"""safetensors_io.py — a minimal, streaming safetensors writer and header reader.

WHY NOT THE ``safetensors`` PACKAGE. Two reasons, both structural rather than aesthetic.

 1. We must emit BF16 tensors we never decode — the 333 ``model.visual.*`` tensors are copied
    byte-for-byte from the AWQ twin so the vision tower is bit-identical to what the incumbent
    already serves. numpy has no bfloat16, so anything that round-trips through ndarray dtypes
    either loses them or forces a torch dependency into a converter that must run on a machine
    with no GPU stack. Writing the container ourselves lets a tensor be (dtype string, shape,
    raw bytes) with no interpretation.
 2. The writer must stream. The output is ~26 GB across shards; holding a shard in RAM as
    materialised arrays is avoidable, so tensors are handed over as byte-producing callables
    and written straight out.

FORMAT (upstream spec): u64 little-endian header length, then that many bytes of UTF-8 JSON,
then the tensor data. Every JSON entry is ``{"dtype", "shape", "data_offsets": [beg, end)}``
with offsets relative to the start of the data section. ``__metadata__`` is a reserved key
holding a flat string->string map. Data must be C-contiguous in row-major order. Upstream
aligns the data section to 8 bytes; we match that, and additionally pad each tensor's start to
8 so a consumer that mmaps and reinterprets never hits an unaligned load.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Callable, Iterable

#: safetensors dtype strings we emit. The value is bytes-per-element, or None when the element
#: size is not a whole number of bytes (never the case here).
DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

NUMPY_TO_ST = {
    "uint8": "U8", "int8": "I8", "int16": "I16", "uint16": "U16",
    "float16": "F16", "int32": "I32", "uint32": "U32", "float32": "F32",
    "int64": "I64", "uint64": "U64", "float64": "F64", "bool": "BOOL",
}


def numpy_dtype_to_st(dtype) -> str:
    name = str(dtype)
    if name not in NUMPY_TO_ST:
        raise ValueError(f"safetensors: no dtype string for numpy dtype {name!r}")
    return NUMPY_TO_ST[name]


class Tensor:
    """One tensor to write.

    ``data`` is bytes, or a zero-argument callable returning bytes, or — for anything large —
    a callable taking the open file object and writing to it directly (pass
    ``streaming=True``). The streaming form is what keeps peak RSS bounded: ``token_embd`` and
    ``lm_head`` are 2.54 GB each at fp16, and materialising either as a single ``bytes`` object
    alongside its float32 decode intermediate is how a converter OOMs on a box that had plenty
    of room. The declared ``nbytes`` is still checked, so a streaming writer that produces the
    wrong length is caught rather than silently corrupting every later offset."""

    __slots__ = ("name", "dtype", "shape", "nbytes", "_data", "_streaming")

    def __init__(self, name: str, dtype: str, shape: Iterable[int],
                 data: bytes | Callable[..., object], streaming: bool = False) -> None:
        if dtype not in DTYPE_SIZE:
            raise ValueError(f"safetensors: unknown dtype {dtype!r}")
        self.name = name
        self.dtype = dtype
        self.shape = [int(s) for s in shape]
        n = DTYPE_SIZE[dtype]
        for s in self.shape:
            n *= s
        self.nbytes = n
        self._data = data
        self._streaming = streaming

    @classmethod
    def from_numpy(cls, name: str, arr) -> "Tensor":
        import numpy as np
        arr = np.ascontiguousarray(arr)
        return cls(name, numpy_dtype_to_st(arr.dtype), arr.shape, arr.tobytes())

    def bytes(self) -> bytes:
        if self._streaming:
            raise TypeError(f"{self.name} is a streaming tensor; use write_to()")
        b = self._data() if callable(self._data) else self._data
        if len(b) != self.nbytes:
            raise ValueError(f"{self.name}: produced {len(b)} B, header declares {self.nbytes}")
        return b

    def write_to(self, f) -> None:
        if not self._streaming:
            f.write(self.bytes())
            return
        beg = f.tell()
        self._data(f)
        wrote = f.tell() - beg
        if wrote != self.nbytes:
            raise ValueError(f"{self.name}: streamed {wrote} B, header declares {self.nbytes}")


def write_file(path: str, tensors: list[Tensor], metadata: dict[str, str] | None = None) -> int:
    """Write one .safetensors file. Returns the total byte size."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    off = 0
    layout = []
    for t in tensors:
        pad = (-off) % 8
        off += pad
        header[t.name] = {"dtype": t.dtype, "shape": t.shape,
                          "data_offsets": [off, off + t.nbytes]}
        layout.append((t, pad))
        off += t.nbytes

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)          # 8-align the start of the data section

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for t, pad in layout:
            if pad:
                f.write(b"\0" * pad)
            t.write_to(f)
    os.replace(tmp, path)                      # never leave a half-written shard in place
    return 8 + len(blob) + off


#: The AWQ twin's header is 312 KB of JSON describing 2385 tensors, and the vision-tower copy
#: touches it 333 times. Parsing it once is worth the cache; the key includes mtime so a
#: rewritten file is not served stale.
_HEADER_CACHE: dict[tuple[str, int, int], dict] = {}


def read_header(path: str) -> dict:
    """Header JSON of an existing .safetensors file, without mapping the data."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    cached = _HEADER_CACHE.get(key)
    if cached is not None:
        return cached
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    _HEADER_CACHE[key] = hdr
    return hdr


def read_tensor_bytes(path: str, name: str) -> tuple[str, list[int], bytes]:
    """(dtype, shape, raw bytes) of one tensor from an existing file. Used to copy the AWQ
    twin's BF16 vision tower through verbatim without ever interpreting it."""
    hdr = read_header(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        e = hdr[name]
        beg, end = e["data_offsets"]
        f.seek(8 + n + beg)
        raw = f.read(end - beg)
    if len(raw) != end - beg:
        raise ValueError(f"{path}: short read for {name} ({len(raw)} of {end - beg} B)")
    return e["dtype"], e["shape"], raw


class ShardWriter:
    """Accumulates tensors and flushes them into ``model-000xx-of-000yy.safetensors`` shards of
    at most ``max_bytes``, writing the ``model.safetensors.index.json`` HF expects.

    A single tensor larger than ``max_bytes`` gets its own shard rather than being rejected:
    ``lm_head`` at BF16 is 2.5 GB and ``token_embd`` at fp16 the same, and a converter that
    refused them would be useless.
    """

    def __init__(self, out_dir: str, max_bytes: int = 4 * (1 << 30),
                 metadata: dict[str, str] | None = None) -> None:
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.metadata = metadata or {}
        self._pending: list[Tensor] = []
        self._pending_bytes = 0
        self._shards: list[list[Tensor]] = []

    def add(self, t: Tensor) -> None:
        if self._pending and self._pending_bytes + t.nbytes > self.max_bytes:
            self._flush_pending()
        self._pending.append(t)
        self._pending_bytes += t.nbytes

    def _flush_pending(self) -> None:
        if self._pending:
            self._shards.append(self._pending)
            self._pending = []
            self._pending_bytes = 0

    def finish(self) -> dict:
        self._flush_pending()
        os.makedirs(self.out_dir, exist_ok=True)
        n = len(self._shards)
        weight_map: dict[str, str] = {}
        total = 0
        for i, group in enumerate(self._shards, start=1):
            fname = (f"model-{i:05d}-of-{n:05d}.safetensors" if n > 1
                     else "model.safetensors")
            total += write_file(os.path.join(self.out_dir, fname), group, self.metadata)
            for t in group:
                weight_map[t.name] = fname
        index = {"metadata": {"total_size": total, **self.metadata},
                 "weight_map": weight_map}
        if n > 1:
            with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f, indent=1)
        return index
