"""
gguf_raw.py -- minimal GGUF reader that does not care what the tensor types mean.

The upstream `gguf` PyPI package cannot open this file at all: GGUFReader._build_tensors
constructs a gguf.GGMLQuantizationType for EVERY tensor in the directory, and 252 is not
a member, so the ValueError kills the file open before a single tensor is yielded.  That
is also why vLLM's gguf.py loader path is unusable (plan §5.1) -- not a preference, a
hard stop.

This reader parses the header, the KV table and the tensor directory with `struct`, then
memory-maps the data section.  Tensor byte-lengths are derived from the sorted offset
list, so unknown type ids cost nothing.
"""

from __future__ import annotations

import mmap
import os
import struct

# GGUF metadata value type tags
(T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL,
 T_STR, T_ARR, T_U64, T_I64, T_F64) = range(13)

_SCALAR_FMT = {
    T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
    T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d",
}
_SCALAR_SIZE = {k: struct.calcsize(v) for k, v in _SCALAR_FMT.items()}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 8: "Q8_0", 14: "Q6_K", 30: "BF16", 39: "MXFP4",
    248: "PXQ1", 252: "PXQ4", 253: "PXQ4HQ", 254: "PXQ2", 255: "PXQ3", 256: "PXQ6",
}


class GGUFTensor:
    __slots__ = ("name", "dims", "type_id", "offset", "nbytes")

    def __init__(self, name, dims, type_id, offset):
        self.name = name
        self.dims = dims
        self.type_id = type_id
        self.offset = offset
        self.nbytes = -1        # filled in by GGUFRaw once all offsets are known

    @property
    def K(self) -> int:
        """ggml ne[0] -- the contiguous/input dimension.  For a PXQ4 weight this is the
        matmul K, i.e. the axis chopped into 32-column slabs."""
        return self.dims[0]

    @property
    def rows(self) -> int:
        """ggml ne[1] -- the output-row count, chopped into 64-row panels."""
        return self.dims[1] if len(self.dims) > 1 else 1

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.type_id, f"UNKNOWN_{self.type_id}")

    def __repr__(self):
        return (f"GGUFTensor({self.name!r}, ne={self.dims}, {self.type_name}, "
                f"{self.nbytes} B @ {self.offset})")


class GGUFRaw:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._pos = 0
        magic = self._rd(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (magic {magic!r})")
        self.version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()

        self.kv = {}
        for _ in range(n_kv):
            key = self._str()
            self.kv[key] = self._value(self._u32())

        self.tensors = {}
        order = []
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u32()
            dims = [self._u64() for _ in range(nd)]
            type_id = self._u32()
            offset = self._u64()
            t = GGUFTensor(name, dims, type_id, offset)
            self.tensors[name] = t
            order.append(t)

        align = int(self.kv.get("general.alignment", 32))
        dir_end = self._pos
        self.data_start = (dir_end + align - 1) // align * align
        self.file_size = os.path.getsize(path)

        # Sizes from the offset gaps: works for type ids this reader has never heard of,
        # and doubles as a check that the file has no inter-tensor padding.
        by_off = sorted(order, key=lambda t: t.offset)
        data_len = self.file_size - self.data_start
        for i, t in enumerate(by_off):
            nxt = by_off[i + 1].offset if i + 1 < len(by_off) else data_len
            t.nbytes = nxt - t.offset

        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    # --- byte readers -------------------------------------------------------------
    def _rd(self, n):
        b = self._f.read(n)
        if len(b) != n:
            raise EOFError(f"{self.path}: truncated at {self._pos}")
        self._pos += n
        return b

    def _u32(self):
        return struct.unpack("<I", self._rd(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._rd(8))[0]

    def _str(self):
        return self._rd(self._u64()).decode("utf-8", "replace")

    def _value(self, t):
        if t == T_ARR:
            et = self._u32()
            n = self._u64()
            if et == T_STR:
                return [self._str() for _ in range(n)]
            sz = _SCALAR_SIZE[et]
            raw = self._rd(sz * n)
            fmt = _SCALAR_FMT[et][1]
            vals = list(struct.unpack(f"<{n}{fmt}", raw))
            return [bool(v) for v in vals] if et == T_BOOL else vals
        if t == T_STR:
            return self._str()
        v = struct.unpack(_SCALAR_FMT[t], self._rd(_SCALAR_SIZE[t]))[0]
        return bool(v) if t == T_BOOL else v

    # --- data access --------------------------------------------------------------
    def raw(self, name: str, panel0: int = 0, npanels: int = -1, panel_bytes: int = -1):
        """Return a tensor's bytes, optionally only panels [panel0, panel0+npanels).

        A panel subrange IS a valid standalone PXQ4 tensor (that is the whole point of the
        column-shard argument), so this is how a 47 MB ffn_gate becomes a 350 KB test
        fixture without weakening the test.
        """
        t = self.tensors[name]
        base = self.data_start + t.offset
        if npanels < 0:
            return bytes(self._mm[base:base + t.nbytes])
        if panel_bytes < 0:
            raise ValueError("panel_bytes required when slicing panels")
        s = base + panel0 * panel_bytes
        return bytes(self._mm[s:s + npanels * panel_bytes])

    def close(self):
        try:
            self._mm.close()
        finally:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
