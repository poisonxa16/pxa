"""gguf_raw.py — a dependency-free GGUF v2/v3 reader that survives unknown tensor types.

WHY THIS EXISTS. The obvious path (``from gguf import GGUFReader``) is dead, and not by a
little: ``GGUFReader.__init__`` builds the whole tensor table eagerly, and
``gguf.GGMLQuantizationType(252)`` raises ``ValueError`` because upstream's enum has no
PXQ4 member. One PXQ4 tensor therefore kills the *file open*, before a single tensor can be
yielded. Fixing that upstream means forking the ``gguf`` PyPI package, vLLM's
``quantization/gguf.py`` type sets, and vLLM's vendored ggml ``_custom_ops`` — and even then
you land on a sharder that assumes a weight row is a contiguous byte run, which 64-row panel
interleave violates. So we parse the container ourselves and never name a type we don't know.

The container format is stable and small: magic, version, tensor count, KV count, then the
KV table, then a tensor directory of (name, ndim, dims..., type_id, offset), then padding to
``general.alignment``, then the data section. Nothing in the container depends on knowing
what a type id means — only the *size* of a tensor does, and we sidestep that too (see
``TensorInfo.nbytes``).

SIZING WITHOUT A TRAITS TABLE. ggml writes tensors into the data section in directory order
with no inter-tensor padding (verified byte-exactly for this artifact: the panel formula
``(rows/64)*(128 + (K/32)*1088)`` reproduces all six PXQ4 on-disk shapes, and consecutive
offsets differ by exactly that). We therefore derive each tensor's byte length from the
*next* offset in offset-sorted order, with the file tail closing the last one. That makes the
reader correct for types it has never heard of, which is the entire point. Where we do know
the traits (``TRAITS`` below) we cross-check the derived size and raise on a mismatch, so a
future file with padding or a reordered data section fails loudly instead of silently
handing out short buffers.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# ---------------------------------------------------------------------------------------------
# GGUF metadata value types (gguf_metadata_value_type in ggml/src/gguf.cpp)
# ---------------------------------------------------------------------------------------------
(GT_U8, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR, GT_ARR, GT_U64,
 GT_I64, GT_F64) = range(13)

_SCALAR_FMT = {
    GT_U8: "<B", GT_I8: "<b", GT_U16: "<H", GT_I16: "<h", GT_U32: "<I", GT_I32: "<i",
    GT_F32: "<f", GT_BOOL: "<B", GT_U64: "<Q", GT_I64: "<q", GT_F64: "<d",
}
_SCALAR_SIZE = {k: struct.calcsize(v) for k, v in _SCALAR_FMT.items()}

# ---------------------------------------------------------------------------------------------
# ggml type ids. Only the ids that actually occur in a PXA artifact are named; anything else
# stays a bare integer so an unexpected type produces a readable error rather than a KeyError.
# PXQ ladder ids are documented at ggml/include/ggml.h:455-505; 250/251 are RETIRED and must
# never be reused (ggml.c:1402-1406).
# ---------------------------------------------------------------------------------------------
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q6_K = 14
GGML_MXFP4 = 39
GGML_PXQ1 = 248
GGML_PXQ4 = 252
GGML_PXQ4HQ = 253
GGML_PXQ2 = 254
GGML_PXQ3 = 255
GGML_PXQ6 = 256

TYPE_NAMES = {
    GGML_F32: "f32", GGML_F16: "f16", GGML_Q8_0: "q8_0", GGML_Q6_K: "q6_K",
    GGML_MXFP4: "mxfp4", GGML_PXQ1: "pxq1", GGML_PXQ4: "pxq4", GGML_PXQ4HQ: "pxq4hq",
    GGML_PXQ2: "pxq2", GGML_PXQ3: "pxq3", GGML_PXQ6: "pxq6",
    2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 9: "q8_1", 10: "q2_K", 11: "q3_K",
    12: "q4_K", 13: "q5_K", 15: "q8_K", 20: "iq4_nl", 23: "iq4_xs", 30: "bf16",
}

#: (block elements, bytes per block, bytes of per-row metadata). ``row_meta`` is ggml's
#: ``row_meta_size`` — nonzero only for the PXQ panel types, where it encodes the 2 B/row
#: fp16 anchor that lives in the panel header rather than in any block (ggml.c:1421-1428).
TRAITS: dict[int, tuple[int, int, int]] = {
    GGML_F32: (1, 4, 0),
    GGML_F16: (1, 2, 0),
    GGML_Q8_0: (32, 34, 0),
    GGML_Q6_K: (256, 210, 0),
    GGML_MXFP4: (32, 17, 0),
    GGML_PXQ4: (32, 17, 2),
    GGML_PXQ4HQ: (32, 18, 2),
    GGML_PXQ2: (32, 9, 2),
    GGML_PXQ3: (32, 13, 2),
}

#: Types this converter can decode. Anything else in a file is a hard error, not a warning:
#: silently dropping a tensor produces a model that loads and is wrong.
SUPPORTED = frozenset({GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q6_K, GGML_MXFP4, GGML_PXQ4})


def type_name(t: int) -> str:
    return TYPE_NAMES.get(t, f"type_{t}")


def row_size(type_id: int, k: int) -> int:
    """Bytes ggml charges for one logical row of ``k`` elements.

    Mirrors ``ggml_row_size`` (ggml.c:4903-4906): ``row_meta + type_size*k/blck_size``. For
    PXQ4 this is ``2 + 17*k/32``, which is where the documented ``4.25 + 16/K`` bpw comes
    from — the 2 B is the row's share of its panel's 128 B fp16 anchor header.
    """
    if type_id not in TRAITS:
        raise KeyError(f"no traits for ggml type {type_name(type_id)} ({type_id})")
    blck, tsize, meta = TRAITS[type_id]
    if k % blck != 0:
        raise ValueError(f"{type_name(type_id)}: k={k} not a multiple of block size {blck}")
    return meta + tsize * k // blck


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]          # ggml ne order: dims[0] is the fastest-varying axis (K)
    type_id: int
    offset: int                    # relative to the start of the data section
    nbytes: int = 0                # filled in by GGUFFile after offset-sorting
    index: int = 0                 # position in the tensor directory

    @property
    def ne0(self) -> int:
        return self.dims[0]

    @property
    def ne1(self) -> int:
        return self.dims[1] if len(self.dims) > 1 else 1

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def type(self) -> str:
        return type_name(self.type_id)

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """Row-major (torch/HF) shape: ggml ``ne`` reversed. A ggml 2D weight is
        ``ne = (K, N)`` and torch wants ``[N, K]`` — i.e. ``out_features x in_features``,
        which is exactly what ``nn.Linear.weight`` holds and what vLLM's loaders narrow."""
        return tuple(reversed(self.dims))

    def __repr__(self) -> str:  # keeps failure messages readable
        return (f"TensorInfo({self.name!r}, ne={self.dims}, {self.type}, "
                f"off={self.offset}, nbytes={self.nbytes})")


class GGUFFile:
    """Read-only GGUF accessor. Holds an ``mmap`` over the whole file; tensor payloads are
    returned as zero-copy ``memoryview``s so a 15 GiB checkpoint never lands in RAM."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._f: BinaryIO = open(path, "rb")
        self.file_size = os.fstat(self._f.fileno()).st_size
        self.kv: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.order: list[str] = []
        self._parse_header()
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    # -- context manager -----------------------------------------------------------------
    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._f.close()

    # -- parsing -------------------------------------------------------------------------
    def _rd(self, n: int) -> bytes:
        b = self._f.read(n)
        if len(b) != n:
            raise EOFError(f"{self.path}: truncated at {self._f.tell()} (wanted {n} B)")
        return b

    def _scalar(self, t: int) -> Any:
        v = struct.unpack(_SCALAR_FMT[t], self._rd(_SCALAR_SIZE[t]))[0]
        return bool(v) if t == GT_BOOL else v

    def _string(self) -> str:
        n = struct.unpack("<Q", self._rd(8))[0]
        # surrogateescape rather than 'replace': a tokenizer byte-fallback token is not valid
        # UTF-8 and must survive a round trip if anything ever re-emits it.
        return self._rd(n).decode("utf-8", "surrogateescape")

    def _value(self, t: int) -> Any:
        if t == GT_STR:
            return self._string()
        if t == GT_ARR:
            et = struct.unpack("<I", self._rd(4))[0]
            n = struct.unpack("<Q", self._rd(8))[0]
            if et == GT_STR:
                return [self._string() for _ in range(n)]
            if et == GT_ARR:
                raise ValueError("nested GGUF arrays are not supported")
            sz, fmt = _SCALAR_SIZE[et], _SCALAR_FMT[et][1]
            raw = self._rd(sz * n)
            vals = list(struct.unpack(f"<{n}{fmt}", raw))
            return [bool(v) for v in vals] if et == GT_BOOL else vals
        return self._scalar(t)

    def _parse_header(self) -> None:
        magic = self._rd(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{self.path}: not a GGUF file (magic {magic!r})")
        self.version = struct.unpack("<I", self._rd(4))[0]
        if self.version not in (2, 3):
            raise ValueError(f"{self.path}: unsupported GGUF version {self.version}")
        n_tensors = struct.unpack("<Q", self._rd(8))[0]
        n_kv = struct.unpack("<Q", self._rd(8))[0]

        for _ in range(n_kv):
            key = self._string()
            t = struct.unpack("<I", self._rd(4))[0]
            self.kv[key] = self._value(t)

        infos: list[TensorInfo] = []
        for i in range(n_tensors):
            name = self._string()
            nd = struct.unpack("<I", self._rd(4))[0]
            dims = struct.unpack(f"<{nd}Q", self._rd(8 * nd))
            type_id = struct.unpack("<I", self._rd(4))[0]
            offset = struct.unpack("<Q", self._rd(8))[0]
            infos.append(TensorInfo(name=name, dims=dims, type_id=type_id, offset=offset,
                                    index=i))

        self.alignment = int(self.kv.get("general.alignment", 32))
        dir_end = self._f.tell()
        self.data_start = (dir_end + self.alignment - 1) // self.alignment * self.alignment

        # Derive lengths from neighbouring offsets so unknown types still get exact bounds.
        by_off = sorted(infos, key=lambda t: t.offset)
        data_len = self.file_size - self.data_start
        for i, ti in enumerate(by_off):
            end = by_off[i + 1].offset if i + 1 < len(by_off) else data_len
            ti.nbytes = end - ti.offset
            if ti.nbytes <= 0:
                raise ValueError(f"{ti.name}: non-positive derived size {ti.nbytes}")

        # Cross-check against the traits table where we have one. A mismatch means either a
        # padded data section or a wrong traits entry; both would corrupt every later step,
        # so this raises rather than warns.
        for ti in by_off:
            if ti.type_id in TRAITS:
                want = row_size(ti.type_id, ti.ne0)
                for d in ti.dims[1:]:
                    want *= d
                if want != ti.nbytes:
                    raise ValueError(
                        f"{ti.name}: derived {ti.nbytes} B but traits predict {want} B "
                        f"(ne={ti.dims}, type={ti.type}); the data section is not densely "
                        f"packed or TRAITS[{ti.type_id}] is wrong")

        for ti in infos:
            if ti.name in self.tensors:
                raise ValueError(f"duplicate tensor name {ti.name!r}")
            self.tensors[ti.name] = ti
            self.order.append(ti.name)

    # -- access --------------------------------------------------------------------------
    def raw(self, name: str) -> memoryview:
        """Zero-copy view of a tensor's on-disk bytes."""
        ti = self.tensors[name]
        beg = self.data_start + ti.offset
        return memoryview(self._mm)[beg:beg + ti.nbytes]

    def kv_get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    def type_histogram(self) -> dict[str, tuple[int, int]]:
        hist: dict[str, list[int]] = {}
        for ti in self.tensors.values():
            e = hist.setdefault(ti.type, [0, 0])
            e[0] += 1
            e[1] += ti.nbytes
        return {k: (v[0], v[1]) for k, v in sorted(hist.items())}

    def assert_all_supported(self) -> None:
        bad = sorted({t.type for t in self.tensors.values() if t.type_id not in SUPPORTED})
        if bad:
            raise ValueError(
                f"{self.path}: contains ggml types this converter cannot decode: {bad}. "
                f"Add a decoder in dequant_ref.py — do NOT skip the tensors.")


class GGUFHeaderOnly(GGUFFile):
    """Parses the header of a file whose data section is absent or truncated.

    Used by the offline tests, which carry a real 10.5 MB header slice of the 15.7 GB
    artifact so name/shape/type/geometry checks run on this machine with no DGX access. The
    derived-size cross-check is skipped for the final tensor only, since its length depends
    on a file tail we do not have.
    """

    def __init__(self, path: str, true_file_size: int) -> None:
        self._true_file_size = true_file_size
        super().__init__(path)

    def _parse_header(self) -> None:
        real = self.file_size
        self.file_size = self._true_file_size
        try:
            super()._parse_header()
        finally:
            self.file_size = real

    def raw(self, name: str) -> memoryview:
        raise NotImplementedError("header-only GGUF: no data section")
