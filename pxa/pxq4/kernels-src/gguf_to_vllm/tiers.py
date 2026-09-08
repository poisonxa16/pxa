"""tiers.py -- the PXQ tier table: geometry, frozen books, and a numpy decode reference.

WHY THIS MODULE EXISTS. Until PXQ2/PXQ3 support, "the layout" was a set of module-level
constants in ``layout.py`` because there was exactly one of it. There are now three, they
differ in exactly two numbers (code bytes per row per slab, and therefore slab bytes), and
every place that used to say ``1088`` has to say "this tensor's slab stride" instead. Getting
that wrong is the worst class of bug this package has: a PXQ2 tensor read with a PXQ4 stride
is a well-formed array of the wrong bytes, and it produces a model that loads, shards, passes
every shape check and generates fluent garbage. So the geometry lives in ONE table, is looked
up by ggml type id, and the id travels with the tensor from the GGUF directory all the way
into the checkpoint's config.json.

SOURCE OF TRUTH (read-only; do not edit values by hand):
    ggml/include/ggml-pxq2-tables.h, ggml-pxq3-tables.h, ggml-pxq6-tables.h
    ggml/src/ggml-cuda/pxa/pxq23.cuh   (the decode policies these functions mirror)
The C++ twin of this table is ``pxq23_kernel_tables.h``; the two are checked against each
other by the kernel self-test (``torch.ops.pxq4.pxq_selftest``), which decodes the same
synthetic bytes with a host oracle written from the same spec.

THE LAYOUT, common to all three tiers:

    tensor  = P panels, row-major in P                     P = N/64
    panel   = 128 B anchor header + S slabs, K-major       S = K/32
    header  = 64 x fp16 row anchors, anchor[r] at byte 2*r
    slab    = 64 B sub-scale SoA (byte r is row r's scale byte for THIS 32-column block:
                low nibble  -> elements 0..15, high nibble -> elements 16..31)
            + 64 code rows of CODE_BYTES (row r at slab[64 + CODE_BYTES*r])

    tier   id   code bits   CODE_BYTES   SLAB_BYTES   bpw (excl. row meta)
    pxq2  254       2            8           576          2.25
    pxq3  255       3           12           832          3.25
    pxq4  252       4           16          1088          4.25

DEQUANT CONTRACT, identical for all three and PARITY-LOCKED (the multiply order is
load-bearing; ggml/src/pxq-cpu.h:16-18):
    eff = fp32(anchor_fp16) * SUB16[sub_nibble]      once per 16-element block
    w   = eff * fp32(book[code])                     fp32; the GEMM snaps float16(w)

The SUB16 LUT is SHARED by all three tiers -- both the PXQ2 and PXQ3 engine headers say the
PXQ6 table is reused verbatim, and the values in the shipped files confirm it byte for byte.
Only the BOOK differs: 4 Lloyd-fit entries for pxq2, 8 for pxq3, the 16-entry PX16 book for
pxq4. The LM4/LM8 books have NO zero entry and absmax != 1 by design, so the PXQ4/PX16
invariants (book[7] == 0, book[15] == 1) deliberately do not apply to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ggml type ids. Used verbatim as the tier key everywhere in this package so a tier can never
# be confused with an array index or a version number.
PXQ2 = 254
PXQ3 = 255
PXQ4 = 252

PANEL_ROWS = 64
SLAB_COLS = 32
HEADER_BYTES = 128
ROW_META = 2
CODE_OFF = 64

#: Frozen v1 books, decimal values of the exact fp32 hex literals in the engine headers. They
#: are fp16-exact by construction; ``check_book`` re-asserts that rather than trusting it.
BOOK_PXQ2 = np.array(
    [-0.70556640625, -0.1876220703125, 0.186767578125, 0.70263671875], dtype=np.float32)
BOOK_PXQ3 = np.array(
    [-0.90673828125, -0.5478515625, -0.2978515625, -0.0931396484375,
     0.0919189453125, 0.295654296875, 0.54541015625, 0.90576171875], dtype=np.float32)

#: argmin |book| -- the code the quantizer writes for an exactly-zero block. Recorded because
#: it is part of the format contract, not because anything here writes codes.
ZIDX = {PXQ2: 2, PXQ3: 4}


@dataclass(frozen=True)
class Tier:
    type_id: int
    name: str
    code_bits: int
    code_bytes: int      # per row per 32-column slab
    slab_bytes: int
    type_size: int       # ggml bytes per 32 elements per row, excluding row meta
    book_n: int

    @property
    def bpw_fn(self):
        """bits per weight as a function of K (the +16/K is the panel's fp16 row anchor)."""
        return lambda K: self.type_size * 8.0 / 32.0 + 16.0 / K


TIERS: dict[int, Tier] = {
    PXQ2: Tier(PXQ2, "pxq2", 2,  8,  576,  9, 4),
    PXQ3: Tier(PXQ3, "pxq3", 3, 12,  832, 13, 8),
    PXQ4: Tier(PXQ4, "pxq4", 4, 16, 1088, 17, 16),
}

BY_NAME: dict[str, Tier] = {t.name: t for t in TIERS.values()}


def tier_of(type_id: int) -> Tier:
    t = TIERS.get(int(type_id))
    if t is None:
        raise ValueError(
            f"pxq: ggml type id {type_id} is not a PXQ panel tier. The panel layout is only "
            f"defined for {sorted(TIERS)}; anything else has to be decoded to fp16 instead.")
    return t


def is_pxq(type_id: int) -> bool:
    return int(type_id) in TIERS


def slab_bytes(type_id: int) -> int:
    return tier_of(type_id).slab_bytes


def panel_bytes(type_id: int, K: int) -> int:
    if K % SLAB_COLS:
        raise ValueError(f"pxq: K={K} is not a multiple of {SLAB_COLS}")
    return HEADER_BYTES + (K // SLAB_COLS) * slab_bytes(type_id)


def tensor_bytes(type_id: int, N: int, K: int) -> int:
    """On-disk size of a whole [N, K] tensor of this tier.

    Equals ggml's own accounting, ``N * ggml_row_size(t, K)`` = ``N * (2 + type_size*K/32)``.
    """
    assert_geometry(N, K)
    return (N // PANEL_ROWS) * panel_bytes(type_id, K)


def assert_geometry(N: int, K: int) -> None:
    """The quantizer's eligibility gate restated as a load-time invariant, tier-independent.

    ``pxq*_tensor_eligible`` requires ``ne[1] % 64 == 0 && ne[0] % 32 == 0`` and demotes the
    tensor otherwise; the CUDA dequant kernels hard-abort on the same condition. Anything
    reaching here has already claimed to be a panel tier, so a violation means a corrupt file
    or a bad shard boundary.
    """
    if N <= 0 or K <= 0:
        raise ValueError(f"pxq: non-positive geometry N={N} K={K}")
    if N % PANEL_ROWS:
        raise ValueError(
            f"pxq: N={N} is not a multiple of {PANEL_ROWS} -- a partial panel has no valid "
            f"anchor header and cannot be addressed")
    if K % SLAB_COLS:
        raise ValueError(
            f"pxq: K={K} is not a multiple of {SLAB_COLS} -- a partial slab has no valid "
            f"sub-scale byte")


# ---------------------------------------------------------------------------------------------
# blob <-> (slabs, anchor). A PURE SPLIT: no byte is reordered and no value is recomputed, so
# the emitted checkpoint is provably the same weights as the GGUF, verifiable by a BYTE
# comparison rather than by a numeric tolerance. ``join_blob`` is the exact inverse and the
# converter runs the round trip on every tensor it writes.
# ---------------------------------------------------------------------------------------------
def split_blob(blob, type_id: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF panel blob -> (slabs uint8[P, S, SLAB], anchor float16[P, 64])."""
    t = tier_of(type_id)
    assert_geometry(N, K)
    P, S = N // PANEL_ROWS, K // SLAB_COLS
    need = tensor_bytes(type_id, N, K)
    a = np.frombuffer(blob, dtype=np.uint8)
    if a.size != need:
        raise ValueError(f"pxq{t.code_bits}: blob is {a.size} B, expected {need} B "
                         f"for N={N} K={K}")
    a = a.reshape(P, HEADER_BYTES + S * t.slab_bytes)
    # .copy() is deliberate: the source is a read-only mmap of the GGUF, and safetensors
    # writing plus the shard tests both want owned, C-contiguous arrays.
    anchor = a[:, :HEADER_BYTES].copy().view("<f2")
    slabs = a[:, HEADER_BYTES:].copy().reshape(P, S, t.slab_bytes)
    if anchor.shape != (P, PANEL_ROWS):
        raise AssertionError(f"pxq: anchor reinterpret gave {anchor.shape}")
    return slabs, anchor


def join_blob(slabs: np.ndarray, anchor: np.ndarray) -> bytes:
    """(slabs, anchor) -> the original GGUF panel blob. Exact inverse of ``split_blob``."""
    if slabs.dtype != np.uint8 or slabs.ndim != 3:
        raise ValueError(f"pxq: slabs must be uint8 [P,S,SLAB], got {slabs.dtype} {slabs.shape}")
    P, S, sb = slabs.shape
    if sb not in {t.slab_bytes for t in TIERS.values()}:
        raise ValueError(f"pxq: slab stride {sb} is not a known tier "
                         f"({sorted(t.slab_bytes for t in TIERS.values())})")
    if anchor.dtype != np.float16 or anchor.shape != (P, PANEL_ROWS):
        raise ValueError(f"pxq: anchor must be float16 [{P},{PANEL_ROWS}], got "
                         f"{anchor.dtype} {anchor.shape}")
    hdr = np.ascontiguousarray(anchor).view(np.uint8).reshape(P, HEADER_BYTES)
    return np.concatenate([hdr, slabs.reshape(P, S * sb)], axis=1).tobytes()


# ---------------------------------------------------------------------------------------------
# CODE EXTRACTION -- written from the packing spec, vectorised over (P, S, 64 rows).
#
# pxq2: two LE u32 words per row-block. word h covers elements 16h..16h+15; element j of that
#       half sits at bits 2*(j&15).
# pxq3: BIT-PLANE. three LE u32 words w0 w1 w2. w0/w1 are the LOW planes (2 b/elem) of
#       elements 0..15 / 16..31; w2 is the HIGH plane, bit j = element j's bit 2 for the low
#       half and bit 16+j for the high half. code(j) = (lo>>2*(j&15) & 3) | ((w2>>j & 1) << 2).
# pxq4: nibbles. byte b of the 16-byte row holds code(2b) low and code(2b+1) high.
#
# The pxq3 arrangement is not a convenience: it is what makes the DEVICE decode branch-free
# (both the 4-bit fp16-LUT and the int8-LUT extraction paths need the planes contiguous), and
# it is locked by the format contract. Do not "simplify" it to packed 3-bit fields.
# ---------------------------------------------------------------------------------------------
def _codes_pxq2(rows: np.ndarray) -> np.ndarray:
    """rows uint8 [..., 8] -> codes uint8 [..., 32]."""
    w = rows.view("<u4").reshape(*rows.shape[:-1], 2)          # [..., 2]
    j = np.arange(32, dtype=np.uint32)
    half = (j >> 4).astype(np.intp)                            # which u32 word
    sh = (2 * (j & 15)).astype(np.uint32)                      # bit offset inside it
    return ((w[..., half] >> sh) & np.uint32(3)).astype(np.uint8)


def _codes_pxq3(rows: np.ndarray) -> np.ndarray:
    """rows uint8 [..., 12] -> codes uint8 [..., 32]."""
    w = rows.view("<u4").reshape(*rows.shape[:-1], 3)
    j = np.arange(32, dtype=np.uint32)
    half = (j >> 4).astype(np.intp)
    sh = (2 * (j & 15)).astype(np.uint32)
    lo = (w[..., half] >> sh) & np.uint32(3)
    hi = (w[..., 2:3] >> j) & np.uint32(1)                     # broadcasts over all 32 j
    return (lo | (hi << np.uint32(2))).astype(np.uint8)


def _codes_pxq4(rows: np.ndarray) -> np.ndarray:
    """rows uint8 [..., 16] -> codes uint8 [..., 32]."""
    out = np.empty((*rows.shape[:-1], 32), dtype=np.uint8)
    out[..., 0::2] = rows & 0x0F
    out[..., 1::2] = rows >> 4
    return out


_CODES = {PXQ2: _codes_pxq2, PXQ3: _codes_pxq3, PXQ4: _codes_pxq4}


def dequant(slabs: np.ndarray, anchor: np.ndarray, type_id: int, book: np.ndarray,
            sub: np.ndarray) -> np.ndarray:
    """(slabs, anchor) -> float32 [N, K], following the parity-locked contract exactly.

    Returned as float32 because that is what the contract computes; the caller snaps to fp16
    if it is writing a dense tensor. The multiply order (anchor*SUB first, then *book) is NOT
    an implementation detail -- reassociating it changes the last bit of a large fraction of
    the weights.
    """
    t = tier_of(type_id)
    P, S, sb = slabs.shape
    if sb != t.slab_bytes:
        raise ValueError(f"pxq: {t.name} wants slab stride {t.slab_bytes}, got {sb} -- this "
                         f"array is a different tier")
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    if book.size != t.book_n:
        raise ValueError(f"pxq: {t.name} wants a {t.book_n}-entry book, got {book.size}")
    if sub.size != 16:
        raise ValueError(f"pxq: the SUB16 LUT is 16 entries, got {sub.size}")

    scale_bytes = slabs[:, :, :CODE_OFF]                                  # [P, S, 64]
    code_rows = slabs[:, :, CODE_OFF:].reshape(P, S, PANEL_ROWS, t.code_bytes)
    codes = _CODES[type_id](code_rows)                                    # [P, S, 64, 32]

    anch = np.asarray(anchor, dtype=np.float16).astype(np.float32)        # [P, 64]
    eff_lo = anch[:, None, :] * sub[(scale_bytes & 0x0F).astype(np.intp)].transpose(0, 1, 2)
    eff_hi = anch[:, None, :] * sub[(scale_bytes >> 4).astype(np.intp)]
    eff = np.empty((P, S, PANEL_ROWS, 2), dtype=np.float32)
    eff[..., 0] = eff_lo
    eff[..., 1] = eff_hi

    w = book[codes.astype(np.intp)]                                       # [P, S, 64, 32]
    w[..., :16] *= eff[..., 0:1]
    w[..., 16:] *= eff[..., 1:2]
    # [P, S, 64, 32] -> [P, 64, S, 32] -> [N, K]
    return w.transpose(0, 2, 1, 3).reshape(P * PANEL_ROWS, S * SLAB_COLS)


def check_book(book, type_id: int) -> None:
    """The engine's ``pxa_pxq23_book_ok``, restated: fp16-snapped, strictly ascending,
    sign-straddling, |v| < 1. NOTE this is deliberately NOT the PXQ4/PX16 check -- the LM4/LM8
    books have no zero entry and absmax != 1 by design, and applying the PX16 invariants to
    them would reject a perfectly good table."""
    b = np.asarray(book, dtype=np.float32)
    t = tier_of(type_id)
    if b.size != t.book_n:
        raise ValueError(f"{t.name}: book must have {t.book_n} entries, got {b.size}")
    if type_id == PXQ4:
        return  # PXQ4/PX16 invariants live in reference.check_tables; not our business here
    if not (b[0] < 0.0 and b[-1] > 0.0):
        raise ValueError(f"{t.name}: book does not straddle zero: {b.tolist()}")
    if not np.all(np.diff(b) > 0):
        raise ValueError(f"{t.name}: book is not strictly ascending: {b.tolist()}")
    if not np.all(np.abs(b) < 1.0):
        raise ValueError(f"{t.name}: |book| must be < 1 for a Lloyd-fit LM book: {b.tolist()}")
    if not np.array_equal(b, b.astype(np.float16).astype(np.float32)):
        raise ValueError(f"{t.name}: book is not fp16-exact: {b.tolist()}")


def check_sub(sub) -> None:
    """SUB16 invariants, shared by every tier: 16 entries, ascending, positive, fp16-exact."""
    s = np.asarray(sub, dtype=np.float32)
    if s.size != 16:
        raise ValueError(f"SUB16 must have 16 entries, got {s.size}")
    if not (s[0] > 0.0 and np.all(np.diff(s) > 0)):
        raise ValueError(f"SUB16 must be positive and strictly ascending: {s.tolist()}")
    if not np.array_equal(s, s.astype(np.float16).astype(np.float32)):
        raise ValueError(f"SUB16 is not fp16-exact: {s.tolist()}")
