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

THE LAYOUT, common to all four tiers:

    tensor  = P panels, row-major in P                     P = N/64
    panel   = 128 B anchor header + S slabs, K-major       S = K/32
    header  = 64 x fp16 row anchors, anchor[r] at byte 2*r
    slab    = SCALE SoA, SUB_PER_ROW bytes per row, row r's bytes at slab[SPR*r ..]
                each byte holds TWO 4-bit sub indices: low nibble first, high nibble second
            + 64 code rows of CODE_BYTES (row r at slab[CODE_OFF + CODE_BYTES*r])

    tier     id   code bits   CODE_BYTES   SPR   CODE_OFF   SLAB_BYTES   sub block   bpw
    pxq2    254       2            8        1       64          576          16      2.25
    pxq3    255       3           12        1       64          832          16      3.25
    pxq4    252       4           16        1       64         1088          16      4.25
    pxq4hq  253       4           16        2      128         1152           8      4.50

DEQUANT CONTRACT, identical for all four and PARITY-LOCKED (the multiply order is
load-bearing; ggml/src/pxq-cpu.h:16-18):
    eff = fp32(anchor_fp16) * SUB[sub_nibble]        once per sub block
    w   = eff * fp32(book[code])                     fp32; the GEMM snaps float16(w)

THE SUB LUT IS SHARED BY THREE OF THE FOUR, AND PXQ4HQ IS THE EXCEPTION. Both the PXQ2 and
PXQ3 engine headers say the PXQ6 SUB16 table is reused verbatim, and the values in the shipped
files confirm it byte for byte. pxq4hq does NOT reuse it: it spends one sub index per EIGHT
elements against its own SUB8 fit (ggml-pxq6-tables.h:47-51, PXQ6_SUB8_INIT), which is a
different 16 floats. Decoding pxq4hq bytes against SUB16 produces a well-formed tensor that is
uniformly wrong, so the sub table is looked up per tier here exactly as the book is.

Only the BOOK differs otherwise: 4 Lloyd-fit entries for pxq2, 8 for pxq3, the 16-entry PX16
book for pxq4 -- and pxq4hq shares PX16 with pxq4 bit for bit. The LM4/LM8 books have NO zero
entry and absmax != 1 by design, so the PXQ4/PX16 invariants (book[7] == 0, book[15] == 1)
deliberately do not apply to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ggml type ids. Used verbatim as the tier key everywhere in this package so a tier can never
# be confused with an array index or a version number.
PXQ2 = 254
PXQ3 = 255
PXQ4 = 252
PXQ4HQ = 253

PANEL_ROWS = 64
SLAB_COLS = 32
HEADER_BYTES = 128
ROW_META = 2
#: The pxq2/pxq3/pxq4 code offset. NOT a universal constant any more -- pxq4hq's is 128 -- so
#: every decode path reads ``tier.code_off``. Kept as a module name because ``layout.py`` and
#: the parity harness import it for the tiers whose value it still is.
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
    #: scale bytes per row per slab. 1 for every tier whose sub block is 16 elements; 2 for
    #: pxq4hq, whose sub block is 8. Also fixes the code offset (64 * sub_per_row) and the
    #: number of effective scales in a 32-element block (2 * sub_per_row).
    sub_per_row: int = 1
    #: True when this tier's sub-scale LUT is NOT the shared SUB16.
    own_sub: bool = False

    @property
    def code_off(self) -> int:
        """Byte offset of the code rows inside a slab."""
        return PANEL_ROWS * self.sub_per_row

    @property
    def neff(self) -> int:
        """Distinct effective scales in one (row, 32-column block)."""
        return 2 * self.sub_per_row

    @property
    def sub_block(self) -> int:
        """Elements covered by one sub index."""
        return SLAB_COLS // self.neff

    @property
    def bpw_fn(self):
        """bits per weight as a function of K (the +16/K is the panel's fp16 row anchor)."""
        return lambda K: self.type_size * 8.0 / 32.0 + 16.0 / K


TIERS: dict[int, Tier] = {
    PXQ2: Tier(PXQ2, "pxq2", 2,  8,  576,  9, 4),
    PXQ3: Tier(PXQ3, "pxq3", 3, 12,  832, 13, 8),
    PXQ4: Tier(PXQ4, "pxq4", 4, 16, 1088, 17, 16),
    PXQ4HQ: Tier(PXQ4HQ, "pxq4hq", 4, 16, 1152, 18, 16, sub_per_row=2, own_sub=True),
}

BY_NAME: dict[str, Tier] = {t.name: t for t in TIERS.values()}


#: tier -> the frozen v1 book above. PXQ4's lives in reference.py (it is the PX16 book the
#: encoder shares) and pxq4hq's IS that same table, so this covers only the tiers whose tables
#: are defined in this module.
_DEFAULT_BOOKS = {PXQ2: BOOK_PXQ2, PXQ3: BOOK_PXQ3}

#: E8-row 4-bit energy-weighted sublevels, the bs8 HQ tier's OWN table. Transcribed from
#: ggml/include/ggml-pxq6-tables.h:47-51 (PXQ6_SUB8_INIT) as C99 hex float literals exactly as
#: they appear in the header, so the transcription is diffable by eye and cannot drift through
#: decimal rounding. Every entry is fp16-exact by construction; ``check_sub`` re-asserts it.
_SUB8_HEX = (
    "0x1.58c0000000000p-3", "0x1.e440000000000p-3", "0x1.2640000000000p-2",
    "0x1.5280000000000p-2", "0x1.7a80000000000p-2", "0x1.a040000000000p-2",
    "0x1.c4c0000000000p-2", "0x1.e900000000000p-2", "0x1.07c0000000000p-1",
    "0x1.1c80000000000p-1", "0x1.32c0000000000p-1", "0x1.4bc0000000000p-1",
    "0x1.68c0000000000p-1", "0x1.8b40000000000p-1", "0x1.b700000000000p-1",
    "0x1.f380000000000p-1",
)
SUB8 = np.array([float.fromhex(h) for h in _SUB8_HEX], dtype=np.float32)
SUB8.flags.writeable = False

#: tier -> its OWN compiled-in sub table, for the tiers that do not use the shared SUB16.
_DEFAULT_SUBS = {PXQ4HQ: SUB8}


def sub_of(type_id: int):
    """The engine's compiled-in sub-scale table for a tier that owns one.

    Raises for every tier that indexes the shared SUB16 -- asking for "pxq3's own sub" is a
    category error and returning SUB16 would hide it. Only ``--books-from-defaults`` and the
    self-tests call this; a real conversion reads the table out of the file.
    """
    s = _DEFAULT_SUBS.get(int(type_id))
    if s is None:
        raise ValueError(
            f"ggml type {type_id} does not own a sub-scale table -- it indexes the shared "
            f"SUB16, which lives in reference.SUB")
    return s


def book_of(type_id: int):
    """The engine's compiled-in book for a tier, for a file that did not record its own.

    Only ``--books-from-defaults`` calls this. A file built with PXA_PXQ2_V3 /
    PXA_PXQ_CEIL_V2 / PXA_PXQ*_BOOK has a DIFFERENT book and nothing in the file says so,
    which is why substituting this is a deliberate, flagged, recorded act and never a
    fallback.
    """
    b = _DEFAULT_BOOKS.get(int(type_id))
    if b is None:
        raise ValueError(f"no compiled-in default book for ggml type {type_id}")
    return b


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


# pxq4hq's code rows are BYTE-IDENTICAL to pxq4's -- the HQ tier subdivides the SCALE, not the
# codes -- so it shares the extractor rather than getting a copy of it.
_CODES = {PXQ2: _codes_pxq2, PXQ3: _codes_pxq3, PXQ4: _codes_pxq4, PXQ4HQ: _codes_pxq4}


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
        raise ValueError(f"pxq: a PXQ sub-scale LUT is 16 entries, got {sub.size}")

    # SCALE SoA. sub_per_row bytes per row, INTERLEAVED BY ROW (row r's bytes are adjacent at
    # slab[SPR*r ..], not two separate planes), each byte carrying two 4-bit indices low-first.
    # For every tier but pxq4hq this is the historical [P, S, 64] single byte per row.
    scale_bytes = slabs[:, :, :t.code_off].reshape(P, S, PANEL_ROWS, t.sub_per_row)
    nib = np.empty((P, S, PANEL_ROWS, t.neff), dtype=np.uint8)
    nib[..., 0::2] = scale_bytes & 0x0F
    nib[..., 1::2] = scale_bytes >> 4

    code_rows = slabs[:, :, t.code_off:].reshape(P, S, PANEL_ROWS, t.code_bytes)
    codes = _CODES[type_id](code_rows)                                    # [P, S, 64, 32]

    anch = np.asarray(anchor, dtype=np.float16).astype(np.float32)        # [P, 64]
    # eff = fp32(anchor) * SUB[nibble], ONE PER SUB BLOCK. anchor broadcasts over slabs and
    # over the blocks within a row.
    eff = anch[:, None, :, None] * sub[nib.astype(np.intp)]               # [P, S, 64, neff]

    w = book[codes.astype(np.intp)]                                       # [P, S, 64, 32]
    # Element j belongs to block j // sub_block, which is exactly what this reshape says.
    w = w.reshape(P, S, PANEL_ROWS, t.neff, t.sub_block)
    w *= eff[..., None]
    w = w.reshape(P, S, PANEL_ROWS, SLAB_COLS)
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
    if type_id in (PXQ4, PXQ4HQ):
        # PXQ4/PX16 invariants live in reference.check_tables; not our business here. pxq4hq
        # shares that book bit for bit, so it shares the exemption.
        return
    if not (b[0] < 0.0 and b[-1] > 0.0):
        raise ValueError(f"{t.name}: book does not straddle zero: {b.tolist()}")
    if not np.all(np.diff(b) > 0):
        raise ValueError(f"{t.name}: book is not strictly ascending: {b.tolist()}")
    if not np.all(np.abs(b) < 1.0):
        raise ValueError(f"{t.name}: |book| must be < 1 for a Lloyd-fit LM book: {b.tolist()}")
    if not np.array_equal(b, b.astype(np.float16).astype(np.float32)):
        raise ValueError(f"{t.name}: book is not fp16-exact: {b.tolist()}")


def check_sub(sub) -> None:
    """Sub-scale LUT invariants, shared by every tier and by both tables (SUB16 and pxq4hq's
    SUB8): 16 four-bit levels, ascending, positive, fp16-exact."""
    s = np.asarray(sub, dtype=np.float32)
    if s.size != 16:
        raise ValueError(f"SUB16 must have 16 entries, got {s.size}")
    if not (s[0] > 0.0 and np.all(np.diff(s) > 0)):
        raise ValueError(f"SUB16 must be positive and strictly ascending: {s.tolist()}")
    if not np.array_equal(s, s.astype(np.float16).astype(np.float32)):
        raise ValueError(f"SUB16 is not fp16-exact: {s.tolist()}")
