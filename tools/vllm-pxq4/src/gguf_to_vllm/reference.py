"""reference.py — the numpy bit-exactness oracle for PXQ4. Plan §6.3.

This is the single most load-bearing file in the port. Gate G1 pins it against the engine's
own CPU decoder ``pxa_deq_row_pxq6`` (ggml/src/pxq-cpu.c:135-158) by exact fp32 equality;
gate G6 then pins the CUDA extension against this. If G1 passes, the panel arithmetic, slab
offsets, nibble order, table values and multiply order are all correct, which is the bulk of
the port's risk — and none of it needs a GPU.

THE PARITY-LOCKED CONTRACT (pxq-cpu.h:16-18 states it explicitly: the dequant IS the
contract; the GEMM kernels are deliberately NOT bit-exact with it because they snap products
to fp16 inside the MMA):

    eff = float32(anchor_fp16) * SUB[s4]     once per (row, 16-element block)
    w   = eff * float32(BOOK[code])          per element

``eff`` must be formed FIRST, as its own rounded float32 product. Reassociating to
``anchor * (sub * book)`` changes the rounding and breaks bit-exactness against both the C
reference and the CUDA kernel. There is no zero point and no offset: a zero row is
``anchor = fp16(+0)`` and dequantizes to exact zeros everywhere.
"""

from __future__ import annotations

import numpy as np

from .layout import (CODE_BYTES, CODE_OFF, PANEL_ROWS, SLAB_BYTES, SLAB_COLS)

# ---------------------------------------------------------------------------------------------
# The frozen tables, transcribed from ggml/include/ggml-pxq6-tables.h:33-44 as C99 hex float
# literals exactly as they appear in the header. float.fromhex parses the same grammar C does,
# so the transcription is diffable by eye against the header and cannot drift through decimal
# rounding. (The same 32 values are also stored in the file as the gguf KVs pxa.pxq6.book /
# pxa.pxq6.sub — the converter reads those and cross-checks them against these, because
# PXA_PXQ6_BOOK / PXA_PXQ6_SUB can override the compiled-in tables at build time.)
# ---------------------------------------------------------------------------------------------
_BOOK_HEX = (
    "-0x1.f9c0000000000p-1", "-0x1.7880000000000p-1", "-0x1.1e00000000000p-1",
    "-0x1.adc0000000000p-2", "-0x1.3440000000000p-2", "-0x1.8e40000000000p-3",
    "-0x1.8740000000000p-4", "0x0.0p+0",
    "0x1.5b00000000000p-4", "0x1.5ec0000000000p-3", "0x1.0c40000000000p-2",
    "0x1.7140000000000p-2", "0x1.e280000000000p-2", "0x1.3380000000000p-1",
    "0x1.8800000000000p-1", "0x1.0000000000000p+0",
)
_SUB16_HEX = (
    "0x1.b7c0000000000p-3", "0x1.36c0000000000p-2", "0x1.72c0000000000p-2",
    "0x1.a2c0000000000p-2", "0x1.ccc0000000000p-2", "0x1.f300000000000p-2",
    "0x1.0bc0000000000p-1", "0x1.1e00000000000p-1", "0x1.3040000000000p-1",
    "0x1.4380000000000p-1", "0x1.5800000000000p-1", "0x1.6ec0000000000p-1",
    "0x1.8880000000000p-1", "0x1.a640000000000p-1", "0x1.cac0000000000p-1",
    "0x1.f9c0000000000p-1",
)

BOOK = np.array([float.fromhex(h) for h in _BOOK_HEX], dtype=np.float32)
SUB = np.array([float.fromhex(h) for h in _SUB16_HEX], dtype=np.float32)
BOOK.flags.writeable = False
SUB.flags.writeable = False


def check_tables(book: np.ndarray = BOOK, sub: np.ndarray = SUB) -> None:
    """The engine's own table invariants, replicated so a checkpoint-supplied table can be
    validated before anything is emitted against it (ggml-pxq6-tables.h:32,39 document the
    invariants; the CUDA side re-asserts them at upload)."""
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    if book.shape != (16,) or sub.shape != (16,):
        raise ValueError("pxq4 tables: both must have exactly 16 entries")
    if book[7] != 0.0 or book[15] != 1.0:
        raise ValueError("pxq4 book: book[7] must be 0 and book[15] must be 1 (absmax==1)")
    if not np.all(book[:-1] < book[1:]):
        raise ValueError("pxq4 book: must be strictly ascending")
    if not np.all(sub[:-1] < sub[1:]):
        raise ValueError("pxq4 sub: must be strictly ascending")
    if sub[0] <= 0.0:
        raise ValueError("pxq4 sub: SUB16[0] must be > 0")
    # Every entry must be fp16-exact. The quantizer guarantees it (the tables are fp16-snapped
    # at generation), and the dequant depends on it: the anchor is stored fp16, and the CUDA
    # kernels form the same products in fp32 from fp16-exact operands.
    for name, t in (("book", book), ("sub", sub)):
        if not np.array_equal(t.astype(np.float16).astype(np.float32), t):
            raise ValueError(f"pxq4 {name}: entries are not fp16-exact")


def dequant(slabs: np.ndarray, anchor: np.ndarray, *, book: np.ndarray = BOOK,
            sub: np.ndarray = SUB) -> np.ndarray:
    """slabs uint8[P,S,1088], anchor float16[P,64] -> float32[P*64, S*32].

    Reproduces ``pxa_deq_row_pxq6`` (pxq-cpu.c:135-158) exactly, in float32:

        anch    = float32(anchor[p, r])
        sb      = slabs[p, kb, r]
        eff[0]  = anch * sub[sb & 0xF]                 # elements  0..15
        eff[1]  = anch * sub[sb >> 4]                  # elements 16..31
        byte    = slabs[p, kb, 64 + 16*r + b]          # b = 0..15
        w[2b]   = eff[(2b)   >> 4] * book[byte & 0xF]
        w[2b+1] = eff[(2b+1) >> 4] * book[byte >> 4]

    Output row = ``p*64 + r``, column = ``kb*32 + element``.

    (The C writes ``eff[4]`` with ``eff[0]==eff[1]`` and ``eff[2]==eff[3]`` and indexes it
    ``eff[i >> 3]``; that is the same two distinct values selected by element<16 vs >=16.)
    """
    if slabs.dtype != np.uint8 or slabs.ndim != 3 or slabs.shape[2] != SLAB_BYTES:
        raise ValueError(f"pxq4: slabs must be uint8 [P,S,{SLAB_BYTES}], got "
                         f"{slabs.dtype} {slabs.shape}")
    P, S, _ = slabs.shape
    if anchor.shape != (P, PANEL_ROWS):
        raise ValueError(f"pxq4: anchor must be [{P},{PANEL_ROWS}], got {anchor.shape}")

    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)

    anch = anchor.astype(np.float32)                      # fp16 -> fp32 is always exact
    sb = slabs[:, :, :CODE_OFF]                           # [P,S,64] scale SoA, byte r = row r
    eff_lo = anch[:, None, :] * sub[sb & 0x0F]            # [P,S,64] for elements 0..15
    eff_hi = anch[:, None, :] * sub[sb >> 4]              # [P,S,64] for elements 16..31

    codes = slabs[:, :, CODE_OFF:].reshape(P, S, PANEL_ROWS, CODE_BYTES)
    vals = np.empty((P, S, PANEL_ROWS, SLAB_COLS), dtype=np.float32)
    vals[..., 0::2] = book[codes & 0x0F]                  # element 2b   <- low nibble
    vals[..., 1::2] = book[codes >> 4]                    # element 2b+1 <- high nibble

    # Single multiplies, so operand order is irrelevant to the result; what must not change is
    # that eff_lo/eff_hi were already rounded to float32 before being applied.
    vals[..., :16] *= eff_lo[..., None]
    vals[..., 16:] *= eff_hi[..., None]

    # [P,S,64,32] -> [P,64,S,32] -> [P*64, S*32]: the transpose IS the de-interleave. Rows are
    # scattered across slabs on disk and contiguous in the output.
    return vals.transpose(0, 2, 1, 3).reshape(P * PANEL_ROWS, S * SLAB_COLS)


def dequant_scalar(slabs: np.ndarray, anchor: np.ndarray, *, book: np.ndarray = BOOK,
                   sub: np.ndarray = SUB) -> np.ndarray:
    """Deliberately independent scalar transcription of pxa_deq_row_pxq6, written from the C
    control flow rather than from ``dequant``'s index algebra. Slow; used by the tests only.
    Two implementations that disagree localise a transcription bug without a GPU or a DGX."""
    P, S, _ = slabs.shape
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    out = np.zeros((P * PANEL_ROWS, S * SLAB_COLS), dtype=np.float32)
    for p in range(P):
        for r in range(PANEL_ROWS):
            anch = np.float32(anchor[p, r])
            for kb in range(S):
                slab = slabs[p, kb]
                eff = [np.float32(anch * sub[slab[r] & 0xF]),
                       np.float32(anch * sub[slab[r] >> 4])]
                q = slab[CODE_OFF + r * CODE_BYTES: CODE_OFF + (r + 1) * CODE_BYTES]
                dst = out[p * PANEL_ROWS + r, kb * SLAB_COLS:(kb + 1) * SLAB_COLS]
                for b in range(CODE_BYTES):
                    i0, i1 = 2 * b, 2 * b + 1
                    dst[i0] = eff[i0 >> 4] * book[q[b] & 0xF]
                    dst[i1] = eff[i1 >> 4] * book[q[b] >> 4]
    return out


def dequant_blob(blob, N: int, K: int, *, book: np.ndarray = BOOK,
                 sub: np.ndarray = SUB) -> np.ndarray:
    """Convenience: raw GGUF panel blob -> float32[N, K]."""
    from .layout import split_blob
    slabs, anchor = split_blob(blob, N, K)
    return dequant(slabs, anchor, book=book, sub=sub)
