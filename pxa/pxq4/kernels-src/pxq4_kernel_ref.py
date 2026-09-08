"""pxq4_kernel_ref.py — numpy CPU reference for PXQ4 (ggml type id 252), plus the panel/slab
layout arithmetic and a synthetic encoder used only by the tests.

This module is the bit-exactness ORACLE for the CUDA extension. It has no torch, no CUDA and
no vLLM dependency, and it must stay that way: gates G1/G3 run here, on any machine, before
the GPU lease is ever taken.

RELATIONSHIP TO THE PLAN. Plan §6.2 (`src/pxq4_vllm/layout.py`) and §6.3
(`src/pxq4_vllm/reference.py`) are owned by agent A. The names and semantics below are the
ones those files must expose, and they are reproduced here verbatim rather than imported so
that this component can be validated standalone. When both exist they must be diffed, not
forked: `dequant`, `BOOK`, `SUB`, `split_blob`, `join_blob`, `panel_bytes`, `tensor_bytes`,
`assert_geometry`, `slab_shape`, `anchor_shape` are the shared contract surface.

FORMAT, restated so this file is self-contained:

    panel (64 weight rows) = 128 B anchor header (64 x fp16, anchor[r] at byte 2*r)
                           + (K/32) slabs of 1088 B, K-major
    slab  (32 columns)     = 64 B sub-scale SoA (byte r belongs to row r:
                               low nibble  -> elements  0..15 of this 32-column block,
                               high nibble -> elements 16..31)
                           + 64 x 16 B code rows (row r at slab[64 + 16*r];
                               byte b holds code(2b) in its low nibble, code(2b+1) in its high)

    dequant, PARITY-LOCKED (ggml/src/pxq-cpu.c:135-158, ggml/src/ggml-cuda/pxq6.cuh:326-331):
        eff = float32(anchor_fp16) * SUB[s4]        once per (row, 16-element block)
        w   = eff * float32(BOOK[code])             per element

    The multiply order is load-bearing. Do not reassociate to anchor*(sub*book).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------
# geometry (plan §6.2)
# ---------------------------------------------------------------------------------------------
PANEL_ROWS = 64
SLAB_COLS = 32
SLAB_BYTES = 1088
HEADER_BYTES = 128
CODE_OFF = 64
CODE_BYTES = 16
TYPE_ID = 252


def panel_bytes(K: int) -> int:
    """Bytes occupied by one 64-row panel of a K-wide tensor."""
    return HEADER_BYTES + (K // SLAB_COLS) * SLAB_BYTES


def tensor_bytes(N: int, K: int) -> int:
    """On-disk size of an [N, K] PXQ4 tensor. Equals ggml's 2*N + 17*N*K/32."""
    return (N // PANEL_ROWS) * panel_bytes(K)


def slab_shape(N: int, K: int) -> tuple[int, int, int]:
    return (N // PANEL_ROWS, K // SLAB_COLS, SLAB_BYTES)


def anchor_shape(N: int) -> tuple[int, int]:
    return (N // PANEL_ROWS, PANEL_ROWS)


def assert_geometry(N: int, K: int) -> None:
    """The quantizer's geometry gate. A tensor that fails it was demoted to another type at
    quantize time and is NOT PXQ4 in the artifact, so any caller that reaches here with a
    failing shape has a name-mapping bug, not a layout bug."""
    if N <= 0 or K <= 0:
        raise ValueError(f"pxq4 geometry: non-positive shape N={N} K={K}")
    if N % PANEL_ROWS:
        raise ValueError(f"pxq4 geometry: N={N} is not a multiple of {PANEL_ROWS} (panel rows)")
    if K % SLAB_COLS:
        raise ValueError(f"pxq4 geometry: K={K} is not a multiple of {SLAB_COLS} (slab columns)")


def split_blob(blob, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF blob -> (slabs uint8 [P,S,1088], anchor float16 [P,64]).

    A pure partition: no byte is reordered and no value is recomputed. This is the whole of
    plan §5.3, and `join_blob(*split_blob(b)) == b` is gate G2."""
    assert_geometry(N, K)
    P, S = N // PANEL_ROWS, K // SLAB_COLS
    a = np.frombuffer(blob, dtype=np.uint8)
    if a.size != tensor_bytes(N, K):
        raise ValueError(f"pxq4: blob is {a.size} B, expected {tensor_bytes(N, K)} B for [{N},{K}]")
    a = a.reshape(P, HEADER_BYTES + S * SLAB_BYTES)
    anchor = a[:, :HEADER_BYTES].copy().view("<f2")
    slabs = a[:, HEADER_BYTES:].copy().reshape(P, S, SLAB_BYTES)
    return slabs, anchor


def join_blob(slabs: np.ndarray, anchor: np.ndarray) -> bytes:
    """Inverse of split_blob."""
    P, S, sb = slabs.shape
    if sb != SLAB_BYTES:
        raise ValueError(f"pxq4: slab stride {sb} != {SLAB_BYTES}")
    if anchor.shape != (P, PANEL_ROWS) or anchor.dtype != np.float16:
        raise ValueError(f"pxq4: anchor must be float16 {(P, PANEL_ROWS)}, got {anchor.dtype} {anchor.shape}")
    hdr = np.ascontiguousarray(anchor).view(np.uint8).reshape(P, HEADER_BYTES)
    return np.concatenate([hdr, slabs.reshape(P, S * SLAB_BYTES)], axis=1).tobytes()


# ---------------------------------------------------------------------------------------------
# frozen tables — transcribed from ggml/include/ggml-pxq6-tables.h:33-44 as C99 hex float
# literals, exactly as they appear in the header, so the transcription is diffable by eye and
# cannot drift through decimal rounding. float.fromhex parses the same grammar C uses.
# ---------------------------------------------------------------------------------------------
_BOOK_HEX = [
    "-0x1.f9c0000000000p-1", "-0x1.7880000000000p-1", "-0x1.1e00000000000p-1", "-0x1.adc0000000000p-2",
    "-0x1.3440000000000p-2", "-0x1.8e40000000000p-3", "-0x1.8740000000000p-4", "0x0.0p+0",
    "0x1.5b00000000000p-4", "0x1.5ec0000000000p-3", "0x1.0c40000000000p-2", "0x1.7140000000000p-2",
    "0x1.e280000000000p-2", "0x1.3380000000000p-1", "0x1.8800000000000p-1", "0x1.0000000000000p+0",
]
_SUB16_HEX = [
    "0x1.b7c0000000000p-3", "0x1.36c0000000000p-2", "0x1.72c0000000000p-2", "0x1.a2c0000000000p-2",
    "0x1.ccc0000000000p-2", "0x1.f300000000000p-2", "0x1.0bc0000000000p-1", "0x1.1e00000000000p-1",
    "0x1.3040000000000p-1", "0x1.4380000000000p-1", "0x1.5800000000000p-1", "0x1.6ec0000000000p-1",
    "0x1.8880000000000p-1", "0x1.a640000000000p-1", "0x1.cac0000000000p-1", "0x1.f9c0000000000p-1",
]

BOOK = np.array([float.fromhex(h) for h in _BOOK_HEX], dtype=np.float32)
SUB = np.array([float.fromhex(h) for h in _SUB16_HEX], dtype=np.float32)


def check_tables(book: np.ndarray = BOOK, sub: np.ndarray = SUB) -> None:
    """The engine's own startup self-check (ggml/src/ggml-cuda/pxq6.cuh:98-108), replicated so a
    checkpoint-supplied table can be validated before it is uploaded to the device."""
    if book.shape != (16,) or sub.shape != (16,):
        raise ValueError("pxq4 tables: both must be 16 entries")
    if not (book[7] == 0.0 and book[15] == 1.0):
        raise ValueError("pxq4 book: book[7] must be 0 and book[15] must be 1")
    if not np.all(book[:-1] < book[1:]):
        raise ValueError("pxq4 book: must be strictly ascending")
    if not np.all(sub[:-1] < sub[1:]):
        raise ValueError("pxq4 sub: must be strictly ascending")
    if not sub[0] > 0.0:
        raise ValueError("pxq4 sub: sub[0] must be > 0")
    # fp16-snap idempotence: every table entry must survive a float32 -> float16 -> float32
    # round trip unchanged. The quantizer guarantees this; the dequant contract depends on it
    # because the anchor is stored as fp16 and the effective scale must be exactly
    # representable in the products the kernels form.
    for name, t in (("book", book), ("sub", sub)):
        rt = t.astype(np.float16).astype(np.float32)
        if not np.array_equal(rt, t.astype(np.float32)):
            raise ValueError(f"pxq4 {name}: entries are not fp16-exact")


# ---------------------------------------------------------------------------------------------
# dequant (plan §6.3)
# ---------------------------------------------------------------------------------------------
def dequant(slabs: np.ndarray, anchor: np.ndarray, *, book: np.ndarray = BOOK,
            sub: np.ndarray = SUB) -> np.ndarray:
    """slabs uint8[P,S,1088], anchor float16[P,64] -> float32[P*64, S*32].

    Reproduces pxa_deq_row_pxq6 (ggml/src/pxq-cpu.c:135-158) exactly, in float32:
        anch   = float32(anchor[p, r])
        sb     = slabs[p, kb, r]
        eff[0] = anch * sub[sb & 0xF]      # elements 0..15
        eff[1] = anch * sub[sb >> 4]       # elements 16..31
        byte   = slabs[p, kb, 64 + 16*r + b]           b = 0..15
        w[2b]   = eff[(2b)  >> 4] * book[byte & 0xF]
        w[2b+1] = eff[(2b+1)>> 4] * book[byte >> 4]
    Output row = p*64 + r, column = kb*32 + element.
    """
    if slabs.ndim != 3 or slabs.shape[2] != SLAB_BYTES or slabs.dtype != np.uint8:
        raise ValueError(f"pxq4: slabs must be uint8 [P,S,{SLAB_BYTES}], got {slabs.dtype} {slabs.shape}")
    P, S, _ = slabs.shape
    if anchor.shape != (P, PANEL_ROWS):
        raise ValueError(f"pxq4: anchor must be [{P},{PANEL_ROWS}], got {anchor.shape}")

    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)

    anch = anchor.astype(np.float32)                              # [P,64] — half->float is exact
    sb = slabs[:, :, :CODE_OFF]                                   # [P,S,64] scale SoA
    eff_lo = anch[:, None, :] * sub[sb & 0x0F]                    # [P,S,64] elements 0..15
    eff_hi = anch[:, None, :] * sub[sb >> 4]                      # [P,S,64] elements 16..31

    codes = slabs[:, :, CODE_OFF:].reshape(P, S, PANEL_ROWS, CODE_BYTES)
    vals = np.empty((P, S, PANEL_ROWS, SLAB_COLS), dtype=np.float32)
    vals[..., 0::2] = book[codes & 0x0F]                          # element 2b <- low nibble
    vals[..., 1::2] = book[codes >> 4]                            # element 2b+1 <- high nibble

    # A single multiply is order-independent in IEEE-754, so eff*book and book*eff agree
    # bit-for-bit; what must NOT change is that `eff` is formed first, as its own rounded
    # float32 product of anchor and sub.
    vals[..., :16] *= eff_lo[..., None]
    vals[..., 16:] *= eff_hi[..., None]

    # [P,S,64,32] -> [P,64,S,32] -> [P*64, S*32]
    return vals.transpose(0, 2, 1, 3).reshape(P * PANEL_ROWS, S * SLAB_COLS)


def dequant_naive(slabs: np.ndarray, anchor: np.ndarray, *, book: np.ndarray = BOOK,
                  sub: np.ndarray = SUB) -> np.ndarray:
    """Deliberately independent scalar transcription of pxa_deq_row_pxq6, written from the C
    source rather than from `dequant` above. Slow, and used only to cross-check `dequant`'s
    vectorised index algebra — two implementations that disagree localise a transcription bug
    that a single implementation cannot see."""
    P, S, _ = slabs.shape
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    out = np.zeros((P * PANEL_ROWS, S * SLAB_COLS), dtype=np.float32)
    for p in range(P):
        for r in range(PANEL_ROWS):
            a = np.float32(anchor[p, r])
            for kb in range(S):
                slab = slabs[p, kb]
                eff = [a * sub[slab[r] & 0x0F], a * sub[slab[r] >> 4]]
                q = slab[CODE_OFF + CODE_BYTES * r: CODE_OFF + CODE_BYTES * (r + 1)]
                for b in range(16):
                    i0, i1 = 2 * b, 2 * b + 1
                    out[p * PANEL_ROWS + r, kb * SLAB_COLS + i0] = eff[i0 >> 4] * book[q[b] & 0x0F]
                    out[p * PANEL_ROWS + r, kb * SLAB_COLS + i1] = eff[i1 >> 4] * book[q[b] >> 4]
    return out


# ---------------------------------------------------------------------------------------------
# the mmv fold, replicated in float32 (validation oracle for k_pxq4_mmv)
# ---------------------------------------------------------------------------------------------
MMV_KSEG = 4
CANON_CMAX = 16


def canon_nfix(kslabs: int, cmax: int = CANON_CMAX) -> int:
    """pxq6_canon_nfix, ggml/src/ggml-cuda/pxq6.cuh:826-833. Shape-only by construction — that
    is what makes a K-split launch bit-identical to an unsplit one, and it is why the vLLM
    kernel reproduces the same chunking."""
    lim = kslabs // MMV_KSEG
    lim = max(1, min(lim, cmax))
    n = 1
    while n * 2 <= lim:
        n *= 2
    return n


def mmv_fold(slabs: np.ndarray, anchor: np.ndarray, x: np.ndarray, *, book: np.ndarray = BOOK,
             sub: np.ndarray = SUB) -> np.ndarray:
    """out[M,N] float32 = x[M,K] * W[N,K]^T, folded in the EXACT order k_pxq4_mmv uses.

    Order matters and is the whole point of this function:
      * lane (kseg) l accumulates slabs l, l+4, l+8 ... within each canonical chunk;
      * chunk partials are added into `su` in ascending chunk order;
      * the 4 lane partials are then summed in ascending lane order;
      * within one 32-element block, t[0] accumulates byte pairs 0..7 and t[1] pairs 8..15,
        each as `acc + (a0*x0 + a1*x1)` (PXQ4_CANON_V2 == 0, the shipping form), and the block
        result is eff[0]*t[0] + eff[1]*t[1].
    Every operation is performed in float32. x is taken as float16 and widened, which is what
    the kernel does when staging into shared memory."""
    P, S, _ = slabs.shape
    N, K = P * PANEL_ROWS, S * SLAB_COLS
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    xf = np.asarray(x, dtype=np.float16).astype(np.float32)
    M = xf.shape[0]
    nfix = canon_nfix(S)

    f32 = np.float32
    out = np.zeros((M, N), dtype=np.float32)
    for iy in range(M):
        xv = xf[iy]
        for p in range(P):
            for row in range(PANEL_ROWS):
                anch = f32(anchor[p, row])
                lane = [f32(0.0)] * MMV_KSEG
                for kseg in range(MMV_KSEG):
                    su = f32(0.0)
                    for c in range(nfix):
                        b0 = (S * c) // nfix
                        b1 = (S * (c + 1)) // nfix
                        t = f32(0.0)
                        for kb in range(b0 + kseg, b1, MMV_KSEG):
                            slab = slabs[p, kb]
                            eff0 = anch * sub[slab[row] & 0x0F]
                            eff1 = anch * sub[slab[row] >> 4]
                            q = slab[CODE_OFF + CODE_BYTES * row: CODE_OFF + CODE_BYTES * (row + 1)]
                            xk = xv[kb * SLAB_COLS: (kb + 1) * SLAB_COLS]
                            acc = [f32(0.0), f32(0.0)]
                            for b in range(16):
                                a0 = book[q[b] & 0x0F]
                                a1 = book[q[b] >> 4]
                                j = (b * 2) >> 4          # NEFF == 2 -> j == (b >= 8)
                                acc[j] = f32(acc[j] + f32(f32(a0 * xk[2 * b]) + f32(a1 * xk[2 * b + 1])))
                            t = f32(t + f32(f32(eff0 * acc[0]) + f32(eff1 * acc[1])))
                        su = f32(su + t)
                    lane[kseg] = su
                u = f32(0.0)
                for s in range(MMV_KSEG):
                    u = f32(u + lane[s])
                out[iy, p * PANEL_ROWS + row] = u
    return out


# ---------------------------------------------------------------------------------------------
# synthetic encoder — TESTS ONLY
# ---------------------------------------------------------------------------------------------
def pack_pxq4(anchor: np.ndarray, sub_idx: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (slabs, anchor) from explicit fields.

    anchor   float16 [P,64]
    sub_idx  uint8   [P,S,64,2]   -> [...,0] governs elements 0..15, [...,1] elements 16..31
    codes    uint8   [P,64,S*32]  book indices 0..15, in NATURAL element order

    Exists so the tests can drive known bit patterns through both the numpy reference and the
    real kernel source. It is NOT a quantizer: the production encoder is the engine's
    pxq6_quantize_expert (imatrix-calibrated, deterministic tie-break seeded by the absolute
    row offset), and nothing here may ever be used to produce a shipping artifact."""
    P, S, _, _ = sub_idx.shape
    if anchor.shape != (P, PANEL_ROWS) or anchor.dtype != np.float16:
        raise ValueError("pack_pxq4: anchor must be float16 [P,64]")
    if codes.shape != (P, PANEL_ROWS, S * SLAB_COLS):
        raise ValueError("pack_pxq4: codes must be [P,64,S*32]")
    if codes.max(initial=0) > 15 or sub_idx.max(initial=0) > 15:
        raise ValueError("pack_pxq4: codes and sub indices are 4-bit")

    slabs = np.zeros((P, S, SLAB_BYTES), dtype=np.uint8)
    slabs[:, :, :CODE_OFF] = (sub_idx[..., 0] | (sub_idx[..., 1] << 4)).astype(np.uint8)
    c = codes.reshape(P, PANEL_ROWS, S, SLAB_COLS).transpose(0, 2, 1, 3)   # [P,S,64,32]
    packed = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)         # [P,S,64,16]
    slabs[:, :, CODE_OFF:] = packed.reshape(P, S, PANEL_ROWS * CODE_BYTES)
    return slabs, anchor


def quantize_pxq4_naive(w: np.ndarray, rng: np.random.Generator | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-value encoder for [N,K] float32 -> (slabs, anchor). TESTS ONLY (see pack_pxq4).

    Per row: anchor = fp16(absmax). Per 16-element block: pick the SUB entry whose
    anchor*sub*book grid minimises squared error, then the nearest book code per element.
    Good enough to produce realistic bit patterns; it makes no claim to match the engine's
    imatrix-weighted optimiser, and it must never be used to build a checkpoint."""
    del rng
    N, K = w.shape
    assert_geometry(N, K)
    P, S = N // PANEL_ROWS, K // SLAB_COLS
    w = np.asarray(w, dtype=np.float32)

    absmax = np.abs(w).max(axis=1)
    anchor = absmax.astype(np.float16)
    anch32 = anchor.astype(np.float32)

    blocks = w.reshape(N, K // 16, 16)                                    # 16-element groups
    cand = anch32[:, None, None, None] * SUB[None, None, :, None] * BOOK[None, None, None, :]
    # err[n, g, s] = sum over the 16 elements of (w - nearest grid point)^2
    d = np.abs(blocks[:, :, None, :, None] - cand[:, :, :, None, :])      # [N,G,16sub,16elem,16book]
    best_code = d.argmin(axis=4).astype(np.uint8)                         # [N,G,16sub,16elem]
    err = (d.min(axis=4) ** 2).sum(axis=3)                                # [N,G,16sub]
    s_idx = err.argmin(axis=2).astype(np.uint8)                           # [N,G]
    codes = np.take_along_axis(best_code, s_idx[:, :, None, None], axis=2)[:, :, 0, :]
    codes = codes.reshape(N, K).astype(np.uint8)

    # [N, K/16] sub indices -> [P, S, 64, 2] (two 16-element groups per 32-column slab)
    si = s_idx.reshape(P, PANEL_ROWS, S, 2).transpose(0, 2, 1, 3).astype(np.uint8)
    cd = codes.reshape(P, PANEL_ROWS, K)
    return pack_pxq4(anchor.reshape(P, PANEL_ROWS), si, cd)
