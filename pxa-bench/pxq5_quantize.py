#!/usr/bin/env python3
# RETIRED 2026-07-21: type id 251 (PXQ5) was REMOVED from the fork — the engine refuses
# id-251 files. Do NOT produce new PXQ5 files; kept only as a historical reference.
# pxq5_quantize.py — PXQ5 reference quantizer / dequantizer / book learner (spec v1, frozen 2026-07-16).
#
# LEGACY (2026-07-19): PXQ5 is superseded by the 4-bit tier now displayed PXQ4 (formerly PXQ6),
# which adds E16-row scales on the same numerics. Kept for reproducibility.
#
# PXQ5 = the PXA fully-proprietary 4-bit quant: LEARNED numerics in the legacy slab LAYOUT. 4.25 bpw.
#   value(code c, scale byte s) = scale_tab[s] * book[c]
#   - book: 16 fp16 levels (default PX16 = pooled Lloyd-Max on real MoE expert weights),
#     sorted ascending, book[7] == 0, absmax == 1. Frozen u16 bit patterns below.
#   - scale byte: s == 0 -> 0.0 (all-zero block); s in [1,255] -> 2^((s-160)/8)
#     (log-uniform, 2^(1/8) steps). fp32-exact table, sha256-locked below.
#   - quantize: per 32-block, scale candidates {fit-2..fit+1} (fit = smallest s covering absmax),
#     RTN against the sorted book (midpoint rule), keep the (imatrix-weighted) min-MSE scale.
#   - slab layout: IDENTICAL to PXQ4 (64 B scale SoA + 64 rows x 16 B nibbles, sequential pairs,
#     K-major slabs in 64-row panels, experts outermost; 17 B / 32 elems).
#
# This file is the BIT-PARITY REFERENCE for the C++ quantizer (llama-quantize PXQ5) and the
# CUDA kernels (pxq5.cuh): same tables, same double-precision error accumulation, same
# assignment rule.
#
# Usage:
#   pxq5_quantize.py --selftest
#   pxq5_quantize.py --learn-book t0.npy t1.npy ...   (each: 2D f32/f16 weight array)
import argparse, hashlib, sys
import numpy as np

QK, BM, SLAB = 32, 64, 1088

# ---- frozen tables ----
_BOOK_U16 = [48103, 47586, 47224, 46775, 46289, 45625, 44573, 0,
             11628, 12667, 13361, 13765, 14218, 14542, 14880, 15360]
BOOK16 = np.array(_BOOK_U16, dtype=np.uint16).view(np.float16)
BOOK32 = BOOK16.astype(np.float32)
BOOK64 = BOOK32.astype(np.float64)
BMAX = float(np.abs(BOOK64).max())            # == 1.0
_MIDS = (BOOK64[1:] + BOOK64[:-1]) / 2

SCALE_TAB = np.array([0.0] + [np.float32(2.0 ** ((s - 160) / 8.0)) for s in range(1, 256)],
                     dtype=np.float32)
_TAB_SHA = "35524b0008e4b3991d947d268932135d172b0211e22f13c2eb2e65411cdc8b07"
assert hashlib.sha256(SCALE_TAB.tobytes()).hexdigest() == _TAB_SHA, \
    "PXQ5 scale table drifted on this platform — use the frozen literal table from ggml-pxq5-tables.h"

def set_book(book16):
    """Install a custom (per-model) book: 16 fp16 values, sorted, containing 0."""
    global BOOK16, BOOK32, BOOK64, BMAX, _MIDS
    BOOK16 = np.asarray(book16, dtype=np.float16)
    assert BOOK16.size == 16 and np.all(np.diff(BOOK16.astype(np.float64)) > 0) and 0.0 in BOOK16
    BOOK32 = BOOK16.astype(np.float32); BOOK64 = BOOK32.astype(np.float64)
    BMAX = float(np.abs(BOOK64).max()); _MIDS = (BOOK64[1:] + BOOK64[:-1]) / 2

def se8_fit(amax):
    with np.errstate(divide="ignore"):
        s = np.ceil(8.0 * np.log2(np.maximum(amax, 1e-30) / BMAX)) + 160.0
    return np.clip(s, 1, 255).astype(np.int32)

def quantize_blocks(Wb, wcol=None):
    """Wb (nb,32) f32/f64 -> (scale_bytes (nb,), codes (nb,32)). Deterministic."""
    Wb = np.asarray(Wb, dtype=np.float64)
    amax = np.abs(Wb).max(axis=1)
    s0 = se8_fit(amax)
    zero = amax == 0
    best_err = best_s = best_c = None
    for k in (-2, -1, 0, 1):
        s = np.clip(s0 + k, 1, 255)
        d = SCALE_TAB[s].astype(np.float64)
        c = np.searchsorted(_MIDS, Wb / d[:, None]).astype(np.uint8)
        rec = (SCALE_TAB[s][:, None] * BOOK32[c]).astype(np.float64)   # fp32 product = kernel math
        e = (Wb - rec) ** 2
        if wcol is not None:
            e = e * wcol
        err = e.sum(axis=1)
        if best_err is None:
            best_err, best_s, best_c = err, s.copy(), c.copy()
        else:
            m = err < best_err
            best_err = np.where(m, err, best_err)
            best_s = np.where(m, s, best_s)
            best_c = np.where(m[:, None], c, best_c)
    best_s = np.where(zero, 0, best_s).astype(np.uint8)
    best_c = np.where(zero[:, None], np.uint8(int(np.argmin(np.abs(BOOK64)))), best_c)
    return best_s, best_c

def dequant_blocks(scales, codes):
    return (SCALE_TAB[scales][:, None] * BOOK32[codes]).astype(np.float32)

def pack_slabs(scales, codes, R, K):
    KB, P = K // QK, R // BM
    s2 = scales.reshape(R, KB)
    c2 = codes.reshape(R, KB, QK)
    out = np.empty((P, KB, SLAB), dtype=np.uint8)
    for p in range(P):
        rs = slice(p * BM, (p + 1) * BM)
        out[p, :, :64] = s2[rs].T
        cc = c2[rs]
        nib = (cc[:, :, 0::2] | (cc[:, :, 1::2] << 4)).astype(np.uint8)
        out[p, :, 64:] = nib.transpose(1, 0, 2).reshape(KB, BM * 16)
    return out.reshape(-1)

def unpack_slabs(buf, R, K):
    KB, P = K // QK, R // BM
    slabs = buf.reshape(P, KB, SLAB)
    scales = slabs[:, :, :64].transpose(0, 2, 1).reshape(R, KB)
    nib = slabs[:, :, 64:].reshape(P, KB, BM, 16).transpose(0, 2, 1, 3)
    codes = np.empty((P, BM, KB, QK), dtype=np.uint8)
    codes[..., 0::2] = nib & 0x0F
    codes[..., 1::2] = nib >> 4
    return scales.reshape(-1), codes.reshape(R * KB, QK)

def quantize_tensor(W, wcol_blocks=None):
    R, K = W.shape
    assert R % BM == 0 and K % QK == 0, (R, K)
    s, c = quantize_blocks(W.reshape(-1, QK), wcol_blocks)
    return pack_slabs(s, c, R, K)

def dequant_tensor(buf, R, K):
    s, c = unpack_slabs(np.asarray(buf, dtype=np.uint8), R, K)
    return dequant_blocks(s, c).reshape(R, K)

# ---- per-model book learner (offline; pass result via PXA_PXQ5_BOOK to quantizer+runtime) ----
def learn_book(sample_blocks, iters=60, rounds=3):
    book = BOOK64.copy()
    for _ in range(rounds):
        amax = np.abs(sample_blocks).max(axis=1)
        d = SCALE_TAB[np.clip(se8_fit(amax), 1, 255)].astype(np.float64)
        x = np.sort((sample_blocks / d[:, None]).ravel())
        for _ in range(iters):
            mids = (book[1:] + book[:-1]) / 2
            idx = np.searchsorted(mids, x)
            sums = np.bincount(idx, weights=x, minlength=16)
            cnts = np.bincount(idx, minlength=16)
            book = np.where(cnts > 0, sums / np.maximum(cnts, 1), book)
            book[np.argmin(np.abs(book))] = 0.0
            book = np.sort(book)
        book = book / np.abs(book).max()
    return book.astype(np.float16)

def _selftest():
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.03, (128, 96)).astype(np.float32)
    buf = quantize_tensor(W)
    assert buf.size == W.size * 17 // 32
    s, c = quantize_blocks(W.reshape(-1, QK))
    assert np.array_equal(dequant_tensor(buf, 128, 96), dequant_blocks(s, c).reshape(W.shape)), \
        "pack/unpack not transparent"
    # zero-block + clamp edges
    W2 = np.zeros((64, 32), np.float32); W2[0, 0] = 65504.0; W2[1, 0] = 1e-9
    b2 = quantize_tensor(W2)
    Wh2 = dequant_tensor(b2, 64, 32)
    assert Wh2[2, 0] == 0.0 and np.isfinite(Wh2).all()
    rel = np.linalg.norm(dequant_tensor(buf, 128, 96) - W) / np.linalg.norm(W)
    print(f"SELFTEST PASS  (gaussian rel_l2={rel:.4f}, 4.25 bpw, layout transparent)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--learn-book", nargs="+", metavar="NPY")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.learn_book:
        blocks = []
        rng = np.random.default_rng(0)
        for f in args.learn_book:
            W = np.load(f).astype(np.float64)
            Wb = W.reshape(-1, QK)
            take = min(40_000, Wb.shape[0])
            blocks.append(Wb[rng.choice(Wb.shape[0], take, replace=False)])
        book = learn_book(np.concatenate(blocks))
        print("PXA_PXQ5_BOOK=" + ",".join(repr(float(v)) for v in book))
    else:
        ap.print_help()
