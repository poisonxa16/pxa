#!/usr/bin/env python3
# pxqu_golden.py — Q-G1 byte-parity gate for the PXQ2/PXQ3 native quantizers.
#
# Independently re-implements the PXQ2/PXQ3 quantize (numpy, the pxqu_lab.py quant_cands
# algorithm with the SHIPPED fp16-snapped tables) + the frozen slab packing, runs
# pxqu_ref quant on the same input, and requires byte-identical output.
# Covers: random gaussian slices, imatrix-weighted, scale-outlier rows, exact-zero rows,
# denormal-small rows. Run after building pxqu_ref (see its header).
#
# Usage: python3 pxqu_golden.py /path/to/pxqu_ref
import subprocess, sys, os, tempfile
import numpy as np

REF = sys.argv[1] if len(sys.argv) > 1 else "./pxqu_ref"

def fp16_snap(x): return np.asarray(np.float16(x), np.float64)

# SHIPPED tables (fp16-snapped) — must equal the C headers
LM4 = fp16_snap([-0.70556640625, -0.1876220703125, 0.186767578125, 0.70263671875])
LM8 = fp16_snap([-0.90673828125, -0.5478515625, -0.2978515625, -0.0931396484375,
                 0.0919189453125, 0.295654296875, 0.54541015625, 0.90576171875])
SUB16_RAW = [0.21474449881887758, 0.3034315324052146, 0.3621376080689431, 0.409029341443377,
             0.4499691625865238, 0.4874126695940804, 0.5231273281861334, 0.5585506918793897,
             0.5944712912749424, 0.6319333068553891, 0.6720072088379927, 0.7165121390870518,
             0.7665223658027503, 0.8247180704599771, 0.89578060308503, 0.9878587276071165]
SUB16 = fp16_snap(SUB16_RAW)   # the C quantizer uses the fp16-snapped LUT (PXQ6_SUB16_INIT)
ZIDX = {2: 2, 3: 4}
BOOK = {2: LM4, 3: LM8}

def quant_slice_np(W, sig, book, zidx):
    """pxqu_lab.py quant_cands semantics with the shipped snapped tables + the C zero-block
    convention (s4=0, codes=zidx for amax==0 blocks). Returns anchors(fp16 f32), s4, codes."""
    R, K = W.shape
    b32 = book.astype(np.float32); b64 = book
    mids = (b64[1:] + b64[:-1]) / 2
    rmax = fp16_snap(np.abs(W).max(axis=1))                    # per-row fp16 anchor (as f64)
    Wb = W.reshape(-1, 16).astype(np.float64)
    anch = np.repeat(rmax, K // 16)
    wcol = np.tile(sig.reshape(-1, 16), (R, 1)).astype(np.float64) if sig is not None else None
    cand = (anch[:, None] * SUB16[None, :]).astype(np.float32)  # f64 product, single f32 round
    best_err = None; best_s = None; best_c = None
    for j in range(16):
        d32 = cand[:, j]
        d64 = np.maximum(d32.astype(np.float64), 1e-30)
        c = np.searchsorted(mids, Wb / d64[:, None]).astype(np.uint8)
        rec = d32[:, None] * b32[c]                             # fp32 product == kernel math
        e = (Wb - rec.astype(np.float64)) ** 2
        if wcol is not None: e = e * wcol
        # sequential left-fold, NOT e.sum(): numpy's pairwise/unrolled reduction differs from
        # the C quantizer's sequential double accumulation in the last ulp, which can flip
        # near-tie argmin picks and fail the BYTE gate spuriously.
        err = e[:, 0].copy()
        for i in range(1, 16): err = err + e[:, i]
        if best_err is None:
            best_err, best_s, best_c = err, np.full(err.shape, j, np.uint8), c
        else:
            m = err < best_err
            best_err = np.where(m, err, best_err)
            best_s = np.where(m, np.uint8(j), best_s)
            best_c = np.where(m[:, None], c, best_c)
    zero = np.abs(Wb).max(axis=1) == 0
    best_s = np.where(zero, np.uint8(0), best_s)
    best_c = np.where(zero[:, None], np.uint8(zidx), best_c)
    zrow = np.repeat(rmax == 0, K // 16)                        # zero rows: same convention
    best_s = np.where(zrow, np.uint8(0), best_s)
    best_c = np.where(zrow[:, None], np.uint8(zidx), best_c)
    return rmax, best_s.reshape(R, K // 16), best_c.reshape(R, K)

def pack(type_, R, K, rmax, s4, codes):
    slab_bytes = 576 if type_ == 2 else 832
    code_b = 8 if type_ == 2 else 12
    KB, P = K // 32, R // 64
    out = bytearray(P * (128 + KB * slab_bytes))
    off = 0
    for p in range(P):
        hdr = np.asarray(np.float16(rmax[p*64:(p+1)*64]), np.float16).tobytes()
        out[off:off+128] = hdr
        for kb in range(KB):
            so = off + 128 + kb * slab_bytes
            for r in range(64):
                row = p*64 + r
                s = s4[row, kb*2:kb*2+2]
                out[so + r] = int(s[0]) | (int(s[1]) << 4)
                c = codes[row, kb*32:(kb+1)*32].astype(np.uint32)
                if type_ == 2:
                    w0 = int((c[:16] << (2*np.arange(16, dtype=np.uint32))).sum())
                    w1 = int((c[16:] << (2*np.arange(16, dtype=np.uint32))).sum())
                    out[so + 64 + r*8: so + 64 + r*8 + 8] = \
                        w0.to_bytes(4, "little") + w1.to_bytes(4, "little")
                else:
                    lo0 = int(((c[:16] & 3)  << (2*np.arange(16, dtype=np.uint32))).sum())
                    lo1 = int(((c[16:] & 3)  << (2*np.arange(16, dtype=np.uint32))).sum())
                    hi  = int(((c[:16] >> 2) << np.arange(16, dtype=np.uint32)).sum()) | \
                          (int(((c[16:] >> 2) << np.arange(16, dtype=np.uint32)).sum()) << 16)
                    out[so + 64 + r*12: so + 64 + r*12 + 12] = \
                        lo0.to_bytes(4, "little") + lo1.to_bytes(4, "little") + hi.to_bytes(4, "little")
        off += 128 + KB * slab_bytes
    return bytes(out)

def run_case(name, type_, W, sig):
    R, K = W.shape
    with tempfile.TemporaryDirectory() as td:
        wf, sf, qf = f"{td}/w.f32", f"{td}/sig.f32", f"{td}/q.bin"
        W.astype(np.float32).tofile(wf)
        if sig is not None: sig.astype(np.float32).tofile(sf)
        subprocess.run([REF, "quant", str(type_), str(R), str(K), wf, sf if sig is not None else "-", qf],
                       check=True)
        got = open(qf, "rb").read()
    rmax, s4, codes = quant_slice_np(W.astype(np.float64), sig, BOOK[type_], ZIDX[type_])
    want = pack(type_, R, K, rmax, s4, codes)
    ok = got == want
    if not ok:
        diff = next(i for i in range(min(len(got), len(want))) if got[i] != want[i])
        print(f"  FAIL {name}: first byte diff at {diff} (got {got[diff]:02x} want {want[diff]:02x}), "
              f"len {len(got)}/{len(want)}")
    else:
        print(f"  PASS {name} ({len(got)} bytes)")
    return ok

rng = np.random.default_rng(1337)
allok = True
for type_ in (2, 3):
    print(f"PXQ{type_}:")
    W = rng.standard_normal((128, 64)).astype(np.float32) * 0.02
    allok &= run_case("random-unweighted", type_, W, None)
    sig = (rng.random(64).astype(np.float32) * 4 + 0.1)
    allok &= run_case("random-imatrix", type_, W, sig)
    W2 = W.copy(); W2[3, :] *= 300.0; W2[77, 5] = 4.0        # outlier rows
    allok &= run_case("outlier-rows", type_, W2, sig)
    W3 = W.copy(); W3[10, :] = 0.0; W3[64, :] = 0.0          # exact-zero rows
    allok &= run_case("zero-rows", type_, W3, sig)
    W4 = (rng.standard_normal((64, 32)).astype(np.float32) * 1e-24)   # denormal-small
    allok &= run_case("tiny-denormal", type_, W4, None)
print("Q-G1", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
