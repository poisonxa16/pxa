#!/usr/bin/env python3
# pxqu_wrel.py — Q-G2 gate: the C PXQ2/PXQ3 quantizers must reproduce the numpy lab's wrel
# on the frozen 36-slice rng-42 eval protocol.
#
# Protocol = pxqu_lab.py stage_eval conditions: EVAL_LAYERS [5,16,25,35] x 3 proj x 3 experts,
# production imatrix (ornith-maxquality), map_gate=True, no subsample. For each slice the C
# binary (pxqu_ref roundtrip) produces the reconstruction; wrel = mean over slices of
# sqrt(sum sig*(W-rec)^2 / sum sig*W^2).
#
# TWO comparisons, in order of strictness:
#   (1) C vs the SNAPPED-SUB16 numpy oracle (same tables as the C code): |d| <= 1e-6 —
#       this is the true implementation-parity check (identical algorithm, identical tables).
#   (2) C vs the FROZEN lab oracle numbers (computed with the raw float64 SUB16 the lab used):
#       |d| <= 1e-4 per the pre-registered Q-G2 gate. If (1) passes but (2) fails, the delta
#       is pure SUB16 fp16-snap displacement — report both and stop for a human call; do NOT
#       widen the gate silently.
#
# Frozen lab oracle (eval.json, e-row-2bit-experts-mixed):
#   PXQ2 (b2_e16): wrel 0.3020488067298746
#   PXQ3 (b3_e16): wrel 0.14353147387217413
#
# Usage: PXQ_LAB_DIRS=/path/to/lab1:/path/to/lab2 python3 pxqu_wrel.py /path/to/pxqu_ref
# (pxqu_lab + its weight/imatrix fixtures are the internal calibration lab, not shipped in this
#  repo — point PXQ_LAB_DIRS at your own checkout; defaults to . and ./pxa-bench)
import sys, os, subprocess, tempfile, json
for _d in os.environ.get("PXQ_LAB_DIRS", ".:./pxa-bench").split(":"):
    if _d: sys.path.insert(0, _d)
import numpy as np
import pxqu_lab as lab   # reuses build_sets/load_imatrix/quant_slice/mk_book — the oracle itself

REF = sys.argv[1] if len(sys.argv) > 1 else "./pxqu_ref"
ORACLE = {2: 0.3020488067298746, 3: 0.14353147387217413}

def fp16_snap64(x): return np.asarray(np.float16(x), np.float64)

BOOKS = json.load(open(os.environ.get("PXQ_BOOKS_JSON", "books.json")))  # lab codebook file (sha256-pinned in ggml-pxq2-tables.h)
BOOK = {2: np.array(BOOKS["b2_e16"]["book"], np.float64),
        3: np.array(BOOKS["b3_e16"]["book"], np.float64)}
SUB16_SNAP = fp16_snap64(lab.SUB16)   # the C code's LUT (PXQ6_SUB16_INIT is fp16-snapped)

IM = lab.load_imatrix(lab.IMX_PROD)
_, ev = lab.build_sets(IM, map_gate=True)
assert len(ev) == 36, len(ev)

def c_wrel(type_):
    wrels = []
    with tempfile.TemporaryDirectory() as td:
        for tag, W, sig in ev:
            R, K = W.shape
            wf, sf, rf = f"{td}/w.f32", f"{td}/s.f32", f"{td}/r.f32"
            W.astype(np.float32).tofile(wf)
            if sig is not None: sig.astype(np.float32).tofile(sf)
            subprocess.run([REF, "roundtrip", str(type_), str(R), str(K), wf,
                            sf if sig is not None else "-", rf], check=True)
            rec = np.fromfile(rf, np.float32).reshape(R, K).astype(np.float64)
            W64 = W.astype(np.float64)
            w = np.tile(sig.astype(np.float64), R).reshape(R, K) if sig is not None else np.ones_like(W64)
            num = float((w * (W64 - rec) ** 2).sum())
            den = float((w * W64 ** 2).sum())
            wrels.append(np.sqrt(num / max(den, 1e-30)))
    return float(np.mean(wrels))

def np_wrel_snapped(type_):
    """the numpy oracle re-run with the SNAPPED SUB16 (matching the C tables exactly)."""
    book = lab.mk_book(BOOK[type_])
    _, wrel = lab.eval_set_metrics(ev, 16, SUB16_SNAP, book)
    return wrel

ok = True
for t in (2, 3):
    c = c_wrel(t)
    o_snap = np_wrel_snapped(t)
    o_froz = ORACLE[t]
    d_snap, d_froz = c - o_snap, c - o_froz
    p1, p2 = abs(d_snap) <= 1e-6, abs(d_froz) <= 1e-4
    print(f"PXQ{t}: C wrel={c:.9f}  snapped-oracle={o_snap:.9f} (d={d_snap:+.2e} {'PASS' if p1 else 'FAIL'})"
          f"  frozen-oracle={o_froz:.9f} (d={d_froz:+.2e} {'PASS' if p2 else 'FAIL'})")
    ok &= p1 and p2
print("Q-G2", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
