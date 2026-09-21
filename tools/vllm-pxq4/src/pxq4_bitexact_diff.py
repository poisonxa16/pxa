#!/usr/bin/env python3
"""Differential bit-exactness gate: variant hostsim .so vs baseline hostsim .so.

Compares BOTH the fp32 partials (`part`, ~400x more sensitive) and the fp16 output,
across every real nfix/vecx/M/shape class, with poisoned buffers each iteration.

  usage: pxq4_bitexact_diff.py BASE.so VARIANT.so
"""
import ctypes, sys
import numpy as np

base, var = ctypes.CDLL(sys.argv[1]), ctypes.CDLL(sys.argv[2])
c = lambda a: a.ctypes.data_as(ctypes.c_void_p)
for L in (base, var):
    L.pxq4_hostsim_canon_nfix.restype = ctypes.c_int

# (panels, kslabs, M) — kslabs 48/136/160/192/544 are the model's real TP1/2/4 K classes;
# 4/8/16 exercise nfix 1/2/4 which production never hits but a variant might.
CASES = [(2, 4, 1), (2, 8, 1), (2, 16, 2), (2, 48, 1), (3, 48, 8), (2, 136, 1),
         (2, 160, 1), (1, 160, 3), (2, 192, 2), (2, 544, 1), (5, 160, 4)]
fails = 0
for (panels, kslabs, M) in CASES:
    nfix = base.pxq4_hostsim_canon_nfix(kslabs)
    assert nfix == var.pxq4_hostsim_canon_nfix(kslabs), "nfix disagrees"
    rng = np.random.default_rng(1000 + kslabs * 31 + M)
    slabs = rng.integers(0, 256, size=(panels, kslabs, 1088), dtype=np.uint8)
    # signed anchors, an exact +0 row, and an fp16-subnormal row
    anchor = (rng.standard_normal((panels, 64)) * 0.05).astype(np.float16)
    anchor[0, 0] = np.float16(0.0)
    anchor[0, 1] = np.float16(6e-8)
    x = (rng.standard_normal((M, kslabs * 32)) * 0.3).astype(np.float16)
    for vecx in (1, 0):
        got = {}
        for name, L in (("base", base), ("var", var)):
            # poison every buffer every time: a kernel that writes nothing must not pass
            o_m = np.full((M, panels * 64), np.float16(np.nan))
            o_s = np.full((M, panels * 64), np.float16(-np.inf))
            part = np.full(M * panels * nfix * 256, np.float32(np.nan))
            L.pxq4_hostsim_mmv_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                   c(o_m.view(np.uint16)), M, panels, kslabs, vecx)
            L.pxq4_hostsim_mmv_split_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                         c(part), c(o_s.view(np.uint16)), M, panels, kslabs, vecx)
            got[name] = (o_m.copy(), o_s.copy(), part.copy())
        (am, as_, ap), (bm, bs, bp) = got["base"], got["var"]
        chk = [
            ("part_fp32", ap.view(np.uint32), bp.view(np.uint32)),   # sensitive probe
            ("mono_fp16", am.view(np.uint16), bm.view(np.uint16)),
            ("split_fp16", as_.view(np.uint16), bs.view(np.uint16)),
            ("mono==split(var)", bm.view(np.uint16), bs.view(np.uint16)),
        ]
        bad = [(n, int((a != b).sum()), a.size) for n, a, b in chk if not np.array_equal(a, b)]
        assert not np.isnan(ap).any(), "baseline left part[] unwritten"
        tag = f"panels={panels} kslabs={kslabs} K={kslabs*32} M={M} vecx={vecx} nfix={nfix}"
        if bad:
            fails += 1
            print(f"FAIL {tag}: " + ", ".join(f"{n} {d}/{t} differ" for n, d, t in bad))
        else:
            print(f"ok   {tag}  (part {ap.size} fp32 + out {am.size} fp16 identical)")
print("\nBIT-EXACT" if not fails else f"\nNOT BIT-EXACT ({fails} cases)")
sys.exit(1 if fails else 0)
