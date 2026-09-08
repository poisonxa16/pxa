#!/usr/bin/env python3
"""Differential bit-exactness gate for the v4 fused mmv: VARIANT's fused path vs BASE's
split AND mono paths, comparing fp32 part[] as uint32 and fp16 out as uint16.

  usage: pxq4_bitexact_diff_v4.py BASE.so VARIANT.so
"""
import ctypes, sys
import numpy as np

base, var = ctypes.CDLL(sys.argv[1]), ctypes.CDLL(sys.argv[2])
c = lambda a: a.ctypes.data_as(ctypes.c_void_p)
for L in (base, var):
    L.pxq4_hostsim_canon_nfix.restype = ctypes.c_int

CASES = [(2, 8, 1), (2, 16, 2), (2, 48, 1), (3, 48, 8), (2, 136, 1), (2, 160, 1),
         (1, 160, 3), (2, 192, 2), (2, 544, 1), (5, 160, 4), (2, 40, 8)]
fails = 0
for (panels, kslabs, M) in CASES:
    nfix = base.pxq4_hostsim_canon_nfix(kslabs)
    if nfix < 2:
        continue
    rng = np.random.default_rng(2000 + kslabs * 31 + M)
    slabs = rng.integers(0, 256, size=(panels, kslabs, 1088), dtype=np.uint8)
    anchor = (rng.standard_normal((panels, 64)) * 0.05).astype(np.float16)
    anchor[0, 0] = np.float16(0.0)
    anchor[0, 1] = np.float16(6e-8)
    x = (rng.standard_normal((M, kslabs * 32)) * 0.3).astype(np.float16)
    for vecx in (1, 0):
        o_bm = np.full((M, panels * 64), np.float16(np.nan))
        o_bs = np.full((M, panels * 64), np.float16(-np.inf))
        p_b = np.full(M * panels * nfix * 256, np.float32(np.nan))
        base.pxq4_hostsim_mmv_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                  c(o_bm.view(np.uint16)), M, panels, kslabs, vecx)
        base.pxq4_hostsim_mmv_split_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                        c(p_b), c(o_bs.view(np.uint16)), M, panels, kslabs, vecx)
        o_vf = np.full((M, panels * 64), np.float16(np.inf))
        p_v = np.full(M * panels * nfix * 256, np.float32(np.nan))
        var.pxq4_hostsim_mmv_fused_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                       c(p_v), c(o_vf.view(np.uint16)), M, panels, kslabs, vecx)
        assert not np.isnan(p_b).any() and not np.isnan(p_v).any(), "part[] left unwritten"
        chk = [("part_fp32 fused vs base-split", p_b.view(np.uint32), p_v.view(np.uint32)),
               ("out_fp16 fused vs base-split", o_bs.view(np.uint16), o_vf.view(np.uint16)),
               ("out_fp16 fused vs base-mono",  o_bm.view(np.uint16), o_vf.view(np.uint16))]
        bad = [(n, int((a != b).sum()), a.size) for n, a, b in chk if not np.array_equal(a, b)]
        tag = f"panels={panels} kslabs={kslabs} K={kslabs*32} M={M} vecx={vecx} nfix={nfix}"
        if bad:
            fails += 1
            print(f"FAIL {tag}: " + ", ".join(f"{n} {d}/{t} differ" for n, d, t in bad))
        else:
            print(f"ok   {tag}  (part {p_v.size} fp32 + out {o_vf.size} fp16 identical)")
print("\nFUSED BIT-EXACT vs v3" if not fails else f"\nNOT BIT-EXACT ({fails} cases)")
sys.exit(1 if fails else 0)
