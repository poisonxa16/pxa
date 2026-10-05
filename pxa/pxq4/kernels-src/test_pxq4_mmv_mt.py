#!/usr/bin/env python3
"""Gate: the v6 multi-token fused mmv is per-token bit-identical to the monolithic
k_pxq4_mmv, on the REAL kernel source via the host simulator, for every dispatchable M (1..8),
both vecx arms, nfix 8 and 16, and part[] compared as uint32 against the two-launch split.

Run by build_hostsim.sh after test_pxq4_mmv_split.py. Needs ./libpxq4_hostsim.so.
Same caveat as the split gate: values only; the device stress harness owns the race.
"""
import ctypes, os, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpxq4_hostsim.so"))
lib.pxq4_hostsim_canon_nfix.restype = ctypes.c_int
rng = np.random.default_rng(11)
fails = 0
c = lambda a: a.ctypes.data_as(ctypes.c_void_p)
cases = []
# every M 1..8 on a small nfix-16 shape, plus nfix-8 and the widest-K class at selected Ms
for M in range(1, 9):
    cases.append((2, 160, M))
cases += [(2, 48, 2), (2, 48, 5), (2, 48, 8), (1, 544, 2), (1, 544, 8), (3, 136, 7)]
for (panels, kslabs, M) in cases:
    nfix = lib.pxq4_hostsim_canon_nfix(kslabs)
    slabs = rng.integers(0, 256, size=(panels, kslabs, 1088), dtype=np.uint8)
    anchor = (rng.standard_normal((panels, 64)) * 0.05).astype(np.float16)
    anchor[0, 0] = np.float16(0.0)
    anchor[0, 1] = np.float16(6e-8)
    x = (rng.standard_normal((M, kslabs * 32)) * 0.1).astype(np.float16)
    for vecx in (1, 0):
        out_a = np.full((M, panels * 64), np.float16(np.nan))
        out_b = np.full((M, panels * 64), np.float16(-np.inf))
        out_m = np.full((M, panels * 64), np.float16(np.inf))
        part_b = np.full(M * panels * nfix * 256, np.float32(np.nan))
        part_m = np.full(M * panels * nfix * 256, np.float32(np.nan))
        lib.pxq4_hostsim_mmv_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                 c(out_a.view(np.uint16)), M, panels, kslabs, vecx)
        lib.pxq4_hostsim_mmv_split_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                       c(part_b), c(out_b.view(np.uint16)), M, panels, kslabs, vecx)
        lib.pxq4_hostsim_mmv_fused_mt_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                          c(part_m), c(out_m.view(np.uint16)), M, panels, kslabs, vecx)
        assert not np.isnan(part_m).any(), "mt left part[] unwritten"
        eq_o = np.array_equal(out_a.view(np.uint16), out_m.view(np.uint16))
        eq_p = np.array_equal(part_b.view(np.uint32), part_m.view(np.uint32))
        print(f"panels={panels} kslabs={kslabs} M={M} vecx={vecx} nfix={nfix} "
              f"mt_out={eq_o} mt_part_fp32={eq_p}")
        if not (eq_o and eq_p):
            fails += 1
print("mmv mt parity:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
