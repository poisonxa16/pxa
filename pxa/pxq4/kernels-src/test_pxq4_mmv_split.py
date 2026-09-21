#!/usr/bin/env python3
"""Gate: the K-chunk-split mmv (two-launch) AND the v4 single-launch fused mmv are both
bit-identical to the monolithic k_pxq4_mmv, executed on the REAL kernel source via the host
simulator.

Run by build_hostsim.sh after test_pxq4_kernel_ref.py. Needs ./libpxq4_hostsim.so.

Three things this checks that the pre-v4 version did not:
  * the fp32 `part` buffer is compared as uint32 between the split and fused paths. The final
    __float2half_rn masks essentially all fp32 fold changes -- an FMA-contraction-class change
    measured 0/128 fp16 words different but 27% of part[] different -- so comparing only fp16
    output is a weak probe. part[] costs nothing extra; it was already allocated.
  * every output buffer is re-poisoned before EVERY call. The old version allocated out_a/out_b
    once outside the `for vecx` loop, so on the second iteration both already held identical
    values and a kernel that wrote nothing passed.
  * WHAT THIS CANNOT DO: the simulator runs blocks strictly sequentially, so the fused path's
    arrival counter degenerates to ++ and its fences are no-ops. This gates the VALUES only and
    can NEVER observe the cross-block race. The device stress harness
    (src/device_gates/pxq4_v5_gpu.cu, `stress` mode) is mandatory, not optional, and is not
    replaced by a pass here.
"""
import ctypes, os, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpxq4_hostsim.so"))
lib.pxq4_hostsim_canon_nfix.restype = ctypes.c_int
rng = np.random.default_rng(7)
fails = 0
c = lambda a: a.ctypes.data_as(ctypes.c_void_p)
# (panels, kslabs, M): covers nfix 8 and 16, every real K class of the model at reduced panel
# count (the fold is per-panel, so panel count only multiplies cases), M across the mmv range.
for (panels, kslabs, M) in [(2, 48, 1), (2, 136, 1), (1, 160, 3), (2, 544, 2), (3, 40, 8)]:
    nfix = lib.pxq4_hostsim_canon_nfix(kslabs)
    slabs = rng.integers(0, 256, size=(panels, kslabs, 1088), dtype=np.uint8)
    anchor = (rng.standard_normal((panels, 64)) * 0.05).astype(np.float16)
    anchor[0, 0] = np.float16(0.0)          # exact +0 row
    anchor[0, 1] = np.float16(6e-8)         # fp16-subnormal row
    x = (rng.standard_normal((M, kslabs * 32)) * 0.1).astype(np.float16)
    for vecx in (1, 0):
        # poison every buffer before every call
        out_a = np.full((M, panels * 64), np.float16(np.nan))
        out_b = np.full((M, panels * 64), np.float16(-np.inf))
        out_c = np.full((M, panels * 64), np.float16(np.inf))
        part_b = np.full(M * panels * nfix * 256, np.float32(np.nan))
        part_c = np.full(M * panels * nfix * 256, np.float32(np.nan))
        lib.pxq4_hostsim_mmv_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                 c(out_a.view(np.uint16)), M, panels, kslabs, vecx)
        lib.pxq4_hostsim_mmv_split_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                       c(part_b), c(out_b.view(np.uint16)), M, panels, kslabs, vecx)
        lib.pxq4_hostsim_mmv_fused_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                       c(part_c), c(out_c.view(np.uint16)), M, panels, kslabs, vecx)
        assert not np.isnan(part_b).any(), "split left part[] unwritten"
        assert not np.isnan(part_c).any(), "fused left part[] unwritten"
        eq_s = np.array_equal(out_a.view(np.uint16), out_b.view(np.uint16))
        eq_f = np.array_equal(out_a.view(np.uint16), out_c.view(np.uint16))
        eq_p = np.array_equal(part_b.view(np.uint32), part_c.view(np.uint32))
        print(f"panels={panels} kslabs={kslabs} M={M} vecx={vecx} nfix={nfix} "
              f"split={eq_s} fused={eq_f} part_fp32={eq_p}")
        if not (eq_s and eq_f and eq_p):
            fails += 1
print("mmv split parity:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
