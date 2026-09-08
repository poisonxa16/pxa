"""
GATES H1-H6 -- run agent C's ACTUAL kernel, on the CPU, with no GPU and no lease.

`pxq4_kernel_hostsim.cpp` compiles the real `k_pxq4_dequant_matrix` / `k_pxq4_mmv` device
code against a host shim for blockIdx/threadIdx/__syncthreads.  That moves the substance
of G6 and G8a -- layout addressing, table values, accumulation order, the fp16 store, and
the shard invariant, all inside the kernel's own source -- from "blocked on borrowed
hardware" to "runs in a second, here".

A hostsim PASS is NECESSARY, NOT SUFFICIENT.  Still owed to real sm_70 hardware:
  * the launch configuration and grid mapping (the shim supplies its own)
  * the dynamic-shared-memory opt-in for K >= 12288 (test_d_ops_abi G8h)
  * CUDA-graph capture (G8g)
  * any MODE that uses warp primitives (__shfl_sync / prmt) -- the host shim cannot
    execute those, so whichever MODE the sm_70 build selects must still be checked on
    device against this same oracle
  * nvcc's fp32 contraction choices, which differ from the host compiler's
"""

from __future__ import annotations

import numpy as np

from . import compare, fixtures, hostsim_bridge as HS
from . import oracle as O
from .test_a_dequant import Skip


def _need():
    if not HS.available():
        raise Skip("libpxq4_hostsim.so not built (run impl/build_hostsim.sh)")


# The host shim launches one OS THREAD per CUDA thread, serially over blocks
# (pxq4_kernel_hostsim.cpp `launch`), measured here at ~6.7 ms per 64-thread block.  A
# full real ffn_down is 4 panels x 544 slabs = 2176 blocks PER DEQUANT, and H4 does
# several -- minutes each, for information the small shapes already carry.
#
# Trimming is sound, and it is sound for the SAME REASON the port works: a panel
# subrange and a slab subrange are each themselves a valid PXQ4 tensor (that is exactly
# what G3a and G3b prove), so a trimmed real fixture is still real artifact bytes with
# the real anchors and the real code distribution -- just fewer of them.  The untrimmed
# real tensors stay covered bitwise by G1a/G3a/G3b through the numpy oracle, which G1a
# pins to the production C.
HOSTSIM_MAX_PANELS = 4       # 256 rows: still splits cleanly at TP=4 on the column axis
HOSTSIM_MAX_SLABS = 4        # K=128:    still splits cleanly at TP=4 on the row axis

# ...BUT THE SAME TRIM IS FATAL TO THE MMV GATES, so they do not use it.
#
# The dequant kernel has no fold: one block per slab, and a slab is decoded identically no
# matter how many of them there are.  4 slabs therefore costs nothing in coverage.
#
# The MMV kernel is the opposite.  Its whole structure is the canonical fold
# (pxq4_kernel.cuh:297-318): nfix = pxq4_canon_nfix(kslabs) chunks, chunk c spanning
# [b0, b1) = [(kslabs*c)/nfix, (kslabs*(c+1))/nfix), with agent C's EDIT 3 staging ONLY that
# chunk's activations into smem and re-basing the read as `pxq4_xs + (kb - b0)*PXQ4_QK`
# (pxq4_kernel.cuh:315).  At kslabs = 4, `lim = 4/PXQ4_MMV_KSEG = 1` so canon_nfix == 1: one
# chunk, b0 == 0, `kb - b0 == kb`.  Trimming the mmv cases to 4 slabs makes the entire fold
# loop degenerate and the re-basing a no-op, so EDIT 3 -- the one substantive deviation from
# the shipping engine, and the reason K = 17408 works without a capture-hostile cudaMalloc
# workspace -- is not tested at all.
#
# VERIFIED, not argued: with the old uniform 4-slab trim, mutating :315 to
# `pxq4_xs + (size_t)kb * PXQ4_QK` (i.e. deleting the re-basing) left H1-H7 at 7 passed,
# 0 failed.  With the shapes below the same mutant matches NONE of the nine fold variants.
#
# Cost is why the trim existed, so the mmv cases are shaped to buy nfix, not area: the host
# shim spawns 256 OS threads per block and the mmv grid is (N/64, M), so cost scales with
# PANELS x M and only weakly with kslabs.  Two panels and 64 slabs is ~0.8 s for M = 1,2,5.
MMV_SHAPES = {
    # label        (N,   K)      kslabs  nfix  chunk sizes
    "nfix1":       (128,  128),  #   4     1    [4]              degenerate arm kept on purpose
    "nfix4":       (128,  512),  #  16     4    [4,4,4,4]        first arm with b0 != 0
    "nfix4_ragged": (192, 576),  #  18     4    [4,5,4,5]        unequal boundaries, 3 panels
    "nfix8_ragged": (128, 1088), #  34     8    [4,4,4,5,4,4,4,5]
    "nfix16":      (128, 2048),  #  64    16    [4]*16           CANON_CMAX saturated
}

# Real fixtures in the mmv gates: enough slabs for a non-degenerate fold (16 -> nfix 4)
# without paying for a full K = 17408 tensor.  The synthetic shapes above carry the deep
# nfix coverage; the real ones are here for the real code/anchor distribution.
HOSTSIM_MMV_MAX_PANELS = 4
HOSTSIM_MMV_MAX_SLABS = 16


def _trim(N, K, slabs, anchor, max_panels=None, max_slabs=None):
    p = min(slabs.shape[0], HOSTSIM_MAX_PANELS if max_panels is None else max_panels)
    q = min(slabs.shape[1], HOSTSIM_MAX_SLABS if max_slabs is None else max_slabs)
    return (p * O.PANEL_ROWS, q * O.SLAB_COLS,
            np.ascontiguousarray(slabs[:p, :q, :]),
            np.ascontiguousarray(anchor[:p]))


def _cases(real=None, profile="extreme", max_panels=None, max_slabs=None):
    for label in ("narrowK", "wideK", "oddpanels", "shardable4"):
        N, K = fixtures.SMALL_SHAPES[label]
        slabs, anchor = fixtures.synth_parts(N, K, seed=0xF0 ^ (abs(hash(label)) % 6971),
                                             profile=profile)
        yield (f"synth:{label}",) + _trim(N, K, slabs, anchor, max_panels, max_slabs)
    if real:
        for label, v in real.items():
            if label.startswith("__"):
                continue
            yield (f"real:{label}",) + _trim(*v, max_panels=max_panels,
                                             max_slabs=max_slabs)


def _mmv_cases(real=None, profile="realistic"):
    """Cases for the gates that run `k_pxq4_mmv`.  NOT `_cases`: see MMV_SHAPES above."""
    for label, (N, K) in MMV_SHAPES.items():
        slabs, anchor = fixtures.synth_parts(N, K, seed=0x3D ^ (abs(hash(label)) % 6971),
                                             profile=profile)
        yield (f"synth:{label}", N, K, slabs, anchor)
    if real:
        for label, v in real.items():
            if label.startswith("__"):
                continue
            yield (f"real:{label}",) + _trim(*v, max_panels=HOSTSIM_MMV_MAX_PANELS,
                                             max_slabs=HOSTSIM_MMV_MAX_SLABS)


def _fold_shape(K):
    """(kslabs, nfix, ragged) for a K, from the MODEL's canon_nfix.  H7 pins the model's
    canon_nfix to the kernel's, so using the model here is safe."""
    kslabs = K // O.SLAB_COLS
    nfix = O.canon_nfix(kslabs)
    return kslabs, nfix, bool(kslabs % nfix)


def test_h1_hostsim_tables_match():
    """The kernel TU's compiled-in PXQ4_BOOK_INIT / PXQ4_SUB16_INIT vs
    ggml-pxq6-tables.h.  Catches a stale object file and a bad transcription in one
    check, and it is the cheapest gate in the whole harness."""
    _need()
    book, sub = HS.builtin_tables()
    assert compare.bitwise_equal(book, O.BOOK), compare.bit_diff_report(
        O.BOOK, book, "tables.h", "kernel TU")
    assert compare.bitwise_equal(sub, O.SUB), compare.bit_diff_report(
        O.SUB, sub, "tables.h", "kernel TU")


def test_h2_hostsim_dequant_f32_bitexact(real=None):
    """G6, minus the GPU.  fp32 dequant is the parity-locked contract, so this is a
    bitwise comparison against the oracle -- which G1a has already pinned to the
    production C."""
    _need()
    for label, N, K, slabs, anchor in _cases(real):
        got = HS.dequant_f32(slabs, anchor)
        want = O.dequant(slabs, anchor)
        assert compare.bitwise_equal(got, want), (
            f"[{label}] N={N} K={K} kernel dequant (hostsim) != oracle\n"
            + compare.bit_diff_report(want, got, "oracle", "kernel"))


def test_h3_hostsim_dequant_f16_rounding(real=None):
    """The op writes fp16 (plan §7.1).  Confirm it is exactly the fp32 result with one
    round-to-nearest-even, i.e. that the kernel is not accumulating in fp16 anywhere."""
    _need()
    for label, N, K, slabs, anchor in _cases(real):
        got = HS.dequant_f16(slabs, anchor)
        want = O.dequant(slabs, anchor).astype(np.float16)
        assert compare.bitwise_equal(got, want), (
            f"[{label}] fp16 dequant != oracle rounded once\n"
            + compare.bit_diff_report(want, got, "oracle_f16", "kernel_f16"))


def test_h4_hostsim_shard_invariant(real=None):
    """G3, executed through the KERNEL rather than through the oracle.

    This is the strongest single statement the harness can make without hardware:
    dequantizing a shard with agent C's own code gives bit-identical values to the
    corresponding slice of the unsharded result, on both axes, at TP 1/2/4.
    """
    _need()
    for label, N, K, slabs, anchor in _cases(real):
        full = HS.dequant_f32(slabs, anchor)
        for tp in (1, 2, 4):
            # Ranks 0 and tp-1 only.  Every rank is the same code path with a different
            # offset, and these two bracket it: rank 0 catches a base-pointer error,
            # rank tp-1 catches an end-of-tensor overrun.  The middle ranks add cost
            # (the host shim spawns an OS thread per CUDA thread) without adding a
            # distinct failure mode -- and G3a/G3b already check ALL ranks through the
            # numpy oracle, which G1a pins to the production C.
            ranks = sorted({0, tp - 1})
            if N % tp == 0 and (N // tp) % O.PANEL_ROWS == 0:
                rows = N // tp
                for r in ranks:
                    s, a = O.shard_column(slabs, anchor, r * rows, rows)
                    assert compare.bitwise_equal(HS.dequant_f32(s, a),
                                                 full[r * rows:(r + 1) * rows]), (
                        f"[{label}] TP={tp} rank={r}: kernel column shard != slice")
            if K % tp == 0 and (K // tp) % O.SLAB_COLS == 0:
                kk = K // tp
                for r in ranks:
                    s, a = O.shard_row(slabs, anchor, r * kk, kk)
                    assert compare.bitwise_equal(HS.dequant_f32(s, a),
                                                 full[:, r * kk:(r + 1) * kk]), (
                        f"[{label}] TP={tp} rank={r}: kernel K shard != column slice")


def test_h5_hostsim_mmv_matches_fold_model(real=None, report=None):
    """G8a, minus the GPU: the kernel's mmv against the bit-exact CPU fold model.

    HONEST LIMITATION, reported rather than hidden: at fp16 output the nine
    FMA-contraction variants of the model are usually indistinguishable, because the
    final round-to-nearest-even absorbs a sub-ULP fp32 difference.  So a match here
    confirms the fold STRUCTURE (nfix chunking, KSEG lane assignment, ascending
    reduction, eff applied once per 32-block) but does NOT pin the contraction.  The gate
    reports how many variants matched: 9 means the test could not discriminate, 1 means
    it did.  The fp32 dequant gate (H2) is what carries the bit-exactness weight.

    WHAT THE FOLD SHAPES BELOW DO AND DO NOT BUY, measured on this harness rather than
    asserted.  Mutants of pxq4_kernel.cuh, rebuilt and run through this gate:
      * :315 `pxq4_xs + (kb - b0)*PXQ4_QK` -> `+ kb*PXQ4_QK` (EDIT 3's re-basing deleted)
        -- CAUGHT, at every case with nfix > 1.  Was NOT caught before this gate stopped
        trimming its cases to 4 slabs; that is the hole these shapes exist to close.
      * :308 `xt[b0*PXQ4_QK + idx]` -> `xt[idx]` (staging reads the wrong chunk of x)
        -- CAUGHT.  Also invisible at nfix == 1, where b0 is always 0.
      * nfix pinned to 1 (EDIT 3 disabled, whole-K staging) -- CAUGHT.
      * :300-301 chunk bounds `(kslabs*c)/nfix` -> `c*(kslabs/nfix)` -- NOT CAUGHT, and
        deliberately not chased.  That mutant keeps every activation paired with its own
        weight and changes only which lane sums which slab, so it perturbs the fp32
        reduction order and nothing else; the fp16 store absorbs it in ~17 of 18
        seed/M combinations tried.  Making this gate catch it would mean picking the
        one lucky seed, i.e. overfitting to a single mutant.  It is a real deviation
        from the engine's byte-for-byte fold and it is owed to G8a on device, where an
        fp32 comparison is available -- listed here rather than hidden.
      * over-staging (a fixed `n` for every chunk) -- NOT CAUGHT, and correctly so: the
        surplus floats are staged but never read.  The real defect in that mutant is an
        out-of-bounds read of `x` on the last chunk, which is an ASAN/compute-sanitizer
        question, not a numerical one.
    """
    _need()
    # Synthetic shapes only: the exhaustive variant count below costs nine full model
    # evaluations, and on real ffn_down (K=17408) that is minutes for information the
    # small shapes already carry.  H2/H4 cover the real tensors bitwise.
    cases = [c for c in _mmv_cases(None)]

    # ANTI-REGRESSION, and the reason this gate is worth anything.  Every claim in the
    # docstring above about the fold STRUCTURE is vacuous unless the cases actually make
    # the fold loop iterate: at nfix == 1 there is one chunk, b0 == 0, and EDIT 3's smem
    # re-basing (pxq4_kernel.cuh:315) is the identity.  A cap change that silently trimmed
    # these shapes back to 4 slabs would restore that hole, so assert the shape budget
    # here rather than trusting a comment.
    folds = [_fold_shape(K) for _, _, K, _, _ in cases]
    assert any(nfix > 1 for _, nfix, _ in folds), (
        "H5 is vacuous: every case has canon_nfix == 1, so the kernel's chunk loop runs "
        "once with b0 == 0 and EDIT 3's activation re-basing is never exercised. "
        f"kslabs seen: {sorted({k for k, _, _ in folds})}")
    assert max(nfix for _, nfix, _ in folds) >= O.MMV_CANON_CMAX, (
        f"H5 never saturates CANON_CMAX={O.MMV_CANON_CMAX}: max nfix is "
        f"{max(nfix for _, nfix, _ in folds)}; keep a >= {O.MMV_CANON_CMAX * O.MMV_KSEG}"
        " slab case in MMV_SHAPES")
    assert any(ragged for _, _, ragged in folds), (
        "H5 has no case where kslabs % nfix != 0, so every chunk has the same length and "
        "the staged span `n = (b1-b0)*PXQ4_QK` never varies between chunks. Keep a ragged "
        "shape in MMV_SHAPES")

    for i, (label, N, K, slabs, anchor) in enumerate(cases):
        for M in (1, 2, 5):
            x = fixtures.synth_activations(M, K, seed=M * 7, scale="normalized")
            got = HS.mmv_f16(x, slabs, anchor, vecx=1)
            # Count all nine on ONE case so the report can say whether the test
            # discriminates; short-circuit on the rest so the gate stays fast.
            exhaustive = (i == 0 and M == 1)
            hits, tried = O.match_mmv_variants(got, x, slabs, anchor,
                                               first_only=not exhaustive)
            if report is not None and exhaustive:
                ks, nfix, ragged = _fold_shape(K)
                report.append((f"hostsim mmv {label} M={M}",
                               {"variants_matched": len(hits), "variants_tried": tried,
                                "discriminating": len(hits) == 1,
                                "kslabs": ks, "nfix": nfix, "ragged_chunks": ragged,
                                "note": "9 matched => fp16 rounding hides the FMA "
                                        "contraction; H2 carries the bit-exactness"}))
            if not hits:
                ks, nfix, ragged = _fold_shape(K)
                base = O.mmv(x.astype(np.float32), slabs, anchor).astype(np.float16)
                raise AssertionError(
                    f"[{label}] M={M} K={K} kslabs={ks} nfix={nfix} ragged={ragged}: "
                    f"the kernel's mmv matches NONE of the 9 fold variants. That is a "
                    f"structural disagreement, not rounding.\n"
                    + compare.bit_diff_report(base, got, "model", "kernel"))

    if report is not None:
        report.append(("hostsim mmv fold coverage",
                       {"kslabs_nfix": [(k, n) for k, n, _ in folds],
                        "max_nfix": max(n for _, n, _ in folds),
                        "ragged_cases": sum(1 for *_, r in folds if r)}))


def test_h6_hostsim_vecx_is_bit_identical(real=None):
    """`k_pxq4_mmv<VECX>`'s two template arms load x as float4 or scalar but accumulate
    into the same two accumulators in the same ascending b order, so they must be bit
    identical.  If they are not, one of them has the accumulator index wrong -- and only
    one of the two is ever instantiated on device, so a divergence here is a latent bug
    that a device-only test would never see.

    Uses the mmv case set (MMV_SHAPES), not the dequant trim: with nfix == 1 the two arms
    would only ever be compared on a single-chunk fold, and the accumulator-index bug this
    gate exists to catch lives inside the chunked b-loop."""
    _need()
    for label, N, K, slabs, anchor in _mmv_cases(real):
        x = fixtures.synth_activations(3, K, seed=13, scale="normalized")
        a = HS.mmv_f16(x, slabs, anchor, vecx=1)
        b = HS.mmv_f16(x, slabs, anchor, vecx=0)
        assert compare.bitwise_equal(a, b), (
            f"[{label}] VECX=1 and VECX=0 disagree\n"
            + compare.bit_diff_report(b, a, "VECX=0", "VECX=1"))


def test_h7_hostsim_canon_nfix_matches_model(real=None):
    """The kernel's own canon_nfix vs the model's, over every kslabs the real shapes
    produce at TP 1/2/4.  A mismatch would make the fold model silently wrong and every
    mmv comparison meaningless."""
    _need()
    ks = set()
    for K in (5120, 6144, 10240, 12288, 17408):
        for tp in (1, 2, 4):
            ks.add((K // tp) // 32)
    for k in sorted(ks | {1, 2, 4, 8, 16, 32, 64, 128}):
        assert HS.canon_nfix(k) == O.canon_nfix(k), (
            f"kslabs={k}: kernel nfix {HS.canon_nfix(k)} != model {O.canon_nfix(k)}")
