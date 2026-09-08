"""
GATE (b) / G8 -- single-linear-layer parity:  y = x W^T.

Three ways to compute the same linear, and the harness pins the relationship between all
three so a regression can be attributed:

  MMV      torch.ops.pxq4.mmv_out          -- decode path, M <= PXQ4_MMV_MAX_M (plan §6.6)
  DEQ+MM   dequant_out then torch.mm       -- prefill path, and the fallback for large M
  EXACT    float64 GEMM on the fp32 dequant -- the arbiter neither path can beat

The CPU leg needs no GPU: `oracle.mmv` is a bit-exact model of k_pxq6_mmv's fp32
accumulation ORDER (pxq6.cuh:914-971), not just of its value.  That distinction is what
makes this gate diagnostic instead of decorative -- if the CUDA mmv disagrees with the
model by more than the FMA-contraction ambiguity, the bug is in the port, not in float
arithmetic, and the model tells you so.

Two facts drive the tolerances, and both are documented rather than fudged:

1. k_pxq6_mmv consumes fp32 activations and emits fp32 (pxq6.cuh:920-923, :968-969).
   vLLM hands us fp16.  The plan stages fp16->fp32->fp16 around the call (§7.3).  So the
   INPUT is exactly representable and the only lossy step is the final fp16 store.

2. DEQ+MM is NOT expected to match MMV bit-for-bit and never will be: torch.mm on a V100
   runs HMMA with fp16 multiplicands and fp32 accumulate over a completely different
   reduction tree.  Their agreement is a tolerance question; MMV vs the CPU model is an
   exactness question.  Conflating the two is how a real bug gets waved through as
   "float noise".
"""

from __future__ import annotations

import numpy as np

from . import adapters, compare, fixtures
from . import oracle as O
from .test_a_dequant import Skip

# M values to exercise.  PXQ4_MMV_MAX_M defaults to 8 (plan §6.7, matching
# PXA_PXQ4_2D_MAX_NY at ggml-cuda.cu:4021), so 1/2/8 are the mmv path and 16 is the
# crossover the plan flags as risk 4.
MMV_M = (1, 2, 8)
BIG_M = (16, 64)

# Shapes small enough to model in numpy in seconds while keeping every structural
# property that matters (see fixtures.SMALL_SHAPES).
_B_SHAPES = ("narrowK", "wideK", "oddpanels", "shardable4")


def _cases(real=None, profile="realistic"):
    for label in _B_SHAPES:
        N, K = fixtures.SMALL_SHAPES[label]
        slabs, anchor = fixtures.synth_parts(N, K, seed=0xB0 ^ abs(hash(label)) % 9973,
                                            profile=profile)
        yield (f"synth:{label}", N, K, slabs, anchor)
    if real:
        for label, (N, K, slabs, anchor) in real.items():
            if label.startswith("__"):
                continue
            yield (f"real:{label}", N, K, slabs, anchor)


# ---------------------------------------------------------------------------------------
# CPU: characterise the mmv fold against an exact GEMM.  No GPU needed.
# ---------------------------------------------------------------------------------------
def test_b_cpu_mmv_model_vs_exact(real=None, report=None):
    """The mmv fold is a reordered fp32 sum, so it differs from an exact GEMM on the SAME
    dequantized weights.  This records by how much, which is the number every later
    tolerance is derived from -- rather than a tolerance picked to make the test pass."""
    for label, N, K, slabs, anchor in _cases(real):
        w = O.dequant(slabs, anchor)                       # fp32, the parity-locked values
        x16 = fixtures.synth_activations(2, K, seed=3, scale="normalized")
        x32 = x16.astype(np.float32)                       # exact: the staging conversion
        got = O.mmv(x32, slabs, anchor)                    # kernel-order fp32 fold
        exact = x32.astype(np.float64) @ w.astype(np.float64).T
        st = compare.err_stats(got, exact)
        if report is not None:
            report.append((f"mmv-fold {label} N={N} K={K}", st))
        # A reordered fp32 sum of K terms should stay far inside fp32's own resolution.
        # 2^-20 of the row absmax is ~4000x looser than one fp32 ULP relative and ~500x
        # tighter than one fp16 ULP, so it catches a wrong fold without tripping on noise.
        assert st["max_abs_over_absmax"] < 2 ** -20, (
            f"[{label}] mmv fold drifts from exact by {st['max_abs_over_absmax']:.3e} "
            f"of absmax -- that is too much for a reassociation of {K} fp32 terms; "
            f"suspect the eff application point or the nfix chunking, not rounding")


def test_b_cpu_fold_is_deterministic(real=None):
    """canon_nfix is `the largest power of two <= min(kslabs/4, 16)` (pxq6.cuh:826-834).
    Its whole purpose is that any pow2 split S <= nfix divides nfix, so the K-split
    kernels are bit-identical to the unsplit one.  Pin the property here: if someone
    changes PXQ6_CANON_CMAX or KSEG the model must be updated with it."""
    for kslabs, want in ((1, 1), (4, 1), (8, 2), (16, 4), (48, 8), (64, 16),
                         (160, 16), (544, 16), (136, 16)):
        got = O.canon_nfix(kslabs)
        assert got == want, f"canon_nfix({kslabs}) = {got}, expected {want}"
        assert want & (want - 1) == 0, "nfix must be a power of two"
    for kslabs in (8, 16, 48, 64, 160, 544):
        n = O.canon_nfix(kslabs)
        for s in (1, 2, 4, 8, 16):
            if s <= n:
                assert n % s == 0, f"S={s} does not divide nfix={n} at kslabs={kslabs}"


def test_b_cpu_fp16_output_headroom(real=None, report=None):
    """The op returns fp16 (plan §7.1).  Check the fp32->fp16 store, not the arithmetic,
    is what dominates the error -- if it is not, something upstream is wrong."""
    for label, N, K, slabs, anchor in _cases(real):
        w = O.dequant(slabs, anchor)
        x16 = fixtures.synth_activations(2, K, seed=11, scale="normalized")
        x32 = x16.astype(np.float32)
        exact = x32.astype(np.float64) @ w.astype(np.float64).T
        fold = O.mmv(x32, slabs, anchor)
        stored = fold.astype(np.float16).astype(np.float64)
        e_fold = compare.err_stats(fold, exact)["max_abs"]
        e_store = compare.err_stats(stored, exact)["max_abs"]
        if report is not None:
            report.append((f"fp16-store {label}", {"fold_err": e_fold, "stored_err": e_store,
                                                   "ratio": e_store / max(e_fold, 1e-30)}))
        assert e_store >= e_fold * 0.5, (
            f"[{label}] the fp16 store is not the dominant error term "
            f"(fold {e_fold:.3e} vs stored {e_store:.3e}) -- if the fp32 fold has become "
            f"comparable to a whole fp16 ULP, the accumulation is drifting")
        assert not np.isinf(fold).any(), f"[{label}] mmv overflowed fp32"
        # The op's output dtype is fp16 (plan §7.1).  With realistic per-row anchors and
        # normalized activations the product must sit far inside fp16 range; if it does
        # not, ffn_down (K=17408, the widest reduction in the model) is the tensor that
        # would silently produce inf in production, so this is checked rather than assumed.
        assert np.abs(fold).max() < 65504.0 / 8, (
            f"[{label}] |y|max={np.abs(fold).max():.1f} leaves less than 3 bits of "
            f"headroom below the fp16 max of 65504 -- an fp16 output for this module "
            f"is not safe")


# ---------------------------------------------------------------------------------------
# GPU: MMV and DEQ+MM against the CPU model.
# ---------------------------------------------------------------------------------------
def _to_cuda(torch, slabs, anchor):
    dev = torch.device("cuda")
    return (torch.from_numpy(np.ascontiguousarray(slabs)).to(dev),
            torch.from_numpy(np.ascontiguousarray(anchor)).to(dev))


def test_g8_cuda_mmv_vs_model(real=None, report=None):
    """MMV vs the bit-exact CPU model.

    Expected outcome: bit-exact for exactly one of the nine (acc_variant, tail_variant)
    combinations -- see oracle.ACC_VARIANTS.  Which one depends on how nvcc contracted
    `acc + (a0*x0 + a1*x1)` (pxq6.cuh:603-609) under the default -fmad=true, and that is
    a codegen fact we cannot read out of the source.  The test SEARCHES and REPORTS the
    match rather than asserting a guess; failure to match ANY variant is the real signal,
    because no contraction choice can rescue a wrong layout.
    """
    ops = adapters.pxq4_ops()
    if not ops:
        raise Skip(str(ops))
    if not adapters.cuda_available():
        raise Skip("no CUDA device")
    torch = adapters.torch_module()

    for label, N, K, slabs, anchor in _cases(real):
        ts, ta = _to_cuda(torch, slabs, anchor)
        for M in MMV_M:
            x16 = fixtures.synth_activations(M, K, seed=M, scale="normalized")
            tx = torch.from_numpy(x16).cuda()
            out = torch.empty((M, N), dtype=torch.float16, device="cuda")
            ops.mmv_out(out, tx, ts, ta)
            torch.cuda.synchronize()
            got = out.cpu().numpy()

            hits, tried = O.match_mmv_variants(got, x16, slabs, anchor, first_only=True)
            matched = hits[0] if hits else None
            if report is not None:
                report.append((f"mmv-variant {label} M={M}",
                               {"matched": matched, "variants_tried": tried}))
            if matched is None:
                base = O.mmv(x16.astype(np.float32), slabs, anchor)
                st = compare.err_stats(got.astype(np.float64), base.astype(np.float64))
                raise AssertionError(
                    f"[{label}] M={M}: torch.ops.pxq4.mmv_out matches NONE of the 9 "
                    f"FMA-contraction variants of the CPU model. This is not float noise. "
                    f"error vs the unfused model: {st}\n"
                    + compare.bit_diff_report(base.astype(np.float16), got, "model", "cuda"))


def test_g8_cuda_mmv_vs_deqmm(real=None, report=None):
    """MMV vs dequant+torch.mm -- the two paths plan §6.6 dispatches between at
    M == PXQ4_MMV_MAX_M.  They must agree to fp16 tolerance, because a request that
    straddles the crossover would otherwise produce a visible discontinuity in output."""
    ops = adapters.pxq4_ops()
    if not ops:
        raise Skip(str(ops))
    if not adapters.cuda_available():
        raise Skip("no CUDA device")
    torch = adapters.torch_module()

    for label, N, K, slabs, anchor in _cases(real):
        ts, ta = _to_cuda(torch, slabs, anchor)
        w = torch.empty((N, K), dtype=torch.float16, device="cuda")
        ops.dequant_out(w, ts, ta)
        w_np = O.dequant(slabs, anchor)
        for M in MMV_M:
            x16 = fixtures.synth_activations(M, K, seed=100 + M, scale="normalized")
            tx = torch.from_numpy(x16).cuda()
            out_mmv = torch.empty((M, N), dtype=torch.float16, device="cuda")
            ops.mmv_out(out_mmv, tx, ts, ta)
            out_mm = torch.mm(tx, w.t())
            torch.cuda.synchronize()

            exact = x16.astype(np.float64) @ w_np.astype(np.float64).T
            s_mmv = compare.err_stats(out_mmv.cpu().numpy(), exact)
            s_mm = compare.err_stats(out_mm.cpu().numpy(), exact)
            if report is not None:
                report.append((f"mmv-vs-mm {label} M={M}", {"mmv": s_mmv, "mm": s_mm}))
            # torch.mm on sm_70 is HMMA: fp16 multiplicands, fp32 accumulate.  Its error
            # is dominated by the fp16 rounding of W, which is ~2^-11 relative per term
            # and partially cancels over K.  2^-8 of absmax is generous for both and
            # still ~100x tighter than a genuine layout error would produce.
            assert s_mmv["max_abs_over_absmax"] < 2 ** -8, f"[{label}] M={M} mmv: {s_mmv}"
            assert s_mm["max_abs_over_absmax"] < 2 ** -8, f"[{label}] M={M} deq+mm: {s_mm}"


def test_g8_cuda_apply_shape_contract(real=None):
    """Reproduce plan §6.6's `apply` body exactly and check the shapes it depends on.

    The reshape/`out.reshape(*x.shape[:-1], N)` dance is where a 3-D activation tensor
    (batch, seq, hidden) silently becomes a 2-D one; catching it here costs nothing.
    """
    ops = adapters.pxq4_ops()
    if not ops:
        raise Skip(str(ops))
    if not adapters.cuda_available():
        raise Skip("no CUDA device")
    torch = adapters.torch_module()

    N, K = fixtures.SMALL_SHAPES["shardable4"]
    slabs, anchor = fixtures.synth_parts(N, K, seed=42, profile="realistic")
    ts, ta = _to_cuda(torch, slabs, anchor)

    for shape in ((4, K), (2, 3, K), (1, 1, K)):
        x = torch.from_numpy(
            fixtures.synth_activations(int(np.prod(shape[:-1])), K, seed=5,
                                       scale="normalized")
        ).reshape(*shape).cuda()
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        M = x2.shape[0]
        out = torch.empty((M, N), dtype=torch.float16, device="cuda")
        if M <= 8:
            ops.mmv_out(out, x2, ts, ta)
        else:
            w = torch.empty((N, K), dtype=torch.float16, device="cuda")
            ops.dequant_out(w, ts, ta)
            torch.mm(x2, w.t(), out=out)
        torch.cuda.synchronize()
        y = out.reshape(*x.shape[:-1], N)
        assert y.shape == (*shape[:-1], N), f"apply() shape contract broken for {shape}"
        assert y.dtype == torch.float16
        assert torch.isfinite(y).all()
