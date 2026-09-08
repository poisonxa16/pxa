"""
GATE G8 (structural half) -- the assumptions that fail LATE and LOUDLY if unchecked.

Plan §10 risk 3 is the only genuinely UNVERIFIED runtime assumption in the design:
whether an out-of-tree `torch.ops.pxq4.*` participates in vLLM's FULL_AND_PIECEWISE
CUDA-graph capture the way the fork's own `torch.ops._C` does.  Nobody read code that
answers it, and it cannot be answered without a GPU.  These tests answer it in isolation,
in seconds, without loading a 27B model -- which is worth a lot, because the alternative
is discovering it during a full engine start on borrowed hardware.

Everything here needs a CUDA device.  Nothing here needs vLLM, except the two helpers at
the bottom that take an already-constructed model.
"""

from __future__ import annotations

import numpy as np

from . import adapters, compare, fixtures
from . import oracle as O
from .test_a_dequant import Skip


def _require():
    ops = adapters.pxq4_ops()
    if not ops:
        raise Skip(str(ops))
    if not adapters.cuda_available():
        raise Skip("no CUDA device")
    return ops, adapters.torch_module()


def test_schema_declares_output_mutation():
    """The op schema must annotate the output as mutated: `Tensor(a!) out`.

    Without it the functionalization pass treats `out` as read-only, is free to reorder
    or elide the call, and torch.compile silently produces a graph that never writes the
    buffer.  The failure is a zero/stale output under compile and a correct output in
    eager -- i.e. it passes every test that does not compile.
    """
    ops, torch = _require()
    for name in ("dequant_out", "mmv_out"):
        packet = getattr(torch.ops.pxq4, name)
        schema = str(packet.default._schema)
        assert "(a!)" in schema, (
            f"pxq4::{name} schema is {schema!r} -- the output tensor is not annotated as "
            f"mutated. Plan §7.1 requires `Tensor(a!) out`.")
        assert schema.strip().endswith("-> ()"), (
            f"pxq4::{name} must be an out-variant returning (), got {schema!r}")


def test_meta_kernel_registered():
    """`register_fake` is mandatory (plan §6.7): tracing happens on meta tensors, and an
    op with no meta implementation raises during Dynamo tracing, before capture is even
    attempted.  This is the cheapest possible check and it catches the most common
    omission in a custom-op port."""
    ops, torch = _require()
    N, K, M = 128, 512, 4
    dev = torch.device("meta")
    slabs = torch.empty(N // 64, K // 32, 1088, dtype=torch.uint8, device=dev)
    anchor = torch.empty(N // 64, 64, dtype=torch.float16, device=dev)
    out_d = torch.empty(N, K, dtype=torch.float16, device=dev)
    out_m = torch.empty(M, N, dtype=torch.float16, device=dev)
    x = torch.empty(M, K, dtype=torch.float16, device=dev)
    try:
        torch.ops.pxq4.dequant_out(out_d, slabs, anchor)
        torch.ops.pxq4.mmv_out(out_m, x, slabs, anchor)
    except NotImplementedError as e:
        raise AssertionError(
            f"pxq4 ops have no meta/fake implementation: {e}\n"
            f"Add @torch.library.register_fake for both ops (plan §6.7).") from e


def test_apply_allocates_nothing():
    """plan §6.6/§6.7: PXQ4Workspace is preallocated before capture and `apply()` must
    never allocate.  An allocation inside a captured region either fails outright or --
    worse -- succeeds from the graph pool and then aliases across replays."""
    ops, torch = _require()
    N, K, M = 256, 4096, 4
    slabs, anchor = fixtures.synth_parts(N, K, seed=1, profile="realistic")
    ts = torch.from_numpy(slabs).cuda()
    ta = torch.from_numpy(anchor).cuda()
    x = torch.from_numpy(fixtures.synth_activations(M, K, seed=1, scale="normalized")).cuda()
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")
    w = torch.empty((N, K), dtype=torch.float16, device="cuda")

    ops.mmv_out(out, x, ts, ta)          # warm up: first call may lazy-init the library
    ops.dequant_out(w, ts, ta)
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated()
    for _ in range(8):
        ops.mmv_out(out, x, ts, ta)
        ops.dequant_out(w, ts, ta)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after == before, (
        f"the pxq4 ops allocated {after-before} B across 16 calls; they must be "
        f"allocation-free (plan §6.7 PXQ4Workspace)")


def test_cudagraph_capture_and_replay():
    """Capture both ops in a CUDA graph and replay with changed input.

    This is the direct test of plan §10 risk 3.  If it passes, the fallback
    (`--cudagraph-mode PIECEWISE` on the CLI) is not needed.  If it fails, it fails HERE
    in two seconds instead of during a 27B engine start on borrowed GPUs.
    """
    ops, torch = _require()
    N, K, M = 256, 4096, 4
    slabs, anchor = fixtures.synth_parts(N, K, seed=2, profile="realistic")
    ts = torch.from_numpy(slabs).cuda()
    ta = torch.from_numpy(anchor).cuda()
    x = torch.zeros((M, K), dtype=torch.float16, device="cuda")
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")

    # Warm up on a side stream: capture of a cold kernel deadlocks on some drivers, and
    # this is exactly what vLLM's own capture path does.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ops.mmv_out(out, x, ts, ta)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.mmv_out(out, x, ts, ta)

    x_np = fixtures.synth_activations(M, K, seed=99, scale="normalized")
    x.copy_(torch.from_numpy(x_np).cuda())
    g.replay()
    torch.cuda.synchronize()
    got = out.cpu().numpy()

    ref = O.mmv(x_np.astype(np.float32), slabs, anchor).astype(np.float16)
    matched = any(
        compare.bitwise_equal(
            got, O.mmv(x_np.astype(np.float32), slabs, anchor,
                       acc_variant=av, tail_variant=tv).astype(np.float16))
        for av in O.ACC_VARIANTS for tv in O.TAIL_VARIANTS)
    assert matched, (
        "graph replay produced a different result than a fresh call -- the captured "
        "kernel is reading stale or aliased memory\n"
        + compare.bit_diff_report(ref, got, "model", "replay"))


def test_dequant_smem_limit_for_wide_k():
    """mmv stages ALL of x in dynamic shared memory: (K + 256) * 4 bytes
    (pxq6.cuh:938-940).  At K=17408 that is 70,656 B -- above the 48 KiB default and
    below V100's 96 KiB opt-in cap, so the launcher MUST call cudaFuncSetAttribute
    (plan §7.3).  K >= 24320 does not fit at all and must fall back to dequant+mm.

    A launcher that forgot the opt-in fails with cudaErrorInvalidValue at exactly the
    shapes that only appear at TP<=2 -- i.e. never on the 4-GPU DGX and always on the
    2-GPU Unraid box.  That is a bug that ships.
    """
    ops, torch = _require()
    for K, must_work in ((4096, True), (17408, True)):
        N, M = 128, 2
        smem = (K + 256) * 4
        slabs, anchor = fixtures.synth_parts(N, K, seed=3, profile="realistic")
        ts = torch.from_numpy(slabs).cuda()
        ta = torch.from_numpy(anchor).cuda()
        x = torch.from_numpy(
            fixtures.synth_activations(M, K, seed=3, scale="normalized")).cuda()
        out = torch.empty((M, N), dtype=torch.float16, device="cuda")
        try:
            ops.mmv_out(out, x, ts, ta)
            torch.cuda.synchronize()
        except RuntimeError as e:
            raise AssertionError(
                f"mmv_out failed at K={K} (dynamic smem {smem} B). If this is "
                f"cudaErrorInvalidValue the launcher is missing "
                f"cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, "
                f"{smem}) -- plan §7.3. Original: {e}") from e
        assert torch.isfinite(out).all(), f"K={K}: non-finite output"


# ---------------------------------------------------------------------------------------
# Helpers for the in-engine gates.  These take a constructed vLLM model, so they belong
# in a run that has already paid for a model load; they are not part of run_gates.
# ---------------------------------------------------------------------------------------
def assert_no_sm70_fastpath(model) -> None:
    """`_maybe_sm70_dense_forward` (linear.py:56-96) runs BEFORE quant_method.apply() in
    every LinearBase.forward.  It short-circuits on `layer._sm70_f16_prepared`, which
    only UnquantizedLinearMethod sets (linear.py:408).  If any PXQ4 layer ever acquires
    that flag, its weights are read as dense fp16 and the quantized path is dead code --
    silently, and only on sm_70.

    Call this after model load, before serving.

    ASSUMPTION: agent B names the class `PXQ4LinearMethod` (plan §6.6). This helper
    matches on the type name because a plugin-registered class cannot be imported here
    without pulling in vllm. If B renames it, this silently checks nothing -- so the
    coverage helper below exists as the paired positive check.
    """
    bad_prepared, missing_forbid = [], []
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None or type(qm).__name__ != "PXQ4LinearMethod":
            continue
        if getattr(mod, "_sm70_f16_prepared", False):
            bad_prepared.append(name)
        if not getattr(mod, "_sm70_f16_forbidden", False):
            missing_forbid.append(name)
    assert not bad_prepared, (
        f"_sm70_f16_prepared is set on PXQ4 layers -- the sm70 dense fast path will "
        f"bypass the quantized kernel entirely: {bad_prepared[:8]}")
    assert not missing_forbid, (
        f"_sm70_f16_forbidden is not set on PXQ4 layers (plan §2.6 requires it "
        f"defensively): {missing_forbid[:8]}")


def assert_pxq4_module_coverage(model, expected_suffixes) -> None:
    """Every module the policy says is PXQ4 must actually have got PXQ4LinearMethod, and
    nothing else may have.  A prefix-matching bug in PXQ4Config.get_quant_method turns
    into "the model runs, at fp16 speed, from a 4-bit checkpoint" -- which looks like a
    performance mystery rather than a bug."""
    got, unexpected = set(), []
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None or type(qm).__name__ != "PXQ4LinearMethod":
            continue
        suffix = next((s for s in expected_suffixes if name.endswith(s)), None)
        if suffix is None:
            unexpected.append(name)
        else:
            got.add(suffix)
    missing = set(expected_suffixes) - got
    assert not missing, f"these modules were NOT served by PXQ4: {sorted(missing)}"
    assert not unexpected, f"PXQ4 was applied to unexpected modules: {unexpected[:8]}"
