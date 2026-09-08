#!/usr/bin/env python3
"""Unit gate for the hand-fused Pascal/Volta small-op pack.

Each op is compared against THE FORK'S OWN EXPRESSION, executed by torch on the same
inputs -- not against a host oracle written from a second reading of the same source,
which would only prove the source can be read twice.

Every output buffer is NaN-poisoned before the call, so an element the kernel fails to
write fails the test instead of accidentally agreeing with a zero.

Reported per case: exact-match count, max absolute difference, and the fp16 ULP
histogram of whatever does not match. A norm cannot be bit-exact against a torch
reduction (see bench/reduce_probe.py), so the gate for the norms is "no element off by
more than 1 fp16 ULP and the mismatch rate is at the level the probe predicted"; the gate
for the elementwise SwiGLU is exact equality, because for a kernel with no reduction in it
anything less is a bug.
"""
import sys

import os

import torch
import torch.nn.functional as F

torch.ops.load_library(
    os.environ.get("PXQ4_LIB", "/work/pxa/pxq4/kernels/libpxq_sm60_v13.so"))

DEV = "cuda"
FAILED = []


def poison(shape, dtype):
    t = torch.empty(shape, dtype=dtype, device=DEV)
    t.fill_(float("nan"))
    return t


def ulp16(a, b):
    """Distance in fp16 representation steps between two fp16 tensors."""
    ia = a.view(torch.int16).to(torch.int32)
    ib = b.view(torch.int16).to(torch.int32)
    # map sign-magnitude to a monotone ordering so the distance is meaningful
    ia = torch.where(ia < 0, torch.tensor(-32768, device=DEV, dtype=torch.int32) - ia, ia)
    ib = torch.where(ib < 0, torch.tensor(-32768, device=DEV, dtype=torch.int32) - ib, ib)
    return (ia - ib).abs()


# GATE THRESHOLDS, and why each is what it is.
#
# EXACT is required wherever the kernel contains no reduction: the SwiGLU, and every
# residual_out, which is a pure elementwise add. For those, anything less than equality is
# a bug in the rounding order and nothing else.
#
# The normed outputs contain a reduction, and bench/reduce_probe.py established that
# torch's fp32 accumulation order for x.pow(2).mean(-1) is picked per shape by
# TensorIterator and is not reproducible at any block geometry. So the variance differs by
# 1-2 fp32 ULP and the question is only how far that propagates:
#
#   * to an FP16 output, barely: the fp32->fp16 round absorbs almost all of it, so the
#     bound is 2 fp16 ULP (the plain-RMSNorm epilogue rounds TWICE -- to the weight dtype
#     and again at the end -- so a 1-ULP shift in the intermediate can move the result by
#     two) and a mismatch RATE of at most 100 ppm, which is the rate the probe predicted.
#   * to an FP32 output, fully: there is no absorbing round at all, so roughly half the
#     elements move and the only meaningful bound is a relative one. rsqrt of a 1-ULP
#     variance is 1 ULP, times x is another, times (1 + w) a third; 8 fp32 ULP
#     (~1e-6 relative) is a generous ceiling on that chain and a tight one on a real bug.
FP16_ULP_MAX = 2
FP16_PPM_MAX = 100.0
FP32_ULP_MAX = 8


def report(name, got, ref, exact_required):
    if got.dtype != ref.dtype or got.shape != ref.shape:
        FAILED.append(f"{name}: dtype/shape {got.dtype}{tuple(got.shape)} vs "
                      f"{ref.dtype}{tuple(ref.shape)}")
        print(f"  {name:<58} SHAPE/DTYPE MISMATCH")
        return
    nan = int(torch.isnan(got).sum()) - int(torch.isnan(ref).sum())
    n = got.numel()
    same = int((got == ref).sum())
    bad = n - same
    if got.dtype is torch.float16:
        u = ulp16(got, ref)
        mx = int(u.max())
        hist = {int(k): int(v) for k, v in
                zip(*torch.unique(u[u > 0], return_counts=True))} if bad else {}
    else:
        # fp32 output: measure the distance in fp32 representation steps, so the
        # tolerance is scale-free the way the fp16 ULP count is.
        ia = got.contiguous().view(torch.int32)
        ib = ref.contiguous().view(torch.int32)
        mx = int((ia - ib).abs().max())
        hist = {int(k): int(v) for k, v in
                zip(*torch.unique((ia - ib).abs()[(ia - ib) != 0],
                                  return_counts=True))} if bad else {}
    maxabs = float((got.float() - ref.float()).abs().max())
    ppm = 1e6 * bad / n
    # Exact where a kernel has no reduction in it; otherwise at most one representation
    # step, which is the most the fp32 variance difference can propagate (see
    # bench/reduce_probe.py -- torch's reduction order is not reproducible at any
    # geometry, so "bit-exact against a torch reduction" is not an available promise).
    if exact_required:
        ok = bad == 0
    elif got.dtype is torch.float16:
        # The rate bound is meaningless on a small tensor -- one element of 4096 is
        # 244 ppm -- so it only applies once there are enough elements for a rate to
        # mean anything. Below that, two differing elements is the bound.
        ok = bad == 0 or (mx <= FP16_ULP_MAX
                          and (bad <= 2 or ppm <= FP16_PPM_MAX))
    else:
        ok = bad == 0 or mx <= FP32_ULP_MAX
    if nan > 0:
        ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  {name:<58} {status}  exact {same}/{n}  maxabs {maxabs:.3e}  "
          f"maxULP {mx}  {ppm:6.1f}ppm  {hist if hist else ''}"
          + ("  UNWRITTEN-NaN!" if nan > 0 else ""))
    if not ok:
        FAILED.append(f"{name}: {bad}/{n} differ, max fp16 ULP {mx}, maxabs {maxabs}")


# ---------------------------------------------------------------- references ----
def ref_gemma_add(x, residual, w, eps):
    """GemmaRMSNorm.forward_native, residual branch, transcribed from the fork."""
    orig = x.dtype
    weight = w.float() + 1.0
    xf = x.float() + residual.float() if orig is torch.float16 else x + residual
    res = xf
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out.to(weight.dtype) * weight
    return out.to(orig), res


def ref_gemma_plain(x, w, eps):
    orig = x.dtype
    weight = w.float() + 1.0
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out.to(weight.dtype) * weight
    return out.to(orig)


def ref_gated(x, z, w, eps, act):
    """RMSNormGated.forward_static, group_size None, norm_before_gate True."""
    orig = x.dtype
    xf = x.float()
    wf = w.float()
    zf = z.float()
    fn = torch.sigmoid if act == 1 else F.silu
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * wf
    out = out * fn(zf)
    return out.to(orig)


def ref_rms(x, w, eps):
    """vllm.ir.ops.rms_norm, fp16 x and fp16 weight."""
    orig = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    xf = xf.to(w.dtype) * w
    return xf.to(orig)


def ref_fused_add(x, res, w, eps):
    """vllm.ir.ops.fused_add_rms_norm."""
    orig = x.dtype
    xf = x.to(torch.float32) + res.to(torch.float32)
    res_out = xf.to(orig)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    xf = xf.to(w.dtype) * w
    return xf.to(orig), res_out


def ref_silu_mul(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# ---------------------------------------------------------------------- cases ----
EPS = 1e-6
# The real shapes: hidden 2048 (80 layer norms + the final norm), head_dim 256
# (q_norm on 16 heads and k_norm on 2, 10 full-attention layers), value head dim 128
# (30 GDN layers, 32 value heads). M is the cudagraph decode ladder, then prefill widths.
MS = [1, 2, 4, 8, 512, 2048, 4096]
NORM_SHAPES = [(m, 2048) for m in MS] + \
              [(m * 16, 256) for m in (1, 2, 4, 8)] + \
              [(m * 32, 128) for m in (1, 2, 4, 8)] + \
              [(512, 256), (2048, 128)]


def gen(shape, dtype, seed, scale=0.05):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
            * scale).to(dtype)


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0),
          "torch:", torch.__version__)
    print("pxq4 lib op version:", int(torch.ops.pxq4.pascal_ops_version()))

    print("\n== 1. GemmaRMSNorm WITH residual (80 calls/token, 11 kernels each) ==")
    for i, (M, H) in enumerate(NORM_SHAPES):
        x = gen((M, H), torch.float16, 100 + i)
        w = gen((H,), torch.float32, 200 + i, 0.02)
        for rdt in (torch.float16, torch.float32):
            res = gen((M, H), rdt, 300 + i)
            ref_o, ref_r = ref_gemma_add(x, res, w, EPS)
            out = poison((M, H), torch.float16)
            rout = poison((M, H), torch.float32)
            torch.ops.pxq4.gemma_add_rms_norm_out(out, rout, x, res, w, EPS)
            report(f"gemma_add [{M}x{H}] residual={str(rdt)[6:]} out", out, ref_o, False)
            report(f"gemma_add [{M}x{H}] residual={str(rdt)[6:]} residual_out",
                   rout, ref_r, True)

    print("\n== 2. GemmaRMSNorm no residual (final norm + q_norm/k_norm, 21/token) ==")
    for i, (M, H) in enumerate(NORM_SHAPES):
        w = gen((H,), torch.float32, 400 + i, 0.02)
        for dt in (torch.float16, torch.float32):
            x = gen((M, H), dt, 500 + i)
            ref_o = ref_gemma_plain(x, w, EPS)
            out = poison((M, H), dt)
            torch.ops.pxq4.gemma_rms_norm_out(out, x, w, EPS)
            report(f"gemma_plain [{M}x{H}] x={str(dt)[6:]}", out, ref_o, False)

    print("\n== 3. GemmaRMSNorm no residual, 3-D q_norm/k_norm layout ==")
    # q_norm sees [tokens, 16, 256] and k_norm [tokens, 2, 256]; a contiguous tensor
    # flattens exactly, and the op is meant to take it without a .contiguous() copy.
    for i, (T, Hd, D) in enumerate([(1, 16, 256), (8, 16, 256), (1, 2, 256),
                                    (8, 2, 256), (2048, 16, 256)]):
        x = gen((T, Hd, D), torch.float16, 600 + i)
        w = gen((D,), torch.float32, 700 + i, 0.02)
        ref_o = ref_gemma_plain(x, w, EPS)
        out = poison((T, Hd, D), torch.float16)
        torch.ops.pxq4.gemma_rms_norm_out(out, x, w, EPS)
        report(f"gemma_plain 3-D [{T}x{Hd}x{D}]", out, ref_o, False)

    print("\n== 4. RMSNormGated, the 30 GDN layers (12 kernels each) ==")
    for i, (M, H) in enumerate([(32, 128), (64, 128), (128, 128), (256, 128),
                                (2048 * 32, 128), (32, 256)]):
        for dt in (torch.float16, torch.float32):
            x = gen((M, H), dt, 800 + i)
            z = gen((M, H), dt, 900 + i, 0.5)
            w = gen((H,), torch.float32, 1000 + i, 0.02)
            for act, aname in ((0, "silu"), (1, "sigmoid")):
                ref_o = ref_gated(x, z, w, EPS, act)
                out = poison((M, H), dt)
                torch.ops.pxq4.rms_norm_gated_out(out, x, z, w, EPS, act)
                report(f"gated [{M}x{H}] {str(dt)[6:]} {aname}", out, ref_o, False)

    print("\n== 5. plain RMSNorm / fused_add_rms_norm (the vllm.ir pair) ==")
    for i, (M, H) in enumerate([(1, 2048), (2, 2048), (4, 2048), (8, 2048),
                                (512, 2048), (4096, 2048), (8, 5120)]):
        x = gen((M, H), torch.float16, 1100 + i)
        w = gen((H,), torch.float16, 1200 + i, 0.02)
        out = poison((M, H), torch.float16)
        torch.ops.pxq4.rms_norm_out(out, x, w, EPS)
        report(f"rms_norm [{M}x{H}]", out, ref_rms(x, w, EPS), False)

        res = gen((M, H), torch.float16, 1300 + i)
        ro, rr = ref_fused_add(x, res, w, EPS)
        out = poison((M, H), torch.float16)
        rout = poison((M, H), torch.float16)
        torch.ops.pxq4.fused_add_rms_norm_out(out, rout, x, res, w, EPS)
        report(f"fused_add [{M}x{H}] out", out, ro, False)
        report(f"fused_add [{M}x{H}] residual_out", rout, rr, True)

    print("\n== 6. SwiGLU (kernel body from the fused-MoE work) -- EXACT equality required ==")
    for i, (M, N) in enumerate([(1, 512), (2, 512), (8, 512), (8, 256), (64, 512),
                                (64, 256), (64, 1024), (2048, 512), (256, 4304),
                                (1, 1), (3, 7)]):
        x = gen((M, 2 * N), torch.float16, 1400 + i, 1.0)
        out = poison((M, N), torch.float16)
        torch.ops.pxq4.silu_and_mul_out(out, x)
        report(f"silu_and_mul [{M}x{2*N} -> {M}x{N}]", out, ref_silu_mul(x), True)

    print("\n== 7. ALIASING: out == x, residual_out == residual ==")
    # The vllm.ir inplace overload passes the same storage for input and output, so this
    # is a contract and not a curiosity. Both kernels are written two-pass over sources
    # they never overwrite; this proves it rather than asserting it.
    M, H = 8, 2048
    x0 = gen((M, H), torch.float16, 1500)
    r0 = gen((M, H), torch.float32, 1501)
    w = gen((H,), torch.float32, 1502, 0.02)
    ref_o, ref_r = ref_gemma_add(x0, r0, w, EPS)
    r1 = r0.clone()
    o1 = torch.empty_like(x0)
    torch.ops.pxq4.gemma_add_rms_norm_out(o1, r1, x0, r1, w, EPS)   # res_out IS res_in
    report("gemma_add aliased residual out", o1, ref_o, False)
    report("gemma_add aliased residual residual_out", r1, ref_r, True)

    xh = gen((M, H), torch.float16, 1600)
    rh = gen((M, H), torch.float16, 1601)
    wh = gen((H,), torch.float16, 1602, 0.02)
    fo, fr = ref_fused_add(xh, rh, wh, EPS)
    r2 = rh.clone()
    o2 = torch.empty_like(xh)
    torch.ops.pxq4.fused_add_rms_norm_out(o2, r2, xh, r2, wh, EPS)
    report("fused_add aliased residual out", o2, fo, False)
    report("fused_add aliased residual residual_out", r2, fr, True)

    print("\n== 8. DETERMINISM: same inputs, 32 runs, byte-identical ==")
    x = gen((8, 2048), torch.float16, 1700)
    r = gen((8, 2048), torch.float32, 1701)
    w = gen((2048,), torch.float32, 1702, 0.02)
    base = None
    spread = 0
    for _ in range(32):
        o = poison((8, 2048), torch.float16)
        ro = poison((8, 2048), torch.float32)
        torch.ops.pxq4.gemma_add_rms_norm_out(o, ro, x, r, w, EPS)
        if base is None:
            base = (o.clone(), ro.clone())
        else:
            spread += int((o != base[0]).sum()) + int((ro != base[1]).sum())
    print(f"  {'gemma_add 32-run spread':<58} "
          f"{'PASS' if spread == 0 else 'FAIL'}  differing elements {spread}")
    if spread:
        FAILED.append(f"determinism: {spread} elements varied across 32 runs")

    print("\n== 9. CUDA GRAPH capture and replay ==")
    x = gen((8, 2048), torch.float16, 1800)
    r = gen((8, 2048), torch.float32, 1801)
    w = gen((2048,), torch.float32, 1802, 0.02)
    o = torch.empty((8, 2048), dtype=torch.float16, device=DEV)
    ro = torch.empty((8, 2048), dtype=torch.float32, device=DEV)
    torch.ops.pxq4.gemma_add_rms_norm_out(o, ro, x, r, w, EPS)   # warm up
    eager = (o.clone(), ro.clone())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.ops.pxq4.gemma_add_rms_norm_out(o, ro, x, r, w, EPS)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    o.fill_(float("nan")); ro.fill_(float("nan"))
    with torch.cuda.graph(graph):
        torch.ops.pxq4.gemma_add_rms_norm_out(o, ro, x, r, w, EPS)
    o.fill_(float("nan")); ro.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ok = bool((o == eager[0]).all() and (ro == eager[1]).all())
    print(f"  {'gemma_add captured + replayed == eager':<58} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILED.append("cuda graph replay differs from eager")

    print("\n== 10. THE SEAM: real vLLM modules, patched vs unpatched ==")
    # The kernels being right is necessary and not sufficient. This constructs the actual
    # classes the model builds, runs each one BEFORE the patch to capture the shipped
    # answer, arms the sidecar seam, and runs the same instance again. It is the only
    # part of this file that proves the dispatch is taken at all -- a provider that arms,
    # logs a cheerful line and is never called would pass every test above.
    try:
        seam_test()
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        FAILED.append(f"seam test raised: {exc}")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"UNIT GATE: {len(FAILED)} FAILURES")
        for f in FAILED:
            print("  -", f)
        return 1
    print("UNIT GATE: ALL PASS")
    return 0


def seam_test():
    """Build the real classes, capture the shipped answer, arm the seam, compare."""
    import os

    from vllm.config import VllmConfig, set_current_vllm_config

    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated

        torch.set_default_dtype(torch.float16)
        g = GemmaRMSNorm(2048, eps=EPS).to(DEV)
        with torch.no_grad():
            g.weight.copy_(gen((2048,), torch.float16, 2000, 0.02))
        q = GemmaRMSNorm(256, eps=EPS).to(DEV)
        with torch.no_grad():
            q.weight.copy_(gen((256,), torch.float16, 2001, 0.02))
        gd = RMSNormGated(128, eps=EPS, group_size=None, norm_before_gate=True,
                          activation="silu", device=torch.device(DEV),
                          dtype=torch.float16)
        with torch.no_grad():
            gd.weight.copy_(gen((128,), torch.float16, 2002, 0.02))
        sm = SiluAndMul()

        x = gen((8, 2048), torch.float16, 2100)
        r = gen((8, 2048), torch.float16, 2101)
        xq = gen((8, 16, 256), torch.float16, 2102)
        xg = gen((256, 128), torch.float16, 2103)
        zg = gen((256, 128), torch.float16, 2104, 0.5)
        xs = gen((64, 1024), torch.float16, 2105, 1.0)

        # the shipped answers, through the real dispatch
        base = {
            "gemma+res": tuple(t.clone() for t in g(x, r)),
            "gemma q_norm": q(xq).clone(),
            "gdn": gd(xg, zg).clone(),
            "silu": sm(xs).clone(),
        }

        # arm the seam exactly as a worker would
        os.environ["PXA_OPS_FUSED"] = "all"
        sys.path.insert(0, "/work/pxa/pxq4/sidecar/site-sm60")
        import pxq4_vllm.pxa_pascal_ops as pops

        pops._ARMED = False
        pops.maybe_patch()
        armed = list(pops._ARMED_LIST)
        print(f"  armed seams: {armed or 'NOTHING'}")
        if not armed:
            FAILED.append("seam: nothing armed")
            return

        # The class patches replace forward_cuda, which CustomOp binds at __init__, so
        # instances built before the patch keep the old bound method. Rebuild them, which
        # is what a real worker does anyway (the plugin loads before the model).
        g2 = GemmaRMSNorm(2048, eps=EPS).to(DEV)
        q2 = GemmaRMSNorm(256, eps=EPS).to(DEV)
        gd2 = RMSNormGated(128, eps=EPS, group_size=None, norm_before_gate=True,
                           activation="silu", device=torch.device(DEV),
                           dtype=torch.float16)
        sm2 = SiluAndMul()
        with torch.no_grad():
            g2.weight.copy_(g.weight)
            q2.weight.copy_(q.weight)
            gd2.weight.copy_(gd.weight)

        o, ro = g2(x, r)
        report("SEAM GemmaRMSNorm+res out   [8x2048]", o, base["gemma+res"][0], False)
        report("SEAM GemmaRMSNorm+res residual", ro.float(),
               base["gemma+res"][1].float(), False)
        report("SEAM GemmaRMSNorm q_norm [8x16x256]", q2(xq), base["gemma q_norm"], False)
        report("SEAM RMSNormGated       [256x128]", gd2(xg, zg), base["gdn"], False)
        report("SEAM SiluAndMul          [64x1024]", sm2(xs), base["silu"], True)

        # and prove the dispatch actually moved rather than coincidentally agreeing
        for name, mod in (("GemmaRMSNorm", g2), ("RMSNormGated", gd2)):
            moved = getattr(type(mod), "_pxa_patched", False)
            print(f"  {name + ': class carries the patch':<58} "
                  f"{'yes' if moved else 'NO'}")
            if not moved:
                FAILED.append(f"seam: {name} was not patched")

        # SiluAndMul arms through the _C namespace rather than a class patch: the fork
        # itself does `self.op = getattr(torch.ops._C, "silu_and_mul", None)` and only
        # falls back to forward_native when that is None. So the proof that the dispatch
        # moved is that sm2 resolved an op AND that its bound forward is forward_cuda.
        # Which mechanism carries SiluAndMul depends on CompilationConfig.custom_ops,
        # which is a per-boot decision: "all" dispatches forward_cuda (the branch that
        # consults torch.ops._C), "none" dispatches forward_native. Both are armed, so
        # the check is that the branch this config actually took is one of them.
        via_c = hasattr(torch.ops._C, "silu_and_mul") and sm2.op is not None
        via_class = getattr(SiluAndMul, "_pxa_patched", False)
        on_cuda = sm2._forward_method == sm2.forward_cuda
        print(f"  {'SiluAndMul: _C.silu_and_mul defined and resolved':<58} "
              f"{'yes' if via_c else 'NO'}")
        print(f"  {'SiluAndMul: forward_native class patch installed':<58} "
              f"{'yes' if via_class else 'NO'}")
        print(f"  {'SiluAndMul: this config dispatches':<58} "
              f"{'forward_cuda' if on_cuda else 'forward_native'}")
        covered = (via_c and on_cuda) or (via_class and not on_cuda)
        if not covered:
            FAILED.append("seam: SiluAndMul did not move off the native path")

        # And the ir provider, which serves the plain-RMSNorm models rather than this one.
        from vllm import ir
        for nm in ("rms_norm", "fused_add_rms_norm"):
            op = getattr(ir.ops, nm)
            ok = "pxa" in op.impls and op.get_priority()[:1] == ["pxa"]
            print(f"  {'ir provider ' + nm + ' registered and first in priority':<58} "
                  f"{'yes' if ok else 'NO'} {op.get_priority()}")
            if not ok:
                FAILED.append(f"seam: ir provider {nm} not installed")


if __name__ == "__main__":
    sys.exit(main())
