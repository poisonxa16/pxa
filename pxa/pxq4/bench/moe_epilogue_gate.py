# SPDX-License-Identifier: Apache-2.0
"""Gate for the v11 MoE epilogue change, run on GPU 3 before it ever sees the pair.

Three things have to be true before this change is allowed near a window:
  1. legacy mode is BIT-IDENTICAL to the pre-change code. If it is not, the "no-op"
     refactor changed the model and everything measured after it is worthless.
  2. fused mode is numerically close (it is not promised bit-identical -- silu in fp32
     and a bmm reduction order are both deliberate departures) and the size of the
     departure is stated, not waved at.
  3. fused mode actually removes the kernels it claims to remove.

Run under bench/run-gpu3d.sh. GP102 timing caveat as always: counts are real, us are not.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.environ.get(
    "PXQ4_SITE", os.environ.get("PXA_PP_ROOT", ".") + "/pxa/pxq4/sidecar/site-sm60"))
from pxq4_vllm import ops as pxops  # noqa: E402

PANEL, SLABC, SLABB = 64, 32, 1088
E, H, TOPK = 256, 2048, 8


def build(tp, dev="cuda"):
    I = 512 // tp
    n13 = 2 * I
    torch.manual_seed(11)
    return (I, n13,
            torch.randint(0, 256, (E, n13 // PANEL, H // SLABC, SLABB), dtype=torch.uint8, device=dev),
            (torch.randn(E, n13 // PANEL, PANEL, device=dev) * 0.02).half(),
            torch.randint(0, 256, (E, H // PANEL, I // SLABC, SLABB), dtype=torch.uint8, device=dev),
            (torch.randn(E, H // PANEL, PANEL, device=dev) * 0.02).half())


def reference(x2, ids64, wts, w13_s, w13_a, w2_s, w2_a, I, n13):
    """The PRE-CHANGE indexed branch, copied out of the shipped moe.py verbatim."""
    M = x2.shape[0]
    top_k = int(ids64.shape[-1])
    s_rows = M * top_k
    ids = ids64.reshape(-1).to(torch.int32)
    xg = (x2.unsqueeze(1).expand(M, top_k, H).reshape(s_rows, H).contiguous())
    gu = torch.empty((s_rows, n13), dtype=torch.float16, device=x2.device)
    torch.ops.pxq4.moe_mmv_out(gu, xg, ids, w13_s, w13_a)
    act = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    dn = torch.empty((s_rows, H), dtype=torch.float16, device=x2.device)
    torch.ops.pxq4.moe_mmv_out(dn, act.contiguous(), ids, w2_s, w2_a)
    w = wts.to(torch.float32).reshape(M, top_k, 1)
    return (dn.view(M, top_k, H).to(torch.float32) * w).sum(dim=1).to(torch.float16)


def n_kernels(fn, iters=20):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    n = t = 0
    for e in p.key_averages():
        if e.key.startswith("void ") or "kernel" in e.key.lower():
            if getattr(e, "self_device_time_total", 0) > 0:
                n += e.count
                t += e.self_device_time_total
    return n / iters, t / iters


def main() -> int:
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if not pxops.load_library(required=False):
        print("FAIL: ops not loaded"); return 2
    dev = "cuda"
    print("device", torch.cuda.get_device_name(0), "| TP =", tp)
    I, n13, w13_s, w13_a, w2_s, w2_a = build(tp, dev)

    # arena warmup for every S a graph could ask for
    s_max = 8 * TOPK
    z = torch.zeros((s_max,), dtype=torch.int32, device=dev)
    a13 = torch.zeros((s_max, H), dtype=torch.float16, device=dev)
    b13 = torch.empty((s_max, n13), dtype=torch.float16, device=dev)
    a2 = torch.zeros((s_max, I), dtype=torch.float16, device=dev)
    b2 = torch.empty((s_max, H), dtype=torch.float16, device=dev)
    for s in range(1, s_max + 1):
        torch.ops.pxq4.moe_mmv_out(b13[:s], a13[:s], z[:s], w13_s, w13_a)
        torch.ops.pxq4.moe_mmv_out(b2[:s], a2[:s], z[:s], w2_s, w2_a)
    torch.cuda.synchronize()

    class L:  # the bits of the layer object apply() touches
        pass
    layer = L()
    layer.w13_pxq4_slabs, layer.w13_pxq4_anchor = w13_s, w13_a
    layer.w2_pxq4_slabs, layer.w2_pxq4_anchor = w2_s, w2_a
    layer.pxq4_I, layer.pxq4_H, layer.pxq4_n13 = I, H, n13
    layer.pxq4_moe_indexed_ok, layer.pxq4_moe_smax = True, s_max

    fails = 0
    for M in (1, 2, 4, 8):
        x = torch.randn(M, H, device=dev, dtype=torch.float16)
        ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int64)
        wts = torch.rand(M, TOPK, device=dev, dtype=torch.float16)
        ref = reference(x, ids, wts, w13_s, w13_a, w2_s, w2_a, I, n13)

        outs = {}
        for mode in ("legacy", "fused"):
            os.environ["PXQ4_MOE_EPILOGUE"] = mode
            for m in list(sys.modules):
                if m.startswith("pxq4_vllm.moe"):
                    del sys.modules[m]
            import importlib
            moe = importlib.import_module("pxq4_vllm.moe")
            meth = moe.PXQ4MoEMethod.__new__(moe.PXQ4MoEMethod)
            outs[mode] = meth.apply(layer, x, wts, ids)

        same = torch.equal(outs["legacy"], ref)
        d = (outs["fused"].float() - ref.float()).abs()
        rel = (d / ref.float().abs().clamp_min(1e-4)).max().item()
        tokmatch = torch.equal(outs["fused"], ref)
        print(f"  M={M}: legacy bit-identical to pre-change reference: "
              f"{'PASS' if same else '*** FAIL ***'} | "
              f"fused max abs {d.max().item():.3e} max rel {rel:.3e} "
              f"bit-identical {tokmatch}")
        if not same:
            fails += 1

    # kernel counts
    x = torch.randn(1, H, device=dev, dtype=torch.float16)
    ids = torch.randint(0, E, (1, TOPK), device=dev, dtype=torch.int64)
    wts = torch.rand(1, TOPK, device=dev, dtype=torch.float16)
    print()
    print(f"  {'variant':<28}{'kernels/layer':>15}{'us/layer':>11}{'kernels/token (x40)':>22}")
    n, t = n_kernels(lambda: reference(x, ids, wts, w13_s, w13_a, w2_s, w2_a, I, n13))
    print(f"  {'pre-change reference':<28}{n:>15.1f}{t:>11.1f}{40*n:>22.0f}")
    for mode in ("legacy", "fused"):
        os.environ["PXQ4_MOE_EPILOGUE"] = mode
        for m in list(sys.modules):
            if m.startswith("pxq4_vllm.moe"):
                del sys.modules[m]
        import importlib
        moe = importlib.import_module("pxq4_vllm.moe")
        meth = moe.PXQ4MoEMethod.__new__(moe.PXQ4MoEMethod)
        n, t = n_kernels(lambda: meth.apply(layer, x, wts, ids))
        print(f"  {'v11 ' + mode:<28}{n:>15.1f}{t:>11.1f}{40*n:>22.0f}")

    print("\n  GATE", "PASS" if fails == 0 else "FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
