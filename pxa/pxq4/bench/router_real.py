# SPDX-License-Identifier: Apache-2.0
"""Correction bench: what the REAL vLLM router costs at this model's shape.

layer_census.py stood in a plain torch softmax+topk+renormalise for the router and
measured 25.7 us/layer. That overstates it: vLLM's fused_topk() does not use torch
here, it calls the fused CUDA op ops.topk_softmax, and the fork's Triton fast path
(_sm70_qwen38_router_topk) is gated on E=512/K=10/sm_70 so it does NOT apply to this
model (E=256, K=8, sm_60). This measures the op that actually runs, plus the torch
stand-in beside it so the size of my error is on the record.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.environ.get(
    "PXQ4_SITE", os.environ.get("PXA_PP_ROOT", ".") + "/pxa/pxq4/sidecar/site-sm60"))
from pxq4_vllm import ops as pxops  # noqa: E402

E, TOPK, M = 256, 8, 1


def graph_us(fn, iters=300):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    try:
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run, tag = g.replay, ""
    except Exception as e:
        run, tag = fn, " (eager; capture failed: %s)" % str(e)[:60]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters, tag


def main() -> int:
    pxops.load_library(required=False)
    dev = "cuda"
    print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    gating = torch.randn(M, E, device=dev, dtype=torch.float16)

    tw = torch.empty(M, TOPK, dtype=torch.float32, device=dev)
    ti = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
    tei = torch.empty(M, TOPK, dtype=torch.int32, device=dev)

    try:
        import vllm._custom_ops as vops
        us, tag = graph_us(lambda: vops.topk_softmax(tw, ti, tei, gating, True))
        print("  REAL vllm ops.topk_softmax (E=%d K=%d M=%d): %8.2f us/layer -> "
              "%6.2f ms/token over 40 layers%s" % (E, TOPK, M, us, 40 * us / 1000, tag))
    except Exception as e:
        print("  ops.topk_softmax unavailable:", str(e)[:160])

    def torch_route():
        p = torch.softmax(gating.float(), dim=-1)
        w, i = torch.topk(p, TOPK, dim=-1)
        return w / w.sum(-1, keepdim=True), i
    us2, tag2 = graph_us(torch_route)
    print("  torch stand-in (what layer_census used):        %8.2f us/layer -> "
          "%6.2f ms/token%s" % (us2, 40 * us2 / 1000, tag2))

    # The gate GEMV that feeds it, for completeness, at both shard widths.
    for tp in (1, 2):
        w = (torch.randn(E, 2048, device=dev) * 0.02).half()
        x = torch.randn(1, 2048, device=dev, dtype=torch.float16)
        o = torch.empty(1, E, device=dev, dtype=torch.float16)
        us3, _ = graph_us(lambda: torch.mm(x, w.t(), out=o))
        print("  router gate GEMV [%d,2048] (replicated, TP-independent): %6.2f us" % (E, us3))
        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
