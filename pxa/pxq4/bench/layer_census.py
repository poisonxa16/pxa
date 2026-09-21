# SPDX-License-Identifier: Apache-2.0
"""Per-token cost model for the whole 35B MoE decode step, built from the REAL shapes.

Companion to moe_census.py. That one showed the routed MoE costs ~12 kernels and
~99 us per layer at the TP=2 shard on GP102 -- which, at 40 layers, is far too little
to account for a 33.5 ms token. So the question becomes: where is the rest? This
harness times every OTHER weight-consuming operation in one decoder layer, at decode
width M=1, at its exact checkpoint shape, each inside a captured CUDA graph so the
number is GPU work and not Python.

Shapes are read off the safetensors headers of coder35-moe-pxq4-m1, not guessed:
  GDN layer   (30 of 40): in_proj_qkv PXQ4 N=8192 K=2048 | in_proj_z PXQ4 N=4096 K=2048
                          out_proj    F16  [2048,4096]   | in_proj_a/b F16 [32,2048]
  full attn   (10 of 40): q_proj F16 [8192,2048] | k,v F16 [512,2048] each
                          o_proj PXQ4 N=2048 K=4096
  every layer:            router F16 [256,2048] + softmax + top-8
                          shared expert PXQ4 gate,up N=512 K=2048 ; down N=2048 K=512
  once per token:         lm_head F16 [248320,2048]

TIMING CAVEAT, same as moe_census.py: GP102 is 1/64-rate fp16 and 484 GB/s GDDR5X
against GP100's full-rate fp16 and 732 GB/s HBM2. Absolute microseconds are NOT P100
numbers and no board row may quote them. What survives the arch change is the SHAPE of
the answer: which operations dominate, and by roughly how much.

  python3 layer_census.py [tp]
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.environ.get(
    "PXQ4_SITE", os.environ.get("PXA_PP_ROOT", ".") + "/pxa/pxq4/sidecar/site-sm60"))

from pxq4_vllm import ops as pxops  # noqa: E402

PANEL, SLABC, SLABB = 64, 32, 1088
H, VOCAB, TOPK, E = 2048, 248320, 8, 256
N_GDN, N_FULL, N_LAYER = 30, 10, 40


def pxq4_pair(N: int, K: int, dev="cuda"):
    s = torch.randint(0, 256, (N // PANEL, K // SLABC, SLABB), dtype=torch.uint8,
                      device=dev)
    a = (torch.randn(N // PANEL, PANEL, device=dev) * 0.02).half()
    return s, a


def graph_us(fn, iters=200):
    """Capture fn once and time the replay. Returns us/call, or eager us on failure."""
    import time
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    try:
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run, tag = g.replay, ""
    except Exception:
        run, tag = fn, " (eager, capture failed)"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters, tag


def main() -> int:
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dev = "cuda"
    if not pxops.load_library(required=False):
        print("FAIL: pxq4 ops did not load")
        return 2
    print("device", torch.cuda.get_device_name(0), "cc",
          torch.cuda.get_device_capability(0), "| TP =", tp)
    x = torch.randn(1, H, device=dev, dtype=torch.float16)

    rows = []   # (label, per-layer count, us, note)

    def pxq4_row(label, N, K, count, shard="N"):
        Ns, Ks = (N // tp, K) if shard == "N" else (N, K // tp)
        if Ns % PANEL or Ks % SLABC:
            rows.append((label, count, float("nan"), "geometry rejects TP=%d" % tp))
            return
        s, a = pxq4_pair(Ns, Ks)
        xin = torch.randn(1, Ks, device=dev, dtype=torch.float16)
        out = torch.empty(1, Ns, device=dev, dtype=torch.float16)
        us, tag = graph_us(lambda: torch.ops.pxq4.linear_out(out, xin, s, a))
        rows.append((label, count, us, "PXQ4 %dx%d%s" % (Ns, Ks, tag)))
        del s, a
        torch.cuda.empty_cache()

    def f16_row(label, N, K, count, shard="N"):
        Ns, Ks = (N // tp, K) if shard == "N" else (N, K // tp)
        w = (torch.randn(Ns, Ks, device=dev) * 0.02).half()
        xin = torch.randn(1, Ks, device=dev, dtype=torch.float16)
        out = torch.empty(1, Ns, device=dev, dtype=torch.float16)
        us, tag = graph_us(lambda: torch.mm(xin, w.t(), out=out))
        note = "F16 cuBLAS %dx%d%s" % (Ns, Ks, tag)
        if hasattr(torch.ops.pxq4, "f16_mmv_out"):
            try:
                us2, _ = graph_us(lambda: torch.ops.pxq4.f16_mmv_out(out, xin, w))
                note += "  | pxq4 f16_mmv %.1f us" % us2
                us = min(us, us2)
            except Exception as e:
                note += "  | f16_mmv failed: %s" % str(e)[:40]
        rows.append((label, count, us, note))
        del w
        torch.cuda.empty_cache()

    # ---- GDN layers (30) -------------------------------------------------
    pxq4_row("GDN in_proj_qkv", 8192, H, N_GDN)
    pxq4_row("GDN in_proj_z", 4096, H, N_GDN)
    f16_row("GDN out_proj", H, 4096, N_GDN, shard="K")
    f16_row("GDN in_proj_a+b", 64, H, N_GDN)
    # ---- full-attention layers (10) --------------------------------------
    f16_row("ATT q_proj", 8192, H, N_FULL)
    f16_row("ATT k_proj+v_proj", 1024, H, N_FULL)
    pxq4_row("ATT o_proj", H, 4096, N_FULL, shard="K")
    # ---- every layer -----------------------------------------------------
    f16_row("router gate", E, H, N_LAYER)
    pxq4_row("shared gate+up", 1024, H, N_LAYER)
    pxq4_row("shared down", H, 512, N_LAYER, shard="K")
    # router epilogue: softmax + top-8 + renorm, the shape vLLM actually runs
    g = torch.randn(1, E, device=dev, dtype=torch.float16)

    def route():
        p = torch.softmax(g.float(), dim=-1)
        w, i = torch.topk(p, TOPK, dim=-1)
        return w / w.sum(-1, keepdim=True), i
    us, tag = graph_us(route)
    rows.append(("router softmax+top8", N_LAYER, us, "E=%d%s" % (E, tag)))
    # ---- once per token --------------------------------------------------
    f16_row("lm_head", VOCAB, H, 1)

    print("\n  %-24s%7s%11s%11s   %s" % ("op", "n/tok", "us each", "us/token", "note"))
    tot = 0.0
    for label, n, us, note in sorted(rows, key=lambda r: -(r[1] * (r[2] if r[2] == r[2] else 0))):
        t = n * us
        tot += 0 if t != t else t
        print("  %-24s%7d%11.1f%11.1f   %s" % (label, n, us, t, note))
    print("  %-24s%7s%11s%11.1f   (routed MoE excluded; see moe_census.py)"
          % ("TOTAL", "", "", tot))
    print("\n  Add the routed MoE from moe_census.py (graph replay us/layer x 40) "
          "to get the whole weight-consuming cost of a token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
