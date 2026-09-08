# SPDX-License-Identifier: Apache-2.0
"""Kernel census and cost split for ONE PXQ4 MoE layer at decode width.

WHAT THIS IS FOR. The 35B MoE decodes at 29.8 tok/s (33.5 ms/token) under vLLM and
95.6 under the llama engine on the same two P100s. Both engines are at 6-19% of the
memory roofline, so the gap is not where the weights live -- it is how many kernels
the step is made of and how much each one costs. This harness answers that for the
MoE half of a layer WITHOUT a two-card window: it builds one layer's worth of PXQ4
expert weights with the real geometry, runs the exact body of PXQ4MoEMethod.apply's
indexed path, and reports (a) how many CUDA kernels one layer costs per token, (b)
how that time splits between the two moe_mmv_out calls and the glue around them, and
(c) what a CUDA graph does to it.

RUNS ON GPU 3. The sm_60 cubin in libpxq4_sm60_v10.so loads on the 1080 Ti's sm_61 by
CUDA's forward minor-version binary compatibility (verified: cuobjdump lists exactly
one sm_60 image and no PTX, and sm_61 >= sm_60 in the same major).

TIMING CAVEAT, STATED UP FRONT SO NO NUMBER FROM HERE IS MISQUOTED. GP102 (1080 Ti)
runs fp16 arithmetic at 1/64 rate and has GDDR5X at 484 GB/s against GP100's full-rate
fp16 and 732 GB/s HBM2. Absolute microseconds from this script are NOT P100 numbers and
must never appear on a board. Kernel COUNTS, kernel NAMES, launch ORDER and the shape
of the split (glue vs mmv) are architecture-independent and are what this is for.

  python3 moe_census.py [tp]      tp = 1 or 2 (default 2, the shipped shape)
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get(
    "PXQ4_SITE", os.environ.get("PXA_PP_ROOT", ".") + "/pxa/pxq4/sidecar/site-sm60"))

from pxq4_vllm import ops as pxops  # noqa: E402

PANEL, SLABC, SLABB = 64, 32, 1088

# The real thing, from coder35-moe-pxq4-m1/config.json.
E, H, I_FULL, TOPK = 256, 2048, 512, 8


def build(tp: int, dev: str = "cuda"):
    I = I_FULL // tp
    n13 = 2 * I
    torch.manual_seed(7)
    # Built straight on the device: these are hundreds of MiB and a host round trip
    # buys nothing. Contents are irrelevant to kernel count and to a bandwidth-bound
    # kernel's timing; the geometry is what has to be exact.
    w13_s = torch.randint(0, 256, (E, n13 // PANEL, H // SLABC, SLABB),
                          dtype=torch.uint8, device=dev)
    w13_a = (torch.randn(E, n13 // PANEL, PANEL, device=dev) * 0.02).half()
    w2_s = torch.randint(0, 256, (E, H // PANEL, I // SLABC, SLABB),
                         dtype=torch.uint8, device=dev)
    w2_a = (torch.randn(E, H // PANEL, PANEL, device=dev) * 0.02).half()
    return I, n13, w13_s, w13_a, w2_s, w2_a


def warm_arenas(w13_s, w13_a, w2_s, w2_a, n13, I, s_max, dev):
    """Mirror process_weights_after_loading: every S the graphs can replay must run
    once eagerly, because the mmv partial arenas refuse to grow under capture."""
    ids = torch.zeros((s_max,), dtype=torch.int32, device=dev)
    x13 = torch.zeros((s_max, H), dtype=torch.float16, device=dev)
    o13 = torch.empty((s_max, n13), dtype=torch.float16, device=dev)
    x2 = torch.zeros((s_max, I), dtype=torch.float16, device=dev)
    o2 = torch.empty((s_max, H), dtype=torch.float16, device=dev)
    for s in range(1, s_max + 1):
        torch.ops.pxq4.moe_mmv_out(o13[:s], x13[:s], ids[:s], w13_s, w13_a)
        torch.ops.pxq4.moe_mmv_out(o2[:s], x2[:s], ids[:s], w2_s, w2_a)
    torch.cuda.synchronize()


def apply_shipped(x2, topk_ids, topk_weights, w13_s, w13_a, w2_s, w2_a, I, n13):
    """VERBATIM the indexed branch of PXQ4MoEMethod.apply (moe.py), so what this
    measures is the shipped path and not a paraphrase of it."""
    M = x2.shape[0]
    out = torch.zeros((M, H), dtype=torch.float16, device=x2.device)  # unused here
    top_k = int(topk_ids.shape[-1])
    s_rows = M * top_k
    ids = topk_ids.reshape(-1).to(torch.int32)
    xg = (x2.unsqueeze(1).expand(M, top_k, H).reshape(s_rows, H).contiguous())
    gu = torch.empty((s_rows, n13), dtype=torch.float16, device=x2.device)
    torch.ops.pxq4.moe_mmv_out(gu, xg, ids, w13_s, w13_a)
    act = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    dn = torch.empty((s_rows, H), dtype=torch.float16, device=x2.device)
    torch.ops.pxq4.moe_mmv_out(dn, act.contiguous(), ids, w2_s, w2_a)
    wts = topk_weights.to(torch.float32).reshape(M, top_k, 1)
    folded = (dn.view(M, top_k, H).to(torch.float32) * wts).sum(dim=1)
    return folded.to(torch.float16), out


def timeit(fn, iters=200):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters  # us per call


def census(fn, iters=20, label=""):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = {}
    for e in p.key_averages():
        if e.device_type.name != "CUDA" and getattr(e, "self_device_time_total", 0) <= 0:
            continue
        t = getattr(e, "self_device_time_total", 0)
        if t <= 0:
            continue
        rows[e.key] = (e.count, t)
    tot_n = sum(v[0] for v in rows.values())
    tot_t = sum(v[1] for v in rows.values())
    print(f"\n  {label}: {tot_n/iters:.1f} CUDA kernels per call, "
          f"{tot_t/iters:.1f} us GPU per call")
    print(f"  {'kernel':<62}{'n/call':>8}{'us/call':>10}{'share':>8}")
    for k, (n, t) in sorted(rows.items(), key=lambda x: -x[1][1]):
        print(f"  {k[:62]:<62}{n/iters:>8.1f}{t/iters:>10.1f}{100*t/tot_t:>7.1f}%")
    return tot_n / iters, tot_t / iters


def main() -> int:
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dev = "cuda"
    if not pxops.load_library(required=False):
        print("FAIL: pxq4 ops did not load; set PXQ4_LIB")
        return 2
    print("pxq4 ops version", pxops.ops_version(),
          "| device", torch.cuda.get_device_name(0),
          "| cc", torch.cuda.get_device_capability(0))

    I, n13, w13_s, w13_a, w2_s, w2_a = build(tp, dev)
    print(f"TP={tp}  E={E} H={H} I_p={I} n13={n13} top_k={TOPK}  "
          f"w13 {w13_s.numel()/2**20:.0f} MiB  w2 {w2_s.numel()/2**20:.0f} MiB")
    for k in (H, I):
        if not torch.ops.pxq4.mmv_supported(int(k)):
            print(f"FAIL: mmv_supported({k}) is False; the indexed path would be off")
            return 3

    s_max = 8 * TOPK
    warm_arenas(w13_s, w13_a, w2_s, w2_a, n13, I, s_max, dev)

    M = 1
    x = torch.randn(M, H, device=dev, dtype=torch.float16)
    ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int64)
    wts = torch.rand(M, TOPK, device=dev, dtype=torch.float16)

    def once():
        apply_shipped(x, ids, wts, w13_s, w13_a, w2_s, w2_a, I, n13)

    us = timeit(once)
    n_k, gpu_us = census(once, label=f"SHIPPED apply(), M=1 TP={tp}")
    print(f"\n  eager wall {us:.1f} us/layer   GPU {gpu_us:.1f} us/layer   "
          f"host gap {us-gpu_us:.1f} us")
    print(f"  x40 layers -> {40*n_k:.0f} MoE kernels and {40*us/1000:.2f} ms of "
          f"eager wall per token from the MoE alone")

    # What a captured graph does to the same body.
    try:
        gpool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                once()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=gpool):
            once()
        gus = timeit(lambda: g.replay())
        print(f"  captured graph replay {gus:.1f} us/layer  "
              f"({us/max(gus,1e-9):.2f}x vs eager)")
    except Exception as e:
        print("  graph capture failed:", str(e)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
