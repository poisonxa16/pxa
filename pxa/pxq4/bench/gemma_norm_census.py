# SPDX-License-Identifier: Apache-2.0
"""CORRECTION BENCH: what this model's norms actually are, and what they actually cost.

My first norm census (bench/norm_census.py) measured vllm.ir.ops.rms_norm and
fused_add_rms_norm directly. This model never calls fused_add_rms_norm. Qwen3_5 aliases
its norm to GemmaRMSNorm (qwen3_5.py:41 'GemmaRMSNorm as Qwen3_5RMSNorm', qwen3_next.py:35
for the decoder layer and q/k norms), and GemmaRMSNorm.forward_native does the residual
add ITSELF and then calls ir.ops.rms_norm - never the fused variant - with two extra
wrinkles that change the kernel entirely:
  * the weight is recomputed every single call: ``weight = self.weight.data.float() + 1.0``
    (Gemma's 1+w convention), which is a cast and an add over the hidden vector per call;
  * for fp16 input the residual is carried in FP32 between layers, not fp16.
The GDN layers use RMSNormGated instead, which the fork's own comment calls "a nine-kernel
native chain" (qwen_gdn_linear_attn.py:4028).

So the acceptance table a kernel needs is THIS one, not the earlier one. Instantiating
the real classes rather than transcribing them, so the dispatch and the pass_weight logic
are the ones that run.

GP102 caveat unchanged: kernel counts exact, microseconds not P100 microseconds.
"""
from __future__ import annotations
import time
import torch

H, HEAD_DIM, V_HEAD, N_V_HEADS = 2048, 256, 128, 32


def counts(fn, iters=30):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    n = 0
    for e in p.key_averages():
        if e.key.startswith("void ") and getattr(e, "self_device_time_total", 0) > 0:
            n += e.count
    return n / iters


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
        run = g.replay
    except Exception:
        run = fn
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main() -> int:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
    dev = "cuda"
    torch.set_default_dtype(torch.float16)
    cfg = VllmConfig()
    rows = []
    with set_current_vllm_config(cfg):
        print("device", torch.cuda.get_device_name(0))
        print("ir_op_priority:", cfg.kernel_config.ir_op_priority)

        g_h = GemmaRMSNorm(H, eps=1e-6).to(dev).half()
        print("GemmaRMSNorm attrs:", [a for a in ("pass_weight","pass_weight_add",
              "has_weight","variance_size_override") if hasattr(g_h, a)])
        x = torch.randn(1, H, device=dev, dtype=torch.float16)
        res32 = torch.randn(1, H, device=dev, dtype=torch.float32)
        rows.append(("GemmaRMSNorm h=2048 +residual", 80,
                     counts(lambda: g_h(x, res32)), graph_us(lambda: g_h(x, res32))))
        rows.append(("GemmaRMSNorm h=2048 no residual", 1,
                     counts(lambda: g_h(x)), graph_us(lambda: g_h(x))))

        g_q = GemmaRMSNorm(HEAD_DIM, eps=1e-6).to(dev).half()
        xq = torch.randn(1 * 16, HEAD_DIM, device=dev, dtype=torch.float16)
        rows.append(("GemmaRMSNorm q_norm h=256", 10,
                     counts(lambda: g_q(xq)), graph_us(lambda: g_q(xq))))
        xk = torch.randn(1 * 2, HEAD_DIM, device=dev, dtype=torch.float16)
        rows.append(("GemmaRMSNorm k_norm h=256", 10,
                     counts(lambda: g_q(xk)), graph_us(lambda: g_q(xk))))

        try:
            gg = RMSNormGated(V_HEAD, eps=1e-6, group_size=None).to(dev).half()
        except TypeError:
            gg = RMSNormGated(V_HEAD, eps=1e-6).to(dev).half()
        xg = torch.randn(1 * N_V_HEADS, V_HEAD, device=dev, dtype=torch.float16)
        zg = torch.randn(1 * N_V_HEADS, V_HEAD, device=dev, dtype=torch.float16)
        try:
            rows.append(("RMSNormGated h=128 (GDN)", 30,
                         counts(lambda: gg(xg, zg)), graph_us(lambda: gg(xg, zg))))
        except Exception as e:
            print("  RMSNormGated could not be exercised:", str(e)[:140])

    print("\n  %-34s%9s%12s%12s%11s%11s"
          % ("norm", "n/token", "kern each", "kern/token", "us each", "ms/token"))
    tk = tt = 0.0
    for label, n, k, us in rows:
        tk += n * k; tt += n * us / 1000
        print("  %-34s%9d%12.1f%12.0f%11.2f%11.3f" % (label, n, k, n * k, us, n * us / 1000))
    print("  %-34s%9s%12s%12.0f%11s%11.3f" % ("TOTAL norms", "", "", tk, "", tt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
