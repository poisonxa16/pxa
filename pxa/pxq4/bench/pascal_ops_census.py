#!/usr/bin/env python3
"""Kernel-count census: the acceptance number for the fused small-op pack.

main's acceptance criterion is a COUNT, not a microsecond: 1450 norm kernels per decode
token down to roughly one launch per call. GPU 3 is a GP102 with 1/64-rate fp16, so no
timing from this box is a P100 number and none of it goes on a board -- but a kernel count
is exact and architecture-independent, which is precisely why it is the criterion.

Method: build the REAL vLLM module (not a transcription), run it once to warm every
allocator and autotune path, then profile a fixed number of iterations with the CUDA
activity on and count the device kernel launches. Native and fused are measured in the
same process on the same tensors, so the only variable is the seam.
"""
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

torch.ops.load_library(
    os.environ.get("PXQ4_LIB", "/work/pxa/pxq4/kernels/libpxq_sm60_v13.so"))

DEV = "cuda"
ITERS = 20


def kernels_per_call(fn, *args):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn(*args)
        torch.cuda.synchronize()
    n = 0
    for e in prof.events():
        # device kernels only: no memcpy, no runtime/host rows
        if str(getattr(e, "device_type", "")).endswith("CUDA") and e.self_device_time_total >= 0:
            if getattr(e, "key", "").startswith(("cudaLaunch", "Memcpy", "Memset")):
                continue
    ka = prof.key_averages()
    total = 0
    for e in ka:
        if str(e.device_type).endswith("CUDA") or e.self_device_time_total > 0:
            pass
    # Simpler and unambiguous: count kernel events in the raw trace.
    total = sum(1 for e in prof.events()
                if getattr(e, "device_type", None) is not None
                and str(e.device_type) == "DeviceType.CUDA"
                and getattr(e, "self_device_time_total", 0) is not None)
    return total / ITERS


def gen(shape, dtype, seed, scale=0.05):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
            * scale).to(dtype)


def main():
    from vllm.config import VllmConfig, set_current_vllm_config

    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated

        torch.set_default_dtype(torch.float16)

        # the real shapes: coder35 / fusion2-35b, TP=2, decode M=8 (top of the ladder)
        M = 8
        g2048 = GemmaRMSNorm(2048, eps=1e-6).to(DEV)
        g256 = GemmaRMSNorm(256, eps=1e-6).to(DEV)
        gdn = RMSNormGated(128, eps=1e-6, group_size=None, norm_before_gate=True,
                           activation="silu", device=torch.device(DEV),
                           dtype=torch.float16)
        silu = SiluAndMul()

        x2048 = gen((M, 2048), torch.float16, 10)
        r2048f = gen((M, 2048), torch.float32, 11)
        xq = gen((M, 16, 256), torch.float16, 12)
        xk = gen((M, 2, 256), torch.float16, 13)
        xg = gen((M * 32, 128), torch.float16, 14)
        zg = gen((M * 32, 128), torch.float16, 15, 0.5)
        xs = gen((M, 512), torch.float16, 16, 1.0)

        CASES = [
            ("GemmaRMSNorm h=2048 + residual", 80, lambda: g2048(x2048, r2048f)),
            ("GemmaRMSNorm h=2048 no residual", 1, lambda: g2048(x2048)),
            ("GemmaRMSNorm q_norm h=256", 10, lambda: g256(xq)),
            ("GemmaRMSNorm k_norm h=256", 10, lambda: g256(xk)),
            ("RMSNormGated h=128 (GDN)", 30, lambda: gdn(xg, zg)),
            ("SwiGLU (SiluAndMul)", 80, lambda: silu(xs)),
        ]

        native = [(n, c, kernels_per_call(f)) for n, c, f in CASES]

        os.environ["PXA_OPS_FUSED"] = "all"
        sys.path.insert(0, "/work/pxa/pxq4/sidecar/site-sm60")
        import pxq4_vllm.pxa_pascal_ops as pops
        pops._ARMED = False
        pops.maybe_patch()
        print("armed:", pops._ARMED_LIST)

        g2048b = GemmaRMSNorm(2048, eps=1e-6).to(DEV)
        g256b = GemmaRMSNorm(256, eps=1e-6).to(DEV)
        gdnb = RMSNormGated(128, eps=1e-6, group_size=None, norm_before_gate=True,
                            activation="silu", device=torch.device(DEV),
                            dtype=torch.float16)
        silub = SiluAndMul()
        CASES2 = [
            ("GemmaRMSNorm h=2048 + residual", 80, lambda: g2048b(x2048, r2048f)),
            ("GemmaRMSNorm h=2048 no residual", 1, lambda: g2048b(x2048)),
            ("GemmaRMSNorm q_norm h=256", 10, lambda: g256b(xq)),
            ("GemmaRMSNorm k_norm h=256", 10, lambda: g256b(xk)),
            ("RMSNormGated h=128 (GDN)", 30, lambda: gdnb(xg, zg)),
            ("SwiGLU (SiluAndMul)", 80, lambda: silub(xs)),
        ]
        fused = [(n, c, kernels_per_call(f)) for n, c, f in CASES2]

        print(f"\n{'call site':<34}{'per token':>10}{'native':>9}{'fused':>7}"
              f"{'native/tok':>12}{'fused/tok':>11}")
        tn = tf = 0
        for (n, c, kn), (_, _, kf) in zip(native, fused):
            tn += c * kn
            tf += c * kf
            print(f"{n:<34}{c:>10}{kn:>9.1f}{kf:>7.1f}{c*kn:>12.0f}{c*kf:>11.0f}")
        print(f"{'TOTAL':<34}{'':>10}{'':>9}{'':>7}{tn:>12.0f}{tf:>11.0f}")
        print(f"\nkernels per decode token: {tn:.0f} -> {tf:.0f}  "
              f"({tn - tf:.0f} deleted, {100*(tn-tf)/tn:.1f}%)")


if __name__ == "__main__":
    main()
