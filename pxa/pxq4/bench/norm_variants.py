# SPDX-License-Identifier: Apache-2.0
"""Can the un-fused norm be made cheaper WITHOUT writing a CUDA kernel?

vllm.ir gives a clean seam for this: every op carries a provider registry and a
priority list (ir/op.py:241 register_impl, :409 set_default), and with no priority set
it logs "Priority not set for op ..., using native implementation" and takes the torch
decomposition. So a plugin can register a leaner provider for sm_60 without touching the
fork. This measures the candidates against the native path for kernel count, time, and
-- the part that decides whether any of them may ship -- exact agreement.
"""
from __future__ import annotations
import time
import torch

H = 2048


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


def graph_us(fn, iters=400):
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
    dev = "cuda"
    eps = 1e-6
    print("device", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    x = torch.randn(1, H, device=dev, dtype=torch.float16)
    r = torch.randn(1, H, device=dev, dtype=torch.float16)
    w = torch.randn(H, device=dev, dtype=torch.float16)

    def native(x, r):
        """vllm.ir.ops.fused_add_rms_norm native, transcribed from ir/ops/layernorm.py."""
        o = x.dtype
        f = x.to(torch.float32) + r.to(torch.float32)
        res = f.to(o)
        var = f.pow(2).mean(dim=-1, keepdim=True)
        f = f * torch.rsqrt(var + eps)
        f = f.to(w.dtype) * w
        return f.to(o), res

    def v_frms(x, r):
        """F.rms_norm over the fp32 sum; same arithmetic order, fewer torch calls."""
        f = x.to(torch.float32) + r.to(torch.float32)
        res = f.to(x.dtype)
        n = torch.nn.functional.rms_norm(f, (H,), None, eps)
        return (n.to(w.dtype) * w).to(x.dtype), res

    def v_frms_w(x, r):
        """F.rms_norm doing the weight multiply too."""
        f = x.to(torch.float32) + r.to(torch.float32)
        res = f.to(x.dtype)
        return torch.nn.functional.rms_norm(f, (H,), w.float(), eps).to(x.dtype), res

    def v_fp16add(x, r):
        """Residual add in fp16 (the model's own dtype), norm in fp32."""
        res = x + r
        n = torch.nn.functional.rms_norm(res.float(), (H,), None, eps)
        return (n.to(w.dtype) * w).to(x.dtype), res

    ref_o, ref_r = native(x, r)
    print("\n  %-26s%10s%11s%14s%14s" % ("variant", "kernels", "us", "max|dO|", "bit-exact"))
    for name, fn in (("native (shipped)", native), ("F.rms_norm", v_frms),
                     ("F.rms_norm+weight", v_frms_w), ("fp16 residual add", v_fp16add)):
        o, res = fn(x, r)
        d = (o.float() - ref_o.float()).abs().max().item()
        dr = (res.float() - ref_r.float()).abs().max().item()
        k = counts(lambda: fn(x, r))
        us = graph_us(lambda: fn(x, r))
        print("  %-26s%10.1f%11.2f%14.3e%14s%s"
              % (name, k, us, d, torch.equal(o, ref_o) and torch.equal(res, ref_r),
                 "" if dr == 0 else "  (residual differs %.1e)" % dr))
    print("\n  x80 fused_add_rms_norm per token: kernels and ms scale directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
