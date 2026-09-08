#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Count the CUDA kernel launches ONE MoE layer costs, unfused vs fused.

A COUNT, not a time. It is run inside a correctness-only window on purpose: the number it
produces is a structural property of the code path (how many nodes the captured graph holds
for this layer), which is exactly what the speed window then needs in order to interpret its
own t/s. No wall-clock number is taken or claimed here.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxq_moe_fused_test import (TIER_OPS, load_lib, read_cfg, tiers_of, upload_books,
                                stack_experts)


def count(fn) -> int:
    fn()   # warm: first-call autotune/alloc must not be counted as steady state
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(int(e.count) for e in p.key_averages()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total >= 0
               and e.key not in ("cudaDeviceSynchronize",) and e.device_time_total > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--M", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device("cuda")
    load_lib()
    cfg = read_cfg(a.model)
    t13, t2 = tiers_of(cfg)
    upload_books(cfg, {t13, t2})
    w13_s, w13_a, w2_s, w2_a, H, Ip = stack_experts(a.model, 0, a.experts, 0, a.tp, dev)
    E, top_k, M = w13_s.shape[0], 8, a.M
    S = M * top_k
    x = (torch.randn((M, H), device=dev) * 0.5).to(torch.float16)
    ids = torch.randint(0, E, (S,), dtype=torch.int32, device=dev)
    wts = torch.rand((S,), dtype=torch.float32, device=dev)
    act = torch.empty((S, Ip), dtype=torch.float16, device=dev)
    dn = torch.empty((S, H), dtype=torch.float16, device=dev)
    out = torch.empty((M, H), dtype=torch.float16, device=dev)

    def unfused():
        # verbatim the shipped indexed path, including the dead zeros the branch never reads
        _ = torch.zeros((M, H), dtype=torch.float16, device=dev)
        i32 = ids.reshape(-1).to(torch.int32)
        xg = x.unsqueeze(1).expand(M, top_k, H).reshape(S, H).contiguous()
        gu = torch.empty((S, 2 * Ip), dtype=torch.float16, device=dev)
        getattr(torch.ops.pxq4, TIER_OPS[t13]["moe_mmv"])(gu, xg, i32, w13_s, w13_a)
        ac = torch.nn.functional.silu(gu[:, :Ip]) * gu[:, Ip:]
        d = torch.empty((S, H), dtype=torch.float16, device=dev)
        getattr(torch.ops.pxq4, TIER_OPS[t2]["moe_mmv"])(d, ac.contiguous(), i32, w2_s, w2_a)
        w = wts.to(torch.float32).reshape(M, top_k, 1)
        return (d.view(M, top_k, H).to(torch.float32) * w).sum(dim=1).to(torch.float16)

    def fused_a():
        getattr(torch.ops.pxq4, TIER_OPS[t13]["gateup"])(act, x, ids, w13_s, w13_a, top_k)
        getattr(torch.ops.pxq4, TIER_OPS[t2]["down_fold"])(out, act, ids, wts, w2_s, w2_a, top_k)

    def fused_b():
        getattr(torch.ops.pxq4, TIER_OPS[t13]["gateup"])(act, x, ids, w13_s, w13_a, top_k)
        getattr(torch.ops.pxq4, TIER_OPS[t2]["down_part"])(dn, act, ids, w2_s, w2_a)
        torch.ops.pxq4.moe_slot_fold_out(out, dn, wts, top_k)

    nu, na, nb = count(unfused), count(fused_a), count(fused_b)
    L = int(cfg.get("text_config", cfg).get("num_hidden_layers", 0))
    print(f"\nCUDA kernel launches for ONE MoE layer at M={M} top_k={top_k} (H={H} Ip={Ip}):")
    print(f"  unfused (shipped indexed path) : {nu}")
    print(f"  fused form A (2 ops)           : {na}   -> {nu - na} fewer per layer")
    print(f"  fused form B (3 ops)           : {nb}   -> {nu - nb} fewer per layer")
    if L:
        print(f"  over {L} layers: form A saves {(nu - na) * L} launches per decode step, "
              f"form B saves {(nu - nb) * L}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
