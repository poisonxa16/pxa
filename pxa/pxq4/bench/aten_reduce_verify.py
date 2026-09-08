#!/usr/bin/env python3
"""Does the COMPUTED ATen reduce config reproduce torch bit-for-bit?

bench/reduce_probe.py searched and failed at one shape. This does not search: it reads the
configuration out of at::native::setReduceConfig, replays that exact accumulation order on
the host in float32, and compares the bits. A pass here means the fused norm can be made
bit-identical to the native path at the decode shapes, which turns the promotion argument
from "trust the tolerance" into "it is the same number".
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aten_reduce_config import Config, aten_row_sum  # noqa: E402

F32 = np.float32


def main():
    dev = torch.cuda.get_device_properties(0)
    print(f"device: {dev.name}  SMs: {dev.multi_processor_count}  torch: {torch.__version__}")
    print(f"{'shape':>12} {'block':>10} {'vec':>4} {'xtree':>7} "
          f"{'rows exact':>12} {'max ULP':>8}")
    allok = True
    for H in (2048, 256, 128):
        for M in (1, 2, 4, 8, 16):
            cfg = Config(M, H, num_mp=dev.multi_processor_count,
                         max_threads_per_mp=dev.max_threads_per_multi_processor)
            ok = 0
            mx = 0
            n = 0
            for seed in range(6):
                g = torch.Generator(device="cuda").manual_seed(seed * 97 + M + H)
                x = torch.randn(M, H, generator=g, device="cuda", dtype=torch.float32)
                ref = x.pow(2).mean(dim=-1).cpu().numpy().astype(F32)
                xc = x.cpu().numpy().astype(F32)
                for r in range(M):
                    sq = (xc[r] * xc[r]).astype(F32)
                    got = F32(aten_row_sum(sq, cfg) / F32(H))
                    a = np.asarray(got, dtype=F32).view(np.int32).astype(np.int64)
                    b = np.asarray(ref[r], dtype=F32).view(np.int32).astype(np.int64)
                    d = int(abs(a - b))
                    mx = max(mx, d)
                    ok += (d == 0)
                    n += 1
            print(f"{M:5d}x{H:<6d} ({cfg.bw:3d},{cfg.bh:3d}) {cfg.vec:4d} "
                  f"{'hybrid' if cfg.bw > 32 else 'warp':>7} {ok:6d}/{n:<5d} {mx:8d}")
            allok &= (ok == n)
    print("\n" + ("ALL SHAPES BIT-EXACT -- the config is replicable and the fused norm "
                  "can be made bit-identical" if allok else
                  "NOT all shapes match; the model is incomplete, see the ULP column"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
