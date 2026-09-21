# SPDX-License-Identifier: Apache-2.0
"""smallm2 (2026-09-07): PXQ4 small-M verify gate for the vLLM engine.

Two questions, one process each (every routing switch in the library is a
``static`` read ONCE, so an arm is a process, never a call):

  IDENTITY  is the arm's [M, N] output equal, row for row, to the SAME arm run
            one token at a time?  That is the per-token path the spec names.
            Bitwise for any SIMT arm; the HMMA arm cannot be and does not claim
            it (pxq4_mma.cu header), so it is reported as ULP distance instead.
  SPEED     per-layer-call microseconds at M = 1,2,4,8,16 on the four real
            rank-local Qwen3.8-27B TP2 shapes, and the ratio to M=1.

  ARM=new|mt|mma|old  python3 smallm_verify.py [identity|bench|both]
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("PXQ4_SITE", "/work/sidecar/site-union"))
from pxq4_vllm import ops as pxops  # noqa: E402

PANEL, SLABC, SLABB = 64, 32, 1088

# Rank-local TP2 shapes of Qwen3.8-27B, exactly the call mix upstream's M=5 AWQ
# contract lists (docs/design/sm70_awq_small_n_hmma_operator.md).
SHAPES = [
    ("gate_up", 17408, 5120),
    ("down",     5120, 8704),
    ("qkv_z",    8192, 5120),
    ("out_proj", 5120, 3072),
]
MS = [1, 2, 4, 8, 16]


def pxq4_pair(N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randint(0, 256, (N // PANEL, K // SLABC, SLABB), dtype=torch.uint8,
                      device="cuda", generator=g)
    a = (torch.randn(N // PANEL, PANEL, device="cuda", generator=g) * 0.02).half()
    return s.contiguous(), a.contiguous()


def call(out, x, s, a, op):
    getattr(torch.ops.pxq4, op)(out, x, s, a)


def timed_us(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    arm = os.environ.get("ARM", "new")
    op = os.environ.get("PXQ4_OP", "mmv_out")
    if not pxops.load_library(required=True):
        raise SystemExit("pxq4 library did not load")
    dev = torch.cuda.get_device_name(0)
    print(f"# arm={arm} op={op} dev={dev} lib={os.environ.get('PXQ4_LIB')}", flush=True)
    print(f"# PXA_PXQ4_SMALLM={os.environ.get('PXA_PXQ4_SMALLM','1')} "
          f"PXQ4_MMV_MMA={os.environ.get('PXQ4_MMV_MMA','<default>')} "
          f"PXQ4_MMV_MMA_MIN_M={os.environ.get('PXQ4_MMV_MMA_MIN_M','<default>')} "
          f"PXQ4_MMV_MT={os.environ.get('PXQ4_MMV_MT','<default>')}", flush=True)

    for name, N, K in SHAPES:
        s, a = pxq4_pair(N, K, seed=hash(name) & 0xffff)
        xg = torch.Generator(device="cuda").manual_seed(1234)
        x16 = (torch.randn(16, K, device="cuda", generator=xg) * 0.5).half().contiguous()

        # WARM THE ARENA AT THE CEILING FIRST. The partials arena refuses to grow
        # once frozen/sealed and vLLM warms largest-M-first for the same reason;
        # a bench that warmed at M=1 would measure a different allocation path.
        ow = torch.empty((16, N), dtype=torch.float16, device="cuda")
        call(ow, x16, s, a, op)
        torch.cuda.synchronize()

        if what in ("identity", "both"):
            for M in MS:
                x = x16[:M].contiguous()
                out = torch.empty((M, N), dtype=torch.float16, device="cuda")
                call(out, x, s, a, op)
                ref = torch.empty((M, N), dtype=torch.float16, device="cuda")
                for m in range(M):
                    r = torch.empty((1, N), dtype=torch.float16, device="cuda")
                    call(r, x[m:m + 1].contiguous(), s, a, op)
                    ref[m] = r[0]
                torch.cuda.synchronize()
                eq = int(torch.equal(out, ref))
                du = (out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
                d = (out.float() - ref.float()).abs()
                print(f"IDENT {name:8s} M={M:2d} bitwise={'YES' if eq else 'NO '} "
                      f"maxulp={int(du.max()):d} maxabs={float(d.max()):.3e} "
                      f"relL2={float(d.norm() / ref.float().norm()):.3e}", flush=True)

        if what in ("bench", "both"):
            base = None
            for M in MS:
                x = x16[:M].contiguous()
                out = torch.empty((M, N), dtype=torch.float16, device="cuda")
                us = timed_us(lambda: call(out, x, s, a, op))
                if M == 1:
                    base = us
                print(f"BENCH {name:8s} M={M:2d} {us:8.1f} us  ratio={us / base:5.2f}x",
                      flush=True)
        del s, a, x16, ow
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
