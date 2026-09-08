#!/usr/bin/env python3
"""Part C of the reduction-order probe: pick the kernel's actual fold.

Part A established that torch's fp32 accumulation order for x.pow(2).mean(-1)
is SHAPE-DEPENDENT -- layouts exist that reproduce it at 1x2048, 2x2048, 4x2048,
16x256, 32x128, 512x2048 and 2048x2048, but no single layout reproduces it at
all of them and none at all was found for 8x2048. Bit-identity with torch across
the decode ladder AND arbitrary prefill widths is therefore not on the table.

So the question becomes the one that decides the default: for the layouts a real
one-block-per-row kernel could plausibly use, how far from torch's variance do
they land, and how many fp16 elements of the FINAL normed output actually
change? That is measured here on the real shapes, against the exact native
GemmaRMSNorm chain, with activation magnitudes in the range a served model
produces.
"""
import numpy as np
import torch

F32 = np.float32


def _seq_rows(a):
    return np.cumsum(np.asarray(a, dtype=F32), axis=-1, dtype=F32)[..., -1]


def _tree_pairwise_rows(p):
    """(0+1)+(2+3) pairing along the last axis; the warp-shuffle combine."""
    p = np.array(p, dtype=F32)
    n = p.shape[-1]
    off = 1
    while off < n:
        q = p.copy()
        q[..., : n - off] = (p[..., : n - off] + p[..., off:]).astype(F32)
        p = q
        off <<= 1
    return p[..., 0]


def fold(sq, layout):
    """sq: [R, H] float32 squares. Returns [R] float32 row sums."""
    kind, bw, vec = layout
    R, H = sq.shape
    if kind == "strided":
        # thread t: sq[t], sq[t+bw], ... sequential; then pairwise tree over bw
        a = sq.reshape(R, -1, bw)                    # [R, k, bw]
        part = _seq_rows(np.moveaxis(a, 1, -1))      # [R, bw]
        return _tree_pairwise_rows(part)
    if kind == "vec":
        # thread t: vec-wide groups, vec accumulators, merged sequentially
        a = sq.reshape(R, -1, bw, vec)               # [R, k, bw, vec]
        acc = _seq_rows(np.moveaxis(a, 1, -1))       # [R, bw, vec]
        part = _seq_rows(acc)                        # [R, bw]
        return _tree_pairwise_rows(part)
    if kind == "canon":
        # PXQ_CANON_v1 discipline: fixed contiguous chunks, then a flat fold.
        # Independent of block geometry, so identical at every launch config.
        a = sq.reshape(R, -1, bw)                    # [R, nchunk, bw]
        return _seq_rows(_seq_rows(a))
    raise ValueError(kind)


LAYOUTS = [
    ("strided", 256, 1), ("strided", 512, 1), ("strided", 1024, 1),
    ("vec", 256, 4), ("vec", 256, 8), ("vec", 128, 8), ("vec", 64, 8),
    ("canon", 64, 1), ("canon", 128, 1),
]


def ulp_dist(a, b):
    ia = np.asarray(a, dtype=F32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, dtype=F32).view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0),
          "torch:", torch.__version__, flush=True)
    eps = 1e-6
    g = torch.Generator(device="cuda").manual_seed(7)
    shapes = [(1, 2048), (2, 2048), (4, 2048), (8, 2048),
              (16, 256), (32, 128), (512, 2048), (4096, 2048)]

    print(f"\n{'shape':>12} {'layout':>16} {'max var ULP':>12} "
          f"{'rsqrt differs':>14} {'fp16 out differs':>20} {'ppm':>8}")
    summary = {}
    for (M, H) in shapes:
        x = (torch.randn(M, H, generator=g, device="cuda",
                         dtype=torch.float32) * 0.05).half()
        res = torch.randn(M, H, generator=g, device="cuda",
                          dtype=torch.float32) * 0.05
        w = torch.randn(H, generator=g, device="cuda",
                        dtype=torch.float32) * 0.02

        # exact native GemmaRMSNorm chain
        weight = w.float() + 1.0
        xf = x.float() + res
        var_t = xf.pow(2).mean(dim=-1, keepdim=True)
        out_t = ((xf * torch.rsqrt(var_t + eps)) * weight).to(torch.float16)

        sq = (xf.cpu().numpy().astype(F32) ** 2).astype(F32)
        for layout in LAYOUTS:
            kind, bw, vec = layout
            if H % (bw * vec):
                continue
            s = fold(sq, layout)
            var_c = (s * F32(1.0 / H)).astype(F32)
            u = ulp_dist(var_c, var_t.squeeze(-1).cpu().numpy().astype(F32))
            vt = torch.from_numpy(var_c).cuda().view(M, 1)
            rs_t = torch.rsqrt(var_t + eps)
            rs_c = torch.rsqrt(vt + eps)
            rdiff = int((rs_t != rs_c).sum().item())
            out_c = ((xf * rs_c) * weight).to(torch.float16)
            d = int((out_c != out_t).sum().item())
            n = out_t.numel()
            print(f"{M:6d}x{H:<5d} {kind+str(bw)+'x'+str(vec):>16} "
                  f"{int(u.max()):12d} {rdiff:>7d} / {M:<4d} "
                  f"{d:>12d} / {n:<6d} {1e6*d/n:8.1f}", flush=True)
            summary.setdefault(layout, []).append((M, H, int(u.max()), d, n))

    print("\nPER-LAYOUT TOTALS (all shapes pooled):")
    for layout, rows in summary.items():
        tot_d = sum(r[3] for r in rows)
        tot_n = sum(r[4] for r in rows)
        mx = max(r[2] for r in rows)
        print(f"  {str(layout):>26}  max var ULP {mx:3d}   fp16 differing "
              f"{tot_d:6d} / {tot_n:<9d}  = {1e6*tot_d/tot_n:7.1f} ppm")


if __name__ == "__main__":
    main()
