#!/usr/bin/env python3
"""Part D: is the decode-shape zero-divergence result robust, or one lucky draw?

Part C found that at every decode shape a one-block-per-row fold differs from
torch's variance by at most 1 fp32 ULP and yet produces a BIT-IDENTICAL fp16
output, because rsqrtf's 2-ULP slack absorbs the difference before the fp16
round ever sees it. That was one draw per shape. This repeats it over many
seeds and over the three norm widths the model actually uses, and it also
sweeps M past the capture ladder to find where divergence starts, so the
supports_args cut can be placed on a measured boundary instead of a guess.
"""
import numpy as np
import torch

F32 = np.float32


def _seq_rows(a):
    return np.cumsum(np.asarray(a, dtype=F32), axis=-1, dtype=F32)[..., -1]


def _tree_pairwise_rows(p):
    p = np.array(p, dtype=F32)
    n = p.shape[-1]
    off = 1
    while off < n:
        q = p.copy()
        q[..., : n - off] = (p[..., : n - off] + p[..., off:]).astype(F32)
        p = q
        off <<= 1
    return p[..., 0]


def fold_vec(sq, bw, vec):
    """The kernel's fold: block of bw threads per row, each thread holding vec
    lane accumulators over a vec-wide strided walk, lanes merged in index
    order, then the offset-doubling shuffle tree over the block."""
    R, H = sq.shape
    a = sq.reshape(R, -1, bw, vec)
    acc = _seq_rows(np.moveaxis(a, 1, -1))
    return _tree_pairwise_rows(_seq_rows(acc))


def plan(H):
    """blockDim and vector width the kernel would choose for this width."""
    for vec in (8, 4, 2, 1):
        if H % vec == 0 and (H // vec) <= 1024:
            return H // vec, vec
    return 1024, H // 1024


def main():
    print("device:", torch.cuda.get_device_name(0), "torch:", torch.__version__,
          flush=True)
    eps = 1e-6
    widths = [2048, 256, 128]
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
    NSEED = 12
    print(f"\n{'H':>6} {'M':>6} {'bw x vec':>10} {'seeds':>6} "
          f"{'max var ULP':>12} {'fp16 elems differing':>22} {'ppm':>9}")
    for H in widths:
        bw, vec = plan(H)
        for M in Ms:
            tot_d = tot_n = 0
            mx = 0
            for s in range(NSEED):
                g = torch.Generator(device="cuda").manual_seed(1000 * s + M + H)
                x = (torch.randn(M, H, generator=g, device="cuda",
                                 dtype=torch.float32) * 0.05).half()
                res = torch.randn(M, H, generator=g, device="cuda",
                                  dtype=torch.float32) * 0.05
                w = torch.randn(H, generator=g, device="cuda",
                                dtype=torch.float32) * 0.02
                weight = w.float() + 1.0
                xf = x.float() + res
                var_t = xf.pow(2).mean(dim=-1, keepdim=True)
                out_t = ((xf * torch.rsqrt(var_t + eps)) * weight).to(torch.float16)

                sq = (xf.cpu().numpy().astype(F32) ** 2).astype(F32)
                var_c = (fold_vec(sq, bw, vec) * F32(1.0 / H)).astype(F32)
                ia = var_c.view(np.int32).astype(np.int64)
                ib = (var_t.squeeze(-1).cpu().numpy().astype(F32)
                      .view(np.int32).astype(np.int64))
                mx = max(mx, int(np.abs(ia - ib).max()))
                vt = torch.from_numpy(var_c).cuda().view(M, 1)
                out_c = ((xf * torch.rsqrt(vt + eps)) * weight).to(torch.float16)
                tot_d += int((out_c != out_t).sum().item())
                tot_n += out_t.numel()
            flag = "" if tot_d == 0 else "   <-- diverges"
            print(f"{H:6d} {M:6d} {bw:6d}x{vec:<3d} {NSEED:6d} {mx:12d} "
                  f"{tot_d:>12d} / {tot_n:<8d} {1e6*tot_d/tot_n:8.2f}{flag}",
                  flush=True)


if __name__ == "__main__":
    main()
