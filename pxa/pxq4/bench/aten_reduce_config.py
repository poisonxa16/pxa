#!/usr/bin/env python3
"""Replicate ATen's reduction configuration instead of searching for it (main #1075(4)).

WHY THE FIRST PROBE FAILED AT ONE SHAPE. bench/reduce_probe.py searched 366 candidate
accumulation layouts and found that every real shape had a layout reproducing torch
bit-for-bit EXCEPT 8x2048, which matched none of them. That was not evidence that no layout
exists. It was evidence that my candidate space was missing the layout torch actually uses,
and reading at::native::setReduceConfig and ReduceConfig::block_x_reduce says exactly which
one:

  * block_x_reduce is a HYBRID tree whenever block_width exceeds the warp size. It first
    runs a HALVING reduction in shared memory from offset = dim_x/2 down to warpSize, and
    only then the offset-DOUBLING warp shuffle from offset = 1 up to warpSize. My candidates
    were pure-halving or pure-doubling and never the two spliced together.
  * At 8x2048 the config comes out with block_width = 64 > 32, which is precisely the case
    that takes the hybrid path. That is why that one shape had no match while 1x2048
    (block_width 32 or 512 depending on the height) had 466.

So the fold is computable. This module computes it, and thread_fold() reproduces the exact
order for a given shape so it can be checked against torch bit-for-bit rather than searched
for. If it holds at the decode shapes, the fused norm becomes bit-identical to the native
path on the captured ladder and the promotion argument stops depending on tolerances.

Modelled for the case this pack cares about: `x.pow(2).mean(dim=-1)` over a CONTIGUOUS
[M, H] float32 tensor, one reduced dimension, ndim 2, on a CUDA device with warp 32.
"""
import math

import numpy as np

F32 = np.float32
WARP = 32
MAX_NUM_THREADS = 512          # mnt_wrapper<float>::MAX_NUM_THREADS
VT0 = 4                        # gpu_reduce_kernel default; input_vec_size == vt0


def last_pow2(n):
    return 1 << (int(n).bit_length() - 1) if n > 0 else 0


def div_up(a, b):
    return -(-a // b)


class Config:
    """The fields of at::native::ReduceConfig that determine the ORDER."""

    def __init__(self, M, H, num_mp=56, max_threads_per_mp=2048):
        self.M, self.H = M, H
        num_outputs, inputs_per_output = M, H

        # 2-D contiguous, reducing the last (fastest) dimension: strides[0] == 4 <
        # strides[1] == 4H, so reduction_on_fastest_striding_dimension is true.
        dim0, dim1 = inputs_per_output, num_outputs
        self.vec = 1
        # fastest_moving_stride == sizeof(float), so vectorisation is considered.
        if dim0 > 128 and VT0 >= VT0:
            self.vec = VT0
            dim0 //= self.vec

        # set_block_dimension (output_vec_size == 1)
        mnt = MAX_NUM_THREADS
        d0p = last_pow2(dim0) if dim0 < mnt else mnt
        d1p = last_pow2(dim1) if dim1 < mnt else mnt
        bw = min(d0p, WARP)
        bh = min(d1p, mnt // bw)
        bw = min(d0p, mnt // bh)
        self.bw, self.bh = bw, bh
        self.num_threads = bw * bh

        self.step_input = 1
        self.input_mult = [0, 0, 0]

        # input_mult[0] = split_input(block_width)
        self.input_mult[0] = self.step_input
        self.step_input *= bw

        vpt = div_up(H, self.step_input)          # num_inputs is H, not H/vec
        warp_split_threshold = min(bh * 16, 256)
        self.split_across_warps = vpt >= warp_split_threshold
        if self.split_across_warps:
            self.input_mult[1] = self.step_input
            self.step_input *= bh

        # ctas_per_output: only when the y-split happened AND the work is large enough.
        self.ctas_per_output = 1
        if self.input_mult[1] != 0 and div_up(H, self.step_input) >= 256:
            blocks_per_sm = max_threads_per_mp // self.num_threads
            target = num_mp * blocks_per_sm
            grid = div_up(M, 1)                    # config.grid().x == num_outputs / step_output
            if grid <= target:
                c1 = div_up(target, grid)
                c2 = div_up(div_up(H, self.step_input), 16)
                c3 = div_up(div_up(H, self.step_input), 256)
                self.ctas_per_output = max(min(c1, c2), c3)
                if self.ctas_per_output > 1:
                    self.input_mult[2] = self.step_input
                    self.step_input *= self.ctas_per_output

        self.values_per_thread = div_up(H, self.step_input)

    def __repr__(self):
        return (f"[{self.M}x{self.H}] block=({self.bw},{self.bh}) threads={self.num_threads} "
                f"vec={self.vec} input_mult={self.input_mult} step_input={self.step_input} "
                f"vpt={self.values_per_thread} ctas={self.ctas_per_output} "
                f"x_tree={'hybrid' if self.bw > WARP else 'warp'} "
                f"y_reduce={'yes' if self.input_mult[1] else 'no'}")


# ------------------------------------------------------------------- the fold ---
def _seq(v):
    v = np.asarray(v, dtype=F32)
    return np.cumsum(v, dtype=F32)[-1] if v.size else F32(0.0)


def thread_partials(sq, cfg):
    """Per-thread accumulator after the vectorised strided walk.

    Mirrors ReduceConfig::input_idx() and the vectorised loop in thread_reduce_impl:
    idx = lane*input_mult[0] + warp*input_mult[1] (+ cta*input_mult[2]), stepping by
    step_input in VECTOR units, with `vec` independent accumulators merged in index order
    at the end.
    """
    H, vec, bw, bh = cfg.H, cfg.vec, cfg.bw, cfg.bh
    nvec = H // vec
    out = np.zeros((bh, bw), dtype=F32)
    for ty in range(bh):
        for tx in range(bw):
            idx = tx * cfg.input_mult[0] + ty * cfg.input_mult[1]
            acc = np.zeros(vec, dtype=F32)
            while idx * vec + vec - 1 < H:
                base = idx * vec
                for j in range(vec):
                    acc[j] = F32(acc[j] + sq[base + j])
                idx += cfg.step_input
            out[ty, tx] = _seq(acc)          # combine accumulators, in index order
    return out


def block_x_reduce(row, bw):
    """ReduceConfig::block_x_reduce -- HALVING in shared memory down to the warp size,
    then the offset-DOUBLING warp shuffle. The splice is the part every hand-written
    candidate gets wrong."""
    v = np.array(row, dtype=F32)
    dim_x = bw
    if dim_x > WARP:
        off = dim_x // 2
        while off >= WARP:
            for tx in range(off):
                if tx + off < bw:
                    v[tx] = F32(v[tx] + v[tx + off])
            off >>= 1
        dim_x = WARP
    off = 1
    while off < dim_x:
        nv = v.copy()
        for tx in range(dim_x):
            if tx + off < dim_x:
                nv[tx] = F32(v[tx] + v[tx + off])
        v = nv
        off <<= 1
    return v[0]


def block_y_reduce(col, bh):
    """ReduceConfig::block_y_reduce -- halving over threadIdx.y."""
    v = np.array(col, dtype=F32)
    off = bh // 2
    while off > 0:
        for ty in range(off):
            if ty + off < bh:
                v[ty] = F32(v[ty] + v[ty + off])
        off >>= 1
    return v[0]


def aten_row_sum(sq, cfg):
    """The complete fp32 sum of one row, in ATen's order."""
    parts = thread_partials(sq, cfg)                       # [bh, bw]
    xs = np.array([block_x_reduce(parts[ty], cfg.bw) for ty in range(cfg.bh)],
                  dtype=F32)
    return block_y_reduce(xs, cfg.bh) if cfg.input_mult[1] else xs[0]


def aten_mean(row_f32, cfg):
    return F32(aten_row_sum((row_f32 * row_f32).astype(F32), cfg) / F32(cfg.H))


if __name__ == "__main__":
    print("Computed ATen reduce configs for x.pow(2).mean(-1) on contiguous [M,H] fp32:\n")
    for H in (2048, 256, 128):
        for M in (1, 2, 4, 8, 16):
            print("  ", Config(M, H))
        print()
    print("The 8x2048 row is the one bench/reduce_probe.py could not match: block_width 64")
    print("exceeds the warp, so its x-tree is the halving/doubling HYBRID that no candidate")
    print("in that search contained.")
