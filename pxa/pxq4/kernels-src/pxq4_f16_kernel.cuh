// pxq4_f16_kernel.cuh — plain-fp16 dense decode GEMV/multi-token GEMM for sm_60.
//
// WHY. On P100 the model's FP16 (unquantized) linears run through cuBLAS's
// gemv2T_kernel_val at decode, measured at 22.7% of the decode step's self-CUDA time —
// second only to the PXQ4 weight reads themselves — at a fraction of achievable
// bandwidth. The fork's own fast path (torch.ops._C.sm70_f16_gemm, TurboMind) is not
// compiled into the Pascal _C build and its prepare/gemm ops are absent at runtime.
// This kernel is the sm_60 replacement: pure streaming, no format decode, fp32
// accumulation, one warp per output row, lanes striding K with uint4 (8-half) loads so
// every warp transaction is a contiguous 512 B read of the weight row.
//
// MT: like k_pxq4_mmv_fused_mt, one weight read serves all M tokens (M <= 8): the lane
// keeps M fp32 accumulators and re-uses each loaded weight vector M times. x rows are
// read via __ldg (L1/L2 resident: x is [M,K] fp16, <= 80 KB).
//
// NUMERICS. fp32 accumulation, lane-order (k ascending in strides of 32*8), then a
// shuffle reduction (offset descending 16,8,4,2,1), one __float2half_rn at the end.
// NOT bit-identical to cuBLAS (different association) — gated by relative-error parity
// against torch.mm and by the serving 391/greedy gates, same policy as every other
// non-vendored kernel here.
//
// Shapes: K % 8 == 0 (uint4 loads), any N (tail rows exit), M 1..8 dispatched to an
// exact template so the accumulator array unrolls.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define PXA_F16_ROWS_PER_BLOCK 8   // 8 warps of 32 lanes = 256 threads

template <int MT>
static __global__ void __launch_bounds__(256)
k_pxa_f16_mmv_mt(const __half * __restrict__ w,   // [N, K] row-major
                 const __half * __restrict__ x,   // [MT, K]
                 __half       * __restrict__ out, // [MT, N]
                 const int N, const int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * PXA_F16_ROWS_PER_BLOCK + warp;
    if (row >= N) return;

    const __half * wr = w + (size_t)row * K;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;

    // lane strides 8 halfs; warp covers 32*8 = 256 halfs (512 B) per iteration, contiguous.
    for (int k = lane * 8; k < K; k += 32 * 8) {
        const uint4 wv = *(const uint4 *)(wr + k);
        const __half2 * wh = (const __half2 *)&wv;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint4 xv = __ldg((const uint4 *)(x + (size_t)m * K + k));
            const __half2 * xh = (const __half2 *)&xv;
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 wf = __half22float2(wh[i]);
                const float2 xf = __half22float2(xh[i]);
                s = __fmaf_rn(wf.x, xf.x, s);
                s = __fmaf_rn(wf.y, xf.y, s);
            }
            acc[m] += s;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            out[(size_t)m * N + row] = __float2half_rn(acc[m]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// v10: K-tiled smem-staged variant for M >= 4. The v9 kernel re-reads x via __ldg once
// per (warp, lane-iteration, token): at M=8 that is 8x the weight traffic again out of
// L2, which is why measured bandwidth sagged 552 -> 201 GB/s as M grew. Here each BLOCK
// stages an [MT, TILE] tile of x into shared memory once per K-tile; the inner loop is
// then a pure weight stream plus conflict-free 16B smem reads. Accumulation stays fp32
// per token, k ascending within a lane, warp shuffle reduce at the end — same numeric
// shape as v9 (association differs from cuBLAS; gated the same way).
// smem = MT * TILE * 2 B (16 KiB at MT=16, TILE=512). Requires K % 8 == 0; any N.
// ---------------------------------------------------------------------------------------------
#define PXA_F16_TILE 512

template <int MT>
static __global__ void __launch_bounds__(256)
k_pxa_f16_mmv_smem(const __half * __restrict__ w,   // [N, K]
                   const __half * __restrict__ x,   // [MT, K]
                   __half       * __restrict__ out, // [MT, N]
                   const int N, const int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * PXA_F16_ROWS_PER_BLOCK + warp;
    const bool rowok = row < N;

    __shared__ __half xs[MT * PXA_F16_TILE];

    const __half * wr = w + (size_t)(rowok ? row : 0) * K;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;

    for (int k0 = 0; k0 < K; k0 += PXA_F16_TILE) {
        const int tw = min(PXA_F16_TILE, K - k0);
        __syncthreads();
        // cooperative fill, 16 B per thread-step, coalesced along each token row
        for (int idx = threadIdx.x * 8; idx < MT * PXA_F16_TILE; idx += 256 * 8) {
            const int m = idx / PXA_F16_TILE;
            const int c = idx % PXA_F16_TILE;
            if (c < tw) {
                *(uint4 *)&xs[m * PXA_F16_TILE + c] =
                    *(const uint4 *)(x + (size_t)m * K + k0 + c);
            }
        }
        __syncthreads();
        if (rowok) {
            for (int k = lane * 8; k < tw; k += 32 * 8) {
                const uint4 wv = *(const uint4 *)(wr + k0 + k);
                const __half2 * wh = (const __half2 *)&wv;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const __half2 * xh = (const __half2 *)&xs[m * PXA_F16_TILE + k];
                    float s = 0.f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 wf = __half22float2(wh[i]);
                        const float2 xf = __half22float2(xh[i]);
                        s = __fmaf_rn(wf.x, xf.x, s);
                        s = __fmaf_rn(wf.y, xf.y, s);
                    }
                    acc[m] += s;
                }
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
    }
    if (lane == 0 && rowok) {
#pragma unroll
        for (int m = 0; m < MT; ++m) out[(size_t)m * N + row] = __float2half_rn(acc[m]);
    }
}
