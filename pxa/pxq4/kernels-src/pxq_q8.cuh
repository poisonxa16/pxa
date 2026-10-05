// pxq_q8.cuh -- int8 weights with a per-ROW fp16 scale: kernels for a quantized LM head.
//
// WHY THIS FORMAT AND NOT A PXQ TIER. The LM head is not shaped like a body matrix. It is one
// [V, H] tensor with V = 248,320 rows against H = 2048, it is read once per token in full, and
// it is the single largest item in the decode byte budget (970 MiB of 2.78 GiB/token on the
// 35B MoE). What it needs is the cheapest possible per-row scale and a layout vLLM's vocab
// loader can shard without help -- NOT the 64-row panel layout, whose vocab axis is panels
// rather than rows and which is exactly what makes a PXQ4 head an engine change.
//
//   weight        int8    [V, H]   row-major, one row per vocabulary entry
//   weight_scale  fp16    [V]      scale[n] = absmax(row n) / 127, symmetric, no zero point
//   dequant       w[n,k] = float(q[n,k]) * float(scale[n])
//
// Per-row rather than per-tensor is not a refinement, it is the whole difference between a
// usable head and a broken one: a single scale across 248,320 rows is set by the largest
// logit row in the vocabulary and quantizes every other row into a handful of levels. Per-row
// also shards for FREE -- the vocab axis is exactly the axis vLLM's vocab loader narrows, so
// the scale rides along with output_dim=0 and needs no custom loader.
//
// THE ARITHMETIC CONTRACT, fixed here because this is a new format and someone will need to
// reproduce it: accumulate float(q)*float(x) in fp32 across the whole row, THEN multiply by
// the row scale once, then round to fp16 exactly once.
//     out[m, n] = half( ( sum_k float(q[n,k]) * float(x[m,k]) ) * float(scale[n]) )
// Scaling once per row instead of once per element is not just cheaper, it is more accurate:
// it keeps the products integral until the single final scaling. The reduction order is a
// fixed lane-strided tree (see below) and is identical on every launch, so the kernel is
// deterministic; it is NOT bit-identical to a dequant-then-cuBLAS evaluation of the same
// weights, and it is not meant to be -- that path exists for prefill and is a different
// summation order. The self-test pins the GEMV against a host replay of THIS order.
//
// WHY A KERNEL AT ALL, since this is the question that decides whether the format is worth
// anything. At decode M = 1 and the op is a GEMV, so:
//     fp16 head                                  read 1017 MB per token
//     int8 head, this kernel                     read  509 MB per token   <- the win
//     int8 head, dequant to fp16 then cuBLAS     read  509 + write 1017 + read 1017 = 2543 MB
// Emitting an int8 head with no kernel behind it is a 2.5x REGRESSION, not a saving. Above
// the routing threshold the third line stops being absurd -- the dequant amortises over the
// batch -- which is why q8_linear_out routes on M exactly as pxq linear_out does.
//
// LAYOUT CHOICE: ONE WARP PER OUTPUT ROW, lanes strided along K. The obvious alternative --
// one thread per row, mirroring the PXQ mmv -- has 32 threads of a warp reading addresses K
// bytes apart, which on a 2048-wide row is 32 separate sectors per load instruction. Here the
// 32 lanes of a warp read 32 consecutive char4s: 128 contiguous bytes per instruction, one
// sector, fully coalesced. There is no panel structure to respect and no sub-scale to fetch,
// so the PXQ block shape buys nothing and costs the coalescing.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define PXQ_Q8_WARPS_PER_BLOCK 8      // 256 threads
#define PXQ_Q8_LANES           32

// out[M, N] = x[M, K] @ W[N, K]^T with W int8 + per-row fp16 scale.
// grid = (ceil(N / WARPS_PER_BLOCK), M), block = WARPS_PER_BLOCK * 32.
// No shared memory and no barriers: every warp is independent, which is also why the early
// return on an out-of-range row is safe here and would not be in the PXQ mmv.
static __global__ void __launch_bounds__(PXQ_Q8_WARPS_PER_BLOCK * PXQ_Q8_LANES)
k_pxq_q8_mmv(const int8_t * __restrict__ W,
             const __half * __restrict__ scale,     // [N]
             const __half * __restrict__ x,         // [M, K]
             __half       * __restrict__ out,       // [M, N]
             const int N, const int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * PXQ_Q8_WARPS_PER_BLOCK + warp;
    if (row >= N) return;

    const int8_t * w  = W + (size_t)row * K;
    const __half * xt = x + (size_t)blockIdx.y * K;

    float acc = 0.f;
    // Each lane takes 4 consecutive int8 values; the warp covers 128 per iteration. The tail
    // is handled scalar-wise rather than by requiring K % 128 == 0: a head's H is 2048 today
    // but a padded vocab or a different model should not silently produce wrong logits.
    const int kmain = (K / 128) * 128;
    for (int k = lane * 4; k < kmain; k += PXQ_Q8_LANES * 4) {
        // 4 B per lane, 128 contiguous bytes per warp instruction.
        const char4 q = *(const char4 *)(w + k);
        // x is fp16; two half2 loads keep this 8-byte aligned for every k (k is a multiple
        // of 4 by construction).
        const __half2 x01 = *(const __half2 *)(xt + k);
        const __half2 x23 = *(const __half2 *)(xt + k + 2);
        const float2  f01 = __half22float2(x01);
        const float2  f23 = __half22float2(x23);
        acc += (float)q.x * f01.x;
        acc += (float)q.y * f01.y;
        acc += (float)q.z * f23.x;
        acc += (float)q.w * f23.y;
    }
    for (int k = kmain + lane; k < K; k += PXQ_Q8_LANES) {
        acc += (float)w[k] * __half2float(xt[k]);
    }

    // Fixed reduction tree: identical on every launch, so the kernel is deterministic.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        // ONE scaling, ONE rounding, both after the whole row is summed.
        out[(size_t)blockIdx.y * N + row] = __float2half_rn(acc * __half2float(scale[row]));
    }
}

// Full dequant to fp16 [N, K], for the prefill route (then cuBLAS). One warp per row again,
// so both the read and the write are contiguous.
static __global__ void k_pxq_q8_dequant(const int8_t * __restrict__ W,
                                        const __half * __restrict__ scale,
                                        __half * __restrict__ y,
                                        const int N, const int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * PXQ_Q8_WARPS_PER_BLOCK + warp;
    if (row >= N) return;
    const int8_t * w = W + (size_t)row * K;
    __half       * o = y + (size_t)row * K;
    const float    s = __half2float(scale[row]);
    for (int k = lane; k < K; k += PXQ_Q8_LANES) {
        o[k] = __float2half_rn((float)w[k] * s);
    }
}
