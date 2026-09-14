//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// PXA_FA_TILE_V2 — a re-SCHEDULED tile flash-attention kernel for the cards that have no
// tensor cores at all. DEFAULT OFF. The shipping kernel (ggml/src/ggml-cuda/fattn-tile-f16.cu,
// flash_attn_tile_ext_f16) is NOT modified by this file and remains the default path.
//
// WHAT THIS FILE CHANGES, AND WHAT IT DELIBERATELY DOES NOT.
//
// It changes WHERE THE BYTES SIT AND WHEN THEY MOVE. It does not change a single arithmetic
// operation, their order, or their types. Same 64-cell tile walk, same 64 kqsum slots, same
// per-tile rescale by exp(max_old - max_new), same accumulation order over k, same butterfly
// fold, same fp32 running state. That is not a stylistic preference: it is what makes this
// kernel checkable. The correctness bar for a schedule change is BITWISE EQUALITY with the
// kernel it replaces, which is a far stronger and far cheaper claim than any tolerance, and it
// inherits the placement-invariance property that fp32 accumulation bought (see
// tests/test-fa-band-align.cpp and the PXA_FA_TILE_F32ACC note in pxa/pxa-enhance.cuh).
//
// THE RUNNING STATE IS fp32, UNCONDITIONALLY. There is no half-accumulator arm in this file.
// The tile walk groups keys by their CELL index in the shared KV ring, not by their position in
// the sequence, so half running state makes the rounding of a twenty-thousand-term accumulation
// a function of where a request's band was placed. That was measured on hardware and fixed; a
// new kernel does not get to re-open it, so the property is a compile-time fact here rather than
// a template parameter.
//
// WHAT WAS ACTUALLY SLOW. Instruction accounting on the shipping kernel at D = 128, ncols = 32,
// nwarps = 8 — the shape a 2048-token ubatch of a 128-head-dim model runs:
//
//   the QK inner loop is SHARED-MEMORY-LOAD BOUND, not FMA bound. Per k step a lane issues
//   2 LDS.32 for its two K rows and 4 LDS.32 (broadcast) for its four Q columns — six loads to
//   feed eight half2 FMAs. Forty-three percent of the hottest loop in the kernel is address
//   arithmetic and LDS.
//
// and the reason it cannot simply be widened is the shared-memory LAYOUT. The staging tile is
// declared `half2 KV_tmp[64][D/2 + 1]`: the `+ 1` is what makes the QK access pattern (32 lanes
// reading 32 DIFFERENT ROWS at the same column) bank-conflict free, because a row stride of
// D/2 + 1 half2 is odd and walks the banks. But that same odd stride means a row does not start
// on a 16-byte boundary, so a 128-bit load is not merely slower, it is not expressible.
//
// THE FIX: A PHASE XOR SWIZZLE INSTEAD OF A PAD. Drop the pad, keep the row stride a power of
// two, and permute WITHIN the row: row r stores its 16-byte chunk c at physical chunk
// c ^ (r mod R), where R = (D/2)/4 is the number of 16-byte chunks in a row. Because R is a
// multiple of 8 for every head size this kernel serves (D = 64, 128, 256 give R = 8, 16, 32),
// the row base contributes nothing to the bank-group index and the swizzle alone decides it:
//
//   bank group of (r, c) = (r*R + (c ^ (r mod R))) mod 8 = (c ^ r) mod 8.
//
//   * QK reads (lanes = rows, one chunk): eight consecutive lanes of an LDS.128 phase see eight
//     consecutive values of r, so (c ^ r) mod 8 takes all eight values — conflict free.
//   * PV reads (lanes = columns, one row): the 32 lanes cover eight consecutive chunks starting
//     on a multiple of eight, XOR'd by a constant — still eight distinct groups, four banks
//     each, all 32 banks — conflict free.
//   * The staging store (lanes = columns, one row) is the PV pattern — conflict free.
//
// So the swizzle is not a trade of one conflict for another; it is conflict free for all three
// patterns AND 16-byte addressable. With it, the QK loop reads K and Q as 128-bit chunks with k
// unrolled by four: twenty-four LDS.32 per four k steps becomes SIX LDS.128, for identical
// arithmetic in an identical order.
//
// WHAT IS NOT HERE, and why, so it is not mistaken for an oversight:
//   * No cp.async / ldmatrix. Neither exists before Ampere; the whole point of this schedule is
//     to reach the same overlap with manual staging.
//   * No 128-bit GLOBAL loads. They would cut the staging instruction count fourfold, but the
//     staging loop is ~6% of the memory instructions in a tile and a 16-byte global access
//     imposes an alignment precondition on a KV-cache view this kernel cannot prove. The
//     instruction saving does not pay for the risk; the coalescing is already perfect either way
//     (32 lanes x 4 B = one 128-byte transaction).
//   * No split-KV rewrite. This tree already merges split-KV partials in fp32:
//     `parallel_blocks` plus flash_attn_combine_results in fattn-common.cuh, whose meta is
//     float2 and whose merge is fp32. v2 keeps that machinery byte for byte.
//   * No GQA head packing yet. On a card with no tensor cores that is not a wider MMA operand;
//     it is "one block serves G query heads of a KV group so K/V is streamed once instead of G
//     times", which costs G times the Q staging and G times the VKQ registers. Register
//     pressure is the failure mode this campaign has measured repeatedly, so it waits for a
//     number from the schedule change alone.
//

#include "../common.cuh"
#include "../fattn-common.cuh"
#include "fattn-tile-v2.cuh"
#include "pxa-enhance.cuh"

#include <cfloat>

#define FATTN_KQ_STRIDE_TILE_V2 64

// THE 16-BYTE CHUNK IS A BUILT-IN TYPE, AND THAT IS NOT cosmetic. The first version of this file
// declared its own `struct __align__(16) { half2 v[4]; }` and read it through a cast from the
// half2 tile. It compiled, it was correct, and a cuobjdump census of the resulting object found
// ZERO LDS.128 in it: nvcc kept the alignment attribute but still scalarised every one of those
// loads into four LDS.32, so the whole premise of the file reached the source and not the machine
// code. The tile is therefore DECLARED as an array of int4 -- a built-in 16-byte type indexed
// directly, with no cast for the compiler to distrust -- and the scalar (PV and staging) accesses
// take a half2 view of the same storage. Same bytes, same swizzle; the difference is only whether
// ptxas is ever asked to prove an alignment it will not prove.

// Phase XOR swizzle, chunk granularity: logical 16-byte chunk `c` of row `r` lives at physical
// chunk c ^ (r mod R). R is a power of two >= 8 for every head size this kernel serves, so the
// permutation is within the row (never off the end of it) and is its own inverse.
template <int D>
static __device__ __forceinline__ int pxa_v2_chunk(const int r, const int c) {
    constexpr int R = (D/2)/4;
    static_assert(R >= 8 && (R & (R - 1)) == 0, "row must hold a power-of-two number of 16B chunks, at least 8");
    return c ^ (r & (R - 1));
}

// The same swizzle at half2 granularity, for the scalar (PV and staging) accesses.
template <int D>
static __device__ __forceinline__ int pxa_v2_h2(const int r, const int i) {
    return (pxa_v2_chunk<D>(r, i >> 2) << 2) | (i & 3);
}

template<int D, int ncols, int nwarps, int parallel_blocks, bool use_softcap, bool pxa_mask_skip> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_v2(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float softcap,
        const uint32_t n_head_log2,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#ifdef FP16_AVAILABLE
    // Skip unused kernel variants for faster compilation:
    if (use_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const int      stride_mask = nb31 / sizeof(half);
    const half   * maskh = (const half   *)  mask + stride_mask*ic0;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    constexpr int TILE = FATTN_KQ_STRIDE_TILE_V2;
    constexpr int D2   = D/2;           // half2 per row
    constexpr int R    = D2/4;          // 16-byte chunks per row

    __shared__ half KQ[ncols*TILE];
    half2 * KQ2 = (half2 *) KQ;

    // No pad. The row stride is a power of two and the phase XOR swizzle above does the work the
    // pad used to do — see the file header for the bank-group derivation. Declared as chunks so
    // the wide read is a plain indexed load of a 16-byte type; KV_h2 is the same storage seen as
    // half2 for the scalar accesses.
    __shared__ int4 KV_tmp[TILE][R];
    half2 * const KV_h2 = (half2 *) KV_tmp;

    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float2 kqsum[ncols/nwarps] = {{0.0f, 0.0f}};

    float2 VKQ[ncols/nwarps][D2/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and stage it. Q rows are D2 half2 = a whole number of 16-byte chunks and
    // every lane of a warp reads the SAME Q row (a broadcast), so Q needs no swizzle to be both
    // conflict free and 128-bit addressable.
    __shared__ int4 Q_tmp[ncols][R];
    half2 * const Q_h2 = (half2 *) Q_tmp;
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            Q_h2[j*D2 + i] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    __syncthreads();

    __shared__ float pxa_skip_smax[nwarps];

    const int k_start = parallel_blocks == 1 ? 0 : ip*TILE;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*TILE) {
        // PXA_FA_MASK_SKIP_TILE: skip KV tiles whose mask is entirely -inf. Bit-identical by
        // construction (sum == -inf -> kqmax unchanged -> exp(-inf - max) == 0, rescale == 1).
        if (pxa_mask_skip && mask) {
            const half2 * pxa_m2  = (const half2 *) maskh;
            const int pxa_stride2 = stride_mask/2;
            float pxa_max = -INFINITY;
            for (int idx = threadIdx.y*WARP_SIZE + threadIdx.x; idx < ncols*(TILE/2); idx += nwarps*WARP_SIZE) {
                const int j = idx / (TILE/2);
                const int k = idx % (TILE/2);
                const half2 v = pxa_m2[j*pxa_stride2 + k_VKQ_0/2 + k];
                pxa_max = fmaxf(pxa_max, fmaxf(__low2float(v), __high2float(v)));
            }
            pxa_max = warp_reduce_max(pxa_max);
            if (threadIdx.x == 0) {
                pxa_skip_smax[threadIdx.y] = pxa_max;
            }
            __syncthreads();
            float pxa_tile_max = pxa_skip_smax[0];
#pragma unroll
            for (int w = 1; w < nwarps; ++w) {
                pxa_tile_max = fmaxf(pxa_tile_max, pxa_skip_smax[w]);
            }
            __syncthreads(); // protect pxa_skip_smax before the next iteration overwrites it
            if (pxa_tile_max == -INFINITY) {
                continue;
            }
        }

        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        // Stage K. Lanes of a warp walk one row's columns, so this is the swizzle's PV-shaped
        // access: eight distinct chunk groups, four banks each, no conflict.
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < TILE; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                KV_h2[i_KQ*D2 + pxa_v2_h2<D>(i_KQ, k_KQ)] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
            }
        }

        __syncthreads();

        half2 sum2[TILE/WARP_SIZE][ncols/nwarps] = {{{0.0f, 0.0f}}};

        // THE HOT LOOP. k advances four at a time and every operand arrives as one 128-bit
        // shared-memory load. The four half2 inside a chunk are then consumed in ascending k
        // order into the same accumulator, so this is the shipping kernel's k loop with the
        // loads hoisted and widened — the FMA sequence per accumulator is unchanged.
#pragma unroll
        for (int kc = 0; kc < R; ++kc) {
            int4 K_k[TILE/WARP_SIZE];
            int4 Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < TILE; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][pxa_v2_chunk<D>(i_KQ, kc)];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_tmp[j_KQ][kc];
            }

#pragma unroll
            for (int s = 0; s < 4; ++s) {
#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < TILE; i_KQ_0 += WARP_SIZE) {
                    const half2 kk = ((const half2 *) &K_k[i_KQ_0/WARP_SIZE])[s];
#pragma unroll
                    for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                        const half2 qq = ((const half2 *) &Q_k[j_KQ_0/nwarps])[s];
                        sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += kk*qq;
                    }
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < TILE; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                half sum;
                if (use_softcap) {
                    const float2 tmp = __half22float2(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                    sum = softcap * tanhf(tmp.x + tmp.y);
                } else {
                    sum = __low2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]) + __high2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }
                sum += mask ? slopeh*maskh[j_KQ*stride_mask + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], __half2float(sum));

                KQ[j_KQ*TILE + i_KQ] = sum;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < TILE/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 kq = __half22float2(KQ2[j*(TILE/2) + i]);
                const float vx = expf(kq.x - kqmax[j0/nwarps]);
                const float vy = expf(kq.y - kqmax[j0/nwarps]);
                kqsum[j0/nwarps].x = kqsum[j0/nwarps].x*KQ_max_scale + vx;
                kqsum[j0/nwarps].y = kqsum[j0/nwarps].y*KQ_max_scale + vy;
                KQ2[j*(TILE/2) + i] = make_half2(vx, vy);
            }

#pragma unroll
            for (int i0 = 0; i0 < D2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/WARP_SIZE].y *= KQ_max_scale;
            }
        }

        __syncthreads();

        // Stage V into the same tile the K rows just vacated.
#pragma unroll
        for (int k0 = 0; k0 < TILE; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_h2[k*D2 + pxa_v2_h2<D>(k, i)] = V_h2[(k_VKQ_0 + k)*stride_KV2 + i];
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < TILE; k0 += 2) {
            half2  V_k[D2/WARP_SIZE][2];
            half2 KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE][0] = KV_h2[(k0 + 0)*D2 + pxa_v2_h2<D>(k0 + 0, i)];
                V_k[i0/WARP_SIZE][1] = KV_h2[(k0 + 1)*D2 + pxa_v2_h2<D>(k0 + 1, i)];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ2[j*(TILE/2) + k0/2];
            }

#pragma unroll
            for (int i0 = 0; i0 < D2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    const float2 v0  = __half22float2(V_k[i0/WARP_SIZE][0]);
                    const float2 v1  = __half22float2(V_k[i0/WARP_SIZE][1]);
                    const float  kql =  __low2float(KQ_k[j0/nwarps]);
                    const float  kqh = __high2float(KQ_k[j0/nwarps]);
                    VKQ[j0/nwarps][i0/WARP_SIZE].x += v0.x*kql + v1.x*kqh;
                    VKQ[j0/nwarps][i0/WARP_SIZE].y += v0.y*kql + v1.y*kqh;
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float kqsum_j = kqsum[j_VKQ_0/nwarps].x + kqsum[j_VKQ_0/nwarps].y;
        kqsum_j = warp_reduce_sum(kqsum_j);

        // Sinks (launcher forces parallel_blocks == 1 when they are present).
        if (sinks) {
            const float sink      = ((const float *) sinks)[blockIdx.y];
            const float kqmax_old = kqmax[j_VKQ_0/nwarps];
            const float kqmax_n   = fmaxf(kqmax_old, sink);
            const float scale     = expf(kqmax_old - kqmax_n);
            kqsum_j = kqsum_j*scale + expf(sink - kqmax_n);
#pragma unroll
            for (int i0 = 0; i0 < D2/WARP_SIZE; ++i0) {
                VKQ[j_VKQ_0/nwarps][i0].x *= scale;
                VKQ[j_VKQ_0/nwarps][i0].y *= scale;
            }
        }

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            float2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (parallel_blocks == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 0] = dst_val.x;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 1] = dst_val.y;
        }

        if (parallel_blocks != 1 && threadIdx.x == 0) {
            dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int cols_per_block, int parallel_blocks, bool use_softcap, bool pxa_mask_skip>
static void launch_fattn_tile_v2_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int      D = 64;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_v2<D, cols_per_block, nwarps, parallel_blocks, use_softcap, pxa_mask_skip>;
            launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        case 128: {
            constexpr int      D = 128;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_v2<D, cols_per_block, nwarps, parallel_blocks, use_softcap, pxa_mask_skip>;
            launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        case 256: {
            // Same 48KB static-smem constraint as the shipping kernel, one chunk lighter because
            // the pad is gone: KQ 2KB + KV_tmp 64x32 int4 = 32KB + Q_tmp 8KB = 42KB at ncols=16.
            if constexpr (cols_per_block <= 16) {
                constexpr int      D = 256;
                constexpr int nwarps = 8;
                fattn_kernel_t fattn_kernel = flash_attn_tile_ext_v2<D, cols_per_block, nwarps, parallel_blocks, use_softcap, pxa_mask_skip>;
                launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
            } else {
                GGML_ABORT("tile-v2 D=256 requires cols_per_block <= 16 (48KB static smem)");
            }
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64, 128 and 256.");
        } break;
    }
}

template <int cols_per_block, int parallel_blocks, bool use_softcap>
static void launch_fattn_tile_v2_skip_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "PXA_FA_TILE_V2: ENGAGED (swizzled smem, 128-bit LDS in the QK loop, fp32 running state)\n");
    }
    if (pxa_fa_mask_skip_tile()) {
        launch_fattn_tile_v2_64_128<cols_per_block, parallel_blocks, use_softcap, true >(ctx, dst);
    } else {
        launch_fattn_tile_v2_64_128<cols_per_block, parallel_blocks, use_softcap, false>(ctx, dst);
    }
}

bool ggml_cuda_fattn_tile_v2_armed() {
    return pxa_fa_tile_v2();
}

void ggml_cuda_flash_attn_ext_tile_v2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    float softcap;
    memcpy(&softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const bool has_sinks = dst->src[4] != nullptr;

    if (Q->ne[0] == 256) {
        constexpr int cols_per_block = 16;
        if (Q->ne[1] <= 16 && !has_sinks) {
            constexpr int parallel_blocks = 4;
            if (softcap == 0.0f) {
                launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, false>(ctx, dst);
            } else {
                launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, true>(ctx, dst);
            }
        } else {
            constexpr int parallel_blocks = 1;
            if (softcap == 0.0f) {
                launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, false>(ctx, dst);
            } else {
                launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, true>(ctx, dst);
            }
        }
        return;
    }

    if (Q->ne[1] <= 16 && !has_sinks) {
        constexpr int cols_per_block = 16;
        constexpr int parallel_blocks = 4;
        if (softcap == 0.0f) {
            launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, false>(ctx, dst);
        } else {
            launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, true>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 32 && !has_sinks) {
        constexpr int cols_per_block = 32;
        constexpr int parallel_blocks = 4;
        if (softcap == 0.0f) {
            launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, false>(ctx, dst);
        } else {
            launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, true>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int parallel_blocks = 1;
    if (softcap == 0.0f) {
        launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, false>(ctx, dst);
    } else {
        launch_fattn_tile_v2_skip_dispatch<cols_per_block, parallel_blocks, true>(ctx, dst);
    }
}

bool ggml_cuda_fattn_tile_v2_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->ne[0] != V->ne[0]) {
        return false;
    }
    // The chunked F16 K/V route (PXA_FA_F16_KV_CHUNK) is a property of the shipping kernel's
    // launcher pairing; v2 supplies no chunked twin, so it declines whenever that route is armed
    // rather than silently changing which schedule a chunked run gets.
    if (pxa_fa_f16_kv_chunk() > 0) {
        return false;
    }
    return K->ne[0] == 64 || K->ne[0] == 128 || K->ne[0] == 256;
}
