//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// PXA_FA_TILE_512 (2026-09-13) — flash-attention for head sizes 512/512 and 576/512 on the
// cards that have no Turing-class MMA: sm_60 (P100) and sm_70 (V100).
//
// WHAT WAS MISSING. The no-tensor-core tile kernels in this tree stop at head size 256
// (fattn-tile-f16.cu serves 64/128/256, fattn-tile-f32.cu 64/128), and the new-MMA kernel that
// does carry 512/512 and 576/512 needs Turing. ggml_cuda_fattn_is_supported() therefore declines
// a 512-wide attention node on a P100 or a V100, and the graph builder falls back to the unfused
// KQ / soft_max / KQV path: three passes over the KV extent, a materialised ncols x n_kv score
// matrix in HBM, and no online softmax. Two models on this box want that head size —
//   * Gemma 4: the 8 GLOBAL layers of 48 are head_dim 512 (key_length = value_length = 512,
//     head_count 16, head_count_kv 1, so GQA 16); the other 40 are the 256-wide sliding-window
//     layers the D=256 tile already serves.
//   * GLM-5.3 / DeepSeek-class MLA: DKQ 576, DV 512, V a 512-wide view of the same rows as K.
//
// WHY A SEPARATE FILE RATHER THAN A CASE IN fattn-tile-f16.cu. Two reasons, one structural and
// one arithmetic.
//   (1) The shipped tile kernel stages a WHOLE K or V row per KV position in shared memory
//       (KV_tmp[64][D/2 + 1] half2). At D = 512 that tile alone is 65,792 B and cannot be
//       compiled for a 48 KB static-smem budget, let alone co-resident with Q. This kernel
//       CHUNKS the head dimension: only nbatch_K = 64 head elements are staged at a time, and
//       the QK loop and the P.V loop each walk the head dimension in DKQ/64 (resp. DV/64) passes
//       over the same 8,448 B tile. Static smem at 512/512, ncols 16 is 28,928 B, so two blocks
//       still fit per SM. No extra global traffic: each pass reads a DISJOINT slice of the same
//       rows, so every K and V byte is still read exactly once per KV tile.
//   (2) 512 terms of fp16 accumulation is not the same proposition as 128. The shipped kernel
//       accumulates the QK dot product in half2 across the whole head dimension; at D = 512 that
//       is four times the terms, and this box has already paid once for exactly that kind of
//       too-coarse accumulation (PXA_FA_TILE_F32ACC, kvfix-fa-tile-f32acc: fp16 running softmax
//       state on sm_60 flipped decode tokens, and moving the state to fp32 was a measured win).
//       Here the head-dim chunking makes the fix free: each 64-element chunk is accumulated in
//       half2 (P100 does fp16 at twice the fp32 rate, and the QK loop is the hot loop) and the
//       chunk's pair-sum is folded into an fp32 accumulator, so no fp16 accumulator ever carries
//       more than 32 terms. The running softmax state (kqmax, kqsum) and the P.V accumulator are
//       fp32 unconditionally — there is no fp16-state variant of this kernel to choose wrongly.
// Keeping it in its own translation unit also means the 64/128/256 instantiations that ship today
// are not recompiled, not re-scheduled and not re-registered: the lever is default-OFF
// structurally, not just by an if.
//
// SCOPE, stated so a green test on the wrong card cannot be read as a pass. This kernel is only
// ever dispatched when PXA_FA_TILE_512 is armed AND the device is cc 6.0 or cc 7.0 AND the node
// is f16 K/V, GGML_PREC_DEFAULT, no attention sinks, no logit softcap, and DKQ/DV is 512/512 or
// 576/512. Every one of those is checked in ggml_cuda_fattn_tile_big_is_supported(), which the
// dispatcher and the backend's supports-op predicate both call, so the two can never disagree.
//   * cc 6.1 (1080 Ti) is EXCLUDED on purpose: GP104 runs fp16 arithmetic at 1/64 rate, so the
//     half2 QK loop that makes this kernel worth having on a P100 would be a large loss there.
//     A 512-wide fp32 tile for sm_61 is a separate kernel and a separate measurement.
//   * Turing and later are excluded because fattn-new-mma.cu already serves both head sizes there
//     and does it with tensor cores.
//   * Attention sinks and logit softcap are declined rather than implemented: neither model that
//     needs this head size uses them, and an untested branch is worse than a decline.
//
// ARITHMETIC, term by term, against the unfused path this replaces: Q is scaled in fp32 and
// stored as half2 (as in the shipped tile kernel); K and V are read from the fp16 cache and
// converted where they are used; QK products are fp16, summed in fp16 within a 64-element chunk
// and in fp32 across chunks; the mask and the ALiBi slope are applied in fp32; the online softmax
// max and sum are fp32; the post-exp probabilities round-trip through shared memory as fp16 (the
// shipped tile kernel's layout); P.V products and their accumulation are fp32. The result is
// therefore NOT bitwise equal to the unfused fp32 path, and the device test
// tests/test-fa-tile-big-dev.cpp gates it the way the campaign gates every numerics change —
// against the same graph computed on the CPU backend, with the error reported, not assumed.

#include "../common.cuh"
#include "../fattn-common.cuh"
#include "fattn-tile-big.cuh"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// KV positions processed per tile. 64 keeps the K/V staging tile at 8,448 B and matches the
// shipped tile kernel's grain, so the mask, the padding assumption (launch_fattn asserts
// K->ne[1] % FATTN_KQ_STRIDE == 0 with FATTN_KQ_STRIDE == 256) and the parallel_blocks split all
// behave exactly as they do for the 64/128/256 kernels.
#define FATTN_KQ_STRIDE_TILE_BIG 64

// Head elements staged per pass. 64 elements == 32 half2 == one warp's worth of columns, which
// makes the staging loop a single iteration per KV row and keeps the fp16 QK accumulator to 32
// terms. DKQ and DV are both multiples of 64 for every shape this kernel serves.
#define FATTN_NBATCH_K_TILE_BIG 64

// ---------------------------------------------------------------------------------------------
// THE SCORE TILE'S PRECISION, as a compile-time choice.
//
// The running softmax state (kqmax, kqsum) and the P.V accumulator have been fp32 since the first
// version of this kernel. What is NOT fp32 is the score tile itself: the raw logit is rounded to
// fp16 on its way into shared memory BEFORE the running maximum is subtracted from it, the
// probability is rounded to fp16 on its way back out, and the per-chunk QK product is summed in
// half2. Those three fp16 carriers are cheap on a P100 (fp16 runs at twice the fp32 rate and the
// QK loop is the hot loop) and they are what the shipped 64/128/256 tile kernel does too.
//
// The 2026-09-13 device-test run says they are not free at depth: measured against the same graph
// on the CPU backend, the error grows with the KV extent far faster than random accumulation can
// account for. `soft_f32` exists to answer that with a measurement instead of an argument -- it
// puts all three carriers in fp32 so one binary can run both and say whether the depth error is
// precision at all. If it is not, the defect is structural and no amount of fp32 will move it.
template <bool f32> struct pxa_kq_prec;
template <> struct pxa_kq_prec<false> { using elem = half;  using pair = half2;  };
template <> struct pxa_kq_prec<true>  { using elem = float; using pair = float2; };

static __device__ __forceinline__ void pxa_kq_mad(half2 & acc, const half2 k, const half2 q) {
    acc += k*q;
}
static __device__ __forceinline__ void pxa_kq_mad(float2 & acc, const half2 k, const half2 q) {
    const float2 kf = __half22float2(k);
    const float2 qf = __half22float2(q);
    acc.x += kf.x*qf.x;
    acc.y += kf.y*qf.y;
}
static __device__ __forceinline__ float pxa_kq_hsum(const half2 a) {
    const float2 t = __half22float2(a);
    return t.x + t.y;
}
static __device__ __forceinline__ float pxa_kq_hsum(const float2 a) {
    return a.x + a.y;
}
static __device__ __forceinline__ void pxa_kq_zero(half2 & a) {
    a = make_half2(0.0f, 0.0f);
}
static __device__ __forceinline__ void pxa_kq_zero(float2 & a) {
    a = make_float2(0.0f, 0.0f);
}
static __device__ __forceinline__ void pxa_kq_set(half * dst, const float v) {
    *dst = __float2half(v);
}
static __device__ __forceinline__ void pxa_kq_set(float * dst, const float v) {
    *dst = v;
}
static __device__ __forceinline__ float2 pxa_kq_get2(const half2 a) {
    return __half22float2(a);
}
static __device__ __forceinline__ float2 pxa_kq_get2(const float2 a) {
    return a;
}
static __device__ __forceinline__ void pxa_kq_put2(half2 * dst, const float x, const float y) {
    *dst = make_half2(x, y);
}
static __device__ __forceinline__ void pxa_kq_put2(float2 * dst, const float x, const float y) {
    *dst = make_float2(x, y);
}
static __device__ __forceinline__ float pxa_kq_lo(const half2 a)  { return  __low2float(a); }
static __device__ __forceinline__ float pxa_kq_hi(const half2 a)  { return __high2float(a); }
static __device__ __forceinline__ float pxa_kq_lo(const float2 a) { return a.x; }
static __device__ __forceinline__ float pxa_kq_hi(const float2 a) { return a.y; }
// ---------------------------------------------------------------------------------------------

template<int DKQ, int DV, int ncols, int nwarps, int parallel_blocks, int nbatch_K, bool soft_f32>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// The fp16 schedule is register-tight (118-119 of 128 at ncols 16) and is asked for two blocks per
// SM because occupancy is the whole point of it. The fp32 diagnostic is asked for correctness, not
// throughput: holding it to two blocks would spill the QK loop and make its numbers describe the
// spills rather than the precision, so it gets the whole register file and one block.
__launch_bounds__(nwarps*WARP_SIZE, soft_f32 ? 1 : 2)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_big(
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
    GGML_UNUSED(ne00); GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne13);
    GGML_UNUSED(ne31); GGML_UNUSED(nb03); GGML_UNUSED(nb13); GGML_UNUSED(nb23);
    GGML_UNUSED(ne0);  GGML_UNUSED(ne1);  GGML_UNUSED(ne2);  GGML_UNUSED(ne3);
    GGML_UNUSED(softcap);

    constexpr int KQ_STRIDE = FATTN_KQ_STRIDE_TILE_BIG;
    constexpr int NB2       = nbatch_K/2; // head elements per staging pass, in half2 units

    static_assert(DKQ % nbatch_K == 0,       "DKQ not divisible by nbatch_K");
    static_assert(DV  % nbatch_K == 0,       "DV not divisible by nbatch_K");
    static_assert(nbatch_K % (2*WARP_SIZE) == 0, "nbatch_K not divisible by 2*WARP_SIZE == 64");
    static_assert(KQ_STRIDE % nwarps == 0,   "KQ_STRIDE not divisible by nwarps");
    static_assert(KQ_STRIDE % WARP_SIZE == 0,"KQ_STRIDE not divisible by WARP_SIZE");
    static_assert(ncols % nwarps == 0,       "ncols not divisible by nwarps");

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V + nb22*(blockIdx.y / gqa_ratio));
    const int      stride_mask = nb31 / sizeof(half); // mask query-row stride
    const half   * maskh = (const half   *)  mask + stride_mask*ic0;

    // K and V get their OWN row strides: at DKQ 576 / DV 512 (MLA) V is a narrower view of the
    // same rows as K, so nb21 != nb11 and using K's stride for both would read the wrong rows.
    const int stride_K2 = nb11 / sizeof(half2);
    const int stride_V2 = nb21 / sizeof(half2);

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);

    using kq_elem = typename pxa_kq_prec<soft_f32>::elem;
    using kq_pair = typename pxa_kq_prec<soft_f32>::pair;

    __shared__ __align__(sizeof(kq_pair)) kq_elem KQ[ncols*KQ_STRIDE];
    kq_pair * KQ2 = (kq_pair *) KQ;

    // One head-dimension slice of the K (then V) tile. Padded by one half2 to break bank conflicts.
    __shared__ half2 KV_tmp[KQ_STRIDE][NB2 + 1];

    // Q stays resident for the whole block: it is read once per KV tile per head-dim chunk.
    __shared__ half2 Q_h2[ncols][DKQ/2];

    float  kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float2 kqsum[ncols/nwarps] = {{0.0f, 0.0f}};

    float2 VKQ[ncols/nwarps][(DV/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Q -> shared memory, pre-scaled in fp32 and stored as half2 (the shipped tile kernel's layout).
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < DKQ/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            Q_h2[j][i] = make_half2(scale*tmp.x, scale*tmp.y);
        }
    }

    __syncthreads();

    const int k_start = parallel_blocks == 1 ? 0 : ip*KQ_STRIDE;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*KQ_STRIDE) {
        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        // KQ, chunked over the head dimension. sumf is the fp32 accumulator across chunks; each
        // chunk's 32 half2 products are summed in fp16 and folded in once.
        float sumf[KQ_STRIDE/WARP_SIZE][ncols/nwarps] = {{0.0f}};

        // Deliberately NOT unrolled: sumf is indexed by (i, j) only, so nothing here needs kc to
        // be a compile-time constant, and unrolling DKQ/nbatch_K copies of a fully-unrolled inner
        // loop is a large amount of code for no register-allocation benefit.
        for (int kc = 0; kc < DKQ/2; kc += NB2) {
            __syncthreads(); // the previous chunk (or the previous tile's V pass) is done reading KV_tmp

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < KQ_STRIDE; i_KQ_0 += nwarps) {
                const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
                for (int k0 = 0; k0 < NB2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KV_tmp[i_KQ][k] = K_h2[(k_VKQ_0 + i_KQ)*stride_K2 + kc + k];
                }
            }

            __syncthreads();

            kq_pair sum2[KQ_STRIDE/WARP_SIZE][ncols/nwarps];
#pragma unroll
            for (int i = 0; i < KQ_STRIDE/WARP_SIZE; ++i) {
#pragma unroll
                for (int j = 0; j < ncols/nwarps; ++j) {
                    pxa_kq_zero(sum2[i][j]);
                }
            }

#pragma unroll
            for (int k = 0; k < NB2; ++k) {
                half2 K_k[KQ_STRIDE/WARP_SIZE];
                half2 Q_k[ncols/nwarps];

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < KQ_STRIDE; i_KQ_0 += WARP_SIZE) {
                    const int i_KQ = i_KQ_0 + threadIdx.x;

                    K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k];
                }
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    const int j_KQ = j_KQ_0 + threadIdx.y;

                    Q_k[j_KQ_0/nwarps] = Q_h2[j_KQ][kc + k];
                }

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < KQ_STRIDE; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                    for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                        pxa_kq_mad(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps],
                                   K_k[i_KQ_0/WARP_SIZE], Q_k[j_KQ_0/nwarps]);
                    }
                }
            }

#pragma unroll
            for (int i = 0; i < KQ_STRIDE/WARP_SIZE; ++i) {
#pragma unroll
                for (int j = 0; j < ncols/nwarps; ++j) {
                    sumf[i][j] += pxa_kq_hsum(sum2[i][j]);
                }
            }
        }

        // Scores: mask, ALiBi and the running maximum, all in fp32.
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < KQ_STRIDE; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                float sum = sumf[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps];
                sum += mask ? slopef*__half2float(maskh[j_KQ*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], sum);

                pxa_kq_set(&KQ[j_KQ*KQ_STRIDE + i_KQ], sum);
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
            for (int i0 = 0; i0 < KQ_STRIDE/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 kq = pxa_kq_get2(KQ2[j*(KQ_STRIDE/2) + i]);
                const float  vx = expf(kq.x - kqmax[j0/nwarps]);
                const float  vy = expf(kq.y - kqmax[j0/nwarps]);
                kqsum[j0/nwarps].x = kqsum[j0/nwarps].x*KQ_max_scale + vx;
                kqsum[j0/nwarps].y = kqsum[j0/nwarps].y*KQ_max_scale + vy;
                pxa_kq_put2(&KQ2[j*(KQ_STRIDE/2) + i], vx, vy);
            }

#pragma unroll
            for (int i0 = 0; i0 < (DV/2)/WARP_SIZE; ++i0) {
                VKQ[j0/nwarps][i0].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0].y *= KQ_max_scale;
            }
        }

        __syncthreads();

        // P.V, chunked over the head dimension the same way. Each pass owns a disjoint slice of
        // the output accumulator, so the per-element accumulation order over KV positions is the
        // shipped tile kernel's: k0 = 0, 2, 4, ... KQ_STRIDE-2, tile after tile.
#pragma unroll
        for (int ic = 0; ic < DV/2; ic += NB2) {
            __syncthreads();

#pragma unroll
            for (int k0 = 0; k0 < KQ_STRIDE; k0 += nwarps) {
                const int k = k0 + threadIdx.y;

#pragma unroll
                for (int i0 = 0; i0 < NB2; i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    KV_tmp[k][i] = V_h2[(k_VKQ_0 + k)*stride_V2 + ic + i];
                }
            }

            __syncthreads();

            // The ic loop above IS unrolled (VKQ is a register array indexed by the head-dim
            // chunk, so ic has to be a compile-time constant); this one is not, which keeps the
            // unrolled body to one chunk's worth of instructions.
#pragma unroll 4
            for (int k0 = 0; k0 < KQ_STRIDE; k0 += 2) {
                half2  V_k[NB2/WARP_SIZE][2];
                kq_pair KQ_k[ncols/nwarps];

#pragma unroll
                for (int i0 = 0; i0 < NB2; i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    V_k[i0/WARP_SIZE][0] = KV_tmp[k0 + 0][i];
                    V_k[i0/WARP_SIZE][1] = KV_tmp[k0 + 1][i];
                }
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    const int j = j0 + threadIdx.y;

                    KQ_k[j0/nwarps] = KQ2[j*(KQ_STRIDE/2) + k0/2];
                }

#pragma unroll
                for (int i0 = 0; i0 < NB2; i0 += WARP_SIZE) {
#pragma unroll
                    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                        const float2 v0  = __half22float2(V_k[i0/WARP_SIZE][0]);
                        const float2 v1  = __half22float2(V_k[i0/WARP_SIZE][1]);
                        const float  kql = pxa_kq_lo(KQ_k[j0/nwarps]);
                        const float  kqh = pxa_kq_hi(KQ_k[j0/nwarps]);

                        VKQ[j0/nwarps][(ic + i0)/WARP_SIZE].x += v0.x*kql + v1.x*kqh;
                        VKQ[j0/nwarps][(ic + i0)/WARP_SIZE].y += v0.y*kql + v1.y*kqh;
                    }
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

        // Attention sinks are declined by ggml_cuda_fattn_tile_big_is_supported(), so `sinks` is
        // always null here; the parameter is kept because fattn_kernel_t carries it.
        GGML_UNUSED(sinks);

#pragma unroll
        for (int i00 = 0; i00 < DV; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            float2 dst_val = VKQ[j_VKQ_0/nwarps][i00/(2*WARP_SIZE)];
            if (parallel_blocks == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
            dst[j_dst*DV*gridDim.y + DV*blockIdx.y + i0 + 0] = dst_val.x;
            dst[j_dst*DV*gridDim.y + DV*blockIdx.y + i0 + 1] = dst_val.y;
        }

        if (parallel_blocks != 1 && threadIdx.x == 0) {
            dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] =
                make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// PXA_FA_TILE_512 — default OFF. This kernel has no silicon A/B yet; arming it changes WHICH
// path a 512-wide attention node takes (tile kernel instead of the unfused KQ/softmax/KQV graph),
// which is a correctness-relevant change and not one to ship on a compile.
// PXA_FA_TILE_512_FP32 - default OFF, and meaningless unless PXA_FA_TILE_512 is on. It selects the
// all-fp32 score tile described at the top of this file. It exists so that ONE binary can answer
// whether the kernel's depth-dependent error is precision or structure, without a second build.
bool ggml_cuda_fattn_tile_big_fp32_soft() {
    static const bool on = [] {
        const char * e = getenv("PXA_FA_TILE_512_FP32");
        const bool v = e != nullptr && atoi(e) != 0;
        if (v) {
            fprintf(stderr, "PXA_FA_TILE_512_FP32: the 512/576 tile kernel keeps the score tile, the "
                            "probabilities and the QK chunk sum in fp32 (diagnostic; slower)\n");
        }
        return v;
    }();
    return on;
}

bool ggml_cuda_fattn_tile_big_armed() {
    static const bool armed = [] {
        const char * e = getenv("PXA_FA_TILE_512");
        const bool on = e != nullptr && atoi(e) != 0;
        if (on) {
            fprintf(stderr, "PXA_FA_TILE_512: ARMED (head size 512/512 and 576/512 flash-attention "
                            "on sm_60/sm_70 -> pxa tile kernel; unset reverts to the unfused path)\n");
        }
        return on;
    }();
    return armed;
}

template <int DKQ, int DV, int ncols, int parallel_blocks>
static void launch_fattn_tile_big(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps   = 8;
    constexpr int nbatch_K = FATTN_NBATCH_K_TILE_BIG;

    fattn_kernel_t fattn_kernel = ggml_cuda_fattn_tile_big_fp32_soft()
        ? flash_attn_tile_ext_big<DKQ, DV, ncols, nwarps, parallel_blocks, nbatch_K, true>
        : flash_attn_tile_ext_big<DKQ, DV, ncols, nwarps, parallel_blocks, nbatch_K, false>;
    launch_fattn<DKQ, DV, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, ncols, true, true);
}

// Column blocking, mirroring the shipped tile kernel's dispatcher: a small batch gets the
// 4-way parallel_blocks split over the key range (the KV extent, not the batch, is what there is
// to parallelise at decode), a large one runs a single block per column group.
template <int DKQ, int DV>
static void dispatch_fattn_tile_big(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] <= 8) {
        launch_fattn_tile_big<DKQ, DV,  8, 4>(ctx, dst);
        return;
    }
    if (Q->ne[1] <= 16) {
        launch_fattn_tile_big<DKQ, DV, 16, 4>(ctx, dst);
        return;
    }
    launch_fattn_tile_big<DKQ, DV, 16, 1>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_tile_big(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (K->ne[0] == 512 && V->ne[0] == 512) {
        dispatch_fattn_tile_big<512, 512>(ctx, dst);
        return;
    }
    if (K->ne[0] == 576 && V->ne[0] == 512) {
        dispatch_fattn_tile_big<576, 512>(ctx, dst);
        return;
    }

    // Unreachable: ggml_cuda_fattn_tile_big_is_supported() is checked before every call.
    GGML_ABORT("pxa tile-big flash-attention: unsupported head sizes %d / %d",
               (int) K->ne[0], (int) V->ne[0]);
}

bool ggml_cuda_fattn_tile_big_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];

    // sm_60 and sm_70 only -- see the SCOPE note at the top of this file.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !(cc == CC_PASCAL || cc == CC_VOLTA)) {
        return false;
    }

    if (K->ne[0] != Q->ne[0]) {
        return false;
    }
    if (!((K->ne[0] == 512 && V->ne[0] == 512) || (K->ne[0] == 576 && V->ne[0] == 512))) {
        return false;
    }

    // f16 K/V only. launch_fattn() can materialise an f16 copy of a quantised cache, but its
    // conversion table has no entry for these head-size pairs, so a quantised cache would abort
    // inside the launcher rather than here.
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }

    if (sinks != nullptr) {
        return false; // attention sinks are not implemented in this kernel
    }

    if (dst->op_params[3] != GGML_PREC_DEFAULT) {
        return false; // the QK products are fp16; a caller asking for fp32 precision gets neither
    }

    float softcap;
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (softcap != 0.0f) {
        return false; // not implemented (and not needed by either model at this head size)
    }

    return true;
}
