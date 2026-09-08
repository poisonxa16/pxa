//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "fattn-vec-common.cuh"
#include "pxa/pxa-enhance.cuh"   // pxa_fa_gqa_pack_default() -- the per-topology ENHANCE default

// PXA_FA_GQA_PACK build-time floor (0 = let the level/topology resolver decide). Defining this
// to 2/4/8 forces a pack width into the binary regardless of level; it exists for A/B builds.
#ifndef PXA_FA_GQA_PACK_DEFAULT
#define PXA_FA_GQA_PACK_DEFAULT (-1)
#endif
#ifndef PXA_FA_GQA_QSMEM_DEFAULT
#define PXA_FA_GQA_QSMEM_DEFAULT 0
#endif

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int Dk, int Dv, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool pxa_vilp = false> // Dk, Dv == K-, V-head size
#ifndef GGML_USE_HIP
__launch_bounds__(Dk, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int2 * __restrict__ KV_min_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {

    // Skip unused kernel variants for faster compilation:
    if constexpr (Dk == Dv || (Dk == 192 && Dv == 128) || (Dk == 576 && Dv == 512)) {
    if (use_logit_softcap && !(Dk == 128 || Dk == 256)) {
        NO_DEVICE_CODE;
        return;
    }
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f32_t vec_dot_KQ = get_vec_dot_KQ_f32<Dk>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f32_t dequantize_1_v = get_dequantize_1_f32(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(Dk % (2*WARP_SIZE) == 0, "Dk not divisible by 2*WARP_SIZE == 64.");
    static_assert(Dv % (2*WARP_SIZE) == 0, "Dv not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = Dk / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < Dk);

    __shared__ float KQ[ncols*Dk];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*Dk + tid] = -FLT_MAX/2.0f;
    }

    float kqmax[ncols];
    float kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
        kqsum[j] = 0.0f;
    }

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ float maskf_shared[ncols*Dk];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskf_shared[j*Dk + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    float2  Q_f2[ncols][Dk/(2*WARP_SIZE)];
    int    Q_i32[ncols][Dk/(sizeof(int)*QK8_1) == 0 ? 1 : Dk/(sizeof(int)*QK8_1)];
    float2  Q_ds[ncols][Dk/QK8_1 == 0 ? 1 : Dk/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*Dk];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + Dk/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < Dk/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2>(Q_f + 4*i0, scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*Dk];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + Dk/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_f2[j][i0/WARP_SIZE]    = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_f2[j][i0/WARP_SIZE].x *= scale;
                Q_f2[j][i0/WARP_SIZE].y *= scale;
            }
        }
    }

    float VKQ[ncols] = {0.0f};

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;

    K     += (first_y + blockIdx.y*Dk) * nb11;
    V     += (first_y + blockIdx.y*Dv) * nb21;
    maskh += (first_y + blockIdx.y*Dk);
    for (int k_VKQ_0 = first_y + blockIdx.y*Dk; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*Dk,
             // Increment pointers after each loop:
             K += gridDim.y*Dk*nb11, V += gridDim.y*Dv*nb21, maskh += gridDim.y*Dk) {

        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskf_shared[j*Dk + tid] = slope*__half2float(maskh[j*ne11 + tid]);
            }
            __syncthreads();
        }

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < Dk; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > Dk && i_KQ >= Dk) || (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_f2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskf_shared[j*Dk + i_KQ];

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x == 0) {
                    KQ[j*Dk + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(KQ[j*Dk + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*Dk + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

        if constexpr (pxa_vilp && ncols == 1) {
            // PXA_FA_VEC_ILP (2026-08-03): the stock V pass below is a single serial FMA chain
            // of Dv dependent adds per thread (nvcc does not reassociate fp adds), each add
            // waiting on a 2-byte V load. Profiled 403 us/call at 9k KV on P100 (183 GB/s,
            // ~2.5x off achievable). Four independent partial accumulators quadruple the ILP;
            // the fixed pairwise fold keeps the per-tile reduction order deterministic.
            // NOT bit-exact vs the serial chain (summation order) — banner-gated, ppl-verified.
            // Partial unroll only: a full Dv-unroll with 4 chains made the register allocator
            // keep dozens of V loads in flight -> spills, and the spill cost scales with KV
            // tiles (measured: deep decode 26.8 -> 18.6 while shallow was unchanged).
            float vacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
            for (int k0 = 0; k0 < Dv; k0 += 4) {
                if (FATTN_KQ_STRIDE % Dv != 0 && k_VKQ_0 + k0 >= ne11) {
                    break;
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float V_ki = dequantize_1_v(V + (k0 + u)*nb21, tid);
                    vacc[u] += V_ki*KQ[k0 + u];
                }
            }
            VKQ[0] += (vacc[0] + vacc[1]) + (vacc[2] + vacc[3]);
        } else {
#pragma unroll
        for (int k = 0; k < Dv; ++k) {
            if (FATTN_KQ_STRIDE % Dv != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            const float V_ki = dequantize_1_v(V + k*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_ki*KQ[j*Dk + k];
            }
        }
        }

        __syncthreads();
    }

    if (sinksf && blockIdx.y == 0) {
        const float sink = sinksf[head];

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sink);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(sink - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        float dst_val = VKQ[j_VKQ];
        if (gridDim.y == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*Dv + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// =================================================================================================
// PXA_FA_GQA_PACK (2026-08-03) — GQA head-packed D=256 f16-KV decode vec kernel.
//
// WHY. flash_attn_vec_ext_f32 gives one block to ONE Q head, so under grouped-query attention it
// re-reads the whole K and V cache once per Q head in the group. qwen35moe-122B decodes with 32 Q
// heads over 2 KV heads (gqa_ratio 16): profiled at fill 8881 on a P100 the kernel issues ~291 MB
// of K+V loads per full-attention layer against 18.2 MB of unique cache, and costs 396 us/call =
// 4.82 ms of a 36.0 ms token — the largest single attention item in the decode profile. It also
// compiles to 242 registers, so cudaOccupancyMaxActiveBlocksPerMultiprocessor returns 1 and each
// SM runs 8 of its 64 possible warps.
//
// WHAT. One block takes NH Q heads that share a KV head. The K row is loaded into registers ONCE
// per block and dotted against NH query vectors; each V element is loaded once and fused into NH
// accumulators. Load traffic and load-INSTRUCTION count both fall by NH while the FMA count is
// unchanged, trading a memory-issue bound for an arithmetic one. The NH independent VKQ chains
// also supply the instruction-level parallelism that PXA_FA_VEC_ILP had to fake with 4 partial
// accumulators, so the V pass keeps ONE accumulator per head and stays in the stock ascending-k
// summation order (i.e. it is the bit-exact non-ILP order, not the PXA_FA_VEC_ILP order).
//
// GATES. ncols == 1 (decode), K == V == f16, Dk == Dv == 256, no logit softcap, NH divides the
// gqa_ratio and the head count, mask (if any) shared by the group — which it is, the mask is
// indexed by (token, kv position) only. Everything outside that falls back to the stock kernel.
// =================================================================================================
template<int D, int NH, bool QSMEM = false>
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f32_gqa(
        const char  * __restrict__ Q,
        const char  * __restrict__ K,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        const char  * __restrict__ sinks,
        const int2  * __restrict__ KV_min_max,
        float       * __restrict__ dst,
        float2      * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    GGML_UNUSED(logit_softcap); GGML_UNUSED(ne00); GGML_UNUSED(ne10); GGML_UNUSED(ne13);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(nb32);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    static_assert(FATTN_KQ_STRIDE % D == 0, "the KV tail guard is elided on this assumption");

    constexpr int nwarps = D/WARP_SIZE;                      // 8 warps for D == 256
    constexpr int QPL    = D/(2*WARP_SIZE);                  // float2 / half2 per lane per row
    constexpr dequantize_1_f32_t dequantize_1_v = get_dequantize_1_f32(GGML_TYPE_F16);

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    const int ic0       = blockIdx.x;                        // Q column (token); ncols == 1
    const int ne02g     = ne02 / NH;                         // head groups per sequence
    const int sequence  = blockIdx.z / ne02g;
    const int head0     = (blockIdx.z - sequence*ne02g) * NH;
    const int gqa_ratio = ne02 / ne12;

    Q += (size_t)nb03*sequence + (size_t)nb02*head0 + (size_t)nb01*ic0;
    K += (size_t)nb13*sequence + (size_t)nb12*(head0 / gqa_ratio);
    V += (size_t)nb23*sequence + (size_t)nb22*(head0 / gqa_ratio);

    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    float slope[NH];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        slope[j] = get_alibi_slope(max_bias, head0 + j, n_head_log2, m0, m1);
    }

    __shared__ float KQ[NH*D];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        KQ[j*D + tid] = -FLT_MAX/2.0f;
    }

    __shared__ float kqmax_shared[NH][WARP_SIZE];
    __shared__ float kqsum_shared[NH][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    // The mask depends on (token, kv position) only, so ONE row serves the whole head group.
    __shared__ float maskf_shared[D];
    maskf_shared[tid] = 0.0f;

    __syncthreads();

    float kqmax[NH], kqsum[NH], VKQ[NH];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
        kqsum[j] = 0.0f;
        VKQ [j] = 0.0f;
    }

    // Q lives in registers (QSMEM == false) or in shared memory (QSMEM == true).
    //
    // NH*QPL float2 per lane is 8*4 = 64 registers at NH=8, which is what pushes that variant to
    // 161 registers and 1 block/SM. The measured NH sweep is explained exactly by the product
    // (traffic reduction NH) x (blocks/SM): NH=2 2x3=6 -> 29.38, NH=4 4x2=8 -> 29.67, NH=8
    // 8x1=8 -> 29.59. To move past that product, NH=8 has to reach 2 blocks/SM, so Q goes to
    // shared. Layout [j][i][lane] keeps consecutive lanes on consecutive banks.
    float2 Q_reg[QSMEM ? 1 : NH][QPL];
    __shared__ float2 Q_sh[QSMEM ? NH*QPL*WARP_SIZE : 1];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        const float2 * Q_f2_j = (const float2 *) (Q + (size_t)j*nb02);
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            float2 q = Q_f2_j[i0 + threadIdx.x];
            q.x *= scale;
            q.y *= scale;
            if constexpr (QSMEM) {
                if (threadIdx.y == 0) {           // Q is warp-invariant: warp 0 stages it once
                    Q_sh[(j*QPL + i0/WARP_SIZE)*WARP_SIZE + threadIdx.x] = q;
                }
            } else {
                Q_reg[j][i0/WARP_SIZE] = q;
            }
        }
    }
    if constexpr (QSMEM) {
        __syncthreads();
    }

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y   = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;

    K     += (size_t)(first_y + blockIdx.y*D) * nb11;
    V     += (size_t)(first_y + blockIdx.y*D) * nb21;
    maskh += (first_y + blockIdx.y*D);

    for (int k_VKQ_0 = first_y + blockIdx.y*D; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*D,
             K += (size_t)gridDim.y*D*nb11, V += (size_t)gridDim.y*D*nb21, maskh += gridDim.y*D) {

        if (mask) {
            maskf_shared[tid] = __half2float(maskh[tid]);
            __syncthreads();
        }

        float kqmax_new_arr[NH];
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            // THE POINT OF THIS KERNEL: one K-row fetch feeds NH dot products.
            const half2 * K_h2 = (const half2 *) (K + (size_t)i_KQ*nb11);
            half2 K_ik[QPL];
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                K_ik[i0/WARP_SIZE] = K_h2[i0 + threadIdx.x];
            }

#pragma unroll
            for (int j = 0; j < NH; ++j) {
                // same ascending (low, high) accumulation order as vec_dot_fattn_vec_KQ_f16
                float sum = 0.0f;
#pragma unroll
                for (int i = 0; i < QPL; ++i) {
                    const float2 q = QSMEM ? Q_sh[(j*QPL + i)*WARP_SIZE + threadIdx.x]
                                           : Q_reg[QSMEM ? 0 : j][i];
                    sum +=  __low2float(K_ik[i]) * q.x;
                    sum += __high2float(K_ik[i]) * q.y;
                }
                sum = warp_reduce_sum(sum);

                sum += slope[j]*maskf_shared[i_KQ];

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_arr[j];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

        // One V fetch, NH fused multiply-adds. Partial unroll only: a full D-unroll keeps dozens
        // of V loads in flight and spills (the lesson banked by PXA_FA_VEC_ILP).
#pragma unroll 4
        for (int k = 0; k < D; ++k) {
            const float V_ki = dequantize_1_v(V + (size_t)k*nb21, tid);
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                VKQ[j] += V_ki*KQ[j*D + k];
            }
        }

        __syncthreads();
    }

    if (sinksf && blockIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sinksf[head0 + j]);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(sinksf[head0 + j] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        kqsum[j] = kqsum_shared[j][threadIdx.x];
        kqsum[j] = warp_reduce_sum(kqsum[j]);

        float dst_val = VKQ[j];
        if (gridDim.y == 1) {
            dst_val /= kqsum[j];
        }
        dst[(((size_t)(sequence*ne01 + ic0)*ne02 + head0 + j)*gridDim.y + blockIdx.y)*D + tid] = dst_val;
    }

    // NOTE: `j` must stay a compile-time constant here — indexing kqmax[]/kqsum[] with a runtime
    // `tid` (as the stock ncols kernel can, because ncols is 1 there) would force both register
    // arrays into local memory.
    if (gridDim.y != 1) {
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            if (tid == j) {
                dst_meta[((size_t)(sequence*ne01 + ic0)*ne02 + head0 + j)*gridDim.y + blockIdx.y]
                    = make_float2(kqmax[j], kqsum[j]);
            }
        }
    }
}

// PXA_FA_GQA_PACK resolver: 0 = off (stock per-head kernel), else the number of Q heads packed
// into one block. Only 2/4/8 are instantiated.
static inline int pxa_fa_gqa_pack() {
    static const int nh = [](){
        const char * e = getenv("PXA_FA_GQA_PACK");
        int v = e ? atoi(e)
                  : (PXA_FA_GQA_PACK_DEFAULT >= 0 ? PXA_FA_GQA_PACK_DEFAULT : pxa_fa_gqa_pack_default());
        if (v != 0 && v != 2 && v != 4 && v != 8) {
            fprintf(stderr, "PXA_FA_GQA_PACK: %d is not one of 0/2/4/8 — falling back to OFF\n", v);
            v = 0;
        }
        fprintf(stderr, "PXA_FA_GQA_PACK: NH=%d (%s)\n", v,
                v ? "D=256 f16-KV decode packs NH GQA heads per block; PXA_FA_GQA_PACK=0 reverts"
                  : "OFF — stock one-block-per-head vec kernel");
        return v;
    }();
    return nh;
}

// PXA_FA_GQA_QSMEM: stage the NH query rows in shared memory instead of registers, trading a
// shared load per dot term for ~64 registers at NH=8 (the difference between 1 and 2 blocks/SM).
// Only the NH=4 and NH=8 kernels carry the twin.
static inline bool pxa_fa_gqa_qsmem() {
    static const bool v = [](){
        const char * e = getenv("PXA_FA_GQA_QSMEM");
        const bool on = e ? atoi(e) != 0 : (PXA_FA_GQA_QSMEM_DEFAULT != 0);
        fprintf(stderr, "PXA_FA_GQA_QSMEM: %s (Q staged in shared for NH=4/8)\n", on ? "ON" : "OFF");
        return on;
    }();
    return v;
}

template <int Dk, int Dv, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f32_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = Dk/WARP_SIZE;
    fattn_kernel_t fattn_kernel;
    if constexpr (Dk == 256 && Dv == 256 && cols_per_block == 1 && type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
        // PXA_FA_GQA_PACK: pack NH GQA heads per block (see the kernel above). Requires NH to
        // divide BOTH the head count and the gqa_ratio, so the packed heads share one KV head.
        if constexpr (!use_logit_softcap) {
            const ggml_tensor * Qt = dst->src[0];
            const ggml_tensor * Kt = dst->src[1];
            const int nh_env    = pxa_fa_gqa_pack();
            const int n_head    = (int) Qt->ne[2];
            const int gqa_ratio = Kt->ne[2] > 0 ? (int)(Qt->ne[2] / Kt->ne[2]) : 1;
            const int nh = (nh_env && n_head % nh_env == 0 && gqa_ratio % nh_env == 0) ? nh_env : 0;

            const bool qs = pxa_fa_gqa_qsmem();
            constexpr bool need_f16_K_g = false;   // Dk == 256 -> the vec kernel reads f16 K directly
            constexpr bool need_f16_V_g = false;
            constexpr size_t nbytes_shared_g = 0;
            switch (nh) {
                case 2:
                    launch_fattn<Dv, cols_per_block, 2>(ctx, dst, flash_attn_vec_ext_f32_gqa<Dv, 2, false>,
                            nwarps, nbytes_shared_g, Dv, need_f16_K_g, need_f16_V_g);
                    return;
                case 4:
                    launch_fattn<Dv, cols_per_block, 4>(ctx, dst,
                            qs ? flash_attn_vec_ext_f32_gqa<Dv, 4, true> : flash_attn_vec_ext_f32_gqa<Dv, 4, false>,
                            nwarps, nbytes_shared_g, Dv, need_f16_K_g, need_f16_V_g);
                    return;
                case 8:
                    launch_fattn<Dv, cols_per_block, 8>(ctx, dst,
                            qs ? flash_attn_vec_ext_f32_gqa<Dv, 8, true> : flash_attn_vec_ext_f32_gqa<Dv, 8, false>,
                            nwarps, nbytes_shared_g, Dv, need_f16_K_g, need_f16_V_g);
                    return;
                default:
                    break;
            }
        }
        // PXA_FA_VEC_ILP: D=256 f16-KV decode gets the 4-accumulator V pass (see the kernel).
        // Default ON; =0 reverts to the serial-chain form. Only this instantiation carries the
        // twin, so compile cost is one extra kernel.
        static const bool vilp = [](){
            const char * e = getenv("PXA_FA_VEC_ILP");
            const bool on = !(e && atoi(e) == 0);
            fprintf(stderr, "PXA_FA_VEC_ILP: %s (D=256 decode V-pass 4-way ILP; PXA_FA_VEC_ILP=0 reverts)\n", on ? "ON" : "OFF");
            return on;
        }();
        fattn_kernel = vilp ? flash_attn_vec_ext_f32<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, true>
                            : flash_attn_vec_ext_f32<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, false>;
    } else {
        fattn_kernel = flash_attn_vec_ext_f32<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>;
    }
    constexpr bool need_f16_K = Dk != 128 && Dk != 256;
    constexpr bool need_f16_V = Dv != 64 && Dv != 128 && Dv != 256;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<Dv, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, Dv, need_f16_K, need_f16_V);
}

template <int Dk, int Dv, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F32_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f32_case                     \
    <D, D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_F32_CASE_DKDV(Dk, Dv, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f32_case                     \
    <Dk, Dv, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16,  GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE_DKDV(192, 128, GGML_TYPE_F16, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE_DKDV(192, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
