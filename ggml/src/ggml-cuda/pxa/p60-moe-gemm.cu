// pxa / PXA kernel suite -- p60-moe-gemm.cu: fp16 expert GEMM for MoE prompt processing on sm_60 (PXA_P60_MOE_GEMM).
// sm_60 has no dp4a, so int8 MMQ is emulated, and the incumbent prefill path on these files is a per-expert loop of
// dequantize -> cuBLAS hgemm with gather/scatter copies (273k launches per 4096-token prefill on Qwen3.6-35B-A3B).
// Here weights are dequantized to fp16 in shared memory and multiplied with HFMA2, fp32 accumulation per 64-wide K step:
//   1. pairs (token, slot) bucketed by expert on the device (count, scan, scatter) - no host sync
//   2. the src1 rows of the pairs gathered once in that order, converted to fp16
//   3. one block turns the expert bounds into a list of tiles (expert, first pair, pairs)
//   4. a block computes BM rows x BN pairs of one expert (BN 64 / 32 / 16 from the average pairs per expert, 128-row
//      tiles for 64-pair tiles; q4_K / q5_K / q6_K)
// GGML_OP_MOE_FUSED_UP_GATE shares one bucketing and one fp16 gather between the two matrices (silu(gate) * up epilogue).
// Every output is summed by one thread in a fixed K order, so the result does not depend on the pair order inside an
// expert's bucket (the scatter uses atomics): deterministic.
//
// Portions:
//
//   Copyright: see NOTICE at the root of this tree.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
//   the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//   THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//   IN THE SOFTWARE.
#include "../common.cuh"
#include "p60-mega.h"

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int BM = 64, BK = 64, KP = BK / 2, NTH = 256;

struct moe_job { int expert, p0, np; };

__device__ __forceinline__ void p60_scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// ---- device-side bucketing of the (token, slot) pairs by expert ----
__global__ void moe_count(const int32_t * __restrict__ ids, int64_t si1, int n_used, int n_pairs, int * __restrict__ cnt) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_pairs) {
        const int t = p / n_used, s = p % n_used;
        atomicAdd(&cnt[ids[t * si1 + s]], 1);
    }
}

// one block of nthr >= n_experts threads: bounds = exclusive scan of cnt (+ total), cursor = bounds
__global__ void moe_scan(const int * __restrict__ cnt, int n_experts, int * __restrict__ bounds, int * __restrict__ cursor) {
    __shared__ int s[1024];
    const int e = threadIdx.x;
    const int v = e < n_experts ? cnt[e] : 0;
    s[e] = v;
    __syncthreads();
    for (int o = 1; o < blockDim.x; o <<= 1) {
        const int a = e >= o ? s[e - o] : 0;
        __syncthreads();
        s[e] += a;
        __syncthreads();
    }
    if (e < n_experts) {
        bounds[e] = s[e] - v;
        cursor[e] = s[e] - v;
    }
    if (e == blockDim.x - 1) {
        bounds[n_experts] = s[e];
    }
}

__global__ void moe_scatter(const int32_t * __restrict__ ids, int64_t si1, int n_used, int n_pairs, int * __restrict__ cursor,
        int32_t * __restrict__ pair_of) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_pairs) {
        const int t = p / n_used, s = p % n_used;
        const int pos = atomicAdd(&cursor[ids[t * si1 + s]], 1);
        pair_of[pos] = p;
    }
}

// src1 rows of the sorted pairs -> fp16 [sorted pair][K]; pair p = t * n_used + slot reads src1[t][slot % ne11]
__global__ void moe_gather_f16(const char * __restrict__ src1, int64_t nb11, int64_t nb12, int ne11, int n_used,
        const int32_t * __restrict__ pair_of, int K, half2 * __restrict__ xs) {
    const int q = blockIdx.x;
    const int p = pair_of[q];
    const int t = p / n_used, s = p % n_used;
    const float2 * x = (const float2 *) (src1 + (int64_t) t * nb12 + (int64_t) (s % ne11) * nb11);
    half2 * o = xs + (int64_t) q * (K / 2);
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
        const float2 v = x[i];
        o[i] = __floats2half2_rn(v.x, v.y);
    }
}

// expert bounds -> tiles of up to BN pairs; one block of nthr >= n_experts (<= 1024) threads
__global__ void moe_jobs(const int32_t * __restrict__ bounds, int n_experts, int BN, moe_job * __restrict__ jobs, int * __restrict__ njobs) {
    __shared__ int cnt[1024];
    const int e = threadIdx.x;
    const int n = e < n_experts ? bounds[e + 1] - bounds[e] : 0;
    const int nt = (n + BN - 1) / BN;
    cnt[e] = nt;
    __syncthreads();
    // inclusive scan (Hillis-Steele)
    for (int o = 1; o < blockDim.x; o <<= 1) {
        const int v = e >= o ? cnt[e - o] : 0;
        __syncthreads();
        cnt[e] += v;
        __syncthreads();
    }
    const int first = cnt[e] - nt;
    for (int t = 0; t < nt; t++) {
        jobs[first + t] = { e, bounds[e] + t * BN, min(BN, n - t * BN) };
    }
    if (e == blockDim.x - 1) {
        *njobs = cnt[e];
    }
}

// TYPE: GGML_TYPE_Q4_K (144 B / 256), GGML_TYPE_Q5_K (176 B / 256) or GGML_TYPE_Q6_K (210 B / 256)
// BN: pairs per tile (64, 32 or 16: fewer when experts get few tokens, e.g. short prompts)
// BMT: rows per tile (64 or 128; 128 halves the weight-side shared-memory loads per HFMA2)
// output row of sorted pair q: dst + pair_of[q] * nb1 (dst contiguous over (slot, token))
template <int TYPE, int BN, int BMT>
__global__ void __launch_bounds__(NTH, 2) moe_gemm_f16(const uint8_t * __restrict__ W, int64_t row_bytes, int64_t exp_bytes, int K,
        const half2 * __restrict__ X, const moe_job * __restrict__ jobs, const int * __restrict__ njobs,
        const int32_t * __restrict__ ids_dst, char * __restrict__ dst, int64_t nb1) {
    __shared__ __align__(16) half2 AP[KP][BMT];  // weights: AP[kp][row]
    __shared__ __align__(16) half2 BP[KP][BN];   // activations: BP[kp][pair]
    if ((int) blockIdx.x >= *njobs) {
        return;
    }
    const moe_job jb = jobs[blockIdx.x];
    const int m0 = blockIdx.y * BMT;
    constexpr int PPT = BN / 16;           // pairs per thread
    constexpr int RPT = BMT / 16;          // rows per thread
    constexpr int TPP = NTH / BN, KPT = KP / TPP; // activation loader: threads per pair, k-pairs per thread
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const uint8_t * Wexp = W + (int64_t) jb.expert * exp_bytes;
    constexpr int SBB = TYPE == GGML_TYPE_Q4_K ? 144 : TYPE == GGML_TYPE_Q5_K ? 176 : 210, QS = TYPE == GGML_TYPE_Q4_K ? 16 : 48;

    float acc[RPT][PPT];
#pragma unroll
    for (int i = 0; i < RPT; i++)
#pragma unroll
        for (int j = 0; j < PPT; j++) acc[i][j] = 0.f;

    const int lr = tid >> 2, lp = tid & 3;          // weight loader: row, 8 bytes of qs (16 weights)
    const int bpair = tid / TPP, bq = tid % TPP;    // activation loader: pair, KPT k-pairs

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int rr = 0; rr < BMT / 64; rr++) {
            if constexpr (TYPE == GGML_TYPE_Q6_K) {
                // 210 B superblocks (2 B aligned): step j covers y[64 j ..]: half n = j / 2, upper nibbles / qh bits 4-7
                // for odd j; first 32 weights from ql[64 n + l], the next 32 from ql[64 n + 32 + l], l = 8 lp + e
                const int sb = k0 / 256, j = (k0 % 256) / 64, n = j >> 1, odd = j & 1;
                const uint8_t * blk = Wexp + (int64_t) (m0 + lr + 64 * rr) * row_bytes + sb * SBB;
                const unsigned short * ql0 = (const unsigned short *) (blk + 64 * n + 8 * lp);
                const unsigned short * ql1 = (const unsigned short *) (blk + 64 * n + 32 + 8 * lp);
                const unsigned short * qhp = (const unsigned short *) (blk + 128 + 32 * n + 8 * lp);
                uint8_t a0[8], a1[8], hh[8];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const unsigned short u0 = ql0[e], u1 = ql1[e], uh = qhp[e];
                    a0[2 * e] = u0 & 0xFF; a0[2 * e + 1] = u0 >> 8;
                    a1[2 * e] = u1 & 0xFF; a1[2 * e + 1] = u1 >> 8;
                    hh[2 * e] = uh & 0xFF; hh[2 * e + 1] = uh >> 8;
                }
                const int is = lp >> 1;
                const int8_t * sc = (const int8_t *) (blk + 192 + 8 * n + 4 * odd);
                const float d = __half2float(*(const half *) (blk + 208));
                const float dl = d * sc[is], dh = d * sc[is + 2];
                const int sh = 4 * odd;
                float lo[8], hi[8];
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int q0 = ((a0[e] >> sh) & 0xF) | (((hh[e] >> sh) & 3) << 4);
                    const int q1 = ((a1[e] >> sh) & 0xF) | (((hh[e] >> (sh + 2)) & 3) << 4);
                    lo[e] = dl * (q0 - 32);
                    hi[e] = dh * (q1 - 32);
                }
#pragma unroll
                for (int e = 0; e < 8; e += 2) {
                    AP[(8 * lp + e) / 2][lr + 64 * rr]      = __floats2half2_rn(lo[e], lo[e + 1]);
                    AP[(32 + 8 * lp + e) / 2][lr + 64 * rr] = __floats2half2_rn(hi[e], hi[e + 1]);
                }
            } else {
                const int sb = k0 / 256, j = (k0 % 256) / 64;
                const uint8_t * blk = Wexp + (int64_t) (m0 + lr + 64 * rr) * row_bytes + sb * SBB;
                const float d = __half2float(*(const half *) blk), dmin = __half2float(*(const half *) (blk + 2));
                int s0, mn0, s1, mn1;
                p60_scale_min_k4(2 * j, blk + 4, s0, mn0);
                p60_scale_min_k4(2 * j + 1, blk + 4, s1, mn1);
                const float d0 = d * s0, c0 = dmin * mn0, d1 = d * s1, c1 = dmin * mn1;
                const uint2 q8 = *(const uint2 *) (blk + QS + 32 * j + 8 * lp);
                const uint8_t * qb = (const uint8_t *) &q8;
                unsigned hb0 = 0, hb1 = 0;
                if constexpr (TYPE == GGML_TYPE_Q5_K) {
                    const uint2 h8 = *(const uint2 *) (blk + 16 + 8 * lp);
                    const uint8_t * hbb = (const uint8_t *) &h8;
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        hb0 |= ((hbb[e] >> (2 * j)) & 1u) << e;
                        hb1 |= ((hbb[e] >> (2 * j + 1)) & 1u) << e;
                    }
                }
                float lo[8], hi[8];
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int ql = qb[e] & 0xF, qh = qb[e] >> 4;
                    if constexpr (TYPE == GGML_TYPE_Q5_K) { ql |= ((hb0 >> e) & 1) << 4; qh |= ((hb1 >> e) & 1) << 4; }
                    lo[e] = d0 * ql - c0;
                    hi[e] = d1 * qh - c1;
                }
#pragma unroll
                for (int e = 0; e < 8; e += 2) {
                    AP[(8 * lp + e) / 2][lr + 64 * rr]      = __floats2half2_rn(lo[e], lo[e + 1]);
                    AP[(32 + 8 * lp + e) / 2][lr + 64 * rr] = __floats2half2_rn(hi[e], hi[e + 1]);
                }
            }
        }
        {
            if (bpair < jb.np) {
                const half2 * src = X + (int64_t) (jb.p0 + bpair) * (K / 2) + k0 / 2 + KPT * bq;
                half2 hv[KPT];
                if constexpr (KPT % 4 == 0) {
#pragma unroll
                    for (int e = 0; e < KPT; e += 4) *(uint4 *) &hv[e] = *(const uint4 *) (src + e);
                } else {
#pragma unroll
                    for (int e = 0; e < KPT; e += 2) *(uint2 *) &hv[e] = *(const uint2 *) (src + e);
                }
#pragma unroll
                for (int e = 0; e < KPT; e++) BP[KPT * bq + e][bpair] = hv[e];
            } else {
#pragma unroll
                for (int e = 0; e < KPT; e++) BP[KPT * bq + e][bpair] = __float2half2_rn(0.f);
            }
        }
        __syncthreads();
        half2 h[RPT][PPT];
#pragma unroll
        for (int i = 0; i < RPT; i++)
#pragma unroll
            for (int j = 0; j < PPT; j++) h[i][j] = __float2half2_rn(0.f);
#pragma unroll 8
        for (int kp = 0; kp < KP; kp++) {
            half2 ah[RPT];
#pragma unroll
            for (int i = 0; i < RPT; i += 4) *(uint4 *) &ah[i] = *(const uint4 *) &AP[kp][RPT * ty + i];
            half2 bh[PPT];
            if constexpr (PPT == 4) {
                *(uint4 *) bh = *(const uint4 *) &BP[kp][4 * tx];
            } else if constexpr (PPT == 2) {
                *(uint2 *) bh = *(const uint2 *) &BP[kp][2 * tx];
            } else {
                bh[0] = BP[kp][tx];
            }
#pragma unroll
            for (int i = 0; i < RPT; i++)
#pragma unroll
                for (int j = 0; j < PPT; j++) h[i][j] = __hfma2(ah[i], bh[j], h[i][j]);
        }
#pragma unroll
        for (int i = 0; i < RPT; i++)
#pragma unroll
            for (int j = 0; j < PPT; j++) { const float2 f = __half22float2(h[i][j]); acc[i][j] += f.x + f.y; }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PPT; j++) {
        const int p = PPT * tx + j;
        if (p < jb.np) {
            float * o = (float *) (dst + (int64_t) ids_dst[jb.p0 + p] * nb1) + m0 + RPT * ty;
#pragma unroll
            for (int i = 0; i < RPT; i += 4) *(float4 *) (o + i) = make_float4(acc[i][j], acc[i + 1][j], acc[i + 2][j], acc[i + 3][j]);
        }
    }
}

// dst = silu(dst) * up, contiguous f32
__global__ void moe_glu_silu(float * __restrict__ dst, const float * __restrict__ up, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = dst[i];
        dst[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

int pxa_p60_moe_min_tokens() {
    static const int v = [] { const char * e = getenv("PXA_P60_MOE_GEMM_MIN"); return e ? atoi(e) : 32; }();
    return v;
}

bool pxa_p60_moe_gemm_on(int device) {
    static const bool on = [] {
        const char * e = getenv("PXA_P60_MOE_GEMM");
        const bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_P60_MOE_GEMM: 1 (fp16 expert GEMM for MoE prefill on cc 6.0, >= %d tokens)\n", pxa_p60_moe_min_tokens());
        return v;
    }();
    return on && ggml_cuda_info().devices[device].cc == 600;
}

bool pxa_p60_w_ok(const ggml_tensor * w, int64_t K, int64_t M) {
    if (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q5_K && w->type != GGML_TYPE_Q6_K) return false;
    const int64_t sbb = w->type == GGML_TYPE_Q4_K ? 144 : w->type == GGML_TYPE_Q5_K ? 176 : 210;
    return w->ne[0] == K && w->ne[1] == M && w->ne[3] == 1 && K % 256 == 0 && M % BM == 0 && w->ne[2] <= 1024 &&
        w->nb[1] == (size_t) (K / 256 * sbb) && w->nb[2] == w->nb[1] * M;
}

// members are allocated in declaration order and freed in reverse (the VMM pool is LIFO)
struct moe_sorted {
    ggml_cuda_pool_alloc<int>     cnt;
    ggml_cuda_pool_alloc<int>     cursor;
    ggml_cuda_pool_alloc<int32_t> pair_of;
    ggml_cuda_pool_alloc<int32_t> bounds;
    ggml_cuda_pool_alloc<half2>   xs;
    ggml_cuda_pool_alloc<moe_job> jobs;
    ggml_cuda_pool_alloc<int>     njobs;
    int bn = 64, max_jobs = 0;
};

// bucket the pairs, gather src1 as fp16, build the tile list
void moe_prepare(ggml_backend_cuda_context & ctx, moe_sorted & S, const ggml_tensor * src1, const ggml_tensor * ids,
        int64_t K, int64_t n_experts) {
    cudaStream_t stream = ctx.stream();
    const int n_used = (int) ids->ne[0], n_tokens = (int) ids->ne[1], n_pairs = n_used * n_tokens;
    const int64_t si1 = ids->nb[1] / sizeof(int32_t);
    S.cnt.alloc(ctx.pool(), n_experts + 1);
    S.cursor.alloc(ctx.pool(), n_experts + 1);
    S.pair_of.alloc(ctx.pool(), n_pairs);
    S.bounds.alloc(ctx.pool(), n_experts + 1);
    int * cnt = S.cnt.get(), * cursor = S.cursor.get();
    CUDA_CHECK(cudaMemsetAsync(cnt, 0, (n_experts + 1) * sizeof(int), stream));
    moe_count<<<(n_pairs + 255) / 256, 256, 0, stream>>>((const int32_t *) ids->data, si1, n_used, n_pairs, cnt);
    int nthr = 32;
    while (nthr < n_experts) {
        nthr *= 2;
    }
    moe_scan<<<1, nthr, 0, stream>>>(cnt, (int) n_experts, S.bounds.get(), cursor);
    moe_scatter<<<(n_pairs + 255) / 256, 256, 0, stream>>>((const int32_t *) ids->data, si1, n_used, n_pairs, cursor, S.pair_of.get());
    S.xs.alloc(ctx.pool(), (int64_t) n_pairs * K / 2);
    moe_gather_f16<<<n_pairs, 256, 0, stream>>>((const char *) src1->data, src1->nb[1], src1->nb[2], (int) src1->ne[1], n_used,
        S.pair_of.get(), (int) K, S.xs.get());
    // tile width from the average pairs per expert (short prompts give experts few tokens)
    const int64_t avg = n_pairs / n_experts;
    S.bn = avg >= 40 ? 64 : avg >= 12 ? 32 : 16;
    S.max_jobs = (int) ((n_pairs + S.bn - 1) / S.bn + n_experts);
    S.jobs.alloc(ctx.pool(), S.max_jobs);
    S.njobs.alloc(ctx.pool(), 1);
    moe_jobs<<<1, nthr, 0, stream>>>(S.bounds.get(), (int) n_experts, S.bn, S.jobs.get(), S.njobs.get());
}

void moe_run(ggml_backend_cuda_context & ctx, moe_sorted & S, const ggml_tensor * W, int64_t K, int64_t M, float * out, int64_t nb1) {
    const int bmt = S.bn == 64 && M % 128 == 0 ? 128 : 64;
    const dim3 grid(S.max_jobs, M / bmt);
    using kfn = void (*)(const uint8_t *, int64_t, int64_t, int, const half2 *, const moe_job *, const int *, const int32_t *, char *, int64_t);
#define P60_MOE_GEMM_K(TYPE) { moe_gemm_f16<TYPE, 64, 128>, moe_gemm_f16<TYPE, 64, 64>, moe_gemm_f16<TYPE, 32, 64>, moe_gemm_f16<TYPE, 16, 64> }
    static const kfn kern[3][4] = { P60_MOE_GEMM_K(GGML_TYPE_Q4_K), P60_MOE_GEMM_K(GGML_TYPE_Q5_K), P60_MOE_GEMM_K(GGML_TYPE_Q6_K) };
#undef P60_MOE_GEMM_K
    const int ti = W->type == GGML_TYPE_Q4_K ? 0 : W->type == GGML_TYPE_Q5_K ? 1 : 2;
    kern[ti][S.bn == 64 ? (bmt == 128 ? 0 : 1) : S.bn == 32 ? 2 : 3]<<<grid, NTH, 0, ctx.stream()>>>((const uint8_t *) W->data, W->nb[1], W->nb[2], (int) K,
        S.xs.get(), S.jobs.get(), S.njobs.get(), S.pair_of.get(), (char *) out, nb1);
    CUDA_CHECK(cudaGetLastError());
}

bool moe_shapes_ok(const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int64_t K, int64_t M) {
    const int64_t n_used = ids->ne[0], n_tokens = ids->ne[1];
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
        src1->ne[0] == K && src1->ne[3] == 1 && dst->ne[3] == 1 && src1->nb[0] == sizeof(float) && src1->nb[1] % 8 == 0 && src1->nb[2] % 8 == 0 &&
        (src1->ne[1] == 1 || src1->ne[1] == n_used) && src1->ne[2] == n_tokens &&
        dst->ne[0] == M && dst->ne[1] == n_used && dst->ne[2] == n_tokens && dst->nb[0] == sizeof(float) &&
        dst->nb[1] == (size_t) M * sizeof(float) && dst->nb[2] == dst->nb[1] * n_used && ((uintptr_t) dst->data) % 16 == 0 &&
        n_tokens >= pxa_p60_moe_min_tokens();
}

} // namespace

// MUL_MAT_ID with q4_K / q5_K / q6_K experts, >= PXA_P60_MOE_GEMM_MIN tokens
bool pxa_p60_moe_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!pxa_p60_moe_gemm_on(ctx.device)) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0], * src1 = dst->src[1], * ids = dst->src[2];
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if (!pxa_p60_w_ok(src0, K, M) || !moe_shapes_ok(src1, ids, dst, K, M)) {
        return false;
    }
    moe_sorted S;
    moe_prepare(ctx, S, src1, ids, K, src0->ne[2]);
    moe_run(ctx, S, src0, K, M, (float *) dst->data, dst->nb[1]);
    return true;
}

// MOE_FUSED_UP_GATE (silu, no clamp, no biases): dst = silu(gate b) * (up b)
bool pxa_p60_moe_gemm_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!pxa_p60_moe_gemm_on(ctx.device)) {
        return false;
    }
    const ggml_tensor * up = dst->src[0], * gate = dst->src[1], * src1 = dst->src[2], * ids = dst->src[3];
    if (gate == nullptr || dst->src[4] != nullptr || dst->src[5] != nullptr || dst->op_params[0] != GGML_UNARY_OP_SILU || dst->op_params[2] != 0) {
        return false;
    }
    const int64_t K = up->ne[0], M = up->ne[1];
    if (!pxa_p60_w_ok(up, K, M) || !pxa_p60_w_ok(gate, K, M) || up->ne[2] != gate->ne[2] || !moe_shapes_ok(src1, ids, dst, K, M)) {
        return false;
    }
    moe_sorted S;
    moe_prepare(ctx, S, src1, ids, K, up->ne[2]);
    const int64_t n = ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> tup(ctx.pool(), n);
    moe_run(ctx, S, up,   K, M, tup.get(), dst->nb[1]);
    moe_run(ctx, S, gate, K, M, (float *) dst->data, dst->nb[1]);
    moe_glu_silu<<<(n + 255) / 256, 256, 0, ctx.stream()>>>((float *) dst->data, tup.get(), n);
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "PXA_P60_MOE_GEMM: up/gate FIRED (dev %d, %s, K %lld, M %lld, %lld tokens)\n", ctx.device, ggml_type_name(up->type),
                (long long) K, (long long) M, (long long) ids->ne[1]);
    }
    return true;
}
