#pragma once

//
// Shared body of the two sm_60 quantized-KV flash-attention routes (2026-09-27):
//   PXA_FA_QKV_DIRECT  narrow nodes, query width 1..8 (decode, speculative verify)  fattn-qkv-direct.cu
//   PXA_FA_QKV_TILE    wide nodes, query width > 8 (prefill ubatch, wide verify)    fattn-qkv-tile.cu
//
// One block per (column tile, KV head, KV split) serves the G query heads of one GQA group times
// NC query columns (R = G*NC rows), so every K/V byte is fetched once per group and column tile.
//
// Data path per KV tile of T keys:
//   - global -> registers: the tile's raw q4_0 / q8_0 rows (144 / 272 B per key and head, both
//     multiples of 16) are read as 16-byte vectors. The NEXT tile is requested while the current
//     one is consumed (one register set, alternating K and V: V(t) is in flight during the QK pass
//     of tile t, K(t+1) during its softmax and P.V), so the loads overlap compute.
//   - registers -> shared: stored verbatim, one key row per (RB/4 + 1) words; the odd word stride
//     keeps the lane-per-key reads of the QK pass conflict-free. Nothing is repacked: the compute
//     passes read the block scale and the quants from the raw layout (an even block starts on a
//     word, an odd block two bytes into one; both resolve at compile time).
//   - K side: a nibble (or a biased int8) is OR-ed into the mantissa of 1024.0h by one byte-permute
//     and 1024 (+ the type's offset) is subtracted in half2, which gives the exact integer quant as
//     a half2 pair. The 32 products of one quant block are summed in two half2 chains of 8 and
//     folded into the fp32 score once per block, times the block scale.
//   - softmax: fp32 online max / sum per row, one warp per row.
//   - V side: the same byte-permute dequant gives the exact integer quants as half2, which are
//     widened and scaled in fp32 (V exact, where an f16 cache would have rounded it); P.V products
//     and the O accumulators are fp32. (A half2 P.V with fp32 folds every 8 or every 4 keys was
//     implemented first and measured slightly WORSE than the tile-f16 route on max abs error at
//     width 512 / n_kv 16384 (1.12e-4 and 1.08e-4 vs 9.99e-5), so it did not ship; fp32 costs the
//     same instruction count once the fold conversions are counted.)
// No f16 copy of the cache is made, so the node allocates only the split partials.
//
// Split and P.V shape (ctx16k, 2026-09-28). A block's keys are ONE contiguous run of whole tiles,
// [t_beg, t_end) of the node's n_tl tiles, balanced to within one tile across the gridDim.y
// splits (it was a 256-key unit strided by the split count, so at n_kv 16384 over 56 splits eight
// splits carried two units and the rest one: the node ran at twice the mean split). The narrow
// route also lets the launcher split at tile granularity instead of 256 keys. At R <= 8 rows (decode
// width 1 at GQA 4 / 6 / 8) the P.V pass is warp-per-key-slice: each warp owns T/NW keys of the
// tile and every row, each lane eight output dims (one 4-byte quant slice: dims 4s..4s+3 and
// 16+4s..16+4s+3 of block b), so a V byte is dequantized once per block instead of once per row
// group, and the eight warp partials are summed in a fixed order at the end (deterministic).
//
// Scope (the host predicates in the two .cu files): sm_60 (fast fp16, no fp16 MMA), head 256,
// K and V both q4_0 or both q8_0, GQA group 4 / 6 / 8, 16-byte aligned rows, no sinks, no ALiBi,
// no logit softcap, KV length a multiple of 256. Deterministic: fixed tile order, fixed split,
// no atomics.
//

#include "../fattn-vec-common.cuh"

namespace pxa_qkv {

constexpr int NW      = 8;              // warps per block
constexpr int NT      = NW*WARP_SIZE;   // threads per block
constexpr int KV_GRAN = 256;            // keys per split unit (== FATTN_KQ_STRIDE)

template <ggml_type TYPE> struct kv_type;
template <> struct kv_type<GGML_TYPE_Q4_0> { static constexpr int BB = 18; static constexpr int QI = 4; };
template <> struct kv_type<GGML_TYPE_Q8_0> { static constexpr int BB = 34; static constexpr int QI = 8; };

// Compile-time tile plan for R = G*NC rows.
//   RG  row groups in the QK pass (warps sharing a key set), KG = key groups, T = keys per tile.
//   q8_0 keys are twice as wide, so its tiles are half as long at the same row count.
template <int D, int R, ggml_type TYPE> struct plan {
    static constexpr int BB   = kv_type<TYPE>::BB;
    static constexpr int QI   = kv_type<TYPE>::QI;
    static constexpr int BPR  = D/32;                 // quant blocks per head row
    static constexpr int RB   = BPR*BB;               // raw bytes per key row
    static constexpr int CPR  = RB/16;                // 16-byte chunks per key row
    static constexpr int RSW  = RB/4 + 1;             // shared words per key row (odd stride)
    static constexpr int RG   = (R <= 12 ? 2 : 4) * (TYPE == GGML_TYPE_Q8_0 ? 2 : 1);
    static constexpr int KG   = NW/RG;
    static constexpr int T    = WARP_SIZE*KG;
    static constexpr int RPG  = (R + RG - 1)/RG;      // rows per thread, QK pass
    static constexpr int RW   = (R + NW - 1)/NW;      // rows per warp, softmax pass
    static constexpr int NP   = D/2;                  // output dim pairs
    static constexpr int NGRP = NT/NP;                // row groups, P.V pass
    static constexpr int RH   = (R + NGRP - 1)/NGRP;  // rows per thread, P.V pass
    static constexpr int NLD  = (T*CPR + NT - 1)/NT;  // 16-byte loads per thread per tile
    static constexpr int SMEM = R*D*2 + T*RSW*4 + R*T*4 + 3*R*4;
    static constexpr int MINB = (SMEM <= 32*1024 && R > 8) ? 2 : 1;   // R <= 8: the PV8 accumulators need > 128 registers

    static_assert(RB % 16 == 0, "key rows must be whole 16-byte chunks");
    static_assert(RSW % 2 == 1, "odd word stride keeps lane-per-key reads conflict-free");
    static_assert(KV_GRAN % T == 0, "tile must divide the split unit");
    static_assert(T % 8 == 0, "P.V reads P four keys at a time, unrolled by 2");
    static_assert(NT % NP == 0, "P.V pass maps one dim pair per thread");
    static_assert(SMEM <= 48*1024, "static shared memory limit");
};

static __device__ __forceinline__ half2 magic_h2(const int x, const int sel) {
    // byte-permute: {x.byte, 0x64, x.byte', 0x64} -> half2(1024 + byte, 1024 + byte')
    const int h = __byte_perm(x, 0x64, sel);
    return *reinterpret_cast<const half2 *>(&h);
}

template <int NLD>
struct tile_regs {
    uint4 v[NLD];
};

// Request one tile (T keys of one head) from global memory into registers, 16 bytes per load.
template <typename P>
static __device__ __forceinline__ void tile_fetch(tile_regs<P::NLD> & rg, const char * __restrict__ base, const int64_t nb1, const int tid) {
#pragma unroll
    for (int i = 0; i < P::NLD; ++i) {
        const int p = i*NT + tid;
        if ((P::T*P::CPR) % NT == 0 || p < P::T*P::CPR) {
            const int k = p / P::CPR;
            const int c = p % P::CPR;
            rg.v[i] = __ldg(reinterpret_cast<const uint4 *>(base + k*nb1) + c);
        }
    }
}

// Store the fetched tile verbatim into shared memory, RSW words per key row.
template <typename P>
static __device__ __forceinline__ void tile_store(const tile_regs<P::NLD> & rg, int * __restrict__ sKV, const int tid) {
#pragma unroll
    for (int i = 0; i < P::NLD; ++i) {
        const int p = i*NT + tid;
        if ((P::T*P::CPR) % NT == 0 || p < P::T*P::CPR) {
            const int k = p / P::CPR;
            const int c = p % P::CPR;
            int * d = sKV + k*P::RSW + 4*c;
            d[0] = (int) rg.v[i].x;
            d[1] = (int) rg.v[i].y;
            d[2] = (int) rg.v[i].z;
            d[3] = (int) rg.v[i].w;
        }
    }
}

template <int D, int G, int NC, ggml_type TYPE>
__launch_bounds__(NT, (plan<D, G*NC, TYPE>::MINB))
static __global__ void flash_attn_qkv(
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
#if defined(FAST_FP16_AVAILABLE) && (!defined(FP16_MMA_AVAILABLE) || __CUDA_ARCH__ == CC_VOLTA) && !defined(GGML_USE_HIP) // sm_60, and sm_70 narrow (PXA_FA_QKV_DIRECT_VOLTA)
    GGML_UNUSED(sinks); GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1); GGML_UNUSED(n_head_log2);
    GGML_UNUSED(logit_softcap); GGML_UNUSED(ne00); GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne13);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(nb32); GGML_UNUSED(ne12);

    constexpr int R = NC*G;
    using P = plan<D, R, TYPE>;
    constexpr int BPR  = P::BPR;
    constexpr int BB   = P::BB;
    constexpr int QI   = P::QI;
    constexpr int RSW  = P::RSW;
    constexpr int RG   = P::RG;
    constexpr int KG   = P::KG;
    constexpr int T    = P::T;
    constexpr int RPG  = P::RPG;
    constexpr int RW   = P::RW;
    constexpr int NP   = P::NP;
    constexpr int NGRP = P::NGRP;
    constexpr int RH   = P::RH;

    __shared__ __align__(16) half2 sQ[R*(D/2)];
    __shared__ __align__(16) int   sKV[T*RSW];
    __shared__ __align__(16) float sS[R*T];     // scores, then P (fp32)
    __shared__ float sAlpha[R];
    __shared__ float sM[R];
    __shared__ float sL[R];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const int ic0      = blockIdx.x*NC;
    const int ngroups  = ne02/G;
    const int sequence = blockIdx.z / ngroups;
    const int head0    = (blockIdx.z - sequence*ngroups)*G;
    const int head_kv  = head0/G;

    K += nb13*sequence + (int64_t) nb12*head_kv;
    V += nb23*sequence + (int64_t) nb22*head_kv;

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + (int64_t) nb31*ic0) : nullptr;
    const int mask_stride = nb31/(int) sizeof(half);

    // ---- stage the group's query rows, scaled, as half2 --------------------------------------
    for (int idx = tid; idx < R*(D/2); idx += NT) {
        const int r = idx / (D/2);
        const int e = idx % (D/2);
        const int c = r / G;
        const int g = r % G;
        float2 q = make_float2(0.0f, 0.0f);
        if (ic0 + c < ne01) {
            q = ((const float2 *) (Q + (int64_t) nb03*sequence + (int64_t) nb02*(head0 + g) + (int64_t) nb01*(ic0 + c)))[e];
        }
        sQ[idx] = __floats2half2_rn(q.x*scale, q.y*scale);
    }

    float m_run[RW];
    float l_run[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        m_run[i] = -FLT_MAX/2.0f;
        l_run[i] = 0.0f;
    }

    constexpr bool PV8 = R <= 8;              // warp-per-key-slice P.V (see the header)
    float O[PV8 ? 1 : RH][2];
#pragma unroll
    for (int rr = 0; rr < (PV8 ? 1 : RH); ++rr) {
        O[rr][0] = 0.0f;
        O[rr][1] = 0.0f;
    }
    float O8[PV8 ? R : 1][8];
#pragma unroll
    for (int r = 0; r < (PV8 ? R : 1); ++r) {
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            O8[r][d] = 0.0f;
        }
    }
    // PV8 lane map: quant block wb, 4-byte slice ws; the slice's first word and the byte-permute
    // that assembles it (a block that starts two bytes into a word has word-aligned quants).
    const int wb   = lane >> 2;
    const int ws   = lane & 3;
    const int boff = wb*BB;
    const int bodd = (boff >> 1) & 1;
    const int bw0  = boff >> 2;
    const int bwa  = bw0 + ws + bodd;
    const int bsel = bodd ? 0x3210 : 0x5432;

    const half2 off_q = TYPE == GGML_TYPE_Q4_0 ? __floats2half2_rn(1032.0f, 1032.0f) : __floats2half2_rn(1152.0f, 1152.0f);

    // QK pass mapping: lane <-> key, warp -> (key group, row group)
    const int kg = warp % KG;
    const int rg = warp / KG;
    const int kq = kg*WARP_SIZE + lane;

    // P.V pass mapping: thread -> (dim pair, row group). The pair is (b*32 + j, b*32 + j + 16): the
    // two elements one q4_0 byte holds (for q8_0: bytes j and j + 16 of the block).
    const int pp = tid % NP;
    const int pg = tid / NP;
    const int pb = pp / 16;
    const int pj = pp % 16;

    const int k_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;
    // this split's keys: tiles [t_beg, t_end) of the n_tl tiles in [first, k_max) (both multiples of 256)
    const int n_tl  = k_max > first ? (k_max - first)/T : 0;
    const int t_beg = (int) (((int64_t) blockIdx.y*n_tl)/gridDim.y);
    const int t_end = (int) (((int64_t) (blockIdx.y + 1)*n_tl)/gridDim.y);
    const int k_end = first + t_end*T;

    tile_regs<P::NLD> rgs;
    int k0 = first + t_beg*T;
    if (k0 < k_end) {
        tile_fetch<P>(rgs, K + (int64_t) k0*nb11, nb11, tid);
    }

    __syncthreads();

    while (k0 < k_end) {
        // ---- K tile: registers -> shared; request this tile's V -------------------------------
        tile_store<P>(rgs, sKV, tid);
        tile_fetch<P>(rgs, V + (int64_t) k0*nb21, nb21, tid);
        __syncthreads();

        // ---- scores: fp16 products, fp32 fold per quant block ----------------------------------
        {
            float s[RPG];
#pragma unroll
            for (int rr = 0; rr < RPG; ++rr) {
                s[rr] = 0.0f;
            }
            const int * kr = sKV + kq*RSW;
#pragma unroll
            for (int b = 0; b < BPR; ++b) {
                const int off = b*BB;          // compile-time after unrolling
                const int w0  = off >> 2;
                int w[QI + 1];
#pragma unroll
                for (int i = 0; i <= QI; ++i) {
                    w[i] = kr[w0 + i];
                }
                int qs[QI];
                unsigned short d16;
                if (off & 2) {
                    d16 = (unsigned short) (((unsigned int) w[0]) >> 16);
#pragma unroll
                    for (int i = 0; i < QI; ++i) {
                        qs[i] = w[i + 1];
                    }
                } else {
                    d16 = (unsigned short) (((unsigned int) w[0]) & 0xFFFF);
#pragma unroll
                    for (int i = 0; i < QI; ++i) {
                        qs[i] = __byte_perm(w[i], w[i + 1], 0x5432);
                    }
                }
                half2 kv[16];
                if constexpr (TYPE == GGML_TYPE_Q4_0) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int x  = qs[i];
                        const int lo = x & 0x0F0F0F0F;
                        const int hi = (x >> 4) & 0x0F0F0F0F;
                        kv[2*i + 0] = __hsub2(magic_h2(lo, 0x4140), off_q);
                        kv[2*i + 1] = __hsub2(magic_h2(lo, 0x4342), off_q);
                        kv[2*i + 8] = __hsub2(magic_h2(hi, 0x4140), off_q);
                        kv[2*i + 9] = __hsub2(magic_h2(hi, 0x4342), off_q);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int x = qs[i] ^ 0x80808080;
                        kv[2*i + 0] = __hsub2(magic_h2(x, 0x4140), off_q);
                        kv[2*i + 1] = __hsub2(magic_h2(x, 0x4342), off_q);
                    }
                }
                const float dk = __half2float(__ushort_as_half(d16));
#pragma unroll
                for (int rr = 0; rr < RPG; ++rr) {
                    const int r = rg*RPG + rr;
                    if (RPG*RG > R && r >= R) {
                        break;
                    }
                    const uint4 * q4 = reinterpret_cast<const uint4 *>(sQ + r*(D/2) + b*16);
                    half2 qv[16];
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const uint4 u = q4[m];
                        qv[4*m + 0] = *reinterpret_cast<const half2 *>(&u.x);
                        qv[4*m + 1] = *reinterpret_cast<const half2 *>(&u.y);
                        qv[4*m + 2] = *reinterpret_cast<const half2 *>(&u.z);
                        qv[4*m + 3] = *reinterpret_cast<const half2 *>(&u.w);
                    }
                    half2 a0 = __hmul2(qv[0], kv[0]);
                    half2 a1 = __hmul2(qv[8], kv[8]);
#pragma unroll
                    for (int m = 1; m < 8; ++m) {
                        a0 = __hfma2(qv[m], kv[m], a0);
                    }
#pragma unroll
                    for (int m = 9; m < 16; ++m) {
                        a1 = __hfma2(qv[m], kv[m], a1);
                    }
                    const float2 f0 = __half22float2(a0);
                    const float2 f1 = __half22float2(a1);
                    s[rr] = fmaf(dk, (f0.x + f0.y) + (f1.x + f1.y), s[rr]);
                }
            }
#pragma unroll
            for (int rr = 0; rr < RPG; ++rr) {
                const int r = rg*RPG + rr;
                if (RPG*RG > R && r >= R) {
                    break;
                }
                const int c = r / G;
                const float mv = (maskh && ic0 + c < ne01) ? __half2float(maskh[c*mask_stride + k0 + kq]) : 0.0f;
                sS[r*T + kq] = s[rr] + mv;
            }
        }
        __syncthreads();

        // ---- V tile: registers -> shared (the K tile is dead); request the next K tile ----------
        tile_store<P>(rgs, sKV, tid);
        const int kn = k0 + T;
        if (kn < k_end) {
            tile_fetch<P>(rgs, K + (int64_t) kn*nb11, nb11, tid);
        }

        // ---- online softmax, fp32; P back into the row ----------------------------------------
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const int r = warp + NW*i;
            if (RW*NW > R && r >= R) {
                break;
            }
            constexpr int PL = T/WARP_SIZE;
            float x[PL];
            float mx = m_run[i];
#pragma unroll
            for (int j = 0; j < PL; ++j) {
                x[j] = sS[r*T + j*WARP_SIZE + lane];
                mx = fmaxf(mx, x[j]);
            }
            mx = warp_reduce_max(mx);
            const float alpha = expf(m_run[i] - mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PL; ++j) {
                const float pv = expf(x[j] - mx);
                sS[r*T + j*WARP_SIZE + lane] = pv;
                sum += pv;
            }
            sum = warp_reduce_sum(sum);
            l_run[i] = l_run[i]*alpha + sum;
            m_run[i] = mx;
            if (lane == 0) {
                sAlpha[r] = alpha;
            }
        }
        __syncthreads();

        // ---- P.V: exact dequant, fp32 products and accumulators -------------------------------
        if constexpr (PV8) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float a = sAlpha[r];
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    O8[r][d] *= a;
                }
            }
            constexpr int KPW = T/NW;
            const int kb0 = warp*KPW;
            // P is read one key at a time (a broadcast load per row): a float4 of four keys per row
            // cost 4*R registers, which pushed the kernel past 128 and into spills.
#pragma unroll 4
            for (int kk = 0; kk < KPW; ++kk) {
                const int k = kb0 + kk;
                {
                    const int * kr = sKV + k*RSW;
                    const float dv = __half2float(__ushort_as_half(
                        (unsigned short) ((((unsigned int) kr[bw0]) >> (16*bodd)) & 0xFFFF)));
                    float v[8];
                    int x0, x1;
                    if constexpr (TYPE == GGML_TYPE_Q4_0) {
                        const int x = __byte_perm(kr[bwa], kr[bwa + 1], bsel);
                        x0 = x & 0x0F0F0F0F;
                        x1 = (x >> 4) & 0x0F0F0F0F;
                    } else {
                        x0 = __byte_perm(kr[bwa],     kr[bwa + 1], bsel) ^ 0x80808080;
                        x1 = __byte_perm(kr[bwa + 4], kr[bwa + 5], bsel) ^ 0x80808080;
                    }
                    float2 f;
                    f = __half22float2(__hsub2(magic_h2(x0, 0x4140), off_q)); v[0] = f.x*dv; v[1] = f.y*dv;
                    f = __half22float2(__hsub2(magic_h2(x0, 0x4342), off_q)); v[2] = f.x*dv; v[3] = f.y*dv;
                    f = __half22float2(__hsub2(magic_h2(x1, 0x4140), off_q)); v[4] = f.x*dv; v[5] = f.y*dv;
                    f = __half22float2(__hsub2(magic_h2(x1, 0x4342), off_q)); v[6] = f.x*dv; v[7] = f.y*dv;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float pr = sS[r*T + k];
#pragma unroll
                        for (int d = 0; d < 8; ++d) {
                            O8[r][d] = fmaf(pr, v[d], O8[r][d]);
                        }
                    }
                }
            }
        } else {
#pragma unroll
        for (int rr = 0; rr < RH; ++rr) {
            const int r = pg*RH + rr;
            if (RH*NGRP > R && r >= R) {
                break;
            }
            const float a = sAlpha[r];
            O[rr][0] *= a;
            O[rr][1] *= a;
        }
        {
            const char * vb = reinterpret_cast<const char *>(sKV) + pb*BB;
#pragma unroll 2
            for (int k = 0; k < T; k += 4) {
                float2 v[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const char * kb = vb + (k + u)*(RSW*4);
                    const float dv = __half2float(*reinterpret_cast<const half *>(kb));
                    int x;
                    if constexpr (TYPE == GGML_TYPE_Q4_0) {
                        const int by = (int) (unsigned char) kb[2 + pj];
                        x = (by & 0x0F) | ((by & 0xF0) << 12);
                    } else {
                        const int b0 = (int) (unsigned char) kb[2 + pj];
                        const int b1 = (int) (unsigned char) kb[2 + 16 + pj];
                        x = (b0 ^ 0x80) | ((b1 ^ 0x80) << 16);
                    }
                    x |= 0x64006400;
                    const float2 q = __half22float2(__hsub2(*reinterpret_cast<const half2 *>(&x), off_q)); // exact integers
                    v[u] = make_float2(q.x*dv, q.y*dv);                                                  // exact in fp32
                }
#pragma unroll
                for (int rr = 0; rr < RH; ++rr) {
                    const int r = pg*RH + rr;
                    if (RH*NGRP > R && r >= R) {
                        break;
                    }
                    const float4 p4 = *reinterpret_cast<const float4 *>(sS + r*T + k);
                    O[rr][0] = fmaf(p4.x, v[0].x, O[rr][0]); O[rr][1] = fmaf(p4.x, v[0].y, O[rr][1]);
                    O[rr][0] = fmaf(p4.y, v[1].x, O[rr][0]); O[rr][1] = fmaf(p4.y, v[1].y, O[rr][1]);
                    O[rr][0] = fmaf(p4.z, v[2].x, O[rr][0]); O[rr][1] = fmaf(p4.z, v[2].y, O[rr][1]);
                    O[rr][0] = fmaf(p4.w, v[3].x, O[rr][0]); O[rr][1] = fmaf(p4.w, v[3].y, O[rr][1]);
                }
            }
        }
        } // PV8
        __syncthreads();

        k0 = kn;
    }

    // PV8: sum the eight warp partials into warp 0, in warp order, through the dead K/V tile.
    if constexpr (PV8) {
        constexpr int SLOT  = R*D;
        constexpr int NSLOT = (T*RSW)/SLOT;
        static_assert(NSLOT >= 1, "one warp partial must fit the tile buffer");
        float * red = reinterpret_cast<float *>(sKV);
        for (int w1 = 1; w1 < NW; w1 += NSLOT) {
            if (warp >= w1 && warp < w1 + NSLOT) {
                float * sp = red + (warp - w1)*SLOT + lane*8;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    *reinterpret_cast<float4 *>(sp + r*D)     = make_float4(O8[r][0], O8[r][1], O8[r][2], O8[r][3]);
                    *reinterpret_cast<float4 *>(sp + r*D + 4) = make_float4(O8[r][4], O8[r][5], O8[r][6], O8[r][7]);
                }
            }
            __syncthreads();
            if (warp == 0) {
                for (int j = 0; j < NSLOT && w1 + j < NW; ++j) {
                    const float * sp = red + j*SLOT + lane*8;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float4 a = *reinterpret_cast<const float4 *>(sp + r*D);
                        const float4 b = *reinterpret_cast<const float4 *>(sp + r*D + 4);
                        O8[r][0] += a.x; O8[r][1] += a.y; O8[r][2] += a.z; O8[r][3] += a.w;
                        O8[r][4] += b.x; O8[r][5] += b.y; O8[r][6] += b.z; O8[r][7] += b.w;
                    }
                }
            }
            __syncthreads();
        }
    }

    // ---- row statistics -> shared, then write --------------------------------------------------
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const int r = warp + NW*i;
        if (RW*NW > R && r >= R) {
            break;
        }
        if (lane == 0) {
            sM[r] = m_run[i];
            sL[r] = l_run[i];
        }
    }
    __syncthreads();

    if constexpr (PV8) {
        if (warp == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int c = r / G;
                const int g = r % G;
                if (ic0 + c >= ne01) {
                    continue;
                }
                const int64_t row = (((int64_t) sequence*ne01 + ic0 + c)*ne02 + head0 + g)*gridDim.y + blockIdx.y;
                const float inv = gridDim.y == 1 ? 1.0f/sL[r] : 1.0f;
                float * o = dst + row*D + wb*32 + 4*ws;
                *reinterpret_cast<float4 *>(o)      = make_float4(O8[r][0]*inv, O8[r][1]*inv, O8[r][2]*inv, O8[r][3]*inv);
                *reinterpret_cast<float4 *>(o + 16) = make_float4(O8[r][4]*inv, O8[r][5]*inv, O8[r][6]*inv, O8[r][7]*inv);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < (PV8 ? 0 : RH); ++rr) {
        const int r = pg*RH + rr;
        if (RH*NGRP > R && r >= R) {
            break;
        }
        const int c = r / G;
        const int g = r % G;
        if (ic0 + c >= ne01) {
            continue;
        }
        const int64_t row = (((int64_t) sequence*ne01 + ic0 + c)*ne02 + head0 + g)*gridDim.y + blockIdx.y;
        const float inv = gridDim.y == 1 ? 1.0f/sL[r] : 1.0f;
        dst[row*D + pb*32 + pj]      = O[rr][0]*inv;
        dst[row*D + pb*32 + pj + 16] = O[rr][1]*inv;
    }
    if (gridDim.y != 1) {
        for (int r = tid; r < R; r += NT) {
            const int c = r / G;
            const int g = r % G;
            if (ic0 + c < ne01) {
                const int64_t row = (((int64_t) sequence*ne01 + ic0 + c)*ne02 + head0 + g)*gridDim.y + blockIdx.y;
                dst_meta[row] = make_float2(sM[r], sL[r]);
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}

// narrow = the decode / verify route: the launcher may split at tile granularity (a whole tile per
// split at minimum) instead of 256 keys, so short caches still fill the card.
template <int D, int G, int NC, ggml_type TYPE>
static void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool narrow = false) {
    launch_fattn<D, NC, G>(ctx, dst, flash_attn_qkv<D, G, NC, TYPE>, NW, 0, narrow ? plan<D, G*NC, TYPE>::T : KV_GRAN,
                           /*need_f16_K=*/false, /*need_f16_V=*/false);
}

// Everything both routes require of a node except its width, its lever and its precision.
// Returns the GQA group (4, 6 or 8) or 0 when the node is not served.
// volta_ok: the caller is the narrow route and PXA_FA_QKV_DIRECT_VOLTA is armed, so cc 7.0 is served
// too (v100-depth, 2026-09-28: at width 1 over a q4_0 cache the sm_70 fallback is WMMA ncols 8, which
// converts the whole K/V cache to f16 on every decode step; this kernel reads it in place).
static inline int node_group(const ggml_tensor * dst, int cc, const bool volta_ok = false) {
    // sm_60: fast fp16, no fp16 matrix hardware. sm_70 only when the caller allows it.
    const bool volta = volta_ok && cc == CC_VOLTA;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !fast_fp16_available(cc) || (fp16_mma_available(cc) && !volta)) {
        return 0;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    if (!Q || !K || !V || dst->src[4] != nullptr) {
        return 0;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return 0;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return 0;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0)) {
        return 0;
    }
    if (K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2]) {
        return 0;
    }
    const int64_t g = Q->ne[2] / K->ne[2];
    if (g != 4 && g != 6 && g != 8) {
        return 0;
    }
    if (K->ne[1] % KV_GRAN != 0 || V->ne[1] != K->ne[1] || K->ne[3] != V->ne[3]) {
        return 0;
    }
    if (M && M->type != GGML_TYPE_F16) {
        return 0;
    }
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return 0;
    }
    // 16-byte aligned rows: the tile fetch reads 16-byte vectors.
    if (K->nb[1] % 16 || K->nb[2] % 16 || K->nb[3] % 16 || V->nb[1] % 16 || V->nb[2] % 16 || V->nb[3] % 16) {
        return 0;
    }
    if (((uintptr_t) K->data) % 16 || ((uintptr_t) V->data) % 16) {
        return 0;
    }
    if (K->nb[0] != (size_t) ggml_type_size(K->type) || V->nb[0] != (size_t) ggml_type_size(V->type)) {
        return 0;
    }
    // Strides reach the kernel as int32 (as for the vec kernels).
    if (K->nb[1] > INT32_MAX || K->nb[2] > INT32_MAX || V->nb[1] > INT32_MAX || V->nb[2] > INT32_MAX) {
        return 0;
    }
    return (int) g;
}

} // namespace pxa_qkv
