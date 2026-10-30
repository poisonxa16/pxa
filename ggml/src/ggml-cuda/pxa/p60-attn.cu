// pxa / PXA kernel suite -- p60-attn.cu: the full-attention half of a Qwen3.5/3.6 MoE layer at decode on sm_60
// (PXA_P60_MEGA_ATTN, under PXA_P60_MEGA). Three launches instead of ~17 kernels:
//   1. a persistent pre kernel: attn_norm, the q (+gate) / k / v projections as one stream of 9216 q8_0 rows (fp16
//      activations with 16-byte loads, HFMA2), q / k RMS norm, M-RoPE (this tree's rope_multi angles, ext_factor 0,
//      theta 0 where the imrope sections do not cover a pair), this token's f16 k / v cache rows, written through the
//      destination pointers of the graph's own CPY nodes;
//   2. flash attention over the cache: one block per KV head and position range holds all 8 query heads of the group,
//      so K / V are read once (HFMA2 chains of 32 products flushed to fp32, fp32 online softmax);
//   3. a persistent post kernel: merge of the KV ranges, sigmoid gate, the output projection in two K halves, residual.
// Generation grid barrier in a module-global __device__ variable (no host epoch). Under a CUDA-graph capture the half
// declines (this token's cache rows move every step). T = 1..6 (decode and speculative verify).
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

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {
namespace p60mt {

constexpr int NT = 512, NW = NT / 32, TM = 6;

struct bar_t {
    unsigned count;
    unsigned gen;
    unsigned pad[30];
};

// one instance per device (a module global): zero at load, no allocation
__device__ bar_t g_attn_bar;

__device__ __forceinline__ void grid_sync(bar_t * bar) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned * vgen = &bar->gen;
        const unsigned g = *vgen;
        __threadfence();
        const unsigned arrived = atomicAdd(&bar->count, 1u);
        if (arrived == gridDim.x - 1) {
            atomicExch(&bar->count, 0u);
            __threadfence();
            atomicAdd(&bar->gen, 1u);
        } else {
            while (*vgen == g) { }
        }
        __threadfence();
    }
    __syncthreads();
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// 8 halves of padding per 64 elements: every 32-element block starts 16 B aligned (uint4 loads of x in dot_q8h), and the
// 8 lanes of a 128-bit load phase still hit disjoint bank quads. Rows of this layout are D + D / 8 long.
__device__ __forceinline__ int px8(int k) { return k + 8 * (k >> 6); }
// groups of 4 stored as [x0, x2, x1, x3] so that a 32-bit weight word turns into half2 pairs (w0, w2), (w1, w3)
__device__ __forceinline__ int pp8(int k) { const int g = k & 3; return px8((k & ~3) | ((g & 1) << 1) | (g >> 1)); }

__device__ __forceinline__ float h2sum(half2 a) { const float2 f = __half22float2(a); return f.x + f.y; }

// one q8_0 block (staged, 34 B, 2 B aligned) against the pre-permuted fp16 rows of T tokens (px8 layout)
template <int T>
__device__ __forceinline__ void dot_q8h(const uint8_t * stage, int blk, int xblk, const __half * x16, int stride, float acc[TM]) {
    const uint8_t * b = stage + 34 * blk;
    const float d = __half2float(*(const half *) b);
    const unsigned short * qs = (const unsigned short *) (b + 2);
    const half2 off = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)); // 1152 = 1024 + 128
    half2 w02[8], w13[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned v = (qs[2 * i] | ((unsigned) qs[2 * i + 1] << 16)) ^ 0x80808080;
        const unsigned a02 = (v & 0x00FF00FF) | 0x64006400u, a13 = ((v >> 8) & 0x00FF00FF) | 0x64006400u;
        w02[i] = __hsub2(*(const half2 *) &a02, off);
        w13[i] = __hsub2(*(const half2 *) &a13, off);
    }
    const int xo = px8(32 * xblk);
#pragma unroll
    for (int t = 0; t < T; t++) {
        const uint4 * xb = (const uint4 *) (x16 + t * stride + xo);
        float f = 0.f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            half2 a0 = __float2half2_rn(0.f), a1 = a0; // two chains: half the dependency depth
#pragma unroll
            for (int j = 2 * h; j < 2 * h + 2; j++) {
                const uint4 x = xb[j];
                a0 = __hfma2(w02[2 * j],     *(const half2 *) &x.x, a0);
                a1 = __hfma2(w13[2 * j],     *(const half2 *) &x.y, a1);
                a0 = __hfma2(w02[2 * j + 1], *(const half2 *) &x.z, a0);
                a1 = __hfma2(w13[2 * j + 1], *(const half2 *) &x.w, a1);
            }
            f += h2sum(__hadd2(a0, a1));
        }
        acc[t] += d * f;
    }
}

// rms norm of T token rows (fp32, global) into pre-permuted fp16 rows (stride XS), one warp per token
template <int D, int XS>
__device__ __forceinline__ void norm_T(const float * x_in, const float * w, float eps, int T, __half * xs16) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    static_assert(D == 64 * 32, "one warp covers a 2048 row");
    if (warp < T) {
        const int t = warp, k0 = 64 * lane;
        float4 xv[16];
#pragma unroll
        for (int i = 0; i < 16; i++) xv[i] = *(const float4 *) (x_in + t * D + k0 + 4 * i);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) ss += dot4(xv[i], xv[i]);
        const float r = rsqrtf(wsum(ss) / D + eps);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float4 nw = __ldg((const float4 *) (w + k0 + 4 * i));
            const float v[4] = {xv[i].x * r * nw.x, xv[i].y * r * nw.y, xv[i].z * r * nw.z, xv[i].w * r * nw.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                xs16[t * XS + pp8(k0 + 4 * i + c)] = __float2half(v[c]);
            }
        }
    }
    __syncthreads();
}

// warp sums of the first T entries of acc[] at once (butterfly halving the vector per level); afterwards lane l holds
// the sum of token tok = l >> (5 - log2 S); lanes with (l & (32 / S - 1)) == 0 are one writer per token (tok < T)
template <int T>
__device__ __forceinline__ float wsum_multi(const float * acc, int lane, int & tok) {
    constexpr int S = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
    constexpr int LS = S == 1 ? 0 : S == 2 ? 1 : S == 4 ? 2 : 3;
    float v[S];
#pragma unroll
    for (int k = 0; k < S; k++) v[k] = k < T ? acc[k] : 0.f;
#pragma unroll
    for (int lv = 0; lv < LS; lv++) {
        const int o = 16 >> lv, w = S >> lv;
        const bool hi = lane & o;
#pragma unroll
        for (int k = 0; k < w / 2; k++) {
            const float send = hi ? v[k] : v[k + w / 2];
            const float keep = hi ? v[k + w / 2] : v[k];
            v[k] = keep + __shfl_xor_sync(0xffffffff, send, o);
        }
    }
#pragma unroll
    for (int o = 16 >> LS; o > 0; o >>= 1) v[0] += __shfl_xor_sync(0xffffffff, v[0], o);
    tok = lane >> (5 - LS);
    return v[0];
}

template <int T>
__device__ __forceinline__ bool wsum_multi_writer(int lane, int tok) {
    constexpr int S = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
    return (lane & (32 / S - 1)) == 0 && tok < T;
}

} // namespace p60mt

// ================= flash attention over the cache, T query tokens =================
namespace fa {

constexpr int D = 256, G = 8, TILE = 32, NTH = G * 32, KS = D + 8; // KS: padded smem row (halves)
constexpr int MAXSPLIT = 64;

struct Params {
    const float * Q; int64_t q_t, q_h;          // floats: token, head strides
    const half  * K; int64_t k_p, k_h;          // halves: position, head strides
    const half  * V; int64_t v_p, v_h;
    const half  * mask; int64_t m_t;            // halves: token stride (row of positions)
    int L, chunk, nsplit;
    float scale;
    float * part;                               // [nsplit][kvh][T][G][D]
    float2 * ml;                                // [nsplit][kvh][T][G] (max, sum)
};

template <int T>
__global__ void __launch_bounds__(NTH, 1) fa_t_kernel(const Params P) {
    __shared__ __align__(16) half qs[T * G][D];
    __shared__ __align__(16) half kv[TILE][KS];
    __shared__ __align__(16) half ps[T * G][TILE];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5; // warp w = query head within the group
    const int split = blockIdx.x, kvh = blockIdx.y;
    const int p0 = split * P.chunk, p1 = min(P.L, p0 + P.chunk);

    for (int i = threadIdx.x; i < T * G * D / 2; i += NTH) {
        const int r = i / (D / 2), d = 2 * (i % (D / 2)), t = r / G, j = r % G;
        const float2 q = *(const float2 *) (P.Q + t * P.q_t + (kvh * G + j) * P.q_h + d);
        *(half2 *) &qs[r][d] = __floats2half2_rn(q.x, q.y);
    }
    const half * Kh = P.K + kvh * P.k_h;
    const half * Vh = P.V + kvh * P.v_h;

    float m[T], l[T], acc[T][8];
#pragma unroll
    for (int t = 0; t < T; t++) {
        m[t] = -INFINITY; l[t] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[t][i] = 0.f;
    }
    // tile loads: 32 rows x 32 uint4, thread i takes chunks i, i + 256, i + 512, i + 768
    auto ld_tile = [&](const half * base, int64_t stride, int p, uint4 * reg) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int idx = threadIdx.x + NTH * c, row = idx / 32, col = idx % 32;
            reg[c] = p + row < p1 ? __ldg((const uint4 *) (base + (int64_t) (p + row) * stride) + col) : make_uint4(0, 0, 0, 0);
        }
    };
    auto st_tile = [&](const uint4 * reg) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int idx = threadIdx.x + NTH * c, row = idx / 32, col = idx % 32;
            *(uint4 *) &kv[row][8 * col] = reg[c];
        }
    };
    uint4 kreg[4], vreg[4];
    if (p0 < p1) ld_tile(Kh, P.k_p, p0, kreg);
    for (int p = p0; p < p1; p += TILE) {
        __syncthreads(); // previous tile's P.V done with kv
        st_tile(kreg);
        ld_tile(Vh, P.v_p, p, vreg);
        __syncthreads();
        // ---- scores: lane = position, warp w: rows (t, w) ----
        float s[T];
        {
            float f[T];
            half2 a[T];
#pragma unroll
            for (int t = 0; t < T; t++) { f[t] = 0.f; a[t] = __float2half2_rn(0.f); }
#pragma unroll 4
            for (int d = 0; d < D; d += 8) {
                const uint4 k4 = *(const uint4 *) &kv[lane][d];
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const uint4 q4 = *(const uint4 *) &qs[t * G + w][d];
                    a[t] = __hfma2(*(const half2 *) &k4.x, *(const half2 *) &q4.x, a[t]);
                    a[t] = __hfma2(*(const half2 *) &k4.y, *(const half2 *) &q4.y, a[t]);
                    a[t] = __hfma2(*(const half2 *) &k4.z, *(const half2 *) &q4.z, a[t]);
                    a[t] = __hfma2(*(const half2 *) &k4.w, *(const half2 *) &q4.w, a[t]);
                }
                if ((d & 31) == 24) { // flush every 32 elements
#pragma unroll
                    for (int t = 0; t < T; t++) { const float2 v = __half22float2(a[t]); f[t] += v.x + v.y; a[t] = __float2half2_rn(0.f); }
                }
            }
            const int pos = p + lane;
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float mk = pos < p1 ? __half2float(P.mask[t * P.m_t + pos]) : -INFINITY;
                s[t] = f[t] * P.scale + mk;
            }
        }
        // ---- online softmax per row (state replicated in all lanes) ----
        float alpha[T];
#pragma unroll
        for (int t = 0; t < T; t++) {
            float mt = s[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, o));
            const float mn = fmaxf(m[t], mt);
            const float pr = mn == -INFINITY ? 0.f : __expf(s[t] - mn);
            alpha[t] = mn == -INFINITY ? 1.f : __expf(m[t] - mn);
            float sum = pr;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
            l[t] = l[t] * alpha[t] + sum;
            m[t] = mn;
            ps[t * G + w][lane] = __float2half(pr);
        }
        __syncthreads(); // K reads done, probabilities visible
        st_tile(vreg);
        if (p + TILE < p1) ld_tile(Kh, P.k_p, p + TILE, kreg);
        __syncthreads();
        // ---- P.V: lane owns d = 8 lane .. 8 lane + 7 ----
        {
            half2 a[T][4];
#pragma unroll
            for (int t = 0; t < T; t++)
#pragma unroll
                for (int i = 0; i < 4; i++) a[t][i] = __float2half2_rn(0.f);
#pragma unroll 8
            for (int j = 0; j < TILE; j++) {
                const uint4 v4 = *(const uint4 *) &kv[j][8 * lane];
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const half2 pp = __half2half2(ps[t * G + w][j]);
                    a[t][0] = __hfma2(*(const half2 *) &v4.x, pp, a[t][0]);
                    a[t][1] = __hfma2(*(const half2 *) &v4.y, pp, a[t][1]);
                    a[t][2] = __hfma2(*(const half2 *) &v4.z, pp, a[t][2]);
                    a[t][3] = __hfma2(*(const half2 *) &v4.w, pp, a[t][3]);
                }
            }
#pragma unroll
            for (int t = 0; t < T; t++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float2 v = __half22float2(a[t][i]);
                    acc[t][2 * i]     = acc[t][2 * i]     * alpha[t] + v.x;
                    acc[t][2 * i + 1] = acc[t][2 * i + 1] * alpha[t] + v.y;
                }
        }
    }
    const int nkvh = gridDim.y;
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int64_t row = (((int64_t) split * nkvh + kvh) * T + t) * G + w;
        float4 * o = (float4 *) (P.part + row * D + 8 * lane);
        o[0] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
        o[1] = make_float4(acc[t][4], acc[t][5], acc[t][6], acc[t][7]);
        if (lane == 0) P.ml[row] = make_float2(m[t], l[t]);
    }
}

// KV ranges of whole tiles: about one block per SM in total
int fa_splits(int nsm, int nkvh, int L, int & chunk) {
    const int ntile = (L + TILE - 1) / TILE;
    int nsplit = std::max(1, std::min({ nsm / nkvh, ntile, MAXSPLIT }));
    chunk = ((ntile + nsplit - 1) / nsplit) * TILE;
    return (L + chunk - 1) / chunk;
}

} // namespace fa

// ================= attention half =================
namespace at {
using namespace p60mt;

constexpr int D = 2048, NQ = 16, NKV = 2, HD = 256, QF = NQ * 2 * HD, KVN = NKV * HD, O = NQ * HD;
constexpr int XS = D + D / 8, HALF = 1088, Q8_ROW_D = D / 32 * 34, Q8_ROW_O = O / 32 * 34;
constexpr int NROW1 = QF + 2 * KVN; // q (+gate), k, v rows

struct PreParams {
    pxa_p60_attn_args a;
    float theta_scale;
    float * qf;   // [T][8192] q / gate projection
    float * kvn;  // [T][1024] k then v projection
    float * q;    // [T][16][256] normed, rotated q
};

struct PostParams {
    const float * part; const float2 * ml; int nsplit;
    const float * qf;           // gate
    const uint8_t * wo;
    const float * x_in; float * x_out;
    float * o;                  // [T][4096] gated attention output
    float * y;                  // [2][T][2048] output projection halves
};

// rotation of pair p (0 .. n_dims/2 - 1) of token t: this tree's rope_multi with ext_factor = 0 (theta 0 where the
// imrope sections do not cover the pair)
__device__ __forceinline__ void rope_cs(const pxa_p60_attn_args & A, float theta_scale, int T, int t, int p, float & c, float & s) {
    const int sect_dims = A.sections[0] + A.sections[1] + A.sections[2] + A.sections[3];
    const int sec_w = A.sections[0] + A.sections[1];
    const int sector = p % sect_dims;
    float theta_base = 0.0f;
    const float tp = powf(theta_scale, (float) p);
    if (A.imrope) {
        if      (sector % 3 == 1 && sector < 3 * A.sections[1]) theta_base = A.pos[t + T * 1] * tp;
        else if (sector % 3 == 2 && sector < 3 * A.sections[2]) theta_base = A.pos[t + T * 2] * tp;
        else if (sector % 3 == 0 && sector < 3 * A.sections[0]) theta_base = A.pos[t] * tp;
    } else {
        if      (sector < A.sections[0])                theta_base = A.pos[t] * tp;
        else if (sector < sec_w)                        theta_base = A.pos[t + T * 1] * tp;
        else if (sector < sec_w + A.sections[2])        theta_base = A.pos[t + T * 2] * tp;
        else                                            theta_base = A.pos[t + T * 3] * tp;
    }
    const float theta = A.freq_scale * theta_base;
    c = cosf(theta) * A.attn_factor;
    s = sinf(theta) * A.attn_factor;
}

template <int T>
__global__ void __launch_bounds__(NT, 1) attn_pre_kernel(const PreParams P) {
    __shared__ __align__(16) __half xs16[TM * XS];      // P0-P1: normed x; P2: per-warp fp32 head rows
    __shared__ __align__(16) uint8_t stage[NW][HALF];
    const pxa_p60_attn_args & A = P.a;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, nwg = NW * gridDim.x;

    norm_T<D, XS>(A.x_in, A.attn_norm, A.eps_norm, T, xs16);

    // ---- P1: q (+gate), k, v rows streamed in 1088 B chunks, next two chunks in registers ----
    {
        const int nt = gw < NROW1 ? 2 * ((NROW1 - gw + nwg - 1) / nwg) : 0;
        auto rowp = [&](int r) {
            return (const uint8_t *) (r < QF ? A.wq + (size_t) r * Q8_ROW_D : r < QF + KVN ? A.wk + (size_t) (r - QF) * Q8_ROW_D
                                                                                            : A.wv + (size_t) (r - QF - KVN) * Q8_ROW_D);
        };
        auto ld = [&](int c, int4 * bb) {
            const int4 * p = (const int4 *) (rowp(gw + (c >> 1) * nwg) + (c & 1) * HALF);
            bb[0] = __ldg(p + lane); bb[1] = __ldg(p + 32 + lane); if (lane < 4) bb[2] = __ldg(p + 64 + lane);
        };
        int4 b0[3], b1[3];
        if (nt > 0) ld(0, b0);
        if (nt > 1) ld(1, b1);
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; t++) acc[t] = 0.f;
        for (int c = 0; c < nt; c++) {
            int4 * st = (int4 *) stage[warp];
            st[lane] = b0[0]; st[32 + lane] = b0[1]; if (lane < 4) st[64 + lane] = b0[2];
            __syncwarp();
            b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
            if (c + 2 < nt) ld(c + 2, b1);
            dot_q8h<T>(stage[warp], lane, 32 * (c & 1) + lane, xs16, XS, acc);
            __syncwarp();
            if (c & 1) {
                const int r = gw + (c >> 1) * nwg;
                int t;
                const float v = wsum_multi<T>(acc, lane, t);
#pragma unroll
                for (int k = 0; k < TM; k++) acc[k] = 0.f;
                if (wsum_multi_writer<T>(lane, t)) {
                    if (r < QF) P.qf[t * QF + r] = v;
                    else        P.kvn[t * 2 * KVN + r - QF] = v;
                }
            }
        }
    }
    grid_sync(&g_attn_bar);

    // ---- P2: per (token, head): q / k RMS norm + rotation, k / v cache rows; one warp per unit ----
    float * hrow = (float *) xs16 + warp * HD;
    const int nunit = T * (NQ + 2 * NKV);
    for (int u = gw; u < nunit; u += nwg) {
        const int t = u / (NQ + 2 * NKV), h = u % (NQ + 2 * NKV);
        float v[8];
        if (h < NQ + NKV) {
            const float * src = h < NQ ? P.qf + t * QF + h * 2 * HD : P.kvn + t * 2 * KVN + (h - NQ) * HD;
            const float * nw  = h < NQ ? A.q_norm : A.k_norm;
            const float  eps  = h < NQ ? A.eps_q : A.eps_k;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) { v[i] = __ldcg(src + 8 * lane + i); ss += v[i] * v[i]; }
            const float inv = rsqrtf(wsum(ss) / HD + eps);
#pragma unroll
            for (int i = 0; i < 8; i++) v[i] = v[i] * inv * __ldg(nw + 8 * lane + i);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 8; i++) hrow[8 * lane + i] = v[i];
            __syncwarp();
            const int nh = A.n_dims / 2;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = 8 * lane + i;
                if (d < A.n_dims) {
                    const int p = d < nh ? d : d - nh;
                    float c, s;
                    rope_cs(A, P.theta_scale, T, t, p, c, s);
                    const float x0 = hrow[p], x1 = hrow[p + nh];
                    v[i] = d < nh ? x0 * c - x1 * s : x0 * s + x1 * c;
                }
            }
            __syncwarp();
            if (h < NQ) {
#pragma unroll
                for (int i = 0; i < 8; i++) P.q[(t * NQ + h) * HD + 8 * lane + i] = v[i];
            } else {
                half * dst = (half *) A.k_dst + (int64_t) t * A.kv_t + (h - NQ) * HD + 8 * lane;
#pragma unroll
                for (int i = 0; i < 8; i++) dst[i] = __float2half(v[i]);
            }
        } else {
            const int kh = h - NQ - NKV;
            const float * src = P.kvn + t * 2 * KVN + KVN + kh * HD + 8 * lane;
            half * dst = (half *) A.v_dst + (int64_t) t * A.kv_t + kh * HD + 8 * lane;
#pragma unroll
            for (int i = 0; i < 8; i++) dst[i] = __float2half(__ldcg(src + i));
        }
    }
}

template <int T>
__global__ void __launch_bounds__(NT, 1) attn_post_kernel(const PostParams P) {
    __shared__ __align__(16) __half xs16[TM * XS];      // one K half of the gated output, all tokens
    __shared__ __align__(16) uint8_t stage[NW][HALF];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, nwg = NW * gridDim.x;
    constexpr int G = fa::G;

    // ---- C1: merge the KV ranges per (token, head), normalize, sigmoid gate ----
    for (int u = gw; u < T * NQ; u += nwg) {
        const int t = u / NQ, h = u % NQ, kvh = h / G, j = h % G;
        auto mli = [&](int s) { return (((int64_t) s * NKV + kvh) * T + t) * G + j; };
        float M = -INFINITY;
        for (int s = lane; s < P.nsplit; s += 32) M = fmaxf(M, __ldcg(&P.ml[mli(s)].x));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
        float S = 0.f, y[8];
#pragma unroll
        for (int i = 0; i < 8; i++) y[i] = 0.f;
        for (int s = 0; s < P.nsplit; s++) {
            const float2 ml = __ldcg(&P.ml[mli(s)]);
            const float w = ml.x == -INFINITY ? 0.f : __expf(ml.x - M);
            S += w * ml.y;
            const float4 * pr = (const float4 *) (P.part + mli(s) * HD + 8 * lane);
            const float4 a = __ldcg(pr), b = __ldcg(pr + 1);
            y[0] += w * a.x; y[1] += w * a.y; y[2] += w * a.z; y[3] += w * a.w;
            y[4] += w * b.x; y[5] += w * b.y; y[6] += w * b.z; y[7] += w * b.w;
        }
        const float inv = S > 0.f ? 1.f / S : 0.f;
        const float * g = P.qf + t * QF + h * 2 * HD + HD + 8 * lane;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float gv = __ldcg(g + i);
            P.o[t * O + h * HD + 8 * lane + i] = (y[i] * inv) * (1.f / (1.f + expf(-gv)));
        }
    }
    grid_sync(&g_attn_bar);

    // ---- C2: output projection, blocks split in two K halves (2048 of the 4096 inputs each) -> partial sums ----
    {
        const int half_ = blockIdx.x & 1, nbh = gridDim.x >> 1, bh = blockIdx.x >> 1;
        if (warp < T) {
            const int t = warp;
            const float * src = P.o + t * O + half_ * D + 64 * lane;
            float4 xv[16];
#pragma unroll
            for (int i = 0; i < 16; i++) xv[i] = __ldcg((const float4 *) (src + 4 * i));
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float vv[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
#pragma unroll
                for (int c = 0; c < 4; c++) xs16[t * XS + pp8(64 * lane + 4 * i + c)] = __float2half(vv[c]);
            }
        }
        __syncthreads();
        const int gwh = warp * nbh + bh, nwh = NW * nbh;
        const int nt = gwh < D ? 2 * ((D - gwh + nwh - 1) / nwh) : 0;
        auto ld = [&](int c, int4 * bb) {
            const int4 * p = (const int4 *) (P.wo + (size_t) (gwh + (c >> 1) * nwh) * Q8_ROW_O + half_ * 2 * HALF + (c & 1) * HALF);
            bb[0] = __ldg(p + lane); bb[1] = __ldg(p + 32 + lane); if (lane < 4) bb[2] = __ldg(p + 64 + lane);
        };
        int4 b0[3], b1[3];
        if (nt > 0) ld(0, b0);
        if (nt > 1) ld(1, b1);
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; t++) acc[t] = 0.f;
        for (int c = 0; c < nt; c++) {
            int4 * st = (int4 *) stage[warp];
            st[lane] = b0[0]; st[32 + lane] = b0[1]; if (lane < 4) st[64 + lane] = b0[2];
            __syncwarp();
            b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
            if (c + 2 < nt) ld(c + 2, b1);
            dot_q8h<T>(stage[warp], lane, 32 * (c & 1) + lane, xs16, XS, acc);
            __syncwarp();
            if (c & 1) {
                const int r = gwh + (c >> 1) * nwh;
                int t;
                const float a = wsum_multi<T>(acc, lane, t);
#pragma unroll
                for (int k = 0; k < TM; k++) acc[k] = 0.f;
                if (wsum_multi_writer<T>(lane, t)) P.y[((size_t) half_ * TM + t) * D + r] = a;
            }
        }
    }
    grid_sync(&g_attn_bar);

    // ---- C3: residual ----
    for (int i = blockIdx.x * NT + threadIdx.x; i < T * D; i += gridDim.x * NT) {
        const int t = i / D, r = i % D;
        P.x_out[i] = P.x_in[i] + (__ldcg(P.y + (size_t) t * D + r) + __ldcg(P.y + ((size_t) TM + t) * D + r));
    }
}

template <typename KP>
void attn_coop(ggml_backend_cuda_context & ctx, const void * kernel, KP & kp) {
    const int nblk = ggml_cuda_info().devices[ctx.device].nsm;
    void * args[] = { &kp };
    CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, dim3(nblk), dim3(NT), args, 0, ctx.stream()));
}

} // namespace at
} // namespace

bool pxa_p60_mega_attn(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args, int n_tokens) {
    using namespace at;
    const int T = n_tokens;
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    if (T < 1 || T > TM || nsm % 2 != 0) {
        return false;
    }
    // the cache rows of this token move every step and a captured graph would bake today's destination: the graph's
    // own CPY nodes follow the move through the cpy-dest indirection, so under capture the half keeps the normal path
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cap));
    if (cap != cudaStreamCaptureStatusNone) {
        return false;
    }
    int chunk;
    const int nsplit = fa::fa_splits(nsm, NKV, args.n_kv, chunk);
    const size_t n_qf = (size_t) TM * QF, n_kvn = (size_t) TM * 2 * KVN, n_q = (size_t) TM * O, n_o = (size_t) TM * O,
                 n_y = (size_t) 2 * TM * D, n_part = (size_t) nsplit * NKV * T * fa::G * HD;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_qf + n_kvn + n_q + n_o + n_y + n_part);
    ggml_cuda_pool_alloc<float2> ml(ctx.pool(), (size_t) nsplit * NKV * T * fa::G);
    float * qf = scratch.get(), * kvn = qf + n_qf, * q = kvn + n_kvn, * o = q + n_q, * y = o + n_o, * part = y + n_y;

    PreParams pre = {};
    pre.a = args;
    pre.theta_scale = powf(args.freq_base, -2.0f / args.n_dims);
    pre.qf = qf; pre.kvn = kvn; pre.q = q;
    static const void * const kpre[TM] = { (const void *) attn_pre_kernel<1>, (const void *) attn_pre_kernel<2>, (const void *) attn_pre_kernel<3>,
                                           (const void *) attn_pre_kernel<4>, (const void *) attn_pre_kernel<5>, (const void *) attn_pre_kernel<6> };
    attn_coop(ctx, kpre[T - 1], pre);

    fa::Params fp = {};
    fp.Q = q; fp.q_t = NQ * HD; fp.q_h = HD;
    fp.K = (const half *) args.K; fp.k_p = args.k_p; fp.k_h = args.k_h;
    fp.V = (const half *) args.V; fp.v_p = args.v_p; fp.v_h = args.v_h;
    fp.mask = (const half *) args.mask; fp.m_t = args.mask_row;
    fp.L = args.n_kv; fp.chunk = chunk; fp.nsplit = nsplit; fp.scale = args.scale;
    fp.part = part; fp.ml = ml.get();
    const dim3 fgrid(nsplit, NKV);
    switch (T) {
        case 1: fa::fa_t_kernel<1><<<fgrid, fa::NTH, 0, ctx.stream()>>>(fp); break;
        case 2: fa::fa_t_kernel<2><<<fgrid, fa::NTH, 0, ctx.stream()>>>(fp); break;
        case 3: fa::fa_t_kernel<3><<<fgrid, fa::NTH, 0, ctx.stream()>>>(fp); break;
        case 4: fa::fa_t_kernel<4><<<fgrid, fa::NTH, 0, ctx.stream()>>>(fp); break;
        case 5: fa::fa_t_kernel<5><<<fgrid, fa::NTH, 0, ctx.stream()>>>(fp); break;
        case 6: fa::fa_t_kernel<6><<<fgrid, fa::NTH, 0, ctx.stream()>>>(fp); break;
    }

    PostParams post = {};
    post.part = part; post.ml = ml.get(); post.nsplit = nsplit;
    post.qf = qf; post.wo = (const uint8_t *) args.wo;
    post.x_in = args.x_in; post.x_out = args.x_out;
    post.o = o; post.y = y;
    static const void * const kpost[TM] = { (const void *) attn_post_kernel<1>, (const void *) attn_post_kernel<2>, (const void *) attn_post_kernel<3>,
                                            (const void *) attn_post_kernel<4>, (const void *) attn_post_kernel<5>, (const void *) attn_post_kernel<6> };
    attn_coop(ctx, kpost[T - 1], post);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
