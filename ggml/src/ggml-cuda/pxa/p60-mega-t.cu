// pxa / PXA kernel suite -- p60-mega-t.cu: the MoE half of a Qwen3.5/3.6 MoE layer for 2..6 tokens at once on sm_60
// (speculative verify batches; PXA_P60_MEGA). One cooperative launch: per-token norm into fp16 rows, router + shared
// gate, the union of the tokens' experts computed once per weight row (each row is dequantized once and dotted only
// with the tokens routed to it, HFMA2 with short half2 chains flushed to fp32), expert down split across blocks by
// cost, weighted sum, shared expert, residual. Generation grid barrier in a module-global __device__ variable.
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
namespace mt {


constexpr int NT = 512, NW = NT / 32, TM = 6;

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

// padded smem index: lanes reading at k strides of 64 elements land in different banks
__device__ __forceinline__ int px(int k) { return k + 4 * (k >> 6); }
__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

// copy bytes (multiple of 16) of a row into per-warp smem with 16 B loads
__device__ __forceinline__ void stage_row(uint8_t * dst, const uint8_t * src, int bytes, int lane) {
    for (int i = lane; i < bytes / 16; i += 32) {
        ((int4 *) dst)[i] = __ldg((const int4 *) src + i);
    }
    __syncwarp();
}

// 8 halves of padding per 64 elements instead of 4: every 32-element block starts 16 B aligned (uint4 loads of x in
// dot_q8h), and the 8 lanes of a 128-bit load phase still hit disjoint bank quads. Rows of this layout are D + D / 8 long.
__device__ __forceinline__ int px8(int k) { return k + 8 * (k >> 6); }
__device__ __forceinline__ int pp8(int k) { const int g = k & 3; return px8((k & ~3) | ((g & 1) << 1) | (g >> 1)); }

// position of element k in a pre-permuted fp16 row (then px padding)
__device__ __forceinline__ int pp(int k) { const int g = k & 3; return px((k & ~3) | ((g & 1) << 1) | (g >> 1)); }

// 4 pre-permuted halves (8 B aligned) -> float4 in element order
__device__ __forceinline__ float4 ld4h(const __half * p) {
    const uint2 v = *(const uint2 *) p;
    const float2 a = __half22float2(*(const half2 *) &v.x), b = __half22float2(*(const half2 *) &v.y);
    return make_float4(a.x, b.x, a.y, b.y);
}

__device__ __forceinline__ float h2sum(half2 a) { const float2 f = __half22float2(a); return f.x + f.y; }

struct bar_t {
    unsigned count;
    unsigned gen;
    unsigned pad[30];
};

// one instance per device (a module global): zero at load, no allocation
__device__ bar_t g_mt_bar;

// generation grid barrier: the last block to arrive resets the count and bumps the generation; no host-side epoch
__device__ __forceinline__ void tsync() {
    bar_t * bar = &g_mt_bar;
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

// one q8_0 block (staged, 34 B, 2 B aligned) against the pre-permuted fp16 rows of T tokens (row stride `stride`,
// block xblk of x). int8 weights -> exact half2 pairs once; the half2 accumulator is flushed to fp32 every 8 products.
__device__ __forceinline__ void dot_q8h_T(const uint8_t * stage, int blk, int xblk, const __half * x16, int stride, int T, float acc[TM]) {
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
    const int xo = px(32 * xblk); // a 32-element block never crosses a 64-element padding boundary
#pragma unroll
    for (int t = 0; t < TM; t++) {
        if (t < T) {
            const __half * xb = x16 + t * stride + xo;
            float f = 0.f;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                half2 a = __float2half2_rn(0.f);
#pragma unroll
                for (int i = 4 * h; i < 4 * h + 4; i++) {
                    const uint2 x = *(const uint2 *) (xb + 4 * i);
                    a = __hfma2(w02[i], *(const half2 *) &x.x, a);
                    a = __hfma2(w13[i], *(const half2 *) &x.y, a);
                }
                f += h2sum(a);
            }
            acc[t] += d * f;
        }
    }
}


// compile-time T version of dot_q8h_T for x in the px8 layout: no per-token branches, so the compiler interleaves the
// tokens' chains, and x comes in 16 B loads
template <int T>
__device__ __forceinline__ void dot_q8h(const uint8_t * stage, int blk, int xblk, const __half * x16, int stride, float acc[TM]) {
    const uint8_t * b = stage + 34 * blk;
    const float d = __half2float(*(const half *) b);
    const unsigned short * qs = (const unsigned short *) (b + 2);
    const half2 off = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
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

// rms norm of T token rows (fp32, global) into pre-permuted fp16 rows (stride XS), one warp per token; a lane owns 64
// contiguous elements and issues all its loads up front. Optional fp32 sums of the 32-element sub-blocks.
template <int D, int XS>
__device__ __forceinline__ void norm_T(const float * x_in, const float * w, float eps, int T, __half * xs16, float * xsum) {
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
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float4 nw = __ldg((const float4 *) (w + k0 + 4 * i));
            const float v[4] = {xv[i].x * r * nw.x, xv[i].y * r * nw.y, xv[i].z * r * nw.z, xv[i].w * r * nw.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const __half h = __float2half(v[c]);
                xs16[t * XS + pp8(k0 + 4 * i + c)] = h;
                if (i < 8) s0 += __half2float(h); else s1 += __half2float(h);
            }
        }
        if (xsum) { xsum[t * 64 + 2 * lane] = s0; xsum[t * 64 + 2 * lane + 1] = s1; }
    }
    __syncthreads();
}

// Warp sums of the first T entries of acc[] at once: a butterfly that halves the vector at each of the first log2(S)
// levels (S = T rounded up to a power of 2), 9 shuffles for T = 5..8 instead of 5 T. Afterwards lane l holds the sum of
// token tok = l >> (5 - log2 S); the lanes with (l & (32 / S - 1)) == 0 are one writer per token (check tok < T).
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
__device__ __forceinline__ int wsum_multi_lane(int tok) {
    constexpr int S = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
    return tok * (32 / S);
}

template <int T>
__device__ __forceinline__ bool wsum_multi_writer(int lane, int tok) {
    constexpr int S = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
    return (lane & (32 / S - 1)) == 0 && tok < T;
}


} // namespace mt

namespace moe {
using namespace mt;

constexpr int P3_ROW_COST = 4;
constexpr int D = 2048, NE = 256, NU = 8, FF = 512;
constexpr int Q4K_ROW = D / 256 * 144, Q5K_ROW = FF / 256 * 176, Q8_ROW_D = D / 32 * 34, Q8_ROW_F = FF / 32 * 34, HALF = 1088;
constexpr int Q5K_ROW_D = D / 256 * 176, Q6K_ROW = FF / 256 * 210; // q5_K gate / up rows, q6_K down rows (some layers)
constexpr int XS = D + D / 16;       // padded row length (px) of a token's x
constexpr int AS = FF + FF / 16;     // padded row length (px) of a swiglu vector
constexpr int MAXU = TM * NU;        // union size bound

struct Layer {
    const float * norm; const float * router; const float * sgate;
    const uint8_t * gate; const uint8_t * up; const uint8_t * down; const uint8_t * sg; const uint8_t * su; const uint8_t * sd;
};
struct Params {
    Layer L;
    const float * x_in; // [T][D]
    float * x_out;      // [T][D]
    float eps;
    float * logits;     // [TM][NE + 1]
    float * shgu;       // [TM][2 FF]
    float * egu;        // [MAXU][TM][2 FF]
    float * shout;      // [TM][D]
    float * ypart;      // [TM][NU][D]
};

__device__ __forceinline__ half2 nib2h(unsigned v) { // two nibbles at bits 0-3 and 16-19 -> half2 of their values
    return __hsub2(*(const half2 *) &(const unsigned &) (v | 0x64006400u), __halves2half2(__ushort_as_half(0x6400), __ushort_as_half(0x6400)));
}
__device__ __forceinline__ void scale_min_k4(int j, unsigned s0, unsigned s1, unsigned s2, int & d, int & m) {
    const int sh = 8 * (j & 3);
    const int b0 = (s0 >> sh) & 0xFF, b1 = (s1 >> sh) & 0xFF, b2 = (s2 >> sh) & 0xFF;
    // branchless: lanes of one warp take both cases (j = 2q, 2q+1 with q = lane & 3), a branch would diverge
    const bool lo = j < 4;
    d = lo ? (b0 & 63) : ((b2 & 0xF) | ((b0 >> 6) << 4));
    m = lo ? (b1 & 63) : ((b2 >> 4) | ((b1 >> 6) << 4));
}

struct q4_ld { int4 h, qa, qb; };
__device__ __forceinline__ q4_ld load_q4k(const uint8_t * row, int lane) {
    const uint8_t * b = row + (lane >> 2) * 144;
    const int q = lane & 3;
    return {__ldg((const int4 *) b), __ldg((const int4 *) (b + 16 + 32 * q)), __ldg((const int4 *) (b + 32 + 32 * q))};
}
struct q5_ld { int4 h, ha, hb, qa, qb; };
__device__ __forceinline__ q5_ld load_q5k(const uint8_t * b, int q) {
    return {__ldg((const int4 *) b), __ldg((const int4 *) (b + 16)), __ldg((const int4 *) (b + 32)), __ldg((const int4 *) (b + 48 + 32 * q)), __ldg((const int4 *) (b + 64 + 32 * q))};
}

// ---- per-token dots: weights converted once per lane, then a real loop over only the tokens that use the row ----
// (unrolled 6-token loops get predicated: every row paid for all 6 tokens, ~1.6 are used on average)
struct w64 { half2 l02[8], l13[8], h02[8], h13[8]; float dl, dh, ml, mh; };
__device__ __forceinline__ void prep_q4k(const q4_ld & L, int q, w64 & w) {
    int s0, m0, s1, m1;
    scale_min_k4(2 * q, L.h.y, L.h.z, L.h.w, s0, m0);
    scale_min_k4(2 * q + 1, L.h.y, L.h.z, L.h.w, s1, m1);
    const unsigned v[8] = {(unsigned) L.qa.x, (unsigned) L.qa.y, (unsigned) L.qa.z, (unsigned) L.qa.w, (unsigned) L.qb.x, (unsigned) L.qb.y, (unsigned) L.qb.z, (unsigned) L.qb.w};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        w.l02[i] = nib2h(v[i] & 0x000F000F); w.l13[i] = nib2h((v[i] >> 8) & 0x000F000F);
        w.h02[i] = nib2h((v[i] >> 4) & 0x000F000F); w.h13[i] = nib2h((v[i] >> 12) & 0x000F000F);
    }
    const float d = __half2float(__ushort_as_half(L.h.x & 0xFFFF)), dmin = __half2float(__ushort_as_half(L.h.x >> 16));
    w.dl = d * s0; w.dh = d * s1; w.ml = dmin * m0; w.mh = dmin * m1;
}
__device__ __forceinline__ void prep_q5k(const q5_ld & L, int q, w64 & w) {
    int s0, m0, s1, m1;
    scale_min_k4(2 * q, L.h.y, L.h.z, L.h.w, s0, m0);
    scale_min_k4(2 * q + 1, L.h.y, L.h.z, L.h.w, s1, m1);
    const unsigned v[8]  = {(unsigned) L.qa.x, (unsigned) L.qa.y, (unsigned) L.qa.z, (unsigned) L.qa.w, (unsigned) L.qb.x, (unsigned) L.qb.y, (unsigned) L.qb.z, (unsigned) L.qb.w};
    const unsigned hh[8] = {(unsigned) L.ha.x, (unsigned) L.ha.y, (unsigned) L.ha.z, (unsigned) L.ha.w, (unsigned) L.hb.x, (unsigned) L.hb.y, (unsigned) L.hb.z, (unsigned) L.hb.w};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned lo5 = (v[i] & 0x0F0F0F0F) | (((hh[i] >> (2 * q)) & 0x01010101) << 4);
        const unsigned hi5 = ((v[i] >> 4) & 0x0F0F0F0F) | (((hh[i] >> (2 * q + 1)) & 0x01010101) << 4);
        w.l02[i] = nib2h(lo5 & 0x001F001F); w.l13[i] = nib2h((lo5 >> 8) & 0x001F001F);
        w.h02[i] = nib2h(hi5 & 0x001F001F); w.h13[i] = nib2h((hi5 >> 8) & 0x001F001F);
    }
    const float d = __half2float(__ushort_as_half(L.h.x & 0xFFFF)), dmin = __half2float(__ushort_as_half(L.h.x >> 16));
    w.dl = d * s0; w.dh = d * s1; w.ml = dmin * m0; w.mh = dmin * m1;
}
// one token: x16 = its pre-permuted fp16 row (px layout), base = element index of the 64-weight chunk, sl/sh = fp32
// sums of the chunk's two 32-element sub-blocks
__device__ __forceinline__ float dot1(const w64 & w, const __half * x16, int base, float sl, float sh) {
    half2 al = __float2half2_rn(0.f), ah = al;
    // base is a multiple of 64, so the 64-element chunk has one px() padding offset: one pointer, immediate offsets
    const __half * xb = x16 + px(base);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint2 lo = *(const uint2 *) (xb + 4 * i), hi = *(const uint2 *) (xb + 32 + 4 * i);
        al = __hfma2(w.l02[i], *(const half2 *) &lo.x, al); al = __hfma2(w.l13[i], *(const half2 *) &lo.y, al);
        ah = __hfma2(w.h02[i], *(const half2 *) &hi.x, ah); ah = __hfma2(w.h13[i], *(const half2 *) &hi.y, ah);
    }
    return w.dl * h2sum(al) + w.dh * h2sum(ah) - (w.ml * sl + w.mh * sh);
}

// ---- q6_K (down rows of some layers): 210 B superblocks, so the second superblock of a 420 B row is only 2 B aligned.
// Lane chunk q of a superblock = 64 weights: half n = q / 2 of the superblock, part q % 2: part 0 = low nibbles of
// ql[64 n ..] with qh bits 0-1 / 2-3, part 1 = high nibbles with bits 4-5 / 6-7; 4 int8 scales per lane (16 weights each).
struct q6_ld { unsigned ql[17], qh[9], sc[2]; unsigned short d; };
__device__ __forceinline__ q6_ld load_q6k(const uint8_t * b, int q) {
    const int n = q >> 1, part = q & 1;
    const unsigned * ql = (const unsigned *) ((uintptr_t) (b + 64 * n) & ~(uintptr_t) 3);
    const unsigned * qh = (const unsigned *) ((uintptr_t) (b + 128 + 32 * n) & ~(uintptr_t) 3);
    const unsigned * sc = (const unsigned *) ((uintptr_t) (b + 192 + 8 * n + 4 * part) & ~(uintptr_t) 3);
    const bool mis = ((uintptr_t) b) & 2; // the same for all fields of the superblock
    q6_ld L;
#pragma unroll
    for (int k = 0; k < 16; k++) L.ql[k] = __ldg(ql + k);
    L.ql[16] = mis ? __ldg(ql + 16) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++) L.qh[k] = __ldg(qh + k);
    L.qh[8] = mis ? __ldg(qh + 8) : 0u;
    L.sc[0] = __ldg(sc);
    L.sc[1] = mis ? __ldg(sc + 1) : 0u;
    L.d = __ldg((const unsigned short *) (b + 208));
    return L;
}
struct w66 { half2 l02[8], l13[8], h02[8], h13[8]; float s[4]; };
__device__ __forceinline__ half2 q6h2(unsigned v) { // two 6-bit values at bits 0-5 and 16-21 -> half2 of (value - 32)
    return __hsub2(*(const half2 *) &(const unsigned &) (v | 0x64006400u), __halves2half2(__ushort_as_half(0x6420), __ushort_as_half(0x6420)));
}
__device__ __forceinline__ void prep_q6k(const q6_ld & L, const uint8_t * b, int q, w66 & w) {
    const int part = q & 1;
    const unsigned sh = (((uintptr_t) b) & 2) * 8;
    const int ns = 4 * part;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned v1 = __funnelshift_r(L.ql[i], L.ql[i + 1], sh), v2 = __funnelshift_r(L.ql[i + 8], L.ql[i + 9], sh);
        const unsigned vh = __funnelshift_r(L.qh[i], L.qh[i + 1], sh);
        const unsigned lo1 = (v1 >> ns) & 0x0F0F0F0F, lo2 = (v2 >> ns) & 0x0F0F0F0F;
        const unsigned q1 = lo1 | (((vh >> ns) & 0x03030303) << 4), q2 = lo2 | (((vh >> (ns + 2)) & 0x03030303) << 4);
        w.l02[i] = q6h2(q1 & 0x00FF00FF); w.l13[i] = q6h2((q1 >> 8) & 0x00FF00FF);
        w.h02[i] = q6h2(q2 & 0x00FF00FF); w.h13[i] = q6h2((q2 >> 8) & 0x00FF00FF);
    }
    const unsigned scw = __funnelshift_r(L.sc[0], L.sc[1], sh);
    const float d = __half2float(__ushort_as_half(L.d));
#pragma unroll
    for (int k = 0; k < 4; k++) w.s[k] = d * (float) (int8_t) (scw >> (8 * k));
}
// one token against a q6_K lane chunk: weights 0..31 of the chunk at element base.., 32..63 at base + 32..
__device__ __forceinline__ float dot6(const w66 & w, const __half * x16, int base) {
    half2 a[4];
#pragma unroll
    for (int k = 0; k < 4; k++) a[k] = __float2half2_rn(0.f);
    const __half * xb = x16 + px(base);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint2 lo = *(const uint2 *) (xb + 4 * i), hi = *(const uint2 *) (xb + 32 + 4 * i);
        a[i >> 2]       = __hfma2(w.l02[i], *(const half2 *) &lo.x, a[i >> 2]);
        a[i >> 2]       = __hfma2(w.l13[i], *(const half2 *) &lo.y, a[i >> 2]);
        a[2 + (i >> 2)] = __hfma2(w.h02[i], *(const half2 *) &hi.x, a[2 + (i >> 2)]);
        a[2 + (i >> 2)] = __hfma2(w.h13[i], *(const half2 *) &hi.y, a[2 + (i >> 2)]);
    }
    return w.s[0] * h2sum(a[0]) + w.s[1] * h2sum(a[1]) + w.s[2] * h2sum(a[2]) + w.s[3] * h2sum(a[3]);
}


// GU5: gate / up rows are q5_K (else q4_K); DN6: down rows are q6_K (else q5_K)
template <int TT, bool GU5, bool DN6>
__global__ void __launch_bounds__(NT, 1) moe_t_kernel(const Params P) {
    // region A: P0-P2 xs16 [TM][XS]; P3 act16 [2 experts * TM][AS] + shared act16 [TM][AS]
    __shared__ __align__(16) __half regA[TM * XS];
    __shared__ __align__(16) uint8_t stage[NW][Q4K_ROW]; // 1152 B: one q4_K row, or a q8_0 half row
    __shared__ float red[NW][TM], rms[TM], sgv[TM], selw[TM][NU];
    __shared__ int sel[TM][NU];
    __shared__ int ulist[MAXU], umaskl[MAXU], wcnt[NW + 1], nU;
    __shared__ float xsum[TM * 64], asum[2 * TM * 16];
    __shared__ unsigned char upos[NE], kof[MAXU][TM]; // expert -> union slot; (union slot, token) -> top-8 position
    const Layer & W = P.L;
    constexpr int T = TT, allm = (1 << T) - 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, nwg = NW * gridDim.x;
    __half * xs16 = regA;

    // ---- P0: rms norm of each token -> xs16 (pre-permuted fp16) + fp32 sub-block sums. One warp per token; a lane
    // owns 64 contiguous elements, all loads issued up front (a load-use loop would pay an L2 round trip per step) ----
    if (warp < T) {
        const int t = warp, k0 = 64 * lane;
        float4 xv[16];
#pragma unroll
        for (int i = 0; i < 16; i++) xv[i] = *(const float4 *) (P.x_in + t * D + k0 + 4 * i);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) ss += dot4(xv[i], xv[i]);
        const float r = rsqrtf(wsum(ss) / D + P.eps);
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float4 nw = __ldg((const float4 *) (W.norm + k0 + 4 * i));
            const float v[4] = {xv[i].x * r * nw.x, xv[i].y * r * nw.y, xv[i].z * r * nw.z, xv[i].w * r * nw.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const __half h = __float2half(v[c]);
                xs16[t * XS + pp(k0 + 4 * i + c)] = h;
                if (i < 8) s0 += __half2float(h); else s1 += __half2float(h);
            }
        }
        xsum[t * 64 + 2 * lane] = s0;
        xsum[t * 64 + 2 * lane + 1] = s1;
    }
    __syncthreads();

    // ---- P1: router + shared gate (f32 rows), shared gate/up (q8_0 rows) ----
    if (gw <= NE) {
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; t++) acc[t] = 0.f;
        const float * w = gw < NE ? W.router + (size_t) gw * D : W.sgate;
#pragma unroll 4
        for (int i = 0; i < D / 128; i++) {
            const float4 v = __ldg((const float4 *) (w + 128 * i) + lane);
#pragma unroll
            for (int t = 0; t < TM; t++) if (t < T) acc[t] += dot4(v, ld4h(xs16 + t * XS + px(128 * i + 4 * lane)));
        }
        { int t; const float a = wsum_multi<T>(acc, lane, t); if (wsum_multi_writer<T>(lane, t)) P.logits[t * (NE + 1) + gw] = a; }
    }
    {
        // shared gate/up rows as a stream of 1088 B chunks (2 per row), the next 2 chunks in registers: staging a row
        // half by half with nothing in flight cost ~24 us for 2.2 MB
        const int u0 = gw <= NE ? gw + nwg : gw, NS = NE + 1 + 2 * FF;
        const int nt = u0 < NS ? 2 * ((NS - u0 + nwg - 1) / nwg) : 0;
        auto rowp = [&](int u) { const int r = u - NE - 1; return r < FF ? W.sg + (size_t) r * Q8_ROW_D : W.su + (size_t) (r - FF) * Q8_ROW_D; };
        auto ld = [&](int t_, int4 * bb) {
            const int4 * p = (const int4 *) (rowp(u0 + (t_ >> 1) * nwg) + (t_ & 1) * HALF);
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
            dot_q8h_T(stage[warp], lane, 32 * (c & 1) + lane, xs16, XS, T, acc);
            __syncwarp();
            if (c & 1) {
                const int r = u0 + (c >> 1) * nwg - NE - 1;
                { int t; const float a = wsum_multi<T>(acc, lane, t); if (wsum_multi_writer<T>(lane, t)) P.shgu[t * 2 * FF + r] = a; }
#pragma unroll
                for (int t = 0; t < TM; t++) acc[t] = 0.f;
            }
        }
    }
    tsync();

    // ---- P2 prologue: top-8 per token (warp t), shared-gate sigmoid, union of experts ----
    if (warp < T) {
        const int t = warp;
        float v[8];
        for (int i = 0; i < 8; i++) v[i] = __ldcg(P.logits + t * (NE + 1) + lane + 32 * i);
        float top0 = 0.f;
        for (int k = 0; k < NU; k++) {
            float bv = -INFINITY; int bi = 0x7fffffff;
            for (int i = 0; i < 8; i++) if (v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, o);
                const int oi = __shfl_xor_sync(0xffffffff, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (k == 0) top0 = bv;
            if (lane == 0) { sel[t][k] = bi; selw[t][k] = __expf(bv - top0); }
#pragma unroll
            for (int i = 0; i < 8; i++) if (bi == lane + 32 * i) v[i] = -INFINITY; // static index: v stays in registers
        }
        __syncwarp();
        if (lane == 0) {
            float s = 0.f;
            for (int k = 0; k < NU; k++) s += selw[t][k];
            for (int k = 0; k < NU; k++) selw[t][k] /= s;
            sgv[t] = 1.f / (1.f + __expf(-__ldcg(P.logits + t * (NE + 1) + NE))); // shared-expert gate
        }
    }
    __syncthreads();
    unsigned um = 0, ub = 0;
    if (warp < NE / 32) {
        // token mask of expert threadIdx.x: compare against the T x 8 selections (no bitmap, no atomics)
        for (int t = 0; t < T; t++)
#pragma unroll
            for (int k = 0; k < NU; k++) um |= (sel[t][k] == (int) threadIdx.x) ? 1u << t : 0u;
        ub = __ballot_sync(0xffffffff, um != 0);
        if (lane == 0) wcnt[warp] = __popc(ub);
    }
    __syncthreads();
    if (warp < NE / 32) {
        int off = 0;
        for (int w = 0; w < warp; w++) off += wcnt[w];
        if (um) { const int pos = off + __popc(ub & ((1u << lane) - 1)); ulist[pos] = threadIdx.x; umaskl[pos] = um; upos[threadIdx.x] = pos; }
        if (threadIdx.x == NE - 1) { int s = 0; for (int w = 0; w < NE / 32; w++) s += wcnt[w]; nU = s; }
    }
    __syncthreads();
    if (threadIdx.x < T * NU) { const int t = threadIdx.x / NU, k = threadIdx.x % NU; kof[upos[sel[t][k]]][t] = k; } // slot of (u, t)
    __syncthreads();
    const int U = nU;

    // ---- P2: expert gate/up rows of the union: rows streamed with coalesced 16 B loads, next 2 rows in registers ----
    if constexpr (!GU5) {
        const int N2 = U * 2 * FF;
        auto q4row = [&](int v) {
            const int u = v / (2 * FF), j = v % (2 * FF);
            return (j < FF ? W.gate : W.up) + ((size_t) ulist[u] * FF + (j % FF)) * Q4K_ROW;
        };
        int4 b0[3], b1[3];
        auto ld = [&](int v, int4 * bb) {
            const int4 * p = (const int4 *) q4row(v);
            bb[0] = __ldg(p + lane); bb[1] = __ldg(p + 32 + lane); if (lane < 8) bb[2] = __ldg(p + 64 + lane);
        };
        int v = gw;
        if (v < N2) ld(v, b0);
        if (v + nwg < N2) ld(v + nwg, b1);
        for (; v < N2; v += nwg) {
            int4 * st = (int4 *) stage[warp];
            st[lane] = b0[0]; st[32 + lane] = b0[1]; if (lane < 8) st[64 + lane] = b0[2];
            __syncwarp();
            b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
            if (v + 2 * nwg < N2) ld(v + 2 * nwg, b1);
            const uint8_t * sb = stage[warp] + (lane >> 2) * 144;
            const q4_ld cur = {*(const int4 *) sb, *(const int4 *) (sb + 16 + 32 * (lane & 3)), *(const int4 *) (sb + 32 + 32 * (lane & 3))};
            __syncwarp();
            const int u = v / (2 * FF), j = v % (2 * FF);
            w64 w;
            prep_q4k(cur, lane & 3, w);
            const int base = (lane >> 2) * 256 + 64 * (lane & 3), j0 = base >> 5;
            for (unsigned m = umaskl[u]; m; m &= m - 1) { // warp-uniform loop over the tokens of this expert
                const int t = __ffs(m) - 1;
                const float a = wsum(dot1(w, xs16 + t * XS, base, xsum[t * 64 + j0], xsum[t * 64 + j0 + 1]));
                if (lane == 0) P.egu[((size_t) u * TM + t) * 2 * FF + j] = a;
            }
        }
    } else {
        // q5_K rows (1408 B) do not fit the staging buffer: lane (superblock, quarter) loads its 64 weights directly,
        // the next row's loads in flight during this row's math
        const int N2 = U * 2 * FF;
        const int sbk = lane >> 2, q = lane & 3;
        auto q5row = [&](int v) {
            const int u = v / (2 * FF), j = v % (2 * FF);
            return load_q5k((j < FF ? W.gate : W.up) + ((size_t) ulist[u] * FF + (j % FF)) * Q5K_ROW_D + sbk * 176, q);
        };
        int v = gw;
        q5_ld Lc;
        if (v < N2) Lc = q5row(v);
        for (; v < N2; v += nwg) {
            q5_ld Ln;
            if (v + nwg < N2) Ln = q5row(v + nwg);
            const int u = v / (2 * FF), j = v % (2 * FF);
            w64 w;
            prep_q5k(Lc, q, w);
            Lc = Ln;
            const int base = sbk * 256 + 64 * q, j0 = base >> 5;
            for (unsigned m = umaskl[u]; m; m &= m - 1) {
                const int t = __ffs(m) - 1;
                const float a = wsum(dot1(w, xs16 + t * XS, base, xsum[t * 64 + j0], xsum[t * 64 + j0 + 1]));
                if (lane == 0) P.egu[((size_t) u * TM + t) * 2 * FF + j] = a;
            }
        }
    }
    tsync();

    // ---- P3: expert down rows in contiguous per-block ranges (<= 2 experts per block), shared down rows ----
    {
        const int N3 = U * D;
        // block ranges split by cost, not rows: a row of union expert u costs 1 + (tokens routed to u) dot products.
        // Equal-row ranges left a 40 us arrival spread at T = 6. A block's share (total / grid) is always below the cheapest
        // expert's 2 * D units (U + 8 T < 112), so a range spans at most 2 experts. Ranges are aligned to 4 rows: a warp's
        // group of 4 rows never spans two experts, so its token mask is uniform.
        long long C = 0;
        for (int u = 0; u < U; u++) C += (long long) D * (P3_ROW_COST + 2 * __popc(umaskl[u]));
        auto row_at = [&](long long c) {
            long long acc = 0;
            for (int u = 0; u < U; u++) {
                const int w = P3_ROW_COST + 2 * __popc(umaskl[u]);
                if (c < acc + (long long) D * w) return (u * D + (int) ((c - acc) / w)) & ~3;
                acc += (long long) D * w;
            }
            return U * D;
        };
        const int r0 = row_at(C * blockIdx.x / gridDim.x);
        const int r1 = blockIdx.x + 1 == gridDim.x ? N3 : row_at(C * (blockIdx.x + 1) / gridDim.x);
        const int ulo = r0 / D, uhi = r1 > r0 ? (r1 - 1) / D : ulo;
        __half * act16 = regA;                      // [2][TM][AS]
        __half * ash16 = regA + 2 * TM * AS;        // [TM][AS]
        // swiglu into fp16 (pre-permuted): warps 0..11 = (expert slot, token) pairs, a lane owns 16 contiguous elements
        // (two lanes per 32-element sub-block); warps 12..15 = shared expert, tokens w-12 and w-8. Loads issued up front.
        if (warp < 2 * TM) {
            const int s_ = warp, uu = ulo + s_ / TM, t = s_ % TM;
            if (uu <= uhi && uu < U && (umaskl[uu] & (1 << t))) {
                const float * gu = P.egu + ((size_t) uu * TM + t) * 2 * FF + 16 * lane;
                float4 g4[4], u4[4];
#pragma unroll
                for (int i = 0; i < 4; i++) { g4[i] = __ldcg((const float4 *) gu + i); u4[i] = __ldcg((const float4 *) (gu + FF) + i); }
                float sum = 0.f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float g[4] = {g4[i].x, g4[i].y, g4[i].z, g4[i].w}, u_[4] = {u4[i].x, u4[i].y, u4[i].z, u4[i].w};
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const __half h = __float2half(silu(g[c]) * u_[c]);
                        act16[s_ * AS + pp(16 * lane + 4 * i + c)] = h;
                        sum += __half2float(h);
                    }
                }
                sum += __shfl_xor_sync(0xffffffff, sum, 1);
                if ((lane & 1) == 0) asum[s_ * 16 + (lane >> 1)] = sum;
            }
        } else {
            for (int t = warp - 2 * TM; t < T; t += NW - 2 * TM) {
                const float * gu = P.shgu + t * 2 * FF + 16 * lane;
                float4 g4[4], u4[4];
#pragma unroll
                for (int i = 0; i < 4; i++) { g4[i] = __ldcg((const float4 *) gu + i); u4[i] = __ldcg((const float4 *) (gu + FF) + i); }
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float g[4] = {g4[i].x, g4[i].y, g4[i].z, g4[i].w}, u_[4] = {u4[i].x, u4[i].y, u4[i].z, u4[i].w};
#pragma unroll
                    for (int c = 0; c < 4; c++) ash16[t * AS + pp(16 * lane + 4 * i + c)] = __float2half(silu(g[c]) * u_[c]);
                }
            }
        }
        __syncthreads();
        // groups of 4 rows per warp: lane = row in group * 8 + superblock * 4 + chunk
        const int sb = (lane >> 2) & 1, q = lane & 3, ri = lane >> 3;
        if constexpr (!DN6) {
            auto q5at = [&](int g) { return load_q5k(W.down + ((size_t) ulist[g / D] * D + g % D + ri) * Q5K_ROW + sb * 176, q); };
            int g0 = r0 + 4 * warp;
            q5_ld Lc;
            if (g0 < r1) Lc = q5at(g0);
            for (; g0 < r1; g0 += 4 * NW) {
                q5_ld Ln;
                if (g0 + 4 * NW < r1) Ln = q5at(g0 + 4 * NW); // next group's loads in flight during this group's math
                const int u = g0 / D, r = g0 % D + ri;
                w64 w;
                prep_q5k(Lc, q, w);
                Lc = Ln;
                const int base = sb * 256 + 64 * q, j0 = base >> 5;
                const __half * a16 = act16 + (u - ulo) * TM * AS;
                const float * as = asum + (u - ulo) * TM * 16;
                for (unsigned m = umaskl[u]; m; m &= m - 1) {
                    const int t = __ffs(m) - 1;
                    float a = dot1(w, a16 + t * AS, base, as[t * 16 + j0], as[t * 16 + j0 + 1]);
                    a += __shfl_xor_sync(0xffffffff, a, 4); a += __shfl_xor_sync(0xffffffff, a, 2); a += __shfl_xor_sync(0xffffffff, a, 1);
                    if ((lane & 7) == 0) {
                        const int k = kof[u][t];
                        P.ypart[((size_t) t * NU + k) * D + r] = selw[t][k] * a;
                    }
                }
            }
        } else {
            auto q6b = [&](int g) { return W.down + ((size_t) ulist[g / D] * D + g % D + ri) * Q6K_ROW + sb * 210; };
            for (int g0 = r0 + 4 * warp; g0 < r1; g0 += 4 * NW) {
                const uint8_t * bq = q6b(g0);
                const q6_ld L = load_q6k(bq, q);
                const int u = g0 / D, r = g0 % D + ri;
                w66 w;
                prep_q6k(L, bq, q, w);
                const int base = sb * 256 + 64 * q;
                const __half * a16 = act16 + (u - ulo) * TM * AS;
                for (unsigned m = umaskl[u]; m; m &= m - 1) {
                    const int t = __ffs(m) - 1;
                    float a = dot6(w, a16 + t * AS, base);
                    a += __shfl_xor_sync(0xffffffff, a, 4); a += __shfl_xor_sync(0xffffffff, a, 2); a += __shfl_xor_sync(0xffffffff, a, 1);
                    if ((lane & 7) == 0) {
                        const int k = kof[u][t];
                        P.ypart[((size_t) t * NU + k) * D + r] = selw[t][k] * a;
                    }
                }
            }
        }
        // shared down: pairs of q8_0 rows (K = 512) for all tokens
        for (int u = gw; u < D / 2; u += nwg) {
            stage_row(stage[warp], W.sd + (size_t) 2 * u * Q8_ROW_F, 2 * Q8_ROW_F, lane);
            const int sub = lane >> 4, blk = lane & 15;
            float acc[TM];
#pragma unroll
            for (int t = 0; t < TM; t++) acc[t] = 0.f;
            dot_q8h_T(stage[warp] + sub * Q8_ROW_F, blk, blk, ash16, AS, T, acc);
            __syncwarp();
#pragma unroll
            for (int t = 0; t < TM; t++) {
                float a = acc[t];
                for (int o = 8; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
                if (t < T && blk == 0) P.shout[t * D + 2 * u + sub] = a;
            }
        }
    }
    tsync();

    // ---- P4: combine ----
    for (int idx = blockIdx.x * NT + threadIdx.x; idx < T * D; idx += gridDim.x * NT) {
        const int t = idx / D, r = idx % D;
        float y = 0.f;
#pragma unroll
        for (int k = 0; k < NU; k++) y += __ldcg(P.ypart + ((size_t) t * NU + k) * D + r);
        P.x_out[idx] = P.x_in[idx] + (y + sgv[t] * __ldcg(P.shout + idx));
    }
    __syncthreads();
}

} // namespace moe

} // namespace

bool pxa_p60_mega_moe_t(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens) {
    using namespace moe;
    if (n_tokens < 1 || n_tokens > TM) {
        return false;
    }
    const bool gu5 = args.gu_type == GGML_TYPE_Q5_K, dn6 = args.dn_type == GGML_TYPE_Q6_K;
    if (!((args.gu_type == GGML_TYPE_Q4_K || gu5) && (args.dn_type == GGML_TYPE_Q5_K || dn6) && (!gu5 || dn6))) {
        return false;
    }
    const size_t n_logits = (TM * (NE + 1) + 3) & ~3, // float4 alignment of what follows
                  n_shgu = TM * 2 * FF, n_egu = (size_t) MAXU * TM * 2 * FF, n_shout = TM * D;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_logits + n_shgu + n_egu + n_shout + (size_t) TM * NU * D);
    Params kp = {};
    kp.L = { args.norm, args.router, args.sgate, (const uint8_t *) args.gate, (const uint8_t *) args.up, (const uint8_t *) args.down,
             (const uint8_t *) args.sg, (const uint8_t *) args.su, (const uint8_t *) args.sd };
    kp.x_in   = args.x_in;
    kp.x_out  = args.x_out;
    kp.eps    = args.eps;
    kp.logits = scratch.get();
    kp.shgu   = kp.logits + n_logits;
    kp.egu    = kp.shgu + n_shgu;
    kp.shout  = kp.egu + n_egu;
    kp.ypart  = kp.shout + n_shout;
#define P60_MOE_T_KERNELS(GU5, DN6) { (const void *) moe_t_kernel<1, GU5, DN6>, (const void *) moe_t_kernel<2, GU5, DN6>, \
        (const void *) moe_t_kernel<3, GU5, DN6>, (const void *) moe_t_kernel<4, GU5, DN6>, (const void *) moe_t_kernel<5, GU5, DN6>, \
        (const void *) moe_t_kernel<6, GU5, DN6> }
    static const void * const kern[3][TM] = { P60_MOE_T_KERNELS(false, false), P60_MOE_T_KERNELS(false, true), P60_MOE_T_KERNELS(true, true) };
#undef P60_MOE_T_KERNELS
    const int nblk = ggml_cuda_info().devices[ctx.device].nsm;
    void * kargs[] = { &kp };
    CUDA_CHECK(cudaLaunchCooperativeKernel(kern[gu5 ? 2 : dn6 ? 1 : 0][n_tokens - 1], dim3(nblk), dim3(mt::NT), kargs, 0, ctx.stream()));
    return true;
}
