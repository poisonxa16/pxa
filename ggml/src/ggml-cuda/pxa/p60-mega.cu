// pxa / PXA kernel suite -- p60-mega.cu: persistent single-token layer-half kernels for the Qwen3.5/3.6 MoE layers on
// sm_60 (PXA_P60_MEGA). One cooperative launch runs a whole half of a layer: all blocks stay resident (one block of
// 512 threads per SM) and sync with a software grid barrier between phases, so the ~25 kernels of a half become one
// launch with 3 phases. fp32 math throughout; k-quant and q8_0 weights are dequantized in registers.
//   * grid barrier: a generation barrier (arrival count reset by the last block, generation bumped), no host-side epoch,
//     so a captured launch replays correctly in a CUDA graph; its state is a module-global __device__ variable (one
//     instance per device, zero at load): no allocation, so even a first launch inside a capture works;
//   * MoE half templated on the expert types: gate/up Q4_K or Q5_K, down Q5_K or Q6_K (all 40 layers of the UD-Q4_K_XL
//     file, where 34/38/39 carry Q6_K down and 39 Q5_K gate/up);
//   * GDN half in this tree's DELTA_NET semantics (state S[k][v], v contiguous; reset mask; l2-norm eps under the sqrt).
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
#include <map>
#include <mutex>

namespace {

constexpr int NT = 512, NW = NT / 32;
constexpr int D = 2048;
// MoE
constexpr int NE = 256, NU = 8, FF = 512;
constexpr int Q4K_ROW_D = D / 256 * 144, Q5K_ROW_D = D / 256 * 176;
constexpr int Q5K_ROW_F = FF / 256 * 176, Q6K_ROW_F = FF / 256 * 210;
constexpr int Q8_ROW_D = D / 32 * 34, Q8_ROW_F = FF / 32 * 34, CHUNK = 1088; // 32 q8_0 blocks

// ---------------- grid barrier (generation; no host epoch) ----------------

struct p60_bar_t {
    unsigned count;
    unsigned gen;
    unsigned pad[30];
};

// one instance per device (a module global): zero at load, no allocation, so a first launch inside a CUDA-graph
// capture needs nothing the capture forbids
__device__ p60_bar_t g_p60_bar;

__device__ __forceinline__ void grid_sync(p60_bar_t * bar) {
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

// padded smem index for float4 reads at strides of 64 floats
__device__ __forceinline__ int px(int k) { return k + 4 * (k >> 6); }
// padded smem index for scalar reads at strides of 32 floats (one q8_0 block per lane)
__device__ __forceinline__ int py(int k) { return k + (k >> 5); }

// 4 bytes of u to floats without I2F (quarter rate on sm_60): 0x4B0000bb = 2^23 + bb
__device__ __forceinline__ float4 bytes_f(unsigned u, float bias) {
    return make_float4(__int_as_float(__byte_perm(u, 0x4B000000, 0x7540)) - bias, __int_as_float(__byte_perm(u, 0x4B000000, 0x7541)) - bias,
                       __int_as_float(__byte_perm(u, 0x4B000000, 0x7542)) - bias, __int_as_float(__byte_perm(u, 0x4B000000, 0x7543)) - bias);
}
__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }

__device__ __forceinline__ void stage_row(uint8_t * dst, const char * src, int bytes, int lane) {
    for (int i = lane; i < bytes / 16; i += 32) {
        ((int4 *) dst)[i] = __ldg((const int4 *) src + i);
    }
    __syncwarp();
}

// one q8_0 block (34 B, 2 B aligned in smem) against x[32 * xblk ...] in py (PYL) or px layout
template <bool PYL = false>
__device__ __forceinline__ float dot_q8_staged(const uint8_t * stage, int blk, int xblk, const float * xs) {
    const uint8_t * b = stage + 34 * blk;
    const float d = __half2float(*(const half *) b);
    const unsigned short * q = (const unsigned short *) (b + 2);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned v = (q[2 * i] | ((unsigned) q[2 * i + 1] << 16)) ^ 0x80808080;
        const int k = 32 * xblk + 4 * i;
        const float4 xv = PYL ? make_float4(xs[py(k)], xs[py(k + 1)], xs[py(k + 2)], xs[py(k + 3)])
                              : make_float4(xs[px(k)], xs[px(k + 1)], xs[px(k + 2)], xs[px(k + 3)]);
        a += dot4(bytes_f(v, 8388608.f + 128.f), xv);
    }
    return d * a;
}

// q8_0 rows as a stream of 1088 B chunks per warp (items u0, u0 + nwg, ... < uend; NCH chunks each). The next two
// chunks stay in registers while the current one is computed from smem. pre(u) runs when an item starts so that its
// epilogue loads are in flight during the math; done(u, sum, pre_result) runs when it ends.
template <int NCH, bool PYL, typename RowFn, typename PreFn, typename DoneFn>
__device__ __forceinline__ void q8_stream_p(int u0, int uend, int nwg, int lane, uint8_t * stage, const float * xv, RowFn rowp, PreFn pre, DoneFn done) {
    const int nitems = u0 < uend ? (uend - u0 + nwg - 1) / nwg : 0;
    const int nt = nitems * NCH;
    int4 b0[3], b1[3];
    auto ld = [&](int t, int4 * bb) {
        const int4 * p = (const int4 *) (rowp(u0 + (t / NCH) * nwg) + (t % NCH) * CHUNK);
        bb[0] = __ldg(p + lane);
        bb[1] = __ldg(p + 32 + lane);
        if (lane < 4) {
            bb[2] = __ldg(p + 64 + lane);
        }
    };
    if (nt > 0) ld(0, b0);
    if (nt > 1) ld(1, b1);
    float a = 0.f;
    decltype(pre(0)) pv = {};
    for (int t = 0; t < nt; t++) {
        if (t % NCH == 0) {
            pv = pre(u0 + (t / NCH) * nwg);
        }
        int4 * s = (int4 *) stage;
        s[lane] = b0[0];
        s[32 + lane] = b0[1];
        if (lane < 4) {
            s[64 + lane] = b0[2];
        }
        __syncwarp();
        b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
        if (t + 2 < nt) {
            ld(t + 2, b1);
        }
        a += dot_q8_staged<PYL>(stage, lane, 32 * (t % NCH) + lane, xv);
        __syncwarp();
        if (t % NCH == NCH - 1) {
            a = wsum(a);
            done(u0 + (t / NCH) * nwg, a, pv);
            a = 0.f;
        }
    }
}

template <int NCH, typename RowFn, typename DoneFn>
__device__ __forceinline__ void q8_stream(int u0, int uend, int nwg, int lane, uint8_t * stage, const float * xv, RowFn rowp, DoneFn done) {
    const int nitems = u0 < uend ? (uend - u0 + nwg - 1) / nwg : 0;
    const int nt = nitems * NCH;
    int4 b0[3], b1[3];
    auto ld = [&](int t, int4 * bb) {
        const int4 * p = (const int4 *) (rowp(u0 + (t / NCH) * nwg) + (t % NCH) * CHUNK);
        bb[0] = __ldg(p + lane);
        bb[1] = __ldg(p + 32 + lane);
        if (lane < 4) {
            bb[2] = __ldg(p + 64 + lane);
        }
    };
    if (nt > 0) ld(0, b0);
    if (nt > 1) ld(1, b1);
    float a = 0.f;
    for (int t = 0; t < nt; t++) {
        int4 * s = (int4 *) stage;
        s[lane] = b0[0];
        s[32 + lane] = b0[1];
        if (lane < 4) {
            s[64 + lane] = b0[2];
        }
        __syncwarp();
        b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
        if (t + 2 < nt) {
            ld(t + 2, b1);
        }
        a += dot_q8_staged(stage, lane, 32 * (t % NCH) + lane, xv);
        __syncwarp();
        if (t % NCH == NCH - 1) {
            a = wsum(a);
            done(u0 + (t / NCH) * nwg, a);
            a = 0.f;
        }
    }
}

// xs = rms_norm(x) * w for one 2048 vector, per block, px (or py: PYL) layout
template <bool PYL = false>
__device__ __forceinline__ void block_rms_norm(const float * x, const float * w, float eps, float * xs, float * red, float * scal) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float xv[D / NT], ss = 0.f;
#pragma unroll
    for (int j = 0; j < D / NT; j++) {
        xv[j] = x[threadIdx.x + j * NT];
        ss += xv[j] * xv[j];
    }
    ss = wsum(ss);
    if (lane == 0) {
        red[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        float v = lane < NW ? red[lane] : 0.f;
        v = wsum(v);
        if (lane == 0) {
            scal[0] = rsqrtf(v / D + eps);
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < D / NT; j++) {
        const int k = threadIdx.x + j * NT;
        xs[PYL ? py(k) : px(k)] = xv[j] * scal[0] * __ldg(w + k);
    }
    __syncthreads();
}

// ---------------- k-quant helpers ----------------

// 6-bit scale/min j (0..7) from the 12 scale bytes in words s0, s1, s2 (no dynamic register indexing)
__device__ __forceinline__ void scale_min_k4(int j, unsigned s0, unsigned s1, unsigned s2, int & d, int & m) {
    const int sh = 8 * (j & 3);
    const int b0 = (s0 >> sh) & 0xFF, b1 = (s1 >> sh) & 0xFF, b2 = (s2 >> sh) & 0xFF;
    if (j < 4) {
        d = b0 & 63; m = b1 & 63;
    } else {
        d = (b2 & 0xF) | ((b0 >> 6) << 4); m = (b2 >> 4) | ((b1 >> 6) << 4);
    }
}

struct q4_ld { int4 h, qa, qb; };
// q4_K row of K = 2048, one warp: lane = superblock * 4 + 64-weight chunk
__device__ __forceinline__ q4_ld load_q4k(const char * row, int lane) {
    const char * b = row + (lane >> 2) * 144;
    const int q = lane & 3;
    return {__ldg((const int4 *) b), __ldg((const int4 *) (b + 16 + 32 * q)), __ldg((const int4 *) (b + 32 + 32 * q))};
}
__device__ __forceinline__ float dot_q4k(const q4_ld & L, const float * xs, int lane) {
    const int sb = lane >> 2, q = lane & 3;
    const half2 dm = __halves2half2(__ushort_as_half(L.h.x & 0xFFFF), __ushort_as_half(L.h.x >> 16));
    int s0, m0, s1, m1;
    scale_min_k4(2 * q, L.h.y, L.h.z, L.h.w, s0, m0);
    scale_min_k4(2 * q + 1, L.h.y, L.h.z, L.h.w, s1, m1);
    const int base = sb * 256 + 64 * q;
    const unsigned w[8] = {(unsigned) L.qa.x, (unsigned) L.qa.y, (unsigned) L.qa.z, (unsigned) L.qa.w,
                           (unsigned) L.qb.x, (unsigned) L.qb.y, (unsigned) L.qb.z, (unsigned) L.qb.w};
    float a0 = 0.f, a1 = 0.f, x0 = 0.f, x1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float4 lo = *(const float4 *) &xs[px(base + 4 * i)];
        const float4 hi = *(const float4 *) &xs[px(base + 32 + 4 * i)];
        a0 += dot4(bytes_f(w[i] & 0x0F0F0F0F, 8388608.f), lo);
        a1 += dot4(bytes_f((w[i] >> 4) & 0x0F0F0F0F, 8388608.f), hi);
        x0 += lo.x + lo.y + lo.z + lo.w;
        x1 += hi.x + hi.y + hi.z + hi.w;
    }
    return __low2float(dm) * (s0 * a0 + s1 * a1) - __high2float(dm) * (m0 * x0 + m1 * x1);
}

// q5_K superblock, 64-weight chunk q against act (px layout), base = element index of the chunk
__device__ __forceinline__ float dot_q5k(const char * b, int q, const float * act, int base) {
    const int4 h  = __ldg((const int4 *) b);
    const int4 ha = __ldg((const int4 *) (b + 16));
    const int4 hb = __ldg((const int4 *) (b + 32));
    const int4 qa = __ldg((const int4 *) (b + 48 + 32 * q));
    const int4 qb = __ldg((const int4 *) (b + 64 + 32 * q));
    const half2 dm = __halves2half2(__ushort_as_half(h.x & 0xFFFF), __ushort_as_half(h.x >> 16));
    int s0, m0, s1, m1;
    scale_min_k4(2 * q, h.y, h.z, h.w, s0, m0);
    scale_min_k4(2 * q + 1, h.y, h.z, h.w, s1, m1);
    const unsigned w[8]  = {(unsigned) qa.x, (unsigned) qa.y, (unsigned) qa.z, (unsigned) qa.w, (unsigned) qb.x, (unsigned) qb.y, (unsigned) qb.z, (unsigned) qb.w};
    const unsigned hh[8] = {(unsigned) ha.x, (unsigned) ha.y, (unsigned) ha.z, (unsigned) ha.w, (unsigned) hb.x, (unsigned) hb.y, (unsigned) hb.z, (unsigned) hb.w};
    const unsigned bl = 2 * q, bh = 2 * q + 1;
    float a0 = 0.f, a1 = 0.f, x0 = 0.f, x1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float4 lo = *(const float4 *) &act[px(base + 4 * i)];
        const float4 hi = *(const float4 *) &act[px(base + 32 + 4 * i)];
        a0 += dot4(bytes_f((w[i] & 0x0F0F0F0F) | (((hh[i] >> bl) & 0x01010101) << 4), 8388608.f), lo);
        a1 += dot4(bytes_f(((w[i] >> 4) & 0x0F0F0F0F) | (((hh[i] >> bh) & 0x01010101) << 4), 8388608.f), hi);
        x0 += lo.x + lo.y + lo.z + lo.w;
        x1 += hi.x + hi.y + hi.z + hi.w;
    }
    return __low2float(dm) * (s0 * a0 + s1 * a1) - __high2float(dm) * (m0 * x0 + m1 * x1);
}

// q6_K superblock (210 B, only 2 B aligned), 64-weight chunk q (half n = q / 2, hi = q % 2) against act (px layout).
// Weights 64q + l (l < 32) come from ql[64n + l] nibble hi, qh[32n + l] bits 4hi..4hi+1, scale sc[8n + l/16 + 4hi];
// weights 64q + 32 + l from ql[64n + 32 + l], qh bits 4hi+2..4hi+3, scale sc[8n + l/16 + 2 + 4hi]; value d*sc*(q6-32).
__device__ __forceinline__ float dot_q6k(const char * b, int q, const float * act, int base) {
    const int n = q >> 1, hi = q & 1;
    const unsigned short * ql = (const unsigned short *) (b + 64 * n);
    const unsigned short * qh = (const unsigned short *) (b + 128 + 32 * n);
    const int8_t * sc = (const int8_t *) (b + 192 + 8 * n);
    const float d = __half2float(__ushort_as_half(__ldg((const unsigned short *) (b + 208))));
    const int sh_lo = 4 * hi, sh_q = 4 * hi;
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 16; i++) { // two weights l = 2i, 2i+1 of each 32-run per iteration
        const unsigned qa = __ldg(ql + i), qb = __ldg(ql + 16 + i), hh = __ldg(qh + i);
        const int l = 2 * i;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned ba = (qa >> (8 * c)) & 0xFF, bb = (qb >> (8 * c)) & 0xFF, bh = (hh >> (8 * c)) & 0xFF;
            const int w0 = (int) (((ba >> sh_q) & 0xF) | (((bh >> sh_lo) & 3) << 4)) - 32;
            const int w1 = (int) (((bb >> sh_q) & 0xF) | (((bh >> (sh_lo + 2)) & 3) << 4)) - 32;
            const int g = (l + c) >> 4; // 0 or 1: which 16-run of the 32
            a[g]     += (float) w0 * act[px(base + l + c)];
            a[2 + g] += (float) w1 * act[px(base + 32 + l + c)];
        }
    }
    return d * ((float) sc[4 * hi] * a[0] + (float) sc[4 * hi + 1] * a[1] + (float) sc[4 * hi + 2] * a[2] + (float) sc[4 * hi + 3] * a[3]);
}

// ---------------- MoE half ----------------

struct moe_params {
    pxa_p60_moe_args a;
    float * logits; // [NE + 1]: router, shared-expert gate
    float * shgu;   // [2 * FF]
    float * egu;    // [NU][2 * FF]
    float * shout;  // [D]
};

template <int GU, int DN> // GU: 4 = q4_K, 5 = q5_K gate/up; DN: 5 = q5_K, 6 = q6_K down
__global__ void __launch_bounds__(NT, 1) p60_mega_moe_kernel(const moe_params P) {
    __shared__ __align__(16) float xs[D + D / 16];             // normed x, px layout
    __shared__ __align__(16) float act[NU * FF + NU * FF / 16]; // expert swiglu (px); first FF: shared swiglu
    __shared__ __align__(16) uint8_t stage[NW][CHUNK];
    __shared__ float red[NW], scal[2], red_lg[NE], selw[NU];
    __shared__ unsigned long long red_key[64];
    __shared__ int sel[NU];
    const pxa_p60_moe_args & A = P.a;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, nwg = NW * gridDim.x;

    block_rms_norm(A.x_in, A.norm, A.eps, xs, red, scal);

    // phase 1: router + shared gate input (f32 rows), shared gate/up (q8_0)
    if (gw <= NE) {
        const float * w = gw < NE ? A.router + (size_t) gw * D : A.sgate;
        float a = 0.f;
#pragma unroll 4
        for (int i = 0; i < D / 128; i++) {
            a += dot4(__ldg((const float4 *) (w + 128 * i) + lane), *(const float4 *) &xs[px(128 * i + 4 * lane)]);
        }
        a = wsum(a);
        if (lane == 0) {
            P.logits[gw] = a;
        }
    }
    q8_stream<2>(gw <= NE ? gw + nwg : gw, NE + 1 + 2 * FF, nwg, lane, stage[warp], xs,
        [&](int u) { const int r = u - NE - 1; return r < FF ? A.sg + (size_t) r * Q8_ROW_D : A.su + (size_t) (r - FF) * Q8_ROW_D; },
        [&](int u, float a) { if (lane == 0) P.shgu[u - NE - 1] = a; });
    grid_sync(&g_p60_bar);

    // phase 2 prologue (per block): top-8 of the router (weights renormalized over the top 8, so the softmax
    // denominator cancels); sort key = order-preserving logit bits, then 255 - index (ties to the lower index).
    // Each of 8 warps bitonic-sorts its 32 keys and keeps its top 8, then the 64 candidates are ranked.
    if (threadIdx.x == NE) {
        scal[1] = 1.f / (1.f + __expf(-__ldcg(P.logits + NE)));
    }
    if (threadIdx.x < NE) {
        const float v = __ldcg(P.logits + threadIdx.x);
        const unsigned uu = __float_as_uint(v);
        const unsigned o = uu ^ ((unsigned) ((int) uu >> 31) | 0x80000000u);
        unsigned long long k = ((unsigned long long) o << 8) | (255 - threadIdx.x);
        red_lg[threadIdx.x] = v;
#pragma unroll
        for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
            for (int stride = size / 2; stride > 0; stride >>= 1) {
                const unsigned long long ot = __shfl_xor_sync(0xffffffff, k, stride);
                const bool keep_max = ((lane & stride) == 0) == ((lane & size) == 0);
                k = keep_max ? (ot > k ? ot : k) : (ot < k ? ot : k);
            }
        }
        if (lane < NU) {
            red_key[warp * NU + lane] = k;
        }
    }
    {
        float gv = 0.f, uv = 0.f;
        if (threadIdx.x < FF) {
            gv = __ldcg(P.shgu + threadIdx.x);
            uv = __ldcg(P.shgu + FF + threadIdx.x);
        }
        __syncthreads();
        if (threadIdx.x < FF) {
            act[px(threadIdx.x)] = silu_f(gv) * uv;
        }
    }
    if (threadIdx.x < 64) {
        const unsigned long long k = red_key[threadIdx.x];
        int rank = 0;
#pragma unroll 16
        for (int j = 0; j < 64; j++) {
            rank += red_key[j] > k;
        }
        if (rank < NU) {
            const int idx = 255 - (int) (k & 0xFF);
            sel[rank] = idx;
            selw[rank] = red_lg[idx];
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float e = lane < NU ? __expf(selw[lane] - selw[0]) : 0.f;
        const float sum = wsum(e);
        __syncwarp();
        if (lane < NU) {
            selw[lane] = e / sum;
        }
    }
    __syncthreads();

    // phase 2: shared down (pairs of q8_0 rows, K = 512), then expert gate/up (next row prefetched for q4_K)
    constexpr int NSD = D / 2, N2 = NSD + NU * 2 * FF;
    int u = gw;
    for (; u < NSD; u += nwg) {
        const int r2 = 2 * u;
        stage_row(stage[warp], A.sd + (size_t) r2 * Q8_ROW_F, 2 * Q8_ROW_F, lane);
        const int sub = lane >> 4, blk = lane & 15;
        float a = dot_q8_staged(stage[warp] + sub * Q8_ROW_F, blk, blk, act);
        __syncwarp();
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, o);
        }
        if (blk == 0) {
            P.shout[r2 + sub] = a;
        }
    }
    if (GU == 4) {
        auto q4row = [&](int uu) {
            const int r = uu - NSD, e = r / (2 * FF), j = r % (2 * FF);
            return (j < FF ? A.gate : A.up) + ((size_t) sel[e] * FF + (j % FF)) * Q4K_ROW_D;
        };
        if (u < N2) {
            q4_ld cur = load_q4k(q4row(u), lane);
            for (; u < N2; u += nwg) {
                q4_ld nxt;
                if (u + nwg < N2) {
                    nxt = load_q4k(q4row(u + nwg), lane);
                }
                const float a = wsum(dot_q4k(cur, xs, lane));
                if (lane == 0) {
                    P.egu[u - NSD] = a;
                }
                cur = nxt;
            }
        }
    } else {
        for (; u < N2; u += nwg) {
            const int r = u - NSD, e = r / (2 * FF), j = r % (2 * FF);
            const char * row = (j < FF ? A.gate : A.up) + ((size_t) sel[e] * FF + (j % FF)) * Q5K_ROW_D;
            const int sb = lane >> 2, q = lane & 3;
            const float a = wsum(dot_q5k(row + sb * 176, q, xs, sb * 256 + 64 * q));
            if (lane == 0) {
                P.egu[r] = a;
            }
        }
    }
    grid_sync(&g_p60_bar);

    // phase 3: expert swiglu (per block), expert down + weighted sum + shared expert + residual
    {
        float gv[NU * FF / NT], uv[NU * FF / NT];
#pragma unroll
        for (int j = 0; j < NU * FF / NT; j++) {
            const int k = threadIdx.x + j * NT, e = k / FF, i = k % FF;
            gv[j] = __ldcg(P.egu + e * 2 * FF + i);
            uv[j] = __ldcg(P.egu + e * 2 * FF + FF + i);
        }
#pragma unroll
        for (int j = 0; j < NU * FF / NT; j++) {
            act[px(threadIdx.x + j * NT)] = silu_f(gv[j]) * uv[j];
        }
    }
    __syncthreads();
    for (int r = gw; r < D; r += nwg) {
        // lane = (expert in the half) * 8 + superblock * 4 + 64-weight chunk
        const int eg = lane >> 3, sb = (lane >> 2) & 1, q = lane & 3;
        float x0 = 0.f, sh = 0.f;
        if (lane == 0) {
            x0 = A.x_in[r];
            sh = __ldcg(P.shout + r);
        }
        float a = 0.f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int e = eg + 4 * h;
            if (DN == 5) {
                a += selw[e] * dot_q5k(A.down + ((size_t) sel[e] * D + r) * Q5K_ROW_F + sb * 176, q, act, e * FF + sb * 256 + 64 * q);
            } else {
                a += selw[e] * dot_q6k(A.down + ((size_t) sel[e] * D + r) * Q6K_ROW_F + sb * 210, q, act, e * FF + sb * 256 + 64 * q);
            }
        }
        a = wsum(a);
        if (lane == 0) {
            A.x_out[r] = x0 + (a + scal[1] * sh);
        }
    }
}


// ---------------- gated delta net half (PXA DELTA_NET semantics) ----------------

constexpr int HK = 16, HV = 32, SD = 128, QKV = 2 * HK * SD + HV * SD, VD = HV * SD;
constexpr int Q8_ROW_V = VD / 32 * 34;
constexpr int CONV_N = 3 * QKV; // conv-state floats at the start of a recurrent-cache row

struct gdn_params {
    pxa_p60_gdn_args a;
    float * qkvc; // [QKV] conv + silu output
    float * zv;   // [VD]
    float * gb;   // [2][HV] g = softplus(alpha + dt) * a | raw beta
    float * o;    // [VD] recurrence output
};

struct conv_pre { float4 w; float s0, s1, s2; };

__global__ void __launch_bounds__(NT, 1) p60_mega_gdn_kernel(const gdn_params P) {
    __shared__ __align__(16) float xs[D + D / 32];   // normed x, py layout
    __shared__ __align__(16) float on[VD + VD / 32]; // gated-normed recurrence output, py layout
    __shared__ __align__(16) uint8_t stage[NW][CHUNK];
    __shared__ float red[NW], scal[1];
    const pxa_p60_gdn_args & A = P.a;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, nwg = NW * gridDim.x;
    const float m = A.mask[0];
    const float * s_rd = (const float *) (A.s_rd_base + (int64_t) A.rd_ids[0] * A.rd_stride);
    float *       s_wr = (float *) (A.s_wr_base + (int64_t) A.wr_ids[0] * A.wr_stride);

    block_rms_norm<true>(A.x_in, A.attn_norm, A.eps_norm, xs, red, scal);

    // phase 1: alpha, beta (f32 rows), qkv rows with the causal conv + silu + conv-state update, z rows
    constexpr int NAB = 2 * HV, N1 = NAB + QKV + VD;
    if (gw < NAB) {
        const int h = gw % HV;
        const float * w = (gw < HV ? A.alpha : A.beta) + (size_t) h * D;
        float a = 0.f;
#pragma unroll 4
        for (int i = 0; i < D / 128; i++) {
            const int k = 128 * i + 4 * lane;
            a += dot4(__ldg((const float4 *) (w + 128 * i) + lane), make_float4(xs[py(k)], xs[py(k + 1)], xs[py(k + 2)], xs[py(k + 3)]));
        }
        a = wsum(a);
        if (lane == 0) {
            if (gw < HV) {
                const float xb = a + __ldg(A.dt + h);
                P.gb[h] = (xb > 20.0f ? xb : logf(1.0f + expf(xb))) * __ldg(A.a + h);
            } else {
                P.gb[HV + h] = a;
            }
        }
    }
    q8_stream_p<2, true>(gw < NAB ? gw + nwg : gw, N1, nwg, lane, stage[warp], xs,
        [&](int u) { const int r = u - NAB; return r < QKV ? A.qkv + (size_t) r * Q8_ROW_D : A.z + (size_t) (r - QKV) * Q8_ROW_D; },
        [&](int u) {
            const int r = u - NAB;
            conv_pre p = {};
            if (lane == 0 && r < QKV) {
                p.w  = __ldg((const float4 *) (A.conv + 4 * r));
                p.s0 = s_rd[3 * r] * m; p.s1 = s_rd[3 * r + 1] * m; p.s2 = s_rd[3 * r + 2] * m;
            }
            return p;
        },
        [&](int u, float a, const conv_pre & p) {
            const int r = u - NAB;
            if (lane != 0) {
                return;
            }
            if (r < QKV) {
                // conv over [s0, s1, s2, x], state <- [s1, s2, x]
                const float c = p.w.x * p.s0 + p.w.y * p.s1 + p.w.z * p.s2 + p.w.w * a;
                s_wr[3 * r] = p.s1; s_wr[3 * r + 1] = p.s2; s_wr[3 * r + 2] = a;
                P.qkvc[r] = silu_f(c);
            } else {
                P.zv[r - QKV] = a;
            }
        });
    grid_sync(&g_p60_bar);

    // phase 2: recurrence. Item = (v head h, 32-wide value chunk c), one item per block; warp w owns key rows
    // 8w .. 8w+7, lane = value column, so every state load / store of a warp is one 128 B line (S[k][v], v contiguous).
    {
        float * qs = (float *) stage;            // [SD] l2-normed q * 1/sqrt(SD)
        float * ks = qs + SD;                    // [SD] l2-normed k
        float * r1 = ks + SD;                    // [NW][33]
        float * r2 = r1 + NW * 33;               // [NW][33]
        float * sv = r2 + NW * 33;               // [32]
        float * sc = sv + 32;                    // [4]: nq, nk, score, decay
        const float * S_in  = s_rd + CONV_N;
        float *       S_out = s_wr + CONV_N;
        for (int item = blockIdx.x; item < HV * (SD / 32); item += gridDim.x) {
            const int h = item / (SD / 32), c = item % (SD / 32);
            const int kh = A.repeat_type == 0 ? h / (HV / HK) : h % HK;
            if (threadIdx.x < SD) {
                qs[threadIdx.x] = __ldcg(P.qkvc + kh * SD + threadIdx.x);
            } else if (threadIdx.x < 2 * SD) {
                ks[threadIdx.x - SD] = __ldcg(P.qkvc + HK * SD + kh * SD + threadIdx.x - SD);
            }
            // the state loads do not depend on q / k: issue them first
            float s[8];
            const size_t sbase = (size_t) h * SD * SD + 32 * c + lane;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                s[j] = S_in[sbase + (size_t) (8 * warp + j) * SD] * m;
            }
            __syncthreads();
            if (warp < 2) {
                const float * v = warp == 0 ? qs : ks;
                float ss = 0.f;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    ss += v[lane + 32 * j] * v[lane + 32 * j];
                }
                ss = wsum(ss);
                if (lane == 0) {
                    sc[warp] = warp == 0 ? rsqrtf(ss + A.eps_l2q) * rsqrtf((float) SD) : rsqrtf(ss + A.eps_l2k);
                }
            }
            __syncthreads();
            const float nq = sc[0], nk = sc[1];
            if (warp == 2) {
                float kq = 0.f;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    kq += (ks[lane + 32 * j] * nk) * (qs[lane + 32 * j] * nq);
                }
                kq = wsum(kq);
                if (lane == 0) {
                    sc[2] = kq;
                    sc[3] = expf(fminf(__ldcg(P.gb + h), 50.0f));
                }
            }
            float p1 = 0.f, p2 = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kk = 8 * warp + j;
                p1 += s[j] * (ks[kk] * nk);
                p2 += s[j] * (qs[kk] * nq);
            }
            r1[warp * 33 + lane] = p1;
            r2[warp * 33 + lane] = p2;
            __syncthreads();
            const float decay = sc[3];
            if (warp == 0) {
                float s1 = 0.f, s2 = 0.f;
#pragma unroll
                for (int w = 0; w < NW; w++) {
                    s1 += r1[w * 33 + lane];
                    s2 += r2[w * 33 + lane];
                }
                const float beta = 1.0f / (1.0f + expf(-__ldcg(P.gb + HV + h)));
                const float vv = __ldcg(P.qkvc + 2 * HK * SD + h * SD + 32 * c + lane);
                const float svn = beta * (vv - s1 * decay);
                P.o[h * SD + 32 * c + lane] = s2 * decay + svn * sc[2];
                sv[lane] = svn;
            }
            __syncthreads();
            const float svn = sv[lane];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kk = 8 * warp + j;
                const float ns = decay * s[j] + svn * (ks[kk] * nk);
                S_out[sbase + (size_t) kk * SD] = fminf(fmaxf(ns, -1e6f), 1e6f);
            }
            __syncthreads();
        }
    }
    grid_sync(&g_p60_bar);

    // phase 3: gated norm (per block, 8 elements per thread, 16 threads per head), ssm_out rows + residual
    {
        const int e = 8 * threadIdx.x, i0 = e % SD;
        const float4 o0 = __ldcg((const float4 *) (P.o + e)),  o1 = __ldcg((const float4 *) (P.o + e + 4));
        const float4 z0 = __ldcg((const float4 *) (P.zv + e)), z1 = __ldcg((const float4 *) (P.zv + e + 4));
        const float4 n0 = __ldg((const float4 *) (A.snorm + i0)), n1 = __ldg((const float4 *) (A.snorm + i0 + 4));
        float ssq = dot4(o0, o0) + dot4(o1, o1);
#pragma unroll
        for (int mm = 8; mm > 0; mm >>= 1) {
            ssq += __shfl_xor_sync(0xffffffff, ssq, mm);
        }
        const float inv = rsqrtf(ssq / SD + A.eps_gn);
        const float ov[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
        const float zz[8] = {z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
        const float nn[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
#pragma unroll
        for (int j = 0; j < 8; j++) {
            on[py(e + j)] = (ov[j] * inv * nn[j]) * silu_f(zz[j]);
        }
    }
    __syncthreads();
    q8_stream_p<4, true>(gw, D, nwg, lane, stage[warp], on,
        [&](int r) { return A.out + (size_t) r * Q8_ROW_V; },
        [&](int r) { return lane == 0 ? A.x_in[r] : 0.f; },
        [&](int r, float a, float x0) { if (lane == 0) A.x_out[r] = x0 + a; });
}

template <typename KP>
void mega_launch(ggml_backend_cuda_context & ctx, void (*kernel)(const KP), KP & kp) {
    const int nblk = ggml_cuda_info().devices[ctx.device].nsm;
    void * args[] = { &kp };
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) kernel, dim3(nblk), dim3(NT), args, 0, ctx.stream()));
}

} // namespace

bool pxa_p60_mega_moe(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args) {
    void (*kern)(const moe_params) = nullptr;
    if (args.gu_type == GGML_TYPE_Q4_K && args.dn_type == GGML_TYPE_Q5_K) kern = p60_mega_moe_kernel<4, 5>;
    if (args.gu_type == GGML_TYPE_Q4_K && args.dn_type == GGML_TYPE_Q6_K) kern = p60_mega_moe_kernel<4, 6>;
    if (args.gu_type == GGML_TYPE_Q5_K && args.dn_type == GGML_TYPE_Q5_K) kern = p60_mega_moe_kernel<5, 5>;
    if (args.gu_type == GGML_TYPE_Q5_K && args.dn_type == GGML_TYPE_Q6_K) kern = p60_mega_moe_kernel<5, 6>;
    if (kern == nullptr) {
        return false;
    }
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), 260 + 2 * FF + NU * 2 * FF + D);
    moe_params kp = {};
    kp.a      = args;
    kp.logits = scratch.get();
    kp.shgu   = kp.logits + 260;
    kp.egu    = kp.shgu + 2 * FF;
    kp.shout  = kp.egu + NU * 2 * FF;
    mega_launch(ctx, kern, kp);
    return true;
}

bool pxa_p60_mega_gdn(ggml_backend_cuda_context & ctx, const pxa_p60_gdn_args & args) {
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), QKV + VD + 2 * HV + VD);
    gdn_params kp = {};
    kp.a    = args;
    kp.qkvc = scratch.get();
    kp.zv   = kp.qkvc + QKV;
    kp.gb   = kp.zv + VD;
    kp.o    = kp.gb + 2 * HV;
    mega_launch(ctx, p60_mega_gdn_kernel, kp);
    return true;
}

// ---------------- BF16 router weights (MTP head): one f32 copy per device and source tensor ----------------
namespace {
__global__ void p60_bf16_f32(const uint16_t * __restrict__ x, float * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __uint_as_float(((unsigned) x[i]) << 16);
    }
}
}

const float * pxa_p60_bf16_to_f32(ggml_backend_cuda_context & ctx, const void * src, int64_t n) {
    static std::mutex mtx;
    static std::map<std::pair<int, const void *>, float *> cache;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cache.find({ ctx.device, src });
    if (it != cache.end()) {
        return it->second;
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &st));
    if (st != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
    float * dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, n * sizeof(float)));
    p60_bf16_f32<<<(n + 255) / 256, 256, 0, ctx.stream()>>>((const uint16_t *) src, dst, n);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    cache[{ ctx.device, src }] = dst;
    return dst;
}
