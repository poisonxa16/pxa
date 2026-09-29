// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// kq-mmv.cuh -- decode GEMV for the standard k-quants (Q4_K, Q5_K, Q6_K) and Q8_0 at ny 1..8.
//
// WHY. The incumbent q8_1 MMVQ gives every output row its own 128-thread block, so every block re-reads
// the whole q8_1 activation (for Q6_K at K=5120 that is ~1.4x the weight bytes of the row), loads the
// 2-byte-aligned Q6_K / Q8_0 blocks with 16-bit loads, needs a separate quantize launch per src1, one
// launch per matrix, and on sm_60 runs the int8 dot on an emulated dp4a.
//
// WHAT THIS KERNEL DOES INSTEAD.
//   * ROWS output rows per block, ONE WARP PER ROW (8 warps; 4 for small-R shards). The activation is
//     staged in shared memory ONCE per block and read by every row of the block.
//   * The activation is staged straight from the f32 src1: no q8_1 launch on any path.
//       h2 (sm_60): per 32-block a power-of-two scale that brings |x|max into [16, 32), x stored as
//                   fp16 (exact scaling, one rounding), plus the fp32 per-16 sums of the STAGED x for the
//                   K-quant min term (Q4_K / Q5_K): the min term sees the same x as the dot, so the two
//                   halves of the dequantized weight (d*sc*q - dmin*m) keep cancelling as they should.
//       i8 (sm_61+): the q8_1 arithmetic (d = |x|max/127, q = round(x/d)) with an fp32 d, plus the same
//                   per-16 sums (of q*d); the dot is dp4a.
//   * Weights: each warp copies its row in 1024-element chunks global -> shared with 16-byte loads
//     (aligned down, so 2-byte-aligned Q6_K / Q8_0 rows are fine), and the NEXT chunk is already in
//     flight in registers while the current one is consumed.
//   * sm_60 dot without dp4a: quants are expanded to half2 by a byte permute into the mantissa of
//     1024.0h and one HSUB2, then HFMA2 chains of <= 8 products against the staged fp16 x, folded to
//     fp32 once per 16 elements with d*scale*xscale. The x scale bounds every partial: |q| <= 128,
//     |x| < 32, 8 products per half lane -> < 32768 < 65504.
//   * A group of up to KQMMV_MAX_MATS matrices that share src1 is ONE launch (grid = concatenated row
//     ranges); the per-row arithmetic is identical to a one-matrix launch (bit-identical by design).
//   * FUSED_UP_GATE: one warp computes up[r] and gate[r] with the same per-matrix code and applies the
//     silu(gate)*up epilogue (kq_glu, also used by the standalone GLU kernel below).
//
// DETERMINISM / INVARIANCE. Lane l always takes task l of each 1024-element weight chunk, chunks are
// visited in K order, the fold is per task and the warp reduce is a fixed xor tree. The x chunking
// (which depends on ny) only decides when x is re-staged, never the per-lane accumulation order, so
// column c at ny=k is bit-identical to ny=1, and a grouped launch is bit-identical to per-matrix ones.
//
// Header-only kernels (no ggml includes) so tests/test-kq-mmv.cu can instantiate them directly.
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


#define KQMMV_MAX_NY   8
#define KQMMV_MAX_MATS 4
#define KQMMV_CHUNK    1024      // elements per weight chunk = 32 tasks of 32 elements
#define KQMMV_XSINGLE  24576     // staged-activation bytes up to which the whole K is staged once per block
#define KQMMV_XMULTI   12288     // staged-activation bytes per K chunk when it is not

enum kqmmv_kind { KQ_Q4_K = 0, KQ_Q5_K = 1, KQ_Q6_K = 2, KQ_Q8_0 = 3 };
enum kqmmv_path { KQMMV_H2 = 0, KQMMV_I8 = 1 };

template <int T> struct kq_traits;
template <> struct kq_traits<KQ_Q4_K> { static constexpr int QK = 256, BB = 144, BPC = 4,  TPB = 8; static constexpr bool SUM = true;  };
template <> struct kq_traits<KQ_Q5_K> { static constexpr int QK = 256, BB = 176, BPC = 4,  TPB = 8; static constexpr bool SUM = true;  };
template <> struct kq_traits<KQ_Q6_K> { static constexpr int QK = 256, BB = 210, BPC = 4,  TPB = 8; static constexpr bool SUM = false; };
template <> struct kq_traits<KQ_Q8_0> { static constexpr int QK = 32,  BB = 34,  BPC = 32, TPB = 1; static constexpr bool SUM = false; };

// 16-byte units of one chunk copy (the chunk's bytes plus up to 15 bytes of align-down slack), per lane
template <int T> static constexpr __host__ __device__ int kq_n16() { return (kq_traits<T>::BPC*kq_traits<T>::BB + 15 + 15)/16; }
template <int T> static constexpr __host__ __device__ int kq_nld() { return (kq_n16<T>() + 31)/32; }
// per-warp weight buffer: the chunk + 16 bytes so an unaligned 16-byte read never leaves the buffer
template <int T> static constexpr __host__ __device__ int kq_wbuf() { return kq_n16<T>()*16 + 16; }

struct kqmmv_mat {
    const char * w;      // row 0 of the matrix (GU: up)
    const char * w2;     // GU: gate
    float      * dst;    // output column 0
    const float * bias;  // nullptr, or a per-row vector added to every column (a fused ADD: dst = W*x + bias)
    int64_t      nb01;   // bytes per row
    int64_t      sd;     // dst column stride (floats)
    int          nrows;
    int          vb0;    // first virtual block of this matrix
};

struct kqmmv_args {
    kqmmv_mat     m[KQMMV_MAX_MATS];
    int           nmat;
    int           nvb;   // virtual blocks (row groups of NW rows) over all matrices
    const float * x;     // src1 column 0 (16-byte aligned)
    int64_t       sx;    // src1 column stride (floats, multiple of 4)
    int           K;
    int           XC;    // activation chunk (elements, multiple of KQMMV_CHUNK); XC >= K = staged once
    float         limit; // GU: the op's limit (< 1e-6 = the plain silu form)
};

static inline double kqmmv_bpe(int T, int path) {
    const bool sm = T == KQ_Q4_K || T == KQ_Q5_K;
    return (path == KQMMV_H2 ? 2.0 : 1.0) + 4.0/32 + (sm ? 4.0/16 : 0.0);
}

// host: bytes of dynamic shared memory for a launch
static inline int kqmmv_smem_bytes(int T, int path, int ny, int XC, int nwarps) {
    const int xe  = path == KQMMV_H2 ? 2 : 1;
    const bool sm = T == KQ_Q4_K || T == KQ_Q5_K;
    const int wb  = T == KQ_Q4_K ? kq_wbuf<KQ_Q4_K>() : T == KQ_Q5_K ? kq_wbuf<KQ_Q5_K>()
                  : T == KQ_Q6_K ? kq_wbuf<KQ_Q6_K>() : kq_wbuf<KQ_Q8_0>();
    return ny*XC*xe + ny*(XC/32)*4 + (sm ? ny*(XC/16)*4 : 0) + nwarps*wb;
}

// host: the activation chunk for (type, path, ny, K). The whole (1024-rounded) K when it fits
// KQMMV_XSINGLE: the block then stages x once and walks many row groups. Otherwise K chunks that keep
// the block small (one row group per block, x re-staged per chunk).
static inline int kqmmv_xchunk(int T, int path, int ny, int K) {
    const double bpe = kqmmv_bpe(T, path);
    const int kr = (K + KQMMV_CHUNK - 1)/KQMMV_CHUNK*KQMMV_CHUNK;
    if (ny*kr*bpe <= KQMMV_XSINGLE) return kr;
    int xc = (int)(KQMMV_XMULTI/(ny*bpe)) / KQMMV_CHUNK * KQMMV_CHUNK;
    if (xc < KQMMV_CHUNK) xc = KQMMV_CHUNK;
    return xc < kr ? xc : kr;
}


// ------------------------------------------------------------------------------------------------
// device helpers
// ------------------------------------------------------------------------------------------------
static __device__ __forceinline__ half2 kq_u2h2(uint32_t u) {
    half2 h; *reinterpret_cast<uint32_t *>(&h) = u; return h;
}

// 16 bytes at an arbitrary byte offset of a shared buffer -> four little-endian words
static __device__ __forceinline__ void kq_ld16(const unsigned char * b, int o, uint32_t w[4]) {
    const uint32_t * p = reinterpret_cast<const uint32_t *>(b + (o & ~3));
    const uint32_t sh = (uint32_t)(o & 3)*8;
    const uint32_t u0 = p[0], u1 = p[1], u2 = p[2], u3 = p[3], u4 = p[4];
    w[0] = __funnelshift_r(u0, u1, sh);
    w[1] = __funnelshift_r(u1, u2, sh);
    w[2] = __funnelshift_r(u2, u3, sh);
    w[3] = __funnelshift_r(u3, u4, sh);
}

static __device__ __forceinline__ float kq_ldh(const unsigned char * b, int o) {
    const unsigned short v = (unsigned short)(b[o] | (b[o + 1] << 8));
    return __half2float(__ushort_as_half(v));
}

static __device__ __forceinline__ int kq_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * va = reinterpret_cast<const int8_t *>(&a);
    const int8_t * vb = reinterpret_cast<const int8_t *>(&b);
    return c + va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
#endif
}

// the K-quant 6-bit scale / min of sub-block j (0..7) from the 12 packed bytes, as three little-endian
// words W0 = q[0..3], W1 = q[4..7], W2 = q[8..11]; branch-free (neighbouring lanes take sub-blocks from both halves)
static __device__ __forceinline__ void kq_scale_min_w(uint32_t W0, uint32_t W1, uint32_t W2, int j, int & sc, int & m) {
    const int jl = (j & 3)*8;
    const int b0 = (W0 >> jl) & 0xff, b1 = (W1 >> jl) & 0xff, b2 = (W2 >> jl) & 0xff;
    const int sc_a = b0 & 63,                          m_a = b1 & 63;
    const int sc_b = (b2 & 0xF) | ((b0 >> 6) << 4),    m_b = (b2 >> 4) | ((b1 >> 6) << 4);
    sc = j < 4 ? sc_a : sc_b;
    m  = j < 4 ? m_a  : m_b;
}

// the GLU of FUSED_UP_GATE (silu), g = gate result, u = up result. Same expression and selection as
// the engine's fused_mul_silu (limit < 1e-6 -> the plain form).
static __device__ __forceinline__ float kq_glu(float g, float u, float limit) {
    if (limit < 1e-6f) {
        return g * u / (1.0f + expf(-g));
    }
    float s = g / (1.0f + expf(-g));
    s = fminf(s, limit);
    return s * fmaxf(-limit, fminf(limit, u));
}

// ------------------------------------------------------------------------------------------------
// one 16-element part: v[4] = the 16 quant bytes (element e in byte e%4 of word e/4), already in the
// path's form (h2: unsigned code, bias subtracted by HSUB2; i8: signed int8 value)
// ------------------------------------------------------------------------------------------------
template <int NY, int P>
struct kq_part {
    // h2: decoded half2 pairs; i8: the words themselves
    uint32_t q[8];

    static __device__ __forceinline__ void make(kq_part & r, const uint32_t v[4], uint32_t bias2) {
        if constexpr (P == KQMMV_H2) {
            const half2 b = kq_u2h2(bias2);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const half2 lo = __hsub2(kq_u2h2(__byte_perm(v[i], 0x64646464u, 0x4140)), b);
                const half2 hi = __hsub2(kq_u2h2(__byte_perm(v[i], 0x64646464u, 0x4342)), b);
                r.q[2*i + 0] = *reinterpret_cast<const uint32_t *>(&lo);
                r.q[2*i + 1] = *reinterpret_cast<const uint32_t *>(&hi);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) r.q[i] = v[i];
        }
    }

    // raw dot of this part with 16 staged elements of one column (h2: fp16 pairs; i8: int8)
    __device__ __forceinline__ float dot(const unsigned char * xe) const {
        if constexpr (P == KQMMV_H2) {
            const uint4 xa = *reinterpret_cast<const uint4 *>(xe);
            const uint4 xb = *reinterpret_cast<const uint4 *>(xe + 16);
            half2 a = __hmul2(kq_u2h2(q[0]), kq_u2h2(xa.x));
            a = __hfma2(kq_u2h2(q[1]), kq_u2h2(xa.y), a);
            a = __hfma2(kq_u2h2(q[2]), kq_u2h2(xa.z), a);
            a = __hfma2(kq_u2h2(q[3]), kq_u2h2(xa.w), a);
            a = __hfma2(kq_u2h2(q[4]), kq_u2h2(xb.x), a);
            a = __hfma2(kq_u2h2(q[5]), kq_u2h2(xb.y), a);
            a = __hfma2(kq_u2h2(q[6]), kq_u2h2(xb.z), a);
            a = __hfma2(kq_u2h2(q[7]), kq_u2h2(xb.w), a);
            return __low2float(a) + __high2float(a);
        } else {
            const uint4 xa = *reinterpret_cast<const uint4 *>(xe);
            int s = kq_dp4a((int)q[0], (int)xa.x, 0);
            s = kq_dp4a((int)q[1], (int)xa.y, s);
            s = kq_dp4a((int)q[2], (int)xa.z, s);
            s = kq_dp4a((int)q[3], (int)xa.w, s);
            return (float)s;
        }
    }
};

// shared-memory view of one staged activation chunk
template <int P>
struct kq_xview {
    const unsigned char * xq;    // column j at xq + j*XC*XE
    const float         * xs;    // column j at xs + j*(XC/32)
    const float         * xsum;  // column j at xsum + j*(XC/16)
    int                   XC;
    static constexpr int XE = P == KQMMV_H2 ? 2 : 1;
    __device__ __forceinline__ const unsigned char * elem(int j, int e) const { return xq + ((size_t)j*XC + e)*XE; }
    __device__ __forceinline__ float scale(int j, int e) const { return xs[j*(XC/32) + (e >> 5)]; }
    __device__ __forceinline__ float sum16(int j, int e) const { return xsum[j*(XC/16) + (e >> 4)]; }
};

// ------------------------------------------------------------------------------------------------
// one task (32 elements of one row) of one weight chunk; wb = this warp's chunk buffer, o = byte
// offset of the chunk's first block in wb, e0 = chunk-relative element base of the chunk in x.
// ------------------------------------------------------------------------------------------------
template <int T, int NY, int P>
static __device__ __forceinline__ void kq_task(const unsigned char * wb, int o, int t, int e0,
                                               const kq_xview<P> & X, float (&acc)[NY]) {
    if constexpr (T == KQ_Q6_K) {
        const int sb = t >> 3, s = t & 7, n = s >> 2, hh = (s >> 1) & 1, p = s & 1;
        const int ob = o + sb*210;
        uint32_t ql[4], qh[4], lo[4], hi[4];
        kq_ld16(wb, ob + n*64 + 32*p + 16*hh, ql);
        kq_ld16(wb, ob + 128 + n*32 + 16*hh, qh);
        const int   sc_lo = (int8_t)wb[ob + 192 + n*8 + 2*p + hh];
        const int   sc_hi = (int8_t)wb[ob + 192 + n*8 + 2*p + 4 + hh];
        const float d     = kq_ldh(wb, ob + 208);
        const int   s_lo  = 2*p, s_hi = 2*p + 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] = (ql[i] & 0x0F0F0F0Fu)        | (((qh[i] >> s_lo) & 0x03030303u) << 4);
            hi[i] = ((ql[i] >> 4) & 0x0F0F0F0Fu) | (((qh[i] >> s_hi) & 0x03030303u) << 4);
            if constexpr (P == KQMMV_I8) {
                lo[i] = __vsub4(lo[i], 0x20202020u);
                hi[i] = __vsub4(hi[i], 0x20202020u);
            }
        }
        kq_part<NY, P> A, B;
        kq_part<NY, P>::make(A, lo, 0x64206420u);   // 1056 = 1024 + 32
        kq_part<NY, P>::make(B, hi, 0x64206420u);
        const float ds_lo = d*(float)sc_lo, ds_hi = d*(float)sc_hi;
        const int E_lo = e0 + sb*256 + n*128 + 32*p + 16*hh, E_hi = E_lo + 64;
#pragma unroll
        for (int j = 0; j < NY; ++j) {
            acc[j] = fmaf(A.dot(X.elem(j, E_lo)), ds_lo*X.scale(j, E_lo), acc[j]);
            acc[j] = fmaf(B.dot(X.elem(j, E_hi)), ds_hi*X.scale(j, E_hi), acc[j]);
        }
    } else if constexpr (T == KQ_Q4_K || T == KQ_Q5_K) {
        constexpr int BB = kq_traits<T>::BB;
        constexpr int QS = T == KQ_Q4_K ? 16 : 48;
        const int sb = t >> 3, s = t & 7, jj = s >> 1, tt = s & 1;
        const int ob = o + sb*BB;
        uint32_t qs[4], lo[4], hi[4];
        kq_ld16(wb, ob + QS + 32*jj + 16*tt, qs);
        if constexpr (T == KQ_Q5_K) {
            uint32_t qh[4];
            kq_ld16(wb, ob + 16 + 16*tt, qh);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                lo[i] = (qs[i] & 0x0F0F0F0Fu)        | (((qh[i] >> (2*jj    )) & 0x01010101u) << 4);
                hi[i] = ((qs[i] >> 4) & 0x0F0F0F0Fu) | (((qh[i] >> (2*jj + 1)) & 0x01010101u) << 4);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                lo[i] = qs[i] & 0x0F0F0F0Fu;
                hi[i] = (qs[i] >> 4) & 0x0F0F0F0Fu;
            }
        }
        // header in one 16-byte read: d | dmin, then the 12 packed scale/min bytes; branch-free 6-bit unpack
        uint32_t hd[4];
        kq_ld16(wb, ob, hd);
        const half2 dd = kq_u2h2(hd[0]);
        const float d = __low2float(dd), dmin = __high2float(dd);
        int sc_lo, m_lo, sc_hi, m_hi;
        kq_scale_min_w(hd[1], hd[2], hd[3], 2*jj,     sc_lo, m_lo);
        kq_scale_min_w(hd[1], hd[2], hd[3], 2*jj + 1, sc_hi, m_hi);
        kq_part<NY, P> A, B;
        kq_part<NY, P>::make(A, lo, 0x64006400u);
        kq_part<NY, P>::make(B, hi, 0x64006400u);
        const float ds_lo = d*(float)sc_lo, ds_hi = d*(float)sc_hi;
        const float dm_lo = dmin*(float)m_lo, dm_hi = dmin*(float)m_hi;
        const int E_lo = e0 + sb*256 + 64*jj + 16*tt, E_hi = E_lo + 32;
#pragma unroll
        for (int j = 0; j < NY; ++j) {
            acc[j] = fmaf(A.dot(X.elem(j, E_lo)), ds_lo*X.scale(j, E_lo), acc[j]);
            acc[j] = fmaf(-dm_lo, X.sum16(j, E_lo), acc[j]);
            acc[j] = fmaf(B.dot(X.elem(j, E_hi)), ds_hi*X.scale(j, E_hi), acc[j]);
            acc[j] = fmaf(-dm_hi, X.sum16(j, E_hi), acc[j]);
        }
    } else {   // Q8_0
        const int ob = o + t*34;
        uint32_t a[4], b[4];
        kq_ld16(wb, ob + 2,  a);
        kq_ld16(wb, ob + 18, b);
        if constexpr (P == KQMMV_H2) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] ^= 0x80808080u; b[i] ^= 0x80808080u; }
        }
        const float d = kq_ldh(wb, ob);
        kq_part<NY, P> A, B;
        kq_part<NY, P>::make(A, a, 0x64806480u);    // 1152 = 1024 + 128
        kq_part<NY, P>::make(B, b, 0x64806480u);
        const int E = e0 + t*32;
#pragma unroll
        for (int j = 0; j < NY; ++j) {
            const float f = A.dot(X.elem(j, E)) + B.dot(X.elem(j, E + 16));
            acc[j] = fmaf(f, d*X.scale(j, E), acc[j]);
        }
    }
}

// stage columns 0..NY-1 of x[k0, k0 + Kc) into shared memory (h2: fp16 + per-32 power-of-two scale; i8: q8_1
// arithmetic), plus the per-16 sums of the staged x when SUM. Element e of the chunk depends only on its own
// 32-block, so any 32-aligned chunking stages bit-identical values (the invariance of every mode rests on it).
template <int NY, int P, bool SUM>
static __device__ __forceinline__ void kq_stage(const float * x, int64_t sx, int XC, int k0, int Kc,
                                                unsigned char * xq, float * xs, float * xsum, int lane, int warp, int NW) {
    // 128-element groups: lane l holds elements 4l..4l+3 (a 32-block = 8 lanes, a 16-run = 4 lanes);
    // four groups per warp per step, all loads first
    const int nb32  = Kc/32;
    const int ngr   = (nb32 + 3)/4;
    const int total = NY*ngr;
    for (int g0 = warp; g0 < total; g0 += 4*NW) {
        float4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int g = g0 + u*NW;
            const int j = g/ngr, gi = g - j*ngr;
            const bool ok = g < total && gi*4 + (lane >> 3) < nb32;
            v[u] = ok ? *reinterpret_cast<const float4 *>(x + (int64_t)j*sx + k0 + gi*128 + lane*4)
                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int g = g0 + u*NW;
            if (g >= total) break;                       // warp-uniform
            const int j = g/ngr, gi = g - j*ngr;
            const int b32 = gi*4 + (lane >> 3);
            const bool ok = b32 < nb32;
            float am = fmaxf(fmaxf(fabsf(v[u].x), fabsf(v[u].y)), fmaxf(fabsf(v[u].z), fabsf(v[u].w)));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            const int e = gi*128 + lane*4;                 // chunk-relative element
            float sc, s;
            if constexpr (P == KQMMV_H2) {
                const int E = (int)((__float_as_uint(am) >> 23) & 0xff);
                const bool okE = E >= 5 && E < 255;
                const float inv = okE ? __uint_as_float((uint32_t)(258 - E) << 23) : 1.0f;
                sc = okE ? __uint_as_float((uint32_t)(E - 4) << 23) : 1.0f;
                const half2 h01 = __floats2half2_rn(v[u].x*inv, v[u].y*inv);
                const half2 h23 = __floats2half2_rn(v[u].z*inv, v[u].w*inv);
                if (ok) {
                    uint2 st;
                    st.x = *reinterpret_cast<const uint32_t *>(&h01);
                    st.y = *reinterpret_cast<const uint32_t *>(&h23);
                    *reinterpret_cast<uint2 *>(xq + ((size_t)j*XC + e)*2) = st;
                }
                // the min term sees the staged x (what the dot sees)
                s = (__low2float(h01) + __high2float(h01) + __low2float(h23) + __high2float(h23))*sc;
            } else {
                const float d = am/127;
                int q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                if (am != 0.0f) {
                    q0 = __float2int_rn(__fdiv_rn(v[u].x, d)); q1 = __float2int_rn(__fdiv_rn(v[u].y, d));
                    q2 = __float2int_rn(__fdiv_rn(v[u].z, d)); q3 = __float2int_rn(__fdiv_rn(v[u].w, d));
                }
                if (ok) {
                    *reinterpret_cast<uint32_t *>(xq + (size_t)j*XC + e) =
                        (uint32_t)(q0 & 255) | ((uint32_t)(q1 & 255) << 8) | ((uint32_t)(q2 & 255) << 16) | ((uint32_t)(q3 & 255) << 24);
                }
                sc = d;
                s = (float)(q0 + q1 + q2 + q3)*d;
            }
            if (ok && (lane & 7) == 0) xs[j*(XC/32) + b32] = sc;
            if constexpr (SUM) {
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                if (ok && (lane & 3) == 0) xsum[j*(XC/16) + (e >> 4)] = s;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// the kernel. blockDim = (32, NW): NW rows per virtual block (vb), one per warp. Virtual blocks are the
// concatenated row groups of the launch's matrices. When the whole K is staged (XC >= K) a block stages
// x once and walks vb = blockIdx.x, +gridDim.x, ...; otherwise gridDim.x == nvb and the block walks the
// K chunks of its one vb. GU: m[0].w = up, m[0].w2 = gate, the glu goes to m[0].dst.
// Per warp the weight chunks are a flat item list (vb, matrix, chunk); two items are always in flight in
// registers while the current one is consumed from the warp's shared buffer.
// ------------------------------------------------------------------------------------------------
template <int T, int NY, int P, bool GU>
static __global__ void __launch_bounds__(256) k_kq_mmv(const kqmmv_args a) {
    using TR = kq_traits<T>;
    constexpr int NMAT = GU ? 2 : 1;
    constexpr int NLD  = kq_nld<T>();
    constexpr int WBUF = kq_wbuf<T>();
    constexpr int CHB  = TR::BPC*TR::BB;               // bytes of a full 1024-element chunk
    constexpr int XE   = P == KQMMV_H2 ? 2 : 1;

    extern __shared__ __align__(16) unsigned char kq_smem[];

    const int lane = threadIdx.x, warp = threadIdx.y, NW = blockDim.y;
    const int XC = a.XC, K = a.K;

    unsigned char * xq   = kq_smem;
    float         * xs   = reinterpret_cast<float *>(kq_smem + (size_t)NY*XC*XE);
    float         * xsum = xs + NY*(XC/32);
    unsigned char * wbuf = reinterpret_cast<unsigned char *>(xsum + (TR::SUM ? NY*(XC/16) : 0)) + (size_t)warp*WBUF;
    const kq_xview<P> X{xq, xs, xsum, XC};

    const bool single = XC >= K;
    const int  nxc    = single ? 1 : (K + XC - 1)/XC;
    const int  vstep  = single ? (int)gridDim.x : a.nvb;
    const int  nvk    = ((int)a.nvb - (int)blockIdx.x + vstep - 1)/vstep;

    // the warp's row of virtual block vb (constant-index parameter reads only)
    struct rowref { const char * w; const char * w2; float * dst; const float * bias; int64_t sd; int row; };
    auto resolve = [&](int vb, rowref & r) -> bool {
        const char * w = a.m[0].w; const char * w2 = a.m[0].w2; float * dst = a.m[0].dst; const float * bias = a.m[0].bias;
        int64_t nb01 = a.m[0].nb01, sd = a.m[0].sd; int nrows = a.m[0].nrows, vb0 = a.m[0].vb0;
        if constexpr (!GU) {
#pragma unroll
            for (int i = 1; i < KQMMV_MAX_MATS; ++i) {
                if (i < a.nmat && vb >= a.m[i].vb0) {
                    w = a.m[i].w; dst = a.m[i].dst; bias = a.m[i].bias; nb01 = a.m[i].nb01; sd = a.m[i].sd; nrows = a.m[i].nrows; vb0 = a.m[i].vb0;
                }
            }
        }
        const int row = (vb - vb0)*NW + warp;
        if (row >= nrows) return false;
        r.w = w + (int64_t)row*nb01; r.w2 = GU ? w2 + (int64_t)row*nb01 : nullptr; r.dst = dst; r.bias = bias; r.sd = sd; r.row = row;
        return true;
    };

    float acc0[NY], acc1[NY];
#pragma unroll
    for (int j = 0; j < NY; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }

    uint4 pA[NLD], pB[NLD];

    for (int xc = 0; xc < nxc; ++xc) {
        const int k0  = xc*XC;
        const int Kc  = min(XC, K - k0);
        const int nwc = (Kc + KQMMV_CHUNK - 1)/KQMMV_CHUNK;
        const int cb0 = k0/KQMMV_CHUNK;
        const bool lastx = xc == nxc - 1;

        const int tailb = (Kc - (nwc - 1)*KQMMV_CHUNK)/TR::QK*TR::BB;   // bytes of the last chunk

        // a position in the warp's item list (virtual block, matrix, chunk) and its row pointers
        struct kq_cur { int vk, m, c; const char * w0; const char * w1; bool valid; };
        auto cur_row = [&](kq_cur & q) {
            rowref r;
            q.valid = q.vk < nvk && resolve((int)blockIdx.x + q.vk*vstep, r);
            if (q.valid) { q.w0 = r.w; q.w1 = r.w2; }
        };
        auto cur_next = [&](kq_cur & q) {
            if (++q.c == nwc) { q.c = 0; if (++q.m == NMAT) { q.m = 0; ++q.vk; cur_row(q); } }
        };
        auto cur_addr = [&](const kq_cur & q, int & nbytes) -> const char * {
            nbytes = q.c == nwc - 1 ? tailb : CHB;
            return (q.m == 0 ? q.w0 : q.w1) + (int64_t)(cb0 + q.c)*CHB;
        };
        auto issue = [&](uint4 (&p)[NLD], const kq_cur & q) {
            if (q.vk >= nvk || !q.valid) return;
            int nbytes; const char * A = cur_addr(q, nbytes);
            const uint4 * A0 = reinterpret_cast<const uint4 *>((uintptr_t)A & ~(uintptr_t)15);
            const int n16 = (int)(((uintptr_t)A & 15) + nbytes + 15) >> 4;
#pragma unroll
            for (int i = 0; i < NLD; ++i) {
                const int idx = lane + 32*i;
                if (idx < n16) p[i] = __ldg(A0 + idx);
            }
        };
        kq_cur qi{0, 0, 0, nullptr, nullptr, false}, qc{0, 0, 0, nullptr, nullptr, false};
        cur_row(qi); cur_row(qc);

        // consume the chunk at qc from p; p is refilled with the chunk at qi (two ahead) as soon as it is free
        auto consume = [&](uint4 (&p)[NLD]) {
            int nbytes = 0, off0 = 0;
            if (qc.valid) {
                const char * A = cur_addr(qc, nbytes);
                off0 = (int)((uintptr_t)A & 15);
                const int n16 = (off0 + nbytes + 15) >> 4;
                __syncwarp();
#pragma unroll
                for (int i = 0; i < NLD; ++i) {
                    const int idx = lane + 32*i;
                    if (idx < n16) reinterpret_cast<uint4 *>(wbuf)[idx] = p[i];
                }
                __syncwarp();
            }
            issue(p, qi); cur_next(qi);
            if (qc.valid) {
                const int ntask = nbytes/TR::BB*TR::TPB;
                if (lane < ntask) {
                    if (GU && qc.m == 1) kq_task<T, NY, P>(wbuf, off0, lane, qc.c*KQMMV_CHUNK, X, acc1);
                    else                 kq_task<T, NY, P>(wbuf, off0, lane, qc.c*KQMMV_CHUNK, X, acc0);
                }
                if (lastx && qc.m == NMAT - 1 && qc.c == nwc - 1) {
                    rowref r; resolve((int)blockIdx.x + qc.vk*vstep, r);
#pragma unroll
                    for (int j = 0; j < NY; ++j) {
                        float v0 = acc0[j], v1 = acc1[j];
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            v0 += __shfl_xor_sync(0xffffffffu, v0, off);
                            if (GU) v1 += __shfl_xor_sync(0xffffffffu, v1, off);
                        }
                        if (lane == 0) {
                            if (GU)          r.dst[(int64_t)j*r.sd + r.row] = kq_glu(v1, v0, a.limit);
                            else if (r.bias) r.dst[(int64_t)j*r.sd + r.row] = v0 + r.bias[r.row];   // == GEMV then ADD
                            else             r.dst[(int64_t)j*r.sd + r.row] = v0;
                        }
                        acc0[j] = 0.0f; acc1[j] = 0.0f;
                    }
                }
            }
            cur_next(qc);
        };

        // the first two weight chunks are in flight while the activation is staged
        issue(pA, qi); cur_next(qi);
        issue(pB, qi); cur_next(qi);

        if (xc > 0) __syncthreads();
        kq_stage<NY, P, TR::SUM>(a.x, a.sx, XC, k0, Kc, xq, xs, xsum, lane, warp, NW);
        __syncthreads();

        while (qc.vk < nvk) {
            consume(pA);
            if (qc.vk >= nvk) break;
            consume(pB);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// the wide form (PXA_KQMMV_WIDE), Q8_0: a non-GU launch whose activation does not fit the block (h2 ny >= 3,
// i8 ny >= 5 at K = 5120: the LM head and the attention / delta-net projections at spec-verify widths).
//   * Each block owns NW*2*RS rows of one matrix over the whole K; x is staged in K chunks of XC elements and
//     every staged chunk is consumed by all of the block's rows before the next one is staged, so x is
//     re-read once per NW*2*RS rows (k_kq_mmv's chunked form re-reads it per NW rows).
//   * A warp works on a PAIR of rows at a time (RS pairs in sequence): both rows' chunks sit in the warp's two
//     shared buffers, and each staged x value a lane loads from shared memory feeds both rows' dots, which
//     halves the shared-memory x traffic that bounds the h2 dot at ny > 1 on sm_60.
//   * The next pair's two chunks are in flight in registers while the current pair is consumed.
// Per row: lane l takes task l of each 1024-element weight chunk, chunks in K order, the Q8_0 task arithmetic
// of kq_task written out once per row of the pair, the same staged x (kq_stage) and the same xor reduce as
// k_kq_mmv -> column c at ny = k is bit-identical to the ny = 1 launch of column c.
// ------------------------------------------------------------------------------------------------
template <int NY, int P>
static __device__ __forceinline__ void kq_task_q8_pair(const unsigned char * wb0, int o0, const unsigned char * wb1, int o1,
                                                       int t, int e0, const kq_xview<P> & X,
                                                       float (&acc0)[NY], float (&acc1)[NY]) {
    uint32_t a0[4], b0[4], a1[4], b1[4];
    kq_ld16(wb0, o0 + t*34 + 2,  a0);
    kq_ld16(wb0, o0 + t*34 + 18, b0);
    kq_ld16(wb1, o1 + t*34 + 2,  a1);
    kq_ld16(wb1, o1 + t*34 + 18, b1);
    if constexpr (P == KQMMV_H2) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { a0[i] ^= 0x80808080u; b0[i] ^= 0x80808080u; a1[i] ^= 0x80808080u; b1[i] ^= 0x80808080u; }
    }
    const float d0 = kq_ldh(wb0, o0 + t*34);
    const float d1 = kq_ldh(wb1, o1 + t*34);
    kq_part<NY, P> A0, B0, A1, B1;
    kq_part<NY, P>::make(A0, a0, 0x64806480u);
    kq_part<NY, P>::make(B0, b0, 0x64806480u);
    kq_part<NY, P>::make(A1, a1, 0x64806480u);
    kq_part<NY, P>::make(B1, b1, 0x64806480u);
    const int E = e0 + t*32;
#pragma unroll
    for (int j = 0; j < NY; ++j) {
        const unsigned char * xe = X.elem(j, E);
        const unsigned char * xf = X.elem(j, E + 16);
        const float sc = X.scale(j, E);
        const float f0 = A0.dot(xe) + B0.dot(xf);
        const float f1 = A1.dot(xe) + B1.dot(xf);
        acc0[j] = fmaf(f0, d0*sc, acc0[j]);
        acc1[j] = fmaf(f1, d1*sc, acc1[j]);
    }
}

template <int NY, int P, int RS>
static __global__ void __launch_bounds__(256) k_kq_mmv_rw(const kqmmv_args a) {
    using TR = kq_traits<KQ_Q8_0>;
    constexpr int NLD  = kq_nld<KQ_Q8_0>();
    constexpr int WBUF = kq_wbuf<KQ_Q8_0>();
    constexpr int CHB  = TR::BPC*TR::BB;
    constexpr int XE   = P == KQMMV_H2 ? 2 : 1;

    extern __shared__ __align__(16) unsigned char kq_smem[];

    const int lane = threadIdx.x, warp = threadIdx.y, NW = blockDim.y;
    const int XC = a.XC, K = a.K;

    unsigned char * xq   = kq_smem;
    float         * xs   = reinterpret_cast<float *>(kq_smem + (size_t)NY*XC*XE);
    unsigned char * wb0  = reinterpret_cast<unsigned char *>(xs + NY*(XC/32)) + (size_t)(2*warp)*WBUF;
    unsigned char * wb1  = wb0 + WBUF;
    const kq_xview<P> X{xq, xs, nullptr, XC};

    // this block's matrix (virtual blocks of NW*2*RS rows, concatenated over the launch's matrices)
    const int vb = blockIdx.x;
    const char * w = a.m[0].w; float * dst = a.m[0].dst; const float * bias = a.m[0].bias;
    int64_t nb01 = a.m[0].nb01, sd = a.m[0].sd; int nrows = a.m[0].nrows, vb0 = a.m[0].vb0;
#pragma unroll
    for (int i = 1; i < KQMMV_MAX_MATS; ++i) {
        if (i < a.nmat && vb >= a.m[i].vb0) {
            w = a.m[i].w; dst = a.m[i].dst; bias = a.m[i].bias; nb01 = a.m[i].nb01; sd = a.m[i].sd; nrows = a.m[i].nrows; vb0 = a.m[i].vb0;
        }
    }
    const int rbase = (vb - vb0)*NW*2*RS + warp;          // row of slot q (0 .. 2*RS-1): rbase + q*NW

    const int nwc   = (K + KQMMV_CHUNK - 1)/KQMMV_CHUNK;  // weight chunks per row
    const int tailb = (K - (nwc - 1)*KQMMV_CHUNK)/TR::QK*TR::BB;
    const int nit   = nwc*RS;                             // items (chunk g, pair s), it = g*RS + s

    // chunk of item it, row q (0/1) of its pair; nullptr = past the end or no such row
    auto item_addr = [&](int it, int q, int & nbytes) -> const char * {
        if (it >= nit) return nullptr;
        const int g = it/RS, sp = it - (it/RS)*RS;
        const int row = rbase + (2*sp + q)*NW;
        if (row >= nrows) return nullptr;
        nbytes = g == nwc - 1 ? tailb : CHB;
        return w + (int64_t)row*nb01 + (int64_t)g*CHB;
    };
    auto issue = [&](uint4 (&p)[NLD], int it, int q) {
        int nbytes = 0; const char * A = item_addr(it, q, nbytes);
        if (!A) return;
        const uint4 * A0 = reinterpret_cast<const uint4 *>((uintptr_t)A & ~(uintptr_t)15);
        const int n16 = (int)(((uintptr_t)A & 15) + nbytes + 15) >> 4;
#pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int idx = lane + 32*i;
            if (idx < n16) p[i] = __ldg(A0 + idx);
        }
    };
    auto put = [&](const uint4 (&p)[NLD], unsigned char * wb, const char * A, int nbytes) {
        const int n16 = ((int)((uintptr_t)A & 15) + nbytes + 15) >> 4;
#pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int idx = lane + 32*i;
            if (idx < n16) reinterpret_cast<uint4 *>(wb)[idx] = p[i];
        }
    };

    float acc[RS][2][NY];
#pragma unroll
    for (int sp = 0; sp < RS; ++sp)
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int j = 0; j < NY; ++j) acc[sp][q][j] = 0.0f;

    uint4 p0[NLD], p1[NLD];
    issue(p0, 0, 0);
    issue(p1, 0, 1);

    const int gpx = XC/KQMMV_CHUNK;                       // weight chunks per x chunk
    const int nxc = (K + XC - 1)/XC;
    for (int xc = 0; xc < nxc; ++xc) {
        const int k0 = xc*XC;
        const int Kc = min(XC, K - k0);
        if (xc > 0) __syncthreads();
        kq_stage<NY, P, false>(a.x, a.sx, XC, k0, Kc, xq, xs, nullptr, lane, warp, NW);
        __syncthreads();
        const int g1 = min(nwc, (xc + 1)*gpx);
        for (int g = xc*gpx; g < g1; ++g) {
#pragma unroll
            for (int sp = 0; sp < RS; ++sp) {
                const int it = g*RS + sp;
                int n0 = 0, n1 = 0;
                const char * A0 = item_addr(it, 0, n0);
                const char * A1 = item_addr(it, 1, n1);
                __syncwarp();
                if (A0) put(p0, wb0, A0, n0);
                if (A1) put(p1, wb1, A1, n1);
                __syncwarp();
                issue(p0, it + 1, 0);
                issue(p1, it + 1, 1);
                if (A0) {                                   // warp-uniform; the pair's rows share the chunk shape
                    const int ntask = n0/TR::BB*TR::TPB;
                    if (lane < ntask) kq_task_q8_pair<NY, P>(wb0, (int)((uintptr_t)A0 & 15), wb1, A1 ? (int)((uintptr_t)A1 & 15) : 0,
                                                             lane, (g - xc*gpx)*KQMMV_CHUNK, X, acc[sp][0], acc[sp][1]);
                }
            }
        }
    }

#pragma unroll
    for (int sp = 0; sp < RS; ++sp) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int row = rbase + (2*sp + q)*NW;
            if (row >= nrows) continue;                    // warp-uniform
#pragma unroll
            for (int j = 0; j < NY; ++j) {
                float v0 = acc[sp][q][j];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) v0 += __shfl_xor_sync(0xffffffffu, v0, off);
                if (lane == 0) {
                    if (bias) dst[(int64_t)j*sd + row] = v0 + bias[row];   // == GEMV then ADD
                    else      dst[(int64_t)j*sd + row] = v0;
                }
            }
        }
    }
}

// the standalone GLU (tests: GEMV + GEMV + GLU must equal the fused epilogue bit for bit)
static __global__ void k_kq_glu(const float * __restrict__ gate, const float * __restrict__ up,
                                float * __restrict__ dst, int n, float limit) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = kq_glu(gate[i], up[i], limit);
}

// host: resident blocks per SM of one instance at (nwarps, smem), cached (packed: smem << 12 | nw << 6 | occ)
template <int T, int NY, int P, bool GU>
static int kqmmv_occupancy(int nwarps, int smem) {
    static uint64_t cache[8] = {0};
    const uint64_t key = ((uint64_t)smem << 12) | ((uint64_t)nwarps << 6);
    for (int i = 0; i < 8; ++i) {
        const uint64_t c = __atomic_load_n(&cache[i], __ATOMIC_ACQUIRE);
        if (c && (c & ~(uint64_t)63) == key) return (int)(c & 63);
    }
    int occ = 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_kq_mmv<T, NY, P, GU>, 32*nwarps, smem) != cudaSuccess || occ < 1) {
        (void) cudaGetLastError();
        occ = 1;
    }
    if (occ > 63) occ = 63;
    for (int i = 0; i < 8; ++i) {
        uint64_t z = 0;
        if (__atomic_compare_exchange_n(&cache[i], &z, key | (uint64_t)occ, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) break;
    }
    return occ;
}

// host: the wide form's x chunk: the largest multiple of KQMMV_CHUNK whose ny staged columns fit
// KQMMV_XRW bytes (PXA_KQMMV_XRW overrides; A/B only). Small on purpose: the x chunk is consumed by all of the
// block's rows, so a short chunk costs only a barrier, and the two chunk buffers per warp need the room.
#define KQMMV_XRW 12288
static inline int kqmmv_xchunk_rw(int T, int path, int ny, int K) {
    static const int bud = [](){ const char * e = getenv("PXA_KQMMV_XRW"); const int v = e ? atoi(e) : 0; return v >= 4096 && v <= 40960 ? v : KQMMV_XRW; }();
    const int kr = (K + KQMMV_CHUNK - 1)/KQMMV_CHUNK*KQMMV_CHUNK;
    int xc = (int)(bud/(ny*kqmmv_bpe(T, path))) / KQMMV_CHUNK * KQMMV_CHUNK;
    if (xc < KQMMV_CHUNK) xc = KQMMV_CHUNK;
    return xc < kr ? xc : kr;
}

// host: row pairs per warp of the wide form (1 or 2; PXA_KQMMV_RS overrides), from the rows of the launch
static inline int kqmmv_rs(int64_t rows, int nsm) {
    static const int ov = [](){ const char * e = getenv("PXA_KQMMV_RS"); const int v = e ? atoi(e) : 0; return v == 1 || v == 2 ? v : 0; }();
    if (ov) return ov;
    return rows >= (int64_t)32*2*nsm ? 2 : 1;
}

// host: dynamic shared memory of the wide form (staged x + scales, two chunk buffers per warp)
static inline int kqmmv_smem_rw(int path, int ny, int XC, int nwarps) {
    return ny*XC*(path == KQMMV_H2 ? 2 : 1) + ny*(XC/32)*4 + 2*nwarps*kq_wbuf<KQ_Q8_0>();
}

template <int P>
static cudaError_t kqmmv_launch_rw(const kqmmv_args & a, int ny, int nwarps, int rs, cudaStream_t stream) {
    const int smem = kqmmv_smem_rw(P, ny, a.XC, nwarps);
#define KQRW_CASE(N) case N: if (rs == 2) k_kq_mmv_rw<N, P, 2><<<a.nvb, dim3(32, nwarps, 1), smem, stream>>>(a); \
                             else         k_kq_mmv_rw<N, P, 1><<<a.nvb, dim3(32, nwarps, 1), smem, stream>>>(a); break;
    switch (ny) {
        KQRW_CASE(1) KQRW_CASE(2) KQRW_CASE(3) KQRW_CASE(4) KQRW_CASE(5) KQRW_CASE(6) KQRW_CASE(7) KQRW_CASE(8)
        default: return cudaErrorInvalidValue;
    }
#undef KQRW_CASE
    return cudaGetLastError();
}

// the wide form serves this (type, path, ny, K) of a non-GU launch: x does not fit the block, Q8_0 only
// (the LM head, attention k / v, delta-net beta / alpha; the k-quants keep the chunked k_kq_mmv form)
static inline bool kqmmv_rw_fits(int T, int path, int ny, int K) {
    return T == KQ_Q8_0 && kqmmv_xchunk(T, path, ny, K) < K && ny >= 1 && ny <= KQMMV_MAX_NY;
}

template <int T, int NY, int P, bool GU>
static cudaError_t kqmmv_launch_t(const kqmmv_args & a, int nwarps, int nsm, cudaStream_t stream) {
    const int smem = kqmmv_smem_bytes(T, P, NY, a.XC, nwarps);   // < 48 KiB by construction
    int grid = a.nvb;
    if (a.XC >= a.K) {
        const int cap = nsm*kqmmv_occupancy<T, NY, P, GU>(nwarps, smem);
        if (grid > cap) grid = cap;
    }
    k_kq_mmv<T, NY, P, GU><<<grid, dim3(32, nwarps, 1), smem, stream>>>(a);
    return cudaGetLastError();
}

template <int T, int P, bool GU>
static cudaError_t kqmmv_launch_ny(const kqmmv_args & a, int ny, int nwarps, int nsm, cudaStream_t stream) {
    switch (ny) {
        case 1: return kqmmv_launch_t<T, 1, P, GU>(a, nwarps, nsm, stream);
        case 2: return kqmmv_launch_t<T, 2, P, GU>(a, nwarps, nsm, stream);
        case 3: return kqmmv_launch_t<T, 3, P, GU>(a, nwarps, nsm, stream);
        case 4: return kqmmv_launch_t<T, 4, P, GU>(a, nwarps, nsm, stream);
        case 5: return kqmmv_launch_t<T, 5, P, GU>(a, nwarps, nsm, stream);
        case 6: return kqmmv_launch_t<T, 6, P, GU>(a, nwarps, nsm, stream);
        case 7: return kqmmv_launch_t<T, 7, P, GU>(a, nwarps, nsm, stream);
        case 8: return kqmmv_launch_t<T, 8, P, GU>(a, nwarps, nsm, stream);
        default: return cudaErrorInvalidValue;
    }
}

template <bool GU>
static cudaError_t kqmmv_launch(int T, int P, const kqmmv_args & a, int ny, int nwarps, int nsm, cudaStream_t stream) {
    if (P == KQMMV_H2) {
        switch (T) {
            case KQ_Q4_K: return kqmmv_launch_ny<KQ_Q4_K, KQMMV_H2, GU>(a, ny, nwarps, nsm, stream);
            case KQ_Q5_K: return kqmmv_launch_ny<KQ_Q5_K, KQMMV_H2, GU>(a, ny, nwarps, nsm, stream);
            case KQ_Q6_K: return kqmmv_launch_ny<KQ_Q6_K, KQMMV_H2, GU>(a, ny, nwarps, nsm, stream);
            case KQ_Q8_0: return kqmmv_launch_ny<KQ_Q8_0, KQMMV_H2, GU>(a, ny, nwarps, nsm, stream);
        }
    } else {
        switch (T) {
            case KQ_Q4_K: return kqmmv_launch_ny<KQ_Q4_K, KQMMV_I8, GU>(a, ny, nwarps, nsm, stream);
            case KQ_Q5_K: return kqmmv_launch_ny<KQ_Q5_K, KQMMV_I8, GU>(a, ny, nwarps, nsm, stream);
            case KQ_Q6_K: return kqmmv_launch_ny<KQ_Q6_K, KQMMV_I8, GU>(a, ny, nwarps, nsm, stream);
            case KQ_Q8_0: return kqmmv_launch_ny<KQ_Q8_0, KQMMV_I8, GU>(a, ny, nwarps, nsm, stream);
        }
    }
    return cudaErrorInvalidValue;
}

// host: one launch over n matrices sharing x (gu: n == 2, ms[0] = up, ms[1] = gate, output ms[0].dst).
// Used by the engine (kq-mmv.cu) and by tests/test-kq-mmv.cu, so both build the same launch.
struct kqmmv_mdesc { const void * w; float * dst; int64_t nb01; int64_t sd; int nrows; const float * bias; };

// wide = the non-GU launches whose x does not fit the block take the wide form (k_kq_mmv_rw); false = the
// chunked k_kq_mmv form (one row group per block, x re-staged per K chunk).
static cudaError_t kqmmv_run_mats(int T, int P, int ny, const float * x, int64_t sx, int K,
                                  const kqmmv_mdesc * ms, int n, bool gu, float limit, int nsm, cudaStream_t stream,
                                  bool wide = true) {
    kqmmv_args a;
    memset(&a, 0, sizeof(a));
    a.x = x; a.sx = sx; a.K = K; a.XC = kqmmv_xchunk(T, P, ny, K); a.limit = limit;
    const int nm = gu ? 1 : n;
    if (wide && !gu && kqmmv_rw_fits(T, P, ny, K)) {
        int64_t rows = 0;
        for (int i = 0; i < nm; ++i) rows += ms[i].nrows;
        const int rs  = kqmmv_rs(rows, nsm);
        static const int nw_ov = [](){ const char * e = getenv("PXA_KQMMV_RW_NW"); const int v = e ? atoi(e) : 0; return v == 4 || v == 8 ? v : 0; }();
        int nw = rows >= (int64_t)8*2*rs*nsm ? 8 : 4;
        // two blocks per SM on sm_60 (64 KiB): 8 warps only while the block stays under 32 KiB
        if (nw == 8 && kqmmv_smem_rw(P, ny, kqmmv_xchunk_rw(T, P, ny, K), 8) > 32768) nw = 4;
        if (nw_ov) nw = nw_ov;
        const int rpb = nw*2*rs;
        a.XC = kqmmv_xchunk_rw(T, P, ny, K);
        int vb = 0;
        for (int i = 0; i < nm; ++i) {
            a.m[i].w = (const char *) ms[i].w; a.m[i].dst = ms[i].dst; a.m[i].bias = ms[i].bias;
            a.m[i].nb01 = ms[i].nb01; a.m[i].sd = ms[i].sd; a.m[i].nrows = ms[i].nrows; a.m[i].vb0 = vb;
            vb += (ms[i].nrows + rpb - 1)/rpb;
        }
        a.nmat = nm;
        a.nvb  = vb;
        return P == KQMMV_H2 ? kqmmv_launch_rw<KQMMV_H2>(a, ny, nw, rs, stream)
                             : kqmmv_launch_rw<KQMMV_I8>(a, ny, nw, rs, stream);
    }
    int64_t blocks8 = 0;
    for (int i = 0; i < nm; ++i) blocks8 += (ms[i].nrows + 7)/8;
    const int nw = blocks8 < 2*(int64_t)nsm ? 4 : 8;       // small-R shards: 4 rows per block, twice the blocks
    int vb = 0;
    for (int i = 0; i < nm; ++i) {
        a.m[i].w = (const char *) ms[i].w; a.m[i].w2 = gu ? (const char *) ms[1].w : nullptr;
        a.m[i].dst = ms[i].dst; a.m[i].bias = gu ? nullptr : ms[i].bias; a.m[i].nb01 = ms[i].nb01; a.m[i].sd = ms[i].sd; a.m[i].nrows = ms[i].nrows;
        a.m[i].vb0 = vb;
        vb += (ms[i].nrows + nw - 1)/nw;
    }
    a.nmat = nm;
    a.nvb  = vb;
    return gu ? kqmmv_launch<true >(T, P, a, ny, nw, nsm, stream)
              : kqmmv_launch<false>(T, P, a, ny, nw, nsm, stream);
}


// ------------------------------------------------------------------------------------------------
// engine glue (kq-mmv.cu; ggml-cuda.cu repeats these declarations). Forward declarations only, so this
// header stays free of ggml includes.
// ------------------------------------------------------------------------------------------------
struct ggml_tensor;

// PXA_KQMMV for this device / type (a ggml_type value) / K / width: true = take the kernel above. Taken:
// shapes whose whole activation fits the block's shared memory (x staged once per block, many row groups
// per block), and Q8_0 at the widths the wide form wins (PXA_KQMMV_WIDE: h2 ny 3..4 at K = 5120; the
// launch must also have >= 8192 rows). Other wide shapes stay on the incumbent.
bool ggml_cuda_kqmmv_take(int device, int type, int64_t K, int64_t ny);
// launches below this many rows (a lone small matrix) stay on the incumbent: latency, not bytes
#define KQMMV_MIN_ROWS 2048
// the node's tensors are a shape the kernel serves (types, strides, widths); buffers are the caller's
bool ggml_cuda_kqmmv_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// PXA_KQMMV_GROUP (default on): consecutive same-src1 MUL_MATs share one launch
bool ggml_cuda_kqmmv_group_on(void);
// n MUL_MATs that all read src1 (each already take + shape_ok): consecutive same-type runs of up to
// KQMMV_MAX_MATS are one launch each (one per matrix when PXA_KQMMV_GROUP=0). false = declined
// (fewer than KQMMV_MIN_ROWS rows in all), nothing launched: the caller runs its incumbent path.
bool ggml_cuda_kqmmv_mul_mats(int device, cudaStream_t stream, const ggml_tensor * src1,
                              const ggml_tensor * const * src0, ggml_tensor * const * dst, int n);
// one MUL_MAT whose next node ADDs a per-row f32 vector to it (the incumbent mmvq_biased form: the
// residual / bias add): the sum goes straight to add->data (bit-identical to the GEMV followed by the ADD,
// one f32 add per output). false = declined, nothing launched.
bool ggml_cuda_kqmmv_mul_mat_bias(int device, cudaStream_t stream, const ggml_tensor * src1, const ggml_tensor * src0,
                                  ggml_tensor * add, const ggml_tensor * bias);
// FUSED_UP_GATE (SILU, no bias) with same-type k-quant up/gate at ny <= 8: 0 = served, -1 = declined
int  ggml_cuda_kqmmv_up_gate(int device, cudaStream_t stream, ggml_tensor * dst);
