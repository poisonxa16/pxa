// pxq23_kernel.cuh -- PXQ2 / PXQ3 device decode primitives for sm_60 and sm_70.
//
// RELATIONSHIP TO pxq4_kernel.cuh, stated once so nobody has to reverse-engineer it.
// pxq4_kernel.cuh is FROZEN: it carries the shipped v12b PXQ4 kernels, whose bit-exactness
// against the llama engine is the entire correctness argument of the PXQ4 line, and this file
// does not include, edit or re-instantiate any of it. What it does instead is take the SAME
// kernels -- which pxq4_kernel.cuh already wrote against a policy struct -- and re-express
// them as POLICY TEMPLATES, then instantiate them for pxq2_pol and pxq3_pol only.
//
// So: PXQ4 keeps exactly the object code it has today, and the new tiers get the same
// algorithm, the same canonical chunk fold, the same accumulation shape and the same single
// final rounding, with three things changed and nothing else:
//   1. SLAB / CODE_OFF / CODE_BYTES / CODE_WORDS  (the geometry, pxq23_kernel_tables.h)
//   2. pair()          (2-bit fields, or the 3-bit bit-plane reassembly)
//   3. stage_tabs()    (a 4- or 8-entry book zero-filled into the same 16-float tab[])
// Every policy body below is a line-for-line transcription of pxq6_pol_p2 / pxq6_pol_p3 in
// ggml/src/ggml-cuda/pxa/pxq23.cuh. The multiply order in row_effs and the pair-then-scale
// order in the dot product are the engine's parity-locked dequant contract and must not be
// reassociated:  eff = fp32(anchor_fp16) * SUB16[s4] ;  w = eff * fp32(book[c]).
//
// THE DUPLICATION IS DELIBERATE, and it is bought back by a gate rather than by trust: this
// header also compiles for pxq4_pol (PXQ_SELFTEST_POL4), and pxq_selftest() runs the templated
// PXQ4 instantiation against the shipped k_pxq4_dequant_matrix on the same bytes and requires
// max-abs-diff 0. If the transcription ever drifts, that gate fails before a model is loaded.
//
// WHAT IS NOT PORTED, and why. The K-chunk-SPLIT / fused-last-arriver mmv family
// (k_pxq4_mmv_part / _reduce / _fused / _fused_mt) is PXQ4-only here. It exists because the
// monolithic mmv starves an 80-SM V100 at the dense 27B's TP4 decode shapes (64-136 blocks),
// and it pays for that with persistent fp32 partial and arrival-counter arenas that must not
// grow under stream capture -- the single largest source of capture hazards in this package
// (see the v12b use-after-free note in pxq4_kernel_torch.cpp). The shapes this file serves are
// the opposite case: the MoE decode path launches grid = (panels, S) with S = tokens*top_k,
// i.e. 16x64 = 1024 blocks for w13 and 32x64 = 2048 for w2 at M=8/top_k=8 on the 35B. There is
// no starvation to fix, so the monolithic kernels are both faster to trust and free of every
// arena hazard. If a future shape does starve, port k_pxq4_mmv_part/_reduce here -- its
// bit-exactness argument is written out in pxq4_kernel.cuh and applies verbatim.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#include "pxq23_kernel_tables.h"
#include "pxq4_kernel_tables.h"     // PXQ4_SUB16_INIT (== PXQ6_SUB16_INIT), PXQ4_MMV_KSEG,
                                    // PXQ4_CANON_CMAX, PXQ4_CANON_V2, PXQ4_BM, PXQ4_QK

#ifndef PXQ_EXTERN_SHARED
#define PXQ_EXTERN_SHARED extern __shared__ __align__(16)
#endif

// ---------------------------------------------------------------------------------------------
// device-resident tables. One copy per translation unit, as in the engine (pxq23.cuh:57-59).
// The SUB16 table is SHARED with PXQ4/PXQ6 -- the engine states this in both tier headers and
// the values are literally PXQ6_SUB16_INIT -- so a checkpoint that recorded a custom
// pxa.pxq6.sub applies to every tier at once, which is what pxq23_upload_tables honours.
// ---------------------------------------------------------------------------------------------
static __device__ float pxq2_book_g[PXQ2_BOOK_N]  = PXQ2_BOOK_INIT;
static __device__ float pxq3_book_g[PXQ3_BOOK_N]  = PXQ3_BOOK_INIT;
static __device__ float pxq23_sub16_g[16]         = PXQ4_SUB16_INIT;
// PXQ4 book, for the self-test instantiation ONLY (pxq4_selftest_pol). The shipped PXQ4
// kernels use pxq4_book_g in their own TU; this copy exists so the transcription gate can
// run the templated kernels on PXQ4 bytes without touching that TU.
static __device__ float pxq4_selftest_book_g[16]  = PXQ4_BOOK_INIT;

// ---------------------------------------------------------------------------------------------
// scale-byte load; carries the same evict-first cache policy as the code row it belongs to
// (v11 change 1, pxq4_kernel.cuh:117-123). Slab bytes are read once and never reused.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ int pxq23_ldscale(const uint8_t * p) {
#ifdef __CUDA_ARCH__
    return (int)__ldcs(p);
#else
    return (int)*p;              // hostsim
#endif
}

// ---------------------------------------------------------------------------------------------
// per-row code load, POLICY-DISPATCHED ON WIDTH. See pxq23_kernel_tables.h "ALIGNMENT": the
// 12-byte PXQ3 row is only 4-byte aligned on odd rows, so it takes three scalar loads and must
// NEVER take a vector load. (Engine: pxq6_ldcodes, pxq6.cuh:455-479.)
// ---------------------------------------------------------------------------------------------
template <class POL>
static __device__ __forceinline__ void pxq23_ldcodes(const uint8_t * p, uint32_t * q) {
    if constexpr (POL::CODE_WORDS == 4) {            // PXQ4: 16 B row, 16 B aligned
#ifdef __CUDA_ARCH__
        *(uint4 *)q = __ldcs((const uint4 *)p);
#else
        *(uint4 *)q = *(const uint4 *)p;
#endif
    } else if constexpr (POL::CODE_WORDS == 2) {     // PXQ2: 8 B row, 8 B aligned
#ifdef __CUDA_ARCH__
        *(uint2 *)q = __ldcs((const uint2 *)p);
#else
        *(uint2 *)q = *(const uint2 *)p;
#endif
    } else {                                         // PXQ3: 12 B row, 4 B aligned only
        const uint32_t * s = (const uint32_t *)p;
#ifdef __CUDA_ARCH__
        q[0] = __ldcs(s); q[1] = __ldcs(s + 1); q[2] = __ldcs(s + 2);
#else
        q[0] = s[0]; q[1] = s[1]; q[2] = s[2];
#endif
    }
}

// ---------------------------------------------------------------------------------------------
// format policies. Transcribed from pxq23.cuh pxq6_pol_p2 / pxq6_pol_p3, with the
// panel-relative anchor() accessor removed (the anchor arrives as its own tensor -- the same
// addressing edit pxq4_kernel.cuh documents as edit 1).
//
// stage_tabs takes tab/sub by ARRAY REFERENCE, exactly as pxq4_pol does, so widening the
// shared table becomes a compile error rather than a silent bank-conflict regression.
// Entries past the book are zero-filled and never indexed: a 2-bit code cannot exceed 3 and a
// 3-bit code cannot exceed 7. The zero fill is the engine's, kept so the two stage the same
// 64 bytes.
// ---------------------------------------------------------------------------------------------
struct pxq2_pol {
    static constexpr int TIER       = PXQ_TIER_PXQ2;
    static constexpr int SLAB       = PXQ2_SLAB_BYTES;
    static constexpr int CODE_OFF   = PXQ2_CODE_OFF;
    static constexpr int CODE_BYTES = PXQ2_CODE_BYTES;
    static constexpr int CODE_WORDS = PXQ2_CODE_WORDS;
    static constexpr int NEFF       = PXQ2_NEFF;
    static constexpr int BOOK_N     = PXQ2_BOOK_N;

    __device__ static void stage_tabs(float (&tab)[16], float (&sub)[16], int tid) {
        static_assert(sizeof(tab) == 64 && sizeof(sub) == 64,
                      "pxq2: the book/sublevel tables must stay 16 floats = 64 bytes");
        if      (tid < 16) tab[tid]      = tid < PXQ2_BOOK_N ? pxq2_book_g[tid] : 0.f;
        else if (tid < 32) sub[tid - 16] = pxq23_sub16_g[tid - 16];
    }

    // eff = fp32(anchor_fp16) * SUB16[nibble]; ORDER IS LOAD-BEARING (parity-locked contract).
    __device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                    const float * sub, float * eff) {
        const int sb = pxq23_ldscale(slab + row);
        eff[0] = anch * sub[sb & 0xf];    // elements  0-15 of this 32-element block
        eff[1] = anch * sub[sb >> 4];     // elements 16-31
    }

    // pair b covers elements 2b and 2b+1. LE u32 word h = b>>3 holds elements 16h..16h+15 at
    // 2 bits each; element j of that half sits at bit 2*(j&15).
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const uint32_t w  = q[b >> 3];
        const int      sh = 2 * ((2 * b) & 15);
        return make_float2(tab[(w >> sh) & 3], tab[(w >> (sh + 2)) & 3]);
    }
};

struct pxq3_pol {
    static constexpr int TIER       = PXQ_TIER_PXQ3;
    static constexpr int SLAB       = PXQ3_SLAB_BYTES;
    static constexpr int CODE_OFF   = PXQ3_CODE_OFF;
    static constexpr int CODE_BYTES = PXQ3_CODE_BYTES;
    static constexpr int CODE_WORDS = PXQ3_CODE_WORDS;
    static constexpr int NEFF       = PXQ3_NEFF;
    static constexpr int BOOK_N     = PXQ3_BOOK_N;

    __device__ static void stage_tabs(float (&tab)[16], float (&sub)[16], int tid) {
        static_assert(sizeof(tab) == 64 && sizeof(sub) == 64,
                      "pxq3: the book/sublevel tables must stay 16 floats = 64 bytes");
        if      (tid < 16) tab[tid]      = tid < PXQ3_BOOK_N ? pxq3_book_g[tid] : 0.f;
        else if (tid < 32) sub[tid - 16] = pxq23_sub16_g[tid - 16];
    }

    __device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                    const float * sub, float * eff) {
        const int sb = pxq23_ldscale(slab + row);
        eff[0] = anch * sub[sb & 0xf];
        eff[1] = anch * sub[sb >> 4];
    }

    // BIT-PLANE decode, branch-free: lo word by 16-element half, high plane in q[2].
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int      h  = b >> 3;               // 16-element half (0 or 1)
        const int      j0 = (2 * b) & 15;         // first element within the half
        const uint32_t lo = q[h];
        const uint32_t hi = q[2] >> (16 * h);     // this half's high plane in bits 0..15
        const int c0 = (int)((lo >> (2 * j0))     & 3) | (int)(((hi >> j0)       & 1) << 2);
        const int c1 = (int)((lo >> (2 * j0 + 2)) & 3) | (int)(((hi >> (j0 + 1)) & 1) << 2);
        return make_float2(tab[c0], tab[c1]);
    }
};

// The PXQ4 policy, RE-EXPRESSED here for the self-test only (see the header comment). It is
// never used to serve a tensor: pxq4_kernel.cuh's own kernels do that, and the ops in this
// file refuse tier 252.
struct pxq4_selftest_pol {
    static constexpr int TIER       = PXQ_TIER_PXQ4;
    static constexpr int SLAB       = 1088;
    static constexpr int CODE_OFF   = 64;
    static constexpr int CODE_BYTES = 16;
    static constexpr int CODE_WORDS = 4;
    static constexpr int NEFF       = 2;
    static constexpr int BOOK_N     = 16;

    __device__ static void stage_tabs(float (&tab)[16], float (&sub)[16], int tid) {
        if      (tid < 16) tab[tid]      = pxq4_selftest_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq23_sub16_g[tid - 16];
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                    const float * sub, float * eff) {
        const int sb = pxq23_ldscale(slab + row);
        eff[0] = anch * sub[sb & 0xf];
        eff[1] = anch * sub[sb >> 4];
    }
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8 * (b & 3))) & 0xff;
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
};

// ---------------------------------------------------------------------------------------------
// inner accumulation shape and the 32-element dot product. Verbatim from pxq4_kernel.cuh
// (pxq6.cuh:575-608, :634-674 MODE_TAB arm) with pxq4_pol replaced by the template parameter.
// PXQ_CANON_V2 is a BUILD-TIME re-baselining switch, never a runtime flag; it must match the
// engine build that produced the artifact (default 0).
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ float pxq23_acc2(float acc, float a0, float x0,
                                                   float a1, float x1) {
#if PXQ4_CANON_V2
    return __fmaf_rn(a1, x1, __fmaf_rn(a0, x0, acc));
#else
    return acc + (a0 * x0 + a1 * x1);
#endif
}

template <class POL>
struct pxq23_slabreg { uint32_t q[POL::CODE_WORDS]; int sb; };

template <class POL>
static __device__ __forceinline__ void pxq23_load_slab(const uint8_t * __restrict__ slab,
                                                       int row, pxq23_slabreg<POL> & r) {
    r.sb = pxq23_ldscale(slab + row);
    pxq23_ldcodes<POL>(slab + POL::CODE_OFF + row * POL::CODE_BYTES, r.q);
}

template <class POL, bool VECX>
static __device__ __forceinline__ float pxq23_dot32_reg(const pxq23_slabreg<POL> & r, float anch,
                                                        const float * __restrict__ xk,
                                                        const float * __restrict__ tab,
                                                        const float * __restrict__ sub) {
    float eff[POL::NEFF];
    eff[0] = anch * sub[r.sb & 0xf];
    eff[1] = anch * sub[r.sb >> 4];
    const uint32_t * q = r.q;

    // t[0] accumulates the eff[0] half (elements 0-15, pair indices b = 0..7), t[1] the other.
    // `(b*NEFF) >> 4` is the engine's index expression; with NEFF == 2 it is exactly (b >= 8).
    float t[POL::NEFF];
#pragma unroll
    for (int i = 0; i < POL::NEFF; ++i) t[i] = 0.f;

    if (VECX) {
        // float4 activation loads; &xk[0] is 32-float aligned at every call site.
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2 * b];
            const float2 p0 = POL::pair(q, b,     tab);
            const float2 p1 = POL::pair(q, b + 1, tab);
            t[(b * POL::NEFF) >> 4]       = pxq23_acc2(t[(b * POL::NEFF) >> 4],       p0.x, xv.x, p0.y, xv.y);
            t[((b + 1) * POL::NEFF) >> 4] = pxq23_acc2(t[((b + 1) * POL::NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = POL::pair(q, b, tab);
            t[(b * POL::NEFF) >> 4] = pxq23_acc2(t[(b * POL::NEFF) >> 4], p.x, xk[2 * b], p.y, xk[2 * b + 1]);
        }
    }
    return eff[0] * t[0] + eff[1] * t[1];
}

template <class POL, bool VECX>
static __device__ __forceinline__ float pxq23_dot32(const uint8_t * __restrict__ slab, int row,
                                                    float anch,
                                                    const float * __restrict__ xk,
                                                    const float * __restrict__ tab,
                                                    const float * __restrict__ sub) {
    pxq23_slabreg<POL> r;
    pxq23_load_slab<POL>(slab, row, r);
    return pxq23_dot32_reg<POL, VECX>(r, anch, xk, tab, sub);
}

// Canonical chunk count -- a function of SHAPE ONLY, tier-independent (it counts SLABS, and
// every tier has 32 columns per slab). Reproduced rather than reused so this header stands
// alone; the value is identical to pxq4_canon_nfix by construction. (pxq6.cuh:826-833)
static __host__ __device__ __forceinline__ int pxq23_canon_nfix(int kslabs, int cmax) {
    int lim = kslabs / PXQ4_MMV_KSEG;
    if (lim < 1)    lim = 1;
    if (lim > cmax) lim = cmax;
    int n = 1;
    while (n * 2 <= lim) n *= 2;
    return n;
}

static __host__ __forceinline__ int pxq23_canon_max_chunk(int kslabs) {
    const int nfix = pxq23_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    return (kslabs + nfix - 1) / nfix;
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant. One block per slab, 64 threads, one row each.
// out is [N, K] row-major fp16 -- vLLM's weight layout for torch.mm(x, w.t()).
// The smem tile + write-back-along-K store coalescing is the engine's 2026-07-27 change, kept
// verbatim: same values, same addresses, different instruction mapping.
// ---------------------------------------------------------------------------------------------
template <class POL, typename dst_t>
static __global__ void k_pxq23_dequant_matrix(const uint8_t * __restrict__ slabs,
                                              const __half  * __restrict__ anchor,
                                              dst_t * __restrict__ y,
                                              const int kslabs, const int64_t K) {
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ dst_t tile[PXQ4_BM][PXQ4_QK + 2];   // +2 -> 17 four-byte banks, gcd(17,32)==1

    POL::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();

    const int64_t slab_id = blockIdx.x;
    const int64_t p       = slab_id / kslabs;
    const int     kb      = (int)(slab_id % kslabs);
    const int     row     = threadIdx.x;

    const uint8_t * slab = slabs + ((size_t)p * kslabs + kb) * POL::SLAB;
    const float     anch = __half2float(anchor[(size_t)p * PXQ4_BM + row]);

    float eff[POL::NEFF];
    POL::row_effs(slab, row, anch, sub, eff);
    uint32_t q[POL::CODE_WORDS];
    pxq23_ldcodes<POL>(slab + POL::CODE_OFF + row * POL::CODE_BYTES, q);

#pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float  e = eff[(b * POL::NEFF) >> 4];
        const float2 v = POL::pair(q, b, tab);
        tile[row][2 * b]     = (dst_t)(e * v.x);
        tile[row][2 * b + 1] = (dst_t)(e * v.y);
    }
    __syncthreads();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x  >> 5;
    for (int r = warp; r < PXQ4_BM; r += nwarp) {
        y[(p * PXQ4_BM + r) * K + kb * PXQ4_QK + lane] = tile[r][lane];
    }
}

// ---------------------------------------------------------------------------------------------
// decode matrix-vector: out[M, N] = x[M, K] * W[N, K]^T, small M only (the caller caps M at
// PXQ4_MMV_MAX_M and uses dequant + cuBLAS above that).
// grid = (N/64, M), block = PXQ4_MMV_KSEG*64 = 256, dynamic smem = max_chunk*32 floats.
// PXQ_CANON_v1 two-level fixed-chunk fold per lane, then a kseg-ordered fold and ONE rounding.
// ---------------------------------------------------------------------------------------------
template <class POL, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq23_mmv(const uint8_t * __restrict__ slabs,
            const __half  * __restrict__ anchor,
            const __half  * __restrict__ x,          // [M, K]
            __half        * __restrict__ out,        // [M, N]
            const int R, const int K) {
    const int p  = blockIdx.x;                       // panel = 64 output rows
    const int iy = blockIdx.y;                       // token

    PXQ_EXTERN_SHARED float pxq23_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * POL::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int nfix = pxq23_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        // barrier 1 also covers the stage_tabs writes on the first iteration, and protects
        // the previous chunk's readers from this chunk's writers on every later iteration.
        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq23_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq23_dot32<POL, VECX>(pan_slabs + (size_t)kb * POL::SLAB, row, anch,
                                        pxq23_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
        }
        su += t;
    }

    red[kseg * PXQ4_BM + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}


// =============================================================================================
// K-CHUNK-SPLIT DECODE FAMILY for PXQ2/PXQ3 (v16).
//
// WHAT CHANGED AND WHY THE HEADER COMMENT AT THE TOP OF THIS FILE NO LONGER APPLIES.
// That comment says the split/fused-last-arriver family is PXQ4-only here, and gives the
// reason: "The shapes this file serves are the opposite case: the MoE decode path launches
// grid = (panels, S) with S = tokens*top_k, i.e. 16x64 = 1024 blocks ... There is no
// starvation to fix." That was true of the MoE file it was written against. It is NOT true of
// a DENSE model whose linear layers are pxq2/pxq3: there k_pxq23_mmv launches grid = (panels,
// M), which at decode (M = 1) is 40-136 blocks of 256 threads on an 80-SM card -- at most
// 12.5% occupancy, latency-bound, and exactly the case k_pxq4_mmv_part was built for. The
// same comment names the remedy: "If a future shape does starve, port k_pxq4_mmv_part/_reduce
// here -- its bit-exactness argument is written out in pxq4_kernel.cuh and applies verbatim."
// This is that port, extended to the fused (v4) and multi-token (v6) forms, because the
// two-launch pair pays a device drain + refill per call and the per-token form re-reads the
// whole weight tensor once per token.
//
// BIT-EXACTNESS. Reproduced from pxq4_kernel.cuh and true here for the same reasons:
//   * k_pxq23_mmv computes, per (row, kseg) lane, su = ((0 + t_0) + t_1) + ... + t_{nfix-1}
//     with t_c = 0 + dot32(kb = b0+kseg) + dot32(+KSEG) + ..., then folds across ksegs in
//     ASCENDING kseg order and applies ONE __float2half_rn.
//   * k_pxq23_mmv_part computes exactly t_c -- identical staging, identical pxq23_dot32 calls,
//     identical kb order -- and stores it UNSUMMED.
//   * k_pxq23_mmv_reduce (and the winner block of the fused kernels) replays literally those
//     two folds, in that order, in fp32, with the same single final rounding.
// No addition is reassociated and no rounding point moves, so every output is BIT-IDENTICAL to
// the monolithic kernel. The atomic is an ARRIVAL COUNTER, never an accumulator: no floating-
// point value is ever atomically combined.
//
// WHAT WOULD NOT BE BIT-EXACT (rejected; do not "simplify" into these):
//   - one block summing a RANGE of chunks (grid-stride / persistent blocks). A left-associated
//     chain cannot be split: ((t0+t1)+t2) + (t3+t4) != ((((t0+t1)+t2)+t3)+t4).
//   - atomicAdd of floats into an accumulator: nondeterministic order, breaks the contract.
//   - cooperative_groups::grid_group::sync(): needs cudaLaunchCooperativeKernel, which caps the
//     grid at what fits resident, forcing exactly the persistent-block reassociation above.
//
// FORWARD PROGRESS. No block ever spins. Every block increments and exits; only the one that
// observes old == nfix-1 continues. Deadlock is impossible regardless of block scheduling.
//
// CONCURRENCY. Exactly one pxq2/pxq3 split mmv may be in flight per device at a time -- the
// partials and counter arenas are shared across every module, so this is a scratch-aliasing
// constraint AND, for the fused forms, a correctness dependency of the arrival barrier.
//
// FAILURE MODE. A launch torn down mid-flight leaves ctr non-zero; no later block then observes
// old == nfix-1, out[] is never written, and the caller silently consumes stale fp16. The
// failure is SILENT. Mitigation: the arena zeroes the counter region once at allocation, and
// any error path that abandons a launch must reset it (pxq23_torch.cpp).
// =============================================================================================

// ---- arrival-barrier primitives -------------------------------------------------------------
// Transcribed from pxq4_kernel.cuh. They cannot be reused from that header: this TU does not
// include it (and must not -- the frozen PXQ4 kernels keep their own translation unit), and the
// helpers there are `static __device__`, i.e. TU-local by construction.
#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
// sm_60 (P100): the PTX memory consistency model (.release/.acquire scopes, fence.acq_rel) is
// sm_70+; ptxas rejects atom.release.gpu below that. Fall back to the classic pre-Volta
// threadFenceReduction pattern. The atomic is an ARRIVAL COUNTER, never an FP accumulator, so
// no floating-point expression tree is touched and the outputs stay bit-identical to the sm_70
// build. This arm is reached: the sm60 library builds this TU for compute_60 AND compute_70.
static __device__ __forceinline__ unsigned pxq23_arrive_release(unsigned * p) {
    __threadfence();                 // release: this thread's prior part[] stores are visible
    return atomicAdd(p, 1u);
}
static __device__ __forceinline__ void pxq23_fence_acq_rel() {
    __threadfence();                 // acquire: other blocks' part[] stores become visible
}
#else
// atom.release.gpu: the release fence is fused into the RMW. A separate __threadfence() before
// a plain atomicAdd is also bit-exact but measurably slower (every block pays a full membar.gl).
static __device__ __forceinline__ unsigned pxq23_arrive_release(unsigned * p) {
    unsigned old;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
    return old;
}
static __device__ __forceinline__ void pxq23_fence_acq_rel() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
#endif
// __ldcg (ld.global.cg, L2-only) exists on every arch built here; on sm_60 L1 is not coherent
// across SMs either, so the same hint is wanted there.
static __device__ __forceinline__ float pxq23_ld_part(const float * p) { return __ldcg(p); }
#else
// hostsim: blocks run strictly sequentially, so the RMW degenerates to ++ and the fences are
// no-ops. This gates the VALUES the fused path computes; it can NEVER observe the race, which
// is why the device differential in pxq23_selftest() is mandatory rather than optional.
static inline unsigned pxq23_arrive_release(unsigned * p) { const unsigned old = *p; *p = old + 1u; return old; }
static inline void  pxq23_fence_acq_rel() {}
static inline float pxq23_ld_part(const float * p) { return *p; }
#endif

// ---------------------------------------------------------------------------------------------
// v3: two-launch K-chunk split. Kept because it is the only form the device differential can
// compare the fused kernel against without the barrier in the picture, and as a one-line
// fallback if a future shape ever misbehaves.
//
// GRID ORDER (chunk-major): c is the FASTEST-varying grid dimension, so a panel's nfix chunk
// blocks are adjacent in launch order and the concurrently resident set reads a few panels'
// full K range rather than one k-chunk of every panel. Pure addressing: part[] keeps the
// (iy, p, c, tid) layout and every lane visits the same kb in the same order.
//
// grid = (nfix, panels, M), block = 256; nfix MUST equal pxq23_canon_nfix(kslabs, CMAX).
// nfix and panels arrive as EXPLICIT ARGUMENTS rather than off gridDim, so one value feeds
// both kernels and the launcher can assert it.
// dynamic smem = pxq23_canon_max_chunk(K/32) * 32 floats, the same bound as k_pxq23_mmv.
// ---------------------------------------------------------------------------------------------
#define PXQ23_MMV_PART_CHUNK_MAJOR 1

template <class POL, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq23_mmv_part(const uint8_t * __restrict__ slabs,
                 const __half  * __restrict__ anchor,
                 const __half  * __restrict__ x,        // [M, K]
                 float         * __restrict__ part,     // [M, panels, nfix, KSEG*64]
                 const int K, const int nfix, const int panels) {
    const int c  = blockIdx.x;                          // canonical chunk (fastest-varying)
    const int p  = blockIdx.y;                          // panel
    const int iy = blockIdx.z;                          // token

    PXQ_EXTERN_SHARED float pxq23_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * POL::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    // this block's canonical chunk -- the same b0/b1 the monolithic loop computes for c
    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    __syncthreads();                                    // covers the stage_tabs writes
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        pxq23_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
    }
    __syncthreads();

    float t = 0.f;
    for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
        t += pxq23_dot32<POL, VECX>(pan_slabs + (size_t)kb * POL::SLAB, row, anch,
                                    pxq23_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
    }
    // one fully-coalesced 1024-B store per block: threadIdx.x == kseg*64 + row, matching the
    // red[] layout of k_pxq23_mmv so the reduce below replays its fold verbatim.
    part[(((size_t)iy * panels + p) * nfix + c) * (PXQ4_MMV_KSEG * PXQ4_BM)
         + threadIdx.x] = t;
}

// grid = (panels, M), block = 64 (one weight row each). Reads back the [nfix, KSEG*64] tile of
// one (panel, token) and performs k_pxq23_mmv's two folds in its exact order. All loads are
// coalesced: 64 consecutive floats per (c, s). Tier-independent -- it touches only fp32
// partials -- so it is not a template.
static __global__ void __launch_bounds__(PXQ4_BM)
k_pxq23_mmv_reduce(const float * __restrict__ part,    // [M, panels, nfix, KSEG*64]
                   __half      * __restrict__ out,     // [M, R]
                   const int nfix, const int R) {
    const int p   = blockIdx.x;
    const int iy  = blockIdx.y;
    const int row = threadIdx.x;

    const float * base = part + (((size_t)iy * gridDim.x + p) * nfix)
                              * (PXQ4_MMV_KSEG * PXQ4_BM);
    float su[PXQ4_MMV_KSEG];
#pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) su[s] = 0.f;
    for (int c = 0; c < nfix; ++c) {                   // chunk fold: su_s = ((0+t_0)+t_1)+...
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            su[s] += base[(size_t)c * (PXQ4_MMV_KSEG * PXQ4_BM) + s * PXQ4_BM + row];
        }
    }
    float u = 0.f;                                     // kseg fold, in kseg order
#pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += su[s];
    out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
}

// ---------------------------------------------------------------------------------------------
// v4: single-launch fused split mmv -- k_pxq23_mmv_part and k_pxq23_mmv_reduce in one kernel.
// The reduce runs in whichever block of a (panel, token) arrives LAST, so the device drain +
// refill between the two kernels disappears and the decode graph loses one launch per module.
//
// ctr: M*panels unsigned, ZERO on entry. The winner rearms its slot to 0 before returning, so
// a completed launch leaves the buffer ready for the next one -- no memset launch is needed and
// nothing is allocated in-capture.
// ---------------------------------------------------------------------------------------------
template <class POL, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq23_mmv_fused(const uint8_t * __restrict__ slabs,
                  const __half  * __restrict__ anchor,
                  const __half  * __restrict__ x,        // [M, K]
                  float         * __restrict__ part,     // [M, panels, nfix, KSEG*64]
                  unsigned      * __restrict__ ctr,      // [M, panels], zero on entry and exit
                  __half        * __restrict__ out,      // [M, R]
                  const int R, const int K, const int nfix, const int panels) {
    const int c  = blockIdx.x;                           // canonical chunk (fastest-varying)
    const int p  = blockIdx.y;                           // panel
    const int iy = blockIdx.z;                           // token

    PXQ_EXTERN_SHARED float pxq23_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * POL::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        pxq23_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
    }
    // This kernel owns exactly ONE chunk, so pxq23_xs has no previous readers to protect and
    // this single barrier already orders the stage_tabs writes above against every tab/sub read
    // below. (k_pxq23_mmv's first barrier is NOT redundant: its chunk loop rewrites pxq23_xs
    // underneath the previous chunk's readers.)
    __syncthreads();

    // Register double-buffering: slab kb+KSEG's global loads are ISSUED before slab kb is
    // FOLDED, so a warp always has a second weight load in flight. The accumulation is
    // untouched -- t still takes dot32(kb) for ascending kb, left-associated, with the
    // identical operands -- so this is scheduling only and stays bit-identical.
    float t = 0.f;
    {
        int kb = b0 + kseg;
        if (kb < b1) {
            pxq23_slabreg<POL> r0, r1;
            pxq23_load_slab<POL>(pan_slabs + (size_t)kb * POL::SLAB, row, r0);
            for (;;) {
                const int  kn   = kb + PXQ4_MMV_KSEG;
                const bool more = (kn < b1);
                if (more) pxq23_load_slab<POL>(pan_slabs + (size_t)kn * POL::SLAB, row, r1);
                t += pxq23_dot32_reg<POL, VECX>(r0, anch,
                                                pxq23_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
                if (!more) break;
                r0 = r1;
                kb = kn;
            }
        }
    }

    const size_t  tile  = (size_t)(PXQ4_MMV_KSEG * PXQ4_BM);
    float * const pbase = part + (((size_t)iy * panels + p) * nfix) * tile;
    pbase[(size_t)c * tile + threadIdx.x] = t;

    // ---- arrival barrier over this (panel, token)'s nfix blocks --------------------------
    // The __syncthreads() below is LOAD-BEARING and is the easiest line in this kernel to omit.
    // The release orders only thread 0's OWN prior accesses, so without it lanes 1..255 may not
    // have issued their part[] stores when the counter is bumped, and the winner reads stale
    // words. Omitting it still passes a single-shot parity check, by luck. Do not remove it.
    __syncthreads();
    int won = 0;
    if (threadIdx.x == 0) {
        const unsigned old = pxq23_arrive_release(&ctr[(size_t)iy * panels + p]);
        won = (old == (unsigned)(nfix - 1));
        if (won) {
            ctr[(size_t)iy * panels + p] = 0u;            // rearm for the next launch
            pxq23_fence_acq_rel();                        // acquire the other blocks' part[]
        }
    }
    const int last = __syncthreads_or(won);               // barrier + broadcast in one
    if (!last) return;

    // ---- k_pxq23_mmv_reduce's fold, verbatim ---------------------------------------------
    // __ldcg keeps the read off L1 (not coherent across SMs); the loaded value is the same fp32
    // word either way, so this is a cache-hint change only. This block's OWN chunk is still
    // live in t -- exactly the word it just stored at pbase[c*tile+tid] -- so reading the
    // register removes 1/nfix of the reduce's global traffic and yields the identical value.
    float su = 0.f;
    for (int cc = 0; cc < nfix; ++cc) {                   // chunk fold, ascending c
        su += (cc == c) ? t : pxq23_ld_part(&pbase[(size_t)cc * tile + threadIdx.x]);
    }
    red[threadIdx.x] = su;                                // threadIdx.x == kseg*64 + row
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;                                    // kseg fold, ascending s
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}

// ---------------------------------------------------------------------------------------------
// v6: MULTI-TOKEN (MT) fused split mmv -- the concurrency kernel.
//
// WHY. Every mmv variant above re-reads the whole weight tensor once PER TOKEN (grid.z = M),
// so a decode batch of M costs ~M times the weight traffic of M = 1. This kernel gives one
// block ALL M tokens of its (chunk, panel): each weight byte is decoded once and folded into M
// accumulators, so the weight traffic of a decode step is ~constant in M (activations and
// partials still scale with M, but they are KB, not MB).
//
// BIT-EXACTNESS. Per token the fold is UNCHANGED: pxq23_dot32_mt performs, for each token m,
// exactly the b-loop and pxq23_acc2 calls of pxq23_dot32 in the same order on t[m]; the caller
// accumulates per-kb results in the same kb order; the partials keep the identical
// [M, panels, nfix, 256] layout; and the winner replays k_pxq23_mmv_reduce's fold verbatim per
// token. Interleaving tokens in the inner loop reorders nothing WITHIN any token's expression
// tree, so each token's output is bit-identical to the monolithic kernel's.
//
// GRID = (nfix, panels, 1), block = 256, template MT == M (1..16).
// dynamic smem = pxq23_canon_max_chunk(kslabs) * 32 * MT floats: token m's chunk slice starts
// at pxq23_xs + m*n (n = this chunk's float count, a multiple of 32, so every token slice keeps
// the 16-B alignment the float4 loads need).
// ctr = panels unsigned (token axis collapsed), same zero-on-entry/exit contract as v4.
// ---------------------------------------------------------------------------------------------
template <class POL, bool VECX, int MT>
static __device__ __forceinline__ void pxq23_dot32_mt(const uint8_t * __restrict__ slab, int row,
                                                      float anch,
                                                      const float * __restrict__ xk,
                                                      const int xs_stride,
                                                      const float * __restrict__ tab,
                                                      const float * __restrict__ sub,
                                                      float (&res)[MT]) {
    float eff[POL::NEFF];
    POL::row_effs(slab, row, anch, sub, eff);
    uint32_t q[POL::CODE_WORDS];
    pxq23_ldcodes<POL>(slab + POL::CODE_OFF + row * POL::CODE_BYTES, q);

    float t[MT][POL::NEFF];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int i = 0; i < POL::NEFF; ++i) t[m][i] = 0.f;
    }

    if (VECX) {
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float2 p0 = POL::pair(q, b,     tab);
            const float2 p1 = POL::pair(q, b + 1, tab);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const float4 xv = *(const float4 *)&xk[m * xs_stride + 2 * b];
                t[m][(b * POL::NEFF) >> 4]       = pxq23_acc2(t[m][(b * POL::NEFF) >> 4],       p0.x, xv.x, p0.y, xv.y);
                t[m][((b + 1) * POL::NEFF) >> 4] = pxq23_acc2(t[m][((b + 1) * POL::NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
            }
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = POL::pair(q, b, tab);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                t[m][(b * POL::NEFF) >> 4] = pxq23_acc2(t[m][(b * POL::NEFF) >> 4],
                                                        p.x, xk[m * xs_stride + 2 * b],
                                                        p.y, xk[m * xs_stride + 2 * b + 1]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) res[m] = eff[0] * t[m][0] + eff[1] * t[m][1];
}

template <class POL, bool VECX, int MT>
static __global__ void __launch_bounds__(256)
k_pxq23_mmv_fused_mt(const uint8_t * __restrict__ slabs,
                     const __half  * __restrict__ anchor,
                     const __half  * __restrict__ x,        // [MT, K]
                     float         * __restrict__ part,     // [MT, panels, nfix, KSEG*64]
                     unsigned      * __restrict__ ctr,      // [panels], zero on entry and exit
                     __half        * __restrict__ out,      // [MT, R]
                     const int R, const int K, const int nfix, const int panels) {
    const int c = blockIdx.x;                               // canonical chunk (fastest-varying)
    const int p = blockIdx.y;                               // panel

    PXQ_EXTERN_SHARED float pxq23_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];
    __shared__ int   last;

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * POL::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);

    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    __syncthreads();                                        // covers the stage_tabs writes
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const __half * xt = x + (size_t)m * K;
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq23_xs[m * n + idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
    }
    __syncthreads();

    float tacc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) tacc[m] = 0.f;
    for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
        float res[MT];
        pxq23_dot32_mt<POL, VECX, MT>(pan_slabs + (size_t)kb * POL::SLAB, row, anch,
                                      pxq23_xs + (size_t)(kb - b0) * PXQ4_QK, n, tab, sub, res);
#pragma unroll
        for (int m = 0; m < MT; ++m) tacc[m] += res[m];
    }

    const size_t tile = (size_t)(PXQ4_MMV_KSEG * PXQ4_BM);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        part[(((size_t)m * panels + p) * nfix + c) * tile + threadIdx.x] = tacc[m];
    }

    // ---- arrival barrier over this panel's nfix blocks (see k_pxq23_mmv_fused) -----------
    __syncthreads();                                        // all lanes' part[] stores issued
    if (threadIdx.x == 0) {
        const unsigned old = pxq23_arrive_release(&ctr[p]);
        last = (old == (unsigned)(nfix - 1));
        if (last) {
            ctr[p] = 0u;                                    // rearm for the next launch
            pxq23_fence_acq_rel();                          // acquire the other blocks' part[]
        }
    }
    __syncthreads();                                        // propagates the acquire
    if (!last) return;

    // ---- k_pxq23_mmv_reduce's fold, verbatim, once per token ------------------------------
    for (int m = 0; m < MT; ++m) {
        const float * pbase = part + (((size_t)m * panels + p) * nfix) * tile;
        float su = 0.f;
        for (int cc = 0; cc < nfix; ++cc) {                 // chunk fold, ascending c
            su += pxq23_ld_part(&pbase[(size_t)cc * tile + threadIdx.x]);
        }
        red[threadIdx.x] = su;
        __syncthreads();
        if (kseg == 0) {
            float u = 0.f;                                  // kseg fold, ascending s
#pragma unroll
            for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
            out[(size_t)m * R + p * PXQ4_BM + row] = __float2half_rn(u);
        }
        __syncthreads();                                    // red[] is reused by the next token
    }
}


// ---------------------------------------------------------------------------------------------
// expert-indexed mmv for the FusedMoE decode path: out[s] = x[s] @ W[ids[s]]^T, ids resident
// on DEVICE. No host sync, so it is legal under CUDA-graph capture -- which is the whole point
// (an eager-only MoE decode measured ~3.4x slower on this stack).
//
// Weights carry the expert as the slowest axis, matching the stacked FusedMoE parameters:
//     pan_slabs = slabs  + ((size_t)e * panels + p) * kslabs * SLAB
//     anch      = anchor[((size_t)e * panels + p) * 64 + row]
// An id outside [0, E) (vLLM emits -1 padding slots) contributes ZERO: the weight loop is
// skipped and the zero accumulator flows through the unchanged fold, so out[s] is always
// WRITTEN and never left stale. Per row the fold is k_pxq23_mmv's fold verbatim on that
// expert's 2-D slice, so a row is bit-identical to a direct mmv on it.
// grid = (panels, S), block = 256, dynamic smem as k_pxq23_mmv.
// ---------------------------------------------------------------------------------------------
template <class POL, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq23_moe_mmv(const uint8_t * __restrict__ slabs,      // [E, panels, kslabs, SLAB]
                const __half  * __restrict__ anchor,     // [E, panels, 64]
                const __half  * __restrict__ x,          // [S, K]
                const int32_t * __restrict__ ids,        // [S]
                __half        * __restrict__ out,        // [S, R]
                const int R, const int K, const int E, const int panels) {
    const int p  = blockIdx.x;                           // panel
    const int iy = blockIdx.y;                           // (token, slot) row

    PXQ_EXTERN_SHARED float pxq23_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const int  e     = ids[iy];
    const bool valid = (e >= 0) && (e < E);
    const int  esafe = valid ? e : 0;

    const uint8_t * pan_slabs = slabs + ((size_t)esafe * panels + p) * (size_t)kslabs * POL::SLAB;
    const float     anch      = __half2float(anchor[((size_t)esafe * panels + p) * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int nfix = pxq23_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq23_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        if (valid) {
            float t = 0.f;
            for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                t += pxq23_dot32<POL, VECX>(pan_slabs + (size_t)kb * POL::SLAB, row, anch,
                                            pxq23_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
            }
            su += t;
        }
    }

    red[kseg * PXQ4_BM + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}
