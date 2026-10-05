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
