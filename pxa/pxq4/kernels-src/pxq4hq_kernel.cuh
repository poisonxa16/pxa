// pxq4hq_kernel.cuh -- PXQ4HQ (ggml type id 253) device decode primitives for sm_60 and sm_70.
//
// RELATIONSHIP TO THE OTHER TWO KERNEL HEADERS, stated once so nobody has to reverse-engineer
// it. pxq4_kernel.cuh is FROZEN: it carries the shipped v12b PXQ4 kernels whose bit-exactness
// against the llama engine is the correctness argument of the whole PXQ4 line. pxq23_kernel.cuh
// re-expressed those kernels as POLICY TEMPLATES and instantiated them for the 2- and 3-bit
// tiers. Neither file is included, edited or re-instantiated here.
//
// This header is the third instance of the same deliberate transcription, and it exists as its
// own translation unit for one structural reason: pxq23_kernel.cuh's inner product hardcodes
// TWO effective scales per 32-element block (one scale byte, two nibbles). PXQ4HQ has FOUR
// (two scale bytes, four nibbles), which is not a policy constant in that header but a shape
// baked into pxq23_slabreg and pxq23_dot32_reg. Generalising those would edit a file that is
// the sole decode path of two shipping tiers; transcribing them here does not.
//
// WHAT CHANGES relative to the PXQ4 policy, exhaustively -- everything else in this file is a
// line-for-line transcription and the self-test is what proves it:
//   1. SLAB 1088 -> 1152, CODE_OFF 64 -> 128          (pxq4hq_kernel_tables.h)
//   2. SCALE_BYTES 1 -> 2, NEFF 2 -> 4                 (row_effs / slabreg / dot32_reg)
//   3. the sub LUT is SUB8, not the shared SUB16       (pxq4hq_sub8_g, its own device symbol)
// The book, the code packing, the canonical chunk fold, the accumulation shape and the single
// final rounding are identical, and the multiply ORDER in row_effs and the pair-then-scale
// order in the dot product are the engine's parity-locked dequant contract:
//     eff = fp32(anchor_fp16) * SUB8[s4] ;  w = eff * fp32(book[c])
// They must not be reassociated.
//
// THE DUPLICATION IS BOUGHT BACK BY A GATE, not by trust: pxq4hq_selftest() builds a
// deterministic synthetic panel set, decodes it on the host straight from the format spec in
// pxq4hq_kernel_tables.h, runs these kernels on the same bytes and requires max-abs-diff 0.
// If a transcription ever drifts, that fails before a model is loaded.
//
// WHAT IS NOT PORTED, and why. The K-chunk-SPLIT / fused-last-arriver mmv family
// (k_pxq4_mmv_part / _reduce / _fused / _fused_mt) and the whole MoE op family are PXQ4-only.
// PXQ4HQ exists on this engine to serve the ATTENTION block of a promoted tier profile -- the
// tier a policy buys attention up to, never the tier that carries the FFN or the experts -- so
// its shapes are dense LINEAR shapes and the arena hazards those kernels bring with them buy
// nothing here. An expert tensor at this tier is refused by the converter with a message
// naming the missing op, which is a decision recorded at conversion time rather than a silent
// dequant at serving time.
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#include "pxq4hq_kernel_tables.h"

#ifndef PXQ_EXTERN_SHARED
#define PXQ_EXTERN_SHARED extern __shared__ __align__(16)
#endif

// ---------------------------------------------------------------------------------------------
// device-resident tables. One copy per translation unit, as in the engine (pxq6.cuh:79-81).
// The book is the shared PX16 table; the sub LUT is this tier's OWN SUB8 and is deliberately
// not the pxq23/pxq4 SUB16 symbol -- see the header comment in pxq4hq_kernel_tables.h.
// ---------------------------------------------------------------------------------------------
static __device__ float pxq4hq_book_g[PXQ4HQ_BOOK_N] = PXQ4HQ_BOOK_INIT;
static __device__ float pxq4hq_sub8_g[16]            = PXQ4HQ_SUB8_INIT;

// ---------------------------------------------------------------------------------------------
// scale-byte load; carries the same evict-first cache policy as the code row it belongs to
// (v11 change 1, pxq4_kernel.cuh:117-123). Slab bytes are read once and never reused.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ int pxq4hq_ldscale(const uint8_t * p) {
#ifdef __CUDA_ARCH__
    return (int)__ldcs(p);
#else
    return (int)*p;              // hostsim
#endif
}

// 16 B code row at 128 + 16*r inside a 1152 B slab: always 16 B aligned, so one uint4, exactly
// as PXQ4's rows are. (Engine: pxq6_ldcodes, pxq6.cuh:455-479.)
static __device__ __forceinline__ void pxq4hq_ldcodes(const uint8_t * p, uint32_t * q) {
#ifdef __CUDA_ARCH__
    *(uint4 *)q = __ldcs((const uint4 *)p);
#else
    *(uint4 *)q = *(const uint4 *)p;
#endif
}

// ---------------------------------------------------------------------------------------------
// format policy. Same shape as pxq4_pol / pxq2_pol / pxq3_pol, so the kernel bodies below read
// identically to their PXQ4 originals.
//
// stage_tabs takes tab/sub by ARRAY REFERENCE, exactly as pxq4_pol does, so widening the
// shared table becomes a compile error rather than a silent bank-conflict regression (staging
// book PAIRS was built, measured bit-exact, and was 0.73-0.87x the speed -- see
// docs/10-kernel-speed.md, "The MODE_TAB table-delivery angle is closed").
// ---------------------------------------------------------------------------------------------
struct pxq4hq_pol {
    static constexpr int TIER        = PXQ_TIER_PXQ4HQ;
    static constexpr int SLAB        = PXQ4HQ_SLAB_BYTES;
    static constexpr int CODE_OFF    = PXQ4HQ_CODE_OFF;
    static constexpr int CODE_BYTES  = PXQ4HQ_CODE_BYTES;
    static constexpr int CODE_WORDS  = PXQ4HQ_CODE_WORDS;
    static constexpr int SCALE_BYTES = PXQ4HQ_SCALE_BYTES;
    static constexpr int NEFF        = PXQ4HQ_NEFF;
    static constexpr int BOOK_N      = PXQ4HQ_BOOK_N;

    __device__ static void stage_tabs(float (&tab)[16], float (&sub)[16], int tid) {
        static_assert(sizeof(tab) == 64 && sizeof(sub) == 64,
                      "pxq4hq: the book/sublevel tables must stay 16 floats = 64 bytes; a wider "
                      "table reintroduces shared-memory bank conflicts (measured 0.797x)");
        if      (tid < 16) tab[tid]      = pxq4hq_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq4hq_sub8_g[tid - 16];
    }

    // Load this row's TWO scale bytes for one 32-column block. Byte 0 covers elements 0..15,
    // byte 1 covers 16..31; within each, the low nibble is the first 8 elements.
    __device__ static void load_scales(const uint8_t * slab, int row, int * sb) {
        sb[0] = pxq4hq_ldscale(slab + 2 * row);
        sb[1] = pxq4hq_ldscale(slab + 2 * row + 1);
    }

    // The four effective scales of one (row, 32-column block), from scale bytes already loaded.
    // eff[i] covers elements 8i .. 8i+7. ORDER IS LOAD-BEARING (parity-locked contract).
    __device__ static void effs_from(const int * sb, float anch, const float * sub, float * eff) {
        eff[0] = anch * sub[sb[0] & 0xf];    // elements  0.. 7
        eff[1] = anch * sub[sb[0] >>  4];    // elements  8..15
        eff[2] = anch * sub[sb[1] & 0xf];    // elements 16..23
        eff[3] = anch * sub[sb[1] >>  4];    // elements 24..31
    }

    __device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                    const float * sub, float * eff) {
        int sb[SCALE_BYTES];
        load_scales(slab, row, sb);
        effs_from(sb, anch, sub, eff);
    }

    // one code byte -> (book[low nibble], book[high nibble]). b indexes the LE byte of the
    // 16-byte code row held in q[0..3]; element 2b takes the low nibble, 2b+1 the high.
    // Byte-identical packing to PXQ4 (ggml/src/pxq-cpu.c:23).
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8 * (b & 3))) & 0xff;
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
};

// ---------------------------------------------------------------------------------------------
// inner accumulation shape and the 32-element dot product. Transcribed from pxq4_kernel.cuh
// (pxq6.cuh:575-608, :634-674 MODE_TAB arm) with the two-scale fold widened to four.
// PXQ4_CANON_V2 is a BUILD-TIME re-baselining switch, never a runtime flag; it must match the
// engine build that produced the artifact (default 0).
//
// WHY (b * NEFF) >> 4 IS THE RIGHT INDEX AND WAS NOT INVENTED HERE. It is the engine's own
// expression, already used verbatim by the PXQ2/PXQ3/PXQ4 bodies with NEFF == 2, where it
// evaluates to (b >= 8) -- the 16-element half. Pair b covers elements 2b and 2b+1, so with
// NEFF == 4 it evaluates to b >> 2, i.e. elements 0..7 -> eff[0], 8..15 -> eff[1],
// 16..23 -> eff[2], 24..31 -> eff[3]: exactly the bs8 blocks. The expression is unchanged; only
// the constant it is evaluated with differs.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ float pxq4hq_acc2(float acc, float a0, float x0,
                                                    float a1, float x1) {
#if PXQ4_CANON_V2
    return __fmaf_rn(a1, x1, __fmaf_rn(a0, x0, acc));
#else
    return acc + (a0 * x0 + a1 * x1);
#endif
}

struct pxq4hq_slabreg { uint32_t q[PXQ4HQ_CODE_WORDS]; int sb[PXQ4HQ_SCALE_BYTES]; };

static __device__ __forceinline__ void pxq4hq_load_slab(const uint8_t * __restrict__ slab,
                                                        int row, pxq4hq_slabreg & r) {
    pxq4hq_pol::load_scales(slab, row, r.sb);
    pxq4hq_ldcodes(slab + pxq4hq_pol::CODE_OFF + row * pxq4hq_pol::CODE_BYTES, r.q);
}

template <bool VECX>
static __device__ __forceinline__ float pxq4hq_dot32_reg(const pxq4hq_slabreg & r, float anch,
                                                         const float * __restrict__ xk,
                                                         const float * __restrict__ tab,
                                                         const float * __restrict__ sub) {
    constexpr int NEFF = pxq4hq_pol::NEFF;
    float eff[NEFF];
    pxq4hq_pol::effs_from(r.sb, anch, sub, eff);
    const uint32_t * q = r.q;

    // t[i] accumulates the eff[i] block, so each effective scale multiplies its own partial
    // exactly once at the end -- the same "pair, then scale" order the PXQ4 kernel uses.
    float t[NEFF];
#pragma unroll
    for (int i = 0; i < NEFF; ++i) t[i] = 0.f;

    if (VECX) {
        // float4 activation loads; &xk[0] is 32-float aligned at every call site.
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2 * b];
            const float2 p0 = pxq4hq_pol::pair(q, b,     tab);
            const float2 p1 = pxq4hq_pol::pair(q, b + 1, tab);
            t[(b * NEFF) >> 4]       = pxq4hq_acc2(t[(b * NEFF) >> 4],       p0.x, xv.x, p0.y, xv.y);
            t[((b + 1) * NEFF) >> 4] = pxq4hq_acc2(t[((b + 1) * NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = pxq4hq_pol::pair(q, b, tab);
            t[(b * NEFF) >> 4] = pxq4hq_acc2(t[(b * NEFF) >> 4], p.x, xk[2 * b], p.y, xk[2 * b + 1]);
        }
    }
    // Summed in ascending block order, which is the order the two-scale form folds in too
    // (eff[0]*t[0] + eff[1]*t[1]); widening it keeps the same left-to-right association.
    float u = eff[0] * t[0];
#pragma unroll
    for (int i = 1; i < NEFF; ++i) u += eff[i] * t[i];
    return u;
}

template <bool VECX>
static __device__ __forceinline__ float pxq4hq_dot32(const uint8_t * __restrict__ slab, int row,
                                                     float anch,
                                                     const float * __restrict__ xk,
                                                     const float * __restrict__ tab,
                                                     const float * __restrict__ sub) {
    pxq4hq_slabreg r;
    pxq4hq_load_slab(slab, row, r);
    return pxq4hq_dot32_reg<VECX>(r, anch, xk, tab, sub);
}

// Canonical chunk count -- a function of SHAPE ONLY and tier-independent (it counts SLABS, and
// every tier has 32 columns per slab). Reproduced rather than reused so this header stands
// alone; the value is identical to pxq4_canon_nfix by construction. (pxq6.cuh:826-833)
static __host__ __device__ __forceinline__ int pxq4hq_canon_nfix(int kslabs, int cmax) {
    int lim = kslabs / PXQ4_MMV_KSEG;
    if (lim < 1)    lim = 1;
    if (lim > cmax) lim = cmax;
    int n = 1;
    while (n * 2 <= lim) n *= 2;
    return n;
}

static __host__ __forceinline__ int pxq4hq_canon_max_chunk(int kslabs) {
    const int nfix = pxq4hq_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    return (kslabs + nfix - 1) / nfix;
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant. One block per slab, 64 threads, one row each.
// out is [N, K] row-major fp16 -- vLLM's weight layout for torch.mm(x, w.t()).
// The smem tile + write-back-along-K store coalescing is the engine's 2026-07-27 change, kept
// verbatim: same values, same addresses, different instruction mapping.
// ---------------------------------------------------------------------------------------------
template <typename dst_t>
static __global__ void k_pxq4hq_dequant_matrix(const uint8_t * __restrict__ slabs,
                                               const __half  * __restrict__ anchor,
                                               dst_t * __restrict__ y,
                                               const int kslabs, const int64_t K) {
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ dst_t tile[PXQ4HQ_BM][PXQ4HQ_QK + 2];   // +2 -> 17 four-byte banks, gcd(17,32)==1

    pxq4hq_pol::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();

    const int64_t slab_id = blockIdx.x;
    const int64_t p       = slab_id / kslabs;
    const int     kb      = (int)(slab_id % kslabs);
    const int     row     = threadIdx.x;

    const uint8_t * slab = slabs + ((size_t)p * kslabs + kb) * pxq4hq_pol::SLAB;
    const float     anch = __half2float(anchor[(size_t)p * PXQ4HQ_BM + row]);

    float eff[pxq4hq_pol::NEFF];
    pxq4hq_pol::row_effs(slab, row, anch, sub, eff);
    uint32_t q[pxq4hq_pol::CODE_WORDS];
    pxq4hq_ldcodes(slab + pxq4hq_pol::CODE_OFF + row * pxq4hq_pol::CODE_BYTES, q);

#pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float  e = eff[(b * pxq4hq_pol::NEFF) >> 4];
        const float2 v = pxq4hq_pol::pair(q, b, tab);
        tile[row][2 * b]     = (dst_t)(e * v.x);
        tile[row][2 * b + 1] = (dst_t)(e * v.y);
    }
    __syncthreads();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x  >> 5;
    for (int r = warp; r < PXQ4HQ_BM; r += nwarp) {
        y[(p * PXQ4HQ_BM + r) * K + kb * PXQ4HQ_QK + lane] = tile[r][lane];
    }
}

// ---------------------------------------------------------------------------------------------
// decode matrix-vector: out[M, N] = x[M, K] * W[N, K]^T, small M only (the caller caps M and
// uses dequant + cuBLAS above that).
// grid = (N/64, M), block = PXQ4_MMV_KSEG*64 = 256, dynamic smem = max_chunk*32 floats.
// PXQ_CANON_v1 two-level fixed-chunk fold per lane, then a kseg-ordered fold and ONE rounding.
// ---------------------------------------------------------------------------------------------
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4hq_mmv(const uint8_t * __restrict__ slabs,
             const __half  * __restrict__ anchor,
             const __half  * __restrict__ x,          // [M, K]
             __half        * __restrict__ out,        // [M, N]
             const int R, const int K) {
    const int p  = blockIdx.x;                        // panel = 64 output rows
    const int iy = blockIdx.y;                        // token

    PXQ_EXTERN_SHARED float pxq4hq_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4HQ_BM];

    pxq4hq_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4HQ_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4hq_pol::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4HQ_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int nfix = pxq4hq_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4HQ_QK;

        // barrier 1 also covers the stage_tabs writes on the first iteration, and protects
        // the previous chunk's readers from this chunk's writers on every later iteration.
        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq4hq_xs[idx] = __half2float(xt[b0 * PXQ4HQ_QK + idx]);
        }
        __syncthreads();

        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq4hq_dot32<VECX>(pan_slabs + (size_t)kb * pxq4hq_pol::SLAB, row, anch,
                                    pxq4hq_xs + (size_t)(kb - b0) * PXQ4HQ_QK, tab, sub);
        }
        su += t;
    }

    red[kseg * PXQ4HQ_BM + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4HQ_BM + row];
        out[(size_t)iy * R + p * PXQ4HQ_BM + row] = __float2half_rn(u);
    }
}
