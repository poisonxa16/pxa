// pxq-mmvq.cuh — PXQ inside MMVQ: the int8/DP4A GEMV kernel every stock 4-bit type rides.
//
// WHY. ncu on the bespoke PXQ decode mmv (k_pxq6_mmv family) reports Block Limit = Registers,
// 19.8-38.6% achieved occupancy and 6-36% of DRAM peak: that kernel is register-limited and
// latency-bound, not bandwidth-bound. MMVQ's whole design avoids exactly that — one output row
// per block, a handful of live registers per thread, thousands of blocks — which is why MXFP4
// decodes faster on sm_70 despite carrying the same 4.25 bpw. Registering PXQ into MMVQ is the
// structural fix; capping registers on the existing kernel was already measured at -27.7%.
//
// NUMERIC CONTRACT — the FROZEN q3-s8 snap of pxq6i8.cuh:19-31, unchanged:
//   q_i = rint(book_i * 127/absmax)  (PX16 absmax == 1 -> the frozen s8 book below)
//   w   = (anchor * SUB[s] * (absmax/127)) * q_s8
// i.e. the /127 and the book absmax fold into the per-group fp32 eff scale and the stored fp16
// row anchor is untouched. Activations are quantized q8_1-style (per-32 absmax/127) — the SAME
// activation quantization MXFP4 gets on this path, so the comparison is like-for-like.
// NOT bit-exact vs the fused fp16 decode kernels: a fidelity gate is mandatory before shipping.
//
// SCOPE. p6 (GGML_TYPE_PXQ4) and p6hq (GGML_TYPE_PXQ4HQ) only — both are PX16-book tiers and
// both are covered by the snap validation. PXQ4's own E2M1 tier and the LM32/LM4/LM8 books are
// deliberately NOT here. If any PXA_PXQ6_BOOK/_SUB/_SUB_HQ table override is set the gate turns
// itself OFF (this TU carries its own frozen copies and does not take runtime uploads).
//
// ENV: PXA_PXQ_MMVQ = 0 (default, OFF: byte-identical dispatch to the previous build)
//                   | 1 (sm_70+ — the arch this was built for)
//                   | 2 (any arch with real DP4A, i.e. cc >= 610; TEST)
#pragma once

#include "../common.cuh"
#include "pxa-enhance.cuh"   // pxa_pxq_mmvq_auto_default: ENHANCE x model x device auto-set
#include "../../../include/ggml-pxq6-tables.h"

// ---------------------------------------------------------------------------------------------
// frozen tables (this TU's own copies; see the header note about env overrides)
// ---------------------------------------------------------------------------------------------
// s8 snap of the frozen PX16 book: rint(book_i * 127) — reproduces the frozen Q3 s8 book exactly.
static const __device__ __align__(16) int8_t pxq_mmvq_book_s8[16] = {
    -125, -93, -71, -53, -38, -25, -12, 0, 11, 22, 33, 46, 60, 76, 97, 127
};
#define PXQ_MMVQ_BFOLD (1.0f/127.0f)          // book absmax (== 1) / 127, folded into eff

static const __device__ float pxq_mmvq_sub16[16] = PXQ6_SUB16_INIT;
static const __device__ float pxq_mmvq_sub8 [16] = PXQ6_SUB8_INIT;

// ---------------------------------------------------------------------------------------------
// 4 nibble codes -> 4 s8 book values IN K ORDER (k = 2b, 2b+1 per code byte b).
// This is get_int_from_table_16() stopped one step early: its internal v3/v4 are already the
// sequential pairs, and the 0x6420/0x7531 remix at its tail is what splits them into the
// even/odd halves that Q4_0-style layouts need. PXQ stores consecutive pairs, so we want v3/v4.
//   .x = [w(k0), w(k1), w(k2), w(k3)]   .y = [w(k4), w(k5), w(k6), w(k7)]
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ int2 pxq_mmvq_table16_seq(const int q4, const int8_t * values) {
#if defined(__CUDA_ARCH__)
    const uint32_t * values32 = (const uint32_t *) values;
    const uint32_t mask = (0x32103210 | ((q4 & 0x88888888) >> 1));
    uint32_t v1 = __byte_perm(values32[0], values32[1], q4);
    uint32_t v2 = __byte_perm(values32[2], values32[3], q4);
    const uint32_t lo = __byte_perm(v1, v2, mask);            // codes 0..3, in order
    v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
    v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
    const uint32_t hi = __byte_perm(v1, v2, mask >> 16);      // codes 4..7, in order
    return make_int2((int) lo, (int) hi);
#else
    char4 a, b;
    a.x = values[(q4 >>  0) & 0xf]; a.y = values[(q4 >>  4) & 0xf];
    a.z = values[(q4 >>  8) & 0xf]; a.w = values[(q4 >> 12) & 0xf];
    b.x = values[(q4 >> 16) & 0xf]; b.y = values[(q4 >> 20) & 0xf];
    b.z = values[(q4 >> 24) & 0xf]; b.w = values[(q4 >> 28) & 0xf];
    return make_int2(*((const int *) &a), *((const int *) &b));
#endif
}

// ---------------------------------------------------------------------------------------------
// layout policies. Both tiers: panel = 128 B fp16 row-anchor header + kslabs slabs; slab = scale
// SoA + 64 x 16 B nibble code rows; panels row-major; experts outermost (the caller has already
// applied the expert offset, so e == 0 here). Codes are 16 B aligned for every row.
//   p6   : slab 1088 B, 64 B SoA, one 4-bit SUB16 per 16 elems (2 nibbles/row/slab)
//   p6hq : slab 1152 B, 128 B SoA, one 4-bit SUB8 per 8 elems (4 nibbles/row/slab)
// m = code word index 0..3 == k-group 8m..8m+7.
// ---------------------------------------------------------------------------------------------
struct pxq_mmvq_pol_p6 {
    static constexpr int SLAB = PXQ6_SLAB_BYTES, CODE_OFF = 64, NEFF = 2, SPR = 1;
    __device__ static const float * subtab() { return pxq_mmvq_sub16; }
    __device__ static int sidx(int i, int m) { return i; }             // 1 scale byte per row
    __device__ static int snib(int sb, int m) { return (sb >> (4*(m >> 1))) & 0xf; }
};

struct pxq_mmvq_pol_p6hq {
    static constexpr int SLAB = PXQ6HQ_SLAB_BYTES, CODE_OFF = 128, NEFF = 4, SPR = 2;
    __device__ static const float * subtab() { return pxq_mmvq_sub8; }
    __device__ static int sidx(int i, int m) { return 2*i + (m >> 1); } // 2 scale bytes per row
    __device__ static int snib(int sb, int m) { return (sb >> (4*(m & 1))) & 0xf; }
};

// The block's ROWS rows are adjacent, so their scale bytes are an adjacent run inside the slab's
// scale SoA: one word load per k-block instead of one byte load per row per k-block.
template <class POL, int ROWS>
struct pxq_mmvq_scales {
    static constexpr int NB = POL::SPR*ROWS;
    uint32_t w[(NB + 3)/4];
    __device__ void load(const uint8_t * __restrict__ slab, int r0) {
        const uint8_t * p = slab + POL::SPR*r0;
        if constexpr (NB % 4 == 0) {          // r0 is a multiple of ROWS, so p is 4 B aligned
#pragma unroll
            for (int i = 0; i < NB/4; ++i) w[i] = ((const uint32_t *) p)[i];
        } else {
#pragma unroll
            for (int i = 0; i < NB; ++i) ((uint8_t *) w)[i] = p[i];
        }
    }
    __device__ int byte(int i) const { return (w[i >> 2] >> (8*(i & 3))) & 0xff; }
};

#define VDR_PXQ_Q8_1_MMVQ 2      // MMVQ's own table wants a constant; the live value is templated

// ---------------------------------------------------------------------------------------------
// the vec_dot itself. Unlike the stock signature this takes (row, kb) instead of the flat block
// index: the flat index is recoverable only with blocks_per_row_x, which the PXQ panel address
// arithmetic needs anyway (panel stride depends on kslabs). k_mul_mat_vec_q hands both over.
// One thread covers 16 of the 32 k-values of one (row, slab): iqs == 0 -> code words 0,1
// (k 0..15), iqs == 2 -> words 2,3 (k 16..31).
// ---------------------------------------------------------------------------------------------
// Everything that depends only on (row) or only on (kb) is hoisted by the caller: `slab` is the
// per-kb slab base (one 64-bit IMAD per k iteration instead of two per row-and-k), and `anch` is
// the row's fp16 anchor with the frozen 1/127 book fold already applied (one LDG.U16 per row for
// the whole K instead of one per k-block). What is left is the irreducible per-(row,kb) work:
// one 8 B code load, one scale byte, the book gather, 4 dp4a.
// VDR = code words (8 k-values each) this thread owns: 2 -> a thread pair splits the 16 B code
// row and each issues LDG.64; 4 -> one thread owns the whole row and issues a single LDG.128,
// halving the memory instructions per useful byte at the cost of half the k-parallelism.
template <class POL, int VDR, int ROWS>
static __device__ __forceinline__ float vec_dot_pxq_q8_1(
        const uint8_t * __restrict__ slab, const int r, const int i, const float anch,
        const float * __restrict__ sub, const pxq_mmvq_scales<POL, ROWS> & sc,
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {

    // this thread's code bytes: CODE_OFF and SLAB are multiples of 64 and iqs is a multiple of
    // VDR, so the address is 4*VDR aligned.
    uint32_t qw[VDR];
    const uint8_t * cp = slab + POL::CODE_OFF + 16*r + 4*iqs;
    if constexpr (VDR == 4) { *(uint4 *)qw = *(const uint4 *)cp; }
    else                    { *(uint2 *)qw = *(const uint2 *)cp; }

    const int * q8 = (const int *) bq8_1->qs;

    float sum = 0.0f;
#pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const int  m = iqs + l;                            // code word == 8-element k group
        const int2 v = pxq_mmvq_table16_seq((int) qw[l], pxq_mmvq_book_s8);
        int s = ggml_cuda_dp4a(v.x, q8[2*m + 0], 0);
        s     = ggml_cuda_dp4a(v.y, q8[2*m + 1], s);
        sum  += sub[POL::snib(sc.byte(POL::sidx(i, m)), m)] * (float) s;
    }

    return anch * __low2float(bq8_1->ds) * sum;
}

// ---------------------------------------------------------------------------------------------
// host gates
// ---------------------------------------------------------------------------------------------
static inline bool pxa_pxq_mmvq_type(ggml_type t) {
    return t == GGML_TYPE_PXQ4 || t == GGML_TYPE_PXQ4HQ;
}

// MODEL-AWARE DEFAULT (2026-07-29): explicit PXA_PXQ_MMVQ always wins; with the env
// unset, ENHANCE auto-sets the mode from (model PXQ4/PXQ4HQ census x device DP4A
// capability) — see pxa_pxq_mmvq_auto_default() in pxa-enhance.cuh. Cached against the
// model-profile generation (the profile can land after CUDA init in the server flow).
static inline int pxa_pxq_mmvq_mode() {
    static int cached_gen = -1;
    static int cached     = 0;
    const int gen = ggml_pxa_model_profile_generation();
    if (cached_gen == gen) return cached;
    const char * e = getenv("PXA_PXQ_MMVQ");
    int m = e ? atoi(e) : pxa_pxq_mmvq_auto_default();
    if (m < 0 || m > 2) m = 0;
    // this TU holds frozen copies of the book / SUB tables; a runtime override would silently
    // diverge from the fused kernels, so decline instead — LOUDLY, auto-set or not.
    if (m && (getenv("PXA_PXQ6_BOOK") || getenv("PXA_PXQ6_SUB") || getenv("PXA_PXQ6_SUB_HQ"))) {
        fprintf(stderr, "PXA_PXQ_MMVQ: DISABLED — a PXA_PXQ6_BOOK/_SUB/_SUB_HQ override is set%s\n",
                e ? "" : " (declining the ENHANCE auto-set)");
        m = 0;
    }
    if (m) {
        fprintf(stderr, "PXA_PXQ_MMVQ: mode %d%s (PXQ4/PXQ4HQ decode via the q8_1 MMVQ kernel, "
                        "s8 book snap — NOT bit-exact vs the fused fp16 mmv, fidelity-gated)\n",
                m, e ? "" : " [AUTO: DEFAULT/ENHANCE x PXQ4-bearing model x DP4A device; override PXA_PXQ_MMVQ=0]");
    }
    cached = m;
    cached_gen = gen;
    return m;
}

// PXQ tile height for the MMVQ block, 1|2|4|8|16 (default 4). See mmvq-templates.cuh.
static inline int pxa_pxq_mmvq_rows() {
    static const int rows = [](){
        const char * e = getenv("PXA_PXQ_MMVQ_ROWS");
        int r = e ? atoi(e) : 4;
        if (r != 1 && r != 2 && r != 4 && r != 8 && r != 16) r = 4;
        fprintf(stderr, "PXA_PXQ_MMVQ_ROWS: %d rows per block\n", r);
        return r;
    }();
    return rows;
}

// code words per thread: 2 (LDG.64, a thread pair per code row) or 4 (LDG.128, one thread).
static inline int pxa_pxq_mmvq_vdr() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQ_MMVQ_VDR");
        int r = e ? atoi(e) : 2;
        if (r != 2 && r != 4) r = 2;
        fprintf(stderr, "PXA_PXQ_MMVQ_VDR: %d code words per thread\n", r);
        return r;
    }();
    return v;
}

static inline bool pxa_pxq_mmvq_on(int cc) {
    const int m = pxa_pxq_mmvq_mode();
    if (m == 0) return false;
    if (m == 1) return cc >= CC_VOLTA;
    return cc >= 610;                 // real DP4A only; sm_60 would run the emulation
}
