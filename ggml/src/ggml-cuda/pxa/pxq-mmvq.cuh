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
// SCOPE. The two PX16-book tiers p6 (GGML_TYPE_PXQ4) and p6hq (GGML_TYPE_PXQ4HQ) unconditionally,
// and since 2026-09-09 the two LOW tiers PXQ2 (LM4) and PXQ3 (LM8, bit-plane) behind their own
// lever PXA_PXQ23_MMVQ — same contract, their own frozen s8 books and their own fidelity
// evidence; the weight decode for those two lives in pxq23-mmvq.h so it can be tested on a CPU.
// PXQ4's own E2M1 tier and the LM32 book are deliberately NOT here. If any table override is set
// (PXA_PXQ6_BOOK/_SUB/_SUB_HQ here, PXA_PXQ2_BOOK/PXQ3_BOOK/CEIL_V2/PXQ2_V3/_SUB there) the
// affected gate turns itself OFF — this TU carries frozen copies and takes no runtime upload.
//
// ENV: PXA_PXQ_MMVQ = 0 (default, OFF: byte-identical dispatch to the previous build)
//                   | 1 (sm_70+ — the arch this was built for)
//                   | 2 (any arch with real DP4A, i.e. cc >= 610; TEST)
#pragma once

#include "../common.cuh"
#include <atomic>
#include <cstdlib>
#include "pxa-enhance.cuh"   // pxa_pxq_mmvq_auto_default: ENHANCE x model x device auto-set
#include "pxq23-mmvq.h"      // the PXQ2/PXQ3 weight decode (host-testable; see that header)
#include "../../../include/ggml-pxq2-tables.h"
#include "../../../include/ggml-pxq3-tables.h"
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
// layout policies. Every tier: panel = 128 B fp16 row-anchor header + kslabs slabs; slab = scale
// SoA + 64 code rows; panels row-major; experts outermost (the caller has already applied the
// expert offset, so e == 0 here).
//   p6   : slab 1088 B, 64 B SoA, 16 B code rows, one 4-bit SUB16 per 16 elems
//   p6hq : slab 1152 B, 128 B SoA, 16 B code rows, one 4-bit SUB8 per 8 elems
//   p2   : slab  576 B, 64 B SoA,  8 B code rows, one 4-bit SUB16 per 16 elems (as p6)
//   p3   : slab  832 B, 64 B SoA, 12 B code rows, one 4-bit SUB16 per 16 elems (as p6)
// m = code group index 0..3 == k-group 8m..8m+7. A policy owns exactly three things the tiers
// disagree about: which scale nibble a group takes, how many u32 words of code a thread's run
// spans, and how a group's 8 codes become 8 s8 book values in k order. Everything else — the
// q8_1 side, the accumulate, the launcher, the row/anchor hoisting — is shared verbatim.
//   BFOLD is the tier's book absmax / 127; it rides in the row anchor (see pxq_mmvq_rowbase) so
// the inner loop carries one fewer multiply. For the two PX16-book tiers it is 1/127 exactly,
// which is what PXQ_MMVQ_BFOLD has always been.
// ---------------------------------------------------------------------------------------------
struct pxq_mmvq_pol_p6 {
    static constexpr int SLAB = PXQ6_SLAB_BYTES, CODE_OFF = 64, NEFF = 2, SPR = 1;
    static constexpr float BFOLD = PXQ_MMVQ_BFOLD;
    __device__ static const float * subtab() { return pxq_mmvq_sub16; }
    __device__ static int sidx(int i, int m) { return i; }             // 1 scale byte per row
    __device__ static int snib(int sb, int m) { return (sb >> (4*(m >> 1))) & 0xf; }

    // one u32 of nibble codes per group, so a thread's VDR groups are one contiguous load
    template <int VDR> static constexpr int NWORDS = VDR;
    template <int VDR>
    __device__ static void load_words(const uint8_t * __restrict__ slab, int r, int iqs, uint32_t * qw) {
        const uint8_t * cp = slab + CODE_OFF + 16*r + 4*iqs;
        if constexpr (VDR == 4) { *(uint4 *)qw = *(const uint4 *)cp; }
        else                    { *(uint2 *)qw = *(const uint2 *)cp; }
    }
    template <int VDR>
    __device__ static int2 group(const uint32_t * qw, int l, int iqs) {
        return pxq_mmvq_table16_seq((int) qw[l], pxq_mmvq_book_s8);
    }
};

struct pxq_mmvq_pol_p6hq {
    static constexpr int SLAB = PXQ6HQ_SLAB_BYTES, CODE_OFF = 128, NEFF = 4, SPR = 2;
    static constexpr float BFOLD = PXQ_MMVQ_BFOLD;
    __device__ static const float * subtab() { return pxq_mmvq_sub8; }
    __device__ static int sidx(int i, int m) { return 2*i + (m >> 1); } // 2 scale bytes per row
    __device__ static int snib(int sb, int m) { return (sb >> (4*(m & 1))) & 0xf; }

    template <int VDR> static constexpr int NWORDS = VDR;
    template <int VDR>
    __device__ static void load_words(const uint8_t * __restrict__ slab, int r, int iqs, uint32_t * qw) {
        const uint8_t * cp = slab + CODE_OFF + 16*r + 4*iqs;
        if constexpr (VDR == 4) { *(uint4 *)qw = *(const uint4 *)cp; }
        else                    { *(uint2 *)qw = *(const uint2 *)cp; }
    }
    template <int VDR>
    __device__ static int2 group(const uint32_t * qw, int l, int iqs) {
        return pxq_mmvq_table16_seq((int) qw[l], pxq_mmvq_book_s8);
    }
};

// PXQN4S8: its policy is defined in the closed libggml-pxqn instance TU (the only TU that instantiates it)
struct pxqn_mmvq_pol_n4s8;

// PXQ2: 2-bit codes, 8 B rows, one u32 per 16 elems -- so ONE word carries two groups and a
// thread's whole VDR run is one 4 B (VDR 2) or 8 B (VDR 4) load, half of what the 4-bit tiers
// move. iqs is always even (it is VDR * a thread index and VDR is 2 or 4), so group l takes the
// low half of its word when l is even and the high half when l is odd -- a compile-time choice,
// not a runtime one.
struct pxq_mmvq_pol_p2 {
    static constexpr int SLAB = PXQ2_SLAB_BYTES, CODE_OFF = 64, NEFF = 2, SPR = 1;
    static constexpr float BFOLD = PXQ2_MMVQ_BFOLD;
    __device__ static const float * subtab() { return pxq_mmvq_sub16; }
    __device__ static int sidx(int i, int m) { return i; }
    __device__ static int snib(int sb, int m) { return (sb >> (4*(m >> 1))) & 0xf; }

    template <int VDR> static constexpr int NWORDS = VDR/2;
    template <int VDR>
    __device__ static void load_words(const uint8_t * __restrict__ slab, int r, int iqs, uint32_t * qw) {
        const uint8_t * cp = slab + CODE_OFF + 8*r + 4*(iqs >> 1);
        if constexpr (VDR == 4) { *(uint2 *)qw = *(const uint2 *)cp; }
        else                    { qw[0] = *(const uint32_t *)cp; }
    }
    template <int VDR>
    __device__ static int2 group(const uint32_t * qw, int l, int iqs) {
        const pxq_mmvq_g8 g = pxq2_mmvq_gather8(pxq2_mmvq_lo16(qw[l >> 1], l));
        return make_int2(g.x, g.y);
    }
};

// PXQ3: bit-plane. The low plane is addressed exactly as PXQ2's codes are; the high plane is one
// more u32 for the whole 32-element slab, of which a group takes byte m. A thread therefore loads
// its low word(s) plus that one shared word -- 12 B rows never reach 8 B alignment, so these stay
// separate 4 B loads rather than a wide one that could walk off the last row of the last slab.
struct pxq_mmvq_pol_p3 {
    static constexpr int SLAB = PXQ3_SLAB_BYTES, CODE_OFF = 64, NEFF = 2, SPR = 1;
    static constexpr float BFOLD = PXQ3_MMVQ_BFOLD;
    __device__ static const float * subtab() { return pxq_mmvq_sub16; }
    __device__ static int sidx(int i, int m) { return i; }
    __device__ static int snib(int sb, int m) { return (sb >> (4*(m >> 1))) & 0xf; }

    template <int VDR> static constexpr int NWORDS = VDR/2 + 1;   // low-plane words + the high plane
    template <int VDR>
    __device__ static void load_words(const uint8_t * __restrict__ slab, int r, int iqs, uint32_t * qw) {
        const uint8_t * cp = slab + CODE_OFF + 12*r;
#pragma unroll
        for (int t = 0; t < VDR/2; ++t) qw[t] = *(const uint32_t *)(cp + 4*((iqs >> 1) + t));
        qw[VDR/2] = *(const uint32_t *)(cp + 8);
    }
    template <int VDR>
    __device__ static int2 group(const uint32_t * qw, int l, int iqs) {
        const pxq_mmvq_g8 g = pxq3_mmvq_gather8(pxq2_mmvq_lo16(qw[l >> 1], l),
                                                pxq3_mmvq_hi8 (qw[VDR/2], iqs + l));
        return make_int2(g.x, g.y);
    }
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

// PXA_SPEC_FAST_VERIFY (2026-09-28): 0 (default) = every decode GEMV at 2..8 columns (the MTP /
// draft verify widths) is bit-identical to the same column decoded alone (batch-invariant); 1 = the verify-width
// GEMVs take cheaper per-column arithmetic (V100 MMVQ: one int->float per 16-group and the row anchor hoisted). Deterministic run to run (fixed order, no
// atomics); greedy output with speculation on may then differ from speculation off. ONE definition shared by the
// PXQN kernel header and the MMVQ header (guarded: both may be in one TU).
#ifndef PXA_SPEC_FAST_VERIFY_DEFINED
#define PXA_SPEC_FAST_VERIFY_DEFINED
#include <cstdio>
#include <cstdlib>
inline int pxa_spec_fast_verify_env() {
    static const int v = [](){
        const char * e = getenv("PXA_SPEC_FAST_VERIFY");
        const int r = e ? atoi(e) : 0;
        if (r) fprintf(stderr, "PXA_SPEC_FAST_VERIFY: 1 (verify-width PXQN4 GEMVs use fast arithmetic: deterministic, "
                               "NOT bit-identical to 1-column decode)\n");
        return r ? 1 : 0;
    }();
    return v;
}
static inline bool pxa_spec_fast_verify(int cc) { (void) cc; return pxa_spec_fast_verify_env() != 0; }
#endif

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
//
// This whole-dot form is what the ncols_y == 1 launch runs. Multi-column launches run the split
// form below (pxq_mmvq_wfrag + pxq_mmvq_dot_frag), which is this function with its weight-only
// prologue lifted out of the column loop and is bit-identical to it; this one then doubles as the
// reference arm of tests/test-pxq-mmvq-cols.cu, so KEEP THE TWO ARITHMETICALLY IN STEP.
template <class POL, int VDR, int ROWS>
static __device__ __forceinline__ float vec_dot_pxq_q8_1(
        const uint8_t * __restrict__ slab, const int r, const int i, const float anch,
        const float * __restrict__ sub, const pxq_mmvq_scales<POL, ROWS> & sc,
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {

    // this thread's code words, however many the tier packs its VDR groups into
    uint32_t qw[POL::template NWORDS<VDR>];
    POL::template load_words<VDR>(slab, r, iqs, qw);

    const int * q8 = (const int *) bq8_1->qs;

    float sum = 0.0f;
#pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const int  m = iqs + l;                            // code group == 8-element k group
        const int2 v = POL::template group<VDR>(qw, l, iqs);
        int s = ggml_cuda_dp4a(v.x, q8[2*m + 0], 0);
        s     = ggml_cuda_dp4a(v.y, q8[2*m + 1], s);
        sum  += sub[POL::snib(sc.byte(POL::sidx(i, m)), m)] * (float) s;
    }

    return anch * __low2float(bq8_1->ds) * sum;
}

// ---------------------------------------------------------------------------------------------
// WEIGHT-SIDE DECODE, HOISTED OUT OF THE COLUMN LOOP (2026-09-08)
// ---------------------------------------------------------------------------------------------
// Everything vec_dot_pxq_q8_1() does before its first dp4a is a function of the WEIGHT alone: the
// 8 B/16 B code load, the byte-perm book gather, and the per-code-word sub-scale lookup all depend
// on (slab, r, i, iqs) and never on which activation column is being scored. A GEMV that scores
// several activation columns against the same rows -- the spec-verify / multi-slot widths this
// kernel is explicitly built for, see the ROWS note in mmvq-templates.cuh -- was therefore
// unpacking the same nibbles ncols_y times over. Split the work in two: pxq_mmvq_wfrag decodes one
// (row, k-block) once, pxq_mmvq_dot_frag consumes it once per column.
//
// The per-column arithmetic and its ORDER are untouched. For every (row, column) pair the l loop
// still runs 0..VDR-1 and still does dp4a(v.x, ...) -> dp4a(v.y, ...) -> fma of eff*(float)s into
// `sum`, and still finishes with anch * ds * sum, so each output is bit-identical to the
// unhoisted vec_dot_pxq_q8_1() by construction. The caller's j/i loop swap does not change it
// either: every tmp[j][i] still receives exactly one addend per k-block, in k-block order.
// tests/test-pxq-mmvq-cols.cu proves it differentially against vec_dot_pxq_q8_1() above, which is
// kept verbatim as the reference arm (it is not instantiated by the engine build).
//
// COST. VDR=2 -> 4 int + 2 float, VDR=4 -> 8 int + 4 float held live across the column loop. The
// ncols_y == 1 caller does not use this path at all (see k_mul_mat_vec_q), so the single-column
// register budget -- the one the ROWS sweep was tuned against -- is untouched.
// ---------------------------------------------------------------------------------------------
template <class POL, int VDR, int ROWS>
struct pxq_mmvq_wfrag {
    int2  v  [VDR];      // book values in k order: .x = k 0..3 of the word, .y = k 4..7
    float eff[VDR];      // that code word's sub-scale

    __device__ __forceinline__ void load(
            const uint8_t * __restrict__ slab, const int r, const int i,
            const float * __restrict__ sub, const pxq_mmvq_scales<POL, ROWS> & sc, const int iqs) {
        uint32_t qw[POL::template NWORDS<VDR>];
        POL::template load_words<VDR>(slab, r, iqs, qw);
#pragma unroll
        for (int l = 0; l < VDR; ++l) {
            const int m = iqs + l;                         // code group == 8-element k group
            v  [l] = POL::template group<VDR>(qw, l, iqs);
            eff[l] = sub[POL::snib(sc.byte(POL::sidx(i, m)), m)];
        }
    }
};

// The irreducibly per-column half: 2*VDR dp4a and VDR fma against an already-decoded fragment.
template <class POL, int VDR, int ROWS>
static __device__ __forceinline__ float pxq_mmvq_dot_frag(
        const pxq_mmvq_wfrag<POL, VDR, ROWS> & w, const float anch,
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {

    const int * q8 = (const int *) bq8_1->qs;

    float sum = 0.0f;
#pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const int m = iqs + l;
        int s = ggml_cuda_dp4a(w.v[l].x, q8[2*m + 0], 0);
        s     = ggml_cuda_dp4a(w.v[l].y, q8[2*m + 1], s);
        sum  += w.eff[l] * (float) s;
    }

    return anch * __low2float(bq8_1->ds) * sum;
}

// PXA_SPEC_FAST_VERIFY (default 0 = off): the per-column half at verify widths (2..8 columns) with
// cheaper arithmetic. NOT bit-identical to pxq_mmvq_dot_frag and NOT batch-invariant against the 1-column launch
// (a verify column differs from the same token decoded alone in the last bits); deterministic run to run (fixed
// order, no atomics). Two relaxations:
//   (1) the thread's two code words of one 16-group share ONE sub-scale (tiers with one SUB16 per 16 elements,
//       NEFF == 2, iqs even), so their 4 dp4a run as one integer chain and take one int->float and one fma
//       (exact: eff*s0 + eff*s1 in fp32; fast: eff*(s0 + s1), the integer sum is exact);
//   (2) the row anchor leaves the per-k-block product: the caller multiplies the finished row sum by it once.
template <class POL, int VDR, int ROWS>
static __device__ __forceinline__ float pxq_mmvq_dot_frag_fast(
        const pxq_mmvq_wfrag<POL, VDR, ROWS> & w, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    static_assert(POL::NEFF == 2 && VDR % 2 == 0, "fast verify: one sub-scale per code-word pair");
    const int * q8 = (const int *) bq8_1->qs;
    float sum = 0.0f;
#pragma unroll
    for (int l = 0; l < VDR; l += 2) {
        const int m = iqs + l;
        int s = ggml_cuda_dp4a(w.v[l    ].x, q8[2*m + 0], 0);
        s     = ggml_cuda_dp4a(w.v[l    ].y, q8[2*m + 1], s);
        s     = ggml_cuda_dp4a(w.v[l + 1].x, q8[2*m + 2], s);
        s     = ggml_cuda_dp4a(w.v[l + 1].y, q8[2*m + 3], s);
        sum   = fmaf(w.eff[l], (float) s, sum);
    }
    return __low2float(bq8_1->ds) * sum;
}


// ---------------------------------------------------------------------------------------------
// host gates
// ---------------------------------------------------------------------------------------------
// Which PXQ tiers may take this path. The two PX16-book tiers always may (they are what it was
// built for); the two LOW tiers may when PXA_PXQ23_MMVQ admits them — see the lever, its two
// refusals and the s8 self-check in pxa-enhance.cuh. Every caller that routes a node to MMVQ
// asks this one question, so one predicate moves the whole dispatch.
// PXA_PXQN_MMVQ: 1 (default) = PXQN4 / PXQN4S8 decode GEMVs (width <= 8, not the fused up/gate) on
// sm_70 take the PXQ4 / PXQ4-HQ q8_1 MMVQ kernel -- the stored bytes, book and SUB table are PXQ4's, so it computes
// the same w; x is q8_1-quantized (not the PXQN GEMV's staging), so numerics are the PXQ4 MMVQ class. Measured cause
// (V100 tg64 nvprof): PXQN GEMVs 1250 ms vs PXQ4 MMVQ 1152 ms for the same bytes; 0 = the PXQN GEMV.
static inline bool pxqn_mmvq_route_on() {
    static const bool v = [](){ const char * e = getenv("PXA_PXQN_MMVQ"); return !(e && atoi(e) == 0); }();
    return v;
}
static inline bool pxa_pxq_mmvq_type(ggml_type t) {
    if (t == GGML_TYPE_PXQ4 || t == GGML_TYPE_PXQ4HQ) return true;
    const int m23 = pxa_pxq23_mmvq_mask();
    return (t == GGML_TYPE_PXQ2 && (m23 & 1)) || (t == GGML_TYPE_PXQ3 && (m23 & 2));
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
    const bool e = pxa_lever_set_by_user(PXA_LEVER_PXQ_MMVQ);   // the variable present, else the registry's auto rule
    int m = (int) pxa_lever(PXA_LEVER_PXQ_MMVQ);
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

// MULTI-COLUMN TILE HEIGHT (PXA_PXQ_MMVQ_COLS, default on at sm_70+ since 2026-09-27, see below).
//
// The multi-column MMVQ block - the one a speculative verify batch of 2..8 tokens lands on - used
// to be clamped to 2 rows at ncols_y > 2, on the reasoning that ncols_y*ROWS accumulators would
// otherwise exhaust the register file. They never were in the register file: the kernel's
// write-back indexed the accumulator array with threadIdx.x, which put the whole array in LOCAL
// memory, so raising the tile raised the spill by the same factor it cut the block count and
// every tile sweep read as a null. With that fixed (pxa_lane_pick in mmvq-templates.cuh) the
// clamp is backwards, and the optimum moves the other way.
//
// Measured standalone with nvcc on the real dense ffn up/gate half-shape (K = 5120, 17408 rows,
// PXQ4), us per launch. TWO changes landed together and they must be read apart: the write-back
// fix above is UNCONDITIONAL and moves every width on its own, and only the third column below is
// this lever. Quoting the first column against the third credits the lever with both.
//
//   sm_70 (V100), the architecture the engine dispatches this kernel on:
//
//   width   before the write-back fix   this branch, lever OFF   lever ON      lever's own delta
//     1              62.9                       63.9            63.9   R4        untouched
//     2              95.3                       67.5            67.5   R4        untouched
//     4             150.9                      122.6            88.0   R8          -28.2%
//     8             211.5                      185.3           136.9   R4          -26.1%
//
// Against this branch's own width-1 step (63.9 us) a width-4 verify step costs 1.92x with the
// lever off and 1.38x with it on. Every arm is BIT-IDENTICAL to the clamped one: the tile height
// changes which rows a block owns, not which k-blocks a thread accumulates, nor the order it
// accumulates them.
//
// The same harness on a P100 reads 134.9 / 171.5 / 350.1 / 582.8 us with the lever off and
// 134.9 / 171.5 / 257.0 / 420.2 with it on (-26.6% and -27.9% at widths 4 and 8), but THOSE ARE
// STANDALONE-KERNEL NUMBERS ONLY and no P100 can reproduce them from the engine: the standalone
// harness compiles its own DP4A emulation for __CUDA_ARCH__ < 610, while pxa_pxq_mmvq_on() below
// admits cc >= 700 (mode 1) or cc >= 610 (mode 2) and never cc 600, so a P100 keeps the bespoke
// 2D PXQ drivers and this lever is inert there. Confirmed on hardware 2026-09-20, P100 pair,
// PXA_PXQ_MMVQ=1 forced and PXA_PXQ_MMVQ_COLS=1: zero multi-column launches, the lever's banner
// never printed, PXA_PXQ4_2D_SPLIT and PXA_PXQ_DENSE_GATEUP owned the node on both devices.
// Reaching it on sm_60 means moving the admission gate, which is a different piece of work.
//
// ncols_y <= 2 is declined outright, so plain decode and a width-2 verify cannot move whatever
// this is set to.
//
// DEFAULT (2026-09-27, , plan item S7a): unset = ON on sm_70 and newer at ENHANCE, OFF below
// (sm_61 fleets, the only other arch this kernel runs on, never measured it). Measured +4.4% greedy
// decode on the V100 pair under MTP verify (58.64 -> 61.21 t/s, same acceptance; RESEARCH-NEW-LEVERS
// row 2); re-probed 2026-09-27 on the release build (impl s7a-mmvq-cols-sm70), 2x V100, 27B PXQ4, MTP,
// REPS 3: -sm layer off/on/off 44.39/49.14/44.60 control, 64.81/77.12/67.12 repetition (+10.2%/+14.9%
// on the close bracket), under the pair's default -sm tensor a wash inside a drifting bracket, inert at
// width 1 (plain decode), greedy shas identical in every arm. =0 turns it off everywhere (the old
// default), =1 arms it on every card the kernel runs on.
inline void pxa_pxq_mmvq_cols_arm();            // defined below, next to the counter it reports
static inline int pxa_pxq_mmvq_cols_env() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQ_MMVQ_COLS");
        return e ? (atoi(e) != 0 ? 1 : 0) : -1;
    }();
    return v;
}
static inline bool pxa_pxq_mmvq_cols(int cc) {
    const int  e = pxa_pxq_mmvq_cols_env();
    // the default follows the config level like every ENHANCE lever: PXA_ENHANCE=0 / PXA_REFERENCE=1 keep the clamp
    const bool v = e >= 0 ? e == 1 : (cc >= CC_VOLTA && ggml_pxa_config_level() >= 2);
    if (v) {
        static const bool said = [e](){
            fprintf(stderr, "PXA_PXQ_MMVQ_COLS: multi-column PXQ MMVQ tile = 8 rows at 3-4 columns, 4 at 5-8 "
                            "(%s; =0 restores 2; PXA_PXQ_MMVQ_ROWS is not consulted above 2 columns)\n",
                            e == 1 ? "armed by PXA_PXQ_MMVQ_COLS=1" : "default on sm_70+");
            pxa_pxq_mmvq_cols_arm();   // banner and exit report always come as a pair
            return true;
        }();
        (void) said;
    }
    return v;
}

// How many multi-column launches actually took the lever's tile, reported at exit so a run that
// quotes a number can say whether the lever ran at all -- a lever that printed its banner and
// then engaged zero nodes has cost a measurement window before now.
//
// THE REPORT IS ARMED FROM THE LEVER, NOT FROM THE FIRST ENGAGEMENT. Arming it where the counter
// is incremented would have made the zero case -- the one case the counter exists to catch --
// print nothing at all, byte-identical to a lever-off run, so a harness grepping for the line
// could never tell "did not engage" from "was not armed".
//
// ONE counter and ONE report line for the whole process: the four PXQ tiers are four translation
// units, so a `static` here would tally and print four times and a reader would have to add them
// up. C++17 inline linkage merges both the counter and the guard that arms the report.
//
// The count is launches SUBMITTED. That equals launches executed on every path this engine ships,
// because CUDA-graph replay at decode is env-only and default OFF (pxa_cuda_graph_decode_enabled);
// under PXA_CUDA_GRAPH_DECODE=1 a captured launch would be counted once per CAPTURE and not once
// per replayed token, so two arms' counts would compare captures, not work.
inline std::atomic<unsigned long long> pxa_pxq_mmvq_cols_n{0};
inline void pxa_pxq_mmvq_cols_report() {
    fprintf(stderr, "PXA_PXQ_MMVQ_COLS: %llu multi-column launches took the wide tile\n",
            pxa_pxq_mmvq_cols_n.load(std::memory_order_relaxed));
}
inline void pxa_pxq_mmvq_cols_arm() {
    static const bool armed = [] { atexit(pxa_pxq_mmvq_cols_report); return true; }();
    (void) armed;
}

// The tile height for one launch. Up to 2 columns it is the PXA_PXQ_MMVQ_ROWS sweep value; above
// 2 columns it is the clamp (lever off) or this lever's fixed table (lever on), and in BOTH of
// those cases PXA_PXQ_MMVQ_ROWS is NOT consulted -- the clamp has always ignored it and the
// lever's table is a measured pair of tile heights, not a ceiling on a sweep. A ROWS x COLS
// sweep therefore reads flat above two columns for a reason outside the ROWS knob; the lever's
// banner says so, because a reader who does not know it records a wrong null.
//
// With the lever off this reproduces the clamp that used to live in mmvq_rows_per_block exactly,
// which is why an unarmed launch is bit-identical to the previous build.
//
// `fused` is the up/gate twin, which carries TWO accumulator tiles and two cross-warp staging
// buffers, so it gets half the tile. THE TWIN'S OWN OPTIMUM WAS NOT MEASURED - halving is a
// reading of the same register and shared budget, not a number. It also decides which arm the
// table above describes: PXA_PXQ_MMVQ_FUGSPLIT is auto-ON at sm_70+, so on a Volta card the dense
// ffn up/gate runs as two PLAIN launches and gets exactly those numbers, and the twin's halved
// tile is taken on trust. The twin's branch is reachable on sm_70 only with
// PXA_PXQ_MMVQ_FUGSPLIT=0, and it has no number: measure it there, or drop the fused arm.
static inline int pxa_pxq_mmvq_tile(int ncols_y, bool fused, int cc) {
    const int r = pxa_pxq_mmvq_rows();
    // Read the lever at EVERY width, before the early return, so the banner and the exit report
    // are armed by the first PXQ MMVQ launch of any width. The pair is then a usable signal:
    // banner without report is impossible, and no banner at all means this kernel never ran.
    const bool lever = pxa_pxq_mmvq_cols(cc);
    if (ncols_y <= 2) {
        return r;
    }
    const int clamped = r > 2 ? 2 : r;
    if (lever) {
        const int wide = fused ? (ncols_y <= 4 ? 4 : 2) : (ncols_y <= 4 ? 8 : 4);
        if (wide != clamped) {                 // count launches the lever actually MOVED
            pxa_pxq_mmvq_cols_n.fetch_add(1, std::memory_order_relaxed);
        }
        return wide;
    }
    return clamped;
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

// Same question, asked with the tier in hand. Mode 2 (an all-sm_61 fleet) carries PXQ4 and
// PXQ4HQ on their own 2026-07-28 evidence; it does NOT carry the two low tiers, whose decode win
// was measured on sm_70 and only on sm_70. So a 1080 Ti fleet keeps the bespoke fused decode for
// PXQ2/PXQ3 until somebody measures it there — a lever should arm on the arch its gate passed on.
static inline bool pxa_pxq_mmvq_on_type(int cc, ggml_type t) {
    if (!pxa_pxq_mmvq_on(cc)) return false;
    if (t == GGML_TYPE_PXQ2 || t == GGML_TYPE_PXQ3) return cc >= CC_VOLTA;
    return true;
}
