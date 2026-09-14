// pxq23-mmvq.h — the PXQ2 / PXQ3 weight side of the q8_1 MMVQ decode path.
//
// WHY THIS IS A PLAIN HEADER AND NOT A .cuh. Everything below is pure integer bit work on the
// code stream plus one PRMT-shaped table gather, and every line of it is a correctness question
// that does not need a GPU to answer: which two bits of which word are element j's code, which
// bit of the high plane carries its third bit, and which book entry that pair names. Compiling
// the SAME source on the host — with PRMT emulated exactly — lets tests/test-pxq23-mmvq-snap.cpp
// check all 2^16 PXQ2 code patterns and all 2^24 PXQ3 patterns exhaustively on a CPU, so the
// device build is left proving only what a device can prove (see tests/test-pxq-mmvq-cols.cu).
//
// NUMERIC CONTRACT. Identical in shape to the PXQ4 contract frozen in pxq-mmvq.cuh:
//
//     q_i = rint(book_i * 127/absmax)            (the frozen s8 books below)
//     w   = (anchor * SUB16[s] * (absmax/127)) * q_i
//
// i.e. the book's own absmax and the /127 fold into a per-tier fp32 constant that rides along in
// the fp16 row anchor, and the stored anchor and sub-scale are untouched. This is NOT bit-exact
// against the fused fp16 decode kernels — the weights are snapped to int8 and the activations are
// quantised q8_1-style, exactly as on the PXQ4 path — so a fidelity gate is mandatory.
//
// WHY 127/absmax IS THE RIGHT SCALE and not something fitted. A dot product propagates the
// ABSOLUTE snap error, |dw| <= 0.5/S in units of (anchor*SUB16), so the best scale is the largest
// one that keeps every |q_i| <= 127 — that is exactly 127/absmax. (Scales that minimise the
// worst RELATIVE error instead were tried and are strictly worse here: they shrink S, which
// enlarges every absolute error to buy accuracy on the book's smallest entry alone.) The
// resulting worst-case per-weight error is 0.28% of absmax for PXQ2 and 0.36% for PXQ3, which is
// 0.5% and 1.8% of those books' own level spacing against PXQ4's 3% — the two new tiers snap
// RELATIVELY CLEANER than the tier already shipped on this path.
//
// FORMAT (from ggml-pxq2-tables.h / ggml-pxq3-tables.h, restated here as the thing being decoded;
// slab = 64 B scale SoA + 64 code rows, panel = 128 B fp16 anchors + kslabs slabs, 32 elems/slab):
//   PXQ2  8 B/row : two LE u32 words, word h covers elems 16h..16h+15, elem j at bits 2*(j&15).
//   PXQ3 12 B/row : three LE u32 words, BIT-PLANE. w0/w1 = the low 2 bits of elems 0-15 / 16-31
//                   at bits 2*(j&15); w2 = the high bit, bit e for elem e (0..31).
// Both tiers carry PXQ6-core's scale machinery unchanged: one scale byte per row per slab, low
// nibble for elems 0-15 and high nibble for elems 16-31, through the shared frozen SUB16 LUT.
//
// UNIT OF WORK. MMVQ's thread map is expressed in "code groups" of 8 k-values (qi == 4 groups per
// 32-element block, the MXFP4/PXQ4 shape), and everything here is written in those terms, so the
// q8_1 side, the scale side and the launcher are shared with PXQ4 verbatim.
#pragma once

#include <stdint.h>

#include "../../../include/ggml-pxq2-tables.h"
#include "../../../include/ggml-pxq3-tables.h"

#if defined(__CUDACC__)
#  define PXQ23_HD __host__ __device__ __forceinline__
#else
#  define PXQ23_HD inline
#endif

// ---------------------------------------------------------------------------------------------
// frozen s8 books. Derived from PXQ2_BOOK_INIT / PXQ3_BOOK_INIT by the contract above; the host
// gate in pxq-mmvq.cuh recomputes them from those tables at startup and refuses the path if the
// tables ever move, so these constants can never silently disagree with the codec.
// ---------------------------------------------------------------------------------------------
#define PXQ2_MMVQ_S8_INIT { -127, -34,  34, 126 }
#define PXQ3_MMVQ_S8_INIT { -127, -77, -42, -13,  13,  41,  76, 127 }

#define PXQ2_MMVQ_ABSMAX 0.70556640625f     // max|PXQ2_BOOK_INIT|
#define PXQ3_MMVQ_ABSMAX 0.90673828125f     // max|PXQ3_BOOK_INIT|

// absmax/127, folded into the row anchor exactly as PXQ_MMVQ_BFOLD is on the PXQ4 path
#define PXQ2_MMVQ_BFOLD (PXQ2_MMVQ_ABSMAX/127.0f)
#define PXQ3_MMVQ_BFOLD (PXQ3_MMVQ_ABSMAX/127.0f)

// The books as PRMT sources: 4 s8 bytes per word, so a code is a byte index into them. These are
// compile-time constants, not device memory — the gather costs no load at all.
static constexpr uint32_t pxq_mmvq_pack4(int a, int b, int c, int d) {
    return ((uint32_t)(uint8_t)a)        | ((uint32_t)(uint8_t)b <<  8)
         | ((uint32_t)(uint8_t)c << 16)  | ((uint32_t)(uint8_t)d << 24);
}

static constexpr uint32_t PXQ2_MMVQ_BK  = pxq_mmvq_pack4(-127, -34,  34, 126);   // codes 0..3
static constexpr uint32_t PXQ3_MMVQ_BKA = pxq_mmvq_pack4(-127, -77, -42, -13);   // codes 0..3
static constexpr uint32_t PXQ3_MMVQ_BKB = pxq_mmvq_pack4(  13,  41,  76, 127);   // codes 4..7

// ---------------------------------------------------------------------------------------------
// PRMT, and its exact emulation for the host build. CUDA's __byte_perm(a, b, s) builds 4 result
// bytes from the 8-byte concatenation {a.b0..a.b3, b.b0..b.b3}: result byte i takes nibble i of
// s, uses its low 3 bits as the source index, and replicates that byte's sign bit instead if
// bit 3 of the nibble is set. Every selector this file builds has bit 3 clear (codes are 0..7).
// ---------------------------------------------------------------------------------------------
PXQ23_HD uint32_t pxq_mmvq_prmt(uint32_t a, uint32_t b, uint32_t s) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(a, b, s);
#else
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = (s >> (4*i)) & 0xFu;
        const uint32_t idx = sel & 7u;
        uint32_t byte = (((idx < 4u) ? a : b) >> (8*(idx & 3u))) & 0xFFu;
        if (sel & 8u) byte = (byte & 0x80u) ? 0xFFu : 0x00u;
        r |= byte << (8*i);
    }
    return r;
#endif
}

// A decoded group: 8 s8 book values IN K ORDER, packed for dp4a. .x = k 0..3, .y = k 4..7 —
// the same shape the PXQ4 gather returns, so the dot half is literally the same code.
struct pxq_mmvq_g8 { int x, y; };

// ---------------------------------------------------------------------------------------------
// the two spreads. Both are the classic bit-interleave ladder; each step is one shift plus an
// OR-AND pair that ptxas folds into a single LOP3, so a step costs two instructions.
// ---------------------------------------------------------------------------------------------
// 8 two-bit fields (elem j at bits 2j of the low 16 bits) -> 8 nibbles (elem j at bits 4j).
PXQ23_HD uint32_t pxq_mmvq_spread2(uint32_t x) {
    x = (x | (x << 8)) & 0x00FF00FFu;   // elems 0-3 -> bits 0-7, elems 4-7 -> bits 16-23
    x = (x | (x << 4)) & 0x0F0F0F0Fu;   // pairs to their own bytes
    x = (x | (x << 2)) & 0x33333333u;   // one code per nibble, nibble bits 2-3 clear
    return x;
}

// 8 one-bit fields (elem j at bit j of the low 8 bits) -> bit 2 of nibble j (bits 4j+2).
PXQ23_HD uint32_t pxq_mmvq_spread1(uint32_t x) {
    x = (x | (x << 12)) & 0x000F000Fu;              // elems 0-3 -> bits 0-3, 4-7 -> bits 16-19
    x = (x | (x <<  6)) & 0x03030303u;              // pairs to their own bytes
    x = ((x << 2) | (x << 5)) & 0x44444444u;        // one bit per nibble, at the nibble's bit 2
    return x;
}

// ---------------------------------------------------------------------------------------------
// the gathers. One PRMT per 4 codes: a PXQ2 code indexes 4 bytes held in one register, a PXQ3
// code indexes the 8 bytes of two — which is exactly PRMT's addressing range, so neither tier
// needs the two-step remix the 16-entry PXQ4 book does.
// ---------------------------------------------------------------------------------------------
PXQ23_HD pxq_mmvq_g8 pxq2_mmvq_gather8(uint32_t lo16) {
    const uint32_t n = pxq_mmvq_spread2(lo16);
    pxq_mmvq_g8 g;
    g.x = (int) pxq_mmvq_prmt(PXQ2_MMVQ_BK, PXQ2_MMVQ_BK, n);
    g.y = (int) pxq_mmvq_prmt(PXQ2_MMVQ_BK, PXQ2_MMVQ_BK, n >> 16);
    return g;
}

PXQ23_HD pxq_mmvq_g8 pxq3_mmvq_gather8(uint32_t lo16, uint32_t hi8) {
    const uint32_t n = pxq_mmvq_spread2(lo16) | pxq_mmvq_spread1(hi8);
    pxq_mmvq_g8 g;
    g.x = (int) pxq_mmvq_prmt(PXQ3_MMVQ_BKA, PXQ3_MMVQ_BKB, n);
    g.y = (int) pxq_mmvq_prmt(PXQ3_MMVQ_BKA, PXQ3_MMVQ_BKB, n >> 16);
    return g;
}

// ---------------------------------------------------------------------------------------------
// group m (k-values 8m..8m+7) out of a row's code words. m is 0..3 within a 32-element slab.
// PXQ2 word m>>1 carries groups 2h and 2h+1 in its two halves; PXQ3's low plane is addressed the
// same way and its high plane is byte m of w2 — both halves of every group are contiguous, which
// is what makes the two spreads above the whole of the decode.
// ---------------------------------------------------------------------------------------------
PXQ23_HD uint32_t pxq2_mmvq_lo16(uint32_t word, int m) { return (word >> (16*(m & 1))) & 0xFFFFu; }
PXQ23_HD uint32_t pxq3_mmvq_hi8 (uint32_t w2,   int m) { return (w2   >> ( 8*m))       & 0x00FFu; }
