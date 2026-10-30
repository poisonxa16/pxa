// pxq-dot.c — integer-domain PXQ x Q8 dot products. Contract + rationale: pxq-dot.h.
//
// TIERS COVERED: PXQ4 (252) and PXQ4-HQ (253) — the two 4-bit nibble-plane tiers. They share
// every byte of their addressing except the sub-scale granularity, so one kernel serves both.
//
// LAYOUT (ground truth: ggml/src/pxq-cpu.c's format summary, src/pxq6-quantize.inc.cpp's
// writer, ggml/include/ggml-pxq6-tables.h):
//   panel p = row >> 6, r = row & 63; panel stride = 128 + (k/32)*SLAB
//   anchor  = fp16 at panel[2*r]                      (128 B header = 64 fp16 row anchors)
//   slab kb = panel + 128 + kb*SLAB                   (SLAB 1088 core / 1152 HQ)
//   subs    : core — slab[r], lo nibble = elems 0-15, hi nibble = elems 16-31
//             HQ   — slab[2*r] lo/hi = elems 0-7 / 8-15, slab[2*r+1] lo/hi = 16-23 / 24-31
//   codes   = slab + CODE_OFF + r*16 (CODE_OFF 64 core / 128 HQ); byte b holds elem 2b in its
//             low nibble and elem 2b+1 in its high nibble
//   value   : w[j] = anchor * SUB[s4(j)] * PX16_BOOK[code(j)]      (the parity-locked contract)
//
// THE DOT. For one 32-element block, with activations x[j] = d * a[j] (a int8):
//   sum_j w[j]*x[j] = anchor*d * ( SUB[s0] * sum_{j<16} BOOK[c_j]*a[j]
//                                + SUB[s1] * sum_{j>=16} BOOK[c_j]*a[j] )
// so the book enters only inside an integer sum. Replacing BOOK by an int8 image B with
// BOOK[c] ~= B[c]*bs turns each half into an exact int32 dot, and bs comes out at the end.
//
// WHY int8 AND NOT fp16 PAIRS. int8 is what makes the lookup free: a 16-entry int8 table is
// exactly one _mm256_shuffle_epi8 operand, so 32 nibbles become 32 book values in one uop, and
// the products then run through maddubs/madd at 32 lanes per instruction. An fp16 or int16 book
// needs two shuffles plus an interleave to assemble, and halves the lanes per multiply — call
// it 2x the kernel. What it buys is bounded by the frozen PX16 book: absmax == 1.0 exactly, so
// bs = 1/127 and the worst absolute book error is 0.5/127 = 3.9e-3 in book units, i.e. 3.9e-3
// of the block's own effective scale. The Q8_0-shaped activation quantisation next to it
// already contributes 0.5/127 of the activation block's amax. The two are the same order, so
// widening the book alone would not move the result — measured in tests/test-pxq-cpu-dot.cpp,
// which reports the book-only error and the total side by side. int8 it is.
//
// The book image is derived from the LIVE table (pxa_pxq_float_tables), so a model shipped with
// a custom PXA_PXQ6_BOOK keeps working here exactly as it does in the dequant path.
//
// OVERFLOW. maddubs saturates at int16. The operands are |B| <= 127 (abs of the book image)
// and |a| <= 127, two per lane: 2*127*127 = 32258 < 32767. Exact, never saturating — this is
// why the kernel uses the sign_epi8 trick rather than a +128 bias on the book, whose products
// would reach 255*127*2 = 64770 and would need a bsum correction to undo.

#include "pxa-env-flag.h"
#include "pxq-dot.h"
#include "pxq-cpu.h"

#include "ggml-impl.h"          // GGML_COMPUTE_FP16_TO_FP32
#include "ggml-pxq6-tables.h"   // PXQ6_SLAB_BYTES / PXQ6HQ_SLAB_BYTES / PXQ6_HDR_BYTES

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define PXA_DOT_ASSERT(x) \
    do { if (!(x)) { fprintf(stderr, "PXQ CPU dot assert failed: %s at %s:%d\n", #x, __FILE__, __LINE__); abort(); } } while (0)

// ---------------------------------------------------------------------------------------------
// int8 image of the PX16 book
// ---------------------------------------------------------------------------------------------

static int8_t pxa_book_i8[16];
static float  pxa_book_i8_scale = 0.0f;   // book[c] ~= pxa_book_i8[c] * pxa_book_i8_scale

// Idempotent and deterministic, like pxa_pxq_ensure_tables(): several compute threads may run
// this at once and every one of them writes the same 16 bytes and the same float.
static void pxa_dot_ensure_book(void) {
    static volatile int done = 0;
    if (done) return;
    const float * book16 = NULL;
    pxa_pxq_float_tables(&book16, NULL, NULL);      // also forces the env-override table init
    float amax = 0.0f;
    for (int c = 0; c < 16; ++c) {
        const float a = fabsf(book16[c]);
        if (a > amax) amax = a;
    }
    // 127/amax is the finest scale that keeps every entry inside int8; with the frozen PX16
    // book (absmax == 1.0) it is exactly 127, and book[15] == 1.0 maps to exactly 127.
    const float to_i8 = amax > 0.0f ? 127.0f/amax : 0.0f;
    for (int c = 0; c < 16; ++c) {
        long v = lrintf(book16[c]*to_i8);
        if (v >  127) v =  127;
        if (v < -127) v = -127;                     // -128 would break _mm256_sign_epi8's abs
        pxa_book_i8[c] = (int8_t)v;
    }
    pxa_book_i8_scale = amax > 0.0f ? amax/127.0f : 0.0f;
    done = 1;
}

const int8_t * pxa_pxq_dot_book_i8(float * scale_out) {
    pxa_dot_ensure_book();
    if (scale_out) *scale_out = pxa_book_i8_scale;
    return pxa_book_i8;
}

// PXA_PXQ_CPU_DOT=0 sends the 4-bit tiers back to the phase-1 dequant-and-f32-dot path in
// pxq-cpu.c. Default ON. This is the A/B lever the speedup is measured with, and the escape
// hatch if a host ever disagrees with the integer arithmetic: the dequant path is the
// parity-locked contract and is always still there.
static bool pxa_dot_enabled(void) {
    static volatile int state = -1;   // benign race: every racer computes the same answer
    int v = state;
    if (v < 0) {
        const char * e = getenv("PXA_PXQ_CPU_DOT");
        v = (e && pxa_env_value_off(e)) ? 0 : 1;
        state = v;
    }
    return v != 0;
}

bool pxa_pxq_dot_supported(enum ggml_type type) {
    if (!pxa_dot_enabled()) return false;
    return type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQ4HQ;
}

bool pxa_pxq_dot_has_simd(void) {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------------------------
// activation quantisation
// ---------------------------------------------------------------------------------------------

void pxa_pxq_quantize_row_q8(const float * x, struct pxa_pxq_q8 * y, int64_t k) {
    PXA_DOT_ASSERT(k % PXA_PXQ_DOT_QK == 0);
    const int64_t nb = k/PXA_PXQ_DOT_QK;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*PXA_PXQ_DOT_QK;
        float amax = 0.0f;
        for (int j = 0; j < PXA_PXQ_DOT_QK; ++j) {
            const float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
        const float d  = amax/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        y[ib].d = d;
        for (int j = 0; j < PXA_PXQ_DOT_QK; ++j) {
            const float v = roundf(xb[j]*id);
            y[ib].qs[j] = (int8_t)(v >  127.0f ?  127.0f :
                                  (v < -127.0f ? -127.0f : v));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// per-row addressing shared by both arms
// ---------------------------------------------------------------------------------------------

struct pxa_dot_row {
    const uint8_t * panel;
    const float   * sub;
    float           anchor;
    int             r;
    int             slab_bytes;
    int             code_off;
    int64_t         kb_n;
};

static void pxa_dot_row_setup(enum ggml_type type, const void * base, int64_t row, int64_t k,
                              struct pxa_dot_row * o) {
    PXA_DOT_ASSERT(pxa_pxq_dot_supported(type));
    PXA_DOT_ASSERT(k % PXA_PXQ_DOT_QK == 0);
    const bool hq = type == GGML_TYPE_PXQ4HQ;
    const float * sub16 = NULL;
    const float * sub8  = NULL;
    pxa_pxq_float_tables(NULL, &sub16, &sub8);
    o->kb_n       = k/32;
    o->slab_bytes = hq ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES;
    o->code_off   = hq ? 128 : 64;
    o->sub        = hq ? sub8 : sub16;
    o->r          = (int)(row & 63);
    o->panel      = (const uint8_t *)base + (row >> 6)*(PXQ6_HDR_BYTES + o->kb_n*o->slab_bytes);
    o->anchor     = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t *)o->panel)[o->r]);
}

// the four per-8-element scales of one block, already carrying anchor and the activation scale.
// The core tier repeats each 16-element sub across its two halves, which is what makes one
// kernel serve both tiers.
static inline void pxa_dot_block_scales(const struct pxa_dot_row * R, const uint8_t * slab,
                                        float d, bool hq, float s[4]) {
    const float ad = R->anchor*d;
    if (hq) {
        const uint8_t b0 = slab[2*R->r], b1 = slab[2*R->r + 1];
        s[0] = ad*R->sub[b0 & 0xf];
        s[1] = ad*R->sub[b0 >>  4];
        s[2] = ad*R->sub[b1 & 0xf];
        s[3] = ad*R->sub[b1 >>  4];
    } else {
        const uint8_t b = slab[R->r];
        s[0] = s[1] = ad*R->sub[b & 0xf];
        s[2] = s[3] = ad*R->sub[b >>  4];
    }
}

// ---------------------------------------------------------------------------------------------
// scalar reference
// ---------------------------------------------------------------------------------------------

float pxa_pxq_dot_q8_ref(enum ggml_type type, const void * base, int64_t row, int64_t k,
                         const struct pxa_pxq_q8 * y) {
    pxa_dot_ensure_book();
    struct pxa_dot_row R;
    pxa_dot_row_setup(type, base, row, k, &R);
    const bool hq = type == GGML_TYPE_PXQ4HQ;

    float sumf = 0.0f;
    for (int64_t kb = 0; kb < R.kb_n; ++kb) {
        const uint8_t * slab = R.panel + PXQ6_HDR_BYTES + kb*R.slab_bytes;
        const uint8_t * q    = slab + R.code_off + R.r*16;
        const int8_t  * a    = y[kb].qs;
        float s[4];
        pxa_dot_block_scales(&R, slab, y[kb].d, hq, s);
        // one accumulator per 8 elements: the finest granularity either tier needs, and the
        // same partition the SIMD arm's int32 lanes land on.
        int32_t isum[4] = { 0, 0, 0, 0 };
        for (int b = 0; b < 16; ++b) {
            const int e0 = 2*b, e1 = 2*b + 1;
            isum[e0 >> 3] += (int32_t)pxa_book_i8[q[b] & 0xf]*(int32_t)a[e0];
            isum[e1 >> 3] += (int32_t)pxa_book_i8[q[b] >>  4]*(int32_t)a[e1];
        }
        sumf += s[0]*(float)isum[0] + s[1]*(float)isum[1]
              + s[2]*(float)isum[2] + s[3]*(float)isum[3];
    }
    return sumf*pxa_book_i8_scale;
}

// ---------------------------------------------------------------------------------------------
// AVX2
// ---------------------------------------------------------------------------------------------

#if defined(__AVX2__)

// AVX2 without FMA is a paper configuration (every shipping AVX2 part has FMA3), but the
// macros are independent, so do not assume it.
static inline __m256 pxa_dot_fma(__m256 a, __m256 b, __m256 c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

static inline float pxa_dot_hsum(__m256 x) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}

static float pxa_pxq_dot_q8_avx2(enum ggml_type type, const void * base, int64_t row, int64_t k,
                                 const struct pxa_pxq_q8 * y) {
    struct pxa_dot_row R;
    pxa_dot_row_setup(type, base, row, k, &R);
    const bool hq = type == GGML_TYPE_PXQ4HQ;

    // byte i of each 128-bit lane takes code byte i/2 of that lane's half of the code row, so
    // output byte i carries element i of the lane's 16 elements: lane 0 -> elems 0-15 from code
    // bytes 0-7, lane 1 -> elems 16-31 from code bytes 8-15.
    const __m256i idx_dup = _mm256_setr_epi8(
        0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,
        8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15);
    // element e takes the low nibble when e is even and the high nibble when e is odd; e and its
    // byte position within a lane share parity because a lane holds 16 elements.
    const __m256i odd_byte = _mm256_setr_epi8(
        0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,
        0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1);
    const __m256i m0f   = _mm256_set1_epi8(0x0f);
    const __m256i ones  = _mm256_set1_epi16(1);
    const __m128i bk128 = _mm_loadu_si128((const __m128i *)pxa_book_i8);
    const __m256i book  = _mm256_set_m128i(bk128, bk128);

    __m256 acc = _mm256_setzero_ps();
    for (int64_t kb = 0; kb < R.kb_n; ++kb) {
        const uint8_t * slab = R.panel + PXQ6_HDR_BYTES + kb*R.slab_bytes;
        float s[4];
        pxa_dot_block_scales(&R, slab, y[kb].d, hq, s);

        const __m128i qc  = _mm_loadu_si128((const __m128i *)(slab + R.code_off + R.r*16));
        const __m256i dup = _mm256_shuffle_epi8(_mm256_set_m128i(qc, qc), idx_dup);
        const __m256i lo  = _mm256_and_si256(dup, m0f);
        const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(dup, 4), m0f);
        const __m256i cod = _mm256_blendv_epi8(lo, hi, odd_byte);      // element-order codes
        const __m256i bv  = _mm256_shuffle_epi8(book, cod);            // element-order book, int8

        const __m256i av  = _mm256_loadu_si256((const __m256i *)y[kb].qs);
        // |bv| as the unsigned operand and av*sign(bv) as the signed one: exact, and bounded by
        // 2*127*127 so maddubs never saturates. bv == 0 (book[7]) zeroes both sides.
        const __m256i ax  = _mm256_sign_epi8(bv, bv);
        const __m256i sy  = _mm256_sign_epi8(av, bv);
        const __m256i p16 = _mm256_maddubs_epi16(ax, sy);              // pairs of elements
        const __m256i p32 = _mm256_madd_epi16(p16, ones);              // int32 lane i = elems 4i..4i+3

        // lane i of p32 covers elements 4i..4i+3, so lanes 0-1 / 2-3 / 4-5 / 6-7 map onto the
        // four 8-element scales.
        const __m256 sv = _mm256_setr_ps(s[0], s[0], s[1], s[1], s[2], s[2], s[3], s[3]);
        acc = pxa_dot_fma(_mm256_cvtepi32_ps(p32), sv, acc);
    }
    return pxa_dot_hsum(acc)*pxa_book_i8_scale;
}

// EIGHT ROWS AT ONCE, core tier (2026-10-19). The CPU cold path of the expert cache (Nemotron 3.5 Bolt PXQ4 on
// 2x P100: the decode waited ~1 ms per cold layer on this dot) needs more weights per cycle than the one-row kernel's ~2.4,
// which spends most of its issue slots on port 5 (lane duplication, blendv, four-scalar setr_ps per block). Eight consecutive
// rows of one 64-row panel (row0 % 8 == 0) share the activation block, the slab, 128 contiguous code bytes, 8 sub bytes and
// 8 anchors, so per block:
//   - codes: one 256-bit load = two rows; and / srli / and, then unpacklo / unpackhi put each row's elements 0-15 (ul) and
//     16-31 (uh) in element order inside each 128-bit lane: ul = [row j 0-15 | row j+1 0-15], uh = [row j 16-31 | row j+1 16-31]
//   - activation: [a 0-15 | a 0-15] and [a 16-31 | a 16-31], built once per block for all eight rows
//   - scales: the eight anchors, d and the two subs per row as vectors (anchor*d, then *sub: the very products the one-row kernel
//     forms one by one), spread per row pair with broadcasts + one blend
// EXACTNESS. Every int32 lane holds exactly the one-row kernel's integer (the four products of elements 4i..4i+3, exact), every
// lane gets the same scale value and the same FMA in the same block order; only WHERE a lane sits in a register differs (row j's
// lanes 0-3 ride accA, its lanes 4-7 accB). At the end each row's eight lanes are put back in the one-row layout and reduced by
// the same horizontal sum and the same final scale, so out[j] is bit-identical to pxa_pxq_dot_q8(row0 + j)
// (tests/test-pxq-cpu-dot.cpp checks it bit for bit).
static void pxa_pxq_dot_q8_x8_core_avx2(const void * base, int64_t row0, int64_t k, const struct pxa_pxq_q8 * y, float out[8]) {
    struct pxa_dot_row R;
    pxa_dot_row_setup(GGML_TYPE_PXQ4, base, row0, k, &R);
    const __m256  anchor = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const uint16_t *)R.panel + R.r)));

    const __m256i m0f   = _mm256_set1_epi8(0x0f);
    const __m256i ones  = _mm256_set1_epi16(1);
    const __m128i bk128 = _mm_loadu_si128((const __m128i *)pxa_book_i8);
    const __m256i book  = _mm256_set_m128i(bk128, bk128);
    const __m256  sub_a = _mm256_loadu_ps(R.sub), sub_b = _mm256_loadu_ps(R.sub + 8);   // the 16-entry sub table

    __m256 accA[4], accB[4];   // pair p = rows 2p, 2p+1: accA lanes 0-3 / 4-7 = row 2p / 2p+1 elems 0-15; accB the same for 16-31
    for (int p = 0; p < 4; ++p) { accA[p] = _mm256_setzero_ps(); accB[p] = _mm256_setzero_ps(); }
    for (int64_t kb = 0; kb < R.kb_n; ++kb) {
        const uint8_t * slab = R.panel + PXQ6_HDR_BYTES + kb*R.slab_bytes;
        const uint8_t * qrow = slab + R.code_off + R.r*16;
        const __m256i a32 = _mm256_loadu_si256((const __m256i *)y[kb].qs);
        const __m256i avA = _mm256_permute2x128_si256(a32, a32, 0x00);     // [a 0-15 | a 0-15]
        const __m256i avB = _mm256_permute2x128_si256(a32, a32, 0x11);     // [a 16-31 | a 16-31]

        // scales of the eight rows: ad = anchor*d, slo = ad*sub[lo nibble] (elems 0-15), shi = ad*sub[hi nibble] (16-31)
        const __m256i sb  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(slab + R.r)));
        const __m256i ilo = _mm256_and_si256(sb, _mm256_set1_epi32(0xf));
        const __m256i ihi = _mm256_srli_epi32(sb, 4);
        // sub[idx] for idx 0..15: entries 0-7 from sub_a, 8-15 from sub_b, picked by bit 3 of idx (moved to the sign bit)
        const __m256  vlo = _mm256_blendv_ps(_mm256_permutevar8x32_ps(sub_a, ilo), _mm256_permutevar8x32_ps(sub_b, ilo),
                                             _mm256_castsi256_ps(_mm256_slli_epi32(ilo, 28)));
        const __m256  vhi = _mm256_blendv_ps(_mm256_permutevar8x32_ps(sub_a, ihi), _mm256_permutevar8x32_ps(sub_b, ihi),
                                             _mm256_castsi256_ps(_mm256_slli_epi32(ihi, 28)));
        const __m256  ad  = _mm256_mul_ps(anchor, _mm256_set1_ps(y[kb].d));
        float slo[8], shi[8];
        _mm256_storeu_ps(slo, _mm256_mul_ps(ad, vlo));
        _mm256_storeu_ps(shi, _mm256_mul_ps(ad, vhi));

        for (int p = 0; p < 4; ++p) {
            const __m256i q  = _mm256_loadu_si256((const __m256i *)(qrow + 32*p));
            const __m256i lo = _mm256_and_si256(q, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m0f);
            const __m256i ul = _mm256_unpacklo_epi8(lo, hi);
            const __m256i uh = _mm256_unpackhi_epi8(lo, hi);

            const __m256i bvA = _mm256_shuffle_epi8(book, ul);
            const __m256i bvB = _mm256_shuffle_epi8(book, uh);
            const __m256i pA  = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(bvA, bvA), _mm256_sign_epi8(avA, bvA)), ones);
            const __m256i pB  = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(bvB, bvB), _mm256_sign_epi8(avB, bvB)), ones);

            const __m256 svA = _mm256_blend_ps(_mm256_broadcast_ss(&slo[2*p]), _mm256_broadcast_ss(&slo[2*p + 1]), 0xF0);
            const __m256 svB = _mm256_blend_ps(_mm256_broadcast_ss(&shi[2*p]), _mm256_broadcast_ss(&shi[2*p + 1]), 0xF0);
            accA[p] = pxa_dot_fma(_mm256_cvtepi32_ps(pA), svA, accA[p]);
            accB[p] = pxa_dot_fma(_mm256_cvtepi32_ps(pB), svB, accB[p]);
        }
    }
    for (int p = 0; p < 4; ++p) {
        // the one-row layout: lanes 0-3 = elems 0-15 (accA), lanes 4-7 = elems 16-31 (accB)
        out[2*p]     = pxa_dot_hsum(_mm256_permute2f128_ps(accA[p], accB[p], 0x20))*pxa_book_i8_scale;
        out[2*p + 1] = pxa_dot_hsum(_mm256_permute2f128_ps(accA[p], accB[p], 0x31))*pxa_book_i8_scale;
    }
}

#endif // __AVX2__

float pxa_pxq_dot_q8(enum ggml_type type, const void * base, int64_t row, int64_t k,
                     const struct pxa_pxq_q8 * y) {
#if defined(__AVX2__)
    pxa_dot_ensure_book();
    return pxa_pxq_dot_q8_avx2(type, base, row, k, y);
#else
    return pxa_pxq_dot_q8_ref(type, base, row, k, y);
#endif
}

// PXA_PXQ_DOT_X8=0: the callers keep one row per call (the off switch; the outputs are the same bits either way)
bool pxa_pxq_dot_x8_on(enum ggml_type type) {
#if defined(__AVX2__) && defined(__F16C__)
    if (type != GGML_TYPE_PXQ4 || !pxa_dot_enabled()) return false;
    static volatile int state = -1;   // benign race: every racer computes the same answer
    int v = state;
    if (v < 0) {
        const char * e = getenv("PXA_PXQ_DOT_X8");
        v = (e && *e && pxa_env_value_off(e)) ? 0 : 1;
        state = v;
    }
    return v != 0;
#else
    (void) type;
    return false;
#endif
}

void pxa_pxq_dot_q8_x8(enum ggml_type type, const void * base, int64_t row0, int64_t k,
                       const struct pxa_pxq_q8 * y, float out[8]) {
#if defined(__AVX2__) && defined(__F16C__)
    if (type == GGML_TYPE_PXQ4 && (row0 & 7) == 0) {
        pxa_dot_ensure_book();
        pxa_pxq_dot_q8_x8_core_avx2(base, row0, k, y, out);
        return;
    }
#endif
    for (int j = 0; j < 8; ++j) out[j] = pxa_pxq_dot_q8(type, base, row0 + j, k, y);
}
