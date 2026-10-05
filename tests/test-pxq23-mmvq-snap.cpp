// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq23-mmvq-snap.cpp -- the PXQ2 / PXQ3 weight side of the q8_1 MMVQ decode path, proved
// on a CPU.
//
// WHAT IS UNDER TEST. ggml-cuda/pxa/pxq23-mmvq.h decodes a PXQ2 or PXQ3 code group into 8 int8
// book values for dp4a: a bit-interleave ladder that turns 2-bit fields (and, for PXQ3, a
// separate 1-bit plane) into nibble selectors, then one PRMT per 4 codes against a frozen s8
// snap of the tier's book. That header compiles for the host with PRMT emulated exactly, so
// every one of its claims is checkable here, without a GPU and without a model.
//
// THE FOUR CLAIMS, and which of them is bitwise:
//
//   1. THE FROZEN s8 BOOKS ARE THE CONTRACT'S. Recompute rint(book_i * 127/absmax) from
//      PXQ2_BOOK_INIT / PXQ3_BOOK_INIT and require the frozen tables exactly. BITWISE. This is
//      the same check the runtime gate runs before it admits the path; it is here so a table
//      re-freeze fails a test rather than a perplexity run.
//
//   2. THE GROUP DECODE IS THE FORMAT. EXHAUSTIVELY and BITWISE: all 2^16 PXQ2 group patterns
//      and all 2^24 PXQ3 (low-plane, high-plane) pairs, each decoded and compared value by value
//      against the format spec read straight out of the table headers. There is no sampling and
//      no epsilon here -- if any bit pattern in either format decodes to a different book entry
//      than the codec says it should, this fails.
//
//   3. THE ADDRESSING IS THE ENGINE'S. On random panels, reconstruct every weight from what the
//      decoder extracted -- anchor * SUB16[nibble] * book[code], in that order -- and require
//      BIT-IDENTICAL agreement with pxa_pxq_dequant_row(), the CPU decoder the engine actually
//      ships. This is what proves the group -> (word, half, plane byte) and row -> (panel, slab,
//      scale nibble) arithmetic, not just the bit twiddling.
//
//   4. THE DOT IS ACCURATE, NOT EXACT. BOUNDED, and it must be: the weights are snapped to int8
//      and the activations are quantised q8_1-style, so the fused fp32 path and this one cannot
//      agree bit for bit and never claimed to. The test measures the disagreement on random
//      panels and holds it to a bound derived from the contract (below), and prints it, because
//      the number is the point -- the shipping decision is a fidelity gate, and this is its
//      cheapest early warning.
//
// THE BOUND IN CLAIM 4. Per weight the snap error is at most 0.5/S of (anchor * SUB16), S being
// 127/absmax; per activation the q8_1 error is at most 0.5 of its own step. For a K-term dot of
// random signs those errors add in quadrature, so the relative error of the result scales as
// ~1/sqrt(K) against the fp32 dot's own magnitude. The test asserts a fixed, generous ceiling
// rather than that model, and reports the observed maximum so a regression shows up as a moved
// number even when it stays under the ceiling.
//
// Nonzero exit on any failure. No CUDA, no device, no model, no network.

#include "ggml.h"
#include "pxq-cpu.h"
#include "ggml-cuda/pxa/pxq23-mmvq.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void fail(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  FAIL: ");
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    if (++g_fail > 20) {
        printf("  (more than 20 failures; stopping)\n");
        exit(1);
    }
}

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// fp16 -> fp32, spelled out. ggml_fp16_to_fp32() goes through a lookup table that only exists
// after ggml_init(), and this test has no context; the conversion is exact in every case, so a
// direct one agrees with the engine's bit for bit.
static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                   // subnormal: renormalise into fp32's range
            uint32_t m = man;
            int e = -1;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float anchor_of(const std::vector<uint8_t> & W, size_t pstride, int row) {
    uint16_t h;
    memcpy(&h, &W[(size_t) (row >> 6)*pstride + 2*(row & 63)], sizeof(h));
    return f16_to_f32(h);
}

// ---------------------------------------------------------------------------------------------
// claim 1 -- the frozen s8 books are exactly what the contract derives from the float books
// ---------------------------------------------------------------------------------------------
static void check_snap_tables() {
    printf("[1] frozen s8 books vs rint(book * 127/absmax)\n");
    const float  b2[4] = PXQ2_BOOK_INIT,    b3[8] = PXQ3_BOOK_INIT;
    const int8_t q2[4] = PXQ2_MMVQ_S8_INIT, q3[8] = PXQ3_MMVQ_S8_INIT;

    float a2 = 0.0f, a3 = 0.0f;
    for (int i = 0; i < 4; ++i) a2 = fabsf(b2[i]) > a2 ? fabsf(b2[i]) : a2;
    for (int i = 0; i < 8; ++i) a3 = fabsf(b3[i]) > a3 ? fabsf(b3[i]) : a3;
    if (a2 != PXQ2_MMVQ_ABSMAX) fail("PXQ2 absmax %.10g != PXQ2_MMVQ_ABSMAX %.10g", a2, PXQ2_MMVQ_ABSMAX);
    if (a3 != PXQ3_MMVQ_ABSMAX) fail("PXQ3 absmax %.10g != PXQ3_MMVQ_ABSMAX %.10g", a3, PXQ3_MMVQ_ABSMAX);

    double w2 = 0.0, w3 = 0.0;
    for (int i = 0; i < 4; ++i) {
        const float  x = b2[i] * (127.0f/a2);
        const int    q = (int) rintf(x);
        if (q != (int) q2[i]) fail("PXQ2 book[%d]: contract %d, frozen %d", i, q, (int) q2[i]);
        if (abs(q) > 127)     fail("PXQ2 book[%d]: |%d| > 127", i, q);
        w2 = fmax(w2, fabs((double) x - q));
    }
    for (int i = 0; i < 8; ++i) {
        const float  x = b3[i] * (127.0f/a3);
        const int    q = (int) rintf(x);
        if (q != (int) q3[i]) fail("PXQ3 book[%d]: contract %d, frozen %d", i, q, (int) q3[i]);
        if (abs(q) > 127)     fail("PXQ3 book[%d]: |%d| > 127", i, q);
        w3 = fmax(w3, fabs((double) x - q));
    }
    for (int i = 0; i + 1 < 4; ++i) if (q2[i] >= q2[i+1]) fail("PXQ2 s8 book not strictly ascending at %d", i);
    for (int i = 0; i + 1 < 8; ++i) if (q3[i] >= q3[i+1]) fail("PXQ3 s8 book not strictly ascending at %d", i);
    if (!(q2[0] < 0 && q2[3] > 0)) fail("PXQ2 s8 book does not straddle zero");
    if (!(q3[0] < 0 && q3[7] > 0)) fail("PXQ3 s8 book does not straddle zero");

    printf("      PXQ2 absmax %.11g s8 {%d,%d,%d,%d} worst snap %.4f LSB = %.4f%% of absmax\n",
           a2, q2[0], q2[1], q2[2], q2[3], w2, 100.0*w2/127.0);
    printf("      PXQ3 absmax %.11g s8 {%d,%d,%d,%d,%d,%d,%d,%d} worst snap %.4f LSB = %.4f%% of absmax\n",
           a3, q3[0], q3[1], q3[2], q3[3], q3[4], q3[5], q3[6], q3[7], w3, 100.0*w3/127.0);

    // For scale: the same snap, applied to the PX16 book of the tier that has been on this path
    // since it existed. All three spend the full int8 range, so all three pay about half an LSB.
    const float * book16 = nullptr; const float * sub16 = nullptr; const float * sub8 = nullptr;
    pxa_pxq_float_tables(&book16, &sub16, &sub8);
    (void) sub16; (void) sub8;
    double a16 = 0.0, w16 = 0.0;
    for (int i = 0; i < 16; ++i) a16 = fmax(a16, fabs((double) book16[i]));
    for (int i = 0; i < 16; ++i) {
        const double x = book16[i] * (127.0/a16);
        w16 = fmax(w16, fabs(x - rint(x)));
    }
    printf("      (for scale) PXQ4 PX16 book absmax %.11g worst snap %.4f LSB = %.4f%% of absmax\n",
           a16, w16, 100.0*w16/127.0);
}

// ---------------------------------------------------------------------------------------------
// claim 2 -- the group decode is the format, for every bit pattern the format admits
// ---------------------------------------------------------------------------------------------
static inline int g8_byte(const pxq_mmvq_g8 & g, int j) {
    const uint32_t w = (j < 4) ? (uint32_t) g.x : (uint32_t) g.y;
    return (int8_t) ((w >> (8*(j & 3))) & 0xFFu);
}

static void check_group_exhaustive() {
    const int8_t q2[4] = PXQ2_MMVQ_S8_INIT, q3[8] = PXQ3_MMVQ_S8_INIT;

    printf("[2a] PXQ2 group decode, all 65536 code patterns\n");
    long long n2 = 0;
    for (uint32_t lo = 0; lo < (1u << 16); ++lo) {
        const pxq_mmvq_g8 g = pxq2_mmvq_gather8(lo);
        for (int j = 0; j < 8; ++j) {
            const int c = (int) ((lo >> (2*j)) & 3u);           // the format: elem j at bits 2j
            const int got = g8_byte(g, j);
            if (got != (int) q2[c]) {
                fail("PXQ2 lo=0x%04x elem %d: code %d -> %d, book says %d", lo, j, c, got, (int) q2[c]);
            }
            ++n2;
        }
    }
    printf("      %lld values, %d mismatches\n", n2, g_fail);

    printf("[2b] PXQ3 group decode, all 16777216 (low-plane, high-plane) pairs\n");
    const int before = g_fail;
    long long n3 = 0;
    for (uint32_t lo = 0; lo < (1u << 16); ++lo) {
        for (uint32_t hi = 0; hi < (1u << 8); ++hi) {
            const pxq_mmvq_g8 g = pxq3_mmvq_gather8(lo, hi);
            for (int j = 0; j < 8; ++j) {
                // the format: low 2 bits at bits 2j of the low plane, high bit at bit j of w2
                const int c = (int) (((lo >> (2*j)) & 3u) | (((hi >> j) & 1u) << 2));
                if (g8_byte(g, j) != (int) q3[c]) {
                    fail("PXQ3 lo=0x%04x hi=0x%02x elem %d: code %d -> %d, book says %d",
                         lo, hi, j, c, g8_byte(g, j), (int) q3[c]);
                }
                ++n3;
            }
        }
    }
    printf("      %lld values, %d mismatches\n", n3, g_fail - before);
}

// ---------------------------------------------------------------------------------------------
// a synthetic panel buffer in the shipped layout, and the group addressing under test
// ---------------------------------------------------------------------------------------------
struct panels {
    int rows, K, kslabs, np, slab_bytes, code_bytes;
    size_t pstride;
    std::vector<uint8_t> W;
};

static void make_panels(panels & P, int slab_bytes, int code_bytes, int rows, int K, uint64_t & seed) {
    P.rows = rows; P.K = K; P.kslabs = K/32; P.np = rows/64;
    P.slab_bytes = slab_bytes; P.code_bytes = code_bytes;
    P.pstride = (size_t) 128 + (size_t) P.kslabs*slab_bytes;
    P.W.assign((size_t) P.np*P.pstride, 0);
    for (size_t i = 0; i < P.W.size(); ++i) P.W[i] = (uint8_t) (rs(seed) & 0xff);
    for (int p = 0; p < P.np; ++p) {                    // sane fp16 anchors: no inf/nan/denormal
        uint16_t * an = (uint16_t *) &P.W[(size_t) p*P.pstride];
        for (int r = 0; r < 64; ++r) {
            const int e = 10 + (int) (rs(seed) % 6);    // exp 10..15 -> 2^-5..2^0
            an[r] = (uint16_t) (((rs(seed) & 1) << 15) | (e << 10) | (rs(seed) & 0x3ff));
        }
    }
}

static const uint8_t * slab_of(const panels & P, int row, int kb) {
    return &P.W[(size_t) (row >> 6)*P.pstride] + 128 + (size_t) kb*P.slab_bytes;
}

// Read one code word the way the policies do: PXQ2 word w of a row, PXQ3 word w (0,1 low, 2 high).
static uint32_t word_of(const panels & P, const uint8_t * slab, int row, int w) {
    const uint8_t * cp = slab + 64 + (size_t) (row & 63)*P.code_bytes + 4*w;
    uint32_t v = 0;
    memcpy(&v, cp, 4);                                  // the format is little-endian by contract
    return v;
}

// The 32 codes of one (row, slab), decoded exactly as the device policies decode them: four
// groups of 8, each group taking its low half-word (and, for PXQ3, its byte of the high plane).
static void decode_slab_codes(const panels & P, bool p3, int row, int kb, int * codes32) {
    const int8_t q2[4] = PXQ2_MMVQ_S8_INIT, q3[8] = PXQ3_MMVQ_S8_INIT;
    const uint8_t * slab = slab_of(P, row, kb);
    const uint32_t w2 = p3 ? word_of(P, slab, row, 2) : 0;
    for (int m = 0; m < 4; ++m) {
        const uint32_t lw   = word_of(P, slab, row, m >> 1);
        const uint32_t lo16 = pxq2_mmvq_lo16(lw, m);
        const pxq_mmvq_g8 g = p3 ? pxq3_mmvq_gather8(lo16, pxq3_mmvq_hi8(w2, m))
                                 : pxq2_mmvq_gather8(lo16);
        for (int j = 0; j < 8; ++j) {
            const int s8 = g8_byte(g, j);
            int c = -1;
            if (p3) { for (int t = 0; t < 8; ++t) if ((int) q3[t] == s8) c = t; }
            else    { for (int t = 0; t < 4; ++t) if ((int) q2[t] == s8) c = t; }
            codes32[8*m + j] = c;                       // the s8 books are injective by claim 1
        }
    }
}

// ---------------------------------------------------------------------------------------------
// claim 3 -- reconstruct from what the decoder saw and require the engine's own decoder, bitwise
// ---------------------------------------------------------------------------------------------
static void check_against_engine(bool p3, uint64_t & seed) {
    const char * name = p3 ? "PXQ3" : "PXQ2";
    printf("[3%c] %s addressing vs pxa_pxq_dequant_row(), bit for bit\n", p3 ? 'b' : 'a', name);

    const float b2[4] = PXQ2_BOOK_INIT, b3[8] = PXQ3_BOOK_INIT;
    const float * book16 = nullptr; const float * sub16 = nullptr; const float * sub8 = nullptr;
    pxa_pxq_float_tables(&book16, &sub16, &sub8);   // the SUB16 the engine will actually use
    (void) book16; (void) sub8;

    const int shapes[][2] = { {64, 64}, {128, 256}, {64, 1024}, {256, 512} };
    long long nval = 0;
    const int before = g_fail;

    for (const auto & sh : shapes) {
        panels P;
        make_panels(P, p3 ? PXQ3_SLAB_BYTES : PXQ2_SLAB_BYTES, p3 ? 12 : 8, sh[0], sh[1], seed);

        std::vector<float> ref(P.K);
        std::vector<int>   codes(32);
        for (int row = 0; row < P.rows; ++row) {
            pxa_pxq_dequant_row(p3 ? GGML_TYPE_PXQ3 : GGML_TYPE_PXQ2, P.W.data(), row, P.K, ref.data());
            const float anchor = anchor_of(P.W, P.pstride, row);
            for (int kb = 0; kb < P.kslabs; ++kb) {
                const uint8_t * slab = slab_of(P, row, kb);
                const int sb = slab[row & 63];
                const float eff[2] = { anchor * sub16[sb & 0xf], anchor * sub16[sb >> 4] };
                decode_slab_codes(P, p3, row, kb, codes.data());
                for (int j = 0; j < 32; ++j) {
                    if (codes[j] < 0) { fail("%s row %d kb %d elem %d: decoded s8 is not a book entry",
                                             name, row, kb, j); continue; }
                    const float mine = eff[j >> 4] * (p3 ? b3[codes[j]] : b2[codes[j]]);
                    const float got  = ref[kb*32 + j];
                    if (memcmp(&mine, &got, sizeof(float)) != 0) {
                        fail("%s row %d kb %d elem %d: reconstructed %.9g, engine %.9g (code %d)",
                             name, row, kb, j, mine, got, codes[j]);
                    }
                    ++nval;
                }
            }
        }
    }
    printf("      %lld weights over %zu shapes, %d mismatches\n",
           nval, sizeof(shapes)/sizeof(shapes[0]), g_fail - before);
}

// ---------------------------------------------------------------------------------------------
// claim 4 -- the dot: exact fp32 dequant path vs the snapped/q8_1 MMVQ arithmetic, bounded
// ---------------------------------------------------------------------------------------------
struct q8_1_blk { float d; int8_t q[32]; };

static void quantize_q8_1(const float * x, int K, std::vector<q8_1_blk> & out) {
    out.resize(K/32);
    for (int b = 0; b < K/32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[32*b + j]));
        const float d = amax/127.0f;
        out[b].d = d;
        for (int j = 0; j < 32; ++j) {
            out[b].q[j] = (int8_t) (d > 0.0f ? (int) rintf(x[32*b + j]/d) : 0);
        }
    }
}

static void check_dot(bool p3, uint64_t & seed) {
    const char * name = p3 ? "PXQ3" : "PXQ2";
    printf("[4%c] %s dot: exact fp32 dequant vs s8-snap + q8_1, bounded\n", p3 ? 'b' : 'a', name);

    const float b2[4] = PXQ2_BOOK_INIT, b3[8] = PXQ3_BOOK_INIT;
    const int8_t q2[4] = PXQ2_MMVQ_S8_INIT, q3[8] = PXQ3_MMVQ_S8_INIT;
    const float bfold = p3 ? PXQ3_MMVQ_BFOLD : PXQ2_MMVQ_BFOLD;
    const float * book16 = nullptr; const float * sub16 = nullptr; const float * sub8 = nullptr;
    pxa_pxq_float_tables(&book16, &sub16, &sub8);   // the SUB16 the engine will actually use
    (void) book16; (void) sub8;

    const int K = 4096, rows = 256;
    panels P;
    make_panels(P, p3 ? PXQ3_SLAB_BYTES : PXQ2_SLAB_BYTES, p3 ? 12 : 8, rows, K, seed);

    std::vector<float> x(K);
    double xn = 0.0;
    for (int i = 0; i < K; ++i) {
        const float u = (float) ((rs(seed) >> 11) * (1.0/9007199254740992.0));
        x[i] = 2.0f*u - 1.0f;
        xn  += (double) x[i]*x[i];
    }
    xn = sqrt(xn);
    std::vector<q8_1_blk> y;
    quantize_q8_1(x.data(), K, y);

    // NORMALISATION. A dot of random signs cancels, so |err|/|dot| is a lottery, not a metric:
    // it is unbounded wherever the exact dot happens to land near zero (and the ACTIVATION
    // quantisation alone scores worse on it than anything here does). The scale a dot's error is
    // actually bounded by is ||w|| * ||x||, so that is what these are relative to.
    //
    // THREE ERRORS, separated, because only the first is this path's doing:
    //   snap  = snapped weights against exact activations  -> the s8 book snap alone
    //   act   = exact weights against q8_1 activations      -> what any q8_1 GEMV pays, PXQ4
    //           and MXFP4 included, and not a property of this tier
    //   mmvq  = the kernel's own arithmetic, in its own order -> what actually ships
    std::vector<float> ref(K);
    std::vector<int>   codes(32);
    double worst_snap = 0.0, worst_act = 0.0, worst_mmvq = 0.0;
    double sq_mmvq = 0.0, sq_ref = 0.0;

    for (int row = 0; row < rows; ++row) {
        pxa_pxq_dequant_row(p3 ? GGML_TYPE_PXQ3 : GGML_TYPE_PXQ2, P.W.data(), row, K, ref.data());
        const float anchor = anchor_of(P.W, P.pstride, row);
        const float anch   = anchor * bfold;

        double dot_ref = 0.0, dot_act = 0.0, dot_snap = 0.0, dot_mmvq = 0.0, wn = 0.0;
        for (int i = 0; i < K; ++i) {
            dot_ref += (double) ref[i] * x[i];
            dot_act += (double) ref[i] * (double) y[i/32].q[i & 31] * y[i/32].d;
            wn      += (double) ref[i] * ref[i];
        }
        wn = sqrt(wn);

        for (int kb = 0; kb < K/32; ++kb) {
            const uint8_t * slab = slab_of(P, row, kb);
            const int sb = slab[row & 63];
            decode_slab_codes(P, p3, row, kb, codes.data());

            // the MMVQ arithmetic, in the kernel's own order: per group an int8 dot, scaled by
            // that group's sub-scale and summed; then the row anchor (carrying the book fold)
            // and the block's q8_1 d.
            float sum = 0.0f;
            for (int m = 0; m < 4; ++m) {
                int s = 0;
                for (int j = 0; j < 8; ++j) {
                    const int e  = 8*m + j;
                    const int qi = p3 ? (int) q3[codes[e]] : (int) q2[codes[e]];
                    s += qi * (int) y[kb].q[e];
                }
                sum += sub16[(sb >> (4*(m >> 1))) & 0xf] * (float) s;
            }
            dot_mmvq += (double) (anch * y[kb].d * sum);

            // the same snapped weights against the EXACT activations
            for (int e = 0; e < 32; ++e) {
                const int   qi = p3 ? (int) q3[codes[e]] : (int) q2[codes[e]];
                const float w  = anch * sub16[(sb >> (4*((e >> 3) >> 1))) & 0xf] * (float) qi;
                dot_snap += (double) w * x[kb*32 + e];
            }
        }

        const double nrm = fmax(wn*xn, 1e-30);
        worst_snap = fmax(worst_snap, fabs(dot_snap - dot_ref)/nrm);
        worst_act  = fmax(worst_act,  fabs(dot_act  - dot_ref)/nrm);
        worst_mmvq = fmax(worst_mmvq, fabs(dot_mmvq - dot_ref)/nrm);
        sq_mmvq += (dot_mmvq - dot_ref)*(dot_mmvq - dot_ref);
        sq_ref  += dot_ref*dot_ref;
    }

    const double rms_vs_dot = sqrt(sq_mmvq/fmax(sq_ref, 1e-30));
    printf("      K=%d rows=%d, errors relative to ||w||*||x||:  weight snap %.4f%%  "
           "activation q8_1 %.4f%%  both (the kernel) %.4f%%\n",
           K, rows, 100.0*worst_snap, 100.0*worst_act, 100.0*worst_mmvq);
    printf("      rms error against the exact dot's own rms: %.4f%%\n", 100.0*rms_vs_dot);

    // Ceilings, set well above what a correct decode produces so they catch a defect rather than
    // noise. A decode bug that still yields valid codes -- a swapped plane, the wrong half-word,
    // a mis-indexed sub-scale -- randomises the weights and lands at tens of percent here.
    if (!(worst_snap < 0.010)) fail("%s weight-snap error %.4f%% of ||w||*||x|| exceeds 1%%",  name, 100.0*worst_snap);
    if (!(worst_mmvq < 0.020)) fail("%s kernel error %.4f%% of ||w||*||x|| exceeds 2%%",       name, 100.0*worst_mmvq);
    // and the snap must not DOMINATE: the activation quantisation is the shared cost of any q8_1
    // GEMV -- PXQ4 and MXFP4 pay it too -- and both are half-LSB int8 roundings, so they land in
    // the same place. A weight decode several times the size of it would mean the snap, not the
    // shared int8 arithmetic, is what a fidelity gate is measuring.
    if (!(worst_snap < 2.0*worst_act)) {
        fail("%s weight snap (%.4f%%) dominates the activation quantisation (%.4f%%)",
             name, 100.0*worst_snap, 100.0*worst_act);
    }
}

int main() {
    printf("== PXQ2 / PXQ3 MMVQ weight decode (host)\n");
    // The frozen tables are the subject; an override would make every claim above vacuous.
    for (const char * v : { "PXA_PXQ2_BOOK", "PXA_PXQ3_BOOK", "PXA_PXQ_CEIL_V2", "PXA_PXQ2_V3",
                            "PXA_PXQ6_SUB", "PXA_PXQ2_SUB", "PXA_PXQ3_SUB" }) {
        if (getenv(v)) { printf("  %s is set; unset it -- this test is about the frozen tables\n", v); return 1; }
    }

    check_snap_tables();
    check_group_exhaustive();

    uint64_t seed = 0x9E3779B97F4A7C15ull;
    check_against_engine(false, seed);
    check_against_engine(true,  seed);
    check_dot(false, seed);
    check_dot(true,  seed);

    printf("== %s (%d failure%s)\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
