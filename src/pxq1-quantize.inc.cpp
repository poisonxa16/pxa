// pxq1-quantize.inc.cpp — PXQ1 native quantizer (1-bit sign codes x E16-row scales).
// Cloned from pxq2-quantize.inc.cpp; the only differences are book size (2 vs 4), the
// 1-bit code packing (8/byte -> 4 B/row-block), and mids = {0}. Self-contained; spliced
// into src/llama-quantize.cpp next to the PXQ2/PXQ3 block, and compiled standalone by
// pxa-bench/pxq1_ref.cpp for the byte-parity gate.
//
// FORMAT: values 1-bit sign into the frozen 2-level book {-1,+1}; scales = per-ROW fp16
// anchor (128 B header/64-row panel) x one 4-bit SUB16 sub-scale per 16-elem block (two
// nibbles per scale byte, 64 B scale SoA per slab). dequant: eff = fp32(anchor)*SUB16[s4];
// w = eff * book[sign]  (fp32 muls, parity-locked). Code row = one LE uint32 (4 B), bit j
// = elem j sign. Zero blocks: s4=0 / code=PXQ1_ZIDX (recon != 0; symmetric book has no zero).

#include "../ggml/include/ggml-pxq1-tables.h"
#include "../ggml/include/ggml-pxq6-tables.h"   // PXQ6_SUB16_INIT (frozen, shared verbatim)

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>

static const float pxq1_book_q_[2]   = PXQ1_BOOK_INIT;
static const float pxq1_sub16_q_[16] = PXQ6_SUB16_INIT;

#ifndef PXQ_FIX_GATES_DEFINED
#define PXQ_FIX_GATES_DEFINED
// ---------------------------------------------------------------------------------------------
// PXQ codec-fix gates (2026-08-09). ALL default OFF: with none of these env vars set every
// quantizer below is byte-identical to the release tree. Shared by the whole PXQ family
// (guarded like PXQ_TIE_BREAK_DEFINED so any single .inc still compiles standalone).
//   PXA_PXQ_CEIL_V2=1  PXQ2/PXQ3 ceiling fix: quantize against the v2 books (the frozen LM4/LM8
//                      rescaled so max|book| == 1.0), restoring the representable ceiling to
//                      anchor*0.987793 = exact PXQ4/PXQ6 parity (v1: 0.697/0.896 * anchor).
//                      Files bake pxa.pxq{2,3}.version=2 + the v2 values in pxa.pxq{2,3}.book;
//                      decode needs the same env on the runtime (or PXA_PXQ2_BOOK/PXA_PXQ3_BOOK
//                      on an older build -- the books are runtime-uploaded tables everywhere).
//   PXA_PXQ_SMIN=1     hard representability floor in the sub-scale search: sub candidates that
//                      cannot represent the block peak within half a top book step are excluded
//                      (search [s_min,15] instead of [0,15]).
//   PXA_PXQ_KQW        k-quant error weighting on the IMATRIX PATH ONLY (DEFAULT ON since
//                      2026-08-24): w_i = imx_i * sqrt(mean(x_row^2) + x_i^2). Set =0 for the
//                      legacy raw-imx weighting (measured ppl regression; A/B use only).
//                      No imatrix => w = 1 exactly, so imatrix-free artifacts do not move.
// ---------------------------------------------------------------------------------------------
static inline bool pxq_fix_env_on(const char * name) {
    const char * e = getenv(name);
    return e && atoi(e) != 0;
}
static inline bool pxq_ceil_v2_enabled() {
    static const bool on = [](){
        const bool v = pxq_fix_env_on("PXA_PXQ_CEIL_V2");
        if (v) fprintf(stderr, "PXA_PXQ_CEIL_V2 ARMED: PXQ2/PXQ3 v2 books (max|book|=1 ceiling restore); baking pxa.pxq{2,3}.version=2\n");
        return v;
    }();
    return on;
}
static inline bool pxq_smin_enabled() {
    static const bool on = [](){
        const bool v = pxq_fix_env_on("PXA_PXQ_SMIN");
        if (v) fprintf(stderr, "PXA_PXQ_SMIN ARMED: representability floor in the sub-scale search\n");
        return v;
    }();
    return on;
}
// PXA_PXQ_IMX (DEFAULT OFF, 2026-08-24): whether PXQ tiers consume an offered imatrix AT ALL.
// Measured on Ornith-9B (580 chunks, wiki.test, clean A/B, P100, cross-card-verified): EVERY
// consumption path is net-negative for PXQ4 —
//   no-imatrix 8.3076 | kqw+SMIN 8.3317 | kqw 8.3420 | kqw+anchor-fit 8.3491 | raw 8.3542
// (and the MXFP4-backbone exponent-search path carries ~0.019 of that damage on its own),
// while the SAME imatrix file improves Q4_K_M by -0.0766. The PXQ lattice (fixed absmax
// anchor, 16-step sub-scale) cannot express per-column importance without clamping
// cold-column outliers. Until a codec-level design exists, an offered imatrix is IGNORED
// for PXQ tiers (loud notice, once) so --imatrix cannot silently degrade a PXQ artifact;
// PXA_PXQ_IMX=1 re-enables consumption (KQW form by default; PXA_PXQ_KQW=0 for raw).
static inline bool pxq_imx_optin_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ_IMX");
        return e && atoi(e) != 0;
    }();
    return on;
}
static inline const float * pxq_imx_gate(const float * imx) {
    if (!imx || pxq_imx_optin_enabled()) return imx;
    static const bool warned = [](){
        fprintf(stderr, "PXQ tiers: imatrix IGNORED (measured net-negative on the PXQ lattice; PXA_PXQ_IMX=1 to consume it — see pxq6-quantize.inc.cpp)\n");
        return true;
    }();
    (void)warned;
    return nullptr;
}
static inline bool pxq_kqw_enabled() {
    // DEFAULT ON since 2026-08-24. The prior default (raw imatrix values as MSE weights) was
    // MEASURED to make PXQ4 WORSE than no imatrix at all (Ornith-9B, 580 chunks, clean A/B:
    // no-imx 8.3076 vs raw-imx 8.3542), while the SAME imatrix file improves Q4_K_M by -0.0766
    // under stock llama.cpp weighting w = imx * sqrt(sigma2_row + x^2). Mechanism: raw imatrix
    // columns span orders of magnitude, so inside a 16/8-wide sub-block the weighted argmin fits
    // only the hottest columns and the coarse 16-candidate sub-scale then clamps the block's
    // large-|x| entries on cold columns; the sqrt(sigma2+x^2) factor keeps per-element magnitude
    // in the objective so the block scale still respects outliers. PXA_PXQ_KQW=0 restores the
    // raw-imx weighting for A/B archaeology. No-imatrix artifacts are unaffected either way
    // (w = 1 exactly on that path).
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ_KQW");
        const bool v = !e || atoi(e) != 0;
        if (v) fprintf(stderr, "PXQ imatrix weighting: k-quant form w = imx * sqrt(sigma2_row + x^2) (default; PXA_PXQ_KQW=0 for legacy raw-imx)\n");
        else   fprintf(stderr, "PXA_PXQ_KQW=0: LEGACY raw-imatrix MSE weights (measured ppl REGRESSION vs no-imatrix on PXQ4; A/B use only)\n");
        return v;
    }();
    return on;
}
// PXA_PXQ_SMIN: smallest sub index whose ceiling covers the block peak within half a top step,
// sign-aware (the peak's own side of the book sets the cap and the top gap):
//     anchor*sub[j]*(bcap + gap/2) >= |peak|
// Returns 0 when the gate is off (search unchanged). If even sub[15] cannot satisfy it, the
// floor clamps to 15 (the best available ceiling). Degenerates gracefully for the sign book
// (nbook==2: bcap=1, gap=2 -> d >= |peak|/2, the natural sign-quantizer bound).
static inline int pxq_sub_floor(const float * x, int bs, float anchor,
                                const float * book, int nbook, const float * sub) {
    if (!pxq_smin_enabled() || !(anchor > 0.f)) return 0;
    float apk = 0.f, xpk = 0.f;
    for (int i = 0; i < bs; ++i) { const float a = fabsf(x[i]); if (a > apk) { apk = a; xpk = x[i]; } }
    if (!(apk > 0.f)) return 0;
    const double bcap = xpk >= 0.f ?  (double)book[nbook-1] : -(double)book[0];
    const double gap  = xpk >= 0.f ?  (double)book[nbook-1] - (double)book[nbook-2]
                                   :  (double)book[1]       - (double)book[0];
    if (!(bcap > 0.0)) return 0;   // degenerate book side: keep the full search
    const double need = (double)apk / ((double)anchor * (bcap + 0.5*gap));
    int j = 0;
    while (j < 15 && (double)sub[j] < need) ++j;
    return j;
}
// PXA_PXQ_KQW: build the k-quant row weights into buf and return them. Returns imx (or null)
// unchanged when the gate is off or there is no imatrix -- the unweighted path NEVER moves.
static inline const float * pxq_kqw_row_weights(const float * x, const float * imx, int64_t K,
                                                std::vector<float> & buf) {
    if (!imx || !pxq_kqw_enabled()) return imx;
    double s2 = 0.0;
    for (int64_t i = 0; i < K; ++i) s2 += (double)x[i]*(double)x[i];
    s2 /= (double)K;
    buf.resize((size_t)K);
    for (int64_t i = 0; i < K; ++i) {
        buf[(size_t)i] = (float)((double)imx[i] * sqrt(s2 + (double)x[i]*(double)x[i]));
    }
    return buf.data();
}
#endif

static inline const float * pxq1_book_q() { return pxq1_book_q_; }
static inline const float * pxq1_sub_q()  { return pxq1_sub16_q_; }

// one 16-elem sub-block against a fixed effective scale d (fp32), double-accum weighted err.
static inline double pxq1_block_err(const float * x, const float * w, float d,
                                    const float * book, uint8_t * codes) {
    const double d64 = (double)d > 1e-30 ? (double)d : 1e-30;
    double err = 0.0;
    for (int i = 0; i < 16; ++i) {
        const int c = ((double)x[i] / d64) > 0.0 ? 1 : 0;   // sign -> book index (mids = {0})
        codes[i] = (uint8_t)c;
        const float rec = d * book[c];                       // fp32 product == kernel math
        const double e = (double)x[i] - (double)rec;
        err += (w ? (double)w[i] : 1.0) * e * e;
    }
    return err;
}

// pick the best 4-bit sub for one 16-elem block under a fixed row anchor (FULL 16-cand search).
static inline double pxq1_quant_subblock(const float * x, const float * w, float anchor,
                                         const float * book, const float * sub,
                                         uint8_t * s4_out, uint8_t * codes_out) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f) || !(anchor > 0.f)) {
        *s4_out = 0;
        for (int i = 0; i < 16; ++i) codes_out[i] = PXQ1_ZIDX;
        return 0.0;
    }
    double best = 1e300;
    uint8_t codes[16];
    const int j0 = pxq_sub_floor(x, 16, anchor, book, 2, sub);   // PXA_PXQ_SMIN (0 when off)
    for (int j = j0; j < 16; ++j) {
        const float d = (float)((double)anchor * (double)sub[j]);   // fp64 product, single fp32 round
        const double err = pxq1_block_err(x, w, d, book, codes);
        if (err < best) {
            best = err;
            *s4_out = (uint8_t)j;
            memcpy(codes_out, codes, 16);
        }
    }
    return best;
}

static double pxq1_quant_row(const float * x, const float * w, int64_t K, float anchor,
                             const float * book, const float * sub,
                             uint8_t * s4_flat /*K/16*/, uint8_t * codes_flat /*K*/) {
    double err = 0.0;
    for (int64_t b = 0; b < K/16; ++b) {
        err += pxq1_quant_subblock(x + b*16, w ? w + b*16 : nullptr, anchor,
                                   book, sub, &s4_flat[b], &codes_flat[b*16]);
    }
    return err;
}

static float pxq1_pick_anchor(const float * x, int64_t K) {
    float amax = 0.f;
    for (int64_t i = 0; i < K; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax > 65504.f) amax = 65504.f;                    // fp16 ceiling (hardening)
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
}

// one [R,K] expert -> PXQ1 panels (bs16 subs, slab 320, 1-bit codes).
static void pxq1_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                 const float * imx /*K vals or null*/) {
    const float * book = pxq1_book_q();
    const float * sub  = pxq1_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ1_HDR_BYTES + KB*PXQ1_SLAB_BYTES;
    const int64_t nsub = K / 16;

    std::vector<uint8_t> s4(nsub), codes(K);
    std::vector<float> kqw;
    for (int64_t p = 0; p < P; ++p) {
        uint8_t * panel = dst + p*panel_bytes;
        ggml_fp16_t * anchors = (ggml_fp16_t *)panel;      // 64 x fp16 header
        for (int64_t r = 0; r < 64; ++r) {
            const float * x = src + (p*64 + r)*K;
            const float anchor = pxq1_pick_anchor(x, K);
            anchors[r] = ggml_fp32_to_fp16(anchor);
            const float * wq = pxq_kqw_row_weights(x, imx, K, kqw);   // PXA_PXQ_KQW (== imx when off)
            pxq1_quant_row(x, wq, K, anchor, book, sub, s4.data(), codes.data());
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ1_HDR_BYTES + kb*PXQ1_SLAB_BYTES;
                const uint8_t * s = &s4[kb*2];             // 2 subs per 32 elems: lo nibble = elems 0-15
                slab[r] = (uint8_t)(s[0] | (s[1] << 4));
                uint8_t * out = slab + 64 + r*4;           // 4 code bytes / row (8 codes/byte)
                const uint8_t * c = &codes[kb*32];
                for (int by = 0; by < 4; ++by) {
                    out[by] = (uint8_t)( c[8*by]   | (c[8*by+1] << 1) | (c[8*by+2] << 2) | (c[8*by+3] << 3)
                                       | (c[8*by+4] << 4) | (c[8*by+5] << 5) | (c[8*by+6] << 6) | (c[8*by+7] << 7));
                }
            }
        }
    }
}

// reference dequant (CPU) — the parity-locked contract. Used by pxq1_ref + golden tests.
static void pxq1_dequant_expert(const uint8_t * src, float * dst, int64_t R, int64_t K) {
    const float * book = pxq1_book_q();
    const float * sub  = pxq1_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ1_HDR_BYTES + KB*PXQ1_SLAB_BYTES;
    for (int64_t p = 0; p < P; ++p) {
        const uint8_t * panel = src + p*panel_bytes;
        const ggml_fp16_t * anchors = (const ggml_fp16_t *)panel;
        for (int64_t r = 0; r < 64; ++r) {
            const float anchor = ggml_fp16_to_fp32(anchors[r]);
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t * slab = panel + PXQ1_HDR_BYTES + kb*PXQ1_SLAB_BYTES;
                float eff[2];
                eff[0] = anchor * sub[slab[r] & 0xf];      // elems 0-15
                eff[1] = anchor * sub[slab[r] >> 4];       // elems 16-31
                const uint8_t * q = slab + 64 + r*4;
                float * o = dst + (p*64 + r)*K + kb*32;
                for (int j = 0; j < 32; ++j) {
                    const int c = (q[j >> 3] >> (j & 7)) & 1;
                    o[j] = eff[j >> 4] * book[c];
                }
            }
        }
    }
}

// full 3D expert tensor, threaded across experts.
// imatrix semantics: imx_size == K*E -> per-expert columns; == K -> shared; else ignored.
static void pxq1_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                 const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = (R/64)*(PXQ1_HDR_BYTES + (K/32)*(int64_t)PXQ1_SLAB_BYTES);
    auto imx_for = [&](int64_t e) -> const float * {
        if (!pxq_imx_gate(imx)) return nullptr;
        // dead-column guard (llama-quantize.cpp): an all-zero / non-finite imatrix column
        // makes every candidate score 0.0, degenerating the weighted argmin into a tie-break
        const float * w = nullptr;
        if      (imx_size == K*E) w = imx + e*K;
        else if (imx_size == K)   w = imx;
        return pxq_imatrix_column_usable(w, K) ? w : nullptr;
    };
    // P15 (2026-07-27): thread over (expert, panel-chunk) jobs, not experts alone. A DENSE
    // tensor (E == 1) previously fell into the serial branch and ran its whole R x K on ONE
    // thread (measured: 103% CPU at -t 32 quantizing the dense 27B, ~9.7 s/tensor). Panels
    // are independent and panel-major with per-panel headers, so a 64-row slice at a panel
    // boundary is self-contained and the output is byte-identical to the serial order by
    // construction (verified: -t 1 vs -t 32 md5-identical on a mixed dense+MoE fixture).
    const int64_t panels      = R/64;
    const int64_t panel_bytes = panels > 0 ? exp_bytes/panels : 0;
    // Job granularity: fixed CHUNK=8 leaves cores idle on wide-and-short tensors --
    // a dense R=5120 gives 80 panels = 10 jobs, so 10 of 72 cores work. Size it so there
    // are ~4 jobs per thread, clamped to [1,8] panels (a 64-row panel x K is already
    // substantial work, so CHUNK=1 costs nothing in scheduling overhead).
    const int64_t CHUNK       = std::max<int64_t>(1, std::min<int64_t>(8, (panels*E)/(4*(int64_t)std::max(1, nthread))));
    const int64_t chunks_per_e = panels > 0 ? (panels + CHUNK - 1)/CHUNK : 1;
    const int64_t n_jobs      = E*chunks_per_e;
    if (nthread <= 1 || n_jobs <= 1) {
        for (int64_t e = 0; e < E; ++e) {
            pxq1_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
        return;
    }
    std::atomic<int64_t> counter{0};
    auto compute = [&]() {
        while (true) {
            const int64_t j = counter.fetch_add(1);
            if (j >= n_jobs) break;
            const int64_t e  = j / chunks_per_e;
            const int64_t c  = j % chunks_per_e;
            const int64_t p0 = c*CHUNK;
            const int64_t nr = std::min<int64_t>(CHUNK, panels - p0)*64;
            pxq1_quantize_expert(src + e*exp_elems + p0*64*K,
                 dst + e*exp_bytes + p0*panel_bytes,
                 nr, K, imx_for(e));
        }
    };
    std::vector<std::thread> th;
    const int n = (int) std::min<int64_t>(nthread, n_jobs);
    th.reserve(n);
    for (int i = 0; i < n; ++i) th.emplace_back(compute);
    for (auto & t : th) t.join();
}
