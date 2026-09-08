// pxq2-quantize.inc.cpp — PXQ2 native quantizer (2-bit LM4 codes x E16-row scales;
// spec: ggml/include/ggml-pxq2-tables.h). Cloned from pxq6-quantize.inc.cpp (core tier only).
//
// SELF-CONTAINED functions; spliced into src/llama-quantize.cpp next to the PXQ6 block, and
// compiled standalone by pxa-bench/pxqu_ref.cpp (the correctness / wrel-reproduction tool).
//
// FORMAT (frozen):
//   values : 2-bit codes into the frozen LM4 book (4 entries, NO zero entry, absmax != 1)
//   scales : per-ROW fp16 anchor (64 anchors = 128 B header at the head of each 64-row panel)
//            x one 4-bit sub-scale per 16-elem block through the frozen PXQ6 SUB16 table
//            (two nibbles pack into the one scale byte per 32 elems -> slab scale SoA = 64 B,
//            IDENTICAL to PXQ6 core)
//   dequant: eff = fp32(anchor) * SUB16[s4]; w = eff * fp32(book[c])   (fp32 muls, parity-locked)
//   layout : slab = 64 B scale SoA + 64 x 8 B code rows; a code row = two LE uint32 words,
//            word h = elems 16h..16h+15, elem j at bits 2*(j&15) (4 codes/byte, low bits first)
//            panel = 128 B anchor header + kslabs x 576 B slabs; panels row-major; experts outer.
//
// QUANTIZE ALGORITHM — identical to PXQ6 (= the numpy reference quant_cands, E16 scheme):
//   per row   : anchor = fp16_rn(row absmax)  [optional weighted-MSE anchor fit, env-gated]
//   per block : FULL search over all 16 sub-levels; codes = RTN against the sorted book
//               (midpoint rule = numpy searchsorted 'left'); keep argmin sum w_i (x_i - w_hat_i)^2
//               (double accumulation, imatrix column weights when present, per-expert slices).
//   zero rows (anchor snaps to 0): s4=0 / codes=PXQ2_ZIDX -> recon exactly 0 (eff==0), matching
//   the numpy oracle bit-exactly. Zero BLOCKS inside a nonzero row: s4=0 / codes=PXQ2_ZIDX;
//   ⚠ recon = anchor*SUB16[0]*book[ZIDX] != 0 here whereas the numpy oracle forces d=0 (the LM4
//   book has no zero entry, so an exact-zero recon is unrepresentable). This is the ONLY
//   quantizer/oracle divergence; all-zero 16-blocks inside nonzero rows do not occur in real
//   bf16 expert weights (verified none on Ornith-35B), and pxa-bench/pxqu_wrel.py measures the
//   gate with the same convention on both sides.
//
// GATES (Q-G1/Q-G2, see PXQ-UNIVERSAL-2026-07-17.md B1): byte-parity vs the numpy reference;
// wrel on the frozen 36-slice rng-42 protocol reproduced the lab number (0.3020488) per
// pxa-bench/pxqu_wrel.py (±1e-4, snapped-SUB16-adjusted oracle). ⚠ That lab number belongs
// to the RETIRED absmax-0.706 book; rerunning the wrel gate against the current book needs
// the oracle refit (or PXA_PXQ2_BOOK set to the old book for a pure parity check).
//
// Env:
//   PXA_PXQ2_ANCHOR_FIT=1  widen the row-anchor search (5 fp16 candidates around absmax,
//                          weighted-MSE argmin over the full row; default OFF = lab recipe).
//   PXA_PXQ2_BOOK / PXA_PXQ2_SUB  comma-separated table overrides (4 / 16 floats,
//                          fp16-snapped; MUST match the runtime kernels' tables).

#include "../ggml/include/ggml-pxq2-tables.h"
#include "../ggml/include/ggml-pxq6-tables.h"   // PXQ6_SUB16_INIT (frozen, shared verbatim)

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>

static const float pxq2_book_q_[4]   = PXQ2_BOOK_INIT;
static const float pxq2_sub16_q_[16] = PXQ6_SUB16_INIT;

#ifndef PXQ_TIE_BREAK_DEFINED
#define PXQ_TIE_BREAK_DEFINED
// deterministic tie-break for reproducible quantization (shared by all PXQ-family quantizers;
// applies ONLY when two candidates carry bit-identical weighted error — quality-neutral)
static inline bool pxq_tie_take_hi(int64_t row, int64_t blk) {
    uint64_t v = ((uint64_t)row ^ (uint64_t)blk) ^ 0x5Aull;
    v ^= v >> 32; v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
    return (v & 1) != 0;
}
#endif

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

#ifndef PXQ2_V3_GATE_DEFINED
#define PXQ2_V3_GATE_DEFINED
// ---------------------------------------------------------------------------------------------
// PXA_PXQ2_V3=1 (2026-08-10): PXQ2 v3 model-family refit book (LM4R), default OFF
// (byte-identical when unset). PXQ2-only -- PXQ3/PXQ4/PXQ6 are untouched. Wins over
// PXA_PXQ_CEIL_V2 for PXQ2; an explicit PXA_PXQ2_BOOK override still wins over both. Files
// bake pxa.pxq2.version=3 + the v3 values in pxa.pxq2.book; decode needs PXA_PXQ2_V3=1 on the
// runtime (or PXA_PXQ2_BOOK with the exact v3 values on an older build). Self-guarded OUTSIDE
// the shared PXQ_FIX_GATES_DEFINED block: pxq6-quantize.inc.cpp's copy of that block wins the
// include order in llama-quantize.cpp, so a pxq2-specific gate cannot live inside it.
static inline bool pxq2_v3_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ2_V3");
        const bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_PXQ2_V3 ARMED: PXQ2 v3 refit book; baking pxa.pxq2.version=3\n");
        return v;
    }();
    return on;
}
#endif

// ── ceiling check (from ds4-merge-dsa) ──────────────────────────────────────────────────────
// A tier whose book x SUB16 composition cannot reach the row anchor structurally clips every
// row's peak weights — the defect that made PXQ2-as-built degenerate on DS4-Flash while
// PASSING its global-wrel gate (ceiling 0.6970: 18.8% of squared error in the top-1% weights,
// output "dekameters"). That must fail HERE, at build time, not after an 83 GB quantize.
// PXA_PXQ_CEILING_ACCEPT=1 downgrades to a warning (deliberate experiments only).
//
// Deliberately NOT inside PXQ_FIX_GATES_DEFINED: pxq6-quantize.inc.cpp is included first and
// wins that macro, so a definition placed in there from this file is silently skipped and
// pxq3's call site fails to resolve. Its own guard makes include order irrelevant.
#ifndef PXQ_CEILING_CHECK_DEFINED
#define PXQ_CEILING_CHECK_DEFINED
static void pxq_ceiling_check(const char * tier, const float * book, int nbook,
                              const float * sub, int nsub, const char * legacy_note = nullptr) {
    float bmax = 0.f, smax = 0.f;
    for (int i = 0; i < nbook; ++i) { float a = fabsf(book[i]); if (a > bmax) bmax = a; }
    for (int i = 0; i < nsub;  ++i) { float a = fabsf(sub[i]);  if (a > smax) smax = a; }
    const float ceiling = bmax * smax;
    if (ceiling >= 0.95f) {
        return;
    }
    if (ceiling >= 0.80f) {
        fprintf(stderr,
            "%s: WARNING: reconstruction ceiling %.4f x row absmax (max|book| %.5f x max(SUB16) %.5f "
            "< 0.95) — row-peak weights are mildly clipped; expect elevated top-1%% error\n",
            tier, ceiling, bmax, smax);
        return;
    }
    // A table we froze on purpose and still ship as the default is a documented limitation,
    // not an unvetted book. It warns and names the lever; everything else still aborts.
    if (legacy_note) {
        fprintf(stderr,
            "%s: WARNING: reconstruction ceiling %.4f x row absmax (max|book| %.5f x max(SUB16) %.5f) "
            "— row-peak weights clipped by %.0f%%. Known limitation of a deliberately retained "
            "table: %s\n",
            tier, ceiling, bmax, smax, 100.f*(1.f - ceiling), legacy_note);
        return;
    }
    fprintf(stderr,
        "%s: FATAL: reconstruction ceiling %.4f x row absmax (max|book| %.5f x max(SUB16) %.5f). "
        "The book/SUB16 composition cannot reach the row anchor: every row's peak weights would be "
        "clipped by %.0f%%. This produced a degenerate model once (PXQ2 + the absmax-0.706 book) "
        "and is refused at build time. Refit the book with absmax ~1.0 (or SUB16 for the book's "
        "range); PXA_PXQ_CEILING_ACCEPT=1 overrides deliberately.\n",
        tier, ceiling, bmax, smax, 100.f*(1.f - ceiling));
    if (!getenv("PXA_PXQ_CEILING_ACCEPT")) {
        exit(EXIT_FAILURE);
    }
}
#endif

// True when the effective book is still the frozen v1 LM4 table (pxq2_book_q_ == PXQ2_BOOK_INIT).
static inline bool pxq2_book_is_frozen_v1(const float * book) {
    for (int i = 0; i < 4; ++i) { if (book[i] != pxq2_book_q_[i]) return false; }
    return true;
}

static inline bool pxq2_parse_n(const char * e, float * out, int want) {
    int n = 0; float v[16];
    char * dup = strdup(e);
    for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
    free(dup);
    if (n != want) return false;
    for (int i = 0; i < want; ++i) out[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[i]));  // fp16-snap (spec)
    return true;
}

static inline const float * pxq2_book_q() {
    static float book[4];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(book, pxq2_book_q_, sizeof(book));
        if (pxq_ceil_v2_enabled()) {   // ceiling fix: v2 book (PXA_PXQ2_BOOK below still wins)
            static const float v2[4] = PXQ2_BOOK_V2_INIT;
            memcpy(book, v2, sizeof(book));
            fprintf(stderr, "PXQ2 quantize: v2 book (PXA_PXQ_CEIL_V2 ceiling fix)\n");
        }
        if (pxq2_v3_enabled()) {       // v3 refit book (wins over CEIL_V2 for PXQ2; PXA_PXQ2_BOOK below still wins)
            static const float v3[4] = PXQ2_BOOK_V3_INIT;
            memcpy(book, v3, sizeof(book));
            fprintf(stderr, "PXQ2 quantize: v3 refit book (PXA_PXQ2_V3)\n");
        }
        if (const char * e = getenv("PXA_PXQ2_BOOK")) {
            if (pxq2_parse_n(e, book, 4)) fprintf(stderr, "PXQ2 quantize: custom codebook from PXA_PXQ2_BOOK\n");
            else fprintf(stderr, "PXA_PXQ2_BOOK: expected 4 floats — IGNORED\n");
        }
    }
    return book;
}

static inline uint8_t pxq2_zidx_q() {
    // argmin |book| of the ACTIVE book -- the code written for exactly-zero rows/blocks.
    // Index 2 for the v1, v2 AND v3 books (so stock output stays byte-identical); derived
    // dynamically so a PXA_PXQ2_BOOK override or a future book with a different min-|v|
    // index stays correct without touching this file.
    static const uint8_t zi = [](){
        const float * b = pxq2_book_q();
        int best = 0;
        for (int i = 1; i < 4; ++i) if (fabsf(b[i]) < fabsf(b[best])) best = i;
        return (uint8_t)best;
    }();
    return zi;
}

static inline const float * pxq2_sub_q() {
    static float sub[16];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(sub, pxq2_sub16_q_, sizeof(sub));
        if (const char * e = getenv("PXA_PXQ2_SUB")) {
            if (pxq2_parse_n(e, sub, 16)) fprintf(stderr, "PXQ2 quantize: custom SUB16 from PXA_PXQ2_SUB\n");
        }
    }
    return sub;
}

static inline const double * pxq2_mids_q() {
    static double mids[3];
    static bool init = false;
    if (!init) {
        init = true;
        const float * b = pxq2_book_q();
        for (int i = 0; i < 3; ++i) mids[i] = ((double)b[i] + (double)b[i+1]) * 0.5;
    }
    return mids;
}

static inline int pxq2_code(double xn, const double * mids) {
    int c = 0;
    for (int i = 0; i < 3; ++i) c += xn > mids[i];   // == numpy searchsorted(mids, xn, 'left')
    return c;
}

// quantize one 16-elem sub-block against a fixed effective scale d (fp32), double-accum weighted err.
static inline double pxq2_block_err(const float * x, const float * w, float d,
                                    const float * book, const double * mids, uint8_t * codes) {
    const double d64 = (double)d > 1e-30 ? (double)d : 1e-30;
    double err = 0.0;
    for (int i = 0; i < 16; ++i) {
        const int c = pxq2_code((double)x[i] / d64, mids);
        codes[i] = (uint8_t)c;
        const float rec = d * book[c];                    // fp32 product == kernel math
        const double e = (double)x[i] - (double)rec;
        err += (w ? (double)w[i] : 1.0) * e * e;
    }
    return err;
}

// pick the best 4-bit sub for one 16-elem block under a fixed row anchor (FULL 16-cand search).
static inline double pxq2_quant_subblock(const float * x, const float * w, float anchor,
                                         const float * book, const double * mids, const float * sub,
                                         uint8_t * s4_out, uint8_t * codes_out, int64_t row, int64_t blk) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f) || !(anchor > 0.f)) {
        *s4_out = 0;
        const uint8_t zi = pxq2_zidx_q();   // argmin |book| of the ACTIVE book (== PXQ2_ZIDX for every shipped book)
        for (int i = 0; i < 16; ++i) codes_out[i] = zi;
        return 0.0;
    }
    double best = 1e300;
    uint8_t codes[16];
    const int j0 = pxq_sub_floor(x, 16, anchor, book, 4, sub);   // PXA_PXQ_SMIN (0 when off)
    for (int j = j0; j < 16; ++j) {
        const float d = (float)((double)anchor * (double)sub[j]);   // fp64 product, single fp32 round
        const double err = pxq2_block_err(x, w, d, book, mids, codes);
        if (err < best || (err == best && pxq_tie_take_hi(row, blk))) {   // deterministic tie-break for reproducible quantization
            best = err;
            *s4_out = (uint8_t)j;
            memcpy(codes_out, codes, 16);
        }
    }
    return best;
}

static double pxq2_quant_row(const float * x, const float * w, int64_t K, float anchor,
                             const float * book, const double * mids, const float * sub,
                             uint8_t * s4_flat /*K/16*/, uint8_t * codes_flat /*K*/, int64_t row) {
    double err = 0.0;
    for (int64_t b = 0; b < K/16; ++b) {
        err += pxq2_quant_subblock(x + b*16, w ? w + b*16 : nullptr, anchor,
                                   book, mids, sub, &s4_flat[b], &codes_flat[b*16], row, b);
    }
    return err;
}

static inline bool pxq2_anchor_fit_enabled() {
    static const bool on = [](){ const char * e = getenv("PXA_PXQ2_ANCHOR_FIT"); return e && atoi(e) != 0; }();
    return on;
}

static float pxq2_pick_anchor(const float * x, const float * w, int64_t K,
                              const float * book, const double * mids, const float * sub,
                              uint8_t * s4_tmp, uint8_t * codes_tmp, int64_t row) {
    float amax = 0.f;
    for (int64_t i = 0; i < K; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax > 65504.f) amax = 65504.f;                    // fp16 ceiling (hardening)
    const float a0 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
    if (!pxq2_anchor_fit_enabled() || !(a0 > 0.f)) return a0;
    float best_a = a0; double best_e = -1.0;
    float prev[5]; int np = 0;
    for (int k = -2; k <= 2; ++k) {
        const float cand = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax * (float)exp2(k / 16.0)));
        if (!(cand > 0.f)) continue;
        bool dup = false;
        for (int i = 0; i < np; ++i) if (prev[i] == cand) { dup = true; break; }
        if (dup) continue;
        prev[np++] = cand;
        const double e = pxq2_quant_row(x, w, K, cand, book, mids, sub, s4_tmp, codes_tmp, row);
        if (best_e < 0.0 || e < best_e) { best_e = e; best_a = cand; }
    }
    return best_a;
}

// one [R,K] expert -> PXQ2 panels (bs16 subs, slab 576).
static void pxq2_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                 const float * imx /*K vals or null*/,
                                 int64_t row0 /*absolute row of src's first row*/) {
    const float  * book = pxq2_book_q();
    const double * mids = pxq2_mids_q();
    const float  * sub  = pxq2_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ2_HDR_BYTES + KB*PXQ2_SLAB_BYTES;
    const int64_t nsub = K / 16;

    std::vector<uint8_t> s4(nsub), codes(K);
    std::vector<float> kqw;
    for (int64_t p = 0; p < P; ++p) {
        uint8_t * panel = dst + p*panel_bytes;
        ggml_fp16_t * anchors = (ggml_fp16_t *)panel;      // 64 x fp16 header
        for (int64_t r = 0; r < 64; ++r) {
            const float * x = src + (p*64 + r)*K;
            const int64_t row = row0 + p*64 + r;
            const float * wq = pxq_kqw_row_weights(x, imx, K, kqw);   // PXA_PXQ_KQW (== imx when off)
            const float anchor = pxq2_pick_anchor(x, wq, K, book, mids, sub, s4.data(), codes.data(), row);
            anchors[r] = ggml_fp32_to_fp16(anchor);
            pxq2_quant_row(x, wq, K, anchor, book, mids, sub, s4.data(), codes.data(), row);
            // scatter into slabs
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ2_HDR_BYTES + kb*PXQ2_SLAB_BYTES;
                const uint8_t * s = &s4[kb*2];             // 2 subs per 32 elems: lo nibble = elems 0-15
                slab[r] = (uint8_t)(s[0] | (s[1] << 4));
                uint8_t * out = slab + 64 + r*8;           // 8 code bytes / row
                const uint8_t * c = &codes[kb*32];
                for (int by = 0; by < 8; ++by) {           // 4 codes/byte, low bits first (LE words)
                    out[by] = (uint8_t)(c[4*by] | (c[4*by+1] << 2) | (c[4*by+2] << 4) | (c[4*by+3] << 6));
                }
            }
        }
    }
}

// reference dequant (CPU) — the parity-locked contract. Used by pxqu_ref + golden tests.
static void pxq2_dequant_expert(const uint8_t * src, float * dst, int64_t R, int64_t K) {
    const float * book = pxq2_book_q();
    const float * sub  = pxq2_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ2_HDR_BYTES + KB*PXQ2_SLAB_BYTES;
    for (int64_t p = 0; p < P; ++p) {
        const uint8_t * panel = src + p*panel_bytes;
        const ggml_fp16_t * anchors = (const ggml_fp16_t *)panel;
        for (int64_t r = 0; r < 64; ++r) {
            const float anchor = ggml_fp16_to_fp32(anchors[r]);
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t * slab = panel + PXQ2_HDR_BYTES + kb*PXQ2_SLAB_BYTES;
                float eff[2];
                eff[0] = anchor * sub[slab[r] & 0xf];      // elems 0-15
                eff[1] = anchor * sub[slab[r] >> 4];       // elems 16-31
                const uint8_t * q = slab + 64 + r*8;
                float * o = dst + (p*64 + r)*K + kb*32;
                for (int j = 0; j < 32; ++j) {
                    const int c = (q[j >> 2] >> (2*(j & 3))) & 3;
                    o[j] = eff[j >> 4] * book[c];
                }
            }
        }
    }
}

// full 3D expert tensor, threaded across experts (same pattern as pxq6_quantize_tensor).
// imatrix semantics: imx_size == K*E -> per-expert columns; == K -> shared; else ignored.
static void pxq2_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                 const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = (R/64)*(PXQ2_HDR_BYTES + (K/32)*(int64_t)PXQ2_SLAB_BYTES);
    auto imx_for = [&](int64_t e) -> const float * {
        if (!pxq_imx_gate(imx)) return nullptr;
        // dead-column guard (llama-quantize.cpp): an all-zero / non-finite imatrix column
        // makes every candidate score 0.0, degenerating the weighted argmin into a tie-break
        const float * w = nullptr;
        if      (imx_size == K*E) w = imx + e*K;
        else if (imx_size == K)   w = imx;
        return pxq_imatrix_column_usable(w, K) ? w : nullptr;
    };
    (void)pxq2_book_q(); (void)pxq2_sub_q(); (void)pxq2_mids_q();   // init tables before threading
    // The frozen v1 book is below the bar by construction (0.70557 x 0.98779 = 0.6970) and is
    // STILL the default, because every PXQ2 file already shipped decodes against it. That
    // clipping is documented in ggml-pxq2-tables.h and the remedy is a lever, so for that one
    // table this warns. Any OTHER sub-bar book — including a bad PXA_PXQ2_BOOK — still aborts.
    pxq_ceiling_check("PXQ2", pxq2_book_q(), 4, pxq2_sub_q(), 16,
                      pxq2_book_is_frozen_v1(pxq2_book_q())
                        ? "the frozen v1 LM4 book; set PXA_PXQ_CEIL_V2=1 (or PXA_PXQ2_V3=1) for a ceiling-correct book"
                        : nullptr);
    // P15 (2026-07-27): thread over (expert, panel-chunk) jobs, not experts alone. A DENSE
    // tensor (E == 1) previously fell into the serial branch and ran its whole R x K on ONE
    // thread (measured: 103% CPU at -t 32 quantizing the dense 27B, ~9.7 s/tensor). Panels
    // are independent and panel-major with per-panel headers, so a 64-row slice at a panel
    // boundary is self-contained. The chunk's ABSOLUTE row offset (row0) must be passed in:
    // `row` seeds pxq_tie_take_hi, so a chunk-local row would flip the deterministic
    // tie-break and make the artifact differ across thread counts.
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
            pxq2_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e), 0);
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
            pxq2_quantize_expert(src + e*exp_elems + p0*64*K,
                 dst + e*exp_bytes + p0*panel_bytes,
                 nr, K, imx_for(e), p0*64);
        }
    };
    std::vector<std::thread> th;
    const int n = (int) std::min<int64_t>(nthread, n_jobs);
    th.reserve(n);
    for (int i = 0; i < n; ++i) th.emplace_back(compute);
    for (auto & t : th) t.join();
}
