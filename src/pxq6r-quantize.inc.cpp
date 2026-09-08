// pxq6r-quantize.inc.cpp — PXQ6R "real PXQ6" native quantizer (5-bit LM32 codes, nibble-plane
// + hi-bit-plane packed, x E16-row scales; spec PXQ6-REAL v1.0-FINAL 2026-07-21). Cloned from
// the pxq3-quantize.inc.cpp pattern (book size, mids count and emission differ from the frozen
// pxq6-quantize.inc.cpp — that file is NOT perturbed).
//
// SELF-CONTAINED functions; spliced into src/llama-quantize.cpp next to the PXQ3 block, and
// compiled standalone by the golden tests (tests/test-pxq-cpu-dequant.cpp) and the parity
// harness.
//
// FORMAT (frozen — §2 of the spec):
//   values : 5-bit codes into the frozen LM32 book (32 entries, book[16] == +0.0f, sorted asc)
//   scales : per-ROW fp16 anchor (128 B header per 64-row panel) x one 4-bit sub-scale per
//            16-elem block through the frozen PXQ6 SUB16 table (= PXQ6 core; scale SoA 64 B)
//   dequant: eff = fp32(anchor) * SUB16[s4]; w = eff * fp32(book[c])   (fp32 muls, parity-locked)
//   layout : slab = 64 B scale SoA + 64 x 20 B code rows; a code row = Option-A packing:
//              bytes 0..15 — nibble plane, byte-identical to the PXQ6 core rows:
//                byte b = lo4(c[2b]) | lo4(c[2b+1]) << 4          (b = 0..15)
//              bytes 16..19 — one LE u32 hi-bit plane: bit j = bit 4 of c[j] (j = 0..31)
//            panel = 128 B anchor header + kslabs x 1344 B slabs; panels row-major; experts outer.
//            Every code row starts at 64 + 20r (4-byte aligned for every r; NOT 8/16-aligned
//            for odd r — consumers must use scalar u32 loads only).
//
// QUANTIZE ALGORITHM — identical to PXQ6/PXQ2/PXQ3 (E16 scheme, 32-entry book):
//   anchor = fp16_rn(row absmax) (65504 clamp); FULL 16-cand sub search per 16-elem block;
//   RTN codes against the sorted book by the midpoint rule (31 mids, linear searchsorted-left
//   = the DECIDED reference semantics); double-accum imatrix-weighted SSE argmin; sub ties
//   resolve via the shared deterministic tie-break for reproducible quantization.
//   zero 16-blocks: s4=0 / codes=PXQ6R_ZIDX (book zero) -> recon exactly 0; zero rows
//   (absmax==0): anchor fp16 +0, whole row s4=0/codes=PXQ6R_ZIDX -> exact 0 on every path.
//
// GATES: Q-G1 byte parity vs the independent spec-derived Python reference; Q-G2' wrel
// agreement (C vs spec model); staged provenance wrel 0.034301 @ 5.2656 bpw (gate Q-G2b).
//
// Env:
//   PXA_PXQ6R_ANCHOR_FIT=1        widened row-anchor search (default OFF = lab recipe).
//   PXA_PXQ6R_BOOK / PXA_PXQ6R_SUB comma-separated overrides (32 / 16 floats, fp16-snapped).

#include "../ggml/include/ggml-pxq6-tables.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>

static const float pxq6r_book_q_[32] = PXQ6_LM32_INIT;
static const float pxq6r_sub16_q_[16] = PXQ6_SUB16_INIT;

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

// n-wide parser (the existing pxq6_parse16 is 16-only; the LM32 book needs 32)
static inline bool pxq6r_parse_n(const char * e, float * out, int want) {
    int n = 0; float v[32];
    char * dup = strdup(e);
    for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
    free(dup);
    if (n != want) return false;
    for (int i = 0; i < want; ++i) out[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[i]));  // fp16-snap
    return true;
}

static inline const float * pxq6r_book_q() {
    static float book[32];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(book, pxq6r_book_q_, sizeof(book));
        if (const char * e = getenv("PXA_PXQ6R_BOOK")) {
            if (pxq6r_parse_n(e, book, 32)) fprintf(stderr, "PXQ6 quantize: custom codebook from PXA_PXQ6R_BOOK\n");
            else fprintf(stderr, "PXA_PXQ6R_BOOK: expected 32 floats — IGNORED\n");
        }
    }
    return book;
}

static inline const float * pxq6r_sub_q() {
    static float sub[16];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(sub, pxq6r_sub16_q_, sizeof(sub));
        if (const char * e = getenv("PXA_PXQ6R_SUB")) {
            if (pxq6r_parse_n(e, sub, 16)) fprintf(stderr, "PXQ6 quantize: custom SUB16 from PXA_PXQ6R_SUB\n");
        }
    }
    return sub;
}

static inline const double * pxq6r_mids_q() {
    static double mids[31];
    static bool init = false;
    if (!init) {
        init = true;
        const float * b = pxq6r_book_q();
        for (int i = 0; i < 31; ++i) mids[i] = ((double)b[i] + (double)b[i+1]) * 0.5;
    }
    return mids;
}

static inline int pxq6r_code(double xn, const double * mids) {
    int c = 0;
    for (int i = 0; i < 31; ++i) c += xn > mids[i];   // == numpy searchsorted(mids, xn, 'left')
    return c;
}

static inline double pxq6r_block_err(const float * x, const float * w, float d,
                                     const float * book, const double * mids, uint8_t * codes) {
    const double d64 = (double)d > 1e-30 ? (double)d : 1e-30;
    double err = 0.0;
    for (int i = 0; i < 16; ++i) {
        const int c = pxq6r_code((double)x[i] / d64, mids);
        codes[i] = (uint8_t)c;
        const float rec = d * book[c];                    // fp32 product == kernel math
        const double e = (double)x[i] - (double)rec;
        err += (w ? (double)w[i] : 1.0) * e * e;
    }
    return err;
}

static inline double pxq6r_quant_subblock(const float * x, const float * w, float anchor,
                                          const float * book, const double * mids, const float * sub,
                                          uint8_t * s4_out, uint8_t * codes_out, int64_t row, int64_t blk) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f) || !(anchor > 0.f)) {
        *s4_out = 0;
        for (int i = 0; i < 16; ++i) codes_out[i] = PXQ6R_ZIDX;   // book[16] == 0 -> exact zero
        return 0.0;
    }
    double best = 1e300;
    uint8_t codes[16];
    const int j0 = pxq_sub_floor(x, 16, anchor, book, 32, sub);   // PXA_PXQ_SMIN (0 when off)
    for (int j = j0; j < 16; ++j) {
        const float d = (float)((double)anchor * (double)sub[j]);   // == fp32 anchor*sub (exact-product, single round)
        const double err = pxq6r_block_err(x, w, d, book, mids, codes);
        if (err < best || (err == best && pxq_tie_take_hi(row, blk))) {   // deterministic tie-break for reproducible quantization
            best = err;
            *s4_out = (uint8_t)j;
            memcpy(codes_out, codes, 16);
        }
    }
    return best;
}

static double pxq6r_quant_row(const float * x, const float * w, int64_t K, float anchor,
                              const float * book, const double * mids, const float * sub,
                              uint8_t * s4_flat /*K/16*/, uint8_t * codes_flat /*K*/, int64_t row) {
    double err = 0.0;
    for (int64_t b = 0; b < K/16; ++b) {
        err += pxq6r_quant_subblock(x + b*16, w ? w + b*16 : nullptr, anchor,
                                    book, mids, sub, &s4_flat[b], &codes_flat[b*16], row, b);
    }
    return err;
}

static inline bool pxq6r_anchor_fit_enabled() {
    static const bool on = [](){ const char * e = getenv("PXA_PXQ6R_ANCHOR_FIT"); return e && atoi(e) != 0; }();
    return on;
}

static float pxq6r_pick_anchor(const float * x, const float * w, int64_t K,
                               const float * book, const double * mids, const float * sub,
                               uint8_t * s4_tmp, uint8_t * codes_tmp, int64_t row) {
    float amax = 0.f;
    for (int64_t i = 0; i < K; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax > 65504.f) amax = 65504.f;
    const float a0 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
    if (!pxq6r_anchor_fit_enabled() || !(a0 > 0.f)) return a0;
    float best_a = a0; double best_e = -1.0;
    float prev[5]; int np = 0;
    for (int k = -2; k <= 2; ++k) {
        const float cand = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax * (float)exp2(k / 16.0)));
        if (!(cand > 0.f)) continue;
        bool dup = false;
        for (int i = 0; i < np; ++i) if (prev[i] == cand) { dup = true; break; }
        if (dup) continue;
        prev[np++] = cand;
        const double e = pxq6r_quant_row(x, w, K, cand, book, mids, sub, s4_tmp, codes_tmp, row);
        if (best_e < 0.0 || e < best_e) { best_e = e; best_a = cand; }
    }
    return best_a;
}

// Option-A pack/unpack for one 32-elem code row (20 B; see the header comment for the layout).
static inline void pxq6r_pack32(const uint8_t * c /*32 codes 0-31*/, uint8_t * out /*20 B*/) {
    uint32_t hi = 0;
    for (int b = 0; b < 16; ++b) {
        out[b] = (uint8_t)((c[2*b] & 0xf) | ((c[2*b+1] & 0xf) << 4));   // nibble plane == PXQ6 core
        hi |= (uint32_t)(c[2*b]   >> 4) << (2*b);                       // element 2b   -> bit 2b
        hi |= (uint32_t)(c[2*b+1] >> 4) << (2*b + 1);                   // element 2b+1 -> bit 2b+1
    }
    // LE byte emission (deterministic on any host endianness)
    for (int i = 0; i < 4; ++i) out[16 + i] = (uint8_t)(hi >> (8*i));
}
static inline void pxq6r_unpack32(const uint8_t * in /*20 B*/, uint8_t * c /*32 codes*/) {
    uint32_t hi = 0;
    for (int i = 0; i < 4; ++i) hi |= (uint32_t)in[16 + i] << (8*i);
    for (int b = 0; b < 16; ++b) {
        c[2*b]   = (uint8_t)((in[b] & 0xf) | (((hi >> (2*b    )) & 1) << 4));
        c[2*b+1] = (uint8_t)((in[b] >> 4)  | (((hi >> (2*b + 1)) & 1) << 4));
    }
}

// one [R,K] expert -> PXQ6R panels (bs16 subs, slab 1344).
static void pxq6r_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                  const float * imx /*K vals or null*/,
                                 int64_t row0 /*absolute row of src's first row*/) {
    const float  * book = pxq6r_book_q();
    const double * mids = pxq6r_mids_q();
    const float  * sub  = pxq6r_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ6R_HDR_BYTES + KB*PXQ6R_SLAB_BYTES;
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
            const float anchor = pxq6r_pick_anchor(x, wq, K, book, mids, sub, s4.data(), codes.data(), row);
            anchors[r] = ggml_fp32_to_fp16(anchor);
            pxq6r_quant_row(x, wq, K, anchor, book, mids, sub, s4.data(), codes.data(), row);
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ6R_HDR_BYTES + kb*PXQ6R_SLAB_BYTES;
                const uint8_t * s = &s4[kb*2];             // 2 subs per 32 elems: lo nibble = elems 0-15
                slab[r] = (uint8_t)(s[0] | (s[1] << 4));
                pxq6r_pack32(&codes[kb*32], slab + PXQ6R_CODE_OFF + r*20);
            }
        }
    }
}

// reference dequant (CPU) — the parity-locked contract (§2 exact inverse, fp32). This is the
// golden contract for the CPU tests AND the CUDA memcmp harness.
static void pxq6r_dequant_expert(const uint8_t * src, float * dst, int64_t R, int64_t K) {
    const float * book = pxq6r_book_q();
    const float * sub  = pxq6r_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ6R_HDR_BYTES + KB*PXQ6R_SLAB_BYTES;
    uint8_t c[32];
    for (int64_t p = 0; p < P; ++p) {
        const uint8_t * panel = src + p*panel_bytes;
        const ggml_fp16_t * anchors = (const ggml_fp16_t *)panel;
        for (int64_t r = 0; r < 64; ++r) {
            const float anchor = ggml_fp16_to_fp32(anchors[r]);
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t * slab = panel + PXQ6R_HDR_BYTES + kb*PXQ6R_SLAB_BYTES;
                float eff[2];
                eff[0] = anchor * sub[slab[r] & 0xf];      // elems 0-15
                eff[1] = anchor * sub[slab[r] >> 4];       // elems 16-31
                pxq6r_unpack32(slab + PXQ6R_CODE_OFF + r*20, c);
                float * o = dst + (p*64 + r)*K + kb*32;
                for (int j = 0; j < 32; ++j) o[j] = eff[j >> 4] * book[c[j]];
            }
        }
    }
}

// full 3D expert tensor, threaded across experts (same pattern as pxq6_quantize_tensor).
// imatrix semantics: imx_size == K*E -> per-expert columns; == K -> shared; else ignored.
static void pxq6r_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                  const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = (R/64)*(PXQ6R_HDR_BYTES + (K/32)*(int64_t)PXQ6R_SLAB_BYTES);
    auto imx_for = [&](int64_t e) -> const float * {
        if (!pxq_imx_gate(imx)) return nullptr;
        // dead-column guard (llama-quantize.cpp): an all-zero / non-finite imatrix column
        // makes every candidate score 0.0, degenerating the weighted argmin into a tie-break
        const float * w = nullptr;
        if      (imx_size == K*E) w = imx + e*K;
        else if (imx_size == K)   w = imx;
        return pxq_imatrix_column_usable(w, K) ? w : nullptr;
    };
    (void)pxq6r_book_q(); (void)pxq6r_sub_q(); (void)pxq6r_mids_q();   // init tables before threading
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
            pxq6r_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e), 0);
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
            pxq6r_quantize_expert(src + e*exp_elems + p0*64*K,
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
