// PXA_MMID_F16_GUARD on CUDA (bug #338 nano-zc-nan): a MUL_MAT_ID whose per-expert route converts the gathered f32 rows
// to f16 (dequant + cuBLAS: F16/PXQ4 experts at > 8 rows per expert, MMQ types at >= PXA_VOLTA_CUBLAS_NE11 rows on
// Volta) must survive an activation outside the f16 range. Nemotron-3-Nano-30B PXQ4 on a V100 fed one token's relu^2
// down-projection input of 426395 into that conversion; the row became inf/NaN and every later chunk of the KLD run was NaN.
// The PXQ4 cases are that type (random panels with real-model fp16 anchors up to 2, pxq4_real_anchor; reference =
// pxa_pxq_dequant_2d; on an f16-OUTPUT route -- P100 FP16_GEMM, cc >= 7.5 -- the 65504 spike times |w| > 1 overflows the
// GEMM output, which PXA_F16_GUARD_OUT recomputes; the "big-w" F16 cases make that overflow deterministic, so =0 fails
// them on an f16-output card by design and passes them on V100); F16 and Q8_0 cover
// the same route with a host-quantised reference, and Q8_0 at T24 is the MMQ (q8_1) control the guard must not touch.
//
// For each case the same graph runs twice on the device with the same routing:
//   (1) CLEAN: every activation in [-1, 1];
//   (2) SPIKED: the same data with a few elements set far beyond 65504 (one row per spike).
// Checks:
//   - every output of both runs is finite and within tolerance of a double-precision host reference (the spiked rows
//     fail this without the guard: inf/NaN); on 32-blocks that hold a spike the q8_1 rounding an MMQ route applies
//     (d = amax/127 per element) is allowed on top, since that loss is the route's own and not an overflow;
//   - every output row that the spikes do not touch is BITWISE equal between (1) and (2): the guard must not move a row
//     that does not overflow.
// Exit 77 without a CUDA device. PXA_MMID_F16_GUARD=0 turns the guard off (the spiked rows then fail on a card whose
// route is the f16 one; a P100 takes that route only with PXA_P100_FP16_GEMM=1, else f32 cuBLAS).
// Guard on vs off, bitwise: PXA_MMID_TEST_DUMP=<file> writes every output of both passes of every case; a later run with
// PXA_MMID_TEST_REF=<that file> (made with PXA_MMID_F16_GUARD=0) checks that every output that was finite with the guard
// off has the same bits with it on -- on the q8_1/f32 routes that is every output, spiked rows included.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "pxq-cpu.h"   // pxa_pxq_dequant_2d: the host reference for the PXQ4 cases (PXQ has no type-traits CPU codec)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <vector>

struct mmid_case {
    const char * name;
    ggml_type    type;
    int64_t      K;        // row length (ne00 == ne10)
    int64_t      N;        // output rows per expert (ne01)
    int64_t      n_exp;    // experts (ne02)
    int64_t      n_used;   // experts per token (ids ne0)
    int64_t      n_tok;    // tokens (ne12)
    bool         bcast;    // b ne11 == 1 (the up-projection form) instead of n_used (the down-projection form)
    bool         bigw;     // F16 only: rows 0-7 of every expert get w = k_big_w at the 65504 spike's k (big_w_rows)
};

static const mmid_case cases[] = {
    { "f16  down-form K256 N128 E4 u2 T96",   GGML_TYPE_F16,  256, 128, 4, 2,  96, false },
    { "f16  up-form   K256 N128 E4 u2 T96",   GGML_TYPE_F16,  256, 128, 4, 2,  96, true  },
    { "f16  down-form K1856 N256 E8 u6 T64",  GGML_TYPE_F16, 1856, 256, 8, 6,  64, false },
    { "q8_0 down-form K256 N128 E4 u2 T512",  GGML_TYPE_Q8_0, 256, 128, 4, 2, 512, false },
    { "q8_0 down-form K256 N128 E4 u2 T24",   GGML_TYPE_Q8_0, 256, 128, 4, 2,  24, false },
    { "pxq4 down-form K256 N128 E4 u2 T96",   GGML_TYPE_PXQ4, 256, 128, 4, 2,  96, false },   // ~48 rows/expert
    { "pxq4 down-form K1856 N128 E8 u6 T64",  GGML_TYPE_PXQ4,1856, 128, 8, 6,  64, false },   // nano's expert K
    { "f16  down-form big-w K256 N128 E4 u2 T96", GGML_TYPE_F16, 256, 128, 4, 2, 96, false, true },   // f16-output overflow
};

// PXQ4 slab bytes of one [K x R] expert: 64-row panels of 128 anchor bytes + K/32 x 1088 (R % 64 == 0, K % 32 == 0)
static size_t pxq4_bytes(int64_t R, int64_t K) { return (size_t) (R/64)*(128 + (size_t) (K/32)*1088); }

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; }
};

struct spike { int64_t tok, slot, k; float v; };

// ---- PXA_F16_GUARD_DENSE and the fused MoE up/gate loop (code review rc22 items 2 and 3) ----
// Spikes here also cover 16- and 32-wide runs of large values (a whole q8_1/f16 block), not only single channels.
struct wspike { int64_t tok, k, w; float v; };   // channels [k, k + w) of token tok set to v

// The column scale k_pxa_mmid_f32_to_f16_guard picks: 2^(e - 12) for a finite absmax >= 65520 (frexp exponent e), else 1.
static int guard_kexp(const float * x, int64_t n) {
    float am = 0.0f;
    for (int64_t i = 0; i < n; ++i) am = std::max(am, std::fabs(x[i]));
    if (!(am >= 65520.0f) || !std::isfinite(am)) return 0;
    int e;
    std::frexp(am, &e);
    return e - 12;
}

// A PXQ4 row anchor drawn like a real model's: Nemotron-3-Nano-30B-A3B-PXQ4.gguf (127 PXQ4 tensors, 13.8M rows, scanned
// 2026-10-21) has nonzero anchors p0.01 = 0.038, p50 = 0.080, p99.9 = 0.20, max 1.75 (ffn_down_exps max 1.61), 0.13% zero
// and 18 rows below the f16 normal range; |w| <= anchor * max|SUB16| (0.988) * max|book| (1.0). Log-uniform over
// [0.03, 2) covers that whole range with the large rows over-represented (|w| > 1 with a 65504 activation overflows an
// f16-output GEMM: PXA_F16_GUARD_OUT), plus 1/16 zero rows. The older draw (random fp16 bits, exponents down to 2^-24)
// put most rows in the f16 subnormal range, where an f16-accumulating GEMM (P100 FP16_GEMM) loses the products of a
// clean column to subnormal rounding: a route-precision artefact on rows no real model has, not an overflow.
static float pxq4_real_anchor(lcg & r) {
    const float u = 0.5f*(r.unit() + 1.0f);
    const bool zero = (r.next() & 15) == 0;
    return zero ? 0.0f : 0.03f*std::exp(u*std::log(2.0f/0.03f));
}

// The f16-OUTPUT overflow, deterministically: |w| = 1.73 is the largest weight Nemotron-3-Nano PXQ4 can hold (anchor max 1.75 x
// 0.988 x 1.0). Rows 0-7 of each n_mat get it at column k of the raw (pre-quantisation) weights; the 65504 spike (no input
// guard: finite in f16) sits at that k, so those outputs are ~1.73 * 65504 = 113k: inf on an f16-output GEMM (P100 FP16_GEMM,
// cc >= 7.5) without PXA_F16_GUARD_OUT, finite on every f32-output route.
static const float k_big_w = 1.73f;
static void big_w_rows(std::vector<float> & raw, int64_t K, int64_t N, int64_t n_mat, int64_t k) {
    for (int64_t e = 0; e < n_mat; ++e) for (int64_t n = 0; n < 8 && n < N; ++n) raw[((size_t) e*N + n)*K + k] = k_big_w;
}

// Weights for n_mat matrices of N rows x K (F16 or PXQ4): device bytes + the exact f32 values the device reads.
static void make_weights(ggml_type type, int64_t K, int64_t N, int64_t n_mat, lcg & r, std::vector<char> & bytes,
                         std::vector<float> & deq, bool real_anchors = false, int64_t big_k = -1) {
    deq.assign((size_t) K*N*n_mat, 0.0f);
    if (type == GGML_TYPE_PXQ4) {
        const size_t eb = pxq4_bytes(N, K), ps = eb/(size_t) (N/64);
        bytes.resize(eb*(size_t) n_mat);
        for (auto & v : bytes) v = (char) (r.next() & 0xff);
        for (int64_t e = 0; e < n_mat; ++e) {
            for (int64_t p = 0; p < N/64; ++p) {
                uint16_t * anchors = (uint16_t *) (bytes.data() + (size_t) e*eb + (size_t) p*ps);
                for (int i = 0; i < 64; ++i) {
                    const ggml_fp16_t h = ggml_fp32_to_fp16(real_anchors ? pxq4_real_anchor(r) : 0.03f + 0.02f*(r.unit() + 1.0f));
                    memcpy(&anchors[i], &h, sizeof(h));
                }
            }
            pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, bytes.data() + (size_t) e*eb, deq.data() + (size_t) e*N*K, N, K);
        }
        return;
    }
    std::vector<float> raw(deq.size());
    for (auto & v : raw) v = 0.05f*r.unit();
    if (big_k >= 0) big_w_rows(raw, K, N, n_mat, big_k);
    const size_t row_bytes = ggml_row_size(type, K);
    bytes.resize(row_bytes*(size_t) N*n_mat);
    ggml_quantize_chunk(type, raw.data(), bytes.data(), 0, N*n_mat, K, nullptr, nullptr);
    const ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    for (int64_t row = 0; row < N*n_mat; ++row) {
        tt.to_float(bytes.data() + (size_t) row*row_bytes, deq.data() + (size_t) row*K, (int) K);
    }
}

// guard on vs off (PXA_MMID_TEST_DUMP / PXA_MMID_TEST_REF): every output finite in the =0 reference keeps its bits
static bool dump_and_ref(FILE * fdump, FILE * fref, const std::vector<float> & yc, const std::vector<float> & ys,
                         size_t & n_ref, size_t & n_ref_moved) {
    const size_t ny = yc.size();
    if (fdump) {
        fwrite(yc.data(), sizeof(float), ny, fdump);
        fwrite(ys.data(), sizeof(float), ny, fdump);
    }
    if (!fref) return true;
    std::vector<float> rr(2*ny);
    if (fread(rr.data(), sizeof(float), 2*ny, fref) != 2*ny) return false;
    for (size_t i = 0; i < 2*ny; ++i) {
        if (!std::isfinite(rr[i])) continue;
        n_ref++;
        const float got = i < ny ? yc[i] : ys[i - ny];
        if (memcmp(&got, &rr[i], sizeof(float)) != 0) n_ref_moved++;
    }
    return true;
}

// Route precision is the route's own (P100 f16 accumulation reaches a few 1e-2 of sum|w*x| on a clean column); the
// reference check here only has to catch an overflow, a missing or double scale (>= 2^5 off) or a scale applied after the
// GLU, so it allows 1e-1. The exactness of the dense guard is checked bitwise instead (pre-scaled pass below).
static const double k_tol_loose = 1e-1;

struct dense_case { const char * name; ggml_type type; int64_t K, N, n_tok; bool real_anchors; bool bigw; };
static const dense_case dense_cases[] = {
    { "f16  dense K256 N128 T32",    GGML_TYPE_F16,   256, 128,  32, false },
    { "f16  dense K1856 N256 T200",  GGML_TYPE_F16,  1856, 256, 200, false },
    { "pxq4 dense K256 N128 T32",    GGML_TYPE_PXQ4,  256, 128,  32, false },   // > 8 columns: dequant + cuBLAS f16
    { "pxq4 dense K1856 N128 T64",   GGML_TYPE_PXQ4, 1856, 128,  64, false },   // nano's down K
    { "pxq4 dense real-anchor K1856 N128 T64", GGML_TYPE_PXQ4, 1856, 128, 64, true },   // pxq4_real_anchor draw (to 2)
    { "f16  dense big-w K256 N128 T32", GGML_TYPE_F16, 256, 128, 32, false, true },   // f16-output overflow (GUARD_OUT)
    { "pxq4 dense K4224 N64 T16",    GGML_TYPE_PXQ4, 4224,  64,  16, false },   // K > 4096: the two-pass conversion kernel
};

// MUL_MAT on the cuBLAS f16 route, three passes: CLEAN, SPIKED, and PRE-SCALED (each overflowing spiked column divided on
// the host by the 2^k the guard picks, so the device takes the stock path for it). Checks: every output finite and within
// k_tol_loose of the host reference; columns without a spike bitwise equal between CLEAN and SPIKED; every SPIKED column
// bitwise equal to its PRE-SCALED column times 2^k (the guard is exact: it moves nothing but the overflow).
static int run_dense_case(ggml_backend_t cuda, const dense_case & c, FILE * fdump, FILE * fref) {
    lcg r(0xD5EEDull + (uint64_t) (c.K*131 + c.N*7 + c.n_tok));
    std::vector<char> w_bytes; std::vector<float> w_deq;
    make_weights(c.type, c.K, c.N, 1, r, w_bytes, w_deq, c.real_anchors, c.bigw ? 33 % c.K : -1);   // 65504 spike's k
    std::vector<float> x_clean((size_t) c.K*c.n_tok);
    for (auto & v : x_clean) v = r.unit();
    const wspike spikes[] = {
        { 5 % c.n_tok,           7 % c.K,    1,  300000.0f },
        { (c.n_tok/2) % c.n_tok, 100 % c.K,  1, -1.5e6f    },
        { (c.n_tok - 3) % c.n_tok, c.K - 1,  1,  65520.0f  },   // the first value f16 rounds to inf
        { (c.n_tok - 1) % c.n_tok, 33 % c.K, 1,  65504.0f  },   // f16 max: no guard needed, must stay as is
        { 1 % c.n_tok,           48,        16,  524200.0f },   // a 16-wide run
        { 3 % c.n_tok,           64,        32,  300000.0f },   // a whole 32-block
    };
    std::vector<float> x_spiked = x_clean;
    for (const wspike & s : spikes) for (int64_t j = 0; j < s.w; ++j) x_spiked[(size_t) s.tok*c.K + s.k + j] = s.v;
    std::vector<int> kexp((size_t) c.n_tok);
    std::vector<float> x_pre = x_spiked;
    for (int64_t t = 0; t < c.n_tok; ++t) {
        kexp[t] = guard_kexp(x_spiked.data() + (size_t) t*c.K, c.K);
        for (int64_t k = 0; k < c.K; ++k) x_pre[(size_t) t*c.K + k] = std::ldexp(x_spiked[(size_t) t*c.K + k], -kexp[t]);
    }

    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, c.type, c.K, c.N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.n_tok);
    ggml_tensor * y = ggml_mul_mat(ctx, w, b);
    if (ggml_nbytes(w) != w_bytes.size() || !ggml_backend_supports_op(cuda, y)) {
        printf("  %-46s FAIL (weight bytes / CUDA declines the op)\n", c.name);
        ggml_free(ctx); return 1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    if (!buf) { printf("  %-46s SKIPPED (allocation failed)\n", c.name); ggml_free(ctx); return 0; }
    ggml_backend_tensor_set(w, w_bytes.data(), 0, w_bytes.size());
    const size_t ny = (size_t) c.N*c.n_tok;
    std::vector<float> y_clean(ny), y_spiked(ny), y_pre(ny);
    const std::vector<float> * xs[3] = { &x_clean, &x_spiked, &x_pre };
    std::vector<float>       * ys[3] = { &y_clean, &y_spiked, &y_pre };
    for (int pass = 0; pass < 3; ++pass) {
        ggml_backend_tensor_set(b, xs[pass]->data(), 0, xs[pass]->size()*sizeof(float));
        if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
            printf("  %-46s FAIL (graph compute)\n", c.name);
            ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
        }
        ggml_backend_tensor_get(y, ys[pass]->data(), 0, ny*sizeof(float));
    }

    size_t n_nonfinite = 0, n_tol = 0, n_moved = 0, n_untouched = 0, n_scaled = 0, n_inexact = 0;
    double worst_rel = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<float> & x  = *xs[pass];
        const std::vector<float> & yy = *ys[pass];
        for (int64_t t = 0; t < c.n_tok; ++t) {
            bool touched = false;
            for (const wspike & s : spikes) if (s.tok == t) touched = true;
            const float * xr = x.data() + (size_t) t*c.K;
            for (int64_t n = 0; n < c.N; ++n) {
                const float * wr = w_deq.data() + (size_t) n*c.K;
                double ref = 0.0, mag = 0.0;
                for (int64_t k = 0; k < c.K; ++k) { ref += (double) wr[k]*xr[k]; mag += std::fabs((double) wr[k]*xr[k]); }
                const size_t iy = (size_t) t*c.N + n;
                const float got = yy[iy];
                if (!std::isfinite(got)) { n_nonfinite++; continue; }
                const double rel = std::fabs((double) got - ref)/(mag + 1e-6);
                worst_rel = std::max(worst_rel, rel);
                if (rel > k_tol_loose) n_tol++;
                if (pass == 1 && !touched) {
                    n_untouched++;
                    if (memcmp(&y_clean[iy], &y_spiked[iy], sizeof(float)) != 0) n_moved++;
                }
                if (pass == 1 && touched) {
                    const float want = std::ldexp(y_pre[iy], kexp[t]);
                    n_scaled++;
                    if (memcmp(&want, &got, sizeof(float)) != 0) n_inexact++;
                }
            }
        }
    }
    size_t n_ref = 0, n_ref_moved = 0;
    const bool ref_ok = dump_and_ref(fdump, fref, y_clean, y_spiked, n_ref, n_ref_moved);
    const bool pass_ok = n_nonfinite == 0 && n_tol == 0 && n_moved == 0 && n_inexact == 0 && n_ref_moved == 0 && ref_ok;
    printf("  %-46s %s  non-finite %zu, out of tolerance %zu (worst rel %.2e), untouched columns moved %zu/%zu, "
           "spiked != 2^k * pre-scaled %zu/%zu", c.name, pass_ok ? "OK  " : "FAIL", n_nonfinite, n_tol, worst_rel, n_moved,
           n_untouched, n_inexact, n_scaled);
    if (fref) printf(", vs =0 finite outputs moved %zu/%zu%s", n_ref_moved, n_ref, ref_ok ? "" : " (ref too short)");
    printf("\n");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return pass_ok ? 0 : 1;
}

struct fused_case { const char * name; int64_t K, N, n_exp, n_used, n_tok; bool merged; int64_t Kd; };
static const fused_case fused_cases[] = {
    { "f16  fused up/gate silu K256 N128 E4 u2 T96",        256, 128, 4, 2, 96, false,   0 },
    { "f16  fused merged up|gate silu K256 N128 E4 u2 T96", 256, 128, 4, 2, 96, true,    0 },
    { "f16  fused up/gate silu +down D128 E4 u2 T96",       256, 128, 4, 2, 96, false, 128 },
};

// MOE_FUSED_UP_GATE (+ the fused down MUL_MAT_ID): CLEAN and SPIKED passes; every output finite and within k_tol_loose of the
// host reference (silu(g)*u, through W_down for the +down case), rows of unspiked tokens bitwise equal between the passes.
static int run_fused_case(ggml_backend_t cuda, const fused_case & c, FILE * fdump, FILE * fref) {
    lcg r(0xF05EDull + (uint64_t) (c.K*131 + c.N*7 + c.n_tok + c.Kd + (c.merged ? 1 : 0)));
    const int64_t n_rows = c.merged ? 2*c.N : c.N;
    std::vector<char> wu_b, wg_b, wd_b; std::vector<float> wu, wg, wd;
    make_weights(GGML_TYPE_F16, c.K, n_rows, c.n_exp, r, wu_b, wu);
    if (!c.merged) make_weights(GGML_TYPE_F16, c.K, c.N, c.n_exp, r, wg_b, wg);
    if (c.Kd > 0)  make_weights(GGML_TYPE_F16, c.N, c.Kd, c.n_exp, r, wd_b, wd);
    // merged layout: rows 0..N-1 gate, N..2N-1 up
    auto w_gate = [&](int64_t e, int64_t n) { return c.merged ? wu.data() + ((size_t) e*n_rows + n)*c.K : wg.data() + ((size_t) e*c.N + n)*c.K; };
    auto w_up   = [&](int64_t e, int64_t n) { return c.merged ? wu.data() + ((size_t) e*n_rows + c.N + n)*c.K : wu.data() + ((size_t) e*c.N + n)*c.K; };
    std::vector<int32_t> ids((size_t) c.n_used*c.n_tok);
    for (int64_t t = 0; t < c.n_tok; ++t) {
        const int32_t base = (int32_t) (r.next() % (uint32_t) c.n_exp);
        for (int64_t u = 0; u < c.n_used; ++u) ids[(size_t) t*c.n_used + u] = (int32_t) ((base + u) % c.n_exp);
    }
    std::vector<float> x_clean((size_t) c.K*c.n_tok);
    for (auto & v : x_clean) v = r.unit();
    const wspike spikes[] = {
        { 5 % c.n_tok,             7 % c.K,   1,  300000.0f },
        { (c.n_tok/2) % c.n_tok,   100 % c.K, 1, -1.5e6f    },
        { (c.n_tok - 3) % c.n_tok, c.K - 1,   1,  65520.0f  },
        { (c.n_tok - 1) % c.n_tok, 33 % c.K,  1,  65504.0f  },
        { 1 % c.n_tok,             48,       16,  524200.0f },
    };
    std::vector<float> x_spiked = x_clean;
    for (const wspike & s : spikes) for (int64_t j = 0; j < s.w; ++j) x_spiked[(size_t) s.tok*c.K + s.k + j] = s.v;

    // weights first: ggml_moe_up_gate builds the fused node only for weights that already sit in a device buffer
    ggml_init_params ipw = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx_w = ggml_init(ipw);
    ggml_tensor * as_up   = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F16, c.K, n_rows, c.n_exp);
    ggml_tensor * as_gate = c.merged ? nullptr : ggml_new_tensor_3d(ctx_w, GGML_TYPE_F16, c.K, c.N, c.n_exp);
    ggml_tensor * as_down = c.Kd > 0 ? ggml_new_tensor_3d(ctx_w, GGML_TYPE_F16, c.N, c.Kd, c.n_exp) : nullptr;
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, cuda);
    if (!buf_w) { printf("  %-46s SKIPPED (allocation failed)\n", c.name); ggml_free(ctx_w); return 0; }
    ggml_backend_tensor_set(as_up, wu_b.data(), 0, wu_b.size());
    if (as_gate) ggml_backend_tensor_set(as_gate, wg_b.data(), 0, wg_b.size());
    if (as_down) ggml_backend_tensor_set(as_down, wd_b.data(), 0, wd_b.size());

    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, 1, c.n_tok);
    ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tok);
    ggml_tensor * y   = ggml_moe_up_gate(ctx, as_up, as_gate, b, idt, GGML_UNARY_OP_SILU);
    ggml_tensor * out = c.Kd > 0 ? ggml_mul_mat_id(ctx, as_down, y, idt) : y;
    const int64_t n_out = c.Kd > 0 ? c.Kd : c.N;
    const size_t ny = (size_t) n_out*c.n_used*c.n_tok;
    std::vector<float> y_clean(ny), y_spiked(ny);
    int rc = 0;
    ggml_backend_buffer_t buf = nullptr;
    if (y->op != GGML_OP_MOE_FUSED_UP_GATE) {
        printf("  %-46s FAIL (no fused node: %s)\n", c.name, ggml_op_name(y->op)); rc = 1;
    } else if (!ggml_backend_supports_op(cuda, y) || !ggml_backend_supports_op(cuda, out)) {
        printf("  %-46s FAIL (CUDA declines the op)\n", c.name); rc = 1;
    } else {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
        if (!buf) {
            printf("  %-46s SKIPPED (allocation failed)\n", c.name); rc = -1;
        } else {
            ggml_backend_tensor_set(idt, ids.data(), 0, ids.size()*sizeof(int32_t));
            for (int pass = 0; pass < 2 && rc == 0; ++pass) {
                const std::vector<float> & x = pass == 0 ? x_clean : x_spiked;
                ggml_backend_tensor_set(b, x.data(), 0, x.size()*sizeof(float));
                if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
                    printf("  %-46s FAIL (graph compute)\n", c.name); rc = 1; break;
                }
                ggml_backend_tensor_get(out, (pass == 0 ? y_clean : y_spiked).data(), 0, ny*sizeof(float));
            }
        }
    }
    if (rc != 0) {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(buf_w); ggml_free(ctx); ggml_free(ctx_w);
        return rc > 0 ? 1 : 0;
    }

    size_t n_nonfinite = 0, n_tol = 0, n_moved = 0, n_untouched = 0;
    double worst_rel = 0.0;
    std::vector<double> yr((size_t) c.N), yb((size_t) c.N);
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<float> & x  = pass == 0 ? x_clean : x_spiked;
        const std::vector<float> & yy = pass == 0 ? y_clean : y_spiked;
        for (int64_t t = 0; t < c.n_tok; ++t) {
            const float * xr = x.data() + (size_t) t*c.K;
            bool touched = false;
            for (const wspike & s : spikes) if (s.tok == t) touched = true;
            for (int64_t u = 0; u < c.n_used; ++u) {
                const int32_t e = ids[(size_t) t*c.n_used + u];
                for (int64_t n = 0; n < c.N; ++n) {
                    const float * wgr = w_gate(e, n);
                    const float * wur = w_up(e, n);
                    double g = 0.0, mg = 0.0, uu = 0.0, mu = 0.0;
                    for (int64_t k = 0; k < c.K; ++k) {
                        g  += (double) wgr[k]*xr[k]; mg += std::fabs((double) wgr[k]*xr[k]);
                        uu += (double) wur[k]*xr[k]; mu += std::fabs((double) wur[k]*xr[k]);
                    }
                    const double sg = g/(1.0 + std::exp(-g));
                    yr[n] = sg*uu;
                    yb[n] = 1.1*mg*std::fabs(uu) + std::fabs(sg)*mu;   // error scale of the GLU output
                }
                for (int64_t m = 0; m < n_out; ++m) {
                    double ref = 0.0, mag = 0.0;
                    if (c.Kd > 0) {
                        const float * wdr = wd.data() + ((size_t) e*c.Kd + m)*c.N;
                        for (int64_t n = 0; n < c.N; ++n) {
                            ref += (double) wdr[n]*yr[n];
                            mag += std::fabs((double) wdr[n])*(std::fabs(yr[n]) + yb[n]);
                        }
                    } else {
                        ref = yr[m];
                        mag = yb[m];
                    }
                    const size_t iy = ((size_t) t*c.n_used + u)*n_out + m;
                    const float got = yy[iy];
                    if (!std::isfinite(got)) { n_nonfinite++; continue; }
                    const double rel = std::fabs((double) got - ref)/(mag + 1e-6);
                    worst_rel = std::max(worst_rel, rel);
                    if (rel > k_tol_loose) n_tol++;
                    if (pass == 1 && !touched) {
                        n_untouched++;
                        if (memcmp(&y_clean[iy], &y_spiked[iy], sizeof(float)) != 0) n_moved++;
                    }
                }
            }
        }
    }
    size_t n_ref = 0, n_ref_moved = 0;
    const bool ref_ok = dump_and_ref(fdump, fref, y_clean, y_spiked, n_ref, n_ref_moved);
    const bool pass_ok = n_nonfinite == 0 && n_tol == 0 && n_moved == 0 && n_ref_moved == 0 && ref_ok;
    printf("  %-46s %s  non-finite %zu, out of tolerance %zu (worst rel %.2e), untouched rows moved %zu/%zu",
           c.name, pass_ok ? "OK  " : "FAIL", n_nonfinite, n_tol, worst_rel, n_moved, n_untouched);
    if (fref) printf(", vs =0 finite outputs moved %zu/%zu%s", n_ref_moved, n_ref, ref_ok ? "" : " (ref too short)");
    printf("\n");
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return pass_ok ? 0 : 1;
}

int main() {
    {   // ggml_init fills the f16 -> f32 table the host dequantisation below reads (without it the first case's reference is 0)
        ggml_init_params ip0 = { 1024, nullptr, true };
        ggml_free(ggml_init(ip0));
    }
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- SKIP\n");
        return 77;
    }
    const char * g = getenv("PXA_MMID_F16_GUARD");
    printf("PXA_MMID_F16_GUARD=%s\n", g ? g : "(unset: default on)");

    // PXA_MMID_TEST_CASE="f16|q8_0,K,N,E,u,T,bcast" runs that one shape instead of the table (debug aid);
    // PXA_MMID_TEST_VERBOSE=1 prints the first mismatches of each case.
    std::vector<mmid_case> run(std::begin(cases), std::end(cases));
    if (const char * cs = getenv("PXA_MMID_TEST_CASE")) {
        char ty[16] = {0}; long long K, N, E, U, T; int bc = 0;
        if (sscanf(cs, "%15[^,],%lld,%lld,%lld,%lld,%lld,%d", ty, &K, &N, &E, &U, &T, &bc) == 7) {
            const ggml_type t = strcmp(ty, "q8_0") == 0 ? GGML_TYPE_Q8_0 : strcmp(ty, "pxq4") == 0 ? GGML_TYPE_PXQ4 : GGML_TYPE_F16;
            run = { { cs, t, K, N, E, U, T, bc != 0 } };
        }
    }
    const bool verbose = getenv("PXA_MMID_TEST_VERBOSE") && atoi(getenv("PXA_MMID_TEST_VERBOSE")) != 0;
    FILE * fdump = nullptr, * fref = nullptr;
    if (const char * d = getenv("PXA_MMID_TEST_DUMP")) fdump = fopen(d, "wb");
    if (const char * d = getenv("PXA_MMID_TEST_REF"))  {
        fref = fopen(d, "rb");
        if (!fref) { printf("PXA_MMID_TEST_REF=%s: cannot open -- FAIL\n", d); return 1; }
    }
    int rc = 0;
    for (const mmid_case & c : run) {
        const int64_t ne11 = c.bcast ? 1 : c.n_used;
        lcg r(0x5EEDull + (uint64_t) (c.K*131 + c.N*7 + c.n_tok));

        // weights: quantise on the host so the reference reads the exact values the device reads
        std::vector<float> w_deq((size_t) c.K*c.N*c.n_exp);
        std::vector<char> w_bytes;
        if (c.type == GGML_TYPE_PXQ4) {
            // random panel bytes, then real-model fp16 anchors (pxq4_real_anchor: log-uniform [0.03, 2), 1/16 zero)
            const size_t eb = pxq4_bytes(c.N, c.K), ps = eb/(size_t) (c.N/64);
            w_bytes.resize(eb*(size_t) c.n_exp);
            for (auto & v : w_bytes) v = (char) (r.next() & 0xff);
            for (int64_t e = 0; e < c.n_exp; ++e) {
                for (int64_t p = 0; p < c.N/64; ++p) {
                    uint16_t * anchors = (uint16_t *) (w_bytes.data() + (size_t) e*eb + (size_t) p*ps);
                    for (int i = 0; i < 64; ++i) {
                        const ggml_fp16_t h = ggml_fp32_to_fp16(pxq4_real_anchor(r));
                        memcpy(&anchors[i], &h, sizeof(h));
                    }
                }
                pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, w_bytes.data() + (size_t) e*eb, w_deq.data() + (size_t) e*c.N*c.K, c.N, c.K);
            }
        } else {
            std::vector<float> w_raw(w_deq.size());
            for (auto & v : w_raw) v = 0.05f*r.unit();
            if (c.bigw) big_w_rows(w_raw, c.K, c.N, c.n_exp, 33 % c.K);   // the 65504 spike's k (spikes[3])
            const size_t row_bytes = ggml_row_size(c.type, c.K);
            w_bytes.resize(row_bytes*(size_t) c.N*c.n_exp);
            ggml_quantize_chunk(c.type, w_raw.data(), w_bytes.data(), 0, c.N*c.n_exp, c.K, nullptr, nullptr);
            const ggml_type_traits_t tt = ggml_internal_get_type_traits(c.type);
            for (int64_t row = 0; row < c.N*c.n_exp; ++row) {
                tt.to_float(w_bytes.data() + (size_t) row*row_bytes, w_deq.data() + (size_t) row*c.K, (int) c.K);
            }
        }

        // routing: distinct experts per token
        std::vector<int32_t> ids((size_t) c.n_used*c.n_tok);
        for (int64_t t = 0; t < c.n_tok; ++t) {
            const int32_t base = (int32_t) (r.next() % (uint32_t) c.n_exp);
            for (int64_t u = 0; u < c.n_used; ++u) ids[(size_t) t*c.n_used + u] = (int32_t) ((base + u) % c.n_exp);
        }

        std::vector<float> x_clean((size_t) c.K*ne11*c.n_tok);
        for (auto & v : x_clean) v = r.unit();
        const spike spikes[] = {
            { 5 % c.n_tok,               0,            7 % c.K,          300000.0f },
            { (c.n_tok/2) % c.n_tok,     ne11 - 1,     100 % c.K,       -1.5e6f    },
            { (c.n_tok - 3) % c.n_tok,   0,            (c.K - 1),        65520.0f  },   // the first value f16 rounds to inf
            { (c.n_tok - 1) % c.n_tok,   ne11 > 1 ? 1 : 0, 33 % c.K,     65504.0f  },   // f16 max: no guard needed, must stay as is
        };
        std::vector<float> x_spiked = x_clean;
        for (const spike & s : spikes) x_spiked[((size_t) s.tok*ne11 + s.slot)*c.K + s.k] = s.v;

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * as  = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.n_exp);
        ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, ne11, c.n_tok);
        ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tok);
        ggml_tensor * y   = ggml_mul_mat_id(ctx, as, b, idt);
        if (ggml_nbytes(as) != w_bytes.size()) {
            printf("  %-40s FAIL (weight bytes %zu != ggml_nbytes %zu)\n", c.name, w_bytes.size(), ggml_nbytes(as));
            rc = 1; ggml_free(ctx); continue;
        }
        if (!ggml_backend_supports_op(cuda, y)) {
            printf("  %-40s FAIL (CUDA declines the op)\n", c.name);
            rc = 1; ggml_free(ctx); continue;
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
        if (!buf) {
            printf("  %-40s SKIPPED (allocation failed)\n", c.name);
            ggml_free(ctx); continue;
        }
        ggml_backend_tensor_set(as,  w_bytes.data(), 0, w_bytes.size());
        ggml_backend_tensor_set(idt, ids.data(),     0, ids.size()*sizeof(int32_t));

        const size_t ny = (size_t) c.N*c.n_used*c.n_tok;
        std::vector<float> y_clean(ny), y_spiked(ny);
        bool ok = true;
        for (int pass = 0; pass < 2 && ok; ++pass) {
            const std::vector<float> & x = pass == 0 ? x_clean : x_spiked;
            ggml_backend_tensor_set(b, x.data(), 0, x.size()*sizeof(float));
            if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) { ok = false; break; }
            ggml_backend_tensor_get(y, (pass == 0 ? y_clean : y_spiked).data(), 0, ny*sizeof(float));
        }
        if (!ok) {
            printf("  %-40s FAIL (graph compute)\n", c.name);
            rc = 1; ggml_backend_buffer_free(buf); ggml_free(ctx); continue;
        }

        // reference + checks
        size_t n_nonfinite = 0, n_tol = 0, n_moved = 0, n_untouched = 0;
        double worst_rel = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            const std::vector<float> & x  = pass == 0 ? x_clean : x_spiked;
            const std::vector<float> & yy = pass == 0 ? y_clean : y_spiked;
            for (int64_t t = 0; t < c.n_tok; ++t) {
                for (int64_t u = 0; u < c.n_used; ++u) {
                    const int32_t e = ids[(size_t) t*c.n_used + u];
                    const float * xr = x.data() + ((size_t) t*ne11 + (u % ne11))*c.K;
                    bool touched = false;
                    for (const spike & s : spikes) if (s.tok == t && (ne11 == 1 || s.slot == u)) touched = true;
                    // a route that quantises the activation per 32-block (MMQ: q8_1, d = amax/127) loses the rest of a block
                    // that holds a spike: allow that rounding (|err| <= d/2 per element) on blocks with amax > 1 only
                    std::vector<float> qd((size_t) c.K);
                    for (int64_t k0 = 0; k0 < c.K; k0 += 32) {
                        float am = 0.0f;
                        for (int64_t k = k0; k < std::min(c.K, k0 + 32); ++k) am = std::max(am, std::fabs(xr[k]));
                        for (int64_t k = k0; k < std::min(c.K, k0 + 32); ++k) qd[k] = am > 1.0f ? am/127.0f : 0.0f;
                    }
                    for (int64_t n = 0; n < c.N; ++n) {
                        const float * wr = w_deq.data() + ((size_t) e*c.N + n)*c.K;
                        double ref = 0.0, mag = 0.0, qb = 0.0;
                        for (int64_t k = 0; k < c.K; ++k) { ref += (double) wr[k]*xr[k]; mag += std::fabs((double) wr[k]*xr[k]); qb += std::fabs((double) wr[k])*qd[k]; }
                        const size_t iy = ((size_t) t*c.n_used + u)*c.N + n;
                        const float got = yy[iy];
                        const double rel = std::max(0.0, std::fabs((double) got - ref) - qb)/(mag + 1e-6);
                        if (verbose && (!std::isfinite(got) || rel > 1.5e-2) && n_nonfinite + n_tol < 6) {
                            printf("    pass %d tok %lld slot %lld exp %d row %lld: got %.6g ref %.6g mag %.4g%s\n", pass, (long long) t,
                                   (long long) u, e, (long long) n, got, ref, mag, touched ? " (spiked row)" : "");
                        }
                        if (!std::isfinite(got)) { n_nonfinite++; continue; }
                        if (rel > worst_rel) worst_rel = rel;
                        if (rel > 1.5e-2) n_tol++;
                        if (pass == 1 && !touched) {
                            n_untouched++;
                            if (memcmp(&y_clean[iy], &y_spiked[iy], sizeof(float)) != 0) n_moved++;
                        }
                    }
                }
            }
        }
        // guard on vs off: every output finite in the =0 reference must keep its bits
        size_t n_ref = 0, n_ref_moved = 0;
        bool ref_ok = true;
        if (fdump) {
            fwrite(y_clean.data(),  sizeof(float), ny, fdump);
            fwrite(y_spiked.data(), sizeof(float), ny, fdump);
        }
        if (fref) {
            std::vector<float> rr(2*ny);
            if (fread(rr.data(), sizeof(float), 2*ny, fref) != 2*ny) {
                printf("  %-40s FAIL (PXA_MMID_TEST_REF too short)\n", c.name);
                ref_ok = false;
            } else {
                for (size_t i = 0; i < 2*ny; ++i) {
                    if (!std::isfinite(rr[i])) continue;
                    n_ref++;
                    const float got = i < ny ? y_clean[i] : y_spiked[i - ny];
                    if (memcmp(&got, &rr[i], sizeof(float)) != 0) n_ref_moved++;
                }
            }
        }
        const bool pass_ok = n_nonfinite == 0 && n_tol == 0 && n_moved == 0 && n_ref_moved == 0 && ref_ok;
        if (!pass_ok) rc = 1;
        printf("  %-40s %s  non-finite %zu, out of tolerance %zu (worst rel %.2e), untouched rows moved %zu/%zu",
               c.name, pass_ok ? "OK  " : "FAIL", n_nonfinite, n_tol, worst_rel, n_moved, n_untouched);
        if (fref) printf(", vs =0 finite outputs moved %zu/%zu", n_ref_moved, n_ref);
        printf("\n");

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // the env-selected single MMID shape (PXA_MMID_TEST_CASE) skips these sections
    if (!getenv("PXA_MMID_TEST_CASE")) {
        const char * gd = getenv("PXA_F16_GUARD_DENSE");
        const char * go = getenv("PXA_F16_GUARD_OUT");
        printf("MUL_MAT f16 cuBLAS route (PXA_F16_GUARD_DENSE=%s, PXA_F16_GUARD_OUT=%s):\n", gd ? gd : "(unset: on)",
               go ? go : "(unset: on)");
        for (const dense_case & c : dense_cases) rc |= run_dense_case(cuda, c, fdump, fref);
        printf("MOE_FUSED_UP_GATE per-expert loop:\n");
        for (const fused_case & c : fused_cases) rc |= run_fused_case(cuda, c, fdump, fref);
    }

    if (fdump) fclose(fdump);
    if (fref)  fclose(fref);
    ggml_backend_free(cuda);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
