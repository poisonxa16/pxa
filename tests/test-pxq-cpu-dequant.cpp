// test-pxq-cpu-dequant.cpp — correctness self-test for the PXQ CPU panel-dequant fallback
// (ggml/src/pxq-cpu.c). NOT registered with CMake — standalone build (needs F16C, x86):
//
//   cc  -O2 -std=c11   -Iggml/include -Iggml/src -c ggml/src/pxq-cpu.c -o /tmp/pxq-cpu.o
//   c++ -O2 -std=c++17 -mf16c -Iggml/include -Iggml/src tests/test-pxq-cpu-dequant.cpp
//       /tmp/pxq-cpu.o -o /tmp/test-pxq-cpu-dequant -lm -lpthread   (one line)
//   /tmp/test-pxq-cpu-dequant
//
// Three layers of checking, per tier:
//   1. HANDCRAFTED PANELS (the strong test): buffers filled with random (sanitized) bytes are
//      dequantized by pxa_pxq_dequant_2d and compared BIT-EXACT against an INDEPENDENT
//      element-indexed scalar reference written straight from the format spec (its own table
//      copies, its own byte arithmetic — it never calls pxq-cpu.c code).
//   2. QUANTIZER ROUNDTRIP: random f32 experts are quantized by the NATIVE quantizers
//      (src/pxq{2,3,6}-quantize.inc.cpp, included below) and pxa_pxq_dequant_2d is compared
//      BIT-EXACT against the quantizers' own parity-locked reference dequant
//      (pxq6_dequant_expert / pxq2_dequant_expert / pxq3_dequant_expert), plus a
//      reconstruction-error sanity bound vs the f32 source (all tiers).
//   3. FUSED/MATMUL WRAPPERS: pxa_pxq_moe_up_gate_cpu (dense + routed-row mapping + biases +
//      every supported unary op + limit clamps + multi-thread partition) and
//      pxa_pxq_mul_mat_cpu are compared against a naive f32 evaluation built on
//      pxa_pxq_dequant_2d.

#include "pxq-cpu.h"   // pulls in ggml.h (enum ggml_type, ggml_unary_op)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <immintrin.h>

// --- native quantizers, compiled standalone exactly like pxa-bench/pxq6_ref.cpp -------------
static inline float    test_fp16_to_fp32(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t test_fp32_to_fp16(float f)    { return _cvtss_sh(f, 0 /*RN*/); }
#define ggml_fp16_to_fp32 test_fp16_to_fp32
#define ggml_fp32_to_fp16 test_fp32_to_fp16
// shim for the inc.cpp dead-column guard (defined in src/llama-quantize.cpp in the real build)
#include <atomic>
static std::atomic<int64_t> g_pxq_imx_dead_cols{0};
static bool pxq_imatrix_column_usable(const float * w, int64_t K) {
    if (!w) return false;
    double s = 0.0;
    for (int64_t i = 0; i < K; ++i) {
        if (!std::isfinite(w[i]) || w[i] < 0.0f) { g_pxq_imx_dead_cols.fetch_add(1); return false; }
        s += (double) w[i];
    }
    if (s > 0.0) return true;
    g_pxq_imx_dead_cols.fetch_add(1);
    return false;
}
#include "../src/pxq6-quantize.inc.cpp"
#include "../src/pxq2-quantize.inc.cpp"
#include "../src/pxq3-quantize.inc.cpp"
#undef ggml_fp16_to_fp32
#undef ggml_fp32_to_fp16

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

// --- independent element-indexed reference (format spec only, own tables) -------------------

namespace ref {

static const float px16_book[16] = PXQ6_BOOK_INIT;   // frozen PX16 book contract
static const float sub16[16]     = PXQ6_SUB16_INIT;
static const float sub8[16]      = PXQ6_SUB8_INIT;
static const float lm4[4]        = PXQ2_BOOK_INIT;
static const float lm8[8]        = PXQ3_BOOK_INIT;

static size_t panel_stride(ggml_type t, int64_t K) {
    const int64_t KB = K/32;
    switch (t) {
        case GGML_TYPE_PXQ4:   return 128 + KB*1088;
        case GGML_TYPE_PXQ4HQ: return 128 + KB*1152;
        case GGML_TYPE_PXQ2:   return 128 + KB*576;
        case GGML_TYPE_PXQ3:   return 128 + KB*832;
        default: abort();
    }
}
static size_t buf_size(ggml_type t, int64_t R, int64_t K) { return (R/64)*panel_stride(t, K); }

// one element, indexed from scratch
static float elem(ggml_type t, const uint8_t * base, int64_t row, int64_t col, int64_t K) {
    const int64_t p = row/64, kb = col/32;
    const int r = (int)(row%64), j = (int)(col%32);
    const uint8_t * panel = base + p*panel_stride(t, K);
    int slab_sz, hdr;
    switch (t) {
        case GGML_TYPE_PXQ4:   slab_sz = 1088; hdr = 128; break;
        case GGML_TYPE_PXQ4HQ: slab_sz = 1152; hdr = 128; break;
        case GGML_TYPE_PXQ2:   slab_sz = 576;  hdr = 128; break;
        case GGML_TYPE_PXQ3:   slab_sz = 832;  hdr = 128; break;
        default: abort();
    }
    const uint8_t * slab = panel + hdr + kb*slab_sz;
    const float anchor = test_fp16_to_fp32(((const uint16_t *)panel)[r]);
    if (t == GGML_TYPE_PXQ4 || t == GGML_TYPE_PXQ4HQ) {
        float eff;
        int code_off;
        if (t == GGML_TYPE_PXQ4) {
            const uint8_t sc = slab[r];
            eff = anchor * sub16[j < 16 ? (sc & 0xf) : (sc >> 4)];
            code_off = 64;
        } else {
            const uint8_t sc = slab[2*r + (j >> 4)];          // b0: elems 0-15, b1: 16-31
            eff = anchor * sub8[((j >> 3) & 1) ? (sc >> 4) : (sc & 0xf)];
            code_off = 128;
        }
        const uint8_t byte = slab[code_off + r*16 + j/2];
        const int c = (j & 1) ? byte >> 4 : byte & 0xf;
        return eff * px16_book[c];
    }
    // PXQ2/PXQ3: shared E16 scale machinery
    const uint8_t sc = slab[r];
    const float eff = anchor * sub16[j < 16 ? (sc & 0xf) : (sc >> 4)];
    if (t == GGML_TYPE_PXQ2) {
        const uint8_t * q = slab + 64 + r*8;
        const int c = (q[j >> 2] >> (2*(j & 3))) & 3;
        return eff * lm4[c];
    }
    // PXQ3 bit-plane: w0/w1 = low 2-bit planes, w2 = bit2 plane
    const uint8_t * q = slab + 64 + r*12;
    uint32_t w[3] = {0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        w[0] |= (uint32_t)q[i]     << (8*i);
        w[1] |= (uint32_t)q[4 + i] << (8*i);
        w[2] |= (uint32_t)q[8 + i] << (8*i);
    }
    const uint32_t lo = j < 16 ? w[0] : w[1];
    const int c = (int)(((lo >> (2*(j & 15))) & 3) | (((w[2] >> j) & 1) << 2));
    return eff * lm8[c];
}

} // namespace ref

// --- handcrafted-panel test -----------------------------------------------------------------

static void sanitize_anchors(ggml_type t, uint8_t * buf, int64_t R, int64_t K, std::mt19937 & rng) {
    const size_t ps = ref::panel_stride(t, K);
    for (int64_t p = 0; p < R/64; ++p) {
        uint16_t * anchors = (uint16_t *)(buf + p*ps);
        for (int r = 0; r < 64; ++r) {
            uint16_t h = (uint16_t)(rng() & 0x7FFF);           // positive
            if ((h & 0x7C00) == 0x7C00) h = 0x7BFF;            // no inf/nan
            if ((rng() & 15) == 0) h = 0;                      // some zero rows
            anchors[r] = h;
        }
    }
}

static void test_handcrafted(ggml_type t, const char * name) {
    const int64_t R = 128, K = 96;                             // 2 panels, 3 slabs
    std::mt19937 rng(42 + (int)t);
    std::vector<uint8_t> buf(ref::buf_size(t, R, K));
    for (auto & b : buf) b = (uint8_t)(rng() & 0xff);
    sanitize_anchors(t, buf.data(), R, K, rng);

    std::vector<float> out(R*K, -777.0f);
    pxa_pxq_dequant_2d(t, buf.data(), out.data(), R, K);

    bool ok = true;
    for (int64_t r = 0; r < R && ok; ++r) {
        for (int64_t c = 0; c < K; ++c) {
            const float want = ref::elem(t, buf.data(), r, c, K);
            const float got  = out[r*K + c];
            if (memcmp(&want, &got, 4) != 0) {
                printf("    MISMATCH %s row %lld col %lld: want %a got %a\n", name, (long long)r, (long long)c, want, got);
                ok = false;
                break;
            }
        }
    }
    char what[128];
    snprintf(what, sizeof(what), "%s handcrafted panels vs independent ref (bit-exact)", name);
    check(ok, what);
}

// --- quantizer roundtrip test ---------------------------------------------------------------

static std::vector<float> rand_matrix(int64_t R, int64_t K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    std::vector<float> m(R*K);
    for (auto & v : m) v = nd(rng);
    // a couple of exact-zero rows (anchor-zero path)
    for (int64_t c = 0; c < K; ++c) m[3*K + c] = 0.0f;
    return m;
}

static double rel_rms(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double)(a[i]-b[i])*(a[i]-b[i]);
        den += (double)b[i]*b[i];
    }
    return den > 0 ? sqrt(num/den) : sqrt(num);
}

static void test_roundtrip(ggml_type t, const char * name, double max_rel) {
    const int64_t R = 128, K = 64;
    std::vector<float> src = rand_matrix(R, K, 7 + (int)t);
    std::vector<uint8_t> q(ref::buf_size(t, R, K), 0);
    std::vector<float> ref_deq(R*K, 0.0f), got(R*K, -777.0f);
    bool have_ref = true;

    switch (t) {
        case GGML_TYPE_PXQ4:
            pxq6_quantize_expert(src.data(), q.data(), R, K, nullptr, 0, 0);
            pxq6_dequant_expert(q.data(), ref_deq.data(), R, K, 0);
            break;
        case GGML_TYPE_PXQ4HQ:
            pxq6_quantize_expert(src.data(), q.data(), R, K, nullptr, 1, 0);
            pxq6_dequant_expert(q.data(), ref_deq.data(), R, K, 1);
            break;
        case GGML_TYPE_PXQ2:
            pxq2_quantize_expert(src.data(), q.data(), R, K, nullptr, 0);
            pxq2_dequant_expert(q.data(), ref_deq.data(), R, K);
            break;
        case GGML_TYPE_PXQ3:
            pxq3_quantize_expert(src.data(), q.data(), R, K, nullptr, 0);
            pxq3_dequant_expert(q.data(), ref_deq.data(), R, K);
            break;
        default: abort();
    }
    pxa_pxq_dequant_2d(t, q.data(), got.data(), R, K);

    char what[128];
    if (have_ref) {
        const bool bit_ok = memcmp(got.data(), ref_deq.data(), got.size()*4) == 0;
        snprintf(what, sizeof(what), "%s quantizer roundtrip vs native reference (bit-exact)", name);
        check(bit_ok, what);
    }
    // independent-ref bit-check on the quantizer's real output bytes too
    bool iok = true;
    for (int64_t r = 0; r < R && iok; ++r) {
        for (int64_t c = 0; c < K; ++c) {
            const float want = ref::elem(t, q.data(), r, c, K);
            if (memcmp(&want, &got[r*K + c], 4) != 0) { iok = false; break; }
        }
    }
    snprintf(what, sizeof(what), "%s quantizer output vs independent ref (bit-exact)", name);
    check(iok, what);

    const double rel = rel_rms(got, src);
    snprintf(what, sizeof(what), "%s reconstruction error sanity (rel RMS %.4f < %.2f)", name, rel, max_rel);
    check(rel < max_rel, what);
}

// --- fused up/gate + matmul wrapper tests ---------------------------------------------------

static float naive_act(int op, float x) {
    switch (op) {
        case GGML_UNARY_OP_RELU: return x > 0 ? x : 0;
        case GGML_UNARY_OP_SILU: return x/(1.0f + expf(-x));
        case GGML_UNARY_OP_GELU: return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
        case GGML_UNARY_OP_SWIGLU_OAI: { float xi = std::min(x, 7.0f); return xi/(1.0f + expf(-xi*1.702f)); }
        default: abort();
    }
}

static void test_fused(int unary_op, float limit, const char * opname) {
    const int64_t R = 64, K = 64;
    // real quantized PXQ6 up + PXQ2 gate (a mixed PXQ-UNIVERSAL pair)
    std::vector<float> upf = rand_matrix(R, K, 100), gatef = rand_matrix(R, K, 101);
    std::vector<uint8_t> up_q(ref::buf_size(GGML_TYPE_PXQ4, R, K)), gate_q(ref::buf_size(GGML_TYPE_PXQ2, R, K));
    pxq6_quantize_expert(upf.data(), up_q.data(), R, K, nullptr, 0, 0);
    pxq2_quantize_expert(gatef.data(), gate_q.data(), R, K, nullptr, 0);

    std::vector<float> up_d(R*K), gate_d(R*K);
    pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, up_q.data(), up_d.data(), R, K);
    pxa_pxq_dequant_2d(GGML_TYPE_PXQ2, gate_q.data(), gate_d.data(), R, K);

    std::vector<float> up_b(R), gate_b(R);
    std::mt19937 rng(55);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (auto & v : up_b) v = nd(rng);
    for (auto & v : gate_b) v = nd(rng);

    // routed-row mapping over 2 tokens x 2 slots, ne11 = 1 (broadcast activations)
    const int ne11 = 1, ntok = 2, nslot = 2;
    std::vector<float> x(ntok*ne11*K);
    for (auto & v : x) v = nd(rng);
    const pxa_pxq_rowmap rows[3] = { {0, 0}, {1, 0}, {0, 1} };
    const int64_t ny = 3;

    const size_t nb11 = K*4, nb12 = ne11*K*4;
    const size_t nb1 = R*4, nb2 = nslot*R*4;
    std::vector<float> dst(ntok*nslot*R, 0.0f), want(ntok*nslot*R, 0.0f);

    // naive reference from the dequantized f32 matrices (same accumulation order)
    for (int64_t iy = 0; iy < ny; ++iy) {
        const float * xr = &x[(size_t)rows[iy].i2*(nb12/4) + (size_t)(rows[iy].i1 % ne11)*(nb11/4)];
        float * out = &want[(size_t)rows[iy].i1*(nb1/4) + (size_t)rows[iy].i2*(nb2/4)];
        for (int64_t ix = 0; ix < R; ++ix) {
            double gd = 0, ud = 0;
            for (int64_t j = 0; j < K; ++j) {
                gd += (double)gate_d[ix*K + j]*(double)xr[j];
                ud += (double)up_d[ix*K + j]*(double)xr[j];
            }
            float act = naive_act(unary_op, (float)gd + gate_b[ix]);
            if (limit > 1e-6f) act = std::min(act, limit);
            float uv = (float)ud + up_b[ix];
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) uv = 1.0f + std::max(std::min(uv, 7.0f), -7.0f);
            else if (limit > 1e-6f) uv = std::max(-limit, std::min(limit, uv));
            out[ix] = uv*act;
        }
    }

    // exercise the thread partition: 3 "threads"
    for (int ith = 0; ith < 3; ++ith) {
        pxa_pxq_moe_up_gate_cpu(GGML_TYPE_PXQ4, up_q.data(), GGML_TYPE_PXQ2, gate_q.data(), R, K,
                up_b.data(), gate_b.data(),
                (const char *)x.data(), nb11, nb12,
                (char *)dst.data(), nb1, nb2,
                rows, ne11, ny, unary_op, limit, ith, 3);
    }

    bool ok = memcmp(dst.data(), want.data(), dst.size()*4) == 0;
    char what[128];
    snprintf(what, sizeof(what), "fused up/gate (%s, limit %.1f, mapped, mixed pair, nth=3)", opname, limit);
    check(ok, what);

    // dense mode (rows == NULL), no biases, single thread
    std::vector<float> ddst(ny*R, 0.0f), dwant(ny*R, 0.0f);
    for (int64_t iy = 0; iy < ny; ++iy) {
        const float * xr = &x[iy*K];
        for (int64_t ix = 0; ix < R; ++ix) {
            double gd = 0, ud = 0;
            for (int64_t j = 0; j < K; ++j) {
                gd += (double)gate_d[ix*K + j]*(double)xr[j];
                ud += (double)up_d[ix*K + j]*(double)xr[j];
            }
            float act = naive_act(unary_op, (float)gd);
            if (limit > 1e-6f) act = std::min(act, limit);
            float uv = (float)ud;
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) uv = 1.0f + std::max(std::min(uv, 7.0f), -7.0f);
            else if (limit > 1e-6f) uv = std::max(-limit, std::min(limit, uv));
            dwant[iy*R + ix] = uv*act;
        }
    }
    pxa_pxq_moe_up_gate_cpu(GGML_TYPE_PXQ4, up_q.data(), GGML_TYPE_PXQ2, gate_q.data(), R, K,
            NULL, NULL, (const char *)x.data(), K*4, 0,
            (char *)ddst.data(), R*4, 0, NULL, (int)ny, ny, unary_op, limit, 0, 1);
    ok = memcmp(ddst.data(), dwant.data(), ddst.size()*4) == 0;
    snprintf(what, sizeof(what), "fused up/gate (%s, limit %.1f, dense)", opname, limit);
    check(ok, what);
}

static void test_mul_mat() {
    const int64_t R = 128, K = 64;
    std::vector<float> af = rand_matrix(R, K, 200);
    std::vector<uint8_t> aq(ref::buf_size(GGML_TYPE_PXQ3, R, K));
    pxq3_quantize_expert(af.data(), aq.data(), R, K, nullptr, 0);
    std::vector<float> ad(R*K);
    pxa_pxq_dequant_2d(GGML_TYPE_PXQ3, aq.data(), ad.data(), R, K);

    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t ny = 5;
    std::vector<float> x(ny*K);
    for (auto & v : x) v = nd(rng);

    std::vector<float> dst(ny*R, 0.0f), want(ny*R, 0.0f);
    for (int64_t iy = 0; iy < ny; ++iy) {
        for (int64_t ix = 0; ix < R; ++ix) {
            double acc = 0;
            for (int64_t j = 0; j < K; ++j) acc += (double)ad[ix*K + j]*(double)x[iy*K + j];
            want[iy*R + ix] = (float)acc;
        }
    }
    for (int ith = 0; ith < 4; ++ith) {
        pxa_pxq_mul_mat_cpu(GGML_TYPE_PXQ3, aq.data(), R, K,
                (const char *)x.data(), K*4, 0,
                (char *)dst.data(), R*4, 0, NULL, (int)ny, ny, ith, 4);
    }
    check(memcmp(dst.data(), want.data(), dst.size()*4) == 0, "mul_mat fallback (PXQ3, dense, nth=4)");
}

int main() {
    printf("PXQ CPU panel-dequant fallback self-test\n");

    printf("[1] handcrafted panels vs independent per-element reference\n");
    test_handcrafted(GGML_TYPE_PXQ4,   "PXQ4");
    test_handcrafted(GGML_TYPE_PXQ4HQ, "PXQ4-HQ");
    test_handcrafted(GGML_TYPE_PXQ2,   "PXQ2");
    test_handcrafted(GGML_TYPE_PXQ3,   "PXQ3");

    printf("[2] native-quantizer roundtrip\n");
    test_roundtrip(GGML_TYPE_PXQ4,   "PXQ4",   0.30);
    test_roundtrip(GGML_TYPE_PXQ4HQ, "PXQ4-HQ", 0.30);
    test_roundtrip(GGML_TYPE_PXQ2,   "PXQ2",   0.80);
    test_roundtrip(GGML_TYPE_PXQ3,   "PXQ3",   0.50);

    printf("[3] fused up/gate + mul_mat wrappers\n");
    test_fused(GGML_UNARY_OP_SILU,       0.0f, "silu");
    test_fused(GGML_UNARY_OP_GELU,       2.5f, "gelu");
    test_fused(GGML_UNARY_OP_RELU,       0.0f, "relu");
    test_fused(GGML_UNARY_OP_SWIGLU_OAI, 7.0f, "swiglu_oai");
    test_mul_mat();

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
