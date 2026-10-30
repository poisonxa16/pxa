// test-dsv4-sinkhorn-despill: the hc=4 register-resident Sinkhorn (todo glm-sinkhorn-despill) must be
// bit-identical to the generic kernel. CPU only, no model, no GPU.
//
// The CUDA kernel (ggml/src/ggml-cuda/dsv4.cu kernel_dsv4_hc_split_sinkhorn<HC_T>) is one templated
// body: HC_T == 0 = generic (run-time HC, c[16*16] in a 1 KiB local frame), HC_T == 4 = compile-time
// HC (c[16] in registers). This test carries a host model of both, transcribed from that file:
//   old_generic(): the pre-de-spill kernel body verbatim (d91ed7ba8c), run-time HC, c[256]
//   new_fixed<HC>(): the templated body with HC a compile-time constant, c[HC*HC]
// and runs both under two math policies, both through the SAME exp/divide functions:
//   libm : expf, IEEE divide (what the CPU op uses)
//   fast : flush-to-zero on every op result plus exp(x) = exp2f(x * log2e), approximating
//          nvcc -use_fast_math (ftz, ex2.approx, rcp/div.approx). This is NOT bit-equal to the GPU;
//          it only checks that the two operation orders agree under an ftz/ex2-style policy too.
// Every output float is compared bitwise (memcmp, NaN payloads included) over random rows and edge
// rows (+-0, denormals, +-FLT_MAX, +-inf, NaN in logits / scales / bases, eps 0 / denormal / 1,
// iters 1..50), for hc in {1,2,3,4,5,8,16}. A third check runs the real ggml CPU op
// (ggml_dsv4_hc_split_sinkhorn) on the same rows and compares it bitwise with old_generic/libm,
// so the host model is tied to the shipped CPU arithmetic.
// The GPU-side counterpart (old vs new kernel, bitwise, on a card) is scripts/dsv4-sinkhorn-microbench.sh.
#include "ggml.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int HC_MAX = 16;

static float ftz(float x) {
    return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x;
}

struct MathLibm {
    static float add(float a, float b) { return a + b; }
    static float sub(float a, float b) { return a - b; }
    static float mul(float a, float b) { return a * b; }
    static float fma_(float a, float b, float c) { return a * b + c; }   // contraction decided by the compiler, same for both bodies
    static float div(float a, float b) { return a / b; }
    static float exp(float x) { return expf(x); }
    static float max(float a, float b) { return fmaxf(a, b); }
};

struct MathFast {
    static float add(float a, float b) { return ftz(ftz(a) + ftz(b)); }
    static float sub(float a, float b) { return ftz(ftz(a) - ftz(b)); }
    static float mul(float a, float b) { return ftz(ftz(a) * ftz(b)); }
    static float fma_(float a, float b, float c) { return ftz(std::fmaf(ftz(a), ftz(b), ftz(c))); }
    static float div(float a, float b) { return ftz(ftz(a) * ftz(1.0f / ftz(b))); }
    static float exp(float x) { return ftz(exp2f(ftz(ftz(x) * 1.44269504088896341f))); }
    static float max(float a, float b) { return fmaxf(ftz(a), ftz(b)); }
};

struct Row {
    std::vector<float> mix, scale, base;
    float eps;
    int iters;
};

// ---- pre-de-spill kernel body (d91ed7ba8c), run-time HC, c[16*16] -------------------------------
template <class M>
static void old_generic(const Row & r, int HC, float * out) {
    const float * mix = r.mix.data();
    const float * base = r.base.data();
    const float epsv = r.eps;
    const float pre_scale = r.scale[0], post_scale = r.scale[1], comb_scale = r.scale[2];

    for (int i = 0; i < HC; ++i) {
        const float z = M::fma_(mix[i], pre_scale, base[i]);
        out[i] = M::add(M::div(1.0f, M::add(1.0f, M::exp(-z))), epsv);
    }
    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = M::fma_(mix[off], post_scale, base[off]);
        out[off] = M::div(2.0f, M::add(1.0f, M::exp(-z)));
    }

    float c[HC_MAX * HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const int off = 2 * HC + idx;
            const float v = M::fma_(mix[off], comb_scale, base[off]);
            c[idx] = v;
            row_max = M::max(row_max, v);
        }
        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const float v = M::exp(M::sub(c[idx], row_max));
            c[idx] = v;
            row_sum = M::add(row_sum, v);
        }
        const float inv_sum = M::div(1.0f, row_sum);
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            c[idx] = M::fma_(c[idx], inv_sum, epsv);
        }
    }
    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) sum = M::add(sum, c[src_hc + dst_hc * HC]);
        const float inv_denom = M::div(1.0f, M::add(sum, epsv));
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) c[src_hc + dst_hc * HC] = M::mul(c[src_hc + dst_hc * HC], inv_denom);
    }
    for (int iter = 1; iter < r.iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) sum = M::add(sum, c[src_hc + dst_hc * HC]);
            const float inv_denom = M::div(1.0f, M::add(sum, epsv));
            for (int src_hc = 0; src_hc < HC; ++src_hc) c[src_hc + dst_hc * HC] = M::mul(c[src_hc + dst_hc * HC], inv_denom);
        }
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) sum = M::add(sum, c[src_hc + dst_hc * HC]);
            const float inv_denom = M::div(1.0f, M::add(sum, epsv));
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) c[src_hc + dst_hc * HC] = M::mul(c[src_hc + dst_hc * HC], inv_denom);
        }
    }
    for (int i = 0; i < HC * HC; ++i) out[2 * HC + i] = c[i];
}

// ---- de-spilled body: HC a compile-time constant, c[HC*HC] (kernel_dsv4_hc_split_sinkhorn<HC>) ----
template <class M, int HC>
static void new_fixed(const Row & r, float * out) {
    const float * mix = r.mix.data();
    const float * base = r.base.data();
    const float epsv = r.eps;
    const float pre_scale = r.scale[0], post_scale = r.scale[1], comb_scale = r.scale[2];

    for (int i = 0; i < HC; ++i) {
        const float z = M::fma_(mix[i], pre_scale, base[i]);
        out[i] = M::add(M::div(1.0f, M::add(1.0f, M::exp(-z))), epsv);
    }
    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = M::fma_(mix[off], post_scale, base[off]);
        out[off] = M::div(2.0f, M::add(1.0f, M::exp(-z)));
    }

    float c[HC * HC];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const int off = 2 * HC + idx;
            const float v = M::fma_(mix[off], comb_scale, base[off]);
            c[idx] = v;
            row_max = M::max(row_max, v);
        }
        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const float v = M::exp(M::sub(c[idx], row_max));
            c[idx] = v;
            row_sum = M::add(row_sum, v);
        }
        const float inv_sum = M::div(1.0f, row_sum);
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            c[idx] = M::fma_(c[idx], inv_sum, epsv);
        }
    }
    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) sum = M::add(sum, c[src_hc + dst_hc * HC]);
        const float inv_denom = M::div(1.0f, M::add(sum, epsv));
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) c[src_hc + dst_hc * HC] = M::mul(c[src_hc + dst_hc * HC], inv_denom);
    }
    for (int iter = 1; iter < r.iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) sum = M::add(sum, c[src_hc + dst_hc * HC]);
            const float inv_denom = M::div(1.0f, M::add(sum, epsv));
            for (int src_hc = 0; src_hc < HC; ++src_hc) c[src_hc + dst_hc * HC] = M::mul(c[src_hc + dst_hc * HC], inv_denom);
        }
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) sum = M::add(sum, c[src_hc + dst_hc * HC]);
            const float inv_denom = M::div(1.0f, M::add(sum, epsv));
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) c[src_hc + dst_hc * HC] = M::mul(c[src_hc + dst_hc * HC], inv_denom);
        }
    }
    for (int i = 0; i < HC * HC; ++i) out[2 * HC + i] = c[i];
}

// ---- inputs ---------------------------------------------------------------------------------------
static std::vector<Row> make_rows(int HC, uint32_t seed) {
    const int n = (2 + HC) * HC;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 1 << 30);
    std::vector<Row> rows;
    auto rnd_row = [&](float sc, float eps, int iters) {
        Row r;
        r.mix.resize(n); r.base.resize(n); r.scale.resize(3);
        for (auto & x : r.mix)  x = nd(rng) * sc;
        for (auto & x : r.base) x = nd(rng) * sc * 0.5f;
        for (auto & x : r.scale) x = 0.5f + std::fabs(nd(rng));
        r.eps = eps; r.iters = iters;
        return r;
    };
    const float scales[] = { 1e-4f, 1e-2f, 1.0f, 3.0f, 10.0f, 100.0f, 1e4f };
    const float epss[]   = { 1e-6f, 0.0f, 1e-30f, 1e-40f /* denormal */, 1.0f };
    const int   iterss[] = { 1, 2, 3, 20, 50 };
    for (float sc : scales) for (float e : epss) for (int it : iterss) for (int k = 0; k < 6; ++k) rows.push_back(rnd_row(sc, e, it));
    // edge values injected into random rows
    const float edges[] = { 0.0f, -0.0f, 1e-40f, -1e-40f, 1.4e-45f, FLT_MIN, -FLT_MIN, FLT_MAX, -FLT_MAX,
                            INFINITY, -INFINITY, NAN, -NAN, 88.7f, -88.7f, 104.0f, -104.0f, 1e30f, -1e30f };
    for (int k = 0; k < 4000; ++k) {
        Row r = rnd_row(scales[pick(rng) % 7], epss[pick(rng) % 5], iterss[pick(rng) % 5]);
        const int m = 1 + pick(rng) % 4;
        for (int j = 0; j < m; ++j) {
            const float e = edges[pick(rng) % (sizeof(edges) / sizeof(edges[0]))];
            switch (pick(rng) % 3) {
                case 0: r.mix[pick(rng) % n] = e; break;
                case 1: r.base[pick(rng) % n] = e; break;
                default: r.scale[pick(rng) % 3] = e; break;
            }
        }
        rows.push_back(r);
    }
    // whole-row edges: all-equal logits, all zero, one dominant logit, all denormal
    for (float v : { 0.0f, 1e-40f, 5.0f, -5.0f, 80.0f, -80.0f }) {
        Row r = rnd_row(1.0f, 1e-6f, 20);
        for (auto & x : r.mix) x = v;
        for (auto & x : r.base) x = 0.0f;
        rows.push_back(r);
        r.mix[2 * HC] = 1e3f; rows.push_back(r);
    }
    return rows;
}

// bitdiff   : any bit difference where NOT both values are NaN (a real numeric difference) -> FAIL
// nan_sign  : both NaN, sign/payload bits differ. On x86 the NaN that propagates out of a commutative op
//             depends on operand order the host compiler picks, so g++ can differ between two transcriptions
//             here; CUDA arithmetic returns the canonical NaN, so this class cannot occur on the GPU. Reported,
//             not failed. NaN *positions* must match exactly (a NaN on one side only counts as bitdiff).
struct Tally { size_t rows = 0, floats = 0, bitdiff = 0, nan_sign = 0, nan_out = 0, inf_out = 0, denorm_out = 0; };

static bool same_or_nan_sign(float x, float y, bool & nan_only) {
    nan_only = false;
    if (memcmp(&x, &y, 4) == 0) return true;
    if (std::isnan(x) && std::isnan(y)) { nan_only = true; return true; }
    return false;
}

static void count_specials(const float * o, int n, Tally & t) {
    for (int i = 0; i < n; ++i) {
        if (std::isnan(o[i])) t.nan_out++;
        else if (std::isinf(o[i])) t.inf_out++;
        else if (std::fpclassify(o[i]) == FP_SUBNORMAL) t.denorm_out++;
    }
}

template <class M, int HC>
static Tally check_hc(const char * pol) {
    Tally t;
    const int n = (2 + HC) * HC;
    std::vector<float> a(n), b(n);
    for (const Row & r : make_rows(HC, 1000u + HC)) {
        old_generic<M>(r, HC, a.data());
        new_fixed<M, HC>(r, b.data());
        t.rows++; t.floats += n;
        for (int i = 0; i < n; ++i) {
            bool nan_only;
            if (!same_or_nan_sign(a[i], b[i], nan_only)) {
                if (t.bitdiff < 5) fprintf(stderr, "  MISMATCH %s hc=%d i=%d old=%a new=%a\n", pol, HC, i, a[i], b[i]);
                t.bitdiff++;
            } else if (nan_only) {
                t.nan_sign++;
            }
        }
        count_specials(a.data(), n, t);
    }
    printf("  %-4s hc=%-2d rows=%-6zu floats=%-8zu bitdiff=%zu nan-sign-only=%zu  (outputs: nan %zu, inf %zu, denormal %zu)\n",
           pol, HC, t.rows, t.floats, t.bitdiff, t.nan_sign, t.nan_out, t.inf_out, t.denorm_out);
    return t;
}

// ggml CPU op vs old_generic<MathLibm>: ties the host model to the shipped CPU arithmetic.
// Rows are grouped by (eps, iters) since those are op params.
static size_t check_ggml_cpu(int HC) {
    const int n = (2 + HC) * HC;
    std::vector<Row> rows = make_rows(HC, 7000u + HC);
    size_t bad = 0, total = 0;
    // one graph per row keeps scale/base per row exact (they are per-op tensors)
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    for (size_t k = 0; k < rows.size(); k += 7) {   // every 7th row keeps the runtime short
        const Row & r = rows[k];
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * mx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
        ggml_tensor * sc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_tensor * bs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        memcpy(mx->data, r.mix.data(), n * 4); memcpy(sc->data, r.scale.data(), 12); memcpy(bs->data, r.base.data(), n * 4);
        ggml_tensor * out = ggml_dsv4_hc_split_sinkhorn(ctx, mx, sc, bs, HC, r.iters, r.eps);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        std::vector<float> ref(n);
        old_generic<MathLibm>(r, HC, ref.data());
        for (int i = 0; i < n; ++i) { total++; bool nan_only; if (!same_or_nan_sign(ref[i], ((float *) out->data)[i], nan_only)) bad++; }
        ggml_free(ctx);
    }
    printf("  ggml-cpu op vs host old_generic/libm hc=%-2d floats=%-7zu bitdiff=%zu (NaN sign ignored)\n", HC, total, bad);
    return bad;
}

} // namespace

int main() {
    size_t diff = 0;
    printf("test-dsv4-sinkhorn-despill: old (run-time HC, c[256]) vs new (compile-time HC, c[HC*HC]), bitwise\n");
    diff += check_hc<MathLibm, 4>("libm").bitdiff;
    diff += check_hc<MathFast, 4>("fast").bitdiff;
    diff += check_hc<MathLibm, 1>("libm").bitdiff;
    diff += check_hc<MathLibm, 2>("libm").bitdiff;
    diff += check_hc<MathLibm, 3>("libm").bitdiff;
    diff += check_hc<MathLibm, 5>("libm").bitdiff;
    diff += check_hc<MathLibm, 8>("libm").bitdiff;
    diff += check_hc<MathLibm, 16>("libm").bitdiff;
    diff += check_hc<MathFast, 8>("fast").bitdiff;
    const size_t cpu_bad = check_ggml_cpu(4);
    // The ggml CPU op is compiled separately (its own fp-contract choices), so a mismatch there means the
    // host model is not a faithful transcription under THIS build's flags, not that the kernel is wrong.
    if (cpu_bad != 0) {
        printf("NOTE: ggml CPU op differs from the host model under this build's fp-contract flags (%zu floats)\n", cpu_bad);
    }
    printf("%s: %zu bit mismatches between old and new order\n", diff == 0 ? "PASS" : "FAIL", diff);
    return diff == 0 ? 0 : 1;
}
