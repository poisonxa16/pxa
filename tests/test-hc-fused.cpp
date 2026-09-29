// test-hc-fused: PXA_QWEN4EXP_HC_FUSED ops against the stock qwen4exp hc chain, on the CUDA backend.
// Builds both on identical random inputs and reports max |diff| and the count of bit mismatches.
// Build: g++ -O2 -std=c++17 -I ggml/include tests/test-hc-fused.cpp -L<build>/ggml/src -lggml
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int run(ggml_backend_t be, int n_embd, int hc, int nt, bool gamma_on) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int hc_dim = n_embd*hc;
    ggml_tensor * r   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nt);
    ggml_tensor * inj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);
    ggml_tensor * gam = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor * g   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt);
    const float eps = 1e-6f;
    // stock combine (build_qwen4exp_hc_combine)
    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inj, 1.0f/(float) hc));
    w = ggml_scale(ctx, w, 2.0f);
    w = ggml_reshape_3d(ctx, w, 1, hc, nt);
    ggml_tensor * bb = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, b, n_embd, 1, nt), n_embd, hc, nt, 1);
    ggml_tensor * s_res = ggml_add(ctx, r, ggml_mul(ctx, bb, w));
    // stock norm (build_qwen4exp_hc_mix)
    ggml_tensor * s_xn = ggml_mul(ctx, ggml_reshape_2d(ctx, ggml_rms_norm(ctx, s_res, eps), hc_dim, nt), gam);
    // stock gate mix, fed from the stock xn
    ggml_tensor * gated = ggml_reshape_3d(ctx, ggml_mul(ctx, s_xn, ggml_sigmoid(ctx, g)), n_embd, hc, nt);
    ggml_tensor * s_mix = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, nt, gated->nb[1]*hc, 0));
    for (int c = 1; c < hc; ++c) s_mix = ggml_add(ctx, s_mix, ggml_view_2d(ctx, gated, n_embd, nt, gated->nb[1]*hc, gated->nb[1]*c));
    s_mix = ggml_scale(ctx, s_mix, 1.0f/(float) hc);
    // fused
    ggml_tensor * both = ggml_hc_combine_norm(ctx, r, b, inj, gamma_on ? gam : nullptr, eps);
    ggml_tensor * f_res = ggml_view_3d(ctx, both, n_embd, hc, nt, both->nb[1], both->nb[2], 0);
    ggml_tensor * f_xn = gamma_on ? ggml_view_2d(ctx, both, hc_dim, nt, both->nb[2], both->nb[3]) : nullptr;
    ggml_tensor * f_mix = ggml_hc_gate_mix(ctx, s_xn, g, hc);   // same xn input as the stock mix
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, s_mix);
    ggml_build_forward_expand(gf, s_res);
    ggml_build_forward_expand(gf, both);
    ggml_build_forward_expand(gf, f_mix);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    std::mt19937 rng(1234 + nt);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill = [&](ggml_tensor * t, float sc) { std::vector<float> v(ggml_nelements(t)); for (auto & x : v) x = nd(rng)*sc; ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t)); };
    fill(r, 3.0f); fill(b, 1.0f); fill(inj, 2.0f); fill(g, 2.0f);
    { std::vector<float> v(hc_dim); for (auto & x : v) x = 1.0f + 0.1f*nd(rng); ggml_backend_tensor_set(gam, v.data(), 0, ggml_nbytes(gam)); }
    ggml_backend_graph_compute(be, gf);
    auto get = [&](ggml_tensor * t) { std::vector<float> v(ggml_nelements(t)); ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t)); return v; };
    auto cmp = [&](const char * nm, ggml_tensor * a, ggml_tensor * f) {
        auto va = get(a), vf = get(f);
        double md = 0; size_t nbit = 0;
        for (size_t i = 0; i < va.size(); ++i) { md = std::max(md, (double) std::fabs(va[i]-vf[i])); if (memcmp(&va[i], &vf[i], 4)) nbit++; }
        printf("  %-8s n=%zu max|diff|=%.3g bit-mismatch=%zu\n", nm, va.size(), md, nbit);
        return nbit;
    };
    printf("n_embd=%d hc=%d nt=%d gamma=%d\n", n_embd, hc, nt, (int) gamma_on);
    size_t bad = cmp("combine", s_res, f_res);
    if (gamma_on) bad += cmp("norm", s_xn, f_xn);
    bad += cmp("gatemix", s_mix, f_mix);
    ggml_gallocr_free(ga); ggml_backend_buffer_free(buf); ggml_free(ctx);
    return bad ? 1 : 0;
}

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 2; }
    int rc = 0;
    for (int nt : {1, 3, 8}) { rc |= run(be, 2560, 4, nt, true); rc |= run(be, 2560, 4, nt, false); }
    rc |= run(be, 512, 4, 2, true);
    printf(rc ? "RESULT: BIT MISMATCH (see above)\n" : "RESULT: BIT-IDENTICAL\n");
    ggml_backend_free(be);
    return rc;
}
