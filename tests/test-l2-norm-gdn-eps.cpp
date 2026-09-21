// mj2-audit finding (item B): ggml_l2_norm's epsilon convention does not match the Gated
// DeltaNet Q/K normalization formula this tree already establishes elsewhere.
//
// ggml_l2_norm (ggml/src/ggml.c, ggml_compute_forward_l2_norm_f32) computes
//     scale = 1 / max(sqrt(sum(x^2)), eps)                 -- an epsilon FLOOR on the norm
// Our own GGML_OP_DELTA_NET CPU kernel (ggml/src/ggml.c, ggml_compute_forward_delta_net_f32,
// the q_norm_inv/k_norm_inv computation) and this engine's own test-kda-delta-net.cpp
// reference oracle (its qn/kn computation) both instead use
//     scale = 1 / sqrt(sum(x^2) + eps)                     -- an epsilon ADDEND under the sqrt
// citing "llama.cpp PR #27773's ggml_compute_forward_gated_delta_net_one_chunk()" as the
// upstream Gated DeltaNet Q/K normalization this op is matching. These are different
// functions of the input norm: for the same eps, ggml_l2_norm's protection band (where eps
// changes the output vs a plain 1/sqrt(sum(x^2))) is entered once sum(x^2) < eps^2, while the
// reference formula's protection band is entered once sum(x^2) is merely comparable to eps --
// a completely different length scale whenever eps < 1, i.e. always.
//
// build_qkv() (src/llama-delta-net.cpp) calls ggml_l2_norm on q_conv/k_conv and then
// unconditionally passes the result into ggml_delta_net(), whose own CPU kernel re-normalizes
// them AGAIN with the (reference-matching) add-eps formula -- so on the CPU backend the wrong
// first pass is largely masked for well-conditioned inputs (the second, correct pass
// renormalizes whatever the first pass produced back to very nearly unit length, and the
// residual correction from re-normalizing an already ~unit vector is far below one float32
// ULP). ggml-cuda/pxa/delta-net.cu's kernel, by contrast, loads its q/k shared-memory tile as
// "(normalized)" already and contains no internal renormalization pass (no eps/rsqrt of any
// kind appears in that file) -- so the CUDA backend depends on ggml_l2_norm's own output being
// correct on its own, with no second pass available to hide a discrepancy. The CPU and CUDA
// backends of the nominally-same GGML_OP_DELTA_NET therefore do not treat their q/k inputs the
// same way (CPU: renormalize unconditionally; CUDA: trust the caller already did).
//
// This test isolates ggml_l2_norm itself against the reference eps formula; it needs no GPU
// and does not attempt to model the CPU double-normalization masking effect described above.
//
// CPU only, no GPU, no model: deterministic, fixed hand-picked inputs (no RNG needed -- the
// discrepancy is a formula-shape mismatch, not something that needs many random samples to
// surface).

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// The Gated DeltaNet reference Q/K normalization this tree already establishes elsewhere:
// ggml/src/ggml.c ggml_compute_forward_delta_net_f32 (q_norm_inv/k_norm_inv), and
// tests/test-kda-delta-net.cpp's own reference_delta_net() (qn/kn), both citing upstream
// llama.cpp PR #27773.
static void reference_l2_norm(const std::vector<float> & x, float eps, std::vector<float> & out) {
    float sumsq = 0.0f;
    for (float v : x) sumsq += v * v;
    const float scale = 1.0f / std::sqrt(sumsq + eps);
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * scale;
}

static bool run_row(const char * name, const std::vector<float> & x, float eps, double tol) {
    printf("  %-36s eps=%.1e ... ", name, (double) eps);
    fflush(stdout);

    std::vector<float> ref;
    reference_l2_norm(x, eps, ref);

    const int64_t n = (int64_t) x.size();
    struct ggml_init_params ip = { 16ull*1024ull*1024ull, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { printf("FAIL (ggml_init)\n"); return false; }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(t->data, x.data(), ggml_nbytes(t));

    ggml_tensor * r = ggml_l2_norm(ctx, t, eps);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float * got = (const float *) r->data;

    double max_diff = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        max_diff = std::fmax(max_diff, std::fabs((double) got[i] - (double) ref[i]));
    }

    ggml_free(ctx);

    const bool ok = max_diff < tol;
    printf("%s  (max |d| %.6g, tol %.3g)\n", ok ? "OK" : "FAIL", max_diff, tol);
    return ok;
}

int main() {
    printf("test-l2-norm-gdn-eps: ggml_l2_norm vs the Gated DeltaNet reference eps formula\n");
    printf("(the eps is applied INSIDE the square root, per the reference implementation)\n");
    bool ok = true;

    // Control row: sum(x^2) = 1.0 >> eps, so neither formula's protection band is meaningfully
    // engaged; the two conventions must agree to within float32 rounding regardless of which
    // eps convention is "correct". This shows the discrepancy below is regime-specific, not a
    // wholesale mismatch.
    ok &= run_row("well-conditioned (control)", {0.6f, 0.8f, 0.0f, 0.0f}, 1e-6f, 1e-4);

    // Demonstration row: x = [1e-4, 0, 0, 0], sum(x^2) = 1e-8, eps = 1e-6.
    //   ggml_l2_norm:  sqrt(1e-8) = 1e-4 >= eps(1e-6)  -> NO floor engaged -> scale = 1/1e-4 = 1e4
    //                  -> y = [1.0, 0, 0, 0]  (treated as already unit-scale)
    //   reference:     scale = 1/sqrt(1e-8 + 1e-6) = 1/sqrt(1.01e-6) ~= 995.0
    //                  -> y ~= [0.0995, 0, 0, 0]  (still meaningfully damped)
    // A ~10x scale disagreement (|d| ~= 0.9) -- not a rounding-level difference, and squarely
    // inside the regime a masked/reset delta-net Q or K row can land in.
    ok &= run_row("small-norm (demonstrates discrepancy)", {1e-4f, 0.0f, 0.0f, 0.0f}, 1e-6f, 1e-4);

    printf("%s\n", ok ? "ALL OK" : "FAILURES (expected -- see mj2-audit report, item B)");
    return ok ? 0 : 1;
}
