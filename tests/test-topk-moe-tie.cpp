// test-topk-moe-tie -- bug #224: the fused CUDA softmax->top-k router (topk-moe.cu) had no
// tie-break in its warp argmax, and padding lanes (n_experts < 32) started at weight 0.
//
//   (a) two experts with bit-equal softmax weight in the top-k: each owning lane kept its own
//       expert, both marked it -INF and both stored into slot k, so one tied expert was dropped
//       and the next-lower expert took its place;
//   (b) n_experts < 32 with underflowed weights (exactly 0): a padding lane (id >= n_experts)
//       could win the tie at 0 and emit an out-of-range expert id.
//
// The graph is the exact node chain llm_build_moe_ffn emits (soft_max -> reshape -> argsort ->
// view -> get_rows [-> reshape -> sum_rows -> div]), which ggml-cuda fuses into topk_moe_cuda.
// The expected answer is computed here: weights descending, ties to the LOWER expert index.
// Set PXA_GLUE_DBG=1 to see the "topk_moe FIRING" banner proving the fused kernel ran.
// Needs a CUDA device; exits 0 with SKIP when there is none.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

static int g_fail = 0;

// logits -> expected ids (ties to the lower index) using the SAME softmax semantics the kernel
// uses: equal logits give bit-equal probabilities, distinct logits never tie.
static std::vector<int> expected_ids(const float * lg, int n_exp, int k) {
    std::vector<int> idx(n_exp);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize(k);
    return idx;
}

static void run_case(ggml_backend_t be, const char * name, int n_exp, int k, int n_tok, bool norm,
                     const std::vector<float> & logits_h) {
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * logits  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_exp, n_tok);
    ggml_tensor * probs   = ggml_soft_max(ctx, logits);                               // [n_exp, n_tok]
    ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, n_exp, n_tok);
    ggml_tensor * sel     = ggml_top_k(ctx, probs, k);                                // [k, n_tok]
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, sel);                          // [1, k, n_tok]
    if (norm) {
        weights = ggml_reshape_2d(ctx, weights, k, n_tok);
        ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
        weights = ggml_div(ctx, weights, wsum);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, weights);
    ggml_build_forward_expand(gf, sel);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(logits, logits_h.data(), 0, logits_h.size() * sizeof(float));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  FAIL %s: graph_compute\n", name);
        ++g_fail;
    } else {
        // the fused kernel writes ids into the argsort's buffer with row stride n_exp
        std::vector<int32_t> ids((size_t) n_exp * n_tok);
        ggml_backend_tensor_get(sel->view_src ? sel->view_src : sel, ids.data(), 0, ids.size() * sizeof(int32_t));
        std::vector<float> w((size_t) k * n_tok);
        ggml_backend_tensor_get(weights, w.data(), 0, w.size() * sizeof(float));

        int bad = 0;
        for (int t = 0; t < n_tok; ++t) {
            const float * lg = logits_h.data() + (size_t) t * n_exp;
            const std::vector<int> want = expected_ids(lg, n_exp, k);
            float wsum = 0.f;
            for (int j = 0; j < k; ++j) {
                const int got = ids[(size_t) t * n_exp + j];
                if (got != want[j]) {
                    if (bad < 4) {
                        printf("  FAIL %s: tok %d slot %d id %d, want %d\n", name, t, j, got, want[j]);
                    }
                    ++bad;
                }
                const float wj = w[(size_t) t * k + j];
                if (!std::isfinite(wj)) { ++bad; }
                wsum += wj;
            }
            if (norm && std::fabs(wsum - 1.0f) > 1e-4f) {
                if (bad < 4) printf("  FAIL %s: tok %d normalised weights sum %g\n", name, t, (double) wsum);
                ++bad;
            }
        }
        printf("  %s %s (n_exp=%d k=%d n_tok=%d norm=%d)\n", bad ? "FAIL" : "ok  ", name, n_exp, k, n_tok, (int) norm);
        g_fail += bad ? 1 : 0;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { printf("SKIP: cuda init failed\n"); return 0; }

    // (a) exact ties in the top-k, several router widths, decode and multi-row, with/without norm.
    for (int n_exp : { 8, 16, 64, 128, 256, 512 }) {
        for (int n_tok : { 1, 7 }) {
            for (bool norm : { false, true }) {
                const int k = n_exp >= 64 ? 8 : 4;
                std::vector<float> lg((size_t) n_exp * n_tok);
                unsigned s = 12345u + n_exp * 7 + n_tok;
                for (auto & v : lg) { s = s * 1664525u + 1013904223u; v = -4.0f + 3.0f * (float) (s >> 8) / (float) (1u << 24); }
                for (int t = 0; t < n_tok; ++t) {
                    float * r = lg.data() + (size_t) t * n_exp;
                    // a two-way tie at the top between experts in DIFFERENT lanes and lane slots,
                    // and a three-way tie further down (still inside the top-k)
                    const int a = (3 + t) % n_exp, b = (n_exp - 1 - t) % n_exp;
                    r[a] = r[b] = 5.0f;
                    const int c = (n_exp / 2 + 1) % n_exp, d = (n_exp / 4 + 2) % n_exp, e = (5 * t + 1) % n_exp;
                    if (c != a && c != b && d != a && d != b && e != a && e != b) { r[c] = r[d] = r[e] = 3.0f; }
                }
                char name[64];
                snprintf(name, sizeof(name), "tie");
                run_case(be, name, n_exp, k, n_tok, norm, lg);
            }
        }
    }

    // (b) n_experts < 32, all but one expert underflow to exactly 0 after the softmax: padding
    //     lanes must never win the tie at 0 (ids must stay < n_exp, lowest indices first).
    for (int n_exp : { 4, 8, 16 }) {
        const int k = 3;
        std::vector<float> lg(n_exp, -1.0e4f);
        lg[n_exp - 1] = 0.0f;
        run_case(be, "underflow", n_exp, k, 1, false, lg);
    }

    ggml_backend_free(be);
    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}
