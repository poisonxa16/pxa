// GGML_OP_CLAMP on the CPU with several threads must clamp EVERY row (todo dsv41-bringup, 10-31).
// The CPU compute loop runs each node on all n_threads; clamp_f32 used to return early for ith != 0
// while striding rows by nth, so rows j % nth != 0 stayed unclamped (the DeepSeek-V4/V4.1 and GLM-5
// SwiGLU limits on CPU). This builds clamp(x) for a 2304 x 97 tensor (a shared-expert up row count
// and an odd row count) with 1..8 threads, also as the in-place view feeding a second op, and checks
// every element against min(max(x, lo), hi) exactly.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int run(int n_threads, float lo, float hi) {
    const int64_t ne0 = 2304, ne1 = 97;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * c = ggml_clamp(ctx, ggml_scale(ctx, x, 1.0f), lo, hi);   // clamp a computed node (in-place view)
    ggml_tensor * y = ggml_scale(ctx, c, 2.0f);                              // a consumer of the clamped view
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, n_threads);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    // every node of this small graph lives in buf (the clamp result is a view of its input)

    std::vector<float> xv(ne0 * ne1);
    uint32_t s = 12345u;
    for (auto & v : xv) { s = s * 1664525u + 1013904223u; v = ((int) (s >> 8) % 4001 - 2000) * 0.05f; }   // [-100, 100]
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
    std::vector<float> yv(xv.size());
    ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
    int64_t bad = 0, first = -1;
    for (size_t i = 0; i < xv.size(); ++i) {
        const float want = 2.0f * fminf(fmaxf(xv[i], lo), hi);
        if (yv[i] != want) { if (first < 0) first = (int64_t) i; ++bad; }
    }
    printf("clamp [%g, %g] threads %d: %lld / %zu wrong%s\n", lo, hi, n_threads, (long long) bad, xv.size(),
           bad ? "" : " (ok)");
    if (bad) printf("  first wrong element %lld (row %lld)\n", (long long) first, (long long) (first / ne0));
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

int main() {
    int rc = 0;
    for (int t : {1, 2, 3, 8}) {
        rc |= run(t, -10.0f, 10.0f);          // the DS4/DS4.1 up clamp
        rc |= run(t, -INFINITY, 10.0f);       // the DS4/DS4.1 gate clamp
    }
    printf(rc ? "FAIL\n" : "PASS\n");
    return rc;
}
