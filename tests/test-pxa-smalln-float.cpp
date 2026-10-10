// test-pxa-smalln-float.cpp (default): PXA_SMALLN_FLOAT GEMV vs a double-precision host reference, F32/F16/BF16 weights,
// ne11 2..8, the hc_inject shape (K=10240, R=4) and a few others. Run once with PXA_SMALLN_FLOAT=1 (the kernel) and once
// without (the cuBLAS fallback); both must pass the same tolerance. Optional argv[1] = file to dump the outputs to, so
// the two runs can be compared directly (tests/run-pxa-smalln-float.sh).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>

int main(int argc, char ** argv) {
    FILE * dump = argc > 1 ? fopen(argv[1], "wb") : nullptr;
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { fprintf(stderr, "no CUDA device\n"); return 2; }
    struct shape { int K, R; ggml_type t; };
    const shape shapes[] = { {10240, 4, GGML_TYPE_F16}, {10240, 4, GGML_TYPE_F32}, {2560, 64, GGML_TYPE_F16},
                             {4096, 512, GGML_TYPE_F32}, {1280, 48, GGML_TYPE_BF16}, {5120, 4096, GGML_TYPE_F16} };
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0, cases = 0; double worst = 0;
    for (const auto & sh : shapes) for (int n = 2; n <= 8; ++n) {
        ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, sh.t, sh.K, sh.R);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.K, n);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> wf((size_t) sh.K*sh.R), xf((size_t) sh.K*n);
        for (auto & v : wf) v = nd(rng)*0.05f; for (auto & v : xf) v = nd(rng);
        // round the weights to the storage type so the reference sees the same values
        std::vector<uint8_t> wb(ggml_nbytes(w));
        if (sh.t == GGML_TYPE_F32) memcpy(wb.data(), wf.data(), wb.size());
        else if (sh.t == GGML_TYPE_F16) { auto * p = (ggml_fp16_t *) wb.data(); for (size_t i = 0; i < wf.size(); ++i) { p[i] = ggml_fp32_to_fp16(wf[i]); wf[i] = ggml_fp16_to_fp32(p[i]); } }
        else { auto * p = (ggml_bf16_t *) wb.data(); for (size_t i = 0; i < wf.size(); ++i) { p[i] = ggml_fp32_to_bf16(wf[i]); wf[i] = ggml_bf16_to_fp32(p[i]); } }
        ggml_backend_tensor_set(w, wb.data(), 0, wb.size());
        ggml_backend_tensor_set(x, xf.data(), 0, xf.size()*sizeof(float));
        ggml_backend_graph_compute(be, gf);
        std::vector<float> yo((size_t) sh.R*n);
        ggml_backend_tensor_get(y, yo.data(), 0, yo.size()*sizeof(float));
        double maxrel = 0;
        for (int c = 0; c < n; ++c) for (int r = 0; r < sh.R; ++r) {
            double ref = 0, mag = 0;
            for (int k = 0; k < sh.K; ++k) { ref += (double) wf[(size_t) r*sh.K + k]*xf[(size_t) c*sh.K + k]; mag += fabs((double) wf[(size_t) r*sh.K + k]*xf[(size_t) c*sh.K + k]); }
            const double rel = fabs(yo[(size_t) c*sh.R + r] - ref)/(mag + 1e-12);
            if (rel > maxrel) maxrel = rel;
        }
        // cuBLAS on sm_60 may run an f16 weight through an f16 GEMM (x rounded to f16): 2e-3 of the |w||x| mass covers it
        const double tol = 2e-3;
        ++cases; if (maxrel > tol) { ++fails; fprintf(stderr, "FAIL %s K=%d R=%d n=%d maxrel=%.3g\n", ggml_type_name(sh.t), sh.K, sh.R, n, maxrel); }
        if (maxrel > worst) worst = maxrel;
        if (dump) fwrite(yo.data(), sizeof(float), yo.size(), dump);
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
    if (dump) fclose(dump);
    printf("%s: %d cases, %d failed, worst relative error %.3g (PXA_SMALLN_FLOAT=%s)\n", fails ? "TEST FAILED" : "TEST PASSED",
           cases, fails, worst, getenv("PXA_SMALLN_FLOAT") ? getenv("PXA_SMALLN_FLOAT") : "unset");
    ggml_backend_free(be);
    return fails ? 1 : 0;
}
