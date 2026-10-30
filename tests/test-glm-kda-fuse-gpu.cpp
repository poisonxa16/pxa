// PXA_GLM_KDA_FUSE GPU bit-identity driver (needs one CUDA device; NOT a ctest; 2026-10-18).
//
// Builds the two glm5next KDA glue chains (gate: ADD -> MUL(per head) -> SCALE -> SIGMOID -> SCALE -> CONT(permute);
// out-gate: SIGMOID -> MUL -> CONT) exactly as build_glm5next.cpp does, at decode (T=1) and prefill (T=64) widths, with
// the real GLM-5.3-Flash KDA shape (head_dim 128, 64 heads) and edge values, runs them on CUDA device 0 and writes every
// CONT output to OUT as raw float32. Run it twice, PXA_GLM_KDA_FUSE=0 and =1 (the switch is read once per process), and
// cmp the two files: the fused kernels must reproduce the per-node kernels bit for bit. PXA_GLM_KDA_FUSE_LOG=1 on the
// fused run prints the fires, so "identical because it never fired" cannot pass unnoticed (gpu-pre.sh checks both).
//
//   test-glm-kda-fuse-gpu OUT
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s OUT\n", argv[0]); return 2; }
#ifndef GGML_USE_CUDA
    printf("built without CUDA: nothing to test\n");
    return 0;
#else
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { fprintf(stderr, "no CUDA device\n"); return 1; }
    FILE * fo = fopen(argv[1], "wb");
    if (!fo) { perror(argv[1]); return 1; }
    const int64_t hd = 128, nh = 64, d_inner = hd*nh;
    std::mt19937 rng(20261018);
    for (int64_t T : { 1, 64 }) {
        ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        // gate chain
        ggml_tensor * g1  = ggml_new_tensor_2d(c, GGML_TYPE_F32, d_inner, T);
        ggml_tensor * dtb = ggml_new_tensor_1d(c, GGML_TYPE_F32, d_inner);
        ggml_tensor * sa  = ggml_new_tensor_1d(c, GGML_TYPE_F32, nh);
        ggml_tensor * add = ggml_add(c, g1, dtb);
        ggml_tensor * A   = ggml_reshape_3d(c, sa, 1, nh, 1);
        ggml_tensor * x   = ggml_mul(c, ggml_reshape_3d(c, add, hd, nh, T), A);
        x = ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, x, -1.0f)), -5.0f);
        ggml_tensor * gate = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, x, hd, nh, T, 1), 0, 2, 1, 3));
        // out-gate chain
        ggml_tensor * dn  = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, nh, T);
        ggml_tensor * w   = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
        ggml_tensor * g2  = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, nh, T);
        ggml_tensor * o   = ggml_mul(c, ggml_rms_norm(c, dn, 1e-6f), w);
        ggml_tensor * og  = ggml_cont_2d(c, ggml_mul(c, o, ggml_sigmoid(c, g2)), d_inner, T);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, gate);
        ggml_build_forward_expand(gf, og);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
        auto put = [&](ggml_tensor * t, float lo, float hi, bool edges) {
            std::uniform_real_distribution<float> d(lo, hi);
            std::vector<float> v(ggml_nelements(t));
            for (auto & f : v) f = d(rng);
            if (edges && v.size() > 16) {
                const float e[] = { 0.0f, -0.0f, 1e-40f, -1e-40f, 88.0f, -88.0f, 1e30f, -1e30f, INFINITY, -INFINITY, 1e-8f, -1e-8f };
                for (size_t k = 0; k < sizeof(e)/sizeof(e[0]); ++k) v[k*7 % v.size()] = e[k];
            }
            ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(float));
        };
        put(g1, -12, 12, true); put(dtb, -2, 2, false); put(sa, -4, -0.01f, false);
        put(dn, -3, 3, true);   put(w, 0.2f, 1.8f, false); put(g2, -10, 10, true);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        for (ggml_tensor * t : { gate, og }) {
            std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(float));
            fwrite(v.data(), sizeof(float), v.size(), fo);
        }
        printf("T=%lld: gate %lld + out-gate %lld floats written\n", (long long) T, (long long) ggml_nelements(gate), (long long) ggml_nelements(og));
        ggml_backend_buffer_free(buf);
        ggml_free(c);
    }
    fclose(fo);
    ggml_backend_free(be);
    return 0;
#endif
}
