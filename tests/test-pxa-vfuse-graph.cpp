// test-pxa-vfuse-graph.cpp -- PXA_VFUSE (2026-10-26): the three qwen4exp verify chains as real ggml graphs on
// the CUDA backend, every intermediate marked as an output, dumped to a file. Run once with PXA_VFUSE unset and once with
// PXA_VFUSE=1 (the switch is read once per process) and compare the two dumps byte for byte:
//   test-pxa-vfuse-graph off.bin && PXA_VFUSE=1 test-pxa-vfuse-graph on.bin && cmp off.bin on.bin
// Chains (shapes of the Flash-Next layer, widths 2..8):
//   A  HC lo     x[320, ny] -> scale(1/4) -> silu -> mul_mat(W[320 -> 1024] Q4_0)
//   B  out gate  z, o [6144, ny] as [128, 48*ny] -> sigmoid(z) -> mul(o, sig) -> reshape -> mul_mat(W[6144 -> 512] Q4_0)
//   C  shexp     g[1, ny], o, r [2560, ny] -> sigmoid(g) -> mul(o, sig) -> add(r, mul)
//   E  shexp gate GEMV  w[2560, 1] f32 x x[2560, ny] -> sigmoid -> mul(o, sig) -> add(r, mul) (E2: logit + sig internal, in place)
//   D  gate_mix  xn, g [2560*4, ny] -> hc_gate_mix -> mul_mat(W[2560 -> 1024] Q4_0) and mul_mat(W[2560 -> 256] Q4_0)
//   G  hc combine_norm  residual [2560, 4, ny], block_out [2560, ny], inject [4, ny], gamma [2560*4] (bit 256)
//   F  DeltaNet gated norm  x, z [128, 48*ny] -> fused_rms_norm(x, w[128]) -> sigmoid(z) -> mul -> reshape -> mul_mat(W[6144 -> 512])
// The inputs include exact zeros (an all-zero q8_1 block: d = 0), negatives, large and tiny magnitudes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill(std::vector<float> & v, std::mt19937 & rng, int kind) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < v.size(); ++i) {
        float x = nd(rng);
        switch (kind) {
            case 1: x *= 8.0f; break;                               // pre-activation range of the gates
            case 2: x *= (i % 97 == 0) ? 1e-30f : 1.0f; break;      // a few tiny values
            default: break;
        }
        if (i % 61 == 0) x = 0.0f;
        v[i] = x;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s OUT.bin\n", argv[0]); return 2; }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 3; }

    const int widths[] = { 2, 3, 4, 5, 6, 8 };
    const int NW = (int) (sizeof(widths)/sizeof(widths[0]));
    struct ggml_init_params ip = { (size_t) 512*ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    struct leaf { ggml_tensor * t; int kind; };
    std::vector<leaf> leaves;
    std::vector<ggml_tensor *> outs;
    auto out = [&](ggml_tensor * t) { ggml_set_output(t); outs.push_back(t); return t; };

    ggml_tensor * W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 320, 1024);
    ggml_tensor * W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 6144, 512);
    ggml_tensor * W3 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 2560, 1024);
    ggml_tensor * W4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 2560, 256);
    ggml_tensor * WG = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * WN = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * WC = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560*4);
    for (int w = 0; w < NW; ++w) {
        const int ny = widths[w];
        // A
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 320, ny); leaves.push_back({x, w == 1 ? 2 : 0});
        ggml_tensor * t = out(ggml_scale(ctx, x, 0.25f));
        ggml_tensor * s = out(ggml_silu(ctx, t));
        ggml_tensor * y1 = out(ggml_mul_mat(ctx, W1, s));
        ggml_build_forward_expand(gf, y1);
        // B
        ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, ny); leaves.push_back({z, 1});
        ggml_tensor * o = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, ny); leaves.push_back({o, w == 2 ? 2 : 0});
        ggml_tensor * sg = out(ggml_sigmoid(ctx, ggml_reshape_2d(ctx, z, 128, 48*ny)));
        ggml_tensor * m  = out(ggml_mul(ctx, ggml_reshape_2d(ctx, o, 128, 48*ny), sg));
        ggml_tensor * y2 = out(ggml_mul_mat(ctx, W2, ggml_reshape_2d(ctx, m, 6144, ny)));
        ggml_build_forward_expand(gf, y2);
        // C
        ggml_tensor * g  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ny);    leaves.push_back({g, 1});
        ggml_tensor * so = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({so, 0});
        ggml_tensor * r  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({r, w == 3 ? 2 : 0});
        ggml_tensor * sgc = out(ggml_sigmoid(ctx, g));
        ggml_tensor * mc  = out(ggml_mul(ctx, so, sgc));
        ggml_tensor * ad  = out(ggml_add(ctx, r, mc));
        ggml_build_forward_expand(gf, ad);
        // D: hc gate_mix -> two quantized projections on the same mixed (DeltaNet qkvz then z)
        ggml_tensor * xn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560*4, ny); leaves.push_back({xn, 0});
        ggml_tensor * gg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560*4, ny); leaves.push_back({gg, 1});
        ggml_tensor * mx = out(ggml_hc_gate_mix(ctx, xn, gg, 4));
        ggml_tensor * y3 = out(ggml_mul_mat(ctx, W3, mx));
        ggml_tensor * y4 = out(ggml_mul_mat(ctx, W4, mx));
        ggml_build_forward_expand(gf, y3);
        ggml_build_forward_expand(gf, y4);
        // E: shexp gate GEMV (f32 1-row) -> sigmoid -> mul -> add
        ggml_tensor * xe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({xe, 0});
        ggml_tensor * oe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({oe, 0});
        ggml_tensor * re = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({re, 0});
        ggml_tensor * ge = out(ggml_mul_mat(ctx, WG, xe));
        ggml_tensor * se = out(ggml_sigmoid(ctx, ge));
        ggml_tensor * me = out(ggml_mul(ctx, oe, se));
        ggml_tensor * ae = out(ggml_add(ctx, re, me));
        ggml_build_forward_expand(gf, ae);
        // E2: the same chain with the gate logit and sig left internal, so the allocator may run the SIGMOID in place
        ggml_tensor * xe2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({xe2, 0});
        ggml_tensor * oe2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({oe2, 0});
        ggml_tensor * re2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny); leaves.push_back({re2, 0});
        ggml_tensor * se2 = ggml_sigmoid(ctx, ggml_mul_mat(ctx, WG, xe2));
        ggml_tensor * me2 = out(ggml_mul(ctx, oe2, se2));
        ggml_tensor * ae2 = out(ggml_add(ctx, re2, me2));
        ggml_build_forward_expand(gf, ae2);
        // F: DeltaNet gated norm (FUSED_RMS_NORM -> SIGMOID -> MUL -> quantized ssm_out)
        ggml_tensor * xf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 48*ny); leaves.push_back({xf, w == 4 ? 2 : 0});
        ggml_tensor * zf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 48*ny); leaves.push_back({zf, 1});
        ggml_tensor * fn = out(ggml_fused_rms_norm(ctx, xf, WN, 1e-6f));
        ggml_tensor * sf = out(ggml_sigmoid(ctx, zf));
        ggml_tensor * mf = out(ggml_mul(ctx, fn, sf));
        ggml_tensor * yf = out(ggml_mul_mat(ctx, W2, ggml_reshape_2d(ctx, mf, 6144, ny)));
        ggml_build_forward_expand(gf, yf);
        // G: hc combine + next norm
        ggml_tensor * rs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 4, ny); leaves.push_back({rs, 0});
        ggml_tensor * bo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ny);    leaves.push_back({bo, w == 5 ? 2 : 0});
        ggml_tensor * ij = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, ny);       leaves.push_back({ij, 1});
        ggml_tensor * cn = out(ggml_hc_combine_norm(ctx, rs, bo, ij, WC, 1e-6f));
        ggml_build_forward_expand(gf, cn);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 4; }

    std::mt19937 rng(1234567);
    for (ggml_tensor * W : { W1, W2, W3, W4 }) {
        std::vector<float> f((size_t) W->ne[0]*W->ne[1]);
        fill(f, rng, 0);
        std::vector<char> q(ggml_nbytes(W));
        ggml_quantize_chunk(W->type, f.data(), q.data(), 0, W->ne[1], W->ne[0], nullptr, nullptr);
        ggml_backend_tensor_set(W, q.data(), 0, q.size());
    }
    {
        std::vector<float> f(2560);
        fill(f, rng, 0);
        ggml_backend_tensor_set(WG, f.data(), 0, f.size()*sizeof(float));
        std::vector<float> n(128);
        fill(n, rng, 0);
        for (auto & v : n) v = 1.0f + 0.25f*v;   // (1 + w) folded gamma
        ggml_backend_tensor_set(WN, n.data(), 0, n.size()*sizeof(float));
        std::vector<float> c(2560*4);
        fill(c, rng, 0);
        for (auto & v : c) v = 1.0f + 0.25f*v;
        ggml_backend_tensor_set(WC, c.data(), 0, c.size()*sizeof(float));
    }
    for (auto & l : leaves) {
        std::vector<float> f((size_t) ggml_nelements(l.t));
        fill(f, rng, l.kind);
        ggml_backend_tensor_set(l.t, f.data(), 0, f.size()*sizeof(float));
    }
    // twice: the second run is the one dumped (any first-use allocation happened in the first)
    for (int rep = 0; rep < 2; ++rep) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 5; }
    }
    // relative timing (same process, same graph): median of 5 batches of 40 computes, host wall incl. launches
    double us_med = 0.0;
    {
        std::vector<double> v;
        for (int b = 0; b < 5; ++b) {
            ggml_backend_synchronize(be);
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 40; ++k) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            v.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/40.0);
        }
        std::sort(v.begin(), v.end());
        us_med = v[2];
    }
    FILE * fo = fopen(argv[1], "wb");
    if (!fo) return 6;
    size_t tot = 0;
    for (ggml_tensor * t : outs) {
        std::vector<char> h(ggml_nbytes(t));
        ggml_backend_tensor_get(t, h.data(), 0, h.size());
        fwrite(h.data(), 1, h.size(), fo);
        tot += h.size();
    }
    fclose(fo);
    printf("VFUSE-GRAPH wrote %zu bytes (%zu outputs, %d graph nodes, PXA_VFUSE=%s) time %.1f us per graph\n", tot, outs.size(),
           ggml_graph_n_nodes(gf), getenv("PXA_VFUSE") ? getenv("PXA_VFUSE") : "unset", us_med);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
