// test-fa-route-qkv -- flash attention over a QUANTIZED K/V cache on the pre-Turing cards:
//
//   bug #235: on sm_60 / sm_61 the planner returned the vec kernel's "unsupported" for K/V types
//             it has no instance for (head 256 with q4_0 / q5_0 / q4_1 ...) without trying the tile
//             kernels, so the node went to the CPU backend (P100 decode 0.44 t/s, and on sm_61 at
//             every width because sm_61 sends every head-256 node to the vec kernel).
//   bug #236: on sm_70 a narrow node over a quantized cache goes to WMMA, which converts the whole
//             K/V view to f16 each token; past a 128 MiB whole-cache conversion PXA_FA_DEEP_QKV_TILE
//             (default on) sends it to the vec kernel, which reads the cache in place. This binary
//             checks support and accuracy only (pass n_kv as the 4th argument for a deep case); the
//             route itself is asserted by test-fa-route-qkv-volta.sh.
//
// For every (head, K/V type, width) case this asks the CUDA backend whether it takes the node
// (on cc 600/610/700 it MUST, at default precision -- a "no" is the bug), runs it, and scores the
// output against an independent double-precision reference computed here from the dequantized
// K/V. Run with PXA_CORE_ROUTES=1 to see which kernel served each node. Needs a CUDA device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static float frand(unsigned & s) { s = s * 1664525u + 1013904223u; return (float) (s >> 8) / (float) (1u << 24) * 2.0f - 1.0f; }

static void run_case(ggml_backend_t be, int D, ggml_type tkv, int nb, int n_kv, bool must_support, bool prec_f32 = false) {
    const int nh = 8, nh_kv = 2;
    const float scale = 1.0f / sqrtf((float) D);
    unsigned s = 0x235u + D * 131u + (unsigned) tkv * 7u + nb;

    std::vector<float> qh((size_t) D * nb * nh), kf((size_t) D * n_kv * nh_kv), vf((size_t) D * n_kv * nh_kv);
    for (auto & x : qh) x = frand(s);
    for (auto & x : kf) x = frand(s);
    for (auto & x : vf) x = frand(s);
    const size_t row_sz = ggml_row_size(tkv, D);
    std::vector<uint8_t> kq(row_sz * n_kv * nh_kv), vq(row_sz * n_kv * nh_kv);
    ggml_quantize_chunk(tkv, kf.data(), kq.data(), 0, (int64_t) n_kv * nh_kv, D, nullptr, nullptr);
    ggml_quantize_chunk(tkv, vf.data(), vq.data(), 0, (int64_t) n_kv * nh_kv, D, nullptr, nullptr);
    // the reference sees exactly the values the kernel sees
    const auto tr = ggml_internal_get_type_traits(tkv);
    std::vector<float> kd(kf.size()), vd(vf.size());
    for (int r = 0; r < n_kv * nh_kv; ++r) {
        tr.to_float(kq.data() + (size_t) r * row_sz, kd.data() + (size_t) r * D, D);
        tr.to_float(vq.data() + (size_t) r * row_sz, vd.data() + (size_t) r * D, D);
    }
    const int nb_pad = GGML_PAD(nb, GGML_KQ_MASK_PAD);
    std::vector<ggml_fp16_t> mh((size_t) n_kv * nb_pad, ggml_fp32_to_fp16(0.0f));

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nb, nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, tkv, D, n_kv, nh_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, tkv, D, n_kv, nh_kv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nb_pad, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);   // default precision
    if (prec_f32) {
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);   // the arches llama builds with f32 attention
    }

    const bool sup = ggml_backend_supports_op(be, out);
    char name[96];
    snprintf(name, sizeof(name), "D=%d %s width %d n_kv %d%s", D, ggml_type_name(tkv), nb, n_kv, prec_f32 ? " prec f32" : "");
    if (!sup) {
        if (must_support) { printf("  FAIL %s: CUDA declines the node (it would run on the CPU backend)\n", name); ++g_fail; }
        else              { printf("  skip %s: not supported on this card\n", name); }
        ggml_free(ctx);
        return;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * sizeof(float));
    ggml_backend_tensor_set(k, kq.data(), 0, kq.size());
    ggml_backend_tensor_set(v, vq.data(), 0, vq.size());
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  FAIL %s: graph_compute\n", name); ++g_fail;
    } else {
        // out: [D, nh, nb]
        std::vector<float> o((size_t) D * nh * nb);
        ggml_backend_tensor_get(out, o.data(), 0, o.size() * sizeof(float));
        double err = 0.0, ref2 = 0.0;
        std::vector<double> sc(n_kv), acc(D);
        for (int t = 0; t < nb; ++t) {
            for (int h = 0; h < nh; ++h) {
                const int hk = h / (nh / nh_kv);
                const float * qr = qh.data() + ((size_t) h * nb + t) * D;
                double mx = -1e300;
                for (int j = 0; j < n_kv; ++j) {
                    const float * kr = kd.data() + ((size_t) hk * n_kv + j) * D;
                    double d = 0; for (int c = 0; c < D; ++c) d += (double) qr[c] * kr[c];
                    sc[j] = d * scale; if (sc[j] > mx) mx = sc[j];
                }
                double sum = 0; for (int j = 0; j < n_kv; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
                for (int c = 0; c < D; ++c) acc[c] = 0;
                for (int j = 0; j < n_kv; ++j) {
                    const float * vr = vd.data() + ((size_t) hk * n_kv + j) * D;
                    for (int c = 0; c < D; ++c) acc[c] += sc[j] * vr[c];
                }
                const float * orow = o.data() + ((size_t) t * nh + h) * D;
                for (int c = 0; c < D; ++c) {
                    const double r = acc[c] / sum;
                    err += (orow[c] - r) * (orow[c] - r); ref2 += r * r;
                    if (!std::isfinite(orow[c])) err += 1e30;
                }
            }
        }
        const double nmse = err / (ref2 > 0 ? ref2 : 1.0);
        const bool ok = nmse < 5e-3;   // fp16 Q / f16-converted K,V in the tile kernels
        printf("  %s %s: nmse %.3g\n", ok ? "ok  " : "FAIL", name, nmse);
        if (!ok) ++g_fail;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    // optional filter: test-fa-route-qkv <D> <type name> <width> [n_kv, default 1024]
    const int  fD = argc > 1 ? atoi(argv[1]) : 0;
    const char * fT = argc > 2 ? argv[2] : nullptr;
    const int  fW = argc > 3 ? atoi(argv[3]) : 0;
    const int  nkv = argc > 4 ? atoi(argv[4]) : 1024;
    // The reference dequantizes through type_traits.to_float, which on x86 reads the fp16 table
    // that the first ggml_init() of the process fills. run_case fills its reference before its own
    // ggml_init, so without this the first case's reference is all zeros (that was hive #268:
    // a test artefact, not an engine defect).
    {
        ggml_init_params ip0 = { 1024, nullptr, true };
        ggml_free(ggml_init(ip0));
    }
    if (ggml_backend_cuda_get_device_count() < 1) { printf("SKIP: no CUDA device\n"); return 0; }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { printf("SKIP: cuda init failed\n"); return 0; }
    const int cc = ggml_backend_cuda_get_device_cc(0);
    // On the pre-Turing cards this repo ships for, every case below must stay on the GPU.
    const bool pascal_volta = cc == 600 || cc == 610 || cc == 700;
    printf("test-fa-route-qkv: device 0 cc %d (%s)\n", cc, pascal_volta ? "must support every case" : "support not asserted");
    for (int D : { 256, 128 }) {
        for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1 }) {
            for (int nb : { 1, 4, 32 }) {
                if ((fD && fD != D) || (fT && strcmp(fT, ggml_type_name(t)) != 0) || (fW && fW != nb)) continue;
                run_case(be, D, t, nb, nkv, pascal_volta);
            }
        }
    }
    // Bug #235 residual: at f32 precision the tile-f16 kernel cannot run, so a head-256 node whose
    // K/V pair has no vec instance needs tile-f32 at head 256 to stay on a Pascal card.
    const bool pascal = cc == 600 || cc == 610;
    if (!fT || strcmp(fT, "f32prec") == 0) {
        for (ggml_type t : { GGML_TYPE_Q5_0, GGML_TYPE_Q4_1 }) {
            for (int nb : { 1, 32 }) {
                if (fD && fD != 256) continue;
                run_case(be, 256, t, nb, nkv, pascal, true);
            }
        }
    }
    ggml_backend_free(be);
    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}
