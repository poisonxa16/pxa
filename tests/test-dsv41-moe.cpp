// test-dsv41-moe -- G5 for DeepSeek-V4.1-Flash: 384 routed experts, top-6, sqrtsoftplus
// scoring with the selection bias, normalised weights x 1.5, then the routed SwiGLU experts
// through mul_mat_id. The graph is the node chain llm_build_moe_ffn emits for the
// sqrtsoftplus / norm_w / scale_w case (logits at F32 precision -> sqrt(softplus) -> + bias ->
// top_k -> get_rows of the UNBIASED probs -> sum_rows/div -> scale; up/gate/down mul_mat_id).
// The expected answer is computed here in double precision.
//
//   test-dsv41-moe            CPU backend, and CUDA device 0 when built with CUDA and one is visible
//   test-dsv41-moe cpu        CPU only
//
// Checks per backend, expert type (F32, Q8_0) and token count (1, 5, 33):
//   ids     the selected expert set equals the reference top-6 (a swap is accepted only when the
//           two reference scores are within 1e-6, i.e. a float-vs-double near tie)
//   weights relative error <= 1e-5, and they sum to 1.5
//   experts per-slot SwiGLU expert outputs vs the reference using the backend's own ids,
//           NMSE <= 1e-10 (F32) or <= 2e-3 (Q8_0 weights; reference uses the dequantised weights)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static const int N_EXP  = 384;
static const int TOP_K  = 6;
static const int N_EMBD = 64;
static const int N_FF   = 32;
static const float W_SCALE = 1.5f;

static int g_fail = 0;

static float frand(unsigned & s) {
    s = s * 1664525u + 1013904223u;
    return (float) (s >> 8) / (float) (1u << 24) * 2.0f - 1.0f;
}

static double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

// expert weights as stored (F32 or Q8_0 bytes) and as the reference sees them (dequantised)
struct expert_tensor {
    std::vector<uint8_t> bytes;
    std::vector<float>   deq;
};

static expert_tensor make_experts(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, unsigned seed, float amp) {
    expert_tensor e;
    std::vector<float> f((size_t) (ne0 * ne1 * ne2));
    for (auto & v : f) { v = amp * frand(seed); }
    if (type == GGML_TYPE_F32) {
        e.bytes.resize(f.size() * sizeof(float));
        memcpy(e.bytes.data(), f.data(), e.bytes.size());
        e.deq = f;
        return e;
    }
    const auto tt = ggml_internal_get_type_traits(type);
    const size_t row_size = ggml_row_size(type, ne0);
    e.bytes.resize(row_size * (size_t) (ne1 * ne2));
    e.deq.resize(f.size());
    for (int64_t r = 0; r < ne1 * ne2; r++) {
        tt.from_float_ref(f.data() + r * ne0, e.bytes.data() + r * row_size, ne0);
        tt.to_float(e.bytes.data() + r * row_size, e.deq.data() + r * ne0, ne0);
    }
    return e;
}

static void run_case(ggml_backend_t be, const char * be_name, ggml_type etype, int n_tok, unsigned seed) {
    // ---- inputs ----
    std::vector<float> x((size_t) N_EMBD * n_tok), g((size_t) N_EMBD * N_EXP), bias(N_EXP);
    for (auto & v : x)    { v = frand(seed); }
    for (auto & v : g)    { v = 0.35f * frand(seed); }
    for (auto & v : bias) { v = 0.05f * frand(seed); }
    expert_tensor up   = make_experts(etype, N_EMBD, N_FF, N_EXP, seed + 1, 0.2f);
    expert_tensor gate = make_experts(etype, N_EMBD, N_FF, N_EXP, seed + 2, 0.2f);
    expert_tensor down = make_experts(etype, N_FF, N_EMBD, N_EXP, seed + 3, 0.2f);

    // ---- graph ----
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tx    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n_tok);
    ggml_tensor * tg    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_EXP);
    ggml_tensor * tb    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EXP);
    ggml_tensor * tup   = ggml_new_tensor_3d(ctx, etype, N_EMBD, N_FF, N_EXP);
    ggml_tensor * tgate = ggml_new_tensor_3d(ctx, etype, N_EMBD, N_FF, N_EXP);
    ggml_tensor * tdown = ggml_new_tensor_3d(ctx, etype, N_FF, N_EMBD, N_EXP);

    ggml_tensor * logits = ggml_mul_mat(ctx, tg, tx);                  // [n_exp, n_tok]
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_tensor * probs  = ggml_sqrt(ctx, ggml_softplus(ctx, logits));  // [n_exp, n_tok]
    ggml_tensor * selp   = ggml_add(ctx, probs, tb);
    ggml_tensor * sel    = ggml_top_k(ctx, selp, TOP_K);                // [k, n_tok]
    ggml_tensor * w      = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, N_EXP, n_tok), sel); // [1, k, n_tok]
    w = ggml_reshape_2d(ctx, w, TOP_K, n_tok);
    w = ggml_div(ctx, w, ggml_sum_rows(ctx, w));
    w = ggml_scale(ctx, w, W_SCALE);                                    // [k, n_tok]

    ggml_tensor * cur  = ggml_reshape_3d(ctx, tx, N_EMBD, 1, n_tok);
    ggml_tensor * u    = ggml_mul_mat_id(ctx, tup,   cur, sel);         // [n_ff, k, n_tok]
    ggml_tensor * gt   = ggml_mul_mat_id(ctx, tgate, cur, sel);
    ggml_tensor * act  = ggml_mul(ctx, ggml_silu(ctx, gt), u);
    ggml_tensor * outs = ggml_mul_mat_id(ctx, tdown, act, sel);         // [n_embd, k, n_tok]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, w);
    ggml_build_forward_expand(gf, outs);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(tx, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(tg, g.data(), 0, g.size() * sizeof(float));
    ggml_backend_tensor_set(tb, bias.data(), 0, bias.size() * sizeof(float));
    ggml_backend_tensor_set(tup,   up.bytes.data(),   0, up.bytes.size());
    ggml_backend_tensor_set(tgate, gate.bytes.data(), 0, gate.bytes.size());
    ggml_backend_tensor_set(tdown, down.bytes.data(), 0, down.bytes.size());

    char name[128];
    snprintf(name, sizeof(name), "%s %s n_tok=%d", be_name, ggml_type_name(etype), n_tok);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  FAIL %s: graph_compute\n", name);
        ++g_fail;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return;
    }

    // top_k is argsort viewed to k: read through the view's own strides
    std::vector<int32_t> ids((size_t) TOP_K * n_tok);
    for (int t = 0; t < n_tok; t++) {
        ggml_backend_tensor_get(sel, ids.data() + (size_t) t * TOP_K, t * sel->nb[1], TOP_K * sizeof(int32_t));
    }
    std::vector<float> wv((size_t) TOP_K * n_tok), ov((size_t) N_EMBD * TOP_K * n_tok);
    ggml_backend_tensor_get(w, wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_get(outs, ov.data(), 0, ov.size() * sizeof(float));

    int bad_ids = 0, near_ties = 0, bad_w = 0;
    double err2 = 0.0, ref2 = 0.0;
    for (int t = 0; t < n_tok; t++) {
        const float * xt = x.data() + (size_t) t * N_EMBD;
        std::vector<double> p(N_EXP), s(N_EXP);
        for (int e = 0; e < N_EXP; e++) {
            double l = 0.0;
            for (int i = 0; i < N_EMBD; i++) { l += (double) g[(size_t) e * N_EMBD + i] * xt[i]; }
            p[e] = std::sqrt(softplus(l));
            s[e] = p[e] + bias[e];
        }
        std::vector<int> order(N_EXP);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s[a] > s[b]; });
        std::vector<int> want(order.begin(), order.begin() + TOP_K);
        std::vector<int> got(ids.begin() + (size_t) t * TOP_K, ids.begin() + (size_t) (t + 1) * TOP_K);
        std::vector<int> ws = want, gs = got;
        std::sort(ws.begin(), ws.end());
        std::sort(gs.begin(), gs.end());
        if (ws != gs) {
            // accept only a near tie at the k-th place (float vs double scores)
            bool tie = true;
            for (int e : gs) {
                if (e < 0 || e >= N_EXP) { tie = false; break; }
                if (!std::binary_search(ws.begin(), ws.end(), e) && std::fabs(s[e] - s[want[TOP_K - 1]]) > 1e-6) { tie = false; }
            }
            if (tie) { near_ties++; } else {
                if (bad_ids < 3) { printf("  FAIL %s: tok %d expert set differs\n", name, t); }
                bad_ids++;
                continue;
            }
        }
        // weights for the backend's ids, from the unbiased probs
        double psum = 0.0;
        for (int j = 0; j < TOP_K; j++) { psum += p[got[j]]; }
        double wsum = 0.0;
        for (int j = 0; j < TOP_K; j++) {
            const double ref = p[got[j]] / psum * W_SCALE;
            const double gv  = wv[(size_t) t * TOP_K + j];
            wsum += gv;
            if (std::fabs(gv - ref) > 1e-5 * std::fabs(ref)) { bad_w++; }
        }
        if (std::fabs(wsum - W_SCALE) > 1e-4) { bad_w++; }
        // expert outputs per slot
        for (int j = 0; j < TOP_K; j++) {
            const int e = got[j];
            std::vector<double> a(N_FF);
            for (int f = 0; f < N_FF; f++) {
                double uu = 0.0, gg = 0.0;
                for (int i = 0; i < N_EMBD; i++) {
                    uu += (double) up.deq[((size_t) e * N_FF + f) * N_EMBD + i] * xt[i];
                    gg += (double) gate.deq[((size_t) e * N_FF + f) * N_EMBD + i] * xt[i];
                }
                a[f] = gg / (1.0 + std::exp(-gg)) * uu;
            }
            for (int o = 0; o < N_EMBD; o++) {
                double r = 0.0;
                for (int f = 0; f < N_FF; f++) { r += (double) down.deq[((size_t) e * N_EMBD + o) * N_FF + f] * a[f]; }
                const double d = ov[((size_t) t * TOP_K + j) * N_EMBD + o] - r;
                err2 += d * d;
                ref2 += r * r;
            }
        }
    }
    const double nmse = ref2 > 0 ? err2 / ref2 : err2;
    const double tol  = etype == GGML_TYPE_F32 ? 1e-10 : 2e-3;
    const bool ok = bad_ids == 0 && bad_w == 0 && nmse <= tol;
    printf("  %s %s: ids ok%s, weights %s, experts nmse %.3g (tol %.0g)\n", ok ? "ok  " : "FAIL", name,
           near_ties ? (" (" + std::to_string(near_ties) + " near ties)").c_str() : "",
           bad_w ? "BAD" : "ok", nmse, tol);
    g_fail += ok ? 0 : 1;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void run_backend(ggml_backend_t be, const char * be_name) {
    unsigned seed = 4141u;
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_Q8_0 }) {
        for (int n_tok : { 1, 5, 33 }) {
            run_case(be, be_name, t, n_tok, seed);
            seed += 977u;
        }
    }
}

int main(int argc, char ** argv) {
    const bool cpu_only = argc > 1 && std::string(argv[1]) == "cpu";
    printf("test-dsv41-moe: %d experts, top-%d, sqrtsoftplus, norm x %.1f\n", N_EXP, TOP_K, (double) W_SCALE);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    run_backend(cpu, "CPU");
    ggml_backend_free(cpu);

#ifdef GGML_USE_CUDA
    if (!cpu_only) {
        if (ggml_backend_cuda_get_device_count() > 0) {
            ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
            if (cuda) {
                run_backend(cuda, "CUDA0");
                ggml_backend_free(cuda);
            }
        } else {
            printf("  (no CUDA device visible: CUDA part skipped)\n");
        }
    }
#else
    (void) cpu_only;
#endif

    printf("test-dsv41-moe: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
