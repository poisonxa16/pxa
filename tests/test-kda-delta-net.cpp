// PXA_GLM5NEXT: GGML_OP_DELTA_NET with a per-channel forget gate (Kimi Delta Attention).
//
// GLM-5.3-Flash's KDA layers are the same delta rule as our Qwen3-Next / Qwen4Exp Gated
// DeltaNet except that the forget gate is a vector over the value channel rather than one
// scalar per head. This test pins that down two ways:
//
//   1. KDA vs an INDEPENDENT reference. The reference below is written in UPSTREAM's state
//      layout -- state[key][value], the layout llama.cpp PR #27773's
//      build_delta_net_autoregressive uses -- while our kernel works in state[value][key].
//      Agreeing therefore checks the transposition, not just the arithmetic.
//
//   2. GDN REGRESSION. With g_per_channel == 0 and the same numbers, the op must still
//      produce exactly what it produced before this feature existed, which the reference
//      reproduces by broadcasting one scalar across the channels. This is the guard that says
//      the Qwen4Exp path was not disturbed.
//
// CPU only, no GPU, no model: deterministic inputs from a fixed LCG.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t rng_state = 1234567u;
static float frand() { // deterministic, in [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float) (rng_state >> 8) / (float) (1u << 23)) - 1.0f;
}

// The caller-side q/k normalisation the DeltaNet op now relies on (see ggml.h's contract note
// on ggml_delta_net): the reference formula x / sqrt(sum(x^2) + eps), applied per row of S.
static void l2_norm_rows(std::vector<float> & x, int S, float eps) {
    for (size_t base = 0; base + (size_t) S <= x.size(); base += (size_t) S) {
        float sumsq = 0.0f;
        for (int i = 0; i < S; ++i) sumsq += x[base + i]*x[base + i];
        const float sc = 1.0f/std::sqrt(sumsq + eps);
        for (int i = 0; i < S; ++i) x[base + i] *= sc;
    }
}

// Reference delta rule in upstream's [key][value] state layout.
//   S'[k][v] = decay[v] * S[k][v]
//   d[v]     = beta * (V[v] - sum_k S'[k][v] * K_hat[k])
//   S''[k][v]= S'[k][v] + K_hat[k] * d[v]
//   O[v]     = sum_k S''[k][v] * Q_hat[k] * scale
// beta comes through a sigmoid and the state is clamped, both of which our op does
// internally. Q_hat/K_hat are the CALLER's already-L2-normalised q/k: as of PXA_KERNFIX
// 2026-09-09 (defect B) the op normalises nothing on either backend, so this test feeds it
// pre-normalised q/k exactly the way the production graphs do (src/llama-delta-net.cpp
// build_qkv, src/graphs/build_glm5next.cpp:228) and the oracle below does the same.
static void reference_delta_net(
        int S, int H, int T, int n_seqs, bool per_channel,
        const std::vector<float> & q,    // [S, T, H, n_seqs]
        const std::vector<float> & k,    // [S, T, H, n_seqs]
        const std::vector<float> & v,    // [S, T, H, n_seqs]
        const std::vector<float> & g,    // per_channel ? [S, T, H, n_seqs] : [T, 1, H, n_seqs]
        const std::vector<float> & beta, // [1, T, H, n_seqs]
        const std::vector<float> & s0,   // [S, S*H, 1, n_seqs], s0[value + key*S] per head
        std::vector<float> & out,        // [S, H, T, n_seqs]
        std::vector<float> & s1) {       // same layout as s0
    const float scale = 1.0f / std::sqrt((float) S);

    out.assign((size_t) S*H*T*n_seqs, 0.0f);
    s1.assign(s0.size(), 0.0f);

    std::vector<float> St((size_t) S*S); // St[key*S + value]

    for (int b = 0; b < n_seqs; ++b) {
        for (int h = 0; h < H; ++h) {
            // load: our layout is state[value + key*S] per head, transpose into [key][value]
            const float * s_in = s0.data() + (size_t) b*S*S*H + (size_t) h*S*S;
            for (int kk = 0; kk < S; ++kk) {
                for (int vv = 0; vv < S; ++vv) {
                    St[(size_t) kk*S + vv] = s_in[vv + kk*S];
                }
            }

            for (int t = 0; t < T; ++t) {
                const float * qt = q.data() + (size_t) b*S*T*H + (size_t) h*S*T + (size_t) t*S;
                const float * kt = k.data() + (size_t) b*S*T*H + (size_t) h*S*T + (size_t) t*S;
                const float * vt = v.data() + (size_t) b*S*T*H + (size_t) h*S*T + (size_t) t*S;

                // q/k are pre-normalised by the caller (see the header note); the op and this
                // oracle both consume them as-is.
                const float qn = 1.0f, kn = 1.0f;

                const float braw = beta[(size_t) b*T*H + (size_t) h*T + t];
                const float bval = 1.0f/(1.0f + std::exp(-braw));

                std::vector<float> decay(S);
                if (per_channel) {
                    const float * gt = g.data() + (size_t) b*S*T*H + (size_t) h*S*T + (size_t) t*S;
                    for (int i = 0; i < S; ++i) decay[i] = std::exp(std::fmin(gt[i], 50.0f));
                } else {
                    const float gv = std::exp(std::fmin(g[(size_t) b*T*H + (size_t) h*T + t], 50.0f));
                    for (int i = 0; i < S; ++i) decay[i] = gv;
                }

                // S' = decay * S. The KDA forget gate is indexed by the KEY channel: upstream
                // llama.cpp PR #27773's ggml_compute_forward_gated_delta_net_one_chunk() does
                // S[i][j] *= exp(g[i]) with i = key, j = value, and fla's chunk_kda carries g
                // with the same (B,T,H,K) shape as k. (Upstream's own build_delta_net_autoregressive
                // fallback decays by the VALUE channel instead; the engine never takes that path --
                // fused_gdn_ar/ch are on -- and following it is what made our first port diverge
                // from the reference build at the second token of every KDA layer.)
                for (int kk = 0; kk < S; ++kk)
                    for (int vv = 0; vv < S; ++vv)
                        St[(size_t) kk*S + vv] *= decay[kk];

                // d = beta * (V - S'^T K_hat)
                std::vector<float> d(S);
                for (int vv = 0; vv < S; ++vv) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < S; ++kk) acc += St[(size_t) kk*S + vv] * (kt[kk]*kn);
                    d[vv] = bval * (vt[vv] - acc);
                }

                // S'' = S' + K_hat d^T, clamped exactly like the kernel does
                for (int kk = 0; kk < S; ++kk) {
                    const float kh = kt[kk]*kn;
                    for (int vv = 0; vv < S; ++vv) {
                        float x = St[(size_t) kk*S + vv] + kh*d[vv];
                        St[(size_t) kk*S + vv] = std::fmin(std::fmax(x, -1e6f), 1e6f);
                    }
                }

                // O = S''^T Q_hat * scale
                float * ot = out.data() + (size_t) b*S*H*T + (size_t) t*S*H + (size_t) h*S;
                for (int vv = 0; vv < S; ++vv) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < S; ++kk) acc += St[(size_t) kk*S + vv] * (qt[kk]*qn*scale);
                    ot[vv] = acc;
                }
            }

            float * s_out = s1.data() + (size_t) b*S*S*H + (size_t) h*S*S;
            for (int kk = 0; kk < S; ++kk)
                for (int vv = 0; vv < S; ++vv)
                    s_out[vv + kk*S] = St[(size_t) kk*S + vv];
        }
    }
}

static bool run_case(int S, int H, int T, int n_seqs, bool per_channel) {
    const char * name = per_channel ? "KDA (per-channel gate)" : "GDN (scalar gate)";
    printf("  %-24s S=%d H=%d T=%d n_seqs=%d ... ", name, S, H, T, n_seqs);
    fflush(stdout);

    rng_state = 987654321u + (per_channel ? 7u : 0u);

    const size_t n_qkv = (size_t) S*T*H*n_seqs;
    std::vector<float> hq(n_qkv), hk(n_qkv), hv(n_qkv);
    std::vector<float> hg(per_channel ? n_qkv : (size_t) T*H*n_seqs);
    std::vector<float> hb((size_t) T*H*n_seqs);
    std::vector<float> hs((size_t) S*S*H*n_seqs);

    for (auto & x : hq) x = frand();
    for (auto & x : hk) x = frand();
    for (auto & x : hv) x = frand();
    // Production pre-normalisation of q/k: the reference GDN L2 norm x/sqrt(sum(x^2)+eps),
    // which is exactly what ggml_l2_norm(ctx, x, 1e-12f) computes in the real graphs. Rows of
    // length S are contiguous in this [S, T, H, n_seqs] layout.
    l2_norm_rows(hq, S, 1e-12f);
    l2_norm_rows(hk, S, 1e-12f);
    // log-decay: keep it negative so exp() lands in (0,1], which is the real regime
    for (auto & x : hg) x = -0.5f*(frand() + 1.0f) - 0.05f;
    for (auto & x : hb) x = frand();
    for (auto & x : hs) x = 0.1f*frand();

    struct ggml_init_params ip = { 1024ull*1024ull*1024ull, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { printf("FAIL (ggml_init)\n"); return false; }

    // q/k/v: [S, T, H, n_seqs]
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
    // g: KDA [S, T, H, n_seqs]; GDN [T, 1, H, n_seqs]
    ggml_tensor * g = per_channel
        ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs)
        : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, 1, H, n_seqs);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, H, n_seqs);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, n_seqs);

    memcpy(q->data, hq.data(), ggml_nbytes(q));
    memcpy(k->data, hk.data(), ggml_nbytes(k));
    memcpy(v->data, hv.data(), ggml_nbytes(v));
    memcpy(g->data, hg.data(), ggml_nbytes(g));
    memcpy(b->data, hb.data(), ggml_nbytes(b));
    memcpy(s->data, hs.data(), ggml_nbytes(s));

    ggml_tensor * r = ggml_delta_net_ext(ctx, q, k, v, g, b, s, nullptr, per_channel ? 1 : 0);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::vector<float> ref_out, ref_state;
    reference_delta_net(S, H, T, n_seqs, per_channel, hq, hk, hv, hg, hb, hs, ref_out, ref_state);

    const float * got = (const float *) r->data;
    const size_t n_out = (size_t) S*H*T*n_seqs;

    double max_out = 0.0, max_state = 0.0;
    for (size_t i = 0; i < n_out; ++i) {
        max_out = std::fmax(max_out, std::fabs((double) got[i] - ref_out[i]));
    }
    for (size_t i = 0; i < ref_state.size(); ++i) {
        max_state = std::fmax(max_state, std::fabs((double) got[n_out + i] - ref_state[i]));
    }

    ggml_free(ctx);

    const double tol = 2e-4;
    const bool ok = max_out < tol && max_state < tol;
    printf("%s  (max |d| out %.3e, state %.3e)\n", ok ? "OK" : "FAIL", max_out, max_state);
    return ok;
}

int main() {
    printf("test-kda-delta-net: GGML_OP_DELTA_NET forget-gate variants\n");
    bool ok = true;

    // decode-shaped and prefill-shaped, one and several sequences
    ok &= run_case(64, 2,  1, 1, true);
    ok &= run_case(64, 2,  7, 1, true);
    ok &= run_case(64, 3,  5, 2, true);
    ok &= run_case(128, 2, 4, 1, true);   // the real GLM-5.3-Flash head width

    // the scalar path must be untouched
    ok &= run_case(64, 2,  1, 1, false);
    ok &= run_case(64, 2,  7, 1, false);
    ok &= run_case(128, 2, 4, 1, false);

    printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
