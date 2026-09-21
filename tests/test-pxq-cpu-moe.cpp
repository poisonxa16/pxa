// test-pxq-cpu-moe.cpp — graph-level integration test for the PXQ CPU fallback wiring in
// ggml.c (GGML_OP_MOE_FUSED_UP_GATE + GGML_OP_MUL_MAT_ID with PXQ weights on the CPU
// backend — the exact partial-offload situation that used to GGML_ABORT / segfault).
// NOT registered with CMake — standalone build against the CPU libggml:
//
//   c++ -O2 -std=c++17 -Iggml/include -Iggml/src tests/test-pxq-cpu-moe.cpp
//       -Lbuild-cpu/ggml/src -lggml -o /tmp/test-pxq-cpu-moe -lm   (one line)
//   LD_LIBRARY_PATH=build-cpu/ggml/src /tmp/test-pxq-cpu-moe
//
// Fills PXQ4 (4-bit tier) up / PXQ2 gate (mixed pair) + PXQ3 down expert tensors with random (sanitized)
// panel bytes, runs a real ggml graph (2 threads), and checks the result against a naive
// f32 evaluation built on pxa_pxq_dequant_2d.

#include "ggml.h"
#include "pxq-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

static size_t pxq_bytes(ggml_type t, int64_t R, int64_t K) {
    const int64_t KB = K/32;
    size_t ps;
    switch (t) {
        case GGML_TYPE_PXQ4:   ps = 128 + KB*1088; break;
        case GGML_TYPE_PXQ4HQ: ps = 128 + KB*1152; break;
        case GGML_TYPE_PXQ2:   ps = 128 + KB*576;  break;
        case GGML_TYPE_PXQ3:   ps = 128 + KB*832;  break;
        default: abort();
    }
    return (R/64)*ps;
}

static void fill_pxq(ggml_type t, uint8_t * dst, int64_t R, int64_t K, int64_t E, std::mt19937 & rng) {
    const size_t eb = pxq_bytes(t, R, K);
    for (size_t i = 0; i < eb*E; ++i) dst[i] = (uint8_t)(rng() & 0xff);
    // sanitize the fp16 anchor headers (finite, positive, some zeros)
    const size_t ps = eb/(R/64);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t p = 0; p < R/64; ++p) {
            uint16_t * anchors = (uint16_t *)(dst + e*eb + p*ps);
            for (int r = 0; r < 64; ++r) {
                uint16_t h = (uint16_t)(rng() & 0x7FFF);
                if ((h & 0x7C00) == 0x7C00) h = 0x3C00;
                if ((rng() & 15) == 0) h = 0;
                anchors[r] = h;
            }
        }
    }
    // keep magnitudes tame so silu/dot products stay in a comparable range: clamp anchors < 4
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t p = 0; p < R/64; ++p) {
            uint16_t * anchors = (uint16_t *)(dst + e*eb + p*ps);
            for (int r = 0; r < 64; ++r) if ((anchors[r] & 0x7C00) >= 0x4400) anchors[r] = (anchors[r] & 0x03FF) | 0x3C00;
        }
    }
}

int main() {
    printf("PXQ CPU fallback ggml-graph integration test\n");

    const int64_t K = 64, R = 64, E = 4, T = 3, n_ids = 2;

    struct ggml_init_params ip = { /*mem_size*/ 64u*1024*1024, /*mem_buffer*/ NULL, /*no_alloc*/ false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { printf("ggml_init failed\n"); return 1; }

    struct ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_PXQ4, K, R, E);
    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_PXQ2, K, R, E);  // mixed pair
    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_PXQ3, R, K, E);
    struct ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, T);

    // sanity: ggml sizes the PXQ tensors exactly like the panel math says
    check(ggml_nbytes(up)   == pxq_bytes(GGML_TYPE_PXQ4, R, K)*E, "ggml_nbytes(PXQ4 [64x64x4]) == panel bytes");
    check(ggml_nbytes(gate) == pxq_bytes(GGML_TYPE_PXQ2, R, K)*E, "ggml_nbytes(PXQ2 [64x64x4]) == panel bytes");
    check(ggml_nbytes(down) == pxq_bytes(GGML_TYPE_PXQ3, K, R)*E, "ggml_nbytes(PXQ3 [64x64x4]) == panel bytes");

    std::mt19937 rng(4242);
    fill_pxq(GGML_TYPE_PXQ4, (uint8_t *)up->data,   R, K, E, rng);
    fill_pxq(GGML_TYPE_PXQ2, (uint8_t *)gate->data, R, K, E, rng);
    fill_pxq(GGML_TYPE_PXQ3, (uint8_t *)down->data, K, R, E, rng);

    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (int64_t i = 0; i < K*T; ++i) ((float *)b->data)[i] = nd(rng);

    // routing: token 0 -> {0, 2}, token 1 -> {3, 0}, token 2 -> {1, -1} (-1 = SER hole)
    const int32_t route[T][n_ids] = { {0, 2}, {3, 0}, {1, -1} };
    memcpy(ids->data, route, sizeof(route));

    struct ggml_tensor * fused = ggml_moe_up_gate(ctx, up, gate, b, ids, GGML_UNARY_OP_SILU);
    struct ggml_tensor * out   = ggml_mul_mat_id(ctx, down, fused, ids);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 2);
    check(st == GGML_STATUS_SUCCESS, "graph compute (2 threads) returns success");

    // ---- naive reference via pxa_pxq_dequant_2d ---------------------------------------------
    std::vector<float> upf(E*R*K), gatef(E*R*K), downf(E*K*R);
    for (int64_t e = 0; e < E; ++e) {
        pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, (const uint8_t *)up->data   + e*pxq_bytes(GGML_TYPE_PXQ4, R, K), &upf[e*R*K],   R, K);
        pxa_pxq_dequant_2d(GGML_TYPE_PXQ2, (const uint8_t *)gate->data + e*pxq_bytes(GGML_TYPE_PXQ2, R, K), &gatef[e*R*K], R, K);
        pxa_pxq_dequant_2d(GGML_TYPE_PXQ3, (const uint8_t *)down->data + e*pxq_bytes(GGML_TYPE_PXQ3, K, R), &downf[e*K*R], K, R);
    }

    std::vector<float> fused_ref(R*n_ids*T, 0.0f), out_ref(K*n_ids*T, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        const float * x = (const float *)b->data + t*K;
        for (int64_t s = 0; s < n_ids; ++s) {
            const int e = route[t][s];
            if (e < 0) continue;
            float * fr = &fused_ref[s*R + t*R*n_ids];
            for (int64_t ix = 0; ix < R; ++ix) {
                double gd = 0, ud = 0;
                for (int64_t j = 0; j < K; ++j) {
                    gd += (double)gatef[e*R*K + ix*K + j]*(double)x[j];
                    ud += (double)upf[e*R*K + ix*K + j]*(double)x[j];
                }
                const float gv = (float)gd;
                fr[ix] = (float)ud * (gv/(1.0f + expf(-gv)));   // silu, limit == 0
            }
            float * orow = &out_ref[s*K + t*K*n_ids];
            for (int64_t ix = 0; ix < K; ++ix) {
                double acc = 0;
                for (int64_t j = 0; j < R; ++j) acc += (double)downf[e*K*R + ix*R + j]*(double)fr[j];
                orow[ix] = (float)acc;
            }
        }
    }

    auto maxrel = [](const float * a, const float * b, int64_t n) {
        double m = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double d = fabs((double)a[i] - (double)b[i]);
            const double r = d/(fabs((double)b[i]) + 1e-9);
            if (d > 1e-6 && r > m) m = r;
        }
        return m;
    };

    const double mr_f = maxrel((const float *)fused->data, fused_ref.data(), R*n_ids*T);
    char what[128];
    snprintf(what, sizeof(what), "MOE_FUSED_UP_GATE result vs naive (max rel %.2e)", mr_f);
    check(mr_f < 1e-5, what);

    const double mr_o = maxrel((const float *)out->data, out_ref.data(), K*n_ids*T);
    snprintf(what, sizeof(what), "MUL_MAT_ID (PXQ3 down) result vs naive (max rel %.2e)", mr_o);
    check(mr_o < 1e-5, what);

    // the SER hole (expert -1) must be zeroed, not garbage
    bool ser_ok = true;
    const float * fd = (const float *)fused->data;
    for (int64_t ix = 0; ix < R; ++ix) if (fd[1*R + 2*R*n_ids + ix] != 0.0f) ser_ok = false;
    check(ser_ok, "SER hole (expert -1) rows are zeroed");

    // ---- dense FUSED_UP_GATE (same-type pair) + plain dense MUL_MAT -------------------------
    {
        struct ggml_tensor * dup   = ggml_new_tensor_2d(ctx, GGML_TYPE_PXQ4, K, R);
        struct ggml_tensor * dgate = ggml_new_tensor_2d(ctx, GGML_TYPE_PXQ4, K, R);
        struct ggml_tensor * db    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        fill_pxq(GGML_TYPE_PXQ4, (uint8_t *)dup->data,   R, K, 1, rng);
        fill_pxq(GGML_TYPE_PXQ4, (uint8_t *)dgate->data, R, K, 1, rng);
        for (int64_t i = 0; i < K*T; ++i) ((float *)db->data)[i] = nd(rng);

        struct ggml_tensor * dfused = ggml_fused_up_gate(ctx, dup, dgate, db, GGML_UNARY_OP_GELU);
        check(dfused->op == GGML_OP_FUSED_UP_GATE, "dense pair fuses into GGML_OP_FUSED_UP_GATE");
        struct ggml_tensor * dmm = ggml_mul_mat(ctx, dup, db);   // plain dense MUL_MAT on PXQ
        struct ggml_cgraph * gf2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf2, dfused);
        ggml_build_forward_expand(gf2, dmm);
        check(ggml_graph_compute_with_ctx(ctx, gf2, 2) == GGML_STATUS_SUCCESS, "dense graph compute returns success");

        std::vector<float> duf(R*K), dgf(R*K);
        pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, dup->data,   duf.data(), R, K);
        pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, dgate->data, dgf.data(), R, K);
        std::vector<float> fref(T*R), mref(T*R);
        for (int64_t t = 0; t < T; ++t) {
            const float * x = (const float *)db->data + t*K;
            for (int64_t ix = 0; ix < R; ++ix) {
                double gd = 0, ud = 0;
                for (int64_t j = 0; j < K; ++j) {
                    gd += (double)dgf[ix*K + j]*(double)x[j];
                    ud += (double)duf[ix*K + j]*(double)x[j];
                }
                const float gv = (float)gd;
                const float act = 0.5f*gv*(1.0f + tanhf(0.79788456080286535587989211986876f*gv*(1.0f + 0.044715f*gv*gv)));
                fref[t*R + ix] = (float)ud*act;
                mref[t*R + ix] = (float)ud;
            }
        }
        const double mr1 = maxrel((const float *)dfused->data, fref.data(), T*R);
        char w2[128];
        snprintf(w2, sizeof(w2), "dense FUSED_UP_GATE vs naive (max rel %.2e)", mr1);
        check(mr1 < 1e-5, w2);
        const double mr2 = maxrel((const float *)dmm->data, mref.data(), T*R);
        snprintf(w2, sizeof(w2), "dense MUL_MAT vs naive (max rel %.2e)", mr2);
        check(mr2 < 1e-5, w2);
    }

    // ---- interleaved MoE (gate == NULL: first half rows gate, second half up) ---------------
    {
        const int64_t R2 = 128;   // 2 panels; halves are panel-aligned (64 rows each)
        struct ggml_tensor * iug = ggml_new_tensor_3d(ctx, GGML_TYPE_PXQ4, K, R2, E);
        fill_pxq(GGML_TYPE_PXQ4, (uint8_t *)iug->data, R2, K, E, rng);
        struct ggml_tensor * ifused = ggml_moe_up_gate(ctx, iug, NULL, b, ids, GGML_UNARY_OP_SILU);
        check(ifused->ne[0] == R2/2, "interleaved MoE dst has ne0 == rows/2");
        struct ggml_cgraph * gf3 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf3, ifused);
        check(ggml_graph_compute_with_ctx(ctx, gf3, 2) == GGML_STATUS_SUCCESS, "interleaved graph compute returns success");

        const size_t eb = pxq_bytes(GGML_TYPE_PXQ4, R2, K);
        std::vector<float> wf(R2*K), iref(R2/2*n_ids*T, 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            const float * x = (const float *)b->data + t*K;
            for (int64_t s = 0; s < n_ids; ++s) {
                const int e = route[t][s];
                if (e < 0) continue;
                pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, (const uint8_t *)iug->data + e*eb, wf.data(), R2, K);
                const float * gatef2 = wf.data();              // rows 0..R2/2-1
                const float * upf2   = wf.data() + (R2/2)*K;   // rows R2/2..R2-1
                float * fr = &iref[s*(R2/2) + t*(R2/2)*n_ids];
                for (int64_t ix = 0; ix < R2/2; ++ix) {
                    double gd = 0, ud = 0;
                    for (int64_t j = 0; j < K; ++j) {
                        gd += (double)gatef2[ix*K + j]*(double)x[j];
                        ud += (double)upf2[ix*K + j]*(double)x[j];
                    }
                    const float gv = (float)gd;
                    fr[ix] = (float)ud * (gv/(1.0f + expf(-gv)));
                }
            }
        }
        const double mr3 = maxrel((const float *)ifused->data, iref.data(), (R2/2)*n_ids*T);
        char w3[128];
        snprintf(w3, sizeof(w3), "interleaved MOE_FUSED_UP_GATE vs naive (max rel %.2e)", mr3);
        check(mr3 < 1e-5, w3);
    }

    ggml_free(ctx);
    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
