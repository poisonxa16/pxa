// test-spec-rollback-cost.cpp -- recurrent-state rollback on a rejected draft: per-step CHECKPOINT vs REPLAY, on the GPU.
//
// A hybrid (Gated-DeltaNet) model cannot rewind its recurrent state by dropping KV rows: after a verify of T tokens of which
// k+1 are accepted, the state must be the one after token k, not after token T-1. Two exact ways to get there:
//
//   CHECKPOINT  (per-step) the verify writes the state after every step into its own row; a rejection copies row k back
//               into the live state. VRAM: T-1 state rows per layer (144 MiB a row over the 27B's 48 layers).
//   REPLAY      (PXA_SPEC_CKPT_REPLAY, 2026-10-14) the verify writes the state it STARTS from plus every token's inputs
//               (q, k, v, g, beta) into a pack (ggml_delta_net_set_replay_capture); a rejection re-runs the SAME kernel
//               over tokens 0..k from that start state (ggml_backend_cuda_delta_net_replay). VRAM: one state row + T rows
//               of inputs (~1.9 MiB a token over 48 layers), whatever the depth.
//
// This test runs the real CUDA DELTA_NET op at the 27B's per-layer shape (48 value heads, 16 key heads, head dim 128) for
// 200 random rounds: verify width T in 2..9 (both block-size classes: <= 8 and > 8), both GQA head mappings (repeat_type
// 0/1), random inputs and start state. Per round it checks, BITWISE:
//   (1) the replay-capture verify computes exactly the per-step verify's output and final state (capturing changes nothing);
//   (2) for every k in 0..T-1 the replay over k+1 tokens equals per-step row k (k < T-1) or the final state (k = T-1).
// Then it times, on the real 48-layer shape at T = 5 (MTP n_max 4), the per-step verify vs the replay-capture verify and the
// 48-layer replay for k = 0..3 (what a rejected round adds), and prints the table.
//
// Usage: test-spec-rollback-cost [device]   (exit 0 = bit-exact everywhere; no CUDA device -> skipped, exit 0)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef GGML_USE_CUDA
int main() {
    printf("test-spec-rollback-cost: built without CUDA; skipped\n");
    return 0;
}
#else

extern "C" void ggml_delta_net_set_replay_capture(struct ggml_tensor * a, struct ggml_tensor * pack);
extern "C" GGML_CALL bool ggml_backend_cuda_delta_net_replay(ggml_backend_t backend, int n, const void * descs);

// mirrors pxa_dn_replay_desc (ggml/src/ggml-cuda/pxa/delta-net.cuh) and pxa_dn_replay_desc_host (src/llama.cpp)
struct replay_desc {
    const float * pack;
    float       * state_dst;
    float       * scratch;
    int64_t       head_dim;
    int64_t       n_heads;
    int64_t       n_heads_kq;
    int32_t       repeat_type;
    int32_t       n_verify;
    int32_t       n_replay;
    int32_t       pad;
};

static const int64_t HD = 128, H = 48, HK = 16;          // Qwen3.8-27B DeltaNet layer
static const int64_t S  = HD * HD * H;                    // state floats per layer (3 MiB)

static int64_t pack_elems(int64_t T) { return S + T * (2 * HD * HK + HD * H + 2 * H); }

struct round_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> q, k, v_raw, g_raw, b_raw, st, saved, pack, n_ps, n_rp;
    ggml_tensor * rstate = nullptr;   // replay destination
    ggml_tensor * scratch = nullptr;  // replay attention output (dropped)
    ggml_cgraph * gf_ps = nullptr;
    ggml_cgraph * gf_rp = nullptr;
};

// L layers of one verify of width T, each with its own inputs; per-step nodes and replay-capture nodes over the SAME inputs
static void build(round_graph & R, int L, int64_t T, int repeat_type, ggml_backend_t be) {
    ggml_init_params ip = { (size_t) (L * 32 + 64) * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(16 * L + 16, false), nullptr, true };
    R.ctx = ggml_init(ip);
    ggml_context * c = R.ctx;
    R.gf_ps = ggml_new_graph_custom(c, 16 * L + 16, false);
    R.gf_rp = ggml_new_graph_custom(c, 16 * L + 16, false);
    for (int l = 0; l < L; ++l) {
        ggml_tensor * q  = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, T, HK, 1);
        ggml_tensor * k  = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, T, HK, 1);
        ggml_tensor * vr = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, H, T, 1);
        ggml_tensor * gr = ggml_new_tensor_3d(c, GGML_TYPE_F32, H, T, 1);
        ggml_tensor * br = ggml_new_tensor_4d(c, GGML_TYPE_F32, H, 1, T, 1);
        ggml_tensor * st = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, HD * H, 1, 1);
        ggml_tensor * sv = ggml_new_tensor_1d(c, GGML_TYPE_F32, std::max<int64_t>(1, (T - 1) * S));
        ggml_tensor * pk = ggml_new_tensor_1d(c, GGML_TYPE_F32, pack_elems(T));
        // the builder's views (llama-delta-net.cpp build_fused_delta_net)
        ggml_tensor * v = ggml_permute(c, vr, 0, 2, 1, 3);
        ggml_tensor * g = ggml_permute(c, gr, 2, 0, 3, 1);
        ggml_tensor * b = ggml_permute(c, br, 2, 0, 1, 3);
        ggml_tensor * nps = ggml_delta_net(c, q, k, v, g, b, st, T > 1 ? sv : nullptr);
        nps->op_params[0] = repeat_type;
        ggml_tensor * nrp = ggml_delta_net(c, q, k, v, g, b, st, nullptr);
        nrp->op_params[0] = repeat_type;
        ggml_delta_net_set_replay_capture(nrp, pk);
        ggml_build_forward_expand(R.gf_ps, nps);
        ggml_build_forward_expand(R.gf_rp, nrp);
        R.q.push_back(q); R.k.push_back(k); R.v_raw.push_back(vr); R.g_raw.push_back(gr); R.b_raw.push_back(br);
        R.st.push_back(st); R.saved.push_back(sv); R.pack.push_back(pk); R.n_ps.push_back(nps); R.n_rp.push_back(nrp);
    }
    R.rstate  = ggml_new_tensor_1d(c, GGML_TYPE_F32, S);
    R.scratch = ggml_new_tensor_1d(c, GGML_TYPE_F32, HD * H * T);
    R.buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!R.buf) { fprintf(stderr, "alloc failed\n"); exit(2); }
}

static void release(round_graph & R) {
    if (R.buf) ggml_backend_buffer_free(R.buf);
    if (R.ctx) ggml_free(R.ctx);
    R = round_graph();
}

static void fill(std::mt19937 & rng, round_graph & R, int l, int64_t T) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-4.0f, 0.0f);
    auto set = [&](ggml_tensor * t, const std::vector<float> & x) { ggml_backend_tensor_set(t, x.data(), 0, x.size() * sizeof(float)); };
    std::vector<float> q(HD * T * HK), k(HD * T * HK);
    for (auto & x : q) x = nd(rng);
    for (auto & x : k) x = nd(rng);
    for (int64_t r = 0; r < T * HK; ++r) {   // l2-normalised rows, as the graph feeds them
        double sq = 0, sk = 0;
        for (int64_t d = 0; d < HD; ++d) { sq += q[r*HD+d]*q[r*HD+d]; sk += k[r*HD+d]*k[r*HD+d]; }
        for (int64_t d = 0; d < HD; ++d) { q[r*HD+d] /= (float) std::sqrt(sq); k[r*HD+d] /= (float) std::sqrt(sk); }
    }
    std::vector<float> v(HD * H * T), g(H * T), b(H * T), st(S);
    for (auto & x : v) x = nd(rng);
    for (auto & x : g) x = ud(rng);
    for (auto & x : b) x = nd(rng);
    for (auto & x : st) x = 0.05f * nd(rng);
    set(R.q[l], q); set(R.k[l], k); set(R.v_raw[l], v); set(R.g_raw[l], g); set(R.b_raw[l], b); set(R.st[l], st);
}

static std::vector<float> get(ggml_tensor * t, size_t off_elems, size_t n) {
    std::vector<float> x(n);
    ggml_backend_tensor_get(t, x.data(), off_elems * sizeof(float), n * sizeof(float));
    return x;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static replay_desc desc(round_graph & R, int l, int64_t T, int repeat_type, int n_replay) {
    replay_desc d;
    d.pack = (const float *) R.pack[l]->data;
    d.state_dst = (float *) R.rstate->data;
    d.scratch = (float *) R.scratch->data;
    d.head_dim = HD; d.n_heads = H; d.n_heads_kq = HK;
    d.repeat_type = repeat_type; d.n_verify = (int32_t) T; d.n_replay = n_replay; d.pad = 0;
    return d;
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    if (ggml_backend_cuda_get_device_count() <= dev) {
        printf("test-spec-rollback-cost: no CUDA device %d; skipped\n", dev);
        return 0;
    }
    ggml_backend_t be = ggml_backend_cuda_init(dev, nullptr);
    if (!be) { printf("cuda init failed\n"); return 2; }

    std::mt19937 rng(20261014);
    int fails = 0, checks = 0, rounds = 0;
    const int ROUNDS = 200;
    for (int it = 0; it < ROUNDS; ++it) {
        const int64_t T = 2 + (int64_t) (rng() % 8);       // verify width 2..9 (n_max 1..8)
        const int rt = (int) (rng() % 2);                   // both key-head mappings
        round_graph R;
        build(R, 1, T, rt, be);
        fill(rng, R, 0, T);
        ggml_backend_graph_compute(be, R.gf_ps);
        ggml_backend_graph_compute(be, R.gf_rp);
        ggml_backend_synchronize(be);
        const size_t OUT = (size_t) (HD * H * T);
        const auto out_ps = get(R.n_ps[0], 0, OUT), out_rp = get(R.n_rp[0], 0, OUT);
        const auto fin_ps = get(R.n_ps[0], OUT, S), fin_rp = get(R.n_rp[0], OUT, S);
        ++checks;
        if (!same(out_ps, out_rp) || !same(fin_ps, fin_rp)) {
            ++fails;
            printf("  FAIL round %d T=%lld rt=%d: replay-capture verify differs from the per-step verify\n", it, (long long) T, rt);
        }
        const auto pre = get(R.pack[0], 0, S), st0 = get(R.st[0], 0, S);
        ++checks;
        if (!same(pre, st0)) { ++fails; printf("  FAIL round %d: captured start state != input state\n", it); }
        for (int kk = 0; kk < T; ++kk) {
            const replay_desc d = desc(R, 0, T, rt, kk + 1);
            if (!ggml_backend_cuda_delta_net_replay(be, 1, &d)) { printf("replay call failed\n"); return 2; }
            ggml_backend_synchronize(be);
            const auto got  = get(R.rstate, 0, S);
            const auto want = kk + 1 < T ? get(R.saved[0], (size_t) kk * S, S) : fin_ps;
            ++checks;
            if (!same(got, want)) {
                ++fails;
                size_t nd = 0; float mx = 0;
                for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) { ++nd; mx = std::max(mx, std::fabs(got[i] - want[i])); }
                printf("  FAIL round %d T=%lld rt=%d k=%d: replay != per-step row (%zu of %zu differ, max |d| %g)\n",
                        it, (long long) T, rt, kk, nd, got.size(), mx);
            }
        }
        release(R);
        ++rounds;
    }
    printf("bit-exact: %d rounds, %d checks, %d failures (T 2..9, k 0..T-1, repeat_type 0/1)\n", rounds, checks, fails);

    // cost on the real 48-layer shape at T = 5 (MTP n_max 4)
    {
        const int L = 48; const int64_t T = 5; const int rt = 1;
        round_graph R;
        build(R, L, T, rt, be);
        for (int l = 0; l < L; ++l) fill(rng, R, l, T);
        auto time_graph = [&](ggml_cgraph * gf) {
            double best = 1e30;
            for (int r = 0; r < 12; ++r) {
                ggml_backend_synchronize(be);
                const auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            return best;
        };
        const double t_ps = time_graph(R.gf_ps);
        const double t_rp = time_graph(R.gf_rp);
        printf("48 layers, verify T=%lld: per-step verify %.3f ms (writes %lld state rows), replay-capture verify %.3f ms (1 start state + inputs)\n",
               (long long) T, t_ps, (long long) (T - 1), t_rp);
        for (int kk = 0; kk + 1 < T; ++kk) {
            std::vector<replay_desc> ds;
            for (int l = 0; l < L; ++l) ds.push_back(desc(R, l, T, rt, kk + 1));
            double best = 1e30;
            for (int r = 0; r < 12; ++r) {
                ggml_backend_synchronize(be);
                const auto t0 = std::chrono::steady_clock::now();
                ggml_backend_cuda_delta_net_replay(be, L, ds.data());
                ggml_backend_synchronize(be);
                best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            printf("  replay k=%d (%d tokens x 48 layers, host+GPU wall): %.3f ms\n", kk, kk + 1, best);
        }
        printf("VRAM per chain at T=%lld: per-step %.1f MiB of state rows vs replay %.1f MiB of packs (+ the same conv rows)\n",
               (long long) T, (T - 1) * S * 4.0 * L / 1048576.0, pack_elems(T) * 4.0 * L / 1048576.0);
        release(R);
    }
    ggml_backend_free(be);
    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
#endif
