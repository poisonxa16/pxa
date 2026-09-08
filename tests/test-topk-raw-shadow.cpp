// PXA_TOPK_RAW correctness harness (CPU only). Loads a tiny model, decodes one token so a real
// logits row exists, then overwrites that row with synthetic distributions (gaussian, injected
// exact ties at the top and at the K boundary, -inf holes) and samples through common_sampler_sample
// under a matrix of sampler chains. Run it three ways:
//   PXA_TOPK_RAW=3 test-topk-raw-shadow model.gguf   -> aborts on the first genuine mismatch (the gate)
//   PXA_TOPK_RAW=0 ... > a ; PXA_TOPK_RAW=1 ... > b ; diff a b
//       the per-config "notie" hash (iterations without injected ties) must be identical, the
//       "all" hash may differ only where ties were injected.
// The last config decodes for real (feeds the sampled token back) instead of using synthetic rows.
#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <functional>

static uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); return h; }

struct cfg {
    std::string name;
    std::function<void(common_params_sampling &)> set;
    bool server_bias = false;
    bool real_decode = false;
};

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_iters]\n", argv[0]); return 2; }
    const int n_iters = argc > 2 ? atoi(argv[2]) : 400;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 32; cp.n_ubatch = 32; cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_set_rng_seed(ctx, 42);   // the context rng feeds XTC; seed it so mode 0 and mode 1 runs are comparable
    const int n_vocab = llama_n_vocab(model);
    const char * mode = getenv("PXA_TOPK_RAW");
    printf("model=%s n_vocab=%d PXA_TOPK_RAW=%s n_iters=%d\n", argv[1], n_vocab, mode ? mode : "(unset)", n_iters);

    auto decode_one = [&](llama_token tok, llama_pos pos) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
        const int ret = llama_decode(ctx, b);
        llama_batch_free(b);
        return ret;
    };
    if (decode_one(1 % n_vocab, 0) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    std::vector<cfg> cfgs = {
        {"defaults_topk40",             [](common_params_sampling & p) { p.top_k = 40; }},
        {"topk40_t0.7_p1_m0",           [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.7f; p.top_p = 1.0f; p.min_p = 0.0f; }},
        {"topk40_p0.9_m0.05_t0.8",      [](common_params_sampling & p) { p.top_k = 40; p.top_p = 0.9f; p.min_p = 0.05f; p.temp = 0.8f; }},
        {"topk1_t1",                    [](common_params_sampling & p) { p.top_k = 1; p.temp = 1.0f; }},
        {"topk128_p0.5_t1.2",           [](common_params_sampling & p) { p.top_k = 128; p.top_p = 0.5f; p.min_p = 0.0f; p.temp = 1.2f; }},
        {"topk100_m0.2_t1",             [](common_params_sampling & p) { p.top_k = 100; p.top_p = 1.0f; p.min_p = 0.2f; p.temp = 1.0f; }},
        {"greedy",                      [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.0f; }},
        {"greedy_pen1.2",               [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.0f; p.penalty_repeat = 1.2f; p.penalty_last_n = 64; }},
        {"topk40_pen1.15_f0.2_p0.3",    [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.9f; p.penalty_repeat = 1.15f; p.penalty_freq = 0.2f; p.penalty_present = 0.3f; p.penalty_last_n = 64; }},
        {"topk40_pen0.85_raises",       [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.9f; p.penalty_repeat = 0.85f; p.penalty_last_n = 32; }},
        {"topk20_pen_penalize_nl",      [](common_params_sampling & p) { p.top_k = 20; p.temp = 1.0f; p.penalty_repeat = 1.3f; p.penalty_last_n = -1; p.penalize_nl = true; }},
        {"topk40_logit_bias",           [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.8f; p.logit_bias[5] = 2.0f; p.logit_bias[7] = -INFINITY; p.logit_bias[11] = 30.0f; }},
        {"topk40_server_bias",          [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.8f; }, true},
        {"topk40_minkeep50_p0.01",      [](common_params_sampling & p) { p.top_k = 40; p.min_keep = 50; p.top_p = 0.01f; p.min_p = 0.5f; p.temp = 1.0f; }},
        {"topk5_minkeep3_m0.9",         [](common_params_sampling & p) { p.top_k = 5; p.min_keep = 3; p.top_p = 1.0f; p.min_p = 0.9f; p.temp = 0.5f; }},
        // ineligible chains: must silently take the default path
        {"ineligible_topk0",            [](common_params_sampling & p) { p.top_k = 0; p.temp = 0.8f; }},
        {"ineligible_topk256",          [](common_params_sampling & p) { p.top_k = 256; p.temp = 0.8f; }},
        {"ineligible_xtc",              [](common_params_sampling & p) { p.top_k = 40; p.xtc_probability = 0.5f; p.xtc_threshold = 0.1f; }},
        {"ineligible_pen+server_bias",  [](common_params_sampling & p) { p.top_k = 40; p.penalty_repeat = 1.1f; }, true},
        {"ineligible_greedy_nprobs",    [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.0f; p.n_probs = 5; }},
        {"real_decode_topk40_pen1.1",   [](common_params_sampling & p) { p.top_k = 40; p.temp = 0.8f; p.penalty_repeat = 1.1f; p.penalty_last_n = 64; }, false, true},
    };

    int64_t st_prev[5] = {0, 0, 0, 0, 0};
    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        const cfg & c = cfgs[ci];
        common_params_sampling p;
        p.seed = 1234 + (uint32_t) ci;
        c.set(p);
        common_sampler * smpl = common_sampler_init(model, p);
        std::vector<float> sb;
        if (c.server_bias) {
            sb.assign(n_vocab, 0.0f);
            std::mt19937 g(77);
            for (int i = 0; i < n_vocab; i += 3) sb[i] = -0.5f + (float) (g() % 100) / 25.0f;
            smpl->server_biases = &sb;
        }
        std::mt19937 gen(1000 + (uint32_t) ci);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        uint64_t h_all = 1, h_notie = 1;
        int n_tie_iters = 0;
        llama_pos pos = 1;
        for (int it = 0; it < n_iters; ++it) {
            float * row = llama_get_logits_ith(ctx, 0);
            bool tie = false;
            if (!c.real_decode) {
                for (int i = 0; i < n_vocab; ++i) row[i] = nd(gen);
                if (it % 7 == 3) {            // exact ties at the top
                    float mx = -INFINITY; for (int i = 0; i < n_vocab; ++i) mx = std::max(mx, row[i]);
                    for (int k = 0; k < 6; ++k) row[gen() % n_vocab] = mx;
                    tie = true;
                }
                if (it % 11 == 5) {           // -inf holes, including among the top
                    for (int k = 0; k < 50; ++k) row[gen() % n_vocab] = -INFINITY;
                }
                if (it % 13 == 8) {           // plateau straddling the K boundary
                    const float v = 4.0f + (float) (gen() % 5);
                    for (int k = 0; k < 200; ++k) row[gen() % n_vocab] = v;
                    tie = true;
                }
                if (it % 17 == 9) {           // a sharp spike (prob ~1) far from id 0
                    row[(size_t) gen() % n_vocab] = 60.0f;
                }
            }
            const llama_token id = common_sampler_sample(smpl, ctx, 0);
            common_sampler_accept(smpl, ctx, id, true);
            h_all = mix(h_all, (uint64_t) id);
            if (!tie) h_notie = mix(h_notie, (uint64_t) id); else n_tie_iters++;
            if (c.real_decode) {
                if (decode_one(id, pos++) != 0) { fprintf(stderr, "decode failed at it=%d\n", it); return 1; }
            }
        }
        int64_t st[5]; common_sampler_pxa_topk_raw_stats(st);
        printf("cfg %-28s hash_all=%016llx hash_notie=%016llx tie_iters=%d | fast_taken=%lld compared=%lld tie=%lld win_mismatch=%lld mismatch=%lld\n",
               c.name.c_str(), (unsigned long long) h_all, (unsigned long long) h_notie, n_tie_iters,
               (long long) (st[0]-st_prev[0]), (long long) (st[1]-st_prev[1]), (long long) (st[2]-st_prev[2]), (long long) (st[3]-st_prev[3]), (long long) (st[4]-st_prev[4]));
        for (int i = 0; i < 5; ++i) st_prev[i] = st[i];
        common_sampler_free(smpl);
    }
    int64_t st[5]; common_sampler_pxa_topk_raw_stats(st);
    printf("TOTAL fast_taken=%lld compared=%lld tie=%lld win_mismatch=%lld mismatch=%lld -> %s\n",
           (long long) st[0], (long long) st[1], (long long) st[2], (long long) st[3], (long long) st[4], st[4] == 0 && st[3] == 0 ? "PASS" : "FAIL");
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return (st[4] == 0 && st[3] == 0) ? 0 : 1;
}
