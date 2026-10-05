// PXA_REP_PREV_v1 + PXA_REP_DRY_GREEDY_v1 regression test (CPU only, one tiny gguf, no card).
//
// Two defects made every repetition setting a no-op on a looping greedy request (// 2026-10-04: repeat 1.05 .. 2.0, frequency, presence 2.0 and DRY 0.8 at temperature 0 gave output
// byte-identical to plain greedy on a prompt that loops):
//   1. the sampler's history `prev` was a ONE-token window (never sized without a grammar, and the
//      accept rule erased the oldest entry before every push), so the penalties only ever saw the
//      token just emitted -- a loop of period > 1 was never touched;
//   2. the temperature-0 branch skipped the whole chain after the penalty pass, so DRY never ran.
// This test synthesises the logit row (the existing test-topk-raw-shadow does the same), so the
// expected token is known exactly, and drives the real common_sampler: plain draws, the PXA_VERIFY_ARGMAX
// gate, and the speculative verify (common_sampler_sample_and_accept_n).
//   test-rep-penalty-window model.gguf        (run it with PXA_TOPK_RAW=0 and =1: both paths must agree)
// Every "must differ" case below FAILS on the pre-fix sampler.
#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { g_fail++; printf("FAIL  %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
    else { printf("ok    "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 32; cp.n_ubatch = 32; cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int n_vocab = llama_n_vocab(model);
    const char * mode = getenv("PXA_TOPK_RAW");
    printf("model=%s n_vocab=%d PXA_TOPK_RAW=%s\n", argv[1], n_vocab, mode ? mode : "(unset)");
    if (n_vocab < 460) { fprintf(stderr, "needs a vocab of at least 460\n"); return 2; }

    // one real decode so the logits rows exist; the rows are then overwritten
    {
        llama_batch b = llama_batch_init(4, 0, 1);
        for (int i = 0; i < 4; ++i) {
            b.token[i] = 1 + i; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = 4;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        llama_batch_free(b);
    }

    using Logits = std::vector<std::pair<int, float>>;
    const auto set_row = [&](int row, const Logits & l) {
        float * r = llama_get_logits_ith(ctx, row);
        for (int i = 0; i < n_vocab; ++i) r[i] = 0.0f;
        for (const auto & kv : l) r[kv.first] = kv.second;
    };
    const auto make = [&](const std::function<void(common_params_sampling &)> & f) {
        common_params_sampling p;
        p.seed = 7;
        p.top_k = 40;
        p.dry_sequence_breakers.clear();   // the fixture's fillers are not breakers either way
        f(p);
        return common_sampler_init(model, p);
    };
    const auto feed = [&](common_sampler * s, const std::vector<int> & toks) {
        for (const int t : toks) common_sampler_accept(s, ctx, t, false);   // the prompt, as the server does
    };
    const auto pick = [&](common_sampler * s, const Logits & l) {
        set_row(0, l);
        return (int) common_sampler_sample(s, ctx, 0, false);
    };

    const int A = 300, B = 301, C = 302, D = 303, E = 304, X = 310, F0 = 420;

    // 1. a token that is NOT the last one emitted is penalised (the pre-fix window held one token)
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 2.0f; p.penalty_last_n = 64; });
        feed(s, {A, B, C, D, E});
        CHECK(pick(s, {{A, 10.0f}, {X, 9.5f}}) == X, "temp0 repeat 2.0: an older token in the window is penalised (10/2 < 9.5)");
        common_sampler_free(s);
    }
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; });
        feed(s, {A, B, C, D, E});
        CHECK(pick(s, {{A, 10.0f}, {X, 9.5f}}) == A, "temp0 default (no penalty): unchanged, the argmax");
        common_sampler_free(s);
    }
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_present = 6.0f; p.penalty_last_n = 64; });
        feed(s, {A, B, C, D, E});
        CHECK(pick(s, {{A, 10.0f}, {X, 9.5f}}) == X, "temp0 presence 6.0: an older token in the window is penalised (10-6 < 9.5)");
        common_sampler_free(s);
    }
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_freq = 2.0f; p.penalty_last_n = 64; });
        feed(s, {A, A, B, C, D, E});
        CHECK(pick(s, {{A, 10.0f}, {X, 9.5f}}) == X, "temp0 frequency 2.0 x 2 occurrences: 10-4 < 9.5");
        common_sampler_free(s);
    }
    // 2. the window is exactly repeat_last_n
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 2.0f; p.penalty_last_n = 3; });
        feed(s, {A, B, C, D, E});   // window = {C, D, E}
        CHECK(pick(s, {{A, 10.0f}, {X, 9.5f}}) == A, "repeat_last_n 3: a token 5 back is outside the window");
        CHECK(pick(s, {{C, 10.0f}, {X, 9.5f}}) == X, "repeat_last_n 3: a token 3 back is inside the window");
        common_sampler_free(s);
    }
    // 3. repeat_last_n larger than n_prev (the guard's 256 against the default n_prev 64)
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 2.0f; p.penalty_last_n = 256; });
        std::vector<int> prompt = {F0};
        for (int i = 0; i < 200; ++i) prompt.push_back(311 + (i % 40));
        feed(s, prompt);
        CHECK(pick(s, {{F0, 10.0f}, {X + 120, 9.5f}}) == X + 120, "repeat_last_n 256 > n_prev 64: a token 200 back is penalised");
        common_sampler_free(s);
    }
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 2.0f; p.penalty_last_n = 256; });
        std::vector<int> prompt = {F0};
        for (int i = 0; i < 300; ++i) prompt.push_back(311 + (i % 40));
        feed(s, prompt);
        CHECK(pick(s, {{F0, 10.0f}, {X + 120, 9.5f}}) == F0, "repeat_last_n 256: a token 300 back is outside the window");
        common_sampler_free(s);
    }
    // 4. a long prompt keeps the history bounded (a 100k-token prompt must not grow without bound)
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 1.3f; p.penalty_last_n = 128; });
        for (int i = 0; i < 20000; ++i) common_sampler_accept(s, ctx, 311 + (i % 97), false);
        CHECK(s->prev.size() >= 128 && s->prev.size() <= 2 * 128, "history stays within [keep, 2*keep] after 20000 tokens (size %zu)", s->prev.size());
        common_sampler_free(s);
    }
    // 5. DRY now applies at temperature 0
    {
        const auto dry = [](common_params_sampling & p) {
            p.temp = 0.0f; p.dry_multiplier = 0.8f; p.dry_base = 1.75f; p.dry_allowed_length = 2; p.dry_penalty_last_n = 64;
        };
        common_sampler * s = make(dry);
        feed(s, {A, B, C, D, A, B, C, D, A, B});
        CHECK(common_sampler_dry_active(s), "DRY reports active");
        CHECK(pick(s, {{C, 10.0f}, {X, 9.9f}}) == X, "temp0 DRY 0.8: the token that extends a 6-token repeat is penalised");
        common_sampler_free(s);
        common_sampler * s0 = make([](common_params_sampling & p) { p.temp = 0.0f; });
        feed(s0, {A, B, C, D, A, B, C, D, A, B});
        CHECK(!common_sampler_dry_active(s0), "DRY reports inactive at its default");
        CHECK(pick(s0, {{C, 10.0f}, {X, 9.9f}}) == C, "temp0 no DRY: unchanged, the argmax");
        common_sampler_free(s0);
    }
    // 6. the device-argmax verify gate (PXA_VERIFY_ARGMAX) must not claim a penalised request
    {
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; });
        feed(s, {A, B, C});
        CHECK(common_sampler_pure_greedy(s), "default greedy request is pure greedy (device argmax verify allowed)");
        common_sampler_free(s);
        s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 1.3f; });
        feed(s, {A, B, C});
        CHECK(!common_sampler_pure_greedy(s), "repeat 1.3 request is NOT pure greedy");
        common_sampler_free(s);
        s = make([](common_params_sampling & p) { p.temp = 0.0f; p.dry_multiplier = 0.8f; p.dry_penalty_last_n = 64; });
        feed(s, {A, B, C});
        CHECK(!common_sampler_pure_greedy(s), "DRY request is NOT pure greedy");
        common_sampler_free(s);
    }
    // 7. speculative verify: rows 0..2 of the batch, draft {A, B}; the bonus row must see A and B in the
    //    penalty window (pre-fix it saw only B)
    {
        const auto verify = [&](const std::function<void(common_params_sampling &)> & f) {
            common_sampler * s = make(f);
            feed(s, {C, D, E});
            // set_row() writes batch row 0 only, so the three rows are filled by hand
            float * r0 = llama_get_logits_ith(ctx, 0); for (int i = 0; i < n_vocab; ++i) r0[i] = 0.0f; r0[A] = 10.0f; r0[X] = 5.0f;
            float * r1 = llama_get_logits_ith(ctx, 1); for (int i = 0; i < n_vocab; ++i) r1[i] = 0.0f; r1[B] = 10.0f; r1[X] = 5.0f;
            float * r2 = llama_get_logits_ith(ctx, 2); for (int i = 0; i < n_vocab; ++i) r2[i] = 0.0f; r2[A] = 10.0f; r2[X] = 9.5f;
            const std::vector<int> idxs = {0, 1, 2};
            const std::vector<llama_token> draft = {A, B};
            const std::vector<llama_token> out = common_sampler_sample_and_accept_n(s, ctx, idxs, draft);
            common_sampler_free(s);
            return out;
        };
        const auto plain = verify([](common_params_sampling & p) { p.temp = 0.0f; });
        CHECK(plain == std::vector<llama_token>({A, B, A}), "spec verify, no penalty: draft accepted, bonus = argmax A (%zu tokens)", plain.size());
        const auto pen = verify([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 2.0f; p.penalty_last_n = 64; });
        CHECK(pen == std::vector<llama_token>({A, B, X}), "spec verify, repeat 2.0: bonus row is penalised for A (10/2 < 9.5) -> X (%zu tokens)", pen.size());
        // a draft the penalty rejects: row 0 favours A, which is in the prompt window
        common_sampler * s = make([](common_params_sampling & p) { p.temp = 0.0f; p.penalty_repeat = 2.0f; p.penalty_last_n = 64; });
        feed(s, {A, D, E});
        float * r0 = llama_get_logits_ith(ctx, 0); for (int i = 0; i < n_vocab; ++i) r0[i] = 0.0f; r0[A] = 10.0f; r0[X] = 9.5f;
        float * r1 = llama_get_logits_ith(ctx, 1); for (int i = 0; i < n_vocab; ++i) r1[i] = 0.0f; r1[A] = 10.0f; r1[X] = 9.5f;
        const std::vector<llama_token> out = common_sampler_sample_and_accept_n(s, ctx, {0, 1}, {A});
        CHECK(out == std::vector<llama_token>({X}), "spec verify, repeat 2.0: a draft of the penalised token is rejected at its own position (%zu tokens)", out.size());
        common_sampler_free(s);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    printf("%s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
