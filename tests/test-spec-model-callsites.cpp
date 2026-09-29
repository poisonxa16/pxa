// Speculative-verify call sites exercised on a REAL model context (CPU is enough). Covers the code
// the server runs, not a copy of its arithmetic:
//
//   #213  common_sampler_sample_and_accept_n, relaxed branch (PXA_SPEC_RELAXED=1): the pmin floor
//         must be compared against the normalised post-chain mass. The target distribution at the
//         verify row is pinned exactly with logit_bias (argmax 0.30, draft token 0.02, floor 0.05).
//         This case uses only API that exists on rc-base, so it builds there and FAILS there.
//   #209  the server's rollback sequence (save = common_sampler_clone, verify =
//         common_sampler_sample_and_accept_n with a rejected draft, restore =
//         common_sampler_rewind_keep_rng + common_sampler_accept) on the default DIST chain AND with
//         the adaptive-p sampler (its own RNG): the step after a rollback must not replay the
//         uniforms of the rolled-back step.
//   #228  common_speculative_draft / set_verified_len / accept with an ngram_map_k stage: a verify
//         the caller cut short, with everything verified accepted, must not shrink the key's draft
//         length; a real short accept still does (positive control).
//
// Every case builds on rc-base too: compiled with -DPXA_CALLSITES_BASE, the two server calls that
// the fixes changed are replaced by what rc-base's server-context.cpp calls in their place, so the
// SAME test run against the rc-base libraries shows the defects:
//   restore:  rc-base restore_speculative_checkpoint calls common_sampler_clone(ckpt, live), which
//             copies the checkpoint's RNG (and adaptive-p's own RNG) back into the live sampler;
//   verify:   rc-base's server never records the verified draft length (no set_verified_len).
// Expected on rc-base: #213, #209 (DIST and adaptive-p) and #228 FAIL, the controls pass.
//
// Usage: test-spec-model-callsites <model.gguf>   (any small GGUF; exit 77 = skipped, no model)

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The two server call sites the fixes changed (see the header).
#ifdef PXA_CALLSITES_BASE
#define SPEC_RESTORE_SAMPLER(ckpt, live)    common_sampler_clone((ckpt), (live))
#define SPEC_SET_VERIFIED_LEN(spec, seq, n) ((void) (spec), (void) (seq), (void) (n))
#else
#define SPEC_RESTORE_SAMPLER(ckpt, live)    common_sampler_rewind_keep_rng((ckpt), (live))
#define SPEC_SET_VERIFIED_LEN(spec, seq, n) common_speculative_set_verified_len((spec), (seq), (n))
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

static llama_model   * g_model = nullptr;
static llama_context * g_ctx   = nullptr;
static std::vector<float> g_row0; // logits of batch row 0 (the row every verify position reads)

// The sampler adds logit_bias to the context's logits IN PLACE (llama_sampling_prepare_impl), exactly
// once per verify row in the server. The test reuses row 0, so it puts the row back before each call.
static void reset_row0() {
    float * lg = llama_get_logits_ith(g_ctx, 0);
    std::copy(g_row0.begin(), g_row0.end(), lg);
}

// Bias that puts token t at exactly logit (base + ln p) after the model's own logit is added.
static void pin(common_params_sampling & sp, llama_token t, double p) {
    sp.logit_bias[t] = (float) (200.0 + std::log(p) - (double) g_row0[t]);
}

static common_params_sampling base_params(uint32_t seed) {
    common_params_sampling sp;
    sp.temp       = 1.0f;
    sp.top_k      = 40;
    sp.top_p      = 1.0f;
    sp.min_p      = 0.0f;
    sp.seed       = seed;
    return sp;
}

// ---------------------------------------------------------------- #213
static void case_relaxed_pmin() {
    // argmax A = 0.30, draft D = 0.02 (under the 0.05 floor), the rest fills to 1.
    const llama_token A = 1001, D = 1002, C = 1003, E = 1004, F = 1005, G = 1006;
    int acc_low = 0, drew_low = 0, acc_high = 0;
    const int N = 200;
    for (int s = 0; s < N; ++s) {
        common_params_sampling sp = base_params(1000 + s);
        pin(sp, A, 0.30); pin(sp, D, 0.02); pin(sp, C, 0.28); pin(sp, E, 0.25); pin(sp, F, 0.10); pin(sp, G, 0.05);
        common_sampler * smpl = common_sampler_init(g_model, sp);
        // draft = {D}: the relaxed rule may keep D only if its mass clears 0.05 -- it does not.
        reset_row0();
        auto ids = common_sampler_sample_and_accept_n(smpl, g_ctx, {0, 0}, {D});
        if (ids.size() == 2) ++acc_low;
        // how often the target's own draw WAS D (exact acceptance, allowed at ~2%)
        common_sampler_free(smpl);
        smpl = common_sampler_init(g_model, sp);
        reset_row0();
        if (common_sampler_sample(smpl, g_ctx, 0) == D) ++drew_low;
        common_sampler_free(smpl);

        // positive control: draft = {F} (0.10 >= 0.05) is kept by the relaxed rule every time
        smpl = common_sampler_init(g_model, sp);
        reset_row0();
        ids = common_sampler_sample_and_accept_n(smpl, g_ctx, {0, 0}, {F});
        if (ids.size() == 2 && ids[0] == F) ++acc_high;
        common_sampler_free(smpl);
    }
    fprintf(stderr, "#213 relaxed: draft@0.02 accepted %d/%d (target drew it %d/%d), draft@0.10 accepted %d/%d\n",
            acc_low, N, drew_low, N, acc_high, N);
    CHECK(acc_high == N, "#213 positive control: a 0.10 draft must pass a 0.05 floor every time (%d/%d)", acc_high, N);
    CHECK(acc_low <= drew_low + N / 20,
          "#213 a 0.02 draft token passed the 0.05 relaxed floor %d/%d times (target drew it %d) -- floor compared against an unnormalised p",
          acc_low, N, drew_low);
}

// ---------------------------------------------------------------- #209
// One speculative step as server-context.cpp runs it on a hybrid model, with a draft the target
// always rejects: save (clone), verify (sample_and_accept_n), rejection -> restore (rewind + accept).
static int count_replays(bool adaptive_p) {
    common_params_sampling sp = base_params(4242);
    sp.top_k = 64;
    for (int k = 0; k < 64; ++k) pin(sp, 2000 + k, 1.0 / 64.0); // 64 equiprobable candidates
    if (adaptive_p) {
        sp.adaptive_target = 0.05f; // opt-in request parameter; adaptive-p draws from its own RNG
    }
    common_sampler * live = common_sampler_init(g_model, sp);
    common_sampler * ckpt = common_sampler_init(g_model, sp);
    const llama_token never = 5; // not in the 64-token window: never matches, never relaxed-kept

    const int R = 40;
    int replays = 0;
    llama_token prev = -1;
    for (int r = 0; r < R; ++r) {
        common_sampler_clone(live, ckpt);                                                   // save_speculative_checkpoint
        reset_row0();
        auto ids = common_sampler_sample_and_accept_n(live, g_ctx, {0, 0}, {never});        // verify
        GGML_ASSERT(ids.size() == 1);
        SPEC_RESTORE_SAMPLER(ckpt, live);                                                   // restore_speculative_checkpoint
        for (llama_token id : ids) common_sampler_accept(live, g_ctx, id, true);
        if (ids[0] == prev) ++replays;
        prev = ids[0];
    }
    common_sampler_free(live);
    common_sampler_free(ckpt);
    return replays;
}

static void case_rewind_rng() {
    const int r_dist = count_replays(false);
    const int r_adp  = count_replays(true);
    fprintf(stderr, "#209 rewind: consecutive identical first tokens over 40 rolled-back steps: dist %d, adaptive-p %d (replay = 39)\n",
            r_dist, r_adp);
    // 64 equiprobable candidates: an honest stream repeats ~0.6 times in 39 pairs
    CHECK(r_dist <= 6, "#209 DIST chain: the step after a rollback replayed the rolled-back uniforms (%d/39)", r_dist);
    CHECK(r_adp  <= 6, "#209 adaptive-p: the step after a rollback replayed the adaptive-p RNG (%d/39)", r_adp);
}

// ---------------------------------------------------------------- #228
static void case_ngram_map_k_truncated_verify() {
    common_params_speculative sp;
    sp.type         = COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    sp.n_max        = 16;
    sp.n_min        = 0;
    sp.ngram_size_n = 2;
    sp.ngram_size_m = 6;
    sp.ngram_min_hits = 1;

    // a history in which the key (106, 107) is always followed by 108 .. 113
    llama_tokens prompt;
    for (int rep = 0; rep < 3; ++rep) for (int t = 3; t < 20; ++t) prompt.push_back(100 + t);
    for (int t = 3; t < 7; ++t) prompt.push_back(100 + t);      // ... 103 104 105 106
    const llama_token id_last = 107;                            // key (106, 107) -> 108 .. 113
    auto run = [&](bool truncated, size_t & first, size_t & second) {
        common_speculative * spec = common_speculative_init(sp, g_ctx);
        GGML_ASSERT(spec != nullptr);
        common_speculative_begin(spec, prompt);
        llama_tokens d1 = common_speculative_draft(spec, sp, prompt, id_last, -1, 0);
        first = d1.size();
        if (truncated) {
            SPEC_SET_VERIFIED_LEN(spec, 0, 2);                  // the server verified 2 of them ...
        }
        common_speculative_accept(spec, 2, 0);                  // ... and both were accepted
        llama_tokens d2 = common_speculative_draft(spec, sp, prompt, id_last, -1, 0);
        second = d2.size();
        common_speculative_free(spec);
    };
    size_t t1 = 0, t2 = 0, c1 = 0, c2 = 0;
    run(true,  t1, t2);
    run(false, c1, c2);
    fprintf(stderr, "#228 ngram_map_k: truncated verify %zu -> %zu, real short accept %zu -> %zu\n", t1, t2, c1, c2);
    CHECK(t1 == 6, "#228 setup: the key should draft its 6-token value (got %zu)", t1);
    CHECK(t2 == t1, "#228 a verify cut to 2 with 2/2 accepted shrank the key's draft %zu -> %zu", t1, t2);
    CHECK(c1 == 6 && c2 == 2, "#228 control: an untruncated 2/6 accept must still shrink the draft (%zu -> %zu)", c1, c2);
}


int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '\0') {
        fprintf(stderr, "usage: %s <model.gguf> -- no model given, skipped\n", argv[0]);
        return 77;
    }
    if (FILE * f = fopen(argv[1], "rb")) { fclose(f); } else {
        fprintf(stderr, "model %s not found, skipped\n", argv[1]);
        return 77;
    }
    setenv("PXA_SPEC_RELAXED", "1", 1);        // the ENHANCE default, pinned
    setenv("PXA_SPEC_RELAXED_PMIN", "0.05", 1);
    unsetenv("PXA_SPEC_SAMPLED");

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(argv[1], mp);
    GGML_ASSERT(g_model != nullptr);
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    g_ctx = llama_init_from_model(g_model, cp);
    GGML_ASSERT(g_ctx != nullptr);

    auto toks = common_tokenize(g_ctx, "The quick brown fox jumps over the lazy dog.", true, false);
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) common_batch_add(b, toks[i], (llama_pos) i, {0}, true);
    GGML_ASSERT(llama_decode(g_ctx, b) == 0);
    llama_batch_free(b);
    // every verify position below reads batch row 0; its distribution is pinned with logit_bias
    const int n_vocab = llama_n_vocab(g_model);
    const float * lg = llama_get_logits_ith(g_ctx, 0);
    GGML_ASSERT(lg != nullptr);
    g_row0.assign(lg, lg + n_vocab);

    case_relaxed_pmin();
    case_rewind_rng();
    case_ngram_map_k_truncated_verify();

    llama_free(g_ctx);
    llama_free_model(g_model);
    llama_backend_free();
    if (g_fail) { fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
    fprintf(stderr, "all call-site checks passed\n");
    return 0;
}
