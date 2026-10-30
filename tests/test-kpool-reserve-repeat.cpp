// PXA_GLM5NEXT k-pool: a greedy request must not depend on the requests before it.
//
// PXA_SCHED_RESERVE_REAL re-reserves the compute buffers with a real-path graph built from a
// SYNTHETIC batch whenever an ubatch's graph is not covered by the reserved plan - in a server,
// on the first decode token of every request after the first (the very first decode of a context
// is exempt: n_eval == 0). That synthetic build used to re-plan the k-pool grid through the
// carried incremental plan and leave the synthetic batch's new-pool marks behind; the real
// ubatch's rebuild kept them, so a decode token that completed a pool lost its pooled-key write.
// Seen as GLM plain output changing between identical greedy requests (request 1 != request 7).
//
// The test runs, per prompt length, three identical greedy passes on one fresh context (the KV
// cleared between passes, as a server does between requests) and requires every pass to match
// the first bit for bit - tokens and the logits row of every step. Several prompt lengths, so
// that the first decode lands on a pool-completing position for any k-pool width up to 8.
//
// Needs a model with a k-pool indexer, so it is built but NOT registered with ctest; drive it
// with the tiny synthetic glm5next GGUF from tests/gen_tiny_glm5next.py:
//   test-kpool-reserve-repeat tiny-glm5next.gguf [n_lo] [n_hi] [n_predict]
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct pass_out {
    std::vector<llama_token> toks;
    std::vector<uint64_t>    lhash;   // FNV-1a of the logits row bits, per step
};

static uint64_t fnv_row(const float * v, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * p = (const unsigned char *) v;
    for (size_t i = 0; i < (size_t) n * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static bool run_pass(llama_context * ctx, const std::vector<llama_token> & prompt, int n_predict, int n_vocab, pass_out & out) {
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    const int n = (int) prompt.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = prompt[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (int8_t) (i == n - 1);
    }
    b.n_tokens = n;
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    if (!ok) { return false; }

    llama_pos pos = n;
    for (int t = 0; t < n_predict; ++t) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { return false; }
        llama_token best = 0;
        for (int i = 1; i < n_vocab; ++i) { if (lg[i] > lg[best]) { best = i; } }
        out.toks.push_back(best);
        out.lhash.push_back(fnv_row(lg, n_vocab));

        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0;
        d.logits[0] = 1; d.n_tokens = 1;
        const bool dok = llama_decode(ctx, d) == 0;
        llama_batch_free(d);
        if (!dok) { return false; }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_lo] [n_hi] [n_predict]\n", argv[0]); return 2; }
    const int n_lo      = argc > 2 ? atoi(argv[2]) : 29;
    const int n_hi      = argc > 3 ? atoi(argv[3]) : 36;
    const int n_predict = argc > 4 ? atoi(argv[4]) : 12;
    const int n_pass    = 3;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    const int n_vocab = llama_n_vocab(model);

    int n_fail = 0;
    for (int n = n_lo; n <= n_hi; ++n) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 512;
        cp.n_batch         = 512;
        cp.n_ubatch        = 512;
        cp.n_seq_max       = 1;
        cp.n_threads       = 4;
        cp.n_threads_batch = 4;
        cp.flash_attn      = false;   // glm5next MLA needs the transposed latent store
        cp.mla_attn        = 1;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "n=%d ctx failed\n", n); return 1; }

        std::vector<llama_token> prompt(n);
        for (int i = 0; i < n; ++i) {
            prompt[i] = (llama_token) (5 + (i*37) % (n_vocab > 16 ? n_vocab - 8 : 8));
        }

        std::vector<pass_out> outs(n_pass);
        bool ok = true;
        for (int r = 0; r < n_pass && ok; ++r) {
            ok = run_pass(ctx, prompt, n_predict, n_vocab, outs[r]);
        }
        int first_bad_pass = -1, first_bad_step = -1;
        for (int r = 1; ok && r < n_pass && first_bad_pass < 0; ++r) {
            for (int t = 0; t < n_predict; ++t) {
                if (outs[r].toks[t] != outs[0].toks[t] || outs[r].lhash[t] != outs[0].lhash[t]) {
                    first_bad_pass = r; first_bad_step = t; break;
                }
            }
        }
        const bool pass = ok && first_bad_pass < 0;
        printf("prompt_n=%2d %s", n, !ok ? "DECODE-FAILED" : pass ? "repeatable" : "DIFFERS");
        if (first_bad_pass >= 0) {
            printf(" (pass %d step %d: token %d vs %d)", first_bad_pass, first_bad_step,
                   outs[first_bad_pass].toks[first_bad_step], outs[0].toks[first_bad_step]);
        }
        printf("\n");
        fflush(stdout);
        if (!pass) { n_fail++; }
        llama_free(ctx);
    }

    llama_free_model(model);
    llama_backend_free();

    if (n_fail) {
        fprintf(stderr, "FAIL: %d of %d prompt lengths gave a different answer on a repeated request\n", n_fail, n_hi - n_lo + 1);
        return 1;
    }
    printf("OK: prompt lengths %d..%d, %d passes each, all bit-identical\n", n_lo, n_hi, n_pass);
    return 0;
}
