// PXA_GLM5NEXT short-prompt regression (CPU only).
//
// The bug this guards: prompts of certain SHORT lengths made the glm5next graph emit NaN
// logits ('!' = token 0 forever), and because the NaN lands in the persistent k-pool /
// recurrent state, every later request on that context is garbage too.
//
// One FRESH context per prompt length -- the whole point is that the state must not be
// poisoned, so a shared context would hide which length actually broke.
//
//   test-glm5next-short-prompts model.gguf [n_max] [n_predict]
//
// Exit 0 when every length 1..n_max produced finite logits at every step.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool row_finite(const float * v, int n, int & bad_at) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { bad_at = i; return false; }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_max] [n_predict]\n", argv[0]); return 2; }
    const int n_max     = argc > 2 ? atoi(argv[2]) : 16;
    const int n_predict = argc > 3 ? atoi(argv[3]) : 4;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }

    const int n_vocab = llama_n_vocab(model);
    int n_fail = 0;

    for (int n = 1; n <= n_max; ++n) {
        llama_context_params cp = llama_context_default_params();
        const char * ce = getenv("PXA_NAN_CTX");
        cp.n_ctx           = ce ? (uint32_t) atoi(ce) : 512;
        cp.n_batch         = 256;
        cp.n_ubatch        = 128;   // the whole prompt in ONE ubatch, as a server prefill does
        cp.n_seq_max       = 1;
        const char * te = getenv("PXA_NAN_THREADS");
        cp.n_threads       = te ? atoi(te) : 4;
        cp.n_threads_batch = cp.n_threads;
        cp.flash_attn      = false;   // glm5next MLA needs the transposed latent store
        cp.mla_attn        = 1;

        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "n=%2d ctx failed\n", n); return 1; }

        // a deterministic, in-vocab prompt; the content does not matter, the LENGTH does
        std::vector<llama_token> prompt(n);
        for (int i = 0; i < n; ++i) {
            prompt[i] = (llama_token) (5 + (i*37) % (n_vocab > 16 ? n_vocab - 8 : 8));
        }

        bool ok = true;
        int  bad_at = -1;
        int  bad_step = -1;

        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = prompt[i];
            b.pos[i]   = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (int8_t) (i == n - 1);
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "n=%2d prefill decode failed\n", n); ok = false; }
        llama_batch_free(b);

        llama_pos pos = n;
        for (int t = 0; ok && t < n_predict; ++t) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            if (!lg || !row_finite(lg, n_vocab, bad_at)) { ok = false; bad_step = t; break; }

            // greedy
            llama_token best = 0;
            float bv = lg[0];
            for (int i = 1; i < n_vocab; ++i) { if (lg[i] > bv) { bv = lg[i]; best = i; } }

            llama_batch d = llama_batch_init(1, 0, 1);
            d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0;
            d.logits[0] = 1; d.n_tokens = 1;
            if (llama_decode(ctx, d) != 0) { fprintf(stderr, "n=%2d decode t=%d failed\n", n, t); ok = false; }
            llama_batch_free(d);
        }
        if (ok) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            if (!lg || !row_finite(lg, n_vocab, bad_at)) { ok = false; bad_step = n_predict; }
        }

        printf("prompt_n=%2d %s", n, ok ? "ok" : "NONFINITE");
        if (!ok && bad_step >= 0) printf(" (step %d, logit[%d])", bad_step, bad_at);
        printf("\n");
        fflush(stdout);

        if (!ok) n_fail++;
        llama_free(ctx);
    }

    llama_free_model(model);
    llama_backend_free();

    if (n_fail) {
        fprintf(stderr, "FAIL: %d of %d prompt lengths produced non-finite logits\n", n_fail, n_max);
        return 1;
    }
    printf("OK: lengths 1..%d all finite\n", n_max);
    return 0;
}
