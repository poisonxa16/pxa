// PXA_GLM5NEXT_GRAPH_REUSE regression (CPU only, no GPU).
//
// glm5next decode used to rebuild its graph every token, because the MLA/DSA blocks wrote
// their K/V and idx_l side cache with raw ggml_cpy and never registered the writes in
// cache_copies. With PXA_GLM5NEXT_GRAPH_REUSE=1 they are registered and re-pointed at the new
// kv_head on reuse. This test runs the SAME greedy decode twice:
//   A: graph_reuse = false (the reference: every token is a fresh graph)
//   B: graph_reuse = true  (the switch armed)
// and requires bit-identical logits at every step plus at least one actual reuse in B.
//
//   test-glm5next-graph-reuse tiny-glm5next.gguf [n_prompt] [n_predict]
#include "llama.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool run(llama_model * model, bool reuse, int n_prompt, int n_predict,
                std::vector<float> & out, std::vector<llama_token> & toks, int64_t & n_reuse) {
    const int n_vocab = llama_n_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 512;
    cp.n_batch         = 256;
    cp.n_ubatch        = 128;
    cp.n_seq_max       = 1;
    const char * te = getenv("PXA_TEST_THREADS");
    cp.n_threads       = te ? atoi(te) : 4;
    cp.n_threads_batch = cp.n_threads;
    cp.flash_attn      = false;   // glm5next MLA needs the transposed latent store
    cp.mla_attn        = 1;
    cp.graph_reuse     = reuse;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return false; }

    llama_batch b = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        b.token[i] = (llama_token) (5 + (i*37) % (n_vocab > 16 ? n_vocab - 8 : 8));
        b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (int8_t) (i == n_prompt - 1);
    }
    b.n_tokens = n_prompt;
    bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);

    llama_pos pos = n_prompt;
    for (int t = 0; ok && t < n_predict; ++t) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { ok = false; break; }
        out.insert(out.end(), lg, lg + n_vocab);
        llama_token best = 0; float bv = lg[0];
        for (int i = 1; i < n_vocab; ++i) { if (lg[i] > bv) { bv = lg[i]; best = i; } }
        toks.push_back(best);
        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0;
        d.logits[0] = 1; d.n_tokens = 1;
        if (llama_decode(ctx, d) != 0) { fprintf(stderr, "decode t=%d failed\n", t); ok = false; }
        llama_batch_free(d);
    }
    n_reuse = llama_pxa_n_graph_reuse(ctx);
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_prompt] [n_predict]\n", argv[0]); return 2; }
    const int n_prompt  = argc > 2 ? atoi(argv[2]) : 7;
    const int n_predict = argc > 3 ? atoi(argv[3]) : 96;

    // arm the switch for this process unless the caller set it explicitly
    setenv("PXA_GLM5NEXT_GRAPH_REUSE", "1", 0);
    const bool armed = llama_pxa_glm5next_graph_reuse();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    const int n_vocab = llama_n_vocab(model);

    std::vector<float> la, lb; std::vector<llama_token> ta, tb; int64_t ra = 0, rb = 0;
    if (!run(model, false, n_prompt, n_predict, la, ta, ra)) return 1;
    if (!run(model, true,  n_prompt, n_predict, lb, tb, rb)) return 1;

    int n_fail = 0;
    if (la.size() != lb.size()) { fprintf(stderr, "FAIL: logit count %zu vs %zu\n", la.size(), lb.size()); n_fail++; }
    else {
        for (size_t i = 0; i < la.size(); ++i) {
            if (memcmp(&la[i], &lb[i], sizeof(float)) != 0) {
                fprintf(stderr, "FAIL: step %zu logit[%zu] ref %.9g reuse %.9g\n",
                        i / n_vocab, i % n_vocab, la[i], lb[i]);
                n_fail++;
                break;
            }
        }
    }
    for (size_t i = 0; i < la.size(); ++i) { if (!std::isfinite(la[i])) { fprintf(stderr, "FAIL: non-finite ref logit\n"); n_fail++; break; } }
    printf("armed=%d steps=%d reuse(ref)=%lld reuse(armed)=%lld tokens_equal=%d\n",
           (int) armed, n_predict, (long long) ra, (long long) rb, (int) (ta == tb));
    if (ra != 0) { fprintf(stderr, "FAIL: reference context reused a graph\n"); n_fail++; }
    if (armed && rb == 0) { fprintf(stderr, "FAIL: armed context never reused a graph\n"); n_fail++; }

    llama_free_model(model);
    llama_backend_free();
    if (n_fail) return 1;
    printf("OK: graph-reuse decode bit-identical to fresh-graph decode over %d steps\n", n_predict);
    return 0;
}
