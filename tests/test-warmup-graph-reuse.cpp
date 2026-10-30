// Graph reuse must never change a greedy answer, whatever ran on the context before.
//
// The load-time warmup decodes a single BOS at n_eval == 0, which llama_build_graph turns into the
// all-experts WARMUP graph (n_expert_used = n_expert). That graph used to be recorded as the
// context's reuse record (`prev`). A prompt of 2..PXA_VERIFY_GRAPH_CACHE tokens runs in its own
// verify-width graph slot and leaves the main reuse record alone, so the first 1-token decode
// after it found the warmup graph still in `prev` at the same n_kv and REUSED it: every decode
// token of that request mixed all experts. Any prompt wider than the slot range rebuilt `prev` on
// the main graph and the next request decoded right - the Gemma 4 server's "output flips after a
// wider prompt" cross-request carry (impl gemma-cross-request-carry).
//
// The test runs the same request list on two fresh contexts of one model, each warmed up the
// way common_init_from_params does it (one BOS, KV cleared, timings reset): REF with graph reuse
// off (every graph built for its own ubatch) and REUSE with graph reuse on. Every request (KV
// cleared before it, as a server with cache_prompt off does) must give the same tokens and the
// same logits bits at every step in both contexts. The list puts short prompts (slot widths)
// first, a wide one in the middle and short ones again after it.
//
// The model is built in the test: a tiny llama-architecture MoE (4 experts, 2 used, so the warmup
// graph really differs) with random-but-fixed weights and a minimal SPM vocab that has a BOS,
// written as a GGUF to a temp file - no data dependencies, CPU only, registered with ctest.
// Any other MoE GGUF can be given instead (e.g. tests/gen_tiny_qwen4exp.py), on a card too:
//   test-warmup-graph-reuse [model.gguf|-] [n_predict] [n_gpu_layers]
// PXA_VERIFY_GRAPH_CACHE defaults to 6 here (the ENHANCE default), so the slot path is exercised
// whatever level the build resolves; set it in the environment to test another width.
#include "llama.h"
#include "ggml.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------
// the tiny MoE model
// ---------------------------------------------------------------------------------------------

static const int N_VOCAB   = 256;
static const int N_EMBD    = 64;
static const int N_HEAD    = 4;
static const int N_HEAD_KV = 2;
static const int N_LAYER   = 2;
static const int N_FF      = 64;
static const int N_EXPERT  = 4;
static const int N_USED    = 2;
static const int HEAD_DIM  = N_EMBD / N_HEAD;

static uint32_t g_rng = 20261017u;

static float frand() { // deterministic, small, zero-mean
    g_rng = g_rng*1664525u + 1013904223u;
    return (((float) (g_rng >> 8) / (float) (1u << 23)) - 1.0f) * 0.25f;
}

static void mk(ggml_context * ctx, gguf_context * gg, const char * name, int64_t ne0, int64_t ne1 = 0, int64_t ne2 = 0) {
    ggml_tensor * t = ne2 > 0 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2)
                    : ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                              : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = ne1 > 0 ? frand() : 1.0f + 0.1f*frand();   // rms_norm gains near 1
    }
    gguf_add_tensor(gg, t);
}

static bool write_tiny_moe(const char * path) {
    ggml_init_params ip = { /*.mem_size =*/ 16u*1024u*1024u, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { return false; }
    gguf_context * gg = gguf_init_empty();
    gguf_set_val_str(gg, "general.architecture", "llama");
    gguf_set_val_str(gg, "general.name",         "tiny-warmup-reuse");
    gguf_set_val_u32(gg, "general.file_type",    0);
    gguf_set_val_u32(gg, "llama.block_count",              N_LAYER);
    gguf_set_val_u32(gg, "llama.context_length",           4096);
    gguf_set_val_u32(gg, "llama.embedding_length",         N_EMBD);
    gguf_set_val_u32(gg, "llama.feed_forward_length",      N_FF);
    gguf_set_val_u32(gg, "llama.expert_count",             N_EXPERT);
    gguf_set_val_u32(gg, "llama.expert_used_count",        N_USED);
    gguf_set_val_u32(gg, "llama.attention.head_count",     N_HEAD);
    gguf_set_val_u32(gg, "llama.attention.head_count_kv",  N_HEAD_KV);
    gguf_set_val_u32(gg, "llama.attention.key_length",     HEAD_DIM);
    gguf_set_val_u32(gg, "llama.attention.value_length",   HEAD_DIM);
    gguf_set_val_f32(gg, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gg, "llama.rope.dimension_count",     HEAD_DIM);
    gguf_set_val_f32(gg, "llama.rope.freq_base",           10000.0f);
    gguf_set_val_u32(gg, "llama.vocab_size",               N_VOCAB);
    // a minimal SPM vocab: the warmup is keyed to the BOS id, so the model must have one
    std::vector<std::string> words(N_VOCAB);
    std::vector<const char *> wp(N_VOCAB);
    std::vector<float>   scores(N_VOCAB, 0.0f);
    std::vector<int32_t> types(N_VOCAB, 1);
    for (int i = 0; i < N_VOCAB; ++i) {
        words[i] = i == 0 ? "<unk>" : i == 1 ? "<s>" : i == 2 ? "</s>" : "w" + std::to_string(i);
        wp[i] = words[i].c_str();
    }
    types[0] = 2; types[1] = 3; types[2] = 3;
    gguf_set_val_str(gg, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(gg, "tokenizer.ggml.tokens", wp.data(), N_VOCAB);
    gguf_set_arr_data(gg, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), N_VOCAB);
    gguf_set_arr_data(gg, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), N_VOCAB);
    gguf_set_val_u32(gg, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(gg, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(gg, "tokenizer.ggml.unknown_token_id", 0);

    mk(ctx, gg, "token_embd.weight", N_EMBD, N_VOCAB);
    for (int il = 0; il < N_LAYER; ++il) {
        char n[64];
        snprintf(n, sizeof(n), "blk.%d.attn_norm.weight",    il); mk(ctx, gg, n, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.attn_q.weight",       il); mk(ctx, gg, n, N_EMBD, N_HEAD*HEAD_DIM);
        snprintf(n, sizeof(n), "blk.%d.attn_k.weight",       il); mk(ctx, gg, n, N_EMBD, N_HEAD_KV*HEAD_DIM);
        snprintf(n, sizeof(n), "blk.%d.attn_v.weight",       il); mk(ctx, gg, n, N_EMBD, N_HEAD_KV*HEAD_DIM);
        snprintf(n, sizeof(n), "blk.%d.attn_output.weight",  il); mk(ctx, gg, n, N_HEAD*HEAD_DIM, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.ffn_norm.weight",     il); mk(ctx, gg, n, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp.weight", il); mk(ctx, gg, n, N_EMBD, N_EXPERT);
        snprintf(n, sizeof(n), "blk.%d.ffn_gate_exps.weight",il); mk(ctx, gg, n, N_EMBD, N_FF, N_EXPERT);
        snprintf(n, sizeof(n), "blk.%d.ffn_up_exps.weight",  il); mk(ctx, gg, n, N_EMBD, N_FF, N_EXPERT);
        snprintf(n, sizeof(n), "blk.%d.ffn_down_exps.weight",il); mk(ctx, gg, n, N_FF, N_EMBD, N_EXPERT);
    }
    mk(ctx, gg, "output_norm.weight", N_EMBD);
    mk(ctx, gg, "output.weight",      N_EMBD, N_VOCAB);

    gguf_write_to_file(gg, path, false);
    gguf_free(gg);
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------------------------
// the requests
// ---------------------------------------------------------------------------------------------

struct req_out {
    std::vector<llama_token> toks;
    std::vector<uint64_t>    lhash;   // FNV-1a of the logits row bits, per step
};

static uint64_t fnv_row(const float * v, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * p = (const unsigned char *) v;
    for (size_t i = 0; i < (size_t) n * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static llama_token greedy(const float * lg, int n_vocab) {
    llama_token best = 0;
    for (int i = 1; i < n_vocab; ++i) { if (lg[i] > lg[best]) { best = i; } }
    return best;
}

// the warmup of common_init_from_params: one BOS (EOS when there is none), then a clean context
static bool warmup(llama_context * ctx, const llama_model * model) {
    llama_token t = llama_token_bos(model);
    if (t == -1) { t = llama_token_eos(model); }
    const bool ok = llama_decode(ctx, llama_batch_get_one(&t, 1, 0, 0)) == 0;
    llama_kv_cache_clear(ctx);
    llama_synchronize(ctx);
    llama_reset_timings(ctx);
    return ok;
}

static bool run_request(llama_context * ctx, const std::vector<llama_token> & prompt, int n_predict, int n_vocab, req_out & out) {
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
        const llama_token best = greedy(lg, n_vocab);
        out.toks.push_back(best);
        out.lhash.push_back(fnv_row(lg, n_vocab));
        if (t + 1 == n_predict) { break; }
        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0;
        d.logits[0] = 1; d.n_tokens = 1;
        const bool dok = llama_decode(ctx, d) == 0;
        llama_batch_free(d);
        if (!dok) { return false; }
    }
    return true;
}

static bool run_list(const llama_model * model, bool graph_reuse, int n_ctx, const std::vector<std::vector<llama_token>> & reqs,
                     int n_predict, int n_vocab, std::vector<req_out> & outs) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.n_ubatch        = n_ctx;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.graph_reuse     = graph_reuse;
    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx) { fprintf(stderr, "context failed (graph_reuse=%d)\n", (int) graph_reuse); return false; }
    bool ok = warmup(ctx, model);
    outs.assign(reqs.size(), req_out{});
    for (size_t r = 0; ok && r < reqs.size(); ++r) {
        ok = run_request(ctx, reqs[r], n_predict, n_vocab, outs[r]);
    }
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool own_model = argc < 2 || strcmp(argv[1], "-") == 0;
    const int  n_predict = argc > 2 ? atoi(argv[2]) : 8;
    const int  n_gpu     = argc > 3 ? atoi(argv[3]) : 0;
    setenv("PXA_VERIFY_GRAPH_CACHE", "6", /* overwrite */ 0);

    std::string path = own_model ? "./test-warmup-graph-reuse.gguf" : argv[1];
    if (own_model) {
        if (const char * t = getenv("TMPDIR")) { path = std::string(t) + "/test-warmup-graph-reuse.gguf"; }
        if (!write_tiny_moe(path.c_str())) { fprintf(stderr, "could not write the test model\n"); return 1; }
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (own_model) { remove(path.c_str()); }
    if (!model) { fprintf(stderr, "load failed: %s\n", path.c_str()); return 1; }
    printf("test-warmup-graph-reuse: graph reuse must not change any greedy answer after the warmup\n");
    const int n_vocab = llama_n_vocab(model);

    // prompt widths: slot widths (2..6) first, one main-graph width (12), slot widths again
    const int widths[] = { 4, 4, 2, 6, 12, 4, 3, 5 };
    std::vector<std::vector<llama_token>> reqs;
    for (int w : widths) {
        std::vector<llama_token> p(w);
        for (int i = 0; i < w; ++i) {
            p[i] = (llama_token) (5 + (i*37 + w*11) % (n_vocab > 16 ? n_vocab - 8 : 8));
        }
        reqs.push_back(p);
    }

    std::vector<req_out> ref, reuse;
    if (!run_list(model, false, 256, reqs, n_predict, n_vocab, ref) ||
        !run_list(model, true,  256, reqs, n_predict, n_vocab, reuse)) {
        fprintf(stderr, "FAIL: a decode failed\n");
        return 1;
    }

    int n_fail = 0;
    for (size_t r = 0; r < reqs.size(); ++r) {
        int bad = -1;
        for (int t = 0; t < n_predict; ++t) {
            if (reuse[r].toks[t] != ref[r].toks[t] || reuse[r].lhash[t] != ref[r].lhash[t]) { bad = t; break; }
        }
        printf("request %zu prompt_n=%2d %s", r, (int) reqs[r].size(), bad < 0 ? "same as no-reuse" : "DIFFERS");
        if (bad >= 0) {
            printf(" (step %d: token %d vs %d)", bad, reuse[r].toks[bad], ref[r].toks[bad]);
            n_fail++;
        }
        printf("\n");
    }
    fflush(stdout);

    llama_free_model(model);
    llama_backend_free();

    if (n_fail) {
        fprintf(stderr, "FAIL: %d of %zu requests decode differently with graph reuse on\n", n_fail, reqs.size());
        return 1;
    }
    printf("OK: %zu requests x %d steps, graph reuse on == off bit for bit\n", reqs.size(), n_predict);
    return 0;
}
