// Gemma 4 with flash attention OFF must give the logits flash attention ON gives (bug #332).
//
// With -fa off the V cache is stored transposed (kv.v_trans): llm_build_kv_store writes
// ggml_transpose(V) into an [n_tokens, n_embd_v_gqa] view of the cache. Gemma 4 hands V over
// per head, [head_dim, n_head_kv, n_tokens] (it is rms-normed per head), and ggml_transpose of
// that 3D tensor swaps head_dim and n_head_kv, not the token axis, so the copy scrambled every V
// row: -fa off (and PXA_MODE=max, which turns flash attention off) answered garbage on every
// card and on the CPU. PXA_VTRANS_2D (default on since the fix; =0 brings the bug back) reshapes
// a per-head V to [n_embd_v_gqa, n_tokens] first. The same test covers the case the first
// version of the lever missed: a ONE-KV-head layer, whose V [head_dim, 1, n_tokens] already has
// ne[0] == n_embd_v_gqa and was still scrambled at n_tokens > 1.
//
// The model is built in the test: a tiny dense Gemma 4 (three blocks: two sliding-window blocks
// with their own V projection and 2 KV heads, one global block that reads V = K with ONE KV head,
// the real 26B's k_eq_v layout) with random-but-fixed weights and a minimal SPM vocab, written
// as a GGUF to a temp file - no data dependencies, CPU only, registered with ctest. The prompt is
// longer than the sliding window, so the window mask and the sliding cache are exercised too.
//
// Every arm decodes the same prompt then n_predict greedy steps; each -fa off arm (one prompt
// ubatch, and a small ubatch that splits the prompt across several stores) must match the
// -fa on arm: same greedy tokens and logits within a tolerance far below what the scrambled
// store produces (the two paths reduce in different orders, so bits are not expected to match).
//   test-gemma4-fa-off-vstore [model.gguf|-] [n_predict] [n_gpu_layers]
#include "llama.h"
#include "ggml.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "pxa-test-portable.h"

// ---------------------------------------------------------------------------------------------
// the tiny Gemma 4
// ---------------------------------------------------------------------------------------------

static const int N_VOCAB      = 256;
static const int N_EMBD       = 128;
static const int N_FF         = 256;
static const int N_LAYER      = 3;
static const int N_HEAD       = 4;
static const int HEAD_SWA     = 64;    // sliding blocks: head 64, 2 KV heads, own V projection
static const int N_HEAD_KV_SWA = 2;
static const int HEAD_FULL    = 128;   // global block: head 128, ONE KV head, V = K (no attn_v)
static const int N_HEAD_KV_FULL = 1;
static const int N_SWA        = 8;
static const bool IS_SWA[N_LAYER] = { true, true, false };

static uint32_t g_rng = 20261022u;

static float frand() { // deterministic, small, zero-mean
    g_rng = g_rng*1664525u + 1013904223u;
    return (((float) (g_rng >> 8) / (float) (1u << 23)) - 1.0f) * 0.25f;
}

static void mk(ggml_context * ctx, gguf_context * gg, const char * name, int64_t ne0, int64_t ne1 = 0, float base = 1.0f) {
    ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                              : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = ne1 > 0 ? frand() : base + 0.1f*frand();   // norm gains near `base`
    }
    gguf_add_tensor(gg, t);
}

static bool write_tiny_gemma4(const char * path) {
    ggml_init_params ip = { /*.mem_size =*/ 32u*1024u*1024u, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { return false; }
    gguf_context * gg = gguf_init_empty();
    gguf_set_val_str(gg, "general.architecture", "gemma4");
    gguf_set_val_str(gg, "general.name",         "tiny-gemma4-fa-off-vstore");
    gguf_set_val_u32(gg, "general.file_type",    0);
    gguf_set_val_u32(gg, "gemma4.block_count",              N_LAYER);
    gguf_set_val_u32(gg, "gemma4.context_length",           4096);
    gguf_set_val_u32(gg, "gemma4.embedding_length",         N_EMBD);
    gguf_set_val_u32(gg, "gemma4.feed_forward_length",      N_FF);
    gguf_set_val_u32(gg, "gemma4.attention.head_count",     N_HEAD);
    int32_t kv_arr[N_LAYER];
    bool    swa_arr[N_LAYER];
    for (int il = 0; il < N_LAYER; ++il) {
        kv_arr[il]  = IS_SWA[il] ? N_HEAD_KV_SWA : N_HEAD_KV_FULL;
        swa_arr[il] = IS_SWA[il];
    }
    gguf_set_arr_data(gg, "gemma4.attention.head_count_kv", GGUF_TYPE_INT32, kv_arr, N_LAYER);
    gguf_set_arr_data(gg, "gemma4.attention.sliding_window_pattern", GGUF_TYPE_BOOL, swa_arr, N_LAYER);
    gguf_set_val_u32(gg, "gemma4.attention.sliding_window",       N_SWA);
    gguf_set_val_u32(gg, "gemma4.attention.shared_kv_layers",     0);
    gguf_set_val_u32(gg, "gemma4.attention.key_length",           HEAD_FULL);
    gguf_set_val_u32(gg, "gemma4.attention.value_length",         HEAD_FULL);
    gguf_set_val_u32(gg, "gemma4.attention.key_length_swa",       HEAD_SWA);
    gguf_set_val_u32(gg, "gemma4.attention.value_length_swa",     HEAD_SWA);
    gguf_set_val_f32(gg, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(gg, "gemma4.embedding_length_per_layer_input", 0);
    gguf_set_val_u32(gg, "gemma4.rope.dimension_count",     HEAD_FULL);
    gguf_set_val_u32(gg, "gemma4.rope.dimension_count_swa", HEAD_SWA);
    gguf_set_val_f32(gg, "gemma4.rope.freq_base",           1000000.0f);
    gguf_set_val_f32(gg, "gemma4.rope.freq_base_swa",       10000.0f);
    gguf_set_val_f32(gg, "gemma4.final_logit_softcapping",  30.0f);
    // a minimal SPM vocab with a BOS
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
    mk(ctx, gg, "rope_freqs.weight", HEAD_FULL/2);   // proportional rope of the global block: factors near 1
    for (int il = 0; il < N_LAYER; ++il) {
        const int hd = IS_SWA[il] ? HEAD_SWA : HEAD_FULL;
        const int nk = IS_SWA[il] ? N_HEAD_KV_SWA : N_HEAD_KV_FULL;
        char n[64];
        snprintf(n, sizeof(n), "blk.%d.attn_norm.weight",           il); mk(ctx, gg, n, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.attn_q.weight",              il); mk(ctx, gg, n, N_EMBD, N_HEAD*hd);
        snprintf(n, sizeof(n), "blk.%d.attn_k.weight",              il); mk(ctx, gg, n, N_EMBD, nk*hd);
        if (IS_SWA[il]) {
            snprintf(n, sizeof(n), "blk.%d.attn_v.weight",          il); mk(ctx, gg, n, N_EMBD, nk*hd);
        }
        snprintf(n, sizeof(n), "blk.%d.attn_output.weight",         il); mk(ctx, gg, n, N_HEAD*hd, N_EMBD);
        // q/k norm gains: scale 1.0 attention (no 1/sqrt(d)), so keep the logits in a sane range
        snprintf(n, sizeof(n), "blk.%d.attn_q_norm.weight",         il); mk(ctx, gg, n, hd, 0, 0.35f);
        snprintf(n, sizeof(n), "blk.%d.attn_k_norm.weight",         il); mk(ctx, gg, n, hd, 0, 0.35f);
        snprintf(n, sizeof(n), "blk.%d.post_attention_norm.weight", il); mk(ctx, gg, n, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.layer_output_scale.weight",  il); mk(ctx, gg, n, 1);
        snprintf(n, sizeof(n), "blk.%d.ffn_norm.weight",            il); mk(ctx, gg, n, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.ffn_gate.weight",            il); mk(ctx, gg, n, N_EMBD, N_FF);
        snprintf(n, sizeof(n), "blk.%d.ffn_up.weight",              il); mk(ctx, gg, n, N_EMBD, N_FF);
        snprintf(n, sizeof(n), "blk.%d.ffn_down.weight",            il); mk(ctx, gg, n, N_FF, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.post_ffw_norm.weight",       il); mk(ctx, gg, n, N_EMBD);
    }
    mk(ctx, gg, "output_norm.weight", N_EMBD);

    gguf_write_to_file(gg, path, false);
    gguf_free(gg);
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------------------------
// one arm: prompt, then n_predict greedy steps; keep every step's logits
// ---------------------------------------------------------------------------------------------

struct arm_out {
    std::vector<llama_token>        toks;
    std::vector<std::vector<float>> logits;
};

static llama_token greedy(const float * lg, int n_vocab) {
    llama_token best = 0;
    for (int i = 1; i < n_vocab; ++i) { if (lg[i] > lg[best]) { best = i; } }
    return best;
}

static bool run_arm(const llama_model * model, bool fa, int n_ubatch, const std::vector<llama_token> & prompt,
                    int n_predict, int n_vocab, arm_out & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 256;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn      = fa;
    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx) { fprintf(stderr, "context failed (fa=%d ub=%d)\n", (int) fa, n_ubatch); return false; }

    const int n = (int) prompt.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = prompt[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (int8_t) (i == n - 1);
    }
    b.n_tokens = n;
    bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);

    llama_pos pos = n;
    for (int t = 0; ok && t < n_predict; ++t) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { ok = false; break; }
        const llama_token best = greedy(lg, n_vocab);
        out.toks.push_back(best);
        out.logits.emplace_back(lg, lg + n_vocab);
        if (t + 1 == n_predict) { break; }
        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0;
        d.logits[0] = 1; d.n_tokens = 1;
        ok = llama_decode(ctx, d) == 0;
        llama_batch_free(d);
    }
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool own_model = argc < 2 || strcmp(argv[1], "-") == 0;
    const int  n_predict = argc > 2 ? atoi(argv[2]) : 12;
    const int  n_gpu     = argc > 3 ? atoi(argv[3]) : 0;

    // one file per process: the -bug twin runs the same binary and ctest -j runs them side by side
    const std::string own_name = "test-gemma4-fa-off-vstore-" + std::to_string(pxa_getpid()) + ".gguf";
    std::string path = own_model ? "./" + own_name : argv[1];
    if (own_model) {
        if (const char * t = getenv("TMPDIR")) { path = std::string(t) + "/" + own_name; }
        if (!write_tiny_gemma4(path.c_str())) { fprintf(stderr, "could not write the test model\n"); return 1; }
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (own_model) { remove(path.c_str()); }
    if (!model) { fprintf(stderr, "load failed: %s\n", path.c_str()); return 1; }
    printf("test-gemma4-fa-off-vstore: -fa off must give the -fa on logits (bug #332, PXA_VTRANS_2D=%s)\n",
           getenv("PXA_VTRANS_2D") ? getenv("PXA_VTRANS_2D") : "default");
    const int n_vocab = llama_n_vocab(model);

    // 30 tokens: longer than the sliding window (8), so the window mask and the sliding cache work
    std::vector<llama_token> prompt(30);
    prompt[0] = 1;
    for (size_t i = 1; i < prompt.size(); ++i) {
        prompt[i] = (llama_token) (5 + (i*37 + 11) % (n_vocab - 8));
    }

    arm_out ref;
    if (!run_arm(model, true, 256, prompt, n_predict, n_vocab, ref)) {
        fprintf(stderr, "FAIL: the -fa on arm did not decode\n");
        return 1;
    }
    double ref_max = 0.0;
    for (const auto & row : ref.logits) { for (float v : row) { ref_max = std::max(ref_max, (double) std::fabs(v)); } }

    // tolerance: the two attention paths reduce in different orders over f16 K/V; the scrambled
    // store moves the logits by a sizeable fraction of their range (see the commit message)
    const double tol = 2e-3 * std::max(1.0, ref_max);
    struct arm_cfg { const char * name; int n_ubatch; };
    const arm_cfg cfgs[] = { { "fa off, one prompt ubatch", 256 }, { "fa off, ubatch 7", 7 } };

    int n_fail = 0;
    for (const auto & c : cfgs) {
        arm_out out;
        if (!run_arm(model, false, c.n_ubatch, prompt, n_predict, n_vocab, out)) {
            fprintf(stderr, "FAIL: arm '%s' did not decode\n", c.name);
            n_fail++;
            continue;
        }
        double max_d = 0.0;
        int    first_tok_diff = -1;
        for (int t = 0; t < n_predict; ++t) {
            for (int i = 0; i < n_vocab; ++i) {
                max_d = std::max(max_d, (double) std::fabs(out.logits[t][i] - ref.logits[t][i]));
            }
            if (first_tok_diff < 0 && out.toks[t] != ref.toks[t]) { first_tok_diff = t; }
        }
        const bool ok = max_d <= tol && first_tok_diff < 0;
        printf("%-28s max |logit - fa on| = %.3e (tol %.1e, |logit| max %.3f)%s %s\n", c.name, max_d, tol, ref_max,
               first_tok_diff >= 0 ? (std::string(", greedy differs at step ") + std::to_string(first_tok_diff)).c_str() : "",
               ok ? "OK" : "DIFFERS");
        if (!ok) { n_fail++; }
    }
    fflush(stdout);

    llama_free_model(model);
    llama_backend_free();

    if (n_fail) {
        fprintf(stderr, "FAIL: %d -fa off arm(s) do not match -fa on\n", n_fail);
        return 1;
    }
    printf("OK: -fa off == -fa on (greedy %d steps, logits within tolerance)\n", n_predict);
    return 0;
}
