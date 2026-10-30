// tiny-llama-fixture: writes the model test-eval-callback runs on, so the test needs no download.
//
//   llama-eval-callback-fixture <vocab.gguf> <out.gguf>
//
// A 2-layer llama with random-but-fixed f32 weights (n_embd 32, 4 heads, 2 KV heads, n_ff 64) whose
// tokenizer is copied verbatim from an in-tree vocab-only file (models/ggml-vocab-llama-spm.gguf), so
// a text prompt tokenizes exactly as it would for a real llama-spm model. The weights are not trained:
// the test checks that the eval callback sees every tensor of a real forward pass, not what it says.

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <string>

static const int N_EMBD    = 32;
static const int N_HEAD    = 4;
static const int N_HEAD_KV = 2;
static const int N_LAYER   = 2;
static const int N_FF      = 64;
static const int N_CTX     = 256;
static const int HEAD_DIM  = N_EMBD / N_HEAD;

static uint32_t g_rng = 20261018u;

static float frand() { // deterministic, small, zero-mean
    g_rng = g_rng*1664525u + 1013904223u;
    return (((float) (g_rng >> 8) / (float) (1u << 23)) - 1.0f) * 0.25f;
}

static void add(ggml_context * ctx, gguf_context * gg, const std::string & name, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                              : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name.c_str());
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = ne1 > 0 ? frand() : 1.0f + 0.1f*frand(); // norm gains near 1
    }
    gguf_add_tensor(gg, t);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <vocab.gguf> <out.gguf>\n", argv[0]);
        return 2;
    }
    gguf_init_params vp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * vocab = gguf_init_from_file(argv[1], vp);
    if (!vocab) {
        fprintf(stderr, "%s: cannot read the vocab file %s\n", argv[0], argv[1]);
        return 1;
    }
    const int key = gguf_find_key(vocab, "tokenizer.ggml.tokens");
    if (key < 0) {
        fprintf(stderr, "%s: %s has no tokenizer.ggml.tokens\n", argv[0], argv[1]);
        gguf_free(vocab);
        return 1;
    }
    const int n_vocab = gguf_get_arr_n(vocab, key);

    gguf_context * gg = gguf_init_empty();
    gguf_set_kv(gg, vocab); // every tokenizer.* key (and the source model's hparams, overwritten below)
    gguf_free(vocab);

    gguf_set_val_str(gg, "general.architecture", "llama");
    gguf_set_val_str(gg, "general.name",         "tiny-eval-callback");
    gguf_set_val_u32(gg, "general.file_type",    0);
    gguf_set_val_u32(gg, "llama.block_count",              N_LAYER);
    gguf_set_val_u32(gg, "llama.context_length",           N_CTX);
    gguf_set_val_u32(gg, "llama.embedding_length",         N_EMBD);
    gguf_set_val_u32(gg, "llama.feed_forward_length",      N_FF);
    gguf_set_val_u32(gg, "llama.attention.head_count",     N_HEAD);
    gguf_set_val_u32(gg, "llama.attention.head_count_kv",  N_HEAD_KV);
    gguf_set_val_u32(gg, "llama.attention.key_length",     HEAD_DIM);
    gguf_set_val_u32(gg, "llama.attention.value_length",   HEAD_DIM);
    gguf_set_val_f32(gg, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gg, "llama.rope.dimension_count",     HEAD_DIM);
    gguf_set_val_f32(gg, "llama.rope.freq_base",           10000.0f);
    gguf_set_val_u32(gg, "llama.vocab_size",               n_vocab);

    const size_t n_floats = (size_t) 2*N_EMBD*n_vocab + N_EMBD +
                            (size_t) N_LAYER*(2*N_EMBD + 2*N_EMBD*N_EMBD + 2*N_EMBD*N_HEAD_KV*HEAD_DIM + 3*N_EMBD*N_FF);
    ggml_init_params ip = { /*.mem_size =*/ n_floats*sizeof(float) + 64*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", argv[0]);
        gguf_free(gg);
        return 1;
    }

    add(ctx, gg, "token_embd.weight", N_EMBD, n_vocab);
    for (int il = 0; il < N_LAYER; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        add(ctx, gg, b + "attn_norm.weight",   N_EMBD, 0);
        add(ctx, gg, b + "attn_q.weight",      N_EMBD, N_HEAD*HEAD_DIM);
        add(ctx, gg, b + "attn_k.weight",      N_EMBD, N_HEAD_KV*HEAD_DIM);
        add(ctx, gg, b + "attn_v.weight",      N_EMBD, N_HEAD_KV*HEAD_DIM);
        add(ctx, gg, b + "attn_output.weight", N_HEAD*HEAD_DIM, N_EMBD);
        add(ctx, gg, b + "ffn_norm.weight",    N_EMBD, 0);
        add(ctx, gg, b + "ffn_gate.weight",    N_EMBD, N_FF);
        add(ctx, gg, b + "ffn_up.weight",      N_EMBD, N_FF);
        add(ctx, gg, b + "ffn_down.weight",    N_FF,   N_EMBD);
    }
    add(ctx, gg, "output_norm.weight", N_EMBD, 0);
    add(ctx, gg, "output.weight",      N_EMBD, n_vocab);

    gguf_write_to_file(gg, argv[2], false);
    gguf_free(gg);
    ggml_free(ctx);
    printf("%s: wrote %s (llama, %d layers, n_embd %d, n_vocab %d)\n", argv[0], argv[2], N_LAYER, N_EMBD, n_vocab);
    return 0;
}
