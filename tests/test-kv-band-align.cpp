// KV BAND ALIGNMENT: a sequence's logits must not depend on WHERE its cells were placed.
//
// The unified KV cache is one ring shared by every sequence. `llama_kv_cache_find_slot` lays a
// request's cells down at the first free run it finds, so the same prompt can start at cell 0 in
// one request and at cell 101 in the next, purely because of what another sequence still holds.
// Nothing about the arithmetic of attention is allowed to notice that: every cell of another
// sequence is masked to -inf, so the set of terms that reach the softmax, and the order they
// reach it in, is the same at any starting cell.
//
// This test asserts exactly that, end to end through a real llama_context on the CPU: prefill one
// prompt with the ring empty, then prefill the SAME prompt with a filler sequence of F cells
// already resident, for a set of F that puts the band's first cell on a tile boundary (F = 0,
// 256), one cell past one (F = 1, 257) and deep inside one (F = 100, 101, 137, 255) -- the class
// that the 4x P100 seat answered differently.
//
// UNDER -fa ON THE COMPARISON IS BITWISE, and that strictness is earned rather than hoped for:
// the CPU flash-attention kernel skips a masked key outright (ggml.c, "if (mv == -INFINITY)
// continue"), so the terms that reach the softmax and the order they reach it in do not move when
// the band does. A correct engine returns the same bytes, not merely close ones.
//
// UNDER -fa OFF THE COMPARISON IS A BOUND, and the reason is not a weaker claim about the KV
// cache. That path multiplies the whole padded key range and softmaxes it, so the vector length
// and the base pointer of every KQ row move with the band; the CPU's SIMD reduction order moves
// with them and the logits differ in the last bits (measured: 2.7e-5 on a 4.6 logit, only at
// offsets that are not a multiple of 4 -- the vector width). That is float addition being
// non-associative, not a cache defect, so this arm requires the same argmax and a small bound
// instead of equality.
//
// It is a host-side test by design. Placement, the exact cell_max, the width of the attention
// window, and the mask fill are all backend-independent, so a defect in any of them shows up
// here with no card in the machine. A defect that lives only in a device kernel will not; that
// case is measured on the seat instead.
//
// The model is built in the test: a tiny dense transformer with random-but-fixed weights, written
// as a GGUF to a temp file and loaded through the normal path, so the test has no data
// dependencies and no network.
//
//   test-kv-band-align [fa|nofa|both]

#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

static const int N_VOCAB   = 512;
static const int N_EMBD    = 64;
static const int N_HEAD    = 4;
static const int N_HEAD_KV = 2;
static const int N_LAYER   = 2;
static const int N_FF      = 128;
static const int HEAD_DIM  = N_EMBD / N_HEAD;

static uint32_t g_rng = 20260909u;

static float frand() { // deterministic, small, zero-mean
    g_rng = g_rng*1664525u + 1013904223u;
    return (((float) (g_rng >> 8) / (float) (1u << 23)) - 1.0f) * 0.25f;
}

static ggml_tensor * mk(ggml_context * ctx, gguf_context * gg, const char * name, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                              : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = frand();
    }
    if (ne1 <= 0) { // an rms_norm gain of ~1 keeps the tiny stack in a sane range
        for (int64_t i = 0; i < n; ++i) {
            d[i] = 1.0f + 0.1f*d[i];
        }
    }
    gguf_add_tensor(gg, t);
    return t;
}

static bool write_tiny_model(const char * path) {
    ggml_init_params ip = { /*.mem_size =*/ 16u*1024u*1024u, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    gguf_context * gg = gguf_init_empty();

    gguf_set_val_str(gg, "general.architecture", "llama");
    gguf_set_val_str(gg, "general.name",         "tiny-band-align");
    gguf_set_val_u32(gg, "general.file_type",    0);
    gguf_set_val_u32(gg, "llama.block_count",              N_LAYER);
    gguf_set_val_u32(gg, "llama.context_length",           8192);
    gguf_set_val_u32(gg, "llama.embedding_length",         N_EMBD);
    gguf_set_val_u32(gg, "llama.feed_forward_length",      N_FF);
    gguf_set_val_u32(gg, "llama.attention.head_count",     N_HEAD);
    gguf_set_val_u32(gg, "llama.attention.head_count_kv",  N_HEAD_KV);
    gguf_set_val_u32(gg, "llama.attention.key_length",     HEAD_DIM);
    gguf_set_val_u32(gg, "llama.attention.value_length",   HEAD_DIM);
    gguf_set_val_f32(gg, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gg, "llama.rope.dimension_count",     HEAD_DIM);
    gguf_set_val_f32(gg, "llama.rope.freq_base",           10000.0f);
    gguf_set_val_str(gg, "tokenizer.ggml.model", "none");
    gguf_set_val_u32(gg, "llama.vocab_size",     N_VOCAB);

    mk(ctx, gg, "token_embd.weight", N_EMBD, N_VOCAB);
    for (int il = 0; il < N_LAYER; ++il) {
        char n[64];
        snprintf(n, sizeof(n), "blk.%d.attn_norm.weight",   il); mk(ctx, gg, n, N_EMBD, 0);
        snprintf(n, sizeof(n), "blk.%d.attn_q.weight",      il); mk(ctx, gg, n, N_EMBD, N_HEAD*HEAD_DIM);
        snprintf(n, sizeof(n), "blk.%d.attn_k.weight",      il); mk(ctx, gg, n, N_EMBD, N_HEAD_KV*HEAD_DIM);
        snprintf(n, sizeof(n), "blk.%d.attn_v.weight",      il); mk(ctx, gg, n, N_EMBD, N_HEAD_KV*HEAD_DIM);
        snprintf(n, sizeof(n), "blk.%d.attn_output.weight", il); mk(ctx, gg, n, N_HEAD*HEAD_DIM, N_EMBD);
        snprintf(n, sizeof(n), "blk.%d.ffn_norm.weight",    il); mk(ctx, gg, n, N_EMBD, 0);
        snprintf(n, sizeof(n), "blk.%d.ffn_gate.weight",    il); mk(ctx, gg, n, N_EMBD, N_FF);
        snprintf(n, sizeof(n), "blk.%d.ffn_up.weight",      il); mk(ctx, gg, n, N_EMBD, N_FF);
        snprintf(n, sizeof(n), "blk.%d.ffn_down.weight",    il); mk(ctx, gg, n, N_FF,   N_EMBD);
    }
    mk(ctx, gg, "output_norm.weight", N_EMBD, 0);
    mk(ctx, gg, "output.weight",      N_EMBD, N_VOCAB);

    gguf_write_to_file(gg, path, false);
    gguf_free(gg);
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------------------------
// the arms
// ---------------------------------------------------------------------------------------------

static const int PROMPT_LEN = 300; // spans several ubatches at n_ubatch = 128
static const int N_BATCH    = 128;

static llama_token tok(int i) {
    return (llama_token) ((i*2654435761u + 17u) % (unsigned) N_VOCAB);
}

// feed `n` tokens of sequence `seq` starting at position 0, in chunks of N_BATCH.
// returns the logits of the LAST token when want_logits, else nullptr.
static const float * feed(llama_context * ctx, llama_seq_id seq, int n, bool want_logits) {
    llama_batch b = llama_batch_init(N_BATCH, 0, 1);
    const float * out = nullptr;
    for (int i0 = 0; i0 < n; i0 += N_BATCH) {
        const int cur = n - i0 < N_BATCH ? n - i0 : N_BATCH;
        for (int i = 0; i < cur; ++i) {
            b.token[i]     = tok(i0 + i);
            b.pos[i]       = i0 + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq;
            b.logits[i]    = 0;
        }
        const bool last_chunk = i0 + cur >= n;
        if (want_logits && last_chunk) {
            b.logits[cur - 1] = 1;
        }
        b.n_tokens = cur;
        const int ret = llama_decode(ctx, b);
        if (ret != 0) {
            printf("    llama_decode failed: %d\n", ret);
            llama_batch_free(b);
            return nullptr;
        }
        if (want_logits && last_chunk) {
            out = llama_get_logits_ith(ctx, cur - 1);
        }
    }
    llama_batch_free(b);
    return out;
}

static int g_fail = 0;

static int run(llama_model * model, bool fa) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 2048;
    cp.n_batch         = N_BATCH;
    cp.n_ubatch        = N_BATCH;
    cp.n_seq_max       = 2;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    cp.flash_attn      = fa;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        printf("  ctx init failed (fa=%d)\n", (int) fa);
        return 1;
    }

    // F = the number of cells another sequence already holds, i.e. the cell the subject's band
    // starts at. 0/256 are on the 256-cell attention-window grain, 1/257 one cell past it, and
    // 100/101/137/255 land deep inside a tile -- the class the seat answered differently.
    const int offsets[] = { 0, 1, 100, 101, 137, 255, 256, 257, 384 };
    const int n_off     = (int) (sizeof(offsets)/sizeof(offsets[0]));

    std::vector<float> ref;
    printf("  fa=%-5s  subject = %d tokens on seq 1, n_ubatch=%d\n", fa ? "on" : "off", PROMPT_LEN, N_BATCH);

    for (int a = 0; a < n_off; ++a) {
        const int F = offsets[a];
        llama_kv_cache_clear(ctx);
        if (F > 0) {
            // a co-resident sequence of exactly F cells, so the subject's band starts at cell F
            if (!feed(ctx, 0, F, false) && F > 0) {
                // feed() returns nullptr when want_logits is false; only a decode error prints
            }
        }
        const float * lg = feed(ctx, 1, PROMPT_LEN, true);
        if (!lg) {
            printf("    offset %-4d DECODE FAILED\n", F);
            g_fail = 1;
            continue;
        }
        std::vector<float> cur(lg, lg + N_VOCAB);

        int    top   = 0;
        for (int i = 1; i < N_VOCAB; ++i) {
            if (cur[i] > cur[top]) top = i;
        }

        if (a == 0) {
            ref = cur;
            printf("    offset %-4d top1=%-4d logit=%.9g   (reference)\n", F, top, (double) cur[top]);
            continue;
        }

        double maxdiff = 0.0;
        int    ndiff   = 0;
        int    first   = -1;
        for (int i = 0; i < N_VOCAB; ++i) {
            if (memcmp(&cur[i], &ref[i], sizeof(float)) != 0) {
                ++ndiff;
                if (first < 0) first = i;
                const double d = std::fabs((double) cur[i] - (double) ref[i]);
                if (d > maxdiff) maxdiff = d;
            }
        }
        int reftop = 0;
        for (int i = 1; i < N_VOCAB; ++i) {
            if (ref[i] > ref[reftop]) reftop = i;
        }
        // -fa on: bitwise. -fa off: same argmax and within the CPU's own reduction-order noise.
        const double bound = 1e-3;
        const bool   ok    = fa ? ndiff == 0 : (top == reftop && maxdiff < bound);
        printf("    offset %-4d top1=%-4d logit=%.9g   %s", F, top, (double) cur[top],
               ok ? (ndiff == 0 ? "IDENTICAL" : "within bound") : "DIFFERS");
        if (ndiff != 0) {
            printf("  (%d/%d logits, max|d|=%.6g, first=%d, ref top1=%d)", ndiff, N_VOCAB, maxdiff, first, reftop);
        }
        if (!ok) {
            g_fail = 1;
        }
        printf("\n");
    }

    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "both";

    std::string path = "./test-kv-band-align.gguf";
    if (const char * t = getenv("TMPDIR")) {
        path = std::string(t) + "/test-kv-band-align.gguf";
    }
    if (!write_tiny_model(path.c_str())) {
        printf("could not write the test model\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        printf("could not load the test model at %s\n", path.c_str());
        remove(path.c_str());
        return 1;
    }
    printf("test-kv-band-align: a sequence's logits must not depend on the cell its band starts at\n");

    if (mode != "nofa") run(model, true);
    if (mode != "fa")   run(model, false);

    llama_free_model(model);
    llama_backend_free();
    remove(path.c_str());

    printf("%s\n", g_fail ? "FAIL" : "OK");
    return g_fail;
}
