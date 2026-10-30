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
// here with no card in the machine. In a CUDA build with a card the same ngl 0 run still puts the
// attention chain on the card (the 128-token ubatches offload their weight GEMMs and the chain
// follows them), and this model's head size 16 has no fused flash-attention kernel there, so
// PXA_FA_GPU_FALLBACK builds the unfused chain for -fa on as well: on a V100 both arms then run
// cuBLAS's tensor-core P.V product, which reduced the cell axis four cells at a time from cell 0
// and answered 0.0107..0.0133 apart at offsets that are not a multiple of 4 (todo
// v100-kv-placement-dependence; fixed by PXA_KV_BAND_EXACT, ggml-cuda/pxa/kv-band.cuh). Offsets
// 0..8 walk every phase of that grain and of an 8-cell one, and three one-token decode steps
// follow each prefill (the shape a server slot runs most). `gpu` runs both arms again with every
// layer and the cache on a card (a no-op without one); PXA_KV_BAND_STRICT=1 holds -fa off to
// bitwise as well, for a run where the whole chain is on the device.
//
// The model is built in the test: a tiny dense transformer with random-but-fixed weights, written
// as a GGUF to a temp file and loaded through the normal path, so the test has no data
// dependencies and no network.
//
//   test-kv-band-align [fa|nofa|both|gpu|mha|gpu-mha]
//
// PXA_KV_BAND_ANCHOR0 (default on): at offset 0 the host proves every row opens at cell 0, so the reference
// run takes the plain path (no band work) and every other offset, which keeps the band, is compared with it:
// the skip is the band's own answer byte for byte. PXA_KV_BAND_ANCHOR0=0 runs the band at offset 0 too.
//
// "mha" / "gpu-mha" build the model with as many KV heads as query heads (no GQA). On a tensor-core
// card (cc >= 7.0) the banded P.V product then takes the pointer-array route like the GQA modes (B4:
// the cache view at anchor 0, the shifted copy past it), and a proven offset-0 call keeps that route,
// so the gpu-mha row checks that the routes match; Pascal keeps the strided cuBLAS route, which reads
// the shifted copy of V. PXA_KV_BAND_STRIDED_COPY=1 restores the strided copy on every card (A/B only).

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
static int       N_HEAD_KV = 2; // "mha"/"gpu-mha" set N_HEAD (before the GGUF is written)
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

// decode steps after the prefill: a one-token ubatch is the shape a server slot runs most, and its
// attention reduces over the same band with one query row (a different kernel than the prefill's).
static const int N_DECODE = 3;

// one token of sequence `seq` at position `pos`; returns its logits or nullptr
static const float * step(llama_context * ctx, llama_seq_id seq, int pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0]     = tok(pos);
    b.pos[0]       = pos;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = seq;
    b.logits[0]    = 1;
    b.n_tokens     = 1;
    const int ret = llama_decode(ctx, b);
    llama_batch_free(b);
    if (ret != 0) {
        printf("    llama_decode (step) failed: %d\n", ret);
        return nullptr;
    }
    return llama_get_logits_ith(ctx, 0);
}

static int argmax(const float * v) {
    int top = 0;
    for (int i = 1; i < N_VOCAB; ++i) {
        if (v[i] > v[top]) top = i;
    }
    return top;
}

// strict: every arm must be bitwise (PXA_KV_BAND_STRICT=1 also holds -fa off to it, for a run on a card
// where the whole chain is on the device and the CPU's SIMD reduction-order noise never enters).
static int run(llama_model * model, bool fa, int ngl_label) {
    llama_context_params cp = llama_context_default_params();
    // KVBA_NCTX (default 2048): 1024 puts the last offsets' moved reads (cell 384 + n_kv ~704) past the end of the
    // ring, so the Pascal pointer route reads its last V rows from the zero pad (PXA_KV_BAND_VPAD) or the copy
    cp.n_ctx           = getenv("KVBA_NCTX") && atoi(getenv("KVBA_NCTX")) > 0 ? (uint32_t) atoi(getenv("KVBA_NCTX")) : 2048;
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
    const bool strict = fa || (getenv("PXA_KV_BAND_STRICT") && atoi(getenv("PXA_KV_BAND_STRICT")) != 0);

    // F = the number of cells another sequence already holds, i.e. the cell the subject's band
    // starts at. 0/256 are on the 256-cell attention-window grain, 1/257 one cell past it, and
    // 100/101/137/255 land deep inside a tile -- the class the seat answered differently. 0..8
    // walk every phase of a 4- and an 8-cell grain: a V100's tensor cores reduce the P.V product
    // over the cell axis 4 cells at a time, so a band at an offset that is not a multiple of 4
    // answered differently there (v100-kv-placement-dependence, 10-19).
    const int offsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 137, 255, 256, 257, 384 };
    const int n_off     = (int) (sizeof(offsets)/sizeof(offsets[0]));
    const int n_blk     = 1 + N_DECODE; // the prefill's last logits, then one row per decode step

    std::vector<float> ref;
    printf("  fa=%-5s ngl=%-2d subject = %d tokens on seq 1 + %d decode steps, n_ubatch=%d\n", fa ? "on" : "off", ngl_label,
           PROMPT_LEN, N_DECODE, N_BATCH);

    for (int a = 0; a < n_off; ++a) {
        const int F = offsets[a];
        llama_kv_cache_clear(ctx);
        if (F > 0) {
            // a co-resident sequence of exactly F cells, so the subject's band starts at cell F
            feed(ctx, 0, F, false);
        }
        const float * lg = feed(ctx, 1, PROMPT_LEN, true);
        if (!lg) {
            printf("    offset %-4d DECODE FAILED\n", F);
            g_fail = 1;
            continue;
        }
        std::vector<float> cur(lg, lg + N_VOCAB);
        for (int j = 0; j < N_DECODE && lg; ++j) {
            lg = step(ctx, 1, PROMPT_LEN + j);
            if (lg) cur.insert(cur.end(), lg, lg + N_VOCAB);
        }
        if ((int) cur.size() != n_blk*N_VOCAB) {
            printf("    offset %-4d DECODE STEP FAILED\n", F);
            g_fail = 1;
            continue;
        }

        const int top = argmax(cur.data());
        if (a == 0) {
            ref = cur;
            // FNV-1a over the reference bytes: two builds or two settings can be compared for identity from the log
            uint64_t h = 1469598103934665603ull;
            const unsigned char * bytes = (const unsigned char *) ref.data();
            for (size_t i = 0; i < ref.size()*sizeof(float); ++i) {
                h = (h ^ bytes[i])*1099511628211ull;
            }
            printf("    offset %-4d top1=%-4d logit=%.9g   (reference, logits fnv1a %016llx)\n", F, top, (double) cur[top],
                   (unsigned long long) h);
            continue;
        }

        double maxdiff = 0.0;
        int    ndiff   = 0;
        int    first   = -1;
        for (int i = 0; i < n_blk*N_VOCAB; ++i) {
            if (memcmp(&cur[i], &ref[i], sizeof(float)) != 0) {
                ++ndiff;
                if (first < 0) first = i;
                const double d = std::fabs((double) cur[i] - (double) ref[i]);
                if (d > maxdiff) maxdiff = d;
            }
        }
        bool same_top = true;
        for (int k = 0; k < n_blk; ++k) {
            same_top = same_top && argmax(cur.data() + k*N_VOCAB) == argmax(ref.data() + k*N_VOCAB);
        }
        // -fa on: bitwise. -fa off: same argmax and within the CPU's own reduction-order noise.
        const double bound = 1e-3;
        const bool   ok    = strict ? ndiff == 0 : (same_top && maxdiff < bound);
        printf("    offset %-4d top1=%-4d logit=%.9g   %s", F, top, (double) cur[top],
               ok ? (ndiff == 0 ? "IDENTICAL" : "within bound") : "DIFFERS");
        if (ndiff != 0) {
            printf("  (%d/%d logits, max|d|=%.6g, first=%d (row %d), same top1 every row=%d)", ndiff, n_blk*N_VOCAB, maxdiff,
                   first, first / N_VOCAB, (int) same_top);
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

    const bool mha = mode == "mha" || mode == "gpu-mha";
    if (mha) {
        N_HEAD_KV = N_HEAD;
    }
    // one file per mode: ctest -j runs the entries side by side
    std::string path = "./test-kv-band-align-" + mode + ".gguf";
    if (const char * t = getenv("TMPDIR")) {
        path = std::string(t) + "/test-kv-band-align-" + mode + ".gguf";
    }
    if (!write_tiny_model(path.c_str())) {
        printf("could not write the test model\n");
        return 1;
    }

    // "gpu" runs both arms again with every layer (and the cache) on a card, when there is one: the
    // shape a seat runs. The default modes stay host-side (ngl 0), so the test means the same thing
    // in a CPU-only build.
    const bool gpu = mode == "gpu" || mode == "gpu-mha";
    llama_backend_init();
    if (gpu && !llama_supports_gpu_offload()) {
        printf("test-kv-band-align gpu: no GPU offload in this build, nothing to do\nOK\n");
        remove(path.c_str());
        llama_backend_free();
        return 0;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu ? 99 : 0;
    if (gpu) { // one card: a bare ctest must not spread the model over every visible card
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu   = 0;
    }
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        printf("could not load the test model at %s\n", path.c_str());
        remove(path.c_str());
        return 1;
    }
    printf("test-kv-band-align (%s, %d query / %d KV heads): a sequence's logits must not depend on the cell its band starts at\n", mode.c_str(), N_HEAD, N_HEAD_KV);

    if (mode != "nofa") run(model, true,  mp.n_gpu_layers);
    if (mode != "fa")   run(model, false, mp.n_gpu_layers);

    llama_free_model(model);
    llama_backend_free();
    remove(path.c_str());

    printf("%s\n", g_fail ? "FAIL" : "OK");
    return g_fail;
}
