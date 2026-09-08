// PXA_KV_SEQ_SOA end-to-end parity: drive a real llama_context on the CPU through every KV
// mutation the server uses (multi-seq prompt, seq_cp, seq_rm, seq_add, seq_div, defrag, seq_keep,
// clear), and after every decode hash the bytes of the KQ_mask input tensor that llama_set_inputs
// produced. Run the two arms and diff stdout: the predicate is unchanged, so the hash lines must
// be identical.
//
// NOTE (2026-09-03): PXA_KV_SEQ_SOA now defaults ON at the ENHANCE config level, which is itself
// the default, so the arms are `PXA_ENHANCE=0` (OFF, the red-black-tree find) against a bare run
// (ON, the inline shadow). Setting PXA_KV_SEQ_SOA=0/1 by hand still overrides either way.
//
//   test-kv-mask-parity <model.gguf> [fa|nofa]
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

struct capture {
    std::set<const ggml_tensor *> seen;
    std::vector<std::string> lines;
};

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static bool is_mask(const ggml_tensor * t) {
    return t && (strncmp(t->name, "KQ_mask", 7) == 0);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    capture * cap = (capture *) ud;
    bool any = false;
    for (int k = 0; k < GGML_MAX_SRC; ++k) if (is_mask(t->src[k])) any = true;
    if (ask) return any;
    if (!any) return true;
    for (int k = 0; k < GGML_MAX_SRC; ++k) {
        const ggml_tensor * m = t->src[k];
        if (!is_mask(m) || cap->seen.count(m)) continue;
        cap->seen.insert(m);
        std::vector<uint8_t> buf(ggml_nbytes(m));
        ggml_backend_tensor_get(m, buf.data(), 0, buf.size());
        // count finite (non -inf) entries in row 0 so the log shows the mask is non-trivial
        int n_open = 0;
        if (m->type == GGML_TYPE_F16) {
            const ggml_fp16_t * d = (const ggml_fp16_t *) buf.data();
            for (int64_t i = 0; i < m->ne[0]; ++i) if (ggml_fp16_to_fp32(d[i]) == 0.0f) n_open++;
        } else {
            const float * d = (const float *) buf.data();
            for (int64_t i = 0; i < m->ne[0]; ++i) if (d[i] == 0.0f) n_open++;
        }
        char line[256];
        snprintf(line, sizeof(line), "  %s type=%s ne=[%lld,%lld] row0_open=%d hash=%016llx",
                 m->name, ggml_type_name(m->type), (long long) m->ne[0], (long long) m->ne[1], n_open, (unsigned long long) fnv1a(buf.data(), buf.size()));
        cap->lines.push_back(line);
    }
    return true;
}

static llama_batch mk(const std::vector<std::pair<llama_seq_id, llama_pos>> & toks, int n_vocab) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = (llama_token) ((toks[i].first*7919 + toks[i].second*104729 + 3) % n_vocab);
        b.pos[i]       = toks[i].second;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = toks[i].first;
        b.logits[i]    = 1;
    }
    b.n_tokens = (int) toks.size();
    return b;
}

static int step(llama_context * ctx, capture & cap, const char * label, std::vector<std::pair<llama_seq_id, llama_pos>> toks, int n_vocab) {
    cap.seen.clear(); cap.lines.clear();
    llama_batch b = mk(toks, n_vocab);
    const int ret = llama_decode(ctx, b);
    llama_batch_free(b);
    printf("step %-28s n_tokens=%zu ret=%d\n", label, toks.size(), ret);
    for (auto & l : cap.lines) printf("%s\n", l.c_str());
    if (cap.lines.empty()) printf("  (no KQ_mask captured)\n");
    return ret;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [fa|nofa]\n", argv[0]); return 2; }
    const bool fa = argc < 3 || strcmp(argv[2], "fa") == 0;
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const int n_vocab = llama_n_vocab(model);

    capture cap;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 6;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.flash_attn = fa;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    printf("model=%s fa=%d n_vocab=%d PXA_KV_SEQ_SOA=%s\n", argv[1], (int) fa, n_vocab, getenv("PXA_KV_SEQ_SOA") ? getenv("PXA_KV_SEQ_SOA") : "(unset)");

    std::vector<std::pair<llama_seq_id, llama_pos>> t;
    // 1. four sequences, 8 tokens each, one batch
    for (llama_seq_id s = 0; s < 4; ++s) for (llama_pos p = 0; p < 8; ++p) t.push_back({s, p});
    step(ctx, cap, "prompt4x8", t, n_vocab);
    // 2. seq_cp 0->4 (multi-seq cells), decode one token on each of 0..4
    llama_kv_cache_seq_cp(ctx, 0, 4, -1, -1);
    t.clear(); for (llama_seq_id s = 0; s < 5; ++s) t.push_back({s, 8});
    step(ctx, cap, "after_seq_cp", t, n_vocab);
    // 3. rm / add / div
    llama_kv_cache_seq_rm (ctx, 1, 4, -1);
    llama_kv_cache_seq_add(ctx, 2, -1, -1, 3);
    llama_kv_cache_seq_div(ctx, 3, -1, -1, 2);
    t = {{1, 4}, {2, 12}, {3, 5}};
    step(ctx, cap, "after_rm_add_div", t, n_vocab);
    // 4. holes + defrag, then decode-shaped single-token steps
    llama_kv_cache_seq_rm(ctx, 0, 2, 5);
    llama_kv_cache_seq_rm(ctx, 4, 6, 8);
    llama_kv_cache_defrag(ctx);
    llama_kv_cache_update(ctx);
    step(ctx, cap, "defrag_decode_s0", {{0, 9}}, n_vocab);
    step(ctx, cap, "defrag_decode_s4", {{4, 9}}, n_vocab);
    step(ctx, cap, "decode_s2", {{2, 13}}, n_vocab);
    // 5. keep only seq 4
    llama_kv_cache_seq_keep(ctx, 4);
    step(ctx, cap, "after_keep4", {{4, 10}}, n_vocab);
    // 6. clear, fresh sequence
    llama_kv_cache_clear(ctx);
    t.clear(); for (llama_pos p = 0; p < 5; ++p) t.push_back({5, p});
    step(ctx, cap, "after_clear", t, n_vocab);
    step(ctx, cap, "decode_s5", {{5, 5}}, n_vocab);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
