// Hybrid (attention + Gated DeltaNet) KV / recurrent-state regressions, CPU only, on the tiny
// qwen4exp fixture (tests/gen_tiny_qwen4exp.py; add --ple for the PLE case).
//
//   test-kv-hybrid-state tiny-qwen4exp.gguf [tiny-qwen4exp-ple.gguf]
//
// Each case is a bug from the 2026-09-25 engine bug hunt, measured end to end through the
// public API. A case compares logits (or state sizes) against a reference context that took
// the plain path to the same tokens; the broken path differs by orders of magnitude more than
// the batch-shape round-off the tolerance allows for.
//
//   #210  a PARTIAL_ONLY restore at np 2 with interleaved cells keeps the attention cells
//         where their K/V is (it used to re-place them in fresh cells with no K/V bytes)
//   #223  defrag keeps the per-index recurrent state-row permutation (cells[i].src), so a
//         sequence's state blob still carries its state row after a defrag
//   #218  llama_decode never returns 1 after committing earlier ubatches of the call; a
//         fragmented ring is compacted and the whole batch lands exactly once, also when
//         the compaction needs more than one defrag pass; a batch larger than the free
//         cells is refused with 1 before any ubatch is committed
//   #231  gpu-fallback recurrent checkpoints resolve to the per-sequence cpu mode at np 2
//   #214  a partially rejected speculative verify keeps the qwen4exp PLE conv window and
//         n-gram tail, also 40 positions deep; a pre-fix PLE state blob still restores
//         (PLE fixture only)
//   and   defrag of a live hybrid sequence moves its V rows in the layout they were written in
//         (hybrids never transpose V; defrag keyed the layout on flash_attn instead)
//
// Every case runs with flash attention off and on (the V cache layout differs).
//
// Exit 0 when every case passes.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct tok { llama_token t; llama_pos p; llama_seq_id s; bool out; };

llama_model * g_model   = nullptr;
int           g_n_vocab = 0;
int           g_fail    = 0;
bool          g_fa      = false;   // the suite runs twice: V transposed (off) and not (on)

llama_context * make_ctx(llama_model * model, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch, uint32_t n_seq) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = n_seq;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn      = g_fa;
    cp.defrag_thold    = -1.0f;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context creation failed\n");
        exit(1);
    }
    return ctx;
}

int decode(llama_context * ctx, const std::vector<tok> & ts) {
    llama_batch b = llama_batch_init((int32_t) ts.size(), 0, 1);
    for (size_t i = 0; i < ts.size(); ++i) {
        b.token[i]     = ts[i].t;
        b.pos[i]       = ts[i].p;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = ts[i].s;
        b.logits[i]    = ts[i].out;
    }
    b.n_tokens = (int32_t) ts.size();
    const int ret = llama_decode(ctx, b);
    llama_batch_free(b);
    return ret;
}

std::vector<float> logits_of(llama_context * ctx, int i) {
    const float * l = llama_get_logits_ith(ctx, i);
    return std::vector<float>(l, l + g_n_vocab);
}

float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

llama_token T(int i) { return (llama_token) (3 + (i * 37 + 11) % 250); }  // byte tokens

void verdict(const char * name, bool ok, const char * fmt, double v) {
    printf("%-6s [fa %s] %s  ", ok ? "PASS" : "FAIL", g_fa ? "on " : "off", name);
    printf(fmt, v);
    printf("\n");
    if (!ok) g_fail++;
}

constexpr float TOL = 1e-3f;

// ---- #210 -------------------------------------------------------------------------------
void test_partial_restore_interleaved() {
    const int n0 = 12;
    auto prefix = [&](llama_context * ctx) {
        for (int i = 0; i < n0; ++i) {
            // one token per sequence per decode: the cells of seq 0 and seq 1 interleave
            if (decode(ctx, { { T(i), i, 0, false }, { T(100 + i), i, 1, false } }) != 0) {
                fprintf(stderr, "#210 prefix decode failed\n"); exit(1);
            }
        }
    };

    llama_context * ref = make_ctx(g_model, 256, 64, 64, 2);
    prefix(ref);
    decode(ref, { { T(50), n0, 0, true } });
    const auto l_ref = logits_of(ref, 0);
    llama_free(ref);

    llama_context * ctx = make_ctx(g_model, 256, 64, 64, 2);
    prefix(ctx);
    const size_t sz = llama_state_seq_get_size(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> blob(sz);
    const size_t got = llama_state_seq_get_data(ctx, blob.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    blob.resize(got);
    // both sequences move on, then seq 0 rolls back to the checkpoint
    for (int i = n0; i < n0 + 4; ++i) {
        decode(ctx, { { T(200 + i), i, 0, false }, { T(300 + i), i, 1, false } });
    }
    const size_t set = llama_state_seq_set_data(ctx, blob.data(), blob.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (set != blob.size()) {
        verdict("#210 partial restore np2", false, "set_data returned %.0f", (double) set);
        llama_free(ctx);
        return;
    }
    decode(ctx, { { T(50), n0, 0, true } });
    const auto l_got = logits_of(ctx, 0);
    const llama_pos pmax = llama_kv_cache_seq_pos_max(ctx, 0);
    llama_free(ctx);

    const float d = max_diff(l_ref, l_got);
    verdict("#210 partial restore np2", d < TOL && pmax == n0, "max|dlogit| = %.3g", d);
}

// ---- #223 -------------------------------------------------------------------------------
void test_defrag_keeps_state_rows() {
    llama_context * ref = make_ctx(g_model, 256, 64, 64, 2);
    std::vector<tok> s1;
    for (int i = 0; i < 4; ++i) s1.push_back({ T(10 + i), i, 1, false });
    decode(ref, s1);
    const size_t want = llama_state_seq_get_size(ref, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    llama_free(ref);

    llama_context * ctx = make_ctx(g_model, 256, 64, 64, 2);
    std::vector<tok> s0;
    for (int i = 0; i < 4; ++i) s0.push_back({ T(20 + i), i, 0, false });
    decode(ctx, s0);   // cells 0..3
    decode(ctx, s1);   // cells 4..7
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);   // a hole at cells 0..3, below n_seq
    llama_kv_cache_defrag(ctx);
    llama_kv_cache_update(ctx);              // cells 4..7 move down into 0..3
    const size_t got = llama_state_seq_get_size(ctx, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    llama_free(ctx);

    verdict("#223 defrag keeps state row", got == want, "state blob bytes short by %.0f", (double) want - (double) got);
}

// ---- #218 -------------------------------------------------------------------------------
void test_decode_no_partial_commit() {
    const int n_pre = 48;   // 96 alternating cells, then seq 1 is freed: 48 one-cell holes
    const int n_new = 40;   // > the 32-cell free tail, so its third ubatch finds no run

    std::vector<tok> fresh;
    for (int i = 0; i < n_new; ++i) fresh.push_back({ T(400 + i), i, 1, i == n_new - 1 });

    llama_context * ref = make_ctx(g_model, 128, 64, 16, 2);
    if (decode(ref, fresh) != 0) { fprintf(stderr, "#218 reference decode failed\n"); exit(1); }
    const auto l_ref = logits_of(ref, n_new - 1);
    llama_free(ref);

    llama_context * ctx = make_ctx(g_model, 128, 64, 16, 2);
    for (int i = 0; i < n_pre; ++i) {
        decode(ctx, { { T(i), i, 0, false }, { T(100 + i), i, 1, false } });
    }
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);
    const int ret = decode(ctx, fresh);
    const int used = llama_get_kv_cache_used_cells(ctx);
    const llama_pos pmax = llama_kv_cache_seq_pos_max(ctx, 1);
    std::vector<float> l_got;
    if (ret == 0) l_got = logits_of(ctx, n_new - 1);
    llama_free(ctx);

    if (ret != 0) {
        // the old behaviour: 1 after two ubatches were already committed (a caller retry
        // would then decode them twice)
        verdict("#218 decode across a fragmented ring", false, "llama_decode returned %.0f", (double) ret);
        printf("       used cells after the refused call: %d (committed before refusal: %d)\n",
               used, used - n_pre);
        return;
    }
    const float d = max_diff(l_ref, l_got);
    const bool ok = used == n_pre + n_new && pmax == n_new - 1 && d < TOL;
    verdict("#218 decode across a fragmented ring", ok, "max|dlogit| = %.3g", d);
    if (!ok) printf("       used=%d (want %d) pos_max=%d (want %d)\n", used, n_pre + n_new, (int) pmax, n_new - 1);
}

// ---- #218, heavy fragmentation: one defrag pass (max_moves blocks) is not enough -----------
void test_decode_multi_pass_defrag() {
    // 430 one-cell holes below a 164-cell free tail. The first 160-token ubatch lands in the
    // tail; the second needs 160 cells, and one defrag pass fills at most max_moves holes
    // (136 on this fixture: (32*308 - 2*12)/(6*12)), freeing 136 + 4 < 160 tail cells, so the
    // mid-call compaction must run a second pass.
    const int n_pre = 430;
    const int n_new = 320;

    std::vector<tok> fresh;
    for (int i = 0; i < n_new; ++i) fresh.push_back({ T(700 + i), i, 1, i == n_new - 1 });

    llama_context * ref = make_ctx(g_model, 1024, 512, 160, 2);
    if (decode(ref, fresh) != 0) { fprintf(stderr, "#218 multi-pass reference decode failed\n"); exit(1); }
    const auto l_ref = logits_of(ref, n_new - 1);
    llama_free(ref);

    llama_context * ctx = make_ctx(g_model, 1024, 512, 160, 2);
    for (int i = 0; i < n_pre; ++i) {
        decode(ctx, { { T(i), i, 0, false }, { T(100 + i), i, 1, false } });
    }
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);
    const int ret = decode(ctx, fresh);
    const int used = llama_get_kv_cache_used_cells(ctx);
    std::vector<float> l_got;
    if (ret == 0) l_got = logits_of(ctx, n_new - 1);
    llama_free(ctx);

    if (ret != 0) {
        verdict("#218 decode across a heavily fragmented ring", false, "llama_decode returned %.0f", (double) ret);
        return;
    }
    const float d = max_diff(l_ref, l_got);
    const bool ok = used == n_pre + n_new && d < TOL;
    verdict("#218 decode across a heavily fragmented ring", ok, "max|dlogit| = %.3g", d);
    if (!ok) printf("       used=%d (want %d)\n", used, n_pre + n_new);
}

// ---- defrag of a LIVE hybrid sequence (the server's reactive defrag) ----------------------
void test_defrag_live_seq() {
    std::vector<tok> a, b, c;
    for (int i = 0; i < 8; ++i) {
        a.push_back({ T(600 + i), i, 0, false });
        b.push_back({ T(700 + i), i, 1, false });
        c.push_back({ T(800 + i), i, 2, false });
    }
    llama_context * ref = make_ctx(g_model, 256, 64, 64, 3);
    decode(ref, c);
    decode(ref, { { T(900), 8, 2, true } });
    const auto l_ref = logits_of(ref, 0);
    llama_free(ref);

    llama_context * ctx = make_ctx(g_model, 256, 64, 64, 3);
    decode(ctx, a);   // cells 0..7
    decode(ctx, b);   // cells 8..15
    decode(ctx, c);   // cells 16..23
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);
    llama_kv_cache_defrag(ctx);
    llama_kv_cache_update(ctx);   // seq 2 moves down into 8..15
    decode(ctx, { { T(900), 8, 2, true } });
    const auto l_got = logits_of(ctx, 0);
    llama_free(ctx);
    const float d = max_diff(l_ref, l_got);
    verdict("defrag of a live hybrid sequence", d < TOL, "max|dlogit| = %.3g", d);
}

// ---- #231 -------------------------------------------------------------------------------
void test_gpu_fallback_np2() {
    llama_context * ctx = make_ctx(g_model, 256, 64, 64, 2);
    const int mode = llama_spec_ckpt_init(ctx, LLAMA_SPEC_CKPT_GPU_FALLBACK, 4);
    llama_free(ctx);
    verdict("#231 gpu-fallback at np2 -> per-seq", mode != LLAMA_SPEC_CKPT_GPU_FALLBACK, "resolved mode %.0f", (double) mode);
}

// ---- #214 -------------------------------------------------------------------------------
void test_ple_spec_rollback(llama_model * model, int mode_req, const char * name, int n_drafts = 3) {
    const int n_p = 10;
    std::vector<tok> prompt;
    for (int i = 0; i < n_p; ++i) prompt.push_back({ T(500 + i), i, 0, false });
    const llama_token t1 = T(77), t2 = T(78);

    llama_context * ref = make_ctx(model, 256, 64, 64, 1);
    decode(ref, prompt);
    decode(ref, { { t1, n_p, 0, true } });
    decode(ref, { { t2, n_p + 1, 0, true } });
    const auto l_ref = logits_of(ref, 0);
    llama_free(ref);

    llama_context * ctx = make_ctx(model, 256, 64, 64, 1);
    decode(ctx, prompt);
    const int mode = llama_spec_ckpt_init(ctx, mode_req, 1 + n_drafts);
    if (mode != mode_req) {
        printf("SKIP   [fa %s] %s (mode %d resolved to %d on this backend)\n", g_fa ? "on " : "off", name, mode_req, mode);
        llama_free(ctx);
        return;
    }
    llama_spec_ckpt_save(ctx, 0);
    // verify t1 plus n_drafts drafts that are all rejected: only t1 is accepted, so the
    // sequence rewinds n_drafts positions
    std::vector<tok> verify = { { t1, n_p, 0, true } };
    for (int d = 1; d <= n_drafts; ++d) verify.push_back({ T(d), n_p + d, 0, true });
    decode(ctx, verify);
    const bool in_place = llama_spec_ckpt_restore(ctx, 0, n_p, 0);
    llama_spec_ckpt_discard(ctx);
    if (!in_place) {
        decode(ctx, { { t1, n_p, 0, true } });   // re-decode the accepted token
    }
    decode(ctx, { { t2, n_p + 1, 0, true } });
    const auto l_got = logits_of(ctx, 0);
    llama_free(ctx);

    const float d = max_diff(l_ref, l_got);
    verdict(name, d < TOL, "max|dlogit| = %.3g", d);
}


// ---- #214: a PLE state blob written before the rollback slack existed still restores -------
// The pre-fix blob had the n-gram tail and conv window at their tap width and no `end` field.
// Build one from a current blob (drop the slack columns / tokens and the end field) and check
// that restoring it continues exactly like the context that wrote it.
void test_ple_legacy_blob(llama_model * model) {
    const char * name = "#214 PLE legacy state blob restores";
    const int n_p = 12;
    std::vector<tok> prompt;
    for (int i = 0; i < n_p; ++i) prompt.push_back({ T(520 + i), i, 0, false });
    const llama_token t1 = T(79);

    llama_context * ref = make_ctx(model, 256, 64, 64, 1);
    decode(ref, prompt);
    const size_t sz = llama_state_seq_get_size(ref, 0, 0);
    std::vector<uint8_t> blob(sz);
    blob.resize(llama_state_seq_get_data(ref, blob.data(), sz, 0, 0));
    decode(ref, { { t1, n_p, 0, true } });
    const auto l_ref = logits_of(ref, 0);
    llama_free(ref);

    // the PLE section ends the blob: have, n_prev, hist, hc_dim, next_pos, valid, end,
    // toks[n_prev], window[hist][hc_dim]. SLACK = the widening, 64.
    const uint32_t SLACK = 64;
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, blob.data() + o, 4); return v; };
    size_t off = SIZE_MAX;
    for (size_t o = 0; o + 28 <= blob.size(); o += 4) {
        const uint64_t n_prev = u32(o + 4), hist = u32(o + 8), hc = u32(o + 12);
        if (u32(o) == 1 && n_prev > SLACK && n_prev < 4096 && hist > SLACK && hist < 4096 && hc > 0 && hc < (1u << 20) &&
                o + 28 + 4*n_prev + 4*hist*hc == blob.size()) {
            off = o;
            break;
        }
    }
    if (off == SIZE_MAX) {
        verdict(name, false, "no PLE section with a %.0f-position slack in the blob", (double) SLACK);
        return;
    }
    const uint32_t n_prev = u32(off + 4), hist = u32(off + 8), hc = u32(off + 12);
    std::vector<uint8_t> old(blob.begin(), blob.begin() + off);
    auto put = [&](uint32_t v) { const uint8_t * b = (const uint8_t *) &v; old.insert(old.end(), b, b + 4); };
    put(1); put(n_prev - SLACK); put(hist - SLACK); put(hc);
    put(u32(off + 16)); put(u32(off + 20));                       // next_pos, valid; no end
    const size_t toks = off + 28, win = toks + 4*(size_t) n_prev;
    old.insert(old.end(), blob.begin() + toks + 4*SLACK, blob.begin() + win);
    old.insert(old.end(), blob.begin() + win + 4*(size_t) SLACK*hc, blob.end());

    llama_context * ctx = make_ctx(model, 256, 64, 64, 1);
    const size_t set = llama_state_seq_set_data(ctx, old.data(), old.size(), 0, 0);
    if (set != old.size()) {
        llama_free(ctx);
        verdict(name, false, "set_data returned %.0f (refused)", (double) set);
        return;
    }
    decode(ctx, { { t1, n_p, 0, true } });
    const auto l_got = logits_of(ctx, 0);
    llama_free(ctx);
    const float d = max_diff(l_ref, l_got);
    verdict(name, d < TOL, "max|dlogit| = %.3g", d);
}

// ---- #218: a batch larger than the free cells is refused before anything is committed -----
void test_decode_refused_whole() {
    llama_context * ctx = make_ctx(g_model, 128, 64, 16, 2);
    const int n_ctx = (int) llama_n_ctx(ctx);
    const int n_fill = n_ctx - 20;   // 20 free cells; the 40-token batch below needs 40
    for (int i = 0; i < n_fill; i += 32) {
        std::vector<tok> ts;
        for (int j = i; j < std::min(n_fill, i + 32); ++j) ts.push_back({ T(j), j, 0, false });
        if (decode(ctx, ts) != 0) { fprintf(stderr, "#218 fill decode failed\n"); exit(1); }
    }
    std::vector<tok> big;
    for (int i = 0; i < 40; ++i) big.push_back({ T(900 + i), i, 1, i == 39 });
    const llama_pos pmax0 = llama_kv_cache_seq_pos_max(ctx, 1);
    const int ret = decode(ctx, big);
    const int used = llama_get_kv_cache_used_cells(ctx);
    const llama_pos pmax = llama_kv_cache_seq_pos_max(ctx, 1);
    llama_free(ctx);
    const bool ok = ret == 1 && used == n_fill && pmax == pmax0;
    verdict("#218 over-full batch refused whole (1, nothing committed)", ok, "llama_decode returned %.0f", (double) ret);
    if (!ok) printf("       used=%d (want %d) seq 1 pos_max=%d (want %d, as before the call)\n", used, n_fill, (int) pmax, (int) pmax0);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s tiny-qwen4exp.gguf [tiny-qwen4exp-ple.gguf]\n", argv[0]);
        return 2;
    }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(argv[1], mp);
    if (!g_model) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    g_n_vocab = llama_n_vocab(g_model);

    llama_model * ple = nullptr;
    if (argc > 2) {
        ple = llama_model_load_from_file(argv[2], mp);
        if (!ple) { fprintf(stderr, "load failed: %s\n", argv[2]); return 1; }
    }

    for (bool fa : { false, true }) {
        g_fa = fa;
        test_partial_restore_interleaved();
        test_defrag_keeps_state_rows();
        test_defrag_live_seq();
        test_decode_no_partial_commit();
        test_decode_multi_pass_defrag();
        test_decode_refused_whole();
        test_gpu_fallback_np2();
        if (ple) {
            test_ple_spec_rollback(ple, LLAMA_SPEC_CKPT_CPU,          "#214 PLE rollback (cpu, control)");
            test_ple_spec_rollback(ple, LLAMA_SPEC_CKPT_GPU_FALLBACK, "#214 PLE rollback (gpu-fallback)");
            test_ple_spec_rollback(ple, LLAMA_SPEC_CKPT_PER_STEP,     "#214 PLE rollback (per-step)");
            // 40 rejected drafts: deeper than the first fix's 16-position slack, within the
            // shipped n-gram stage's n_max=64
            test_ple_spec_rollback(ple, LLAMA_SPEC_CKPT_GPU_FALLBACK, "#214 PLE rollback 40 deep (gpu-fallback)", 40);
            test_ple_spec_rollback(ple, LLAMA_SPEC_CKPT_PER_STEP,     "#214 PLE rollback 40 deep (per-step)", 40);
            test_ple_legacy_blob(ple);
        }
    }

    if (ple) llama_free_model(ple);
    llama_free_model(g_model);
    llama_backend_free();

    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
