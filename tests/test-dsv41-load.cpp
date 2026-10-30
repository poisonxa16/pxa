// DeepSeek-V4.1 ("deepseek41") loader check.
//
//   test-dsv41-load MODEL.gguf [ok|fail[:substring]]
//
// ok   : the model must load (CPU, mmap, no context) and every per-layer tensor the
//        validated hparams call for must be bound, every tensor V4.1 lacks must be null.
// fail : loading must fail (and the error log must contain `substring` if one is given).
//
// The loader itself already enforces "every tensor in the file is bound" (n_created ==
// n_tensors), so a stray tensor (hc head, tid2eid, a q-head norm) fails the load; this
// test adds the opposite direction: the layout the graph will rely on.
#include "llama.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        g_log += text;
    }
    if (getenv("DSV41_TEST_VERBOSE")) {
        fputs(text, stderr);
    }
}

static int n_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { n_fail++; fprintf(stderr, "CHECK FAILED: " __VA_ARGS__); fputc('\n', stderr); } } while (0)

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [ok|fail[:substring]]\n", argv[0]);
        return 2;
    }
    const std::string expect = argc > 2 ? argv[2] : "ok";
    const bool want_ok = expect == "ok";
    const std::string want_sub = (!want_ok && expect.size() > 5) ? expect.substr(5) : "";

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = true;
    mp.use_mlock    = false;
    // DSV41_DRY=1: bind and shape-check every tensor without reading weight data
    // (no MAP_POPULATE of a 138 GB set); used for the real-file bind check.
    if (const char * e = getenv("DSV41_DRY")) { mp.dry_run = atoi(e) != 0; }

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!want_ok) {
        const bool sub_ok = want_sub.empty() || g_log.find(want_sub) != std::string::npos;
        if (model) {
            fprintf(stderr, "FAIL: %s loaded but was expected to be rejected\n", argv[1]);
            llama_free_model(model);
            return 1;
        }
        if (!sub_ok) {
            fprintf(stderr, "FAIL: rejected, but the error does not mention '%s'. Log:\n%s", want_sub.c_str(), g_log.c_str());
            return 1;
        }
        printf("PASS rejected %s%s%s\n", argv[1], want_sub.empty() ? "" : " with ", want_sub.c_str());
        return 0;
    }
    if (!model) {
        fprintf(stderr, "FAIL: %s did not load. Log:\n%s", argv[1], g_log.c_str());
        return 1;
    }

    const auto & hp = model->hparams;
    CHECK(model->arch == LLM_ARCH_DEEPSEEK41, "arch is not deepseek41");
    CHECK(model->tok_embd && model->output && model->output_norm, "embd/output missing");
    CHECK(!model->hc_head_fn && !model->hc_head_base && !model->hc_head_scale, "hc head must be absent");
    CHECK(hp.dsv4_hash_layer_count == 0, "hash layers");
    CHECK(hp.expert_gating_func == LLM_EXPERT_GATING_FUNC_SQRT_SOFTPLUS, "gating func");
    CHECK(hp.dsv41_n_kv_source > 0 && hp.dsv41_n_index_source > 0, "source lists");
    CHECK(hp.dsv41_is_kv_source[hp.dsv41_cand_source_layer], "candidate source is not a kv source");
    if (hp.dsv41_engram_present) {
        CHECK(model->engram_token_map && model->engram_primes && model->engram_offsets && model->engram_multipliers, "engram tables");
    } else {
        CHECK(!model->engram_token_map, "engram tables bound without engram.present");
    }

    int n_csa2 = 0, n_dec = 0, n_raw = 0, n_gate = 0, n_idx = 0, n_eng = 0;
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const auto & l = model->layers[il];
        const uint32_t r = hp.dsv4_compress_ratios[il];
        (r == 0 ? n_raw : r == 1 ? n_dec : n_csa2)++;
        CHECK(l.attn_norm && l.attn_sinks && l.wq_a && l.attn_q_a_norm && l.wq_b && l.wkv && l.attn_kv_a_norm && l.wo_a && l.wo_b,
              "layer %u attention core", il);
        CHECK(l.hc_attn_fn && l.hc_attn_base && l.hc_attn_scale && l.hc_ffn_fn && l.hc_ffn_base && l.hc_ffn_scale, "layer %u hc", il);
        CHECK(l.ffn_gate_inp && l.ffn_exp_probs_b && l.ffn_norm && l.ffn_gate_exps && l.ffn_up_exps && l.ffn_down_exps,
              "layer %u moe", il);
        CHECK(l.ffn_gate_shexp && l.ffn_up_shexp && l.ffn_down_shexp, "layer %u shexp", il);
        CHECK(!l.ffn_gate_tid2eid && !l.attn_comp_ape && !l.indexer_comp_wkv && !l.indexer_comp_wgate && !l.indexer_comp_ape &&
              !l.indexer_comp_norm && !l.attn_q_norm && !l.attn_k_norm, "layer %u carries a tensor V4.1 does not have", il);
        CHECK(l.wo_a && l.wo_a->ne[2] == (int64_t) hp.dsv4_o_group_count, "layer %u wo_a is not [.., o_lora, o_groups]", il);
        const bool kv = hp.dsv41_is_kv_source[il], ix = hp.dsv41_is_index_source[il], en = hp.dsv41_is_engram[il];
        CHECK(!!l.attn_comp_wkv == kv && !!l.attn_comp_norm == kv && !!l.indexer_attn_k == kv && !!l.indexer_k_norm == kv,
              "layer %u kv-source tensors do not match the flag (%d)", il, kv);
        CHECK(!!l.attn_comp_wgate == (kv && r > 1), "layer %u compressor gate (ratio %u, kv %d)", il, r, kv);
        CHECK(!!l.indexer_proj == ix && !!l.indexer_attn_q_b == ix, "layer %u index-source tensors (flag %d)", il, ix);
        CHECK(!!l.engram_q == en && !!l.engram_k == en && !!l.engram_wkv == en, "layer %u engram tensors (flag %d)", il, en);
        CHECK(!l.engram_embd, "layer %u engram_embd bound without the sidecar", il);
        n_gate += !!l.attn_comp_wgate;
        n_idx  += ix;
        n_eng  += en;
    }

    printf("deepseek41: n_layer %u, n_tensors %zu, n_expert %u x top-%u, ratios raw %d / csa2 %d / dec %d, "
           "kv-src %u (gated %d), idx-src %d, engram layers %d, n_params %.3fB\n",
           hp.n_layer, model->tensors_by_name.size(), hp.n_expert, hp.n_expert_used, n_raw, n_csa2, n_dec,
           hp.dsv41_n_kv_source, n_gate, n_idx, n_eng, llama_model_n_params(model) / 1e9);

    llama_free_model(model);
    llama_backend_free();
    if (n_fail) {
        fprintf(stderr, "FAIL: %d checks\n", n_fail);
        return 1;
    }
    printf("PASS %s\n", argv[1]);
    return 0;
}
