#pragma once
// PXA_EXPERT_LOG_v1 (fusion-v2 profiling, 2026-07-18)
// Env-gated per-request MoE expert-routing histograms.
//   PXA_EXPERT_LOG=/some/dir  -> for every completed generation request, write one JSON file
//   with the per-layer expert-activation histogram (expert index -> count of (token, top-k slot)
//   selections) accumulated over ALL graph evals of that request (prompt prefill + decode).
//
// Mechanism: a ggml_backend_sched eval-callback observes every tensor named "ffn_moe_topk-<il>"
// (the [n_expert_used, n_tokens] I32 view produced by ggml_top_k in llm_build_moe_ffn) right
// after it is computed, copies the top-k indices to host (stride-aware: the tensor is a VIEW into
// the argsort buffer with nb[1] == n_expert*4; the fused CUDA topk_moe kernel only writes the
// first k entries of each row, so we must honor the view strides and never read the full
// view_src), and accumulates counts. The server hooks request boundaries:
//   launch_slot_with_task -> pxa_expert_log_begin()   (resets the accumulator)
//   send_final_response   -> pxa_expert_log_flush()   (writes <dir>/<completion_id>.json + index.jsonl)
// ONLY VALID AT np1 (single slot): with concurrent slots the histograms would interleave.
// When the env var is unset, nothing is installed and the server is byte-identical to stock.

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <vector>

struct pxa_expert_log_state {
    bool        checked = false;
    std::string dir;

    std::mutex mtx;

    // accumulator (since last begin)
    int      n_expert = 0;   // max n_expert seen (from view nb[1]/4)
    int      top_k    = 0;   // n_expert_used (view ne[0])
    std::vector<std::vector<uint64_t>> counts;           // [layer][expert]
    std::vector<uint64_t>              tokens_per_layer; // [layer]
    std::vector<int32_t>               hostbuf;

    // request meta
    std::atomic<uint64_t> seq{0};
    uint64_t              cur_seq = 0;
    int                   cur_id_task = -1;
    std::string           cur_cmpl_id;
};

static pxa_expert_log_state & pxa_elog() {
    static pxa_expert_log_state s;
    return s;
}

static bool pxa_expert_log_enabled() {
    auto & S = pxa_elog();
    if (!S.checked) {
        const char * d = getenv("PXA_EXPERT_LOG");
        if (d && d[0]) {
            S.dir = d;
            mkdir(S.dir.c_str(), 0755); // best-effort
        }
        S.checked = true;
    }
    return !S.dir.empty();
}

static void pxa_expert_log_reset_locked() {
    auto & S = pxa_elog();
    for (auto & v : S.counts) {
        std::fill(v.begin(), v.end(), 0);
    }
    std::fill(S.tokens_per_layer.begin(), S.tokens_per_layer.end(), 0);
}

// the ggml_backend_sched eval callback
static bool pxa_expert_log_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    }
    // computed + backend synchronized: harvest
    const int il = atoi(t->name + 13);
    if (il < 0 || il > 512) {
        return true;
    }
    const int     k        = (int) t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (k <= 0 || n_tokens <= 0) {
        return true;
    }
    // stride-aware span read (t is a view into the argsort result)
    const size_t nb1  = t->nb[1];
    const size_t span = (size_t)(n_tokens - 1) * nb1 + (size_t) k * sizeof(int32_t);
    const int    row_stride_i32 = (int)(nb1 / sizeof(int32_t));
    // n_expert = full sorted row width (view_src row) when available, else at least max index+1
    int n_exp_here = t->view_src ? (int) t->view_src->ne[0] : row_stride_i32;

    auto & S = pxa_elog();
    std::lock_guard<std::mutex> lock(S.mtx);

    S.hostbuf.resize(span / sizeof(int32_t) + 1);
    ggml_backend_tensor_get(t, S.hostbuf.data(), 0, span);

    if ((int) S.counts.size() <= il) {
        S.counts.resize(il + 1);
        S.tokens_per_layer.resize(il + 1, 0);
    }
    if (n_exp_here > S.n_expert) S.n_expert = n_exp_here;
    if ((int) S.counts[il].size() < S.n_expert) S.counts[il].resize(S.n_expert, 0);
    S.top_k = k;

    for (int64_t ti = 0; ti < n_tokens; ++ti) {
        const int32_t * row = S.hostbuf.data() + ti * row_stride_i32;
        for (int j = 0; j < k; ++j) {
            const int32_t e = row[j];
            if (e >= 0) {
                if (e >= (int) S.counts[il].size()) {
                    S.counts[il].resize(e + 1, 0);
                    if (e + 1 > S.n_expert) S.n_expert = e + 1;
                }
                S.counts[il][e]++;
            }
        }
    }
    S.tokens_per_layer[il] += (uint64_t) n_tokens;
    return true; // continue graph eval
}

static void pxa_expert_log_begin(int id_task, const std::string & cmpl_id) {
    if (!pxa_expert_log_enabled()) return;
    auto & S = pxa_elog();
    std::lock_guard<std::mutex> lock(S.mtx);
    pxa_expert_log_reset_locked();
    S.cur_seq     = S.seq.fetch_add(1) + 1;
    S.cur_id_task = id_task;
    S.cur_cmpl_id = cmpl_id;
}

static std::string pxa_expert_log_sanitize(const std::string & s) {
    std::string out;
    for (char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_') {
            out += c;
        }
    }
    return out;
}

static std::string pxa_expert_log_escape(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += (char) c;
    }
    return out;
}

// returns the path written ("" if disabled / nothing accumulated)
static std::string pxa_expert_log_flush(int id_task, const std::string & cmpl_id,
                                        int n_prompt_tokens, int n_decoded,
                                        const std::string & prompt_preview) {
    if (!pxa_expert_log_enabled()) return "";
    auto & S = pxa_elog();
    std::lock_guard<std::mutex> lock(S.mtx);

    int n_layers = (int) S.counts.size();
    if (n_layers == 0) return "";

    std::string key = pxa_expert_log_sanitize(!cmpl_id.empty() ? cmpl_id : S.cur_cmpl_id);
    char fname[512];
    if (!key.empty()) {
        snprintf(fname, sizeof(fname), "%s/%s.json", S.dir.c_str(), key.c_str());
    } else {
        snprintf(fname, sizeof(fname), "%s/req-%06llu.json", S.dir.c_str(), (unsigned long long) S.cur_seq);
    }

    FILE * f = fopen(fname, "w");
    if (!f) return "";

    fprintf(f, "{\n");
    fprintf(f, "  \"seq\": %llu,\n", (unsigned long long) S.cur_seq);
    fprintf(f, "  \"id_task\": %d,\n", id_task);
    fprintf(f, "  \"completion_id\": \"%s\",\n", pxa_expert_log_escape(cmpl_id).c_str());
    fprintf(f, "  \"ts_unix\": %lld,\n", (long long) time(NULL));
    fprintf(f, "  \"n_layers\": %d,\n", n_layers);
    fprintf(f, "  \"n_experts\": %d,\n", S.n_expert);
    fprintf(f, "  \"top_k\": %d,\n", S.top_k);
    fprintf(f, "  \"n_prompt_tokens\": %d,\n", n_prompt_tokens);
    fprintf(f, "  \"n_decoded\": %d,\n", n_decoded);
    fprintf(f, "  \"prompt_preview\": \"%s\",\n", pxa_expert_log_escape(prompt_preview).c_str());
    fprintf(f, "  \"tokens_per_layer\": [");
    for (int il = 0; il < n_layers; ++il) {
        fprintf(f, "%s%llu", il ? "," : "", (unsigned long long) S.tokens_per_layer[il]);
    }
    fprintf(f, "],\n");
    fprintf(f, "  \"counts\": [\n");
    for (int il = 0; il < n_layers; ++il) {
        fprintf(f, "    [");
        const auto & row = S.counts[il];
        for (int e = 0; e < S.n_expert; ++e) {
            uint64_t v = e < (int) row.size() ? row[e] : 0;
            fprintf(f, "%s%llu", e ? "," : "", (unsigned long long) v);
        }
        fprintf(f, "]%s\n", il + 1 < n_layers ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);

    // append to the index for order-based correlation
    char iname[512];
    snprintf(iname, sizeof(iname), "%s/index.jsonl", S.dir.c_str());
    FILE * fi = fopen(iname, "a");
    if (fi) {
        fprintf(fi, "{\"seq\":%llu,\"id_task\":%d,\"completion_id\":\"%s\",\"file\":\"%s\",\"n_prompt_tokens\":%d,\"n_decoded\":%d}\n",
                (unsigned long long) S.cur_seq, id_task, pxa_expert_log_escape(cmpl_id).c_str(), fname,
                n_prompt_tokens, n_decoded);
        fclose(fi);
    }

    pxa_expert_log_reset_locked();
    return fname;
}
