// pxa / PXA expert-cache cold path without host round trips -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-async.cpp -- model-facing side of PXA_XCACHE_ASYNC; the mechanism is llama-pxa-xcache-async-core.cpp.
#include "llama-pxa-xcache-async.h"
#include "llama-pxa-xcache-async-core.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-xcache.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

bool llama_pxa_xcache_async_wanted(void) {
#ifdef GGML_USE_CUDA
    static const bool on = [] { const char * e = getenv("PXA_XCACHE_ASYNC"); return e && *e && atoi(e) != 0; }();
    return on;
#else
    return false;
#endif
}

int llama_pxa_xcache_async_max_tokens(void) {
    static const int n = [] {
        const char * e = getenv("PXA_XCACHE_ASYNC_MAXTOK");
        return std::min(PXA_XCA_MAX_TOK, std::max(1, e && *e ? atoi(e) : 8));
    }();
    return n;
}

ggml_tensor * llama_pxa_xcache_async_check_node(struct ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, ggml_tensor * ids_cold, void * slot) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && ggml_are_same_shape(a, b) && slot != nullptr);
    ggml_tensor * r = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    r->op     = GGML_OP_MOE_SPLIT_IDS;
    r->src[0] = ids_cold;
    r->src[1] = a;
    r->src[2] = b;
    ((int32_t *) r->op_params)[0] = 4;
    memcpy((char *) r->op_params + 2*sizeof(int32_t), &slot, sizeof(slot));
    return r;
}

#ifndef GGML_USE_CUDA

void * llama_pxa_xcache_async_slot(const llama_model &, int, int, int, int, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int, int, bool) { return nullptr; }
void llama_pxa_xcache_async_free(llama_model &) {}

#else

namespace {

bool stats_cb(struct ggml_cuda_cold_slot * s, unsigned long long out[GGML_CUDA_COLD_STAT_N], double * khz) {
    if (khz) *khz = (double) ggml_backend_cuda_cold_clock_khz(s->device);
    return ggml_backend_cuda_cold_slot_stats(s, out);
}

std::mutex g_mu;
// how often graphs asked for the async path, by width, and how often it declined (summary at exit)
uint64_t g_asked[PXA_XCA_MAX_TOK + 2] = {}, g_declined[PXA_XCA_MAX_TOK + 2] = {};
// why: 0 not a CPU-readable stack, 1 no device, 2 no slot, 3 slot shape, 4 sub-graph refused, 5 chain not built through ggml_moe_up_gate
uint64_t g_why[8] = {};
std::string g_why_buft;

}

void * llama_pxa_xcache_async_slot(const llama_model & model, int il, int n_tokens, int n_embd, int n_used,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * up_gate, ggml_tensor * down, int unary_op, int n_threads, bool fused_chain) {
    if (n_tokens < 1 || n_tokens > llama_pxa_xcache_async_max_tokens() || !down) return nullptr;
    // the warm-up decode routes every token to ALL experts (n_used = n_expert): that graph is no decode shape and keeps the split path
    if (n_used != (int) model.hparams.n_expert_used) return nullptr;
    static const bool allow_mixed = [] { const char * e = getenv("PXA_XCACHE_ASYNC_MIXED"); return e && *e && atoi(e) != 0; }();
    struct counter_t {
        int w; bool ok = false;
        counter_t(int w_) : w(w_) { std::lock_guard<std::mutex> l(g_mu); ++g_asked[w]; }
        ~counter_t() { if (!ok) { std::lock_guard<std::mutex> l(g_mu); ++g_declined[w]; } }
    } counted(n_tokens);
    if (!fused_chain && !allow_mixed) { std::lock_guard<std::mutex> l(g_mu); ++g_why[5]; return nullptr; }
    // the CPU must be able to read all of the cold stacks straight from the pinned bytes: the aliases of a narrow graph. A stack
    // still in its CUDA host buffer (zero-copy cold path) is read by the GPU in place and has nothing to hand to a worker.
    for (ggml_tensor * t : { up, gate, up_gate, down }) {
        if (t && (!t->buffer || ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cpu_buffer_type())) {
            std::lock_guard<std::mutex> l(g_mu);
            ++g_why[0];
            if (g_why_buft.empty()) g_why_buft = std::string(t->name) + (t->buffer ? std::string(" in ") + ggml_backend_buffer_name(t->buffer) : " (no buffer)");
            return nullptr;
        }
    }
    int dev = -1;
    for (const auto & f : model.pxa_xc_fills) if (f.il == il) { dev = f.dev; break; }
    if (dev < 0) { std::lock_guard<std::mutex> l(g_mu); ++g_why[1]; return nullptr; }
    if (const char * e = getenv("PXA_XCACHE_ASYNC_THREADS")) if (atoi(e) > 0) n_threads = atoi(e);

    std::lock_guard<std::mutex> lock(g_mu);
    if (!model.pxa_xc_async) {
        model.pxa_xc_async = new pxa_xca_core(ggml_backend_cuda_cold_slot_new, ggml_backend_cuda_cold_slot_free, stats_cb);
    }
    pxa_xca_core * core = (pxa_xca_core *) model.pxa_xc_async;
    // one slot per layer, sized for the widest graph the lever takes
    struct ggml_cuda_cold_slot * slot = core->slot(il, dev, n_embd, n_used, llama_pxa_xcache_async_max_tokens());
    if (!slot) { ++g_why[2]; return nullptr; }
    if (slot->n_embd != n_embd || slot->n_used != n_used) {
        ++g_why[3];
        if (g_why_buft.empty()) { char b[160]; snprintf(b, sizeof(b), "layer %d width %d: slot %dx%d, asked %dx%d", il, n_tokens, slot->n_embd, slot->n_used, n_embd, n_used); g_why_buft = b; }
        return nullptr;
    }
    if (!core->add_width(slot, n_tokens, up, gate, up_gate, down, unary_op, n_threads)) { ++g_why[4]; return nullptr; }
    static bool said = false;
    if (!said) {
        said = true;
        LLAMA_LOG_INFO("PXA_XCACHE_ASYNC: cold experts of split layers computed by a host worker thread (%d CPU threads, widths 1..%d), "
                       "no scheduler split and no host sync per cold layer; CUDA-graph capture is off for these graphs\n",
                       n_threads, llama_pxa_xcache_async_max_tokens());
    }
    counted.ok = true;
    return slot;
}

void llama_pxa_xcache_async_free(llama_model & model) {
    pxa_xca_core * core = (pxa_xca_core *) model.pxa_xc_async;
    if (!core) return;
    model.pxa_xc_async = nullptr;
    const std::string line = core->summary_line();     // stops the worker first
    if (core->totals(false).requests > 0 || getenv("PXA_XCACHE_SUMMARY")) {
        LLAMA_LOG_INFO("%s\n", line.c_str());
        std::string by_w;
        for (int w = 1; w <= PXA_XCA_MAX_TOK; ++w) {
            if (!g_asked[w]) continue;
            char b[64];
            snprintf(b, sizeof(b), " %d:%llu/%llu", w, (unsigned long long) (g_asked[w] - g_declined[w]), (unsigned long long) g_asked[w]);
            by_w += b;
        }
        LLAMA_LOG_INFO("PXA_XCACHE_ASYNC: graph layers built on the async path / asked, by width:%s; declined: stack not CPU-readable %llu (%s), no device %llu, "
                       "no slot %llu, slot shape %llu, sub-graph refused %llu, split-path chain (activation on the GPU there) %llu\n", by_w.c_str(), (unsigned long long) g_why[0], g_why_buft.c_str(),
                       (unsigned long long) g_why[1], (unsigned long long) g_why[2], (unsigned long long) g_why[3], (unsigned long long) g_why[4], (unsigned long long) g_why[5]);
    }
    delete core;
}

#endif
