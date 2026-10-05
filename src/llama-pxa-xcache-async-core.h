// pxa / PXA expert-cache cold path without host round trips -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-async-core.h -- PXA_XCACHE_ASYNC, the host half: rendezvous slots, per-(layer, width) CPU sub-graphs and the worker
// thread that serves them. The model-facing wrapper is llama-pxa-xcache-async.h; this core takes plain ggml tensors and a slot
// factory so tests/test-pxa-xcache-async.cpp can drive it without a GPU (the "GPU" is another thread that writes the same flags).
#pragma once

#include "ggml.h"
#include "ggml-cuda-xcache.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

constexpr int PXA_XCA_MAX_TOK    = 32;     // widest graph the cold sub-graph table covers (PXA_XCACHE_ASYNC_MAXTOK is clamped to it)
constexpr int PXA_XCA_MAX_LAYERS = 1024;

typedef struct ggml_cuda_cold_slot * (*pxa_xca_slot_new_fn)(int device, int n_embd, int n_used, int n_tok_max);
typedef void (*pxa_xca_slot_free_fn)(struct ggml_cuda_cold_slot * slot);
// device counters of a slot and the SM clock (kHz) of its device, to turn the counter clocks into time
typedef bool (*pxa_xca_slot_stats_fn)(struct ggml_cuda_cold_slot * slot, unsigned long long out[GGML_CUDA_COLD_STAT_N], double * khz);

struct pxa_xca_sub;
struct pxa_xca_layer;

// Worker-side totals (the GPU-side ones live in the slot's device counters).
struct pxa_xca_totals {
    uint64_t requests = 0;       // requests the worker served (each one a layer of a token with at least one cold slot)
    uint64_t compute_us = 0;     // wall time of the CPU sub-graphs
    uint64_t max_us = 0;         // the longest one
    uint64_t errors = 0;         // sub-graph missing / failed: those layers' rows are zero (must stay 0)
    // GPU side, summed over the slots (needs the stats callback)
    uint64_t gpu_waits = 0, gpu_skipped = 0, gpu_timeouts = 0;
    double   gpu_wait_us = 0, gpu_wait_max_us = 0;
    // PXA_XCACHE_ASYNC_CHECK: rows compared against the scheduler split, rows with a differing bit, largest |difference|, layers with a difference
    uint64_t chk_rows = 0, chk_bad = 0; float chk_maxd = 0; int chk_layers_bad = 0; int chk_first_bad_il = -1; std::string chk_bad_layers;
};

struct pxa_xca_core {
    pxa_xca_core(pxa_xca_slot_new_fn nf, pxa_xca_slot_free_fn ff, pxa_xca_slot_stats_fn sf = nullptr);
    ~pxa_xca_core();
    pxa_xca_core(const pxa_xca_core &) = delete;
    pxa_xca_core & operator=(const pxa_xca_core &) = delete;

    // the layer's slot, created on first use (nullptr when the factory fails); starts the worker with the first slot
    struct ggml_cuda_cold_slot * slot(int il, int device, int n_embd, int n_used, int n_tok_max);
    // build (once) and register the CPU sub-graph of this slot for a graph of n_tok tokens. up / gate (or up_gate) / down are the
    // layer's cold stacks as the CPU reads them. False when the stacks cannot take the path (shape, type): the caller keeps the split.
    bool add_width(struct ggml_cuda_cold_slot * slot, int n_tok, struct ggml_tensor * up, struct ggml_tensor * gate,
                   struct ggml_tensor * up_gate, struct ggml_tensor * down, int unary_op, int n_threads);

    // stop and join the worker (idempotent); the summary is only complete after this
    void stop();
    pxa_xca_totals totals(bool with_gpu);
    std::string summary_line();

    // test hook: serve one pending request of every layer on the calling thread (when the worker is not running)
    int serve_pending();

private:
    void run();
    void serve(pxa_xca_layer * L, uint64_t req);
    void start();

    pxa_xca_slot_new_fn   new_fn_;
    pxa_xca_slot_free_fn  free_fn_;
    pxa_xca_slot_stats_fn stats_fn_;
    // sub-graph tensors come from a few shared ggml contexts (ggml has a small fixed pool of contexts: one per sub-graph would exhaust it)
    struct ggml_context * sub_ctx();
    std::vector<struct ggml_context *> arenas_;
    int                   arena_used_ = 0;
    pxa_xca_layer *       layers_[PXA_XCA_MAX_LAYERS] = {};
    std::atomic<int>      n_layers_{0};
    std::atomic<bool>     stop_{false};
    std::thread           th_;
    bool                  started_ = false;
    int                   spin_us_ = 3000;
    int                   cpu_ = -1;
};
