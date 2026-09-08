#pragma once

#include "ggml.h"

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    bool use_cpy_indirection = false;
    std::vector<char *> cpy_dest_ptrs;
    char ** dest_ptrs_d;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;

    // PXA_CUDA_GRAPH_V2 (G1, 2026-07-19): replace the PERMANENT too-many-updates disable with a
    // cooldown -- early churn (prompt boundaries, first-token topology variants) can no longer kill
    // steady-state replay for the rest of the session. While cooldown_remaining > 0 the key runs
    // eager; at 0 it re-arms (consecutive-update counter reset).
    int cooldown_remaining = 0;
    // Pool generation at last property store; a bump (real cudaFree of pool memory) forces recapture
    // so a replay can never dereference baked pool pointers that were returned to the driver.
    uint64_t pool_generation = 0;
    // LRU bookkeeping for the keyed cache (each captured exec holds device memory).
    uint64_t last_use = 0;
    // PXA_CUDA_GRAPH_PREFILL (2026-09-02): ported from mainline llama.cpp 9400c8946e (MIT), which
    // replaced the permanent "too many consecutive updates" disable with a two-call warmup rule:
    // a graph is captured only after the node properties have been IDENTICAL on two consecutive
    // evaluations, and any later property change resets the flag and runs eager again instead of
    // burning a capture. That is what makes capture viable for multi-ubatch prefill, where the
    // early ubatches churn (n_kv grows in 256-cell steps) before the shape settles.
    bool warmup_complete = false;

    // Honest per-key counters (PXA_CUDA_GRAPH_LOG instrumentation).
    long n_captures = 0;
    long n_replays  = 0;
    long n_eager    = 0;
    int  last_n_nodes = 0;
#endif
};

