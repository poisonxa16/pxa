#pragma once

// PXA_HEAD_DUAL_LAYER fit guard. Pure, header-only, no ggml.
//
// The dual-layer head cut copies rows [0, N/2) of the LM head to the first card of a '-sm layer' pair AFTER the weights
// are loaded (src/llama.cpp, llm_build_head_dual_layer). No layer placement or --fit plan counts that copy (~258 MiB on
// the 35B-A3B), so a placement that fills the first card to the brim used to arm the cut and then fail at context
// creation (the GEMM-pool pre-grow or its free-memory floor) where PXA_HEAD_DUAL_LAYER=0 boots.
//
// The guard: arm the cut only when the first card's free memory, measured after the weights are on it, covers
//     the copy + the KV cache of the layers on that card + the compute-buffer estimate + the floor that the
//     context keeps free (PXA_POOL_FLOOR_MB) + PXA_HEAD_DUAL_SLACK_MB
// The slack covers what the loader's estimate does not see: the real compute buffer is larger than the per-layer
// estimate, plus the per-step verify buffers, the GEMM temporary pool and the checkpoint shadows. Measured on the
// 35B-A3B P100 pair at -c 8192: 378 MiB on the first card past the weights, of which the estimate sees 141 MiB.
// Not armed = the head stays whole = exactly PXA_HEAD_DUAL_LAYER=0, so the guard never changes an output either way
// (the cut is bit-identical to the whole head).

#include <cstddef>

#define PXA_HEAD_DUAL_SLACK_MB 512

// bytes the first card must still have free after the weights for the cut to be armed
static inline size_t pxa_head_dual_need_bytes(size_t copy_bytes, size_t kv_first_bytes, size_t compute_est_bytes,
                                              size_t floor_bytes) {
    return copy_bytes + kv_first_bytes + compute_est_bytes + floor_bytes + ((size_t) PXA_HEAD_DUAL_SLACK_MB << 20);
}

static inline bool pxa_head_dual_fits(size_t free_now_bytes, size_t copy_bytes, size_t kv_first_bytes,
                                      size_t compute_est_bytes, size_t floor_bytes) {
    return free_now_bytes >= pxa_head_dual_need_bytes(copy_bytes, kv_first_bytes, compute_est_bytes, floor_bytes);
}
