// PXA_KV_K_GUARD (4.0.4): the K-cache guard for the Qwen2 family.
//
// Qwen2-family attention (qwen2, qwen2vl = Qwen2-VL / Qwen2.5-VL, qwen2moe) has a K projection bias with a few
// very large channels that a 4-bit K cache cannot hold. With -ctk q4_0 such a model answers garbage text and a
// one-token image reply, on the P100 flash-attention path AND on the CPU, so it is the model's numbers, not a
// kernel: K q4_0 + V f16 = garbage, K f16 + V q4_0 = fine, q8_0/q8_0 = fine.
//
// For those archs a requested K cache of 4 bits or fewer is raised to q8_0 when the context is created (and in
// the load-time fit, so the estimate sizes the cache that is really allocated). V stays as asked. Every other
// arch keeps the K type it asked for. PXA_KV_K_GUARD=0 restores the old behaviour.
//
// Header-only, so tests/test-pxa-kv-k-guard.cpp checks the code the engine runs.
#pragma once

#include "llama-arch.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// the archs whose K cache the guard raises
static inline bool llama_pxa_kv_k_guard_arch(llm_arch arch) {
    return arch == LLM_ARCH_QWEN2 || arch == LLM_ARCH_QWEN2VL || arch == LLM_ARCH_QWEN2MOE;
}

// a quantized type of 4 bits or fewer per value: at most 5.0 bits per weight with its block scales
// (q4_0 4.5, q4_1 5.0, iq4_nl 4.5, and every 2- and 3-bit type). The 5-bit types start at 5.5 (q5_0) and stay.
static inline bool llama_pxa_kv_k_guard_low_bits(ggml_type type) {
    if ((int) type < 0 || (int) type >= (int) GGML_TYPE_COUNT || !ggml_is_quantized(type)) {
        return false;
    }
    const int64_t blck = ggml_blck_size(type);
    const size_t  size = ggml_type_size(type);
    return blck > 0 && size > 0 && 8.0*(double) size <= 5.0*(double) blck;
}

// the K type a context of `arch` gets when `type_k` is asked (no environment read here)
static inline ggml_type llama_pxa_kv_k_guard_pick(llm_arch arch, ggml_type type_k, bool enabled) {
    return enabled && llama_pxa_kv_k_guard_arch(arch) && llama_pxa_kv_k_guard_low_bits(type_k) ? GGML_TYPE_Q8_0 : type_k;
}

// PXA_KV_K_GUARD: on unless it is set to 0. Read on every call (one load and one context init read it).
static inline bool llama_pxa_kv_k_guard_enabled() {
    const char * e = getenv("PXA_KV_K_GUARD");
    return e == nullptr || atoi(e) != 0;
}

// ---- the V q8_0 hint (#62282) ------------------------------------------------------------------------------
// The guard leaves V as asked. Measured on a P100 (#62276, the supporter's file, tg128 at depth 16000): K q8_0 + V q4_0
// 41.53 t/s, K q8_0 + V q8_0 56.54 (+36%): a matched pair reaches the faster attention kernels. The guard does not raise V
// on its own; its line tells the user what -ctv q8_0 would cost.

// a V type that takes fewer bytes per value than q8_0 (8.5 bits per weight with its block scale)
static inline bool llama_pxa_kv_v_narrower_than_q8(ggml_type type) {
    if ((int) type < 0 || (int) type >= (int) GGML_TYPE_COUNT) {
        return false;
    }
    const int64_t blck = ggml_blck_size(type);
    const size_t  size = ggml_type_size(type);
    return blck > 0 && size > 0 &&
           (double) size*(double) ggml_blck_size(GGML_TYPE_Q8_0) < (double) ggml_type_size(GGML_TYPE_Q8_0)*(double) blck;
}

// the exact extra V-cache bytes if the V cache were q8_0 instead of type_v: summed over the layers that hold a V cache of
// type_v (n_embd_v_gqa(il) values per cell; a width of 0 = no V cache in that layer), the q8_0 row minus the type_v row,
// times kv_cells. 0 when type_v is q8_0 or wider.
static inline uint64_t llama_pxa_kv_v_q8_extra_bytes(const int64_t * n_embd_v, int n_layers, int64_t kv_cells, ggml_type type_v) {
    if (!llama_pxa_kv_v_narrower_than_q8(type_v) || n_embd_v == nullptr || n_layers <= 0 || kv_cells <= 0) {
        return 0;
    }
    uint64_t extra = 0;
    for (int il = 0; il < n_layers; ++il) {
        const int64_t n = n_embd_v[il];
        if (n <= 0 || n % ggml_blck_size(type_v) != 0 || n % ggml_blck_size(GGML_TYPE_Q8_0) != 0) {
            continue;
        }
        extra += (uint64_t) (ggml_row_size(GGML_TYPE_Q8_0, n) - ggml_row_size(type_v, n))*(uint64_t) kv_cells;
    }
    return extra;
}

// the hint the guard's line carries, "" when V is q8_0 or wider: whole MiB from 10 MiB up, one decimal below
static inline std::string llama_pxa_kv_k_guard_v_hint(ggml_type type_v, uint64_t extra_bytes) {
    if (!llama_pxa_kv_v_narrower_than_q8(type_v) || extra_bytes == 0) {
        return "";
    }
    const double mib = (double) extra_bytes/1048576.0;
    char buf[128];
    snprintf(buf, sizeof(buf), "for faster decode also set V to q8_0 (-ctv q8_0): +%.*f MiB at this context",
             mib >= 10.0 ? 0 : 1, mib);
    return buf;
}
