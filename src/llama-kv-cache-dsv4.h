#pragma once

// DeepSeek-V4 memory module — INTERFACE.
//
// Adapted from llama.cpp `src/llama-kv-cache-dsv4.h` @ upstream commit 82dbc4f01
// (PR #24162 and follow-ups). Copyright (c) 2023-2026 The ggml authors. MIT.
//
// Upstream implements this against llama_memory_i / llama_kv_cache_iswa /
// llama_kv_cells / llama_io_{read,write}_i, none of which exist in this tree
// (our memory layer is the flat POD `struct llama_kv_cache` in llama-context.h).
// Per the port spec §1.4 this module is therefore a REWRITE against our POD
// cache, not a file-level port. `llama_dsv4_comp_plan` below is a verbatim
// structural copy of upstream's `llama_kv_cache_dsv4_context::comp_plan` and
// MUST NOT drift — it is the contract with src/graphs/build_deepseek4.cpp.
//
// ###################################################################
// #  STATUS: INTERFACE ONLY. THE IMPLEMENTATION IS NOT WRITTEN.     #
// #  llama_dsv4_memory_init() DELIBERATELY FAILS so that a          #
// #  DeepSeek-V4 model REFUSES TO LOAD rather than running on an    #
// #  empty cache and emitting fluent garbage. See the .cpp.         #
// ###################################################################

#include "ggml.h"

#include <cstdint>
#include <vector>

struct llama_context;
struct llama_batch;

// which of the three compressed block streams
enum llama_dsv4_stream {
    LLAMA_DSV4_CSA = 0,   // compress ratio 4   (+ lightning indexer)
    LLAMA_DSV4_HCA = 1,   // compress ratio 128
    LLAMA_DSV4_LID = 2,   // lightning-indexer key stream (ratio 4)
};

// verbatim from upstream llama_kv_cache_dsv4_context::comp_plan
struct llama_dsv4_comp_plan {
    std::vector<int32_t> state_pos;                 // APE row ids = pos % ratio
    std::vector<int32_t> state_persist_src_idxs;    // current-ubatch source rows
    std::vector<int32_t> state_persist_dst_idxs;    // unique persistent-state dest rows
    std::vector<int32_t> state_read_idxs;           // rows into [persistent_state | ubatch_scratch]
    std::vector<int64_t> state_write_idxs;          // dest rows in the compressed cache
    std::vector<int32_t> state_write_pos;           // RoPE positions for state-backed commits
    std::vector<int32_t> n_visible;                 // completed compressed rows visible per query token
    int64_t n_stream = 1;
    int64_t n_kv     = 0;                           // graph width >= max(n_visible)
};

struct llama_dsv4_comp_inputs {
    ggml_tensor * state_pos              = nullptr; // I32 [n_state]
    ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_persist_src]
    ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_persist_dst]
    ggml_tensor * state_read_idxs        = nullptr; // I32 [ratio*n_write] (overlap: 2*ratio*n_write)
    ggml_tensor * state_write_idxs       = nullptr; // I64 [n_write]
    ggml_tensor * state_write_pos        = nullptr; // I32 [n_write]
    ggml_tensor * kq_mask                = nullptr; // F32 [n_kv, n_tokens, 1, 1]
    ggml_tensor * k_rot                  = nullptr; // F32 [nrot, nrot] or null
};

struct llama_dsv4_inputs {
    ggml_tensor * raw_k_idxs  = nullptr;            // I64 [n_tokens]
    ggml_tensor * raw_kq_mask = nullptr;            // F32 [n_kv_raw, n_tokens, 1, 1]
    ggml_tensor * raw_win_idxs = nullptr;           // I32 [n_swa+n_tokens] (window mode only)
    ggml_tensor * raw_k_rot   = nullptr;            // F32 [nrot, nrot] or null
    llama_dsv4_comp_inputs comp[3];                 // indexed by llama_dsv4_stream
};

//
// lifecycle hooks — called from src/llama.cpp (chunk B)
//

bool llama_dsv4_memory_init(llama_context & lctx, ggml_type type_k, ggml_type type_v,
                            uint32_t kv_size, bool offload);
// Derive the three compression plans for this batch. MUST run before the graph for the
// same batch is built: build_dsv4_inputs() sizes every plan-derived graph input from the
// stored plans, and llama_dsv4_set_inputs() then fills those tensors from the same plans.
// Building the graph against a stale plan and filling from a fresh one is a size mismatch.
void llama_dsv4_build_plans(llama_context & lctx, const llama_batch & batch);
void llama_dsv4_set_inputs (llama_context & lctx, const llama_batch & batch);
void llama_dsv4_memory_free(llama_context & lctx);

//
// graph-side accessors — called from src/graphs/build_deepseek4.cpp (chunk D)
//

llama_dsv4_inputs &          llama_dsv4_get_inputs   (llama_context & lctx);
const llama_dsv4_comp_plan & llama_dsv4_get_plan     (const llama_context & lctx, llama_dsv4_stream s);
uint32_t                     llama_dsv4_get_raw_n_kv (const llama_context & lctx);
uint32_t                     llama_dsv4_get_raw_swa  (const llama_context & lctx);
// Raw-window lever: raw attention spans [prior-window gather | in-batch keys]
// = n_swa + n_tokens rows instead of the whole padded ring. Same visible key set,
// far fewer rows (prefill 640 vs 768 at ub=512; decode 192 vs 768).
// DEFAULT: ON for non-flash-attn contexts (proven byte-identical there), OFF under
// flash-attn (the fallback FA kernel rejects the layout — checkpoint 12).
// PXA_DSV4_RAW_WINDOW_KV=0|1|2 overrides unconditionally.
bool                         llama_dsv4_raw_window_enabled(const llama_context & lctx);
// 0 = ring, 1 = window lever, 2 = identity-gather bisect (ring maths, window plumbing)
int                          llama_dsv4_raw_window_mode(const llama_context & lctx);
// window K width, padded so the attention kernel never sees a partial trailing tile
uint32_t                     llama_dsv4_raw_window_width(const llama_context & lctx, int64_t n_tokens);
int64_t                      llama_dsv4_get_raw_nrot (const llama_context & lctx);  // 0 => no raw k_rot
int64_t                      llama_dsv4_get_comp_nrot(const llama_context & lctx, llama_dsv4_stream s);

ggml_tensor * llama_dsv4_get_raw_k(const llama_context & lctx, ggml_context * ctx, int32_t il);
// `ring_w` MUST be the tensor llama_dsv4_cpy_raw_k() returned for this layer, not the
// ring leaf: the gather has to depend on the scatter or the scheduler may run it first.
ggml_tensor * llama_dsv4_get_raw_win_k(const llama_context & lctx, ggml_context * ctx,
                                       ggml_tensor * ring_w, ggml_tensor * win_idxs, int32_t il);
ggml_tensor * llama_dsv4_cpy_raw_k(const llama_context & lctx, ggml_context * ctx,
                                   ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il);
ggml_tensor * llama_dsv4_get_comp_k(const llama_context & lctx, ggml_context * ctx,
                                    llama_dsv4_stream s, int32_t il);
ggml_tensor * llama_dsv4_cpy_comp_k(const llama_context & lctx, ggml_context * ctx,
                                    llama_dsv4_stream s, ggml_tensor * k_cur,
                                    ggml_tensor * k_idxs, int32_t il);
ggml_tensor * llama_dsv4_get_state_kv(const llama_context & lctx, ggml_context * ctx,
                                      llama_dsv4_stream s, int32_t il);
ggml_tensor * llama_dsv4_get_state_score(const llama_context & lctx, ggml_context * ctx,
                                         llama_dsv4_stream s, int32_t il);
ggml_tensor * llama_dsv4_cpy_state_kv(const llama_context & lctx, ggml_context * ctx,
                                      llama_dsv4_stream s, ggml_tensor * cur,
                                      ggml_tensor * idxs, int32_t il);
ggml_tensor * llama_dsv4_cpy_state_score(const llama_context & lctx, ggml_context * ctx,
                                         llama_dsv4_stream s, ggml_tensor * cur,
                                         ggml_tensor * idxs, int32_t il);

//
// speculative rollback (M5) — see the block comment in the .cpp for WHY the compressor
// ring cannot self-heal. Components are llama_dsv4_spec_component in llama.h.
//
bool     llama_dsv4_spec_snapshot      (llama_context & lctx, const llama_batch & batch);
bool     llama_dsv4_spec_restore       (llama_context & lctx, uint32_t components);
void     llama_dsv4_spec_discard       (llama_context & lctx);
size_t   llama_dsv4_spec_snapshot_bytes(const llama_context & lctx);
uint64_t llama_dsv4_debug_digest       (const llama_context & lctx, int component);
size_t   llama_dsv4_debug_export       (const llama_context & lctx, int component,
                                        float * out, size_t max_n);
