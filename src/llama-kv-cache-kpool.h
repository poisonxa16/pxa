#pragma once

// PXA_GLM5NEXT: the pooled ("k-pool") indexer cache of GLM-5.3-Flash's DeepSeek sparse
// attention — INTERFACE.
//
// Ported from llama.cpp PR #27773 (`src/llama-memory-hybrid-idx.cpp`,
// `llama_memory_hybrid_idx_context::kpool_build_layout / kpool_build_state /
// set_input_kpool`). Copyright (c) 2023-2026 The ggml authors. MIT.
//
// WHAT THIS IS
// ------------
// GLM-5.3-Flash's indexer does not score tokens, it scores POOLS of `kpool` (= 4)
// CONSECUTIVE tokens of one sequence. So the indexer's own cache is kpool times shorter
// than the KV cache, and the host has to hand the graph, every ubatch:
//
//   pool_cells    [n_pool]              the cell whose row holds each pool's pooled key
//   pool_idxs     [kpool, n_pool]       the member cells of each pool (sentinel for padding)
//   pool_mask     [n_pool, n_tokens]    0 where a pool is visible to a token, -inf elsewhere
//   tail_idxs     [kpool-1, n_tokens]   the incomplete tail, appended to the selection
//   gather_mask   [n_sel, n_tokens]     which slots of the decode gather are real
//   new_pool_idxs [kpool, n_new_g]      the members of the pools THIS ubatch completes,
//                                       padded to a fixed row count (see n_new_g)
//   new_pool_rep  [n_new_g]             the row each of those pooled keys is written into
//
// plus a `cache_safe` flag saying whether the pooled write-back may be done before the
// pool gather in the same graph.
//
// Everything above is derived from POSITIONS, not from cell order: the pool grid starts at
// each sequence's first cached position and a pool exists only where `kpool` cells carry
// consecutive positions. That is the whole off-by-one surface of this architecture, so the
// derivation lives here as PURE functions over a plain description of the cache's occupied
// cells, and `tests/test-kpool-cache.cpp` drives them with synthetic token streams — no
// model, no GPU, no ggml.
//
// DIFFERENCES FROM UPSTREAM, ALL DELIBERATE
// -----------------------------------------
//  * Single stream. Upstream's cache can be non-unified (one stream per sequence) and folds
//    a stream offset into every cell index. Our tree's cache is the flat POD ring in
//    llama-context.h: there is exactly one cell index space. Upstream's `gcell()` is
//    therefore the identity here and is folded out, not carried as dead generality.
//  * The mask is F32. Upstream can emit F16 for its fused lightning-indexer kernel; we have
//    no fused kernel for this yet and the mask is consumed by a plain ggml_add onto an F32
//    score.
//  * The storage is not a separate memory object. The indexer rows live in
//    `llama_kv_cache::idx_l[il]`, a per-layer F32 side buffer over the SAME cell indices as
//    k_l, so a cell is one token in both by construction.

#include "llama.h"

#include <cstdint>
#include <utility>
#include <vector>

struct llama_context;
struct llama_batch;
struct ggml_tensor;
struct ggml_context;

//
// ---------------------------------------------------------------------------------------
// the pure core: the pool grid
// ---------------------------------------------------------------------------------------
//

// One occupied cell of the attention cache, as the pool grid sees it.
struct llama_kpool_cell_desc {
    llama_pos                 pos  = 0;
    uint32_t                  cell = 0;
    std::vector<llama_seq_id> seqs;
};

// One token of the ubatch.
struct llama_kpool_tok_desc {
    llama_pos                 pos = 0;
    std::vector<llama_seq_id> seqs;   // seqs[0] is the token's "own" sequence
};

struct llama_kpool_seq {
    llama_pos pos_min = 0;

    // (position, cell) of every cached token of this sequence, sorted by position
    std::vector<std::pair<llama_pos, uint32_t>> cells;

    // index into `cells` of the FIRST member of each complete pool, in position order
    std::vector<uint32_t> pools;

    // 1 where this ubatch (re)wrote a member of the pool, so its pooled key must be recomputed
    std::vector<uint8_t> is_new;

    // PXA_QSA: where the pool scan stopped, i.e. the first index of `cells` it did not fully
    // consider (the loop needs `kpool` cells in hand). Appending cells that are all NEWER than
    // everything already here cannot change any decision the scan made below this cursor, so
    // resuming from it reproduces the from-scratch grid exactly -- which is the whole basis of
    // llama_kpool_append_layout(). Written by llama_kpool_build_layout().
    uint32_t scan_j = 0;
};

struct llama_kpool_state {
    std::vector<llama_kpool_seq> seqs;      // indexed by seq_id, sized n_seq_max

    uint32_t n_pool_real = 0;               // complete pools across all sequences
    uint32_t n_new       = 0;               // of those, the ones this ubatch must (re)pool

    // false when a cell is shared by more than one sequence: the pool grid is
    // sequence-relative, so one cached pooled key cannot serve both. The graph then pools
    // in-place instead of writing back and gathering.
    bool cache_safe = true;
};

// The padded pool count handed to the graph. The LAST padded pool is always unused, which is
// what lets `pool_idxs` use a sentinel row without colliding with a real pool.
uint32_t llama_kpool_pad(uint32_t n_pool_real);

// Pools per token in a selection, and the total selected cell count.
uint32_t llama_kpool_n_top(uint32_t n_pool, uint32_t indexer_top_k, uint32_t kpool);
uint32_t llama_kpool_n_sel(uint32_t n_top, uint32_t kpool, bool select_tail);

// The FIXED number of new-pool rows a graph for this ubatch width is built with. An ubatch of
// n_tokens can complete at most n_tokens/kpool + 1 pools, and the grid always keeps one padded
// pool spare, so this is both an upper bound for the ordinary case and >= 1 for every ubatch.
// PXA_QSA_GRID_VERIFY's comparison as a pure predicate: returns nullptr when the two states
// describe the SAME GRID, else the name of the first grid field that differs (and, via bad_seq,
// the sequence it differed in, or -1 for a whole-state field). The per-ubatch delta fields
// (n_new, is_new) are deliberately NOT part of the answer -- see the block comment on the
// implementation. Exposed so tests/test-qsa-grid-verify.cpp can assert on it without aborting.
const char * llama_kpool_grid_diff(const struct llama_kpool_state & a,
                                   const struct llama_kpool_state & b, int * bad_seq);

uint32_t llama_kpool_n_new_floor(uint32_t n_pool, uint32_t n_tokens, uint32_t kpool);

// Derive the pool grid from the cache's occupied cells. `cells` need not be sorted.
llama_kpool_state llama_kpool_build_layout(
        const std::vector<llama_kpool_cell_desc> & cells,
        uint32_t kpool,
        uint32_t n_seq_max);

// Mark the pools this ubatch completes or rewrites. `all_new` re-pools everything, which is
// what a sequence edit (which regrids the pools) and a !cache_safe layout both require.
void llama_kpool_mark_new(
        llama_kpool_state & st,
        const std::vector<llama_kpool_tok_desc> & toks,
        uint32_t kpool,
        bool all_new);

// PXA_QSA: the APPEND-ONLY grid update.
//
// llama_kpool_build_layout() is O(cached cells) and the engine ran it, from a fresh O(cache
// capacity) scan, on EVERY ubatch -- i.e. on every decoded token. Measured on this box at the
// seat's -c 150016: 4.5 ms per token at 86k fill and 7.2 ms at 140k, single-threaded host time
// that no device profiler can see, against a measured 11.6 ms/token gap between the QSA gather
// arm and dense at 86k.
//
// At decode the cache only ever GROWS, by the ubatch's own cells, and every appended position
// is newer than everything cached for that sequence. Under that precondition the grid is a
// pure append: pos_min does not move, no already-formed pool changes, and the pool scan resumes
// at the cursor it stopped on. So the from-scratch layout and the appended one are IDENTICAL,
// which tests/test-qsa-select.cpp asserts over long synthetic token streams rather than
// asserting it here in prose.
//
// Returns false -- leaving `st` untouched -- when the precondition does not hold (a cell shared
// between sequences, an out-of-order or duplicate position, an unknown sequence). The caller
// must then rebuild from scratch, which is always correct and is what every non-decode path
// still does.
bool llama_kpool_append_layout(
        llama_kpool_state & st,
        const std::vector<llama_kpool_cell_desc> & added,
        uint32_t kpool);

// The append-only counterpart of llama_kpool_mark_new(): exactly the pools formed by the cells
// just appended are new. An appended position is newer than every position already cached for
// its sequence, so it cannot fall inside a pool that already existed -- there is nothing to
// search for. `pools_before[s]` is st.seqs[s].pools.size() as it was BEFORE the append.
void llama_kpool_mark_new_appended(
        llama_kpool_state & st,
        const std::vector<uint32_t> & pools_before);

struct llama_kpool_dims {
    uint32_t kpool    = 0;
    uint32_t n_pool   = 0;   // == llama_kpool_pad(st.n_pool_real)
    uint32_t n_tokens = 0;
    uint32_t n_kv     = 0;   // the attention graph's key width; also the scatter sentinel
    uint32_t n_new    = 0;   // == st.n_new
    // The number of new-pool rows the GRAPH builds: always >= 1 and, for a given n_tokens,
    // CONSTANT in the steady state, so that the graph's node count does not depend on whether
    // this particular ubatch happened to complete a pool. Rows [n_new, n_new_g) are padding:
    // they read `dummy_cell` and write `sink`. See llama_kpool_fill().
    uint32_t n_new_g  = 0;
    uint32_t sink     = 0;   // idx_l row the padded write-backs land in (== the cache size)
    uint32_t n_sel    = 0;   // gather only
    uint32_t n_top    = 0;   // gather only: pools per token in the selection
    bool     gather   = false;

    // PXA_QSA: which member of a pool lends the pooled key its ROPE POSITION. qwen4exp pools
    // the RAW indexer key and rotates the pooled result once, at one position per block, so a
    // position has to be chosen; the architecture's reference carries a `blk_pos` input and
    // does not say which member it names. 0 (the block's first member) is the default and the
    // reading the block-start convention gives; PXA_QSA_BLKPOS selects another member so the
    // selection-recall probe can decide it by measurement. Only read when `new_blk_pos` is
    // requested, so the glm5next path never sees it.
    uint32_t blk_pos_member = 0;
};

// Destination buffers. Any of gather_mask / new_pool_idxs / new_pool_rep may be null when
// the corresponding count is zero (see the asserts in the .cpp).
struct llama_kpool_bufs {
    int32_t * pool_cells    = nullptr;   // [n_pool]
    int32_t * pool_idxs     = nullptr;   // [kpool, n_pool]
    float   * pool_mask     = nullptr;   // [n_pool, n_tokens]
    int32_t * tail_idxs     = nullptr;   // [kpool - 1, n_tokens]
    float   * gather_mask   = nullptr;   // [n_sel, n_tokens]
    int32_t * new_pool_idxs = nullptr;   // [kpool, n_new]
    int64_t * new_pool_rep  = nullptr;   // [n_new]

    // PXA_QSA, optional: the rope position of the pooled key of each pool in the fixed-width
    // NEW-pool group, laid out the way this tree's mrope inp_pos is -- SECTION MAJOR,
    // new_blk_pos[s*n_new_g + i] for s < 4, with the fourth section 0 (see llama_set_inputs).
    // Only the pools this ubatch (re)pools need a position: every other pooled key was
    // rotated when it was written. Null for glm5next, whose pooled key is not rotated at all.
    int32_t * new_blk_pos   = nullptr;   // [4*n_new_g]
};

// Fill every host-side input of one ubatch. Pure: no ggml, no context, no cache.
void llama_kpool_fill(
        const llama_kpool_state & st,
        const llama_kpool_dims  & d,
        const std::vector<llama_kpool_tok_desc> & toks,
        const llama_kpool_bufs  & b);

//
// ---------------------------------------------------------------------------------------
// the per-context plan — lifecycle hooks, mirroring llama-kv-cache-dsv4.h
// ---------------------------------------------------------------------------------------
//

// The graph input tensors the glm5next builder registers, filled by llama_kpool_set_inputs().
struct llama_kpool_inputs {
    ggml_tensor * pool_cells    = nullptr;   // I32 [n_pool]
    ggml_tensor * pool_idxs     = nullptr;   // I32 [kpool, n_pool]
    ggml_tensor * pool_mask     = nullptr;   // F32 [n_pool, n_tokens]
    ggml_tensor * tail_idxs     = nullptr;   // I32 [kpool - 1, n_tokens]
    ggml_tensor * gather_mask   = nullptr;   // F32 [n_sel, 1, 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr;   // I32 [kpool, n_new]
    ggml_tensor * new_pool_rep  = nullptr;   // I64 [n_new]
    ggml_tensor * new_blk_pos   = nullptr;   // I32 [4*n_new_g] (PXA_QSA only; see the bufs note)
};

// Derive this batch's pool grid from the CURRENT contents of lctx.kv_self. MUST run after
// the batch's cells have been allocated (llama_kv_cache_find_slot) and BEFORE the graph is
// built: the graph sizes every k-pool input from this plan and llama_kpool_set_inputs()
// then fills those tensors from the same plan.
// `worst_case` is the RESERVE graph: no batch is ever run through it, so instead of reading a
// batch that has no cells yet, the dims are set to the largest the context can ever need. The
// plan a reserve build leaves behind is never filled (llama_kpool_set_inputs refuses it).
void llama_kpool_build_plan(llama_context & lctx, const llama_batch & batch, bool worst_case);

// Rebuild the plan for an ubatch whose graph the engine wants to REUSE. Returns true only if
// every shape the reused graph baked in is unchanged, in which case the plan now describes
// this ubatch and set_inputs is safe; false means "rebuild the graph".
bool llama_kpool_replan_for_reuse(llama_context & lctx, const llama_batch & batch);

const llama_kpool_state & llama_kpool_get_state (const llama_context & lctx);
const llama_kpool_dims  & llama_kpool_get_dims  (const llama_context & lctx);
llama_kpool_inputs      & llama_kpool_get_inputs(llama_context & lctx);

// Size the plan-derived parts of the graph. Both read the plan built above.
uint32_t llama_kpool_get_n_pool     (const llama_context & lctx);
uint32_t llama_kpool_get_n_new      (const llama_context & lctx);
bool     llama_kpool_get_cache_safe (const llama_context & lctx);

void llama_kpool_set_inputs(llama_context & lctx, const llama_batch & batch);

// PXA_QSA: whether qwen4exp's query-time sparse attention is live for this context, i.e.
// exactly the condition build_qwen4exp() plans a k-pool grid under. Every call site that has
// to agree with the graph -- set_inputs, the graph-reuse replan -- asks this and not its own
// approximation of it: reading a plan that was never built is an assert, and filling inputs a
// graph never registered is silent nonsense.
bool llama_qsa_planned(const llama_context & lctx);

// Any sequence edit regrids the pools (the grid is relative to each sequence's first cached
// position), which invalidates every cached pooled key. The next batch then re-pools
// everything from the still-valid key|gate rows.
void llama_kpool_mark_stale(llama_context & lctx);
