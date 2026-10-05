// PXA_GLM5NEXT: the pooled ("k-pool") indexer cache of GLM-5.3-Flash — the per-context plan.
//
// The grid derivation itself lives in src/llama-kpool-grid.cpp (pure, unit-tested). This file
// is only the glue: read the current cache cells, build one ubatch's plan, and fill the graph
// inputs the glm5next builder registered. Mirrors llama-kv-cache-dsv4.cpp's lifecycle shape.

#include "llama-kv-cache-kpool.h"

#include <random>   // llama-context.h -> llama-sampling.h uses std::mt19937 without including it

#include "llama-context.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-qsa-prof.h"

#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>

//
// ---------------------------------------------------------------------------------------
// the per-context plan
// ---------------------------------------------------------------------------------------
//

struct llama_kpool_plan {
    llama_kpool_state  st;
    llama_kpool_dims   dims;
    llama_kpool_inputs inputs;

    // a sequence edit regrids the pools; the next batch's FIRST ubatch re-pools everything
    bool stale = true;

    // guards a read before llama_kpool_build_plan() ran for this batch
    bool built = false;

    // the plan describes the RESERVE graph's worst case, not a real batch
    bool reserve_only = false;

    // host staging kept alive across set_inputs
    std::vector<llama_kpool_tok_desc> toks;

    // PXA_QSA: the worst-case (reserve) plan's state. Kept SEPARATE so that a reserve build --
    // which happens between two real ubatches -- does not destroy the incremental layout below.
    llama_kpool_state st_ws;

    // PXA_QSA, the append-only grid (see llama_kpool_append_layout): `st` is carried across
    // ubatches instead of being rebuilt, and `applied_used` is the kv.used it describes. The
    // fast path is taken only when the cache has grown by exactly this ubatch's own cells and
    // every one of them checks out; anything else falls back to the full rebuild.
    bool     have_state   = false;
    uint32_t applied_used = 0;
    uint32_t applied_size = 0;
    uint32_t applied_nseq = 0;
    uint32_t applied_kpool = 0;

    // Occupied cells the carried layout describes. Counted the way the from-scratch path counts
    // them (cells whose seq set is non-empty), not read from kv.used, so that d.gather is
    // decided on exactly the same number both paths always used.
    uint32_t n_occupied = 0;
};

static llama_kpool_plan g_kpool;   // single-context module (first cut is -np 1 / one context)

const llama_kpool_state & llama_kpool_get_state(const llama_context & /*lctx*/) {
    GGML_ASSERT(g_kpool.built && "k-pool plan read before llama_kpool_build_plan()");
    return g_kpool.reserve_only ? g_kpool.st_ws : g_kpool.st;
}

const llama_kpool_dims & llama_kpool_get_dims(const llama_context & /*lctx*/) {
    GGML_ASSERT(g_kpool.built && "k-pool plan read before llama_kpool_build_plan()");
    return g_kpool.dims;
}

llama_kpool_inputs & llama_kpool_get_inputs(llama_context & /*lctx*/) {
    return g_kpool.inputs;
}

uint32_t llama_kpool_get_n_pool    (const llama_context & lctx) { return llama_kpool_get_dims(lctx).n_pool; }
uint32_t llama_kpool_get_n_new     (const llama_context & lctx) { return llama_kpool_get_dims(lctx).n_new;  }
bool     llama_kpool_get_cache_safe(const llama_context & lctx) { return llama_kpool_get_state(lctx).cache_safe; }

void llama_kpool_mark_stale(llama_context & /*lctx*/) {
    g_kpool.stale = true;
    // A sequence edit regrids the pools, so the carried layout describes a cache that no longer
    // exists. Drop it; the next ubatch rebuilds from scratch.
    g_kpool.have_state = false;
}

// PXA_QSA_GRID_INCR (default ON): carry the pool grid across ubatches and extend it with the
// ubatch's own cells instead of rescanning the whole cache and rebuilding it, which is what
// this did on every decoded token. =0 forces the from-scratch path, which is the A/B control
// and the fallback every non-append case takes anyway.
static bool kpool_grid_incr_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXA_QSA_GRID_INCR");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

// PXA_QSA_GRID_VERIFY=1: after every incremental update, rebuild the layout from scratch and
// abort on any difference. The claim the fast path makes is EQUALITY, not approximation, so it
// is checkable -- expensively, which is why it is a lever and not the default.
static bool kpool_grid_verify_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXA_QSA_GRID_VERIFY");
        return e != nullptr && atoi(e) != 0;
    }();
    return v;
}

// Does the carried layout already hold exactly this (pos, cell) for this sequence? Used to
// re-verify the "the plan for this ubatch is already applied" shortcut, which kv.used alone
// cannot justify: a rollback that REMOVES d cells followed by an ubatch that adds d leaves
// kv.used exactly where it was, and the carried layout would then describe cells that are gone.
static bool kpool_layout_holds(const llama_kpool_state & st, llama_seq_id s,
                               llama_pos pos, uint32_t cell) {
    if (s < 0 || (size_t) s >= st.seqs.size()) {
        return false;
    }
    const auto & cells = st.seqs[s].cells;
    const auto it = std::lower_bound(cells.begin(), cells.end(),
                                     std::make_pair(pos, 0u));
    return it != cells.end() && it->first == pos && it->second == cell;
}

// WHAT THIS COMPARES, AND WHAT IT DELIBERATELY DOES NOT (2026-09-09).
//
// The check used to include `a.n_new == b.n_new` and `seqs[s].is_new`, and both are PER-UBATCH
// DELTA fields, not properties of the grid: the header defines n_new as "of those, the ones this
// ubatch must (re)pool" and is_new as "1 where THIS UBATCH (re)wrote a member of the pool". The
// incremental path knows what this ubatch added; a from-scratch rebuild has no history and
// necessarily reports everything it just formed. They can only agree by accident. On the real
// Flash-Next seat at 86,401 fill the lever aborted on exactly that: "n_pool_real 256 vs 256,
// n_new 0 vs 128" -- the grid itself agreed and the bookkeeping did not.
//
// Worse, `ok` was folded with && and the per-sequence loop was guarded by it, so the abort fired
// BEFORE cells, pools or pos_min were ever compared. The instrument whose entire purpose is to
// prove the appended grid equals the rebuilt one had therefore never once compared the grid.
//
// So: every GRID field is compared, unconditionally, and the message names the field and the
// sequence that diverged instead of printing two numbers that may have nothing to do with it.
// The delta fields are reported alongside as context, never as a failure.
// Split out as a PURE PREDICATE so it can be tested without aborting: the abort below is the
// only caller that turns a difference into a crash, and tests/test-qsa-grid-verify.cpp asserts
// the predicate's answers directly (a grid difference is named; a delta-field-only difference is
// not a difference at all).
const char * llama_kpool_grid_diff(const llama_kpool_state & a, const llama_kpool_state & b, int * bad_seq) {
    const char * what = nullptr;
    int          bad  = -1;

    if (a.n_pool_real  != b.n_pool_real ) { what = "n_pool_real";  }
    else if (a.cache_safe != b.cache_safe) { what = "cache_safe";  }
    else if (a.seqs.size() != b.seqs.size()) { what = "seqs.size"; }
    else {
        for (size_t s = 0; s < a.seqs.size(); ++s) {
            const auto & x = a.seqs[s];
            const auto & y = b.seqs[s];
            if      (x.pos_min != y.pos_min) { what = "pos_min"; bad = (int) s; break; }
            else if (x.cells   != y.cells  ) { what = "cells";   bad = (int) s; break; }
            else if (x.pools   != y.pools  ) { what = "pools";   bad = (int) s; break; }
            // scan_j is the pool scan's resume cursor. The append path's whole argument is that
            // resuming from it reproduces the from-scratch grid, so it SHOULD agree -- but it is
            // an internal cursor rather than something the graph consumes, and the point of this
            // rewrite is to stop the instrument inventing failures. It is reported, not gated.
        }
    }

    if (bad_seq) { *bad_seq = bad; }
    return what;
}

static void kpool_state_equal_or_die(const llama_kpool_state & a, const llama_kpool_state & b) {
    int bad = -1;
    const char * what = llama_kpool_grid_diff(a, b, &bad);
    if (what) {
        GGML_ABORT("PXA_QSA_GRID_VERIFY: the incremental pool grid differs from the from-scratch "
                   "grid in %s%s%d (n_pool_real %u vs %u, scan_j %u vs %u; per-ubatch deltas, "
                   "NOT compared: n_new %u vs %u)",
                   what, bad >= 0 ? " for seq " : "", bad,
                   a.n_pool_real, b.n_pool_real,
                   bad >= 0 ? a.seqs[bad].scan_j : 0u, bad >= 0 ? b.seqs[bad].scan_j : 0u,
                   a.n_new, b.n_new);
    }
}

bool llama_qsa_planned(const llama_context & lctx) {
    if (lctx.model.arch != LLM_ARCH_QWEN4EXP || !llama_qsa_enabled()) {
        return false;
    }
    if (lctx.model.hparams.indexer_kpool <= 1) {
        return false;
    }
    // PXA_QSA_MIN_FILL, the depth gate. Asked HERE, in the one predicate every call site shares
    // -- the graph builder, llama_kpool_set_inputs and the graph-reuse replan -- because a gate
    // that some of them disagreed with would fill inputs no graph registered, or replan a plan
    // that was never built. The count is the cache's occupied cells, the same quantity the
    // gather/scatter branch is decided on, so the two rules read the same number.
    //
    // It moves once per context in practice (fill only grows at decode), so the graph-shape
    // change it causes costs one re-reserve at the crossing rather than one per token. A
    // context that crosses back down -- a seq_rm, a shorter prompt on the same slot -- pays it
    // again; that is the honest cost of a gate and it is far smaller than running QSA where it
    // is measured to lose.
    if (lctx.kv_self.used < llama_qsa_min_fill()) {
        return false;
    }
    // the side rows are allocated per full-attention layer and only when the lever was on at
    // cache-init time, so their presence is the honest test that the graph can build the path
    for (ggml_tensor * t : lctx.kv_self.idx_l) {
        if (t != nullptr) {
            return true;
        }
    }
    return false;
}

// The reserve / worst-case graph is built from llama_batch_get_one(), whose pos array is null
// and whose positions are implied by all_pos_0/all_pos_1 (same trap as dsv4_batch_pos()).
static llama_pos kpool_batch_pos(const llama_batch & batch, int32_t i) {
    return batch.pos ? batch.pos[i] : batch.all_pos_0 + (llama_pos) i*batch.all_pos_1;
}

static void kpool_batch_seqs(const llama_batch & batch, int32_t i, std::vector<llama_seq_id> & out) {
    out.clear();
    const int32_t n = batch.n_seq_id ? batch.n_seq_id[i] : 1;
    for (int32_t k = 0; k < n; ++k) {
        out.push_back(batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][k] : batch.all_seq_id);
    }
    if (out.empty()) {
        out.push_back(batch.all_seq_id);
    }
}

void llama_kpool_build_plan(llama_context & lctx, const llama_batch & batch, bool worst_case) {
    const auto & hparams = lctx.model.hparams;
    const auto & kv      = lctx.kv_self;

    const uint32_t kpool     = hparams.indexer_kpool;
    const uint32_t n_seq_max = std::max<uint32_t>(1, lctx.cparams.n_seq_max);

    GGML_ASSERT(kpool > 1);

    // The decode gather keeps attention cost proportional to top_k instead of to context length;
    // it only pays off (and is only correct to prefer) when the context is genuinely longer than
    // the selection.
    constexpr uint32_t max_ub = 16;

    auto & d = g_kpool.dims;

    d.kpool    = kpool;
    d.n_tokens = (uint32_t) batch.n_tokens;
    d.n_kv     = kv.n;

    // PXA_QSA_BLKPOS: which member of a block lends the pooled indexer key its rope position.
    // Read once; only qwen4exp asks for new_blk_pos at all, so this is inert for glm5next.
    static const uint32_t blk_pos_member = [] {
        const char * e = getenv("PXA_QSA_BLKPOS");
        const long v = e != nullptr ? atol(e) : 0;
        return (uint32_t) (v > 0 ? v : 0);
    }();
    d.blk_pos_member = std::min(blk_pos_member, kpool - 1);

    if (worst_case) {
        // The reserve graph is built from llama_batch_get_one() against a cache that has no
        // cells yet, so deriving a plan from it would size every k-pool input to nothing and
        // ggml-alloc would then have to grow the compute buffer on the first real decode.
        // Size it to the largest grid the context can hold instead, and take the SCATTER path,
        // whose n_kv-wide mask is the bigger of the two.
        g_kpool.st_ws = llama_kpool_state();
        g_kpool.st_ws.seqs.resize(n_seq_max);
        g_kpool.toks.clear();

        d.n_pool = llama_kpool_pad(kv.size/kpool);
        d.n_new  = llama_kpool_n_new_floor(d.n_pool, d.n_tokens, kpool);
        d.n_new_g = d.n_new;
        d.sink    = kv.size;
        d.n_top  = llama_kpool_n_top(d.n_pool, hparams.indexer_top_k, kpool);
        d.n_sel  = llama_kpool_n_sel(d.n_top, kpool, hparams.indexer_kpool_select_tail);
        // Take the SAME branch a real ubatch of this size will take, against a full cache. The
        // scatter path is the bigger of the two in memory, so a reserve that always chose it
        // sized the buffer safely -- but it also built a different NODE COUNT from the decode
        // graph, and this engine re-plans whenever the count moves, which it then did on every
        // single token. Deciding it the same way here costs nothing and stops that.
        d.gather = d.n_tokens <= max_ub && kv.size > d.n_sel;

        g_kpool.built        = true;
        g_kpool.reserve_only = true;
        return;
    }

    // PXA_QSA_PROF: the three host stages below are the part of the per-token cost no device
    // profiler can see -- a full rescan of the cache's cells, a layout rebuild and a mark pass.
    // Before PXA_QSA_GRID_INCR they ran on EVERY ubatch, i.e. on every decoded token: 4.5 ms at
    // 86k fill and 7.2 ms at 140k on this box, against an 11.6 ms/token gap to dense at 86k.
    const bool prof = pxa_qsa_prof_on();
    int64_t t_prof = prof ? ggml_time_us() : 0;

    g_kpool.toks.resize(batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        g_kpool.toks[i].pos = kpool_batch_pos(batch, i);
        kpool_batch_seqs(batch, i, g_kpool.toks[i].seqs);
    }

    //
    // The append-only fast path.
    //
    // It is taken only when the carried layout provably still describes the cache. The three
    // structural facts it rests on:
    //
    //  * every mutation that is not an append either shrinks kv.used (seq_rm, seq_keep, the
    //    checkpoint rollbacks) or grows it by more than this ubatch (a state restore), so
    //    `kv.used == applied_used + n_tokens` is a real guard and not a hope;
    //  * the six public sequence-edit entry points call llama_kpool_mark_stale(), which drops
    //    the carried layout outright;
    //  * a DEFRAG moves cells while leaving kv.used alone, which this guard would not catch --
    //    so QSA turns defrag off at context creation (see llama.cpp) rather than pretending.
    //
    // Everything else is re-verified per cell below, and any failure falls through to the
    // from-scratch rebuild, which is always correct.
    //
    bool appended = false;
    if (kpool_grid_incr_enabled() && g_kpool.have_state && !g_kpool.stale &&
        g_kpool.applied_size == kv.size && g_kpool.applied_nseq == n_seq_max &&
        g_kpool.applied_kpool == kpool) {

        if (g_kpool.applied_used == kv.used) {
            // The plan for THIS ubatch has already been applied -- llama_kpool_build_plan runs
            // more than once per ubatch (the graph-reuse replan, then the rebuild, and a
            // reserve in between). Re-deriving the dims below is all that is left to do.
            //
            // But kv.used being unchanged does NOT by itself mean the cache is unchanged: a
            // speculative rollback removes d cells and the next ubatch adds d back, landing on
            // the same count with different contents. So verify that the layout actually holds
            // this ubatch's own cells, at this ubatch's own positions, before believing it.
            const uint32_t head = kv.head;
            bool ok = head + (uint32_t) batch.n_tokens <= kv.size;
            for (int32_t i = 0; ok && i < batch.n_tokens; ++i) {
                const auto & c = kv.cells[head + i];
                ok = !c.is_empty() && c.pos == g_kpool.toks[i].pos &&
                     !g_kpool.toks[i].seqs.empty() &&
                     kpool_layout_holds(g_kpool.st, g_kpool.toks[i].seqs[0], c.pos, head + i);
            }
            appended = ok;
        } else if (g_kpool.applied_used + (uint32_t) batch.n_tokens == kv.used) {
            const uint32_t head = kv.head;
            std::vector<llama_kpool_cell_desc> added;
            added.reserve(batch.n_tokens);
            bool ok = head + (uint32_t) batch.n_tokens <= kv.size;
            for (int32_t i = 0; ok && i < batch.n_tokens; ++i) {
                const auto & c = kv.cells[head + i];
                if (c.is_empty() || c.pos != g_kpool.toks[i].pos) {
                    ok = false;
                    break;
                }
                llama_kpool_cell_desc cd;
                cd.pos  = c.pos;
                cd.cell = head + i;
                cd.seqs.assign(c.seqs().begin(), c.seqs().end());
                if (cd.seqs != g_kpool.toks[i].seqs) {
                    ok = false;
                    break;
                }
                added.push_back(std::move(cd));
            }
            if (ok) {
                std::vector<uint32_t> pools_before(g_kpool.st.seqs.size());
                for (size_t sq = 0; sq < g_kpool.st.seqs.size(); ++sq) {
                    pools_before[sq] = (uint32_t) g_kpool.st.seqs[sq].pools.size();
                }
                if (llama_kpool_append_layout(g_kpool.st, added, kpool)) {
                    llama_kpool_mark_new_appended(g_kpool.st, pools_before);
                    g_kpool.n_occupied += (uint32_t) added.size();
                    appended = true;
                }
            }
        }
    }

    if (appended) {
        if (prof) { pxa_qsa_prof_host_add(PXA_QSA_H_LAYOUT, ggml_time_us() - t_prof); }
        if (kpool_grid_verify_enabled()) {
            std::vector<llama_kpool_cell_desc> ref_cells;
            ref_cells.reserve(kv.used);
            for (uint32_t i = 0; i < kv.size; ++i) {
                const auto & c = kv.cells[i];
                if (c.is_empty()) {
                    continue;
                }
                llama_kpool_cell_desc cd;
                cd.pos  = c.pos;
                cd.cell = i;
                cd.seqs.assign(c.seqs().begin(), c.seqs().end());
                ref_cells.push_back(std::move(cd));
            }
            llama_kpool_state ref = llama_kpool_build_layout(ref_cells, kpool, n_seq_max);
            llama_kpool_mark_new(ref, g_kpool.toks, kpool, /*all_new*/ false);
            kpool_state_equal_or_die(g_kpool.st, ref);
        }
    } else {
        // the occupied cells, as the pool grid sees them
        std::vector<llama_kpool_cell_desc> cells;
        cells.reserve(kv.used);
        for (uint32_t i = 0; i < kv.size; ++i) {
            const auto & c = kv.cells[i];
            if (c.is_empty()) {
                continue;
            }
            llama_kpool_cell_desc d;
            d.pos  = c.pos;
            d.cell = i;
            d.seqs.assign(c.seqs().begin(), c.seqs().end());
            cells.push_back(std::move(d));
        }
        if (prof) { const int64_t t = ggml_time_us(); pxa_qsa_prof_host_add(PXA_QSA_H_SCAN, t - t_prof); t_prof = t; }

        g_kpool.st = llama_kpool_build_layout(cells, kpool, n_seq_max);
        g_kpool.n_occupied = (uint32_t) cells.size();
        if (prof) { const int64_t t = ggml_time_us(); pxa_qsa_prof_host_add(PXA_QSA_H_LAYOUT, t - t_prof); t_prof = t; }

        llama_kpool_mark_new(g_kpool.st, g_kpool.toks, kpool, g_kpool.stale);
        if (prof) { pxa_qsa_prof_host_add(PXA_QSA_H_MARKNEW, ggml_time_us() - t_prof); }
    }

    g_kpool.stale = false;
    g_kpool.have_state   = true;
    g_kpool.applied_used = kv.used;
    g_kpool.applied_size = kv.size;
    g_kpool.applied_nseq = n_seq_max;
    g_kpool.applied_kpool = kpool;

    d.n_pool = llama_kpool_pad(g_kpool.st.n_pool_real);
    d.n_new  = g_kpool.st.n_new;
    // The graph is built with a FIXED number of new-pool rows: the floor below, which depends
    // only on the ubatch width and the grid size, and never fewer than the pools this ubatch
    // actually completed (a sequence edit re-pools everything and legitimately needs more).
    // In steady-state decode n_new alternates 0/1 while the floor is a constant 1, so the
    // node count -- and therefore the reserved plan -- stops moving every single token.
    d.n_new_g = std::max(d.n_new, llama_kpool_n_new_floor(d.n_pool, d.n_tokens, kpool));
    d.sink    = kv.size;

    d.n_top  = llama_kpool_n_top(d.n_pool, hparams.indexer_top_k, kpool);
    d.n_sel  = llama_kpool_n_sel(d.n_top, kpool, hparams.indexer_kpool_select_tail);
    // Decided on the OCCUPIED CELL COUNT, not on kv.n. Same intent -- "is the context longer
    // than the selection, so that attending only the selection is the cheaper graph" -- but
    // kv.n is the wrong reading of it here: pxa_reserve_real_graph builds its reserve with
    // kv.n forced to kv.size, so a kv.n rule had the reserve take the gather branch while the
    // decode it was reserving for took the scatter branch (kv.n = 128 <= n_sel = 259). Four
    // nodes per MLA block apart, 44 over the trunk, and this engine answers a moved node count
    // with a full re-reserve of every compute buffer -- on every single token. The reserve
    // does NOT touch the cells, so counting them is a reading both sides agree on.
    // (Capacity, kv.size, would agree too, but it forces gather at every fill, and the gather
    // path is only exercised by decode-shaped ubatches -- a short PROMPT is <= max_ub as well,
    // and taking gather there produces garbage. Measured, 2026-09-08.)
    d.gather = d.n_tokens <= max_ub && g_kpool.n_occupied > d.n_sel;

    g_kpool.built        = true;
    g_kpool.reserve_only = false;
}

bool llama_kpool_replan_for_reuse(llama_context & lctx, const llama_batch & batch) {
    if (!g_kpool.built || g_kpool.reserve_only) {
        return false;
    }
    // A REUSED graph is never rebuilt, so llama_kpool_build_plan() is never called for that
    // ubatch -- and llama_kpool_set_inputs() would then fill this ubatch's k-pool tensors from
    // the PREVIOUS ubatch's grid: silently wrong attention, not a crash. Rebuild the plan here
    // and let the caller reuse the graph only if every shape the graph baked in is unchanged.
    // A mismatch is not an error: the caller falls back to a full rebuild, which builds the
    // plan again from the same inputs and reaches the same answer.
    const llama_kpool_dims before = g_kpool.dims;
    llama_kpool_build_plan(lctx, batch, /*worst_case*/ false);
    const llama_kpool_dims & d = g_kpool.dims;
    return d.kpool    == before.kpool    && d.n_pool == before.n_pool &&
           d.n_tokens == before.n_tokens && d.n_kv   == before.n_kv   &&
           d.n_new_g  == before.n_new_g  && d.n_sel  == before.n_sel  &&
           d.n_top    == before.n_top    && d.gather == before.gather &&
           d.sink     == before.sink;
}

void llama_kpool_set_inputs(llama_context & lctx, const llama_batch & batch) {
    GGML_ASSERT(g_kpool.built && "k-pool inputs filled before llama_kpool_build_plan()");
    GGML_ASSERT(!g_kpool.reserve_only &&
                "k-pool inputs filled from a RESERVE plan -- the real batch's plan was not built");

    auto & in = g_kpool.inputs;
    const auto & d = g_kpool.dims;

    if (in.pool_cells == nullptr) {
        return;   // no glm5next graph was built for this batch
    }

    GGML_ASSERT((uint32_t) batch.n_tokens == d.n_tokens &&
                "the k-pool plan was built for a different batch than the graph");

    GGML_ASSERT(ggml_backend_buffer_is_host(in.pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(in.pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(in.pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(in.tail_idxs->buffer));

    GGML_ASSERT(in.pool_cells->ne[0] == (int64_t) d.n_pool);
    GGML_ASSERT(in.pool_idxs->ne[0] == (int64_t) d.kpool && in.pool_idxs->ne[1] == (int64_t) d.n_pool);
    GGML_ASSERT(in.pool_mask->ne[0] == (int64_t) d.n_pool && in.pool_mask->ne[1] == (int64_t) d.n_tokens);
    GGML_ASSERT(in.tail_idxs->ne[0] == (int64_t) d.kpool - 1 && in.tail_idxs->ne[1] == (int64_t) d.n_tokens);

    llama_kpool_bufs b;
    b.pool_cells = (int32_t *) in.pool_cells->data;
    b.pool_idxs  = (int32_t *) in.pool_idxs->data;
    b.pool_mask  = (float   *) in.pool_mask->data;
    b.tail_idxs  = (int32_t *) in.tail_idxs->data;

    if (in.gather_mask) {
        GGML_ASSERT(ggml_backend_buffer_is_host(in.gather_mask->buffer));
        GGML_ASSERT(in.gather_mask->ne[0] == (int64_t) d.n_sel && in.gather_mask->ne[3] == (int64_t) d.n_tokens);
        b.gather_mask = (float *) in.gather_mask->data;
    }
    if (in.new_pool_idxs) {
        GGML_ASSERT(ggml_backend_buffer_is_host(in.new_pool_idxs->buffer));
        GGML_ASSERT(in.new_pool_idxs->ne[0] == (int64_t) d.kpool && in.new_pool_idxs->ne[1] == (int64_t) d.n_new_g);
        b.new_pool_idxs = (int32_t *) in.new_pool_idxs->data;
    }
    if (in.new_pool_rep) {
        GGML_ASSERT(ggml_backend_buffer_is_host(in.new_pool_rep->buffer));
        GGML_ASSERT(in.new_pool_rep->ne[0] == (int64_t) d.n_new_g);
        b.new_pool_rep = (int64_t *) in.new_pool_rep->data;
    }
    if (in.new_blk_pos) {
        GGML_ASSERT(ggml_backend_buffer_is_host(in.new_blk_pos->buffer));
        GGML_ASSERT(in.new_blk_pos->ne[0] == (int64_t) 4*d.n_new_g);
        b.new_blk_pos = (int32_t *) in.new_blk_pos->data;
    }

    const int64_t t_fill = pxa_qsa_prof_on() ? ggml_time_us() : 0;
    llama_kpool_fill(g_kpool.st, d, g_kpool.toks, b);
    if (pxa_qsa_prof_on()) { pxa_qsa_prof_host_add(PXA_QSA_H_FILL, ggml_time_us() - t_fill); }

    GGML_UNUSED(lctx);
}
