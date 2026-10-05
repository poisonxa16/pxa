// PXA_GLM5NEXT: the pooled ("k-pool") indexer GRID of GLM-5.3-Flash — the PURE core.
//
// This file deliberately depends on nothing but ggml.h and llama.h so that
// tests/test-kpool-cache.cpp can compile it straight into a host-only unit test: no context,
// no model, no cache object, no graph.
//
// Ported from llama.cpp PR #27773 `src/llama-memory-hybrid-idx.cpp`. Copyright (c) 2023-2026
// The ggml authors. MIT. See the header for what is deliberately different.
//
// WHAT IS TRANSCRIBED AND MUST NOT BE "IMPROVED"
// ----------------------------------------------
// llama_kpool_build_layout() and llama_kpool_fill() below are upstream's grid derivation and
// input fill, with only the single-stream folding and the F32-mask narrowing applied. Getting
// them wrong does not crash: the model loads, attends to the WRONG cells, and emits fluent
// garbage. tests/test-kpool-cache.cpp is the guard.

#include "llama-kv-cache-kpool.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

//
// ---------------------------------------------------------------------------------------
// the pure core
// ---------------------------------------------------------------------------------------
//

uint32_t llama_kpool_pad(uint32_t n_pool_real) {
    // The last padded pool is always unused, so `pool_idxs` can carry a sentinel row for the
    // padded entries without ever colliding with a real pool. GGML_PAD(n+1, 64) guarantees at
    // least one spare even when n_pool_real is already a multiple of 64.
    return std::max<uint32_t>(64u, GGML_PAD(n_pool_real + 1, 64u));
}

uint32_t llama_kpool_n_top(uint32_t n_pool, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    return std::min<uint32_t>(n_pool, indexer_top_k / kpool);
}

uint32_t llama_kpool_n_sel(uint32_t n_top, uint32_t kpool, bool select_tail) {
    return kpool*n_top + (select_tail ? kpool - 1 : 0);
}

uint32_t llama_kpool_n_new_floor(uint32_t n_pool, uint32_t n_tokens, uint32_t kpool) {
    GGML_ASSERT(kpool > 0 && n_pool > 1);
    return std::min<uint32_t>(n_pool - 1, n_tokens/kpool + 1);
}

llama_kpool_state llama_kpool_build_layout(
        const std::vector<llama_kpool_cell_desc> & cells,
        uint32_t kpool,
        uint32_t n_seq_max) {
    GGML_ASSERT(kpool > 1);
    GGML_ASSERT(n_seq_max > 0);

    llama_kpool_state st;
    st.seqs.resize(n_seq_max);

    for (const auto & c : cells) {
        if (c.seqs.empty()) {
            continue;
        }
        for (const llama_seq_id s : c.seqs) {
            if (s < 0 || (uint32_t) s >= n_seq_max) {
                continue;
            }
            st.seqs[s].cells.emplace_back(c.pos, c.cell);
        }
        if (c.seqs.size() > 1) {
            // A cell shared between sequences cannot carry ONE cached pooled key: the pool grid
            // is relative to each sequence's own first position, so the same cell can sit at a
            // different pool offset in each. The graph then pools in place every ubatch.
            st.cache_safe = false;
        }
    }

    for (auto & sq : st.seqs) {
        if (sq.cells.empty()) {
            continue;
        }
        if (!std::is_sorted(sq.cells.begin(), sq.cells.end())) {
            std::sort(sq.cells.begin(), sq.cells.end());
        }

        sq.pos_min = sq.cells.front().first;

        // Pools start at the sequence's first cached position and cover `kpool` CONSECUTIVE
        // positions. A gap (a seq_rm hole, a shifted context) simply yields no pool there; the
        // scan then steps one cell and re-tries, so the grid stays anchored on pos_min.
        size_t j = 0;
        for (; j + kpool <= sq.cells.size(); ) {
            const llama_pos p0 = sq.cells[j].first;
            if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                ++j;
                continue;
            }
            bool ok = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sq.pools.push_back((uint32_t) j);
                j += kpool;
            } else {
                ++j;
            }
        }
        // where the scan stopped: llama_kpool_append_layout() resumes here (see the header)
        sq.scan_j = (uint32_t) j;

        st.n_pool_real += (uint32_t) sq.pools.size();
    }

    for (auto & sq : st.seqs) {
        sq.is_new.assign(sq.pools.size(), 0);
    }

    return st;
}

void llama_kpool_mark_new(
        llama_kpool_state & st,
        const std::vector<llama_kpool_tok_desc> & toks,
        uint32_t kpool,
        bool all_new) {
    st.n_new = 0;

    all_new = all_new || !st.cache_safe;

    // positions this ubatch wrote, per sequence
    std::vector<std::vector<llama_pos>> upos(st.seqs.size());
    if (!all_new) {
        for (const auto & t : toks) {
            for (const llama_seq_id s : t.seqs) {
                if (s >= 0 && (size_t) s < upos.size()) {
                    upos[s].push_back(t.pos);
                }
            }
        }
        for (auto & v : upos) {
            if (!std::is_sorted(v.begin(), v.end())) {
                std::sort(v.begin(), v.end());
            }
        }
    }

    for (size_t s = 0; s < st.seqs.size(); ++s) {
        auto & sq = st.seqs[s];

        sq.is_new.assign(sq.pools.size(), all_new ? 1 : 0);
        if (all_new) {
            st.n_new += (uint32_t) sq.pools.size();
            continue;
        }

        const auto & up = upos[s];
        if (up.empty()) {
            continue;
        }

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const llama_pos p0 = sq.cells[sq.pools[pi]].first;

            // any written position inside [p0, p0 + kpool) makes the pool's key stale
            auto it = std::lower_bound(up.begin(), up.end(), p0);
            if (it != up.end() && *it < p0 + (llama_pos) kpool) {
                sq.is_new[pi] = 1;
                st.n_new++;
            }
        }
    }
}

bool llama_kpool_append_layout(
        llama_kpool_state & st,
        const std::vector<llama_kpool_cell_desc> & added,
        uint32_t kpool) {
    GGML_ASSERT(kpool > 1);

    if (added.empty()) {
        return true;
    }
    if (!st.cache_safe) {
        // a cache with cells shared between sequences re-pools everything every ubatch anyway
        return false;
    }

    // Validate BEFORE touching anything: this function's contract is that a false return leaves
    // the layout exactly as it was, so the caller can fall back to a full rebuild.
    std::vector<std::pair<llama_pos, uint32_t>> per_seq_last(st.seqs.size(), { 0, 0 });
    std::vector<uint8_t> per_seq_has(st.seqs.size(), 0);
    for (size_t s = 0; s < st.seqs.size(); ++s) {
        if (!st.seqs[s].cells.empty()) {
            per_seq_last[s] = st.seqs[s].cells.back();
            per_seq_has[s]  = 1;
        }
    }
    for (const auto & c : added) {
        if (c.seqs.size() != 1) {
            return false;   // a shared cell flips cache_safe; that is a full rebuild
        }
        const llama_seq_id s = c.seqs[0];
        if (s < 0 || (size_t) s >= st.seqs.size()) {
            return false;
        }
        if (per_seq_has[s] && c.pos <= per_seq_last[s].first) {
            return false;   // not strictly newer: the scan cursor would no longer be sound
        }
        per_seq_last[s] = { c.pos, c.cell };
        per_seq_has[s]  = 1;
    }

    // Append. `added` may interleave sequences and need not be sorted across them, but within
    // one sequence the check above has already proved it is strictly increasing.
    for (const auto & c : added) {
        auto & sq = st.seqs[c.seqs[0]];
        if (sq.cells.empty()) {
            sq.pos_min = c.pos;
        }
        sq.cells.emplace_back(c.pos, c.cell);
    }

    // Resume the pool scan from the cursor. Identical, decision for decision, to what
    // llama_kpool_build_layout() would produce over the extended cell list: every j below the
    // cursor was decided from cells that have not changed, and pos_min has not moved.
    for (auto & sq : st.seqs) {
        if (sq.cells.empty()) {
            continue;
        }
        size_t j = sq.scan_j;
        for (; j + kpool <= sq.cells.size(); ) {
            const llama_pos p0 = sq.cells[j].first;
            if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                ++j;
                continue;
            }
            bool ok = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sq.pools.push_back((uint32_t) j);
                j += kpool;
                st.n_pool_real++;
            } else {
                ++j;
            }
        }
        sq.scan_j = (uint32_t) j;
    }

    return true;
}

void llama_kpool_mark_new_appended(
        llama_kpool_state & st,
        const std::vector<uint32_t> & pools_before) {
    GGML_ASSERT(pools_before.size() == st.seqs.size());

    st.n_new = 0;
    for (size_t s = 0; s < st.seqs.size(); ++s) {
        auto & sq = st.seqs[s];
        const uint32_t before = pools_before[s];
        GGML_ASSERT(before <= sq.pools.size());

        // The previous ubatch's flags have to go: is_new is read by llama_kpool_fill() to
        // decide which pooled keys the graph recomputes, and a stale 1 would re-pool a block
        // that is already cached -- correct, but exactly the per-token work this removes.
        sq.is_new.assign(sq.pools.size(), 0);
        for (size_t pi = before; pi < sq.pools.size(); ++pi) {
            sq.is_new[pi] = 1;
            st.n_new++;
        }
    }
}

void llama_kpool_fill(
        const llama_kpool_state & st,
        const llama_kpool_dims  & d,
        const std::vector<llama_kpool_tok_desc> & toks,
        const llama_kpool_bufs  & b) {
    const uint32_t kpool    = d.kpool;
    const uint32_t n_pool   = d.n_pool;
    const uint32_t n_tokens = d.n_tokens;
    const uint32_t n_new    = d.n_new;

    GGML_ASSERT(kpool > 1);
    GGML_ASSERT(n_tokens > 0 && toks.size() == n_tokens);
    GGML_ASSERT(n_pool == llama_kpool_pad(st.n_pool_real));
    GGML_ASSERT(n_new == st.n_new);
    GGML_ASSERT(b.pool_cells && b.pool_idxs && b.pool_mask && b.tail_idxs);
    // The new-pool block is a FIXED n_new_g rows wide and is always built (n_new_g >= 1), so
    // that the graph's node count does not move with the number of pools an ubatch completes.
    GGML_ASSERT(d.n_new_g >= n_new && d.n_new_g >= 1 && d.n_new_g < n_pool);
    GGML_ASSERT(b.new_pool_idxs != nullptr);
    GGML_ASSERT(st.cache_safe == (b.new_pool_rep != nullptr));
    GGML_ASSERT(d.gather == (b.gather_mask != nullptr));

    // The gather path indexes real cache rows, so its padding must point at a REAL row and be
    // masked separately; the scatter path builds an n_kv+1 row mask, whose extra row IS the
    // sentinel. Upstream uses the first ubatch token's cell for the gather padding.
    int32_t dummy_cell = 0;
    {
        GGML_ASSERT(!toks[0].seqs.empty());
        const llama_seq_id s = toks[0].seqs[0];
        GGML_ASSERT(s >= 0 && (size_t) s < st.seqs.size());
        const auto & sq = st.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(),
                                   std::make_pair(toks[0].pos, 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == toks[0].pos &&
                    "the ubatch's own tokens must already be in the cache when the plan is built");
        dummy_cell = (int32_t) it->second;
    }

    const int32_t sentinel = d.gather ? dummy_cell : (int32_t) d.n_kv;

    // sequences present in this ubatch
    std::vector<uint8_t> seq_in_ub(st.seqs.size(), 0);
    for (const auto & t : toks) {
        for (const llama_seq_id s : t.seqs) {
            if (s >= 0 && (size_t) s < seq_in_ub.size()) {
                seq_in_ub[s] = 1;
            }
        }
    }

    // pools are laid out sequence by sequence; pool_end[ip] is the LAST member's position
    std::vector<uint32_t>  seq_pool_start(st.seqs.size(), 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    uint32_t i_new = 0;
    for (size_t s = 0; s < st.seqs.size(); ++s) {
        const auto & sq = st.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        // Upstream parks the pools of an absent sequence on the scatter sentinel, but ONLY when
        // the cache is non-unified: there cell indices are STREAM-LOCAL, so another sequence's
        // pool would scatter its "visible" zero onto a cell of the current stream. Our cache has
        // one global cell index space (n_stream == 1), which is exactly upstream's unified case,
        // where its own condition `n_stream_kv > 1` makes this false — an absent sequence's cells
        // are distinct cells, they are dropped by pool_mask, and the causal kq_mask that is added
        // to the scatter selection masks them regardless. Kept as a named constant so the
        // divergence from upstream is visible rather than silently omitted.
        const bool inert = false;
        GGML_UNUSED(seq_in_ub);

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // the pooled key lives in the LAST member's row: that row is complete exactly when
            // the pool is, so a pool is never read before it has been written
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            b.pool_cells[ip] = (int32_t) rep;

            for (uint32_t k = 0; k < kpool; ++k) {
                b.pool_idxs[(size_t) ip*kpool + k] =
                    inert ? sentinel : (int32_t) sq.cells[j + k].second;
            }

            if (sq.is_new[pi]) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    b.new_pool_idxs[(size_t) i_new*kpool + k] = (int32_t) sq.cells[j + k].second;
                }
                if (b.new_pool_rep) {
                    b.new_pool_rep[i_new] = (int64_t) rep;
                }
                // PXA_QSA: the pooled key's rope position. A pool covers `kpool` CONSECUTIVE
                // positions of one sequence, so any member names the block; which one is a
                // convention, and blk_pos_member carries it (see llama_kpool_dims).
                if (b.new_blk_pos) {
                    GGML_ASSERT(d.blk_pos_member < kpool);
                    const llama_pos bp = sq.cells[j + d.blk_pos_member].first;
                    b.new_blk_pos[                      i_new] = (int32_t) bp;
                    b.new_blk_pos[    d.n_new_g + i_new] = (int32_t) bp;
                    b.new_blk_pos[2*d.n_new_g + i_new] = (int32_t) bp;
                    b.new_blk_pos[3*d.n_new_g + i_new] = 0;   // the 4th mrope section is 0 for text
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Pad the fixed-width new-pool block. A padded row re-pools `dummy_cell` four times, which
    // is a real, in-bounds cache row, and writes the result into the sink row that idx_l keeps
    // past its last cell -- so the padding costs one 128-wide softmax per unused row and can
    // never touch a pooled key any pool actually reads.
    for (uint32_t ip = n_new; ip < d.n_new_g; ++ip) {
        for (uint32_t k = 0; k < kpool; ++k) {
            b.new_pool_idxs[(size_t) ip*kpool + k] = dummy_cell;
        }
        if (b.new_pool_rep) {
            b.new_pool_rep[ip] = (int64_t) d.sink;
        }
        if (b.new_blk_pos) {
            // a padded row's pooled key is written to the sink and never read, so its rotation
            // is arbitrary; 0 keeps it in range for the rope kernel's position lookup
            for (uint32_t s = 0; s < 4; ++s) {
                b.new_blk_pos[(size_t) s*d.n_new_g + ip] = 0;
            }
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    GGML_ASSERT(n_pool_real == st.n_pool_real);

    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        b.pool_cells[ip] = dummy_cell;   // pool_cells always addresses a real storage row
        for (uint32_t k = 0; k < kpool; ++k) {
            b.pool_idxs[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible to a token when it belongs to the token's sequence AND ends at or
    // before the token's own position
    for (uint32_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(!toks[i].seqs.empty());
        const llama_seq_id s = toks[i].seqs[0];
        const llama_pos    p = toks[i].pos;
        GGML_ASSERT(s >= 0 && (size_t) s < st.seqs.size());

        float * row = b.pool_mask + (size_t) i*n_pool;
        std::fill(row, row + n_pool, -INFINITY);

        const uint32_t p0 = seq_pool_start[s];
        const uint32_t p1 = p0 + (uint32_t) st.seqs[s].pools.size();
        const uint32_t nv = (uint32_t) (std::upper_bound(pool_end.begin() + p0,
                                                         pool_end.begin() + p1, p)
                                        - (pool_end.begin() + p0));
        std::fill(row + p0, row + p0 + nv, 0.0f);

        // The selection is ordered by descending score and the -inf pools sort last, so the
        // finite (visible) pools occupy exactly the first min(nv, n_top) ranked slots.
        if (b.gather_mask) {
            const uint32_t nvc  = std::min(nv, d.n_top);
            float * grow = b.gather_mask + (size_t) i*d.n_sel;
            std::fill(grow,                          grow + (size_t) nvc*kpool,     0.0f);
            std::fill(grow + (size_t) nvc*kpool,     grow + (size_t) d.n_top*kpool, -INFINITY);
        }
    }

    // the incomplete tail: the newest (pos - pos_min + 1) % kpool tokens, which belong to no
    // complete pool yet and are appended to every token's selection unconditionally
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = toks[i].seqs[0];
        const llama_pos    p = toks[i].pos;
        const auto & sq = st.seqs[s];

        const uint32_t n_tail = sq.cells.empty()
            ? 0u
            : (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(),
                                           std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            b.tail_idxs[(size_t) i*(kpool - 1) + k] = cell;

            if (b.gather_mask && d.n_sel % kpool != 0) {
                b.gather_mask[(size_t) i*d.n_sel + (size_t) d.n_top*kpool + k] =
                    real ? 0.0f : -INFINITY;
            }
        }
    }
}

