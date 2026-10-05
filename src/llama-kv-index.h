#pragma once

// PXA_KV_SEQ_RM_INDEX (house): a per-sequence ordered position -> cell index for the unified KV
// cache, so that removing a position RANGE of one sequence costs O(#cells actually removed) instead
// of O(cache.size).
//
// Why it exists. llama_kv_cache_seq_rm() is the rollback primitive of the speculative decode loop:
// every rejected draft tail calls it, and on a long-context seat it walked all `cache.size` cells to
// erase a handful. That scan is pure host time on a decode path our own profiling already shows to
// be host-submit-bound, and it grows with -c while the work it does stays constant.
//
// Shape. `per_seq[s]` is an ordered multimap keyed by position; a range removal is
// [lower_bound(p0), lower_bound(p1)). A multimap (not a map) because nothing in the cache API
// forbids two cells carrying the same (seq, pos) -- llama_kv_cache_seq_cp() onto a non-empty
// destination sequence would produce exactly that -- and silently dropping one of them would leave a
// cell behind that the scan would have freed. Duplicates are therefore represented, not assumed away.
//
// Maintenance discipline. The index is LAZY and FAIL-CLOSED:
//   * `valid == false` means "no index"; every consumer then falls back to the full scan, and the
//     next indexed call rebuilds from the cells themselves. Construction starts invalid.
//   * Cheap, per-cell mutations (slot placement, single-cell frees) are mirrored incrementally.
//   * Every bulk or exotic mutation (seq_cp, seq_keep, seq_add, seq_div, seq_share_prefix, defrag,
//     checkpoint restore, state load, any recurrent-cache path) simply calls invalidate(). Those
//     are all already O(cache.size), so a rebuild adds nothing asymptotically.
//   * A cell is indexed only while `pos >= 0`, which is exactly the set seq_rm can act on
//     (its range is clamped to p0 >= 0).
//   * NONE OF THAT IS TRUSTED. The two bullets above are a whitelist, and a whitelist is only as
//     good as the last person to add a mutation site; llama_kv_cache_seq_share_prefix was added to
//     the file and to neither list, and the index went on answering from contents that no longer
//     described the cache. So every change to a cell now bumps a tag epoch and the index answers
//     only while it has accounted for the current value -- see llama_kv_tag_epoch() below. "Forgot
//     to mirror it" is a rebuild because the counter says so, not because the list is complete.
//
// Guards. Every index entry is re-checked against the cell it points at before that cell is touched
// (position still equal, sequence still present). A disagreement is counted in `n_stale` and the
// entry is dropped without mutating the cell, so an index bug can never erase the wrong row.
// PXA_KV_INDEX_CHECK=1 adds two things on every indexed removal: the index is verified against a
// from-scratch rebuild (llama_kv_index_verify), and -- the one that matters -- the removal is run
// BOTH ways from the same starting state and the OUTCOMES are compared cell by cell
// (llama_kv_index_diff). A disagreement is printed with the seq/p0/p1 that produced it and the first
// cell that differs, and the scan's answer is the one kept.
//
// Levers: PXA_KV_SEQ_RM_INDEX=1 opts INTO the indexed path; PXA_KV_INDEX_CHECK=1 turns on both checks.
//
// DEFAULT CHANGED TO OFF, 2026-09-09, on measurement. Gating the unified head on the V100 pair with
// Qwen3.8-27B PXQ4 found non-finite greedy logits on a default-on path once a server had served
// real traffic (det_gate then smoke then speed, then a fresh capture): 153
// "PXA_SAMPLE_SOFTFAIL_v1: greedy argmax logit is non-finite" lines and 9 truncated slot releases,
// with the first one landing immediately after "slot apply_checkp: erased invalidated context
// checkpoint" and two "kv cache rm [p0, end)" calls at p0=0 then p0=117 -- the invalidate-and-refill
// path this index reimplements. Same cards, same session, same sequence:
//   rc1 release binary          non-finite 0    breaker 0    truncated 0
//   unified head, index ON      non-finite 153  breaker 1+   truncated 9
//   unified head, index OFF     non-finite 0    breaker 0    truncated 0
// The evidence is in the unify gate logs {server-unify-def.log, server-rc1-ctrl.log,
// server-kvidx-off.log}. The index is a speed change, so it does not get to ship on by default
// while it can produce a NaN.
//
// STILL OFF, 2026-09-09, after the repair below. Three defects were found and fixed -- the
// unmirrored seq_share_prefix, an inverted range that walked off the end of the map, and the
// whitelist itself, now replaced by the tag epoch -- and tests/test-kv-seq-rm-index.cpp covers all
// three (its guard_only arm re-introduces the share_prefix defect and passes anyway, on the epoch
// alone). tests/kvindex-live-test.sh reproduces the server-side sequence on CPU and is clean with
// the index on and off. But the run that produced the NaN was on two V100s with a 27B hybrid, and
// nothing on CPU reproduces it, so the flip back to ON belongs to the window that can re-run that
// exact sequence with PXA_KV_INDEX_CHECK=1: clean, with no PXA_KV_INDEX_DIVERGENCE line, is the
// evidence the default needs. A divergence line names the bug outright.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

// PXA_KV_SEQ_RM_INDEX: the tag epoch.
//
// The maintenance discipline above is a WHITELIST -- it is correct exactly as long as every site
// that edits a cell either mirrors the edit or invalidates. A site that does neither is silent: the
// index keeps answering, from contents that no longer describe the cache, and the cells it fails to
// find are cells the scan would have freed. llama_kv_cache_seq_share_prefix was such a site.
//
// So the whitelist is no longer trusted. Every change to a cell's sequence set, and every whole-cell
// copy, bumps this counter (llama_kv_cell does it -- see llama-context.h). The index records the
// value it has accounted for, and answers only while the two agree; anything that edits a cell
// without telling the index moves the counter past the index's own, and the next indexed call
// rebuilds. "Forgot to mirror it" is now a rebuild, not a wrong answer -- which is what the
// paragraph above always claimed and never enforced.
//
// A mirrored operation brackets itself with llama_kv_index_edit (below), which advances the index's
// epoch only if the index was ALREADY in sync when the operation began. Without that, an operation
// that mirrors its own edits would also silently absolve every un-mirrored edit made before it.
// Not atomic, deliberately. The KV cache is edited only from the decode thread -- find_slot, every
// seq_* entry point, defrag and the state save/restore all run inside update_slots, and the HTTP
// threads never touch a cell -- so the counter inherits that serialisation. Making it atomic would
// buy nothing: relaxed atomics give no ordering either, and if a second thread ever did edit cells
// the index would be the smaller of the problems.
inline uint64_t & llama_kv_tag_epoch() {
    static uint64_t epoch = 0;
    return epoch;
}

struct llama_kv_seq_index {
    // per sequence id: ordered (pos -> cell index)
    std::vector<std::multimap<llama_pos, uint32_t>> per_seq;

    bool valid = false; // false => no usable index, rebuild before use

    // the llama_kv_tag_epoch() value this index has accounted for; meaningless while !valid
    uint64_t epoch = 0;

    // diagnostics (host-side only, never affects a computed value)
    uint64_t n_rebuild = 0;
    uint64_t n_indexed = 0; // indexed removals served
    uint64_t n_stale   = 0; // entries that disagreed with their cell (must stay 0)

    std::multimap<llama_pos, uint32_t> & map_for(llama_seq_id s) {
        if ((size_t) s >= per_seq.size()) {
            per_seq.resize((size_t) s + 1);
        }
        return per_seq[(size_t) s];
    }

    void note_add(llama_seq_id s, llama_pos pos, uint32_t i) {
        if (!valid || s < 0 || pos < 0) {
            return;
        }
        auto & m = map_for(s);
        // the cell holds a SET of sequence ids, so adding the same id twice must not add a second
        // entry (a batch is free to list a sequence more than once for one token)
        auto r = m.equal_range(pos);
        for (auto it = r.first; it != r.second; ++it) {
            if (it->second == i) {
                return;
            }
        }
        m.emplace(pos, i);
    }

    void note_erase(llama_seq_id s, llama_pos pos, uint32_t i) {
        if (!valid || s < 0 || pos < 0 || (size_t) s >= per_seq.size()) {
            return;
        }
        auto & m = per_seq[(size_t) s];
        auto r = m.equal_range(pos);
        for (auto it = r.first; it != r.second; ++it) {
            if (it->second == i) {
                m.erase(it);
                return;
            }
        }
    }

    // usable RIGHT NOW: built, and nothing has touched a cell behind its back since
    bool in_sync() const {
        return valid && epoch == llama_kv_tag_epoch();
    }

    // "everything up to this point is accounted for"
    void sync() {
        epoch = llama_kv_tag_epoch();
    }

    void invalidate() {
        valid = false;
        per_seq.clear();
    }

    // the cache is now empty: an empty index is exact whatever happened before, so this may sync
    // unconditionally -- it is the one state that needs no history.
    void clear_all() {
        for (auto & m : per_seq) {
            m.clear();
        }
        valid = true;
        sync();
    }

    size_t size() const {
        size_t n = 0;
        for (const auto & m : per_seq) {
            n += m.size();
        }
        return n;
    }
};

// Bracket for a block that edits cells AND mirrors those edits into the index. If the index was in
// sync when the block began it is in sync when it ends; if it was not, it stays out of sync and the
// next indexed call rebuilds. Declare one at the top of any such block -- never call sync() by hand.
struct llama_kv_index_edit {
    llama_kv_seq_index & idx;
    const bool           was_in_sync;

    explicit llama_kv_index_edit(llama_kv_seq_index & i) : idx(i), was_in_sync(i.in_sync()) {}

    ~llama_kv_index_edit() {
        if (was_in_sync) {
            idx.sync();
        }
    }

    llama_kv_index_edit(const llama_kv_index_edit &) = delete;
    llama_kv_index_edit & operator=(const llama_kv_index_edit &) = delete;
};

// PXA_KV_SEQ_RM_INDEX=1 -> take the indexed path. Unset or 0 (the default since 2026-09-09, see the
// note at the top of this file) -> never take it: the scan stays the only implementation.
inline bool llama_kv_index_enabled() {
    static const bool on = getenv("PXA_KV_SEQ_RM_INDEX") && atoi(getenv("PXA_KV_SEQ_RM_INDEX")) != 0;
    return on;
}

// PXA_KV_INDEX_CHECK=1 -> verify the maintained index against a fresh rebuild on every use.
inline bool llama_kv_index_check() {
    static const bool on = getenv("PXA_KV_INDEX_CHECK") && atoi(getenv("PXA_KV_INDEX_CHECK")) != 0;
    return on;
}

// Build the index from the cells. O(cache.size), same cost as one old-style scan.
// Templated on the cell type so this header does not have to see llama_kv_cell's definition
// (llama-context.h includes this header in order to hold the index by value).
template <typename Cell>
void llama_kv_index_rebuild(llama_kv_seq_index & idx, const std::vector<Cell> & cells) {
    idx.per_seq.clear();
    for (uint32_t i = 0; i < (uint32_t) cells.size(); ++i) {
        const Cell & c = cells[i];
        if (c.pos < 0) {
            continue;
        }
        for (const auto s : c.seqs()) {
            if (s < 0) {
                continue;
            }
            idx.map_for(s).emplace(c.pos, i);
        }
    }
    idx.valid = true;
    idx.sync(); // the index now describes the cells exactly as they are at this instant
    idx.n_rebuild++;
}

// Returns true when the maintained index agrees exactly with a rebuild from the cells.
template <typename Cell>
bool llama_kv_index_verify(const llama_kv_seq_index & idx, const std::vector<Cell> & cells, const char * where) {
    if (!idx.valid) {
        return true;
    }
    llama_kv_seq_index ref;
    ref.valid = true;
    llama_kv_index_rebuild(ref, cells);

    // compare as SETS of (pos, cell): a multimap keeps equal keys in insertion order, which the
    // incremental path and a fresh rebuild have no reason to agree on, and nothing in the removal
    // depends on that order
    auto flatten = [](const std::multimap<llama_pos, uint32_t> & m) {
        std::vector<std::pair<llama_pos, uint32_t>> v(m.begin(), m.end());
        std::sort(v.begin(), v.end());
        return v;
    };

    const std::multimap<llama_pos, uint32_t> empty;
    const size_t n = idx.per_seq.size() > ref.per_seq.size() ? idx.per_seq.size() : ref.per_seq.size();
    bool ok = true;
    for (size_t s = 0; s < n; ++s) {
        const auto & a = s < idx.per_seq.size() ? idx.per_seq[s] : empty;
        const auto & b = s < ref.per_seq.size() ? ref.per_seq[s] : empty;
        if (flatten(a) != flatten(b)) {
            fprintf(stderr, "%s: KV seq index drift at seq %d (index has %zu entries, cells have %zu)\n",
                    where ? where : "llama_kv_index_verify", (int) s, a.size(), b.size());
            ok = false;
        }
    }
    return ok;
}

// PXA_KV_INDEX_CHECK: the shadow-scan differential.
//
// llama_kv_index_verify answers "is the index consistent with the cells". This answers the question
// that actually decides whether the lever is safe: "did the indexed removal and the old scan do the
// SAME THING to the cache". It is given the state the indexed removal produced and the state the
// scan produced from the same starting point, and it names the first place they differ -- with the
// removal that produced it, so the line is enough on its own to identify the bug. Returns true when
// they agree; on a disagreement the caller keeps the SCAN's result, so the check mode is a safety
// net and not only an instrument.
template <typename Cell>
bool llama_kv_index_diff(
        const std::vector<Cell> & got,  uint32_t got_used,  uint32_t got_head,
        const std::vector<Cell> & want, uint32_t want_used, uint32_t want_head,
        llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const char * what = nullptr;

    if (got.size() != want.size()) {
        what = "cell count";
    } else if (got_used != want_used) {
        what = "used";
    } else if (got_head != want_head) {
        what = "head";
    }

    if (what != nullptr) {
        fprintf(stderr, "PXA_KV_INDEX_DIVERGENCE: indexed seq_rm(seq %d, [%d, %d)) disagrees with the scan "
                "on %s: indexed used=%u head=%u, scan used=%u head=%u\n",
                seq_id, p0, p1, what, got_used, got_head, want_used, want_head);
        return false;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        const Cell & a = got[i];
        const Cell & b = want[i];
        if (a.pos == b.pos && a.src == b.src && a.delta == b.delta && a.seqs() == b.seqs()) {
            continue;
        }
        fprintf(stderr, "PXA_KV_INDEX_DIVERGENCE: indexed seq_rm(seq %d, [%d, %d)) disagrees with the scan "
                "at cell %zu: indexed pos=%d src=%d n_seq=%zu, scan pos=%d src=%d n_seq=%zu\n",
                seq_id, p0, p1, i,
                (int) a.pos, (int) a.src, a.n_seq(),
                (int) b.pos, (int) b.src, b.n_seq());
        return false;
    }

    return true;
}

// The indexed equivalent of the whole-cache seq_rm scan.
//
// Preconditions the caller must establish (they are the cases the scan keeps for itself):
//   * idx.in_sync(), seq_id >= 0, the cache is not recurrent, and p0/p1 are already normalised
//     (p0 >= 0, p1 == INT_MAX for "to the end"). An inverted or empty range is handled here rather
//     than assumed away.
//
// Semantics are exactly the scan's: erase seq_id from every cell of that sequence whose position is
// in [p0, p1); free (pos = -1, and src = i when qnext state storage is on) every cell that thereby
// became empty, decrement `used` once per freed cell, and move `head` back to the LOWEST freed cell
// index if that is earlier than the current head. Only the order in which freed cells are visited
// differs, and `head` is derived from the minimum rather than from the first hit, so the resulting
// state is identical.
template <typename Cell>
void llama_kv_index_seq_rm(
        llama_kv_seq_index & idx,
        std::vector<Cell>  & cells,
        llama_seq_id         seq_id,
        llama_pos            p0,
        llama_pos            p1,
        uint32_t           & used,
        uint32_t           & head,
        bool                 has_qnext_state) {
    const uint32_t n_cells = (uint32_t) cells.size();
    uint32_t new_head = n_cells;

    idx.n_indexed++;

    // An empty or INVERTED range removes nothing -- the scan's `pos >= p0 && pos < p1` matches no
    // cell for p1 <= p0. Without this the walk below starts at lower_bound(p0), which is AFTER
    // lower_bound(p1), so the loop can never reach its end iterator and runs off the end of the map.
    // Nothing in llama_kv_cache_seq_rm's contract forbids the caller from computing such a range.
    if (p1 <= p0) {
        return;
    }

    if ((size_t) seq_id >= idx.per_seq.size()) {
        return; // this sequence holds no cell at all
    }

    auto & m = idx.per_seq[(size_t) seq_id];

    auto it        = m.lower_bound(p0);
    const auto end = m.lower_bound(p1);

    while (it != end) {
        const uint32_t i   = it->second;
        const llama_pos pk = it->first;

        if (i >= n_cells) {
            idx.n_stale++;
            it = m.erase(it);
            continue;
        }

        Cell & cell = cells[i];

        // the index is never trusted over the cell it points at
        if (cell.pos != pk || !cell.has_seq_id(seq_id)) {
            idx.n_stale++;
            it = m.erase(it);
            continue;
        }

        cell.erase_seq(seq_id);

        if (cell.is_empty()) {
            if (cell.pos >= 0) {
                used--;
            }
            cell.pos = -1;
            if (has_qnext_state) {
                cell.src = (int32_t) i;
            }
            if (i < new_head) {
                new_head = i;
            }
        }

        it = m.erase(it);
    }

    if (new_head != n_cells && new_head < head) {
        head = new_head;
    }
}
