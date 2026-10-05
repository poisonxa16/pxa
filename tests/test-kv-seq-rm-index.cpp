// PXA_KV_SEQ_RM_INDEX differential test.
//
// llama_kv_cache_seq_rm() used to be an unconditional scan of every cell in the cache. It is now
// served, for the common case (one sequence, non-recurrent cache), by a per-sequence ordered
// position index (src/llama-kv-index.h). This test asserts the two produce BYTE-IDENTICAL cache
// state -- every cell's pos/src/delta and sequence set, plus `used` and `head` -- after long random
// sequences of the cache mutations the engine actually performs: slot placement (both the unified
// allocator's contiguous run and single-cell frees), range removals INCLUDING INVERTED ONES,
// cross-sequence copies, keep-one-sequence, position shifts, defrag-style cell moves, whole-cache
// clears, checkpoint snapshot/restore round-trips, live prefix forks (PXA_SLOT_FORK_v1) and the
// server's state-restore round trip (wipe the sequence, re-place the same positions).
//
// Each modelled operation is a copy of the REAL call site, including its index maintenance or the
// absence of it, so that the test measures what the engine does rather than what the index's
// documented discipline says it should do.
//
// The reference below is a verbatim copy of the scan that llama_kv_cache_seq_rm() still runs
// whenever the indexed path does not apply, so "identical to the old behaviour" is checked against
// the old code itself rather than against a restatement of it.
//
// The index is additionally verified against a from-scratch rebuild after EVERY operation, which is
// what PXA_KV_INDEX_CHECK=1 does at runtime in the server.
//
// CPU only, no model, no GPU. Exit 0 on parity.

#include "llama-context.h"
#include "llama-kv-index.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------
// a minimal stand-in for the parts of llama_kv_cache that seq_rm touches

struct tcache {
    std::vector<llama_kv_cell> cells;
    uint32_t head = 0;
    uint32_t used = 0;
    llama_kv_seq_index idx;

    explicit tcache(uint32_t n) : cells(n) {}

    uint32_t size() const { return (uint32_t) cells.size(); }
};

// The reference cache B is a MODEL of what the scan would have produced, not a second live cache.
// Editing its cells bumps the same process-wide tag epoch the index watches, so every B operation --
// and every read that copies cells, such as taking a snapshot -- is wrapped in one of these, which
// puts the epoch back. Without it the test would be measuring an artefact of running the arm and its
// reference in one process rather than the behaviour of either.
struct epoch_scope {
    uint64_t saved;
    epoch_scope() : saved(llama_kv_tag_epoch()) {}
    ~epoch_scope() { llama_kv_tag_epoch() = saved; }
};

// ---------------------------------------------------------------------------------------------
// REFERENCE: the pre-index scan, copied unchanged from llama_kv_cache_seq_rm()

static bool ref_seq_rm(tcache & c, llama_seq_id seq_id, llama_pos p0, llama_pos p1, bool has_qnext_state) {
    uint32_t new_head = c.size();

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < c.size(); ++i) {
        if (c.cells[i].pos >= p0 && c.cells[i].pos < p1) {
            if (seq_id < 0) {
                c.cells[i].clear_seq();
            } else if (c.cells[i].has_seq_id(seq_id)) {
                c.cells[i].erase_seq(seq_id);
            } else {
                continue;
            }
            if (c.cells[i].is_empty()) {
                // keep count of the number of used cells
                if (c.cells[i].pos >= 0) c.used--;

                c.cells[i].pos = -1;
                if (has_qnext_state) {
                    c.cells[i].src = i;
                }
                if (new_head == c.size()) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != c.size() && new_head < c.head) c.head = new_head;

    return true;
}

// UNDER TEST: the same call routed exactly the way llama_kv_cache_seq_rm() routes it now.
static bool idx_seq_rm(tcache & c, llama_seq_id seq_id, llama_pos p0, llama_pos p1, bool has_qnext_state) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    const bool can_index = seq_id >= 0; // never recurrent here, lever forced on by the test

    if (can_index) {
        if (!c.idx.in_sync()) {
            llama_kv_index_rebuild(c.idx, c.cells);
        }
        llama_kv_index_edit idx_edit(c.idx);
        llama_kv_index_seq_rm(c.idx, c.cells, seq_id, p0, p1, c.used, c.head, has_qnext_state);
        return true;
    }

    ref_seq_rm(c, seq_id, p0, p1, has_qnext_state);
    c.idx.invalidate();
    return true;
}

// ---------------------------------------------------------------------------------------------
// the other cache mutations, mirrored into the index exactly as src/llama.cpp mirrors them

// unified allocator: a contiguous run of n free cells at/after head (llama_kv_cache_find_slot)
static bool op_place(tcache & c, const std::vector<llama_pos> & pos, const std::vector<std::vector<llama_seq_id>> & seqs) {
    const uint32_t n = (uint32_t) pos.size();
    if (n == 0 || n > c.size()) return false;

    uint32_t n_tested = 0;
    while (true) {
        if (c.head + n > c.size()) {
            n_tested += c.size() - c.head;
            c.head = 0;
            if (n_tested >= c.size()) return false;
            continue;
        }
        bool found = true;
        for (uint32_t i = 0; i < n; i++) {
            if (c.cells[c.head + i].pos >= 0) {
                found = false;
                c.head    += i + 1;
                n_tested  += i + 1;
                break;
            }
        }
        if (found) break;
        if (n_tested >= c.size()) return false;
    }

    {
        llama_kv_index_edit idx_edit(c.idx);
        for (uint32_t i = 0; i < n; i++) {
            c.cells[c.head + i].pos = pos[i];
            for (const llama_seq_id s : seqs[i]) {
                c.cells[c.head + i].add_seq(s);
                c.idx.note_add(s, pos[i], c.head + i);
            }
        }
    }
    c.used += n;
    return true;
}

// single-cell free (llama_kv_swa_evict / llama_kv_swa_release_slot)
static void op_free_cell(tcache & c, uint32_t i) {
    auto & cell = c.cells[i];
    if (cell.pos < 0) return;
    llama_kv_index_edit idx_edit(c.idx);
    for (const auto s : cell.seqs()) {
        c.idx.note_erase(s, cell.pos, i);
    }
    cell.clear_seq();
    cell.pos   = -1;
    cell.delta = 0;
    c.used--;
    if (i < c.head) c.head = i;
}

// llama_kv_cache_seq_cp -- bulk, invalidates
static void op_seq_cp(tcache & c, llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    c.idx.invalidate();
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    c.head = 0;
    for (uint32_t i = 0; i < c.size(); ++i) {
        if (c.cells[i].has_seq_id(src) && c.cells[i].pos >= p0 && c.cells[i].pos < p1) {
            c.cells[i].add_seq(dst);
        }
    }
}

// llama_kv_cache_seq_keep -- bulk, invalidates
static void op_seq_keep(tcache & c, llama_seq_id seq_id) {
    c.idx.invalidate();
    uint32_t new_head = c.size();
    for (uint32_t i = 0; i < c.size(); ++i) {
        if (!c.cells[i].has_seq_id(seq_id)) {
            if (c.cells[i].pos >= 0) c.used--;
            c.cells[i].pos = -1;
            c.cells[i].clear_seq();
            if (new_head == c.size()) new_head = i;
        } else {
            c.cells[i].clear_seq();
            c.cells[i].add_seq(seq_id);
        }
    }
    if (new_head != c.size() && new_head < c.head) c.head = new_head;
}

// llama_kv_cache_seq_add -- bulk, invalidates (positions move)
static void op_seq_add(tcache & c, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    c.idx.invalidate();
    uint32_t new_head = c.size();
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    if (p0 == p1) return;
    for (uint32_t i = 0; i < c.size(); ++i) {
        if (c.cells[i].has_seq_id(seq_id) && c.cells[i].pos >= p0 && c.cells[i].pos < p1) {
            c.cells[i].pos   += delta;
            c.cells[i].delta += delta;
            if (c.cells[i].pos < 0) {
                if (!c.cells[i].is_empty()) c.used--;
                c.cells[i].pos = -1;
                c.cells[i].clear_seq();
                if (new_head == c.size()) new_head = i;
            }
        }
    }
    c.head = new_head != c.size() ? new_head : 0;
}

// defrag: compact every live cell toward index 0 -- cell INDICES change
static void op_defrag(tcache & c) {
    c.idx.invalidate();
    std::vector<llama_kv_cell> live;
    live.reserve(c.size());
    for (uint32_t i = 0; i < c.size(); ++i) {
        if (!c.cells[i].is_empty()) live.push_back(c.cells[i]);
    }
    c.cells.assign(c.size(), llama_kv_cell());
    for (uint32_t i = 0; i < (uint32_t) live.size(); ++i) c.cells[i] = live[i];
    c.head = (uint32_t) live.size();
}

static void op_clear(tcache & c) {
    for (uint32_t i = 0; i < c.size(); ++i) {
        c.cells[i].pos = -1;
        c.cells[i].src = (int32_t) i;
        c.cells[i].clear_seq();
    }
    c.head = 0;
    c.used = 0;
    c.idx.clear_all();
}


// llama_kv_cache_seq_share_prefix (PXA_SLOT_FORK_v1) -- copied from the real function, including
// its index maintenance. It erases the destination's own cells in [0, p1) and tags the source's
// cells in [0, p1) with the destination id; it used to do neither mirroring nor invalidation, which
// is what this test caught, and it now invalidates like every other bulk edit in the file.
static void op_share_prefix(tcache & c, llama_seq_id src, llama_seq_id dst, llama_pos p1, bool has_qnext_state,
                            bool tell_the_index) {
    if (src < 0 || dst < 0 || src == dst || p1 <= 0) return;

    // tell_the_index == true is the engine as it now stands. tell_the_index == false is the engine
    // as it stood when this defect shipped, and running the whole fuzz that way is the proof that
    // the tag epoch alone -- with no site-specific fix at all -- keeps the answers right.
    if (tell_the_index) {
        c.idx.invalidate();
    }

    uint32_t new_head = c.size();
    for (uint32_t i = 0; i < c.size(); ++i) {
        auto & cell = c.cells[i];
        if (cell.pos < 0 || cell.pos >= p1)   continue;
        if (!cell.has_seq_id(dst))            continue;
        cell.erase_seq(dst);
        if (cell.is_empty()) {
            c.used--;
            cell.pos = -1;
            if (has_qnext_state) cell.src = (int32_t) i;
            if (new_head == c.size()) new_head = i;
        }
    }
    if (new_head != c.size() && new_head < c.head) c.head = new_head;

    for (uint32_t i = 0; i < c.size(); ++i) {
        auto & cell = c.cells[i];
        if (cell.pos < 0 || cell.pos >= p1) continue;
        if (!cell.has_seq_id(src))          continue;
        cell.add_seq(dst);
    }

    c.head = 0;
}

// llama_state_seq_set_data / read_kv_cache_meta: the server's context-checkpoint restore and the
// speculative rollback both take this path. It wipes the sequence and immediately re-places the
// same positions with llama_kv_cache_find_slot. Under LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY the K/V
// BYTES are not rewritten, so the restore is only correct if the re-placement lands on the same
// physical cells the wipe just freed -- which makes the head that seq_rm leaves behind part of the
// contract, not an implementation detail. Modelled with both halves mirrored, as the engine does.
static bool op_state_restore(tcache & c, llama_seq_id s, bool indexed, bool has_qnext_state,
                             std::vector<uint32_t> & placed_at) {
    std::vector<llama_pos> pos;
    for (uint32_t i = 0; i < c.size(); ++i) {
        if (c.cells[i].pos >= 0 && c.cells[i].has_seq_id(s)) pos.push_back(c.cells[i].pos);
    }
    if (pos.empty()) return false;
    std::sort(pos.begin(), pos.end());

    if (indexed) idx_seq_rm(c, s, 0, -1, has_qnext_state);
    else         ref_seq_rm(c, s, 0, -1, has_qnext_state);

    std::vector<std::vector<llama_seq_id>> seqs(pos.size(), std::vector<llama_seq_id>{s});
    if (!op_place(c, pos, seqs)) return false;
    // op_place leaves head at the first cell of the run it found, which is the physical row the
    // restored K/V is about to be read from.
    placed_at.assign(1, c.head);
    return true;
}

// ---------------------------------------------------------------------------------------------

static bool cells_equal(const tcache & a, const tcache & b, std::string & why) {
    if (a.used != b.used) { why = "used " + std::to_string(a.used) + " vs " + std::to_string(b.used); return false; }
    if (a.head != b.head) { why = "head " + std::to_string(a.head) + " vs " + std::to_string(b.head); return false; }
    for (uint32_t i = 0; i < a.size(); ++i) {
        const auto & x = a.cells[i];
        const auto & y = b.cells[i];
        if (x.pos != y.pos || x.src != y.src || x.delta != y.delta || x.seqs() != y.seqs()) {
            why = "cell " + std::to_string(i) + ": pos " + std::to_string(x.pos) + " vs " + std::to_string(y.pos)
                + ", src " + std::to_string(x.src) + " vs " + std::to_string(y.src)
                + ", nseq " + std::to_string(x.n_seq()) + " vs " + std::to_string(y.n_seq());
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const long  n_steps = argc > 1 ? atol(argv[1]) : 60000;
    const uint32_t n_cells = argc > 2 ? (uint32_t) atol(argv[2]) : 256;
    const int   n_seq   = 6;

    long bad = 0;

    // guard_only == 1 removes llama_kv_cache_seq_share_prefix's invalidate again, leaving ONLY the
    // tag epoch to notice that a path edited cells without telling the index. Both arms must pass:
    // the first says the engine is right, the second says it would still be right if someone added
    // another such path tomorrow and forgot.
    for (int guard_only = 0; guard_only < 2 && bad == 0; ++guard_only)
    for (int qnext = 0; qnext < 2 && bad == 0; ++qnext) {
        const bool has_qnext_state = qnext != 0;

        long n_rm = 0, n_rm_all = 0, n_place = 0, n_free = 0, n_cp = 0, n_keep = 0, n_add = 0, n_defrag = 0, n_clear = 0, n_snap = 0;
        long n_fork = 0, n_restore = 0;

        std::mt19937 rng(0xC0FFEE + qnext);

        tcache A(n_cells); // indexed
        tcache B(n_cells); // reference scan
        op_clear(A);
        { epoch_scope g; op_clear(B); B.idx.invalidate(); } // the reference never uses an index

        // per-sequence next position, so placements look like real prefill/decode
        std::vector<llama_pos> next_pos(n_seq, 0);

        std::vector<llama_kv_cell> snapA, snapB;
        uint32_t snap_headA = 0, snap_usedA = 0, snap_headB = 0, snap_usedB = 0;
        bool have_snap = false;

        for (long step = 0; step < n_steps && bad == 0; ++step) {
            const llama_seq_id s = (llama_seq_id) (rng() % n_seq);
            const uint32_t roll = rng() % 100;
            const char * what = "?";

            if (roll < 28) {
                // place a run of tokens for one sequence (prefill chunk or decode step)
                what = "place";
                const uint32_t n = 1 + (rng() % 8);
                std::vector<llama_pos> pos(n);
                std::vector<std::vector<llama_seq_id>> seqs(n);
                for (uint32_t i = 0; i < n; ++i) {
                    pos[i] = next_pos[s] + (llama_pos) i;
                    seqs[i].push_back(s);
                    if ((rng() % 10) == 0) { // occasionally a shared (multi-sequence) cell
                        seqs[i].push_back((llama_seq_id) (rng() % n_seq));
                    }
                }
                const bool okA = op_place(A, pos, seqs);
                bool okB; { epoch_scope g; okB = op_place(B, pos, seqs); }
                if (okA != okB) { fprintf(stderr, "place disagreed at step %ld\n", step); bad++; break; }
                if (okA) { next_pos[s] += (llama_pos) n; n_place++; }
            } else if (roll < 64) {
                // the hot case: roll back a tail, or trim a prefix, of ONE sequence
                what = "seq_rm";
                llama_pos p0, p1;
                switch (rng() % 5) {
                    case 0: p0 = next_pos[s] - (llama_pos) (rng() % 6); p1 = -1;  break; // rejected draft tail
                    case 1: p0 = 0; p1 = (llama_pos) (rng() % 32); break;                // prefix trim
                    case 2: p0 = (llama_pos) (rng() % 64); p1 = p0 + (llama_pos) (rng() % 16); break;
                    // an INVERTED range. Nothing in llama_kv_cache_seq_rm's contract forbids one --
                    // p0/p1 are whatever the caller computed -- and the scan answers it correctly by
                    // matching no cell at all. The indexed path has to agree.
                    case 3: p0 = 32 + (llama_pos) (rng() % 32); p1 = p0 - (llama_pos) (1 + rng() % 8); break;
                    default: p0 = -1; p1 = -1; break;                                    // whole sequence
                }
                idx_seq_rm(A, s, p0, p1, has_qnext_state);
                { epoch_scope g; ref_seq_rm(B, s, p0, p1, has_qnext_state); }
                n_rm++;
            } else if (roll < 68) {
                // seq_id < 0: every sequence in the range (keeps the scan)
                what = "seq_rm_all";
                const llama_pos p0 = (llama_pos) (rng() % 32);
                const llama_pos p1 = p0 + (llama_pos) (rng() % 32);
                idx_seq_rm(A, -1, p0, p1, has_qnext_state);
                { epoch_scope g; ref_seq_rm(B, -1, p0, p1, has_qnext_state); }
                n_rm_all++;
            } else if (roll < 75) {
                what = "free_cell";
                const uint32_t i = rng() % n_cells;
                op_free_cell(A, i);
                { epoch_scope g; op_free_cell(B, i); }
                n_free++;
            } else if (roll < 79) {
                what = "seq_cp";
                const llama_seq_id d = (llama_seq_id) (rng() % n_seq);
                op_seq_cp(A, s, d, 0, -1);
                { epoch_scope g; op_seq_cp(B, s, d, 0, -1); }
                n_cp++;
            } else if (roll < 82) {
                what = "seq_keep";
                op_seq_keep(A, s);
                { epoch_scope g; op_seq_keep(B, s); }
                n_keep++;
            } else if (roll < 85) {
                what = "seq_add";
                const llama_pos delta = (llama_pos) (rng() % 17) - 8;
                op_seq_add(A, s, 0, -1, delta);
                { epoch_scope g; op_seq_add(B, s, 0, -1, delta); }
                n_add++;
            } else if (roll < 88) {
                what = "defrag";
                op_defrag(A);
                { epoch_scope g; op_defrag(B); }
                n_defrag++;
            } else if (roll < 89) {
                what = "clear";
                op_clear(A);
                { epoch_scope g; op_clear(B); B.idx.invalidate(); }
                for (auto & p : next_pos) p = 0;
                n_clear++;
            } else if (roll < 93) {
                // PXA_SLOT_FORK_v1: a live prefix fork. Metadata only, and -- as the engine writes
                // it -- invisible to the index.
                what = "share_prefix";
                const llama_seq_id d = (llama_seq_id) (rng() % n_seq);
                const llama_pos p1 = (llama_pos) (1 + rng() % 64);
                op_share_prefix(A, s, d, p1, has_qnext_state, !guard_only);
                { epoch_scope g; op_share_prefix(B, s, d, p1, has_qnext_state, !guard_only); }
                n_fork++;
            } else if (roll < 97) {
                // the server's context-checkpoint restore / speculative rollback: wipe the
                // sequence and immediately re-place the same positions.
                what = "state_restore";
                std::vector<uint32_t> hA, hB;
                const bool okA = op_state_restore(A, s, /* indexed = */ true,  has_qnext_state, hA);
                bool okB; { epoch_scope g; okB = op_state_restore(B, s, /* indexed = */ false, has_qnext_state, hB); }
                if (okA != okB || (okA && hA != hB)) {
                    fprintf(stderr, "STATE RESTORE LANDED ELSEWHERE qnext=%d guard_only=%d step=%ld: indexed head %u vs scan head %u\n",
                            qnext, guard_only, step, hA.empty() ? 0u : hA[0], hB.empty() ? 0u : hB[0]);
                    bad++;
                }
                n_restore++;
            } else if (roll < 99) {
                // a snapshot READS the cells (llama_kv_cache::checkpoint_save does the same), so it
                // must not move the epoch: nothing the index describes has changed.
                what = "snapshot";
                epoch_scope g;
                snapA = A.cells; snap_headA = A.head; snap_usedA = A.used;
                snapB = B.cells; snap_headB = B.head; snap_usedB = B.used;
                have_snap = true;
                n_snap++;
            } else if (have_snap) {
                // checkpoint restore: the whole cell vector comes back, so the index is invalidated
                what = "restore";
                A.cells = snapA; A.head = snap_headA; A.used = snap_usedA; A.idx.invalidate();
                { epoch_scope g; B.cells = snapB; B.head = snap_headB; B.used = snap_usedB; }
                n_snap++;
            }

            std::string why;
            if (!cells_equal(A, B, why)) {
                fprintf(stderr, "MISMATCH qnext=%d guard_only=%d step=%ld op=%s: %s\n", qnext, guard_only, step, what, why.c_str());
                bad++;
            }
            // only meaningful while the index claims to be usable: out of sync it is knowingly
            // behind the cells and gets rebuilt before anything reads it (which is the guard_only
            // arm's whole point), so verifying it there would be asserting the wrong thing.
            if (A.idx.in_sync() && !llama_kv_index_verify(A.idx, A.cells, "test")) {
                fprintf(stderr, "INDEX DRIFT qnext=%d guard_only=%d step=%ld op=%s\n", qnext, guard_only, step, what);
                bad++;
            }
            if (A.idx.n_stale != 0) {
                fprintf(stderr, "STALE INDEX ENTRY qnext=%d guard_only=%d step=%ld op=%s (n_stale=%llu)\n",
                        qnext, guard_only, step, what, (unsigned long long) A.idx.n_stale);
                bad++;
            }
        }

        // The epoch guard makes an unaccounted-for edit cost a rebuild. That is the point, but it
        // would also be a silent way to lose the whole optimisation: an index that rebuilds on every
        // removal is O(cache.size) per call, which is the scan it replaced. Hold it to answering
        // most removals from the index it already has.
        if (bad == 0 && A.idx.n_rebuild * 2 > A.idx.n_indexed) {
            fprintf(stderr, "INDEX REBUILT TOO OFTEN qnext=%d guard_only=%d: %llu rebuilds for %llu indexed removals "
                    "-- the index is no longer buying anything\n", qnext, guard_only,
                    (unsigned long long) A.idx.n_rebuild, (unsigned long long) A.idx.n_indexed);
            bad++;
        }

        if (bad == 0) {
            printf("test-kv-seq-rm-index: qnext_state=%d guard_only=%d cells=%u steps=%ld -> "
                   "place=%ld rm=%ld rm_all=%ld free=%ld cp=%ld keep=%ld add=%ld defrag=%ld clear=%ld snap=%ld "
                   "fork=%ld state_restore=%ld "
                   "| indexed_rm=%llu rebuilds=%llu final_index_entries=%zu\n",
                   qnext, guard_only, n_cells, n_steps, n_place, n_rm, n_rm_all, n_free, n_cp, n_keep, n_add, n_defrag, n_clear, n_snap,
                   n_fork, n_restore,
                   (unsigned long long) A.idx.n_indexed, (unsigned long long) A.idx.n_rebuild, A.idx.size());
        }
    }

    printf("test-kv-seq-rm-index: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
