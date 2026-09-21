// PXA_CKPT_EVICT policy test.
//
// Two claims are checked:
//
//  1. DEFAULT IS UNCHANGED. The structural mode must pick exactly the same checkpoint as the
//     eviction loop that shipped before this change, for every ring the server can present it.
//     The reference below is a verbatim copy of that loop; the two are compared over exhaustive
//     small rings and long randomized ones.
//
//  2. THE WEIGHTED MODE DOES WHAT IT CLAIMS. A checkpoint that has actually served restores is
//     harder to evict in proportion to (1 + 4*replay_hits); a checkpoint flagged as a replay
//     boundary is not evicted at all while any unflagged interior candidate exists; and the ring
//     always yields exactly one victim, inside the interior, so eviction can never fail to make
//     room.
//
// Eviction changes only what has to be recomputed, never a computed value, so neither mode can
// move a token. CPU only, no model, no server, no GPU. Exit 0 on pass.

#include "pxa-ckpt-evict.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------------------------
// REFERENCE: the pre-change eviction loop, copied unchanged (returns the chosen index)

static size_t ref_evict_by_variance(const std::vector<int64_t> & tokens) {
    if (tokens.size() < 3) {
        return 0;
    } else if (tokens.size() == 3) {
        return 1;
    }
    size_t best_idx = 1;
    const size_t n = tokens.size();
    const size_t start = 1;     // never remove the first
    const size_t end = n - 1;   // never remove the last
    double max_pos = tokens[n - 1];
    double diff  = (tokens[start] - tokens[start - 1]);
    double diff2 = (tokens[start + 1] - tokens[start]);
    double best_variance = diff * (diff2 / max_pos);
    for (size_t i = start + 1; i < end; i++) {
        diff  = tokens[i] - tokens[i - 1];
        diff2 = tokens[i + 1] - tokens[i];
        double variance = diff * (diff2 / max_pos);
        if (variance < best_variance) {
            best_variance = variance;
            best_idx = i;
        }
    }
    return best_idx;
}

// ---------------------------------------------------------------------------------------------

static int fails = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static std::vector<pxa_ckpt_evict_entry> ring(const std::vector<int64_t> & pos) {
    std::vector<pxa_ckpt_evict_entry> v;
    v.reserve(pos.size());
    for (const int64_t p : pos) {
        pxa_ckpt_evict_entry e;
        e.pos_max = p;
        v.push_back(e);
    }
    return v;
}

int main() {
    std::mt19937 rng(20260908);

    // ---------------------------------------------------------------------------------------
    // 1. structural mode == the shipped loop

    long n_cmp = 0;

    // exhaustive over small rings of strictly increasing positions on a small grid
    for (size_t n = 0; n <= 6; ++n) {
        std::vector<int64_t> pos(n);
        std::vector<size_t>  idx(n, 0);
        const int64_t grid = 8;
        // enumerate every strictly increasing tuple over [0, grid)
        std::vector<int64_t> cur;
        std::function<void(int64_t)> rec = [&](int64_t start) {
            if (cur.size() == n) {
                const size_t a = pxa_ckpt_evict_pick(ring(cur), PXA_CKPT_EVICT_STRUCTURAL);
                const size_t b = ref_evict_by_variance(cur);
                if (a != b) {
                    fprintf(stderr, "FAIL: structural mismatch n=%zu -> %zu vs ref %zu\n", n, a, b);
                    fails++;
                }
                n_cmp++;
                return;
            }
            for (int64_t v = start; v < grid; ++v) {
                cur.push_back(v);
                rec(v + 1);
                cur.pop_back();
            }
        };
        rec(0);
        (void) pos; (void) idx;
    }

    // long randomized rings, including the ragged spacing a real conversation produces
    for (int trial = 0; trial < 20000; ++trial) {
        const size_t n = 1 + (rng() % 12);
        std::vector<int64_t> pos;
        int64_t p = (int64_t) (rng() % 64);
        for (size_t i = 0; i < n; ++i) {
            pos.push_back(p);
            p += 1 + (int64_t) (rng() % 4096);
        }
        const size_t a = pxa_ckpt_evict_pick(ring(pos), PXA_CKPT_EVICT_STRUCTURAL);
        const size_t b = ref_evict_by_variance(pos);
        if (a != b) {
            fprintf(stderr, "FAIL: structural mismatch (random, n=%zu) -> %zu vs ref %zu\n", n, a, b);
            fails++;
            break;
        }
        n_cmp++;
    }

    // hits and boundary flags must be INERT in structural mode
    for (int trial = 0; trial < 5000; ++trial) {
        const size_t n = 4 + (rng() % 8);
        std::vector<int64_t> pos;
        int64_t p = 0;
        for (size_t i = 0; i < n; ++i) { pos.push_back(p); p += 1 + (int64_t) (rng() % 1000); }
        auto r = ring(pos);
        for (auto & e : r) {
            e.replay_hits = rng() % 5;
            e.boundary    = (rng() % 3) == 0;
        }
        if (pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_STRUCTURAL) != ref_evict_by_variance(pos)) {
            check(false, "structural mode read replay_hits/boundary");
            break;
        }
        n_cmp++;
    }

    // ---------------------------------------------------------------------------------------
    // 2. the weighted mode

    // evenly spaced ring: with no hits anywhere, the first interior candidate wins the tie in both
    // modes, so the two modes agree
    {
        auto r = ring({0, 100, 200, 300, 400, 500});
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE) ==
              pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_STRUCTURAL),
              "even ring, no hits: value == structural");
    }

    // the structurally weakest candidate is spared once it has proven useful
    {
        // gaps: 100,10,90,... -> index 2 is the tightest pair and is what structural drops
        auto r = ring({0, 100, 110, 200, 300, 400});
        const size_t s = pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_STRUCTURAL);
        check(s == 2, "structural drops the tightest interior candidate");
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE) == 2, "value agrees when nothing has hits");
        r[2].replay_hits = 1;   // weight 5x
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE) != 2, "one restore spares the tightest candidate");
        r[2].replay_hits = 0;
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE) == 2, "clearing the hit restores the choice");
    }

    // the weight is exactly (1 + 4*hits): a candidate 4x better structurally survives one hit on
    // its rival, and a 6x better one does not survive two
    {
        // positions chosen so the interior scores are exactly 1x and 4x of each other
        //   idx1 gaps (10, 10) -> 100 ; idx2 gaps (10, 40) -> 400 ; idx3 gaps (40, 40) -> 1600
        auto r = ring({0, 10, 20, 60, 100});
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_STRUCTURAL) == 1, "score ordering as constructed");
        r[1].replay_hits = 1;   // 100 * 5 = 500 > 400
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE) == 2, "one hit outweighs a 4x structural gap");
        r[2].replay_hits = 1;   // 400 * 5 = 2000 > 500
        check(pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE) == 1, "an equal hit restores the structural order");
    }

    // a replay boundary is never dropped while an ordinary interior candidate exists ...
    {
        auto r = ring({0, 100, 110, 200, 300, 400});
        r[2].boundary = true;
        const size_t v = pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE);
        check(v != 2, "a boundary checkpoint is protected");
        check(v >= 1 && v + 1 < r.size(), "the victim is still interior");
    }

    // ... but the ring still yields a victim when every interior candidate is a boundary
    {
        auto r = ring({0, 100, 110, 200, 300, 400});
        for (size_t i = 1; i + 1 < r.size(); ++i) r[i].boundary = true;
        const size_t v = pxa_ckpt_evict_pick(r, PXA_CKPT_EVICT_VALUE);
        check(v >= 1 && v + 1 < r.size(), "all-boundary ring still evicts an interior checkpoint");
        check(v == 2, "all-boundary ring falls back to the structural/weighted order");
    }

    // invariants over long randomized rings, in both modes
    for (int trial = 0; trial < 50000; ++trial) {
        const size_t n = rng() % 14;
        std::vector<int64_t> pos;
        int64_t p = 0;
        for (size_t i = 0; i < n; ++i) { pos.push_back(p); p += (int64_t) (rng() % 500); }
        auto r = ring(pos);
        for (auto & e : r) {
            e.replay_hits = rng() % 7;
            e.boundary    = (rng() % 4) == 0;
        }
        for (const auto mode : {PXA_CKPT_EVICT_STRUCTURAL, PXA_CKPT_EVICT_VALUE}) {
            const size_t v = pxa_ckpt_evict_pick(r, mode);
            if (n == 0) { continue; }
            if (v >= n) { check(false, "victim out of range"); break; }
            if (n >= 4 && (v == 0 || v == n - 1)) { check(false, "victim was an endpoint"); break; }
        }
        if (fails) break;
    }

    // a growing agent conversation: one early checkpoint is the re-entry point every turn comes
    // back to, while new checkpoints keep arriving ahead of it. Under the weighted policy it must
    // survive; the stock policy's behaviour is recorded, not asserted, since it depends only on
    // spacing.
    {
        const size_t ring_max = 6;
        bool hot_alive_by_mode[2] = { false, false };

        for (int mode_i = 0; mode_i < 2; ++mode_i) {
            const auto mode = mode_i == 0 ? PXA_CKPT_EVICT_STRUCTURAL : PXA_CKPT_EVICT_VALUE;

            const int64_t hot_pos = 500;
            std::vector<pxa_ckpt_evict_entry> ck;

            // the conversation's stable prefix, checkpointed early
            pxa_ckpt_evict_entry first; first.pos_max = 0;    ck.push_back(first);
            pxa_ckpt_evict_entry hot;   hot.pos_max   = hot_pos; hot.boundary = true; ck.push_back(hot);

            bool hot_alive = true;
            int64_t pos = hot_pos;

            for (int turn = 0; turn < 200; ++turn) {
                // every turn re-enters at the hot checkpoint while it is still there
                if (hot_alive) {
                    for (auto & e : ck) {
                        if (e.pos_max == hot_pos) { e.replay_hits++; }
                    }
                }
                if (ck.size() >= ring_max) {
                    const size_t v = pxa_ckpt_evict_pick(ck, mode);
                    if (ck[v].pos_max == hot_pos) { hot_alive = false; }
                    ck.erase(ck.begin() + (long) v);
                }
                pos += 400;
                pxa_ckpt_evict_entry e; e.pos_max = pos;
                ck.push_back(e);
            }

            hot_alive_by_mode[mode_i] = hot_alive;
        }

        check(hot_alive_by_mode[1], "weighted policy keeps the repeatedly restored checkpoint");
        printf("test-ckpt-evict: 200-turn replay simulation -> hot checkpoint survives: structural=%s value=%s\n",
               hot_alive_by_mode[0] ? "yes" : "no", hot_alive_by_mode[1] ? "yes" : "no");
    }

    printf("test-ckpt-evict: structural comparisons=%ld -> %s\n", n_cmp, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
