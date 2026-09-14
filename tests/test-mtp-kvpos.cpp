// PXA_MTP_KVPOS_v1 + PXA_MTP_ZERO_OUTPUT_COMMIT: the MTP head's K/V position policy, on the CPU, with no
// model, no weights, no context and no GPU.
//
// The head's cache has three writers -- the prompt warm-up, the draft loop and the accepted-token
// commit -- plus one eraser, the draft-region purge. They only work if all four agree on ONE
// convention, and until 2026-09-09 they did not: the commit wrote every row a position low. The
// arithmetic they now share lives in common/pxa-mtp-kvpos.h, and this test replays the real control
// flow of common/speculative.cpp over it:
//
//   warm_up(N)                        mtp_update_kv_cache(..., is_prompt_warmup = true)
//   draft(n_past, cached, K)          mtp_speculative_gen_draft() + its purge
//   commit(pos_base, n_ids)           common_speculative_apply_hidden_rows() / ..._batched()
//
// Every cell carries the identity of the pair it should hold -- the target hidden produced after
// some token and the token that follows it -- so the invariant can be checked directly:
//
//   INVARIANT   cell at position q holds (h_{q-1}, x_q), the cache is contiguous from 0, and it
//               ends at exactly the position of the newest token the target has produced.
//
// The expectations are computed from the token stream itself, never read back out of the helpers
// under test. The last case replays the SAME scenario with the pre-fix arithmetic and requires the
// invariant to break, so a future regression cannot pass by making the test vacuous.
//
// A separate case covers the acceptance rule the speed work rides on: at temperature 0 the verify
// is exact-match, so the emitted token stream must equal the no-speculation stream whatever the
// drafter proposes and however deep it proposes it. (This is the POLICY, not the arithmetic of the
// verify GEMM -- a batch-shape near-tie flip in the target's own numerics is a separate, measured,
// pre-existing property; measured 2026-09-08)

#include "pxa-mtp-kvpos.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

static int          g_fail = 0;
static const char * g_case = "";

static void begin(const char * name) { g_case = name; printf("  %s\n", name); fflush(stdout); }

static void fail(const char * fmt, ...) {
    fprintf(stderr, "  FAIL [%s]: ", g_case);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    ++g_fail;
}

#define CHECK(cond, ...) do { if (!(cond)) { fail(__VA_ARGS__); } } while (0)

// ---------------------------------------------------------------------------------------------
// A model of the MTP head's K/V cache.
//
// A cell records WHICH pair was written into it: `h` is the position of the token whose target
// hidden state conditioned the row (so the pair is (h_{h}, x_{tok})), and `tok` is the position of
// the token whose embedding it carries. Positions stand in for the tensors; that is exactly what
// the bug was about.
// ---------------------------------------------------------------------------------------------
struct cell {
    int32_t h   = -2;   // position of the token whose target hidden conditioned this row
    int32_t tok = -2;   // position of the token this row carries
};

struct mtp_cache {
    std::map<int32_t, cell> cells;

    // A real K/V cache is a flat ring of cells, not a map: writing a row for a logical position
    // that is still occupied leaves TWO cells claiming it (the hazard the draft purge exists to
    // prevent -- "cache state corruption where two cells map to the same logical position").
    // The map cannot represent that, so the collision is counted instead.
    int                  dup_stores = 0;
    std::vector<int32_t> dup_pos;

    void store(int32_t pos, int32_t h_pos, int32_t tok_pos) {
        if (cells.count(pos)) {
            ++dup_stores;
            dup_pos.push_back(pos);
        }
        cells[pos] = cell{h_pos, tok_pos};
    }

    // llama_kv_cache_seq_rm(ctx, seq, p0, -1)
    void rm_from(int32_t p0) {
        for (auto it = cells.begin(); it != cells.end(); ) {
            it = (it->first >= p0) ? cells.erase(it) : std::next(it);
        }
    }

    // llama_kv_cache_seq_rm(ctx, seq, p0, p1)
    void rm_range(int32_t p0, int32_t p1) {
        for (auto it = cells.begin(); it != cells.end(); ) {
            it = (it->first >= p0 && it->first < p1) ? cells.erase(it) : std::next(it);
        }
    }

    int32_t pos_max() const { return cells.empty() ? -1 : cells.rbegin()->first; }
};

// INVARIANT: cell at q holds (h_{q-1}, x_q); the cache is contiguous 0..pos_end and ends there.
static void check_invariant(const mtp_cache & c, int32_t pos_end, const char * where) {
    CHECK(c.pos_max() == pos_end, "%s: cache ends at %d, expected %d", where, c.pos_max(), pos_end);
    CHECK(c.dup_stores == 0, "%s: %d write(s) landed on an already-occupied logical position "
                             "(first at %d)", where, c.dup_stores,
          c.dup_pos.empty() ? -1 : c.dup_pos.front());

    int32_t expect = 0;
    for (const auto & kv : c.cells) {
        CHECK(kv.first == expect, "%s: hole/duplicate in the cache at %d (expected %d)",
              where, kv.first, expect);
        ++expect;

        CHECK(kv.second.tok == kv.first,
              "%s: cell %d carries the token from position %d (must carry its own)",
              where, kv.first, kv.second.tok);
        CHECK(kv.second.h == kv.first - 1,
              "%s: cell %d is conditioned on the hidden after position %d (must be %d)",
              where, kv.first, kv.second.h, kv.first - 1);
    }
    CHECK(expect - 1 == pos_end, "%s: cache holds %d cells, expected %d",
          where, expect, pos_end + 1);
}

// ---------------------------------------------------------------------------------------------
// The engine's three writers, replayed. `legacy` selects the pre-fix arithmetic so the last case
// can prove the invariant actually catches it.
// ---------------------------------------------------------------------------------------------
struct engine {
    mtp_cache c;
    bool      legacy = false;

    // mtp_update_kv_cache(..., is_prompt_warmup = true): the prompt batch at its true positions,
    // with the hidden rows shifted right by one (upstream f5e5753c #1987).
    void warm_up(int32_t n_prompt) {
        for (int32_t q = 0; q < n_prompt; ++q) {
            c.store(q, q - 1, q);
        }
    }

    // mtp_speculative_gen_draft(): n_past is the position of the token being drafted from.
    // `cached` is i0 == 1 (the commit left a free token behind), `k` the proposed chain length.
    // Returns the K/V positions of the rows the draft loop wrote for the PROPOSED tokens, indexed
    // by draft slot: draft slot j is the token the verify batch places at n_past + 1 + j.
    std::vector<int32_t> draft(int32_t n_past, bool cached, int32_t k) {
        std::vector<int32_t> draft_row_pos(k, -1);

        const int32_t i0 = cached ? 1 : 0;

        // The fix's pre-loop cleanup: without a cached token this call is about to (re)write the
        // committed row at n_past, so drop any stale cell there first. (Legacy never needed it,
        // because a commit never stored anything at n_past.)
        if (!legacy && i0 == 0 && c.pos_max() >= n_past) {
            c.rm_from(n_past);
        }

        int32_t n_decode = 0;
        for (int32_t i = i0; i < k; ++i) {
            // row i sits at n_past + i: for i == 0 that is the committed row for x_{n_past}
            // (written here because no commit left one behind), for i >= 1 it is draft slot i-1.
            const int32_t pos = pxa_mtp_draft_pos(n_past, i);
            c.store(pos, pos - 1, pos);
            if (i >= 1) {
                draft_row_pos[i - 1] = pos;
            }
            ++n_decode;
        }

        if (n_decode > 0) {
            const int32_t p0 = legacy ? (n_past + 1 - i0) : pxa_mtp_draft_region_pos0(n_past);
            c.rm_range(p0, n_past + n_decode + 2);
        }

        return draft_row_pos;
    }

    // common_speculative_apply_hidden_rows() / ..._batched(): pos_base is the position of
    // `sampled_before`, hidden row i is h_{pos_base+i}, and ids[i] is the token at pos_base+1+i.
    // Returns the K/V positions the commit wrote, in id order.
    std::vector<int32_t> commit(int32_t pos_base, int32_t n_ids) {
        std::vector<int32_t> commit_row_pos(n_ids, -1);

        // pre-decode cleanup (mtp_update_kv_cache: seq_rm from batch.pos[0])
        const int32_t start = legacy ? pos_base : pxa_mtp_commit_pos0(pos_base);
        if (c.pos_max() >= start) {
            c.rm_from(start);
        }

        for (int32_t i = 0; i < n_ids; ++i) {
            const int32_t pos = legacy ? (pos_base + i) : pxa_mtp_commit_pos(pos_base, i);
            // the pair really carried by this row, whatever position it is written to
            c.store(pos, pos_base + i, pos_base + i + 1);
            commit_row_pos[i] = pos;
        }

        return commit_row_pos;
    }
};

// ---------------------------------------------------------------------------------------------
// One decode cycle of the server: draft from n_past, verify, accept a prefix, commit.
// Returns the new n_past (the position of the next token to be drafted from).
// ---------------------------------------------------------------------------------------------
static int32_t cycle(engine & e, int32_t n_past, bool cached, int32_t k, int32_t n_accept,
                     bool check_positions, const char * where) {
    const std::vector<int32_t> draft_row_pos = e.draft(n_past, cached, k);

    // The commit covers the accepted drafts plus the one token the target produced beyond them.
    const int32_t n_ids = n_accept + 1;
    const std::vector<int32_t> commit_row_pos = e.commit(n_past, n_ids);

    if (check_positions) {
        // THE IDENTITY THE FIX IS ABOUT: an accepted draft token is stored by the draft path and
        // then again by the commit path, and both must place it at its own sequence position.
        for (int32_t i = 0; i < n_accept; ++i) {
            const int32_t tok_pos = n_past + 1 + i;   // where the verify batch put that token
            CHECK(commit_row_pos[i] == tok_pos,
                  "%s: commit stored the token at position %d in cell %d",
                  where, tok_pos, commit_row_pos[i]);
            if (draft_row_pos[i] >= 0) {
                CHECK(draft_row_pos[i] == commit_row_pos[i],
                      "%s: draft wrote token %d into cell %d, commit into cell %d "
                      "(the two writers must agree)",
                      where, tok_pos, draft_row_pos[i], commit_row_pos[i]);
            }
        }
    }

    return n_past + n_ids;   // the new sampled token's position
}

// ---------------------------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------------------------
static void case_warmup_then_first_draft() {
    begin("warm-up then the first draft: the cache ends on the token being drafted from");

    engine e;
    e.warm_up(5);                       // prompt x_0..x_4
    check_invariant(e.c, 4, "after warm-up");

    // The target sampled x_5; the first draft has no cached free token (warm-up invalidates it).
    e.draft(/* n_past = */ 5, /* cached = */ false, /* k = */ 3);
    check_invariant(e.c, 5, "after the first (uncached) draft + purge");
}

static void case_three_deep_after_commit() {
    begin("a 3-deep draft after a commit: both writers place every token at its own position");

    engine e;
    e.warm_up(5);
    int32_t n_past = 5;

    // first cycle: uncached, propose 3, target accepts 2 of them
    n_past = cycle(e, n_past, /* cached = */ false, /* k = */ 3, /* n_accept = */ 2, true,
                   "cycle 1 (uncached, k=3, 2 accepted)");
    check_invariant(e.c, n_past, "after cycle 1");

    // every later cycle starts with the free token the commit sampled
    n_past = cycle(e, n_past, true, 3, 3, true, "cycle 2 (cached, k=3, all accepted)");
    check_invariant(e.c, n_past, "after cycle 2");

    n_past = cycle(e, n_past, true, 3, 0, true, "cycle 3 (cached, k=3, all rejected)");
    check_invariant(e.c, n_past, "after cycle 3");

    n_past = cycle(e, n_past, true, 3, 1, true, "cycle 4 (cached, k=3, 1 accepted)");
    check_invariant(e.c, n_past, "after cycle 4");
}

static void case_depth_and_accept_sweep() {
    begin("every depth 1..6 x every accepted prefix, 40 cycles each");

    for (int32_t k = 1; k <= 6; ++k) {
        for (int32_t acc = 0; acc <= k; ++acc) {
            engine e;
            e.warm_up(7);
            int32_t n_past = 7;
            bool cached = false;

            for (int step = 0; step < 40; ++step) {
                char where[128];
                snprintf(where, sizeof(where), "k=%d accept=%d step=%d", k, acc, step);
                n_past = cycle(e, n_past, cached, k, acc, true, where);
                check_invariant(e.c, n_past, where);
                cached = true;   // a successful commit always leaves the free token behind
            }
        }
    }
}

static void case_uncached_resync_midstream() {
    begin("a mid-stream uncached draft (cleared hidden state) rewrites the committed row cleanly");

    engine e;
    e.warm_up(4);
    int32_t n_past = 4;

    n_past = cycle(e, n_past, false, 2, 2, true, "warm cycle");
    check_invariant(e.c, n_past, "after the warm cycle");

    // The commit left a row at n_past AND a cached free token; now the cached draft is invalidated
    // (context shift / cleared hidden state / a failed commit), so the next draft runs with i0 = 0
    // and writes the row at n_past itself. Without the pre-loop cleanup this is where a second cell
    // for the same logical position would appear.
    e.draft(n_past, /* cached = */ false, /* k = */ 4);
    check_invariant(e.c, n_past, "after the uncached resync draft");
}

// ---------------------------------------------------------------------------------------------
// PXA_MTP_ZERO_OUTPUT_COMMIT (P2): the commit decode asks for no outputs, so it carries no free
// token out and the next draft starts at i0 == 0, re-decoding the last position as its own step 0.
// The K/V must be indistinguishable from the lever-off flow: same cells, same contents.
// ---------------------------------------------------------------------------------------------
static void case_zero_output_commit_writes_exactly_the_verified_rows() {
    begin("zero-output commit: the cells written are the verified positions, and only those");

    engine e;
    e.warm_up(6);
    int32_t n_past = 6;

    for (int step = 0; step < 12; ++step) {
        const int32_t k   = 1 + (step % 4);          // depth 1..4
        const int32_t acc = step % (k + 1);          // 0..k accepted

        // With the lever on there is never a cached free token: every cycle is i0 == 0.
        const std::vector<int32_t> draft_row_pos = e.draft(n_past, /* cached = */ false, k);

        // every draft row this cycle wrote must sit strictly above n_past (the committed row), and
        // the purge must have taken all of them back out
        for (int32_t j = 0; j < k; ++j) {
            if (draft_row_pos[j] >= 0) {
                CHECK(draft_row_pos[j] > n_past,
                      "step %d: a draft row landed at %d, at or below the committed row %d",
                      step, draft_row_pos[j], n_past);
            }
        }
        CHECK(e.c.pos_max() == n_past,
              "step %d: after the purge the cache ends at %d, expected the committed row %d "
              "(a draft row survived)", step, e.c.pos_max(), n_past);

        const int32_t n_ids = acc + 1;
        const std::vector<int32_t> commit_row_pos = e.commit(n_past, n_ids);

        // the commit writes EVERY verified position and nothing else
        for (int32_t i = 0; i < n_ids; ++i) {
            CHECK(commit_row_pos[i] == n_past + 1 + i,
                  "step %d: verified token %d was committed to cell %d, expected %d",
                  step, i, commit_row_pos[i], n_past + 1 + i);
        }

        n_past += n_ids;
        char where[96];
        snprintf(where, sizeof(where), "zero-output step %d (k=%d acc=%d)", step, k, acc);
        check_invariant(e.c, n_past, where);
    }
}

static void case_zero_output_commit_matches_the_free_token_flow() {
    begin("zero-output commit: the resulting cache is identical to the free-carried-token flow");

    // Same token stream, same acceptance pattern, run twice: once with the commit carrying a free
    // token out (lever off -> every later cycle is i0 == 1) and once without it (lever on -> every
    // cycle is i0 == 0 and re-decodes the last position). Only WHICH decode writes the row at
    // n_past differs; the cell contents cannot.
    engine off, on;
    off.warm_up(6);
    on.warm_up(6);

    int32_t n_past_off = 6, n_past_on = 6;
    bool cached_off = false;

    for (int step = 0; step < 12; ++step) {
        const int32_t k   = 1 + (step % 4);
        const int32_t acc = step % (k + 1);

        off.draft(n_past_off, cached_off, k);
        off.commit(n_past_off, acc + 1);
        n_past_off += acc + 1;
        cached_off  = true;

        on.draft(n_past_on, /* cached = */ false, k);
        on.commit(n_past_on, acc + 1);
        n_past_on += acc + 1;
    }

    CHECK(n_past_off == n_past_on, "the two flows emitted a different number of tokens (%d vs %d)",
          n_past_off, n_past_on);
    CHECK(off.c.cells.size() == on.c.cells.size(),
          "lever off holds %zu cells, lever on holds %zu", off.c.cells.size(), on.c.cells.size());

    for (const auto & kv : off.c.cells) {
        auto it = on.c.cells.find(kv.first);
        if (it == on.c.cells.end()) {
            fail("cell %d exists with the lever off and not with it on", kv.first);
            continue;
        }
        CHECK(it->second.h == kv.second.h && it->second.tok == kv.second.tok,
              "cell %d differs: off=(h %d, tok %d) on=(h %d, tok %d)",
              kv.first, kv.second.h, kv.second.tok, it->second.h, it->second.tok);
    }
}

static void case_legacy_arithmetic_breaks_it() {
    begin("the pre-fix arithmetic must FAIL this invariant (the test has teeth)");

    engine e;
    e.legacy = true;
    e.warm_up(5);
    int32_t n_past = 5;

    // Same scenario as case_three_deep_after_commit, minus the position identity checks (they are
    // what we are proving would have fired).
    n_past = cycle(e, n_past, false, 3, 2, false, "legacy cycle 1");
    n_past = cycle(e, n_past, true,  3, 3, false, "legacy cycle 2");

    // Count the violations by hand rather than through check_invariant(), which reports failures.
    int violations = 0;
    for (const auto & kv : e.c.cells) {
        if (kv.second.tok != kv.first || kv.second.h != kv.first - 1) {
            ++violations;
        }
    }
    if (e.c.pos_max() != n_past) {
        ++violations;
    }

    CHECK(violations > 0,
          "the legacy commit arithmetic produced a cache this invariant accepts -- the invariant "
          "is not testing anything");
    printf("    legacy arithmetic: %d violated cell(s)/end position, cache ends at %d, "
           "the newest token is at %d\n", violations, e.c.pos_max(), n_past);
}

// ---------------------------------------------------------------------------------------------
// Greedy losslessness of the acceptance policy: at temperature 0 the target's own token stream must
// come out of the speculative loop unchanged, whatever the drafter proposes.
// ---------------------------------------------------------------------------------------------
static int32_t target_token_at(int32_t pos) {          // the model, as a pure function of position
    uint32_t x = (uint32_t) pos * 2654435761u + 12345u;
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return (int32_t) (x % 50000u);
}

static void case_greedy_losslessness() {
    begin("greedy losslessness: exact-match verification cannot change the temp-0 token stream");

    // the no-speculation reference stream
    std::vector<int32_t> reference;
    for (int32_t pos = 1; pos <= 200; ++pos) {
        reference.push_back(target_token_at(pos));
    }

    // sweep the drafter's depth and its quality, including a drafter that is always wrong and one
    // that is always right
    for (int32_t k = 1; k <= 6; ++k) {
        for (int quality = 0; quality <= 10; ++quality) {
            std::vector<int32_t> emitted;
            int32_t n_past = 0;                 // position of the token we draft from
            uint32_t rng   = 0x9e3779b9u ^ (uint32_t) (k * 977 + quality);

            while ((int32_t) emitted.size() < 200) {
                // propose k tokens for positions n_past+1 .. n_past+k
                std::vector<int32_t> proposal;
                for (int32_t j = 0; j < k; ++j) {
                    rng = rng * 1664525u + 1013904223u;
                    const bool right = (int) (rng >> 16) % 10 < quality;
                    proposal.push_back(right ? target_token_at(n_past + 1 + j)
                                             : target_token_at(n_past + 1 + j) + 1);
                }

                // exact-match verify: accept the longest prefix that equals the target's own tokens
                int32_t n_accept = 0;
                while (n_accept < k && proposal[n_accept] == target_token_at(n_past + 1 + n_accept)) {
                    ++n_accept;
                }

                // ids = the accepted prefix plus the token the target produced beyond it; every one
                // of them is emitted (server-context.cpp emits the whole ids vector)
                for (int32_t i = 0; i <= n_accept; ++i) {
                    emitted.push_back(target_token_at(n_past + 1 + i));
                }
                n_past += n_accept + 1;
            }

            emitted.resize(200);
            bool same = true;
            size_t first_diff = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                if (emitted[i] != reference[i]) { same = false; first_diff = i; break; }
            }
            CHECK(same, "k=%d quality=%d: speculation changed the temp-0 stream at token %zu "
                        "(%d vs %d)", k, quality, first_diff,
                  same ? 0 : emitted[first_diff], same ? 0 : reference[first_diff]);
        }
    }
}

int main() {
    printf("test-mtp-kvpos: MTP K/V position policy (PXA_MTP_KVPOS_v1)\n");

    case_warmup_then_first_draft();
    case_three_deep_after_commit();
    case_depth_and_accept_sweep();
    case_uncached_resync_midstream();
    case_zero_output_commit_writes_exactly_the_verified_rows();
    case_zero_output_commit_matches_the_free_token_flow();
    case_legacy_arithmetic_breaks_it();
    case_greedy_losslessness();

    if (g_fail) {
        fprintf(stderr, "test-mtp-kvpos: %d FAILURE(S)\n", g_fail);
        return 1;
    }
    printf("test-mtp-kvpos: OK\n");
    return 0;
}
