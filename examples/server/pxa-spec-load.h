#pragma once

// PXA_SPEC_ADAPTIVE_LOAD: choose the speculative draft depth from the CURRENT load instead of a
// single static configured depth.
//
// Why. Speculation's payoff and its cost scale differently with concurrency. The payoff is
// latency for ONE request: the verify batch carries `1 + K` rows through one target decode, and on
// a memory-bandwidth-bound decode rows 2..K are nearly free, so a single active request gets most
// of the chain for the price of one step. The cost is per drafted token and per drafting slot: each
// proposed token is a draft-head forward with its own lm_head GEMV, and each drafting slot adds its
// own per-step checkpoint save, hidden-row copies and rollback. With S slots decoding together the
// machine is already saturated by real work -- the free rows are gone, because the verify batch is
// now S times wider -- while the per-slot draft overheads are paid S times per wall-clock step.
// Screening data on comparable hardware shows the pattern plainly: a large per-request win at one
// active request, and an aggregate wash at four. A depth chosen for the solo case is therefore
// left switched on into the region where it no longer pays.
//
// The policy. Two independent ceilings, and the depth is the smaller:
//
//   1. CONCURRENCY. At one active slot the configured depth is used unchanged. Above that the
//      useful depth falls roughly as 1/S, because that is how the share of the machine this slot
//      can still win back scales. At or above `off_at` active slots the depth is zero: the drafting
//      is a pure tax there, not a smaller win.
//
//   2. ACCEPTANCE. Proposing further than the target will follow is pure cost. Under the standard
//      geometric model, if a drafted token is accepted with probability p, the expected number of
//      accepted tokens before the first rejection is p/(1-p) -- 9 at p=0.9, 3 at p=0.75, 1 at
//      p=0.5. Chain length beyond that buys nothing, so it is a ceiling, applied only when the
//      caller actually has an acceptance estimate.
//
// A depth below the stage's own minimum usable length is reported as zero (skip speculation) rather
// than as an unusable stub, matching what the server already does with get_n_draft_max().
//
// This changes only HOW MANY tokens are proposed. Acceptance stays exact-greedy, so the accepted
// text cannot change: the target verifies every proposal it is given, and a proposal not made is a
// token the target produces itself. There is no determinism exposure at any depth.
//
// Lever: PXA_SPEC_ADAPTIVE_LOAD=1 turns the policy on. Default OFF -- the configured depth is used
// exactly as before. (This line read "Default OFF" while the default was briefly ON, 2026-09-10 to
// 2026-09-11; it is accurate again as of the revert. See pxa_spec_load_enabled below for why the
// flip was withdrawn.) PXA_SPEC_ADAPTIVE_LOAD_OFF_AT=<n> moves the concurrency at which the depth
// drops to zero (default 4, minimum 2).
//
// Pure header: no server types, no llama types, no I/O, so the policy is unit-tested directly
// (tests/test-spec-load.cpp).

#include <algorithm>
#include <cstdlib>
#include <cmath>

// The concurrency at which speculation is switched off entirely. Default from the screening
// pattern above (a wash at four concurrent requests); never below 2, since one active slot is the
// case speculation is FOR.
inline int pxa_spec_load_off_at() {
    static const int off_at = [] {
        const char * e = getenv("PXA_SPEC_ADAPTIVE_LOAD_OFF_AT");
        const int    v = e ? atoi(e) : 0;
        return v >= 2 ? v : 4;
    }();
    return off_at;
}

// DEFAULT OFF. It was flipped ON on 2026-09-10 on a +37% four-client number, and that number was
// WITHDRAWN on 2026-09-11 when the same binary, arms, cards and arm order 29 h later did not
// reproduce it. The reason it could be believed at all is recorded here because it is the whole
// lesson: the 4-client arm's own control drifts 16.7% run to run and the server-TTFT arms 20.3%,
// so no delta under ~20% from either arm class is resolvable at n=5 -- and the claim being flipped
// on was exactly one such delta, read as if it were signal.
//
// So the mechanism stays (it is correct, depth-only, and cannot move a token at any temperature --
// acceptance is still exact-greedy, the target verifies every proposal it is given) and the LEVER
// stays one env var away. What is withdrawn is only the evidence that it should be the default.
// PXA_SPEC_ADAPTIVE_LOAD=1 re-arms it.
//
// To re-open this as a default, the harness has to come first: per-rep interleaving of the two arms
// with a co-measured control published beside every delta, and a pre-registered gate that VOIDs the
// run when the control's own drift is as large as the effect claimed. Until such a gate exists and
// passes, OFF is the only defensible default -- and OFF is also what every release before
// 2026-09-10 shipped, so returning to it is a return to a measured state, not a new bet.
inline bool pxa_spec_load_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_SPEC_ADAPTIVE_LOAD");
        return e ? atoi(e) != 0 : false;
    }();
    return on;
}

struct pxa_spec_load_state {
    int   n_active_slots = 1;     // slots decoding this tick (>= 1 while this slot is one of them)
    int   n_draft_max    = 0;     // the depth this slot would otherwise attempt
    int   n_min_usable   = 1;     // below this the server skips speculation for the slot
    float accept_ema     = -1.0f; // recent share of drafted tokens accepted; < 0 = no estimate yet
    int   off_at         = 4;     // concurrency at which the depth becomes 0
};

// Returns the draft depth to attempt: 0 (skip speculation) or a value in [n_min_usable, n_draft_max].
inline int pxa_spec_load_depth(const pxa_spec_load_state & st) {
    if (st.n_draft_max <= 0) {
        return 0;
    }

    const int n_active = st.n_active_slots > 1 ? st.n_active_slots : 1;
    const int off_at   = st.off_at >= 2 ? st.off_at : 2;

    int depth;

    if (n_active <= 1) {
        // one active request: this is the case the configured depth was chosen for, so it is used
        // unchanged -- with the policy on and a single slot, behaviour is identical to it being off
        depth = st.n_draft_max;
    } else if (n_active >= off_at) {
        return 0;
    } else {
        // ceiling 1: the share of the machine one slot can still win back scales as 1/S. Rounded UP
        // so the policy narrows the chain rather than extinguishing it before `off_at`.
        depth = (st.n_draft_max + n_active - 1) / n_active;

        // ceiling 2: never propose further than the target is expected to follow
        if (st.accept_ema >= 0.0f) {
            const float p = st.accept_ema < 0.99f ? st.accept_ema : 0.99f;
            const int   expected_run = (int) std::floor((double) p / (double) (1.0f - p));
            depth = std::min(depth, std::max(1, expected_run));
        }

        depth = std::min(depth, st.n_draft_max);
    }

    // a chain shorter than the stage can use is not a smaller win, it is overhead
    if (depth < st.n_min_usable) {
        return 0;
    }

    return depth;
}
