#pragma once

// PXA_RESERVE_GUARD run accounting (issue #11, 2026-10-17). Pure, header-only, no ggml.
//
// The thrash guard used to latch the real-path reserve off once a context had re-reserved
// PXA_RESERVE_MAX times over its whole LIFETIME. A long-lived server legitimately re-reserves
// twice per request (the prefill and the decode graph have different node counts), so after about
// a dozen requests the guard latched off for good and every later prefill re-planned and drained
// per ubatch. What the guard is for is a plan that is replaced before it has done any work: two
// shapes taking turns, a reallocation per ubatch, forever.
//
// So the guard now counts a RUN: consecutive re-reserves made before PXA_RESERVE_AMORTIZED_MIN
// ubatches of the context ran on the plan they replace. Every ubatch counts, the ones a verify-width
// or MTP graph slot runs on its own plan included: with speculative decoding and short prompts the
// main plan itself serves only 1-2 ubatches per request, yet the reserve was paid for by the work in
// between (measured 2026-10-17: counting main-plan ubatches only let the run reach 7 in 40 short
// requests, counting all of them kept it at 3). A handoff after that many ubatches is amortized and
// restarts the run at 1. PXA_RESERVE_MAX caps the run; PXA_RESERVE_LIFETIME=1
// restores the old cap on the lifetime count. The lifetime count itself is kept for the logs.

#include <climits>

// a plan replaced after at least this many ubatches of its context ran paid for its reserve
#define PXA_RESERVE_AMORTIZED_MIN 4

struct pxa_reserve_run_state {
    int run    = 0;   // consecutive unamortized re-reserves, the latest one included
    int served = 0;   // ubatches the context ran since the current plan was reserved (graph slots included)
};

// one more ubatch ran on this context (saturating: a server can live for a very long time)
static inline void pxa_reserve_run_note_ubatch(pxa_reserve_run_state & st) {
    if (st.served < INT_MAX) {
        st.served++;
    }
}

// a re-reserve replaced the current plan: returns the new run length and starts counting the new plan
static inline int pxa_reserve_run_note_reserve(pxa_reserve_run_state & st) {
    st.run    = st.served >= PXA_RESERVE_AMORTIZED_MIN ? 1 : (st.run < INT_MAX ? st.run + 1 : INT_MAX);
    st.served = 0;
    return st.run;
}

// does the guard latch after this re-reserve? lifetime_mode = the pre-fix rule on the lifetime count
static inline bool pxa_reserve_run_should_latch(int run, int lifetime_count, int max_reserves, bool lifetime_mode) {
    return lifetime_mode ? lifetime_count > max_reserves : run > max_reserves;
}
