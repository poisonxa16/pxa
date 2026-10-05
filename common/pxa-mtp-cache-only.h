#pragma once

// PXA_MTP_DRAFT_CACHE_ONLY: when may an MTP head's decode build the K/V-only graph?
//
// The MTP draft head has two jobs. Sometimes it is asked to PROPOSE - produce a hidden row and a
// logit so the next draft token can be sampled - and sometimes it is only asked to CATCH UP: to
// advance its own K/V cache over tokens the target has already verified, so that the next real
// draft attends to a complete cache. The second job needs the K/V projection, the rotation and the
// cache write, and nothing else. Everything downstream of that - the query projection, the
// attention scores and softmax, the output projection, the FFN and the (150k-row) LM head - is
// computed and then thrown away, for every token of the catch-up.
//
// The graph builders can therefore emit a reduced graph, but ONLY when the caller genuinely wants
// no output. Taking that path when an output IS wanted would return a garbage logit and change the
// generated text, so the decision is isolated here as one pure predicate, exhaustively tested
// (tests/test-mtp-cache-only.cpp), rather than spread across call sites.
//
// The four conditions, all required:
//
//   1. the lever is on (PXA_MTP_DRAFT_CACHE_ONLY=1; default OFF, so nothing changes until measured);
//   2. the caller asked for ZERO outputs - not "few", zero. This is the load-bearing one;
//   3. the head is a SINGLE grafted layer (nextn_predict_layers == 1). A multi-layer or
//      shared-memory draft head can carry state between its layers that later layers read back, so
//      "only K/V matters" is not established for it;
//   4. there is at least one token to store.
//
// When all four hold, the accepted text cannot change: the cache-store arithmetic (projection +
// rotation + write) is exactly what the full graph computes for K/V, and only unused downstream
// work is dropped. The draft still proposes and the target still verifies every proposal.

#include <cstdlib>

// Reads the lever once.
inline bool pxa_mtp_cache_only_enabled() {
    static const bool on = getenv("PXA_MTP_DRAFT_CACHE_ONLY") &&
                           atoi(getenv("PXA_MTP_DRAFT_CACHE_ONLY")) == 1;
    return on;
}

struct pxa_mtp_cache_only_state {
    bool enabled              = false; // the lever
    bool wants_output         = true;  // does the caller read a logit or hidden row back?
    int  n_layer_nextn        = 0;     // grafted MTP/NextN layers in this model
    int  n_tokens             = 0;     // tokens in the refresh batch
};

// True when the decode may be built as a K/V-only refresh.
inline bool pxa_mtp_cache_only_ok(const pxa_mtp_cache_only_state & st) {
    if (!st.enabled) {
        return false;
    }
    if (st.wants_output) {
        return false;
    }
    if (st.n_layer_nextn != 1) {
        return false;
    }
    if (st.n_tokens <= 0) {
        return false;
    }
    return true;
}
