#pragma once
// PXA_MTP_FIRST_GATE (2026-10-28, C063): the MTP draft's FIRST token pays its verify width only
// when the head is confident in it.
//
// mtp_speculative_gen_draft (common/speculative.cpp) always keeps the first draft token -- the "cached free
// token" the commit decode already drew, or row 0 of a fresh chain -- and p_min only ends the chain after
// it. That rests on "a 2-row verify costs about what a 1-row decode costs", which holds when every weight
// is on the card. It does not hold when the routed experts sit on the host: a 2-token verify reads the
// union of both tokens' experts from host RAM, and on GLM-5.3-Flash 2x V100 (31 of 42 MoE layers on the
// host, C063) a verify round costs ~1.73x a plain decode while the head is accepted ~63% of the time, so
// MTP n1 is ~12% SLOWER per token than our own plain decode (10.45 vs 11.96 t/s, out-g63 bracket).
//
// Armed, a first token whose reported probability is under the stage's p_min is dropped: the draft comes
// back empty and the server runs that step as a plain one-row decode ("ignoring small draft"), and the
// next draft re-seeds from the target's hidden row (i0 = 0, the same path a selector plain round takes).
// Greedy output: every emitted token is still the target's argmax (exact-match verify); only the verify
// width of some steps changes, so the bytes can differ at a near-tie the way any spec-vs-plain width
// change does (PXA_SPEC_BATCH_FIDELITY). Sampled drafts: the probability the chain reports is max q
// (PXA_SPEC_SAMPLED_GATE_v1, a function of the head's row before x_0 is drawn), so dropping the whole
// draft on it is exact; with PXA_SPEC_EXACT=0 (A/B of the old gate) sampled drafts are left alone.
// p_min <= 0, or no probability computed for the row, never drops.
//
// Switch: PXA_MTP_FIRST_GATE=1 arms it, =0 (or unset, for now) leaves today's behaviour. The floor is the
// MTP stage's p_min (0.75 stock; --spec-type mtp:...,p_min=X or a per-request speculative.p_min move it).
#include "pxa-env-flag.h"

#include <cstdlib>

// -1 = unset (caller decides the default), 0 = off, 1 = on
inline int pxa_mtp_first_gate_env() {
    static const int v = [] {
        const char * e = std::getenv("PXA_MTP_FIRST_GATE");
        if (e == nullptr) return -1;
        const char * p = e;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0') return -1;
        return pxa_env_value_off(e) ? 0 : 1;
    }();
    return v;
}

// The decision for the first draft token of a cycle. Pure: no model, no env (tested in
// tests/test-pxa-mtp-first-gate.cpp).
//   armed     gate armed for this draft
//   p_min     the MTP stage's floor for this request
//   has_prob  the draw reported a probability for this row (p_min > 0, stats or depth cutoff on)
//   p         the reported probability (argmax: its top-1 prob; sampled: max q)
//   sampled   the token was drawn from q (PXA_SPEC_SAMPLED), not the argmax
//   exact_on  PXA_SPEC_EXACT (the reported probability of a sampled draft is max q)
inline bool pxa_mtp_first_gate_drop(bool armed, float p_min, bool has_prob, float p, bool sampled, bool exact_on) {
    if (!armed || !(p_min > 0.0f) || !has_prob) {
        return false;
    }
    if (sampled && !exact_on) {
        return false;
    }
    return p < p_min;
}

// PXA_MTP_STATS: bucket of a first-token probability, 10 buckets of width 0.1 (1.0 -> bucket 9)
inline int pxa_mtp_first_gate_bucket(float p) {
    if (!(p > 0.0f)) return 0;
    const int b = (int) (p * 10.0f);
    return b < 0 ? 0 : (b > 9 ? 9 : b);
}
