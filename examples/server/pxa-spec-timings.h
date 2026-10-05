#pragma once

// PXA_SPEC_TIMINGS_v1 (2026-09-09): the per-request draft accounting a client sees.
//
// A slot counts what speculation did for a request in two integers -- how many draft tokens were
// PROPOSED (server_slot::n_draft_total) and how many the target ACCEPTED (n_draft_accepted) -- and
// they are the only acceptance numbers a user can get without rebuilding the engine with the lab
// counters on. START-HERE points people at `timings.draft_n` / `timings.draft_n_accepted`.
//
// They were missing from every `/completion` response, for EVERY stage type (measured on the
// 2026-09-09 V100 matrix: nine arms, MTP drafting on eight of them, `draft_n` absent from all).
// Not because the counters were empty -- the server's own end-of-request log printed
// "draft acceptance rate = 0.96154 (125 accepted / 130 generated)" for the same requests -- but
// because the final result carries TWO timings objects and the legacy `/completion` path emits the
// one that never had the fields:
//
//   * result_timings (server-task.h) -> to_json(): has draft_n / draft_n_accepted, and is what the
//     OAI-compatible endpoints serialise;
//   * server_slot::get_formated_timings() -> a raw json built by hand, stored into res->data
//     ["timings"], and returned verbatim by to_json_non_oaicompat_final(). It had no draft fields,
//     and because nlohmann::json::push_back does not overwrite an existing key, the later
//     `res.push_back({"timings", timings.to_json()})` in the OAI path is a silent no-op for any
//     result whose `data` already carries "timings".
//
// So the rule for reporting them lives here, once, and both serialisers ask it the same question.

#include <cstdint>

struct pxa_spec_draft_counters {
    int32_t proposed = 0;   // draft tokens generated for this request
    int32_t accepted = 0;   // of those, how many the target kept
};

// Report the accounting only when this request actually drafted. A request that never speculated
// (no stage armed, or every step skipped) must not carry an all-zero draft block: a client cannot
// tell "0 of 0" from "speculation is broken", and the fields were originally made conditional for
// exactly that reason.
static inline bool pxa_spec_draft_reported(const pxa_spec_draft_counters & c) {
    return c.proposed > 0;
}

// The acceptance rate a reader would compute from the pair. Defined here so the "0 proposed" case
// has one answer instead of a division by zero at each call site.
static inline double pxa_spec_draft_accept_rate(const pxa_spec_draft_counters & c) {
    return c.proposed > 0 ? (double) c.accepted / (double) c.proposed : 0.0;
}
