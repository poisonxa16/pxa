#pragma once

// Bug #228 (spec-accept-feedback-uses-untruncated-draft): a speculative stage learns from the ratio
// of accepted to drafted tokens (ngram_mod's low-acceptance reset streak, the suffix stage's p_min
// back-off, MTP adaptive K, the autotuner) and ngram_map_k stores the accepted count as the value's
// future draft length. The server may verify FEWER tokens than a stage drafted -- n_draft_max near
// the end of a request, the np>1 uniform-batch equalizer -- and the tokens it cut off were never
// tested. Every ratio has to be taken against the verified length, and a verify that was cut short
// with everything verified accepted says nothing about the rest.
//
// Pure arithmetic, tested by tests/test-spec-verified.cpp.

#include <algorithm>
#include <cstddef>
#include <cstdint>

// n_verified == SIZE_MAX means the caller verified the whole draft
inline size_t pxa_spec_n_tested(size_t n_drafted, size_t n_verified) {
    return std::min(n_drafted, n_verified);
}

// accepted / tested, or a negative value when nothing was tested (no ratio to learn from)
inline double pxa_spec_accept_ratio(size_t n_accepted, size_t n_drafted, size_t n_verified) {
    const size_t n = pxa_spec_n_tested(n_drafted, n_verified);
    return n > 0 ? (double) n_accepted / (double) n : -1.0;
}

// the verify was truncated below the draft and every verified token was accepted: the tokens past the
// cut are unknown, so a stage must not record a "the draft was only this long" verdict from this step
inline bool pxa_spec_verify_truncated_all_accepted(size_t n_drafted, size_t n_verified, size_t n_accepted) {
    return n_verified < n_drafted && n_accepted >= n_verified;
}
