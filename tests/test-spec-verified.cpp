// Bug #228 (spec-accept-feedback-uses-untruncated-draft): acceptance feedback must be measured against
// the drafted tokens the server VERIFIED, not the stage's untruncated draft. See
// common/pxa-spec-verified.h. Pure arithmetic, no model, no GPU.

#include "pxa-spec-verified.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); ++g_fail; } } while (0)

int main() {
    const size_t ALL = SIZE_MAX;

    // the report's case: slot A drafts 4, the equalizer cuts it to 2, 1 is accepted -> 0.5, not 0.25
    CHECK(std::fabs(pxa_spec_accept_ratio(1, 4, 2) - 0.5) < 1e-12, "4 drafted / 2 verified / 1 accepted is 0.5");
    CHECK((double) 1 / 4 < 0.5, "the old ratio fell under ngram_mod's 0.5 reset threshold (defect witness)");

    // untruncated verify: unchanged behaviour
    CHECK(std::fabs(pxa_spec_accept_ratio(1, 4, ALL) - 0.25) < 1e-12, "no truncation keeps n/drafted");
    CHECK(std::fabs(pxa_spec_accept_ratio(3, 4, 4) - 0.75) < 1e-12, "verified == drafted keeps n/drafted");
    CHECK(pxa_spec_n_tested(4, ALL) == 4, "unset verified length is the whole draft");
    CHECK(pxa_spec_n_tested(4, 9) == 4, "a verified length above the draft is capped by the draft");

    // nothing tested -> no ratio
    CHECK(pxa_spec_accept_ratio(0, 4, 0) < 0.0, "a verify of zero tokens has no ratio");
    CHECK(pxa_spec_accept_ratio(0, 0, ALL) < 0.0, "an empty draft has no ratio");

    // ngram_map_k: cut to 2, both accepted -> must not record 2 as the value's length
    CHECK(pxa_spec_verify_truncated_all_accepted(4, 2, 2), "cut short and all verified accepted");
    CHECK(!pxa_spec_verify_truncated_all_accepted(4, 2, 1), "a rejection inside the verified part is real");
    CHECK(!pxa_spec_verify_truncated_all_accepted(4, 4, 4), "no truncation");
    CHECK(!pxa_spec_verify_truncated_all_accepted(4, ALL, 2), "unset verified length");

    // exhaustive: the ratio is always in [0,1] and never uses the cut-off tokens
    for (size_t d = 0; d <= 8; ++d) {
        for (size_t v = 0; v <= 9; ++v) {
            const size_t t = pxa_spec_n_tested(d, v);
            for (size_t a = 0; a <= t; ++a) {
                const double r = pxa_spec_accept_ratio(a, d, v);
                if (t == 0) {
                    CHECK(r < 0.0, "no ratio without a tested token");
                } else {
                    CHECK(r >= 0.0 && r <= 1.0, "ratio in [0,1]");
                }
            }
        }
    }

    if (g_fail) {
        fprintf(stderr, "test-spec-verified: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-spec-verified: OK\n");
    return 0;
}
