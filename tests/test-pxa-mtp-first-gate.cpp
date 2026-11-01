// PXA_MTP_FIRST_GATE (2026-10-28, C063): the decision that drops a cycle's FIRST MTP draft
// token (common/pxa-mtp-first-gate.h). Pure function checks:
//   - disarmed, p_min <= 0 (or NaN), or no probability for the row: never drops (today's behaviour);
//   - armed: drops exactly when p < p_min (p == p_min is kept, the same side as the in-loop p_min stop);
//   - sampled drafts: dropped on the reported max q only under PXA_SPEC_EXACT; with the exact rule off
//     (the A/B of the old q(x) gate) a sampled draft is left alone;
//   - the stats bucket: 10 buckets of 0.1, clamped, NaN/0 -> 0, 1.0 -> 9.
#include "pxa-mtp-first-gate.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // disarmed: never drops, whatever the probability
    for (float p : {0.0f, 0.1f, 0.5f, 0.74f, 0.99f}) {
        CHECK(!pxa_mtp_first_gate_drop(false, 0.75f, true, p, false, true));
        CHECK(!pxa_mtp_first_gate_drop(false, 0.75f, true, p, true,  true));
    }
    // armed, p_min off (0, negative, NaN): never drops
    for (float pm : {0.0f, -0.5f, nan}) {
        CHECK(!pxa_mtp_first_gate_drop(true, pm, true, 0.01f, false, true));
    }
    // armed, no probability computed for the row: never drops
    CHECK(!pxa_mtp_first_gate_drop(true, 0.75f, false, 0.0f, false, true));
    // armed, argmax draft: drops iff p < p_min
    CHECK( pxa_mtp_first_gate_drop(true, 0.75f, true, 0.7499f, false, true));
    CHECK(!pxa_mtp_first_gate_drop(true, 0.75f, true, 0.75f,   false, true));
    CHECK(!pxa_mtp_first_gate_drop(true, 0.75f, true, 0.9f,    false, true));
    CHECK( pxa_mtp_first_gate_drop(true, 0.75f, true, 0.0f,    false, true));
    CHECK( pxa_mtp_first_gate_drop(true, 1.01f, true, 1.0f,    false, true));   // "always gate" calibration arm
    // argmax draft: the exact flag does not matter
    CHECK( pxa_mtp_first_gate_drop(true, 0.75f, true, 0.5f, false, false));
    // sampled draft: exact rule on -> gated on max q; off -> left alone
    CHECK( pxa_mtp_first_gate_drop(true, 0.75f, true, 0.5f, true, true));
    CHECK(!pxa_mtp_first_gate_drop(true, 0.75f, true, 0.5f, true, false));
    CHECK(!pxa_mtp_first_gate_drop(true, 0.75f, true, 0.9f, true, true));

    // buckets
    CHECK(pxa_mtp_first_gate_bucket(0.0f)   == 0);
    CHECK(pxa_mtp_first_gate_bucket(nan)    == 0);
    CHECK(pxa_mtp_first_gate_bucket(-1.0f)  == 0);
    CHECK(pxa_mtp_first_gate_bucket(0.05f)  == 0);
    CHECK(pxa_mtp_first_gate_bucket(0.1f)   == 1);
    CHECK(pxa_mtp_first_gate_bucket(0.75f)  == 7);
    CHECK(pxa_mtp_first_gate_bucket(0.999f) == 9);
    CHECK(pxa_mtp_first_gate_bucket(1.0f)   == 9);
    CHECK(pxa_mtp_first_gate_bucket(5.0f)   == 9);

    if (g_fail) {
        std::fprintf(stderr, "test-pxa-mtp-first-gate: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("test-pxa-mtp-first-gate: OK\n");
    return 0;
}
