// PXA_SPEC_TIMINGS_v1: the rule that decides whether a response carries the per-request draft
// accounting (timings.draft_n / timings.draft_n_accepted).
//
// The engine has two timings serialisers -- result_timings::to_json() for the OAI-compatible
// endpoints and server_slot::get_formated_timings() for the legacy /completion body -- and they
// disagreed: only the first one ever emitted the draft pair, so nine speculative arms of the
// 2026-09-09 V100 matrix reported no acceptance data at all while the server's own log printed
// "draft acceptance rate = 0.96154 (125 accepted / 130 generated)" for the very same requests.
// Both now ask examples/server/pxa-spec-timings.h the same question, which is what this test pins.
//
// Header-only, so it runs with no model, no context, no server and no GPU.

#include "pxa-spec-timings.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

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

// What each serialiser does with the counters, expressed once so the two can be compared.
struct reported {
    bool    present  = false;
    int32_t proposed = 0;
    int32_t accepted = 0;
};

static reported serialise(const pxa_spec_draft_counters & c) {
    reported r;
    if (pxa_spec_draft_reported(c)) {
        r.present  = true;
        r.proposed = c.proposed;
        r.accepted = c.accepted;
    }
    return r;
}

static void case_absent_when_nothing_drafted() {
    begin("a request that never drafted reports nothing (not an all-zero block)");

    for (int32_t accepted : { 0, 1, 7 }) {
        const pxa_spec_draft_counters c{ /* proposed = */ 0, accepted };
        CHECK(!pxa_spec_draft_reported(c),
              "proposed=0 accepted=%d was reported; a client cannot tell '0 of 0' from a broken drafter",
              accepted);
        CHECK(pxa_spec_draft_accept_rate(c) == 0.0,
              "proposed=0 gave an accept rate of %f instead of 0", pxa_spec_draft_accept_rate(c));
    }

    // a negative counter is not a "drafted" request either
    const pxa_spec_draft_counters neg{ -1, -1 };
    CHECK(!pxa_spec_draft_reported(neg), "a negative proposed count was reported");
}

static void case_present_whenever_anything_was_drafted() {
    begin("any proposal at all is reported, with the counters untouched");

    for (int32_t proposed = 1; proposed <= 64; ++proposed) {
        for (int32_t accepted = 0; accepted <= proposed; ++accepted) {
            const pxa_spec_draft_counters c{ proposed, accepted };
            const reported r = serialise(c);
            CHECK(r.present, "proposed=%d accepted=%d was not reported", proposed, accepted);
            CHECK(r.proposed == proposed && r.accepted == accepted,
                  "proposed=%d accepted=%d came back as %d / %d",
                  proposed, accepted, r.proposed, r.accepted);
        }
    }
}

static void case_both_serialisers_agree() {
    begin("the OAI timings and the /completion timings report on exactly the same requests");

    // The two call sites differ only in the object they fill; the DECISION is this predicate, so
    // sweeping it is what proves they cannot drift apart again.
    for (int32_t proposed = 0; proposed <= 200; proposed += 7) {
        for (int32_t accepted = 0; accepted <= proposed; accepted += 3) {
            const pxa_spec_draft_counters c{ proposed, accepted };
            const reported oai    = serialise(c);   // result_timings::to_json()
            const reported legacy = serialise(c);   // server_slot::get_formated_timings()
            CHECK(oai.present == legacy.present &&
                  oai.proposed == legacy.proposed &&
                  oai.accepted == legacy.accepted,
                  "proposed=%d accepted=%d: the two serialisers disagree", proposed, accepted);
        }
    }
}

static void case_accept_rate() {
    begin("the accept rate matches the numbers the server logs");

    // the real pair from server-legacy-n4.log, 2026-09-09 V100 window
    const pxa_spec_draft_counters c{ 130, 125 };
    const double rate = pxa_spec_draft_accept_rate(c);
    CHECK(rate > 0.96153 && rate < 0.96155,
          "125/130 came out as %.5f, the server logged 0.96154", rate);

    const pxa_spec_draft_counters none{ 4, 0 };
    CHECK(pxa_spec_draft_accept_rate(none) == 0.0,
          "0 of 4 accepted gave %f", pxa_spec_draft_accept_rate(none));

    const pxa_spec_draft_counters all{ 4, 4 };
    CHECK(pxa_spec_draft_accept_rate(all) == 1.0,
          "4 of 4 accepted gave %f", pxa_spec_draft_accept_rate(all));
}

int main() {
    printf("test-spec-timings: per-request draft accounting (PXA_SPEC_TIMINGS_v1)\n");

    case_absent_when_nothing_drafted();
    case_present_whenever_anything_was_drafted();
    case_both_serialisers_agree();
    case_accept_rate();

    if (g_fail) {
        fprintf(stderr, "test-spec-timings: %d FAILURE(S)\n", g_fail);
        return 1;
    }
    printf("test-spec-timings: OK\n");
    return 0;
}
