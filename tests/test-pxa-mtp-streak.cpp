// PXA_MTP_STREAK (common/pxa-mtp-streak.h, Overdrive 2.0 B8): parse, depth cap, full flag, cumulative stop. Pure, no model.
#include "pxa-mtp-streak.h"

#include <cstdio>
#include <initializer_list>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main() {
    // parse: off forms
    for (const char * e : {(const char *) nullptr, "", "0", "-1", "x", "5", "5:0.25:5", "7:1.5", "7:-0.1", "7:0.25:0"}) {
        CHECK(!pxa_mtp_streak_parse(e).on);
    }
    // parse: defaults and explicit fields
    {
        const auto c = pxa_mtp_streak_parse("7");
        CHECK(c.on); CHECK(c.cap == 7); CHECK(c.floor == 0.25f); CHECK(c.deep == 5);
        const auto d = pxa_mtp_streak_parse("6:0.3:4");
        CHECK(d.on); CHECK(d.cap == 6); CHECK(d.floor == 0.3f); CHECK(d.deep == 4);
    }
    const auto c = pxa_mtp_streak_parse("7:0.25:5");
    const pxa_mtp_streak_cfg off;

    // depth: off or at/below D -> untouched (the PXA_MTP_DEEP chain); above D -> D unless the last cycle was full
    CHECK(pxa_mtp_streak_depth(off, 7, true) == 7);
    CHECK(pxa_mtp_streak_depth(c, 3, false) == 3);    // EMA below the DEEP threshold: base depth stays
    CHECK(pxa_mtp_streak_depth(c, 5, false) == 5);
    CHECK(pxa_mtp_streak_depth(c, 7, false) == 5);    // no streak: today's deep cap
    CHECK(pxa_mtp_streak_depth(c, 7, true) == 7);     // streak: the extension opens
    CHECK(pxa_mtp_streak_depth(c, 9, true) == 7);     // never past S
    CHECK(pxa_mtp_streak_depth(c, 6, true) == 6);     // never past the configured ceiling

    // full: a chain of >= D drafts kept whole
    CHECK(!pxa_mtp_streak_full(off, 5, 5));
    CHECK(pxa_mtp_streak_full(c, 5, 5));
    CHECK(pxa_mtp_streak_full(c, 7, 7));
    CHECK(!pxa_mtp_streak_full(c, 7, 6));
    CHECK(!pxa_mtp_streak_full(c, 3, 3));             // a short full chain does not open the extension
    CHECK(!pxa_mtp_streak_full(c, 0, 0));

    // stop: never inside the first D rows, past them only under the floor
    CHECK(!pxa_mtp_streak_stop(c, 4, 0.01f));
    CHECK(!pxa_mtp_streak_stop(c, 5, 0.25f));
    CHECK(pxa_mtp_streak_stop(c, 5, 0.2499f));
    CHECK(pxa_mtp_streak_stop(c, 6, 0.10f));
    CHECK(!pxa_mtp_streak_stop(off, 6, 0.0f));

    // a run of cycles: (drafted, accepted, chain probabilities) -> depth the next cycle gets
    {
        bool full = false;
        struct cyc { int depth_in; int accepted; float p; } run[] = {
            {7, 5, 0.9f},   // no streak yet: held at 5, all 5 kept -> full
            {7, 7, 0.95f},  // streak: 7 allowed; 0.95^7 = 0.70 clears 0.25 -> 7 drafted, all kept -> full
            {7, 3, 0.95f},  // streak: 7 drafted, 3 kept -> not full
            {7, 5, 0.9f},   // held at 5 again
        };
        int want_depth[] = {5, 7, 7, 5};
        for (int k = 0; k < 4; ++k) {
            const int depth = pxa_mtp_streak_depth(c, run[k].depth_in, full);
            CHECK(depth == want_depth[k]);
            int drafted = 0; float cum = 1.0f;
            for (int i = 0; i < depth; ++i) {
                cum *= run[k].p;
                if (pxa_mtp_streak_stop(c, i, cum)) break;
                ++drafted;
            }
            const int acc = run[k].accepted < drafted ? run[k].accepted : drafted;
            full = pxa_mtp_streak_full(c, drafted, acc);
        }
        // low-confidence chain on a streak: 0.75^6 = 0.178 < 0.25 -> the 6th row is not kept
        int drafted = 0; float cum = 1.0f;
        for (int i = 0; i < pxa_mtp_streak_depth(c, 7, true); ++i) {
            cum *= 0.75f;
            if (pxa_mtp_streak_stop(c, i, cum)) break;
            ++drafted;
        }
        CHECK(drafted == 5);
    }

    if (g_fail == 0) printf("test-pxa-mtp-streak: OK\n");
    return g_fail == 0 ? 0 : 1;
}
