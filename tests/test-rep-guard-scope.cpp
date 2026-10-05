// PXA_REP_GUARD scope (src/pxa-rep-guard-scope.h): which weight mixes arm the repetition guard.
// Header-only, no model, no card. Byte counts below are the Flash-Next 32 GB file's layout in
// miniature (gate/up experts PXQN1, down experts PXQN2) plus the files that must NOT arm.
#include "pxa-rep-guard-scope.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { g_fail++; printf("FAIL  %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
    else { printf("ok    "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

int main() {
    const uint64_t MB = 1ull << 20;
    {   // empty / no PXQ at all
        pxa_rep_guard_scope s;
        s.add(GGML_TYPE_F16, 100 * MB, false); s.add(GGML_TYPE_Q4_K, 900 * MB, false);
        CHECK(!s.eligible(), "k-quant file: guard stays off");
    }
    {   // classic PXQ1: any tensor of any size (the 2026-07-23 rule, unchanged)
        pxa_rep_guard_scope s;
        s.add(GGML_TYPE_PXQ4, 5000 * MB, true); s.add(GGML_TYPE_PXQ1, 1 * MB, true);
        CHECK(s.eligible(), "one tiny classic PXQ1 tensor in a PXQ4 file: guard arms (unchanged rule)");
    }
    {   // the file that loops: the Flash-Next 32 GB file's real per-type bytes (encode log, MB)
        pxa_rep_guard_scope s;
        s.add(GGML_TYPE_PXQN1,   11730 * MB, true);    // gate_exps + up_exps
        s.add(GGML_TYPE_PXQN2,   13820 * MB, true);    // down_exps
        s.add(GGML_TYPE_PXQN4,    1560 * MB, true);
        s.add(GGML_TYPE_PXQN4S8,   190 * MB, true);
        s.add(GGML_TYPE_PXQN5,     690 * MB, true);
        s.add(GGML_TYPE_Q6_K,      520 * MB, false);
        s.add(GGML_TYPE_Q8_0,    54410 * MB, false);   // per-layer embedding table: 54 GB of q8_0, host side
        CHECK(s.eligible() && s.pxqn1_eligible(), "Flash-Next 32 GB layout (PXQN1 %.0f%% of the PXQ-family bytes, 14%% of the 83 GB file): guard arms", 100.0 * s.pxqn1_share());
    }
    {   // PXQN2-heavy MoE (the 48 GB layout): no PXQN1 at all
        pxa_rep_guard_scope s;
        s.add(GGML_TYPE_PXQN2, 30000 * MB, true); s.add(GGML_TYPE_PXQN4, 8000 * MB, true); s.add(GGML_TYPE_Q8_0, 1000 * MB, false);
        CHECK(!s.eligible(), "PXQN2-heavy file without PXQN1: guard stays off (relmse 0.11, output error 0.037)");
    }
    {   // a stray PXQN1 tensor in a high-tier file
        pxa_rep_guard_scope s;
        s.add(GGML_TYPE_PXQN4, 20000 * MB, true); s.add(GGML_TYPE_PXQN1, 200 * MB, true);
        CHECK(!s.eligible(), "PXQN1 at %.1f%% of the bytes: below the 5%% share, guard stays off", 100.0 * s.pxqn1_share());
    }
    {   // exactly at the threshold arms; just under does not
        pxa_rep_guard_scope a; a.add(GGML_TYPE_PXQN4, 9500 * MB, true); a.add(GGML_TYPE_PXQN1, 500 * MB, true);
        CHECK(a.eligible(), "PXQN1 at exactly 5%%: arms");
        pxa_rep_guard_scope b; b.add(GGML_TYPE_PXQN4, 9501 * MB, true); b.add(GGML_TYPE_PXQN1, 499 * MB, true);
        CHECK(!b.eligible(), "PXQN1 just under 5%%: stays off");
    }
    {   // the other tiers never count
        pxa_rep_guard_scope s;
        s.add(GGML_TYPE_PXQ2, 1000 * MB, true); s.add(GGML_TYPE_PXQ3, 1000 * MB, true); s.add(GGML_TYPE_PXQ4, 1000 * MB, true);
        s.add(GGML_TYPE_PXQ6, 1000 * MB, true); s.add(GGML_TYPE_PXQN3, 1000 * MB, true); s.add(GGML_TYPE_PXQN4, 1000 * MB, true);
        s.add(GGML_TYPE_PXQN5, 1000 * MB, true); s.add(GGML_TYPE_PXQN2, 1000 * MB, true);
        CHECK(!s.eligible(), "PXQ2/3/4/6 + PXQN2/3/4/5 only: guard stays off");
    }
    printf("%s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
