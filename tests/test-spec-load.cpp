// PXA_SPEC_ADAPTIVE_LOAD policy test.
//
// The policy decides only HOW MANY tokens a slot proposes. Acceptance is untouched and stays
// exact-greedy, so no depth this function can return is able to change the accepted text; what is
// tested here is therefore the shape of the depth curve and its invariants, not any output.
//
// Checked:
//   * a solo request is never narrowed (the configured depth is what it was measured with);
//   * depth falls monotonically as concurrency rises, and reaches 0 at the off-at threshold;
//   * the acceptance ceiling is exactly the geometric expected run p/(1-p), and only applies when
//     an acceptance estimate is present;
//   * the result is always 0 or inside [n_min_usable, n_draft_max] -- never an unusable stub, never
//     longer than the caller allowed;
//   * a slot that could not speculate at all (n_draft_max <= 0) is left at 0.
//
// CPU only, no model, no server, no GPU. Exit 0 on pass.

#include "pxa-spec-load.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int fails = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static pxa_spec_load_state mk(int active, int n_max, int n_min = 1, float ema = -1.0f, int off_at = 4) {
    pxa_spec_load_state st;
    st.n_active_slots = active;
    st.n_draft_max    = n_max;
    st.n_min_usable   = n_min;
    st.accept_ema     = ema;
    st.off_at         = off_at;
    return st;
}

int main() {
    // nothing to speculate with stays nothing
    check(pxa_spec_load_depth(mk(1, 0)) == 0, "n_draft_max 0 -> 0");
    check(pxa_spec_load_depth(mk(8, 0)) == 0, "n_draft_max 0 under load -> 0");

    // a solo request is never narrowed, at any acceptance
    for (int n_max = 1; n_max <= 16; ++n_max) {
        for (float ema : {-1.0f, 0.0f, 0.25f, 0.5f, 0.9f, 1.0f}) {
            if (pxa_spec_load_depth(mk(1, n_max, 1, ema)) != n_max) {
                check(false, "solo request narrowed");
                break;
            }
        }
    }
    // 0 active slots cannot happen (this slot is decoding) but must behave as 1
    check(pxa_spec_load_depth(mk(0, 4)) == 4, "n_active 0 treated as solo");

    // the concurrency ceiling is ceil(n_max / S), and 0 at off_at
    check(pxa_spec_load_depth(mk(2, 4)) == 2, "S=2, n_max=4 -> 2");
    check(pxa_spec_load_depth(mk(3, 4)) == 2, "S=3, n_max=4 -> ceil(4/3)=2");
    check(pxa_spec_load_depth(mk(4, 4)) == 0, "S=4 -> off by default");
    check(pxa_spec_load_depth(mk(9, 4)) == 0, "S=9 -> off");
    check(pxa_spec_load_depth(mk(4, 4, 1, -1.0f, 8)) == 1, "off_at=8 keeps S=4 alive at ceil(4/4)");
    check(pxa_spec_load_depth(mk(8, 8, 1, -1.0f, 8)) == 0, "off_at=8 turns off at S=8");
    check(pxa_spec_load_depth(mk(2, 4, 1, -1.0f, 1)) == 0, "off_at below 2 is clamped to 2");

    // depth is monotonically non-increasing in concurrency
    for (int n_max = 1; n_max <= 12; ++n_max) {
        int prev = pxa_spec_load_depth(mk(1, n_max));
        for (int s = 2; s <= 12; ++s) {
            const int d = pxa_spec_load_depth(mk(s, n_max, 1, -1.0f, 12));
            if (d > prev) { check(false, "depth rose with concurrency"); break; }
            prev = d;
        }
    }

    // the acceptance ceiling is exactly floor(p/(1-p)), clamped to at least 1, and only above solo
    {
        // p = 0.5 -> expected run 1
        check(pxa_spec_load_depth(mk(2, 8, 1, 0.5f, 16)) == 1, "p=0.5 -> run 1");
        // p = 0.75 -> expected run 3, but the concurrency ceiling at S=2 of n_max=8 is 4
        check(pxa_spec_load_depth(mk(2, 8, 1, 0.75f, 16)) == 3, "p=0.75 -> run 3 (tighter than 1/S)");
        // p = 0.9 -> expected run 9, so the concurrency ceiling wins
        check(pxa_spec_load_depth(mk(2, 8, 1, 0.9f, 16)) == 4, "p=0.9 -> 1/S ceiling wins");
        // a hopeless acceptance still leaves a 1-token chain, not a negative one
        check(pxa_spec_load_depth(mk(2, 8, 1, 0.0f, 16)) == 1, "p=0 -> 1, never below");
        // p at/above 1 must not divide by zero
        check(pxa_spec_load_depth(mk(2, 8, 1, 1.0f, 16)) == 4, "p=1 clamped, 1/S ceiling wins");
        // with no estimate the acceptance ceiling does not apply
        check(pxa_spec_load_depth(mk(2, 8, 1, -1.0f, 16)) == 4, "no estimate -> concurrency ceiling only");
    }

    // a chain shorter than the stage can use is reported as "skip", not as a stub
    check(pxa_spec_load_depth(mk(3, 4, 3)) == 0, "depth below n_min_usable -> 0");
    check(pxa_spec_load_depth(mk(2, 4, 2)) == 2, "depth exactly n_min_usable is kept");

    // invariants over the whole reachable input space
    long n = 0;
    std::mt19937 rng(20260908);
    for (int trial = 0; trial < 400000; ++trial) {
        const int   active = (int) (rng() % 17);
        const int   n_max  = (int) (rng() % 33) - 4;
        const int   n_min  = (int) (rng() % 6);
        const int   off_at = (int) (rng() % 12);
        const float ema    = (rng() % 4) == 0 ? -1.0f : (float) (rng() % 1001) / 1000.0f;

        const int d = pxa_spec_load_depth(mk(active, n_max, n_min, ema, off_at));

        if (d != 0) {
            if (d < 1 || d > n_max)      { check(false, "depth outside [1, n_draft_max]"); break; }
            if (d < n_min)               { check(false, "depth below n_min_usable"); break; }
            if (active <= 1 && d != n_max) { check(false, "solo request narrowed"); break; }
        } else if (active <= 1 && n_max > 0 && n_max >= n_min) {
            check(false, "solo request skipped");
            break;
        }
        n++;
    }

    printf("test-spec-load: %ld randomized states checked -> %s\n", n, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
