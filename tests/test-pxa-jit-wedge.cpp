// PXA_JIT_FIRST_START_v1 + PXA_JIT_WEDGE_v1: the SASS-vs-JIT verdict and the wedge-limit policy.
// Header-only logic (ggml/src/ggml-cuda/pxa/pxa-jit-sass.h, examples/server/pxa-wedge-policy.h):
// no GPU, no model, no server.
#include "../ggml/src/ggml-cuda/pxa/pxa-jit-sass.h"
#include "../examples/server/pxa-wedge-policy.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

struct fake_env {
    std::map<std::string, std::string> kv;
    const char * operator()(const char * n) const {
        auto it = kv.find(n);
        return it == kv.end() ? nullptr : it->second.c_str();
    }
};

static void test_sass() {
    const char * rel = "60_61_70_75_86_89";            // the release list (minus 80)
    CHECK(pxa_jit_needs_ptx(rel, 600)  == 0);           // P100
    CHECK(pxa_jit_needs_ptx(rel, 610)  == 0);           // P40 / GTX 10
    CHECK(pxa_jit_needs_ptx(rel, 700)  == 0);           // V100
    CHECK(pxa_jit_needs_ptx(rel, 720)  == 0);           // Xavier: sm_70 SASS runs on 7.2
    CHECK(pxa_jit_needs_ptx(rel, 750)  == 0);           // Turing
    CHECK(pxa_jit_needs_ptx(rel, 800)  == 1);           // A100: 8.0 < 8.6, no SASS runs
    CHECK(pxa_jit_needs_ptx(rel, 860)  == 0);           // RTX 30
    CHECK(pxa_jit_needs_ptx(rel, 870)  == 0);           // Orin: 8.6 SASS runs on 8.7
    CHECK(pxa_jit_needs_ptx(rel, 890)  == 0);           // RTX 40
    CHECK(pxa_jit_needs_ptx(rel, 900)  == 1);           // H100
    CHECK(pxa_jit_needs_ptx(rel, 1000) == 1);           // B200
    CHECK(pxa_jit_needs_ptx(rel, 1200) == 1);           // RTX 50
    CHECK(pxa_jit_needs_ptx(rel, 500)  == 1);           // Maxwell (PTX 60 cannot run there either; still "no SASS")

    const char * full = "60_61_70_75_80_86_89";
    CHECK(pxa_jit_needs_ptx(full, 800) == 0);
    CHECK(pxa_jit_needs_ptx("70", 700) == 0);
    CHECK(pxa_jit_needs_ptx("70", 600) == 1);
    CHECK(pxa_jit_needs_ptx("60_61_70", 750) == 0);     // sm_70 SASS runs on Turing 7.5 (same major)
    CHECK(pxa_jit_needs_ptx("60_61", 700) == 1);        // ... but sm_6x SASS never runs on Volta
    CHECK(pxa_jit_needs_ptx("none", 600) == 1);         // a PTX-only ("60-virtual") build: every card JITs
    CHECK(pxa_jit_needs_ptx("none", 0) == -1);
#define PXA_TEST_TOKEN3 none
    CHECK(strcmp(PXA_JIT_STR(PXA_TEST_TOKEN3), "none") == 0);
    CHECK(pxa_jit_needs_ptx("100_120", 1200) == 0);     // three-digit SMs
    CHECK(pxa_jit_needs_ptx("100_120", 1210) == 0);
    CHECK(pxa_jit_needs_ptx("100_120", 1100) == 1);
    CHECK(pxa_jit_needs_ptx("90a", 900) == 0);          // arch-specific: exactly that cc
    CHECK(pxa_jit_needs_ptx("90a", 910) == 1);
    CHECK(pxa_jit_needs_ptx("89_90a_100a", 1000) == 0);
    CHECK(pxa_jit_needs_ptx("89_90a_100a", 1030) == 1);

    CHECK(pxa_jit_needs_ptx("", 900) == -1);            // unknown list ("native", "all")
    CHECK(pxa_jit_needs_ptx(nullptr, 900) == -1);
    CHECK(pxa_jit_needs_ptx(rel, 0) == -1);
    CHECK(pxa_jit_needs_ptx("__", 900) == -1);

    CHECK(!pxa_jit_forced(nullptr));
    CHECK(!pxa_jit_forced(""));
    CHECK(!pxa_jit_forced("0"));
    CHECK(pxa_jit_forced("1"));
    CHECK(pxa_jit_forced("yes"));

    // the CMake token reaches the code as one string
#define PXA_TEST_TOKEN 60_61_70_75_86_89
    CHECK(strcmp(PXA_JIT_STR(PXA_TEST_TOKEN), "60_61_70_75_86_89") == 0);
#define PXA_TEST_TOKEN2 89_90a_100a
    CHECK(strcmp(PXA_JIT_STR(PXA_TEST_TOKEN2), "89_90a_100a") == 0);
}

static void test_policy() {
    // defaults: no JIT -> exactly the old limit, at any decode count
    fake_env e;
    pxa_wedge_policy p = pxa_wedge_policy_from(false, e, 180000);
    CHECK(p.base_ms == 180000 && p.jit_ms == 1800000 && p.jit_decodes == 256 && !p.jit);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0) == 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 10000) == 180000);
    CHECK(pxa_wedge_limit_ms(p, 60000, 0) == 60000);              // /health's own limit untouched
    CHECK(!pxa_wedge_jit_window(p, 0));

    // JIT: stretched for the first 256 decodes, then back to 180 s
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_jit_window(p, 0));
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0)   == 1800000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 255) == 1800000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 256) == 180000);
    CHECK(pxa_wedge_limit_ms(p, 60000, 3)       == 1800000);      // /health says "compiling", not "stalled"
    CHECK(!pxa_wedge_jit_window(p, 256));

    // the RTX 3080 Ti case: a first decode 180-411 s long, JIT on -> not a wedge any more
    {
        const int64_t inflight_ms = 411000;
        CHECK(inflight_ms > p.base_ms);                                   // the old monitor fired here
        CHECK(!(inflight_ms > pxa_wedge_limit_ms(p, p.base_ms, 0)));      // the new one does not
    }

    // the monitor off stays off (0 / negative / "" / junk, exactly the old atoll behaviour)
    e.kv["PXA_WEDGE_EXIT_MS"] = "0";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0) == 0);
    e.kv["PXA_WEDGE_EXIT_MS"] = "";
    CHECK(pxa_wedge_policy_from(true, e, 180000).base_ms == 0);
    e.kv["PXA_WEDGE_EXIT_MS"] = "abc";
    CHECK(pxa_wedge_policy_from(true, e, 180000).base_ms == 0);
    e.kv["PXA_WEDGE_EXIT_MS"] = "-5";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0) == -5);

    // a base above the JIT limit is never lowered
    e.kv["PXA_WEDGE_EXIT_MS"] = "3600000";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0) == 3600000);

    // off switch: PXA_WEDGE_JIT_MS=0 -> the old limit even on a JIT card
    e.kv.clear();
    e.kv["PXA_WEDGE_JIT_MS"] = "0";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0) == 180000);
    CHECK(!pxa_wedge_jit_window(p, 0));

    // window length 0 = no window; a custom length and limit
    e.kv.clear();
    e.kv["PXA_WEDGE_JIT_DECODES"] = "0";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 0) == 180000);
    e.kv["PXA_WEDGE_JIT_DECODES"] = "4";
    e.kv["PXA_WEDGE_JIT_MS"] = "900000";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 3) == 900000);
    CHECK(pxa_wedge_limit_ms(p, p.base_ms, 4) == 180000);
    // junk in the new knobs keeps their defaults
    e.kv["PXA_WEDGE_JIT_DECODES"] = "x";
    e.kv["PXA_WEDGE_JIT_MS"] = "";
    p = pxa_wedge_policy_from(true, e, 180000);
    CHECK(p.jit_decodes == 256 && p.jit_ms == 1800000);
}

int main() {
    test_sass();
    test_policy();
    if (g_fail) {
        fprintf(stderr, "test-pxa-jit-wedge: %d FAILED\n", g_fail);
        return 1;
    }
    printf("test-pxa-jit-wedge: all checks passed\n");
    return 0;
}
