// test-pxa-xcache-prefill-freeze.cpp -- PXA_XCACHE_PREFILL_FREEZE / PXA_XCACHE_DECODE_MAXW (src/llama-pxa-xcache.h):
// the decision the expert-cache adaptor makes after every batch: adapt (decode / MTP verify) or keep the residency frozen
// (a prefill ubatch). Pure header, no model, no GPU. The mechanism half (no flip, no swap start after a frozen tick, every expert
// still behind its map entry) is in the closed tree's tests/test-pxa-xcache-adapt.cpp.
#include "llama-pxa-xcache.h"

#include <cstdio>
#include <initializer_list>

static int g_fail = 0;

static void check(const char * what, bool ok) {
    printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

int main() {
    // the default: on exactly where the 1x V100 Overdrive recipe runs the adaptor (RTIER copy-completion), off in the
    // deterministic schedule (GLM eng-m1a, lab ADAPT=1 runs: their swap schedule and answers stay as measured)
    {
        const auto r = llama_pxa_xc_freeze_parse(nullptr, nullptr, /*rtier =*/ true);
        const auto d = llama_pxa_xc_freeze_parse(nullptr, nullptr, /*rtier =*/ false);
        check("unset: on under RTIER, cap 16", r.on && r.maxw == 16);
        check("unset: off in the deterministic schedule", !d.on && d.maxw == 16);
        const auto e = llama_pxa_xc_freeze_parse("", "", true);
        check("empty strings = unset", e.on && e.maxw == 16);
    }
    // the switch
    {
        check("=0 off under RTIER", !llama_pxa_xc_freeze_parse("0", nullptr, true).on);
        check("=0 off in the deterministic schedule", !llama_pxa_xc_freeze_parse("0", nullptr, false).on);
        check("=1 on in the deterministic schedule", llama_pxa_xc_freeze_parse("1", nullptr, false).on);
        check("=1 on under RTIER", llama_pxa_xc_freeze_parse("1", nullptr, true).on);
        check("=2 (any non-zero) on", llama_pxa_xc_freeze_parse("2", nullptr, false).on);
    }
    // the width
    {
        check("DECODE_MAXW=8", llama_pxa_xc_freeze_parse(nullptr, "8", true).maxw == 8);
        check("DECODE_MAXW=1 (plain decode only)", llama_pxa_xc_freeze_parse(nullptr, "1", true).maxw == 1);
        check("DECODE_MAXW=0 -> the default 16", llama_pxa_xc_freeze_parse(nullptr, "0", true).maxw == 16);
        check("DECODE_MAXW=-5 -> the default 16", llama_pxa_xc_freeze_parse(nullptr, "-5", true).maxw == 16);
        check("DECODE_MAXW=100000 -> clamped to 256", llama_pxa_xc_freeze_parse(nullptr, "100000", true).maxw == 256);
    }
    // the decision at the widths this box runs: decode 1, MTP n_max 3 verify 4, PLAN_V2 verify 6, async cold lane <= 8, the
    // cap itself; prefill: the adaptor's own prefill-sized observation (32), a chat prompt (869), the CPU-alias width (1638),
    // the prompt tails of the pp8k / pp28k cells (2663, 3331) and the server's -ub 4096 (PXA_AUTO_UB_STREAM)
    {
        const auto on = llama_pxa_xc_freeze_parse(nullptr, nullptr, true);
        bool dec = true;
        for (int w : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16}) dec = dec && !llama_pxa_xc_frozen(on, w);
        check("decode / verify widths 1..16 adapt", dec);
        bool pre = true;
        for (int w : {17, 32, 64, 512, 869, 1637, 1638, 2663, 3331, 4096}) pre = pre && llama_pxa_xc_frozen(on, w);
        check("prefill widths 17..4096 run frozen", pre);
        const auto off = llama_pxa_xc_freeze_parse("0", nullptr, true);
        bool none = true;
        for (int w : {1, 4, 16, 17, 869, 4096}) none = none && !llama_pxa_xc_frozen(off, w);
        check("=0: no width is frozen (the RC behaviour)", none);
        const auto w8 = llama_pxa_xc_freeze_parse(nullptr, "8", true);
        check("DECODE_MAXW=8: 8 adapts, 9 frozen", !llama_pxa_xc_frozen(w8, 8) && llama_pxa_xc_frozen(w8, 9));
        check("a zero / negative token count never freezes", !llama_pxa_xc_frozen(on, 0) && !llama_pxa_xc_frozen(on, -1));
    }
    printf("RESULT: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
