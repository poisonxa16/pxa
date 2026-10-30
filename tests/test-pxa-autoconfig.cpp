// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// The engine's own serve-flag defaults (common/pxa-registry.cpp) over a case table: the four
// topologies the release is proved on (1x P100, 2x P100, 2x V100, 4x P100) plus the edges that
// decide a pick (mixed pair, gemma4, the wrong tier, too few KV heads, -fa off, -ts, a flag the user
// set, the reference level). No GPU and no model file: the resolver is pure.
//
// Every expectation here is what tools/pxa-launch.py resolve_auto_split() and its recipe table
// answer for the same case, so this test is the engine half of the launcher/engine parity proof.

#include "pxa-registry.h"
#include "ggml-pxqn-overdrive.h"
#include "llama.h"
#include <set>
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv, unsetenv
#include <stdlib.h>
#ifndef environ
#define environ _environ   // the CRT name
#endif
#else
#include <unistd.h>

extern char ** environ;
#endif
#include <algorithm>
#include <utility>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

// A pick's evidence id: compiled in only with -DPXA_REGISTRY_EVIDENCE=ON (dev builds); every other build, release
// included, runs the scrubbed registry whose ids are "" (common/CMakeLists.txt), and then the id must be empty.
static bool ev_is(const std::string & got, const char * id) {
#if defined(PXA_REGISTRY_EVIDENCE_IDS)
    return got == id;
#else
    (void) id;
    return got.empty();
#endif
}

static pxa_model_info mk(const char * arch, const char * tier, int n_expert, int n_head_kv) {
    pxa_model_info m;
    m.ok = true;
    m.arch = arch;
    m.tier = tier;
    m.n_expert = n_expert;
    m.n_head_kv = n_head_kv;
    m.n_layer = 64;
    return m;
}

static const pxa_pick * pick(const pxa_autoconfig & ac, const char * flag) {
    for (const auto & p : ac.picks) {
        if (p.flag == flag) {
            return &p;
        }
    }
    return nullptr;
}

// the value the autoconfig sets for an environment variable, or nullptr when it does not set it
static const char * env_of(const pxa_autoconfig & ac, const char * name) {
    for (const auto & e : ac.env) {
        if (e.first == name) {
            return e.second.c_str();
        }
    }
    return nullptr;
}

static void expect(const char * name, const char * topo, const pxa_model_info & m, const pxa_user_set & u, int level,
                   int want_split, int want_b, int want_ub, const char * want_sm_value) {
    const pxa_topology t = pxa_topology_parse(topo);
    const pxa_autoconfig ac = pxa_autoconfig_resolve(t, m, u, level, 0);
    const pxa_pick * sm = pick(ac, "-sm");
    // a P100 tensor split also carries PXA_PXQN_SM60_RBSHAPE=0 (e2d9716c05, measured by rel-gate2 #14803): its own
    // entry, checked by name, and left out of the count of the split's two
    bool any_sm60 = false;
    for (int cc : t.cc) any_sm60 = any_sm60 || cc == 600;
    const char * rb = env_of(ac, "PXA_PXQN_SM60_RBSHAPE");
    const bool rb_ok = (rb != nullptr && std::string(rb) == "0") == (ac.split == PXA_SM_TENSOR && any_sm60);
    // the PXA_AUTO_PF16K prompt-attention pair rides on the dense 27B 2x P100 tensor cell too (p100-pf16k-dense, own block below)
    const size_t n_pf16k = (env_of(ac, "PXA_FA_TILE_H2CHUNK") != nullptr ? 1 : 0) + (env_of(ac, "PXA_FA_LONGPREFILL") != nullptr ? 1 : 0);
    // the V70 six ride on the qwen35 2x V100 tensor pair (main 2026-10-19, v70-v24-tensor; own block below): left out of the count
    size_t n_v70 = 0;
    if (const pxa_pick * pv = pick(ac, "PXA_AUTO_V70"); pv && pv->applied) {
        for (const char * n : { "PXA_DN_CONVFUSE", "PXA_DN_INPLACE", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX", "PXA_DN_CF_DIRECT",
                                "PXA_SCHED_ASYNC_INPUTS" }) {
            n_v70 += env_of(ac, n) != nullptr ? 1 : 0;
        }
    }
    // the P2T six (PXA_PXQN_RBP + PXA_PXQN_RBP_TP + four decode levers) ride on the dense 27B PXQN4 P100 tensor cell
    // (own block below)
    size_t n_p2t = 0;
    if (const pxa_pick * pp = pick(ac, "PXA_AUTO_P2T"); pp && pp->applied) {
        for (const char * n : { "PXA_PXQN_RBP", "PXA_PXQN_RBP_TP", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX", "PXA_DN_CF_DIRECT",
                                "PXA_PXQN_RMS_RHT_GB" }) {
            n_p2t += env_of(ac, n) != nullptr ? 1 : 0;
        }
    }
    // the verify-identity pair (PXA_FA_VEC_NY1_SPLIT + PXA_KQMMV_VERIFY_ID_SMALL) rides on qwen35 with an sm_60 device (and on
    // Nemotron-H MoE): its own rows below ("qwen35 verify identity: sm_60 only"), left out of the count of the split's two
    const size_t n_vid = (env_of(ac, "PXA_FA_VEC_NY1_SPLIT") != nullptr ? 1 : 0) + (env_of(ac, "PXA_KQMMV_VERIFY_ID_SMALL") != nullptr ? 1 : 0) +
                         (env_of(ac, "PXA_KQMMV_VERIFY_ID_SMALL_W") != nullptr ? 1 : 0);
    const size_t n_split_env = ac.env.size() - (rb != nullptr ? 1 : 0) - n_pf16k - n_v70 - n_p2t - n_vid;
    const bool ok = rb_ok && ac.split == want_split && ac.n_batch == want_b && ac.n_ubatch == want_ub &&
                    sm != nullptr && sm->value == want_sm_value &&
                    // the tensor split always travels with its environment, and only it does. Checked by NAME:
                    // REDUCE=fused + FALLBACK (REDUCE_PREFILL dropped in b74c512371). A count of the entries
                    // cannot tell them from the other picks that set an env on a layer run (PXA_DN_CONVFUSE=0 on
                    // any no-tensor topology, PXA_PXQN_RHT_FUSE=0 on one V100): 94e2509867 made that count 2 on
                    // 1x V100 and six rows failed on it, the picks being measured and intended. Those picks have
                    // their own block below.
                    (env_of(ac, "PXA_TSPLIT_REDUCE") != nullptr && std::string(env_of(ac, "PXA_TSPLIT_REDUCE")) == "fused") ==
                        (ac.split == PXA_SM_TENSOR) &&
                    (env_of(ac, "PXA_TSPLIT_FALLBACK") != nullptr && std::string(env_of(ac, "PXA_TSPLIT_FALLBACK")) == "1") ==
                        (ac.split == PXA_SM_TENSOR) &&
                    // Flash-Next (qwen4exp) on a P100 pair carries its 6 split switches too (HC, UNPROVEN_ARCH, PAIRLOCAL, WORKERS,
                    // REPLAY, HC_FUSED; 2026-10-15)
                    ((ac.split != PXA_SM_TENSOR) || n_split_env == 2u + (m.arch == "qwen4exp" ? 6u : 0u)) &&
                    ((ac.split == PXA_SM_TENSOR) == ac.ts_even);
    printf("%-44s %-10s split=%-2d b=%-5d ub=%-5d sm=%-7s %s\n", name, topo, ac.split, ac.n_batch, ac.n_ubatch,
           sm ? sm->value.c_str() : "?", ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("    want split=%d b=%d ub=%d sm=%s; why: %s\n", want_split, want_b, want_ub, want_sm_value,
               sm ? sm->why.c_str() : "?");
        g_fail++;
    }
}

int main() {
    unsetenv("PXA_AUTO_SM");
    unsetenv("PXA_AUTO_SM_PXQN");
    unsetenv("PXA_TSPLIT_ALLOW_4WAY");
    unsetenv("PXA_AUTO_UB_LONG");
    const pxa_user_set none;
    const auto q27   = mk("qwen35",    "PXQ4", 0, 4);    // Qwen3.8-27B PXQ4 (dense hybrid)
    const auto q27p3 = mk("qwen35",    "PXQ3", 0, 4);
    const auto moe   = mk("qwen35moe", "PXQ4", 256, 2);  // the 35B MoE flagship
    const auto fn    = mk("qwen4exp",  "PXQ_UNIVERSAL", 512, 2);
    const auto g4    = mk("gemma4",    "", 128, 8);
    const auto q27kv1 = mk("qwen35",   "PXQ4", 0, 1);
    const auto pxqu  = mk("qwen35moe", "PXQ_UNIVERSAL", 256, 2);
    const auto l1080 = mk("qwen35moe", "PXQ2", 256, 2);

    // ---- the four proof topologies, the 27B --------------------------------------------------
    expect("1x P100 27B PXQ4",                  "1x600", q27, none, 2, PXA_SM_LAYER,  0,    0,    "layer");
    expect("2x P100 27B PXQ4",                  "2x600", q27, none, 2, PXA_SM_TENSOR, 8192, 256,  "tensor");
    expect("2x V100 27B PXQ4",                  "2x700", q27, none, 2, PXA_SM_TENSOR, 8192, 2048, "tensor");
    // 4x P100 (2026-09-27): the one set past a pair the split is measured faster on
    expect("4x P100 27B PXQ4 (4-way tensor)",   "4x600", q27, none, 2, PXA_SM_TENSOR, 2048, 256,  "tensor");
    expect("3x P100 27B PXQ4 (unmeasured)",     "3x600", q27, none, 2, PXA_SM_LAYER,  0,    0,    "layer");
    expect("4x V100 27B PXQ4 (unmeasured)",     "4x700", q27, none, 2, PXA_SM_LAYER,  0,    0,    "layer");
    setenv("PXA_TSPLIT_ALLOW_4WAY", "0", 1);
    expect("4x P100 27B PXA_TSPLIT_ALLOW_4WAY=0", "4x600", q27, none, 2, PXA_SM_LAYER, 2048, 256,  "layer");
    unsetenv("PXA_TSPLIT_ALLOW_4WAY");
    // ---- other files on the same cards -------------------------------------------------------
    expect("2x P100 MoE PXQ4 (flagship cell)",  "2x600", moe, none, 2, PXA_SM_LAYER,  8192, 2048, "layer");
    expect("2x V100 MoE PXQ4",                  "2x700", moe, none, 2, PXA_SM_LAYER,  8192, 2048, "layer");
    expect("4x P100 Flash-Next (expert)",       "4x600", fn,  none, 2, PXA_SM_LAYER,  2048, 2048, "layer");
    //  (2026-10-12): never -b 8192 for Flash-Next on 16 GB cards (the 2x P100 cell is a dense
    // measurement; CriticalTechGuy's 2x P100 seat got it). The dense 27B keeps its measured 8192 on the same pair.
    //  2026-10-15: Flash-Next on a P100 PAIR takes the tensor split (+7.4/+5.4% MTP, +9.4/+10.0% plain);
    // a V100 pair keeps layer (MTP tie). The -b cap still applies.
    expect("2x P100 Flash-Next: tensor, -b capped 2048", "2x600", fn,  none, 2, PXA_SM_TENSOR, 2048, 256,  "tensor");
    expect("2x P100 16 GB Flash-Next: -b 2048",  "600:16280,600:16280", fn, none, 2, PXA_SM_TENSOR, 2048, 256, "tensor");
    expect("2x V100 Flash-Next: -b capped",      "2x700", fn,  none, 2, PXA_SM_LAYER,  2048, 2048, "layer");
    expect("2x V100 32 GB Flash-Next keeps 8192", "700:32768,700:32768", fn, none, 2, PXA_SM_LAYER, 8192, 2048, "layer");
    {
        // and a measured KV pick for it on P100/V100: q8_0, never a silent f16
        const pxa_autoconfig ac  = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), fn, none, 2, 0);
        const pxa_autoconfig ac1 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), fn, none, 2, 0);
        const pxa_autoconfig acg = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), g4, none, 2, 0);
        const pxa_pick * k = pick(ac, "-ctk");
        const bool ok = ac.kv_type && std::string(ac.kv_type) == "q8_0" && k && k->value == "q8_0" &&
                        ac1.kv_type && std::string(ac1.kv_type) == "q8_0" && acg.kv_type == nullptr;
        g_fail += ok ? 0 : 1;
        printf("%-44s %-10s %s\n", "Flash-Next P100/V100 KV q8_0 (gemma4 untouched)", "2x600", ok ? "PASS" : "FAIL");
    }
    {
        // ac.batch_cell is an OWNED string (2026-10-22): the Flash-Next -b cap writes its why into a static buffer, and
        // -sm layer typed on a P100 pair re-resolves the picks for the cost note, which rewrote that buffer under a stored pointer
        pxa_user_set lay;
        lay.sm = true; lay.sm_value = PXA_SM_LAYER;
        const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), fn, lay, 2, 0);
        const pxa_pick * b = pick(ac, "-b");
        bool cost = false;
        for (const auto & c : ac.costs) cost = cost || c.what == "-sm layer";
        const bool ok = cost && b && ac.batch_cell.find("-b capped at 2048 for the Flash-Next class") != std::string::npos &&
                        b->why == ac.batch_cell;
        g_fail += ok ? 0 : 1;
        printf("%-44s %-10s %s\n", "Flash-Next -sm layer: batch_cell owned", "2x600", ok ? "PASS" : "FAIL");
    }
    expect("2x V100 gemma4",                    "2x700", g4,  none, 2, PXA_SM_LAYER,  2048, 512,  "layer");
    expect("1x V100 gemma4",                    "1x700", g4,  none, 2, PXA_SM_LAYER,  2048, 512,  "layer");
    expect("2x P100 gemma4 expert (measured)",  "2x600", g4,  none, 2, PXA_SM_LAYER,  2048, 512,  "layer");
    // INFERRED launcher rows are not cells: these fall through to the card table, as before step 3
    expect("1x P100 gemma4 (measured 2026-10-05)", "1x600", g4, none, 2, PXA_SM_LAYER, 2048, 256,  "layer");
    expect("4x P100 gemma4 (INFERRED -> table)",  "4x600", g4, none, 2, PXA_SM_LAYER, 2048, 2048, "layer");
    {
        const auto g4d = mk("gemma4", "", 0, 8);
        expect("2x P100 gemma4 dense (INFERRED)",  "2x600", g4d, none, 2, PXA_SM_LAYER, 8192, 256,  "layer");
        expect("2x V100 gemma4 dense (INFERRED)",  "2x700", g4d, none, 2, PXA_SM_LAYER, 8192, 2048, "layer");
    }
    expect("2x P100 27B PXQ3 (tier)",           "2x600", q27p3, none, 2, PXA_SM_LAYER, 8192, 256, "layer");
    // ---- PXQ-Next (2026-09-27): the 2-card sizes take the split on a pair, the 1-card sizes do not ----
    {
        const auto n4   = mk("qwen35", "PXQN4",   0, 4);
        const auto n4s8 = mk("qwen35", "PXQN4S8", 0, 4);
        const auto n5   = mk("qwen35", "PXQN5",   0, 4);
        const auto n3   = mk("qwen35", "PXQN3",   0, 4);
        expect("2x P100 27B PXQN4",                "2x600", n4,   none, 2, PXA_SM_TENSOR, 8192, 256,  "tensor");
        expect("2x V100 27B PXQN4",                "2x700", n4,   none, 2, PXA_SM_TENSOR, 8192, 2048, "tensor");
        expect("2x V100 27B PXQN4S8",              "2x700", n4s8, none, 2, PXA_SM_TENSOR, 8192, 2048, "tensor");
        expect("2x P100 27B PXQN5",                "2x600", n5,   none, 2, PXA_SM_TENSOR, 8192, 256,  "tensor");
        expect("2x P100 27B PXQN3 (1-card size)",  "2x600", n3,   none, 2, PXA_SM_LAYER,  8192, 256,  "layer");
        setenv("PXA_AUTO_SM_PXQN", "0", 1);
        expect("2x V100 27B PXQN4 PXA_AUTO_SM_PXQN=0", "2x700", n4, none, 2, PXA_SM_LAYER, 8192, 2048, "layer");
        unsetenv("PXA_AUTO_SM_PXQN");
    }
    expect("2x P100 1 KV head",                 "2x600", q27kv1, none, 2, PXA_SM_LAYER, 8192, 256, "layer");
    expect("1x P100 PXQU MoE",                  "1x600", pxqu, none, 2, PXA_SM_LAYER, 2048, 2048, "layer");
    expect("1x 1080 Ti PXQ2 MoE",               "1x610", l1080, none, 2, PXA_SM_LAYER, 2048, 768, "layer");
    expect("mixed V100+P100",                   "700,600", q27, none, 2, PXA_SM_LAYER, 0, 0,      "layer");
    // ---- narrow PCIe links (2026-09-27, bug #280): a x1/x2 riser never gets the auto tensor split ----
    {
        pxa_topology t1 = pxa_topology_parse("2x600");
        t1.pcie_width = { 1, 1 };
        pxa_topology t4 = pxa_topology_parse("2x600");
        t4.pcie_width = { 4, 4 };
        const pxa_autoconfig a1 = pxa_autoconfig_resolve(t1, q27, none, 2, 0);
        const pxa_autoconfig a4 = pxa_autoconfig_resolve(t4, q27, none, 2, 0);
        const pxa_pick * s1 = pick(a1, "-sm");
        const bool ok = a1.split == PXA_SM_LAYER && s1 && s1->value == "layer" && ev_is(s1->evidence, "") &&
                        a4.split == PXA_SM_TENSOR && t1.narrow_link() == 0 && t4.narrow_link() == -1;
        printf("%-44s %-10s %s\n", "2x P100 on x1 risers -> layer (x4 -> tensor)", "2x600", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- what the user set wins --------------------------------------------------------------
    {
        pxa_user_set u; u.sm = true;
        expect("2x V100, -sm given",            "2x700", q27, u, 2, PXA_SM_KEEP,   8192, 2048, "");
    }
    {
        pxa_user_set u; u.fa = true; u.fa_value = false;
        expect("2x V100, -fa off given",        "2x700", q27, u, 2, PXA_SM_LAYER,  8192, 2048, "layer");
    }
    {
        pxa_user_set u; u.ts = true;
        expect("2x V100, -ts given",            "2x700", q27, u, 2, PXA_SM_LAYER,  8192, 2048, "layer");
    }
    {
        pxa_user_set u; u.ngl_partial = true;
        expect("2x V100, partial -ngl",         "2x700", q27, u, 2, PXA_SM_LAYER,  8192, 2048, "layer");
    }
    {
        pxa_user_set u; u.b = true; u.ub = true;
        // the cell is still REPORTED (n_batch/n_ubatch), it is just not applied over the user's
        const pxa_topology t = pxa_topology_parse("2x600");
        const pxa_autoconfig ac = pxa_autoconfig_resolve(t, q27, u, 2, 0);
        const pxa_pick * ub = pick(ac, "-ub");
        const bool ok = ub && !ub->applied && ub->status == "USER";
        printf("%-44s %-10s %s\n", "2x P100, -b/-ub given (not applied)", "2x600", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- levels and levers -------------------------------------------------------------------
    expect("2x V100 REFERENCE",                 "2x700", q27, none, 0, PXA_SM_KEEP,   0,    0,    "layer");
    expect("2x V100 DEFAULT (ENHANCE=0)",       "2x700", q27, none, 1, PXA_SM_KEEP,   0,    0,    "layer");
    setenv("PXA_AUTO_SM", "0", 1);
    expect("2x V100 PXA_AUTO_SM=0",             "2x700", q27, none, 2, PXA_SM_KEEP,   8192, 2048, "layer");
    unsetenv("PXA_AUTO_SM");
    setenv("PXA_AUTO_UB_LONG", "0", 1);
    expect("4x P100 27B PXA_AUTO_UB_LONG=0",    "4x600", q27, none, 2, PXA_SM_TENSOR, 2048, 2048, "tensor");
    unsetenv("PXA_AUTO_UB_LONG");
    // ---- the two single-device environment picks: PXA_DN_CONVFUSE and PXA_PXQN_RHT_FUSE ----------------
    // Both are measured defaults, not accidents. The six 1x V100 rows above first failed at 94e2509867
    // (RHT_FUSE=0 joined DN_CONVFUSE=0, so a layer run on one V100 carries 2 env entries, the count the old
    // check read as "tensor split"): a stale expectation, replaced by the by-name check in expect() and by
    // these rows. Labels follow where the A/B was actually run:
    //   PXA_DN_CONVFUSE=0 without a tensor split (dd121111aa).
    //     MEASURED 2026-09-29, 27B one-card file, llama-bench tg128 ABAB, -fa 1 q4_0 KV: 1x V100 fused
    //     23.3/22.4 vs unfused 32.3/33.6 t/s (-31%), 1x P100 19.4/19.3 vs 24.6/24.2 (-21%). With -sm tensor the
    //     fusion wins (2x P100 +4.4%, 2x V100 +5.3%) and stays on. No A/B ran on a layer-split pair, 3-4 cards or
    //     a 1080 Ti: the same pick there is carried over from the one-card runs, so INFERRED.
    //   PXA_PXQN_RHT_FUSE=0 on one V100 (94e2509867, evidence rhtfuse-single-v100-2026-09-29, ledger
    //     measured 2026-09-29 20:20 EDT): 1x V100, llama-bench tg128 -fa 1 q4_0 KV,
    //     REPS 3, 3 interleaved rounds: one-card 27B 34.8-35.0 unfused vs 32.3-33.4 fused t/s (+5..8%), PXQN4
    //     ladder 34.8/33.8 vs 31.0/32.6, greedy512 sha identical (435d10f612f952d1). P100 and pairs were not
    //     measured and keep the fusion (no pick there). REMOVED 2026-10-14 (K3):
    //     on that binary the fusions are a one-V100 WIN (tg128 40.41 -> 41.04, server
    //     +1.3%, sha identical), so one V100 has no RHT_FUSE pick either (want_rht nullptr below).
    {
        struct { const char * name; const char * topo; const pxa_model_info * m; int level;
                 const char * want_conv; const char * want_conv_status; const char * want_rht; } envrows[] = {
            {"1x V100 27B PXQ3",                "1x700", &q27p3, 2, "0",     "MEASURED", nullptr},
            {"1x V100 gemma4",                  "1x700", &g4,    2, "0",     "MEASURED", nullptr},
            {"1x P100 27B PXQ4",                "1x600", &q27,   2, "0",     "MEASURED", nullptr},
            {"1x 1080 Ti PXQ2 MoE (unmeasured)", "1x610", &l1080, 2, "0",    "INFERRED", nullptr},
            {"2x P100 MoE layer (unmeasured)",  "2x600", &moe,   2, "0",     "INFERRED", nullptr},
            {"3x P100 27B layer (unmeasured)",  "3x600", &q27,   2, "0",     "INFERRED", nullptr},
            {"2x V100 27B tensor (fusion stays)", "2x700", &q27, 2, nullptr, "MEASURED", nullptr},
            {"2x P100 27B tensor (fusion stays)", "2x600", &q27, 2, nullptr, "MEASURED", nullptr},
            {"4x P100 27B tensor (fusion stays)", "4x600", &q27, 2, nullptr, "MEASURED", nullptr},
            {"1x V100 27B DEFAULT level (no picks)", "1x700", &q27p3, 1, nullptr, nullptr, nullptr},
            {"1x V100 27B REFERENCE (no picks)",    "1x700", &q27p3, 0, nullptr, nullptr, nullptr},
        };
        // these rows are the picks WITHOUT the one-V100 qwen35 stack (its stand-down switch); the stack has its own block below
        setenv("PXA_AUTO_V70", "0", 1);
        for (const auto & r : envrows) {
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, r.level, 0);
            const char * conv = env_of(ac, "PXA_DN_CONVFUSE");
            const char * rht  = env_of(ac, "PXA_PXQN_RHT_FUSE");
            const pxa_pick * pc = pick(ac, "PXA_DN_CONVFUSE");
            const pxa_pick * pr = pick(ac, "PXA_PXQN_RHT_FUSE");
            bool ok = (conv == nullptr) == (r.want_conv == nullptr) && (!conv || std::string(conv) == r.want_conv) &&
                      (rht == nullptr) == (r.want_rht == nullptr) && (!rht || std::string(rht) == r.want_rht);
            if (r.want_conv_status) {
                ok = ok && pc && pc->status == r.want_conv_status && pc->applied == (r.want_conv != nullptr) &&
                     pc->value == (r.want_conv ? "off" : "on") && ev_is(pc->evidence, "");
            } else {
                ok = ok && pc == nullptr;
            }
            if (r.want_rht) {
                ok = ok && pr && pr->status == "MEASURED" && pr->applied && pr->value == "off" &&
                     ev_is(pr->evidence, "");
            } else {
                ok = ok && pr == nullptr;
            }
            printf("%-44s %-10s DN_CONVFUSE=%s RHT_FUSE=%s %s\n", r.name, r.topo, conv ? conv : "-", rht ? rht : "-",
                   ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    unsetenv("PXA_AUTO_V70");
    setenv("PXA_AUTO_V70_G4", "0", 1);   // the qwen35 block checks its own stack; the Gemma 4 cell has its own block below
    setenv("PXA_AUTO_P1", "0", 1);       // the one-P100 qwen35 PXQN cell has its own block below
    // ---- the one-V100 qwen35 plain-decode stack (2026-10-16) ----
    // CONVFUSE + INPLACE + ARGMAX_VAL_MB + PLAIN_GREEDY_AMAX + DN_CF_DIRECT + SCHED_ASYNC_INPUTS = "1" exactly on qwen35, one sm_70 card, no
    // tensor split, level 2.
    {
        const auto n4 = mk("qwen35", "PXQN4", 0, 4);
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        pxa_user_set sm_l = none; sm_l.sm = true; sm_l.sm_value = PXA_SM_LAYER;
        static const char * const v70[] = { "PXA_DN_CONVFUSE", "PXA_DN_INPLACE", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX",
                                            "PXA_DN_CF_DIRECT", "PXA_SCHED_ASYNC_INPUTS" };
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off_switch; bool want; } rows[] = {
            {"1x V100 27B PXQN4 (the measured cell)", "1x700", &n4,    &none, 2, false, true },
            {"1x V100 27B PXQ3",                      "1x700", &q27p3, &none, 2, false, true },
            {"1x V100 27B PXQN4 PXA_AUTO_V70=0",      "1x700", &n4,    &none, 2, true,  false},
            {"1x V100 27B PXQN4 DEFAULT level",       "1x700", &n4,    &none, 1, false, false},
            {"1x V100 27B PXQN4 user -sm tensor",     "1x700", &n4,    &sm_t, 2, false, false},
            {"1x V100 gemma4",                        "1x700", &g4,    &none, 2, false, false},
            {"1x P100 27B PXQN4",                     "1x600", &n4,    &none, 2, false, false},
            // the 2x V100 pair resolves to -sm tensor on its own and takes the six (main 2026-10-19, v70-v24-tensor); a layer
            // split pair and the pair's off switch do not
            {"2x V100 27B PXQN4 (auto tensor)",        "2x700", &n4,    &none, 2, false, true },
            {"2x V100 27B PXQN4 user -sm layer",       "2x700", &n4,    &sm_l, 2, false, false},
            {"2x V100 27B PXQN4 PXA_AUTO_V70=0",       "2x700", &n4,    &none, 2, true,  false},
        };
        for (const auto & r : rows) {
            if (r.off_switch) setenv("PXA_AUTO_V70", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_V70");
            int n_on = 0;
            for (const char * n : v70) {
                const char * v = env_of(ac, n);
                n_on += v && std::string(v) == "1";
            }
            const pxa_pick * pv = pick(ac, "PXA_AUTO_V70");
            const pxa_pick * pc = pick(ac, "PXA_DN_CONVFUSE");
            const bool all = n_on == (int) (sizeof(v70) / sizeof(v70[0]));
            const bool pair = std::string(r.topo) == "2x700";
            bool ok = r.want ? (all && pv && pv->applied && pv->status == "MEASURED" && pc && pc->value == "on" &&
                                ev_is(pc->evidence, "") &&
                                (pair ? ev_is(pv->evidence, "") : ev_is(pv->evidence, "")))
                             : (pv == nullptr && env_of(ac, "PXA_DN_INPLACE") == nullptr && env_of(ac, "PXA_PLAIN_GREEDY_AMAX") == nullptr &&
                                env_of(ac, "PXA_DN_CF_DIRECT") == nullptr && env_of(ac, "PXA_SCHED_ASYNC_INPUTS") == nullptr);
            printf("%-44s %-10s V70 stack %-4s %s\n", r.name, r.topo, all ? "on" : (n_on ? "part" : "off"), ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    unsetenv("PXA_AUTO_V70_G4");
    unsetenv("PXA_AUTO_P1");
    // ---- the dense 27B PXQN4 on the P100 tensor pair: PXA_PXQN_RBP + PXA_PXQN_RBP_TP (2026-10-20) ----
    // all six "1" exactly on two sm_60 cards, tensor split, qwen35, PXQN4, level 2; PXA_AUTO_P2T=0 stands it down. The one-P100 and
    // the 35B P60 picks set PXA_PXQN_RBP on their own cells but never RBP_TP.
    {
        const auto n4   = mk("qwen35",    "PXQN4", 0, 4);
        const auto moen = mk("qwen35moe", "PXQN4", 256, 2);
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        pxa_user_set sm_l = none; sm_l.sm = true; sm_l.sm_value = PXA_SM_LAYER;
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off; bool want; } rows[] = {
            {"2x P100 27B PXQN4 (auto tensor)",         "2x600", &n4,   &none, 2, false, true },
            {"2x P100 27B PXQN4 user -sm tensor",       "2x600", &n4,   &sm_t, 2, false, true },
            {"2x P100 27B PXQN4 user -sm layer",        "2x600", &n4,   &sm_l, 2, false, false},
            {"2x P100 27B PXQN4 PXA_AUTO_P2T=0",        "2x600", &n4,   &none, 2, true,  false},
            {"2x P100 27B PXQN4 DEFAULT level",         "2x600", &n4,   &none, 1, false, false},
            {"2x P100 27B PXQ4 (other tier)",           "2x600", &q27,  &none, 2, false, false},
            {"2x P100 35B PXQN4 (P60 cell)",            "2x600", &moen, &none, 2, false, false},
            {"2x V100 27B PXQN4 (unmeasured)",          "2x700", &n4,   &none, 2, false, false},
            {"4x P100 27B PXQN4 (unmeasured)",          "4x600", &n4,   &none, 2, false, false},
            {"1x P100 27B PXQN4 (its own RBP pick)",    "1x600", &n4,   &none, 2, false, false},
        };
        for (const auto & r : rows) {
            if (r.off) setenv("PXA_AUTO_P2T", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_P2T");
            const char * tp = env_of(ac, "PXA_PXQN_RBP_TP");
            const pxa_pick * pp = pick(ac, "PXA_AUTO_P2T");
            int n_on = 0;
            for (const char * n : { "PXA_PXQN_RBP", "PXA_PXQN_RBP_TP", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX", "PXA_DN_CF_DIRECT",
                                    "PXA_PXQN_RMS_RHT_GB" }) {
                const char * v = env_of(ac, n);
                n_on += v && std::string(v) == "1";
            }
            const bool on = n_on == 6;
            const bool ok = r.want ? (on && pp && pp->applied && pp->status == "MEASURED" && pp->value == "on" && ev_is(pp->evidence, ""))
                                   : (tp == nullptr && pp == nullptr);
            printf("%-44s %-10s P2T %-3s %s\n", r.name, r.topo, on ? "on" : "off", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    // ---- the dense 27B PXQN4 on the P100 LAYER pair: the one-P100 decode stack on both cards (2026-10-21) ----
    // all six "1" exactly on two sm_60 cards, layer split, qwen35, PXQN4, level 2, and the DN_CONVFUSE pick says on (no "0" entry);
    // PXA_AUTO_P2L=0 stands it down (then the no-tensor-split rule's DN_CONVFUSE=0 is back). Never on the tensor pair (P2T's cell),
    // the 35B P60 cell, another tier, a V100 pair, 4 cards or one card.
    {
        const auto n4   = mk("qwen35",    "PXQN4", 0, 4);
        const auto n3   = mk("qwen35",    "PXQN3", 0, 4);
        const auto moen = mk("qwen35moe", "PXQN4", 256, 2);
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        pxa_user_set sm_l = none; sm_l.sm = true; sm_l.sm_value = PXA_SM_LAYER;
        static const char * const p2l[] = { "PXA_PXQN_RBP", "PXA_PXQN_RMS_RHT_GB", "PXA_ARGMAX_VAL_MB", "PXA_DN_CONVFUSE",
                                            "PXA_DN_CF_DIRECT", "PXA_PLAIN_GREEDY_AMAX" };
        // want_conv: the DN_CONVFUSE value the resolve must set ("1" the cell, "0" the no-tensor-split rule, nullptr = not checked)
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off; bool want;
                 const char * want_conv; } rows[] = {
            {"2x P100 27B PXQN4 user -sm layer (G10)",    "2x600", &n4,   &sm_l, 2, false, true,  "1"    },
            {"2x P100 27B PXQN4 -sm layer PXA_AUTO_P2L=0", "2x600", &n4,  &sm_l, 2, true,  false, "0"    },
            {"2x P100 27B PXQN4 -sm layer DEFAULT level", "2x600", &n4,   &sm_l, 1, false, false, nullptr},
            {"2x P100 27B PXQN4 (auto tensor)",           "2x600", &n4,   &none, 2, false, false, nullptr},
            {"2x P100 27B PXQN4 user -sm tensor",         "2x600", &n4,   &sm_t, 2, false, false, nullptr},
            {"2x P100 27B PXQN3 -sm layer (other tier)",  "2x600", &n3,   &sm_l, 2, false, false, "0"    },
            {"2x P100 35B PXQN4 -sm layer (P60 cell)",    "2x600", &moen, &sm_l, 2, false, false, nullptr},
            {"2x V100 27B PXQN4 -sm layer",               "2x700", &n4,   &sm_l, 2, false, false, "0"    },
            {"4x P100 27B PXQN4 -sm layer",               "4x600", &n4,   &sm_l, 2, false, false, "0"    },
            {"1x P100 27B PXQN4 (the one-card cells)",    "1x600", &n4,   &none, 2, false, false, nullptr},
        };
        for (const auto & r : rows) {
            if (r.off) setenv("PXA_AUTO_P2L", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_P2L");
            const pxa_pick * pp = pick(ac, "PXA_AUTO_P2L");
            const pxa_pick * pc = pick(ac, "PXA_DN_CONVFUSE");
            int n_on = 0;
            for (const char * n : p2l) {
                const char * v = env_of(ac, n);
                n_on += v && std::string(v) == "1";
            }
            // env_of returns the FIRST entry for a name, the one applied (setenv never overwrites)
            const char * conv = env_of(ac, "PXA_DN_CONVFUSE");
            const bool on = n_on == (int) (sizeof(p2l) / sizeof(p2l[0]));
            bool ok = r.want ? (on && pp && pp->applied && pp->status == "MEASURED" && pp->value == "on" &&
                                ev_is(pp->evidence, "") && pc && pc->value == "on" && !pc->applied &&
                                ev_is(pc->evidence, ""))
                             : pp == nullptr;
            if (r.want_conv) {
                ok = ok && conv && std::string(conv) == r.want_conv;
            }
            printf("%-44s %-10s P2L %-3s DN_CONVFUSE=%s %s\n", r.name, r.topo, on ? "on" : "off", conv ? conv : "-", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    // ---- the one-V100 Gemma 4 MoE decode stack (2026-10-17, ledger gemma-v100-decode-stack) ----
    // CUDA_GRAPH_MOE + ARGMAX_VAL_MB + PLAIN_GREEDY_AMAX + SCHED_ASYNC_INPUTS + G2_NORMFUSE + NORM_ADD_SCALE = "1" exactly on gemma4 with experts, one sm_70 card,
    // no tensor split, level 2; never the qwen35 stack's DN levers.
    {
        const auto g4n = mk("gemma4", "PXQN4", 128, 8);
        const auto g4d = mk("gemma4", "PXQN4", 0, 8);
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        static const char * const g4s[] = { "PXA_CUDA_GRAPH_MOE", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX", "PXA_SCHED_ASYNC_INPUTS",
                                            "PXA_G2_NORMFUSE", "PXA_NORM_ADD_SCALE" };
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off_switch; bool want; } rows[] = {
            {"1x V100 gemma4 26B PXQN4 (the G50 cell)", "1x700", &g4n, &none, 2, false, true },
            {"1x V100 gemma4 (no tier)",                "1x700", &g4,  &none, 2, false, true },
            {"1x V100 gemma4 PXA_AUTO_V70_G4=0",        "1x700", &g4n, &none, 2, true,  false},
            {"1x V100 gemma4 DEFAULT level",            "1x700", &g4n, &none, 1, false, false},
            {"1x V100 gemma4 user -sm tensor",          "1x700", &g4n, &sm_t, 2, false, false},
            {"1x V100 gemma4 dense",                    "1x700", &g4d, &none, 2, false, false},
            {"1x P100 gemma4 PXQN4",                    "1x600", &g4n, &none, 2, false, false},
            {"2x V100 gemma4 PXQN4",                    "2x700", &g4n, &none, 2, false, false},
        };
        for (const auto & r : rows) {
            if (r.off_switch) setenv("PXA_AUTO_V70_G4", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_V70_G4");
            int n_on = 0;
            for (const char * n : g4s) {
                const char * v = env_of(ac, n);
                n_on += v && std::string(v) == "1";
            }
            const pxa_pick * pg = pick(ac, "PXA_AUTO_V70_G4");
            const bool all = n_on == (int) (sizeof(g4s) / sizeof(g4s[0]));
            bool ok = r.want ? (all && pg && pg->applied && pg->status == "MEASURED" && ev_is(pg->evidence, "") &&
                                env_of(ac, "PXA_DN_INPLACE") == nullptr && pick(ac, "PXA_AUTO_V70") == nullptr)
                             : (pg == nullptr && env_of(ac, "PXA_CUDA_GRAPH_MOE") == nullptr && env_of(ac, "PXA_PLAIN_GREEDY_AMAX") == nullptr);
            printf("%-44s %-10s V70_G4 stack %-4s %s\n", r.name, r.topo, all ? "on" : (n_on ? "part" : "off"), ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    // ---- the P60 verify stack: Qwen3.6-35B-A3B (qwen35moe, PXQN4) on layer-split P100s (2026-10-16; 4 cards 2026-10-20) ----------
    // Eleven levers, OFF in the engine and ON in the board recipe, are the registry's pick for exactly this cell
    // (plain 83.1 -> 133.1 t/s, MTP 99.6 -> 138.1). Every other cell carries none of them; the DN_CONVFUSE pick flips to "on" only here.
    // The cell is two or more cards that are ALL sm_60 (measured on 2 and 4 = MEASURED; 3 = INFERRED); one card, a mixed
    // P100+V100 set or a tensor split never takes it.
    {
        const auto moen = mk("qwen35moe", "PXQN4", 256, 2);   // the 35B-A3B MTP PXQN4 file
        const auto n4d  = mk("qwen35",    "PXQN4", 0, 4);     // the dense 27B PXQN4 file
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        pxa_user_set sm_l = none; sm_l.sm = true; sm_l.sm_value = PXA_SM_LAYER;
        static const char * const on1[] = { "PXA_P60_MEGA", "PXA_P60_MEGA_PXQN_T", "PXA_KQMMV_VW", "PXA_PXQN_VW", "PXA_P60_PRE_VW",
                                            "PXA_PXQN_RBP", "PXA_PXQN_RMS_RHT_GB", "PXA_ARGMAX_VAL_MB", "PXA_DN_CONVFUSE", "PXA_PXQN_QPN" };
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off_switch; bool want;
                 const char * status = "MEASURED"; } rows[] = {
            {"2x P100 35B PXQN4 layer (the board cell)", "2x600", &moen, &none, 2, false, true },
            {"2x P100 35B PXQN4, user -sm layer",        "2x600", &moen, &sm_l, 2, false, true },
            {"2x P100 35B PXQN4, user -sm tensor",       "2x600", &moen, &sm_t, 2, false, false},
            {"2x P100 35B PXQN4 PXA_AUTO_P60=0",         "2x600", &moen, &none, 2, true,  false},
            {"2x P100 35B PXQN4 DEFAULT level",          "2x600", &moen, &none, 1, false, false},
            {"2x V100 35B PXQN4 (unmeasured)",           "2x700", &moen, &none, 2, false, false},
            {"1x P100 35B PXQN4 (one card: no)",         "1x600", &moen, &none, 2, false, false},
            {"4x P100 35B PXQN4 layer (CX51-53)",        "4x600", &moen, &none, 2, false, true },
            {"4x P100 35B PXQN4, user -sm layer",        "4x600", &moen, &sm_l, 2, false, true },
            {"4x P100 35B PXQN4, user -sm tensor",       "4x600", &moen, &sm_t, 2, false, false},
            {"4x P100 35B PXQN4 PXA_AUTO_P60=0",         "4x600", &moen, &none, 2, true,  false},
            {"3x P100 35B PXQN4 layer (INFERRED)",       "3x600", &moen, &none, 2, false, true, "INFERRED"},
            {"P100+V100 35B PXQN4 (mixed pair: no)",     "600,700", &moen, &none, 2, false, false},
            {"3x P100+V100 35B PXQN4 (mixed: no)",       "600,600,600,700", &moen, &none, 2, false, false},
            {"4x V100 35B PXQN4 (no)",                   "4x700", &moen, &none, 2, false, false},
            {"2x P100 35B PXQ4 (other tier)",            "2x600", &moe,  &none, 2, false, false},
            {"4x P100 35B PXQ4 (other tier)",            "4x600", &moe,  &none, 2, false, false},
            {"2x P100 27B PXQN4 dense (tensor)",         "2x600", &n4d,  &none, 2, false, false},
        };
        for (const auto & r : rows) {
            if (r.off_switch) setenv("PXA_AUTO_P60", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_P60");
            int n_on = 0;
            for (const char * n : on1) {
                const char * v = env_of(ac, n);
                n_on += v && std::string(v) == "1";
            }
            const char * gp = env_of(ac, "PXA_FA_GQA_PACK");
            const bool all = n_on == (int) (sizeof(on1) / sizeof(on1[0])) && gp && std::string(gp) == "0";
            const bool none_set = env_of(ac, "PXA_P60_MEGA") == nullptr && env_of(ac, "PXA_KQMMV_VW") == nullptr && gp == nullptr;
            const pxa_pick * pm = pick(ac, "PXA_P60_MEGA");
            const pxa_pick * pc = pick(ac, "PXA_DN_CONVFUSE");
            bool ok = r.want ? (all && pm && pm->status == r.status && pm->applied && pc && pc->value == "on" && pc->status == r.status)
                             : (none_set && pm == nullptr);
            printf("%-44s %-10s P60 stack %-4s %s\n", r.name, r.topo, all ? "on" : (none_set ? "off" : "part"), ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
        //  (order #22136): llama-bench resolves with every placement/batch axis given (-sm layer is its
        // default) and must get the same stack; PXA_AUTOCONFIG=0 stands the whole resolve down (no env, no -sm/-ngl/-c pick)
        {
            pxa_user_set bench = none;
            bench.sm = true; bench.sm_value = PXA_SM_LAYER;
            bench.b = bench.ub = bench.fa = bench.ngl = bench.ctx = bench.kv = true;
            const pxa_autoconfig ab = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), moen, bench, 2, 0);
            const bool ok_b = env_of(ab, "PXA_P60_MEGA") && env_of(ab, "PXA_PXQN_QPN") && env_of(ab, "PXA_FA_GQA_PACK");
            printf("%-44s %-10s P60 stack %-4s %s\n", "2x P100 35B PXQN4, llama-bench user set", "2x600", ok_b ? "on" : "off",
                   ok_b ? "PASS" : "FAIL");
            g_fail += !ok_b;
            setenv("PXA_AUTOCONFIG", "0", 1);
            const pxa_autoconfig a0 = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), moen, none, 2, 0);
            const pxa_autoconfig a1 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), n4d, none, 2, 0);
            unsetenv("PXA_AUTOCONFIG");
            const bool ok0 = a0.env.empty() && a1.env.empty() && a0.split == PXA_SM_KEEP && a0.n_gpu_layers < 0 &&
                             a0.n_ctx == 0 && a0.kv_type == nullptr && a1.kv_type == nullptr && a1.n_gpu_layers < 0;
            printf("%-44s %-10s env %zu+%zu %s\n", "PXA_AUTOCONFIG=0 (whole registry off)", "2x600/1x700",
                   a0.env.size(), a1.env.size(), ok0 ? "PASS" : "FAIL");
            g_fail += !ok0;
        }
    }
    // ---- 2x P100 35B PXQN4 long prefill trio (gated) -------------------------------------
    {
        const auto moen = mk("qwen35moe", "PXQN4", 256, 2);
        const auto n4d  = mk("qwen35",    "PXQN4", 0, 4);
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off; bool want; } rows[] = {
            {"2x P100 35B PXQN4 -> PF16K trio on",        "2x600", &moen, &none, 2, false, true },
            {"2x P100 35B PXQN4 user -sm tensor -> on",   "2x600", &moen, &sm_t, 2, false, true },
            {"2x P100 35B PXQN4 PXA_AUTO_PF16K=0",        "2x600", &moen, &none, 2, true,  false},
            {"2x P100 35B PXQN4 DEFAULT level",           "2x600", &moen, &none, 1, false, false},
            {"2x V100 35B PXQN4 (unmeasured)",            "2x700", &moen, &none, 2, false, false},
            {"1x P100 35B PXQN4 (unmeasured)",            "1x600", &moen, &none, 2, false, false},
            {"4x P100 35B PXQN4 -> on (2026-10-20)",      "4x600", &moen, &none, 2, false, true },
            {"4x P100 35B PXQN4 PXA_AUTO_PF16K=0",        "4x600", &moen, &none, 2, true,  false},
            {"3x P100 35B PXQN4 -> on (INFERRED)",        "3x600", &moen, &none, 2, false, true },
            {"3x P100 + 1x V100 35B PXQN4 (mixed)",       "600,600,600,700", &moen, &none, 2, false, false},
            {"4x V100 35B PXQN4 (unmeasured)",            "4x700", &moen, &none, 2, false, false},
            {"2x P100 27B PXQN4 dense -> H2CHUNK+LONG on", "2x600", &n4d,  &none, 2, false, true },
            {"2x P100 27B dense user -sm tensor -> on",   "2x600", &n4d,  &sm_t, 2, false, true },
            {"2x P100 27B PXQN4 dense PXA_AUTO_PF16K=0",  "2x600", &n4d,  &none, 2, true,  false},
            {"2x P100 27B PXQN4 dense DEFAULT level",     "2x600", &n4d,  &none, 1, false, false},
            {"2x V100 27B PXQN4 dense (unmeasured)",      "2x700", &n4d,  &none, 2, false, false},
            {"1x P100 27B PXQN4 dense -> on (2026-10-18)", "1x600", &n4d,  &none, 2, false, true },
            {"1x P100 27B PXQN4 dense PXA_AUTO_PF16K=0",  "1x600", &n4d,  &none, 2, true,  false},
            {"1x P100 27B PXQN4 dense DEFAULT level",     "1x600", &n4d,  &none, 1, false, false},
            {"1x V100 27B PXQN4 dense (unmeasured)",      "1x700", &n4d,  &none, 2, false, false},
        };
        for (const auto & r : rows) {
            if (r.off) setenv("PXA_AUTO_PF16K", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_PF16K");
            const char * s1 = env_of(ac, "PXA_PF_STABLE_SHAPE");
            const char * s2 = env_of(ac, "PXA_FA_TILE_H2CHUNK");
            const char * s3 = env_of(ac, "PXA_FA_LONGPREFILL");
            const bool attn = s2 && s3 && std::string(s2) == "1" && std::string(s3) == "3";
            // the MoE cell carries the stable shape too; the dense cell (p100-pf16k-dense) only the two attention halves
            const bool on = r.m == &moen ? (attn && s1 && std::string(s1) == "1") : attn;
            const bool ok = r.want ? on : (s2 == nullptr && s3 == nullptr);   // s1 is the sm_60 MoE rule's (its own rows)
            printf("%-44s %-10s PF16K %-3s %s\n", r.name, r.topo, on ? "on" : "off", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    // ---- one P100, PXQN file: RMS_RHT_GB + ARGMAX_VAL_MB ----------------------
    {
        const auto n4d  = mk("qwen35",    "PXQN4", 0, 4);
        const auto moen = mk("qwen35moe", "PXQN4", 256, 2);
        const auto moe4 = mk("qwen35moe", "PXQ4", 256, 2);
        struct { const char * name; const char * topo; const pxa_model_info * m; int level; bool want; bool want_rbp; } rows[] = {
            {"1x P100 27B PXQN4 -> K3 pair + RBP on", "1x600", &n4d,  2, true,  true },
            {"1x P100 35B PXQN4 -> K3 pair on",       "1x600", &moen, 2, true,  false},
            {"1x P100 27B PXQN4 DEFAULT level",       "1x600", &n4d,  1, false, false},
            {"1x P100 PXQ4 file (unmeasured)",        "1x600", &moe4, 2, false, false},
            {"1x V100 27B PXQN4 (init rule instead)", "1x700", &n4d,  2, false, false},
            {"2x P100 27B PXQN4 (K3 unmeasured)",     "2x600", &n4d,  2, false, false},
        };
        for (const auto & r : rows) {
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, r.level, 0);
            const char * g = env_of(ac, "PXA_PXQN_RMS_RHT_GB");
            const char * a = env_of(ac, "PXA_ARGMAX_VAL_MB");
            const pxa_pick * pg = pick(ac, "PXA_PXQN_RMS_RHT_GB");
            const bool on = g && a && std::string(g) == "1" && std::string(a) == "1";
            const char * rb = env_of(ac, "PXA_PXQN_RBP");
            const pxa_pick * pr = pick(ac, "PXA_PXQN_RBP");
            const bool rbp_on = rb && std::string(rb) == "1";
            const bool rbp_ok = r.want_rbp ? (rbp_on && pr && ev_is(pr->evidence, "") && pr->applied)
                                           // the P100 tensor pair's RBP is PXA_AUTO_P2T's (own block), never this pick's
                                           : ((rb == nullptr || pick(ac, "PXA_AUTO_P2T") != nullptr) && pr == nullptr);
            // the P100 tensor pair: PXA_AUTO_P2T owns RMS_RHT_GB + ARGMAX_VAL_MB there, never this pick
            const bool p2t = pick(ac, "PXA_AUTO_P2T") != nullptr;
            const bool ok = rbp_ok && (r.want ? (on && pg && ev_is(pg->evidence, "") && pg->applied)
                                              : ((g == nullptr || p2t) && pg == nullptr &&
                                                 // one V100 qwen35: PXA_AUTO_V70 owns ARGMAX_VAL_MB (pass 5: v100-plain)
                                                 (a == nullptr || p2t || std::string(r.topo) == "1x700")));
            printf("%-44s %-10s K3 pair %-3s RBP %-3s %s\n", r.name, r.topo, on ? "on" : "off", rbp_on ? "on" : "off", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    // ---- the one-P100 qwen35 PXQN additions (2026-10-18): CONVFUSE + CF_DIRECT + PLAIN_GREEDY_AMAX = "1" ----
    {
        const auto n4p = mk("qwen35", "PXQN4", 0, 4);
        const auto n2p = mk("qwen35", "PXQN2", 0, 4);
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        setenv("PXA_AUTO_V70", "0", 1);   // the one-V100 row checks only that this cell stays off there
        static const char * const p1[] = { "PXA_DN_CONVFUSE", "PXA_DN_CF_DIRECT", "PXA_PLAIN_GREEDY_AMAX" };
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; bool off_switch; bool want; } rows[] = {
            {"1x P100 27B PXQN4 (the measured cell)", "1x600", &n4p, &none, 2, false, true },
            {"1x P100 27B PXQN2",                     "1x600", &n2p, &none, 2, false, true },
            {"1x P100 27B PXQN4 PXA_AUTO_P1=0",       "1x600", &n4p, &none, 2, true,  false},
            {"1x P100 27B PXQN4 DEFAULT level",       "1x600", &n4p, &none, 1, false, false},
            {"1x P100 27B PXQ4 (classic, unmeasured)", "1x600", &q27, &none, 2, false, false},
            {"1x P100 35B PXQN4 MoE (unmeasured)",    "1x600", &moe, &none, 2, false, false},
            {"1x V100 27B PXQN4 (the V70 cell)",      "1x700", &n4p, &none, 2, false, false},
            {"2x P100 27B PXQN4",                     "2x600", &n4p, &none, 2, false, false},
            {"1x P100 27B PXQN4 user -sm tensor",     "1x600", &n4p, &sm_t, 2, false, false},
        };
        for (const auto & r : rows) {
            if (r.off_switch) setenv("PXA_AUTO_P1", "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            unsetenv("PXA_AUTO_P1");
            int n_on = 0;
            for (const char * n : p1) {
                const char * v = env_of(ac, n);
                n_on += v && std::string(v) == "1";
            }
            const pxa_pick * pp = pick(ac, "PXA_AUTO_P1");
            const pxa_pick * pc = pick(ac, "PXA_DN_CONVFUSE");
            const char * conv = env_of(ac, "PXA_DN_CONVFUSE");
            bool ok;
            if (r.want) {
                ok = n_on == 3 && pp && pp->applied && pp->status == "MEASURED" && pc && pc->value == "on" &&
                     ev_is(pc->evidence, "") && std::string(conv) == "1";
            } else {
                // the P100 tensor pair: PXA_AUTO_P2T owns DN_CF_DIRECT + PLAIN_GREEDY_AMAX there
                const bool p2t = pick(ac, "PXA_AUTO_P2T") != nullptr;
                ok = pp == nullptr && ((env_of(ac, "PXA_DN_CF_DIRECT") == nullptr && env_of(ac, "PXA_PLAIN_GREEDY_AMAX") == nullptr) || p2t);
                // the old one-card pick (off) still holds wherever the cell does not apply, on one P100 at level 2
                if (std::string(r.topo) == "1x600" && r.level == 2 && r.u == &none) {
                    ok = ok && conv && std::string(conv) == "0";
                }
            }
            printf("%-44s %-10s P1 cell %-4s %s\n", r.name, r.topo, n_on == 3 ? "on" : (n_on ? "part" : "off"), ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
        unsetenv("PXA_AUTO_V70");
    }
    // ---- PXA Overdrive: the recipe and the decision are the closed PXQN library's (ggml-pxqn-overdrive.h;
    // 2026-10-19). This open test spells no row. Library present: the Flash-Next class larger than the cards gets the
    // library's rows (its library-only rows found through ggml_pxqn_overdrive_is_member / pxa_lever_is_closed, the CPU
    // pool derived by the engine appended with them), the EXPLAIN json carries those rows as the alias and the banner
    // counts them, the off switches and the non-matching seats get nothing. Library absent (a public tree): nothing,
    // ever. The rows themselves are pinned by tests/test-pxqn-overdrive.cpp (closed tree).
    {
        const bool lib = ggml_pxqn_overdrive_available();
        printf("%-44s %-12s %s\n", "Overdrive library table", "-", lib ? "present" : "absent: Overdrive off");
        auto env_names = [](std::set<std::string> & out) {
            out.clear();
            for (char ** e = environ; e && *e; ++e) {
                const char * eq = strchr(*e, '=');
                if (eq) out.insert(std::string(*e, (size_t) (eq - *e)));
            }
        };
        auto added_since = [&](const std::set<std::string> & before, std::vector<std::string> & added) {
            std::set<std::string> now;
            env_names(now);
            added.clear();
            for (const auto & n : now) if (!before.count(n) && n != "PXA_OVERDRIVE_LIB") added.push_back(n);   // the alias itself is set by the test
        };
        // the alias (PXA_OVERDRIVE_LIB): unset / =0 -> nothing; =1 -> only library-only rows of the set, each at the
        // recipe value; a value already in the environment (empty included) is kept, as the launcher's user-wins merge kept it
        {
            unsetenv("PXA_OVERDRIVE_LIB");
            std::set<std::string> before;
            env_names(before);
            std::vector<std::string> added;
            ggml_pxqn_overdrive_alias_apply();
            added_since(before, added);
            bool alias_ok = added.empty();
            setenv("PXA_OVERDRIVE_LIB", "0", 1);
            ggml_pxqn_overdrive_alias_apply();
            added_since(before, added);
            alias_ok = alias_ok && added.empty();
            setenv("PXA_OVERDRIVE_LIB", "1", 1);
            ggml_pxqn_overdrive_alias_apply();
            added_since(before, added);
            alias_ok = alias_ok && (lib ? added.size() >= 2 : added.empty());
            for (const auto & n : added) {
                const char * v = getenv(n.c_str());
                alias_ok = alias_ok && ggml_pxqn_overdrive_is_member(n.c_str()) && pxa_lever_is_closed(n) && v && strcmp(v, "1") == 0;
            }
            if (added.size() >= 2) {
                setenv(added[0].c_str(), "0", 1);
                setenv(added[1].c_str(), "", 1);
                for (size_t i = 2; i < added.size(); ++i) unsetenv(added[i].c_str());
                ggml_pxqn_overdrive_alias_apply();
                const char * v0 = getenv(added[0].c_str());
                const char * v1 = getenv(added[1].c_str());
                alias_ok = alias_ok && v0 && strcmp(v0, "0") == 0 && v1 && v1[0] == '\0';
                for (size_t i = 2; i < added.size(); ++i) {
                    const char * v = getenv(added[i].c_str());
                    alias_ok = alias_ok && v && strcmp(v, "1") == 0;
                }
            }
            unsetenv("PXA_OVERDRIVE_LIB");
            for (const auto & n : added) unsetenv(n.c_str());
            alias_ok = alias_ok && !ggml_pxqn_overdrive_is_member("PXA_AUTO_OVERDRIVE") && !ggml_pxqn_overdrive_is_member("");
            printf("%-44s %-12s %s\n", "PXA_OVERDRIVE_LIB: library rows only, user wins", "-", alias_ok ? "PASS" : "FAIL");
            g_fail += !alias_ok;
        }
        auto fnm = mk("qwen4exp", "PXQ_UNIVERSAL", 512, 2);
        fnm.n_nextn = 1;
        fnm.bytes = 58ull << 30;                          // fn32 + the PLE table: larger than any one or two 16 GiB cards
        auto fnn2 = fnm; fnn2.tier = "PXQN2";
        auto fnq4 = fnm; fnq4.tier = "PXQ4";              // not a tier the set was measured on
        auto fnnomtp = fnm; fnnomtp.n_nextn = 0;           // no MTP head
        auto fnsmall = fnm; fnsmall.bytes = 20ull << 30;   // fits two 16 GiB cards
        pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        pxa_user_set sm_l = none; sm_l.sm = true; sm_l.sm_value = PXA_SM_LAYER;
        struct { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level; const char * off; bool want; } rows[] = {
            {"1x P100 Flash-Next mixed (measured preset)", "1x600:16384", &fnm,     &none, 2, nullptr, true },
            {"1x V100 Flash-Next PXQN2",                   "1x700:16384", &fnn2,    &none, 2, nullptr, true },
            {"2x V100 Flash-Next mixed -sm layer",         "2x700:16384", &fnm,     &sm_l, 2, nullptr, true },
            {"2x P100 Flash-Next mixed -sm layer",         "2x600:16384", &fnm,     &sm_l, 2, nullptr, true },
            // the P100 pair's tensor split is the registry's own default split for this class there and takes the set (2026-10-20);
            // a tensor split elsewhere keeps its own env
            {"2x P100 Flash-Next mixed -sm tensor",        "2x600:16384", &fnm,     &sm_t, 2, nullptr, true },
            {"2x V100 Flash-Next mixed -sm tensor",        "2x700:16384", &fnm,     &sm_t, 2, nullptr, false},
            {"1x P100 Flash-Next PXA_AUTO_OVERDRIVE=0",    "1x600:16384", &fnm,     &none, 2, "PXA_AUTO_OVERDRIVE", false},
            {"1x P100 Flash-Next PXA_NO_OVERDRIVE=1",      "1x600:16384", &fnm,     &none, 2, "PXA_NO_OVERDRIVE",   false},
            {"1x P100 Flash-Next PXA_OVERDRIVE=0",         "1x600:16384", &fnm,     &none, 2, "PXA_OVERDRIVE",      false},
            {"1x P100 Flash-Next DEFAULT level",           "1x600:16384", &fnm,     &none, 1, nullptr, false},
            {"1x P100 Flash-Next PXQ4 tier",               "1x600:16384", &fnq4,    &none, 2, nullptr, false},
            {"1x P100 Flash-Next without MTP head",        "1x600:16384", &fnnomtp, &none, 2, nullptr, false},
            {"2x P100 20 GiB file (fits)",                 "2x600:16384", &fnsmall, &sm_l, 2, nullptr, false},
            {"1x P100 VRAM unknown",                       "1x600",       &fnm,     &none, 2, nullptr, false},
            {"1x P100 35B qwen35moe",                      "1x600:16384", &moe,     &none, 2, nullptr, false},
        };
        for (const auto & r : rows) {
            const bool want = r.want && lib;
            if (r.off) setenv(r.off, std::string(r.off) == "PXA_NO_OVERDRIVE" ? "1" : "0", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, *r.u, r.level, 0);
            if (r.off) unsetenv(r.off);
            const pxa_pick * po = pick(ac, "PXA_AUTO_OVERDRIVE");
            std::vector<std::string> members;   // the library-only rows the set put into the env
            for (const auto & e : ac.env) {
                if (ggml_pxqn_overdrive_is_member(e.first.c_str())) members.push_back(e.first);
            }
            const char * pl = env_of(ac, "PXA_XCACHE_CPUPOOL_CPUS");   // derived from this process's allowed CPUs (>= 4 on any CI box)
#if !defined(__linux__)
            // od_pool_cpus() reads the Linux cpuset and is omitted elsewhere. A missing pool is correct there.
            const bool pool_ok = pl == nullptr || (*pl && std::count(pl, pl + strlen(pl), ',') <= 15);
#else
            const bool pool_ok = pl && *pl && std::count(pl, pl + strlen(pl), ',') <= 15;
#endif
            const bool on  = po && po->applied && po->status == "MEASURED" && ev_is(po->evidence, "") &&
                             members.size() >= 2 && pool_ok;
            const bool off = po == nullptr && members.empty() && pl == nullptr;
            // the user-facing log and the EXPLAIN json never name a library-only row (2026-10-18): the banner counts
            // them, the json carries the alias PXA_OVERDRIVE_LIB=1 instead
            bool quiet_ok = true;
            if (want) {
                const std::string js = pxa_autoconfig_json(pxa_topology_parse(r.topo), *r.m, ac, r.level);
                quiet_ok = js.find("\"PXA_OVERDRIVE_LIB\":\"1\"") != std::string::npos;
                for (const auto & c : members) quiet_ok = quiet_ok && js.find(c) == std::string::npos;
                static bool banner_done = false;
                if (!banner_done && !members.empty()) {   // the banner prints once per process
                    banner_done = true;
                    setenv(members.back().c_str(), "0", 1);   // one row kept at the user's value
                    FILE * f = tmpfile();
                    pxa_registry_banner(f, pxa_topology_parse(r.topo), *r.m, ac);
                    unsetenv(members.back().c_str());
                    std::string log;
                    rewind(f);
                    for (int ch; (ch = fgetc(f)) != EOF; ) log += (char) ch;
                    fclose(f);
                    for (const auto & c : members) quiet_ok = quiet_ok && log.find(c) == std::string::npos;
                    const std::string line = "Overdrive: " + std::to_string(members.size() - 1) + " tuned library lever(s) on, 1 kept at your value";
                    quiet_ok = quiet_ok && log.find(line) != std::string::npos;
                    printf("%-44s %-12s %s\n", "banner: library rows counted, not named", r.topo, quiet_ok ? "PASS" : "FAIL");
                }
            }
            const bool ok = want ? (on && quiet_ok) : off;
            printf("%-44s %-12s Overdrive %-3s %s\n", r.name, r.topo, on ? "on" : "off", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }

    // ---- bug #207: 1x V100, q8_0 KV, long -c -> -ub 1024 (the Volta MMA q8 route keeps its margin) ----
    {
        unsetenv("PXA_AUTO_UB_VOLTA_Q8");
        pxa_user_set q8l; q8l.ctx = true; q8l.n_ctx_value = 65536; q8l.kv_q8 = true;
        pxa_user_set q8s = q8l; q8s.n_ctx_value = 32768;
        pxa_user_set f16l = q8l; f16l.kv_q8 = false;
        pxa_user_set q8u = q8l; q8u.ub = true;
        expect("1x V100 27B PXQ3 q8 -c 65536 -> ub 1024",  "1x700", q27p3, q8l,  2, PXA_SM_LAYER, 0, 1024, "layer");
        expect("1x V100 27B PXQ3 q8 -c 32768 (ladder)",    "1x700", q27p3, q8s,  2, PXA_SM_LAYER, 0, 0,    "layer");
        expect("1x V100 27B PXQ3 f16 KV -c 65536 (ladder)", "1x700", q27p3, f16l, 2, PXA_SM_LAYER, 0, 0,    "layer");
        expect("1x V100 27B PXQ3 q8 -c 65536 -ub given",   "1x700", q27p3, q8u,  2, PXA_SM_LAYER, 0, 0,    "layer");
        expect("1x P100 27B PXQ3 q8 -c 65536 (ladder)",    "1x600", q27p3, q8l,  2, PXA_SM_LAYER, 0, 0,    "layer");
        expect("1x V100 27B PXQ3 q8 -c 65536 DEFAULT",     "1x700", q27p3, q8l,  1, PXA_SM_KEEP,  0, 0,    "layer");
        setenv("PXA_AUTO_UB_VOLTA_Q8", "0", 1);
        expect("1x V100 q8 -c 65536 PXA_AUTO_UB_VOLTA_Q8=0", "1x700", q27p3, q8l, 2, PXA_SM_LAYER, 0, 0,   "layer");
        unsetenv("PXA_AUTO_UB_VOLTA_Q8");
        const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), q27p3, q8l, 2, 0);
        const pxa_pick * ub = pick(ac, "-ub");
        const bool ok = ub && ub->value == "1024" && ub->status == "RULE" && ub->applied && !ac.ub_adaptive &&
                        ac.ub_rule != nullptr;
        printf("%-44s %-10s %s\n", "1x V100 q8 long: -ub pick is RULE #207", "1x700", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- an ENGINE-picked -ub has to fit (bug #207 follow-up, ws6-fix) ------------------------
    // The walk common.cpp runs at load, driven here by a table of build outcomes (no card, no model).
    // The first table is the rule's own target as the verifier found it: -ub 1024 allocated with
    // 27 MiB left and the server died on its first decode.
    {
        const size_t MiB = 1024ull * 1024ull;
        struct row { int ub; bool alloc; bool oom; size_t free_mib; };
        struct wcase {
            const char * name; int start; int origin; size_t reserve_mib;
            std::vector<row> rows; int want_ub; int want_builds; int want_discards; int floor_ub = 0;
        };
        const std::vector<wcase> cases = {
            { "rule: 1024 leaves 27 MiB -> steps to 512",     1024, PXA_UB_RULE,   512,
              { {1024, true, false, 27}, {768, true, false, 280}, {512, true, false, 532} }, 512, 3, 2 },
            { "rule: 1024 fits -> kept, nothing freed",       1024, PXA_UB_RULE,   512,
              { {1024, true, false, 1631} },                                                1024, 1, 0 },
            { "rule: no rung leaves the reserve -> refusal",  1024, PXA_UB_RULE,   512,
              { {1024, true, false, 27}, {768, true, false, 280}, {512, true, false, 400},
                {256, true, false, 480} },                                                  0,    4, 4 },
            { "rule: a non-memory failure -> no retry",       1024, PXA_UB_RULE,   512,
              { {1024, false, false, 0} },                                                  0,    1, 0 },
            { "rule: reserve 0 (check off) -> kept",          1024, PXA_UB_RULE,   0,
              { {1024, true, false, 27} },                                                  1024, 1, 0 },
            { "ladder: first pick allocates -> kept as before", 2048, PXA_UB_LADDER, 512,
              { {2048, true, false, 27} },                                                  2048, 1, 0 },
            { "ladder: 2048 OOM -> 1024,768 thin -> 512",     2048, PXA_UB_LADDER, 512,
              { {2048, false, true, 0}, {1024, true, false, 27}, {768, true, false, 280},
                {512, true, false, 532} },                                                  512,  4, 2 },
            { "ladder: every rung OOMs -> refusal",           2048, PXA_UB_LADDER, 512,
              { {2048, false, true, 0}, {1024, false, true, 0}, {768, false, true, 0},
                {512, false, true, 0}, {256, false, true, 0} },                             0,    5, 0 },
            // floor_ub (a FIT walk, PXA_AUTO_UB_DENSE_2048): never below the floor, the floor kept without the reserve
            { "fit floor 512: 2048 fits -> kept",             2048, PXA_UB_FIT,    512,
              { {2048, true, false, 700} },                                                 2048, 1, 0, 512 },
            { "fit floor 512: 2048 OOM, 1024/768 thin -> 512", 2048, PXA_UB_FIT,   512,
              { {2048, false, true, 0}, {1024, true, false, 300}, {768, true, false, 400},
                {512, true, false, 100} },                                                  512,  4, 2, 512 },
            { "fit floor 512: floor OOM -> refusal, no 256",  2048, PXA_UB_FIT,    512,
              { {2048, false, true, 0}, {1024, false, true, 0}, {768, false, true, 0},
                {512, false, true, 0} },                                                    0,    4, 0, 512 },
            { "fit floor 600: 768 -> clamps to the floor",    1024, PXA_UB_FIT,    512,
              { {1024, true, false, 10}, {768, true, false, 20}, {600, true, false, 30} },  600,  3, 2, 600 },
            // measured fit math: 27B dense PXQN4 on one 16 GB V100, llama-bench pp at -c 4k/16k/32k (logs 2026-10-18,
            // free MiB as printed by the walk; bench reserve 256, server reserve 512)
            { "27B V100 4k: 2048 leaves 819 -> 2048",         2048, PXA_UB_FIT,    256,
              { {2048, true, false, 819} },                                                 2048, 1, 0, 512 },
            { "27B V100 16k: 2048 OOM, 1024 leaves 385 -> 1024", 2048, PXA_UB_FIT, 256,
              { {2048, false, true, 0}, {1024, true, false, 385} },                         1024, 2, 0, 512 },
            { "27B V100 32k: 2048 OOM, 1024 leaves 469 -> 1024", 2048, PXA_UB_FIT, 256,
              { {2048, false, true, 0}, {1024, true, false, 469} },                         1024, 2, 0, 512 },
            { "27B V100 16k at reserve 512: 1024/768 thin -> 512", 2048, PXA_UB_FIT, 512,
              { {2048, false, true, 0}, {1024, true, false, 385}, {768, true, false, 469},
                {512, true, false, 179} },                                                  512,  4, 2, 512 },
        };
        for (const auto & c : cases) {
            int builds = 0, discards = 0;
            bool order_ok = true;
            std::string trail;
            auto build = [&](int ub) {
                pxa_ub_attempt a;
                if (builds >= (int) c.rows.size() || c.rows[builds].ub != ub) {
                    order_ok = false;
                    builds++;
                    return a;
                }
                const row & r = c.rows[builds++];
                a.allocated = r.alloc; a.alloc_failure = r.oom; a.free_min = r.free_mib * MiB; a.worst_dev = 0;
                return a;
            };
            auto discard = [&]() { discards++; };
            const int got = pxa_registry_ub_walk(c.start, c.origin, c.reserve_mib * MiB, build, discard, &trail, c.floor_ub);
            const bool ok = order_ok && got == c.want_ub && builds == c.want_builds && discards == c.want_discards;
            printf("%-44s %-10s ub=%-5d builds=%d discards=%d %s\n", c.name, "walk", got, builds, discards,
                   ok ? "PASS" : "FAIL");
            if (!ok) {
                printf("    want ub=%d builds=%d discards=%d order_ok=%d; trail: %s\n", c.want_ub, c.want_builds,
                       c.want_discards, (int) order_ok, trail.c_str());
                g_fail++;
            }
        }
        const bool rungs = pxa_registry_ub_next_rung(4096) == 2048 && pxa_registry_ub_next_rung(2048) == 1024 &&
                           pxa_registry_ub_next_rung(1024) == 768 && pxa_registry_ub_next_rung(768) == 512 &&
                           pxa_registry_ub_next_rung(512) == 256 && pxa_registry_ub_next_rung(256) == 0 &&
                           pxa_registry_ub_next_rung(1000) == 768 && pxa_registry_ub_next_rung(64) == 0;
        printf("%-44s %-10s %s\n", "ub ladder rungs 2048>1024>768>512>256", "", rungs ? "PASS" : "FAIL");
        g_fail += !rungs;
        unsetenv("PXA_AUTO_UB_RESERVE_MB");
        const bool r_def = pxa_registry_ub_reserve_bytes() == 512 * MiB;
        setenv("PXA_AUTO_UB_RESERVE_MB", "0", 1);
        const bool r_off = pxa_registry_ub_reserve_bytes() == 0;
        setenv("PXA_AUTO_UB_RESERVE_MB", "1024", 1);
        const bool r_set = pxa_registry_ub_reserve_bytes() == 1024 * MiB;
        unsetenv("PXA_AUTO_UB_RESERVE_MB");
        printf("%-44s %-10s %s\n", "PXA_AUTO_UB_RESERVE_MB default 512, 0 off", "", r_def && r_off && r_set ? "PASS" : "FAIL");
        g_fail += !(r_def && r_off && r_set);
    }
    // ---- the posture: max (fa off) keeps layer -----------------------------------------------
    {
        const pxa_topology t = pxa_topology_parse("2x700");
        const pxa_autoconfig ac = pxa_autoconfig_resolve(t, q27, none, 2, 1);
        const bool ok = ac.split == PXA_SM_LAYER && ac.flash_attn == 0;
        printf("%-44s %-10s %s\n", "2x V100 PXA_MODE=max -> fa off, layer", "2x700", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- -ngl: every layer on the cards when unset, the user's value otherwise -----------------
    {
        const pxa_topology t = pxa_topology_parse("1x600");
        pxa_user_set u; u.ngl = true;
        const bool ok = pxa_autoconfig_resolve(t, q27, none, 2, 0).n_gpu_layers == 999 &&
                        pxa_autoconfig_resolve(t, q27, u, 2, 0).n_gpu_layers == -1 &&
                        pxa_autoconfig_resolve(pxa_topology_parse("0x600"), q27, none, 2, 0).n_gpu_layers == -1;
        printf("%-44s %-10s %s\n", "-ngl unset -> 999; given -> kept; no card -> kept", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- the rollback level keeps the old behaviour end to end: no -ngl, no -c pick ------------
    {
        const pxa_topology t = pxa_topology_parse("2x700");
        const pxa_autoconfig ac = pxa_autoconfig_resolve(t, q27, none, 1, 0);
        const bool ok = ac.n_gpu_layers == -1 && ac.n_ctx == 0 && ac.n_batch == 0;
        printf("%-44s %-10s %s\n", "DEFAULT (ENHANCE=0): -ngl/-c/-b not picked", "2x700", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- -c: np * 4096 when unset, capped at the trained window; the user's value kept ---------
    {
        const pxa_topology t = pxa_topology_parse("1x700");
        auto m = q27; m.n_ctx_train = 262144;
        pxa_user_set np4; np4.n_parallel = 4;
        pxa_user_set uc;  uc.ctx = true;
        auto small = q27; small.n_ctx_train = 2048;
        const bool ok = pxa_autoconfig_resolve(t, m, none, 2, 0).n_ctx == 4096 &&
                        pxa_autoconfig_resolve(t, m, np4, 2, 0).n_ctx == 16384 &&
                        pxa_autoconfig_resolve(t, m, uc, 2, 0).n_ctx == 0 &&
                        pxa_autoconfig_resolve(t, small, none, 2, 0).n_ctx == 2048;
        printf("%-44s %-10s %s\n", "-c unset -> np*4096 (capped); given -> kept", "1x700", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- -c: a covered cell's OWN ctx is used - the same number tools/pxa-launch.py's matching
    //      recipe row passes - instead of the np*4096 formula, and it is NEVER smaller than the
    //      -b just picked (#8382 problem 1: a bare 2x P100 boot picked -c 4096 here, np defaulting
    //      to 1 with no -np given, while pxa-launch passed -c 32768 for the identical cards and
    //      file, and the unmatched 4096 silently clamped the -b 8192 pick down to 4096) ----------
    {
        const pxa_autoconfig ac2p    = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), q27, none, 2, 0);
        const pxa_autoconfig acflag  = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), moe, none, 2, 0);
        const pxa_autoconfig ac1080  = pxa_autoconfig_resolve(pxa_topology_parse("1x610"), l1080, none, 2, 0);
        const bool ok = ac2p.n_ctx == 32768 && ac2p.n_ctx >= ac2p.n_batch &&      // 2xp100-dense-pxq4
                        acflag.n_ctx == 8192 && acflag.n_ctx >= acflag.n_batch && // 2xpair-flagship-moe
                        ac1080.n_ctx == 8192 && ac1080.n_ctx >= ac1080.n_batch;   // 1x1080ti-pxq2
        printf("%-44s %-10s %s\n", "-c: cell-sourced, matches the launcher recipe", "2x600/1x610", ok ? "PASS" : "FAIL");
        if (!ok) {
            printf("    2xP100 dense ctx=%d(want 32768) b=%d; flagship ctx=%d(want 8192) b=%d; "
                   "1080Ti ctx=%d(want 8192) b=%d\n",
                   ac2p.n_ctx, ac2p.n_batch, acflag.n_ctx, acflag.n_batch, ac1080.n_ctx, ac1080.n_batch);
        }
        g_fail += !ok;
    }
    {
        // the general form: over every topology this table has a cell for, -c is never smaller
        // than the -b/-ub cell just picked, cell-sourced -c or formula fallback alike
        struct { const char * topo; const pxa_model_info * m; } covered[] = {
            {"1x600", &pxqu}, {"1x700", &pxqu}, {"2x600", &q27}, {"2x700", &q27}, {"2x600", &moe},
            {"2x700", &moe},  {"1x610", &l1080}, {"1x700", &g4}, {"2x700", &g4}, {"2x600", &g4},
            {"4x600", &q27},
        };
        bool ok = true;
        for (const auto & c : covered) {
            const pxa_autoconfig a = pxa_autoconfig_resolve(pxa_topology_parse(c.topo), *c.m, none, 2, 0);
            const bool row_ok = a.n_batch == 0 || a.n_ctx >= a.n_batch;
            if (!row_ok) {
                printf("    %s ctx=%d < b=%d\n", c.topo, a.n_ctx, a.n_batch);
            }
            ok = ok && row_ok;
        }
        printf("%-44s %-10s %s\n", "-c >= -b on every covered cell (never clamps it)", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- the gemma4 cell's status follows the file's tier, not a blanket MEASURED (#8382 problem
    //      2: "the rule ignores tier"). Matches tools/pxa-launch.py's GEMMA4_CELLS exactly: the
    //      PXQ3/PXQ_UNIVERSAL pair rows are INFERRED ("the pair was not booted on the PXQ3 file");
    //      the same tier on ONE card, and any other tier on a pair, is MEASURED -----------------
    {
        struct { const char * topo; const char * tier; const char * want; } g4st[] = {
            {"1x700", "PXQ_UNIVERSAL", "MEASURED"},   // one V100, our PXQ3 file: measured 2026-09-20
            {"1x700", "PXQ3",          "MEASURED"},   // one V100, a pure PXQ3 tensor set: same boot
            {"2x700", "PXQ_UNIVERSAL", "INFERRED"},   // the pair was NOT booted on this tier
            {"2x600", "PXQ_UNIVERSAL", "INFERRED"},
            {"2x700", "PXQ3",          "INFERRED"},
            {"2x700", "",              "MEASURED"},   // the Google QAT q4_0 file: no PXQ tensor type
            {"2x600", "",              "MEASURED"},
        };
        bool ok = true;
        for (const auto & c : g4st) {
            const auto m = mk("gemma4", c.tier, 128, 8);
            int b, ub, cx; const char * why; const char * st = nullptr;
            pxa_registry_batch_cell(pxa_topology_parse(c.topo), m, 2, &b, &ub, &why, &st, &cx);
            const bool row_ok = st && std::string(st) == c.want;
            if (!row_ok) {
                printf("    %s tier=%-14s got %-9s want %s\n", c.topo, c.tier, st ? st : "(null)", c.want);
            }
            ok = ok && row_ok;
        }
        printf("%-44s %-10s %s\n", "gemma4 cell status is tier-aware (fix 2)", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- gemma4 on ONE 16 GB card: the measured cell's -ub is the FLOOR of a fit walk from -ub 1024
    //      (2026-10-16). The cell itself (n_ubatch) is unchanged; the -ub pick
    //      says "fit 1024>N" (not a number, so the launcher leaves -ub to the engine's walk at load) -------
    {
        unsetenv("PXA_AUTO_UB_FASTEST");
        struct { const char * name; const char * topo; const pxa_model_info * m; int level; bool user_ub;
                 int want_fit; int want_ub; const char * want_value; } fr[] = {
            {"1x P100 gemma4 expert: fit 1024>256",  "1x600", &g4,  2, false, 1024, 256, "fit 1024>256"},
            {"1x V100 gemma4 expert: fit 1024>512",  "1x700", &g4,  2, false, 1024, 512, "fit 1024>512"},
            {"2x V100 gemma4: no fit rule (pairs)",  "2x700", &g4,  2, false, 0,    512, "512"},
            {"2x P100 gemma4: no fit rule (pairs)",  "2x600", &g4,  2, false, 0,    512, "512"},
            {"1x P100 27B: no fit rule (not gemma4)", "1x600", &q27, 2, false, 0,   0,   nullptr},
            {"1x P100 gemma4 DEFAULT level: none",   "1x600", &g4,  1, false, 0,    0,   nullptr},
            {"1x P100 gemma4 -ub given: none",       "1x600", &g4,  2, true,  0,    256, nullptr},
        };
        bool ok = true;
        for (const auto & r : fr) {
            pxa_user_set u;
            u.ub = r.user_ub;
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, u, r.level, 0);
            int b = 0, ub = 0, cx = 0; const char * why = nullptr;
            const int cell = pxa_registry_batch_cell(pxa_topology_parse(r.topo), *r.m, r.level, &b, &ub, &why, nullptr, &cx);
            const int fit  = r.user_ub ? 0 : pxa_registry_ub_fit_rule(pxa_topology_parse(r.topo), *r.m, r.level, cell ? ub : 0, nullptr);
            const pxa_pick * pu = pick(ac, "-ub");
            bool row_ok = fit == r.want_fit && ac.ub_fit_ceiling == r.want_fit && ac.n_ubatch == r.want_ub &&
                          (r.want_value == nullptr || (pu && pu->value == r.want_value)) &&
                          (r.want_fit == 0 || (pu && pu->status == "RULE" && pu->applied));
            printf("%-44s %-10s fit=%-5d ub=%-5d value=%-14s %s\n", r.name, r.topo, fit, ac.n_ubatch,
                   pu ? pu->value.c_str() : "?", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        // set by hand, PXA_AUTO_UB_FASTEST owns the walk: =0 keeps the measured cell, =1 is the generic walk
        for (const char * v : {"0", "1"}) {
            setenv("PXA_AUTO_UB_FASTEST", v, 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x600"), g4, none, 2, 0);
            const pxa_pick * pu = pick(ac, "-ub");
            const bool row_ok = ac.ub_fit_ceiling == 0 && ac.n_ubatch == 256 && pu && pu->value == "256" &&
                                pxa_registry_ub_fit_rule(pxa_topology_parse("1x600"), g4, 2, 256, nullptr) == 0;
            printf("%-44s %-10s %s\n", (std::string("1x P100 gemma4 PXA_AUTO_UB_FASTEST=") + v + ": no fit rule").c_str(),
                   "1x600", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        unsetenv("PXA_AUTO_UB_FASTEST");
        // PXQN4 on 1x V100 / 1x P100 with an f16 KV cache: the ceiling is 2048 (2026-10-17 V100, 2026-10-18 P100); a
        // quantized KV cache, another tier or PXA_AUTO_UB_GEMMA_2048=0 keep 1024 (the switch latches at
        // first use, so its =0 path is the run-time env, not tested here)
        {
            const auto g4n = mk("gemma4", "PXQN4", 128, 8);
            const auto g4q = mk("gemma4", "PXQ4",  128, 8);
            struct { const char * name; const char * topo; const pxa_model_info * m; bool kvq; int want; } gr[] = {
                {"1x V100 gemma4 PXQN4 f16 KV: fit 2048",  "1x700", &g4n, false, 2048},
                {"1x V100 gemma4 PXQN4 q8 KV: fit 1024",   "1x700", &g4n, true,  1024},
                {"1x P100 gemma4 PXQN4 f16 KV: fit 2048",  "1x600", &g4n, false, 2048},
                {"1x P100 gemma4 PXQN4 q8 KV: fit 1024",   "1x600", &g4n, true,  1024},
                {"1x P100 gemma4 PXQ4 f16 KV: fit 1024",   "1x600", &g4q, false, 1024},
                {"1x V100 gemma4 PXQ4 f16 KV: fit 1024",   "1x700", &g4q, false, 1024},
            };
            for (const auto & r : gr) {
                int b = 0, ub = 0, cx = 0; const char * why = nullptr;
                const int cell = pxa_registry_batch_cell(pxa_topology_parse(r.topo), *r.m, 2, &b, &ub, &why, nullptr, &cx);
                const int fit  = pxa_registry_ub_fit_rule(pxa_topology_parse(r.topo), *r.m, 2, cell ? ub : 0, nullptr, r.kvq);
                const bool row_ok = fit == r.want;
                printf("%-44s %-10s fit=%-5d cell_ub=%-5d %s\n", r.name, r.topo, fit, ub, row_ok ? "PASS" : "FAIL");
                ok = ok && row_ok;
            }
        }
        printf("%-44s %-10s %s\n", "gemma4 one-card -ub fit rule", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_AUTO_UB_DENSE_2048 (2026-10-18): the 27B dense PXQN4 file on ONE V100, f16 KV, -ub and -b
    //      unset: -ub 2048 is a ceiling walked down to the stock 512 (llama-bench and the server) -------------------------
    {
        bool ok = true;
        const auto d27  = mk("qwen35",    "PXQN4", 0, 4);
        const auto d27q = mk("qwen35",    "PXQ4",  0, 4);
        const auto moe  = mk("qwen35moe", "PXQN4", 256, 2);
        auto d9 = mk("qwen35", "PXQN4", 0, 4); d9.n_layer = 32;
        auto d65 = mk("qwen35", "PXQN4", 0, 4); d65.n_layer = 65;   // the shipped file: 64 blocks + 1 nextn block
        auto d66 = mk("qwen35", "PXQN4", 0, 4); d66.n_layer = 66;
        struct { const char * name; const char * topo; const pxa_model_info * m; int level; bool kvq; int want; } dr[] = {
            {"dense27 PXQN4 1x V100 f16 KV: 2048",     "1x700", &d27,  2, false, 2048},
            {"dense27 PXQN4 1x V100 q8 KV: none",      "1x700", &d27,  2, true,  0},
            {"dense27 PXQN4 1x V100 DEFAULT level: none", "1x700", &d27, 1, false, 0},
            {"dense27 PXQN4 1x P100 f16 KV: 2048 (#23320)", "1x600", &d27, 2, false, 2048},
            {"dense27 PXQN4 1x P100 q8 KV: none",      "1x600", &d27,  2, true,  0},
            {"dense27 PXQN4 2x P100: none",            "2x600", &d27,  2, false, 0},
            {"dense27 PXQN4 1x P40 (sm_61): none",     "1x610", &d27,  2, false, 0},
            {"dense27 + nextn (65 blocks) 1x P100: 2048", "1x600", &d65, 2, false, 2048},
            {"qwen35moe PXQN4 1x P100: none (experts)","1x600", &moe,  2, false, 0},
            {"dense27 PXQN4 2x V100: none",            "2x700", &d27,  2, false, 0},
            {"dense27 PXQ4 1x V100: none (tier)",      "1x700", &d27q, 2, false, 0},
            {"qwen35moe PXQN4 1x V100: none (experts)","1x700", &moe,  2, false, 0},
            {"dense 32-block PXQN4 1x V100: none",     "1x700", &d9,   2, false, 0},
            {"dense27 + nextn (65 blocks) 1x V100: 2048", "1x700", &d65, 2, false, 2048},
            {"dense 66-block PXQN4 1x V100: none",     "1x700", &d66,  2, false, 0},
        };
        for (const auto & r : dr) {
            const int got = pxa_registry_ub_fit_dense(pxa_topology_parse(r.topo), *r.m, r.level, r.kvq, nullptr);
            const bool row_ok = got == r.want;
            printf("%-44s %-10s fit=%-5d %s\n", r.name, r.topo, got, row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        // the off switch and a hand-set PXA_AUTO_UB_FASTEST stand it down (read per call, not latched)
        for (const char * k : {"PXA_AUTO_UB_DENSE_2048", "PXA_AUTO_UB_FASTEST"}) {
            setenv(k, "0", 1);
            const int got = pxa_registry_ub_fit_dense(pxa_topology_parse("1x700"), d27, 2, false, nullptr);
            unsetenv(k);
            printf("%-44s %-10s fit=%-5d %s\n", (std::string(k) + "=0: none").c_str(), "1x700", got, got == 0 ? "PASS" : "FAIL");
            ok = ok && got == 0;
        }
        setenv("PXA_AUTO_UB_DENSE_2048", "0", 1);
        {
            const int got = pxa_registry_ub_fit_dense(pxa_topology_parse("1x600"), d27, 2, false, nullptr);
            printf("%-44s %-10s fit=%-5d %s\n", "PXA_AUTO_UB_DENSE_2048=0 (P100): none", "1x600", got, got == 0 ? "PASS" : "FAIL");
            ok = ok && got == 0;
        }
        setenv("PXA_AUTO_UB_DENSE_2048", "1", 1);
        ok = ok && pxa_registry_ub_fit_dense(pxa_topology_parse("1x700"), d27, 2, false, nullptr) == 2048;
        ok = ok && pxa_registry_ub_fit_dense(pxa_topology_parse("1x600"), d27, 2, false, nullptr) == 2048;
        unsetenv("PXA_AUTO_UB_DENSE_2048");
        // the resolver: -ub is a RULE pick "fit 2048>512" with no -ub and no -b; either flag given = no rule
        {
            pxa_user_set u0;
            // the engine's own KV pick for this file on one 16 GB card is quantized (the server default): the f16-KV
            // rule stands down there; PXA_KV_AUTO_OFF=1 keeps f16 KV and the rule fires
            const pxa_autoconfig acq = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), d27, u0, 2, 0);
            const pxa_pick * puq = pick(acq, "-ub");
            const bool rq = acq.ub_fit_ceiling == 0 && acq.kv_type != nullptr && strcmp(acq.kv_type, "f16") != 0 &&
                            puq && puq->status == "OFF" && !puq->applied;
            setenv("PXA_KV_AUTO_OFF", "1", 1);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), d27, u0, 2, 0);
            unsetenv("PXA_KV_AUTO_OFF");
            const pxa_pick * pu = pick(ac, "-ub");
            const bool r0 = rq && ac.ub_fit_ceiling == 2048 && pu && pu->value == "fit 2048>512" && pu->status == "RULE" && pu->applied;
            pxa_user_set ub_set; ub_set.ub = true;
            const pxa_autoconfig ac1 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), d27, ub_set, 2, 0);
            const pxa_pick * pu1 = pick(ac1, "-ub");
            const bool r1 = ac1.ub_fit_ceiling == 0 && pu1 && pu1->status == "USER";
            pxa_user_set b_set; b_set.b = true;
            const pxa_autoconfig ac2 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), d27, b_set, 2, 0);
            const bool r2 = ac2.ub_fit_ceiling == 0;
            pxa_user_set kv_set; kv_set.kv = true; kv_set.kv_q8 = true;
            const pxa_autoconfig ac3 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), d27, kv_set, 2, 0);
            const bool r3 = ac3.ub_fit_ceiling == 0;
            printf("%-44s %-10s %s %s %s %s\n", "dense27 resolve: q4 pick+rule / -ub / -b / q8", "1x700",
                   r0 ? "PASS" : "FAIL", r1 ? "PASS" : "FAIL", r2 ? "PASS" : "FAIL", r3 ? "PASS" : "FAIL");
            ok = ok && r0 && r1 && r2 && r3;
            // one P100 (#23320): with f16 KV the resolver names the same ceiling and the P100 evidence row
            setenv("PXA_KV_AUTO_OFF", "1", 1);
            const pxa_autoconfig ap = pxa_autoconfig_resolve(pxa_topology_parse("1x600"), d27, u0, 2, 0);
            unsetenv("PXA_KV_AUTO_OFF");
            const pxa_pick * pup = pick(ap, "-ub");
            const bool rp = ap.ub_fit_ceiling == 2048 && pup && pup->value == "fit 2048>512" && pup->status == "RULE" && pup->applied &&
                            ev_is(pup->evidence, "");
            printf("%-44s %-10s ceiling=%-5d %s\n", "dense27 resolve 1x P100 f16 KV: fit 2048>512", "1x600", ap.ub_fit_ceiling,
                   rp ? "PASS" : "FAIL");
            ok = ok && rp;
        }
        // the Nemotron-H MoE verify-identity pair is registry-scoped to that arch: other archs keep their verify paths
        {
            auto has = [](const pxa_autoconfig & ac, const char * k) {
                for (const auto & e : ac.env) if (e.first == k) return true;
                return false;
            };
            pxa_user_set u0;
            const pxa_model_info nemo = mk("nemotron_h_moe", "PXQN2", 512, 2);
            const pxa_autoconfig n1 = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), nemo, u0, 2, 0);
            const pxa_autoconfig n2 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), nemo, u0, 2, 0);
            // a dense qwen35 on V100: the nemo rule does not reach it (on sm_60 the qwen35 rule below sets the same pair)
            const pxa_autoconfig n3 = pxa_autoconfig_resolve(pxa_topology_parse("2x700"), d27, u0, 2, 0);
            const pxa_autoconfig n4 = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), nemo, u0, 0, 0);
            const bool q1 = has(n1, "PXA_FA_VEC_NY1_SPLIT") && has(n1, "PXA_KQMMV_VERIFY_ID_SMALL");
            const bool q2 = has(n2, "PXA_FA_VEC_NY1_SPLIT") && has(n2, "PXA_KQMMV_VERIFY_ID_SMALL");
            const bool q3 = !has(n3, "PXA_FA_VEC_NY1_SPLIT") && !has(n3, "PXA_KQMMV_VERIFY_ID_SMALL");
            const bool q4 = !has(n4, "PXA_FA_VEC_NY1_SPLIT");
            printf("%-44s %-10s %s\n", "nemo verify identity: nemotron_h_moe only", "", q1 && q2 && q3 && q4 ? "PASS" : "FAIL");
            ok = ok && q1 && q2 && q3 && q4;
            // qwen4exp (Hemlock) takes the same on/off switch (option A), not the FA split; the value is "1" (on, not a width)
            const pxa_model_info q4x = mk("qwen4exp", "PXQN2", 512, 2);
            const pxa_autoconfig h1 = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), q4x, u0, 2, 0);
            const pxa_autoconfig h2 = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), q4x, u0, 0, 0);
            auto val = [](const pxa_autoconfig & ac, const char * k) -> std::string {
                for (const auto & e : ac.env) if (e.first == k) return e.second;
                return "";
            };
            const bool r1 = val(h1, "PXA_KQMMV_VERIFY_ID_SMALL") == "1" && !has(h1, "PXA_FA_VEC_NY1_SPLIT") &&
                            val(n1, "PXA_KQMMV_VERIFY_ID_SMALL") == "1" && !has(h1, "PXA_KQMMV_VERIFY_ID_SMALL_W");
            const bool r2 = !has(h2, "PXA_KQMMV_VERIFY_ID_SMALL");
            printf("%-44s %-10s %s\n", "verify-id small switch: nemotron_h_moe+qwen4exp", "", r1 && r2 ? "PASS" : "FAIL");
            ok = ok && r1 && r2;
        }
        // the same pair for qwen35 on sm_60: 2x P100 / 1x P100 / P100+V100 yes, V100 alone no, REFERENCE level no
        {
            auto has = [](const pxa_autoconfig & ac, const char * k) {
                for (const auto & e : ac.env) if (e.first == k) return true;
                return false;
            };
            pxa_user_set u0;
            auto both = [&](const pxa_autoconfig & ac) { return has(ac, "PXA_FA_VEC_NY1_SPLIT") && has(ac, "PXA_KQMMV_VERIFY_ID_SMALL"); };
            auto none = [&](const pxa_autoconfig & ac) { return !has(ac, "PXA_FA_VEC_NY1_SPLIT") && !has(ac, "PXA_KQMMV_VERIFY_ID_SMALL"); };
            const pxa_autoconfig a2p = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), d27, u0, 2, 0);
            const char * w_w = nullptr;
            for (const auto & e : a2p.env) if (e.first == "PXA_KQMMV_VERIFY_ID_SMALL_W") w_w = e.second.c_str();
            const bool w1 = both(a2p) && w_w != nullptr && std::string(w_w) == "4";   // the width the 27B measured, not qwen4exp's 8
            const bool w2 = both(pxa_autoconfig_resolve(pxa_topology_parse("1x600"), d27, u0, 2, 0));
            const bool w3 = both(pxa_autoconfig_resolve(pxa_topology_parse("700,600"), d27, u0, 2, 0));
            const bool w4 = none(pxa_autoconfig_resolve(pxa_topology_parse("2x700"), d27, u0, 2, 0));
            const bool w5 = none(pxa_autoconfig_resolve(pxa_topology_parse("2x600"), d27, u0, 0, 0));
            printf("%-44s %-10s %s\n", "qwen35 verify identity: sm_60 only", "", w1 && w2 && w3 && w4 && w5 ? "PASS" : "FAIL");
            ok = ok && w1 && w2 && w3 && w4 && w5;
        }
        // PXA_AUTO_PF16K now also covers ONE P100 for the dense 27B PXQN4 file (2026-10-18: pp16384 +5.9%)
        {
            auto has = [](const pxa_autoconfig & ac, const char * k) {
                for (const auto & e : ac.env) if (e.first == k) return true;
                return false;
            };
            pxa_user_set u0;
            const pxa_autoconfig a1 = pxa_autoconfig_resolve(pxa_topology_parse("1x600"), d27, u0, 2, 0);
            const pxa_autoconfig a2 = pxa_autoconfig_resolve(pxa_topology_parse("2x600"), d27, u0, 2, 0);
            const pxa_autoconfig a3 = pxa_autoconfig_resolve(pxa_topology_parse("1x700"), d27, u0, 2, 0);
            const pxa_autoconfig a4 = pxa_autoconfig_resolve(pxa_topology_parse("1x600"), d27q, u0, 2, 0);
            const bool p1 = has(a1, "PXA_FA_TILE_H2CHUNK") && has(a1, "PXA_FA_LONGPREFILL");
            const bool p2 = has(a2, "PXA_FA_TILE_H2CHUNK") && has(a2, "PXA_FA_LONGPREFILL");
            const bool p3 = !has(a3, "PXA_FA_LONGPREFILL");
            const bool p4 = !has(a4, "PXA_FA_LONGPREFILL");
            setenv("PXA_AUTO_PF16K", "0", 1);
            const pxa_autoconfig a5 = pxa_autoconfig_resolve(pxa_topology_parse("1x600"), d27, u0, 2, 0);
            unsetenv("PXA_AUTO_PF16K");
            const bool p5 = !has(a5, "PXA_FA_LONGPREFILL");
            printf("%-44s %-10s %s\n", "PF16K dense27: 1x/2x P100 on, V100/PXQ4/=0 off", "", p1 && p2 && p3 && p4 && p5 ? "PASS" : "FAIL");
            ok = ok && p1 && p2 && p3 && p4 && p5;
        }
        printf("%-44s %-10s %s\n", "dense27 one-V100 -ub fit rule", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_AUTO_UB_STREAM_DECODE (2026-10-18, C042): a qwen35moe PXQN4 file on ONE V100 that streams a
    //      small tail (file <= 1.5x free) gets -ub 2048 instead of the stream rule's 4096; elsewhere 0 (the 4096 stands) ----
    {
        bool ok = true;
        const uint64_t MB = 1ull << 20;
        const auto moe  = mk("qwen35moe", "PXQN4", 256, 2);
        const auto moeq = mk("qwen35moe", "PXQ4",  256, 2);
        const auto nxt  = mk("qwen3next", "PXQN4", 512, 2);
        const auto d27  = mk("qwen35",    "PXQN4", 0, 4);
        struct { const char * name; const char * topo; const pxa_model_info * m; uint64_t file, free; int want; } sr[] = {
            {"stream decode: 35B-A3B 1x V100 18617/15827: 2048", "1x700", &moe,  18617*MB, 15827*MB, 2048},
            {"stream decode: fits (file <= free): none",        "1x700", &moe,  15000*MB, 15827*MB, 0},
            {"stream decode: file > 1.5x free: none",           "1x700", &moe,  24000*MB, 15827*MB, 0},
            {"stream decode: file = 1.5x free: 2048",           "1x700", &moe,  3*8000*MB, 2*8000*MB, 2048},
            {"stream decode: 1x P100 (unmeasured): none",       "1x600", &moe,  18617*MB, 15827*MB, 0},
            {"stream decode: 2x V100: none",                    "2x700", &moe,  18617*MB, 15827*MB, 0},
            {"stream decode: PXQ4 tier (unmeasured): none",     "1x700", &moeq, 18617*MB, 15827*MB, 0},
            {"stream decode: Flash-Next qwen3next: none",       "1x700", &nxt,  18617*MB, 15827*MB, 0},
            {"stream decode: dense qwen35: none",               "1x700", &d27,  18617*MB, 15827*MB, 0},
            {"stream decode: free unknown (0): none",           "1x700", &moe,  18617*MB, 0,        0},
        };
        for (const auto & r : sr) {
            const char * why = nullptr;
            const int got = pxa_registry_stream_ub_decode(pxa_topology_parse(r.topo), *r.m, r.file, r.free, &why);
            const bool row_ok = got == r.want && (got == 0 || (why && strstr(why, "PXA_AUTO_UB_STREAM_DECODE=0")));
            printf("%-44s %-10s ub=%-5d %s\n", r.name, r.topo, got, row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        setenv("PXA_AUTO_UB_STREAM_DECODE", "0", 1);
        {
            const int got = pxa_registry_stream_ub_decode(pxa_topology_parse("1x700"), moe, 18617*MB, 15827*MB, nullptr);
            printf("%-44s %-10s ub=%-5d %s\n", "PXA_AUTO_UB_STREAM_DECODE=0: none", "1x700", got, got == 0 ? "PASS" : "FAIL");
            ok = ok && got == 0;
        }
        setenv("PXA_AUTO_UB_STREAM_DECODE", "1", 1);
        ok = ok && pxa_registry_stream_ub_decode(pxa_topology_parse("1x700"), moe, 18617*MB, 15827*MB, nullptr) == 2048;
        unsetenv("PXA_AUTO_UB_STREAM_DECODE");
        printf("%-44s %-10s %s\n", "stream -ub decode-sized rule (C042)", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_STREAM_UB_CKPT (2026-10-18): the stream rule's -ub is a ceiling walked down by 512 (floor 2048)
    //      until the speculative chain keeps its checkpoint rows; the measured Flash-Next 1x V100 shape: 4096 affords 2 of 4 ----
    {
        bool ok = true;
        struct { const char * name; int ub0, cap0, target; std::vector<std::pair<int,int>> caps; int want; std::vector<int> tried; } wr[] = {
            {"ckpt walk: 4096 2/4, 3584 4/4 -> 3584",       4096, 2, 4, {{3584, 4}},                                   3584, {3584}},
            {"ckpt walk: rows fit already -> no rebuild",    4096, 4, 4, {},                                            4096, {}},
            {"ckpt walk: target 1 (no chain) -> none",       4096, 1, 1, {},                                            4096, {}},
            {"ckpt walk: never full -> most rows, larger",   4096, 1, 4, {{3584, 2}, {3072, 3}, {2560, 3}, {2048, 3}}, 3072, {3584, 3072, 2560, 2048}},
            {"ckpt walk: nothing better -> keep 4096",       4096, 2, 4, {{3584, 2}, {3072, 2}, {2560, 2}, {2048, 2}}, 4096, {3584, 3072, 2560, 2048}},
            {"ckpt walk: 3584 does not allocate -> 4096",    4096, 2, 4, {{3584, 0}},                                   4096, {3584}},
            {"ckpt walk: at the floor -> none",              2048, 2, 4, {},                                            2048, {}},
            {"ckpt walk: odd ceiling 3000 -> 2560 first",    3000, 2, 4, {{2560, 4}},                                   2560, {2560}},
        };
        for (const auto & r : wr) {
            std::vector<int> tried;
            auto cap_at = [&](int ub) -> int {
                tried.push_back(ub);
                for (const auto & c : r.caps) if (c.first == ub) return c.second;
                return -1;   // a rung the row did not expect: fails the row through `tried`
            };
            std::string trail;
            const int got = pxa_registry_stream_ub_ckpt_walk(r.ub0, r.cap0, r.target, 2048, cap_at, &trail);
            const bool row_ok = got == r.want && tried == r.tried;
            printf("%-44s %-10s ub=%-5d %s\n", r.name, "", got, row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        const bool unset_off = !pxa_registry_stream_ub_ckpt_on();   // opt-in lever
        setenv("PXA_STREAM_UB_CKPT", "0", 1);
        const bool off = !pxa_registry_stream_ub_ckpt_on();
        setenv("PXA_STREAM_UB_CKPT", "1", 1);
        const bool on1 = pxa_registry_stream_ub_ckpt_on();
        unsetenv("PXA_STREAM_UB_CKPT");
        printf("%-44s %-10s %s\n", "PXA_STREAM_UB_CKPT unset/=0 off, =1 on", "", unset_off && off && on1 ? "PASS" : "FAIL");
        ok = ok && unset_off;
        ok = ok && off && on1;
        printf("%-44s %-10s %s\n", "stream -ub checkpoint-chain walk", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_AUTO_XCACHE_SMALLTAIL (2026-10-19, C042/C114/C115): the same qwen35moe PXQN4 one-V100 small-tail class
    //      (VRAM < file <= 1.5x VRAM) arms the async CPU-pool cold path; nothing elsewhere; =0 off ----
    {
        bool ok = true;
        const uint64_t MB = 1ull << 20;
        auto sized = [](pxa_model_info m, uint64_t b) { m.bytes = b; return m; };
        const auto moe  = sized(mk("qwen35moe", "PXQN4", 256, 2), 18617*MB);
        const auto moef = sized(mk("qwen35moe", "PXQN4", 256, 2), 15000*MB);
        const auto moeb = sized(mk("qwen35moe", "PXQN4", 256, 2), 25000*MB);
        const auto moeq = sized(mk("qwen35moe", "PXQ4",  256, 2), 18617*MB);
        const auto nxt  = sized(mk("qwen3next", "PXQN4", 512, 2), 18617*MB);
        const auto d27  = sized(mk("qwen35",    "PXQN4", 0, 4),   18617*MB);
        struct { const char * name; const char * topo; const pxa_model_info * m; bool want; } sr[] = {
            {"xc small tail: 35B-A3B 1x V100 16 GB: on",     "1x700:16144", &moe,  true},
            {"xc small tail: fits (file <= VRAM): off",      "1x700:16144", &moef, false},
            {"xc small tail: file > 1.5x VRAM: off",         "1x700:16144", &moeb, false},
            {"xc small tail: VRAM unknown: off",             "1x700",       &moe,  false},
            {"xc small tail: 1x P100 (unmeasured): off",     "1x600:16276", &moe,  false},
            {"xc small tail: 2x V100: off",                  "2x700:16144", &moe,  false},
            {"xc small tail: PXQ4 tier (unmeasured): off",   "1x700:16144", &moeq, false},
            {"xc small tail: Flash-Next qwen3next: off",     "1x700:16144", &nxt,  false},
            {"xc small tail: dense qwen35: off",             "1x700:16144", &d27,  false},
        };
        const pxa_user_set none;
        for (const auto & r : sr) {
            const bool got = pxa_registry_xcache_smalltail(pxa_topology_parse(r.topo), *r.m, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, 2, 0);
            const pxa_pick * pk = pick(ac, "PXA_AUTO_XCACHE_SMALLTAIL");
            bool env_async = false, env_pool = false;
            for (const auto & e : ac.env) {
                env_async = env_async || (e.first == "PXA_XCACHE_ASYNC" && e.second == "1");
                env_pool  = env_pool  || (e.first == "PXA_XCACHE_CPUPOOL" && e.second == "1");
            }
            const bool row_ok = got == r.want && (pk != nullptr) == r.want && env_async == r.want && env_pool == r.want;
            printf("%-44s %-12s %-3s %s\n", r.name, r.topo, got ? "on" : "off", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        {   // the reference level (PXA_ENHANCE=0 class) applies no lever
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700:16144"), moe, none, 0, 0);
            const bool row_ok = pick(ac, "PXA_AUTO_XCACHE_SMALLTAIL") == nullptr;
            printf("%-44s %-12s %s\n", "xc small tail: level 0: off", "1x700:16144", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        setenv("PXA_AUTO_XCACHE_SMALLTAIL", "0", 1);
        {
            const bool got = pxa_registry_xcache_smalltail(pxa_topology_parse("1x700:16144"), moe, nullptr);
            printf("%-44s %-12s %s\n", "PXA_AUTO_XCACHE_SMALLTAIL=0: off", "1x700:16144", !got ? "PASS" : "FAIL");
            ok = ok && !got;
        }
        unsetenv("PXA_AUTO_XCACHE_SMALLTAIL");
        printf("%-44s %-12s %s\n", "xcache small-tail async cold path (C042)", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_AUTO_FLINT_1CARD (2026-10-23): a flint PXQN4 file on ONE P100 with VRAM < file <= 1.5x VRAM
    //      sets PXA_XCACHE_SMALLTAIL_SM60=1 (the small-tail plan on sm_60) and nothing else; one V100 is left to the generic
    //      small-tail scope; nothing elsewhere; =0 off ----
    {
        bool ok = true;
        const uint64_t MB = 1ull << 20;
        auto sized = [](pxa_model_info m, uint64_t b) { m.bytes = b; return m; };
        const auto fl   = sized(mk("flint",     "PXQN4", 256, 4), 20561*MB);
        const auto flf  = sized(mk("flint",     "PXQN4", 256, 4), 15000*MB);
        const auto flb  = sized(mk("flint",     "PXQN4", 256, 4), 25000*MB);
        const auto flq  = sized(mk("flint",     "PXQ4",  256, 4), 20561*MB);
        const auto flm  = [&]{ auto m = sized(mk("flint", "PXQ_UNIVERSAL", 256, 4), 23719*MB); m.pxq_mask = (1u << (259-248)) | (1u << (263-248)); return m; }();   // Flint B: PXQN4+PXQN5 experts
        const auto flx  = [&]{ auto m = sized(mk("flint", "PXQ_UNIVERSAL", 256, 4), 23719*MB); m.pxq_mask = (1u << (259-248)) | (1u << (260-248)); return m; }();   // any other PXQ mix
        const auto moe  = sized(mk("qwen35moe", "PXQN4", 256, 2), 18617*MB);
        const auto nxt  = sized(mk("qwen3next", "PXQN4", 512, 2), 20561*MB);
        struct { const char * name; const char * topo; const pxa_model_info * m; bool want; } fr[] = {
            {"flint 1card: 1x P100 16 GB: on",            "1x600:16276", &fl,   true },
            {"flint 1card: 1x V100 (generic scope): off", "1x700:16144", &fl,   false},
            {"flint 1card: fits (file <= VRAM): off",     "1x600:16276", &flf,  false},
            {"flint 1card: file > 1.5x VRAM: off",        "1x600:16276", &flb,  false},
            {"flint 1card: VRAM unknown: off",            "1x600",       &fl,   false},
            {"flint 1card: 2x P100: off",                 "2x600:16276", &fl,   false},
            {"flint 1card: 1x sm_61 (unmeasured): off",   "1x610:24576", &fl,   false},
            {"flint 1card: PXQ4 tier (unmeasured): off",  "1x600:16276", &flq,  false},
            {"flint 1card: B mix (N4+N5, PXQ_UNIVERSAL) P100: on", "1x600:16276", &flm, true },
            {"flint 1card: other PXQ_UNIVERSAL mix: off", "1x600:16276", &flx,  false},
            {"flint 1card: qwen35moe 1x P100: off",       "1x600:16276", &moe,  false},
            {"flint 1card: qwen3next 1x P100: off",       "1x600:16276", &nxt,  false},
        };
        const pxa_user_set none;
        for (const auto & r : fr) {
            const bool got = pxa_registry_flint_1card(pxa_topology_parse(r.topo), *r.m, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, 2, 0);
            const pxa_pick * pk = pick(ac, "PXA_AUTO_FLINT_1CARD");
            bool env_st = false, env_other = false;
            for (const auto & e : ac.env) {
                env_st    = env_st    || (e.first == "PXA_XCACHE_SMALLTAIL_SM60" && e.second == "1");
                env_other = env_other || e.first == "PXA_XCACHE_ASYNC" || e.first == "PXA_XCACHE_COST_LAYER_US";
            }
            const bool row_ok = got == r.want && (pk != nullptr) == r.want && env_st == r.want && (r.m == &moe || !env_other);
            printf("%-44s %-12s %-3s %s\n", r.name, r.topo, got ? "on" : "off", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        {   // the one-P100 K3 block (RMS_RHT_GB + ARGMAX_VAL_MB) is arch-agnostic ON PURPOSE: single-type Flint PXQN4 got it through the old tier test, and
            // Flint B (PXQN4+PXQN5, arch flint) is single-type Flint to this block; any other PXQ mix still gets neither (review finding 1)
            auto k3 = [&](const pxa_model_info & mm) {
                const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x600:16276"), mm, none, 2, 0);
                const char * a = env_of(ac, "PXA_PXQN_RMS_RHT_GB"); const char * b = env_of(ac, "PXA_ARGMAX_VAL_MB");
                return a && b && std::string(a) == "1" && std::string(b) == "1";
            };
            const bool row_ok = k3(flm) && k3(fl) && !k3(flx);
            printf("%-44s %-12s %s\n", "flint 1card: K3 picks on B mix + PXQN4, not other mix", "1x600:16276", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        {   // the reference level (PXA_ENHANCE=0 class) applies no lever
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x600:16276"), fl, none, 0, 0);
            const bool row_ok = pick(ac, "PXA_AUTO_FLINT_1CARD") == nullptr;
            printf("%-44s %-12s %s\n", "flint 1card: level 0: off", "1x600:16276", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        setenv("PXA_AUTO_FLINT_1CARD", "0", 1);
        {
            const bool got = pxa_registry_flint_1card(pxa_topology_parse("1x600:16276"), fl, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x600:16276"), fl, none, 2, 0);
            bool env_any = false;
            for (const auto & e : ac.env) env_any = env_any || e.first == "PXA_XCACHE_SMALLTAIL_SM60";
            const bool row_ok = !got && pick(ac, "PXA_AUTO_FLINT_1CARD") == nullptr && !env_any;
            printf("%-44s %-12s %s\n", "PXA_AUTO_FLINT_1CARD=0: off, no env", "1x600:16276", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        unsetenv("PXA_AUTO_FLINT_1CARD");
        printf("%-44s %-12s %s\n", "flint one-P100 small-tail plan", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- the published OneCard 27B files (Qwen3.8-27B-PXQN-OneCard 8fccd7aa, Swift-1.5 4674a8c8): PXQN4 + PXQN3 + one PXQN5 = tier
    //      PXQ_UNIVERSAL (2026-10-24, PXQ audit). The 27B levers that do not read the FFN weight format take the mix
    //      (dense -ub fit, one-P100 K3 / convfuse stack, RBP, the P100 prompt-attention pair, P2L, P2T); any other mix stays off ----
    {
        bool ok = true;
        auto mixed = [](uint32_t mask) { auto m = mk("qwen35", "PXQ_UNIVERSAL", 0, 4); m.n_layer = 65; m.pxq_mask = mask; return m; };
        const auto oc    = mixed((1u << (259-248)) | (1u << (257-248)) | (1u << (263-248)));   // PXQN4 + PXQN3 + PXQN5: the OneCard map
        const auto oc43  = mixed((1u << (259-248)) | (1u << (257-248)));                       // PXQN4 + PXQN3
        const auto ocn2  = mixed((1u << (259-248)) | (1u << (260-248)));                       // PXQN4 + PXQN2: not measured
        const auto ocq4  = mixed((1u << (259-248)) | (1u << (252-248)));                       // PXQN4 + PXQ4: not measured
        const auto ocn3  = mixed((1u << (257-248)) | (1u << (263-248)));                       // no PXQN4 tensor at all: not measured
        pxa_user_set none; pxa_user_set sm_t = none; sm_t.sm = true; sm_t.sm_value = PXA_SM_TENSOR;
        pxa_user_set sm_l = none; sm_l.sm = true; sm_l.sm_value = PXA_SM_LAYER;
        struct { const char * name; const pxa_model_info * m; bool want; } mm[] = {
            {"OneCard mix PXQN4+N3+N5", &oc, true }, {"PXQN4+PXQN3 mix (no N5 head)", &oc43, true }, {"PXQN4+PXQN2 mix", &ocn2, false},
            {"PXQN4+PXQ4 mix", &ocq4, false}, {"PXQN3+PXQN5 (no N4)", &ocn3, false},
        };
        for (const auto & r : mm) {
            const int u7 = pxa_registry_ub_fit_dense(pxa_topology_parse("1x700:16144"), *r.m, 2, false, nullptr);
            const int u6 = pxa_registry_ub_fit_dense(pxa_topology_parse("1x600:16276"), *r.m, 2, false, nullptr);
            const pxa_autoconfig a1 = pxa_autoconfig_resolve(pxa_topology_parse("1x600:16276"), *r.m, none, 2, 0);
            const pxa_autoconfig al = pxa_autoconfig_resolve(pxa_topology_parse("2x600:16276"), *r.m, sm_l, 2, 0);
            const pxa_autoconfig at = pxa_autoconfig_resolve(pxa_topology_parse("2x600:16276"), *r.m, sm_t, 2, 0);
            const pxa_autoconfig ad = pxa_autoconfig_resolve(pxa_topology_parse("2x600:16276"), *r.m, none, 2, 0);
            auto on1 = [&](const pxa_autoconfig & a, const char * k) { const char * v = env_of(a, k); return v && std::string(v) == "1"; };
            const bool p1 = pick(a1, "PXA_AUTO_P1") != nullptr && on1(a1, "PXA_DN_CONVFUSE") && on1(a1, "PXA_PLAIN_GREEDY_AMAX");
            const bool k3 = on1(a1, "PXA_PXQN_RMS_RHT_GB") && on1(a1, "PXA_ARGMAX_VAL_MB");
            const bool rbp = on1(a1, "PXA_PXQN_RBP");
            const bool pf = on1(a1, "PXA_FA_TILE_H2CHUNK") && env_of(a1, "PXA_FA_LONGPREFILL") != nullptr;
            const bool p2l = pick(al, "PXA_AUTO_P2L") != nullptr && on1(al, "PXA_PXQN_RBP");
            const bool p2t = pick(at, "PXA_AUTO_P2T") != nullptr && on1(at, "PXA_PXQN_RBP_TP");
            const bool pf2 = on1(at, "PXA_FA_TILE_H2CHUNK");
            const bool ub = u7 == 2048 && u6 == 2048;
            // 2026-10-24 (+31% decode on 2x P100): the exact OneCard map PXQN4+N3+N5 takes the tensor split by default on a
            // P100 PAIR; every other mix, and a V100 pair, stays layer; PXA_AUTO_SM_PXQN=0 puts layer back.
            const pxa_autoconfig av = pxa_autoconfig_resolve(pxa_topology_parse("2x700:16144"), *r.m, none, 2, 0);
            setenv("PXA_AUTO_SM_PXQN", "0", 1);
            const pxa_autoconfig a0 = pxa_autoconfig_resolve(pxa_topology_parse("2x600:16276"), *r.m, none, 2, 0);
            unsetenv("PXA_AUTO_SM_PXQN");
            const bool want_ts = r.m == &oc;
            const bool tsplit_untouched = (ad.split == PXA_SM_TENSOR) == want_ts && av.split != PXA_SM_TENSOR && a0.split != PXA_SM_TENSOR;
            const bool all_on = ub && p1 && k3 && rbp && pf && p2l && p2t && pf2;
            const bool all_off = u7 == 0 && u6 == 0 && !p1 && !k3 && !rbp && !pf && !p2l && !p2t && !pf2;
            const bool row_ok = (r.want ? all_on : all_off) && tsplit_untouched;
            printf("%-44s ub %d/%d p1 %d k3 %d rbp %d pf %d p2l %d p2t %d %s\n", r.name, u7, u6, p1, k3, rbp, pf, p2l, p2t, row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        printf("%-44s %s\n", "OneCard 27B mix: levers on, other mixes off", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_AUTO_FLINT_ROUTER (2026-10-23): a flint PXQN4 file on ONE P100 or V100 with VRAM < file <= 1.5x VRAM
    //      sets PXA_ROUTER_BF16=2 (the BF16 router GEMV on any card); nothing elsewhere; PXA_AUTO_FLINT_ROUTER=0 and
    //      PXA_AUTO_FLINT_1CARD=0 each turn it off ----
    {
        bool ok = true;
        const uint64_t MB = 1ull << 20;
        auto sized = [](pxa_model_info m, uint64_t b) { m.bytes = b; return m; };
        const auto fl   = sized(mk("flint",     "PXQN4", 256, 4), 20561*MB);
        const auto flf  = sized(mk("flint",     "PXQN4", 256, 4), 15000*MB);
        const auto flb  = sized(mk("flint",     "PXQN4", 256, 4), 25000*MB);
        const auto flq  = sized(mk("flint",     "PXQ4",  256, 4), 20561*MB);
        const auto flm  = [&]{ auto m = sized(mk("flint", "PXQ_UNIVERSAL", 256, 4), 23719*MB); m.pxq_mask = (1u << (259-248)) | (1u << (263-248)); return m; }();   // Flint B: PXQN4+PXQN5 experts
        const auto flx  = [&]{ auto m = sized(mk("flint", "PXQ_UNIVERSAL", 256, 4), 23719*MB); m.pxq_mask = (1u << (259-248)) | (1u << (260-248)); return m; }();   // any other PXQ mix
        const auto moe  = sized(mk("qwen35moe", "PXQN4", 256, 2), 18617*MB);
        struct { const char * name; const char * topo; const pxa_model_info * m; bool want; } fr[] = {
            {"flint router: 1x P100 16 GB: on",           "1x600:16276", &fl,   true },
            {"flint router: 1x V100 16 GB: on",           "1x700:16144", &fl,   true },
            {"flint router: fits (file <= VRAM): off",    "1x700:16144", &flf,  false},
            {"flint router: file > 1.5x VRAM: off",       "1x600:16276", &flb,  false},
            {"flint router: VRAM unknown: off",           "1x700",       &fl,   false},
            {"flint router: 2x V100: off",                "2x700:16144", &fl,   false},
            {"flint router: 1x sm_61 (unmeasured): off",  "1x610:24576", &fl,   false},
            {"flint router: PXQ4 tier (unmeasured): off", "1x600:16276", &flq,  false},
            {"flint router: B mix (N4+N5) 1x P100: on",   "1x600:16276", &flm,  true },
            {"flint router: B mix (N4+N5) 1x V100: on",   "1x700:16144", &flm,  true },
            {"flint router: other PXQ_UNIVERSAL mix: off", "1x700:16144", &flx,  false},
            {"flint router: qwen35moe 1x V100: off",      "1x700:16144", &moe,  false},
        };
        const pxa_user_set none;
        for (const auto & r : fr) {
            const bool got = pxa_registry_flint_1card_router(pxa_topology_parse(r.topo), *r.m, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, 2, 0);
            const pxa_pick * pk = pick(ac, "PXA_AUTO_FLINT_ROUTER");
            bool env_r = false;
            for (const auto & e : ac.env) env_r = env_r || (e.first == "PXA_ROUTER_BF16" && e.second == "2");
            const bool row_ok = got == r.want && (pk != nullptr) == r.want && env_r == r.want;
            printf("%-44s %-12s %-3s %s\n", r.name, r.topo, got ? "on" : "off", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        {   // the reference level (PXA_ENHANCE=0 class) applies no lever
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700:16144"), fl, none, 0, 0);
            const bool row_ok = pick(ac, "PXA_AUTO_FLINT_ROUTER") == nullptr;
            printf("%-44s %-12s %s\n", "flint router: level 0: off", "1x700:16144", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        for (const char * sw : {"PXA_AUTO_FLINT_ROUTER", "PXA_AUTO_FLINT_1CARD"}) {
            setenv(sw, "0", 1);
            const bool got = pxa_registry_flint_1card_router(pxa_topology_parse("1x700:16144"), fl, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700:16144"), fl, none, 2, 0);
            bool env_any = false;
            for (const auto & e : ac.env) env_any = env_any || e.first == "PXA_ROUTER_BF16";
            const bool row_ok = !got && pick(ac, "PXA_AUTO_FLINT_ROUTER") == nullptr && !env_any;
            printf("%-44s %-12s %s\n", sw, "=0: off", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
            unsetenv(sw);
        }
        printf("%-44s %-12s %s\n", "flint one-card BF16 router GEMV", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- PXA_AUTO_FLINT_V100 (2026-10-23): a flint PXQN4 file on ONE V100 with VRAM < file <= 1.5x VRAM arms
    //      the six-lever plain-decode stack (graphs on the decode segments ...); never on a P100; PXA_AUTO_FLINT_V100=0 and
    //      PXA_AUTO_FLINT_1CARD=0 each turn it off ----
    {
        bool ok = true;
        const uint64_t MB = 1ull << 20;
        auto sized = [](pxa_model_info m, uint64_t b) { m.bytes = b; return m; };
        const auto fl   = sized(mk("flint",     "PXQN4", 256, 4), 20561*MB);
        const auto flf  = sized(mk("flint",     "PXQN4", 256, 4), 15000*MB);
        const auto flb  = sized(mk("flint",     "PXQN4", 256, 4), 25000*MB);
        const auto flq  = sized(mk("flint",     "PXQ4",  256, 4), 20561*MB);
        const auto flm  = [&]{ auto m = sized(mk("flint", "PXQ_UNIVERSAL", 256, 4), 23719*MB); m.pxq_mask = (1u << (259-248)) | (1u << (263-248)); return m; }();   // Flint B: PXQN4+PXQN5 experts
        const auto flx  = [&]{ auto m = sized(mk("flint", "PXQ_UNIVERSAL", 256, 4), 23719*MB); m.pxq_mask = (1u << (259-248)) | (1u << (260-248)); return m; }();   // any other PXQ mix
        const auto moe  = sized(mk("qwen35moe", "PXQN4", 256, 2), 18617*MB);
        static const char * const six[] = { "PXA_CUDA_GRAPH_DECODE", "PXA_CUDA_GRAPH_MOE", "PXA_CUDA_GRAPH_LRU",
                                             "PXA_SCHED_ASYNC_INPUTS", "PXA_PLAIN_GREEDY_AMAX", "PXA_ARGMAX_VAL_MB" };
        auto n_six = [&](const pxa_autoconfig & ac) {
            int n = 0;
            for (const char * k : six) {
                for (const auto & e : ac.env) {
                    if (e.first == k && e.second == (std::string(k) == "PXA_CUDA_GRAPH_LRU" ? "256" : "1")) { n++; break; }
                }
            }
            return n;
        };
        struct { const char * name; const char * topo; const pxa_model_info * m; bool want; } fv[] = {
            {"flint v100 stack: 1x V100 16 GB: on",          "1x700:16144", &fl,   true },
            {"flint v100 stack: 1x V100 32 GB fits: off",    "1x700:32510", &fl,   false},
            {"flint v100 stack: 1x P100 16 GB: off",         "1x600:16276", &fl,   false},
            {"flint v100 stack: fits (file <= VRAM): off",   "1x700:16144", &flf,  false},
            {"flint v100 stack: file > 1.5x VRAM: off",      "1x700:16144", &flb,  false},
            {"flint v100 stack: VRAM unknown: off",          "1x700",       &fl,   false},
            {"flint v100 stack: 2x V100: off",               "2x700:16144", &fl,   false},
            {"flint v100 stack: PXQ4 tier: off",             "1x700:16144", &flq,  false},
            {"flint v100 stack: B mix (N4+N5) 1x V100: on",  "1x700:16144", &flm,  true },
            {"flint v100 stack: B mix (N4+N5) 1x P100: off", "1x600:16276", &flm,  false},
            {"flint v100 stack: other PXQ_UNIVERSAL mix: off", "1x700:16144", &flx, false},
            {"flint v100 stack: qwen35moe 1x V100: off",     "1x700:16144", &moe,  false},
        };
        const pxa_user_set none;
        for (const auto & r : fv) {
            const bool got = pxa_registry_flint_1card_v100(pxa_topology_parse(r.topo), *r.m, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, 2, 0);
            const pxa_pick * pk = pick(ac, "PXA_AUTO_FLINT_V100");
            const int n = n_six(ac);
            const bool row_ok = got == r.want && (pk != nullptr) == r.want && (r.want ? n == 6 : pk == nullptr);
            printf("%-44s %-12s %-3s %s\n", r.name, r.topo, got ? "on" : "off", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        {   // the reference level (PXA_ENHANCE=0 class) applies no lever
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700:16144"), fl, none, 0, 0);
            const bool row_ok = pick(ac, "PXA_AUTO_FLINT_V100") == nullptr && n_six(ac) == 0;
            printf("%-44s %-12s %s\n", "flint v100 stack: level 0: off", "1x700:16144", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
        }
        for (const char * sw : {"PXA_AUTO_FLINT_V100", "PXA_AUTO_FLINT_1CARD"}) {
            setenv(sw, "0", 1);
            const bool got = pxa_registry_flint_1card_v100(pxa_topology_parse("1x700:16144"), fl, nullptr);
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse("1x700:16144"), fl, none, 2, 0);
            bool env_g = false;
            for (const auto & e : ac.env) env_g = env_g || e.first == "PXA_CUDA_GRAPH_DECODE";
            const bool row_ok = !got && pick(ac, "PXA_AUTO_FLINT_V100") == nullptr && !env_g;
            printf("%-44s %-12s %s\n", sw, "=0: off", row_ok ? "PASS" : "FAIL");
            ok = ok && row_ok;
            unsetenv(sw);
        }
        printf("%-44s %-12s %s\n", "flint one-V100 plain-decode stack", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- the 35B MoE flagship on a V100 pair is INFERRED, matching tools/pxa-launch.py's
    //      "2xpair-flagship-moe-v100" row (todo registry-flagship-moe-v100-label; same pattern as the
    //      gemma4 fix above). The V100 pair was never booted on that file: its -b/-ub are borrowed
    //      from the P100 pair. The dense 27B PXQ4 pair cell stays MEASURED (2026-09-03 boots), the
    //      P100 flagship cell stays MEASURED, and the picks themselves do not change -------------
    {
        struct { const char * topo; const char * arch; const char * tier; int n_expert; const char * want;
                 int want_b; int want_ub; } flst[] = {
            {"2x700", "qwen35moe", "PXQ4",    256, "INFERRED", 8192, 2048},   // the flagship on a V100 pair
            {"2x700", "qwen35moe", "PXQ6",    256, "INFERRED", 8192, 2048},
            {"2x700", "qwen35moe", "PXQ4-HQ", 256, "INFERRED", 8192, 2048},
            {"2x700", "qwen35",    "PXQ4",    0,   "MEASURED", 8192, 2048},   // dense 27B PXQ4: the booted cell
            {"2x600", "qwen35moe", "PXQ4",    256, "MEASURED", 8192, 2048},   // the P100 flagship boot
        };
        bool ok = true;
        for (const auto & c : flst) {
            const auto m = mk(c.arch, c.tier, c.n_expert, 2);
            int b = 0, ub = 0, cx = 0; const char * why = nullptr; const char * st = nullptr;
            const int have = pxa_registry_batch_cell(pxa_topology_parse(c.topo), m, 2, &b, &ub, &why, &st, &cx);
            bool row_ok = have && st && std::string(st) == c.want && b == c.want_b && ub == c.want_ub;
            // the autoconfig carries the same status onto -b and -ub, and still applies the pick
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(c.topo), m, none, 2, 0);
            const pxa_pick * pb = pick(ac, "-b");
            const pxa_pick * pu = pick(ac, "-ub");
            row_ok = row_ok && pb && pu && pb->status == c.want && pu->status == c.want &&
                     pb->applied && pu->applied;
            if (!row_ok) {
                printf("    %s %s/%s got %-9s want %s (b=%d ub=%d)\n", c.topo, c.arch, c.tier,
                       st ? st : "(null)", c.want, b, ub);
            }
            ok = ok && row_ok;
        }
        printf("%-44s %-10s %s\n", "MoE flagship on 2x V100 is INFERRED", "2x700", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- the qwen4exp rule is reported as a RULE, not a measurement ----------------------------
    {
        const pxa_topology t = pxa_topology_parse("6x600");   // off the measured 4x P100 cell
        const pxa_pick * ub = nullptr;
        const pxa_autoconfig ac = pxa_autoconfig_resolve(t, fn, none, 2, 0);
        ub = pick(ac, "-ub");
        const bool ok = ub && ub->value == "1024" && ub->status == "RULE";
        printf("%-44s %-10s %s\n", "6x P100 Flash-Next ub1024 status RULE", "6x600", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- SM75 file-layout cells, including the mixed pair, retain a load-time fit walk ---------
    {
        auto f = fn;
        f.n_layer = 49;
        f.n_pxq = 681;
        f.bytes = 83201745728ULL;
        f.pxa_kv.push_back({"pxa.pxqn.encoder", "PXQN"});
        struct { const char * topo; int b, ub; const char * status; } cases[] = {
            {"750:22528", 2048, 2048, "FIT"},
            {"750:22000", 2048, 2048, "FIT"},
            {"750:22528,700:32768", 2048, 2048, "FIT"},
            {"700:32494,750:22000", 2048, 2048, "FIT"},
            {"750:11264", 1024, 1024, "RULE"},
            {"750:16384", 1024, 1024, "RULE"},
            {"750:24576", 1024, 1024, "RULE"},
            {"750", 1024, 1024, "RULE"},
            {"750:11264,700:32768", 0, 0, nullptr},
            {"750:22528,700:16384", 0, 0, nullptr},
            {"750:22528,600:16384", 0, 0, nullptr},
            {"750:22528,700:32768,700:32768", 0, 0, nullptr},
        };
        bool ok = true;
        for (const auto & c : cases) {
            int b = 0, ub = 0;
            const char * st = nullptr;
            pxa_registry_batch_cell(pxa_topology_parse(c.topo), f, 2, &b, &ub, nullptr, &st);
            const bool row_ok = b == c.b && ub == c.ub &&
                ((st == nullptr && c.status == nullptr) || (st && c.status && strcmp(st, c.status) == 0));
            ok &= row_ok;
            if (!row_ok) printf("    SM75 batch %s got %d/%d %s\n", c.topo, b, ub, st ? st : "none");
        }
        for (int mismatch = 0; mismatch < 5; ++mismatch) {
            auto other = f;
            if (mismatch == 0) other.bytes++;
            if (mismatch == 1) other.n_pxq--;
            if (mismatch == 2) other.pxa_kv.clear();
            if (mismatch == 3) other.n_layer--;
            if (mismatch == 4) other.n_expert = 256;
            int b = 0, ub = 0;
            const int have = pxa_registry_batch_cell(pxa_topology_parse("750:22528,700:32768"), other, 2,
                                                     &b, &ub, nullptr);
            ok &= !have && b == 0 && ub == 0;
        }
        for (int level : {0, 1, 2}) {
            pxa_user_set u;
            const auto ac = pxa_autoconfig_resolve(pxa_topology_parse("750:22528,700:32768"), f, u, level, 0);
            const auto * ub = pick(ac, "-ub");
            ok &= ub && (level == 2 ? ub->value == "fit 2048" && ub->status == "FIT" : ac.n_ubatch == 0);
            u.b = u.ub = true;
            const auto explicit_ac = pxa_autoconfig_resolve(pxa_topology_parse("750:22528,700:32768"), f, u, level, 0);
            const auto * explicit_b = pick(explicit_ac, "-b");
            const auto * explicit_ub = pick(explicit_ac, "-ub");
            ok &= explicit_b && explicit_ub && !explicit_b->applied && !explicit_ub->applied &&
                  explicit_b->status == "USER" && explicit_ub->status == "USER";
        }
        printf("%-44s %s\n", "SM75/pair file-layout cells and rollback", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    // ---- cost lines: an explicit known-worse value is kept and named once -----
    {
        auto has_cost = [](const pxa_autoconfig & ac, const char * what) {
            for (const auto & c : ac.costs) {
                if (c.what == what) {
                    return true;
                }
            }
            return false;
        };
        const auto n4 = mk("qwen35", "PXQN4", 0, 4);
        pxa_user_set lay;
        lay.sm = true;
        lay.sm_value = PXA_SM_LAYER;
        pxa_user_set ten;
        ten.sm = true;
        ten.sm_value = PXA_SM_TENSOR;
        struct cc { const char * name; const char * topo; const pxa_model_info * m; const pxa_user_set * u; int level;
                    const char * red; const char * rpf; const char * want; bool want_on; };
        const cc cases[] = {
            { "cost: -sm layer on 2x P100 PXQN4",         "2x600", &n4,  &lay,  2, nullptr, nullptr, "-sm layer", true  },
            { "cost: -sm layer on 2x V100 PXQN4",         "2x700", &n4,  &lay,  2, nullptr, nullptr, "-sm layer", true  },
            { "cost: -sm layer on 4x P100 PXQN4",         "4x600", &n4,  &lay,  2, nullptr, nullptr, "-sm layer", true  },
            { "cost: none for -sm layer on 1x P100",      "1x600", &n4,  &lay,  2, nullptr, nullptr, "-sm layer", false },
            { "cost: none for -sm layer on mixed pair",   "600,700", &n4, &lay, 2, nullptr, nullptr, "-sm layer", false },
            { "cost: none for -sm layer on PXQ3 pair",    "2x600", &q27p3, &lay, 2, nullptr, nullptr, "-sm layer", false },
            { "cost: none for -sm layer at DEFAULT",      "2x600", &n4,  &lay,  1, nullptr, nullptr, "-sm layer", false },
            { "cost: REDUCE=off left to the reduce's line",     "2x600", &n4,  &ten,  2, "off",   nullptr, "PXA_TSPLIT_REDUCE=off", false },
            { "cost: REDUCE=off left to the reduce (auto)",            "2x700", &n4,  &none, 2, "off",   nullptr, "PXA_TSPLIT_REDUCE=off", false },
            { "cost: none for REDUCE=fused",              "2x600", &n4,  &ten,  2, "fused", nullptr, "PXA_TSPLIT_REDUCE=fused", false },
            { "cost: none for REDUCE=off on layer",       "2x600", &n4,  &lay,  2, "off",   nullptr, "PXA_TSPLIT_REDUCE=off", false },
            { "cost: REDUCE_PREFILL=1 under tensor",      "2x600", &n4,  &ten,  2, nullptr, "1",     "PXA_TSPLIT_REDUCE_PREFILL=1", true },
            { "cost: none for REDUCE_PREFILL=0",          "2x600", &n4,  &ten,  2, nullptr, "0",     "PXA_TSPLIT_REDUCE_PREFILL=0", false },
            { "cost: none under PXA_REFERENCE",           "2x600", &n4,  &ten,  0, "off",   "1",     "PXA_TSPLIT_REDUCE=off", false },
        };
        for (const auto & c : cases) {
            if (c.red) { setenv("PXA_TSPLIT_REDUCE", c.red, 1); } else { unsetenv("PXA_TSPLIT_REDUCE"); }
            if (c.rpf) { setenv("PXA_TSPLIT_REDUCE_PREFILL", c.rpf, 1); } else { unsetenv("PXA_TSPLIT_REDUCE_PREFILL"); }
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(c.topo), *c.m, *c.u, c.level, 0);
            const bool ok = has_cost(ac, c.want) == c.want_on && (c.u->sm ? ac.split == PXA_SM_KEEP : true);
            printf("%-44s %-10s costs=%zu %s\n", c.name, c.topo, ac.costs.size(), ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
        unsetenv("PXA_TSPLIT_REDUCE");
        unsetenv("PXA_TSPLIT_REDUCE_PREFILL");
    }
    // ---- the registry in every tool: several models, tool kinds, bench instances (#22182 F1 + F3) -----
    {
        typedef std::vector<std::pair<std::string, std::string>> envv;
        const auto q27n = mk("qwen35",    "PXQN4", 0, 4);     // the dense 27B PXQN4 file
        const auto m35n = mk("qwen35moe", "PXQN4", 256, 2);   // the 35B-A3B PXQN4 file (the P60 cell)
        const pxa_topology t2 = pxa_topology_parse("2x600");
        const pxa_autoconfig a27 = pxa_autoconfig_resolve(t2, q27n, none, 2, 0);
        const pxa_autoconfig a35 = pxa_autoconfig_resolve(t2, m35n, none, 2, 0);
        auto clear = [&]() {
            for (const auto & e : a27.env) unsetenv(e.first.c_str());
            for (const auto & e : a35.env) unsetenv(e.first.c_str());
            unsetenv("PXA_HS_ENV_UNION");
        };
        auto env_is = [](const envv & v) {   // every pick of v is the process value
            for (const auto & e : v) { const char * c = getenv(e.first.c_str()); if (!c || e.second != c) return false; }
            return true;
        };
        auto only_in = [](const envv & a, const envv & b) {   // picks of a whose name b does not set
            envv o;
            for (const auto & e : a) { bool in = false; for (const auto & f : b) in = in || f.first == e.first; if (!in) o.push_back(e); }
            return o;
        };
        auto none_set = [](const envv & v) {
            for (const auto & e : v) if (getenv(e.first.c_str())) return false;
            return true;
        };
        const envv x35 = only_in(a35.env, a27.env), x27 = only_in(a27.env, a35.env);
        // F1: -m 27B + hot 35B: the 27B's picks only, the 35B's differing picks named
        clear();
        std::vector<std::string> d1;
        pxa_registry_env_apply(a27.env, true, nullptr);
        pxa_registry_env_apply(a35.env, false, &d1);
        const bool ok1 = !x35.empty() && env_is(a27.env) && none_set(x35) && d1.size() >= x35.size();
        printf("%-44s %-10s warn %zu %s\n", "F1 -m 27B + hot 35B: 27B picks only", "2x600", d1.size(), ok1 ? "PASS" : "FAIL");
        g_fail += !ok1;
        // F1 reverse: -m 35B + hot 27B
        clear();
        std::vector<std::string> d2;
        pxa_registry_env_apply(a35.env, true, nullptr);
        pxa_registry_env_apply(a27.env, false, &d2);
        const bool ok2 = !x27.empty() && env_is(a35.env) && none_set(x27) && d2.size() >= x27.size();
        printf("%-44s %-10s warn %zu %s\n", "F1 -m 35B + hot 27B: 35B picks only", "2x600", d2.size(), ok2 ? "PASS" : "FAIL");
        g_fail += !ok2;
        // F1 PXA_HS_ENV_UNION=1: the old first-wins union, no warning for what it applied
        clear();
        setenv("PXA_HS_ENV_UNION", "1", 1);
        std::vector<std::string> d3;
        pxa_registry_env_apply(a27.env, true, nullptr);
        pxa_registry_env_apply(a35.env, false, &d3);
        const bool ok3 = env_is(a27.env) && env_is(x35);
        printf("%-44s %-10s warn %zu %s\n", "F1 PXA_HS_ENV_UNION=1: old union", "2x600", d3.size(), ok3 ? "PASS" : "FAIL");
        g_fail += !ok3;
        // F1: an empty PXA_X= counts as unset (the pick lands); a real user value stays
        clear();
        setenv("PXA_FA_GQA_PACK", "", 1);
        setenv("PXA_P60_MEGA", "0", 1);
        pxa_registry_env_apply(a35.env, true, nullptr);
        const char * gq = getenv("PXA_FA_GQA_PACK");
        const char * mg = getenv("PXA_P60_MEGA");
        const bool ok4 = env_of(a35, "PXA_FA_GQA_PACK") && gq && std::string(gq) == env_of(a35, "PXA_FA_GQA_PACK") &&
                         mg && std::string(mg) == "0";
        printf("%-44s %-10s %s\n", "F1 empty PXA_X= is unset, PXA_X=0 kept", "2x600", ok4 ? "PASS" : "FAIL");
        g_fail += !ok4;
        clear();
        // F3(a): the tool kinds; a gate tool resolves lever-only (no -c / KV / batch / split / -ngl pick)
        const bool ok5 = pxa_tool_kind_of("/e/bin/llama-perplexity") == PXA_TOOL_GATE && pxa_tool_kind_of("llama-imatrix") == PXA_TOOL_GATE &&
                         pxa_tool_kind_of("./llama-bench") == PXA_TOOL_BENCH && pxa_tool_kind_of("llama-server") == PXA_TOOL_SERVER &&
                         pxa_tool_kind_of("/usr/bin/llama-cli") == PXA_TOOL_CLI && pxa_tool_kind_of(nullptr) == PXA_TOOL_CLI;
        printf("%-44s %-10s %s\n", "F3a tool kinds from argv[0]", "", ok5 ? "PASS" : "FAIL");
        g_fail += !ok5;
        const auto fnx = mk("qwen4exp", "PXQ_UNIVERSAL", 512, 2);
        const pxa_autoconfig gs = pxa_autoconfig_resolve(t2, fnx, none, 2, 0);
        const pxa_autoconfig gg = pxa_autoconfig_resolve(t2, fnx, pxa_user_set_lever_only(PXA_SM_LAYER, false, false), 2, 0);
        const pxa_autoconfig gm = pxa_autoconfig_resolve(t2, m35n, pxa_user_set_lever_only(PXA_SM_LAYER, false, false), 2, 0);
        const bool ok6 = gs.kv_type != nullptr && gg.kv_type == nullptr && gg.n_ctx == 0 &&
                         (pick(gg, "-b") == nullptr || !pick(gg, "-b")->applied) && (pick(gg, "-ub") == nullptr || !pick(gg, "-ub")->applied) &&
                         gg.split == PXA_SM_KEEP && gg.n_gpu_layers < 0 && !gg.ts_even &&
                         env_of(gm, "PXA_P60_MEGA") != nullptr && gm.kv_type == nullptr && gm.n_ctx == 0;
        printf("%-44s %-10s kv %s -> %s %s\n", "F3a gate tool: Flash-Next KV pick dropped", "2x600", gs.kv_type ? gs.kv_type : "-",
               gg.kv_type ? gg.kv_type : "-", ok6 ? "PASS" : "FAIL");
        g_fail += !ok6;
        // F3(a): a CLI -c pick only where no -c was given and it does not shrink the trained window
        const bool ok7 = pxa_tool_ctx_pick(4096, false, 262144) == 0 && pxa_tool_ctx_pick(4096, true, 2048) == 0 &&
                         pxa_tool_ctx_pick(0, false, 4096) == 0 && pxa_tool_ctx_pick(8192, false, -1) == 0 &&
                         pxa_tool_ctx_pick(8192, false, 4096) == 8192 && pxa_tool_ctx_pick(8192, false, 8192) == 8192;
        printf("%-44s %-10s %s\n", "F3a cli -c pick never shrinks the window", "", ok7 ? "PASS" : "FAIL");
        g_fail += !ok7;
        // F3(b): PXA_AUTOCONFIG=0 is what the server's batch/ub block reads (a measured cell exists at level 2)
        int cb = 0, cub = 0;
        const bool cell = pxa_registry_batch_cell(t2, m35n, 2, &cb, &cub, nullptr) != 0;
        unsetenv("PXA_AUTOCONFIG");
        const bool on_before = !pxa_autoconfig_off();
        setenv("PXA_AUTOCONFIG", "0", 1);
        const bool off_now = pxa_autoconfig_off();
        setenv("PXA_AUTOCONFIG", "1", 1);
        const bool on_one = !pxa_autoconfig_off();
        unsetenv("PXA_AUTOCONFIG");
        const bool ok8 = cell && on_before && off_now && on_one;
        printf("%-44s %-10s cell -b %d -ub %d %s\n", "F3b PXA_AUTOCONFIG=0 stops server batch/ub", "2x600", cb, cub, ok8 ? "PASS" : "FAIL");
        g_fail += !ok8;
        // F3(c): llama-bench -sm layer,row on the P60 cell: the P60 picks are layer-only, so neither instance gets them
        const pxa_autoconfig bl = pxa_autoconfig_resolve(t2, m35n, pxa_user_set_lever_only(PXA_SM_LAYER, false, false), 2, 0);
        const pxa_autoconfig br = pxa_autoconfig_resolve(t2, m35n, pxa_user_set_lever_only(2 /* row */, false, false), 2, 0);
        std::vector<std::string> dr;
        const envv cm = pxa_env_common({ bl.env, br.env }, &dr);
        bool p60_in = false;
        for (const auto & e : cm) p60_in = p60_in || e.first == "PXA_P60_MEGA";
        const envv one = pxa_env_common({ bl.env }, nullptr);
        const envv same = pxa_env_common({ bl.env, bl.env }, nullptr);
        const bool ok9 = env_of(bl, "PXA_P60_MEGA") && !env_of(br, "PXA_P60_MEGA") && !p60_in && !dr.empty() &&
                         one == bl.env && same == bl.env;
        printf("%-44s %-10s shared %zu dropped %zu %s\n", "F3c bench -sm layer,row: no cross picks", "2x600", cm.size(), dr.size(),
               ok9 ? "PASS" : "FAIL");
        g_fail += !ok9;
        // F3(c): bench -m 27B -m 35B (both layer): only shared picks
        const pxa_autoconfig b27 = pxa_autoconfig_resolve(t2, q27n, pxa_user_set_lever_only(PXA_SM_LAYER, false, false), 2, 0);
        const envv cm2 = pxa_env_common({ b27.env, bl.env }, nullptr);
        bool cross = false;
        for (const auto & e : cm2) cross = cross || e.first == "PXA_P60_MEGA";
        printf("%-44s %-10s shared %zu %s\n", "F3c bench -m 27B -m 35B: shared only", "2x600", cm2.size(), !cross ? "PASS" : "FAIL");
        g_fail += cross;
    }
    // ---- sm_60 MoE prefill: PXA_PF_STABLE_SHAPE (order #22214) -------------------------------
    {
        const auto m35n = mk("qwen35moe", "PXQN4", 256, 2);
        const auto m35q = mk("qwen35moe", "PXQ4", 256, 2);
        const auto q27n = mk("qwen35",    "PXQN4", 0, 4);
        struct { const char * name; const char * topo; const pxa_model_info * m; int level; bool want; } rows[] = {
            { "PF_STABLE_SHAPE 2x P100 35B PXQN4",      "2x600",   &m35n, 2, true  },
            { "PF_STABLE_SHAPE 1x P100 35B PXQ4",       "1x600",   &m35q, 2, true  },
            { "PF_STABLE_SHAPE 4x P100 35B PXQN4",      "4x600",   &m35n, 2, true  },
            { "PF_STABLE_SHAPE not on 2x V100 35B",     "2x700",   &m35n, 2, false },
            { "PF_STABLE_SHAPE not on a mixed pair",    "600,700", &m35n, 2, false },
            { "PF_STABLE_SHAPE not for the dense 27B",  "2x600",   &q27n, 2, false },
            { "PF_STABLE_SHAPE not at DEFAULT level",   "2x600",   &m35n, 1, false },
            { "PF_STABLE_SHAPE not at REFERENCE",       "2x600",   &m35n, 0, false },
        };
        for (const auto & r : rows) {
            const pxa_autoconfig ac = pxa_autoconfig_resolve(pxa_topology_parse(r.topo), *r.m, none, r.level, 0);
            const char * v = env_of(ac, "PXA_PF_STABLE_SHAPE");
            const bool ok = (v != nullptr && std::string(v) == "1") == r.want;
            printf("%-44s %-10s %s %s\n", r.name, r.topo, v ? v : "-", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
    }
    // ---- PXA_SPEC_STREAM_MTP: streamed Nemotron-H MoE experts get no auto speculation (#22519) ----
    {
        unsetenv("PXA_SPEC_STREAM_MTP");
        struct row { const char * name; const char * arch; bool head, fits, cpu, xoff, sforce, xmtp, want; };
        const row rows[] = {
            { "nemo stream: XCACHE=0, file > cards",        "nemotron_h_moe", true,  false, false, true,  false, false, true  },
            { "nemo stream: cache declines (head loaded)",  "nemotron_h_moe", true,  false, false, false, false, false, true  },
            { "nemo stream: PXA_STREAM_WEIGHTS forced",     "nemotron_h_moe", true,  true,  false, false, true,  false, true  },
            { "nemo resident: file fits the cards",         "nemotron_h_moe", true,  true,  false, true,  false, false, false },
            { "nemo cache + chain: PXA_XCACHE_MTP=1",       "nemotron_h_moe", true,  false, false, false, false, true,  false },
            { "nemo -ot CPU experts (not streamed)",        "nemotron_h_moe", true,  false, true,  true,  false, false, false },
            { "nemo without an MTP head",                   "nemotron_h_moe", false, false, false, true,  false, false, false },
            { "qwen35moe streamed: other arch untouched",   "qwen35moe",      true,  false, false, true,  false, false, false },
        };
        for (const auto & r : rows) {
            pxa_spec_stream_in in;
            in.arch = r.arch; in.has_head = r.head; in.fits = r.fits; in.cpu_override = r.cpu;
            in.xcache_off = r.xoff; in.stream_forced = r.sforce; in.xcache_mtp = r.xmtp;
            std::string why;
            const bool got = pxa_registry_spec_stream_off(in, &why);
            const bool ok  = got == r.want && (!got || why.find("PXA_SPEC_STREAM_MTP=0") != std::string::npos);
            printf("%-44s %-10s %s\n", r.name, "", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
        // Nemotron 3.5 Super ships its NextN head as a separate mtp-*.gguf (PXA_MTP_HEAD_FILE): the attached head counts as a head, so the
        // 2x P100 host-expert layout (file > cards, cache declines with the head) keeps auto MTP off; a trunk without either has none
        {
            const bool h1 = pxa_registry_spec_head_present(0, "/head/mtp-Super-Q8_0.gguf");
            const bool h2 = !pxa_registry_spec_head_present(0, nullptr) && !pxa_registry_spec_head_present(0, "") && pxa_registry_spec_head_present(1, nullptr);
            pxa_spec_stream_in in;
            in.arch = "nemotron_h_moe"; in.has_head = h1; in.fits = false;
            std::string why;
            const bool off = pxa_registry_spec_stream_off(in, &why);
            const bool ok = h1 && h2 && off && why.find("expert cache declines") != std::string::npos;
            printf("%-44s %-10s %s\n", "nemo super: attached head file, 2x P100", "", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
        setenv("PXA_SPEC_STREAM_MTP", "0", 1);   // the off switch restores the old auto layer
        pxa_spec_stream_in in;
        in.arch = "nemotron_h_moe"; in.has_head = true; in.fits = false; in.xcache_off = true;
        const bool ok = !pxa_registry_spec_stream_off(in, nullptr);
        printf("%-44s %-10s %s\n", "nemo stream: PXA_SPEC_STREAM_MTP=0 off switch", "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
        unsetenv("PXA_SPEC_STREAM_MTP");
    }
    // ---- PXA_GLM5NEXT_MTP auto scope: GLM MTP auto-arms only with resident experts ----
    {
        struct row { const char * name; const char * env; bool head, fits, cpu, xoff, sforce, want; };
        const row rows[] = {
            { "glm resident experts: auto layer decides",  nullptr, true,  true,  false, false, false, false },
            { "glm -ot CPU experts (V24/Overdrive): off",  nullptr, true,  false, true,  false, false, true  },
            { "glm -ot CPU experts, file fits: off",       nullptr, true,  true,  true,  false, false, true  },
            { "glm streamed (PXA_STREAM_WEIGHTS): off",    nullptr, true,  true,  false, false, true,  true  },
            { "glm file > cards, XCACHE=0: off",           nullptr, true,  false, false, true,  false, true  },
            { "glm without a head: untouched",             nullptr, false, false, true,  false, false, false },
            { "glm -ot CPU + PXA_GLM5NEXT_MTP=1: arms",    "1",     true,  false, true,  false, false, false },
            { "glm resident + PXA_GLM5NEXT_MTP=0: off",    "0",     true,  true,  false, false, false, true  },
        };
        for (const auto & r : rows) {
            if (r.env) setenv("PXA_GLM5NEXT_MTP", r.env, 1); else unsetenv("PXA_GLM5NEXT_MTP");
            pxa_spec_stream_in in;
            in.arch = "glm5next"; in.has_head = r.head; in.fits = r.fits; in.cpu_override = r.cpu;
            in.xcache_off = r.xoff; in.stream_forced = r.sforce;
            std::string why;
            const bool got = pxa_registry_spec_stream_off(in, &why);
            const bool ok  = got == r.want && (!got || why.find("PXA_GLM5NEXT_MTP") != std::string::npos);
            printf("%-44s %-10s %s\n", r.name, "", ok ? "PASS" : "FAIL");
            g_fail += !ok;
        }
        unsetenv("PXA_GLM5NEXT_MTP");
    }
    // ---- the catalog declares the levers this change reads -----------------------------------
    for (const char * n : { "PXA_AUTO_SM", "PXA_AUTO_SM_PXQN", "PXA_EXPLAIN", "PXA_TOPOLOGY", "PXA_TSPLIT_ALLOW_4WAY", "PXA_ENHANCE",
                            "PXA_TSPLIT_FALLBACK", "PXA_AUTO_UB_LONG", "PXA_AUTO_UB_VOLTA_Q8",
                            "PXA_CACHE_HYBRID_USABLE", "PXA_SLOT_VICTIM", "PXA_AUTO_UB_RESERVE_MB",
                            "PXA_DN_GM_FAULT", "PXA_SPEC_STREAM_MTP", "PXA_GLM5NEXT_MTP" }) {
        const bool ok = pxa_lever_find(n) != nullptr;
        printf("%-44s %-10s %s\n", (std::string("catalog declares ") + n).c_str(), "", ok ? "PASS" : "FAIL");
        g_fail += !ok;
    }
    size_t n = 0;
    pxa_lever_catalog(&n);
    printf("catalog: %zu levers\n", n);
    printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
