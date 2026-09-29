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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

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

static void expect(const char * name, const char * topo, const pxa_model_info & m, const pxa_user_set & u, int level,
                   int want_split, int want_b, int want_ub, const char * want_sm_value) {
    const pxa_topology t = pxa_topology_parse(topo);
    const pxa_autoconfig ac = pxa_autoconfig_resolve(t, m, u, level, 0);
    const pxa_pick * sm = pick(ac, "-sm");
    const bool ok = ac.split == want_split && ac.n_batch == want_b && ac.n_ubatch == want_ub &&
                    sm != nullptr && sm->value == want_sm_value &&
                    // the tensor split always travels with its environment, and only it does
                    ((ac.split == PXA_SM_TENSOR) == (ac.env.size() == 2)) &&   // REDUCE=fused + FALLBACK (REDUCE_PREFILL dropped in b74c512371)
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
    // 4x P100 (2026-09-27, tsplit-4card-default): the one set past a pair the split is measured faster on
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
    expect("2x V100 gemma4",                    "2x700", g4,  none, 2, PXA_SM_LAYER,  2048, 512,  "layer");
    expect("1x V100 gemma4",                    "1x700", g4,  none, 2, PXA_SM_LAYER,  2048, 512,  "layer");
    expect("2x P100 gemma4 expert (measured)",  "2x600", g4,  none, 2, PXA_SM_LAYER,  2048, 512,  "layer");
    // INFERRED launcher rows are not cells: these fall through to the card table, as before step 3
    expect("1x P100 gemma4 (INFERRED -> ladder)", "1x600", g4, none, 2, PXA_SM_LAYER, 0,    0,    "layer");
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
        const bool ok = a1.split == PXA_SM_LAYER && s1 && s1->value == "layer" && s1->evidence == "bug-280" &&
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
        printf("%-44s %-10s %s\n", "1x V100 q8 long: -ub pick is RULE bug-207", "1x700", ok ? "PASS" : "FAIL");
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
            std::vector<row> rows; int want_ub; int want_builds; int want_discards;
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
            const int got = pxa_registry_ub_walk(c.start, c.origin, c.reserve_mib * MiB, build, discard, &trail);
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
    // ---- the catalog declares the levers this change reads -----------------------------------
    for (const char * n : { "PXA_AUTO_SM", "PXA_AUTO_SM_PXQN", "PXA_EXPLAIN", "PXA_TOPOLOGY", "PXA_TSPLIT_ALLOW_4WAY", "PXA_ENHANCE",
                            "PXA_TSPLIT_FALLBACK", "PXA_AUTO_UB_LONG", "PXA_AUTO_UB_VOLTA_Q8",
                            "PXA_CACHE_HYBRID_USABLE", "PXA_SLOT_VICTIM", "PXA_AUTO_UB_RESERVE_MB",
                            "PXA_DN_GM_FAULT" }) {
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
