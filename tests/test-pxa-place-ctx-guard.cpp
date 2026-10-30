// test-pxa-place-ctx-guard.cpp -- bug #339 (xcache-slack-ctx-refuse): the PXA_STREAM_WEIGHTS=auto planner (src/llama-pxa-place.cpp)
// must never plan a layout the context then refuses (pxa_pool_pregrow: the floor must stay free after the GEMM pool) or runs out of
// VRAM in, at any -c/-b/-ub, on the CPU and the zero-copy cold path:
//   (A) the head GEMM's pool need at an all-outputs width follows the PXA_CUBLAS_SRC0_SLICE row slices,
//   (B) Nemotron-3-Nano-30B PXQ4 on one V100, -c 2048 -ub 512 perplexity (the #339 repro numbers): the C042 small-tail slack credit
//       left 7.5 MiB after the pool; the guarded plan leaves the floor + the head's GEMM growth free by the planner's estimate,
//   (C) a context that takes the logits cap (server, bench, cli) gets the previous plan exactly, at every deficit / arch / cold path,
//   (D) an all-outputs context: the invariant holds (or the plan says STILL SHORT), and every plan that already met it is unchanged,
//   (E) R1: an all-outputs context fits resident only if the full logits fit,
//   (F) ctx_guard off (PXA_PLACE_CTX_GUARD=0) = the previous plan byte for byte, whatever the fields hold,
//   (G) whole-layer and dense-spill plans meet the same invariant,
//   (H) an explicit PXA_XCACHE_SLACK_MB larger than the margin no longer plans below the floor.
// ctest: test-pxa-place-ctx-guard. standalone: g++ -O1 -std=c++17 -DPXA_TEST_STANDALONE -Isrc -Iinclude -Iggml/include -Iggml/src \
//   tests/test-pxa-place-ctx-guard.cpp src/llama-pxa-place.cpp -o /tmp/test-ctx-guard
#include "llama-pxa-place.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "pxa-test-portable.h"   // setenv / unsetenv on native Windows too

#ifdef PXA_TEST_STANDALONE
extern "C" bool ggml_pxqn_cpu_mmv_available(void) { return true; }
#endif

static const size_t MiB = 1u << 20;
static int fails = 0, checks = 0;
#define CHECK(c, ...) do { ++checks; if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct rng { uint64_t s; uint32_t next() { s = s*6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 33); } double u() { return (next() + 0.5)/2147483648.0; } };

// Nemotron-3-Nano-30B-A3B PXQ4 on one V100 (the #339 log, 2026-10-21 05:14): 23 MoE layers x 128 experts, up + down expert
// tensors, 5.06 MiB per expert (186 cold experts = 942 MiB; a one-layer ring of an 18-expert cap = 169 MiB), resident need
// 16310 MiB (logits capped) + pool 55 MiB vs 15828 MiB free, +53 MiB split-graph compute, uncapped logits +192 MiB.
static const int NL = 23, NE = 128;
static const size_t HALF_EXPERT = 2655000;   // one expert in one of the two expert tensors

struct scen {
    int    cc = 70; bool zc = false; bool counts = true; bool dense = false;
    double need = 16310, pool = 55, free_ = 15828, extra = 53;   // MiB
    double ctx_l = 0, ctx_h = 0, floor_ = 128, slack = -1, spec = 0, spill = 0;   // slack < 0: the auto credit (no PXA_XCACHE_SLACK_MB)
    bool   guard = true;
};

static pxa_place_input make(const scen & s) {
    pxa_place_input in;
    in.n_layer = NL; in.mode_req = 4; in.n_expert = NE; in.n_expert_used = 6;
    in.xc_cold_zc = s.zc; in.xc_cold_pxqn = false;
    in.floor_bytes = (size_t)(s.floor_*MiB);
    in.xc_ring1_auto = true;
    if (s.slack < 0) in.xc_slack_auto = (size_t) PXA_XCACHE_SLACK_AUTO_MB*MiB; else in.xc_slack_credit = (size_t)(s.slack*MiB);
    in.xc_extra_compute = s.counts && !s.dense ? (size_t)(s.extra*MiB) : 0;
    in.spill_extra = (size_t)(s.spill*MiB);
    if (s.spec > 0) in.spec_reserve = { (size_t)(s.spec*MiB) };
    in.ctx_guard = s.guard;
    pxa_place_device d; d.id = 0; d.cc = s.cc;
    d.need_resident = (size_t)(s.need*MiB); d.pool_need = (size_t)(s.pool*MiB); d.free_bytes = (size_t)(s.free_*MiB);
    d.ctx_logits = (size_t)(s.ctx_l*MiB); d.ctx_head = (size_t)(s.ctx_h*MiB);
    in.devices.push_back(d);
    rng r{339};
    if (s.counts && !s.dense) in.xc_counts.assign(NL, {});
    for (int l = 0; l < NL; ++l) {
        if (s.dense) {
            pxa_place_tensor dn; dn.name = "blk." + std::to_string(l) + ".ffn_down.weight"; dn.layer = l; dn.device = 0; dn.kind = PXA_PLACE_FFN_DOWN; dn.bytes = 120u*MiB;
            pxa_place_tensor up = dn; up.name = "blk." + std::to_string(l) + ".ffn_up.weight"; up.kind = PXA_PLACE_FFN_UPGATE;
            pxa_place_tensor gt = up; gt.name = "blk." + std::to_string(l) + ".ffn_gate.weight";
            in.tensors.push_back(dn); in.tensors.push_back(up); in.tensors.push_back(gt);
            continue;
        }
        if (s.counts) {
            in.xc_counts[l].resize(NE);
            for (int e = 0; e < NE; ++e) {
                double g = 0; for (int k = 0; k < 12; ++k) g += r.u(); g -= 6;
                in.xc_counts[l][e] = std::exp(1.3*g)*50.0;
            }
        }
        for (int t = 0; t < 2; ++t) {
            pxa_place_tensor pt; pt.name = "blk." + std::to_string(l) + (t == 0 ? ".ffn_up_exps.weight" : ".ffn_down_exps.weight");
            pt.layer = l; pt.device = 0; pt.kind = PXA_PLACE_EXPERTS; pt.bytes = (size_t) NE*HALF_EXPERT;
            in.tensors.push_back(pt);
        }
    }
    return in;
}

static bool same_plan(const pxa_place_result & a, const pxa_place_result & b) {
    return a.mode == b.mode && a.xc_cold == b.xc_cold && a.dev_ring == b.dev_ring && a.streamed == b.streamed && a.xc_cap == b.xc_cap &&
           a.names == b.names && a.dev_stream == b.dev_stream && a.dev_deficit == b.dev_deficit && a.cpu_instead == b.cpu_instead;
}
static size_t n_cold(const pxa_place_result & r) { size_t n = 0; for (auto & v : r.xc_cold) n += v.size(); return n; }

// the planner's own estimate of what stays free after the context's pool pre-grow for this plan, for a context that reserves
// ctx_l more than the capped plan (0 for a capped context), in MiB. The invariant: a streaming plan leaves >= floor + spec + ctx_h;
// a resident plan (R1) leaves >= floor with the full logits (its head growth is today's: every resident plan the old planner made
// with the full logits stays exactly as it was)
static double free_after_pool(const scen & s, const pxa_place_result & r) {
    if (r.mode == 0) return s.free_ - s.need - s.pool - s.ctx_l;
    const double net = ((double) r.dev_stream[0] - (double) r.dev_ring[0])/MiB;
    return s.free_ - s.need - s.pool - (r.xc_layers > 0 ? s.extra : 0.0) - s.ctx_l + net;
}
static bool meets(const scen & s, const pxa_place_result & r) { return free_after_pool(s, r) + 1e-6 >= s.floor_ + (r.mode == 0 ? 0.0 : s.spec + s.ctx_h); }
static bool short_flag(const pxa_place_result & r) { return r.dev_deficit[0] > 0 || r.reason.find("STILL SHORT") != std::string::npos; }

int main() {
#ifdef _WIN32
    setvbuf(stdout, nullptr, _IONBF, 0);   // the MSVC CRT rejects _IOLBF with size 0 (invalid parameter -> 0xc0000409)
#else
    setvbuf(stdout, nullptr, _IOLBF, 0);
#endif
    unsetenv("PXA_XCACHE_COST_LAYER_US"); unsetenv("PXA_XCACHE_COST_MISS_US"); unsetenv("PXA_XCACHE_ASYNC"); unsetenv("PXA_XCACHE_PLANNER");
    unsetenv("PXA_XCACHE_LAYERS"); unsetenv("PXA_XCACHE_LAYER_WEIGHT");

    // (A) head pool need: Q8_0 output.weight 2688 x 131072 at T=512 on sm_70 (cuBLAS dequant, no 2TN term on Volta)
    {
        const int64_t K = 2688, N = 131072, T = 512;
        const size_t lib = (size_t)(2*K*N + 2*T*K);
        const size_t slice = 256*MiB;
        const int64_t rows = 49920;   // 268435456 / 5376 = 49932 -> down to a multiple of 64
        const size_t want = (size_t)(rows*K*2 + 2*T*K);
        const size_t got = pxa_place_head_pool_need(lib, K, N, T, true, slice);
        CHECK(got == want, "sliced head pool %zu != %zu", got, want);
        CHECK(std::fabs(got/(double)MiB - 55 - 203.57) < 0.1, "nano head growth over the 55 MiB pool %.2f MiB, want ~203.6", got/(double)MiB - 55);
        CHECK(pxa_place_head_pool_need(lib, K, N, T, true, 0) == lib, "PXA_CUBLAS_SRC0_SLICE_MIB=0 must keep the whole copy");
        CHECK(pxa_place_head_pool_need(lib, K, N, T, false, slice) == lib, "an F16/F32 head is not sliced");
        const size_t mmq = (size_t)(T*K/32*36) + MiB;
        CHECK(pxa_place_head_pool_need(mmq, K, N, T, true, slice) == mmq, "an MMQ head (no f16 copy) keeps the library's estimate");
        const size_t small = (size_t)(2*1024*4096 + 2*T*1024);
        CHECK(pxa_place_head_pool_need(small, 1024, 4096, T, true, slice) == small, "a head under the slice size keeps the incumbent path");
        // a slice smaller than 64 rows (64 KiB / 5376 B = 12 rows) rounds up to 64; 1 MiB = 195 rows -> 192
        CHECK(pxa_place_head_pool_need(lib, K, N, T, true, 64*1024) == (size_t)(64*K*2 + 2*T*K), "a slice smaller than 64 rows rounds up to 64");
        CHECK(pxa_place_head_pool_need(lib, K, N, T, true, MiB) == (size_t)(192*K*2 + 2*T*K), "1 MiB slices hold 192 rows");
    }

    // (B) the #339 repro: all-outputs context (perplexity), auto slack credit, CPU cold path
    {
        scen bug; bug.ctx_l = 192; bug.ctx_h = 203.57;
        scen old = bug; old.guard = false;
        scen wa = old; wa.slack = 0;   // the workaround: PXA_XCACHE_SLACK_MB=0
        const auto r_old = pxa_place_plan(make(old)), r_new = pxa_place_plan(make(bug)), r_wa = pxa_place_plan(make(wa));
        printf("(B) nano30b 1xV100 ppl -ub 512: old %zu cold / %.0f MiB free after pool | guard %zu cold / %.0f MiB | SLACK_MB=0 %zu cold / %.0f MiB (need >= %.0f)\n",
               n_cold(r_old), free_after_pool(bug, r_old), n_cold(r_new), free_after_pool(bug, r_new), n_cold(r_wa), free_after_pool(bug, r_wa), bug.floor_ + bug.ctx_h);
        printf("    guard reason: %s\n", r_new.reason.c_str());
        CHECK(r_old.mode == 2 && r_new.mode == 2 && r_new.xc_layers > 0, "both plans must be expert-granular");
        CHECK(free_after_pool(bug, r_old) < bug.floor_, "the old plan must reproduce the refusal (%.1f MiB after the pool)", free_after_pool(bug, r_old));
        CHECK(meets(bug, r_new), "the guarded plan must leave floor + head growth free (%.1f < %.1f)", free_after_pool(bug, r_new), bug.floor_ + bug.ctx_h);
        CHECK(!short_flag(r_new), "the guarded plan must not be short");
        CHECK(n_cold(r_new) <= n_cold(r_wa), "the guarded plan must keep at least the workaround's hot experts (%zu vs %zu)", n_cold(r_new), n_cold(r_wa));
        CHECK(r_new.reason.find("ctx guard") != std::string::npos, "the reason must say the guard engaged");
        // the same load for the server (capped context): the previous plan exactly, reason included
        scen srv; scen srv_old = srv; srv_old.guard = false;
        const auto a = pxa_place_plan(make(srv)), b = pxa_place_plan(make(srv_old));
        CHECK(same_plan(a, b) && a.reason == b.reason, "the capped (server) plan must not change");
        CHECK(n_cold(a) == n_cold(r_old), "the server plan is the old perplexity plan");
        // ub 1024: the uncapped logits (~448 MiB) exceed the whole margin even without a slack credit -> the old plan is refused too
        scen u1k = bug; u1k.ctx_l = 448; u1k.ctx_h = 206; u1k.slack = 0; u1k.need = 16310 + 40; u1k.pool = 56;
        scen u1k_old = u1k; u1k_old.guard = false;
        const auto c = pxa_place_plan(make(u1k)), d = pxa_place_plan(make(u1k_old));
        CHECK(!meets(u1k, d), "ub1024 old plan should miss the invariant (%.1f)", free_after_pool(u1k, d));
        CHECK(meets(u1k, c) && !short_flag(c), "ub1024 guarded plan must meet it (%.1f)", free_after_pool(u1k, c));
    }

    // (C) capped contexts: the previous plan exactly (ctx fields 0), over deficits, arch, cold path, slack, spec, spill
    {
        int n = 0, same = 0;
        for (double def : {120.0, 1050.0, 3500.0, 9000.0})
        for (int cc : {60, 70})
        for (bool zc : {false, true})
        for (bool counts : {true, false})
        for (double slack : {-1.0, 200.0})
        for (int sp : {0, 1}) {
            const double spec = sp ? 300.0 : 0.0, spill = sp ? 150.0 : 0.0;
            scen s; s.cc = cc; s.zc = zc; s.counts = counts; s.slack = slack; s.spec = spec; s.spill = spill;
            s.free_ = s.need + s.pool + 512 + spec - def;
            scen o = s; o.guard = false;
            const auto a = pxa_place_plan(make(s)), b = pxa_place_plan(make(o));
            ++n; const bool eq = same_plan(a, b) && a.reason == b.reason; if (eq) ++same;
            CHECK(eq, "capped plan changed: def %.0f cc %d zc %d counts %d slack %.0f spec %.0f spill %.0f (%zu vs %zu cold, %.1f vs %.1f MiB free after pool)",
                       def, cc, zc, counts, slack, spec, spill, n_cold(a), n_cold(b), free_after_pool(s, a), free_after_pool(o, b));
        }
        printf("(C) capped contexts: %d/%d plans identical\n", same, n);
    }

    // (D) all-outputs contexts: invariant or STILL SHORT; unchanged wherever the old plan already met it
    {
        int n = 0, met_old = 0, unchanged = 0, guarded = 0, short_n = 0;
        const double ctx_pairs[4][2] = {{0, 204}, {96, 120}, {192, 204}, {768, 204}};
        const int cc = 70;
        for (double def : {300.0, 1050.0, 3500.0})
        for (bool zc : {false, true})
        for (bool counts : {true, false})
        for (double slack : {-1.0, 0.0, 800.0})
        for (const auto & cp : ctx_pairs)
        for (double fl : {128.0, 256.0}) {
            const double cl = cp[0], ch = cp[1];
            scen s; s.cc = cc; s.zc = zc; s.counts = counts; s.slack = slack; s.ctx_l = cl; s.ctx_h = ch; s.floor_ = fl;
            s.free_ = s.need + s.pool + 512 - def;
            scen o = s; o.guard = false;
            const auto a = pxa_place_plan(make(s)), b = pxa_place_plan(make(o));
            ++n;
            const bool ok = meets(s, a) || short_flag(a);
            CHECK(ok, "all-outputs plan misses the invariant: def %.0f cc %d zc %d counts %d slack %.0f ctx %.0f+%.0f floor %.0f: %.1f MiB free after pool",
                           def, cc, zc, counts, slack, cl, ch, fl, free_after_pool(s, a));
            if (short_flag(a)) ++short_n;
            if (b.mode != 0 && meets(s, b)) {
                ++met_old;
                const bool eq = same_plan(a, b); if (eq) ++unchanged;
                CHECK(eq, "a plan that met the invariant changed: def %.0f cc %d zc %d counts %d slack %.0f ctx %.0f+%.0f floor %.0f", def, cc, zc, counts, slack, cl, ch, fl);
            } else if (!same_plan(a, b)) ++guarded;
        }
        printf("(D) all-outputs contexts: %d plans, %d met the invariant already (%d unchanged), %d re-planned by the guard, %d still short\n",
               n, met_old, unchanged, guarded, short_n);
    }

    // (E) R1: fits resident with the logits capped but not with them full -> an all-outputs context must stream
    {
        scen s; s.ctx_l = 192; s.ctx_h = 0; s.free_ = s.need + s.pool + 128 + 100;   // fits by 100 MiB capped, short by 92 with full logits
        scen o = s; o.guard = false;
        const auto a = pxa_place_plan(make(s)), b = pxa_place_plan(make(o));
        CHECK(b.mode == 0, "old plan: resident with the cap");
        CHECK(a.mode == 2 && meets(s, a) && !short_flag(a), "guarded: must stream enough for the full logits (mode %d, %.1f MiB)", a.mode, free_after_pool(s, a));
        scen f = s; f.free_ = s.need + s.pool + 128 + 200;   // the full logits fit too
        scen fo = f; fo.guard = false;
        const auto c = pxa_place_plan(make(f)), d = pxa_place_plan(make(fo));
        CHECK(c.mode == 0 && same_plan(c, d), "the full logits fit: resident, unchanged");
        scen h = f; h.ctx_l = 0; h.ctx_h = 204; scen ho = h; ho.guard = false;   // not short with the full logits: the head term never moves R1
        CHECK(same_plan(pxa_place_plan(make(h)), pxa_place_plan(make(ho))), "a resident plan must not change for the head term");
    }

    // (F) the off switch: ctx_guard false = the previous plan byte for byte, whatever the fields hold
    {
        int n = 0, same = 0;
        for (double def : {300.0, 1050.0, 3500.0})
        for (bool zc : {false, true})
        for (double slack : {-1.0, 800.0}) {
            scen s; s.zc = zc; s.slack = slack; s.guard = false; s.ctx_l = 192; s.ctx_h = 204; s.free_ = s.need + s.pool + 512 - def;
            scen z = s; z.ctx_l = 0; z.ctx_h = 0;
            const auto a = pxa_place_plan(make(s)), b = pxa_place_plan(make(z));
            ++n; const bool eq = same_plan(a, b) && a.reason == b.reason; if (eq) ++same; CHECK(eq, "guard off changed a plan (def %.0f zc %d)", def, zc);
        }
        printf("(F) guard off: %d/%d identical\n", same, n);
    }

    // (G) whole-layer (no counts) and dense spill: same invariant
    {
        for (bool dense : {false, true}) {
            scen s; s.counts = false; s.dense = dense; s.ctx_l = 192; s.ctx_h = 204; s.free_ = s.need + s.pool + 512 - 1050;
            const auto a = pxa_place_plan(make(s));
            CHECK(a.mode == (dense ? 3 : 2) && meets(s, a) && !short_flag(a), "%s: %.1f MiB free after pool (mode %d)", dense ? "dense spill" : "whole-layer", free_after_pool(s, a), a.mode);
        }
    }

    // (H) an explicit PXA_XCACHE_SLACK_MB past the margin (capped context): no plan below the floor
    {
        scen s; s.slack = 900; s.free_ = s.need + s.pool + 512 - 1050;
        scen o = s; o.guard = false;
        const auto a = pxa_place_plan(make(s)), b = pxa_place_plan(make(o));
        CHECK(free_after_pool(o, b) < o.floor_, "old: an oversized explicit slack plans below the floor (%.1f)", free_after_pool(o, b));
        CHECK(meets(s, a), "guarded: the floor holds (%.1f)", free_after_pool(s, a));
        scen ok = s; ok.slack = 300;   // inside the margin: unchanged
        scen oko = ok; oko.guard = false;
        CHECK(same_plan(pxa_place_plan(make(ok)), pxa_place_plan(make(oko))), "an explicit slack inside the margin must not change the plan");
    }

    printf("%d checks, %s\n", checks, fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
