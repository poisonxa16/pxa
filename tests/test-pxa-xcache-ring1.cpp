// test-pxa-xcache-ring1.cpp -- standalone check of the C042 small-tail rule of the expert-cache planner (src/llama-pxa-place.cpp,
// 2026-10-18): on ONE sm_70 device with a CPU cold path and a deficit of at most half its free VRAM,
//   (1) the cold stacks' ring holds one layer (largest cold layer + largest cold tensor + 32 MiB) instead of two,
//   (2) the auto slack credit (pxa_place_input::xc_slack_auto) moves more experts back to VRAM,
//   (3) outside the scope (sm_60, zero-copy cold path, a large deficit, the switches off) the plan is the previous one exactly.
// build: g++ -O1 -std=c++17 -Iinclude -Iggml/include -Iggml/src -Isrc tests/test-pxa-xcache-ring1.cpp src/llama-pxa-place.cpp -o /tmp/test-xc-ring1
#include "llama-pxa-place.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" bool ggml_pxqn_cpu_mmv_available(void) { return true; }

static const int NL = 40, NE = 256;
static const size_t MiB = 1u << 20;
static const size_t PER_EXPERT = 3*(559241u);   // ~1.6 MiB per expert over the three tensors (35B-A3B PXQN4 class)

struct rng { uint64_t s; uint32_t next() { s = s*6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 33); } double u() { return (next() + 0.5)/2147483648.0; } };

static pxa_place_input make_input(int cc, bool zc, size_t deficit_mib, bool ring1, size_t slack_mib) {
    pxa_place_input in;
    in.n_layer = NL; in.mode_req = 4; in.n_expert = NE; in.n_expert_used = 8;
    in.xc_cold_pxqn = !zc; in.xc_cold_zc = zc;
    in.floor_bytes = 128u*MiB;
    in.xc_ring1_auto = ring1; in.xc_slack_auto = slack_mib*MiB;
    pxa_place_device d; d.id = 0; d.cc = cc;
    const size_t experts = (size_t) NL*NE*PER_EXPERT;
    d.need_resident = experts + 2000u*MiB;
    d.free_bytes = d.need_resident + 512u*MiB - deficit_mib*MiB;   // the planner's deficit is exactly deficit_mib
    d.pool_need = 0;
    in.devices.push_back(d);
    rng r{4242};
    in.xc_counts.assign(NL, {});
    for (int l = 0; l < NL; ++l) {
        in.xc_counts[l].resize(NE);
        for (int e = 0; e < NE; ++e) {
            double g = 0; for (int k = 0; k < 12; ++k) g += r.u(); g -= 6;
            in.xc_counts[l][e] = std::exp(1.2*g)*40.0;
        }
        for (int t = 0; t < 3; ++t) {
            pxa_place_tensor pt; pt.name = "blk." + std::to_string(l) + (t == 0 ? ".ffn_up_exps.weight" : t == 1 ? ".ffn_gate_exps.weight" : ".ffn_down_exps.weight");
            pt.layer = l; pt.device = 0; pt.kind = PXA_PLACE_EXPERTS; pt.bytes = (size_t) NE*(PER_EXPERT/3);
            in.tensors.push_back(pt);
        }
    }
    return in;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static bool same_plan(const pxa_place_result & a, const pxa_place_result & b) {
    return a.mode == b.mode && a.xc_cold == b.xc_cold && a.dev_ring == b.dev_ring && a.streamed == b.streamed && a.xc_cap == b.xc_cap;
}
static size_t max_cold(const pxa_place_result & r) {
    size_t m = 0;
    for (const auto & v : r.xc_cold) m = std::max(m, v.size());
    return m;
}

int main() {
    unsetenv("PXA_XCACHE_COST_LAYER_US"); unsetenv("PXA_XCACHE_COST_MISS_US"); unsetenv("PXA_XCACHE_ASYNC"); unsetenv("PXA_XCACHE_PLANNER");
    unsetenv("PXA_XCACHE_LAYERS"); unsetenv("PXA_XCACHE_LAYER_WEIGHT"); unsetenv("PXA_XCACHE_SMALLTAIL_SM60");
    // the scope predicate
    CHECK( pxa_xc_small_tail(1, 70, false, 3000*MiB, 15800*MiB), "1x V100 cpu 3000/15800 must be in scope");
    CHECK(!pxa_xc_small_tail(1, 60, false, 3000*MiB, 15800*MiB), "sm_60 must be out of scope");
    CHECK(!pxa_xc_small_tail(1, 70, true,  3000*MiB, 15800*MiB), "zero-copy cold path must be out of scope");
    CHECK(!pxa_xc_small_tail(2, 70, false, 3000*MiB, 15800*MiB), "two devices must be out of scope");
    CHECK(!pxa_xc_small_tail(1, 70, false, 9000*MiB, 15800*MiB), "a deficit above half the free VRAM must be out of scope");
    CHECK(!pxa_xc_small_tail(1, 70, false, 0, 15800*MiB), "no deficit, no rule");
    // PXA_XCACHE_SMALLTAIL_SM60=1 (the registry's PXA_AUTO_FLINT_1CARD) admits one sm_60 card under the same bounds, nothing else
    setenv("PXA_XCACHE_SMALLTAIL_SM60", "1", 1);
    CHECK( pxa_xc_small_tail(1, 60, false, 3000*MiB, 15800*MiB), "SMALLTAIL_SM60=1: 1x P100 cpu 3000/15800 must be in scope");
    CHECK(!pxa_xc_small_tail(1, 61, false, 3000*MiB, 15800*MiB), "SMALLTAIL_SM60=1: sm_61 stays out of scope");
    CHECK(!pxa_xc_small_tail(2, 60, false, 3000*MiB, 15800*MiB), "SMALLTAIL_SM60=1: two P100s stay out of scope");
    CHECK(!pxa_xc_small_tail(1, 60, true,  3000*MiB, 15800*MiB), "SMALLTAIL_SM60=1: zero-copy stays out of scope");
    CHECK(!pxa_xc_small_tail(1, 60, false, 9000*MiB, 15800*MiB), "SMALLTAIL_SM60=1: deficit above half the free VRAM stays out");
    setenv("PXA_XCACHE_SMALLTAIL_SM60", "0", 1);
    CHECK(!pxa_xc_small_tail(1, 60, false, 3000*MiB, 15800*MiB), "SMALLTAIL_SM60=0: sm_60 out of scope");
    unsetenv("PXA_XCACHE_SMALLTAIL_SM60");

    const auto old = pxa_place_plan(make_input(70, false, 3000, false, 0));
    const auto r1  = pxa_place_plan(make_input(70, false, 3000, true,  0));
    const auto r1s = pxa_place_plan(make_input(70, false, 3000, true,  PXA_XCACHE_SLACK_AUTO_MB));
    CHECK(old.mode == 2 && r1.mode == 2 && r1s.mode == 2, "all three plans stream experts (%d %d %d)", old.mode, r1.mode, r1s.mode);
    CHECK(old.xc_layers > 0 && r1.xc_layers > 0, "the expert-granular plan was not chosen");
    size_t n_old = 0, n_r1 = 0, n_r1s = 0;
    for (const auto & v : old.xc_cold) n_old += v.size();
    for (const auto & v : r1.xc_cold)  n_r1  += v.size();
    for (const auto & v : r1s.xc_cold) n_r1s += v.size();
    printf("two-layer ring: %zu cold, ring %.0f MiB, streamed %.0f MiB, L=%d cap=%d\n", n_old, old.dev_ring[0]/(double)MiB, old.streamed/(double)MiB, old.xc_layers, old.xc_cap);
    printf("one-layer ring: %zu cold, ring %.0f MiB, streamed %.0f MiB, L=%d cap=%d\n", n_r1, r1.dev_ring[0]/(double)MiB, r1.streamed/(double)MiB, r1.xc_layers, r1.xc_cap);
    printf("  + slack %d : %zu cold, ring %.0f MiB, streamed %.0f MiB, L=%d cap=%d\n", PXA_XCACHE_SLACK_AUTO_MB, n_r1s, r1s.dev_ring[0]/(double)MiB, r1s.streamed/(double)MiB, r1s.xc_layers, r1s.xc_cap);
    // (1) the ring is one layer: largest cold layer + largest cold tensor + 32 MiB (the three expert tensors are equal here)
    const size_t want_r1 = max_cold(r1)*PER_EXPERT + max_cold(r1)*(PER_EXPERT/3) + 32*MiB;
    const size_t want_old = 2*max_cold(old)*PER_EXPERT + max_cold(old)*(PER_EXPERT/3) + 32*MiB;
    CHECK(r1.dev_ring[0] == want_r1, "one-layer ring %zu != %zu", r1.dev_ring[0], want_r1);
    CHECK(old.dev_ring[0] == want_old, "two-layer ring %zu != %zu", old.dev_ring[0], want_old);
    CHECK(n_r1 < n_old && r1.streamed < old.streamed, "the one-layer ring must keep more experts resident (%zu vs %zu)", n_r1, n_old);
    // VRAM the experts + ring take never grows past the old plan's (the freed ring goes to hot experts, not past the margin)
    CHECK((old.streamed - r1.streamed) <= (old.dev_ring[0] - r1.dev_ring[0]) + 64*MiB,
          "the plan took back %.0f MiB of experts for a %.0f MiB smaller ring", (old.streamed - r1.streamed)/(double)MiB, (old.dev_ring[0] - r1.dev_ring[0])/(double)MiB);
    // (2) the slack credit keeps still more experts resident, by about the credit
    CHECK(n_r1s < n_r1, "the slack credit must keep more experts resident (%zu vs %zu)", n_r1s, n_r1);
    const double moved = ((double) r1.streamed - (double) r1s.streamed + (double) r1s.dev_ring[0] - (double) r1.dev_ring[0])/MiB;
    CHECK(moved > PXA_XCACHE_SLACK_AUTO_MB - 64 && moved < PXA_XCACHE_SLACK_AUTO_MB + 64, "the credit moved %.0f MiB, not ~%d", moved, PXA_XCACHE_SLACK_AUTO_MB);
    // (3) outside the scope: the previous plan exactly
    CHECK(same_plan(pxa_place_plan(make_input(60, false, 3000, true, PXA_XCACHE_SLACK_AUTO_MB)), pxa_place_plan(make_input(60, false, 3000, false, 0))), "sm_60 plan changed");
    CHECK(same_plan(pxa_place_plan(make_input(70, true, 3000, true, PXA_XCACHE_SLACK_AUTO_MB)), pxa_place_plan(make_input(70, true, 3000, false, 0))), "zero-copy plan changed");
    CHECK(same_plan(pxa_place_plan(make_input(70, false, 9000, true, PXA_XCACHE_SLACK_AUTO_MB)), pxa_place_plan(make_input(70, false, 9000, false, 0))), "large-deficit plan changed");
    // an explicit PXA_XCACHE_SLACK_MB (xc_slack_credit) still applies everywhere, as before
    {
        auto a = make_input(60, false, 3000, false, 0); a.xc_slack_credit = 384*MiB;
        auto b = make_input(60, false, 3000, false, 0);
        CHECK(pxa_place_plan(a).streamed < pxa_place_plan(b).streamed, "an explicit slack credit must still move experts on sm_60");
    }
    // (4) C042 small tail on the async CPU-pool cold path (xc_async_cost, 2026-10-19): the per-layer price drops to
    //     PXA_XC_SMALLTAIL_ASYNC_LAYER_US, so the cold set spreads over more layers with a lower miss; outside the scope nothing moves
    {
        auto a = make_input(70, false, 3000, true, PXA_XCACHE_SLACK_AUTO_MB); a.xc_async_cost = true;
        const auto ra = pxa_place_plan(a);
        printf("  + async cost: L=%d cap=%d miss=%.2f%% ring %.0f MiB (layer price %.0f us) vs L=%d miss=%.2f%%\n", ra.xc_layers, ra.xc_cap,
               100.0*ra.xc_miss, ra.dev_ring[0]/(double)MiB, ra.xc_cost_layer_us, r1s.xc_layers, 100.0*r1s.xc_miss);
        CHECK(ra.xc_cost_layer_us < r1s.xc_cost_layer_us, "the async small-tail price must be below the split path's (%.0f vs %.0f)", ra.xc_cost_layer_us, r1s.xc_cost_layer_us);
        CHECK(ra.xc_layers > r1s.xc_layers && ra.xc_miss < r1s.xc_miss, "the async price must spread the cold set (L %d vs %d, miss %.4f vs %.4f)",
              ra.xc_layers, r1s.xc_layers, ra.xc_miss, r1s.xc_miss);
        auto b60 = make_input(60, false, 3000, true, PXA_XCACHE_SLACK_AUTO_MB); b60.xc_async_cost = true;
        CHECK(same_plan(pxa_place_plan(b60), pxa_place_plan(make_input(60, false, 3000, true, PXA_XCACHE_SLACK_AUTO_MB))), "sm_60 plan changed by the async price");
        auto bzc = make_input(70, true, 3000, true, PXA_XCACHE_SLACK_AUTO_MB); bzc.xc_async_cost = true;
        CHECK(same_plan(pxa_place_plan(bzc), pxa_place_plan(make_input(70, true, 3000, true, PXA_XCACHE_SLACK_AUTO_MB))), "zero-copy plan changed by the async price");
        auto bbig = make_input(70, false, 9000, true, PXA_XCACHE_SLACK_AUTO_MB); bbig.xc_async_cost = true;
        CHECK(same_plan(pxa_place_plan(bbig), pxa_place_plan(make_input(70, false, 9000, true, PXA_XCACHE_SLACK_AUTO_MB))), "large-deficit plan changed by the async price");
        setenv("PXA_XCACHE_COST_LAYER_US", "1300", 1);   // an explicit price still wins
        CHECK(same_plan(pxa_place_plan(a), r1s), "PXA_XCACHE_COST_LAYER_US must override the async small-tail price");
        unsetenv("PXA_XCACHE_COST_LAYER_US");
    }
    // a fitting model stays resident
    CHECK(pxa_place_plan(make_input(70, false, 0, true, PXA_XCACHE_SLACK_AUTO_MB)).mode == 0, "a fitting model must stay resident");
    printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
