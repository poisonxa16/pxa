// test-pxa-xcache-ring-lend.cpp -- R3a ring lending in the expert-cache planner (src/llama-pxa-place.cpp, PXA_XCACHE_RING_LEND,
// 2026-10-25): with the CPU cold path the cold stacks' ring is reported but not charged, so its bytes go to resident experts; lend
// off and the zero-copy cold path keep the previous plan exactly. The backend half (on-demand ring, zero-copy fallback) is in
// ggml/src/ggml-cuda/pxa/pxa-stream.cuh and is checked on a card (tests/run-xcache-ring-lend.sh).
// build: g++ -O1 -std=c++17 -Iinclude -Iggml/include -Iggml/src -Isrc tests/test-pxa-xcache-ring-lend.cpp src/llama-pxa-place.cpp -o /tmp/t
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

static pxa_place_input make_input(int cc, bool zc, size_t deficit_mib, bool ring1, size_t slack_mib, bool lend = false) {
    pxa_place_input in;
    in.n_layer = NL; in.mode_req = 4; in.n_expert = NE; in.n_expert_used = 8;
    in.xc_cold_pxqn = !zc; in.xc_cold_zc = zc;
    in.floor_bytes = 128u*MiB;
    in.xc_ring1_auto = ring1; in.xc_slack_auto = slack_mib*MiB;
    in.xc_ring_lend = lend;
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


static size_t n_cold(const pxa_place_result & r) { size_t n = 0; for (const auto & v : r.xc_cold) n += v.size(); return n; }

int main() {
    unsetenv("PXA_XCACHE_COST_LAYER_US"); unsetenv("PXA_XCACHE_COST_MISS_US"); unsetenv("PXA_XCACHE_ASYNC"); unsetenv("PXA_XCACHE_PLANNER");
    unsetenv("PXA_XCACHE_LAYERS"); unsetenv("PXA_XCACHE_LAYER_WEIGHT"); unsetenv("PXA_XCACHE_SMALLTAIL_SM60");
    // (1) lend off = the plan exactly as before (a hand-built input never sets it)
    const auto base = pxa_place_plan(make_input(70, false, 9000, true, 0));
    const auto off  = pxa_place_plan(make_input(70, false, 9000, true, 0, false));
    CHECK(same_plan(base, off) && !off.xc_ring_lent, "lend off changed the plan");
    CHECK(base.mode == 2 && base.xc_layers > 0 && base.dev_ring[0] > 0, "the base plan must be expert-granular with a ring");
    // (2) lend on, CPU cold path, large deficit (the G30 shape: two-layer ring): the ring is reported, flagged lent and not charged,
    //     so about its bytes go back to resident experts; never more than the ring (+ one layer's worth of rounding)
    const auto lend = pxa_place_plan(make_input(70, false, 9000, true, 0, true));
    printf("base: %zu cold, ring %.0f MiB, streamed %.0f MiB, L=%d cap=%d miss %.2f%%\n", n_cold(base), base.dev_ring[0]/(double)MiB,
           base.streamed/(double)MiB, base.xc_layers, base.xc_cap, 100.0*base.xc_miss);
    printf("lend: %zu cold, ring %.0f MiB (lent), streamed %.0f MiB, L=%d cap=%d miss %.2f%%\n", n_cold(lend), lend.dev_ring[0]/(double)MiB,
           lend.streamed/(double)MiB, lend.xc_layers, lend.xc_cap, 100.0*lend.xc_miss);
    CHECK(lend.mode == 2 && lend.xc_ring_lent, "lend on must flag the ring lent");
    CHECK(n_cold(lend) < n_cold(base) && lend.streamed < base.streamed, "lending must keep more experts resident (%zu vs %zu)", n_cold(lend), n_cold(base));
    CHECK(lend.xc_miss < base.xc_miss, "more resident experts must lower the predicted miss (%.4f vs %.4f)", lend.xc_miss, base.xc_miss);
    const double moved = ((double) base.streamed - (double) lend.streamed)/MiB;
    CHECK(moved > 0.8*base.dev_ring[0]/(double)MiB && moved <= base.dev_ring[0]/(double)MiB + 64,
          "lending moved %.0f MiB of experts for a %.0f MiB ring", moved, base.dev_ring[0]/(double)MiB);
    // the reported (on-demand) ring is the two-layer rule of the lent plan's own cold set
    const size_t want = 2*max_cold(lend)*PER_EXPERT + max_cold(lend)*(PER_EXPERT/3) + 32*MiB;
    CHECK(lend.dev_ring[0] == want, "lent ring %zu != %zu", lend.dev_ring[0], want);
    // the plan still frees the deficit: streamed bytes alone (no ring charge) cover it
    CHECK(lend.streamed + 64*MiB >= 9000*MiB, "the lent plan does not free the deficit (%.0f MiB)", lend.streamed/(double)MiB);
    // (3) the small-tail scope keeps its one-layer ring size, lent
    {
        const auto st  = pxa_place_plan(make_input(70, false, 3000, true, 0));
        const auto stl = pxa_place_plan(make_input(70, false, 3000, true, 0, true));
        const size_t w1 = max_cold(stl)*PER_EXPERT + max_cold(stl)*(PER_EXPERT/3) + 32*MiB;
        CHECK(stl.xc_ring_lent && stl.dev_ring[0] == w1, "small tail: lent one-layer ring %zu != %zu", stl.dev_ring[0], w1);
        CHECK(stl.streamed < st.streamed, "small tail: lending must keep more resident");
    }
    // (4) zero-copy cold path: the ring is read from 65 tokens up, never lent: the plan exactly as before
    {
        const auto zc  = pxa_place_plan(make_input(70, true, 9000, true, 0));
        const auto zcl = pxa_place_plan(make_input(70, true, 9000, true, 0, true));
        CHECK(same_plan(zc, zcl) && !zcl.xc_ring_lent, "zero-copy plan changed by lending");
    }
    // (5) an explicit slack credit stacks with the lent ring (G30: SLACK_MB=1000 + lend)
    {
        auto a = make_input(70, false, 9000, true, 0, true); a.xc_slack_credit = 1000*MiB;
        const auto ra = pxa_place_plan(a);
        const double m2 = ((double) lend.streamed - (double) ra.streamed)/MiB;
        CHECK(m2 > 1000 - 64 && m2 < 1000 + 64, "slack 1000 on top of lending moved %.0f MiB", m2);
    }
    // (6) a fitting model stays resident, lend or not
    CHECK(pxa_place_plan(make_input(70, false, 0, true, 0, true)).mode == 0, "a fitting model must stay resident");
    printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
