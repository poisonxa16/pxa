// test-pxa-xcache-plan-v2.cpp -- standalone check of the expert-cache planner changes of (src/llama-pxa-place.cpp):
//   (1) the speculative-chain reserve (pxa_place_input::spec_reserve) is taken out of the experts' share: the plan streams at least that much more,
//   (2) PXA_XCACHE_PLAN_V2 off (the default) is the previous plan whatever the verify width is,
//   (3) PLAN_V2 on with a verify width of 4 never has more misses than the same plan at width 1, and spreads over at least as many layers,
//   (4) the async cold path (PXA_XCACHE_ASYNC) prices a cold layer lower than the synchronous one, so the plan can only spread wider.
// build: g++ -O1 -std=c++17 -Iinclude -Iggml/include -Iggml/src -Isrc tests/test-pxa-xcache-plan-v2.cpp src/llama-pxa-place.cpp -o /tmp/test-xc-plan-v2
#include "llama-pxa-place.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" bool ggml_pxqn_cpu_mmv_available(void) { return true; }

static const int NL = 40, NE = 512, MiB = 1 << 20;

struct rng { uint64_t s; uint32_t next() { s = s*6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 33); } double u() { return (next() + 0.5)/2147483648.0; } };

static pxa_place_input make_input(size_t deficit_mib, size_t reserve_mib, bool v2, double width) {
    pxa_place_input in;
    in.n_layer = NL; in.mode_req = 4; in.n_expert = NE; in.n_expert_used = 10;
    in.xc_cold_pxqn = true; in.xc_cold_zc = false;
    in.floor_bytes = 128u*MiB;
    pxa_place_device d; d.id = 0; d.cc = 60;
    const size_t per_expert = 3*(349525u);                  // ~1 MiB per expert over the three tensors (a whole number of bytes per expert and tensor)
    const size_t experts = (size_t) NL*NE*per_expert;
    d.need_resident = experts + 3000u*MiB;
    d.free_bytes = d.need_resident + 128u*MiB + 512u*MiB - deficit_mib*MiB;   // short by exactly deficit_mib
    d.pool_need = 0;
    in.devices.push_back(d);
    if (reserve_mib) in.spec_reserve.assign(1, reserve_mib*MiB);
    rng r{12345};
    in.xc_counts.assign(NL, {});
    for (int l = 0; l < NL; ++l) {
        in.xc_counts[l].resize(NE);
        for (int e = 0; e < NE; ++e) {
            double g = 0; for (int k = 0; k < 12; ++k) g += r.u(); g -= 6;
            in.xc_counts[l][e] = std::exp(1.2*g)*40.0;
        }
        for (int t = 0; t < 3; ++t) {
            pxa_place_tensor pt; pt.name = "blk." + std::to_string(l) + (t == 0 ? ".ffn_up_exps.weight" : t == 1 ? ".ffn_gate_exps.weight" : ".ffn_down_exps.weight");
            pt.layer = l; pt.device = 0; pt.kind = PXA_PLACE_EXPERTS; pt.bytes = (size_t) NE*(per_expert/3);
            in.tensors.push_back(pt);
        }
    }
    in.xc_plan_v2 = v2; in.xc_verify_width = width;
    return in;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main() {
    unsetenv("PXA_XCACHE_COST_LAYER_US"); unsetenv("PXA_XCACHE_COST_MISS_US"); unsetenv("PXA_XCACHE_ASYNC"); unsetenv("PXA_XCACHE_VERIFY_KAPPA");
    // (1) the reserve
    auto a = pxa_place_plan(make_input(6000, 0, false, 1.0));
    auto b = pxa_place_plan(make_input(6000, 700, false, 1.0));
    CHECK(a.mode == 2 && b.mode == 2, "both plans stream experts (%d %d)", a.mode, b.mode);
    const double sa = a.streamed/(double) MiB, sb = b.streamed/(double) MiB;
    printf("reserve: streamed %.0f MiB without, %.0f MiB with a 700 MiB reserve (ring %.0f / %.0f)\n", sa, sb, a.dev_ring[0]/(double) MiB, b.dev_ring[0]/(double) MiB);
    CHECK(sb >= sa + 650.0, "the reserve moved only %.0f MiB of experts out", sb - sa);
    CHECK(a.xc_layers > 0, "the expert-granular plan was not chosen");
    // a model that fits resident keeps today's plan whatever the reserve
    auto f = pxa_place_plan(make_input(0, 700, false, 1.0));
    CHECK(f.mode == 0, "a fitting model must stay resident (mode %d)", f.mode);
    // (2) PLAN_V2 off ignores the width
    auto o1 = pxa_place_plan(make_input(6000, 0, false, 1.0));
    auto o4 = pxa_place_plan(make_input(6000, 0, false, 4.0));
    CHECK(o1.xc_layers == o4.xc_layers && o1.xc_cap == o4.xc_cap && o1.xc_miss == o4.xc_miss, "off: the width changed the plan");
    // (3) PLAN_V2 on
    auto v1 = pxa_place_plan(make_input(6000, 0, true, 1.0));
    auto v4 = pxa_place_plan(make_input(6000, 0, true, 4.0));
    printf("v2 width 1: L=%d cap=%d miss=%.2f%%   width 4: L=%d cap=%d miss=%.2f%%   off: L=%d miss=%.2f%%\n", v1.xc_layers, v1.xc_cap, 100*v1.xc_miss, v4.xc_layers, v4.xc_cap, 100*v4.xc_miss, o1.xc_layers, 100*o1.xc_miss);
    CHECK(v4.xc_miss <= v1.xc_miss + 1e-12, "width 4 missed more than width 1");
    CHECK(v4.xc_layers >= v1.xc_layers, "width 4 used fewer cold layers than width 1");
    CHECK(v1.xc_cost_layer_us == 700.0, "v2 layer price %.0f", v1.xc_cost_layer_us);
    // (4) async prices the layer lower
    setenv("PXA_XCACHE_ASYNC", "1", 1);
    auto s4 = pxa_place_plan(make_input(6000, 0, true, 4.0));
    unsetenv("PXA_XCACHE_ASYNC");
    printf("v2 + async width 4: L=%d miss=%.2f%% layer price %.0f us\n", s4.xc_layers, 100*s4.xc_miss, s4.xc_cost_layer_us);
    CHECK(s4.xc_cost_layer_us < v4.xc_cost_layer_us, "async layer price %.0f not below %.0f", s4.xc_cost_layer_us, v4.xc_cost_layer_us);
    CHECK(s4.xc_layers >= v4.xc_layers, "async spread over fewer layers");
    printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
