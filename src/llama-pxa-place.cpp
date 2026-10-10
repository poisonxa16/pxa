// pxa / PXA placement planner -- authored by PXA Network (https://pxanetwork.com).
// See llama-pxa-place.h for the decision rules.
#include "llama-pxa-place.h"
#include "ggml.h"
#include "ggml-pxqn-api.h"   // ggml_pxqn_cpu_mmv_available

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

static const size_t MiB = size_t(1) << 20;

size_t pxa_place_floor_bytes() {
    long long mb = PXA_POOL_FLOOR_MB_DEFAULT;
    if (const char * e = getenv("PXA_POOL_FLOOR_MB")) mb = std::max(0LL, atoll(e));
    return (size_t)mb * MiB;
}

size_t pxa_place_ring_bytes(const std::vector<pxa_place_tensor> & sel) {
    if (sel.empty()) return 0;
    std::map<int, size_t> per_layer;
    size_t mt = 0;
    for (const auto & t : sel) { per_layer[t.layer] += t.bytes; mt = std::max(mt, t.bytes); }
    size_t ml = 0;
    for (auto & kv : per_layer) ml = std::max(ml, kv.second);
    return 2*ml + mt + 32*MiB;
}

// k of n spread evenly (the loader's Bresenham quota: layer i is picked when k*(i+1)/n steps
// past k*i/n); returns positions into [0, n)
static std::vector<int> spread(int k, int n) {
    std::vector<int> r;
    if (n <= 0 || k <= 0) return r;
    for (int i = 0; i < n; ++i) if ((int64_t)k*(i + 1)/n > (int64_t)k*i/n) r.push_back(i);
    return r;
}

// R3x (PXA_XCACHE_PLANNER=r3): cold experts from every layer, least routed first under a per-layer cap
static bool pxa_xc_choose_r3x(const std::vector<int> & layers, const std::vector<size_t> & expert_bytes,
        const std::vector<size_t> & max_tensor_bytes, const std::vector<std::vector<double>> & counts,
        int n_expert, size_t target, pxa_xc_choice & out) {
    const int nl = (int)layers.size();
    if (nl == 0 || n_expert <= 1 || (int)counts.size() != nl) return false;
    struct item { double c; int l, e; };
    std::vector<item> items;
    items.reserve((size_t)nl*n_expert);
    double total = 0;
    for (int l = 0; l < nl; ++l) {
        if ((int)counts[l].size() != n_expert) return false;
        for (int e = 0; e < n_expert; ++e) { items.push_back({counts[l][e], l, e}); total += counts[l][e]; }
    }
    // least routed first; ties by (layer, expert) so the choice is a pure function of the counts
    std::sort(items.begin(), items.end(), [](const item & a, const item & b) {
        if (a.c != b.c) return a.c < b.c;
        if (a.l != b.l) return a.l < b.l;
        return a.e < b.e;
    });
    // A per-layer cap trades routing mass against the ring: the ring holds two layers' cold bytes,
    // so a flat layer that takes many cold experts grows it for every layer. Try every cap and keep
    // the feasible selection that leaves the least routed mass on the cold side.
    bool found = false;
    double best_miss = 0; size_t best_ring = 0; int best_cap = 0;
    std::vector<int> cnt(nl);
    for (int cap = 1; cap < n_expert; ++cap) {
        std::fill(cnt.begin(), cnt.end(), 0);
        size_t bytes = 0, max_l = 0, max_t = 0; double miss = 0;
        bool ok = false;
        for (const auto & it : items) {
            if (cnt[it.l] >= cap) continue;
            ++cnt[it.l];
            bytes += expert_bytes[it.l];
            miss  += it.c;
            max_l = std::max(max_l, (size_t)cnt[it.l]*expert_bytes[it.l]);
            max_t = std::max(max_t, (size_t)cnt[it.l]*max_tensor_bytes[it.l]);
            const size_t ring = 2*max_l + max_t + 32*MiB;
            if (bytes > ring && bytes - ring >= target) {
                ok = true;
                if (!found || miss < best_miss || (miss == best_miss && ring < best_ring)) {
                    found = true; best_miss = miss; best_ring = ring; best_cap = cap;
                }
                break;
            }
        }
        if (!ok && cap*(size_t)nl >= items.size()) break;   // every expert already allowed: larger caps change nothing
    }
    if (!found) return false;
    // replay the winning cap to materialise the selection
    out = pxa_xc_choice();
    out.cold.assign(nl, {});
    std::fill(cnt.begin(), cnt.end(), 0);
    size_t bytes = 0, max_l = 0, max_t = 0; double miss = 0;
    for (const auto & it : items) {
        if (cnt[it.l] >= best_cap) continue;
        ++cnt[it.l];
        bytes += expert_bytes[it.l];
        miss  += it.c;
        max_l = std::max(max_l, (size_t)cnt[it.l]*expert_bytes[it.l]);
        max_t = std::max(max_t, (size_t)cnt[it.l]*max_tensor_bytes[it.l]);
        out.cold[it.l].push_back(it.e);
        const size_t ring = 2*max_l + max_t + 32*MiB;
        if (bytes > ring && bytes - ring >= target) { out.ring = ring; break; }
    }
    for (auto & v : out.cold) std::sort(v.begin(), v.end());
    out.cold_bytes = bytes;
    out.miss  = miss;
    out.total = total;
    out.cap   = best_cap;
    for (auto & v : out.cold) out.n_layers += v.empty() ? 0 : 1;
    return true;
}

bool pxa_type_cpu_fast(int t) {
    switch ((ggml_type)t) {
        // PXQN: fast when the closed library carries its CPU matmul (ggml_pxqn_cpu_mmv_available);
        // an older library, or PXA_PXQN_CPU_MMV=0, has only the row-dequant path (zero-copy over PCIe wins there)
        case GGML_TYPE_PXQN1: case GGML_TYPE_PXQN2: case GGML_TYPE_PXQN3: case GGML_TYPE_PXQN3S8:
        case GGML_TYPE_PXQN4: case GGML_TYPE_PXQN4S8: case GGML_TYPE_PXQN5: case GGML_TYPE_PXA4:
            return ggml_pxqn_cpu_mmv_available();
        case GGML_TYPE_PXQ6:
            return false;
        default:
            return true;
    }
}

// PXQN experts read on the CPU by the library's matmul (2026-10-04, Flash-Next fn32 PXQN-32 on ONE P100, -sm layer, -t 16):
//   the CPU reads a cold layer's routed experts at ~20 GB/s effective (the matmul alone streams 30-39 GB/s over 16 threads; the cold
//   path's narrow ops, ids and hand-offs cost the rest), and every layer that holds a cold expert costs ~1.3 ms per token in
//   fixed overhead (two GPU<->CPU hand-offs, ~7 CPU ops each with a thread barrier) against ~0.4 ms of CPU math for 10 routed
//   experts. A cold layer is therefore about 4x more expensive in overhead than in bytes, so the plan prefers FEW layers that are
//   (nearly) entirely cold over many layers with a few cold experts: whole-layer streaming 20.8 vs the L=48 partial split 13.5 tok/s.
#define PXA_XC_PXQN_CPU_GBS      20.0
#define PXA_XC_PXQN_CPU_LAYER_US 1300.0
// PXA_XCACHE_PLAN_V2: the same constant re-fit on the rel-next head (fn32 on ONE P100, quiet box, REPS 3, static plan 20.5 t/s = 48.8 ms per
// token with 31 cold layers): the cold path's fixed cost measured ~0.7 ms per cold layer (two host syncs, the CPU op chain and the merge;
// , mail #14095), against the 1.3 ms the first measurement (a loaded box) priced.
#define PXA_XC_PXQN_CPU_LAYER_US_V2 700.0
// With PXA_XCACHE_ASYNC the cold layer no longer stops the decode twice (no scheduler split, no host sync, no copy call): what is left of its
// fixed cost is the submit / wait kernels and the worker's op chain (~8 CPU ops with a thread barrier each). A first estimate, to be replaced by the
// measured value (PXA_XCACHE_COST_LAYER_US overrides it).
#define PXA_XC_PXQN_CPU_LAYER_US_V2_ASYNC 250.0

double pxa_xc_width_factor(double width) {
    if (!(width > 1.0)) return 1.0;
    double kappa = 0.75;
    if (const char * e = getenv("PXA_XCACHE_VERIFY_KAPPA")) kappa = std::min(1.0, std::max(0.0, atof(e)));
    return 1.0 + kappa*(width - 1.0);
}

// R4 decode-cost table (2026-09-28 data; this box: every card on PCIe x4, 8 quiet CPU threads).
//   miss_us = (bytes a 1% miss moves per token) / effective read rate, where a 1% miss moves
//             0.01 x n_used x sum over the device's candidate layers of one expert's bytes:
//     CPU cold path  3.78 GB/s: the least-squares fit on Ornith-35B PXQ4, one V100, L16/20/24/30
//                    (1420 us per % at 40 layers x 1.60 MiB x top-8, the constant R4 shipped with).
//     zero-copy      3.29 GB/s: Ornith-35B PXQ4, one V100, zero-copy L10 (15.8% miss) 19.49 t/s and
//                    L16 (4.8%) 23.50 t/s, solved against the CPU fit's 10.2 ms base -> 1631 us per %
//                    at the same geometry = PCIe x4 read rate, so it holds for any model's bytes.
//   layer_us = fixed cost per layer that holds a cold expert (hand-off / extra launches + merge):
//     CPU cold path  161 us (the same Ornith V100 fit; no P100 CPU-path sweep, same value used).
//     zero-copy      sm_70 1530 us (the same two Ornith zero-copy points);
//                    sm_60  585 us (Flash-Next PXQN 2x P100 -sm layer, 48 cold layers, ~10.5% miss,
//                    12.43 t/s, base taken as the 4x P100 resident 36.26 t/s; one point, so coarse).
//   PXA_XCACHE_COST_LAYER_US / PXA_XCACHE_COST_MISS_US override either constant.
pxa_xc_cost pxa_xc_cost_for(int cc, bool zc, int n_used, const std::vector<size_t> & expert_bytes, bool pxqn_cpu) {
    pxa_xc_cost c;
    double sum = 0;
    for (size_t b : expert_bytes) sum += (double)b;
    double gbs = zc ? 3.29 : 3.78;
    if (pxqn_cpu && !zc) gbs = PXA_XC_PXQN_CPU_GBS;   // PXQN experts on the library's CPU matmul (see the table above)
    if (n_used > 0 && sum > 0) c.miss_us = 0.01*n_used*sum/(gbs*1e3);   // bytes / (GB/s) -> us
    else                       c.miss_us = zc ? 1631.0 : 1420.0;
    if (!zc)            c.layer_us = pxqn_cpu ? PXA_XC_PXQN_CPU_LAYER_US : 161.0;
    else if (cc >= 70)  c.layer_us = 1530.0;
    else                c.layer_us = 585.0;
    if (const char * e = getenv("PXA_XCACHE_COST_LAYER_US")) c.layer_us = std::max(0.0, atof(e));
    if (const char * e = getenv("PXA_XCACHE_COST_MISS_US"))  c.miss_us  = std::max(0.0, atof(e));
    return c;
}

// R4 (default): confine the cold experts to the L least-hot layers. Every layer that holds a cold
// expert pays the cold-path latency per token (a CPU dispatch or a zero-copy read + the merge), so
// spreading a 2-5% miss over all layers costs more than whole-layer placement. For each per-layer
// cap, layers are ranked by the routing mass of their `cap` least-routed experts; for L = 1, 2, ...
// the cold set is taken (least routed first, under the cap) only from the first L layers of that
// ranking and every other layer stays fully resident. At each L the least missed mass wins among
// the feasible caps, then the smaller ring. L itself is chosen by a measured decode-cost model:
//   cost(L) = PXA_XCACHE_COST_LAYER_US * (layers holding a cold expert)
//           + PXA_XCACHE_COST_MISS_US  * (percent of routings that miss)
// minimised over every feasible L (ties: fewer layers). The constants come from pxa_xc_cost_for():
// per cold path (CPU / zero-copy) and card, the miss priced as the bytes it moves per token (table
// and data above). The CPU fit holds for misses up to ~5%; above that it over-prices the miss,
// which only pushes L up. The ring is VRAM only (decode never reads it), so
// it enters as a tie-break, not a cost.
// PXA_XCACHE_LAYERS=N forces L = N (sweeps; the smallest feasible L >= N if N is not feasible).
bool pxa_xc_choose(const std::vector<int> & layers, const std::vector<size_t> & expert_bytes,
        const std::vector<size_t> & max_tensor_bytes, const std::vector<std::vector<double>> & counts,
        int n_expert, size_t target, pxa_xc_choice & out, const pxa_xc_cost & cost) {
    if (const char * pl = getenv("PXA_XCACHE_PLANNER")) {
        if (strcmp(pl, "r3") == 0 || strcmp(pl, "r3x") == 0) {
            return pxa_xc_choose_r3x(layers, expert_bytes, max_tensor_bytes, counts, n_expert, target, out);
        }
    }
    const int nl = (int)layers.size();
    if (nl == 0 || n_expert <= 1 || (int)counts.size() != nl) return false;
    // PXA_XCACHE_LAYER_WEIGHT (LEVERS-6 A1; default off = byte-identical plan): "il:w,il-il:w,..." scales the
    // routing counts of those model layers for the CHOICE of the cold set only (item order + the per-cap
    // layer rank); the reported miss stays in raw counts. The boot-corpus prior under-states how spread
    // the live routing is on some layers (layers 0-2: live hit 63-69% vs ~80% elsewhere), so w > 1 keeps
    // more of those layers' experts hot out of the same budget (pair with PXA_XCACHE_SLACK_MB to grow the
    // budget itself). Weights are clamped to [0.05, 20].
    std::vector<double> lw(nl, 1.0);
    if (const char * ws = getenv("PXA_XCACHE_LAYER_WEIGHT")) {
        std::string spec(ws);
        size_t p = 0;
        int n_set = 0;
        while (p < spec.size()) {
            size_t q = spec.find(',', p);
            if (q == std::string::npos) q = spec.size();
            const std::string tok = spec.substr(p, q - p);
            p = q + 1;
            const size_t c = tok.find(':');
            if (c == std::string::npos) continue;
            const std::string rg = tok.substr(0, c);
            const size_t dash = rg.find('-');
            const int a = atoi(rg.c_str());
            const int b = dash == std::string::npos ? a : atoi(rg.c_str() + dash + 1);
            const double w = std::min(20.0, std::max(0.05, atof(tok.c_str() + c + 1)));
            for (int l = 0; l < nl; ++l) if (layers[l] >= a && layers[l] <= b) { lw[l] = w; ++n_set; }
        }
        fprintf(stderr, "PXA_XCACHE_LAYER_WEIGHT: %d layer weight(s) set from \"%s\" (cold-set choice only)\n", n_set, ws);
    }
    struct item { double c; int l, e; double raw; };
    std::vector<item> items;
    items.reserve((size_t)nl*n_expert);
    double total = 0;
    // per layer: counts ascending, prefix sums = mass of the bottom-k experts
    std::vector<std::vector<double>> pre(nl);
    for (int l = 0; l < nl; ++l) {
        if ((int)counts[l].size() != n_expert) return false;
        std::vector<double> c = counts[l];
        if (lw[l] != 1.0) for (auto & v : c) v *= lw[l];
        std::sort(c.begin(), c.end());
        pre[l].assign(n_expert + 1, 0.0);
        for (int e = 0; e < n_expert; ++e) pre[l][e + 1] = pre[l][e] + c[e];
        for (int e = 0; e < n_expert; ++e) { items.push_back({counts[l][e]*lw[l], l, e, counts[l][e]}); total += counts[l][e]; }
    }
    std::sort(items.begin(), items.end(), [](const item & a, const item & b) {
        if (a.c != b.c) return a.c < b.c;
        if (a.l != b.l) return a.l < b.l;
        return a.e < b.e;
    });
    // layer rank per cap (least bottom-cap mass first; ties by layer so the plan is a pure function of the counts)
    std::vector<std::vector<int>> rank(n_expert);
    for (int cap = 1; cap < n_expert; ++cap) {
        auto & r = rank[cap];
        r.resize(nl);
        for (int l = 0; l < nl; ++l) r[l] = l;
        std::sort(r.begin(), r.end(), [&](int a, int b) {
            if (pre[a][cap] != pre[b][cap]) return pre[a][cap] < pre[b][cap];
            return a < b;
        });
    }
    int l0 = 1;
    bool forced = false;
    if (const char * e = getenv("PXA_XCACHE_LAYERS")) { l0 = std::max(1, std::min(nl, atoi(e))); forced = true; }
    const double c_layer = cost.layer_us, c_miss = cost.miss_us;   // pxa_xc_cost_for (table + env)
    std::vector<int> cnt(nl);
    std::vector<char> allow(nl);
    // one (L, cap) selection; returns true when feasible and fills bytes/ring/miss (and `cold` when asked)
    auto run = [&](int L, int cap, size_t & bytes_o, size_t & ring_o, double & miss_o,
                   std::vector<std::vector<int32_t>> * cold) -> bool {
        std::fill(allow.begin(), allow.end(), 0);
        for (int k = 0; k < L; ++k) allow[rank[cap][k]] = 1;
        std::fill(cnt.begin(), cnt.end(), 0);
        size_t bytes = 0, max_l = 0, max_t = 0; double miss = 0;
        for (const auto & it : items) {
            if (!allow[it.l] || cnt[it.l] >= cap) continue;
            ++cnt[it.l];
            bytes += expert_bytes[it.l];
            miss  += it.raw;
            max_l = std::max(max_l, (size_t)cnt[it.l]*expert_bytes[it.l]);
            max_t = std::max(max_t, (size_t)cnt[it.l]*max_tensor_bytes[it.l]);
            if (cold) (*cold)[it.l].push_back(it.e);
            const size_t ring = 2*max_l + max_t + 32*MiB;
            if (bytes > ring && bytes - ring >= target) { bytes_o = bytes; ring_o = ring; miss_o = miss; return true; }
        }
        return false;
    };
    bool have = false;
    int  win_L = 0, win_cap = 0; double win_cost = 0, win_miss = 0; size_t win_ring = 0;
    for (int L = l0; L <= nl; ++L) {
        bool found = false;
        double best_miss = 0; size_t best_ring = 0; int best_cap = 0;
        for (int cap = 1; cap < n_expert; ++cap) {
            size_t b = 0, rg = 0; double m = 0;
            if (!run(L, cap, b, rg, m, nullptr)) continue;
            if (!found || m < best_miss || (m == best_miss && rg < best_ring)) {
                found = true; best_miss = m; best_ring = rg; best_cap = cap;
            }
        }
        if (!found) continue;
        const double pct  = total > 0 ? 100.0*best_miss/total : 0.0;
        const double cost = c_layer*L + c_miss*pct;
        if (!have || cost < win_cost || (cost == win_cost && best_ring < win_ring)) {
            have = true; win_L = L; win_cap = best_cap; win_cost = cost; win_miss = best_miss; win_ring = best_ring;
        }
        if (forced) break;   // PXA_XCACHE_LAYERS: the first feasible L >= N
    }
    if (!have) return false;
    (void)win_miss;
    out = pxa_xc_choice();
    out.cold.assign(nl, {});
    size_t b = 0, rg = 0; double m = 0;
    run(win_L, win_cap, b, rg, m, &out.cold);
    for (auto & v : out.cold) std::sort(v.begin(), v.end());
    out.cold_bytes = b;
    out.ring  = rg;
    out.miss  = m;
    out.total = total;
    out.cap   = win_cap;
    out.cost_us = win_cost;
    for (auto & v : out.cold) out.n_layers += v.empty() ? 0 : 1;
    return true;
}

pxa_place_result pxa_place_plan(const pxa_place_input & in) {
    pxa_place_result r;
    const size_t nd = in.devices.size();
    r.xc_cold.assign(in.n_layer, {});
    r.dev_stream.assign(nd, 0);
    r.dev_deficit.assign(nd, 0);
    r.dev_ring.assign(nd, 0);
    if (in.mode_req == 0) { r.reason = "PXA_STREAM_WEIGHTS=off"; return r; }

    std::string why;
    bool any_expert_stream = false, any_dense_stream = false;
    for (size_t d = 0; d < nd; ++d) {
        const auto & dev = in.devices[d];
        // the same test the context applies after the pool pre-grow (bug #266): the floor must stay free
        const size_t fit_need = dev.need_resident + dev.pool_need + in.floor_bytes;
        char buf[512];
        if (fit_need <= dev.free_bytes && in.spill_extra == 0) {
            snprintf(buf, sizeof(buf), "dev%d fits resident (%.0f of %.0f MiB)", dev.id, fit_need/(double)MiB, dev.free_bytes/(double)MiB);
            why += (why.empty() ? "" : "; ") + std::string(buf);
            continue;
        }
        // what streaming must free: the deficit with the MMA q8 staging margin (#207) and 256 MiB
        // for lazily loaded kernels / cuBLAS, +5%, plus whatever was asked for on top
        const size_t spec_res = d < in.spec_reserve.size() ? in.spec_reserve[d] : 0;
        const size_t want_free = dev.need_resident + dev.pool_need + 512*MiB + spec_res;
        const size_t deficit = want_free > dev.free_bytes ? want_free - dev.free_bytes : 0;
        const size_t target = deficit + deficit/20 + in.spill_extra;

        // candidate groups on this device, by layer
        std::map<int, std::vector<const pxa_place_tensor *>> exps, down, upgate;
        for (const auto & t : in.tensors) {
            if (t.device != (int)d || t.layer < 0) continue;
            if (t.kind == PXA_PLACE_EXPERTS) exps[t.layer].push_back(&t);
            else if (t.kind == PXA_PLACE_FFN_DOWN) down[t.layer].push_back(&t);
            else if (t.kind == PXA_PLACE_FFN_UPGATE) upgate[t.layer].push_back(&t);
        }
        std::vector<pxa_place_tensor> best;
        bool ok = false;
        auto try_select = [&](const std::map<int, std::vector<const pxa_place_tensor *>> & groups,
                              const std::vector<pxa_place_tensor> & base) -> bool {
            std::vector<int> layers;
            for (auto & kv : groups) layers.push_back(kv.first);
            const int n = (int)layers.size();
            for (int k = 1; k <= n; ++k) {
                std::vector<pxa_place_tensor> sel = base;
                for (int p : spread(k, n)) for (auto * t : groups.at(layers[p])) sel.push_back(*t);
                size_t bytes = 0;
                for (auto & t : sel) bytes += t.bytes;
                const size_t ring = pxa_place_ring_bytes(sel);
                best = sel;
                if (bytes > ring && bytes - ring >= target) return true;
            }
            return false;
        };
        const char * what = "";
        char xc_what[512] = "";
        bool xc_done = false;
        if (!exps.empty() && in.n_expert > 1 && (int)in.xc_counts.size() == in.n_layer) {
            // PXA_XCACHE: expert-granular when every candidate layer has routing counts
            std::vector<int> layers;
            std::vector<size_t> eb, tb;
            std::vector<std::vector<double>> cnts;
            bool have = true;
            for (auto & kv : exps) {
                const int il = kv.first;
                if (il < 0 || il >= in.n_layer || (int)in.xc_counts[il].size() != in.n_expert) { have = false; break; }
                size_t e_all = 0, e_max = 0;
                for (auto * t : kv.second) {
                    if (t->bytes % (size_t)in.n_expert) { have = false; break; }
                    e_all += t->bytes/in.n_expert;
                    e_max  = std::max(e_max, t->bytes/(size_t)in.n_expert);
                }
                if (!have) break;
                layers.push_back(il); eb.push_back(e_all); tb.push_back(e_max); cnts.push_back(in.xc_counts[il]);
            }
            pxa_xc_choice xc;
            pxa_xc_cost cost = pxa_xc_cost_for(dev.cc, in.xc_cold_zc, in.n_expert_used, eb, in.xc_cold_pxqn);
            if (in.xc_miss_scale > 0 && in.xc_miss_scale != 1.0 && !getenv("PXA_XCACHE_COST_MISS_US")) cost.miss_us *= in.xc_miss_scale;
            if (in.xc_plan_v2) {   // PXA_XCACHE_PLAN_V2: measured per-layer cost, and the verify width prices the miss per ROUND
                if (in.xc_cold_pxqn && !in.xc_cold_zc && !getenv("PXA_XCACHE_COST_LAYER_US")) {
                    const char * as = getenv("PXA_XCACHE_ASYNC");
                    cost.layer_us = as && atoi(as) != 0 ? PXA_XC_PXQN_CPU_LAYER_US_V2_ASYNC : PXA_XC_PXQN_CPU_LAYER_US_V2;
                }
                if (in.xc_verify_width > 1.0 && !getenv("PXA_XCACHE_COST_MISS_US")) cost.miss_us *= pxa_xc_width_factor(in.xc_verify_width);
            }
            const size_t xc_target = target + in.xc_extra_compute > in.xc_slack_credit ? target + in.xc_extra_compute - in.xc_slack_credit : 1;
            if (have && pxa_xc_choose(layers, eb, tb, cnts, in.n_expert, xc_target, xc, cost)) {
                r.xc_cost_layer_us = cost.layer_us;
                r.xc_cost_miss_us  = cost.miss_us;
                size_t n_cold = 0;
                for (size_t k = 0; k < layers.size(); ++k) {
                    r.xc_cold[layers[k]] = xc.cold[k];
                    n_cold += xc.cold[k].size();
                }
                r.xc_miss = xc.total > 0 ? xc.miss/xc.total : 0;
                r.xc_cap  = xc.cap;
                r.xc_layers = xc.n_layers;
                r.dev_stream[d] = xc.cold_bytes;
                r.dev_ring[d]   = xc.ring;
                r.streamed     += xc.cold_bytes;
                ok = true; xc_done = true;
                any_expert_stream = true;
                snprintf(xc_what, sizeof(xc_what), "routed experts EXPERT-GRANULAR: %zu of %zu experts cold in %d of %zu layers "
                        "(cap %d/layer, %.2f%% of routings per the counts; +%.0f MiB split-graph compute; cold path %s, "
                        "cost %.0f us/layer + %.0f us/%% miss, sm_%d)",
                        n_cold, layers.size()*(size_t)in.n_expert, xc.n_layers, layers.size(), xc.cap, 100.0*r.xc_miss,
                        in.xc_extra_compute/(double)MiB, in.xc_cold_zc ? "zero-copy" : "cpu", cost.layer_us, cost.miss_us, dev.cc);
                what = xc_what;
            }
        }
        if (xc_done) {
        } else if (!exps.empty()) {
            ok = try_select(exps, {});
            what = "routed experts";
            any_expert_stream = true;
        } else {
            ok = try_select(down, {});
            what = "dense ffn_down";
            if (!ok) {
                std::vector<pxa_place_tensor> all_down;
                for (auto & kv : down) for (auto * t : kv.second) all_down.push_back(*t);
                ok = try_select(upgate, all_down);
                what = "dense FFN (all ffn_down + ffn_up/gate pairs)";
            }
            any_dense_stream = true;
        }
        size_t bytes = r.dev_stream[d];
        if (!xc_done) {
            bytes = 0;
            for (auto & t : best) { bytes += t.bytes; r.names.push_back(t.name); }
            r.dev_stream[d] = bytes;
            r.dev_ring[d] = pxa_place_ring_bytes(best);
            r.streamed += bytes;
        }
        if (!ok) {
            const size_t freed = bytes > r.dev_ring[d] ? bytes - r.dev_ring[d] : 0;
            r.dev_deficit[d] = target > freed ? target - freed : 0;
        }
        snprintf(buf, sizeof(buf), "dev%d short by %.0f MiB (resident need %.0f + pool %.0f + 512 margin%s vs %.0f free): "
                "streams %.0f MiB of %s, ring %.0f MiB%s",
                dev.id, deficit/(double)MiB, dev.need_resident/(double)MiB, dev.pool_need/(double)MiB,
                spec_res ? (" + " + std::to_string((unsigned long long)(spec_res/MiB)) + " speculative chain").c_str() : "", dev.free_bytes/(double)MiB,
                bytes/(double)MiB, what, r.dev_ring[d]/(double)MiB, ok ? "" : " -- STILL SHORT, the load may fail");
        why += (why.empty() ? "" : "; ") + std::string(buf);
    }
    if (r.streamed > 0) {
        r.mode = any_expert_stream && !any_dense_stream ? 2 : 3;
        if (in.pin_ok > 0 && r.streamed > in.pin_ok) {
            r.cpu_instead = true;
            char buf[256];
            snprintf(buf, sizeof(buf), "; pinning %.0f MiB exceeds pin_ok %.0f MiB: the same tensors go to mmap'd host RAM (no ring, CPU compute)",
                    r.streamed/(double)MiB, in.pin_ok/(double)MiB);
            why += buf;
        }
    }
    r.reason = why;
    return r;
}

std::string pxa_place_json(const pxa_place_result & r) {
    std::string s = "{\"mode\":\"";
    s += r.mode == 0 ? "resident" : r.mode == 2 ? "experts" : "spill";
    s += "\",\"cpu_instead\":"; s += r.cpu_instead ? "true" : "false";
    char buf[128];
    snprintf(buf, sizeof(buf), ",\"streamed_mib\":%.1f,\"tensors\":%zu", r.streamed/(double)MiB, r.names.size());
    s += buf;
    s += ",\"devices\":[";
    for (size_t d = 0; d < r.dev_stream.size(); ++d) {
        snprintf(buf, sizeof(buf), "%s{\"stream_mib\":%.1f,\"ring_mib\":%.1f,\"short_mib\":%.1f}", d ? "," : "",
                r.dev_stream[d]/(double)MiB, r.dev_ring[d]/(double)MiB, r.dev_deficit[d]/(double)MiB);
        s += buf;
    }
    s += "]";
    {
        size_t n_layers = 0, n_cold = 0;
        for (auto & v : r.xc_cold) if (!v.empty()) { ++n_layers; n_cold += v.size(); }
        if (n_layers) {
            snprintf(buf, sizeof(buf), ",\"xcache\":{\"layers\":%zu,\"cold_experts\":%zu,\"cap\":%d,\"miss\":%.4f}",
                    n_layers, n_cold, r.xc_cap, r.xc_miss);
            s += buf;
        }
    }
    s += ",\"reason\":\"";
    for (char c : r.reason) { if (c == '"' || c == '\\') s += '\\'; s += c; }
    s += "\"}";
    return s;
}
