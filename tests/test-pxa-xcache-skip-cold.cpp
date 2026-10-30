// test-pxa-xcache-skip-cold.cpp -- PXA_XCACHE_SKIP_COLD_RANK: the split-ids op that drops the cold tail of a decode / verify window.
//
// ggml_moe_split_ids_set_skip_cold (ggml.h) marks the node that writes the cold ids of a split expert layer: a COLD expert whose best
// 1-based rank over every token of the window is >= R gets cold id -1, so it is not computed and GGML_OP_MOE_MERGE writes 0 for its
// slots (no renormalisation). This test checks, against an independent reference written here, on hundreds of random windows
// (widths 1..40, n_used 4..10, 8..512 experts, 0-100% resident, tied weights, strided ids, map with and without the adaptation block):
//   1. lever off: the cold / hot / both nodes write exactly the plain split (the code path the lever leaves alone)
//   2. lever on : the hot ids (side 0 and plane 0 of the both-node) are untouched; the cold ids are the plain ones except exactly the
//                 experts the reference drops; R > n_used drops nothing
//   3. merge + weighting: a dropped slot merges to exactly 0, the weighted sum equals the reference sum over the kept slots with the
//                 ORIGINAL weights (no renormalisation), and with the lever off it equals the one-stack sum
//   4. (CPU) the same through mul_mat_id over a hot and a cold F32 expert stack (the chain the engine builds)
// Backends: the CPU always; with a CUDA build and a visible device also the CUDA kernels, which must give the same I32 ids bit for bit,
// the same routing counters in the map's adaptation block as the plain kernel, and device skip counters
// (ggml_backend_cuda_xcache_skip_stats) equal to the reference counts. CPU part: no GPU, no model.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-xcache.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

struct win_case {
    int n_exp = 0, n_used = 0, n_tok = 0, R = 0, n_in = 8, n_out = 6, pad = 3;
    bool adapt = false, w2d = false;
    std::vector<int32_t> ids;     // [n_used + pad, n_tok] (the op sees the first n_used rows: a strided view, as ggml_top_k gives)
    std::vector<int32_t> map;     // [n_exp] (+ [n_exp + 8] zeroed adaptation block)
    std::vector<float>   w;       // [n_used, n_tok]
    std::vector<float>   eh, ec;  // synthetic hot / cold expert outputs [n_out, n_used, n_tok]
    std::vector<float>   x;       // [n_in, n_tok]
    std::vector<float>   wh, wc;  // expert stacks [n_in, n_out, n_hot] / [n_in, n_out, n_cold]
    int n_hot = 0, n_cold = 0;
    int32_t id(int k, int t) const { return ids[(size_t) t*(n_used + pad) + k]; }
    float   wt(int k, int t) const { return w[(size_t) t*n_used + k]; }
};

struct win_ref {
    std::vector<int32_t> hot, cold_plain, cold_skip;   // [n_used, n_tok]
    unsigned long long cold_x = 0, skip_x = 0, cold_s = 0, skip_s = 0, hit_windows = 0;
    std::vector<uint32_t> routings;                     // per expert, valid routed slots
    uint32_t hits = 0, misses = 0;
};

static win_ref reference(const win_case & c) {
    win_ref r;
    const int n = c.n_used*c.n_tok;
    r.hot.assign(n, -1); r.cold_plain.assign(n, -1); r.cold_skip.assign(n, -1);
    r.routings.assign(c.n_exp, 0);
    // best rank of every expert over the window, from a full sort of each token's weights (stable: ties keep slot order)
    std::vector<int> best(c.n_exp, 1 << 30);
    for (int t = 0; t < c.n_tok; ++t) {
        std::vector<int> order(c.n_used);
        for (int k = 0; k < c.n_used; ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return c.wt(a, t) > c.wt(b, t); });
        for (int pos = 0; pos < c.n_used; ++pos) {
            const int32_t e = c.id(order[pos], t);
            if (e >= 0 && e < c.n_exp) best[e] = std::min(best[e], pos + 1);
        }
    }
    std::set<int32_t> cold_seen, skip_seen;
    for (int t = 0; t < c.n_tok; ++t) for (int k = 0; k < c.n_used; ++k) {
        const int i = t*c.n_used + k;
        const int32_t e = c.id(k, t);
        if (e < 0 || e >= c.n_exp) continue;
        const int32_t v = c.map[e];
        ++r.routings[e];
        if (v >= 0) { r.hot[i] = v; ++r.hits; } else ++r.misses;
        if (v <= -2) {
            r.cold_plain[i] = -2 - v;
            r.cold_skip[i]  = (c.R > 0 && best[e] >= c.R) ? -1 : -2 - v;
            ++r.cold_s; cold_seen.insert(e);
            if (r.cold_skip[i] < 0) { ++r.skip_s; skip_seen.insert(e); }
        }
    }
    r.cold_x = cold_seen.size(); r.skip_x = skip_seen.size(); r.hit_windows = skip_seen.empty() ? 0 : 1;
    return r;
}

static win_case make_case(std::mt19937 & rng, int n_exp, int n_used, int n_tok, int R, double hot_frac, bool ties, bool adapt, bool w2d) {
    win_case c;
    c.n_exp = n_exp; c.n_used = n_used; c.n_tok = n_tok; c.R = R; c.adapt = adapt; c.w2d = w2d;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    // residency: each expert hot with probability hot_frac; a few map entries -1 (an expert between the two copies of a swap)
    c.map.assign(n_exp, -1);
    for (int e = 0; e < n_exp; ++e) {
        const double u = U(rng);
        if (u < 0.02) continue;
        if (U(rng) < hot_frac) c.map[e] = c.n_hot++; else c.map[e] = -2 - c.n_cold++;
    }
    if (c.n_hot == 0)  { c.map[0] = 0; c.n_hot = 1; }    // stacks need one slot each (the slot numbers above stay valid)
    if (c.n_cold == 0) c.n_cold = 1;
    if (adapt) c.map.resize((size_t) 2*n_exp + 8, 0);
    // ids: n_used distinct experts per token (sometimes a -1 slot, the SER convention), pad rows are garbage the op must not read
    c.ids.assign((size_t) (n_used + c.pad)*n_tok, 0x7f7f7f7f);
    std::vector<int32_t> all(n_exp);
    for (int e = 0; e < n_exp; ++e) all[e] = e;
    for (int t = 0; t < n_tok; ++t) {
        std::shuffle(all.begin(), all.end(), rng);
        for (int k = 0; k < n_used; ++k) c.ids[(size_t) t*(n_used + c.pad) + k] = (U(rng) < 0.03) ? -1 : all[k];
    }
    // weights: positive, unsorted per token; with ties, values from a small set
    c.w.resize((size_t) n_used*n_tok);
    for (auto & v : c.w) v = ties ? (float) (1 + (int) (U(rng)*4))*0.125f : (float) (0.01 + U(rng));
    std::normal_distribution<float> N(0.0f, 1.0f);
    c.eh.resize((size_t) c.n_out*n_used*n_tok); for (auto & v : c.eh) v = N(rng);
    c.ec.resize((size_t) c.n_out*n_used*n_tok); for (auto & v : c.ec) v = N(rng);
    c.x.resize((size_t) c.n_in*n_tok);           for (auto & v : c.x) v = N(rng);
    c.wh.resize((size_t) c.n_in*c.n_out*c.n_hot);  for (auto & v : c.wh) v = N(rng);
    c.wc.resize((size_t) c.n_in*c.n_out*c.n_cold); for (auto & v : c.wc) v = N(rng);
    return c;
}

struct run_out {
    std::vector<int32_t> hot, cold_plain, cold_skip, both_plain, both_skip;
    std::vector<float> merged_skip, out_skip, out_plain, mm_skip, mm_plain;
    std::vector<uint32_t> adapt_block;   // the map's counters after the graph (CUDA counts there; the CPU op does not)
};

// one graph with every node of the case on backend `be`; mmid = also the mul_mat_id chain (CPU)
static bool run_case(ggml_backend_t be, const win_case & c, bool mmid, run_out & o) {
    ggml_init_params ip = { ggml_tensor_overhead()*96 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ids_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used + c.pad, c.n_tok);
    ggml_tensor * map = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) c.map.size());
    ggml_tensor * w = c.w2d ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_used, c.n_tok) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c.n_used, c.n_tok);
    ggml_tensor * eh_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_out, c.n_used, c.n_tok);
    ggml_tensor * eh_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_out, c.n_used, c.n_tok);
    ggml_tensor * ec   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_out, c.n_used, c.n_tok);
    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_in, 1, c.n_tok);
    ggml_tensor * wh = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_in, c.n_out, c.n_hot);
    ggml_tensor * wc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_in, c.n_out, c.n_cold);
    for (ggml_tensor * t : { ids_full, map, w, eh_a, eh_b, ec, x, wh, wc }) ggml_set_input(t);
    ggml_tensor * ids = ggml_view_2d(ctx, ids_full, c.n_used, c.n_tok, ids_full->nb[1], 0);

    ggml_tensor * hot = ggml_moe_split_ids(ctx, ids, map, 0);
    ggml_tensor * cold_plain = ggml_moe_split_ids(ctx, ids, map, 1);
    ggml_tensor * cold_skip  = ggml_moe_split_ids(ctx, ids, map, 1);
    ggml_moe_split_ids_set_skip_cold(cold_skip, w, c.R);
    ggml_tensor * both_plain = ggml_moe_split_ids_both(ctx, ids, map);
    ggml_tensor * both_skip  = ggml_moe_split_ids_both(ctx, ids, map);
    ggml_moe_split_ids_set_skip_cold(both_skip, w, c.R);
    if (c.adapt) for (ggml_tensor * t : { hot, both_plain, both_skip }) ((int32_t *) t->op_params)[1] = c.n_exp;
    ggml_tensor * w3 = c.w2d ? ggml_reshape_3d(ctx, w, 1, c.n_used, c.n_tok) : w;
    // merge + weighting over synthetic expert outputs (in place over a copy of the hot output each)
    auto wsum = [&](ggml_tensor * merged) {
        ggml_tensor * y = ggml_mul(ctx, merged, w3);                                   // [n_out, n_used, n_tok]
        y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));                          // [n_used, n_out, n_tok]
        return ggml_sum_rows(ctx, y);                                                  // [1, n_out, n_tok]
    };
    ggml_tensor * merged_skip = ggml_moe_merge(ctx, eh_a, ec, hot, cold_skip);
    ggml_tensor * out_skip  = wsum(merged_skip);
    ggml_tensor * out_plain = wsum(ggml_moe_merge(ctx, eh_b, ec, hot, cold_plain));
    std::vector<ggml_tensor *> outs = { hot, cold_plain, cold_skip, both_plain, both_skip, merged_skip, out_skip, out_plain };
    ggml_tensor * mm_skip = nullptr, * mm_plain = nullptr;
    if (mmid) {
        ggml_tensor * mh = ggml_mul_mat_id(ctx, wh, x, hot);
        mm_skip  = wsum(ggml_moe_merge(ctx, mh, ggml_mul_mat_id(ctx, wc, x, cold_skip), hot, cold_skip));
        ggml_tensor * mh2 = ggml_mul_mat_id(ctx, wh, x, hot);
        mm_plain = wsum(ggml_moe_merge(ctx, mh2, ggml_mul_mat_id(ctx, wc, x, cold_plain), hot, cold_plain));
        outs.push_back(mm_skip); outs.push_back(mm_plain);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : outs) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
    ggml_set_output(map);
    ggml_set_output(eh_a);   // merged_skip is a view of it: keep the allocator from reusing it after the merge (we read it back)
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(ctx); return false; }
    ggml_backend_tensor_set(ids_full, c.ids.data(), 0, c.ids.size()*4);
    ggml_backend_tensor_set(map, c.map.data(), 0, c.map.size()*4);
    ggml_backend_tensor_set(w, c.w.data(), 0, c.w.size()*4);
    ggml_backend_tensor_set(eh_a, c.eh.data(), 0, c.eh.size()*4);
    ggml_backend_tensor_set(eh_b, c.eh.data(), 0, c.eh.size()*4);
    ggml_backend_tensor_set(ec, c.ec.data(), 0, c.ec.size()*4);
    if (mmid) {
        // bug #341: x / wh / wc feed only the mul_mat_id chain; without it (the CUDA pass) they are not in the graph, the
        // allocator gives them no buffer, and setting them asserted "tensor buffer not set" whenever a device was visible
        ggml_backend_tensor_set(x, c.x.data(), 0, c.x.size()*4);
        ggml_backend_tensor_set(wh, c.wh.data(), 0, c.wh.size()*4);
        ggml_backend_tensor_set(wc, c.wc.data(), 0, c.wc.size()*4);
    }
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    auto geti = [](ggml_tensor * t) { std::vector<int32_t> v(ggml_nelements(t)); ggml_backend_tensor_get(t, v.data(), 0, v.size()*4); return v; };
    auto getf = [](ggml_tensor * t) { std::vector<float> v(ggml_nelements(t)); ggml_backend_tensor_get(t, v.data(), 0, v.size()*4); return v; };
    if (ok) {
        o.hot = geti(hot); o.cold_plain = geti(cold_plain); o.cold_skip = geti(cold_skip);
        o.both_plain = geti(both_plain); o.both_skip = geti(both_skip);
        o.merged_skip = getf(merged_skip); o.out_skip = getf(out_skip); o.out_plain = getf(out_plain);
        if (mmid) { o.mm_skip = getf(mm_skip); o.mm_plain = getf(mm_plain); }
        if (c.adapt) {
            std::vector<int32_t> m = geti(map);
            o.adapt_block.assign((const uint32_t *) m.data() + c.n_exp, (const uint32_t *) m.data() + m.size());
        }
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

static bool close_enough(float a, double b) { return std::fabs((double) a - b) <= 1e-5*std::max(1.0, std::fabs(b)); }

// the weighted sum of the synthetic outputs over the kept slots, original weights
static std::vector<double> ref_wsum(const win_case & c, const std::vector<int32_t> & hot, const std::vector<int32_t> & cold, bool mmid) {
    std::vector<double> y((size_t) c.n_out*c.n_tok, 0.0);
    for (int t = 0; t < c.n_tok; ++t) for (int k = 0; k < c.n_used; ++k) {
        const int i = t*c.n_used + k;
        const bool h = hot[i] >= 0, cl = !h && cold[i] >= 0;
        if (!h && !cl) continue;
        for (int o = 0; o < c.n_out; ++o) {
            double v;
            if (!mmid) v = (h ? c.eh : c.ec)[((size_t) t*c.n_used + k)*c.n_out + o];
            else {
                const std::vector<float> & W = h ? c.wh : c.wc;
                const int slot = h ? hot[i] : cold[i];
                v = 0.0;
                for (int j = 0; j < c.n_in; ++j) v += (double) W[((size_t) slot*c.n_out + o)*c.n_in + j]*c.x[(size_t) t*c.n_in + j];
            }
            y[(size_t) t*c.n_out + o] += (double) c.wt(k, t)*v;
        }
    }
    return y;
}

static void check_case(const char * be_name, const win_case & c, const win_ref & r, const run_out & o, bool mmid, int ci) {
    const int n = c.n_used*c.n_tok;
    char tag[160];
    snprintf(tag, sizeof(tag), "%s case %d (n_exp %d n_used %d n_tok %d R %d%s%s)", be_name, ci, c.n_exp, c.n_used, c.n_tok, c.R,
             c.adapt ? " adapt" : "", c.w2d ? " w2d" : "");
    // 1. the plain nodes are the plain split
    CHECK(o.hot == r.hot, "%s: hot ids differ from the plain split", tag);
    CHECK(o.cold_plain == r.cold_plain, "%s: cold ids (lever off) differ from the plain split", tag);
    CHECK(std::equal(r.hot.begin(), r.hot.end(), o.both_plain.begin()) && std::equal(r.cold_plain.begin(), r.cold_plain.end(), o.both_plain.begin() + n),
          "%s: both-node (lever off) differs from the plain split", tag);
    // 2. the skip nodes: hot plane untouched, cold = the reference
    CHECK(std::equal(r.hot.begin(), r.hot.end(), o.both_skip.begin()), "%s: the skip changed a HOT id", tag);
    CHECK(o.cold_skip == r.cold_skip, "%s: cold ids with the skip differ from the reference", tag);
    CHECK(std::equal(r.cold_skip.begin(), r.cold_skip.end(), o.both_skip.begin() + n), "%s: both-node cold plane with the skip differs", tag);
    if (c.R > c.n_used || c.R <= 0) CHECK(o.cold_skip == r.cold_plain, "%s: R outside 1..n_used dropped an expert", tag);
    // 3. merge: dropped slots are exactly 0; weighted sums are the references (kept slots, original weights)
    for (int i = 0; i < n; ++i) {
        if (r.hot[i] >= 0 || r.cold_skip[i] >= 0) continue;
        bool zero = true;
        for (int q = 0; q < c.n_out; ++q) zero &= o.merged_skip[(size_t) i*c.n_out + q] == 0.0f;
        CHECK(zero, "%s: slot %d (dropped or unrouted) did not merge to 0 (ids hot %d cold %d / op %d %d; merged[0] %g eh %g ec %g)", tag, i,
              r.hot[i], r.cold_skip[i], o.hot[i], o.cold_skip[i], o.merged_skip[(size_t) i*c.n_out], c.eh[(size_t) i*c.n_out], c.ec[(size_t) i*c.n_out]);
    }
    const std::vector<double> ys = ref_wsum(c, r.hot, r.cold_skip, false), yp = ref_wsum(c, r.hot, r.cold_plain, false);
    bool oks = true, okp = true;
    for (size_t i = 0; i < ys.size(); ++i) { oks &= close_enough(o.out_skip[i], ys[i]); okp &= close_enough(o.out_plain[i], yp[i]); }
    CHECK(oks, "%s: weighted sum with the skip != sum over the kept slots with the original weights", tag);
    CHECK(okp, "%s: weighted sum with the lever off != the one-stack sum", tag);
    if (mmid) {
        const std::vector<double> ms = ref_wsum(c, r.hot, r.cold_skip, true), mp = ref_wsum(c, r.hot, r.cold_plain, true);
        bool a = true, b = true;
        for (size_t i = 0; i < ms.size(); ++i) { a &= close_enough(o.mm_skip[i], ms[i]); b &= close_enough(o.mm_plain[i], mp[i]); }
        CHECK(a, "%s: mul_mat_id chain with the skip != reference", tag);
        CHECK(b, "%s: mul_mat_id chain with the lever off != one-stack reference", tag);
    }
}

int main(int argc, char ** argv) {
    int n_cases = 400;
    unsigned seed = 1234;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--cases") && i + 1 < argc) n_cases = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned) atoi(argv[++i]);
    }
    std::mt19937 rng(seed);
    std::vector<win_case> cases;
    const int widths[] = { 1, 2, 3, 4, 5, 6, 8, 9, 16, 26, 40 };
    for (int i = 0; i < n_cases; ++i) {
        const int n_used = i % 7 == 0 ? 4 : (i % 5 == 0 ? 8 : 10);
        const int n_exp  = i % 9 == 0 ? 512 : (i % 4 == 0 ? 128 : (i % 3 == 0 ? 16 : 64));
        const int n_tok  = widths[i % (sizeof(widths)/sizeof(widths[0]))];
        const int Rs[]   = { 8, 9, 10, 1, 2, 5, 7, 11, 12, 0 };
        const int R      = Rs[(i/3) % 10];
        const double hf  = (i % 11 == 0) ? 1.0 : ((i % 13 == 0) ? 0.0 : 0.1 + 0.8*((i*37) % 100)/100.0);
        cases.push_back(make_case(rng, std::max(n_exp, n_used + 1), n_used, n_tok, R, hf, i % 6 == 0, i % 2 == 1, i % 10 == 3));
    }

    // ---- CPU ----------------------------------------------------------------------------------------------------------------------------
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 3);
    std::vector<run_out> cpu_out(cases.size());
    unsigned long long tot_cold = 0, tot_skip = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const win_ref r = reference(cases[i]);
        tot_cold += r.cold_x; tot_skip += r.skip_x;
        const bool ok = run_case(cpu, cases[i], true, cpu_out[i]);
        CHECK(ok, "cpu case %zu: graph failed", i);
        if (ok) check_case("cpu", cases[i], r, cpu_out[i], true, (int) i);
    }
    printf("cpu: %zu windows, %llu distinct cold experts, %llu dropped by the reference; %d checks, %d failures\n",
           cases.size(), tot_cold, tot_skip, g_checks, g_fail);
    CHECK(tot_skip > 0 && tot_skip < tot_cold, "the case mix must drop some cold experts and keep others (dropped %llu of %llu)", tot_skip, tot_cold);
    ggml_backend_free(cpu);

    // ---- CUDA: the kernels against the same reference, the plain kernel's routing counters, the device skip counters --------------------
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        const int fail0 = g_fail, checks0 = g_checks;
        ggml_backend_t cu = ggml_backend_cuda_init(0, nullptr);
        CHECK(cu != nullptr, "cuda: no backend on device 0");
        if (cu) {
            for (size_t i = 0; i < cases.size(); ++i) {
                const win_case & c = cases[i];
                const win_ref r = reference(c);
                unsigned long long s0[GGML_CUDA_SKIP_STAT_N] = {}, s1[GGML_CUDA_SKIP_STAT_N] = {};
                CHECK(ggml_backend_cuda_xcache_skip_stats(0, s0), "cuda: skip stats not readable");
                run_out o;
                const bool ok = run_case(cu, c, false, o);
                CHECK(ok, "cuda case %zu: graph failed", i);
                if (!ok) continue;
                ggml_backend_synchronize(cu);
                CHECK(ggml_backend_cuda_xcache_skip_stats(0, s1), "cuda: skip stats not readable");
                check_case("cuda", c, r, o, false, (int) i);
                CHECK(o.hot == cpu_out[i].hot && o.cold_skip == cpu_out[i].cold_skip && o.both_skip == cpu_out[i].both_skip &&
                      o.both_plain == cpu_out[i].both_plain && o.cold_plain == cpu_out[i].cold_plain, "cuda case %zu: ids differ from the CPU op", i);
                // two nodes carry the skip (the cold node and the both-node) when R > 0
                const unsigned long long m = c.R > 0 ? 2 : 0;
                CHECK(s1[GGML_CUDA_SKIP_STAT_WINDOWS] - s0[GGML_CUDA_SKIP_STAT_WINDOWS] == m &&
                      s1[GGML_CUDA_SKIP_STAT_TOKENS] - s0[GGML_CUDA_SKIP_STAT_TOKENS] == m*c.n_tok &&
                      s1[GGML_CUDA_SKIP_STAT_COLD_EXPERTS] - s0[GGML_CUDA_SKIP_STAT_COLD_EXPERTS] == m*r.cold_x &&
                      s1[GGML_CUDA_SKIP_STAT_SKIP_EXPERTS] - s0[GGML_CUDA_SKIP_STAT_SKIP_EXPERTS] == m*r.skip_x &&
                      s1[GGML_CUDA_SKIP_STAT_COLD_SLOTS] - s0[GGML_CUDA_SKIP_STAT_COLD_SLOTS] == m*r.cold_s &&
                      s1[GGML_CUDA_SKIP_STAT_SKIP_SLOTS] - s0[GGML_CUDA_SKIP_STAT_SKIP_SLOTS] == m*r.skip_s &&
                      s1[GGML_CUDA_SKIP_STAT_WINDOWS_HIT] - s0[GGML_CUDA_SKIP_STAT_WINDOWS_HIT] == m*r.hit_windows,
                      "cuda case %zu: device skip counters (windows %llu cold %llu skip %llu) != reference x%llu (cold %llu skip %llu)", i,
                      s1[0] - s0[0], s1[2] - s0[2], s1[3] - s0[3], m, r.cold_x, r.skip_x);
                if (c.adapt) {
                    // three counting nodes (hot, both_plain, both_skip): the skip kernel counts exactly as the plain kernel does
                    bool cnt_ok = o.adapt_block.size() == (size_t) c.n_exp + 8;
                    for (int e = 0; cnt_ok && e < c.n_exp; ++e) cnt_ok = o.adapt_block[e] == 3*r.routings[e];
                    cnt_ok = cnt_ok && o.adapt_block[c.n_exp + 0] == 3*r.hits && o.adapt_block[c.n_exp + 1] == 3*r.misses &&
                             o.adapt_block[c.n_exp + 2] == 3u*(uint32_t) c.n_tok;
                    CHECK(cnt_ok, "cuda case %zu: routing counters of the adaptation block differ from the plain kernel's", i);
                }
            }
            ggml_backend_free(cu);
        }
        printf("cuda: %zu windows; %d checks, %d failures\n", cases.size(), g_checks - checks0, g_fail - fail0);
    } else {
        printf("cuda: no device visible, kernel comparison skipped\n");
    }
#endif
    printf("%s: %d checks, %d failures\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
