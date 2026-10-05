// pxa / PXA expert-cache online adaptation -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache.cpp -- see llama-pxa-xcache.h for the design. The policy (what to swap, how many per step) is the closed
// library's (ggml_pxqn_xcache_policy); this file is the mechanism: counter read-back, the two-phase slot copy, the map flips.
#include "llama-pxa-xcache.h"
#include "llama.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-pxqn-api.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-xcache.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef GGML_USE_CUDA

bool llama_pxa_xcache_residency_blocks(void) { return false; }
void llama_pxa_xcache_tick(llama_model &, int) {}
void llama_pxa_xcache_adaptor_free(llama_model &) {}
bool llama_pxa_xcache_stats(const struct llama_model *, uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *) { return false; }

#else

// a residency group (hot swap) is active: its weights are mirrored to host RAM once and restored at unpark, so a cache that rewrites the
// hot stacks and the routing maps while the model runs would come back wrong. The active group can only be read through bind(): put
// back what was taken out (load time and the first tick are single-threaded in the thread that owns the model)
bool llama_pxa_xcache_residency_blocks(void) {
    void * prev = ggml_backend_cuda_residency_bind(nullptr);
    if (prev) ggml_backend_cuda_residency_bind(prev);
    return prev != nullptr;
}

namespace {

constexpr size_t PXA_XC_COPY_CHUNK = 512u << 10;   // a decode's own small transfers never wait behind more than one chunk (~0.16 ms at x4)

int env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

}

// One-time probe of what the CPU can stream out of the pinned cold stack: the cold path's matvec is bound by this (GB/s of weight bytes).
// `threads` readers sum their slice of up to 256 MiB twice; the second pass is timed. Reported, and used for the miss price log only.
static double pxa_xc_probe_cpu_gbs(const void * base, size_t bytes, int threads) {
    const size_t n = std::min<size_t>(bytes, 256u << 20) & ~(size_t) 63;
    if (n < (8u << 20) || threads < 1) return 0.0;
    std::vector<uint64_t> sink((size_t) threads, 0);
    auto pass = [&]() {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) {
            th.emplace_back([&, t]() {
                const uint64_t * p = (const uint64_t *) base + ((n/8)*(size_t) t)/(size_t) threads;
                const uint64_t * e = (const uint64_t *) base + ((n/8)*(size_t) (t + 1))/(size_t) threads;
                uint64_t a = 0, b = 0, c = 0, d = 0;
                for (; p + 4 <= e; p += 4) { a += p[0]; b += p[1]; c += p[2]; d += p[3]; }
                sink[(size_t) t] = a + b + c + d;
            });
        }
        for (auto & x : th) x.join();
    };
    pass();                                        // first pass: page in / warm the TLB
    const int64_t t0 = ggml_time_us();
    pass();
    const double us = (double) (ggml_time_us() - t0);
    uint64_t keep = 0;
    for (uint64_t v : sink) keep ^= v;
    if (keep == 0x9e3779b97f4a7c15ull) fputc(0, stderr);   // the sums are used
    return us > 0 ? (double) n/(us*1e3) : 0.0;
}

struct llama_model::pxa_xc_adaptor {
    struct lay {
        int il = -1, dev = 0, n_expert = 0;
        ggml_tensor * mapT = nullptr;
        std::vector<ggml_tensor *> hotT, coldT;   // parallel: one per expert tensor kind (up / gate / down, or gate_up / down)
        size_t eb = 0;                            // bytes of one expert over all its tensors
        std::vector<int32_t> map;                 // host mirror of the device map: >= 0 hot slot, <= -2 cold slot
        std::vector<uint8_t> busy;                // an in-flight swap involves this expert
        std::vector<int32_t> spare;               // free COLD slots (pinned host RAM)
        size_t stage_off = 0;                     // offset (in u32) of this layer's counter block in the pinned staging buffer
        bool dirty = false;                       // map changed this tick, flush to the device
    };
    // one swap = evict e_out (hot slot s -> a free cold slot), then admit e_in (its cold slot -> the hot slot s that freed up)
    struct op {
        int li = 0, e_in = 0, e_out = 0;
        int s = -1;            // hot slot of e_out, then the slot e_in lands in
        int c_free = -1;       // free cold slot e_out is copied into
        int c_in = -1;         // cold slot e_in is copied from; free again once it is admitted
        int phase = 0;         // 0 evict copy in flight (hot -> cold), 1 admit copy in flight (cold -> hot)
        uint64_t due = 0;      // tick at which the current phase's flip is applied (earliest, in non-deterministic mode)
        void * ev = nullptr;
    };

    llama_model & model;
    const ggml_pxqn_xcache_policy * pol = nullptr;
    void * pst = nullptr;
    bool active = false;
    bool adapt = false;      // false = observe only (hit-rate statistics, no swaps)
    bool broken = false;     // a copy could not be enqueued: no new swaps
    bool det = true;
    int  obs_every = 4;
    int  delay = 2;
    int  log_every = 0;
    int  n_used = 0;
    FILE * trace = nullptr;    // PXA_XCACHE_TRACE=<file>: the cumulative routing counters of every decode (offline replay of policies, tests/replay-xcache-trace.cpp)
    bool   trace_hdr = false;
    std::string hold_flag;     // PXA_XCACHE_ADAPT_FLAG=<file>: while the file exists and starts with '0' no new swap is started (A/B inside one boot)
    int  chaos = 0;            // PXA_XCACHE_ADAPT_CHAOS=n: n random swaps per observation instead of the policy (reproducibility / numerics test)
    uint64_t chaos_rng = 0x9e3779b97f4a7c15ull;
    uint64_t tick = 0;
    uint32_t tokens_since_obs = 0;
    int  ticks_since_obs = 0;
    std::vector<lay> L;
    std::vector<op>  ops;
    std::vector<ggml_pxqn_xc_layer_cfg> lcfg;
    std::vector<ggml_pxqn_xc_obs>  obs;
    std::vector<ggml_pxqn_xc_view> views;
    std::vector<ggml_pxqn_xc_swap> plan_out;
    std::vector<std::vector<double>> prior;
    uint32_t * stage = nullptr;
    size_t stage_u32 = 0;
    std::vector<int> devs;
    double pcie_gbs = 0;
    uint64_t n_obs = 0, n_planned = 0, n_started = 0, n_admitted = 0, n_evicted = 0, n_waits = 0;
    int64_t  us_tick = 0, us_obs = 0, n_ticks = 0;     // host time the adaptation itself costs the decode thread
    std::vector<uint32_t> prev_hit, prev_miss;       // per layer: counters at the last observation
    std::vector<uint64_t> lay_hit, lay_miss;         // per layer: routings served by the hot stack / sent to the cold path since the start
    uint64_t tot_hit = 0, tot_miss = 0;               // routings served by the hot stack / that went to the cold stack since the start
    uint64_t win_hit = 0, win_miss = 0;               // the same since the last log line
    std::mutex mu;

    explicit pxa_xc_adaptor(llama_model & m) : model(m) {}

    ~pxa_xc_adaptor() {
        if (trace) fclose(trace);
        drain();
        if (pst && pol) pol->destroy(pst);
        pst = nullptr;
        if (stage) ggml_backend_cuda_xcache_pinned_free(stage);
    }

    bool init() {
        // PXA_XCACHE_ADAPT: unset / 0 off (static plan) | observe (hit-rate statistics only) | 1 adapt when the library has the
        // policy. Off by default (main, 2026-10-04): adaptation moves experts between the CPU q8 path and the GPU kernel, so a
        // greedy answer depends on the requests before it, and the measured gain (+4% prose / +8% code on the 32 GB Flash-Next,
        // one P100) sits inside this box's load noise. The expert cache's ship rule is "beats the fallback AND stays exact".
        const char * am = getenv("PXA_XCACHE_ADAPT");
        const bool mode_obs = am && strcmp(am, "observe") == 0;
        const bool mode_off = !mode_obs && !(am && strcmp(am, "1") == 0);
        log_every = env_int("PXA_XCACHE_LOG", 0);
        if (mode_off) return false;
        if (llama_pxa_xcache_residency_blocks()) return false;      // hot swap restores the weights from a one-time mirror (see llama.cpp)
        pol = ggml_pxqn_xcache_policy_get();
        adapt = pol && !mode_obs;
        if (!adapt && !(mode_obs || log_every > 0 || getenv("PXA_XCACHE_SUMMARY"))) return false;   // nothing asked for
        det       = env_int("PXA_XCACHE_ADAPT_DET", 1) != 0;
        obs_every = std::max(1, env_int("PXA_XCACHE_ADAPT_EVERY", 4));
        delay     = std::max(1, env_int("PXA_XCACHE_ADAPT_DELAY", 1));
        chaos     = std::max(0, env_int("PXA_XCACHE_ADAPT_CHAOS", 0));
        if (const char * hf = getenv("PXA_XCACHE_ADAPT_FLAG"); hf && *hf) hold_flag = hf;
        if (const char * tf = getenv("PXA_XCACHE_TRACE"); tf && *tf) trace = fopen(tf, "wb");
        n_used    = (int) model.hparams.n_expert_used;
        size_t any_spare = 0;
        for (const auto & mp : model.pxa_xc_maps) {
            lay l;
            l.il = mp.second;
            l.mapT = mp.first;
            l.n_expert = (int) ((mp.first->ne[0] - 8)/2);
            for (const auto & f : model.pxa_xc_fills) {
                if (f.il != l.il) continue;
                l.hotT.push_back(f.hot);
                l.coldT.push_back(f.cold);
                l.dev = f.dev;
                l.eb += f.hot->nb[2];
            }
            if (l.hotT.empty() || l.eb == 0 || l.n_expert <= 1) continue;
            const auto & xl = model.pxa_xc[l.il];
            l.map.assign(l.n_expert, -1);
            l.busy.assign(l.n_expert, 0);
            for (size_t k = 0; k < xl.hot.size(); ++k) if (xl.hot[k] >= 0) l.map[xl.hot[k]] = (int32_t) k;
            for (size_t k = 0; k < xl.cold.size(); ++k) {
                if (xl.cold[k] >= 0) l.map[xl.cold[k]] = -2 - (int32_t) k;
                else l.spare.push_back((int32_t) k);
            }
            any_spare += l.spare.size();
            L.push_back(std::move(l));
        }
        if (L.empty()) return false;
        if (adapt && any_spare == 0) adapt = false;       // the plan left no free cold slot: statistics only
        std::sort(L.begin(), L.end(), [](const lay & a, const lay & b) { return a.il < b.il; });
        for (auto & l : L) {
            l.stage_off = stage_u32;
            stage_u32 += (size_t) l.n_expert + 8;
            if (std::find(devs.begin(), devs.end(), l.dev) == devs.end()) devs.push_back(l.dev);
        }
        stage = (uint32_t *) ggml_backend_cuda_xcache_pinned_alloc(stage_u32*sizeof(uint32_t));
        if (!stage) return false;
        for (int d : devs) {    // what the plan left free once the weights, KV cache and compute buffers exist (slot-pool headroom)
            size_t fr = 0, tot = 0;
            ggml_backend_cuda_get_device_memory(d, &fr, &tot);
            LLAMA_LOG_INFO("PXA_XCACHE: device %d has %.0f MiB of %.0f MiB free after the load, the KV cache and the compute buffers\n", d, fr/1048576.0, tot/1048576.0);
        }
        prev_hit.assign(L.size(), 0);
        prev_miss.assign(L.size(), 0);
        lay_hit.assign(L.size(), 0);
        lay_miss.assign(L.size(), 0);
        obs.resize(L.size());
        views.resize(L.size());
        plan_out.resize(std::max<size_t>(8, 2*L.size()*8));
        if (!pol) {
            LLAMA_LOG_INFO("PXA_XCACHE: routing statistics ON (no adaptation policy in this build): %zu split layers\n", L.size());
            return true;
        }
        // measure the link once: the swap budget per step is a fraction of what the card's link carries
        for (int d : devs) {
            double h2d = 0, d2h = 0;
            ggml_backend_cuda_xcache_probe_link(d, &h2d, &d2h);
            if (h2d > 0 && (pcie_gbs == 0 || h2d < pcie_gbs)) pcie_gbs = h2d;
        }
        // prior: the static counts the plan was made from, as expected selections per token (sum over experts = n_used)
        prior.assign(L.size(), {});
        lcfg.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            auto & l = L[i];
            lcfg[i].n_expert     = l.n_expert;
            lcfg[i].n_slots      = (int32_t) model.pxa_xc[l.il].hot.size();
            lcfg[i].expert_bytes = l.eb;
            lcfg[i].prior        = nullptr;
            if (l.il < (int) model.pxa_xc_prior.size() && (int) model.pxa_xc_prior[l.il].size() == l.n_expert) {
                double tot = 0;
                for (double c : model.pxa_xc_prior[l.il]) tot += c;
                if (tot > 0) {
                    prior[i].resize(l.n_expert);
                    for (int e = 0; e < l.n_expert; ++e) prior[i][e] = model.pxa_xc_prior[l.il][e] * n_used / tot;
                    lcfg[i].prior = prior[i].data();
                }
            }
        }
        ggml_pxqn_xc_cfg cfg = {};
        cfg.n_layers = (int32_t) L.size();
        cfg.layers   = lcfg.data();
        cfg.n_used   = n_used;
        cfg.pcie_gbs = pcie_gbs;
        cfg.cpu_gbs  = 0;
        cfg.flags    = det ? 1u : 0u;
        pst = pol->create(&cfg);
        if (!pst) return false;
        {   // what one routed expert costs when it misses: on the CPU (cold path) vs read over the measured link (the miss split's prices)
            double cpu_us = 0, link_us = 0;
            size_t eb_sum = 0;
            for (const auto & l : L) eb_sum += l.eb;
            double cpu_gbs = 0;
            if (!L.empty() && !L[0].coldT.empty() && L[0].coldT[0]->data) {
                const int nt = env_int("PXA_XCACHE_PROBE_THREADS", 16);
                cpu_gbs = pxa_xc_probe_cpu_gbs(L[0].coldT[0]->data, ggml_nbytes(L[0].coldT[0]), nt);
            }
            // a matvec streams its weights at ~two thirds of what a plain read gets
            const double cpu_use = cpu_gbs > 0 ? 0.65*cpu_gbs : 20.0;
            if (pol->miss_cost_us && pcie_gbs > 0 && pol->miss_cost_us(eb_sum/L.size(), pcie_gbs, cpu_use, &cpu_us, &link_us)) {
                LLAMA_LOG_INFO("PXA_XCACHE: a missed expert (%.2f MiB) costs ~%.0f us on the CPU (cold path, %.1f GB/s streamed from the pinned stack%s) against ~%.0f us over the %.2f GB/s link (zero-copy / DMA): %s\n",
                        eb_sum/(double)L.size()/1048576.0, cpu_us, cpu_use, cpu_gbs > 0 ? ", measured" : ", assumed", link_us, pcie_gbs,
                        link_us < cpu_us ? "the link is the cheaper way to serve a miss" : "the CPU serves misses, the link carries the background swaps");
            }
        }
        LLAMA_LOG_INFO("PXA_XCACHE: online adaptation %s: %zu split layers, %zu free cold slots, re-rank every %d decodes, link %.2f GB/s, %s schedule\n",
                adapt ? "ON" : "observing only", L.size(), any_spare, obs_every, pcie_gbs, det ? "deterministic" : "copy-completion");
        return true;
    }

    // ---- copies -------------------------------------------------------------------------------------------------------------

    // to_hot: cold slot -> hot slot (H2D); else hot slot -> cold slot (D2H)
    bool copy_slot(lay & l, bool to_hot, int hot_slot, int cold_slot) {
        for (size_t k = 0; k < l.hotT.size(); ++k) {
            const size_t nb = l.hotT[k]->nb[2];
            char * hp = (char *) l.hotT[k]->data  + (size_t) hot_slot*nb;
            char * cp = (char *) l.coldT[k]->data + (size_t) cold_slot*nb;
            if (!ggml_backend_cuda_xcache_copy_async(l.dev, to_hot ? hp : cp, to_hot ? cp : hp, nb, PXA_XC_COPY_CHUNK)) return false;
        }
        return true;
    }

    void flush_maps() {
        for (auto & l : L) {
            if (!l.dirty) continue;
            ggml_backend_tensor_set(l.mapT, l.map.data(), 0, (size_t) l.n_expert*sizeof(int32_t));
            l.dirty = false;
        }
    }

    bool wait_ev(op & o) {
        if (!o.ev) return false;
        if (ggml_backend_cuda_xcache_event_done(o.ev) != 1) {
            ggml_backend_cuda_xcache_event_sync(o.ev);
            ++n_waits;
        }
        ggml_backend_cuda_xcache_event_free(o.ev);
        o.ev = nullptr;
        return true;
    }

    void apply_due() {
        std::vector<size_t> to_issue;      // swaps whose evict landed this tick: the admit copy goes out once the new map is on the device
        size_t w = 0;
        for (size_t i = 0; i < ops.size(); ++i) {
            op o = ops[i];
            bool keep = true;
            if (tick >= o.due && (det || (o.ev && ggml_backend_cuda_xcache_event_done(o.ev) == 1))) {
                lay & l = L[o.li];
                auto & xl = model.pxa_xc[l.il];
                if (!wait_ev(o)) {
                    // a copy could not be enqueued (an error): abandon this swap; no new swaps from here on. Phase 0 changed
                    // nothing yet; in phase 1 e_out already lives in its cold slot and the hot slot stays vacant, e_in keeps its
                    // cold slot, so every expert still has exactly one valid home
                    broken = true;
                    if (o.phase == 0) l.spare.push_back(o.c_free);
                    l.busy[o.e_in] = l.busy[o.e_out] = 0;
                    keep = false;
                } else if (o.phase == 0) {
                    // the evicted expert's bytes are in the free cold slot: from now on it is served by the cold path and its hot
                    // slot is vacant
                    l.map[o.e_out] = -2 - o.c_free;
                    l.dirty = true;
                    xl.cold[o.c_free] = o.e_out;
                    xl.hot[o.s] = -1;
                    ++n_evicted;
                    o.phase = 1;
                    o.due = tick + (det ? delay : 1);
                    to_issue.push_back(w);
                } else {
                    // the admitted expert's bytes are in the vacated hot slot: route it there; its cold slot is free again
                    l.map[o.e_in] = o.s;
                    l.dirty = true;
                    xl.hot[o.s] = o.e_in;
                    xl.cold[o.c_in] = -1;
                    l.spare.push_back(o.c_in);
                    l.busy[o.e_in] = l.busy[o.e_out] = 0;
                    ++n_admitted;
                    keep = false;
                }
            }
            if (keep) ops[w++] = o;
        }
        ops.resize(w);
        flush_maps();       // the vacated hot slots are unmapped on the device before anything is written into them
        for (size_t idx : to_issue) {
            op & o = ops[idx];
            lay & l = L[o.li];
            if (copy_slot(l, true, o.s, o.c_in)) o.ev = ggml_backend_cuda_xcache_event_record(l.dev);
            else o.ev = nullptr;     // picked up as a failed copy at its due tick
        }
    }

    // finish everything in flight (shutdown): complete both phases of every swap so the slot table is consistent
    void drain() {
        size_t guard = 0;
        while (!ops.empty() && guard++ < 8) {
            const uint64_t save_tick = tick;
            tick = ~0ull;
            bool d = det; det = true;
            apply_due();
            det = d;
            tick = save_tick;
        }
        for (auto & o : ops) if (o.ev) { ggml_backend_cuda_xcache_event_sync(o.ev); ggml_backend_cuda_xcache_event_free(o.ev); o.ev = nullptr; }
        ops.clear();
    }

    // ---- observation + planning ---------------------------------------------------------------------------------------------

    bool read_counters() {
        for (auto & l : L) {
            const size_t off = (size_t) l.n_expert*sizeof(int32_t);
            if (!ggml_backend_cuda_xcache_copy_async(l.dev, stage + l.stage_off, (char *) l.mapT->data + off, ((size_t) l.n_expert + 8)*sizeof(uint32_t), 0)) return false;
        }
        for (int d : devs) {
            void * ev = ggml_backend_cuda_xcache_event_record(d);
            if (!ev) return false;
            ggml_backend_cuda_xcache_event_sync(ev);
            ggml_backend_cuda_xcache_event_free(ev);
        }
        return true;
    }

    bool held() const {
        if (hold_flag.empty()) return false;
        FILE * f = fopen(hold_flag.c_str(), "r");
        if (!f) return false;
        const int c = fgetc(f);
        fclose(f);
        return c == '0';
    }

    void observe_and_plan() {
        if (!read_counters()) return;
        for (size_t i = 0; i < L.size(); ++i) {
            const uint32_t * b = stage + L[i].stage_off;
            obs[i].counts = b;
            obs[i].hits   = b[L[i].n_expert + 0];
            obs[i].misses = b[L[i].n_expert + 1];
            const uint32_t dh = obs[i].hits - prev_hit[i], dm = obs[i].misses - prev_miss[i];   // u32 wrap-safe
            prev_hit[i] = obs[i].hits; prev_miss[i] = obs[i].misses;
            tot_hit += dh; tot_miss += dm; win_hit += dh; win_miss += dm;
            lay_hit[i] += dh; lay_miss[i] += dm;
        }
        ++n_obs;
        const uint32_t n_tok = tokens_since_obs;
        tokens_since_obs = 0;
        ticks_since_obs = 0;
        if (!pst) return;
        pol->observe(pst, n_tok, obs.data());
        if (!adapt || broken || held()) return;
        size_t eb_sum = 0;
        for (size_t i = 0; i < L.size(); ++i) {
            views[i].map     = L[i].map.data();
            views[i].busy    = L[i].busy.data();
            views[i].n_spare = (int32_t) L[i].spare.size();
            eb_sum += L[i].eb;
        }
        if (chaos > 0 && !held()) {      // not the policy: a fixed pseudo-random schedule, the same on every run
            int32_t n = 0;
            for (int k = 0; k < chaos && n < (int32_t) plan_out.size(); ++k) {
                auto rnd = [&]() { chaos_rng = chaos_rng*6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (chaos_rng >> 33); };
                const int li = (int) (rnd() % L.size());
                const lay & l = L[li];
                if (l.spare.empty()) continue;
                int e_in = -1, e_out = -1;
                for (int tries = 0; tries < 64 && (e_in < 0 || e_out < 0); ++tries) {
                    const int e = (int) (rnd() % (uint32_t) l.n_expert);
                    if (l.busy[e]) continue;
                    if (l.map[e] <= -2 && e_in < 0) e_in = e;
                    else if (l.map[e] >= 0 && e_out < 0) e_out = e;
                }
                if (e_in >= 0 && e_out >= 0) plan_out[n++] = { li, e_in, e_out };
            }
            n_planned += n;
            for (int32_t k = 0; k < n; ++k) start_swap(plan_out[k]);
            return;
        }
        const int32_t n_budget = pol->swap_budget(pst, n_tok);
        if (n_budget <= 0) return;
        const uint64_t budget_bytes = (uint64_t) n_budget * (eb_sum / L.size());
        const int32_t n = pol->plan(pst, views.data(), budget_bytes, plan_out.data(), (int32_t) plan_out.size());
        n_planned += n > 0 ? n : 0;
        for (int32_t k = 0; k < n; ++k) start_swap(plan_out[k]);
    }

    void start_swap(const ggml_pxqn_xc_swap & sw) {
        if (sw.layer < 0 || sw.layer >= (int) L.size()) return;
        lay & l = L[sw.layer];
        if (sw.e_in < 0 || sw.e_in >= l.n_expert || sw.e_out < 0 || sw.e_out >= l.n_expert) return;
        if (l.map[sw.e_in] > -2 || l.map[sw.e_out] < 0 || l.busy[sw.e_in] || l.busy[sw.e_out] || l.spare.empty()) return;
        op o;
        o.li = sw.layer; o.e_in = sw.e_in; o.e_out = sw.e_out;
        o.c_in   = -2 - l.map[sw.e_in];
        o.s      = l.map[sw.e_out];
        o.c_free = l.spare.back();
        if (!copy_slot(l, false, o.s, o.c_free)) return;      // evict first: hot slot -> free cold slot
        o.ev = ggml_backend_cuda_xcache_event_record(l.dev);
        if (!o.ev) return;
        l.spare.pop_back();
        l.busy[o.e_in] = l.busy[o.e_out] = 1;
        o.phase = 0;
        o.due = tick + (det ? delay : 1);
        ops.push_back(o);
        ++n_started;
    }

    // ---- self test (PXA_XCACHE_ADAPT_SELFTEST=1) ---------------------------------------------------------------------------------
    // Moves one hot and one cold expert of a few layers through the real two-phase machinery, checks that every byte of both
    // experts arrived where the map now says it is, then moves them back.
    bool slot_bytes(lay & l, bool hot, int slot, std::vector<std::vector<char>> & out) {
        out.clear();
        for (size_t k = 0; k < l.hotT.size(); ++k) {
            ggml_tensor * t = hot ? l.hotT[k] : l.coldT[k];
            const size_t nb = t->nb[2];
            std::vector<char> b(nb);
            ggml_backend_tensor_get(t, b.data(), (size_t) slot*nb, nb);
            out.push_back(std::move(b));
        }
        return true;
    }

    int selftest() {
        int tested = 0, bad = 0;
        const size_t stride = std::max<size_t>(1, L.size()/4);
        for (size_t li = 0; li < L.size() && tested < 4; li += stride) {
            lay & l = L[li];
            if (l.spare.empty()) continue;
            int e_in = -1, e_out = -1;
            for (int e = 0; e < l.n_expert && (e_in < 0 || e_out < 0); ++e) {
                if (l.map[e] <= -2 && e_in < 0) e_in = e;
                if (l.map[e] >= 0 && e_out < 0) e_out = e;
            }
            if (e_in < 0 || e_out < 0) continue;
            std::vector<std::vector<char>> in_bytes, out_bytes, got;
            slot_bytes(l, false, -2 - l.map[e_in], in_bytes);      // the cold expert, as the cold stack holds it
            slot_bytes(l, true,  l.map[e_out],     out_bytes);     // the hot expert, as the hot stack holds it
            auto run = [&](int a, int b) {
                ggml_pxqn_xc_swap sw = { (int32_t) li, a, b };
                const size_t n0 = ops.size();
                start_swap(sw);
                if (ops.size() == n0) return false;
                for (int t = 0; t < 4 && !ops.empty(); ++t) { ++tick; apply_due(); }
                return ops.empty();
            };
            bool ok = run(e_in, e_out);
            ok = ok && l.map[e_in] >= 0 && l.map[e_out] <= -2;
            bool same_in = false, same_out = false;
            if (ok) {
                slot_bytes(l, true, l.map[e_in], got);  same_in = got == in_bytes;
                slot_bytes(l, false, -2 - l.map[e_out], got); same_out = got == out_bytes;
            }
            ok = ok && same_in && same_out;
            // and back
            bool back = run(e_out, e_in) && l.map[e_out] >= 0 && l.map[e_in] <= -2;
            if (back) {
                slot_bytes(l, true, l.map[e_out], got);  back = got == out_bytes;
                slot_bytes(l, false, -2 - l.map[e_in], got); back = back && got == in_bytes;
            }
            LLAMA_LOG_INFO("PXA_XCACHE adapt selftest: layer %d expert %d in / %d out: %s, back: %s\n", l.il, e_in, e_out, ok ? "ok" : "FAILED", back ? "ok" : "FAILED");
            ++tested;
            if (!ok || !back) ++bad;
        }
        LLAMA_LOG_INFO("PXA_XCACHE adapt selftest: %s (%d layers)\n", tested > 0 && bad == 0 ? "PASS" : "FAIL", tested);
        return bad == 0 && tested > 0;
    }

    // ---- routing trace (PXA_XCACHE_TRACE) ------------------------------------------------------------------------------------
    // header: "XCTR1", n_layers, then per layer {il, n_expert, n_slots_hot, n_free_cold} + the initial map (int32 x n_expert) + {has_prior, float x n_expert};
    // records: {tick, n_tokens} + per layer the cumulative counters (u32 x n_expert, hits, misses)
    void trace_tick(int n_tokens) {
        if (!trace || !read_counters()) return;
        auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, trace); };
        if (!trace_hdr) {
            fwrite("XCTR1", 1, 5, trace);
            w32((uint32_t) L.size());
            for (const auto & l : L) {
                w32((uint32_t) l.il); w32((uint32_t) l.n_expert);
                w32((uint32_t) model.pxa_xc[l.il].hot.size()); w32((uint32_t) l.spare.size());
                fwrite(l.map.data(), 4, (size_t) l.n_expert, trace);
                const size_t li = (size_t) (&l - L.data());
                const bool hp = li < lcfg.size() && lcfg[li].prior != nullptr;
                w32(hp ? 1u : 0u);
                if (hp) for (int e = 0; e < l.n_expert; ++e) { const float v = (float) lcfg[li].prior[e]; fwrite(&v, 4, 1, trace); }
            }
            trace_hdr = true;
        }
        w32((uint32_t) tick); w32((uint32_t) n_tokens);
        for (size_t i = 0; i < L.size(); ++i) {
            const uint32_t * b = stage + L[i].stage_off;
            fwrite(b, 4, (size_t) L[i].n_expert, trace);
            w32(b[L[i].n_expert + 0]); w32(b[L[i].n_expert + 1]);
        }
        fflush(trace);
    }

    // ---- reporting ----------------------------------------------------------------------------------------------------------

    void stats_line(const char * tag) {
        ggml_pxqn_xc_stats st = {};
        if (pol && pst) pol->stats(pst, &st);
        const double all = tot_hit + tot_miss ? 100.0*tot_hit/(double)(tot_hit + tot_miss) : 0.0;
        const double win = win_hit + win_miss ? 100.0*win_hit/(double)(win_hit + win_miss) : 0.0;
        LLAMA_LOG_INFO("PXA_XCACHE%s: tick %llu, expert hit rate %.1f%% since start (%llu routings) / %.1f%% since the last line (%llu); "
                "swaps started %llu admitted %llu evicted %llu, in flight %zu, copy waits %llu, observations %llu, adaptation cost %.0f us per decode (%.0f us per observation)\n",
                tag, (unsigned long long) tick, all, (unsigned long long)(tot_hit + tot_miss), win, (unsigned long long)(win_hit + win_miss),
                (unsigned long long) n_started, (unsigned long long) n_admitted, (unsigned long long) n_evicted, ops.size(),
                (unsigned long long) n_waits, (unsigned long long) n_obs,
                n_ticks ? (double) us_tick/(double) n_ticks : 0.0, n_obs ? (double) us_obs/(double) n_obs : 0.0);
        win_hit = win_miss = 0;
        if (tag && *tag && env_int("PXA_XCACHE_SUMMARY_LAYERS", 1)) {      // the summary also names where the misses are
            std::string ln;
            char b[48];
            for (size_t i = 0; i < L.size(); ++i) {
                const uint64_t t = lay_hit[i] + lay_miss[i];
                snprintf(b, sizeof(b), "%s%d:%.0f", i ? " " : "", L[i].il, t ? 100.0*lay_hit[i]/(double) t : 0.0);
                ln += b;
            }
            LLAMA_LOG_INFO("PXA_XCACHE%s: hit %% per split layer (layer:hit): %s\n", tag, ln.c_str());
        }
    }

    void on_tick(int n_tokens) {
        const int64_t t0 = ggml_time_us();
        ++tick;
        if (trace) trace_tick(n_tokens);
        if (adapt) apply_due();
        tokens_since_obs += (uint32_t) std::max(n_tokens, 0);
        ++ticks_since_obs;
        // a prefill-sized decode is an observation of its own: its routing is the prompt's, which is what the answer will use
        if (ticks_since_obs >= obs_every || n_tokens >= 32) {
            const int64_t t1 = ggml_time_us();
            observe_and_plan();
            us_obs += ggml_time_us() - t1;
            if (log_every > 0 && n_obs % (uint64_t) log_every == 0) stats_line("");
        }
        us_tick += ggml_time_us() - t0;
        ++n_ticks;
    }
};

void llama_pxa_xcache_tick(llama_model & model, int n_tokens) {
    if (model.pxa_xc.empty()) return;
    static std::mutex g_mu;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!model.pxa_xc_ad) {
        auto * a = new llama_model::pxa_xc_adaptor(model);
        a->active = a->init();
        model.pxa_xc_ad = a;
        if (a->active && a->adapt && env_int("PXA_XCACHE_ADAPT_SELFTEST", 0) != 0) a->selftest();
    }
    if (!model.pxa_xc_ad->active) return;
    model.pxa_xc_ad->on_tick(n_tokens);
}

bool llama_pxa_xcache_stats(const struct llama_model * model, uint64_t * hits, uint64_t * misses,
        uint64_t * swaps_started, uint64_t * swaps_admitted, uint64_t * swaps_evicted) {
    if (!model || !model->pxa_xc_ad || !model->pxa_xc_ad->active) return false;
    const auto * a = model->pxa_xc_ad;
    if (hits)           *hits           = a->tot_hit;
    if (misses)         *misses         = a->tot_miss;
    if (swaps_started)  *swaps_started  = a->n_started;
    if (swaps_admitted) *swaps_admitted = a->n_admitted;
    if (swaps_evicted)  *swaps_evicted  = a->n_evicted;
    return true;
}

void llama_pxa_xcache_adaptor_free(llama_model & model) {
    if (!model.pxa_xc_ad) return;
    auto * a = model.pxa_xc_ad;
    model.pxa_xc_ad = nullptr;
    if (a->active) {
        a->drain();
        if (a->n_obs > 0 && (a->log_every > 0 || getenv("PXA_XCACHE_SUMMARY"))) a->stats_line(" summary");
    }
    delete a;
}

#endif
