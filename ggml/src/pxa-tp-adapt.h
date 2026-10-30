#pragma once
// PXA_TSPLIT_WORKERS_ADAPT state machine and the default worker cpu pick (2026-10-21, cells C031/C125/C126).
// Pure logic, no ggml state: ggml-backend.cpp drives it once per graph, tests/test-pxa-tp-adapt.cpp drives it from a table.
//
// WHY. The persistent per-device enqueue workers (PXA_TSPLIT_WORKERS, auto on a 4x sm_60 tensor split) win ~10% on a quiet
// box, but every token crosses ~144 host rendezvous; when a worker shares its core with a busy co-tenant each rendezvous waits
// out a scheduler slice and the token collapses (measured 38.8 -> 2.5-4.4 t/s, 2x P100). The plain loop (one thread enqueues
// every device) holds up under the same load. Both put the same launches on every stream in the same order, so the choice can
// be made per graph, by measurement, without changing a bit of the output.
//
// HOW. A sample = the period between the entry of two consecutive decode-width graphs with the same split count, filed under
// the mode the earlier graph ran. The preferred mode runs; every `interval` graphs (16..512) the other mode runs for a probe of
// up to 3 graphs; it takes over when its median is >= 3% better. A losing probe doubles the interval (x4 / x8 when it lost by
// more than 1.5x / 3x: a collapsed mode costs a whole slow graph per probe graph; and at least 100 x the probe graph's cost
// in preferred graphs, <= 8192, so a collapsed mode's probes stay near 1% of the run). Periods up to 10 s are samples (a
// burned-core worker graph takes ~1 s), a probe graph's at any length. Drift (the preferred median 25% above the
// value it was chosen at, or its last two periods twice that value) re-probes at once, against the preferred mode's newest
// two periods only. The first period after a mode change
// carries the change itself and is dropped, unless a probe's first period is already twice the preferred median.
// Graphs wider than 4 tokens are not sampled; they run the workers until a decode probe has decided, then the preferred mode.
#include <algorithm>
#include <cstdint>
#include <vector>

struct pxa_tp_adapt {
    int64_t last_us     = 0;    // entry time of the previous graph, 0 = none
    int     last_mode   = -1;   // 1 = the previous graph ran on the workers, 0 = on the plain loop
    int     last_splits = -1;   // its split count (a period is only a sample between two same-shape graphs)
    int     pref        = 0;    // the mode decode graphs run (starts on the plain loop: a probe of a collapsed worker mode
                                // costs one graph, a few plain graphs on a quiet box cost ~nothing)
    int     since       = 0;    // graphs run in the preferred mode since the last probe
    int     interval    = 16;   // graphs between probes
    int     probe_left  = 0;    // graphs left in the running probe of the other mode
    int     skip        = 0;    // samples to drop after a mode change
    int     switches    = 0;
    int     decisions   = 0;    // finished probes
    long    graphs[2]   = {0, 0};
    float   ring[2][8]  = {{0}}; // the last 8 periods per mode, us
    int     nring[2]    = {0, 0};
    int     pos[2]      = {0, 0};
    float   ref         = 0;    // the preferred mode's median when it was last chosen
};

struct pxa_tp_adapt_decision {
    bool  decided = false;      // a probe finished in this step
    int   probed  = -1;         // the mode it probed
    int   kept    = -1;         // the mode preferred before it
    float m_probe = 0;          // its median, us
    float m_kept  = 0;          // the preferred median, us
};

static inline float pxa_tp_adapt_median(const pxa_tp_adapt & a, int m) {
    const int n = a.nring[m];
    if (n <= 0) { return 0.0f; }
    float v[8];
    for (int i = 0; i < n; ++i) { v[i] = a.ring[m][i]; }
    std::sort(v, v + n);
    return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static inline void pxa_tp_adapt_push(pxa_tp_adapt & a, int m, float us) {
    a.ring[m][a.pos[m]] = us;
    a.pos[m] = (a.pos[m] + 1) & 7;
    if (a.nring[m] < 8) { a.nring[m]++; }
}

static inline float pxa_tp_adapt_last(const pxa_tp_adapt & a, int m, int back) {   // back = 1: the newest sample
    return a.ring[m][(a.pos[m] + 8 - back) & 7];
}

// One graph. now_us = its entry time, n_splits = its split count, ntok = the ubatch width llama stated (0 = not told = small).
// Returns 1 = this graph runs on the workers, 0 = on the plain loop.
static inline int pxa_tp_adapt_step(pxa_tp_adapt & a, int64_t now_us, int n_splits, int ntok, pxa_tp_adapt_decision * dec) {
    if (dec) { *dec = pxa_tp_adapt_decision(); }
    if (ntok > 4) {
        // a prompt ubatch / a wide multi-slot or verify batch: not sampled, breaks the period chain; the workers (the
        // pre-adapt behaviour) until a decode probe has measured this box, then the preferred mode
        a.last_us   = 0;
        a.last_mode = -1;
        return a.decisions == 0 ? 1 : a.pref;
    }
    bool sampled = false;
    if (a.last_us != 0 && a.last_mode >= 0 && a.last_splits == n_splits) {
        const int64_t dt = now_us - a.last_us;
        // A collapsed worker graph takes 0.7-1.0 s on a burned core (P4 quick 10-21); the old 1 s cut dropped those
        // periods, so a probe ran all its graphs at the mildest backoff and a collapse of the preferred workers was
        // never seen. Periods up to 10 s are samples now (a new request starts with a wide prompt graph, which breaks
        // the chain, so an idle gap rarely lands here, and one outlier moves neither a median of 8 nor the two-period
        // drift test); a probe graph's period is a sample at any length (a gap after a probe costs only that probe).
        const bool probe_graph = a.probe_left > 0 && a.last_mode != a.pref;
        if (dt > 0 && (dt < 10000000 || probe_graph)) {
            const float mp0 = pxa_tp_adapt_median(a, a.pref);
            const bool  bad = probe_graph && mp0 > 0 && (float) dt > 2.0f * mp0;
            if (a.skip > 0 && !bad) {
                a.skip--;
            } else {
                a.skip = 0;
                pxa_tp_adapt_push(a, a.last_mode, (float) dt);
                sampled = true;
            }
        }
    }
    const int pref  = a.pref;
    const int other = 1 - pref;
    if (a.probe_left > 0) {
        // a probe of `other` is running: its sample (if any) just landed in ring[other]
        bool done = --a.probe_left == 0;
        int  back = 2;
        const float mp = pxa_tp_adapt_median(a, pref);
        if (sampled && a.last_mode == other && mp > 0 && pxa_tp_adapt_last(a, other, 1) > 1.5f * mp) {
            done = true;                                  // already clearly slower: stop paying for it
            back = pxa_tp_adapt_last(a, other, 1) > 3.0f * mp ? 8 : 4;
        }
        if (done) {
            const float mo  = pxa_tp_adapt_median(a, other);
            const bool  win = mo > 0 && mp > 0 && mo < 0.97f * mp;
            if (win) {
                a.pref     = other;
                a.interval = 16;
                a.ref      = mo;
                a.switches++;
            } else {
                a.interval = std::min(a.interval * back, 512);
                // time-based floor: a probe graph that cost a whole slow graph (a collapsed mode) may take at most ~1% of
                // the preferred mode's time, so the next probe waits >= 100 x its cost in preferred graphs (<= 8192)
                const float worst = a.nring[other] > 0 ? std::max(pxa_tp_adapt_last(a, other, 1), mo) : 0.0f;
                if (mp > 0 && worst > 1.5f * mp) {
                    a.interval = std::max(a.interval, std::min(8192, (int) (100.0f * worst / mp)));
                }
                a.ref      = mp;
            }
            a.decisions++;
            if (dec) { dec->decided = true; dec->probed = other; dec->kept = pref; dec->m_probe = mo; dec->m_kept = mp; }
            a.probe_left = 0;
            a.since      = 0;
        }
    } else {
        a.since++;
        const float mp    = pxa_tp_adapt_median(a, pref);
        const float last1 = pxa_tp_adapt_last(a, pref, 1);
        const float last2 = pxa_tp_adapt_last(a, pref, 2);
        const bool  drift = a.ref > 0 && ((a.nring[pref] >= 4 && mp > 1.25f * a.ref) ||
                                          (sampled && a.nring[pref] >= 2 && last1 > 2.0f * a.ref && last2 > 2.0f * a.ref));
        if (a.since >= a.interval || drift || (a.nring[other] == 0 && a.nring[pref] >= 2)) {
            if (drift) {
                // judge the probe against what the preferred mode does NOW, not against a ring still full of the old level
                a.interval     = 16;
                a.nring[pref]  = 0;
                a.pos[pref]    = 0;
                pxa_tp_adapt_push(a, pref, last2);
                pxa_tp_adapt_push(a, pref, last1);
            }
            a.probe_left   = 3;
            a.nring[other] = 0;
            a.pos[other]   = 0;
        }
    }
    const int mode = a.probe_left > 0 ? 1 - a.pref : a.pref;
    if (mode != a.last_mode) { a.skip = 1; }
    a.last_us     = now_us;
    a.last_mode   = mode;
    a.last_splits = n_splits;
    a.graphs[mode]++;
    return mode;
}

// The default worker cpus (PXA_TSPLIT_WORKER_PIN=1): count down from the top of the driver node's cpus that the PROCESS may
// run on, pool `pool_index` taking the next n_workers below the previous pool's.
//   node_cpus   the driver's node, ascending;
//   allowed     the process mask's cpus, ascending (empty = unknown: the whole node);
//   driver_only the driver thread's cpu when it is pinned to exactly one cpu, else -1: left out of the pick while the
//               allowed node cpus can still give every worker a cpu of its own without it.
static inline std::vector<int> pxa_tp_pick_cpus(const std::vector<int> & node_cpus, const std::vector<int> & allowed,
                                                int driver_only, int n_workers, int pool_index) {
    std::vector<int> cand;
    for (int c : node_cpus) {
        if (allowed.empty() || std::binary_search(allowed.begin(), allowed.end(), c)) { cand.push_back(c); }
    }
    if (cand.empty()) { cand = node_cpus; }   // a mask that misses the node entirely: the node top, as before
    if (driver_only >= 0 && (int) cand.size() > (pool_index + 1) * n_workers) {
        cand.erase(std::remove(cand.begin(), cand.end(), driver_only), cand.end());
    }
    std::vector<int> v;
    const int skip = pool_index * n_workers;
    for (int i = (int) cand.size() - 1 - skip; i >= 0 && (int) v.size() < n_workers; --i) {
        v.push_back(cand[i]);
    }
    return v;
}
