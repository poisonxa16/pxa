// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_THREAD_PLACE (default OFF): which CPU each long-lived thread of the Overdrive path runs on.
//
// The cold-expert CPU pool (PXA_XCACHE_CPUPOOL) pins one spinning worker per entry of PXA_XCACHE_CPUPOOL_CPUS and leaves every
// other thread of the process to the scheduler: the host thread that serves the cold-layer doorbells and takes part in the pool,
// the thread that drives the CUDA graphs, the CUDA driver's own helper threads and the HTTP threads. A thread that is not pinned
// lands wherever the scheduler puts it, including on the core of a spinning pool worker; the two then time-slice, and the worker
// (which every cold layer waits for) loses half its speed or pays a wake-up. With the lever on:
//
//   workers  the pool keeps PXA_XCACHE_CPUPOOL_CPUS[0 .. n-2] exactly as the library pins them today (n = pool threads, -t)
//   leader   the host thread that serves the doorbells (and is the n-th participant of every pool job) is pinned to the NEXT
//            entry of the list, the one the library never gives to a worker: PXA_XCACHE_CPUPOOL_CPUS[n-1]
//   decode   the thread that calls llama_decode (the server's task loop) is pinned to a core of the cards' node that is
//            neither a worker's core, nor the leader's, nor a worker's SMT sibling
//   aux      the HTTP listener (and so the HTTP pool it creates) and the CUDA driver's helper threads get a MASK, not a core:
//            every remaining CPU of the node that is not a worker's core or sibling
//
// It never moves a pool worker, never touches the ggml CPU thread pool, and never leaves the CPU set the process was given
// (taskset / cgroup / PXA_NUMA_BIND). Placement moves threads only: no arithmetic, no chunking, no ordering changes; the
// cold-expert result of a job is a function of (expert, row, chunk), never of the core that ran the chunk. Bytes are unchanged by
// construction, so this is a lever-window item with no quality gate.
//
// This header is the pure layer (no syscalls): the plan from a CPU set, a pool list and the SMT siblings, unit-tested in
// tests/test-pxa-thread-place.cpp. The runtime (sysfs, sched_setaffinity, the /proc/self/task sweep) is llama-pxa-thread-place.cpp.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

// "0-15,36-40" -> {0..15,36..40}, in the order written (the pool pins worker i to the i-th entry, so the order matters)
inline std::vector<int> pxa_tp_parse_cpulist(const char * s) {
    std::vector<int> v;
    if (!s) return v;
    const char * p = s;
    while (*p) {
        char * q = nullptr;
        const long a = strtol(p, &q, 10);
        if (q == p) break;
        long b = a;
        if (*q == '-') { p = q + 1; b = strtol(p, &q, 10); }
        for (long c = a; c <= b && c < 4096; ++c) v.push_back((int) c);
        p = (*q == ',') ? q + 1 : q;
    }
    return v;
}

// {0,1,2,3,7} -> "0-3,7" (ascending, merged; for log lines)
inline std::string pxa_tp_fmt_cpulist(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    std::string out;
    for (size_t i = 0; i < v.size();) {
        size_t j = i;
        while (j + 1 < v.size() && v[j + 1] == v[j] + 1) ++j;
        char b[48];
        if (j > i) snprintf(b, sizeof(b), "%s%d-%d", out.empty() ? "" : ",", v[i], v[j]);
        else       snprintf(b, sizeof(b), "%s%d",    out.empty() ? "" : ",", v[i]);
        out += b;
        i = j + 1;
    }
    return out.empty() ? "none" : out;
}

struct pxa_tp_input {
    std::vector<int>                  allowed;      // CPUs the process may run on (sched_getaffinity)
    std::vector<int>                  node_cpus;    // CPUs of the NUMA node the cards (and so the pool) sit on; empty = unknown
    std::vector<int>                  pool;         // PXA_XCACHE_CPUPOOL_CPUS, in order
    int                               pool_threads = 0;   // threads the pool is created with, the leader included; 0 = the list's width
    std::map<int, std::vector<int>>   sibs;         // cpu -> its SMT siblings (itself included); a missing cpu is a core of one thread
};

struct pxa_tp_plan {
    bool             ok = false;       // at least one thing is placed
    std::string      why;              // when nothing is, or what was left out
    std::vector<int> workers;          // the pool's worker CPUs (informational: the library pins them)
    int              leader = -1;      // the host thread that serves the doorbells
    int              decode = -1;      // the thread that calls llama_decode
    std::vector<int> aux;              // the mask of the driver and HTTP threads (empty = leave them)
    std::vector<int> cand;             // the CPUs the plan chose from
};

inline int pxa_tp_core_of(const pxa_tp_input & in, int cpu) {
    const auto it = in.sibs.find(cpu);
    if (it == in.sibs.end() || it->second.empty()) return cpu;
    return *std::min_element(it->second.begin(), it->second.end());
}

// the plan: pure, deterministic. See the file comment for the roles.
inline pxa_tp_plan pxa_tp_make_plan(const pxa_tp_input & in) {
    pxa_tp_plan p;
    if (in.pool.empty()) {
        p.why = "no PXA_XCACHE_CPUPOOL_CPUS: there is no cold-expert pool to keep threads off";
        return p;
    }
    std::set<int> allowed(in.allowed.begin(), in.allowed.end());
    std::vector<int> cand;
    if (!in.node_cpus.empty()) {
        for (int c : in.node_cpus) if (allowed.count(c)) cand.push_back(c);
    }
    if (cand.empty()) cand.assign(allowed.begin(), allowed.end());
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    p.cand = cand;
    const std::set<int> candset(cand.begin(), cand.end());

    // the library: n threads = the caller + n-1 workers, worker i pinned to list[i] (a list shorter than n-1 leaves the rest unpinned)
    const int n = in.pool_threads > 0 ? in.pool_threads : (int) in.pool.size();
    const int nw = std::max(0, n - 1);
    std::set<int> wcores, wcpus;
    for (int i = 0; i < nw && i < (int) in.pool.size(); ++i) {
        p.workers.push_back(in.pool[i]);
        wcpus.insert(in.pool[i]);
        wcores.insert(pxa_tp_core_of(in, in.pool[i]));
    }
    // the leader: the list entry the library never hands to a worker
    int lcore = -1;
    if ((int) in.pool.size() >= n && n >= 1) {
        const int l = in.pool[n - 1];
        if (candset.count(l) && !wcpus.count(l)) { p.leader = l; lcore = pxa_tp_core_of(in, l); }
        else p.why += "leader: list entry " + std::to_string(l) + " is outside the allowed set; ";
    } else {
        p.why += "leader: the pool list (" + std::to_string(in.pool.size()) + " CPUs) has no spare entry for " + std::to_string(n) +
                 " pool threads; ";
    }

    // free CPUs: allowed, not a worker, not the leader, and not on a worker's core (a worker's SMT sibling shares its execution units)
    std::vector<int> freec;
    for (int c : cand) {
        if (c == p.leader || wcpus.count(c)) continue;
        if (wcores.count(pxa_tp_core_of(in, c))) continue;
        freec.push_back(c);
    }
    // the decode core: the highest-numbered free core that is not the leader's; its lowest CPU
    std::map<int, int> core_first;                       // core -> lowest free cpu on it
    for (int c : freec) {
        const int k = pxa_tp_core_of(in, c);
        if (!core_first.count(k)) core_first[k] = c;
    }
    for (auto it = core_first.rbegin(); it != core_first.rend(); ++it) {
        if (it->first == lcore) continue;
        p.decode = it->second;
        break;
    }
    if (p.decode < 0) {
        // nothing but the leader's own core: its SMT sibling still keeps the thread off every worker core
        for (int c : freec) if (pxa_tp_core_of(in, c) == lcore) { p.decode = c; break; }
    }
    if (p.decode < 0) p.why += "decode: every CPU of the node belongs to the pool; the decode thread stays where the scheduler puts it; ";

    // aux: the rest of the free CPUs, the decode core's SMT siblings first, the leader's siblings only when nothing else is left
    const int dcore = p.decode >= 0 ? pxa_tp_core_of(in, p.decode) : -1;
    std::vector<int> a1, a2, a3;
    for (int c : freec) {
        if (c == p.decode) continue;
        const int k = pxa_tp_core_of(in, c);
        if (k == dcore) a1.push_back(c);
        else if (k == lcore) a3.push_back(c);
        else a2.push_back(c);
    }
    p.aux = a1;
    p.aux.insert(p.aux.end(), a2.begin(), a2.end());
    if (p.aux.empty()) p.aux = a3;
    if (p.aux.empty()) p.why += "aux: no CPU left for the driver and HTTP threads; they stay where the scheduler puts them; ";

    p.ok = p.leader >= 0 || p.decode >= 0 || !p.aux.empty();
    if (!p.why.empty() && p.why.size() >= 2) p.why.resize(p.why.size() - 2);   // the trailing "; "
    return p;
}

// ---- the runtime (llama-pxa-thread-place.cpp) -------------------------------------------------------------------------------

// PXA_THREAD_PLACE=1
bool pxa_thread_place_on(void);
// the effective pool width for a create(n_threads_hint) call: PXA_XCACHE_CPUPOOL_THREADS overrides, as the library does
int  pxa_thread_place_pool_threads(int n_threads_hint);
// the cold-path host thread (calling thread) pins itself for a pool of `pool_threads` threads; logs one line
void pxa_thread_place_leader(int pool_threads);

// While alive, the calling thread runs with the CPU set the process had before the lever pinned it, so a thread the scope creates
// (the pool's workers beyond the list) does not inherit the decode core. No-op with the lever off or the thread not pinned.
struct pxa_thread_place_scope {
    pxa_thread_place_scope();
    ~pxa_thread_place_scope();
    pxa_thread_place_scope(const pxa_thread_place_scope &) = delete;
    pxa_thread_place_scope & operator=(const pxa_thread_place_scope &) = delete;
private:
    bool active_ = false;
};
