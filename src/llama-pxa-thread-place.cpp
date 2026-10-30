// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_THREAD_PLACE runtime: see llama-pxa-thread-place.h for the roles and the plan. Everything here is sched_setaffinity on
// threads of this process; no compute, no buffers, no ordering. Default OFF.
#include "llama-pxa-thread-place.h"
#include "llama-impl.h"
#include "llama.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

struct pxa_tp_placed { std::string role; int tid; std::string cpus; };

struct pxa_tp_state {
    std::mutex               mu;
    bool                     inited = false;
    pxa_tp_input             in;
    pxa_tp_plan              plan;
    std::string              node_src;
    int                      node = -1;
    cpu_set_t                orig;                 // the process's CPU set when the plan was made (the pre-lever mask)
    cpu_set_t                aux;
    bool                     have_aux = false;
    std::vector<pxa_tp_placed> placed;
    std::set<int>            reg;                  // tids the lever pinned on purpose: the sweep never touches them
    std::set<int>            decode_tids;
    int                      decode_cpu = -1;      // the core the decode threads were pinned to
    int                      leader_cpu = -1;
    int                      nlink = -1;           // /proc/self/task link count at the last sweep
    int                      n_driver = 0, n_repaired = 0;
    bool                     reported = false;
};

pxa_tp_state & st() { static pxa_tp_state s; return s; }

int gettid_() { return (int) syscall(SYS_gettid); }

bool env_on() {
    static const bool on = [] { const char * e = getenv("PXA_THREAD_PLACE"); return e && *e && atoi(e) != 0; }();
    return on;
}

std::string read_line(const std::string & path) {
    std::string out;
    if (FILE * f = fopen(path.c_str(), "r")) {
        char b[2048];
        if (fgets(b, sizeof(b), f)) out = b;
        fclose(f);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
    return out;
}

std::vector<int> mask_to_vec(const cpu_set_t & m) {
    std::vector<int> v;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &m)) v.push_back(c);
    return v;
}

cpu_set_t vec_to_mask(const std::vector<int> & v) {
    cpu_set_t m; CPU_ZERO(&m);
    for (int c : v) if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &m);
    return m;
}

bool set_self(const cpu_set_t & m) { return sched_setaffinity(0, sizeof(m), &m) == 0; }

// the NUMA node of `cpu` and that node's CPU list, from /sys/devices/system/node
bool node_of(int cpu, int * node, std::vector<int> * cpus) {
    if (DIR * d = opendir("/sys/devices/system/node")) {
        bool found = false;
        while (const dirent * de = readdir(d)) {
            int n = -1;
            if (sscanf(de->d_name, "node%d", &n) != 1) continue;
            const std::vector<int> l = pxa_tp_parse_cpulist(read_line(std::string("/sys/devices/system/node/") + de->d_name + "/cpulist").c_str());
            if (std::find(l.begin(), l.end(), cpu) != l.end()) { *node = n; *cpus = l; found = true; break; }
        }
        closedir(d);
        return found;
    }
    return false;
}

bool node_cpus_of(int node, std::vector<int> * cpus) {
    const std::string s = read_line("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    if (s.empty()) return false;
    *cpus = pxa_tp_parse_cpulist(s.c_str());
    return !cpus->empty();
}

// first call decides (before any thread of the lever is pinned: the callers are the HTTP listener, the task loop or the cold-path host
// thread, all created or running before the decode pin)
void init_locked(pxa_tp_state & s) {
    if (s.inited) return;
    s.inited = true;
    cpu_set_t cur; CPU_ZERO(&cur);
    sched_getaffinity(0, sizeof(cur), &cur);
    s.orig = cur;
    s.in.allowed = mask_to_vec(cur);
    s.in.pool = pxa_tp_parse_cpulist(getenv("PXA_XCACHE_CPUPOOL_CPUS"));
    if (const char * e = getenv("PXA_XCACHE_CPUPOOL_THREADS")) if (atoi(e) > 0) s.in.pool_threads = atoi(e);
    for (int c : s.in.allowed) {
        const std::string sl = read_line("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!sl.empty()) s.in.sibs[c] = pxa_tp_parse_cpulist(sl.c_str());
    }
    for (int c : s.in.pool) {      // a pool CPU outside the allowed set still needs its siblings to rank cores
        if (s.in.sibs.count(c)) continue;
        const std::string sl = read_line("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!sl.empty()) s.in.sibs[c] = pxa_tp_parse_cpulist(sl.c_str());
    }
    // the node: an explicit one, else the node of the first pool CPU (Control / the preset derive the pool from the cards' node)
    int node = -1;
    if (const char * e = getenv("PXA_THREAD_PLACE_NODE")) {
        if (*e && node_cpus_of(atoi(e), &s.in.node_cpus)) { node = atoi(e); s.node_src = "PXA_THREAD_PLACE_NODE"; }
    }
    if (node < 0 && !s.in.pool.empty() && node_of(s.in.pool[0], &node, &s.in.node_cpus)) s.node_src = "the pool list";
    if (node < 0) s.node_src = "unknown (the allowed CPUs as given)";
    s.node = node;
    if (!s.in.pool.empty()) {
        std::set<int> nodes;
        for (int c : s.in.pool) { int n = -1; std::vector<int> l; if (node_of(c, &n, &l)) nodes.insert(n); }
        if (nodes.size() > 1) s.node_src += " (the pool spans " + std::to_string(nodes.size()) + " nodes)";
    }
    s.plan = pxa_tp_make_plan(s.in);
    s.have_aux = !s.plan.aux.empty();
    if (s.have_aux) s.aux = vec_to_mask(s.plan.aux);
}

void note_placed(pxa_tp_state & s, const char * role, int tid, const std::string & cpus) {
    s.placed.push_back({ role, tid, cpus });
    s.reg.insert(tid);
}

// move the threads the lever did not place itself that sit where they should not: the CUDA driver's helper threads (named
// "cuda..."), and any thread that inherited the decode or the leader core from the thread that created it
void sweep_locked(pxa_tp_state & s, bool force) {
    struct stat sb;
    if (stat("/proc/self/task", &sb) != 0) return;
    if (!force && (int) sb.st_nlink == s.nlink) return;
    s.nlink = (int) sb.st_nlink;
    DIR * d = opendir("/proc/self/task");
    if (!d) return;
    while (const dirent * de = readdir(d)) {
        const int tid = atoi(de->d_name);
        if (tid <= 0 || s.reg.count(tid)) continue;
        cpu_set_t m; CPU_ZERO(&m);
        if (sched_getaffinity(tid, sizeof(m), &m) != 0) continue;
        const std::string comm = read_line(std::string("/proc/self/task/") + de->d_name + "/comm");
        if (comm.compare(0, 4, "cuda") == 0) {
            if (s.have_aux && !CPU_EQUAL(&m, &s.aux) && sched_setaffinity(tid, sizeof(s.aux), &s.aux) == 0) ++s.n_driver;
            continue;
        }
        if (CPU_COUNT(&m) == 1) {
            const std::vector<int> v = mask_to_vec(m);
            if (v[0] == s.decode_cpu || v[0] == s.leader_cpu) {
                // pinned to a core the lever reserved but not by the lever: it inherited the mask from its creator
                if (sched_setaffinity(tid, sizeof(s.orig), &s.orig) == 0) ++s.n_repaired;
            }
        }
    }
    closedir(d);
}

}  // namespace

bool pxa_thread_place_on(void) { return env_on(); }

int pxa_thread_place_pool_threads(int n_threads_hint) {
    const char * e = getenv("PXA_XCACHE_CPUPOOL_THREADS");
    int n = (e && atoi(e) > 0) ? atoi(e) : n_threads_hint;
    return n < 1 ? 1 : n;
}

void pxa_thread_place_leader(int pool_threads) {
    if (!env_on()) return;
    pxa_tp_state & s = st();
    std::lock_guard<std::mutex> l(s.mu);
    init_locked(s);
    pxa_tp_input in = s.in;
    in.pool_threads = pool_threads;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    if (p.leader < 0) {
        LLAMA_LOG_INFO("PXA_THREAD_PLACE: cold-path host thread not pinned (%s)\n", p.why.empty() ? "no CPU for it" : p.why.c_str());
        return;
    }
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(p.leader, &m);
    const int tid = gettid_();
    if (!set_self(m)) {
        LLAMA_LOG_INFO("PXA_THREAD_PLACE: cold-path host thread: sched_setaffinity(cpu %d) failed (%s)\n", p.leader, strerror(errno));
        return;
    }
    s.leader_cpu = p.leader;
    note_placed(s, "leader", tid, std::to_string(p.leader));
    LLAMA_LOG_INFO("PXA_THREAD_PLACE: cold-path host thread tid %d -> cpu %d (pool: %d workers on %s)\n", tid, p.leader,
                   pool_threads - 1, pxa_tp_fmt_cpulist(p.workers).c_str());
}

pxa_thread_place_scope::pxa_thread_place_scope() {
    if (!env_on()) return;
    pxa_tp_state & s = st();
    std::lock_guard<std::mutex> l(s.mu);
    if (!s.inited || !s.decode_tids.count(gettid_())) return;
    active_ = set_self(s.orig);
}

pxa_thread_place_scope::~pxa_thread_place_scope() {
    if (!active_) return;
    pxa_tp_state & s = st();
    std::lock_guard<std::mutex> l(s.mu);
    if (s.decode_cpu >= 0) { cpu_set_t m; CPU_ZERO(&m); CPU_SET(s.decode_cpu, &m); set_self(m); }
}

bool llama_pxa_thread_place_on(void) { return env_on(); }

void llama_pxa_thread_place_self(const char * role) {
    if (!env_on() || !role) return;
    pxa_tp_state & s = st();
    std::lock_guard<std::mutex> l(s.mu);
    init_locked(s);
    const int tid = gettid_();
    if (!strcmp(role, "decode")) {
        if (s.plan.decode < 0) return;
        cpu_set_t m; CPU_ZERO(&m); CPU_SET(s.plan.decode, &m);
        if (!set_self(m)) {
            LLAMA_LOG_INFO("PXA_THREAD_PLACE: decode thread: sched_setaffinity(cpu %d) failed (%s)\n", s.plan.decode, strerror(errno));
            return;
        }
        s.decode_cpu = s.plan.decode;
        s.decode_tids.insert(tid);
        note_placed(s, "decode", tid, std::to_string(s.plan.decode));
    } else if (!strcmp(role, "aux")) {
        if (!s.have_aux) return;
        if (!set_self(s.aux)) {
            LLAMA_LOG_INFO("PXA_THREAD_PLACE: %s thread: sched_setaffinity(%s) failed (%s)\n", role, pxa_tp_fmt_cpulist(s.plan.aux).c_str(), strerror(errno));
            return;
        }
        note_placed(s, "http", tid, pxa_tp_fmt_cpulist(s.plan.aux));
    }
}

void llama_pxa_thread_place_tick(void) {
    if (!env_on()) return;
    pxa_tp_state & s = st();
    std::lock_guard<std::mutex> l(s.mu);
    if (!s.inited) return;
    sweep_locked(s, false);
}

void llama_pxa_thread_place_report(void) {
    if (!env_on()) return;
    pxa_tp_state & s = st();
    std::lock_guard<std::mutex> l(s.mu);
    init_locked(s);
    sweep_locked(s, true);
    if (s.reported) return;
    s.reported = true;
    if (!s.plan.ok) {
        LLAMA_LOG_INFO("PXA_THREAD_PLACE=1: nothing placed: %s\n", s.plan.why.c_str());
        return;
    }
    // threads that carry the aux mask: the HTTP listener and the pool it created
    int n_on_aux = 0;
    if (s.have_aux) {
        if (DIR * d = opendir("/proc/self/task")) {
            while (const dirent * de = readdir(d)) {
                const int tid = atoi(de->d_name);
                cpu_set_t m; CPU_ZERO(&m);
                if (tid > 0 && sched_getaffinity(tid, sizeof(m), &m) == 0 && CPU_EQUAL(&m, &s.aux) && !s.reg.count(tid)) ++n_on_aux;
            }
            closedir(d);
        }
    }
    std::string roles;
    for (const auto & p : s.placed) {
        if (!roles.empty()) roles += ", ";
        roles += p.role + " tid " + std::to_string(p.tid) + " -> " + p.cpus;
    }
    const std::string leader_s = s.leader_cpu >= 0 ? "pinned to cpu " + std::to_string(s.leader_cpu)
                               : "pinned when the pool starts (planned cpu " + (s.plan.leader >= 0 ? std::to_string(s.plan.leader) : std::string("none")) + ")";
    LLAMA_LOG_INFO("PXA_THREAD_PLACE=1: node %d from %s | pool workers %s | placed: %s | aux mask %s: %d more threads (HTTP pool), "
                   "%d CUDA driver threads moved, %d inherited masks repaired | cold-path host thread %s | not placed: %s\n",
                   s.node, s.node_src.c_str(), pxa_tp_fmt_cpulist(s.plan.workers).c_str(), roles.empty() ? "nothing yet" : roles.c_str(),
                   pxa_tp_fmt_cpulist(s.plan.aux).c_str(), n_on_aux, s.n_driver, s.n_repaired, leader_s.c_str(),
                   s.plan.why.empty() ? "nothing" : s.plan.why.c_str());
}

#else   // not Linux: the lever is a no-op

bool pxa_thread_place_on(void) { return false; }
int  pxa_thread_place_pool_threads(int n) { return n < 1 ? 1 : n; }
void pxa_thread_place_leader(int) {}
pxa_thread_place_scope::pxa_thread_place_scope() {}
pxa_thread_place_scope::~pxa_thread_place_scope() {}
bool llama_pxa_thread_place_on(void) { return false; }
void llama_pxa_thread_place_self(const char *) {}
void llama_pxa_thread_place_tick(void) {}
void llama_pxa_thread_place_report(void) {}

#endif
