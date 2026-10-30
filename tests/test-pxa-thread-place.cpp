// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_THREAD_PLACE: (1) the pure plan (src/llama-pxa-thread-place.h) on the box layouts the lever is measured on and on random CPU
// sets, (2) the runtime on this machine: a thread that asks for a role lands where the plan says (read back with
// sched_getaffinity), a thread that inherited the decode core is repaired by the sweep, a thread of a pool the lever does not own is
// left alone. No model, no GPU.

#include "llama-pxa-thread-place.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static std::vector<int> range(int a, int b) { std::vector<int> v; for (int c = a; c <= b; ++c) v.push_back(c); return v; }
static std::vector<int> cat(std::vector<int> a, const std::vector<int> & b) { a.insert(a.end(), b.begin(), b.end()); return a; }

// the 2-socket box the lever is measured on: node0 = 0-17,36-53, node1 = 18-35,54-71; cpu c and c+36 are the two threads of a core
static void box72(pxa_tp_input & in) {
    for (int c = 0; c < 72; ++c) in.sibs[c] = { c % 36, c % 36 + 36 };
}
static std::vector<int> node0() { return cat(range(0, 17), range(36, 53)); }
static std::vector<int> node1() { return cat(range(18, 35), range(54, 71)); }

static std::string str(const std::vector<int> & v) { return pxa_tp_fmt_cpulist(v); }

static void test_parse_fmt() {
    CHECK(pxa_tp_parse_cpulist("0-3,7,9-10") == (std::vector<int>{0,1,2,3,7,9,10}), "parse");
    CHECK(pxa_tp_parse_cpulist("") .empty() && pxa_tp_parse_cpulist(nullptr).empty(), "empty");
    CHECK(pxa_tp_parse_cpulist("5,3,4") == (std::vector<int>{5,3,4}), "order kept");
    CHECK(str({0,1,2,3,7}) == "0-3,7", "fmt %s", str({0,1,2,3,7}).c_str());
    CHECK(str({}) == "none", "fmt empty");
}

// the shipped 1-card preset on node 0: pool 0-15, -t 16
static void test_overdrive_node0() {
    pxa_tp_input in; box72(in);
    in.allowed = node0(); in.node_cpus = node0(); in.pool = range(0, 15); in.pool_threads = 16;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.ok, "ok");
    CHECK(p.workers == range(0, 14), "workers %s", str(p.workers).c_str());
    CHECK(p.leader == 15, "leader %d", p.leader);
    CHECK(p.decode == 17, "decode %d", p.decode);
    CHECK(str(p.aux) == "16,52-53", "aux %s", str(p.aux).c_str());
    CHECK(p.why.empty(), "why '%s'", p.why.c_str());
}

// the launcher's pool derivation gives the same list as the preset when the width is 16; the list is the input, the width defaults to it
static void test_width_defaults_to_list() {
    pxa_tp_input in; box72(in);
    in.allowed = node0(); in.node_cpus = node0(); in.pool = range(0, 15);       // pool_threads = 0 -> 16
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.leader == 15 && p.decode == 17, "leader %d decode %d", p.leader, p.decode);
}

// a launcher run under taskset 54-71 (this box's build CPUs): the pool is whatever the list says inside the allowed set
static void test_node1_cpuset() {
    pxa_tp_input in; box72(in);
    in.allowed = range(54, 71); in.node_cpus = node1(); in.pool = range(54, 61); in.pool_threads = 8;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.workers == range(54, 60), "workers %s", str(p.workers).c_str());
    CHECK(p.leader == 61, "leader %d", p.leader);
    CHECK(p.decode == 71, "decode %d", p.decode);
    CHECK(str(p.aux) == "62-70", "aux %s", str(p.aux).c_str());
}

// the CPU set is only the pool: nothing is left for decode and aux; the leader still gets its entry
static void test_no_spare() {
    pxa_tp_input in; box72(in);
    in.allowed = range(0, 15); in.node_cpus = node0(); in.pool = range(0, 15); in.pool_threads = 16;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.ok && p.leader == 15, "leader %d", p.leader);
    CHECK(p.decode == -1 && p.aux.empty(), "decode %d aux %s", p.decode, str(p.aux).c_str());
    CHECK(p.why.find("decode") != std::string::npos && p.why.find("aux") != std::string::npos, "why '%s'", p.why.c_str());
}

// a pool wider than its list: every list CPU is a worker, there is no leader entry
static void test_list_too_short() {
    pxa_tp_input in; box72(in);
    in.allowed = node0(); in.node_cpus = node0(); in.pool = range(0, 7); in.pool_threads = 16;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.leader == -1, "leader %d", p.leader);
    CHECK(p.workers == range(0, 7), "workers %s", str(p.workers).c_str());
    CHECK(p.decode == 17, "decode %d", p.decode);
    CHECK(p.why.find("no spare entry") != std::string::npos, "why '%s'", p.why.c_str());
}

// a pool narrower than its list: -t 8 on 0-15: the leader is list[7], list[8..15] are free cores again
static void test_narrow_pool() {
    pxa_tp_input in; box72(in);
    in.allowed = node0(); in.node_cpus = node0(); in.pool = range(0, 15); in.pool_threads = 8;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.workers == range(0, 6) && p.leader == 7 && p.decode == 17, "workers %s leader %d decode %d", str(p.workers).c_str(), p.leader, p.decode);
    const std::set<int> aux(p.aux.begin(), p.aux.end());
    CHECK(aux.count(8) && aux.count(15) && aux.count(16) && aux.count(53), "aux %s", str(p.aux).c_str());
    CHECK(!aux.count(7) && !aux.count(43), "the leader and its sibling are not aux: %s", str(p.aux).c_str());
}

static void test_no_pool_no_smt() {
    pxa_tp_input in;
    in.allowed = range(0, 7);
    CHECK(!pxa_tp_make_plan(in).ok, "no pool: nothing placed");
    in.pool = range(0, 3); in.pool_threads = 4;           // no SMT info: every CPU is a core
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.leader == 3 && p.decode == 7 && str(p.aux) == "4-6", "leader %d decode %d aux %s", p.leader, p.decode, str(p.aux).c_str());
}

// the node filter: allowed spans both sockets, the pool is on node 0: nothing lands on node 1
static void test_node_filter() {
    pxa_tp_input in; box72(in);
    in.allowed = range(0, 71); in.node_cpus = node0(); in.pool = range(0, 15); in.pool_threads = 16;
    const pxa_tp_plan p = pxa_tp_make_plan(in);
    CHECK(p.decode == 17, "decode %d", p.decode);
    for (int c : p.aux) CHECK(c <= 17 || (c >= 36 && c <= 53), "aux cpu %d off the node", c);
}

// properties over random layouts: the roles never overlap, never sit on a worker's core, never leave the allowed set
static void test_random() {
    unsigned long long x = 88172645463325252ull;
    auto rnd = [&](int n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (int) (x % (unsigned long long) n); };
    for (int it = 0; it < 4000; ++it) {
        pxa_tp_input in;
        const int ncore = 4 + rnd(40), smt = 1 + rnd(2);
        const int ncpu = ncore * smt;
        for (int c = 0; c < ncpu; ++c) {
            if (smt == 2) in.sibs[c] = { c % ncore, c % ncore + ncore };
        }
        std::vector<int> all = range(0, ncpu - 1);
        for (int c : all) if (rnd(4) != 0) in.allowed.push_back(c);
        if (rnd(2)) in.node_cpus = in.allowed;
        const int pw = 1 + rnd(20);
        for (int i = 0; i < pw; ++i) in.pool.push_back(rnd(ncpu));
        std::sort(in.pool.begin(), in.pool.end()); in.pool.erase(std::unique(in.pool.begin(), in.pool.end()), in.pool.end());
        in.pool_threads = rnd(3) ? 0 : 1 + rnd(24);
        const pxa_tp_plan p = pxa_tp_make_plan(in);
        const std::set<int> allowed(in.allowed.begin(), in.allowed.end());
        std::set<int> wcores;
        for (int w : p.workers) wcores.insert(pxa_tp_core_of(in, w));
        std::set<int> seen;
        auto use = [&](int c, const char * role) {
            if (c < 0) return;
            CHECK(allowed.count(c), "%s cpu %d outside the allowed set (it %d)", role, c, it);
            // the leader is the list's own entry, whatever its SMT sibling is; decode and aux never share a worker's core
            if (strcmp(role, "leader")) CHECK(!wcores.count(pxa_tp_core_of(in, c)), "%s cpu %d on a worker's core (it %d)", role, c, it);
            else CHECK(std::find(p.workers.begin(), p.workers.end(), c) == p.workers.end(), "leader cpu %d is a worker (it %d)", c, it);
            CHECK(seen.insert(c).second, "%s cpu %d used twice (it %d)", role, c, it);
        };
        use(p.leader, "leader"); use(p.decode, "decode");
        for (int c : p.aux) use(c, "aux");
        if (p.leader >= 0 && p.decode >= 0) CHECK(p.leader != p.decode, "leader == decode");
    }
}

#if defined(__linux__)
static std::vector<int> self_mask(pid_t tid = 0) {
    cpu_set_t m; CPU_ZERO(&m);
    sched_getaffinity(tid, sizeof(m), &m);
    std::vector<int> v;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &m)) v.push_back(c);
    return v;
}

// the runtime on this machine. The pool list is the first CPUs of whatever set the test runs in; the checks read the masks back.
static void test_runtime() {
    const std::vector<int> allowed = self_mask();
    if (allowed.size() < 6) { fprintf(stderr, "SKIP runtime: %zu CPUs allowed (<6)\n", allowed.size()); return; }
    const int k = (int) std::min<size_t>(8, allowed.size() - 2);
    const std::vector<int> pool(allowed.begin(), allowed.begin() + k);
    setenv("PXA_THREAD_PLACE", "1", 1);
    setenv("PXA_XCACHE_CPUPOOL_CPUS", pxa_tp_fmt_cpulist(pool).c_str(), 1);
    setenv("PXA_XCACHE_CPUPOOL_THREADS", std::to_string(k).c_str(), 1);
    CHECK(pxa_thread_place_on() && llama_pxa_thread_place_on(), "lever on");

    // what the plan says for this machine's SMT layout is what the runtime must do; recompute it here from sysfs
    pxa_tp_input in; in.allowed = allowed; in.pool = pool; in.pool_threads = k;
    for (int c : allowed) {
        char path[128]; snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        if (FILE * f = fopen(path, "r")) { char b[256]; if (fgets(b, sizeof(b), f)) in.sibs[c] = pxa_tp_parse_cpulist(b); fclose(f); }
    }
    {   // the node, the way the runtime finds it: node of pool[0]
        char path[128];
        for (int n = 0; n < 16; ++n) {
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
            if (FILE * f = fopen(path, "r")) {
                char b[1024]; std::vector<int> l;
                if (fgets(b, sizeof(b), f)) l = pxa_tp_parse_cpulist(b);
                fclose(f);
                if (std::find(l.begin(), l.end(), pool[0]) != l.end()) { in.node_cpus = l; break; }
            }
        }
    }
    const pxa_tp_plan plan = pxa_tp_make_plan(in);
    fprintf(stderr, "runtime plan: pool %s workers %s leader %d decode %d aux %s\n", str(pool).c_str(), str(plan.workers).c_str(),
            plan.leader, plan.decode, str(plan.aux).c_str());
    if (plan.decode < 0 || plan.aux.empty()) { fprintf(stderr, "SKIP runtime: no spare CPU on this machine's cpuset\n"); return; }

    // aux role: an HTTP-like thread gets the mask; a thread it spawns inherits it
    std::vector<int> aux_mask, child_mask;
    std::thread t([&] {
        llama_pxa_thread_place_self("aux");
        aux_mask = self_mask();
        std::thread c([&] { child_mask = self_mask(); });
        c.join();
    });
    t.join();
    CHECK(aux_mask == plan.aux || str(aux_mask) == str(plan.aux), "aux mask %s want %s", str(aux_mask).c_str(), str(plan.aux).c_str());
    CHECK(child_mask == aux_mask, "the HTTP pool inherits the aux mask: %s", str(child_mask).c_str());

    // leader role: a cold-path host thread pins to list[n-1] when the list has the entry
    {
        std::vector<int> lm;
        std::thread l([&] { pxa_thread_place_leader(k); lm = self_mask(); });
        l.join();
        if (plan.leader >= 0) CHECK(lm == std::vector<int>{ plan.leader }, "leader mask %s want %d", str(lm).c_str(), plan.leader);
        else CHECK(lm == allowed, "no leader entry: the thread stays put");
    }

    // decode role on a worker thread of the test (the real one is the server's task loop); the sweep repairs a thread that inherited it
    std::vector<int> dm, inherited_before, inherited_after;
    std::thread d([&] {
        llama_pxa_thread_place_self("decode");
        dm = self_mask();
        pid_t child_tid = 0;
        std::atomic<bool> go{false}, done{false};
        std::thread c([&] {
            child_tid = (pid_t) syscall(SYS_gettid);
            inherited_before = self_mask();                          // inherited the decode core from the creator
            go = true;
            while (!done) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
        while (!go) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        llama_pxa_thread_place_report();                             // the boot sweep
        inherited_after = self_mask(child_tid);
        done = true;
        c.join();
    });
    d.join();
    CHECK(dm == std::vector<int>{ plan.decode }, "decode mask %s want %d", str(dm).c_str(), plan.decode);
    CHECK(inherited_before == dm, "a thread created by the pinned decode thread starts on its core: %s", str(inherited_before).c_str());
    CHECK(inherited_after == allowed, "the sweep gives it the process set back: %s want %s", str(inherited_after).c_str(), str(allowed).c_str());
}
#endif

int main() {
    test_parse_fmt();
    test_overdrive_node0();
    test_width_defaults_to_list();
    test_node1_cpuset();
    test_no_spare();
    test_list_too_short();
    test_narrow_pool();
    test_no_pool_no_smt();
    test_node_filter();
    test_random();
#if defined(__linux__)
    test_runtime();
#endif
    if (g_fail) { fprintf(stderr, "%d failure(s)\n", g_fail); return 1; }
    printf("test-pxa-thread-place: all passed\n");
    return 0;
}
