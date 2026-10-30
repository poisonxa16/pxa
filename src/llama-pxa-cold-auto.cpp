// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// Cold path auto choice for the Nemotron-H expert split: see llama-pxa-cold-auto.h.

#include "llama-pxa-cold-auto.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <unistd.h>
#endif

std::vector<pxa_cpu_ticks> pxa_cold_parse_proc_stat(const std::string & text) {
    std::vector<pxa_cpu_ticks> v;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 4 || line.compare(0, 3, "cpu") != 0 || line[3] < '0' || line[3] > '9') continue;
        char * end = nullptr;
        const long cpu = strtol(line.c_str() + 3, &end, 10);
        if (cpu < 0 || cpu > 65535 || !end) continue;
        unsigned long long f[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // user nice system idle iowait irq softirq steal
        int n = 0;
        const char * p = end;
        while (n < 8) {
            char * q = nullptr;
            const unsigned long long x = strtoull(p, &q, 10);
            if (q == p) break;
            f[n++] = x;
            p = q;
        }
        if (n < 4) continue;
        if ((size_t) cpu >= v.size()) v.resize(cpu + 1);
        pxa_cpu_ticks & t = v[cpu];
        t.busy = f[0] + f[2] + f[5] + f[6] + f[7];
        t.nice = f[1];
        t.idle = f[3] + f[4];
        t.seen = true;
    }
    return v;
}

pxa_cold_load pxa_cold_load_from(const std::vector<pxa_cpu_ticks> & a, const std::vector<pxa_cpu_ticks> & b,
        const std::vector<int> & set, double own_ticks) {
    pxa_cold_load r;
    r.set_cpus = (int) set.size();
    if (set.empty()) return r;
    double busy = 0, nice = 0, idle = 0;
    for (int c : set) {
        if (c < 0 || (size_t) c >= a.size() || (size_t) c >= b.size() || !a[c].seen || !b[c].seen) return r;
        // a counter that went backwards (CPU hot-unplugged and back) counts as no time
        busy += b[c].busy >= a[c].busy ? (double) (b[c].busy - a[c].busy) : 0.0;
        nice += b[c].nice >= a[c].nice ? (double) (b[c].nice - a[c].nice) : 0.0;
        idle += b[c].idle >= a[c].idle ? (double) (b[c].idle - a[c].idle) : 0.0;
    }
    const double total = busy + nice + idle;
    if (total <= 0) return r;
    // this process's own time is not load: take it from the un-niced busy time first, the rest from the niced time
    double own = std::max(0.0, own_ticks);
    const double from_busy = std::min(own, busy);
    busy -= from_busy; own -= from_busy;
    nice -= std::min(own, nice);
    const double per_cpu = total/(double) set.size();   // ticks one CPU has over the window
    r.other_busy = busy/per_cpu;
    r.other_nice = nice/per_cpu;
    r.idle       = idle/per_cpu;
    r.sampled    = true;
    return r;
}

pxa_cold_auto_out pxa_cold_auto_decide(const pxa_cold_auto_in & in) {
    pxa_cold_auto_out o;
    char buf[256];
    const int set = in.load.set_cpus;
    const int threads = in.threads > 0 ? in.threads : set;
    o.needed = std::max(0.0, in.need)*threads;
    if (in.forced == 1) { o.zc = true;  o.reason = "PXA_XCACHE_NEMO_ZC=1"; return o; }
    if (in.forced == 0) { o.zc = false; o.reason = "PXA_XCACHE_NEMO_ZC=0"; return o; }
    o.is_auto = true;
    if (set > 0 && threads > set) {
        o.zc = true;
        snprintf(buf, sizeof(buf), "auto: %d threads on %d CPUs, the CPU path would time-slice", threads, set);
        o.reason = buf;
        return o;
    }
    if (!in.load.sampled) {
        o.zc = false;
        snprintf(buf, sizeof(buf), "auto: CPU load not readable, %d threads fit the %d CPUs", threads, set);
        o.reason = buf;
        return o;
    }
    const double w = std::min(1.0, std::max(0.0, in.nice_weight));
    o.avail = std::max(0.0, (double) set - in.load.other_busy - w*in.load.other_nice);
    o.zc = o.avail < o.needed;
    snprintf(buf, sizeof(buf), "auto: %d threads, %.1f of %d CPUs free (others busy %.1f, niced %.1f), CPU path needs %.1f",
            threads, o.avail, set, in.load.other_busy, in.load.other_nice, o.needed);
    o.reason = buf;
    return o;
}

bool pxa_cold_auto_confirmable(const pxa_cold_auto_in & in, const pxa_cold_auto_out & o) {
    const int set = in.load.set_cpus;
    const int threads = in.threads > 0 ? in.threads : set;
    return o.is_auto && o.zc && in.load.sampled && !(set > 0 && threads > set);
}

pxa_cold_auto_out pxa_cold_auto_confirm(const pxa_cold_auto_in & in, const std::vector<pxa_cold_load> & more) {
    pxa_cold_auto_out o = pxa_cold_auto_decide(in);
    if (!pxa_cold_auto_confirmable(in, o)) return o;
    std::vector<double> avail = { o.avail };
    std::string seen;
    char buf[64];
    snprintf(buf, sizeof(buf), "%.1f", o.avail);
    seen = buf;
    for (const pxa_cold_load & l : more) {
        if (!l.sampled) continue;
        pxa_cold_auto_in w = in;
        w.load = l;
        const double a = pxa_cold_auto_decide(w).avail;
        avail.push_back(a);
        snprintf(buf, sizeof(buf), " %.1f", a);
        seen += buf;
    }
    if (avail.size() < 2) return o;
    std::sort(avail.begin(), avail.end());
    o.avail = avail[avail.size()/2];
    o.zc = o.avail < o.needed;
    snprintf(buf, sizeof(buf), "; confirmed over %zu windows: free ", avail.size());
    o.reason += buf;
    o.reason += seen;
    snprintf(buf, sizeof(buf), ", median %.1f -> %s", o.avail, o.zc ? "zero-copy" : "cpu");
    o.reason += buf;
    return o;
}

#if defined(__linux__)
static bool pxa_cold_read_file(const char * path, std::string & out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return !out.empty();
}

static double pxa_cold_own_ticks(void) {
    std::string s;
    if (!pxa_cold_read_file("/proc/self/stat", s)) return -1;
    const size_t rp = s.rfind(')');   // the command name may hold spaces and parentheses
    if (rp == std::string::npos) return -1;
    std::istringstream in(s.substr(rp + 2));
    std::string tok;
    double ut = -1, st = -1;
    for (int field = 3; in >> tok; ++field) {   // field 3 = state; 14 = utime, 15 = stime
        if (field == 14) ut = atof(tok.c_str());
        if (field == 15) { st = atof(tok.c_str()); break; }
    }
    return ut >= 0 && st >= 0 ? ut + st : -1;
}
#endif

pxa_cold_load pxa_cold_sample_load(int window_ms) {
    pxa_cold_load r;
#if defined(__linux__)
    std::vector<int> set;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &cs)) set.push_back(c);
    }
    r.set_cpus = (int) set.size();
    std::string s0, s1;
    const double own0 = pxa_cold_own_ticks();
    if (set.empty() || !pxa_cold_read_file("/proc/stat", s0)) return r;
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(20, window_ms)));
    const double own1 = pxa_cold_own_ticks();
    if (!pxa_cold_read_file("/proc/stat", s1)) return r;
    r = pxa_cold_load_from(pxa_cold_parse_proc_stat(s0), pxa_cold_parse_proc_stat(s1), set,
            own0 >= 0 && own1 >= own0 ? own1 - own0 : 0.0);
    r.set_cpus = (int) set.size();
#else
    (void) window_ms;
#endif
    return r;
}

pxa_cold_auto_out pxa_cold_auto_from_env(int threads) {
    pxa_cold_auto_in in;
    in.threads = threads;
    if (const char * e = getenv("PXA_XCACHE_NEMO_ZC")) {
        if (*e) in.forced = atoi(e) != 0 ? 1 : 0;
    }
    if (const char * e = getenv("PXA_XCACHE_COLD_THREADS")) in.threads = atoi(e);
    if (const char * e = getenv("PXA_XCACHE_NEMO_NICE_W"))  in.nice_weight = atof(e);
    if (const char * e = getenv("PXA_XCACHE_NEMO_CPU_NEED")) in.need = atof(e);
    if (in.forced < 0) {
        int ms = 300;
        if (const char * e = getenv("PXA_XCACHE_NEMO_SAMPLE_MS")) ms = std::max(20, atoi(e));
        in.load = pxa_cold_sample_load(ms);
        // bug #347: a zero-copy verdict holds for the whole run, so one window caught in a burst must not decide it
        int n_win = 5, gap_ms = 700;
        if (const char * e = getenv("PXA_XCACHE_NEMO_CONFIRM"))        n_win  = std::max(1, atoi(e));
        if (const char * e = getenv("PXA_XCACHE_NEMO_CONFIRM_GAP_MS")) gap_ms = std::max(0, atoi(e));
        if (n_win > 1 && pxa_cold_auto_confirmable(in, pxa_cold_auto_decide(in))) {
            std::vector<pxa_cold_load> more;
            for (int k = 1; k < n_win; ++k) {
                if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
                more.push_back(pxa_cold_sample_load(ms));
            }
            return pxa_cold_auto_confirm(in, more);
        }
    }
    return pxa_cold_auto_decide(in);
}

// ---- runtime guard

bool pxa_cold_guard_step(pxa_cold_guard_state & s, const pxa_cold_guard_cfg & c, double wait, double avail) {
    const int n_up = std::max(1, c.n_up);
    if (s.up_need <= 0) s.up_need = n_up;
    ++s.since;
    if (!s.zc) {
        s.above = 0;
        if (s.since >= std::max(1, c.n_settle)) s.up_need = n_up;   // the CPU path held: the next move back waits the base time
        s.below = wait > c.wait_down ? s.below + 1 : 0;
        if (s.below >= std::max(1, c.n_down)) {
            if (s.moves > 0 && s.since <= std::max(1, c.n_trial)) s.up_need = std::min(std::max(n_up, c.max_up), s.up_need*2);
            s.zc = true; s.below = 0; s.since = 0; ++s.moves;
            return true;
        }
    } else {
        s.below = 0;
        s.above = avail >= c.up_frac*c.needed ? s.above + 1 : 0;
        if (s.above >= s.up_need) { s.zc = false; s.above = 0; s.since = 0; ++s.moves; return true; }
    }
    return false;
}

struct pxa_cold_guard_rt {
    pxa_cold_guard_cfg   cfg;
    pxa_cold_guard_state st;
    std::vector<int>     set;
    std::vector<pxa_cpu_ticks> ticks;
    double               own  = 0;   // utime+stime (jiffies)
    double               wait = 0;   // sum of this process's threads' run-queue wait (ns)
    std::chrono::steady_clock::time_point t;
    int64_t              interval_us = 1000000;
    std::mutex           mu;
};

#if defined(__linux__)
// run-queue wait of every thread of this process (ns; /proc/self/task/<tid>/schedstat: run wait slices)
static double pxa_cold_own_wait_ns(void) {
    DIR * d = opendir("/proc/self/task");
    if (!d) return -1;
    double sum = 0;
    bool any = false;
    while (struct dirent * e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char path[300];
        snprintf(path, sizeof(path), "/proc/self/task/%s/schedstat", e->d_name);
        FILE * f = fopen(path, "r");
        if (!f) continue;   // the thread ended
        unsigned long long run = 0, wt = 0;
        if (fscanf(f, "%llu %llu", &run, &wt) == 2) { sum += (double) wt; any = true; }
        fclose(f);
    }
    closedir(d);
    return any ? sum : -1;
}
#endif

bool pxa_cold_guard_wanted(void) {
    const char * e = getenv("PXA_XCACHE_NEMO_GUARD");
    return e && *e && atoi(e) != 0;
}

pxa_cold_guard_rt * pxa_cold_guard_new(const pxa_cold_auto_out & choice) {
#if defined(__linux__)
    if (!choice.is_auto || choice.zc) return nullptr;
    // opt-in since 2026-10-19: a move changes the path, and so the rounding, of the tokens after it, so a greedy
    // request's text depended on the box's load during that request (Bolt-74B 2x P100 RC: 2-5 different greedy shas over 6 reps)
    if (!pxa_cold_guard_wanted()) return nullptr;
    auto * g = new pxa_cold_guard_rt();
    g->cfg.needed = choice.needed;
    if (const char * e = getenv("PXA_XCACHE_NEMO_GUARD_WAIT")) g->cfg.wait_down = std::max(0.0, atof(e));
    if (const char * e = getenv("PXA_XCACHE_NEMO_GUARD_UP"))   g->cfg.up_frac = std::max(0.0, atof(e));
    if (const char * e = getenv("PXA_XCACHE_NEMO_NICE_W"))     g->cfg.nice_weight = atof(e);
    if (const char * e = getenv("PXA_XCACHE_NEMO_GUARD_MS"))   g->interval_us = (int64_t) std::max(100, atoi(e))*1000;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &cs)) g->set.push_back(c);
    }
    std::string s0;
    g->wait = pxa_cold_own_wait_ns();
    g->own  = pxa_cold_own_ticks();
    if (g->set.empty() || g->wait < 0 || g->own < 0 || !pxa_cold_read_file("/proc/stat", s0)) { delete g; return nullptr; }
    g->ticks = pxa_cold_parse_proc_stat(s0);
    g->t = std::chrono::steady_clock::now();
    return g;
#else
    (void) choice;
    return nullptr;
#endif
}

void pxa_cold_guard_free(pxa_cold_guard_rt * g) { delete g; }

int pxa_cold_guard_poll(pxa_cold_guard_rt * g, std::string & why) {
#if defined(__linux__)
    if (!g) return 0;
    std::unique_lock<std::mutex> lk(g->mu, std::try_to_lock);
    if (!lk.owns_lock()) return 0;
    const auto now = std::chrono::steady_clock::now();
    const int64_t dt_us = std::chrono::duration_cast<std::chrono::microseconds>(now - g->t).count();
    if (dt_us < g->interval_us) return 0;
    std::string s1;
    const double wait = pxa_cold_own_wait_ns();
    const double own  = pxa_cold_own_ticks();
    if (wait < 0 || own < 0 || !pxa_cold_read_file("/proc/stat", s1)) return 0;
    std::vector<pxa_cpu_ticks> ticks = pxa_cold_parse_proc_stat(s1);
    const pxa_cold_load l = pxa_cold_load_from(g->ticks, ticks, g->set, std::max(0.0, own - g->own));
    const double wait_cpus = std::max(0.0, wait - g->wait)/(dt_us*1e3);
    g->ticks = std::move(ticks); g->own = own; g->wait = wait; g->t = now;
    if (!l.sampled) return 0;
    const double w = std::min(1.0, std::max(0.0, g->cfg.nice_weight));
    const double avail = std::max(0.0, (double) l.set_cpus - l.other_busy - w*l.other_nice);
    const bool moved = pxa_cold_guard_step(g->st, g->cfg, wait_cpus, avail);
    char b[240];
    snprintf(b, sizeof(b), "threads waited %.2f CPUs for a core, %.1f of %d CPUs free (others busy %.1f, niced %.1f), CPU path needs %.1f, back after %d s at %.1f free",
            wait_cpus, avail, l.set_cpus, l.other_busy, l.other_nice, g->cfg.needed, g->st.up_need, g->cfg.up_frac*g->cfg.needed);
    why = b;
    if (moved) return g->st.zc ? 1 : -1;
    return g->st.zc && g->st.since > 0 && g->st.since % 10 == 0 ? 2 : 0;
#else
    (void) g; (void) why;
    return 0;
#endif
}
