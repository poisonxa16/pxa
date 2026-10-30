// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// Nemotron-H expert split cold path auto choice (src/llama-pxa-cold-auto.h): /proc/stat parsing, the per-set load with this
// process's own time subtracted, the decision on the box cases it was made for, and one live sample of this machine.
// No model, no GPU.

#include "llama-pxa-cold-auto.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv, unsetenv
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

// a /proc/stat text with n CPUs; cpu c gets (user, nice, system, idle) from f(c)
template <typename F>
static std::string stat_text(int n, F f) {
    std::string s = "cpu  1 2 3 4 5 6 7 8 0 0\n";
    for (int c = 0; c < n; ++c) {
        unsigned long long u, ni, sy, id;
        f(c, u, ni, sy, id);
        char b[160];
        snprintf(b, sizeof(b), "cpu%d %llu %llu %llu %llu 0 0 0 0 0 0\n", c, u, ni, sy, id);
        s += b;
    }
    s += "intr 1 2 3\nctxt 99\nbtime 1\nprocesses 5\n";
    return s;
}

static std::vector<int> range(int a, int b) { std::vector<int> v; for (int c = a; c <= b; ++c) v.push_back(c); return v; }

int main() {
    // ---- parsing
    {
        const std::string t = "cpu  100 0 0 0\ncpu0 10 20 30 40 5 1 2 3 0 0\ncpu2 1 1 1 1\nintr 5\ncpufreq junk\n";
        auto v = pxa_cold_parse_proc_stat(t);
        CHECK(v.size() == 3, "size %zu", v.size());
        CHECK(v[0].seen && v[0].busy == 10 + 30 + 1 + 2 + 3 && v[0].nice == 20 && v[0].idle == 45, "cpu0 %llu %llu %llu",
                (unsigned long long) v[0].busy, (unsigned long long) v[0].nice, (unsigned long long) v[0].idle);
        CHECK(!v[1].seen, "cpu1 absent");
        CHECK(v[2].seen && v[2].busy == 2 && v[2].idle == 1, "cpu2 four fields");
        CHECK(pxa_cold_parse_proc_stat("").empty(), "empty text");
    }
    // ---- load over a set (72 CPUs, 100 ticks per CPU over the window)
    const int N = 72;
    auto a = pxa_cold_parse_proc_stat(stat_text(N, [](int, unsigned long long & u, unsigned long long & ni, unsigned long long & sy, unsigned long long & id) { u = ni = sy = id = 1000; }));
    {
        // quiet node 1 (54-71) while node 0 builds: the set sees no load
        auto b = pxa_cold_parse_proc_stat(stat_text(N, [](int c, unsigned long long & u, unsigned long long & ni, unsigned long long & sy, unsigned long long & id) {
            u = 1000; ni = 1000; sy = 1000; id = 1000;
            if (c >= 36 && c <= 53) ni += 100; else id += 100; }));
        pxa_cold_load l = pxa_cold_load_from(a, b, range(54, 71), 0);
        CHECK(l.sampled && l.set_cpus == 18 && near(l.other_busy, 0) && near(l.other_nice, 0) && near(l.idle, 18), "quiet node1 %f %f %f", l.other_busy, l.other_nice, l.idle);
        pxa_cold_auto_in in; in.threads = 16; in.load = l;
        pxa_cold_auto_out o = pxa_cold_auto_decide(in);
        CHECK(!o.zc && near(o.avail, 18) && near(o.needed, 14.4), "quiet -t 16 -> cpu (%s)", o.reason.c_str());
        // the same reading seen from node-0 cores 36-43 (-t 8): eight CPUs of niced build work -> zero-copy at weight 1, cpu at weight 0
        pxa_cold_load l0 = pxa_cold_load_from(a, b, range(36, 43), 0);
        CHECK(l0.sampled && near(l0.other_nice, 8) && near(l0.other_busy, 0), "node0 niced %f", l0.other_nice);
        in.threads = 8; in.load = l0;
        o = pxa_cold_auto_decide(in);
        CHECK(o.zc && near(o.avail, 0), "busy niced, weight 1 -> zc (%s)", o.reason.c_str());
        in.nice_weight = 0;
        o = pxa_cold_auto_decide(in);
        CHECK(!o.zc && near(o.avail, 8), "busy niced, weight 0 -> cpu (%s)", o.reason.c_str());
        in.nice_weight = 0.5;
        o = pxa_cold_auto_decide(in);
        CHECK(o.zc && near(o.avail, 4), "busy niced, weight 0.5 -> 4 free < 7.2 -> zc (%s)", o.reason.c_str());
    }
    {
        // un-niced load on 3 of 18 CPUs, -t 16: 15 free >= 14.4 -> cpu; on 4: 14 < 14.4 -> zero-copy
        for (int busy_cpus : {3, 4}) {
            auto b = pxa_cold_parse_proc_stat(stat_text(N, [busy_cpus](int c, unsigned long long & u, unsigned long long & ni, unsigned long long & sy, unsigned long long & id) {
                u = ni = sy = id = 1000;
                if (c >= 54 && c < 54 + busy_cpus) { u += 80; sy += 20; } else id += 100; }));
            pxa_cold_auto_in in; in.threads = 16; in.load = pxa_cold_load_from(a, b, range(54, 71), 0);
            pxa_cold_auto_out o = pxa_cold_auto_decide(in);
            CHECK(near(in.load.other_busy, busy_cpus), "busy %d -> %f", busy_cpus, in.load.other_busy);
            CHECK(o.zc == (busy_cpus == 4), "busy %d -> %s (%s)", busy_cpus, o.zc ? "zc" : "cpu", o.reason.c_str());
        }
    }
    {
        // this process's own spinning threads are not load: 16 CPUs busy, all of it our own time -> free
        auto b = pxa_cold_parse_proc_stat(stat_text(N, [](int c, unsigned long long & u, unsigned long long & ni, unsigned long long & sy, unsigned long long & id) {
            u = ni = sy = id = 1000;
            if (c >= 54 && c < 70) u += 100; else id += 100; }));
        pxa_cold_load l = pxa_cold_load_from(a, b, range(54, 71), 1600);
        CHECK(l.sampled && near(l.other_busy, 0) && near(l.other_nice, 0), "own time %f %f", l.other_busy, l.other_nice);
        // own time beyond the un-niced busy time comes off the niced time; never negative
        l = pxa_cold_load_from(a, b, range(54, 71), 5000);
        CHECK(near(l.other_busy, 0) && near(l.other_nice, 0), "own overflow %f %f", l.other_busy, l.other_nice);
        // half of it ours -> 8 CPUs of others
        l = pxa_cold_load_from(a, b, range(54, 71), 800);
        CHECK(near(l.other_busy, 8), "half own %f", l.other_busy);
    }
    {
        // a CPU of the set missing from either reading -> not sampled; empty set -> not sampled
        auto b = pxa_cold_parse_proc_stat(stat_text(40, [](int, unsigned long long & u, unsigned long long & ni, unsigned long long & sy, unsigned long long & id) { u = ni = sy = id = 2000; }));
        CHECK(!pxa_cold_load_from(a, b, range(54, 71), 0).sampled, "cpu missing");
        CHECK(!pxa_cold_load_from(a, a, {}, 0).sampled, "empty set");
        CHECK(!pxa_cold_load_from(a, a, range(0, 3), 0).sampled, "no time passed");
        // counters that went backwards count as no time (not a huge unsigned delta)
        auto back = pxa_cold_parse_proc_stat(stat_text(N, [](int c, unsigned long long & u, unsigned long long & ni, unsigned long long & sy, unsigned long long & id) {
            u = c == 0 ? 10 : 1000; ni = 1000; sy = 1000; id = 1100; }));
        pxa_cold_load l = pxa_cold_load_from(a, back, range(0, 3), 0);
        CHECK(l.sampled && near(l.other_busy, 0) && near(l.idle, 4), "backwards %f %f", l.other_busy, l.idle);
    }
    {
        // forced switches win over everything; oversubscription and an unreadable load
        pxa_cold_auto_in in; in.threads = 16;
        in.load.sampled = true; in.load.set_cpus = 18; in.load.other_busy = 18;
        in.forced = 0; CHECK(!pxa_cold_auto_decide(in).zc, "forced cpu under load");
        in.load.other_busy = 0;
        in.forced = 1; CHECK(pxa_cold_auto_decide(in).zc, "forced zc on idle CPUs");
        in.forced = -1; in.threads = 32;
        pxa_cold_auto_out o = pxa_cold_auto_decide(in);
        CHECK(o.zc, "32 threads on 18 CPUs -> zc (%s)", o.reason.c_str());
        in.threads = 16; in.load.sampled = false;
        o = pxa_cold_auto_decide(in);
        CHECK(!o.zc, "load unknown, threads fit -> cpu (%s)", o.reason.c_str());
        in.load.set_cpus = 0;
        CHECK(!pxa_cold_auto_decide(in).zc, "nothing known -> cpu");
        // threads unknown: every CPU of the set
        in.threads = 0; in.load.sampled = true; in.load.set_cpus = 18; in.load.other_busy = 1;
        o = pxa_cold_auto_decide(in);
        CHECK(!o.zc && near(o.needed, 16.2) && near(o.avail, 17), "threads unknown, 1 busy -> cpu (%s)", o.reason.c_str());
        in.load.other_busy = 2;
        CHECK(pxa_cold_auto_decide(in).zc, "threads unknown, 2 busy -> zc");
    }
    {
        // bug #347: a zero-copy verdict is confirmed over more windows (median free CPUs); a cpu verdict is never re-read
        auto win = [](double busy) { pxa_cold_load l; l.sampled = true; l.set_cpus = 18; l.other_busy = busy; return l; };
        pxa_cold_auto_in in; in.threads = 16; in.load = win(9.5);   // Bolt-74B 2x P100 st0 boot: 7.4 free (here 8.5) < 14.4
        const pxa_cold_auto_out first = pxa_cold_auto_decide(in);
        CHECK(first.zc && pxa_cold_auto_confirmable(in, first), "burst window -> confirmable zc (%s)", first.reason.c_str());
        // a burst: the later windows are free -> median 17.6 -> cpu
        pxa_cold_auto_out o = pxa_cold_auto_confirm(in, { win(0.2), win(0.4), win(9.0), win(0.1) });
        CHECK(!o.zc && near(o.avail, 17.6) && near(o.needed, 14.4), "burst -> cpu (%s)", o.reason.c_str());
        CHECK(o.reason.rfind(first.reason, 0) == 0 && o.reason.find("confirmed over 5 windows: free 8.5 17.8 17.6 9.0 17.9, median 17.6 -> cpu") != std::string::npos,
                "reason names every window (%s)", o.reason.c_str());
        // persistent load: free 8.5 8.0 7.0 17.0 9.0, the median (8.5) stays busy -> zero-copy
        o = pxa_cold_auto_confirm(in, { win(10), win(11), win(1), win(9) });
        CHECK(o.zc && near(o.avail, 8.5), "persistent -> zc (%s)", o.reason.c_str());
        // an even count takes the upper middle: one free window of two overturns the first
        o = pxa_cold_auto_confirm(in, { win(1) });
        CHECK(!o.zc && near(o.avail, 17), "two windows -> upper middle (%s)", o.reason.c_str());
        // unsampled windows are skipped; none read -> the first verdict, unchanged
        pxa_cold_load un; un.set_cpus = 18;
        o = pxa_cold_auto_confirm(in, { un, un });
        CHECK(o.zc && o.reason == first.reason, "no window read -> first verdict (%s)", o.reason.c_str());
        o = pxa_cold_auto_confirm(in, { un, win(0), un });
        CHECK(!o.zc && o.reason.find("over 2 windows") != std::string::npos, "unsampled skipped (%s)", o.reason.c_str());
        // a cpu verdict is final: the same output and reason as the single window, whatever later windows say
        in.load = win(1.0);
        const pxa_cold_auto_out quiet = pxa_cold_auto_decide(in);
        CHECK(!quiet.zc && !pxa_cold_auto_confirmable(in, quiet), "quiet window not confirmable");
        o = pxa_cold_auto_confirm(in, { win(18), win(18), win(18), win(18) });
        CHECK(!o.zc && o.reason == quiet.reason && near(o.avail, quiet.avail), "cpu verdict unchanged (%s)", o.reason.c_str());
        // forced and oversubscribed verdicts are not re-read
        in.load = win(18); in.forced = 1;
        CHECK(!pxa_cold_auto_confirmable(in, pxa_cold_auto_decide(in)), "forced zc not confirmable");
        CHECK(pxa_cold_auto_confirm(in, { win(0) }).zc, "forced zc stays");
        in.forced = -1; in.threads = 32;
        o = pxa_cold_auto_confirm(in, { win(0), win(0) });
        CHECK(o.zc && o.reason.find("confirmed") == std::string::npos, "oversubscribed stays zc (%s)", o.reason.c_str());
        in.threads = 16; in.load.sampled = false;
        CHECK(!pxa_cold_auto_confirmable(in, pxa_cold_auto_decide(in)), "unreadable load not confirmable");
    }
    {
        // the confirm windows through the environment: a need no box can meet makes every window say zero-copy
        setenv("PXA_XCACHE_NEMO_SAMPLE_MS", "20", 1);
        setenv("PXA_XCACHE_NEMO_CPU_NEED", "1000", 1);
        setenv("PXA_XCACHE_NEMO_CONFIRM_GAP_MS", "0", 1);
        setenv("PXA_XCACHE_COLD_THREADS", "1", 1);
        pxa_cold_auto_out o = pxa_cold_auto_from_env();
#if defined(__linux__)
        CHECK(o.zc && o.reason.find("confirmed over 5 windows") != std::string::npos, "env: 5 windows by default (%s)", o.reason.c_str());
        setenv("PXA_XCACHE_NEMO_CONFIRM", "3", 1);
        o = pxa_cold_auto_from_env();
        CHECK(o.zc && o.reason.find("confirmed over 3 windows") != std::string::npos, "env: CONFIRM=3 (%s)", o.reason.c_str());
        // these two read the CPU load too (Linux /proc/stat): elsewhere the load is not readable and the verdict is the CPU path
        setenv("PXA_XCACHE_NEMO_CONFIRM", "0", 1);
        o = pxa_cold_auto_from_env();
        CHECK(o.zc && o.reason.find("confirmed") == std::string::npos, "env: CONFIRM=0 = one window, the off switch (%s)", o.reason.c_str());
        setenv("PXA_XCACHE_NEMO_CONFIRM", "1", 1);
        o = pxa_cold_auto_from_env();
        CHECK(o.zc && o.reason.find("confirmed") == std::string::npos, "env: CONFIRM=1 = one window (%s)", o.reason.c_str());
#endif
        unsetenv("PXA_XCACHE_NEMO_CONFIRM");
        unsetenv("PXA_XCACHE_NEMO_CONFIRM_GAP_MS");
        unsetenv("PXA_XCACHE_NEMO_CPU_NEED");
        unsetenv("PXA_XCACHE_NEMO_SAMPLE_MS");
        unsetenv("PXA_XCACHE_COLD_THREADS");
    }
    {
        // the environment front end: forced values never sample; the live sample of this machine is sane
        setenv("PXA_XCACHE_NEMO_ZC", "1", 1);
        pxa_cold_auto_out o = pxa_cold_auto_from_env();
        CHECK(o.zc && o.reason == "PXA_XCACHE_NEMO_ZC=1", "env =1 (%s)", o.reason.c_str());
        setenv("PXA_XCACHE_NEMO_ZC", "0", 1);
        o = pxa_cold_auto_from_env();
        CHECK(!o.zc && o.reason == "PXA_XCACHE_NEMO_ZC=0", "env =0 (%s)", o.reason.c_str());
        unsetenv("PXA_XCACHE_NEMO_ZC");
        setenv("PXA_XCACHE_NEMO_SAMPLE_MS", "60", 1);
        setenv("PXA_XCACHE_COLD_THREADS", "1", 1);
        o = pxa_cold_auto_from_env();
        CHECK(o.reason.rfind("auto:", 0) == 0, "auto reason (%s)", o.reason.c_str());
#if defined(__linux__)
        pxa_cold_load l = pxa_cold_sample_load(60);
        CHECK(l.sampled && l.set_cpus > 0 && l.other_busy >= 0 && l.other_nice >= 0 && l.idle >= 0 &&
              l.other_busy + l.other_nice + l.idle <= l.set_cpus + 1e-6, "live sample %d %f %f %f", l.set_cpus, l.other_busy, l.other_nice, l.idle);
        printf("live: %d CPUs, others busy %.2f, niced %.2f, idle %.2f; auto -t 1 -> %s (%s)\n", l.set_cpus, l.other_busy, l.other_nice,
                l.idle, o.zc ? "zero-copy" : "cpu", o.reason.c_str());
#endif
    }
    {
        // this load's cold-path threads come with the model params (llama_model_params::pxa_cold_threads, common init's -t);
        // an exported PXA_XCACHE_COLD_THREADS wins; the choice never writes the environment (a value written there would reach
        // every later load of the process and every child process as if exported)
        unsetenv("PXA_XCACHE_COLD_THREADS");
        setenv("PXA_XCACHE_NEMO_ZC", "1", 1);   // forced: no sample; needed = 0.9 x threads
        pxa_cold_auto_out o = pxa_cold_auto_from_env(12);
        CHECK(near(o.needed, 10.8), "threads from the load params (needed %f)", o.needed);
        CHECK(getenv("PXA_XCACHE_COLD_THREADS") == nullptr, "the choice wrote no PXA_XCACHE_COLD_THREADS");
        o = pxa_cold_auto_from_env(6);
        CHECK(near(o.needed, 5.4), "a second load with other threads gets its own (needed %f)", o.needed);
        setenv("PXA_XCACHE_COLD_THREADS", "4", 1);
        o = pxa_cold_auto_from_env(12);
        CHECK(near(o.needed, 3.6), "an exported PXA_XCACHE_COLD_THREADS wins over the load params (needed %f)", o.needed);
        unsetenv("PXA_XCACHE_COLD_THREADS");
        unsetenv("PXA_XCACHE_NEMO_ZC");
    }
    {
        // runtime guard hysteresis: CPU path -> zero-copy after n_down intervals of waiting above wait_down; back after n_up
        // intervals with the free CPUs covering the need; a single quiet / busy interval resets the run
        pxa_cold_guard_cfg c; c.needed = 14.4; c.wait_down = 1.0; c.n_down = 2; c.n_up = 5;
        pxa_cold_guard_state s;
        CHECK(!pxa_cold_guard_step(s, c, 0.2, 18) && !s.zc, "quiet stays cpu");
        CHECK(!pxa_cold_guard_step(s, c, 3.0, 2) && !s.zc && s.below == 1, "one busy interval");
        CHECK(!pxa_cold_guard_step(s, c, 0.5, 2) && !s.zc && s.below == 0, "a quiet interval resets");
        CHECK(!pxa_cold_guard_step(s, c, 3.0, 2), "busy 1");
        CHECK(pxa_cold_guard_step(s, c, 1.5, 2) && s.zc, "busy 2 -> zc");
        // on zero-copy the wait no longer matters (our threads idle): only the free CPUs, at up_frac (0.75) of the need = 10.8
        for (int i = 0; i < 4; ++i) CHECK(!pxa_cold_guard_step(s, c, 0, 11) && s.zc, "free %d of 5", i + 1);
        CHECK(!pxa_cold_guard_step(s, c, 0, 10) && s.zc && s.above == 0, "busy again resets the climb");
        for (int i = 0; i < 4; ++i) pxa_cold_guard_step(s, c, 0, 10.8);
        CHECK(pxa_cold_guard_step(s, c, 9.0, 10.8) && !s.zc, "5 free intervals -> cpu (trial)");
        // the trial fails at once: back down, and the next move back waits twice as long (10), then 20, capped at 64
        pxa_cold_guard_step(s, c, 3.0, 2);
        CHECK(pxa_cold_guard_step(s, c, 3.0, 2) && s.zc && s.up_need == 10, "failed trial doubles the wait (%d)", s.up_need);
        for (int i = 0; i < 9; ++i) CHECK(!pxa_cold_guard_step(s, c, 0, 18) && s.zc, "waiting 10: %d", i + 1);
        CHECK(pxa_cold_guard_step(s, c, 0, 18) && !s.zc, "10 free intervals -> cpu");
        pxa_cold_guard_step(s, c, 3.0, 2);
        CHECK(pxa_cold_guard_step(s, c, 3.0, 2) && s.zc && s.up_need == 20, "second failed trial -> 20 (%d)", s.up_need);
        pxa_cold_guard_cfg c2 = c; c2.max_up = 16;
        pxa_cold_guard_state s2; s2.zc = true; s2.moves = 3; s2.up_need = 16;
        for (int i = 0; i < 16; ++i) pxa_cold_guard_step(s2, c2, 0, 18);
        CHECK(!s2.zc, "move back at the cap");
        pxa_cold_guard_step(s2, c2, 3.0, 2); pxa_cold_guard_step(s2, c2, 3.0, 2);
        CHECK(s2.zc && s2.up_need == 16, "capped at max_up (%d)", s2.up_need);
        // a trial that holds n_settle intervals resets the wait to n_up
        for (int i = 0; i < 16; ++i) pxa_cold_guard_step(s2, c2, 0, 18);
        for (int i = 0; i < 30; ++i) CHECK(!pxa_cold_guard_step(s2, c2, 0.1, 18) && !s2.zc, "settled %d", i);
        CHECK(s2.up_need == c2.n_up, "settle resets (%d)", s2.up_need);
        // the factory: opt-in (PXA_XCACHE_NEMO_GUARD=1, 2026-10-19: the guard made greedy output depend on the load);
        // even then no guard for a forced or a zero-copy choice, nor with PXA_XCACHE_NEMO_GUARD=0
        pxa_cold_auto_out ch; ch.is_auto = true; ch.zc = false; ch.needed = 1;
        pxa_cold_auto_out forced = ch; forced.is_auto = false;
        pxa_cold_auto_out zc = ch; zc.zc = true;
        unsetenv("PXA_XCACHE_NEMO_GUARD");
        CHECK(!pxa_cold_guard_wanted() && pxa_cold_guard_new(ch) == nullptr, "guard off by default (deterministic)");
        setenv("PXA_XCACHE_NEMO_GUARD", "1", 1);
        CHECK(pxa_cold_guard_wanted(), "PXA_XCACHE_NEMO_GUARD=1 asks for the guard");
        CHECK(pxa_cold_guard_new(forced) == nullptr && pxa_cold_guard_new(zc) == nullptr, "guard only for an auto cpu choice");
        setenv("PXA_XCACHE_NEMO_GUARD", "0", 1);
        CHECK(pxa_cold_guard_new(ch) == nullptr, "guard off switch");
        setenv("PXA_XCACHE_NEMO_GUARD", "1", 1);
#if defined(__linux__)
        // live: a guard polls nothing before its interval, then reads this machine without moving on one interval
        setenv("PXA_XCACHE_NEMO_GUARD_MS", "100", 1);
        pxa_cold_guard_rt * g = pxa_cold_guard_new(ch);
        CHECK(g != nullptr, "live guard");
        std::string why;
        CHECK(pxa_cold_guard_poll(g, why) == 0, "no poll before the interval");
        volatile double x = 0; const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(150)) x += 1;
        CHECK(pxa_cold_guard_poll(g, why) == 0, "one interval never moves (n_down 2)");
        pxa_cold_guard_free(g);
#endif
        unsetenv("PXA_XCACHE_NEMO_GUARD");
    }
    if (g_fail) { fprintf(stderr, "test-pxa-cold-auto: %d FAILED\n", g_fail); return 1; }
    printf("test-pxa-cold-auto: OK\n");
    return 0;
}
