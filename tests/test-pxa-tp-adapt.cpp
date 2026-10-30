// PXA_TSPLIT_WORKERS_ADAPT state machine + the default worker cpu pick (ggml/src/pxa-tp-adapt.h), table-driven. Pure logic:
// no GPU, no model, no threads. The "box" is a function (graph index, mode) -> graph period in us; a graph's period is the
// time from its entry to the next graph's entry, as the scheduler measures it.
#include "pxa-tp-adapt.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct run_out {
    std::vector<int> mode;       // per graph
    std::vector<int> pref;       // a.pref after the graph's step
    int max_interval = 0;
    int decisions    = 0;
    pxa_tp_adapt a;
};

// period(i, mode) in us; ntok(i) the stated width (default 1)
static run_out run(int n, const std::function<float(int, int)> & period, const std::function<int(int)> & ntok = nullptr) {
    run_out r;
    int64_t now = 1000000;
    for (int i = 0; i < n; ++i) {
        pxa_tp_adapt_decision d;
        const int w = ntok ? ntok(i) : 1;
        const int m = pxa_tp_adapt_step(r.a, now, 144, w, &d);
        r.mode.push_back(m);
        r.pref.push_back(r.a.pref);
        r.max_interval = std::max(r.max_interval, r.a.interval);
        r.decisions += d.decided ? 1 : 0;
        now += (int64_t) period(i, m);
    }
    return r;
}

static int count_mode(const run_out & r, int m, int from, int to) {
    int c = 0;
    for (int i = from; i < to && i < (int) r.mode.size(); ++i) { c += r.mode[i] == m; }
    return c;
}

// time spent in graphs of mode m (m = -1: all graphs), us
static double run_time(const run_out & r, const std::function<float(int, int)> & period, int m) {
    double t = 0;
    for (int i = 0; i < (int) r.mode.size(); ++i) { if (m < 0 || r.mode[i] == m) { t += period(i, r.mode[i]); } }
    return t;
}

static int first_pref(const run_out & r, int m, int from) {
    for (int i = from; i < (int) r.pref.size(); ++i) { if (r.pref[i] == m) { return i; } }
    return -1;
}

int main() {
    const float W = 24750.0f;      // C031 quiet: workers 40.4 t/s
    const float P = 27200.0f;      // plain loop 36.75 t/s
    const float C = 250000.0f;     // workers on a contended core: ~4 t/s

    // 1. quiet box: starts plain, the first probe moves to the workers within a few graphs, they keep >= 97% of the run
    {
        run_out r = run(3000, [&](int, int m) { return m ? W : P; });
        const int f = first_pref(r, 1, 0);
        CHECK(f >= 0 && f <= 8, "quiet: workers preferred at graph %d (want <= 8)", f);
        CHECK(r.a.pref == 1, "quiet: final pref %d", r.a.pref);
        CHECK(count_mode(r, 1, 0, 3000) >= 2910, "quiet: %d/3000 graphs on the workers", count_mode(r, 1, 0, 3000));
        CHECK(r.max_interval == 512, "quiet: interval reaches the 512 cap (max %d)", r.max_interval);
    }
    // 2. collapsed workers from the start: stays plain, one graph per probe, the time-based floor holds the probes near 1%
    //    of the run: interval = 100 x C/P (919 graphs at 250 ms vs 27.2 ms), up to 8192
    {
        run_out r = run(3000, [&](int, int m) { return m ? C : P; });
        CHECK(r.a.pref == 0, "collapse: final pref %d", r.a.pref);
        CHECK(r.a.switches == 0, "collapse: %d switches", r.a.switches);
        const int floor_c = (int) (100.0f * C / P);
        CHECK(r.max_interval == floor_c, "collapse: interval = the time floor %d (max %d)", floor_c, r.max_interval);
        CHECK(count_mode(r, 1, 0, 3000) <= 6, "collapse: %d worker graphs", count_mode(r, 1, 0, 3000));
        const int probes = r.decisions;
        CHECK(probes >= 2 && probes <= 5, "collapse: %d probes in 3000 graphs", probes);
        const double t_w = run_time(r, [&](int, int m) { return m ? C : P; }, 1);
        const double t   = run_time(r, [&](int, int m) { return m ? C : P; }, -1);
        CHECK(t_w <= 0.02 * t, "collapse: worker graphs %.2f%% of the run (want <= 2%%)", 100.0 * t_w / t);
    }
    // 2b. collapsed workers slower than the old 1 s sample cut (a burned core: 0.9 s, 1.5 s per graph): the probe graph is
    //     still a sample, so each probe ends after ONE graph and the floor holds the workers <= 2% of the run
    for (const float CS : {900000.0f, 1500000.0f}) {
        auto per = [&](int, int m) { return m ? CS : P; };
        run_out r = run(20000, per);
        CHECK(r.a.pref == 0 && r.a.switches == 0, "slow collapse %.0f: pref %d switches %d", CS, r.a.pref, r.a.switches);
        CHECK(count_mode(r, 1, 0, 20000) == r.decisions, "slow collapse %.0f: %d worker graphs for %d probes (want 1 each)",
              CS, count_mode(r, 1, 0, 20000), r.decisions);
        CHECK(r.max_interval == std::min(8192, (int) (100.0f * CS / P)), "slow collapse %.0f: interval %d", CS, r.max_interval);
        const double t_w = run_time(r, per, 1), t = run_time(r, per, -1);
        CHECK(t_w <= 0.02 * t, "slow collapse %.0f: worker graphs %.2f%% of the run (want <= 2%%)", CS, 100.0 * t_w / t);
    }
    // 3. contention arrives at graph 1000 while the workers are preferred: back on the plain loop within 6 graphs
    {
        run_out r = run(2000, [&](int i, int m) { return m ? (i >= 1000 ? C : W) : P; });
        CHECK(r.pref[999] == 1, "arrive: workers preferred before the contention (pref %d)", r.pref[999]);
        const int f = first_pref(r, 0, 1000);
        CHECK(f >= 1000 && f <= 1006, "arrive: plain preferred at graph %d (want <= 1006)", f);
        CHECK(count_mode(r, 1, 1000, 2000) <= 20, "arrive: %d worker graphs after the contention", count_mode(r, 1, 1000, 2000));
    }
    // 4. contention leaves at graph 1500: the workers come back within one maximal interval (+ probe)
    {
        run_out r = run(3000, [&](int i, int m) { return m ? (i < 1500 ? C : W) : P; });
        CHECK(r.pref[1499] == 0, "leave: plain preferred during the contention");
        const int f = first_pref(r, 1, 1500);
        CHECK(f >= 1500 && f <= 1500 + (int) (100.0f * C / P) + 8, "leave: workers preferred again at graph %d", f);
        CHECK(r.a.pref == 1, "leave: final pref %d", r.a.pref);
    }
    // 4b. the preferred workers collapse past 1 s per graph (a burner lands on the pinned cores): still seen, back on the
    //     plain loop within 6 graphs
    {
        run_out r = run(2000, [&](int i, int m) { return m ? (i >= 1000 ? 1500000.0f : W) : P; });
        CHECK(r.pref[999] == 1, "arrive slow: workers preferred before (pref %d)", r.pref[999]);
        const int f = first_pref(r, 0, 1000);
        CHECK(f >= 1000 && f <= 1006, "arrive slow: plain preferred at graph %d (want <= 1006)", f);
        CHECK(count_mode(r, 1, 1000, 2000) <= 6, "arrive slow: %d worker graphs after", count_mode(r, 1, 1000, 2000));
    }
    // 4c. one 5 s idle gap between two decode graphs (no prompt graph in between) on a quiet box: one outlier, no switch
    {
        run_out r = run(3000, [&](int i, int m) { return i == 1500 ? 5000000.0f : (m ? W : P); });
        CHECK(r.a.pref == 1 && r.pref[1499] == 1, "gap: workers stay preferred (pref %d)", r.a.pref);
        CHECK(count_mode(r, 1, 1500, 3000) >= 1450, "gap: %d/1500 worker graphs after the gap", count_mode(r, 1, 1500, 3000));
    }
    // 5. a mild 2% difference is below the 3% bar: never switches away from the plain start
    {
        run_out r = run(2000, [&](int, int m) { return m ? 0.98f * P : P; });
        CHECK(r.a.switches == 0 && r.a.pref == 0, "mild: switches %d pref %d", r.a.switches, r.a.pref);
    }
    // 6. drift: the preferred workers slow 2x without a full collapse (2x worse than plain): re-probe and leave quickly
    {
        run_out r = run(1500, [&](int i, int m) { return m ? (i >= 1000 ? 2.0f * W : W) : P; });
        const int f = first_pref(r, 0, 1000);
        CHECK(f >= 1000 && f <= 1008, "drift: plain preferred at graph %d (want <= 1008)", f);
    }
    // 7. the first period after a mode change is not filed (it carries the switch), unless a probe's first period is > 2x
    {
        pxa_tp_adapt a;
        int64_t now = 1000;
        // two plain graphs -> one plain sample (the very first graph has no previous entry)
        pxa_tp_adapt_step(a, now, 10, 1, nullptr); now += 1000;
        pxa_tp_adapt_step(a, now, 10, 1, nullptr); now += 1000;
        CHECK(a.nring[0] == 0, "skip: the period after the first mode set is dropped (nring %d)", a.nring[0]);
        pxa_tp_adapt_step(a, now, 10, 1, nullptr); now += 1000;
        CHECK(a.nring[0] == 1, "skip: then plain samples land (nring %d)", a.nring[0]);
        // a different split count breaks the chain: no sample
        pxa_tp_adapt_step(a, now, 11, 1, nullptr); now += 1000;
        CHECK(a.nring[0] == 1, "skip: a split-count change is not a sample (nring %d)", a.nring[0]);
    }
    // 8. wide graphs: the workers until a decode probe decided, then the preferred mode; never sampled
    {
        pxa_tp_adapt a;
        CHECK(pxa_tp_adapt_step(a, 1000, 50, 512, nullptr) == 1, "wide: workers before any decode decision");
        CHECK(a.nring[0] == 0 && a.nring[1] == 0, "wide: not sampled");
        run_out r = run(200, [&](int, int m) { return m ? C : P; }, [](int i) { return i < 3 ? 512 : 1; });
        CHECK(r.mode[0] == 1 && r.mode[1] == 1 && r.mode[2] == 1, "wide: the prompt graphs ran the workers");
        CHECK(r.decisions >= 1 && r.a.pref == 0, "wide: decode decided plain (decisions %d pref %d)", r.decisions, r.a.pref);
        CHECK(pxa_tp_adapt_step(r.a, 999999999, 50, 512, nullptr) == 0, "wide: after the decision a wide graph follows pref");
    }
    // 9. the default cpu pick: node0 = 0-17,36-53
    {
        std::vector<int> node0;
        for (int c = 0; c <= 17; ++c) { node0.push_back(c); }
        for (int c = 36; c <= 53; ++c) { node0.push_back(c); }
        auto range = [](int a, int b) { std::vector<int> v; for (int c = a; c <= b; ++c) { v.push_back(c); } return v; };
        auto eq = [](const std::vector<int> & x, std::vector<int> y) { return x == y; };
        CHECK(eq(pxa_tp_pick_cpus(node0, range(36, 47), -1, 4, 0), {47, 46, 45, 44}), "pick: cpuset 36-47 -> 47..44");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(36, 47), -1, 4, 1), {43, 42, 41, 40}), "pick: second pool 43..40");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(36, 53), -1, 4, 0), {53, 52, 51, 50}), "pick: the RC's 36-53 unchanged 53..50");
        CHECK(eq(pxa_tp_pick_cpus(node0, {}, -1, 4, 0), {53, 52, 51, 50}), "pick: unknown mask = the node top");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(0, 71), -1, 4, 0), {53, 52, 51, 50}), "pick: whole-box mask = the node top");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(36, 47), 47, 4, 0), {46, 45, 44, 43}), "pick: a pinned driver keeps its cpu");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(44, 47), 47, 4, 0), {47, 46, 45, 44}), "pick: no spare cpu -> driver's cpu shared");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(18, 35), -1, 4, 0), {53, 52, 51, 50}), "pick: mask off the node -> node top");
        CHECK(eq(pxa_tp_pick_cpus(node0, range(36, 37), -1, 4, 0), {37, 36}), "pick: 2 allowed cpus for 4 workers -> 2 pinned");
    }
    if (fails) { printf("test-pxa-tp-adapt: %d FAILED\n", fails); return 1; }
    printf("test-pxa-tp-adapt: OK\n");
    return 0;
}
