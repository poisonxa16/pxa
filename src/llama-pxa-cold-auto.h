// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// Cold path auto choice for the Nemotron-H expert split (nemotron_h_moe, non-PXQN routed experts, file larger than the card):
// the experts the card has no room for stay in pinned host RAM and are either computed by the CPU (cold path = cpu: one
// GPU -> CPU -> GPU hand-off per cold layer, the CPU reads only the routed experts from DRAM) or read in place by the GPU over the
// card's link (cold path = zero-copy). On a free CPU the CPU path is about twice as fast (Nano 30B PXQ4, one V100 on PCIe x4:
// 82-92 vs 45 t/s prose, 78-89 vs 38 code); on CPUs other work keeps busy the hand-offs wait for descheduled threads and the CPU
// path collapses (1.8 t/s at box load 36-40 on the cores it ran on), where zero-copy does not care. So the choice is made at load
// from what the CPUs this process may run on are doing:
//
//   PXA_XCACHE_NEMO_ZC=1  zero-copy (the documented off switch: the pre-auto default, bit-identical to it)
//   PXA_XCACHE_NEMO_ZC=0  cpu
//   unset                 auto: sample /proc/stat over a short window (PXA_XCACHE_NEMO_SAMPLE_MS, default 300) on this process's
//                         CPU set; CPUs busy with OTHER processes' work (this process's own time is subtracted) are not free,
//                         niced work counts PXA_XCACHE_NEMO_NICE_W of a CPU (default 1); cpu when the free CPUs cover
//                         PXA_XCACHE_NEMO_CPU_NEED (default 0.9) per cold-path thread (-t, handed over by the front end with
//                         the model load, llama_model_params::pxa_cold_threads; an exported PXA_XCACHE_COLD_THREADS wins;
//                         unknown = every CPU of the set), else zero-copy. More
//                         threads than CPUs in the set -> zero-copy. No /proc/stat (non-Linux) -> cpu when the threads fit the set.
//                         A zero-copy verdict is CONFIRMED (bug #347, 2026-10-22): the choice holds for the whole run, so when the
//                         first window says zero-copy the sampler reads PXA_XCACHE_NEMO_CONFIRM - 1 more windows (default 5 in
//                         all), PXA_XCACHE_NEMO_CONFIRM_GAP_MS (default 700) apart, and decides on the MEDIAN free CPUs. A burst of
//                         other work at the moment of the load no longer locks a boot into the slower path (Bolt-74B PXQ4, 2x P100
//                         x4: one window saw 7.4 of 18 CPUs free while every other boot of the bracket saw 15.5-17.9 -> zero-copy
//                         ~10 t/s and its own greedy bytes vs 22-24 t/s on the CPU path); load that persists over the windows still
//                         picks zero-copy. A first window that says cpu decides at once, as before (no extra time, same line).
//                         PXA_XCACHE_NEMO_CONFIRM=0 (or 1) = the single-window rule (the off switch).
//
// Runtime guard (PXA_XCACHE_NEMO_GUARD=1, OPT-IN since 2026-10-19; only after an auto cpu choice): load that starts after the load would
// still collapse the CPU path, so once a second (PXA_XCACHE_NEMO_GUARD_MS) the decode thread reads how long this process's own
// threads waited for a CPU (/proc/self/task/*/schedstat). While on the CPU path, more than PXA_XCACHE_NEMO_GUARD_WAIT (default 1.0)
// CPUs' worth of waiting for 2 intervals in a row moves the narrow graphs to the zero-copy read of the same pinned stacks; while on
// zero-copy, 5 intervals in a row with the free CPUs at 0.75 of the CPU path's need move them back as a trial (a trial that
// falls back within 5 intervals doubles the next wait, up to 64). Each move is one log line. A
// moved run's output then follows the path each token took (both correct; not the forced runs' bytes). That is why the guard is
// opt-in: the two paths round differently, so with the guard a greedy request's text depended on the load during it (Bolt-74B PXQ4
// on 2x P100, 4.0 RC: 2-5 different greedy shas over 6 reps of one prompt). Off, a boot keeps the path it chose at load.
//
// The choice only picks which existing path runs (the same two plans PXA_XCACHE_NEMO_ZC=0 / unset produced before): a run's output
// equals the forced run of the path it chose. This header is the pure layer, unit-tested in tests/test-pxa-cold-auto.cpp; the
// sampler (/proc/stat, /proc/self/stat, sched_getaffinity) is llama-pxa-cold-auto.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct pxa_cpu_ticks {
    uint64_t busy = 0;   // user + system + irq + softirq + steal (jiffies)
    uint64_t nice = 0;   // niced user time
    uint64_t idle = 0;   // idle + iowait
    bool     seen = false;
};

// "/proc/stat" text -> per-CPU ticks indexed by CPU number (the aggregate "cpu " line is skipped)
std::vector<pxa_cpu_ticks> pxa_cold_parse_proc_stat(const std::string & text);

struct pxa_cold_load {
    bool   sampled    = false;   // two readings of every CPU of the set
    int    set_cpus   = 0;       // CPUs this process may run on (0 = unknown)
    double other_busy = 0;       // CPUs (in CPU units over the window) busy with other processes' un-niced work
    double other_nice = 0;       // CPUs busy with niced work (this process's own time subtracted first from busy, then nice)
    double idle       = 0;       // CPUs idle (idle + iowait)
};

// a, b = two readings of /proc/stat; set = the CPUs this process may run on; own_ticks = this process's utime+stime between them
pxa_cold_load pxa_cold_load_from(const std::vector<pxa_cpu_ticks> & a, const std::vector<pxa_cpu_ticks> & b,
        const std::vector<int> & set, double own_ticks);

struct pxa_cold_auto_in {
    int    forced      = -1;    // PXA_XCACHE_NEMO_ZC: -1 unset (auto), 0 cpu, 1 zero-copy
    int    threads     = 0;     // CPU threads the cold path runs on (-t); <= 0 = unknown (every CPU of the set)
    pxa_cold_load load;
    double nice_weight = 1.0;   // share of a CPU that niced work takes from the cold path
    double need        = 0.9;   // free CPUs needed per cold-path thread
};

struct pxa_cold_auto_out {
    bool        zc     = false;
    bool        is_auto = false;   // the choice was made from the load (not forced)
    double      avail  = 0;     // free CPUs the decision saw
    double      needed = 0;     // free CPUs the CPU path needs
    std::string reason;
};

pxa_cold_auto_out pxa_cold_auto_decide(const pxa_cold_auto_in & in);

// true when `o` (= pxa_cold_auto_decide(in)) is a zero-copy verdict made from a sampled load that more windows may overturn
// (not forced, not the oversubscribed set, not an unreadable load)
bool pxa_cold_auto_confirmable(const pxa_cold_auto_in & in, const pxa_cold_auto_out & o);

// the confirmed choice: in.load = the first window, more = the later windows (unsampled ones are skipped). When the first
// verdict is confirmable and some later window was read, the free CPUs are the median of all the windows (the upper middle
// of an even count) and the reason names every window; otherwise the first verdict, unchanged.
pxa_cold_auto_out pxa_cold_auto_confirm(const pxa_cold_auto_in & in, const std::vector<pxa_cold_load> & more);

// the runtime: this process's CPU set, two /proc/stat readings window_ms apart, own time subtracted (not sampled on failure)
pxa_cold_load pxa_cold_sample_load(int window_ms);

// the whole auto choice: threads = this load's cold-path threads (llama_model_params::pxa_cold_threads, <= 0 = unknown) unless
// PXA_XCACHE_COLD_THREADS is exported; the rest from the environment (PXA_XCACHE_NEMO_ZC, PXA_XCACHE_NEMO_SAMPLE_MS,
// PXA_XCACHE_NEMO_NICE_W, PXA_XCACHE_NEMO_CPU_NEED, PXA_XCACHE_NEMO_CONFIRM, PXA_XCACHE_NEMO_CONFIRM_GAP_MS); samples only when
// the choice is auto, and more than one window only to confirm a zero-copy verdict. Reads the environment, never writes it.
pxa_cold_auto_out pxa_cold_auto_from_env(int threads = 0);

// ---- runtime guard
struct pxa_cold_guard_cfg {
    double needed      = 0;     // free CPUs the CPU path needs (pxa_cold_auto_out::needed)
    double wait_down   = 1.0;   // CPU path -> zero-copy when this process's threads waited more than this many CPUs' worth ...
    int    n_down      = 2;     // ... for this many intervals in a row
    int    n_up        = 5;     // zero-copy -> CPU path (a trial) after this many intervals in a row with avail >= up_frac x needed
    double up_frac     = 0.75;  // (the reading on zero-copy undercounts nothing the CPU path would fight for; the trial confirms it)
    int    n_trial     = 5;     // a move back down within this many intervals of a trial doubles the next wait (n_up .. max_up)
    int    max_up      = 64;
    int    n_settle    = 30;    // this many intervals on the CPU path without a move down reset the wait to n_up
    double nice_weight = 1.0;
};
struct pxa_cold_guard_state {
    bool zc      = false;
    int  below   = 0;
    int  above   = 0;
    int  up_need = 0;   // intervals the next move back waits for (0 = n_up)
    int  since   = 0;   // intervals since the last move
    int  moves   = 0;
};
// one interval: wait = CPUs' worth of this process's run-queue wait, avail = free CPUs of the set; true when the path moves.
// CPU path: n_down intervals of wait > wait_down -> zero-copy. Zero-copy: up_need intervals of avail >= up_frac x needed -> a
// trial on the CPU path; a trial that moves down again within n_trial intervals doubles up_need (to max_up), n_settle quiet
// intervals on the CPU path reset it.
bool pxa_cold_guard_step(pxa_cold_guard_state & s, const pxa_cold_guard_cfg & c, double wait, double avail);

struct pxa_cold_guard_rt;   // the readings between polls
// PXA_XCACHE_NEMO_GUARD set and non-zero (default off: deterministic output)
bool pxa_cold_guard_wanted(void);
// a guard for a load whose auto choice was cpu (nullptr unless PXA_XCACHE_NEMO_GUARD=1, or when nothing can be read)
pxa_cold_guard_rt * pxa_cold_guard_new(const pxa_cold_auto_out & choice);
void pxa_cold_guard_free(pxa_cold_guard_rt * g);
// called once per decode (cheap until an interval has passed): +1 moved to zero-copy, -1 moved back to cpu, 0 unchanged,
// 2 = no move, but a reading to log (every 10th interval while on zero-copy)
int pxa_cold_guard_poll(pxa_cold_guard_rt * g, std::string & why);
