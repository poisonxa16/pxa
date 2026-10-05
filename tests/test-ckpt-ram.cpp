// PXA_CKPT_RAM_v1 sizing test.
//
// A model with recurrent state keeps its context checkpoints and its RAM prompt cache in HOST RAM.
// The rule under test (common/pxa-ckpt-ram.h) sizes both from the free RAM, and ONLY when the user
// did not pass the flag. What is checked:
//   * a big machine keeps today's behaviour exactly: 32 checkpoints per slot, 8192 MiB of cache;
//   * the 8 GB machine of the Discord report gets a number that fits, on the real 27B-class state
//     size (100 MiB), and the total never exceeds 15% of the free RAM;
//   * the count falls monotonically with fewer free bytes, more slots and a bigger checkpoint;
//   * one checkpoint per slot survives while it takes at most half of the free RAM, else 0;
//   * an explicit --ctx-checkpoints / --cache-ram is returned untouched, for any RAM and any value,
//     including 0, -1 and a number far above what the machine holds;
//   * unknown free RAM (-1, not Linux) leaves the defaults;
//   * the readers: /proc/meminfo, cgroup memory.max / memory.current / memory.stat (v2 and v1's
//     "unlimited"), and the PXA_MEM_AVAILABLE_MIB test hook;
//   * the truth lines say what was decided.
//
// CPU only, no model, no server, no GPU. Exit 0 on pass.

#include "pxa-ckpt-ram.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static constexpr int64_t MiB = 1024ll * 1024;
static constexpr int64_t GiB = 1024ll * MiB;

int main() {
    const int64_t state_27b = 100 * MiB;                        // measured 60-130 MiB for the 27B class
    const int64_t est       = pxa_ckpt_ram_est_bytes(state_27b, 262144);
    check(est == state_27b + 8 * 262144, "estimate = state + 8 bytes per cell of the slot context");
    check(pxa_ckpt_ram_est_bytes(0, 4096) == 128 * MiB + 8 * 4096, "an unreadable probe falls back to 128 MiB");
    check(pxa_ckpt_ram_est_bytes(200, 4096) == 128 * MiB + 8 * 4096, "a header-only probe falls back to 128 MiB");
    check(pxa_ckpt_ram_est_bytes(200, 4096, false) == 200 + 8 * 4096, "a measured checkpoint is taken as it is");

    // ---- a big machine keeps today's behaviour ---------------------------------------------------
    check(pxa_ckpt_ram_plan_per_slot(64 * GiB, 1, est, 32) == 32, "64 GiB free, 1 slot -> 32");
    check(pxa_ckpt_ram_plan_per_slot(128 * GiB, 4, est, 32) == 32, "128 GiB free, 4 slots -> 32");
    check(pxa_ckpt_ram_plan_per_slot(64 * GiB, 4, est, 32) == 24, "64 GiB free, 4 slots -> 24 (32 x 4 x 102 MiB is over 15%)");
    check(pxa_ckpt_ram_plan_per_slot(32 * GiB, 1, 130 * MiB, 32) == 32, "32 GiB free, 130 MiB state -> 32");
    check(pxa_cache_ram_plan_mib(64 * GiB, 8192) == 8192, "64 GiB free -> cache 8192 MiB");
    check(pxa_cache_ram_plan_mib(54614 * MiB, 8192) == 8192, "cache hits its default at ~53.3 GiB free");
    check(pxa_cache_ram_plan_mib(54613 * MiB, 8192) == 8191, "and is one MiB short just under it");

    // ---- the 8 GB machine of the report -----------------------------------------------------------
    {
        const int64_t avail = 7987 * MiB; // 7.8 GiB free
        const int n = pxa_ckpt_ram_plan_per_slot(avail, 1, est, 32);
        check(n >= 1 && n < 32, "7.8 GiB free, 1 slot -> a bounded count");
        check((int64_t) n * est <= avail / 100 * 15 + 1, "the total is within 15% of the free RAM");
        check((int64_t) (n + 1) * est > avail / 100 * 15, "and it is the largest count that is");
        printf("7.8 GiB free, 1 slot, 100 MiB state -> %d per slot\n", n);
        const int c = pxa_cache_ram_plan_mib(avail, 8192);
        check(c == (int) ((avail / 100 * 15 + (avail % 100) * 15 / 100) >> 20) && c < 8192, "cache capped to 15% of free RAM");
        printf("7.8 GiB free -> cache %d MiB\n", c);
    }

    // ---- monotone in every input -------------------------------------------------------------------
    {
        int prev = 1 << 30;
        for (int64_t a = 32 * GiB; a >= 0; a -= 256 * MiB) {          // fewer bytes -> never more
            const int n = pxa_ckpt_ram_plan_per_slot(a, 1, est, 32);
            check(n <= prev, "count never rises as free RAM falls");
            prev = n;
        }
        prev = 1 << 30;
        for (int s = 1; s <= 16; s++) {                               // more slots -> never more per slot
            const int n = pxa_ckpt_ram_plan_per_slot(8 * GiB, s, est, 32);
            check(n <= prev, "count never rises with more slots");
            check((int64_t) n * s * est <= 8 * GiB / 100 * 15 || n <= 1, "all slots together stay in the share");
            prev = n;
        }
        prev = 1 << 30;
        for (int64_t e = 1 * MiB; e <= 400 * MiB; e += 5 * MiB) {     // bigger checkpoint -> never more
            const int n = pxa_ckpt_ram_plan_per_slot(8 * GiB, 1, e, 32);
            check(n <= prev, "count never rises with a bigger checkpoint");
            prev = n;
        }
        int pc = 1 << 30;
        for (int64_t a = 16 * GiB; a >= 0; a -= 64 * MiB) {
            const int c = pxa_cache_ram_plan_mib(a, 8192);
            check(c <= pc || c == 0, "cache never rises as free RAM falls");
            pc = c;
        }
    }

    // ---- the floor: one per slot while it takes at most half of the free RAM, else off -----------------
    check(pxa_ckpt_ram_plan_per_slot(500 * MiB, 1, 100 * MiB, 32) == 1, "500 MiB free, 100 MiB state -> floor of 1 (15% = 75 MiB fits none)");
    check(pxa_ckpt_ram_plan_per_slot(400 * MiB, 1, 100 * MiB, 32) == 1, "400 MiB free, 100 MiB state -> floor of 1 (15% fits none)");
    check(pxa_ckpt_ram_plan_per_slot(200 * MiB, 1, 100 * MiB, 32) == 1, "200 MiB free: one checkpoint is exactly half -> 1");
    check(pxa_ckpt_ram_plan_per_slot(199 * MiB, 1, 100 * MiB, 32) == 0, "199 MiB free: one checkpoint is over half -> 0");
    check(pxa_ckpt_ram_plan_per_slot(0, 1, 100 * MiB, 32) == 0, "no free RAM -> 0");
    check(pxa_ckpt_ram_plan_per_slot(1 * GiB, 4, 100 * MiB, 32) == 1, "1 GiB free, 4 slots -> 1 each (400 MiB of 1 GiB)");
    check(pxa_ckpt_ram_plan_per_slot(700 * MiB, 4, 100 * MiB, 32) == 0, "700 MiB free, 4 slots -> 0 (400 MiB is over half)");
    check(pxa_cache_ram_plan_mib(500 * MiB, 8192) == 0, "500 MiB free -> 15% is 75 MiB, under 128 -> cache off");
    check(pxa_cache_ram_plan_mib(1 * GiB, 8192) == 153, "1 GiB free -> cache 153 MiB");
    check(pxa_cache_ram_plan_mib(100 * GiB, 4096) == 4096, "a smaller requested cache is never raised");

    // ---- the engine's own ceiling is never exceeded ---------------------------------------------------------
    check(pxa_ckpt_ram_plan_per_slot(1024 * GiB, 1, 1 * MiB, 8) == 8, "a ceiling of 8 is respected");
    check(pxa_ckpt_ram_plan_per_slot(1024 * GiB, 1, 1 * MiB, 0) == 0, "a ceiling of 0 stays 0");

    // ---- unknown free RAM keeps the defaults ----------------------------------------------------------------
    check(pxa_ckpt_ram_plan_per_slot(-1, 1, est, 32) == 32, "unknown free RAM -> 32");
    check(pxa_cache_ram_plan_mib(-1, 8192) == 8192, "unknown free RAM -> 8192");
    check(pxa_ckpt_ram_plan_per_slot(8 * GiB, 1, 0, 32) == 32, "unknown checkpoint size -> 32");

    // ---- an explicit flag always wins ------------------------------------------------------------------------
    {
        const int64_t avails[] = { -1, 0, 64 * MiB, 1 * GiB, 7987 * MiB, 64 * GiB };
        const int reqs[]       = { -1, 0, 1, 2, 32, 1000 };
        for (int64_t a : avails) {
            for (int r : reqs) {
                check(pxa_ckpt_ram_decide_per_slot(true, r, a, 1, est, 32) == r, "explicit --ctx-checkpoints is untouched");
                check(pxa_ckpt_ram_decide_per_slot(true, r, a, 8, 400 * MiB, 32) == r, "explicit --ctx-checkpoints is untouched (8 slots, big state)");
                check(pxa_cache_ram_decide_mib(true, r, a) == r, "explicit --cache-ram is untouched");
            }
            check(pxa_cache_ram_decide_mib(true, 8192, a) == 8192, "explicit --cache-ram 8192 on a small machine is untouched");
            check(pxa_cache_ram_decide_mib(false, 8192, a) == pxa_cache_ram_plan_mib(a, 8192), "defaulted cache is planned");
        }
        check(pxa_ckpt_ram_decide_per_slot(false, 32, 7987 * MiB, 1, est, 32) == pxa_ckpt_ram_plan_per_slot(7987 * MiB, 1, est, 32),
               "a defaulted --ctx-checkpoints is planned");
    }

    // ---- readers ---------------------------------------------------------------------------------------------
    {
        const std::string mi =
            "MemTotal:       16337540 kB\nMemFree:          412345 kB\nMemAvailable:    7987200 kB\nBuffers:  1 kB\n";
        check(pxa_ckpt_ram_parse_meminfo(mi) == 7987200ll * 1024, "meminfo MemAvailable in kB -> bytes");
        check(pxa_ckpt_ram_parse_meminfo(mi, "MemTotal") == 16337540ll * 1024, "meminfo MemTotal");
        check(pxa_ckpt_ram_parse_meminfo(mi, "Nope") == -1, "meminfo missing key -> -1");
        check(pxa_ckpt_ram_parse_meminfo("") == -1, "meminfo empty -> -1");
        check(pxa_ckpt_ram_parse_meminfo("MemAvailable: 12\n") == 12, "meminfo without a unit is bytes");
        check(pxa_ckpt_ram_parse_meminfo("MemAvailable: abc kB\n") == -1, "meminfo garbage -> -1");
        check(pxa_ckpt_ram_parse_stat("anon 100\ninactive_file 4096\nfile 9\n", "inactive_file") == 4096, "memory.stat key");
        check(pxa_ckpt_ram_parse_stat("anon 100\n", "inactive_file") == -1, "memory.stat missing key");
        check(pxa_ckpt_ram_stat_reclaimable("anon 5\ninactive_file 100\nactive_file 20\n", false) == 120, "reclaimable = inactive + active file (v2)");
        check(pxa_ckpt_ram_stat_reclaimable("total_inactive_file 7\ntotal_active_file 3\ninactive_file 99\n", true) == 10, "reclaimable (v1 total_ keys)");
        check(pxa_ckpt_ram_stat_reclaimable("anon 5\n", false) == 0, "reclaimable absent -> 0");

        const std::string g8 = std::to_string(8 * GiB);
        check(pxa_ckpt_ram_cgroup_headroom(g8 + "\n", std::to_string(3 * GiB) + "\n", 0) == 5 * GiB, "limit 8 GiB, used 3 GiB -> 5 GiB");
        check(pxa_ckpt_ram_cgroup_headroom(g8, std::to_string(3 * GiB), 1 * GiB) == 6 * GiB, "reclaimable file cache is not counted as used");
        check(pxa_ckpt_ram_cgroup_headroom(g8, std::to_string(3 * GiB), 9 * GiB) == 8 * GiB, "reclaimable above usage clamps to the whole limit");
        check(pxa_ckpt_ram_cgroup_headroom(g8, std::to_string(9 * GiB), 0) == 0, "over the limit -> 0 headroom");
        check(pxa_ckpt_ram_cgroup_headroom("max\n", "123", 0) == -1, "memory.max = max -> unlimited");
        check(pxa_ckpt_ram_cgroup_headroom("9223372036854771712", "123", 0) == -1, "cgroup v1 unlimited -> unlimited");
        check(pxa_ckpt_ram_cgroup_headroom("", "123", 0) == -1, "absent files -> unlimited");
        check(pxa_ckpt_ram_cgroup_headroom(g8, "", 0) == -1, "absent usage -> unlimited");
    }
    {
        setenv("PXA_MEM_AVAILABLE_MIB", "4096", 1);
        std::string why;
        check(pxa_ckpt_ram_available_bytes(&why) == 4096 * MiB && why == "PXA_MEM_AVAILABLE_MIB", "PXA_MEM_AVAILABLE_MIB test hook");
        setenv("PXA_MEM_AVAILABLE_MIB", "junk", 1);
        const int64_t live = pxa_ckpt_ram_available_bytes(&why);
        check(live != 4096 * MiB, "a junk hook value is ignored");
#if defined(__linux__)
        check(live > 0, "the live reading works on Linux");
        printf("live free RAM here: %.1f GiB (%s)\n", (double) live / (double) GiB, why.c_str());
#endif
        unsetenv("PXA_MEM_AVAILABLE_MIB");
    }

    // ---- the truth lines -------------------------------------------------------------------------------------------
    {
        auto has = [](const std::string & s, const char * sub) { return s.find(sub) != std::string::npos; };
        const int64_t avail = 7987 * MiB;
        const int n = pxa_ckpt_ram_plan_per_slot(avail, 1, est, 32);
        const std::string l = pxa_ckpt_ram_describe(n, 1, est, avail, 32, false);
        printf("%s\n", l.c_str());
        check(has(l, "context checkpoints: ") && has(l, " per slot (") && has(l, "7.8 GiB free RAM") &&
              has(l, "--ctx-checkpoints N to override"), "auto line names the count, the free RAM and the override");
        check(!has(l, "up to"), "a limited plan does not say 'up to'");
        const std::string big = pxa_ckpt_ram_describe(32, 1, est, 64 * GiB, 32, false);
        printf("%s\n", big.c_str());
        check(has(big, "32 per slot") && has(big, "up to ~"), "an unconstrained plan says 'up to'");
        const std::string multi = pxa_ckpt_ram_describe(3, 4, est, avail, 32, false);
        check(has(multi, "x 4 slots"), "multi-slot line names the slot count");
        const std::string off = pxa_ckpt_ram_describe(0, 1, est, 150 * MiB, 32, false);
        printf("%s\n", off.c_str());
        check(has(off, "OFF") && has(off, "--ctx-checkpoints N to force"), "0 plan warns and names the override");
        const std::string ex = pxa_ckpt_ram_describe(32, 1, est, 4 * GiB, 32, true);
        printf("%s\n", ex.c_str());
        check(has(ex, "set by --ctx-checkpoints") && has(ex, "expect swapping"), "explicit line on a small box warns about swapping");
        check(!has(pxa_ckpt_ram_describe(2, 1, est, 4 * GiB, 32, true), "swapping"), "explicit 2 is quiet");
        check(has(pxa_ckpt_ram_describe(0, 1, est, 4 * GiB, 32, true), "off (--ctx-checkpoints 0)"), "explicit 0 line");
        check(has(pxa_ckpt_ram_describe(7, 1, est, -1, 32, false), "cannot be read"), "unknown RAM line");
        const std::string c = pxa_cache_ram_describe(1170, 8192, avail);
        printf("%s\n", c.c_str());
        check(has(c, "1170 MiB") && has(c, "8192 MiB") && has(c, "--cache-ram N"), "cache line");
        check(has(pxa_cache_ram_describe(0, 8192, 500 * MiB), "OFF"), "cache off line");
    }

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    printf("test-ckpt-ram: all checks passed\n");
    return 0;
}
