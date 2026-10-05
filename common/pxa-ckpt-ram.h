#pragma once

// PXA_CKPT_RAM_v1: size the host-RAM consumers of a recurrent / hybrid model from the host.
//
// For a model with recurrent state (hybrid Gated-DeltaNet: Qwen3.8, Flash-Next, ...) the server
// keeps context checkpoints in HOST RAM: one std::vector of the recurrent state (60-130 MiB for
// the 27B class) every --ctx-checkpoints-interval tokens, up to --ctx-checkpoints (32) per slot.
// At long context that is 2-4 GB per slot -- invisible on a 64 GB box, fatal on an 8 GB one,
// where the machine starts swapping while the VRAM is still free (Discord report 2026-10-04,
// Swift-1.5-Qwen3.8-27B OneCard on 8 GB of system RAM). The RAM prompt cache (--cache-ram,
// default 8192 MiB) is the second consumer.
//
// This header is the sizing rule and nothing else. Both numbers are decided ONLY when the user
// did not pass the flag (an explicit --ctx-checkpoints / --cache-ram always wins, untouched):
//
//   checkpoints per slot = min(32, floor(15% of free RAM / (n_slots * checkpoint bytes)))
//        never below 1 while one checkpoint per slot takes at most half of the free RAM, else 0
//   RAM prompt cache     = min(8192 MiB, 15% of free RAM), or 0 when that is under 128 MiB
//
// "Free RAM" is the smaller of /proc/meminfo MemAvailable and the headroom of the process's
// cgroup memory limit (docker --memory, systemd MemoryMax): MemAvailable inside a container still
// shows the whole host, which is exactly the machine that is NOT the problem. A big machine
// resolves to today's 32 / 8192; where free RAM cannot be read (not Linux) the defaults stand.
//
// Checkpoints and the prompt cache only decide what has to be re-prefilled. They never change a
// computed value, so no setting this header can produce moves a logit or a token.
//
// Pure: the planning functions take numbers, the readers take text, so the whole rule is
// unit-tested without a model or a server (tests/test-ckpt-ram.cpp). Only
// pxa_ckpt_ram_available_bytes() touches the machine.
//
// PXA_MEM_AVAILABLE_MIB=<n> overrides the reading (a test hook: it lets a bench pretend to be a
// small machine without a container).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

// share of the free RAM that checkpoints (and, separately, the RAM prompt cache) may take
static constexpr int      PXA_CKPT_RAM_SHARE_PCT   = 15;
// a RAM prompt cache smaller than this cannot hold one useful entry; it is switched off instead
static constexpr int      PXA_CACHE_RAM_MIN_MIB    = 128;
// one checkpoint per slot is kept as long as it takes no more than 1/N of the free RAM
static constexpr int      PXA_CKPT_RAM_FLOOR_DIV   = 2;
// per-cell metadata a checkpoint also carries (position + seq count, see write_kv_cache_meta)
static constexpr int64_t  PXA_CKPT_RAM_CELL_BYTES  = 8;

// ---- readers (text in, bytes out; -1 = absent / unparseable / unlimited) ------------------------

// "MemAvailable:    7981236 kB" -> bytes
static inline int64_t pxa_ckpt_ram_parse_meminfo(const std::string & text, const char * key = "MemAvailable") {
    const std::string k = std::string(key) + ":";
    size_t p = 0;
    while (p < text.size()) {
        size_t e = text.find('\n', p);
        if (e == std::string::npos) {
            e = text.size();
        }
        if (text.compare(p, k.size(), k) == 0) {
            const char * s = text.c_str() + p + k.size();
            char * end = nullptr;
            const long long v = strtoll(s, &end, 10);
            if (end == s || v < 0) {
                return -1;
            }
            while (*end == ' ' || *end == '\t') {
                ++end;
            }
            int64_t mul = 1;
            if (strncmp(end, "kB", 2) == 0) {
                mul = 1024;
            }
            return (int64_t) v * mul;
        }
        p = e + 1;
    }
    return -1;
}

// "key 12345" lines of a cgroup memory.stat
static inline int64_t pxa_ckpt_ram_parse_stat(const std::string & text, const char * key) {
    const std::string k = std::string(key) + " ";
    size_t p = 0;
    while (p < text.size()) {
        size_t e = text.find('\n', p);
        if (e == std::string::npos) {
            e = text.size();
        }
        if (text.compare(p, k.size(), k) == 0) {
            const char * s = text.c_str() + p + k.size();
            char * end = nullptr;
            const long long v = strtoll(s, &end, 10);
            return (end == s || v < 0) ? -1 : (int64_t) v;
        }
        p = e + 1;
    }
    return -1;
}

// headroom of ONE cgroup: limit - (usage - reclaimable page cache). "max" / absent / a limit of
// 2^60 or more (cgroup v1's "unlimited") -> -1. The reclaimable term is the file cache (active +
// inactive): the kernel gives it back under pressure, and /proc/meminfo's MemAvailable counts it
// as available for the same reason, so a container that merely read a big model file is not
// reported as full and the two readings mean the same thing.
static inline int64_t pxa_ckpt_ram_cgroup_headroom(const std::string & limit_s, const std::string & usage_s,
                                                   int64_t reclaimable) {
    if (limit_s.empty() || usage_s.empty()) {
        return -1;
    }
    if (limit_s.compare(0, 3, "max") == 0) {
        return -1;
    }
    char * e1 = nullptr;
    char * e2 = nullptr;
    const long long limit = strtoll(limit_s.c_str(), &e1, 10);
    const long long usage = strtoll(usage_s.c_str(), &e2, 10);
    if (e1 == limit_s.c_str() || e2 == usage_s.c_str() || limit <= 0 || limit >= (1LL << 60) || usage < 0) {
        return -1;
    }
    const long long used = std::max<long long>(0, usage - std::max<int64_t>(0, reclaimable));
    return (int64_t) std::max<long long>(0, limit - used);
}

// active + inactive file cache of a memory.stat (v2 keys, or v1's total_ keys)
static inline int64_t pxa_ckpt_ram_stat_reclaimable(const std::string & stat, bool v1) {
    const int64_t inact = pxa_ckpt_ram_parse_stat(stat, v1 ? "total_inactive_file" : "inactive_file");
    const int64_t act   = pxa_ckpt_ram_parse_stat(stat, v1 ? "total_active_file"   : "active_file");
    return std::max<int64_t>(0, inact) + std::max<int64_t>(0, act);
}

static inline std::string pxa_ckpt_ram_read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        return std::string();
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The smallest headroom over this process's cgroup and every ancestor (cgroup v2), or the v1 memory
// controller's. -1 when no memory limit applies.
static inline int64_t pxa_ckpt_ram_cgroup_available_bytes() {
    int64_t best = -1;
    auto take = [&](int64_t v) {
        if (v >= 0 && (best < 0 || v < best)) {
            best = v;
        }
    };

    // cgroup v2: "0::/path" in /proc/self/cgroup; a container with its own cgroup namespace sees
    // "/" there and its own limits at the mount root.
    std::string rel = "/";
    {
        std::stringstream ss(pxa_ckpt_ram_read_file("/proc/self/cgroup"));
        std::string line;
        while (std::getline(ss, line)) {
            if (line.compare(0, 3, "0::") == 0) {
                rel = line.substr(3);
                break;
            }
        }
    }
    const std::string root = "/sys/fs/cgroup";
    std::string dir = rel;
    while (true) {
        const std::string base = root + (dir == "/" ? std::string() : dir);
        take(pxa_ckpt_ram_cgroup_headroom(
            pxa_ckpt_ram_read_file(base + "/memory.max"),
            pxa_ckpt_ram_read_file(base + "/memory.current"),
            pxa_ckpt_ram_stat_reclaimable(pxa_ckpt_ram_read_file(base + "/memory.stat"), false)));
        if (dir == "/" || dir.empty()) {
            break;
        }
        const size_t s = dir.find_last_of('/');
        dir = (s == std::string::npos || s == 0) ? std::string("/") : dir.substr(0, s);
    }

    // cgroup v1: the memory controller mounted at the container's root
    take(pxa_ckpt_ram_cgroup_headroom(
        pxa_ckpt_ram_read_file(root + "/memory/memory.limit_in_bytes"),
        pxa_ckpt_ram_read_file(root + "/memory/memory.usage_in_bytes"),
        pxa_ckpt_ram_stat_reclaimable(pxa_ckpt_ram_read_file(root + "/memory/memory.stat"), true)));
    return best;
}

// Free host RAM in bytes, or -1 when it cannot be read (the caller then keeps its defaults).
// `why` (optional) names which reading won, for the log line.
static inline int64_t pxa_ckpt_ram_available_bytes(std::string * why = nullptr) {
    if (const char * e = getenv("PXA_MEM_AVAILABLE_MIB")) {
        const long long mib = atoll(e);
        if (mib > 0) {
            if (why) {
                *why = "PXA_MEM_AVAILABLE_MIB";
            }
            return (int64_t) mib * 1024 * 1024;
        }
    }
#if defined(__linux__)
    const int64_t host = pxa_ckpt_ram_parse_meminfo(pxa_ckpt_ram_read_file("/proc/meminfo"));
    const int64_t cg   = pxa_ckpt_ram_cgroup_available_bytes();
    if (host < 0 && cg < 0) {
        return -1;
    }
    if (cg >= 0 && (host < 0 || cg < host)) {
        if (why) {
            *why = "container memory limit";
        }
        return cg;
    }
    if (why) {
        *why = "MemAvailable";
    }
    return host;
#else
    return -1;
#endif
}

// ---- the sizing rule ----------------------------------------------------------------------------

// Bytes one checkpoint of this model costs, from the state size the engine reports for a
// sequence. Adds the per-cell metadata a checkpoint grows by as the context fills.
static inline int64_t pxa_ckpt_ram_est_bytes(int64_t state_bytes, int64_t n_ctx_per_slot, bool probe = true) {
    // `probe`: the size came from asking the engine about a sequence BEFORE any checkpoint exists.
    // An unreadable or implausibly small answer (an empty sequence reports little more than its
    // header) falls back to the top of the measured range for the 27B class (130 MiB); the first
    // real checkpoint then re-plans with its true size (probe = false: taken as it is).
    constexpr int64_t FALLBACK  = 128ll * 1024 * 1024;
    constexpr int64_t TOO_SMALL = 64ll * 1024;
    if (probe && state_bytes < TOO_SMALL) {
        state_bytes = FALLBACK;
    }
    return std::max<int64_t>(1, state_bytes) + PXA_CKPT_RAM_CELL_BYTES * std::max<int64_t>(0, n_ctx_per_slot);
}

// Checkpoints per slot. `max_per_slot` is the engine's own default (32); a result never exceeds it.
// avail < 0 (unknown) or est <= 0 returns max_per_slot unchanged.
static inline int pxa_ckpt_ram_plan_per_slot(int64_t avail, int n_slots, int64_t est, int max_per_slot) {
    if (max_per_slot <= 0 || avail < 0 || est <= 0) {
        return max_per_slot;
    }
    const int64_t slots    = std::max(1, n_slots);
    const int64_t one_each = slots * est;
    const int64_t budget   = avail / 100 * PXA_CKPT_RAM_SHARE_PCT + (avail % 100) * PXA_CKPT_RAM_SHARE_PCT / 100;
    const int64_t fit      = budget / one_each;
    if (fit >= max_per_slot) {
        return max_per_slot;
    }
    if (fit >= 1) {
        return (int) fit;
    }
    return (one_each <= avail / PXA_CKPT_RAM_FLOOR_DIV) ? 1 : 0;
}

// RAM prompt cache size in MiB. want_mib <= 0 (off / no-limit) is the caller's choice and returned
// as is. Returns 0 when 15% of the free RAM is under PXA_CACHE_RAM_MIN_MIB.
static inline int pxa_cache_ram_plan_mib(int64_t avail, int want_mib) {
    if (want_mib <= 0 || avail < 0) {
        return want_mib;
    }
    const int64_t budget = avail / 100 * PXA_CKPT_RAM_SHARE_PCT + (avail % 100) * PXA_CKPT_RAM_SHARE_PCT / 100;
    const int64_t cap    = budget >> 20;
    if (cap >= want_mib) {
        return want_mib;
    }
    return cap < PXA_CACHE_RAM_MIN_MIB ? 0 : (int) cap;
}

// The decision itself, with the one rule that matters most: a value the user passed is never touched.
static inline int pxa_ckpt_ram_decide_per_slot(bool explicit_flag, int requested, int64_t avail, int n_slots,
                                               int64_t est, int max_per_slot) {
    return explicit_flag ? requested : pxa_ckpt_ram_plan_per_slot(avail, n_slots, est, max_per_slot);
}

static inline int pxa_cache_ram_decide_mib(bool explicit_flag, int requested_mib, int64_t avail) {
    return explicit_flag ? requested_mib : pxa_cache_ram_plan_mib(avail, requested_mib);
}

// ---- the truth lines ----------------------------------------------------------------------------

static inline std::string pxa_ckpt_ram_fmt_bytes(int64_t b) {
    char buf[32];
    if (b >= (1ll << 30)) {
        snprintf(buf, sizeof(buf), "%.1f GiB", (double) b / (double) (1ll << 30));
    } else if (b >= (1ll << 20)) {
        snprintf(buf, sizeof(buf), "%.0f MiB", (double) b / (double) (1ll << 20));
    } else {
        snprintf(buf, sizeof(buf), "%.0f KiB", (double) b / 1024.0);
    }
    return buf;
}

// One plain line for the boot log. `explicit_flag`: the user passed --ctx-checkpoints, so the
// number is theirs and the line only says what it costs.
static inline std::string pxa_ckpt_ram_describe(int per_slot, int n_slots, int64_t est, int64_t avail,
                                                int max_per_slot, bool explicit_flag) {
    char buf[512];
    const int64_t slots = std::max(1, n_slots);
    if (per_slot <= 0) {
        if (explicit_flag) {
            return "context checkpoints: off (--ctx-checkpoints " + std::to_string(per_slot) + ")";
        }
        snprintf(buf, sizeof(buf),
                 "context checkpoints: OFF - only %s of RAM is free and one checkpoint per slot takes ~%s "
                 "(--ctx-checkpoints N to force them on)",
                 pxa_ckpt_ram_fmt_bytes(std::max<int64_t>(0, avail)).c_str(),
                 pxa_ckpt_ram_fmt_bytes(slots * est).c_str());
        return buf;
    }
    if (avail < 0) {
        snprintf(buf, sizeof(buf),
                 "context checkpoints: %d per slot (free RAM cannot be read here, so the default stands; "
                 "--ctx-checkpoints N to override)", per_slot);
        return buf;
    }
    const int64_t total = (int64_t) per_slot * slots * est;
    const std::string slot_note = slots > 1 ? (" x " + std::to_string(slots) + " slots") : std::string();
    if (explicit_flag) {
        snprintf(buf, sizeof(buf),
                 "context checkpoints: %d per slot%s (set by --ctx-checkpoints; up to ~%s of %s free RAM)%s",
                 per_slot, slot_note.c_str(), pxa_ckpt_ram_fmt_bytes(total).c_str(),
                 pxa_ckpt_ram_fmt_bytes(avail).c_str(),
                 total > avail / PXA_CKPT_RAM_FLOOR_DIV ? " - that is over half of it, expect swapping" : "");
        return buf;
    }
    snprintf(buf, sizeof(buf),
             "context checkpoints: %d per slot%s (%s~%s of %s free RAM, ~%s each; --ctx-checkpoints N to override)",
             per_slot, slot_note.c_str(), per_slot >= max_per_slot ? "up to " : "",
             pxa_ckpt_ram_fmt_bytes(total).c_str(), pxa_ckpt_ram_fmt_bytes(avail).c_str(),
             pxa_ckpt_ram_fmt_bytes(est).c_str());
    return buf;
}

static inline std::string pxa_cache_ram_describe(int mib, int want_mib, int64_t avail) {
    char buf[384];
    if (mib <= 0) {
        snprintf(buf, sizeof(buf),
                 "prompt cache (RAM): OFF - 15%% of the %s of free RAM is under %d MiB (--cache-ram N to force it on)",
                 pxa_ckpt_ram_fmt_bytes(std::max<int64_t>(0, avail)).c_str(), PXA_CACHE_RAM_MIN_MIB);
        return buf;
    }
    snprintf(buf, sizeof(buf),
             "prompt cache (RAM): %d MiB, capped from %d MiB to 15%% of the %s of free RAM (--cache-ram N to override)",
             mib, want_mib, pxa_ckpt_ram_fmt_bytes(avail).c_str());
    return buf;
}
