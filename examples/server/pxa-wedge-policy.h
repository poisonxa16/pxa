#pragma once
// PXA_JIT_WEDGE_v1: the stall limit the PXA_WEDGE_EXIT_v1 watchdog (and /health) applies to one
// llama_decode in flight.
//
// A card with no native SASS in this build (H100/B200/RTX 50 on a release build, or any card under
// CUDA_FORCE_PTX_JIT=1) has its kernels compiled by the driver from PTX the first time each module
// is used. With a cold JIT cache that is minutes per decode for the first decodes of the process,
// and the 180 s watchdog used to kill the server (exit 41) before it ever answered.
//
// Policy: when the device runs JIT code, the limit is stretched to PXA_WEDGE_JIT_MS (default 30 min)
// until PXA_WEDGE_JIT_DECODES (default 256) decodes have completed in this process: by then the
// prompt, generation and warm-up shapes have loaded (and compiled) their modules. After that the
// normal PXA_WEDGE_EXIT_MS limit applies again, so a genuinely hung driver is still caught. With a
// warm cache the first decodes are fast and the window simply runs out.
//   PXA_WEDGE_JIT_MS=0       -> never stretch (the old behaviour; the off switch)
//   PXA_WEDGE_JIT_DECODES=N  -> window length in completed decodes (0 = no window = never stretch)
//   PXA_WEDGE_EXIT_MS=0      -> the whole monitor off (unchanged)
// A card with native SASS gets exactly the old limit: nothing changes there.
//
// Header-only so tests/test-pxa-jit-wedge.cpp checks it without a server, a model or a GPU.

#include <cstdint>
#include <cstdlib>

struct pxa_wedge_policy {
    int64_t base_ms     = 180000;    // PXA_WEDGE_EXIT_MS (<= 0: monitor off)
    int64_t jit_ms      = 1800000;   // PXA_WEDGE_JIT_MS (<= 0: never stretch)
    int64_t jit_decodes = 256;       // PXA_WEDGE_JIT_DECODES (<= 0: no window)
    bool    jit         = false;     // a device of this process runs driver-JIT code
};

static inline int64_t pxa_wedge_env_i64(const char * v, int64_t dflt) {
    if (v == nullptr || *v == '\0') {
        return dflt;
    }
    char * end = nullptr;
    const long long x = strtoll(v, &end, 10);
    if (end == v) {
        return dflt;
    }
    return (int64_t) x;
}

// get(name) -> value or nullptr; getenv in the server, a table in the test.
template <typename Get>
static inline pxa_wedge_policy pxa_wedge_policy_from(bool jit, Get get, int64_t base_default) {
    pxa_wedge_policy p;
    const char * b = get("PXA_WEDGE_EXIT_MS");
    p.base_ms     = b ? (int64_t) atoll(b) : base_default;   // atoll exactly as before: "" / junk = 0 = off
    p.jit_ms      = pxa_wedge_env_i64(get("PXA_WEDGE_JIT_MS"),      p.jit_ms);
    p.jit_decodes = pxa_wedge_env_i64(get("PXA_WEDGE_JIT_DECODES"), p.jit_decodes);
    p.jit         = jit;
    return p;
}

// true while the stretched limit is in force (JIT device, inside the first-decodes window)
static inline bool pxa_wedge_jit_window(const pxa_wedge_policy & p, int64_t decodes_done) {
    return p.jit && p.jit_ms > 0 && p.jit_decodes > 0 && decodes_done < p.jit_decodes;
}

// the limit for a decode that starts after `decodes_done` completed ones; `base` is the limit the
// caller would use without JIT (the watchdog's PXA_WEDGE_EXIT_MS or /health's PXA_HEALTH_STALL_MS).
// <= 0 stays <= 0 (that monitor is off).
static inline int64_t pxa_wedge_limit_ms(const pxa_wedge_policy & p, int64_t base, int64_t decodes_done) {
    if (base <= 0) {
        return base;
    }
    if (!pxa_wedge_jit_window(p, decodes_done)) {
        return base;
    }
    return p.jit_ms > base ? p.jit_ms : base;
}
