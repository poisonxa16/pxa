#include "levers.cuh"

#include "../../common.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// One row per declared lever. The row is the whole declaration: the variable a user sets, the
// short name the boot report prints, the value that means "behave as the base engine did", and
// the one line that says what it is for. The rule that turns an environment string into a value
// lives in pxa_lever_resolve() below, one case per row, because the rules are genuinely different
// from each other and hiding that behind a common parser would be a lie.
struct pxa_lever_row {
    const char * env;
    const char * report_name;
    int64_t      off;
    const char * doc;
};

static const pxa_lever_row g_pxa_lever_rows[PXA_LEVER_COUNT] = {
    { "PXA_FA_D512_VOLTA",     "FA_D512_VOLTA",   0,
      "sm_70 head 512/512 flash attention: 0 the unfused chain, 1 the vendored m8n8k4 MMA kernel, "
      "2 the vendored tile kernel (the ENHANCE default). Fenced by the graph builder's width cut, "
      "context floor, every-card probe, architecture and offload checks." },
    { "PXA_FA_D256_VOLTA_TILE", "FA_D256_VOLTA_TILE", 0,
      "sm_70 head 256/256 flash attention on the vendored no-tensor-core tile kernel: 0 the route "
      "this fork always had (vector at width 1, legacy WMMA above it), 1 the tile kernel for the "
      "speculative verify widths, 2 for width 1 as well. Fenced by a KV floor and a width window." },
    { "PXA_FA_D256_VOLTA_TILE_MINKV", "FA_D256_VOLTA_TILE_MINKV", 1280,
      "the KV length at or above which PXA_FA_D256_VOLTA_TILE takes a node. The gain is at depth, "
      "and a fused attention kernel on this architecture has a known short-KV defect at head 512." },
    { "PXA_FA_D256_VOLTA_TILE_MAXCOLS", "FA_D256_VOLTA_TILE_MAXCOLS", 8,
      "the widest query batch PXA_FA_D256_VOLTA_TILE takes. Above it the node keeps the route it "
      "always had, which at prefill width is the vendored m8n8k4 MMA kernel." },
    { "PXA_FA_MMA_VOLTA",      "FA_MMA_VOLTA",    0,
      "sm_70 large-batch flash attention on the vendored mainline m8n8k4 MMA kernel. Default on; "
      "=0 hands those shapes back to the tile route. Decode is never routed here." },
    { "PXA_FA_MMA_VOLTA_Q8",   "FA_MMA_VOLTA_Q8", 0,
      "sm_70 head-256 prefill with a MATCHED q8_0 K/V cache also reaches that MMA kernel, staged "
      "to f16 through the CUDA pool. Default on; =0 restores the f16-only predicate. The admission "
      "declines itself when the staging would not fit in free device memory." },
    { "PXA_FA_TILE_VOLTA",     "FA_TILE_VOLTA",   0,
      "sm_70 flash attention on the no-tensor-core tile kernel: 0 off (default -- this route emits "
      "corrupt output on this architecture), 1 every batch size, 2 large batch only. Benchmarking." },
    { "PXA_FA_TILE256",        "FA_TILE256",      0,
      "pre-Volta D=256 attention at batch > 8 uses the tile-f16 ncols=16 kernel instead of the "
      "single-column vec kernel. Default on; =0 is a kernel-selection rollback, not a fix." },
    { "PXA_FA_SWA_SLICE",      "FA_SWA_SLICE",    0,
      "restore the windowed KV slice at dispatch. UNSOUND with np>1 or slot reuse (it can cut a "
      "sequence's own cells out of the view and produce NaN logits). Single-sequence benchmarking only." },
    { "PXA_FA_SWA_KEEP",       "FA_SWA_KEEP",     0,
      "keep n_swa in op_params[4] so the kernels' mask-driven KV_min_max scan bounds the SWA "
      "iteration. Default on; =0 restores the old full-range dispatch behaviour." },
    { "PXQ_SM60_FA_VEC_F32",   "SM60_FA_VEC_F32", 0,
      "sm_60 decode (batch <= 8) uses the fp32-accumulating vec kernel; the fp16 one flips ~3-4% "
      "of top-1 tokens and is not faster on a bandwidth-bound card. Default on." },
    { "PXA_CORE_ROUTES",       "CORE_ROUTES",     0,
      "print, at exit, how many flash-attention nodes each route served on each device. Default off." },
};

struct pxa_lever_state {
    int64_t      value;
    bool         set;   // the user set the variable to something this build understands
    const char * why;
};

static bool pxa_env_on_unless_zero(const char * v) { return !(v && v[0] == '0'); }
static bool pxa_env_on_if_one    (const char * v) { return   v && v[0] == '1';  }

// A lever whose value is a number rather than a mode. An unset, empty, unparsable or negative
// string keeps the row's default instead of silently meaning zero -- a KV floor of 0 and a KV floor
// of "nonsense" are very different instructions and only one of them was asked for.
static int64_t pxa_env_int(const char * env, const char * v, int64_t def) {
    if (!v || v[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const long long n = strtoll(v, &end, 10);
    if (end == v || *end != '\0' || n < 0) {
        fprintf(stderr, "%s=%s is not a non-negative integer; keeping %lld\n", env, v, (long long) def);
        return def;
    }
    return (int64_t) n;
}

static const pxa_lever_state * pxa_lever_table() {
    static pxa_lever_state st[PXA_LEVER_COUNT];
    static const bool resolved = [] {
        const int    level      = ggml_pxa_config_level();
        const char * level_name = ggml_pxa_config_level_name();

        for (int i = 0; i < PXA_LEVER_COUNT; ++i) {
            const char * v   = getenv(g_pxa_lever_rows[i].env);
            const bool   set = v && v[0] != '\0';
            int64_t      value;

            switch ((pxa_lever_id) i) {
                case PXA_LEVER_FA_D512_VOLTA:
                    // The default follows the config level like every other default: REFERENCE and
                    // DEFAULT keep the unfused chain, ENHANCE (the shipping level) takes the tile kernel.
                    value = !set ? (level >= 2 ? 2 : 0)
                                 : (v[0] == '1' && v[1] == '\0') ? 1
                                 : (v[0] == '2' && v[1] == '\0') ? 2 : 0;
                    if (set && value == 0 && strcmp(v, "0") != 0) {
                        fprintf(stderr, "PXA_FA_D512_VOLTA=%s is not a value this lever has (0 = off, 1 = MMA kernel, "
                                        "2 = tile kernel); treating it as 0 and keeping the unfused attention chain\n", v);
                    }
                    break;

                case PXA_LEVER_FA_D256_VOLTA_TILE:
                    // DEFAULT 2 AT THE ENHANCE LEVEL, like every other default that follows the
                    // config level. 2 rather than 1 because width 1 -- plain decode -- turned out to
                    // be the same win: on the 15k long class the plain step gains 9.3% and the
                    // speculative arms 12-18%, and per node at width 1 the kernel beats not only the
                    // vector kernel this engine had but mainline's own by 1.6-1.7x at depth.
                    value = !set ? (level >= 2 ? 2 : 0)
                                 : (v[0] == '1' && v[1] == '\0') ? 1
                                 : (v[0] == '2' && v[1] == '\0') ? 2 : 0;
                    if (set && value == 0 && strcmp(v, "0") != 0) {
                        fprintf(stderr, "PXA_FA_D256_VOLTA_TILE=%s is not a value this lever has (0 = off, "
                                        "1 = verify widths, 2 = width 1 as well); treating it as 0\n", v);
                    }
                    break;

                case PXA_LEVER_FA_D256_VOLTA_TILE_MINKV:
                case PXA_LEVER_FA_D256_VOLTA_TILE_MAXCOLS:
                    value = pxa_env_int(g_pxa_lever_rows[i].env, v, g_pxa_lever_rows[i].off);
                    break;

                case PXA_LEVER_FA_MMA_VOLTA:
                    value = pxa_env_on_unless_zero(v) ? 1 : 0;
                    // =2 also routed DECODE here. DISABLED 2026-09-20: the decode-width instances of
                    // that kernel are stubs on sm_70, so lifting the gate faults the card with an
                    // unspecified launch failure at widths 1, 2 and 4. Accepted and treated as =1, loudly.
                    if (set && v[0] == '2') {
                        fprintf(stderr, "PXA_FA_MMA_VOLTA=2 is disabled in this build (its decode-width kernel instances are not "
                                        "built for sm_70 and the launch faults); treating it as =1\n");
                    }
                    break;

                case PXA_LEVER_FA_TILE_VOLTA:
                    value = !set ? 0 : (v[0] == '0') ? 0 : (v[0] == '1') ? 1 : (v[0] == '2') ? 2 : 0;
                    if (value == 2) {
                        fprintf(stderr, "PXA_FA_TILE_VOLTA=2: sm_70 large-batch flash-attention -> tile kernel "
                                        "(KNOWN TO PRODUCE CORRUPT OUTPUT ON THIS ARCH -- benchmarking only)\n");
                    }
                    break;

                case PXA_LEVER_FA_MMA_VOLTA_Q8: value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
                case PXA_LEVER_FA_TILE256:      value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
                case PXA_LEVER_FA_SWA_KEEP:     value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
                case PXA_LEVER_SM60_FA_VEC_F32: value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
                case PXA_LEVER_FA_SWA_SLICE:    value = pxa_env_on_if_one(v)     ? 1 : 0; break;
                case PXA_LEVER_CORE_ROUTES:     value = pxa_env_on_if_one(v)     ? 1 : 0; break;

                default:                        value = g_pxa_lever_rows[i].off; break;
            }

            st[i].value = value;
            st[i].set   = set;
            // The reason is recorded WITH the value, at the moment the value is decided. This is the
            // whole point of the registry: the report cannot say "explicit env override" about a
            // default, because it does not re-derive anything -- it reads this field.
            st[i].why   = set ? "explicit env override" : level_name;
        }
        return true;
    }();
    (void) resolved;
    return st;
}

int64_t pxa_lever(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return 0;
    }
    return pxa_lever_table()[id].value;
}

bool pxa_lever_set_by_user(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return false;
    }
    return pxa_lever_table()[id].set;
}

const char * pxa_lever_why(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return "?";
    }
    return pxa_lever_table()[id].why;
}

void pxa_core_lever_report(void) {
    static std::atomic<bool> told{false};
    if (told.exchange(true)) {
        return;
    }
    const pxa_lever_state * st = pxa_lever_table();
    const bool verbose = pxa_env_on_if_one(getenv("PXA_CORE_LEVERS"));
    for (int i = 0; i < PXA_LEVER_COUNT; ++i) {
        fprintf(stderr, "PXA_AUTO: %s=%lld (%s; override %s)\n",
                g_pxa_lever_rows[i].report_name, (long long) st[i].value,
                st[i].why, g_pxa_lever_rows[i].env);
        if (verbose) {
            fprintf(stderr, "PXA_AUTO:   %s\n", g_pxa_lever_rows[i].doc);
        }
    }
}

void pxa_core_lever_check_unknown(void) {
    // Gated until the registry is complete: today most PXA_* levers are still read by their own
    // getenv, so an ungated warning would name dozens of variables that work perfectly well.
    // PXA_CORE_LEVERS=1 turns it on for the levers that HAVE moved.
    if (!pxa_env_on_if_one(getenv("PXA_CORE_LEVERS"))) {
        return;
    }
    extern char ** environ;
    for (char ** e = environ; e && *e; ++e) {
        if (strncmp(*e, "PXA_", 4) != 0 && strncmp(*e, "PXQ_", 4) != 0) { // the registry declares PXQ_ rows too
            continue;
        }
        const char * eq = strchr(*e, '=');
        if (!eq) {
            continue;
        }
        const size_t n = (size_t) (eq - *e);
        bool known = false;
        for (int i = 0; i < PXA_LEVER_COUNT && !known; ++i) {
            known = strlen(g_pxa_lever_rows[i].env) == n && strncmp(g_pxa_lever_rows[i].env, *e, n) == 0;
        }
        if (!known) {
            fprintf(stderr, "PXA_CORE_LEVERS: %.*s is not a lever the core declares (it may still be read "
                            "by its own site; the registry is being filled in step by step)\n", (int) n, *e);
        }
    }
}
