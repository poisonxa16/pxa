// ggml-pxqn-api.h -- how the open engine reaches the closed PXQN library (libggml-pxqn), .
//
// PXQN kernels are not part of the open source tree. They ship compiled, as libggml-pxqn.so next to libggml.so in
// the PXA release tarball and images. libggml loads it at first use (ggml-pxqn-loader.cpp): from its own directory,
// else by name (RPATH / LD_LIBRARY_PATH), or from PXA_PXQN_LIB=<path>; PXA_PXQN_DISABLE=1 skips it. The library
// exports ONE symbol, ggml_pxqn_get_api, returning the table below. Without it (or on any version / layout
// mismatch) PXQN stays unavailable: files with PXQN tensors are refused at load with GGML_PXQN_MISSING_MSG, and
// classic PXQ, k-quants and every other type run exactly as before.
#pragma once

#include "ggml.h"
#include "ggml-pxqn-spec.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_PXQN_LIB_VERSION 1u
#define GGML_PXQN_LIB_NAME    "libggml-pxqn.so"
#define GGML_PXQN_MISSING_MSG "PXQN models need the PXA release build (libggml-pxqn); classic PXQ/k-quants work in this build"

struct ggml_pxqn_cpu_api {
    uint32_t version;   // GGML_PXQN_LIB_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_cpu_api)
    void (*deq_row)(enum ggml_type type, const void * data, int64_t row, int64_t k, float * dst);   // PXQN types only
    void (*rht)(struct ggml_tensor * dst, int ith, int nth);                                        // GGML_OP_PXQN_RHT
};

// Optional extra symbol (not part of the table above, so a library without it and an engine without the hook still match):
//   ggml_pxqn_cpu_mul_mat -- the fast CPU matmul of PXQN host-RAM weights.
// The loader resolves it with dlsym; an older library simply lacks it. Same addressing as the engine's pxa_pxq_mul_mat_cpu:
// a = base of one 2D [k x nr0] expert slice; out row ix of column iy is dst_row(iy)[ix] with x(iy) / dst_row(iy) taken from
// `rows` (routed: x = src1f + rows[iy].i2*nb12 + (rows[iy].i1 % ne11)*nb11, out = dst + rows[iy].i1*nb1 + rows[iy].i2*nb2) or,
// rows == NULL, x = src1f + iy*nb11, out = dst + iy*nb1. Every compute thread calls it with its own (ith, nth); the library
// splits the rows itself and needs no barrier.
#define GGML_PXQN_CPU_MUL_MAT_SYM "ggml_pxqn_cpu_mul_mat"

// one routed row of a mul_mat_id (layout-identical to the engine's pxa_pxq_rowmap): i1 = expert slot, i2 = token row
struct ggml_pxqn_rowmap {
    int32_t i1;
    int32_t i2;
};

typedef void (*ggml_pxqn_cpu_mul_mat_fn)(enum ggml_type type, const void * a, int64_t nr0, int64_t k,
                                         const char * src1f, size_t nb11, size_t nb12, char * dst, size_t nb1, size_t nb2,
                                         const struct ggml_pxqn_rowmap * rows, int ne11, int64_t ny, int ith, int nth);

// NULL when the library is absent, has no fast CPU matmul, or PXA_PXQN_CPU_MMV=0
ggml_pxqn_cpu_mul_mat_fn ggml_pxqn_cpu_mul_mat_get(void);
// true when ggml_pxqn_cpu_mul_mat_get() is non-NULL: the planner then reads PXQN cold experts on the CPU (llama-pxa-place.cpp)
GGML_API bool ggml_pxqn_cpu_mmv_available(void);

// Optional extra symbol: ggml_pxqn_xcache_get -- the online-adaptation POLICY of PXA_XCACHE (expert-granular hot cache).
// The engine (open, src/llama-pxa-xcache.cpp) owns the mechanism: per-layer routing counters on the device, the slot table, the
// copy stream, the map flips between two decodes. The library owns the decisions: how the routing counts are decayed and ranked,
// which cold expert replaces which hot one, how many swaps one step may start (byte budget), what a miss costs on the CPU vs a copy.
// Absent (older library, PXA_PXQN_XCACHE=0) the engine keeps today's static plan exactly. Pure functions of what they are given:
// no clocks, no threads, no randomness, so the same token stream produces the same swap schedule.
#define GGML_PXQN_XCACHE_SYM     "ggml_pxqn_xcache_get"
#define GGML_PXQN_XCACHE_VERSION 1u

struct ggml_pxqn_xc_layer_cfg {
    int32_t        n_expert;
    int32_t        n_slots;          // hot stack capacity (slots of the hot stack)
    uint64_t       expert_bytes;     // bytes of one expert over all its tensors
    const double * prior;            // [n_expert] relative routing frequency from the static counts, NULL when unknown
};
struct ggml_pxqn_xc_cfg {
    int32_t                          n_layers;     // split layers
    const struct ggml_pxqn_xc_layer_cfg * layers;
    int32_t                          n_used;       // routed experts per token
    double                           pcie_gbs;     // measured pinned host -> device GB/s (0 unknown)
    double                           cpu_gbs;      // measured CPU matvec GB/s of the cold path (0 unknown)
    uint32_t                         flags;
};
// one layer's observation: the device counters as read back (u32, cumulative, wrap-safe: the policy differences snapshots)
struct ggml_pxqn_xc_obs {
    const uint32_t * counts;         // [n_expert]
    uint32_t         hits, misses;   // cumulative routings served by the hot stack / that went to the cold stack
};
// one layer's state when a plan is asked for
struct ggml_pxqn_xc_view {
    const int32_t * map;             // [n_expert] >= 0 hot slot, <= -2 cold slot (-2 - slot)
    const uint8_t * busy;            // [n_expert] 1 = a swap involving this expert is in flight: leave it alone
    int32_t         n_spare;         // free cold-stack slots right now (each swap in flight holds one: the evicted expert is copied into it)
};
struct ggml_pxqn_xc_swap { int32_t layer; int32_t e_in; int32_t e_out; };   // layer = index into cfg.layers; e_in cold -> hot, e_out hot -> cold
struct ggml_pxqn_xc_stats {
    uint64_t obs_steps, plans, swaps_planned;
    double   hit_rate_all;           // hits / (hits + misses) since the start
    double   hit_rate_recent;        // over the decayed window
    double   window_tokens;
};
struct ggml_pxqn_xcache_policy {
    uint32_t version;                // GGML_PXQN_XCACHE_VERSION
    uint32_t size;                   // sizeof(struct ggml_pxqn_xcache_policy)
    void * (*create)(const struct ggml_pxqn_xc_cfg * cfg);
    void   (*destroy)(void * st);
    // n_tokens = tokens decoded since the previous observe (all layers share it); obs has cfg.n_layers entries
    void   (*observe)(void * st, uint32_t n_tokens, const struct ggml_pxqn_xc_obs * obs);
    // swaps to start now, at most max_out, together moving at most budget_bytes host->device; returns how many
    int32_t (*plan)(void * st, const struct ggml_pxqn_xc_view * views, uint64_t budget_bytes, struct ggml_pxqn_xc_swap * out, int32_t max_out);
    void   (*stats)(const void * st, struct ggml_pxqn_xc_stats * out);
    // swaps per observation step the link can carry without crowding a decode (quantised so the schedule is reproducible)
    int32_t (*swap_budget)(const void * st, uint32_t obs_every_tokens);
    // the miss price of the planner's cost row for this cold path: us a cold expert costs on the CPU / over the link, from the
    // measured throughputs. Returns 0 when the throughputs are unknown (the planner keeps its table).
    int    (*miss_cost_us)(uint64_t expert_bytes, double pcie_gbs, double cpu_gbs, double * cpu_us, double * link_us);
    // the factor the static planner applies to its price of a missed routing when the adaptation is on (the resident set tracks the
    // routing, so fewer routings miss than the static counts predict); 1 = the static price
    double (*plan_miss_scale)(void);
};
typedef const struct ggml_pxqn_xcache_policy * (*ggml_pxqn_xcache_get_fn)(uint32_t version);
// NULL when the library is absent, has no policy, or PXA_PXQN_XCACHE=0
GGML_API const struct ggml_pxqn_xcache_policy * ggml_pxqn_xcache_policy_get(void);

// Optional symbol: build one learned counts table from a packed prior and packed device rows.
// The engine writes the csv. NULL when the library is absent: the engine does not save learned counts.
// 0 = out_counts filled, 1 = nothing observed, 2 = *wait_layer is not ready, -1 = bad arguments.
#define GGML_PXQN_XCACHE_LEARN_BUILD_SYM "ggml_pxqn_xcache_learn_build"
typedef int (*ggml_pxqn_xcache_learn_build_fn)(int n_layer, int n_expert, int n_main,
                                               const double * prior,
                                               const int * obs_il, int n_obs, const uint32_t * obs,
                                               const uint8_t * routed,
                                               uint64_t * out_counts,
                                               int * wait_layer);
GGML_API ggml_pxqn_xcache_learn_build_fn ggml_pxqn_xcache_learn_build_get(void);

// Optional symbol: which counts file a boot opens. 0 = curated, 1 = learned.
// explicit_flag nonzero returns 0. Means are filled when a measurement was read, else -1.
// NULL when the library is absent: the engine keeps the curated file and does not open a learned one.
#define GGML_PXQN_XCACHE_COUNTS_PICK_SYM "ggml_pxqn_xcache_counts_pick"
typedef int (*ggml_pxqn_xcache_counts_pick_fn)(const char * curated, const char * learned, int explicit_flag,
                                               double * learned_mean, double * curated_mean);
GGML_API ggml_pxqn_xcache_counts_pick_fn ggml_pxqn_xcache_counts_pick_get(void);

#define GGML_PXQN_XCACHE_SPEED_WRITE_SYM "ggml_pxqn_xcache_speed_write"
typedef int (*ggml_pxqn_xcache_speed_write_fn)(const char * csv_path, double prose, double code);
GGML_API ggml_pxqn_xcache_speed_write_fn ggml_pxqn_xcache_speed_write_get(void);

// Optional symbol: the next calibration step. which 0 logs; which 1 measures.
// Writes kind and prompt. 1 = a step, 0 = past the end, -1 = bad arguments.
// NULL when the library is absent: the tool does not calibrate.
#define GGML_PXQN_XCACHE_CALIB_STEP_SYM "ggml_pxqn_xcache_calib_step"
typedef int (*ggml_pxqn_xcache_calib_step_fn)(int which, int index, char * kind, size_t kind_n,
                                              char * prompt, size_t prompt_n, int * n_predict, double * temperature);
GGML_API ggml_pxqn_xcache_calib_step_fn ggml_pxqn_xcache_calib_step_get(void);

// Optional symbol: whether the newest measurement is kept. hist is prose,code pairs.
// 0 = keep it, 1 = stop, -1 = bad arguments. *best_index is the kept row.
#define GGML_PXQN_XCACHE_CALIB_CONSIDER_SYM "ggml_pxqn_xcache_calib_consider"
typedef int (*ggml_pxqn_xcache_calib_consider_fn)(const double * hist, int n_hist, double prose, double code, int * best_index);
GGML_API ggml_pxqn_xcache_calib_consider_fn ggml_pxqn_xcache_calib_consider_get(void);

// Optional symbols: the cold-expert CPU pool. NULL when the library is absent: the switch stays off.
#define GGML_PXQN_XCACHE_CPUPOOL_CREATE_SYM "ggml_pxqn_xcache_cpupool_create"
typedef void * (*ggml_pxqn_xcache_cpupool_create_fn)(int n_threads_hint, int spin_us);
GGML_API ggml_pxqn_xcache_cpupool_create_fn ggml_pxqn_xcache_cpupool_create_get(void);

#define GGML_PXQN_XCACHE_CPUPOOL_DESTROY_SYM "ggml_pxqn_xcache_cpupool_destroy"
typedef void (*ggml_pxqn_xcache_cpupool_destroy_fn)(void * pool);
GGML_API ggml_pxqn_xcache_cpupool_destroy_fn ggml_pxqn_xcache_cpupool_destroy_get(void);

#define GGML_PXQN_XCACHE_CPUPOOL_STEP_SYM "ggml_pxqn_xcache_cpupool_step"
typedef int (*ggml_pxqn_xcache_cpupool_step_fn)(void * pool, struct ggml_tensor * up, struct ggml_tensor * gate,
                                                struct ggml_tensor * unary, struct ggml_tensor * down,
                                                const int32_t * ids, int n_used, int ntok);
GGML_API ggml_pxqn_xcache_cpupool_step_fn ggml_pxqn_xcache_cpupool_step_get(void);

struct ggml_pxqn_lib_api {
    uint32_t                         version;   // GGML_PXQN_LIB_VERSION
    uint32_t                         size;      // sizeof(struct ggml_pxqn_lib_api)
    const char *                     abi;       // ggml_pxqn_abi_sig() of the engine build the library was built with
    const char *                     build;     // free-form build id (engine commit), printed once at load
    const struct ggml_pxqn_cpu_api * cpu;
    const void *                     cuda;      // struct ggml_pxqn_cuda_api (ggml-cuda/pxa/pxqn-api.cuh) or NULL
};

typedef const struct ggml_pxqn_lib_api * (*ggml_pxqn_get_api_fn)(uint32_t version);

// the layouts the table's callers and callees share; a library built against other layouts is refused
static inline void ggml_pxqn_abi_sig(char * buf, size_t n) {
    snprintf(buf, n, "pxqn1:t%u:o%d:y%d:s%d:p%d:d%d", (unsigned) sizeof(struct ggml_tensor), (int) GGML_OP_COUNT,
             (int) GGML_TYPE_COUNT, (int) GGML_MAX_SRC, (int) GGML_MAX_OP_PARAMS, (int) GGML_MAX_DIMS);
}

// NULL when libggml-pxqn is absent or refused (the reason is printed once to stderr)
const struct ggml_pxqn_lib_api * ggml_pxqn_lib(void);

// ---- locked models (Pro encoder: pxa.lock.mode = supporters | personal) ----------------------------------------------------------------------------------
// A model written with a lock has its PXQN tensors stored encrypted and listed in the tensor directory under type id GGML_GGUF_LOCKED_TYPE_BASE + type, which a
// reader that does not know the lock refuses at load. Everything that matters -- the key handling, the licence check, the decryption -- lives in libggml-pxqn
// behind this second, optional entry point (a library without it, or no library at all, simply cannot open a locked model). The engine only hands over the
// file's pxa.lock.* header keys once, and each locked tensor's bytes after they are read and before they are uploaded.
#define GGML_PXQN_LOCK_VERSION 1u
#define GGML_PXQN_LOCKED_MSG  "This model is locked to PXA supporters - see https://ko-fi.com/shatteredrealms1 and /encoder in our Discord"
#define GGML_PXQN_LOCKED_OLD_MSG "This model is locked to PXA supporters and this libggml-pxqn is too old to unlock it - update to the latest PXA release " \
                                 "(https://ko-fi.com/shatteredrealms1 and /encoder in our Discord)"

struct ggml_pxqn_lock_api {
    uint32_t version;   // GGML_PXQN_LOCK_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_lock_api)
    // keys/vals: the file's pxa.lock.* header entries as strings (numbers in decimal). Returns a handle, or NULL with a plain sentence in err.
    void * (*open)(const char * const * keys, const char * const * vals, int n, char * err, int errlen);
    // decrypt n bytes of tensor `name` that start at byte `off` of the tensor's data, in place; thread-safe, no-op cost-free for other tensors
    void   (*decrypt)(void * handle, const char * name, uint64_t off, void * buf, size_t n);
    void   (*close)(void * handle);
};
typedef const struct ggml_pxqn_lock_api * (*ggml_pxqn_get_lock_api_fn)(uint32_t version);

// NULL when libggml-pxqn is absent, refused, or has no lock entry point
const struct ggml_pxqn_lock_api * ggml_pxqn_lock(void);

// ---- the speculation selector: an OPTIONAL second table ---------------------------------
// Looked up by name (ggml_pxqn_get_spec_api) after the main table passed its checks, so a library without the
// selector, or an engine that never asks, is unaffected by it. The engine reaches it only through the
// ggml_pxqn_spec_* wrappers in ggml-pxqn-spec.h (ggml-pxqn-loader.cpp).
#define GGML_PXQN_SPEC_VERSION 1u

struct ggml_pxqn_spec_api {
    uint32_t version;   // GGML_PXQN_SPEC_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_spec_api)
    void * (*create)(const struct ggml_pxqn_spec_cfg * cfg);
    void   (*destroy)(void * h);
    void   (*reset)(void * h, int32_t seq_id);
    void   (*choose)(void * h, const struct ggml_pxqn_spec_offer * offer, struct ggml_pxqn_spec_pick * pick);
    void   (*observe)(void * h, int32_t seq_id, int32_t n_verified, int32_t n_accepted);
    int    (*describe)(void * h, char * buf, size_t n);
};

typedef const struct ggml_pxqn_spec_api * (*ggml_pxqn_get_spec_api_fn)(uint32_t version);

// NULL when libggml-pxqn is absent / refused, does not carry the selector, or PXA_SPEC_SELECT=0
const struct ggml_pxqn_spec_api * ggml_pxqn_spec_table(void);


// ---- per-model / per-card speculation defaults: an OPTIONAL third table -----------------------------
// Same rule as the selector table: looked up by name after the main table passed its checks; a library without it,
// or an engine that never asks, is unaffected. The engine reaches it through the ggml_pxqn_specdef_* wrappers in
// ggml-pxqn-spec.h.
#define GGML_PXQN_SPECDEF_VERSION 1u

struct ggml_pxqn_specdef_api {
    uint32_t version;   // GGML_PXQN_SPECDEF_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_specdef_api)
    bool    (*choose)(const struct ggml_pxqn_specdef_in * in, struct ggml_pxqn_specdef_out * out);
    int32_t (*select)(const char * arch, int32_t has_ngram, int32_t has_mtp, int32_t has_assistant, int32_t n_dev, int32_t cc_min);
    int32_t (*shortlist)(const char * arch, int32_t n_vocab, int32_t head_type, int32_t n_dev, int32_t cc_min);
};

typedef const struct ggml_pxqn_specdef_api * (*ggml_pxqn_get_specdef_api_fn)(uint32_t version);

// NULL when libggml-pxqn is absent / refused, does not carry the table, or PXA_SPEC_DEFAULTS=0
const struct ggml_pxqn_specdef_api * ggml_pxqn_specdef_table(void);

#ifdef __cplusplus
}
#endif
