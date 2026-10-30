// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// ggml-pxqn-overdrive.h -- PXA Overdrive: the per-class lever set the closed PXQN library (libggml-pxqn) carries, and
// the decision that applies it. Open: this header holds the shapes only. No lever name, no value, no class rule.
//
// The engine's autoconfig registry (common/pxa-registry.cpp) hands the library what it knows about the seat (the
// cards, the GGUF header, the user's picks, the configuration level) and appends the rows the library answers with,
// in the library's order, with setenv that never overwrites (a value already in the environment wins, as always).
// Without the library, with PXA_PXQN_DISABLE=1, or for a seat the library has no recipe for, nothing is appended and
// the engine runs exactly as before. Same rule as the other optional tables (ggml-pxqn-spec.h, ggml-pxqn-tune.h):
// looked up by name after the main table passed its checks, so a library without it and an engine that never asks
// are unaffected by each other.
//
// Switches (public, documented in the lever catalog): PXA_AUTO_OVERDRIVE=0 or PXA_NO_OVERDRIVE=1 stands the whole set
// down (read by the engine, below); PXA_OVERDRIVE=1 asks the library for the class recipe even where its fit test
// would not pick it, =0 stands it down (read by the library); PXA_OVERDRIVE_LIB=1 turns on only the rows of the set
// whose readers live inside the library (the alias the presets and the launcher used before 2026-10-19).
#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_PXQN_OVERDRIVE_SYM     "ggml_pxqn_get_overdrive_api"
#define GGML_PXQN_OVERDRIVE_VERSION 1u

// what the engine knows about the seat before any weight is loaded
struct ggml_pxqn_od_in {
    uint32_t         size;          // sizeof(struct ggml_pxqn_od_in): the library refuses another layout
    const char     * arch;          // general.architecture of the file ("qwen4exp", "qwen35moe", ...)
    const char     * tier;          // the file's PXQ tier ("PXQN2", "PXQN4", "PXQ_UNIVERSAL", "PXQ4", ...), "" when not PXQ
    int32_t          n_expert;      // routed experts per layer: > 0 MoE, 0 dense, -1 unknown
    int32_t          n_nextn;       // MTP head layers in the file: > 0 a head, 0 none, -1 unknown
    int32_t          n_layer;       // block count, -1 unknown
    uint64_t         model_bytes;   // the file over every shard, 0 unknown
    int32_t          n_dev;         // CUDA devices the model is spread over
    const int32_t  * cc;            // [n_dev] compute capability, 100 * major + 10 * minor (600, 610, 700), NULL when n_dev == 0
    const uint64_t * vram_bytes;    // [n_dev] total memory per device, 0 = unknown, NULL when n_dev == 0
    int32_t          split;         // the split mode in effect (llama_split_mode: 1 layer, 4 tensor), -1 none / kept
    int32_t          level;         // configuration level: 0 reference, 1 default, 2 enhance
    uint32_t         flags;         // GGML_PXQN_OD_IN_*
};
#define GGML_PXQN_OD_IN_SPLIT_FORCED 1u   // the user named -sm (split is the user's, not the registry's pick)

// one row of the answer: an environment variable and the value the recipe was measured with
struct ggml_pxqn_od_row {
    const char * name;
    const char * value;
    int32_t      closed;            // 1 = read only inside the library (never named in a user-facing log; see below)
};

// the answer: static storage inside the library, valid for the life of the process
struct ggml_pxqn_od_out {
    uint32_t                        size;        // sizeof(struct ggml_pxqn_od_out)
    int32_t                         recipe;      // 0 = no recipe for this seat (nothing to append); > 0 = the recipe id
    int32_t                         n_rows;
    const struct ggml_pxqn_od_row * rows;        // [n_rows] in apply order
    const char                    * pool_name;   // non-NULL: the recipe wants the engine's derived cold-expert CPU pool appended
                                                 // under this name when the environment has none (the derivation is the engine's)
    char                            why[512];    // the pick's reason for the boot banner; never names a closed row
};

struct ggml_pxqn_overdrive_api {
    uint32_t version;   // GGML_PXQN_OVERDRIVE_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_overdrive_api)
    // the decision and the rows: 1 = out filled with a recipe, 0 = no recipe for this seat (out->recipe == 0), -1 = bad input
    int32_t (*choose)(const struct ggml_pxqn_od_in * in, struct ggml_pxqn_od_out * out);
    // PXA_OVERDRIVE_LIB=1: the library-only rows of the recipe are set to their recipe value (setenv, never overwriting)
    void    (*alias_apply)(void);
    // 1 when NAME is a library-only row of the recipe (the EXPLAIN json carries such rows as the alias, not by name)
    int32_t (*is_member)(const char * name);
};
typedef const struct ggml_pxqn_overdrive_api * (*ggml_pxqn_get_overdrive_api_fn)(uint32_t version);

// The engine's side (ggml-pxqn-loader.cpp). All of them are inert without the library.
GGML_API bool ggml_pxqn_overdrive_available(void);
// false (and out->recipe = 0, out->n_rows = 0) without the library, with PXA_AUTO_OVERDRIVE=0 or PXA_NO_OVERDRIVE=1,
// or when the library has no recipe for this seat
GGML_API bool ggml_pxqn_overdrive_choose(const struct ggml_pxqn_od_in * in, struct ggml_pxqn_od_out * out);
GGML_API void ggml_pxqn_overdrive_alias_apply(void);
GGML_API int  ggml_pxqn_overdrive_is_member(const char * name);

#ifdef __cplusplus
}
#endif
