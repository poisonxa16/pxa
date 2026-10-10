// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// ggml-pxqn-tune.h -- optional tuning switches supplied by the closed PXQN library (libggml-pxqn). Open.
// The library decides which switches are on and why; the engine only asks. Without the library (or with
// PXA_PXQN_DISABLE=1) every switch reads 0 and the engine runs its plain path.
#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_pxqn_tune_bit {
    GGML_PXQN_TUNE_COMPANION = 1u << 0,   // llama: the speculation companion context takes the library's quiet path
    GGML_PXQN_TUNE_ORDER     = 1u << 1,   // llama: the expert-cache graph takes the library's node order
};

#define GGML_PXQN_TUNE_VERSION 1u
#define GGML_PXQN_TUNE_SYM     "ggml_pxqn_get_tune_api"

struct ggml_pxqn_tune_api {
    uint32_t version;   // GGML_PXQN_TUNE_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_tune_api)
    uint32_t (*flags)(void);
};
typedef const struct ggml_pxqn_tune_api * (*ggml_pxqn_get_tune_api_fn)(uint32_t version);

// the switches the library turns on (read once); 0 without the library
GGML_API uint32_t ggml_pxqn_tune_flags(void);

#ifdef __cplusplus
}
#endif
