// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// ggml-pxqn-levers.h -- does the closed PXQN library (libggml-pxqn) read this lever? Open.
// The engine's lever catalog is the public table; the levers the library reads are not in it, so the engine asks the
// library before it calls a PXA_* variable a typo. 0 without the library (or with PXA_PXQN_DISABLE=1): then nothing
// reads them, and the warning is right.
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_PXQN_LEVER_SYM "ggml_pxqn_lever_known"
typedef int (*ggml_pxqn_lever_known_fn)(const char * name);

// 1 when the loaded closed library reads the environment variable NAME
GGML_API int ggml_pxqn_lever_builtin(const char * name);

#ifdef __cplusplus
}
#endif
