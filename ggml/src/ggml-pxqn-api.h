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

#ifdef __cplusplus
}
#endif
