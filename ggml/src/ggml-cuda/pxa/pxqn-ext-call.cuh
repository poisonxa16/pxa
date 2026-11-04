// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// pxqn-ext-call.cuh -- PXA_CLOSED_HOOK: the CUDA backend's calls into the PXQN library's role tables
// (ggml-pxqn-ext.h). Each helper offers one launch to a role and answers true when the library took it; false
// (no library, no table, or the library declined) leaves the caller's own route to run unchanged.
#pragma once

#include "../../../include/ggml-pxqn-ext.h"

#include <cstring>

// role "gemv": MD = the caller's matrix descriptor ({ w, dst, nb01, sd, nrows, bias })
template <typename MD>
static inline bool pxqn_ext_gemv(int device, int cc, int nsm, int type, int64_t ny, int64_t K, const float * x, int64_t sx,
                                 const MD * ms, int n, uint32_t flags, float limit, void * stream) {
    if (ggml_pxqn_gemv_api_get() == nullptr || n < 1 || n > GGML_PXQN_GEMV_MAX_MATS) return false;
    ggml_pxqn_gemv_args a;
    memset(&a, 0, sizeof(a));
    a.size = (uint32_t) sizeof(a);
    a.type = type; a.device = device; a.cc = cc; a.nsm = nsm;
    a.ny = (int32_t) ny; a.K = K; a.x = x; a.sx = sx; a.n_mats = n; a.flags = flags; a.limit = limit;
    for (int i = 0; i < n; ++i) {
        a.mats[i].w = ms[i].w; a.mats[i].dst = ms[i].dst; a.mats[i].bias = ms[i].bias;
        a.mats[i].nb01 = ms[i].nb01; a.mats[i].sd = ms[i].sd; a.mats[i].rows = ms[i].nrows;
    }
    return ggml_pxqn_gemv_launch(&a, stream) == 0;
}

// role "xchg": one card's half of a tensor-split all-reduce, the engine's exchange descriptor `task` (task_size bytes)
static inline bool pxqn_ext_xchg(int device, int type, int nrank, int64_t nelem, int64_t ne0, int64_t nrows, float * data,
                                 const void * task, size_t task_size, int token, int * err_flag, size_t area_bytes,
                                 int lines, void * stream) {
    if (ggml_pxqn_xchg_api_get() == nullptr) return false;
    ggml_pxqn_xchg_args a;
    memset(&a, 0, sizeof(a));
    a.size = (uint32_t) sizeof(a);
    a.device = device; a.type = type; a.nrank = nrank; a.nelem = nelem; a.ne0 = ne0; a.nrows = (int32_t) nrows;
    a.data = data; a.task = task; a.task_size = (uint32_t) task_size; a.token = token; a.err_flag = err_flag;
    a.area_bytes = area_bytes; a.lines = lines;
    return ggml_pxqn_xchg_launch(&a, stream) == 0;
}

// role "xchg": the most rows of a reduce whose multi-row epilogue the library runs (0 = none)
static inline int pxqn_ext_xchg_rows(int nrank, int64_t ne0, int lines) {
    return ggml_pxqn_xchg_api_get() == nullptr ? 0 : ggml_pxqn_xchg_rows(nrank, ne0, lines);
}

// role "fa": the library's pick among the engine's own flash-attention kernels for one node (enum ggml_pxqn_fa_route;
// -1 = no library, no table or no answer: the engine's own rule)
static inline int pxqn_ext_fa_pick(int cc, int d_k, int d_v, int64_t ncols, int64_t n_kv, int type_k, int type_v, int prec) {
    if (ggml_pxqn_fa_api_get() == nullptr) return -1;
    ggml_pxqn_fa_args a;
    memset(&a, 0, sizeof(a));
    a.size = (uint32_t) sizeof(a);
    a.cc = cc; a.d_k = d_k; a.d_v = d_v; a.ncols = ncols; a.n_kv = n_kv; a.type_k = type_k; a.type_v = type_v; a.prec = prec;
    return ggml_pxqn_fa_pick(&a);
}
