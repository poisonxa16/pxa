#pragma once

// PXA_GLM_CONCAT_I32_CUDA (default OFF): let the CUDA backend claim I32 CONCAT.
//
// Why: glm5next's DSA indexer appends the always-attended tail cells to the selected cell table
// with an I32 concat (src/graphs/build_glm5next.cpp, "indexer_kpool_select_tail"). CUDA's
// supports_op refused every I32 CONCAT, so the scheduler put that one node on the CPU backend:
// per MLA block (11 per token) a D2H of the cell table, a full device drain, a CPU concat and an
// H2D back, plus two extra split boundaries.
//
// Why it is bit-exact: ggml_cuda_op_concat's two paths this predicate admits never do arithmetic
// on the payload. The contiguous dim 1/2/3 path is two cudaMemcpyAsync; the dim-0 path moves
// 4-byte words through float-typed load/store (no conversion, no canonicalisation), exactly as it
// already does for F32. Every other I32 layout (non-contiguous dim 1/2/3) would reach the F32-only
// kernels and stays refused, as before.
//
// Host-only (no CUDA types) so the CPU test (tests/test-concat-i32.cpp) checks the very predicate
// the CUDA backend uses.

#include "ggml.h"

#include <cstdio>
#include <cstdlib>

static inline bool pxa_cuda_concat_i32_env_on(void) {
    static const bool on = [] {
        // default ON since 10-24 (bit-exact copy paths only); =0 = the CPU round trip
        const char * e = getenv("PXA_GLM_CONCAT_I32_CUDA");
        const bool v = !e || atoi(e) != 0;
        if (v) {
            fprintf(stderr, "PXA_GLM_CONCAT_I32_CUDA=1: I32 CONCAT runs on the CUDA backend (bit-exact copy paths only)\n");
        }
        return v;
    }();
    return on;
}

// Would ggml_cuda_op_concat serve this I32 CONCAT on a copy-only path? (Independent of the switch.)
static inline bool pxa_cuda_concat_i32_shape_ok(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CONCAT || !op->src[0] || !op->src[1]) return false;
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (a->type != GGML_TYPE_I32 || b->type != GGML_TYPE_I32 || op->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(op)) return false;
    const int32_t dim = op->op_params[0];
    if (dim < 0 || dim > 3) return false;
    // path 1: both sources contiguous and the concat is two back-to-back blocks -> two memcpys
    if (ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
        (dim == 3 || (dim == 2 && op->ne[3] == 1) || (dim == 1 && op->ne[2]*op->ne[3] == 1))) {
        return true;
    }
    // path 2: dim 0, unit element stride, word-aligned rows -> the 4-byte word copy kernels
    // (grid y/z carry ne[1..3] in some of them: keep inside the 65535 limit)
    if (dim == 0 &&
        a->nb[0] == sizeof(int32_t) && b->nb[0] == sizeof(int32_t) &&
        a->nb[1] % sizeof(float) == 0 && b->nb[1] % sizeof(float) == 0 &&
        op->ne[2] < 65536 && op->ne[3] < 65536) {
        return true;
    }
    return false;
}

static inline bool pxa_cuda_concat_i32_supported(const ggml_tensor * op, bool switch_on) {
    return switch_on && pxa_cuda_concat_i32_shape_ok(op);
}
