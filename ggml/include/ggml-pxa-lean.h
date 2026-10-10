#pragma once
// ggml-pxa-lean.h -- PXA_XCACHE_LEAN (2026-10-06): exported pieces of the CPU MUL_MAT_ID (one expert's rows, PXQ/PXQN host
// weights) and FUSED_MUL_UNARY (a row range) kernels of ggml.c, bit-identical to what ggml_graph_compute runs for those nodes.
// rows = (i1 slot, i2 token) int32 pairs, grouped per expert as MUL_MAT_ID groups them. A header of its own (not ggml.h) so adding it
// does not touch every translation unit. Used by src/llama-pxa-xcache-async-core.cpp only.
#include "ggml.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
GGML_API bool ggml_pxa_lean_mmid_ok(const struct ggml_tensor * node);
GGML_API void ggml_pxa_lean_mmid_expert(const struct ggml_tensor * node, int expert, const int32_t * rows, int64_t ny, int ith, int nth);
GGML_API void ggml_pxa_lean_fused_mul_unary(struct ggml_tensor * node, int ith, int nth);
#ifdef __cplusplus
}
#endif
