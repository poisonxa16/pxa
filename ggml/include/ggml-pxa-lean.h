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
// gate-less chains (Nemotron-H MoE): node = relu(x) or sqr(relu(x)) over the up matmul's rows; rows [ith/nth] from x, bit-identical
GGML_API bool ggml_pxa_lean_relu_ok(const struct ggml_tensor * node);
GGML_API void ggml_pxa_lean_relu_act(struct ggml_tensor * node, int ith, int nth);
// PXA_CPU_MMID_TILED override (tests): 1 = work queue + same-order tiles (default), 0 = the old per-expert loop.
GGML_API void ggml_pxa_set_cpu_mmid_tiled(int mode);
// PXA_PXQN_CPU_MMID_Q override (tests): 1 = host PXQN experts on the work queue (default), 0 = the old per-expert call;
// and the queue's rows per item (PXA_PXQN_CPU_MMID_RC, a multiple of 8)
GGML_API void ggml_pxa_set_cpu_mmid_pxqn(int mode);
GGML_API void ggml_pxa_set_cpu_mmid_pxqn_rc(int rc);
#ifdef __cplusplus
}
#endif
