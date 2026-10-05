// pxq-cpu.h — CPU panel-dequant + slow-but-correct matmul fallbacks for the PXQ slab types
// (A5: CPU PXQ panel-dequant fallback, 2026-07-21).
//
// The PXQ tensor types (PXQ4 252, PXQ4HQ 253, PXQ2 254, PXQ3 255; the retired legacy ids
// 250 + 251 were removed 2026-07-21, and the 5-bit PXQ6 id 256 has no CPU fallback yet)
// are 64-row PANEL-interleaved CUDA-consumer formats: there is no
// per-row CPU codec (a ggml to_float/vec_dot gets a single row pointer, but a PXQ row's
// bytes are scattered across the slabs of its 64-row panel), so their type_traits
// to_float/from_float/vec_dot stay NULL on purpose. These entry points instead operate
// on a WHOLE 2D matrix (or expert slice) base pointer, where the panel geometry is known,
// and let the CPU backend run partial-offload (--cpu-moe / -ngl < 99) instead of hitting
// GGML_ABORT in ggml.c's fused-MoE / mul_mat(_id) paths.
//
// Layout ground truth: src/pxq{2,3,6}-quantize.inc.cpp + ggml/include/ggml-pxq{2,3,6}-tables.h.
// This is a COMPATIBILITY fallback: correct and coherent, not fast, and not required to
// be bit-exact with the CUDA GEMM kernels (which snap products to fp16 inside the MMA).
// The dequant itself IS the parity-locked contract (fp32 eff/book products).

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// row mapping entry, layout-identical to ggml.c's local `struct mmid_row_mapping`
// and iqk_common.h's `mmid_row_mapping` (i1 = expert-slot/route index, i2 = token row).
struct pxa_pxq_rowmap {
    int32_t i1;
    int32_t i2;
};

// true for the PXQ slab types this CPU fallback can decode
bool pxa_pxq_is_cpu_supported(enum ggml_type type);

// dequant one row (global row index into the panel-interleaved 2D matrix) to k floats.
// data = base of the 2D [k x nrows] slice (e.g. expert base = tensor->data + e*nb[2]).
void pxa_pxq_dequant_row(enum ggml_type type, const void * data, int64_t row, int64_t k, float * dst);

// dequant a whole [k x nrows] 2D slice (row-major f32 out, dst[row*k + col]).
// nrows must be a multiple of 64 and k a multiple of 32 (quantizer eligibility mirrors
// this: pxq*_tensor_eligible requires ne[1] % 64 == 0 && ne[0] % 32 == 0; the CUDA
// dequant kernels hard-abort on the same condition).
void pxa_pxq_dequant_2d(enum ggml_type type, const void * data, float * dst, int64_t nrows, int64_t k);

// fused up/gate MoE fallback == CPU re-implementation of iqk_moe_fused_up_gate for PXQ
// weights (see iqk_mul_mat.cpp MulMat::mul_mat_up_gate_NxM for the mirrored semantics):
//   act  = unary(gate_row . x + gate_b);        if (limit > 1e-6) act = min(act, limit)
//   up   = up_row . x + up_b
//   if (unary == SWIGLU_OAI) up = 1 + clamp(up, -7, 7)   else if (limit > 1e-6) up = clamp(up, -limit, limit)
//   dst[row] = up * act
// up/gate may be DIFFERENT PXQ types (the PXQ-UNIVERSAL mixed-pair case).
// src1f = f32 activations base; rows != NULL: routed-row mode, for iy in [0,ny):
//   x   = src1f + rows[iy].i2*nb12 + (rows[iy].i1 % ne11)*nb11
//   out = dst   + rows[iy].i1*nb1  +  rows[iy].i2*nb2
// rows == NULL: dense mode, x = src1f + iy*nb11, out = dst + iy*nb1.
// Threading: src0 rows are split across [ith, nth) — call from every compute thread.
void pxa_pxq_moe_up_gate_cpu(
        enum ggml_type type_up,   const void * up,
        enum ggml_type type_gate, const void * gate,
        int64_t nr0, int64_t k,
        const float * up_bias, const float * gate_bias,   // per-src0-row f32 biases or NULL
        const char * src1f, size_t nb11, size_t nb12,
        char * dst, size_t nb1, size_t nb2,
        const struct pxa_pxq_rowmap * rows, int ne11, int64_t ny,
        int unary_op, float limit,
        int ith, int nth);

// plain matmul fallback (dense MUL_MAT slice or one expert of MUL_MAT_ID):
// dst[out_row][ix] = src0_row(ix) . x(iy), addressing as in pxa_pxq_moe_up_gate_cpu.
void pxa_pxq_mul_mat_cpu(
        enum ggml_type type, const void * a,
        int64_t nr0, int64_t k,
        const char * src1f, size_t nb11, size_t nb12,
        char * dst, size_t nb1, size_t nb2,
        const struct pxa_pxq_rowmap * rows, int ne11, int64_t ny,
        int ith, int nth);

#ifdef __cplusplus
}
#endif
