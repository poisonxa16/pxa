// pxq-cpu.h — CPU panel-dequant + slow-but-correct matmul fallbacks for the PXQ slab types
// (A5: CPU PXQ panel-dequant fallback, 2026-07-21).
//
// The PXQ tensor types (PXQ1 248, PXQ4 252, PXQ4HQ 253, PXQ2 254, PXQ3 255, PXQ6 256; the
// retired legacy ids 250 + 251 were removed 2026-07-21) are 64-row PANEL-interleaved
// CUDA-consumer formats: there is no
// per-row CPU codec (a ggml to_float/vec_dot gets a single row pointer, but a PXQ row's
// bytes are scattered across the slabs of its 64-row panel), so their type_traits
// to_float/from_float/vec_dot stay NULL on purpose. These entry points instead operate
// on a WHOLE 2D matrix (or expert slice) base pointer, where the panel geometry is known,
// and let the CPU backend run partial-offload (--cpu-moe / -ngl < 99) instead of hitting
// GGML_ABORT in ggml.c's fused-MoE / mul_mat(_id) paths.
//
// Layout ground truth: src/pxq{1,2,3,6,6r}-quantize.inc.cpp + ggml/include/ggml-pxq{1,2,3,6}-tables.h.
// Every tier decodes here as of 2026-09-01 (PXQ1 + PXQ6 completed), which is what lets
// llama-quantize requantize FROM a PXQ source and llama-pxq-export run without a GPU.
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

// the live numeric tables, after the frozen-header defaults and any PXA_PXQ*_BOOK / _SUB env
// override have been applied. Any of the three out-pointers may be NULL. Calling this is what
// forces the (idempotent) table init, so a consumer that needs the tables and nothing else --
// pxq-dot.c builds its int8 book image from the PX16 book here -- can ask for them directly
// instead of duplicating the env parsing and drifting from the dequant path.
//   book16 : 16-entry PX16 book   (PXQ4 / PXQ4-HQ codes)
//   sub16  : 16-entry SUB16 subs  (every tier's per-16-element sub-scale)
//   sub8   : 16-entry SUB8 subs   (PXQ4-HQ's per-8-element sub-scale)
void pxa_pxq_float_tables(const float ** book16, const float ** sub16, const float ** sub8);

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

// MXFP4 dot product against STOCK q8_0 activations (PXA 2026-08-20).
//
// MXFP4 is a mainline llama.cpp format, but this fork inherited its CPU vec_dot from ik --
// where the non-ik body was commented out entirely, so the function returned WITHOUT ever
// assigning *s and the caller read uninitialized stack (harness: "mxfp4 dot product FAILED
// (inf)"; inference: non-finite logits). Our own PXQ4 files carry mxfp4 tensors, so we owned
// the exposure without owning the code.
//
// This is the PXA implementation and it takes PLAIN block_q8_0 activations, not ik's x4
// interleave -- so the mxfp4 path has no ik dependency of any kind.
void pxa_mxfp4_dot_q8_0(int n, float * s, const void * vx, const void * vy);

// Q6_0 (4-bit nibble + 2-bit high plane) x STOCK block_q8_0 activations.
// ggml_vec_dot_q6_0_q8_0 in ggml-quants.c was never written -- below its ik
// early-return it read "// TODO / *s = 0;", so every non-ik build silently
// scored every Q6_0 row as zero (test-quantize-fns: dot product error 1.090274,
// which is just the norm of the reference). Layout taken from dequantize_row_q6_0.
void pxa_q6_0_dot_q8_0(int n, float * s, const void * vx, const void * vy);

// Q8_KV x Q8_KV -- the KV-cache int8 row format selected by -ctk q8_KV / -ctv q8_KV.
// vec_dot_q8_KV_q8_KV had no body below its ik early-return, so with the accelerated
// matmul compiled out it asserted and returned WITHOUT ever assigning *s: the generic
// flash-attention loop then fed an uninitialized stack float into the online softmax,
// and the generic mul_mat chunk loop fed it stale tmp[]. A silently wrong KV cache, not
// a crash. Row layout (see pxq-cpu.c) is a flat header, not a block struct.
void pxa_q8_KV_dot_q8_KV(int n, float * s, const void * vx, const void * vy);

void pxa_pxq_mul_mat_cpu(
        enum ggml_type type, const void * a,
        int64_t nr0, int64_t k,
        const char * src1f, size_t nb11, size_t nb12,
        char * dst, size_t nb1, size_t nb2,
        const struct pxa_pxq_rowmap * rows, int ne11, int64_t ny,
        int ith, int nth);

// -------------------------------------------------------------------------------------------------
// Graph ops and helpers reowned from ggml/src/iqk/iqk_cpu_ops.cpp (PXA 2026-08-21, ik
// separation phase 2). See the block comment in pxq-cpu.c for the per-function provenance.
//
// Three of these are FORK OPS with no stock ggml body anywhere in the tree -- MUL_MULTI_ADD,
// HADAMARD and FUSED_RMS_RMS_ADD. They are not accelerators with a fallback beneath them;
// they are the whole implementation, which is why they had to be moved rather than dropped.
// The other three are a MoE fusion body (SUM_ROWS+DIV) and two helpers that are not ggml ops
// at all: pxa_has_fancy_simd is a logging predicate for llama.cpp, and pxa_exp_with_thresh is
// llama-sampling's adaptive-p accumulator.
//
// ARGSORT and GROUPED_TOPK live in pxq-cpu-ops.cpp instead -- they need std::partial_sort.
//
// All of these are called from every compute thread with a row split; see pxq-cpu.c.
// -------------------------------------------------------------------------------------------------

// true when the host has the AVX512 feature set (F + VNNI + VL + BW + DQ). Logging only.
bool pxa_has_fancy_simd(void);

// SUM_ROWS+DIV fused: div[ir][j] = src[ir][j] / sum_j(src[ir][j]), 0 when the row sums to <= 0.
void pxa_sumrows_div(struct ggml_tensor * div, int ith, int nth);

// GGML_OP_MUL_MULTI_ADD -- weighted sum of ne01 expert outputs, optionally with a second
// per-expert scale drawn from src[2] via the ids in src[3]. No stock ggml body exists.
void pxa_mul_multi_add(struct ggml_tensor * dst, int ith, int nth);

// GGML_OP_HADAMARD -- in-place fast Walsh-Hadamard transform over every op_params[0]-wide
// chunk of every row, f32 or any quantized source with a to_float. No stock ggml body exists.
void pxa_hadamard(struct ggml_tensor * dst, int ith, int nth);

// GGML_OP_FUSED_RMS_RMS_ADD -- two independent RMS norms of the same shape, added.
// src0/src2 are f32/f16/bf16 and share a type; src1/src3 are f32 gain rows. No stock body.
void pxa_rms_rms_add(struct ggml_tensor * dst, int ith, int nth);

// llama-sampling adaptive-p: rewrite logits[] in place as exp(logit - max) for logits >= min
// (zero otherwise) and return the sum. Not a ggml op.
float pxa_exp_with_thresh(int n, float * logits, float max, float min);

#ifdef __cplusplus
}
#endif
