// pxq-dot.h — integer-domain PXQ x Q8 dot products for the CPU backend.
//
// PXA 2026-09-01, PXQ CPU codec phase 2. Phase 1 (pxq-cpu.c) made every PXQ tier DECODE on the
// host: panel dequant to f32, then a double-precision f32 dot. That is correct and it is what
// llama-quantize / llama-pxq-export use, but as a matmul it is the slow path twice over — it
// materialises 4 bytes per weight and it does the arithmetic in scalar doubles.
//
// This file is the fast path for the 4-bit tiers: the weights stay in their nibble codes, the
// activations are quantised to int8 once per row, and the whole 32-element block is one
// integer dot. On AVX2 the nibble -> book lookup is a single _mm256_shuffle_epi8 against a
// 16-entry int8 image of the PX16 book, so the codebook indirection costs one instruction for
// 32 weights instead of 32 table loads.
//
// ---------------------------------------------------------------------------------------------
// WHY THIS IS NOT A ggml .vec_dot
// ---------------------------------------------------------------------------------------------
// It cannot be, and that is structural, not an omission. ggml hands .vec_dot (and .to_float) a
// single ROW POINTER; a PXQ row's bytes are scattered across the slabs of its 64-row panel and
// the panel base is not recoverable from a row pointer. So the addressing here is the addressing
// the format actually has — tensor base plus a GLOBAL ROW INDEX — exactly like
// pxa_pxq_dequant_row(). The type_traits entries for the PXQ types keep .to_float /
// .from_float / .vec_dot NULL (with the PXA_NO_CPU_VEC_DOT backstops in ggml.c), unchanged by
// this file; the callers are pxa_pxq_mul_mat_cpu / pxa_pxq_moe_up_gate_cpu in pxq-cpu.c, which
// mul_mat and mul_mat_id already reach through their panel-dequant early return.
//
// ---------------------------------------------------------------------------------------------
// ACTIVATION FORMAT
// ---------------------------------------------------------------------------------------------
// Q8_0-SHAPED, 32 values per block, because a PXQ K-block is 32 values: one activation block
// lines up with one slab, so a block's two 16-element sub-scales are the only per-block floats
// the dot needs. (Q4_K in this tree pairs with block_q8_K, a 256-element superblock with an
// f32 scale and per-16 bsums; that shape exists because Q4_K's own superblock is 256 and its
// mins need the bsums. PXQ has neither — the E16-row family is symmetric, no zero point and no
// min, so there is nothing for a bsum to correct.)
//
// The scale is f32, not the ggml_half of block_q8_0, because this buffer is scratch: it is
// built per matmul call from f32 activations and never reaches a file, so paying an fp16
// round-trip on it would cost accuracy for nothing.
// ---------------------------------------------------------------------------------------------

#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PXA_PXQ_DOT_QK 32

// one 32-element activation block. 36 B; deliberately not block_q8_0 (see the header comment).
struct pxa_pxq_q8 {
    float  d;                        // dequant scale: x[j] ~= d * qs[j]
    int8_t qs[PXA_PXQ_DOT_QK];
};

// true for the tiers with an integer-domain dot here. The others (PXQ1/PXQ2/PXQ3/PXQ6) still
// go through the pxq-cpu.c dequant-and-f32-dot path; nothing breaks, it is just slower.
bool pxa_pxq_dot_supported(enum ggml_type type);

// f32 activation row -> k/32 blocks. k % 32 == 0.
void pxa_pxq_quantize_row_q8(const float * x, struct pxa_pxq_q8 * y, int64_t k);

// dot of PXQ row `row` (global row index into the panel-major tensor slice at `base`, k values)
// against the quantised activation row `y` (k/32 blocks).
//   _ref  — the scalar reference. Same integer arithmetic as the SIMD arm, no intrinsics.
//   plain — dispatches to the AVX2 arm where the build has it, otherwise to _ref.
// Both require pxa_pxq_dot_supported(type); anything else aborts.
float pxa_pxq_dot_q8_ref(enum ggml_type type, const void * base, int64_t row, int64_t k,
                         const struct pxa_pxq_q8 * y);
float pxa_pxq_dot_q8    (enum ggml_type type, const void * base, int64_t row, int64_t k,
                         const struct pxa_pxq_q8 * y);

// true when the plain entry point above is the AVX2 arm and not a redirect to _ref.
bool pxa_pxq_dot_has_simd(void);

// the int8 image of the PX16 book this file dots with, and the scale that maps it back:
//   book[c] ~= book_i8[c] * (*scale_out)
// Exposed for the self-test (tests/test-pxq-cpu-dot.cpp), which reports the quantisation error
// of the book itself separately from the activation error. Returns 16 entries.
const int8_t * pxa_pxq_dot_book_i8(float * scale_out);

#ifdef __cplusplus
}
#endif
