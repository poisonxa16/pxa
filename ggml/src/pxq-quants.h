// pxq-quants.h -- the per-type quant codecs PXA owns after the phase-3 prune of the 53
// ik-only quant types (2026-08-21). Everything declared here has a real, architecture-neutral
// scalar implementation in pxq-quants.cpp; nothing here has an accelerator underneath it that
// can silently claim the call and leave the output unwritten.
//
// Provenance: relocated verbatim from ggml/src/iqk/iqk_quantize.h and .cpp, derived from work
// by Iwan Kawrakow, MIT licensed.

#pragma once

#include <stdint.h>
#include <stddef.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#ifdef __cplusplus
#define GGML_RESTRICT
extern "C" {
#else
#define GGML_RESTRICT restrict
#endif

struct quantize_user_data;
struct ggml_tensor;

// ---- MXFP4 (type 39, ours) --------------------------------------------------------------
void   quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
void   quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * user_data);
void   dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_mxfp4_q8_0_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// ---- q8_0_x4 (type 97): the activation layout q1_0_g128's dot actually reads ---------------
void   quantize_row_q8_0_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// ---- q8_KV (type 151): KV cache selectable with -ctk / -ctv q8_KV --------------------------
void   quantize_row_q8_KV_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void   quantize_row_q8_KV(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q8_KV(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * user_data);
void   dequantize_row_q8_KV(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q8_KV_q8_KV(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// ---- q1_0_g128 (type 41): Bonsai 1-bit files ----------------------------------------------
void   quantize_row_q1_0_g128_ref(const float * GGML_RESTRICT x, block_q1_0_g128 * GGML_RESTRICT y, int64_t k);
void   quantize_row_q1_0_g128(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q1_0_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * user_data);
void   dequantize_row_q1_0_g128(const block_q1_0_g128 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q1_0_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// ---- I2_S (type 36): Microsoft BitNet files -----------------------------------------------
void   dequantize_row_ms_i2s(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// ---- generic conversion + validation -------------------------------------------------------
typedef void (*pxa_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
typedef void (*pxa_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);

// dequantise a row to f32 and requantise it into to_type. Replaces iqk_quantize_any(); the
// row-interleave factor it used to look up is gone with the repacked types, so it is 1.
void pxa_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * GGML_RESTRICT x, void * GGML_RESTRICT y, void * work_buffer,
                      pxa_to_float_t to_float, pxa_from_float_t from_float, int ith, int nth);

// --check-tensors / validate_quants: scan a tensor's fp16 scale fields for NaN/inf.
bool pxa_validate_tensor(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
