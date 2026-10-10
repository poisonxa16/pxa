// pxqn.cuh -- the PXQN entry points of the CUDA backend (ggml-cuda.cu, reduce.cu, convert.cu, mmvq.cu call these).
// In the open engine they are thin forwards (pxqn-stub.cu) into the closed libggml-pxqn, loaded at run time next to
// libggml; without it every entry point declines (or aborts with ONE clear message where a decline is impossible),
// PXQN files are refused at load, and every other type runs exactly as before. The closed library implements them.
#pragma once

#include "../common.cuh"

#define PXQN_MMV_MAX_NY 32

bool ggml_cuda_pxqn_type(ggml_type t);

void ggml_cuda_pxqn_ctx_init(int device);

int  ggml_cuda_pxqn_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst);

int  ggml_cuda_pxqn_mul_mat_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0a, ggml_tensor * dsta,
                                 const ggml_tensor * src0b, ggml_tensor * dstb, const ggml_tensor * src1);

int  ggml_cuda_pxqn_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

int  ggml_cuda_pxqn_rht_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * rht, ggml_tensor * mm);
int  ggml_cuda_pxqn_rht_ranges_place(ggml_backend_cuda_context & ctx, const float * x, int n, const int64_t * off,
                                     const int64_t * k0, uint64_t seed, int layer, int site, ggml_tensor * cat,
                                     const ggml_tensor * mm);

bool ggml_cuda_pxqn_rht_supported(const ggml_tensor * op);
void ggml_cuda_op_pxqn_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_pxqn_rms_rht_ok(const ggml_tensor * add, const ggml_tensor * norm, const ggml_tensor * rht);
const char * ggml_cuda_pxqn_q8_lookup(int device, const ggml_tensor * src1, int64_t ne10_padded);
void ggml_cuda_pxqn_rms_rht(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * norm, ggml_tensor * rht,
                            bool want_q8 = true);

float * ggml_cuda_pxqn_xmax_buf(int device, int64_t ncols, int64_t nrows);
void    ggml_cuda_pxqn_xmax_stamp(int device, const ggml_tensor * rht);
bool    ggml_cuda_pxqn_q8sc_active(int device);
void *  ggml_cuda_pxqn_q8sc_buf(int device, int64_t ncols, int64_t * kpad);
void    ggml_cuda_pxqn_q8sc_stamp(int device, const ggml_tensor * rht);

int  ggml_cuda_pxqn_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
int  ggml_cuda_pxqn_moe_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * down, ggml_tensor * kpad);

struct pxa_ts_task;
// the fused tensor-split reduce epilogue with the PXQN tail (norm + RHT of the next site), launched by the library
void ggml_cuda_pxqn_epi_rht_launch(cudaStream_t stream, size_t smem, const float * pa, const float * pb, float * xout,
                                   const pxa_ts_task & task, int token, int * err_flag, const float * w, float * y,
                                   float * mx, int ncols, float eps, uint64_t seed, int layer, int site, int64_t k0,
                                   int push_payload, void * q8, int kpad);

// dequant converters (panel -> row-major; nrows multiple of 64, n_per_row of the slab K)
void dequantize_row_pxqn3_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn3_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn3s8_f16(const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn3s8_f32(const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn2_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn2_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn1_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn1_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4s8_f16(const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4s8_f32(const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn5_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn5_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxa4_f16   (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxa4_f32   (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
