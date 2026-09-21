// pxq_q8_launch.h -- CUDA-free launcher declarations for the int8 per-row-scale head kernels.
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

// out[M, N] = x[M, K] @ W[N, K]^T. W int8 [N, K], scale fp16 [N]. Small M (decode).
void pxq_q8_launch_mmv_f16(const int8_t * W, const void * scale, const void * x, void * out,
                           int M, int N, int K, cudaStream_t stream);

// y[N, K] fp16 = W * scale, for the prefill route (then cuBLAS).
void pxq_q8_launch_dequant_f16(const int8_t * W, const void * scale, void * y,
                               int N, int K, cudaStream_t stream);

// Self-test: synthetic int8 weights and per-row scales, decoded and multiplied on the host
// from the format spec, required to match the device BIT-EXACTLY for the dequant and within
// the stated fp32 contraction allowance for the GEMV. 0 = pass.
int pxq_q8_selftest();
