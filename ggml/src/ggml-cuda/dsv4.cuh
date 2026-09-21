// DeepSeek-V4 fused hyper-connection ops.
//
// Transcribed from llama.cpp `ggml/src/ggml-cuda/dsv4.cu` @ upstream commit
// 44c7b01de (deepseek-v4-flash CUDA branch). Copyright (c) 2023-2026 The ggml
// authors. MIT.

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_weighted_sum (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_expand       (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
