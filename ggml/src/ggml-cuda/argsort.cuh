//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_CUDA_USE_CUB
// Adapted from upstream llama.cpp ggml/src/ggml-cuda/argsort.cuh @ 82dbc4f01 (MIT).
// CUB-backed full-row argsort, used when ncols exceeds what the bitonic
// shared-memory kernel can handle (> 1024 columns).
int  argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows);
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif // GGML_CUDA_USE_CUB

// PXA_FLINT_GATE_FUSE: sigmoid -> argsort DESC -> top-k gather -> sum_rows/div in one launch (argsort.cu); false = declined,
// nothing launched.
bool ggml_cuda_sigmoid_topk_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, ggml_tensor * probs, ggml_tensor * ids,
        ggml_tensor * w, ggml_tensor * wsum, ggml_tensor * wn, int k);

void ggml_cuda_op_argsort_thresh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_grouped_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void cuda_bailingmoev2_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk);

void cuda_glm45moe_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk);

void cuda_openai_experts(ggml_backend_cuda_context & ctx, ggml_tensor * topk, ggml_tensor * softmax);
