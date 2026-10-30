//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_id_cuda(
    const float * x, void * vy, const char * row_mapping, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

// For now only applicable for tensors with ne[1] = 1, ne[3] = 1, and useful if ne[2] > 1
void quantize_tensor_q8_1_cuda(const struct ggml_tensor * src, void * vy, const enum ggml_type type, cudaStream_t stream);

// PXA_VFUSE (pxa/pxa-vfuse.cuh, defined in quantize.cu): verify-width elementwise chain + q8_1 in one launch
void pxa_vf_ew_q8_1_cuda(int op, const float * a, const float * b, float * o1, float * o2, void * vy, float p0, float p1,
                         int64_t kx, int64_t kx1, int64_t kx0_padded, cudaStream_t stream);
void pxa_vf_sig_mul_add_cuda(const float * g, const float * o, const float * r, float * sig, float * mul, float * add,
                             int64_t ne0, int64_t nelem, cudaStream_t stream);
void pxa_vf_gnorm_q8_cuda(const float * x, const float * w, const float * z, float * fo, float * sig, float * mul, void * q8,
                          int64_t nrows, float eps, cudaStream_t stream);
void pxa_vf_gate_tail_cuda(const float * w, const float * x, size_t nb11, float * gout, size_t nb1, int n,
                           const float * o, const float * r, float * sig, float * mul, float * add, int64_t ne0, int64_t nelem,
                           cudaStream_t stream);
