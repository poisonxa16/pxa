//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"

#define CUDA_MULTI_ADD_BLOCK_SIZE 256

void ggml_cuda_op_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_mul_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// G2-F4 ADDFUSE: mul_multi_add with the following residual ADD folded into the epilogue.
// Writes the ADD node's output; experts are summed first, the residual is added LAST
// (fp32 add is bitwise commutative => bit-identical to the standalone k_add_same).
void ggml_cuda_op_mul_multi_add_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * add);
