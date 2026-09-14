#pragma once

#include "common.cuh"

// PXA_QSA: GGML_OP_QSA_TOPK -- the block top-k as a radix SELECT, one launch per graph node.
// See ggml/include/ggml.h for the contract and ggml.c's ggml_qsa_topk_row_f32() for the
// reference algorithm this kernel reproduces, decision for decision.
void ggml_cuda_op_qsa_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
