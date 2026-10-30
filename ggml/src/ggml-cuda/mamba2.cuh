#include "common.cuh"

// PXA Nemotron-H phase 2: the Mamba-2 mixer as two in-place ops (GGML_OP_MAMBA2_CONV, GGML_OP_MAMBA2_SCAN).
// Both read the recurrent state rows (masked) and write them back directly, and both keep the arithmetic of
// every value of the unfused graph, so the result is bit-identical to it.
void ggml_cuda_op_mamba2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mamba2_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// relu(x)^2 in one launch for the adjacent RELU -> SQR pair (Nemotron / Nemotron-H relu^2 FFN); writes both
// outputs, so a second reader of the RELU output still sees it. Bit-identical to relu_f32 + sqr_f32.
bool ggml_cuda_relu_sqr_fusable(const ggml_tensor * relu, const ggml_tensor * sqr);
void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu, ggml_tensor * sqr);
