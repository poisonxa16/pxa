#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool is_norm = false);

void ggml_cuda_op_fused_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * dst);

void ggml_cuda_op_fused_add_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * add1, ggml_tensor * add2, ggml_tensor * dst);

void ggml_cuda_op_fused_rms_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * rms1, ggml_tensor * rms2);

void ggml_cuda_op_fused_rms_rms_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// G2-F3 NORMFUSE: fused rms-norm emitting a q8_1 sidecar of its own output (bit-identical to
// norm-then-quantize_q8_1). Returns false if the shape is outside the bit-exact envelope.
bool ggml_cuda_op_fused_rms_norm_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8, int64_t ncols_padded);
