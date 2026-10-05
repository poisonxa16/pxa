// qwen4exp hyper-connection glue, fused (PXA_QWEN4EXP_HC_FUSED; 2026-09-25).
#include "common.cuh"

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_gate_mix    (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
