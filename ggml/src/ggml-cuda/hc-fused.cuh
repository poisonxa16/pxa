// qwen4exp hyper-connection glue, fused (PXA_QWEN4EXP_HC_FUSED; 2026-09-25).
#include "common.cuh"

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// PXA_VFUSE bit 256: the same op with its values kept in flight (pxa/pxa-hccn-pf.cuh); false = not launched, run the op above
bool ggml_cuda_op_hc_combine_norm_pf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_gate_mix    (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// PXA_VFUSE bit 32: the same op + the q8_1 of its output into q8 (rows laid out unpadded); false = not launched
bool ggml_cuda_op_hc_gate_mix_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8);
