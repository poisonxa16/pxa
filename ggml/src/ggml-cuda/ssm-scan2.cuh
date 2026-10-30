#include "common.cuh"

// PXA Nemotron-H: Mamba-2 selective scan (GGML_OP_SSM_SCAN2)
void ggml_cuda_op_ssm_scan2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
