// pxq4-rb.cuh -- PXA_PXQ4_RB: the sm_60 one-column decode GEMV for the classic PXQ4 tier (GGML_TYPE_PXQ4). Shipped
// compiled in the closed libggml-pxqn; without it PXQ4 keeps its open decode path. PXA_PXQ4_RB=0
// turns it off.
#pragma once

#include "../common.cuh"

// 0 = served, -1 = declined (nothing launched)
int  ggml_cuda_pxq4_rb_mul_mat(int device, cudaStream_t stream, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);
// FUSED_UP_GATE: dst->src[0] = up, dst->src[1] = gate, dst->src[2] = x. 0 = served, -1 = declined.
int  ggml_cuda_pxq4_rb_up_gate(int device, cudaStream_t stream, ggml_tensor * dst);
bool ggml_cuda_pxq4_rb_on();
