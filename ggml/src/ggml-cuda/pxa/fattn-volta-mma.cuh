#pragma once

#include "../common.cuh"

// sm_70 flash-attention on the vendored mainline MMA kernel (mma.sync.m8n8k4).
// Thin interface on purpose: none of the vendored macros (FATTN_KQ_STRIDE and
// friends, which this fork also defines with different values in places) escape
// into the dispatcher's translation unit.
bool ggml_cuda_fattn_volta_mma_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_volta_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// PXA_FA_D512_VOLTA: the 512/512 family, which the predicate above refuses.
// Structural predicate only -- the lever that arms it lives in fattn.cu.
bool ggml_cuda_fattn_volta_mma_d512_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_volta_mma_d512(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_fattn_volta_mma_d512_arm_report(void);
