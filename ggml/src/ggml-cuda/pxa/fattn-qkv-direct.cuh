#pragma once

#include "../common.cuh"

// PXA_FA_QKV_DIRECT -- narrow flash attention (decode width 1 and speculative-verify widths 2..8)
// over a q4_0 or q8_0 K/V cache on sm_60, reading the quantized cache straight into shared memory
// and dequantizing there, with every query head of one GQA group and every query column handled by
// ONE block, so each K/V byte is read from memory once per group instead of once per (head, column).
// See fattn-qkv-direct.cu and fattn-qkv-common.cuh for the design.
//
// A dispatch-side SUBSTITUTION for the vec route: the support answer does not change.
bool ggml_cuda_fattn_qkv_direct_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_qkv_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
