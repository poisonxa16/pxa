#pragma once

#include "../common.cuh"

// PXA_FA_QKV_TILE -- wide flash attention (prefill ubatch, wide speculative verify: query width > 8)
// over a q4_0 or q8_0 K/V cache on sm_60. The K/V tiles are read in place and dequantized in shared
// memory (the same data path as PXA_FA_QKV_DIRECT, fattn-qkv-common.cuh), so the node makes no f16
// copy of the cache: no whole-cache or chunked conversion and no scratch that grows with depth.
//
// A dispatch-side SUBSTITUTION for the tile-f16 route (which converts the cache first): the support
// answer does not change, and every shape it declines keeps its route.
bool ggml_cuda_fattn_qkv_tile_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_qkv_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
