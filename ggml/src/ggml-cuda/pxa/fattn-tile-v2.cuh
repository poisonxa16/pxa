#pragma once

#include "../common.cuh"

// PXA_FA_TILE_V2 — an alternative tile SCHEDULE for the no-tensor-core batched flash-attention
// path. See fattn-tile-v2.cu for the whole argument. Default OFF; the shipping kernel in
// ggml/src/ggml-cuda/fattn-tile-f16.cu is not touched by this file.
// PXA_FA_TILE_V2 armed? (the lever itself lives with the other levers in pxa/pxa-enhance.cuh;
// this wrapper keeps the dispatcher's include set unchanged.)
bool ggml_cuda_fattn_tile_v2_armed();

void ggml_cuda_flash_attn_ext_tile_v2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_tile_v2_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
