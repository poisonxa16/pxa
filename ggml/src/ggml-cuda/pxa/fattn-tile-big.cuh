#pragma once

#include "../common.cuh"

// PXA_FA_TILE_512 — a no-tensor-core flash-attention tile kernel for the LARGE head sizes
// (512/512 and 576/512) on sm_60 and sm_70. See fattn-tile-big.cu for the whole argument.
// Default OFF; the shipping tile kernels (ggml/src/ggml-cuda/fattn-tile-f16.cu, fattn-tile-f32.cu)
// serve 64/128/256 and are not touched by this file.

// PXA_FA_TILE_512 armed?
bool ggml_cuda_fattn_tile_big_armed();

// PXA_FA_TILE_512_FP32 — the all-fp32 score tile (diagnostic, slower). Only read when armed.
bool ggml_cuda_fattn_tile_big_fp32_soft();

// Does this node have a shape/dtype/op_params combination this kernel serves on this device?
bool ggml_cuda_fattn_tile_big_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_tile_big(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
