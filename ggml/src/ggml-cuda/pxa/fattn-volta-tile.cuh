#pragma once

#include "../common.cuh"

// PXA_FA_D512_VOLTA=2 -- head 512/512 on the vendored no-tensor-core tile kernel.
// Thin interface on purpose: none of the vendored macros escape into the
// dispatcher's translation unit.
bool ggml_cuda_fattn_volta_tile_d512_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_volta_tile_d512(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Called once when the lever selects this route, whether or not a node ever reaches it, so that
// the exit report exists on an armed run that served nothing and can therefore say zero.
void ggml_cuda_fattn_volta_tile_d512_arm_report(void);

// PXA_FA_D256_VOLTA_TILE -- head 256/256 on the same vendored tile kernel, for the speculative
// verify step. Same thin interface, same reason.
bool ggml_cuda_fattn_volta_tile_d256_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_volta_tile_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_fattn_volta_tile_d256_arm_report(void);
