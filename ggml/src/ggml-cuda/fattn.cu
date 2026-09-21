//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "pxa/fattn-volta-mma.cuh"
#include "pxa/fattn-volta-tile.cuh"
#include "pxa/fattn-tile-v2.cuh"
#include "pxa/fattn-tile-big.cuh"
#include "pxa/core/fa-route.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn.cuh"
#include "convert.cuh"
#include "dsa_attn.cuh"

// Which kernel runs a FLASH_ATTN_EXT node is decided in ggml-cuda/pxa/core/fa-route.cu and
// nowhere else. This file used to carry that decision twice -- once here and once in
// ggml_cuda_fattn_is_supported() -- with "must mirror exactly" comments in place of a compiler
// check, plus nine PXA_* levers read inline. Both are gone: this is an upstream-shaped file
// again, it reads no environment variable, and the two entry points below are the same question
// asked with a different verb.

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    // The window as the router sees it, taken before any rewrite: the SWA slice zeroes
    // op_params[4] on the node it hands the kernel, and routing has always used the original.
    const int32_t n_swa = ((const int32_t *) dst->op_params)[4];

    pxa_fa_scratch_t scratch;
    ggml_tensor * node = pxa_fa_prepare_node(dst, scratch);

    const pxa_fa_query_t  q    = { node, n_swa, PXA_FA_QUERY_DISPATCH };
    const pxa_fa_plan_t   plan = pxa_fa_plan_node(ctx, q);

    pxa_fa_route_census(ctx.device, plan.route);

    switch (plan.route) {
        case PXA_FA_ROUTE_DSA:       ggml_cuda_dsa_attn_ext(ctx, node);                   return;
        case PXA_FA_ROUTE_VEC_F16:   ggml_cuda_flash_attn_ext_vec_f16(ctx, node);         return;
        case PXA_FA_ROUTE_VEC_F32:   ggml_cuda_flash_attn_ext_vec_f32(ctx, node);         return;
        case PXA_FA_ROUTE_TILE_F16:  ggml_cuda_flash_attn_ext_tile_f16(ctx, node);        return;
        case PXA_FA_ROUTE_TILE_F32:  ggml_cuda_flash_attn_ext_tile_f32(ctx, node);        return;
        case PXA_FA_ROUTE_TILE_V2:   ggml_cuda_flash_attn_ext_tile_v2(ctx, node);         return;
        case PXA_FA_ROUTE_TILE_BIG:  ggml_cuda_flash_attn_ext_tile_big(ctx, node);        return;
        case PXA_FA_ROUTE_D512_MMA:  ggml_cuda_flash_attn_ext_volta_mma_d512(ctx, node);  return;
        case PXA_FA_ROUTE_D512_TILE: ggml_cuda_flash_attn_ext_volta_tile_d512(ctx, node); return;
        case PXA_FA_ROUTE_D256_TILE: ggml_cuda_flash_attn_ext_volta_tile_d256(ctx, node); return;
        case PXA_FA_ROUTE_VOLTA_MMA: ggml_cuda_flash_attn_ext_volta_mma(ctx, node);       return;
        case PXA_FA_ROUTE_WMMA_F16:  ggml_cuda_flash_attn_ext_wmma_f16(ctx, node);        return;
        case PXA_FA_ROUTE_MMA_F16:   ggml_cuda_flash_attn_ext_mma_f16(ctx, node);         return;
        case PXA_FA_ROUTE_MMA_NEW:   ggml_cuda_flash_attn_ext_mma_new(ctx, node);         return;

        case PXA_FA_ROUTE_DSA_UNSERVED:
            // The node asked for sparse attention and the sparse kernel declined it. Falling
            // through to a dense kernel would hand it to code that ignores src[5], and would do
            // it silently. Unreachable in both configurations of the switch: with PXA_DSA_ATTN
            // off no graph in this tree builds an index list at all, and with it on
            // build_dsv4_attn_mha() mirrors this predicate before emitting one.
            GGML_ABORT("FLASH_ATTN_EXT carries a DSA index list (src[5]) that ggml_cuda_dsa_attn_supported() "
                       "declined, and no dense CUDA kernel honours src[5]. Set PXA_DSA_ATTN=1, or do not "
                       "attach an index list to this node.");

        default:
            GGML_ABORT("the flash-attention planner returned no route for a node the backend accepted");
    }
}

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const pxa_fa_query_t q = { dst, ((const int32_t *) dst->op_params)[4], PXA_FA_QUERY_SUPPORT };
    return pxa_fa_plan_node(ctx, q).supported;
}
