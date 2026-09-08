#include "fattn-volta-mma.cuh"
#include "volta-mma/fattn-mma-ml.cuh"

#include <cstring>

// PXA_FA_MMA_VOLTA (2026-09-02) -- sm_70 flash-attention on mainline llama.cpp's
// MMA kernel, vendored under volta-mma/ (see the provenance headers there).
//
// WHY: nvprof on the 2xV100 pair, Qwable-27B-PXQ4 at a 20801-token prefill,
// showed both engines issuing the SAME 352 flash-attention calls -- 25.17 s here
// on this fork's legacy nvcuda::wmma kernel against 9.22 s in mainline, a 2.73x
// deficit that accounted for the entire GPU-compute gap (61.1 s vs 45.8 s).
// mainline's Volta kernel uses the m8n8k4 tensor-core instruction directly and,
// via ncols2, reads each K/V row once for several of the Q heads that share it.
//
// SCOPE:
//   * cc == 7.0 only. Every other arch keeps its existing dispatch untouched.
//   * head sizes 128 and 256 (DKQ == DV). 128 covers the MoE and P100-seat
//     models if they are ever run on Volta; 256 is this rig's model.
//   * f16 K and V only. mainline's launch_fattn stages a quantized KV cache into
//     scratch obtained from ggml_cuda_flash_attn_ext_get_f16_extra_data(), which
//     is backed by mainline's ggml_cuda_flash_attn_ext_get_alloc_size() hook in
//     its CUDA backend. This fork has no such hook, so a non-f16 KV cache would
//     hand the kernel an unallocated pointer. f16 is what this rig runs, so the
//     predicate requires it rather than porting the allocator.
//   * batch > 8 only; decode keeps the incumbent path until measured.
// Anything declined falls through to the tile route, which is itself a large win
// over the WMMA kernel, so this can only change which working kernel runs.

// mainline fattn.cu:183-194. ncols2 > 1 packs several Q heads into one K/V read
// and is only valid with a mask, no ALiBi, K padded to FATTN_KQ_STRIDE, and
// 16-byte-aligned higher-dimensional strides. Otherwise ncols2 must stay 1.
static bool pxa_volta_mma_use_gqa_opt(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % 256 == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }
    return use_gqa_opt;
}

bool ggml_cuda_fattn_volta_mma_supported(const ggml_tensor * dst, int cc) {
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) {
        return false;
    }
    const int64_t DKQ = Q->ne[0];
    if (DKQ != 128 && DKQ != 256) {
        return false; // only these two families are instantiated
    }
    if (K->ne[0] != DKQ || V->ne[0] != DKQ) {
        return false; // DKQ == DV only
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false; // see SCOPE above
    }
    if (!mask) {
        return false; // the vendored kernel's KV_max bound reads the mask
    }
    if (mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false; // mainline rejects these in ggml_cuda_get_best_fattn_kernel
    }
    if (dst->src[4]) {
        return false; // attention sinks: not exercised here, do not claim it
    }
    if (K->ne[1] % 256 != 0) {
        return false; // FATTN_KQ_STRIDE padding the vendored launcher assumes
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    // NB: the batch-size gate lives in pxa_fa_mma_volta_take() in fattn.cu, so
    // that PXA_FA_MMA_VOLTA=2 can lift it for a decode measurement without this
    // structural predicate having to know about the lever.
    return true;
}

// mainline's ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1, specialised for
// Volta: turing_mma_available() is false there, so the 8/ncols2 rung is skipped.
template <int DKQ, int DV, int ncols2>
static void pxa_volta_mma_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] <= 16/ncols2) {
        pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }
    if (Q->ne[1] <= 32/ncols2) {
        pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }
    pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

// mainline's switch_ncols2, Volta branch (fattn.cu:200-222). Upstream comment:
// "On Volta the GQA optimizations aren't as impactful vs. minimizing wasted
// compute", hence the descending 8/4/2/1 ladder rather than a fixed ncols2.
template <int DKQ, int DV>
static void pxa_volta_mma_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const bool use_gqa_opt = pxa_volta_mma_use_gqa_opt(dst);
    const int  gqa_ratio   = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        pxa_volta_mma_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        pxa_volta_mma_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        pxa_volta_mma_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }
    pxa_volta_mma_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_volta_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[0]) {
        case 128: pxa_volta_mma_switch_ncols2<128, 128>(ctx, dst); return;
        case 256: pxa_volta_mma_switch_ncols2<256, 256>(ctx, dst); return;
        default:  GGML_ABORT("PXA: unreachable, guarded by ggml_cuda_fattn_volta_mma_supported");
    }
}
