//
// PXA_FA_QKV_DIRECT (2026-09-27, lanes qkv-attn / hp-qkv-attn) -- narrow flash attention over a
// quantized K/V cache on sm_60, reading the cache straight into shared memory.
//
// WHY. On a P100 a narrow attention node (decode width 1, speculative-verify widths 2..8) over a
// q4_0 / q8_0 cache ran on the fp32 vec kernel, which gives one block to ONE query head and ONE
// query column. Under grouped-query attention (Qwen3.8-27B: 24 query heads over 4 KV heads, a
// group of 6) the same K/V bytes were therefore fetched 6x at width 1 and 30x at width 5, and every
// fetch dotted the key against a q8_1-requantized query with an emulated dp4a (sm_60 has none).
// At 16k-128k that is the attention half of a deep decode or verify step.
//
// WHAT. One block per (column tile, KV head, KV split) serves every query head of the group and
// every query column of the tile; the kernel and its data path are in fattn-qkv-common.cuh
// (16-byte tile fetch with a register prefetch of the next tile, in-kernel byte-permute dequant,
// fp16 products with fp32 folds, fp32 softmax and accumulators). The split / combine machinery is
// the vec route's own launcher.
//
// Column tiles: width w runs as one tile of w columns (R = G*w rows), except GQA 8 at widths 7
// and 8, which run as two tiles of 4 so the block's shared memory stays under 48 KB.
//
// LEVER. PXA_FA_QKV_DIRECT: dispatch-side substitution for the vec route (the support answer is
// unchanged). On at the ENHANCE level; =0 restores the vec-f32 route.
//

#include "fattn-qkv-direct.cuh"
#include "fattn-qkv-common.cuh"
#include "core/levers.cuh"

#include <atomic>
#include <cstdio>
#include <cstring>

namespace {

using namespace pxa_qkv;

template <ggml_type TYPE>
static void qkvd_launch_g6(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int nc) {
    switch (nc) {
        case 1: launch<256, 6, 1, TYPE>(ctx, dst, true); return;
        case 2: launch<256, 6, 2, TYPE>(ctx, dst, true); return;
        case 3: launch<256, 6, 3, TYPE>(ctx, dst, true); return;
        case 4: launch<256, 6, 4, TYPE>(ctx, dst, true); return;
        case 5: launch<256, 6, 5, TYPE>(ctx, dst, true); return;
        case 6: launch<256, 6, 6, TYPE>(ctx, dst, true); return;
        case 7: launch<256, 6, 7, TYPE>(ctx, dst, true); return;
        case 8: launch<256, 6, 8, TYPE>(ctx, dst, true); return;
        default: GGML_ABORT("PXA_FA_QKV_DIRECT: width %d outside 1..8", nc);
    }
}

template <ggml_type TYPE>
static void qkvd_launch_g4(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int nc) {
    switch (nc) {
        case 1: launch<256, 4, 1, TYPE>(ctx, dst, true); return;
        case 2: launch<256, 4, 2, TYPE>(ctx, dst, true); return;
        case 3: launch<256, 4, 3, TYPE>(ctx, dst, true); return;
        case 4: launch<256, 4, 4, TYPE>(ctx, dst, true); return;
        case 5: launch<256, 4, 5, TYPE>(ctx, dst, true); return;
        case 6: launch<256, 4, 6, TYPE>(ctx, dst, true); return;
        case 7: launch<256, 4, 7, TYPE>(ctx, dst, true); return;
        case 8: launch<256, 4, 8, TYPE>(ctx, dst, true); return;
        default: GGML_ABORT("PXA_FA_QKV_DIRECT: width %d outside 1..8", nc);
    }
}

template <ggml_type TYPE>
static void qkvd_launch_g8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int nc) {
    switch (nc) {
        case 1: launch<256, 8, 1, TYPE>(ctx, dst, true); return;
        case 2: launch<256, 8, 2, TYPE>(ctx, dst, true); return;
        case 3: launch<256, 8, 3, TYPE>(ctx, dst, true); return;
        case 4: launch<256, 8, 4, TYPE>(ctx, dst, true); return;
        case 5: launch<256, 8, 5, TYPE>(ctx, dst, true); return;
        case 6: launch<256, 8, 6, TYPE>(ctx, dst, true); return;
        case 7:
        case 8: launch<256, 8, 4, TYPE>(ctx, dst, true); return;   // two column tiles of 4 (R = 64 does not fit)
        default: GGML_ABORT("PXA_FA_QKV_DIRECT: width %d outside 1..8", nc);
    }
}

template <ggml_type TYPE>
static void qkvd_launch_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int g, int nc) {
    switch (g) {
        case 4: qkvd_launch_g4<TYPE>(ctx, dst, nc); return;
        case 6: qkvd_launch_g6<TYPE>(ctx, dst, nc); return;
        case 8: qkvd_launch_g8<TYPE>(ctx, dst, nc); return;
        default: GGML_ABORT("PXA_FA_QKV_DIRECT: GQA group %d not instantiated", g);
    }
}

} // namespace

bool ggml_cuda_fattn_qkv_direct_supported(const ggml_tensor * dst, int cc) {
    if (pxa_lever(PXA_LEVER_FA_QKV_DIRECT) == 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    if (!Q || Q->ne[1] < 1 || Q->ne[1] > 8) {
        return false;
    }
    return pxa_qkv::node_group(dst, cc, pxa_lever(PXA_LEVER_FA_QKV_DIRECT_VOLTA) != 0) != 0;
}

void ggml_cuda_flash_attn_ext_qkv_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int g  = (int) (Q->ne[2] / K->ne[2]);
    const int nc = (int) Q->ne[1];

    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        fprintf(stderr, "PXA_FA_QKV_DIRECT: engaged (narrow attention reads the %s K/V cache in place, "
                        "GQA group %d per block; first node width %d n_kv %d; PXA_FA_QKV_DIRECT=0 reverts)\n",
                ggml_type_name(K->type), g, nc, (int) K->ne[1]);
    }

    switch (K->type) {
        case GGML_TYPE_Q4_0: qkvd_launch_type<GGML_TYPE_Q4_0>(ctx, dst, g, nc); return;
        case GGML_TYPE_Q8_0: qkvd_launch_type<GGML_TYPE_Q8_0>(ctx, dst, g, nc); return;
        default: GGML_ABORT("PXA_FA_QKV_DIRECT: K/V type %s not served", ggml_type_name(K->type));
    }
}
