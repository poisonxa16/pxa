//
// PXA_FA_QKV_TILE (2026-09-27) -- wide flash attention over a quantized K/V cache
// on sm_60, without converting the cache to f16.
//
// WHY. On a P100 a wide head-256 node (prefill ubatch, wide verify) over a q4_0 / q8_0 cache ran on
// the tile-f16 kernel, which needs f16 K and V: the launcher converted the WHOLE cache of that
// layer to f16 on every node (or, past 128 MiB of scratch that does not fit, in 8192-token chunks
// with a fold pass per chunk). That is traffic and pool memory that grow with depth -- the same
// scratch that pushed deep-context MTP out of memory.
//
// WHAT. The kernel of PXA_FA_QKV_DIRECT (fattn-qkv-common.cuh) with a column tile of 8 queries
// (GQA 4 / 6: 32 / 48 rows per block) or 4 (GQA 8: 32 rows): quantized K/V tiles are fetched in
// 16-byte vectors and dequantized in shared memory, fp16 products with fp32 folds per quant block,
// fp32 softmax, fp16 P.V pairs folded into fp32 accumulators every 8 keys. Nothing is converted;
// the only allocation is the vec launcher's split partials (output-sized, independent of depth).
// Column tiles of 8 / 4 divide the mask's 16-row padding, so no tile reads a mask row past it.
//
// SCOPE. sm_60, head 256, K and V both q4_0 or both q8_0, GQA 4 / 6 / 8, width > 8, default
// precision (an f32-precision request keeps its route), no sinks / ALiBi / softcap.
//
// LEVER. PXA_FA_QKV_TILE=1 takes those nodes; default OFF. Parity (tests/test-fa-qkv-direct,
// P100, 2026-09-27): normalised MSE 2-17x below the tile-f16 route on all 12 wide cases and max abs
// error below it on 11, but 1.07e-4 vs 9.99e-5 on q4_0 width 512 / n_kv 16384 -- the gate is
// "max abs error <= the incumbent's", so it ships as a lever. Both routes stage Q in fp16, which
// is the shared error floor at that shape.
//

#include "fattn-qkv-tile.cuh"
#include "fattn-qkv-common.cuh"
#include "core/levers.cuh"

#include <atomic>
#include <cstdio>

namespace {

using namespace pxa_qkv;

template <ggml_type TYPE>
static void qkvt_launch_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int g) {
    switch (g) {
        case 4: launch<256, 4, 8, TYPE>(ctx, dst); return;
        case 6: launch<256, 6, 8, TYPE>(ctx, dst); return;
        case 8: launch<256, 8, 4, TYPE>(ctx, dst); return;
        default: GGML_ABORT("PXA_FA_QKV_TILE: GQA group %d not instantiated", g);
    }
}

} // namespace

bool ggml_cuda_fattn_qkv_tile_supported(const ggml_tensor * dst, int cc) {
    if (pxa_lever(PXA_LEVER_FA_QKV_TILE) == 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    if (!Q || Q->ne[1] <= 8) {
        return false;
    }
    if (((const int32_t *) dst->op_params)[3] != GGML_PREC_DEFAULT) {
        return false;
    }
    return pxa_qkv::node_group(dst, cc) != 0;
}

void ggml_cuda_flash_attn_ext_qkv_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int g = (int) (Q->ne[2] / K->ne[2]);

    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        fprintf(stderr, "PXA_FA_QKV_TILE: engaged (sm_60 wide attention reads the %s K/V cache in place, no f16 "
                        "conversion; first node width %d n_kv %d GQA %d; PXA_FA_QKV_TILE=0 reverts)\n",
                ggml_type_name(K->type), (int) Q->ne[1], (int) K->ne[1], g);
    }

    switch (K->type) {
        case GGML_TYPE_Q4_0: qkvt_launch_type<GGML_TYPE_Q4_0>(ctx, dst, g); return;
        case GGML_TYPE_Q8_0: qkvt_launch_type<GGML_TYPE_Q8_0>(ctx, dst, g); return;
        default: GGML_ABORT("PXA_FA_QKV_TILE: K/V type %s not served", ggml_type_name(K->type));
    }
}
