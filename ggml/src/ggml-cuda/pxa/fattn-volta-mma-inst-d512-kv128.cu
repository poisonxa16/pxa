// PXA_FA_D512_KVSTEP=128 (2026-10-18): a second copy of the vendored Volta MMA flash-attention
// family, compiled under its own namespace with the 512/512 64-column row staged as 128 keys per softmax step
// and 32-wide K/V half2 loads (the vendored default is 32 keys, 128-wide loads; see fattn-mma-ml.cuh). The
// D512 kernel at 64 columns keeps the whole 512-wide V accumulator per warp in registers and spills it, so a
// deeper key step pays the spill traffic and the rescale fewer times per key. Same MMA arithmetic; the online
// softmax rescales at a different granularity, so the output is not bit-identical to the 32-key step (the
// lever value 32 restores it). Reached from ggml_cuda_flash_attn_ext_volta_mma_d512() at prefill widths only.

#define PXA_VOLTA_FA_NS  pxa_volta_fa_kv128
#define PXA_VOLTA_FA_ALT 1
#define PXA_VOLTA_D512_C64_FA 128
#define PXA_VOLTA_D512_C64_K2 32
#define PXA_VOLTA_D512_C64_V2 32
#include "volta-mma/fattn-mma-ml.cuh"
#include "fattn-volta-mma.cuh"

namespace pxa_volta_fa_kv128 {
DECL_FATTN_MMA_F16_CASE(512, 512, 8, 8);
}

void ggml_cuda_flash_attn_ext_volta_mma_d512_kv128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    pxa_volta_fa_kv128::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 8, 8>(ctx, dst);
}
