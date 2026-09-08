#include "common.cuh"

// DSA = "DeepSeek sparse attention": a FLASH_ATTN_EXT node that carries a per-query
// index list in src[5] (produced by GGML_OP_MASK_TO_IDX) attends only the listed KV
// rows. Head-dim agnostic and arch-neutral (cuBLAS fp16 GEMM + plain kernels), so it
// is the only CUDA attention path that runs at head 512 on sm_70.
//
// ggml_cuda_dsa_attn_supported() is the SINGLE predicate used by both
// ggml_cuda_flash_attn_ext() (dispatch) and ggml_cuda_fattn_is_supported() (the
// scheduler gate). They cannot disagree because they call the same function.
bool ggml_cuda_dsa_attn_supported(const ggml_tensor * dst, int cc);

void ggml_cuda_dsa_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
