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

// The companion question, and the one that makes PXA_DSA_ATTN mean the same thing at
// every head size: does this node ASK for sparse attention at all?
//
// src[5] is a request, not a hint. A node that carries an index list wants the listed KV
// rows attended and nothing else; every dense CUDA kernel ignores src[5] and attends
// everything the mask admits. Those two answers differ whenever the list is narrower
// than the visible set -- a wrong answer, not a slow one, and a silent one. It is the
// same class of defect that ggml_cuda_dsa_attn_supported() already declines for ALiBi
// and logit softcapping, and it must be declined for the same reason.
//
// Without this the switch only APPEARS to hold. At head 384/512 no dense CUDA kernel is
// instantiated on sm_60/sm_70, so an index-list node is refused whatever the switch
// says; at head 128/256 a dense kernel exists, accepts, and quietly drops the request.
// The rule below is what the comment above ggml_cuda_dsa_attn_supported()'s call sites
// always claimed -- "nodes WITHOUT src[5] fall through unchanged" -- stated as code.
bool ggml_cuda_dsa_attn_requested(const ggml_tensor * dst);

void ggml_cuda_dsa_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
