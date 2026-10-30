#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_op_fused_rope_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2);

bool ggml_cuda_op_fused_rms_rope_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2);

bool ggml_cuda_op_rope_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2);
// PXA_ROPE_KV_FUSE: true when the fused rope launch has no input/output byte overlap (vs = the V source, or nullptr)
bool pxa_rope_fuse_disjoint(const ggml_tensor * dst1, const ggml_tensor * dst2, const ggml_tensor * vs);
// PXA_ROPE_KV_FUSE=2: the decode token's Q rope + K rope + the two f16 KV-cache stores in one launch (bit-identical)
bool ggml_cuda_op_rope_rope_kv(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2,
                               const ggml_tensor * cpy_k, const ggml_tensor * cpy_v);
