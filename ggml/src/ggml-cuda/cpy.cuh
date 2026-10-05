#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1,  bool disable_indirection = false);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_cpy_dest_ptrs_copy(ggml_cuda_graph * cuda_graph, char ** host_dest_ptrs, const int host_dest_ptrs_size, cudaStream_t stream);

// PXA_FUSE_SIBLINGS: N consecutive same-shape CPY nodes in one launch (bit-identical, see cpy.cu).
// false = outside the merged kernel's envelope, nothing launched.
bool ggml_cuda_cpy_n(ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n, bool disable_indirection = false);

bool ggml_cuda_cpy_2(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst1, ggml_tensor * dst2, bool disable_indirection = false);

bool ggml_cuda_concat_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * dst,
        bool disable_indirection = false);
