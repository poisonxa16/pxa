#include "kpool-score.cuh"

// PXA_GLM5NEXT: the k-pool indexer's per-pool score in one kernel.
//
// Replaced chain (src/graphs/build_glm5next.cpp):
//
//   kq    = mul_mat(k_p, q_p)          [n_pool, n_tokens, n_head]
//   kq    = cont(permute(kq,2,1,0,3))  [n_head, n_tokens, n_pool]   <-- full copy
//   score = relu(kq)
//   score = mul(score, weights)
//   score = sum_rows(score)            [1, n_tokens, n_pool]
//   score = cont(permute(score,...))   [n_pool, n_tokens, 1]
//   score = add(score, pool_mask)
//
// NUMERIC CONTRACT. k_sum_rows_f32 launches one warp per row with ncols == n_head
// <= 32, so lane h holds exactly one term and the row total is warp_reduce_sum's
// xor butterfly:
//
//   for (mask = 16; mask; mask >>= 1) x += __shfl_xor_sync(0xffffffff, x, mask, 32);
//
// Lane 0's result is, add for add,
//
//   for (off = n_head/2; off; off >>= 1) for (i < off) t[i] += t[i + off];
//
// which is what one thread does below. Reproducing it in-thread is what lets a
// thread own one output pool (and a warp own 32 CONSECUTIVE pools, so the loads
// stay coalesced) while still being BIT-IDENTICAL to the unfused chain.
//
// Lanes above n_head in the unfused kernel contribute 0.0f, and adding 0.0f is
// exact for every finite value; the only value it changes is a NaN payload, and
// the chain cannot produce one here (relu of a finite dot product). For
// n_head < 32 the loop above therefore still matches.

template <int N_HEAD>
static __global__ void k_kpool_score_f32(
        const float * __restrict__ kq,
        const float * __restrict__ w,
        const float * __restrict__ mask,
        float       * __restrict__ dst,
        const int64_t n_pool,
        const int64_t n_tok) {

    const int64_t ip = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t it = blockIdx.y;

    if (ip >= n_pool) {
        return;
    }

    const float * kp = kq + it*n_pool + ip;
    const float * wr = w  + it*N_HEAD;

    float t[N_HEAD];
#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
        const float v = kp[(int64_t) h*n_pool*n_tok];
        t[h] = fmaxf(v, 0.0f) * wr[h];   // relu_f32 uses fmaxf, so this one does too
    }

#pragma unroll
    for (int off = N_HEAD/2; off > 0; off >>= 1) {
#pragma unroll
        for (int i = 0; i < off; ++i) {
            t[i] += t[i + off];
        }
    }

    dst[it*n_pool + ip] = mask ? t[0] + mask[it*n_pool + ip] : t[0];
}

void ggml_cuda_op_kpool_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kq   = dst->src[0];
    const ggml_tensor * w    = dst->src[1];
    const ggml_tensor * mask = dst->src[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && kq->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kq) && ggml_is_contiguous(w) && ggml_is_contiguous(dst));
    GGML_ASSERT(!mask || (mask->type == GGML_TYPE_F32 && ggml_is_contiguous(mask)));

    const int64_t n_pool = kq->ne[0];
    const int64_t n_tok  = kq->ne[1];
    const int64_t n_head = kq->ne[2];

    GGML_ASSERT(kq->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == n_pool && dst->ne[1] == n_tok);
    GGML_ASSERT(n_tok <= INT32_MAX);

    const float * kq_d   = (const float *) kq->data;
    const float * w_d    = (const float *) w->data;
    const float * mask_d = mask ? (const float *) mask->data : nullptr;
    float       * dst_d  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    constexpr int block = 256;
    const dim3 blocks_num((n_pool + block - 1)/block, n_tok, 1);
    const dim3 block_dims(block, 1, 1);

    switch (n_head) {
        case  4: k_kpool_score_f32< 4><<<blocks_num, block_dims, 0, stream>>>(kq_d, w_d, mask_d, dst_d, n_pool, n_tok); break;
        case  8: k_kpool_score_f32< 8><<<blocks_num, block_dims, 0, stream>>>(kq_d, w_d, mask_d, dst_d, n_pool, n_tok); break;
        case 16: k_kpool_score_f32<16><<<blocks_num, block_dims, 0, stream>>>(kq_d, w_d, mask_d, dst_d, n_pool, n_tok); break;
        case 32: k_kpool_score_f32<32><<<blocks_num, block_dims, 0, stream>>>(kq_d, w_d, mask_d, dst_d, n_pool, n_tok); break;
        default: GGML_ABORT("kpool_score: unsupported indexer head count %d", (int) n_head);
    }

    CUDA_CHECK(cudaGetLastError());
}
