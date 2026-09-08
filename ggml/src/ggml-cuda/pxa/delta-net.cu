// -----------------------------------------------------------------------------
// Vendored from ik_llama.cpp, commit 3c58ae37, file
// ggml/src/ggml-cuda/delta-net.cu -- the DeltaNet linear-attention kernel for
// Qwen3-Next. First imported here 2026-07-17.
//
// ik_llama.cpp and the llama.cpp / ggml it derives from are MIT licensed:
//   MIT License, Copyright (c) 2023-2026 the ik_llama.cpp authors
//                Copyright (c) 2023-2026 The ggml authors
//   (full text: the LICENSE file at the root of each repository)
//
// The recurrence itself is upstream's and is unmodified. PXA's edits are the
// state-destination redirects that the DeltaNet decode fusion drives, and they
// are the only reason this file is not byte-identical to upstream's:
//   * ggml_cuda_op_delta_net_ex   -- PXA_FUSE_DELTANET, writes the new ssm state
//                                    into a caller-supplied destination instead
//                                    of through a separate CONCAT.
//   * ggml_cuda_op_delta_net_ex2  -- PXA_DN_SCATTER_FUSE, resolves the
//                                    destination ROW on the device from the
//                                    index tensor the fused-away SET_ROWS would
//                                    have read.
// Both take nullptr to mean "do exactly what upstream does". Every edited site
// names the lever it belongs to in a comment; see ggml-cuda/pxa/README.md and
// pxa-deltanet-fuse.cuh for what drives them.
// -----------------------------------------------------------------------------
#include "../common.cuh"
#include "delta-net.cuh"
#include <cstdlib>
#include <cstring>

// Delta Net Linear Attention Kernel for Qwen3-Next (HEAD_DIM=128)
// State layout: [S_v, S_v*H_v, 1, n_seqs] (column-major)

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <int block_size>
__device__ __forceinline__ float reduce_sum(float x, float * s) {
    x = warp_reduce_sum(x);
    if constexpr (block_size > WARP_SIZE) {
        //__shared__ float s[block_size/WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s[warp_id] = x;
        }
        __syncthreads();
        x = lane_id < block_size/WARP_SIZE ? s[lane_id] : 0.0f;
        x = warp_reduce_sum(x);
    }
    return x;
}

template <int HEAD_DIM, int block_size>
__global__ void delta_net_recurrent_f32(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,         // [n_tokens, 1, n_heads, n_seqs]
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * __restrict__ state_in,  // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    float * __restrict__ dst,             // output + new_state(s) concatenated
    float * __restrict__ saved_states,
    float * __restrict__ state_ext,       // PXA_FUSE_DELTANET: optional external new-state destination
    const int32_t * __restrict__ state_ext_idx, // PXA_DN_SCATTER_FUSE: optional DEVICE-side row index
    const int64_t state_ext_row_stride,         //   (elements) row stride of the destination buffer
    const int64_t n_heads,
    const int64_t gqa_ratio,
    const int repeat_type,
    const int64_t n_tokens,
    const int64_t n_seqs,
    const int64_t output_offset,          // offset where state starts in output
    size_t vnb1, size_t vnb2, size_t vnb3) {
    constexpr int warps_per_head = HEAD_DIM/WARP_SIZE;
    const int batch_idx = blockIdx.x / (warps_per_head*n_heads);
    const int sub_head_idx  = blockIdx.x % (warps_per_head*n_heads);
    const int head_idx = sub_head_idx / warps_per_head;
    const int sub_idx  = sub_head_idx % warps_per_head;
    const int head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);
    const int tid = threadIdx.x;

    // Strides for input tensors (column-major)
    // Q/K/V: [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t qkv_stride_batch_kq = qkv_stride_batch / gqa_ratio;

    // G/Beta: [n_tokens, 1, n_heads, n_seqs] / [1, n_tokens, n_heads, n_seqs]
    //const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    // State: [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    // For head h: columns h*HEAD_DIM to (h+1)*HEAD_DIM
    // state[row, col] for head h = state[row, h*HEAD_DIM + col]
    // Linear index: row + (h*HEAD_DIM + col) * HEAD_DIM = row + h*HEAD_DIM^2 + col*HEAD_DIM
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // State step stride for save_all_states: HEAD_DIM^2 * n_heads * n_seqs
    const int64_t state_step_stride = HEAD_DIM * HEAD_DIM * n_heads * n_seqs;

    // Pointers for this batch/head
    const float * q_ptr = q + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * v_ptr = v + batch_idx * vnb3 + head_idx * vnb2;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    // For [dim, head, token, batch]: index = dim + head*S_v + token*S_v*H_v + batch*S_v*H_v*n_tokens
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;  // stride between tokens
    // PXA_DN_SCATTER_FUSE: when state_ext_idx is given, state_ext is the BASE of the recurrent
    // state buffer (already advanced past the conv prefix) and this sequence's row is chosen on the
    // DEVICE from the same index tensor the fused-away SET_ROWS would have used. The host cannot
    // do it: the index lives in a graph input buffer whose value is only known on the device
    // without a sync, and reading it host-side would also break CUDA-graph replay. Rows are
    // disjoint per sequence, so the per-batch packing stride is replaced (not added to) here.
    float * state_dst = (state_ext && state_ext_idx)
        ? state_ext + (int64_t)state_ext_idx[batch_idx] * state_ext_row_stride + state_head_offset
        : (state_ext ? state_ext : dst + output_offset) + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for current token's Q, K, V (normalized), and intermediate results
    extern __shared__ float smem[];
    float * sQ = smem;                          // HEAD_DIM
    float * sK = sQ + HEAD_DIM;                 // HEAD_DIM

    const float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float sum_helper[block_size/WARP_SIZE];

    constexpr int num_warps = block_size/WARP_SIZE;
    const int row = tid % WARP_SIZE;
    const int col_idx_0 = tid / WARP_SIZE;
    const int row_out = row + sub_idx * WARP_SIZE;

    // Keep the state in registers, copy the final state to its destination at the end
    float state_local[HEAD_DIM/num_warps];
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_local[i] = state_src[col*HEAD_DIM + row_out];
    }

    constexpr int WARP_SIZE_S = WARP_SIZE + 1;
    constexpr int num_stored_rows = block_size/WARP_SIZE;
    __shared__ float all_sum[2*WARP_SIZE_S*num_stored_rows];
    auto all_sum1 = all_sum;
    auto all_sum2 = all_sum1 + WARP_SIZE_S*num_stored_rows;

    for (int64_t t = 0; t < n_tokens; t++) {
        float sum_kq = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += block_size) {
            sQ[i] = q_ptr[t * qkv_stride_token + i] * scale;
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sum_kq += sK[i] * sQ[i];
        }

        float attn_score = reduce_sum<block_size>(sum_kq, sum_helper);

        float beta_val = sigmoid_f(beta_ptr[t*n_heads]);
        float decay    = expf(fminf(g_ptr[t*n_heads], 50.0f));

        float sum1 = 0, sum2 = 0;
#pragma unroll
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            sum1 += state_local[i] * sK[col];
            sum2 += state_local[i] * sQ[col];
        }
        all_sum1[col_idx_0*WARP_SIZE_S + row] = sum1;
        all_sum2[col_idx_0*WARP_SIZE_S + row] = sum2;

        __syncthreads();

        sum1 = sum2 = 0;
#pragma unroll
        for (int i = 0; i < block_size/WARP_SIZE; ++i) {
            sum1 += all_sum1[i*WARP_SIZE_S + row];
            sum2 += all_sum2[i*WARP_SIZE_S + row];
        }

        //float sv_new = beta_val * (v_ptr[t * qkv_stride_token + row_out] - sum1 * decay);
        float sv_new = beta_val * (v_ptr[t * vnb1 + row_out] - sum1 * decay);
        if (col_idx_0 == 0) {
            out_base[t * out_token_stride + row_out] = sum2 * decay + sv_new * attn_score;
        }

        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            float new_state_val = decay * state_local[i] + sv_new * sK[col];
            new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
            state_local[i] = new_state_val;
        }

        // Save per-step state if requested
        if (saved_states && t < n_tokens - 1) {
            float * state_step_dst = saved_states + batch_idx * state_batch_stride + state_head_offset + t * state_step_stride;
            for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
                int col = num_warps*i + col_idx_0;
                state_step_dst[col*HEAD_DIM + row_out] = state_local[i];
            }
        }

        // Barrier required: (a) sK reads in the state update above must complete
        // before next iteration overwrites sK at the top of the loop, and (b) this
        // single barrier also orders all_sum1/all_sum2 reads above vs. the next
        // iteration's writes — subsuming the prior barriers after the cross-warp
        // reduction and after the loop exit.
        __syncthreads();
    }
    // Copy the final state to its destination
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_dst[col*HEAD_DIM + row_out] = state_local[i];
    }
}

static void delta_net_f32_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state_in,
    float * dst,
    float * saved_states,
    float * state_ext,
    const int32_t * state_ext_idx,
    const int64_t state_ext_row_stride,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t gqa_ratio,
    const int     repeat_type,
    const int64_t n_seqs,
    size_t vnb1, size_t vnb2, size_t vnb3,
    const int device_id,
    const int cc, // compute capability (e.g., 890 for SM 8.9, 1200 for SM 12.0)
    cudaStream_t stream) {
    GGML_UNUSED(device_id);
    GGML_UNUSED(cc);

    const int64_t output_offset = head_dim * n_tokens * n_heads * n_seqs;

    if (head_dim != 64 && head_dim != 128) {
        GGML_ABORT("Unsupported delta net head size");
    }

    GGML_ASSERT(head_dim % WARP_SIZE == 0);
    const int num_blocks = n_seqs * n_heads * (head_dim/WARP_SIZE);
    const size_t smem_size = 2 * head_dim * sizeof(float);

    if (n_tokens <= 8) {
        constexpr int threads_per_block = 256;
        if (head_dim == 64) {
            delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, state_ext, state_ext_idx, state_ext_row_stride, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        } else {
            delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, state_ext, state_ext_idx, state_ext_row_stride, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        }
    } else {
        constexpr int threads_per_block = 128;
        if (head_dim == 64) {
            delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, state_ext, state_ext_idx, state_ext_row_stride, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        } else {
            delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, state_ext, state_ext_idx, state_ext_row_stride, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        }
    }

    CUDA_CHECK(cudaGetLastError());

}

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_delta_net_ex2(ctx, dst, nullptr, nullptr, 0);
}

void ggml_cuda_op_delta_net_ex(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override) {
    ggml_cuda_op_delta_net_ex2(ctx, dst, state_dst_override, nullptr, 0);
}

void ggml_cuda_op_delta_net_ex2(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override,
                                const int32_t * state_dst_row_idx, int64_t state_dst_row_stride) {
    const ggml_tensor * src0 = dst->src[0];  // q
    const ggml_tensor * src1 = dst->src[1];  // k
    const ggml_tensor * src2 = dst->src[2];  // v
    const ggml_tensor * src3 = dst->src[3];  // g
    const ggml_tensor * src4 = dst->src[4];  // beta
    const ggml_tensor * src5 = dst->src[5];  // state
    const ggml_tensor * src6 = dst->src[6];  // when not null, state for token 0...n_token-1

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t head_dim = src0->ne[0];
    const int64_t n_tokens = src0->ne[1];
    const int64_t n_heads = src2->ne[2];
    const int64_t n_heads_kq = src0->ne[2];
    const int64_t n_seqs = src0->ne[3];
    GGML_ASSERT(n_heads % n_heads_kq == 0);
    const int64_t gqa_ratio = n_heads / n_heads_kq;

    // Dimension validation
    // Q/K: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src1->ne[0] == head_dim && src1->ne[1] == n_tokens && src1->ne[2] == n_heads_kq && src1->ne[3] == n_seqs);
    // V: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src2->ne[0] == head_dim && src2->ne[1] == n_tokens && src2->ne[2] == n_heads && src2->ne[3] == n_seqs);
    // G: [n_tokens, 1, n_heads, n_seqs]
    GGML_ASSERT(src3->ne[0] == n_tokens && src3->ne[1] == 1 && src3->ne[2] == n_heads && src3->ne[3] == n_seqs);
    // Beta: [1, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src4->ne[0] == 1 && src4->ne[1] == n_tokens && src4->ne[2] == n_heads && src4->ne[3] == n_seqs);
    // State: [head_dim, head_dim*n_heads, 1, n_seqs]
    GGML_ASSERT(src5->ne[0] == head_dim && src5->ne[1] == head_dim * n_heads && src5->ne[2] == 1 && src5->ne[3] == n_seqs);

    // Verify output tensor size
    const int64_t output_size = head_dim * n_tokens * n_heads * n_seqs;
    const int64_t state_size = head_dim * head_dim * n_heads * n_seqs;

    int repeat_type = dst->op_params[0];
    if (src6) {
        GGML_ASSERT(src6->type == GGML_TYPE_F32);
        GGML_ASSERT(src6->ne[0] >= (n_tokens - 1)*state_size);
    }

    const int64_t expected_size = output_size + state_size;
    GGML_ASSERT(ggml_nelements(dst) == expected_size);

    GGML_ASSERT(head_dim <= 256);  // Reasonable limit for shared memory

    // Get device info from ctx (avoids calling CUDA runtime APIs inside dispatch)
    const int device_id = ctx.device;
    const int cc = ggml_cuda_info().devices[device_id].cc;

    delta_net_f32_cuda(
        (const float *)src0->data,
        (const float *)src1->data,
        (const float *)src2->data,
        (const float *)src3->data,
        (const float *)src4->data,
        (const float *)src5->data,
        (float *)dst->data,
        src6 ? (float *)src6->data : nullptr,
        state_dst_override,
        state_dst_row_idx,
        state_dst_row_stride,
        head_dim, n_tokens, n_heads, gqa_ratio, repeat_type, n_seqs,
        src2->nb[1]/sizeof(float), src2->nb[2]/sizeof(float), src2->nb[3]/sizeof(float),
        device_id, cc,
        ctx.stream());

}
