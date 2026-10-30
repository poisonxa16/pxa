// PXA Nemotron-H: Mamba-2 selective scan for GGML_OP_SSM_SCAN2.
// Kernel shape ported from mainline llama.cpp ggml-cuda/ssm-scan.cu (ssm_scan_f32_group, MIT, see NOTICE):
// one warp per (head, head-dim row), each lane keeps d_state/32 state values in registers, the output row is a
// warp reduction. Differences from mainline: no ids/rollback snapshots (the caller gathers states in batch order),
// and the state tail of dst is always the destination.
#include "ssm-scan2.cuh"

template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    ssm_scan2_f32_group(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
        const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
        float * __restrict__ dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok) {
    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x  * c_factor + warp;

    if (warp_idx >= n_head * d_head) {
        return;
    }

    const int head_idx = warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * sizeof(float);
    const int seq_idx  = blockIdx.y;

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    const float * s0_warp = (const float *) ((const char *) src0 + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_warp  = (const float *) ((const char *) src1 + seq_idx * src1_nb3 + warp_idx * sizeof(float));
    const float * dt_warp = (const float *) ((const char *) src2 + seq_idx * src2_nb2 + head_idx * sizeof(float));
    const float * A_warp  = src3 + head_idx;
    const float * B_warp  = (const float *) ((const char *) src4 + seq_idx * src4_nb3 + group_off);
    const float * C_warp  = (const float *) ((const char *) src5 + seq_idx * src5_nb3 + group_off);
    float *       y_warp  = dst + seq_idx * n_tok * n_head * d_head + warp_idx;
    float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[c_factor];

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    const float A = A_warp[0];

    for (int64_t i = 0; i < n_tok; i++) {
        const float dt_raw = dt_warp[i * stride_dt];
        const float dt_sp  = dt_raw <= 20.0f ? log1pf(expf(dt_raw)) : dt_raw;
        const float dA     = expf(dt_sp * A);
        const float x_dt   = x_warp[i * stride_x] * dt_sp;

        float state_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = state[j] * dA + B_val * x_dt;
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }
    }

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

template <int d_state>
static void ssm_scan2_launch(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = dst->src[3];
    const ggml_tensor * src4 = dst->src[4];
    const ggml_tensor * src5 = dst->src[5];

    const int64_t head_dim = src0->ne[1];
    const int64_t n_head   = src1->ne[1];
    const int64_t n_group  = src4->ne[1];
    const int64_t n_tok    = src1->ne[2];
    const int64_t n_seq    = src1->ne[3];
    const int64_t s_off    = ggml_nelements(src1) * sizeof(float);

    constexpr int threads   = d_state;
    constexpr int num_warps = threads / WARP_SIZE;
    const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
    ssm_scan2_f32_group<d_state / WARP_SIZE, d_state><<<blocks, threads, 0, stream>>>(
        (const float *) src0->data, (const float *) src1->data, (const float *) src2->data,
        (const float *) src3->data, (const float *) src4->data, (const float *) src5->data,
        (float *) dst->data,
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, n_head, head_dim, n_group, n_tok);
}

void ggml_cuda_op_ssm_scan2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t d_state = dst->src[0]->ne[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    cudaStream_t stream = ctx.stream();
    switch (d_state) {
        case  64: ssm_scan2_launch< 64>(dst, stream); break;
        case 128: ssm_scan2_launch<128>(dst, stream); break;
        case 256: ssm_scan2_launch<256>(dst, stream); break;
        default: GGML_ABORT("ssm_scan2: d_state must be 64, 128 or 256");
    }
    CUDA_CHECK(cudaGetLastError());
}
