#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}


// PXA_ARGMAX_VAL: one block per row -> (index, max, sumexp). Deterministic: the (value, index)
// order is "larger value wins, equal values -> lower index", so the result does not depend on
// the thread schedule and matches the host greedy sampler's first-maximal-id rule.
static __device__ __forceinline__ void pxa_amv_merge(float & v, int & i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}

static __global__ void argmax_val_f32(const float * __restrict__ x, float * __restrict__ dst, const int64_t ncols, const int64_t row_stride) {
    const int64_t row = blockIdx.x;
    const float * rowx = x + row * row_stride;

    float maxval = -INFINITY;
    int   argmax = 0x7fffffff;
    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        pxa_amv_merge(maxval, argmax, rowx[col], (int) col);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float v2 = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   i2 = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        pxa_amv_merge(maxval, argmax, v2, i2);
    }
    __shared__ float s_v[32];
    __shared__ int   s_i[32];
    __shared__ float s_s[32];
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) { s_v[warp_id] = maxval; s_i[warp_id] = argmax; }
    __syncthreads();
    if (warp_id == 0) {
        maxval = lane_id < n_warps ? s_v[lane_id] : -INFINITY;
        argmax = lane_id < n_warps ? s_i[lane_id] : 0x7fffffff;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float v2 = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
            const int   i2 = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
            pxa_amv_merge(maxval, argmax, v2, i2);
        }
        if (lane_id == 0) { s_v[0] = maxval; s_i[0] = argmax; }
    }
    __syncthreads();
    maxval = s_v[0];
    argmax = s_i[0];

    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        sum += expf(rowx[col] - maxval);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
    __syncthreads();
    if (lane_id == 0) { s_s[warp_id] = sum; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < n_warps; ++w) tot += s_s[w];
        dst[3*row + 0] = (float) argmax;
        dst[3*row + 1] = maxval;
        dst[3*row + 2] = tot;
    }
}

// PXA_ARGMAX_COMBINE (op_params[0] == 2): src[k] = the [3, rows] (index, max, sumexp) of vocab slice k,
// op_params[1 + k] = that slice's first column, op_params[5] = number of slices. dst I32 [rows] = the
// global argmax, slices scanned in column order with strictly-greater wins (lowest id on exact ties).
struct pxa_amc_args { const float * p[4]; int off[4]; int n; };
static __global__ void argmax_combine_f32(pxa_amc_args a, int32_t * __restrict__ dst, const int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float best_v = 0.0f; int best = -1;
    for (int k = 0; k < a.n; ++k) {
        const float * q = a.p[k] + 3*r;
        if (best < 0 || q[1] > best_v) { best_v = q[1]; best = a.off[k] + (int) q[0]; }
    }
    dst[r] = best;
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    if (dst->op_params[0] == 2) {
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        pxa_amc_args a = {};
        a.n = dst->op_params[5];
        GGML_ASSERT(a.n >= 1 && a.n <= 4);
        for (int k = 0; k < a.n; ++k) {
            GGML_ASSERT(dst->src[k] && dst->src[k]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[k]));
            a.p[k] = (const float *) dst->src[k]->data;
            a.off[k] = dst->op_params[1 + k];
        }
        const int rows = (int) dst->ne[0];
        argmax_combine_f32<<<(rows + 127) / 128, 128, 0, ctx.stream()>>>(a, (int32_t *) dst->data, rows);
        return;
    }

    if (dst->op_params[0] == 1) {
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
        const int64_t nrows = src0->ne[1];
        argmax_val_f32<<<dim3(nrows, 1, 1), dim3(1024, 1, 1), 0, ctx.stream()>>>(
            (const float *) src0->data, (float *) dst->data, src0->ne[0], src0->nb[1] / sizeof(float));
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
