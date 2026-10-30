//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "softmax.cuh"
#include "pxa/kv-band.cuh"

// pxa/kv-band.cu: PXA_KV_BAND_EXACT on NVIDIA cc >= 6.0 (Pascal added 2026); HIP keeps the old walk
bool pxa_kv_band_softmax_applies(int cc);

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

// PXA_KV_BAND_EXACT (pxa/kv-band.cuh). A KV band can start at any cell of the unified ring and every other cell of
// the row is masked to -inf, so the terms that reach the row sum do not depend on where the band sits -- but the
// thread slot a column is summed in does (column c goes to thread c % block_size), and so does the order of the
// tree that folds the slots. The row is therefore walked from its first open column: column `first + i` takes the
// slot column `i` took, and a band at any cell is summed exactly as the same band at cell 0. When that column is 0
// (no band offset, no mask, or nothing open) the walk is the one of before, bit for bit. Returns the anchor; every
// thread's vals[] is visible to every other thread when it returns.
static __device__ __forceinline__ int soft_max_band_warp_min(int x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = min(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ int soft_max_band_anchor(const float * vals, const int ncols, const int block_size,
        const int tid, const int warp_id, const int lane_id, float * buf_iw) {
    int a = ncols;
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;
        if (col >= ncols) {
            break;
        }
        if (!(vals[col] == -INFINITY)) { // open: anything but a masked -inf (a NaN counts, as it did in the sum)
            a = col; // this thread's columns rise, so its first open one is its smallest
            break;
        }
    }
    a = soft_max_band_warp_min(a);
    if (block_size > WARP_SIZE) {
        int * buf = (int *) buf_iw;
        __syncthreads();
        if (warp_id == 0) {
            buf[lane_id] = ncols;
        }
        __syncthreads();
        if (lane_id == 0) {
            buf[warp_id] = a;
        }
        __syncthreads();
        a = soft_max_band_warp_min(buf[lane_id]);
    }
    __syncthreads();
    return a < ncols ? a : 0;
}

template <bool vals_smem, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32_nosinks(const float * x, const T * mask, float * dst, const int ncols_par, const int nrows_y, const float scale, const float max_bias, const float m0, const float m1, uint32_t n_head_log2, float cap_params0, float cap_params1, bool do_softcap, bool band) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(max_bias, rowx/nrows_y, n_head_log2, m0, m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = vals_smem ? buf_iw + WARP_SIZE : dst + (int64_t)rowx*ncols;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = do_softcap ? scale*cap_params1*tanhf(cap_params0*x[ix]) + (mask ? slope*t2f32(mask[iy]) : 0.0f) :
                                       scale*x[ix] + (mask ? slope*t2f32(mask[iy]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    const int anchor = band && mask ? soft_max_band_anchor(vals, ncols, block_size, tid, warp_id, lane_id, buf_iw) : 0;

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = anchor; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if ((ncols_template == 0 || anchor != 0) && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }
    if (anchor > 0) {
        for (int col = tid; col < anchor; col += block_size) {
            vals[col] = 0.0f; // masked: what expf(-inf - max) gave
        }
        __syncthreads();
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t idst = (int64_t)rowx*ncols + col;
        dst[idst] = vals[col] * inv_sum;
    }
}

template<typename T>
static void soft_max_f32_cuda_nosinks(const float * x, const T * mask, float * dst, const int ncols_x, const int nrows_x, const int nrows_y, const float scale, const float max_bias, float cap_params0, float cap_params1, bool do_softcap, cudaStream_t stream, bool band = false) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    const uint32_t n_head      = nrows_x/nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    if (shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        switch (ncols_x) {
            case 32:
                soft_max_f32_nosinks<true, 32, 32><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 64:
                soft_max_f32_nosinks<true, 64, 64><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 128:
                soft_max_f32_nosinks<true, 128, 128><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 256:
                soft_max_f32_nosinks<true, 256, 256><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 512:
                soft_max_f32_nosinks<true, 512, 512><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 1024:
                soft_max_f32_nosinks<true, 1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 2048:
                soft_max_f32_nosinks<true, 2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            case 4096:
                soft_max_f32_nosinks<true, 4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
            default:
                soft_max_f32_nosinks<true, 0, 0><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
                break;
        }
    } else {
        const size_t shmem_low = WARP_SIZE*sizeof(float);
        soft_max_f32_nosinks<false, 0, 0><<<block_nums, block_dims, shmem_low, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2, cap_params0, cap_params1, do_softcap, band);
    }
}

#if 0
void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const void  * src1_d = src1 ? (const void *)src1->data : nullptr;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    if (use_f16) {
        const half * src1_dd = (const half *)src1_d;

        soft_max_f32_cuda_nosinks(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, 0, 0, false, stream);
    } else {
        const float * src1_dd = (const float *)src1_d;

        soft_max_f32_cuda_nosinks(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, 0, 0, false, stream);
    }
}
#endif

void ggml_cuda_op_soft_cap_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const void  * src1_d = src1 ? (const void *)src1->data : nullptr;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    float params[4];
    memcpy(params, dst->op_params, sizeof(params));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);
    //printf("%s: %g, %g, %g, %g, %p, %d\n", __func__, params[0], params[1], params[2], params[3], (const void *)src1, use_f16);
    // PXA_KV_BAND_ANCHOR0: every row proven to open at column 0 walks from 0 either way (the walk of before)
    const bool band = src1 != nullptr && pxa_kv_band_softmax_applies(ggml_cuda_info().devices[ctx.device].cc) &&
                      !pxa_kv_band_anchor0_proven(src1, ne00, nrows_y, false, stream);

    if (use_f16) {
        const half * src1_dd = (const half *)src1_d;

        soft_max_f32_cuda_nosinks(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, params[0], params[1], params[2], params[3], true, stream,
                                  band);
    } else {
        const float * src1_dd = (const float *)src1_d;

        soft_max_f32_cuda_nosinks(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, params[0], params[1], params[2], params[3], true, stream,
                                  band);
    }
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    bool  band; // PXA_KV_BAND_EXACT: walk each row from its first open column
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid  = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    //TODO: noncontigous inputs/outputs
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = sinks ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    const int anchor = p.band && mask ? soft_max_band_anchor(vals, ncols, block_size, tid, warp_id, lane_id, buf_iw) : 0;

    float tmp = 0.0f; // partial sum
#pragma unroll
    for (int col0 = anchor; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if ((ncols_template == 0 || anchor != 0) && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }
    if (anchor > 0) {
        for (int col = tid; col < anchor; col += block_size) {
            vals[col] = 0.0f; // masked: what expf(-inf - max) gave
        }
        __syncthreads();
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    if (sinks) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template<int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, 0, 0, T>), smpbo);
    soft_max_f32<true, 0, 0><<<block_nums, block_dims, nbytes_shared, stream>>>(x, mask, sinks, dst, p);
}

template<typename T>
static void soft_max_f32_cuda(const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params & params, cudaStream_t stream) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;


    if (nbytes_shared <= smpbo) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
    } else {
        const size_t nbytes_shared_low = WARP_SIZE*sizeof(float);
        soft_max_f32<false, 0, 0><<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, sinks, dst, params);
    }
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;
    params.band = src1 != nullptr && pxa_kv_band_softmax_applies(ggml_cuda_info().devices[ctx.device].cc) &&
                  !pxa_kv_band_anchor0_proven(src1, ne00, nrows_y, false, stream); // PXA_KV_BAND_ANCHOR0, as above

    if (use_f16) {
        soft_max_f32_cuda(src0_d, (const half  *) src1_d, (const float *) src2_d, dst_d, params, stream);
    } else {
        soft_max_f32_cuda(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream);
    }
}

