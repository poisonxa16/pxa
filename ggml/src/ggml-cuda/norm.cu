#include "norm.cuh"

template <int block_size, typename T>
static __global__ void norm_f32(const T * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float)x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = (T)(((float)x[row*ncols + col] - mean) * inv_std);
    }
}

template <int block_size, typename T>
static __global__ void fused_norm_f32(const T * x, const float * c, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    if constexpr (std::is_same_v<T, block_q8_0>) {
        static_assert(block_size % QK8_0 == 0);
        auto xr = x + (row*ncols)/QK8_0;
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = (float)xr[col / QK8_0].d * xr[col / QK8_0].qs[col % QK8_0];
            mean_var.x += xi;
            mean_var.y += xi * xi;
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = (float)x[row*ncols + col];
            mean_var.x += xi;
            mean_var.y += xi * xi;
        }
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    if constexpr (std::is_same_v<T, block_q8_0>) {
        static_assert(block_size % QK8_0 == 0);
        auto xr = x + (row*ncols)/QK8_0;
        for (int col = tid; col < ncols; col += block_size) {
            dst[row*ncols + col] = ((float)xr[col/QK8_0].d*xr[col/QK8_0].qs[col%QK8_0] - mean) * inv_std * c[col];
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[row*ncols + col] = ((float)x[row*ncols + col] - mean) * inv_std * c[col];
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    int start = blockIdx.x * group_size;
    int end = start + group_size;

    start += threadIdx.x;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float variance = tmp / group_size;
    float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// PXA_NORM_REGCACHE (default OFF): keep the row in registers across the two passes.
//
// The rms/l2 norm kernels read x twice -- once for the sum of squares, once to scale. At
// single-token decode the grid is one block, so the kernel occupies one SM and is limited by
// the bytes it pulls through it; dropping the second read is worth roughly a fifth of the
// launch. It only pays when the row needs more than one value per thread and still fits in a
// small fixed register array, so it is confined to the block_size == 1024 instantiations and
// to rows of at most 4*block_size.
//
// The cached value is exactly the value the second pass would have re-read, and the scaling
// expression is otherwise untouched, so the output is bit-identical.
#define PXA_NORM_MAX_CACHE 4

// PXA_NORM_REGCACHE=1 enables the register-cached norm variants.
static bool pxa_norm_regcache_enabled() {
    static const bool v = [] {
        // DEFAULT 2026-09-03: ON at ENHANCE (the default level), OFF at DEFAULT/REFERENCE. The
        // cached value is exactly the value the second pass would have re-read and the scaling
        // expression is untouched, so the output is bit-identical (see the comment above); the
        // variants are confined to the block_size == 1024 instantiations on every architecture.
        // PXA_NORM_REGCACHE still overrides in both directions.
        return pxa_cuda_house_lever("PXA_NORM_REGCACHE");
    }();
    return v;
}

template <int block_size, bool regcache = false>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    const bool cached = regcache && ncols > block_size && ncols <= PXA_NORM_MAX_CACHE*block_size;
    float xv[regcache ? PXA_NORM_MAX_CACHE : 1];

    if (cached) {
#pragma unroll
        for (int k = 0; k < (regcache ? PXA_NORM_MAX_CACHE : 1); ++k) {
            const int   col = tid + k*block_size;
            const float xi  = col < ncols ? x[row*ncols + col] : 0.0f;
            xv[k] = xi;
            tmp  += xi * xi;
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[row*ncols + col];
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if (cached) {
#pragma unroll
        for (int k = 0; k < (regcache ? PXA_NORM_MAX_CACHE : 1); ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                dst[row*ncols + col] = scale * xv[k];
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[row*ncols + col] = scale * x[row*ncols + col];
        }
    }
}

template <int block_size, bool regcache = false>
static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f;

    const bool cached = regcache && ncols > block_size && ncols <= PXA_NORM_MAX_CACHE*block_size;
    float xv[regcache ? PXA_NORM_MAX_CACHE : 1];

    if (cached) {
#pragma unroll
        for (int k = 0; k < (regcache ? PXA_NORM_MAX_CACHE : 1); ++k) {
            const int   col = tid + k*block_size;
            const float xi  = col < ncols ? x[row*ncols + col] : 0.0f;
            xv[k] = xi;
            tmp  += xi * xi;
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[row * ncols + col];
            tmp += xi * xi;
        }
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size / WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    if (cached) {
#pragma unroll
        for (int k = 0; k < (regcache ? PXA_NORM_MAX_CACHE : 1); ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                dst[row*ncols + col] = scale * xv[k];
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[row * ncols + col] = scale * x[row * ncols + col];
        }
    }
}

template <int block_size>
static __global__ void rms_norm_f32_nc(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

template <int block_size>
static __global__ void l2_norm_f32_nc(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    float tmp = 0.0f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

// The q8_0 source is deliberately NOT cached: its scaling pass evaluates
// scale * y[col] * (float)d * qs, and folding d*qs into one cached value re-associates the
// product and would change the last bit. Every other source type re-reads a value that
// converts to exactly the cached float, so those stay bit-identical.
template <int block_size, typename src_t, bool regcache = false>
static __global__ void fused_rms_norm_f32(const src_t * x, const float * y, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    constexpr bool can_cache = regcache && !std::is_same_v<src_t, block_q8_0>;
    const bool cached = can_cache && ncols > block_size && ncols <= PXA_NORM_MAX_CACHE*block_size;
    float xv[can_cache ? PXA_NORM_MAX_CACHE : 1];

    if constexpr (std::is_same_v<src_t, block_q8_0>) {
        static_assert(block_size % QK8_0 == 0);
        auto xr = x + (row*ncols)/QK8_0;
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = (float)xr[col / QK8_0].d * xr[col / QK8_0].qs[col % QK8_0];
            tmp += xi * xi;
        }
    } else if constexpr (std::is_same_v<src_t, nv_bfloat16>) {
        if (cached) {
#pragma unroll
            for (int k = 0; k < (can_cache ? PXA_NORM_MAX_CACHE : 1); ++k) {
                const int   col = tid + k*block_size;
                const float xi  = col < ncols ? __bfloat162float(x[row*ncols + col]) : 0.0f;
                xv[k] = xi;
                tmp  += xi * xi;
            }
        } else {
            for (int col = tid; col < ncols; col += block_size) {
                const float xi = __bfloat162float(x[row*ncols + col]);
                tmp += xi * xi;
            }
        }
    } else {
        if (cached) {
#pragma unroll
            for (int k = 0; k < (can_cache ? PXA_NORM_MAX_CACHE : 1); ++k) {
                const int   col = tid + k*block_size;
                const float xi  = col < ncols ? (float)x[row*ncols + col] : 0.0f;
                xv[k] = xi;
                tmp  += xi * xi;
            }
        } else {
            for (int col = tid; col < ncols; col += block_size) {
                const float xi = (float)x[row*ncols + col];
                tmp += xi * xi;
            }
        }
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (std::is_same_v<src_t, block_q8_0>) {
        auto xr = x + (row*ncols)/QK8_0;
        for (int col = tid; col < ncols; col += block_size) {
            dst[row*ncols + col] = scale * y[col] * (float)xr[col / QK8_0].d * xr[col / QK8_0].qs[col % QK8_0];
        }
    } else if (cached) {
#pragma unroll
        for (int k = 0; k < (can_cache ? PXA_NORM_MAX_CACHE : 1); ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                dst[row*ncols + col] = scale * y[col] * xv[k];
            }
        }
    } else if constexpr (std::is_same_v<src_t, nv_bfloat16>) {
        for (int col = tid; col < ncols; col += block_size) {
            dst[row*ncols + col] = scale * y[col] * __bfloat162float(x[row*ncols + col]);
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[row*ncols + col] = scale * y[col] * (float)x[row*ncols + col];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// G2-F3 NORMFUSE (2026-07-19): fused rms-norm that ALSO emits the q8_1 quantization of its own
// output as a sidecar, so the consuming mul_mat_vec_q chain needs no separate quantize_q8_1
// launch (and no extra global read pass). The f32 dst is written EXACTLY like fused_rms_norm_f32
// (same block size selection -> same reduction order -> bit-identical), and the q8_1 epilogue
// replicates quantize_q8_1's arithmetic verbatim (one warp per 32-wide block: warp amax/sum
// reduce, d = amax/127, q = roundf(xi/d)) on identically-computed values => the sidecar is
// bit-identical to what quantize_q8_1 would have produced from the standalone norm output.
// ---------------------------------------------------------------------------------------------
template <int block_size>
static __global__ void fused_rms_norm_q8_f32(const float * x, const float * y, float * dst,
        void * vq8, const int ncols, const int ncols_padded, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * y[col] * x[row*ncols + col];
    }

    // q8_1 epilogue — quantize_q8_1 replicated, one warp per 32-wide block
    block_q8_1 * q8 = (block_q8_1 *)vq8 + (int64_t)row*(ncols_padded/QK8_1);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    for (int ib = warp_id; ib < ncols_padded/QK8_1; ib += block_size/WARP_SIZE) {
        const int col = ib*QK8_1 + lane_id;
        const float xi = col < ncols ? scale * y[col] * x[row*ncols + col] : 0.0f;
        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max(amax);
        sum  = warp_reduce_sum(sum);
        const float d = amax / 127;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        q8[ib].qs[lane_id] = q;
        if (lane_id == 0) {
            reinterpret_cast<half&>(q8[ib].ds.x) = d;
            reinterpret_cast<half&>(q8[ib].ds.y) = sum;
        }
    }
}

template <int block_size, typename src_t>
static __global__ void fused_rms_norm_f32_nc(
        const src_t * x, const float * y, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    //const int channel   = blockIdx.y * blockDim.y + threadIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float)x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        //if constexpr (block_size == 1024) {
        //    tmp = s_sum[lane_id];
        //} else {
        //    tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        //}
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * y[col] * (float)x[col];
    }
}

template <typename T>
static void norm_f32_cuda(const T * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE, T><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024, T><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

static void group_norm_f32_cuda(const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    // Why did we have this assert?
    //GGML_ASSERT(ncols % WARP_SIZE == 0);
    constexpr int kBlockSize = 256;
    if (ncols < 1024) {
        const dim3 block_dims(kBlockSize, 1, 1);
        rms_norm_f32<kBlockSize><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        if (pxa_norm_regcache_enabled()) {
            rms_norm_f32<1024, true><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
        } else {
            rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
        }
    }
}

static void rms_norm_f32_nc_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32_nc<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32_nc<1024><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void l2_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    constexpr int kBlockSize = 256;
    if (ncols < 1024) {
        const dim3 block_dims(kBlockSize, 1, 1);
        l2_norm_f32<kBlockSize><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        if (pxa_norm_regcache_enabled()) {
            l2_norm_f32<1024, true><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
        } else {
            l2_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
        }
    }
}

static void l2_norm_f32_nc_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        l2_norm_f32_nc<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        l2_norm_f32_nc<1024><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

template <typename src_t>
static void fused_rms_norm_f32_cuda(const src_t * x, const float * y, float * dst,
        const int ncols, const int nrows, const float eps, bool is_norm, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (is_norm) {
        if (ncols < kBlockSize) {
            switch (ncols) {
                case  32: fused_norm_f32< 32><<<nrows,  32, 0, stream>>>(x, y, dst, ncols, eps); break;
                case  64: fused_norm_f32< 64><<<nrows,  64, 0, stream>>>(x, y, dst, ncols, eps); break;
                case  96: fused_norm_f32< 96><<<nrows,  96, 0, stream>>>(x, y, dst, ncols, eps); break;
                case 128: fused_norm_f32<128><<<nrows, 128, 0, stream>>>(x, y, dst, ncols, eps); break;
                case 160: fused_norm_f32<160><<<nrows, 160, 0, stream>>>(x, y, dst, ncols, eps); break;
                case 192: fused_norm_f32<192><<<nrows, 192, 0, stream>>>(x, y, dst, ncols, eps); break;
                default : fused_norm_f32<224><<<nrows, 224, 0, stream>>>(x, y, dst, ncols, eps); break;
            }
        }
        else if (ncols < 1024) {
            const dim3 block_dims(kBlockSize, 1, 1);
            fused_norm_f32<kBlockSize><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
        } else {
            const dim3 block_dims(1024, 1, 1);
            fused_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
        }
    } else {
        if (ncols < kBlockSize) {
            switch (ncols) {
                case  32: fused_rms_norm_f32< 32><<<nrows,  32, 0, stream>>>(x, y, dst, ncols, eps); break;
                case  64: fused_rms_norm_f32< 64><<<nrows,  64, 0, stream>>>(x, y, dst, ncols, eps); break;
                case  96: fused_rms_norm_f32< 96><<<nrows,  96, 0, stream>>>(x, y, dst, ncols, eps); break;
                case 128: fused_rms_norm_f32<128><<<nrows, 128, 0, stream>>>(x, y, dst, ncols, eps); break;
                case 160: fused_rms_norm_f32<160><<<nrows, 160, 0, stream>>>(x, y, dst, ncols, eps); break;
                case 192: fused_rms_norm_f32<192><<<nrows, 192, 0, stream>>>(x, y, dst, ncols, eps); break;
                default : fused_rms_norm_f32<224><<<nrows, 224, 0, stream>>>(x, y, dst, ncols, eps); break;
            }
        }
        else if (ncols < 1024) {
            const dim3 block_dims(kBlockSize, 1, 1);
            fused_rms_norm_f32<kBlockSize><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
        } else {
            const dim3 block_dims(1024, 1, 1);
            if (pxa_norm_regcache_enabled()) {
                fused_rms_norm_f32<1024, src_t, true><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
            } else {
                fused_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
            }
        }
    }
}

template <typename src_t>
static void fused_rms_norm_f32_nc_cuda(
        const src_t * x, const float * y, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        fused_rms_norm_f32_nc<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        //constexpr int kBlockSize = 256;

        //if (nchannels%4 == 0) {
        //    const dim3 blocks_num(nrows, nchannels/4, nsamples);
        //    const dim3 block_dims(kBlockSize, 4, 1);
        //    fused_rms_norm_f32_nc<kBlockSize><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        //} else {
        //    const dim3 block_dims(kBlockSize, 1, 1);
        //    fused_rms_norm_f32_nc<kBlockSize><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        //}
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_rms_norm_f32_nc<1024><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F32) {
        norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
    } else {
        norm_f32_cuda((const half *)src0_d, dst_d, ne00, nrows, eps, stream);
    }
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    if (ggml_is_contiguous(src0)) {
        const int64_t nrows = ggml_nrows(src0);
        rms_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
    } else {
        auto ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        auto s01 = src0->nb[1] / ts0;
        auto s02 = src0->nb[2] / ts0;
        auto s03 = src0->nb[3] / ts0;
        rms_norm_f32_nc_cuda(src0_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    if (ggml_is_contiguous(src0)) {
        const int64_t nrows = ggml_nrows(src0);
        l2_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
    } else {
        const size_t ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        const int64_t s01 = src0->nb[1] / ts0;
        const int64_t s02 = src0->nb[2] / ts0;
        const int64_t s03 = src0->nb[3] / ts0;
        l2_norm_f32_nc_cuda(src0_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }
}

void ggml_cuda_op_fused_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool is_norm) {
    if (!dst->src[1]) {
        ggml_cuda_op_rms_norm(ctx, dst);
        return;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
               (ggml_is_contiguous(src0) && src0->type == GGML_TYPE_Q8_0));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_nrows(src1) == 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];

    if (ggml_is_contiguous(src0)) {
        const int64_t nrows = ggml_nrows(src0);
        if (src0->type == GGML_TYPE_F32) {
            fused_rms_norm_f32_cuda(src0_d, src1_d, dst_d, ne00, nrows, eps, is_norm, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            fused_rms_norm_f32_cuda((const half *)src0_d, src1_d, dst_d, ne00, nrows, eps, is_norm, stream);
        } else if (src0->type == GGML_TYPE_Q8_0) {
            fused_rms_norm_f32_cuda((const block_q8_0 *)src0_d, src1_d, dst_d, ne00, nrows, eps, is_norm, stream);
        } else {
            fused_rms_norm_f32_cuda((const nv_bfloat16 *)src0_d, src1_d, dst_d, ne00, nrows, eps, is_norm, stream);
        }
    } else {
        if (is_norm) {
            GGML_ABORT("Non-contiguous norm is not implemented");
        }
        auto ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        auto s01 = src0->nb[1] / ts0;
        auto s02 = src0->nb[2] / ts0;
        auto s03 = src0->nb[3] / ts0;
        if (src0->type == GGML_TYPE_F32) {
            fused_rms_norm_f32_nc_cuda(src0_d, src1_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            fused_rms_norm_f32_nc_cuda((const nv_bfloat16 *)src0_d, src1_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
        } else {
            fused_rms_norm_f32_nc_cuda((const half *)src0_d, src1_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
        }
    }
}

template <int block_size>
static __global__ void fused_add_rms_norm_f32(const float * a, const float * b, const float * c,
        float * dst_add, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a[row*ncols + col] + b[row*ncols + col];
        tmp += xi * xi;
        dst_add[row*ncols + col] = xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * c[col] * dst_add[row*ncols + col];
    }
}

template <int block_size>
static __global__ void fused_add_add_rms_norm_f32(const float * a1, const float * a2, const float * b, const float * c,
        float * dst_add, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a1[row*ncols + col] + a2[row*ncols + col] + b[row*ncols + col];
        tmp += xi * xi;
        dst_add[row*ncols + col] = xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * c[col] * dst_add[row*ncols + col];
    }
}

static void fused_add_rms_norm_f32_cuda(const float * a, const float * b, const float * c, float * dst_add, float * dst,
        const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        fused_add_rms_norm_f32<256><<<nrows, block_dims, 0, stream>>>(a, b, c, dst_add, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_add_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(a, b, c, dst_add, dst, ncols, eps);
    }
}

void ggml_cuda_op_fused_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    //const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(add->data == src0->data);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(add->src[0]));
    GGML_ASSERT(ggml_is_contiguous(add->src[1]));
    GGML_ASSERT(ggml_are_same_shape(add->src[0], add->src[1]));
    GGML_ASSERT(ggml_are_same_shape(add->src[0], src0));
    GGML_ASSERT(add->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(add->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_nrows(src1) == 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];

    const int64_t nrows = ggml_nrows(src0);
    fused_add_rms_norm_f32_cuda((const float *)add->src[0]->data, (const float *)add->src[1]->data,
            src1_d, (float *)add->data, dst_d, ne00, nrows, eps, stream);
}

static void fused_add_add_rms_norm_f32_cuda(const float * a1, const float * a2, const float * b, const float * c, float * dst_add, float * dst,
        const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        fused_add_add_rms_norm_f32<256><<<nrows, block_dims, 0, stream>>>(a1, a2, b, c, dst_add, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_add_add_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(a1, a2, b, c, dst_add, dst, ncols, eps);
    }
}

void ggml_cuda_op_fused_add_add_rms_norm(ggml_backend_cuda_context & ctx,
        ggml_tensor * add1, ggml_tensor * add2, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    //const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(add1->data == add2->src[0]->data);
    GGML_ASSERT(add2->data == src0->data);
    GGML_ASSERT(ggml_is_contiguous(src0));
    //GGML_ASSERT(ggml_is_contiguous(add->src[0]));
    //GGML_ASSERT(ggml_is_contiguous(add->src[1]));
    //GGML_ASSERT(ggml_are_same_shape(add->src[0], add->src[1]));
    //GGML_ASSERT(ggml_are_same_shape(add->src[0], src0));
    //GGML_ASSERT(add->src[0]->type == GGML_TYPE_F32);
    //GGML_ASSERT(add->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_nrows(src1) == 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];

    const int64_t nrows = ggml_nrows(src0);
    fused_add_add_rms_norm_f32_cuda((const float *)add1->src[0]->data, (const float *)add1->src[1]->data, (const float *)add2->src[1]->data,
            src1_d, (float *)add2->data, dst_d, ne00, nrows, eps, stream);
}

template <int block_size>
static __global__ void fused_rms_rms_norm_f32(int ncols, int nrows1, int nrows2, size_t nb1, size_t nb2, float eps,
        const char *x1, const char * x2, const float * c1, const float * c2, float * y1, float * y2) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    auto x_row = (const float *)(row < nrows1 ? x1 + row*nb1 : x2 + (row - nrows1)*nb2);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x_row[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    auto dst = row < nrows1 ? y1 + row*ncols : y2 + (row - nrows1)*ncols;
    auto   c = row < nrows1 ? c1 : c2;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * c[col] * x_row[col];
    }
}

static void fused_rms_rms_norm_f32_cuda(int ncols, int nrows1, int nrows2, size_t nb1, size_t nb2, float eps,
        const char * x1, const char * x2, const float * c1, const float * c2, float * y1, float * y2, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    int nrows = nrows1 + nrows2;
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        fused_rms_rms_norm_f32<256><<<nrows, block_dims, 0, stream>>>(ncols, nrows1, nrows2, nb1, nb2, eps, x1, x2, c1, c2, y1, y2);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_rms_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(ncols, nrows1, nrows2, nb1, nb2, eps, x1, x2, c1, c2, y1, y2);
    }
}

void ggml_cuda_op_fused_rms_rms_norm([[maybe_unused]] ggml_backend_cuda_context & ctx, [[maybe_unused]] ggml_tensor * rms1, [[maybe_unused]] ggml_tensor * rms2) {
    GGML_ASSERT(rms1->ne[2] == 1 && rms1->ne[3] == 1);
    GGML_ASSERT(rms2->ne[2] == 1 && rms2->ne[3] == 1);
    GGML_ASSERT(rms1->ne[0] == rms2->ne[0]);
    GGML_ASSERT(rms1->type == GGML_TYPE_F32);
    GGML_ASSERT(rms2->type == GGML_TYPE_F32);
    GGML_ASSERT(rms1->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms2->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms1->src[0]->ne[0] == rms1->src[1]->ne[0]);
    GGML_ASSERT(rms2->src[0]->ne[0] == rms2->src[1]->ne[0]);
    GGML_ASSERT(ggml_nrows(rms1->src[1]) == 1);
    GGML_ASSERT(ggml_nrows(rms2->src[1]) == 1);
    GGML_ASSERT(rms1->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms2->src[1]->type == GGML_TYPE_F32);

    float eps1, eps2;
    memcpy(&eps1, rms1->op_params, sizeof(float));
    memcpy(&eps2, rms2->op_params, sizeof(float));
    GGML_ASSERT(eps1 == eps2);

    fused_rms_rms_norm_f32_cuda(rms1->ne[0], rms1->ne[1], rms2->ne[1], rms1->nb[1], rms2->nb[1], eps1,
            (const char  *)rms1->src[0]->data, (const char *)rms2->src[0]->data,
            (const float *)rms1->src[1]->data, (const float *)rms2->src[1]->data,
            (float *)rms1->data, (float *)rms2->data, ctx.stream());


}

template <int block_size, typename src_t>
static __global__ void fused_rms_rms_add_f32(int ncols, int nrows, float * dst,
        const src_t * x1, const float * c1, const src_t * x2, const float * c2, float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    auto x1_row = x1 + row*ncols;
    auto x2_row = x2 + row*ncols;

    float tmp1 = 0.0f, tmp2 = 0.0f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi1 = (float)x1_row[col];
        const float xi2 = (float)x2_row[col];
        tmp1 += xi1 * xi1;
        tmp2 += xi2 * xi2;
    }

    tmp1 = warp_reduce_sum(tmp1);
    tmp2 = warp_reduce_sum(tmp2);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[2*WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[2*warp_id+0] = tmp1;
            s_sum[2*warp_id+1] = tmp2;
        }
        __syncthreads();
        tmp1 = lane_id < block_size/WARP_SIZE ? s_sum[2*lane_id+0] : 0.0f;
        tmp2 = lane_id < block_size/WARP_SIZE ? s_sum[2*lane_id+1] : 0.0f;
        tmp1 = warp_reduce_sum(tmp1);
        tmp2 = warp_reduce_sum(tmp2);
    }

    const float mean1 = tmp1 / ncols;
    const float mean2 = tmp2 / ncols;
    const float scale1 = rsqrtf(mean1 + eps);
    const float scale2 = rsqrtf(mean2 + eps);

    dst += row*ncols;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale1 * c1[col] * (float)x1_row[col] + scale2 * c2[col] * (float)x2_row[col];
    }
}

template <typename src_t>
static void fused_rms_rms_add_f32_cuda(int ncols, int nrows, float * dst,
        const src_t * x1, const float * c1, const src_t * x2, const float * c2,
        float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        fused_rms_rms_add_f32<256><<<nrows, block_dims, 0, stream>>>(ncols, nrows, dst, x1, c1, x2, c2, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_rms_rms_add_f32<1024><<<nrows, block_dims, 0, stream>>>(ncols, nrows, dst, x1, c1, x2, c2, eps);
    }
}

void ggml_cuda_op_fused_rms_rms_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst->src[2]));
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nrows(dst->src[1]) == 1 && dst->src[1]->ne[0] == dst->src[0]->ne[0]);
    GGML_ASSERT(ggml_nrows(dst->src[3]) == 1 && dst->src[3]->ne[0] == dst->src[2]->ne[0]);
    GGML_ASSERT(dst->src[0]->type == dst->src[2]->type);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32 && dst->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    int nrows = ggml_nrows(dst);
    int ncols = dst->ne[0];

    if (dst->src[0]->type == GGML_TYPE_F32) {
        fused_rms_rms_add_f32_cuda(ncols, nrows, (float *)dst->data,
                (const float *)dst->src[0]->data, (const float *)dst->src[1]->data,
                (const float *)dst->src[2]->data, (const float *)dst->src[3]->data,
                eps, ctx.stream());
    }
    else if (dst->src[0]->type == GGML_TYPE_F16) {
        fused_rms_rms_add_f32_cuda(ncols, nrows, (float *)dst->data,
                (const half *)dst->src[0]->data, (const float *)dst->src[1]->data,
                (const half *)dst->src[2]->data, (const float *)dst->src[3]->data,
                eps, ctx.stream());
    }
    else if (dst->src[0]->type == GGML_TYPE_BF16) {
        fused_rms_rms_add_f32_cuda(ncols, nrows, (float *)dst->data,
                (const nv_bfloat16 *)dst->src[0]->data, (const float *)dst->src[1]->data,
                (const nv_bfloat16 *)dst->src[2]->data, (const float *)dst->src[3]->data,
                eps, ctx.stream());
    }
    else {
        GGML_ABORT("Not implemented");
    }
}

// G2-F3 NORMFUSE launcher/entry (see kernel comment above). Returns false when the shape is
// outside the bit-exact-parity envelope (ncols < 256 would pick a different block size than the
// standalone launcher) — the caller then falls back to the plain norm + quantize path.
static bool fused_rms_norm_q8_f32_cuda(const float * x, const float * y, float * dst, void * q8,
        const int ncols, const int ncols_padded, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 256 || ncols % WARP_SIZE != 0 || ncols_padded % QK8_1 != 0) return false;
    if (ncols < 1024) {
        fused_rms_norm_q8_f32<256><<<nrows, 256, 0, stream>>>(x, y, dst, q8, ncols, ncols_padded, eps);
    } else {
        fused_rms_norm_q8_f32<1024><<<nrows, 1024, 0, stream>>>(x, y, dst, q8, ncols, ncols_padded, eps);
    }
    return true;
}

bool ggml_cuda_op_fused_rms_norm_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8, int64_t ncols_padded) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src1) return false;
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;
    if (ggml_nrows(src1) != 1 || src0->ne[0] != src1->ne[0]) return false;
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    return fused_rms_norm_q8_f32_cuda((const float *)src0->data, (const float *)src1->data, (float *)dst->data,
            q8, src0->ne[0], (int)ncols_padded, ggml_nrows(src0), eps, ctx.stream());
}
