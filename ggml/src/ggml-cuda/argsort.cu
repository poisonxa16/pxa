//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "argsort.cuh"
#include "pxa-env-flag.h"
#include "sumrows.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>

//
// ---------------------------------------------------------------------------
// CUB-backed argsort. Taken substantially verbatim from upstream llama.cpp
// ggml/src/ggml-cuda/argsort.cu @ commit 82dbc4f01.
// Copyright (c) 2023-2026 The ggml authors. MIT license.
//
// WHY: this file's bitonic kernel launches one thread per padded column
// (block_dims(next_power_of_2(ncols))), so a CUDA block's 1024-thread limit caps
// it at ncols <= 1024. DeepSeek-V4's lightning indexer needs top-512 out of
// ceil(n_ctx/4) candidates - 2048 at 8k context, 32768 at 128k - which aborts on
// that assert. CUB DeviceSegmentedRadixSort / DeviceSegmentedSort are plain
// radix sorts with no compute-capability floor, so this path is valid on
// sm_60 (P100), sm_61 (1080 Ti) and sm_70 (V100).
//
// NOTE: upstream's faster cub::DeviceTopK path needs CCCL >= 3.2 (CUDA 13) and is
// deliberately not ported - on CUDA 12.x we get the radix-sort path, which is the
// same path upstream itself takes there.
//
// This does NOT touch GGML_OP_ARGSORT_THRESH / ggml_top_k_thresh / grouped_topk,
// which are fork-local MoE-routing work with no upstream counterpart and stay on
// the bitonic kernel.
// ---------------------------------------------------------------------------
//

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define PXA_CUB_STRIDED_ITERATOR_AVAILABLE
#        include <cuda/iterator>
#    endif
using namespace cub;

static __global__ void pxa_cub_init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#    ifndef PXA_CUB_STRIDED_ITERATOR_AVAILABLE
static __global__ void pxa_cub_init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#    endif // PXA_CUB_STRIDED_ITERATOR_AVAILABLE

// returns the suggested maximum number of rows to process during one
// argsort_f32_i32_cuda_cub() call
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows) {
    // perform argsort in chunks up to approximately this size (currently 64MB)
    // to avoid excessive temporary buffers memory usage
    const int chunk_bytes = 1 << 26;

    // calculate how many rows will fit in one chunk (must be at least one)
    const int chunk_nrows = std::max((int) (chunk_bytes / nb01), 1);

    // limit the resulting amount to total nrows
    return (int) std::min((int64_t) chunk_nrows, nrows);
}

void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, (size_t) ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, (size_t) ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    pxa_cub_init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#    ifdef PXA_CUB_STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#    else
    // offset_iterator needs to populate nrows + 1 elements, so we also have to
    // ceildiv nrows + 1 by block_size
    const int                 nrows_offset = nrows + 1;
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows_offset);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows_offset + block_size - 1) / block_size);
    pxa_cub_init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#    endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, (size_t) ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    bool is_capturing = false;
#    ifdef USE_CUDA_GRAPH
    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support
    // stream capture, while DeviceSegmentedRadixSort does.
    // See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = (capture_status != cudaStreamCaptureStatusNone);
#    endif // USE_CUDA_GRAPH

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                  temp_indices, dst,
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(
                nullptr, temp_storage_bytes, temp_keys, temp_keys,
                temp_indices, dst,
                ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys,
                                                      temp_keys,
                                                      temp_indices, dst,
                                                      ncols * nrows, nrows,
                                                      offset_iterator, offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys,
                                                            temp_indices, dst,
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                                temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys,
                                                  temp_indices, dst,
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                           offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                      temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                      offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys,
                                                            temp_indices, dst,
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                                temp_keys, temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }
}
#endif // GGML_CUDA_USE_CUB

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

struct store_ser {
    constexpr static bool has_thresh = true;
    int   min_experts;
    float thresh_experts;
    store_ser(int min, float thresh) : min_experts(min), thresh_experts(thresh) {}
};

struct store {
    constexpr static bool has_thresh = false;
};

template<ggml_sort_order order>
static __device__ __forceinline__ void sort(int ncols_pad, int ncols, int col, const float * x_row, int * dst_row) {
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<ggml_sort_order order, typename Store, typename dst_t>
static __global__ void k_argsort_f32_T(const float * x, dst_t * dst, const int ncols, int ncols_pad, int ntop, Store s) {
//        int min_experts, float thresh_experts) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if constexpr (Store::has_thresh) {
        __syncthreads();
        float max_val = x_row[dst_row[0]];
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts*max_val ? dst_row[col] : -1;
            } else {
                dst[row * ntop + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts*max_val ? x_row[dst_row[col]] : 0.f;
            }
        }
    } else {
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = dst_row[col];
            } else {
                dst[row * ntop + col] = x_row[dst_row[col]];
            }
        }
    }
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_u8(const float * x, uint8_t * dst, const int ncols, int ncols_pad, int ntop) {
//        int min_experts, float thresh_experts) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (col < ncols) dst[row*ncols + dst_row[col]] = col < ntop ? 1 : 0;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_f32_i32(const float * x_biased, const float * x, const uint8_t * group_mask,
        float * weights, int * ids, const int ncols, int ncols_pad, int ntop, size_t nb_ids, int n_per_group, int n_groups) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    extern __shared__ int dst_row[];
    auto x_row = (float *)(dst_row + ncols_pad);

    // initialize indices
    dst_row[col] = col;
    int ig = col / n_per_group;
    x_row[col] = ig < n_groups && group_mask[row*n_groups + ig] ? x_biased[row * ncols + col] : -INFINITY;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (col < ntop) {
        weights[row * ntop + col] = 1/(1 + expf(-x[row * ncols + dst_row[col]]));
        auto row_ids = (int *)((char *)ids + row*nb_ids);
        row_ids[col] = dst_row[col];
    }
}

template<ggml_sort_order order>
static __global__ void k_argsort_biased_f32_f32_i32(const float * x, const float * bias, float * weights, int * ids, const int ncols, int ncols_pad, int ntop,
        size_t nb_ids) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    extern __shared__ int dst_row[];
    auto x_row = (float *)(dst_row + ncols_pad);

    // initialize indices
    dst_row[col] = col;
    x_row[col]   = col < ncols ? 1/(1 + expf(-x[row*ncols + col])) + bias[col] : -INFINITY;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (col < ntop) {
        weights[row * ntop + col] = 1/(1 + expf(-x[row * ncols + dst_row[col]]));
        auto row_ids = (int *)((char *)ids + row*nb_ids);
        row_ids[col] = dst_row[col];
    }
}

template<ggml_sort_order order>
static __global__ void k_openai_f32_f32_i32(const float * x, float * weights, int * ids, const int ncols, int ncols_pad, int ntop,
        size_t nb_ids) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    extern __shared__ int dst_row[];
    auto x_row = x + row*ncols;

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    float max = x_row[dst_row[0]];
    float val = col < ntop ? expf(x_row[dst_row[col]] - max) : 0.0f;
    float sum = warp_reduce_sum(val);
    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
        float * s_sum = (float *)(dst_row + ncols_pad);
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = sum;
        }
        __syncthreads();
        sum = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            sum = s_sum[lane_id];
        }
        sum = warp_reduce_sum(sum);
    }
    float norm = 1/sum;
    if (col < ntop) {
        weights[row * ntop + col] = norm*val;
        auto row_ids = (int *)((char *)ids + row*nb_ids);
        row_ids[col] = dst_row[col];
    }
}

template<ggml_sort_order order>
static __global__ void k_topk_sum(const float * x, const float * bias, float * x_p, float * dst,
        const int ne00, const int ncols, int ncols_pad, int n_top_k) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;
    if (bias && x_p) {
        float * x_p_row = x_p + row * ncols;
        if (col < ncols) {
            x_p_row[col] = 1/(1 + expf(-x_row[col])) + bias[(row * ncols + col)%ne00];
        }
        x_row = x_p_row;
    }

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (n_top_k == 2) {
        float val = x_row[dst_row[0]] + x_row[dst_row[1]];
        if (col == 0) dst[row] = val;
    } else {
    float val = col < n_top_k ? x_row[dst_row[col]] : 0;
    val = warp_reduce_sum(val);
    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
        float * s_sum = (float *)(dst_row + ncols_pad);
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = val;
        }
        __syncthreads();
        val = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            val = s_sum[lane_id];
        }
        val = warp_reduce_sum(val);
    }

    if (col == 0) {
        dst[row] = val;
    }
    }
}

static __global__ void k_apply_mask(float * dst, const int * groups,
        const int n_top_groups, const int n_per_group, const int ncols) {
    int row = blockIdx.x;
    for (int col = threadIdx.x; col < n_top_groups*n_per_group; col += blockDim.x) {
        int ig = groups[row*n_top_groups + col / n_per_group];
        int ic = col % n_per_group;
        dst[row*ncols + ig*n_per_group + ic] = -INFINITY;
    }
}

// PXA_ARGSORT_SMEM (default ON, =0 off; 2026-10-23, ledger argsort-keys-smem): the same bitonic
// network as k_argsort_f32_T with each thread holding one (index, key) pair in registers. A stage whose partner distance j
// is below the warp size swaps through __shfl_xor_sync (no barrier); a wider stage exchanges through shared memory. The
// swap decision of a pair is the original lower thread's test on (index, key) of the lower and upper slots, computed
// identically by both threads, and the key a slot carries is always x[its index], so every compare sees the values the
// global-key kernel compares: same network, same ties, bit-identical indices and values (test-argsort-smem checks a host
// copy of the network). The global-key kernel spends ~0.4 us per stage on two dependent global loads and a barrier
// (a 256-wide Flint router sort: 36 stages, 16.9 us per call on a P100, 47 calls per token).
// The register/shuffle bitonic network shared by k_argsort_f32_T_reg and k_sigmoid_topk_norm_f32: thread col enters with
// (my_i = col, my_k = x[col] or 0 for a padded slot) and leaves holding the (index, key) of sorted position col.
// s_xchg = the two exchange buffers ([2][ncols_pad] indices, [2][ncols_pad] keys); every thread of the block must call it.
template<ggml_sort_order order>
static __device__ __forceinline__ void argsort_reg_network(const int col, const int ncols, const int ncols_pad,
        int & my_i, float & my_k, int * s_xchg) {
    int   * s_i = s_xchg;
    float * s_k = (float *)(s_xchg + 2*ncols_pad);
    int pb = 0;                                              // alternating buffers: one barrier per wide stage

    const unsigned mask = ncols_pad >= WARP_SIZE ? 0xffffffffu : ((1u << ncols_pad) - 1u);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int   o_i;
            float o_k;
            if (j >= WARP_SIZE) {
                // a thread that reaches the next wide stage has passed this barrier, so every read of the buffer it
                // then writes (two wide stages back) is done
                int   * bi = s_i + pb*ncols_pad;
                float * bk = s_k + pb*ncols_pad;
                pb ^= 1;
                bi[col] = my_i;
                bk[col] = my_k;
                __syncthreads();
                o_i = bi[col ^ j];
                o_k = bk[col ^ j];
            } else {
                o_i = __shfl_xor_sync(mask, my_i, j);
                o_k = __shfl_xor_sync(mask, my_k, j);
            }
            const bool  lower = (col & j) == 0;              // this thread holds the pair's lower slot
            const int   a_i = lower ? my_i : o_i;            // lower slot
            const float a_k = lower ? my_k : o_k;
            const int   b_i = lower ? o_i  : my_i;           // upper slot
            const float b_k = lower ? o_k  : my_k;
            bool swap;
            if ((col & k) == 0) {                            // bit k is the same for both slots (j < k)
                swap = a_i >= ncols || (b_i < ncols && (order == GGML_SORT_ORDER_ASC ? a_k > b_k : a_k < b_k));
            } else {
                swap = b_i >= ncols || (a_i < ncols && (order == GGML_SORT_ORDER_ASC ? a_k < b_k : a_k > b_k));
            }
            if (swap) {
                my_i = o_i;
                my_k = o_k;
            }
        }
    }
}

template<ggml_sort_order order, typename Store, typename dst_t>
static __global__ void k_argsort_f32_T_reg(const float * x, dst_t * dst, const int ncols, int ncols_pad, int ntop, Store s) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int s_xchg[];                          // two exchange buffers: [2][ncols_pad] indices, [2][ncols_pad] keys

    int   my_i = col;
    float my_k = col < ncols ? x_row[col] : 0.0f;            // a padded slot's key is never compared (index test first)

    argsort_reg_network<order>(col, ncols, ncols_pad, my_i, my_k, s_xchg);

    if constexpr (Store::has_thresh) {
        __shared__ float s_max;                              // the first sorted key == x_row[dst_row[0]]
        if (col == 0) {
            s_max = my_k;
        }
        __syncthreads();
        const float max_val = s_max;
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = col < s.min_experts || my_k >= s.thresh_experts*max_val ? my_i : -1;
            } else {
                dst[row * ntop + col] = col < s.min_experts || my_k >= s.thresh_experts*max_val ? my_k : 0.f;
            }
        }
    } else {
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = my_i;
            } else {
                dst[row * ntop + col] = my_k;
            }
        }
    }
}

// PXA_ARGSORT_SMEM: default ON (k_argsort_f32_T_reg); =0 restores the global-key bitonic kernel (read once per process).
static bool pxa_argsort_smem_on() {
    static const bool v = [] {
        const char * e = getenv("PXA_ARGSORT_SMEM");
        return !pxa_env_value_off(e);
    }();
    return v;
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template <typename dst_t>
static void argsort_f32_T_cuda(const float * x, dst_t * dst, const int ncols, const int nrows, int ntop,
        ggml_sort_order order, int min_experts, float thresh_experts, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    const size_t smpb = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    GGML_ASSERT(shared_mem <= smpb);

    // PXA_ARGSORT_SMEM: the register/shuffle network when its two index + key exchange buffers fit (bit-identical, see the kernel)
    const size_t shared_mem_keys = 2 * ncols_pad * (sizeof(int) + sizeof(float));
    if (pxa_argsort_smem_on() && shared_mem_keys <= smpb) {
        if (order == GGML_SORT_ORDER_ASC) {
            if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
                k_argsort_f32_T_reg<GGML_SORT_ORDER_ASC, store_ser, dst_t><<<block_nums, block_dims, shared_mem_keys, stream>>>(x, dst, ncols, ncols_pad,
                        ntop, {min_experts, thresh_experts});
            } else {
                k_argsort_f32_T_reg<GGML_SORT_ORDER_ASC, store, dst_t><<<block_nums, block_dims, shared_mem_keys, stream>>>(x, dst, ncols, ncols_pad, ntop, {});
            }
            return;
        } else if (order == GGML_SORT_ORDER_DESC) {
            if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
                k_argsort_f32_T_reg<GGML_SORT_ORDER_DESC, store_ser, dst_t><<<block_nums, block_dims, shared_mem_keys, stream>>>(x, dst, ncols, ncols_pad,
                        ntop, {min_experts, thresh_experts});
            } else {
                k_argsort_f32_T_reg<GGML_SORT_ORDER_DESC, store, dst_t><<<block_nums, block_dims, shared_mem_keys, stream>>>(x, dst, ncols, ncols_pad, ntop, {});
            }
            return;
        }
    }

    if (order == GGML_SORT_ORDER_ASC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_T<GGML_SORT_ORDER_ASC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad,
                    ntop, {min_experts, thresh_experts});
        } else {
            k_argsort_f32_T<GGML_SORT_ORDER_ASC, store><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop, {});
        }
    } else if (order == GGML_SORT_ORDER_DESC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_T<GGML_SORT_ORDER_DESC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad,
                    ntop, {min_experts, thresh_experts});
        } else {
            k_argsort_f32_T<GGML_SORT_ORDER_DESC, store><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop, {});
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

// PXA_FLINT_GATE_FUSE (default OFF since #354, =1 opt-in; 2026-10-23, ledger flint-gate-fuse): the norm_w sigmoid router
// tail without a selection bias (Flint: SIGMOID -> ARGSORT DESC -> VIEW top-k -> GET_ROWS -> SUM_ROWS -> DIV) as ONE launch,
// one block per token row. Every intermediate tensor is still written (probs, the full argsort, the gathered weights, the
// row sum, the normalized weights), each with the arithmetic of the kernel it replaces: sigmoid_f32's expression, the
// k_argsort_f32_T_reg network (argsort_reg_network: the key a slot carries is x[its index], so the gathered weight of sorted
// position j is the key thread j holds = get_rows' copy), and k_sum_rows_div_f32's warp-0 sum (lane j < k adds weight j to
// 0.0f, warp_reduce_sum, norm = 1/sum or 0, weight * norm). Bit-identical; test-flint-gate-fuse compares it with the op chain.
// #354 (2026-10-24): the graph allocator lays the chain out for its op order, where probs (and logits)
// are dead before SUM_ROWS/DIV run, so it may place wsum/wn (or ids/w) inside a buffer the chain already freed. One launch
// writes them all at once from different blocks, so such a layout is an inter-block race (P100 one-card Flint: block r''
// writing probs row r'' over the wn rows of a block that already finished; greedy output differed run to run).
// ggml_cuda_sigmoid_topk_norm checks the byte ranges first: probs shared only with w/wsum/wn -> probs is not written (it is
// dead); any other sharing except the exact in-place pairs (probs == logits, wn == w; one thread reads then writes the same
// element) -> the fusion declines and the op chain runs. Every tensor that is live after the tail gets the same bytes.
static __global__ void k_sigmoid_topk_norm_f32(const float * __restrict__ logits, float * __restrict__ probs, int * __restrict__ ids,
        float * __restrict__ w, float * __restrict__ wsum, float * __restrict__ wn, const int ncols, const int ncols_pad, const int k) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;

    extern __shared__ int s_xchg[];

    int   my_i = col;
    float my_k = 0.0f;                                       // a padded slot's key is never compared (index test first)
    if (col < ncols) {
        const float p = 1.0f / (1.0f + expf(-logits[row * ncols + col]));
        if (probs) {                                         // nullptr = probs is dead and its buffer holds w/wsum/wn (#354)
            probs[row * ncols + col] = p;
        }
        my_k = p;
    }

    argsort_reg_network<GGML_SORT_ORDER_DESC>(col, ncols, ncols_pad, my_i, my_k, s_xchg);

    if (col < ncols) {
        ids[row * ncols + col] = my_i;
    }
    if (col < WARP_SIZE) {                                   // warp 0 = k_sum_rows_div_f32's 32-thread block (k <= 32)
        float sum = 0.0f;
        if (col < k) {
            w[row * k + col] = my_k;
            sum += my_k;
        }
        sum = warp_reduce_sum(sum);
        float norm = sum > 0 ? 1/sum : 0.0f;
        if (col < k) {
            wn[row * k + col] = my_k * norm;
        }
        if (col == 0) {
            wsum[row] = sum;
        }
    }
}

// default OFF since 2026-10-24: a measured wash that raced once on the default path (#354);
// PXA_FLINT_GATE_FUSE=1 opts in (the #354 alias guard below still applies)
static bool pxa_flint_gate_fuse_on() {
    static const bool v = pxa_env_flag("PXA_FLINT_GATE_FUSE", 0) != 0;
    return v;
}

bool ggml_cuda_sigmoid_topk_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, ggml_tensor * probs, ggml_tensor * ids,
        ggml_tensor * w, ggml_tensor * wsum, ggml_tensor * wn, int k) {
    if (!pxa_flint_gate_fuse_on() || !pxa_argsort_smem_on()) {
        return false;
    }
    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || w->type != GGML_TYPE_F32 ||
        wsum->type != GGML_TYPE_F32 || wn->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(probs) || !ggml_is_contiguous(ids) || !ggml_is_contiguous(w) ||
        !ggml_is_contiguous(wsum) || !ggml_is_contiguous(wn)) {
        return false;
    }
    const int64_t ncols = probs->ne[0];
    const int64_t nrows = ggml_nrows(probs);
    if (!ggml_are_same_shape(logits, probs) || !ggml_are_same_shape(ids, probs) || ncols > 1024 || nrows > INT_MAX ||
        k < 1 || k > WARP_SIZE || k > ncols || ggml_nelements(w) != (int64_t) k*nrows || ggml_nelements(wn) != (int64_t) k*nrows ||
        ggml_nelements(wsum) != nrows) {
        return false;
    }
    const int ncols_pad = next_power_of_2((int) ncols);
    const size_t smpb = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const size_t shared_mem_keys = 2 * ncols_pad * (sizeof(int) + sizeof(float));
    // the reference ARGSORT takes k_argsort_f32_T_reg only in this range (CUB above 1024 columns or past smpb)
    if (ncols_pad < WARP_SIZE || ncols_pad > 1024 || (size_t) ncols_pad * sizeof(int) > smpb || shared_mem_keys > smpb) {
        return false;
    }
    // #354: buffers the allocator shared inside the chain (see the kernel comment)
    auto ov = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    const ggml_tensor * late[3] = { w, wsum, wn };
    bool probs_live = true;
    bool unsafe = (ov(probs, logits) && probs->data != logits->data) || ov(ids, logits) || ov(ids, probs) ||
                  ov(w, wsum) || ov(wsum, wn) || (ov(w, wn) && w->data != wn->data);
    for (const ggml_tensor * t : late) {
        unsafe = unsafe || ov(t, logits) || ov(t, ids);  // a row another block may still read (logits) or a live output (ids)
        if (ov(t, probs)) {
            probs_live = false;                          // probs was freed inside the chain: nothing reads it after the tail
        }
    }
    if (unsafe) {
        static bool said_decl = false;
        if (!said_decl) {
            said_decl = true;
            fprintf(stderr, "PXA_FLINT_GATE_FUSE: declined where the allocator shares a buffer inside the router tail (op chain runs; #354)\n");
        }
        return false;
    }
    if (!probs_live) {
        static bool said_dead = false;
        if (!said_dead) {
            said_dead = true;
            fprintf(stderr, "PXA_FLINT_GATE_FUSE: router probs buffer reused by the tail outputs (dead after it): probs not written (#354)\n");
        }
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "PXA_FLINT_GATE_FUSE: ON (opt-in; sigmoid + top-%d + norm_w router tail as one launch, %d experts, bit-identical)\n",
                k, (int) ncols);
    }
    k_sigmoid_topk_norm_f32<<<dim3((unsigned) nrows, 1, 1), dim3(ncols_pad, 1, 1), shared_mem_keys, ctx.stream()>>>(
            (const float *) logits->data, probs_live ? (float *) probs->data : nullptr, (int *) ids->data, (float *) w->data, (float *) wsum->data,
            (float *) wn->data, (int) ncols, ncols_pad, k);
    return true;
}

static void argsort_f32_u8_cuda(const float * x, uint8_t * dst, const int ncols, const int nrows, int ntop,
        ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_u8<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_u8<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argsort_f32_f32_i32_cuda(const float * x_biased, const float * x, const uint8_t * group_mask,
        float * weights, int * ids, const int ncols, const int nrows, int ntop,
        size_t nb_ids, int n_per_group, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x_biased, x, group_mask, weights, ids,
                ncols, ncols_pad, ntop, nb_ids, n_per_group, ncols/n_per_group);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x_biased, x, group_mask, weights, ids,
                ncols, ncols_pad, ntop, nb_ids, n_per_group, ncols/n_per_group);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argsort_biased_f32_f32_i32_cuda(const float * x, const float * bias, float * weights, int * ids, const int ncols, const int nrows, int ntop,
        size_t nb_ids, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_biased_f32_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, bias, weights, ids,
                ncols, ncols_pad, ntop, nb_ids);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_biased_f32_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, bias, weights, ids,
                ncols, ncols_pad, ntop, nb_ids);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argsort_openai_f32_f32_i32_cuda(const float * x, float * weights, int * ids, const int ncols, const int nrows, int ntop,
        size_t nb_ids, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    // k_openai_f32_f32_i32 always stores dst_row[ncols_pad] in dynamic shared memory.
    // When blockDim > WARP_SIZE it also uses a per-warp float scratch segment (s_sum).
    const int n_warps = (ncols_pad + WARP_SIZE - 1) / WARP_SIZE;
    const size_t shared_mem = ncols_pad * sizeof(int) + (ncols_pad > WARP_SIZE ? n_warps * sizeof(float) : 0);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_openai_f32_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, weights, ids,
                ncols, ncols_pad, ntop, nb_ids);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_openai_f32_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, weights, ids,
                ncols, ncols_pad, ntop, nb_ids);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

#ifdef GGML_CUDA_USE_CUB
    // The bitonic kernel launches next_power_of_2(ncols) threads in ONE block, so
    // it dies on the 1024-thread block limit above ncols == 1024. Route wider rows
    // to CUB. Upstream llama.cpp does exactly this (ggml_cuda_op_argsort @ 82dbc4f01).
    // ncols <= 1024 keeps the existing kernel bit-for-bit.
    {
        const int    ncols_pad      = next_power_of_2((int) ncols);
        const size_t shared_mem     = (size_t) ncols_pad * sizeof(int);
        const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

        if (ncols > 1024 || shared_mem > max_shared_mem) {
            const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

            ggml_cuda_pool & pool = ctx.pool();

            const float * x_d = src0_d;
            int *         o_d = (int *) dst_d;

            for (int64_t i = 0; i < nrows; i += chunk_nrows) {
                const int iter_nrows = (int) std::min((int64_t) chunk_nrows, nrows - i);

                argsort_f32_i32_cuda_cub(pool, x_d, o_d, (int) ncols, iter_nrows, order, stream);

                x_d += (size_t) ncols * iter_nrows;
                o_d += (size_t) ncols * iter_nrows;
            }

            return;
        }
    }
#endif // GGML_CUDA_USE_CUB

    argsort_f32_T_cuda(src0_d, (int *)dst_d, ncols, nrows, ncols, order, -1, 0.f, stream);
}

void ggml_cuda_op_argsort_thresh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    int min_experts = dst->op_params[0];
    float thresh;
    memcpy(&thresh, dst->op_params + 1, sizeof(float));

    argsort_f32_T_cuda(src0_d, (int *)dst_d, ncols, nrows, ncols, GGML_SORT_ORDER_DESC, min_experts, thresh, stream);
}

static void ggml_cuda_op_topk_sum(ggml_backend_cuda_context & ctx, const float * src, const float * bias, float * src_p, float * dst,
        int ne00, int ncols, int nrows, int n_top_k) {

    GGML_ASSERT(n_top_k <= ncols);

    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = (ncols_pad + WARP_SIZE) * sizeof(int);
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    k_topk_sum<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, ctx.stream()>>>(src, bias, src_p, dst, ne00, ncols, ncols_pad, n_top_k);
}

void ggml_cuda_op_grouped_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nrows(src) == ggml_nrows(dst));

    auto nrows = ggml_nrows(src);

    int n_groups     = dst->op_params[0];
    int n_top_groups = dst->op_params[1];
    int nk           = dst->op_params[2];

    int ne00 = src->ne[0];
    int ne0  = dst->ne[0];
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups < n_groups);
    int n_discarded_groups = n_groups - n_top_groups;

#if 0
    ggml_cuda_pool_alloc<float> sorted_group_scores(ctx.pool(), nk*nrows*n_groups);
    argsort_f32_T_cuda((const float *)src->data, sorted_group_scores.get(), n_per_group, nrows*n_groups, nk,
            GGML_SORT_ORDER_DESC, -1, 0.0f, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_pool_alloc<float> group_scores(ctx.pool(), nrows*n_groups);
    sum_rows_f32_cuda((const float *)sorted_group_scores.get(), group_scores.get(), nk, nrows*n_groups, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
#else
    ggml_cuda_pool_alloc<float> group_scores(ctx.pool(), nrows*n_groups);
    ggml_cuda_op_topk_sum(ctx, (float *)src->data, nullptr, nullptr, group_scores.get(), ne00, n_per_group, nrows*n_groups, nk);
    CUDA_CHECK(cudaGetLastError());
#endif

    ggml_cuda_pool_alloc<int> discarded_groups(ctx.pool(), nrows*n_discarded_groups);
    argsort_f32_T_cuda(group_scores.get(), discarded_groups.get(), n_groups, nrows, n_discarded_groups, GGML_SORT_ORDER_ASC, -1, 0.0f, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const dim3 block_nums(nrows, 1, 1);
        cudaStream_t stream = ctx.stream();
        k_apply_mask<<<block_nums, block_dims, 0, ctx.stream()>>>((float *)src->data, discarded_groups.get(), n_discarded_groups, n_per_group, ne00);
        CUDA_CHECK(cudaGetLastError());
    }

    argsort_f32_T_cuda((const float *)src->data, (int *)dst->data, ne00, nrows, ne0, GGML_SORT_ORDER_DESC, -1, 0.0f, ctx.stream());

}

void cuda_bailingmoev2_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk) {
    auto topk_src = topk->src[0];
    auto probs    = topk_src->src[0]->src[0];
    auto bias     = topk_src->src[1];

    auto nrows = ggml_nrows(probs);

    int n_groups     = topk->op_params[0];
    int n_top_groups = topk->op_params[1];
    int nk           = topk->op_params[2];

    int ne00 = probs->ne[0];
    int ne0  = topk->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(bias->ne[1] == 1);
    GGML_ASSERT(bias->ne[0] == probs->ne[0]);
    GGML_ASSERT(ne0 == dst->ne[1]);
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups <= n_groups);

    ggml_cuda_pool_alloc<float> group_scores(ctx.pool(), nrows*n_groups);
    ggml_cuda_op_topk_sum(ctx, (const float *)probs->data, (const float *)bias->data, (float *)topk_src->data, group_scores.get(),
            ne00, n_per_group, nrows*n_groups, nk);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<uint8_t> group_mask(ctx.pool(), nrows*n_groups);
    argsort_f32_u8_cuda(group_scores.get(), group_mask.get(), n_groups, nrows, n_top_groups, GGML_SORT_ORDER_DESC, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    argsort_f32_f32_i32_cuda((const float *)topk_src->data, (const float *)probs->data, group_mask.get(),
            (float *)dst->data, (int *)topk->data,
            ne00, nrows, ne0, topk->nb[1], n_per_group, GGML_SORT_ORDER_DESC, ctx.stream());

}

void cuda_glm45moe_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk_view) {
    GGML_ASSERT(topk_view->op == GGML_OP_VIEW);
    auto topk     = topk_view->src[0];
    auto topk_src = topk->src[0];
    auto probs    = topk_src->src[0]->src[0];
    auto bias     = topk_src->src[1];

    auto nrows = ggml_nrows(probs);

    int ne00 = probs->ne[0];
    int ne0  = topk_view->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(bias->ne[1] == 1);
    GGML_ASSERT(bias->ne[0] == probs->ne[0]);
    GGML_ASSERT(ne0 == dst->ne[1]);
    GGML_ASSERT(ne0 <= ne00);

    argsort_biased_f32_f32_i32_cuda((const float *)probs->data, (const float *)bias->data, (float *)dst->data, (int *)topk->data,
            ne00, nrows, ne0, topk->nb[1], GGML_SORT_ORDER_DESC, ctx.stream());

}

void cuda_openai_experts(ggml_backend_cuda_context & ctx, ggml_tensor * topk, ggml_tensor * softmax) {

    auto probs    = topk->src[0];
    int  ntop     = topk->op_params[1];

    auto nrows = ggml_nrows(probs);
    int ne00 = probs->ne[0];
    int ne0  = softmax->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(ggml_is_contiguous(softmax));
    GGML_ASSERT(ne0 <= ne00);
    if (ntop != ne0) {
        printf("Oops: ntop = %d, ne0 = %d\n", ntop, ne0);
        GGML_ASSERT(false);
    }
    //GGML_ASSERT(ne0 == ntop);

    argsort_openai_f32_f32_i32_cuda((const float *)probs->data, (float *)softmax->data, (int *)topk->data,
            ne00, nrows, ne0, topk->nb[1], GGML_SORT_ORDER_DESC, ctx.stream());

}
