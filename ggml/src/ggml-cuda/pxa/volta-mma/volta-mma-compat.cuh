#pragma once
// -----------------------------------------------------------------------------
// PXA compatibility shim for the vendored mainline llama.cpp Volta MMA
// flash-attention path (see mma-ml.cuh / fattn-mma-ml.cuh).
//
// The vendored files are taken verbatim from ggml-org/llama.cpp commit
// 9400c8946e4da5e7694f2c26d6d4e50e14b690fa (2026-09-02), which is MIT licensed:
//   MIT License, Copyright (c) 2023-2026 The ggml authors
// Small excerpts of mainline's ggml/src/ggml-cuda/common.cuh are reproduced here
// under the same licence, because this fork's common.cuh is ik_llama-derived and
// does not carry them (different names, or absent entirely).
//
// This header maps mainline spellings onto this fork's common.cuh so the
// vendored math compiles unmodified.
// -----------------------------------------------------------------------------

#include "../../common.cuh"
#include "../../convert.cuh"   // PXA: to_fp16_cuda_t / ggml_get_to_fp16_(nc_)cuda live here in this fork
#include "../../../ggml-impl.h" // PXA: ggml_get_op_params_i32

#include <cstdint>
#include <limits>
#include <utility>

// ---- compute-capability spellings -------------------------------------------
// This fork uses CC_*; mainline uses GGML_CUDA_CC_*.
#ifndef GGML_CUDA_CC_PASCAL
#define GGML_CUDA_CC_PASCAL       600
#endif
#ifndef GGML_CUDA_CC_DP4A
#define GGML_CUDA_CC_DP4A         610
#endif
#ifndef GGML_CUDA_CC_VOLTA
#define GGML_CUDA_CC_VOLTA        700
#endif
#ifndef GGML_CUDA_CC_TURING
#define GGML_CUDA_CC_TURING       750
#endif
#ifndef GGML_CUDA_CC_AMPERE
#define GGML_CUDA_CC_AMPERE       800
#endif
#ifndef GGML_CUDA_CC_ADA_LOVELACE
#define GGML_CUDA_CC_ADA_LOVELACE 890
#endif
#ifndef GGML_CUDA_CC_HOPPER
#define GGML_CUDA_CC_HOPPER       900
#endif
#ifndef GGML_CUDA_CC_BLACKWELL
#define GGML_CUDA_CC_BLACKWELL    1200
#endif
#ifndef GGML_CUDA_CC_DGX_SPARK
#define GGML_CUDA_CC_DGX_SPARK    1210
#endif
#ifndef GGML_CUDA_CC_RUBIN
#define GGML_CUDA_CC_RUBIN        1300
#endif

// ---- device-side arch macros (mainline common.cuh:262-296) ------------------
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif
#ifndef FLASH_ATTN_AVAILABLE
#define FLASH_ATTN_AVAILABLE
#endif
// PXA: this fork spells the fast-fp16 device macro FP16_AVAILABLE/FAST_FP16_AVAILABLE
// in common.cuh already; only define them if it did not.
#if !defined(FAST_FP16_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif

// PDL is not enabled in this fork's build; keep mainline's spelling.
#ifndef GGML_CUDA_RESTRICT
#define GGML_CUDA_RESTRICT __restrict__
#endif

// ---- host-side arch predicates ----------------------------------------------
static bool pxa_ml_volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

// ---- mainline common.cuh:374 -------------------------------------------------
#ifndef PXA_ML_HAVE_PHYSICAL_WARP_SIZE
#define PXA_ML_HAVE_PHYSICAL_WARP_SIZE
static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 32; // PXA: CUDA-only fork build; the HIP 64-lane branch is dropped.
}
#endif

// ---- mainline common.cuh:744-784 (ggml_cuda_mad overloads) -------------------
static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// ---- mainline common.cuh:909-946 (fastdiv) ----------------------------------
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

// ---- mainline common.cuh:1554-1568 + 1659-1678 (kernel launch) --------------
struct ggml_cuda_kernel_launch_params {
    dim3 block_nums;
    dim3 block_dims;
    size_t shmem;
    cudaStream_t stream;

    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const size_t shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem(shmem_), stream(stream_) {}

    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const int shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem((size_t)shmem_), stream(stream_) {}
};

template<typename Kernel, typename... Args>
static __inline__ void ggml_cuda_kernel_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {
    kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );
    CUDA_CHECK(cudaGetLastError());
}

// ---- mainline ggml.h:262 -----------------------------------------------------
#ifndef GGML_UNUSED_VARS
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)
#endif

// ---- mainline common.cuh:383 / 792-833 (aligned register<->smem copies) -----
// PXA: NVIDIA-only build, so the widest coalesced transfer is 16 B (the HIP
// branch of mainline's ggml_cuda_get_max_cpy_bytes is dropped).
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
    return 16;
}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// ---- host-side arch predicates under mainline spellings ---------------------
static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}
static bool ampere_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}
static bool volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}
static bool amd_mfma_available(const int) { return false; } // PXA: CUDA-only build
static bool amd_wmma_available(const int) { return false; } // PXA: CUDA-only build

// ---- mainline common.cuh: warp_reduce_all -----------------------------------
template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

// ---- PDL (programmatic dependent launch) is Hopper+ and is not enabled in this
// fork's build; mainline compiles these to nothing without GGML_CUDA_USE_PDL.
static __device__ __forceinline__ void ggml_cuda_pdl_sync() {}
static __device__ __forceinline__ void ggml_cuda_pdl_lc()   {}

// ---- mainline common.cuh: compile-time unrolled index helper ----------------
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};
