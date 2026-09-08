// pxa_pascal_ops_launch.h -- CUDA-free declarations for the host TU.
// Kept free of <cuda_fp16.h> so pxa_pascal_ops_torch.cpp compiles with g++.
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace pxa_pascal {

// dtype tags used across the host boundary (no torch types in the kernel TU)
enum DType { DT_F16 = 0, DT_F32 = 1 };

// GemmaRMSNorm with residual: out fp16, res_out fp32, x fp16|fp32, res_in fp16|fp32.
void launch_gemma_add_rms_norm(void* out, void* res_out, const void* x,
                               const void* res_in, const float* w, float eps,
                               int rows, int H, int x_dtype, int r_dtype,
                               int64_t x_stride, int64_t ri_stride,
                               int64_t r_stride, int64_t o_stride,
                               cudaStream_t stream);

// GemmaRMSNorm without residual: out dtype follows x dtype.
void launch_gemma_rms_norm(void* out, const void* x, const float* w, float eps,
                           int rows, int H, int dtype,
                           int64_t x_stride, int64_t o_stride,
                           cudaStream_t stream);

// RMSNormGated (the GDN norm): out/x/z all one dtype, weight fp32, act 0=silu 1=sigmoid.
void launch_rms_norm_gated(void* out, const void* x, const void* z,
                           const float* w, float eps, int rows, int H,
                           int dtype, int act,
                           int64_t x_stride, int64_t z_stride, int64_t o_stride,
                           cudaStream_t stream);

// Plain RMSNorm (fp16 x, fp16 weight) -- the vllm.ir op family.
void launch_rms_norm_h16(void* out, const void* x, const void* w, float eps,
                         int rows, int H, int64_t x_stride, int64_t o_stride,
                         cudaStream_t stream);
void launch_fused_add_rms_norm_h16(void* out, void* res_out, const void* x,
                                   const void* res_in, const void* w, float eps,
                                   int rows, int H, int64_t x_stride,
                                   int64_t r_stride, int64_t o_stride,
                                   cudaStream_t stream);

// SwiGLU (kernel body from the fused-MoE work, pxa_nat_silu_and_mul.cuh).
void launch_silu_and_mul_f16(void* out, const void* x, int rows, int N,
                             int64_t x_stride, int64_t o_stride,
                             cudaStream_t stream);

int pascal_ops_version();

}  // namespace pxa_pascal
