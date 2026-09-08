// pxa_pascal_ops.cuh -- hand-fused Pascal/Volta kernels for the small ops modern vLLM
// ships as DECOMPOSITIONS.
//
// WHY THIS FILE EXISTS. vLLM no longer ships fused CUDA kernels for its norms and
// activations; it ships torch expression trees and expects Inductor to fuse them back.
// On sm_60 there is no Inductor -- TORCHDYNAMO_DISABLE=1 is load-bearing, and this
// stack's recipe also sets VLLM_USE_BREAKABLE_CUDAGRAPH=1, which by itself disables the
// compile pipeline -- so every norm and every activation runs as its eager
// decomposition. The whole layernorm / activation / positional-encoding family that
// would otherwise cover this lives in the _C_stable_libtorch extension, which needs
// torch >= 2.8 stable-ABI headers and is skipped outright on this build
// (CMakeLists.txt VLLM_SKIP_C_STABLE); there is no such library in the serving image and
// torch.ops._C holds ten unrelated ops. Measured on the 35B MoE at TP=2: 1450 norm
// kernels and 3.35 ms of a 33.5 ms decode token, plus 160 SwiGLU kernels and 0.81 ms --
// as much as every routed expert in the model.
//
// ARITHMETIC IS THE WHOLE POINT OF THIS FILE. Each kernel reproduces the ORDER of the
// torch chain it replaces, not merely its algebra, because the acceptance gate is greedy
// byte-identity against the path being replaced and a more accurate answer fails that
// gate exactly as a less accurate one does. Two rules are applied everywhere:
//
//   1. NO CONTRACTION. nvcc contracts `acc += a * b` into a single FMA under the default
//      -fmad=true, which keeps the product at extended precision and is a DIFFERENT
//      number from two separately rounded fp32 operations. Every place the reference
//      rounds twice, this file writes __fadd_rn(acc, __fmul_rn(a, b)).
//   2. ROUND WHERE ATEN ROUNDS. aten evaluates half elementwise ops in float and rounds
//      the RESULT back to half at every step, so a chain of three torch ops on half
//      tensors is three roundings and not one fp32 expression.
//
// WHAT CANNOT BE PROMISED, stated here rather than discovered in a window: the variance
// is a REDUCTION, and torch's accumulation order for x.pow(2).mean(-1) is chosen per
// shape by TensorIterator. A probe over 366 candidate layouts (bench/reduce_probe.py)
// found that every real shape has some layout reproducing torch bit-for-bit, that no
// layout reproduces two shapes, and that 8x2048 is reproduced by none of them. So
// bit-identity with the native path is not attainable at any block geometry. The fold
// below was chosen as the closest of the deterministic options: pooled over eight shapes
// it lands within 2 fp32 ULP of torch's variance and changes ~20 fp16 output elements
// per million, against ~50-60 ppm for a fixed-chunk canonical fold. The fold order is a
// function of H alone -- not of block scheduling, not of grid size, no atomics anywhere
// -- so run-to-run and np1-vs-np2 determinism hold by construction.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace pxa_pascal {

// ---------------------------------------------------------------------------------
// element access: one template so a kernel body can be written once for fp16 and fp32
// ---------------------------------------------------------------------------------
template <typename T> struct Elem;

template <> struct Elem<__half> {
    __device__ __forceinline__ static float load(const __half* p) { return __half2float(*p); }
    __device__ __forceinline__ static void  store(__half* p, float v) { *p = __float2half_rn(v); }
};

template <> struct Elem<float> {
    __device__ __forceinline__ static float load(const float* p) { return *p; }
    __device__ __forceinline__ static void  store(float* p, float v) { *p = v; }
};

// ---------------------------------------------------------------------------------
// THE FOLD. One block per row. Thread t owns the VEC-wide groups g = t, t+bw, t+2bw ...
// in increasing g, with VEC independent lane accumulators; the lanes are merged in index
// order; the block is then combined by the offset-doubling tree that torch's own
// reduce_kernel uses for its warp combine (v += shfl_down(v, off), off = 1, 2, 4 ...).
// bw is always a power of two by construction (it is H/VEC, and every H this serves --
// 128, 256, 2048 -- divides that way), so the tree is exact with no ragged tail.
//
// Shared memory rather than __shfl_down_sync for the combine: bw exceeds a warp at
// H=2048, and one code path that is correct at every width is worth more than a
// warp-specialised one that has to be re-argued at each new shape.
// ---------------------------------------------------------------------------------
__device__ __forceinline__ float block_fold_pairwise(float v, int bw, float* sm, int tid)
{
    sm[tid] = v;
    __syncthreads();
    for (int off = 1; off < bw; off <<= 1) {
        float add = (tid + off < bw) ? sm[tid + off] : 0.0f;
        __syncthreads();                       // every read finished before any write
        sm[tid] = __fadd_rn(sm[tid], add);
        __syncthreads();
    }
    return sm[0];
}

// ---------------------------------------------------------------------------------
// GemmaRMSNorm WITH residual -- 80 calls per decode token on the 35B, 11 kernels each.
//
// The reference is GemmaRMSNorm.forward_native (vllm/model_executor/layers/layernorm.py),
// which on this platform is what forward_cuda falls through to:
//     weight = self.weight.data.float() + 1.0        # rebuilt EVERY call
//     x = x.float() + residual.float()               # residual is carried in FP32
//     residual = x                                   # fp32 out
//     out = ir.ops.rms_norm(x, weight, eps)          # fp32 in, fp32 out
//     return out.to(orig_dtype), residual            # fp16 out
// and ir.ops.rms_norm with an fp32 x and an fp32 weight is
//     var = x.pow(2).mean(-1); x = x * rsqrt(var + eps); x = x * weight
// Note the 1.0 + w is folded into this kernel, which deletes the two extra launches the
// python does per call ON TOP of the eleven the chain itself costs.
// ---------------------------------------------------------------------------------
template <typename XT, typename RT, int VEC>
__global__ void k_gemma_add_rms_norm(
    __half* __restrict__ out,            // [R, H] fp16
    float*  __restrict__ res_out,        // [R, H] fp32   (may alias res_in when RT=float)
    const XT* __restrict__ x,            // [R, H] fp16 or fp32
    const RT* __restrict__ res_in,       // [R, H] fp32, or fp16 on the first call
    const float* __restrict__ w,         // [H] fp32, RAW gemma weight; +1 applied here
    float eps, int H, int bw,
    int64_t x_stride, int64_t ri_stride, int64_t r_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const XT*    xr  = x       + (int64_t)row * x_stride;
    const RT*    rr  = res_in  + (int64_t)row * ri_stride;
    float*       ro  = res_out + (int64_t)row * r_stride;
    __half*      orow = out    + (int64_t)row * o_stride;

    // pass 1: the fp32 residual add, the fp32 residual write, and the fold
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    const int ngroups = H / VEC;
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float s = __fadd_rn(Elem<XT>::load(&xr[base + j]),
                                      Elem<RT>::load(&rr[base + j]));
            ro[base + j] = s;
            acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));   // never one FMA
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    const float total = block_fold_pairwise(part, bw, sm, tid);

    // torch's mean is sum / N; for the power-of-two H this serves, sum * (1/N) is the
    // same bits, but the division is what the reference literally does.
    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    // pass 2: scale, weight, round to fp16. Two separately rounded fp32 multiplies,
    // exactly as `x * rsqrt(...)` followed by `x * weight` rounds twice.
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float t1 = __fmul_rn(ro[base + j], scale);
            const float wf = __fadd_rn(1.0f, w[base + j]);
            orow[base + j] = __float2half_rn(__fmul_rn(t1, wf));
        }
    }
}

// ---------------------------------------------------------------------------------
// GemmaRMSNorm WITHOUT residual -- the final model norm (h=2048) and q_norm/k_norm
// (h=256 on the 10 full-attention layers), 21 calls per token, 10 kernels each.
// Identical tail; the input may be fp16 (q/k straight from the projection) or fp32.
// The output dtype is the input's original dtype, which is what `.to(orig_dtype)` means.
// ---------------------------------------------------------------------------------
template <typename XT, typename OT, int VEC>
__global__ void k_gemma_rms_norm(
    OT* __restrict__ out,
    const XT* __restrict__ x,
    const float* __restrict__ w,
    float eps, int H, int bw,
    int64_t x_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const XT* xr = x   + (int64_t)row * x_stride;
    OT*    orow  = out + (int64_t)row * o_stride;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    const int ngroups = H / VEC;
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float s = Elem<XT>::load(&xr[base + j]);
            acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    const float total = block_fold_pairwise(part, bw, sm, tid);
    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float t1 = __fmul_rn(Elem<XT>::load(&xr[base + j]), scale);
            const float wf = __fadd_rn(1.0f, w[base + j]);
            Elem<OT>::store(&orow[base + j], __fmul_rn(t1, wf));
        }
    }
}

// ---------------------------------------------------------------------------------
// RMSNormGated -- the 30 GDN layers, head_v_dim 128, 12 kernels each.
//
// Reference is RMSNormGated.forward_static with group_size=None and
// norm_before_gate=True (layernorm.py):
//     x = x.float(); weight = weight.float(); z = z.float()
//     var = x.pow(2).mean(-1); x_normed = x * rsqrt(var + eps)
//     out = x_normed * weight
//     out = out * act(z)                       # act is F.silu or sigmoid
//     return out.to(orig_dtype)
// Note this is a PLAIN weight, not Gemma's 1+w. And aten's silu is x / (1 + exp(-x)) --
// a division, not x * sigmoid(x); those are different fp32 numbers, and ::expf is the
// accurate exponential, never the __expf intrinsic.
//
// The fork ships an sm_70 one-pass route for this op but gates it to a 12x128 shape
// (qwen_gdn_linear_attn.py), so it never fires for the 32x128 geometry this model has on
// a Pascal card, which is why these 360 kernels a token are still on the table.
// ---------------------------------------------------------------------------------
enum GateAct { GATE_SILU = 0, GATE_SIGMOID = 1 };

__device__ __forceinline__ float gate_apply(float z, int act)
{
    if (act == GATE_SIGMOID) {
        // aten sigmoid on float: 1 / (1 + exp(-x))
        return __fdiv_rn(1.0f, __fadd_rn(1.0f, ::expf(-z)));
    }
    // aten silu on float: x / (1 + exp(-x))
    return __fdiv_rn(z, __fadd_rn(1.0f, ::expf(-z)));
}

template <typename XT, int VEC>
__global__ void k_rms_norm_gated(
    XT* __restrict__ out,
    const XT* __restrict__ x,
    const XT* __restrict__ z,
    const float* __restrict__ w,          // fp32 view of the weight
    float eps, int H, int bw, int act,
    int64_t x_stride, int64_t z_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const XT* xr = x   + (int64_t)row * x_stride;
    const XT* zr = z   + (int64_t)row * z_stride;
    XT*    orow  = out + (int64_t)row * o_stride;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    const int ngroups = H / VEC;
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float s = Elem<XT>::load(&xr[base + j]);
            acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    const float total = block_fold_pairwise(part, bw, sm, tid);
    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float t1 = __fmul_rn(Elem<XT>::load(&xr[base + j]), scale);
            const float t2 = __fmul_rn(t1, w[base + j]);
            const float gz = gate_apply(Elem<XT>::load(&zr[base + j]), act);
            Elem<XT>::store(&orow[base + j], __fmul_rn(t2, gz));
        }
    }
}

// ---------------------------------------------------------------------------------
// PLAIN RMSNorm -- not used by the 35B (its norms are all Gemma), but it is what
// vllm.ir.ops.rms_norm and fused_add_rms_norm mean for every model that uses the
// standard RMSNorm class, which is the dense 27B on the V100 half of the window.
//
// The epilogue is genuinely different from Gemma's and must not share a code path:
//     x = x * rsqrt(var + eps)         # fp32
//     x = x.to(weight.dtype) * weight  # rounds to the WEIGHT dtype, then multiplies
//     return x.to(orig_dtype)
// With an fp16 weight that is TWO roundings where Gemma has one, so `out` here is
// round(float(round_half(x*scale)) * float(w)) and not round(x * scale * w).
// ---------------------------------------------------------------------------------
template <int VEC>
__global__ void k_rms_norm_h16(          // fp16 x, fp16 weight: the double-round form
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    float eps, int H, int bw,
    int64_t x_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* xr = x   + (int64_t)row * x_stride;
    __half*    orow  = out + (int64_t)row * o_stride;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    const int ngroups = H / VEC;
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float s = __half2float(xr[base + j]);
            acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    const float total = block_fold_pairwise(part, bw, sm, tid);
    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float t1 = __fmul_rn(__half2float(xr[base + j]), scale);
            const __half h1 = __float2half_rn(t1);          // .to(weight.dtype)
            orow[base + j] = __float2half_rn(
                __fmul_rn(__half2float(h1), __half2float(w[base + j])));
        }
    }
}

// fused_add_rms_norm: the same, with the residual added in fp32 first and written back
// in the ORIGINAL dtype (fp16), which is where it differs from Gemma's fp32 residual.
template <int VEC>
__global__ void k_fused_add_rms_norm_h16(
    __half* __restrict__ out,
    __half* __restrict__ res_out,
    const __half* __restrict__ x,
    const __half* __restrict__ res_in,
    const __half* __restrict__ w,
    float eps, int H, int bw,
    int64_t x_stride, int64_t r_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* xr = x       + (int64_t)row * x_stride;
    const __half* rr = res_in  + (int64_t)row * r_stride;
    __half*       ro = res_out + (int64_t)row * r_stride;
    __half*     orow = out     + (int64_t)row * o_stride;

    // The variance is taken over the UNROUNDED fp32 sum, while the residual that is
    // written out is that sum rounded to fp16. Rounding first and folding the rounded
    // value -- which is what the upstream vLLM kernel does -- is a different number and
    // fails the gate. Hence sq is kept in registers rather than re-read from res_out.
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    const int ngroups = H / VEC;
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float s = __fadd_rn(__half2float(xr[base + j]),
                                      __half2float(rr[base + j]));
            acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    const float total = block_fold_pairwise(part, bw, sm, tid);
    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    // Pass 2 recomputes the UNROUNDED sum from the two inputs -- neither of which pass 1
    // wrote -- and only then rounds it into res_out. Writing res_out in pass 1 and
    // reading it back here would fold the fp16-rounded residual into the output, which
    // is what the upstream vLLM kernel does and is a different number; and res_out is
    // allowed to alias res_in (the in-place overload does exactly that), so a pass-1
    // write would also destroy pass 2's only source. Reading and writing the same
    // element within one statement is safe under that aliasing.
    for (int g = tid; g < ngroups; g += bw) {
        const int base = g * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float s  = __fadd_rn(__half2float(xr[base + j]),
                                       __half2float(rr[base + j]));
            ro[base + j]   = __float2half_rn(s);
            const float t1 = __fmul_rn(s, scale);
            const __half h1 = __float2half_rn(t1);
            orow[base + j] = __float2half_rn(
                __fmul_rn(__half2float(h1), __half2float(w[base + j])));
        }
    }
}


// =================================================================================
// THE EXACT FOLD: ATen's own reduction geometry, replicated rather than approximated.
//
// The kernels above use a fold chosen for accuracy and determinism. It lands within 2 fp32
// ULP of torch and changes ~20 fp16 output elements per million, which is enough to fail a
// greedy byte gate over 40 layers -- measured, 11/20 on the P100 pair.
//
// That was never necessary. torch's accumulation order is not arbitrary, it is COMPUTED by
// at::native::setReduceConfig, and reading that function gives the exact layout for any
// shape. An earlier search over 366 candidate layouts missed it, because every candidate
// used a single tree shape and the real one is a HYBRID:
//
//     ReduceConfig::block_x_reduce
//       if (dim_x > warpSize) {
//         for (int offset = dim_x/2; offset >= warpSize; offset >>= 1)  // shared memory
//         dim_x = warpSize;
//       }
//       for (int offset = 1; offset < dim_x; offset <<= 1)              // warp shuffle
//
// halving down to the warp, then offset-doubling inside it. With that, plus the four
// independent per-thread accumulators merged in index order, the fold reproduces torch
// BIT FOR BIT: 462 rows across 15 shapes, zero ULP (bench/aten_reduce_verify.py).
//
// The geometry is a 2-D block: threadIdx.y selects the ROW (bh rows per block), threadIdx.x
// walks that row in vec-wide strides of bw. For every shape this model serves the y-split
// never engages, so ctas_per_output stays 1 and the multiprocessor count never enters the
// answer -- the same fold is correct on a GP102, a P100 and a V100.
//
// Outside the modelled family the launcher falls back to the kernels above, because a fold
// that is bit-exact where it applies and merely deterministic elsewhere is strictly better
// than one that is neither.
// =================================================================================

// ReduceConfig::block_x_reduce, verbatim in structure. `sm` is bw*bh floats.
__device__ __forceinline__ float aten_block_x_reduce(float v, int bw, float* sm,
                                                     int tx, int ty)
{
    int dim_x = bw;
    const int base = tx + ty * bw;
    if (dim_x > warpSize) {
        sm[base] = v;
        for (int off = dim_x / 2; off >= warpSize; off >>= 1) {
            __syncthreads();
            if (tx < off && tx + off < bw) {
                v = __fadd_rn(v, sm[base + off]);
                sm[base] = v;
            }
        }
        dim_x = warpSize;
    }
    __syncthreads();
    // The shuffle half is unconditional in ATen: every lane combines, and only lane 0's
    // value is ever read. Reproducing the condition rather than the result would be a
    // different sum for the lanes that do not matter and the same one for the lane that
    // does -- but it is the shape of the reference and cheap, so it is kept.
    for (int off = 1; off < dim_x; off <<= 1) {
        const float other = __shfl_down_sync(0xffffffffu, v, off, warpSize);
        v = __fadd_rn(v, other);
    }
    // BROADCAST. After a shuffle-down tree only LANE 0 holds the row's sum -- in ATen that
    // is sufficient, because should_store() lets only threadIdx.x == 0 write the output.
    // Here every thread of the row needs the total, since each one scales its own slice of
    // the row in the second pass. Returning `v` directly gives threads 1..bw-1 a partial
    // sum, which is a wrong variance for every element they touch: the first version of
    // this kernel did exactly that and disagreed with the reference on 99% of elements.
    __syncthreads();
    if (tx == 0) sm[ty * bw] = v;
    __syncthreads();
    return sm[ty * bw];
}

// GemmaRMSNorm with residual, ATen fold. grid.x = ceil(rows / bh), block = (bw, bh).
template <typename XT, typename RT, int VEC>
__global__ void k_gemma_add_rms_norm_aten(
    __half* __restrict__ out, float* __restrict__ res_out,
    const XT* __restrict__ x, const RT* __restrict__ res_in,
    const float* __restrict__ w, float eps, int H, int rows, int bw,
    int64_t x_stride, int64_t ri_stride, int64_t r_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.x * blockDim.y + ty;
    const bool live = row < rows;

    const XT*    xr = x       + (int64_t)(live ? row : 0) * x_stride;
    const RT*    rr = res_in  + (int64_t)(live ? row : 0) * ri_stride;
    float*       ro = res_out + (int64_t)(live ? row : 0) * r_stride;
    __half*    orow = out     + (int64_t)(live ? row : 0) * o_stride;

    const int nvec = H / VEC;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    if (live) {
        for (int idx = tx; idx < nvec; idx += bw) {
            const int base = idx * VEC;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float s = __fadd_rn(Elem<XT>::load(&xr[base + j]),
                                          Elem<RT>::load(&rr[base + j]));
                ro[base + j] = s;
                acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));
            }
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    // Every thread of the block must reach the barriers inside the x-reduce, live or not.
    const float total = aten_block_x_reduce(part, bw, sm, tx, ty);
    if (!live) return;

    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    for (int idx = tx; idx < nvec; idx += bw) {
        const int base = idx * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float t1 = __fmul_rn(ro[base + j], scale);
            const float wf = __fadd_rn(1.0f, w[base + j]);
            orow[base + j] = __float2half_rn(__fmul_rn(t1, wf));
        }
    }
}

// GemmaRMSNorm without residual, ATen fold.
template <typename XT, typename OT, int VEC>
__global__ void k_gemma_rms_norm_aten(
    OT* __restrict__ out, const XT* __restrict__ x, const float* __restrict__ w,
    float eps, int H, int rows, int bw, int64_t x_stride, int64_t o_stride)
{
    extern __shared__ float sm[];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.x * blockDim.y + ty;
    const bool live = row < rows;

    const XT* xr  = x   + (int64_t)(live ? row : 0) * x_stride;
    OT*     orow  = out + (int64_t)(live ? row : 0) * o_stride;

    const int nvec = H / VEC;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;

    if (live) {
        for (int idx = tx; idx < nvec; idx += bw) {
            const int base = idx * VEC;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float s = Elem<XT>::load(&xr[base + j]);
                acc[j] = __fadd_rn(acc[j], __fmul_rn(s, s));
            }
        }
    }
    float part = acc[0];
#pragma unroll
    for (int j = 1; j < VEC; ++j) part = __fadd_rn(part, acc[j]);

    const float total = aten_block_x_reduce(part, bw, sm, tx, ty);
    if (!live) return;

    const float var   = __fdiv_rn(total, (float)H);
    const float scale = rsqrtf(__fadd_rn(var, eps));

    for (int idx = tx; idx < nvec; idx += bw) {
        const int base = idx * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float t1 = __fmul_rn(Elem<XT>::load(&xr[base + j]), scale);
            const float wf = __fadd_rn(1.0f, w[base + j]);
            Elem<OT>::store(&orow[base + j], __fmul_rn(t1, wf));
        }
    }
}

}  // namespace pxa_pascal
