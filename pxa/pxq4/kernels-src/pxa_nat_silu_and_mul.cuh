// pxa_nat_silu_and_mul.cuh -- fp16 SwiGLU for Pascal and Volta, BIT-IDENTICAL to the vLLM
// fork's SiluAndMul.forward_native.
//
// FOR LANE pascal-ops, to the signature they posted. Header only: no launcher, no .cu, no
// build script, no torch, no device tables. One owner of the build and the ABI (theirs), one
// owner of the arithmetic (this file).
//
// WHY THE FORK NEEDS IT: activation.py:144-147 resolves SiluAndMul to torch.ops._C.silu_and_mul
// and says in its own comment that this fork cannot build that against torch 2.7, so every
// SwiGLU in every model on this stack falls back to forward_native -- two kernels and a full
// intermediate round trip, per FFN, per layer, per token.
//
// ============================================================================================
// THE ARITHMETIC, which is the whole point of the file. forward_native is
//
//     d = x.shape[-1] // 2 ;  F.silu(x[..., :d]) * x[..., d:]
//
// and on HALF tensors aten does NOT evaluate that as one fp32 expression. Every torch op
// promotes to float, computes, and rounds the RESULT back to half. So this is TWO roundings:
//
//     s   = __float2half_rn( silu_f32( __half2float(gate) ) )
//     out = __float2half_rn( __half2float(s) * __half2float(up) )
//
// A single-expression fp32 version is MORE ACCURATE and NOT BIT-COMPARABLE, which means it can
// never pass a byte-identity gate against the path it replaces and therefore can never be
// turned on by default. Agreement is the goal here, not accuracy.
//
// Two details inside silu that are right by construction and wrong when retyped from memory:
//
//   * IT IS A DIVISION.  aten's silu is  x / (1 + exp(-x)),  NOT  x * sigmoid(x)  and NOT
//     x * __frcp_rn(1 + exp(-x)). Those are different fp32 numbers. (Confirmed against the
//     aten kernel, and this is the form already gated bit-exact inside the fused MoE gateup.)
//   * IT IS ::expf, NOT __expf.  aten uses c10::cuda::compat::exp, which is the accurate
//     ::expf. The fast intrinsic is a silent numeric change wearing a bit-exactness claim.
//
// And the products/sums are written with __fmul_rn / __fadd_rn / __fdiv_rn rather than plain
// operators, because nvcc contracts a*b+c into a single FMA by default (-fmad=true) and an FMA
// keeps the product in extended precision -- a different number from two separately rounded
// fp32 operations, which is what torch performs. There is no a*b+c in THIS kernel, so the
// explicit intrinsics here are belt-and-braces; they are load-bearing in anything that
// accumulates, which is why they are called out.
//
// PROVENANCE: these three lines are not new. They are the epilogue of the fused MoE gateup
// kernel in pxq_moe_fused.cuh, which passed its bit-exactness gate against the real unfused
// path on the first run -- 32,768 elements at a time, PXQ4 and PXQ2 tiers, two TP shard
// widths, every gate an equality on raw fp16 bits.
//
// A NOTE ON SCOPE, since pascal-ops raised it: this kernel is pure elementwise, so bit-exact
// is genuinely achievable. That is NOT true of anything containing a torch reduction -- their
// probe found torch's fp32 order for x.pow(2).mean(-1) is shape-dependent and unreproducible,
// and my own fused MoE sees the same thing from the other side: my ascending top_k fold differs
// from torch's sum(dim=1) on 0 to 5 elements per 2,048-16,384, always by exactly 1 ULP. For
// reductions the only honest shape is bit-exact against an explicit reference plus a byte gate
// against torch. For this file, bit-exact against torch itself is the gate.
// ============================================================================================
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace pxa_nat {

// VEC contiguous halves as one load. VEC is chosen at the call site by divisibility, so the
// kernel never branches on it. 1/2/4/8 halves = 2/4/8/16 bytes.
template <int VEC> struct hvec;
template <> struct hvec<1> { using type = __half; };
template <> struct hvec<2> { using type = uint32_t; };
template <> struct hvec<4> { using type = uint2; };
template <> struct hvec<8> { using type = uint4; };

// aten silu on one half, then the SwiGLU multiply. Two roundings, a division, and ::expf.
__device__ __forceinline__ __half silu_and_mul_one(__half gate, __half up) {
    const float g = __half2float(gate);
    const __half s = __float2half_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))));
    return __float2half_rn(__fmul_rn(__half2float(s), __half2float(up)));
}

// ---------------------------------------------------------------------------------------------
// grid.x = rows (M never enters the kernel, so the same instantiation serves a dense FFN at
// [M, 2N] and the routed-expert call at [M*top_k, 2*I_p]); the block walks N in a stride loop.
// Row strides are in ELEMENTS: pass 2N and N for the contiguous case and they fold away.
//
// CONTRACT: VEC must divide N, and the row bases must be VEC*2-byte aligned -- both are the
// caller's to guarantee, since the caller is what picks VEC. The `c < N` guard below therefore
// never fires on a well-formed call; it is kept because this kernel is memory-bound, so one
// predicate per element is free, and because the alternative to a free guard is an
// out-of-bounds write on the first caller who gets the divisibility wrong.
// ---------------------------------------------------------------------------------------------
template <int VEC>
__global__ void k_silu_and_mul_f16(__half * __restrict__ out,          // [M, N]
                                   const __half * __restrict__ x,      // [M, 2N], gate then up
                                   int N,
                                   int64_t x_row_stride,
                                   int64_t out_row_stride) {
    using V = typename hvec<VEC>::type;

    const __half * __restrict__ gate = x + (int64_t)blockIdx.x * x_row_stride;
    const __half * __restrict__ up   = gate + N;
    __half * __restrict__ o          = out + (int64_t)blockIdx.x * out_row_stride;

    const int step = (int)blockDim.x * VEC;
    for (int base = (int)threadIdx.x * VEC; base < N; base += step) {
        __half gv[VEC], uv[VEC], ov[VEC];

        if (VEC > 1 && base + VEC <= N) {
            *(V *)gv = *(const V *)(gate + base);
            *(V *)uv = *(const V *)(up + base);
        } else {
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                const int c = base + v;
                gv[v] = c < N ? gate[c] : __ushort_as_half((unsigned short)0);
                uv[v] = c < N ? up[c]   : __ushort_as_half((unsigned short)0);
            }
        }

#pragma unroll
        for (int v = 0; v < VEC; ++v) ov[v] = silu_and_mul_one(gv[v], uv[v]);

        if (VEC > 1 && base + VEC <= N) {
            *(V *)(o + base) = *(const V *)ov;
        } else {
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                const int c = base + v;
                if (c < N) o[c] = ov[v];
            }
        }
    }
}

}  // namespace pxa_nat
