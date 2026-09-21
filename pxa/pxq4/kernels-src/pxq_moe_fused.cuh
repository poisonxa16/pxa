// pxq_moe_fused.cuh -- the FUSED MoE decode block: the whole routed-expert MLP of one layer
// in two kernel launches instead of the ten-to-thirteen the sidecar spends today.
//
// WHY THIS FILE EXISTS. PXQ4MoEMethod.apply()'s capture-safe indexed path
// (sidecar/site-union/pxq4_vllm/moe.py) is correct and graph-legal, but it expresses the MoE
// block as two PXQ GEMVs surrounded by nine stock torch kernels:
//
//     out = zeros(M,H)                         <- allocated, zeroed, and NEVER USED by this branch
//     ids = topk_ids.reshape(-1).to(int32)
//     xg  = x.unsqueeze(1).expand(M,top_k,H).reshape(S,H).contiguous()   <- x materialised top_k times
//     moe_mmv(gu, xg, ids, w13)                <- the only real work
//     act = silu(gu[:, :I]) * gu[:, I:]        <- two elementwise passes over S*2I
//     moe_mmv(dn, act, ids, w2)                <- the only real work
//     wts = topk_weights.to(f32)
//     folded = (dn.view(M,top_k,H).to(f32) * wts).sum(dim=1)   <- an S*H fp16->fp32 expansion,
//     return folded.to(f16)                                       a broadcast multiply, a
//                                                                 reduction and a cast
//
// The 35B decodes 40 of those per token. The arithmetic says this is not a bandwidth problem:
// 8 routed experts of 256 is ~13 MB of expert bytes per token, about 0.1 ms of P100 HBM, while
// the measured step is 33.6 ms. It is a launch/host problem, so the fix is to stop launching.
//
// WHAT IS FUSED, AND WHAT IS DELIBERATELY NOT.
//   k_pxq_moe_gateup_glu   gate GEMV + up GEMV + SwiGLU, one launch, x staged ONCE per token
//   k_pxq_moe_down_fold    down GEMV + the router-weighted top_k fold, one launch, no atomics
// That is the entire routed block. It is NOT one launch, and the reason is structural rather
// than lazy: the down projection contracts over the intermediate axis, so a block owning 64
// rows of H needs the COMPLETE act[I] vector for its (token, slot). Fusing to a single launch
// would mean either recomputing the whole gate+up in every one of the H/64 down blocks (32x
// redundant at H=2048) or a grid-wide barrier via a cooperative launch, which is not reliably
// stream-capturable on this CUDA/driver set. Two launches with a real data dependency is the
// honest shape, and it is the shape the llama engine ships (ggml-cuda.cu:5472
// pxa_pxq4_moe_fast_tg launches exactly one gateup grid and one down grid per MoE layer).
//
// NUMERICS: NOTHING NEW IS INVENTED HERE. Every dot is the enclosing TU's existing per-slab
// dot (pxq23_dot32 / pxq4_dot32) at the existing PXQ_CANON_v1 two-level fixed-chunk fold with
// the existing nfix, which is a function of SHAPE ONLY. What changes is addressing and the
// epilogue. That is the same provenance argument pxa_expert_shard.cuh makes for the engine's
// sharded MoE: only WHERE the identical math runs moves, never WHAT it computes.
//
// THE EPILOGUE IS ROUNDING-COMPATIBLE WITH TORCH ON PURPOSE. To stay comparable with the path
// it replaces, the GLU reproduces torch's half-precision silu and multiply exactly:
//     g,u  = folds in fp32
//     gh   = __float2half_rn(g);  uh = __float2half_rn(u)            (moe_mmv writes fp16)
//     sh   = __float2half_rn( f(gh) / (1 + expf(-f(gh))) )           (aten silu: opmath float,
//                                                                     ::expf, result rounded
//                                                                     back to Half)
//     act  = __float2half_rn( f(sh) * f(uh) )                        (at::Half operator* is a
//                                                                     float multiply rounded)
// expf, not __expf: aten uses c10::cuda::compat::exp, which is the accurate ::expf. The fast
// intrinsic would be a silent numeric change wearing a bit-exactness claim.
//
// THE SLOT FOLD REPRODUCES TORCH'S OWN ACCUMULATION ORDER, which is what lets the fused block
// be BYTE-IDENTICAL to the path it replaces rather than merely close.
//
// The first version of this kernel folded the top_k slots in a simple ascending loop. That is a
// perfectly good canonical order, and it cost the byte gate: 7 of 20 greedy completions matched
// (19/20 on the first token), because a 1-ULP difference per layer compounds over 40 layers and
// 128 tokens. The reduce axis is only top_k long, though, so the space of plausible fp32 orders
// is small enough to ENUMERATE rather than despair over. Measured against
// (dn.float() * w).sum(dim=1) at H=2048, top_k=8, over M in 1/2/4/8:
//
//     4 accumulators, strided (acc[j & 3])   40/40 EXACT
//     ascending / descending / pairwise tree / 2 accumulators / 4 blocked   0/40
//
// So torch reduces this axis with four accumulators taken stride-wise and then folds them
// ascending, and the kernel below does exactly that. PXQ_MOE_FOLD_ACC is the accumulator count
// and is a BUILD-TIME constant, never a runtime flag, because changing it re-bases every output
// bit. Note the claim is shape-scoped and honest: it is verified at top_k = 8, the width this
// stack ships. At other top_k the fold is still fixed, deterministic and batch-invariant -- so
// the kernel is still bit-exact against its own reference and still safe -- it simply may not
// coincide with torch's choice there, and the unit test says which case it measured.
//
// Adding zero is exact in fp32, so a top_k below the accumulator count degrades correctly: the
// unused accumulators stay at 0.0f and contribute nothing.
//
// THE FOLD IS ATOMIC-FREE, and that is a hard requirement, not taste. An
// atomicAdd over the top_k slots would make the result depend on block scheduling, and this
// campaign gates on determinism (12/12 at np=1 AND np=2). So the slot axis leaves the grid and
// becomes a fixed loop inside one block, whose fp32 accumulation order is a function of top_k
// alone and of nothing else -- not of M, not of the block schedule, not of the device.
//
// CAPTURE SAFETY. No host reads, no allocation, no data-dependent launch geometry: ids and
// router weights stay on device and every grid dimension is a function of (M, top_k, shape).
// An id outside [0, E) contributes exactly zero and its output row is still WRITTEN, never
// left stale -- the same padding contract k_pxq23_moe_mmv already established for vLLM's -1
// slots.
//
// TU OWNERSHIP. This header defines templates only; it declares no device tables of its own.
// That is deliberate. pxq2_book_g / pxq3_book_g / pxq23_sub16_g are `static __device__` in
// pxq23_kernel.cuh, so a header that included it from a NEW .cu would get that TU's OWN
// silent copies, which pxq23_upload_book would never update -- a checkpoint with a custom book
// would then decode correctly through the old ops and wrongly through the new ones, with no
// error anywhere. So this file is #included FROM pxq23_kernel.cu (pxq2/pxq3, via pxq23_dot32)
// and FROM pxq4_kernel.cu (pxq4, via pxq4_dot32), and each TU supplies its own dot adapter and
// keeps its own tables. The same reasoning is why pxq23_kernel.cuh's pxq4_selftest_pol must
// NOT be used here: pxq23_upload_book explicitly refuses tier 252 (pxq23_kernel.cu:153), so
// that policy reads a book nobody uploads to.
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#include "pxq4_kernel_tables.h"   // PXQ4_BM, PXQ4_QK, PXQ4_MMV_KSEG, PXQ4_CANON_CMAX

#ifndef PXQ_MOE_EXTERN_SHARED
#define PXQ_MOE_EXTERN_SHARED extern __shared__ __align__(16)
#endif

// Canonical chunk count. Reproduced (not reused) so this header stands alone in either TU;
// the value is identical to pxq4_canon_nfix and pxq23_canon_nfix by construction, and the
// unit test asserts the fused output against a reference built on those, so a drift here
// fails a gate rather than silently re-rounding a model.
// Accumulators in the top_k fold. BUILD-TIME, never a runtime flag: changing it re-bases every
// output bit. 4 is torch's own choice for this axis (see the header note) and is what makes the
// fused block byte-identical to the path it replaces.
#ifndef PXQ_MOE_FOLD_ACC
#define PXQ_MOE_FOLD_ACC 4
#endif

static __host__ __device__ __forceinline__ int pxq_moe_canon_nfix(int kslabs, int cmax) {
    int lim = kslabs / PXQ4_MMV_KSEG;
    if (lim < 1)    lim = 1;
    if (lim > cmax) lim = cmax;
    int n = 1;
    while (n * 2 <= lim) n *= 2;
    return n;
}

static __host__ __forceinline__ int pxq_moe_max_chunk_floats(int kslabs) {
    const int nfix = pxq_moe_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    return ((kslabs + nfix - 1) / nfix) * PXQ4_QK;
}

// aten's silu on a Half tensor, reproduced exactly: promote to float, x/(1+exp(-x)) with the
// accurate expf, round the RESULT back to half (silu returns scalar_t), then the SwiGLU
// multiply is at::Half operator*, i.e. a float multiply rounded once more.
static __device__ __forceinline__ __half pxq_moe_swiglu_h(float g, float u) {
    const float gf  = __half2float(__float2half_rn(g));
    const float uf  = __half2float(__float2half_rn(u));
    const float sil = __fdiv_rn(gf, __fadd_rn(1.0f, expf(-gf)));
    return __float2half_rn(__fmul_rn(__half2float(__float2half_rn(sil)), uf));
}

// ---------------------------------------------------------------------------------------------
// LAUNCH 1 -- gate GEMV + up GEMV + SwiGLU.
//
// grid  = (Ip/64, S)   S = M*top_k, block = PXQ4_MMV_KSEG*64 = 256
// smem  = max_chunk_floats(kslabs) floats (the k_pxq23_mmv chunked staging, verbatim)
//
// w13 is ONE tensor, [E, 2*Ip, H], gate in panels [0, Ip/64) and up in [Ip/64, 2*Ip/64) --
// that is the layout PXQ4MoEMethod._weight_loader builds (shard_id w1 -> beg 0, w3 -> beg per).
// So one block walks BOTH halves for the same 64 output rows, which is what lets it stage x
// once for two GEMVs. Total weight traffic is unchanged; the block count halves and the
// activation staging halves.
//
// x is indexed by TOKEN (s / top_k), not by the flattened (token,slot) row. That single change
// is what deletes the `xg` expand+contiguous from the python side: every slot of a token reads
// the same activation straight out of x, so there is nothing to materialise.
// ---------------------------------------------------------------------------------------------
template <class POL, class DOT, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq_moe_gateup_glu(const uint8_t * __restrict__ slabs,    // [E, panels13, kslabs, SLAB]
                     const __half  * __restrict__ anchor,   // [E, panels13, 64]
                     const __half  * __restrict__ x,        // [M, K]   K == H
                     const int32_t * __restrict__ ids,      // [S]
                     __half        * __restrict__ act,      // [S, Ip]
                     const int Ip, const int K, const int E,
                     const int panels13, const int top_k) {
    const int p = blockIdx.x;                 // gate panel; up panel is p + half
    const int s = blockIdx.y;                 // flattened (token, slot) row
    const int t = s / top_k;                  // token -- x is per token, not per row

    PXQ_MOE_EXTERN_SHARED float pxq_moe_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[2 * PXQ4_MMV_KSEG * PXQ4_BM];   // gate half, then up half

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;
    const int half   = panels13 >> 1;         // == Ip/64

    const int  e     = ids[s];
    const bool valid = (e >= 0) && (e < E);
    const int  esafe = valid ? e : 0;

    const size_t pg = (size_t)esafe * panels13 + p;
    const size_t pu = (size_t)esafe * panels13 + half + p;
    const uint8_t * pan_g = slabs + pg * (size_t)kslabs * POL::SLAB;
    const uint8_t * pan_u = slabs + pu * (size_t)kslabs * POL::SLAB;
    const float     anch_g = __half2float(anchor[pg * PXQ4_BM + row]);
    const float     anch_u = __half2float(anchor[pu * PXQ4_BM + row]);
    const __half  * xt     = x + (size_t)t * K;

    const int nfix = pxq_moe_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float sg = 0.f, su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        // barrier 1 also covers the stage_tabs writes on the first iteration, and protects the
        // previous chunk's readers from this chunk's writers on every later iteration.
        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq_moe_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        if (valid) {
            float tg = 0.f, tu = 0.f;
            for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                const float * xk = pxq_moe_xs + (size_t)(kb - b0) * PXQ4_QK;
                tg += DOT::dot(pan_g + (size_t)kb * POL::SLAB, row, anch_g, xk, tab, sub);
                tu += DOT::dot(pan_u + (size_t)kb * POL::SLAB, row, anch_u, xk, tab, sub);
            }
            sg += tg;
            su += tu;
        }
    }

    red[kseg * PXQ4_BM + row]                            = sg;
    red[PXQ4_MMV_KSEG * PXQ4_BM + kseg * PXQ4_BM + row]  = su;
    __syncthreads();
    if (kseg == 0) {
        float g = 0.f, u = 0.f;
#pragma unroll
        for (int i = 0; i < PXQ4_MMV_KSEG; ++i) {
            g += red[i * PXQ4_BM + row];
            u += red[PXQ4_MMV_KSEG * PXQ4_BM + i * PXQ4_BM + row];
        }
        act[(size_t)s * Ip + p * PXQ4_BM + row] = pxq_moe_swiglu_h(g, u);
    }
}

// ---------------------------------------------------------------------------------------------
// LAUNCH 2, FORM A -- down GEMV with the router-weighted top_k fold folded in.
//
// grid  = (H/64, M), block = 256, smem = max_chunk_floats(kslabs) floats.
//
// The slot axis is GONE from the grid and lives inside the block as a fixed ascending loop.
// Total weight traffic is identical to today's w2 moe_mmv (M*top_k panel reads either way),
// only regrouped -- and the regrouping is what makes the fold free, deterministic, and
// atomic-free. Steps 8-12 of the python path (the fp32 expansion, the broadcast multiply, the
// sum over dim 1, the cast back) all disappear into the last four lines of this kernel, along
// with the dead `out = torch.zeros(M,H)` at the top of apply().
//
// The trade this form makes, stated because it is real and the window has to arbitrate it:
// grid.y loses the top_k factor, so at M=1 the launch is H/64 = 32 blocks instead of 256. On a
// 56-SM P100 that is well under one wave. FORM B below keeps the blocks and pays one extra
// launch; both produce bit-identical values, so the choice is purely a measurement.
// ---------------------------------------------------------------------------------------------
template <class POL, class DOT, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq_moe_down_fold(const uint8_t * __restrict__ slabs,     // [E, panels2, kslabs, SLAB]
                    const __half  * __restrict__ anchor,    // [E, panels2, 64]
                    const __half  * __restrict__ act,       // [S, K]   K == Ip
                    const int32_t * __restrict__ ids,       // [S]
                    const float   * __restrict__ wts,       // [S] fp32 router weights
                    __half        * __restrict__ out,       // [M, R]   R == H
                    const int R, const int K, const int E,
                    const int panels2, const int top_k) {
    const int p = blockIdx.x;                 // output panel of H
    const int t = blockIdx.y;                 // token

    PXQ_MOE_EXTERN_SHARED float pxq_moe_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;
    const int nfix   = pxq_moe_canon_nfix(kslabs, PXQ4_CANON_CMAX);

    // fp32 accumulators over the slots, in torch's own order; kseg 0 owns them.
    float acc[PXQ_MOE_FOLD_ACC];
#pragma unroll
    for (int i = 0; i < PXQ_MOE_FOLD_ACC; ++i) acc[i] = 0.f;

    for (int j = 0; j < top_k; ++j) {
        const int  s     = t * top_k + j;
        const int  e     = ids[s];
        const bool valid = (e >= 0) && (e < E);
        const int  esafe = valid ? e : 0;

        const size_t pe = (size_t)esafe * panels2 + p;
        const uint8_t * pan  = slabs + pe * (size_t)kslabs * POL::SLAB;
        const float     anch = __half2float(anchor[pe * PXQ4_BM + row]);
        const __half  * as   = act + (size_t)s * K;

        float sd = 0.f;
        for (int c = 0; c < nfix; ++c) {
            const int b0 = (kslabs * c) / nfix;
            const int b1 = (kslabs * (c + 1)) / nfix;
            const int n  = (b1 - b0) * PXQ4_QK;

            __syncthreads();
            for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
                pxq_moe_xs[idx] = __half2float(as[b0 * PXQ4_QK + idx]);
            }
            __syncthreads();

            if (valid) {
                float tt = 0.f;
                for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                    tt += DOT::dot(pan + (size_t)kb * POL::SLAB, row, anch,
                                   pxq_moe_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
                }
                sd += tt;
            }
        }

        red[kseg * PXQ4_BM + row] = sd;
        __syncthreads();
        if (kseg == 0) {
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < PXQ4_MMV_KSEG; ++i) d += red[i * PXQ4_BM + row];
            // Round to fp16 FIRST: that is the value the unfused moe_mmv would have written
            // into `dn`, and the fold that follows is the one the reference performs on it.
            //
            // __fmul_rn/__fadd_rn, NOT `acc += a * b`. nvcc contracts a*b+c into a single FMA
            // by default (-fmad=true), which keeps the product in extended precision and is a
            // DIFFERENT number from the reference's two separately-rounded fp32 operations
            // (`dn.float() * w`, then the accumulate). One fused multiply-add here would cost
            // the bit-exactness gate for a rounding nobody asked for.
            const int b = j % PXQ_MOE_FOLD_ACC;
            acc[b] = __fadd_rn(acc[b], __fmul_rn(__half2float(__float2half_rn(d)), wts[s]));
        }
        // the next slot's first chunk barrier separates its xs writes from this slot's xs
        // readers, and this slot's red readers from the next slot's red writers.
    }

    if (kseg == 0) {
        float u = acc[0];                     // then fold the accumulators ascending, as torch does
#pragma unroll
        for (int i = 1; i < PXQ_MOE_FOLD_ACC; ++i) u = __fadd_rn(u, acc[i]);
        out[(size_t)t * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}

// ---------------------------------------------------------------------------------------------
// LAUNCH 2, FORM B -- the same values with the slot axis kept in the grid.
//
// k_pxq_moe_down_part: grid = (H/64, S), identical to today's w2 moe_mmv geometry and block
// count, writing the per-slot fp16 partials exactly as `dn` holds them today.
// k_pxq_moe_slot_fold: grid = (ceil(M*R/256)), one thread per output element, folding the
// top_k slots in the SAME ascending fp32 order form A uses.
//
// So form B is three launches per layer instead of two and is bit-identical to form A by
// construction: same dot, same fold order, same two roundings. It exists because form A trades
// blocks for launches and only a measurement can say which side of that trade this rig is on.
// ---------------------------------------------------------------------------------------------
template <class POL, class DOT, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq_moe_down_part(const uint8_t * __restrict__ slabs,
                    const __half  * __restrict__ anchor,
                    const __half  * __restrict__ act,
                    const int32_t * __restrict__ ids,
                    __half        * __restrict__ dn,        // [S, R]
                    const int R, const int K, const int E, const int panels2) {
    const int p  = blockIdx.x;
    const int s  = blockIdx.y;

    PXQ_MOE_EXTERN_SHARED float pxq_moe_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const int  e     = ids[s];
    const bool valid = (e >= 0) && (e < E);
    const int  esafe = valid ? e : 0;

    const size_t pe = (size_t)esafe * panels2 + p;
    const uint8_t * pan  = slabs + pe * (size_t)kslabs * POL::SLAB;
    const float     anch = __half2float(anchor[pe * PXQ4_BM + row]);
    const __half  * as   = act + (size_t)s * K;

    const int nfix = pxq_moe_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float sd = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq_moe_xs[idx] = __half2float(as[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        if (valid) {
            float tt = 0.f;
            for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                tt += DOT::dot(pan + (size_t)kb * POL::SLAB, row, anch,
                               pxq_moe_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
            }
            sd += tt;
        }
    }

    red[kseg * PXQ4_BM + row] = sd;
    __syncthreads();
    if (kseg == 0) {
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < PXQ4_MMV_KSEG; ++i) d += red[i * PXQ4_BM + row];
        dn[(size_t)s * R + p * PXQ4_BM + row] = __float2half_rn(d);
    }
}

// One thread per (token, output element). Ascending over slots in fp32 -- the same order and
// the same roundings as form A's epilogue, so the two forms are bitwise equal.
static __global__ void __launch_bounds__(256)
k_pxq_moe_slot_fold(const __half * __restrict__ dn,         // [M*top_k, R]
                    const float  * __restrict__ wts,        // [M*top_k]
                    __half       * __restrict__ out,        // [M, R]
                    const int M, const int R, const int top_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * R) return;
    const int t = i / R;
    const int r = i - t * R;
    // Same accumulator count, same order, same no-FMA rule as form A's epilogue: form B must be
    // bitwise equal to form A, and both must equal torch's fold.
    float acc[PXQ_MOE_FOLD_ACC];
#pragma unroll
    for (int i = 0; i < PXQ_MOE_FOLD_ACC; ++i) acc[i] = 0.f;
    for (int j = 0; j < top_k; ++j) {
        const int s = t * top_k + j;
        const int b = j % PXQ_MOE_FOLD_ACC;
        acc[b] = __fadd_rn(acc[b], __fmul_rn(__half2float(dn[(size_t)s * R + r]), wts[s]));
    }
    float u = acc[0];
#pragma unroll
    for (int i = 1; i < PXQ_MOE_FOLD_ACC; ++i) u = __fadd_rn(u, acc[i]);
    out[(size_t)t * R + r] = __float2half_rn(u);
}
