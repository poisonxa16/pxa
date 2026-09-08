// pxq4_moe_kernel.cuh — expert-indexed PXQ4 mmv kernels for the FusedMoE path.
//
// WHY THESE EXIST. PXQ4MoEMethod.apply() (pxq4_moe.py) drove a per-expert Python loop off a
// host copy of topk_ids. A host sync is illegal under stream capture, so the MoE path was
// eager-only ("CUDA error: operation not permitted when stream is capturing"), and eager
// decode on this stack costs ~3.4x (docs/12: 3.90 vs 13.27 tok/s on the dense 27B).
//
// THE SHAPE OF THE FIX. vLLM routes every token to exactly `topk` experts, so a batch of M
// tokens is S = M*topk (token, expert-slot) rows — a STATIC shape. The only data-dependent
// quantity is WHICH expert serves each row, and that is an integer per row that the kernel
// can read from device memory itself. So: out[s] = x[s] @ W[ids[s]]^T with ids resident on
// device. No host read, no sort, no compaction; every control decision visible to the CPU is
// a function of shape only. That is the entire capture-safety argument.
//
// BIT-EXACTNESS. Per row s, the fold below is k_pxq4_mmv's fold verbatim on expert ids[s]'s
// [panels, kslabs, 1088] slice: same canonical chunks, same kb order, same pxq4_acc2 calls,
// same single __float2half_rn. A row's output is therefore bit-identical to
// mmv_out(out_row, x_row, slabs[e], anchor[e]) — which is what the parity gate asserts.
//
// EXPERT INDIRECTION ADDRESSING. The MoE parameters carry the expert as the slowest axis
// (pxq4_moe.py: slabs [E, P, Sk, 1088], anchor [E, P, 64]), so
//     pan_slabs = slabs  + ((size_t)e * panels + p) * kslabs * SLAB
//     anch      = anchor[((size_t)e * panels + p) * 64 + row]
// i.e. exactly the 2-D addressing with a per-row base offset. An id outside [0, E) (vLLM can
// emit -1 padding slots) contributes ZERO: the weight loop is skipped and the zero
// accumulator flows through the unchanged fold, so out[s] is written (never left stale).
//
// The fused-split variant replays k_pxq4_mmv_fused's arrival-barrier design unchanged; the
// counter is indexed [s * panels + p] (slot-major, mirroring the token axis of the 2-D
// kernel). All barrier caveats from k_pxq4_mmv_fused apply verbatim: one PXQ4 mmv in flight
// per device, ctr zero on entry and exit, a torn-down launch leaves the counter poisoned.

#pragma once

#include "pxq4_kernel.cuh"

// ---------------------------------------------------------------------------------------------
// mono: grid = (panels, S), block = 256, dynamic smem as k_pxq4_mmv.
// ---------------------------------------------------------------------------------------------
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4_moe_mmv(const uint8_t * __restrict__ slabs,      // [E, panels, kslabs, 1088]
               const __half  * __restrict__ anchor,     // [E, panels, 64]
               const __half  * __restrict__ x,          // [S, K]
               const int32_t * __restrict__ ids,        // [S]
               __half        * __restrict__ out,        // [S, R]
               const int R, const int K, const int E, const int panels) {
    const int p  = blockIdx.x;                          // panel
    const int iy = blockIdx.y;                          // (token, slot) row

    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const int  e     = ids[iy];
    const bool valid = (e >= 0) && (e < E);
    const int  esafe = valid ? e : 0;

    const uint8_t * pan_slabs = slabs + ((size_t)esafe * panels + p) * (size_t)kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[((size_t)esafe * panels + p) * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int nfix = pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq4_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        if (valid) {
            float t = 0.f;
            for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                                      pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
            }
            su += t;
        }
    }

    red[kseg * PXQ4_BM + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}

// ---------------------------------------------------------------------------------------------
// fused split: grid = (nfix, panels, S), block = 256; ctr = [S * panels] unsigned, zero on
// entry and exit. Same barrier as k_pxq4_mmv_fused (sm_60 fallback included via
// pxq4_arrive_release / pxq4_fence_acq_rel).
// ---------------------------------------------------------------------------------------------
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4_moe_mmv_fused(const uint8_t * __restrict__ slabs,   // [E, panels, kslabs, 1088]
                     const __half  * __restrict__ anchor,  // [E, panels, 64]
                     const __half  * __restrict__ x,       // [S, K]
                     const int32_t * __restrict__ ids,     // [S]
                     float         * __restrict__ part,    // [S, panels, nfix, KSEG*64]
                     unsigned      * __restrict__ ctr,     // [S, panels]
                     __half        * __restrict__ out,     // [S, R]
                     const int R, const int K, const int E,
                     const int nfix, const int panels) {
    const int c  = blockIdx.x;                          // canonical chunk (fastest-varying)
    const int p  = blockIdx.y;                          // panel
    const int iy = blockIdx.z;                          // (token, slot) row

    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];
    __shared__ int   last;

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const int  e     = ids[iy];
    const bool valid = (e >= 0) && (e < E);
    const int  esafe = valid ? e : 0;

    const uint8_t * pan_slabs = slabs + ((size_t)esafe * panels + p) * (size_t)kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[((size_t)esafe * panels + p) * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    __syncthreads();                                    // covers the stage_tabs writes
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        pxq4_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
    }
    __syncthreads();

    float t = 0.f;
    if (valid) {
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                                  pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
        }
    }

    const size_t  tile  = (size_t)(PXQ4_MMV_KSEG * PXQ4_BM);
    float * const pbase = part + (((size_t)iy * panels + p) * nfix) * tile;
    pbase[(size_t)c * tile + threadIdx.x] = t;

    // ---- arrival barrier over this (panel, row)'s nfix blocks ---------------------------
    // The __syncthreads() below is LOAD-BEARING (see k_pxq4_mmv_fused). Do not remove it.
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = pxq4_arrive_release(&ctr[(size_t)iy * panels + p]);
        last = (old == (unsigned)(nfix - 1));
        if (last) {
            ctr[(size_t)iy * panels + p] = 0u;          // rearm for the next launch
            pxq4_fence_acq_rel();                       // acquire the other blocks' part[]
        }
    }
    __syncthreads();                                    // propagates the acquire to the block
    if (!last) return;

    // ---- k_pxq4_mmv_reduce's fold, verbatim ---------------------------------------------
    float su = 0.f;
    for (int cc = 0; cc < nfix; ++cc) {                 // chunk fold, ascending c
        su += pxq4_ld_part(&pbase[(size_t)cc * tile + threadIdx.x]);
    }
    red[threadIdx.x] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;                                  // kseg fold, ascending s
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}
