// pxq4_gemm2d_kernel.cuh — fused 4-bit prefill GEMM for sm_60, vendored from the engine's
// k_pxq6_gemm_grouped (pxq6.cuh:2519, POL=pxq6_pol_p6, RAG+PIPE arms) with the E==1 idiom
// and the sidecar's two-tensor addressing (slabs [P,S,1088] + anchor [P,64]).
//
// WHY. On P100 (full-rate fp16, no tensor cores) the engine measured this tile +35% on
// dense prefill AGAINST THE SAME dequant+cuBLAS incumbent the sidecar runs
// (ggml-cuda.cu:4414 "+35% P100 dense prefill measured 2026-07-28"). On sm_70 it is a
// measured -18.6% LOSS (HMMA cuBLAS wins) — never route it there.
//
// SHAPE. 64-thread blocks computing a 64-row x 64-token tile (8x8 per thread, half2
// chains); grid = (panels, ceil(M/64)); tile (row0, nrows) derived from blockIdx.y so no
// tiles array exists (capture-safe by construction, though prefill never captures).
//
// NUMERICS. Accumulation is fp16 (__hfma2 chains) exactly as the engine ships it on P100;
// the k-order per output element is the engine's (slab-major, strict). This is NOT the
// dequant+cuBLAS numerics — the serving quality gate (same-top-token) is MANDATORY before
// this route is enabled anywhere, and it ships default-OFF behind PXQ4_GEMM2D=1.
//
// Addressing edits vs the engine (same contract as pxq4_kernel.cuh's header):
//   pan = slabs + p*kslabs*SLAB (HDR removed); anch = anchor[p*64 + tid]; A fp16 [M,K]
//   row-major; C fp16 [M,N] row-major (engine wrote fp32 + bias; vLLM adds bias outside).

#pragma once

#include "pxq4_kernel.cuh"

#define PXQ4_G2D_BN 64   // token tile width (engine PXQ4_BN)

template <bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq4_gemm2d(const uint8_t * __restrict__ slabs,   // [panels, kslabs, 1088]
              const __half  * __restrict__ anchor,  // [panels, 64]
              const __half  * __restrict__ A,       // [M, K]
              __half        * __restrict__ C,       // [M, N]
              const int M, const int N, const int K) {
    const int panels = N / PXQ4_BM;
    const int kslabs = K / PXQ4_QK;
    const int p    = blockIdx.x;
    const int row0 = blockIdx.y * PXQ4_G2D_BN;
    const int nrows = min(PXQ4_G2D_BN, M - row0);
    (void)panels;

    const uint8_t * pan = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const __half  * At  = A + (size_t)row0 * K;
    __half        * Ct  = C + (size_t)row0 * N + (size_t)p * PXQ4_BM;

    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ __half sW[PXQ4_QK][PXQ4_BM];      // [32 k][64 rows]
    __shared__ __half sA[PXQ4_QK][PXQ4_G2D_BN];  // [32 k][64 tokens]
    const int tid = threadIdx.x;
    pxq4_pol::stage_tabs(tab, sub, tid);
    const float anch = __half2float(anchor[(size_t)p * PXQ4_BM + tid]);

    const int tx = tid & 7, ty = tid >> 3;
    __half2 acc[8][4];
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < nrows;
    const bool fma_on  = !RAG || (8 * ty) < nrows;

    uint32_t qn[4] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {
        pxq4_ldcodes(pan + (size_t)0 * pxq4_pol::SLAB + pxq4_pol::CODE_OFF
                     + tid * pxq4_pol::CODE_BYTES, qn);
        if (a_valid) {
            const __half * src = At + (size_t)tid * K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + (size_t)kb * pxq4_pol::SLAB;
        uint32_t q[4]; uint4 a0, a1, a2, a3;
        if (PIPE) {
#pragma unroll
            for (int i = 0; i < 4; ++i) q[i] = qn[i];
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {
                const uint8_t * slabn = pan + (size_t)(kb + 1) * pxq4_pol::SLAB;
                pxq4_ldcodes(slabn + pxq4_pol::CODE_OFF + tid * pxq4_pol::CODE_BYTES, qn);
                if (a_valid) {
                    const __half * src = At + (size_t)tid * K + (kb + 1) * PXQ4_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + tid * pxq4_pol::CODE_BYTES, q);
            if (a_valid) {
                const __half * src = At + (size_t)tid * K + kb * PXQ4_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        {   // pxq6_deq_slab_cm, pxq4_pol arm: thread = weight row, 32 cols to sW[c][row]
            float eff[PXQ4_NEFF];
            pxq4_pol::row_effs(slab, tid, anch, sub, eff);
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const float  e = eff[(b * PXQ4_NEFF) >> 4];
                const float2 vv = pxq4_pol::pair(q, b, tab);
                sW[2 * b][tid]     = __float2half_rn(e * vv.x);
                sW[2 * b + 1][tid] = __float2half_rn(e * vv.y);
            }
        }
        if (a_valid) {
            const __half * h0 = (const __half *)&a0; const __half * h1 = (const __half *)&a1;
            const __half * h2 = (const __half *)&a2; const __half * h3 = (const __half *)&a3;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sA[i][tid] = h0[i]; sA[8 + i][tid] = h1[i];
                sA[16 + i][tid] = h2[i]; sA[24 + i][tid] = h3[i];
            }
        } else {
            const __half hz = __float2half_rn(0.f);
#pragma unroll
            for (int i = 0; i < PXQ4_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
#pragma unroll 4
            for (int kk = 0; kk < PXQ4_QK; ++kk) {
                __half2 a2v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const __half2 *)&sA[kk][8 * ty + 2 * j];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __half2 wp  = *(const __half2 *)&sW[kk][8 * tx + 2 * i];
                    const __half2 wlo = __low2half2(wp), whi = __high2half2(wp);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[2 * i][j]     = __hfma2(wlo, a2v[j], acc[2 * i][j]);
                        acc[2 * i + 1][j] = __hfma2(whi, a2v[j], acc[2 * i + 1][j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8 * tx + r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8 * ty + 2 * j;
            if (t < nrows)     Ct[(size_t)t * N + row]       = __low2half(acc[r][j]);
            if (t + 1 < nrows) Ct[(size_t)(t + 1) * N + row] = __high2half(acc[r][j]);
        }
    }
}
