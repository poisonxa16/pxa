// pxq4.cuh — PXQ4: the PXA-native quant format (MXFP4 numerics, GEMM-tile-ordered layout).
//
// Design: PXA quant design notes, 2026-07-15 (internal lab). PoC: pxa-bench/pxq4_bench.cu
// (measured 2.53x vs the prod dequant+cublas16F path on P100 at live gpt-oss expert shapes).
//
// On-disk layout (v1, frozen):
//   slab = 64 weight rows x 32 K-values = 1088 B, 64 B aligned:
//     [64 B scales]   s[row] = E8M0 byte for (row, this 32-K block)      (SoA, coalesced)
//     [1024 B nibbles] row r = 16 B at offset 64 + 16*r
//                      byte b of a row = code(k=2b) lo | code(k=2b+1) hi (sequential pairs)
//   slabs K-major within a 64-row panel; panels row-major; experts outermost.
//   Constraints: rows % 64 == 0, K % 32 == 0 (the repack tool enforces; other tensors stay MXFP4).
//   Numerics are EXACTLY MXFP4 (E2M1 codes x E8M0 power-of-two scale): MXFP4 <-> PXQ4 is a lossless
//   bit permutation. type_size=17 / blck_size=32 keeps ggml_row_size/nbytes arithmetic identical.
//
// Env: PXA_PXQ4=0 disables the fused kernels (PXQ4 tensors then run dequant->cublas fallback,
// which is bit-identical f16 to the MXFP4 path). Default ON.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#define PXQ4_QK         32
#define PXQ4_TYPE_SIZE  17
#define PXQ4_BM         64                 // rows per panel (= fused kernel tile height)
#define PXQ4_BN         64                 // token tile width
#define PXQ4_SLAB_BYTES 1088

// (pxq4_kvalues / pxq4_e8m0_to_f32 / pxa_pxq4_enabled — legacy id-250-only helpers — were
//  removed 2026-07-21 with the retirement of GGML_TYPE_PXQ4_LEGACY.)


// per-(expert, token-tile) descriptor for the grouped prefill GEMM
struct pxq4_tile_info {
    int32_t e;      // expert id
    int32_t row0;   // first flat activation row of this tile (expert-grouped order)
    int32_t nrows;  // 1..64 valid tokens in this tile
    int32_t _pad;
};

// layout-compatible with ggml-cuda.cu's mmid_row_mapping {int32 i1, i2}
struct pxq4_rowmap { int32_t i1, i2; };

// E==1 tile map for the 2D (non-MoE) grouped GEMM: one 64-token tile per entry, expert 0, the
// ragged tail carried in the last entry's nrows. Built ON DEVICE on purpose: the MoE prefill
// driver builds its map host-side and cudaMemcpyAsync's it, which is a pageable H2D and is NOT
// legal under stream capture (it survives only because prefill graphs are not captured today).
// A kernel launch is capture-legal and cheaper than the copy.
static __global__ void k_pxq_tiles_2d(pxq4_tile_info * __restrict__ tiles,
                                      const int ny, const int ntiles) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ntiles) return;
    const int r0   = i*PXQ4_BN;
    const int left = ny - r0;
    pxq4_tile_info t;
    t.e = 0; t.row0 = r0; t.nrows = left < PXQ4_BN ? left : PXQ4_BN; t._pad = 0;
    tiles[i] = t;
}

// (The legacy full-matrix dequant kernel k_pxq4_dequant_matrix / dequantize_row_pxq4_cuda,
//  which served only type id 250, was removed 2026-07-21.)

// ---------------------------------------------------------------------------------------------
// activation gather: flat expert-grouped f32 rows -> contiguous f16 [total][K]
// (same mapping semantics as k_copy_src_to_contiguous, plus the f32->f16 convert the
//  cublas fp16 path does anyway — so the fused path sees the same f16 activations.)
// ---------------------------------------------------------------------------------------------
static __global__ void k_pxq4_gather_a_f16(const char * __restrict__ src1, half * __restrict__ A,
                                           const pxq4_rowmap * __restrict__ map,
                                           const int64_t K, const int64_t ne11,
                                           const size_t nb11, const size_t nb12) {
    const int i = blockIdx.x;
    half * dst = A + (size_t)i*K;
    if (map[i].i1 < 0) {   // device-built map: slot past the routed total, no source row
        for (int64_t j = threadIdx.x; j < K; j += blockDim.x) dst[j] = __float2half(0.0f);
        return;
    }
    const int32_t i11 = map[i].i1 % ne11;
    const int32_t i12 = map[i].i2;
    const float * src = (const float *)(src1 + i11*nb11 + i12*nb12);
    for (int64_t j = threadIdx.x; j < K; j += blockDim.x) {
        dst[j] = __float2half(src[j]);
    }
}

// (The legacy grouped prefill GEMM k_pxq4_gemm_grouped, which served only type id 250, was
//  removed 2026-07-21. Its PoC lineage and tile shape live on in the pxq5/pxq6 kernel families.)

// ---------------------------------------------------------------------------------------------
// PXQ4 GLU epilogue: unary==1 => SWIGLU_OAI (gpt-oss: alpha=1.702, gate clamped/silu'd, up=(1+.)),
// unary==0 => plain SILU-swiglu (qwen: silu(gate)*up, with the exact limit<1e-6 branch prod uses).
// xg = GATE buffer, gu = UP buffer (biases already folded into the GEMM epilogues). Output f16/f32
// feeds the PXQ4 down-proj GEMM. Bit-faithful to unary.cu swiglu_oai_kernel / fused_mul_silu_f32.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ float pxq4_glu_apply(float g, float u, int unary, float alpha, float limit) {
    if (unary == 1) {                    // SWIGLU_OAI
        float gi = fminf(g, limit);
        float ui = fmaxf(fminf(u, limit), -limit);
        return gi / (1.0f + expf(-gi * alpha)) * (1.0f + ui);
    }
    // SILU-swiglu
    if (limit < 1e-6f) return (g / (1.0f + expf(-g))) * u;
    float gs = g / (1.0f + expf(-g));
    gs = fminf(gs, limit);
    return gs * fmaxf(-limit, fminf(limit, u));
}

template <typename dst_t>
static __global__ void k_pxq4_glu(const float * __restrict__ xg, const float * __restrict__ gu,
                                  dst_t * __restrict__ dst, const int64_t k,
                                  const int unary, const float alpha, const float limit) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = (dst_t)pxq4_glu_apply(xg[i], gu[i], unary, alpha, limit);
}

// ---------------------------------------------------------------------------------------------
#define PXQ4_MMV_KSEG 4
// (The legacy decode kernels k_pxq4_gateup_mmv / k_pxq4_mmv, which served only type id 250,
//  were removed 2026-07-21. PXQ4_MMV_KSEG above is SHARED — the pxq6 policy family and the
//  ggml-cuda.cu smem sizing still use it.)
