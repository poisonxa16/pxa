#include "../common.cuh"

// PXA_SPEC_SMALLN (B4 SPEC_VERIFY_ENGINE, 2026-07-22):
// multi-column dequant-FMA GEMV for the dense quantized backbone at ne11 = 2..8 on
// pre-Volta cards (P100 sm_60 has NO dp4a; 1080Ti sm_61 emulates poorly at tiny N).
//
// Why: speculative/MTP verify runs the whole dense backbone (MXFP4 attn/shexp projections +
// the q8_0 output head) at ne11 = k+1. Today that shape rides int8 MMVQ, whose emulated dp4a
// compute scales with columns — measured verify(4)/verify(1) = 1.646x on P100 (B4 audit,
// 2026-07-22), which eats most of the MTP acceptance win. The cuBLAS redirect was tried and
// measured WORSE (2.28x — per-call dequant + setup dominates at tiny N). This kernel is the
// third option: one weight load -> R column FMAs (the weight stream is the bandwidth wall;
// extra columns are near-free), fp32 accumulation, deterministic per-launch.
// MXFP4 scale decode is byte-identical to dequantize_block_mxfp4 in convert.cu.
//
// Env: PXA_SPEC_SMALLN=1 (default OFF). G3-class vs the mmvq path it replaces (different
// reduction order); run-to-run deterministic.

void ggml_cuda_op_pxa_smalln(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// true when the (type, shape, cc) combination is one this path serves
bool ggml_cuda_pxa_smalln_supported(ggml_type src0_type);
