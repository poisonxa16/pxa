// PXA_SPEC_SMALLN (B4 SPEC_VERIFY_ENGINE) — see pxa-smalln.cuh for the full rationale.
#include "pxa-smalln.cuh"

// One warp per output row; R spec-batch columns accumulated per weight load.
// MXFP4: warp covers 2 blocks/iter (16 code bytes per block, 1 byte -> elems j and j+16).
// Scale decode is byte-identical to convert.cu:dequantize_block_mxfp4.
template <int R>
static __global__ void k_pxa_smalln_mxfp4(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols, const int nrows, const int stride_y, const int nrows_dst) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    __shared__ float tab[16];
    if (threadIdx.y == 0 && threadIdx.x < 16) {
        tab[threadIdx.x] = (float) kvalues_mxfp4[threadIdx.x];
    }
    __syncthreads();
    if (row >= nrows) {
        return;
    }
    constexpr uint32_t uval[2] = { 0x00200000, 0x00400000 };
    const int nblk = ncols/QK_MXFP4;
    const block_mxfp4 * xr = (const block_mxfp4 *) vx + (size_t) row*nblk;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    const int sub  = threadIdx.x & 15; // code byte within the block
    const int boff = threadIdx.x >> 4; // which of the 2 blocks this iteration

    for (int b = boff; b < nblk; b += 2) {
        const block_mxfp4 * blk = xr + b;
        union { float f; uint32_t u; } helper;
        helper.u = blk->e >= 2 ? uint32_t(blk->e - 1) << 23u : uval[blk->e];
        const float d  = helper.f;
        const int   q  = blk->qs[sub];
        const float v0 = d*tab[q & 0xf];
        const float v1 = d*tab[q >> 4];
        const int   k0 = b*QK_MXFP4 + sub;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float * yr = y + (size_t) r*stride_y;
            acc[r] += v0*yr[k0] + v1*yr[k0 + 16];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float s = warp_reduce_sum(acc[r]);
        if (threadIdx.x == 0) {
            dst[(size_t) r*nrows_dst + row] = s;
        }
    }
}

// Q8_0: warp covers 2 blocks/iter (16 threads x 2 consecutive int8 codes each).
template <int R>
static __global__ void k_pxa_smalln_q8_0(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols, const int nrows, const int stride_y, const int nrows_dst) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int nblk = ncols/QK8_0;
    const block_q8_0 * xr = (const block_q8_0 *) vx + (size_t) row*nblk;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    const int sub  = threadIdx.x & 15; // int8 pair within the block
    const int boff = threadIdx.x >> 4;

    for (int b = boff; b < nblk; b += 2) {
        const block_q8_0 * blk = xr + b;
        const float d = __half2float(blk->d);
        // one aligned 16-bit load for the code pair
        const int16_t qq = *(const int16_t *)(blk->qs + 2*sub);
        const float v0 = d*(float)(int8_t)(qq & 0xff);
        const float v1 = d*(float)(int8_t)(qq >> 8);
        const int  k0 = b*QK8_0 + 2*sub;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float * yr = y + (size_t) r*stride_y;
            acc[r] += v0*yr[k0] + v1*yr[k0 + 1];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float s = warp_reduce_sum(acc[r]);
        if (threadIdx.x == 0) {
            dst[(size_t) r*nrows_dst + row] = s;
        }
    }
}

bool ggml_cuda_pxa_smalln_supported(ggml_type src0_type) {
    return src0_type == GGML_TYPE_MXFP4 || src0_type == GGML_TYPE_Q8_0;
}

template <int R>
static void pxa_smalln_launch(ggml_type type, const void * vx, const float * y, float * dst,
        const int ncols, const int nrows, const int stride_y, const int nrows_dst, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = 2;
    const dim3 block_dims(WARP_SIZE, ROWS_PER_BLOCK, 1);
    const dim3 block_nums((nrows + ROWS_PER_BLOCK - 1)/ROWS_PER_BLOCK, 1, 1);
    switch (type) {
        case GGML_TYPE_MXFP4:
            k_pxa_smalln_mxfp4<R><<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows, stride_y, nrows_dst);
            break;
        case GGML_TYPE_Q8_0:
            k_pxa_smalln_q8_0<R><<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows, stride_y, nrows_dst);
            break;
        default:
            GGML_ABORT("pxa_smalln: unsupported type");
    }
}

void ggml_cuda_op_pxa_smalln(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;
    const int64_t ne0      = dst->ne[0];

    const int id = ggml_cuda_get_device();
    // same convention as mmvq: the main device buffer holds the full-ne0 result rows
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src1_ncols) {
        // R=1 (2026-08-03): the plain decode GEMV. Added after the dmmv route was measured a
        // LOSS on the 122B 4xP100 rig (deep decode 24.97 -> 14.7 t/s): the legacy dmmv template
        // indexes the device-global kvalues table divergently and re-decodes the E8M0 scale per
        // value pair. This kernel stages the table in smem and decodes the scale once per block.
        case 1: pxa_smalln_launch<1>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 2: pxa_smalln_launch<2>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 3: pxa_smalln_launch<3>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 4: pxa_smalln_launch<4>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 5: pxa_smalln_launch<5>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 6: pxa_smalln_launch<6>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 7: pxa_smalln_launch<7>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        case 8: pxa_smalln_launch<8>(src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, ne00, nrows_dst, stream); break;
        default: GGML_ABORT("pxa_smalln: ncols %d out of range", (int) src1_ncols);
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
