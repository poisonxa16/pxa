// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-dn-outgate-row.cuh -- the DeltaNet out-gate row body (FUSED_RMS_NORM(ssm_norm) + FUSED_MUL_UNARY(z, SILU)) shared,
// source for source, by the open pxa_dn_rms_silu_gate_f32 (pxa-deltanet-fuse.cuh) and the closed k_pxqn_dn_outgate_rq8
// (pxqn-rq8.cuh, PXA_PXQN_RQ8): one definition, so both kernels compute M with the same expressions and reduction tree.
#pragma once
#include "../common.cuh"

template <int block_size>
static __device__ __forceinline__ void pxa_dn_rms_silu_gate_row(
        const float * __restrict__ xr, const float * __restrict__ w, const float * __restrict__ zr, float * __restrict__ dr,
        const int ncols, const float eps) {
    const int tid = threadIdx.x;
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = xr[col];
        tmp += xi * xi;
    }
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float v  = scale * w[col] * xr[col];
        const float zi = zr[col];
        dr[col] = zi * v / (1.0f + expf(-zi));
    }
}
