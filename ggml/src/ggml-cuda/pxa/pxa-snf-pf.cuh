// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-snf-pf.cuh -- PXA_VFUSE bit 8 (2026-10-26): k_pxa_smalln_float with the weight row in flight at once.
// The Flash-Next router (f32 [2560 -> 512] at the 2..8-column verify) runs k_pxa_smalln_float: one block per output row,
// thread t walks k = t, t + bs, ... and accumulates acc[c] += w[k] * x[c][k], then a warp xor tree and a block tree.
// The k loop has a run-time trip count, so each step waits for its own weight load (~17 us per call on V100, ~300 GB/s).
// Here the trip count J = ceil(K / bs) is a template constant: every weight of the thread's k sequence is loaded first,
// then the same FMAs run in the same k order on the same lanes, and the reduction is the original's code. Bit-identical
// to k_pxa_smalln_float by construction (tests/test-pxa-snf.cu compares bytes against a verbatim copy of it).
#pragma once

template <typename T, int NC, int J>
static __global__ void k_pxa_smalln_float_pf(const T * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int K, const int ncols, const int64_t w_row_stride, const int64_t x_col_stride, const int64_t y_col_stride) {
    const int row = blockIdx.x;
    const T * wr = w + (int64_t) row*w_row_stride;
    float wv[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int k = threadIdx.x + j*blockDim.x;
        wv[j] = k < K ? pxa_snf_load(wr, k) : 0.0f;
    }
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int k = threadIdx.x + j*blockDim.x;
        if (k < K) {
#pragma unroll
            for (int c = 0; c < NC; ++c) if (c < ncols) acc[c] += wv[j]*x[(int64_t) c*x_col_stride + k];
        }
    }
    __shared__ float red[NC][32];
    const int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE, nw = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o, WARP_SIZE);
        if (lane == 0) red[c][wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float v = lane < nw ? red[c][lane] : 0.0f;
#pragma unroll
            for (int o = WARP_SIZE/2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o, WARP_SIZE);
            if (lane == 0 && c < ncols) y[(int64_t) c*y_col_stride + row] = v;
        }
    }
}

// true = launched (J = ceil(K / bs) in 1..16, 2..4 columns: past that the hoisted x loads cost more than the weight
// prefetch saves; 2560x512 V100: +22/+18/+10% at 2..4, -2/-11% at 5/6; RTX 3080 Ti +25/+26/+18% at 2..4, -11% at 8).
// NC = the column count exactly: the original's NC = 8 with a run-time
// column guard reduces zeros for the absent columns and never writes them, so the written columns are the same sums.
template <typename T, int NC>
static bool pxa_smalln_float_pf_launch_nc(const T * w, const float * x, float * y, int K, int nc, int64_t ws, int64_t xs, int64_t ys,
                                          unsigned R, int bs, cudaStream_t stream) {
    const int J = (K + bs - 1)/bs;
    const dim3 grid(R);
    switch (J) {
#define PXA_SNF_PF(JJ) case JJ: k_pxa_smalln_float_pf<T, NC, JJ><<<grid, bs, 0, stream>>>(w, x, y, K, nc, ws, xs, ys); return true;
        PXA_SNF_PF(1) PXA_SNF_PF(2) PXA_SNF_PF(3) PXA_SNF_PF(4) PXA_SNF_PF(5) PXA_SNF_PF(6) PXA_SNF_PF(7) PXA_SNF_PF(8)
        PXA_SNF_PF(9) PXA_SNF_PF(10) PXA_SNF_PF(11) PXA_SNF_PF(12) PXA_SNF_PF(13) PXA_SNF_PF(14) PXA_SNF_PF(15) PXA_SNF_PF(16)
#undef PXA_SNF_PF
        default: return false;
    }
}
template <typename T>
static bool pxa_smalln_float_pf_launch(const T * w, const float * x, float * y, int K, int nc, int64_t ws, int64_t xs, int64_t ys,
                                       unsigned R, int bs, cudaStream_t stream) {
    switch (nc) {
        case 2: return pxa_smalln_float_pf_launch_nc<T, 2>(w, x, y, K, nc, ws, xs, ys, R, bs, stream);
        case 3: return pxa_smalln_float_pf_launch_nc<T, 3>(w, x, y, K, nc, ws, xs, ys, R, bs, stream);
        case 4: return pxa_smalln_float_pf_launch_nc<T, 4>(w, x, y, K, nc, ws, xs, ys, R, bs, stream);
        default: return false;
    }
}
