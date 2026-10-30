// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-f16ny-pf.cuh -- PXA_VFUSE bit 16 (2026-10-26): k_pxa_gemv_f16_ny with the thread's whole k sequence in flight.
// The qwen4exp hyper-connection inject (f16 [10240 -> 4], two per layer) at the 2..8-column verify runs k_pxa_gemv_f16_ny:
// 4 x ny blocks of 128 threads, each thread walking 40 half2 steps whose loads wait one after the other (~8 us per call on
// V100 for an 80 KB weight). Here the step count J is a template constant: every weight and activation pair of the thread
// is loaded first, then the same expression accumulates in the same k order, and the fold is the original's code.
// Bit-identical to k_pxa_gemv_f16_ny by construction (tests/test-pxa-f16ny.cu: bytes vs a verbatim copy).
#pragma once

template <int J>
static __global__ void __launch_bounds__(128) k_pxa_gemv_f16_ny_pf(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01, const size_t xs_tok, const size_t ys_tok) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const half2  * w2 = (const half2  *)((const char *) w + (size_t) row * nb01);
    const float2 * x2 = (const float2 *)((const char *) x + (size_t) tok * xs_tok);
    const int k2max = ne00/2;
    half2  hw[J];
    float2 fx[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int k = threadIdx.x + j*128;
        if (k < k2max) { hw[j] = w2[k]; fx[j] = x2[k]; }
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int k = threadIdx.x + j*128;
        if (k < k2max) sum += __low2float(hw[j])*fx[j].x + __high2float(hw[j])*fx[j].y;
    }
    __shared__ float warpsum[4];
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        ((float *)((char *) y + (size_t) tok * ys_tok))[row] = (warpsum[0] + warpsum[1]) + (warpsum[2] + warpsum[3]);
    }
}

// true = launched (J = ceil(K/2 / 128) in {8, 16, 24, 32, 40, 48}: the step count rounded up to a multiple of 8, the extra
// steps predicated off, so K up to 12288)
static bool pxa_gemv_f16_ny_pf_launch(const half * w, const float * x, float * y, int ne00, size_t nb01, size_t xs, size_t ys,
                                      dim3 grid, cudaStream_t stream) {
    const int J = (ne00/2 + 127)/128;
    switch ((J + 7)/8) {
        case 1: k_pxa_gemv_f16_ny_pf< 8><<<grid, 128, 0, stream>>>(w, x, y, ne00, nb01, xs, ys); return true;
        case 2: k_pxa_gemv_f16_ny_pf<16><<<grid, 128, 0, stream>>>(w, x, y, ne00, nb01, xs, ys); return true;
        case 3: k_pxa_gemv_f16_ny_pf<24><<<grid, 128, 0, stream>>>(w, x, y, ne00, nb01, xs, ys); return true;
        case 4: k_pxa_gemv_f16_ny_pf<32><<<grid, 128, 0, stream>>>(w, x, y, ne00, nb01, xs, ys); return true;
        case 5: k_pxa_gemv_f16_ny_pf<40><<<grid, 128, 0, stream>>>(w, x, y, ne00, nb01, xs, ys); return true;
        case 6: k_pxa_gemv_f16_ny_pf<48><<<grid, 128, 0, stream>>>(w, x, y, ne00, nb01, xs, ys); return true;
        default: return false;
    }
}
