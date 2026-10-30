// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxa-f16ny.cu -- PXA_VFUSE bit 16: k_pxa_gemv_f16_ny_pf (pxa/pxa-f16ny-pf.cuh) is byte-identical to
// k_pxa_gemv_f16_ny (ggml-cuda.cu, copied below verbatim) on the hc_inject shape (f16 10240 -> 4) and others at 2..8
// columns, then the graph-timed cost (weights rotated over 8 copies).
// Build: nvcc -O3 -std=c++17 -use_fast_math -arch=sm_70 -I ggml/src/ggml-cuda/pxa -o test-pxa-f16ny tests/test-pxa-f16ny.cu
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define WARP_SIZE 32
// ---- verbatim from ggml/src/ggml-cuda.cu (PXA_F16_GEMV_NY) ----
static __global__ void __launch_bounds__(128) k_pxa_gemv_f16_ny(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01, const size_t xs_tok, const size_t ys_tok) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const half2  * w2 = (const half2  *)((const char *) w + (size_t) row * nb01);
    const float2 * x2 = (const float2 *)((const char *) x + (size_t) tok * xs_tok);
    const int k2max = ne00/2;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < k2max; k += 128) {
        const half2  hw = w2[k];
        const float2 fx = x2[k];
        sum += __low2float(hw)*fx.x + __high2float(hw)*fx.y;
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
// ---- end verbatim ----
#include "pxa-f16ny-pf.cuh"

int main(int argc, char ** argv) {
    bool bench = true;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "-nobench")) bench = false;
    int dev = 0; cudaDeviceProp p; CUCK(cudaGetDevice(&dev)); CUCK(cudaGetDeviceProperties(&p, dev));
    printf("device %s sm_%d%d\n", p.name, p.major, p.minor);
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct shape { int K, R; };
    const shape shapes[] = { {10240, 4}, {10240, 16}, {4096, 8}, {2560, 4}, {12288, 4}, {1000, 3} };
    size_t total = 0, bad = 0;
    for (const shape & s : shapes) {
        std::vector<half> hw((size_t) s.K*s.R); std::vector<float> hx((size_t) s.K*8);
        for (auto & v : hw) v = __float2half(nd(rng)*0.05f);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (i % 41 == 0) ? 0.0f : nd(rng);
        half * dw; float * dx, * dy0, * dy1;
        CUCK(cudaMalloc(&dw, hw.size()*2)); CUCK(cudaMalloc(&dx, hx.size()*4));
        CUCK(cudaMalloc(&dy0, (size_t) s.R*8*4)); CUCK(cudaMalloc(&dy1, (size_t) s.R*8*4));
        CUCK(cudaMemcpy(dw, hw.data(), hw.size()*2, cudaMemcpyHostToDevice));
        CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        for (int ny = 2; ny <= 8; ++ny) {
            CUCK(cudaMemset(dy0, 0xff, (size_t) s.R*8*4)); CUCK(cudaMemset(dy1, 0xff, (size_t) s.R*8*4));
            dim3 grid((unsigned) s.R, (unsigned) ny);
            k_pxa_gemv_f16_ny<<<grid, 128>>>(dw, dx, dy0, s.K, (size_t) s.K*2, (size_t) s.K*4, (size_t) s.R*4);
            const bool ok = pxa_gemv_f16_ny_pf_launch(dw, dx, dy1, s.K, (size_t) s.K*2, (size_t) s.K*4, (size_t) s.R*4, grid, 0);
            CUCK(cudaDeviceSynchronize());
            std::vector<uint32_t> a((size_t) s.R*8), b((size_t) s.R*8);
            CUCK(cudaMemcpy(a.data(), dy0, a.size()*4, cudaMemcpyDeviceToHost));
            CUCK(cudaMemcpy(b.data(), dy1, b.size()*4, cudaMemcpyDeviceToHost));
            ++total;
            if (!ok || memcmp(a.data(), b.data(), a.size()*4)) { ++bad; printf("  F16NY MISMATCH K=%d R=%d ny=%d launched=%d\n", s.K, s.R, ny, (int) ok); }
        }
        CUCK(cudaFree(dw)); CUCK(cudaFree(dx)); CUCK(cudaFree(dy0)); CUCK(cudaFree(dy1));
    }
    printf("hc_inject f16 GEMV prefetch byte-exact vs k_pxa_gemv_f16_ny: %zu/%zu\n", total - bad, total);
    if (bench) {
        const int K = 10240, R = 4, NCOPY = 8, iters = 64;
        std::vector<half> hw((size_t) K*R*NCOPY); std::vector<float> hx((size_t) K*8);
        for (auto & v : hw) v = __float2half(nd(rng)*0.05f);
        for (auto & v : hx) v = nd(rng);
        half * dw; float * dx, * dy;
        CUCK(cudaMalloc(&dw, hw.size()*2)); CUCK(cudaMalloc(&dx, hx.size()*4)); CUCK(cudaMalloc(&dy, (size_t) R*8*4));
        CUCK(cudaMemcpy(dw, hw.data(), hw.size()*2, cudaMemcpyHostToDevice));
        CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        cudaStream_t st; CUCK(cudaStreamCreate(&st)); cudaEvent_t e0, e1; CUCK(cudaEventCreate(&e0)); CUCK(cudaEventCreate(&e1));
        auto timeit = [&](int ny, int which) {
            cudaGraph_t g; cudaGraphExec_t ge; dim3 grid((unsigned) R, (unsigned) ny);
            CUCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int it = 0; it < iters; ++it) {
                const half * w = dw + (size_t) (it % NCOPY)*K*R;
                if (which == 0) k_pxa_gemv_f16_ny<<<grid, 128, 0, st>>>(w, dx, dy, K, (size_t) K*2, (size_t) K*4, (size_t) R*4);
                else pxa_gemv_f16_ny_pf_launch(w, dx, dy, K, (size_t) K*2, (size_t) K*4, (size_t) R*4, grid, st);
            }
            CUCK(cudaStreamEndCapture(st, &g)); CUCK(cudaGraphInstantiate(&ge, g, 0));
            std::vector<float> ms;
            for (int r = 0; r < 7; ++r) { CUCK(cudaEventRecord(e0, st)); CUCK(cudaGraphLaunch(ge, st)); CUCK(cudaEventRecord(e1, st));
                CUCK(cudaEventSynchronize(e1)); float m; CUCK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m); }
            std::sort(ms.begin(), ms.end()); CUCK(cudaGraphExecDestroy(ge)); CUCK(cudaGraphDestroy(g));
            return ms[3]*1000.0f/iters;
        };
        printf("\n[bench] hc_inject f16 %dx%d, GPU us per launch (%d launches in one graph, median of 7 replays)\n", K, R, iters);
        for (int ny = 2; ny <= 8; ++ny) {
            const float a = timeit(ny, 0), b = timeit(ny, 1);
            printf("  ny=%d  shipped %6.2f us  | pf %6.2f us (%+.0f%%)\n", ny, a, b, 100.0*(a - b)/a);
        }
    }
    printf("\nHCF16NY %s (%zu failures)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
