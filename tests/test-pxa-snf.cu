// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxa-snf.cu -- PXA_VFUSE bit 8: k_pxa_smalln_float_pf (pxa/pxa-snf-pf.cuh) is byte-identical to
// k_pxa_smalln_float (ggml-cuda.cu, copied below verbatim) at 2..8 columns over router-like shapes, then the graph-timed
// GPU cost of the Flash-Next router (f32 2560 -> 512; 8 weight copies rotated so every launch reads cold weights).
//
// Build: nvcc -O3 -std=c++17 -use_fast_math -arch=sm_70 -I ggml/src/ggml-cuda/pxa -o test-pxa-snf tests/test-pxa-snf.cu
// Run:   ./test-pxa-snf [-nobench]          exit 0 = every byte compare passed
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
typedef __nv_bfloat16 nv_bfloat16;

// ---- verbatim from ggml/src/ggml-cuda.cu (PXA_SMALLN_FLOAT) ----
static __device__ __forceinline__ float pxa_snf_load(const float * p, int64_t i) { return p[i]; }
static __device__ __forceinline__ float pxa_snf_load(const half * p, int64_t i) { return __half2float(p[i]); }
static __device__ __forceinline__ float pxa_snf_load(const nv_bfloat16 * p, int64_t i) { return __bfloat162float(p[i]); }

template <typename T, int NC>
static __global__ void k_pxa_smalln_float(const T * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int K, const int ncols, const int64_t w_row_stride, const int64_t x_col_stride, const int64_t y_col_stride) {
    const int row = blockIdx.x;
    const T * wr = w + (int64_t) row*w_row_stride;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float wv = pxa_snf_load(wr, k);
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < ncols) acc[c] += wv*x[(int64_t) c*x_col_stride + k];
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
// ---- end verbatim ----

#include "pxa-snf-pf.cuh"

static void ship_launch(const float * w, const float * x, float * y, int K, int nc, int64_t ws, int64_t xs, int64_t ys, unsigned R,
                        int bs, cudaStream_t st) {
    k_pxa_smalln_float<float, 8><<<R, bs, 0, st>>>(w, x, y, K, nc, ws, xs, ys);
}

int main(int argc, char ** argv) {
    bool bench = true;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "-nobench")) bench = false;
    int dev = 0; cudaDeviceProp p; CUCK(cudaGetDevice(&dev)); CUCK(cudaGetDeviceProperties(&p, dev));
    printf("device %s sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct shape { int K, R; };
    const shape shapes[] = { {2560, 512}, {2560, 128}, {1024, 256}, {4096, 512}, {3000, 64}, {640, 40} };
    size_t total = 0, bad = 0;
    for (const shape & s : shapes) {
        const int bs = s.K >= 1024 ? 256 : 128;
        std::vector<float> hw((size_t) s.K*s.R), hx((size_t) s.K*8);
        for (auto & v : hw) v = nd(rng)*0.05f;
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (i % 53 == 0) ? 0.0f : nd(rng);
        float * dw, * dx, * dy0, * dy1;
        CUCK(cudaMalloc(&dw, hw.size()*4)); CUCK(cudaMalloc(&dx, hx.size()*4));
        CUCK(cudaMalloc(&dy0, (size_t) s.R*8*4)); CUCK(cudaMalloc(&dy1, (size_t) s.R*8*4));
        CUCK(cudaMemcpy(dw, hw.data(), hw.size()*4, cudaMemcpyHostToDevice));
        CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        for (int nc = 2; nc <= 8; ++nc) {
            CUCK(cudaMemset(dy0, 0xff, (size_t) s.R*8*4)); CUCK(cudaMemset(dy1, 0xff, (size_t) s.R*8*4));
            ship_launch(dw, dx, dy0, s.K, nc, s.K, s.K, s.R, s.R, bs, 0);
            const bool ok = pxa_smalln_float_pf_launch<float>(dw, dx, dy1, s.K, nc, s.K, s.K, s.R, s.R, bs, 0);
            CUCK(cudaDeviceSynchronize());
            std::vector<uint32_t> a((size_t) s.R*8), b((size_t) s.R*8);
            CUCK(cudaMemcpy(a.data(), dy0, a.size()*4, cudaMemcpyDeviceToHost));
            CUCK(cudaMemcpy(b.data(), dy1, b.size()*4, cudaMemcpyDeviceToHost));
            ++total;
            const bool eq = ok ? !memcmp(a.data(), b.data(), a.size()*4) : nc > 4;   // 5-8 columns: declined by design
            if (!eq) { ++bad; printf("  SNF MISMATCH K=%d R=%d nc=%d launched=%d\n", s.K, s.R, nc, (int) ok); }
        }
        CUCK(cudaFree(dw)); CUCK(cudaFree(dx)); CUCK(cudaFree(dy0)); CUCK(cudaFree(dy1));
    }
    printf("router f32 GEMV prefetch byte-exact vs k_pxa_smalln_float: %zu/%zu\n", total - bad, total);

    if (bench) {
        const int K = 2560, R = 512, bs = 256, NCOPY = 8, iters = 64;
        std::vector<float> hw((size_t) K*R*NCOPY), hx((size_t) K*8);
        for (auto & v : hw) v = nd(rng)*0.05f;
        for (auto & v : hx) v = nd(rng);
        float * dw, * dx, * dy;
        CUCK(cudaMalloc(&dw, hw.size()*4)); CUCK(cudaMalloc(&dx, hx.size()*4)); CUCK(cudaMalloc(&dy, (size_t) R*8*4));
        CUCK(cudaMemcpy(dw, hw.data(), hw.size()*4, cudaMemcpyHostToDevice));
        CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        cudaStream_t st; CUCK(cudaStreamCreate(&st));
        cudaEvent_t e0, e1; CUCK(cudaEventCreate(&e0)); CUCK(cudaEventCreate(&e1));
        auto timeit = [&](int nc, int which) {
            cudaGraph_t g; cudaGraphExec_t ge;
            CUCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int it = 0; it < iters; ++it) {
                const float * w = dw + (size_t) (it % NCOPY)*K*R;
                if (which == 0) ship_launch(w, dx, dy, K, nc, K, K, R, R, bs, st);
                else pxa_smalln_float_pf_launch<float>(w, dx, dy, K, nc, K, K, R, R, bs, st);
            }
            CUCK(cudaStreamEndCapture(st, &g)); CUCK(cudaGraphInstantiate(&ge, g, 0));
            std::vector<float> ms;
            for (int r = 0; r < 7; ++r) {
                CUCK(cudaEventRecord(e0, st)); CUCK(cudaGraphLaunch(ge, st)); CUCK(cudaEventRecord(e1, st));
                CUCK(cudaEventSynchronize(e1)); float m; CUCK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
            }
            std::sort(ms.begin(), ms.end());
            CUCK(cudaGraphExecDestroy(ge)); CUCK(cudaGraphDestroy(g));
            return ms[3]*1000.0f/iters;
        };
        printf("\n[bench] router f32 %dx%d (%.2f MB), GPU us per launch (%d launches in one graph, median of 7 replays)\n",
               K, R, K*R*4/1e6, iters);
        for (int nc = 2; nc <= 4; ++nc) {
            const float a = timeit(nc, 0), b = timeit(nc, 1);
            printf("  ny=%d  shipped %6.2f us  | pf %6.2f us (%+.0f%%)\n", nc, a, b, 100.0*(a - b)/a);
        }
    }
    printf("\nHCSNF %s (%zu failures)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
