// Standalone microbench for the PXA small-R f16 decode GEMV geometry (PXA_GEMV_RPB /
// PXA_GEMV_NWARPS). Kernel bodies are copied verbatim from ggml-cuda.cu so the numbers describe
// the shipped code, not a paraphrase of it.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

#define WARP_SIZE 32

__global__ void __launch_bounds__(128) k_stock(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row = blockIdx.x;
    const half2  * w2 = (const half2  *)((const char *) w + (size_t) row * nb01);
    const float2 * x2 = (const float2 *)x;
    const int k2max = ne00/2;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < k2max; k += 128) {
        const half2  hw = w2[k];
        const float2 fx = x2[k];
        sum += __low2float(hw)*fx.x + __high2float(hw)*fx.y;
    }
    __shared__ float warpsum[4];
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) y[row] = (warpsum[0] + warpsum[1]) + (warpsum[2] + warpsum[3]);
}

template <int NW, int RPB>
__global__ void __launch_bounds__(NW*WARP_SIZE) k_rpb(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row0 = blockIdx.x*RPB;
    const half2  * __restrict__ w2 = (const half2  *)((const char *) w + (size_t) row0 * nb01);
    const float2 * __restrict__ x2 = (const float2 *)x;
    const int k2max  = ne00/2;
    const int rstep2 = (int)(nb01/sizeof(half2));
    float sum[RPB];
#pragma unroll
    for (int r = 0; r < RPB; ++r) sum[r] = 0.0f;
    for (int k = threadIdx.x; k < k2max; k += NW*WARP_SIZE) {
        const float2 fx = x2[k];
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            const half2 hw = w2[k + r*rstep2];
            sum[r] += __low2float(hw)*fx.x + __high2float(hw)*fx.y;
        }
    }
    __shared__ float warpsum[NW*RPB];
#pragma unroll
    for (int r = 0; r < RPB; ++r) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) sum[r] += __shfl_down_sync(0xffffffff, sum[r], off);
        if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[r*NW + (threadIdx.x >> 5)] = sum[r];
    }
    __syncthreads();
    if (threadIdx.x < RPB) {
        const float * ws = warpsum + threadIdx.x*NW;
        float acc;
        if      (NW == 4) acc = (ws[0] + ws[1]) + (ws[2] + ws[3]);
        else if (NW == 2) acc =  ws[0] + ws[1];
        else              acc =  ws[0];
        y[row0 + threadIdx.x] = acc;
    }
}

static half *dW; static float *dX, *dY, *dRef;
static int K_, R_;

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

template <class F> static float timeit(F f, int iters) {
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i=0;i<20;++i) f();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i=0;i<iters;++i) f();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms*1000.0f/iters;   // us
}

static void check(const char * name) {
    float * h = (float*)malloc(R_*sizeof(float));
    float * r = (float*)malloc(R_*sizeof(float));
    CK(cudaMemcpy(h, dY, R_*sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r, dRef, R_*sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i=0;i<R_;++i) if (h[i] != r[i]) ++bad;
    printf("   %-14s bit-exact vs stock: %s (%d/%d rows differ)\n", name, bad?"NO":"YES", bad, R_);
    free(h); free(r);
}

int main(int argc, char ** argv) {
    const int iters = 200;
    struct { int K, R; const char * name; } shapes[] = {
        { 10240, 320, "hc_*_down  K=10240 R=320" },
        {  2560,  64, "attn_gate  K=2560  R=64"  },
        {  2560, 320, "midshape   K=2560  R=320" },
    };
    for (auto & s : shapes) {
        K_ = s.K; R_ = s.R;
        const size_t nb01 = (size_t)K_*sizeof(half);
        CK(cudaMalloc(&dW, (size_t)K_*R_*sizeof(half)));
        CK(cudaMalloc(&dX, (size_t)K_*sizeof(float)));
        CK(cudaMalloc(&dY, (size_t)R_*sizeof(float)));
        CK(cudaMalloc(&dRef,(size_t)R_*sizeof(float)));
        half * hW = (half*)malloc((size_t)K_*R_*sizeof(half));
        float * hX = (float*)malloc((size_t)K_*sizeof(float));
        srand(1234);
        for (size_t i=0;i<(size_t)K_*R_;++i) hW[i] = __float2half((rand()/(float)RAND_MAX)*2.f-1.f);
        for (int i=0;i<K_;++i) hX[i] = (rand()/(float)RAND_MAX)*2.f-1.f;
        CK(cudaMemcpy(dW,hW,(size_t)K_*R_*sizeof(half),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dX,hX,(size_t)K_*sizeof(float),cudaMemcpyHostToDevice));

        const double bytes = (double)K_*R_*sizeof(half);
        printf("\n== %s  (%.2f MB weights) ==\n", s.name, bytes/1048576.0);

        auto run_stock = [&]{ k_stock<<<R_,128>>>(dW,dX,dY,K_,nb01); };
        float t = timeit(run_stock, iters);
        CK(cudaMemcpy(dRef,dY,R_*sizeof(float),cudaMemcpyDeviceToDevice));
        printf("   %-14s %7.2f us  %6.1f GB/s   (baseline)\n", "NW=4 RPB=1", t, bytes/t/1000.0);
        const float t0 = t;

#define ARM(NW,RPB) if (R_ % (RPB) == 0) { \
            auto f = [&]{ k_rpb<NW,RPB><<<R_/(RPB), (NW)*WARP_SIZE>>>(dW,dX,dY,K_,nb01); }; \
            float tt = timeit(f, iters); \
            printf("   NW=%d RPB=%-8d %7.2f us  %6.1f GB/s   %+6.1f%%\n", NW, RPB, tt, bytes/tt/1000.0, 100.0*(t0-tt)/t0); \
            char nm[32]; snprintf(nm,sizeof nm,"NW=%d RPB=%d",NW,RPB); check(nm); }
        ARM(4,2) ARM(4,4)
        ARM(2,1) ARM(2,2)
        ARM(1,1)
#undef ARM
        cudaFree(dW);cudaFree(dX);cudaFree(dY);cudaFree(dRef);free(hW);free(hX);
    }
    return 0;
}
