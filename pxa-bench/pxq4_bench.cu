// pxq4_bench.cu — PoC microbench for the PXQ4 quant format (PXA-native, MXFP4 numerics,
// GEMM-tile-ordered layout) on P100 (sm_60).
//
// Compares, at the EXACT live gpt-oss-120b MoE expert shapes (K=2880, R=2880 rows/proj,
// ~64 tokens/expert at ub=2048):
//   A) prod baseline: per-expert MXFP4->f16 dequant (the PXA_MXFP4_DEQ_V2 kernel) into a
//      reused global f16 buffer + cublasGemmEx fp16 (CUBLAS_COMPUTE_16F) — the exact path
//      a production MoE server takes on P100.
//   B) PXQ4 fused: ONE grouped kernel over all experts reading 4.25bpw slabs straight into
//      smem tiles (dequant never touches global memory), half2 HFMA2 accumulate.
//   C) context: cublas with dequant excluded (pre-dequanted), to isolate the dequant share.
//
// Correctness: PXQ4 in-kernel dequant is bit-checked against the prod dequant kernel; both
// GEMM paths are compared against an fp64 CPU reference (max rel err).
//
// Memory budget deliberately small (~250 MB) so it runs on a P100 slice alongside another
// resident workload on the same card.
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1);} } while(0)

// ---------------- MXFP4 (prod on-disk) ----------------
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;   // 17 B, 32 elems
static __device__ __constant__ int8_t kvalues_c[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static const int8_t kvalues_h[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

static inline float e8m0_to_f32_host(uint8_t e) {
    union { float f; uint32_t u; } h;
    h.u = e >= 2 ? (uint32_t)(e - 1) << 23 : (e ? 0x00400000u : 0x00200000u);
    return h.f;
}

// prod dequant (PXA_MXFP4_DEQ_V2 / deq_v3): smem LUT, thread-pair per block, uint4 stores.
static __global__ void deq_v3(const void * __restrict__ vx, half * __restrict__ yy, const int64_t nblk32) {
    __shared__ float tab[16];
    if (threadIdx.x < 16) tab[threadIdx.x] = (float)kvalues_c[threadIdx.x];
    __syncthreads();
    constexpr uint32_t uval[2] = { 0x00200000, 0x00400000 };
    const int64_t b = (int64_t)blockIdx.x*(blockDim.x/2) + threadIdx.x/2;
    if (b >= nblk32) return;
    const int il = threadIdx.x & 1;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + b;
    union { float f; uint32_t u; } helper;
    helper.u = x->e >= 2 ? uint32_t(x->e - 1) << 23u : uval[x->e];
    const float d = helper.f;
    const uint8_t * q4 = x->qs + 8*il;
    half2 lo[4], hi[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = __floats2half2_rn(d*tab[q4[2*j] & 0xf], d*tab[q4[2*j+1] & 0xf]);
        hi[j] = __floats2half2_rn(d*tab[q4[2*j] >> 4],  d*tab[q4[2*j+1] >> 4]);
    }
    half * y = yy + b*32 + 8*il;
    *(uint4 *)(y)      = *(const uint4 *)lo;
    *(uint4 *)(y + 16) = *(const uint4 *)hi;
}

// ---------------- PXQ4 layout ----------------
// Slab = 64 rows x 32 K-values: [64 scale bytes][64 x 16B nibble rows] = 1088 B, payload 64B-aligned.
// Nibble order: byte b of a row = codes for k=2b (lo) | k=2b+1 (hi)  (sequential pairs -> trivial half2).
// Slabs K-major within a 64-row panel; panels row-major; experts outermost.
#define PXQ4_SLAB_BYTES 1088
#define BM 64
#define BN 64
#define BK 32

static void repack_mxfp4_to_pxq4(const block_mxfp4 * src, uint8_t * dst, int E, int R, int K) {
    const int panels = R / BM, kslabs = K / BK, blk_per_row = K / 32;
    for (int e = 0; e < E; ++e)
    for (int p = 0; p < panels; ++p)
    for (int kb = 0; kb < kslabs; ++kb) {
        uint8_t * slab = dst + ((size_t)(e*panels + p)*kslabs + kb)*PXQ4_SLAB_BYTES;
        for (int row = 0; row < BM; ++row) {
            const block_mxfp4 * b = src + (size_t)e*R*blk_per_row + (size_t)(p*BM + row)*blk_per_row + kb;
            slab[row] = b->e;
            for (int j = 0; j < 16; ++j) {
                // src: qs[i] lo = k=i, hi = k=i+16 ; dst byte j: lo = k=2j, hi = k=2j+1
                int k0 = 2*j, k1 = 2*j + 1;
                uint8_t c0 = k0 < 16 ? (b->qs[k0] & 0xf) : (b->qs[k0-16] >> 4);
                uint8_t c1 = k1 < 16 ? (b->qs[k1] & 0xf) : (b->qs[k1-16] >> 4);
                slab[64 + row*16 + j] = (uint8_t)(c0 | (c1 << 4));
            }
        }
    }
}

// full dequant of one PXQ4 expert to f16 [R][K] (for the bit-identity check vs deq_v3)
static __global__ void pxq4_deq_full(const uint8_t * __restrict__ wq, half * __restrict__ y, int R, int K) {
    __shared__ float tab[16];
    if (threadIdx.x < 16) tab[threadIdx.x] = (float)kvalues_c[threadIdx.x];
    __syncthreads();
    const int kslabs = K / BK;
    const int64_t slab_id = blockIdx.x;               // (panel*kslabs + kb)
    const int p = slab_id / kslabs, kb = slab_id % kslabs;
    const int row = threadIdx.x & 63;
    if (threadIdx.x >= 64) return;
    const uint8_t * slab = wq + (size_t)slab_id*PXQ4_SLAB_BYTES;
    union { float f; uint32_t u; } h;
    uint8_t eb = slab[row];
    h.u = eb >= 2 ? (uint32_t)(eb - 1) << 23 : (eb ? 0x00400000u : 0x00200000u);
    const float d = h.f;
    half * dst = y + (size_t)(p*BM + row)*K + kb*BK;
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
        uint8_t q = slab[64 + row*16 + b];
        dst[2*b]   = __float2half_rn(d * tab[q & 0xf]);
        dst[2*b+1] = __float2half_rn(d * tab[q >> 4]);
    }
}

// ---------------- PXQ4 fused grouped GEMM (v1) ----------------
// grid: (panels, T/64, E); block 128 threads.
// C is col-major [R][T] per expert (matches the prod cublas call), f16.
static __global__ void __launch_bounds__(128)
pxq4_gemm(const uint8_t * __restrict__ Wq, const half * __restrict__ A, half * __restrict__ C,
          int R, int K, int T) {
    const int panels = R / BM, kslabs = K / BK;
    const int e = blockIdx.z, p = blockIdx.x, tt = blockIdx.y;
    const uint8_t * Wexp = Wq + ((size_t)(e*panels + p)*kslabs)*PXQ4_SLAB_BYTES;
    const half    * At   = A + ((size_t)e*T + (size_t)tt*BN)*K;
    half          * Ct   = C + (size_t)e*R*T + (size_t)(tt*BN)*R + (size_t)p*BM;

    __shared__ float tab[16];
    __shared__ half sW[BK][BM];
    __shared__ half sA[BK][BN];
    const int tid = threadIdx.x;
    if (tid < 16) tab[tid] = (float)kvalues_c[tid];

    const int tx = tid & 15, ty = tid >> 4;
    half2 acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        // --- dequant W slab into smem (threads 0..63, one row each) ---
        if (tid < 64) {
            const uint8_t * slab = Wexp + (size_t)kb*PXQ4_SLAB_BYTES;
            const int row = tid;
            union { float f; uint32_t u; } h;
            const uint8_t eb = slab[row];
            h.u = eb >= 2 ? (uint32_t)(eb - 1) << 23 : (eb ? 0x00400000u : 0x00200000u);
            const float d = h.f;
            uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sW[2*b][row]   = __float2half_rn(d * tab[qb[b] & 0xf]);
                sW[2*b+1][row] = __float2half_rn(d * tab[qb[b] >> 4]);
            }
        }
        // --- A tile: 128 threads, token t = tid&63, half-range h = tid>>6 ---
        {
            const int t = tid & 63, hh = tid >> 6;
            const half * src = At + (size_t)t*K + kb*BK + hh*16;
            uint4 v0 = *(const uint4 *)(src);
            uint4 v1 = *(const uint4 *)(src + 8);
            const half * h0 = (const half *)&v0;
            const half * h1 = (const half *)&v1;
            #pragma unroll
            for (int i = 0; i < 8; ++i) sA[16*hh + i][t] = h0[i];
            #pragma unroll
            for (int i = 0; i < 8; ++i) sA[16*hh + 8 + i][t] = h1[i];
        }
        __syncthreads();
        // --- FMA: thread = 4 rows x 8 tokens (4 half2 col-pairs) ---
        #pragma unroll 4
        for (int kk = 0; kk < BK; ++kk) {
            half2 a2[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a2[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const half2 w2 = __half2half2(sW[kk][4*tx + r]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[r][j] = __hfma2(w2, a2[j], acc[r][j]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = 4*tx + r, t = 8*ty + 2*j;
            Ct[row + (size_t)t*R]     = __low2half(acc[r][j]);
            Ct[row + (size_t)(t+1)*R] = __high2half(acc[r][j]);
        }
}

// ---------------- PXQ4 fused v2: 64 threads, 8 rows x 8 tokens per thread (FMA/LDS 4.0) ----------------
static __global__ void __launch_bounds__(64)
pxq4_gemm_v2(const uint8_t * __restrict__ Wq, const half * __restrict__ A, half * __restrict__ C,
             int R, int K, int T) {
    const int panels = R / BM, kslabs = K / BK;
    const int e = blockIdx.z, p = blockIdx.x, tt = blockIdx.y;
    const uint8_t * Wexp = Wq + ((size_t)(e*panels + p)*kslabs)*PXQ4_SLAB_BYTES;
    const half    * At   = A + ((size_t)e*T + (size_t)tt*BN)*K;
    half          * Ct   = C + (size_t)e*R*T + (size_t)(tt*BN)*R + (size_t)p*BM;

    __shared__ float tab[16];
    __shared__ half sW[BK][BM];
    __shared__ half sA[BK][BN];
    const int tid = threadIdx.x;
    if (tid < 16) tab[tid] = (float)kvalues_c[tid];

    const int tx = tid & 7, ty = tid >> 3;      // 8 row-groups x 8 col-groups
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        {   // dequant W slab: 64 threads, one row each
            const uint8_t * slab = Wexp + (size_t)kb*PXQ4_SLAB_BYTES;
            const int row = tid;
            union { float f; uint32_t u; } h;
            const uint8_t eb = slab[row];
            h.u = eb >= 2 ? (uint32_t)(eb - 1) << 23 : (eb ? 0x00400000u : 0x00200000u);
            const float d = h.f;
            uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sW[2*b][row]   = __float2half_rn(d * tab[qb[b] & 0xf]);
                sW[2*b+1][row] = __float2half_rn(d * tab[qb[b] >> 4]);
            }
        }
        {   // A tile: 64 threads, one token each, 32 halves (64B)
            const half * src = At + (size_t)tid*K + kb*BK;
            uint4 v0 = *(const uint4 *)(src);
            uint4 v1 = *(const uint4 *)(src + 8);
            uint4 v2 = *(const uint4 *)(src + 16);
            uint4 v3 = *(const uint4 *)(src + 24);
            const half * h0 = (const half *)&v0; const half * h1 = (const half *)&v1;
            const half * h2 = (const half *)&v2; const half * h3 = (const half *)&v3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        }
        __syncthreads();
        #pragma unroll 4
        for (int kk = 0; kk < BK; ++kk) {
            half2 a2[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a2[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[2*i][j]   = __hfma2(wlo, a2[j], acc[2*i][j]);
                    acc[2*i+1][j] = __hfma2(whi, a2[j], acc[2*i+1][j]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = 8*tx + r, t = 8*ty + 2*j;
            Ct[row + (size_t)t*R]     = __low2half(acc[r][j]);
            Ct[row + (size_t)(t+1)*R] = __high2half(acc[r][j]);
        }
}

// ---------------- PXQ4 fused v3: 128 threads, 8 rows x 4 tokens per thread (FMA/LDS 2.67, 4 warps) ----------------
static __global__ void __launch_bounds__(128)
pxq4_gemm_v3(const uint8_t * __restrict__ Wq, const half * __restrict__ A, half * __restrict__ C,
             int R, int K, int T) {
    const int panels = R / BM, kslabs = K / BK;
    const int e = blockIdx.z, p = blockIdx.x, tt = blockIdx.y;
    const uint8_t * Wexp = Wq + ((size_t)(e*panels + p)*kslabs)*PXQ4_SLAB_BYTES;
    const half    * At   = A + ((size_t)e*T + (size_t)tt*BN)*K;
    half          * Ct   = C + (size_t)e*R*T + (size_t)(tt*BN)*R + (size_t)p*BM;

    __shared__ float tab[16];
    __shared__ half sW[BK][BM];
    __shared__ half sA[BK][BN];
    const int tid = threadIdx.x;
    if (tid < 16) tab[tid] = (float)kvalues_c[tid];

    const int tx = tid & 7, ty = tid >> 3;      // 8 row-groups x 16 col-groups
    half2 acc[8][2];
    #pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r][0] = __floats2half2_rn(0.f,0.f); acc[r][1] = __floats2half2_rn(0.f,0.f); }

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        if (tid < 64) {
            const uint8_t * slab = Wexp + (size_t)kb*PXQ4_SLAB_BYTES;
            const int row = tid;
            union { float f; uint32_t u; } h;
            const uint8_t eb = slab[row];
            h.u = eb >= 2 ? (uint32_t)(eb - 1) << 23 : (eb ? 0x00400000u : 0x00200000u);
            const float d = h.f;
            uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sW[2*b][row]   = __float2half_rn(d * tab[qb[b] & 0xf]);
                sW[2*b+1][row] = __float2half_rn(d * tab[qb[b] >> 4]);
            }
        }
        {
            const int t = tid & 63, hh = tid >> 6;
            const half * src = At + (size_t)t*K + kb*BK + hh*16;
            uint4 v0 = *(const uint4 *)(src);
            uint4 v1 = *(const uint4 *)(src + 8);
            const half * h0 = (const half *)&v0; const half * h1 = (const half *)&v1;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[16*hh + i][t] = h0[i]; sA[16*hh + 8 + i][t] = h1[i]; }
        }
        __syncthreads();
        #pragma unroll 4
        for (int kk = 0; kk < BK; ++kk) {
            half2 a2[2];
            a2[0] = *(const half2 *)&sA[kk][4*ty];
            a2[1] = *(const half2 *)&sA[kk][4*ty + 2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                acc[2*i][0]   = __hfma2(wlo, a2[0], acc[2*i][0]);
                acc[2*i][1]   = __hfma2(wlo, a2[1], acc[2*i][1]);
                acc[2*i+1][0] = __hfma2(whi, a2[0], acc[2*i+1][0]);
                acc[2*i+1][1] = __hfma2(whi, a2[1], acc[2*i+1][1]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = 8*tx + r, t = 4*ty + 2*j;
            Ct[row + (size_t)t*R]     = __low2half(acc[r][j]);
            Ct[row + (size_t)(t+1)*R] = __high2half(acc[r][j]);
        }
}

// ---------------- helpers ----------------
static double ms_of(cudaEvent_t a, cudaEvent_t b) { float m; CK(cudaEventElapsedTime(&m, a, b)); return (double)m; }

int main(int argc, char ** argv) {
    int E = argc > 1 ? atoi(argv[1]) : 8;      // experts benched
    int T = argc > 2 ? atoi(argv[2]) : 64;     // tokens per expert (multiple of 64)
    int R = argc > 3 ? atoi(argv[3]) : 2880;   // rows per projection
    int K = 2880;
    const int iters = 20, warm = 3;
    printf("PXQ4 PoC bench: E=%d experts, R=%d, K=%d, T=%d tokens/expert (gpt-oss-120b shapes)\n", E, R, K, T);

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    size_t freeB, totB; CK(cudaMemGetInfo(&freeB, &totB));
    printf("device: %s sm_%d%d, free VRAM %.0f MiB\n", prop.name, prop.major, prop.minor, freeB/1048576.0);

    const int blk_per_row = K/32, panels = R/BM, kslabs = K/BK;
    const size_t nblk_e = (size_t)R*blk_per_row;            // mxfp4 blocks per expert
    const size_t qbytes_e = nblk_e*sizeof(block_mxfp4);
    const size_t pxbytes_e = (size_t)panels*kslabs*PXQ4_SLAB_BYTES;
    if (qbytes_e + 64 != pxbytes_e*17/17 + (qbytes_e+64)-(qbytes_e+64)) {} // sizes: 17B/blk both
    printf("per-expert: MXFP4 %.1f MiB, PXQ4 %.1f MiB (4.25 bpw both)\n", qbytes_e/1048576.0, pxbytes_e/1048576.0);

    // host data
    std::vector<block_mxfp4> hq(nblk_e*E);
    srand(12345);
    for (size_t i = 0; i < hq.size(); ++i) {
        hq[i].e = 118 + rand()%8;                            // sane scale range
        for (int j = 0; j < 16; ++j) hq[i].qs[j] = (uint8_t)(rand() & 0xff);
    }
    std::vector<uint8_t> hpx(pxbytes_e*E);
    repack_mxfp4_to_pxq4(hq.data(), hpx.data(), E, R, K);
    std::vector<half> hA((size_t)E*T*K);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half_rn(((float)(rand()%2001) - 1000.f)/1000.f);

    // device buffers
    block_mxfp4 *dq;  CK(cudaMalloc(&dq,  qbytes_e*E));
    uint8_t *dpx;     CK(cudaMalloc(&dpx, pxbytes_e*E));
    half *dW16;       CK(cudaMalloc(&dW16, (size_t)R*K*sizeof(half)));       // baseline reuse buffer
    half *dW16b;      CK(cudaMalloc(&dW16b,(size_t)R*K*sizeof(half)));       // identity check
    half *dA;         CK(cudaMalloc(&dA,  (size_t)E*T*K*sizeof(half)));
    half *dC_base;    CK(cudaMalloc(&dC_base, (size_t)E*R*T*sizeof(half)));
    half *dC_fused;   CK(cudaMalloc(&dC_fused,(size_t)E*R*T*sizeof(half)));
    CK(cudaMemcpy(dq, hq.data(), qbytes_e*E, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpx, hpx.data(), pxbytes_e*E, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t cb; CB(cublasCreate(&cb));
    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

    // ---- correctness 1: PXQ4 dequant bit-identity vs prod dequant (expert 0) ----
    {
        int nb = (int)((nblk_e + 63)/64);
        deq_v3<<<nb, 128>>>(dq, dW16, nblk_e);
        pxq4_deq_full<<<panels*kslabs, 64>>>(dpx, dW16b, R, K);
        CK(cudaDeviceSynchronize());
        std::vector<half> a((size_t)R*K), b((size_t)R*K);
        CK(cudaMemcpy(a.data(), dW16, a.size()*2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dW16b, b.size()*2, cudaMemcpyDeviceToHost));
        size_t mism = 0;
        for (size_t i = 0; i < a.size(); ++i) if (memcmp(&a[i], &b[i], 2)) ++mism;
        printf("dequant bit-identity (PXQ4 vs prod MXFP4 path): %zu/%zu mismatches %s\n",
               mism, a.size(), mism ? "*** FAIL ***" : "(bit-exact)");
    }

    const half one = __float2half(1.0f), zero = __float2half(0.0f);
    auto run_baseline = [&](bool with_deq) {
        for (int e = 0; e < E; ++e) {
            if (with_deq) {
                int nb = (int)((nblk_e + 63)/64);
                deq_v3<<<nb, 128>>>(dq + (size_t)e*nblk_e, dW16, nblk_e);
            }
            CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, R, T, K,
                            &one, dW16, CUDA_R_16F, K,
                            dA + (size_t)e*T*K, CUDA_R_16F, K,
                            &zero, dC_base + (size_t)e*R*T, CUDA_R_16F, R,
                            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        }
    };
    auto run_fused = [&]() {
        dim3 grid(panels, T/BN, E);
        pxq4_gemm<<<grid, 128>>>(dpx, dA, dC_fused, R, K, T);
    };
    auto run_fused_v2 = [&]() {
        dim3 grid(panels, T/BN, E);
        pxq4_gemm_v2<<<grid, 64>>>(dpx, dA, dC_fused, R, K, T);
    };
    auto run_fused_v3 = [&]() {
        dim3 grid(panels, T/BN, E);
        pxq4_gemm_v3<<<grid, 128>>>(dpx, dA, dC_fused, R, K, T);
    };

    // ---- v1/v2/v3 cross-check: same strict k-order fp16 accumulation => must be bit-identical ----
    {
        std::vector<half> c1((size_t)E*R*T), c2((size_t)E*R*T);
        run_fused(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(c1.data(), dC_fused, c1.size()*2, cudaMemcpyDeviceToHost));
        run_fused_v2(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(c2.data(), dC_fused, c2.size()*2, cudaMemcpyDeviceToHost));
        int m2 = memcmp(c1.data(), c2.data(), c1.size()*2);
        run_fused_v3(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(c2.data(), dC_fused, c2.size()*2, cudaMemcpyDeviceToHost));
        int m3 = memcmp(c1.data(), c2.data(), c1.size()*2);
        printf("kernel cross-check: v2 %s v1, v3 %s v1 (same k-order accum)\n",
               m2 ? "*** DIFFERS ***" : "bit-identical to", m3 ? "*** DIFFERS ***" : "bit-identical to");
    }

    // ---- correctness 2: both GEMMs vs fp64 CPU ref (expert 0, all rows, all T tokens) ----
    {
        run_baseline(true); run_fused(); CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<half> w16((size_t)R*K);
        CK(cudaMemcpy(w16.data(), dW16b, w16.size()*2, cudaMemcpyDeviceToHost)); // expert 0 f16 weights
        std::vector<half> cb_h((size_t)R*T), cf_h((size_t)R*T);
        CK(cudaMemcpy(cb_h.data(), dC_base, cb_h.size()*2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cf_h.data(), dC_fused, cf_h.size()*2, cudaMemcpyDeviceToHost));
        double maxeb = 0, maxef = 0, ss = 0; long n = 0;
        for (int t = 0; t < T; t += 7) {           // sample tokens
            for (int r = 0; r < R; r += 13) {      // sample rows
                double ref = 0;
                for (int k = 0; k < K; ++k)
                    ref += (double)__half2float(w16[(size_t)r*K + k]) * (double)__half2float(hA[(size_t)t*K + k]);
                ss += ref*ref; ++n;
                double eb = fabs((double)__half2float(cb_h[r + (size_t)t*R]) - ref);
                double ef = fabs((double)__half2float(cf_h[r + (size_t)t*R]) - ref);
                if (eb > maxeb) maxeb = eb;
                if (ef > maxef) maxef = ef;
            }
        }
        double rms = sqrt(ss/n);
        printf("GEMM max err / RMS(ref) vs fp64 (sampled, RMS=%.3g): cublas-16F %.4g | pxq4-fused %.4g\n",
               rms, maxeb/rms, maxef/rms);
    }

    // ---- timing ----
    auto timeit = [&](const char * name, auto fn) {
        for (int i = 0; i < warm; ++i) fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CK(cudaEventRecord(ev1));
        CK(cudaEventSynchronize(ev1));
        double ms = ms_of(ev0, ev1)/iters;
        double flops = 2.0*E*R*(double)K*T;
        double qgb = (double)E*qbytes_e;
        printf("%-38s %8.3f ms   %6.2f TFLOP/s   weightQ %6.1f GB/s\n",
               name, ms, flops/ms/1e9, qgb/ms/1e6);
        return ms;
    };

    printf("---- timing (avg of %d, %d warmup) ----\n", iters, warm);
    double t_base  = timeit("A) prod: per-expert deq_v2 + cublas16F", [&]{ run_baseline(true); });
    double t_nodeq = timeit("C) cublas16F only (deq excluded)",       [&]{ run_baseline(false); });
    double t_f1    = timeit("B1) PXQ4 fused v1 (4rx8t/thr, 128thr)",  [&]{ run_fused(); });
    double t_f2    = timeit("B2) PXQ4 fused v2 (8rx8t/thr, 64thr)",   [&]{ run_fused_v2(); });
    double t_f3    = timeit("B3) PXQ4 fused v3 (8rx4t/thr, 128thr)",  [&]{ run_fused_v3(); });
    double t_fused = t_f1 < t_f2 ? (t_f1 < t_f3 ? t_f1 : t_f3) : (t_f2 < t_f3 ? t_f2 : t_f3);
    printf("\ndequant share of baseline: %.1f%%\n", 100.0*(t_base - t_nodeq)/t_base);
    printf("PXQ4 fused (best) speedup vs prod path: %.2fx  (vs cublas-no-deq: %.2fx)\n",
           t_base/t_fused, t_nodeq/t_fused);

    // dense cublas reference (context: what cublas does with a GOOD shape) — optional, needs VRAM
    {
        half *dAd = nullptr, *dCd = nullptr;
        const int Td = 2048;
        if (cudaMalloc(&dAd, (size_t)Td*K*sizeof(half)) == cudaSuccess &&
            cudaMalloc(&dCd, (size_t)R*Td*sizeof(half)) == cudaSuccess) {
            CK(cudaMemset(dAd, 0x3c, (size_t)Td*K*sizeof(half)));
            auto dense = [&]{
                CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, R, Td, K,
                                &one, dW16, CUDA_R_16F, K, dAd, CUDA_R_16F, K,
                                &zero, dCd, CUDA_R_16F, R, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
            };
            for (int i = 0; i < warm; ++i) dense();
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev0));
            for (int i = 0; i < iters; ++i) dense();
            CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
            double ms = ms_of(ev0, ev1)/iters;
            printf("ref) cublas16F dense n=%d:              %8.3f ms   %6.2f TFLOP/s (cublas ceiling at good shape)\n",
                   Td, ms, 2.0*R*(double)K*Td/ms/1e9);
            cudaFree(dAd); cudaFree(dCd);
        } else {
            if (dAd) cudaFree(dAd);
            printf("ref) dense cublas skipped (VRAM)\n");
        }
    }

    cublasDestroy(cb);
    return 0;
}
