// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxa-hccn.cu -- PXA_VFUSE bit 256: k_hc_combine_norm_pf (pxa/pxa-hccn-pf.cuh) is byte-identical to
// k_hc_combine_norm<1024> (hc-fused.cu, copied below verbatim) over qwen4exp-like shapes (n_embd 2048..4096, hc 4, 1..8 tokens,
// with and without the next norm's gamma), then the graph-timed GPU cost at the Flash-Next shape (2560 x 4 streams).
//
// Build: nvcc -O3 -std=c++17 -use_fast_math -arch=sm_70 -I ggml/src/ggml-cuda/pxa -o test-pxa-hccn tests/test-pxa-hccn.cu
// Run:   ./test-pxa-hccn [-nobench]          exit 0 = every byte compare passed
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
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

// ---- verbatim from ggml/src/ggml-cuda/hc-fused.cu ----
template <int block_size>
static __global__ void k_hc_combine_norm(
        const char * __restrict__ r, const char * __restrict__ b, const char * __restrict__ inj,
        const float * __restrict__ gamma, float * __restrict__ dst,
        const int n_embd, const int n_hc,
        const int64_t r_nb1, const int64_t r_nb2, const int64_t b_nb1, const int64_t j_nb0, const int64_t j_nb1,
        const int64_t d_nb3, const float inv_hc, const float eps) {
    const int c   = blockIdx.x % n_hc;
    const int t   = blockIdx.x / n_hc;
    const int tid = threadIdx.x;

    const float * rr = (const float *) (r + c*r_nb1 + t*r_nb2);
    const float * br = (const float *) (b + t*b_nb1);
    const float   jv = *(const float *) (inj + c*j_nb0 + t*j_nb1);

    // scale(inject, 1/hc) -> sigmoid -> scale(., 2)
    const float js = __fmul_rn(inv_hc, jv);
    const float sg = 1.0f / (1.0f + expf(-js));
    const float w  = __fmul_rn(2.0f, sg);

    float * y = dst + ((int64_t) t*n_hc + c)*n_embd;

    float tmp = 0.0f;
    for (int col = tid; col < n_embd; col += block_size) {
        const float xi = __fadd_rn(rr[col], __fmul_rn(br[col], w));
        y[col] = xi;
        tmp += xi * xi;
    }
    if (gamma == nullptr) {
        return;
    }
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }
    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    float * xn = (float *) ((char *) y + d_nb3);
    const float * gm = gamma + (int64_t) c*n_embd;
    for (int col = tid; col < n_embd; col += block_size) {
        const float v = __fmul_rn(scale, y[col]);
        xn[col] = __fmul_rn(v, gm[col]);
    }
}
// ---- end verbatim ----

#include "pxa-hccn-pf.cuh"

struct dims { int n_embd, n_hc, nt; };
static void ship(const float * r, const float * b, const float * j, const float * g, float * d, const dims & s, cudaStream_t st) {
    const int64_t r_nb1 = (int64_t) s.n_embd*4, r_nb2 = r_nb1*s.n_hc, b_nb1 = (int64_t) s.n_embd*4, j_nb0 = 4, j_nb1 = (int64_t) s.n_hc*4;
    const int64_t d_nb3 = (int64_t) s.n_embd*s.n_hc*s.nt*4;
    k_hc_combine_norm<1024><<<s.n_hc*s.nt, 1024, 0, st>>>((const char *) r, (const char *) b, (const char *) j, g, d, s.n_embd, s.n_hc,
        r_nb1, r_nb2, b_nb1, j_nb0, j_nb1, d_nb3, 1.0f/s.n_hc, 1e-6f);
}
static bool pf(const float * r, const float * b, const float * j, const float * g, float * d, const dims & s, cudaStream_t st) {
    const int64_t r_nb1 = (int64_t) s.n_embd*4, r_nb2 = r_nb1*s.n_hc, b_nb1 = (int64_t) s.n_embd*4, j_nb0 = 4, j_nb1 = (int64_t) s.n_hc*4;
    const int64_t d_nb3 = (int64_t) s.n_embd*s.n_hc*s.nt*4;
    return pxa_hc_combine_norm_pf_launch((const char *) r, (const char *) b, (const char *) j, g, d, s.n_embd, s.n_hc, s.nt,
        r_nb1, r_nb2, b_nb1, j_nb0, j_nb1, d_nb3, 1.0f/s.n_hc, 1e-6f, st);
}

int main(int argc, char ** argv) {
    bool bench = true;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "-nobench")) bench = false;
    int dev = 0; cudaDeviceProp p; CUCK(cudaGetDevice(&dev)); CUCK(cudaGetDeviceProperties(&p, dev));
    printf("device %s sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    size_t total = 0, bad = 0;
    for (int n_embd : { 2560, 2048, 3072, 4096, 1536 }) {
        for (int nt = 1; nt <= 8; ++nt) {
            for (int wg = 0; wg < 2; ++wg) {
                const dims s = { n_embd, 4, nt };
                const size_t nr = (size_t) n_embd*4*nt, nb = (size_t) n_embd*nt, nj = (size_t) 4*nt, ng = (size_t) n_embd*4;
                std::vector<float> hr(nr), hb(nb), hj(nj), hg(ng);
                for (size_t i = 0; i < nr; ++i) hr[i] = (i % 89 == 0) ? 0.0f : nd(rng);
                for (auto & v : hb) v = nd(rng)*2.0f;
                for (auto & v : hj) v = nd(rng)*3.0f;
                for (auto & v : hg) v = 1.0f + 0.2f*nd(rng);
                float * dr, * db, * dj, * dg, * d0, * d1;
                CUCK(cudaMalloc(&dr, nr*4)); CUCK(cudaMalloc(&db, nb*4)); CUCK(cudaMalloc(&dj, nj*4)); CUCK(cudaMalloc(&dg, ng*4));
                CUCK(cudaMalloc(&d0, nr*8)); CUCK(cudaMalloc(&d1, nr*8));
                CUCK(cudaMemcpy(dr, hr.data(), nr*4, cudaMemcpyHostToDevice)); CUCK(cudaMemcpy(db, hb.data(), nb*4, cudaMemcpyHostToDevice));
                CUCK(cudaMemcpy(dj, hj.data(), nj*4, cudaMemcpyHostToDevice)); CUCK(cudaMemcpy(dg, hg.data(), ng*4, cudaMemcpyHostToDevice));
                CUCK(cudaMemset(d0, 0x7f, nr*8)); CUCK(cudaMemset(d1, 0x7f, nr*8));
                const float * g = wg ? dg : nullptr;
                ship(dr, db, dj, g, d0, s, 0);
                const bool ok = pf(dr, db, dj, g, d1, s, 0);
                CUCK(cudaDeviceSynchronize());
                std::vector<uint32_t> a(nr*2), c(nr*2);
                CUCK(cudaMemcpy(a.data(), d0, nr*8, cudaMemcpyDeviceToHost)); CUCK(cudaMemcpy(c.data(), d1, nr*8, cudaMemcpyDeviceToHost));
                ++total;
                const bool eq = ok ? !memcmp(a.data(), c.data(), nr*8) : (n_embd <= 1024 || n_embd > 4096);
                if (!eq) { ++bad; printf("  HCCN MISMATCH n_embd=%d nt=%d gamma=%d launched=%d\n", n_embd, nt, wg, (int) ok); }
                if (!ok && n_embd == 1536) { /* 1536 -> NJ 2: must launch */ }
                CUCK(cudaFree(dr)); CUCK(cudaFree(db)); CUCK(cudaFree(dj)); CUCK(cudaFree(dg)); CUCK(cudaFree(d0)); CUCK(cudaFree(d1));
            }
        }
    }
    printf("hc combine_norm prefetch byte-exact vs k_hc_combine_norm<1024>: %zu/%zu\n", total - bad, total);

    if (bench) {
        const int n_embd = 2560, NCOPY = 8, iters = 64;
        cudaStream_t st; CUCK(cudaStreamCreate(&st));
        cudaEvent_t e0, e1; CUCK(cudaEventCreate(&e0)); CUCK(cudaEventCreate(&e1));
        printf("\n[bench] hc combine_norm %d x 4 streams, GPU us per launch (%d launches in one graph, median of 7 replays)\n", n_embd, iters);
        for (int nt = 2; nt <= 6; ++nt) {
            const dims s = { n_embd, 4, nt };
            const size_t nr = (size_t) n_embd*4*nt;
            float * dr, * db, * dj, * dg, * dd;
            CUCK(cudaMalloc(&dr, nr*4*NCOPY)); CUCK(cudaMalloc(&db, (size_t) n_embd*nt*4*NCOPY)); CUCK(cudaMalloc(&dj, 4*nt*4*NCOPY));
            CUCK(cudaMalloc(&dg, (size_t) n_embd*4*4)); CUCK(cudaMalloc(&dd, nr*8*NCOPY));
            CUCK(cudaMemset(dr, 0, nr*4*NCOPY)); CUCK(cudaMemset(db, 0, (size_t) n_embd*nt*4*NCOPY)); CUCK(cudaMemset(dj, 0, 4*nt*4*NCOPY));
            CUCK(cudaMemset(dg, 0, (size_t) n_embd*4*4));
            auto timeit = [&](int which) {
                cudaGraph_t gr; cudaGraphExec_t ge;
                CUCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                for (int it = 0; it < iters; ++it) {
                    const int k = it % NCOPY;
                    const float * r = dr + (size_t) k*nr, * b = db + (size_t) k*n_embd*nt, * j = dj + (size_t) k*4*nt;
                    float * d = dd + (size_t) k*nr*2;
                    if (which == 0) ship(r, b, j, dg, d, s, st); else pf(r, b, j, dg, d, s, st);
                }
                CUCK(cudaStreamEndCapture(st, &gr)); CUCK(cudaGraphInstantiate(&ge, gr, 0));
                std::vector<float> ms;
                for (int rep = 0; rep < 7; ++rep) {
                    CUCK(cudaEventRecord(e0, st)); CUCK(cudaGraphLaunch(ge, st)); CUCK(cudaEventRecord(e1, st));
                    CUCK(cudaEventSynchronize(e1)); float m; CUCK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                CUCK(cudaGraphExecDestroy(ge)); CUCK(cudaGraphDestroy(gr));
                return ms[3]*1000.0f/iters;
            };
            const float a = timeit(0), b = timeit(1);
            printf("  ny=%d  shipped %6.2f us  | pf %6.2f us (%+.0f%%)\n", nt, a, b, 100.0*(a - b)/a);
            CUCK(cudaFree(dr)); CUCK(cudaFree(db)); CUCK(cudaFree(dj)); CUCK(cudaFree(dg)); CUCK(cudaFree(dd));
        }
    }
    printf("\nHCCN %s (%zu failures)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
