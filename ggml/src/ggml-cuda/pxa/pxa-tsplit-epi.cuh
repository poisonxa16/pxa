// pxa-tsplit-epi.cuh -- the fused tensor-split reduce epilogue kernel (PXA_TSPLIT_EPI), shared between the open
// engine (reduce.cu: the norm tail, PXA_TSPLIT_EPI_NORM) and the closed libggml-pxqn (the PXQN tail, which it
// instantiates and launches itself). Stages A (partial + push), B (peer sum, residual out) and the RMS
// sum-of-squares are here; TAIL::run finishes the row. Moved out of reduce.cu unchanged.
#pragma once

#include "../common.cuh"
#include <cstdint>

#define PXA_TS_POOL          4                   // staging slots per device
#define PXA_TS_FLAG_STRIDE   64                  // bytes: one line per (slot, block)
#define PXA_TS_SPIN_MAX      100000000ll

struct pxa_ts_task {
    void *       wdata[GGML_CUDA_MAX_DEVICES];   // where my payload goes   (nwrite entries)
    int  *       wflag[GGML_CUDA_MAX_DEVICES];   // where my token goes     (nwrite entries)
    const void * rdata[GGML_CUDA_MAX_DEVICES];   // where peer r's payload is, indexed by RANK
    const int  * rflag[GGML_CUDA_MAX_DEVICES];   // where peer r's token is,   indexed by RANK
    int          nwrite = 0;
    int          nrank  = 0;
    int          rank   = 0;
    int          nelem  = 0;
    // PXA_TSPLIT_GRAPH: non-null = the slot and token come from this device-local counter pair
    // ([0] = reduce counter, [1] = blocks that have read it); the four pointer tables above then
    // address slot 0 and the kernel adds slot * stride.
    int *        ctr        = nullptr;
    long long    slot_bytes = 0;         // payload stride per slot, bytes
    long long    slot_ints  = 0;         // arrival-line stride per slot, ints
};

static __device__ __forceinline__ void pxa_ts_pause() {
#if __CUDA_ARCH__ >= CC_VOLTA
    __nanosleep(100);
#else
    // sm_60 has no __nanosleep. A short clock64() backoff keeps the poll off the
    // PCIe BAR between reads without spinning the SM at full rate.
    const long long t0 = clock64();
    while (clock64() - t0 < 128) { __threadfence_block(); }
#endif
}

template <typename T> static __device__ __forceinline__ T pxa_ts_poison();
template <> __device__ __forceinline__ float pxa_ts_poison<float>() { return __int_as_float(0x7fffffff); }
template <> __device__ __forceinline__ half  pxa_ts_poison<half >() { return __ushort_as_half(0x7fff);   }

// the open tail: the norm node's output is written directly, fused_rms_norm_f32<1024>'s scale * y[col] * x[col]
struct pxa_epi_tail_norm {
    static __device__ __forceinline__ void run(float * row, const float * __restrict__ w, float * y, const float scale,
                                               const int ncols, const int tid, const int warp, const int lane,
                                               float * mx, uint64_t seed, int layer, int site, int64_t k0,
                                               void * q8, const int kpad) {
        constexpr int BS = 1024;
        for (int col = tid; col < ncols; col += BS) y[col] = scale * w[col] * row[col];
    }
};

template <int NRANK, class TAIL>
static __global__ void __launch_bounds__(1024)
k_pxa_tsplit_epi(const float * pa, const float * pb, float * xout, pxa_ts_task task, int token, int * __restrict__ err_flag,
                 const float * __restrict__ w, float * y, float * mx, const int ncols, const float eps,
                 uint64_t seed, int layer, int site, int64_t k0, const int push_payload,
                 void * __restrict__ q8 = nullptr, const int kpad = 0) {
    constexpr int BS       = 1024;
    constexpr int ARR_INTS = PXA_TS_FLAG_STRIDE / sizeof(int);
    extern __shared__ float4 pxa_epi_row4[];
    float * row = (float *)pxa_epi_row4;
    __shared__ float s_sum[32];
    __shared__ int s_bad;
    __shared__ int s_tok;
    __shared__ long long s_slot;
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_bad = 0;
        if (task.ctr) {   // the device-side counter, exactly as k_pxa_tsplit_fused (one block: it advances it)
            const unsigned c = (unsigned)*(volatile int *)task.ctr;
            __threadfence();
            const int prev = atomicAdd(task.ctr + 1, 1);
            if (prev == (int)gridDim.x - 1) {
                *(volatile int *)(task.ctr + 1) = 0;
                *(volatile int *)task.ctr = (int)(c + 1u);
                __threadfence();
            }
            s_tok  = (int)(c + 1u);
            s_slot = (long long)(c % (unsigned)PXA_TS_POOL);
        } else {
            s_tok  = token;
            s_slot = 0;
        }
    }
    __syncthreads();
    token = s_tok;
    const long long soff_b = s_slot * task.slot_bytes;
    const long long soff_i = s_slot * task.slot_ints;
    const int n4 = ncols / 4;

    // A -- the partial (k_add's a + b when the ADD is folded), kept in smem and pushed
    for (int i = tid; i < n4; i += BS) {
        float4 a = ((const float4 *)pa)[i];
        if (pb) {
            const float4 b = ((const float4 *)pb)[i];
            a.x = a.x + b.x; a.y = a.y + b.y; a.z = a.z + b.z; a.w = a.w + b.w;
        }
        pxa_epi_row4[i] = a;
        if (!push_payload) continue;   // the producer GEMV already wrote it into the peer's slot
        #pragma unroll 1
        for (int wi = 0; wi < task.nwrite; ++wi) {
            ((float4 *)((char *)task.wdata[wi] + soff_b))[i] = a;
        }
    }
    __threadfence_system();   // the payload commits before the token that announces it
    __syncthreads();

    // flag line 0 of this slot (one block), then poll the peers' line 0 in my own memory
    if (tid == 0) {
        #pragma unroll 1
        for (int wi = 0; wi < task.nwrite; ++wi) {
            *(volatile int *)(task.wflag[wi] + soff_i) = token;
        }
        __threadfence_system();
        long long spins = 0;
        bool bad = false;
        #pragma unroll 1
        for (int r = 0; r < NRANK && !bad; ++r) {
            if (r == task.rank) continue;
            const volatile int * other = (const volatile int *)(task.rflag[r] + soff_i);
            while (*other != token) {
                pxa_ts_pause();
                if (++spins > PXA_TS_SPIN_MAX) { bad = true; break; }
            }
        }
        if (bad) { *(volatile int *)err_flag = 1; s_bad = 1; }
    }
    __syncthreads();
    __threadfence_system();
    const bool bad = s_bad != 0;

    // B -- own first, then the peers in rank order (k_pxa_tsplit_fused's order); residual stream out
    for (int i = tid; i < n4; i += BS) {
        float4 acc = pxa_epi_row4[i];
        #pragma unroll
        for (int r = 0; r < NRANK; ++r) {
            if (r == task.rank) continue;
            const float4 o = ((const float4 *)((const char *)task.rdata[r] + soff_b))[i];
            acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
        }
        if (bad) { const float p = pxa_ts_poison<float>(); acc = make_float4(p, p, p, p); }
        pxa_epi_row4[i] = acc;
        ((float4 *)xout)[i] = acc;
    }
    __syncthreads();

    // C -- RMS norm of the summed row (fused_rms_norm_f32<1024>'s sum-of-squares tree), then the tail
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += BS) {
        const float xi = row[col];
        tmp += xi*xi;
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, m, 32);
    const int warp = tid/32, lane = tid%32;
    if (lane == 0) s_sum[warp] = tmp;
    __syncthreads();
    tmp = lane < BS/32 ? s_sum[lane] : 0.0f;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, m, 32);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    TAIL::run(row, w, y, scale, ncols, tid, warp, lane, mx, seed, layer, site, k0, q8, kpad);
}

