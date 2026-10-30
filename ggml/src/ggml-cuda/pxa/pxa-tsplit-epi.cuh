// pxa-tsplit-epi.cuh -- the fused tensor-split reduce epilogue kernel (PXA_TSPLIT_EPI), shared between the open
// engine (reduce.cu: the norm tail, PXA_TSPLIT_EPI_NORM) and the closed libggml-pxqn (the PXQN tail, which it
// instantiates and launches itself). Stages A (partial + push), B (peer sum, residual out) and the RMS
// sum-of-squares are here; TAIL::run finishes the row. Moved out of reduce.cu unchanged.
#pragma once

#include "../common.cuh"
#include <cstdint>
#include <cstdlib>

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
    // PXA_TSPLIT_EPI_RSAG: arrival lines per (slot, source rank) at wflag / rflag (the pipeline's blocks); the split
    // exchange's second token round uses line 1, so it needs >= 2 (0 = not set up: the one-round exchange)
    int          lines      = 0;
};

static __device__ __forceinline__ void pxa_ts_pause() {
#if __CUDA_ARCH__ >= PXA_CC_VOLTA
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

// PXA_TSPLIT_EPI_LEAN (2026-10-20; default 0 = the epilogue as before (no measured win), 3 = lean; bit-identical either way). Two
// latency cuts on the single-block epilogue's critical path after the peer handshake (P100 pair microbench mb/epi-phase:
// 26.5 -> 24.7 us per exchange with both, peer ready; 23.9 -> 20.8 us with the peer late):
//   bit 0 (W): the tail's norm weights are loaded at kernel start (they land while the partial is pushed and the peer
//              polled) and parked in shared memory after phase A; the tail reads w[col] there instead of from global
//              after the sum-of-squares tree. Same values, same products. Needs 2*ncols floats of dynamic smem: the host
//              sets the bit only when that fits (pxa_epi_lean_opt) and then launches with the larger smem.
//   bit 1 (A): after the poll, thread 0 (the thread that observed the peers' tokens) issues the system fence before
//              the block barrier, instead of every thread fencing after it. The acquire chain is the PTX model's
//              (relaxed observation + fence.sc by the observer, then bar.sync to the readers); the phase-A release
//              fence (every writer thread before the token) is unchanged.
#define PXA_EPI_WPF_MAX 8
// PXA_TSPLIT_EPI_RSAG (2026-10-21; opt bit 2): at 3-4 cards the epilogue's exchange is split. The
// one-round exchange pushes the whole row to every peer (n-1 rows out of each card: 60 KB at 4 cards for a 5120 row,
// from one block) and every card sums all n rows. Split: rank q owns the q-th of n column ranges; round 1 sends each
// column range only to its owner, the owner sums it in the ONE rank order (r = 0..n-1, the same float adds as the
// one-round sum), round 2 sends the summed range to every peer. 2(n-1)/n rows out of each card instead of n-1 (half
// at 4 cards), one more token round. Same bits: each summed element is the same rank-ordered sum, computed once
// instead of n times. Only with the epilogue's own push (the 2-card producer push and PXA_TSPLIT_REDUCE_PULL keep
// the one round) and only when the pipeline has >= 2 arrival lines per rank.
// Default ON at exactly 4 cards (opt bit 3; measured 4x P100 G12 2026-10-22: reduce gap p50 61.5 -> 53.1 us, -1.1 ms per
// token on the timeline, greedy output identical); =1 also at 3 cards (opt bit 2, unmeasured); =0 off (the one round).
static inline int pxa_epi_rsag_bit() {
    static const int v = [] { const char * e = getenv("PXA_TSPLIT_EPI_RSAG"); if (!e || !*e) return 8; return atoi(e) != 0 ? 4 : 0; }();
    return v;
}
static __host__ __device__ inline bool pxa_epi_rsag_on(int opt, int nrank) { return ((opt & 4) && nrank >= 3) || ((opt & 8) && nrank == 4); }
static inline int pxa_epi_lean_opt(int ncols) {
    static const int v = [] { const char * e = getenv("PXA_TSPLIT_EPI_LEAN"); return e && *e ? atoi(e) : 0; }();   // default OFF: no measured win (f4/f6 quick)
    int o = v & 3;
    if ((o & 1) && !(ncols <= PXA_EPI_WPF_MAX*1024 && (size_t)2*ncols*sizeof(float) + 1024 <= 48*1024)) o &= ~1;
    return o | pxa_epi_rsag_bit();
}
template <int BS>
static __device__ __forceinline__ void pxa_epi_w_load(float (&wr)[PXA_EPI_WPF_MAX], const float * __restrict__ w,
                                                      const int ncols, const int tid) {
#pragma unroll
    for (int k = 0; k < PXA_EPI_WPF_MAX; ++k) { const int col = tid + k*BS; wr[k] = col < ncols ? __ldg(w + col) : 0.0f; }
}
template <int BS>
static __device__ __forceinline__ void pxa_epi_w_park(const float (&wr)[PXA_EPI_WPF_MAX], float * wsm, const int ncols,
                                                      const int tid) {
#pragma unroll
    for (int k = 0; k < PXA_EPI_WPF_MAX; ++k) { const int col = tid + k*BS; if (col < ncols) wsm[col] = wr[k]; }
}

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
                 void * __restrict__ q8 = nullptr, const int kpad = 0, const int opt = 0) {
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
    const bool wpf = (opt & 1) && w != nullptr;   // PXA_TSPLIT_EPI_LEAN bit 0 (block-uniform)
    float wr[PXA_EPI_WPF_MAX];
    if (wpf) pxa_epi_w_load<BS>(wr, w, ncols, tid);

    // PXA_TSPLIT_EPI_RSAG: the split exchange (block-uniform). Column range q (float4 units) = [q*n4/NRANK, (q+1)*n4/NRANK)
    const bool rsag = pxa_epi_rsag_on(opt, NRANK) && push_payload && task.lines >= 2 && task.nwrite == NRANK - 1;
    // A -- the partial (k_add's a + b when the ADD is folded), kept in smem and pushed
    if (rsag) {
        // round 1: column i goes to its range's owner only (wdata is in rank order without me). One pass over the
        // row, so every owner receives from every peer at once: a loop over the owners in turn made all n - 1 senders
        // write the same card together (ingress-bound phases, the engine's RSAG ran slower than the one-round push)
        for (int i = tid; i < n4; i += BS) {
            float4 a = ((const float4 *)pa)[i];
            if (pb) {
                const float4 b = ((const float4 *)pb)[i];
                a.x = a.x + b.x; a.y = a.y + b.y; a.z = a.z + b.z; a.w = a.w + b.w;
            }
            pxa_epi_row4[i] = a;
            int q = (int)((long long)i*NRANK/n4);   // owner: q*n4/NRANK <= i < (q + 1)*n4/NRANK (floor ranges)
            while (q + 1 < NRANK && i >= (int)((long long)(q + 1)*n4/NRANK)) ++q;
            while (q > 0 && i < (int)((long long)q*n4/NRANK)) --q;
            if (q != task.rank) ((float4 *)((char *)task.wdata[q < task.rank ? q : q - 1] + soff_b))[i] = a;
        }
    } else
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
    // the payload commits before the token that announces it; a producer-pushed payload was fenced by its GEMV,
    // which completed before this kernel started (stream order), so phase A wrote nothing global here
    if (push_payload) __threadfence_system();
    if (wpf) pxa_epi_w_park<BS>(wr, row + ncols, ncols, tid);
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
        if (opt & 2) __threadfence_system();      // PXA_TSPLIT_EPI_LEAN bit 1: the observer's acquire fence
    }
    __syncthreads();
    if (!(opt & 2)) __threadfence_system();
    const bool bad = s_bad != 0;

    if (rsag) {
        // my range: every rank's piece in ONE rank order (the one-round sum's adds), out, and to every peer at my range
        // (the peer's slot from me, a range round 1 never writes: round 1 wrote only the peer's own range there)
        const int lo = (int)((long long)task.rank*n4/NRANK), hi = (int)((long long)(task.rank + 1)*n4/NRANK);
        for (int i = lo + tid; i < hi; i += BS) {
            float4 acc;
            #pragma unroll
            for (int r = 0; r < NRANK; ++r) {
                const float4 o = r == task.rank ? pxa_epi_row4[i] : ((const float4 *)((const char *)task.rdata[r] + soff_b))[i];
                if (r == 0) { acc = o; } else { acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w; }
            }
            if (bad) { const float p = pxa_ts_poison<float>(); acc = make_float4(p, p, p, p); }
            pxa_epi_row4[i] = acc;
            ((float4 *)xout)[i] = acc;
            #pragma unroll 1
            for (int wi = 0; wi < task.nwrite; ++wi) {
                ((float4 *)((char *)task.wdata[wi] + soff_b))[i] = acc;
            }
        }
        __threadfence_system();
        __syncthreads();
        // round 2 on arrival line 1 of this slot
        if (tid == 0) {
            #pragma unroll 1
            for (int wi = 0; wi < task.nwrite; ++wi) {
                *(volatile int *)(task.wflag[wi] + soff_i + ARR_INTS) = token;
            }
            __threadfence_system();
            long long spins = 0;
            bool bad2 = false;
            #pragma unroll 1
            for (int r = 0; r < NRANK && !bad2; ++r) {
                if (r == task.rank) continue;
                const volatile int * other = (const volatile int *)(task.rflag[r] + soff_i + ARR_INTS);
                while (*other != token) {
                    pxa_ts_pause();
                    if (++spins > PXA_TS_SPIN_MAX) { bad2 = true; break; }
                }
            }
            if (bad2) { *(volatile int *)err_flag = 1; s_bad = 1; }
            if (opt & 2) __threadfence_system();
        }
        __syncthreads();
        if (!(opt & 2)) __threadfence_system();
        const bool bad2 = s_bad != 0;
        // the peers' summed ranges, from my own memory
        #pragma unroll 1
        for (int q = 0; q < NRANK; ++q) {
            if (q == task.rank) continue;
            const int qlo = (int)((long long)q*n4/NRANK), qhi = (int)((long long)(q + 1)*n4/NRANK);
            const float4 * src = (const float4 *)((const char *)task.rdata[q] + soff_b);
            for (int i = qlo + tid; i < qhi; i += BS) {
                float4 v = src[i];
                if (bad2) { const float p = pxa_ts_poison<float>(); v = make_float4(p, p, p, p); }
                pxa_epi_row4[i] = v;
                ((float4 *)xout)[i] = v;
            }
        }
    } else
    // B -- every rank's partial in ONE rank order, own at its rank (k_pxa_tsplit_fused's order since bug #247: the
    // replicas are bit-identical at any n; at n = 2 the same bits as the old own + peer); residual stream out
    for (int i = tid; i < n4; i += BS) {
        float4 acc;
        #pragma unroll
        for (int r = 0; r < NRANK; ++r) {
            const float4 o = r == task.rank ? pxa_epi_row4[i] : ((const float4 *)((const char *)task.rdata[r] + soff_b))[i];
            if (r == 0) { acc = o; } else { acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w; }
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
    TAIL::run(row, wpf ? (const float *)(row + ncols) : w, y, scale, ncols, tid, warp, lane, mx, seed, layer, site, k0, q8, kpad);
}


// PXA_TSPLIT_EPI_MT (2026-10-16): the same epilogue at the speculative verify widths (nrows 2..4 rows
// of ncols, row-contiguous). At nrows > 1 the planner used to decline (t->ne[1] == 1), so every split verify forward ran
// k_pxa_tsplit_fused + k_add + a closed routine per reduce (three launches, ~124 per 27B forward). Here block b IS
// the one-row k_pxa_tsplit_epi on row b: its partial (a + b when the ADD is folded) in shared memory and pushed, its
// own arrival line b (k_pxa_tsplit_fused's per-block lines: nrows <= the pipeline's blocks), own first then the peers
// in rank order, the row's sum-of-squares tree (fused_rms_norm_f32<1024>'s, which a closed routine replays per row)
// and TAIL::run on that row's y and max sidecar slot (y + b*ncols, mx + b*(ncols/128): a closed routine's layout).
// Bit-identical to the unfused launches. A block writes y only after it has read its own row of pa / pb, and no
// block reads another block's row, so a y over a dying residual input is safe as in the one-row kernel (the rows of
// y and of the inputs line up: same ncols stride). The device counter advances once (the last block, as in the fused
// reduce). No q8_1 sidecar (sm_70 only). The producer push is one-row only: every payload is pushed here.
template <int NRANK, class TAIL>
static __global__ void __launch_bounds__(1024)
k_pxa_tsplit_epi_mt(const float * pa, const float * pb, float * xout, pxa_ts_task task, int token, int * __restrict__ err_flag,
                    const float * __restrict__ w, float * y, float * mx, const int ncols, const int nrows, const float eps,
                    uint64_t seed, int layer, int site, int64_t k0, const int opt = 0) {
    constexpr int BS       = 1024;
    constexpr int ARR_INTS = PXA_TS_FLAG_STRIDE / sizeof(int);
    extern __shared__ float4 pxa_epi_row4[];
    float * row = (float *)pxa_epi_row4;
    __shared__ float s_sum[32];
    __shared__ int s_bad;
    __shared__ int s_tok;
    __shared__ long long s_slot;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if (tid == 0) {
        s_bad = 0;
        if (task.ctr) {   // k_pxa_tsplit_fused's counter: every block reads it, the last one advances it
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
    const long long soff_b = s_slot * task.slot_bytes + (long long)bid*ncols*(long long)sizeof(float);
    const long long soff_i = s_slot * task.slot_ints + (long long)bid*ARR_INTS;
    const int n4 = ncols / 4;
    const bool wpf = (opt & 1) && w != nullptr;   // PXA_TSPLIT_EPI_LEAN bit 0 (block-uniform)
    float wr[PXA_EPI_WPF_MAX];
    if (wpf) pxa_epi_w_load<BS>(wr, w, ncols, tid);
    const int64_t ro = (int64_t)bid*ncols;
    pa += ro; if (pb) pb += ro; xout += ro; y += ro;
    if (mx) mx += (int64_t)bid*(ncols/128);

    // A -- this row's partial, kept in smem and pushed
    for (int i = tid; i < n4; i += BS) {
        float4 a = ((const float4 *)pa)[i];
        if (pb) {
            const float4 b = ((const float4 *)pb)[i];
            a.x = a.x + b.x; a.y = a.y + b.y; a.z = a.z + b.z; a.w = a.w + b.w;
        }
        pxa_epi_row4[i] = a;
        #pragma unroll 1
        for (int wi = 0; wi < task.nwrite; ++wi) {
            ((float4 *)((char *)task.wdata[wi] + soff_b))[i] = a;
        }
    }
    __threadfence_system();
    if (wpf) pxa_epi_w_park<BS>(wr, row + ncols, ncols, tid);
    __syncthreads();

    // arrival line `bid` of this slot, then poll the peers' line `bid` in my own memory
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
        if (opt & 2) __threadfence_system();      // PXA_TSPLIT_EPI_LEAN bit 1: the observer's acquire fence
    }
    __syncthreads();
    if (!(opt & 2)) __threadfence_system();
    const bool bad = s_bad != 0;

    // B -- every rank's partial in ONE rank order, own at its rank (as above); residual stream out
    for (int i = tid; i < n4; i += BS) {
        float4 acc;
        #pragma unroll
        for (int r = 0; r < NRANK; ++r) {
            const float4 o = r == task.rank ? pxa_epi_row4[i] : ((const float4 *)((const char *)task.rdata[r] + soff_b))[i];
            if (r == 0) { acc = o; } else { acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w; }
        }
        if (bad) { const float p = pxa_ts_poison<float>(); acc = make_float4(p, p, p, p); }
        pxa_epi_row4[i] = acc;
        ((float4 *)xout)[i] = acc;
    }
    __syncthreads();

    // C -- the row's RMS norm tree, then the tail
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
    TAIL::run(row, wpf ? (const float *)(row + ncols) : w, y, scale, ncols, tid, warp, lane, mx, seed, layer, site, k0, nullptr, 0);
}
