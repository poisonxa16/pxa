//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "reduce.cuh"
#include "ggml-common.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

template <typename T, int block_size>
static __global__ void k_add(int nelem, const T * __restrict__ src, T * __restrict__ dst) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
#if __CUDA_ARCH__ >= CC_AMPERE
        dst[i] += src[i];
#else
        dst[i] = __float2bfloat16((float)src[i] + (float)dst[i]);
#endif
    } else {
        dst[i] += src[i];
    }
}

template <int block_size>
static __global__ void k_add(int nelem, const block_q8_0 * __restrict__ src, block_q8_0 * __restrict__ dst) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    int ib = i / QK8_0;
    int iq = i % QK8_0;
    float x = (float)src[ib].d * src[ib].qs[iq] + (float)dst[ib].d * dst[ib].qs[iq];
    float ax = fabsf(x);
    float max = warp_reduce_max(ax);
    float d = max / 127;
    float id = d > 0 ? 1/d : 0;
    dst[ib].qs[iq] = roundf(x * id);
    if (threadIdx.x % WARP_SIZE == 0) {
        dst[ib].d = (half)d;
    }
}

template <typename T, int block_size>
static __global__ void k_add_sym(int nelem, T * src, T * dst) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    dst[i] += src[i];
    src[i] = dst[i];
}

struct copy_task {
    void * ptrs[GGML_CUDA_MAX_DEVICES];
    int nptr;
    int nelem;
};

template <typename T, int block_size>
static __global__ void k_reduce_add(copy_task task) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= task.nelem) return;
    auto dst = (T *)task.ptrs[0];
    for (int j = 1; j < task.nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        dst[i] += src[i];
    }
    for (int j = 1; j < task.nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        src[i] = dst[i];
    }
}

template <typename T, int block_size, int nptr>
static __global__ void k_reduce_add_T(copy_task task) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= task.nelem) return;
    auto dst = (T *)task.ptrs[0];
    #pragma unroll
    for (int j = 1; j < nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        dst[i] += src[i];
    }
    #pragma unroll
    for (int j = 1; j < nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        src[i] = dst[i];
    }
}

static void copy_missing_tensors(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        int nhave, int ncopy, const int * idx, const int * copy_idx) {

    if (ncopy < 1) return;

    auto & info = ggml_cuda_info();
    auto size = ggml_nbytes(dst);
    int isrc = 0;
    for (int ii = 0; ii < ncopy; ++ii) {
        int i = copy_idx[ii];
        int j = idx[isrc];
        isrc = (isrc + 1)%nhave;
        //printf("%s: copying from device %d to device %d: %p -> %p\n", __func__, j, i, dst->src[j]->data, dst->src[i]->data);
        ggml_cuda_set_device(j);
        CUDA_CHECK(cudaMemcpyPeerAsync(dst->src[i]->data, info.all_ctx[i]->device, dst->src[j]->data, info.all_ctx[j]->device,
                            size, info.all_ctx[j]->stream()));
        CUDA_CHECK(cudaEventRecord(info.all_ctx[j]->copy_event, info.all_ctx[j]->stream()));
    }
    isrc = 0;
    for (int ii = 0; ii < ncopy; ++ii) {
        int i = copy_idx[ii];
        int j = idx[isrc];
        isrc = (isrc + 1)%nhave;
        ggml_cuda_set_device(i);
        CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event, 0));
    }
    ggml_cuda_set_device(ctx.device);
}

// ---------------------------------------------------------------------------
// PXA_REDUCE_PINNED_v1 (2026-09-13) -- a low-latency third route for the
// cross-device REDUCE of the graph/attn split on a two-card, no-NVLink pair.
//
// WHY. On the 2x V100 pair the graph/attn split issues 130 REDUCE ops per
// forward pass (48 delta-net + 16 attention + 64 FFN + 2 in the MTP head),
// every one of them [n_embd, n_tokens] F32 -- 20 KB at decode, 80 KB in the
// MTP verify batch. That is a latency problem, not a bandwidth one, and the
// existing decode route (the p2p-direct branch below) pays for it twice:
//
//   * per reduce it issues 4 cudaEventRecord + 4 cudaStreamWaitEvent + 2
//     kernel launches + ~10 device switches to move 20 KB, and
//   * because its kernel reads the PEER's partial out of peer DEVICE memory,
//     the scheduler must prove the producer kernels COMPLETED before the
//     reduce runs -- that is the per-reduce ggml_backend_synchronize in
//     ggml_backend_sched_compute_splits(). 128 full pipeline drains per pass.
//
// THE ROUTE. Each device stages its own partial into its own pinned host
// slot, publishes a strictly increasing arrival token, spins on the peer's
// token, then reads the peer's slot and sums -- all inside ONE kernel,
// launched on that device's OWN stream. No events, no host handshake, and
// nothing reads peer device memory, so the scheduler's drain is not needed
// either (see pxa_reduce_pinned_handles(), which the scheduler asks).
//
// BIT-IDENTITY. The wire type is the destination type: no narrowing on the
// wire. The p2p-direct kernel computes a0+a1 on one card and a1+a0 on the
// other; IEEE addition is commutative, so summing local+peer in destination
// precision on both cards is bit-identical to the p2p route AND to a
// single-device sum. tests/test-reduce-pinned.cu holds that to the bit.
//
// SLOT SAFETY WITHOUT AN EVENT. Call N writes host slot s = N % POOL. The
// same slot is written again by call N+POOL. Our call N+1 cannot leave its
// spin until the peer has entered ITS call N+1 phase 2, which on the peer's
// stream is ordered after the peer's call N kernel has finished reading slot
// s. Our call N+POOL is ordered after our call N+1 on our own stream, so the
// slot is free by construction and the wraparound needs no cudaEvent at all.
//
// SAFETY VALVE. The spin is bounded (~10 s). On expiry the kernel sets a
// mapped host flag and exits rather than wedging a card; the host reads the
// flag and says so loudly. A wrong number that announces itself beats a GPU
// that has to be reset.
// ---------------------------------------------------------------------------

#define PXA_RP_BLOCKS          8
#define PXA_RP_THREADS         256
#define PXA_RP_ARRIVAL_STRIDE  64                 // one cache line per (slot,rank,block)
#define PXA_RP_POOL            4                  // host staging slots per device
#define PXA_RP_BUF_BYTES       (1024u*1024u)      // per slot per device
#define PXA_RP_SPIN_MAX        100000000ll        // ~10 s at __nanosleep(100)

struct pxa_rp_host_mem {
    uint8_t * host = nullptr;
    uint8_t * dev  = nullptr;
    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc((void **)&host, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) { host = nullptr; return rc; }
        rc = cudaHostGetDevicePointer((void **)&dev, host, 0);
        if (rc != cudaSuccess) { cudaFreeHost(host); host = nullptr; dev = nullptr; }
        return rc;
    }
};

struct pxa_rp_pipeline {
    int             dev[2]    = { 0, 1 };
    size_t          buf_bytes = PXA_RP_BUF_BYTES;
    long long       calls     = 0;
    pxa_rp_host_mem buf[2];
    pxa_rp_host_mem arrival;
    pxa_rp_host_mem err;
};

static bool pxa_reduce_pinned_enabled() {
    static const bool v = [](){ const char * e = getenv("PXA_REDUCE_PINNED"); return e && atoi(e) != 0; }();
    return v;
}

// Built once, on the first predicate call. Returns nullptr when the route is off or the
// machine does not qualify; the decision is therefore per-process and deterministic.
static pxa_rp_pipeline * pxa_rp_get() {
    static pxa_rp_pipeline * p = nullptr;
    static bool tried = false;
    if (tried) return p;
    tried = true;
    if (!pxa_reduce_pinned_enabled()) {
        return nullptr;
    }
    auto & info = ggml_cuda_info();
    if (info.device_count != 2) {
        fprintf(stderr, "PXA_REDUCE_ROUTE: pinned requested but device_count=%d (needs exactly 2) -- staying on the existing routes\n", info.device_count);
        return nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (info.devices[i].cc < CC_VOLTA) {
            fprintf(stderr, "PXA_REDUCE_ROUTE: pinned requested but device %d has cc=%d (needs >= %d for __nanosleep) -- staying on the existing routes\n",
                    i, info.devices[i].cc, CC_VOLTA);
            return nullptr;
        }
    }
    auto * q = new pxa_rp_pipeline();
    const size_t arrival_bytes = (size_t)PXA_RP_POOL * 2 * PXA_RP_BLOCKS * PXA_RP_ARRIVAL_STRIDE;
    const size_t staging_bytes = (size_t)PXA_RP_POOL * q->buf_bytes;
    int cur_dev = 0;
    cudaGetDevice(&cur_dev);
    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
        ggml_cuda_set_device(q->dev[i]);
        ok = q->buf[i].alloc(staging_bytes) == cudaSuccess;
    }
    ggml_cuda_set_device(q->dev[0]);
    ok = ok && q->arrival.alloc(arrival_bytes) == cudaSuccess;
    ok = ok && q->err.alloc(sizeof(int)) == cudaSuccess;
    if (ok) {
        memset(q->arrival.host, 0, arrival_bytes);
        memset(q->err.host, 0, sizeof(int));
    }
    ggml_cuda_set_device(cur_dev);
    if (!ok) {
        fprintf(stderr, "PXA_REDUCE_ROUTE: pinned requested but the pinned-host allocation failed -- staying on the existing routes\n");
        delete q;
        return nullptr;
    }
    fprintf(stderr, "PXA_REDUCE_ROUTE: pinned ON (PXA_REDUCE_PINNED=1) devices=%d,%d cc=%d,%d "
                    "staging=%dx%zuKB/device arrival=%dx2x%dx%dB blocks=%d threads=%d drain-skip=ON\n",
            q->dev[0], q->dev[1], info.devices[0].cc, info.devices[1].cc,
            PXA_RP_POOL, (size_t)(q->buf_bytes >> 10), PXA_RP_POOL, PXA_RP_BLOCKS, PXA_RP_ARRIVAL_STRIDE,
            PXA_RP_BLOCKS, PXA_RP_THREADS);
    p = q;
    return p;
}

static int * pxa_rp_arrival_ptr(const pxa_rp_pipeline * p, int slot, int rank) {
    const size_t off = ((size_t)slot * 2 + rank) * PXA_RP_BLOCKS * PXA_RP_ARRIVAL_STRIDE;
    return (int *)(p->arrival.dev + off);
}

template <typename T> static __device__ __forceinline__ T pxa_rp_add(T a, T b) { a += b; return a; }
template <typename T> static __device__ __forceinline__ T pxa_rp_poison();
template <> __device__ __forceinline__ float pxa_rp_poison<float>() { return __int_as_float(0x7fffffff); }
template <> __device__ __forceinline__ half  pxa_rp_poison<half >() { return __ushort_as_half(0x7fff);   }

// One kernel, three phases. sendbuf and recvbuf are the same in-place tensor.
template <typename T>
static __global__ void k_pxa_reduce_pinned(
        const T * __restrict__ sendbuf,
        T       * __restrict__ recvbuf,
        T       * __restrict__ host_mine,
        const T * __restrict__ host_other,
        int                    count,
        int   *                arrival_mine,
        const int *            arrival_other,
        int                    token,
        int   *                err_flag) {

    constexpr int VEC      = 16 / sizeof(T);   // one 16 B transaction, the widest single copy on Volta
    constexpr int ARR_INTS = PXA_RP_ARRIVAL_STRIDE / sizeof(int);

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;
    const int nvec = count / VEC;
    const int tail = nvec * VEC;

    __shared__ int s_timed_out;
    if (tid == 0) { s_timed_out = 0; }
    __syncthreads();

    // Phase 1: publish our own contribution into our pinned host slot.
    for (int i = gtid; i < nvec; i += gnt) {
        const int off = i * VEC;
        *(int4 *)(host_mine + off) = *(const int4 *)(sendbuf + off);
    }
    if (bid == 0 && tid < count - tail) {
        host_mine[tail + tid] = sendbuf[tail + tid];
    }

    __threadfence_system();   // commit the host writes before the token
    __syncthreads();

    // Phase 2: one arrival slot per block, so blocks proceed independently.
    if (tid == 0) {
        int       * mine  = arrival_mine  + bid * ARR_INTS;
        const int * other = arrival_other + bid * ARR_INTS;
        *(volatile int *)mine = token;
        __threadfence_system();
        long long spins = 0;
        while (*(const volatile int *)other != token) {
#if __CUDA_ARCH__ >= CC_VOLTA
            __nanosleep(100);
#endif
            if (++spins > PXA_RP_SPIN_MAX) { *(volatile int *)err_flag = 1; s_timed_out = 1; break; }
        }
    }
    __syncthreads();
    __threadfence_system();   // acquire the peer's host writes

    // Phase 3: local + peer, in destination precision, in place. A block whose arrival spin
    // timed out writes NaN instead of a sum, so a missed handshake can never be mistaken for a
    // number; the host aborts on the flag at the next reduce.
    const bool bad = s_timed_out != 0;
    for (int i = gtid; i < nvec; i += gnt) {
        const int off = i * VEC;
        T other[VEC];
        *(int4 *)other = *(const int4 *)(host_other + off);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            recvbuf[off + k] = bad ? pxa_rp_poison<T>() : pxa_rp_add(sendbuf[off + k], other[k]);
        }
    }
    if (bid == 0 && tid < count - tail) {
        recvbuf[tail + tid] = bad ? pxa_rp_poison<T>() : pxa_rp_add(sendbuf[tail + tid], host_other[tail + tid]);
    }
}

// The one place eligibility is decided. The scheduler asks this before it drops the
// per-reduce backend drain, and ggml_cuda_op_reduce asks the same function before it
// takes the route -- they can never disagree.
extern "C" bool pxa_reduce_pinned_handles(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_REDUCE)                  return false;
    if ((ggml_op)dst->op_params[0] != GGML_OP_ADD)          return false;
    if (dst->op_params[3] == 1)                             return false;  // reduce-OFF container
    if (dst->op_params[1] != 2 || dst->op_params[2] != 2)   return false;  // nreduce == nhave == 2
    if (dst->op_params[4] != 0)                             return false;  // every device holds a partial
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) return false;
    if (dst->ne[1] >= 32)                                   return false;  // prefill keeps the ring path
    if (!ggml_is_contiguous(dst))                           return false;
    if (!dst->src[0] || !dst->src[1])                       return false;
    return pxa_rp_get() != nullptr;
}

// Returns false only if something the predicate could not see makes the route impossible;
// the caller then has to fall through, so keep this in step with the predicate.
static bool pxa_reduce_pinned_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto * p = pxa_rp_get();
    if (!p) return false;

    auto & info = ggml_cuda_info();

    if (p->err.host && *(volatile int *)p->err.host) {
        fprintf(stderr, "PXA_REDUCE_PINNED: a cross-device arrival spin timed out (>%lld x 100 ns). The reduce that "
                        "timed out wrote NaN rather than a number, so nothing downstream is trustworthy.\n",
                (long long)PXA_RP_SPIN_MAX);
        GGML_ABORT("PXA_REDUCE_PINNED: cross-device arrival spin timed out");
    }

    const int64_t ne        = ggml_nelements(dst);
    const size_t  type_size = ggml_type_size(dst->type);
    const size_t  max_chunk = p->buf_bytes / type_size;

    for (int64_t start = 0; start < ne; start += (int64_t)max_chunk) {
        const int64_t chunk = std::min((int64_t)max_chunk, ne - start);
        const int     slot  = (int)(p->calls % PXA_RP_POOL);
        const int     token = (int)(++p->calls);

        for (int r = 0; r < 2; ++r) {
            const int d    = p->dev[r];
            const int peer = p->dev[1 - r];
            ggml_cuda_set_device(d);
            cudaStream_t stream = info.all_ctx[d]->stream();
            char * data = (char *)dst->src[d]->data + (size_t)start * type_size;
            void * mine  = p->buf[r    ].dev + (size_t)slot * p->buf_bytes;
            void * other = p->buf[1 - r].dev + (size_t)slot * p->buf_bytes;
            if (dst->type == GGML_TYPE_F16) {
                k_pxa_reduce_pinned<half><<<PXA_RP_BLOCKS, PXA_RP_THREADS, 0, stream>>>(
                    (const half *)data, (half *)data, (half *)mine, (const half *)other,
                    (int)chunk, pxa_rp_arrival_ptr(p, slot, r), pxa_rp_arrival_ptr(p, slot, 1 - r),
                    token, (int *)p->err.dev);
            } else {
                k_pxa_reduce_pinned<float><<<PXA_RP_BLOCKS, PXA_RP_THREADS, 0, stream>>>(
                    (const float *)data, (float *)data, (float *)mine, (const float *)other,
                    (int)chunk, pxa_rp_arrival_ptr(p, slot, r), pxa_rp_arrival_ptr(p, slot, 1 - r),
                    token, (int *)p->err.dev);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    ggml_cuda_set_device(ctx.device);
    return true;
}

// ---------------------------------------------------------------------------
// PXA_REDUCE_TIME_v1 (2026-09-13) -- what one reduce actually costs, per route.
//
// PXA_REDUCE_TIME=<n> prints a table every <n> completed samples. A ring of
// event pairs is recorded around the route on the owner's stream and read back
// lazily when its slot comes round again, so the hot path never synchronises;
// a sample whose stop event is not ready yet is dropped rather than waited on.
// The host clock around the same span is the enqueue cost -- the launches,
// the event traffic and the device switches -- which is the half of the bill
// the pinned route is meant to remove.
// ---------------------------------------------------------------------------

enum { PXA_RT_NONE = 0, PXA_RT_NCCL, PXA_RT_RING, PXA_RT_P2P, PXA_RT_STAGED, PXA_RT_PINNED, PXA_RT_N };

static const char * pxa_rt_name(int r) {
    switch (r) {
        case PXA_RT_NCCL:   return "nccl";
        case PXA_RT_RING:   return "ring";
        case PXA_RT_P2P:    return "p2p-direct";
        case PXA_RT_STAGED: return "staged-copy";
        case PXA_RT_PINNED: return "pinned-host";
        default:            return "unrouted";
    }
}

#define PXA_RT_RING_N   64
#define PXA_RT_NE1_MAX  8

struct pxa_rt_bucket { long long n = 0; double gpu_us = 0, gpu_max = 0, host_us = 0, host_max = 0; };

struct pxa_rt_state {
    long long report_every = 0;
    long long done         = 0;
    pxa_rt_bucket b[PXA_RT_N][PXA_RT_NE1_MAX + 1];
    // per-device event rings
    cudaEvent_t ev_a[GGML_CUDA_MAX_DEVICES][PXA_RT_RING_N] = {};
    cudaEvent_t ev_b[GGML_CUDA_MAX_DEVICES][PXA_RT_RING_N] = {};
    int         ev_route[GGML_CUDA_MAX_DEVICES][PXA_RT_RING_N] = {};
    int         ev_ne1  [GGML_CUDA_MAX_DEVICES][PXA_RT_RING_N] = {};
    bool        ev_live [GGML_CUDA_MAX_DEVICES][PXA_RT_RING_N] = {};
    int         head    [GGML_CUDA_MAX_DEVICES] = {};
    bool        made    [GGML_CUDA_MAX_DEVICES] = {};
};

static pxa_rt_state & pxa_rt() {
    static pxa_rt_state st = [](){
        pxa_rt_state s;
        const char * e = getenv("PXA_REDUCE_TIME");
        s.report_every = e ? atoll(e) : 0;
        return s;
    }();
    return st;
}

static void pxa_rt_report() {
    auto & st = pxa_rt();
    fprintf(stderr, "PXA_REDUCE_TIME: %lld samples\n", st.done);
    fprintf(stderr, "PXA_REDUCE_TIME  %-12s %-5s %9s %10s %10s %10s %10s\n",
            "route", "ne1", "n", "gpu_us", "gpu_max", "host_us", "host_max");
    for (int r = 1; r < PXA_RT_N; ++r) {
        for (int k = 0; k <= PXA_RT_NE1_MAX; ++k) {
            const auto & b = st.b[r][k];
            if (!b.n) continue;
            fprintf(stderr, "PXA_REDUCE_TIME  %-12s %-5d %9lld %10.2f %10.2f %10.2f %10.2f\n",
                    pxa_rt_name(r), k, b.n, b.gpu_us / b.n, b.gpu_max, b.host_us / b.n, b.host_max);
        }
    }
}

struct pxa_rt_scope {
    int      dev  = -1;
    int      slot = -1;
    int      r    = PXA_RT_NONE;
    int      ne1  = 0;
    bool     on   = false;
    std::chrono::steady_clock::time_point t0;

    pxa_rt_scope(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
        auto & st = pxa_rt();
        if (st.report_every <= 0) return;
        on  = true;
        dev = ctx.device;
        ne1 = (int)std::min<int64_t>(dst->ne[1], PXA_RT_NE1_MAX);
        if (!st.made[dev]) {
            ggml_cuda_set_device(dev);
            for (int i = 0; i < PXA_RT_RING_N; ++i) {
                CUDA_CHECK(cudaEventCreate(&st.ev_a[dev][i]));
                CUDA_CHECK(cudaEventCreate(&st.ev_b[dev][i]));
            }
            st.made[dev] = true;
        }
        slot = st.head[dev];
        if (st.ev_live[dev][slot]) {
            st.ev_live[dev][slot] = false;
            if (cudaEventQuery(st.ev_b[dev][slot]) == cudaSuccess) {
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, st.ev_a[dev][slot], st.ev_b[dev][slot]) == cudaSuccess) {
                    auto & b = st.b[st.ev_route[dev][slot]][st.ev_ne1[dev][slot]];
                    const double us = ms * 1000.0;
                    b.gpu_us += us;
                    if (us > b.gpu_max) b.gpu_max = us;
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(st.ev_a[dev][slot], ctx.stream()));
        t0 = std::chrono::steady_clock::now();
    }

    void route(int rr) { r = rr; }

    ~pxa_rt_scope() {
        if (!on) return;
        const double host_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        auto & st = pxa_rt();
        auto & ctxinfo = ggml_cuda_info();
        ggml_cuda_set_device(dev);
        cudaEventRecord(st.ev_b[dev][slot], ctxinfo.all_ctx[dev]->stream());
        st.ev_route[dev][slot] = r;
        st.ev_ne1  [dev][slot] = ne1;
        st.ev_live [dev][slot] = true;
        st.head[dev] = (slot + 1) % PXA_RT_RING_N;
        auto & b = st.b[r][ne1];
        b.n++;
        b.host_us += host_us;
        if (host_us > b.host_max) b.host_max = host_us;
        if (++st.done % st.report_every == 0) {
            pxa_rt_report();
        }
    }
};

// The scheduler drops its per-reduce backend drain for exactly the nodes the predicate above
// accepts. Registered from a file-scope constructor; the slot it writes is a plain function
// pointer, zero-initialised before any dynamic initialisation runs, so the order is safe.
static struct pxa_reduce_pinned_registrar {
    pxa_reduce_pinned_registrar() {
        ggml_backend_set_reduce_pinned_predicate(&pxa_reduce_pinned_handles);
    }
} g_pxa_reduce_pinned_registrar;

void ggml_cuda_op_reduce([[maybe_unused]] ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    auto op = (ggml_op)dst->op_params[0];
    GGML_ASSERT(op == GGML_OP_ADD);
    int nreduce = dst->op_params[1];
    int nhave   = dst->op_params[2];
    if (getenv("PXA_REDUCE_CAPTURE")) { // PXA_REDUCE_PATH diag: shape + p2p + is-this-reduce-being-captured
        static long _n=0;
        cudaStreamCaptureStatus _st=cudaStreamCaptureStatusNone; cudaStreamIsCapturing(ctx.stream(),&_st);
        if ((_n++ % 500)==0)
            fprintf(stderr,"PXA_REDUCE_PATH ne0=%ld ne1=%ld nhave=%d p2p=%d op3=%d capturing=%d call=%ld\n",
                (long)dst->ne[0],(long)dst->ne[1],nhave,(int)ctx.p2p_enabled,(int)dst->op_params[3],(int)(_st==cudaStreamCaptureStatusActive),_n);
    }
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32 ||
                dst->type == GGML_TYPE_Q8_0 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(nhave >= 2 && nhave <= nreduce);
    // PXA_RDBG: per-reduce diagnostic (env-gated, first N calls)
    static const long _rdbg_n = getenv("PXA_RDBG") ? atol(getenv("PXA_RDBG")) : 0;
    static long _rdbg_i = 0;
    const bool _rdbg = _rdbg_i < _rdbg_n;
    if (_rdbg) {
        fprintf(stderr, "PXA_RDBG[%ld] %s ne=%ldx%ld nreduce=%d nhave=%d op3=%d op4=%d ctxdev=%d p2p=%d dstdata_is_src%d\n",
            _rdbg_i, dst->name, (long)dst->ne[0], (long)dst->ne[1], nreduce, nhave,
            dst->op_params[3], dst->op_params[4], ctx.device, (int)ctx.p2p_enabled,
            (dst->src[ctx.device] && dst->data == dst->src[ctx.device]->data) ? ctx.device : -1);
        for (int _j = 0; _j < nreduce; ++_j) fprintf(stderr, "PXA_RDBG[%ld]   src[%d]=%s\n", _rdbg_i, _j, dst->src[_j] ? dst->src[_j]->name : "(null)");
        ++_rdbg_i;
    }
    if (dst->op_params[3] == 1) {
        if (_rdbg) fprintf(stderr, "PXA_RDBG   -> BRANCH reduce-OFF\n");
        // The dst tensor is just a container for the sources and the reduce op is turned off
        return;
    }

    // PXA_REDUCE_TIME_v1: times whichever route is taken, per route and per row count. The
    // destructor closes the measurement on every return path below.
    pxa_rt_scope _rt(ctx, dst);

    // PXA_REDUCE_PINNED_v1: the third route. The scheduler has already dropped its per-reduce
    // backend drain for exactly the nodes this predicate accepts, so a "yes" here is binding.
    if (pxa_reduce_pinned_handles(dst)) {
        if (_rdbg) fprintf(stderr, "PXA_RDBG   -> BRANCH pinned-host\n");
        _rt.route(PXA_RT_PINNED);
        if (!pxa_reduce_pinned_run(ctx, dst)) {
            GGML_ABORT("PXA_REDUCE_PINNED: the predicate accepted this reduce but the route refused it; "
                       "the scheduler has already skipped the drain, so falling through would be unsound");
        }
        return;
    }

    auto & info = ggml_cuda_info();
#ifdef GGML_USE_NCCL
    // Somehow I'm not able to figure out how to use NCCL correctly.
    // It does not work at all if not all GPUs participate in the reduce op, and we
    // get suboptimal prompt processing performance when we have more than 2 GPUs.
    // Hence, if enabled, we use NCCL only for the cases where it works and performs well.
#if __CUDA_ARCH__ >= CC_AMPERE
    constexpr bool bf16_supported = true;
#else
    constexpr bool bf16_supported = false;
#endif
    // PXA_REDUCE_NCCL_v1 (route selector, default = stock ON). Which branch a reduce takes is not
    // obvious from the source and it MATTERS when debugging -sm graph: GGML_NCCL defaults to ON, so
    // on a 4-GPU box every reduce with nhave == nreduce (the delta-net and MoE reduces) is served by
    // NCCL and returns above all the in-tree peer paths, while a reduce with nhave != nreduce (the
    // standard-attention one, nhave=2) falls through to them. Set PXA_REDUCE_NCCL=0 to force the
    // in-tree paths and A/B the two implementations. MEASURED 2026-08-04 (qwen35moe-122B, 4x P100,
    // -sm graph): NCCL=0 and NCCL=1 are BOTH degenerate and byte-identical, so the graph-split
    // corruption on this arch is NOT the NCCL route -- do not re-chase it. See PXA_RDBG above.
    static const bool _pxa_nccl_ok = getenv("PXA_REDUCE_NCCL") ? atoi(getenv("PXA_REDUCE_NCCL")) != 0 : true;
    if (_pxa_nccl_ok && info.have_nccl && dst->type != GGML_TYPE_Q8_0 && nhave == nreduce && (nhave == 2 || dst->ne[1] < 32) &&
       (dst->type != GGML_TYPE_BF16 || bf16_supported)) {
        GGML_ASSERT(info.have_nccl);
        GGML_ASSERT(info.device_count == nreduce);
        _rt.route(PXA_RT_NCCL);
        auto data_type = dst->type == GGML_TYPE_F32 ? ncclFloat : dst->type == GGML_TYPE_BF16 ? ncclBfloat16 : ncclHalf;
        ncclGroupStart();
        for (int i = 0; i < nreduce; ++i) {
            ggml_cuda_set_device(i);
            auto status = ncclAllReduce(dst->src[i] ? dst->src[i]->data : nullptr,
                    dst->src[i] ? dst->src[i]->data : nullptr,
                    ggml_nelements(dst), data_type, ncclSum, info.nccl_coms[i], info.all_ctx[i]->stream());
            if (status != ncclSuccess) {
                fprintf(stderr, "%s: ncclAllReduce failed with status %d\n", __func__, (int)status);
                GGML_ABORT("Fatal error");
            }
        }
        ncclGroupEnd();
        ggml_cuda_set_device(ctx.device);
        return;
    }
#endif
    GGML_ASSERT(dst->data == dst->src[ctx.device]->data);
    auto nbytes = ggml_nbytes(dst);
    int idx[GGML_CUDA_MAX_DEVICES];
    int copy_idx[GGML_CUDA_MAX_DEVICES];
    int ncopy = 0;
    {
        int ii = 0;
        bool have_this_device = false;
        for (int i = 0; i < nreduce; ++i) {
            if (dst->op_params[4] & (1u << i)) {
                copy_idx[ncopy++] = i;
            }
            else {
                if (dst->src[i]) {
                    idx[ii++] = i;
                    if (i == ctx.device) have_this_device = true;
                }
            }
        }
        GGML_ASSERT(ii == nhave);
        GGML_ASSERT(have_this_device);
    }
    //
    // For prompt processing) the objective is to minimize the amount of data being exchanged between
    // the GPUs, even if this means we need to launch a larger number of kernels (we are bandwidth
    // bound rather than latency bound).
    // The following implements a ring communication+reduction that achieves this goal.
    // I would have thought that this is automatically done by NCCL, but it doesn't look that
    // way (or I simply don't understand how to use NCCL) as the ring implementation bellow achieves quite a bit
    // better performance compared to what I get with NCCL.
    //
    // We do the data reduction in stages. Let's N be the number of GPUs.
    // In each stage, each GPU sends 1/N'th of the data to a peer GPU in a ring fashion
    // (i.e. 0->1, 1->2, 2->3, ..., N-1 ->0). Each GPU then performs the addition with the
    // portion just received. After N-1 stages, each GPU ends up having the full sum for 1/N'th
    // of the data. We then do a second round of N-1 stages where each GPU sends a fully reduced
    // portion to its peer. The following shows how all this works for 2, 3, and 4 GPUs:
    // Worth noting that because in each round each GPU sends and receives data, we use the
    // bidirectional p2p bandwidth, which tends to be 2X the unidirectional bandwidth.
    //
    // Examples
    //
    // ======================== 2 devices:
    // stage 0:
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, add -> device 0 has part 0 complete
    //   i = 1, peer = 0, ichunk = 1 -> copy part 1 from device 0, add -> device 1 has part 1 complete
    // second loop
    // stage 0
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1 -> device 0 has parts 0, 1 complete
    //   i = 1, peer = 0, ichunk = 0 -> copy part 0 from device 0 -> device 1 has parts 0, 1 complete
    //
    // ======================== 3 devices
    // stage 0
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, add -> part 0 = 0+1
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, add -> part 1 = 1+2
    //   i = 2, peer = 0, ichunk = 2 -> copy part 2 from device 0, add -> part 2 = 0+2
    // stage 1
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1, add -> part 1 = 0+1+2
    //   i = 1, peer = 2, ichunk = 2 -> copy part 2 from device 2, add -> part 2 = 0+1+2
    //   i = 2, peer = 0, ichunk = 0 -> copy part 0 from device 0, add -> part 0 = 0+1+2
    // second loop
    // stage 0
    //   i = 0, peer = 1, ichunk = 2 -> copy part 2 from device 1, device 0 now has parts 1, 2 complete
    //   i = 1, peer = 2, ichunk = 0 -> copy part 0 from device 2, device 1 now has parts 0, 2 complete
    //   i = 2, peer = 0, ichunk = 1 -> copy part 1 from device 0, device 2 now has parts 0, 1 complete
    // stage 1
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, device 0 now has parts 0, 1, 2, complete
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, device 1 now has parts 0, 1, 2, complete
    //   i = 2, peer = 0, ichunk = 2 -> copy part 2 from device 0, device 2 now has parts 0, 1, 2, complete
    //
    // ======================== 4 devices
    // stage 0
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, add -> part 0 = 0+1
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, add -> part 1 = 1+2
    //   i = 2, peer = 3, ichunk = 2 -> copy part 2 from device 3, add -> part 2 = 2+3
    //   i = 3, peer = 0, ichunk = 3 -> copy part 3 from device 0, add -> part 3 = 0+3
    // stage 1
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1, add -> part 1 = 0+1+2
    //   i = 1, peer = 2, ichunk = 2 -> copy part 2 from device 2, add -> part 2 = 1+2+3
    //   i = 2, peer = 3, ichunk = 3 -> copy part 3 from device 3, add -> part 3 = 0+2+3
    //   i = 3, peer = 0, ichunk = 0 -> copy part 0 from device 0, add -> part 0 = 0+1+3
    // stage 2
    //   i = 0, peer = 1, ichunk = 2 -> copy part 2 from device 1, add -> part 2 = 0+1+2+3
    //   i = 1, peer = 2, ichunk = 3 -> copy part 3 from device 2, add -> part 3 = 0+1+2+3
    //   i = 2, peer = 3, ichunk = 0 -> copy part 0 from device 3, add -> part 0 = 0+1+2+3
    //   i = 3, peer = 0, ichunk = 1 -> copy part 1 from device 0, add -> part 1 = 0+1+2+3
    // second loop
    // stage 0
    //   i = 0, peer = 1, ichunk = 3 -> copy part 3 from device 1, device 0 now has parts 2, 3
    //   i = 1, peer = 2, ichunk = 0 -> copy part 0 from device 2, device 1 now has parts 3, 0
    //   i = 2, peer = 3, ichunk = 1 -> copy part 1 from device 3, device 2 now has parts 0, 1
    //   i = 3, peer = 0, ichunk = 2 -> copy part 2 from device 0, device 3 now has parts 1, 2
    // stage 1
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, device 0 now has parts 0, 2, 3
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, device 1 now has parts 3, 0, 1
    //   i = 2, peer = 3, ichunk = 2 -> copy part 2 from device 3, device 2 now has parts 0, 1, 2
    //   i = 3, peer = 0, ichunk = 3 -> copy part 3 from device 0, device 3 now has parts 1, 2, 3
    // stage 2
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1, device 0 now has parts 0, 1, 2, 3
    //   etc.
    //
    if (dst->ne[1] >= 32) {
        if (_rdbg) fprintf(stderr, "PXA_RDBG   -> BRANCH ring(ne1>=32)\n");
        _rt.route(PXA_RT_RING);
        auto nelem = ggml_nelements(dst);
        auto tt = ggml_internal_get_type_traits(dst->type);
        GGML_ASSERT(nelem % tt.blck_size == 0);
        auto nblocks = nelem / tt.blck_size;
        auto nblocks_per_device = (nblocks + nhave - 1)/nhave;
        auto nelem_per_device = nblocks_per_device * tt.blck_size;
        auto size_per_device  = nblocks_per_device * tt.type_size;
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            auto this_ctx = info.all_ctx[i];
            if (!this_ctx->copy_event || !this_ctx->compute_event || size_per_device > this_ctx->copy_size) {
                ggml_cuda_set_device(this_ctx->device);
                if (!this_ctx->copy_event) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&this_ctx->copy_event, cudaEventDisableTiming));
                }
                if (!this_ctx->compute_event) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&this_ctx->compute_event, cudaEventDisableTiming));
                }
                if (size_per_device > this_ctx->copy_size) {
                    if (this_ctx->copy_buffer) {
                        CUDA_CHECK(cudaFree(this_ctx->copy_buffer));
                    }
                    CUDA_CHECK(ggml_cuda_device_malloc(&this_ctx->copy_buffer, size_per_device, this_ctx->device));
                    this_ctx->copy_size = size_per_device;
                }
            }
        }
        for (int stage = 0; stage < nhave-1; ++stage) {
            int ichunk = stage;
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                int peer = idx[(ii+1)%nhave];
                auto this_nelem = std::min(nelem_per_device, nelem - ichunk*nelem_per_device);
                auto this_size  = (this_nelem / tt.blck_size) * tt.type_size;
                ggml_cuda_set_device(info.all_ctx[peer]->device);
                if (stage > 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[peer]->stream(), info.all_ctx[i]->compute_event, 0));
                }
                CUDA_CHECK(cudaMemcpyPeerAsync(info.all_ctx[i]->copy_buffer, info.all_ctx[i]->device,
                            (const char *)dst->src[peer]->data + ichunk*size_per_device, info.all_ctx[peer]->device,
                            this_size, info.all_ctx[peer]->stream()));
                CUDA_CHECK(cudaEventRecord(info.all_ctx[peer]->copy_event, info.all_ctx[peer]->stream()));
                ichunk = (ichunk + 1)%nhave;
            }
            ichunk = stage;
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                int peer = idx[(ii+1)%nhave];
                auto this_nelem = std::min(nelem_per_device, nelem - ichunk*nelem_per_device);
                ggml_cuda_set_device(info.all_ctx[i]->device);
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[peer]->copy_event, 0));
                int num_blocks = (this_nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
                if (dst->type == GGML_TYPE_F16) {
                    k_add<half, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(this_nelem,
                            (const half *)info.all_ctx[i]->copy_buffer, (half *)dst->src[i]->data + ichunk*nelem_per_device);
                } else if (dst->type == GGML_TYPE_Q8_0) {
                    k_add<CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(this_nelem,
                            (const block_q8_0 *)info.all_ctx[i]->copy_buffer, (block_q8_0 *)dst->src[i]->data + ichunk*nelem_per_device/tt.blck_size);
                } else if (dst->type == GGML_TYPE_BF16) {
                    k_add<nv_bfloat16, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(
                            this_nelem, (const nv_bfloat16 *)info.all_ctx[i]->copy_buffer,
                            (nv_bfloat16 *)dst->src[i]->data + ichunk*nelem_per_device);
                } else {
                    k_add<float, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(this_nelem,
                            (const float *)info.all_ctx[i]->copy_buffer, (float *)dst->src[i]->data + ichunk*nelem_per_device);
                }
                CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->compute_event, info.all_ctx[i]->stream()));
                ichunk = (ichunk + 1)%nhave;
            }
        }
        for (int stage = 0; stage < nhave-1; ++stage) {
            int ichunk = (nhave - 1 + stage)%nhave;
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                int peer = idx[(ii+1)%nhave];
                auto this_nelem = std::min(nelem_per_device, nelem - ichunk*nelem_per_device);
                auto this_size  = (this_nelem / tt.blck_size) * tt.type_size;
                ggml_cuda_set_device(info.all_ctx[peer]->device);
                if (stage == 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[peer]->stream(), info.all_ctx[i]->compute_event, 0));
                }
                CUDA_CHECK(cudaMemcpyPeerAsync((char *)dst->src[i]->data + ichunk*size_per_device, info.all_ctx[i]->device,
                            (const char *)dst->src[peer]->data + ichunk*size_per_device, info.all_ctx[peer]->device,
                            this_size, info.all_ctx[peer]->stream()));
                CUDA_CHECK(cudaEventRecord(info.all_ctx[peer]->copy_event, info.all_ctx[peer]->stream()));
                ichunk = (ichunk + 1)%nhave;
            }
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                int peer = idx[(ii+1)%nhave];
                ggml_cuda_set_device(info.all_ctx[i]->device);
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[peer]->copy_event, 0));
            }
        }
        ggml_cuda_set_device(ctx.device);
        if (ncopy > 0) {
            copy_missing_tensors(ctx, dst, nhave, ncopy, idx, copy_idx);
        }
        return;
    }
    if (false && nhave == 4 && dst->ne[1] <= 8 && ctx.p2p_enabled) {
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            GGML_ASSERT(dst->src[i]->type == dst->type);
            GGML_ASSERT(ggml_are_same_shape(dst, dst->src[i]));
            ggml_cuda_set_device(i);
            if (!info.all_ctx[i]->copy_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[i]->copy_event, cudaEventDisableTiming));
            }
        }
        auto nelem = ggml_nelements(dst);
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+0];
            int nblocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            copy_task task;
            task.nptr = nhave/2;
            task.nelem = nelem;
            task.ptrs[0] = (char *)dst->src[i]->data;
            int j = idx[2*ii+1];
            ggml_cuda_set_device(j);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[j]->copy_event, info.all_ctx[j]->stream()));
            task.ptrs[1] = (char *)dst->src[j]->data;
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            if (dst->type == GGML_TYPE_F16) {
                k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            } else {
                k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            }
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+0];
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+1];
            int nblocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            copy_task task;
            task.nptr = nhave/2;
            task.nelem = nelem;
            task.ptrs[0] = (char *)dst->src[i]->data;
            int j = idx[(2*ii+2)%nhave];
            task.ptrs[1] = (char *)dst->src[j]->data;
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            if (dst->type == GGML_TYPE_F16) {
                k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            } else {
                k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            }
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+1];
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[(2*ii+2)%nhave];
            ggml_cuda_set_device(i);
            int j = idx[2*ii+1];
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
        }
        ggml_cuda_set_device(ctx.device);
        if (ncopy > 0) {
            copy_missing_tensors(ctx, dst, nhave, ncopy, idx, copy_idx);
        }
        return;
    }
    if (dst->ne[1] < 32 && ctx.p2p_enabled) {
        if (_rdbg) fprintf(stderr, "PXA_RDBG   -> BRANCH p2p-direct\n");
        _rt.route(PXA_RT_P2P);
        GGML_ASSERT(dst->type != GGML_TYPE_Q8_0);
        // PXA_REDUCE_CAPTURE: make this cross-device direct-peer reduce CUDA-graph-capturable.
        static const bool _pxa_rc = getenv("PXA_REDUCE_CAPTURE") != nullptr;
        if (_pxa_rc) {
            { // PXA_RC_DIAG: is this reduce running INSIDE a CUDA-graph capture or eager?
                static long _ncap=0, _neager=0;
                cudaStreamCaptureStatus _st = cudaStreamCaptureStatusNone;
                cudaStreamIsCapturing(ctx.stream(), &_st);
                if (_st == cudaStreamCaptureStatusActive) ++_ncap; else ++_neager;
                if (((_ncap+_neager) % 500) == 0)
                    fprintf(stderr, "PXA_RC_DIAG reduce: captured=%ld eager=%ld\n", _ncap, _neager);
            }
            // (1) Pre-create all events EAGERLY -- allocation is forbidden during capture, and
            // the graph gate only permits capture once these exist, so capture never allocates.
            for (int ii = 0; ii < nhave; ++ii) {
                auto c = info.all_ctx[idx[ii]];
                ggml_cuda_set_device(c->device);
                if (!c->copy_event)           CUDA_CHECK(cudaEventCreateWithFlags(&c->copy_event, cudaEventDisableTiming));
                if (!c->reduce_kickoff_event) CUDA_CHECK(cudaEventCreateWithFlags(&c->reduce_kickoff_event, cudaEventDisableTiming));
            }
            // (2) KICKOFF FORK: record a kickoff on the owner (ctx) capture stream and make every
            // peer stream wait on it, so the peer-stream work below is pulled INTO the owner stream
            // capture (otherwise that work is on non-captured streams and is absent on graph replay).
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaEventRecord(ctx.reduce_kickoff_event, ctx.stream()));
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                if (i != ctx.device) {
                    CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), ctx.reduce_kickoff_event, 0));
                }
            }
        }
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            GGML_ASSERT(dst->src[i]->type == dst->type);
            GGML_ASSERT(ggml_are_same_shape(dst, dst->src[i]));
            ggml_cuda_set_device(i);
            if (!info.all_ctx[i]->copy_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[i]->copy_event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        //printf("Recorded events\n");
        auto nelem = ggml_nelements(dst);
        auto nelem8 = (nelem + 7)/8;
        auto nelem_per_device = 8*((nelem8 + nhave - 1)/nhave);
        //auto nelem_per_device = (nelem + nhave - 1)/nhave;
        auto elem_size = ggml_element_size(dst);
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            ggml_cuda_set_device(i);
            int this_nelem = std::min(nelem_per_device, nelem - ii*nelem_per_device);
            copy_task task;
            task.nptr = nhave;
            task.nelem = this_nelem;
            task.ptrs[0] = (char *)dst->src[i]->data + ii*nelem_per_device*elem_size;
            int k = 1;
            for (int jj = 0; jj < nhave; ++jj) {
                if (jj == ii) continue;
                int j = idx[jj];
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
                task.ptrs[k++] = (char *)dst->src[j]->data + ii*nelem_per_device*elem_size;
            }
            int nblock = (this_nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            if (dst->type == GGML_TYPE_F16) {
                switch (nhave) {
                    case 2:
                        k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 3:
                        k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 3><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 4:
                        k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 4><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    default:
                        k_reduce_add<half, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                }
            } else {
                switch (nhave) {
                    case 2:
                        k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 3:
                        k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 3><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 4:
                        k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 4><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    default:
                        k_reduce_add<float, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                }
            }
        }
        //printf("Submitted kernels\n");
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        //printf("Recorded events again\n");
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            ggml_cuda_set_device(i);
            for (int jj = 0; jj < nhave; ++jj) {
                if (jj == ii) continue;
                int j = idx[jj];
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            }
        }
        ggml_cuda_set_device(ctx.device);
        if (ncopy > 0) {
            copy_missing_tensors(ctx, dst, nhave, ncopy, idx, copy_idx);
        }
        return;
    }
    if (_rdbg) fprintf(stderr, "PXA_RDBG   -> BRANCH staged-copy(no-p2p)\n");
    _rt.route(PXA_RT_STAGED);
    auto required_size = nbytes*(nhave-1);
    if (required_size > ctx.copy_size) {
        if (ctx.copy_buffer) {
            CUDA_CHECK(cudaFree(ctx.copy_buffer));
        }
        CUDA_CHECK(ggml_cuda_device_malloc(&ctx.copy_buffer, required_size, ctx.device));
        ctx.copy_size = required_size;
    }
    auto ptr = (char *)ctx.copy_buffer;
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        GGML_ASSERT(dst->src[i]->type == dst->type);
        GGML_ASSERT(ggml_are_same_shape(dst, dst->src[i]));
        if (i == ctx.device) continue;
        ggml_cuda_set_device(i);
        CUDA_CHECK(cudaMemcpyPeerAsync(ptr, ctx.device, dst->src[i]->data, i, nbytes, info.all_ctx[i]->stream()));
        if (!info.all_ctx[i]->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[i]->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        ptr += nbytes;
    }
    auto nelem = ggml_nelements(dst);
    int num_blocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
    ggml_cuda_set_device(ctx.device);
    ptr = (char *)ctx.copy_buffer;
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        if (i == ctx.device) continue;
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), info.all_ctx[i]->copy_event, 0));
        if (dst->type == GGML_TYPE_F16) {
            k_add<half, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, ctx.stream()>>>(nelem, (const half *)ptr, (half *)dst->data);
        } else if (dst->type == GGML_TYPE_BF16) {
            k_add<nv_bfloat16, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, ctx.stream()>>>(nelem,
                    (const nv_bfloat16*)ptr, (nv_bfloat16 *)dst->data);
        } else if (dst->type == GGML_TYPE_Q8_0) {
            k_add<CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, ctx.stream()>>>(nelem, (const block_q8_0 *)ptr,
                    (block_q8_0 *)dst->data);
        } else {
            k_add<float, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, ctx.stream()>>>(nelem, (const float *)ptr, (float *)dst->data);
        }
        ptr += nbytes;
    }
    if (!ctx.copy_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.copy_event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(ctx.copy_event, ctx.stream()));
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        if (i == ctx.device) continue;
        ggml_cuda_set_device(i);
        CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), ctx.copy_event, 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(dst->src[i]->data, i, dst->data, ctx.device, nbytes, info.all_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
    }
    ggml_cuda_set_device(ctx.device);
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        if (i == ctx.device) continue;
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), info.all_ctx[i]->copy_event, 0));
    }
    if (ncopy > 0) {
        copy_missing_tensors(ctx, dst, nhave, ncopy, idx, copy_idx);
    }
}
