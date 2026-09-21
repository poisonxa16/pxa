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
#include <atomic>
#include <mutex>
#include <vector>

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
    static const bool v = [](){
        const char * e = getenv("PXA_REDUCE_PINNED");
        if (e && atoi(e) != 0) return true;
        // PXA_TSPLIT_REDUCE=pinned selects this route as a comparison arm through the one lever.
        const char * t = getenv("PXA_TSPLIT_REDUCE");
        return t && !strcmp(t, "pinned");
    }();
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
// PXA_TSPLIT_REDUCE_v1 (2026-09-20) -- the low-latency small-payload all-reduce
// for the PXA tensor split.
//
// THE PROBLEM, measured in-engine and not on a bench. At batch-1 decode a
// tensor-split all-reduce moves 10-20 KB and costs 47-67 us of GPU span, x130
// per token -- about 30% of a split token. The payload is nothing; the span is
// fixed overhead. The existing p2p-direct route (below) pays, per reduce and
// per pair: 4 cudaEventRecord, 4 cudaStreamWaitEvent, ~9 device switches, two
// kernel launches and TWO cross-device rendezvous -- one forward (wait until
// the peer's partial exists) and one back (wait until the peer has finished
// writing into MY buffer, because that route's kernel writes its reduced chunk
// straight into the peer's tensor).
//
// THE ROUTE. Give every device a small DEVICE-memory staging ring of its own
// and make the reduce one kernel per device, on that device's own stream:
//
//   phase 1  copy my own partial d_i  ->  my staging slot  stage_i[slot]
//   phase 2  publish an arrival token in my own device memory, then spin on
//            every peer's token for this slot (volatile loads over the P2P
//            mapping; this box has NO peer atomics, cuDeviceGetP2PAttribute
//            NATIVE_ATOMIC == 0 on every pair, so a plain volatile store plus
//            __threadfence_system is the only handshake available)
//   phase 3  d_i = d_i + sum_j stage_j[slot], reading the peers' staging slots
//            directly through the P2P mapping, summing in destination
//            precision, writing my own tensor in place.
//
// WHY THAT IS THE WHOLE POINT. Nobody ever writes into a peer's memory, so the
// BACK rendezvous disappears: after phase 3 my next compute segment may read
// d_i immediately, it is already ordered behind my own kernel on my own stream.
// And because the arrival handshake happens inside the kernel, the forward
// rendezvous costs no host API calls at all. Per reduce and per device the host
// issues ONE kernel launch and nothing else: no events, no stream waits, no
// device switches.
//
// WIRE TRAFFIC is unchanged at two devices: each card reads S bytes of peer
// memory (against S/2 read + S/2 written by the p2p-direct route), so each link
// direction still carries exactly S. The extra local copy in phase 1 is a
// 20 KB device-memory write, ~0.04 us at P100 bandwidth.
//
// SLOT SAFETY WITHOUT AN EVENT, the same argument the pinned route uses, moved
// to device memory. Call N writes stage slot s = N % POOL; the same slot is
// written again by call N+POOL. My call N+1 cannot leave its spin until every
// peer has entered ITS call N+1 phase 2, and on the peer's stream that is
// ordered after the peer's call N kernel COMPLETED -- i.e. after the peer
// finished reading my slot s. My call N+2 is ordered behind my call N+1 on my
// own stream, so POOL >= 2 makes the wraparound safe with no cudaEvent. POOL is
// 4, for margin.
//
// BIT-IDENTITY AT TWO DEVICES. Phase 3 computes own + peer in destination
// precision on both cards. The p2p-direct route computes own + peer for the
// half-chunk it owns and receives the other half computed as own + peer on the
// peer. IEEE addition is commutative, so the two routes produce bit-identical
// bytes on both cards, and both are bit-identical to a single-device sum.
// At THREE OR MORE devices this route sums every element in one device-rank
// order while p2p-direct sums each chunk starting from that chunk's owner, so
// the orders differ and floating-point associativity does not save us. The
// predicate therefore refuses nhave > 2 unless PXA_TSPLIT_REDUCE_NWAY=1 is set,
// and the banner says so when it is.
//
// PASCAL. The pinned route needs __nanosleep and is gated at cc >= 7.0. This
// one backs off with clock64() below Volta, so it runs on sm_60 -- which is the
// point, because the four-card Pascal island is where a tensor split has the
// most to gain and where no other engine's tensor split boots at all.
//
// SAFETY VALVE. The spin is bounded. On expiry the kernel poisons its output
// with NaN and sets a mapped host flag, and the host turns that into an abort at
// the next reduce. A wrong number that announces itself beats a wedged card.
// ---------------------------------------------------------------------------

#define PXA_TS_POOL          4                   // staging slots per device
#define PXA_TS_BLOCKS        8
#define PXA_TS_THREADS       256
#define PXA_TS_FLAG_STRIDE   64                  // bytes: one line per (slot, block)
#define PXA_TS_SPIN_MAX      100000000ll

enum pxa_ts_route { PXA_TS_OFF = 0, PXA_TS_FUSED, PXA_TS_P2P, PXA_TS_PINNED, PXA_TS_NCCL };

static int pxa_tsplit_route() {
    static const int v = [](){
        const char * e = getenv("PXA_TSPLIT_REDUCE");
        if (!e || !*e)                 return (int)PXA_TS_OFF;
        if (!strcmp(e, "fused"))       return (int)PXA_TS_FUSED;
        if (!strcmp(e, "p2p"))         return (int)PXA_TS_P2P;
        if (!strcmp(e, "pinned"))      return (int)PXA_TS_PINNED;
        if (!strcmp(e, "nccl"))        return (int)PXA_TS_NCCL;
        if (!strcmp(e, "off") || !strcmp(e, "0")) return (int)PXA_TS_OFF;
        fprintf(stderr, "PXA_TSPLIT_REDUCE=%s is not one of fused|p2p|pinned|nccl|off -- refusing to guess\n", e);
        GGML_ABORT("PXA_TSPLIT_REDUCE: unknown route");
        return (int)PXA_TS_OFF;
    }();
    return v;
}

static int  pxa_ts_env_int(const char * name, int dflt) {
    const char * e = getenv(name);
    return (e && *e) ? atoi(e) : dflt;
}

static bool pxa_ts_nway()     { static const bool v = pxa_ts_env_int("PXA_TSPLIT_REDUCE_NWAY", 0)    != 0; return v; }
static bool pxa_ts_prefill()  { static const bool v = pxa_ts_env_int("PXA_TSPLIT_REDUCE_PREFILL",0) != 0; return v; }
// Default OFF: this one widens the route predicate of a path that ships, so with every lever unset
// the shipping selection must stay exactly what it was. Turn it on for a process that holds cards
// from more than one peer island (see the note at the p2p-direct branch).
static bool pxa_ts_pairmap()  { static const bool v = pxa_ts_env_int("PXA_TSPLIT_P2P_PAIR", 0)      != 0; return v; }
static bool pxa_ts_pull()     { static const bool v = pxa_ts_env_int("PXA_TSPLIT_REDUCE_PULL", 0)   != 0; return v; }

// The arrival token and the ring slot are KERNEL ARGUMENTS, so a captured graph would replay a
// reduce with the token and the slot it was captured with: the poll would either pass immediately
// on a flag the capture itself left behind and sum stale staging, or never match and spin out.
// Refuse to be captured instead.
//
// The query is UNCONDITIONAL. Graph use is ON by default in this backend (both the context flag and
// the cuda-params default are true; only GGML_CUDA_DISABLE_GRAPHS turns it off), and what keeps a
// reduce out of a capture today is the graph admission test, which drops graph use for any graph
// containing a REDUCE node unless it is told otherwise. That test can be relaxed behind any lever
// name, or through the `graphs=` cuda-params string with no environment variable at all, so a guard
// that only looked at a fixed list of environment names would silently stop guarding at exactly the
// moment it was needed. One capture-status query per half is cheap against a route whose GPU span
// is ~20 us.
static void pxa_ts_refuse_capture(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) { (void)cudaGetLastError(); return; }
    if (st == cudaStreamCaptureStatusActive) {
        GGML_ABORT("PXA_TSPLIT_REDUCE: this reduce is being CUDA-graph captured, but its arrival token "
                   "and ring slot are kernel arguments, so a replay would read a stale rendezvous. "
                   "Either keep the graph off for this graph, or move the generation counter into "
                   "device memory and pin the slot per graph position first");
    }
}

// ---- the device side -------------------------------------------------------

// PUSH vs PULL, and why the default is PUSH.
//
// The handshake needs my data to be visible to a peer before my token is. In the PULL shape my
// data and my token both sit in MY memory and the peer reads both over the peer mapping: a
// __threadfence_system on my side orders them, and the peer's two reads walk the same path, so it
// is sound -- but every payload byte crosses the link as a READ, and a PCIe read is a round trip
// with a limited number of outstanding requests.
//
// In the PUSH shape I write my payload INTO the peer's memory and then write my token into the
// peer's memory too. Both are posted writes to the same completer, and PCIe does not let a posted
// write pass another posted write, so the token cannot arrive before the data it announces. The
// peer then polls a flag in its OWN memory -- a local read, not a link round trip -- and reads its
// payload locally as well. Every byte that crosses the link is a posted write, and nothing spins on
// the link at all.
//
// One kernel serves both: only the pointer wiring differs.
struct pxa_ts_task {
    void *       wdata[GGML_CUDA_MAX_DEVICES];   // where my payload goes   (nwrite entries)
    int  *       wflag[GGML_CUDA_MAX_DEVICES];   // where my token goes     (nwrite entries)
    const void * rdata[GGML_CUDA_MAX_DEVICES];   // where peer r's payload is, indexed by RANK
    const int  * rflag[GGML_CUDA_MAX_DEVICES];   // where peer r's token is,   indexed by RANK
    int          nwrite = 0;
    int          nrank  = 0;
    int          rank   = 0;
    int          nelem  = 0;
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

template <typename T, int NRANK>
static __global__ void k_pxa_tsplit_fused(
        const T *              sendbuf,     // NOT __restrict__: recvbuf is the same buffer
        T       *              recvbuf,
        pxa_ts_task            task,
        int                    token,
        int     * __restrict__ err_flag) {

    constexpr int VEC      = 16 / sizeof(T);
    constexpr int ARR_INTS = PXA_TS_FLAG_STRIDE / sizeof(int);

    const int tid   = threadIdx.x;
    const int bid   = blockIdx.x;
    const int gtid  = bid * blockDim.x + tid;
    const int gnt   = gridDim.x * blockDim.x;
    const int count = task.nelem;
    const int nvec  = count / VEC;
    const int tail  = nvec * VEC;

    __shared__ int s_bad;
    if (tid == 0) { s_bad = 0; }
    __syncthreads();

    // Phase 1 -- publish this device's partial where its consumers will read it. Every block writes
    // exactly the elements it will later read back, so the per-block flag below covers precisely
    // the bytes that block depends on and the blocks stay independent of each other.
    #pragma unroll 1
    for (int w = 0; w < task.nwrite; ++w) {
        T * dstb = (T *)task.wdata[w];
        for (int i = gtid; i < nvec; i += gnt) {
            const int off = i * VEC;
            *(int4 *)(dstb + off) = *(const int4 *)(sendbuf + off);
        }
        if (bid == 0 && tid < count - tail) {
            dstb[tail + tid] = sendbuf[tail + tid];
        }
    }

    __threadfence_system();   // the payload commits before the token that announces it
    __syncthreads();

    // Phase 2 -- one arrival line per block.
    if (tid == 0) {
        #pragma unroll 1
        for (int w = 0; w < task.nwrite; ++w) {
            *(volatile int *)(task.wflag[w] + bid * ARR_INTS) = token;
        }
        __threadfence_system();
        long long spins = 0;
        bool bad = false;
        #pragma unroll 1
        for (int r = 0; r < NRANK && !bad; ++r) {
            if (r == task.rank) continue;
            const volatile int * other = (const volatile int *)(task.rflag[r] + bid * ARR_INTS);
            while (*other != token) {
                pxa_ts_pause();
                if (++spins > PXA_TS_SPIN_MAX) { bad = true; break; }
            }
        }
        if (bad) { *(volatile int *)err_flag = 1; s_bad = 1; }
    }
    __syncthreads();
    __threadfence_system();   // acquire whatever the peers published

    // Phase 3 -- own first, then the peers in rank order, in destination precision, in place into
    // our own tensor. Nobody else reads this buffer, so there is no back edge to wait for.
    const bool bad = s_bad != 0;
    for (int i = gtid; i < nvec; i += gnt) {
        const int off = i * VEC;
        T acc[VEC];
        *(int4 *)acc = *(const int4 *)(sendbuf + off);
        #pragma unroll
        for (int r = 0; r < NRANK; ++r) {
            if (r == task.rank) continue;
            T o[VEC];
            *(int4 *)o = *(const int4 *)((const T *)task.rdata[r] + off);
            #pragma unroll
            for (int k = 0; k < VEC; ++k) acc[k] += o[k];
        }
        #pragma unroll
        for (int k = 0; k < VEC; ++k) recvbuf[off + k] = bad ? pxa_ts_poison<T>() : acc[k];
    }
    if (bid == 0 && tid < count - tail) {
        T acc = sendbuf[tail + tid];
        #pragma unroll
        for (int r = 0; r < NRANK; ++r) {
            if (r == task.rank) continue;
            acc += ((const T *)task.rdata[r])[tail + tid];
        }
        recvbuf[tail + tid] = bad ? pxa_ts_poison<T>() : acc;
    }
}

// ---- the host side ---------------------------------------------------------

struct pxa_ts_dev {
    int       dev   = -1;
    uint8_t * stage = nullptr;   // [PXA_TS_POOL][nrank] payload areas of buf_bytes, on `dev`
    int     * flag  = nullptr;   // [PXA_TS_POOL][nrank][blocks][PXA_TS_FLAG_STRIDE/4] arrival lines
};

struct pxa_ts_pipeline {
    int         ndev      = 0;
    int         dev[GGML_CUDA_MAX_DEVICES] = {};
    int         rank_of[GGML_CUDA_MAX_DEVICES];          // device id -> rank, -1 if not in the group
    pxa_ts_dev  d[GGML_CUDA_MAX_DEVICES];
    size_t      buf_bytes = 0;
    long long   calls     = 0;
    int         blocks    = PXA_TS_BLOCKS;
    int         threads   = PXA_TS_THREADS;
    uint8_t *   err_host  = nullptr;
    uint8_t *   err_dev   = nullptr;
};

// The seam's publication record: (pointer, event, generation) per device per reduce index. The
// per-device workers that will replace the driver below publish into it from whichever thread owns
// the device; a half may then spin, lock free, until every peer's generation for THIS reduce index
// has been published. The single-thread driver in this file publishes all N before issuing any
// half, so its spin always completes on the first read -- the machinery is here so the seam does
// not move when the workers land.
//
// The reduce index is a process-wide sequence and is 64-bit end to end: a slot is taken modulo the
// ring size as UNSIGNED, so a counter past 2^31 can never produce a negative subscript (at ~130
// reduces per token a long-lived server reaches 2^31 in days).
#define PXA_TS_PUB_SLOTS 1024
struct pxa_ts_pub {
    std::atomic<unsigned long long> gen;
    void *      ptr;
    cudaEvent_t ev;
    int         nelem;
};
static pxa_ts_pub g_pxa_ts_pub[GGML_CUDA_MAX_DEVICES][PXA_TS_PUB_SLOTS];

static inline int pxa_ts_pub_slot(long long reduce_index) {
    return (int)((unsigned long long)reduce_index % (unsigned long long)PXA_TS_PUB_SLOTS);
}

extern "C" void pxa_tsplit_reduce_publish(int dev_i, long long reduce_index, void * ptr, cudaEvent_t ev,
                                          int nelem, unsigned long long gen) {
    auto & p = g_pxa_ts_pub[dev_i][pxa_ts_pub_slot(reduce_index)];
    p.ptr   = ptr;
    p.ev    = ev;
    p.nelem = nelem;
    p.gen.store(gen, std::memory_order_release);
}

static bool pxa_ts_await_publication(long long reduce_index, int n_dev, const int * devs,
                                     unsigned long long gen) {
    const int slot = pxa_ts_pub_slot(reduce_index);
    for (int k = 0; k < n_dev; ++k) {
        int tries = 0;
        while (g_pxa_ts_pub[devs[k]][slot].gen.load(std::memory_order_acquire) < gen) {
            if (++tries > 100000000) return false;      // ~seconds; a lost publication, not a slow one
        }
    }
    return true;
}

// A pipeline belongs to ONE device group, because its staging ring, arrival lines and call counter
// are laid out for that group's ranks. One graph legitimately contains reduces over DIFFERENT
// groups -- on a four-card split the delta-net and FFN reduces take all four devices while a
// standard-attention reduce takes only the devices its heads landed on -- so the pipelines are kept
// in a small table keyed by the (ascending) device list and a new group builds its own.
//
// The table is built on first use, and first use is reached from pxa_tsplit_reduce_half(), which
// the seam lets N host threads enter at once (one per device). Two threads racing the build would
// each get their own staging ring, and then no device would ever see a peer's arrival token.
// Double-checked: an acquire load of the published count on the hot path, the build under a mutex.
#define PXA_TS_MAX_GROUPS 8
static std::atomic<int>   g_pxa_ts_ngroup{0};
static pxa_ts_pipeline *  g_pxa_ts_pipe[PXA_TS_MAX_GROUPS];
static std::mutex         g_pxa_ts_pipe_mutex;

static inline bool pxa_ts_same_group(const pxa_ts_pipeline * p, const int * idx, int nhave) {
    if (p->ndev != nhave) return false;
    for (int i = 0; i < nhave; ++i) if (p->dev[i] != idx[i]) return false;
    return true;
}

static pxa_ts_pipeline * pxa_ts_get(const int * idx, int nhave, size_t need_bytes) {
    // Every failure below aborts, so a built pipeline is the only outcome a second caller can
    // observe: a group's slow path is taken exactly once, under the mutex, and everyone else takes
    // the acquire load. (There is no "tried and failed" state to remember.)
    {
        const int n = g_pxa_ts_ngroup.load(std::memory_order_acquire);
        for (int k = 0; k < n; ++k) {
            if (pxa_ts_same_group(g_pxa_ts_pipe[k], idx, nhave)) return g_pxa_ts_pipe[k];
        }
    }
    std::lock_guard<std::mutex> lock(g_pxa_ts_pipe_mutex);
    const int have = g_pxa_ts_ngroup.load(std::memory_order_relaxed);
    for (int k = 0; k < have; ++k) {
        if (pxa_ts_same_group(g_pxa_ts_pipe[k], idx, nhave)) return g_pxa_ts_pipe[k];
    }
    if (have >= PXA_TS_MAX_GROUPS) {
        GGML_ABORT("PXA_TSPLIT_REDUCE: more than %d distinct device groups in one process", PXA_TS_MAX_GROUPS);
    }

    auto & info = ggml_cuda_info();
    if (!pxa_tsplit_p2p_group(idx, nhave)) {
        fprintf(stderr, "PXA_TSPLIT_REDUCE: the requested group does not peer end to end -- refusing.\n");
        pxa_tsplit_p2p_dump("PXA_TSPLIT_REDUCE");
        GGML_ABORT("PXA_TSPLIT_REDUCE: a half was asked to cross a non-peering pair; there is no silent "
                   "fallback to a blocking copy from here");
    }

    auto * q = new pxa_ts_pipeline();
    q->ndev = nhave;
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) q->rank_of[i] = -1;
    for (int i = 0; i < nhave; ++i) { q->dev[i] = idx[i]; q->rank_of[idx[i]] = i; }
    q->blocks    = pxa_ts_env_int("PXA_TSPLIT_REDUCE_BLOCKS",  PXA_TS_BLOCKS);
    q->threads   = pxa_ts_env_int("PXA_TSPLIT_REDUCE_THREADS", PXA_TS_THREADS);
    const int want_mb_i = pxa_ts_env_int("PXA_TSPLIT_REDUCE_BUF_MB", 1);
    // A geometry knob that is out of range does not degrade gracefully here: zero blocks or zero
    // threads is a grid of no threads, so every device would publish nothing and then spin on a
    // token that is never written. Say so instead of hanging.
    if (q->blocks < 1 || q->blocks > 1024 || q->threads < 32 || q->threads > 1024 || (q->threads % 32) != 0 ||
        want_mb_i < 1 || want_mb_i > 512) {
        GGML_ABORT("PXA_TSPLIT_REDUCE: geometry out of range (blocks=%d must be 1..1024, threads=%d must be "
                   "32..1024 and a multiple of 32, buf_mb=%d must be 1..512)", q->blocks, q->threads, want_mb_i);
    }
    const size_t want_mb = (size_t)want_mb_i;
    q->buf_bytes = std::max(need_bytes, want_mb * 1024u * 1024u);
    q->buf_bytes = (q->buf_bytes + 255) & ~(size_t)255;

    // One payload area and one arrival line per SOURCE RANK, so a push lands in a place only its
    // sender writes; the pull shape uses rank slot 0 of the same allocation.
    const size_t stage_bytes = (size_t)PXA_TS_POOL * nhave * q->buf_bytes;
    const size_t flag_bytes  = (size_t)PXA_TS_POOL * nhave * q->blocks * PXA_TS_FLAG_STRIDE;

    int cur_dev = 0;
    cudaGetDevice(&cur_dev);
    bool ok = true;
    for (int i = 0; i < nhave && ok; ++i) {
        const int d = idx[i];
        q->d[i].dev = d;
        ggml_cuda_set_device(d);
        ok = ok && ggml_cuda_device_malloc((void **)&q->d[i].stage, stage_bytes, d) == cudaSuccess;
        ok = ok && ggml_cuda_device_malloc((void **)&q->d[i].flag,  flag_bytes,  d) == cudaSuccess;
        // The arrival lines must read ZERO before ANY half of this group is enqueued, on any
        // device: a peer's very first kernel pushes its token straight into these lines, and a
        // zeroing that landed afterwards would erase it and leave this device spinning to its
        // timeout. The compute streams are non-blocking, so the legacy null stream orders nothing
        // against them -- zero on the device's own compute stream and drain it below, inside the
        // build, which no half of this group can overtake because the build holds the mutex.
        if (ok) ok = cudaMemsetAsync(q->d[i].flag, 0, flag_bytes, info.all_ctx[d]->stream()) == cudaSuccess;
    }
    for (int i = 0; i < nhave && ok; ++i) {
        const int d = idx[i];
        ggml_cuda_set_device(d);
        ok = ok && cudaStreamSynchronize(info.all_ctx[d]->stream()) == cudaSuccess;
    }
    // One mapped host word, shared by every device, for the spin-timeout poison flag.
    ggml_cuda_set_device(idx[0]);
    ok = ok && cudaHostAlloc((void **)&q->err_host, sizeof(int),
                             cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostGetDevicePointer((void **)&q->err_dev, q->err_host, 0) == cudaSuccess;
    if (ok) memset(q->err_host, 0, sizeof(int));
    ggml_cuda_set_device(cur_dev);

    if (!ok) {
        fprintf(stderr, "PXA_TSPLIT_REDUCE: allocation failed -- refusing (no silent fallback).\n");
        GGML_ABORT("PXA_TSPLIT_REDUCE: pipeline allocation failed");
    }

    char devs[128]; int off = 0;
    for (int i = 0; i < nhave; ++i) off += snprintf(devs + off, sizeof(devs) - off, "%s%d", i ? "," : "", idx[i]);
    fprintf(stderr,
        "PXA_TSPLIT: reduce=fused dir=%s devices=%s cc=%d spin=%s staging=%dx%dx%zuKB/device "
        "blocks=%d threads=%d nway=%s prefill=%s pairmap=%s bit-identity=%s\n",
        pxa_ts_pull() ? "pull(peer-read)" : "push(peer-write,local-poll)",
        devs, info.devices[idx[0]].cc,
        info.devices[idx[0]].cc >= CC_VOLTA ? "nanosleep" : "clock64",
        PXA_TS_POOL, nhave, (size_t)(q->buf_bytes >> 10), q->blocks, q->threads,
        pxa_ts_nway() ? "on" : "off", pxa_ts_prefill() ? "on" : "off", pxa_ts_pairmap() ? "on" : "off",
        nhave == 2 ? "exact-vs-p2p-direct" : "ORDER DIFFERS from p2p-direct (n>2)");
    g_pxa_ts_pipe[have] = q;
    g_pxa_ts_ngroup.store(have + 1, std::memory_order_release);
    return q;
}

// The spin-timeout fault, read from the host. A half whose arrival spin ran out wrote NaN instead
// of a sum and set this flag; this is the only thing that turns it into a stop. It is checked at
// the entry of the next half AND at the token join (the backend synchronize), because the host runs
// far ahead of the device: without the join check the last reduce of a run -- the final decode step,
// or the step before a server goes idle -- would put a NaN into the logits and exit cleanly.
void pxa_tsplit_reduce_check_fault(void) {
    const int n = g_pxa_ts_ngroup.load(std::memory_order_acquire);
    for (int k = 0; k < n; ++k) {
        const pxa_ts_pipeline * p = g_pxa_ts_pipe[k];
        if (p->err_host && *(volatile int *)p->err_host) {
            fprintf(stderr, "PXA_TSPLIT_REDUCE: a cross-device arrival spin timed out (>%lld polls). The reduce "
                            "that timed out wrote NaN rather than a number, so nothing downstream is trustworthy.\n",
                    (long long)PXA_TS_SPIN_MAX);
            GGML_ABORT("PXA_TSPLIT_REDUCE: cross-device arrival spin timed out");
        }
    }
}

// The one place eligibility is decided. If this says yes the route MUST run: the caller aborts
// rather than falling through, exactly as the pinned route does, because a "yes" here is the same
// promise the scheduler's drain-skip is allowed to trust.
static bool pxa_tsplit_reduce_handles(const struct ggml_tensor * dst) {
    const int _r = pxa_tsplit_route();
    if (_r != PXA_TS_FUSED && _r != PXA_TS_P2P)             return false;
    if (!dst || dst->op != GGML_OP_REDUCE)                  return false;
    if ((ggml_op)dst->op_params[0] != GGML_OP_ADD)          return false;
    if (dst->op_params[3] == 1)                             return false;  // reduce-OFF container
    if (dst->op_params[4] != 0)                             return false;  // a device's partial is late-filled
    // Q8_0 reduce is a REQUANTISING add (k_add above recomputes the block scale): it is not a plain
    // element-wise sum, so it cannot be expressed as stage-then-add. Refuse it, loudly, rather than
    // quietly producing a different quantisation.
    if (dst->type == GGML_TYPE_Q8_0)                        return false;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) return false;
    if (!ggml_is_contiguous(dst))                           return false;
    const int nhave = dst->op_params[2];
    if (nhave < 2)                                          return false;
    if (_r == PXA_TS_FUSED && nhave > 2 && !pxa_ts_nway())  return false;  // see the bit-identity note
    if (dst->ne[1] >= 32 && !pxa_ts_prefill())              return false;  // prefill keeps the ring
    // The kernel treats every participant as one flat, contiguous, 16-byte-aligned run of `nelem`
    // elements and moves it with int4 loads and stores. Nothing upstream promises that, so check it
    // here rather than discover it as a misaligned access or a short copy. A "no" at this point is
    // free: no promise has been made yet and the stock routes below take the reduce.
    if (((uintptr_t)dst->data & 15) != 0)                   return false;
    for (int i = 0; i < dst->op_params[1]; ++i) {
        if (!dst->src[i]) continue;
        if (dst->src[i]->type != dst->type)                 return false;
        if (!ggml_are_same_shape(dst, dst->src[i]))         return false;
        if (!ggml_is_contiguous(dst->src[i]))               return false;
        if (((uintptr_t)dst->src[i]->data & 15) != 0)       return false;
    }
    return true;
}

// The seam's per-device HALF. Enqueued only onto device dev_i's compute stream -- the stream of the
// context passed in, which is the context that produced this device's partial -- by whichever host
// thread owns dev_i. It never synchronises the host on the GPU and never blocks; the cross-device
// ordering is the in-kernel arrival token.
extern "C" void pxa_tsplit_reduce_half(ggml_backend_cuda_context & ctx_i, ggml_tensor * dst,
                                       long long reduce_index, int n_dev, int dev_i,
                                       const int * devs, int slot, int token) {
    auto * p = pxa_ts_get(devs, n_dev, ggml_nbytes(dst));
    GGML_ASSERT(p && "PXA_TSPLIT_REDUCE: no pipeline for this group");

    // The fault path. A half that spun out wrote poison instead of a sum, so the first thing the
    // next half does is look. This sits in the half rather than in the single-thread driver so that
    // it still fires when per-device workers call the half directly.
    pxa_tsplit_reduce_check_fault();

    const int rank = p->rank_of[dev_i];
    GGML_ASSERT(rank >= 0);

    const size_t type_size  = ggml_type_size(dst->type);
    const int64_t nelem     = ggml_nelements(dst);
    const int64_t max_chunk = (int64_t)(p->buf_bytes / type_size);
    // A zero-sized slice is a first-class case: it still has to take part in the handshake, so it
    // runs exactly one kernel with nelem == 0 instead of none.
    const int     nchunk    = nelem > 0 ? (int)((nelem + max_chunk - 1) / max_chunk) : 1;

    pxa_ts_task task;
    task.nrank = n_dev;
    task.rank  = rank;

    const bool  pull        = pxa_ts_pull();
    const int   flag_ints   = PXA_TS_FLAG_STRIDE / (int)sizeof(int);
    const size_t flag_stride_rank = (size_t)p->blocks * flag_ints;              // one rank's lines
    const size_t flag_stride_slot = (size_t)n_dev * flag_stride_rank;           // one slot's lines

    // The caller's own stream, not a globally registered one: this half stages dst->src[dev_i]->data
    // and nothing orders it against the kernels that produced it except being on the same stream.
    GGML_ASSERT(ctx_i.device == dev_i && "PXA_TSPLIT_REDUCE: a half was given another device's context");
    ggml_cuda_set_device(dev_i);
    cudaStream_t stream = ctx_i.stream();
    pxa_ts_refuse_capture(stream);

    for (int c = 0; c < nchunk; ++c) {
        const int64_t start = (int64_t)c * max_chunk;
        const int64_t chunk = nelem > 0 ? std::min(max_chunk, nelem - start) : 0;
        const int this_slot  = (slot + c) % PXA_TS_POOL;
        const int this_token = token + c;

        // area(dev, slot, src_rank) -- the payload written by src_rank, living on `dev`
        auto area = [&](int d_rank, int src_rank) {
            return p->d[d_rank].stage + ((size_t)this_slot * n_dev + src_rank) * p->buf_bytes;
        };
        auto line = [&](int d_rank, int src_rank) {
            return p->d[d_rank].flag + (size_t)this_slot * flag_stride_slot + (size_t)src_rank * flag_stride_rank;
        };

        if (pull) {
            // I publish once, into my own area; every peer reads it from me.
            task.nwrite   = 1;
            task.wdata[0] = area(rank, 0);
            task.wflag[0] = line(rank, 0);
            for (int r = 0; r < n_dev; ++r) {
                task.rdata[r] = area(r, 0);
                task.rflag[r] = line(r, 0);
            }
        } else {
            // I push a copy into every peer's area reserved for MY rank, and poll my own memory.
            int w = 0;
            for (int r = 0; r < n_dev; ++r) {
                if (r == rank) continue;
                task.wdata[w] = area(r, rank);
                task.wflag[w] = line(r, rank);
                ++w;
            }
            task.nwrite = w;
            for (int r = 0; r < n_dev; ++r) {
                task.rdata[r] = area(rank, r);
                task.rflag[r] = line(rank, r);
            }
        }
        task.nelem = (int)chunk;

        char * data = (char *)dst->src[dev_i]->data + (size_t)start * type_size;

        if (dst->type == GGML_TYPE_F16) {
            switch (n_dev) {
                case 2: k_pxa_tsplit_fused<half, 2><<<p->blocks, p->threads, 0, stream>>>(
                            (const half *)data, (half *)data, task, this_token, (int *)p->err_dev); break;
                case 3: k_pxa_tsplit_fused<half, 3><<<p->blocks, p->threads, 0, stream>>>(
                            (const half *)data, (half *)data, task, this_token, (int *)p->err_dev); break;
                case 4: k_pxa_tsplit_fused<half, 4><<<p->blocks, p->threads, 0, stream>>>(
                            (const half *)data, (half *)data, task, this_token, (int *)p->err_dev); break;
                default: GGML_ABORT("PXA_TSPLIT_REDUCE: unsupported device count %d", n_dev);
            }
        } else {
            switch (n_dev) {
                case 2: k_pxa_tsplit_fused<float, 2><<<p->blocks, p->threads, 0, stream>>>(
                            (const float *)data, (float *)data, task, this_token, (int *)p->err_dev); break;
                case 3: k_pxa_tsplit_fused<float, 3><<<p->blocks, p->threads, 0, stream>>>(
                            (const float *)data, (float *)data, task, this_token, (int *)p->err_dev); break;
                case 4: k_pxa_tsplit_fused<float, 4><<<p->blocks, p->threads, 0, stream>>>(
                            (const float *)data, (float *)data, task, this_token, (int *)p->err_dev); break;
                default: GGML_ABORT("PXA_TSPLIT_REDUCE: unsupported device count %d", n_dev);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    GGML_UNUSED(reduce_index);
}

// The SINGLE-THREAD DRIVER. Calls half 0..N-1 in order so the new route is measurable under
// today's sequential scheduler, before the per-device workers exist. Those workers replace this
// loop with N threads calling pxa_tsplit_reduce_half() concurrently; nothing else changes.
static bool pxa_tsplit_reduce_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const int * idx, int nhave) {
    auto & info = ggml_cuda_info();
    auto * p = pxa_ts_get(idx, nhave, ggml_nbytes(dst));
    if (!p) return false;

    const int64_t nelem     = ggml_nelements(dst);
    const size_t  type_size = ggml_type_size(dst->type);
    const int64_t max_chunk = (int64_t)(p->buf_bytes / type_size);
    const int     nchunk    = (int)((nelem + max_chunk - 1) / max_chunk);

    // Slot and arrival token come from this GROUP's own call counter, because the staging ring they
    // address belongs to the group. The publication record is shared by every group, so its index
    // and generation come from a process-wide sequence instead: two groups that share a device must
    // not write each other's slot, and a generation must never go backwards on a device.
    const long long call = p->calls;
    p->calls += std::max(nchunk, 1);
    const int slot  = (int)(call % PXA_TS_POOL);
    const int token = (int)(call + 1);

    static std::atomic<unsigned long long> seq_next{0};
    const unsigned long long gen = seq_next.fetch_add(1, std::memory_order_relaxed) + 1;
    const long long          seq = (long long)gen;

    // Publish first, then issue every half. The publication is what the per-device workers spin on.
    for (int ii = 0; ii < nhave; ++ii) {
        const int d = idx[ii];
        pxa_tsplit_reduce_publish(d, seq, dst->src[d]->data, nullptr, (int)nelem, gen);
    }
    if (!pxa_ts_await_publication(seq, nhave, idx, gen)) {
        GGML_ABORT("PXA_TSPLIT_REDUCE: a peer never published its partial for this reduce index");
    }

    for (int ii = 0; ii < nhave; ++ii) {
        pxa_tsplit_reduce_half(*info.all_ctx[idx[ii]], dst, seq, nhave, idx[ii], idx, slot, token);
    }
    ggml_cuda_set_device(ctx.device);
    return true;
}


// ---------------------------------------------------------------------------
// PXA_TSPLIT_REDUCE=p2p -- the comparison arm that keeps today's mesh ALGORITHM
// and removes only the host overhead around it, so the two halves of the bill can
// be told apart. Same kernel, same chunking, same summation order, therefore the
// same bytes out. What changes:
//   * every event is created once at pipeline build, never lazily in the hot path;
//   * PER-PAIR events instead of one shared copy_event per device, so two
//     independent reduces can be in flight without serialising on one object
//     (today every reduce on a device re-records the same copy_event);
//   * the current device is cached in this translation unit, so the ~9
//     cudaGetDevice/cudaSetDevice calls per reduce become at most two.
// The back-edge rendezvous stays, because this route's kernel writes its reduced
// chunk into the peer's tensor. Removing the back edge is what the fused route
// is for; this arm exists to price the difference.
// ---------------------------------------------------------------------------

#define PXA_TS_EV_RING 64
struct pxa_ts_ev {
    // ev[gen % RING][phase]: recorded on this device's stream. A ring, not one shared object, so
    // consecutive reduces never serialise on the same event -- which is what one copy_event per
    // device does today.
    cudaEvent_t ev[PXA_TS_EV_RING][2] = {};
    bool made = false;
};
static long long g_pxa_ts_p2p_calls = 0;
static pxa_ts_ev g_pxa_ts_ev[GGML_CUDA_MAX_DEVICES];
static int       g_pxa_ts_cur_dev = -1;

static inline void pxa_ts_set_device(int d) {
    if (g_pxa_ts_cur_dev == d) return;
    CUDA_CHECK(cudaSetDevice(d));
    g_pxa_ts_cur_dev = d;
}

static void pxa_ts_make_events(const int * idx, int nhave) {
    for (int ii = 0; ii < nhave; ++ii) {
        const int i = idx[ii];
        if (g_pxa_ts_ev[i].made) continue;
        ggml_cuda_set_device(i);
        for (int ph = 0; ph < 2; ++ph) {
            for (int k = 0; k < PXA_TS_EV_RING; ++k) {
                CUDA_CHECK(cudaEventCreateWithFlags(&g_pxa_ts_ev[i].ev[k][ph], cudaEventDisableTiming));
            }
        }
        g_pxa_ts_ev[i].made = true;
    }
}

static bool pxa_tsplit_reduce_run_p2p(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                      const int * idx, int nhave) {
    auto & info = ggml_cuda_info();
    if (!pxa_tsplit_p2p_group(idx, nhave)) {
        pxa_tsplit_p2p_dump("PXA_TSPLIT_REDUCE");
        GGML_ABORT("PXA_TSPLIT_REDUCE=p2p: a half was asked to cross a non-peering pair; there is no "
                   "silent fallback to a blocking copy from here");
    }
    static bool banner = false;
    if (!banner) {
        banner = true;
        pxa_ts_make_events(idx, nhave);
        fprintf(stderr, "PXA_TSPLIT: reduce=p2p devices=%d event-ring=%d setdev-cache=on "
                        "back-edge=kept bit-identity=exact-vs-p2p-direct\n", nhave, PXA_TS_EV_RING);
    }
    const int ering = (int)(g_pxa_ts_p2p_calls++ % PXA_TS_EV_RING);

    const auto nelem    = ggml_nelements(dst);
    const auto nelem8   = (nelem + 7)/8;
    const auto npd      = 8*((nelem8 + nhave - 1)/nhave);
    const auto elem_sz  = ggml_element_size(dst);

    int cur = 0; CUDA_CHECK(cudaGetDevice(&cur)); g_pxa_ts_cur_dev = cur;

    // phase 0: my partial is ready
    for (int ii = 0; ii < nhave; ++ii) {
        const int i = idx[ii];
        pxa_ts_set_device(i);
        CUDA_CHECK(cudaEventRecord(g_pxa_ts_ev[i].ev[ering][0], info.all_ctx[i]->stream()));
    }
    // phase 1: wait on every peer, then reduce my chunk and write it to the peers
    for (int ii = 0; ii < nhave; ++ii) {
        const int i = idx[ii];
        pxa_ts_set_device(i);
        const int64_t this_nelem = std::max<int64_t>(0, std::min<int64_t>(npd, nelem - (int64_t)ii*npd));
        copy_task task;
        task.nptr  = nhave;
        task.nelem = (int)this_nelem;
        task.ptrs[0] = (char *)dst->src[i]->data + (size_t)ii*npd*elem_sz;
        int k = 1;
        for (int jj = 0; jj < nhave; ++jj) {
            if (jj == ii) continue;
            const int j = idx[jj];
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), g_pxa_ts_ev[j].ev[ering][0]));
            task.ptrs[k++] = (char *)dst->src[j]->data + (size_t)ii*npd*elem_sz;
        }
        if (this_nelem <= 0) continue;
        const int nblock = (int)((this_nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE);
        if (dst->type == GGML_TYPE_F16) {
            switch (nhave) {
                case 2: k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task); break;
                case 3: k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 3><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task); break;
                case 4: k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 4><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task); break;
                default: k_reduce_add<half, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            }
        } else {
            switch (nhave) {
                case 2: k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task); break;
                case 3: k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 3><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task); break;
                case 4: k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 4><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task); break;
                default: k_reduce_add<float, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    // phase 2: the back edge -- nobody may read its own buffer until every peer's write landed
    for (int ii = 0; ii < nhave; ++ii) {
        const int i = idx[ii];
        pxa_ts_set_device(i);
        CUDA_CHECK(cudaEventRecord(g_pxa_ts_ev[i].ev[ering][1], info.all_ctx[i]->stream()));
    }
    for (int ii = 0; ii < nhave; ++ii) {
        const int i = idx[ii];
        pxa_ts_set_device(i);
        for (int jj = 0; jj < nhave; ++jj) {
            if (jj == ii) continue;
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), g_pxa_ts_ev[idx[jj]].ev[ering][1]));
        }
    }
    pxa_ts_set_device(ctx.device);
    return true;
}

static int pxa_ts_collect_idx(const ggml_tensor * dst, int * idx) {
    int n = 0;
    const int nreduce = dst->op_params[1];
    for (int i = 0; i < nreduce; ++i) {
        if (dst->op_params[4] & (1u << i)) continue;
        if (dst->src[i]) idx[n++] = i;
    }
    return n;
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

enum { PXA_RT_NONE = 0, PXA_RT_NCCL, PXA_RT_RING, PXA_RT_P2P, PXA_RT_STAGED, PXA_RT_PINNED,
       PXA_RT_TSPLIT, PXA_RT_TSPLIT_P2P, PXA_RT_N };

static const char * pxa_rt_name(int r) {
    switch (r) {
        case PXA_RT_NCCL:   return "nccl";
        case PXA_RT_RING:   return "ring";
        case PXA_RT_P2P:    return "p2p-direct";
        case PXA_RT_STAGED: return "staged-copy";
        case PXA_RT_PINNED: return "pinned-host";
        case PXA_RT_TSPLIT:     return "pxa-fused";
        case PXA_RT_TSPLIT_P2P: return "pxa-p2p";
        default:            return "unrouted";
    }
}

#define PXA_RT_RING_N   64
#define PXA_RT_NE1_MAX  8

// PXA_TSPLIT: a mean hides exactly the thing a rendezvous costs, so keep every GPU-span sample
// (capped) and report p50 and p99 beside it. Host spans get the same treatment.
#define PXA_RT_KEEP 200000
struct pxa_rt_bucket {
    long long n = 0; double gpu_us = 0, gpu_max = 0, host_us = 0, host_max = 0;
    std::vector<float> gpu_s, host_s;
};
static double pxa_rt_pct(std::vector<float> & v, double q) {
    if (v.empty()) return 0.0;
    std::vector<float> t = v;
    size_t k = (size_t)(q * (t.size() - 1) + 0.5);
    std::nth_element(t.begin(), t.begin() + k, t.end());
    return t[k];
}

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
    fprintf(stderr, "PXA_REDUCE_TIME  %-12s %-5s %9s %9s %9s %9s %9s %9s %9s %9s\n",
            "route", "ne1", "n", "gpu_avg", "gpu_p50", "gpu_p99", "gpu_max", "host_avg", "host_p50", "host_p99");
    for (int r = 1; r < PXA_RT_N; ++r) {
        for (int k = 0; k <= PXA_RT_NE1_MAX; ++k) {
            auto & b = st.b[r][k];
            if (!b.n) continue;
            fprintf(stderr, "PXA_REDUCE_TIME  %-12s %-5d %9lld %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n",
                    pxa_rt_name(r), k, b.n, b.gpu_us / b.n,
                    pxa_rt_pct(b.gpu_s, 0.50), pxa_rt_pct(b.gpu_s, 0.99), b.gpu_max,
                    b.host_us / b.n, pxa_rt_pct(b.host_s, 0.50), pxa_rt_pct(b.host_s, 0.99));
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
                    if (b.gpu_s.size() < PXA_RT_KEEP) b.gpu_s.push_back((float)us);
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
        if (b.host_s.size() < PXA_RT_KEEP) b.host_s.push_back((float)host_us);
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

    // PXA_TSPLIT_REDUCE_v1: the tensor-split routes. Predicate and route are the same function, so
    // they can never disagree; a "yes" is binding and there is no fall-through from here.
    if (pxa_tsplit_reduce_handles(dst)) {
        int ts_idx[GGML_CUDA_MAX_DEVICES];
        const int ts_n = pxa_ts_collect_idx(dst, ts_idx);
        GGML_ASSERT(ts_n == nhave);
        // Hard constraint: a REDUCE node runs in the context whose device is the slot it aliases.
        // Every half writes only dst->src[dev]->data on its own device, so this still holds.
        GGML_ASSERT(dst->data == dst->src[ctx.device]->data);
        const bool ts_fused = pxa_tsplit_route() == PXA_TS_FUSED;
        _rt.route(ts_fused ? PXA_RT_TSPLIT : PXA_RT_TSPLIT_P2P);
        const bool ok = ts_fused ? pxa_tsplit_reduce_run    (ctx, dst, ts_idx, ts_n)
                                 : pxa_tsplit_reduce_run_p2p(ctx, dst, ts_idx, ts_n);
        if (!ok) {
            GGML_ABORT("PXA_TSPLIT_REDUCE: the predicate accepted this reduce but the route refused it; "
                       "falling through would be unsound");
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
    // The main NCCL communicator spans EVERY visible device, so it can only serve a reduce in which
    // every visible device takes part. A tensor split over a sub-group inside a larger process (the
    // island shape: tensor-parallel inside one island, pipelined between islands) has
    // nreduce < device_count and used to hit the assert below and abort the run. Skip the route in
    // that case instead, and let the in-tree peer routes take it.
    if (_pxa_nccl_ok && info.have_nccl && dst->type != GGML_TYPE_Q8_0 && nhave == nreduce && (nhave == 2 || dst->ne[1] < 32) &&
        info.device_count == nreduce &&
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
    // PXA_TSPLIT_P2P_PAIR: ctx.p2p_enabled is an AND over EVERY visible device, so a process that
    // also holds a card outside this reduce's peer island used to lose the fast route here even
    // between cards that peer perfectly well. Ask the per-pair matrix about the participating group
    // instead. On a single-island group the two agree, so this changes nothing there -- but on a
    // mixed fleet it changes which route a SHIPPING path takes, so it is off unless asked for
    // (PXA_TSPLIT_P2P_PAIR=1).
    const bool _pxa_pair_ok = ctx.p2p_enabled || (pxa_ts_pairmap() && pxa_tsplit_p2p_group(idx, nhave));
    if (dst->ne[1] < 32 && _pxa_pair_ok) {
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
