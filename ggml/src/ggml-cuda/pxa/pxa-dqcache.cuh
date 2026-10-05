// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-dqcache.cuh — K8-C: weight-stationary prefetch arena for the dequant->cuBLAS prefill path.
//
// WHAT IT IS. Not a cache. A per-device fp16 ring arena plus a two-node lookahead walker that
// runs the src0 weight dequant of an upcoming MUL_MAT on a low-priority side stream, so the
// consumer GEMM finds the fp16 operand already materialized instead of stalling behind its own
// dequant. Any hit from intra-graph reuse is a bonus, not the mechanism: the per-device fp16
// working set (~15-17 GB) dwarfs any affordable arena, and the access order is a clean cyclic
// scan, so an LRU cache over it would get ~0% hit rate. Prefetch is the only form that pays.
//
// WHY NOT ggml_cuda_pool. ggml_cuda_pool_vmm::free asserts strict LIFO order and
// ggml_cuda_pool_alloc's RAII hands the buffer back when the enclosing call returns, so a
// buffer that must outlive ggml_cuda_op_mul_mat_cublas cannot come from there. One cudaMalloc,
// owned here, is the only workable backing.
//
// SAFETY.
//  * IMMUTABILITY: only tensors in a GGML_BACKEND_BUFFER_USAGE_WEIGHTS, non-split CUDA buffer
//    are ever prefetched. Model weights are immutable for process lifetime, so there is no
//    invalidation problem. pxa_dqc_invalidate_all() (called from buffer free) is belt and braces.
//  * KEY: the exact tuple the consumer sees — (src0_dd_i, row_diff, ne00, src0->type). The
//    walker only accepts src0->ne[2] == ne[3] == 1 on a non-split buffer, which is precisely the
//    condition under which src0_dd_i == src0->data and row_diff == src0->ne[1], so the walker's
//    key and the consumer's key are the same tuple. The FULL tuple is compared, never a hash.
//  * ORDERING: every arena WRITE happens on the dq stream; every arena READ happens on the
//    compute stream. acquire() issues cudaStreamWaitEvent(compute, produced) so a read cannot
//    outrun its write. Reclaim is a full-drain wrap: before head resets to 0 an event is
//    recorded on the compute stream and the dq stream is made to wait on it, so no write can
//    outrun a still-pending read of the region it is about to overwrite. Offsets otherwise only
//    increase, so an in-use region can never be overwritten without passing that barrier.
//  * CUDA GRAPHS: acquire() declines while the consumer stream is capturing. The walker already
//    stands down under use_cuda_graph, but regions left live by an earlier eager eval remain
//    findable, and a hit under capture is doubly wrong — the wait-event crosses the capture
//    boundary, and a graph that replays a baked arena pointer reads a recycled region.
//  * LIFETIME: invalidate_all() runs from a buffer free, which is NOT ordered after the
//    destruction of the context owning the stored compute stream, so it synchronizes the device
//    and forgets that stream instead of recording the wrap barrier on a dead handle.
//
// GATE. PXA_DQC_MB, default 0 = OFF. This component changes allocation counts and stream
// structure — exactly the perturbation class the known delta-net layout aliasing bug is
// sensitive to on a hybrid model — so it ships armed only by env, separately measured.
// PXA_DQC_MIN_NY (default 64) is a second independent bar on top of the M>8 routing threshold
// that already keeps decode off this path entirely.
#pragma once

#include "../common.cuh"
#include "../convert.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------------------------
// env
// ---------------------------------------------------------------------------------------------
static inline size_t pxa_dqc_mb() {
    static const size_t v = [](){
        const char * e = getenv("PXA_DQC_MB");
        long x = e ? atol(e) : 0;
        if (x < 0) x = 0;
        if (x) fprintf(stderr, "PXA_DQC: prefetch arena ARMED, %ld MiB/device\n", x);
        return (size_t) x;
    }();
    return v;
}

static inline bool pxa_dqc_on() { return pxa_dqc_mb() != 0; }

static inline int64_t pxa_dqc_min_ny() {
    static const int64_t v = [](){
        const char * e = getenv("PXA_DQC_MIN_NY");
        long x = e ? atol(e) : 64;
        return (int64_t)(x < 1 ? 1 : x);
    }();
    return v;
}

static inline int pxa_dqc_depth() {
    static const int v = [](){
        const char * e = getenv("PXA_DQC_DEPTH");
        int x = e ? atoi(e) : 2;   // ~0.34 ms dequant vs ~4 ms consumer GEMM: one node is runway
        if (x < 1) x = 1;
        if (x > 8) x = 8;
        return x;
    }();
    return v;
}

// ---------------------------------------------------------------------------------------------
// counters (folded into the PXA_DQ_PROF periodic line by pxa_dqc_report_line)
// ---------------------------------------------------------------------------------------------
struct pxa_dqc_stat {
    std::atomic<uint64_t> hits{0}, misses{0}, prefetches{0}, bytes{0}, drains{0}, refused{0};
};
static pxa_dqc_stat g_pxa_dqc_stat[GGML_CUDA_MAX_DEVICES];

// ---------------------------------------------------------------------------------------------
// arena
// ---------------------------------------------------------------------------------------------
struct pxa_dqc_key {
    const void * ptr  = nullptr;
    int64_t      rows = 0;
    int64_t      cols = 0;
    int          type = -1;
    bool operator==(const pxa_dqc_key & o) const {
        return ptr == o.ptr && rows == o.rows && cols == o.cols && type == o.type;
    }
};

struct pxa_dqc_region {
    pxa_dqc_key key;
    size_t      off      = 0;
    size_t      bytes    = 0;
    cudaEvent_t produced = nullptr;
};

struct pxa_dqc_arena {
    std::mutex   mtx;
    bool         tried   = false;
    void *       base    = nullptr;
    size_t       cap     = 0;
    size_t       head    = 0;
    cudaStream_t dq      = nullptr;
    cudaEvent_t  barrier = nullptr;
    cudaStream_t last_compute = nullptr;
    std::deque<pxa_dqc_region> live;
    std::vector<cudaEvent_t>   evfree;
};

static pxa_dqc_arena g_pxa_dqc_arena[GGML_CUDA_MAX_DEVICES];

// caller holds a.mtx
static bool pxa_dqc_ensure(pxa_dqc_arena & a, int device) {
    if (a.tried) {
        return a.base != nullptr;
    }
    a.tried = true;
    const size_t want = pxa_dqc_mb() * 1024ull * 1024ull;
    if (!want) {
        return false;
    }
    ggml_cuda_set_device(device);
    if (cudaMalloc(&a.base, want) != cudaSuccess) {
        a.base = nullptr;
        (void) cudaGetLastError();
        fprintf(stderr, "PXA_DQC dev%d: cudaMalloc(%zu MiB) FAILED — prefetch disabled, "
                        "incumbent dequant path in use\n", device, want/(1024*1024));
        return false;
    }
    a.cap = want;
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    // `least` is the numerically-largest, lowest-priority value: cuBLAS CTAs win SM arbitration.
    if (cudaStreamCreateWithPriority(&a.dq, cudaStreamNonBlocking, least) != cudaSuccess) {
        (void) cudaGetLastError();
        CUDA_CHECK(cudaStreamCreateWithFlags(&a.dq, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&a.barrier, cudaEventDisableTiming));
    fprintf(stderr, "PXA_DQC dev%d: arena %zu MiB, lookahead %d, min_ny %lld\n",
            device, a.cap/(1024*1024), pxa_dqc_depth(), (long long) pxa_dqc_min_ny());
    return true;
}

// caller holds a.mtx
static cudaEvent_t pxa_dqc_take_event(pxa_dqc_arena & a) {
    if (!a.evfree.empty()) {
        cudaEvent_t e = a.evfree.back();
        a.evfree.pop_back();
        return e;
    }
    cudaEvent_t e = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return e;
}

// caller holds a.mtx. Recycle every live region's event and rewind the ring. The ORDERING that
// makes the rewind safe is the caller's job: nothing here keeps a future write off a region a
// pending GEMM is still reading.
static void pxa_dqc_reset(pxa_dqc_arena & a, int device) {
    for (auto & r : a.live) {
        a.evfree.push_back(r.produced);
    }
    a.live.clear();
    a.head = 0;
    g_pxa_dqc_stat[device].drains.fetch_add(1, std::memory_order_relaxed);
}

// caller holds a.mtx. Wrap-reclaim: order every future arena write behind everything already
// enqueued on the compute stream, which includes every GEMM that reads a live region. Only ever
// called from the producer, which refreshes a.last_compute to the live stream first.
static void pxa_dqc_drain(pxa_dqc_arena & a, int device) {
    if (a.last_compute) {
        CUDA_CHECK(cudaEventRecord(a.barrier, a.last_compute));
        CUDA_CHECK(cudaStreamWaitEvent(a.dq, a.barrier, 0));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(a.dq));
    }
    pxa_dqc_reset(a, device);
}

// caller holds a.mtx
static pxa_dqc_region * pxa_dqc_find(pxa_dqc_arena & a, const pxa_dqc_key & k) {
    for (auto & r : a.live) {
        if (r.key == k) {
            return &r;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------------------------
// producer: dequant one weight tensor into the arena on the dq stream
// ---------------------------------------------------------------------------------------------
static void pxa_dqc_prefetch(int device, const pxa_dqc_key & k, cudaStream_t compute) {
    pxa_dqc_arena & a = g_pxa_dqc_arena[device];
    std::lock_guard<std::mutex> lk(a.mtx);
    if (!pxa_dqc_ensure(a, device)) {
        return;
    }
    a.last_compute = compute;
    if (pxa_dqc_find(a, k)) {
        return;                                     // already resident (or already in flight)
    }
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda((ggml_type) k.type);
    if (!to_fp16) {
        return;
    }
    const size_t bytes = (size_t) k.rows * (size_t) k.cols * sizeof(half);
    const size_t need  = (bytes + 255) & ~(size_t)255;
    if (need > a.cap) {
        g_pxa_dqc_stat[device].refused.fetch_add(1, std::memory_order_relaxed);
        return;                                     // one tensor larger than the whole arena
    }
    if (a.head + need > a.cap) {
        pxa_dqc_drain(a, device);
    }

    pxa_dqc_region r;
    r.key      = k;
    r.off      = a.head;
    r.bytes    = bytes;
    r.produced = pxa_dqc_take_event(a);
    a.head    += need;

    to_fp16(k.ptr, (half *)((char *) a.base + r.off), k.rows, k.cols, a.dq);
    CUDA_CHECK(cudaEventRecord(r.produced, a.dq));
    a.live.push_back(r);

    g_pxa_dqc_stat[device].prefetches.fetch_add(1, std::memory_order_relaxed);
    g_pxa_dqc_stat[device].bytes.fetch_add(bytes, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------
// consumer: hand the GEMM an already-materialized fp16 operand, or nullptr to fall through to
// the byte-identical incumbent pool path
// ---------------------------------------------------------------------------------------------
static const half * pxa_dqc_acquire(int device, const void * src0_dd_i, int64_t row_diff,
                                    int64_t ne00, int type, cudaStream_t compute) {
    if (!pxa_dqc_on()) {
        return nullptr;
    }
    // NEVER serve a hit into a CUDA-graph capture. pxa_dqc_walk() declines to prefetch while
    // use_cuda_graph is set, but regions left live by an earlier EAGER eval are still findable
    // here, and both halves of a hit are wrong under capture:
    //   * cudaStreamWaitEvent(compute, produced) on an event recorded outside this capture is
    //     cudaErrorStreamCaptureIsolation, which CUDA_CHECK turns into an abort; and
    //   * the arena pointer would be baked into the executable graph and replayed long after the
    //     ring recycled that region, i.e. silent garbage weights.
    // Same "decline mid-capture" rule the pool and the zero-buffer paths already follow.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(compute, &cap) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    if (cap != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
    pxa_dqc_arena & a = g_pxa_dqc_arena[device];
    std::lock_guard<std::mutex> lk(a.mtx);
    if (!a.base) {
        return nullptr;
    }
    a.last_compute = compute;
    const pxa_dqc_key k{ src0_dd_i, row_diff, ne00, type };
    pxa_dqc_region * r = pxa_dqc_find(a, k);
    if (!r) {
        g_pxa_dqc_stat[device].misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    CUDA_CHECK(cudaStreamWaitEvent(compute, r->produced, 0));
    g_pxa_dqc_stat[device].hits.fetch_add(1, std::memory_order_relaxed);
    return (const half *)((char *) a.base + r->off);
}

// ---------------------------------------------------------------------------------------------
// invalidation (belt and braces; WEIGHTS buffers are immutable for process lifetime)
// ---------------------------------------------------------------------------------------------
static void pxa_dqc_invalidate_all() {
    if (!pxa_dqc_on()) {
        return;
    }
    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess) {
        (void) cudaGetLastError();
        cur = -1;
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        pxa_dqc_arena & a = g_pxa_dqc_arena[d];
        std::lock_guard<std::mutex> lk(a.mtx);
        if (!a.base) {
            continue;
        }
        // NOT pxa_dqc_drain(): this runs from a buffer free, and a buffer free is NOT ordered
        // after the destruction of the context that owns a.last_compute. llama_free(ctx) runs
        // before llama_free_model(model) in main, llama-bench and the server, so by the time the
        // weight buffers are released ggml_backend_cuda_context::~ggml_backend_cuda_context has
        // already cudaStreamDestroy'd every stream: recording the wrap barrier on a.last_compute
        // here would pass a destroyed stream handle to cudaEventRecord, which CUDA_CHECK turns
        // into an abort on the way out. Synchronize the device instead — strictly stronger than
        // the wrap barrier, and this path only fires on a buffer free.
        ggml_cuda_set_device(d);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            (void) cudaGetLastError();
        }
        a.last_compute = nullptr;   // never hold a stream handle across a free
        pxa_dqc_reset(a, d);
    }
    if (cur >= 0) {
        (void) cudaSetDevice(cur);
    }
}

static void pxa_dqc_report_line() {
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        const uint64_t h = g_pxa_dqc_stat[d].hits.load(), m = g_pxa_dqc_stat[d].misses.load();
        if (!h && !m) {
            continue;
        }
        fprintf(stderr, "PXA_DQC dev%d: %llu hits / %llu misses (%.1f%%), %llu prefetches, "
                        "%.2f GiB prefetched, %llu drains, %llu refused\n",
                d, (unsigned long long) h, (unsigned long long) m,
                (h + m) ? 100.0*h/(double)(h + m) : 0.0,
                (unsigned long long) g_pxa_dqc_stat[d].prefetches.load(),
                g_pxa_dqc_stat[d].bytes.load()/(1024.0*1024.0*1024.0),
                (unsigned long long) g_pxa_dqc_stat[d].drains.load(),
                (unsigned long long) g_pxa_dqc_stat[d].refused.load());
    }
}

// ---------------------------------------------------------------------------------------------
// lookahead walker — called immediately before ggml_cuda_compute_forward for node i
// ---------------------------------------------------------------------------------------------
// `src0_is_split` is supplied by the caller: ggml_backend_buft_is_cuda_split has internal
// linkage in ggml-cuda.cu, and keeping it out of here is what lets the harness link this
// predicate directly.
static bool pxa_dqc_node_ok(const ggml_tensor * node, int64_t min_ny, bool src0_is_split) {
    if (!node || node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1 || !src0->buffer) {
        return false;
    }
    // IMMUTABILITY GATE: model weights only, never a compute buffer.
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    if (src0_is_split) {
        return false;
    }
    if (!ggml_is_quantized(src0->type) || !ggml_is_contiguous(src0)) {
        return false;
    }
    if (!ggml_get_to_fp16_cuda(src0->type)) {
        return false;
    }
    // The guarantee that the walker's key == the consumer's key: with ne[2] == ne[3] == 1 and a
    // non-split buffer, src0_dd_i is src0->data and row_diff is src0->ne[1].
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->ne[1] < min_ny) {
        return false;
    }
    if ((size_t) src0->ne[0] * (size_t) src0->ne[1] * sizeof(half) < 1024*1024) {
        return false;
    }
    return true;
}

#ifndef PXA_DQC_NO_WALKER
static void pxa_dqc_walk(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, bool use_cuda_graph) {
    if (!pxa_dqc_on() || use_cuda_graph || !cgraph) {
        return;
    }
    const int device = ctx.device;
    if (ggml_cuda_info().devices[device].cc != CC_VOLTA) {
        return;
    }
    const int64_t     min_ny  = pxa_dqc_min_ny();
    const int         depth   = pxa_dqc_depth();
    const cudaStream_t compute = ctx.stream();
    for (int d = 0; d <= depth; ++d) {
        const int j = i + d;
        if (j >= cgraph->n_nodes) {
            break;
        }
        const ggml_tensor * node = cgraph->nodes[j];
        const ggml_tensor * src0 = node->src[0];
        const bool is_split = src0 && src0->buffer && ggml_backend_buft_is_cuda_split(src0->buffer->buft);
        if (!pxa_dqc_node_ok(node, min_ny, is_split)) {
            continue;
        }
        pxa_dqc_prefetch(device, pxa_dqc_key{ src0->data, src0->ne[1], src0->ne[0], (int) src0->type }, compute);
    }
}
#endif // PXA_DQC_NO_WALKER
