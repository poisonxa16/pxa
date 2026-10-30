// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-stream.cuh -- PXA_STREAM_WEIGHTS: weights that live in pinned host RAM and are streamed to
// the GPU once per graph evaluation (one ubatch pass), double-buffered through a device ring so
// the host->device copy of the next tensors overlaps the compute of the current ones.
//
// WHAT IT IS. A buffer type per device, "CUDA<d>_Stream", whose memory is cudaHostAlloc'd pinned
// host RAM but which presents itself to the rest of the CUDA backend as a CUDA buffer on device d
// (same get_name function, same ggml_backend_cuda_buffer_context layout). The scheduler therefore
// assigns every op that reads such a weight to device d exactly as if the weight were resident:
// no graph splits, no CPU fallback, and every PXQ fast path that checks
// ggml_backend_buffer_is_cuda(src0->buffer) keeps working. The loader puts weights there when
// PXA_STREAM_WEIGHTS=layers|experts (src/llama-load-tensors.cpp).
//
// HOW A GRAPH RUNS. At the start of an eager evaluation (CUDA graphs are forced off for a graph
// that reads a streamed weight) pxa_stream_begin() builds the plan: every streamed tensor the
// graph reads, with the node index of its first and last use, in first-use order. Before each
// node, pxa_stream_walk():
//   1. releases the ring regions of tensors whose last use is behind the walk (one event recorded
//      on the compute stream per region);
//   2. issues as many copies as the ring has room for, in first-use order, on a per-device copy
//      stream. A copy that reuses ring bytes waits (GPU side, cudaStreamWaitEvent) on the release
//      event of the region it overwrites, so the host never blocks. Issuing swaps tensor->data
//      (and any view of it the graph reads) to the ring slot; pxa_stream_end() restores them;
//   3. makes the compute stream wait for the copy-done event of every tensor first used within
//      the next LOOKAHEAD nodes (fusions read the srcs of nodes ahead of the current one).
// If a tensor that is needed within the lookahead cannot get ring space (ring too small for the
// live set), it is left pointing at its pinned host bytes, which are device-readable under UVA:
// correct, slow (zero-copy over PCIe), counted in the stats line.
//
// WHAT IT COSTS. Every streamed byte crosses PCIe once per graph evaluation. A prefill ubatch of
// width N pays bytes/BW once for N tokens, so above a break-even width the copy hides under the
// compute; decode (N=1) pays it per token. That trade is the lever, and it is off by default.
//
// ENV (read once):
//   PXA_STREAM_LOOKAHEAD  nodes ahead whose streamed weights must be resident (default 16)
//   PXA_STREAM_ZC_NY      graphs whose widest streamed-weight consumer has <= this many tokens
//                         read the pinned host bytes directly (zero-copy) instead of copying.
//                         Exported, it is authoritative for every stream buffer of the process.
//                         Unset, each stream buffer carries its own model's width, set by the
//                         loader with ggml_backend_cuda_stream_buffer_set_zc_ny (default 0 =
//                         always copy; 32 for experts mode, where a narrow batch touches only the
//                         routed experts' rows; 64 for the expert cache's zero-copy cold path and
//                         its runtime guard). The engine never writes this variable: a width it
//                         picked for one model cannot reach another model, a later load or a
//                         child process.
//   PXA_STREAM_LOG        1 = one stats line per 64 graphs per device and at exit
//
// RING LENDING (PXA_XCACHE_RING_LEND, R3a 2026-10-25; the loader arms it per device with ggml_backend_cuda_stream_set_ring_lent).
// With the expert cache's CPU cold path a graph narrower than the CPU-alias width (32 x n_expert / n_expert_used tokens,
// 1638 on Flash-Next) never reads a streamed weight on the GPU, so the ring idles through every decode. A lent ring is not
// allocated at load: the planner gave its bytes to resident experts. The first graph that does copy streamed weights
// (pxa_stream_begin, not all zero-copy) allocates it on demand from the VRAM left free, at most the planned size and never
// below PXA_XCACHE_RING_LEND_KEEP_MB (default 512) of free VRAM; ggml_backend_cuda_stream_ring_return() (llama_synchronize)
// frees it again. When that allocation cannot be had (the card is full, or PXA_XCACHE_RING_LEND_FAIL=1 in a test) the graph
// reads every streamed weight in place over the link, exactly as an all-zero-copy graph does (PXA_STREAM_ZC_NY), so a long
// prefill never aborts on the ring. Measured G30 (w2/w3, 2234-token prompt, 64 greedy tokens): the fallback's answer
// equals PXA_STREAM_ZC_NY=4096 on the same plan (sha 95427b95f060 both); a copied run of that plan answered f8df01b88110, so
// copy vs in-place reads are correct-not-identical at some residencies (equal on the stock plan), like any zero-copy graph.
#pragma once

#include "pxa-env-flag.h"
#include "../common.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------------------------
// env
// ---------------------------------------------------------------------------------------------
static int pxa_stream_lookahead() {
    static const int v = [](){ const char * e = getenv("PXA_STREAM_LOOKAHEAD"); int x = e ? atoi(e) : 16; return x < 1 ? 1 : x; }();
    return v;
}
// the exported PXA_STREAM_ZC_NY (any value, empty = 0, as the old reader took it), or -1 when unset: then each buffer's own width
static int64_t pxa_stream_zc_ny_env() {
    static const int64_t v = [](){ const char * e = getenv("PXA_STREAM_ZC_NY"); return e ? (int64_t)atoll(e) : (int64_t)-1; }();
    return v;
}
static bool pxa_stream_log() {
    static const bool v = [](){ const char * e = getenv("PXA_STREAM_LOG"); return e && atoi(e) != 0; }();
    return v;
}

// true once any stream buffer has been allocated in this process -- keeps the per-graph scan off
// the hot path of every build that never arms the lever.
static std::atomic<bool> g_pxa_stream_any{false};

// ---------------------------------------------------------------------------------------------
// the buffer type
// ---------------------------------------------------------------------------------------------
static size_t g_pxa_stream_ring_req[GGML_CUDA_MAX_DEVICES] = {0};
static bool   g_pxa_stream_ring_lent[GGML_CUDA_MAX_DEVICES] = {false};   // PXA_XCACHE_RING_LEND (see the header comment)
static std::atomic<bool> g_pxa_stream_lent_live{false};                 // some lent ring is allocated right now

struct pxa_stream_buffer_context : public ggml_backend_cuda_buffer_context {
    void * host_ptr;
    int64_t zc_ny = 0;   // graphs no wider than this read this buffer's weights in place (the owning model's width; 0 = always copy)
    pxa_stream_buffer_context(int device, void * p) : ggml_backend_cuda_buffer_context(device, p), host_ptr(p) {
        // the base class frees dev_ptr with cudaFree in its destructor; ours is host memory,
        // freed by the buffer's free function, so the base must see nullptr.
    }
};

static void pxa_stream_ring_ensure(int device);

GGML_CALL static void pxa_stream_buffer_free(ggml_backend_buffer_t buffer) {
    pxa_stream_buffer_context * ctx = (pxa_stream_buffer_context *)buffer->context;
    void * p = ctx->host_ptr;
    ctx->dev_ptr = nullptr;
    delete ctx;
    if (p) CUDA_CHECK(cudaFreeHost(p));
}

GGML_CALL static void * pxa_stream_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((pxa_stream_buffer_context *)buffer->context)->host_ptr;
}

GGML_CALL static void pxa_stream_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != NULL) return;
    // zero the row padding like the device buffer does (the kernels read it)
    const size_t original_size = ggml_nbytes(tensor);
    const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);
    if (padded_size > original_size) {
        memset((char *)tensor->data + original_size, 0, padded_size - original_size);
    }
}

GGML_CALL static void pxa_stream_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

GGML_CALL static void pxa_stream_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

GGML_CALL static void pxa_stream_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

GGML_CALL static bool pxa_stream_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer); GGML_UNUSED(src); GGML_UNUSED(dst);
    return false;
}

GGML_CALL static void pxa_stream_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(((pxa_stream_buffer_context *)buffer->context)->host_ptr, value, buffer->size);
}

// get_name is the DEVICE buffer's function on purpose: ggml_backend_buffer_is_cuda() must be true
// for these buffers so the PXQ/MMVQ fast paths that check it treat a streamed weight as on-device.
static ggml_backend_buffer_i pxa_stream_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ pxa_stream_buffer_free,
    /* .get_base        = */ pxa_stream_buffer_get_base,
    /* .init_tensor     = */ pxa_stream_buffer_init_tensor,
    /* .memset_tensor   = */ pxa_stream_buffer_memset_tensor,
    /* .set_tensor      = */ pxa_stream_buffer_set_tensor,
    /* .get_tensor      = */ pxa_stream_buffer_get_tensor,
    /* .cpy_tensor      = */ pxa_stream_buffer_cpy_tensor,
    /* .clear           = */ pxa_stream_buffer_clear,
    /* .reset           = */ NULL,
};

static inline bool pxa_stream_buffer_is(const ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.free_buffer == pxa_stream_buffer_free;
}

GGML_CALL static ggml_backend_buffer_t pxa_stream_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
    ggml_cuda_set_device(buft_ctx->device);
    size = std::max(size, (size_t)1);
    void * p = nullptr;
    // portable: any device's copy engine reads it at full rate; under UVA it is also device-
    // addressable at the same pointer, which is what the zero-copy fallback relies on.
    // PXA_NUMA_BIND: the streamed experts are read by the host CPU on every token (the expert cache's cold path), so their pages go on the
    // NODE THE CARDS HANG OFF -- a hard bind here (reclaiming page cache of that node if it has to), where the process-wide preference alone
    // would let the per-layer-embedding table take that node's free pages first. A failed bound allocation is retried under the old policy.
    cudaError_t err = cudaErrorMemoryAllocation;
#if defined(__linux__)
    // PXA_NUMA_HOST_BIND: when the cards' node has no allowed CPU, the node the CPU threads sit on (pxa_numa_stream_node)
    if (const int bind_node = pxa_numa_stream_node(size); bind_node >= 0) {
        unsigned long m[2] = { 0, 0 };
        m[bind_node / 64] = 1UL << (bind_node % 64);
        const long brc = syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, m, (unsigned long) 128);
        err = cudaHostAlloc(&p, size, cudaHostAllocPortable | cudaHostAllocMapped);
        if (g_pxa_numa_bound >= 0) {
            syscall(SYS_set_mempolicy, 1 /* MPOL_PREFERRED */, m, (unsigned long) 128);
        } else {
            syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, (unsigned long) 0);
            GGML_CUDA_LOG_INFO("PXA_NUMA_HOST_BIND: %.2f GiB of streamed experts pinned with their pages bound to node %d%s\n",
                    size / 1073741824.0, bind_node, brc == 0 ? "" : " (set_mempolicy refused: --cap-add SYS_NICE in a container; placed unbound)");
        }
        if (err != cudaSuccess) { cudaGetLastError(); p = nullptr; GGML_CUDA_LOG_WARN("PXA_NUMA_BIND: the bound pinned allocation of %.0f MiB failed; retrying unbound\n", size/1024.0/1024.0); }
    }
#endif
    if (err != cudaSuccess) err = cudaHostAlloc(&p, size, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        cudaGetLastError();
        GGML_CUDA_LOG_ERROR("%s: pinning %.2f MiB for device %d failed: %s\n", __func__, size/1024.0/1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }
    // the ring is allocated together with the first stream buffer of the device, i.e. during model
    // load and BEFORE the KV cache and compute buffers, so their sizing sees it.
    pxa_stream_ring_ensure(buft_ctx->device);
    g_pxa_stream_any.store(true);
    GGML_CUDA_LOG_INFO("PXA_STREAM: device %d: %.2f MiB of weights pinned in host RAM (streamed per graph)\n",
            buft_ctx->device, size/1024.0/1024.0);
    auto * ctx = new pxa_stream_buffer_context(buft_ctx->device, p);
    return ggml_backend_buffer_init(buft, pxa_stream_buffer_interface, ctx, size);
}

// ---------------------------------------------------------------------------------------------
// per-device streaming state
// ---------------------------------------------------------------------------------------------
struct pxa_stream_dev {
    int          device      = -1;
    char       * ring        = nullptr;
    size_t       ring_bytes  = 0;
    cudaStream_t cstream     = nullptr;

    struct region { size_t off, size; cudaEvent_t rel_ev; bool released; };
    std::vector<region> live;         // allocated ring regions (non-overlapping)
    size_t head = 0;
    std::vector<cudaEvent_t> ev_pool;

    struct view_fix { ggml_tensor * t; size_t offs; };
    struct item {
        ggml_tensor * base; int first, last; size_t bytes;
        int state;          // 0 pending, 1 issued (copy enqueued), 2 compute waited
        bool zc;            // left on host (zero-copy)
        size_t off; cudaEvent_t done_ev; bool released;
        std::vector<view_fix> views;
    };
    std::vector<item> plan;
    std::vector<int>  by_last;
    size_t issue_idx = 0, wait_idx = 0, rel_idx = 0;
    std::vector<std::pair<ggml_tensor *, void *>> restore;
    bool active = false;

    // PXA_XCACHE_RING_LEND: the ring is allocated on demand (lend_want bytes at most) and returned at the next sync
    bool   lent      = false;
    size_t lend_want = 0;
    long long n_lend_alloc = 0, n_lend_fail = 0, n_lend_short = 0;

    // stats
    long long n_graphs = 0, n_copies = 0, n_zc = 0, n_zc_graphs = 0;
    double bytes = 0;

    cudaEvent_t ev_get() {
        if (!ev_pool.empty()) { cudaEvent_t e = ev_pool.back(); ev_pool.pop_back(); return e; }
        cudaEvent_t e; CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e;
    }
    void ev_put(cudaEvent_t e) { if (e) ev_pool.push_back(e); }
};
static pxa_stream_dev g_pxa_stream[GGML_CUDA_MAX_DEVICES];

static void pxa_stream_ring_ensure(int device) {
    pxa_stream_dev & s = g_pxa_stream[device];
    if (s.ring || s.lent) return;
    size_t want = g_pxa_stream_ring_req[device];
    if (const char * e = getenv("PXA_STREAM_RING_MB")) want = (size_t)atoll(e) << 20;
    if (want == 0) want = (size_t)512 << 20;
    want = GGML_PAD(want, 1 << 20);
    ggml_cuda_set_device(device);
    if (g_pxa_stream_ring_lent[device]) {
        // PXA_XCACHE_RING_LEND: nothing in VRAM now; pxa_stream_lend_alloc takes it when a graph streams
        s.device = device;
        s.lent = true;
        s.lend_want = want;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.cstream, cudaStreamNonBlocking));
        GGML_CUDA_LOG_INFO("PXA_STREAM: device %d: ring %.1f MiB LENT to the expert cache (PXA_XCACHE_RING_LEND): not in VRAM; a graph "
                "that streams weights allocates it from the free VRAM and the next sync returns it (short of VRAM: zero-copy reads)\n",
                device, want/1048576.0);
        return;
    }
    void * p = nullptr;
    cudaError_t err = ggml_cuda_device_malloc(&p, want, device);
    if (err != cudaSuccess) {
        cudaGetLastError();
        GGML_ABORT("PXA_STREAM: device %d: ring of %.1f MiB could not be allocated (%s)", device, want/1048576.0, cudaGetErrorString(err));
    }
    s.device = device;
    s.ring = (char *)p;
    s.ring_bytes = want;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.cstream, cudaStreamNonBlocking));
    if (pxa_stream_zc_ny_env() >= 0) {
        GGML_CUDA_LOG_INFO("PXA_STREAM: device %d: ring %.1f MiB in VRAM, lookahead %d nodes, zero-copy at ny <= %lld (PXA_STREAM_ZC_NY)\n",
                device, want/1048576.0, pxa_stream_lookahead(), (long long)pxa_stream_zc_ny_env());
    } else {
        GGML_CUDA_LOG_INFO("PXA_STREAM: device %d: ring %.1f MiB in VRAM, lookahead %d nodes, zero-copy width per model\n",
                device, want/1048576.0, pxa_stream_lookahead());
    }
}

// the loader's per-model zero-copy width for one stream buffer (ggml_backend_cuda_stream_buffer_set_zc_ny); an exported
// PXA_STREAM_ZC_NY still decides over it (pxa_stream_zc_ny_of)
static void pxa_stream_buffer_set_zc_ny(ggml_backend_buffer_t buffer, int64_t ny) {
    if (!pxa_stream_buffer_is(buffer)) return;
    ((pxa_stream_buffer_context *)buffer->context)->zc_ny = ny < 0 ? 0 : ny;
}

static void pxa_stream_stats(pxa_stream_dev & s, const char * why) {
    fprintf(stderr, "PXA_STREAM stats dev=%d (%s): graphs=%lld copies=%lld bytes=%.2f GiB zero-copy-fallbacks=%lld zc-graphs=%lld\n",
            s.device, why, s.n_graphs, s.n_copies, s.bytes/1073741824.0, s.n_zc, s.n_zc_graphs);
    if (s.lent) {
        fprintf(stderr, "PXA_STREAM lent ring dev=%d (%s): taken back %lld times (%lld below the planned size), unavailable %lld times\n",
                s.device, why, s.n_lend_alloc, s.n_lend_short, s.n_lend_fail);
    }
}
static void pxa_stream_atexit() {
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) if (g_pxa_stream[d].ring || g_pxa_stream[d].lent) pxa_stream_stats(g_pxa_stream[d], "exit");
}

static inline ggml_tensor * pxa_stream_base_of(ggml_tensor * t) {
    if (!t) return nullptr;
    ggml_tensor * b = t->view_src ? t->view_src : t;
    return (b->buffer && pxa_stream_buffer_is(b->buffer)) ? b : nullptr;
}

// tokens a weight consumer processes: dst is [rows, tokens] for a matmul and [rows, n_used, tokens]
// for the expert ops (whose activation operand is not src[1] in the fused up/gate form)
static inline int64_t pxa_stream_node_width(const ggml_tensor * n) {
    if (n->op == GGML_OP_MUL_MAT_ID || n->op == GGML_OP_MOE_FUSED_UP_GATE) return n->ne[2]*n->ne[3];
    return n->ne[1]*n->ne[2]*n->ne[3];
}

// zero-copy width for a streamed base tensor (pxa_stream_base_of): the exported PXA_STREAM_ZC_NY, else its buffer's own
static inline int64_t pxa_stream_zc_ny_of(const ggml_tensor * b) {
    const int64_t e = pxa_stream_zc_ny_env();
    return e >= 0 ? e : ((const pxa_stream_buffer_context *)b->buffer->context)->zc_ny;
}
// a node that reads streamed base b reads it in place when its width fits b's zero-copy width
static inline bool pxa_stream_node_zc(const ggml_tensor * n, const ggml_tensor * b) {
    const int64_t zc_ny = pxa_stream_zc_ny_of(b);
    return zc_ny > 0 && pxa_stream_node_width(n) <= zc_ny;
}

// PXA_STREAM_ZC_GRAPHS (default 1): a graph that reads every streamed weight in place
// (zero-copy, no wider than PXA_STREAM_ZC_NY) has nothing host-driven -- no ring, no copy stream,
// no pointer swaps -- so it may be captured and replayed like a resident graph.
static bool pxa_stream_zc_graphs() {
    static const bool v = [](){ const char * e = getenv("PXA_STREAM_ZC_GRAPHS"); return !(e && pxa_env_value_off(e)); }();
    return v;
}

// cheap scan used to force CUDA graphs off for graphs that read a streamed weight
static bool pxa_stream_graph_uses(const ggml_cgraph * cgraph) {
    if (!g_pxa_stream_any.load(std::memory_order_relaxed)) return false;
    bool uses = false;
    bool all_zc = true;   // every streamed weight the graph reads is read in place (its buffer's width; one model = one width)
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (const ggml_tensor * b = pxa_stream_base_of(n->src[j])) {
                uses = true;
                if (all_zc && !pxa_stream_node_zc(n, b)) all_zc = false;
            }
        }
    }
    if (!uses) return false;
    if (all_zc && pxa_stream_zc_graphs()) return false;
    return true;
}

// ring allocation: returns false (nothing changed) if any overlapping region is still in use
static bool pxa_stream_ring_alloc(pxa_stream_dev & s, size_t size, size_t & off_out) {
    size = GGML_PAD(size, 256);
    if (size > s.ring_bytes) return false;
    size_t p = s.head;
    if (p + size > s.ring_bytes) p = 0;
    for (auto & r : s.live) {
        const bool ov = r.off < p + size && p < r.off + r.size;
        if (ov && !r.released) return false;
    }
    for (size_t k = 0; k < s.live.size(); ) {
        auto & r = s.live[k];
        const bool ov = r.off < p + size && p < r.off + r.size;
        if (ov) {
            CUDA_CHECK(cudaStreamWaitEvent(s.cstream, r.rel_ev, 0));
            s.ev_put(r.rel_ev);
            s.live[k] = s.live.back(); s.live.pop_back();
        } else {
            ++k;
        }
    }
    s.live.push_back({p, size, nullptr, false});
    s.head = p + size;
    off_out = p;
    return true;
}

static void pxa_stream_release(pxa_stream_dev & s, pxa_stream_dev::item & it, cudaStream_t compute) {
    if (it.released) return;
    it.released = true;
    if (it.zc || it.state == 0) return;
    for (auto & r : s.live) {
        if (r.off == it.off && !r.released) {
            r.rel_ev = s.ev_get();
            CUDA_CHECK(cudaEventRecord(r.rel_ev, compute));
            r.released = true;
            return;
        }
    }
}

static void pxa_stream_swap(pxa_stream_dev & s, pxa_stream_dev::item & it, char * dst) {
    s.restore.push_back({it.base, it.base->data});
    for (auto & v : it.views) {
        s.restore.push_back({v.t, v.t->data});
        v.t->data = dst + v.offs;
    }
    it.base->data = dst;
}

// try to issue the next planned copy; returns false if the ring has no room for it yet
static bool pxa_stream_issue_one(pxa_stream_dev & s, pxa_stream_dev::item & it) {
    size_t off = 0;
    if (!pxa_stream_ring_alloc(s, it.bytes, off)) return false;
    it.off = off;
    char * dst = s.ring + off;
    CUDA_CHECK(cudaMemcpyAsync(dst, it.base->data, it.bytes, cudaMemcpyHostToDevice, s.cstream));
    it.done_ev = s.ev_get();
    CUDA_CHECK(cudaEventRecord(it.done_ev, s.cstream));
    it.state = 1;
    s.n_copies++;
    s.bytes += (double)it.bytes;
    pxa_stream_swap(s, it, dst);
    return true;
}

// PXA_XCACHE_RING_LEND_KEEP_MB: free VRAM a lent ring never takes (the CUDA pool grows during the graph; default 512)
static size_t pxa_stream_lend_keep() {
    static const size_t v = [](){ const char * e = getenv("PXA_XCACHE_RING_LEND_KEEP_MB"); long long x = e ? atoll(e) : 512; return (size_t)(x < 0 ? 0 : x) << 20; }();
    return v;
}
// PXA_XCACHE_RING_LEND_FAIL=1 (tests only): every on-demand allocation fails, so the zero-copy fallback runs
static bool pxa_stream_lend_fail_test() {
    static const bool v = [](){ const char * e = getenv("PXA_XCACHE_RING_LEND_FAIL"); return e && atoi(e) != 0; }();
    return v;
}

// a lent ring for this graph: at most lend_want, at least min_bytes (the largest streamed tensor), keeping
// pxa_stream_lend_keep() of the device free; false = no ring (the caller reads everything in place)
static bool pxa_stream_lend_alloc(pxa_stream_dev & s, size_t min_bytes) {
    if (pxa_stream_lend_fail_test()) return false;
    ggml_cuda_set_device(s.device);
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); return false; }
    const size_t keep = pxa_stream_lend_keep();
    size_t want = std::min(s.lend_want, fr > keep ? fr - keep : (size_t)0);
    want &= ~(((size_t)1 << 20) - 1);
    min_bytes = GGML_PAD(min_bytes, 1 << 20);
    if (want < s.lend_want) s.n_lend_short++;
    for (; want >= min_bytes && want > 0; want = (want/2) & ~(((size_t)1 << 20) - 1)) {
        void * p = nullptr;
        if (ggml_cuda_device_malloc(&p, want, s.device) == cudaSuccess) {
            s.ring = (char *)p;
            s.ring_bytes = want;
            s.head = 0;
            s.n_lend_alloc++;
            g_pxa_stream_lent_live.store(true);
            if (s.n_lend_alloc == 1 || pxa_stream_log()) {
                GGML_CUDA_LOG_INFO("PXA_STREAM: device %d: lent ring taken back for a streaming graph: %.1f of %.1f MiB (%.0f MiB free before)\n",
                        s.device, want/1048576.0, s.lend_want/1048576.0, fr/1048576.0);
            }
            return true;
        }
        cudaGetLastError();
    }
    return false;
}

// return a lent ring that a graph took (the caller has synchronised the device's streams)
static void pxa_stream_lend_return(pxa_stream_dev & s) {
    if (!s.lent || !s.ring || s.active) return;
    ggml_cuda_set_device(s.device);
    CUDA_CHECK(cudaStreamSynchronize(s.cstream));
    for (auto & r : s.live) s.ev_put(r.rel_ev);
    s.live.clear();
    s.head = 0;
    CUDA_CHECK(cudaFree(s.ring));
    s.ring = nullptr;
    s.ring_bytes = 0;
}

static void pxa_stream_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (!g_pxa_stream_any.load(std::memory_order_relaxed)) return;
    const int dev = ctx.device;
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return;
    pxa_stream_dev & s = g_pxa_stream[dev];
    s.active = false;
    if (!s.ring && !s.lent) return;
    s.plan.clear(); s.by_last.clear(); s.restore.clear();
    s.issue_idx = s.wait_idx = s.rel_idx = 0;

    // first/last use of every streamed base tensor, plus every view of it the graph reads
    bool all_zc = true;   // every read fits its buffer's zero-copy width (pxa_stream_node_zc)
    std::vector<std::pair<ggml_tensor *, int>> idx;   // base -> plan index (linear search; tens of entries per graph)
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * n = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = n->src[j];
            ggml_tensor * b = pxa_stream_base_of(src);
            if (!b) continue;
            if (all_zc && !pxa_stream_node_zc(n, b)) all_zc = false;
            int pi = -1;
            for (int k = (int)idx.size() - 1; k >= 0 && k >= (int)idx.size() - 64; --k) if (idx[k].first == b) { pi = idx[k].second; break; }
            if (pi < 0) for (auto & pr : idx) if (pr.first == b) { pi = pr.second; break; }
            if (pi < 0) {
                pxa_stream_dev::item it{};
                it.base = b; it.first = i; it.last = i;
                it.bytes = ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
                it.state = 0; it.zc = false; it.off = 0; it.done_ev = nullptr; it.released = false;
                s.plan.push_back(std::move(it));
                pi = (int)s.plan.size() - 1;
                idx.push_back({b, pi});
            }
            auto & it = s.plan[pi];
            it.last = i;
            if (src != b) {
                bool seen = false;
                for (auto & v : it.views) if (v.t == src) { seen = true; break; }
                if (!seen) it.views.push_back({src, (size_t)((char *)src->data - (char *)b->data)});
            }
        }
    }
    if (s.plan.empty()) return;
    s.n_graphs++;
    s.active = true;

    // PXA_XCACHE_RING_LEND: a graph that copies needs the lent ring back; none to be had -> every read in place
    if (!all_zc && !s.ring) {
        size_t mx = 0;
        for (const auto & it : s.plan) mx = std::max(mx, it.bytes);
        if (!pxa_stream_lend_alloc(s, mx)) {
            s.n_lend_fail++;
            if (s.n_lend_fail == 1 || pxa_stream_log()) {
                GGML_CUDA_LOG_WARN("PXA_STREAM: device %d: the lent ring could not be taken back (%s); this graph reads its %zu streamed "
                        "weights in place over the link (zero-copy, as PXA_STREAM_ZC_NY would) [%lld such graphs]\n", dev,
                        pxa_stream_lend_fail_test() ? "PXA_XCACHE_RING_LEND_FAIL" : "not enough free VRAM", s.plan.size(), s.n_lend_fail);
            }
            all_zc = true;
        }
    }

    // decode-width graphs: read the pinned bytes in place instead of copying them
    if (all_zc) {
        for (auto & it : s.plan) { it.zc = true; it.state = 2; it.released = true; }
        s.issue_idx = s.wait_idx = s.plan.size();
        s.n_zc_graphs++;
        return;
    }

    s.by_last.resize(s.plan.size());
    for (size_t k = 0; k < s.plan.size(); ++k) s.by_last[k] = (int)k;
    std::sort(s.by_last.begin(), s.by_last.end(), [&](int a, int b) { return s.plan[a].last < s.plan[b].last; });

    // make the copy stream see everything the compute stream did before this graph (the weights
    // are immutable, but the ring regions freed at the end of the previous graph are not)
    GGML_UNUSED(ctx);
}

static void pxa_stream_walk(ggml_backend_cuda_context & ctx, int i) {
    const int dev = ctx.device;
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return;
    pxa_stream_dev & s = g_pxa_stream[dev];
    if (!s.active) return;
    cudaStream_t compute = ctx.stream();
    const int la = pxa_stream_lookahead();

    // 1. release regions whose last consumer has been enqueued (nodes < i). A fusion launched at
    //    node k only reads srcs of nodes >= k, so anything with last use < i is done being enqueued.
    while (s.rel_idx < s.by_last.size() && s.plan[s.by_last[s.rel_idx]].last < i) {
        pxa_stream_release(s, s.plan[s.by_last[s.rel_idx]], compute);
        s.rel_idx++;
    }
    // 2. issue as far ahead as the ring allows; anything needed within the lookahead that does not
    //    fit is read in place (zero-copy) instead.
    while (s.issue_idx < s.plan.size()) {
        auto & it = s.plan[s.issue_idx];
        if (it.state != 0) { s.issue_idx++; continue; }
        if (pxa_stream_issue_one(s, it)) { s.issue_idx++; continue; }
        if (it.first <= i + la) {
            it.zc = true; it.state = 2;
            s.n_zc++;
            s.issue_idx++;
            continue;
        }
        break;
    }
    // 3. the compute stream waits for every copy consumed within the lookahead
    while (s.wait_idx < s.plan.size() && s.plan[s.wait_idx].first <= i + la) {
        auto & it = s.plan[s.wait_idx];
        if (it.state == 1) {
            CUDA_CHECK(cudaStreamWaitEvent(compute, it.done_ev, 0));
            s.ev_put(it.done_ev); it.done_ev = nullptr;
            it.state = 2;
        }
        s.wait_idx++;
    }
}

static void pxa_stream_end(ggml_backend_cuda_context & ctx) {
    const int dev = ctx.device;
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return;
    pxa_stream_dev & s = g_pxa_stream[dev];
    if (!s.active) return;
    cudaStream_t compute = ctx.stream();
    for (auto & it : s.plan) {
        if (it.state == 1) {   // issued but never waited (cannot happen: every item is used) -- keep ordering sane
            CUDA_CHECK(cudaStreamWaitEvent(compute, it.done_ev, 0));
            s.ev_put(it.done_ev); it.done_ev = nullptr;
            it.state = 2;
        }
        pxa_stream_release(s, it, compute);
    }
    for (auto it = s.restore.rbegin(); it != s.restore.rend(); ++it) it->first->data = it->second;
    s.restore.clear();
    s.plan.clear();
    s.active = false;
    if (pxa_stream_log()) {
        static bool reg = false;
        if (!reg) { reg = true; atexit(pxa_stream_atexit); }
        if ((s.n_graphs & 63) == 1) pxa_stream_stats(s, "periodic");
    }
}
