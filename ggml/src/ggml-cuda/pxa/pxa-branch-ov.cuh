#pragma once
// PXA_DN_BA_OVERLAP (2026-10-20; default 0 = off (measured -10% on 27B 2x P100), 1 = decode rows, 2 = any width; sm_60).
//
// A delta-net layer's input projections read the same activation: the PXQN qkv(+z) GEMV (the persistent row-block
// kernel, ~55 us per card on the 2x P100 27B split) and the small beta/alpha GEMV right after it (one k_kq_mmv launch of
// 12 blocks, ~14 us, pure latency). Nothing orders them but the stream. Here the beta/alpha node(s) run on a side
// stream forked before the qkv launch and joined right after it, so the small launch hides under the big one (both fit
// on an SM together: 2 row-block blocks of 256 threads at 96 registers + one 128-thread kq block). Same kernels, same
// inputs, same outputs: bit-identical. Under CUDA graph capture the fork/join become graph edges.
//
// Taken only when (all checked here, else the graph runs as before):
//   - the beta/alpha MUL_MATs follow the qkv node with nothing but no-op views and PXQN siblings on the same activation
//     (the z GEMV the qkv launch groups) in between, and their inputs are not produced by those nodes;
//   - their outputs overlap no tensor those nodes read or write (the allocator may hand a buffer whose last reader is the
//     qkv GEMV to a later node: sequential order made that safe, concurrency would not);
//   - no bias ADD follows them (the kq dispatcher would fold it and write a tensor not checked here).
// The side window holds the context's GEMM pool out of reach: if the side path allocates anyway, main joins at once
// (no overlap, still in order) and the overlap is not tried again.

#define PXA_OV_SIDX (GGML_CUDA_MAX_STREAMS - 1)

static int pxa_ba_ov_mode() {
    static const int v = [] {
        const char * e = getenv("PXA_DN_BA_OVERLAP");
        return e && *e ? atoi(e) : 0;   // default OFF: measured -10% (job3 f6 17:38, P56 quiet)
    }();
    return v;
}
static std::atomic<bool> pxa_ba_ov_off{false};   // set once the side path misbehaved (allocated, consumed too much)
// per device (the graph thread of that device's context): the fork/join events, the side-launched node range of the
// current graph, and whether a node runs on the side stream right now. Nothing is added to ggml_backend_cuda_context: the
// side window swaps the context's stream 0 handle with the side stream's, so every launch of the node(s) computed there
// (ctx.stream() in any TU) goes to the side stream, and swaps back after.
struct pxa_ov_state {
    cudaEvent_t ev[2] = { nullptr, nullptr };
    const void * ctx = nullptr, * g = nullptr;
    int k0 = -1, k1 = -1;
    bool side = false;
};
static pxa_ov_state pxa_ov_st[GGML_CUDA_MAX_DEVICES];

static bool pxa_ba_weight(const ggml_tensor * w) {
    return w && (strstr(w->name, "ssm_beta") || strstr(w->name, "ssm_alpha"));
}
static bool pxa_ov_viewlike(const ggml_tensor * n) {
    return ggml_is_empty(n) || n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE;
}
static bool pxa_ov_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst, const ggml_cgraph * cgraph, int & i);

// at node i (a PXQN MUL_MAT about to launch on the main stream): launch the beta/alpha node(s) that follow it on the side
// stream. true = a join is owed after node i's launch (pxa_ba_overlap_join); the launched nodes are skipped later.
static bool pxa_ba_overlap_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    const int mode = pxa_ba_ov_mode();
    pxa_ov_state & st = pxa_ov_st[ctx.device];
    if (mode <= 0 || !cgraph || pxa_ba_ov_off.load(std::memory_order_relaxed) || st.k0 >= 0 || st.side) return false;
    if (ggml_cuda_info().devices[ctx.device].cc != 600 || pxa_x_cache_on()) return false;
    const ggml_tensor * mm = cgraph->nodes[i];
    const ggml_tensor * x  = mm->src[1];
    if (!mm->src[0] || !x || !ggml_cuda_pxqn_type(mm->src[0]->type)) return false;
    if (mode == 1 && x->ne[1] != 1) return false;
    int k0 = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 8; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (pxa_ov_viewlike(n)) continue;
        if (n->op == GGML_OP_MUL_MAT && pxa_ba_weight(n->src[0])) { k0 = j; break; }
        if (n->op == GGML_OP_MUL_MAT && n->src[1] == x && n->src[0] && ggml_cuda_pxqn_type(n->src[0]->type)) continue;
        break;
    }
    // a PXQN beta/alpha weight could be grouped into the main launch itself: not this path
    if (k0 < 0 || ggml_cuda_pxqn_type(cgraph->nodes[k0]->src[0]->type)) return false;
    const ggml_tensor * xb = cgraph->nodes[k0]->src[1];
    if (!xb || (mode == 1 && xb->ne[1] != 1)) return false;
    int k1 = k0;
    for (int j = k0 + 1; j < cgraph->n_nodes && j <= k0 + 6; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (pxa_ov_viewlike(n)) continue;
        if (n->op == GGML_OP_MUL_MAT && n->src[1] == xb && pxa_ba_weight(n->src[0])) { k1 = j; continue; }
        break;
    }
    // no bias ADD folded behind a beta/alpha node; inputs not produced, outputs not overlapping, by nodes [i, k0)
    for (int j = k0; j <= k1; ++j) {
        const ggml_tensor * b = cgraph->nodes[j];
        if (b->op != GGML_OP_MUL_MAT) continue;
        if (j + 1 < cgraph->n_nodes && cgraph->nodes[j + 1]->op == GGML_OP_ADD && cgraph->nodes[j + 1]->src[0] == b) return false;
        for (int m = i; m < k0; ++m) {
            const ggml_tensor * n = cgraph->nodes[m];
            if (n == xb || xb->view_src == n || n == b->src[0]) return false;
            if (pxa_ov_overlap(b, n) || (pxa_ov_overlap(xb, n) && !pxa_ov_viewlike(n))) return false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) if (pxa_ov_overlap(b, n->src[s])) return false;
        }
    }
    if (!st.ev[0]) {
        CUDA_CHECK(cudaEventCreateWithFlags(&st.ev[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&st.ev[1], cudaEventDisableTiming));
    }
    cudaStream_t s_main = ctx.stream(ctx.device, 0), s_side = ctx.stream(ctx.device, PXA_OV_SIDX);
    CUDA_CHECK(cudaEventRecord(st.ev[0], s_main));
    CUDA_CHECK(cudaStreamWaitEvent(s_side, st.ev[0], 0));
    std::unique_ptr<ggml_cuda_pool> held = std::move(ctx.pools[ctx.device]);   // the main pool is out of reach meanwhile
    std::swap(ctx.streams[ctx.device][0], ctx.streams[ctx.device][PXA_OV_SIDX]);  // ctx.stream() = the side stream
    st.side = true;
    // the side launches see the graph cut after k1, so no dispatcher chain or fusion there reaches past the checked
    // nodes; a chain that stops short is continued node by node up to k1
    ggml_cgraph sub = *cgraph;
    sub.n_nodes = k1 + 1;
    int kk = k0;
    bool ok = ggml_cuda_compute_forward(ctx, sub.nodes[k0], &sub, kk);
    while (ok && kk < k1) {
        ++kk;
        ok = ggml_cuda_compute_forward(ctx, sub.nodes[kk], &sub, kk);
    }
    st.side = false;
    std::swap(ctx.streams[ctx.device][0], ctx.streams[ctx.device][PXA_OV_SIDX]);
    const bool used_pool = ctx.pools[ctx.device] != nullptr;
    if (used_pool) {   // kept for the process's life: graphs captured meanwhile may point into it
        static std::vector<std::unique_ptr<ggml_cuda_pool>> side_pools;
        side_pools.push_back(std::move(ctx.pools[ctx.device]));
    }
    ctx.pools[ctx.device] = std::move(held);
    CUDA_CHECK(cudaEventRecord(st.ev[1], s_side));
    GGML_ASSERT(ok);
    st.ctx = &ctx; st.g = cgraph; st.k0 = k0; st.k1 = kk;
    static std::atomic<uint32_t> seen{0};
    const uint32_t bit = 1u << (ctx.device & 31);
    if (used_pool || kk != k1) {
        // not the expected side path: join now (in order, nothing overlaps; the record still skips [k0, kk]) and never again
        CUDA_CHECK(cudaStreamWaitEvent(s_main, st.ev[1], 0));
        pxa_ba_ov_off.store(true, std::memory_order_relaxed);
        fprintf(stderr, "PXA_DN_BA_OVERLAP dev%d: side path %s at node %d (%s): overlap off from here\n", ctx.device,
                used_pool ? "allocated from the pool" : "ended off the beta/alpha range", k0, cgraph->nodes[k0]->name);
        return false;
    }
    if (!(seen.fetch_or(bit) & bit)) {
        fprintf(stderr, "PXA_DN_BA_OVERLAP dev%d: beta/alpha GEMV (%s, nodes %d..%d) on a side stream under the %s launch "
                        "(bit-identical; PXA_DN_BA_OVERLAP=0 off)\n", ctx.device, cgraph->nodes[k0]->name, k0, kk, mm->name);
    }
    return true;
}

// after node i's launch(es) on the main stream (i_end = the last node they consumed)
static void pxa_ba_overlap_join(ggml_backend_cuda_context & ctx, const int i_end) {
    pxa_ov_state & st = pxa_ov_st[ctx.device];
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(ctx.device, 0), st.ev[1], 0));
    if (i_end >= st.k0) {   // the main launch consumed the overlapped nodes as well (not seen): stop
        pxa_ba_ov_off.store(true, std::memory_order_relaxed);
        fprintf(stderr, "PXA_DN_BA_OVERLAP dev%d: the main launch reached node %d too: overlap off from here\n", ctx.device, st.k0);
        st.k0 = -1;
    }
}

// top of compute_forward: a node already launched on the side stream is skipped (stale records are dropped)
static inline bool pxa_ba_overlap_skip(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int & i) {
    pxa_ov_state & st = pxa_ov_st[ctx.device];
    if (st.k0 < 0 || st.side) return false;
    if (st.ctx == &ctx && st.g == cgraph && i == st.k0) {
        i = st.k1;
        st.k0 = -1;
        return true;
    }
    if (st.ctx != &ctx || st.g != cgraph || i > st.k1) st.k0 = -1;
    return false;
}
