// pxa_expert_shard.cuh — PXA-SHARD M2: the sharded MoE op path.
//
// Shards the top-8-of-256 expert compute of ONE fused-MoE layer across a
// MATCHED, P2P-connected device group (the V100 pair for M2; the P100 quad in
// M4). Experts [k*EP .. (k+1)*EP) live on group device k (EP = n_expert/n_shard)
// — placed there by the M3 loader (prepare_split_tensors(2,...) → the expert
// tensor carries a ggml_split_tensor_t extra whose splits[k]->data is device k's
// resident expert slice). Because top-8 routing writes ONE disjoint dst row per
// (token,slot), each device owns a DISJOINT set of output rows: there is NO
// reduction. Each group device runs its owned expert bins with the ENGINE's OWN
// bit-exact vec_dot math (reused from grouped_moe_verify.cuh — same q8_1
// activations, same MXFP4 dot, same SILU epilogue), writing its disjoint rows
// DIRECTLY into the home-device dst over P2P; one event sync per MoE wall.
//
// ⭐ CORRECTNESS (the M2 gate): the per-member accumulation is byte-identical to
// the per-token reference — only ADDRESSING differs (expert base per device,
// owning-device filter, RT-1 scatter key). So the exact-equality shadow diff
// (PXA_MOE_GROUPED_VERIFY=1) must report ZERO mismatch. This is the same
// bit-exact provenance as gk6; PXA-SHARD only moves WHERE the identical math
// runs (across 2 cards instead of 1), not WHAT it computes.
//
// FLAG-GATED: nothing routes an expert tensor into the shard buffer type unless
// PXA_EXPERT_SHARD is set (M3 loader). With the flag unset this file's host
// entry is never called (the op-path branch is guarded on
// pxa_buft_is_expert_shard(src0_1->buffer->buft), which is false for every
// non-shard buft), so the binary is bit-identical to the flag-off build.
//
// Design notes (load-bearing):
//  * DIRECT PEER WRITE, no gather copy: with peer access enabled, device k's
//    kernel scatters straight into the home dst pointer (UVA). Disjoint rows ⇒
//    no clobber, no all-reduce, no staging buffer. Just an event per peer so the
//    home stream sees a complete dst.
//  * The plan (pxa_build_plan CSR) + the q8_1 activation columns live on the
//    home device and are BROADCAST once per layer to each non-home peer (tiny:
//    <=64 ints + Ny*ddq bytes). Each device's kernel then reads its LOCAL copies
//    (avoids d_ff peer reads of the activation column in the GEMV hot loop).
//  * Capture guard: peer copies + the pool allocs are not CUDA-graph-capture
//    safe; inside a capture we return false and the caller runs the per-token
//    path (graphs buy ~0 on this rig — measured).
#pragma once
#include <cuda_runtime.h>
#include "../grouped_moe_verify.cuh"   // pxa_build_plan, pxa_mxfp4_d, block_q8_1 dot reuse, shadow diff

// ---------------------------------------------------------------------------
// env: PXA_EXPERT_SHARD set ⇒ the shard op path is live (the M3 loader also
// gates the weight placement on the same var). Read once. Default OFF.
// ---------------------------------------------------------------------------
static inline bool pxa_expert_shard_enabled() {
    // 2026-07-30: was presence-tested, so PXA_EXPERT_SHARD=0 armed the shard OP path while
    // the M3 loader (src/llama.cpp) correctly refused to shard (it needs a >=2-device group).
    // Mirror the loader semantic: live only when the value parses to >= 2 group devices.
    static const bool v = [](){
        const char * e = getenv("PXA_EXPERT_SHARD");
        if (!e) return false;
        int n = 0; bool in_tok = false;
        for (const char * p = e; ; ++p) {
            if (*p == ',' || *p == 0) { if (in_tok) ++n; in_tok = false; if (*p == 0) break; }
            else if (*p != ' ') in_tok = true;
        }
        return n >= 2;
    }();
    return v;
}

// ---------------------------------------------------------------------------
// PXA_SHARD_TIMING: per-layer MoE up+gate op GPU-time meter (the M2 mechanism
// gate — immune to sampler/prompt/MTP/harness noise). cudaEventElapsedTime
// measures PURE GPU time between two stream events, so the per-op host sync used
// to read it does NOT dilute the measured op time. Runs on both paths: flag-off
// times the UNSHARDED op, flag-on times the SHARDED op, on the SAME V100 layer
// (filter by device). Ratio = unsharded/sharded on dev 0/1 = the mechanism win.
// ---------------------------------------------------------------------------
static inline bool pxa_shard_timing() {
    static const bool v = getenv("PXA_SHARD_TIMING") != nullptr;
    return v;
}
struct pxa_shard_tmr_t { cudaEvent_t a=nullptr,b=nullptr; double sum=0; long n=0; };
static inline pxa_shard_tmr_t & pxa_shard_tmr(int dev) { static pxa_shard_tmr_t t[16]; return t[dev & 15]; }
static inline void pxa_shard_time_begin(int dev, cudaStream_t s) {
    if (!pxa_shard_timing()) return;
    auto & t = pxa_shard_tmr(dev);
    if (!t.a) { cudaEventCreate(&t.a); cudaEventCreate(&t.b); }
    cudaEventRecord(t.a, s);
}
static inline void pxa_shard_time_end(int dev, cudaStream_t s, const char * tag) {
    if (!pxa_shard_timing()) return;
    auto & t = pxa_shard_tmr(dev);
    if (!t.a) return;
    cudaEventRecord(t.b, s);
    cudaEventSynchronize(t.b);
    float ms = 0.f; cudaEventElapsedTime(&ms, t.a, t.b);
    t.sum += ms; t.n++;
    if (t.n % 200 == 0)
        fprintf(stderr, "PXA_SHARD_TIMING %-9s dev=%d moe_upgate_avg=%.5f ms (n=%ld)\n", tag, dev, t.sum/t.n, t.n);
}

// ---------------------------------------------------------------------------
// Sharded up+gate kernel: gk6's bit-exact engine-vec_dot instantiation, plus
//   (a) expert_base  : local expert index = g_experts[u] - expert_base
//                      (device k holds global experts [k*EP..(k+1)*EP) as local
//                       0..EP-1), and
//   (b) an OWNING-DEVICE FILTER: this launch (on device k) processes ONLY bins
//       whose expert is owned by k. All group devices launch over the SAME
//       oversubscribed grid (grid.y = NN bins); non-owned bins early-out. This
//       reuses the single home-built plan (broadcast to each device) without a
//       host sync or per-device plan compaction.
// The float accumulation is byte-identical to pxa_gk6_gateup_iqk (and thus to
// the per-token reference); the ONLY deltas are the two addressing changes
// above. dst_base is the HOME dst pointer (direct P2P scatter).
// ---------------------------------------------------------------------------
template<int MAXC>
__global__ void pxa_shard_gateup_iqk(
        const void* __restrict__ up_base,      // device-k local UP slice (splits[k]->data)
        const void* __restrict__ gate_base,    // device-k local GATE slice
        int64_t w_nb1 /*row_size bytes*/, int64_t w_nb2 /*expert stride bytes*/,
        const void* __restrict__ vy_base,      // device-k local q8_1 activations, all Ny cols
        int blocks_per_col_y,
        const int* __restrict__ g_experts, const int* __restrict__ g_offsets,
        const int* __restrict__ g_tok, const int* __restrict__ g_slot,
        const int* __restrict__ n_union,
        float* __restrict__ dst_base, int64_t dst_nb2_f /*floats per token (HOME dst)*/,
        int ncols_x /*d_model*/, int d_ff, float limit,
        int expert_base, int experts_per_shard) {

    constexpr int qk  = ggml_cuda_type_traits<GGML_TYPE_MXFP4>::qk;   // 32
    constexpr int qi  = ggml_cuda_type_traits<GGML_TYPE_MXFP4>::qi;
    constexpr int vdr = VDR_MXFP4_Q8_1_MMVQ;
    constexpr int nwarps = 1;
    constexpr int rows_per_cuda_block = 1;

    const int u = blockIdx.y;
    if (u >= *n_union) return;
    const int m0 = g_offsets[u], m = g_offsets[u+1] - m0;
    if (m <= 0) return;
    const int e = g_experts[u];
    if (e < 0) return;                                   // SER guard
    // OWNING-DEVICE FILTER: skip bins this device does not own.
    if (e / experts_per_shard != expert_base / experts_per_shard) return;
    const int e_local = e - expert_base;                 // local expert index on this device

    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block*blockIdx.x;
    if (row0 >= d_ff) return;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter  = vdr*nwarps*WARP_SIZE / qi;

    const char* vup = (const char*)up_base   + (int64_t)e_local*w_nb2 + (int64_t)row0*w_nb1;
    const char* vgt = (const char*)gate_base + (int64_t)e_local*w_nb2 + (int64_t)row0*w_nb1;
    const block_q8_1* y = (const block_q8_1*)vy_base;

    float tmp_u[MAXC]; float tmp_g[MAXC];
    #pragma unroll
    for (int j = 0; j < MAXC; ++j) { tmp_u[j] = 0.0f; tmp_g[j] = 0.0f; }

    for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(tid % (qi/vdr));
        #pragma unroll
        for (int j = 0; j < MAXC; ++j) {
            if (j >= m) break;
            const int t = g_tok[m0+j];
            const block_q8_1* yj = y + (int64_t)t*blocks_per_col_y + kby;
            tmp_u[j] += vec_dot_mxfp4_q8_1((const void*)vup, yj, kbx, kqs);
            tmp_g[j] += vec_dot_mxfp4_q8_1((const void*)vgt, yj, kbx, kqs);
        }
    }
    #pragma unroll
    for (int j = 0; j < MAXC; ++j) {
        if (j >= m) break;
        tmp_u[j] = warp_reduce_sum(tmp_u[j]);
        tmp_g[j] = warp_reduce_sum(tmp_g[j]);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int j = 0; j < MAXC; ++j) {
            if (j >= m) break;
            // ENGINE SILU epilogue VERBATIM: u=up, g=gate (sanitized limit).
            float uu = tmp_u[j], g = tmp_g[j];
            g = g/(1.0f + expf(-g));
            g = fminf(g, limit);
            const float r = fmaxf(-limit, fminf(limit, uu)) * g;
            const int key = g_slot[m0+j];                // (t<<16)|s
            dst_base[(size_t)(key >> 16) * dst_nb2_f + (size_t)(key & 0xffff) * d_ff + row0] = r;
        }
    }
}

// Per-device slice bases + group come from the shard buffer context (NOT ->extra
// — extra stays NULL so the fused MoE path is kept). Provided in-TU by
// pxa_expert_shard_tensor_info() (defined in ggml-cuda.cu before this include).
GGML_CALL bool pxa_expert_shard_tensor_info(const ggml_tensor * t, void ** bases, int * group, int * n_shard);

// PXA-SHARD ntok>1 reconstruct: rebuild the FULL n_expert weight on `home` from the
// per-device slices so the stock (ntok>1) MMQ can read a complete tensor (no OOB).
// Each slot k's slice (EP experts, stride nb2) is placed at the GLOBAL range [k*EP,(k+1)*EP).
static inline bool pxa_shard_reconstruct(ggml_backend_cuda_context & ctx, const ggml_tensor * root,
        ggml_cuda_pool_alloc<char> & full, int home, cudaStream_t stream) {
    void * bases[GGML_CUDA_MAX_DEVICES]; int grp[GGML_CUDA_MAX_DEVICES]; int ns = 0;
    if (!pxa_expert_shard_tensor_info(root, bases, grp, &ns) || ns < 2) return false;
    const int n_expert = (int)root->ne[2];
    if (n_expert % ns != 0) return false;
    const int EP = n_expert / ns;
    const size_t es = (size_t)root->nb[2];
    full.alloc(ctx.pool(home), (size_t)n_expert * es);
    char * base = full.get();
    for (int k = 0; k < ns; ++k) {
        char * dstk = base + (size_t)k * EP * es;
        const size_t bytes = (size_t)EP * es;
        if (grp[k] == home) CUDA_CHECK(cudaMemcpyAsync (dstk, bases[k], bytes, cudaMemcpyDeviceToDevice, stream));
        else                CUDA_CHECK(cudaMemcpyPeerAsync(dstk, home, bases[k], grp[k], bytes, stream));
    }
    return true;
}
// RAII (default-constructible): temporarily repoint a tensor's ->data; restore on scope exit.
struct pxa_data_swap {
    ggml_tensor * t = nullptr; void * orig = nullptr;
    void set(const ggml_tensor * t_, void * nd) { t = const_cast<ggml_tensor*>(t_); orig = t_->data; t->data = nd; }
    ~pxa_data_swap() { if (t) t->data = orig; }
};
// Fix B merge: dst[i] += src[i] (add a peer's disjoint output rows, gathered home).
__global__ void pxa_add_into(float * __restrict__ dst, const float * __restrict__ src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ---------------------------------------------------------------------------
// Host entry for the sharded up+gate. Returns true if it produced dst (skip the
// per-token iy-loop). Bit-exact by construction (see header). Falls back
// (returns false) on any geometry it doesn't cover, inside a graph capture, or
// if peer access can't be established — the caller then runs the per-token path.
//
// dst is the HOME up_gate output; each device scatters its owned rows into it
// over P2P. In shadow mode it writes a private HOME scratch and returns false so
// the per-token path still fills dst; pxa_moe_grouped_shadow_check() then diffs.
// ---------------------------------------------------------------------------
static inline bool pxa_moe_shard_gateup(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_1 /*UP, shard*/, const ggml_tensor * src0_2 /*GATE, shard*/,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        int Ny, int n_ids, float limit,
        const char * src1_q, size_t src1_ddq_size,
        cudaStream_t stream) {

    if (!pxa_expert_shard_enabled()) return false;
    // capture guard (peer copies + pool allocs not capture-safe)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) return false;
    }
    // SILU limit sanitize (mmvq.cu:45 parity — qwen35moe leaves op_params[1]=0).
    limit = limit > 1e-6f ? limit : INFINITY;
    // resolve views to the ROOT weights for correct ne/nb (a view may be re-based)
    { const ggml_tensor * r1 = src0_1; while (r1->view_src) r1 = r1->view_src; src0_1 = r1;
      const ggml_tensor * r2 = src0_2; while (r2->view_src) r2 = r2->view_src; src0_2 = r2; }

    // group membership + per-device bases from the shard buffer context (up &
    // gate must map to the same group). Each device k holds experts
    // [k*EP..(k+1)*EP); base_u[k]/base_g[k] point at that slice's expert 0.
    void * base_u[GGML_CUDA_MAX_DEVICES], * base_g[GGML_CUDA_MAX_DEVICES];
    int    grp_u[GGML_CUDA_MAX_DEVICES],  grp_g[GGML_CUDA_MAX_DEVICES];
    int n_shard = 0, n_shard_g = 0;
    if (!pxa_expert_shard_tensor_info(src0_1, base_u, grp_u, &n_shard))   return false;
    if (!pxa_expert_shard_tensor_info(src0_2, base_g, grp_g, &n_shard_g)) return false;
    if (n_shard < 2 || n_shard != n_shard_g) return false;
    for (int k = 0; k < n_shard; ++k) if (grp_u[k] != grp_g[k]) return false;

    const int n_expert = (int)src0_1->ne[2];
    if (n_expert % n_shard != 0) return false;           // uniform shards for M2
    const int EP = n_expert / n_shard;                   // experts per device

    const int home = ctx.device;
    // PXA-SHARD: only shard when THIS op's runtime home is a group device. An
    // over-selected layer (home outside the group) makes the group's kernels write
    // dst on a non-group device over a link with no P2P -> illegal access.
    { bool ig=false; for (int k=0;k<n_shard;++k) if (grp_u[k]==home){ig=true;break;}
      if (!ig) { if (getenv("PXA_MOE_DEBUG")) fprintf(stderr,"PXA-SHARD-GUARD home=%d NOT in group -> fallback\n", home); return false; } }
    const int d_model = (int)src0_1->ne[0];
    const int d_ff    = (int)dst->ne[0];
    const int NN      = Ny * n_ids;
    if (Ny < 1 || Ny > 8 || n_ids < 1 || n_ids > 8 || NN > 64) return false;
    if (d_model % 128 != 0) return false;
    if (Ny*(size_t)n_ids >= (1u<<16)) return false;

    // per-expert weight strides: the full-tensor nb apply to each device's slice
    // (same [d_model,d_ff] per expert; the slice just has fewer experts).
    const int64_t w_nb1 = (int64_t)src0_1->nb[1]; // row_size bytes
    const int64_t w_nb2 = (int64_t)src0_1->nb[2]; // expert stride bytes

    // ensure P2P among the group in BOTH directions: each group device must be
    // able to write home memory (its disjoint dst rows) AND read home for the
    // plan/activation broadcast. ggml_cuda_set_peer_access(d) enables d->others,
    // so enable it for every group member (idempotent; AlreadyEnabled is fine).
    for (int k = 0; k < n_shard; ++k) ggml_cuda_set_peer_access(grp_u[k]);
    ggml_cuda_set_device(home);

    pxa_shard_time_begin(home, stream);   // meter the sharded op (GPU time)

    // build the CSR plan on the HOME device (no host sync)
    ggml_cuda_pool_alloc<int> pl_exp(ctx.pool(home), NN);
    ggml_cuda_pool_alloc<int> pl_off(ctx.pool(home), NN+1);
    ggml_cuda_pool_alloc<int> pl_tok(ctx.pool(home), NN);
    ggml_cuda_pool_alloc<int> pl_key(ctx.pool(home), NN);
    ggml_cuda_pool_alloc<int> pl_nu (ctx.pool(home), 1);
    pxa_build_plan<<<1, 32, 0, stream>>>((const int*)ids->data, (int64_t)(ids->nb[1]/sizeof(int)),
            Ny, n_ids, pl_exp.get(), pl_off.get(), pl_tok.get(), pl_key.get(), pl_nu.get());
    CUDA_CHECK(cudaGetLastError());

    // output: a raw-cudaMalloc HOME staging buffer (NOT dst directly). PEER kernels
    // cannot store into the home dst — dst lives in the VMM pool, cuMemSetAccess'd only
    // for its owner; cudaDeviceEnablePeerAccess does NOT grant kernel peer access to VMM
    // allocs (only the copy engine crosses it). A raw cudaMalloc IS peer-kernel-writable.
    // Home + peers scatter into sh.scratch; a home-local D2D lands it in dst after the
    // peer sync. (Fix A; Fix B = bulk peer-gather is the honest op-ratio config.)
    const int64_t dst_nb2_f = (int64_t)(dst->nb[2]/sizeof(float));
    const size_t need = (size_t)Ny*dst_nb2_f*sizeof(float);
    auto & sh = pxa_moe_shadow_st(ctx.device);
    if (sh.cap < need) { if (sh.scratch) cudaFree(sh.scratch); cudaMalloc(&sh.scratch, need); sh.cap = need; }
    // seed staging with the current dst so untouched rows (SER -1) survive the copy-back
    // Fix B: non-shadow writes DIRECTLY into dst (zeroed first); home writes its rows,
    // each peer's disjoint rows arrive via a device-local staging gathered + added home.
    float * out;
    if (pxa_moe_grouped_shadow()) {
        cudaMemcpyAsync(sh.scratch, dst->data, need, cudaMemcpyDeviceToDevice, stream);
        out = (float*)sh.scratch;
    } else {
        cudaMemsetAsync(dst->data, 0, need, stream);
        out = (float*)dst->data;
    }
    if (pxa_moe_grouped_shadow()) {
        if (!sh.viol) cudaMalloc(&sh.viol, sizeof(int));
        cudaMemsetAsync(sh.viol, 0, sizeof(int), stream);
        sh.pending = true; sh.Ny = Ny; sh.n_ids = n_ids; sh.d_ff = d_ff; sh.dst_nb2_f = dst_nb2_f; sh.exact = true;
    }

    const int blocks_per_col_y = (int)(src1_ddq_size / sizeof(block_q8_1));
    dim3 grid6(d_ff, NN);
    dim3 blk6(WARP_SIZE, 1);

    // Cross-stream ordering: the plan (pxa_build_plan) + the q8_1 quantization
    // (done by the caller on this same HOME stream) must complete before any peer
    // reads them over P2P. Record one home event and gate each peer stream on it.
    cudaEvent_t ev_home;
    cudaEventCreateWithFlags(&ev_home, cudaEventDisableTiming);
    cudaEventRecord(ev_home, stream);

    // Function-scope holders: default-constructed (the pool-alloc type is
    // move/copy-deleted, so no assignment) and alive until this function returns
    // (AFTER the sync) — the peer kernels read them asynchronously, so their
    // backing pool memory must outlive the loop iteration.
    ggml_cuda_pool_alloc<int>  bexp[GGML_CUDA_MAX_DEVICES], boff[GGML_CUDA_MAX_DEVICES],
                               btok[GGML_CUDA_MAX_DEVICES], bkey[GGML_CUDA_MAX_DEVICES],
                               bnu [GGML_CUDA_MAX_DEVICES];
    ggml_cuda_pool_alloc<char> bvy [GGML_CUDA_MAX_DEVICES];
    ggml_cuda_pool_alloc<float> lstage[GGML_CUDA_MAX_DEVICES];  // Fix B: peer-local output staging (device k)
    ggml_cuda_pool_alloc<float> hstage[GGML_CUDA_MAX_DEVICES];  // Fix B: home temp per peer for the gather
    cudaEvent_t ev[GGML_CUDA_MAX_DEVICES];

    for (int k = 0; k < n_shard; ++k) {
        const int dev = grp_u[k];
        ggml_cuda_set_device(dev);
        cudaStream_t dstream = (dev == home) ? stream : ctx.stream(dev, 0);
        const void * vy = src1_q;      // home copy (used directly on the home device)
        const int * p_exp = pl_exp.get(), * p_off = pl_off.get(), * p_tok = pl_tok.get(),
                  * p_key = pl_key.get(), * p_nu = pl_nu.get();

        if (dev != home) {
            // broadcast the plan + q8_1 activations to the peer (tiny), then the
            // peer reads its LOCAL copies (keeps the GEMV hot loop device-local).
            cudaStreamWaitEvent(dstream, ev_home, 0);
            bexp[k].alloc(ctx.pool(dev), NN);
            boff[k].alloc(ctx.pool(dev), NN+1);
            btok[k].alloc(ctx.pool(dev), NN);
            bkey[k].alloc(ctx.pool(dev), NN);
            bnu [k].alloc(ctx.pool(dev), 1);
            bvy [k].alloc(ctx.pool(dev), (size_t)Ny*src1_ddq_size);
            cudaMemcpyPeerAsync(bexp[k].get(), dev, pl_exp.get(), home, NN*sizeof(int),   dstream);
            cudaMemcpyPeerAsync(boff[k].get(), dev, pl_off.get(), home, (NN+1)*sizeof(int),dstream);
            cudaMemcpyPeerAsync(btok[k].get(), dev, pl_tok.get(), home, NN*sizeof(int),   dstream);
            cudaMemcpyPeerAsync(bkey[k].get(), dev, pl_key.get(), home, NN*sizeof(int),   dstream);
            cudaMemcpyPeerAsync(bnu [k].get(), dev, pl_nu.get(),  home, sizeof(int),       dstream);
            cudaMemcpyPeerAsync(bvy [k].get(), dev, (const void*)src1_q, home, (size_t)Ny*src1_ddq_size, dstream);
            vy = bvy[k].get(); p_exp = bexp[k].get(); p_off = boff[k].get(); p_tok = btok[k].get();
            p_key = bkey[k].get(); p_nu = bnu[k].get();
        }

        // Fix B: home writes `out` directly (local); peer writes a device-LOCAL zeroed
        // staging (no P2P store in the kernel — that was the illegal VMM peer write).
        float * out_k;
        if (dev == home) { out_k = out; }
        else {
            lstage[k].alloc(ctx.pool(dev), (size_t)(need/sizeof(float)));
            cudaMemsetAsync(lstage[k].get(), 0, need, dstream);
            out_k = lstage[k].get();
        }
        auto launch = [&](auto maxc_tag) {
            constexpr int MAXC = decltype(maxc_tag)::value;
            pxa_shard_gateup_iqk<MAXC><<<grid6, blk6, 0, dstream>>>(
                    base_u[k], base_g[k], w_nb1, w_nb2,
                    vy, blocks_per_col_y,
                    p_exp, p_off, p_tok, p_key, p_nu,
                    out_k, dst_nb2_f, d_model, d_ff, limit,
                    /*expert_base=*/ k*EP, /*experts_per_shard=*/ EP);
        };
        if      (Ny <= 2) launch(std::integral_constant<int,2>{});
        else if (Ny <= 4) launch(std::integral_constant<int,4>{});
        else              launch(std::integral_constant<int,8>{});
        CUDA_CHECK(cudaGetLastError());

        if (dev != home) {
            cudaEventCreateWithFlags(&ev[k], cudaEventDisableTiming);
            cudaEventRecord(ev[k], dstream);
        }
    }

    // Fix B gather: pull each peer's device-local staging home (copy engine crosses VMM)
    // and ADD its disjoint rows into out. Disjoint + zeroed-elsewhere => the add is exact.
    ggml_cuda_set_device(home);
    const size_t out_floats = need/sizeof(float);
    for (int k = 0; k < n_shard; ++k) {
        if (grp_u[k] == home) continue;
        cudaStreamWaitEvent(stream, ev[k], 0);
        hstage[k].alloc(ctx.pool(home), out_floats);
        cudaMemcpyPeerAsync(hstage[k].get(), home, lstage[k].get(), grp_u[k], need, stream);
        const int TPB = 256;
        pxa_add_into<<<(unsigned)((out_floats+TPB-1)/TPB), TPB, 0, stream>>>(out, hstage[k].get(), out_floats);
        CUDA_CHECK(cudaGetLastError());
        cudaEventDestroy(ev[k]);
    }
    cudaEventDestroy(ev_home);
    pxa_shard_time_end(home, stream, "SHARDED");   // kernel + local writes + bulk gather+add
    return !pxa_moe_grouped_shadow();
}

// ---------------------------------------------------------------------------
// Top-level op dispatcher: called from ggml_cuda_moe_up_gate_unary when
// src0_1's buffer is an expert-shard buft (flag-on only). Mirrors the engine's
// fast-TG q8_1 quantization, runs the sharded up+gate, then the shadow check.
// Returns the graph index to continue from, or -1 to fall back to the stock op
// (geometry not covered / not shardable this step).
//
// ⚠ M2 SCOPE: this wires the UP+GATE shard (the manual's anchor). The fused
// down-proj shard (fuse_next block) is the symmetric "repeat 5-7 keyed on the
// SAME g_slot" and is the remaining M2 op-path piece; until it + the M3 loader
// (prepare_split_tensors(2) placement of the group's *_exps in the shard buft,
// in LAYER mode) land, this dispatcher is only reachable flag-on. Flag-off it is
// never called ⇒ the binary is bit-identical to the flag-off build.
// ---------------------------------------------------------------------------
static inline int pxa_moe_shard_up_gate_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];
    if (!src0_2 || !src1 || !ids) return -1;
    if (getenv("PXA_MOE_DEBUG")) { static int _e=0; if(_e++<50) fprintf(stderr,"PXA-DISP name=%s dev=%d ne1=%ld ne2=%ld ne3=%ld s1type=%d w1type=%d w2type=%d unary=%d\n", src0_1->name,(int)ctx.device,(long)src1->ne[1],(long)src1->ne[2],(long)src1->ne[3],(int)src1->type,(int)src0_1->type,(int)src0_2->type,(int)(ggml_unary_op)dst->op_params[0]); }
    // fast-TG shape only (single-token decode or Ny<=8 MTP verify)
    if (!(src1->ne[1] == 1 && src1->ne[2] <= 8 && src1->ne[3] == 1 && src1->type == GGML_TYPE_F32)) return -1;
    if (src0_1->type != GGML_TYPE_MXFP4 || src0_2->type != GGML_TYPE_MXFP4) return -1;
    auto unary_op = (ggml_unary_op)dst->op_params[0];
    if (unary_op != GGML_UNARY_OP_SILU) return -1;

    const int device_id = ctx.device;
    auto stream = ctx.stream(device_id, 0);
    const int64_t n_ids = ids->ne[0];
    const int Ny = (int)src1->ne[2];
    float limit = *(const float *)(dst->op_params + 1);

    // quantize activations to q8_1 (verbatim from the fast-TG path)
    const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
    auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
    src1_quantized.alloc(src_1_ddq_size * Ny);
    quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], Ny, 1,
            src1_padded_col_size, src0_1->type, stream);
    CUDA_CHECK(cudaGetLastError());

    const bool done = pxa_moe_shard_gateup(ctx, src0_1, src0_2, src1, ids, dst,
            Ny, (int)n_ids, limit, (const char *)src1_quantized.get(), src_1_ddq_size, stream);
    if (!done && !pxa_moe_grouped_shadow()) return -1;   // shard path declined ⇒ stock op

    pxa_moe_grouped_shadow_check(ctx, dst, stream);       // exact-equality diff if armed

    // NOTE (M2 remaining): the down-proj fuse_next block is NOT yet sharded here;
    // returning i lets the scheduler run the following down op on the stock path.
    // Correct flag-on end-to-end requires the symmetric down shard + M3 loader.
    return i;
}
