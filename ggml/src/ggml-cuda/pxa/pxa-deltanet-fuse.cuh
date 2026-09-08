// PXA_FUSE_DELTANET — R2 DeltaNet decode glue-kernel fusion (2026-07-07)
//
// The qwen35moe/qwen3next DeltaNet decode path (bs=1) runs a chain of small glue kernels
// around the fused delta_net_recurrent_f32 core, each paying a ~3-10us execution-latency
// floor on Pascal/Volta:
//
//   SILU(conv_out) -> L2_NORM(q) -> L2_NORM(k)          3 kernels -> 1  (pxa_dn_silu_qknorm_f32)
//   CONT(conv tail) + CONCAT(state writeback, 4.2MB x2) 2 kernels -> 1  (pxa_dn_conv_tail_f32)
//     + delta_net writes its new ssm state DIRECTLY into the cache row (state_dst override),
//       eliminating the CONCAT's 4.15MB read + 4.15MB write per layer per token entirely.
//   FUSED_RMS_NORM(ssm_norm) + FUSED_MUL_UNARY(silu z)  2 kernels -> 1  (pxa_dn_rms_silu_gate_f32)
//
//   GET_ROWS(state) + MUL(reset mask)                   2 kernels -> 1  (pxa_dn_gather_mask_f32)
//   CONCAT(new row) + SET_ROWS(scatter to cache)        2 kernels -> 0  (written in place)
//
// Env gate: PXA_FUSE_DELTANET (bitmask, default 3 = bits 0|1; =0 restores the eager path)
//   bit0 (1): the qk-norm/state-writeback cluster (anchored at the SILU node; consumes the
//             23-node SILU..CONCAT window incl. the already-fused ADD+SOFTPLUS+MUL beta-gate).
//             GUARDED: fires only on the exact in-place state alias (PXA_FUSE_DELTANET_WAR).
//   bit1 (2): the out-gate rms+silu fusion (anchored at FUSED_RMS_NORM). GUARDED: the
//             sole-consumer claim is verified against the graph, not asserted.
//   bit2 (4): the state-row gather+reset-mask fusion (anchored at GET_ROWS; both dsts written,
//             so it makes no dead-store claim). OFF by default -- measured never to match the
//             live decode graph; bit-exact and free, but unexercised. =7 re-arms it.
//   bit3 (8): LEGACY row absorb. Absorb the trailing SET_ROWS by writing the new row straight
//             into the cache row. OFF by default; it is held to pxa_dn_scatter_dst_safe(), the
//             same exact-alias predicate as bit0, so on the SAFE carry it can never fire. Kept
//             only as the historical reproducer; bit5 is the shippable form.
//   bit4 (16): SAFE-CARRY cluster (2026-09-03, decode2 lane). Exactly bit0's three kernels, but
//             the conv tail and the new ssm state are written into CC's OWN storage -- the
//             destination the graph allocated for the CONCAT -- instead of into the recurrent
//             cache row. Nothing is written outside the graph's dataflow, so the exact-alias
//             predicate is not needed; what IS needed, and IS checked, is that CC's storage is
//             disjoint from every tensor the surviving kernels read or write (ggml-alloc is free
//             to place CC over a dead parent), and that the CONT whose dst is left unwritten has
//             no consumer but the CONCAT. See pxa_dn_scratch_window_safe().
//   bit5 (32): PROVABLE row absorb, the safe-carry replacement for bit3. Requires bit0 or bit4.
//             Writes the packed row straight into the recurrent cache row and drops the trailing
//             SET_ROWS. Admitted only when a reader-set SCAN of the split graph proves that no
//             node strictly between this cluster and the SET_ROWS it absorbs touches the state
//             tensor's storage -- i.e. there is provably no reader of the old row contents in the
//             window the early write opens. See pxa_dn_row_window_clear(); the scan itself is
//             printable with PXA_DN_ROWSCAN=1, which NAMES any such reader.
//   bits 0|1 default ON: measured +3.7% P100 decode on the published U16 config (docs/LEVERS.md
//   §2), bit-exact vs the eager kernels; pattern mismatch still falls through to eager per-node.
//
// Correctness notes:
//  - Pure eval-time pattern fusion: the ggml graph is UNCHANGED; only the launched kernel
//    sequence differs. Any structural mismatch falls through to the eager per-node path.
//  - Bit-exact math vs the eager kernels: silu x/(1+e^-x); l2_norm rsqrtf(fmaxf(sumsq,eps^2));
//    fused_rms_norm rsqrtf(sumsq/ncols+eps) * w; fused_mul_unary(SILU, no-limit) silu(z)*y.
//  - The state-row conv-tail write (kernel B) is ordered AFTER the SSM_CONV read of the same
//    region by stream order; the delta-net in-place ssm-state update is race-free because every
//    state element is read-at-start/written-at-end by the SAME thread (block-disjoint ownership).
//  - CUDA-graph capture safe: no host syncs; all pointers are graph-node pointers covered by
//    ggml_graph_node_has_matching_properties (slot/pointer changes force re-capture).
//  - Gated to n_tokens==1 && n_seqs==1 && no per-step ckpt (src6==NULL): plain decode only.
//    Prefill/MTP-verify/mixed-seq batches keep the eager path (their node order differs anyway).

#pragma once

#include "pxa-enhance.cuh"   // level default: REFERENCE -> 0 (eager path); env always wins
#include <vector>
#include <atomic>

static inline int pxa_fuse_deltanet_mask() {
    static const int mask = getenv("PXA_FUSE_DELTANET") ? atoi(getenv("PXA_FUSE_DELTANET")) : pxa_fuse_deltanet_default();
    return mask;
}

// PXA_FUSE_DELTANET_WAR (default ON, =0 restores the unguarded fusion for A/B only).
//
// Kernel B and the delta-net state redirect below write through CC->data, and the redirect
// hands the delta-net core a state DESTINATION that is not the tensor the graph allocated for
// it. That is only safe when CC is the in-place concat into the recurrent cache row, which is
// the shape this cluster was written against (PXA_DN_NP1_FASTPATH: ggml_concat_inplace into
// state_dst). On the SHIPPED safe carry the same 23-node window still matches op-for-op, but
// CC is a plain ggml_concat into a compute-buffer scratch whose overlap with the delta-net's
// own state INPUT is whatever ggml-alloc happened to choose. An exact alias is index-for-index
// safe - every state element is read-at-start and written-at-end by the same thread - but a
// SHIFTED overlap is a read/write race inside one kernel, and which one you get depends on the
// compute-buffer layout.
//
// Measured 2026-09-03, 2x V100 -sm layer, Qwable-27B-PXQ4core, 6 identical needle20801
// requests per arm, PXA_DN_SHARE_INPUTS=1 (which shifts the layout):
//   -cuda fusion=0        6/6 identical
//   PXA_FUSE_DELTANET=0   6/6 identical, same sha as fusion=0
//   default (unguarded)   3 shas incl. the all-"!" 539b277a recurrent-corruption signature
//   every PXA_G2_* fusion gate off, one at a time: still 3 shas -> none of those is the cause
static inline bool pxa_fuse_deltanet_war() {
    static const bool on = [](){
        const char * e = getenv("PXA_FUSE_DELTANET_WAR");
        const bool v = e ? atoi(e) != 0 : true;
        if (!v) fprintf(stderr, "PXA_FUSE_DELTANET_WAR: OFF -- the DeltaNet cluster redirect is "
                        "UNGUARDED (known read/write race; measurement only, never ship)\n");
        return v;
    }();
    return on;
}

// exact alias (index-for-index) or fully disjoint; anything shifted is a race
static inline bool pxa_dn_range_safe(const void * dst, size_t dst_bytes, const ggml_tensor * src) {
    if (!dst || !src || !src->data || dst_bytes == 0) return true;
    const char * a0 = (const char *) dst;
    const char * a1 = a0 + dst_bytes;
    const char * b0 = (const char *) src->data;
    const char * b1 = b0 + ggml_nbytes(src);
    if (a0 == b0 && dst_bytes == ggml_nbytes(src)) return true;   // exact alias
    return a1 <= b0 || b1 <= a0;                                  // disjoint
}

// PXA_FUSE_DELTANET bit3 (PXA_DN_SCATTER_FUSE) destination guard.
//
// bit3 points kernel B and the delta-net new-state write straight at a row of the recurrent
// state buffer, so it writes through a pointer the graph's own dataflow does not know about.
// That is the SAME act the PXA_FUSE_DELTANET_WAR guard exists to forbid for bit0, and it is
// held to the same predicate: the destination must BE the delta-net's state tensor,
// index-for-index, not merely a disjoint buffer that happens to be the right size.
//
// The catch is that bit3's destination ROW is resolved on the DEVICE from SET_ROWS' index
// tensor, so the host can only prove the predicate when the index cannot select anything but
// row 0 -- i.e. the destination storage view is a single row starting at SR->data. Reading the
// index host-side would need a sync and would break graph replay, so anything else is
// UNPROVABLE, and unprovable is decline: the trailing SET_ROWS then runs eagerly and the
// cluster takes only bit0's own (already exact-alias-guarded) redirect.
//
// With the guard on this is expected to decline on the safe gather/scatter carry -- which is
// the carry bit3 was written for -- and that is the point: the +3.3% it measured there also
// moved the greedy sha with no arithmetic change, i.e. it was reading or writing a byte the
// eager path did not. bit3 stays in the tree, off, as a narrow reproducer for that hunt.
static inline bool pxa_dn_scatter_dst_safe(const ggml_tensor * SR, const ggml_tensor * RS,
                                           int64_t conv_elems, int64_t ssm_elems) {
    if (!pxa_fuse_deltanet_war()) return true;    // unguarded A/B only, exactly as for bit0
    if (!SR || !RS || !SR->src[2]) return false;
    if (ggml_nrows(SR->src[2]) != 1) return false;                    // row index is a device value
    if (ggml_nelements(RS) != ssm_elems) return false;
    return (const void *)((const float *) SR->data + conv_elems) == (const void *) RS->data;
}

// ---------------------------------------------------------------------------------------------
// decode2 lane, 2026-09-03: the reader-set scan behind bit5 (and the PXA_DN_ROWSCAN diagnostic).
//
// bit3's defect was never an arithmetic one: it wrote the packed state row EARLY (at the top of
// the cluster, instead of at the trailing SET_ROWS) and asserted -- in a comment -- that "nobody
// reads state_storage again" inside that window. That assertion was never checked against the
// graph. It is the same species as bit0's aliasing claim and bit1's sole-consumer claim, and it
// is the reason the greedy sha moved with zero arithmetic change.
//
// So check it. One pass over the SPLIT graph records every node whose dst or any src occupies
// storage in the buffer that holds the recurrent state, with its byte range and node index. The
// predicate is then exact: the early write is observable only by a node that runs strictly after
// our first kernel and strictly before the SET_ROWS we are absorbing. A toucher at an index
// BELOW the cluster ran before the write in stream order (it reads the old row in both arms); a
// toucher ABOVE the SET_ROWS index reads the new row in both arms. Only the open window matters.
//
// Splits: ggml_backend_cuda_graph_compute is handed ONE split's graph, so the scan only sees this
// split. That is sufficient, not lucky: the scheduler runs splits in order, and both the early
// write and the SET_ROWS it replaces live in THIS split, so no node of another split can fall
// between them.
struct pxa_dn_touch { const char * lo; const char * hi; int idx; };

struct pxa_dn_touch_cache {
    const ggml_cgraph *   g       = nullptr;
    ggml_tensor **        nodes   = nullptr;
    int                   n_nodes = 0;
    const void *          d0      = nullptr;
    const void *          dl      = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    std::vector<pxa_dn_touch> v;
};

static const std::vector<pxa_dn_touch> & pxa_dn_touches(const ggml_cgraph * cgraph,
                                                        ggml_backend_buffer_t buf) {
    static thread_local pxa_dn_touch_cache C[4];
    static thread_local int next_slot = 0;

    const int n = cgraph->n_nodes;
    const void * d0 = n > 0 ? (const void *) cgraph->nodes[0]->data     : nullptr;
    const void * dl = n > 0 ? (const void *) cgraph->nodes[n-1]->data   : nullptr;

    for (int k = 0; k < 4; ++k) {
        if (C[k].buf == buf && C[k].g == cgraph && C[k].nodes == cgraph->nodes &&
            C[k].n_nodes == n && C[k].d0 == d0 && C[k].dl == dl) {
            return C[k].v;
        }
    }

    pxa_dn_touch_cache & c = C[next_slot];
    next_slot = (next_slot + 1) & 3;
    c.g = cgraph; c.nodes = cgraph->nodes; c.n_nodes = n; c.d0 = d0; c.dl = dl; c.buf = buf;
    c.v.clear();
    for (int j = 0; j < n; ++j) {
        ggml_tensor * nd = cgraph->nodes[j];
        for (int s = -1; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = s < 0 ? nd : nd->src[s];
            if (!t || !t->data || t->buffer != buf) continue;
            const char * lo = (const char *) t->data;
            c.v.push_back({ lo, lo + ggml_nbytes(t), j });
        }
    }
    return c.v;
}

// PXA_DN_ROWSCAN=1: print, once per state buffer region, every node that touches it. This is the
// instrument that NAMES the reader bit3 raced with (or shows there is none).
static void pxa_dn_rowscan_dump(const ggml_cgraph * cgraph, int i, int sr_idx,
                                const ggml_tensor * SR, const char * lo, const char * hi) {
    static const bool on = getenv("PXA_DN_ROWSCAN") != nullptr;
    if (!on) return;
    static thread_local int left = 6;
    if (left-- <= 0) return;
    fprintf(stderr, "PXA_DN_ROWSCAN: cluster@%d set_rows@%d nodes=%d row=[%p,%p) tensor=%s\n",
            i, sr_idx, cgraph->n_nodes, (const void *) lo, (const void *) hi, SR->name);
    const auto & tv = pxa_dn_touches(cgraph, SR->buffer);
    for (const auto & e : tv) {
        if (e.hi <= lo || hi <= e.lo) continue;
        const ggml_tensor * nd = cgraph->nodes[e.idx];
        const char * where = e.idx <  i      ? "before"
                           : e.idx == sr_idx ? "SET_ROWS(absorbed)"
                           : e.idx >  sr_idx ? "after"
                                             : "*** INSIDE WINDOW ***";
        fprintf(stderr, "   node[%4d] %-22s op=%-16s %s\n",
                e.idx, nd->name, ggml_op_name(nd->op), where);
    }
    fflush(stderr);
}

// bit5's predicate: no node strictly inside (i, sr_idx) touches the state tensor's storage.
static bool pxa_dn_row_window_clear(const ggml_cgraph * cgraph, int i, int sr_idx,
                                    const ggml_tensor * SR) {
    if (!SR->buffer || !SR->data) return false;
    const char * lo = (const char *) SR->data;
    const char * hi = lo + ggml_nbytes(SR);
    pxa_dn_rowscan_dump(cgraph, i, sr_idx, SR, lo, hi);
    const auto & tv = pxa_dn_touches(cgraph, SR->buffer);
    for (const auto & e : tv) {
        if (e.hi <= lo || hi <= e.lo) continue;   // disjoint from this layer's row storage
        if (e.idx <  i)      continue;            // ran before the early write, in both arms
        if (e.idx == sr_idx) continue;            // the SET_ROWS we are absorbing
        if (e.idx >  sr_idx) continue;            // runs after the eager SET_ROWS too
        return false;                             // a reader of the old row inside the window
    }
    return true;
}

// bit4's predicate: CC is compute scratch, so nothing is written outside the graph's dataflow --
// but ggml-alloc may have placed CC's storage over a tensor that is still live for the kernels we
// keep. Require CC's whole range to be disjoint from every tensor those kernels read or write.
// PXA_FUSE_DELTANET_LOG=1: a fire/decline census per fusion site, dumped every 20000 decisions.
// A lever that silently declines is the difference between a measurement and a wasted GPU hour,
// so make the fact visible instead of inferring it from an op profile.
static void pxa_dn_fire_log(const char * site, bool fired) {
    static const bool on = getenv("PXA_FUSE_DELTANET_LOG") != nullptr;
    if (!on) return;
    struct row { const char * name; long fire; long decline; };
    static thread_local row rows[8] = {};
    static thread_local long total = 0;
    row * r = nullptr;
    for (auto & x : rows) {
        if (x.name == site) { r = &x; break; }
        if (!x.name) { x.name = site; r = &x; break; }
    }
    if (!r) return;
    (fired ? r->fire : r->decline)++;
    if ((++total % 20000) != 0) return;
    fprintf(stderr, "==== PXA_FUSE_DELTANET census (mask=%d) ====\n", pxa_fuse_deltanet_mask());
    for (auto & x : rows) {
        if (!x.name) break;
        fprintf(stderr, "  %-18s fired=%-10ld declined=%-10ld\n", x.name, x.fire, x.decline);
    }
    fflush(stderr);
}

// PXA_DN_OUTGATE_DBG=1: print, for the first 40 out-gate sites, where M sits relative to the
// two inputs the fused kernel re-reads. d(x) == 0 is the harmless exact alias; a small non-zero
// d(x) is the shifted overlap the guard above declines.
static inline void pxa_dn_outgate_dbg(const ggml_tensor * m, const ggml_tensor * x,
                                      const ggml_tensor * z, bool fired) {
    static const bool on = getenv("PXA_DN_OUTGATE_DBG") != nullptr;
    if (!on) return;
    static std::atomic<int> left{40};
    if (left.fetch_sub(1) <= 0) return;
    auto delta = [](const ggml_tensor * d, const ggml_tensor * s) -> long long {
        if (!d || !s || d->buffer != s->buffer || !d->data || !s->data) return (long long)1e18;
        return (long long)((const char *)d->data - (const char *)s->data);
    };
    fprintf(stderr, "PXA_DN_OUTGATE_DBG: %-22s %s  m=%p x=%p z=%p  d(x)=%lld d(z)=%lld "
            "nb_m=%zu nb_x=%zu nb_z=%zu\n", m->name, fired ? "FUSE " : "DECLINE",
            m->data, x ? x->data : nullptr, z ? z->data : nullptr, delta(m, x), delta(m, z),
            ggml_nbytes(m), x ? ggml_nbytes(x) : 0, z ? ggml_nbytes(z) : 0);
}

static inline bool pxa_dn_disjoint(const void * a, size_t an, const ggml_tensor * t) {
    if (!a || an == 0 || !t || !t->data) return true;
    const char * a0 = (const char *) a;
    const char * a1 = a0 + an;
    const char * b0 = (const char *) t->data;
    const char * b1 = b0 + ggml_nbytes(t);
    return a1 <= b0 || b1 <= a0;
}

// ---------------------------------------------------------------------------------------------
// Kernel A: silu over the full conv output + per-head l2-norm of the q/k regions.
//   raw      [total]          pre-activation conv output (token columns of conv_output_raw)
//   silu_out [total]          silu(raw) — the full tensor is written (v region consumed by delta-net)
//   qn/kn    [hd*nh] each     l2-normalized silu'd q/k head blocks (the L2_NORM node dsts)
// Blocks [0, 2*nh): one per q/k head (block reduction). Blocks >= 2*nh: elementwise tail.
static __global__ void pxa_dn_silu_qknorm_f32(
        const float * __restrict__ raw, float * __restrict__ silu_out,
        float * __restrict__ qn, float * __restrict__ kn,
        const int hd, const int nh, const int total, const float eps) {
    const int nqk = 2*nh;
    if ((int)blockIdx.x < nqk) {
        const int base = blockIdx.x * hd;
        const int tid  = threadIdx.x;
        float vals[2]; // hd <= 2*blockDim.x (hd in {64,128}, blockDim.x = 128)
        int nv = 0;
        float sumsq = 0.0f;
        for (int c = tid; c < hd; c += blockDim.x) {
            const float x  = raw[base + c];
            const float sv = x / (1.0f + expf(-x));
            silu_out[base + c] = sv;
            vals[nv++] = sv;
            sumsq += sv*sv;
        }
        sumsq = warp_reduce_sum(sumsq);
        __shared__ float smem[8];
        if (blockDim.x > WARP_SIZE) {
            const int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
            if (lid == 0) smem[wid] = sumsq;
            __syncthreads();
            sumsq = tid < blockDim.x/WARP_SIZE ? smem[tid] : 0.0f;
            sumsq = warp_reduce_sum(sumsq);
            if (tid == 0) smem[0] = sumsq;
            __syncthreads();
            sumsq = smem[0];
        }
        const float scale = rsqrtf(fmaxf(sumsq, eps*eps));
        float * dstp = (int)blockIdx.x < nh ? qn + blockIdx.x*hd : kn + (blockIdx.x - nh)*hd;
        nv = 0;
        for (int c = tid; c < hd; c += blockDim.x) {
            dstp[c] = vals[nv++] * scale;
        }
    } else {
        const int idx = nqk*hd + (blockIdx.x - nqk)*blockDim.x + threadIdx.x;
        if (idx < total) {
            const float x = raw[idx];
            silu_out[idx] = x / (1.0f + expf(-x));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B: gather the strided conv-tail view [dc, cd] straight into the state cache row,
// replacing CONT + the conv half of the CONCAT. dst layout = the CONT layout [t + c*dc].
static __global__ void pxa_dn_conv_tail_f32(
        const float * __restrict__ src, float * __restrict__ dst,
        const int dc, const int64_t cd, const int64_t se0, const int64_t se1,
        const int32_t * __restrict__ row_idx = nullptr, const int64_t row_stride = 0) {
    const int64_t idx = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= dc*cd) return;
    const int64_t t = idx % dc;
    const int64_t c = idx / dc;
    // PXA_DN_SCATTER_FUSE: with row_idx given, dst is the BASE of the recurrent state buffer and
    // the row is picked on the device (n_seqs == 1 at this fusion site, so row_idx[0]).
    float * dp = row_idx ? dst + (int64_t)row_idx[0]*row_stride : dst;
    dp[idx] = src[t*se0 + c*se1];
}

// ---------------------------------------------------------------------------------------------
// Kernel D (PXA_DN_GATHER_FUSE): GET_ROWS(state, idx) immediately followed by MUL(., mask), the
// read half of the SAFE delta-net state path, in one launch.
//   dst[s*ncols + e] = src[idx[s]*src_row_stride + e] * mask[s*mask_row_stride]
// Identical arithmetic to the two eager kernels (a copy and a broadcast multiply), so bit-exact;
// it removes one launch and one full round trip of the ~4 MB state row per delta-net layer per
// token. The mask is the [1, n_seqs] per-sequence state-reset mask (1 = carry, 0 = fresh seq).
static __global__ void pxa_dn_gather_mask_f32(
        const float * __restrict__ src, const int32_t * __restrict__ idx,
        const float * __restrict__ mask, float * __restrict__ gdst, float * __restrict__ dst,
        const int64_t ncols, const int64_t src_row_stride, const int64_t mask_row_stride) {
    const int64_t s = blockIdx.y;
    const float m = mask[s*mask_row_stride];
    const float * sp = src + (int64_t)idx[s]*src_row_stride;
    float * gp = gdst + s*ncols;
    float * dp = dst  + s*ncols;
    for (int64_t e = (int64_t)blockIdx.x*blockDim.x + threadIdx.x; e < ncols;
         e += (int64_t)gridDim.x*blockDim.x) {
        const float v = sp[e];
        gp[e] = v;          // the GET_ROWS dst, so the fused pair leaves the graph in exactly the
        dp[e] = v * m;      // state the two eager kernels would have -- no dead-store assumption
    }
}

// Fires on the GET_ROWS node when the next non-no-op node is the broadcast MUL of it. Returns the
// number of nodes consumed beyond the GET_ROWS (the caller advances by it), or 0 for no match.
//
// Deliberately NOT a dead-store fusion: both destinations are written, so no claim is made about
// who else in the graph reads the GET_ROWS output. What it buys is the launch and the re-READ of
// the ~4 MB state row that the eager MUL would do -- per delta-net layer, per token, 48 layers.
// The read side of the safe gather/scatter state path is the reason PXA_DN_NP1_FASTPATH could be
// turned off (it aliased the live cache row); this keeps that fix and takes back part of its cost.
//
// STATUS 2026-09-03: this pattern DOES NOT CURRENTLY FIRE on the Qwable-27B decode graph, and the
// bit claims no win. Evidence, PXA_PROFILE at bs=1 decode, per token: MUL stays at 64 with the bit
// on and 64 with it off (a fire would drop it by 48), and PXA_FUSE_DELTANET=7 is byte-identical to
// =3 on the greedy sha. The most likely cause is that the scheduler puts a split boundary between
// the GET_ROWS and the MUL -- the mask is a host-buffer INPUT, so the split that consumes it can
// start at the MUL -- in which case nodes[i+1] is simply not the MUL and the match falls through
// to eager. Left armed because the check is one compare per GET_ROWS and it is exact when it does
// match, but do NOT credit it in any table until a profile shows MUL at 16 per token.
static int pxa_try_deltanet_gather_mask(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!(pxa_fuse_deltanet_mask() & 4)) return 0;
    if (i + 1 >= cgraph->n_nodes) return 0;
    // decode2 2026-09-03: the pair is not always ADJACENT in the node list -- the reshape/view
    // nodes the graph builder emits between them are no-ops the executor already skips, but the
    // old i+1 test treated one as a mismatch and the bit never fired. Skip no-ops (and nothing
    // else) when looking for the MUL; the count of consumed nodes is returned so the caller
    // advances past exactly the nodes covered here.
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_is_noop(cgraph->nodes[j])) ++j;
    if (j >= cgraph->n_nodes) return 0;
    const ggml_tensor * G = cgraph->nodes[i];      // GET_ROWS
    const ggml_tensor * M = cgraph->nodes[j];      // MUL by the state mask
    if (G->op != GGML_OP_GET_ROWS || M->op != GGML_OP_MUL) { pxa_dn_fire_log("gather-mask", false); return 0; }
    if (M->src[0] != G || !M->src[1]) return 0;
    const ggml_tensor * SRC = G->src[0];
    const ggml_tensor * IDX = G->src[1];
    const ggml_tensor * MSK = M->src[1];
    if (!SRC || !IDX) return 0;
    if (SRC->type != GGML_TYPE_F32 || G->type != GGML_TYPE_F32 || M->type != GGML_TYPE_F32) return 0;
    if (IDX->type != GGML_TYPE_I32 || MSK->type != GGML_TYPE_F32) return 0;
    // 2-D gather only: [ncols, n_seqs] out of [ncols, n_rows]; contiguous rows both sides.
    if (G->ne[2] != 1 || G->ne[3] != 1 || SRC->ne[2] != 1 || SRC->ne[3] != 1) return 0;
    if (IDX->ne[1] != 1 || IDX->ne[2] != 1 || IDX->ne[3] != 1) return 0;
    if (IDX->ne[0] != G->ne[1]) return 0;
    if (!ggml_is_contiguous(G) || !ggml_is_contiguous(M)) return 0;
    if (!ggml_is_contiguous_rows(SRC)) return 0;
    if (SRC->nb[0] != sizeof(float) || SRC->nb[1] % sizeof(float)) return 0;
    if (IDX->nb[0] != sizeof(int32_t)) return 0;
    // the mask must broadcast over dim 0 and match one value per gathered row
    if (MSK->ne[0] != 1 || MSK->ne[1] != G->ne[1] || MSK->ne[2] != 1 || MSK->ne[3] != 1) return 0;
    if (MSK->nb[1] % sizeof(float)) return 0;
    if (!ggml_are_same_shape(G, M)) return 0;
    // this is the state row, not some small gather: only worth a fused launch when it is big
    if (G->ne[0] < 4096) return 0;
    if (!ops_are_same_device(cgraph, i, j)) return 0;

    const int64_t ncols = G->ne[0];
    const int64_t nseqs = G->ne[1];
    const int block = 256;
    const int64_t want = (ncols + block - 1)/block;
    dim3 grid((unsigned)(want < 65535 ? want : 65535), (unsigned)nseqs, 1);
    pxa_dn_gather_mask_f32<<<grid, block, 0, ctx.stream()>>>(
            (const float *)SRC->data, (const int32_t *)IDX->data,
            (const float *)MSK->data, (float *)G->data, (float *)M->data,
            ncols, (int64_t)(SRC->nb[1]/sizeof(float)), (int64_t)(MSK->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    pxa_dn_fire_log("gather-mask", true);
    return j - i;
}

// ---------------------------------------------------------------------------------------------
// Kernel C: fused rms-norm(x)*w * silu(z) — the DeltaNet gated-output epilogue.
static __global__ void pxa_dn_rms_silu_gate_f32(
        const float * __restrict__ x, const float * __restrict__ w,
        const float * __restrict__ z, float * __restrict__ dst,
        const int ncols, const float eps, void * __restrict__ vq8 = nullptr) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * xr = x + (int64_t)row*ncols;
    const float * zr = z + (int64_t)row*ncols;
    float       * dr = dst + (int64_t)row*ncols;
    float sumsq = 0.0f;
    for (int c = tid; c < ncols; c += blockDim.x) {
        const float v = xr[c];
        sumsq += v*v;
    }
    sumsq = warp_reduce_sum(sumsq);
    __shared__ float smem[8];
    if (blockDim.x > WARP_SIZE) {
        const int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
        if (lid == 0) smem[wid] = sumsq;
        __syncthreads();
        sumsq = tid < blockDim.x/WARP_SIZE ? smem[tid] : 0.0f;
        sumsq = warp_reduce_sum(sumsq);
        if (tid == 0) smem[0] = sumsq;
        __syncthreads();
        sumsq = smem[0];
    }
    const float scale = rsqrtf(sumsq/ncols + eps);
    for (int c = tid; c < ncols; c += blockDim.x) {
        const float zi = zr[c];
        dr[c] = xr[c]*scale*w[c] * (zi / (1.0f + expf(-zi)));
    }
    // G2-F2 QUANTFOLD epilogue: emit the FLAT q8_1 sidecar of dst (bit-identical to
    // quantize_q8_1 of the flattened output; requires ncols % 32 == 0, driver-guarded).
    // This row owns flat chunks [row*ncols/32, (row+1)*ncols/32), one warp per chunk.
    if (vq8) {
        block_q8_1 * q8 = (block_q8_1 *)vq8;
        const int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
        for (int cb = wid; cb < ncols/WARP_SIZE; cb += blockDim.x/WARP_SIZE) {
            const int c = cb*WARP_SIZE + lid;
            const float zi = zr[c];
            const float xi = xr[c]*scale*w[c] * (zi / (1.0f + expf(-zi)));
            float amax = fabsf(xi);
            float sum  = xi;
            amax = warp_reduce_max(amax);
            sum  = warp_reduce_sum(sum);
            const float d = amax / 127;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            const int64_t ib = (int64_t)row*(ncols/WARP_SIZE) + cb;
            q8[ib].qs[lid] = q;
            if (lid == 0) {
                reinterpret_cast<half&>(q8[ib].ds.x) = d;
                reinterpret_cast<half&>(q8[ib].ds.y) = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Cluster handler. Anchored at the UNARY(SILU) node of a DeltaNet layer at bs=1 decode.
// Matches the fixed 23-node window (verified against the live decode graph, both the
// steady-state and the pos-0 state-reset variants have identical structure from the anchor):
//   i+0  UNARY SILU   conv_output_silu       i+12 ADD    alpha_biased
//   i+1  VIEW         conv tail (of raw)     i+13 UNARY  SOFTPLUS
//   i+2  CONT         new_conv_states_cont   i+14 MUL    g_in
//   i+3  RESHAPE                             i+15 PERMUTE g_fused
//   i+4  VIEW         q_conv                 i+16 PERMUTE beta_fused
//   i+5  L2_NORM                             i+17 RESHAPE state_fused
//   i+6  PERMUTE      q_fused                i+18 DELTA_NET
//   i+7  VIEW         k_conv                 i+19 VIEW    (new ssm state)
//   i+8  L2_NORM                             i+20 RESHAPE
//   i+9  PERMUTE      k_fused                i+21 RESHAPE
//   i+10 VIEW         v_in                   i+22 CONCAT  state_cpy (dst = cache row)
//   i+11 PERMUTE      v_fused
// Returns the number of EXTRA nodes consumed (22) or 0 for no-match (eager fallback).
static int pxa_try_deltanet_cluster(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int pxa_mask = pxa_fuse_deltanet_mask();
    if (!(pxa_mask & (1 | 16))) return 0;
    if (i + 22 >= cgraph->n_nodes) return 0;

    ggml_tensor * S   = cgraph->nodes[i];
    ggml_tensor * T   = cgraph->nodes[i+1];
    ggml_tensor * C   = cgraph->nodes[i+2];
    ggml_tensor * R1  = cgraph->nodes[i+3];
    ggml_tensor * Q   = cgraph->nodes[i+4];
    ggml_tensor * LQ  = cgraph->nodes[i+5];
    ggml_tensor * PQ  = cgraph->nodes[i+6];
    ggml_tensor * K   = cgraph->nodes[i+7];
    ggml_tensor * LK  = cgraph->nodes[i+8];
    ggml_tensor * PK  = cgraph->nodes[i+9];
    ggml_tensor * Vv  = cgraph->nodes[i+10];
    ggml_tensor * PV  = cgraph->nodes[i+11];
    ggml_tensor * AD  = cgraph->nodes[i+12];
    ggml_tensor * SP  = cgraph->nodes[i+13];
    ggml_tensor * MU  = cgraph->nodes[i+14];
    ggml_tensor * PG  = cgraph->nodes[i+15];
    ggml_tensor * PB  = cgraph->nodes[i+16];
    ggml_tensor * RS  = cgraph->nodes[i+17];
    ggml_tensor * D   = cgraph->nodes[i+18];
    ggml_tensor * NV  = cgraph->nodes[i+19];
    ggml_tensor * NR1 = cgraph->nodes[i+20];
    ggml_tensor * NR2 = cgraph->nodes[i+21];
    ggml_tensor * CC  = cgraph->nodes[i+22];

    // --- op skeleton ---
    if (T->op   != GGML_OP_VIEW      || C->op   != GGML_OP_CONT      || R1->op  != GGML_OP_RESHAPE ||
        Q->op   != GGML_OP_VIEW      || LQ->op  != GGML_OP_L2_NORM   || PQ->op  != GGML_OP_PERMUTE ||
        K->op   != GGML_OP_VIEW      || LK->op  != GGML_OP_L2_NORM   || PK->op  != GGML_OP_PERMUTE ||
        Vv->op  != GGML_OP_VIEW      || PV->op  != GGML_OP_PERMUTE   ||
        AD->op  != GGML_OP_ADD       || SP->op  != GGML_OP_UNARY     || MU->op  != GGML_OP_MUL     ||
        PG->op  != GGML_OP_PERMUTE   || PB->op  != GGML_OP_PERMUTE   || RS->op  != GGML_OP_RESHAPE ||
        D->op   != GGML_OP_DELTA_NET || NV->op  != GGML_OP_VIEW      || NR1->op != GGML_OP_RESHAPE ||
        NR2->op != GGML_OP_RESHAPE   || CC->op  != GGML_OP_CONCAT) {
        return 0;
    }
    if ((ggml_unary_op)SP->op_params[0] != GGML_UNARY_OP_SOFTPLUS) return 0;

    // --- SILU anchor: bs=1 slice of an SSM_CONV output ---
    if (S->type != GGML_TYPE_F32 || S->ne[1] != 1 || S->ne[2] != 1 || S->ne[3] != 1) return 0;
    if (!S->src[0] || S->src[0]->op != GGML_OP_VIEW) return 0;
    ggml_tensor * RAW = S->src[0]->src[0];
    if (!RAW || RAW->op != GGML_OP_SSM_CONV || RAW->type != GGML_TYPE_F32) return 0;
    if (S->src[0]->data != RAW->data) return 0;         // token slice at offset 0
    if (!ggml_is_contiguous(S)) return 0;
    const int64_t cd = S->ne[0];                        // conv_dim

    // --- conv-tail view + CONT + reshape ---
    if (T->src[0] != RAW || T->type != GGML_TYPE_F32) return 0;
    const int64_t dc = T->ne[0];                        // d_conv - 1
    if (dc < 1 || dc > 8 || T->ne[1] != cd || T->ne[2] != 1 || T->ne[3] != 1) return 0;
    if (C->src[0] != T || !ggml_is_contiguous(C) || C->ne[0] != dc || C->ne[1] != cd) return 0;
    if (R1->src[0] != C) return 0;
    if (T->nb[0] % sizeof(float) || T->nb[1] % sizeof(float)) return 0;
    const int64_t tail_off = ((const char *)T->data - (const char *)RAW->data);
    if (tail_off < 0 || tail_off % sizeof(float)) return 0;

    // --- q/k views + l2 norms + permutes ---
    if (Q->src[0] != S || K->src[0] != S || Vv->src[0] != S) return 0;
    const int64_t hd = Q->ne[0], nh = Q->ne[1];
    if ((hd != 64 && hd != 128) || nh < 1 || nh > 256) return 0;
    if (Q->ne[2] != 1 || Q->ne[3] != 1) return 0;
    if (K->ne[0] != hd || K->ne[1] != nh || K->ne[2] != 1 || K->ne[3] != 1) return 0;
    if (Q->nb[1] != hd*sizeof(float) || K->nb[1] != hd*sizeof(float)) return 0;
    if (Q->data != S->data) return 0;                                   // q at offset 0
    if ((const char *)K->data - (const char *)S->data != (int64_t)(nh*hd*sizeof(float))) return 0;
    if ((const char *)Vv->data - (const char *)S->data != (int64_t)(2*nh*hd*sizeof(float))) return 0;
    if (2*nh*hd + Vv->ne[0]*Vv->ne[1] != cd) return 0;                  // q+k+v tile the conv row
    if (LQ->src[0] != Q || LK->src[0] != K) return 0;
    if (LQ->type != GGML_TYPE_F32 || LK->type != GGML_TYPE_F32) return 0;
    if (!ggml_is_contiguous(LQ) || !ggml_is_contiguous(LK)) return 0;
    float epsq, epsk;
    memcpy(&epsq, LQ->op_params, sizeof(float));
    memcpy(&epsk, LK->op_params, sizeof(float));
    if (epsq != epsk) return 0;
    if (PQ->src[0] != LQ || PK->src[0] != LK || PV->src[0] != Vv) return 0;

    // --- beta-gate chain (delegated to the existing fused kernel) ---
    if (SP->src[0] != AD || MU->src[0] != SP) return 0;
    if (!AD->src[1] || !MU->src[1]) return 0;
    if (ggml_nrows(AD->src[1]) != 1 || ggml_nrows(MU->src[1]) != 1) return 0;
    if (AD->src[1]->ne[0] != AD->src[0]->ne[0] || MU->src[1]->ne[0] != MU->src[0]->ne[0]) return 0;
    if (AD->type != GGML_TYPE_F32 || MU->type != GGML_TYPE_F32 ||
        AD->src[0]->type != GGML_TYPE_F32 || AD->src[1]->type != GGML_TYPE_F32) return 0;
    if (!ggml_is_contiguous(AD->src[0])) return 0;
    if (PG->src[0] != MU) return 0;

    // --- delta-net core: exactly the tensors produced above, plain decode only ---
    if (D->src[0] != PQ || D->src[1] != PK || D->src[2] != PV ||
        D->src[3] != PG || D->src[4] != PB || D->src[5] != RS || D->src[6] != nullptr) {
        return 0;
    }
    if (D->src[0]->ne[1] != 1 || D->src[0]->ne[3] != 1) return 0;       // n_tokens==1, n_seqs==1

    // --- state writeback: CONCAT(conv_tail_cont, new_ssm_state) into the cache row ---
    if (NV->src[0] != D || NR1->src[0] != NV || NR2->src[0] != NR1) return 0;
    if (CC->op_params[0] != 0) return 0;                                // concat along dim 0
    if (CC->src[0] != R1 || CC->src[1] != NR2) return 0;
    if (CC->type != GGML_TYPE_F32 || !ggml_is_contiguous(CC)) return 0;
    const int64_t conv_elems = ggml_nelements(R1);
    const int64_t ssm_elems  = ggml_nelements(NR2);
    if (conv_elems != dc*cd) return 0;
    if (ggml_nelements(CC) != conv_elems + ssm_elems) return 0;

    if (!ops_are_same_device(cgraph, i, i+22)) return 0;

    // --- shape guard (see PXA_FUSE_DELTANET_WAR above) ------------------------------------
    //
    // The cluster writes the conv tail through CC->data and hands the delta-net core a state
    // DESTINATION of CC->data + conv_elems, bypassing the graph's own dataflow for both. That
    // is only defensible on the ONE shape it was written for: CC is ggml_concat_inplace into
    // the recurrent cache row, so the ssm half of CC IS the delta-net's state tensor and every
    // element is read-at-start / written-at-end by the same thread.
    //
    // kernel A writes silu_out / qn / kn while reading RAW: same-index or disjoint only.
    // Common to both shapes below; unconditional, because it costs five compares.
    if (pxa_fuse_deltanet_war()) {
        if (!pxa_dn_range_safe(S->data,  ggml_nbytes(S),  RAW)) return 0;
        if (!pxa_dn_range_safe(LQ->data, ggml_nbytes(LQ), RAW)) return 0;
        if (!pxa_dn_range_safe(LK->data, ggml_nbytes(LK), RAW)) return 0;
        if (!pxa_dn_range_safe(LQ->data, ggml_nbytes(LQ), S))   return 0;
        if (!pxa_dn_range_safe(LK->data, ggml_nbytes(LK), S))   return 0;
    }

    // Both shapes leave the CONT's own dst unwritten (kernel B writes the conv tail straight into
    // CC). bit0 asserted that; bit1's defect was exactly an unverified claim of this form, so
    // verify it: C's only consumer must be R1, and R1's only consumer must be CC.
    if (pxa_fuse_deltanet_war()) {
        if (!pxa_g2_sole_consumer(cgraph, i+2, C,   R1)) return 0;
        if (!pxa_g2_sole_consumer(cgraph, i+3, R1,  CC)) return 0;
        // likewise the delta-net's state half of D's dst is left unwritten; its view chain
        // NV -> NR1 -> NR2 must end at CC and have no other reader.
        if (!pxa_g2_sole_consumer(cgraph, i+19, NV,  NR1)) return 0;
        if (!pxa_g2_sole_consumer(cgraph, i+20, NR1, NR2)) return 0;
        if (!pxa_g2_sole_consumer(cgraph, i+21, NR2, CC))  return 0;
    }

    const bool pxa_inplace_shape =
        (const void *)((const float *) CC->data + conv_elems) == RS->data &&
        ggml_nelements(RS) == ssm_elems &&
        (const void *) CC->data == (const void *) ((const float *) RS->data - conv_elems);


    // --- PXA_DN_SCATTER_FUSE: absorb the SET_ROWS that scatters the packed row back -------------
    // On the SAFE state path (PXA_DN_NP1_FASTPATH=0, the default) the layer ends
    //     CONCAT(conv_tail, new_ssm) -> pxa_new_row ; SET_ROWS(state_storage, pxa_new_row, idx)
    // i.e. it builds the whole ~4 MB row in a scratch buffer and then copies it into the cache.
    // Both producers of that row are already OURS (kernel B and the delta-net state write), so
    // point them at the cache row itself and the SET_ROWS copy disappears: one launch and a full
    // ~4 MB read + ~4 MB write per delta-net layer per token.
    //
    // The claim this rests on is that nothing observes the row between the early write and the
    // point the SET_ROWS would have run: the read side is untouched (the layer reads a PRIVATE
    // gathered copy made by the GET_ROWS above), and within the layer nothing reads state_storage
    // again. bit3 ASSERTED that in this comment and never checked it, which is why it moved the
    // greedy sha with no arithmetic change. bit5 CHECKS it -- pxa_dn_row_window_clear() scans this
    // split's node list for any node between the cluster and the SET_ROWS whose dst or srcs land
    // in the state tensor's storage, and declines if there is one. PXA_DN_ROWSCAN=1 prints it.
    // The destination ROW is resolved on the device (see ggml_cuda_op_delta_net_ex2): the index is
    // a graph input, so a host-side read would need a sync and would break graph replay.
    float * pxa_state_base = nullptr;             // base of state_storage, or null to keep CONCAT dst
    const int32_t * pxa_state_idx = nullptr;
    int64_t pxa_state_row_stride = 0;
    int pxa_extra = 0;
    if ((pxa_mask & (8 | 32)) && i + 23 < cgraph->n_nodes) {
        const ggml_tensor * SR = cgraph->nodes[i+23];
        if (SR->op == GGML_OP_SET_ROWS && SR->src[0] == CC && SR->src[1] && SR->src[2] &&
            SR->src[1]->type == GGML_TYPE_I32 &&
            SR->type == GGML_TYPE_F32 && SR->src[2]->type == GGML_TYPE_F32 &&
            SR->data == SR->src[2]->data &&                    // SET_ROWS' dst IS the storage view
            SR->ne[0] == CC->ne[0] && SR->ne[2] == 1 && SR->ne[3] == 1 &&
            SR->src[1]->ne[0] == CC->ne[1] &&                  // one index per packed row
            CC->ne[1] == 1 && CC->ne[2] == 1 && CC->ne[3] == 1 &&   // n_seqs == 1 at this site
            SR->src[1]->nb[0] == sizeof(int32_t) &&
            SR->nb[0] == sizeof(float) && (SR->nb[1] % sizeof(float)) == 0 &&
            ggml_is_contiguous_rows(SR) &&
            ops_are_same_device(cgraph, i, i+23) &&
            // bit3 (legacy): the same exact-alias predicate the WAR guard holds bit0 to, which
            //   on the safe carry is always false, so bit3 can never fire there.
            // bit5 (decode2, 2026-09-03): the honest predicate for the safe carry -- a reader-set
            //   scan of this split graph proving that nothing between here and the SET_ROWS we
            //   are absorbing touches the state tensor's storage, so the early write is
            //   unobservable. Not asserted: scanned. PXA_DN_ROWSCAN=1 prints the reader set.
            (((pxa_mask & 8)  && pxa_dn_scatter_dst_safe(SR, RS, conv_elems, ssm_elems)) ||
             ((pxa_mask & 32) && (!pxa_fuse_deltanet_war() ||
                                  pxa_dn_row_window_clear(cgraph, i, i+23, SR)))) &&
            // with the absorb live NOTHING writes CC, so CC's own dst becomes a dead store:
            // verify (do not assert) that the SET_ROWS is its only consumer.
            (!pxa_fuse_deltanet_war() || pxa_g2_sole_consumer(cgraph, i+22, CC, SR))) {
            pxa_state_base       = (float *)SR->data;
            pxa_state_idx        = (const int32_t *)SR->src[1]->data;
            pxa_state_row_stride = (int64_t)(SR->nb[1]/sizeof(float));
            pxa_extra            = 1;
        }
    }

    // --- which shape are we on, and is every write we make provable? -----------------------
    //
    // ROW ABSORB LIVE (pxa_extra): every byte this cluster writes outside its own kernels' node
    // storage goes into the recurrent cache ROW, and nothing at all is written to CC. The row
    // lives in the KV buffer, a different allocation from the compute buffer that holds RAW, S,
    // LQ, LK, RS and D, so the writes cannot collide with anything the surviving kernels read;
    // and pxa_dn_row_window_clear() above proved the early write is unobservable. That is the
    // whole proof obligation, and it is discharged. (This is also why the absorb is not merely
    // an extra saving on top of bit4: it REMOVES bit4's only hazard, because the compute-buffer
    // block ggml-alloc hands the CONCAT is very often the block the masked state copy just
    // freed -- same 4 MB shape, dead at exactly that point -- and writing it early would clobber
    // the state the delta-net core is still reading.)
    //
    // IN-PLACE, NO ABSORB (bit0): CC IS the recurrent cache row (ggml_concat_inplace, the
    // PXA_DN_NP1_FASTPATH carry). The ssm half of CC is then the delta-net's state tensor
    // index-for-index, so every element is read-at-start / written-at-end by the same thread.
    // Exact alias or decline: an earlier version accepted "exact alias OR disjoint", and disjoint
    // is not enough -- on the safe carry that let the cluster write through a pointer nothing in
    // the graph knows about and reproduced 539b277a at -np 2.
    //
    // SCRATCH, NO ABSORB (bit4): we write only into CC, the destination the graph allocated for
    // the CONCAT -- the same bytes the CONCAT would have written, from the same inputs, just
    // earlier and in two kernels instead of three. Nothing leaves the graph's dataflow, but the
    // write is EARLY, so CC's range must be disjoint from everything the surviving kernels still
    // read or write. Checked exhaustively; expect this to decline whenever the absorb declines.
    // PXA_DN_ROWSCAN: if the absorb declined, say WHY -- shape mismatch vs a reader in the window
    // are very different findings and the log must not blur them.
    if (getenv("PXA_DN_ROWSCAN") && !pxa_extra && i + 23 < cgraph->n_nodes) {
        static thread_local int left_r = 3;
        if (left_r-- > 0) {
            const ggml_tensor * SR = cgraph->nodes[i+23];
            fprintf(stderr, "PXA_DN_ROWSCAN: absorb DECLINED at %d: next op=%s src0==CC=%d "
                    "dst==src2=%d ne0 %ld vs CC %ld ccne1=%ld inplace=%d mask=%d\n",
                    i, ggml_op_name(SR->op), SR->src[0] == CC,
                    SR->src[2] ? (SR->data == SR->src[2]->data) : -1,
                    (long) SR->ne[0], (long) CC->ne[0], (long) CC->ne[1],
                    (int) pxa_inplace_shape, pxa_mask);
            fflush(stderr);
        }
    }

    bool pxa_shape_ok = false;
    if (pxa_extra) {
        pxa_shape_ok = pxa_inplace_shape ? ((pxa_mask & 1) != 0) : ((pxa_mask & 16) != 0);
    } else if ((pxa_mask & 1) && (pxa_inplace_shape || !pxa_fuse_deltanet_war())) {
        pxa_shape_ok = true;
    } else if ((pxa_mask & 16) && !pxa_inplace_shape && pxa_fuse_deltanet_war()) {
        const size_t cc_bytes = ggml_nbytes(CC);
        const ggml_tensor * others[] = {
            RAW, S, S->src[0], LQ, LK, RS, D, MU, AD, AD->src[0], AD->src[1], MU->src[1],
            D->src[0], D->src[1], D->src[2], D->src[3], D->src[4], D->src[5],
            PB->src[0], Vv, T
        };
        pxa_shape_ok = ggml_is_contiguous(CC) && CC->nb[0] == sizeof(float);
        for (const ggml_tensor * o : others) {
            if (!pxa_shape_ok) break;
            if (!pxa_dn_disjoint(CC->data, cc_bytes, o)) pxa_shape_ok = false;
        }
    }
    if (!pxa_shape_ok) {
        pxa_dn_fire_log(pxa_inplace_shape ? "cluster-inplace" : "cluster-scratch", false);
        pxa_dn_fire_log("row-absorb", false);
        return 0;
    }

    // ------------------------------------ execute ------------------------------------
    cudaStream_t stream = ctx.stream();

    { // A: silu + q/k l2-norm (replaces SILU, L2_NORM, L2_NORM)
        const int nqk = 2*(int)nh;
        const int block = 128;
        const int tail_blocks = (int)(((cd - (int64_t)nqk*hd) + block - 1)/block);
        const int nblocks = nqk + tail_blocks;
        pxa_dn_silu_qknorm_f32<<<nblocks, block, 0, stream>>>(
                (const float *)S->src[0]->data, (float *)S->data,
                (float *)LQ->data, (float *)LK->data,
                (int)hd, (int)nh, (int)cd, epsq);
        CUDA_CHECK(cudaGetLastError());
    }

    { // B: conv tail straight into the state cache row (replaces CONT + the conv half of CONCAT)
        const int block = 256;
        const int nblocks = (int)((dc*cd + block - 1)/block);
        pxa_dn_conv_tail_f32<<<nblocks, block, 0, stream>>>(
                (const float *)((const char *)RAW->data + tail_off),
                pxa_state_base ? pxa_state_base : (float *)CC->data,
                (int)dc, cd, (int64_t)(T->nb[0]/sizeof(float)), (int64_t)(T->nb[1]/sizeof(float)),
                pxa_state_idx, pxa_state_row_stride);
        CUDA_CHECK(cudaGetLastError());
    }

    // beta-gate: the fork's existing fused ADD+SOFTPLUS+MUL kernel
    ggml_cuda_fused_softplus(ctx, MU);

    // delta-net core with the new ssm state redirected into the cache row (replaces the
    // ssm half of CONCAT — and its 2x ~4MB of pure copy traffic)
    if (pxa_state_base) {
        // straight into the cache row; the trailing SET_ROWS is then a no-op and is consumed too
        ggml_cuda_op_delta_net_ex2(ctx, D, pxa_state_base + conv_elems,
                                   pxa_state_idx, pxa_state_row_stride);
    } else {
        ggml_cuda_op_delta_net_ex(ctx, D, (float *)CC->data + conv_elems);
    }

    // CONCAT (i+22) is fully covered by B + the redirect; all other nodes in the window are views.
    // With PXA_DN_SCATTER_FUSE live, the SET_ROWS at i+23 is covered as well.
    pxa_dn_fire_log(pxa_inplace_shape ? "cluster-inplace" : "cluster-scratch", true);
    pxa_dn_fire_log("row-absorb", pxa_extra != 0);
    return 22 + pxa_extra;
}

// ---------------------------------------------------------------------------------------------
// Out-gate handler: FUSED_RMS_NORM(delta-net output, ssm_norm) + FUSED_MUL_UNARY(z, ., SILU) -> 1.
// Returns true if fused (caller then skips one node).
static bool pxa_try_deltanet_outgate(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!(pxa_fuse_deltanet_mask() & 2)) return false;
    if (i + 1 >= cgraph->n_nodes) return false;

    ggml_tensor * F = cgraph->nodes[i];
    ggml_tensor * M = cgraph->nodes[i+1];
    if (M->op != GGML_OP_FUSED_MUL_UNARY) return false;
    if ((ggml_unary_op)M->op_params[0] != GGML_UNARY_OP_SILU) return false;
    float limit;
    memcpy(&limit, (const float *)M->op_params + 1, sizeof(float));
    if (limit >= 1e-6f) return false;                    // limited variant not replicated
    if (M->src[1] != F || !M->src[0]) return false;
    // anchor strictly to the DeltaNet out-gate: rms input is a reshape of a view of DELTA_NET
    if (!F->src[0] || F->src[0]->op != GGML_OP_RESHAPE ||
        !F->src[0]->src[0] || F->src[0]->src[0]->op != GGML_OP_VIEW ||
        !F->src[0]->src[0]->src[0] || F->src[0]->src[0]->src[0]->op != GGML_OP_DELTA_NET) {
        return false;
    }
    if (F->type != GGML_TYPE_F32 || M->type != GGML_TYPE_F32) return false;
    if (F->src[0]->type != GGML_TYPE_F32 || M->src[0]->type != GGML_TYPE_F32) return false;
    if (F->ne[2] != 1 || F->ne[3] != 1) return false;
    if (!F->src[1] || ggml_nrows(F->src[1]) != 1 || F->src[1]->ne[0] != F->ne[0]) return false;
    if (F->src[1]->type != GGML_TYPE_F32) return false;
    if (!ggml_are_same_shape(M->src[0], M) || !ggml_are_same_shape(F, M)) return false;
    if (!ggml_is_contiguous(F->src[0]) || !ggml_is_contiguous(M->src[0]) || !ggml_is_contiguous(M)) return false;
    if (!ops_are_same_device(cgraph, i, i+1)) return false;

    // The kernel below writes M and deliberately leaves F's own dst UNWRITTEN, on the claim
    // that F's only consumer is M. That claim was asserted from the anchor pattern, never
    // checked against the graph, and it is not always true: with the bit0 cluster guarded,
    // this is what was left producing 539b277a (1 in 6 needle20801 requests at -np 1, 1 in 12
    // slot-pinned at -np 2). Verify it instead - the same check the MUL_MULTI_ADD epilogue
    // fusion already does before it swallows a node.
    if (pxa_fuse_deltanet_war() && !pxa_g2_sole_consumer(cgraph, i, F, M)) {
        pxa_dn_fire_log("outgate", false);
        return false;
    }

    // ---------------------------------------------------------------------------------------
    // G2-F4a CLASS WAR (nondet lane, 2026-09-03). The SECOND thing this fusion removes, besides
    // F's own store, is the kernel boundary between READING x / w / z and WRITING M. Unfused,
    // the launch boundary is a grid-wide barrier: FUSED_RMS_NORM has read every byte of x before
    // FUSED_MUL_UNARY writes a byte of M. Fused, the only barrier is the per-block
    // __syncthreads() inside the RMS reduction, which orders nothing BETWEEN blocks -- and this
    // kernel is one block per row, each block reading its whole row twice (once for sumsq, once
    // for the store). If M's storage overlaps an input at a SHIFTED base, block r's store to
    // dr[c] is the same memory as block r''s xr[c'] that block r' has not read yet: a
    // write-after-read race whose outcome depends on block scheduling.
    //
    // Exact aliasing is harmless -- every access inside a block is index-for-index -- so only a
    // shifted overlap declines, exactly as for the ADD+FUSED_RMS_NORM fusion whose identical
    // defect pxa_g2_addfuse_no_shifted_overlap() was written for (see its note in ggml-cuda.cu:
    // same symptom, every greedy temp-0 completion differing, first divergence AT the fused node
    // with both inputs bit-identical).
    //
    // MEASURED, 2026-09-03, 2x V100 -sm layer, Qwable-27B-PXQ4core (arch qwen35), needle3121,
    // 10 identical greedy requests per arm, metric = spread of the token-0 top-1 probability:
    //   PXA_FUSE_DELTANET=55 (ship default)   spread 1.26e-02, 9/10 distinct values
    //   PXA_FUSE_DELTANET=53 (this bit off)   spread 0.00e+00, 1/10   <- deterministic
    //   PXA_FUSE_DELTANET=48 (bits 4|5 only)  spread 0.00e+00, 1/10   <- deterministic
    //   PXA_FUSE_DELTANET=0                   spread 0.00e+00, same value as 53 and 48
    //   every other lever off one at a time (PIPELINE_PP, VOLTA_CUBLAS_NE11, PXQ_MMVQ,
    //   DN_SHARE_INPUTS, CUBLAS_WORKSPACE_CONFIG) left the spread at 1.3e-02..4.0e-02.
    // The greedy sha is stable only where the top-2 margin exceeds that wobble; the release
    // gate's 1-in-4 needle20801 flip and 1-in-12 np=2 flip are the tokens where it does not.
    // PXA_FUSE_DELTANET_WAR=0 restores the unguarded fusion for A/B only.
    if (pxa_fuse_deltanet_war() &&
        !(pxa_g2_addfuse_no_shifted_overlap(M, F->src[0]) &&
          pxa_g2_addfuse_no_shifted_overlap(M, M->src[0]) &&
          pxa_g2_addfuse_no_shifted_overlap(M, F->src[1]))) {
        pxa_dn_outgate_dbg(M, F->src[0], M->src[0], false);
        pxa_dn_fire_log("outgate", false);
        return false;
    }
    pxa_dn_outgate_dbg(M, F->src[0], M->src[0], true);

    float eps;
    memcpy(&eps, F->op_params, sizeof(float));

    const int ncols = (int)F->ne[0];
    const int nrows = (int)ggml_nrows(F);
    // G2-F2 QUANTFOLD: if a q8_1-GEMV consumer of M follows, emit the sidecar in the same launch
    void * g2_q8 = nullptr;
    int64_t g2_padded = 0;
    if (pxa_g2_quantfold() && (ncols % WARP_SIZE) == 0 &&
        pxa_g2_normfuse_wanted(ctx, cgraph, i + 1, M, g2_padded)) {
        g2_q8 = pxa_g2_q8_buf(ctx.device, ctx.stream(), (size_t)(g2_padded/QK8_1)*sizeof(block_q8_1));
    }
    pxa_dn_rms_silu_gate_f32<<<nrows, 128, 0, ctx.stream()>>>(
            (const float *)F->src[0]->data, (const float *)F->src[1]->data,
            (const float *)M->src[0]->data, (float *)M->data, ncols, eps, g2_q8);
    CUDA_CHECK(cudaGetLastError());
    if (g2_q8) {
        auto & sc = pxa_g2_q8sc[ctx.device];
        sc.t = M; sc.data = M->data; sc.padded = g2_padded; sc.eval = pxa_g2_eval_serial;
    }
    // F's own dst is deliberately left unwritten: its only consumer is M (verified above against
    // the graph by pxa_g2_sole_consumer, not asserted from the anchor pattern).
    pxa_dn_fire_log("outgate", true);
    return true;
}
