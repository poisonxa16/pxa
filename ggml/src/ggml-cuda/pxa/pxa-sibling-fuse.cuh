#pragma once

// PXA_FUSE_SIBLINGS: collapse a run of consecutive same-op, same-shape graph nodes into one
// launch (env-gated, default OFF).
//
// WHY: this is the other half of the launch-count problem PXA_EW_FUSE attacks. PXA_EW_FUSE
// collapses a straight-line producer -> consumer chain: node k+1 reads what node k wrote, so the
// intermediate write can be skipped entirely. That mechanism is blind to the opposite shape --
// runs of INDEPENDENT nodes sitting next to each other in the graph, each doing the same work on
// a different tensor of the same shape. Those cannot be collapsed into fewer arithmetic steps,
// because nothing is redundant; but N launches of a 2-3us kernel still cost N launches, and on a
// 4-card decode step where the host is the wall, only the launch count matters.
//
// WHAT: at eval time, find the longest run nodes[i..i+n-1] such that every node has the same op,
// the same shape, the same types and the same device, and merge it into one kernel whose grid is
// the N single-node grids concatenated along y. NOTHING IS ELIMINATED: every node reads and
// writes exactly what it read and wrote alone, with the same kernel body and the same launch
// geometry per node, so the merged result is bit-identical by construction.
//
// The one thing merging DOES change is order: unfused, node k's write completes before node k+1
// starts; fused, they run concurrently. That is only observable if one node in the run writes
// into another node's source or destination, so the run is admitted only after a runtime
// byte-range check over the actual device pointers proves no such overlap exists. Ranges are
// [data, data + ggml_nbytes), which for a strided view is the first-to-last-byte span -- i.e.
// conservative in the safe direction. A null data pointer or a cross-device pair declines.
//
// Ops covered: CPY (the contiguous same-shape flat copy, cpy.cu ggml_cuda_cpy_n) and L2_NORM
// (norm.cu ggml_cuda_op_l2_norm_multi). Both were picked because they are the ops that actually
// occur in runs on our decode graphs; anything else ends the run.
//
// Env:
//   PXA_FUSE_SIBLINGS=1        enable
//   PXA_FUSE_SIBLINGS_MIN=n    minimum run length to merge (default 2)
//   PXA_FUSE_SIBLINGS_LOG=1    print each distinct (op, len, shape) signature on first fire
//   PXA_FUSE_SIBLINGS_CENSUS=1 detect and tally runs but NEVER merge, and dump the table at exit.
//                              This is the honest way to ask "does this pattern occur at all, and
//                              how often" without changing a single result -- run it on the OFF
//                              arm before trusting any throughput number from the ON arm.

#include <map>
#include <set>
#include <string>

static inline bool pxa_fuse_siblings_enabled() {
    static const int v = [] {
        const char * e = getenv("PXA_FUSE_SIBLINGS");
        const int t = e ? atoi(e) : 0;
        if (t) fprintf(stderr, "PXA_FUSE_SIBLINGS: armed (same-op same-shape run merging, bit-exact; "
                               "an A/B without a RUN line under PXA_FUSE_SIBLINGS_LOG is unverified)\n");
        return t;
    }();
    return v != 0;
}

static inline int pxa_fuse_siblings_min() {
    static const int v = [] {
        const char * e = getenv("PXA_FUSE_SIBLINGS_MIN");
        int t = e ? atoi(e) : 2;
        return t < 2 ? 2 : (t > PXA_SIB_MAX ? PXA_SIB_MAX : t);
    }();
    return v;
}

static inline bool pxa_fuse_siblings_log() {
    static const bool v = getenv("PXA_FUSE_SIBLINGS_LOG") != nullptr;
    return v;
}

// Census mode: find runs, count them, merge nothing. Also tallies the ADD -> FUSED_MUL_UNARY
// adjacency, which is the other "does this pattern occur on our graphs" question the launch-count
// audit needs answered (a bias ADD sitting in front of a fused mul+unary is one k_bin_bcast
// launch per occurrence that a fold would remove).
static inline bool pxa_fuse_siblings_census() {
    static const bool v = getenv("PXA_FUSE_SIBLINGS_CENSUS") != nullptr;
    return v;
}

struct pxa_sib_census {
    std::map<std::string, long long> runs;      // "op xN ne=..." -> times seen
    std::map<std::string, long long> nodes;     // "op"           -> nodes seen
    long long add_before_fmu = 0;               // ADD immediately followed by FUSED_MUL_UNARY
    long long add_feeding_fmu = 0;              // ... and the FMU actually reads the ADD's output
    long long add_total = 0;

    ~pxa_sib_census() {
        if (runs.empty() && nodes.empty() && add_total == 0) return;
        fprintf(stderr, "\n=== PXA_FUSE_SIBLINGS census ===\n");
        for (const auto & kv : nodes) {
            fprintf(stderr, "  nodes  %-16s %lld\n", kv.first.c_str(), kv.second);
        }
        for (const auto & kv : runs) {
            fprintf(stderr, "  run    %-48s %lld\n", kv.first.c_str(), kv.second);
        }
        fprintf(stderr, "  ADD nodes                                        %lld\n", add_total);
        fprintf(stderr, "  ADD immediately followed by FUSED_MUL_UNARY      %lld\n", add_before_fmu);
        fprintf(stderr, "  ... and consumed by it (a foldable bias ADD)     %lld\n", add_feeding_fmu);
        fprintf(stderr, "=== end census ===\n");
    }
};

static pxa_sib_census g_pxa_sib_census;

// Byte range of what a node READS and what it WRITES. For most ops the write target is the node
// itself; CPY is the exception -- the node is a handle, src[1] is the tensor written.
static inline const ggml_tensor * pxa_sib_written(const ggml_tensor * node) {
    return node->op == GGML_OP_CPY ? node->src[1] : node;
}

static inline bool pxa_sib_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) return false;
    if (!a->data || !b->data) return true;   // unknown placement: assume the worst, decline
    const char * a0 = (const char *) a->data;
    const char * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *) b->data;
    const char * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// Can nodes[i..i+n-1] run concurrently and still produce what they produce in order?
// Yes iff no node's written range touches any OTHER node's read or written range. A node
// overlapping its own source is left exactly as it is today (that is the single-node kernel's
// business, and merging does not change it).
static inline bool pxa_sib_independent(const ggml_cgraph * cgraph, int i, int n) {
    for (int a = 0; a < n; ++a) {
        const ggml_tensor * wa = pxa_sib_written(cgraph->nodes[i+a]);
        if (!wa) return false;
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            const ggml_tensor * nb = cgraph->nodes[i+b];
            if (pxa_sib_overlap(wa, pxa_sib_written(nb))) return false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (!nb->src[s]) continue;
                if (nb->op == GGML_OP_CPY && s == 1) continue;   // that is nb's write, counted above
                if (pxa_sib_overlap(wa, nb->src[s])) return false;
            }
        }
    }
    return true;
}

// Is this node the kind of node a run can be built from at all?
static inline bool pxa_sib_eligible(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_CPY: {
            const ggml_tensor * s = node->src[0];
            const ggml_tensor * d = node->src[1];
            if (!s || !d) return false;
            if (!ggml_is_contiguous(s) || !ggml_is_contiguous(d)) return false;
            if (!ggml_are_same_shape(s, d)) return false;
            if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_F16) return false;
            if (d->type != GGML_TYPE_F32 && d->type != GGML_TYPE_F16 && d->type != GGML_TYPE_BF16) return false;
            return true;
        }
        case GGML_OP_L2_NORM: {
            const ggml_tensor * s = node->src[0];
            if (!s) return false;
            if (s->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(s)) return false;
            if (s->ne[0] % WARP_SIZE != 0) return false;
            return true;
        }
        default:
            return false;
    }
}

// Two eligible nodes are siblings if they are the same op on the same shape with the same types
// (and, for L2_NORM, the same epsilon -- a different eps is a different kernel argument).
static inline bool pxa_sib_match(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != b->op) return false;
    if (a->type != b->type) return false;
    if (!ggml_are_same_shape(a, b)) return false;
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * sa = a->src[s];
        const ggml_tensor * sb = b->src[s];
        if ((sa == nullptr) != (sb == nullptr)) return false;
        if (!sa) continue;
        if (sa->type != sb->type) return false;
        if (!ggml_are_same_shape(sa, sb)) return false;
    }
    if (a->op == GGML_OP_L2_NORM) {
        if (memcmp(a->op_params, b->op_params, sizeof(float)) != 0) return false;
    }
    return true;
}

static inline std::string pxa_sib_sig(const ggml_tensor * node, int n) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s x%d ne=[%lld,%lld,%lld,%lld]", ggml_op_name(node->op), n,
             (long long) node->ne[0], (long long) node->ne[1],
             (long long) node->ne[2], (long long) node->ne[3]);
    return std::string(buf);
}

// Length of the sibling run starting at i (1 if there is none), capped at PXA_SIB_MAX and at the
// point where independence stops holding.
static inline int pxa_sib_run_len(const ggml_cgraph * cgraph, int i) {
    if (!pxa_sib_eligible(cgraph->nodes[i])) return 1;

    int n = 1;
    while (n < PXA_SIB_MAX && i + n < cgraph->n_nodes) {
        const ggml_tensor * cand = cgraph->nodes[i+n];
        if (ggml_is_noop((ggml_tensor *) cand)) break;
        if (!pxa_sib_eligible(cand)) break;
        if (!pxa_sib_match(cgraph->nodes[i], cand)) break;
        if (!ops_are_same_device(cgraph, i, i+n)) break;
        if (!pxa_sib_independent(cgraph, i, n+1)) break;
        ++n;
    }
    return n;
}

// Census hook: called for every node whether or not fusion is armed. Counts what is there.
static inline void pxa_sib_census_node(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_ADD) {
        ++g_pxa_sib_census.add_total;
        if (i + 1 < cgraph->n_nodes && cgraph->nodes[i+1]->op == GGML_OP_FUSED_MUL_UNARY) {
            ++g_pxa_sib_census.add_before_fmu;
            const ggml_tensor * fmu = cgraph->nodes[i+1];
            if (fmu->src[0] == node || fmu->src[1] == node) {
                ++g_pxa_sib_census.add_feeding_fmu;
            }
        }
    }

    if (node->op != GGML_OP_CPY && node->op != GGML_OP_L2_NORM) return;
    g_pxa_sib_census.nodes[ggml_op_name(node->op)]++;

    // Only report a run at its head, so a run of 3 counts once, not three times.
    if (i > 0 && pxa_sib_eligible(cgraph->nodes[i-1]) &&
        pxa_sib_match(cgraph->nodes[i-1], node)) {
        return;
    }
    const int n = pxa_sib_run_len(cgraph, i);
    if (n >= 2) {
        g_pxa_sib_census.runs[pxa_sib_sig(node, n)]++;
    }
}

// Try to merge the sibling run starting at nodes[i]. Returns the number of nodes consumed (>=2),
// or 0 (nothing launched, caller falls through to the per-node switch unchanged).
static int pxa_try_sibling_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int n = pxa_sib_run_len(cgraph, i);
    if (n < pxa_fuse_siblings_min()) return 0;

    ggml_tensor * nodes[PXA_SIB_MAX];
    for (int k = 0; k < n; ++k) {
        nodes[k] = cgraph->nodes[i+k];
    }

    bool fired = false;
    switch (nodes[0]->op) {
        case GGML_OP_CPY:     fired = ggml_cuda_cpy_n(ctx, nodes, n, false);   break;
        case GGML_OP_L2_NORM: fired = ggml_cuda_op_l2_norm_multi(ctx, nodes, n); break;
        default:              fired = false; break;
    }
    if (!fired) return 0;

    if (pxa_fuse_siblings_log()) {
        static std::set<std::string> seen;
        const std::string sig = pxa_sib_sig(nodes[0], n);
        if (seen.insert(sig).second) {
            fprintf(stderr, "PXA_FUSE_SIBLINGS dev%d: RUN [%s] (head=%s tail=%s)\n",
                    ctx.device, sig.c_str(), nodes[0]->name, nodes[n-1]->name);
        }
    }
    return n;
}
