// PXA_EW_FUSE: generic straight-line elementwise chain fusion (env-gated, default OFF).
//
// WHY: on the 5-card P100 decode seat the nsys host timeline shows ~1340 kernel launches
// per token costing ~12.4ms of host time in a ~37ms token — and 38% of ALL launches are
// four trivial elementwise kernels (add 12.7%, scale 11.3%, mul 7.3%, sigmoid 7.0%) whose
// GPU time is 1.6-3.2us each, i.e. pure launch overhead. The hybrid side-paths (hc gates,
// PLE gate math, conv-as-shifted-copies) emit long straight runs of these.
//
// WHAT: at eval time, collapse a run of consecutive nodes where each node's input is the
// previous node's output (sole consumer, same shape, f32, contiguous, same device, not a
// view, not an OUTPUT) into ONE interpreter kernel applying the same scalar formulas in
// the same order per element. Elementwise ops carry no cross-element reduction, so the
// fused result is BIT-IDENTICAL to the unfused kernels by construction (each formula
// below mirrors the standalone kernel's expression verbatim; float add/mul commute, so
// accepting the chain value on either binary operand is exact).
//
// Ops covered: ADD/MUL (same-shape, no broadcast), SCALE (a*x+b), SQRT, CLAMP,
// UNARY sigmoid/silu/abs/sgn/neg. Anything else ends the chain.
//
// Env: PXA_EW_FUSE=1 enables; PXA_EW_FUSE_MIN (default 2) = minimum run length;
// PXA_EW_FUSE_LOG=1 prints each distinct chain signature on first fire.

#define PXA_EW_MAX_OPS 8

enum pxa_ew_opcode {
    PXA_EW_ADD = 0, PXA_EW_MUL, PXA_EW_SCALE, PXA_EW_SQRT, PXA_EW_CLAMP,
    PXA_EW_SIGMOID, PXA_EW_SILU, PXA_EW_ABS, PXA_EW_SGN, PXA_EW_NEG,
};

struct pxa_ew_prog {
    int n_ops;
    int op[PXA_EW_MAX_OPS];
    const float * ext[PXA_EW_MAX_OPS];   // ADD/MUL second operand; null otherwise
    float a[PXA_EW_MAX_OPS];             // SCALE scale / CLAMP min
    float b[PXA_EW_MAX_OPS];             // SCALE bias  / CLAMP max
};

static __global__ void k_pxa_ew_chain(const float * __restrict__ x, float * __restrict__ dst,
                                      const int64_t n, const pxa_ew_prog p) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    #pragma unroll
    for (int k = 0; k < PXA_EW_MAX_OPS; ++k) {
        if (k >= p.n_ops) break;
        switch (p.op[k]) {
            case PXA_EW_ADD:     v = v + p.ext[k][i]; break;
            case PXA_EW_MUL:     v = v * p.ext[k][i]; break;
            case PXA_EW_SCALE:   v = p.a[k]*v + p.b[k]; break;                       // scale_f32
            case PXA_EW_SQRT:    v = sqrtf(v); break;                                // sqrt_f32
            case PXA_EW_CLAMP:   v = v < p.a[k] ? p.a[k] : (v > p.b[k] ? p.b[k] : v); break; // clamp_f32
            case PXA_EW_SIGMOID: v = 1.0f / (1.0f + expf(-v)); break;                // sigmoid_f32
            case PXA_EW_SILU:    v = v / (1.0f + expf(-v)); break;                   // silu_f32
            case PXA_EW_ABS:     v = fabsf(v); break;                                // op_abs
            case PXA_EW_SGN:     v = (v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f)); break; // op_sgn
            case PXA_EW_NEG:     v = -v; break;                                      // op_neg
        }
    }
    dst[i] = v;
}

static inline bool pxa_ew_fuse_enabled() {
    static const int v = [] {
        const char * e = getenv("PXA_EW_FUSE");
        const int t = e ? atoi(e) : 0;
        if (t) fprintf(stderr, "PXA_EW_FUSE: armed (elementwise chain fusion, bit-exact; "
                               "an A/B without an in-run CHAIN line under PXA_EW_FUSE_LOG is unverified)\n");
        return t;
    }();
    return v != 0;
}
static inline int pxa_ew_fuse_min() {
    static const int v = [] { const char * e = getenv("PXA_EW_FUSE_MIN");
        int t = e ? atoi(e) : 2; return t < 2 ? 2 : (t > PXA_EW_MAX_OPS ? PXA_EW_MAX_OPS : t); }();
    return v;
}
static inline bool pxa_ew_fuse_log() {
    static const bool v = getenv("PXA_EW_FUSE_LOG") != nullptr;
    return v;
}

// Classify one node as a chain member fed by `prev_out` (null => head: input is src0, and
// for ADD/MUL src1 is the ext operand). Returns false if not fusable in this position.
static bool pxa_ew_classify(const ggml_tensor * node, const ggml_tensor * prev_out,
                            int & code, const ggml_tensor * & ext) {
    ext = nullptr;
    if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) return false;
    if (node->view_src != nullptr) return false;  // inplace/view dst: skipping its write is unsafe

    const ggml_tensor * s0 = node->src[0];
    const ggml_tensor * s1 = node->src[1];

    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            if (!s0 || !s1) return false;
            // strict same-shape (no broadcast), both f32 contiguous
            if (s0->type != GGML_TYPE_F32 || s1->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(s0) || !ggml_is_contiguous(s1)) return false;
            if (!ggml_are_same_shape(s0, node) || !ggml_are_same_shape(s1, node)) return false;
            code = node->op == GGML_OP_ADD ? PXA_EW_ADD : PXA_EW_MUL;
            if (prev_out == nullptr || s0 == prev_out) { ext = s1; return true; }
            if (s1 == prev_out)                        { ext = s0; return true; }  // commutative: exact
            return false;
        }
        case GGML_OP_SCALE:
            if (!s0 || s0->type != GGML_TYPE_F32 || !ggml_is_contiguous(s0)) return false;
            if (prev_out && s0 != prev_out) return false;
            code = PXA_EW_SCALE; return true;
        case GGML_OP_SQRT:
            if (!s0 || s0->type != GGML_TYPE_F32 || !ggml_is_contiguous(s0)) return false;
            if (prev_out && s0 != prev_out) return false;
            code = PXA_EW_SQRT; return true;
        case GGML_OP_CLAMP:
            if (!s0 || s0->type != GGML_TYPE_F32 || !ggml_is_contiguous(s0)) return false;
            if (prev_out && s0 != prev_out) return false;
            code = PXA_EW_CLAMP; return true;
        case GGML_OP_UNARY: {
            if (!s0 || s0->type != GGML_TYPE_F32 || !ggml_is_contiguous(s0)) return false;
            if (prev_out && s0 != prev_out) return false;
            switch (ggml_get_unary_op((ggml_tensor *) node)) {
                case GGML_UNARY_OP_SIGMOID: code = PXA_EW_SIGMOID; return true;
                case GGML_UNARY_OP_SILU:    code = PXA_EW_SILU;    return true;
                case GGML_UNARY_OP_ABS:     code = PXA_EW_ABS;     return true;
                case GGML_UNARY_OP_SGN:     code = PXA_EW_SGN;     return true;
                case GGML_UNARY_OP_NEG:     code = PXA_EW_NEG;     return true;
                default: return false;
            }
        }
        default: return false;
    }
}

// Try to fuse a chain starting at nodes[i]. Returns the number of nodes consumed (>=2), or 0.
static int pxa_try_ew_chain(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * head = cgraph->nodes[i];
    int code; const ggml_tensor * ext;
    if (!pxa_ew_classify(head, nullptr, code, ext)) return 0;

    pxa_ew_prog prog;
    prog.n_ops = 0;
    const int64_t n = ggml_nelements(head);

    auto push = [&](const ggml_tensor * node, int c, const ggml_tensor * e) {
        prog.op[prog.n_ops]  = c;
        prog.ext[prog.n_ops] = e ? (const float *) e->data : nullptr;
        float pa = 0.f, pb = 0.f;
        if (c == PXA_EW_SCALE || c == PXA_EW_CLAMP) {
            memcpy(&pa, (const char *) node->op_params + 0, sizeof(float));
            memcpy(&pb, (const char *) node->op_params + sizeof(float), sizeof(float));
        }
        prog.a[prog.n_ops] = pa;
        prog.b[prog.n_ops] = pb;
        ++prog.n_ops;
    };
    push(head, code, ext);

    int j = i;
    const ggml_tensor * prev = head;
    while (prog.n_ops < PXA_EW_MAX_OPS && j + 1 < cgraph->n_nodes) {
        const ggml_tensor * cand = cgraph->nodes[j+1];
        if (ggml_is_noop((ggml_tensor *) cand)) break;
        if (!pxa_ew_classify(cand, prev, code, ext)) break;
        if (ggml_nelements(cand) != n) break;
        if (!ops_are_same_device(cgraph, j, j+1)) break;
        // prev's write is skipped => prev must feed ONLY cand, must not be a graph output,
        // and nothing may view into it later.
        if (prev->flags & GGML_TENSOR_FLAG_OUTPUT) break;
        if (!pxa_g2_sole_consumer(cgraph, j, prev, cand)) break;
        push(cand, code, ext);
        prev = cand;
        ++j;
    }

    const int len = j - i + 1;
    if (len < pxa_ew_fuse_min()) return 0;

    if (pxa_ew_fuse_log()) {
        static std::set<std::string> seen;
        std::string sig;
        static const char * nm[] = {"add","mul","scale","sqrt","clamp","sigmoid","silu","abs","sgn","neg"};
        for (int k = 0; k < prog.n_ops; ++k) { sig += nm[prog.op[k]]; sig += k+1 < prog.n_ops ? "+" : ""; }
        if (seen.insert(sig).second) {
            fprintf(stderr, "PXA_EW_FUSE dev%d: CHAIN [%s] n=%lld (head=%s tail=%s)\n",
                    ctx.device, sig.c_str(), (long long) n, head->name, cgraph->nodes[j]->name);
        }
    }

    const float * x = (const float *) head->src[0]->data;
    float * out     = (float *) cgraph->nodes[j]->data;
    const int64_t blocks = (n + 255) / 256;
    k_pxa_ew_chain<<<(unsigned) blocks, 256, 0, ctx.stream()>>>(x, out, n, prog);
    CUDA_CHECK(cudaGetLastError());
    return len;
}
