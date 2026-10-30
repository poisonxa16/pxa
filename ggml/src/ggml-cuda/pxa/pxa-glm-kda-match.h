#pragma once

// PXA_GLM_KDA_FUSE (default OFF; 2026-10-18, todo glm-mc-3 item #2): the pattern half of two
// glm5next KDA decode glue fusions. Host-only (no CUDA types), so tests/test-glm-kda-fuse-match.cpp runs the very
// matcher the CUDA backend uses over a REAL glm5next graph (the tiny model) on the CPU.
//
//   GATE    (bit0)  ADD(g1, ssm_dt_b) -> MUL(., A) -> SCALE(-1) -> SIGMOID -> SCALE(lower_bound) -> CONT(permute)
//                   src/graphs/build_glm5next.cpp "forget gate", bounded form; 6 launches -> 1
//   OUTGATE (bit1)  SIGMOID(g2) -> MUL(o, .) -> CONT (cont_2d)                                    3 launches -> 1
//
// Only no-op nodes (RESHAPE / VIEW / PERMUTE / TRANSPOSE) may sit between the chain nodes; any other node, any
// extra consumer of a skipped intermediate, an OUTPUT flag, a broadcast other than the ones named here or a
// non-F32 tensor declines the match and the per-node path runs unchanged. The math (and why it is bit-identical
// to the per-node kernels) lives with the kernels in pxa-glm-kda-fuse.cuh.

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __CUDACC__
#define PXA_GLM_KDA_HD __host__ __device__
#else
#define PXA_GLM_KDA_HD
#endif

// PXA_GLM_KDA_FUSE: unset = both fusions (default ON since 10-24); 0 = off (the off switch).
// "1" = both fusions (mask 3); any other value = the bitmask (1 gate, 2 outgate). The parser keeps its old contract
// (nullptr/"0" -> 0); the default lives in pxa_glm_kda_fuse_mask_env.
static inline int pxa_glm_kda_fuse_parse(const char * e) {
    if (!e || !*e) return 0;
    const int v = atoi(e);
    if (v <= 0) return 0;
    return v == 1 ? 3 : (v & 3);
}

static inline int pxa_glm_kda_fuse_mask_env(void) {
    static const int m = [] {
        const char * e = getenv("PXA_GLM_KDA_FUSE");
        const int v = e ? pxa_glm_kda_fuse_parse(e) : 3;
        if (v) {
            fprintf(stderr, "PXA_GLM_KDA_FUSE=%d: glm5next KDA glue fusion armed (%s%s; bit-exact, pattern-matched)\n", v,
                    (v & 1) ? "gate chain" : "", (v & 3) == 3 ? " + out-gate" : ((v & 2) ? "out-gate" : ""));
        }
        return v;
    }();
    return m;
}

static inline bool pxa_glm_kda_fuse_log_on(void) {
    static const bool v = [] { const char * e = getenv("PXA_GLM_KDA_FUSE_LOG"); return e && atoi(e) != 0; }();
    return v;
}

static inline bool pxa_glm_kda_is_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// index of the first non-no-op node after j (or -1)
static inline int pxa_glm_kda_next_real(const ggml_cgraph * g, int j) {
    for (int k = j + 1; k < g->n_nodes; ++k) {
        if (!pxa_glm_kda_is_noop(g->nodes[k])) return k;
    }
    return -1;
}

static inline bool pxa_glm_kda_f32c(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

// every node after i that reads t (src or view_src) is one of allowed[]
typedef bool (*pxa_glm_kda_consumers_fn)(const ggml_cgraph * g, int i, const ggml_tensor * t,
                                         const ggml_tensor * const * allowed, int n_allowed);

static inline bool pxa_glm_kda_consumers_scan(const ggml_cgraph * g, int i, const ggml_tensor * t,
                                              const ggml_tensor * const * allowed, int n_allowed) {
    for (int j = i + 1; j < g->n_nodes; ++j) {
        const ggml_tensor * n = g->nodes[j];
        bool ok = false;
        for (int a = 0; a < n_allowed; ++a) ok = ok || allowed[a] == n;
        if (ok) continue;
        if (n->view_src == t) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == t) return false;
        }
    }
    return true;
}

struct pxa_glm_kda_gate_match {
    int last;                       // node index of the CONT (the caller resumes after it)
    const ggml_tensor * add, * mul, * s1, * sig, * s2, * cont;
    const float * g, * dtb, * A;    // ADD src0, ADD src1 (bias row), MUL src1 (one value per head)
    float * dst;                    // the CONT's storage
    int64_t n0;                     // ADD ne[0] (d_inner): bias period
    int64_t hd, nh;                 // MUL ne[0], ne[1]: A index = (e / hd) % nh
    int64_t ne[4];                  // CONT shape
    int64_t nb[4];                  // the CONT source view's strides, in floats, over the chain's contiguous layout
    int64_t offs;                   // its offset, in floats
    int64_t n;                      // elements
    float s1_scale, s1_bias, s2_scale, s2_bias;
};

struct pxa_glm_kda_outgate_match {
    int last;                       // node index of the CONT
    const ggml_tensor * sig, * mul, * cont;
    const float * o, * g;           // MUL's other operand, SIGMOID's input
    float * dst;
    int64_t n;
};

// The logical chain index the GATE kernel reads for output element k (row-major over ne[0..3]).
PXA_GLM_KDA_HD static inline int64_t pxa_glm_kda_gate_src_index(const pxa_glm_kda_gate_match & m, int64_t k) {
    int64_t r = k;
    const int64_t i0 = r % m.ne[0]; r /= m.ne[0];
    const int64_t i1 = r % m.ne[1]; r /= m.ne[1];
    const int64_t i2 = r % m.ne[2]; r /= m.ne[2];
    const int64_t i3 = r;
    return m.offs + i0*m.nb[0] + i1*m.nb[1] + i2*m.nb[2] + i3*m.nb[3];
}

static inline const char * pxa_glm_kda_set_why(const char ** why, const char * w) {
    if (why) *why = w;
    return w;
}

// GATE match anchored at nodes[i] (an ADD). Returns false (with *why) on any mismatch.
static inline bool pxa_glm_kda_match_gate(const ggml_cgraph * g, int i, pxa_glm_kda_gate_match & m,
                                          pxa_glm_kda_consumers_fn consumers, const char ** why) {
    const ggml_tensor * add = g->nodes[i];
    if (add->op != GGML_OP_ADD) return !pxa_glm_kda_set_why(why, "head not ADD");
    const ggml_tensor * a0 = add->src[0], * a1 = add->src[1];
    if (!pxa_glm_kda_f32c(add) || !pxa_glm_kda_f32c(a0) || !pxa_glm_kda_f32c(a1)) return !pxa_glm_kda_set_why(why, "add not f32 contiguous");
    if (!ggml_are_same_shape(a0, add)) return !pxa_glm_kda_set_why(why, "add src0 broadcast");
    if (a1->ne[0] != add->ne[0] || ggml_nrows(a1) != 1) return !pxa_glm_kda_set_why(why, "add src1 not a bias row");
    // MUL by one value per head
    const int jm = pxa_glm_kda_next_real(g, i);
    if (jm < 0) return !pxa_glm_kda_set_why(why, "end of graph");
    const ggml_tensor * mul = g->nodes[jm];
    if (mul->op != GGML_OP_MUL) return !pxa_glm_kda_set_why(why, "no MUL after ADD");
    const ggml_tensor * m0 = mul->src[0], * m1 = mul->src[1];
    if (!pxa_glm_kda_f32c(mul) || !pxa_glm_kda_f32c(m0) || !pxa_glm_kda_f32c(m1)) return !pxa_glm_kda_set_why(why, "mul not f32 contiguous");
    const bool m0_view = m0 != add;
    if (m0_view && !(m0->view_src == add && m0->view_offs == 0 && m0->op == GGML_OP_RESHAPE))
        return !pxa_glm_kda_set_why(why, "mul src0 not the ADD (or its reshape)");
    if (ggml_nelements(m0) != ggml_nelements(add) || !ggml_are_same_shape(m0, mul)) return !pxa_glm_kda_set_why(why, "mul src0 shape");
    if (m1->ne[0] != 1 || m1->ne[1] != mul->ne[1] || m1->ne[2] != 1 || m1->ne[3] != 1) return !pxa_glm_kda_set_why(why, "mul src1 not [1, n_head]");
    if (mul->ne[0]*mul->ne[1] != add->ne[0]) return !pxa_glm_kda_set_why(why, "head_dim*n_head != d_inner");
    // SCALE -> SIGMOID -> SCALE
    const int j1 = pxa_glm_kda_next_real(g, jm);
    if (j1 < 0 || g->nodes[j1]->op != GGML_OP_SCALE || g->nodes[j1]->src[0] != mul) return !pxa_glm_kda_set_why(why, "no SCALE after MUL");
    const ggml_tensor * s1 = g->nodes[j1];
    const int j2 = pxa_glm_kda_next_real(g, j1);
    if (j2 < 0 || g->nodes[j2]->op != GGML_OP_UNARY || ggml_get_unary_op(g->nodes[j2]) != GGML_UNARY_OP_SIGMOID ||
        g->nodes[j2]->src[0] != s1) return !pxa_glm_kda_set_why(why, "no SIGMOID after SCALE");
    const ggml_tensor * sig = g->nodes[j2];
    const int j3 = pxa_glm_kda_next_real(g, j2);
    if (j3 < 0 || g->nodes[j3]->op != GGML_OP_SCALE || g->nodes[j3]->src[0] != sig) return !pxa_glm_kda_set_why(why, "no SCALE after SIGMOID");
    const ggml_tensor * s2 = g->nodes[j3];
    if (!pxa_glm_kda_f32c(s1) || !pxa_glm_kda_f32c(sig) || !pxa_glm_kda_f32c(s2)) return !pxa_glm_kda_set_why(why, "scale/sigmoid not f32 contiguous");
    if (ggml_nelements(s1) != ggml_nelements(mul) || ggml_nelements(sig) != ggml_nelements(mul) ||
        ggml_nelements(s2) != ggml_nelements(mul)) return !pxa_glm_kda_set_why(why, "scale/sigmoid element count");
    // CONT of a (permuted) view of the last SCALE
    const int jc = pxa_glm_kda_next_real(g, j3);
    if (jc < 0 || g->nodes[jc]->op != GGML_OP_CONT) return !pxa_glm_kda_set_why(why, "no CONT after the gate");
    const ggml_tensor * cont = g->nodes[jc];
    const ggml_tensor * v = cont->src[0];
    if (!pxa_glm_kda_f32c(cont) || !v || v->type != GGML_TYPE_F32) return !pxa_glm_kda_set_why(why, "cont not f32 contiguous");
    if (!(v == s2 || v->view_src == s2)) return !pxa_glm_kda_set_why(why, "cont does not read the gate");
    if (ggml_nelements(cont) != ggml_nelements(s2) || !ggml_are_same_shape(v, cont)) return !pxa_glm_kda_set_why(why, "cont shape");
    const size_t offs_b = v == s2 ? 0 : v->view_offs;
    if (offs_b % sizeof(float)) return !pxa_glm_kda_set_why(why, "view offset not float-aligned");
    for (int d = 0; d < 4; ++d) {
        if (v->nb[d] % sizeof(float)) return !pxa_glm_kda_set_why(why, "view stride not float-aligned");
    }
    // the skipped intermediates: no OUTPUT flag, no reader outside the chain
    const ggml_tensor * skipped[] = { add, m0_view ? m0 : nullptr, mul, s1, sig, s2 };
    for (const ggml_tensor * t : skipped) {
        if (t && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) return !pxa_glm_kda_set_why(why, "intermediate is an OUTPUT");
    }
    {
        // views of s2 between it and the CONT (reshape_4d, permute) are part of the chain
        const ggml_tensor * allowed[16];
        int na = 0;
        if (m0_view) allowed[na++] = m0;
        allowed[na++] = mul; allowed[na++] = s1; allowed[na++] = sig; allowed[na++] = s2; allowed[na++] = cont;
        for (int k = j3 + 1; k < jc && na < 16; ++k) {
            const ggml_tensor * n = g->nodes[k];
            if (n->view_src == s2 && pxa_glm_kda_is_noop(n)) allowed[na++] = n;
        }
        for (const ggml_tensor * t : skipped) {
            if (t && !consumers(g, i, t, allowed, na)) return !pxa_glm_kda_set_why(why, "an intermediate has a reader outside the chain");
        }
        for (int k = j3 + 1; k < jc; ++k) {
            const ggml_tensor * n = g->nodes[k];
            if (n->view_src == s2 && !consumers(g, i, n, allowed, na)) return !pxa_glm_kda_set_why(why, "a view of the gate has a reader outside the chain");
        }
    }
    m.last = jc;
    m.add = add; m.mul = mul; m.s1 = s1; m.sig = sig; m.s2 = s2; m.cont = cont;
    m.g = (const float *) a0->data; m.dtb = (const float *) a1->data; m.A = (const float *) m1->data;
    m.dst = (float *) cont->data;
    m.n0 = add->ne[0]; m.hd = mul->ne[0]; m.nh = mul->ne[1];
    for (int d = 0; d < 4; ++d) { m.ne[d] = cont->ne[d]; m.nb[d] = (int64_t) (v->nb[d] / sizeof(float)); }
    m.offs = (int64_t) (offs_b / sizeof(float));
    m.n = ggml_nelements(cont);
    // the same two floats ggml_cuda_op_scale reads
    memcpy(&m.s1_scale, (const float *) s1->op_params + 0, sizeof(float));
    memcpy(&m.s1_bias,  (const float *) s1->op_params + 1, sizeof(float));
    memcpy(&m.s2_scale, (const float *) s2->op_params + 0, sizeof(float));
    memcpy(&m.s2_bias,  (const float *) s2->op_params + 1, sizeof(float));
    // every source index stays inside the chain's elements
    const int64_t n_chain = ggml_nelements(add);
    int64_t lo = m.offs, hi = m.offs;
    for (int d = 0; d < 4; ++d) {
        const int64_t span = (m.ne[d] - 1)*m.nb[d];
        if (span < 0) lo += span; else hi += span;
    }
    if (lo < 0 || hi >= n_chain) return !pxa_glm_kda_set_why(why, "view reaches outside the chain");
    return true;
}

// OUTGATE match anchored at nodes[i] (a SIGMOID).
static inline bool pxa_glm_kda_match_outgate(const ggml_cgraph * g, int i, pxa_glm_kda_outgate_match & m,
                                             pxa_glm_kda_consumers_fn consumers, const char ** why) {
    const ggml_tensor * sig = g->nodes[i];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) return !pxa_glm_kda_set_why(why, "head not SIGMOID");
    if (!pxa_glm_kda_f32c(sig) || !pxa_glm_kda_f32c(sig->src[0]) || ggml_nelements(sig->src[0]) != ggml_nelements(sig))
        return !pxa_glm_kda_set_why(why, "sigmoid not f32 contiguous");
    const int jm = pxa_glm_kda_next_real(g, i);
    if (jm < 0 || g->nodes[jm]->op != GGML_OP_MUL) return !pxa_glm_kda_set_why(why, "no MUL after SIGMOID");
    const ggml_tensor * mul = g->nodes[jm];
    const ggml_tensor * o = mul->src[1] == sig ? mul->src[0] : (mul->src[0] == sig ? mul->src[1] : nullptr);
    if (!o || o == sig) return !pxa_glm_kda_set_why(why, "mul does not read the sigmoid once");
    if (!pxa_glm_kda_f32c(mul) || !pxa_glm_kda_f32c(o)) return !pxa_glm_kda_set_why(why, "mul not f32 contiguous");
    if (!ggml_are_same_shape(o, mul) || !ggml_are_same_shape(sig, mul)) return !pxa_glm_kda_set_why(why, "mul broadcasts");
    const int jc = pxa_glm_kda_next_real(g, jm);
    if (jc < 0 || g->nodes[jc]->op != GGML_OP_CONT || g->nodes[jc]->src[0] != mul) return !pxa_glm_kda_set_why(why, "no CONT after MUL");
    const ggml_tensor * cont = g->nodes[jc];
    if (!pxa_glm_kda_f32c(cont) || ggml_nelements(cont) != ggml_nelements(mul)) return !pxa_glm_kda_set_why(why, "cont shape");
    if ((sig->flags | mul->flags) & GGML_TENSOR_FLAG_OUTPUT) return !pxa_glm_kda_set_why(why, "intermediate is an OUTPUT");
    const ggml_tensor * a_sig[] = { mul };
    const ggml_tensor * a_mul[] = { cont };
    if (!consumers(g, i, sig, a_sig, 1) || !consumers(g, jm, mul, a_mul, 1))
        return !pxa_glm_kda_set_why(why, "an intermediate has a reader outside the chain");
    m.last = jc;
    m.sig = sig; m.mul = mul; m.cont = cont;
    m.o = (const float *) o->data; m.g = (const float *) sig->src[0]->data;
    m.dst = (float *) cont->data;
    m.n = ggml_nelements(cont);
    return true;
}
