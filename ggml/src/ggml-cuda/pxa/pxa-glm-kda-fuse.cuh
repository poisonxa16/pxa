// PXA_GLM_KDA_FUSE (default OFF; 2026-10-18, todo glm-mc-3 item #2): two glm5next KDA decode glue
// chains as one launch each. Pure eval-time pattern fusion: the ggml graph is unchanged, the matcher
// (pxa-glm-kda-match.h, host-only and CPU-tested on the real tiny-model graph) declines anything off-pattern and the
// per-node path runs as before.
//
//   bit0 GATE     ADD(g1, dt_b) -> MUL(., A) -> SCALE -> SIGMOID -> SCALE -> CONT(permute)    6 launches -> 1 (x 34/token)
//   bit1 OUTGATE  SIGMOID(g2) -> MUL(o, .) -> CONT                                              3 launches -> 1 (x 34/token)
//
// BIT-IDENTICAL BY CONSTRUCTION. Every value the fused kernel produces goes through the per-node kernels' own float
// operations in their order, each rounded once, with nothing contracted across an op boundary:
//   ADD     binbcast op_add  a + b                 -> __fadd_rn (never merged into an FMA)
//   MUL     binbcast op_mul  a * b                 -> __fmul_rn (never merged into an FMA)
//   SCALE   scale_f32        scale * x + bias      -> __fmaf_rn: the per-node kernel's own expression is contracted
//                                                     to one FFMA by nvcc (-fmad=true); scripts/glm-kda-fuse/sass-census.sh
//                                                     checks that FFMA in scale_f32's SASS and the fused kernel's
//   SIGMOID sigmoid_f32      1.0f/(1.0f+expf(-x))  -> the same source expression (same flags, same lowering)
//   CONT    a copy                                 -> the store itself
// The mapping from the CONT's (permuted) element to the chain element is the CONT source view's own strides.
// The WAR guard: the GATE kernel writes a PERMUTED order, so its output may not overlap the chain input at all;
// the OUTGATE kernel is index-for-index, so an exact alias is fine and only a shifted overlap declines.
//
// Env: PXA_GLM_KDA_FUSE=1 (both) | 2 (outgate only) | 3 (both); PXA_GLM_KDA_FUSE_LOG=1 prints the first fire of each
// kind and the first 8 near-miss decline reasons.

#include "pxa-glm-kda-match.h"
#include "pxa-fastdiv.cuh"

static __global__ void k_pxa_glm_kda_gate(const pxa_glm_kda_gate_match m) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= m.n) return;
    const int64_t e = pxa_glm_kda_gate_src_index(m, k);
    float v = __fadd_rn(m.g[e], m.dtb[e % m.n0]);           // ADD   g1 + ssm_dt_b
    v = __fmul_rn(v, m.A[(e / m.hd) % m.nh]);               // MUL   * A (per head)
    v = __fmaf_rn(m.s1_scale, v, m.s1_bias);                // SCALE scale_f32: scale * x + bias
    v = 1.0f / (1.0f + expf(-v));                           // SIGMOID sigmoid_f32
    v = __fmaf_rn(m.s2_scale, v, m.s2_bias);                // SCALE
    m.dst[k] = v;                                           // CONT  (permuted store)
}

// The same kernel with 32-bit multiply-shift index math (pxa-fastdiv.cuh) when every index fits in int32 (always on GLM:
// decode 8192, a 512-token ubatch 4.2 M): the int64 divisions of the generic kernel cost tens of instructions each on
// GP100/GV100, which would eat the launch saving at prefill width. Index math only; the float path is the same text.
struct pxa_glm_kda_gate_fd { uint3 ne0, ne1, ne2, n0, hd, nh; };
static __global__ void k_pxa_glm_kda_gate_i32(const pxa_glm_kda_gate_match m, const pxa_glm_kda_gate_fd fd) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= m.n) return;
    uint2 q = pxa_fast_div_modulo((uint32_t) k, fd.ne0); const uint32_t i0 = q.y;
    q = pxa_fast_div_modulo(q.x, fd.ne1);                 const uint32_t i1 = q.y;
    q = pxa_fast_div_modulo(q.x, fd.ne2);                 const uint32_t i2 = q.y, i3 = q.x;
    const uint32_t e = (uint32_t) (m.offs + (int64_t) i0*m.nb[0] + (int64_t) i1*m.nb[1] + (int64_t) i2*m.nb[2] + (int64_t) i3*m.nb[3]);
    float v = __fadd_rn(m.g[e], m.dtb[pxa_fastmodulo(e, fd.n0)]);       // ADD
    v = __fmul_rn(v, m.A[pxa_fastmodulo(pxa_fastdiv(e, fd.hd), fd.nh)]); // MUL
    v = __fmaf_rn(m.s1_scale, v, m.s1_bias);                             // SCALE
    v = 1.0f / (1.0f + expf(-v));                                        // SIGMOID
    v = __fmaf_rn(m.s2_scale, v, m.s2_bias);                             // SCALE
    m.dst[k] = v;                                                        // CONT
}

static __global__ void k_pxa_glm_kda_outgate(const float * __restrict__ o, const float * __restrict__ g,
                                             float * dst, const int64_t n) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float s = 1.0f / (1.0f + expf(-g[k]));            // SIGMOID sigmoid_f32
    dst[k] = __fmul_rn(o[k], s);                            // MUL op_mul (commutative: exact either way), CONT copy
}

static bool pxa_glm_kda_consumers_idx(const ggml_cgraph * cgraph, int i, const ggml_tensor * t,
                                      const ggml_tensor * const * allowed, int n_allowed) {
    if (pxa_consumer_index_on() && t && cgraph->n_nodes > 64) {
        const pxa_cons_index & x = pxa_cons_get(cgraph);
        uint32_t h = pxa_cons_hash(t) & x.mask;
        for (;;) {
            const pxa_cons_slot & sl = x.tab[h];
            if (!sl.key) return true;
            if (sl.key == t) {
                if (sl.cnt > 3) break;                      // many readers: the exact scan below
                for (int k = 0; k < sl.cnt; ++k) {
                    if (sl.idx[k] <= i) continue;
                    const ggml_tensor * n = cgraph->nodes[sl.idx[k]];
                    bool ok = false;
                    for (int a = 0; a < n_allowed; ++a) ok = ok || allowed[a] == n;
                    if (!ok) return false;
                }
                return true;
            }
            h = (h + 1) & x.mask;
        }
    }
    return pxa_glm_kda_consumers_scan(cgraph, i, t, allowed, n_allowed);
}

// t lives in a plain (not streamed) CUDA buffer of this device
static inline bool pxa_glm_kda_on_dev(const ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    const ggml_tensor * b = t->view_src ? t->view_src : t;
    if (!b->buffer || !ggml_backend_buffer_is_cuda(b->buffer) || pxa_stream_buffer_is(b->buffer)) return false;
    return ((const ggml_backend_cuda_buffer_context *) b->buffer->context)->device == ctx.device;
}

static inline bool pxa_glm_kda_disjoint(const ggml_tensor * d, const ggml_tensor * s) {
    const ggml_tensor * db = d->view_src ? d->view_src : d, * sb = s->view_src ? s->view_src : s;
    if (!d->data || !s->data || db->buffer != sb->buffer) return true;
    const char * d0 = (const char *) d->data, * d1 = d0 + ggml_nbytes(d);
    const char * s0 = (const char *) s->data, * s1 = s0 + ggml_nbytes(s);
    return d1 <= s0 || s1 <= d0;
}

static void pxa_glm_kda_log(const char * kind, bool fired, const char * why, const ggml_tensor * t) {
    if (!pxa_glm_kda_fuse_log_on()) return;
    static int n_fire_gate = 0, n_fire_out = 0, n_miss = 0;
    if (fired) {
        int & c = kind[0] == 'g' ? n_fire_gate : n_fire_out;
        if (c++ == 0) fprintf(stderr, "PXA_GLM_KDA_FUSE: first %s fire at %s (%lld elements)\n", kind, t->name, (long long) ggml_nelements(t));
    } else if (n_miss < 8) {
        ++n_miss;
        fprintf(stderr, "PXA_GLM_KDA_FUSE: %s near miss at %s: %s\n", kind, t->name, why ? why : "?");
    }
}

// Returns the node index to resume after (the CONT), or -1 when nothing fused.
static int pxa_try_glm_kda_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int mask = pxa_glm_kda_fuse_mask_env();
    const ggml_tensor * head = cgraph->nodes[i];
    const char * why = nullptr;
    if ((mask & 1) && head->op == GGML_OP_ADD) {
        pxa_glm_kda_gate_match m;
        if (!pxa_glm_kda_match_gate(cgraph, i, m, pxa_glm_kda_consumers_idx, &why)) {
            // near miss = it got past the MUL test (a [1, n_head] multiplier after a bias ADD)
            if (why && strcmp(why, "head not ADD") && strcmp(why, "no MUL after ADD") && strcmp(why, "add src1 not a bias row") &&
                strcmp(why, "mul src1 not [1, n_head]") && strcmp(why, "add src0 broadcast") && strcmp(why, "add not f32 contiguous")) {
                pxa_glm_kda_log("gate", false, why, head);
            }
            return -1;
        }
        if (!ops_are_same_device(cgraph, i, m.last) || !pxa_glm_kda_on_dev(ctx, m.add->src[0]) || !pxa_glm_kda_on_dev(ctx, m.add->src[1]) ||
            !pxa_glm_kda_on_dev(ctx, m.mul->src[1]) || !pxa_glm_kda_on_dev(ctx, m.cont)) {
            pxa_glm_kda_log("gate", false, "not all on this device", head);
            return -1;
        }
        if (!pxa_glm_kda_disjoint(m.cont, m.add->src[0]) || !pxa_glm_kda_disjoint(m.cont, m.add->src[1]) ||
            !pxa_glm_kda_disjoint(m.cont, m.mul->src[1])) {
            pxa_glm_kda_log("gate", false, "output overlaps an input (permuted store)", head);
            return -1;
        }
        const int64_t blocks = (m.n + 255) / 256;
        if (m.n <= INT32_MAX && ggml_nelements(m.add) <= INT32_MAX) {
            const pxa_glm_kda_gate_fd fd = { pxa_init_fastdiv_values((uint32_t) m.ne[0]), pxa_init_fastdiv_values((uint32_t) m.ne[1]),
                                             pxa_init_fastdiv_values((uint32_t) m.ne[2]), pxa_init_fastdiv_values((uint32_t) m.n0),
                                             pxa_init_fastdiv_values((uint32_t) m.hd),    pxa_init_fastdiv_values((uint32_t) m.nh) };
            k_pxa_glm_kda_gate_i32<<<(unsigned) blocks, 256, 0, ctx.stream()>>>(m, fd);
        } else {
            k_pxa_glm_kda_gate<<<(unsigned) blocks, 256, 0, ctx.stream()>>>(m);
        }
        CUDA_CHECK(cudaGetLastError());
        pxa_glm_kda_log("gate", true, nullptr, m.cont);
        return m.last;
    }
    if ((mask & 2) && head->op == GGML_OP_UNARY && ggml_get_unary_op(head) == GGML_UNARY_OP_SIGMOID) {
        pxa_glm_kda_outgate_match m;
        if (!pxa_glm_kda_match_outgate(cgraph, i, m, pxa_glm_kda_consumers_idx, &why)) {
            if (why && strcmp(why, "no MUL after SIGMOID") && strcmp(why, "head not SIGMOID")) pxa_glm_kda_log("outgate", false, why, head);
            return -1;
        }
        const ggml_tensor * o = m.mul->src[0] == m.sig ? m.mul->src[1] : m.mul->src[0];
        if (!ops_are_same_device(cgraph, i, m.last) || !pxa_glm_kda_on_dev(ctx, o) || !pxa_glm_kda_on_dev(ctx, m.sig->src[0]) ||
            !pxa_glm_kda_on_dev(ctx, m.cont)) {
            pxa_glm_kda_log("outgate", false, "not all on this device", head);
            return -1;
        }
        // index-for-index: an exact alias is safe, a shifted overlap is a write-after-read race
        if (!pxa_g2_addfuse_no_shifted_overlap(m.cont, o) || !pxa_g2_addfuse_no_shifted_overlap(m.cont, m.sig->src[0])) {
            pxa_glm_kda_log("outgate", false, "output overlaps an input at a shift", head);
            return -1;
        }
        const int64_t blocks = (m.n + 255) / 256;
        k_pxa_glm_kda_outgate<<<(unsigned) blocks, 256, 0, ctx.stream()>>>(m.o, m.g, m.dst, m.n);
        CUDA_CHECK(cudaGetLastError());
        pxa_glm_kda_log("outgate", true, nullptr, m.cont);
        return m.last;
    }
    return -1;
}
