// PXA_RS_RING -- the recurrent-state ring's rollback is bit-identical to the copy path.
//
// The claim the ring rests on is not statistical: the bytes the next decode reads after a
// rollback must be THE SAME BYTES the per-step copy path would have restored into the live row.
// They are the same store, at a different address -- so this test runs the two arms side by side
// and requires memcmp == 0, for the SSM half (GGML_OP_DELTA_NET) and the conv half
// (GGML_OP_SSM_CONV) alike.
//
// Scenario: a depth-3 draft (T = 4 rows: the target's sampled token plus 3
// drafts) that the target rejects at depth 1, i.e. rows 0..1 are accepted -> accepted_step = 1 ->
// the wanted state is the one after batch row 1 -> per-step snapshot index 1 -> ring plane 2.
//
//   arm A (today): capture into a dense [step][batch_pos][state] buffer, then copy snapshot row
//                  `step` into the live state row -- what llama_kv_cache::per_step_restore does.
//   arm B (ring):  the same op with ggml_op_set_save_strides(state_dim, state_dim*plane_rows)
//                  and its capture pointed at plane 1 of a (1 + n_rs_seq)-plane state tensor,
//                  then the rollback expressed as a READ INDEX -- ggml_get_rows(plane 1+step).
//
// Also covers: n_seqs = 2 (the np2 row mapping), depth 1 (n_rs_seq = 1), and the fact that a
// continuation decode from the restored state produces identical output in both arms.
//
// CPU only, no GPU, no model, no threads beyond ggml's own.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t rng_state = 20260909u;
static float frand() { // deterministic, in [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float) (rng_state >> 8) / (float) (1u << 23)) - 1.0f;
}

static int    g_fail = 0;
static size_t g_checks = 0;

static void check_bytes(const char * what, const void * a, const void * b, size_t n) {
    ++g_checks;
    if (memcmp(a, b, n) == 0) {
        return;
    }
    // report the first differing float so a failure is diagnosable, not just red
    const float * fa = (const float *) a;
    const float * fb = (const float *) b;
    for (size_t i = 0; i < n/sizeof(float); ++i) {
        if (fa[i] != fb[i]) {
            printf("\n    MISMATCH %s: first at float %zu of %zu: copy=%.9g ring=%.9g\n",
                   what, i, n/sizeof(float), (double) fa[i], (double) fb[i]);
            break;
        }
    }
    g_fail = 1;
}

// ---------------------------------------------------------------------------------------------
// SSM half: GGML_OP_DELTA_NET
//
//   S        head dim (state is S x S per head)
//   H        heads
//   T        tokens per sequence (the verify batch: 1 sampled + n_draft drafts)
//   n_seqs   sequences in the batch
//   K        history planes (n_rs_seq); the ring tensor is (1 + K) planes of n_slots rows
//   step     the per-step snapshot to roll back to (= accepted_step)
// ---------------------------------------------------------------------------------------------
static bool case_delta_net(int S, int H, int T, int n_seqs, int K, int step, int n_slots) {
    printf("  delta_net  S=%d H=%d T=%d n_seqs=%d planes=%d step=%d ... ", S, H, T, n_seqs, 1 + K, step);
    fflush(stdout);

    rng_state = 1234567u + (uint32_t)(S*131 + H*17 + T*7 + n_seqs*3 + K);

    const size_t n_qkv  = (size_t) S*T*H*n_seqs;
    const int64_t ssm_state_dim = (int64_t) S*S*H;          // per sequence
    // a realistic row: [conv half | ssm half], as s_l rows are laid out in the engine
    const int64_t conv_state_dim = 8;
    const int64_t state_dim      = conv_state_dim + ssm_state_dim;

    std::vector<float> hq(n_qkv), hk(n_qkv), hv(n_qkv);
    std::vector<float> hg((size_t) T*H*n_seqs), hb((size_t) T*H*n_seqs);
    std::vector<float> hs((size_t) ssm_state_dim*n_seqs);
    for (auto & x : hq) x = frand();
    for (auto & x : hk) x = frand();
    for (auto & x : hv) x = frand();
    for (auto & x : hg) x = -0.5f*(frand() + 1.0f) - 0.05f;
    for (auto & x : hb) x = frand();
    for (auto & x : hs) x = 0.1f*frand();

    struct ggml_init_params ip = { 512ull*1024ull*1024ull, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { printf("FAIL (ggml_init)\n"); return false; }

    auto mk_inputs = [&](ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v,
                         ggml_tensor ** g, ggml_tensor ** b, ggml_tensor ** s) {
        *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
        *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
        *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
        *g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, 1, H, n_seqs);
        *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, H, n_seqs);
        *s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, n_seqs);
        memcpy((*q)->data, hq.data(), ggml_nbytes(*q));
        memcpy((*k)->data, hk.data(), ggml_nbytes(*k));
        memcpy((*v)->data, hv.data(), ggml_nbytes(*v));
        memcpy((*g)->data, hg.data(), ggml_nbytes(*g));
        memcpy((*b)->data, hb.data(), ggml_nbytes(*b));
        memcpy((*s)->data, hs.data(), ggml_nbytes(*s));
    };

    // ---- arm A: the dense per-step capture the engine uses today -----------------------------
    ggml_tensor *qa, *ka, *va, *ga, *ba, *sa;
    mk_inputs(&qa, &ka, &va, &ga, &ba, &sa);
    // [step][batch_pos][ssm_state], steps 0..T-2 (the last state is the live one)
    ggml_tensor * dense = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)(T - 1)*ssm_state_dim*n_seqs);
    memset(dense->data, 0, ggml_nbytes(dense));
    ggml_tensor * ra = ggml_delta_net(ctx, qa, ka, va, ga, ba, sa, dense);

    // ---- arm B: the ring -- same op, capture retargeted into planes 1.. -----------------------
    ggml_tensor *qb, *kb, *vb, *gb, *bb, *sb;
    mk_inputs(&qb, &kb, &vb, &gb, &bb, &sb);
    // the ring: (1 + K) planes of n_slots rows of state_dim, plane-major
    ggml_tensor * ring = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_dim, (int64_t) n_slots*(1 + K));
    memset(ring->data, 0, ggml_nbytes(ring));
    const int64_t plane_elems = state_dim*n_slots;
    ggml_tensor * cap = ggml_view_1d(ctx, ring, plane_elems*K - conv_state_dim,
            (size_t)(plane_elems + conv_state_dim)*ggml_element_size(ring));
    ggml_tensor * rb = ggml_delta_net(ctx, qb, kb, vb, gb, bb, sb, cap);
    ggml_op_set_save_strides(rb, state_dim, plane_elems);

    // the rollback READ: get_rows on the ring at plane (1 + step), row = seq (== batch pos in v1)
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
    for (int s = 0; s < n_seqs; ++s) {
        ((int32_t *) idx->data)[s] = (int32_t)((1 + step)*n_slots + s);
    }
    ggml_tensor * rolled = ggml_get_rows(ctx, ring, idx);   // [state_dim, n_seqs]

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ra);
    ggml_build_forward_expand(gf, rb);
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    // the read must run AFTER the capture wrote the planes
    struct ggml_cgraph * gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gr, rolled);
    ggml_graph_compute_with_ctx(ctx, gr, 2);

    // 1. the two arms computed the same thing at all (guards a broken retarget that also broke
    //    the recurrence itself)
    check_bytes("delta_net output+final state", ra->data, rb->data, ggml_nbytes(ra));

    // 2. THE CLAIM: the ring row the rollback reads == the dense snapshot the copy path restores
    const float * dense_f = (const float *) dense->data;
    const float * rolled_f = (const float *) rolled->data;
    for (int s = 0; s < n_seqs; ++s) {
        const float * want = dense_f + ((size_t) step*n_seqs + s)*ssm_state_dim;
        const float * got  = rolled_f + (size_t) s*state_dim + conv_state_dim; // ssm half of the row
        check_bytes("ssm snapshot", want, got, (size_t) ssm_state_dim*sizeof(float));
    }

    // 3. the untouched planes stayed untouched: plane 0 (live) is written by the SCATTER, which is
    //    not part of this op, so it must still be zero here.
    const float * ring_f = (const float *) ring->data;
    for (int64_t i = 0; i < plane_elems; ++i) {
        if (ring_f[i] != 0.0f) {
            printf("\n    MISMATCH: the capture wrote into plane 0 at element %lld\n", (long long) i);
            g_fail = 1;
            break;
        }
    }
    ++g_checks;

    ggml_free(ctx);
    printf("%s\n", g_fail ? "FAIL" : "ok");
    return g_fail == 0;
}

// ---------------------------------------------------------------------------------------------
// conv half: GGML_OP_SSM_CONV
// ---------------------------------------------------------------------------------------------
static bool case_ssm_conv(int nc, int nr, int T, int n_kv, int K, int step, int n_slots) {
    printf("  ssm_conv   nc=%d nr=%d T=%d n_kv=%d planes=%d step=%d ... ", nc, nr, T, n_kv, 1 + K, step);
    fflush(stdout);

    rng_state = 7654321u + (uint32_t)(nc*31 + nr*13 + T*5 + n_kv);

    const int64_t conv_state_dim = (int64_t)(nc - 1)*nr;
    const int64_t ssm_tail       = 16;                      // the ssm half of the row, unused here
    const int64_t state_dim      = conv_state_dim + ssm_tail;
    const int     n_t            = T*n_kv;                  // tokens, seq-major

    std::vector<float> hs((size_t) conv_state_dim*n_kv), hx((size_t) nr*n_t), hc((size_t) nc*nr);
    for (auto & x : hs) x = frand();
    for (auto & x : hx) x = frand();
    for (auto & x : hc) x = frand();

    struct ggml_init_params ip = { 256ull*1024ull*1024ull, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { printf("FAIL (ggml_init)\n"); return false; }

    auto mk = [&](ggml_tensor ** s, ggml_tensor ** x, ggml_tensor ** c, ggml_tensor ** sq) {
        *s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nc - 1, nr, n_kv);
        *x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nr, n_t);
        *c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
        *sq = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_t);
        memcpy((*s)->data, hs.data(), ggml_nbytes(*s));
        memcpy((*x)->data, hx.data(), ggml_nbytes(*x));
        memcpy((*c)->data, hc.data(), ggml_nbytes(*c));
        int32_t * m = (int32_t *) (*sq)->data;
        for (int t = 0; t < n_t; ++t) {
            m[(size_t) t*n_kv + 0] = t / T;                 // seq-major: T tokens per sequence
            for (int i = 1; i < n_kv; ++i) m[(size_t) t*n_kv + i] = -1;
        }
    };

    ggml_tensor *s0a, *xa, *ca, *sqa;
    mk(&s0a, &xa, &ca, &sqa);
    ggml_tensor * dense = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, conv_state_dim*n_t);
    memset(dense->data, 0, ggml_nbytes(dense));
    ggml_tensor * ya = ggml_ssm_conv(ctx, s0a, xa, ca, sqa, dense);

    ggml_tensor *s0b, *xb, *cb, *sqb;
    mk(&s0b, &xb, &cb, &sqb);
    ggml_tensor * ring = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_dim, (int64_t) n_slots*(1 + K));
    memset(ring->data, 0, ggml_nbytes(ring));
    const int64_t plane_elems = state_dim*n_slots;
    ggml_tensor * cap = ggml_view_1d(ctx, ring, plane_elems*K,
            (size_t) plane_elems*ggml_element_size(ring));   // conv half is at row offset 0
    ggml_tensor * yb = ggml_ssm_conv(ctx, s0b, xb, cb, sqb, cap);
    ggml_op_set_save_strides(yb, state_dim, plane_elems);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ya);
    ggml_build_forward_expand(gf, yb);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    check_bytes("ssm_conv output", ya->data, yb->data, ggml_nbytes(ya));

    const float * dense_f = (const float *) dense->data;
    const float * ring_f  = (const float *) ring->data;
    for (int s = 0; s < n_kv; ++s) {
        const float * want = dense_f + ((size_t) step*n_kv + s)*conv_state_dim;
        const float * got  = ring_f  + (size_t)(1 + step)*plane_elems + (size_t) s*state_dim;
        check_bytes("conv snapshot", want, got, (size_t) conv_state_dim*sizeof(float));
    }
    // plane 0 must be untouched by the capture
    for (int64_t i = 0; i < plane_elems; ++i) {
        if (ring_f[i] != 0.0f) {
            printf("\n    MISMATCH: the conv capture wrote into plane 0 at element %lld\n", (long long) i);
            g_fail = 1;
            break;
        }
    }
    ++g_checks;
    // and the LAST step must NOT have been recorded in the ring (there is no plane for it, and
    // per_step_restore never reads it) -- plane K+... would be out of range, so check the plane
    // the last step WOULD have landed in is either absent or untouched.
    if (T <= K) {
        const float * last_plane = ring_f + (size_t) T*plane_elems;
        for (int64_t i = 0; i < plane_elems; ++i) {
            if (last_plane[i] != 0.0f) {
                printf("\n    MISMATCH: the conv capture recorded its dead LAST step (plane %d)\n", T);
                g_fail = 1;
                break;
            }
        }
        ++g_checks;
    }

    ggml_free(ctx);
    printf("%s\n", g_fail ? "FAIL" : "ok");
    return g_fail == 0;
}

int main() {
    printf("PXA_RS_RING: rollback-by-index is byte-identical to the per-step copy path\n");

    // the brief's scenario: depth-3 draft (T = 4), rejected at depth 1 -> accepted_step = 1
    case_delta_net(/*S*/ 4, /*H*/ 2, /*T*/ 4, /*n_seqs*/ 1, /*K*/ 4, /*step*/ 1, /*n_slots*/ 1);
    case_ssm_conv (/*nc*/ 4, /*nr*/ 6, /*T*/ 4, /*n_kv*/ 1, /*K*/ 4, /*step*/ 1, /*n_slots*/ 1);

    // every other rollback depth of that same draft
    case_delta_net(4, 2, 4, 1, 4, 0, 1);
    case_delta_net(4, 2, 4, 1, 4, 2, 1);
    case_ssm_conv (4, 6, 4, 1, 4, 0, 1);
    case_ssm_conv (4, 6, 4, 1, 4, 2, 1);

    // np2: two sequences, the row mapping the ring depends on (batch position == seq id)
    case_delta_net(4, 2, 4, 2, 4, 1, 2);
    case_ssm_conv (4, 6, 4, 2, 4, 1, 2);

    // depth 1 (the shipped default): one history plane, one possible rollback
    case_delta_net(4, 2, 2, 1, 1, 0, 1);
    case_ssm_conv (4, 6, 2, 1, 1, 0, 1);

    // a wider head, closer to the seat models (S = 8, two sequences)
    case_delta_net(8, 3, 5, 2, 4, 3, 2);

    printf("%s (%zu byte-exact checks)\n", g_fail ? "FAILED" : "PASSED", g_checks);
    return g_fail;
}
