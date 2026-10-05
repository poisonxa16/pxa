// PXA_GLM5NEXT: GGML_OP_KPOOL_SCORE must be a drop-in for the unfused indexer chain.
//
// build_glm5next.cpp scores every k-pool for every token as
//
//   kq    = mul_mat(pooled_k, indexer_q)      [n_pool, n_tokens, n_head]
//   kq    = cont(permute(kq, 2,1,0,3))        [n_head, n_tokens, n_pool]
//   score = relu(kq)
//   score = mul(score, weights)               weights broadcast over ne2
//   score = sum_rows(score)                   [1, n_tokens, n_pool]
//   score = cont(permute(score, 2,1,0,3))     [n_pool, n_tokens, 1]
//   score = add(score, pool_mask)
//
// The cont() at the head of that chain is a second full-size f32 tensor beside the
// mul_mat output, and it is the term that decides how large -ub can be. The fused op
// reads kq in its native layout instead.
//
// This test builds BOTH graphs over the same buffers and requires the outputs to be
// EQUAL BIT FOR BIT -- not close. On CPU that pins the accumulation order to
// ggml_vec_sum_f32's (sequential, in ggml_float); the CUDA kernel carries the matching
// claim against k_sum_rows_f32's warp butterfly and is covered by the same test when it
// is run against a CUDA backend.
//
// CPU only, no GPU, no model: deterministic inputs from a fixed LCG.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t rng_state = 987654321u;
static float frand() { // deterministic, in [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float) (rng_state >> 8) / (float) (1u << 23)) - 1.0f;
}

static int n_fail = 0;

static void run_case(int n_pool, int n_tokens, int n_head, bool with_mask, bool with_negatives) {
    const size_t n_kq   = (size_t) n_pool*n_tokens*n_head;
    const size_t n_w    = (size_t) n_head*n_tokens;
    const size_t n_mask = (size_t) n_pool*n_tokens;

    std::vector<float> kq(n_kq), w(n_w), mask(n_mask);
    for (auto & x : kq) {
        // half the entries must be clipped by the relu, or the test never exercises it
        x = with_negatives ? frand() : fabsf(frand());
    }
    for (auto & x : w)  { x = frand(); }
    for (size_t i = 0; i < n_mask; ++i) {
        // the real pool_mask is 0 for a visible pool and -inf for one this token cannot see
        mask[i] = (i % 7 == 3) ? -INFINITY : 0.0f;
    }

    const size_t mem = 64u*1024*1024 + 8*(n_kq + n_w + n_mask)*sizeof(float);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * t_kq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_pool, n_tokens, n_head);
    ggml_tensor * t_w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_head, n_tokens);
    ggml_tensor * t_m  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_pool, n_tokens);
    memcpy(t_kq->data, kq.data(),   n_kq*sizeof(float));
    memcpy(t_w->data,  w.data(),    n_w*sizeof(float));
    memcpy(t_m->data,  mask.data(), n_mask*sizeof(float));

    // --- the chain the fused op replaces, verbatim from build_glm5next.cpp
    ggml_tensor * ref = ggml_cont(ctx, ggml_permute(ctx, t_kq, 2, 1, 0, 3));
    ref = ggml_relu(ctx, ref);
    ref = ggml_mul(ctx, ref, t_w);
    ref = ggml_sum_rows(ctx, ref);
    ref = ggml_cont(ctx, ggml_permute(ctx, ref, 2, 1, 0, 3));
    if (with_mask) {
        ref = ggml_add(ctx, ref, t_m);
    }

    ggml_tensor * fused = ggml_kpool_score(ctx, t_kq, t_w, with_mask ? t_m : nullptr);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, fused);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // ref is [n_pool, n_tokens, 1]; fused is [n_pool, n_tokens]
    const float * a = (const float *) ref->data;
    const float * b = (const float *) fused->data;

    int bad = 0;
    for (size_t i = 0; i < n_mask; ++i) {
        const uint32_t ua = *(const uint32_t *) &a[i];
        const uint32_t ub = *(const uint32_t *) &b[i];
        // -inf == -inf compares equal bitwise too, so no special case is needed
        if (ua != ub && !(std::isnan(a[i]) && std::isnan(b[i]))) {
            if (bad < 4) {
                fprintf(stderr, "  MISMATCH at %zu: ref %.9g (0x%08x) vs fused %.9g (0x%08x)\n",
                        i, (double) a[i], ua, (double) b[i], ub);
            }
            ++bad;
        }
    }

    printf("  n_pool=%-5d n_tokens=%-4d n_head=%-3d mask=%d neg=%d : %s (%d/%zu differ)\n",
           n_pool, n_tokens, n_head, (int) with_mask, (int) with_negatives,
           bad == 0 ? "BIT-IDENTICAL" : "FAIL", bad, n_mask);
    if (bad) {
        ++n_fail;
    }

    ggml_free(ctx);
}

int main() {
    printf("test-kpool-score: GGML_OP_KPOOL_SCORE vs the unfused indexer chain\n");

    // the shape GLM-5.3-Flash actually builds: 32 indexer heads, n_pool = n_kv/4
    run_case(2048, 1,   32, true,  true);
    run_case(2048, 128, 32, true,  true);
    run_case( 512, 64,  32, true,  true);
    run_case( 512, 64,  32, false, true);
    run_case( 512, 64,  32, true,  false);
    // the other head counts the op is instantiated for
    run_case( 257, 5,    4, true,  true);
    run_case( 257, 5,    8, true,  true);
    run_case( 257, 5,   16, true,  true);

    if (n_fail) {
        printf("test-kpool-score: FAILED (%d case(s))\n", n_fail);
        return 1;
    }
    printf("test-kpool-score: OK\n");
    return 0;
}
