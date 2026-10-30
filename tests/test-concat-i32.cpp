// PXA_GLM_CONCAT_I32_CUDA, CPU-runnable coverage (no GPU needed or used).
//
// 1. The CPU backend's I32 CONCAT (the reference the CUDA arm must match, and the arm that
//    runs with the switch OFF) is bit-exact on adversarial bit patterns -- NaN/Inf/denormal
//    float encodings, INT_MIN, -1 -- across dims 0..3, contiguous and viewed sources.
// 2. The predicate the CUDA backend's supports_op uses (ggml-cuda/pxa/pxa-concat-i32.h) admits
//    exactly the copy-only layouts: refused with the switch off, the glm5next indexer tail shape
//    admitted with it on, and the F32-kernel-only layouts refused either way.
//
//   test-concat-i32            (exit 0 = pass; reads PXA_GLM_CONCAT_I32_CUDA only to report it)
#include "ggml.h"
#include "ggml-cuda/pxa/pxa-concat-i32.h"

#include <array>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

static const uint32_t k_patterns[] = {
    0x7f800001u, 0x7fc00000u, 0xffc00000u, 0x7fbfffffu, 0xff800001u, // NaN encodings incl. signalling
    0x7f800000u, 0xff800000u,                                         // +-Inf
    0x00000001u, 0x807fffffu,                                         // denormals
    0x80000000u, 0x7fffffffu, 0xffffffffu, 0x00000000u,               // INT_MIN, INT_MAX, -1, 0
};

static int32_t pat(uint32_t seed) {
    const uint32_t n = sizeof(k_patterns)/sizeof(k_patterns[0]);
    uint32_t x = seed*2654435761u + 12345u;
    uint32_t v = (x >> 7) % 3 == 0 ? k_patterns[x % n] : x ^ (x >> 13);
    int32_t r; memcpy(&r, &v, 4); return r;
}

// element (i0..i3) of a (possibly viewed) tensor
static int32_t at(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    int32_t v;
    memcpy(&v, (const char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], 4);
    return v;
}

static ggml_tensor * make_src(ggml_context * ctx, std::array<int64_t, 4> ne, bool view, uint32_t salt) {
    ggml_tensor * t;
    if (view) {
        auto big = ne; big[0] *= 2; big[1] *= 3; big[2] *= 2;
        t = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, big.data());
        int32_t * d = (int32_t *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = pat((uint32_t) i + salt);
        t = ggml_view_4d(ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
    } else {
        t = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, ne.data());
        int32_t * d = (int32_t *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = pat((uint32_t) i + salt);
    }
    return t;
}

// compute one I32 concat on the CPU and check it bit for bit against a scalar reference
static void check_cpu(std::array<int64_t, 4> ne_a, int64_t ne_b_d, int dim, int v, int n_threads) {
    ggml_init_params ip = { 64u*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    auto ne_b = ne_a; ne_b[dim] = ne_b_d;
    ggml_tensor * a = make_src(ctx, ne_a, v & 1, 17u);
    ggml_tensor * b = make_src(ctx, ne_b, v & 2, 91u);
    ggml_tensor * c = ggml_concat(ctx, a, b, dim);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    int64_t bad = 0;
    for (int64_t i3 = 0; i3 < c->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < c->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < c->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < c->ne[0]; ++i0) {
        int64_t j[4] = { i0, i1, i2, i3 };
        int32_t ref;
        if (j[dim] < a->ne[dim]) {
            ref = at(a, j[0], j[1], j[2], j[3]);
        } else {
            j[dim] -= a->ne[dim];
            ref = at(b, j[0], j[1], j[2], j[3]);
        }
        const int32_t got = at(c, i0, i1, i2, i3);
        if (got != ref) bad++;
    }
    CHECK(bad == 0, "CPU I32 concat dim=%d v=%d ne_a=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nt=%d: %" PRId64 " elements differ",
          dim, v, ne_a[0], ne_a[1], ne_a[2], ne_a[3], n_threads, bad);
    ggml_free(ctx);
}

static ggml_tensor * build_only(ggml_context * ctx, ggml_type type, std::array<int64_t, 4> ne_a, int64_t ne_b_d, int dim, int v) {
    auto ne_b = ne_a; ne_b[dim] = ne_b_d;
    auto mk = [&](std::array<int64_t, 4> ne, bool view) {
        if (!view) return ggml_new_tensor(ctx, type, 4, ne.data());
        auto big = ne; big[0] *= 2; big[1] *= 3; big[2] *= 2;
        ggml_tensor * t = ggml_new_tensor(ctx, type, 4, big.data());
        return ggml_view_4d(ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
    };
    return ggml_concat(ctx, mk(ne_a, v & 1), mk(ne_b, v & 2), dim);
}

int main() {
    // ---- 1. CPU bit-exactness ------------------------------------------------------------------
    for (int nt : { 1, 4 }) {
        for (int v = 0; v < 4; ++v) {
            for (int dim = 0; dim < 4; ++dim) {
                check_cpu({ 11, 12, 13, 14 }, 7, dim, v, nt);   // test-backend-ops' shape
                check_cpu({ 32, 1, 1, 1 }, 3, dim, v, nt);      // a decode-sized table
            }
        }
        // the glm5next indexer tail shapes: [kpool*n_top, n_tokens] ++ [n_tail, n_tokens] on dim 0
        check_cpu({ 32, 1, 1, 1 }, 3, 0, 0, nt);
        check_cpu({ 2048, 1, 1, 1 }, 3, 0, 0, nt);
        check_cpu({ 2048, 128, 1, 1 }, 3, 0, 0, nt);
    }

    // ---- 2. the CUDA supports_op predicate ----------------------------------------------------
    ggml_init_params ip = { 16u*1024*1024, nullptr, true };   // no_alloc: shapes only
    ggml_context * ctx = ggml_init(ip);

    // glm5next indexer tail: decode (n_tokens 1) and a prefill ubatch
    for (int64_t nt : { 1, 128 }) {
        ggml_tensor * sel  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2048, nt);
        ggml_tensor * tail = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, nt);
        ggml_tensor * c = ggml_concat(ctx, sel, tail, 0);
        CHECK(pxa_cuda_concat_i32_shape_ok(c), "indexer tail n_tokens=%" PRId64 " not admitted", nt);
        CHECK(!pxa_cuda_concat_i32_supported(c, false), "switch off must refuse I32 concat");
        CHECK(pxa_cuda_concat_i32_supported(c, true), "switch on must admit the indexer tail");
    }

    // the test-backend-ops grid {11,12,13,14} ++ 7: admitted exactly where a copy-only path exists
    for (int v = 0; v < 4; ++v) {
        for (int dim = 0; dim < 4; ++dim) {
            ggml_tensor * c = build_only(ctx, GGML_TYPE_I32, { 11, 12, 13, 14 }, 7, dim, v);
            const bool want = dim == 0 || (dim == 3 && v == 0);
            CHECK(pxa_cuda_concat_i32_shape_ok(c) == want, "I32 dim=%d v=%d: predicate %d, want %d",
                  dim, v, (int) pxa_cuda_concat_i32_shape_ok(c), (int) want);
            CHECK(!pxa_cuda_concat_i32_supported(c, false), "I32 dim=%d v=%d admitted with switch off", dim, v);
            ggml_tensor * f = build_only(ctx, GGML_TYPE_F32, { 11, 12, 13, 14 }, 7, dim, v);
            CHECK(!pxa_cuda_concat_i32_shape_ok(f), "F32 concat must not go through the I32 predicate");
        }
    }
    // dim 1 / dim 2 become memcpy-able when the trailing dims are 1
    {
        ggml_tensor * c1 = build_only(ctx, GGML_TYPE_I32, { 5, 6, 1, 1 }, 3, 1, 0);
        ggml_tensor * c2 = build_only(ctx, GGML_TYPE_I32, { 5, 6, 7, 1 }, 3, 2, 0);
        CHECK(pxa_cuda_concat_i32_shape_ok(c1), "contiguous dim 1 with ne2*ne3 == 1 should be admitted");
        CHECK(pxa_cuda_concat_i32_shape_ok(c2), "contiguous dim 2 with ne3 == 1 should be admitted");
    }
    // I16 stays refused (not an I32 node); a non-CONCAT node is refused
    {
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_I16, 8);
        ggml_tensor * c = ggml_concat(ctx, a, a, 0);
        CHECK(!pxa_cuda_concat_i32_shape_ok(c), "I16 concat must not be admitted");
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
        CHECK(!pxa_cuda_concat_i32_shape_ok(ggml_cont(ctx, x)), "non-CONCAT node admitted");
    }
    ggml_free(ctx);

    const bool env = pxa_cuda_concat_i32_env_on();
    const char * e = getenv("PXA_GLM_CONCAT_I32_CUDA");
    CHECK(env == (!e || atoi(e) != 0), "env gate disagrees with PXA_GLM_CONCAT_I32_CUDA (default ON, =0 off)");
    printf("PXA_GLM_CONCAT_I32_CUDA=%s -> gate %s\n", e ? e : "(unset)", env ? "ON" : "off");

    if (g_fail) { fprintf(stderr, "FAIL: %d check(s)\n", g_fail); return 1; }
    printf("OK: CPU I32 concat bit-exact; CUDA I32 concat predicate admits exactly the copy-only layouts\n");
    return 0;
}
