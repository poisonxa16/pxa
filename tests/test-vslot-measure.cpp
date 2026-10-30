// bug #348: the measure-only compute-buffer plan used to size speculative verify graph slots.
//
// A verify graph slot's arena is now reserved when the slot is created, for the largest KV extent
// that leaves a VRAM margin free; the extent is found by measuring candidate graphs without
// allocating (pxa_ggml_backend_sched_measure -> pxa_ggml_gallocr_measure_n). The decision is only
// as good as the measurement, so this test requires, on the CPU backend:
//   1. the measured bytes equal what a real reserve of the same graph allocates (gallocr and sched);
//   2. a measurement allocates nothing, and the allocator it ran on still allocates and computes the
//      graph correctly afterwards (the unbacked plan is re-planned, never used);
//   3. the measured size does not shrink when the KV extent grows (the cap search assumes it);
//   4. a scheduler reserved at the largest extent runs every smaller extent without growing.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" bool pxa_ggml_gallocr_measure_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
        const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes);
extern "C" int pxa_ggml_backend_sched_measure(ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        ggml_backend_buffer_type_t * bufts, size_t * sizes, int n_max);

#define PXA_VSLOT_MAX_BUFS 64   // >= GGML_SCHED_MAX_BACKENDS (ggml-backend.cpp)

// an attention-shaped toy: q [d, w], k [d, n_kv] -> kq [n_kv, w] -> softmax -> kqv with v [n_kv, d]
struct toy { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * q; ggml_tensor * k; ggml_tensor * v; ggml_tensor * out; };

static toy build(int64_t n_kv, int64_t w) {
    const int64_t d = 64;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    toy t;
    t.ctx = ggml_init(ip);
    t.gf  = ggml_new_graph(t.ctx);
    t.q = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, d, w);     ggml_set_input(t.q);
    t.k = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, d, n_kv);  ggml_set_input(t.k);
    t.v = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, n_kv, d);  ggml_set_input(t.v);
    ggml_tensor * kq  = ggml_mul_mat(t.ctx, t.k, t.q);                 // [n_kv, w]
    ggml_tensor * sm  = ggml_soft_max(t.ctx, ggml_scale(t.ctx, kq, 0.125f));
    ggml_tensor * kqv = ggml_mul_mat(t.ctx, t.v, sm);                  // [d, w]
    t.out = ggml_add(t.ctx, kqv, t.q);
    ggml_set_output(t.out);
    ggml_build_forward_expand(t.gf, t.out);
    return t;
}

static void fill(ggml_tensor * x, float s) {
    std::vector<float> v(ggml_nelements(x));
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.01f * (float) i + s);
    ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
}

static bool inside(const ggml_tensor * t) {
    if (!t->buffer || !t->data) return false;
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(t->buffer);
    const uintptr_t end  = base + ggml_backend_buffer_get_size(t->buffer);
    return (uintptr_t) t->data >= base && (uintptr_t) t->data + ggml_nbytes(t) <= end;
}

int main() {
    int fails = 0;
    const int64_t W = 4;

    // 1+2 (gallocr): measure == reserve, measurement allocates nothing, the allocator still works after it
    for (int64_t n_kv : { 256, 1024, 4096 }) {
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        ggml_gallocr_t gb = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        toy a = build(n_kv, W), b = build(n_kv, W);
        size_t meas[1] = { 0 };
        if (!pxa_ggml_gallocr_measure_n(ga, a.gf, nullptr, nullptr, meas)) { fprintf(stderr, "FAIL measure n_kv=%lld\n", (long long) n_kv); fails++; }
        if (ggml_gallocr_get_buffer_size(ga, 0) != 0) { fprintf(stderr, "FAIL measure allocated %zu bytes\n", ggml_gallocr_get_buffer_size(ga, 0)); fails++; }
        if (!ggml_gallocr_reserve(gb, b.gf)) { fprintf(stderr, "FAIL reserve\n"); fails++; }
        const size_t real = ggml_gallocr_get_buffer_size(gb, 0);
        if (meas[0] != real || real == 0) { fprintf(stderr, "FAIL n_kv=%lld: measured %zu, reserved %zu\n", (long long) n_kv, meas[0], real); fails++; }
        // the measured allocator must re-plan, not place tensors through its NULL buffer
        if (!ggml_gallocr_alloc_graph(ga, a.gf)) { fprintf(stderr, "FAIL alloc after measure\n"); fails++; }
        else if (!inside(a.out) || !inside(a.q)) { fprintf(stderr, "FAIL tensors outside the buffer after measure\n"); fails++; }
        printf("gallocr n_kv=%-5lld measured %8zu reserved %8zu\n", (long long) n_kv, meas[0], real);
        ggml_free(a.ctx); ggml_free(b.ctx);
        ggml_gallocr_free(ga); ggml_gallocr_free(gb);
    }

    // 1+3+4 (scheduler): measure == reserve, monotone in n_kv, the reserved max extent covers smaller ones
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[1] = { cpu };
    const size_t gsize = 1024;
    size_t prev_meas = 0;
    std::vector<float> ref_out, got_out;
    for (int64_t n_kv : { 256, 512, 2048, 8192 }) {
        ggml_backend_sched_t sm = ggml_backend_sched_new(backends, nullptr, 1, gsize, false);
        ggml_backend_sched_t sr = ggml_backend_sched_new(backends, nullptr, 1, gsize, false);
        toy a = build(n_kv, W), b = build(n_kv, W);
        ggml_backend_buffer_type_t bufts[PXA_VSLOT_MAX_BUFS];
        size_t sizes[PXA_VSLOT_MAX_BUFS];
        const int n = pxa_ggml_backend_sched_measure(sm, a.gf, bufts, sizes, PXA_VSLOT_MAX_BUFS);
        if (n != 1) { fprintf(stderr, "FAIL sched measure returned %d\n", n); fails++; }
        if (ggml_backend_sched_get_buffer_size(sm, cpu) != 0) { fprintf(stderr, "FAIL sched measure allocated\n"); fails++; }
        if (!ggml_backend_sched_reserve(sr, b.gf)) { fprintf(stderr, "FAIL sched reserve\n"); fails++; }
        const size_t real = ggml_backend_sched_get_buffer_size(sr, cpu);
        if (n == 1 && (sizes[0] != real || bufts[0] != ggml_backend_cpu_buffer_type())) {
            fprintf(stderr, "FAIL sched n_kv=%lld: measured %zu, reserved %zu\n", (long long) n_kv, sizes[0], real); fails++;
        }
        if (n == 1 && sizes[0] < prev_meas) { fprintf(stderr, "FAIL not monotone: %zu < %zu\n", sizes[0], prev_meas); fails++; }
        if (n == 1) prev_meas = sizes[0];
        // the measured scheduler computes correctly afterwards, and matches the reserved one bit for bit
        ggml_backend_sched_reset(sm);
        if (!ggml_backend_sched_alloc_graph(sm, a.gf)) { fprintf(stderr, "FAIL sched alloc after measure\n"); fails++; }
        else {
            fill(a.q, 0.1f); fill(a.k, 0.2f); fill(a.v, 0.3f);
            ggml_backend_sched_graph_compute(sm, a.gf);
            got_out.resize(ggml_nelements(a.out));
            ggml_backend_tensor_get(a.out, got_out.data(), 0, ggml_nbytes(a.out));
            ggml_backend_sched_reset(sr);
            ggml_backend_sched_alloc_graph(sr, b.gf);
            fill(b.q, 0.1f); fill(b.k, 0.2f); fill(b.v, 0.3f);
            ggml_backend_sched_graph_compute(sr, b.gf);
            ref_out.resize(ggml_nelements(b.out));
            ggml_backend_tensor_get(b.out, ref_out.data(), 0, ggml_nbytes(b.out));
            if (memcmp(ref_out.data(), got_out.data(), ref_out.size() * sizeof(float)) != 0) { fprintf(stderr, "FAIL outputs differ n_kv=%lld\n", (long long) n_kv); fails++; }
        }
        printf("sched   n_kv=%-5lld measured %8zu reserved %8zu\n", (long long) n_kv, n == 1 ? sizes[0] : 0, real);
        ggml_free(a.ctx); ggml_free(b.ctx);
        ggml_backend_sched_free(sm); ggml_backend_sched_free(sr);
    }

    // 4: reserve at the largest extent, then run the smaller extents: the buffer must not grow
    {
        ggml_backend_sched_t s = ggml_backend_sched_new(backends, nullptr, 1, gsize, false);
        toy big = build(8192, W);
        ggml_backend_sched_reserve(s, big.gf);
        const size_t cap = ggml_backend_sched_get_buffer_size(s, cpu);
        ggml_free(big.ctx);
        for (int64_t n_kv : { 256, 1024, 4096, 8192 }) {
            ggml_backend_sched_reset(s);
            toy t = build(n_kv, W);
            if (!ggml_backend_sched_alloc_graph(s, t.gf)) { fprintf(stderr, "FAIL alloc n_kv=%lld under the cap\n", (long long) n_kv); fails++; }
            if (ggml_backend_sched_get_buffer_size(s, cpu) != cap) { fprintf(stderr, "FAIL buffer grew at n_kv=%lld\n", (long long) n_kv); fails++; }
            ggml_free(t.ctx);
        }
        ggml_backend_sched_free(s);
    }
    ggml_backend_free(cpu);

    if (fails) { fprintf(stderr, "test-vslot-measure: %d FAIL\n", fails); return 1; }
    printf("test-vslot-measure: OK\n");
    return 0;
}
