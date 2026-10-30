// ggml-alloc: a graph leaf that no node consumes must still be re-planned when it grows.
//
// ggml_gallocr_alloc_graph() reuses the reserved plan unless ggml_gallocr_needs_realloc() finds a
// tensor that no longer fits. That check used to cover nodes and the sources nodes read, never the
// leaves themselves, so an input expanded into the graph with no consumer (glm5next's KQ_mask on
// the decode-gather path / in its NextN head) kept its old slot when it grew with n_kv while the
// graph's node and leaf counts stayed the same; the input fill then wrote past the slot and past the
// end of the compute buffer (heap corruption at MTP n_max=4 on the tiny glm5next fixture).
//
// The test reserves a graph whose unconsumed leaf is [64,16] f32, then allocates the same-topology
// graph with the leaf at [96,16] (and once more at [64,16] to check reuse still works) and requires
// every tensor to lie inside the buffer and the leaf not to overlap the node.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>

struct built { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * mask; ggml_tensor * x; ggml_tensor * y; };

static built build(int64_t n_kv) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    built b;
    b.ctx = ggml_init(ip);
    b.gf  = ggml_new_graph(b.ctx);
    b.mask = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, n_kv, 16);   // an input no node reads
    ggml_set_name(b.mask, "KQ_mask"); ggml_set_input(b.mask);
    b.x = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 256);
    ggml_set_name(b.x, "x"); ggml_set_input(b.x);
    b.y = ggml_scale(b.ctx, b.x, 2.0f);
    ggml_set_name(b.y, "y"); ggml_set_output(b.y);
    ggml_build_forward_expand(b.gf, b.mask);
    ggml_build_forward_expand(b.gf, b.y);
    return b;
}

static bool inside(const ggml_tensor * t) {
    if (!t->buffer || !t->data) return false;
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(t->buffer);
    const uintptr_t end  = base + ggml_backend_buffer_get_size(t->buffer);
    const uintptr_t lo   = (uintptr_t) t->data;
    return lo >= base && lo + ggml_nbytes(t) <= end;
}

static bool disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
    const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
    return a1 <= b0 || b1 <= a0;
}

int main() {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    int fails = 0;

    built r = build(64);
    if (r.gf->n_leafs < 2) { fprintf(stderr, "FAIL setup: the mask is not a leaf (n_leafs=%d)\n", r.gf->n_leafs); return 1; }
    if (!ggml_gallocr_reserve(galloc, r.gf)) { fprintf(stderr, "FAIL reserve\n"); return 1; }
    const size_t reserved = ggml_gallocr_get_buffer_size(galloc, 0);
    ggml_free(r.ctx);

    const int64_t steps[] = { 96, 64, 160 };
    for (int64_t n_kv : steps) {
        built b = build(n_kv);
        if (!ggml_gallocr_alloc_graph(galloc, b.gf)) {
            fprintf(stderr, "FAIL n_kv=%lld: alloc_graph refused\n", (long long) n_kv); fails++; ggml_free(b.ctx); continue;
        }
        const bool ok = inside(b.mask) && inside(b.x) && inside(b.y) && disjoint(b.mask, b.y) && disjoint(b.mask, b.x);
        printf("%s n_kv=%lld: mask %zu B, buffer %zu B (reserved %zu B)\n", ok ? "PASS" : "FAIL", (long long) n_kv,
               ggml_nbytes(b.mask), ggml_gallocr_get_buffer_size(galloc, 0), reserved);
        fails += ok ? 0 : 1;
        ggml_free(b.ctx);
    }
    ggml_gallocr_free(galloc);
    printf("RESULT %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
