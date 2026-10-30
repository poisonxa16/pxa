// ggml-backend: a CPU buffer's reported size must be usable from its aligned base.
//
// The CPU buffer type mallocs size + TENSOR_ALIGNMENT (32) bytes and get_base() rounds the malloc
// pointer up to 32. It used to report the INFLATED size (size + 32) as the buffer size, so when
// malloc returned an address that is 16 mod 32, [base, base + get_size) ran 16 bytes past the block.
// ggml_gallocr_reserve_n() keeps a buffer while the new plan is <= get_size, so a re-plan that grew
// by exactly one alignment unit was placed in the old, too small block and its last tensor wrote
// into the next heap chunk's header. Seen as "free(): invalid next size" / "double free or
// corruption (!prev)" with glm5next MTP n_max=4 (verify graph plan 231712 -> 231744 bytes, the
// kda_delta_net state at the end of the buffer); only with a heap layout that hands out a 16 mod 32
// pointer, so ASan and other glibc versions hid it.
//
// Part 1 allocates CPU buffers of many sizes and requires [base, base + size) inside the block.
// Part 2 replays the grow-by-one-unit gallocr re-plan and requires every tensor inside the block.
// The block end is malloc_usable_size (glibc only; elsewhere the test only checks alignment).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#define NH_HAVE_USABLE 1
#endif

static int g_fail = 0, g_misaligned = 0;

// true when [p, p + n) lies inside the malloc'd block that starts at blk
static bool in_block(const void * blk, const void * p, size_t n) {
#ifdef NH_HAVE_USABLE
    const uintptr_t b = (uintptr_t) blk, e = b + malloc_usable_size(const_cast<void *>(blk));
    return (uintptr_t) p >= b && (uintptr_t) p + n <= e;
#else
    (void) blk; (void) p; (void) n;
    return true;
#endif
}

static void check_buffer(ggml_backend_buffer_t buf, size_t want, const char * what) {
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(buf);
    const size_t    size = ggml_backend_buffer_get_size(buf);
    if (((uintptr_t) buf->context) % 32 != 0) g_misaligned++;
    if (base % 32 != 0 || size < want || !in_block(buf->context, (void *) base, size)) {
        fprintf(stderr, "FAIL %s: block=%p base=%p size=%zu want=%zu (base+size outside the block)\n",
                what, buf->context, (void *) base, size, want);
        g_fail++;
    }
}

struct built { ggml_context * ctx; ggml_cgraph * gf; };

// a [n] input, b [8] input, out = view(a, 8) + b: every tensor's size is a multiple of 32, so the
// tensor at the end of the plan fills its slot exactly; growing a by 8 floats grows the plan by 32.
static built build(int64_t n) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    built r;
    r.ctx = ggml_init(ip);
    r.gf  = ggml_new_graph(r.ctx);
    ggml_tensor * a = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, n); ggml_set_name(a, "a"); ggml_set_input(a);
    ggml_tensor * b = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, 8); ggml_set_name(b, "b"); ggml_set_input(b);
    ggml_tensor * o = ggml_add(r.ctx, ggml_view_1d(r.ctx, a, 8, 0), b); ggml_set_name(o, "out"); ggml_set_output(o);
    ggml_build_forward_expand(r.gf, o);
    return r;
}

static void check_graph(const built & g, const char * what) {
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
        const ggml_tensor * t = ggml_graph_node(g.gf, i);
        if (t->view_src || !t->buffer) continue;
        if (!in_block(t->buffer->context, t->data, ggml_nbytes(t))) {
            fprintf(stderr, "FAIL %s: node %s [%p, +%zu) outside block %p\n", what, t->name, t->data, ggml_nbytes(t), t->buffer->context);
            g_fail++;
        }
    }
    for (int i = 0; i < g.gf->n_leafs; ++i) {
        const ggml_tensor * t = g.gf->leafs[i];
        if (t->view_src || !t->buffer) continue;
        if (!in_block(t->buffer->context, t->data, ggml_nbytes(t))) {
            fprintf(stderr, "FAIL %s: leaf %s [%p, +%zu) outside block %p\n", what, t->name, t->data, ggml_nbytes(t), t->buffer->context);
            g_fail++;
        }
    }
}

int main() {
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    // part 1: plain buffers, kept alive together so the heap hands out both 0 and 16 mod 32 blocks
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<void *> pads;
    for (int k = 0; k < 96; ++k) {
        const size_t want = 1024 + 32 * (size_t) k;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, want);
        if (!buf) { fprintf(stderr, "FAIL alloc %zu\n", want); return 1; }
        check_buffer(buf, want, "buffer");
        bufs.push_back(buf);
        pads.push_back(malloc(16 + 16 * (k % 3)));   // shift the next block's alignment
    }
    for (auto * b : bufs) ggml_backend_buffer_free(b);

    // part 2: reserve a plan, then a same-topology graph whose plan is exactly one 32-byte unit larger
    for (int k = 0; k < 64; ++k) {
        const int64_t n = 1024 + 8 * k;
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        built g1 = build(n);
        if (!ggml_gallocr_alloc_graph(ga, g1.gf)) { fprintf(stderr, "FAIL alloc g1\n"); return 1; }
        check_graph(g1, "g1");
        built g2 = build(n + 8);
        if (!ggml_gallocr_alloc_graph(ga, g2.gf)) { fprintf(stderr, "FAIL alloc g2\n"); return 1; }
        const ggml_tensor * out = ggml_graph_node(g2.gf, -1);
        if (out->buffer && ((uintptr_t) out->buffer->context) % 32 != 0) g_misaligned++;
        check_graph(g2, "g2 (plan grown by 32 bytes)");
        ggml_free(g1.ctx);
        ggml_free(g2.ctx);
        ggml_gallocr_free(ga);
        pads.push_back(malloc(16 + 16 * (k % 3)));
    }
    for (auto * p : pads) free(p);

    printf("test-cpu-buffer-bounds: %d failure(s); %d block(s) were 16 mod 32 (the case under test)%s\n",
           g_fail, g_misaligned,
#ifdef NH_HAVE_USABLE
           ""
#else
           " [no malloc_usable_size: alignment checked only]"
#endif
    );
    return g_fail == 0 ? 0 : 1;
}
