// PXA_REDUCE_PINNED_v1 (2026-09-13): the cross-device REDUCE, route by route, on real cards.
//
// GGML_OP_REDUCE is an ALL-reduce: after it runs, EVERY participating source must hold the sum,
// not just the node's home device. It has three device routes -- the in-tree peer path, NCCL, and
// the pinned-host path added with PXA_REDUCE_PINNED -- and the thing that has to be true of the
// two in-tree ones is that they are BIT-IDENTICAL to a single-device sum of the same two
// partials. That is not decoration: the pinned route is what lets the scheduler drop its
// per-reduce backend drain, so if its arithmetic ever drifted from the route it replaces, the
// speed harness would still be green and only a fidelity gate would notice, much later.
//
// The reference is a real single-device sum (ggml_add of the two partials on device 0), not a
// host-side sum, so "bit-identical" means what it says.
//
// Shapes: the decode reduce [n_embd, 1], the MTP verify batch [n_embd, 1 + n_max], a shape whose
// row count is not a multiple of the kernel's vector width (tail coverage), and [n_embd, 64],
// which is DELIBERATELY INELIGIBLE for the pinned route (ne[1] >= 32 belongs to the prefill ring
// path) -- the positive control on the negative, proving the fallback still runs and still adds up.
//
// Route is chosen by argv so the per-process env reads resolve before the first reduce:
//   test-reduce-pinned intree | pinned | nccl
// Needs two CUDA devices; skips cleanly with one.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Defined in ggml/src/ggml-cuda/reduce.cu. The single place route eligibility is decided; the
// scheduler asks the same function before it drops the drain.
extern "C" bool pxa_reduce_pinned_handles(const struct ggml_tensor * dst);

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("%-70s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) { g_fail++; }
}

struct dev_tensor {
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

// One case: build two partials on two devices, reduce, compare to a single-device sum.
static void run_case(ggml_backend_t back[2], int64_t ne0, int64_t ne1,
                     const char * label, bool expect_pinned, bool lever_on) {
    const size_t nelem  = (size_t)(ne0 * ne1);
    const size_t nbytes = nelem * sizeof(float);

    std::mt19937 rng(1234567u ^ (unsigned)(ne0 * 1000003u + ne1));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> h0(nelem), h1(nelem);
    for (size_t i = 0; i < nelem; ++i) { h0[i] = dist(rng); h1[i] = dist(rng); }

    ggml_init_params ip = {};
    ip.mem_size   = 16u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * a1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(a0, "partial0");
    ggml_set_name(a1, "partial1");

    ggml_backend_buffer_t b0 = ggml_backend_alloc_buffer(back[0], nbytes + 256);
    ggml_backend_buffer_t b1 = ggml_backend_alloc_buffer(back[1], nbytes + 256);
    a0->buffer = b0; a0->data = ggml_backend_buffer_get_base(b0);
    a1->buffer = b1; a1->data = ggml_backend_buffer_get_base(b1);

    // ggml_reduce aliases its LAST non-null source, so the node's home device is device 1 and the
    // reduce must be evaluated on that backend -- exactly how the scheduler pins it.
    ggml_tensor * srcs[2] = { a0, a1 };
    ggml_tensor * r = ggml_reduce(ctx, srcs, 2, GGML_OP_ADD);
    r->buffer = a1->buffer;
    r->data   = a1->data;
    ggml_set_name(r, "reduced");

    {
        char what[160];
        snprintf(what, sizeof(what), "%s: pinned route %s this shape", label, expect_pinned ? "accepts" : "refuses");
        check(pxa_reduce_pinned_handles(r) == (expect_pinned && lever_on), what);
    }

    ggml_backend_tensor_set(a0, h0.data(), 0, nbytes);
    ggml_backend_tensor_set(a1, h1.data(), 0, nbytes);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    const bool ok_compute = ggml_backend_graph_compute(back[1], gf) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(back[0]);
    ggml_backend_synchronize(back[1]);
    {
        char what[160];
        snprintf(what, sizeof(what), "%s: reduce computed", label);
        check(ok_compute, what);
    }

    // The comparator: the same two partials added on ONE device.
    ggml_init_params ip2 = ip;
    ggml_context * ctx2 = ggml_init(ip2);
    ggml_tensor * x = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * y = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * z = ggml_add(ctx2, x, y);
    ggml_backend_buffer_t bx = ggml_backend_alloc_buffer(back[0], nbytes + 256);
    ggml_backend_buffer_t by = ggml_backend_alloc_buffer(back[0], nbytes + 256);
    ggml_backend_buffer_t bz = ggml_backend_alloc_buffer(back[0], nbytes + 256);
    x->buffer = bx; x->data = ggml_backend_buffer_get_base(bx);
    y->buffer = by; y->data = ggml_backend_buffer_get_base(by);
    z->buffer = bz; z->data = ggml_backend_buffer_get_base(bz);
    ggml_backend_tensor_set(x, h0.data(), 0, nbytes);
    ggml_backend_tensor_set(y, h1.data(), 0, nbytes);
    ggml_cgraph * gf2 = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf2, z);
    ggml_backend_graph_compute(back[0], gf2);
    ggml_backend_synchronize(back[0]);

    std::vector<float> ref(nelem), got0(nelem), got1(nelem);
    ggml_backend_tensor_get(z,  ref.data(),  0, nbytes);
    ggml_backend_tensor_get(a0, got0.data(), 0, nbytes);
    ggml_backend_tensor_get(a1, got1.data(), 0, nbytes);

    size_t bad0 = 0, bad1 = 0;
    double maxabs = 0.0;
    for (size_t i = 0; i < nelem; ++i) {
        if (memcmp(&got0[i], &ref[i], sizeof(float)) != 0) { bad0++; }
        if (memcmp(&got1[i], &ref[i], sizeof(float)) != 0) { bad1++; }
        const double d = fabs((double)got0[i] - (double)ref[i]);
        if (d > maxabs) { maxabs = d; }
    }

    char what[200];
    snprintf(what, sizeof(what), "%s: device 0 partial holds the sum, bit-identical (%zu/%zu differ)", label, bad0, nelem);
    check(bad0 == 0, what);
    snprintf(what, sizeof(what), "%s: device 1 partial holds the sum, bit-identical (%zu/%zu differ)", label, bad1, nelem);
    check(bad1 == 0, what);
    if (bad0 || bad1) {
        printf("   max |reduce - single-device sum| = %.9g\n", maxabs);
    }

    ggml_backend_buffer_free(bx); ggml_backend_buffer_free(by); ggml_backend_buffer_free(bz);
    ggml_backend_buffer_free(b0); ggml_backend_buffer_free(b1);
    ggml_free(ctx2);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const std::string route = argc > 1 ? argv[1] : "intree";

    // Resolved before the first reduce, because every route selector is a per-process static.
    bool lever_on = false;
    if (route == "pinned") {
        setenv("PXA_REDUCE_PINNED", "1", 1);
        setenv("PXA_REDUCE_NCCL",   "0", 1);
        lever_on = true;
    } else if (route == "nccl") {
        setenv("PXA_REDUCE_NCCL",   "1", 1);
        unsetenv("PXA_REDUCE_PINNED");
    } else {
        setenv("PXA_REDUCE_NCCL",   "0", 1);
        unsetenv("PXA_REDUCE_PINNED");
    }
    setenv("PXA_RDBG", "400", 1);   // every branch this run takes is named in the log

    const int ndev = ggml_backend_cuda_get_device_count();
    printf("test-reduce-pinned: route=%s devices=%d\n", route.c_str(), ndev);
    if (ndev < 2) {
        printf("SKIP: needs two CUDA devices, found %d\n", ndev);
        return 0;
    }

    ggml_backend_t back[2] = { ggml_backend_cuda_init(0, nullptr), ggml_backend_cuda_init(1, nullptr) };
    if (!back[0] || !back[1]) {
        printf("FAIL: could not initialise both CUDA backends\n");
        return 1;
    }

    const int64_t n_embd = 5120;   // Qwen3.8-27B
    run_case(back, n_embd,  1, "decode      [5120,1] ", true,  lever_on);
    run_case(back, n_embd,  4, "verify batch[5120,4] ", true,  lever_on);
    run_case(back,     37,  3, "tail        [37,3]   ", true,  lever_on);
    run_case(back, n_embd, 64, "prefill     [5120,64]", false, lever_on);   // ineligible on purpose

    ggml_backend_free(back[0]);
    ggml_backend_free(back[1]);

    printf("test-reduce-pinned: route=%s failures=%d\n", route.c_str(), g_fail);
    return g_fail == 0 ? 0 : 1;
}
