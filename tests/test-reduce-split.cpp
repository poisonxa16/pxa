// PXA 2026-09-08: GGML_OP_REDUCE -- the -sm graph all-reduce primitive.
//
// WHY THIS TEST EXISTS. `-sm graph` (build-time tensor parallelism) corrupted the DeltaNet
// hybrid architectures for over a month, and the cause was not in any kernel: it was that the
// REDUCE node lost its consumer during GRAPH CONSTRUCTION. Nothing could catch that, because
// GGML_OP_REDUCE had no CPU implementation at all and `-sm graph` needs >= 2 CUDA devices, so
// there was no test of any kind for this op. This file is the regression test that the CPU
// fallback was added to make possible.
//
// Three cases, in increasing order of what they would have caught:
//
//   1. ARITHMETIC. The reduce is an ALL-reduce: every participating source must end up holding
//      the sum, not just the node. Consumers read their own src[id], so a reduce-to-one would
//      look correct on the home device and silently wrong everywhere else.
//   2. CONTAINER MODE. op_params[3] == 1 means the reduce is switched off and the node is only a
//      container for its sources (PXA_REPLICATE_RECURRENT and friends). The sources must be left
//      untouched.
//   3. THE VIEW INVARIANT -- the one that actually mattered. ggml_reduce builds its result as a
//      view of its last non-null source, and TWO consumers used to identify that source by
//      pointer identity against result->view_src:
//        * the scheduler's REDUCE backend pinning, and
//        * llm_build_context::get_input_tensor_sm_graph, which is what gives the HOME device an
//          edge to the reduce node.
//      But ggml_new_tensor_impl deliberately collapses one level of view onto the base, so when
//      the last partial is ITSELF a view (the DeltaNet partial is a bare ggml_reshape_2d at
//      n_tokens <= 32, i.e. every decode step) that identity fails, and the reduce ends up with
//      NO consumer in the graph. Case 3 pins the collapse down as a property of ggml, so that
//      the "identify the home source by its SLOT, not by pointer identity" rule the consumers
//      now use cannot be quietly reverted.
//
// CPU only, no GPU, no model.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("%-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

// The rule ggml_reduce uses to pick the source its result aliases, and therefore the rule the
// consumers must use to find the home device. Deliberately written out here rather than shared,
// so that a change to it has to be made in two places on purpose.
static int last_non_null_slot(const ggml_tensor * reduce) {
    int last = -1;
    for (int j = 0; j < reduce->op_params[1]; ++j) {
        if (reduce->src[j]) last = j;
    }
    return last;
}

static ggml_context * make_ctx(size_t mb) {
    ggml_init_params p = {};
    p.mem_size   = mb * 1024 * 1024;
    p.mem_buffer = NULL;
    p.no_alloc   = false;
    return ggml_init(p);
}

// ---------------------------------------------------------------------------------------------
// 1 + 2: arithmetic, and the container early-out
// ---------------------------------------------------------------------------------------------
static void test_reduce_add(bool container_mode) {
    const int64_t NE0 = 37;   // deliberately not a multiple of anything
    const int64_t NE1 = 5;
    const int     N   = 4;    // four "devices"

    ggml_context * ctx = make_ctx(16);

    ggml_tensor * parts[N];
    std::vector<float> expected((size_t)(NE0*NE1), 0.0f);

    for (int j = 0; j < N; ++j) {
        parts[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, NE1);
        float * d = (float *) parts[j]->data;
        for (int64_t i = 0; i < NE0*NE1; ++i) {
            // distinct per (slot, element) so a dropped or double-counted slot is visible
            d[i] = (float)(j + 1) * 0.25f + (float)i * 0.001f;
            expected[i] += d[i];
        }
    }

    ggml_tensor * red = ggml_reduce(ctx, parts, N, GGML_OP_ADD);
    if (container_mode) {
        red->op_params[3] = 1;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, red);
    ggml_graph_compute_with_ctx(ctx, gf, 3); // >1 thread: the op splits over the element range

    bool all_ok = true;
    for (int j = 0; j < N; ++j) {
        const float * d = (const float *) parts[j]->data;
        for (int64_t i = 0; i < NE0*NE1; ++i) {
            const float want = container_mode
                             ? (float)(j + 1) * 0.25f + (float)i * 0.001f   // untouched
                             : expected[i];                                  // the sum
            if (fabsf(d[i] - want) > 1e-4f * fmaxf(1.0f, fabsf(want))) {
                if (all_ok) {
                    printf("  first mismatch: slot %d elem %lld got %.6f want %.6f\n",
                           j, (long long) i, (double) d[i], (double) want);
                }
                all_ok = false;
            }
        }
    }
    check(all_ok, container_mode
                ? "container mode (op_params[3]=1) leaves every source untouched"
                : "all-reduce: EVERY source holds the sum, not just the node");

    if (!container_mode) {
        // The node is a view of its last source, so it must agree with it.
        const float * node = (const float *) red->data;
        const float * home = (const float *) parts[N-1]->data;
        check(node == home, "reduce node aliases its last source");
    }

    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------------
// 3: the view collapse, and the slot rule that survives it
// ---------------------------------------------------------------------------------------------
static void test_view_collapse(void) {
    const int64_t NE0 = 16;
    const int64_t NE1 = 4;
    const int     N   = 3;

    ggml_context * ctx = make_ctx(16);

    ggml_tensor * parts[N];
    for (int j = 0; j < N; ++j) {
        ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, NE1);
        float * d = (float *) base->data;
        for (int64_t i = 0; i < NE0*NE1; ++i) d[i] = (float)(j + 1);
        // The LAST partial is a view of its base -- exactly the shape build_gated_output
        // produces (it ends in ggml_reshape_2d) whenever the cast that would de-view it is
        // skipped, i.e. at n_tokens <= 32 or under -grt f32.
        parts[j] = (j == N-1) ? ggml_reshape_2d(ctx, base, NE0, NE1) : base;
    }

    ggml_tensor * red = ggml_reduce(ctx, parts, N, GGML_OP_ADD);

    // The property that broke everything: with a view-valued last source, view_src is the BASE.
    check(red->view_src != parts[N-1],
          "ggml collapses the view chain: view_src != the last source");
    check(red->view_src == parts[N-1]->view_src,
          "  ...it points at that source's base instead");

    // ...but the data pointer is still the last source's, so the SLOT rule is sound and is what
    // the scheduler and get_input_tensor_sm_graph must use to find the home device.
    check(red->data == parts[N-1]->data,
          "the node still aliases the last source's data");
    check(last_non_null_slot(red) == N-1,
          "the last-non-null-slot rule finds the home source anyway");

    // And a sparse slot map (the nhave != nreduce case, e.g. attention where the head count does
    // not divide the device count) must not confuse it.
    ggml_tensor * sparse[4] = { parts[0], NULL, parts[1], NULL };
    ggml_tensor * red2 = ggml_reduce(ctx, sparse, 4, GGML_OP_ADD);
    check(red2->op_params[2] == 2,          "nhave counts only non-null sources");
    check(red2->op_params[1] == 4,          "nreduce keeps the full slot count");
    check(last_non_null_slot(red2) == 2,    "slot rule skips null slots (nhave != nreduce)");

    // The arithmetic must still be right through a view-valued source.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, red);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    const float want = 1.0f + 2.0f + 3.0f;
    bool ok = true;
    for (int j = 0; j < N; ++j) {
        const float * d = (const float *) parts[j]->data;
        for (int64_t i = 0; i < NE0*NE1; ++i) {
            if (fabsf(d[i] - want) > 1e-5f) { ok = false; break; }
        }
    }
    check(ok, "all-reduce is correct when a source is a view");

    ggml_free(ctx);
}

int main(void) {
    printf("test-reduce-split: GGML_OP_REDUCE (the -sm graph all-reduce primitive)\n\n");

    test_reduce_add(/*container_mode =*/ false);
    test_reduce_add(/*container_mode =*/ true);
    test_view_collapse();

    printf("\n%s\n", g_fail == 0 ? "ALL OK" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
