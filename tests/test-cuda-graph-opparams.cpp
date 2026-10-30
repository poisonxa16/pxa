// bug #343 regression: a CUDA graph replay must pick up a change of op_params.
//
// The CUDA backend replays a captured graph when every node still has the same op, shapes, strides and
// addresses (ggml_graph_node_has_matching_properties, ggml-cuda.cu). Most ops take their kernel arguments from
// op_params (pad offsets, clamp bounds, the leaky-relu slope, the norm eps, ...), and the match used to compare
// op_params for GGML_OP_SCALE only: a graph that differed from the captured one ONLY in an op_param replayed the
// old kernel arguments. CUDA graphs are on by default on sm_80+, so an RTX 3080 Ti failed test-pxqn-backend's
// PAD_EXT, K-STRADDLE (ntok=1) and PXQN_RHT checks while a P100 / V100 (graphs off by arch) passed them.
//
// Each case below builds ONE node on the CUDA backend, computes it three times with parameters A (capture +
// replays), rewrites the node's op_params in place to parameters B (same node struct, same graph, same
// addresses: exactly the replay-match situation) and computes twice more. The A result and both B results must equal
// fresh CUDA evaluations of an A and a B node (own node structs and graphs) bit for bit, and the two fresh results must
// differ (else the case would prove nothing). PXA_CUDA_GRAPHS_PASCAL=1 is set before the backend starts so the graphs
// are captured on sm_60 / sm_70 too: the check runs on every card, not only where graphs are on by default.
// PXA_CGRAPH_OPPARAMS=0 (the old SCALE-only compare) must make the non-SCALE cases fail.
//
// Needs a CUDA device at run time: built, not registered with ctest (like test-pxqn-backend).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#include "pxa-test-portable.h"   // setenv / unsetenv on native Windows too

using mk_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor *)>;

static int g_fail = 0, g_pass = 0;

static void check(bool ok, const char * msg) {
    printf("%s %s\n", ok ? "  ok  " : "  FAIL", msg);
    (ok ? g_pass : g_fail)++;
}

static ggml_context * new_ctx() {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    return ggml_init(ip);
}

static bool compute(ggml_backend_t be, ggml_cgraph * gf) {
    return ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
}

static std::vector<float> get(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(float));
    return v;
}

// a fresh evaluation of mk(x): its own context, node struct, buffer and graph. The context and buffer stay alive until
// the case ends (keep), so no later node of the case can reuse this node's address or this graph's key (the graph cache is
// keyed by the first node's address: a reused address would turn a fresh eval into a replay candidate itself).
struct keep_t { std::vector<ggml_context *> ctx; std::vector<ggml_backend_buffer_t> buf;
    ~keep_t() { for (auto b : buf) ggml_backend_buffer_free(b); for (auto c : ctx) ggml_free(c); } };

static std::vector<float> fresh(ggml_backend_t be, const mk_fn & mk, const std::vector<float> & xv, int64_t n0, int64_t n1, int64_t n2,
                                keep_t & keep) {
    ggml_context * ctx = new_ctx();
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
    ggml_tensor * y = mk(ctx, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    keep.ctx.push_back(ctx); keep.buf.push_back(buf);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    std::vector<float> r;
    if (compute(be, gf)) r = get(y);
    return r;
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static void run_case(ggml_backend_t be, const char * name, const mk_fn & mk_a, const mk_fn & mk_b) {
    const int64_t n0 = 320, n1 = 10, n2 = 7;
    std::mt19937 rng(0x343u);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);
    std::vector<float> xv(n0*n1*n2);
    for (auto & v : xv) v = dist(rng);

    keep_t keep;
    const auto ref_a = fresh(be, mk_a, xv, n0, n1, n2, keep);
    const auto ref_b = fresh(be, mk_b, xv, n0, n1, n2, keep);

    ggml_context * ctx = new_ctx();
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
    ggml_tensor * y = mk_a(ctx, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    keep.ctx.push_back(ctx); keep.buf.push_back(buf);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    bool ok = compute(be, gf) && compute(be, gf) && compute(be, gf);   // capture, then replays
    const auto got_a = get(y);

    // parameters B on the SAME node: take the op_params of a B node built in a scratch context
    ggml_context * cs = new_ctx();
    ggml_tensor * yb = mk_b(cs, ggml_new_tensor_3d(cs, GGML_TYPE_F32, n0, n1, n2));
    const bool same_shape = ggml_are_same_shape(yb, y) && yb->op == y->op;
    memcpy(y->op_params, yb->op_params, sizeof(y->op_params));
    ggml_free(cs);

    ok = ok && compute(be, gf);
    const auto got_b1 = get(y);
    ok = ok && compute(be, gf);
    const auto got_b2 = get(y);

    const bool valid = !same_bits(ref_a, ref_b);   // else the case would prove nothing
    const bool a = same_bits(got_a, ref_a), b1 = same_bits(got_b1, ref_b), b2 = same_bits(got_b2, ref_b);
    char msg[256];
    snprintf(msg, sizeof msg, "%-10s one captured node, op_params A -> B in place; == fresh CUDA eval bit for bit: A %s, B 1st %s, B 2nd %s%s%s",
             name, a ? "yes" : "NO", b1 ? "yes" : "NO", b2 ? "yes" : "NO", valid ? "" : "  [A == B: case proves nothing]",
             same_shape ? "" : "  [B changes the shape: bad case]");
    check(ok && same_shape && valid && a && b1 && b2, msg);
}

int main() {
    setenv("PXA_CUDA_GRAPHS_PASCAL", "1", 0);   // capture on sm_60 / sm_70 too (graphs are on by default from sm_80)
    if (getenv("GGML_CUDA_DISABLE_GRAPHS")) {
        printf("GGML_CUDA_DISABLE_GRAPHS is set: no graph is captured, the cases below cannot fail\n");
    }
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) { printf("no CUDA device\n"); return 2; }
    char desc[256] = {0};
    ggml_backend_cuda_get_device_description(0, desc, sizeof desc);
    printf("test-cuda-graph-opparams: %s (cc %d), PXA_CGRAPH_OPPARAMS=%s\n", desc, ggml_backend_cuda_get_device_cc(0),
           getenv("PXA_CGRAPH_OPPARAMS") ? getenv("PXA_CGRAPH_OPPARAMS") : "unset (1)");

    // front pad <-> back pad: same output shape, different placement (the K-STRADDLE shard pair)
    run_case(cuda, "PAD_EXT",
        [](ggml_context * c, ggml_tensor * x) { return ggml_pad_ext(c, x, 0, 64, 0, 0, 0, 0, 0, 0); },
        [](ggml_context * c, ggml_tensor * x) { return ggml_pad_ext(c, x, 64, 0, 0, 0, 0, 0, 0, 0); });
    run_case(cuda, "CLAMP",
        [](ggml_context * c, ggml_tensor * x) { return ggml_clamp(c, x, -1.0f, 1.0f); },
        [](ggml_context * c, ggml_tensor * x) { return ggml_clamp(c, x, -0.25f, 0.5f); });
    run_case(cuda, "LEAKY_RELU",
        [](ggml_context * c, ggml_tensor * x) { return ggml_leaky_relu(c, x, 0.1f, false); },
        [](ggml_context * c, ggml_tensor * x) { return ggml_leaky_relu(c, x, 0.3f, false); });
    run_case(cuda, "RMS_NORM",
        [](ggml_context * c, ggml_tensor * x) { return ggml_rms_norm(c, x, 1e-6f); },
        [](ggml_context * c, ggml_tensor * x) { return ggml_rms_norm(c, x, 1.0f); });
    // SCALE was already compared before the fix: the control
    run_case(cuda, "SCALE",
        [](ggml_context * c, ggml_tensor * x) { return ggml_scale(c, x, 0.5f); },
        [](ggml_context * c, ggml_tensor * x) { return ggml_scale(c, x, 2.0f); });

    ggml_backend_free(cuda);
    printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "ALL PASS", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
