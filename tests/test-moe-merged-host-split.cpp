// v4-fix-mmq-id-host-experts: pins the trigger of the mmq_id.cu "contiguously allocated" abort that
// every Gemma 4 QAT run with host-placed experts (-ot exps=CPU) hit, and the two invariants the fix
// relies on. CPU-only: the device half (the scheduler's partial expert copy + the MMQ padding clear)
// needs a CUDA backend and is proven on a GPU.
//
// A merged ffn_gate_up_exps that sits in a HOST buffer cannot take the fused up/gate node
// (ggml_fused_up_gate_placement_ok), so ggml_moe_up_gate() splits it into two strided views and
// two MUL_MAT_ID nodes. The scheduler copies each view's weight to the device as
// ggml_dup_tensor_layout(view): same ne, same (merged) strides, no view_src. Checks:
//   1. both MUL_MAT_ID weights are views of the merged tensor with the merged expert stride;
//   2. their device copies are NOT contiguously allocated (the old assert's condition) -- and are
//      not views, so the relaxed check (!view_src) holds;
//   3. ggml_nbytes of such a copy is the end of its last row: the padding clear
//      [nbytes, nbytes + pad) never touches a byte of any expert's rows;
//   4. a copy run ending at the last expert must be clamped to ggml_nbytes (the up half's last
//      expert ends nff rows before ne[2]*nb[2]); the clamp is a no-op for a contiguous weight.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// the clamp in ggml_backend_sched's only_active_experts copy (ggml-backend.cpp copy_experts)
static size_t run_copy_size(const ggml_tensor * t, int first, int last) {
    const int n_expert = (int) t->ne[2];
    const size_t es = t->nb[2], off = first*es, sz = (last - first + 1)*es;
    const size_t pad_end = last < n_expert - 1 ? std::min<size_t>(es, 512) : 0;
    return std::min<size_t>(sz + pad_end, ggml_nbytes(t) - off);
}

int main() {
    const int64_t n_embd = 256, nff = 8, n_expert = 4, n_tok = 3, n_used = 2;
    ggml_init_params wp = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * merged = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_0, n_embd, 2*nff, n_expert);
    ggml_set_name(merged, "blk.0.ffn_gate_up_exps.weight");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(ggml_backend_buffer_is_host(merged->buffer), "weight not in a host buffer");

    ggml_init_params gp = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * out = ggml_moe_up_gate(ctx, merged, nullptr, b, ids, GGML_UNARY_OP_GELU);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_init_params cp = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context * cctx = ggml_init(cp);
    int n_mmid = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (n->op == GGML_OP_MOE_FUSED_UP_GATE) { CHECK(false, "host-resident merged weight built a fused node"); }
        if (n->op != GGML_OP_MUL_MAT_ID) continue;
        ++n_mmid;
        ggml_tensor * w = n->src[0];
        CHECK(w->view_src == merged && w->ne[1] == nff && w->nb[2] == merged->nb[2],
              "weight %d is not a strided half of the merged tensor", n_mmid);
        // the scheduler's device copy: ggml_dup_tensor_layout()
        ggml_tensor * c = ggml_dup_tensor(cctx, w);
        for (int k = 0; k < GGML_MAX_DIMS; ++k) c->nb[k] = w->nb[k];
        CHECK(!ggml_is_contiguously_allocated(c), "copy %d is contiguous: the old assert would not fire", n_mmid);
        CHECK(c->view_src == nullptr, "copy %d is a view", n_mmid);
        const size_t last_row_end = (size_t)(n_expert - 1)*c->nb[2] + (size_t)(nff - 1)*c->nb[1] + ggml_row_size(c->type, n_embd);
        CHECK(ggml_nbytes(c) == last_row_end, "copy %d: nbytes %zu != end of last row %zu", n_mmid, ggml_nbytes(c), last_row_end);
        // every copy run stays inside the copy and (offset by the view) inside the merged tensor
        for (int f = 0; f < n_expert; ++f) for (int l = f; l < n_expert; ++l) {
            const size_t off = f*c->nb[2], sz = run_copy_size(c, f, l);
            CHECK(off + sz <= ggml_nbytes(c), "copy %d run [%d,%d] writes past the copy", n_mmid, f, l);
            CHECK(w->view_offs + off + sz <= ggml_nbytes(merged), "copy %d run [%d,%d] reads past the host tensor", n_mmid, f, l);
        }
        const size_t unclamped = (size_t)c->ne[2]*c->nb[2];
        if (w->view_offs > 0) CHECK(unclamped > ggml_nbytes(c), "up half: the unclamped last run should overflow (test premise)");
    }
    CHECK(n_mmid == 2, "expected 2 MUL_MAT_ID nodes, got %d", n_mmid);
    // contiguous weight: the clamp never changes a run
    for (int f = 0; f < n_expert; ++f) for (int l = f; l < n_expert; ++l) {
        const size_t es = merged->nb[2], pad_end = l < n_expert - 1 ? std::min<size_t>(es, 512) : 0;
        CHECK(run_copy_size(merged, f, l) == (l - f + 1)*es + pad_end, "clamp changed a contiguous run [%d,%d]", f, l);
    }
    ggml_free(cctx); ggml_free(ctx); ggml_backend_buffer_free(wbuf); ggml_free(wctx); ggml_backend_free(cpu);
    printf("%s test-moe-merged-host-split (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
