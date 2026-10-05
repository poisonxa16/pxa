// PXA_MTP_READBACK_v1: the MTP head's own feature row has to be a graph OUTPUT.
//
// Every grafted NextN head in this tree ends the same way:
//
//     cur = <the head block's FFN output>
//     cb(cur, "result_norm", -1);                     <-- THE ROW
//     cur = build_output(..., shared_head_norm, ...); // norm + the head's lm_head
//     cb(cur, "result_output", -1);
//
// and llama_decode() picks that "result_norm" node by name as the embeddings source for any
// MTP op (src/llama.cpp, the backward node scan). It is therefore what
// common/speculative.cpp's mtp_accept_batch() reads back with llama_get_embeddings_ith()
// after the accepted-token commit decode, and what it hands to the next MTP_OP_DRAFT_GEN
// decode as prev_embeddings -- the conditioning hidden the whole draft chain is built on.
//
// ggml-alloc's contract is explicit (ggml/include/ggml-alloc.h):
//
//     ggml_set_output(): output tensors are never freed and never overwritten
//
// Without that flag the row is an ordinary intermediate: the moment the head's own lm_head
// has consumed it, its block goes back on the free list and a later node of the same graph
// takes it. The graph still computes the right logits -- they are produced BEFORE the reuse --
// so the free carried token sampled off that row is perfect while the row itself, copied D2H
// after the whole graph has run, is a different node's data. That asymmetry is exactly what
// was measured on the GPU (2026-09-08): 0.954 accepted at depth 1 from
// the token, 0.045 top-1 at depth 2 for any chain conditioned on the row.
//
// This test needs no model, no weights and no GPU. It builds the head's tail at its real
// shape, allocates it through ggml_gallocr exactly as llama_decode does, and compares the
// row read back out of the arena against the same computation run in a private context where
// every tensor owns its memory.
//
//   case 1  flagged   read-back is bit-identical to the reference        <-- the fix
//   case 2  unflagged the arena reuses the row's block, and the read-back is NOT the row
//                                                                        <-- the teeth
//
// Case 2 failing means the reproduction has gone stale (an allocator change, a shape change),
// not that the engine is fine -- fix the reproduction rather than deleting it, because case 1
// alone cannot tell a working flag from a graph the allocator never wanted to reuse anyway.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void fail(const char * fmt, ...) {
    fprintf(stderr, "  FAIL: ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    ++g_fail;
}

// The head's real proportions, scaled down so the whole thing runs in well under a second:
// a 2-row accepted-token commit batch that asks for ONE output row, a row-select, the block's
// FFN, the shared-head norm and a vocabulary-sized head.
static const int64_t N_EMBD   = 256;
static const int64_t N_FF     = 704;
static const int64_t N_VOCAB  = 4096;
static const int64_t N_TOKENS = 2;    // the commit batch: |accepted prefix| + 1
static const float   NORM_EPS = 1e-6f;

// Deterministic, sign-varied, order-1 values -- no RNG, no fixture file.
static void fill(std::vector<float> & v, uint32_t seed) {
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((float) ((s >> 9) & 0xFFFF) / 32768.0f - 1.0f) * 0.5f;
    }
}

struct inputs {
    std::vector<float>   x;      // [N_EMBD, N_TOKENS]  the head block's input rows
    std::vector<int32_t> ids;    // [1]                 inp_out_ids: the single output row
    std::vector<float>   w_up;   // [N_EMBD, N_FF]
    std::vector<float>   w_down; // [N_FF,   N_EMBD]
    std::vector<float>   g;      // [N_EMBD]            shared_head_norm
    std::vector<float>   w_out;  // [N_EMBD, N_VOCAB]   the head's lm_head

    inputs()
        : x(N_EMBD * N_TOKENS), ids(1), w_up(N_EMBD * N_FF), w_down(N_FF * N_EMBD),
          g(N_EMBD), w_out(N_EMBD * N_VOCAB) {
        fill(x, 1); fill(w_up, 2); fill(w_down, 3); fill(g, 4); fill(w_out, 5);
        ids[0] = (int32_t) (N_TOKENS - 1);   // the commit reads back its LAST row
    }
};

// The tail, built once and used by both runs. `row_out` is the node the engine tags
// "result_norm"; `logits_out` is "result_output".
struct tail {
    ggml_tensor * x      = nullptr;
    ggml_tensor * ids    = nullptr;
    ggml_tensor * w_up   = nullptr;
    ggml_tensor * w_down = nullptr;
    ggml_tensor * g      = nullptr;
    ggml_tensor * w_out  = nullptr;
    ggml_tensor * row    = nullptr;
    ggml_tensor * logits = nullptr;
};

static tail build_tail(ggml_context * ctx, ggml_cgraph * gf, bool flag_row_as_output, bool mark_inputs) {
    tail t;

    t.x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);
    t.ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    t.w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_FF);
    t.w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_FF,   N_EMBD);
    t.g      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EMBD);
    t.w_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_VOCAB);

    if (mark_inputs) {
        ggml_set_input(t.x); ggml_set_input(t.ids); ggml_set_input(t.w_up);
        ggml_set_input(t.w_down); ggml_set_input(t.g); ggml_set_input(t.w_out);
    }
    ggml_set_name(t.x, "x"); ggml_set_name(t.ids, "inp_out_ids");

    // the row-select the multi-row commit does inside the head's attention
    ggml_tensor * cur = ggml_get_rows(ctx, t.x, t.ids);          // [N_EMBD, 1]

    // the head block's FFN
    cur = ggml_mul_mat(ctx, t.w_up,   cur);                      // [N_FF,   1]
    cur = ggml_mul_mat(ctx, t.w_down, cur);                      // [N_EMBD, 1]

    // THE ROW
    t.row = cur;
    ggml_set_name(t.row, "result_norm");
    if (flag_row_as_output) {
        ggml_set_output(t.row);
    }

    // build_output(): shared_head_norm, then the head's own lm_head
    ggml_tensor * n = ggml_rms_norm(ctx, t.row, NORM_EPS);       // [N_EMBD, 1]
    n = ggml_mul(ctx, n, t.g);                                   // [N_EMBD, 1]  <- same size as the row
    t.logits = ggml_mul_mat(ctx, t.w_out, n);                    // [N_VOCAB, 1]
    ggml_set_name(t.logits, "result_output");
    ggml_set_output(t.logits);

    ggml_build_forward_expand(gf, t.logits);
    return t;
}

// The reference: the identical tail in a context where nothing shares memory with anything.
static bool reference_row(const inputs & in, std::vector<float> & row_out, std::vector<float> & logits_out) {
    ggml_init_params ip = { 256ull*1024ull*1024ull, nullptr, /* no_alloc = */ false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fail("reference: ggml_init"); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);
    tail t = build_tail(ctx, gf, /* flag_row_as_output = */ false, /* mark_inputs = */ false);

    memcpy(t.x->data,      in.x.data(),      ggml_nbytes(t.x));
    memcpy(t.ids->data,    in.ids.data(),    ggml_nbytes(t.ids));
    memcpy(t.w_up->data,   in.w_up.data(),   ggml_nbytes(t.w_up));
    memcpy(t.w_down->data, in.w_down.data(), ggml_nbytes(t.w_down));
    memcpy(t.g->data,      in.g.data(),      ggml_nbytes(t.g));
    memcpy(t.w_out->data,  in.w_out.data(),  ggml_nbytes(t.w_out));

    ggml_graph_compute_with_ctx(ctx, gf, 4);

    row_out.assign((const float *) t.row->data,    (const float *) t.row->data    + N_EMBD);
    logits_out.assign((const float *) t.logits->data, (const float *) t.logits->data + N_VOCAB);

    ggml_free(ctx);
    return true;
}

struct arena_result {
    std::vector<float> row;
    std::vector<float> logits;
    bool row_block_reused = false;   // a node scheduled after the row's last consumer overlaps it
    const char * thief   = nullptr;
};

// Run the tail through ggml_gallocr, exactly as llama_decode() allocates a decode graph.
static bool arena_row(const inputs & in, bool flag_row_as_output, arena_result & out) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fail("arena: ggml_backend_cpu_init"); return false; }

    ggml_init_params ip = { ggml_tensor_overhead()*2048 + ggml_graph_overhead_custom(2048, false),
                            nullptr, /* no_alloc = */ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fail("arena: ggml_init"); ggml_backend_free(backend); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);
    tail t = build_tail(ctx, gf, flag_row_as_output, /* mark_inputs = */ true);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        fail("arena: ggml_gallocr_alloc_graph");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(ctx); ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(t.x,      in.x.data(),      0, ggml_nbytes(t.x));
    ggml_backend_tensor_set(t.ids,    in.ids.data(),    0, ggml_nbytes(t.ids));
    ggml_backend_tensor_set(t.w_up,   in.w_up.data(),   0, ggml_nbytes(t.w_up));
    ggml_backend_tensor_set(t.w_down, in.w_down.data(), 0, ggml_nbytes(t.w_down));
    ggml_backend_tensor_set(t.g,      in.g.data(),      0, ggml_nbytes(t.g));
    ggml_backend_tensor_set(t.w_out,  in.w_out.data(),  0, ggml_nbytes(t.w_out));

    // Does the arena hand the row's block to a node that runs after the row is dead? That, and
    // not any particular numeric outcome, is the defect -- read it off the allocation directly.
    {
        int i_row = -1, i_last_consumer = -1;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node == t.row) i_row = i;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s] == t.row) i_last_consumer = i;
            }
        }
        const char * row_lo = (const char *) t.row->data;
        const char * row_hi = row_lo + ggml_nbytes(t.row);
        for (int i = (i_last_consumer >= 0 ? i_last_consumer + 1 : i_row + 1);
             i >= 0 && i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node == t.row || node->data == nullptr) continue;
            const char * lo = (const char *) node->data;
            const char * hi = lo + ggml_nbytes(node);
            if (lo < row_hi && row_lo < hi) {
                out.row_block_reused = true;
                out.thief = ggml_op_name(node->op);
                break;
            }
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fail("arena: ggml_backend_graph_compute");
        ggml_gallocr_free(galloc); ggml_free(ctx); ggml_backend_free(backend);
        return false;
    }

    // Read the row back the way llama_decode() does: after the WHOLE graph has run.
    out.row.resize(N_EMBD);
    out.logits.resize(N_VOCAB);
    ggml_backend_tensor_get(t.row,    out.row.data(),    0, N_EMBD  * sizeof(float));
    ggml_backend_tensor_get(t.logits, out.logits.data(), 0, N_VOCAB * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool identical(const std::vector<float> & a, const std::vector<float> & b, double & max_abs) {
    max_abs = 0.0;
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
        if (d > max_abs) max_abs = d;
    }
    return max_abs == 0.0;
}

int main() {
    printf("test-mtp-head-output: the MTP head's feature row must be a graph output\n");

    const inputs in;

    std::vector<float> ref_row, ref_logits;
    if (!reference_row(in, ref_row, ref_logits)) {
        printf("test-mtp-head-output: FAILED (reference)\n");
        return 1;
    }

    // ---- case 1: flagged -- what the engine ships ------------------------------------------
    {
        printf("  flagged   ggml_set_output(result_norm)\n");
        arena_result r;
        if (!arena_row(in, /* flag_row_as_output = */ true, r)) {
            printf("test-mtp-head-output: FAILED (arena, flagged)\n");
            return 1;
        }
        double d_row = 0.0, d_log = 0.0;
        const bool row_ok = identical(r.row, ref_row, d_row);
        const bool log_ok = identical(r.logits, ref_logits, d_log);
        printf("            read-back row max|d| = %.9f, logits max|d| = %.9f, block reused = %s\n",
               d_row, d_log, r.row_block_reused ? "YES" : "no");
        if (!log_ok) {
            fail("flagged: the head's logits differ from the reference (max|d| = %.9f) -- the test's "
                 "own tail is wrong, not the flag", d_log);
        }
        if (!row_ok) {
            fail("flagged: the read-back feature row is NOT the row the graph computed "
                 "(max|d| = %.9f). ggml_set_output() promises the tensor is never freed and never "
                 "overwritten; something in this tail broke that promise", d_row);
        }
        if (r.row_block_reused) {
            fail("flagged: the arena still handed the row's block to a later %s node -- the output "
                 "flag is not being honoured", r.thief ? r.thief : "?");
        }
    }

    // ---- case 2: unflagged -- the defect, so the test keeps its teeth ------------------------
    {
        printf("  unflagged no output flag on result_norm (the pre-fix engine)\n");
        arena_result r;
        if (!arena_row(in, /* flag_row_as_output = */ false, r)) {
            printf("test-mtp-head-output: FAILED (arena, unflagged)\n");
            return 1;
        }
        double d_row = 0.0, d_log = 0.0;
        const bool row_same = identical(r.row, ref_row, d_row);
        const bool log_ok   = identical(r.logits, ref_logits, d_log);
        printf("            read-back row max|d| = %.9f, logits max|d| = %.9f, block reused = %s (%s)\n",
               d_row, d_log, r.row_block_reused ? "YES" : "no", r.thief ? r.thief : "-");
        if (!log_ok) {
            fail("unflagged: the head's logits differ from the reference (max|d| = %.9f)", d_log);
        }
        // The logits staying right while the row goes wrong IS the signature that made this bug
        // look like a graph/space problem for two windows. Assert both halves of it.
        if (!r.row_block_reused) {
            fail("unflagged: the arena did NOT reuse the row's block, so this test can no longer "
                 "reproduce the defect it exists for. Reshape the tail (the free block has to be "
                 "the right size for a later node) rather than deleting the case");
        }
        if (row_same) {
            fail("unflagged: the read-back row still matched the reference, so nothing here would "
                 "have caught the shipped bug -- see the note above");
        }
    }

    if (g_fail) {
        printf("test-mtp-head-output: FAILED (%d)\n", g_fail);
        return 1;
    }
    printf("test-mtp-head-output: OK\n");
    return 0;
}
