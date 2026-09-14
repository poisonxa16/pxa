#pragma once

#include "llama.h"

#include <vector>

struct llama_context;

enum llama_spec_feature_kind {
    LLAMA_SPEC_FEATURE_NONE,
    LLAMA_SPEC_FEATURE_HIDDEN_STATE,
};

struct llama_spec_feature_row_view {
    llama_seq_id seq_id = 0;
    llama_pos pos = -1;
    const float * data = nullptr;
};

struct llama_spec_feature_view {
    llama_spec_feature_kind kind = LLAMA_SPEC_FEATURE_NONE;
    int32_t width = 0;
    std::vector<llama_spec_feature_row_view> rows;
};

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx);

bool llama_set_draft_input_hidden_state_copy(
        struct llama_context * ctx,
        const float * hidden_state,
        size_t n_floats);

bool llama_spec_get_hidden_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view);

bool llama_spec_get_hidden_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view);

// PXA_MTP_HIDDEN_BY_BATCH_ROW_v1 (2026-09-09): address the target's MTP hidden rows by RAW
// BATCH ROW, and say so.
//
// The storage is already dense by raw batch row on the target whenever MTP is on: llama_decode
// reserves n_outputs_embd = n_tokens_all for an MTP context at MTP_OP_NONE and writes one row per
// batch token, in batch order, rather than one row per OUTPUT (src/llama.cpp, the "reserve output
// buffer" block and the LLAMA_POOLING_TYPE_NONE extraction). That is the same shape mainline
// llama.cpp stores h_nextn in.
//
// What was missing is the CONTRACT. These readers were named and documented as taking output
// indices while every caller passes a batch row -- slot.i_batch_dft entries, slot.i_batch - i, the
// {0..n-1} of a re-decode. The two coincide only while every row of the batch asks for logits AND
// the buffer was not clamped, and where they diverge the old code did not fail: it returned a row
// belonging to a different token. The clamp is reachable -- PXA_MTP_LAZY_WARMUP_v1 sets
// n_outputs_embd = n_outputs for batches over 64 tokens, which is any step where one slot prefills
// while another verifies.
//
// So: a non-negative index is a BATCH ROW and is only answered while the buffer is batch-dense; a
// negative index keeps its old meaning, counting back from the end of the embedding buffer (that
// is a genuine output-space sentinel -- common_speculative_ensure_sequence_hidden's "the last row
// produced"). Not addressable now fails, instead of quietly answering with the wrong token's
// hidden state.
//
// The rule is a pure function so it can be enumerated on the CPU (tests/test-mtp-hidden-rows.cpp).
//   batch_row           the caller's index: >= 0 a raw batch row, < 0 counts back from the end
//   n_rows_batch_dense  ctx->n_embd_rows_batch_dense: how many leading rows of ctx->embd are
//                       addressable by raw batch row (0 = the buffer is output-indexed)
//   n_outputs_embd      ctx->n_outputs_embd: how many rows the buffer actually holds
// Returns the row index into ctx->embd, or -1 when the request cannot be answered.
static inline int32_t llama_spec_hidden_row_for_batch_row(
        int32_t batch_row,
        int32_t n_rows_batch_dense,
        int32_t n_outputs_embd) {
    if (n_outputs_embd <= 0) {
        return -1;
    }
    if (batch_row < 0) {
        const int32_t j = n_outputs_embd + batch_row;
        return (j >= 0) ? j : -1;
    }
    if (n_rows_batch_dense <= 0 || batch_row >= n_rows_batch_dense) {
        return -1;
    }
    return (batch_row < n_outputs_embd) ? batch_row : -1;
}

// True when ctx->embd's leading rows are addressable by raw batch row for the last decode.
bool llama_spec_hidden_rows_are_batch_dense(const struct llama_context * ctx);

bool llama_spec_get_hidden_feature_view_from_batch_row(
        struct llama_context   * ctx,
        int32_t                  batch_row,
        llama_seq_id             seq_id,
        llama_pos                pos,
        llama_spec_feature_view & view);

bool llama_spec_copy_hidden_rows_from_batch_rows(
        struct llama_context * ctx,
        const std::vector<int32_t> & batch_rows,
        std::vector<float> & hidden_rows);