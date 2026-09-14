#pragma once

// PXA_MTP_KVPOS_v1 (2026-09-09): where an MTP self-speculation head's K/V rows live.
//
// THE CONVENTION. The MTP head is one grafted layer whose input row is a pair: the TARGET model's
// hidden state produced after some token, and the embedding of the token that FOLLOWS it. The row
// for the pair
//
//     (h_{q-1}, x_q)          h_{q-1} = target hidden after the token at position q-1
//                             x_q     = the token at position q
//
// occupies MTP K/V position q -- the token's own position -- and its output predicts x_{q+1}.
// That is upstream llama.cpp's convention ("shift the hidden states right by one, leave the tokens
// at their true positions") and it is what three independent places in this tree already do:
//
//   1. the prompt warm-up (common_speculative_on_target_batch, the upstream f5e5753c #1987 port)
//      decodes the prompt batch AT ITS TRUE POSITIONS while shifting the hidden rows right by one,
//      so position i gets (h_{i-1}, x_i). Its comment records that the UNSHIFTED version had "a
//      one-token conditioning skew";
//   2. the drafter's seed (common_speculative_state_mtp::draft) pairs the stored target hidden --
//      captured by the server at draft_base_pos - 1, i.e. h_{n_past-1} -- with id_last = x_{n_past},
//      and mtp_speculative_gen_draft decodes that pair at position n_past;
//   3. the draft-region purge in mtp_speculative_gen_draft deliberately KEEPS the row at n_past
//      ("we want to keep this token in the KV cache") -- that row is committed history, not a draft.
//
// THE BUG THIS FIXES. The commit path (common_speculative_apply_hidden_rows and its batched twin)
// placed row i of an accepted-token batch at pos_base + i, where pos_base is the position of
// `sampled_before` -- the token that OPENED the verify batch. But row i carries ids[i], whose own
// sequence position is pos_base + 1 + i (ids[0] is the token the target produced AFTER
// sampled_before). Every committed row therefore landed one position low: the head's cache held
// (h_q, x_{q+1}) at q instead of (h_{q-1}, x_q), the row for the newest token was overwritten and
// the newest position left empty. Depth-1 drafts survived that (they are conditioned directly on
// the true hidden row handed in as the draft input), but every deeper draft attends the shifted
// cache through RoPE and was rejected: measured 9 accepted of 388 depth->=2 verifies, 2.3 %
// (measured 2026-09-08).
//
// The arithmetic lives here, as pure functions over positions, so that both writers and the purge
// bounds that interact with them can be tested on the CPU with no model
// (tests/test-mtp-kvpos.cpp).

#include <cstdint>

// Position of row i of a commit batch.
//   pos_base = position of `sampled_before` (the token that opened the verify batch)
//   ids[i]   = the i-th token the target produced, whose own position is pos_base + 1 + i
static inline int32_t pxa_mtp_commit_pos(int32_t pos_base, int32_t i) {
    return pos_base + 1 + (int32_t) i;
}

// First position a commit writes. Also the first cell its pre-decode cleanup may drop: the row at
// pos_base itself is committed history (it holds (h_{pos_base-1}, x_{pos_base})) and must survive,
// because the very next draft attends to it.
static inline int32_t pxa_mtp_commit_pos0(int32_t pos_base) {
    return pxa_mtp_commit_pos(pos_base, 0);
}

// First position the DRAFT region may occupy, given the drafter's n_past (the position of the token
// it is drafting from). n_past itself is the committed row, never a draft row:
//   * with a cached free token (i0 == 1) the commit wrote it, and the purge must leave it alone;
//   * without one (i0 == 0) this draft call writes it, and the purge must leave it behind.
// The old bound `n_past + 1 - i0` only looked right because the commit wrote one position low, so
// nothing was ever stored at n_past in the cached-token case.
static inline int32_t pxa_mtp_draft_region_pos0(int32_t n_past) {
    return n_past + 1;
}

// Position of the row the draft loop writes on iteration i (i counted from i0, the loop's first
// index): with a cached free token the loop starts at n_past + 1, without one at n_past.
static inline int32_t pxa_mtp_draft_pos(int32_t n_past, int32_t i) {
    return n_past + (int32_t) i;
}
