#pragma once

#include "llama.h"
#include "llama-spec-features.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

using common_speculative_feature_kind = llama_spec_feature_kind;
using common_speculative_feature_row_view = llama_spec_feature_row_view;
using common_speculative_feature_view = llama_spec_feature_view;

static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_NONE = LLAMA_SPEC_FEATURE_NONE;
static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE = LLAMA_SPEC_FEATURE_HIDDEN_STATE;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// PXA_SHARED_MTP_v1: per-seq begin for the SHARED MTP companion. Resets ONLY this seq's MTP
// caches (target hidden + draft cache) instead of the whole map, so starting a new generation in
// one slot does not wipe the in-flight state of the other slots sharing the same spec. For non-MTP
// (per-slot, unshared) impls this falls back to the whole-spec begin (unchanged behavior).
void common_speculative_begin_seq(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// PXA_SLOT_ERASE_SPEC_HARD_v1 (2026-09-14): full teardown of every impl's persistent per-slot
// state (the n-gram map and any adaptive-ramp counters riding beside it) -- what a slot ERASE
// needs and begin()/begin_seq() deliberately do not do (they keep a stage's learned table warm
// across a slot's own follow-up turns). Call this from the erase path alongside the existing
// hidden/companion clear.
void common_speculative_hard_reset(common_speculative * spec);

// sample up to n_draft tokens and add them to the batch using the draft model
// draft_base_pos/draft_seq_id override the MTP position for id_last
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

// PXA_MTP_BATCH_SLOTS (2026-09-09; ):
// draft for SEVERAL slots at once on the shared MTP companion -- one llama_decode per draft STEP
// with one batch row per still-running slot, instead of one serial chain of decodes per slot.
// Same tokens, same positions, same K/V rows, same conditioning hidden rows, same argmax; only the
// grouping of the decodes changes. Default OFF (env PXA_MTP_BATCH_SLOTS=1); see
// common/pxa-mtp-batch-slots.h.
//
// Returns false when the batched path does not apply (lever off, fewer than 2 requests, a
// single-seq or composite/gemma4 companion, autotune on, or a request without a draft_base_pos) --
// the caller then runs common_speculative_draft() per slot exactly as before. On true every
// request's `result` has been filled, possibly empty.
struct common_speculative_draft_req {
    llama_seq_id seq_id         = 0;
    llama_pos    draft_base_pos = -1;
    llama_token  id_last        = -1;
    common_params_speculative params;   // this slot's params, with any caller-side depth cap applied
    llama_tokens result;
};

bool common_speculative_draft_batched(
                     common_speculative * spec,
                     std::vector<common_speculative_draft_req> & reqs);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

bool common_speculative_ensure_sequence_hidden(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_capture_output_hidden(
    common_speculative * spec,
    llama_context * ctx,
    int32_t output_index,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_copy_output_hidden_rows(
    const common_speculative * spec,
    llama_context * ctx,
    const std::vector<int32_t> & output_indices,
    std::vector<float> & hidden_rows);

// PXA_MTP_ZERO_OUTPUT_COMMIT (2026-09-09): true when the accepted-token catch-up decode runs as a
// zero-output K/V refresh and the next draft re-decodes the last position as its own step 0, instead
// of the commit carrying a free token out on a full-graph pass. Default off; see the block above
// common_speculative_mtp_zero_output_commit() in speculative.cpp. Exposed so the server can print it.
bool common_speculative_mtp_zero_output_commit();

// Arm (or disarm) the zero-output commit as a DEFAULT for a model family that measured it. An
// explicit PXA_MTP_ZERO_OUTPUT_COMMIT in the environment always wins. Call before the first decode.
void common_speculative_mtp_zero_output_commit_set_default(bool on);

// Commit an accepted verify step into the MTP head's K/V cache.
//
// PXA_MTP_KVPOS_v1 (2026-09-09): `pos_base` is the position of `sampled_before` -- the token that
// OPENED the verify batch -- and `hidden_rows` are that batch's target hidden rows in verify order,
// so row i is h_{pos_base+i} and ids[i] is the token at position pos_base+1+i. The committed MTP row
// is the pair (h_{q-1}, x_q) at position q, i.e. row i lands at pxa_mtp_commit_pos(pos_base, i);
// see common/pxa-mtp-kvpos.h for why, and tests/test-mtp-kvpos.cpp for the invariant it keeps.
bool common_speculative_commit_accepted_hidden_rows(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_output(
    common_speculative * spec,
    llama_context * ctx,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<int32_t> & output_indices);

// PXA_SHARED_MTP_v1: BATCHED all-accepted commit across slots on the shared MTP companion.
// One request per accepting seq; runs a single multi-seq llama_decode(ctx_mtp) instead of N
// serial per-slot commit decodes. Returns the seq_ids whose commit failed (caller must clear
// their hidden state). Covers the no-rejection commit path only (pre-captured hidden rows).
struct common_speculative_commit_req {
    llama_seq_id seq_id;
    llama_pos pos_base;
    llama_token sampled_before;
    std::vector<llama_token> ids;
    const std::vector<float> * hidden_rows;
};

std::vector<llama_seq_id> common_speculative_commit_accepted_hidden_rows_batched(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    const std::vector<common_speculative_commit_req> & reqs);

// PXA_MTP_PREFETCH (Layer 1): async variants that submit the identical commit to a dedicated
// ctx_mtp worker thread and return immediately (env PXA_MTP_PREFETCH=1; off => serial call).
bool common_speculative_commit_accepted_hidden_rows_async(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<float> & hidden_rows);

std::vector<llama_seq_id> common_speculative_commit_accepted_hidden_rows_batched_async(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    const std::vector<common_speculative_commit_req> & reqs);

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id);

llama_context * common_speculative_get_companion_ctx(common_speculative * spec);

// PXA_SPEC_SEQ_STATE_v1 (2026-09-13): the drafter's per-sequence carry, as bytes.
//
// A speculative sequence's state lives in three places: the target's KV, the companion context's
// KV, and a small per-sequence carry that belongs to neither - the target hidden row the next
// draft conditions on, and the last drafted embedding/probability/token. Only the first two travel
// in a llama state blob, so anything that moves or restores a sequence (a slot save/restore, a
// parked prompt coming back) has to carry the third one itself, or the restored sequence drafts
// from whatever the previous occupant of that sequence id left behind.
//
// get returns false when there is nothing to carry (no drafter, not a carry-bearing drafter, or
// this sequence has no state yet), which is not an error: the caller simply writes no companion.
// set returns false on a blob it cannot read - a wrong magic, a truncated tail or an embedding
// width from a different model - and leaves the sequence CLEARED rather than half-applied, so a
// stale or foreign carry can never be adopted by accident.
bool common_speculative_get_seq_state(const common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_set_seq_state(common_speculative * spec, llama_seq_id seq_id, const uint8_t * data, size_t size);

int32_t common_speculative_on_target_seq_batch(
    common_speculative * spec,
    llama_context * ctx,
    const llama_batch & batch,
    llama_seq_id seq_id,
    bool is_prompt_warmup);

int32_t common_speculative_on_target_batch(
    common_speculative * spec,
    const llama_batch & batch,
    const common_speculative_feature_view & features,
    bool is_prompt_warmup);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0, int n_past = 0, common_params_speculative * active_params = nullptr);

common_speculative_type common_speculative_current_type(const common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);
