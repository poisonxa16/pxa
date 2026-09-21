#pragma once

// PXA_MTP_BATCH_SLOTS (2026-09-09): one MTP draft decode per STEP across all
// drafting slots, instead of one serial chain of decodes per slot.
//
// Mainline llama.cpp's MTP driver
// (common/speculative.cpp:1596-1746 at ggml-org/llama.cpp 304665fe7) seeds ONE batch row per
// drafting sequence and advances them all in the same llama_decode, so at -np N a depth-K draft
// costs K decodes, not N*K. Ours runs mtp_speculative_gen_draft() once per slot, each with its own
// 1-row decodes, so the seat pays N*K round trips per cycle (diff-table row D6: ~+8.7 ms/cycle per
// extra slot at K=4). Both V100 seats run -np 2.
//
// The plumbing this needs already exists and is NOT part of this change:
//   * the MTP companion is SHARED across slots with n_seq_max = n_parallel
//     (PXA_SHARED_MTP_v1 / PXA_SHARED_MTP_M1a, examples/server/server-context.cpp), and the
//     kv_seq = (n_seq_max <= 1 ? 0 : seq_id) guards in common/speculative.cpp already route every
//     call to that slot's own K/V row;
//   * the MTP draft-input buffer is a MULTI-ROW buffer -- prepare_mtp_graph_inputs()
//     (src/llama.cpp) slices it per token when its float count is a multiple of the batch token
//     count -- which is what the batched COMMIT (common_speculative_commit_accepted_hidden_rows_
//     batched) already relies on;
//   * per-row read-back by BATCH index (llama_get_logits_ith / llama_get_embeddings_ith go through
//     the output-id map) is what that same batched commit already does for its per-seq tails.
//
// So P5 is a SCHEDULING change: same tokens, same positions, same K/V rows, same hidden rows, same
// argmax -- only regrouped from N chains of K decodes into K decodes of N rows. This header holds
// the pieces that decide whether the regrouping is legal and what each sequence does on each step,
// as pure data with no llama_context, so both schedulers can be replayed against each other on the
// CPU with no model (tests/test-mtp-batch-slots.cpp).
//
// THE ROW-WIDTH CONTRACT (PXA_MTP_BATCH_SLOTS_ROWS_v1). One item of plumbing was NOT already in
// place, contrary to the paragraph above, and it is what made the first two-client run abort. The
// MTP tail-only graph allocated its conditioning-hidden input (inp_mtp_states) as [width, n_tokens]
// for the warm-up, the accepted-token commit and the K/V-only refresh, but as a single [width] row
// for the DRAFT step -- correctly, for as long as every draft decode WAS one row. A batched step is
// N rows, the MTP block concatenates that input against the token embeddings (always [width,
// n_tokens]), and ggml_concat requires every dimension but the concatenated one to match: N against
// 1, aborting while the graph was still being assembled, earlier than any input-size check could
// refuse the decode cleanly. Every MTP builder now sizes that input by the batch token count for
// every op type, which is the same tensor it built before whenever n_tokens is 1, and
// llama_model_supports_mtp_multi_row_draft() reports which architectures do so -- the batched path
// refuses outright on one that does not, rather than finding out from a stack trace.
//
// DEFAULT OFF. The win is ~0 at -np 1 and every decode number in the campaign is -np 1; the arm
// that decides it is two concurrent clients against an -np 2 seat (aggregate t/s, per-stream t/s,
// acc_hist, greedy identity per stream). Determinism risk is the reason it is a lever at all:
// multi-seq work on the companion context is the area PXA_LLAMA_MTP_NP_FIX was written to avoid.
// Once measured it can join the per-family PXA_AUTO table in examples/server/server.cpp next to
// PXA_MTP_ZERO_OUTPUT_COMMIT.

#include "pxa-mtp-kvpos.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------------------------
// The lever.
// ---------------------------------------------------------------------------------------------

static inline bool pxa_mtp_batch_slots_enabled() {
    static const int on = []() {
        const char * env = getenv("PXA_MTP_BATCH_SLOTS");
        return (env && atoi(env) != 0) ? 1 : 0;
    }();
    return on == 1;
}

// ---------------------------------------------------------------------------------------------
// When the regrouping is legal.
//
// n_reqs      how many sequences want to draft this step
// n_seq_max   llama_n_seq_max(ctx_mtp) -- the companion's seq rows
// n_ubatch    llama_n_ubatch(ctx_mtp)  -- the companion's micro-batch ceiling
// max_seq_id  the largest seq_id among the requests
//
// The n_seq_max test is the important one: with n_seq_max <= 1 every call is remapped to local K/V
// row 0 (PXA_LLAMA_MTP_NP_FIX), so two sequences in one batch would write each other's cache. The
// n_ubatch test keeps the whole step in ONE micro-batch, because prepare_mtp_graph_inputs() slices
// the draft-input hidden buffer as a CONTIGUOUS [cur_token, cur_token + n_tokens) window of the
// logical batch; a split would still be sliced correctly, but only as long as the split is
// contiguous, and there is no reason to depend on that when the companion's n_ubatch is already
// sized for the whole step (max_stage_n_max + 1, floor 4 -- PXA_MTP_VRAM_FIX).
static inline bool pxa_mtp_batch_slots_applicable(
        int32_t n_reqs,
        uint32_t n_seq_max,
        uint32_t n_ubatch,
        int32_t max_seq_id) {
    if (n_reqs < 2) {
        return false;   // one drafting slot is the serial path, bit-for-bit
    }
    if (n_seq_max <= 1) {
        return false;   // single-seq companion: every seq_id collapses to K/V row 0
    }
    if (max_seq_id < 0 || (uint32_t) max_seq_id >= n_seq_max) {
        return false;
    }
    if ((uint32_t) n_reqs > n_seq_max) {
        return false;
    }
    if (n_ubatch != 0 && (uint32_t) n_reqs > n_ubatch) {
        return false;   // would split the step across micro-batches
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Whether a STEP may be regrouped, once the slots' RUNTIME state is known.
//
// pxa_mtp_batch_slots_applicable() above answers from shapes alone -- how many slots, how many
// companion rows, how wide a micro-batch -- and every one of those is known before a token is
// drafted. It cannot see the one condition that is neither a shape nor a constant: whether a
// particular slot's companion K/V row has caught up to the position its draft would start from.
// A slot that has just joined a running seat is behind for its first few tokens.
//
// The serial drafter tolerates being behind. It logs it and decodes its single row anyway, and the
// row is legal because a 1-row draft decode is the only shape that path can produce. A batched step
// is the only caller that can put that slot's row NEXT TO a caught-up slot's row, so it is the only
// one for which "behind" is a different question, and it must answer it before the batch is built.
//
// The rule is a STEP rule, not a slot rule: if any slot in the step is behind, the WHOLE step goes
// to the serial drafter. Dropping only the lagging slot would be worse in two ways -- that slot
// would silently lose the draft the serial path would have given it, and the equivalence this
// feature rests on (batched == serial, per slot, for every record the test compares) would stop
// holding for it. Handing the step back costs nothing: the caller answers a refusal by running the
// ordinary serial drafter over the same requests, which is exactly what the lever being off does.
//
// The same is true of a step with fewer than two slots left to advance: one drafting slot IS the
// serial path, so there is nothing to regroup and refusing is free.
//
// A refusal must therefore be reachable without having written anything -- which is why every field
// here is READ from the drafter's state rather than taken from it.
enum pxa_mtp_batch_step_decision {
    PXA_MTP_BATCH_STEP_BATCH,   // regroup: two or more slots can be advanced in one decode
    PXA_MTP_BATCH_STEP_SERIAL,  // hand the whole step back to the serial drafter
};

struct pxa_mtp_batch_slot_state {
    int32_t seq_id             = -1;
    int32_t n_past             = 0;     // the position this slot's draft starts from (draft_base_pos)
    int32_t companion_pos_max  = -1;    // highest position present in this slot's companion K/V row
    bool    have_target_hidden = false; // the target stored a hidden row for this slot to draft from
    int32_t n_draft            = 0;     // resolved depth, after any adaptive-K narrowing
};

// The row at n_past is committed history, so a slot is caught up as soon as its companion holds
// n_past - 1. This is the same comparison the serial drafter makes before it logs.
static inline bool pxa_mtp_batch_slot_is_warm(const pxa_mtp_batch_slot_state & s) {
    return s.companion_pos_max >= s.n_past - 1;
}

// A slot with no stored target hidden row, or a resolved depth of zero, produces an empty draft on
// either scheduler; it is not a reason to refuse the step, it just is not one of the slots being
// regrouped.
static inline bool pxa_mtp_batch_slot_can_draft(const pxa_mtp_batch_slot_state & s) {
    return s.have_target_hidden && s.n_draft > 0;
}

static inline pxa_mtp_batch_step_decision pxa_mtp_batch_slots_step_decision(
        const pxa_mtp_batch_slot_state * slots,
        size_t                           n_slots,
        int32_t *                        n_batchable_out = nullptr,
        int32_t *                        cold_slot_out   = nullptr) {
    if (n_batchable_out) { *n_batchable_out = 0; }
    if (cold_slot_out)   { *cold_slot_out   = -1; }

    int32_t n_batchable = 0;
    for (size_t i = 0; i < n_slots; ++i) {
        if (!pxa_mtp_batch_slot_is_warm(slots[i])) {
            if (cold_slot_out) { *cold_slot_out = (int32_t) i; }
            return PXA_MTP_BATCH_STEP_SERIAL;
        }
        if (pxa_mtp_batch_slot_can_draft(slots[i])) {
            ++n_batchable;
        }
    }

    if (n_batchable_out) { *n_batchable_out = n_batchable; }

    return n_batchable >= 2 ? PXA_MTP_BATCH_STEP_BATCH : PXA_MTP_BATCH_STEP_SERIAL;
}

// ---------------------------------------------------------------------------------------------
// One sequence's draft chain.
//
// This is the per-sequence half of mtp_speculative_gen_draft(): what the sequence feeds into the
// next decode, whether it still wants one, and what it does with the sampled token. The scheduler
// -- serial (one chain at a time, 1-row batches) or batched (all chains, one row each, one batch
// per step) -- is the OTHER half, and it is the only thing P5 changes.
//
// The stop codes are PXA_MTP_STATS' codes, unchanged:
//   0 the cached free token collapsed the chain (last.prob < p_min)
//   1 an in-loop p_min break
//   2 the full n_max chain ran
//   3 a decode / embedding failure
enum pxa_mtp_draft_step_action {
    PXA_MTP_DRAFT_STEP_STOP,   // the token was dropped and the chain is finished
    PXA_MTP_DRAFT_STEP_KEEP,   // the token was pushed; read this row's hidden state next
};

struct pxa_mtp_draft_chain {
    int32_t seq_id  = -1;

    int32_t n_draft = 0;    // K for this sequence, AFTER any adaptive-K narrowing
    float   p_min   = 0.0f;
    bool    have_prob = false;  // the caller asks the sampler for a probability at all
    bool    constant_positions = false; // gemma4 external MTP: every draft row sits at n_past

    int32_t n_past  = 0;    // position of the token the draft starts from
    int32_t i       = 0;    // loop index; starts at 1 when a cached free token was consumed
    int32_t cur_id  = -1;   // token fed into the next decode
    int32_t cur_pos = 0;    // position fed into the next decode

    int32_t n_decode = 0;   // decodes this chain has asked for
    int32_t stop     = 2;
    bool    active   = false;
    int32_t pending_id = -1;    // token sampled this step, kept until its hidden row is read

    std::vector<int32_t> drafts;

    // Mirrors the head of mtp_speculative_gen_draft(): the cached cross-step token (produced by the
    // previous commit decode) is draft token 1 for ZERO decodes, and its own probability is what
    // decides whether the chain is allowed to grow past it.
    //
    // Returns false when this chain has nothing to do at all (n_draft <= 0) -- the caller then
    // invalidates the cached draft for the sequence and returns an empty result, as the serial path
    // does.
    bool begin(int32_t seq_id_,
               int32_t n_draft_,
               float   p_min_,
               bool    have_prob_,
               int32_t n_past_,
               int32_t id_last,
               bool    has_cached,
               int32_t cached_id,
               float   cached_prob,
               bool    constant_positions_ = false) {
        seq_id  = seq_id_;
        n_draft = n_draft_;
        p_min   = p_min_;
        have_prob = have_prob_;
        constant_positions = constant_positions_;
        n_past  = n_past_;
        n_decode = 0;
        stop    = 2;
        drafts.clear();

        if (n_draft <= 0) {
            active = false;
            return false;
        }

        cur_id  = id_last;
        cur_pos = n_past;
        i       = 0;

        if (has_cached) {
            if (cached_prob < p_min) {
                n_draft = 1;
                stop    = 0;    // the chain collapsed to the cached token
            }
            cur_id = cached_id;
            drafts.push_back(cur_id);
            cur_pos = n_past + 1;
            i       = 1;
        }

        active = true;
        return true;
    }

    // True when this chain wants a decode on this step.
    bool wants_step() const { return active && i < n_draft; }

    // The position of the row this chain contributes to the next batch.
    int32_t step_pos() const { return constant_positions ? n_past : cur_pos; }

    // The row at n_past is committed history -- (h_{n_past-1}, x_{n_past}) under PXA_MTP_KVPOS_v1.
    // A chain that consumed a cached free token must leave it alone; a chain starting at i == 0 is
    // about to rewrite it and has to drop any stale cell there first. (constant_positions never has
    // a cached token and is deliberately left alone -- see mtp_speculative_gen_draft.)
    bool needs_pre_purge() const { return active && i == 0 && !constant_positions; }

    // A decode has been issued for this chain's row. Counted BEFORE the decode, exactly as the
    // serial loop counts it, because a failed decode still widens the draft-region purge.
    void on_step_issued() { ++n_decode; }

    // What the chain does with the token sampled from its own row of the step's decode. This is the
    // EXACT ordering of mtp_speculative_gen_draft()'s loop tail, split at the point where the loop
    // reads the row's hidden state, and the ordering is the reason it matters:
    //
    //   * at i > 0 a below-floor token is DROPPED and the chain stops (the pre-push compare);
    //   * otherwise the token is pushed and the caller must read this row's hidden state; a failed
    //     read is on_fail() with the token already kept, as in the serial loop;
    //   * on_hidden() then advances the input/position and applies the post-push floor, which is
    //     only reachable at i == 0 because i > 0 already broke above -- so a below-floor FIRST
    //     token is KEPT and stops the chain.
    //
    // NOTE for the merge of P6 ( "make p_min compare a top-k(10)-renormalised
    // probability"): P6 changes WHAT `prob` is, not this ordering. If it ever changes the ordering,
    // it has to change it here AND in mtp_speculative_gen_draft's loop, which is the authority.
    pxa_mtp_draft_step_action on_sample(int32_t id_next, float prob) {
        if (i > 0 && have_prob && prob < p_min) {
            stop   = 1;
            active = false;
            return PXA_MTP_DRAFT_STEP_STOP;
        }

        drafts.push_back(id_next);
        pending_id = id_next;
        return PXA_MTP_DRAFT_STEP_KEEP;
    }

    // The hidden row for the token just pushed has been read successfully.
    void on_hidden(float prob) {
        cur_id  = pending_id;
        cur_pos = cur_pos + 1;
        ++i;

        if (have_prob && prob < p_min) {
            stop   = 1;
            active = false;
            return;
        }

        if (i >= n_draft) {
            active = false;     // stop stays 2: the full chain ran
        }
    }

    // A decode or an embedding read failed for this sequence.
    void on_fail() {
        stop   = 3;
        active = false;
    }

    // The draft-region purge bounds, unchanged from the serial path:
    //   [pxa_mtp_draft_region_pos0(n_past), n_past + n_decode + 2)
    // Only run when this chain actually decoded something.
    bool needs_purge() const { return n_decode > 0; }
    int32_t purge_p0() const { return pxa_mtp_draft_region_pos0(n_past); }
    int32_t purge_p1() const { return n_past + n_decode + 2; }
};

// PXA_MTP_ADAPTIVE_K, as the batched scheduler resolves it. Byte-for-byte the ladder in
// mtp_speculative_gen_draft(); kept here so the batched path cannot drift from it silently and so
// the CPU test can cover both schedulers with one ladder.
static inline int32_t pxa_mtp_adaptive_k_depth(float ema, int32_t n_draft) {
    if (n_draft <= 1) {
        return n_draft;
    }
    if (ema >= 0.55f) {
        return n_draft;                             // strong acceptance -> full depth
    }
    if (ema >= 0.40f) {
        return n_draft < 2 ? n_draft : 2;           // middling -> shallow chain
    }
    return 1;                                       // poor -> single (cached/free) token
}
