#include "speculative.h"
#include "pxa-mtp-batch-slots.h"
#include "pxa-mtp-cache-only.h"
#include "pxa-mtp-kvpos.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>

// ============================================================================
// PXA_MTP_PREFETCH (Layer 1) — overlap the single MTP companion COMMIT decode
// with the same step's process_token + HTTP streaming write.
//   env PXA_MTP_PREFETCH=1  (default 0 => byte-for-byte the serial path).
// One dedicated worker thread owns every async ctx_mtp write; every other
// ctx_mtp writer (draft / ensure / on_target_* / context_shift / clear) waits
// on the pending job first (cancel-before-mutate). The commit code executed is
// IDENTICAL to the serial function — only its thread + timing move — so the
// token stream and the acceptance trajectory are bit-exact. Because the proxy
// caps brainInflightCap=1 (single seq), at most one job is ever in flight.
// ============================================================================
#include <thread>
#include <future>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <memory>
#include <vector>

static bool pxa_mtp_prefetch_enabled() {
    static const int v = getenv("PXA_MTP_PREFETCH") ? atoi(getenv("PXA_MTP_PREFETCH")) : 0;
    return v != 0;
}

// Set while the worker thread runs a commit job. Guarded ctx_mtp entry points that the job may
// re-enter (e.g. clear_sequence_hidden on a failed commit) must NOT wait on the job's own future
// -> that would deadlock. When true, the wait helpers below are no-ops (the worker already owns
// ctx_mtp exclusively for the duration of the job).
static thread_local bool pxa_mtp_in_worker = false;

// self-contained copy of a commit request (the caller's hidden-row buffers are
// stack/locals that die when update_slots returns, so the job must own them).
struct pxa_mtp_owned_req {
    llama_seq_id             seq_id = -1;
    llama_pos                pos_base = 0;
    llama_token              sampled_before = -1;
    std::vector<llama_token> ids;
    std::vector<float>       hidden;
};

struct pxa_mtp_prefetch_worker {
    std::thread                       th;
    std::mutex                        qmtx;
    std::condition_variable           qcv;
    std::deque<std::function<void()>> q;
    bool                              stop = false;
    bool                              started = false;

    std::mutex                                                 jmtx;
    std::unordered_map<llama_seq_id, std::shared_future<void>> jobs; // pending future per seq

    void ensure_started() {
        std::lock_guard<std::mutex> lk(qmtx);
        if (started) return;
        started = true;
        th = std::thread([this]{
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> ul(qmtx);
                    qcv.wait(ul, [this]{ return stop || !q.empty(); });
                    if (stop && q.empty()) return;
                    job = std::move(q.front());
                    q.pop_front();
                }
                pxa_mtp_in_worker = true;
                job(); // runs the (identical) commit function on ctx_mtp
                pxa_mtp_in_worker = false;
            }
        });
    }

    // Register the future for every seq in `seqs` AND enqueue the task under one lock scope,
    // so any wait_seq()/wait_all() (which take jmtx) either runs entirely before registration
    // or sees the registered+enqueued future — never a gap where the job is live but unfindable.
    void submit_for_seqs(const std::vector<llama_seq_id> & seqs, std::function<void()> fn) {
        ensure_started();
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
        std::shared_future<void> fut = task->get_future().share();
        {
            std::lock_guard<std::mutex> jlk(jmtx);
            for (llama_seq_id s : seqs) jobs[s] = fut;      // register first (visible under jmtx)
            {
                std::lock_guard<std::mutex> qlk(qmtx);      // then enqueue (FIFO single-thread => ordering)
                q.push_back([task]{ (*task)(); });
            }
        }
        qcv.notify_one();
    }

    void wait_seq(llama_seq_id seq_id) {
        std::shared_future<void> fut;
        {
            std::lock_guard<std::mutex> jlk(jmtx);
            auto it = jobs.find(seq_id);
            if (it == jobs.end() || !it->second.valid()) return;
            fut = it->second;
            jobs.erase(it);
        }
        fut.wait();
    }

    void wait_all() {
        std::vector<std::shared_future<void>> futs;
        {
            std::lock_guard<std::mutex> jlk(jmtx);
            for (auto & kv : jobs) if (kv.second.valid()) futs.push_back(kv.second);
            jobs.clear();
        }
        for (auto & f : futs) f.wait();
    }
};

static pxa_mtp_prefetch_worker & pxa_mtp_prefetch() {
    static pxa_mtp_prefetch_worker w;
    return w;
}

static inline void pxa_mtp_prefetch_wait_seq(llama_seq_id seq_id) {
    if (pxa_mtp_in_worker) return;
    if (pxa_mtp_prefetch_enabled()) pxa_mtp_prefetch().wait_seq(seq_id);
}
static inline void pxa_mtp_prefetch_wait_all() {
    if (pxa_mtp_in_worker) return;
    if (pxa_mtp_prefetch_enabled()) pxa_mtp_prefetch().wait_all();
}

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

void llama_set_mtp_target_context(struct llama_context * ctx, struct llama_context * target_ctx);

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_SUFFIX
};

// PXA_SPEC_NGRAM_ALIAS_v1 (2026-09-09) -- one word for the drafter a model without an MTP head
// can actually use.
//
// The self-speculation family below is six separate spellings of "guess the next tokens from
// tokens already in the context": ngram_simple, ngram_map_k, ngram_map_k4v, ngram_mod,
// ngram_cache and suffix. A user who loads a plain dense GGUF -- which has no MTP head, so the
// engine's fastest drafter is not available to them at all -- has no way to know which of the six
// to name, and picking wrong is the difference between a measured win and a measured loss. So
// `--spec-type ngram` (or `prompt-lookup`, the name the technique is published under) resolves to
// the variant this campaign MEASURED best, and carries that variant's measured knobs with it.
//
// Two properties this alias deliberately keeps:
//   * It is an ALIAS, not a type. common_speculative_type_to_str() still prints the CONCRETE
//     variant, so a boot log always says what actually ran and no number is ever filed against a
//     name that does not exist in the enum.
//   * It fills only ABSENT knobs. `--spec-type ngram:n_max=8` gets 8, and every key the user names
//     wins, exactly like the PXA_AUTO layer's contract.
//
// The target and its defaults are one edit, on purpose: when a measurement overturns the ranking,
// this table moves and nothing else does.
static constexpr enum common_speculative_type COMMON_SPECULATIVE_NGRAM_ALIAS_TARGET =
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD;

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    // aliases -- resolved to a concrete variant above, never printed back
    {"ngram",         COMMON_SPECULATIVE_NGRAM_ALIAS_TARGET},
    {"prompt_lookup", COMMON_SPECULATIVE_NGRAM_ALIAS_TARGET}
};

// the set of spellings that are aliases rather than types -- the stage parser asks, so that it
// knows whether it may fill the measured defaults below.
static const std::set<std::string> common_speculative_type_alias_names = {
    "ngram", "prompt_lookup"
};

bool common_speculative_type_name_is_alias(const std::string & name) {
    std::string normalized = name;
    std::replace(normalized.begin(), normalized.end(), '-', '_');
    return common_speculative_type_alias_names.count(normalized) > 0;
}

// The measured knobs the alias carries, per concrete target. Sources are named so a future edit
// has to argue with a number rather than with a preference.
void common_speculative_apply_alias_defaults(common_speculative_stage_params & stage) {
    switch (stage.type) {
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
            // n_max=4, n_min=2 is the pair the PXA_AUTO layer already ships for qwen35moe, where it
            // measured +23.0% first-request decode on code traffic (24.44 -> 30.05 t/s) and +4.6%
            // on prose, prefill-neutral. n_min=2 is the load-bearing half: it makes a one-token
            // match fall through instead of paying for a verify batch.
            if (!stage.has_n_max_override())        { stage.n_max        = 4; }
            if (!stage.has_n_min_override())        { stage.n_min        = 2; }
            // ngram_mod hashes WITHOUT collision detection, so the hash width is a quality knob:
            // the implementation warns below 16 and the global default is 12. The alias is the
            // beginner's spelling, so it takes the width the implementation asks for.
            if (!stage.has_ngram_size_n_override()) { stage.ngram_size_n = 16; }
            break;
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            // the map keeps up to four continuations per key and filters by hit count, so it is
            // exact where ngram_mod is lossy; its size knobs keep their stock values (12/48) and
            // only the draft shape is filled.
            if (!stage.has_n_max_override())        { stage.n_max        = 4; }
            if (!stage.has_n_min_override())        { stage.n_min        = 2; }
            break;
        default:
            if (!stage.has_n_max_override())        { stage.n_max        = 4; }
            if (!stage.has_n_min_override())        { stage.n_min        = 2; }
            break;
    }
}

struct common_speculative_config {
    common_speculative_stage_params stage;
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(
            const common_speculative_stage_params & s,
            const common_params_speculative & p = common_params_speculative{})
        : stage(s), type(s.type), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // PXA_SPEC_STAGE_DIAG_v1 (2026-09-15): the shipped counters cannot tell
    // "this stage found nothing" apart from "this stage drafted and the chain floor threw it
    // away", because n_gen_drafts is only incremented once a draft has SURVIVED the floor. That
    // ambiguity is exactly what stalled the n-gram suppression diagnosis: 898 draft calls,
    // 52 wins, and no way to read which failure it was. These three add the missing resolution and touch nothing that already exists --
    // n_gen_drafts / n_acc_drafts / n_gen_tokens / n_acc_tokens keep their exact meanings.
    size_t n_draft_empty      = 0; // draft() returned nothing at all
    size_t n_draft_below_floor= 0; // draft() returned tokens, the chain's n_min discarded them
    size_t n_draft_ctx_zero   = 0; // draft() was called with an EMPTY prompt_tgt
    size_t n_observe          = 0; // told about an acceptance this stage did not win

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_pos draft_base_pos,
            llama_seq_id draft_seq_id,
            llama_tokens & result) {
        GGML_UNUSED(draft_base_pos);
        GGML_UNUSED(draft_seq_id);
        draft(params, prompt_tgt, id_last, result);
    }

    virtual void accept(uint16_t n_accepted) = 0;

    // PXA_SPEC_CHAIN_ACCEPT_v1 (2026-09-15): a chain step is won by ONE stage, and
    // until now only that stage was told anything. Every OTHER stage in the chain is left with
    // stale bookkeeping about a step it took part in (it was asked to draft) -- see the suffix
    // stage, which is the only one that notices, and only via the `!had_accept` branch in its own
    // draft(). This is the explicit version of that: every stage that was ASKED and did not win
    // hears about the step. It carries no statistics (n_acc_drafts / n_acc_tokens / n_call_accept
    // remain the winner's alone, so the diagnosis vocabulary is unchanged); it exists so a stage
    // that keeps learned state can keep it correct. Default no-op.
    virtual void observe_step(uint16_t n_accepted, bool drafted, llama_seq_id seq_id) {
        GGML_UNUSED(n_accepted);
        GGML_UNUSED(drafted);
        GGML_UNUSED(seq_id);
    }

    // Per-stage diagnostics appended to the statistics line. Default: nothing.
    virtual std::string extra_stats() const { return std::string(); }

    // PXA_SLOT_ERASE_SPEC_HARD_v1 (2026-09-14): unconditionally drop any persistent table this
    // stage keeps BEYOND what begin()/begin_seq() already clear per generation. begin() resets
    // the per-generation bookkeeping (i_last, n_draft_last, the low-acceptance streak) but a
    // stage's learned map (e.g. common_ngram_mod's hashed entries) is deliberately left across
    // generations to stay warm within one slot's lifetime -- until a slot ERASE says that slot's
    // lifetime, and everything it learned, is over. Default no-op: most stages have no such table.
    virtual void hard_reset() {}
};

struct common_speculative_state_mtp;

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec);
static const common_speculative_state_mtp * common_speculative_get_mtp_state(const common_speculative * spec);
static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state);
static void mtp_clear_target_hidden(common_speculative_state_mtp & state, llama_seq_id seq_id); // PXA_SHARED_MTP_v1

static std::vector<llama_token> mtp_speculative_gen_draft(
    common_speculative_state_mtp & state,
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    llama_token id_last,
    llama_pos n_past,
    llama_seq_id seq_id,
    bool constant_draft_positions = false);

static int32_t mtp_update_kv_cache(struct llama_context * ctx, const llama_batch & batch, bool is_prompt_warmup);

// PXA_MTP_ADAPTIVE_K: opt-in (env PXA_MTP_ADAPTIVE_K=1). Default OFF -> the MTP draft loop below is
// bit-identical to baseline. When ON, the per-cycle draft depth K is chosen from a running
// acceptance EMA (raise K when drafts land, shrink toward 1 when they get rejected) so a
// near-certain-reject tail draft does not cost an MTP-head forward + a wider verify pass.
// PXA_MTP_ZERO_OUTPUT_COMMIT (2026-09-09; ):
// split the commit decode's two jobs the way mainline llama.cpp does.
//
// Our accepted-token commit does two things in one decode: it advances the MTP head's K/V over the
// verified positions, AND it produces one logit so the next cycle's first draft token can be
// sampled for free. Because of the second job the batch has to ask for an output, so the whole head
// runs for every row -- the FFN and a 248320-row LM-head GEMV over rows that nobody reads.
//
// Mainline splits them: its catch-up decode sets logits = 0 on EVERY row (n_outputs == 0, so the
// row-select gathers nothing and the FFN and LM head compute nothing), and its draft loop then
// re-decodes the last position as its own step 0 to get the logit. The trade is one extra 1-row MTP
// decode (~2.9 ms measured) against a full-graph pass over 1+A rows collapsed to a K/V-only one.
//
// With the lever ON: the commit batch asks for no outputs, the last-row embedding read-back is
// skipped, and the cached free token stays absent -- so mtp_speculative_gen_draft() takes its
// i0 == 0 path and re-decodes the row at n_past itself. That row is committed history under
// PXA_MTP_KVPOS_v1, and the i0 == 0 pre-loop seq_rm added there is what makes rewriting it safe.
// The K/V arithmetic is identical either way (same hidden row, same token, same position), so the
// cache the drafter attends to is unchanged; only which decode produces the first logit moves.
//
// DEFAULT OFF: the window measures it as an arm. It is also the caller-side half of the split --
// with zero outputs our graph already collapses the FFN and the LM head (build_qwen35_mtp builds
// inp_out_ids whenever n_tokens > 1 && n_outputs < n_tokens), but the query projection and the
// attention still run. The MTP_OP_KV_ONLY / store_only graph (introduced 2026-09-08) is what
// removes those, and it composes with this as a one-line change to that graph's own
// wants_output condition.
// -1 = not resolved yet, 0 = off, 1 = on. Resolved from the env on first read, or set earlier by
// the PXA_AUTO layer for a family that measured it (common_speculative_mtp_zero_output_commit_set_default).
static int pxa_mtp_zero_output_commit_state = -1;

bool common_speculative_mtp_zero_output_commit() {
    if (pxa_mtp_zero_output_commit_state < 0) {
        const char * env = getenv("PXA_MTP_ZERO_OUTPUT_COMMIT");
        pxa_mtp_zero_output_commit_state = (env && atoi(env) != 0) ? 1 : 0;
    }
    return pxa_mtp_zero_output_commit_state == 1;
}

void common_speculative_mtp_zero_output_commit_set_default(bool on) {
    // An explicit PXA_MTP_ZERO_OUTPUT_COMMIT always wins over an auto-armed default; the operator
    // naming the lever has decided. Called once, before any decode, from the server's PXA_AUTO block.
    if (getenv("PXA_MTP_ZERO_OUTPUT_COMMIT")) {
        return;
    }
    pxa_mtp_zero_output_commit_state = on ? 1 : 0;
}

// PXA_MTP_PMIN_TOPK_v1: WHICH probability the MTP confidence floor compares against.
//
// The floor was inherited from draft-MODEL speculation, where a proposal costs a whole second
// forward pass. It was applied here to a FULL-VOCABULARY softmax of the head's argmax, and on a
// 248320-token vocabulary that number has mean 0.036 and clears 0.75 in 0.0 % of draft steps
// (measured 2026-09-08) -- so the stock 0.75 floor was not a confidence
// filter, it was an off switch, and it is why n_max was unreachable before PXA_MTP_KVPOS_v1's
// sibling fix. Compare the same quantity a top_k = 10 sampler chain would report for its first
// candidate instead, which is what the upstream llama.cpp drafter compares
// (common/speculative.cpp: it samples through a top_k = 10 chain and tests cur_p->data[0].p).
//
// Default 10. At the family's own auto default (p_min = 0) nothing computes a probability at all,
// so this is INERT on the shipping path -- it changes only the runs that arm a floor
// (PXA_MTP_PMIN=X, --spec-type mtp:...,p_min=X, a per-request speculative.p_min, or any family
// that still inherits the 0.75 global) and the PXA_MTP_STATS histograms, which deliberately
// report the same quantity the floor compares so the instrument measures what the knob does.
// It is also cheaper: 10 exponentials per draft step instead of 248320.
//   PXA_MTP_PMIN_TOPK=0   the historical full-vocabulary softmax
//   PXA_MTP_PMIN_TOPK=k   renormalise over the k largest logits (k clamped to 64)
//   PXA_REFERENCE=1       resolves to 0, like every other level-gated lever
static int pxa_mtp_pmin_topk() {
    static const int k = [](){
        const char * e = getenv("PXA_MTP_PMIN_TOPK");
        if (e) {
            const int v = atoi(e);
            return v < 0 ? 0 : (v > COMMON_SAMPLER_PMIN_TOPK_MAX ? COMMON_SAMPLER_PMIN_TOPK_MAX : v);
        }
        return ggml_pxa_config_level() >= 1 ? 10 : 0;
    }();
    return k;
}

static bool pxa_mtp_adaptive_k_enabled() {
    static const int v = getenv("PXA_MTP_ADAPTIVE_K") ? atoi(getenv("PXA_MTP_ADAPTIVE_K")) : 0;
    return v != 0;
}

// PXA_MTP_STATS (2026-09-08): per-cycle accounting of the MTP self-speculation
// loop, so the reason a draft chain is SHORT is measurable instead of inferred. Off by default
// (one getenv-cached bool); PXA_MTP_STATS=1 turns it on, PXA_MTP_STATS_EVERY (default 200) sets
// the dump period, and a final dump is emitted from common_speculative_print_stats().
// Per cycle we record: how many tokens were proposed, why the chain stopped (the cached token's
// probability gate, the in-loop p_min gate, reaching n_max, or a decode/embedding failure), the
// top-1 probability of every drafted token, the wall time of the draft call, and -- fed from
// accept() -- how many of those tokens the target then accepted.
struct pxa_mtp_stats_t {
    bool     on    = false;
    int      every = 200;
    uint64_t cycles        = 0;
    uint64_t drafted       = 0;
    uint64_t accepted      = 0;
    uint64_t accept_calls  = 0;
    uint64_t mtp_decodes   = 0;
    uint64_t draft_us      = 0;
    double   p_sum         = 0.0;
    uint64_t p_n           = 0;
    uint64_t p_ge_075      = 0;   // drafted steps whose top-1 prob would clear the stock p_min
    uint64_t len_hist[9]   = {0}; // proposed chain length, 0..8+
    uint64_t acc_hist[9]   = {0}; // accepted tokens per verify, 0..8+
    uint64_t trunc_cached  = 0;   // chain collapsed to 1 by the cached token's prob < p_min
    uint64_t trunc_pmin    = 0;   // in-loop p_min break
    uint64_t trunc_nmax    = 0;   // ran the full n_max chain
    uint64_t trunc_fail    = 0;   // llama_decode / embeddings failure
    uint64_t since_dump    = 0;
    pxa_mtp_stats_t() {
        on    = getenv("PXA_MTP_STATS") && atoi(getenv("PXA_MTP_STATS")) != 0;
        every = getenv("PXA_MTP_STATS_EVERY") ? std::max(1, atoi(getenv("PXA_MTP_STATS_EVERY"))) : 200;
    }
};

static pxa_mtp_stats_t & pxa_mtp_stats() {
    static pxa_mtp_stats_t s;
    return s;
}

static void pxa_mtp_stats_dump(const char * why) {
    auto & st = pxa_mtp_stats();
    if (!st.on || st.cycles == 0) {
        return;
    }
    std::string lh, ah;
    for (int i = 0; i < 9; ++i) {
        lh += (i ? "," : "") + std::to_string((unsigned long long) st.len_hist[i]);
        ah += (i ? "," : "") + std::to_string((unsigned long long) st.acc_hist[i]);
    }
    const double d_mean = (double) st.drafted  / (double) st.cycles;
    const double a_mean = st.accept_calls ? (double) st.accepted / (double) st.accept_calls : 0.0;
    const double a_rate = st.drafted ? (double) st.accepted / (double) st.drafted : 0.0;
    const double p_mean = st.p_n ? st.p_sum / (double) st.p_n : -1.0;
    const double p_hi   = st.p_n ? 100.0 * (double) st.p_ge_075 / (double) st.p_n : -1.0;
    LOG_WRN("PXA_MTP_STATS[%s]: cycles=%llu proposed=%llu (mean %.3f/cycle) accepted=%llu "
            "(mean %.3f/verify, accept_rate %.3f) mtp_decodes=%llu draft=%.3f ms/cycle | "
            "len_hist[0..8+]=%s acc_hist[0..8+]=%s | stop: cached_pmin=%llu pmin=%llu nmax=%llu fail=%llu | "
            "top1_p mean=%.3f, %%>=0.75 = %.1f\n",
            why,
            (unsigned long long) st.cycles, (unsigned long long) st.drafted, d_mean,
            (unsigned long long) st.accepted, a_mean, a_rate,
            (unsigned long long) st.mtp_decodes,
            st.cycles ? (double) st.draft_us / (double) st.cycles / 1000.0 : 0.0,
            lh.c_str(), ah.c_str(),
            (unsigned long long) st.trunc_cached, (unsigned long long) st.trunc_pmin,
            (unsigned long long) st.trunc_nmax,   (unsigned long long) st.trunc_fail,
            p_mean, p_hi);
}

// PXA_MTP_STATS: close one draft cycle. `stop` is 0 cached-token p_min collapse, 1 in-loop p_min
// break, 2 the full n_max chain, 3 a decode/embedding failure.
static void pxa_mtp_stats_record(pxa_mtp_stats_t & st, int64_t t0, size_t n_drafted, int n_decode, int stop) {
    if (!st.on) {
        return;
    }
    st.cycles      += 1;
    st.drafted     += n_drafted;
    st.mtp_decodes += (uint64_t) (n_decode > 0 ? n_decode : 0);
    st.draft_us    += (uint64_t) std::max<int64_t>(0, ggml_time_us() - t0);
    st.len_hist[n_drafted < 8 ? n_drafted : 8] += 1;
    switch (stop) {
        case 0:  st.trunc_cached += 1; break;
        case 1:  st.trunc_pmin   += 1; break;
        case 3:  st.trunc_fail   += 1; break;
        default: st.trunc_nmax   += 1; break;
    }
    if (++st.since_dump >= (uint64_t) st.every) {
        st.since_dump = 0;
        pxa_mtp_stats_dump("running");
    }
}

struct mtp_last_embd {
    std::vector<float> embd;
    float prob = 0.0f;
    int   last_id = -1;
    // PXA_SPEC_SAMPLED: the proposal distribution `last_id` was DRAWN from, when it was drawn
    // rather than argmax-ed. Empty means "this token has no q", which the verifier reads as
    // "verify this position by exact match".
    std::vector<pxa_spec_cand> q;
};

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_mtp = nullptr;
    common_sampler * smpl;
    // For Gemma 4 external MTP assistant: draft positions are held constant
    bool constant_draft_positions = false;
    int n_embd = 0;
    std::unordered_map<llama_seq_id, std::vector<float>> target_hidden_by_seq;
    std::unordered_map<llama_seq_id, mtp_last_embd> draft_cache_by_seq;

    // PXA_SPEC_SAMPLED: per-seq, the request's own sampler (its parameters AND its rng, so seeded
    // runs stay reproducible) and the proposal distributions of the draft this seq last produced,
    // one entry per drafted token and in draft order.
    std::unordered_map<llama_seq_id, common_sampler *> pxa_req_smpl_by_seq;
    std::unordered_map<llama_seq_id, std::vector<std::vector<pxa_spec_cand>>> pxa_draft_q_by_seq;

    // PXA_MTP_ADAPTIVE_K controller state (inert unless PXA_MTP_ADAPTIVE_K=1).
    float    pxa_ak_ema     = 0.70f; // running acceptance EMA (optimistic cold-start = full-depth drafts)
    uint32_t pxa_ak_drafted = 0;     // draft tokens emitted last cycle (accept() denominator)

    // PXA_SPEC_MTP_LAZY_v1 (2026-09-15): state for a companion that has not been
    // built yet. See the block comment on the promotion path below.
    llama_context_params lazy_cparams = {};
    bool   lazy_pending  = false; // the chain carries this stage, its companion does not exist yet
    bool   lazy_failed   = false; // a first-use build was attempted and failed: stage is dead
    // PER SEQUENCE. One companion is SHARED by every slot (the server aliases a single chain
    // across them), so a single counter here is a single counter for the whole server: a slot
    // whose earlier stage wins its steps would zero the run of every other slot, and a slot on
    // fresh text -- the traffic this stage exists for -- would never reach the promotion
    // threshold for the life of the boot. The run is a property of one sequence's steps.
    std::unordered_map<llama_seq_id, size_t> lazy_asks_by_seq;
    size_t lazy_asks_max = 0;     // high-water mark across sequences, for the statistics line

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_mtp,
            bool constant_draft_positions = false,
            const llama_context_params * lazy_cparams_in = nullptr)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_mtp(ctx_mtp)
        , constant_draft_positions(constant_draft_positions)
    {
        if (ctx_mtp == nullptr) {
            GGML_ASSERT(lazy_cparams_in != nullptr);
            lazy_cparams = *lazy_cparams_in;
            lazy_pending = true;
            smpl         = nullptr;
            LOG_INF("%s: MTP companion DEFERRED (PXA_SPEC_MTP_LAZY): no context, no KV and no target "
                    "embedding output are claimed until this stage is actually asked to draft\n", __func__);
            return;
        }

        bind_ctx();
    }

    void bind_ctx() {
        struct common_params_sampling sparams;
        sparams.samplers_sequence = {
            llama_sampler_type::DIST,
        };
        smpl = common_sampler_init(llama_get_model(ctx_mtp), sparams);
        llama_set_mtp_target_context(ctx_mtp, ctx_tgt);
        n_embd = llama_mtp_state_n_embd(ctx_mtp);

        LOG_INF("%s: MTP context ready (n_ctx=%d, constant_draft_positions=%s)\n", __func__,
                llama_n_ctx(ctx_mtp), constant_draft_positions ? "true" : "false");
    }

    // PXA_SPEC_MTP_LAZY_v1. THE COMPANION IS A BOOT-TIME CONSTANT PAID BY EVERY REQUEST AND USED
    // BY SOME. Arming an MTP stage at init costs three separate things, every one of them charged
    // to traffic that never promotes: the companion context and its KV; the checkpoint headroom a
    // later companion allocation has to be reserved out of (which is what shortens every other
    // stage's draft); and -- the one nobody had named -- llama_set_embeddings(ctx_tgt, true) plus
    // the per-step ensure_sequence_hidden/commit bookkeeping on the TARGET, which every decode of
    // every request then pays whether or not this stage ever drafts. Building on first use moves
    // all three to the run that asked for them. If the build fails (the VRAM is gone by the time
    // we ask), the chain degrades to whatever else it carries with ONE log line and never aborts.
    bool lazy_ensure() {
        if (ctx_mtp != nullptr) {
            return true;
        }
        if (lazy_failed) {
            return false;
        }

        const llama_model * model = llama_get_model(ctx_tgt);
        ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model), lazy_cparams);
        if (ctx_mtp == nullptr) {
            lazy_failed  = true;
            lazy_pending = false;
            LOG_WRN("%s: MTP companion could not be built on first use - the free VRAM that was there at "
                    "boot is not there now. This chain continues WITHOUT its MTP stage; every other stage "
                    "is unaffected and nothing is aborted.\n", __func__);
            return false;
        }

        bind_ctx();
        lazy_pending = false;
        LOG_INF("%s: MTP companion built ON FIRST USE after %zu consecutive steps on one sequence with "
                "no other stage able to draft (PXA_SPEC_MTP_LAZY_N)\n", __func__, lazy_asks_max);
        return true;
    }

    ~common_speculative_state_mtp() override {
        if (smpl) {
            common_sampler_free(smpl);
        }
        if (ctx_mtp) {
            llama_free(ctx_mtp);
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        target_hidden_by_seq.clear();
        draft_cache_by_seq.clear();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        draft(params, prompt_tgt, id_last, -1, 0, result);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_pos draft_base_pos,
            llama_seq_id seq_id,
            llama_tokens & result) override {

        // PXA_SPEC_MTP_LAZY_v1: this stage is only ever reached when every earlier stage in the
        // chain drafted nothing, so reaching it IS the demand signal. Promote on a run of them,
        // not on one (a single silent step on a class the table otherwise owns must not buy a
        // gigabyte); PXA_SPEC_MTP_LAZY_N is that run length, default 16, 1 = literal first use.
        if (ctx_mtp == nullptr) {
            result.clear();
            if (lazy_failed) {
                return;
            }
            size_t & lazy_asks = lazy_asks_by_seq[seq_id];
            lazy_asks++;
            if (lazy_asks > lazy_asks_max) {
                lazy_asks_max = lazy_asks;
            }
            static const size_t pxa_lazy_n = []() -> size_t {
                const char * e = getenv("PXA_SPEC_MTP_LAZY_N");
                return e ? (size_t) std::max(1, atoi(e)) : (size_t) 16;
            }();
            if (lazy_asks >= pxa_lazy_n) {
                lazy_ensure();
            }
            // Even a successful build cannot draft on THIS step: the target hidden row for this
            // sequence is produced by the server only once it knows the stage is live.
            return;
        }

        // PXA_LLAMA_MTP_NP_FIX: ctx_mtp is single-seq per slot -> query local row 0.
        const llama_seq_id mtp_kv_seq = llama_n_seq_max(ctx_mtp) <= 1 ? 0 : seq_id;
        const llama_pos mtp_pos_max = llama_kv_cache_seq_pos_max(ctx_mtp, mtp_kv_seq);
        const bool has_draft_base_pos = draft_base_pos >= 0;
        // Prefer the target slot position when the caller has it. Gemma4 external MTP reads
        // the target KV cache directly, so ctx_mtp's own KV position is not authoritative.
        const llama_pos n_past = has_draft_base_pos
            ? draft_base_pos
            : (mtp_pos_max >= 0 ? mtp_pos_max + 1 : (llama_pos) prompt_tgt.size());

        if (!has_draft_base_pos && !prompt_tgt.empty() && mtp_pos_max < (llama_pos)prompt_tgt.size() - 1) {
            LOG_WRN("%s: MTP context not fully warmed up: pos_max = %d, expected = %d\n",
                    __func__, (int)mtp_pos_max, (int)prompt_tgt.size() - 1);
        }
        if (has_draft_base_pos && !constant_draft_positions && mtp_pos_max < n_past - 1) {
            LOG_WRN("%s: MTP context not fully warmed up: pos_max = %d, expected >= %d\n",
                    __func__, (int)mtp_pos_max, (int)n_past - 1);
        }

        llama_context * ctx = ctx_mtp;

        const auto hidden_it = target_hidden_by_seq.find(seq_id);
        if (hidden_it == target_hidden_by_seq.end() || (int) hidden_it->second.size() != n_embd) {
            LOG_WRN("%s: missing target hidden state for seq_id %d\n", __func__, (int) seq_id);
            result.clear();
            return;
        }

        if (!llama_set_draft_input_hidden_state_copy(ctx, hidden_it->second.data(), hidden_it->second.size())) {
            result.clear();
            return;
        }

        result = mtp_speculative_gen_draft(
            *this,
            smpl,
            ctx,
            params.n_max,
            params.p_min,
            id_last,
            n_past,
            seq_id,
            constant_draft_positions
        );
    }

    // PXA_SPEC_CHAIN_ACCEPT_v1 + PXA_SPEC_MTP_LAZY_v1: another stage won this step, so the run
    // of "nobody else could draft" is broken. Nothing else in this stage's state may move: its
    // acceptance EMA is a verdict on its own drafts and the target never saw one here.
    void observe_step(uint16_t n_accepted, bool drafted, llama_seq_id seq_id) override {
        GGML_UNUSED(n_accepted);
        GGML_UNUSED(drafted);
        lazy_asks_by_seq.erase(seq_id);
    }

    std::string extra_stats() const override {
        if (!lazy_pending && !lazy_failed && lazy_asks_max == 0) {
            return std::string();
        }
        char buf[192];
        snprintf(buf, sizeof(buf), ", lazy(pending,failed,run_max) = %d %d %zu",
                 (int) lazy_pending, (int) lazy_failed, lazy_asks_max);
        return std::string(buf);
    }

    void accept(uint16_t n_accepted) override {
        // PXA_MTP_STATS: the target has just told us how many of the proposed tokens it kept.
        if (auto & st = pxa_mtp_stats(); st.on) {
            st.accepted     += n_accepted;
            st.accept_calls += 1;
            st.acc_hist[n_accepted < 8 ? n_accepted : 8] += 1;
        }
        // PXA_MTP_ADAPTIVE_K: fold the realized accept ratio into the running EMA that drives K.
        if (pxa_mtp_adaptive_k_enabled() && pxa_ak_drafted > 0) {
            float f = (float) n_accepted / (float) pxa_ak_drafted;
            if (f > 1.0f) f = 1.0f;
            pxa_ak_ema = 0.85f * pxa_ak_ema + 0.15f * f;
            pxa_ak_drafted = 0;
        } else {
            GGML_UNUSED(n_accepted);
        }
    }
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;
        {
            struct common_params_sampling params;
            params.top_k = 10;
            params.samplers_sequence = {
                llama_sampler_type::TOP_K,
                llama_sampler_type::DIST, // needed to get probabilities
            };
            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);



            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_kv_cache_clear(ctx_dft);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_kv_cache_seq_rm (ctx_dft, 0, 0, reuse_i);
                llama_kv_cache_seq_add(ctx_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_kv_cache_seq_rm (ctx_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);
        if (getenv("PXA_DRAFT_DBG")) {
            LOG_WRN("PXA_DRAFT_DBG call: vocab_cmpt=%d prompt_dft=%d n_past=%d id_last=%d\n",
                (int)spec->vocab_cmpt, (int)prompt_dft.size(), (int)n_past, (int)id_last);
        }

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;
            if (getenv("PXA_DRAFT_DBG")) {
                LOG_WRN("PXA_DRAFT_DBG step %d: top id=%d p=%.3f (c1=%d)\n",
                    i, (int)cur_p->data[0].id, (double)cur_p->data[0].p,
                    cur_p->size>1 ? (int)cur_p->data[1].id : -1);
            }

            common_sampler_accept(smpl, nullptr, id, true);

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                if (i == 0) {
                    result.push_back(id);
                }
                break;
            }

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }


            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    // PXA_SPEC_STAGE_DIAG_v1: why a draft call produced nothing, resolved.
    size_t n_miss_first   = 0; // the table had NO continuation for the current suffix (i == 0 miss)
    size_t n_short_of_min = 0; // it had a continuation but fewer than params.n_min of them
    size_t n_ctx_short    = 0; // prompt_tgt shorter than the n-gram width
    size_t n_feed_calls   = 0; // draft calls that actually added entries to the index
    size_t n_feed_lag_sum = 0; // sum over draft calls of (cur_len - n) - i_last BEFORE feeding
    size_t n_feed_lag_max = 0; // worst single lag seen
    size_t n_not_asked    = 0; // steps this stage drafted and was never told the outcome

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;
        n_low = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            n_ctx_short++;
            return;
        }

        const size_t n = mod.get_n();

        // PXA_SPEC_NGRAM_FEED_LAG_v1 (2026-09-15). THE INDEX IS THIS STAGE'S ONLY
        // MEMORY, and accept() never touches it -- it is fed here, from the target's context, and
        // it used to be fed only once every 32 tokens. That chunking is invisible while this stage
        // wins its own steps (it then advances the context ~n_max tokens per step, so the lag is
        // spent in one or two calls), and it is ruinous the moment it SHARES a chain: a stage that
        // advances the context ~3 tokens per step spends a dozen consecutive steps looking up a
        // suffix whose own tail was never indexed. The sibling learned-table stage in this same
        // file (common_speculative_state_suffix) feeds every call with zero lag, which is what the
        // contract should always have been. Lever: PXA_SPEC_NGRAM_FEED_LAG, tokens of permitted
        // lag; 0 = feed every call (this branch's default), 32 = the pre-2026-09-15 behaviour.
        static const size_t pxa_feed_lag = []() -> size_t {
            const char * e = getenv("PXA_SPEC_NGRAM_FEED_LAG");
            return e ? (size_t) std::max(0, atoi(e)) : (size_t) 0;
        }();

        // A CONTEXT SHIFT SHRINKS prompt_tgt. i_last is otherwise only ever moved forward, so a
        // shift leaves it pointing past the end of the context it indexes, and the feed test
        // below is then false on every call until the context grows back past i_last + n -- for
        // the whole of the next n_discard generated tokens this stage indexes nothing and looks
        // up suffixes whose own tail was never added, which is the starvation this lever exists
        // to remove. Re-anchor before the test rather than letting the guard strand the pointer.
        if (i_last > cur_len - n) {
            i_last = cur_len - n;
        }

        // add new ngrams in chunks
        if (i_last + pxa_feed_lag < cur_len && i_last < cur_len - n) {
            const size_t lag = (cur_len - n) - i_last;
            n_feed_lag_sum += lag;
            if (lag > n_feed_lag_max) {
                n_feed_lag_max = lag;
            }
            n_feed_calls++;
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i == 0) {
                    n_miss_first++;
                }
                if (i < params.n_min) {
                    if (i > 0) {
                        n_short_of_min++;
                    }
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            // PXA_NGRAM_RESET_STREAK (2026-07-09): the hardcoded streak-3 full-map wipe
            // thrashes the 4M map on varied-writer models (accept~0.64 -> frequent f_acc<0.5 rounds ->
            // the map never warms). Env-tunable: default 3 = upstream; 0 = never reset on acceptance.
            static const int pxa_reset_streak = getenv("PXA_NGRAM_RESET_STREAK") ? atoi(getenv("PXA_NGRAM_RESET_STREAK")) : 3;
            if (f_acc < 0.5) {
                n_low++;
                if (pxa_reset_streak > 0 && n_low >= pxa_reset_streak) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }

    // PXA_SPEC_CHAIN_ACCEPT_v1: this stage drafted, another stage won the step. Deliberately do
    // NOT fold that step into n_low: the streak is a verdict on THIS table's own drafts, and the
    // target never tested the draft this table produced. What the step is allowed to do is clear
    // the "never told" record so the counter below stays honest.
    void observe_step(uint16_t n_accepted, bool drafted, llama_seq_id seq_id) override {
        GGML_UNUSED(n_accepted);
        GGML_UNUSED(seq_id);
        if (drafted) {
            n_not_asked++;
        }
    }

    std::string extra_stats() const override {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 ", miss@0 = %zu, short<n_min = %zu, ctx<n = %zu, feeds = %zu, lag(sum,max) = %zu %zu, untested = %zu",
                 n_miss_first, n_short_of_min, n_ctx_short, n_feed_calls, n_feed_lag_sum, n_feed_lag_max, n_not_asked);
        return std::string(buf);
    }

    // PXA_SLOT_ERASE_SPEC_HARD_v1: the entries mod.add() accumulates are exactly the "warm map"
    // that lets a second identical request draft longer than the first (begin() only re-hashes
    // the current prompt INTO the same table; it does not clear it, by design, so the table stays
    // useful across a slot's own follow-up turns). A slot ERASE ends that lifetime, so the table
    // -- and the adaptive streak state that rides beside it -- must go with it.
    void hard_reset() override {
        mod.reset();
        i_last       = 0;
        n_draft_last = 0;
        n_low        = 0;
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_suffix : public common_speculative_state {
    common_suffix_tree tree;
    common_suffix_tree corpus_tree;
    bool has_corpus = false;
    size_t cache_size   = 0;

    // Acceptance feedback
    size_t n_draft_last  = 0;
    bool   had_accept    = false;
    int    n_low         = 0;
    float  base_p_min    = 0.1f;
    float  eff_p_min     = 0.1f;

    common_speculative_state_suffix(
            enum common_speculative_type type,
            int max_depth,
            const std::string & corpus_path,
            const llama_model * model)
        : common_speculative_state(type)
        , tree(max_depth)
        , corpus_tree(max_depth)
    {
        if (!corpus_path.empty()) {
            std::function<std::vector<llama_token>(const std::string &)> tokenize_fn;
            if (model) {
                tokenize_fn = [model](const std::string & text) -> std::vector<llama_token> {
                    return common_tokenize(model, text, false, true);
                };
            }
            has_corpus = corpus_tree.load_corpus(corpus_path, tokenize_fn);
        }
    }

    void begin(const llama_tokens & prompt) override {
        cache_size   = 0;
        n_draft_last = 0;
        had_accept   = false;
        n_low        = 0;
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        base_p_min = params.p_min;
        if (n_draft_last > 0 && !had_accept) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        }
        had_accept = false;

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last);

            tree.extend(tokens_new.data(), (int)tokens_new.size());
            cache_size = prompt_tgt.size() + 1;
        }

        const int ctx_len = std::min((int)(prompt_tgt.size() + 1), tree.max_depth());
        llama_tokens context;
        context.reserve(ctx_len);
        const int ctx_start = (int)prompt_tgt.size() + 1 - ctx_len;
        for (int j = ctx_start; j < (int)prompt_tgt.size(); ++j) {
            context.push_back(prompt_tgt[j]);
        }
        context.push_back(id_last);
        const int min_match_len = std::max(1, params.suffix_min_match_len);

        result = tree.speculate(
            context.data(), (int)context.size(),
            params.n_max,
            eff_p_min,
            1,
            min_match_len);

        if (has_corpus) {
            auto corpus_result = corpus_tree.speculate(
                context.data(), (int)context.size(),
                params.n_max,
                eff_p_min,
                1,
                min_match_len);
            if (corpus_result.size() > result.size()) {
                result = std::move(corpus_result);
            }
        }

        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (n_draft_last == 0) {
            return;
        }
        had_accept = true;
        const double f_acc = (double)n_accepted / (double)n_draft_last;
        if (f_acc < 0.5) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        } else {
            n_low = 0;
            if (eff_p_min > base_p_min) {
                eff_p_min = std::max(eff_p_min - 0.05f, base_p_min);
            }
        }
    }
};

struct common_speculative {
    std::vector<common_speculative_config> configs; // resolved stage config for each implementation
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
    // PXA_SPEC_CHAIN_ACCEPT_v1: how many stages the draft loop actually REACHED on this
    // sequence's last step. The loop stops at the first stage that produces a usable draft, so
    // every stage after it was never asked and must not be told it drafted.
    std::unordered_map<llama_seq_id, size_t> n_asked_by_seq;
    std::unique_ptr<spec_tuner> tuner;
    int last_n_drafted = 0;
    int64_t t_step_start_us = 0;
};

static bool common_speculative_stage_chain_matches(
        const std::vector<common_speculative_stage_params> & stages,
        const std::vector<common_speculative_config> & configs) {
    if (stages.size() != configs.size()) {
        return false;
    }

    for (size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].type != configs[i].type) {
            return false;
        }
    }

    return true;
}

// PXA_MTP_PMIN_v1 (2026-09-08; DEFAULT REVERTED 2026-09-09): resolve
// the confidence floor for an MTP stage that carries no explicit `p_min=` in its --spec-type entry.
//
// The floor decides how long a draft chain may get: mtp_speculative_gen_draft() stops at the first
// token whose FULL-VOCAB softmax top-1 probability falls under it, and collapses the whole chain to
// the single cached token when the carried-over token is under it. Our stock draft default is 0.75
// (COMMON_SPEC_P_MIN_DEFAULT); upstream's own MTP drafter defaults the same knob to 0.0.
//
// MEASURED, 2026-09-09 (2x V100-16GB, Qwen3.8-27B PXQ4, decode 256 np1,
// median of 3; measured 2026-09-08):
//
//   plain 36.73 t/s | p_min=0.75: n1 44.71  n2 42.71  n3 42.84  n4 42.35
//                   | p_min=0.00: n1 44.75  n2 35.11  n3 29.55  n4 28.94
//
// Dropping the floor did exactly what it claimed -- the proposed-length histogram moved from "1 in
// 100% of cycles" to "n_max in 100% of cycles" -- and it made the engine SLOWER at every depth
// above 1, because the target accepted a depth->=2 proposal in only 9 of 388 verifies (2.3%). The
// deep chain is not paying for itself while the MTP KV rows the draft attends to are one position
// out (PXA_MTP_KVPOS_v1, below). So the default goes back to the stock floor -- which at the shipped
// n_max=1 is inert anyway (a chain of 1 cannot be truncated below 1) -- and 0.0 stays one env var
// away as the lab value that re-runs the experiment once the alignment fix has been measured.
//
//   unset            -> inherit the global p_min (0.75 stock) -- the measured-best default
//   PXA_MTP_PMIN=0   -> no floor: propose the full n_max every cycle (the 2026-09-08 experiment)
//   PXA_MTP_PMIN=X   -> any explicit floor
//   PXA_MTP_PMIN=-1  -> inherit the global p_min (identical to unset; kept so the lever's
//                       "inherit" spelling keeps working for scripts that already use it)
//
// An explicit `--spec-type mtp:...,p_min=X` always wins over all of the above, and so does a
// per-request `speculative.p_min`.
static float pxa_mtp_stage_p_min(float inherited) {
    static const char * env     = getenv("PXA_MTP_PMIN");
    static const bool   has_env = env != nullptr;
    static const float  env_val = has_env ? (float) atof(env) : 0.0f;

    if (has_env) {
        return env_val < 0.0f ? inherited : env_val;
    }

    return inherited;
}

static common_params_speculative common_speculative_get_runtime_params(
        const common_speculative_config & config,
        const common_params_speculative & params,
        const common_speculative_stage_params & stage) {
    common_params_speculative result = config.params;

    result.type = config.type;
    result.n_max = stage.has_n_max_override() ? stage.n_max : params.n_max;
    result.n_min = stage.has_n_min_override() ? stage.n_min : params.n_min;
    result.p_min = stage.has_p_min_override() ? stage.p_min : params.p_min;

    // PXA_MTP_PMIN_v1: the MTP self-speculation stage does not inherit the draft-model floor.
    if (config.type == COMMON_SPECULATIVE_TYPE_MTP && !stage.has_p_min_override()) {
        result.p_min = pxa_mtp_stage_p_min(result.p_min);
    }

    if (config.type == COMMON_SPECULATIVE_TYPE_SUFFIX) {
        result.suffix_min_match_len = stage.has_suffix_min_match_len_override()
            ? stage.suffix_min_match_len
            : params.suffix_min_match_len;
    }

    result.n_max = std::max(result.n_max, 0);
    result.n_min = std::max(0, std::min(result.n_min, result.n_max));
    result.stages.clear();

    return result;
}

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    std::string normalized = name;
    std::replace(normalized.begin(), normalized.end(), '-', '_');

    const auto it = common_speculative_type_from_name_map.find(normalized);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    bool res = true;

    llama_kv_cache_clear(ctx_tgt);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size(), 0, 0));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_kv_cache_seq_rm(ctx_tgt, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_kv_cache_clear(ctx_tgt);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    std::string chain_error;
    if (!common_speculative_validate_chain(params, &chain_error)) {
        LOG_ERR("%s: invalid speculative stage chain: %s\n", __func__, chain_error.c_str());
        return nullptr;
    }

    // PXA_CKPT_BUDGET: a recurrent target checkpoints every speculation window, and the per-step
    // checkpoint buffers scale linearly with the draft length - 378+227 MiB at 5 tokens, 5988+3593
    // at 65 on a 2x16 GB pair with Qwen3.8-27B PXQ4. Auto mode used to claim them whenever the
    // allocation merely fitted, leaving one card with ~0.7 GB and killing the first sizeable
    // compute-pool allocation a dozen cycles later. Ask what the context can AFFORD before the
    // drafters are built, and clamp the chain to it: a draft length the checkpoint cannot carry is
    // not a faster draft, it is an out-of-memory a minute later.
    // Only the per-step mode claims the buffers the budget is about; gpu-fallback and cpu keep
    // small shadows whatever the draft length is, so clamping the chain for them would cost draft
    // depth for nothing.
    const bool pxa_ckpt_budget_applies = params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_AUTO ||
                                         params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_PER_STEP;
    if (pxa_ckpt_budget_applies && llama_model_has_recurrent(llama_get_model(ctx_tgt))) {
        const int want   = std::max(1, params.get_max_stage_n_max() + 1);
        const int afford = llama_spec_ckpt_budget_max_tokens(ctx_tgt, want);
        // Print it every boot, clamp or no clamp: which capacity a run actually got is not
        // otherwise visible anywhere, and two arms that differ only in it have been compared as
        // if they were like for like.
        const int eff_n_max = afford >= 2 ? std::min(want - 1, afford - 1) : want - 1;
        LOG_INF("%s: recurrent checkpoint budget: the chain drafts up to %d tokens (capacity %d), "
                "this context can checkpoint %d -> effective n_max = %d\n",
                __func__, want - 1, want, afford, eff_n_max);
        if (afford >= 2 && afford < want) {
            const int32_t cap = afford - 1;
            LOG_INF("%s: recurrent checkpoint budget: the chain asks to draft %d tokens, this context can "
                    "checkpoint %d - every stage is clamped to n_max=%d\n",
                    __func__, want - 1, afford, cap);
            if (params.n_max > cap) {
                params.n_max = cap;
            }
            for (auto & st : params.stages) {
                // effective n_max of a stage: its own override, else the global draft length
                if ((st.has_n_max_override() ? st.n_max : params.n_max) > cap) {
                    st.n_max = cap;
                }
            }
            if (params.n_min > cap) {
                params.n_min = cap;
            }
            for (auto & st : params.stages) {
                if (st.n_min > cap) {
                    st.n_min = cap;
                }
            }
        } else if (afford < 2) {
            LOG_WRN("%s: recurrent checkpoint budget: this context cannot afford even a two-token per-step "
                    "checkpoint; the checkpoint mode is left to resolve itself\n", __func__);
        }
    }

    const auto stages = params.get_resolved_stages();
    if (params.model_dft && llama_model_is_gemma4_mtp_assistant(params.model_dft)) {
        const bool has_draft_stage = std::any_of(stages.begin(), stages.end(), [](const common_speculative_stage_params & stage) {
            return stage.type == COMMON_SPECULATIVE_TYPE_DRAFT;
        });

        if (has_draft_stage) {
            LOG_ERR("%s: Gemma4 assistant models only support MTP stages; omit -md for self-spec-only runs or use --spec-type mtp:n_max=1,p_min=0.0 for assistant-backed MTP\n", __func__);
            return nullptr;
        }
    }

    const bool needs_draft_ctx = std::any_of(stages.begin(), stages.end(), [&params](const common_speculative_stage_params & stage) {
        return stage.type == COMMON_SPECULATIVE_TYPE_DRAFT ||
               (stage.type == COMMON_SPECULATIVE_TYPE_MTP && params.model_dft != nullptr);
    });

    llama_context * ctx_dft = nullptr;
    if (needs_draft_ctx) {
        if (!params.model_dft) {
            LOG_ERR("%s: draft speculative stage requires a loaded draft model\n", __func__);
            return nullptr;
        }

        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the resolved stage chain.
    std::vector<common_speculative_config> configs = {};
    configs.reserve(stages.size());

    for (const auto & stage : stages) {
        common_params_speculative stage_params = params.with_stage_overrides(stage);

        if (stage.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD && !stage_params.ngram_mod) {
            stage_params.ngram_mod = std::make_shared<common_ngram_mod>(stage_params.ngram_size_n, 4*1024*1024);

            LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                    stage_params.ngram_size_n, stage_params.ngram_mod->size(),
                    (float)(stage_params.ngram_mod->size_bytes())/1024/1024);

            if (stage_params.ngram_size_n < 16) {
                LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, stage_params.ngram_size_n);
            }
        }

        // PXA_MTP_PMIN_v1: make the resolved MTP confidence floor visible in the boot log -- it is
        // what actually decides the draft chain length, and it is not printed anywhere else.
        if (stage.type == COMMON_SPECULATIVE_TYPE_MTP) {
            const float pxa_p_min = stage.has_p_min_override()
                ? stage.p_min
                : pxa_mtp_stage_p_min(params.p_min);
            const int pxa_topk = pxa_mtp_pmin_topk();
            LOG_INF("%s: MTP stage: n_max=%d, p_min=%.3f (%s), p_min compares %s%s\n", __func__,
                    stage.has_n_max_override() ? stage.n_max : params.n_max,
                    (double) pxa_p_min,
                    stage.has_p_min_override() ? "explicit p_min (CLI or PXA_AUTO)" :
                    (getenv("PXA_MTP_PMIN") ? "PXA_MTP_PMIN" : "stock floor, measured best 2026-09-09"),
                    // PXA_MTP_PMIN_TOPK_v1: the floor's SCALE is as decisive as its value -- 0.75 on a
                    // full-vocab softmax over 248320 tokens is an off switch, 0.75 over 10 candidates
                    // is a filter. Print which one is armed next to the number it is compared with.
                    pxa_topk >= 2 ? "the top-" : "the FULL-VOCABULARY softmax",
                    pxa_topk >= 2 ? (std::to_string(pxa_topk) + "-renormalised probability"
                                     + (getenv("PXA_MTP_PMIN_TOPK") ? " (PXA_MTP_PMIN_TOPK)" : "")).c_str()
                                  : (getenv("PXA_MTP_PMIN_TOPK") ? " (PXA_MTP_PMIN_TOPK=0)" : " (PXA_REFERENCE)"));
        }

        configs.push_back(common_speculative_config(stage, stage_params));
    }

    if (!configs.empty() && llama_model_has_recurrent(llama_get_model(ctx_tgt))) {
        const int ckpt_tokens = std::max(1, params.get_max_stage_n_max() + 1);
        const int actual_mode = llama_spec_ckpt_init(ctx_tgt, params.recurrent_ckpt_mode, ckpt_tokens);
        if (actual_mode == LLAMA_SPEC_CKPT_NONE) {
            LOG_ERR("%s: failed to prepare recurrent checkpoint mode '%s' during speculative init (max_tokens=%d)\n",
                    __func__,
                    params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_PER_STEP ? "per-step" :
                    params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_GPU_FALLBACK ? "gpu-fallback" :
                    params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_CPU ? "cpu" : "auto",
                    ckpt_tokens);
            if (ctx_dft != nullptr) {
                llama_free(ctx_dft);
            }
            return nullptr;
        }
        llama_spec_ckpt_discard(ctx_tgt);
        params.recurrent_ckpt_mode = actual_mode;
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ config.params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                llama_context * ctx_mtp = ctx_dft;

                // PXA_SPEC_MTP_LAZY_v1 (2026-09-15): defer the companion when this
                // stage SHARES a chain -- i.e. when something else can draft if it never does.
                // A chain whose only stage is MTP has nothing to fall back to, so it is built
                // eagerly exactly as before and nothing about a plain `--spec-type mtp` run
                // changes. An externally supplied draft context (Gemma 4 assistant) is likewise
                // never deferred: it is not ours to build.
                static const bool pxa_mtp_lazy = []() {
                    const char * e = getenv("PXA_SPEC_MTP_LAZY");
                    return e ? atoi(e) != 0 : true;
                }();
                const bool pxa_defer = ctx_mtp == nullptr && pxa_mtp_lazy && configs.size() > 1;

                if (!ctx_mtp && !pxa_defer) {
                    const llama_model * model = llama_get_model(ctx_tgt);
                    ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model), config.params.cparams_dft);
                    if (!ctx_mtp) {
                        LOG_ERR("%s: failed to create MTP context\n", __func__);
                        return nullptr;
                    }
                }
                ctx_dft = nullptr;

                const llama_model * model_for_flags = ctx_mtp ? llama_get_model(ctx_mtp) : llama_get_model(ctx_tgt);
                const bool use_constant_draft_positions = llama_model_is_gemma4_mtp_assistant(model_for_flags);
                impls.push_back(std::make_unique<common_speculative_state_mtp>(
                    config.type, ctx_tgt, ctx_mtp, use_constant_draft_positions,
                    pxa_defer ? &config.params.cparams_dft : nullptr));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config.params.lookup_cache_static, config.params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                int depth = config.params.suffix_max_depth > 0 ? config.params.suffix_max_depth : 64;
                const llama_model * model = llama_get_model(ctx_tgt);
                impls.push_back(std::make_unique<common_speculative_state_suffix>(
                    config.type, depth, config.params.suffix_corpus, model));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .configs = */ std::move(configs),
        /* .impls = */ std::move(impls)
    };

    // initialize autotune if requested
    if (params.autotune && params.has_composite_stage_chain()) {
        LOG_WRN("Autotune disabled — explicit speculative stage chains are not supported yet\n");
    } else if (params.autotune && !result->impls.empty()) {
        auto actual_type = result->impls[0]->type;
        if (actual_type != COMMON_SPECULATIVE_TYPE_NONE &&
            actual_type != COMMON_SPECULATIVE_TYPE_EAGLE3) {
            result->tuner = std::make_unique<spec_tuner>();
            result->tuner->init(actual_type, params, llama_get_model(ctx_tgt));
            LOG_DBG("Autotune initialized for %s, tuning %zu parameters\n",
                    common_speculative_type_to_str(actual_type).c_str(),
                    result->tuner->coords.size());
        } else {
            LOG_WRN("Autotune disabled — speculative type %s is not supported for autotuning\n",
                    common_speculative_type_to_str(actual_type).c_str());
        }
    }

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

// PXA_SHARED_MTP_v1: per-seq begin. For the SHARED MTP impl, clear ONLY this seq's caches so a new
// generation in one slot does not wipe the other slots' in-flight MTP state. Non-MTP impls are
// per-slot (unshared) -> their whole-spec begin() is still correct, so fall back to it for those.
void common_speculative_begin_seq(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    auto * mtp_state = common_speculative_get_mtp_state(spec);

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        if (mtp_state != nullptr && impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
            // per-seq reset only (mirrors begin()'s clear, scoped to this seq)
            mtp_clear_target_hidden(*mtp_state, seq_id);
        } else {
            impl->begin(prompt);
        }
        impl->n_call_begin++;
    }
}

// PXA_SLOT_ERASE_SPEC_HARD_v1 (2026-09-14): full teardown of every impl's persistent state, for a
// slot ERASE. Distinct from begin()/begin_seq(), which run at the start of every generation and
// deliberately leave a stage's learned table (e.g. common_ngram_mod) in place so it stays warm
// across a slot's own follow-up turns -- that table is what an erase must actually clear, since
// the slot's next occupant should not draft against the previous occupant's learned map.
void common_speculative_hard_reset(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->hard_reset();
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last,
        llama_pos draft_base_pos,
        llama_seq_id draft_seq_id) {
    llama_tokens result;

    pxa_mtp_prefetch_wait_seq(draft_seq_id); // PXA_MTP_PREFETCH: drain in-flight companion commit before touching ctx_mtp

    spec->t_step_start_us = ggml_time_us();

    // apply autotune proposal if enabled
    if (spec->tuner && spec->tuner->enabled) {
        spec->tuner->propose(params);
    }

    const auto runtime_stages = params.get_resolved_stages();
    const bool use_runtime_stage_overrides = common_speculative_stage_chain_matches(runtime_stages, spec->configs);

    spec->curr_impl = nullptr; // reset current implementation

    size_t n_asked = 0; // stages the loop below actually reached (PXA_SPEC_CHAIN_ACCEPT_v1)

    // PXA_SPEC_SAMPLED: this seq's proposal distributions start EMPTY for every step, whatever
    // stage ends up producing the draft and whatever early return an impl takes on the way there.
    // The MTP drafter clears them as it fills them, but it is not always reached: its own draft()
    // returns before that clear when the seq has no stored target hidden state or the hidden-state
    // copy fails, and in a cascade an earlier stage that hits means it never runs at all. Either
    // way the previous step's q must not survive to be paired with this step's tokens -- that
    // pairing is not a small error, it is a different (wrong) output distribution. Cleared per
    // seq, because one speculative object is shared by every slot and its curr_impl is therefore
    // the last slot to have drafted, not this one.
    if (common_sampler_spec_sampled_active()) {
        if (auto * qst = common_speculative_get_mtp_state(spec)) {
            const auto qit = qst->pxa_draft_q_by_seq.find(draft_seq_id);
            if (qit != qst->pxa_draft_q_by_seq.end()) {
                qit->second.clear();
            }
        }
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto & impl = spec->impls[i];
        const auto & runtime_stage = use_runtime_stage_overrides ? runtime_stages[i] : spec->configs[i].stage;
        common_params_speculative impl_params = common_speculative_get_runtime_params(spec->configs[i], params, runtime_stage);
        result.clear();

        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(impl_params, prompt_tgt, id_last, draft_base_pos, draft_seq_id, result);
            impl->n_call_draft++;
        }
        n_asked = i + 1; // this stage WAS reached, whatever it returned

        if (prompt_tgt.empty()) {
            impl->n_draft_ctx_zero++; // PXA_SPEC_STAGE_DIAG_v1
        }

        if (result.empty()) {
            impl->n_draft_empty++; // PXA_SPEC_STAGE_DIAG_v1
            continue;
        }

        if (common_speculative_type_is_self_spec(impl->type) && impl_params.n_min > 0 && (int)result.size() < impl_params.n_min) {
            LOG_DBG("%s: impl %s drafted %zu tokens, below fallback threshold %d - trying next implementation\n",
                    __func__, common_speculative_type_to_str(impl->type).c_str(), result.size(), impl_params.n_min);
            impl->n_draft_below_floor++; // PXA_SPEC_STAGE_DIAG_v1
            result.clear();
            continue;
        }
        LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                impl.get()->n_call_draft, result.size());

        spec->curr_impl = impl.get();
        impl->n_gen_drafts++;
        impl->n_gen_tokens += result.size();

        break; // We have a draft, so break out of the loop and return it.
    }

    // PXA_SPEC_CHAIN_ACCEPT_v1: the loop above stops at the winner, so stages after it were
    // never invoked this step. Remember how far it got so the accept path can tell a stage that
    // drafted and lost apart from one that was never asked.
    spec->n_asked_by_seq[draft_seq_id] = n_asked;

    // store draft count for tuner feedback
    if (spec->tuner && spec->tuner->enabled) {
        spec->last_n_drafted = (int)result.size();
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted, llama_seq_id seq_id) {
    if (spec->tuner && spec->tuner->enabled && spec->t_step_start_us > 0) {
        int64_t step_time_us = ggml_time_us() - spec->t_step_start_us;
        double step_tps = (step_time_us > 100)
            ? (n_accepted + 1.0) * 1e6 / (double)step_time_us
            : 0.0;
        spec->tuner->accept_feedback(n_accepted, spec->last_n_drafted, step_tps);
        spec->t_step_start_us = 0;
    }

    common_speculative_state * impl = spec->curr_impl;

    if (!impl) {
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }

    // PXA_SPEC_CHAIN_ACCEPT_v1 (2026-09-15): tell every OTHER stage the step
    // happened. No statistics counter moves here -- n_acc_drafts, n_acc_tokens and n_call_accept
    // stay the winner's alone, exactly as they read today -- only per-stage learned state may
    // react, and only a stage that was actually asked hears "drafted". The draft loop stops at
    // the winner, so the stages positioned AFTER it never ran; passing them "drafted" charged
    // every such step to the `untested` counter, whose whole purpose is to separate "this stage
    // drafted and lost" from "this stage was never asked". n_asked_by_seq records how far the
    // loop got on THIS sequence.
    size_t n_asked = spec->impls.size();
    {
        const auto it = spec->n_asked_by_seq.find(seq_id);
        if (it != spec->n_asked_by_seq.end()) {
            n_asked = it->second;
        }
    }

    for (size_t j = 0; j < spec->impls.size(); ++j) {
        auto & other = spec->impls[j];
        if (other.get() == impl) {
            continue;
        }
        other->n_observe++;
        other->observe_step(n_accepted, /* drafted = */ j < n_asked, seq_id);
    }

    if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
        if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
            mtp_invalidate_cached_drafts(*mtp_state);
        }
    }
}

// PXA_SPEC_MTP_LAZY_v1: the server needs two facts it cannot see from the outside -- whether this
// chain carries an MTP stage whose companion has not been built yet (so it must NOT charge the
// target the embedding output or the per-step hidden/commit bookkeeping), and whether that stage
// has since gone live (so it must start charging them).
bool common_speculative_mtp_is_lazy_pending(const common_speculative * spec) {
    const auto * st = common_speculative_get_mtp_state(spec);
    return st != nullptr && st->lazy_pending;
}

bool common_speculative_mtp_is_live(const common_speculative * spec) {
    const auto * st = common_speculative_get_mtp_state(spec);
    return st != nullptr && st->ctx_mtp != nullptr;
}

static bool common_speculative_has_type(const common_speculative * spec, common_speculative_type type) {
    if (spec == nullptr) {
        return false;
    }

    return std::any_of(spec->configs.begin(), spec->configs.end(), [type](const common_speculative_config & config) {
        return config.type == type;
    });
}

static int common_speculative_ctx_mtp_n_embd(llama_context * ctx) {
    return ctx ? (int) llama_mtp_state_n_embd(ctx) : 0;
}

static bool common_speculative_batch_token_has_seq_id(
        const llama_batch & batch,
        int token_index,
        llama_seq_id seq_id) {
    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[token_index] <= 0 || batch.seq_id[token_index] == nullptr) {
        return false;
    }

    for (int i = 0; i < batch.n_seq_id[token_index]; ++i) {
        if (batch.seq_id[token_index][i] == seq_id) {
            return true;
        }
    }

    return false;
}

static bool common_speculative_batch_is_exact_single_seq(
        const llama_batch & batch,
        llama_seq_id seq_id) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return false;
        }
    }

    return true;
}

static int common_speculative_copy_seq_batch(
        const llama_batch & batch,
        llama_seq_id seq_id,
        llama_batch & seq_batch) {
    if (batch.token == nullptr || batch.pos == nullptr) {
        return -1;
    }

    if (batch.n_tokens < 1) {
        return 0;
    }

    std::vector<int> token_indices;
    token_indices.reserve(batch.n_tokens);
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (common_speculative_batch_token_has_seq_id(batch, i, seq_id)) {
            token_indices.push_back(i);
        }
    }

    if (token_indices.empty()) {
        return 0;
    }

    seq_batch = llama_batch_init((int) token_indices.size(), 0, 1);
    for (const int i : token_indices) {
        common_batch_add(seq_batch, batch.token[i], batch.pos[i], { seq_id }, batch.logits != nullptr && batch.logits[i]);
    }

    return (int) token_indices.size();
}

static bool common_speculative_feature_view_copy_batch_rows(
        const common_speculative_feature_view & view,
        const llama_batch & batch,
        llama_seq_id seq_id,
        std::vector<float> * hidden_rows) {
    if (hidden_rows == nullptr || view.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || view.width <= 0 || batch.n_tokens <= 0 || batch.pos == nullptr) {
        return false;
    }

    std::unordered_map<llama_pos, const float *> rows_by_pos;
    rows_by_pos.reserve(view.rows.size());
    for (const auto & row : view.rows) {
        if (row.seq_id == seq_id && row.data != nullptr) {
            rows_by_pos[row.pos] = row.data;
        }
    }

    hidden_rows->clear();
    hidden_rows->reserve((size_t) batch.n_tokens * view.width);
    for (int i = 0; i < batch.n_tokens; ++i) {
        auto it = rows_by_pos.find(batch.pos[i]);
        if (it == rows_by_pos.end()) {
            hidden_rows->clear();
            return false;
        }

        hidden_rows->insert(hidden_rows->end(), it->second, it->second + view.width);
    }

    return hidden_rows->size() == (size_t) batch.n_tokens * view.width;
}

static bool common_speculative_capture_target_features(
        common_speculative * spec,
        const common_speculative_feature_view & features);

static bool common_speculative_feature_view_from_hidden_rows(
        const std::vector<float> & hidden_rows,
        int32_t width,
        llama_seq_id seq_id,
        llama_pos pos_base,
        common_speculative_feature_view & view) {
    view = {};
    view.kind = COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE;
    view.width = width;

    if (width <= 0 || hidden_rows.empty() || hidden_rows.size() % (size_t) width != 0) {
        return false;
    }

    const size_t n_rows = hidden_rows.size() / (size_t) width;
    view.rows.reserve(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        view.rows.push_back({
            /* .seq_id = */ seq_id,
            /* .pos    = */ pos_base + (llama_pos) i,
            /* .data   = */ hidden_rows.data() + i * (size_t) width,
        });
    }

    return true;
}

static bool common_speculative_collect_target_batch_features(
        const common_speculative * spec,
        llama_context * ctx,
        const llama_batch & batch,
        common_speculative_feature_view & features) {
    features = {};
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    if (!llama_spec_get_hidden_feature_view(ctx, batch, features)) {
        return false;
    }

    return true;
}

static bool common_speculative_collect_target_seq_batch_features(
        const common_speculative * spec,
        llama_context * ctx,
        const llama_batch & batch,
        llama_seq_id seq_id,
        common_speculative_feature_view & features) {
    features = {};
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    if (!llama_spec_get_hidden_feature_view_for_seq(ctx, batch, seq_id, features)) {
        return false;
    }

    return true;
}

bool common_speculative_capture_output_hidden(
        common_speculative * spec,
        llama_context * ctx,
        int32_t output_index,
        llama_seq_id seq_id,
        llama_pos pos) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    // PXA_MTP_HIDDEN_BY_BATCH_ROW_v1: `output_index` is, and always was, a RAW BATCH ROW at every
    // caller (the server passes slot.i_batch - i, and ensure_sequence_hidden passes -1 for "the last
    // row produced"). It is read as one now, and refused when the target's embedding buffer is not
    // dense by batch row, instead of silently answering from another token's row.
    common_speculative_feature_view features;
    if (!llama_spec_get_hidden_feature_view_from_batch_row(ctx, output_index, seq_id, pos, features)) {
        return false;
    }

    return common_speculative_capture_target_features(spec, features);
}

bool common_speculative_ensure_sequence_hidden(
        common_speculative * spec,
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos pos) {
    pxa_mtp_prefetch_wait_seq(seq_id); // PXA_MTP_PREFETCH: drain in-flight companion commit before touching ctx_mtp
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP) || common_speculative_has_sequence_hidden(spec, seq_id)) {
        return true;
    }

    return common_speculative_capture_output_hidden(spec, ctx, -1, seq_id, pos);
}

int32_t common_speculative_on_target_seq_batch(
        common_speculative * spec,
        llama_context * ctx_tgt,
        const llama_batch & batch,
        llama_seq_id seq_id,
        bool is_prompt_warmup) {
    pxa_mtp_prefetch_wait_seq(seq_id); // PXA_MTP_PREFETCH: drain in-flight companion commit before touching ctx_mtp
    llama_context * ctx_mtp = common_speculative_get_companion_ctx(spec);
    ctx_mtp = ctx_mtp ? ctx_mtp : ctx_tgt;
    if (ctx_tgt == nullptr || ctx_mtp == nullptr || batch.n_tokens <= 0) {
        return 0;
    }

    // PXA_MTP_LAZY_WARMUP_v1 (2026-07-15): skip the prompt-time companion warmup entirely.
    // The warmup runs feature-row collection (all-token hidden D2H), an H2D copy into the
    // companion, and a companion-layer decode for EVERY prompt batch — measured ~40% of 35B
    // prefill (850 vs 1300+ t/s at 12k cold). The companion KV is an accelerator cache only
    // (see common_speculative_context_shift: dropped wholesale on ctx-shift and rebuilt
    // forward via MTP_OP_UPDATE_ACCEPTED — correctness unaffected, draft quality dips
    // briefly). Lazy mode starts every sequence in that post-shift state: the drafter
    // self-seeds via common_speculative_ensure_sequence_hidden() before the first draft
    // (the server draft path already calls it) and the companion KV fills forward from
    // accepts. Trade: early-draft acceptance dips (PXA_MTP_ADAPTIVE_K shrinks depth
    // automatically) in exchange for full-speed prefill. The level is resolved by
    // llama_pxa_mtp_lazy_warmup() (include/llama.h) — an explicit PXA_MTP_LAZY_WARMUP wins,
    // else PXA_ENHANCE=1 turns it on while PXA_REFERENCE=1 and the plain default leave it
    // off — so this site cannot disagree with the decode-time output reserve or the graph
    // out_ids row-slice. Gates BOTH prompt-warmup callers (text prefill + media warmup); the
    // checkpoint-restore resync uses is_prompt_warmup=false and is untouched.
    if (llama_pxa_mtp_lazy_warmup() && is_prompt_warmup) {
        return 0;
    }

    const int n_embd_src = common_speculative_ctx_mtp_n_embd(ctx_tgt);
    const int n_embd_dst = common_speculative_ctx_mtp_n_embd(ctx_mtp);
    if (n_embd_src <= 0 || n_embd_dst <= 0) {
        return -1;
    }

    if (n_embd_src != n_embd_dst) {
        LOG_ERR("MTP warmup hidden state width mismatch: n_embd_src = %d, n_embd_dst = %d\n", n_embd_src, n_embd_dst);
        return -1;
    }

    common_speculative_feature_view feature_view;
    const llama_batch * batch_for_spec = &batch;
    llama_batch seq_batch = {};
    // PXA_MTP_NP_CTXMTP_SYNC_v1: split a multi-seq batch down to this seq for the GENERATION sync too
    // (is_prompt_warmup=false), not only at prompt warmup. The np>1 non-spec-decode ctx_mtp sync (in
    // server process_batch_tokens) passes the shared multi-seq batch_view with is_prompt_warmup=false;
    // without splitting, on_target_batch's single-seq requirement rejects it and ctx_mtp never advances.
    // The for-seq feature view (collect_target_seq_batch_features) reads the correct embd row per token.
    // Only is_prompt_warmup=true callers existed before, and for those !single_seq is unchanged, so this
    // is behavior-preserving for the prompt-warmup path.
    const bool needs_seq_split = !common_speculative_batch_is_exact_single_seq(batch, seq_id);

    if (needs_seq_split) {
        const int n_seq_tokens = common_speculative_copy_seq_batch(batch, seq_id, seq_batch);
        if (n_seq_tokens <= 0) {
            return n_seq_tokens < 0 ? -1 : 0;
        }

        if (!common_speculative_collect_target_seq_batch_features(spec, ctx_tgt, batch, seq_id, feature_view)) {
            llama_batch_free(seq_batch);
            return -1;
        }

        batch_for_spec = &seq_batch;
    } else {
        if (!common_speculative_collect_target_batch_features(spec, ctx_tgt, batch, feature_view)) {
            return -1;
        }
    }

    const int32_t ret = common_speculative_on_target_batch(spec, *batch_for_spec, feature_view, is_prompt_warmup);
    if (needs_seq_split) {
        llama_batch_free(seq_batch);
    }

    return ret;
}

bool common_speculative_copy_output_hidden_rows(
        const common_speculative * spec,
        llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows) {
    hidden_rows.clear();
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    // PXA_MTP_HIDDEN_BY_BATCH_ROW_v1: these are the verify batch's own row indices (slot.i_batch_dft,
    // or {0..n-1} of a checkpoint re-decode), i.e. RAW BATCH ROWS.
    return llama_spec_copy_hidden_rows_from_batch_rows(ctx, output_indices, hidden_rows);
}

// PXA_MTP_KVPOS_v1 (2026-09-09): the tokens an accepted verify step commits to the
// MTP head's K/V cache, paired row-for-row with the target hidden rows captured for that step.
//
// Hidden row i is h_{pos_base+i} (the target's output at verify row i, whose token sits at position
// pos_base+i), and the MTP row it belongs to is (h_{q-1}, x_q) at q = pos_base+i+1 -- so row i takes
// ids[i], the token the target produced at that verify row, and is written at
// pxa_mtp_commit_pos(pos_base, i). One shape for every stage type.
//
// BEFORE this fix the two stage types disagreed, and BOTH were off by one, differently:
//   * MTP-drafted steps used ids (the right pairing) but wrote them at pos_base+i -- one position
//     LOW, which is the defect measured 2026-09-08;
//   * steps drafted by another stage of a composite chain (ngram + mtp) shifted the TOKENS instead
//     -- [sampled_before, ids[0..n-2]] at pos_base+i -- writing (h_q, x_q) at q: the position was
//     right but the pair was the unshifted one, exactly the "one-token conditioning skew" the
//     prompt warm-up shifts its hidden rows to avoid. Its last row then seeded the cached free
//     draft token with a prediction of a token the sequence already had.
// `sampled_before` is retained in the signature (and in the request struct the batched path fills)
// because it is the caller's own record of the step; the K/V pairing no longer needs it.
static bool common_speculative_build_commit_tokens(
        common_speculative_type spec_type_used,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        std::vector<llama_token> & commit_tokens) {
    GGML_UNUSED(spec_type_used);
    GGML_UNUSED(sampled_before);

    commit_tokens.clear();
    if (ids.empty()) {
        return true;
    }

    commit_tokens = ids;
    return commit_tokens.size() == ids.size();
}

static bool common_speculative_apply_hidden_rows(
        common_speculative * spec,
        llama_seq_id seq_id,
        llama_pos pos_base,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || ids.empty()) {
        return true;
    }

    const size_t expected_floats = ids.size() * (size_t) mtp_state->n_embd;
    if (mtp_state->n_embd <= 0 || hidden_rows.size() != expected_floats) {
        return false;
    }

    // PXA_MTP_KVPOS_v1: row i holds (hidden row i, ids[i]) and belongs at ids[i]'s OWN position,
    // pos_base + 1 + i -- not at pos_base + i, which put every committed row one position low and
    // left the newest position empty for the next draft to attend across. The feature view is keyed
    // by position (common_speculative_feature_view_copy_batch_rows looks each batch row up by
    // batch.pos[i]), so it is built at the same base or the copy finds nothing.
    const llama_pos commit_pos0 = pxa_mtp_commit_pos0(pos_base);

    llama_batch accepted_batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(accepted_batch, ids[i], pxa_mtp_commit_pos(pos_base, (int32_t) i), { seq_id }, true);
    }

    common_speculative_feature_view feature_view;
    const bool have_feature_view = common_speculative_feature_view_from_hidden_rows(
        hidden_rows, mtp_state->n_embd, seq_id, commit_pos0, feature_view);
    const int32_t ret = have_feature_view
        ? common_speculative_on_target_batch(spec, accepted_batch, feature_view, false)
        : -1;

    llama_batch_free(accepted_batch);
    return ret == 0;
}

bool common_speculative_commit_accepted_hidden_rows(
        common_speculative * spec,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP) || ids.empty()) {
        return true;
    }

    std::vector<llama_token> commit_tokens;
    if (!common_speculative_build_commit_tokens(spec_type_used, sampled_before, ids, commit_tokens)) {
        return false;
    }

    return common_speculative_apply_hidden_rows(spec, seq_id, pos_base, commit_tokens, hidden_rows);
}

bool common_speculative_commit_accepted_output(
        common_speculative * spec,
        llama_context * ctx,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<int32_t> & output_indices) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP) || ids.empty()) {
        return true;
    }

    std::vector<float> hidden_rows;
    if (!common_speculative_copy_output_hidden_rows(spec, ctx, output_indices, hidden_rows)) {
        return false;
    }

    return common_speculative_commit_accepted_hidden_rows(
        spec,
        spec_type_used,
        seq_id,
        pos_base,
        sampled_before,
        ids,
        hidden_rows);
}

void common_speculative_print_stats(const common_speculative * spec, double slot_tps, int n_decoded, int n_past, common_params_speculative * active_params) {
    pxa_mtp_stats_dump("final"); // PXA_MTP_STATS: end-of-request roll-up (no-op when the counters are off)
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        // PXA_SPEC_STAGE_DIAG_v1: the four resolution counters ride on the SAME line, after the
        // shipped ones, so no existing reader of this line breaks and no existing number changes.
        const std::string str_diag = impl->extra_stats();
        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s | empty = %zu, below floor = %zu, ctx0 = %zu, observed = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str(),
                impl->n_draft_empty,
                impl->n_draft_below_floor,
                impl->n_draft_ctx_zero,
                impl->n_observe,
                str_diag.c_str());
    }

    if (spec->tuner && spec->tuner->enabled && slot_tps > 0.0 && n_decoded > 0) {
        auto * mutable_spec = const_cast<common_speculative *>(spec);
        if (active_params) {
            mutable_spec->tuner->end_of_request(slot_tps, n_past, *active_params);
        } else {
            common_params_speculative tmp_params;
            mutable_spec->tuner->end_of_request(slot_tps, n_past, tmp_params);
        }
    }
}

// ----------------------------------------------------------------------------
// MTP
// ----------------------------------------------------------------------------

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec) {
    if (!spec) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
            continue;
        }

        if (auto * mtp_state = dynamic_cast<common_speculative_state_mtp *>(impl.get())) {
            return mtp_state;
        }
    }

    return nullptr;
}

static const common_speculative_state_mtp * common_speculative_get_mtp_state(const common_speculative * spec) {
    return common_speculative_get_mtp_state(const_cast<common_speculative *>(spec));
}

static mtp_last_embd & mtp_get_last_embd(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto & last = state.draft_cache_by_seq[seq_id];
    if ((int) last.embd.size() != state.n_embd) {
        last.embd.resize(state.n_embd);
    }
    return last;
}

// PXA_SPEC_SAMPLED: the request sampler to draw this seq's draft tokens with, or nullptr when the
// lever is off or the server never registered one (then the drafter keeps its argmax and the
// verifier keeps exact matching -- the baseline path, byte for byte).
static common_sampler * pxa_spec_req_sampler(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    if (!common_sampler_spec_sampled_active()) {
        return nullptr;
    }
    const auto it = state.pxa_req_smpl_by_seq.find(seq_id);
    return it == state.pxa_req_smpl_by_seq.end() ? nullptr : it->second;
}

// Draw one draft token: sampled from q when the lever allows it for this request, argmax otherwise.
// `q_out` is filled only in the sampled case, and a refusal anywhere falls back cleanly.
static llama_token pxa_spec_draw_draft(common_speculative_state_mtp & state, llama_seq_id seq_id,
        llama_context * ctx, int idx, float * out_prob, std::vector<pxa_spec_cand> & q_out) {
    q_out.clear();
    if (common_sampler * rs = pxa_spec_req_sampler(state, seq_id)) {
        const llama_token id = common_sampler_draft_sample_dist(rs, ctx, idx, q_out, out_prob);
        if (id >= 0) {
            return id;
        }
        q_out.clear();
    }
    return common_sampler_sample_speculative(nullptr, ctx, idx, out_prob, pxa_mtp_pmin_topk());
}

static void mtp_invalidate_cached_draft(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto it = state.draft_cache_by_seq.find(seq_id);
    if (it == state.draft_cache_by_seq.end()) {
        return;
    }

    it->second.last_id = -1;
    it->second.prob = 0.0f;
    it->second.q.clear();
}

static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state) {
    for (auto & entry : state.draft_cache_by_seq) {
        entry.second.last_id = -1;
        entry.second.prob = 0.0f;
        entry.second.q.clear();
    }
}

static void mtp_store_target_hidden(
        common_speculative_state_mtp & state,
        llama_seq_id seq_id,
        const float * hidden,
        int32_t width) {
    if (hidden == nullptr || width <= 0) {
        return;
    }

    auto & stored = state.target_hidden_by_seq[seq_id];
    stored.assign(hidden, hidden + width);
}

// PXA port (upstream f5e5753c #1987): qwen35/qwen35moe MTP heads are conditioned on the
// PREVIOUS token's target hidden state (recurrent conditioning). During eager prompt warmup
// the companion KV must be built from hidden rows shifted right by one so warmup matches how
// drafting seeds (target_hidden_by_seq holds h_last -> predicts the token after last). Gate
// kept to our recurrent hybrid arch; upstream 0d59973e later universalised this for the
// non-recurrent GLM path, which we do not run, so it is deliberately NOT universalised here.
static bool mtp_model_uses_recurrent_conditioning(const common_speculative_state_mtp & state) {
    if (state.ctx_mtp == nullptr) {
        return false;
    }

    const llama_model * model = llama_get_model(state.ctx_mtp);
    if (!llama_model_has_recurrent(model)) {
        return false;
    }

    std::string arch{llama_model_arch_string(model)};
    return arch == "qwen35" || arch == "qwen35moe" || arch == "qwen4exp";
}

static void mtp_clear_target_hidden(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    state.target_hidden_by_seq.erase(seq_id);
    state.draft_cache_by_seq.erase(seq_id);
}

static bool common_speculative_capture_target_features(common_speculative * spec, const common_speculative_feature_view & features) {
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width <= 0) {
        return false;
    }

    bool captured = false;
    for (const auto & row : features.rows) {
        if (row.data == nullptr) {
            continue;
        }

        mtp_store_target_hidden(*mtp_state, row.seq_id, row.data, features.width);
        mtp_invalidate_cached_draft(*mtp_state, row.seq_id);
        captured = true;
    }

    return captured;
}

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id) {
    const auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return false;
    }

    auto it = mtp_state->target_hidden_by_seq.find(seq_id);
    return it != mtp_state->target_hidden_by_seq.end() && !it->second.empty();
}

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id) {
    pxa_mtp_prefetch_wait_seq(seq_id); // PXA_MTP_PREFETCH: drain in-flight companion commit before touching ctx_mtp
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return;
    }

    mtp_clear_target_hidden(*mtp_state, seq_id);
}

llama_context * common_speculative_get_companion_ctx(common_speculative * spec) {
    if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
        return mtp_state->ctx_mtp;
    }

    return nullptr;
}

// PXA_SPEC_SEQ_STATE_v1 (2026-09-13) -- the per-sequence drafter carry as bytes; see speculative.h.
//
// Layout, little-endian, fixed width so a blob written by one build is readable by another with the
// same embedding width (and rejected by one with a different width, which is the point of storing
// it): magic, version, n_embd, then the target hidden row and then the last-draft cache, each
// preceded by its own present flag and length. Nothing here is a pointer or a container header, so
// the blob is safe to write to disk and read back in another process.
namespace {
constexpr uint32_t PXA_SPEC_SEQ_STATE_MAGIC   = 0x31535850u; // "PXS1"
constexpr uint32_t PXA_SPEC_SEQ_STATE_VERSION = 1u;

template <typename T> void pxa_spec_put(std::vector<uint8_t> & out, const T & v) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T> bool pxa_spec_get(const uint8_t * data, size_t size, size_t & off, T & v) {
    if (off + sizeof(T) > size) {
        return false;
    }
    std::memcpy(&v, data + off, sizeof(T));
    off += sizeof(T);
    return true;
}
} // namespace

bool common_speculative_get_seq_state(const common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    data.clear();

    const auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return false;
    }

    const auto it_hidden = mtp_state->target_hidden_by_seq.find(seq_id);
    const auto it_draft  = mtp_state->draft_cache_by_seq.find(seq_id);

    const bool have_hidden = it_hidden != mtp_state->target_hidden_by_seq.end() && !it_hidden->second.empty();
    const bool have_draft  = it_draft  != mtp_state->draft_cache_by_seq.end()   && !it_draft->second.embd.empty();

    if (!have_hidden && !have_draft) {
        return false;
    }

    pxa_spec_put(data, PXA_SPEC_SEQ_STATE_MAGIC);
    pxa_spec_put(data, PXA_SPEC_SEQ_STATE_VERSION);
    pxa_spec_put(data, (int32_t) mtp_state->n_embd);

    pxa_spec_put(data, (uint32_t) (have_hidden ? 1 : 0));
    if (have_hidden) {
        pxa_spec_put(data, (uint64_t) it_hidden->second.size());
        const uint8_t * p = reinterpret_cast<const uint8_t *>(it_hidden->second.data());
        data.insert(data.end(), p, p + it_hidden->second.size() * sizeof(float));
    }

    pxa_spec_put(data, (uint32_t) (have_draft ? 1 : 0));
    if (have_draft) {
        pxa_spec_put(data, (uint64_t) it_draft->second.embd.size());
        const uint8_t * p = reinterpret_cast<const uint8_t *>(it_draft->second.embd.data());
        data.insert(data.end(), p, p + it_draft->second.embd.size() * sizeof(float));
        pxa_spec_put(data, it_draft->second.prob);
        pxa_spec_put(data, (int32_t) it_draft->second.last_id);
    }

    return true;
}

bool common_speculative_set_seq_state(common_speculative * spec, llama_seq_id seq_id, const uint8_t * data, size_t size) {
    pxa_mtp_prefetch_wait_seq(seq_id); // PXA_MTP_PREFETCH: drain in-flight companion commit first

    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return false;
    }

    // A refused blob must leave the sequence with NO carry rather than the previous occupant's.
    mtp_clear_target_hidden(*mtp_state, seq_id);

    if (data == nullptr || size == 0) {
        return false;
    }

    size_t   off = 0;
    uint32_t magic = 0, version = 0, have = 0;
    int32_t  n_embd = 0;

    if (!pxa_spec_get(data, size, off, magic) || magic != PXA_SPEC_SEQ_STATE_MAGIC) {
        LOG_WRN("%s: refusing a speculative carry with the wrong magic\n", __func__);
        return false;
    }
    if (!pxa_spec_get(data, size, off, version) || version != PXA_SPEC_SEQ_STATE_VERSION) {
        LOG_WRN("%s: refusing a speculative carry of version %u\n", __func__, version);
        return false;
    }
    if (!pxa_spec_get(data, size, off, n_embd) || n_embd != mtp_state->n_embd) {
        LOG_WRN("%s: refusing a speculative carry for width %d (this model is %d)\n",
                __func__, (int) n_embd, mtp_state->n_embd);
        return false;
    }

    std::vector<float> hidden;
    mtp_last_embd      draft;
    bool               have_draft = false;

    if (!pxa_spec_get(data, size, off, have)) {
        return false;
    }
    if (have) {
        uint64_t n = 0;
        if (!pxa_spec_get(data, size, off, n) || n == 0 || n > (uint64_t) INT32_MAX ||
                off + n * sizeof(float) > size) {
            LOG_WRN("%s: truncated hidden row in a speculative carry\n", __func__);
            return false;
        }
        hidden.resize((size_t) n);
        std::memcpy(hidden.data(), data + off, (size_t) n * sizeof(float));
        off += (size_t) n * sizeof(float);
        if ((int32_t) hidden.size() != n_embd) {
            LOG_WRN("%s: hidden row of %d floats against a width of %d\n",
                    __func__, (int) hidden.size(), (int) n_embd);
            return false;
        }
    }

    if (!pxa_spec_get(data, size, off, have)) {
        return false;
    }
    if (have) {
        uint64_t n = 0;
        if (!pxa_spec_get(data, size, off, n) || n == 0 || n > (uint64_t) INT32_MAX ||
                off + n * sizeof(float) > size) {
            LOG_WRN("%s: truncated draft cache in a speculative carry\n", __func__);
            return false;
        }
        draft.embd.resize((size_t) n);
        std::memcpy(draft.embd.data(), data + off, (size_t) n * sizeof(float));
        off += (size_t) n * sizeof(float);
        int32_t last_id = -1;
        if (!pxa_spec_get(data, size, off, draft.prob) || !pxa_spec_get(data, size, off, last_id)) {
            LOG_WRN("%s: truncated draft cache tail in a speculative carry\n", __func__);
            return false;
        }
        draft.last_id = (int) last_id;
        have_draft = true;
    }

    if (off != size) {
        LOG_WRN("%s: %zu trailing bytes in a speculative carry\n", __func__, size - off);
        return false;
    }

    if (!hidden.empty()) {
        mtp_state->target_hidden_by_seq[seq_id] = std::move(hidden);
    }
    if (have_draft) {
        mtp_state->draft_cache_by_seq[seq_id] = std::move(draft);
    }

    return true;
}

static int pxa_mtp_readback_check_level() {
    static const int v = getenv("PXA_MTP_READBACK_CHECK") ? atoi(getenv("PXA_MTP_READBACK_CHECK")) : 0;
    return v;
}

// PXA_MTP_READBACK_CHECK: see the call site in mtp_accept_batch. Debug only; it re-decodes the
// commit batch's LAST row on its own and reports how the read-back hidden differs. It rewrites the
// same cell with the same content, restores `last` and the companion's draft-input buffer, so with
// the lever off (the default) nothing here runs and with it on the flow is unchanged apart from
// two extra 1-row decodes and the printout.
static void pxa_mtp_readback_check(
        common_speculative_state_mtp & state,
        const llama_batch & accepted_batch,
        llama_seq_id seq_id,
        const float * hidden_rows,
        mtp_last_embd & last) {
    const int level = pxa_mtp_readback_check_level();
    if (level <= 0 || accepted_batch.n_tokens <= 0 || hidden_rows == nullptr) {
        return;
    }

    llama_context * ctx = state.ctx_mtp;
    const int n_embd = state.n_embd;
    const int n = accepted_batch.n_tokens;
    const llama_seq_id kv_seq = llama_n_seq_max(ctx) <= 1 ? 0 : seq_id;

    const llama_token   tok_last = accepted_batch.token[n - 1];
    const llama_pos     pos_last = accepted_batch.pos[n - 1];
    const float * const h_last   = hidden_rows + (size_t) (n - 1) * n_embd;

    std::vector<float> h_commit(last.embd.begin(), last.embd.end());
    const llama_token tok_commit = last.last_id;
    const float       p_commit   = last.prob;

    // Capture the WHOLE companion embedding buffer as the commit decode left it, so the row the
    // read-back actually returned can be identified among the rows the decode produced.
    std::vector<float> embd_all;
    if (const float * base = llama_get_embeddings(ctx); base != nullptr) {
        embd_all.assign(base, base + (size_t) n * n_embd);
    }

    auto stats = [&](const std::vector<float> & a, const std::vector<float> & b,
                     double & max_abs, double & cosine, double & l2a, double & l2b) {
        double sa = 0, sb = 0, dot = 0; max_abs = 0;
        for (int i = 0; i < n_embd; ++i) {
            const double x = a[i], y = b[i];
            sa += x * x; sb += y * y; dot += x * y;
            const double d = x - y < 0 ? y - x : x - y;
            if (d > max_abs) max_abs = d;
        }
        l2a = sqrt(sa); l2b = sqrt(sb);
        cosine = (sa > 0 && sb > 0) ? dot / (sqrt(sa) * sqrt(sb)) : 0.0;
    };

    // Re-run the last row on its own, once per op type under test.
    auto rerun = [&](llama_mtp_op_type op, std::vector<float> & h_out, llama_token & tok_out, float & p_out) -> bool {
        if (llama_kv_cache_seq_pos_max(ctx, kv_seq) >= pos_last) {
            llama_kv_cache_seq_rm(ctx, kv_seq, pos_last, -1);
        }
        if (!llama_set_draft_input_hidden_state_copy(ctx, h_last, (size_t) n_embd)) {
            return false;
        }
        llama_batch b1 = llama_batch_init(1, 0, 1);
        common_batch_add(b1, tok_last, pos_last, { kv_seq }, true);
        llama_set_mtp_op_type(ctx, op);
        const int32_t rc = llama_decode(ctx, b1);
        llama_set_mtp_op_type(ctx, MTP_OP_NONE);
        llama_batch_free(b1);
        if (rc != 0) {
            return false;
        }
        const float * e = llama_get_embeddings_ith(ctx, 0);
        if (e == nullptr) {
            return false;
        }
        h_out.assign(e, e + n_embd);
        if (!llama_set_draft_input_hidden_state_copy(ctx, h_out.data(), h_out.size())) {
            return false;
        }
        tok_out = common_sampler_sample_speculative(nullptr, ctx, 0, &p_out, pxa_mtp_pmin_topk());
        return true;
    };

    std::vector<float> h_draft;
    llama_token tok_draft = -1;
    float       p_draft   = 0.0f;
    const bool  ok_draft  = rerun(MTP_OP_DRAFT_GEN, h_draft, tok_draft, p_draft);

    if (ok_draft) {
        double max_abs, cosine, l2c, l2d;
        stats(h_commit, h_draft, max_abs, cosine, l2c, l2d);
        LOG_WRN("PXA_MTP_READBACK_CHECK n=%d pos_last=%d tok_last=%d | commit vs DRAFT_GEN: "
                "max|d|=%.6f cos=%.6f L2(commit)=%.4f L2(draft)=%.4f tok %d(p=%.4f) vs %d(p=%.4f)%s\n",
                n, (int) pos_last, (int) tok_last, max_abs, cosine, l2c, l2d,
                (int) tok_commit, (double) p_commit, (int) tok_draft, (double) p_draft,
                max_abs == 0.0 ? " IDENTICAL" : " DIFFER");
        for (int r = 0; r < n && !embd_all.empty(); ++r) {
            std::vector<float> row(embd_all.begin() + (size_t) r * n_embd,
                                   embd_all.begin() + (size_t) (r + 1) * n_embd);
            double ma, cs, la, lb;
            stats(row, h_draft, ma, cs, la, lb);
            LOG_WRN("PXA_MTP_READBACK_CHECK   embd buffer row %d/%d vs DRAFT_GEN: max|d|=%.6f cos=%.6f L2=%.4f%s\n",
                    r, n, ma, cs, la, ma == 0.0 ? "   <== THIS IS THE CORRECT ROW" : "");
        }
    } else {
        LOG_WRN("PXA_MTP_READBACK_CHECK n=%d pos_last=%d: DRAFT_GEN re-run FAILED\n", n, (int) pos_last);
    }

    if (level >= 2) {
        std::vector<float> h_ua1;
        llama_token tok_ua1 = -1;
        float       p_ua1   = 0.0f;
        if (rerun(MTP_OP_UPDATE_ACCEPTED, h_ua1, tok_ua1, p_ua1)) {
            double max_abs, cosine, l2a, l2b;
            stats(h_ua1, ok_draft ? h_draft : h_commit, max_abs, cosine, l2a, l2b);
            LOG_WRN("PXA_MTP_READBACK_CHECK n=%d pos_last=%d | 1-row UPDATE_ACCEPTED vs %s: "
                    "max|d|=%.6f cos=%.6f L2=%.4f/%.4f tok %d(p=%.4f)%s\n",
                    n, (int) pos_last, ok_draft ? "DRAFT_GEN" : "commit",
                    max_abs, cosine, l2a, l2b, (int) tok_ua1, (double) p_ua1,
                    max_abs == 0.0 ? " IDENTICAL" : " DIFFER");
        } else {
            LOG_WRN("PXA_MTP_READBACK_CHECK n=%d pos_last=%d: 1-row UPDATE_ACCEPTED re-run FAILED\n",
                    n, (int) pos_last);
        }
    }

    // Restore the flow: the cached free token and its hidden are the COMMIT's, exactly as before.
    last.embd.assign(h_commit.begin(), h_commit.end());
    last.last_id = tok_commit;
    last.prob    = p_commit;
    llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size());
}

static int32_t mtp_accept_batch(
        common_speculative_state_mtp & state,
        const llama_batch & accepted_batch,
        llama_seq_id seq_id,
        const float * hidden_rows) {
    if (accepted_batch.n_tokens == 0 || hidden_rows == nullptr) {
        return 0;
    }

    const size_t hidden_rows_floats = (size_t) accepted_batch.n_tokens * state.n_embd;
    if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, hidden_rows, hidden_rows_floats)) {
        return -1;
    }
    if (mtp_update_kv_cache(state.ctx_mtp, accepted_batch, false) != 0) {
        return -1;
    }

    auto & last = mtp_get_last_embd(state, seq_id);

    // PXA_MTP_ZERO_OUTPUT_COMMIT: the decode above asked for no outputs, so there is no embedding
    // to read and no free token to sample. Leave the cache empty and the next draft re-decodes this
    // position as its own step 0 (mainline's split).
    if (common_speculative_mtp_zero_output_commit()) {
        last.last_id = -1;
        return 0;
    }

    const float * embd = llama_get_embeddings_ith(state.ctx_mtp, accepted_batch.n_tokens - 1);
    if (embd != nullptr) {
        std::memcpy(last.embd.data(), embd, last.embd.size() * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, last.embd.data(), last.embd.size())) {
            return -1;
        }
        last.last_id = pxa_spec_draw_draft(state, seq_id, state.ctx_mtp, accepted_batch.n_tokens - 1, &last.prob, last.q);

        // PXA_MTP_READBACK_CHECK (debug lever, default off): the third defect's bisect. The commit
        // decode's last row and a 1-row DRAFT_GEN decode of the SAME (hidden, token, position) over
        // the same companion K/V are, on paper, the identical computation -- so their read-back
        // hidden rows must be bit-identical. Measured on the GPU they are not: chains built on the
        // commit's row collapse (measured 2026-09-08). Re-run the last row two more ways and
        // print the deltas, so the cause is localised to (a) the multi-row batch, (b) the op type,
        // or (c) neither.  1 = compare vs DRAFT_GEN, 2 = also compare vs a 1-row UPDATE_ACCEPTED.
        pxa_mtp_readback_check(state, accepted_batch, seq_id, hidden_rows, last);
    }

    return 0;
}

// PXA_SHARED_MTP_v1: BATCHED all-accepted MTP commit across slots.
// -----------------------------------------------------------------------------
// Each accepting slot would otherwise run ONE serial llama_decode(ctx_mtp) to commit its
// accepted tokens (mtp_update_kv_cache) + sample its next free draft token. With the SHARED
// companion (one ctx_mtp, n_seq_max=np) those decodes can be collapsed into ONE batched
// multi-seq decode: every seq's accepted tokens go in the same llama_batch, each token tagged
// with its own seq_id+pos and its own hidden row (the MTP draft-input buffer holds one row per
// batch token; prepare_mtp_graph_inputs slices per token, so multi-seq is supported).
//
// This covers the COMMON no-rejection commit path only (the inputs mirror
// common_speculative_commit_accepted_hidden_rows: spec_type_used + pre-captured hidden rows).
// The rarer rejected/checkpoint-restore commits stay per-slot (untouched) — their hidden rows
// come from a per-slot ctx_tgt re-decode and are deeply coupled to the per-seq checkpoint logic.
//
// np=1 equivalence: with one entry this builds the same single-seq batch the serial path built
// and runs the identical decode + per-seq tail (seq_rm cleanup + last-embd sample), so the
// committed KV + cached free-draft token are byte-identical to the per-slot path.
// (common_speculative_commit_req is declared in speculative.h)
// Returns the set of seq_ids whose commit FAILED (caller must clear their hidden state).
std::vector<llama_seq_id> common_speculative_commit_accepted_hidden_rows_batched(
        common_speculative * spec,
        common_speculative_type spec_type_used,
        const std::vector<common_speculative_commit_req> & reqs) {
    std::vector<llama_seq_id> failed;
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || mtp_state->ctx_mtp == nullptr || reqs.empty()) {
        return failed; // PXA_SPEC_MTP_LAZY_v1: a deferred companion has nothing to commit into
    }
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return failed;
    }

    llama_context * ctx = mtp_state->ctx_mtp;
    const int n_embd = mtp_state->n_embd;
    if (n_embd <= 0) {
        for (const auto & r : reqs) failed.push_back(r.seq_id);
        return failed;
    }

    // Build the per-seq commit token lists (same transform as the serial path) and validate.
    struct prepared_seq {
        llama_seq_id seq_id;
        llama_pos pos_base;
        std::vector<llama_token> commit_tokens;
        const std::vector<float> * hidden_rows;
        int last_token_batch_idx = -1; // index of this seq's last token in the combined batch
    };
    std::vector<prepared_seq> prepared;
    prepared.reserve(reqs.size());

    size_t total_tokens = 0;
    for (const auto & r : reqs) {
        if (r.ids.empty() || r.hidden_rows == nullptr) {
            // nothing to commit for this seq -> treat as no-op success (matches serial early-out)
            continue;
        }
        std::vector<llama_token> commit_tokens;
        if (!common_speculative_build_commit_tokens(spec_type_used, r.sampled_before, r.ids, commit_tokens)) {
            failed.push_back(r.seq_id);
            continue;
        }
        const size_t expected_floats = commit_tokens.size() * (size_t) n_embd;
        if (commit_tokens.empty() || r.hidden_rows->size() != expected_floats) {
            failed.push_back(r.seq_id);
            continue;
        }
        prepared_seq ps;
        ps.seq_id        = r.seq_id;
        ps.pos_base      = r.pos_base;
        ps.commit_tokens = std::move(commit_tokens);
        ps.hidden_rows   = r.hidden_rows;
        total_tokens    += ps.commit_tokens.size();
        prepared.push_back(std::move(ps));
    }

    if (prepared.empty() || total_tokens == 0) {
        return failed;
    }

    // Per-seq KV cleanup BEFORE the batched decode (mirrors mtp_update_kv_cache's pre-decode rm):
    // drop any stale rows at/after this seq's FIRST COMMITTED position so the commit writes clean
    // cells. PXA_MTP_KVPOS_v1: that is pos_base + 1, not pos_base -- the row at pos_base holds
    // (h_{pos_base-1}, x_{pos_base}) and is the committed history this batch continues from.
    for (const auto & ps : prepared) {
        const llama_pos start_pos = pxa_mtp_commit_pos0(ps.pos_base);
        if (llama_kv_cache_seq_pos_max(ctx, ps.seq_id) >= start_pos) {
            llama_kv_cache_seq_rm(ctx, ps.seq_id, start_pos, -1);
        }
    }

    // Assemble ONE batch + ONE concatenated hidden-row buffer in batch-token order.
    llama_batch batch = llama_batch_init((int) total_tokens, 0, 1);
    std::vector<float> hidden_all;
    hidden_all.reserve(total_tokens * (size_t) n_embd);
    // PXA_MTP_ZERO_OUTPUT_COMMIT: same split as the serial path -- no row asks for an output and no
    // free token is sampled, so this batched commit is a pure K/V catch-up too.
    const bool pxa_zero_output = common_speculative_mtp_zero_output_commit();
    for (auto & ps : prepared) {
        for (size_t i = 0; i < ps.commit_tokens.size(); ++i) {
            const bool is_last = (i + 1 == ps.commit_tokens.size());
            common_batch_add(batch, ps.commit_tokens[i], pxa_mtp_commit_pos(ps.pos_base, (int32_t) i), { ps.seq_id }, is_last && !pxa_zero_output);  // PXA_MTP_KVPOS_v1
            if (is_last) {
                ps.last_token_batch_idx = batch.n_tokens - 1;
            }
        }
        const float * src = ps.hidden_rows->data();
        hidden_all.insert(hidden_all.end(), src, src + ps.commit_tokens.size() * (size_t) n_embd);
    }

    if (!llama_set_draft_input_hidden_state_copy(ctx, hidden_all.data(), hidden_all.size())) {
        llama_batch_free(batch);
        for (const auto & ps : prepared) failed.push_back(ps.seq_id);
        return failed;
    }

    llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    const int32_t ret = llama_decode(ctx, batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    if (ret != 0) {
        llama_batch_free(batch);
        for (const auto & ps : prepared) failed.push_back(ps.seq_id);
        return failed;
    }

    // Per-seq tail: read this seq's last-token embedding, cache it as the next draft input, and
    // sample the free draft token (mirrors mtp_accept_batch's tail). Each read is at this seq's
    // own last-token batch index, so the per-seq output map stays correct within the one decode.
    // PXA_MTP_ZERO_OUTPUT_COMMIT: nothing was output, so there is nothing to read -- every seq's
    // next draft starts from its own step 0.
    for (auto & ps : prepared) {
        auto & last = mtp_get_last_embd(*mtp_state, ps.seq_id);
        if (pxa_zero_output) {
            last.last_id = -1;
            continue;
        }
        const float * embd = llama_get_embeddings_ith(ctx, ps.last_token_batch_idx);
        if (embd == nullptr) {
            failed.push_back(ps.seq_id);
            continue;
        }
        std::memcpy(last.embd.data(), embd, last.embd.size() * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            failed.push_back(ps.seq_id);
            continue;
        }
        last.last_id = pxa_spec_draw_draft(*mtp_state, ps.seq_id, ctx, ps.last_token_batch_idx, &last.prob, last.q);
    }

    llama_batch_free(batch);
    return failed;
}

// PXA_MTP_PREFETCH async wrappers. Each submits the EXACT serial commit function to the worker
// (so the ctx_mtp result is bit-identical) and returns immediately, letting the caller proceed
// to process_token + the HTTP streaming write while the companion decode runs off-thread. With
// the env gate off, both fall through to the synchronous serial call unchanged.
bool common_speculative_commit_accepted_hidden_rows_async(
        common_speculative * spec,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    // Overlap applies to the MTP companion decode only; non-MTP "commit" is a cheap hidden
    // sync -> keep it serial. Off-gate -> serial.
    if (!pxa_mtp_prefetch_enabled() || spec_type_used != COMMON_SPECULATIVE_TYPE_MTP) {
        return common_speculative_commit_accepted_hidden_rows(
            spec, spec_type_used, seq_id, pos_base, sampled_before, ids, hidden_rows);
    }
    auto owned = std::make_shared<pxa_mtp_owned_req>();
    owned->seq_id = seq_id;
    owned->pos_base = pos_base;
    owned->sampled_before = sampled_before;
    owned->ids = ids;
    owned->hidden = hidden_rows;
    pxa_mtp_prefetch().submit_for_seqs({ seq_id }, [spec, spec_type_used, owned]{
        const bool ok = common_speculative_commit_accepted_hidden_rows(
            spec, spec_type_used, owned->seq_id, owned->pos_base,
            owned->sampled_before, owned->ids, owned->hidden);
        if (!ok) {
            common_speculative_clear_sequence_hidden(spec, owned->seq_id); // same fail-safe as inline
        }
    });
    return true; // optimistic; a failed commit clears its own seq hidden inside the job
}

std::vector<llama_seq_id> common_speculative_commit_accepted_hidden_rows_batched_async(
        common_speculative * spec,
        common_speculative_type spec_type_used,
        const std::vector<common_speculative_commit_req> & reqs) {
    if (!pxa_mtp_prefetch_enabled()) {
        return common_speculative_commit_accepted_hidden_rows_batched(spec, spec_type_used, reqs);
    }
    auto owned = std::make_shared<std::vector<pxa_mtp_owned_req>>();
    owned->reserve(reqs.size());
    std::vector<llama_seq_id> seqs;
    seqs.reserve(reqs.size());
    for (const auto & r : reqs) {
        pxa_mtp_owned_req o;
        o.seq_id = r.seq_id;
        o.pos_base = r.pos_base;
        o.sampled_before = r.sampled_before;
        o.ids = r.ids;
        if (r.hidden_rows) o.hidden = *r.hidden_rows;
        owned->push_back(std::move(o));
        seqs.push_back(r.seq_id);
    }
    pxa_mtp_prefetch().submit_for_seqs(seqs, [spec, spec_type_used, owned]{
        std::vector<common_speculative_commit_req> jreqs;
        jreqs.reserve(owned->size());
        for (auto & o : *owned) {
            common_speculative_commit_req r;
            r.seq_id = o.seq_id;
            r.pos_base = o.pos_base;
            r.sampled_before = o.sampled_before;
            r.ids = o.ids;
            r.hidden_rows = &o.hidden; // points into owned storage (kept alive by the shared_ptr capture)
            jreqs.push_back(std::move(r));
        }
        const std::vector<llama_seq_id> failed =
            common_speculative_commit_accepted_hidden_rows_batched(spec, spec_type_used, jreqs);
        for (llama_seq_id sid : failed) {
            common_speculative_clear_sequence_hidden(spec, sid); // same fail-safe as the serial flush
        }
    });
    return {}; // async: failures handled inside the job
}

int32_t common_speculative_on_target_batch(
        common_speculative * spec,
        const llama_batch & batch,
        const common_speculative_feature_view & features,
    bool is_prompt_warmup) {
    pxa_mtp_prefetch_wait_all(); // PXA_MTP_PREFETCH: drain in-flight companion commit before touching ctx_mtp
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || mtp_state->ctx_mtp == nullptr) {
        return 0; // PXA_SPEC_MTP_LAZY_v1: deferred companion -> nothing to warm
    }

    if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width <= 0 || batch.n_tokens <= 0) {
        return 0;
    }

    // PXA port (upstream f5e5753c #1987): guard against a target/companion hidden-width mismatch
    // before the row copy below overreads the feature buffer.
    if (features.width != mtp_state->n_embd) {
        LOG_ERR("%s: MTP feature width mismatch: got %d expected %d\n",
                __func__, features.width, mtp_state->n_embd);
        return -1;
    }

    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[0] <= 0 || batch.seq_id[0] == nullptr) {
        return -1;
    }

    const llama_seq_id seq_id = batch.seq_id[0][0];
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return -1;
        }
    }

    std::vector<float> hidden_rows_storage;
    if (!common_speculative_feature_view_copy_batch_rows(features, batch, seq_id, &hidden_rows_storage)) {
        return -1;
    }

    const float * last_hidden = hidden_rows_storage.data() + (size_t) (batch.n_tokens - 1) * features.width;
    mtp_store_target_hidden(*mtp_state, seq_id, last_hidden, features.width);

    if (mtp_state->constant_draft_positions) {
        mtp_invalidate_cached_draft(*mtp_state, seq_id);
        return 0;
    }

    if (!is_prompt_warmup) {
        return mtp_accept_batch(*mtp_state, batch, seq_id, hidden_rows_storage.data());
    }

    // PXA port (upstream f5e5753c #1987): shift hidden rows right by one for recurrent-
    // conditioned MTP (qwen35/qwen35moe) so eager warmup conditions MTP position i on the
    // previous token's target hidden state, matching the drafter's h_last seeding. This whole
    // path is skipped whenever PXA_MTP_LAZY_WARMUP=1 (returns early at the caller), so
    // it only affects an eager-warmup A/B; unshifted warmup had a one-token conditioning skew.
    const bool uses_shifted_hidden_rows = mtp_model_uses_recurrent_conditioning(*mtp_state);
    std::vector<float> previous_hidden_storage;
    if (uses_shifted_hidden_rows) {
        const auto hidden_it = mtp_state->target_hidden_by_seq.find(seq_id);
        if (hidden_it != mtp_state->target_hidden_by_seq.end() && (int32_t) hidden_it->second.size() == features.width) {
            previous_hidden_storage = hidden_it->second;
        } else {
            previous_hidden_storage.assign(features.width, 0.0f);
        }
    }

    const float * conditioned_hidden_rows = hidden_rows_storage.data();
    std::vector<float> conditioned_hidden_storage;
    if (uses_shifted_hidden_rows) {
        conditioned_hidden_storage.resize(hidden_rows_storage.size());
        std::copy(previous_hidden_storage.begin(), previous_hidden_storage.end(), conditioned_hidden_storage.begin());
        if (batch.n_tokens > 1) {
            std::copy(
                hidden_rows_storage.begin(),
                hidden_rows_storage.begin() + (size_t) (batch.n_tokens - 1) * features.width,
                conditioned_hidden_storage.begin() + features.width);
        }
        conditioned_hidden_rows = conditioned_hidden_storage.data();
    }

    if (!llama_set_draft_input_hidden_state_copy(mtp_state->ctx_mtp, conditioned_hidden_rows, hidden_rows_storage.size())) {
        return -1;
    }
    const int32_t ret = mtp_update_kv_cache(mtp_state->ctx_mtp, batch, true);
    mtp_invalidate_cached_draft(*mtp_state, seq_id);
    return ret;
}

// PXA_SPEC_SAMPLED: the server registers the request's sampler for a seq before it asks for a
// draft, so the drafter can draw from the same filter chain with the same rng. No-op with the lever
// off, and for any drafter that is not the MTP head (an n-gram stage proposes tokens with no
// distribution behind them, and those positions stay on exact matching).
// A null sampler ERASES the entry, and the server calls it that way when it releases a seq: the
// map holds a raw pointer into a request's sampler, and the server frees and re-creates that
// sampler for every new request on the slot. An entry that outlived its request would be a
// dangling pointer the next time anything on this seq asked for a draft.
void common_speculative_set_request_sampler(common_speculative * spec, llama_seq_id seq_id, common_sampler * smpl) {
    if (spec == nullptr) {
        return;
    }
    if (smpl == nullptr) {
        // The erase is NOT gated on the lever: a server that armed the lever, ran, and had it
        // read back off would otherwise keep the stale entry.
        if (auto * mtp_state = common_speculative_get_mtp_state(spec)) {
            mtp_state->pxa_req_smpl_by_seq.erase(seq_id);
        }
        return;
    }
    if (!common_sampler_spec_sampled_active()) {
        return;
    }
    if (auto * mtp_state = common_speculative_get_mtp_state(spec)) {
        mtp_state->pxa_req_smpl_by_seq[seq_id] = smpl;
    }
}

// Hand over the proposal distributions of the draft just produced for `seq_id`, and leave the store
// empty so the same q can never be read twice.
bool common_speculative_take_draft_q(common_speculative * spec, llama_seq_id seq_id,
        std::vector<std::vector<pxa_spec_cand>> & out) {
    out.clear();
    if (!common_sampler_spec_sampled_active() || spec == nullptr) {
        return false;
    }
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return false;
    }
    const auto it = mtp_state->pxa_draft_q_by_seq.find(seq_id);
    if (it == mtp_state->pxa_draft_q_by_seq.end()) {
        return false;
    }
    out = std::move(it->second);
    it->second.clear();
    return true;
}

common_speculative_type common_speculative_current_type(const common_speculative * spec) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    return spec->curr_impl->type;
}

void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past) {
    pxa_mtp_prefetch_wait_seq(seq_id); // PXA_MTP_PREFETCH: drain in-flight companion commit before touching ctx_mtp
    if (auto * ctx_mtp = common_speculative_get_companion_ctx(spec); ctx_mtp != nullptr) {
        // PXA_HYBRID_CTX_SHIFT_v3: do NOT mirror the target's shift into the companion with
        // seq_add — that marks has_shift on the MTP context, and its deferred K-shift triggers
        // a worst-case graph re-reserve that builds the full hybrid (delta-net) graph on a
        // context that only carries the nextn layer -> ggml_concat dim assert. (Pre-v2 this
        // path LOOKED fine only because the IMROPE can_shift block made the companion shift a
        // silently-swallowed no-op.) The companion KV is just an accelerator cache: drop this
        // seq's rows and let MTP_OP_UPDATE_ACCEPTED rebuild forward at post-shift positions.
        // Draft quality dips for a few steps after a shift; correctness is unaffected.
        const llama_seq_id kv_seq = llama_n_seq_max(ctx_mtp) <= 1 ? 0 : seq_id;
        llama_kv_cache_seq_rm(ctx_mtp, kv_seq, -1, -1);
        GGML_UNUSED(kv_keep); GGML_UNUSED(kv_discard); GGML_UNUSED(kv_past);
    }
    if (auto * mtp_state = common_speculative_get_mtp_state(spec)) {
        // PXA_HYBRID_CTX_SHIFT_v1: a cached cross-step draft (and its saved hidden state) was
        // produced against pre-shift positions — drop it for this seq. The drafter re-syncs on
        // the next step via the same resume path the adaptive tier-0 uses.
        mtp_invalidate_cached_draft(*mtp_state, seq_id);
    }
}

std::vector<llama_token> mtp_speculative_gen_draft(
    common_speculative_state_mtp & state,
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    llama_token id_last,
    llama_pos n_past,
    llama_seq_id seq_id,
    bool constant_draft_positions) {

    llama_tokens drafts;
    drafts.reserve(n_draft);

    // PXA_LLAMA_MTP_NP_FIX: ctx is the per-slot single-seq MTP context. Address its KV at local row 0
    // when it has only one seq slot; keep the external seq_id for the per-seq hidden/draft caches.
    const llama_seq_id kv_seq = llama_n_seq_max(ctx) <= 1 ? 0 : seq_id;

    // PXA_SPEC_SAMPLED: this seq's proposal distributions, one per drafted token and in draft order.
    // Cleared here, before any early return, so the verifier can never be handed a previous step's
    // q for this step's tokens -- the one way this rule could silently stop being lossless.
    auto & pxa_qs = state.pxa_draft_q_by_seq[seq_id];
    pxa_qs.clear();

    if (!smpl) return drafts;

    if (n_draft <= 0) {
        mtp_invalidate_cached_draft(state, seq_id);
        return drafts;
    }

    // PXA_MTP_ADAPTIVE_K: choose draft depth K in [1, n_draft] from the running acceptance EMA.
    // n_draft (the configured n_max) is the CEILING, so K never exceeds baseline -> the seq_rm/KV
    // sizing below is unchanged and this can only ever draft fewer, never more, tokens.
    if (pxa_mtp_adaptive_k_enabled() && n_draft > 1) {
        const float ema = state.pxa_ak_ema;
        int k_adapt;
        if      (ema >= 0.55f) k_adapt = n_draft;              // strong acceptance -> full depth
        else if (ema >= 0.40f) k_adapt = std::min(n_draft, 2); // middling -> shallow chain
        else                   k_adapt = 1;                    // poor -> single (cached/free) token
        if (getenv("PXA_MTP_DBG")) {
            LOG_WRN("PXA_MTP_ADAPTIVE_K: ema=%.3f n_draft=%d -> K=%d\n", (double)ema, n_draft, k_adapt);
        }
        n_draft = k_adapt;
    }

    common_sampler_reset(smpl);

    // PXA_MTP_STATS: cycle-local accounting (zero cost when the counters are off).
    auto & pxa_st = pxa_mtp_stats();
    const int64_t pxa_st_t0 = pxa_st.on ? ggml_time_us() : 0;
    int  pxa_st_stop = 2;    // 0 cached_pmin, 1 pmin, 2 nmax, 3 fail

    llama_batch mtp_batch = llama_batch_init(1, 0, 1);
    llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN);

    float prob;
    // PXA_MTP_STATS: with p_min == 0 the draft loop skips the full-vocab softmax entirely; ask for
    // the probability anyway while the counters are on, so the p distribution is observable.
    auto prob_ptr = (p_min > 0 || pxa_st.on) ? &prob : nullptr;

    llama_token current_input_id = id_last;
    llama_pos current_n_past = n_past;
    const int n_embd = llama_mtp_state_n_embd(ctx);

    auto & last = mtp_get_last_embd(state, seq_id);
    int i0 = 0;
    if (last.last_id >= 0) {
        if (last.prob < p_min) {
            n_draft = 1;
            pxa_st_stop = 0; // PXA_MTP_STATS: chain collapsed to the cached token by p_min
        }
        current_input_id = last.last_id;
        last.last_id = -1;
        drafts.push_back(current_input_id);
        pxa_qs.push_back(std::move(last.q));   // PXA_SPEC_SAMPLED: the commit drew it; its q travels with it
        last.q.clear();
        current_n_past++;
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            llama_batch_free(mtp_batch);
            llama_set_mtp_op_type(ctx, MTP_OP_NONE);
            pxa_mtp_stats_record(pxa_st, pxa_st_t0, drafts.size(), 0, 3);
            return drafts;
        }
        i0 = 1;
    }

    // PXA_MTP_KVPOS_v1: the row at n_past is committed history -- (h_{n_past-1}, x_{n_past}).
    // With a cached free token the commit has already written it and it must survive untouched;
    // without one, the first loop iteration below is about to (re)write it, so drop any stale cell
    // there (and anything above it) first, exactly as mtp_update_kv_cache does before its decode.
    // Before the commit positions were corrected nothing was ever stored at n_past, so a duplicate
    // cell for that logical position could not arise and this cleanup was not needed.
    // Gemma4 external MTP (constant_draft_positions) keeps every draft row at n_past by design and
    // is deliberately left alone.
    if (i0 == 0 && !constant_draft_positions) {
        if (llama_kv_cache_seq_pos_max(ctx, kv_seq) >= n_past) {
            llama_kv_cache_seq_rm(ctx, kv_seq, n_past, -1);
        }
    }

    int n_decode = 0;
    for (int i = i0; i < n_draft; ++i) {
        mtp_batch.n_tokens = 0;
        const llama_pos draft_pos = constant_draft_positions ? n_past : current_n_past;
        common_batch_add(mtp_batch, current_input_id, draft_pos, {kv_seq}, true);

        ++n_decode;
        if (llama_decode(ctx, mtp_batch) != 0) {
            pxa_st_stop = 3;
            break;
        }

        std::vector<pxa_spec_cand> pxa_q;   // PXA_SPEC_SAMPLED: empty unless this token was drawn from q
        llama_token id_next = pxa_spec_draw_draft(state, seq_id, ctx, 0, prob_ptr, pxa_q);
        if (pxa_st.on && prob_ptr) {
            pxa_st.p_sum += (double) prob;
            pxa_st.p_n   += 1;
            if (prob >= 0.75f) pxa_st.p_ge_075 += 1;
        }
        if (getenv("PXA_MTP_DBG")) {
            LOG_WRN("PXA_MTP_DBG step %d: in_id=%d n_past=%d -> draft=%d prob=%.3f\n",
                i, (int)current_input_id, (int)current_n_past, (int)id_next, prob_ptr ? (double)prob : -1.0);
        }

        if (i > 0 && prob_ptr && prob < p_min) {
            pxa_st_stop = 1;
            break;
        }

        drafts.push_back(id_next);
        pxa_qs.push_back(std::move(pxa_q));

        const float * emb = llama_get_embeddings_ith(ctx, 0);
        if (!emb) {
            pxa_st_stop = 3;
            break;
        }
        if (getenv("PXA_MTP_DBG")) {
            double s=0,mx=0; int nan=0;
            for (int z=0; z<n_embd; ++z){ float v=emb[z]; if(v!=v)nan++; s+=(double)v*v; double a=v<0?-v:v; if(a>mx)mx=a; }
            LOG_WRN("PXA_MTP_HID: L2=%.4f max=%.4f nan=%d n_embd=%d\n", (double)sqrt(s), mx, nan, n_embd);
        }

        // Keep a stable copy because later decode steps reuse ctx->embd storage.
        memcpy(last.embd.data(), emb, n_embd * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            pxa_st_stop = 3;
            break;
        }

        current_input_id = id_next;
        current_n_past++;

        if (prob_ptr && prob < p_min) {
            pxa_st_stop = 1;
            break;
        }
    }
    llama_batch_free(mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    // Purge the metadata for the draft tokens.
    // This prevents cache state corruption where two cells map to the same logical position.
    //
    // PXA_MTP_KVPOS_v1: the draft region starts at n_past + 1 in BOTH cases, because the row at
    // n_past is committed history either way -- with a cached free token the commit wrote it (only
    // now that commits land at the right position; before, nothing was ever stored there, which is
    // why discarding from n_past looked harmless), and without one the loop above just wrote it and
    // the original comment already said to keep it. Discarding from n_past would delete the row the
    // very next draft has to attend to, re-opening the hole this fix closes.
    // (constant_draft_positions never has a cached token, so i0 is always 0 there and the bound is
    // unchanged for Gemma4 external MTP.)
    if (n_decode > 0) {
        llama_kv_cache_seq_rm(ctx, kv_seq, pxa_mtp_draft_region_pos0(n_past), n_past + n_decode + 2);
    }

    // PXA_MTP_ADAPTIVE_K: record how many draft tokens we emitted this cycle so accept() can form
    // the accept ratio for the EMA. (Inert unless the controller is enabled.)
    if (pxa_mtp_adaptive_k_enabled()) {
        state.pxa_ak_drafted = (uint32_t) drafts.size();
    }

    pxa_mtp_stats_record(pxa_st, pxa_st_t0, drafts.size(), n_decode, pxa_st_stop);

    return drafts;
}

// PXA_MTP_BATCH_SLOTS (2026-09-09): the same draft, scheduled across
// slots instead of within one.
// -----------------------------------------------------------------------------
// mtp_speculative_gen_draft() above runs ONE sequence's chain: up to K 1-row decodes on the shared
// companion, each conditioned on the hidden row the previous one produced. At -np N the server
// calls it N times per cycle, so the companion sees N*K round trips. Mainline instead puts one row
// per drafting sequence into the SAME batch and advances them together (common/speculative.cpp:
// 1596-1746 at ggml-org/llama.cpp 304665fe7), paying K decodes of N rows.
//
// This is that regrouping and nothing else. Per sequence the token fed in, the position it is
// written at, the K/V row it lands in, the hidden row it is conditioned on, the argmax that comes
// out and the p_min ordering that decides whether the chain continues are all unchanged -- the
// per-sequence half of the loop lives in common/pxa-mtp-batch-slots.h as pxa_mtp_draft_chain and is
// replayed against a transcription of the serial loop in tests/test-mtp-batch-slots.cpp. What
// changes is that the decodes interleave: chains with different depths simply drop out of the batch
// as they stop, so a step's batch is exactly the still-running chains.
//
// It needs three things that already exist and are NOT introduced here: the SHARED companion with
// n_seq_max = n_parallel (PXA_SHARED_MTP_v1), the multi-row draft-input hidden buffer that
// prepare_mtp_graph_inputs() slices per token, and per-row read-back by batch index. The batched
// COMMIT (common_speculative_commit_accepted_hidden_rows_batched) already uses all three.
//
// Returns false when the batched path does not apply -- the caller then runs the serial
// common_speculative_draft() per slot, unchanged. On true, every request's `result` is filled
// (possibly empty, exactly as the serial call would have left it).
bool common_speculative_draft_batched(
        common_speculative * spec,
        std::vector<common_speculative_draft_req> & reqs) {

    for (auto & r : reqs) {
        r.result.clear();
    }

    // PXA_SPEC_SAMPLED: the batched drafter still draws its tokens with argmax, so every seq it
    // drafts for must be left with NO proposal distributions -- otherwise the verifier could pair
    // this step's argmax tokens with the previous step's q, and the rule's guarantee would quietly
    // stop holding. An empty store makes those positions exact-verified, which is correct.
    if (common_sampler_spec_sampled_active() && spec != nullptr) {
        if (auto * qst = common_speculative_get_mtp_state(spec)) {
            for (const auto & r : reqs) {
                qst->pxa_draft_q_by_seq[r.seq_id].clear();
            }
        }
    }

    if (!pxa_mtp_batch_slots_enabled() || spec == nullptr || reqs.size() < 2) {
        return false;
    }
    // The autotune tuner proposes per-CALL params and reads back a single last_n_drafted, so it has
    // no meaning across a batch of slots. Composite chains (ngram + mtp) fall back too: the batched
    // path is the MTP impl only, and the dispatcher's "try the next impl if this one drafted
    // nothing" rule is per-slot control flow this does not reproduce.
    if (spec->tuner && spec->tuner->enabled) {
        return false;
    }
    if (spec->impls.size() != 1) {
        return false;
    }
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || mtp_state->ctx_mtp == nullptr || mtp_state->n_embd <= 0) {
        return false;
    }
    if (spec->impls[0].get() != static_cast<common_speculative_state *>(mtp_state)) {
        return false;
    }
    // Gemma 4 external MTP holds every draft row at n_past by design, so two steps of one chain
    // already share a position; that path keeps its serial loop.
    if (mtp_state->constant_draft_positions) {
        return false;
    }

    llama_context * ctx    = mtp_state->ctx_mtp;
    const int       n_embd = mtp_state->n_embd;

    int32_t max_seq_id = -1;
    for (const auto & r : reqs) {
        if (r.draft_base_pos < 0) {
            return false;   // MTP slots always carry one; without it the position is guessed
        }
        max_seq_id = std::max(max_seq_id, (int32_t) r.seq_id);
    }
    if (!pxa_mtp_batch_slots_applicable((int32_t) reqs.size(),
                                        llama_n_seq_max(ctx),
                                        llama_n_ubatch(ctx),
                                        max_seq_id)) {
        return false;
    }

    // PXA_MTP_BATCH_SLOTS_ROWS_v1: a batched step puts one row per slot into a SINGLE
    // MTP_OP_DRAFT_GEN decode, so the companion's draft-gen graph has to size its conditioning-hidden
    // input by the batch token count. An architecture whose MTP builder still allocates one [n_embd]
    // row would concatenate it against [n_embd, n_rows] token embeddings and abort inside ggml_concat
    // while the graph is being BUILT -- earlier than prepare_mtp_graph_inputs(), whose float-count
    // check would otherwise have refused the decode cleanly. Ask the model up front and say so once,
    // rather than discovering it from a stack trace.
    if (!llama_model_supports_mtp_multi_row_draft(llama_get_model(ctx))) {
        static bool pxa_bs_arch_warned = false;
        if (!pxa_bs_arch_warned) {
            pxa_bs_arch_warned = true;
            LOG_WRN("%s: PXA_MTP_BATCH_SLOTS is set, but this model's MTP draft graph takes one hidden "
                    "row per decode - drafting serially instead\n", __func__);
        }
        return false;
    }

    spec->t_step_start_us = ggml_time_us();

    auto & impl = spec->impls[0];
    auto & pxa_st = pxa_mtp_stats();

    // The whole batched draft is what the serial path charges to impl->t_draft_us, one call per slot.
    common_time_meas tm(impl->t_draft_us, !impl->gen_perf);

    // ---- per-sequence preparation: exactly common_speculative_state_mtp::draft()'s head ----
    struct prepared_chain {
        pxa_mtp_draft_chain  chain;
        size_t               req_idx    = 0;
        const float *        next_hidden = nullptr;  // the row fed into this chain's next decode
        std::vector<float> * embd_slot   = nullptr;  // where the read-back is stored (last.embd)
        int64_t              t0          = 0;
        int32_t              n_min       = 0;
    };
    std::vector<prepared_chain> chains;
    chains.reserve(reqs.size());

    // PXA_MTP_BATCH_SLOTS_WARM_v1: settle the whole step BEFORE writing any per-sequence state.
    // Preparation used to decide and mutate in one pass, which made a late refusal impossible: by the
    // time a slot turned out to be un-batchable its cached free token had already been consumed
    // (last.last_id cleared), so handing the step back would have silently dropped that token. The
    // caller answers false by running the ORDINARY serial drafter over the same requests, so a
    // refusal has to leave the drafter exactly as it found it. This pass therefore only reads.
    struct scouted_req {
        common_params_speculative  params;
        const std::vector<float> * target_hidden = nullptr;  // null: nothing for this slot to draft from
        pxa_mtp_batch_slot_state   state;
    };
    std::vector<scouted_req> scouted(reqs.size());
    std::vector<pxa_mtp_batch_slot_state> states;
    states.reserve(reqs.size());

    for (size_t ri = 0; ri < reqs.size(); ++ri) {
        auto & r  = reqs[ri];
        auto & sc = scouted[ri];

        // PXA_MTP_PREFETCH: drain any in-flight companion commit for this seq before touching
        // ctx_mtp -- including before reading the K/V position the warm-up test below compares.
        pxa_mtp_prefetch_wait_seq(r.seq_id);

        const auto runtime_stages = r.params.get_resolved_stages();
        const bool use_runtime_stage_overrides = common_speculative_stage_chain_matches(runtime_stages, spec->configs);
        const auto & runtime_stage = use_runtime_stage_overrides ? runtime_stages[0] : spec->configs[0].stage;
        sc.params = common_speculative_get_runtime_params(spec->configs[0], r.params, runtime_stage);

        sc.state.seq_id            = (int32_t) r.seq_id;
        sc.state.n_past            = (int32_t) r.draft_base_pos;
        sc.state.companion_pos_max = (int32_t) llama_kv_cache_seq_pos_max(ctx, r.seq_id);

        const auto hidden_it = mtp_state->target_hidden_by_seq.find(r.seq_id);
        if (hidden_it == mtp_state->target_hidden_by_seq.end() || (int) hidden_it->second.size() != n_embd) {
            LOG_WRN("%s: missing target hidden state for seq_id %d\n", __func__, (int) r.seq_id);
        } else {
            sc.target_hidden = &hidden_it->second;
            sc.state.have_target_hidden = true;
        }

        int32_t n_draft = sc.params.n_max;
        // PXA_MTP_ADAPTIVE_K: same ladder as mtp_speculative_gen_draft(), factored into
        // pxa_mtp_adaptive_k_depth() so the two schedulers cannot drift apart.
        if (pxa_mtp_adaptive_k_enabled() && n_draft > 1) {
            const int32_t k_adapt = pxa_mtp_adaptive_k_depth(mtp_state->pxa_ak_ema, n_draft);
            if (getenv("PXA_MTP_DBG")) {
                LOG_WRN("PXA_MTP_ADAPTIVE_K: ema=%.3f n_draft=%d -> K=%d\n",
                        (double) mtp_state->pxa_ak_ema, n_draft, k_adapt);
            }
            n_draft = k_adapt;
        }
        sc.state.n_draft = n_draft;

        states.push_back(sc.state);
    }

    // The step rule lives in pxa-mtp-batch-slots.h next to the shape rule, so the CPU test drives the
    // same code the server does: a slot whose companion K/V row has not caught up sends the WHOLE step
    // to the serial drafter, and so does a step with fewer than two slots left to regroup.
    int32_t n_batchable = 0;
    int32_t cold_slot   = -1;
    if (pxa_mtp_batch_slots_step_decision(states.data(), states.size(), &n_batchable, &cold_slot)
            != PXA_MTP_BATCH_STEP_BATCH) {
        if (cold_slot >= 0) {
            const auto & cs = states[(size_t) cold_slot];
            LOG_WRN("%s: MTP context not fully warmed up for seq_id %d: pos_max = %d, expected >= %d"
                    " - drafting this step serially\n",
                    __func__, (int) cs.seq_id, (int) cs.companion_pos_max, (int) cs.n_past - 1);
        }
        return false;
    }

    // ---- the commit pass: from here on the step is going to run, so it may write ----
    for (size_t ri = 0; ri < reqs.size(); ++ri) {
        auto & r  = reqs[ri];
        auto & sc = scouted[ri];

        impl->n_call_draft++;

        if (!pxa_mtp_batch_slot_can_draft(sc.state)) {
            if (sc.target_hidden != nullptr) {
                // A resolved depth of zero: the serial drafter drops the cached cross-step token and
                // returns an empty draft for this slot, so do exactly that.
                mtp_invalidate_cached_draft(*mtp_state, r.seq_id);
            }
            continue;
        }

        auto & last = mtp_get_last_embd(*mtp_state, r.seq_id);

        prepared_chain pc;
        pc.req_idx   = ri;
        pc.embd_slot = &last.embd;
        pc.t0        = pxa_st.on ? ggml_time_us() : 0;
        pc.n_min     = sc.params.n_min;

        const bool has_cached = last.last_id >= 0;
        const int32_t cached_id = last.last_id;
        const float   cached_prob = last.prob;
        if (has_cached) {
            last.last_id = -1;
        }

        if (!pc.chain.begin(r.seq_id,
                            sc.state.n_draft,
                            sc.params.p_min,
                            /* have_prob */ sc.params.p_min > 0 || pxa_st.on,
                            r.draft_base_pos,
                            r.id_last,
                            has_cached,
                            cached_id,
                            cached_prob)) {
            mtp_invalidate_cached_draft(*mtp_state, r.seq_id);
            continue;
        }

        // The first row's conditioning hidden: the cached free token's own MTP hidden when the
        // commit produced one, otherwise the target's stored hidden for this sequence.
        pc.next_hidden = has_cached ? last.embd.data() : sc.target_hidden->data();

        chains.push_back(std::move(pc));
    }

    if (chains.empty()) {
        return true;    // every slot resolved to an empty draft; nothing to decode
    }

    common_sampler_reset(mtp_state->smpl);

    // The row at n_past is committed history; a chain that starts at i == 0 is about to rewrite it
    // and must drop any stale cell there first (PXA_MTP_KVPOS_v1).
    for (const auto & pc : chains) {
        if (pc.chain.needs_pre_purge() && llama_kv_cache_seq_pos_max(ctx, pc.chain.seq_id) >= pc.chain.n_past) {
            llama_kv_cache_seq_rm(ctx, pc.chain.seq_id, pc.chain.n_past, -1);
        }
    }

    // ---- the step loop: one decode per STEP, one row per still-running chain ----
    llama_batch mtp_batch = llama_batch_init((int32_t) chains.size(), 0, 1);
    llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN);

    std::vector<float>   hidden_all;
    std::vector<int32_t> row_of;    // batch row -> index into `chains`
    hidden_all.reserve(chains.size() * (size_t) n_embd);
    row_of.reserve(chains.size());

    for (;;) {
        mtp_batch.n_tokens = 0;
        hidden_all.clear();
        row_of.clear();

        for (size_t c = 0; c < chains.size(); ++c) {
            auto & pc = chains[c];
            if (!pc.chain.wants_step()) {
                continue;
            }
            common_batch_add(mtp_batch, pc.chain.cur_id, pc.chain.step_pos(), { (llama_seq_id) pc.chain.seq_id }, true);
            hidden_all.insert(hidden_all.end(), pc.next_hidden, pc.next_hidden + n_embd);
            row_of.push_back((int32_t) c);
            pc.chain.on_step_issued();
        }

        if (mtp_batch.n_tokens == 0) {
            break;
        }

        if (!llama_set_draft_input_hidden_state_copy(ctx, hidden_all.data(), hidden_all.size()) ||
                llama_decode(ctx, mtp_batch) != 0) {
            for (int32_t c : row_of) {
                chains[c].chain.on_fail();
            }
            break;
        }

        for (size_t row = 0; row < row_of.size(); ++row) {
            auto & pc = chains[row_of[row]];

            float prob = 0.0f;
            float * prob_ptr = pc.chain.have_prob ? &prob : nullptr;
            const llama_token id_next = common_sampler_sample_speculative(mtp_state->smpl, ctx, (int) row, prob_ptr);

            if (pxa_st.on && prob_ptr) {
                pxa_st.p_sum += (double) prob;
                pxa_st.p_n   += 1;
                if (prob >= 0.75f) pxa_st.p_ge_075 += 1;
            }
            if (getenv("PXA_MTP_DBG")) {
                LOG_WRN("PXA_MTP_DBG seq %d step %d: in_id=%d n_past=%d -> draft=%d prob=%.3f\n",
                        (int) pc.chain.seq_id, (int) pc.chain.i, (int) pc.chain.cur_id,
                        (int) pc.chain.cur_pos, (int) id_next, prob_ptr ? (double) prob : -1.0);
            }

            if (pc.chain.on_sample(id_next, prob) == PXA_MTP_DRAFT_STEP_STOP) {
                continue;
            }

            const float * emb = llama_get_embeddings_ith(ctx, (int32_t) row);
            if (!emb) {
                pc.chain.on_fail();
                continue;
            }
            // Keep a stable copy: the next step's decode reuses ctx->embd storage.
            std::memcpy(pc.embd_slot->data(), emb, (size_t) n_embd * sizeof(float));
            pc.next_hidden = pc.embd_slot->data();

            pc.chain.on_hidden(prob);
        }
    }

    llama_batch_free(mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    // ---- per-sequence tail: exactly mtp_speculative_gen_draft()'s tail ----
    bool any_draft = false;
    for (auto & pc : chains) {
        auto & chain = pc.chain;

        if (chain.needs_purge()) {
            llama_kv_cache_seq_rm(ctx, chain.seq_id, chain.purge_p0(), chain.purge_p1());
        }

        if (pxa_mtp_adaptive_k_enabled()) {
            mtp_state->pxa_ak_drafted = (uint32_t) chain.drafts.size();
        }

        pxa_mtp_stats_record(pxa_st, pc.t0, chain.drafts.size(), chain.n_decode, chain.stop);

        auto & result = reqs[pc.req_idx].result;
        result = std::move(chain.drafts);

        // The dispatcher's self-spec fallback threshold, per slot (common_speculative_draft).
        if (!result.empty() && pc.n_min > 0 && (int) result.size() < pc.n_min) {
            LOG_DBG("%s: impl %s drafted %zu tokens, below fallback threshold %d - dropping\n",
                    __func__, common_speculative_type_to_str(impl->type).c_str(), result.size(), pc.n_min);
            result.clear();
        }

        if (!result.empty()) {
            any_draft = true;
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();
        }
    }

    // curr_impl drives common_speculative_current_type() and common_speculative_accept(); the shared
    // companion has exactly one impl, so this is the same value the serial path would have left.
    spec->curr_impl = any_draft ? impl.get() : nullptr;

    return true;
}


int32_t mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup) {
    if (batch.n_tokens == 0) {
        return 0;
    }

    llama_seq_id seq_id    = batch.seq_id[0][0];
    llama_pos    start_pos = batch.pos[0];

    // PXA_LLAMA_MTP_NP_FIX: ctx is the per-slot single-seq MTP context. Map the external seq_id to
    // local KV row 0 (avoids the np>1 multi-row recurrent-state cross-context copy crash).
    const llama_seq_id kv_seq = llama_n_seq_max(ctx) <= 1 ? 0 : seq_id;

    if (llama_kv_cache_seq_pos_max(ctx, kv_seq) >= start_pos) {
        llama_kv_cache_seq_rm(ctx, kv_seq, start_pos, -1);
    }

    LOG_WRN("[MTP-UPDATE|%s] Updating %d tokens for seq_id %d from pos %d...\n",
            is_prompt_warmup ? "PROMPT_WARMUP" : "GEN_ACCEPTED", batch.n_tokens, seq_id, (int)start_pos);

    // We never need all logits. We only need the logits of the last token so we can sample
    // the next draft token. In the MTP_OP_WARMUP case we do not need logits at all, but just
    // in case we also get the logits of the last token.
    llama_batch mtp_batch = batch;
    // PXA_LLAMA_MTP_NP_FIX: redirect the decode to local KV row kv_seq in the single-seq MTP context.
    std::vector<llama_seq_id>  mtp_seq_storage;
    std::vector<llama_seq_id*> mtp_seq_ptrs;
    std::vector<int32_t>       mtp_nseq;
    if (kv_seq != seq_id) {
        mtp_seq_storage.assign(mtp_batch.n_tokens, kv_seq);
        mtp_seq_ptrs.resize(mtp_batch.n_tokens);
        mtp_nseq.assign(mtp_batch.n_tokens, 1);
        for (int i = 0; i < mtp_batch.n_tokens; ++i) { mtp_seq_ptrs[i] = &mtp_seq_storage[i]; }
        mtp_batch.seq_id   = mtp_seq_ptrs.data();
        mtp_batch.n_seq_id = mtp_nseq.data();
    }
    // PXA_MTP_DRAFT_CACHE_ONLY: the prompt warm-up is a pure catch-up -- its caller
    // (common_speculative_prompt_warmup) never reads a logit or a hidden row back, it only checks
    // the return code. The accept path is NOT: mtp_accept_batch reads the last row's embedding and
    // samples the next free draft token from it, so it keeps the full graph. When the reduced graph
    // is taken, NO row asks for an output, which is the condition the whole thing is gated on.
    const pxa_mtp_cache_only_state pxa_co = {
        /* .enabled       = */ pxa_mtp_cache_only_enabled(),
        /* .wants_output  = */ !is_prompt_warmup,
        /* .n_layer_nextn = */ llama_model_supports_mtp_kv_only(llama_get_model(ctx)) ? 1 : 0,
        /* .n_tokens      = */ mtp_batch.n_tokens,
    };
    const bool pxa_kv_only = pxa_mtp_cache_only_ok(pxa_co);

    for (int i = 0; i < mtp_batch.n_tokens; ++i) {
        mtp_batch.logits[i] = false;
    }
    // PXA_MTP_ZERO_OUTPUT_COMMIT: with the lever on, an ACCEPTED-token catch-up asks for no outputs
    // at all -- n_outputs == 0, so the row-select gathers nothing and the FFN and the 248320-row
    // LM head compute nothing. The prompt warm-up keeps its trailing output row (it is the
    // cache-only path above, not this lever's).
    const bool pxa_zero_output = !is_prompt_warmup && common_speculative_mtp_zero_output_commit();
    // The two levers compose and are independently default-off: either one suppresses the
    // trailing output row, and the cache-only lever additionally switches the op type, because
    // MTP_OP_KV_ONLY is the stronger statement (store the KV, run nothing else) and it must win
    // when both are on.
    if (!pxa_kv_only && !pxa_zero_output) {
        mtp_batch.logits[mtp_batch.n_tokens-1] = true;
    }
    if (pxa_kv_only) {
        llama_set_mtp_op_type(ctx, MTP_OP_KV_ONLY);
    } else if (is_prompt_warmup) {
        llama_set_mtp_op_type(ctx, MTP_OP_WARMUP);
    } else {
        llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    }

    const int32_t ret = llama_decode(ctx, mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);
    return ret;
}

// ============================================================================================
// PXA_SPEC_POLICY (2026-09-20) — the per-request auto speculation policy.
// The table, the environment overrides and the once-per-choice banner. See speculative.h for
// the measurements this is seeded from.
// ============================================================================================

// split: 0 = any split mode, 1 = layer split, 2 = tensor split. A row whose split matches the
// running one wins over the any-split row for the same arch; that is the hook for the one axis
// the measurements do show a difference on (exact verification under a tensor split prefers a
// SHORT chain to a gated one: 56.5 t/s at n_max=1 against 53.6 at n_max=3 p_min=0.85, while
// under a layer split the two are inside the rep spread at 43.6 and 43.4). Nothing is seeded
// split-specific yet: this cut ships one row per arch and lets the measurement decide.
struct pxa_spec_policy_row {
    const char * arch;
    int          split;
    int32_t      n_max_exact;
    float        p_min_exact;
    int32_t      n_max_relaxed;
    float        p_min_relaxed;
    const char * why_exact;
    const char * why_relaxed;
};

static const pxa_spec_policy_row pxa_spec_policy_tab[] = {
    { "qwen35", 0,
      3, 0.85f,
      3, 0.00f,
      "exact verification keeps a draft token only when the target's own pick equals it, so an "
      "ungated depth-3 chain proposes ~2.95 tokens to deliver ~1.06 at temperature 1. The "
      "confidence floor stops the chain at the first token the head is unsure of: measured "
      "2026-09-20 on a 2x V100 pair, Qwen3.8-27B PXQ4, fresh text, REPS 3 -- 43.4 t/s greedy / "
      "41.6 sampled against 40.0 / 31.7 at p_min=0 and 37.7 with no speculation (layer split)",
      "the relaxed rule keeps a draft token that holds enough of the target's own post-filter "
      "mass, so acceptance runs ~82% and depth pays while a floor only costs proposals: measured "
      "2026-09-20, same pair and prompts, 69.4 t/s at p_min=0 against 60.0 at 0.85 and 63.5 at "
      "0.5 (tensor split, sampled)" },
    { "qwen35moe", 0,
      3, 0.85f,
      3, 0.00f,
      "carried from the dense sibling's measurement (no MoE cell taken); MTP was a tax at every "
      "depth on this family when it was last measured, so measure before trusting this row",
      "carried from the dense sibling's measurement (no MoE cell taken); MTP was a tax at every "
      "depth on this family when it was last measured, so measure before trusting this row" },
};

bool pxa_spec_policy_enabled() {
    static const bool en = [] {
        const char * e = getenv("PXA_SPEC_POLICY");
        return e && atoi(e) != 0;
    }();
    return en;
}

// "n_max=3,p_min=0.85" — a field left out keeps the table's value.
static bool pxa_spec_policy_parse(const char * s, int32_t & n_max, float & p_min, std::string & err) {
    std::string in(s);
    size_t pos = 0;
    while (pos < in.size()) {
        size_t end = in.find(',', pos);
        if (end == std::string::npos) {
            end = in.size();
        }
        std::string kv = in.substr(pos, end - pos);
        pos = end + 1;
        while (!kv.empty() && (kv.front() == ' ' || kv.front() == '\t')) {
            kv.erase(kv.begin());
        }
        if (kv.empty()) {
            continue;
        }
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) {
            err = "'" + kv + "' is not key=value";
            return false;
        }
        const std::string key = kv.substr(0, eq);
        const std::string val = kv.substr(eq + 1);
        if (key == "n_max") {
            n_max = atoi(val.c_str());
            if (n_max < 0) {
                err = "n_max must be >= 0";
                return false;
            }
        } else if (key == "p_min") {
            p_min = (float) atof(val.c_str());
            if (p_min < 0.0f || p_min > 1.0f) {
                err = "p_min must be in [0,1]";
                return false;
            }
        } else {
            err = "unknown key '" + key + "' (only n_max and p_min)";
            return false;
        }
    }
    return true;
}

struct pxa_spec_policy_state {
    bool                   resolved = false;
    bool                   active   = false;
    bool                   cli_n_max = false;
    bool                   cli_p_min = false;
    std::string            arch;
    bool                   tensor_split = false;
    pxa_spec_policy_choice exact;
    pxa_spec_policy_choice relaxed;
};

static pxa_spec_policy_state & pxa_spec_policy_st() {
    static pxa_spec_policy_state st;
    return st;
}

bool pxa_spec_policy_resolve(const std::string & arch, bool tensor_split) {
    pxa_spec_policy_state & st = pxa_spec_policy_st();
    st.resolved     = true;
    st.active       = false;
    st.arch         = arch;
    st.tensor_split = tensor_split;

    if (!pxa_spec_policy_enabled()) {
        return false;
    }

    const int want_split = tensor_split ? 2 : 1;
    const pxa_spec_policy_row * row = nullptr;
    for (const auto & r : pxa_spec_policy_tab) {
        if (arch != r.arch) {
            continue;
        }
        if (r.split == want_split) { row = &r; break; }      // a split-specific row wins outright
        if (r.split == 0 && row == nullptr) { row = &r; }     // else remember the any-split row
    }

    if (row == nullptr) {
        fprintf(stderr, "PXA_SPEC_POLICY: no row for arch=%s -- the per-request policy stands down "
                        "and the shipped auto defaults are used unchanged\n",
                arch.empty() ? "?" : arch.c_str());
        return false;
    }

    st.exact   = { row->n_max_exact,   row->p_min_exact,   row->why_exact,   false };
    st.relaxed = { row->n_max_relaxed, row->p_min_relaxed, row->why_relaxed, false };

    struct { const char * env; pxa_spec_policy_choice * dst; const char * name; } ov[] = {
        { "PXA_SPEC_POLICY_EXACT",   &st.exact,   "EXACT"   },
        { "PXA_SPEC_POLICY_RELAXED", &st.relaxed, "RELAXED" },
    };
    for (const auto & o : ov) {
        const char * e = getenv(o.env);
        if (e == nullptr) {
            continue;
        }
        int32_t n_max = o.dst->n_max;
        float   p_min = o.dst->p_min;
        std::string err;
        if (!pxa_spec_policy_parse(e, n_max, p_min, err)) {
            fprintf(stderr, "PXA_SPEC_POLICY: %s='%s' ignored -- %s (format: n_max=N,p_min=X)\n",
                    o.env, e, err.c_str());
            continue;
        }
        o.dst->n_max    = n_max;
        o.dst->p_min    = p_min;
        o.dst->why      = "named by the operator";
        o.dst->from_env = true;
    }

    st.active = true;
    fprintf(stderr, "PXA_SPEC_POLICY: arch=%s split=%s -> the MTP draft depth and confidence floor "
                    "are chosen PER REQUEST from the acceptance rule that request will run under. "
                    "EXACT (temperature 0, top_k 1, a grammar, mirostat, or relaxed acceptance off) "
                    "-> n_max=%d p_min=%.3f%s. RELAXED (everything else) -> n_max=%d p_min=%.3f%s. "
                    "Anything the request names explicitly still wins, field by field. Override "
                    "PXA_SPEC_POLICY_EXACT / PXA_SPEC_POLICY_RELAXED ('n_max=N,p_min=X'), or "
                    "PXA_SPEC_POLICY=0 for the shipped single-row behaviour\n",
            arch.empty() ? "?" : arch.c_str(), tensor_split ? "tensor" : "layer",
            st.exact.n_max,   (double) st.exact.p_min,   st.exact.from_env   ? " (yours)" : "",
            st.relaxed.n_max, (double) st.relaxed.p_min, st.relaxed.from_env ? " (yours)" : "");
    return true;
}

bool pxa_spec_policy_active() {
    return pxa_spec_policy_enabled() && pxa_spec_policy_st().active;
}

int32_t pxa_spec_policy_n_max_ceiling() {
    const pxa_spec_policy_state & st = pxa_spec_policy_st();
    if (!st.active) {
        return -1;
    }
    return std::max(st.exact.n_max, st.relaxed.n_max);
}

void pxa_spec_policy_set_cli_named(bool n_max_named, bool p_min_named) {
    pxa_spec_policy_state & st = pxa_spec_policy_st();
    st.cli_n_max = n_max_named;
    st.cli_p_min = p_min_named;
    if (st.active && (n_max_named || p_min_named)) {
        fprintf(stderr, "PXA_SPEC_POLICY: the MTP stage's %s%s%s named on the command line, so the "
                        "policy leaves %s alone and fills only the rest\n",
                n_max_named ? "n_max" : "",
                (n_max_named && p_min_named) ? " and " : "",
                p_min_named ? "p_min was" : "was",
                (n_max_named && p_min_named) ? "both" : "it");
    }
}

bool pxa_spec_policy_cli_named_n_max() { return pxa_spec_policy_st().cli_n_max; }
bool pxa_spec_policy_cli_named_p_min() { return pxa_spec_policy_st().cli_p_min; }

pxa_spec_policy_choice pxa_spec_policy_for(bool exact_rule) {
    const pxa_spec_policy_state & st = pxa_spec_policy_st();
    return exact_rule ? st.exact : st.relaxed;
}

void pxa_spec_policy_announce(bool exact_rule, const char * rule_why, int32_t n_max, float p_min) {
    static std::mutex               mtx;
    static std::set<std::string>    seen;

    char key[160];
    snprintf(key, sizeof(key), "%d|%d|%.4f|%s", exact_rule ? 1 : 0, n_max, (double) p_min,
             rule_why ? rule_why : "");

    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!seen.insert(key).second) {
            return;
        }
    }

    const pxa_spec_policy_choice ch = pxa_spec_policy_for(exact_rule);
    fprintf(stderr, "PXA_SPEC_POLICY: request runs under %s acceptance (%s) -> MTP stage n_max=%d "
                    "p_min=%.3f (%s)\n",
            exact_rule ? "EXACT" : "RELAXED", rule_why ? rule_why : "?", n_max, (double) p_min,
            ch.why);
}
