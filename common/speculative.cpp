#include "speculative.h"

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
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX}
};

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
static bool pxa_mtp_adaptive_k_enabled() {
    static const int v = getenv("PXA_MTP_ADAPTIVE_K") ? atoi(getenv("PXA_MTP_ADAPTIVE_K")) : 0;
    return v != 0;
}

struct mtp_last_embd {
    std::vector<float> embd;
    float prob = 0.0f;
    int   last_id = -1;
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

    // PXA_MTP_ADAPTIVE_K controller state (inert unless PXA_MTP_ADAPTIVE_K=1).
    float    pxa_ak_ema     = 0.70f; // running acceptance EMA (optimistic cold-start = full-depth drafts)
    uint32_t pxa_ak_drafted = 0;     // draft tokens emitted last cycle (accept() denominator)

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_mtp,
            bool constant_draft_positions = false)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_mtp(ctx_mtp)
        , constant_draft_positions(constant_draft_positions)
    {
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

    ~common_speculative_state_mtp() override {
        common_sampler_free(smpl);
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

    void accept(uint16_t n_accepted) override {
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
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
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
                if (i < params.n_min) {
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

static common_params_speculative common_speculative_get_runtime_params(
        const common_speculative_config & config,
        const common_params_speculative & params,
        const common_speculative_stage_params & stage) {
    common_params_speculative result = config.params;

    result.type = config.type;
    result.n_max = stage.has_n_max_override() ? stage.n_max : params.n_max;
    result.n_min = stage.has_n_min_override() ? stage.n_min : params.n_min;
    result.p_min = stage.has_p_min_override() ? stage.p_min : params.p_min;

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
                if (!ctx_mtp) {
                    const llama_model * model = llama_get_model(ctx_tgt);
                    ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model), config.params.cparams_dft);
                    if (!ctx_mtp) {
                        LOG_ERR("%s: failed to create MTP context\n", __func__);
                        return nullptr;
                    }
                }
                ctx_dft = nullptr;

                const bool use_constant_draft_positions = llama_model_is_gemma4_mtp_assistant(llama_get_model(ctx_mtp));
                impls.push_back(std::make_unique<common_speculative_state_mtp>(
                    config.type, ctx_tgt, ctx_mtp, use_constant_draft_positions));
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

        if (result.empty()) {
            continue;
        }

        if (common_speculative_type_is_self_spec(impl->type) && impl_params.n_min > 0 && (int)result.size() < impl_params.n_min) {
            LOG_DBG("%s: impl %s drafted %zu tokens, below fallback threshold %d - trying next implementation\n",
                    __func__, common_speculative_type_to_str(impl->type).c_str(), result.size(), impl_params.n_min);
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

    // store draft count for tuner feedback
    if (spec->tuner && spec->tuner->enabled) {
        spec->last_n_drafted = (int)result.size();
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
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

    if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
        if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
            mtp_invalidate_cached_drafts(*mtp_state);
        }
    }
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

    common_speculative_feature_view features;
    if (!llama_spec_get_hidden_feature_view_from_output_index(ctx, output_index, seq_id, pos, features)) {
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

    return llama_spec_copy_hidden_rows_from_output_indices(ctx, output_indices, hidden_rows);
}

static bool common_speculative_build_commit_tokens(
        common_speculative_type spec_type_used,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        std::vector<llama_token> & commit_tokens) {
    commit_tokens.clear();
    if (ids.empty()) {
        return true;
    }

    if (spec_type_used == COMMON_SPECULATIVE_TYPE_MTP) {
        commit_tokens = ids;
        return true;
    }

    commit_tokens.reserve(ids.size());
    commit_tokens.push_back(sampled_before);
    if (ids.size() > 1) {
        commit_tokens.insert(commit_tokens.end(), ids.begin(), ids.end() - 1);
    }

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

    llama_batch accepted_batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(accepted_batch, ids[i], pos_base + (llama_pos) i, { seq_id }, true);
    }

    common_speculative_feature_view feature_view;
    const bool have_feature_view = common_speculative_feature_view_from_hidden_rows(
        hidden_rows, mtp_state->n_embd, seq_id, pos_base, feature_view);
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

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
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

static void mtp_invalidate_cached_draft(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto it = state.draft_cache_by_seq.find(seq_id);
    if (it == state.draft_cache_by_seq.end()) {
        return;
    }

    it->second.last_id = -1;
    it->second.prob = 0.0f;
}

static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state) {
    for (auto & entry : state.draft_cache_by_seq) {
        entry.second.last_id = -1;
        entry.second.prob = 0.0f;
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
    const float * embd = llama_get_embeddings_ith(state.ctx_mtp, accepted_batch.n_tokens - 1);
    if (embd != nullptr) {
        std::memcpy(last.embd.data(), embd, last.embd.size() * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, last.embd.data(), last.embd.size())) {
            return -1;
        }
        last.last_id = common_sampler_sample_speculative(nullptr, state.ctx_mtp, accepted_batch.n_tokens - 1, &last.prob);
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
    if (mtp_state == nullptr || reqs.empty()) {
        return failed;
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
    // drop any stale rows at/after this seq's start position so the commit writes clean cells.
    for (const auto & ps : prepared) {
        const llama_pos start_pos = ps.pos_base;
        if (llama_kv_cache_seq_pos_max(ctx, ps.seq_id) >= start_pos) {
            llama_kv_cache_seq_rm(ctx, ps.seq_id, start_pos, -1);
        }
    }

    // Assemble ONE batch + ONE concatenated hidden-row buffer in batch-token order.
    llama_batch batch = llama_batch_init((int) total_tokens, 0, 1);
    std::vector<float> hidden_all;
    hidden_all.reserve(total_tokens * (size_t) n_embd);
    for (auto & ps : prepared) {
        for (size_t i = 0; i < ps.commit_tokens.size(); ++i) {
            const bool is_last = (i + 1 == ps.commit_tokens.size());
            common_batch_add(batch, ps.commit_tokens[i], ps.pos_base + (llama_pos) i, { ps.seq_id }, is_last);
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
    for (auto & ps : prepared) {
        auto & last = mtp_get_last_embd(*mtp_state, ps.seq_id);
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
        last.last_id = common_sampler_sample_speculative(nullptr, ctx, ps.last_token_batch_idx, &last.prob);
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
    if (mtp_state == nullptr) {
        return 0;
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

    llama_batch mtp_batch = llama_batch_init(1, 0, 1);
    llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN);

    float prob;
    auto prob_ptr = p_min > 0 ? &prob : nullptr;

    llama_token current_input_id = id_last;
    llama_pos current_n_past = n_past;
    const int n_embd = llama_mtp_state_n_embd(ctx);

    auto & last = mtp_get_last_embd(state, seq_id);
    int i0 = 0;
    if (last.last_id >= 0) {
        if (last.prob < p_min) {
            n_draft = 1;
        }
        current_input_id = last.last_id;
        last.last_id = -1;
        drafts.push_back(current_input_id);
        current_n_past++;
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            llama_batch_free(mtp_batch);
            llama_set_mtp_op_type(ctx, MTP_OP_NONE);
            return drafts;
        }
        i0 = 1;
    }

    int n_decode = 0;
    for (int i = i0; i < n_draft; ++i) {
        mtp_batch.n_tokens = 0;
        const llama_pos draft_pos = constant_draft_positions ? n_past : current_n_past;
        common_batch_add(mtp_batch, current_input_id, draft_pos, {kv_seq}, true);

        ++n_decode;
        if (llama_decode(ctx, mtp_batch) != 0) {
            break;
        }

        llama_token id_next = common_sampler_sample_speculative(smpl, ctx, 0, prob_ptr);
        if (getenv("PXA_MTP_DBG")) {
            LOG_WRN("PXA_MTP_DBG step %d: in_id=%d n_past=%d -> draft=%d prob=%.3f\n",
                i, (int)current_input_id, (int)current_n_past, (int)id_next, prob_ptr ? (double)prob : -1.0);
        }

        if (i > 0 && prob_ptr && prob < p_min) {
            break;
        }

        drafts.push_back(id_next);

        const float * emb = llama_get_embeddings_ith(ctx, 0);
        if (!emb) {
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
            break;
        }

        current_input_id = id_next;
        current_n_past++;

        if (prob_ptr && prob < p_min) {
            break;
        }
    }
    llama_batch_free(mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    // Purge the metadata for the draft tokens.
    // This prevents cache state corruption where two cells map to the same logical position.
    // If the state contained in `last` had a valid token id and probability, it means that we
    // have previously run an "accept" batch, where the token sampled from the main model was included.
    // In that case, we need to discard all tokens that we ran here to get the KV cache to the correct state.
    //   => for i0 = 1 we discard from n_past
    // But if we did not have a valid last token_id, it means the first token we run was sampled from the
    // main model. Hence we want to keep this token in the KV cache and discard all other tokens.
    //   => for i0 = 0 we discard from n_past + 1
    if (n_decode > 0) {
        llama_kv_cache_seq_rm(ctx, kv_seq, n_past + 1 - i0, n_past + n_decode + 2);
    }

    // PXA_MTP_ADAPTIVE_K: record how many draft tokens we emitted this cycle so accept() can form
    // the accept ratio for the EMA. (Inert unless the controller is enabled.)
    if (pxa_mtp_adaptive_k_enabled()) {
        state.pxa_ak_drafted = (uint32_t) drafts.size();
    }

    return drafts;
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
    for (int i = 0; i < mtp_batch.n_tokens; ++i) {
        mtp_batch.logits[i] = false;
    }
    mtp_batch.logits[mtp_batch.n_tokens-1] = true;
    if (is_prompt_warmup) {
        llama_set_mtp_op_type(ctx, MTP_OP_WARMUP);
    } else {
        llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    }

    const int32_t ret = llama_decode(ctx, mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);
    return ret;
}
