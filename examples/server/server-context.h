#include "server-task.h"
#include "server-queue.h"
#include "speculative.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <memory>
#include <atomic>



enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_PROCESSING,
};



enum slot_command {
    SLOT_COMMAND_NONE,
    SLOT_COMMAND_LOAD_PROMPT,
    SLOT_COMMAND_RELEASE,
};

struct server_speculative_checkpoint {
    bool valid = false;
    bool per_step_enabled = false; // per-step SSM checkpoints active
    llama_pos n_past = 0;
    llama_token sampled = LLAMA_TOKEN_NULL;
    common_sampler * sampler = nullptr; // saved sampler state

    void clear();
};

struct server_slot {
    int id;
    int id_task = -1;
    int id_multi = -1;

    struct slot_params params;

    llama_batch batch_spec = {};
    llama_context * ctx_dft = nullptr;

    bool released = false;
    slot_state state = SLOT_STATE_IDLE;
    slot_command command = SLOT_COMMAND_NONE;

    llama_context* ctx = nullptr;
    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    std::unique_ptr<const server_task> task;

    // generation props
    int32_t n_ctx = 0;  // context size per slot
    int32_t n_past = 0;
    int32_t n_past_prompt = 0;
    int32_t n_past_offset = 0;
    int32_t n_decoded = 0;
    int32_t n_remaining = -1;
    int32_t n_discarded_prompt = 0;
    int32_t n_kept_prompt = 0;
    int32_t n_softfail_consecutive = 0; // PXA_SOFTFAIL_BREAKER_v1

    int32_t i_batch = -1;
    int32_t n_predict = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens = 0;
    int32_t n_prompt_tokens_cache = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt; // can be either a string, array of strings or array of token ids

    // when a task is submitted, we first tokenize the prompt and store it here
    server_tokens prompt_tokens;
    server_tokens cache_tokens;

    int32_t last_gentxt_size = 0;
    std::string generated_text;

    // idx of draft tokens in the main batch
    // non-empty if we went to evaluate draft tokens
    // ref: https://github.com/ggml-org/llama.cpp/pull/17808
    std::vector<int32_t> i_batch_dft;

    std::vector<completion_token_output> generated_token_probs;


    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    bool saturate_predict = false;
    bool oaicompat = false;

    std::string oaicompat_model;
    std::string stopping_word;
    stop_type stop;

    // For context rewind/ token buffer
    size_t n_buffer = 0;
    int32_t rewind_count = 0;
    int32_t rewind_count_max = -1;
    bool rewind_status = false;
    std::unordered_map<llama_token, float> logit_bias;
    std::vector<std::string> ban_phrases;
    std::vector<std::string> ban_regex;
    std::vector<std::string> ban_regex_ci;
    completion_token_outputs token_buffer;
    float ban_phrases_bias = 0;
    int32_t banned_n = 1;
	std::map<int32_t, std::set<llama_token>> positional_bans;

    // allowlist
    std::vector<std::vector<std::tuple<uint32_t, uint32_t, std::string, float>>> allow_ruless_prev;
    std::vector<std::vector<std::tuple<uint32_t, uint32_t, std::string, float>>> allow_ruless;
    std::vector<std::string> allow_pieces;
    std::vector<std::string> allow_kws;
    size_t allow_kw_delay = 0;
    std::vector<std::vector<float>> allow_biasess;
    size_t allow_idx = 0;

    server_prompt server_cached_prompt;

    void prompt_save(server_prompt_cache& prompt_cache) const;

    void prompt_load(server_prompt_cache& prompt_cache, const server_tokens& tokens, float min_reusable_fraction);

    size_t checkpoint_pos = 0;
    bool do_checkpoint = false;
    bool image_just_processed = false;

    // sampling
    llama_token sampled; // in speculative mode, this is the last accepted token
    llama_tokens drafted;
    common_speculative_type drafted_spec_type = COMMON_SPECULATIVE_TYPE_NONE;

    json json_schema;

    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_msg chat_msg;
    int32_t chat_parse_throttle = 0; // PXA_CHATPARSE_THROTTLE: per-slot counter to throttle the O(N^2) full re-parse
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    bool anthropic_thinking_block_started = false;
    bool anthropic_text_block_started = false;

    bool oai_resp_thinking_block_started = false;
    bool oai_resp_text_block_started = false;

    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    // multimodal
    mtmd_context* mctx = nullptr;

    // speculative decoding
    struct common_speculative * spec = nullptr;
    // PXA_SHARED_MTP_v1: one ctx_mtp/spec is shared across ALL slots (created once on slot 0,
    // n_seq_max=n_parallel). Only the owning slot frees it; the rest hold a non-owning alias so
    // the single shared ctx_mtp is freed exactly once. Per-seq routing is by slot.id (the seq_id).
    bool owns_spec = false;
    struct common_params_sampling sparams;
    common_sampler * ctx_sampling = nullptr;

    // expiring logit bias
    std::vector<common_sampler::elb_state> prev_elb_states;

    bool has_mtp = false;

    // saves recurrent state before a speculative batch so it can be restored on rejection
    server_speculative_checkpoint spec_ckpt;

    // speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    // PXA_MTP_ADAPTIVE: acceptance-EMA-gated draft-length controller (np=1 only; see add_sampled_tokens)
    float spec_accept_ema      = 0.70f; // EMA of per-step draft acceptance ratio (optimistic cold start -> full drafts)
    int   spec_probe_countdown = 0;     // skip-tier steps left until the next full-length probe draft
    int   spec_adaptive_tier   = 2;     // last applied tier (2 = full, 1 = cap-to-1, 0 = skip) for change logging

    int32_t n_past_se = 0; // self-extend

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    void reset();

    bool need_embd() const;

    bool has_budget(gpt_params& global_params);

    bool available() const;

    bool is_processing() const;

    void add_token_string(const completion_token_output& token);

    bool can_speculate() const;

    int get_n_draft_max() const;

    void release();

    json get_formated_timings() const;

    result_timings get_timings() const;

    const common_chat_msg& update_chat_msg(bool is_partial, std::vector<common_chat_msg_diff>& diffs,
        bool filter_tool_calls = false);

    size_t find_stopping_strings(const std::string& text, const size_t last_token_size, bool is_full_stop);

    void print_timings() const;

};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total = 0;
    uint64_t n_tokens_predicted_total = 0;
    uint64_t t_tokens_generation_total = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing = 0;

    uint64_t n_tokens_predicted = 0;
    uint64_t t_tokens_generation = 0;

    void init();

    void on_prompt_eval(const server_slot& slot);

    void on_prediction(const server_slot& slot);

    void reset_bucket();
};

struct server_context {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::vector<llama_lora_adapter_container> lora_adapters;
    std::vector<control_vector_container> control_vectors;

    std::vector<std::string> vocab_pieces;
    size_t max_piece_len = 0;

    gpt_params params_base;

    llama_batch batch = {};

    // PXA_HONEST_HEALTH_v1: decode heartbeat. Stamped around every llama_decode in update_slots so
    // /health can detect a wedged main loop WITHOUT posting a metrics task and blocking on
    // queue_results.recv (the real HTTP-worker thread-saturation feeder under a wedge).
    std::atomic<int64_t> last_decode_start_us{0};
    std::atomic<int64_t> last_decode_done_us{0};
    bool pxa_decode_wedged(int64_t stall_ms) const;

    bool clean_kv_cache = true;
    bool add_bos_token = true;
    bool has_eos_token = false;

    // PXA_PROMPT_INTERLEAVE_v1 (PXA_KERNEL_FIX_20260611): alternation flip-flop between
    // pure-prompt ticks and pure-decode ticks on recurrent/hybrid models (see update_slots).
    bool pxa_prev_tick_was_prompt = false;
    // PXA_INTERLEAVE_DECODE_MS_v1: end of the decode window opened by the last prompt chunk.
    int64_t pxa_decode_window_until_us = 0;

    // multimodal
    mtmd_context* mctx = nullptr;

    // For speculative decoding
    llama_model* model_draft = nullptr;
    llama_context* ctx_draft = nullptr;
    llama_context_params cparams_dft;

    int32_t n_ctx; // total context for all clients / slots

    // ---- PXA_KV_UNIFIED_v1 -------------------------------------------------------------------
    // Effective per-slot attention-context ceiling, computed once in init().
    //   lever OFF (default) : n_ctx / n_parallel   -- byte-identical to the historical behaviour
    //   lever ON            : n_ctx                -- every slot may address the whole shared ring
    // The ring itself is unchanged either way: this base already allocates ONE seq-tagged cell ring
    // of exactly cparams.n_ctx cells (src/llama.cpp:8459) and never multiplies it by n_seq_max, so
    // -np N costs zero extra attention-KV bytes. Only the server's arithmetic is being unified.
    //
    // SCOPE, stated explicitly because it is easy to get wrong on this model family: every number
    // tracked here is ATTENTION KV CELLS (one cell == one token position in the shared ring). The
    // per-sequence recurrent GDN/delta-net state (n_embd_v_s rows, ~112.2 MiB per sequence on
    // qwen4exp) lives in kv_self.s_l, is indexed by absolute seq_id, is NOT part of this ring, does
    // NOT unify, and its cost still scales with -np. It is deliberately absent from every budget
    // below. Nor do we ever let two slots share a cell: src/llama-delta-net.cpp:43 asserts
    // batch.n_seq_id[i] == 1, so cross-slot cell sharing stays forbidden.
    int32_t n_ctx_slot = 0;

    // Attention KV cells the shared ring must still be able to give a slot beyond its prompt before
    // that slot can make forward progress. Keeps admission from handing out a ring with room for the
    // prompt but not for a single generated token.
    int32_t kv_unified_reserve = 0;

    // Cells currently pinned in the shared ring by slots OTHER than `except` that we may not touch:
    // any slot that is mid-request. Idle slots are counted separately because their cells are only a
    // prompt cache and are reclaimable without disturbing a live sequence.
    int32_t kv_unified_cells_busy(const server_slot * except) const;
    int32_t kv_unified_cells_idle_reclaimable(const server_slot * except) const;

    // Drop the prompt-cache KV of idle slots (oldest-used first) until `need` cells are free or
    // nothing reclaimable is left. Returns cells reclaimed. NEVER touches a processing slot.
    int32_t kv_unified_reclaim_idle(int32_t need, const server_slot * except);

    // Admission gate. Decides whether a prompt of n_prompt_tokens may enter the shared ring now.
    enum kv_admission { KV_ADMIT_OK, KV_ADMIT_DEFER, KV_ADMIT_REJECT };
    kv_admission kv_unified_admit(int32_t n_prompt_tokens, const server_slot * except, int32_t & n_free_out);
    // ------------------------------------------------------------------------------------------

    // system prompt
    bool system_need_update = false;

    std::string              system_prompt;
    std::vector<llama_token> system_tokens;

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    server_queue    queue_tasks;
    server_response queue_results;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    common_chat_templates_ptr chat_templates;
    server_chat_params  chat_params;
    std::map<std::string, bool> chat_template_caps;

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;
    int32_t cache_ram_n_min = 0;
    float cache_ram_similarity = 0.5f;

    ~server_context();

    bool load_model(const gpt_params& params_);

    void init();

    std::vector<llama_token> tokenize(const json& json_prompt, bool add_special) const;

    server_slot* get_slot_by_id(int id);

    float calculate_slot_f_keep(const server_slot& slot, llama_context* ctx, const server_tokens& a, const server_tokens& b);

    std::pair<common_prefix, float> calculate_slot_similarity(const server_slot& slot, llama_context* ctx, const server_tokens& a, const server_tokens& b);

    void copy_data_to_cached_prompt(const server_tokens& tokens, server_slot& slot);

    server_slot* get_available_slot(const server_task& task);

    int32_t populate_vocab_pieces();

    bool launch_slot_with_task(server_slot& slot, server_task& task);

    void kv_cache_clear();

    void system_prompt_update();

    bool system_prompt_set(const std::string& sys_prompt);

    bool process_token(completion_token_output& result, server_slot& slot);

    void populate_token_probs(const server_slot& slot, completion_token_output& result, bool post_sampling, bool special, int idx);

    json get_formated_generation(const server_slot& slot) const;

    void send_error(const server_task& task, const std::string& error, const enum error_type type = ERROR_TYPE_SERVER);

    void send_error(const server_slot& slot, const std::string& error, const enum error_type type = ERROR_TYPE_SERVER);

    void send_error(const int id_task, const int id_multi, const std::string& error, const enum error_type type = ERROR_TYPE_SERVER);

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task);

    void send_partial_response(server_slot& slot, completion_token_output tkn);

    void send_final_response(server_slot& slot);

    void send_embedding(const server_slot& slot, const llama_batch& batch);

    void apply_server_biases(server_slot& slot);

    void request_completion(int id_task, int id_multi, json data, bool infill, bool embedding, server_tokens & inputs);

    void request_cancel(int id_task);

    void split_multiprompt_task(int id_multi, server_task& multiprompt_task);

    void process_single_task(server_task&& task);

    void on_finish_multitask(const server_task_multi& multitask);

    void print_tokens(const server_tokens& prompt, const server_tokens& cache, size_t start1 = 0, size_t start2 = 0, size_t length = 10);

    // discard tokens in kv cache and cached tokens
    void discard_n_kv_and_cache_tokens(llama_context* ctx, server_slot& slot, int32_t n_keep, int32_t n_discard);

    // convert keep first few and discard next tokens in a to b
    void context_shift_find_n_tokens(llama_context* ctx, const server_tokens& a, const server_tokens& b, int32_t n_keep,
        int32_t n_discard, int32_t& n_kept, int32_t& n_discarded, bool exact = false);

    // handle context shift for prompt
    void context_shift_prompt(llama_context* ctx, server_slot& slot, bool exact = false);

    void update_slots();

    void release_slots();

    bool slots_idle();

    void context_shift();

    void add_sampled_tokens();

    void batch_pending_prompt(const int32_t n_ubatch, const int32_t n_batch,  int32_t & batch_type);

    void process_batch_tokens(int32_t & n_batch);

    void extend_context(const int32_t n_tokens);

    void speculative_decoding_accept();

    bool accept_special_token(const server_slot& slot, const llama_token token);

    bool has_next_token(const completion_token_output& result, server_slot& slot);

    void send_token_results(completion_token_outputs& results, server_slot& slot, int32_t n = 0);

    void buffer_and_check_string_ban(server_slot& slot, completion_token_output& result);

    void update_allowlist_state(server_slot& slot);

    json model_meta() const;

    // Re-aggregates all active vectors and updates the model state
    bool apply_control_vectors_internal();

    bool create_checkpoint(server_slot & slot);

    void apply_checkpoint(server_slot & slot);

    void create_checkpoint_at_interval(server_slot & slot, const gpt_params & params_base, bool in_prompt = false);

    void release_slot_after_final_response(server_slot & slot);
};
