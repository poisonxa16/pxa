#pragma once

#include "llama-impl.h"
#include "llama-cparams.h"
#include "llama-sampling.h"

#include "llama-spec-features.h"

struct llama_model;

#include <vector>
#include <cstdlib>
#include <map>
#include <set>
#include <memory>

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = 0; // used by recurrent state models to copy states

    // PXA_KV_SEQ_SOA (house custom): the set of sequence ids is PRIVATE. Every mutation goes
    // through add_seq/erase_seq/clear_seq so that the inline shadow `seq0` below can never drift
    // from the set -- a missed mutation site is a compile error, not a stale mask. Whole-cell
    // copies (defrag moves, checkpoint snapshots, `= llama_kv_cell()`) carry the shadow along.
    //
    // seq0 encodes the set for the O(1) has_seq_fast() read used by the decode KQ-mask fill when
    // PXA_KV_SEQ_SOA=1 (the OFF path keeps the red-black-tree has_seq_id() lookup):
    //   -1        : empty
    //   -2        : two or more ids (consult the set; only seq_cp creates these)
    //   otherwise : the single id
    // It lives in the 4-byte hole after `src`, so the cell stays 64 bytes on libstdc++.
    static constexpr int32_t SEQ_NONE  = -1;
    static constexpr int32_t SEQ_MULTI = -2;

private:
    int32_t seq0 = SEQ_NONE;
    std::set<llama_seq_id> seq_id;

    void sync_shadow() {
        seq0 = seq_id.empty() ? SEQ_NONE : (seq_id.size() == 1 ? *seq_id.begin() : SEQ_MULTI);
    }

public:
    void add_seq(llama_seq_id id)   { seq_id.insert(id); sync_shadow(); }
    void erase_seq(llama_seq_id id) { seq_id.erase(id);  sync_shadow(); }
    void clear_seq()                { seq_id.clear();    seq0 = SEQ_NONE; }

    // read-only views of the set
    const std::set<llama_seq_id> & seqs() const { return seq_id; }
    size_t n_seq() const { return seq_id.size(); }
    int32_t seq_shadow() const { return seq0; }

    // the original predicate (default path, PXA_KV_SEQ_SOA unset)
    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    // same predicate, answered from the inline shadow; falls back to the set only for multi-id cells
    bool has_seq_fast(const llama_seq_id & id) const {
        return seq0 >= 0 ? seq0 == id : (seq0 == SEQ_MULTI && seq_id.find(id) != seq_id.end());
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool do_copy   = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool hybrid    = false;
    bool v_trans   = true;  // the value tensor is transposed

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;
    std::vector<struct ggml_tensor *> s_l; // per layer recurrent state storage (Qwen3Next)

    // When true, the delta_net graph builder will enable per-step SSM state saves
    bool save_per_step_ssm = false;

    std::vector<llama_split_tensor> split_k_l;
    std::vector<llama_split_tensor> split_v_l;
    std::vector<llama_split_tensor> split_s_l;

    // Per-device replicas of the MLA compressed-latent KV cache (-sm graph for DEEPSEEK2/GLM_DSA/MISTRAL4).
    std::vector<llama_split_tensor> replicated_k_l;

    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    size_t total_size() const {
        size_t size = 0;
        for (ggml_backend_buffer_t buf : bufs) {
            size += ggml_backend_buffer_get_size(buf);
        }
        return size;
    }

    // GPU-resident checkpoint for recurrent/hybrid speculative decoding
    struct gpu_checkpoint {
        std::vector<llama_kv_cell> cells_snapshot;
        uint32_t head_snapshot = 0;
        uint32_t used_snapshot = 0;

        std::vector<ggml_tensor *> s_l_shadow;

        std::vector<std::vector<ggml_tensor *>> split_s_l_shadow;

        // Per-step SSM state checkpoints for speculative decoding.
        std::vector<std::vector<ggml_tensor *>> per_step_ssm;

        // Per-step conv feature buffer: stores qkv_mixed features from the
        // verification forward pass so conv state can be reconstructed at any step.
        // One tensor per recurrent layer, each sized [conv_dim * max_tokens].
        //std::vector<std::vector<ggml_tensor *>> per_step_qkv;
        std::vector<std::vector<ggml_tensor *>> per_step_conv;

        int32_t per_step_n_tokens = 0;
        int32_t per_step_max_allocated = 0;
        int64_t per_step_ssm_state_size = 0;
        int64_t per_step_conv_state_dim = 0;
        int64_t per_step_conv_dim = 0;
        int32_t per_step_d_conv = 0;

        // PXA_MULTISEQ_CKPT: identity of the LAST per-step capture, recorded by llama_set_inputs
        // (next to the inp_s_seq_qnext fill) whenever save_per_step_ssm is armed and the batch
        // qualifies (fits the per-step buffers AND is decomposable with a uniform tokens-per-seq
        // count). The per-step snapshots are laid out [step][batch_pos][state-row] with a step
        // stride of last_capture_n_seqs rows; per_step_restore() translates an absolute seq_id to
        // batch_pos via last_capture_seqs (the batch's distinct seq ids in first-seen/gathered
        // order). last_capture_n_seqs == 0 -> the last decode produced NO valid capture and any
        // per-step restore must fail closed.
        std::vector<llama_seq_id> last_capture_seqs;
        int32_t last_capture_n_seqs = 0;
        int32_t last_capture_steps  = 0; // tokens per sequence (n_seq_tokens) in the captured batch

        int selected_spec_mode = -1;
        int fixed_spec_mode = LLAMA_SPEC_CKPT_NONE;
        int32_t fixed_max_tokens = 0;

        // Serialised sequence state for CPU mode
        std::vector<uint8_t> cpu_state_data;
        // PXA_LLAMA_MTP_NP3B_FIX: at np>1 the spec-checkpoint save loop runs once PER active slot before
        // the shared verify decode, so a single shared cpu_state_data buffer is clobbered by the last slot
        // to save -> every slot then RESTORES the last slot's recurrent state into its own row (cross-
        // conversation bleed + garbage under concurrent MTP). Key the CPU checkpoint per seq_id so each
        // slot save/restore is isolated. (Single-slot is unchanged: one entry at seq_id 0.)
        std::map<llama_seq_id, std::vector<uint8_t>> cpu_state_by_seq;

        // Separate storage for per-step allocations
        std::vector<struct ggml_context *>   per_step_ctxs;
        std::vector<ggml_backend_buffer_t>   per_step_bufs;

        std::vector<struct ggml_context *>   shadow_ctxs;
        std::vector<ggml_backend_buffer_t>   shadow_bufs;

        bool allocated = false;
        bool shadow_conv_only = false;
        bool saved     = false;

        ~gpu_checkpoint() {
            for (struct ggml_context * ctx : shadow_ctxs) {
                ggml_free(ctx);
            }
            for (ggml_backend_buffer_t buf : shadow_bufs) {
                ggml_backend_buffer_free(buf);
            }
            for (struct ggml_context * ctx : per_step_ctxs) {
                ggml_free(ctx);
            }
            for (ggml_backend_buffer_t buf : per_step_bufs) {
                ggml_backend_buffer_free(buf);
            }
        }
    };

    gpu_checkpoint ckpt;

    bool checkpoint_alloc_shadows(bool conv_only_shadow = false);
    bool checkpoint_supported() const;
    bool checkpoint_save(ggml_backend_sched_t sched);
    bool checkpoint_restore(ggml_backend_sched_t sched);
    void checkpoint_delete();

    // Per-step checkpoint: allocate, restore step k's full state (SSM + conv) to cache
    bool per_step_alloc(const llama_model & model, int max_tokens);
    // PXA_PER_SEQ_CKPT: seq_id >= 0 restricts the restore to that sequence's single recurrent
    // state row (s_l row == seq_id); -1 = legacy all-rows restore (np=1-only semantics).
    bool per_step_restore(const llama_model & model, ggml_backend_sched_t sched, int step, llama_seq_id seq_id = -1);

    ~llama_kv_cache() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

struct llama_control_vector {
    std::vector<struct ggml_tensor *> tensors; // per layer
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    // PXA: directional ablation mode — project the (unit) dir OUT of the residual
    // (cur -= alpha * dir (dir^T cur)) instead of adding it; for refusal removal with
    // all weights frozen (Arditi 2406.11717). Set via llama_control_vector_set_ablation().
    bool  ablation       = false;
    float ablation_alpha = 1.0f;

    struct ggml_tensor * tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            if (ablation) {
                // project dir out of the residual: cur -= alpha * dir (dir^T cur)  (dir unit-norm)
                struct ggml_tensor * coeffs = ggml_mul_mat(ctx, layer_dir, cur);     // [1, n_tokens]
                struct ggml_tensor * proj   = ggml_mul(ctx,
                                                  ggml_repeat(ctx, layer_dir, cur),
                                                  ggml_repeat(ctx, coeffs, cur));     // [n_embd, n_tokens]
                proj = ggml_scale(ctx, proj, ablation_alpha);
                cur  = ggml_sub(ctx, cur, proj);
            } else {
                cur = ggml_add(ctx, cur, layer_dir);
            }
        }
        return cur;
    }

    ~llama_control_vector() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

// PXA_SCHED_PIPELINE: the umbrella switch for cross-ubatch device pipelining. OFF by
// default. ON it turns on all three parts at once, because any two of them without the
// third are either unsafe or worthless:
//
//   1. scheduler copy slots (n_copies = GGML_SCHED_MAX_COPIES = 2), via
//      pxa_pipeline_pp_enabled() below, so consecutive micro-batches do not share the
//      storage of the user graph inputs;
//   2. PXA_SCHED_ASYNC_INPUTS, so no split boundary blocks the host and micro-batch k+1's
//      first split can be enqueued while k's last split is still on the other card;
//   3. the graph-input slot wait in ggml_backend_sched_alloc_graph, which is what makes
//      (2) safe (see the root-cause note there).
//
// Each part still has its own env var and an explicit setting of that var wins, so an arm
// can isolate any one of them.
static inline bool pxa_sched_pipeline_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("PXA_SCHED_PIPELINE");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

// PXA_PIPELINE_PP: prefill pipeline parallelism (scheduler n_copies > 1) plus the in-graph
// PLE conv-history carry it needs.
//
// DEFAULT ON since 2026-09-03, but only for the arches on the caller's allow-list (qwen35 /
// qwen35moe / qwen4exp -- see the pxa_pp_default_on computation at the llama_new_context site).
// Every other arch keeps n_copies = 1 unless PXA_PIPELINE_PP is set explicitly.
//
// Why it flipped: with the mid-prompt context-checkpoint drains gone
// (PXA_CKPT_PROMPT_POLICY / PXA_CKPT_LEAN_SYNC) the copy slots finally buy something. On
// 2x V100, Qwable-27B-PXQ4core, 20801 prefill at -ub 512, default -ts, median of 2:
//   checkpoint levers on, PP off        1074.3 t/s   busy 1.340
//   checkpoint levers on, PP copies=2   1085.3 t/s   busy 1.330
// and in the final bench rows 1083.20 t/s at ub512 / 932.13 at ub2048 against 971.49 / 848.56
// with everything off. The reason it measured flat for three previous lanes is that the host
// was blocked 4-5 times per ubatch by the checkpoint drains and the compute_splits input
// syncs; at n_copies > 1 with those gone, BOTH compute_splits blockers disappear entirely
// (they become a slot-event wait costing 1.5 ms in total across 41 ubatches).
//
// pxa_pipeline_pp_env() reports what the environment asked for, so the arch default can be
// applied without losing an explicit override:
//   -1 = neither PXA_PIPELINE_PP nor PXA_SCHED_PIPELINE set -> use the caller's arch default
//    0 = explicitly forced off
//    1 = explicitly forced on
static inline int pxa_pipeline_pp_env() {
    static const int v = [] {
        const char * e = getenv("PXA_PIPELINE_PP");
        if (e != nullptr && *e != '\0') {
            return atoi(e) != 0 ? 1 : 0;
        }
        if (getenv("PXA_SCHED_PIPELINE") != nullptr) {
            return pxa_sched_pipeline_enabled() ? 1 : 0;
        }
        return -1;
    }();
    return v;
}

static inline bool pxa_pipeline_pp_enabled(bool default_on = false) {
    const int e = pxa_pipeline_pp_env();
    return e < 0 ? default_on : (e != 0);
}

struct llama_context {

    llama_context(const llama_model & model);

    ~llama_context();

    const struct llama_model & model;

    struct llama_cparams        cparams;
    struct llama_sampling       sampling;
    struct llama_kv_cache       kv_self;

    // PXA_SWA_KV: interleaved-SWA KV allocation (opt-in, env PXA_SWA_KV=1).
    //
    // When armed, the model's sliding-window layers get their OWN cache object with its own
    // cells/head/size/n/slot-allocator, sized to the attention window instead of the full context.
    // kv_self then allocates K/V only for the full-attention layers and kv_swa only for the sliding
    // ones (the other layers' k_l/v_l entries are nullptr in each). Two SEPARATE index spaces is the
    // point: it removes the cell-index aliasing that a per-layer ring inside the single shared index
    // space would introduce, rather than trying to mask it.
    //
    // swa_kv_active == false reproduces the original single-cache behaviour exactly.
    bool                        swa_kv_active = false;
    uint32_t                    swa_kv_guard  = 0; // eviction guard band, in positions
    struct llama_kv_cache       kv_swa;

    // true if layer il's K/V lives in kv_swa rather than kv_self
    bool swa_kv_layer(int il) const;

    // the cache that owns layer il
    const struct llama_kv_cache & kv_for_layer(int il) const {
        return swa_kv_layer(il) ? kv_swa : kv_self;
    }
    struct llama_kv_cache & kv_for_layer(int il) {
        return swa_kv_layer(il) ? kv_swa : kv_self;
    }

    struct llama_context      * mtp_target_ctx   = nullptr;
    struct llama_control_vector cvec;

    std::vector<float> scale_data;

    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

    std::vector<ggml_backend_t> backends;
#ifdef GGML_USE_METAL
    ggml_backend_t backend_metal = nullptr;
#endif
#ifdef GGML_USE_BLAS
    ggml_backend_t backend_blas = nullptr;
#endif
    ggml_backend_t backend_cpu = nullptr;

    bool has_evaluated_once = false;

    int64_t t_start_us;
    int64_t t_load_us;
    int64_t t_p_eval_us = 0;
    int64_t t_eval_us   = 0;

    int64_t t_compute_start_us = 0;
    int64_t n_queued_tokens = 0;

    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    int32_t n_eval   = 0; // number of eval calls

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_t buf_output = nullptr;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch
    int32_t n_outputs_embd = 0; // number of embedding rows produced for the current logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    const float * draft_input_hidden_state = nullptr;
    size_t draft_input_hidden_state_n_floats = 0;
    std::vector<float> draft_input_hidden_state_owned;

    // input tensors
    struct ggml_tensor * inp_tokens;      // I32 [n_batch]
    struct ggml_tensor * inp_embd;        // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;         // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;     // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;     // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_dspark_cap = nullptr; // F32 [n_capture*n_embd, 1]
    std::vector<float>   dspark_cap_host;          // host side of the above

    // M4 - the Markov chain's seed. It is the token the TARGET just committed, i.e. the
    // `token` argument of ds4_session_prepare_dspark_draft_impl (ds4.c:59232). Deliberately
    // NOT a view of inp_tokens[1]: that view happens to carry the same value today, but it
    // welds the chain to the block-seeding convention and to a contiguity contract for no
    // gain.
    struct ggml_tensor * inp_dspark_prev = nullptr; // I32 [1]
    int32_t              dspark_prev_host = 0;

    // M4 outputs, extracted once per drafter forward (see llama_dspark_extract_draft).
    // conf holds the confidence head's RAW LOGIT, not sigmoid(x): sigmoid is monotone so
    // the threshold comparison is identical, and leaving it out of the graph removes an op
    // and a rounding step.
    std::vector<llama_token> dspark_draft_ids;
    std::vector<float>       dspark_draft_conf;
    bool                     dspark_draft_ok  = false;
    int64_t                  t_dspark_read_us = 0;   // cost of the per-cycle D2H readbacks
    int64_t                  n_dspark_read    = 0;

    // M5 - the TARGET side of the capture. build_deepseek4() emits dspark_cap_%d nodes
    // under PXA_DSPARK_CAPTURE; this is where they land so a caller can hand them to the
    // drafter without a file. Without it the loop can only ever run on a synthetic
    // capture, which says nothing about acceptance.
    std::vector<float>       dspark_cap_out;
    bool                     dspark_cap_out_ok = false;
    struct ggml_tensor * inp_KQ_mask_swa; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_K_shift;     // I32 [kv_size]
    struct ggml_tensor * inp_K_shift_swa; // I32 [kv_swa.size]  (PXA_SWA_KV)
    struct ggml_tensor * inp_mean;        // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;         // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;      // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;      // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;       // I32 [n_kv, n_batch]
    // qwen4exp PLE: I32 [ple_n_heads * n_batch]. One gather row per (token, head) into
    // per_layer_token_embd. Filled host-side in llama_set_inputs because the hash needs
    // int64 multiply and xor, neither of which ggml has.
    // = nullptr is load-bearing, not tidiness. This is assigned ONLY by
    // build_qwen4exp(), but llama_set_inputs reads it for EVERY architecture. Left
    // uninitialised it holds indeterminate heap bytes on any other model, and the
    // `if (ptr && ptr->buffer)` guard then dereferences a wild pointer. Observed:
    // a 0.6B dense model aborted on the PLE n_gram assert in 1 run of 3, and passed
    // the other 2 -- the value tracked heap layout, so relinking the library changed
    // the outcome. A segfault was equally available.
    struct ggml_tensor * inp_ple_rows = nullptr;

    // ---- qwen4exp PLE per-sequence conv history --------------------------------------
    // The PLE conv is dilated and causal: out[c, t] reads the conv input at
    // t, t-dil, t-2*dil, t-3*dil. Those positions are not recomputable once the layer is
    // done (they depend on the residual there), so the last hist = (conv_kernel-1)*ngram
    // columns of every LIVE SEQUENCE are carried in this one device-resident tensor:
    //
    //     F32 [hc_dim, 1 + n_seq_max*hist]
    //       column 0                    : permanently zero. It is the "no such position"
    //                                     sentinel every gather falls back to - before a
    //                                     sequence's first token, or after a rewind that
    //                                     left the window pointing somewhere else. Keeping
    //                                     one zero column costs 40 KiB and removes every
    //                                     special case a keep/reset flag used to carry.
    //       column 1 + s*hist + j       : sequence s, position ple_seq[s].next_pos-hist+j
    //
    // It used to be ONE window for the whole context plus a scalar "keep" flag, which is
    // why a batch spanning two sequences had to abort: the conv taps were plain shifted
    // VIEWS of concat(window, ubatch), and a view does not know where one sequence ends and
    // the next begins. The taps are gathers now (inp_ple_conv_src), so a ragged batch is
    // just a different index vector.
    struct ggml_context * ctx_ple_capture   = nullptr;
    ggml_backend_buffer_t buf_ple_capture   = nullptr;
    struct ggml_tensor  * ple_conv_hist_dev = nullptr;

    // I32 [conv_kernel * n_tokens]: source column in concat(ple_conv_hist_dev, conv_in) for
    // tap k of token t, at idx[k*n_tokens + t]. Filled host-side in llama_set_inputs, which
    // is the only place that sees each token's sequence and position.
    // = nullptr is load-bearing for the same reason inp_ple_rows above is: graphs that do
    // not rebuild the PLE inputs (the MTP draft graph, k_shift, defrag) must see a null
    // pointer rather than the previous graph's arena address.
    struct ggml_tensor * inp_ple_conv_src   = nullptr;
    // I32 [n_seq_max*hist]: the same gather, used once at the end of the graph to rewrite
    // every sequence's window from concat(window, conv_in). Sequences absent from this
    // ubatch get identity indices, so one copy preserves them and advances the rest.
    struct ggml_tensor * inp_ple_conv_carry = nullptr;

    // Host-side bookkeeping for the window above, one entry per sequence slot. next_pos is
    // the position the window ENDS at (it holds [next_pos-hist, next_pos-1]); toks is the
    // last (n_gram-1) token ids, ending at next_pos-1, for the n-gram hash of the tokens
    // that come next. Nothing here is invalidated by a rewind: both are POSITION-TAGGED, so
    // a request that does not continue where the slot left off simply finds no matching
    // column and reads zeros / EOS padding, which is what a fresh sequence gets. An exact
    // resume instead comes from the server's prompt checkpoint, which round-trips this
    // struct through llama_state_seq_get_data / set_data.
    struct ple_seq_state {
        llama_pos                next_pos = 0;
        bool                     valid    = false;
        std::vector<llama_token> toks;
    };
    std::vector<ple_seq_state> ple_seq;

    // hist and hc_dim, cached from the graph builder so the seq_* hooks and the state
    // serialiser can address ple_conv_hist_dev without reaching into hparams.
    int32_t ple_hist_cols = 0;
    int32_t ple_hc_dim    = 0;

    struct ggml_tensor * inp_s_seq_qnext; // I32 [1, n_batch]
    struct ggml_tensor * inp_conv_seq_map;  // PXA_LLAMA_FIX_v4: I32 [n_batch, n_batch] conv seq-map for batched mixed-seq delta-net
    struct ggml_tensor * inp_qnext_state_mask; // PXA_LLAMA_FIX_v4: F32 [1, n_batch] per-seq recurrent-state reset mask (0=reset)
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_scale = nullptr; // F32 [n_tokens]
    struct ggml_tensor * inp_mtp_states = nullptr;

    ggml_backend_t ggml_backend_by_name(const char * name);

    struct Prev;
    std::unique_ptr<Prev> prev;
    std::unique_ptr<Prev> prev_mtp;

    void reset_scheduler();
    bool can_reuse_graph(const llama_batch & u_batch);

    // PXA_SCHED_RESERVE_REAL: identity of the plan galloc currently holds, as installed by
    // pxa_reserve_real_graph (real decode path, KV window wide open). A ubatch whose graph does
    // not fit this plan would make ggml_backend_sched_alloc_splits re-plan - on that ubatch's
    // own narrow sizes - so the decode loop re-reserves instead. Three keys, because any one of
    // them alone lets a narrow plan survive:
    //   nodes   - the shape. Necessary, not sufficient: a 2-token graph and a 2048-token graph
    //             have the SAME node count, so node count alone accepted the warmup's 2-token
    //             plan as covering the whole prefill.
    //   width   - the ubatch the plan was built at. A wider ubatch overflows the per-node sizes.
    //   replans - ggml_backend_sched_get_n_replans() right after the reserve. If it has moved,
    //             something else (a warmup decode, a K-shift) re-planned and the plan in galloc
    //             is no longer the one reserved here, whatever its node count.
    int  pxa_reserved_nodes   = -1;
    int  pxa_reserved_width   =  0;
    int  pxa_reserved_replans = -1;
    bool pxa_reserve_sizes_logged = false;

    struct CacheCopy {
        ggml_tensor * cpy = nullptr;
        size_t        step = 0;
    };
    std::vector<CacheCopy> cache_copies;

    bool update_cache_copies();

    bool prepare_mtp_graph_inputs(
        struct llama_context & lctx);
    void set_mtp_op_type(llama_mtp_op_type value);

    int max_nodes(int n_tokens, int n_kv) const;

};

