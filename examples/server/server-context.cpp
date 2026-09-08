#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-task.h"
#include "server-queue.h"

#include "common.h"
#include "ggml.h"      // ggml_pxa_config_level (the PXA master config level)
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "pxa-expert-log.hpp"

#include <algorithm> // PXA_KV_UNIFIED_v1: std::sort over reclaim victims
#include <fstream>
#include <iostream>
#include <regex>
#include <exception>

static void server_prompt_checkpoint_update(server_prompt_checkpoint & ckpt, llama_context * ctx, int id, int64_t n_tokens, llama_pos pos_min = -1, llama_pos pos_max = -1, int32_t offset = 0) {
    if (pos_min == -1) {
        pos_min = llama_kv_cache_seq_pos_min(ctx, id);
    }
    if (pos_max == -1) {
        pos_max = llama_kv_cache_seq_pos_max(ctx, id);
    }
    ggml_backend_sync_tag_push("server:create_checkpoint_size");
    const size_t checkpoint_size = llama_state_seq_get_size(ctx, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    ggml_backend_sync_tag_pop();

    ckpt.pos_min = pos_min;
    ckpt.pos_max = pos_max;
    ckpt.pos_max_prompt = pos_max + offset;
    ckpt.pos_min_prompt = pos_min + offset;
    ckpt.n_tokens = n_tokens;
    ckpt.data.resize(checkpoint_size);

    // PXA_CKPT_APPLY_SYNC (PXA_KERNEL_FIX_20260611): drain all in-flight backend work before the
    // bulk D2H state read. With pipeline scheduling (GGML_SCHED_MAX_COPIES=4, 3-device -sm layer)
    // a prior decode's kernels can still be running when the server thread reaches this point;
    // reading (here) or writing (apply_checkpoint) the 60-130MiB recurrent/KV state while kernels
    // are in flight is a data race and the prime suspect for the sporadic sticky
    // "CUDA illegal memory access" surfacing at the next CUDA call (prepare_row_mappings et al).
    // PXA_CKPT_LEAN_SYNC: llama_state_seq_get_data() drains the devices itself immediately
    // below, with no decode in between, so this drain is a second full two-device stall for
    // nothing. Armed, only the one inside the state read survives (the guard the comment above
    // is actually about); unarmed, behaviour is byte-for-byte what it was.
    ggml_backend_sync_tag_push("server:create_checkpoint");
    if (!llama_pxa_ckpt_lean_sync()) {
        llama_synchronize(ctx);
    }
    const size_t n = llama_state_seq_get_data(ctx, ckpt.data.data(), checkpoint_size, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    ggml_backend_sync_tag_pop();
    if (n != checkpoint_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
    }
}

static void log_text(const gpt_params & params_base, const std::string & text) {
    if (params_base.minilog) {
        LOG_TEE("%s\n", text.c_str());
    }
}

static bool params_use_gemma4_external_mtp(const gpt_params & params_base) {
    return params_base.has_mtp &&
        llama_model_is_gemma4_mtp_assistant(params_base.speculative.model_dft);
}

struct server_mtp_warmup {
    llama_context * ctx_tgt;
    server_slot   * slot;
};

static int32_t server_mtp_media_warmup_callback(void * user_data, const llama_batch * batch) {
    auto * data = static_cast<server_mtp_warmup *>(user_data);
    if (data == nullptr || data->slot == nullptr) {
        return 0;
    }

    return batch != nullptr
        ? common_speculative_on_target_seq_batch(data->slot->spec, data->ctx_tgt, *batch, data->slot->id, true)
        : 0;
}

static bool server_response_needs_chat_parse(oaicompat_type oaicompat) {
    return oaicompat == OAICOMPAT_TYPE_CHAT ||
        oaicompat == OAICOMPAT_TYPE_ANTHROPIC ||
        oaicompat == OAICOMPAT_TYPE_RESP;
}

void server_speculative_checkpoint::clear() {
    valid = false;
    per_step_enabled = false;
    n_past = 0;
    sampled = LLAMA_TOKEN_NULL;

    // PXA_SPEC_CKPT_SAMPLER_REUSE: keep the sampler allocation. clear() runs once per verify STEP
    // (save_speculative_checkpoint), and common_sampler_clone() fully overwrites the destination
    // (params, rng, grammar, DRY, adaptive-p, rbudget, prev), so the object is safely reusable.
    // Re-initializing per step was pure allocation churn — and with DRY enabled,
    // common_sampler_init() re-processes the DRY sequence breakers against the WHOLE vocab,
    // which on a 248K-vocab model costs ~34 ms/step (measured 54 -> 19.5 tok/s). The per-slot
    // sampler lives for the server lifetime (bounded by n_parallel).
}

// PXA_CKPT_STEP_SCOPED_DISCARD (PXA_KERNEL_FIX_20260611): llama_spec_ckpt_discard() is CTX-GLOBAL
// (clears save_per_step_ssm, resets selected_spec_mode to NONE and deletes the shadow checkpoint).
// It used to be called PER SLOT inside the verify-accept loop, so with 2+ slots co-decoding in one
// uniform batch the FIRST slot's discard destroyed the step's checkpoint before the remaining
// slots ran their rejected-draft restore: llama_spec_ckpt_restore() then hit mode==NONE and failed
// (fail-closed) -> the later slots kept the rejected drafts' recurrent-state contamination every
// step with any rejection. This was latent while PXA_LLAMA_MTP_NP4_FIX serialized all decoding
// (co-decode never actually happened); PXA_PROMPT_INTERLEAVE_v1 restores true co-decode, so the
// discard MUST be step-scoped: per-slot teardown only clears the slot's own bookkeeping, and the
// caller (speculative_decoding_accept) issues ONE ctx-global discard after ALL slots finished
// their accept/restore work for the step.
static void discard_speculative_checkpoint(server_slot & slot, llama_context * /*ctx*/) {
    slot.spec_ckpt.clear();
}

// PXA_SPEC_TRACE: aggregate CPU-wall decomposition of the speculative round. The GPU-op profiler
// showed MTP mode adds only ~5% GPU time while costing ~27% decode, so the tax is host-side; this
// names it. A "round" is one update_slots tick where at least one slot verified a draft. Phases:
//   draft  = common_speculative_draft (K MTP forwards + sampling + hidden copies)
//   ckpt   = pre-verify recurrent checkpoint save (llama_synchronize + bulk state read)
//   verify = the target llama_decode calls of the tick
//   accept = speculative_decoding_accept in full (sample+accept, restore/commit decodes)
//   commit = the restore/commit sub-block inside accept (subset of accept)
//   other  = tick wall minus the above (batch build, slot walks, token emit)
// update_slots runs on one thread, so plain statics suffice. Env PXA_SPEC_TRACE=1 turns it on,
// PXA_SPEC_TRACE_EVERY (default 64) sets the dump period. Off = zero cost (one getenv-cached bool).
struct pxa_spec_trace_state {
    bool    on;
    int     every;
    int     rounds       = 0;
    bool    tick_spec    = false;
    int64_t tick_t0      = 0;
    int64_t tk[5]        = {0,0,0,0,0}; // draft, ckpt, verify, accept, commit (this tick)
    int64_t ac[6]        = {0,0,0,0,0,0}; // accumulated: draft, ckpt, verify, accept, commit, tick
    pxa_spec_trace_state() {
        on    = getenv("PXA_SPEC_TRACE") && atoi(getenv("PXA_SPEC_TRACE")) != 0;
        every = getenv("PXA_SPEC_TRACE_EVERY") ? std::max(1, atoi(getenv("PXA_SPEC_TRACE_EVERY"))) : 64;
    }
};
static pxa_spec_trace_state pxa_sptr;

enum { PXA_SPTR_DRAFT = 0, PXA_SPTR_CKPT, PXA_SPTR_VERIFY, PXA_SPTR_ACCEPT, PXA_SPTR_COMMIT };

static inline void pxa_sptr_tick_begin() {
    if (!pxa_sptr.on) return;
    pxa_sptr.tick_t0   = ggml_time_us();
    pxa_sptr.tick_spec = false;
    for (auto & v : pxa_sptr.tk) v = 0;
}

static inline void pxa_sptr_add(int phase, int64_t t0) {
    if (!pxa_sptr.on) return;
    pxa_sptr.tk[phase] += ggml_time_us() - t0;
}

static inline void pxa_sptr_tick_end() {
    if (!pxa_sptr.on || !pxa_sptr.tick_spec || pxa_sptr.tick_t0 == 0) return;
    for (int i = 0; i < 5; ++i) pxa_sptr.ac[i] += pxa_sptr.tk[i];
    pxa_sptr.ac[5] += ggml_time_us() - pxa_sptr.tick_t0;
    if (++pxa_sptr.rounds % pxa_sptr.every == 0) {
        const double n = (double) pxa_sptr.every;
        const double tick   = pxa_sptr.ac[5] / n / 1e3;
        const double draft  = pxa_sptr.ac[PXA_SPTR_DRAFT]  / n / 1e3;
        const double ckpt   = pxa_sptr.ac[PXA_SPTR_CKPT]   / n / 1e3;
        const double verify = pxa_sptr.ac[PXA_SPTR_VERIFY] / n / 1e3;
        const double accept = pxa_sptr.ac[PXA_SPTR_ACCEPT] / n / 1e3;
        const double commit = pxa_sptr.ac[PXA_SPTR_COMMIT] / n / 1e3;
        fprintf(stderr,
            "PXA_SPEC_TRACE rounds=%d avg_ms tick=%.2f draft=%.2f ckpt=%.2f verify=%.2f accept=%.2f (commit=%.2f) other=%.2f\n",
            pxa_sptr.rounds, tick, draft, ckpt, verify, accept, commit,
            tick - draft - ckpt - verify - accept);
        for (auto & v : pxa_sptr.ac) v = 0;
    }
}

static bool save_speculative_checkpoint(server_slot & slot, llama_model * model, llama_context * ctx, int ckpt_mode) {
    slot.spec_ckpt.clear();
    const int32_t n_pre_spec_tokens = slot.cache_tokens.n_tokens() - (int32_t)(slot.drafted.size() + 1);
    slot.spec_ckpt.n_past = slot.cache_tokens.pos_next(n_pre_spec_tokens);
    slot.spec_ckpt.sampled = slot.sampled;

    const int max_tokens = (int)slot.drafted.size() + 1;
    const int actual_mode = llama_spec_ckpt_init(ctx, ckpt_mode, max_tokens);
    if (actual_mode == LLAMA_SPEC_CKPT_NONE) {
        return false;
    }
    slot.spec_ckpt.per_step_enabled = (actual_mode == LLAMA_SPEC_CKPT_PER_STEP);

    slot.spec_ckpt.valid = llama_spec_ckpt_save(ctx, slot.id);
    if (!slot.spec_ckpt.valid) {
        // PXA_CKPT_STEP_SCOPED_DISCARD: a failed save invalidates only THIS slot's checkpoint
        // (per-step saves cannot fail; only the per-seq CPU path can). A ctx-global discard here
        // would wipe a sibling slot's just-saved checkpoint in the same pre-verify loop.
        slot.spec_ckpt.clear();
        return false;
    }

    // PXA_SPEC_CKPT_SAMPLER_REUSE: allocate once per slot; common_sampler_clone below overwrites
    // every field, so a reused object is state-identical to a fresh init.
    if (slot.spec_ckpt.sampler == nullptr) {
        slot.spec_ckpt.sampler = common_sampler_init(model, slot.sparams);
    }
    if (slot.spec_ckpt.sampler == nullptr) {
        discard_speculative_checkpoint(slot, ctx);
        return false;
    }

    common_sampler_clone(slot.ctx_sampling, slot.spec_ckpt.sampler);
    return true;
}

static void server_remove_speculative_stage(common_params_speculative & spec, common_speculative_type type) {
    spec.stages.erase(std::remove_if(spec.stages.begin(), spec.stages.end(), [type](const common_speculative_stage_params & stage) {
        return stage.type == type;
    }), spec.stages.end());

    if (spec.type == type) {
        spec.type = COMMON_SPECULATIVE_TYPE_NONE;
        const auto resolved = spec.get_resolved_stages();
        spec.type = resolved.empty() ? COMMON_SPECULATIVE_TYPE_NONE : resolved.front().type;
    }
}

static bool server_speculative_needs_draft_model(const common_params_speculative & spec) {
    return spec.has_stage_type(COMMON_SPECULATIVE_TYPE_DRAFT);
}

static bool server_speculative_has_mtp(const common_params_speculative & spec) {
    return spec.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
}

static bool server_speculative_same_stage_types(
        const common_params_speculative & lhs,
        const common_params_speculative & rhs) {
    const auto lhs_stages = lhs.get_resolved_stages();
    const auto rhs_stages = rhs.get_resolved_stages();

    if (lhs_stages.size() != rhs_stages.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs_stages.size(); ++i) {
        if (lhs_stages[i].type != rhs_stages[i].type) {
            return false;
        }
    }

    return true;
}

static void server_reject_dead_speculative_request_overrides(const json & data) {
    if (json_value_ptr(data, "speculative.type") != nullptr) {
        throw std::runtime_error("Error: speculative.type request override is not supported; keep the startup stage types and use speculative.stages or speculative.n_max/n_min/p_min");
    }

    if (json_value_ptr(data, "speculative.ngram_size_n") != nullptr ||
        json_value_ptr(data, "speculative.ngram_size_m") != nullptr ||
        json_value_ptr(data, "speculative.ngram_min_hits") != nullptr ||
        json_value_ptr(data, "speculative.suffix_min_match_len") != nullptr ||
        json_value_ptr(data, "speculative.suffix_max_depth") != nullptr ||
        json_value_ptr(data, "speculative.suffix_corpus") != nullptr) {
        throw std::runtime_error("Error: structural speculative overrides are startup-only; per-request overrides only support speculative.n_max, speculative.n_min, speculative.p_min, and speculative.stages");
    }
}

static common_speculative_stage_params server_parse_speculative_stage_json(const json & stage_json) {
    if (!stage_json.is_object()) {
        throw std::runtime_error("Error: speculative.stages entries must be objects");
    }
    if (!stage_json.contains("type") || !stage_json["type"].is_string()) {
        throw std::runtime_error("Error: speculative.stages entries must include a string 'type'");
    }

    common_speculative_stage_params stage;
    stage.type = common_speculative_type_from_name(stage_json["type"].get<std::string>());
    if (stage.type == COMMON_SPECULATIVE_TYPE_COUNT) {
        throw std::runtime_error("Error: unknown speculative stage type in speculative.stages");
    }

    for (const auto & item : stage_json.items()) {
        if (item.key() == "type") {
            continue;
        }

        if (item.key() == "n_max") {
            stage.n_max = item.value().get<int32_t>();
            if (stage.n_max < 0) {
                throw std::runtime_error("Error: speculative.stages[].n_max must be >= 0");
            }
            continue;
        }

        if (item.key() == "n_min") {
            stage.n_min = item.value().get<int32_t>();
            if (stage.n_min < 0) {
                throw std::runtime_error("Error: speculative.stages[].n_min must be >= 0");
            }
            continue;
        }

        if (item.key() == "p_min") {
            stage.p_min = item.value().get<float>();
            if (stage.p_min < 0.0f) {
                throw std::runtime_error("Error: speculative.stages[].p_min must be >= 0");
            }
            continue;
        }

        throw std::runtime_error("Error: per-request speculative.stages only support type, n_max, n_min, and p_min; structural stage overrides are startup-only");
    }

    return stage;
}

server_context::~server_context() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }

    if (model) {
        llama_free_model(model);
        model = nullptr;
    }
    // Free multimodal
    mtmd_free(mctx);
    // Free draft model and context if they exist
    if (ctx_draft) {
        llama_free(ctx_draft);
        ctx_draft = nullptr;
    }
    if (model_draft) {
        llama_free_model(model_draft);
        model_draft = nullptr;
    }

    // Clear any sampling context
    for (server_slot& slot : slots) {
        if (slot.ctx_sampling != nullptr) {
            common_sampler_free(slot.ctx_sampling);
        }
        slot.spec_ckpt.clear();
        if (slot.ctx_dft) {
            llama_free(slot.ctx_dft);
        }
        // PXA_SHARED_MTP_v1: a non-owning alias of the shared MTP companion must NOT be freed here
        // (it would double-free the single shared ctx_mtp). Only the owning slot (slot 0 for MTP, or
        // any per-slot non-MTP spec) frees its spec. Null the alias first so nothing dangles.
        if (slot.owns_spec) {
            common_speculative_free(slot.spec);
        }
        slot.spec = nullptr;
        llama_batch_free(slot.batch_spec);
    }

    llama_batch_free(batch);
}

bool server_context::load_model(const gpt_params& params_) {
    params_base = params_;

    // PXA_EXPERT_LOG_v1: env-gated per-request MoE expert-routing histograms
    if (pxa_expert_log_enabled()) {
        params_base.cb_eval           = pxa_expert_log_cb;
        params_base.cb_eval_user_data = nullptr;
        fprintf(stderr, "PXA_EXPERT_LOG: enabled, dir=%s (np1 only!)\n", getenv("PXA_EXPERT_LOG"));
    }

    llama_init_result llama_init = llama_init_from_gpt_params(params_base);

    model = llama_init.model;
    ctx = llama_init.context;
    lora_adapters = llama_init.lora_adapters;

    if (model == nullptr) {
        LOG_ERROR("unable to load model", { {"model", params_base.model} });
        return false;
    }

    n_ctx = llama_n_ctx(ctx);

    add_bos_token = llama_should_add_bos_token(model);
    has_eos_token = llama_add_eos_token(model) != 1;

    if (false /*PXA: MTP np>1 gate stripped for test*/ && params_base.n_parallel > 1 && server_speculative_has_mtp(params_base.speculative)) {
        LOG_WARNING("MTP is not supported with parallel slots yet, removing the MTP stage to avoid cross-slot corruption.\n", {
            {"n_parallel", params_base.n_parallel},
            {"stage_chain", common_speculative_stage_chain_to_str(params_base.speculative)},
        });

        params_base.has_mtp = false;
        server_remove_speculative_stage(params_base.speculative, COMMON_SPECULATIVE_TYPE_MTP);

        if (!server_speculative_needs_draft_model(params_base.speculative)) {
            params_base.speculative.model.clear();
            params_base.speculative.params.clear();
            params_base.speculative.model_dft = nullptr;
        }
    }

    bool has_draft_model = !params_base.speculative.model.empty() || !params_base.speculative.params.empty();
    std::string& mmproj_path = params_base.mmproj.path;
    if (!mmproj_path.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params_base.mmproj_use_gpu;
        mparams.print_timings = false;
        mparams.n_threads = params_base.n_threads_mtmd != -1 ? params_base.n_threads_mtmd
                             : params_base.n_threads_batch != -1 ? params_base.n_threads_batch
                                                                 : params_base.n_threads;
        mparams.flash_attn_type = params_base.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        mparams.verbosity = params_base.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
        mparams.image_min_tokens = params_base.image_min_tokens;
        mparams.image_max_tokens = params_base.image_max_tokens;
        mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
        if (mctx == nullptr) {
            LOG_ERROR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
            return false;
        }
        LOG_INFO("loaded multimodal model, '%s'\n", mmproj_path.c_str());

        //if (params.n_cache_reuse) {
        //    params_base.n_cache_reuse = 0;
        //    SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
        //}

        if (has_draft_model) {
            LOG_ERROR("%s\n", "err: speculative decode is not supported by multimodal");
            return false;
        }
    const auto spec_stages = params_base.speculative.get_resolved_stages();
    const bool multimodal_spec_supported = spec_stages.empty() ||
        (spec_stages.size() == 1 && spec_stages.front().type == COMMON_SPECULATIVE_TYPE_MTP);
    if (!multimodal_spec_supported) {
            params_base.speculative.type = COMMON_SPECULATIVE_TYPE_NONE;
            params_base.speculative.stages.clear();
            params_base.has_mtp = false;
            SRV_WRN("%s\n", "speculative decoding is not supported by multimodal, it will be disabled");
        }
    }
    // Load draft model for speculative decoding if specified
    if (has_draft_model) {
        LLAMA_LOG_INFO("\n\n==================================loading DRAFT model==================================\n\n");

        gpt_params params_dft;
        params_dft.devices      = params_base.speculative.devices;
        params_dft.model        = params_base.speculative.model;
        params_dft.main_gpu     = params_base.main_gpu;
        params_dft.n_gpu_layers = params_base.speculative.n_gpu_layers;
        params_dft.rpc_servers  = params_base.rpc_servers;
        params_dft.cache_type_k = params_base.speculative.cache_type_k.empty() ? params_base.cache_type_k : params_base.speculative.cache_type_k;
        params_dft.cache_type_v = params_base.speculative.cache_type_v.empty() ? params_base.cache_type_v : params_base.speculative.cache_type_v;
        params_dft.flash_attn   = params_base.flash_attn;
        params_dft.k_cache_hadamard = params_base.k_cache_hadamard;
        params_dft.v_cache_hadamard = params_base.v_cache_hadamard;
        if (!params_base.speculative.params.empty()) {
            auto [argc, argv] = parse_command_line("llama-server " + params_base.speculative.params);
            if (!gpt_params_parse(argc, argv, params_dft)) {
                gpt_params_print_usage(argc, argv, params_dft);
                free_command_line(argc, argv);
                return false;
            };
            free_command_line(argc, argv);
        }
        LOG_INFO("", { {"model", params_dft.model} });
        if (params_dft.n_ctx == 0) {
            params_dft.n_ctx = params_base.speculative.n_ctx;
        }
        // PXA_KV_UNIFIED_v1: the draft context must be able to follow the target slot. Under the
        // lever a slot may grow to the full ring, so a draft sized at n_ctx/n_parallel would overrun
        // during long generations. Mirror the target's effective per-slot ceiling. Untouched when
        // the lever is off. (Only reached with a separate -md draft model; the MTP path builds its
        // companion from params_base further down.)
        params_dft.n_ctx = params_dft.n_ctx == 0
                         ? (params_base.kv_unified ? params_base.n_ctx : params_base.n_ctx / params_base.n_parallel)
                         : params_dft.n_ctx;
        params_dft.n_parallel = 1;
        params_dft.n_batch = params_dft.n_ctx;

        params_base.speculative.mparams_dft.path = params_dft.model; //

        llama_model_params mparams_dft = common_model_params_to_llama(params_dft);

        llama_model * model_dft = llama_model_load_from_file(params_dft.model.c_str(), mparams_dft);
        if (model_dft == nullptr) {
            LOG_ERROR("failed to load draft model", { {"model", params_base.speculative.model} });
            return false;
        }

        cparams_dft = common_context_params_to_llama(params_dft);

        params_base.speculative.model_dft = model_dft;
        params_base.speculative.cparams_dft = cparams_dft;

    }
    if (server_speculative_has_mtp(params_base.speculative) &&
        llama_model_n_nextn_layer(model) == 0 &&
        !params_use_gemma4_external_mtp(params_base)) {
        LOG_WARNING("WARNING: MTP speculative stage requested, but model has 0 NextN layers. MTP will be disabled.\n", {});
        params_base.has_mtp = false;
        server_remove_speculative_stage(params_base.speculative, COMMON_SPECULATIVE_TYPE_MTP);
    }
    return true;
}

// PXA_INTERLEAVE_DECODE_MS_v1: how long the OTHER slots get to decode after each prompt chunk.
// PXA_PROMPT_INTERLEAVE_v1 below alternates strictly: one prompt tick, one decode tick. During a
// deep prefill a prompt tick is a whole -b chunk (about 9 s for 2048 tokens at 220 t/s on the
// 4-card 150k seat) while a decode tick costs about 70 ms, so a co-resident short request emitted
// roughly one token per prompt chunk -- measured 0.22 tok/s, a 14-token answer taking 63 s even
// though its own prefill had finished in 477 ms. Run decode-only ticks for this many milliseconds
// after each prompt chunk (or until no slot has decode work left) before taking the next chunk.
// 0 restores the strict one-for-one alternation.
static int32_t pxa_interleave_decode_ms() {
    static const int32_t v = getenv("PXA_INTERLEAVE_DECODE_MS") ? atoi(getenv("PXA_INTERLEAVE_DECODE_MS")) : 1000;
    return v < 0 ? 0 : v;
}

void server_context::init() {
    // PXA_KV_UNIFIED_v1: the ONE line of server arithmetic that statically partitions an already
    // shared, seq-tagged attention-KV ring. With the lever off this is bit-for-bit the historical
    // expression; with it on, every slot may address the full ring and admission control (see
    // kv_unified_admit) takes over the job the division used to do implicitly.
    n_ctx_slot = params_base.kv_unified
               ? n_ctx
               : n_ctx / params_base.n_parallel;

    // Forward-progress reserve, in ATTENTION KV CELLS. A prompt is only admitted if the ring can
    // also give the slot room to actually generate; otherwise we would admit a request that stalls
    // on its first decoded token. Cheap relative to the ring, and bounded so tiny contexts still work.
    kv_unified_reserve = params_base.kv_unified
                       ? std::max(1, std::min(256, n_ctx / (16 * std::max(1, params_base.n_parallel))))
                       : 0;

    LOG_INFO("initializing slots", {
        {"n_slots",    params_base.n_parallel},
        {"kv_unified", params_base.kv_unified},
        {"n_ctx",      n_ctx},
        {"n_ctx_slot", n_ctx_slot},
        });

    LOG_INFO("prompt/decode interleave", {
        {"PXA_INTERLEAVE_DECODE_MS", pxa_interleave_decode_ms()},
        });

    if (params_base.kv_unified) {
        // Loud and unmistakable: this mode changes the capacity contract of every slot.
        LOG_WARNING("kv-unified ENABLED: slots share one attention-KV ring", {
            {"n_ctx",              n_ctx},
            {"n_ctx_slot",         n_ctx_slot},
            {"n_slots",            params_base.n_parallel},
            {"n_ctx_slot_if_split", n_ctx / params_base.n_parallel},
            {"kv_unified_reserve", kv_unified_reserve},
            });
        LOG_WARNING("kv-unified: per-slot recurrent (GDN) state is NOT unified and still scales with -np", {});
    }

    // PXA_SHARED_MTP_v1: the single shared MTP companion spec, created on the first MTP slot and
    // aliased (non-owning) into every other slot. nullptr until the first MTP slot builds it.
    common_speculative * shared_mtp_spec = nullptr;

    for (int i = 0; i < params_base.n_parallel; i++) {
        server_slot slot;

        slot.id = i;
        slot.ctx = ctx;
        slot.n_ctx = n_ctx_slot;
        slot.n_predict = params_base.n_predict;
        slot.mctx = mctx;
        slot.cache_tokens.has_mtmd = mctx != nullptr;
        slot.params.think_tokens = params_base.think_tokens;
        if (params_base.think_tokens.exclude) {
            SRV_WRN("Exclude reasoning tokens when selecting slot based on similarity: start: %s, end: %s\nuse `--reasoning-tokens none` to disable.\n", params_base.think_tokens.begin.c_str(), params_base.think_tokens.end.c_str() );
        }
        else {
            SRV_WRN("%s", "Include reasoning tokens when selecting slot based on similarity\nuse `--reasoning-tokens auto` to exclude reasoning tokens.\n");
        }
        LOG_INFO("new slot", {
            {"id_slot",    slot.id},
            {"n_ctx_slot", slot.n_ctx}
            });

        const int ga_n = params_base.grp_attn_n;
        const int ga_w = params_base.grp_attn_w;

        if (ga_n != 1) {
            GGML_ASSERT(ga_n > 0 && "ga_n must be positive");                       // NOLINT
            GGML_ASSERT(ga_w % ga_n == 0 && "ga_w must be a multiple of ga_n");             // NOLINT
            //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of ga_w");    // NOLINT
            //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * ga_n"); // NOLINT

            LOG_INFO("slot self-extend", {
                {"id_slot", slot.id},
                {"ga_n",    ga_n},
                {"ga_w",    ga_w}
                });
        }

        slot.ga_i = 0;
        slot.ga_n = ga_n;
        slot.ga_w = ga_w;

        slot.params.speculative = params_base.speculative;
        slot.sparams = params_base.sparams;

        const bool wants_mtp_stage = server_speculative_has_mtp(params_base.speculative);
        if (wants_mtp_stage) {
            const bool has_external_mtp = params_use_gemma4_external_mtp(params_base);

            if (llama_model_n_nextn_layer(model) > 0 || has_external_mtp) {
                params_base.pooling_type = LLAMA_POOLING_TYPE_NONE;

                if (!has_external_mtp) {
                    params_base.speculative.cparams_dft = common_context_params_to_llama(params_base);
                    // PXA_SHARED_MTP_M1a / PXA_SHARED_MTP_v1: the MTP companion is now SHARED across all
                    // slots, so it needs one seq row per slot -> n_seq_max = n_parallel. The old per-slot
                    // n_seq_max=1 design (PXA_LLAMA_MTP_NP_FIX) remapped every seq_id to local row 0; that
                    // crashed the old multi-row recurrent copy, but the companion allocates ZERO recurrent
                    // state in MTP_OP mode (delta-net layers skipped; the tail is plain attention) so its
                    // per-seq KV rows are independent ordinary attention rows -> a shared multi-seq companion
                    // is safe (M1a test: np4 loads + runs, 0 crashes / 0 wedges). The kv_seq = (n_seq_max<=1
                    // ? 0 : seq_id) guards in speculative.cpp now fall through to seq_id, giving correct
                    // per-slot routing on the shared context with no further per-call changes.
                    params_base.speculative.cparams_dft.n_seq_max = params_base.n_parallel; /* PXA_SHARED_MTP_M1a */
                    // PXA_MTP_VRAM_FIX: MTP tail processes at most n_max+1 tokens per batch.
                    // Inheriting the main n_ubatch (=n_ctx) makes the worst-case compute buffer
                    // scale with context size (~541 MiB at n_ctx=512 for 248K vocab output projection),
                    // OOMing multi-GPU layer-split configs. Cap n_ubatch to the actual MTP batch
                    // ceiling: saves ~500+ MiB VRAM on the tail-layer GPU.
                    {
                        const int mtp_batch_ceil = std::max(params_base.speculative.get_max_stage_n_max() + 1, 4);
                        params_base.speculative.cparams_dft.n_ubatch = mtp_batch_ceil;
                        // PXA_MTP_NBATCH_FIX: do NOT cap n_batch (only n_ubatch drives the compute buffer). Capping it to 2 asserted (n_tokens_all<=n_batch) on any prompt>2 tokens, so the spec ctx died on first request. n_batch stays at the inherited main value so prompts process; n_ubatch cap above keeps VRAM small.
                    }
                }

                params_base.speculative.cparams_dft.mtp          = true;
                params_base.speculative.cparams_dft.mtp_op_type  = MTP_OP_WARMUP;
                params_base.speculative.cparams_dft.embeddings   = true;

                slot.has_mtp = true;
                slot.params.speculative.cparams_dft = params_base.speculative.cparams_dft;

                slot.batch_spec = llama_batch_init(slot.params.speculative.get_max_stage_n_max() + 1, 0, 1);
                SLT_DBG(slot, "batch_spec contains %d tokens\n", slot.batch_spec.n_tokens);

                SRV_INF("%s\n", "MTP needs embeddings on decode, enabling");
                llama_set_embeddings(ctx, true);
            }
            else {
                SRV_WRN("%s\n", "MTP speculative stage requested, but model has 0 NextN layers. Removing MTP from the configured stage chain.");
                params_base.has_mtp = false;
                server_remove_speculative_stage(params_base.speculative, COMMON_SPECULATIVE_TYPE_MTP);
                slot.params.speculative = params_base.speculative;
                slot.has_mtp = false;
            }
        }

        const bool requested_spec = !params_base.speculative.get_resolved_stages().empty();

        bool can_spec = true;
        if (!params_base.dry_run) {
            can_spec = common_speculative_is_compat(ctx);
        }
        if (!can_spec) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }
        // try speculative decoding
        if (can_spec && requested_spec) {
            // PXA_SHARED_MTP_v1: share ONE ctx_mtp/spec across all MTP slots. The companion was
            // built with n_seq_max=n_parallel (PXA_SHARED_MTP_M1a) and every call routes by slot.id
            // as seq_id, so a single shared spec serves all slots; only the owning slot frees it.
            // Draft-model / ngram specs are NOT seq-keyed -> keep those per-slot (share only has_mtp).
            if (slot.has_mtp && shared_mtp_spec != nullptr) {
                slot.spec = shared_mtp_spec;   // non-owning alias of the shared MTP companion
                slot.owns_spec = false;
            } else {
            slot.spec = common_speculative_init(params_base.speculative, slot.ctx);
            if (slot.spec && slot.has_mtp) {
                shared_mtp_spec = slot.spec;   // first MTP slot owns the shared companion
                slot.owns_spec = true;
            } else if (slot.spec) {
                slot.owns_spec = true;         // non-MTP spec: per-slot, owns its own context
            }
            } // PXA_SHARED_MTP_v1 end share-branch
            if (slot.spec) {
                if (mctx && !slot.has_mtp) {
                    SRV_ERR("%s\n", "speculative decoding is not supported with multimodal");
                    return;
                }
                SLT_INF(slot, "%s", "speculative decoding context initialized\n");
            } else {
                if (llama_model_has_recurrent(model)) {
                    SRV_ERR("%s", "failed to initialize recurrent speculative context\n");
                    throw std::runtime_error("recurrent speculative context initialization failed");
                } else if (slot.has_mtp) {
                    SRV_ERR("%s", "failed to initialize MTP speculative context\n");
                    throw std::runtime_error("MTP speculative context initialization failed");
                } else {
                    SRV_ERR("%s", "failed to initialize speculative decoding context\n");
                    throw std::runtime_error("speculative decoding context initialization failed");
                }
            }
        }

        slot.reset();

        slots.push_back(std::move(slot));
    }

    default_generation_settings_for_props = get_formated_generation(slots.front());
    default_generation_settings_for_props["seed"] = -1;

    // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
    // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
    {
        const int32_t n_batch = llama_n_batch(ctx);

        // only a single seq_id per token is needed
        batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
    }

    metrics.init();

    if (params_base.cache_ram_mib != 0) {
        if (params_base.cache_ram_mib < 0) {
            LLAMA_LOG_INFO("prompt cache is enabled, size limit: %s\n", "no limit");
        }
        else {
            LLAMA_LOG_INFO("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
        }
        LLAMA_LOG_INFO("%s", "use `--cache-ram 0` to disable the prompt cache\n");
        // only apply ram size limit. No token limit for now.
        prompt_cache = std::make_unique<server_prompt_cache>(ctx, params_base.cache_ram_mib, 0);
    }
    else {
        LLAMA_LOG_INFO("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
    }

    // populate chat template params
    {
        common_chat_templates_ptr chat_templates;

        try {
            chat_templates = common_chat_templates_init(model, params_base.chat_template);

            LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

        }
        catch (const std::exception & e) {
            SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
            SRV_ERR("%s: please consider enabling jinja via --jinja, or use a custom chat template via --chat-template\n", __func__);
            SRV_ERR("%s: for example:  --chat-template chatml\n", __func__);
        }

        // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
        const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
        const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;

        chat_params = {
            /* use_jinja             */ params_base.use_jinja,
            /* prefill_assistant     */ params_base.prefill_assistant,
            /* reasoning_format      */ params_base.reasoning_format,
            /* chat_template_kwargs  */ params_base.default_template_kwargs,
            /* tmpls                 */ std::move(chat_templates),
            /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
            /* allow_audio           */ mctx ? mtmd_support_audio(mctx) : false,
            /* enable_thinking       */ enable_thinking,
            /* parallel_tool_calls  */ params_base.parallel_tool_calls,
            /* reasoning_budget      */ params_base.reasoning_budget,
            /* reasoning_budget_msg  */ params_base.reasoning_budget_message,
            /* force_pure_content    */ params_base.force_pure_content_parser
           // /* media_path            */ params_base.media_path,
        };
    }

}


void server_slot::prompt_save(server_prompt_cache& prompt_cache) const {
    assert(server_cached_prompt.data.size() == 0);

    const size_t cur_size = llama_state_seq_get_size(ctx, id, 0);

    LLAMA_LOG_INFO(" - saving prompt with length %d, total state size = %.3f MiB\n",
        (int)server_cached_prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

    auto* cur = prompt_cache.alloc(server_cached_prompt, cur_size);
    if (cur == nullptr) {
        return;
    }

    llama_synchronize(ctx); // PXA_CKPT_APPLY_SYNC: no bulk state read with kernels in flight
    llama_state_seq_get_data(ctx, cur->data.data(), cur_size, id, 0);
}

void server_slot::prompt_load(server_prompt_cache& prompt_cache, const server_tokens& tokens, float min_reusable_fraction) {
    bool res = prompt_cache.load(server_cached_prompt, tokens, ctx, id, min_reusable_fraction);
    if (!res) {
        LLAMA_LOG_INFO("failed to load prompt from cache\n");
    }
}

void server_slot::reset() {
    n_prompt_tokens = 0;
    last_gentxt_size = 0;
    generated_text = "";
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    n_softfail_consecutive = 0; // PXA_SOFTFAIL_BREAKER_v1
    n_past = 0;
    n_past_prompt = 0;
    n_discarded_prompt = 0;
    n_kept_prompt = 0;
    n_sent_text = 0;
    drafted.clear();
    drafted_spec_type = COMMON_SPECULATIVE_TYPE_NONE;
    i_batch_dft.clear();
    spec_ckpt.clear();
    n_sent_token_probs = 0;
    infill = false;
    ga_i = 0;
    n_past_se = 0;
    chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    logit_bias.clear();
    token_buffer.clear();
    rewind_count = 0;
    n_buffer = 0;
    rewind_status = false;

    generated_token_probs.clear();
    checkpoint_pos = 0;
    image_just_processed = false;
    do_checkpoint = false;
    if (spec != nullptr) {
        common_speculative_clear_sequence_hidden(spec, id);
    }

    positional_bans.clear();
    ban_phrases.clear();
    ban_regex.clear();
    ban_regex_ci.clear();

    allow_ruless.clear();
    allow_pieces.clear();
    allow_kws.clear();
    allow_kw_delay = 0;
    allow_idx = 0;

    // Reset speculative decoding stats
    n_draft_total = 0;
    n_draft_accepted = 0;

    // PXA_MTP_ADAPTIVE: reset the adaptive draft-length controller (optimistic so cold-start drafts full)
    spec_accept_ema      = 0.70f;
    spec_probe_countdown = 0;
    spec_adaptive_tier   = 2;
    chat_msg = {};
    json_schema = json();
    generated_tool_call_ids.clear();

    anthropic_thinking_block_started = false;
    anthropic_text_block_started = false;

    oai_resp_thinking_block_started = false;
    oai_resp_text_block_started = false;
    oai_resp_id.clear();
    oai_resp_reasoning_id.clear();
    oai_resp_message_id.clear();
    oai_resp_fc_id.clear();

    task.reset();
}

// PXA_MTP_LAZY_PROMPT_FLAGS (2026-07-15): under lazy MTP warmup (PXA_MTP_LAZY_WARMUP=1) nothing
// consumes per-token outputs on prompt batches (the companion warmup that needed the per-token
// feature rows is skipped). Flagging every prompt token as an output forced n_outputs=n_tokens in
// llama_decode, which (a) computed the lm_head over ALL prompt rows and (b) async-D2H copied the
// full-vocab logits for the whole chunk (2048 x 248320 x 4B = 2.03 GB PER CHUNK on the 35B) -- the
// entire remaining "structural" MTP prefill tax (~620 ms/chunk copy + ~77 ms/chunk lm_head).
// Under lazy, prompt tokens are flagged exactly like a non-MTP slot; the prompt-done path still
// flags the final token (sampling + the lazy self-seed hidden row). Eager mode is byte-unchanged.
static bool pxa_mtp_lazy_prompt_flags() {
    // In lockstep with pxa_mtp_lazy_warmup_enabled() (src/llama.cpp) — keep the three in lockstep:
    // explicit env wins; else ENHANCE -> on (bit-identical, measured), REFERENCE/DEFAULT -> off.
    static const bool on = [](){
        const char * e = getenv("PXA_MTP_LAZY_WARMUP");
        if (e) return atoi(e) == 1;
        return ggml_pxa_config_level() >= 2;
    }();
    return on;
}

bool server_slot::need_embd() const {
    return embedding || has_mtp;
}

bool server_slot::has_budget(gpt_params& global_params) {
    if (params.n_predict == -1 && global_params.n_predict == -1) {
        return true; // limitless
    }

    n_remaining = -1;

    if (params.n_predict != -1) {
        n_remaining = params.n_predict - n_decoded;
    }
    else if (global_params.n_predict != -1) {
        n_remaining = global_params.n_predict - n_decoded;
    }

    return n_remaining > 0; // no budget
}

bool server_slot::available() const {
    return state == SLOT_STATE_IDLE && command == SLOT_COMMAND_NONE;
}

bool server_slot::is_processing() const {
    return (state == SLOT_STATE_IDLE && command == SLOT_COMMAND_LOAD_PROMPT) || state == SLOT_STATE_PROCESSING;
}

void server_slot::add_token_string(const completion_token_output& token) {
    if (command == SLOT_COMMAND_RELEASE) {
        return;
    }
    generated_token_probs.push_back(token);
}

bool server_slot::can_speculate() const {
    return (!!spec || has_mtp);
}

int server_slot::get_n_draft_max() const {
    if (!can_speculate()) {
        return 0;
    }

    // determine the max draft that fits the current slot state
    int n_draft_max = params.speculative.get_max_stage_n_max();

    // note: slot.prompt is not yet expanded with the `id` token sampled above
    //       also, need to leave space for 1 extra token to allow context shifts
    n_draft_max = std::min(n_draft_max, n_ctx - n_past - 2);

    if (n_remaining > 0) {
        n_draft_max = std::min(n_draft_max, n_remaining - 1);
    }

    SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

    const int min_usable_draft = params.speculative.get_min_usable_stage_n_min();
    if (n_draft_max < min_usable_draft) {
        SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, min_usable_draft);
        n_draft_max = 0;
    }
    return n_draft_max;
}

void server_slot::release() {
    // PXA_SLOT_RELEASE_UAF_FIX_v1: idempotent release. Guard on is_processing() so this covers BOTH
    // an actively-decoding slot (PROCESSING) and a just-launched slot whose prompt has not decoded yet
    // (IDLE + command==LOAD_PROMPT) -- the latter must also be cancellable cleanly. A second release()
    // on an already-released slot is a no-op: is_processing() is false (state==IDLE, command==NONE) and
    // task is already null, so nothing re-runs. This is the structural idempotency that prevents the
    // double-release -> re-teardown -> use-after-free (CUDA illegal memory access) under cancel churn.
    if (is_processing()) {
        t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
        command = SLOT_COMMAND_RELEASE;
        state = SLOT_STATE_IDLE;
        task.reset();
    }
    llama_decode_reset();
}


json server_slot::get_formated_timings() const {
    return json{
        {"prompt_n",               n_prompt_tokens_processed},
        {"prompt_ms",              t_prompt_processing},
        {"prompt_per_token_ms",    t_prompt_processing / n_prompt_tokens_processed},
        {"prompt_per_second",      1e3 / t_prompt_processing * n_prompt_tokens_processed},

        {"predicted_n",            n_decoded},
        {"predicted_ms",           t_token_generation},
        {"predicted_per_token_ms", t_token_generation / n_decoded},
        {"predicted_per_second",   1e3 / t_token_generation * n_decoded},

        {"n_ctx",           n_ctx},
        {"n_past",           n_past},
    };
}

result_timings server_slot::get_timings() const {
    result_timings timings;
    timings.prompt_n = n_prompt_tokens_processed;
    timings.prompt_ms = t_prompt_processing;
    timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
    timings.prompt_per_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    timings.predicted_n = n_decoded;
    timings.predicted_ms = t_token_generation;
    timings.predicted_per_token_ms = t_token_generation / n_decoded;
    timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

    timings.n_ctx = n_ctx;
    timings.n_past = n_past;


    // Add speculative metrics
    if (n_draft_total > 0) {
        timings.draft_n = n_draft_total;
        timings.draft_n_accepted = n_draft_accepted;
    }

    return timings;
}

const common_chat_msg& server_slot::update_chat_msg(bool is_partial, std::vector<common_chat_msg_diff>& diffs,
    bool filter_tool_calls) {
    // PXA_CHATPARSE_THROTTLE: common_chat_parse re-parses ALL of generated_text from byte 0 on every
    // call -> O(N^2) over a response (several ms/token deep in long agentic/code outputs, larger than
    // the whole sampler). Raw content still streams every token via text_to_send; the structured diffs
    // only need to be current at flush boundaries. Throttle the full parse to every Kth PARTIAL token,
    // and ALWAYS parse the final token (stop==EOS) so the authoritative message is exact. Skipped tokens
    // emit no diff; the next parse's compute_diffs covers the gap (chat_msg/sent_tool_call_names persist),
    // so nothing is lost -- structured deltas just arrive in K-token chunks (reassembled by clients).
    // Default OFF (K=1): measured +0.5% at 1200 tokens (the PEG parse is cheaper than estimated, and
    // decode here is memory-bound not host-bound). Verified tool-call-safe. Opt in for very long
    // (>4K-token) outputs where the O(N^2) eventually matters: set PXA_CHATPARSE_EVERY=4 (or higher).
    if (is_partial && stop != STOP_TYPE_EOS) {
        static const int pxa_chatparse_k = getenv("PXA_CHATPARSE_EVERY") ? atoi(getenv("PXA_CHATPARSE_EVERY")) : 1;
        if (pxa_chatparse_k > 1 && (++chat_parse_throttle % pxa_chatparse_k) != 0) {
            diffs.clear();
            return chat_msg;
        }
    }
    auto msg_prv_copy = chat_msg;
    // PXA 2026-07-11: a length-truncated / forced-close answer can be unparseable by the PEG chat
    // parser -> NEVER let that become a 500. Fall back to delivering the raw generated text as content so
    // the client still gets the answer (finish_reason is set upstream from the stop type).
    common_chat_msg new_msg;
    try {
        new_msg = common_chat_parse(
            generated_text,
            /* is_partial= */ stop != STOP_TYPE_EOS,
            params.chat_parser_params);
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: chat parse failed (%s); returning raw content instead of erroring\n", __func__, e.what());
        new_msg = msg_prv_copy;
        new_msg.role = "assistant";
        new_msg.content = generated_text;
    }
    if (!new_msg.empty()) {
        //new_msg.ensure_tool_call_ids_set(generated_tool_call_ids, gen_tool_call_id);
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index = i;
                        header.tool_call_delta.id = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index = i;
                        header.tool_call_delta.id = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}


size_t server_slot::find_stopping_strings(const std::string& text, const size_t last_token_size, bool is_full_stop) {
    size_t stop_pos = std::string::npos;

    for (const std::string& word : params.antiprompt) {
        size_t pos;

        if (is_full_stop) {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

            pos = text.find(word, from_pos);
        }
        else {
            pos = string_find_partial_stop(text, word);
        }

        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            if (is_full_stop) {
                stopped_word = true;
                stopping_word = word;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }

    return stop_pos;
}

void server_slot::print_timings() const {
    char buffer[512];
    double t_prompt = t_prompt_processing / n_prompt_tokens_processed;
    double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    double t_gen = t_token_generation / n_decoded;
    double n_gen_second = 1e3 / t_token_generation * n_decoded;

    SLT_INF(*this,
        "\n"
        "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
        "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
        "      total time = %10.2f ms / %5d tokens\n",
        t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
        t_token_generation, n_decoded, t_gen, n_gen_second,
        t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

    if (n_draft_total > 0) {
        const float draft_ratio = (float)n_draft_accepted / n_draft_total;
        SLT_CNT(*this,
            "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
            draft_ratio, n_draft_accepted, n_draft_total
        );
    }
    common_speculative_print_stats(spec, n_gen_second, n_decoded, n_past,
        const_cast<common_params_speculative *>(&params.speculative));
}

void server_metrics::init() {
    t_start = ggml_time_us();
}

void server_metrics::on_prompt_eval(const server_slot& slot) {
    n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
    n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
    t_prompt_processing += slot.t_prompt_processing;
    t_prompt_processing_total += slot.t_prompt_processing;
}

void server_metrics::on_prediction(const server_slot& slot) {
    n_tokens_predicted_total += slot.n_decoded;
    n_tokens_predicted += slot.n_decoded;
    t_tokens_generation += slot.t_token_generation;
    t_tokens_generation_total += slot.t_token_generation;
}

void server_metrics::reset_bucket() {
    n_prompt_tokens_processed = 0;
    t_prompt_processing = 0;
    n_tokens_predicted = 0;
    t_tokens_generation = 0;
}

std::vector<llama_token> server_context::tokenize(const json& json_prompt, bool add_special) const {
    // TODO: currently, we tokenize using special tokens by default
    //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
    //       but it's better compared to completely ignoring ChatML and other chat templates
    const bool TMP_FORCE_SPECIAL = true;

    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    std::vector<llama_token> prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto& p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                std::vector<llama_token> p;
                if (first) {
                    p = ::common_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
                    first = false;
                }
                else {
                    p = ::common_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            }
            else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    }
    else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = ::common_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
    }

    return prompt_tokens;
}

server_slot* server_context::get_slot_by_id(int id) {
    for (server_slot& slot : slots) {
        if (slot.id == id) {
            return &slot;
        }
    }

    return nullptr;
}

float server_context::calculate_slot_f_keep(const server_slot & slot, llama_context * ctx,const server_tokens & a, const server_tokens & b) {
    float f_keep = 0.0f;
    if (!a.empty()) {
        if (slot.ga_n == 1 && slot.n_discarded_prompt > 0 && b.size() >= slot.n_ctx) {
            f_keep = a.get_cached_tokens_similarity(slot.ctx, b, slot.params.n_keep + add_bos_token, slot.n_discarded_prompt);
        }
        else {
            f_keep = a.get_cached_tokens_similarity(slot.ctx, b, 0, 0);
        }
    }
    return f_keep;
}

std::pair<common_prefix, float> server_context::calculate_slot_similarity(const server_slot& slot, llama_context* ctx, const server_tokens& a, const server_tokens& b) {
    std::pair<common_prefix, float> sim;
    // length of the Longest Common Prefix between the current slot's prompt and the input prompt
    common_prefix lcp_len = a.get_common_prefix(slot.ctx, b);
    // fraction of the Longest Common Prefix length with respect to the input prompt and cached prompt length
    float sim_cur = a.get_tokens_similarity(slot.ctx, b, 0, 0);
    // handle context shift
    if (slot.ga_n == 1 && slot.n_discarded_prompt > 0 && b.size() >= slot.n_ctx) {
        float sim_cur_ctx_shift = a.get_tokens_similarity(slot.ctx, b, slot.n_kept_prompt, slot.n_discarded_prompt);
        if (sim_cur_ctx_shift > sim_cur) {
            sim_cur = sim_cur_ctx_shift;
        }
    }
    sim.first = lcp_len;
    sim.second = sim_cur;
    return sim;
}

void server_context::copy_data_to_cached_prompt(const server_tokens & tokens, server_slot & slot) {
    slot.server_cached_prompt.tokens = tokens.clone(); // copy cache tokens
    slot.server_cached_prompt.n_discarded_prompt = slot.n_discarded_prompt;
    slot.server_cached_prompt.n_kept_prompt = slot.n_kept_prompt;
    slot.server_cached_prompt.think_tokens = slot.params.think_tokens;
}

server_slot* server_context::get_available_slot(const server_task& task) {
    server_slot* ret = nullptr;
    bool update_cache = false;

    // find the slot that has at least n% prompt similarity
    if (ret == nullptr && slot_prompt_similarity != 0.0f) {
        int max_lcp_len = 0;
        float sim_best = 0;

        for (server_slot& slot : slots) {
            // skip the slot if it is not available
            if (!slot.available()) {
                continue;
            }
            auto& cache_tokens = slot.cache_tokens;
            // skip the slot if it does not contains prompt
            if (cache_tokens.empty()) {
                continue;
            }
            std::pair<common_prefix, float> sim;
            if (slot.params.think_tokens.exclude) {
                server_tokens cache_tokens_exclude_think = slot.cache_tokens.get_tokens_exclude_think(slot.ctx, slot.params.think_tokens);
                server_tokens prompt_tokens_exclude_think = task.tokens.get_tokens_exclude_think(slot.ctx, slot.params.think_tokens);
                sim = calculate_slot_similarity(slot, ctx, cache_tokens_exclude_think, prompt_tokens_exclude_think);
            }
            else {
                sim = calculate_slot_similarity(slot, ctx, cache_tokens, task.tokens);
            }
            common_prefix lcp_len = sim.first;
            float sim_cur = sim.second;

            // select the current slot if the criteria match
            if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                sim_best = sim_cur;
                max_lcp_len = lcp_len.first;
                ret = &slot;
            }
        }
        if (ret != nullptr) {
            LOG_VERBOSE("selected slot by lcp similarity", {
                {"id_slot", ret->id},
                {"max_lcp_len", max_lcp_len},
                {"similarity", sim_best},
                });
        }
    }

    // find the slot that has been least recently used
    if (ret == nullptr) {
        int64_t t_last = ggml_time_us();
        for (server_slot& slot : slots) {
            // skip the slot if it is not available
            if (!slot.available()) {
                continue;
            }
            // select the current slot if the criteria match
            if (slot.t_last_used < t_last) {
                t_last = slot.t_last_used;
                ret = &slot;
            }
        }

        if (ret != nullptr) {
            LOG_VERBOSE("selected slot by lru", {
                {"id_slot", ret->id},
                {"t_last", t_last},
                });
        }
    }
    if (ret) {
        auto& tokens = ret->cache_tokens;
        float f_keep = 0;
        size_t cache_token_size = tokens.size();
        if (!tokens.empty()) {
            if (ret->params.think_tokens.exclude) {
                server_tokens cache_exclude_think = tokens.get_tokens_exclude_think(ret->ctx, ret->params.think_tokens);
                server_tokens prompt_exclude_think = task.tokens.get_tokens_exclude_think(ret->ctx, ret->params.think_tokens);

                cache_token_size = cache_exclude_think.size();
                f_keep = calculate_slot_f_keep(*ret, ret->ctx, cache_exclude_think, prompt_exclude_think);
            }
            else {
                f_keep = calculate_slot_f_keep(*ret, ret->ctx, tokens, task.tokens);
            }
            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < cache_ram_similarity) {
                update_cache = true;
            }
        }

        update_cache = update_cache && prompt_cache;
        // cache prompts only for completion tasks
        update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

        // don't update the cache if the slot's context is above cache_ram_n_min
        update_cache = update_cache && cache_token_size >= cache_ram_n_min;

        LLAMA_LOG_INFO("======== Prompt cache: cache size: %d, n_keep: %d, n_discarded_prompt: %d, cache_ram_n_min: %d, f_keep: %.2f, cache_ram_similarity: %.2f\n",
            (int)tokens.size(), ret->n_kept_prompt, ret->n_discarded_prompt, cache_ram_n_min, f_keep, cache_ram_similarity);
        if (update_cache) {
            const int64_t t_start = ggml_time_us();
            LLAMA_LOG_INFO("updating prompt cache\n");
            // copy cache tokens
            copy_data_to_cached_prompt(tokens, *ret);

            ret->prompt_save(*prompt_cache);
            LLAMA_LOG_INFO("prompt cache save took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
        }
        // has prompts saved earlier to load
        if (prompt_cache && !prompt_cache->states.empty()) {
            const int64_t t_start = ggml_time_us();
            copy_data_to_cached_prompt(tokens, *ret);

            ret->prompt_load(*prompt_cache, task.tokens, cache_ram_similarity);
            prompt_cache->update();

            ret->cache_tokens = ret->server_cached_prompt.tokens.clone(); // recover cache tokens
            ret->n_discarded_prompt = ret->server_cached_prompt.n_discarded_prompt;
            ret->n_kept_prompt = ret->server_cached_prompt.n_kept_prompt;

            LLAMA_LOG_INFO("prompt cache load took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
        }
    }
    return ret;
}

int32_t server_context::populate_vocab_pieces() {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocab_pieces.size() == n_vocab) {
        return n_vocab;
    }
    vocab_pieces.clear();
    vocab_pieces.reserve(n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        auto piece = common_token_to_piece(ctx, id, true);
        max_piece_len = std::max(piece.length(), max_piece_len);
        vocab_pieces.push_back(std::move(piece));
    }
    return n_vocab;
}

bool server_context::launch_slot_with_task(server_slot& slot, server_task& task) {
    slot_params defaults;
    defaults.speculative = params_base.speculative;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    common_params_sampling default_sparams = params_base.sparams;
    auto& data = task.data;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (data.count("__oaicompat") != 0) {
        slot.oaicompat = true;
        slot.oaicompat_model = task.params.oaicompat_model;
    }
    else {
        slot.oaicompat = false;
        slot.oaicompat_model = "";
    }
    slot.params.oaicompat = task.params.oaicompat;
    slot.params.oaicompat_cmpl_id =task.params.oaicompat_cmpl_id;
    pxa_expert_log_begin(task.id, slot.params.oaicompat_cmpl_id);

    slot.oai_resp_thinking_block_started = false;
    slot.oai_resp_text_block_started = false;
    slot.oai_resp_id = "resp_" + random_string();
    slot.oai_resp_reasoning_id = "rs_" + random_string();
    slot.oai_resp_message_id = "msg_" + random_string();
    slot.oai_resp_fc_id.clear();
    slot.params.timings_per_token = json_value(data, "timings_per_token", false);
    slot.params.stream = json_value(data, "stream", false);
    auto stream_opt = json_value(data, "stream_options", json::object());
    slot.params.include_usage = json_value(stream_opt, "include_usage", false);
    slot.params.cache_prompt = json_value(data, "cache_prompt", true);
    slot.params.n_predict = json_value(data, "n_predict", json_value(data, "max_tokens", defaults.n_predict));
    slot.saturate_predict = json_value(data, "saturate_predict", false);
    slot.sparams.top_k = json_value(data, "top_k", default_sparams.top_k);
    slot.sparams.top_p = json_value(data, "top_p", default_sparams.top_p);
    slot.sparams.min_p = json_value(data, "min_p", default_sparams.min_p);
    slot.sparams.tfs_z = json_value(data, "tfs_z", default_sparams.tfs_z);
    slot.sparams.typical_p = json_value(data, "typical_p", default_sparams.typical_p);
    slot.sparams.temp = json_value(data, "temperature", default_sparams.temp);
    slot.sparams.dynatemp_range = json_value(data, "dynatemp_range", default_sparams.dynatemp_range);
    slot.sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
    slot.sparams.xtc_probability = json_value(data, "xtc_probability", default_sparams.xtc_probability);
    slot.sparams.xtc_threshold = json_value(data, "xtc_threshold", default_sparams.xtc_threshold);
    slot.sparams.top_n_sigma = json_value(data, "top_n_sigma", default_sparams.top_n_sigma);
    slot.sparams.penalty_last_n = json_value(data, "repeat_last_n", default_sparams.penalty_last_n);
    slot.sparams.penalty_repeat = json_value(data, "repeat_penalty", default_sparams.penalty_repeat);
    slot.sparams.penalty_freq = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
    slot.sparams.penalty_present = json_value(data, "presence_penalty", default_sparams.penalty_present);
    slot.sparams.dry_multiplier = json_value(data, "dry_multiplier", default_sparams.dry_multiplier);
    slot.sparams.dry_base = json_value(data, "dry_base", default_sparams.dry_base);
    slot.sparams.dry_allowed_length = json_value(data, "dry_allowed_length", default_sparams.dry_allowed_length);
    slot.sparams.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", default_sparams.dry_penalty_last_n);
    slot.sparams.mirostat = json_value(data, "mirostat", default_sparams.mirostat);
    slot.sparams.mirostat_tau = json_value(data, "mirostat_tau", default_sparams.mirostat_tau);
    slot.sparams.mirostat_eta = json_value(data, "mirostat_eta", default_sparams.mirostat_eta);
    slot.sparams.adaptive_target = json_value(data, "adaptive_target", default_sparams.adaptive_target);
    slot.sparams.adaptive_decay = json_value(data, "adaptive_decay", default_sparams.adaptive_decay);
    slot.sparams.adaptive_updt_w_cur = json_value(data, "adaptive_updt_w_cur", default_sparams.adaptive_updt_w_cur);
    slot.sparams.penalize_nl = json_value(data, "penalize_nl", default_sparams.penalize_nl);
    slot.params.n_keep = json_value(data, "n_keep", slot.params.n_keep);
    slot.params.n_discard = json_value(data, "n_discard", defaults.n_discard);
    slot.sparams.seed = json_value(data, "seed", default_sparams.seed);
    slot.sparams.n_probs = json_value(data, "n_probs", default_sparams.n_probs);
    slot.sparams.min_keep = json_value(data, "min_keep", default_sparams.min_keep);

    slot.params.post_sampling_probs = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    // speculative decoding parameters
    try {
        slot.params.speculative = defaults.speculative;
        const bool has_flat_n_max = json_value_ptr(data, "speculative.n_max") != nullptr;
        const bool has_flat_n_min = json_value_ptr(data, "speculative.n_min") != nullptr;
        const bool has_flat_p_min = json_value_ptr(data, "speculative.p_min") != nullptr;

        slot.params.speculative.n_max = json_value(data, "speculative.n_max", params_base.speculative.n_max);
        slot.params.speculative.n_min = json_value(data, "speculative.n_min", params_base.speculative.n_min);
        slot.params.speculative.p_min = json_value(data, "speculative.p_min", params_base.speculative.p_min);

        server_reject_dead_speculative_request_overrides(data);

        const json stages = json_value(data, "speculative.stages", json());
        if (stages.is_null() && !slot.params.speculative.stages.empty()) {
            for (auto & stage : slot.params.speculative.stages) {
                if (has_flat_n_max) {
                    stage.n_max = -1;
                }
                if (has_flat_n_min) {
                    stage.n_min = -1;
                }
                if (has_flat_p_min) {
                    stage.p_min = -1.0f;
                }
            }
        }

        if (!stages.is_null()) {
            if (!stages.is_array()) {
                throw std::runtime_error("Error: speculative.stages must be an array");
            }

            const auto default_stages = defaults.speculative.get_resolved_stages();
            if (stages.size() != default_stages.size()) {
                throw std::runtime_error("Error: speculative.stages must provide the same number of stages configured at server startup");
            }

            slot.params.speculative.stages = default_stages;
            for (size_t i = 0; i < stages.size(); ++i) {
                const auto stage_override = server_parse_speculative_stage_json(stages[i]);
                if (stage_override.type != default_stages[i].type) {
                    throw std::runtime_error("Error: speculative.stages must preserve the stage types configured at server startup");
                }

                if (stage_override.has_n_max_override()) {
                    slot.params.speculative.stages[i].n_max = stage_override.n_max;
                }
                if (stage_override.has_n_min_override()) {
                    slot.params.speculative.stages[i].n_min = stage_override.n_min;
                }
                if (stage_override.has_p_min_override()) {
                    slot.params.speculative.stages[i].p_min = stage_override.p_min;
                }
            }

            const auto resolved = slot.params.speculative.get_resolved_stages();
            slot.params.speculative.type = resolved.empty() ? COMMON_SPECULATIVE_TYPE_NONE : resolved.front().type;
        }

        slot.params.speculative.n_min = std::min(slot.params.speculative.n_max, slot.params.speculative.n_min);
        slot.params.speculative.n_min = std::max(slot.params.speculative.n_min, 0);
        slot.params.speculative.n_max = std::max(slot.params.speculative.n_max, 0);

        if (slot.can_speculate() &&
            llama_model_has_recurrent(model) &&
            slot.params.speculative.get_max_stage_n_max() > params_base.speculative.get_max_stage_n_max()) {
            send_error(task,
                "Error: speculative n_max=" + std::to_string(slot.params.speculative.get_max_stage_n_max()) +
                " exceeds the recurrent speculative startup limit of " + std::to_string(params_base.speculative.get_max_stage_n_max()) +
                "; restart the server with a higher n_max inside the configured --spec-type stages to reserve checkpoint capacity",
                    ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        if (!server_speculative_same_stage_types(slot.params.speculative, defaults.speculative)) {
            throw std::runtime_error("Error: per-request speculative stages must match the server startup stage types; only stage parameter overrides are supported");
        }

        if (slot.params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) && !slot.has_mtp) {
            throw std::runtime_error("Error: MTP speculative stage requested, but the server was not started with MTP support");
        }

        if (slot.params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_DRAFT) && !params_base.speculative.has_dft()) {
            throw std::runtime_error("Error: draft speculative stage requested, but no draft model is loaded");
        }

        std::string spec_error;
        if (!common_speculative_validate_chain(slot.params.speculative, &spec_error)) {
            throw std::runtime_error("Error: invalid speculative request configuration: " + spec_error);
        }
    } catch (const std::exception & e) {
        send_error(task, e.what(), ERROR_TYPE_INVALID_REQUEST);
        return false;
    }


    if (slot.sparams.penalty_last_n < -1) {
        throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (slot.sparams.dry_penalty_last_n < -1) {
        throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (slot.sparams.penalty_last_n == -1) {
        // note: should be the slot's context and not the full context, but it's ok
        slot.sparams.penalty_last_n = llama_n_ctx(ctx);
    }

    if (slot.sparams.dry_penalty_last_n == -1) {
        slot.sparams.dry_penalty_last_n = llama_n_ctx(ctx);

    }
    if (slot.sparams.dry_base < 1.0f)
    {
        slot.sparams.dry_base = default_sparams.dry_base;
    }

    // sequence breakers for DRY
    {
        // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
        // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

        if (data.contains("dry_sequence_breakers")) {
            slot.sparams.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (slot.sparams.dry_sequence_breakers.empty()) {
                send_error(task, "Error: dry_sequence_breakers must be a non-empty array of strings", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            auto schema = json_value(data, "json_schema", json::object());
            LLAMA_LOG_DEBUG("JSON schema: %s\n", schema.dump(2).c_str());
            std::string grammar_str = json_schema_to_grammar(schema);
            SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
            slot.sparams.grammar = { COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str) };
        }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    }
    else {
        slot.sparams.grammar = default_sparams.grammar;
        std::string grammar_str = json_value(data, "grammar", std::string());
        if (!grammar_str.empty()) {
            // grammar_type key is set by the server when converting chat template grammars
            std::string grammar_type = json_value(data, "grammar_type", std::string());
            if (grammar_type == "tool_calls") {
                slot.sparams.grammar = { COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str) };
            } else {
                // explicit grammar from the user (API field "grammar")
                slot.sparams.grammar = { COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str) };
            }
            LLAMA_LOG_DEBUG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(slot.sparams.grammar).c_str());
        }
        slot.sparams.grammar_lazy = json_value(data, "grammar_lazy", default_sparams.grammar_lazy);
        LLAMA_LOG_DEBUG("Grammar lazy: %s\n", slot.sparams.grammar_lazy ? "true" : "false");
    }

    if (slot.params.cache_prompt && slot.ga_n != 1) {
        LOG_WARNING("cache_prompt is not supported with group-attention", {});
        slot.params.cache_prompt = false;
    }

    if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
        // Might be better to reject the request with a 400 ?
        LOG_WARNING("Max tokens to predict exceeds server configuration", {
            {"params.n_predict", slot.params.n_predict},
            {"slot.n_predict",   slot.n_predict},
            });
        slot.params.n_predict = slot.n_predict;
    }

    // infill
    slot.params.input_prefix = json_value(data, "input_prefix", defaults.input_prefix);
    slot.params.input_suffix = json_value(data, "input_suffix", defaults.input_suffix);

    // get prompt
    if (!task.infill) {
        slot.prompt_tokens = std::move(task.tokens);

        const auto & prompt = data.find("prompt");
        if (prompt != data.end()) {
            if (prompt->is_string() ||
                (prompt->is_array() && !prompt->empty() && prompt->at(0).is_number_integer())) {
                slot.prompt = *prompt;
            } else if (prompt->is_array() && prompt->size() == 1 && prompt->at(0).is_string()) {
                slot.prompt = *prompt;
            } else if (prompt->is_array() && prompt->size() == 1 && prompt->at(0).is_array()) {
                slot.prompt = prompt->at(0);
            }
        }
    }

    // penalize user-provided tokens
    {
        slot.sparams.penalty_prompt_tokens.clear();
        slot.sparams.use_penalty_prompt_tokens = false;

        const auto& penalty_prompt = data.find("penalty_prompt");

        if (penalty_prompt != data.end()) {
            if (penalty_prompt->is_string()) {
                const auto penalty_prompt_string = penalty_prompt->get<std::string>();
                slot.sparams.penalty_prompt_tokens = common_tokenize(model, penalty_prompt_string, false);

                if (slot.params.n_predict > 0) {
                    slot.sparams.penalty_prompt_tokens.reserve(slot.sparams.penalty_prompt_tokens.size() + slot.params.n_predict);
                }
                slot.sparams.use_penalty_prompt_tokens = true;

                LOG_VERBOSE("penalty_prompt_tokens", {
                    {"id_slot", slot.id},
                    {"tokens",  slot.sparams.penalty_prompt_tokens},
                    });
            }
            else if (penalty_prompt->is_array()) {
                const auto n_tokens = penalty_prompt->size();
                slot.sparams.penalty_prompt_tokens.clear();
                slot.sparams.penalty_prompt_tokens.reserve(n_tokens + std::max(0, slot.params.n_predict));

                const int n_vocab = llama_n_vocab(model);
                for (const auto& penalty_token : *penalty_prompt) {
                    if (penalty_token.is_number_integer()) {
                        const auto tok = penalty_token.get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            slot.sparams.penalty_prompt_tokens.push_back(tok);
                        }
                    }
                }
                slot.sparams.use_penalty_prompt_tokens = true;

                LOG_VERBOSE("penalty_prompt_tokens", {
                    {"id_slot", slot.id},
                    {"tokens",  slot.sparams.penalty_prompt_tokens},
                    });
            }
        }
    }
    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            slot.params.chat_parser_params.format = static_cast<common_chat_format>(it->get<int>());
            LLAMA_LOG_DEBUG("Chat format: %s\n", common_chat_format_name(slot.params.chat_parser_params.format));
        }
        else {
            slot.params.chat_parser_params.format = defaults.chat_parser_params.format;
        }
        common_reasoning_format reasoning_format = params_base.reasoning_format;
        if (data.contains("reasoning_format")) {
            reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
        }
        slot.params.chat_parser_params.reasoning_format = reasoning_format;
        slot.params.chat_parser_params.reasoning_in_content = slot.params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        slot.params.chat_parser_params.generation_prompt = json_value(data, "generation_prompt", std::string());
        slot.sparams.generation_prompt = slot.params.chat_parser_params.generation_prompt;
        LLAMA_LOG_DEBUG("Generation prompt: '%s'\n", slot.params.chat_parser_params.generation_prompt.c_str());
        slot.params.chat_parser_params.parse_tool_calls = json_value(data, "parse_tool_calls", false);
        if (data.contains("chat_parser")) {
            slot.params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        } else {
            // Reset to default; otherwise a reused slot would apply the previous chat's PEG parser to non-chat outputs (e.g., /v1/completions).
            slot.params.chat_parser_params.parser = defaults.chat_parser_params.parser;
        }
    }
    {

        const auto preserved_tokens = data.find("preserved_tokens");
        if (preserved_tokens != data.end()) {
            slot.sparams.preserved_tokens.clear();
            for (const auto& t : *preserved_tokens) {
                auto ids = common_tokenize(model, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    LOG("Preserved token: %d\n", ids[0]);
                    slot.sparams.preserved_tokens.insert(ids[0]);
                }
                else {
                    // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                    LOG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                }
            }
        }
        const auto grammar_triggers = data.find("grammar_triggers");
        if (grammar_triggers != data.end()) {
            slot.sparams.grammar_triggers.clear();
            for (const auto& t : *grammar_triggers) {
                server_grammar_trigger ct(t);
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto& word = ct.value.value;
                    auto ids = common_tokenize(model, word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(slot.sparams.preserved_tokens.begin(), slot.sparams.preserved_tokens.end(), (llama_token)token) == slot.sparams.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        LOG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                        common_grammar_trigger trigger;
                        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        slot.sparams.grammar_triggers.push_back(std::move(trigger));
                    }
                    else {
                        LOG("Grammar trigger word: `%s`\n", word.c_str());
                        slot.sparams.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word });
                    }
                }
                else {
                    //slot.sparams.grammar_triggers.push_back(ct);
                    if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                        LLAMA_LOG_DEBUG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                    }
                    else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                        LLAMA_LOG_DEBUG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                    }
                    else {
                        throw std::runtime_error("Unknown grammar trigger type");
                    }
                    slot.sparams.grammar_triggers.emplace_back(std::move(ct.value));
                }
            }
        }

        if (slot.sparams.grammar_lazy && slot.sparams.grammar_triggers.empty()) {
            throw std::runtime_error("Error: no triggers set for lazy grammar!");
        }
    }

    // Parse reasoning budget sampler parameters
    {
        const int32_t budget = json_value(data, "reasoning_budget_tokens", (int32_t)-1);
        const auto start_tag = json_value(data, "reasoning_budget_start_tag", std::string());
        const auto end_tag = json_value(data, "reasoning_budget_end_tag", std::string());
        const auto message = json_value(data, "reasoning_budget_message", std::string());
        slot.sparams.reasoning_budget_tokens = budget;

        if (!start_tag.empty()) {
            slot.sparams.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        }
        if (!end_tag.empty()) {
            slot.sparams.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
            slot.sparams.reasoning_budget_forced = common_tokenize(vocab, message + end_tag, false, true);

            SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu toks, forced=%zu toks\n",
                budget, slot.sparams.generation_prompt.c_str(),
                slot.sparams.reasoning_budget_start.size(),
                slot.sparams.reasoning_budget_end.size(),
                slot.sparams.reasoning_budget_forced.size());
        }
    }

    {  // apply logit bias
        const auto& logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && (logit_bias->is_object() || logit_bias->is_array())) {
            slot.sparams.logit_bias.clear(); // only clear if user sets it
        }
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_n_vocab(model);
            for (const auto& el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    }
                    else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    }
                    else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            slot.sparams.logit_bias[tok] = bias;
                        }
                    }
                    else if (el[0].is_string()) {
                        auto toks = common_tokenize(model, el[0].get<std::string>(), false);
                        for (auto tok : toks) {
                            slot.sparams.logit_bias[tok] = bias;
                        }
                    }
                }
            }
        }
        else if (logit_bias != data.end() && logit_bias->is_object()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto& el : logit_bias->items()) {
                float bias;
                const auto& key = el.key();
                const auto& value = el.value();
                if (value.is_number()) {
                    bias = value.get<float>();
                }
                else if (value.is_boolean() && !value.get<bool>()) {
                    bias = -INFINITY;
                }
                else {
                    continue;
                }

                char* end;
                llama_token tok = strtol(key.c_str(), &end, 10);
                if (*end == 0) {
                    if (tok >= 0 && tok < n_vocab) {
                        slot.sparams.logit_bias[tok] = bias;
                    }
                }
                else {
                    auto toks = common_tokenize(model, key, false);
                    for (auto tok : toks) {
                        slot.sparams.logit_bias[tok] = bias;
                    }
                }
            }
        }
        if (json_value(data, "ignore_eos", false) && has_eos_token) {
            slot.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
        }

    }

    {
        // ban string
        int32_t banbuffer_size = json_value(data, "banbuffer_size", 0);
        slot.n_buffer = 0; // Ensure buffer calculation starts fresh for this slot
        slot.rewind_count_max = json_value(data, "rewind_count_max", -1);

        const auto& banned_strings = data.find("banned_strings");
        if (banned_strings != data.end() && banned_strings->is_array()) {
            slot.ban_phrases.clear();
            for (const auto& val : data["banned_strings"]) {
                if (val.is_string()) {
                    std::string s = val.get<std::string>();
                    if (!s.empty()) {
                        s = string_lower(s);
                        // Use string length instead of token count
                        if (s.length() > slot.n_buffer) {
                            slot.n_buffer = s.length();
                        }
                        slot.ban_phrases.push_back(s);
                    }
                }
            }
            std::sort(slot.ban_phrases.begin(), slot.ban_phrases.end(), [](const std::string& a, const std::string& b) {
                return a.length() > b.length();
                });
        } else if (params_base.ban_phrases.size() > 0) {
            if (params_base.n_buffer == 0) {
                slot.ban_phrases.clear();
                std::sort(params_base.ban_phrases.begin(), params_base.ban_phrases.end(), [](const std::string & a, const std::string & b) {
                    return a.length() > b.length();
                    });
                for (auto & val : params_base.ban_phrases) {
                    if (!val.empty()) {
                        val = string_lower(val);
                        // Use string length instead of token count
                        if (val.length() > slot.n_buffer) {
                            slot.n_buffer = val.length();
                        }
                        slot.ban_phrases.push_back(val);
                    }
                }
                params_base.n_buffer = slot.n_buffer + 1; // buffer is longest string + 1
            } else {
                slot.ban_phrases = params_base.ban_phrases;
                slot.n_buffer = params_base.n_buffer;
            }
        }

        // ban regex
        slot.ban_regex.clear();
        const auto& banned_regex = data.find("banned_regex");
        if (banned_regex != data.end() && banned_regex->is_array()) {
            for (const auto& val : data["banned_regex"]) {
                if (val.is_string()) {
                    std::string s = val.get<std::string>();
                    if (!s.empty()) {
                        try {
                            std::regex re(s);
                            slot.ban_regex.push_back(s);
                            if (s.length() > slot.n_buffer) {
                                slot.n_buffer = s.length();
                            }
                        } catch (const std::regex_error& e) {
                            send_error(task, "Invalid regex in banned_regex: " + s, ERROR_TYPE_INVALID_REQUEST);
                            return false;
                        }
                    }
                }
            }
        }

        // ban regex case insensitive
        slot.ban_regex_ci.clear();
        const auto& banned_regex_ci = data.find("banned_regex_case_insensitive");
        if (banned_regex_ci != data.end() && banned_regex_ci->is_array()) {
            for (const auto& val : data["banned_regex_case_insensitive"]) {
                if (val.is_string()) {
                    std::string s = val.get<std::string>();
                    if (!s.empty()) {
                        try {
                            std::regex re(s, std::regex_constants::icase);
                            slot.ban_regex_ci.push_back(s);
                            if (s.length() > slot.n_buffer) {
                                slot.n_buffer = s.length();
                            }
                        } catch (const std::regex_error& e) {
                            send_error(task, "Invalid regex in banned_regex_case_insensitive: " + s, ERROR_TYPE_INVALID_REQUEST);
                            return false;
                        }
                    }
                }
            }
        }
        if (banned_regex_ci != data.end() || banned_regex != data.end() || banned_strings != data.end()) {
            if (banbuffer_size > 0) {
                slot.n_buffer = banbuffer_size;
            } else {
                slot.n_buffer = slot.n_buffer + 1; // buffer is longest string/regex + 1
            }
        }
		slot.logit_bias = slot.sparams.logit_bias; // keep a copy to restore
        slot.ban_phrases_bias = json_value(data, "banned_bias", params_base.ban_phrases_bias);
        slot.banned_n = json_value(data, "banned_n", params_base.banned_n);
    }

    do  // populate allowlist biases
    {
        // TODO: JSON parsing for rules and keywords
        slot.allow_ruless = params_base.allow_ruless;
        if (slot.allow_ruless.size() == 0) {
            slot.allow_biasess.clear();
            break;
        }
        slot.allow_kws = params_base.allow_kws;

        slot.allow_pieces = params_base.allow_pieces;
        const auto& allowlist_piece_array = data.find("allowlist_piece_array");
        if (allowlist_piece_array != data.end() && allowlist_piece_array->is_array()) {
            slot.allow_pieces.clear();
            for (const auto& piece: *allowlist_piece_array) {
                if (piece.is_string()) {
                    slot.allow_pieces.push_back(piece.get<std::string>());
                }
            }
        }

        slot.allow_kw_delay = json_value(data, "allowlist_keyword_delay", params_base.allow_kw_delay);
        // end of allowlist criteria update

        const int32_t n_vocab = populate_vocab_pieces();

        std::unordered_set<llama_token> allow_settoken;
        for (const auto& piece: slot.allow_pieces) {
            for (const auto token: common_tokenize(model, piece, false, true)) {
                allow_settoken.insert(token);
            }
        }

        auto n_rules = slot.allow_ruless.size();
        if (n_rules > slot.allow_kws.size() + 1) {
            // one more rules than keyword, last rules do not expire
            n_rules = slot.allow_kws.size() + 1;
            slot.allow_ruless.resize(n_rules);
        } else if (n_rules < slot.allow_kws.size()) {
            // every rules expire
            slot.allow_kws.resize(n_rules);
        }
        slot.allow_biasess.resize(n_rules);

        for (size_t i = 0; i < n_rules; ++i) {
            const auto& rules = slot.allow_ruless[i];
            if ((i < slot.allow_ruless_prev.size()) && (rules == slot.allow_ruless_prev[i])) {
                continue;
            }
            LLAMA_LOG_DEBUG("%s: allowlist %zu is new\n", __func__, i);

            auto& biases = slot.allow_biasess[i];
            biases.resize(n_vocab);

            std::vector<uint32_t> cpts;
            std::vector<std::string> scripts;
            for (size_t id = 0; id < n_vocab; ++id) {
                const size_t n_cpt = llama_fill_from_utf8(&vocab_pieces[id], &cpts, &scripts);
                float bias = -INFINITY;

                // each codepoint must be found in
                for (size_t j = 0; j < n_cpt; ++j) {
                    bool in_rule = false;

                    // at least one rule
                    for (const auto& rule: rules) {
                        const bool in_range = (std::get<0>(rule) <= cpts[j]) && (cpts[j] <= std::get<1>(rule));
                        in_rule = in_range && ((std::get<2>(rule) == "*") || std::get<2>(rule) == scripts[j]);
                        if (in_rule) {
                            // earlier rule has higher priority
                            bias = std::max(bias, std::get<3>(rule));
                            break;
                        }
                    }
                    if (!in_rule) {
                        if ((scripts[j] == "common") || (scripts[j] == "inherited")) {
                            // for common or inherited codepoints (e.g. whitespace), defer to other codepoints in the token
                            continue;
                        }

                        // to shadow realm
                        bias = -INFINITY;
                        break;
                    }
                }
                biases[id] = bias;
            }

            float max_bias = -INFINITY;
            for (const auto& rule: rules) {
                max_bias = std::max(max_bias, std::get<3>(rule));
            }
            for (const auto token: allow_settoken) {
                biases[token] = max_bias;
            }
        }
    } while (false);
    slot.allow_ruless_prev = slot.allow_ruless;

    if (llama_model_has_recurrent(llama_get_model(slot.ctx))) {
        params_base.can_ban_phrases = false;
        bool do_checkpoint = params_base.ctx_checkpoints_n > 0;
        // make checkpoints only for completion tasks
        do_checkpoint = do_checkpoint && task.type == SERVER_TASK_TYPE_COMPLETION;
        // make a checkpoint of the parts of the memory that cannot be rolled back.
        // checkpoints are created only if:
        // - the model architecture is marked as recurrent or hybrid
        //
        // TODO: try to make this conditional on the context or the memory module, instead of the model type
        params_base.do_checkpoint = do_checkpoint;
        if (slot.n_buffer != 0) {
            LLAMA_LOG_WARN("banned strings is not supported by recurrent model, it will be disabled.\n");
        }
        if (params_base.ctx_shift && !llama_model_is_hybrid(llama_get_model(slot.ctx))) {
            // PXA_HYBRID_CTX_SHIFT_v1: only PURE recurrent models (mamba-class) cannot shift —
            // their whole memory is one rolling state with no per-token cells to slide. HYBRIDS
            // (qwen3next/qwen35moe Gated-DeltaNet) CAN shift: the attention KV slides via the
            // standard K re-rope (build_k_shift already skips is_recurrent layers), and the
            // recurrent state is position-free and already summarizes the discarded window, so
            // keeping it is strictly better than truncation. Blanket-disabling here meant any
            // slot that outgrew per-slot n_ctx died with a 500 ("context shift is disabled")
            // that frontends swallow — the production "long conversation goes silent" failure.
            params_base.ctx_shift = false;
            LOG_WARNING("%s\n", "ctx_shift is not supported by recurrent model, it will be disabled");
        }
    }
    if (llama_model_is_split_mode_graph(llama_get_model(slot.ctx))) {
        if (params_base.ctx_shift) {
            params_base.ctx_shift = false;
            LOG_WARNING("%s\n", "ctx_shift is not implemented for split mode graph, it will be disabled");
        }
    }
    {
        const auto& stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            slot.params.antiprompt.clear();
            for (const auto& word : *stop) {
                if (!word.empty()) {
                    slot.params.antiprompt.push_back(word);
                }
            }
        }
    }

    {
        const auto samplers = data.find("samplers");
        if (samplers != data.end()) {
            if (samplers->is_array()) {
                slot.sparams.samplers_sequence = llama_sampling_types_from_names(*samplers, false);
            }
            else if (samplers->is_string()) {
                slot.sparams.samplers_sequence = llama_sampling_types_from_chars(samplers->get<std::string>());
            }
            else {
                slot.sparams.samplers_sequence = default_sparams.samplers_sequence;
            }
        }
    }

    {
        try
        {
            if (slot.ctx_sampling != nullptr) {
                common_sampler_free(slot.ctx_sampling);
            }
            slot.ctx_sampling = common_sampler_init(model, slot.sparams);
        }
        catch (std::exception & e) {
            std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
            send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (slot.ctx_sampling == nullptr) {
            // for now, the only error that may happen here is invalid grammar
            send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
    }

    do  // populate expiring logit bias
    {
        const auto prev_elb_params = slot.sparams.elb_params;

        const auto& expiring_logit_bias = data.find("expiring_logit_bias");
        if (expiring_logit_bias != data.end() && expiring_logit_bias->is_array()) {
            // has new params from api
            std::string content;
            for (const auto& line: data["expiring_logit_bias"]) {
                if (line.is_string()) {
                    content += line.get<std::string>() + "\n";
                }
            }
            try {
                argparse_expiring_logit_bias(content, slot.sparams);
            } catch (const std::invalid_argument& e) {
                std::cerr << e.what() << '\n';
            } catch (const std::out_of_range& e) {
                std::cerr << e.what() << '\n';
            }
        }

        const auto& elb_params = slot.sparams.elb_params;
        if (elb_params.empty()) {
            slot.ctx_sampling->elb_states.clear();
            break;
        }

        if (!slot.prev_elb_states.empty() && (elb_params == prev_elb_params)) {
            // reset and reuse previous states
            slot.ctx_sampling->elb_states = slot.prev_elb_states;
            for (auto& elb_state: slot.ctx_sampling->elb_states) {
                elb_state.countup = 0;
            }
            break;
        }

        const auto n_elb_param = elb_params.size();

        slot.ctx_sampling->elb_states.clear();
        slot.ctx_sampling->elb_states.reserve(n_elb_param);

        // 1 state <-> 1 exitword <-> 1+ entries
        for (int32_t i = 0; i < elb_params.size(); ++i) {
            const auto& [entries, exitword, op] = elb_params[i];

            if (op == ">>") {
                for (auto& elb_state: slot.ctx_sampling->elb_states) {
                    if (elb_state.jumpword.empty()) {
                        elb_state.jumpword = exitword;
                        elb_state.jump_idx = i + 1;
                        elb_state.search_word_len = std::max(elb_state.exitword.length(), elb_state.jumpword.length());
                    }
                }
            }

            slot.ctx_sampling->elb_states.push_back({ { }, { }, exitword, 0, 0, 0, "", 0, exitword.length() });

            auto& first_tokens = slot.ctx_sampling->elb_states.back().first_tokens;
            auto& other_tokens = slot.ctx_sampling->elb_states.back().other_tokens;
            auto& delay = slot.ctx_sampling->elb_states.back().delay;
            auto& max_cond_len = slot.ctx_sampling->elb_states.back().max_cond_len;

            // 1 entry <-> 1 phrase <-> 1+ biases
            for (auto& entry: entries) {
                auto biases = entry.biases;
                if (biases.empty()) {
                    // expiring sampler bias
                    continue;
                }

                // expiring logit bias
                for (const auto& phrase: entry.phrases) {
                    auto duration = entry.duration;

                    const auto ids = common_tokenize(model, phrase, false, true);
                    if (!entry.is_range) {
                        // extrapolate
                        biases.resize(ids.size(), biases.back());
                    } else if (ids.size() == 1) {
                        biases[0] = biases.back();
                        biases.resize(1);
                    } else {
                        // interpolate
                        float bb = biases.back();
                        const float inc = (bb - biases.front()) / (ids.size() - 1);
                        biases.resize(ids.size());
                        for (int32_t j = ids.size() - 1; j >= 0; --j) {
                            biases[j] = bb;
                            bb -= inc;
                        }
                    }

                    if (biases[0] != 0.0f) {
                        // cond is piece for first_tokens (no match to bias)
                        first_tokens.push_back({ ids[0], biases[0], size_t(duration), common_token_to_piece(ctx, ids[0], true) });
                    }

                    int32_t m = 1;
                    if (duration < 0) {
                        // -1 is smallest infinite duration
                        duration ^= 0x7FFFFFFF;
                        m = -1;
                    }

                    std::string cond;
                    for (int32_t j = 1; j < ids.size(); ++j) {
                        // cond collects preceding string for other_tokens (match to bias)
                        cond += common_token_to_piece(ctx, ids[j - 1], true);
                        if (biases[j] == 0.0f) {
                            continue;
                        } else if (biases[j] > 0.0f) {
                            delay = std::max(size_t(duration + m * j), delay);
                        }
                        other_tokens.push_back({ ids[j], biases[j], size_t(duration + m * j), cond });
                    }
                    max_cond_len = std::max(int32_t(cond.length()), max_cond_len);
                }
            }
        }

        // sort by duration in descending order
        for (auto& elb_state: slot.ctx_sampling->elb_states) {
            std::sort(elb_state.first_tokens.begin(), elb_state.first_tokens.end(), [](const auto& a, const auto& b) {
                return a.duration > b.duration;
            });
            std::sort(elb_state.other_tokens.begin(), elb_state.other_tokens.end(), [](const auto& a, const auto& b) {
                return a.duration > b.duration;
            });
        }
    } while (false);
    slot.prev_elb_states = slot.ctx_sampling->elb_states;

    slot.command = SLOT_COMMAND_LOAD_PROMPT;
    // slot.prompt_tokens.clear();

    LOG_INFO("slot is processing task", {
        {"id_slot", slot.id},
        {"id_task", slot.id_task},
        });
    slot.task = std::make_unique<const server_task>(std::move(task));
    return true;
}

void server_context::kv_cache_clear() {
    LOG_VERBOSE("clearing KV cache", {});

    // clear the entire KV cache
    llama_kv_cache_clear(ctx);
    for (auto & slot : slots) {
        if (slot.spec == nullptr) {
            continue;
        }

        common_speculative_clear_sequence_hidden(slot.spec, slot.id);
        if (auto * ctx_companion = common_speculative_get_companion_ctx(slot.spec); ctx_companion != nullptr) {
            llama_kv_cache_clear(ctx_companion);
        }
    }
    clean_kv_cache = false;
}

void server_context::system_prompt_update() {
    LOG_VERBOSE("system prompt update", {
        {"system_prompt", system_prompt},
        });

    kv_cache_clear();
    system_tokens.clear();

    if (!system_prompt.empty()) {
        system_tokens = ::common_tokenize(ctx, system_prompt, true);

        const int32_t n_batch = llama_n_batch(ctx);
        const int32_t n_tokens_prompt = system_tokens.size();

        for (int32_t i = 0; i < n_tokens_prompt; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, n_tokens_prompt - i);

            common_batch_clear(batch);

            for (int32_t j = 0; j < n_tokens; ++j) {
                common_batch_add(batch, system_tokens[i + j], i + j, { 0 }, false);
            }

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERROR("llama_decode() failed", {});
                return;
            }
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i <= params_base.n_parallel; ++i) {
            llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
        }
    }

    system_need_update = false;
}

bool server_context::system_prompt_set(const std::string& sys_prompt) {
    system_prompt = sys_prompt;

    LOG_VERBOSE("system prompt process", {
        {"system_prompt",  system_prompt},
        });

    // release all slots
    for (server_slot& slot : slots) {
        slot.release();
        slot.cache_tokens.clear();
        slot.n_past = 0;
        slot.n_past_prompt = 0;
        slot.n_past_offset = 0;
        slot.n_discarded_prompt = 0;
        slot.n_kept_prompt = 0;
        slot.n_prompt_tokens_cache = 0;
        slot.server_cached_prompt.checkpoints.clear();
        slot.checkpoint_pos = 0;
        slot.do_checkpoint = false;
        if (slot.ctx_sampling != nullptr) {
            common_sampler_reset(slot.ctx_sampling);
        }
    }
    if (prompt_cache) {
        prompt_cache->states.clear();
    }

    system_need_update = true;
    return true;
}

// keep in sync with process_token(completion_token_output& result, server_slot& slot)
bool server_context::has_next_token(const completion_token_output& result,  server_slot& slot) {
    bool next = true;
    //std::string generate_text = slot.generated_text + result.text_to_send;
    //bool incomplete = validate_utf8(generate_text) < generate_text.size();
    //if (incomplete) {
    //    next = true;
    //}
    if (slot.n_decoded > 0 && !slot.has_budget(params_base)) {
        next = false;
    }
    // PXA_SOFTFAIL_BREAKER_v1 (keep in sync with process_token)
    {
        static const int32_t softfail_max_consec = []() {
            const char * e = getenv("PXA_SOFTFAIL_MAX_CONSEC");
            const int v = e ? atoi(e) : 16;
            return v > 0 ? v : 16;
        }();
        if (slot.n_softfail_consecutive >= softfail_max_consec) {
            next = false;
        }
    }
    if (llama_token_is_eog(model, result.tok)) {
        next = false;
    }
    auto n_ctx_train = llama_n_ctx_train(model);
    if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.ga_n == 1
        && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
        next = false;
    }
    return next;
}


bool server_context::process_token(completion_token_output& result, server_slot& slot) {
    // remember which tokens were sampled - used for repetition penalties during sampling
    const std::string token_str = result.text_to_send;
    slot.sampled = result.tok;

    // search stop word and delete it
    slot.last_gentxt_size = slot.generated_text.size();
    slot.generated_text += token_str;
    slot.has_next_token = true;

    if (slot.ctx_sampling->params.use_penalty_prompt_tokens && result.tok != -1) {
        // we can change penalty_prompt_tokens because it is always created from scratch each request
        slot.ctx_sampling->params.penalty_prompt_tokens.push_back(result.tok);
    }

    // check if there is incomplete UTF-8 character at the end
    bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

    if (!incomplete) {
        size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

        const std::string str_test = slot.generated_text.substr(pos);
        bool send_text = true;

        size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
        if (stop_pos != std::string::npos) {
            slot.generated_text.erase(
                slot.generated_text.begin() + pos + stop_pos,
                slot.generated_text.end());
            pos = std::min(slot.n_sent_text, slot.generated_text.size());
        }
        else if (slot.has_next_token && !llama_token_is_eog(model, result.tok)) {
            stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
            send_text = stop_pos == std::string::npos;
        }

        // check if there is any token to predict
        if (send_text) {
            // no send the stop word in the response
            result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
            slot.n_sent_text += result.text_to_send.size();
            // add the token to slot queue and cache
        }
        else {
            result.text_to_send = "";
        }

        slot.add_token_string(result);
        if (slot.params.stream) {
            send_partial_response(slot, result);
        }
    }

    if (incomplete) {
        slot.has_next_token = true;
    }

    // check the limits
    if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
        slot.stopped_limit = true;
        slot.has_next_token = false;

        LOG_VERBOSE("stopped by limit", {
            {"id_slot",   slot.id},
            {"id_task",   slot.id_task},
            {"n_decoded", slot.n_decoded},
            {"n_predict", slot.params.n_predict},
            });
    }

    // PXA_SOFTFAIL_BREAKER_v1: stop a generation stuck emitting the unsampleable-fallback token
    // before it runs to the context limit and starves co-resident slots under -sm layer. The
    // sampler surfaces the event per token; the count is maintained per slot at the sample site.
    {
        static const int32_t softfail_max_consec = []() {
            const char * e = getenv("PXA_SOFTFAIL_MAX_CONSEC");
            const int v = e ? atoi(e) : 16;
            return v > 0 ? v : 16;
        }();
        if (slot.has_next_token && slot.n_softfail_consecutive >= softfail_max_consec) {
            slot.truncated = true;
            slot.stopped_limit = true;
            slot.has_next_token = false;
            LOG_WARNING("PXA_SOFTFAIL_BREAKER: stopping degenerate generation", {
                { "id_slot", slot.id },
                { "id_task", slot.id_task },
                { "n_softfail_consecutive", slot.n_softfail_consecutive },
                { "n_decoded", slot.n_decoded },
            });
        }
    }

    if (llama_token_is_eog(model, result.tok)) {
        slot.stopped_eos = true;
        slot.has_next_token = false;

        LOG_VERBOSE("eos token found", {});
    }

    auto n_ctx_train = llama_n_ctx_train(model);
    if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.ga_n == 1
        && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
        LOG_WARNING("n_predict is not set and self-context extend is disabled."
            " Limiting generated tokens to n_ctx_train to avoid EOS-less generation infinite loop", {
        { "id_slot",              slot.id },
        { "params.n_predict",     slot.params.n_predict },
        { "slot.n_prompt_tokens", slot.n_prompt_tokens },
        { "slot.n_decoded",       slot.n_decoded },
        { "slot.n_predict",       slot.n_predict },
        { "n_slots",              params_base.n_parallel },
        { "slot.n_ctx",           slot.n_ctx },
        { "n_ctx",                n_ctx },
        { "n_ctx_train",          n_ctx_train },
        { "ga_n",                 slot.ga_n },
            });
        slot.truncated = true;
        slot.stopped_limit = true;
        slot.has_next_token = false; // stop prediction
    }
    log_text(params_base, "token:"+result.text_to_send);
    LOG_VERBOSE("next token", {
        {"id_slot",        slot.id},
        {"id_task",        slot.id_task},
        {"token",          result.tok},
        {"token_text",     tokens_to_output_formatted_string(ctx, result.tok)},
        {"has_next_token", slot.has_next_token},
        {"n_remain",       slot.n_remaining},
        {"n_decoded",      slot.n_decoded},
        {"stopped_eos",    slot.stopped_eos},
        {"stopped_word",   slot.stopped_word},
        {"stopped_limit",  slot.stopped_limit},
        {"stopping_word",  slot.stopping_word},
        });

    return slot.has_next_token; // continue
}

void server_context::populate_token_probs(const server_slot& slot, completion_token_output& result, bool post_sampling, bool special, int idx) {
    size_t n_probs = slot.sparams.n_probs;
    size_t n_vocab = llama_n_vocab(llama_get_model(ctx));

    if (post_sampling) {
        const auto* cur_p = common_sampler_get_candidates(slot.ctx_sampling);
        const size_t max_probs = cur_p->size;

        // set probability for sampled token
        for (size_t i = 0; i < max_probs; i++) {
            if (cur_p->data[i].id == result.tok) {
                result.prob = cur_p->data[i].p;
                break;
            }
        }

        // set probability for top n_probs tokens
        result.probs.reserve(max_probs);
        for (size_t i = 0; i < std::min(max_probs, n_probs); i++) {
            result.probs.push_back({
                cur_p->data[i].id,
                common_token_to_piece(ctx, cur_p->data[i].id, special),
                cur_p->data[i].p
                });
        }
    }
    else {
        auto&& [sampled_token_p, cur] = get_token_probabilities(ctx, idx, result.tok, n_probs);

        // set probability for sampled token
        result.prob = sampled_token_p;

        // set probability for top n_probs tokens
        result.probs.reserve(n_probs);
        for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
            result.probs.push_back({
                cur[i].id,
                common_token_to_piece(ctx, cur[i].id, special),
                cur[i].p
                });
        }
    }
}

json server_context::get_formated_generation(const server_slot& slot) const {
    const auto eos_bias = slot.sparams.logit_bias.find(llama_token_eos(model));
    const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() && eos_bias->second < 0.0f && std::isinf(eos_bias->second);

    std::vector<std::string> samplers_sequence;
    samplers_sequence.reserve(slot.sparams.samplers_sequence.size());
    for (const auto& sampler_type : slot.sparams.samplers_sequence) {
        samplers_sequence.emplace_back(llama_sampling_type_to_str(sampler_type));
    }

    auto grammar_triggers = json::array();
    for (const auto& trigger : slot.sparams.grammar_triggers) {
        grammar_triggers.push_back(trigger.to_json<json>());
    }

    return json{
        {"n_ctx",                     slot.n_ctx},
        {"n_predict",                 slot.n_predict},     // Server configured n_predict
        {"model",                     params_base.model_alias},
        {"seed",                      slot.sparams.seed},
        {"temperature",               slot.sparams.temp},
        {"dynatemp_range",            slot.sparams.dynatemp_range},
        {"dynatemp_exponent",         slot.sparams.dynatemp_exponent},
        {"top_k",                     slot.sparams.top_k},
        {"top_p",                     slot.sparams.top_p},
        {"min_p",                     slot.sparams.min_p},
        {"tfs_z",                     slot.sparams.tfs_z},
        {"typical_p",                 slot.sparams.typical_p},
        {"repeat_last_n",             slot.sparams.penalty_last_n},
        {"repeat_penalty",            slot.sparams.penalty_repeat},
        {"presence_penalty",          slot.sparams.penalty_present},
        {"frequency_penalty",         slot.sparams.penalty_freq},
        {"penalty_prompt_tokens",     slot.sparams.penalty_prompt_tokens},
        {"use_penalty_prompt_tokens", slot.sparams.use_penalty_prompt_tokens},
        {"dry_multiplier",            slot.sparams.dry_multiplier},
        {"dry_base",                  slot.sparams.dry_base},
        {"dry_allowed_length",        slot.sparams.dry_allowed_length},
        {"dry_penalty_last_n",        slot.sparams.dry_penalty_last_n},
        {"dry_sequence_breakers",     slot.sparams.dry_sequence_breakers},
        {"mirostat",                  slot.sparams.mirostat},
        {"mirostat_tau",              slot.sparams.mirostat_tau},
        {"mirostat_eta",              slot.sparams.mirostat_eta},
        {"adaptive_target",           slot.sparams.adaptive_target},
        {"adaptive_decay",            slot.sparams.adaptive_decay},
        {"adaptive_updt_w_cur",       slot.sparams.adaptive_updt_w_cur},
        {"penalize_nl",               slot.sparams.penalize_nl},
        {"stop",                      slot.params.antiprompt},
        {"max_tokens",                slot.params.n_predict}, // User configured n_predict
        {"n_keep",                    slot.params.n_keep},
        {"n_discard",                 slot.params.n_discard},
        {"ignore_eos",                ignore_eos},
        {"stream",                    slot.params.stream},
        {"logit_bias",                slot.sparams.logit_bias},
        {"n_probs",                   slot.sparams.n_probs},
        {"min_keep",                  slot.sparams.min_keep},
        {"grammar",                   slot.sparams.grammar.grammar},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          slot.sparams.preserved_tokens},
        {"chat_format",               common_chat_format_name(slot.params.chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(slot.params.chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      slot.params.chat_parser_params.reasoning_in_content},
        {"samplers",                  samplers_sequence}
    };
}

void server_context::send_error(const server_task& task, const std::string& error, const enum error_type type) {
    send_error(task.id, task.id_multi, error, type);
}

void server_context::send_error(const server_slot& slot, const std::string& error, const enum error_type type) {
    send_error(slot.id_task, slot.id_multi, error, type);
}

void server_context::send_error(const int id_task, const int id_multi, const std::string& error, const enum error_type type ) {
    LOG_ERROR("task error", {
        {"id_multi", id_multi},
        {"id_task", id_task},
        {"error", error},
        });

    auto res = std::make_unique<server_task_result_error>();
    res->id = id_task;
    res->id_multi = id_multi;
    res->stop = false;
    res->error = true;
    res->err_type = type;
    res->err_msg = error;
    queue_results.send(std::move(res));
}

// if multimodal is enabled, send an error and return false
bool server_context::check_no_mtmd(const int id_task) {
    if (mctx) {
        int id_multi = 0;
        send_error(id_task, id_multi, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
        return false;
    }
    return true;
}

void server_context::send_partial_response(server_slot& slot, completion_token_output tkn) {
    if (slot.task == nullptr) {
        return;
    }
    auto res = std::make_unique<server_task_result_cmpl_partial>();
    res->final_result = false;
    res->id = slot.id_task;
    res->id_multi = slot.id_multi;
    res->index = slot.task->index;
    res->error = false;
    res->stop = false;
    res->stream = slot.params.stream;
    res->content = tkn.text_to_send;
    res->post_sampling_probs = slot.params.post_sampling_probs;
    res->oaicompat = slot.params.oaicompat;
    res->oaicompat_model = slot.task->params.oaicompat_model;
    res->oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;
    res->oai_resp_id = slot.oai_resp_id;
    res->oai_resp_reasoning_id = slot.oai_resp_reasoning_id;
    res->oai_resp_message_id = slot.oai_resp_message_id;
    res->oai_resp_fc_id = slot.oai_resp_fc_id;
    res->n_decoded = slot.n_decoded;
    res->n_prompt_tokens = slot.n_prompt_tokens;
    // PXA_STREAM_SLIM: the OAI-chat partial serializer never reads res->data — skip building this
    // json on the decode thread for chat streams (it is only consumed by the non-chat partial paths).
    if (slot.params.oaicompat != OAICOMPAT_TYPE_CHAT) {
    res->data = json{
        {"content",    tkn.text_to_send},
        {"stop",       false},
        {"id_slot",    slot.id},
        {"multimodal", false}
    };
    }
    if (server_response_needs_chat_parse(slot.params.oaicompat)) {
        slot.update_chat_msg(true, res->oaicompat_msg_diffs);

        res->anthropic_has_reasoning = !slot.chat_msg.reasoning_content.empty();

        res->anthropic_thinking_block_started = slot.anthropic_thinking_block_started;
        res->anthropic_text_block_started = slot.anthropic_text_block_started;

        res->oai_resp_thinking_block_started = slot.oai_resp_thinking_block_started;
        res->oai_resp_text_block_started = slot.oai_resp_text_block_started;

        for (const auto& diff : res->oaicompat_msg_diffs) {
            if (!diff.reasoning_content_delta.empty() && !slot.anthropic_thinking_block_started) {
                slot.anthropic_thinking_block_started = true;
            }
            if (!diff.content_delta.empty() && !slot.anthropic_text_block_started) {
                slot.anthropic_text_block_started = true;
            }
            if (!diff.reasoning_content_delta.empty() && !slot.oai_resp_thinking_block_started) {
                slot.oai_resp_thinking_block_started = true;
            }
            if (!diff.content_delta.empty() && !slot.oai_resp_text_block_started) {
                slot.oai_resp_text_block_started = true;
            }
            if (!diff.tool_call_delta.name.empty()) {
                slot.oai_resp_fc_id = diff.tool_call_delta.id;
            }
        }
    }

    // populate res->probs_output
    if (slot.sparams.n_probs > 0) {
        res->probs_output = { tkn }; // copy the token probs
        res->data["completion_probabilities"] = probs_vector_to_json(ctx, res->probs_output);
    }

    if (slot.oaicompat && slot.params.oaicompat != OAICOMPAT_TYPE_CHAT) { // PXA_STREAM_SLIM
        res->data["oaicompat_token_ctr"] = slot.n_decoded;
        res->data["model"] = slot.oaicompat_model;
    }

    // populate timings if this is final response or timings_per_token is enabled
    if (slot.params.timings_per_token) {
        res->timings = slot.get_timings();
    }
    queue_results.send(std::move(res));
}

void server_context::send_final_response(server_slot& slot) {
    // PXA_UTF8_FINAL_v1 (2026-07-30): a generation cut mid-multibyte-codepoint (n_predict
    // cap / ignore_eos runaway) leaves a trailing partial UTF-8 sequence in generated_text.
    // nlohmann::json refuses to serialise it and the WHOLE response 500s
    // ("incomplete UTF-8 string; last byte: 0x..") AFTER a successful generation — a
    // content-dependent, seed-reproducible failure that masquerades as flaky infrastructure.
    // The streaming path already holds back incomplete sequences per-chunk (process_token);
    // apply the same discipline to the final payload: drop the unrenderable tail bytes.
    {
        const size_t utf8_ok = validate_utf8(slot.generated_text);
        if (utf8_ok < slot.generated_text.size()) {
            LOG_WARNING("final response: dropping incomplete trailing UTF-8 sequence", {
                {"id_slot",       slot.id},
                {"id_task",       slot.id_task},
                {"dropped_bytes", slot.generated_text.size() - utf8_ok},
            });
            slot.generated_text.resize(utf8_ok);
        }
    }
    auto res = std::make_unique<server_task_result_cmpl_final>();
    res->final_result = true;
    res->id = slot.id_task;
    res->id_multi = slot.id_multi;
    res->index = slot.task->index;
    res->error = false;
    res->stop = true;   // "this is the final result" (base-class flag)
    // The actual stop reason. EOS wins over WORD wins over LIMIT: the budget check runs
    // before the EOG check, so a model that emits EOS exactly at the cap really did finish.
    // Anything else is reported as a non-clean stop (-> finish_reason "length"), which is
    // the safe direction: a false "length" only makes a harness re-check, while a false
    // "stop" makes it append a truncated runaway to history and poison the next turn.
    res->stop_kind = slot.stopped_eos   ? STOP_TYPE_EOS   :
                       slot.stopped_word  ? STOP_TYPE_WORD  :
                       slot.stopped_limit ? STOP_TYPE_LIMIT :
                                            STOP_TYPE_NONE;
    res->stream = slot.params.stream;
    res->include_usage = slot.params.include_usage;
    res->content = slot.generated_text;
    res->timings = slot.get_timings();
    res->post_sampling_probs = slot.params.post_sampling_probs;
    res->oaicompat = slot.params.oaicompat;
    res->oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;
    if (server_response_needs_chat_parse(slot.params.oaicompat)) {
        res->oaicompat_msg = slot.update_chat_msg(false, res->oaicompat_msg_diffs);
    }
    res->oai_resp_id = slot.oai_resp_id;
    res->oai_resp_reasoning_id = slot.oai_resp_reasoning_id;
    res->oai_resp_message_id = slot.oai_resp_message_id;
    res->n_decoded = slot.n_decoded;
    res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
    res->anthropic_thinking_block_started = slot.anthropic_thinking_block_started;
    res->anthropic_text_block_started = slot.anthropic_text_block_started;
    res->n_prompt_tokens = slot.n_prompt_tokens;
    res->oaicompat_model = slot.task->params.oaicompat_model;
    res->data = json{
        {"content",             !slot.params.stream ? slot.generated_text : ""},
        {"generated_text",      slot.generated_text},  // Always include full text for finish_reason logic
        {"id_slot",             slot.id},
        {"stop",                true},
        {"model",               params_base.model_alias},
        {"tokens_predicted",    slot.n_decoded},
        {"tokens_evaluated",    slot.n_prompt_tokens},
        {"generation_settings", get_formated_generation(slot)},
        {"prompt",              slot.prompt},
        {"truncated",           slot.truncated},
        {"stopped_eos",         slot.stopped_eos},
        {"stopped_word",        slot.stopped_word},
        {"stopped_limit",       slot.stopped_limit},
        {"stopping_word",       slot.stopping_word},
        {"tokens_cached",       slot.n_past},
        {"timings",             slot.get_formated_timings()},
        //{"oaicompat_chat_format",  slot.params.oaicompat_chat_format},
    };

    // populate res->probs_output
    if (slot.sparams.n_probs > 0) {
        res->probs_output = std::vector<completion_token_output>(
            slot.generated_token_probs.begin(),
            slot.generated_token_probs.end());
        res->data["completion_probabilities"] = probs_vector_to_json(ctx, res->probs_output);
    }

    if (slot.oaicompat) {
        res->data["oaicompat_token_ctr"] = slot.n_decoded;
        res->data["model"] = slot.oaicompat_model;
    }

    // PXA_EXPERT_LOG_v1: dump this request's expert-routing histogram
    {
        std::string prompt_preview;
        try {
            prompt_preview = slot.prompt.is_string() ? slot.prompt.get<std::string>() : slot.prompt.dump();
        } catch (...) {}
        if (prompt_preview.size() > 200) prompt_preview.resize(200);
        const std::string elog_path = pxa_expert_log_flush(slot.id_task, slot.params.oaicompat_cmpl_id,
                slot.n_prompt_tokens, slot.n_decoded, prompt_preview);
        if (!elog_path.empty()) {
            res->data["pxa_expert_log"] = elog_path;
        }
    }

    queue_results.send(std::move(res));
}

void server_context::send_embedding(const server_slot& slot, const llama_batch& batch) {
    auto res = std::make_unique<server_task_result_embd>();
    res->id = slot.task->id;
    res->index = slot.task->index;
    res->server_task_result::index = slot.task->index;
    res->n_tokens = slot.prompt_tokens.size();
    res->oaicompat = slot.task->params.oaicompat;

    const int n_embd = llama_model_n_embd(model);

    std::vector<float> embd_res(n_embd, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
            continue;
        }

        const float* embd = nullptr;
        if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
        }
        else {
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        }

        if (embd == nullptr) {
            SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

            res->embedding.push_back(std::vector<float>(n_embd, 0.0f));
            continue;
        }

        // normalize only when there is pooling
        if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
            common_embd_normalize(embd, embd_res.data(), n_embd, slot.task->params.embd_normalize);
            res->embedding.push_back(embd_res);
            break;
        }

        res->embedding.emplace_back(embd, embd + n_embd);
    }
    queue_results.send(std::move(res));
}

void server_context::apply_server_biases(server_slot& slot) {
    auto& server_biases = slot.ctx_sampling->server_biases;

    if (slot.allow_idx < slot.allow_biasess.size()) {
        server_biases = &slot.allow_biasess[slot.allow_idx];
    } else {
        server_biases = nullptr;
    }
}

void server_context::request_completion(int id_task, int id_multi, json data, bool infill, bool embedding, server_tokens & inputs) {
    server_task task;
    task.id = id_task;
    task.id_multi = id_multi;
    task.id_target = 0;
    task.data = std::move(data);
    task.infill = infill;
    task.embedding = embedding;
    task.type = SERVER_TASK_TYPE_COMPLETION;
    task.tokens = inputs.clone();
    // when a completion task's prompt array is not a singleton, we split it into multiple requests
    // otherwise, it's a single-prompt task, we actually queue it
    // if there's numbers in the prompt array it will be treated as an array of tokens
    if (task.data.count("prompt") != 0 && task.data.at("prompt").size() > 1) {
        bool numbers = false;
        for (const auto& e : task.data.at("prompt")) {
            if (e.is_number()) {
                numbers = true;
                break;
            }
        }

        // NOTE: split_multiprompt_task() does not handle a mix of strings and numbers,
        // it will completely stall the server. I don't know where the bug for this is.
        //
        // if there are numbers, it needs to be treated like a single prompt,
        // queue_tasks handles a mix of strings and numbers just fine.
        if (numbers) {
            queue_tasks.post(std::move(task));
        }
        else {
            split_multiprompt_task(id_task, task);
        }
    }
    else {
        queue_tasks.post(std::move(task));
    }
}

void server_context::request_cancel(int id_task) {
    server_task task;
    task.type = SERVER_TASK_TYPE_CANCEL;
    task.id_target = id_task;

    queue_tasks.post(std::move(task));
}

void server_context::split_multiprompt_task(int id_multi, server_task& multiprompt_task) {
    auto prompts = multiprompt_task.data.at("prompt");
    const int prompt_count = prompts.size();
    if (prompt_count <= 1) {
        send_error(multiprompt_task, "error while handling multiple prompts");
        return;
    }

    // generate all the ID for subtask
    std::vector<int> subtask_ids(prompt_count);
    for (int i = 0; i < prompt_count; i++) {
        subtask_ids[i] = queue_tasks.get_new_id();
    }

    // queue up the multitask so we can track its subtask progression
    queue_tasks.add_multitask(id_multi, subtask_ids);

    // add subtasks
    for (int i = 0; i < prompt_count; i++) {
        json subtask_data = multiprompt_task.data;
        subtask_data["prompt"] = prompts[i];

        // subtasks inherit everything else (infill mode, embedding mode, etc.)
        request_completion(subtask_ids[i], id_multi, subtask_data, multiprompt_task.infill, multiprompt_task.embedding,
            multiprompt_task.tokens);
    }
}



static size_t save_checkpoints_to_file(const std::string & filename, const std::list<server_prompt_checkpoint> & checkpoints) {
    if (checkpoints.size() == 0) {
        return 0;
    }
    std::ofstream file(filename, std::ios::binary);
    uint32_t magic = LLAMA_STATE_SEQ_MAGIC;
    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    uint32_t version = LLAMA_STATE_SEQ_VERSION;
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    size_t count = checkpoints.size();
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));

    for (const auto & checkpoint : checkpoints) {
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_min), sizeof(checkpoint.pos_min));
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_max), sizeof(checkpoint.pos_max));
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_min_prompt), sizeof(checkpoint.pos_min_prompt));
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_max_prompt), sizeof(checkpoint.pos_max_prompt));
        size_t data_len = checkpoint.data.size();
        file.write(reinterpret_cast<const char *>(&data_len), sizeof(data_len));
        if (data_len > 0) {
            file.write(reinterpret_cast<const char *>(checkpoint.data.data()), data_len * sizeof(uint8_t));
        }
    }
    size_t pos = file.tellp();
    file.close();
    return pos;
}

static size_t load_checkpoints_from_file(const std::string & filename, std::list<server_prompt_checkpoint> & checkpoints) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return 0;
    }
    checkpoints.clear();
    // version checks
    {
        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for checkpoint file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }
    // load the checkpoints
    {
        size_t count;
        file.read(reinterpret_cast<char *>(&count), sizeof(count));

        for (int i = 0; i < count; i++) {
            server_prompt_checkpoint checkpoint;
            file.read(reinterpret_cast<char *>(&checkpoint.pos_min), sizeof(checkpoint.pos_min));
            file.read(reinterpret_cast<char *>(&checkpoint.pos_max), sizeof(checkpoint.pos_max));
            file.read(reinterpret_cast<char *>(&checkpoint.pos_min_prompt), sizeof(checkpoint.pos_min_prompt));
            file.read(reinterpret_cast<char *>(&checkpoint.pos_max_prompt), sizeof(checkpoint.pos_max_prompt));

            size_t data_len;
            file.read(reinterpret_cast<char *>(&data_len), sizeof(data_len));
            if (data_len > 0) {
                checkpoint.data.resize(data_len);
                file.read(reinterpret_cast<char *>(checkpoint.data.data()), data_len * sizeof(uint8_t));
            }
            checkpoints.push_back(checkpoint);
        }
    }
    size_t pos = file.tellg();
    file.close();
    return pos;
}

static size_t save_server_tokens_to_file(const std::string & filename, const server_tokens & tokens) {
    std::ofstream file(filename, std::ios::binary);
    json token_json = tokens.to_json();
    token_json["magic"] = LLAMA_SERVER_MAGIC;
    token_json["version"] = LLAMA_SERVER_VERSION;
    size_t pos = 0;
    if (file.is_open()) {
        file << token_json;
        pos = file.tellp();
        file.close();
    }
    return pos;
}

static size_t load_server_tokens_from_file(const std::string & filename,  server_tokens & tokens) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return 0;
    }
    size_t pos = 0;
    json token_json;
    if (file.is_open()) {
        file >> token_json;
        pos = file.tellg();
        file.close();
    }
    uint32_t magic = token_json.value<uint32_t>("magic", 0);
    uint32_t version = token_json.value<uint32_t>("version", 0);
    if (magic != LLAMA_SERVER_MAGIC || version != LLAMA_SERVER_VERSION) {
        LLAMA_LOG_ERROR("%s: unknown (magic, version) for token file: %08x, %08x\n", __func__, magic, version);
        return 0;
    }
    tokens.from_json(token_json);

    return pos;
}

void server_context::process_single_task(server_task&& task) {
    switch (task.type) {
    case SERVER_TASK_TYPE_COMPLETION:
    case SERVER_TASK_TYPE_INFILL:
    case SERVER_TASK_TYPE_EMBEDDING:
    case SERVER_TASK_TYPE_RERANK:
    {
        const int id_slot = json_value(task.data, "id_slot", -1);

        server_slot* slot;

        if (id_slot != -1) {
            slot = get_slot_by_id(id_slot);
        }
        else {
            slot = get_available_slot(task);
        }

        if (slot == nullptr) {
            // if no slot is available, we defer this task for processing later
            // PXA_DEFER_PUMP_v1: visible at default log level + slot state dump — a task
            // parking here with idle-looking slots is the starvation race we hunt.
            std::string st;
            for (const auto & s2 : slots) {
                st += std::to_string(s2.id) + ":s" + std::to_string((int) s2.state) + "c" + std::to_string((int) s2.command) + " ";
            }
            LOG_WARNING("no slot is available - task deferred", { {"id_task", task.id}, {"slot_states", st} });
            queue_tasks.defer(std::move(task));
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_WARNING("requested slot is unavailable - task deferred", { {"id_task", task.id}, {"id_slot_req", slot->id} });
            queue_tasks.defer(std::move(task));
            break;
        }

        // PXA_KV_UNIFIED_v1: shared-ring admission. A free slot is no longer proof of capacity once
        // the n_ctx/n_parallel split is gone, so gate on the ring itself before committing the task.
        // No-op (returns KV_ADMIT_OK immediately) when the lever is off.
        if (params_base.kv_unified && !task.embedding) {
            int32_t n_free = 0;
            const int32_t n_prompt = task.n_tokens();
            const kv_admission adm = kv_unified_admit(n_prompt, slot, n_free);
            if (adm == KV_ADMIT_DEFER) {
                // Park on the existing deferred queue; PXA_DEFER_PUMP_v1 re-dispatches on slot
                // release or its 500 ms heartbeat. The request waits, it does not fail, and no
                // live sequence is disturbed.
                LOG_WARNING("kv-unified: shared KV ring full - task deferred", {
                    {"id_task",         task.id},
                    {"id_slot",         slot->id},
                    {"n_prompt_tokens", n_prompt},
                    {"n_cells_free",    n_free},
                    {"n_ctx",           n_ctx},
                    {"n_cells_used",    llama_get_kv_cache_used_cells(ctx)},
                    {"n_cells_busy",    kv_unified_cells_busy(slot)},
                    {"n_cells_idle",    kv_unified_cells_idle_reclaimable(slot)},
                    });
                queue_tasks.defer(std::move(task));
                break;
            }
            // KV_ADMIT_REJECT is intentionally NOT handled here: the prompt is larger than the whole
            // ring, which is exactly the condition the pre-existing over-length guard in
            // update_slots() reports (slot.n_prompt_tokens >= slot.n_ctx, with slot.n_ctx == n_ctx
            // under the lever). Letting it fall through keeps one error path, not two.
        }

        if (task.data.contains("system_prompt")) {
            std::string sys_prompt = json_value(task.data, "system_prompt", std::string());
            system_prompt_set(sys_prompt);

            for (server_slot& slot : slots) {
                slot.n_past = 0;
                slot.n_past_se = 0;
            }
        }

        slot->reset();

        slot->id_task = task.id;
        slot->id_multi = task.id_multi;
        slot->infill = task.infill;
        slot->embedding = task.embedding;

        if (!launch_slot_with_task(*slot, task)) {
            LOG_ERROR("error while launching slot", task.data);
            break;
        }
    } break;
    case SERVER_TASK_TYPE_CANCEL:
    {
        // PXA_SLOT_RELEASE_UAF_FIX_v1: match the LIVE task pointer (reset to null by release()),
        // not the raw id_task int (which persists after release until the slot is reused). Under
        // request-cancellation churn (den/lab DEDUPE_v1 req.destroy(), bg 25s timeouts) a late/
        // duplicate CANCEL for an already-finished task otherwise re-matched the stale id_task and
        // re-released the slot. That double-release set command=RELEASE again on an IDLE (or freshly
        // relaunched) slot -> release_slots() re-ran the seq-rm/spec-state teardown while a new decode
        // batch referenced the same KV/recurrent state -> CUDA illegal memory access. Guarding on the
        // live unique_ptr makes a stale CANCEL a no-op (slot.task==nullptr after release) and only an
        // actually-processing slot can be cancelled -> release is now idempotent. Mirrors mainline
        // llama.cpp (slot.task && slot.task->id == task.id_target).
        for (auto& slot : slots) {
            if (slot.task && slot.task->id == task.id_target && slot.is_processing()) {
                slot.release();
                break;
            }
        }
    } break;
    case SERVER_TASK_TYPE_NEXT_RESPONSE:
    {
        // do nothing
    } break;
    case SERVER_TASK_TYPE_METRICS:
    {
        json slots_data = json::array();

        int n_idle_slots = 0;
        int n_processing_slots = 0;

        for (server_slot& slot : slots) {
            json slot_data = get_formated_generation(slot);
            slot_data["id"] = slot.id;
            slot_data["id_task"] = slot.id_task;
            slot_data["state"] = slot.state;
            slot_data["prompt"] = slot.prompt;
            slot_data["next_token"] = {
                {"has_next_token", slot.has_next_token},
                {"n_remain",       slot.n_remaining},
                {"n_decoded",      slot.n_decoded},
                {"stopped_eos",    slot.stopped_eos},
                {"stopped_word",   slot.stopped_word},
                {"stopped_limit",  slot.stopped_limit},
                {"stopping_word",  slot.stopping_word},
            };

            if (slot_data["state"] == SLOT_STATE_IDLE) {
                n_idle_slots++;
            }
            else {
                n_processing_slots++;
            }

            slots_data.push_back(slot_data);
        }
        LOG_INFO("slot data", {
            {"id_task",            task.id},
            {"n_idle_slots",       n_idle_slots},
            {"n_processing_slots", n_processing_slots}
            });

        LOG_VERBOSE("slot data", {
            {"id_task",            task.id},
            {"n_idle_slots",       n_idle_slots},
            {"n_processing_slots", n_processing_slots},
            {"slots",              slots_data}
            });

        server_task_result res;
        res.id = task.id;
        res.id_multi = task.id_multi;
        res.stop = true;
        res.error = false;
        res.data = {
            { "idle",                            n_idle_slots       },
            { "processing",                      n_processing_slots },
            { "deferred",                        queue_tasks.queue_tasks_deferred.size() },
            { "t_start",                         metrics.t_start},

            { "n_prompt_tokens_processed_total", metrics.n_prompt_tokens_processed_total},
            { "t_tokens_generation_total",       metrics.t_tokens_generation_total},
            { "n_tokens_predicted_total",        metrics.n_tokens_predicted_total},
            { "t_prompt_processing_total",       metrics.t_prompt_processing_total},

            { "n_prompt_tokens_processed",       metrics.n_prompt_tokens_processed},
            { "t_prompt_processing",             metrics.t_prompt_processing},
            { "n_tokens_predicted",              metrics.n_tokens_predicted},
            { "t_tokens_generation",             metrics.t_tokens_generation},

            { "kv_cache_tokens_count",           llama_get_kv_cache_token_count(ctx)},
            { "kv_cache_used_cells",             llama_get_kv_cache_used_cells(ctx)},

            { "slots",                           slots_data },
        };

        if (json_value(task.data, "reset_bucket", false)) {
            metrics.reset_bucket();
        }
        queue_results.send(res);
    } break;
    case SERVER_TASK_TYPE_SLOT_SAVE:
    {
        int id_slot = task.data.at("id_slot");
        server_slot* slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
            queue_tasks.defer(std::move(task));
            break;
        }

        const size_t token_count = slot->cache_tokens.size();
        const int64_t t_start = ggml_time_us();

        std::string filename = task.data.at("filename");
        std::string filepath = task.data.at("filepath");
        save_server_tokens_to_file(filepath+".tokens.json", slot->cache_tokens);
        size_t saved = save_checkpoints_to_file(filepath + ".checkpoints", slot->server_cached_prompt.checkpoints);

        const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), token_count);

        const int64_t t_end = ggml_time_us();
        const double t_save_ms = (t_end - t_start) / 1000.0;

        server_task_result result;
        result.id = task.id;
        result.stop = true;
        result.error = false;
        result.data = json{
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   token_count }, // tokens saved
            { "n_written", nwrite + saved },      // bytes written
            { "timings", {
                { "save_ms", t_save_ms }
            } }
        };
        queue_results.send(result);
    } break;
    case SERVER_TASK_TYPE_SLOT_RESTORE:
    {
        int id_slot = task.data.at("id_slot");
        server_slot* slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
            queue_tasks.defer(std::move(task));
            break;
        }
        const int64_t t_start = ggml_time_us();

        std::string filename = task.data.at("filename");
        std::string filepath = task.data.at("filepath");

        slot->cache_tokens.resize(slot->n_ctx);
        size_t token_count = 0;
        size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
        if (nread == 0) {
            slot->cache_tokens.resize(0);
            send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        load_server_tokens_from_file(filepath+".tokens.json", slot->cache_tokens);
        size_t loaded = load_checkpoints_from_file(filepath + ".checkpoints", slot->server_cached_prompt.checkpoints);

        const int64_t t_end = ggml_time_us();
        const double t_restore_ms = (t_end - t_start) / 1000.0;

        server_task_result result;
        result.id = task.id;
        result.stop = true;
        result.error = false;
        result.data = json{
            { "id_slot",    id_slot },
            { "filename",   filename },
            { "n_restored", token_count }, // tokens restored
            { "n_read",     nread },       // bytes read
            { "timings", {
                { "restore_ms", t_restore_ms }
            } }
        };
        queue_results.send(result);
    } break;
    case SERVER_TASK_TYPE_SLOT_ERASE:
    {
        int id_slot = task.data.at("id_slot");
        server_slot* slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
            queue_tasks.defer(std::move(task));
            break;
        }
        // Erase token cache
        const size_t n_erased = slot->cache_tokens.size();
        llama_synchronize(ctx); // PXA_WAVE4_FIX_v1: no full-seq wipe with kernels in flight
        llama_kv_cache_seq_rm(ctx, slot->id, -1, -1);
        slot->cache_tokens.keep_first(0);
        //slot->cache_tokens.clear();
        slot->server_cached_prompt.checkpoints.clear();
        slot->server_cached_prompt.data.clear();
        server_task_result result;
        result.id = task.id;
        result.stop = true;
        result.error = false;
        result.data = json{
            { "id_slot",  id_slot },
            { "n_erased", n_erased }
        };
        queue_results.send(result);
    } break;
    case SERVER_TASK_TYPE_SET_LORA:
    {
        llama_lora_adapters_apply(ctx, lora_adapters);
        server_task_result result;
        result.id = task.id;
        result.stop = true;
        result.error = false;
        result.data = json{ { "success", true } };
        queue_results.send(result);
    } break;
    case SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR:
    {
        // Load control vector from file
        std::string path = task.data.at("path");
        float scale = task.data.value("scale", 1.0f);
        int32_t layer_start = task.data.value("layer_start", 1);
        int32_t layer_end = task.data.value("layer_end", llama_n_layer(model));

        // Check if already loaded
        int cv_id = -1;
        for (size_t i = 0; i < control_vectors.size(); i++) {
            if (control_vectors[i].path == path) {
                control_vectors[i].scale = scale;
                control_vectors[i].layer_start = layer_start;
                control_vectors[i].layer_end = layer_end;
                cv_id = i;
                break;
            }
        }

        if (cv_id == -1) {
            control_vector_container new_cv;
            new_cv.path = path;
            new_cv.scale = scale;
            new_cv.layer_start = layer_start;
            new_cv.layer_end = layer_end;
            new_cv.applied = false;

            // Load the control vector data
            llama_control_vector_load_info load_info;
            load_info.fname = path;
            load_info.strength = 1.0f;  // Don't pre-scale here, we'll scale when applying

            std::vector<llama_control_vector_load_info> load_infos = { load_info };
            new_cv.data = llama_control_vector_load(load_infos);

            if (new_cv.data.n_embd == -1) {
                server_task_result result;
                result.id = task.id;
                result.error = true;
                result.data = json{{ "success", false }, { "error", "Failed to load control vector from " + path }};
                queue_results.send(result);
                break;
            }

            // Validate dimension to prevent heap corruption
            if (new_cv.data.n_embd != llama_model_n_embd(model)) {
                server_task_result result;
                result.id = task.id;
                result.error = true;
                result.data = json{{ "success", false },
                                   { "error", "Vector dimension mismatch" }};
                queue_results.send(result);
                break;
            }

            control_vectors.push_back(new_cv);

            cv_id = control_vectors.size() - 1;
        }

        // Auto-apply control vectors after loading
        if (!apply_control_vectors_internal()) {
            server_task_result result;
            result.id = task.id;
            result.error = true;
            result.data = json{{ "success", false }, { "error", "Failed to apply control vectors" }};
            queue_results.send(result);
            break;
        }

        server_task_result result;
        result.id = task.id;
        result.error = false;
        result.data = json{{ "success", true }, { "id", cv_id }};
        queue_results.send(result);
    } break;
    case SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR:
    {
        // Validate that "id" field exists and is a number
        if (!task.data.contains("id") || task.data["id"].is_null() || !task.data["id"].is_number()) {
            server_task_result result;
            result.id = task.id;
            result.error = true;
            result.data = json{{ "success", false }, { "error", "Missing or invalid 'id' field" }};
            queue_results.send(result);
            break;
        }

        int id = task.data.at("id");

        if (id < 0 || id >= (int)control_vectors.size()) {
            server_task_result result;
            result.id = task.id;
            result.error = true;
            result.data = json{{ "success", false }, { "error", "Invalid control vector ID" }};
            queue_results.send(result);
            break;
        }

        // Remove the control vector from the list
        control_vectors.erase(control_vectors.begin() + id);

        // Reapply remaining control vectors
        if (!apply_control_vectors_internal()) {
            server_task_result result;
            result.id = task.id;
            result.error = true;
            result.data = json{{ "success", false }, { "error", "Failed to apply control vectors" }};
            queue_results.send(result);
            break;
        }

        server_task_result result;
        result.id = task.id;
        result.error = false;
        result.data = json{{ "success", true }};
        queue_results.send(result);
    } break;
    case SERVER_TASK_TYPE_SET_CONTROL_VECTOR:
    {
        if (!apply_control_vectors_internal()) {
            server_task_result result;
            result.id = task.id;
            result.error = true;
            result.data = json{{ "success", false }, { "error", "Failed to apply control vectors" }};
            queue_results.send(result);
            break;
        }

        server_task_result result;
        result.id = task.id;
        result.error = false;
        result.data = json{{ "success", true }};
        queue_results.send(result);
    } break;
    }
}

bool server_context::apply_control_vectors_internal() {
    llama_control_vector_data combined_cv = { -1, {} };

    // Check if we have anything to apply
    bool any_active = false;
    for (const auto& cv : control_vectors) {
        if (cv.scale != 0.0f) {
            any_active = true;
            break;
        }
    }

    if (!any_active) {
        // Clear control vectors if nothing is active
        llama_control_vector_apply(ctx, nullptr, 0, 0, 0, 0);
        return true;
    }

    // Aggregate control vectors with scaling
    for (auto& cv : control_vectors) {
        if (cv.scale == 0.0f) {
            cv.applied = false;
            continue;
        }

        if (combined_cv.n_embd == -1) {
            combined_cv.n_embd = cv.data.n_embd;
            combined_cv.data.resize(cv.data.data.size(), 0.0f);
        }

        for (size_t i = 0; i < cv.data.data.size(); i++) {
            combined_cv.data[i] += cv.data.data[i] * cv.scale;
        }
        cv.applied = true;
    }

    // Apply combined control vector
    if (combined_cv.n_embd != -1 && !combined_cv.data.empty()) {
        int32_t min_layer_start = INT32_MAX;
        int32_t max_layer_end = 0;

        for (const auto& cv : control_vectors) {
            if (cv.scale != 0.0f) {
                min_layer_start = std::min(min_layer_start, cv.layer_start);
                max_layer_end = std::max(max_layer_end, cv.layer_end);
            }
        }

        int err = llama_control_vector_apply(ctx,
                                            combined_cv.data.data(),
                                            combined_cv.data.size(),
                                            combined_cv.n_embd,
                                            min_layer_start,
                                            max_layer_end);
        return (err == 0);
    }

    return true;
}

void server_context::on_finish_multitask(const server_task_multi& multitask) {
    // all subtasks done == multitask is done
    server_task_result result;
    result.id = multitask.id;
    result.stop = true;
    result.error = false;

    // collect json results into one json result
    std::vector<json> result_jsons;
    for (const auto& subres : multitask.results) {
        result_jsons.push_back(subres.data);
        result.error = result.error && subres.error;
    }
    result.data = json{
        { "results", result_jsons }
    };

    queue_results.send(result);
}

void server_context::print_tokens(const server_tokens& prompt, const server_tokens& cache, size_t start1, size_t start2, size_t length) {
    if (cache.size() > start2) {
        LLAMA_LOG_INFO("cache : %s\n", cache.detokenize(ctx, true, start2, length).c_str());
    }
    if (prompt.size() > start1) {
        LLAMA_LOG_INFO("prompt: %s\n", prompt.detokenize(ctx, true, start1, length).c_str());
    }

}

void server_context::discard_n_kv_and_cache_tokens(llama_context* ctx, server_slot& slot, int32_t n_keep, int32_t n_discard) {
    auto kv_keep = slot.cache_tokens.pos_next(n_keep);
    auto kv_discard = slot.cache_tokens.pos_next(n_keep + n_discard) - kv_keep;
    auto kv_past = slot.cache_tokens.pos_next(slot.n_past);
    int32_t pos_min = llama_kv_cache_seq_pos_min(slot.ctx, slot.id);
    const auto pos_max = llama_kv_cache_seq_pos_max(slot.ctx, slot.id);
    llama_synchronize(ctx); // PXA_WAVE4_FIX_v1: context-shift trim is the same teardown class
    // PXA_HYBRID_CTX_SHIFT_v1: kv_keep/kv_discard are already POSITIONS (pos_next of token
    // indices above); re-wrapping them in pos_next() treated a position as a token index —
    // identity for text-only, wrong under mtmd, and inconsistent with the seq_add line below.
    LLAMA_LOG_INFO("PXA shift trim: seq=%d n_keep=%d n_discard=%d kv_keep=%d kv_discard=%d kv_past=%d pos_min=%d pos_max=%d\n",
        slot.id, n_keep, n_discard, (int)kv_keep, (int)kv_discard, (int)kv_past, (int)pos_min, (int)pos_max);
    llama_kv_cache_seq_rm(ctx, slot.id, kv_keep, kv_keep + kv_discard);
    llama_kv_cache_seq_add(ctx, slot.id, kv_keep + kv_discard, kv_past, -kv_discard);
    if (slot.has_mtp && slot.spec) {
        common_speculative_context_shift(slot.spec, slot.id, kv_keep, kv_discard, kv_past);
    }
    if (slot.params.cache_prompt) {
        slot.cache_tokens.discard_n_tokens(n_keep, n_discard);
    }
    // PXA_HYBRID_CTX_SHIFT_v1: the shift moves every position after kv_keep down by kv_discard.
    // Prompt checkpoints capture recurrent state keyed by pos ranges in the OLD position space;
    // any checkpoint reaching past the kept prefix is no longer mappable — drop it. A later
    // rewind then falls back to full re-processing, which is correct, just slower.
    {
        auto & ckpts = slot.server_cached_prompt.checkpoints;
        for (auto it = ckpts.begin(); it != ckpts.end();) {
            it = (it->pos_max > kv_keep) ? ckpts.erase(it) : std::next(it);
        }
    }
}


inline static bool tokens_support_context_shift(const server_tokens & tokens, int32_t n_keep,
    int32_t n_discard) {
    bool can_shift = !tokens.has_mtmd;
    if (tokens.has_mtmd) {
        can_shift = true;
        if (n_keep > 0 && n_keep<= tokens.n_tokens()) {
            can_shift = tokens[n_keep - 1] != LLAMA_TOKEN_NULL;
        }
        if (n_discard + n_keep > 0 && n_discard + n_keep <= tokens.n_tokens()) {
            can_shift = can_shift && tokens[n_discard + n_keep - 1] != LLAMA_TOKEN_NULL;
        }
    }
    return can_shift;
}

inline static void adjust_n_to_support_context_shift(const server_tokens & tokens, int32_t & n_keep,
    int32_t & n_discard) {
    if (!tokens.has_mtmd) {
        return;
    }
    if (n_keep > 0 && n_keep <= tokens.n_tokens()) {
        while (tokens[n_keep - 1] == LLAMA_TOKEN_NULL) {
            n_keep--;
            if (n_keep<1 || n_keep>tokens.size()) {
                break;
            }
        }
    }
    if (n_discard + n_keep > 0 && n_discard + n_keep <= tokens.n_tokens()) {
        while (tokens[n_discard + n_keep - 1] == LLAMA_TOKEN_NULL) {
            n_discard++;
            if (n_discard + n_keep<1 || n_discard + n_keep>tokens.size()) {
                break;
            }
        }
    }

}


// convert keep first few and discard next tokens in a to b
void server_context::context_shift_find_n_tokens(llama_context* ctx, const server_tokens& a, const server_tokens& b, int32_t n_keep,
    int32_t n_discard, int32_t& n_kept, int32_t& n_discarded, bool exact) {

    common_prefix ctx_keep_prefix = a.get_common_prefix_first_n(ctx, b, n_keep, exact);
    common_prefix ctx_total_discard_prefix = a.get_common_prefix_first_n(ctx, b, n_discard + n_keep, exact);
    // only if there is enough common token
    int32_t discard_offset = ctx_total_discard_prefix.first - (n_discard + n_keep);
    int32_t keep_offset = ctx_keep_prefix.first - n_keep;
    n_kept = ctx_keep_prefix.second - keep_offset;
    n_discarded = ctx_total_discard_prefix.second - ctx_keep_prefix.second - discard_offset;
    if (n_kept < 0) {
        n_kept = n_keep;
    }
    if (n_discarded < 0) {
        n_discarded = n_discard;
    }
}

void server_context::context_shift_prompt(llama_context* ctx, server_slot& slot, bool exact) {
    int n_keep = std::max(0, slot.params.n_keep + add_bos_token);
    const int n_left = slot.n_ctx - n_keep;
    int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);
    adjust_n_to_support_context_shift(slot.prompt_tokens, n_keep, n_discard);
    if (n_discard<=0 || !tokens_support_context_shift(slot.prompt_tokens, n_keep, n_discard)) {
        return;
    }
    int n_discard_prompt = 0;
    // we still need to truncate input since we have not discarded enough tokens
    while (slot.n_prompt_tokens - slot.n_discarded_prompt >= slot.n_ctx) {
        slot.n_discarded_prompt = slot.n_discarded_prompt + n_discard;
        n_discard_prompt = n_discard_prompt + n_discard;
    }

    // Handle mistokenization between prompt and cache during context shift
    //
    int32_t n_discard_cache = n_discard_prompt;
    int32_t n_kept = n_keep;
    slot.prompt_tokens.discard_n_tokens(n_keep, slot.n_discarded_prompt - n_discard_prompt);
    if (n_discard_prompt > 0) {
        context_shift_find_n_tokens(ctx, slot.prompt_tokens, slot.cache_tokens, n_keep,
            n_discard, n_kept, n_discard_cache, exact);
    }

    int n_discard_cache_max = std::max((int32_t)slot.cache_tokens.size() - n_kept, 0);
    n_discard_cache = std::min(n_discard_cache, n_discard_cache_max);
    // discard matching tokens from cache and kv cache to avoid reprocessing the prompt
    if (n_discard_cache > 0) {
        discard_n_kv_and_cache_tokens(ctx, slot, n_kept, n_discard_cache);
    }
    // discard extra tokens from prompts
    slot.n_kept_prompt = n_keep;
    slot.prompt_tokens.discard_n_tokens(n_keep, n_discard_prompt);
    slot.n_prompt_tokens = slot.prompt_tokens.size();
}

void server_context::release_slots()
{
    for (auto& slot : slots) {
        if (slot.command == SLOT_COMMAND_RELEASE) {
            slot.state = SLOT_STATE_IDLE;
            slot.command = SLOT_COMMAND_NONE;
            slot.t_last_used = ggml_time_us();

            LOG_INFO("slot released", {
                {"id_slot",         slot.id},
                {"id_task",         slot.id_task},
                {"n_ctx",           n_ctx},
                {"n_past",          slot.n_past},
                {"n_system_tokens", system_tokens.size()},
                {"n_cache_tokens",  slot.cache_tokens.size()},
                {"truncated",       slot.truncated}
                });

            queue_tasks.notify_slot_changed();
        }
    }
}

bool server_context::slots_idle(){
        bool all_idle = true;
        for (auto& slot : slots) {
            if (slot.state != SLOT_STATE_IDLE || slot.command != SLOT_COMMAND_NONE) {
                all_idle = false;
                break;
            }
        }

        if (all_idle) {
            LOG_INFO("all slots are idle", {});
            if (system_prompt.empty() && clean_kv_cache) {
                kv_cache_clear();
            }
            all_idle = true;
        }
        return all_idle;
}

// ================= PXA_KV_UNIFIED_v1: shared-ring admission control ==========================
//
// WHY THIS EXISTS. With the lever OFF the server hands each slot a private n_ctx/n_parallel window,
// so a slot can never consume cells another slot needs -- the split IS the admission policy. With
// the lever ON every slot may address the whole ring, and that policy disappears. Without a
// replacement, two long sequences would race for the same cells and llama_kv_cache_find_slot()
// (src/llama.cpp:1509) would eventually return false mid-request: llama_decode fails, the slot dies
// with a 500, and (worse) nothing stops one sequence from starving another. That is strictly worse
// than the fixed split, which is why the flag cannot just delete the division.
//
// WHAT WE ACCOUNT. ATTENTION KV CELLS ONLY. One cell == one token position in the single seq-tagged
// ring of exactly cparams.n_ctx cells (src/llama.cpp:8459, tagging at :1605-1611). The per-sequence
// recurrent GDN/delta-net state (kv_self.s_l, ~112.2 MiB per sequence on qwen4exp) is NOT in this
// ring, is indexed by absolute seq_id, does NOT unify, and its cost still scales with -np. It is
// intentionally absent from every sum below; unifying the attention ring neither helps nor hurts it.
// We also never share a cell between slots: src/llama-delta-net.cpp:43 asserts one seq_id per token.
//
// THE POLICY. Ground truth for occupancy is llama_get_kv_cache_used_cells(), an O(1) read of
// kv_self.used. A request is admitted only if the ring can hold its prompt plus a small forward
// -progress reserve. If it cannot, we prefer, in order:
//   1. reclaim -- drop the prompt CACHE of idle slots (oldest-used first). An idle slot's cells are
//      a disposable optimisation, not a live sequence; dropping them costs a future re-prefill and
//      disturbs nobody. This is the same teardown the SLOT_ERASE task already performs.
//   2. defer  -- park the task on the EXISTING deferred queue (PXA_DEFER_PUMP_v1), which re-dispatches
//      when a slot releases or on its 500 ms heartbeat. The request waits; it does not fail.
//   3. reject -- only when the prompt could not fit even on a completely empty ring. This falls
//      through to the pre-existing context_length_exceeded / "exceeds the available context size"
//      path rather than inventing a second error.
// At no point do we evict cells belonging to a slot that is mid-request.

int32_t server_context::kv_unified_cells_busy(const server_slot * except) const {
    // Cells that in-flight slots have already been ADMITTED for but have not yet materialised in the
    // ring. This is the part llama_get_kv_cache_used_cells() cannot see, and leaving it out is a real
    // bug rather than a conservatism: several requests arriving in the same tick each look at a ring
    // that is still empty, each is admitted, and then they collectively overrun it during prefill --
    // observed as three of four concurrent 2993-token prompts dying on "Input prompt is too big
    // compared to KV size" at n_ctx 8192. Charging the outstanding promise fixes that admission race.
    //
    // Only the REMAINDER is charged: whatever a slot has already decoded is counted by used_cells, so
    // adding its full prompt here as well would double-count it.
    //
    // A slot that has been launched but has not yet had an update_slots() tick still has
    // n_prompt_tokens == 0 (it is set in batch_pending_prompt), while prompt_tokens is already
    // populated by launch_slot_with_task. Take the larger of the two so a slot is charged from the
    // moment it is committed, not from the moment it is first ticked.
    int32_t n = 0;
    for (const auto & s : slots) {
        if (&s == except || !s.is_processing()) {
            continue;
        }
        const int32_t want = std::max(s.n_prompt_tokens, (int32_t) s.prompt_tokens.size()) + kv_unified_reserve;
        n += std::max(0, want - s.n_past);
    }
    return n;
}

int32_t server_context::kv_unified_cells_idle_reclaimable(const server_slot * except) const {
    int32_t n = 0;
    for (const auto & s : slots) {
        if (&s == except || !s.available()) {
            continue;
        }
        n += (int32_t) s.cache_tokens.size();
    }
    return n;
}

int32_t server_context::kv_unified_reclaim_idle(int32_t need, const server_slot * except) {
    if (need <= 0) {
        return 0;
    }

    // Oldest-used idle slot first: that is the cache least likely to be hit again, and it matches
    // the victim order get_available_slot() already uses when it picks a slot to overwrite.
    std::vector<server_slot *> victims;
    for (auto & s : slots) {
        if (&s == except || !s.available() || s.cache_tokens.size() == 0) {
            continue;
        }
        victims.push_back(&s);
    }
    std::sort(victims.begin(), victims.end(), [](const server_slot * a, const server_slot * b) {
        return a->t_last_used < b->t_last_used;
    });

    int32_t reclaimed = 0;
    for (server_slot * s : victims) {
        if (reclaimed >= need) {
            break;
        }
        const int32_t n_cells = (int32_t) s->cache_tokens.size();

        // PXA_WAVE4_FIX_v1 idiom: drain before any full-sequence teardown, on BOTH contexts.
        auto * ctx_companion = s->spec ? common_speculative_get_companion_ctx(s->spec) : nullptr;
        llama_synchronize(ctx);
        if (ctx_companion != nullptr) {
            llama_synchronize(ctx_companion);
        }
        llama_kv_cache_seq_rm(ctx, s->id, -1, -1);
        if (ctx_companion != nullptr) {
            llama_kv_cache_seq_rm(ctx_companion, s->id, -1, -1);
        }
        s->cache_tokens.keep_first(0);
        s->server_cached_prompt.checkpoints.clear();
        s->server_cached_prompt.data.clear();
        s->n_past = 0;
        s->n_past_prompt = 0;
        s->n_prompt_tokens_cache = 0;

        reclaimed += n_cells;

        LOG_INFO("kv-unified: reclaimed idle slot prompt cache", {
            {"id_slot",   s->id},
            {"n_cells",   n_cells},
            {"reclaimed", reclaimed},
            {"need",      need},
            });
    }

    return reclaimed;
}

server_context::kv_admission server_context::kv_unified_admit(int32_t n_prompt_tokens,
                                                              const server_slot * except,
                                                              int32_t & n_free_out) {
    n_free_out = n_ctx;

    if (!params_base.kv_unified) {
        // Lever OFF: the static n_ctx/n_parallel split is the admission policy, exactly as before.
        return KV_ADMIT_OK;
    }

    const int32_t need = n_prompt_tokens + kv_unified_reserve;

    // Could this ever fit, on a completely empty ring? If not, do not park it forever -- let it fall
    // through to the pre-existing over-length error path.
    if (need > n_ctx) {
        n_free_out = n_ctx;
        return KV_ADMIT_REJECT;
    }

    // Ground truth occupancy of the shared ring (O(1): returns kv_self.used) PLUS the cells already
    // promised to slots that are in flight but have not decoded them yet. Both terms are required:
    // used_cells alone loses the admission race described in kv_unified_cells_busy().
    const int32_t used    = llama_get_kv_cache_used_cells(ctx);
    const int32_t pending = kv_unified_cells_busy(except);

    // Cells the requesting slot itself already holds are not an obstacle to its own request: they are
    // either reused as a cache_prompt prefix or trimmed by the reuse path before its prompt lands.
    const int32_t own = except != nullptr ? (int32_t) except->cache_tokens.size() : 0;

    int32_t n_free = n_ctx - used - pending + own;
    if (need <= n_free) {
        n_free_out = n_free;
        return KV_ADMIT_OK;
    }

    // Not enough right now. Try reclaiming disposable idle-slot caches before making anyone wait.
    const int32_t reclaimed = kv_unified_reclaim_idle(need - n_free, except);
    if (reclaimed > 0) {
        n_free = n_ctx - llama_get_kv_cache_used_cells(ctx) - kv_unified_cells_busy(except) + own;
        if (need <= n_free) {
            n_free_out = n_free;
            return KV_ADMIT_OK;
        }
    }

    // Still short: the missing cells are pinned by slots that are mid-request. Wait for them rather
    // than evicting a live sequence.
    n_free_out = n_free;
    return KV_ADMIT_DEFER;
}
// ============================================================================================

void server_context::context_shift() {
    // PXA_KV_UNIFIED_v1: runtime half of admission control.
    //
    // Admission bounds what ENTERS the ring; it cannot bound generation, because n_predict is
    // routinely unbounded. With the lever off, slot.n_ctx == n_ctx/n_parallel and the per-slot test
    // below is itself the backstop -- a slot simply cannot outgrow its private window. With the
    // lever on, slot.n_ctx == n_ctx, so that test only fires when ONE slot has eaten the entire
    // ring, and several co-resident slots could collectively exhaust it first. We therefore add a
    // SHARED-ring trigger and route it into this same, already-correct handler: context shift when
    // enabled (which discards the slot's OWN oldest tokens -- never another sequence's), or the
    // pre-existing clean context_length_exceeded error when --no-context-shift is set.
    //
    // Only the deepest slot is asked to yield. It frees the most cells, it is the slot actually
    // responsible for the pressure, and making one slot shift is far less disruptive than making
    // every slot shift at once. Cells belonging to any other live sequence are never touched.
    // Accounting is ATTENTION KV CELLS ONLY; the per-sequence GDN state is not in this ring.
    const server_slot * pxa_kvu_deepest = nullptr;
    if (params_base.kv_unified) {
        const int32_t margin = std::max(kv_unified_reserve, 4 * std::max(1, params_base.n_parallel));
        if (llama_get_kv_cache_used_cells(ctx) >= n_ctx - margin) {
            // Idle-slot prompt caches are disposable and may be holding the cells that live slots
            // need. Reclaim those FIRST -- no running request is affected by dropping them.
            kv_unified_reclaim_idle(2 * margin, nullptr);
        }
        if (llama_get_kv_cache_used_cells(ctx) >= n_ctx - margin) {
            int32_t best = -1;
            for (const auto & s : slots) {
                if (s.ga_n == 1 && s.state == SLOT_STATE_PROCESSING && s.n_past > best) {
                    best = s.n_past;
                    pxa_kvu_deepest = &s;
                }
            }
            if (pxa_kvu_deepest != nullptr) {
                LOG_WARNING("kv-unified: shared KV ring near capacity - yielding deepest slot", {
                    {"id_slot",      pxa_kvu_deepest->id},
                    {"n_past",       pxa_kvu_deepest->n_past},
                    {"n_cells_used", llama_get_kv_cache_used_cells(ctx)},
                    {"n_ctx",        n_ctx},
                    {"margin",       margin},
                    {"ctx_shift",    params_base.ctx_shift},
                    });
            }
        }
    }

    for (server_slot& slot : slots) {
        if (slot.ga_n == 1) {
            const bool pxa_kvu_ring_pressure = (pxa_kvu_deepest == &slot);
            if (slot.is_processing() && (pxa_kvu_ring_pressure ||
                                         (int)system_tokens.size() + slot.n_past >= slot.n_ctx - 1)) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    slot.print_timings();
                    slot.release();
                    send_error(slot, "context_length_exceeded: the request exceeds the available context size; context shift is disabled", ERROR_TYPE_SERVER);
                    continue;
                }
                // Shift context
                int n_keep = slot.params.n_keep < 0 ? slot.prompt_tokens.size() : slot.params.n_keep;
                if (add_bos_token) {
                    n_keep += 1;
                }
                // PXA_HYBRID_CTX_SHIFT_v2: never discard position 0. BOS-less vocabs (qwen)
                // land here with n_keep=0, and dropping the first token destroys the attention
                // sink -> repetition collapse in long chats (same failure class documented in
                // PXA_MTP_HYBRID_ROLLBACK_FIX).
                if (n_keep < 1) {
                    n_keep = 1;
                }
                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int32_t n_left = (int)system_tokens.size() + slot.n_past - n_keep;
                int32_t n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);
                int32_t n_kept;
                int32_t n_discard_cache;
                adjust_n_to_support_context_shift(slot.cache_tokens, n_keep, n_discard);
                if (n_discard > 0 && tokens_support_context_shift(slot.cache_tokens, n_keep, n_discard)) {
                    context_shift_find_n_tokens(ctx, slot.prompt_tokens, slot.cache_tokens, n_keep,
                        n_discard, n_kept, n_discard_cache);
                    LOG_INFO("slot context shift", {
                                         {"id_slot",         slot.id},
                                         {"id_task",         slot.id_task},
                                         {"n_keep",          n_keep},
                                         {"n_left",          n_left},
                                         {"n_discard",       n_discard},
                                         {"n_ctx",           n_ctx},
                                         {"n_past",          slot.n_past},
                                         {"n_system_tokens", system_tokens.size()},
                                         {"n_cache_tokens",  slot.cache_tokens.size()}
                        });
                    slot.n_discarded_prompt = slot.n_discarded_prompt + n_discard;
                    slot.n_kept_prompt = n_keep;
                    discard_n_kv_and_cache_tokens(ctx, slot, n_kept, n_discard_cache);
                    slot.n_past -= n_discard_cache;
                    slot.truncated = true;
                }

            }
        }
    }
}

void server_context::add_sampled_tokens() {
    // PXA_LLAMA_MTP_NP4_FIX: at np>1 with MTP, slots that co-decode in this one batch can produce
    // DIFFERENT draft lengths (the MTP drafter stops early when its confidence drops below p_min).
    // That makes the shared verify u-batch RAGGED (unequal tokens-per-sequence), which forces the
    // qwen3next delta-net onto the "non-uniform multi-seq -> split per-sequence" fallback. That
    // fallback corrupts the recurrent-state handoff for the 2nd+ sub-sequence (NaN logits / repetition
    // loops). FIX: make the generation portion of the batch UNIFORM by capping every co-decoding
    // MTP slot to a COMMON draft length this step, so the batch always takes the proven-clean batched
    // (uniform n_seqs x n_seq_tokens) delta-net path. Truncating a draft is lossless for correctness
    // (the slot simply verifies fewer speculative tokens this step and re-drafts next step); it costs
    // only a little speculative throughput, which is the accepted trade for concurrency correctness.

    // ---- pass 1: generate drafts for every active slot, but DO NOT add to the batch yet ----
    struct pxa_pending_slot {
        server_slot * slot      = nullptr;
        bool          do_spec   = false;  // n_draft_max_pre > 0 (this slot drafted)
        llama_tokens  draft;              // the generated (and n_draft_max-capped) draft
        int           min_usable = 0;     // this slot's min usable draft length
    };
    std::vector<pxa_pending_slot> pending;
    pending.reserve(slots.size());

    // PXA_NP_SPEC_GATE_v1 (alias PXA_GRAMMAR_CONCURRENCY_v1, PXA_KERNEL_FIX_20260611): when 2+
    // slots co-decode on a recurrent/hybrid model, speculation is a pure tax: the uniform-batch
    // equalizer below chops every draft to the common (usually 0-1) length, so accepted drafts
    // average ~1.05 tokens/step (~= no speedup), while EVERY step still pays K sequential MTP
    // draft generations (GPU round-trips), K per-step checkpoint saves, per-slot D2H hidden-row
    // copies, per-slot rollback restores/commits and a ctx-global discard. Measured on the
    // 35B-A3B Q8 3xP100 prod profile: 4 concurrent plain streams 9.5 t/s/stream WITH speculation
    // vs 12.8 t/s/stream WITHOUT (~25% tax). This also closes the observed "grammar/json_schema
    // requests stall under concurrency" symptom (they are long thinking+constrained generations,
    // so the universal np>1 slowdown made them look wedged). FIX: skip drafting entirely for this
    // tick when 2+ slots are active; a lone slot keeps full MTP depth (solo 49.5 t/s unchanged).
    // The MTP per-seq hidden state self-heals at the next solo draft via
    // common_speculative_ensure_sequence_hidden() (same resume path PXA_MTP_ADAPTIVE tier-0 uses).
    // Kill-switch: PXA_NP_SPEC_GATE=0 restores the old always-draft behavior.
    // PXA_WAVE4_FIX_v1 companion: default OFF -- session-3 gauntlet found isolation K=4 DIRTY
    // with the gate ON (feature shelved until root-caused). Opt-in via PXA_NP_SPEC_GATE=1.
    static const bool pxa_np_spec_gate_on =
        (getenv("PXA_NP_SPEC_GATE") && atoi(getenv("PXA_NP_SPEC_GATE")) == 1);
    int pxa_n_active_slots = 0;
    for (const auto & pxa_s : slots) {
        if (pxa_s.state != SLOT_STATE_IDLE) {
            ++pxa_n_active_slots;
        }
    }
    const bool pxa_skip_spec_this_tick =
        pxa_np_spec_gate_on && pxa_n_active_slots >= 2 && llama_model_has_recurrent(model);

    for (auto& slot : slots) {
        slot.released = false;
        if (slot.state == SLOT_STATE_IDLE) {
            continue;
        }

        pxa_pending_slot ps;
        ps.slot = &slot;

        // generate draft tokens in speculative decoding mode
        // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
        //       perform the speculative drafting for all sequences at the same time in a single batch
        const int n_draft_max_pre = slot.get_n_draft_max();

        // PXA_MTP_ADAPTIVE: acceptance-EMA-gated draft length (np=1 + MTP only; default ON, opt out
        // with PXA_MTP_ADAPTIVE=0). EXPERIMENTAL — defaults OFF pending the skip-tier warmup fix (stuck-skip + per-step capture cost measured 19 tok/s); enable with PXA_MTP_ADAPTIVE=1. Every MTP-drafted token costs one MTP-head forward (incl. an
        // lm_head GEMV) BEFORE verification, so when acceptance collapses (~50% at temp 0.85 chat
        // vs ~95% greedy) speculation is a net LOSS vs plain decode on the memory-bound P100s.
        // Tier the draft length by a per-slot acceptance EMA:
        //   ema >= 0.55        -> full n_draft_max (behaviour unchanged)
        //   0.35 <= ema < 0.55 -> cap the GENERATED draft to 1 token (normally the drafter's cached
        //                         free token, so the draft costs ~no extra MTP forwards)
        //   ema < 0.35         -> skip speculation (plain decode path), but every 24th skipped step
        //                         run one full-length PROBE draft so the controller can recover when
        //                         content turns predictable again (code/tool-call bursts).
        // np>1 is untouched: the gate below leaves the PXA_LLAMA_MTP_NP4_FIX uniform-batch logic
        // and all multi-slot behaviour exactly as-is.
        int pxa_draft_cap = -1; // -1 = uncapped, 1 = cap generated draft to 1, 0 = skip speculation
        {
            static const int pxa_mtp_adaptive = getenv("PXA_MTP_ADAPTIVE") ? atoi(getenv("PXA_MTP_ADAPTIVE")) : 0;
            if (pxa_mtp_adaptive != 0 && params_base.n_parallel == 1 && slot.has_mtp &&
                !slot.params.speculative.autotune && n_draft_max_pre > 0) {
                constexpr float PXA_ADAPT_EMA_FULL = 0.55f; // >= this: full-length drafts
                constexpr float PXA_ADAPT_EMA_CAP1 = 0.35f; // >= this: cap drafts to 1 token
                constexpr int   PXA_ADAPT_PROBE_PERIOD = 24; // skip-tier steps between probe drafts
                bool pxa_probe = false;
                int  pxa_tier;
                if (slot.spec_accept_ema >= PXA_ADAPT_EMA_FULL) {
                    pxa_tier = 2;
                } else if (slot.spec_accept_ema >= PXA_ADAPT_EMA_CAP1) {
                    pxa_tier = 1;
                } else {
                    pxa_tier = 0;
                    if (--slot.spec_probe_countdown <= 0) {
                        slot.spec_probe_countdown = PXA_ADAPT_PROBE_PERIOD;
                        pxa_probe = true;
                    }
                }
                if (pxa_tier > 0) {
                    // re-arm while drafting normally so a fresh skip stretch probes only after a full period
                    slot.spec_probe_countdown = PXA_ADAPT_PROBE_PERIOD;
                }
                if (pxa_tier != slot.spec_adaptive_tier) {
                    SLT_DBG(slot, "PXA_MTP_ADAPTIVE: tier %d -> %d (accept ema = %.3f)\n",
                            slot.spec_adaptive_tier, pxa_tier, slot.spec_accept_ema);
                    slot.spec_adaptive_tier = pxa_tier;
                }
                pxa_draft_cap = (pxa_tier == 2 || pxa_probe) ? -1 : pxa_tier;
            }
        }

        if (n_draft_max_pre > 0 && pxa_draft_cap != 0 && !pxa_skip_spec_this_tick) { // PXA_NP_SPEC_GATE_v1
            if (mctx && !slot.has_mtp) {
                // we should never reach this, as speculative is automatically disabled if mmproj is loaded
                GGML_ABORT("not supported by multimodal");
            }

            static const llama_tokens empty_prompt;
            const llama_tokens & cached_text_tokens = slot.has_mtp && !slot.params.speculative.has_composite_stage_chain()
                ? empty_prompt
                : slot.cache_tokens.get_text_tokens();

            auto & params_spec = slot.params.speculative;
            const llama_pos draft_base_pos = slot.has_mtp ? slot.cache_tokens.pos_next() : -1;

            if (slot.has_mtp) {
                if (!common_speculative_ensure_sequence_hidden(slot.spec, ctx, slot.id, draft_base_pos - 1)) {
                    LOG_ERROR("MTP hidden state is empty during speculation", {});
                }
            }

            // PXA_MTP_ADAPTIVE: apply the tier cap to the draft GENERATION params (not just the
            // verify length) so the per-drafted-token MTP forwards are actually saved. Stage n_max
            // overrides win over params.n_max in common_speculative_get_runtime_params(), so cap
            // both; restored right after the call (these are the request's params).
            const int32_t pxa_saved_n_max = params_spec.n_max;
            std::vector<int32_t> pxa_saved_stage_n_max;
            if (pxa_draft_cap > 0) {
                pxa_saved_stage_n_max.reserve(params_spec.stages.size());
                for (auto & pxa_st : params_spec.stages) {
                    pxa_saved_stage_n_max.push_back(pxa_st.n_max);
                    if (pxa_st.has_n_max_override() && pxa_st.n_max > pxa_draft_cap) {
                        pxa_st.n_max = pxa_draft_cap;
                    }
                }
                if (params_spec.n_max > pxa_draft_cap) {
                    params_spec.n_max = pxa_draft_cap;
                }
            }

            const int64_t pxa_sptr_t0 = pxa_sptr.on ? ggml_time_us() : 0; // PXA_SPEC_TRACE
            llama_tokens draft = common_speculative_draft(
                slot.spec,
                params_spec,
                cached_text_tokens,
                slot.sampled,
                draft_base_pos,
                slot.id);
            if (pxa_sptr.on) pxa_sptr_add(PXA_SPTR_DRAFT, pxa_sptr_t0); // PXA_SPEC_TRACE

            // PXA_MTP_ADAPTIVE: restore the request's speculative params after the capped draft call
            if (pxa_draft_cap > 0) {
                params_spec.n_max = pxa_saved_n_max;
                for (size_t pxa_si = 0; pxa_si < pxa_saved_stage_n_max.size(); ++pxa_si) {
                    params_spec.stages[pxa_si].n_max = pxa_saved_stage_n_max[pxa_si];
                }
            }

            slot.drafted_spec_type = common_speculative_current_type(slot.spec);

            const int n_draft_max = slot.get_n_draft_max();

            if (draft.size() > (size_t)n_draft_max) {
                if (slot.params.speculative.autotune) {
                    // expected near end-of-response when autotune shrinks n_max
                    SLT_DBG(slot, "draft size %d exceeds max %d, truncating\n", (int)draft.size(), n_draft_max);
                } else {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int)draft.size(), n_draft_max);
                }
                draft.resize(n_draft_max);
            }

            ps.do_spec    = true;
            ps.draft      = std::move(draft);
            ps.min_usable = slot.params.speculative.get_min_usable_stage_n_min();
        } else {
            ps.do_spec = false;
        }

        pending.push_back(std::move(ps));
    }

    // ---- equalize: cap all co-decoding drafting slots to a COMMON draft length (uniform verify batch) ----
    // Only matters when 2+ slots decode together on a recurrent/hybrid model (the ragged-batch corruption
    // case). A single decoding slot is the clean single-seq path; non-recurrent models are unaffected by the
    // qwen3next bug, so we leave them at full speculative depth.
    size_t pxa_common_draft_len = SIZE_MAX; // SIZE_MAX = "not constrained"
    if (pending.size() >= 2 && llama_model_has_recurrent(model)) {
        size_t n_spec_slots = 0;
        size_t min_draft    = SIZE_MAX;
        int    max_min_usable = 0;
        for (auto & ps : pending) {
            if (ps.do_spec) {
                ++n_spec_slots;
                min_draft = std::min(min_draft, ps.draft.size());
                max_min_usable = std::max(max_min_usable, ps.min_usable);
            }
        }
        // A non-drafting slot contributes exactly 1 token (just slot.sampled). To keep the WHOLE batch
        // uniform, if any co-decoding slot is NOT drafting this step, every drafting slot must also drop
        // to its sampled token only (common length 0). Likewise if the smallest usable draft falls below
        // the strictest min-usable threshold, drop everyone to non-speculative for this step.
        const bool any_non_spec = (n_spec_slots != pending.size());
        if (n_spec_slots >= 1) {
            if (any_non_spec) {
                pxa_common_draft_len = 0;
            } else {
                pxa_common_draft_len = (min_draft == SIZE_MAX) ? 0 : min_draft;
                if ((int)pxa_common_draft_len < max_min_usable) {
                    pxa_common_draft_len = 0;
                }
            }
            SLT_DBG(*pending[0].slot, "MTP np>1 uniform-batch: capping %d drafting slot(s) to common draft len %d\n",
                    (int)n_spec_slots, (int)pxa_common_draft_len);
        }
    }

    // ---- pass 2: add sampled (+ capped draft) tokens to the shared batch ----
    for (auto & ps : pending) {
        server_slot & slot = *ps.slot;

        // determine this slot's effective draft length for this step
        llama_tokens & draft = ps.draft;
        if (ps.do_spec && pxa_common_draft_len != SIZE_MAX && draft.size() > pxa_common_draft_len) {
            draft.resize(pxa_common_draft_len);
        }

        const bool emit_spec = ps.do_spec
            && (int)draft.size() >= slot.params.speculative.get_min_usable_stage_n_min()
            && !draft.empty();

        if (emit_spec) {
            pxa_sptr.tick_spec = true; // PXA_SPEC_TRACE: this tick verifies a draft
            // add the sampled token to the batch
            slot.i_batch_dft.push_back(batch.n_tokens);
            common_batch_add(batch, slot.sampled, slot.cache_tokens.pos_next(), { slot.id }, true);
            slot.cache_tokens.push_back(slot.sampled);

            // keep track of total number of drafted tokens tested
            slot.n_draft_total += draft.size();

            // add all drafted tokens to the batch
            for (size_t i = 0; i < draft.size(); i++) {
                slot.i_batch_dft.push_back(batch.n_tokens);
                common_batch_add(batch, draft[i], slot.cache_tokens.pos_next(), { slot.id }, true);
                slot.cache_tokens.push_back(draft[i]);
            }
            slot.drafted = std::move(draft);
        } else if (ps.do_spec) {
            // drafted, but the (possibly capped) draft is too small to be worth verifying -> normal decode.
            SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int)draft.size(), slot.params.speculative.get_min_usable_stage_n_min());
            slot.i_batch = batch.n_tokens;
            common_batch_add(batch, slot.sampled, slot.cache_tokens.pos_next(), { slot.id }, true);
            slot.cache_tokens.push_back(slot.sampled);
            slot.drafted.clear();
            slot.drafted_spec_type = COMMON_SPECULATIVE_TYPE_NONE;
            slot.i_batch_dft.clear();
        } else {
            // no speculative decoding
            slot.drafted_spec_type = COMMON_SPECULATIVE_TYPE_NONE;
            slot.i_batch = batch.n_tokens;

            common_batch_add(batch, slot.sampled, slot.cache_tokens.pos_next(), { slot.id }, true);

            slot.cache_tokens.push_back(slot.sampled);

            SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                (int)slot.n_ctx, (int)slot.cache_tokens.size(), (int)slot.truncated);
        }
        slot.n_past = slot.cache_tokens.n_tokens();
    }
}

void  server_context::create_checkpoint_at_interval(server_slot & slot, const gpt_params & params_base, bool in_prompt) {
    if (params_base.do_checkpoint && params_base.ctx_checkpoints_interval > 0) {
        // ---- PXA_CKPT_PROMPT_POLICY (DEFAULT ON since 2026-09-03) ------------------------
        // Mid-prompt, ours fires every --ctx-checkpoints-interval (512) tokens, i.e. once per
        // ubatch at -ub 512: 71 checkpoints on a 20801-token prefill, each one a full
        // two-device drain plus a bulk D2H read of the recurrent state. That is what
        // serializes the V100 pair (measured: --ctx-checkpoints 0 is +8% and takes device-busy
        // from 1.18 to 1.35-1.44), and it is why three separate pipelining mechanisms all
        // measured flat -- they were pipelining between drains.
        //
        // Mainline (tools/server/server-context.cpp, "skip ordinary mid-prompt checkpoints,
        // unless the batch starts a user message or we are near the end of the prompt") keeps
        // only the checkpoints at {4 + n_ubatch, 4} tokens before the end of the prompt, plus
        // user-message boundaries. Armed, we keep the same two: the tolerance-triggered one
        // (slot.do_checkpoint, already in this fork) and one a full ubatch earlier. Those are
        // the ones a later request with a shared prefix can actually restore from; the 69 in
        // the middle were only ever paid for.
        if (in_prompt && llama_pxa_ckpt_prompt_policy()) {
            const int32_t remaining = slot.n_prompt_tokens - slot.n_past_prompt;
            const int32_t near_end  = params_base.n_ubatch + std::max(0, params_base.ctx_checkpoints_tolerance);
            if (remaining > near_end) {
                return;
            }
        }
        auto pos = llama_kv_cache_seq_pos_max(slot.ctx, slot.id);
        if (slot.checkpoint_pos + params_base.ctx_checkpoints_interval <= 1 + pos) {
            bool created = create_checkpoint(slot);
            if (created) {
                slot.checkpoint_pos = pos;
            }
        }
    }
}

void server_context::apply_checkpoint(server_slot & slot) {
    llama_pos pos_next = slot.cache_tokens.pos_next(slot.n_past);
    const auto pos_min_thold = std::max(0, pos_next - 1);
    if (slot.n_past > 0 && slot.n_past < slot.cache_tokens.n_tokens()) {
        int32_t pos_min = llama_kv_cache_seq_pos_min(slot.ctx, slot.id);

        // PXA_CKPT_HYBRID_ROLLBACK_v1 (2026-07-28) — THE recurrent-checkpoint contamination fix.
        //
        // For HYBRID arches (qwen35moe / qwen3next Gated-DeltaNet) the unified cache is
        // one-cell-per-token, so llama_kv_cache_seq_pos_min() returns the ATTENTION side's
        // minimum (~0 with no SWA) and the old `pos_min >= pos_min_thold` gate NEVER fired —
        // while the per-sequence recurrent state row (which llama_kv_cache_seq_rm cannot trim)
        // silently stayed at the END of the previous generation. Any prefix-reuse admission
        // (n_past < cache size: a repeated request, edited history, eval harness re-entry) then
        // decoded new tokens against a recurrent state from the FUTURE of a different
        // generation. One degenerate generation poisoned the slot's state for every later
        // request — measured live: byte-identical greedy requests flipping from clean tool
        // calls to 4000-token loops, `tool_choice:"required"` emitting "{ { { {" x740.
        //
        // The recurrent state's true position IS the newest cached token (seq_pos_max): the
        // state was advanced by every decoded token. Roll back whenever it sits AHEAD of the
        // re-entry point. The checkpoint search must then compare that same quantity —
        // cur.pos_max, the position the SNAPSHOT's recurrent state is at — against the re-entry
        // threshold; the old predicate compared cur.pos_min (== 0 on every hybrid checkpoint),
        // which matched the NEWEST checkpoint unconditionally and restored recurrent state from
        // the poisoned generation itself.
        const bool      is_hybrid = llama_model_is_hybrid(llama_get_model(slot.ctx));
        const llama_pos state_pos = is_hybrid ? llama_kv_cache_seq_pos_max(slot.ctx, slot.id) : -1;

        const bool need_rewind = (pos_min >= pos_min_thold) || (is_hybrid && state_pos > pos_min_thold);

        if (need_rewind) {
            SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, state_pos = %d\n", slot.n_past, (int)slot.cache_tokens.size(), slot.id, pos_min, (int)state_pos);

            // search for a context checkpoint
            const auto it = std::find_if(
                slot.server_cached_prompt.checkpoints.rbegin(),
                slot.server_cached_prompt.checkpoints.rend(),
                [&](const auto & cur) {
                    if (is_hybrid) {
                        // the snapshot's recurrent state includes every token up to cur.pos_max;
                        // it is usable only strictly BEFORE the re-entry point (>= 1 token left
                        // to re-decode for logits, and never state from the divergent future).
                        return cur.pos_max < pos_min_thold;
                    }
                    return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                }
            );

            bool do_reset = it == slot.server_cached_prompt.checkpoints.rend();

            if (!do_reset) {
                // restore the context checkpoint
                const int64_t t_start = ggml_time_us();
                const size_t checkpoint_size = it->data.size();
                // PXA_CKPT_APPLY_SYNC (PXA_KERNEL_FIX_20260611): see create-checkpoint note --
                // never write the bulk recurrent/KV state while prior decode kernels may still
                // be in flight (sticky CUDA illegal-access class).
                llama_synchronize(ctx);
                const size_t n = llama_state_seq_set_data(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                if (n != checkpoint_size) {
                    SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float)checkpoint_size / 1024 / 1024);
                    do_reset = true;
                    //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                } else if (is_hybrid) {
                    // resume EXCLUSIVE of the snapshot position: the restored recurrent state
                    // already contains the token at pos_max — re-decoding it would double-apply
                    // it to the linear-attention state.
                    pos_next = std::min(pos_next, it->pos_max + 1);
                    slot.n_past = slot.cache_tokens.size_up_to_pos(pos_next);

                    pos_next = slot.prompt_tokens.pos_next(slot.n_past_prompt);
                    pos_next = std::min(pos_next, it->pos_max_prompt + 1);
                    slot.n_past_prompt = slot.prompt_tokens.size_up_to_pos(pos_next);
                    SLT_WRN(slot, "restored HYBRID context checkpoint took %.2f ms (state_pos %d -> %d, n_past = %d, size = %.3f MiB)\n", (ggml_time_us() - t_start) / 1000.0, (int)state_pos, it->pos_max, slot.n_past, (float)checkpoint_size / 1024 / 1024);
                } else {
                    pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                    slot.n_past = slot.cache_tokens.size_up_to_pos(pos_next);

                    pos_next = slot.prompt_tokens.pos_next(slot.n_past_prompt);
                    pos_next = std::min(pos_next, std::max(it->pos_min_prompt + 1, it->pos_max_prompt));
                    slot.n_past_prompt = slot.prompt_tokens.size_up_to_pos(pos_next);
                    SLT_WRN(slot, "restored context checkpoint took  %.2f ms (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", (ggml_time_us() - t_start) / 1000.0, it->pos_min, it->pos_max, it->n_tokens, slot.n_past, (float)checkpoint_size / 1024 / 1024);
                }
            }

            if (do_reset) {
                SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA, see %s)\n",
                    "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                slot.n_past = 0;
                slot.n_past_prompt = 0;
                slot.n_past_se = 0;
                slot.ga_i = 0;
                slot.cache_tokens.keep_first(0);
                pos_next = 0;
                common_sampler_reset(slot.ctx_sampling);
            }
        }
    }

    {
        // erase any checkpoints with pos_min > pos_min_thold
        for (auto it = slot.server_cached_prompt.checkpoints.begin(); it != slot.server_cached_prompt.checkpoints.end();) {
            const auto & cur = *it;
            if (cur.pos_max > pos_min_thold) {
                SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, (float)cur.data.size() / 1024 / 1024);
                it = slot.server_cached_prompt.checkpoints.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// PXA port of upstream #2020: when the checkpoint ring is full, evict the interior checkpoint
// whose removal keeps the remaining checkpoints most uniformly spaced (lowest resulting gap
// variance) instead of always dropping the oldest. Preserves long-range re-entry coverage for
// our long multi-turn Discord/agent conversations. First and last checkpoints are never evicted.
static std::list<server_prompt_checkpoint>::iterator evict_checkpoint_by_variance(server_slot & slot, std::list<server_prompt_checkpoint> & ckpts) {
    (void) slot;
    auto it = ckpts.begin();
    if (ckpts.size() < 3) {
        return it;
    } else if (ckpts.size() == 3) {
        std::advance(it, 1);
        return it;
    }
    std::vector<int64_t> tokens;
    tokens.reserve(ckpts.size());
    for (const auto & ckpt : ckpts) {
        tokens.push_back(int64_t(ckpt.pos_max));
    }
    // For each interior checkpoint, the gap variance after removing it is minimised by the
    // smallest product of its two adjacent gaps (first/last are fixed, so the mean is constant):
    // minimise { (x_i - x_{i-1}) * (x_{i+1} - x_i) }.
    size_t best_idx = 1;
    const size_t n = tokens.size();
    const size_t start = 1;     // never remove the first
    const size_t end = n - 1;   // never remove the last
    double max_pos = tokens[n - 1];
    double diff  = (tokens[start] - tokens[start - 1]);
    double diff2 = (tokens[start + 1] - tokens[start]);
    double best_variance = diff * (diff2 / max_pos);
    for (size_t i = start + 1; i < end; i++) {
        diff  = tokens[i] - tokens[i - 1];
        diff2 = tokens[i + 1] - tokens[i];
        double variance = diff * (diff2 / max_pos);
        if (variance < best_variance) {
            best_variance = variance;
            best_idx = i;
        }
    }
    std::advance(it, best_idx);
    return it;
}

bool server_context::create_checkpoint(server_slot & slot) {
    bool do_checkpoint = !slot.image_just_processed;
    int32_t pos_min = llama_kv_cache_seq_pos_min(slot.ctx, slot.id);
    const auto pos_max = llama_kv_cache_seq_pos_max(slot.ctx, slot.id);

    // no need for empty or small checkpoints
    do_checkpoint = do_checkpoint && (pos_min >= 0 && slot.cache_tokens.n_tokens() >= 64);

    // no need to create checkpoints that are too close together
    do_checkpoint = do_checkpoint && (slot.server_cached_prompt.checkpoints.empty() || slot.cache_tokens.n_tokens() > slot.server_cached_prompt.checkpoints.back().n_tokens);

    if (do_checkpoint) {
        const int64_t t_start = ggml_time_us();
        while (slot.server_cached_prompt.checkpoints.size() >= (size_t)params_base.ctx_checkpoints_n) {
            // make room for the new checkpoint, if needed
            auto it = slot.server_cached_prompt.checkpoints.begin();
            if (params_base.ctx_checkpoint_eviction == COMMON_CHECKPOINT_EVICTION_VARIANCE ||
                params_base.ctx_checkpoint_eviction == COMMON_CHECKPOINT_EVICTION_AUTO) {
                it = evict_checkpoint_by_variance(slot, slot.server_cached_prompt.checkpoints);
            }
            const auto & cur = *it;

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                cur.pos_min, cur.pos_max, cur.n_tokens, (float)cur.data.size() / 1024 / 1024);

            slot.server_cached_prompt.checkpoints.erase(it);
        }

        auto & cur = slot.server_cached_prompt.checkpoints.emplace_back();
        server_prompt_checkpoint_update(cur, ctx, slot.id, slot.cache_tokens.n_tokens(), pos_min, pos_max, slot.n_past_offset);

        SLT_WRN(slot, "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB, took %.2f ms)\n",
            (int)slot.server_cached_prompt.checkpoints.size(), params_base.ctx_checkpoints_n, cur.pos_min, cur.pos_max, cur.n_tokens, (float)cur.data.size() / 1024 / 1024,
            (ggml_time_us() - t_start) / 1000.0);
    }
    return do_checkpoint;
}

void server_context::batch_pending_prompt(const int32_t n_ubatch, const int32_t n_batch,  int32_t & batch_type) {
    if (params_base.cont_batching || batch.n_tokens == 0) {
        // PXA_LLAMA_MTP_NP4_FIX: on a recurrent/hybrid model, process at most ONE prompt slot per
        // batch. Two prompt slots of different lengths co-batched make a RAGGED multi-seq u-batch
        // (the qwen3next delta-net split-per-sequence corruption). Stop before adding a second
        // prompt slot; the deferred slot runs alone next tick.
        const bool pxa_one_prompt_per_batch = llama_model_has_recurrent(model);
        bool pxa_prompt_slot_added = false;

        // PXA_PROMPT_FAIR_v1: WHICH pending-prompt slot gets the one prompt seat above.
        // The one-prompt-per-batch invariant is untouched; only the choice of that one slot
        // changes. A slot stays SLOT_STATE_IDLE/SLOT_COMMAND_LOAD_PROMPT for the whole of its
        // prefill (state becomes PROCESSING only once n_past_prompt == n_prompt_tokens), so
        // walking `slots` in id order lets the lowest-numbered slot win every tick until its
        // prompt is fully processed: any request arriving during a deep prefill is head-of-line
        // blocked for that entire prefill. Measured on the 4xP100 150k seat: a 23-token chat on
        // slot 1 waited 463 s behind a 100k prefill on slot 0, its own prompt_ms being 898 ms.
        // Prefer instead a pending slot whose REMAINING prompt fits in a single batch: it
        // completes within this one tick, so the deep prefill yields exactly one tick (a few
        // seconds) instead of blocking short traffic for minutes. Bounded by n_parallel - 1
        // interruptions per tick, since a slot that has been served is no longer pending.
        // PXA_PROMPT_FAIR=0 restores the plain id order.
        static const bool pxa_prompt_fair = !(getenv("PXA_PROMPT_FAIR") && atoi(getenv("PXA_PROMPT_FAIR")) == 0);
        std::vector<server_slot *> pxa_slot_order;
        pxa_slot_order.reserve(slots.size());
        for (auto & pxa_s : slots) { pxa_slot_order.push_back(&pxa_s); }
        if (pxa_prompt_fair && pxa_one_prompt_per_batch && pxa_slot_order.size() > 1) {
            // A launched slot has prompt_tokens populated before its first tick sets
            // n_prompt_tokens, so take the larger of the two (same idiom as kv_unified_cells_busy).
            auto pxa_fits_one_batch = [&](const server_slot * s) {
                if (s->state != SLOT_STATE_IDLE || s->command != SLOT_COMMAND_LOAD_PROMPT) { return false; }
                const int32_t total = std::max(s->n_prompt_tokens, (int32_t) s->prompt_tokens.size());
                const int32_t left  = total - s->n_past_prompt;
                return left > 0 && left <= n_batch;
            };
            const server_slot * pxa_first_pending = nullptr;
            for (const auto * s : pxa_slot_order) {
                if (s->state == SLOT_STATE_IDLE && s->command == SLOT_COMMAND_LOAD_PROMPT) { pxa_first_pending = s; break; }
            }
            std::stable_partition(pxa_slot_order.begin(), pxa_slot_order.end(), pxa_fits_one_batch);
            if (pxa_first_pending != nullptr && pxa_slot_order.front() != pxa_first_pending &&
                pxa_fits_one_batch(pxa_slot_order.front())) {
                LOG_INFO("prompt-fair: short prompt admitted ahead of a deep prefill", {
                    {"id_slot_short",      pxa_slot_order.front()->id},
                    {"n_prompt_tokens",    std::max(pxa_slot_order.front()->n_prompt_tokens,
                                                    (int32_t) pxa_slot_order.front()->prompt_tokens.size())},
                    {"id_slot_deferred",   pxa_first_pending->id},
                    {"n_remaining_deferred", std::max(pxa_first_pending->n_prompt_tokens,
                                                      (int32_t) pxa_first_pending->prompt_tokens.size()) - pxa_first_pending->n_past_prompt},
                    {"n_batch",            n_batch},
                    });
            }
        }

        for (server_slot * pxa_slot_ptr : pxa_slot_order) {
            auto& slot = *pxa_slot_ptr;
            if (pxa_one_prompt_per_batch && pxa_prompt_slot_added) { break; }
            const int32_t pxa_n_tokens_before_slot = batch.n_tokens;
            // this slot still has a prompt to be processed
            if (slot.state == SLOT_STATE_IDLE && slot.command == SLOT_COMMAND_LOAD_PROMPT) {
                auto& prompt_tokens = slot.prompt_tokens;

                // we haven't tokenized the prompt yet - do it now:
                if (prompt_tokens.empty() || slot.n_prompt_tokens == 0) {
                    LOG_VERBOSE("tokenizing prompt", {
                        {"id_slot", slot.id},
                        {"id_task", slot.id_task}
                        });

                    slot.t_start_process_prompt = ggml_time_us();
                    slot.t_start_generation = 0;

                    if (slot.infill) {
                        const bool add_bos = llama_should_add_bos_token(model);
                        bool suff_rm_leading_spc = true;
                        if (params_base.input_suffix.find_first_of(' ') == 0 && params_base.input_suffix.size() > 1) {
                            params_base.input_suffix.erase(0, 1);
                            suff_rm_leading_spc = false;
                        }

                        auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                        auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                        const int space_token = 29871; // TODO: this should not be hardcoded
                        if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token) {
                            suffix_tokens.erase(suffix_tokens.begin());
                        }

                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
                        suffix_tokens.insert(suffix_tokens.begin(), llama_token_suffix(model));

                        auto embd_inp = params_base.spm_infill ? suffix_tokens : prefix_tokens;
                        auto embd_end = params_base.spm_infill ? prefix_tokens : suffix_tokens;
                        if (add_bos) {
                            embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
                        }
                        embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

                        const llama_token middle_token = llama_token_middle(model);
                        if (middle_token >= 0) {
                            embd_inp.push_back(middle_token);
                        }

                        prompt_tokens = server_tokens(embd_inp, false);
                    }
                    else {
                        // prompt_tokens = tokenize(slot.prompt, system_prompt.empty()); // add BOS if there isn't system prompt
                    }

                    slot.n_past = 0;
                    slot.n_prompt_tokens = prompt_tokens.size();

                    LOG_VERBOSE("prompt tokenized", {
                        {"id_slot",         slot.id},
                        {"id_task",         slot.id_task},
                        {"n_ctx",           slot.n_ctx},
                        {"n_keep",          slot.params.n_keep},
                        {"n_prompt_tokens", slot.n_prompt_tokens},
                        {"prompt_tokens", prompt_tokens.detokenize(ctx, true)},
                        });

                    // empty prompt passed -> release the slot and send empty response
                    if (prompt_tokens.empty()) {
                        LOG_INFO("empty prompt - releasing slot", {
                            {"id_slot", slot.id},
                            {"id_task", slot.id_task}
                            });

                        slot.state = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;
                        send_final_response(slot);
                        slot.release();
                        slot.print_timings();
                        continue;
                    }

                    if (slot.embedding) {
                        // this prompt is too large to process - discard it
                        if (slot.n_prompt_tokens > n_ubatch) {
                            slot.state = SLOT_STATE_PROCESSING;
                            slot.command = SLOT_COMMAND_NONE;
                            slot.release();
                            send_error(slot, "input is too large to process. increase the physical batch size", ERROR_TYPE_SERVER);
                            continue;
                        }
                    }
                    else {
                        // if input prompt is too big, truncate it (if group attention self-extend is disabled)
                        // context shift for prompt processing
                        if (slot.ga_n == 1 && slot.n_prompt_tokens >= slot.n_ctx) {
                            if (!params_base.ctx_shift) {
                                send_error(slot, "the request exceeds the available context size, try increasing it", ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }
                            context_shift_prompt(ctx, slot);
                            slot.truncated = true;
                            LOG_VERBOSE("input truncated", {
                                {"id_slot",         slot.id},
                                {"id_task",         slot.id_task},
                                {"n_ctx",           slot.n_ctx},
                                {"n_keep",          slot.params.n_keep},
                                {"n_left",          slot.n_ctx - slot.params.n_keep},
                                {"n_prompt_tokens", slot.n_prompt_tokens},
                                {"prompt_tokens",   prompt_tokens.detokenize(ctx, true)},
                                });

                            GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);

#ifndef NDEBUG
                            // debug
                            common_prefix prefix = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, false);
                            int32_t back = 1;
                            if (slot.cache_tokens.size() && slot.cache_tokens.size() > prefix.first + 20
                                && prefix.second >= back && prefix.first >= back) {
                                LLAMA_LOG_INFO("After context shift :\n");
                                print_tokens(slot.prompt_tokens, slot.cache_tokens, prefix.second - back, prefix.first - back, 50);
                            }
#endif
                        }
                        else {
                            slot.n_discarded_prompt = 0;
                        }
                        common_sampler_reset(slot.ctx_sampling);

                        if (!slot.params.cache_prompt) {
                            slot.n_past_se = 0;
                            slot.ga_i = 0;
                        }
                        else {
                            GGML_ASSERT(slot.ga_n == 1);

                            // reuse any previously computed tokens that are common with the new prompt
                            common_prefix prefix = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, true); // string level match
                            common_prefix prefix_nonexact = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, false);
                            auto n_past0 = slot.cache_tokens.get_common_prefix_exact(prompt_tokens); // token level match
                            LLAMA_LOG_INFO("======== Cache: cache_size = %d, n_past0 =  %d, n_past1 =  %d, n_past_prompt1 = %d,  n_past2 =  %d, n_past_prompt2 =  %d\n", (int32_t)slot.cache_tokens.size(), (int32_t)n_past0, (int32_t)prefix.first, (int32_t)prefix.second, (int32_t)prefix_nonexact.first, (int32_t)prefix_nonexact.second);
                            int32_t size_threshold = 20;
                            if (prefix.first + size_threshold < prefix_nonexact.first) {
                                // LLAMA_LOG_WARN("Common part contains missing or extra space and new line\n");
                                prefix = prefix_nonexact;
                            }
                            slot.n_past = prefix.first;
                            slot.n_past_prompt = prefix.second;
                            slot.n_past_offset = slot.n_past_prompt - slot.n_past;

                            // PXA_SWA_TRUNCATE_GUARD_v1 (2026-08-16): on a masked-SWA arch, reusing
                            // a small prefix of a much larger restored/cached state hands llama_decode
                            // a KV state whose bookkeeping is inconsistent -> an all-non-finite logit
                            // distribution on the very next decode (the Alina NaN cascade). -cram 0
                            // and cache_prompt:false do NOT prevent it: the slot's own cache_tokens
                            // feed the same near-total truncation. When the reuse would discard a
                            // large fraction of a large cached state, force a clean full re-process
                            // (upstream's own SWA answer, PR #13194). Pure continuation (n_past ==
                            // cache size) never triggers, so the growing-conversation fast path is
                            // preserved; the forced reprocess only re-decodes the shared prefix
                            // (~chat template), since the diverging tail was going to be re-decoded
                            // regardless. Thresholds are env-tunable for live sweeps without a rebuild.
                            if (llama_model_n_swa(model) > 0 && !slot.cache_tokens.empty()) {
                                const int32_t cache_sz  = (int32_t) slot.cache_tokens.size();
                                const int32_t discarded = cache_sz - slot.n_past;
                                static const int32_t swa_guard_min_discard = []() {
                                    const char * e = getenv("PXA_SWA_TRUNCATE_MIN_DISCARD");
                                    const int v = e ? atoi(e) : 256;
                                    return v > 0 ? v : 256;
                                }();
                                static const double swa_guard_max_keep_frac = []() {
                                    const char * e = getenv("PXA_SWA_TRUNCATE_MAX_KEEP_FRAC");
                                    const double v = e ? atof(e) : 0.5;
                                    return (v > 0.0 && v < 1.0) ? v : 0.5;
                                }();
                                if (discarded > swa_guard_min_discard &&
                                    slot.n_past < (int32_t)(swa_guard_max_keep_frac * cache_sz)) {
                                    LLAMA_LOG_WARN("PXA_SWA_TRUNCATE_GUARD: forcing full re-process "
                                        "(cache=%d, keep=%d, discard=%d) to avoid restore-then-truncate NaN\n",
                                        cache_sz, (int32_t) slot.n_past, discarded);
                                    slot.n_past = 0;
                                    slot.n_past_prompt = 0;
                                    slot.n_past_offset = 0;
                                    slot.n_discarded_prompt = 0;
                                    slot.n_kept_prompt = 0;
                                    slot.n_prompt_tokens_cache = 0;
                                    slot.server_cached_prompt.checkpoints.clear();
                                    common_sampler_reset(slot.ctx_sampling);
                                }
                            }

                            //if (slot.n_past != slot.n_past_prompt) {
                            //    LLAMA_LOG_INFO("Mistokenization found and handled successfully.\n");
                            //}
                            if ((slot.n_past + size_threshold < slot.cache_tokens.size()))
                            {
                                LLAMA_LOG_WARN("Common part does not match fully\n");
                                int32_t back = 4;
                                if (prefix.second >= back && prefix.first >= back) {
                                    print_tokens(slot.prompt_tokens, slot.cache_tokens, prefix.second - back, prefix.first - back, 30);
                                }
                            }

                            // push the prompt into the sampling context (do not apply grammar)
                            for (int i = 0; i < slot.n_past; ++i) {
                                common_sampler_accept(slot.ctx_sampling, ctx, slot.cache_tokens[i], false);
                            }
                        }
                    }
                    apply_checkpoint(slot);
                    if (slot.n_past_prompt == slot.n_prompt_tokens && slot.n_past_prompt > 0) {
                        // we have to evaluate at least 1 token to generate logits.
                        LOG_INFO("we have to evaluate at least 1 token to generate logits", {
                            { "id_slot", slot.id },
                            { "id_task", slot.id_task }
                            });

                        slot.n_past_prompt--;
                        slot.n_past--;
                        if (slot.ga_i > 0) {
                            slot.n_past_se--;
                        }

                        // PXA_CKPT_HYBRID_ROLLBACK_v1: on a hybrid arch the recurrent state
                        // already contains the token we just un-counted — re-decoding it
                        // without a rollback would double-apply it to the DeltaNet state.
                        // Re-run the rollback against the decremented n_past (it restores a
                        // strictly-older checkpoint or forces a clean full re-process).
                        if (llama_model_is_hybrid(llama_get_model(slot.ctx))) {
                            apply_checkpoint(slot);
                        }
                    }
                    slot.n_prompt_tokens_cache = slot.n_past_prompt;
                    slot.n_prompt_tokens_processed = 0;
                }

                if (slot.embedding) {
                    // cannot fit the prompt in the current batch - will try next iter
                    if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
                        continue;
                    }
                }

                // check that we are in the right batch_type, if not defer the slot
                bool slot_type = slot.embedding ? 1 : 0;
                if (batch_type == -1) {
                    batch_type = slot_type;
                }
                else if (batch_type != slot_type) {
                    continue;
                }

                // keep only the common part
                // remove the non-common part from the cache
                if (slot.n_past < 0)
                {
                    slot.n_past = 0;
                }
                slot.cache_tokens.keep_first(slot.n_past);
                int p0 = (int)system_tokens.size() + slot.n_past;
                p0 = system_tokens.size() + slot.cache_tokens.pos_next();
                auto * ctx_companion = slot.spec ? common_speculative_get_companion_ctx(slot.spec) : nullptr;
                // PXA_WAVE4_FIX_v1: serialize sequence teardown against in-flight compute. Caught
                // on camera 2026-06-11: this seq_rm (p0=0 full wipe on slot reset/new prompt) ran
                // while async work from the previous decode/spec-checkpoint was still in flight
                // (pipeline sched, GGML_SCHED_MAX_COPIES=4, 3-device -sm layer) -> poisoned CUDA
                // context -> cublasSgemm launch failure in the very next graph (fused-MoE path).
                // Mirrors mainline's pipeline-parallelism guard (llama-context.cpp: "we must
                // synchronize before set_inputs") and our own PXA_CKPT_APPLY_SYNC (save side).
                // Drain BOTH contexts before the teardown. Cost: once per prompt admission.
                llama_synchronize(ctx);
                if (ctx_companion != nullptr) {
                    llama_synchronize(ctx_companion);
                }
                const bool target_trimmed = llama_kv_cache_seq_rm(ctx, slot.id, p0, -1);
                const bool companion_trimmed = ctx_companion == nullptr || llama_kv_cache_seq_rm(ctx_companion, slot.id, p0, -1);
                if (!target_trimmed || !companion_trimmed) {
                    // could not partially delete (likely using a non-Transformer model)
                    llama_kv_cache_seq_rm(ctx, slot.id, -1, -1);
                    if (ctx_companion != nullptr) {
                        llama_kv_cache_seq_rm(ctx_companion, slot.id, -1, -1);
                    }

                    p0 = (int)system_tokens.size();
                    if (p0 != 0) {
                        // copy over the system prompt when there is one
                        llama_kv_cache_seq_cp(ctx, 0, slot.id, -1, -1);
                    }

                    // there is no common part left (except for the system prompt)
                    slot.cache_tokens.clear();
                    slot.n_past = 0;
                    slot.n_past_prompt = 0;
                    slot.n_past_offset = 0;
                    slot.n_discarded_prompt = 0;
                    slot.n_kept_prompt = 0;
                    slot.n_past_se = 0;
                    slot.n_prompt_tokens_cache = 0;
                    slot.ga_i = 0;
                    slot.server_cached_prompt.checkpoints.clear();
                    // TODO: is the system prompt ever in the sampling context?
                    common_sampler_reset(slot.ctx_sampling);
                }

                LOG_INFO("kv cache rm [p0, end)", {
                    { "id_slot", slot.id },
                    { "id_task", slot.id_task },
                    { "p0",      p0 }
                    });

                // check if we should process the image
                if (slot.n_past_prompt < slot.n_prompt_tokens
                    && slot.prompt_tokens[slot.n_past_prompt] == LLAMA_TOKEN_NULL) {
                    // process the image
                    size_t n_tokens_out = 0;
                    llama_pos p1 = slot.cache_tokens.pos_next() + slot.n_past_prompt - slot.n_past; // add offset to prompt
                    server_mtp_warmup mtp_media_warmup {
                        ctx,
                        slot.has_mtp && slot.spec ? &slot : nullptr,
                    };
                    mtmd_helper_eval_batch_callback mtp_media_callback =
                        mtp_media_warmup.slot ? server_mtp_media_warmup_callback : nullptr;
                    int32_t res = slot.prompt_tokens.process_chunk(
                            ctx, mctx, slot.n_past_prompt, p1, slot.id, n_tokens_out,
                            mtp_media_callback, &mtp_media_warmup);
                    if (res != 0) {
                        LLAMA_LOG_ERROR("failed to process image, res = %d\n", res);
                        slot.release();
                        send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                        continue;
                    }

                    // add the image chunk to cache
                    {
                        slot.prompt_tokens.free_raw_media_data(slot.n_past_prompt);
                        const auto& chunk = slot.prompt_tokens.find_chunk(slot.n_past_prompt);
                        slot.cache_tokens.push_back(chunk.get()); // copy
                    }

                    slot.n_past += n_tokens_out;
                    slot.n_past_prompt += n_tokens_out;
                    slot.n_prompt_tokens_processed += n_tokens_out;
                    slot.image_just_processed = true; // do not checkpoint right after an image chunk
                }



                int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

                int32_t ga_i = slot.ga_i;
                int32_t ga_n = slot.ga_n;
                int32_t ga_w = slot.ga_w;

                // add prompt tokens for processing in the current batch
                // TODO: the self-extend stuff here is a mess - simplify and/or abstract it somehow
                while (slot.n_past_prompt < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
                    // get next token to process
                    llama_token cur_tok = slot.prompt_tokens[slot.n_past_prompt];
                    if (cur_tok == LLAMA_TOKEN_NULL) {
                        break; // end of text chunk
                    }
                    if (slot.ga_n != 1) {
                        while (slot_npast >= ga_i + ga_w) {
                            const int bd = (ga_w / ga_n) * (ga_n - 1);
                            slot_npast -= bd;
                            ga_i += ga_w / ga_n;
                        }
                    }

                    int p0 = system_tokens.size() + slot.cache_tokens.pos_next();
                    // PXA_MTP_LAZY_PROMPT_FLAGS: see comment above server_slot::need_embd()
                    const bool pxa_tok_embd = pxa_mtp_lazy_prompt_flags() ? slot.embedding : slot.need_embd();
                    common_batch_add(batch, cur_tok, p0, { slot.id }, pxa_tok_embd);

                    slot.cache_tokens.push_back(cur_tok);


                    slot.n_prompt_tokens_processed++;
                    slot_npast++;
                    slot.n_past_prompt++;
                    slot.n_past++;
                    slot.image_just_processed = false;
                    if (params_base.do_checkpoint && slot.n_prompt_tokens - slot.n_past_prompt == params_base.ctx_checkpoints_tolerance) {
                        slot.do_checkpoint = true;
                        break;
                    }

                }
                LOG_VERBOSE("prompt processing progress", {
                    {"id_slot",  slot.id},
                    {"n_past",   slot.n_past},
                    {"n_ctx",    n_ctx},
                    {"n_tokens", batch.n_tokens},
                    {"progress", (float)slot.n_prompt_tokens_processed / slot.n_prompt_tokens},
                    });

                // entire prompt has been processed - start decoding new tokens
                if (slot.n_past_prompt == slot.n_prompt_tokens) {
                    slot.state = SLOT_STATE_PROCESSING;
                    slot.command = SLOT_COMMAND_NONE;
                    GGML_ASSERT(batch.n_tokens > 0);
                    GGML_ASSERT((size_t)slot.n_prompt_tokens == slot.prompt_tokens.size());
                    common_sampler_reset(slot.ctx_sampling);
                    for (int i = 0; i < slot.n_prompt_tokens; ++i) {
                        llama_token id = slot.prompt_tokens[i];
                        if (id != LLAMA_TOKEN_NULL) {
                            common_sampler_accept(slot.ctx_sampling, ctx, id, false);
                        }
                    }

                    // extract the logits only for the last token
                    batch.logits[batch.n_tokens - 1] = true;

                    slot.n_decoded = 0;
                    slot.i_batch = batch.n_tokens - 1;

                    LOG_VERBOSE("prompt done", {
                        {"id_slot",  slot.id},
                        {"n_past",   slot.n_past},
                        {"n_ctx",    n_ctx},
                        {"n_tokens", batch.n_tokens},
                        });
                }
            }
            if (batch.n_tokens > pxa_n_tokens_before_slot) { pxa_prompt_slot_added = true; }

            if (batch.n_tokens >= n_batch) {
                break;
            }
        }
    }
}

void server_context::extend_context(const int32_t n_tokens) {
    for (auto& slot : slots) {
        if (slot.ga_n != 1) {
            // context extension via Self-Extend
            // TODO: simplify and/or abstract this
            while (slot.n_past_se >= slot.ga_i + slot.ga_w) {
                const int ib = (slot.ga_n * slot.ga_i) / slot.ga_w;
                const int bd = (slot.ga_w / slot.ga_n) * (slot.ga_n - 1);
                const int dd = (slot.ga_w / slot.ga_n) - ib * bd - slot.ga_w;

                LOG_TEE("\n");
                LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i, slot.n_past_se, ib * bd, slot.ga_i + ib * bd, slot.n_past_se + ib * bd);
                LOG_TEE("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n, (slot.ga_i + ib * bd) / slot.ga_n, (slot.ga_i + ib * bd + slot.ga_w) / slot.ga_n);
                LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd, slot.ga_i + ib * bd + slot.ga_w + dd, slot.n_past_se + ib * bd + dd);

                llama_kv_cache_seq_add(ctx, slot.id, slot.ga_i, slot.n_past_se, ib * bd);
                llama_kv_cache_seq_div(ctx, slot.id, slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n);
                llama_kv_cache_seq_add(ctx, slot.id, slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd);

                slot.n_past_se -= bd;

                slot.ga_i += slot.ga_w / slot.ga_n;

                LOG_TEE("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", slot.n_past_se + bd, slot.n_past_se, slot.ga_i);
            }

            slot.n_past_se += n_tokens;
        }
    }
}

// Restore recurrent state and re-decode accepted tokens after speculative-decode rejection.
static void restore_speculative_checkpoint(
        server_slot & slot, llama_context * ctx, llama_model * model,
        common_speculative_type spec_type_used,
    llama_token sampled_before,
    const std::vector<llama_token> & ids, int n_draft,
        const std::vector<float> & mtp_hidden_state_pre, int32_t mtp_n_past_base) {
    if (slot.spec_ckpt.per_step_enabled) {
        const int step = (int)ids.size() - 1;
        // PXA_MULTISEQ_CKPT: per-step restore can now legitimately refuse (this seq was not part of
        // the captured batch / the capture was skipped for a non-qualifying batch / step out of the
        // captured range). There is no in-band recovery for per-step mode (no shadow state to replay
        // from), so surface it LOUDLY: on failure the slot's recurrent state still carries the
        // rejected drafts' contamination.
        if (!llama_spec_ckpt_restore(ctx, slot.id, slot.spec_ckpt.n_past, step)) {
            SLT_ERR(slot, "per-step recurrent restore FAILED (step=%d, n_past=%d) - recurrent state may be inconsistent\n",
                    step, slot.spec_ckpt.n_past);
        }

        if (slot.spec_ckpt.sampler) {
            common_sampler_clone(slot.spec_ckpt.sampler, slot.ctx_sampling);
        }
        for (llama_token id : ids) {
            common_sampler_accept(slot.ctx_sampling, ctx, id, true);
        }

        // Update MTP KV cache and hidden state using embeddings collected before checkpoint restore.
        if (slot.has_mtp && !mtp_hidden_state_pre.empty()) {
            if (!common_speculative_commit_accepted_hidden_rows(
                    slot.spec,
                    spec_type_used,
                    slot.id,
                    mtp_n_past_base,
                    sampled_before,
                    ids,
                    mtp_hidden_state_pre)) {
                common_speculative_clear_sequence_hidden(slot.spec, slot.id);
            } else if (spec_type_used != COMMON_SPECULATIVE_TYPE_MTP) {
                SLT_DBG(slot, "%s", "synced MTP target hidden state from accepted-prefix rows after per-step restore");
            }
        }

        SLT_DBG(slot, "per-step restore: step=%d (rejected %d drafts)\n",
            step, (int)(n_draft - (ids.size() - 1)));
    } else {
        // Restore pre-speculation recurrent state then re-decode accepted tokens.
        llama_spec_ckpt_restore(ctx, slot.id, slot.spec_ckpt.n_past, 0);

        if (slot.spec_ckpt.sampler) {
            common_sampler_clone(slot.spec_ckpt.sampler, slot.ctx_sampling);
        }

        if (!ids.empty()) {
            // Re-decode to advance recurrent state to the accepted position.
            const int n_re = (int)ids.size();
            llama_batch re_batch = llama_batch_init(n_re, 0, 1);
            common_batch_add(re_batch, slot.spec_ckpt.sampled, slot.spec_ckpt.n_past, { slot.id }, n_re == 1);
            for (int j = 0; j < n_re - 1; j++) {
                common_batch_add(re_batch, ids[j], slot.spec_ckpt.n_past + 1 + j, { slot.id }, j == n_re - 2);
            }

            if (slot.has_mtp) {
                for (int j = 0; j < re_batch.n_tokens; j++) {
                    re_batch.logits[j] = true;
                }
                llama_set_embeddings(ctx, true);
            }

            const int ret = llama_decode(ctx, re_batch);
            if (ret != 0) {
                SLT_ERR(slot, "failed to re-decode accepted tokens after checkpoint restore: %d\n", ret);
            }
            if (slot.has_mtp) {
                const int n_accepted = (int)ids.size();
                std::vector<int32_t> redecoded_indices(n_accepted);
                for (int j = 0; j < n_accepted; ++j) {
                    redecoded_indices[j] = j;
                }

                if (!common_speculative_commit_accepted_output(
                        slot.spec,
                        ctx,
                        spec_type_used,
                        slot.id,
                        slot.spec_ckpt.n_past,
                        sampled_before,
                        ids,
                        redecoded_indices)) {
                    common_speculative_clear_sequence_hidden(slot.spec, slot.id);
                }
            }

            for (llama_token id : ids) {
                common_sampler_accept(slot.ctx_sampling, ctx, id, true);
            }

            llama_batch_free(re_batch);
            SLT_DBG(slot, "spec checkpoint restored: re-decoded %d tokens (rejected %d drafts)\n",
                n_re, (int)(n_draft - (ids.size() - 1)));
        }
    }

    discard_speculative_checkpoint(slot, ctx);
}

void server_context::speculative_decoding_accept() {
    // PXA_LLAMA_MTP_NP_FIX (blocker #2): the per-slot accept work below performs
    // checkpoint-restore re-decodes and MTP-commit decodes that RESET the global
    // output_ids mapping. At np>1 the verify batch is shared across slots, so a
    // decode triggered while accepting slot A clobbers the verify logits mapping that
    // slot B still needs, causing `llama_get_logits_ith: invalid logits id N`.
    // Fix: do ALL output_ids-dependent reads (sample+accept and accepted-hidden-row
    // copy) for every slot FIRST, while the verify mapping is intact, then run the
    // decoding restore/commit work from the stashed results. Each slot has its own
    // sampler, so pre-sampling per slot is equivalent to the original interleaving.
    std::unordered_map<int, std::vector<llama_token>> pxa_pre_ids;
    std::unordered_map<int, std::vector<float>>       pxa_pre_hidden;
    std::unordered_map<int, bool>                     pxa_pre_sample_failed;
    bool pxa_any_spec_slot = false; // PXA_CKPT_STEP_SCOPED_DISCARD

    // PXA_SHARED_MTP_v1: BATCHED all-accepted commit. When ON (default OFF for safe isolation),
    // the no-rejection MTP commits are DEFERRED and run as ONE multi-seq decode on the shared
    // companion after the per-slot loop, instead of N serial per-slot commit decodes. The rarer
    // rejected/checkpoint-restore commits stay inline+serial (unchanged). Kill-switch: unset the
    // env (default OFF) to restore the fully-serial behavior with zero other changes.
    static const bool pxa_shared_mtp_batch_commit =
        (getenv("PXA_SHARED_MTP_BATCH_COMMIT") && atoi(getenv("PXA_SHARED_MTP_BATCH_COMMIT")) == 1);
    std::vector<common_speculative_commit_req> pxa_batched_commits;
    common_speculative * pxa_batch_commit_spec = nullptr;
    common_speculative_type pxa_batch_commit_type = COMMON_SPECULATIVE_TYPE_NONE;
    for (auto & slot : slots) {
        if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch_dft.empty()) {
            continue;
        }
        slot.ctx_sampling->to_generated_text = &slot.generated_text;
        const size_t n_draft_pp = slot.drafted.size();
        if (n_draft_pp > 0) {
            (void) populate_vocab_pieces();     // max_piece_len
            slot.ctx_sampling->drafted_text.reserve(max_piece_len * n_draft_pp);
            slot.ctx_sampling->drafted_text.clear();
        }
        apply_server_biases(slot);
        std::vector<llama_token> ids_pp;
        try {
            ids_pp = common_sampler_sample_and_accept_n(slot.ctx_sampling, ctx, slot.i_batch_dft, slot.drafted);
        } catch (const std::exception & e) {
            pxa_pre_sample_failed[slot.id] = true;
            pxa_pre_ids[slot.id] = {};
            continue;
        }
        pxa_pre_sample_failed[slot.id] = false;
        // Capture accepted-prefix hidden rows now (output_ids-dependent) for BOTH the
        // any-rejected rollback path AND the accepted-output MTP commit path. Both read
        // output hidden rows by index and must do so before any other slot decodes.
        if (slot.has_mtp && !ids_pp.empty()) {
            std::vector<int32_t> acc_idx(slot.i_batch_dft.begin(), slot.i_batch_dft.begin() + ids_pp.size());
            std::vector<float> hidden_pp;
            if (common_speculative_copy_output_hidden_rows(slot.spec, ctx, acc_idx, hidden_pp)) {
                pxa_pre_hidden[slot.id] = std::move(hidden_pp);
            }
        }
        pxa_pre_ids[slot.id] = std::move(ids_pp);
    }

    for (auto& slot : slots) {

        if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch_dft.empty()) {
            continue;
        }
        pxa_any_spec_slot = true; // PXA_CKPT_STEP_SCOPED_DISCARD

        const llama_token sampled_before = slot.sampled;
        const common_speculative_type spec_type_used = slot.drafted_spec_type;
        size_t n_draft = slot.drafted.size();

        // PXA_LLAMA_MTP_NP_FIX: consume the pre-pass sample result (computed above while the
        // verify output_ids mapping was still intact). Pre-pass already set to_generated_text,
        // drafted_text and applied server biases for this slot.
        if (pxa_pre_sample_failed.count(slot.id) && pxa_pre_sample_failed[slot.id]) {
            LOG_ERROR("speculative sampling failed, releasing slot", {
                {"id_slot", slot.id},
                {"id_task", slot.id_task},
            });
            send_error(slot, std::string("sampling error"), ERROR_TYPE_SERVER);
            slot.release();
            slot.i_batch = -1;
            slot.i_batch_dft.clear();
            slot.drafted.clear();
            continue;
        }
        std::vector<llama_token> ids = std::move(pxa_pre_ids[slot.id]);
        if (getenv("PXA_SPEC_DBG")) {
            std::string ds, is;
            for (auto t : slot.drafted) { ds += std::to_string((int)t); ds += ","; }
            for (auto t : ids)          { is += std::to_string((int)t); is += ","; }
            LLAMA_LOG_INFO("PXA_SPEC_DBG slot %d: draft=[%s] target_ids=[%s] accepted=%d/%d n_idx=%d\n",
                (int)slot.id, ds.c_str(), is.c_str(), (int)ids.size()-1, (int)slot.drafted.size(), (int)slot.i_batch_dft.size());
        }

        const bool any_rejected = (ids.size() - 1) < n_draft;
        int32_t mtp_n_past_base = 0;
        std::vector<float> mtp_hidden_state_pre;
        std::vector<int32_t> accepted_output_indices;
        if (slot.has_mtp) {
            const int32_t n_pre_spec_tokens = slot.cache_tokens.n_tokens() - (int32_t)(slot.drafted.size() + 1);
            mtp_n_past_base = slot.cache_tokens.pos_next(n_pre_spec_tokens);

            if (!ids.empty()) {
                accepted_output_indices.assign(slot.i_batch_dft.begin(), slot.i_batch_dft.begin() + ids.size());
            }

            // PXA_LLAMA_MTP_NP_FIX: use the accepted-prefix hidden rows captured in the pre-pass
            // (reading them here, after another slot may have decoded, would read a clobbered output map).
            if (any_rejected && slot.spec_ckpt.valid && !accepted_output_indices.empty()) {
                auto _ph = pxa_pre_hidden.find(slot.id);
                if (_ph != pxa_pre_hidden.end()) {
                    mtp_hidden_state_pre = std::move(_ph->second);
                } else {
                    mtp_hidden_state_pre.clear();
                }
            }
        }

        slot.i_batch_dft.clear();
        slot.drafted.clear();
        slot.drafted_spec_type = COMMON_SPECULATIVE_TYPE_NONE;

        slot.n_past += ids.size();
        slot.n_decoded += ids.size();
        const int64_t t_current = ggml_time_us();
        slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

        // update how many tokens out of those tested were accepted
        slot.n_draft_accepted += ids.size() - 1;

        // PXA_MTP_ADAPTIVE: feed the per-slot acceptance EMA. Only steps that actually drafted reach
        // this function (i_batch_dft non-empty), so skip-tier steps never decay the EMA - the
        // periodic probe drafts are what keep it alive and let it recover.
        if (n_draft > 0) {
            slot.spec_accept_ema = 0.85f * slot.spec_accept_ema
                                 + 0.15f * ((float) (ids.size() - 1) / (float) n_draft);
        }

        // inform the speculative decoding about the number of accepted tokens
        common_speculative_accept(slot.spec, ids.size() - 1);

        // rollback to the state before sampling the draft tokens
        slot.cache_tokens.keep_first(slot.cache_tokens.n_tokens() - n_draft);

        // add accepted tokens to the prompt
        for (auto it = ids.begin(); it != ids.end() - 1; ++it) {
            slot.cache_tokens.push_back(*it);
        }
        slot.sampled = ids.back(); // last accepted token
        slot.n_past = slot.cache_tokens.n_tokens();

        // for recurrent/hybrid models: if any drafts were rejected, restore recurrent state
        const int64_t pxa_sptr_cm0 = pxa_sptr.on ? ggml_time_us() : 0; // PXA_SPEC_TRACE
        if (any_rejected && slot.spec_ckpt.valid) {
            restore_speculative_checkpoint(slot, ctx, model, spec_type_used, sampled_before, ids, n_draft, mtp_hidden_state_pre, mtp_n_past_base);
        } else {
            if (slot.has_mtp && !accepted_output_indices.empty()) {
                // PXA_LLAMA_MTP_NP_FIX: commit from the pre-captured hidden rows (read while the verify
                // output map was intact) instead of re-reading output hidden rows here, which would hit
                // a clobbered output_ids after a prior slot already decoded its commit/restore.
                auto _ch = pxa_pre_hidden.find(slot.id);
                const bool _have_hidden = (_ch != pxa_pre_hidden.end() && !_ch->second.empty());
                // PXA_SHARED_MTP_v1: DEFER the plain-MTP no-rejection commit into the batched pass.
                // (spec_type_used == MTP is the case where commit_tokens == ids and the batched
                // function applies; the non-MTP-type sync branch below stays inline+serial.)
                if (pxa_shared_mtp_batch_commit && _have_hidden &&
                        spec_type_used == COMMON_SPECULATIVE_TYPE_MTP) {
                    common_speculative_commit_req _req;
                    _req.seq_id         = slot.id;
                    _req.pos_base       = mtp_n_past_base;
                    _req.sampled_before = sampled_before;
                    _req.ids            = ids;             // copy (ids is still used by the emit loop)
                    _req.hidden_rows    = &_ch->second;    // map entry outlives this function
                    pxa_batched_commits.push_back(std::move(_req));
                    pxa_batch_commit_spec = slot.spec;     // identical across all MTP slots (shared)
                    pxa_batch_commit_type = spec_type_used;
                } else {
                    std::vector<float> _commit_hidden;
                    if (_have_hidden) {
                        _commit_hidden = _ch->second;
                    }
                    // PXA_MTP_PREFETCH: route through the async wrapper so the serial ~9ms/round
                    // companion commit decode overlaps token emit + the next tick. The wrapper is
                    // the env gate: unset -> the identical synchronous call, byte-for-byte.
                    bool _committed = !_commit_hidden.empty() &&
                        common_speculative_commit_accepted_hidden_rows_async(
                            slot.spec, spec_type_used, slot.id, mtp_n_past_base,
                            sampled_before, ids, _commit_hidden);
                    if (!_committed) {
                        common_speculative_clear_sequence_hidden(slot.spec, slot.id);
                    } else if (spec_type_used != COMMON_SPECULATIVE_TYPE_MTP) {
                        SLT_DBG(slot, "%s", "synced MTP target hidden state from accepted-prefix rows");
                    }
                }
            }
            llama_kv_cache_seq_rm(ctx, slot.id, slot.cache_tokens.pos_next(slot.n_past), -1);
            discard_speculative_checkpoint(slot, ctx);
        }
        if (pxa_sptr.on) pxa_sptr_add(PXA_SPTR_COMMIT, pxa_sptr_cm0); // PXA_SPEC_TRACE

        for (size_t i = 0; i < ids.size(); ++i) {
            completion_token_output result;

            result.tok = ids[i];
            result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
            result.prob = 1.0f; // set later

            if (slot.sparams.n_probs > 0) {
                populate_token_probs(slot, result, slot.params.post_sampling_probs, params_base.special, i);
            }

            if (slot.n_buffer == 0 || !params_base.can_ban_phrases) {
                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    send_final_response(slot);
                    release_slot_after_final_response(slot);
                    break;
                }
            } else {
                buffer_and_check_string_ban(slot, result);
                if (slot.task == nullptr) {
                    break;
                }
            }

            common_sampler_review(slot.ctx_sampling, slot.token_buffer.size(), slot.rewind_status);

            update_allowlist_state(slot);
        }
        SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int)ids.size() - 1, (int)slot.drafted.size(), slot.n_past);
        LOG_VERBOSE("speculative decoding result", {
            {"id_slot", slot.id},
            {"accepted", (int)ids.size() - 1},
            {"total", (int)slot.drafted.size()},
            {"new_n_past", slot.n_past}
            });
    }

    // PXA_SHARED_MTP_v1: flush the DEFERRED no-rejection MTP commits as ONE batched multi-seq
    // decode on the shared companion (instead of the N serial per-slot commits above). Runs after
    // all per-slot main-ctx work (the pre-captured hidden rows it reads are independent of that
    // work). On a per-seq failure, clear that seq's hidden state (same fail-safe as the inline path).
    if (pxa_shared_mtp_batch_commit && pxa_batch_commit_spec != nullptr && !pxa_batched_commits.empty()) {
        const std::vector<llama_seq_id> _failed =
            common_speculative_commit_accepted_hidden_rows_batched(
                pxa_batch_commit_spec, pxa_batch_commit_type, pxa_batched_commits);
        for (const llama_seq_id _sid : _failed) {
            common_speculative_clear_sequence_hidden(pxa_batch_commit_spec, _sid);
        }
    }

    // PXA_CKPT_STEP_SCOPED_DISCARD: ONE ctx-global discard per verify step, AFTER every
    // co-decoding slot finished its accept/restore against the step's shared checkpoint
    // (stops the per-step capture and releases the shadow state until the next save loop).
    if (pxa_any_spec_slot) {
        llama_spec_ckpt_discard(ctx);
    }
}


bool server_context::accept_special_token(const server_slot& slot, const  llama_token token) {
    return params_base.special || slot.sparams.preserved_tokens.find(token) != slot.sparams.preserved_tokens.end();
}

void server_context::release_slot_after_final_response(server_slot & slot) {
    slot.print_timings();
    if (params_base.do_checkpoint) {
        create_checkpoint(slot);
    }
    slot.release();
    slot.released = true;
    metrics.on_prediction(slot);
}

void server_context::send_token_results(completion_token_outputs& results, server_slot& slot, int32_t n) {
    int count = 0;
    bool released = false;

    int32_t start_pos = slot.n_past - (int32_t)slot.token_buffer.size() + 1;

    for (auto& it : results) {
        bool has_next = process_token(it, slot);

        // Clean up positional bans for the token we just confirmed/sent
        slot.positional_bans.erase(start_pos + count);

        count++;
        if (!has_next) {
            if (slot.stopped_limit && !slot.stopped_eos && !slot.stopped_word) {
                continue;
            }
            send_final_response(slot);
            release_slot_after_final_response(slot);
            released = true;
            break;
        }
        if (n > 0 && count >= n) {
            break;
        }
    }

    if (!released && slot.stopped_limit && !slot.stopped_eos && !slot.stopped_word) {
        send_final_response(slot);
        release_slot_after_final_response(slot);
    }

    if (count > 0) {
        slot.sampled = results[results.size()-1].tok;
        results.erase(results.begin(), results.begin() + count);
    }

}

inline int32_t check_ban_phrase(server_slot& slot) {
    if (slot.token_buffer.empty()) return -1;

    std::string string_buffer;
    std::vector<size_t> token_offsets;

    for (const auto& it : slot.token_buffer) {
        token_offsets.push_back(string_buffer.size());
        string_buffer += it.text_to_send;
    }

    size_t best_start = std::string::npos;
    bool found = false;
    std::string string_buffer_lower = string_lower(string_buffer);

    // 1. Check strings
    for (const auto& phrase : slot.ban_phrases) {
        size_t start = string_buffer_lower.find(phrase);
        if (start != std::string::npos) {
            if (start < best_start) {
                best_start = start;
            found = true;
        }
    }
    }

    // 2. Check regex
    for (const auto& pattern : slot.ban_regex) {
        try {
            std::regex re(pattern);
            std::smatch match;
            if (std::regex_search(string_buffer, match, re)) {
                if (match.position() < best_start) {
                    best_start = match.position();
                    found = true;
                }
            }
        } catch (...) { continue; }
    }

    // 3. Check regex case insensitive
    for (const auto& pattern : slot.ban_regex_ci) {
        try {
            std::regex re(pattern, std::regex_constants::icase);
            std::smatch match;
            if (std::regex_search(string_buffer, match, re)) {
                if (match.position() < best_start) {
                    best_start = match.position();
                    found = true;
                }
            }
        } catch (...) { continue; }
    }

    if (found) {
        int32_t token_idx = -1;
        for (size_t i = 0; i < token_offsets.size(); ++i) {
            size_t len = (i == token_offsets.size() - 1)
                ? string_buffer.size() - token_offsets[i]
                : token_offsets[i+1] - token_offsets[i];

            if (best_start >= token_offsets[i] && best_start < token_offsets[i] + len) {
                token_idx = (int32_t)i;
                break;
    }
}

        if (token_idx != -1) {
            int32_t abs_pos = slot.n_past - (int32_t)slot.token_buffer.size() + 1 + token_idx;
            return abs_pos;
        }
    }

    return -1;
}

inline void rewind_context(server_slot& slot, int32_t ban_pos) {
    slot.rewind_count++;

    int32_t buffer_start_pos = slot.n_past - (int32_t)slot.token_buffer.size() + 1;
    int32_t n_keep_buffer = ban_pos - buffer_start_pos;
    if (n_keep_buffer < 0) n_keep_buffer = 0;

    if (slot.banned_n != 0) {
        int32_t n = 0;
        for (auto result = slot.token_buffer.begin() + n_keep_buffer; result != slot.token_buffer.end(); result++) {
            llama_token banned_tok = result->tok;

            if (n == 0) {
                LLAMA_LOG_DEBUG("Banned pattern detected at pos %d. Banning token %d ('%s') and rewinding.\n",
                    ban_pos, banned_tok, result->text_to_send.c_str());
            }

            slot.positional_bans[ban_pos].insert(banned_tok);
            n++;
            if (slot.banned_n > 0 && n == slot.banned_n) {
                break;
            }
        }
    }

    int32_t n_rewind_total = (slot.n_past + 1) - ban_pos;

    size_t n_keep_cache = 0;
    if (ban_pos > 0) {
        n_keep_cache = (size_t)(ban_pos - 1);
    }

    if (n_keep_cache > slot.cache_tokens.size()) {
        n_keep_cache = slot.cache_tokens.size();
    }

    if (n_keep_cache < slot.cache_tokens.size()) {
        slot.sampled = slot.cache_tokens[n_keep_cache];
    } else {
        slot.sampled = 0;
    }

    // Truncate cache
    slot.cache_tokens.keep_first(n_keep_cache);
    slot.n_past = slot.cache_tokens.n_tokens();

    // Remove from KV cache
    llama_kv_cache_seq_rm(slot.ctx, slot.id, slot.cache_tokens.pos_next(slot.n_past), -1);

    // Truncate buffer
    slot.token_buffer.resize(n_keep_buffer);

    // Adjust decoded count
    if (slot.saturate_predict) {
        slot.n_decoded -= n_rewind_total;
        if (slot.n_decoded < 0) slot.n_decoded = 0;
    }
}

void server_context::buffer_and_check_string_ban(server_slot & slot, completion_token_output & result) {
    slot.token_buffer.push_back(result);

    bool next_token = has_next_token(result, slot);
    // If buffer full or generation stopped, we might send tokens
    bool buffer_full = slot.token_buffer.size() >= slot.n_buffer;

    int32_t ban_pos = -1;
    bool sent_results = false;

    // Always reset logit bias to base before checking bans
    slot.ctx_sampling->params.logit_bias = slot.logit_bias;

    if (slot.ban_phrases.size() > 0 || slot.ban_regex.size() > 0 || slot.ban_regex_ci.size() > 0) {
        ban_pos = check_ban_phrase(slot);
    }

    bool allow_rewind = true;

    if (ban_pos >= 0) {
        if (slot.rewind_count_max == -1) {
            // Automatic / Heuristic logic
            // Account for strings + regex + regex_ci
            size_t total_bans = slot.ban_phrases.size() + slot.ban_regex.size() + slot.ban_regex_ci.size();

            // Heuristic: Allow if under 20 OR under 2 * total_bans
            // Conversely: Stop if >= 20 AND > 2 * total_bans
            if (slot.rewind_count >= 20 && slot.rewind_count > 2 * total_bans) {
                allow_rewind = false;
            }
        }
        else if (slot.rewind_count_max > 0) {
            // Strict limit logic
            if (slot.rewind_count >= slot.rewind_count_max) {
                allow_rewind = false;
            }
        }
        // If slot.rewind_count_max == 0, allow_rewind remains true (Infinite)
    }

    if (ban_pos >= 0 && allow_rewind) {
        rewind_context(slot, ban_pos);
        slot.rewind_status = true;
    }
    else if (buffer_full || !next_token) {
        slot.rewind_status = false;
        slot.rewind_count = 0;

        if (!next_token) {
            // send all remaining tokens
            send_token_results(slot.token_buffer, slot);
        }
        else {
            // send 1 token from the front (FIFO)
            send_token_results(slot.token_buffer, slot, 1);
        }
    }
    else {
        // buffer the result, wait for more tokens to validate string
        slot.sampled = result.tok;
    }
}

void server_context::update_allowlist_state(server_slot& slot) {
    const auto& kws = slot.allow_kws;
    auto& idx = slot.allow_idx;
    if ((slot.allow_kw_delay > slot.n_decoded) || (idx >= kws.size())) {
        return;
    }

    // search for keyword
    auto kw = kws[idx];
    auto pos = slot.generated_text.find(kw, std::max(0, slot.last_gentxt_size - (int32_t)kw.length() + 1));
    while (pos != std::string::npos) {
        if (++idx >= kws.size()) {
            break;
        }
        kw = kws[idx];
        pos = slot.generated_text.find(kw, pos + 1);
    }
}

void server_context::process_batch_tokens(int32_t & n_batch) {
    // PXA_DECODE_WALL_DBG: tick-entry marker
    if (getenv("PXA_DECODE_WALL_DBG")) {
        const int64_t d0 = last_decode_done_us.load(std::memory_order_relaxed);
        if (d0) fprintf(stderr, "PXA_PBT_ENTRY: since_done=%lldus\n", (long long) (ggml_time_us() - d0));
    }
    bool defrag_tried = false; // PXA_HYBRID_CTX_SHIFT_v3: one compact-and-retry per batch pass
    for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
        const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);
        extend_context(n_tokens);

        llama_batch batch_view = {
            n_tokens,
            batch.token + i,
            nullptr,
            batch.pos + i,
            batch.n_seq_id + i,
            batch.seq_id + i,
            batch.logits + i,
            0, 0, 0, // unused
        };

        last_decode_start_us.store(ggml_time_us(), std::memory_order_relaxed);
        const int ret = llama_decode(ctx, batch_view);
        last_decode_done_us.store(ggml_time_us(), std::memory_order_relaxed);
        // PXA_SPEC_TRACE: the target verify/generation decode wall of this tick
        if (pxa_sptr.on) {
            pxa_sptr.tk[PXA_SPTR_VERIFY] +=
                last_decode_done_us.load(std::memory_order_relaxed) -
                last_decode_start_us.load(std::memory_order_relaxed);
        }
        // PXA_SYNC_BISECT: was the 620ms queued during llama_decode?
        if (getenv("PXA_SYNC_BISECT")) {
            const int64_t t0 = ggml_time_us();
            llama_synchronize(ctx);
            const int64_t el = ggml_time_us() - t0;
            fprintf(stderr, "PXA_BISECT post-decode drain us=%lld (n_tok=%d)\n", (long long) el, (int) n_tokens);
        }
        // PXA_DECODE_WALL_DBG: per-call llama_decode wall (profiling aid, env-gated, zero cost off)
        if (getenv("PXA_DECODE_WALL_DBG")) {
            static int64_t pxa_prev_done = 0;
            const int64_t s0 = last_decode_start_us.load(std::memory_order_relaxed);
            const int64_t d0 = last_decode_done_us.load(std::memory_order_relaxed);
            fprintf(stderr, "PXA_DECODE_WALL: n_tok=%d wall=%lldus gap_before=%lldus\n",
                    (int) n_tokens, (long long) (d0 - s0), (long long) (pxa_prev_done ? s0 - pxa_prev_done : 0));
            pxa_prev_done = d0;
        }
        if (ret != 0) {
            // PXA_HYBRID_CTX_SHIFT_v3: concurrent context shifts leave the unified cache
            // FRAGMENTED (observed at np=4: 461 free cells but max contiguous run 9 — a
            // 12-token batch cannot land even though plenty is free). The defrag_thold
            // trigger never fires in that state (a near-full cache has a LOW fragmentation
            // ratio), so compact reactively: defrag once and retry the SAME chunk at the
            // same batch size before falling back to halving. build_defrag skips recurrent
            // layers, so this is hybrid-safe.
            if (ret == 1 && !defrag_tried) {
                defrag_tried = true;
                llama_synchronize(ctx); // PXA_WAVE4_FIX_v1 discipline: never rewrite KV mid-flight
                llama_kv_cache_defrag(ctx);
                llama_kv_cache_update(ctx);
                LOG_WARNING("KV alloc failed - defragmented the cache, retrying the chunk", {
                    {"i", i}, {"n_batch", n_batch},
                    });
                i -= n_batch; // for-increment restores i -> same chunk, same size
                continue;
            }
            if (n_batch == 1 || ret < 0) {
                int user_cancel = -3;
                if (ret == user_cancel) {
                    LLAMA_LOG_INFO("Decode process is cancelled by user.\n");
                }
                else {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_ERROR("failed to decode the batch: KV cache is full - try increasing it via the context size", {
                        {"i",   i},
                        {"n_batch",  ret},
                        {"ret",   ret},
                        });
                }

                for (auto& slot : slots) {
                    if (ret == user_cancel) {
                        // PXA_SCOPED_CANCEL_v1: unwind ONLY slots that were part of the aborted
                        // decode (processing, or a just-launched prompt). The old code flipped EVERY
                        // slot to PROCESSING and released it — one client disconnect nuked all
                        // in-flight gens and their clients hung forever (release() posts no result).
                        // With sole-occupant gating upstream this branch normally only ever sees the
                        // canceller's own slots; the guards below make the racy case merely lossy
                        // (victim gets an immediate, honest error instead of a silent forever-hang).
                        if (!slot.is_processing() &&
                            !(slot.state == SLOT_STATE_IDLE && slot.command == SLOT_COMMAND_LOAD_PROMPT)) {
                            continue; // truly idle slot — leave its state alone
                        }
                        slot.state = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;
                        llama_pos cur_pos = llama_kv_cache_seq_pos_max(slot.ctx, slot.id);
                        slot.n_past = slot.cache_tokens.size_up_to_pos(cur_pos + 1);
                        slot.cache_tokens.keep_first(slot.n_past);
                        // never leave a still-connected client hanging: if the client is already
                        // gone (the normal case — it cancelled us) the error result is dropped.
                        send_error(slot, "request cancelled: decode aborted mid-flight", ERROR_TYPE_SERVER);
                        slot.release();
                    }
                    else {
                        // PXA_KV_UNIFIED_v1: this branch historically unwinds EVERY slot in `slots`,
                        // including ones that were never part of the failing batch -- it force-flips
                        // truly idle slots to PROCESSING and then releases them, posting an error to
                        // a stale id_task. Under the fixed n_ctx/n_parallel split that blast radius
                        // was mostly theoretical, because a slot could not overrun its private window
                        // in the first place. Under the shared ring it is exactly the "one slot's
                        // exhaustion corrupts another slot's request" failure this feature must not
                        // have, so scope the unwind to slots that were actually in flight -- the same
                        // discipline PXA_SCOPED_CANCEL_v1 already applies to the cancel branch above.
                        // Gated on the lever: with it off, behaviour is bit-for-bit as before.
                        if (params_base.kv_unified &&
                            !slot.is_processing() &&
                            !(slot.state == SLOT_STATE_IDLE && slot.command == SLOT_COMMAND_LOAD_PROMPT)) {
                            continue; // untouched bystander — leave its state and its cache alone
                        }
                        slot.state = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;
                        slot.release();
                        LLAMA_LOG_INFO("n_past = %d\n", (int)slot.cache_tokens.size());
                        send_error(slot, "Input prompt is too big compared to KV size. Please try increasing KV size.");
                    }
                }
                break; // break loop of n_batch
            }
            // retry with half the batch size to try to find a free slot in the KV cache
            n_batch /= 2;
            i -= n_batch;

            LOG_WARNING("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation", {
                {"i",   i},
                {"n_batch",  n_batch},
                {"ret",   ret},
                });

            continue; // continue loop of n_batch
        }

    if (server_speculative_has_mtp(params_base.speculative)) {
        const int64_t pxa_wm0 = getenv("PXA_DECODE_WALL_DBG") ? ggml_time_us() : 0;
        for (auto & slot : slots) {
            if (!slot.spec || !slot.has_mtp) {
                continue;
            }

            if ((slot.state != SLOT_STATE_PROCESSING || slot.n_decoded != 0) &&
                (slot.state != SLOT_STATE_IDLE || slot.command != SLOT_COMMAND_LOAD_PROMPT)) {
                continue;
            }

            if (common_speculative_on_target_seq_batch(slot.spec, ctx, batch_view, slot.id, true) != 0) {
                LOG_ERROR("failed to warm up MTP state from prompt batch for slot %d\n", slot.id);
            }
        }
        if (pxa_wm0) fprintf(stderr, "PXA_WARMUP_BLK: %lldus\n", (long long) (ggml_time_us() - pxa_wm0));
    }
    if (getenv("PXA_DECODE_WALL_DBG")) {
        const int64_t d0 = last_decode_done_us.load(std::memory_order_relaxed);
        if (d0) fprintf(stderr, "PXA_PBT_PRESLOTS: since_done=%lldus\n", (long long) (ggml_time_us() - d0));
    }

        for (auto& slot : slots) {
            bool is_active_slot = (slot.state == SLOT_STATE_PROCESSING);

            if (!is_active_slot || slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
                // save checkpoint during prompt processing
                if (slot.command == SLOT_COMMAND_LOAD_PROMPT) {
                    const int64_t pxa_ck0 = getenv("PXA_DECODE_WALL_DBG") ? ggml_time_us() : 0;
                    if (slot.do_checkpoint) {
                        create_checkpoint(slot);
                    } else {
                        create_checkpoint_at_interval(slot, params_base, /* in_prompt = */ true);
                    }
                    if (pxa_ck0) {
                        const int64_t el = ggml_time_us() - pxa_ck0;
                        if (el > 2000) fprintf(stderr, "PXA_CKPT slot=%d do=%d took=%lldus\n", slot.id, (int) slot.do_checkpoint, (long long) el);
                    }
                }
                continue; // continue loop of slots
            }

            // prompt evaluated for embedding
            if (slot.embedding) {
                send_embedding(slot, batch_view);
                slot.release();
                slot.i_batch = -1;
                continue; // continue loop of slots
            }

            if (slot.n_decoded == 0 && slot.can_speculate()) {
                static const llama_tokens empty_prompt;
                const llama_tokens & spec_prompt = slot.has_mtp && !slot.params.speculative.has_composite_stage_chain()
                    ? empty_prompt
                    : slot.cache_tokens.get_text_tokens();
                // PXA_SHARED_MTP_v1: per-seq begin so a new generation in this slot resets ONLY
                // this seq's MTP caches, never the in-flight state of other slots sharing the spec.
                common_speculative_begin_seq(slot.spec, slot.id, spec_prompt);
            }

            if (slot.i_batch_dft.size() > 0) {
                continue; // sample using speculative decoding
            }

            // RESTORE AND APPLY POSITIONAL BANS
            slot.ctx_sampling->params.logit_bias = slot.logit_bias;
            auto ban_it = slot.positional_bans.find(slot.n_past);
            if (ban_it != slot.positional_bans.end()) {
                for (llama_token tok : ban_it->second) {
                    slot.ctx_sampling->params.logit_bias[tok] += slot.ban_phrases_bias;
                }
            }

            slot.ctx_sampling->to_generated_text = &slot.generated_text;

            completion_token_output result;
            const int tok_idx = slot.i_batch - i;

            // PXA_MTP_ADAPTIVE: in the skip tier (spec_adaptive_tier == 0) this slot decodes WITHOUT
            // speculation, so nothing refreshes the MTP target hidden state (normally maintained by
            // the spec commit/restore paths). A later full-length probe would then draft from a
            // hidden state many tokens stale (plus the drafter's stale cached token) and be rejected
            // regardless of content, making recovery impossible. Capture the output hidden on every
            // plain-decode step while in the skip tier: cheap (one feature-row copy; need_embd() is
            // true for MTP slots) and capture also invalidates the drafter's stale cached token.
            // PXA_MTP_NP_CTXMTP_SYNC_v1: a slot reaches this plain-decode branch only when it did NOT
            // speculate this step (the i_batch_dft>0 spec slots continue'd above and are advanced by
            // speculative_decoding_accept()). A non-spec generation decode never passes through the
            // spec-accept commit, so it never advances this slot's per-slot companion ctx_mtp KV (only
            // the commit / prompt-warmup do). On the HYBRID qwen35moe the companion state must stay
            // CONTIGUOUS; at np>1 the uniform-batch equalizer forces these non-spec decodes, so the
            // skipped position leaves a growing gap -> "MTP context not fully warmed up: pos_max=N,
            // expected>=N+1" -> drafts from a stale companion state -> rejected/empty -> wedge (empty
            // gens, 0% util). Advance ctx_mtp by this one just-decoded token via the warmup machinery
            // with is_prompt_warmup=false (stores the fresh target hidden AND runs mtp_update_kv_cache),
            // keeping the companion KV contiguous. Gated to np>1 + n_decoded>0 so np=1 (rock-solid) and
            // the prompt boundary stay byte-unchanged. Fail-safe: on error drop this seq's stale hidden
            // so the next draft bails cleanly (returns empty -> plain decode) instead of corrupting ctx_mtp.
            if (slot.has_mtp && (int) slots.size() > 1 && slot.n_decoded > 0) {
                if (common_speculative_on_target_seq_batch(slot.spec, ctx, batch_view, slot.id, false) != 0) {
                    common_speculative_clear_sequence_hidden(slot.spec, slot.id);
                }
            } else if (slot.has_mtp && (slot.n_decoded == 0 || slot.spec_adaptive_tier == 0)) {
                (void) common_speculative_capture_output_hidden(slot.spec, ctx, tok_idx, slot.id, slot.n_past);
            }

            apply_server_biases(slot);

            llama_token id;
            try {
                id = common_sampler_sample(slot.ctx_sampling, ctx, tok_idx);
                common_sampler_accept(slot.ctx_sampling, ctx, id, true);
                // PXA_SOFTFAIL_BREAKER_v1: the sampler cannot end a request — on an all-non-finite
                // (unsampleable) distribution it can only return a deterministic fallback token and
                // let generation continue. With n_predict=-1 that degenerates into an unbounded run
                // that fills the context, holds the slot, and under -sm layer starves every
                // co-resident generation (the "healthy /health, HTTP=000" outage). Count consecutive
                // fallbacks per slot; process_token stops the slot once the cap is reached.
                if (llama_get_last_sample_softfailed(ctx)) {
                    slot.n_softfail_consecutive++;
                } else {
                    slot.n_softfail_consecutive = 0;
                }
            } catch (const std::exception & e) {
                LOG_ERROR("sampling failed, releasing slot", {
                    {"id_slot", slot.id},
                    {"id_task", slot.id_task},
                    {"error", e.what()},
                });
                send_error(slot, std::string("sampling error: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
                slot.i_batch = -1;
                continue;
            }

            slot.n_decoded += 1;
            const int64_t t_current = ggml_time_us();

            if (slot.n_decoded == 1) {
                slot.t_start_generation = ggml_time_us();
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
                // create checkpoint after prompt processing ends
                if (params_base.ctx_checkpoints_tolerance<=0 && params_base.do_checkpoint) {
                    create_checkpoint(slot);
                }
            }

            // create checkpoint during generation
            if (slot.n_decoded > 1) {
                create_checkpoint_at_interval(slot, params_base);
            }

            slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

            result.tok = id;
            result.prob = 1.0f; // TODO: set it here instead of doing inside populate_token_probs
            result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
            if (slot.sparams.n_probs > 0) {
                populate_token_probs(slot, result, slot.params.post_sampling_probs, params_base.special, tok_idx);
            }

            // no ban string for recurrent/hybrid model
            if (slot.n_buffer == 0 || !params_base.can_ban_phrases) {
                slot.token_buffer = { result };
                send_token_results(slot.token_buffer, slot);
            } else {
                // buffer the result and check string ban.
                // if ban, we need to go back, apply logit bias and regenerate
                buffer_and_check_string_ban(slot, result);
            }

            common_sampler_review(slot.ctx_sampling, slot.token_buffer.size(), slot.rewind_status);

            update_allowlist_state(slot);

            slot.i_batch = -1;
        }
        // PXA_DECODE_WALL_DBG: post-slot-loop marker
        if (getenv("PXA_DECODE_WALL_DBG")) {
            const int64_t d0 = last_decode_done_us.load(std::memory_order_relaxed);
            if (d0) fprintf(stderr, "PXA_PBT_POSTSLOTS: since_done=%lldus\n", (long long) (ggml_time_us() - d0));
        }
        // speculative decoding - main model sample and accept
        {
            const int64_t pxa_ac0 = getenv("PXA_DECODE_WALL_DBG") ? ggml_time_us() : 0;
            const int64_t pxa_sptr_ac0 = pxa_sptr.on ? ggml_time_us() : 0; // PXA_SPEC_TRACE
            speculative_decoding_accept();
            if (pxa_sptr.on) pxa_sptr_add(PXA_SPTR_ACCEPT, pxa_sptr_ac0); // PXA_SPEC_TRACE
            if (pxa_ac0) fprintf(stderr, "PXA_ACCEPT_BLK: %lldus\n", (long long) (ggml_time_us() - pxa_ac0));
        }
    }
}

bool server_context::pxa_decode_wedged(int64_t stall_ms) const {
    if (stall_ms <= 0) return false;
    const int64_t s = last_decode_start_us.load(std::memory_order_relaxed);
    const int64_t d = last_decode_done_us.load(std::memory_order_relaxed);
    if (s <= d) return false; // no decode in flight (idle, or the last one returned)
    return (ggml_time_us() - s) / 1000 > stall_ms;
}

void server_context::update_slots() {
    // PXA_CANCEL_TICK_RESET: clear a sticky process-global stop flag at the very top of
    // each tick. llama_decode_stop() is global; if it was set between ticks (e.g. a client
    // disconnect that landed while the queue was idle, or any path that left it set) it would
    // abort the NEXT, unrelated llama_decode and return ret=-3 (user_cancel) -> wedge. Resetting
    // here is safe: an in-tick abort is still honored because stop_internal_decode is polled
    // INSIDE llama_decode AFTER this reset (src/llama.cpp). Pairs with FIX 1 (gating the trigger).
    llama_decode_reset();

    // PXA_DECODE_WALL_DBG: update_slots section markers
    const bool pxa_us_dbg = getenv("PXA_DECODE_WALL_DBG") != nullptr;
    const int64_t pxa_us_t0 = pxa_us_dbg ? ggml_time_us() : 0;
    auto pxa_us_mark = [&](const char * what) {
        if (pxa_us_dbg) {
            const int64_t d0 = last_decode_done_us.load(std::memory_order_relaxed);
            fprintf(stderr, "PXA_US %s: since_entry=%lldus since_done=%lldus\n",
                    what, (long long) (ggml_time_us() - pxa_us_t0), (long long) (d0 ? ggml_time_us() - d0 : 0));
        }
    };
    pxa_us_mark("entry");
    pxa_sptr_tick_begin(); // PXA_SPEC_TRACE

    if (system_need_update) {
        system_prompt_update();
    }
    // release slots
    release_slots();
    pxa_us_mark("post_release");

    // check if all slots are idle
    if (slots_idle()) {
        return;
    }

    {
        LOG_VERBOSE("posting NEXT_RESPONSE", {});
        server_task task;
        task.type = SERVER_TASK_TYPE_NEXT_RESPONSE;
        task.id_target = -1;

        queue_tasks.post(std::move(task));
    }

    // apply context-shift if needed
    // TODO: simplify and improve
    context_shift();
    pxa_us_mark("post_ctxshift");

    // start populating the batch for this iteration
    common_batch_clear(batch);

    // PXA_PROMPT_INTERLEAVE_v1 (PXA_KERNEL_FIX_20260611): the PXA_LLAMA_MTP_NP4_FIX prompt/decode
    // separation (never co-batch prompt tokens with generation tokens on a recurrent/hybrid model)
    // previously STARVED pending prompts: add_sampled_tokens() refills the batch with generation
    // tokens EVERY tick, so `pxa_defer_prompt` deferred a new request's prefill until EVERY
    // in-flight generation fully completed. Result: np=4 degenerated into a strictly serial server
    // under staggered arrivals (head-of-line blocking of minutes behind -n 4096 runaways) -- the
    // production "HTTP-ingress wedge / 47s hang / empty body" class, and the reason multi-slot
    // co-decode (and every np>1 code path) never actually executed in production.
    // FIX: keep the uniform-batch invariant (a batch is either pure-prompt(one slot) or
    // pure-generation) but ALTERNATE between the two kinds when both are pending: if a prompt is
    // waiting and the previous tick was not a prompt tick, dedicate this tick to ONE pending
    // prompt (skip add_sampled_tokens; the generating slots simply do not advance this tick and
    // resume next tick). The delta-net never sees a ragged batch, prompts are admitted within one
    // generation step instead of after the full drain, and true concurrent decode is restored.
    bool pxa_pending_prompt = false;
    for (auto & pxa_s : slots) {
        if (pxa_s.state == SLOT_STATE_IDLE && pxa_s.command == SLOT_COMMAND_LOAD_PROMPT) {
            pxa_pending_prompt = true;
            break;
        }
    }
    const bool pxa_recurrent_model = llama_model_has_recurrent(model);
    // Ops kill-switch: PXA_PROMPT_INTERLEAVE=0 reverts to the old serialize-behind-decode behavior
    // (no rebuild needed) in case the restored co-decode misbehaves in production.
    static const bool pxa_interleave_on = !(getenv("PXA_PROMPT_INTERLEAVE") && atoi(getenv("PXA_PROMPT_INTERLEAVE")) == 0);
    // PXA_INTERLEAVE_DECODE_MS_v1: hold the next prompt chunk while the decode window is open and
    // some slot still has a token to emit. With the value at 0 this is exactly the previous strict
    // alternation. The window closes early when no slot has decode work, so a lone deep prefill
    // still runs back-to-back chunks and pays nothing.
    bool pxa_decode_pending = false;
    for (auto & pxa_s : slots) {
        if (pxa_s.state == SLOT_STATE_PROCESSING) { pxa_decode_pending = true; break; }
    }
    const bool pxa_hold_for_decode = pxa_interleave_decode_ms() > 0
                                   ? (pxa_decode_pending && ggml_time_us() < pxa_decode_window_until_us)
                                   : pxa_prev_tick_was_prompt;
    const bool pxa_prompt_tick = pxa_interleave_on && pxa_recurrent_model && pxa_pending_prompt && !pxa_hold_for_decode;

    // first, add sampled tokens from any ongoing sequences (skipped on a dedicated prompt tick)
    if (!pxa_prompt_tick) {
        add_sampled_tokens(); // Prepare batch for inference
    }

    // process in chunks of params.n_batch
    int32_t n_batch = llama_n_batch(ctx);
    int32_t n_ubatch = llama_n_ubatch(ctx);

    // track if this is an embedding or non-embedding batch
    // if we've added sampled tokens above, we are in non-embedding mode
    // -1: none, 0: non-embedding, 1: embedding
    int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

    // PXA_LLAMA_MTP_NP4_FIX: on a recurrent/hybrid model, do NOT co-batch pending PROMPT tokens with
    // already-queued GENERATION/verify tokens in the same decode. A prompt slot adds many tokens while a
    // generating slot adds 1+draft, producing a RAGGED multi-seq u-batch that forces the qwen3next
    // delta-net onto the corrupting split-per-sequence fallback. Defer the prompt to the next iteration so
    // the generation batch stays uniform (the pure-prompt batch then runs alone next tick). Non-recurrent
    // models are unaffected by that bug, so keep their continuous-batching behavior.
    const bool pxa_defer_prompt = pxa_recurrent_model && batch.n_tokens > 0;
    if (!pxa_defer_prompt) {
        // next, batch any pending prompts without exceeding n_batch
        batch_pending_prompt(n_ubatch, n_batch, batch_type); // Prepare batch for prompt process
    }

    // PXA_PROMPT_INTERLEAVE_v1: record whether this tick actually ran as a prompt tick (prompt
    // tokens batched while generation was held back) so the next tick goes back to generation.
    // If the prompt tick batched nothing (e.g. the pending slot could not proceed), fall through
    // as a normal tick so the flip-flop cannot livelock.
    pxa_prev_tick_was_prompt = pxa_prompt_tick && batch.n_tokens > 0;

    if (batch.n_tokens == 0) {
        LOG_VERBOSE("no tokens to decode", {});
        return;
    }

    LOG_VERBOSE("decoding batch", {
        {"n_tokens", batch.n_tokens},
        });

    // make sure we're in the right embedding mode
    llama_set_embeddings(ctx, batch_type == 1);

    if (llama_model_has_recurrent(model)) {
        const int ckpt_mode = params_base.speculative.recurrent_ckpt_mode;
        const int64_t pxa_sptr_ck0 = pxa_sptr.on ? ggml_time_us() : 0; // PXA_SPEC_TRACE

        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch_dft.empty()) {
                continue;
            }
            if (save_speculative_checkpoint(slot, model, ctx, ckpt_mode)) {
                const char * mode_name = slot.spec_ckpt.per_step_enabled ? "per-step" : "shadow/cpu";
                SLT_DBG(slot, "spec checkpoint saved (mode=%s), n_past_pre_spec=%d\n",
                    mode_name, slot.spec_ckpt.n_past);
            } else {
                SLT_WRN(slot, "%s", "failed to save spec checkpoint\n");
            }
        }
        if (pxa_sptr.on) pxa_sptr_add(PXA_SPTR_CKPT, pxa_sptr_ck0); // PXA_SPEC_TRACE
    }

    // process the created batch of tokens
    pxa_us_mark("pre_pbt");
    process_batch_tokens(n_batch); // Decode with batch
    pxa_us_mark("us_exit");

    // PXA_INTERLEAVE_DECODE_MS_v1: open the decode window once the prompt chunk has actually been
    // DECODED, not when its batch was built. A prompt chunk gets dramatically more expensive with
    // depth -- 2048 tokens costs 3-5 s early in a prefill and about 19.5 s at 140k on the 4-card
    // seat -- so a window opened before the decode had already expired by the time the next tick
    // began, and the co-resident slot fell back to one token per chunk (measured 20.9 s per token
    // at the tail of a 145k prefill, against 0.06 s per token when the window survived).
    if (pxa_prev_tick_was_prompt && pxa_interleave_decode_ms() > 0) {
        pxa_decode_window_until_us = ggml_time_us() + 1000ll * (int64_t) pxa_interleave_decode_ms();
    }
    pxa_sptr_tick_end(); // PXA_SPEC_TRACE

    LOG_VERBOSE("run slots completed", {});
}

json server_context::model_meta() const {
    return json{
        {"vocab_type",  llama_vocab_type(llama_model_get_vocab(model))},
        {"n_vocab",     llama_n_vocab(model)},
        {"n_ctx_train", llama_n_ctx_train(model)},
        {"n_embd",      llama_model_n_embd(model)},
        {"n_params",    llama_model_n_params(model)},
        {"size",        llama_model_size(model)},
    };
}
