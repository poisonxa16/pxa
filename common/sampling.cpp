#define LLAMA_API_INTERNAL
#include "sampling.h"
#include "llama-vocab.h"
#include "common.h"
#include "reasoning-budget.cpp"
#include "ggml.h"           // ggml_pxa_config_level (the PXA master config level)

#include <cstdlib>
#include <limits>
#include <random>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

struct llama_sampler_adaptive_p * llama_clone_adaptive_p(const struct llama_sampler_adaptive_p * adapt_p_ctx);
void llama_free_adaptive_p(struct llama_sampler_adaptive_p * adapt_p_ctx);

// PXQ repetition guard (2026-07-23; widened to all PXQ tiers 2026-07-24): low-bit tiers
// measurably loop on open-ended prompts (coding answers stay clean — measured on a 122B
// PXQU24 mixed file). Originally gated on PXQ1 (248) alone, which made it dead code for
// PXQ4 (252) — the tier the shipped PXA models actually use.
// HONEST SCOPE: penalties were measured to make a runaway loop NON-VERBATIM, not to end
// it, and the underlying prose-degeneration attractor is a property of a specific MERGE,
// not of the codec. This is a runaway CAP, not the fix. The measured primary lever is
// reasoning-off; `--reasoning-budget 0` was measured INERT (never arms for this template).
// When the loaded model contains PXQ tensors, fill repetition-defense DEFAULTS for any
// knob the user left at its default — explicit user settings ALWAYS win (only untouched
// values are moved). Level logic mirrors PXA_SPEC_RELAXED below / pxa_config_level() in
// ggml/src/ggml-cuda/pxa/pxa-enhance.cuh (non-CUDA TU, so the logic is mirrored): the guard is
// an ENHANCE-class lever; REFERENCE/DEFAULT leave sampling untouched.
//   PXA_PXQ1_REP_GUARD=0  force off at any level
//   PXA_PXQ1_REP_GUARD=1  force on at any level
static void pxa_pxq1_apply_rep_guard(common_params_sampling & sp) {
    static const int mode = [] {   // -1 = forced off, 0 = level default, 1 = forced on
        // PXA_REP_GUARD is the current name; PXA_PXQ1_REP_GUARD kept for back-compat.
        const char * e = std::getenv("PXA_REP_GUARD");
        if (!e) {
            e = std::getenv("PXA_PXQ1_REP_GUARD");
        }
        return e ? (atoi(e) != 0 ? 1 : -1) : 0;
    }();
    if (mode == -1 || !llama_pxa_pxq_content()) {
        return;
    }
    if (mode == 0) {
        // PXA_REP_GUARD_SCOPE fix (2026-07-30): the ENHANCE auto-arm is PXQ1-CONTENT-ONLY
        // again. The 2026-07-24 broadening to any-PXQ was motivated by the Fusion4 prose
        // attractor — which was subsequently root-caused to PXA_PXQ4_2D_SPLIT non-bit-
        // exactness and FIXED by PXQ_CANON_v1 (H1 verified gone at defaults, 2026-07-28/30).
        // Left broad, the guard MEASURABLY CORRUPTS exact answers on high-quality tiers:
        // Laguna PXQ4-core, 4xP100+1080Ti, temp-0/top_k-1 chat probe "4183*391" — DEFAULT
        // level answers 1,635,553 (correct, x3); ENHANCE answered 1,635,593 (x3): the filled
        // repeat/DRY penalties suppress the legitimate repeated "5". Both-direction proof:
        // ENHANCE + PXA_REP_GUARD=0 -> correct x3; DEFAULT + PXA_REP_GUARD=1 -> wrong x3.
        // That flip was chased for a night as an "sm_61 numerics bug" — it was this lever.
        // PXQ1-bearing files (incl. PXQU12/24 mixed maps) keep the guard: the 1-bit tier's
        // looping is measured and real. PXA_REP_GUARD=1 still forces any-PXQ arming.
        if (!llama_pxa_pxq1_content()) {
            return;
        }
        if (ggml_pxa_config_level() < 2) {
            return;   // arms at ENHANCE only (the default level; PXA_ENHANCE=0/PXA_REFERENCE=1 opt out)
        }
    }
    bool applied = false;
    if (sp.penalty_repeat == 1.00f) { sp.penalty_repeat = 1.15f; applied = true; }
    if (sp.penalty_last_n == 64)    { sp.penalty_last_n  = 256;  applied = true; }
    if (sp.dry_multiplier == 0.0f)  { sp.dry_multiplier  = 0.8f; applied = true; }   // DRY: base/allowed-length keep their defaults
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "PXQ content detected: repetition guard ON [%s]%s (repeat_penalty=%.2f repeat_last_n=%d dry_multiplier=%.2f; runaway CAP, not a cure; PXA_REP_GUARD=0 disables)\n",
                mode == 1 ? "FORCED" : "ENHANCE", applied ? "" : " — all knobs user-set, nothing filled",
                sp.penalty_repeat, sp.penalty_last_n, sp.dry_multiplier);
    }
}

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params_in) {
    common_params_sampling params = params_in;   // mutable copy — the PXQ1 guard may fill defaults
    pxa_pxq1_apply_rep_guard(params);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    struct common_sampler * result = new common_sampler();

    result->params  = params;
    result->grammar = nullptr;
    result->rbudget = nullptr;

    struct llama_grammar* grmr = nullptr;
    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
        result->grammar = grmr;
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    }
    else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            {
                const auto & word = trigger.value;
                trigger_patterns.push_back(regex_escape(word));
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            {
                trigger_patterns.push_back(trigger.value);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            {
                const auto & pattern = trigger.value;
                std::string anchored = "^$";
                if (!pattern.empty()) {
                    anchored = (pattern.front() != '^' ? "^" : "")
                        + pattern
                        + (pattern.back() != '$' ? "$" : "");
                }
                trigger_patterns.push_back(anchored);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            {
                const auto token = trigger.token;
                trigger_tokens.push_back(token);
                break;
            }
            default:
                GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
            grmr = params.grammar_lazy
                ? llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                    trigger_patterns_c.data(), trigger_patterns_c.size(),
                    trigger_tokens.data(), trigger_tokens.size())
                : llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
            if (grmr) {
                result->prev.resize(params.n_prev);

                result->grammar = grmr;
            }
        }
        result->n_valid = 0;
        result->grammar_str = grammar_str;
        result->grammar_root = "root";
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_grammar_accept_impl(*grmr, vocab, nullptr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        }
        catch (std::exception & e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0)) {
        result->rbudget = common_reasoning_budget_init(
            vocab,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            common_reasoning_budget_accept(result->rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    llama_sampling_set_rng_seed(result, params.seed);
    for (const auto& cnstr : params.samplers_sequence)
    {
        switch (cnstr)
        {
            case llama_sampler_type::DRY:
            {
                std::vector<const char*> c_breakers;
                c_breakers.reserve(params.dry_sequence_breakers.size());
                for (const auto& str : params.dry_sequence_breakers)
                {
                    c_breakers.push_back(str.c_str());
                }
                result->smpl=llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size());

                break;
            }
            case llama_sampler_type::ADAPTIVE_P:
            {
                if (params.adaptive_target >= 0.0f) {
                    GGML_ASSERT(vocab);
                    auto n_vocab = llama_vocab_n_tokens(vocab);
                    result->adapt_p_ctx = llama_init_adaptive_p(n_vocab, params.adaptive_target, params.adaptive_decay, params.adaptive_updt_w_cur, result->rng());
                }
                break;
            }
            default:
                break;
        }
    }

    result->elb_idx = 0;
    result->elb_search_pos = 0;

    return result;
}

void common_sampler_free(struct common_sampler * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->grammar) {
        llama_grammar_free(ctx->grammar);
    }
    if (ctx->smpl)
        llama_sampler_dry_free(ctx->smpl);
    if (ctx->adapt_p_ctx)
        llama_free_adaptive_p(ctx->adapt_p_ctx);
    if (ctx->rbudget)
        common_reasoning_budget_free(ctx->rbudget);
    delete ctx;
}

static void llama_grammar_reset(common_sampler * ctx) {
    if (!ctx->grammar) {
        return;
    }
    std::vector<const char*>  trigger_patterns_c;
    trigger_patterns_c.reserve(ctx->grammar->trigger_patterns.size());
    for (auto& trigger_pattern : ctx->grammar->trigger_patterns) {
        trigger_patterns_c.push_back(trigger_pattern.pattern.c_str());
    }

    auto* grammar_new = llama_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str(),
        ctx->grammar->lazy, trigger_patterns_c.data(), trigger_patterns_c.size(),
        ctx->grammar->trigger_tokens.data(), ctx->grammar->trigger_tokens.size());

    llama_grammar_free_impl(ctx->grammar);
    ctx->grammar = grammar_new;
}

void common_sampler_reset(common_sampler * ctx) {
    // llama_grammar_reset(ctx);
    ctx->prev.clear();
    llama_sampler_dry_reset(ctx->smpl);
}

void common_sampler_review(common_sampler * ctx, const size_t n_unsent, const bool rewind_status) {
    // add stateful samplers here
    if (ctx->adapt_p_ctx != nullptr) {
        llama_review_adaptive_p(ctx->adapt_p_ctx, n_unsent, rewind_status);
    }
}

void llama_sampling_set_rng_seed(struct common_sampler * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = std::random_device{}();
    }
    ctx->rng.seed(seed);
}

void common_sampler_clone(common_sampler * src, common_sampler * dst) {
    dst->params = src->params;
    dst->mirostat_mu = src->mirostat_mu;
    dst->n_valid = src->n_valid;
    dst->rng = src->rng;
    dst->server_biases = src->server_biases;

    if (dst->grammar) {
        llama_grammar_free(dst->grammar);
        dst->grammar = nullptr;
    }

    if (src->grammar) {
        dst->grammar_root = src->grammar_root;
        dst->grammar_str = src->grammar_str;
        dst->grammar = llama_grammar_copy(src->grammar);
    }

    dst->prev = src->prev;
    if (dst->smpl) {
        llama_sampler_dry_free(dst->smpl);
        dst->smpl = nullptr;
    }
    if (src->smpl) {
        dst->smpl = llama_sampler_dry_clone(src->smpl);
    }

    if (dst->adapt_p_ctx) {
        llama_free_adaptive_p(dst->adapt_p_ctx);
        dst->adapt_p_ctx = nullptr;
    }
    if (src->adapt_p_ctx) {
        dst->adapt_p_ctx = llama_clone_adaptive_p(src->adapt_p_ctx);
    }

    if (dst->rbudget) {
        common_reasoning_budget_free(dst->rbudget);
        dst->rbudget = nullptr;
    }
    if (src->rbudget) {
        dst->rbudget = common_reasoning_budget_clone(src->rbudget);
    }
}

llama_token llama_sampling_last(common_sampler * ctx) {
    return ctx->prev.back();
}

std::string llama_sampling_prev_str(common_sampler * ctx_sampling, llama_context * ctx_main, int n) {
    const int size = ctx_sampling->prev.size();

    n = std::min(n, size);

    std::string result;

    for (int i = size - n; i < size; i++) {
        result += common_token_to_piece(ctx_main, ctx_sampling->prev[i]);
    }

    return result;
}

std::string llama_sampling_print(const common_params_sampling & params) {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f\n"
            "\txtc_probability = %.3f, xtc_threshold = %.3f, top_n_sigma = %.3f\n"
            "\tadaptive_target = %.2f, adaptive_decay = %.2f",
            params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present,
            params.top_k, params.tfs_z, params.top_p, params.min_p, params.typical_p, params.temp,
            params.mirostat, params.mirostat_eta, params.mirostat_tau,
            params.xtc_probability, params.xtc_threshold, params.top_n_sigma,
            params.adaptive_target, params.adaptive_decay);

    return std::string(result);
}

std::string llama_sampling_order_print(const common_params_sampling & params) {
    std::string result = "CFG -> Penalties ";
    if (params.mirostat == 0) {
        for (auto sampler_type : params.samplers_sequence) {
            const auto sampler_type_name = llama_sampling_type_to_str(sampler_type);
            if (!sampler_type_name.empty()) {
                result += "-> " + sampler_type_name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case llama_sampler_type::DRY:         return "dry";
        case llama_sampler_type::TOP_K:       return "top_k";
        case llama_sampler_type::TFS_Z:       return "tfs_z";
        case llama_sampler_type::TYPICAL_P:   return "typical_p";
        case llama_sampler_type::TOP_P:       return "top_p";
        case llama_sampler_type::MIN_P:       return "min_p";
        case llama_sampler_type::TEMPERATURE: return "temperature";
        case llama_sampler_type::XTC        : return "xtc";
        case llama_sampler_type::TOP_N_SIGMA: return "top_n_sigma";
        case llama_sampler_type::ADAPTIVE_P : return "adaptive_p";
        default : return "";
    }
}

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        {"dry",         llama_sampler_type::DRY},
        {"top_k",       llama_sampler_type::TOP_K},
        {"top_p",       llama_sampler_type::TOP_P},
        {"typical_p",   llama_sampler_type::TYPICAL_P},
        {"min_p",       llama_sampler_type::MIN_P},
        {"tfs_z",       llama_sampler_type::TFS_Z},
        {"xtc",         llama_sampler_type::XTC},
        {"top_n_sigma", llama_sampler_type::TOP_N_SIGMA},
        {"temperature", llama_sampler_type::TEMPERATURE},
        {"adaptive_p",  llama_sampler_type::ADAPTIVE_P},
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        {"dry",         llama_sampler_type::DRY},
        {"top-k",       llama_sampler_type::TOP_K},
        {"top-p",       llama_sampler_type::TOP_P},
        {"nucleus",     llama_sampler_type::TOP_P},
        {"typical-p",   llama_sampler_type::TYPICAL_P},
        {"typical",     llama_sampler_type::TYPICAL_P},
        {"min-p",       llama_sampler_type::MIN_P},
        {"tfs-z",       llama_sampler_type::TFS_Z},
        {"tfs",         llama_sampler_type::TFS_Z},
        {"xtc",         llama_sampler_type::XTC},
        {"top-n-sigma", llama_sampler_type::TOP_N_SIGMA},
        {"temp",        llama_sampler_type::TEMPERATURE},
        {"adaptive-p",  llama_sampler_type::ADAPTIVE_P},
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());
    for (const auto & name : names)
    {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end())
        {
            sampler_types.push_back(sampler_item->second);
        }
        else
        {
            if (allow_alt_names)
            {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end())
                {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }
    return sampler_types;
}

std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        {'d', llama_sampler_type::DRY},
        {'k', llama_sampler_type::TOP_K},
        {'p', llama_sampler_type::TOP_P},
        {'y', llama_sampler_type::TYPICAL_P},
        {'m', llama_sampler_type::MIN_P},
        {'f', llama_sampler_type::TFS_Z},
        {'x', llama_sampler_type::XTC},
        {'n', llama_sampler_type::TOP_N_SIGMA},
        {'t', llama_sampler_type::TEMPERATURE},
        {'w', llama_sampler_type::ADAPTIVE_P},
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}

// no reasons to expose this function in header
static void sampler_queue(
    struct llama_context* ctx_main,
    const common_params_sampling& params,
    common_sampler * ctx_sampling,
    llama_token_data_array& cur_p,
    size_t   min_keep) {
    const float         temp = params.temp;
    const float         dynatemp_range = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k = params.top_k;
    const float         top_p = params.top_p;
    const float         min_p = params.min_p;
    const float         tfs_z = params.tfs_z;
    const float         typical_p = params.typical_p;
    const float         xtc_probability = params.xtc_probability;
    const float         xtc_threshold = params.xtc_threshold;
    const float         top_n_sigma = params.top_n_sigma;

    const std::vector<llama_sampler_type> & samplers_sequence = params.samplers_sequence;
    bool use_adaptive_p = false; // see below
    for (auto sampler_type : samplers_sequence) {
        switch (sampler_type) {
            case llama_sampler_type::DRY        : llama_sample_dry      (ctx_main, ctx_sampling->smpl, &cur_p); break;
            case llama_sampler_type::TOP_K      : llama_sample_top_k    (ctx_main, &cur_p, top_k,     min_keep); break;
            case llama_sampler_type::TFS_Z      : llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P  : llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P      : llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P      : llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::XTC        : llama_sample_xtc      (ctx_main, &cur_p, xtc_probability, xtc_threshold, min_keep); break;
            case llama_sampler_type::TOP_N_SIGMA: llama_sample_top_n_sigma(ctx_main, &cur_p, top_n_sigma); break;
            case llama_sampler_type::DIST       : llama_sample_dist  (ctx_main, &cur_p); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sample_temp(ctx_main, &cur_p, temp);
                }
                break;
            case llama_sampler_type::ADAPTIVE_P:  use_adaptive_p = ctx_sampling->adapt_p_ctx != nullptr; break;
            default : break;
        }

    }
    if (use_adaptive_p) {
        // adaptive p should be put to the last, so we ignore the order in the sampler
        llama_sample_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
    }
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grammar) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

// ---- PXA_TOPK_RAW (house custom, DEFAULT OFF) ----------------------------------------------
// Bounded top-K sampling straight off the raw `float * logits` row, for the resolved sampler chain
// that a decode seat actually runs:
//     [no-op samplers]* TOP_K [TOP_P | MIN_P | TEMPERATURE | DIST | no-op samplers]*  then rng pick
// (DIST in this fork is an unnormalized softmax that only fills the p fields, size unchanged)
// or  temp == 0 (greedy argmax), with optional logit_bias / repetition penalties / server biases.
// The default path materializes n_vocab llama_token_data (3 MB at a 248k vocab) and partial_sorts
// them per token; this keeps a W <= 128 sorted window (W = min(max(top_k, min_keep), n_vocab)) and
// does one threshold compare per vocab entry (AVX2-prefiltered), then runs the SAME llama_sample_*
// calls over the window, so float arithmetic, min_keep handling and rng draws are identical.
//
// Exactness notes (the plan's two failure modes, both handled):
//   * penalties are applied IN PLACE on `logits` for the <= penalty_last_n penalized ids BEFORE the
//     scan (so the window is the true post-penalty top-W, including a penalty that RAISES a negative
//     logit under repeat_penalty < 1), and restored afterwards so the row reads exactly like the
//     default path left it (logit_bias applied, penalties not) for anything that inspects it later
//     (n_probs, spec verify);
//   * server_biases (an n_vocab vector) are folded into the scan on the fly and never written back;
//     penalties + server_biases together are gated OFF (the default path penalizes logit+bias and
//     the nl-token exception then drops the bias -- not reproducible without a second pass);
//   * the gate is on the RESOLVED per-request chain: DRY/TFS/typical/XTC/top-n-sigma must be at
//     their no-op parameters, anything before TOP_K in the sequence must be a no-op, no grammar, no
//     reasoning-budget FORCING, no mirostat / adaptive-p / dynatemp / expiring-logit-bias, temp >= 0;
//   * ties: partial_sort's order among equal logits is unspecified, this window keeps the earliest
//     id first. Same distribution; the drawn token can differ on an exact fp32 tie. Shadow mode
//     classifies those as "tie" (both ids carry the identical post-chain logit), not mismatches.
//
// PXA_TOPK_RAW=1  fast path live.
// PXA_TOPK_RAW=2  shadow-compare: the default path is authoritative; the fast path runs first on a
//                 saved copy of (logits, rng), the post-chain windows and the drawn ids are compared,
//                 a genuine mismatch is logged once and the fast path disables itself.
// PXA_TOPK_RAW=3  shadow-compare that aborts on a genuine mismatch (harness use).
// A stats line (compared / tie / mismatch counts) goes to stderr every 1000 compares in modes 2/3.
#include <unordered_map>
#include <cmath>

static int pxa_topk_raw_mode() {
    // DEFAULT 2026-09-03: mode 1 (fast path live) at ENHANCE (the default level), OFF at
    // DEFAULT/REFERENCE. The fast path is the same argmax/top-k over the raw logits that the
    // default chain computes, on the CPU -- there is no architecture in it, and the previous
    // "Volta-only process" gate left it off on the P100 seat it was measured on. It self-disables
    // (g_pxa_topk_raw_disabled) on any sampler configuration it does not cover.
    // PXA_TOPK_RAW still overrides in both directions, and =2 keeps the shadow-compare mode.
    static const int v = [] {
        const char * e = getenv("PXA_TOPK_RAW");
        const int m = e ? atoi(e) : (ggml_pxa_config_level() >= 2 ? 1 : 0);
        return m < 0 ? 0 : m;
    }();
    return v;
}

static bool g_pxa_topk_raw_disabled = false;
static int64_t g_pxa_topk_raw_stats[5] = {0, 0, 0, 0, 0};   // taken, compared, tie, window_mismatch, mismatch

void common_sampler_pxa_topk_raw_stats(int64_t out[5]) {
    for (int i = 0; i < 5; ++i) out[i] = g_pxa_topk_raw_stats[i];
}

static bool pxa_topk_raw_penalties_active(const common_sampler * cs) {
    const auto & p = cs->params;
    const int32_t penalty_last_n = p.penalty_last_n < 0 ? p.n_prev : p.penalty_last_n;
    const auto & penalty_tokens  = p.use_penalty_prompt_tokens ? p.penalty_prompt_tokens : cs->prev;
    const int used = std::min((int) penalty_tokens.size(), penalty_last_n);
    return used > 0 && !(p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f);
}

// Returns the window width W > 0 when the resolved chain is the safe subset, 0 otherwise.
static int pxa_topk_raw_window(const common_sampler * cs, const llama_context * ctx_cfg, bool grammar_first, int n_vocab) {
    const auto & p = cs->params;
    if (grammar_first || ctx_cfg != nullptr || cs->grammar != nullptr) return 0;
    if (cs->rbudget && common_reasoning_budget_get_state(cs->rbudget) == REASONING_BUDGET_FORCING) return 0;
    if (cs->elb_states.size() > cs->elb_idx) return 0;               // expiring logit bias mutates its own state
    if (p.mirostat != 0) return 0;
    if (cs->adapt_p_ctx != nullptr) return 0;                          // adaptive-p branch / ADAPTIVE_P queue entry
    if (!(p.temp >= 0.0f)) return 0;                                   // temp < 0 (greedy with probs) and NaN
    if (p.dynatemp_range > 0.0f) return 0;
    const bool pen_active  = pxa_topk_raw_penalties_active(cs);
    const bool bias_active = cs->server_biases != nullptr && cs->server_biases->size() == (size_t) n_vocab;
    if (pen_active && bias_active) return 0;
    if (p.temp == 0.0f) {
        // greedy: argmax over the penalized row; cur_p differs in content from the default path
        // (window vs the whole unsorted vocab), so only when nothing reads it
        return p.n_probs > 0 ? 0 : 1;
    }
    bool seen_top_k = false;
    for (const auto st : p.samplers_sequence) {
        switch (st) {
            case llama_sampler_type::DRY:
                if (!(p.dry_multiplier == 0.0f || p.dry_base < 1.0f || p.dry_penalty_last_n == 0)) return 0;
                break;
            case llama_sampler_type::TOP_K:
                if (p.top_k <= 0) return 0;
                seen_top_k = true;
                break;
            case llama_sampler_type::TFS_Z:       if (p.tfs_z < 1.0f) return 0; break;
            case llama_sampler_type::TYPICAL_P:   if (p.typical_p < 1.0f) return 0; break;
            case llama_sampler_type::XTC:         if (!(p.xtc_probability <= 0.0f || p.xtc_threshold > 0.5f)) return 0; break;
            case llama_sampler_type::TOP_N_SIGMA: if (p.top_n_sigma > 0.0f) return 0; break;
            case llama_sampler_type::TOP_P:       if (!seen_top_k && p.top_p < 1.0f) return 0; break;
            case llama_sampler_type::MIN_P:       if (!seen_top_k && p.min_p > 0.0f) return 0; break;
            case llama_sampler_type::TEMPERATURE: break;               // per-element scale, commutes with the top-W selection
            case llama_sampler_type::ADAPTIVE_P:  if (cs->adapt_p_ctx != nullptr) return 0; break;
            case llama_sampler_type::DIST:        break;               // unnormalized softmax p-fill: p_i = exp(l_i - max), per element, size unchanged
            default: return 0;                                         // anything new
        }
    }
    if (!seen_top_k) return 0;
    const int W = std::min(std::max(p.top_k, std::max(1, p.min_keep)), n_vocab);
    return W <= 128 ? W : 0;
}

// Sorted-descending window insert; among equal values the earlier id stays first, and an element
// equal to the current minimum of a full window does not enter.
struct pxa_topk_window {
    int   W   = 0;
    int   cnt = 0;
    float thr = -INFINITY;   // = wl[W-1] once full
    float wl[128];
    int   wid[128];

    inline void push(float x, int id) {
        if (cnt == W) {
            if (!(x > thr)) return;
            int j = W - 1;
            while (j > 0 && wl[j-1] < x) { wl[j] = wl[j-1]; wid[j] = wid[j-1]; --j; }
            wl[j] = x; wid[j] = id;
            thr = wl[W-1];
            return;
        }
        int j = cnt;
        while (j > 0 && wl[j-1] < x) { wl[j] = wl[j-1]; wid[j] = wid[j-1]; --j; }
        wl[j] = x; wid[j] = id;
        if (++cnt == W) thr = wl[W-1];
    }
};

static void pxa_topk_scan_scalar(const float * logits, const float * bias, int n, int i0, pxa_topk_window & w) {
    if (bias) {
        for (int i = i0; i < n; ++i) w.push(logits[i] + bias[i], i);
    } else {
        for (int i = i0; i < n; ++i) w.push(logits[i], i);
    }
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
static void pxa_topk_scan_avx2(const float * logits, const float * bias, int n, pxa_topk_window & w) {
    // fill the window from the head, then threshold-prefilter 8 lanes at a time
    int i = 0;
    for (; i < n && w.cnt < w.W; ++i) w.push(bias ? logits[i] + bias[i] : logits[i], i);
    alignas(32) float xb[8];
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(logits + i);
        if (bias) x = _mm256_add_ps(x, _mm256_loadu_ps(bias + i));
        const int m = _mm256_movemask_ps(_mm256_cmp_ps(x, _mm256_set1_ps(w.thr), _CMP_GT_OQ));
        if (m == 0) continue;
        _mm256_store_ps(xb, x);
        for (int b = 0; b < 8; ++b) {
            if (m & (1 << b)) w.push(xb[b], i + b);   // push re-checks against the (possibly raised) threshold
        }
    }
    pxa_topk_scan_scalar(logits, bias, n, i, w);
}
#endif

static void pxa_topk_scan(const float * logits, const float * bias, int n, pxa_topk_window & w) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    static const bool has_avx2 = __builtin_cpu_supports("avx2");
    if (has_avx2 && n >= 8) { pxa_topk_scan_avx2(logits, bias, n, w); return; }
#endif
    pxa_topk_scan_scalar(logits, bias, n, 0, w);
}

// The fast path proper. Leaves ctx_sampling->cur_p = the post-chain window, the rng advanced by
// exactly the draw the default path makes, and `logits` as the default path leaves it.
static llama_token pxa_topk_raw_sample(struct common_sampler * cs, struct llama_context * ctx_main, const int idx, const int W) {
    const auto & params = cs->params;
    const llama_model * model = llama_get_model(ctx_main);
    const int n_vocab = llama_n_vocab(model);
    float * logits = llama_get_logits_ith(ctx_main, idx);

    // 1. logit_bias in place, exactly as llama_sampling_prepare_impl does it (and leaves it)
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); ++it) {
        logits[it->first] += it->second;
    }
    const float * bias = (cs->server_biases != nullptr && cs->server_biases->size() == (size_t) n_vocab) ? cs->server_biases->data() : nullptr;

    // 2. repetition penalties in place on the penalized ids, originals kept for the restore
    struct saved { llama_token id; float v; };
    std::vector<saved> restore;
    if (pxa_topk_raw_penalties_active(cs)) {
        const int32_t penalty_last_n = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
        const auto & penalty_tokens  = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : cs->prev;
        const int used = std::min((int) penalty_tokens.size(), penalty_last_n);
        const llama_token nl_token = llama_token_nl(model);
        const float nl_logit = nl_token != LLAMA_TOKEN_NULL ? logits[nl_token] : 0.0f;
        std::unordered_map<llama_token, int> token_count;
        const llama_token * last = penalty_tokens.data() + penalty_tokens.size() - used;
        for (int i = 0; i < used; ++i) token_count[last[i]]++;
        restore.reserve(token_count.size());
        for (const auto & kv : token_count) {
            const llama_token id = kv.first;
            if (id < 0 || id >= n_vocab) continue;   // the default path's full scan never matches these either
            float & l = logits[id];
            restore.push_back({id, l});
            if (l <= 0) { l *= params.penalty_repeat; } else { l /= params.penalty_repeat; }
            l -= float(kv.second) * params.penalty_freq + float(kv.second > 0) * params.penalty_present;
        }
        if (!params.penalize_nl && nl_token != LLAMA_TOKEN_NULL) {
            logits[nl_token] = nl_logit;
        }
    }

    // 3. bounded top-W scan
    pxa_topk_window w;
    w.W = W;
    pxa_topk_scan(logits, bias, n_vocab, w);

    // 4. restore the row
    for (const auto & r : restore) logits[r.id] = r.v;

    // 5. the window becomes the candidate array (sorted descending)
    auto & cur = cs->cur;
    cur.resize(w.cnt);
    for (int i = 0; i < w.cnt; ++i) cur[i] = llama_token_data{w.wid[i], w.wl[i], 0.0f};
    cs->cur_p = { cur.data(), (size_t) w.cnt, 0, true };
    llama_token_data_array & cur_p = cs->cur_p;

    if (params.temp == 0.0f) {
        // greedy, no probs: first maximal id, softfail flag set exactly like the default path
        return llama_sample_token_greedy(ctx_main, &cur_p);
    }

    // 6. the rest of the chain, in the request's order, over the window
    const size_t min_keep = std::max(1, params.min_keep);
    for (const auto st : params.samplers_sequence) {
        switch (st) {
            case llama_sampler_type::TOP_P:       llama_sample_top_p(ctx_main, &cur_p, params.top_p, min_keep); break;
            case llama_sampler_type::MIN_P:       llama_sample_min_p(ctx_main, &cur_p, params.min_p, min_keep); break;
            case llama_sampler_type::TEMPERATURE: llama_sample_temp (ctx_main, &cur_p, params.temp); break;
            case llama_sampler_type::DIST:        llama_sample_dist (ctx_main, &cur_p); break;
            default: break;   // TOP_K is the window itself; everything else was verified to be a no-op
        }
    }
    return llama_sample_token_with_rng(ctx_main, &cur_p, cs->rng);
}

static llama_token llama_sampling_sample_impl_default(struct common_sampler *, struct llama_context *, struct llama_context *, int, bool);

static llama_token pxa_topk_raw_shadow(struct common_sampler * cs, struct llama_context * ctx_main, struct llama_context * ctx_cfg, const int idx, bool grammar_first, const int W, const int mode) {
    int64_t & n_cmp = g_pxa_topk_raw_stats[1]; int64_t & n_tie = g_pxa_topk_raw_stats[2]; int64_t & n_win_bad = g_pxa_topk_raw_stats[3]; int64_t & n_bad = g_pxa_topk_raw_stats[4];
    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));
    float * logits = llama_get_logits_ith(ctx_main, idx);
    const std::vector<float> logits_copy(logits, logits + n_vocab);
    const std::mt19937 rng_copy = cs->rng;

    const llama_token id_new = pxa_topk_raw_sample(cs, ctx_main, idx, W);
    const std::vector<llama_token_data> win_new(cs->cur_p.data, cs->cur_p.data + cs->cur_p.size);

    std::copy(logits_copy.begin(), logits_copy.end(), logits);
    cs->rng = rng_copy;
    const llama_token id_old = llama_sampling_sample_impl_default(cs, ctx_main, ctx_cfg, idx, grammar_first);
    const llama_token_data_array & cur_old = cs->cur_p;

    ++n_cmp;
    bool ok = true;
    std::string why;
    if (cs->params.temp != 0.0f) {
        // post-chain windows: same length, same logit sequence (both sorted descending)
        if (cur_old.size != win_new.size()) {
            ok = false; why = "window size " + std::to_string(cur_old.size) + " vs " + std::to_string(win_new.size());
        } else {
            for (size_t i = 0; i < win_new.size(); ++i) {
                if (cur_old.data[i].logit != win_new[i].logit) {
                    ok = false; why = "window logit[" + std::to_string(i) + "] " + std::to_string(cur_old.data[i].logit) + " vs " + std::to_string(win_new[i].logit);
                    break;
                }
            }
        }
        if (!ok) ++n_win_bad;
    }
    if (ok && id_old != id_new) {
        // exact tie? both ids must carry the identical post-chain logit
        float l_old = NAN, l_new = NAN;
        if (cs->params.temp == 0.0f) {
            // the default greedy path leaves cur_p as the whole vocab in identity order with the
            // post-penalty logits, which is the value the argmax actually compared
            if ((size_t) id_old < cur_old.size && cur_old.data[id_old].id == id_old) l_old = cur_old.data[id_old].logit;
            if ((size_t) id_new < cur_old.size && cur_old.data[id_new].id == id_new) l_new = cur_old.data[id_new].logit;
        } else {
            // each id in its own path's post-chain window: on a tie at a top_p / min_p cut the two
            // windows hold the same logit sequence (checked above) but can keep different ids
            for (size_t i = 0; i < cur_old.size; ++i) if (cur_old.data[i].id == id_old) l_old = cur_old.data[i].logit;
            for (size_t i = 0; i < win_new.size(); ++i) if (win_new[i].id == id_new) l_new = win_new[i].logit;
        }
        if (l_old == l_new) {
            ++n_tie;
        } else {
            ok = false; why = "id " + std::to_string(id_old) + " (logit " + std::to_string(l_old) + ") vs " + std::to_string(id_new) + " (logit " + std::to_string(l_new) + ")";
        }
    }
    if (!ok) {
        ++n_bad;
        fprintf(stderr, "PXA_TOPK_RAW shadow MISMATCH #%lld after %lld compares: %s | W=%d top_k=%d top_p=%g min_p=%g temp=%g min_keep=%d rep=%g freq=%g pres=%g last_n=%d bias=%d\n",
                (long long) n_bad, (long long) n_cmp, why.c_str(), W, cs->params.top_k, cs->params.top_p, cs->params.min_p, cs->params.temp, cs->params.min_keep,
                cs->params.penalty_repeat, cs->params.penalty_freq, cs->params.penalty_present, cs->params.penalty_last_n, cs->server_biases != nullptr);
        if (mode >= 3) {
            GGML_ABORT("PXA_TOPK_RAW=3: shadow mismatch");
        }
        if (!g_pxa_topk_raw_disabled) {
            g_pxa_topk_raw_disabled = true;
            fprintf(stderr, "PXA_TOPK_RAW: fast path DISABLED for the rest of this process (shadow mismatch)\n");
        }
    }
    if (n_cmp % 1000 == 0 || !ok) {
        fprintf(stderr, "PXA_TOPK_RAW shadow stats: compared=%lld tie=%lld window_mismatch=%lld mismatch=%lld\n",
                (long long) n_cmp, (long long) n_tie, (long long) n_win_bad, (long long) n_bad);
    }
    return id_old;
}

static llama_token llama_sampling_sample_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first) {
    const int mode = pxa_topk_raw_mode();
    if (mode != 0 && !g_pxa_topk_raw_disabled) {
        const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));
        const int W = pxa_topk_raw_window(ctx_sampling, ctx_cfg, grammar_first, n_vocab);
        if (W > 0) {
            if (mode == 1) {
                ++g_pxa_topk_raw_stats[0];
                return pxa_topk_raw_sample(ctx_sampling, ctx_main, idx, W);
            }
            return pxa_topk_raw_shadow(ctx_sampling, ctx_main, ctx_cfg, idx, grammar_first, W, mode);
        }
    }
    return llama_sampling_sample_impl_default(ctx_sampling, ctx_main, ctx_cfg, idx, grammar_first);
}
// ---- end PXA_TOPK_RAW ----------------------------------------------------------------------

static llama_token llama_sampling_sample_impl_default(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first) {
    const common_params_sampling & params = ctx_sampling->params;

    const float   temp            = params.temp;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    const float   adaptive_target = params.adaptive_target;

    std::vector<float> original_logits;
    llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, /* grammar_first= */ grammar_first, &original_logits);
    llama_token_data_array & cur_p = ctx_sampling->cur_p;
    if (ctx_sampling->grammar != NULL && !grammar_first) {
        GGML_ASSERT(!original_logits.empty());
    }
    auto & rbudget = ctx_sampling->rbudget;

    llama_token id = 0;
    float * logits = llama_get_logits_ith(ctx_main, idx);
    // apply reasoning budget first
    common_reasoning_budget_apply(rbudget, &cur_p);
    // Sample grammar first for resampling
    if (ctx_sampling->grammar != NULL && grammar_first && grammar_should_apply(ctx_sampling)) {
        // Apply grammar constraints to all candidates
        llama_grammar_apply(ctx_sampling->grammar, ctx_main, &cur_p);
    }

    // llama_sampler_apply
    if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        } else if (adaptive_target >= 0.0f && ctx_sampling->adapt_p_ctx!=nullptr) {
            // adaptive p sampling
            llama_prep_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
            sampler_queue(ctx_main, params, ctx_sampling, cur_p, std::max(1, params.min_keep));
            id = llama_sample_token_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            sampler_queue(ctx_main, params,ctx_sampling, cur_p, min_keep);
            id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);

        }
    }

    if (grammar_first || !grammar_should_apply(ctx_sampling)) {
        return id;
    }

    if (ctx_sampling->grammar != NULL && !grammar_first && grammar_should_apply(ctx_sampling)) {
        // Get a pointer to the logits
        float * logits = llama_get_logits_ith(ctx_main, idx);

        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        llama_grammar_apply(ctx_sampling->grammar, ctx_main, &single_token_data_array);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid) {
            LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, common_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl_default(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ true);
        }
    }
    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}

static llama_token_data_array llama_sampling_prepare_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  std::vector<float> * original_logits) {
    const common_params_sampling & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    const float   penalty_freq    = params.penalty_freq;
    const float   penalty_present = params.penalty_present;

    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    if (ctx_sampling->grammar != NULL && !grammar_first) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + n_vocab};
    }

    // apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    if (ctx_cfg) {
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    if (ctx_sampling->elb_states.size() > ctx_sampling->elb_idx) {
        common_expiring_logit_bias_apply(ctx_sampling, logits);
    }

    cur.resize(n_vocab);

    if ((ctx_sampling->server_biases != nullptr) && (ctx_sampling->server_biases->size() == n_vocab)) {
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id] + ctx_sampling->server_biases->at(token_id), 0.0f};
        }
    } else {
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    ctx_sampling->cur_p = { cur.data(), cur.size(), false };

    llama_token_data_array & cur_p = ctx_sampling->cur_p;

    // apply penalties
    const auto& penalty_tokens = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : prev;
    const int penalty_tokens_used_size = std::min((int)penalty_tokens.size(), penalty_last_n);
    if (penalty_tokens_used_size) {
        // llama_token_nl() can legitimately return LLAMA_TOKEN_NULL (-1). Some vocabs (e.g. Falcon3's
        // BPE tokenizer) tokenize "\n" to zero tokens, and the loader's fallback
        // (llama-vocab.cpp, linefeed_id = special_pad_id) runs before LLM_KV_TOKENIZER_PAD_ID is read
        // from the GGUF, so it copies the BPE default, which is itself LLAMA_TOKEN_NULL. Reading
        // logits[-1] is then an out-of-bounds access one float before the current position's logit
        // row, which segfaults when that address happens to be unmapped.
        const llama_token nl_token = llama_token_nl(llama_get_model(ctx_main));
        const float nl_logit = nl_token != LLAMA_TOKEN_NULL ? logits[nl_token] : 0.0f;

        llama_sample_repetition_penalties(ctx_main, &cur_p,
                penalty_tokens.data() + penalty_tokens.size() - penalty_tokens_used_size,
                penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);

        if (!penalize_nl && nl_token != LLAMA_TOKEN_NULL) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == nl_token) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    // apply grammar checks before sampling logic
    if (grammar_first && ctx_sampling->grammar != NULL) {
        llama_grammar_apply(ctx_sampling->grammar, ctx_main, &cur_p);
    }

    return cur_p;
}

llama_token common_sampler_sample_legacy(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ false);
}

llama_token common_sampler_sample(
    struct common_sampler * ctx_sampling,
    struct llama_context * ctx_main,
    const int idx,
    bool grammar_first) {
    // PXA_HOST_TIMING: the whole sampler call is one bucket of the per-step host timing
    static const bool pxa_ht = llama_pxa_host_timing_enabled();
    if (!pxa_ht) {
        // Call the implementation function with is_resampling set to false by default
        return llama_sampling_sample_impl(ctx_sampling, ctx_main, nullptr, idx, /* is_resampling= */ grammar_first);
    }
    const int64_t t0 = ggml_time_us();
    const llama_token id = llama_sampling_sample_impl(ctx_sampling, ctx_main, nullptr, idx, /* is_resampling= */ grammar_first);
    llama_pxa_host_timing_add_sample(ggml_time_us() - t0);
    return id;
}

llama_token_data_array llama_sampling_prepare(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  std::vector<float> * original_logits) {
    return llama_sampling_prepare_impl(ctx_sampling,ctx_main, ctx_cfg, idx, grammar_first, original_logits);
}

void common_sampler_accept(
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token token,
        bool is_generated) {
    if (ctx_sampling->prev.size() > 0) {
        ctx_sampling->prev.erase(ctx_sampling->prev.begin());
    }
    ctx_sampling->prev.push_back(token);

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(ctx_sampling);
    if (ctx_sampling->rbudget && is_generated) {
        common_reasoning_budget_accept(ctx_sampling->rbudget, token);
    }

    if (ctx_sampling->grammar && accept_grammar) {
        llama_grammar_accept_token(ctx_sampling->grammar, ctx_main, token);
    }
    if (ctx_sampling->smpl) {
        llama_sampler_dry_accept(ctx_sampling->smpl, token);
    }

    if (ctx_sampling->elb_states.size() > ctx_sampling->elb_idx) {
        common_expiring_logit_bias_accept(ctx_sampling, ctx_main);
    }
}

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
            });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

std::vector<llama_token> llama_sampling_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<llama_token> & draft) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft);
}

// PXA_SPEC_RELAXED (2026-07-08): relaxed speculative acceptance. Instead of accepting a
// draft token ONLY when it exactly equals the token the target sampled, also accept it when
// it lands inside the target's post-sampler-chain candidate set (top_k/top_p/min_p filtered,
// softmaxed cur_p) with probability >= PXA_SPEC_RELAXED_PMIN. The kept token is then the
// DRAFT token (KV/positions after i were decoded conditioned on it), which is a token the
// target itself could have sampled at temp>0 — so coherence is preserved while per-position
// acceptance rises. Env-gated; the CONFIG LEVEL decides the default (2026-09-06 correction:
// this comment and three LEVERS.md rows said "default OFF" while the code below has returned
// ON at ENHANCE — the default level since 2026-09-03 — the whole time):
//   unset                       ENHANCE (level 2, the default) -> ON for spec lanes;
//                               DEFAULT / REFERENCE            -> OFF
//   PXA_SPEC_RELAXED=0 / =1     explicit, always wins over the level
//   PXA_SPEC_RELAXED_PMIN=0.05  probability floor (post-filter softmax mass)
// Auto-disabled for grammar-constrained sampling, mirostat, and temp<=0 (greedy keeps
// exact-match semantics), so every temperature-0 gate and every "with speculation" board row
// measured at temp 0 is exact-verify regardless of this switch; at temp>0 on defaults the
// output legitimately differs from the drafter-off run. The server prints one startup line
// (PXA_SPEC_RELAXED: ON (pmin=...)) whenever a spec stage is armed and this is live.
static bool pxa_spec_relaxed_enabled() {
    static const bool en = [] {
        const char * e = std::getenv("PXA_SPEC_RELAXED");
        if (e) {
            return atoi(e) != 0;   // explicit env always wins over the level default
        }
        // Level default — ggml_pxa_config_level() is the ONE definition (ggml/src/ggml.c), the
        // same one pxa_spec_relaxed_resolve() in ggml/src/ggml-cuda/pxa/pxa-enhance.cuh reads.
        // ENHANCE (the default level) -> on (G3, spec lanes only); DEFAULT/REFERENCE -> off.
        return ggml_pxa_config_level() >= 2;
    }();
    return en;
}
static float pxa_spec_relaxed_pmin() {
    static const float pmin = [] {
        const char * e = std::getenv("PXA_SPEC_RELAXED_PMIN");
        return e ? (float) atof(e) : 0.05f;
    }();
    return pmin;
}

// Public read-only view for startup ledgers (examples/server): what acceptance semantics a
// speculative stage will actually run under, resolved exactly as the sampler resolves them.
bool  common_sampler_spec_relaxed_active() { return pxa_spec_relaxed_enabled(); }
float common_sampler_spec_relaxed_pmin()   { return pxa_spec_relaxed_pmin(); }

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const std::vector<llama_token> & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        bool accept = (draft[i] == id);

        if (!accept && pxa_spec_relaxed_enabled() &&
            gsmpl->grammar == NULL &&
            gsmpl->params.mirostat == 0 &&
            gsmpl->params.temp > 0.0f) {
            // cur_p holds the post-chain candidates for THIS position (filled by the
            // common_sampler_sample call above); probabilities are softmaxed by the
            // token pick. Membership scan is over <= top_k entries.
            const llama_token_data_array & cur_p = gsmpl->cur_p;
            const float pmin = pxa_spec_relaxed_pmin();
            for (size_t j = 0; j < cur_p.size; ++j) {
                if (cur_p.data[j].id == draft[i]) {
                    if (cur_p.data[j].p >= pmin) {
                        id = draft[i];   // keep the draft token — later positions condition on it
                        accept = true;
                    }
                    break;
                }
            }
        }

        gsmpl->drafted_text += common_token_to_piece(ctx, id, true);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);

        if (!accept) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        gsmpl->drafted_text += common_token_to_piece(ctx, id, true);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);
    }

    return result;
}


static void elb_print(common_params_sampling& sparams, const common_params_sampling::elb_param::elb_entry& entry) {
    #undef X
    #define X(T, MEMBER, DV, PRECAST) #MEMBER,
    static const std::vector<std::string> names = { X_COMMON_PARAMS_SAMPLING };
    #undef X
    #define X(T, MEMBER, DV, PRECAST) if (std::abs(entry.addsubs[SPARAMS_ ## MEMBER ## _ENUM]) > 0.0f) \
    { LLAMA_LOG_DEBUG("%s: %s = %f\n", __func__, names[SPARAMS_ ## MEMBER ## _ENUM].c_str(), float(A_DOT_B(sparams, MEMBER))); }
    X_COMMON_PARAMS_SAMPLING
}

static void elb_add(common_params_sampling& sparams, const common_params_sampling::elb_param::elb_entry& entry) {
    #undef X
    #define X(T, MEMBER, _, PRECAST) A_DOT_B(sparams, MEMBER) += static_cast<T>(PRECAST(entry.addsubs[SPARAMS_ ## MEMBER ## _ENUM]));
    X_COMMON_PARAMS_SAMPLING
}

static void elb_sub(common_params_sampling& sparams, const common_params_sampling::elb_param::elb_entry& entry) {
    #undef X
    #define X(T, MEMBER, _, PRECAST) A_DOT_B(sparams, MEMBER) -= static_cast<T>(PRECAST(entry.addsubs[SPARAMS_ ## MEMBER ## _ENUM]));
    X_COMMON_PARAMS_SAMPLING
}

void common_expiring_logit_bias_apply(struct common_sampler* ctx_sampling, float* logits) {
    auto index_first_inactive = [](auto countup, auto& tokens) {
        return std::distance(
            tokens.begin(),
            std::upper_bound(tokens.begin(), tokens.end(), countup, [](const auto& countup, const auto& token) {
                return countup > token.duration;
            })
        );
    };

    const auto& elb = ctx_sampling->elb_states[ctx_sampling->elb_idx];

    std::string combined_text;
    const std::string* search_window = &combined_text;
    if (!ctx_sampling->drafted_text.empty()) {
        // add speculated tokens
        combined_text = ctx_sampling->to_generated_text != nullptr ? (
            ctx_sampling->to_generated_text->substr(std::max(0, int32_t(ctx_sampling->to_generated_text->length()) - elb.max_cond_len))
        ) : "" + ctx_sampling->drafted_text;
    } else if (ctx_sampling->to_generated_text != nullptr) {
        search_window = ctx_sampling->to_generated_text;
    }

    if (!search_window->empty() && !elb.other_tokens.empty() && (elb.other_tokens.front().duration > elb.countup)) {
        const auto ifi = index_first_inactive(elb.countup, elb.other_tokens);
        for (size_t j = 0; j < ifi; ++j) {
            const auto& [id, bias, _, cond] = elb.other_tokens[j];
            if (string_ends_with(*search_window, cond)) {
                logits[id] += bias;
            }
        }
    }

    if (!elb.first_tokens.empty() && (elb.first_tokens.front().duration > elb.countup)) {
        const auto ifi = index_first_inactive(elb.countup, elb.first_tokens);
        if (search_window->empty()) {
            // empty case here
            for (size_t j = 0; j < ifi; ++j) {
                logits[elb.first_tokens[j].id] += elb.first_tokens[j].bias;
            }
        } else {
            for (size_t j = 0; j < ifi; ++j) {
                const auto& [id, bias, _, cond] = elb.first_tokens[j];
                // no bias if seen (probably too late)
                if (!string_ends_with(*search_window, cond)) {
                    logits[id] += bias;
                }
            }
        }
    }

    // expiring sampler bias
    for (auto& entry: ctx_sampling->params.elb_params[ctx_sampling->elb_idx].entries) {
        if (!entry.biases.empty()) {
            continue;   // next entry
        }
        for (size_t j = 0; j < entry.phrases.size(); ++j) {
            const auto& phrase = entry.phrases[j];
            if (phrase.empty()) {
                // duration bound only
                if (elb.countup == 0) {
                    LLAMA_LOG_DEBUG("%s: before add\n", __func__);
                    elb_print(ctx_sampling->params, entry);

                    elb_add(ctx_sampling->params, entry);
                    entry.addflags[j] = true;

                    LLAMA_LOG_DEBUG("%s: after add\n", __func__);
                    elb_print(ctx_sampling->params, entry);
                } else if (elb.countup == entry.duration) {
                    LLAMA_LOG_DEBUG("%s: before sub\n", __func__);
                    elb_print(ctx_sampling->params, entry);

                    elb_sub(ctx_sampling->params, entry);
                    entry.addflags[j] = false;

                    LLAMA_LOG_DEBUG("%s: after sub\n", __func__);
                    elb_print(ctx_sampling->params, entry);
                }
                continue;   // next entry
            }
            size_t count = 0;
            auto pos = ctx_sampling->to_generated_text->find(phrase, entry.posi[j]);
            while (pos != std::string::npos) {
                LLAMA_LOG_DEBUG("%s: found %s @ %zu\n", __func__, phrase.c_str(), pos);
                ++count;
                pos = ctx_sampling->to_generated_text->find(phrase, pos + phrase.length());
            }
            entry.posi[j] = std::max(0, int32_t(ctx_sampling->to_generated_text->length()) - int32_t(phrase.length()) + 1);
            if (count % 2 == 1) {
                // even = no match or cancelled
                LLAMA_LOG_DEBUG("%s: before\n", __func__);
                elb_print(ctx_sampling->params, entry);

                (entry.addflags[j] ? elb_sub : elb_add)(ctx_sampling->params, entry);
                entry.addflags[j] = !entry.addflags[j];

                LLAMA_LOG_DEBUG("%s: after\n", __func__);
                elb_print(ctx_sampling->params, entry);
            }
        }
    }
}

void common_expiring_logit_bias_accept(struct common_sampler* ctx_sampling, struct llama_context * ctx_main) {
    if (ctx_sampling->to_generated_text == nullptr) {
        // prompt processing
        return;
    }

    auto idx = ctx_sampling->elb_idx;
    auto& elb = ctx_sampling->elb_states[idx];
    if ((elb.delay > ++elb.countup) || (elb.search_word_len == 0)) {
        return;
    }

    const std::string window = ctx_sampling->to_generated_text->substr(std::min(
        ctx_sampling->to_generated_text->length(),
        ctx_sampling->elb_search_pos)) + common_token_to_piece(ctx_main, ctx_sampling->prev.back(), true);
    size_t pos = 0;
    if (string_is_found(window, elb.jumpword, pos)) {
        LLAMA_LOG_DEBUG("%s: found %s in %s @ %zu\n", __func__, string_unescape(elb.jumpword).c_str(), string_unescape(window).c_str(), pos);
        pos += ctx_sampling->elb_search_pos + elb.jumpword.length();
        ctx_sampling->elb_idx = elb.jump_idx;
    } else if (string_is_found(window, elb.exitword, pos)) {
        LLAMA_LOG_DEBUG("%s: found %s in %s @ %zu\n", __func__, string_unescape(elb.exitword).c_str(), string_unescape(window).c_str(), pos);
        pos += ctx_sampling->elb_search_pos + elb.exitword.length();
        ++ctx_sampling->elb_idx;
    } else {
        // not found. move search position to include next token
        ctx_sampling->elb_search_pos += std::max(0, int32_t(window.length()) - int32_t(elb.search_word_len) + 1);
        return;
    }

    // single character clearance
    // e.g. stop \n\n from expiring two \n immediately
    ctx_sampling->elb_search_pos = pos + 1;

    // undo current sampler bias
    for (auto& entry: ctx_sampling->params.elb_params[idx].entries) {
        for (const auto addflag: entry.addflags) {
            if (addflag) {
                LLAMA_LOG_DEBUG("%s: before\n", __func__);
                elb_print(ctx_sampling->params, entry);

                elb_sub(ctx_sampling->params, entry);

                LLAMA_LOG_DEBUG("%s: after\n", __func__);
                elb_print(ctx_sampling->params, entry);
            }
        }
    }

    // prepare next sampler bias
    for (auto& entry: ctx_sampling->params.elb_params[ctx_sampling->elb_idx].entries) {
        // no clearance for sampler bias
        std::fill(entry.posi.begin(), entry.posi.end(), pos);
    }
}


template <>
json common_grammar_trigger::to_json() const {
    json out{
        {"type", (int)type},
        {"value", value},
    };
    if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out["token"] = (int)token;
    }
    return out;
}

template <>
common_grammar_trigger common_grammar_trigger::from_json(const json& in) {
    common_grammar_trigger out;
    out.type = (common_grammar_trigger_type)in.at("type").get<int>();
    out.value = in.at("value").get<std::string>();
    if (out.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out.token = (llama_token)in.at("token").get<int>();
    }
    return out;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
static bool common_sampler_speculative_top1_avx2(const float * logits, const int n_vocab, int & best_id, float & max_val) {
    if (n_vocab < 8) {
        return false;
    }

    __m256  max_v = _mm256_loadu_ps(logits);
    __m256i id_v  = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    const __m256i step = _mm256_set1_epi32(8);
    __m256i cur_id = _mm256_add_epi32(id_v, step);

    int i = 8;
    for (; i + 7 < n_vocab; i += 8) {
        const __m256 x = _mm256_loadu_ps(logits + i);
        const __m256 gt_max = _mm256_cmp_ps(x, max_v, _CMP_GT_OQ);

        max_v = _mm256_blendv_ps(max_v, x, gt_max);
        id_v  = _mm256_blendv_epi8(id_v, cur_id, _mm256_castps_si256(gt_max));
        cur_id = _mm256_add_epi32(cur_id, step);
    }

    alignas(32) float max_buf[8];
    alignas(32) int id_buf[8];
    _mm256_store_ps(max_buf, max_v);
    _mm256_store_si256((__m256i *) id_buf, id_v);

    best_id = id_buf[0];
    max_val = max_buf[0];

    for (int j = 1; j < 8; ++j) {
        if (max_buf[j] > max_val) {
            max_val = max_buf[j]; best_id = id_buf[j];
        }
    }
    for (; i < n_vocab; ++i) {

        if (logits[i] > max_val) {
            max_val = logits[i]; best_id = i;
        }
    }
    return true;
}
__attribute__((target("avx2,fma")))
static inline __m256 v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
__attribute__((target("avx2")))
static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
__attribute__((target("avx2")))
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
__attribute__((target("avx2,fma")))
static float prob_avx2(int n, const float * logits, float max_val) {
    float sumf = 0;
    int i = 0;
    if (n >= 8) {
        auto sum_v = _mm256_setzero_ps();
        auto max_v = _mm256_set1_ps(max_val);
        for (; i < n - 7; i += 8) {
            auto x = _mm256_loadu_ps(logits + i);
            auto exp_x = v_expf(_mm256_sub_ps(x, max_v));
            sum_v = _mm256_add_ps(sum_v, exp_x);
        }
        sumf = hsum_float_8(sum_v);
    }
    for (; i < n; ++i) {
        sumf += expf(logits[i] - max_val);
    }
    return 1.0f/sumf;
}
#endif
static float prob_scalar(int n, const float * logits, float max_val) {
    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_exp += exp((double)(logits[i] - max_val));
    }
    return (float)(1./sum_exp);
}

llama_token common_sampler_sample_speculative(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, float * out_prob) {
    GGML_UNUSED(gsmpl);

    float * logits = llama_get_logits_ith(ctx, idx);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    int best_id = 0;
    float max_val = logits[0];
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    static const bool has_avx2 = __builtin_cpu_supports("avx2");
    if (has_avx2 && common_sampler_speculative_top1_avx2(logits, n_vocab, best_id, max_val)) {
        if (out_prob) {
            static const bool has_fma = __builtin_cpu_supports("fma");
            if (has_fma) {
                *out_prob = prob_avx2(n_vocab, logits, max_val);
            } else {
                *out_prob = prob_scalar(n_vocab, logits, max_val);
            }
        }
        return best_id;
    }
#endif
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best_id = i;
        }
    }

    if (out_prob) {
        *out_prob = prob_scalar(n_vocab, logits, max_val);
    }

    return best_id;
}
