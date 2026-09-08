
#include "llama-hparams.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include <map>

#define LLAMA_MAX_EXPERTS 512  // Qwen3 Next

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,   "none"   },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR, "linear" },
    { LLAMA_ROPE_SCALING_TYPE_YARN,   "yarn"   },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

const char * llama_hparams::rope_scaling_type_name(llama_rope_scaling_type type) {
    return LLAMA_ROPE_SCALING_TYPES.at(type);
}

static inline const char * llm_expert_gating_func_name(llm_expert_gating_func_type type) {
    switch (type) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX: return "softmax";
        case LLM_EXPERT_GATING_FUNC_SIGMOID: return "sigmoid";
        case LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT: return "weight";
        case LLM_EXPERT_GATING_FUNC_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default: return "none";
    }
}



// ---------------------------------------------------------------------------
// DSpark support-file hparams.
//
// The DSpark support GGUF carries NINE metadata keys and NO geometry: no
// block_count, no context_length, no embedding_length, no vocab, no tokenizer.
// llm_load_hparams' mandatory prologue reads all three of those unconditionally
// and throws before the arch switch is ever reached, so this arch gets an early
// return and derives its geometry from its OWN tensor shapes instead.
//
// Everything genuinely NOT derivable from a shape (n_expert_used, sinkhorn
// iteration count, rms eps, rope base/scaling, o_group_count, ...) is copied
// later from the bound target model - see llama_dspark_bind_target(). A
// mismatched donor loads cleanly and drafts garbage; the assertions there are
// the only thing standing between us and that.
// ---------------------------------------------------------------------------
static int64_t dspark_tensor_ne(llama_model_loader & ml, const char * name, int dim) {
    const ggml_tensor * t = ml.get_tensor_meta(name);
    if (!t) {
        throw std::runtime_error(format("DSpark support file is missing tensor '%s'", name));
    }
    return t->ne[dim];
}

static void llm_load_hparams_dspark(llama_model_loader & ml, llama_model & model) {
    auto & hparams = model.hparams;

    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    ml.get_key(LLM_KV_DSPARK_BLOCK_SIZE,  hparams.dspark_block_size);
    ml.get_key(LLM_KV_DSPARK_STAGE_COUNT, hparams.dspark_stage_count);
    ml.get_key(LLM_KV_DSPARK_N_LAYERS,    hparams.n_layer);
    ml.get_key(LLM_KV_DSPARK_MARKOV_RANK, hparams.dspark_markov_rank);
    {
        uint32_t noise = 0;
        ml.get_key(LLM_KV_DSPARK_NOISE_TOKEN_ID, noise);
        hparams.dspark_noise_token = (int32_t) noise;
    }
    {
        // only get_arr<std::vector<uint32_t>>(llm_kv, ...) is explicitly instantiated in
        // this tree (llama-model-loader.cpp:1635); the layer ids are small non-negative
        // ints, so read them as uint32 and narrow.
        std::vector<uint32_t> ids;
        ml.get_arr(LLM_KV_DSPARK_TARGET_LAYER_IDS, ids, true);
        if (ids.empty() || ids.size() > hparams.dspark_target_layer_ids.size()) {
            throw std::runtime_error(format("DSpark: dspark.target_layer_ids has %zu entries", ids.size()));
        }
        for (size_t i = 0; i < ids.size(); ++i) hparams.dspark_target_layer_ids[i] = (int32_t) ids[i];
        hparams.dspark_n_target_layers = (uint32_t) ids.size();
    }

    // stage_count IS n_layers - settled by ds4.c metal_graph_eval_dspark_stage_chain,
    // not by the paper's 4-phase pipeline framing.
    if (hparams.dspark_stage_count != hparams.n_layer) {
        throw std::runtime_error(format("DSpark: stage_count=%u != n_layers=%u",
                    hparams.dspark_stage_count, hparams.n_layer));
    }
    const uint32_t last = hparams.n_layer - 1;

    // ---- geometry, derived from the support file's own tensor shapes ----
    const int64_t main_proj_in = dspark_tensor_ne(ml, "mtp.0.main_proj.weight", 0);   // 3*n_embd
    hparams.n_embd = (uint32_t) (main_proj_in / (int64_t) hparams.dspark_n_target_layers);
    if ((int64_t) hparams.n_embd * (int64_t) hparams.dspark_n_target_layers != main_proj_in) {
        throw std::runtime_error("DSpark: main_proj input width is not a multiple of the capture-layer count");
    }
    if (dspark_tensor_ne(ml, "mtp.0.main_norm.weight", 0) != (int64_t) hparams.n_embd) {
        throw std::runtime_error("DSpark: main_norm width disagrees with the derived n_embd");
    }

    const std::string w1 = format("mtp.%u.markov_head.markov_w1.weight", last);
    hparams.n_vocab = (uint32_t) dspark_tensor_ne(ml, w1.c_str(), 1);
    if ((uint32_t) dspark_tensor_ne(ml, w1.c_str(), 0) != hparams.dspark_markov_rank) {
        throw std::runtime_error("DSpark: markov_w1 rank disagrees with dspark.markov_rank");
    }

    hparams.n_head_arr.fill(0);
    hparams.n_head_kv_arr.fill(0);
    const uint32_t n_head = (uint32_t) dspark_tensor_ne(ml, "mtp.0.attn_sinks.weight", 0);
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        hparams.n_head_arr[il]    = n_head;
        hparams.n_head_kv_arr[il] = 1;             // MLA: one shared latent
    }
    // n_embd_head_{k,v} are accessors over the _full/_swa pair in this tree, not fields.
    // The drafter has no SWA-vs-full split (swa_layers stays all-zero), so both slots
    // get the same value and every accessor path lands on it.
    hparams.n_embd_head_k_full = (uint32_t) dspark_tensor_ne(ml, "mtp.0.attn_kv.weight", 1);
    hparams.n_embd_head_k_swa  = hparams.n_embd_head_k_full;
    hparams.n_embd_head_v_full = hparams.n_embd_head_k_full;
    hparams.n_embd_head_v_swa  = hparams.n_embd_head_k_full;
    hparams.swa_layers.fill(0);
    hparams.n_lora_q      = (uint32_t) dspark_tensor_ne(ml, "mtp.0.attn_q_a.weight", 1);
    hparams.n_expert      = (uint32_t) dspark_tensor_ne(ml, "mtp.0.ffn_gate_exps.weight", 2);
    hparams.n_ff_exp      = (uint32_t) dspark_tensor_ne(ml, "mtp.0.ffn_gate_exps.weight", 1);
    hparams.n_ff_arr.fill(hparams.n_ff_exp);
    hparams.dsv4_hc_mult  = (uint32_t) (dspark_tensor_ne(ml, "mtp.0.hc_attn_fn.weight", 0) / (int64_t) hparams.n_embd);

    // attn_output is the DS4 grouped LoRA pair:
    //   a : {n_head*n_embd_head_k / o_groups , o_lora_rank * o_groups}
    //   b : {o_groups * o_lora_rank          , n_embd}
    // Both scalars fall out of a.ne[] given n_head and n_embd_head_k, so the drafter
    // does NOT need the target for them. (Reading a.ne[1] as o_lora_rank directly is
    // wrong by a factor of o_groups - it is o_lora_rank*o_groups.)
    {
        const int64_t a0 = dspark_tensor_ne(ml, "mtp.0.attn_output_a.weight", 0);
        const int64_t a1 = dspark_tensor_ne(ml, "mtp.0.attn_output_a.weight", 1);
        const int64_t qtot = (int64_t) n_head * (int64_t) hparams.n_embd_head_k_full;
        if (a0 <= 0 || qtot % a0 != 0) {
            throw std::runtime_error("DSpark: attn_output_a width does not divide n_head*n_embd_head_k");
        }
        hparams.dsv4_o_group_count = (uint32_t) (qtot / a0);
        if (hparams.dsv4_o_group_count == 0 || a1 % (int64_t) hparams.dsv4_o_group_count != 0) {
            throw std::runtime_error("DSpark: attn_output_a rank is not a multiple of o_groups");
        }
        hparams.dsv4_o_lora_rank = (uint32_t) (a1 / (int64_t) hparams.dsv4_o_group_count);
    }

    // one shared expert of n_ff_exp columns
    hparams.n_expert_shared = (uint32_t) (dspark_tensor_ne(ml, "mtp.0.ffn_gate_shexp.weight", 1)
                                          / (int64_t) hparams.n_ff_exp);

    // The drafter has NO hash-routing table (no mtp.N.ffn_gate_tid2eid) and no
    // compressor/indexer tensors: all three stages are plain raw-SWA blocks.
    hparams.dsv4_hash_layer_count = 0;
    std::fill(hparams.dsv4_compress_ratios.begin(), hparams.dsv4_compress_ratios.end(), 0u);

    // Placeholders; llama_dspark_bind_target() overwrites them from the target.
    hparams.n_ctx_train   = 4096;
    // Provisional. Not implied by any drafter tensor shape and not used by any tensor
    // shape either - it only reaches the graph. llama_dspark_bind_target() overwrites it
    // with the target's value before any drafter graph is built. A non-zero value is
    // required here only because the shared LOADING_PRELUDE rejects n_expert>0 with
    // n_expert_used==0.
    hparams.n_expert_used = 6;

    model.type  = e_model::MODEL_UNKNOWN;
    model.ftype = ml.ftype;

    // llm_load_hparams' epilogue never runs for this arch (we returned early), and
    // that epilogue is what populates rope_type. Unset it stays LLAMA_ROPE_TYPE_NONE
    // (-1), and -1 & 1 == 1 trips ggml_rope's mode assert on the first rope node.
    // Use the same single source of truth every other arch uses.
    hparams.rope_type = llama_rope_type(&model);

    LLAMA_LOG_INFO("%s: DSpark support: stages=%u block_size=%u markov_rank=%u noise_id=%d "
                   "n_embd=%u n_vocab=%u n_head=%u kv_lora=%u q_lora=%u n_expert=%u n_ff_exp=%u hc=%u\n",
                   __func__, hparams.n_layer, hparams.dspark_block_size, hparams.dspark_markov_rank,
                   hparams.dspark_noise_token, hparams.n_embd, hparams.n_vocab, n_head,
                   hparams.n_embd_head_k_full, hparams.n_lora_q, hparams.n_expert, hparams.n_ff_exp,
                   hparams.dsv4_hc_mult);
}

void llm_load_hparams(
        llama_model_loader & ml,
        llama_model & model, bool ignore_vocab) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.meta;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        model.gguf_kv.emplace(name, value);
    }

    // DSpark: the support file has NO geometry keys. Its hparams come from its own
    // tensor shapes; the required-key prologue below would throw before the arch
    // switch is ever reached. Must stay ABOVE the LLM_KV_BLOCK_COUNT read.
    if (model.arch == LLM_ARCH_DEEPSEEK4_DSPARK) {
        llm_load_hparams_dspark(ml, model);
        return;
    }

    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    // get hparams kv
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab, !ignore_vocab);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the per-layer hparams
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(hparams.swa_layers.begin(),    hparams.swa_layers.end(),    0);
    std::fill(hparams.recurrent_layer_arr.begin(), hparams.recurrent_layer_arr.end(), false);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    // by default assume that the sliding-window layers use the same scaling type as the non-sliding-window layers
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON || model.arch == LLM_ARCH_BITNET_25 || model.arch == LLM_ARCH_BITNET_B158 || model.arch == LLM_ARCH_DECI) {
            if (hparams.n_rot != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // arch-specific KVs
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 8) {
                    switch (hparams.n_layer) {
                        case 32: model.type = e_model::MODEL_8x7B; break;
                        case 56: model.type = e_model::MODEL_8x22B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    switch (hparams.n_layer) {
                        case 22: model.type = e_model::MODEL_1B; break;
                        case 26: model.type = e_model::MODEL_3B; break;
                        // granite uses a vocab with len 49152
                        case 32: model.type = hparams.n_vocab == 49152 ? e_model::MODEL_3B : (hparams.n_vocab < 40000 ? e_model::MODEL_7B : e_model::MODEL_8B); break;
                        case 36: model.type = e_model::MODEL_8B; break; // granite
                        case 40: model.type = e_model::MODEL_13B; break;
                        case 48: model.type = e_model::MODEL_34B; break;
                        case 60: model.type = e_model::MODEL_30B; break;
                        case 80: model.type = hparams.n_head() == hparams.n_head_kv() ? e_model::MODEL_65B : e_model::MODEL_70B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_LLAMA4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step);
                hparams.n_swa_pattern = 4;    // pattern: 3 chunked - 1 full
                hparams.n_attn_chunk  = 8192; // should this be a gguf kv? currently it's the same for Scout and Maverick
                hparams.n_swa = 1; // TODO @ngxson : this is added to trigger the SWA branch (we store the chunked attn mask in the SWA tensor), will need to clean this up later

                switch (hparams.n_expert) {
                    case 16:  model.type = MODEL_17B_16E; break;
                    case 128: model.type = MODEL_17B_128E; break;
                    default:  model.type = MODEL_UNKNOWN;
                }

                if (model.type == MODEL_17B_128E) {
                    hparams.use_kq_norm = false;
                    }
            } break;
        case LLM_ARCH_DECI:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
	            case 162: model.type = e_model::MODEL_405B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_2B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GROK:
            {
                // defaults for old GGUFs
                hparams.yarn_beta_fast = 8.0f;
                hparams.f_logit_scale = 0.5773502691896257f;
                hparams.f_embedding_scale = 78.38367176906169f;
                hparams.f_attn_out_scale = 0.08838834764831845f;
                hparams.f_attn_logit_softcapping = 30.0f;
                hparams.f_router_logit_softcapping = 30.0f;
                // no final_logit_softcapping in grok-1
                hparams.f_final_logit_softcapping = 0.0f;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale, false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
                ml.get_key(LLM_KV_ATTENTION_OUTPUT_SCALE, hparams.f_attn_out_scale, false);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_ROUTER_LOGIT_SOFTCAPPING, hparams.f_router_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.attn_temp_length, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR, hparams.yarn_ext_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, hparams.yarn_beta_slow, false);

                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_314B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 60: model.type = e_model::MODEL_40B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                if (model.type == e_model::MODEL_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 42: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        model.type = e_model::MODEL_17M; break; // bge-micro
                    case 6:
                        model.type = e_model::MODEL_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: model.type = e_model::MODEL_33M; break; // MiniLM-L12, bge-small
                            case 768: model.type = e_model::MODEL_109M; break; // bge-base
                        } break;
                    case 24:
                        model.type = e_model::MODEL_335M; break; // bge-large
                }
            } break;
        case LLM_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4: model.type = e_model::MODEL_33M; break; // jina-embeddings-small
                    case 12: model.type = e_model::MODEL_137M; break; // jina-embeddings-base
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    model.type = e_model::MODEL_137M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                        } break;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_MPT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_12B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = hparams.n_head() == 20 ? e_model::MODEL_4B : e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_A2_7B; break;
                    case 28: model.type = e_model::MODEL_57B_A14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;

        case LLM_ARCH_QWEN3:
            {
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_6B : e_model::MODEL_1_7B; break;
                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3VL:
            {
                ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_1_7B; break;
                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                // since vision model stacks deepstack features along feature dim
                // we also create a fake "n_embd" for text model to be the main embd + deepstack embds
                hparams.n_embd *= hparams.n_deepstack_layers + 1;
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_30B_A3B; break;
                    case 94: model.type = e_model::MODEL_235B_A22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MELLUM:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);

                if (hparams.n_swa > 0) {
                    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
                    hparams.rope_freq_scale_train_swa = 1; //hparams.rope_freq_scale_train;

                    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer, false)) {
                        for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                            hparams.swa_layers[i] = ((i + 1) % 4 != 0);
                        }
                    }

                    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                }

                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_12B_A2_5B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3NEXT:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                // PXA qwen3next-MTP: NextN/MTP parameters (mirror qwen35moe). Defaults to 0 on
                // a no-MTP GGUF so today's production model loads unchanged.
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Upstream convention: every 4th layer is full attention, others are recurrent.
                // PXA: the MTP tail layer(s) are FULL-ATTENTION (not delta-net) -> mark them
                // non-recurrent, carved out of the main-layer loop (mirror qwen35moe).
                {
                    const uint32_t n_main_layers = hparams.n_layer - hparams.nextn_predict_layers;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        if (i < n_main_layers) {
                            hparams.recurrent_layer_arr[i] = ((i + 1) % 4 != 0);
                        } else {
                            hparams.recurrent_layer_arr[i] = false;
                        }
                    }
                }

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_80B_A3B; break;
                    case 49: model.type = e_model::MODEL_80B_A3B; break; // 48 main + 1 MTP
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN35MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // Load linear attention (gated delta net) parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Mark recurrent layers (linear attention layers)
                {
                    uint32_t full_attn_interval = 4;
                    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
                    const uint32_t n_main_layers = hparams.n_layer - hparams.nextn_predict_layers;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        if (i < n_main_layers) {
                            hparams.recurrent_layer_arr[i] = ((i + 1) % full_attn_interval != 0);
                        } else {
                            hparams.recurrent_layer_arr[i] = false;
                        }
                    }
                }

                switch (hparams.n_layer) {
                    case 40:
                    case 41:
                        model.type = e_model::MODEL_35B_A3B; break;
                    case 48: model.type = e_model::MODEL_122B_A10B; break;
                    case 60: model.type = e_model::MODEL_397B_A17B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN4EXP:
            {
                // TENSOR-LOADER SCOPE: this case loads exactly the hparams that
                // create_qwen4exp_tensors() needs to compute tensor SHAPES, plus the
                // recurrent-layer map that decides which branch each layer takes. Anything
                // the forward graph needs and the shapes do not (indexer_top_k, the PLE
                // head-offset/vocab-size tables, ple eos/image token ids, HC epsilon,
                // sinkhorn iterations, compress ratios) is deliberately NOT read here - the
                // graph phase owns those and reading them now would just create a second
                // place to keep in sync.
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

                // Gated DeltaNet geometry, same key set as qwen35moe
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // split_recurrent_tensors() derives head_v_dim as ssm_d_inner/ssm_dt_rank,
                // while the qwen4exp tensor shapes use ssm_d_state for BOTH head dims. If the
                // converter ever writes an ssm_d_inner that disagrees, -sm graph would shard
                // the delta-net weights on a different head width than they were created with
                // and silently mis-slice them. Catch it here, at load, not there.
                GGML_ASSERT(hparams.ssm_dt_rank > 0 && hparams.ssm_n_group > 0);
                if (hparams.ssm_d_inner != hparams.ssm_d_state * hparams.ssm_dt_rank) {
                    throw std::runtime_error(format(
                        "qwen4exp: ssm_inner_size (%u) must equal ssm_state_size (%u) * ssm_time_step_rank (%u) = %u",
                        hparams.ssm_d_inner, hparams.ssm_d_state, hparams.ssm_dt_rank,
                        hparams.ssm_d_state * hparams.ssm_dt_rank));
                }

                // Low-rank hyper-connections: hc_dim = dsv4_hc_mult * n_embd
                ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
                ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
                if (hparams.dsv4_hc_mult == 0 || hparams.hc_low_rank == 0) {
                    throw std::runtime_error("qwen4exp: needs a non-zero hyper_connection.count and hyper_connection.low_rank");
                }

                // NextN/MTP graft: a nextn_predict_layers>0 GGUF carries one extra
                // full-attention tail layer (blk.<n_layer-1>) with a blk.*.nextn.* fusion +
                // head mixer. Same n_layer_kv_from_start handling as qwen35/qwen35moe.
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // QSA lightning indexer geometry (shapes only; top_k is a graph concern)
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);

                // PLE hash constants. The tensor-loader scope deliberately skipped these -
                // shapes do not need them - but the forward graph does, so they are read here.
                {
                    std::vector<uint64_t> mult, off, voc;
                    ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, mult, false);
                    ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,      off,  false);
                    ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,  voc,  false);
                    if (mult.size() > llama_hparams::PLE_MAX_NGRAM ||
                        off.size()  > llama_hparams::PLE_MAX_HEADS ||
                        voc.size()  > llama_hparams::PLE_MAX_HEADS) {
                        throw std::runtime_error(format(
                            "qwen4exp: PLE tables larger than this build allows (multipliers %zu/%zu, offsets %zu/%zu, vocab %zu/%zu)",
                            mult.size(), llama_hparams::PLE_MAX_NGRAM,
                            off.size(),  llama_hparams::PLE_MAX_HEADS,
                            voc.size(),  llama_hparams::PLE_MAX_HEADS));
                    }
                    std::copy(mult.begin(), mult.end(), hparams.ple_layer_multipliers.begin());
                    std::copy(off.begin(),  off.end(),  hparams.ple_head_offsets.begin());
                    std::copy(voc.begin(),  voc.end(),  hparams.ple_head_vocab_sizes.begin());
                }
                ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,      hparams.ple_eos_token_id,      false);
                ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,    hparams.ple_image_token_id,    false);

                // PLE n-gram hash embeddings. Absent key group => no PLE at all.
                {
                    hparams.ple_layer_arr.fill(false);
                    hparams.ple_n_heads = 0;

                    uint32_t n_ple = 0;
                    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
                    if (n_ple > 0) {
                        std::vector<uint32_t> ple_layers;
                        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
                        for (uint32_t il : ple_layers) {
                            if (il >= hparams.n_layer) {
                                throw std::runtime_error(format("qwen4exp: ple.layers names layer %u but the model has %u layers", il, hparams.n_layer));
                            }
                            hparams.ple_layer_arr[il] = true;
                        }

                        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,             hparams.ple_ngram_size);
                        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM,        hparams.ple_heads_per_ngram);
                        ml.get_key(LLM_KV_PLE_CONV_KERNEL,            hparams.ple_conv_kernel);
                        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);

                        if (hparams.ple_ngram_size < 2) {
                            throw std::runtime_error("qwen4exp: ple.ngram_size must be >= 2");
                        }
                        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
                        hparams.ple_head_dim = hparams.n_embd_per_layer;
                        if (hparams.ple_n_heads == 0 || hparams.ple_head_dim == 0) {
                            throw std::runtime_error("qwen4exp: PLE key group present but heads_per_ngram / embedding_length_per_layer_input is zero");
                        }
                    }
                }

                // Linear attention everywhere except every full_attention_interval-th layer.
                // The NextN/MTP tail (the last nextn_predict_layers layers of a grafted GGUF)
                // is ALWAYS plain full attention, never delta-net — the interval formula would
                // otherwise call layer 48 of a 49-block graft recurrent ((48+1)%4 != 0).
                {
                    uint32_t full_attn_interval = 4;
                    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
                    GGML_ASSERT(full_attn_interval > 0);
                    const uint32_t n_main_layers = hparams.n_layer - hparams.nextn_predict_layers;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        hparams.recurrent_layer_arr[i] = i < n_main_layers && ((i + 1) % full_attn_interval != 0);
                    }
                }

                switch (hparams.n_layer) {
                    case 48: // plain model
                    case 49: // with the grafted MTP tail (48 main + 1 nextn)
                        model.type = e_model::MODEL_180B_A10B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN35:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // Load linear attention (gated delta net) parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Mark recurrent layers (linear attention layers)
                // MTP layers always use standard attention, not delta-net
                {
                    uint32_t full_attn_interval = 4;
                    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
                    const uint32_t n_main_layers = hparams.n_layer - hparams.nextn_predict_layers;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        if (i < n_main_layers) {
                            hparams.recurrent_layer_arr[i] = ((i + 1) % full_attn_interval != 0);
                        } else {
                            hparams.recurrent_layer_arr[i] = false;
                        }
                    }
                }

                switch (hparams.n_layer) {
                    case 24: // without MTP layer
                    case 25: // with MTP layer (24 main + 1 MTP)
                        model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_8B : e_model::MODEL_2B; break;
                    case 32: // without MTP layer
                    case 33: // with MTP layer (32 main + 1 MTP)
                        model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B   : e_model::MODEL_9B; break;
                    case 64: // without MTP layer
                    case 65: // with MTP layer (64 main + 1 MTP)
                        model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3VLMOE:
            {
                ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_30B_A3B; break;
                    case 94: model.type = e_model::MODEL_235B_A22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                // since vision model stacks deepstack features along feature dim
                // we also create a fake "n_embd" for text model to be the main embd + deepstack embds
                hparams.n_embd *= hparams.n_deepstack_layers + 1;
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI3:
            {
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // for backward compatibility ; see: https://github.com/ggerganov/llama.cpp/pull/8931
                if ((hparams.n_layer == 32 || hparams.n_layer == 40) && hparams.n_ctx_train == 4096) {
                    // default value for Phi-3-mini-4k-instruct and Phi-3-medium-4k-instruct
                    hparams.n_swa = 2047;
                } else if (hparams.n_layer == 32 && hparams.n_head_kv(0) == 32 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-mini-128k-instruct
                    hparams.n_swa = 262144;
                } else if (hparams.n_layer == 40 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-medium-128k-instruct
                    hparams.n_swa = 131072;
                }
                bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (!found_swa && hparams.n_swa == 0) {
                    throw std::runtime_error("invalid value for sliding_window");
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: model.type = e_model::MODEL_SMALL; break;
                    case 24: model.type = e_model::MODEL_MEDIUM; break;
                    case 36: model.type = e_model::MODEL_LARGE; break;
                    case 48: model.type = e_model::MODEL_XL; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_20B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: model.type = e_model::MODEL_2B; break;
                    case 28: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA2:
            {
                hparams.n_swa = 4096; // default value of gemma 2
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
                hparams.attn_soft_cap = true;

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 42: model.type = e_model::MODEL_9B; break;
                    case 46: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA3:
            {
                hparams.n_swa_pattern = 6;

                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 34: model.type = e_model::MODEL_4B; break;
                    case 48: model.type = e_model::MODEL_12B; break;
                    case 62: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                hparams.f_attention_scale = model.type == e_model::MODEL_27B
                    ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
                    : 1.0f / std::sqrt(float(hparams.n_embd_head_k_full));
            } break;
        case LLM_ARCH_MUSE_GLIMMER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);

                // Local (SWA) layers run RoPE at the class freq base (a dedicated
                // swa base overrides when present); full-attention layers are NoPE
                // (handled in the graph builder).
                hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
                hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

                // sliding_window_pattern is a scalar period (4 = local,local,local,global)
                // on released GGUFs, or a per-layer array. NOTE: get_key_or_arr on the
                // array target would broadcast a scalar period into every layer (all-SWA),
                // so the scalar must be read first.
                uint32_t swa_period = 0;
                bool have_period = false;
                try {
                    have_period = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
                } catch (...) {
                    have_period = false;
                }
                if (have_period && swa_period > 0) {
                    hparams.n_swa_pattern = swa_period;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        hparams.swa_layers[i] = ((i + 1) % swa_period != 0);
                    }
                } else {
                    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer, false);
                }

                switch (hparams.n_layer) {
                    case 52: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA4:
            {
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

                uint32_t n_kv_shared_layers = 0;
                ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

                hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t)n_kv_shared_layers;
                hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
                hparams.f_final_logit_softcapping = 30.0f;

                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

                switch (hparams.n_layer) {
                    case 35: model.type = e_model::MODEL_2B; break;
                    case 42: model.type = e_model::MODEL_4B; break; // to confirm: E4B or E5B?
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA4_MTP:
            {
                ml.get_key(LLM_KV_MTP_BACKBONE_EMBEDDING_LENGTH, hparams.mtp_backbone_n_embd);
                ml.get_key(LLM_KV_MTP_USE_ORDERED_EMBEDDINGS,    hparams.mtp_use_ordered_embeddings, false);
                ml.get_key(LLM_KV_MTP_CENTROID_COUNT,            hparams.mtp_num_centroids, false);
                ml.get_key(LLM_KV_MTP_CENTROID_TOP_K,            hparams.mtp_centroid_top_k, false);

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,       hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,             hparams.rope_freq_base_train_swa, false);

                hparams.n_layer_kv_from_start = hparams.n_layer;
                hparams.f_attention_scale     = 1.0f;

                switch (hparams.mtp_backbone_n_embd) {
                    case 5376: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;

        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    case 52: model.type = e_model::MODEL_20B; break; // granite
                    case 88: model.type = e_model::MODEL_34B; break; // granite
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: model.type = e_model::MODEL_SMALL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: model.type = e_model::MODEL_MEDIUM; break;
                            case 1536: model.type = e_model::MODEL_LARGE; break;
                            case 2048: model.type = e_model::MODEL_XL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_XVERSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_65B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_35B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DBRX:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
            ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv);

            switch (hparams.n_layer) {
                case 40: model.type = e_model::MODEL_16x12B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_OLMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

                switch (hparams.n_layer) {
                    case 22: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                case 16: model.type = e_model::MODEL_270M; break;
                case 20: model.type = e_model::MODEL_450M; break;
                case 28: model.type = e_model::MODEL_1B; break;
                case 36: model.type = e_model::MODEL_3B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL, hparams.use_par_res);
                switch (hparams.n_layer) {
                    case 6:
                        switch (hparams.n_ff()) {
                            case 512: model.type = e_model::MODEL_14M; break;
                            case 2048: model.type = e_model::MODEL_70M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_160M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 16:
                        switch (hparams.n_ff()) {
                            case 8192: model.type = e_model::MODEL_1B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096: model.type = e_model::MODEL_410M; break;
                            case 8192: model.type = e_model::MODEL_1_4B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_ff()) {
                            case 10240: model.type = e_model::MODEL_2_8B; break;
                            case 16384: model.type = e_model::MODEL_6_9B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 36:
                        switch (hparams.n_ff()) {
                            case 20480: model.type = e_model::MODEL_12B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 44:
                        switch (hparams.n_ff()) {
                            case 24576: model.type = e_model::MODEL_20B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ARCTIC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 128) {
                    switch (hparams.n_layer) {
                        case 35: model.type = e_model::MODEL_10B_128x3_66B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_DEEPSEEK2:
            {
                int expected_head_size_k = model.arch == LLM_ARCH_DEEPSEEK2 ? 576 : 320;
                int expected_head_size_v = model.arch == LLM_ARCH_DEEPSEEK2 ? 512 : 256;
                if (hparams.n_head_kv() == 1) {
                    int n_nead_kv = hparams.n_gqa();
                    if (n_nead_kv%4 != 0 || hparams.n_embd_head_k(0) != expected_head_size_k || hparams.n_embd_head_v(0) != expected_head_size_v ||
                        hparams.n_rot != 64) {
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        LLAMA_LOG_ERROR("Detected incompatible DeepSeek model without a known way to fix it.\n");
                        LLAMA_LOG_ERROR("Consider making your own pxa compatible model or\n");
                        LLAMA_LOG_ERROR("ask the model provider to make one for you,\n\n");
                        LLAMA_LOG_ERROR("Sorry, uknown model => cannot fix it => bailing out\n");
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        GGML_ABORT("Fatal error");
                    }
                    LLAMA_LOG_INFO("================= Adjusted mainline llama.cpp MLA tensors to pxa\n");
                    for (auto& item : hparams.n_head_kv_arr) item = n_nead_kv;
                    hparams.n_embd_head_k_full = 192;
                    hparams.n_embd_head_v_full = 128;
                    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_full);
                    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_full);
                }
                bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26) || (hparams.n_layer == 48 && hparams.n_vocab == 128256);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_TYPE_NONE;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // Older DeepSeek models from the 2.0/2.5 series may not have the experts gating function recorded in the GGUF.
                    // Such models use SOFTMAX as the experts gating function.
                    // The new (new as of this commit) GLM-4.7-Flash may also be missing the experts gating function.
                    // GLM-4.7-Flash uses SIGMOID as the experts gating function.
                    // Hence, we make the LLM_KV_EXPERT_GATING_FUNC entry optional, and set here if missing.
                    // We distinguish between GLM-4.7-Flash and DeepSeek-2/2.5 models by the number of layers.
                    // GLM-4.7-Flash has 47 layers (or 48, if an MTP layer is included in the GGUF).
                    hparams.expert_gating_func = hparams.n_layer == 47 || hparams.n_layer == 48 ?
                        LLM_EXPERT_GATING_FUNC_SIGMOID : LLM_EXPERT_GATING_FUNC_SOFTMAX;
                    LLAMA_LOG_INFO("================= Missing experts gating function -> set to %s\n",
                            llm_expert_gating_func_name(llm_expert_gating_func_type(hparams.expert_gating_func)));
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false);

                switch (hparams.n_layer) {
                    case 27: model.type = e_model::MODEL_16B; break;
                    case 36: model.type = e_model::MODEL_119B_A6B; break;
                    case 47: model.type = e_model::MODEL_30B_A3B; break; // GLM-4.7-Flash
                    case 60: model.type = e_model::MODEL_236B; break;
                    case 61: model.type = e_model::MODEL_671B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHATGLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_6B; break;
                    case 40: model.type = e_model::MODEL_9B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_9B; break;
                    case 61: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // Expert gating function (GLM4_MOE uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == 0) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // NextN/MTP parameters
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                }
                else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);

                switch (hparams.n_layer) {
                    case 47: model.type = e_model::MODEL_106B_A12B; break; // GLM-4.5-Air (46 layers + 1 NextN layer)
                    case 93: model.type = e_model::MODEL_355B_A32B; break; // GLM-4.5 (92 layers + 1 NextN layer)
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_2B; break; // bitnet2b_2501
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_T5:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

                uint32_t dec_start_token_id;
                if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
                    hparams.dec_start_token_id = dec_start_token_id;
                }

                switch (hparams.n_layer) {
                    case 6:  model.type = e_model::MODEL_60M;  break; // t5-small
                    case 8:  model.type = e_model::MODEL_80M;  break; // flan-t5-small
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_220M; break; // t5-base
                            case 2048: model.type = e_model::MODEL_250M; break; // flan-t5-base
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096:  model.type = e_model::MODEL_770M; break; // t5-large
                            case 2816:  model.type = e_model::MODEL_780M; break; // flan-t5-large
                            case 16384: model.type = e_model::MODEL_3B;   break; // t5-3b
                            case 5120:  model.type = e_model::MODEL_3B;   break; // flan-t5-xl
                            case 65536: model.type = e_model::MODEL_11B;  break; // t5-11b
                            case 10240: model.type = e_model::MODEL_11B;  break; // flan-t5-xxl
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_JAIS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1_3B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    /* TODO: add variants */
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_3B; break;
                    // Add additional layer/vocab/etc checks here for other model sizes
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2:
            {
                hparams.n_swa_pattern = 4;
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
                ml.get_key(LLM_KV_LOGIT_SCALE,                      hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,        hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,       hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,              hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,               hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }
                switch (hparams.n_layer) {
                    case 49: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAILINGMOE2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups);
                ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);

                // PXA_BAILING_MTP_KV_FIX: when MTP is enabled (-mtp / model.mtp) the NextN tail layer
                // is run as a real attention layer in the MTP companion graph, so it MUST have a KV
                // cache cell. has_kv(tail)=false (n_layer_kv_from_start = n_layer-nextn) made
                // llm_build_kqv fall through to kv.k_l[n_layer_kv_from_start-1], which is NULL in the
                // MTP-only context (only the tail layer is allocated there) -> the
                // "k_cache is null in llm_build_kqv" assert. Mirror qwen35moe / qwen35 / glm4_moe:
                // when model.mtp, all layers (incl. the tail) count as KV-bearing so has_kv(tail)=true
                // -> llm_build_kqv uses kv.k_l[tail] directly instead of the null kv.k_l[tail-1] fallback.
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                switch (hparams.n_layer) {
                    case 20: model.type = MODEL_16B_A1B; break;
                    case 21: model.type = MODEL_16B_A1B; break;
                    case 32: model.type = MODEL_100B_A6B; break;
                    case 33: model.type = MODEL_100B_A6B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
	    case LLM_ARCH_DOTS1:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_142B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
            if (model.arch == LLM_ARCH_ERNIE4_5_MOE) {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP, hparams.n_moe_layer_step);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
            }

            switch (hparams.n_layer) {
            case 18: model.type = e_model::MODEL_0_3B; break;
            case 28: model.type = e_model::MODEL_21B_A3B; break;
            case 54: model.type = e_model::MODEL_300B_A47B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_80B_A13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

                //TODO OAI_MOE: SWA
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                //hparams.set_swa_pattern(2);

                // TODO: switch (hparams.n_layer)

            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                // PXA: minimax-m2 is architecturally sigmoid-gated. Sibling arches
                // (deepseek/glm/qwen3next) reset to NONE then force SIGMOID when the GGUF
                // omits the key; the minimax block didn't -> a key-less GGUF routed with the
                // wrong (softmax) gating -> garbage. Match the siblings.
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_TYPE_NONE;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_230B_A10B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.n_no_rope_layer_step = 4;

                switch (hparams.n_layer) {
                    case 36: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE, hparams.f_attn_temp_scale, false);

                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,     hparams.rope_yarn_log_mul, false);

                if (hparams.f_attn_temp_scale != 0.0f) {
                    hparams.n_attn_temp_floor_scale = hparams.n_ctx_orig_yarn;
                    if (hparams.n_attn_temp_floor_scale == 0) {
                        throw std::runtime_error("invalid n_ctx_orig_yarn for attention temperature scaling");
                    }
                }

                // TODO: this seems to be correct with the case of mscale == mscale_all_dims == 1.0f
                //       but may need further verification with other values
                if (hparams.rope_yarn_log_mul != 0.0f) {
                    float factor = 1.0f / hparams.rope_freq_scale_train;
                    float mscale = 1.0f;
                    float mscale_all_dims = hparams.rope_yarn_log_mul;
                    static auto get_mscale = [](float scale, float mscale) {
                        return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
                    };
                    hparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);
                }

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 34: model.type = e_model::MODEL_8B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MIMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,         hparams.rope_freq_base_train_swa);
                ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE,      hparams.f_attn_v_scale, false);
                //TODO
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; // which is the same as OpenAI
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_310B_A15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

            } break;
        case LLM_ARCH_SEED_OSS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_36B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STEP35:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                // MoE + SWA parameters
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                // Step35 uses sigmoid gating by default (if not set in GGUF)
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                bool have_rfb_train_swa = ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
                if (!ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER, hparams.rope_dim_per_layer, hparams.n_layer, false)) {
                    for (int i = 0; i < hparams.n_layer; ++i) {
                        hparams.rope_dim_per_layer[i] = hparams.swa_layers[i] ? hparams.n_rot : hparams.n_rot/2;
                    }
                }
                // The following two parameters: one of the two versions must be present in the GGUF
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_LIMITS,        hparams.swiglu_limits,        hparams.n_layer, false)) {
                     ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_limits, hparams.n_layer, true);
                }
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_LIMITS_SHARED, hparams.swiglu_limits_shared, hparams.n_layer, false)) {
                     ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_limits_shared, hparams.n_layer, true);
                }
                // Optional: Step35-only gating for applying rope scaling (HF: yarn_only_types).
                // Default is 3 (apply on all layers) if the key is absent.
                //ml.get_key(format("%s.rope.scaling.apply_mask", ml.get_arch_name().c_str()),
                //    hparams.rope_scaling_apply_mask, false);
                //hparams.has_rope_freq_base_per_layer = ml.get_key_or_arr(
                //    format("%s.rope.freq_base_per_layer", ml.get_arch_name().c_str()),
                //    hparams.rope_freq_base_per_layer, hparams.n_layer, false);
                ml.get_key(format("%s.rope.scaling.apply_mask", ml.get_arch_name().c_str()),
                    hparams.rope_scaling_apply_mask, false);
                hparams.has_rope_freq_base_per_layer = ml.get_key_or_arr(LLM_KV_ROPE_FREQ_BASE_PER_LAYER,
                    hparams.rope_freq_base_per_layer, hparams.n_layer, false);
                GGML_ASSERT(hparams.has_rope_freq_base_per_layer || have_rfb_train_swa);
            } break;
        case LLM_ARCH_LAGUNA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoE (n_expert / n_expert_used are read by the common loader above)
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                hparams.n_expert_shared = 1;
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                if (hparams.n_ff_shexp == 0) {
                    hparams.n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;
                }
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,      hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,       hparams.expert_weights_norm,  false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,        hparams.expert_gating_func,   false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // Sliding-window attention is OPTIONAL. S-2.1/XS.2 are hybrid
                // full/SWA period 4 (FULL at il%4==0, dense-first); M.1 has no
                // sliding window (all full). window (n_swa) read from the GGUF.
                hparams.n_swa = 0;
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (hparams.n_swa > 0) {
                    hparams.n_swa_pattern = 4;
                    // The unsloth GGUF stores no per-layer SWA pattern array; derive
                    // the period-4 dense-first schedule directly (matches mainline
                    // set_swa_pattern(4, dense_first=true)): FULL at il%4==0.
                    for (int i = 0; i < (int) hparams.n_layer; ++i) {
                        hparams.swa_layers[i] = (i % 4 != 0);
                    }
                    // Dual RoPE: full-attn layers = YaRN (theta from ROPE_FREQ_BASE);
                    // SWA layers = plain RoPE (theta from ROPE_FREQ_BASE_SWA, no YaRN
                    // scaling). The YaRN ext/attn/beta params are zeroed per-SWA-layer
                    // inside build_std_attention.
                    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
                    hparams.rope_freq_scale_train_swa = 1.0f;  // SWA never inherits the YaRN 1/factor
                    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                }
                // Partial rotary dims (n_rot=64 full / n_rot_swa=128 SWA) are already
                // read by the common loader from ROPE_DIMENSION_COUNT[_SWA]. Zero any
                // per-layer rope-dim override so rope_n_rot(il) falls through to
                // n_rot / n_rot_swa (STEP35's swa?n_rot:n_rot/2 wrongly assumes
                // n_rot==head_dim, which is false for Laguna's partial rotary).
                std::fill(hparams.rope_dim_per_layer.begin(), hparams.rope_dim_per_layer.end(), 0);
                hparams.has_rope_freq_base_per_layer = false;

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_118B_A8B; break;   // Laguna-S 2.1
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_HY_V3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoE. n_expert / n_expert_used come from the common loader above
                // (192 / 8). Hy3 is a sigmoid router with a per-expert selection
                // bias, sum-normalised top-k weights and a router_scaling_factor.
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);           // 1536
                hparams.n_expert_shared = 1;
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
                // The converter already writes moe_intermediate_size * num_shared_experts here,
                // so this is the total shared-expert width - do NOT multiply by n_expert_shared again.
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);  // 1536
                if (hparams.n_ff_shexp == 0) {
                    hparams.n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;
                }
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead,   false); // 1
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,      hparams.expert_weights_scale, false); // 2.826
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,       hparams.expert_weights_norm,  false); // true
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,        hparams.expert_gating_func,   false); // SIGMOID
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // NextN/MTP. config.json declares num_nextn_predict_layers=1 but the
                // released checkpoint carries no MTP block, so the converter omits the
                // key entirely and this stays 0 (trunk loop = n_layer, no MTP graph).
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // Plain RoPE ("rope_type": "default"): no YaRN, no sliding window.
                // n_rot falls back to attention.key_length (128) in the common loader.
                switch (hparams.n_layer) {
                    case 80: model.type = e_model::MODEL_295B_A21B; break;
                    case 81: model.type = e_model::MODEL_295B_A21B; break;  // + 1 MTP block
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM_DSA:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // deepseek MLA parameters
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
                //ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
                //ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

                // DSA parameters
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

                // Expert gating function (GLM-4.5 uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);

                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                }
                else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                switch (hparams.n_layer) {
                    case 79: model.type = MODEL_744B_A40B; break;
                    default: model.type = MODEL_UNKNOWN;
                }
                if (hparams.n_head_kv() == 1) {
                    int n_nead_kv = hparams.n_gqa();
                    if (n_nead_kv%4 != 0 || hparams.n_embd_head_k_full != 576 || hparams.n_embd_head_v_full != 512 ||
                        hparams.n_rot != 64) {
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        LLAMA_LOG_ERROR("Detected incompatible DeepSeek model without a known way to fix it.\n");
                        LLAMA_LOG_ERROR("Sorry, uknown model => cannot fix it => bailing out\n");
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        GGML_ABORT("Fatal error");
                    }
                    LLAMA_LOG_INFO("================= Adjusted mainline llama.cpp MLA tensors to pxa\n");
                    for (auto& item : hparams.n_head_kv_arr) item = n_nead_kv;
                    hparams.n_embd_head_k_full = 192;
                    hparams.n_embd_head_v_full = 128;
                    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_full);
                    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_full);
                }
            } break;
        case LLM_ARCH_DEEPSEEK4:
            {
                // Adapted from llama.cpp src/models/deepseek4.cpp
                // (llama_model_deepseek4::load_arch_hparams, lines 18-60) @ upstream
                // commit 82dbc4f01, PR #24162 and follow-ups.
                // Copyright (c) 2023-2026 The ggml authors. MIT.
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);

                // Upstream stores these in hparams.swiglu_clamp_{exp,shexp}. This tree
                // already maps the very same GGUF keys onto swiglu_limits /
                // swiglu_limits_shared (see LLM_ARCH_STEP35 above), so reuse those.
                // NOTE the DS4 semantics: the GATE is clamped to [-inf, limit] BEFORE
                // swiglu_split - it is NOT silu-then-clamp (graph lane, chunk D).
                ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_limits, hparams.n_layer, true);
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_limits_shared, hparams.n_layer, false)) {
                    hparams.swiglu_limits_shared = hparams.swiglu_limits;
                }

                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

                ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
                ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
                ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
                ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
                ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
                ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
                ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

                // Per-layer attention regime. get_key_or_arr() insists the array length is
                // exactly n_layer (upstream accepted >= n_layer) and throws a descriptive
                // error otherwise; a scalar would broadcast, which the validation below
                // still accepts only if it is one of the three legal ratios.
                std::fill(hparams.dsv4_compress_ratios.begin(), hparams.dsv4_compress_ratios.end(), 0u);
                {
                    // DS4 ships MORE ratios than blocks: the 0731 Flash release carries 46 for
                    // 43 layers. Upstream accepts >= n_layer and uses the first n_layer; our
                    // shared get_key_or_arr() demands an exact match and would reject the model
                    // outright. Relaxing the shared guard is not an option -- for every other
                    // arch a short array must stay an error, not a silent broadcast of zeros --
                    // so read the raw array here and take the prefix.
                    //
    // The per-layer regime pattern occupies indices [0, n_layer); anything past that
                    // is padding whose length is NOT fixed -- converters in the wild emit both
                    // n_layer+1 and n_layer+hash_layer_count. Upstream requires only size >= n_layer
                    // and copies the prefix (llama-model.cpp, LLM_ARCH_DEEPSEEK4), so match that.
                    // A SHORT array stays a hard error: broadcasting one value across every layer
                    // would assign the wrong attention regime and yield plausible garbage.
                    //     idx  0  1  2   3  4   5 ...  42 | 43 ...
                    //     val  0, 0, 4, 128, 4, 128 ...  4 |  0 ...
                    std::vector<uint32_t> ratios;
                    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, ratios, true);
                    if (ratios.size() < hparams.n_layer) {
                        throw std::runtime_error(format(
                                    "DeepSeek-V4: attention.compress_ratios has %u entries, need at least n_layer=%u",
                                    (uint32_t) ratios.size(), hparams.n_layer));
                    }
                    if (hparams.n_layer > hparams.dsv4_compress_ratios.size()) {
                        throw std::runtime_error(format("DeepSeek-V4: n_layer=%u exceeds LLAMA_MAX_LAYERS", hparams.n_layer));
                    }
                    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                        hparams.dsv4_compress_ratios[il] = ratios[il];
                    }
                }
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    const uint32_t r = hparams.dsv4_compress_ratios[il];
                    if (r != 0 && r != 4 && r != 128) {
                        throw std::runtime_error(format(
                                    "DeepSeek-V4 loader only supports compression ratios 0, 4 and 128 (layer %u has %u)",
                                    il, r));
                    }
                }

                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
                if (hparams.expert_gating_func != LLM_EXPERT_GATING_FUNC_SQRT_SOFTPLUS) {
                    throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
                }

                // Upstream additionally does:
                //     hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                //     hparams.set_swa_pattern(0);
                // Neither exists in this tree: there is only the scalar n_swa (set above
                // from attention.sliding_window = 128) plus the per-layer swa_layers[]
                // flag. swa_layers[] is deliberately left all-zero - it selects
                // n_embd_head_k_swa inside n_embd_head_k(il), which is NOT what DS4 wants.
                // "Which layers are raw-SWA" is dsv4_compress_ratios[il] == 0, and that
                // decision is carried by the DSV4 memory module (chunk C) and the graph
                // (chunk D), not by swa_layers[].

                // Upstream leaves the model type UNKNOWN for every layer count
                // (deepseek4.cpp:57-60). Do not invent an "<n>B_A<m>B" label until the
                // active-parameter count has actually been measured.
                switch (hparams.n_layer) {
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        default: (void)0;
    }

    model.ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_rope_type(&model);
}
