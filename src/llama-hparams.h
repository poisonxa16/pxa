#pragma once

#include "llama-impl.h"

#include <cstdint>
#include <array>
#include <cmath>

#define LLAMA_MAX_LAYERS  512

enum llm_expert_gating_func_type {
    LLM_EXPERT_GATING_FUNC_TYPE_NONE             = 0,
    LLM_EXPERT_GATING_FUNC_SOFTMAX               = 1,
    LLM_EXPERT_GATING_FUNC_SIGMOID               = 2,
    LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3,
    // DeepSeek-V4. Value 4 is written into the GGUF by the converter and MUST match
    // gguf-py ExpertGatingFuncType.SQRTSOFTPLUS (upstream constants.py:4747).
    LLM_EXPERT_GATING_FUNC_SQRT_SOFTPLUS       = 4,
};

struct llama_hparams {
    bool vocab_only;
    bool rope_finetuned;
    bool use_par_res;

    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_layer;
    int32_t n_layer_kv_from_start = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache
    uint32_t n_rot;
    uint32_t n_rot_swa;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_swa_pattern = 1; // by default, all layers use non-sliding-window attention
    uint32_t n_embd_head_k_full; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v_full; // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types
    uint32_t n_rel_attn_bkts = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_ff_exp           = 0;
    uint32_t n_ff_shexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    float    expert_group_scale   = 0.05f;
    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLM_EXPERT_GATING_FUNC_SOFTMAX;
    uint32_t moe_every_n_layers   = 0;
    uint32_t nextn_predict_layers = 0;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    float f_attn_logit_softcapping   = 50.0f;
    float f_router_logit_softcapping = 30.0f;
    float f_final_logit_softcapping  = 30.0f;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa;
    uint32_t rope_scaling_apply_mask = 0x1;
    bool     has_rope_freq_base_per_layer = false;
    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul = 0.0f;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;

    std::array<int, 4> rope_sections;
    std::array<float,    LLAMA_MAX_LAYERS> rope_freq_base_per_layer;
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_dim_per_layer;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;

    // for hybrid state-space models (e.g. qwen3next)
    std::array<bool, LLAMA_MAX_LAYERS> recurrent_layer_arr;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;
    float f_attn_v_scale    = 1.0f;

    // grok-2
    float    f_attn_out_scale = 0.0f;
    uint32_t attn_temp_length = 0;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    uint32_t n_moe_layer_step        = 0;
    bool     use_kq_norm             = true;
    uint32_t n_attn_chunk            = 0;
    // values below seems to be fixed on llama4
    uint32_t n_no_rope_layer_step    = 4;
    uint32_t n_attn_temp_floor_scale = 8192;
    float    f_attn_temp_scale       = 0.1;

    // DSA (deepseek sparse attention)
    uint32_t indexer_n_head    = 0;
    uint32_t indexer_head_size = 0;
    uint32_t indexer_top_k     = 0;

    // DeepSeek-V4 (mirrors llama.cpp src/llama-hparams.h:241-248 @ upstream 82dbc4f01).
    // NOTE: DS4's swiglu clamp limits live in swiglu_limits / swiglu_limits_shared
    // below - this tree already routes the swiglu_clamp_exp / swiglu_clamp_shexp GGUF
    // keys into those arrays (see LLM_ARCH_STEP35), so upstream's separate
    // swiglu_clamp_exp / swiglu_clamp_shexp fields are deliberately NOT duplicated.
    uint32_t dsv4_o_group_count     = 0;
    uint32_t dsv4_o_lora_rank       = 0;
    uint32_t dsv4_hc_mult           = 0;
    uint32_t dsv4_hc_sinkhorn_iters = 0;
    // qwen4exp: the HC input mixer is low-rank (down: hc_dim -> hc_low_rank, up: back).
    // Zero means "full rank", which is what DeepSeek-V4 uses.
    uint32_t hc_low_rank            = 0;
    uint32_t dsv4_hash_layer_count  = 0;
    float    dsv4_compress_rope_base = 0.0f;
    float    dsv4_hc_eps             = 0.0f;
    // per-layer attention regime: 0 = raw SWA, 4 = CSA + lightning indexer, 128 = HCA
    std::array<uint32_t, LLAMA_MAX_LAYERS> dsv4_compress_ratios = {};

	// qwen3vl deepstack
    uint32_t n_deepstack_layers = 0;

    // gemma4 per-layer embedding; qwen4exp reuses it as the PLE head width
    uint32_t n_embd_per_layer = 0;

    // qwen4exp PLE (per-layer n-gram hash embeddings). ple_n_heads == 0 means the GGUF
    // carries no PLE key group at all and the whole side path is absent.
    uint32_t ple_ngram_size      = 0;
    uint32_t ple_heads_per_ngram = 0;
    uint32_t ple_conv_kernel     = 0;
    uint32_t ple_n_heads         = 0;
    uint32_t ple_head_dim        = 0;
    std::array<bool, LLAMA_MAX_LAYERS> ple_layer_arr = {};

    // PLE hash constants. The row for head h at position p is
    //     mixed = (t[p]*m[0]) ^ (t[p-1]*m[1]) ^ ... ; row = mixed % vocab[h] + offset[h]
    // computed HOST-side in llama_set_inputs, because ggml has neither int64 nor xor.
    // These are UINT64 arrays in the GGUF - reading them needed a UINT64 case in
    // llama_model_loader::get_arr, which both overloads previously threw on.
    // Fixed-size, because llama_hparams must stay trivially copyable (static_assert below).
    // n_gram is 3 and the head count is (n_gram-1)*heads_per_ngram = 16 in the shipped
    // model; these caps leave generous headroom and are checked at load.
    static constexpr size_t PLE_MAX_NGRAM = 16;
    static constexpr size_t PLE_MAX_HEADS = 128;
    std::array<uint64_t, PLE_MAX_NGRAM> ple_layer_multipliers = {};
    std::array<uint64_t, PLE_MAX_HEADS> ple_head_offsets      = {};
    std::array<uint64_t, PLE_MAX_HEADS> ple_head_vocab_sizes  = {};
    uint32_t ple_eos_token_id   = 0;
    uint32_t ple_image_token_id = 0;

    // gemma4 separate assistant MTP
    uint32_t mtp_backbone_n_embd = 0;
    bool     mtp_use_ordered_embeddings = false;
    uint32_t mtp_num_centroids = 0;
    uint32_t mtp_centroid_top_k = 0;

    // DSpark block drafter (arch deepseek4-dspark). Read from the support GGUF;
    // everything else the drafter needs is copied from the bound target model.
    uint32_t dspark_block_size    = 0;   // k candidate positions per proposal
    uint32_t dspark_stage_count   = 0;   // == n_layer, asserted at bind time
    uint32_t dspark_markov_rank   = 0;
    int32_t  dspark_noise_token   = -1;
    std::array<int32_t, 8> dspark_target_layer_ids = {};
    uint32_t dspark_n_target_layers = 0;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = -1;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_layers;

    std::array<float, LLAMA_MAX_LAYERS> swiglu_limits;
    std::array<float, LLAMA_MAX_LAYERS> swiglu_limits_shared;

    bool operator!=(const llama_hparams & other) const {
        if (this->vocab_only    != other.vocab_only)    return true;
        if (this->n_vocab       != other.n_vocab)       return true;
        if (this->n_ctx_train   != other.n_ctx_train)   return true;
        if (this->n_embd        != other.n_embd)        return true;
        if (this->mtp_backbone_n_embd != other.mtp_backbone_n_embd) return true;
        if (this->n_layer       != other.n_layer)       return true;
        if (this->n_rot         != other.n_rot)         return true;
        if (this->n_swa         != other.n_swa)         return true;
        if (this->n_swa_pattern != other.n_swa_pattern) return false;
        if (this->n_embd_head_k_full != other.n_embd_head_k_full) return true;
        if (this->n_embd_head_v_full != other.n_embd_head_v_full) return true;
        if (this->n_expert      != other.n_expert)      return true;
        if (this->n_expert_used != other.n_expert_used) return true;

        if (this->n_head_arr    != other.n_head_arr)    return true;
        if (this->n_head_kv_arr != other.n_head_kv_arr) return true;
        if (this->n_ff_arr      != other.n_ff_arr)      return true;

        if (this->n_rel_attn_bkts    != other.n_rel_attn_bkts)    return true;
        if (this->n_layer_dense_lead != other.n_layer_dense_lead) return true;
        if (this->n_lora_q           != other.n_lora_q)           return true;
        if (this->n_lora_kv          != other.n_lora_kv)          return true;
        if (this->n_ff_exp           != other.n_ff_exp)           return true;
        if (this->n_ff_shexp         != other.n_ff_shexp)         return true;
        if (this->n_expert_shared    != other.n_expert_shared)    return true;

        // DS4: the per-layer compression schedule changes the KV geometry, so it is
        // part of the cache identity. The scalar dsv4_* values follow from it.
        if (this->dsv4_compress_ratios != other.dsv4_compress_ratios) return true;

        if (this->rope_finetuned  != other.rope_finetuned)  return true;
        if (this->n_ctx_orig_yarn != other.n_ctx_orig_yarn) return true;

        if (this->ssm_d_conv  != other.ssm_d_conv)  return true;
        if (this->ssm_d_inner != other.ssm_d_inner) return true;
        if (this->ssm_d_state != other.ssm_d_state) return true;
        if (this->ssm_dt_rank != other.ssm_dt_rank) return true;
        if (this->ssm_n_group != other.ssm_n_group) return true;
        if (this->recurrent_layer_arr != other.recurrent_layer_arr) return true;

        if (this->dec_start_token_id != other.dec_start_token_id) return true;

        const float EPSILON = 1e-9f;

        if (!is_float_close(this->f_norm_eps,            other.f_norm_eps,            EPSILON)) return true;
        if (!is_float_close(this->f_norm_rms_eps,        other.f_norm_rms_eps,        EPSILON)) return true;
        if (!is_float_close(this->rope_attn_factor,      other.rope_attn_factor,      EPSILON)) return true;
        if (!is_float_close(this->rope_freq_base_train,  other.rope_freq_base_train,  EPSILON)) return true;
        if (!is_float_close(this->rope_freq_scale_train, other.rope_freq_scale_train, EPSILON)) return true;
        if (!is_float_close(this->expert_weights_scale,  other.expert_weights_scale,  EPSILON)) return true;
        if (!is_float_close(this->rope_yarn_log_mul,     other.rope_yarn_log_mul,     EPSILON)) return true;
        if (!is_float_close(this->f_residual_scale,      other.f_residual_scale,      EPSILON)) return true;
        if (!is_float_close(this->f_embedding_scale,     other.f_embedding_scale,     EPSILON)) return true;
        if (!is_float_close(this->f_attention_scale,     other.f_attention_scale,     EPSILON)) return true;

        return false;
    }

    bool has_kv(uint32_t il) const {
        return n_layer_kv_from_start > 0 ? il < n_layer_kv_from_start : true;
    }

    uint32_t n_head(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_arr[il];
        }
        printf("%s: Oops, il = %d\n", __func__, il);
        GGML_ABORT("fatal error");
    }

    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_embd_inp() const {
        uint32_t n_embd_inp = n_embd;

        if (n_deepstack_layers > 0) {
            n_embd_inp += n_embd * n_deepstack_layers;
        }

        return n_embd_inp;
    }

    uint32_t n_ff(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_ff_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_gqa(uint32_t il = 0) const {
        const uint32_t n_head    = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);

        if (n_head_kv == 0) {
            return 0;
        }

        return n_head/n_head_kv;
    }

    uint32_t n_embd_head_k(int il) const { return swa_layers[il] ? n_embd_head_k_swa : n_embd_head_k_full; }

    uint32_t n_embd_head_v(int il) const { return swa_layers[il] ? n_embd_head_v_swa : n_embd_head_v_full; }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        return n_head_kv(il) * n_embd_head_k(il);
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const { // dimension of value embeddings across all k-v heads
        return n_head_kv(il) * n_embd_head_v(il);
    }

    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        if (ssm_n_group > 0) {
            // qwen3next keeps all recurrent state in the V-cache tail
            return 0;
        }
        // corresponds to Mamba's conv_states size
        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
    }

    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        if (ssm_n_group > 0) {
            // qwen3next recurrent state packs:
            // 1) conv state: (d_conv - 1) * (2 * key_dim + value_dim)
            // 2) delta-net state: head_v_dim * head_v_dim * num_v_heads
            const uint32_t key_dim        = ssm_d_state * ssm_n_group;
            const uint32_t value_dim      = ssm_d_inner;
            const uint32_t conv_dim       = 2 * key_dim + value_dim;
            const uint32_t conv_state_dim = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * conv_dim;
            const uint32_t head_v_dim     = ssm_dt_rank > 0 ? ssm_d_inner / ssm_dt_rank : 0;
            const uint32_t ssm_state_dim  = head_v_dim * head_v_dim * ssm_dt_rank;
            return conv_state_dim + ssm_state_dim;
        }
        // corresponds to Mamba's ssm_states size
        return ssm_d_state * ssm_d_inner;
    }

    std::pair<uint32_t, uint32_t> n_embd_v_s_dims(int nv) const {
        if (ssm_n_group <= 0 || nv < 1 || ssm_dt_rank < 1) return {0, 0};
        int num_v_heads = ssm_dt_rank;
        int num_k_heads = ssm_n_group;
        int gqa_ratio   = num_v_heads / num_k_heads;
        GGML_ASSERT(nv <= num_v_heads);
        GGML_ASSERT(nv % gqa_ratio == 0);
        int nk = nv / gqa_ratio;
        int head_k_dim  = ssm_d_state;
        int head_v_dim  = ssm_d_inner / num_v_heads;
        uint32_t conv_dim       = 2 * nk * head_k_dim + nv * head_v_dim;
        //uint32_t conv_state_dim = conv_dim * (ssm_d_conv - 1);
        //uint32_t ssm_state_dim  = head_v_dim * head_v_dim * nv;
        //return {conv_state_dim, ssm_state_dim};
        uint32_t ssm_state_dim  = head_v_dim * head_v_dim * nv;
        return {conv_dim, ssm_state_dim};
    }

    uint32_t n_embd_v_s_id(int nv) const {
        auto [conv_dim, ssm_state_dim] = n_embd_v_s_dims(nv);
        return (ssm_d_conv - 1) * conv_dim + ssm_state_dim;
    }

    bool is_recurrent(uint32_t il) const {
        return il < n_layer ? recurrent_layer_arr[il] : false;
    }

    // qwen4exp: true only on the layers named by <arch>.ple.layers
    bool is_ple(uint32_t il) const {
        return (ple_n_heads > 0 && il < n_layer) ? ple_layer_arr[il] : false;
    }

    static bool is_float_close(float a, float b, float abs_tol) {
        // Check for non-negative tolerance
        if (abs_tol < 0.0) {
            throw std::invalid_argument("Tolerance must be non-negative");
        }

        // Exact equality check
        if (a == b) {
            return true;
        }

        // Check for infinities
        if (std::isinf(a) || std::isinf(b)) {
            return false;
        }

        // Regular comparison using the provided absolute tolerance
        return std::fabs(b - a) <= abs_tol;
    }

    uint32_t rope_n_rot(uint32_t il) const {
        const uint32_t v = rope_dim_per_layer[il];
        return v ? v : swa_layers[il] ? n_rot_swa : n_rot;
    }

    static const char * rope_scaling_type_name(llama_rope_scaling_type);

};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
