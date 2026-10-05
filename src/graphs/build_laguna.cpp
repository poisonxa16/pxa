#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Laguna (poolside): sigmoid-routed MoE with a score-correction bias, one shared
// expert, a per-head softplus attention output gate, QK-norm, and per-layer-type
// RoPE (YaRN on full-attention layers, plain RoPE on sliding-window layers).
// Structurally a STEP35 sibling: the softplus gate and the SWA YaRN-zeroing are
// arch-gated inside build_std_attention, so this graph is a near-verbatim copy of
// build_step35 minus the rope_freqs handling (Laguna carries no rope-factor tensor).
ggml_cgraph * llm_build_context::build_laguna() {
    ggml_cgraph * gf = new_graph_custom();
    ggml_tensor * cur;
    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_pos     = build_inp_pos();
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();
    auto KQ_mask_swa = build_inp_KQ_mask_swa();
    // Constant head_dim (128) on every layer, including SWA -> kq_scale is always
    // 1/sqrt(head_dim), independent of the partial rotary dim count.
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];

        // The per-head softplus gate + o_proj + QK-norm + dual RoPE (YaRN on full
        // layers, plain RoPE on SWA layers) are all handled inside
        // build_std_attention (arch-gated for LAGUNA). No rope-factor tensor.
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 && n_tokens > 1 ? inp_out_ids : nullptr,
                /*rope_factors*/ nullptr, is_swa ? KQ_mask_swa : KQ_mask, nullptr, nullptr,
                kq_scale, 0.0f, is_swa ? hparams.n_swa : 0, il, true, false, true);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense FFN (leading dense layer 0)
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            // MoE: sigmoid routing + score-correction bias (exp_probs_b) + sum-norm +
            // routed_scaling_factor (2.5); routed experts scaled, shared added unscaled.
            const bool  norm_w  = hparams.expert_weights_norm;    // true
            const float w_scale = hparams.expert_weights_scale;   // 2.5
            const bool  scale_w = w_scale != 0.0f;                // true
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr,
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, norm_w, scale_w, w_scale,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
