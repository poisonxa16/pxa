#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// PXA_FLINT: FlintForCausalLM (RecursiveSI/flint), ported from upstream PR 30928 onto this tree's
// builders. Per layer: RMS norm -> Q/K/V (+ per-head Q/K RMS norm) -> RoPE on the sliding layers only
// (the full-attention layers are NoPE) -> attention with learnable sinks (sliding mask on the SWA
// layers) -> residual -> RMS norm -> dense SiLU MLP (first n_layer_dense_lead layers) or
// sigmoid-routed top-k MoE (normalised weights, optional selection bias) + ungated shared expert
// -> residual.
ggml_cgraph * llm_build_context::build_flint() {
    ggml_cgraph * gf = new_graph_custom();

    GGML_ASSERT(n_embd_head_k == n_embd_head_v);

    ggml_tensor * cur;
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head_k));

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa  = hparams.swa_layers[il];
        const bool do_rope = !hparams.nope_layers[il];
        const bool use_swa_kv = is_swa && lctx.swa_kv_active;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                is_swa ? KQ_mask_swa : KQ_mask, model.layers[il].attn_sinks,
                nullptr, kq_scale, 0.0f, is_swa ? hparams.n_swa : 0, il, do_rope, false, true,
                false, false, nullptr, false,
                // PXA_FLINT_ISWA: a sliding layer's K/V live in the window cache when it is armed
                // (kv_self holds nothing for that layer then); off -> no override, original graph.
                use_swa_kv ? &kv_swa : nullptr, use_swa_kv ? n_kv_swa : 0, use_swa_kv ? kv_head_swa : 0);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            const float w_scale = hparams.expert_weights_scale;
            const bool  scale_w = w_scale != 0.0f && w_scale != 1.0f;
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,   nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, scale_w, w_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
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
