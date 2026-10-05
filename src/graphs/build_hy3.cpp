#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Hunyuan V3 (hy_v3): GQA (64 q / 8 kv heads, head_dim 128) with per-head RMS
// QK-norm, one leading dense block, then a sigmoid-routed MoE stack with a
// per-expert selection bias (exp_probs_b), sum-normalised top-k weights, a
// router_scaling_factor, and one always-on shared expert. Plain RoPE (NEOX,
// full rotary), no sliding window, no attention gate.
//
// Structurally build_laguna() minus exactly two things: the sliding-window /
// dual-RoPE split (hy_v3 has no SWA, so n_swa is always 0 and there is only one
// KQ mask) and the per-head softplus attention output gate (hy_v3 has no
// attn_gate tensor, so the gate branch inside build_std_attention self-disables
// on layers[il].wqkv_gate == nullptr).
//
// Router order of operations, matching the HF reference HYV3TopKRouter.forward:
//   probs      = sigmoid(logits)                  <- sigmoid FIRST
//   selection  = probs + exp_probs_b              <- bias AFTER sigmoid
//   idx        = top_k(selection)                 <- selection uses the BIASED scores
//   weights    = probs[idx]                       <- weights come from the UNBIASED probs
//   weights   /= sum(weights)                     <- route_norm
//   weights   *= router_scaling_factor            <- scale AFTER the normalisation
// llm_build_moe_ffn implements exactly this; the bias never enters the weights.
//
// Cross-checked against the upstream hy_v3 reference implementation (hy-v3.cpp,
// branch hy3-mtp): identical hparam reads, tensor shapes, residual structure and
// router arguments. The reference also carries a NextN/MTP draft graph; that is
// deliberately not ported - our MTP framework (cparams.mtp_op_type / build_*_mtp)
// is unrelated to the LLM_GRAPH_TYPE_DECODER_MTP plumbing it is written against,
// and the released checkpoint ships no MTP block to validate it with.
ggml_cgraph * llm_build_context::build_hy3() {
    ggml_cgraph * gf = new_graph_custom();

    // head_dim (128) is NOT n_embd/n_head (4096/64 = 64), and the whole head is rotated.
    GGML_ASSERT(n_embd_head_k == hparams.n_embd_head_v(0));
    GGML_ASSERT(n_embd_head_k == (int64_t) hparams.n_rot);

    ggml_tensor * cur;
    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_pos     = build_inp_pos();
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    // NextN/MTP tail blocks (none in the released checkpoint) are loaded but not run here.
    const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;

    for (int il = 0; il < n_transformer_layers; ++il) {
        // attn_norm -> q/k/v -> per-head RMS QK-norm -> NEOX RoPE -> KV/attn ->
        // o_proj -> + residual. All inside build_std_attention (add_input = true).
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_transformer_layers - 1 && n_tokens > 1 ? inp_out_ids : nullptr,
                /*rope_factors*/ nullptr, KQ_mask, /*sinks*/ nullptr, /*inp_attn_scale*/ nullptr,
                kq_scale, 0.0f, /*n_swa*/ 0, il, /*do_rope*/ true, /*add_graph_split*/ false,
                /*add_input*/ true);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // leading dense block (first_k_dense_replace = 1 -> layer 0)
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, /*add_input*/ true);
            cb(cur, "ffn_out", il);
        } else {
            // Routed experts are scaled by router_scaling_factor; the shared expert
            // is added unscaled and ungated, off the same ffn_norm'ed input.
            const bool  norm_w  = hparams.expert_weights_norm;    // route_norm = true
            const float w_scale = hparams.expert_weights_scale;   // router_scaling_factor = 2.826
            const bool  scale_w = w_scale != 0.0f;
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
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, /*add_input*/ true,
                    model.layers[il].ffn_up_gate_exps);
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
