// PXA LFM port: LiquidAI LFM2 dense, LFM2-MoE and LFM3 (mainline src/models/lfm2.cpp, lfm2moe.cpp, lfm3.cpp,
// MIT, see NOTICE). Causal text models only (the non-causal decision-head variants are refused at load).
// Every block: h += mixer(rms_norm(h)); h += ffn(rms_norm(h)), where the mixer is the short-conv operator
// (delta_net::build_layer_shortconv, state in the s_l rows) on blocks without KV heads, else GQA with q/k RMS
// norms and NEOX rope. LFM2-MoE swaps the FFN for a sigmoid-routed SwiGLU MoE after n_layer_dense_lead.
// LFM3 adds sliding-window blocks (their own rope base, no rope factors), rope factors on the full blocks,
// a per-head sigmoid gate on the attention output and optional sandwich (post) norms.
#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

ggml_cgraph * llm_build_context::build_lfm() {
    ggml_cgraph * gf = new_graph_custom();

    const bool is_lfm3 = model.arch == LLM_ARCH_LFM3;
    const bool has_swa = hparams.n_swa > 0;

    delta_net mixer(lctx, batch);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    cb(inpL, "model.embed_tokens", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = has_swa ? build_inp_KQ_mask_swa() : nullptr;

    // the recurrent-state routing inputs the short-conv layers read (filled by llama_set_inputs, shared with DeltaNet)
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);
    lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
    ggml_set_input(lctx.inp_conv_seq_map);
    lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
    ggml_set_input(lctx.inp_qnext_state_mask);

    const bool  scale_w = hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = mixer.build_layer_shortconv(ctx0, gf, cur, il, cb);
        } else {
            const bool    is_swa      = has_swa && hparams.swa_layers[il];
            const int64_t n_embd_head = hparams.n_embd_head_k(il);
            const int64_t n_head_il   = hparams.n_head(il);
            const int     n_rot_l     = hparams.rope_n_rot(il);
            const float   freq_base_l  = is_swa && is_lfm3 ? hparams.rope_freq_base_train_swa  : freq_base;
            const float   freq_scale_l = is_swa && is_lfm3 ? hparams.rope_freq_scale_train_swa : freq_scale;
            ggml_tensor * rope_factors = is_swa ? nullptr : layer.rope_freqs;
            cur = pxqn_rot_in(cur, il, PXQN_SITE_ATTN_IN);   // PXQN attn_in (identity unless the file rotates the site)

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    nullptr, nullptr, nullptr, nullptr,
                    layer.wq, nullptr, layer.wk, nullptr, layer.wv, nullptr,
                    layer.attn_q_norm, layer.attn_k_norm, 0.0f, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            ggml_tensor * mask_l = is_swa ? KQ_mask_swa : KQ_mask;
            const float kq_scale = 1.0f/sqrtf(float(n_embd_head));
            if (layer.wqkv_gate) {
                // LFM3: out = wo @ (attn * sigmoid(gate @ x) per head)
                ggml_tensor * attn = llm_build_kv(ctx0, lctx, kv_self, gf, nullptr, nullptr,
                        Kcur, Vcur, Qcur, mask_l, n_tokens, kv_head, n_kv, kq_scale, cb, il, nullptr,
                        is_swa ? hparams.n_swa : 0);
                cb(attn, "attn_out", il);
                ggml_tensor * gate = ggml_sigmoid(ctx0, llm_build_lora_mm(lctx, ctx0, layer.wqkv_gate, cur));
                cb(gate, "attn_gate", il);
                attn = ggml_mul(ctx0, ggml_reshape_3d(ctx0, attn, n_embd_head, n_head_il, n_tokens),
                                      ggml_reshape_3d(ctx0, gate, 1,           n_head_il, n_tokens));
                attn = ggml_reshape_2d(ctx0, attn, n_embd_head*n_head_il, n_tokens);
                attn = build_pxqn_rht(ctx0, hparams, attn, il, PXQN_SITE_OUT_IN);
                cur = llm_build_lora_mm(lctx, ctx0, layer.wo, attn);
            } else {
                cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, nullptr,
                        Kcur, Vcur, Qcur, mask_l, n_tokens, kv_head, n_kv, kq_scale, cb, il, nullptr,
                        is_swa ? hparams.n_swa : 0);
            }
            cb(cur, "attn_out_proj", il);
        }

        if (layer.attn_post_norm) {
            cur = llm_build_norm(ctx0, cur, hparams, layer.attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_post_norm", il);
        }

        if (il == n_layer - 1 && n_tokens > 1) {
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        ggml_tensor * ffn_inp = cur;

        if (layer.ffn_gate_inp) {
            cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);
            cur = pxqn_rot_in(cur, il, PXQN_SITE_FFN_IN);   // PXQN ffn_in (gate_inp + exps sit on it; identity unless the file rotates the site)
            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    scale_w, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    cb, il, gf);
        } else {
            // the ffn norm goes inside llm_build_ffn (it rotates the PXQN ffn_in / down_in sites itself)
            cur = llm_build_ffn(ctx0, lctx, layer.ffn_norm, cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        }
        cb(cur, "ffn_out", il);

        if (layer.ffn_post_norm) {
            cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
