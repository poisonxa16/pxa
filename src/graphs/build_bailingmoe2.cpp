#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_bailingmoe2() {
    ggml_cgraph * gf = new_graph_custom();
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    //const int64_t n_embd_head = hparams.n_embd_head_v;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // PXA_BAILING_MTP: when running an MTP op (draft / warmup / update-accepted), build the
    // NextN tail-only graph and return early — mirrors build_glm4_moe()/build_qwen35moe().
    if (cparams.mtp_op_type != MTP_OP_NONE) {
        ggml_tensor * hidden_states_from_main_model;
        if (cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        } else {
            hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_embd);
        }
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_bailingmoe2_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    // PXA_BAILING_MTP: when MTP is enabled (-mtp), the main pass must emit ALL tokens' hidden
    // states (the MTP context consumes each), so do NOT crop to the last token -> inp_out_ids = null.
    ggml_tensor * inp_out_ids = (n_tokens > 1 && !lctx.cparams.mtp) ? build_inp_out_ids() : nullptr;

    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;

    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wqkv, model.layers[il].bqkv,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.0f, il);

            if (rope_cache) {
                Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }
        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpSA);
        cb(sa_out, "sa_out", il);

        // MoE branch
        cur = llm_build_norm(ctx0, sa_out, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        // MoE branch
        cur = llm_build_norm(ctx0, sa_out, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {

            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);

    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// PXA_BAILING_MTP: NextN / MTP tail graph for bailingmoe2 (Ling/Ring family).
// Mirrors build_glm4_moe_mtp() — bailing's NextN head is the same DeepSeek-V3-lineage layout
// (enorm/hnorm/eh_proj fuse the embedded next token with the main model's hidden state, then a
// single bailing transformer block = fused-QKV attention + grouped MoE + shared expert, then a
// shared_head_norm + shared_head_head projection to the draft logits). The bailing block ops are
// reused verbatim: build_std_attention() takes the fused wqkv + attn_q_norm/attn_k_norm + NEOX rope
// path (its non-gated else-branch, since arch != QWEN35*), and llm_build_std_moe_ffn() routes through
// llm_build_moe_ffn() which auto-applies bailing's grouped-topk expert routing + expert_weights_norm/
// scale + exp_probs_b. This is intentionally the SINGLE-DEVICE / -sm-layer clean path (no per-device
// nextn mirror, unlike qwen35moe_mtp); -sm graph/attn tensor-parallel MTP for bailing is future work.
struct ggml_tensor * llm_build_context::build_bailingmoe2_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos) {

    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    // Embed the (drafted) next token. Prefer the per-block nextn.embed_tokens if present, else the
    // model's shared tok_embd (Ring shares the input embedding with the NextN module).
    ggml_tensor * mtp_embd_weights = mtp_layer.nextn.embed_tokens;
    if (mtp_embd_weights == nullptr) {
        mtp_embd_weights = model.tok_embd;
    }
    ggml_tensor * token_emb = build_inp_embd_mtp(mtp_embd_weights);

    // NextN input fusion: norm(token_emb) ++ norm(prev_hidden) -> eh_proj  (eh_proj: [2*n_embd, n_embd])
    ggml_tensor * token_emb_norm   = llm_build_norm(ctx0, token_emb,       hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
    cb(combined, "mtp_concat", il);
    ggml_tensor * cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    cb(cur, "mtp_fused", il);

    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!mtp_layer.wqkv || !mtp_layer.wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wqkv=%p, wo=%p)\n",
                __func__, il, (void*)mtp_layer.wqkv, (void*)mtp_layer.wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // Bailing transformer block: attn_norm -> fused-QKV attn (q/k norm + NEOX rope) -> +residual.
    // build_std_attention with add_input=true does the pre-norm + the attention-output residual add
    // internally (the bailing main loop's sa_out = attn + inpSA), and threads inp_out_ids per the
    // glm4_moe_mtp convention so row-cropping stays device-coherent.
    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, inp_out_ids, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, false, nullptr);

    // Bailing MoE FFN + shared expert (+ second residual). llm_build_std_moe_ffn applies ffn_norm,
    // the routed MoE (grouped-topk for bailing via llm_build_moe_ffn), the shared expert, and the
    // residual add (add_input=true) — exactly the bailing main-loop MoE branch.
    cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
            mtp_layer.ffn_gate_inp,  nullptr,
            mtp_layer.ffn_up_exps,   nullptr,
            mtp_layer.ffn_gate_exps, nullptr,
            mtp_layer.ffn_down_exps, nullptr,
            mtp_layer.ffn_exp_probs_b,
            mtp_layer.ffn_up_shexp,    nullptr,
            mtp_layer.ffn_gate_shexp,  nullptr,
            mtp_layer.ffn_down_shexp,  nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps);

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    cb(cur, "result_norm", -1);

    // NextN output head: shared_head_norm -> shared_head_head (falls back to the main LM head when
    // the GGUF ties them, like GLM-4.6 / some Ring exports).
    ggml_tensor * mtp_head_weights = mtp_layer.nextn.shared_head_head;
    if (mtp_head_weights == nullptr) {
        mtp_head_weights = model.output;
    }
    cur = build_output(lctx, ctx0, cur, mtp_head_weights, mtp_layer.nextn.shared_head_norm, cb);
    cb(cur, "result_output", -1);

    return cur;
}
