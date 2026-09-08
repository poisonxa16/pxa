#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

// PXA_MTP_LAZY_WARMUP_v1 (level resolved by llama_pxa_mtp_lazy_warmup(), include/llama.h):
// with lazy warmup on, nothing consumes all-token MTP hidden rows on prompt-sized batches (the
// companion warmup is skipped), so the out_ids row-slice can be re-enabled — restoring "last layer + output norm +
// lm_head compute output rows only" on prefill. That full-vocab lm_head over every prompt token
// was the dominant structural MTP prefill tax. Gen/verify batches (n_tokens <= 64) keep full
// rows for the accept path, exactly as before. Eager mode (lazy resolved off) is byte-unchanged.
static bool pxa_mtp_lazy_out_ids(int64_t n_tokens) {
    const bool lazy = llama_pxa_mtp_lazy_warmup();
    return lazy && n_tokens > 64;
}


ggml_cgraph * llm_build_context::build_qwen3next() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * cur = nullptr;

    // PXA qwen3next-MTP: when running the MTP head (draft/warmup/update), build ONLY the MTP tail
    // from the main model's last-layer hidden state (mirror build_qwen35moe). Otherwise build the
    // normal delta-net + full-attention trunk.
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

        cur = build_qwen3next_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        // PXA: when cparams.mtp is on, the main-model pass must keep its full hidden state (no out-id
        // gather) so the MTP head can read result_mtp_embd, exactly like build_qwen35moe.
        ggml_tensor * inp_out_ids = (n_tokens > 1 && (!lctx.cparams.mtp || pxa_mtp_lazy_out_ids(n_tokens))) ? build_inp_out_ids() : nullptr; // PXA_MTP_LAZY_WARMUP_v1
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        // PXA_LLAMA_FIX_v4: conv seq-map [n_kv=n_tokens, n_tokens] for the ONE-batched mixed-seq delta-net path.
        // Row 0 of column t = state column for token t (identity, since states are gathered in token order); rows >=1 = -1.
        lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
        cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
        ggml_set_input(lctx.inp_conv_seq_map);

        // PXA_LLAMA_FIX_v4: per-seq recurrent-state reset mask (0=reset to zero, 1=keep). One column per token/seq.
        lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
        ggml_set_input(lctx.inp_qnext_state_mask);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        // PXA qwen3next-MTP: bound the trunk to the main (non-MTP) layers; the trailing
        // nextn_predict_layers are built separately by build_qwen3next_mtp.
        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            GGML_ASSERT(model.layers[il].attn_norm != nullptr);
            GGML_ASSERT(model.layers[il].attn_post_norm != nullptr);

            const bool has_moe = model.layers[il].ffn_gate_inp != nullptr;
            const bool has_dense = model.layers[il].ffn_gate != nullptr &&
                                   model.layers[il].ffn_up != nullptr &&
                                   model.layers[il].ffn_down != nullptr;
            GGML_ASSERT(has_moe || has_dense);
            if (has_moe) {
                GGML_ASSERT(model.layers[il].ffn_up_exps != nullptr);
                GGML_ASSERT(model.layers[il].ffn_gate_exps != nullptr);
                GGML_ASSERT(model.layers[il].ffn_down_exps != nullptr);
            }

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, false);
            }

            if (!model.layers[il].ffn_gate_inp) {
                // dense FFN
                cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                        model.layers[il].ffn_up,   nullptr, nullptr,
                        model.layers[il].ffn_gate, nullptr, nullptr,
                        model.layers[il].ffn_down, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
                cb(cur, "ffn_out", il);
            } else {
                cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                        model.layers[il].ffn_gate_inp,  nullptr,
                        model.layers[il].ffn_up_exps,   nullptr,
                        model.layers[il].ffn_gate_exps, nullptr,
                        model.layers[il].ffn_down_exps, nullptr,
                        nullptr,
                        model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                        model.layers[il].ffn_gate_shexp,  nullptr,
                        model.layers[il].ffn_down_shexp,  nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true, false, 0.0f,
                        LLM_EXPERT_GATING_FUNC_SOFTMAX,
                        LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps, nullptr, model.layers[il].ffn_gate_inp_shexp);
            }

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        // PXA qwen3next-MTP: expose the main model's last-layer hidden state for the MTP head
        // (mirror build_qwen35moe). The companion MTP pass reads this via "inp_mtp_states".
        if (lctx.cparams.mtp) {
            cb(inpL, "result_mtp_embd", -1);
            ggml_set_output(inpL);
        }

        cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
        cb(cur, "result_output", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// PXA qwen3next-MTP: near-copy of build_qwen35moe_mtp (concat enorm/hnorm -> eh_proj -> std attention
// -> MoE FFN -> build_output via output_mtp + shared_head_norm), KEEPING the MTP-SPLIT-DEV0-FIX
// inp_out_ids threading and the mtp_split tensor-parallel path (needed for the 3-card -sm topology).
// THE ONE DIFFERENCE: qwen3next is NEOX rope (NOT IMROPE) -> build_std_attention is called with
// is_multi=false (matching the qwen3next trunk). Using is_multi=true here would silently apply
// IMROPE and drop draft acceptance to ~0.
struct ggml_tensor * llm_build_context::build_qwen3next_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos) {

    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    // MTP-UNIFORM (see build_qwen35moe_mtp): when the MTP head runs tensor-parallel (-sm attn/graph,
    // >1 device), build the FULL fused hidden PER DEVICE from per-device mirror replicas of
    // enorm/hnorm/eh_proj, tied together with a reduce-OFF container so each device gets its OWN full
    // fused hidden. Gate on eh_proj->extra (split active); single-GPU / -sm layer / non-split falls
    // through to the original single-device fusion below.
    const bool mtp_split =
        mtp_layer.nextn.eh_proj && mtp_layer.nextn.eh_proj->extra &&
        mtp_layer.nextn.enorm   && mtp_layer.nextn.enorm->extra   &&
        mtp_layer.nextn.hnorm   && mtp_layer.nextn.hnorm->extra;

    ggml_tensor * cur;
    if (mtp_split) {
        auto eh   = (const ggml_split_tensor_t *)mtp_layer.nextn.eh_proj->extra;
        auto en   = (const ggml_split_tensor_t *)mtp_layer.nextn.enorm->extra;
        auto hn   = (const ggml_split_tensor_t *)mtp_layer.nextn.hnorm->extra;
        GGML_ASSERT(eh->n_device == en->n_device && eh->n_device == hn->n_device);
        std::vector<ggml_tensor *> fused(eh->n_device, nullptr);
        int nhave = 0;
        for (int id = 0; id < eh->n_device; ++id) {
            auto eh_id = eh->splits[id];
            auto en_id = en->splits[id];
            auto hn_id = hn->splits[id];
            GGML_ASSERT((eh_id && en_id && hn_id) || (!eh_id && !en_id && !hn_id));
            if (!eh_id) continue;
            int il_cb = 1000*(id+1) + il;
            ggml_tensor * te_norm = llm_build_norm(ctx0, token_emb,       hparams, en_id, NULL, LLM_NORM_RMS, cb, il_cb);
            ggml_tensor * hs_norm = llm_build_norm(ctx0, prev_embeddings, hparams, hn_id, NULL, LLM_NORM_RMS, cb, il_cb);
            ggml_tensor * combined = ggml_concat(ctx0, te_norm, hs_norm, 0);
            cb(combined, "mtp_concat", il_cb);
            ggml_tensor * f = llm_build_lora_mm(lctx, ctx0, eh_id, combined);
            cb(f, "mtp_fused", il_cb);
            ggml_build_forward_expand(gf, f);
            fused[id] = f;
            ++nhave;
        }
        GGML_ASSERT(nhave > 1); // tensor-parallel MTP head requires >=2 participating devices
        cur = ggml_reduce(ctx0, fused.data(), eh->n_device, GGML_OP_ADD);
        cur->op_params[3] = 1; // turn the reduce OFF (container only)
        ggml_build_forward_expand(gf, cur);
        cb(cur, "mtp_fused_reduce", il);
    } else {
        ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
        ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

        if (mtp_layer.nextn.eh_proj != nullptr) {
            ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
            cb(combined, "mtp_concat", il);
            cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
        } else {
            cur = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
        }
        cb(cur, "mtp_fused", il);
    }

    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!mtp_layer.wq || !mtp_layer.wk || !mtp_layer.wv || !mtp_layer.wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wq=%p, wk=%p, wv=%p, wo=%p)\n",
                __func__, il, (void*)mtp_layer.wq, (void*)mtp_layer.wk,
                (void*)mtp_layer.wv, (void*)mtp_layer.wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // MTP-SPLIT-DEV0-FIX (see build_qwen35moe_mtp): thread inp_out_ids INTO build_std_attention so the
    // row-select happens PER-DEVICE inside the attention reduce, keeping the output a per-device REDUCE
    // for the split MoE FFN. NEOX rope -> is_multi=false (qwen3next is NOT IMROPE).
    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, inp_out_ids, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, false, nullptr);

    cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
            mtp_layer.ffn_gate_inp,  nullptr,
            mtp_layer.ffn_up_exps,   nullptr,
            mtp_layer.ffn_gate_exps, nullptr,
            mtp_layer.ffn_down_exps, nullptr,
            nullptr,
            mtp_layer.ffn_up_shexp,    nullptr,
            mtp_layer.ffn_gate_shexp,  nullptr,
            mtp_layer.ffn_down_shexp,  nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true, false, 0.0f,
            LLM_EXPERT_GATING_FUNC_SOFTMAX,
            LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps, nullptr, mtp_layer.ffn_gate_inp_shexp);

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output_mtp, mtp_layer.nextn.shared_head_norm, cb, false);
    cb(cur, "result_output", -1);

    return cur;
}
