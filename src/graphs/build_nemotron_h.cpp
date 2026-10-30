// PXA Nemotron port: Nemotron-H hybrids (mainline src/models/nemotron-h.cpp, MIT, see NOTICE).
// Every block is pre-norm (RMS) + residual and is ONE of: a Mamba-2 mixer (delta_net::build_layer_mamba2),
// an attention mixer without positional encoding, or a relu^2 MLP / sigmoid-routed relu^2 MoE with a shared
// expert. The block kind comes from the per-layer hparams (see create_nemotron_h_tensors).
#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

#include <functional>

ggml_cgraph * llm_build_context::build_nemotron_h() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // the sigmoid-routed relu^2 MoE + shared expert of an FFN block (trunk blocks and the NextN head)
    auto build_moe = [&](ggml_tensor * cur, const llama_layer & layer, int il) -> ggml_tensor * {
        // Nemotron 3.5 Bolt (Puzzle): the layer's own expert count / used count, and the latent MoE -- the router
        // reads the full-width x, the routed experts run on latent_down(x), latent_up maps their weighted sum back.
        // A file without latent tensors and with scalar expert keys builds exactly the previous graph.
        // PXA_XCACHE splits the stack into hot + cold: the routed count is the file's, carried by the layer's map
        // ([map: n_expert][counters: n_expert][stats: 8]); the hot stack's ne[2] is only its slot count
        const int64_t n_expert_l = layer.ffn_exps_xmap ? (layer.ffn_exps_xmap->ne[0] - 8) / 2 : layer.ffn_up_exps->ne[2];
        // warmup runs every expert (n_expert_used == n_expert there): keep that per layer
        const int64_t n_used_l   = n_expert_used >= (int64_t) hparams.n_expert ? n_expert_l
                                 : hparams.n_expert_used_arr[il] ? (int64_t) hparams.n_expert_used_arr[il] : (int64_t) n_expert_used;
        ggml_tensor * moe_in = cur;
        ggml_tensor * logits = nullptr;
        if (layer.ffn_latent_down) {
            logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, cur);
            cb(logits, "ffn_moe_logits", il);
            moe_in = llm_build_lora_mm(lctx, ctx0, layer.ffn_latent_down, cur);
            cb(moe_in, "ffn_latent_in", il);
        }
        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, moe_in,
                logits ? nullptr : layer.ffn_gate_inp,
                layer.ffn_up_exps,
                nullptr,               // no gate
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert_l, n_used_l,
                LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                hparams.expert_weights_scale != 0.0f, hparams.expert_weights_scale,
                LLM_EXPERT_GATING_FUNC_SIGMOID,
                cb, il, gf, false, nullptr, nullptr, logits);
        if (layer.ffn_latent_up) {
            moe_out = llm_build_lora_mm(lctx, ctx0, layer.ffn_latent_up, moe_out);
            cb(moe_out, "ffn_latent_out", il);
        }
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                layer.ffn_up_shexp,   NULL, NULL,
                NULL,                 NULL, NULL,
                layer.ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, cb, il);
        cb(shexp, "ffn_shexp", il);

        return ggml_add(ctx0, moe_out, shexp);
    };

    // PXA_NEMOTRON_MTP: the NextN draft head runs on its own graph (the MTP companion context): one block
    // conditioned on the trunk's final hidden (result_norm, i.e. after output_norm) of the previous position
    // and the embedding of the token that follows it.
    if (cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ASSERT(hparams.nextn_predict_layers == 1 && "nemotron_h MTP: exactly one NextN block is implemented");
        ggml_tensor * prev_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_name(prev_hidden, "inp_mtp_states");
        ggml_set_input(prev_hidden);
        lctx.inp_mtp_states = prev_hidden;

        ggml_tensor * head = build_nemotron_h_mtp(gf, prev_hidden, kq_scale,
                [&](ggml_tensor * x, const llama_layer & l, int i) { return build_moe(x, l, i); });
        ggml_build_forward_expand(gf, head);
        return gf;
    }

    delta_net mixer(lctx, batch);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // the recurrent-state routing inputs the Mamba-2 layers read (filled by llama_set_inputs, shared with DeltaNet)
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);
    lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
    ggml_set_input(lctx.inp_conv_seq_map);
    lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
    ggml_set_input(lctx.inp_qnext_state_mask);



    // PXA_NEMOTRON_MTP: with a NextN head loaded (-mtp) the tail block is the head's, not the trunk's, and the
    // head conditions on EVERY batch row's final hidden, so the trunk keeps all rows (as qwen35 does) unless the
    // lazy prompt warm-up is on for a prompt-sized batch. Without -mtp both reduce to the previous graph.
    const int  n_trunk  = n_layer - (int) hparams.nextn_predict_layers;
    const bool all_rows = lctx.cparams.mtp && !(llama_pxa_mtp_lazy_warmup() && n_tokens > 64);

    for (int il = 0; il < n_trunk; ++il) {
        ggml_tensor * inpSA = inpL;
        const auto & layer = model.layers[il];

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = mixer.build_layer_mamba2(ctx0, gf, cur, il, cb);
        } else if (hparams.n_ff(il) == 0) {
            const int64_t n_head_il    = hparams.n_head(il);
            const int64_t n_head_kv_il = hparams.n_head_kv(il);
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, layer.wq, layer.bq,
                    layer.wk, layer.bk, layer.wv, layer.bv, 0.f, il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_il,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, layer.bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            cb(cur, "attn_out", il);
        } else if (layer.ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up,   layer.ffn_up_b,   NULL,
                    NULL,           NULL,             NULL,
                    layer.ffn_down, layer.ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_moe(cur, layer, il);
            cb(cur, "ffn_out", il);
        }

        if (il == n_trunk - 1 && n_tokens > 1 && !all_rows) {
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);
    // PXA_NEMOTRON_MTP: this row is the MTP head's conditioning hidden; the read-back needs it as an output
    if (lctx.cparams.mtp) {
        ggml_set_output(cur);
    }

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// PXA_NEMOTRON_MTP (2026-10-17): the Nemotron 3.5 NextN / MTP draft head (blk.<n_layer-1>; mainline
// src/models/nemotron-h-moe.cpp graph_mtp, MIT, see NOTICE):
//
//     x   = eh_proj( RMSNorm(embd(tok), enorm) || RMSNorm(h_prev, hnorm) )        [2*n_embd -> n_embd]
//     x   = x + Attn_NoPE( RMSNorm(x, attn_norm) )
//     x   = x + MoE_latent( RMSNorm(x, post_attention_norm) )                     (the trunk's MoE block)
//     h   = LayerNorm(x, shared_head_norm)          -> the next draft step's h_prev ("result_norm")
//     out = output(h)                                  (the trunk's lm_head; no output_norm on top)
//
// The head's final norm is a LayerNorm (mean-centred, eps = layer_norm_epsilon), not the trunk's RMSNorm.
ggml_tensor * llm_build_context::build_nemotron_h_mtp(ggml_cgraph * gf, ggml_tensor * prev_hidden, float kq_scale,
        const std::function<ggml_tensor *(ggml_tensor *, const llama_layer &, int)> & build_moe) {
    const int il = (int) hparams.n_layer - 1;
    const auto & layer = model.layers[il];
    const int64_t n_embd_head = hparams.n_embd_head_v(0);

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.shared_head_norm &&
                "nemotron_h MTP: blk.<n_layer-1>.nextn.* missing -- load the head with -mtp (PXA_MTP_HEAD_FILE for a "
                "separate mtp-*.gguf)");
    GGML_ASSERT(layer.wq && layer.wk && layer.wv && layer.wo && layer.attn_post_norm && layer.ffn_up_exps &&
                "nemotron_h MTP: the head block's attention / MoE tensors are missing");
    GGML_ASSERT(il < (int) kv_self.k_l.size() && kv_self.k_l[il] && kv_self.v_l[il] &&
                "nemotron_h MTP: no KV cache for the head block");

    ggml_tensor * KQ_mask = build_inp_KQ_mask();
    // one output row per requested row (the stable layout every MTP builder uses); none when nothing is read
    ggml_tensor * inp_out_ids = n_outputs > 0 ? build_inp_out_ids() : nullptr;

    ggml_tensor * tok = build_inp_embd_mtp(model.tok_embd);
    ggml_tensor * e = llm_build_norm(ctx0, tok,         hparams, layer.nextn.enorm, nullptr, LLM_NORM_RMS, cb, il);
    ggml_tensor * h = llm_build_norm(ctx0, prev_hidden, hparams, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, cb, il);
    ggml_tensor * cur = llm_build_lora_mm(lctx, ctx0, layer.nextn.eh_proj, ggml_concat(ctx0, e, h, 0));
    cb(cur, "mtp_fused", il);

    // dense NoPE attention, into the head block's own K/V
    ggml_tensor * inpSA = cur;
    cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);
    {
        auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, layer.wq, layer.bq,
                layer.wk, layer.bk, layer.wv, layer.bv, 0.f, il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il),    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, layer.bo,
                Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        cb(cur, "attn_out", il);
    }
    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    // only the rows that are read go through the MoE and the lm_head
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    ggml_tensor * ffn_residual = cur;
    cur = llm_build_norm(ctx0, cur, hparams, layer.attn_post_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "ffn_norm", il);
    cur = build_moe(cur, layer, il);
    cb(cur, "ffn_out", il);
    cur = ggml_add(ctx0, cur, ffn_residual);
    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "l_out", il);

    cur = llm_build_norm(ctx0, cur, hparams, layer.nextn.shared_head_norm, nullptr, LLM_NORM, cb, -1);
    // PXA_MTP_READBACK_v1: the next draft step's conditioning hidden is read back by name -- keep it an output
    if (llama_pxa_mtp_head_output()) {
        ggml_set_output(cur);
    }
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    return cur;
}
