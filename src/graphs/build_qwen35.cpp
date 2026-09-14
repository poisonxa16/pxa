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
// PXA_MTP_STABLE_OUT_IDS (=0 disables): keep the MTP head's graph SHAPE constant across the two
// batches a speculative cycle alternates between. The head decodes an N-row update pass (which
// asks for fewer output rows than it has, so it carries an out_ids row-slice) and then a 1-row
// draft pass (which asks for one row out of one, so the slice was skipped). Two node counts on one
// context means the graph allocator's single plan can never cover both: it re-plans - a full
// backend drain and a compute-buffer reallocation - on every pass, twice per accepted token.
//
// Building the slice unconditionally costs two extra get_rows on the 1-row pass, and at one row
// out of one the slice is the identity, so the values are unchanged; what it buys is one shape,
// one plan, and no re-plan at all in steady state.
static bool pxa_mtp_stable_out_ids() {
    static const bool on = [] {
        const char * e = getenv("PXA_MTP_STABLE_OUT_IDS");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

static bool pxa_mtp_lazy_out_ids(int64_t n_tokens) {
    const bool lazy = llama_pxa_mtp_lazy_warmup();
    if (lazy && n_tokens > 64) {
        static bool logged = false;
        if (!logged) { fprintf(stderr, "PXA_MTP_LAZY_OUT_IDS: active (n_tokens=%lld)\n", (long long) n_tokens); logged = true; }
        return true;
    }
    return false;
}


ggml_cgraph * llm_build_context::build_qwen35moe() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * cur = nullptr;

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // PXA_MTP_BATCH_SLOTS_ROWS_v1: one hidden row per BATCH token, for every MTP op type.
        // The draft-gen graph used to allocate a single [n_embd] row here because every draft decode
        // was a 1-row batch; a multi-row draft step then concatenated it against [n_embd, n_tokens]
        // token embeddings and aborted. See common/pxa-mtp-batch-slots.h. No-op at n_tokens == 1.
        ggml_tensor * hidden_states_from_main_model =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_qwen35moe_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
        if (cur == nullptr) {
            // PXA_MTP_DRAFT_CACHE_ONLY: the K/V cache writes are already expanded into the graph and
            // there is no result tensor by design (the decode requested zero outputs).
            return gf;
        }
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        ggml_tensor * inp_out_ids = (n_tokens > 1 && (!lctx.cparams.mtp || pxa_mtp_lazy_out_ids(n_tokens))) ? build_inp_out_ids() : nullptr; // PXA_MTP_LAZY_WARMUP_v1
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        // PXA_RS_RING: when armed the tensor carries TWO index vectors -- [0, n_tokens) the write rows
        // (plane 0, absolute seq_id) and [n_tokens, 2*n_tokens) the read rows. Ring off -> unchanged.
        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens * (lctx.kv_self.rs_ring_armed() ? 2 : 1));
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        // PXA_LLAMA_FIX_v4: conv seq-map [n_kv=n_tokens, n_tokens] for the ONE-batched mixed-seq delta-net path.
        lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
        cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
        ggml_set_input(lctx.inp_conv_seq_map);

        // PXA_LLAMA_FIX_v4: per-seq recurrent-state reset mask (0=reset to zero, 1=keep).
        lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
        ggml_set_input(lctx.inp_qnext_state_mask);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
            }

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

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

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

ggml_cgraph * llm_build_context::build_qwen35() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // MTP tail-only graph
        // PXA_MTP_BATCH_SLOTS_ROWS_v1: one hidden row per BATCH token, for every MTP op type.
        // The draft-gen graph used to allocate a single [n_embd] row here because every draft decode
        // was a 1-row batch; a multi-row draft step then concatenated it against [n_embd, n_tokens]
        // token embeddings and aborted. See common/pxa-mtp-batch-slots.h. No-op at n_tokens == 1.
        ggml_tensor * hidden_states_from_main_model =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_qwen35_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
        if (cur == nullptr) {
            return gf; // PXA_MTP_DRAFT_CACHE_ONLY: see build_qwen35moe()
        }
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        ggml_tensor * inp_out_ids = (n_tokens > 1 && (!lctx.cparams.mtp || pxa_mtp_lazy_out_ids(n_tokens))) ? build_inp_out_ids() : nullptr; // PXA_MTP_LAZY_WARMUP_v1
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        // PXA_RS_RING: when armed the tensor carries TWO index vectors -- [0, n_tokens) the write rows
        // (plane 0, absolute seq_id) and [n_tokens, 2*n_tokens) the read rows. Ring off -> unchanged.
        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens * (lctx.kv_self.rs_ring_armed() ? 2 : 1));
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        // PXA_LLAMA_FIX_v4: conv seq-map [n_kv=n_tokens, n_tokens] for the ONE-batched mixed-seq delta-net path.
        lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
        cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
        ggml_set_input(lctx.inp_conv_seq_map);

        // PXA_LLAMA_FIX_v4: per-seq recurrent-state reset mask (0=reset to zero, 1=keep).
        lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
        ggml_set_input(lctx.inp_qnext_state_mask);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = nullptr;

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
            }

            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        if (lctx.cparams.mtp) {
            //struct ggml_tensor * embd_copy = ggml_dup(ctx0, inpL);
            //cb(embd_copy, "result_mtp_embd", -1);
            //ggml_set_output(embd_copy);
            cb(inpL, "result_mtp_embd", -1);
            ggml_set_output(inpL);
        }

        cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
        cb(cur, "result_output", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_tensor * llm_build_context::build_qwen35moe_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos) {

    const int il = hparams.n_layer - 1;

    // PXA_MTP_DRAFT_CACHE_ONLY: a K/V-only refresh builds no attention, so it needs no KQ mask --
    // and not building it also removes the O(n_tokens * n_kv) host-side mask fill from the step.
    const bool store_only = cparams.mtp_op_type == MTP_OP_KV_ONLY;

    struct ggml_tensor * KQ_mask = store_only ? nullptr : build_inp_KQ_mask();
    const bool want_out_ids = !store_only && n_outputs > 0 &&
                              (pxa_mtp_stable_out_ids() ? true : (n_tokens > 1 && n_outputs < n_tokens));
    struct ggml_tensor * inp_out_ids = want_out_ids ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    // MTP-UNIFORM: when the MTP head runs tensor-parallel (-sm attn/graph, >1 device), the input
    // fusion (enorm/hnorm/eh_proj) used to run on ONE device, producing a single-device "island"
    // whose output was then read by the row-split attention/MoE on the OTHER devices -> CUDA illegal
    // memory access during the cross-device read. Robust fix: build the FULL fused hidden PER DEVICE,
    // each device using its own mirror replica of enorm/hnorm/eh_proj (loaded by the split-buffer's
    // mirror set_tensor) + a per-device copy of the token-embd / prev-hidden inputs, then hand the
    // result to the existing split attention. The per-device tensors are tied together with a
    // reduce-OFF GGML_OP_REDUCE node: that pins each fused[id] subgraph to backend `id` (scheduler
    // assigns src[j] -> backend j) and is a NO-OP at compute time (ggml_cuda_op_reduce returns early
    // when op_params[3]==1), so get_input_tensor_sm_graph() in build_std_attention hands each device
    // its OWN full fused hidden (input->src[id]) — NO single-device tensor is read by a per-device op.
    // Gate: eh_proj->extra (split active). When null (single-GPU / -sm layer / non-split) -> the exact
    // original single-device fusion below, so there is NO regression for those paths.
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
            // FULL fused hidden computed locally on device `id` from its own replicas.
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
        // reduce-OFF container: holds the per-device full fused hidden, pins each to its backend,
        // computes to a no-op. build_std_attention reads input->src[id] per device.
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

    // MTP-SPLIT-DEV0-FIX: thread inp_out_ids INTO build_std_attention (applied PER-DEVICE inside the
    // attention reduce, exactly like every normal transformer layer in build_qwen35moe()), instead of
    // applying an EXTERNAL ggml_get_rows() on the returned attention REDUCE. The external get_rows
    // collapsed the per-device REDUCE (whose src[id] slices the split MoE FFN reads locally via
    // get_input_tensor_sm_graph) into a SINGLE-device GGML_OP_GET_ROWS tensor; the row-split MoE on the
    // OTHER devices then read that single-device tensor cross-device -> CUDA illegal memory access on
    // device 0 during the MTP-tail decode. Passing inp_out_ids through keeps the attention output a
    // proper per-device reduce (row-selected per device + residual row-selected per device), so the MoE
    // FFN sees a REDUCE input and stays concurrency/device-coherent. Intrinsic to the split head (NOT an
    // offload artifact): the cross-device read exists with or without --n-cpu-moe.
    if (store_only) {
        // K/V projection + rotation + cache write, and nothing else. The head's own KV is what the
        // next real draft attends to; the logits and hidden rows this pass would otherwise produce
        // are not read by anyone (see common/pxa-mtp-cache-only.h for the gate).
        build_std_attention(gf, mtp_layer.attn_norm, cur,
                inp_pos, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr, /* store_only = */ true);
        return nullptr;
    }

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, inp_out_ids, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr);

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
    // PXA_MTP_READBACK_v1: this row IS the head's conditioning hidden for the next draft step --
    // mtp_accept_batch and the draft loop both read it back with llama_get_embeddings_ith() and
    // feed it to the following MTP_OP_DRAFT_GEN decode. Without the output flag ggml-alloc is free
    // to hand its buffer to a later node once the head's own lm_head has consumed it, and the D2H
    // copy in llama_decode() then reads whatever landed there. That is exactly what happened on the
    // multi-row accepted-token commit: the logits were right (they are computed before the reuse)
    // while the hidden came back orthogonal to the true row -- so the free carried token was good
    // and every chain conditioned on it collapsed. The trunk graph already protects its own feature
    // row this way (build_qwen35moe/build_qwen35: ggml_set_output(inpL) under "result_mtp_embd");
    // the head's row was the one left unprotected.
    if (llama_pxa_mtp_head_output()) { // PXA_MTP_HEAD_OUTPUT=0 reproduces the stale read-back (debug only)
        ggml_set_output(cur);
    }

    cur = build_output(lctx, ctx0, cur, model.output_mtp, mtp_layer.nextn.shared_head_norm, cb, false);
    cb(cur, "result_output", -1);

    return cur;
}

struct ggml_tensor * llm_build_context::build_qwen35_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos) {

    const int il = hparams.n_layer - 1;

    // PXA_MTP_DRAFT_CACHE_ONLY: see build_qwen35moe_mtp
    const bool store_only = cparams.mtp_op_type == MTP_OP_KV_ONLY;

    struct ggml_tensor * KQ_mask = store_only ? nullptr : build_inp_KQ_mask();

    const bool want_out_ids = !store_only && n_outputs > 0 &&
                              (pxa_mtp_stable_out_ids() ? true : (n_tokens > 1 && n_outputs < n_tokens));
    struct ggml_tensor * inp_out_ids = want_out_ids ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * cur;
    if (mtp_layer.nextn.eh_proj != nullptr) {
        // Full fusion: concat + project (27B, 4B, 2B, 0.8B)
        ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_concat", il);
        cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    } else {
        // 9B — no fc/eh_proj
        cur = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
    }
    cb(cur, "mtp_fused", il);

    // Self-Attention (wq may be shared from main model's last layer)
    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!model.layers[il].wq || !model.layers[il].wk || !model.layers[il].wv || !model.layers[il].wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wq=%p, wk=%p, wv=%p, wo=%p)\n",
                __func__, il, (void*)model.layers[il].wq, (void*)model.layers[il].wk,
                (void*)model.layers[il].wv, (void*)model.layers[il].wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // MTP-SPLIT-DEV0-FIX (see build_qwen35moe_mtp): thread inp_out_ids INTO build_std_attention so the
    // row-select happens PER-DEVICE inside the attention reduce, keeping the output a per-device REDUCE
    // for the split dense FFN (llm_build_ffn also reads input->src[id] via get_input_tensor_sm_graph).
    // The old external ggml_get_rows() collapsed the reduce to a single-device GET_ROWS read cross-device.
    if (store_only) {
        // K/V projection + rotation + cache write, and nothing else
        build_std_attention(gf, mtp_layer.attn_norm, cur,
                inp_pos, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr, /* store_only = */ true);
        return nullptr;
    }

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, inp_out_ids, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr);

    // Dense FFN — optional (9B and 4B don't have FFN in MTP layer)
    if (mtp_layer.ffn_gate != nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up,   NULL, NULL,
                mtp_layer.ffn_gate, NULL, NULL,
                mtp_layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    // As far as I can tell this was wrong. We need the FFN output, and not the normalized result.
    //cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);
    // PXA_MTP_READBACK_v1: this row IS the head's conditioning hidden for the next draft step --
    // mtp_accept_batch and the draft loop both read it back with llama_get_embeddings_ith() and
    // feed it to the following MTP_OP_DRAFT_GEN decode. Without the output flag ggml-alloc is free
    // to hand its buffer to a later node once the head's own lm_head has consumed it, and the D2H
    // copy in llama_decode() then reads whatever landed there. That is exactly what happened on the
    // multi-row accepted-token commit: the logits were right (they are computed before the reuse)
    // while the hidden came back orthogonal to the true row -- so the free carried token was good
    // and every chain conditioned on it collapsed. The trunk graph already protects its own feature
    // row this way (build_qwen35moe/build_qwen35: ggml_set_output(inpL) under "result_mtp_embd");
    // the head's row was the one left unprotected.
    if (llama_pxa_mtp_head_output()) { // PXA_MTP_HEAD_OUTPUT=0 reproduces the stale read-back (debug only)
        ggml_set_output(cur);
    }

    //cur = build_output(lctx, ctx0, cur, model.output, nullptr, cb);
    cur = build_output(lctx, ctx0, cur, model.output_mtp, mtp_layer.nextn.shared_head_norm, cb, false);
    cb(cur, "result_output", -1);

    return cur;
}
