#include <algorithm>

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
//
// The block structure is not the usual norm -> op -> residual. Instead the residual is WIDE:
// hc parallel streams of n_embd carried as [n_embd, hc, n_tokens]. A low-rank mixer collapses
// the streams into the single [n_embd, n_tokens] the token mixer and the MoE consume, and a
// scatter puts each block output back with a per-stream weight. The mixer IS the layer norm --
// this arch has no attn_norm, no ffn_norm and no output_norm at all.
//
// Reused from the rest of this fork rather than rewritten:
//   * the fused Gated DeltaNet kernel (delta_net, hc_mode) on the linear-attention layers
//   * build_std_attention (gated Q projection + q/k norms + IMRoPE) on the full-attention layers
//   * llm_build_std_moe_ffn for the expert FFN
// Both helpers normally fold in their own norm and residual add; hc_mode / a null norm weight /
// add_input=false turn those off so the hyper-connection owns them.

ggml_tensor * llm_build_context::build_qwen4exp_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // Grouped RMSNorm: the reduction is over ONE stream (ne[0] == n_embd), then a single
    // [hc_dim] gamma scales all of them. The converter folded each gamma to (1 + w).
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo   = llm_build_lora_mm(lctx, ctx0, w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f/(float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, llm_build_lora_mm(lctx, ctx0, w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // Collapse the streams by their mean. Summing hc strided views beats a permute + sum_rows
    // because hc is 4.
    ggml_tensor * mixed = ggml_cont(ctx0,
            ggml_view_2d(ctx0, gated, n_embd, nt, ggml_row_size(gated->type, n_embd)*hc, 0));
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd)*hc,
                ggml_row_size(gated->type, n_embd)*c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f/(float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = llm_build_lora_mm(lctx, ctx0, w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llm_build_context::build_qwen4exp_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add.
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f/(float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}


// PLE: per-layer n-gram hash embeddings, a side path that runs on the layers named by
// <arch>.ple.layers (layer 1 alone in the shipped model) and adds into the wide residual.
//
// The gather table per_layer_token_embd is 95.4 GiB in the shipped model - larger than all
// six cards put together - so it is meant to be pinned to host RAM with
//   -ot per_layer_token_embd=CPU
// The row indices come from inp_ple_rows, hashed host-side in llama_set_inputs.
ggml_tensor * llm_build_context::build_qwen4exp_ple(ggml_cgraph * gf, ggml_tensor * hidden, int il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc*n_embd;
    const int64_t n_heads = hparams.ple_n_heads;
    const int64_t kern    = hparams.ple_conv_kernel;

    GGML_ASSERT(model.tok_embd_per_layer != nullptr);

    // gather, then flatten the heads: get_rows lays the head dimension out slowest
    ggml_tensor * emb = ggml_get_rows(ctx0, model.tok_embd_per_layer, lctx.inp_ple_rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim*n_heads, n_tokens);
    cb(emb, "ple_embd", il);

    ggml_tensor * key   = llm_build_lora_mm(lctx, ctx0, model.layers[il].ple_key,   emb);
    ggml_tensor * value = llm_build_lora_mm(lctx, ctx0, model.layers[il].ple_value, emb);

    // both norms reduce over ONE hc stream and scale with a weight over the whole hc*n_embd
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a SIGNED square root before the sigmoid
    ggml_tensor * sc = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    sc = ggml_scale(ctx0, sc, 1.0f/sqrtf((float) n_embd));
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, sc), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, sc), mag));
    cb(gate, "ple_gate", il);

    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);
    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens), model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // The conv input of the PREVIOUS `hist` positions is not recomputable here: it depends on
    // the residual at those positions, which is gone. So every live sequence's window is
    // carried in ple_conv_hist_dev and prepended below.
    ggml_set_name(normalized, "ple_conv_in");
    ggml_build_forward_expand(gf, normalized);

    GGML_ASSERT(lctx.ple_conv_hist_dev != nullptr && lctx.inp_ple_conv_src != nullptr);
    GGML_ASSERT(lctx.ple_conv_hist_dev->ne[0] == hc_dim);

    // [hc_dim, n_win + n_tokens]: every sequence's carried window, then this ubatch.
    // Column 0 of the window part is the permanent zero column (see llama-context.h), so a
    // gather that finds no source position resolves to zeros - which is exactly what an
    // all-EOS prefix convolves to, and what a fresh sequence must see.
    const int64_t n_win  = lctx.ple_conv_hist_dev->ne[1];
    ggml_tensor * padded = ggml_concat(ctx0, lctx.ple_conv_hist_dev, normalized, 1);
    cb(padded, "ple_conv_padded", il);

    // Depthwise causal conv DILATED by the n-gram size, as a sum of shifted copies:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // written this way because ggml_conv_1d_dw is documented as unreliable.
    //
    // Tap k used to be a shifted VIEW of `padded`, which assumes column t-off of the
    // concatenation IS position pos(t)-off of token t's sequence. That holds for one
    // sequence and fails the moment a batch interleaves two, which is what the old
    // GGML_ABORT was guarding. It is a GATHER now: inp_ple_conv_src[k*n_tokens + t] names
    // the column that really holds (seq(t), pos(t)-off), resolved host-side. The cont() the
    // view needed anyway made this free - one materialised [hc_dim, n_tokens] per tap
    // either way.
    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        ggml_tensor * idx_k = ggml_view_1d(ctx0, lctx.inp_ple_conv_src, n_tokens,
                (size_t) k*n_tokens*ggml_element_size(lctx.inp_ple_conv_src));
        ggml_tensor * shifted = ggml_get_rows(ctx0, padded, idx_k);

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1], k*model.layers[il].ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    // A/B switch for attribution. The PLE conv is the one path here whose history is
    // carried across calls by hand, and with 11 graph splits across 6 GPUs that carry is
    // the least-proven code in this port - the single-GPU fixture could never exercise it.
    // Setting PXA_QWEN4EXP_NO_PLE_CONV=1 drops the conv branch entirely, leaving the rest of
    // the PLE side path intact, so a run with and without it isolates the carry.
    static const bool pxa_no_ple_conv = getenv("PXA_QWEN4EXP_NO_PLE_CONV") != nullptr;
    if (pxa_no_ple_conv) {
        conv_out = ggml_scale(ctx0, conv_out, 0.0f);
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    // Carry every sequence's window forward on-device, in one gather + one copy. Dest column
    // 1 + s*hist + j must end up holding (s, new_next_pos(s) - hist + j); the host resolves
    // that to a column of `padded`, which is either a token of this ubatch, a still-valid
    // column of the OLD window, or the zero column. A sequence with no token in this ubatch
    // gets identity indices, so this one copy preserves it untouched.
    //
    // Reading `padded` (a materialised concat in the graph arena) rather than
    // ple_conv_hist_dev itself is what makes the in-place rewrite safe: the gather's source
    // is already a copy, so overlapping source and destination columns cannot race. The taps
    // above are expanded first so the write-back lands after them in node order, and the
    // read of ple_conv_hist_dev (the concat) precedes both. All on one device stream, so a
    // pipelined next ubatch (n_copies > 1) sees this window only after it is written.
    if (lctx.inp_ple_conv_carry != nullptr) {
        ggml_build_forward_expand(gf, conv_out);
        ggml_tensor * next = ggml_get_rows(ctx0, padded, lctx.inp_ple_conv_carry);
        ggml_tensor * dst  = ggml_view_2d(ctx0, lctx.ple_conv_hist_dev, hc_dim, n_win - 1,
                lctx.ple_conv_hist_dev->nb[1], lctx.ple_conv_hist_dev->nb[1]);
        ggml_tensor * carry = ggml_cpy(ctx0, next, dst);
        cb(carry, "ple_conv_hist_carry", il);
        ggml_build_forward_expand(gf, carry);
    }

    return ggml_add(ctx0, hidden, ggml_add(ctx0, ggml_reshape_3d(ctx0, gated, n_embd, hc, n_tokens), conv_out));
}

ggml_cgraph * llm_build_context::build_qwen4exp() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc          = hparams.dsv4_hc_mult;
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(hc > 0 && "qwen4exp needs a hyper-connection count");

    // NextN/MTP tail-only graph, built on the companion context (--spec-type mtp). Same
    // dispatch shape as build_qwen35moe(), but the hidden state handed over by the target
    // is the WIDE hyper-connection row (hc*n_embd per token), not the collapsed n_embd.
    if (cparams.mtp_op_type != MTP_OP_NONE) {
        const int64_t hc_dim = hc*n_embd;

        ggml_tensor * hidden_states_from_main_model;
        if (cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
        } else {
            hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hc_dim);
        }
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.inp_mtp_states = hidden_states_from_main_model;

        ggml_tensor * inp_pos = build_inp_pos();

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        ggml_tensor * cur = build_qwen4exp_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    delta_net delta(lctx, batch);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // the recurrent-path inputs the fused delta-net kernel reads (same set qwen35moe builds)
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
    ggml_set_input(lctx.inp_conv_seq_map);

    lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
    ggml_set_input(lctx.inp_qnext_state_mask);

    const float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head))
                                                             : hparams.f_attention_scale;

    const bool has_ple = hparams.ple_n_heads > 0;
    if (has_ple) {
        const int64_t hc_dim = hc*n_embd;
        const int64_t hist   = (hparams.ple_conv_kernel - 1)*hparams.ple_ngram_size;

        lctx.inp_ple_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hparams.ple_n_heads*n_tokens);
        cb(lctx.inp_ple_rows, "inp_ple_rows", -1);
        ggml_set_input(lctx.inp_ple_rows);

        // One PLE layer is what this build supports: hparams holds a single set of hash
        // constants, and one window is carried. (Mainline has the same limit.)
        int n_ple_layers = 0;
        for (int lp = 0; lp < n_layer; ++lp) if (hparams.is_ple(lp)) ++n_ple_layers;
        GGML_ASSERT(n_ple_layers == 1 && "this build carries one PLE conv window; more than one PLE layer is not supported");

        const int64_t n_seq_max = std::max<int64_t>(1, lctx.cparams.n_seq_max);
        const int64_t n_win     = 1 + n_seq_max*hist;

        lctx.inp_ple_conv_src = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,
                (int64_t) hparams.ple_conv_kernel*n_tokens);
        cb(lctx.inp_ple_conv_src, "inp_ple_conv_src", -1);
        ggml_set_input(lctx.inp_ple_conv_src);

        lctx.inp_ple_conv_carry = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_win - 1);
        cb(lctx.inp_ple_conv_carry, "inp_ple_conv_carry", -1);
        ggml_set_input(lctx.inp_ple_conv_carry);

        // One-time persistent window. It lives in its OWN backend buffer, NOT in the graph
        // arena: the scheduler reuses arena memory across splits (11 of them here, over 4
        // cards), so anything read back or carried from there comes back clobbered. A
        // pre-allocated tensor written by a ggml_cpy node is the idiom the KV cache already
        // uses, and the scheduler never aliases such a tensor.
        if (lctx.ple_conv_hist_dev == nullptr) {
            ggml_init_params cap_params = {
                /*.mem_size   =*/ ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            lctx.ctx_ple_capture = ggml_init(cap_params);
            lctx.ple_conv_hist_dev = ggml_new_tensor_2d(
                    lctx.ctx_ple_capture, GGML_TYPE_F32, hc_dim, n_win);
            ggml_set_name(lctx.ple_conv_hist_dev, "ple_conv_hist_dev");

            // Allocate on the SAME backend that holds the PLE weights, NOT on the CPU. A CPU
            // buffer makes the carry copy a device->host copy INSIDE the graph, which forces
            // an extra scheduler split and a synchronisation on every decode step. Falls back
            // to CPU if the PLE weights are somehow unplaced, which keeps things working
            // rather than failing.
            ggml_backend_buffer_type_t cap_buft = ggml_backend_cpu_buffer_type();
            for (int lp = 0; lp < n_layer; ++lp) {
                ggml_tensor * probe = model.layers[lp].ple_norm_conv;
                if (probe && probe->buffer) {
                    cap_buft = ggml_backend_buffer_get_type(probe->buffer);
                    break;
                }
            }
            lctx.buf_ple_capture = ggml_backend_alloc_ctx_tensors_from_buft(
                    lctx.ctx_ple_capture, cap_buft);
            GGML_ASSERT(lctx.buf_ple_capture && "failed to allocate the PLE conv window buffer");
            // zeroes column 0 (the sentinel, never written again) and every sequence window
            ggml_backend_buffer_clear(lctx.buf_ple_capture, 0);

            lctx.ple_hist_cols = (int32_t) hist;
            lctx.ple_hc_dim    = (int32_t) hc_dim;
            lctx.ple_seq.assign((size_t) n_seq_max, llama_context::ple_seq_state());
        }
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    // the NextN/MTP tail layer(s) belong to the companion graph above, never to the trunk
    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        const auto & layer = model.layers[il];

        // Second A/B rung for attribution: PXA_QWEN4EXP_NO_PLE=1 skips the whole PLE side
        // path, not just its conv. If the output collapse survives BOTH this and
        // PXA_QWEN4EXP_NO_PLE_CONV, then PLE is exonerated and the fault is in the other
        // state-carrying path - the delta-net recurrent state running under hc_mode on 36
        // of the 48 layers. Diagnostic only: the logits are wrong with PLE off, since the
        // shipped model genuinely has the side path.
        static const bool pxa_no_ple = getenv("PXA_QWEN4EXP_NO_PLE") != nullptr;
        if (has_ple && hparams.is_ple(il) && !pxa_no_ple) {
            res_hc = build_qwen4exp_ple(gf, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_qwen4exp_hc_mix(res_hc,
                layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
                &inject, il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            // hc_mode: no input norm (the mixer above was it), no residual add (the combine
            // below is it), sigmoid output gate instead of Qwen3.5 silu.
            cur = delta.build_layer_attn_linear(ctx0, gf, cur, nullptr, il, cb, /*hc_mode*/ true);
        } else {
            // a null norm weight and add_input=false reduce build_std_attention to the block
            // itself: gated Q projection, q/k norms, IMRoPE (is_multi), attention, wo.
            cur = build_std_attention(gf, /*attn_norm*/ nullptr, cur, inp_pos, /*inp_out_ids*/ nullptr,
                    /*rope_factors*/ nullptr, KQ_mask, /*sinks*/ nullptr, /*inp_attn_scale*/ nullptr,
                    KQ_scale, 0.0f, /*n_swa*/ 0, il,
                    /*do_rope*/ true, /*add_graph_split*/ false, /*add_input*/ false,
                    /*is_norm*/ false, /*is_multi*/ true);
        }
        cb(cur, "attn_block_out", il);

        res_hc = build_qwen4exp_hc_combine(res_hc, cur, inject, il);

        cur = build_qwen4exp_hc_mix(res_hc,
                layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
                &inject, il);

        cur = llm_build_std_moe_ffn(ctx0, lctx, /*ffn_norm*/ nullptr, cur,
                layer.ffn_gate_inp,   nullptr,
                layer.ffn_up_exps,    nullptr,
                layer.ffn_gate_exps,  nullptr,
                layer.ffn_down_exps,  nullptr,
                nullptr,
                layer.ffn_up_shexp,   nullptr,
                layer.ffn_gate_shexp, nullptr,
                layer.ffn_down_shexp, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, /*add_input*/ false,
                layer.ffn_up_gate_exps, nullptr, layer.ffn_gate_inp_shexp);
        cb(cur, "ffn_out", il);

        res_hc = build_qwen4exp_hc_combine(res_hc, cur, inject, il);

        res_hc = lctx.cvec.apply_to(ctx0, res_hc, il);
        cb(res_hc, "l_out", il);
    }

    // NextN/MTP hand-over: the draft consumes the WIDE residual BEFORE the head mixer
    // collapses it — one flat [hc*n_embd, n_tokens] row per token, every token (the wide
    // stream is never row-dropped in this graph, so no out_ids handling is needed here).
    // ggml_cont, not a bare reshape view: an OUTPUT-flagged view does not protect the
    // underlying arena memory from reuse across scheduler splits (see the PLE capture
    // note above) and this node is read back host-side after compute.
    if (lctx.cparams.mtp) {
        ggml_tensor * mtp_flat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, res_hc, hc*n_embd, n_tokens));
        cb(mtp_flat, "result_mtp_embd", -1);
        ggml_set_output(mtp_flat);
        ggml_build_forward_expand(gf, mtp_flat);
    }

    // The head mixer is the output norm; this arch ships no separate one. It takes no
    // injection because nothing is scattered back after it.
    ggml_tensor * cur = build_qwen4exp_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up, nullptr, nullptr, -1);

    // one gather at the very end: the wide residual has to stay whole through every layer,
    // so tokens are not dropped per-layer the way the dense graphs do it
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// NextN/MTP tail graph for the grafted Qwen3.8-Flash-Next MTP block: ONE full-attention
// hyper-connection layer (blk.<n_layer-1>) with its own input fusion and its own head mixer.
//
// Wiring (upstream llama.cpp PR #27739 graph_mtp, adapted to this fork's helpers):
//   hnorm(prev wide row, grouped RMS) + enorm(token embd, repeated per stream)
//     -> per-stream concat(e, h) -> eh_proj -> the layer's input streams
//     -> hc_attn mix -> DENSE attention (same build_std_attention call as the trunk's
//        full-attention branch; the tail carries NO QSA indexer tensors)
//     -> hc_combine -> hc_ffn mix -> MoE FFN -> hc_combine
//     -> the MTP's OWN mixer (nextn.hc_*) -> the shared lm_head.
//
// Differences from build_qwen35moe_mtp:
//   * prev_embeddings is the WIDE hc row [hc*n_embd, n_tokens] (the target's res_hc before
//     its head mixer), not the collapsed n_embd hidden — llama_mtp_state_n_embd() reports
//     the wide width so the whole driver moves hc*n_embd floats per token.
//   * hnorm is a WHOLE-ROW RMS: the reduction runs over the full hc row (hc*n_embd), and
//     only then is the row split into streams — upstream reference semantics ("unlike
//     deepseek4, which norms each stream"). PXA_FN_MTP_HNORM_GROUPED=1 flips to a
//     per-stream (grouped) reduction for A/B; acceptance rate is the oracle.
//   * the head collapse uses layer.nextn.hc_{norm,down,up}, NOT model.hc_head_* (that one
//     is the MAIN model's output mixer; the graft gives the draft its own).
//   * inp_out_ids is NOT threaded into the attention: the wide residual must stay whole
//     through hc_combine, so rows are selected once, on the flat stream, before the mixer.
struct ggml_tensor * llm_build_context::build_qwen4exp_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos) {

    const int il = hparams.n_layer - 1;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;

    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "qwen4exp MTP supports a single nextn block");
    GGML_ASSERT(mtp_layer.nextn.eh_proj && mtp_layer.nextn.enorm && mtp_layer.nextn.hnorm &&
            "qwen4exp MTP block missing nextn fusion tensors");
    GGML_ASSERT(mtp_layer.nextn.hc_norm && mtp_layer.nextn.hc_down && mtp_layer.nextn.hc_up &&
            "qwen4exp MTP block missing its nextn.hc_* head mixer");

    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    // enorm: plain RMS over the token embedding, the same term repeated into every stream
    ggml_tensor * e_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // hnorm: default = WHOLE-ROW RMS (reduce over the full hc_dim row, then split into
    // streams — upstream reference; "unlike deepseek4, which norms each stream").
    // PXA_FN_MTP_HNORM_GROUPED=1 = per-stream reduction, for the acceptance A/B.
    static const bool hnorm_grouped = [] {
        const char * e = getenv("PXA_FN_MTP_HNORM_GROUPED");
        return e && atoi(e) != 0;
    }();
    ggml_tensor * h_norm;
    if (hnorm_grouped) {
        h_norm = ggml_rms_norm(ctx0,
                ggml_reshape_3d(ctx0, prev_embeddings, n_embd, hc, n_tokens), hparams.f_norm_rms_eps);
        h_norm = ggml_mul(ctx0, ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens), mtp_layer.nextn.hnorm);
    } else {
        h_norm = ggml_rms_norm(ctx0,
                ggml_reshape_2d(ctx0, prev_embeddings, hc_dim, n_tokens), hparams.f_norm_rms_eps);
        h_norm = ggml_mul(ctx0, h_norm, mtp_layer.nextn.hnorm);
    }
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // fc_embedding(e) + fc_hidden(h) is ONE projection of the per-stream concat (the graft
    // packed eh_proj = concat([fc_embedding, fc_hidden], dim=1)); broadcast over the streams
    ggml_tensor * inpL = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj,
            ggml_concat(ctx0, e_norm, h_norm, 0));
    cb(inpL, "mtp_eh_proj", il);

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

    const float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head))
                                                             : hparams.f_attention_scale;

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur    = build_qwen4exp_hc_mix(inpL,
            mtp_layer.hc_attn_norm, mtp_layer.hc_attn_down, mtp_layer.hc_attn_up,
            mtp_layer.hc_attn_inject, &inject, il);
    ggml_build_forward_expand(gf, cur);
    cb(cur, "mtp_hc_attn_pre", il);

    // DENSE attention — the exact call the trunk's full-attention branch makes (null norm,
    // add_input=false, IMRoPE is_multi). No QSA: the tail has no indexer, and one layer
    // spends its time reading weights, not attending.
    cur = build_std_attention(gf, /*attn_norm*/ nullptr, cur, inp_pos, /*inp_out_ids*/ nullptr,
            /*rope_factors*/ nullptr, KQ_mask, /*sinks*/ nullptr, /*inp_attn_scale*/ nullptr,
            KQ_scale, 0.0f, /*n_swa*/ 0, il,
            /*do_rope*/ true, /*add_graph_split*/ false, /*add_input*/ false,
            /*is_norm*/ false, /*is_multi*/ true);
    cb(cur, "mtp_attn_block_out", il);

    inpL = build_qwen4exp_hc_combine(inpL, cur, inject, il);
    cb(inpL, "mtp_hc_attn_post", il);

    cur = build_qwen4exp_hc_mix(inpL,
            mtp_layer.hc_ffn_norm, mtp_layer.hc_ffn_down, mtp_layer.hc_ffn_up,
            mtp_layer.hc_ffn_inject, &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = llm_build_std_moe_ffn(ctx0, lctx, /*ffn_norm*/ nullptr, cur,
            mtp_layer.ffn_gate_inp,   nullptr,
            mtp_layer.ffn_up_exps,    nullptr,
            mtp_layer.ffn_gate_exps,  nullptr,
            mtp_layer.ffn_down_exps,  nullptr,
            nullptr,
            mtp_layer.ffn_up_shexp,   nullptr,
            mtp_layer.ffn_gate_shexp, nullptr,
            mtp_layer.ffn_down_shexp, nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true, false, 0.0f,
            LLM_EXPERT_GATING_FUNC_SOFTMAX,
            LLM_FFN_SILU, cb, il, gf, /*add_input*/ false,
            mtp_layer.ffn_up_gate_exps, nullptr, mtp_layer.ffn_gate_inp_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_qwen4exp_hc_combine(inpL, cur, inject, il);
    cb(inpL, "mtp_l_out", il);

    // Chained-draft hand-over: the next draft step is conditioned on the WIDE streams the
    // block just produced, exactly as the target's trunk hands its res_hc over. Row-select
    // here (once, on the flat stream) so ctx->embd rows line up with output_ids; when no
    // selection runs, ggml_cont so the OUTPUT-flagged node owns its memory (a bare reshape
    // view does not survive arena reuse — see the PLE capture note in build_qwen4exp_ple).
    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, hc_dim, n_tokens);
    if (inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
    } else {
        flat = ggml_cont(ctx0, flat);
    }
    cb(flat, "result_mtp_embd", -1);
    ggml_set_output(flat);
    ggml_build_forward_expand(gf, flat);

    // The MTP's OWN head mixer collapses the streams — model.hc_head_* is the MAIN model's
    // collapse and is deliberately NOT shared with the draft.
    cur = build_qwen4exp_hc_mix(ggml_reshape_3d(ctx0, flat, n_embd, hc, flat->ne[1]),
            mtp_layer.nextn.hc_norm, mtp_layer.nextn.hc_down, mtp_layer.nextn.hc_up,
            nullptr, nullptr, -1);
    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output, cb);
    cb(cur, "result_output", -1);

    return cur;
}
