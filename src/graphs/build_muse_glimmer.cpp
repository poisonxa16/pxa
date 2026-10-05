#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Muse Glimmer (muse-glimmer): dense 30B, 52 layers, GQA 32 q / 2 kv heads,
// head_dim 128, interleaved local/global attention with period 4 (three
// SWA-2048 layers with RoPE, then one full-attention NoPE layer), per-head RMS
// QK-norm (q_norm carries the folded qk_scale_factor from conversion; k_norm
// is identity on released checkpoints), a FULL-WIDTH sigmoid attention output
// gate [n_embd -> n_head*head_dim] computed from the pre-attention normed
// hidden state and applied between SDPA and o_proj, sandwich norms around both
// the attention and FFN blocks (the post-norms use a fixed eps 1e-8 that is
// deliberately different from f_norm_rms_eps, matching the reference), an
// unweighted RMS norm applied to the raw token embeddings, a SwiGLU dense FFN,
// and an lm_head followed by a logit_scale multiplier and a tanh softcap.
//
// The wqkv_gate tensor here is NOT the per-head scalar gate that
// build_std_attention assumes -- this graph applies the gate itself and never
// routes through build_std_attention.
ggml_cgraph * llm_build_context::build_muse_glimmer() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == (int64_t) hparams.n_rot);

    // post-attention / post-FFN norm eps, fixed and distinct from f_norm_rms_eps
    const float post_norm_eps = 1e-8f;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // unweighted RMS norm on the embeddings
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    cb(inpL, "embd_norm", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        ggml_tensor * KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;

        ggml_tensor * inpSA = inpL;

        // pre-attention norm ("weight + 1" folded at conversion time)
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * attn_inp = cur; // gate input = pre-attention normed hidden state

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    nullptr, nullptr, nullptr, nullptr,
                    model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.0f, il);

            // full-width attention output gate off the pre-attention hidden state
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, attn_inp);
            cb(gate, "attn_gate_proj", il);

            if (is_swa) {
                // RoPE runs on the SWA layers only; full-attention layers are NoPE
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        hparams.rope_freq_base_train_swa, hparams.rope_freq_scale_train_swa,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        hparams.rope_freq_base_train_swa, hparams.rope_freq_scale_train_swa,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);
                cb(Kcur, "Kcur_rope", il);
            }

            // SDPA with wo deferred: the gate applies between attn out and o_proj
            //
            // PXA_SWA_KV: the sliding layers read and write a SEPARATE cache object with its own
            // cells/head/size — a distinct index space, not a remapping of this one. When the
            // feature is off, kv_swa/n_kv_swa/kv_head_swa alias the unified cache and this is
            // exactly the original call.
            const llama_kv_cache & kv_l   = is_swa ? kv_swa      : kv_self;
            const int32_t          nkv_l  = is_swa ? n_kv_swa    : n_kv;
            const int32_t          khead_l= is_swa ? kv_head_swa : kv_head;

            //
            // The n_swa passed to the attention op is a KERNEL HINT, and one of its consumers --
            // the CPU iqk flash-attention path -- still slices K/V/mask to the LAST
            // pad(n_tokens + n_swa) cells BY INDEX, which assumes cell-index order matches position
            // order. That is the same assumption 16d5c1a8 removed from the CUDA path and did not
            // remove here. It happens to hold for a cache that is appended to in position order; it
            // does NOT hold for the sliding cache, whose live band floats and wraps, so the slice
            // can drop every cell a query can see and leave an all -inf mask row.
            // Suppress the hint when the layer is served by the sliding cache. Nothing is lost
            // correctness-wise (the mask still carries the window) and little is lost in work,
            // because that cache IS the window: there is no long tail left to skip.
            // 2026-08-17 (fable10): the hint is passed again for sliding-cache layers. The reason
            // it was suppressed -- the CPU iqk path slicing K/V/mask by INDEX -- is gone: that
            // slice is now mask-driven (PXA_NANFIX_SWA_SLICE_CPU in iqk_flash_attn.cpp) and is
            // correct for a floating/wrapping band. Passing it again restores the CUDA
            // mask-driven KV_min_max bound that 16d5c1a8 deliberately kept, which the suppression
            // had also switched off for these layers.
            // PXA_SWA_HINT=0 restores the suppression without a rebuild, as a fallback lever.
            static const bool pxa_swa_hint_on = []() {
                const char * e = getenv("PXA_SWA_HINT");
                return !(e && e[0] == '0');
            }();
            const int n_swa_hint = is_swa ? ((lctx.swa_kv_active && !pxa_swa_hint_on) ? 0 : (int) hparams.n_swa) : 0;

            cur = llm_build_kv(ctx0, lctx, kv_l, gf, nullptr, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask_l, n_tokens, khead_l, nkv_l, kq_scale, cb, il,
                    nullptr, n_swa_hint);
            cb(cur, "attn_out", il);

            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sig", il);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "attn_o_proj", il);
        }

        cur = ggml_rms_norm(ctx0, cur, post_norm_eps);
        cur = ggml_mul(ctx0, cur, model.layers[il].attn_post_norm);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // SwiGLU dense FFN with the pre-FFN norm folded in
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_rms_norm(ctx0, cur, post_norm_eps);
        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_post_norm);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head, output multiplier, final tanh softcap
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    if (hparams.f_logit_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    }
    if (hparams.f_final_logit_softcapping != 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
