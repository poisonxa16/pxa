#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>
#include <cstdio>
#include <vector>

// ============================================================================
// DSpark block drafter (M3).
//
// Three DS4 blocks in the RAW-SWA regime, run over a block of block_size+1 rows in ONE
// forward pass. Established from the reference implementation (antirez ds4, HEAD
// 54b36ed), not guessed; the file:line citations below are into that tree.
//
// The five properties that are load-bearing for acceptance (do not "simplify" any of
// them - each is a silent quality bug, not a crash):
//
//   1. ROW 0 IS INVIOLATE ACROSS ALL THREE STAGES (ds4.c:30785-30806). It carries the
//      target signal into every stage. Here it is enforced explicitly: after each
//      stage's FFN hc_post the row-0 slice is restored from the stage-entry stream, so
//      no arithmetic anywhere in the stage can leak into it.
//   2. ATTENTION IS NON-CAUSAL WITHIN THE BLOCK (ds4_cuda.cu:13466-13530). Every draft
//      row sees every other draft row INCLUDING later ones. Without it, block drafting
//      does not work at all. This is a MASK property, not an op: the mask is a graph
//      input and llama_set_inputs fills the intra-block square open in both directions.
//   3. The Markov chain is genuinely sequential (M4).
//   4. Confidence is evaluated BEFORE the Markov bias and short-circuits (M4).
//   5. The scheduler is part of the algorithm (M5).
//
// Row 0 shares its position with row 1: positions are
//   [pos, pos, pos+1, pos+2, pos+3, pos+4]                        (ds4.c:30261-30264)
// and the draft rows are seeded from the TARGET's token embeddings of
//   [last_real_token, NOISE, NOISE, NOISE, NOISE]                 (ds4.c:30266-30270)
// where NOISE = dspark.noise_token_id is a LEARNED placeholder embedding for the
// not-yet-known positions. That learned placeholder is what makes one-pass block
// drafting possible at all.
//
// DELIBERATE DEVIATION, numerically inert: the reference computes Q for rows 1..k only
// while computing KV for all k+1 rows. Here Q is computed for all k+1 rows and row 0's
// attention output is then discarded by the row-0 restore in (1). That keeps every
// shape rectangular and constant, which is what `can_reuse_graph` needs (a variable row
// count would evict the cached drafter graph every cycle - see the plan's 4.2). The cost
// is 1/(k+1) of one attention; the benefit is graph reuse on every cycle.
//
// WHY THE DS4 ATTENTION HELPERS ARE NOT REUSED HERE (verified, and it is not a style
// choice): llama_dsv4_get_inputs() DISCARDS its context argument and returns the
// process-global module (llama-kv-cache-dsv4.cpp:670), and build_dsv4_raw_attention is
// welded to it - it calls llama_dsv4_cpy_raw_k(), which would write the DRAFTER's latent
// K rows into the TARGET model's layer-0/1/2 raw SWA ring. Not a compile error, not a
// crash: silent target-KV corruption presenting as low acceptance. The drafter therefore
// runs its attention against its OWN standard llama_kv_cache. What IS reused is
// everything verified clean of global state: build_dsv4_attn_mha, build_dsv4_hc_pre /
// _post / _head (the 20-iteration Sinkhorn is already one fused node), and the
// deepseek4 MoE path.
//
// A DSpark stage carries no compressor, no lightning indexer and no ffn_gate_tid2eid -
// its tensor set is byte-for-byte DS4's blk.0, i.e. compress ratio 0. So there is no
// CSA/HCA branch here and no hash routing, and llm_load_hparams_dspark() forces
// dsv4_hash_layer_count to 0 so a stage can never take the target's hash-routing branch
// and read a table that does not exist in the file.
// ============================================================================

ggml_cgraph * llm_build_context::build_dspark() {
    const auto & tgt = model.dspark_target;
    GGML_ASSERT(tgt && "DSpark drafter used without llama_dspark_bind_target()");
    GGML_ASSERT(tgt->tok_embd && tgt->output && "DSpark borrows the target's embd/lm_head");

    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t n_block    = hparams.dspark_block_size;      // k candidate positions
    const int64_t n_rows     = n_block + 1;                    // + the target row
    const int64_t n_capture  = hparams.dspark_n_target_layers;

    GGML_ASSERT(hc == 4 && "DSpark graph assumes hyper_connection.count == 4");
    GGML_ASSERT(n_block > 0);
    GGML_ASSERT(n_tokens == n_rows && "the drafter batch is exactly block_size+1 rows");
    GGML_ASSERT(hparams.n_embd_head_k_full == hparams.n_embd_head_v_full);

    // The grouped output LoRA reshapes wo_a to {ne[0], o_lora_rank, o_groups}. Reading
    // o_lora_rank straight off attn_output_a.ne[1] yields o_lora_rank*o_groups (8192 on
    // this checkpoint, not 1024) and this reshape is where that lands. Assert the
    // relation the loader derived so a regression cannot pass silently.
    GGML_ASSERT(hparams.dsv4_o_group_count > 0 && hparams.dsv4_o_lora_rank > 0);
    GGML_ASSERT(model.layers[0].wo_a->ne[1] ==
            (int64_t) hparams.dsv4_o_lora_rank * (int64_t) hparams.dsv4_o_group_count &&
            "wo_a rank must be o_lora_rank*o_groups - see llm_load_hparams_dspark()");

    // ---- inputs -------------------------------------------------------------
    // The captured target hidden states, [n_capture*n_embd, 1], host-fed by the spec loop.
    lctx.inp_dspark_cap = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_capture*n_embd, 1);
    ggml_set_input(lctx.inp_dspark_cap);
    cb(lctx.inp_dspark_cap, "dspark_cap", -1);

    lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
    ggml_set_input(lctx.inp_tokens);
    cb(lctx.inp_tokens, "dspark_inp_tokens", -1);

    // M4: the Markov chain's seed - the token the target just committed.
    lctx.inp_dspark_prev = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(lctx.inp_dspark_prev);
    cb(lctx.inp_dspark_prev, "dspark_prev", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    // Non-causal within the block. Filled by llama_set_inputs; NOT build_inp_KQ_mask(),
    // whose fill is driven by cparams.causal_attn and would hide row 3 from row 1.
    // NOTE: n_kv is the llm_build_context member (worst_case ? kv_self.size : kv_self.n).
    // Do NOT shadow it with kv_self.n: during the worst-case graph reserve at context
    // creation kv_self.n is 0, which builds a zero-width K and a zero-width mask and
    // trips ggml_mul_mat deep inside the attention.
    const auto & kv_self = lctx.kv_self;
    {
        const int64_t n_mask_rows = GGML_PAD(n_rows, GGML_KQ_MASK_PAD);
        lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0,
                cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_mask_rows);
        ggml_set_input(lctx.inp_KQ_mask);
        cb(lctx.inp_KQ_mask, "dspark_kq_mask", -1);
    }

    // ---- 3a: main_x = RMSNorm(main_proj(concat(h40,h41,h42)), main_norm) ----
    //                                                          ds4.c:29981-30012
    ggml_tensor * main_x = llm_build_lora_mm(lctx, ctx0,
            model.layers[0].dspark_main_proj, lctx.inp_dspark_cap);
    main_x = llm_build_norm(ctx0, main_x, hparams,
            model.layers[0].dspark_main_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(main_x, "dspark_main_x", -1);                                    // [n_embd, 1]

    // ---- 3b: assemble the block -------------------------------------------
    // rows 1..k from the TARGET's token embeddings (borrowed, not owned):
    ggml_tensor * tok = ggml_get_rows(ctx0, tgt->tok_embd, lctx.inp_tokens);
    cb(tok, "dspark_tok_embd", -1);                                 // [n_embd, n_rows]

    // row 0 := main_x, rows 1..k := the seeded embeddings.
    ggml_tensor * rows_1k = ggml_view_2d(ctx0, tok, n_embd, n_block,
            tok->nb[1], tok->nb[1]);
    ggml_tensor * block = ggml_concat(ctx0, main_x, ggml_cont(ctx0, rows_1k), 1);
    cb(block, "dspark_block_in", -1);                               // [n_embd, n_rows]

    // widen to the hc branches: [n_embd, hc, n_rows]
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, block, n_embd, 1, n_rows);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_rows, 1);
    cb(inpL, "dspark_hc_init", -1);

    const int64_t n_embd_head      = hparams.n_embd_head_k_full;
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const float   kq_scale         = 1.0f/sqrtf(float(n_embd_head));

    // ratio-0 layers take the PLAIN rope basis, never the compressed one
    // (build_dsv4_attention: use_compress_rope = ratio != 0).
    const float   freq_base_l   = freq_base;
    const float   freq_scale_l  = 1.0f;
    const float   ext_factor_l  = 0.0f;
    // dsv4_rope_attn_factor() returns exactly 1.0f when ext_factor == 0, which is the
    // ratio-0 case unconditionally. Inlined because the helper is file-static in
    // build_deepseek4.cpp and duplicating it would be a second thing to keep in sync.
    const float   attn_factor_l = 1.0f;
    const float   beta_fast_l   = 0.0f;
    const float   beta_slow_l   = 0.0f;
    const int32_t n_ctx_orig_l  = 0;

    // ---- 3c: the three stages ---------------------------------------------
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * stage_in = inpL;          // for the row-0 restore below
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;
        ggml_tensor * cur  = nullptr;

        cur = build_dsv4_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "dspark_attn_norm", il);

        // ---- attention, against the DRAFTER's own cache -------------------
        {
            ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, layer.wq_a, cur);
            qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);

            ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_rows);
            q = ggml_rms_norm(ctx0, q, norm_rms_eps);

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, n_rows,
                    ggml_row_size(q->type, n_embd_head),
                    ggml_row_size(q->type, n_embd_head)*n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, n_rows,
                    ggml_row_size(q->type, n_embd_head),
                    ggml_row_size(q->type, n_embd_head)*n_head,
                    ggml_row_size(q->type, n_embd_head_nope));
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type,
                    n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                    beta_fast_l, beta_slow_l);
            q = ggml_concat(ctx0, q_nope, q_pe, 0);
            cb(q, "dspark_q", il);

            // one shared 512-wide latent for all 64 heads
            ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, layer.wkv, cur);
            kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_rows);

            ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_rows,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_rows,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head_nope));
            kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type,
                    n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                    beta_fast_l, beta_slow_l);
            kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
            cb(kv, "dspark_kv", il);

            // ALL k+1 rows are written, including row 0 (ds4: "KV for all 6 rows").
            // MLA shares one latent, so the same tensor serves as K and as V.
            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf,
                    kv, /*v_cur*/ nullptr, n_rows, kv_head, cb, il);

            ggml_tensor * k = ggml_view_3d(ctx0, kv_self.k_l[il],
                    n_embd_head, 1, n_kv,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head), 0);
            cb(k, "dspark_k_all", il);

            // the window ++ the k+1 block rows, with the intra-block square open in
            // BOTH directions - non-causal (property 2).
            ggml_tensor * out = build_dsv4_attn_mha(gf, q, k, k,
                    lctx.inp_KQ_mask, layer.attn_sinks, kq_scale, il);

            // de-rope, then the grouped output LoRA
            out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, n_rows);
            ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, n_rows,
                    ggml_row_size(out->type, n_embd_head),
                    ggml_row_size(out->type, n_embd_head)*n_head, 0);
            ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, n_rows,
                    ggml_row_size(out->type, n_embd_head),
                    ggml_row_size(out->type, n_embd_head)*n_head,
                    ggml_row_size(out->type, n_embd_head_nope));
            out_pe = ggml_rope_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type,
                    n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                    beta_fast_l, beta_slow_l);
            out = ggml_concat(ctx0, out_nope, out_pe, 0);

            out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, n_rows);
            out = ggml_permute(ctx0, out, 0, 2, 1, 3);
            ggml_tensor * oa = ggml_mul_mat(ctx0,
                    ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
            oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, n_rows);

            cur = llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
            cb(cur, "dspark_attn_out", il);
        }

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        cb(inpL, "dspark_hc_attn_post", il);

        // ---- FFN: 256-expert top-k MoE + one shared expert ----------------
        residual = inpL;
        cur = build_dsv4_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il);
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);

        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                hparams.expert_weights_norm,
                hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, /*add_input*/ false,
                nullptr, nullptr, nullptr, nullptr, /*selected_experts_in*/ nullptr);
        cb(moe_out, "dspark_moe_out", il);

        ggml_tensor * ffn_shexp;
        {
            ggml_tensor * up   = llm_build_lora_mm(lctx, ctx0, layer.ffn_up_shexp,   cur);
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_shexp, cur);
            const float limit = hparams.swiglu_limits_shared[il];
            if (limit > 1e-6f) {
                up   = ggml_clamp(ctx0, up,   -limit,    limit);
                gate = ggml_clamp(ctx0, gate, -INFINITY, limit);
            }
            ffn_shexp = llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp,
                    ggml_swiglu_split(ctx0, gate, up));
        }
        cur = ggml_add(ctx0, moe_out, ffn_shexp);

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        cb(inpL, "dspark_hc_ffn_post", il);

        // ---- PROPERTY 1: row 0 is restored, bit-for-bit, from the stage entry ----
        // The hand-off replaces rows 1..k ONLY (ds4.c:30785-30806). Enforced here as an
        // explicit splice rather than trusted to the arithmetic, so a future change to
        // any stage internal cannot leak into the target row.
        {
            ggml_tensor * row0 = ggml_view_3d(ctx0, stage_in, n_embd, hc, 1,
                    stage_in->nb[1], stage_in->nb[2], 0);
            ggml_tensor * rows = ggml_view_3d(ctx0, inpL, n_embd, hc, n_block,
                    inpL->nb[1], inpL->nb[2], inpL->nb[2]);
            inpL = ggml_concat(ctx0, ggml_cont(ctx0, row0), ggml_cont(ctx0, rows), 2);
            cb(inpL, "dspark_row0_preserved", il);
        }
    }

    // ---- 3d: hc_head -> RMSNorm(norm) -> the TARGET's lm_head --------------
    //                                                          ds4.c:31548-31714
    ggml_tensor * cur = build_dsv4_hc_head(inpL,
            model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "dspark_hc_head", -1);

    cur = llm_build_norm(ctx0, cur, hparams,
            model.layers[n_layer-1].dspark_norm, nullptr, LLM_NORM_RMS, cb, -1);

    // Drop the target row: only the k draft positions get logits. Driven by the batch's
    // own out_ids rather than a hardcoded slice - llama_set_inputs asserts the tensor
    // exists whenever n_outputs < n_tokens, and using it keeps the head in agreement with
    // whatever rows the caller actually flagged.
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    ggml_tensor * head_in = cur;                                  // [n_embd, n_out]
    cb(head_in, "dspark_head_in", -1);

    ggml_tensor * logits = llm_build_lora_mm(lctx, ctx0, tgt->output, head_in);
    cb(logits, "result_output", -1);                              // [n_vocab, n_out]

    // MANDATORY once anything is appended after it, and MEASURED - not a precaution.
    // In every other arch result_output is the graph's last node, so ggml-alloc never gets
    // the chance to recycle it and llama_decode's readback is safe by position. Here the
    // M4 chain consumes it and then keeps allocating: the moment position k's view/cont is
    // done, result_output's last child is gone, ggml_gallocr_free_node hands its 2.5 MB
    // back (ggml-alloc.c:547) and the next 517 KB node in the chain lands on top of ROW 0.
    // The chain itself still computes the right answer - it had already read the row - but
    // llama_get_logits() then returns a corrupted first row, silently. Measured exactly
    // that way: positions 1..4 of the readback matched the tapped values bit-for-bit and
    // position 0 matched nothing at all.
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    // ========================================================================
    // 3e (M4): the rank-256 Markov chain and the confidence head.
    //
    //   state_k = markov_w1[:, prev]                                 ds4.c:32169-32172
    //   feat_k  = concat( hidden[k] (n_embd), state_k (rank) )   <- hidden FIRST
    //                                                                ds4.c:32088-32095
    //   conf_k  = sigmoid( conf_proj . feat_k )                      ds4.c:32180-32190
    //   id_k    = argmax( logits[:,k] + markov_w2^T . state_k )      ds4.c:32242-32257
    //   prev    = id_k                                               ds4.c:32266
    //
    // Composed entirely from existing ggml ops (plan 7.1). The chain is the bit-exactness
    // ORACLE for any later fused GGML_OP_DSPARK_MARKOV_HEAD: a new op must never be the
    // reason temp-0 output drifts.
    //
    // WHERE THIS DIFFERS FROM THE REFERENCE, AND WHY IT IS NOT AN APPROXIMATION: ds4 ships
    // TWO paths. The lazy runtime one BREAKS out of the loop the moment a confidence falls
    // under the threshold (ds4.c:32187-32190); the probe one computes ALL block_size
    // confidences with dspark_eval_confidence_probe (ds4.c:32053-32105) and truncates
    // afterwards with dspark_confident_prefix_len (ds4.c:32330-32341). A static graph can
    // express the second exactly, and the two produce the IDENTICAL proposal - the break is
    // a compute saver, not a semantic. Truncation happens on the host
    // (llama_dspark_confident_prefix), which is the same function ds4 uses.
    //
    // sigmoid is deliberately NOT built. It is monotone, so comparing its argument against
    // logit(threshold) is identical; leaving it out removes an op and a rounding step. The
    // stored value is the RAW LOGIT and the host applies the stable sigmoid.
    //
    // THREE TRAPS, all of them silent rather than loud:
    //  A. the five ids CANNOT be packed into one I32 tensor in-graph. ggml_cuda's
    //     supports_op REFUSES CONCAT on an I32 src (ggml-cuda.cu:7517-7520) and CPY only
    //     handles F32->F32/F16 (:7481-7492), so a pack would be scheduled onto the CPU and
    //     cost a D2H/H2D round trip of the chain per proposal - which reads as "DSpark is
    //     mysteriously slower", not as an error. Five 4-byte reads is the cheaper answer;
    //     the fused op (plan 7.2, one F32 [2,block_size] dst) is the real fix if the
    //     instrumented read cost ever justifies it.
    //  B. for the same reason the argmax result is touched ONLY as src[1] of the next
    //     get_rows. No cont, no reshape, no view-of-a-view - each of those is a cpy.
    //  C. ggml-alloc REUSES a tensor's storage the moment its last consumer is done
    //     (ggml-alloc.c:500,547) unless GGML_TENSOR_FLAG_OUTPUT is set. Without
    //     ggml_set_output the host reads back whatever overwrote id_0..id_3 - measured, in
    //     the M4 preflight, as ids like 999080960, which are float bit patterns. Every id
    //     and the packed confidence MUST be flagged.
    // ========================================================================
    {
        const auto & fin = model.layers[n_layer-1];   // stage 2 carries both heads
        GGML_ASSERT(fin.dspark_markov_w1 && fin.dspark_markov_w2 && fin.dspark_conf_proj &&
                "DSpark M4: the final stage must carry markov_w1/w2 and confidence_head.proj");

        const int64_t rank    = hparams.dspark_markov_rank;
        const int64_t n_vocab = logits->ne[0];
        GGML_ASSERT(fin.dspark_markov_w1->ne[0] == rank && fin.dspark_markov_w2->ne[0] == rank);
        GGML_ASSERT(fin.dspark_conf_proj->ne[0] == n_embd + rank &&
                "conf_proj is [n_embd+markov_rank, 1] - the feature is concat(hidden, state)");
        // The chain indexes rows 0..n_block-1 of BOTH logits and head_in, so the drafter
        // batch must flag exactly the k draft rows and nothing else. In the worst-case
        // reserve n_outputs is the whole batch, which is wider - that graph is never run,
        // and being wider keeps the node count identical so the reserve still covers the
        // real graph.
        GGML_ASSERT(logits->ne[1] >= n_block && head_in->ne[1] >= n_block);
        if (!is_reserve) {
            GGML_ASSERT(logits->ne[1] == n_block &&
                    "DSpark M4: the drafter batch must flag exactly block_size output rows");
        }

        std::vector<ggml_tensor *> state_node(n_block, nullptr);
        std::vector<ggml_tensor *> id_node   (n_block, nullptr);
        std::vector<ggml_tensor *> conf_node (n_block, nullptr);

        // NEGATIVE CONTROL, and the only reason it exists: a gate that cannot fail is not
        // a gate. PXA_DSPARK_BREAK_CHAIN=1 seeds every position from inp_dspark_prev
        // instead of from its predecessor - i.e. it builds exactly the parallelised chain
        // that reproduces DFlash's suffix decay - so both property-3 gates (the pointer
        // asserts below, and the successor self-test in the M4 harness) can be shown to
        // catch it. It is loud, it is off by default, and it must never be set anywhere
        // except a deliberate control run.
        const bool break_chain = getenv("PXA_DSPARK_BREAK_CHAIN") &&
                                 atoi(getenv("PXA_DSPARK_BREAK_CHAIN")) != 0;
        if (break_chain) {
            fprintf(stderr, "%s: *** PXA_DSPARK_BREAK_CHAIN=1: the Markov chain is being "
                            "built PARALLEL. This is a negative control. Proposals from "
                            "this graph are WRONG by construction. ***\n", __func__);
        }

        ggml_tensor * prev = lctx.inp_dspark_prev;

        for (int64_t k = 0; k < n_block; ++k) {
            // ---- the state is SHARED by the confidence at k and the bias at k ----
            ggml_tensor * state = ggml_get_rows(ctx0, fin.dspark_markov_w1, prev);   // F32 [rank,1]
            cb(state, "dspark_state", (int) k);

            // ---- confidence at k: conf_proj . concat(hidden_k, state) ----
            ggml_tensor * h_k = ggml_view_2d(ctx0, head_in, n_embd, 1,
                    head_in->nb[1], (size_t) k * head_in->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, h_k), state, 0);  // F32 [E+r,1]
            ggml_tensor * cfl  = ggml_mul_mat(ctx0, fin.dspark_conf_proj, feat);     // F32 [1,1]
            cb(cfl, "dspark_conf", (int) k);

            // ---- Markov bias + argmax at k ----
            ggml_tensor * bias = ggml_mul_mat(ctx0, fin.dspark_markov_w2, state);    // F32 [V,1]
            cb(bias, "dspark_bias", (int) k);
            ggml_tensor * lgk  = ggml_view_2d(ctx0, logits, n_vocab, 1,
                    logits->nb[1], (size_t) k * logits->nb[1]);
            ggml_tensor * biased = ggml_add(ctx0, ggml_cont(ctx0, lgk), bias);       // F32 [V,1]
            cb(biased, "dspark_biased", (int) k);
            ggml_tensor * id_k   = ggml_argmax(ctx0, biased);                        // I32 [1]

            ggml_format_name(id_k, "dspark_id_%d", (int) k);
            ggml_set_output(id_k);                       // trap C
            ggml_build_forward_expand(gf, id_k);

            state_node[k] = state;
            id_node  [k]  = id_k;
            conf_node[k]  = cfl;

            prev = break_chain ? lctx.inp_dspark_prev : id_k;   // <<<< PROPERTY 3: THE SEQUENTIAL EDGE
        }

        // Pack ONLY the confidences - they are F32, so concat is legal (trap A).
        ggml_tensor * conf = conf_node[0];
        for (int64_t k = 1; k < n_block; ++k) {
            conf = ggml_concat(ctx0, conf, conf_node[k], 1);
        }
        ggml_set_name(conf, "result_dspark_conf");                                  // F32 [1,k]
        ggml_set_output(conf);                           // trap C
        ggml_build_forward_expand(gf, conf);

        // ---- PROPERTY 3, ENFORCED STRUCTURALLY, IN EVERY BUILD ----------------
        // Parallelising the chain (every position reading the last REAL token instead of
        // its predecessor's argmax) reproduces exactly the DFlash suffix decay that DSpark
        // exists to fix - worth +16.3% accepted length. It cannot crash. It surfaces only
        // as low acceptance, which is indistinguishable from "the drafter is mediocre".
        // So it is asserted on POINTER IDENTITY, not on shapes: a batched rewrite (one
        // get_rows over five ids, one argmax over [V,k]) fails here at graph-build time,
        // in every build, before a single token is ever drafted.
        GGML_ASSERT(state_node[0]->src[1] == lctx.inp_dspark_prev &&
                "DSpark M4: step 0's state must be seeded from inp_dspark_prev");
        for (int64_t k = 1; k < n_block && !break_chain; ++k) {
            GGML_ASSERT(state_node[k]->src[1] == id_node[k-1] &&
                    "DSpark M4: the Markov chain has been PARALLELISED - see plan 1.4 property 3");
        }
        if (break_chain) {
            // prove the assert above would have fired, without aborting the control run
            for (int64_t k = 1; k < n_block; ++k) {
                if (state_node[k]->src[1] != id_node[k-1]) {
                    fprintf(stderr, "%s: NEGATIVE CONTROL: the structural assert WOULD have "
                                    "fired at k=%d (state src is not the previous argmax)\n",
                                    __func__, (int) k);
                    break;
                }
            }
        }
        int n_argmax = 0;
        for (int i = 0; i < gf->n_nodes; ++i) {
            if (gf->nodes[i]->op == GGML_OP_ARGMAX) ++n_argmax;
        }
        GGML_ASSERT(n_argmax == (int) n_block &&
                "DSpark M4: exactly one argmax per block position, unrolled");
    }

    return gf;
}
