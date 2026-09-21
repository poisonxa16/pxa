#include <algorithm>

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-kv-cache-kpool.h"
#include "../llama-qsa-prof.h"
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

// =======================================================================================
// PXA_QSA: query-time sparse attention
// =======================================================================================
//
// qwen4exp ships a trained sparse-attention indexer on each of its twelve full-attention
// layers. The reference definition (ref/qwen4exp-mainline.cpp, build_qsa_top_k /
// build_attn_qsa) is:
//
//   k_raw[c]  = index_k_proj * x_c                        ONE 128-wide key per CELL, RAW:
//                                                         pooling precedes norm and rotation
//   kb[p]     = rope_multi(rms_norm(mean_{m<r} k_raw[cell(p,m)]) * index_k_norm, blk_pos[p])
//   q[h,t]    = rope_multi(rms_norm(index_q_proj * x_t) * index_q_norm, pos[t])   h < 4
//   s[p,t]    = sum_h relu( <kb[p], q[h,t]> )             no scale factor at all
//   selection = the top (top_k + r - 1) CELLS of the per-cell expansion of s, with the
//               causal/visibility bias added PER CELL
//   attention = ordinary dense GQA with everything outside the selection masked to -inf
//
// and the whole point of the mechanism is that the last line is where the time goes: masking
// a cell costs a flash-attention kernel exactly what attending to it costs, so the win is in
// physically GATHERING the selected cells and attending densely over the compact set.
//
// The three arms this builds, and what each one claims:
//
//   PXA_QSA unset (the default)   node-for-node the build that shipped; this file is dead
//   PXA_QSA=1 PXA_QSA_GATHER=0    the selection as an n_kv-wide additive mask over the SAME
//                                 dense attention the default arm runs -- i.e. exactly the
//                                 architecture's reference algorithm, and the control
//   PXA_QSA=1 PXA_QSA_GATHER=1    the same selection, physically gathered, O(n_sel) work
//
// Arms 2 and 3 sum the SAME terms in a different order and blocking, so what is claimed
// between them is identical greedy text and a token-0 logit spread inside tolerance, not bit
// identity. Arm 1 is byte-identical to the pre-QSA build by construction: nothing below is
// built, and no side storage is allocated, unless llama_qsa_enabled().
//
// WHERE THE PIECES COME FROM
// --------------------------
// The block grid -- which cells form a block, which blocks a token may see, the incomplete
// tail, the padded block count, the gather mask, and the "blocks this ubatch completed" list
// -- is llama_kpool_* (src/llama-kpool-grid.cpp), already pure and already unit-tested by
// tests/test-kpool-cache.cpp. It is derived from (pos, seq) per occupied cell and never from
// cell order, which is exactly what makes it correct under -np 2 --kv-unified, where a block
// of r consecutive POSITIONS is not r consecutive cells. Nothing in it is architecture
// specific, and nothing in it changes here beyond one added output (new_blk_pos).
//
// WHY MILESTONE 1 NEEDS NO NEW GGML OP
// ------------------------------------
// Because the pooled block key is cached, pooling only ever runs for the blocks an ubatch
// COMPLETES -- one every r tokens at decode, a padded row otherwise -- so the reference's own
// get_rows + r slice-adds + scale is already cheap and a fused pool-and-score kernel would be
// optimising a term that is not there. The score then reuses ggml_kpool_score with a constant
// weight vector: that op computes sum_h relu(kq[p,t,h])*w[h,t] + mask[p,t], which with w = 1
// IS this architecture's relu-per-head-then-sum, in a reduction order
// tests/test-kpool-score.cpp already pins bit-for-bit against the unfused chain.
//
// TWO DELIBERATE DIVERGENCES FROM THE REFERENCE, BOTH BOUNDED
// ----------------------------------------------------------
//  1. THE TAIL IS APPENDED, NOT SCORED. The reference forms a block over every r consecutive
//     cells including the last, incomplete one, so the newest < r tokens are scored like any
//     other block and may or may not win a slot. Our grid forms only COMPLETE blocks and
//     appends the tail unconditionally (indexer_kpool_select_tail), which is the reading that
//     guarantees a token can always attend to itself.
//  2. THE BUDGET IS NOT TOPPED UP. The reference's width is a flat top_k + r - 1 = 2051 cells;
//     ours is n_top whole blocks (2048 cells) plus however many tail cells exist, so when the
//     tail is short we attend to up to r - 1 = 3 FEWER cells than the reference, out of 2051.
//     It is deterministic (a function of the position alone) and identical in both QSA arms,
//     so the three-arm gate is unaffected; closing it would mean changing llama_kpool_n_sel
//     and the gather-mask layout, which glm5next shares and test-kpool-cache.cpp pins.
//     tests/test-qsa-gather.cpp measures the exact size of the gap rather than leaving it as
//     prose here.
//
// THE ONE INFERRED QUANTITY
// -------------------------
// The reference rotates the pooled block key at ONE position per block and does not say which
// of the block's r positions that is. The block's FIRST member is the default here (the
// block-start reading); PXA_QSA_BLKPOS picks another member so the selection-recall probe on
// a real seat can settle it by measurement rather than by argument.
//

void llm_build_context::build_qwen4exp_inp_qsa(ggml_cgraph * gf) {
    build_kpool_inputs(gf);

    auto & in = llama_kpool_get_inputs(lctx);
    const auto & d = llama_kpool_get_dims(lctx);

    // qwen4exp's one addition to the k-pool input set: the rope position of the pooled key of
    // each block in the fixed-width new-block group. Only those blocks need a position -- every
    // other pooled key was rotated by the ubatch that completed it. Laid out exactly like this
    // tree's mrope inp_pos, SECTION MAJOR over four sections, because ggml_rope_multi reads it
    // that way.
    in.new_blk_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*(int64_t) d.n_new_g);
    cb(in.new_blk_pos, "qsa_new_blk_pos", -1);
    ggml_set_input(in.new_blk_pos);
    ggml_build_forward_expand(gf, in.new_blk_pos);
}

ggml_tensor * llm_build_context::build_qwen4exp_qsa_select(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * KQ_mask,
        bool          gather,
        int           il) {
    const auto & layer = model.layers[il];
    const auto & in    = llama_kpool_get_inputs(lctx);
    const auto & d     = llama_kpool_get_dims(lctx);

    const int64_t n_ih   = hparams.indexer_n_head;      // 4
    const int64_t n_ei   = hparams.indexer_head_size;   // 128
    const int64_t kpool  = d.kpool;                     // r = 4
    const int64_t n_pool = d.n_pool;
    const int64_t n_new  = d.n_new_g;                   // the GRAPH width: fixed, always >= 1

    ggml_tensor * idx_l = kv_self.idx_l[il];
    GGML_ASSERT(idx_l && idx_l->ne[0] == 2*n_ei &&
                "qsa: the indexer side cache is missing or the wrong width");
    GGML_ASSERT(idx_l->ne[1] > (int64_t) d.sink &&
                "qsa: the indexer side cache has no sink row");

    int sections[GGML_MROPE_SECTIONS];
    std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS,
              sections);
    const int n_rot_l = hparams.rope_n_rot(il);

    //
    // (A) this ubatch's RAW indexer keys, into the side cache
    //
    // A cache row is [ raw(128) | pooled(128) ]. The pooled half is zeroed for the cells this
    // ubatch writes and (re)written below for the blocks this ubatch COMPLETED; every other
    // row's pooled half still holds whatever ubatch completed its block. Writing the whole
    // 2*n_ei row at once keeps the destination a plain contiguous 2D view, which is what the
    // K/V store already does and what every backend's cpy is happiest with.
    //
    {
        ggml_tensor * k_raw = llm_build_lora_mm(lctx, ctx0, layer.index_k_proj, cur);
        cb(k_raw, "qsa_k_raw", il);
        pxa_qsa_prof_tag(k_raw, PXA_QSA_ST_IPROJ, il);

        ggml_tensor * pzero = ggml_fill(ctx0,
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ei, n_tokens), 0.0f);
        pxa_qsa_prof_tag(pzero, PXA_QSA_ST_IPROJ, il);
        ggml_tensor * packed = ggml_concat(ctx0, k_raw, pzero, 0);

        // this tree's cache hands out a CONTIGUOUS run of n_tokens cells from kv_head, so the
        // write is a plain view -- no scatter, no k_idxs input
        ggml_tensor * dst = ggml_view_2d(ctx0, idx_l, 2*n_ei, n_tokens,
                idx_l->nb[1], idx_l->nb[1]*kv_head);
        ggml_tensor * wr = ggml_cpy(ctx0, packed, dst);
        pxa_qsa_prof_tag(wr, PXA_QSA_ST_IPROJ, il, /*terminal*/ true);
        ggml_build_forward_expand(gf, wr);
    }

    const int64_t n_cells    = idx_l->ne[1];
    ggml_tensor * raw_all    = ggml_view_2d(ctx0, idx_l, n_ei, n_cells, idx_l->nb[1], 0);
    ggml_tensor * pooled_all = ggml_view_2d(ctx0, idx_l, n_ei, n_cells, idx_l->nb[1],
                                            ggml_row_size(idx_l->type, n_ei));

    //
    // (B) pool the blocks this ubatch completed: the mean of the r RAW member keys, then the
    //     indexer's own RMS norm, then IMRoPE at the block's position.
    //
    ggml_tensor * pooled_new = nullptr;
    {
        ggml_tensor * rows = ggml_get_rows(ctx0, raw_all,
                ggml_reshape_1d(ctx0, in.new_pool_idxs, kpool*n_new));
        rows = ggml_reshape_3d(ctx0, rows, n_ei, kpool, n_new);

        // The mean over the r members, in ONE reduction instead of r cont'd views and r-1 adds.
        //
        // That chain was 2*r nodes per layer per token -- 8 of the 20 the profiler attributes to
        // pooling, 96 of the ~768 QSA adds to the graph -- to average four 128-wide rows, on a
        // step whose cost is per-launch host work. ggml_sum_rows reduces ne[0] in ASCENDING
        // INDEX ORDER, so transposing the member axis into ne[0] first gives the identical
        // summation tree ((k0+k1)+k2)+k3 the explicit adds gave: the pooled key is bit-identical,
        // not merely equal, which is what keeps the member order fixed for every backend.
        pooled_new = ggml_cont(ctx0, ggml_permute(ctx0, rows, 1, 0, 2, 3));  // [kpool,n_ei,n_new]
        pooled_new = ggml_sum_rows(ctx0, pooled_new);                        // [1,n_ei,n_new]
        pooled_new = ggml_scale(ctx0, ggml_reshape_2d(ctx0, pooled_new, n_ei, n_new),
                                1.0f/(float) kpool);
        cb(pooled_new, "qsa_pool_new", il);
        pxa_qsa_prof_tag(rows,       PXA_QSA_ST_POOL, il);
        pxa_qsa_prof_tag(pooled_new, PXA_QSA_ST_POOL, il);

        // rope wants [n_dims, n_head, n_tokens]: one "token" per block, one head
        pooled_new = ggml_reshape_3d(ctx0, pooled_new, n_ei, 1, n_new);
        pooled_new = llm_build_norm(ctx0, pooled_new, hparams, layer.index_k_norm, nullptr,
                                    LLM_NORM_RMS, cb, il);
        pooled_new = ggml_rope_multi(ctx0, pooled_new, in.new_blk_pos, nullptr,
                n_rot_l, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        pooled_new = ggml_reshape_2d(ctx0, pooled_new, n_ei, n_new);
        cb(pooled_new, "qsa_pool_new_roped", il);

        if (in.new_pool_rep) {
            // write back BEFORE the gather below, so one get_rows picks up both the blocks
            // just completed and the ones some earlier ubatch completed
            ggml_tensor * wb = ggml_set_rows(ctx0, pooled_all, pooled_new, in.new_pool_rep);
            pxa_qsa_prof_tag(wb, PXA_QSA_ST_POOL, il, /*terminal*/ true);
            ggml_build_forward_expand(gf, wb);
        }
    }

    ggml_tensor * pooled;
    if (llama_kpool_get_cache_safe(lctx)) {
        pooled = ggml_get_rows(ctx0, pooled_all, in.pool_cells);
    } else {
        // cells shared between sequences: the block grid is sequence-relative and one cached
        // pooled key cannot serve both, so every block was recomputed above and the cache is
        // not read at all
        GGML_ASSERT(n_new <= n_pool);
        ggml_tensor * pad = ggml_fill(ctx0,
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ei, n_pool - n_new), 0.0f);
        // rows [d.n_new, n_new_g) of pooled_new are padding: they land on block indices
        // >= n_pool_real, which pool_mask masks with -inf for every token.
        pooled = ggml_concat(ctx0, pooled_new, pad, 1);
    }
    pooled = ggml_reshape_3d(ctx0, pooled, n_ei, 1, n_pool);
    cb(pooled, "qsa_pool_k", il);
    pxa_qsa_prof_tag(pooled, PXA_QSA_ST_POOL, il);

    //
    // (C) score every block for every token: the sum over the 4 indexer heads of relu(dot).
    //
    ggml_tensor * score;
    {
        ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.index_q_proj, cur);
        pxa_qsa_prof_tag(q, PXA_QSA_ST_IPROJ, il);
        q = ggml_reshape_3d(ctx0, q, n_ei, n_ih, n_tokens);
        q = llm_build_norm(ctx0, q, hparams, layer.index_q_norm, nullptr, LLM_NORM_RMS, cb, il);
        q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
                n_rot_l, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q, "qsa_q", il);
        pxa_qsa_prof_tag(q, PXA_QSA_ST_IPROJ, il);

        ggml_tensor * q_p = ggml_permute(ctx0, ggml_cont(ctx0, q), 0, 2, 1, 3); // [n_ei,n_tok,n_ih]
        ggml_tensor * k_p = ggml_permute(ctx0, pooled,             0, 2, 1, 3); // [n_ei,n_pool,1]
        pxa_qsa_prof_tag(q_p, PXA_QSA_ST_IPROJ, il, /*terminal*/ true);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);            // [n_pool, n_tokens, n_ih]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        pxa_qsa_prof_tag(kq, PXA_QSA_ST_SCORE, il);

        // This architecture's indexer has NO per-head weight tensor and no scale, so the fused
        // op's weight vector is a constant 1. (Whatever scale the architecture applies is a
        // single positive factor and top-k is invariant under one, so the SELECTION -- the only
        // thing that leaves this stage -- cannot depend on getting it right.)
        static const bool fused = [] {
            const char * e = getenv("PXA_QSA_FUSED_SCORE");
            return e == nullptr || atoi(e) != 0;   // default ON
        }();
        const bool can_fuse = fused && n_ih >= 4 && n_ih <= 32 && (n_ih & (n_ih - 1)) == 0;

        if (can_fuse) {
            ggml_tensor * ones = ggml_fill(ctx0,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ih, n_tokens), 1.0f);
            pxa_qsa_prof_tag(ones, PXA_QSA_ST_SCORE, il);
            score = ggml_kpool_score(ctx0, kq, ones, in.pool_mask);
        } else {
            ggml_tensor * s = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3)); // [n_ih,n_tok,n_pool]
            s = ggml_relu(ctx0, s);
            s = ggml_sum_rows(ctx0, s);                                            // [1,n_tok,n_pool]
            s = ggml_cont(ctx0, ggml_permute(ctx0, s, 2, 1, 0, 3));                // [n_pool,n_tok,1]
            score = ggml_add(ctx0, s, in.pool_mask);
        }
        score = ggml_reshape_2d(ctx0, score, n_pool, n_tokens);
        cb(score, "qsa_score", il);
        pxa_qsa_prof_tag(score, PXA_QSA_ST_SCORE, il, /*terminal*/ true);
    }

    //
    // (D) the top n_top BLOCKS, expanded to their member cells, then the tail.
    //
    // Scoring and sorting BLOCKS and expanding only the winners is the structural half of the
    // win: the n_kv-wide cell-score vector the reference builds is never materialised. That
    // this lands on the same cells as the reference's cell-wise cut -- including a block that
    // straddles the query position, and including the tie order inside a block -- is what
    // tests/test-qsa-gather.cpp exists to assert.
    //
    // This tree's ggml_top_k IS ggml_argsort(DESC) viewed to the first k, so the selection is
    // already ordered by descending score; above 1024 columns that argsort is a CUB segmented
    // radix sort, which is stable, so equal block scores keep ascending block order and the
    // order the gather then reduces in is reproducible run to run.
    //
    ggml_tensor * sel_idx;
    {
        // PXA_QSA_FAST_TOPK (default OFF): the same selection by a radix SELECT instead of a
        // full sort. ggml_top_k here IS ggml_argsort(DESC) viewed to n_top, and above 1024
        // columns that argsort is a CUB radix sort of the WHOLE row -- 21,632 keys per token
        // per layer at 86k fill, in a dozen-plus kernel launches, to keep 512. ggml_qsa_top_k
        // produces the identical set AND order (ties by ascending block index) in one launch;
        // the arms must therefore be byte-identical, which is what makes it measurable as a
        // pure speed lever. OFF until a window has measured it on a card.
        static const bool fast_topk = [] {
            const char * e = getenv("PXA_QSA_FAST_TOPK");
            return e != nullptr && atoi(e) != 0;
        }();
        ggml_tensor * top;
        if (fast_topk && d.n_top <= 2048 && (int64_t) d.n_top <= score->ne[0]) {
            top = ggml_qsa_top_k(ctx0, score, (int) d.n_top);
        } else {
            top = ggml_cont(ctx0, ggml_top_k(ctx0, score, (int) d.n_top));
        }
        cb(top, "qsa_top_k", il);
        pxa_qsa_prof_tag(top, PXA_QSA_ST_TOPK, il, /*terminal*/ true);

        sel_idx = ggml_get_rows(ctx0, in.pool_idxs,
                ggml_reshape_1d(ctx0, top, (int64_t) d.n_top*n_tokens));   // [kpool, n_top*n_tok]
        pxa_qsa_prof_tag(sel_idx, PXA_QSA_ST_EXPAND, il);
        sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*(int64_t) d.n_top, n_tokens);

        if (hparams.indexer_kpool_select_tail) {
            // the newest < r tokens belong to no complete block yet, and one of them is always
            // the token doing the attending
            sel_idx = ggml_concat(ctx0, sel_idx, in.tail_idxs, 0);
        }
    }
    GGML_ASSERT(sel_idx->ne[0] == (int64_t) d.n_sel);

    if (gather) {
        cb(sel_idx, "qsa_sel_idx", il);
        pxa_qsa_prof_tag(sel_idx, PXA_QSA_ST_EXPAND, il, /*terminal*/ true);
        return sel_idx;
    }

    //
    // The control arm: turn the selection into an additive mask of length n_kv and fold in
    // causality. This IS the reference algorithm -- unmask the selected cells into an all -inf
    // mask, add the causal mask, attend densely -- so arm 2 approximates nothing.
    //
    // The mask is built one column wide and (n_kv + 1) rows tall so ggml_set_rows can scatter
    // along ne[1]; the extra row IS the padding sentinel the host writes into pool_idxs and
    // tail_idxs, so a padded slot lands there and never unmasks a live cell.
    //
    ggml_tensor * seed = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    ggml_tensor * mask_all = ggml_repeat_4d(ctx0, ggml_fill(ctx0, seed, -INFINITY),
                                            1, n_kv + 1, n_tokens, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + 1, n_tokens);

    ggml_tensor * zeros = ggml_repeat_4d(ctx0, ggml_fill(ctx0, seed, 0.0f),
                                         1, d.n_sel, n_tokens, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, d.n_sel, n_tokens);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros,
            ggml_reshape_3d(ctx0, sel_idx, d.n_sel, n_tokens, 1));
    sel = ggml_view_2d(ctx0, sel, n_kv, n_tokens, sel->nb[2], 0);

    // The attention reads a mask of the padded height its own kernels were built for, and (with
    // flash attention on) of the type build_inp_KQ_mask handed out. Match both, so that the only
    // thing this arm changes about the dense path is WHICH cells are visible.
    const int64_t n_pad = KQ_mask->ne[1];
    GGML_ASSERT(n_pad >= n_tokens);
    if (n_pad > n_tokens) {
        ggml_tensor * tailpad = ggml_repeat_4d(ctx0, ggml_fill(ctx0, seed, -INFINITY),
                                               n_kv, n_pad - n_tokens, 1, 1);
        sel = ggml_concat(ctx0, sel, tailpad, 1);
    }

    // Causality is folded in HERE, not left to the attention: a padded selection slot points at
    // the SENTINEL row and is dropped by the view above, but a block whose cells this token may
    // only partly see is dropped whole by pool_mask, and every cell of another sequence has to
    // be masked by the same rule the dense path uses. Adding the causal mask is that rule.
    ggml_tensor * causal = KQ_mask;
    if (causal->type != GGML_TYPE_F32) {
        causal = ggml_cast(ctx0, causal, GGML_TYPE_F32);
    }
    sel = ggml_add(ctx0, sel, causal);
    if (KQ_mask->type != GGML_TYPE_F32) {
        sel = ggml_cast(ctx0, sel, KQ_mask->type);
    }
    cb(sel, "qsa_sel_mask", il);
    // The control arm's n_kv-wide mask IS its expansion stage, and it is a terminal: the dense
    // attention that consumes it is tagged ATTN through Q/K/V, not through the mask.
    pxa_qsa_prof_tag(sel, PXA_QSA_ST_EXPAND, il, /*terminal*/ true);

    return sel;
}

ggml_tensor * llm_build_context::build_qwen4exp_qsa_attention(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * KQ_mask,
        float         KQ_scale,
        int           il) {
    const auto & layer = model.layers[il];
    const auto & d     = llama_kpool_get_dims(lctx);

    // The gather only pays off, and is only correct to prefer, when the context is genuinely
    // longer than the selection and the ubatch is decode-shaped; llama_kpool_build_plan decides
    // that on the OCCUPIED CELL COUNT so the reserve graph and the decode graph agree.
    const bool gather = d.gather && llama_qsa_gather_enabled();

    // The same three projections, two norms and IMRoPE build_std_attention applies to this arch,
    // spelled out here because the attention that follows is not llm_build_kqv. Every node up to
    // and including the rope is the one the dense path builds; if that stops being true, arm 2
    // stops being a control for arm 1.
    auto [Qcur, Kcur, Vcur, gate] = llm_build_mul_mat_qkv_gated(gf, cur,
            layer.wq, layer.wk, layer.wv, layer.attn_q_norm, layer.attn_k_norm, il);

    int sections[GGML_MROPE_SECTIONS];
    std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS,
              sections);
    const int n_rot_l = hparams.rope_n_rot(il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot_l, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot_l, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "Qcur_roped", il);
    cb(Kcur, "Kcur_roped", il);
    // The qkv projections themselves are inside llm_build_mul_mat_qkv_gated and are identical
    // in both arms and in the dense build, so they stay in the profiler's "other" bucket; from
    // the rope on, the nodes are this path's own.
    pxa_qsa_prof_tag(Qcur, PXA_QSA_ST_ATTN, il);
    pxa_qsa_prof_tag(Kcur, PXA_QSA_ST_ATTN, il);
    pxa_qsa_prof_tag(Vcur, PXA_QSA_ST_ATTN, il);

    // The selection is built from `cur`, the block input, so it depends on none of the above; it
    // is built here only because its own side-cache write has to precede its pooling read.
    ggml_tensor * sel = build_qwen4exp_qsa_select(gf, cur, inp_pos, KQ_mask, gather, il);

    ggml_tensor * out;
    if (!gather) {
        // The reference: the dense path with the selection as its mask. Node for node what
        // PXA_QSA=0 builds, KQ_mask swapped -- so it keeps flash attention and every other
        // lever the shipping build has, and the A/B against arm 3 isolates the gather alone.
        out = llm_build_kv(ctx0, lctx, kv_self, gf, /*wo*/ nullptr, /*wo_b*/ nullptr,
                Kcur, Vcur, Qcur, sel, n_tokens, kv_head, n_kv, KQ_scale, cb, il,
                /*sinks*/ nullptr, /*n_swa*/ 0);
    } else {
        //
        // The gather. The K/V store must be EXPANDED before the gather nodes are built: the
        // gather reads rows this same ubatch has just written, and a graph's order is its
        // expansion order, not its data dependencies.
        //
        GGML_ASSERT(!cparams.k_cache_hadamard && !cparams.v_cache_hadamard &&
                    "qsa: the gather path does not implement the KV cache hadamard rotation");

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, Kcur, Vcur,
                           n_tokens, kv_head, cb, il);

        const int64_t n_sel    = d.n_sel;
        const int64_t n_head_l = hparams.n_head(il);
        const int64_t n_hkv    = hparams.n_head_kv(il);
        const int64_t hk       = hparams.n_embd_head_k(il);
        const int64_t hv       = hparams.n_embd_head_v(il);

        GGML_ASSERT(!kv_self.v_trans && "qsa: the gather needs an untransposed V cache");
        GGML_ASSERT(n_hkv > 0 && n_head_l % n_hkv == 0);

        // One cell is ONE contiguous row across both KV heads: k_l is [n_embd_head_k,
        // n_head_kv*kv_size] and v_l is flat n_embd_v_gqa per cell, so a selected cell is one
        // get_rows row in each.
        ggml_tensor * rows_k = ggml_view_2d(ctx0, kv_self.k_l[il], hk*n_hkv, kv_self.size,
                ggml_row_size(kv_self.k_l[il]->type, hk*n_hkv), 0);
        ggml_tensor * rows_v = ggml_view_2d(ctx0, kv_self.v_l[il], hv*n_hkv, kv_self.size,
                ggml_row_size(kv_self.v_l[il]->type, hv*n_hkv), 0);

        ggml_tensor * idx = ggml_reshape_1d(ctx0, sel, n_sel*n_tokens);

        // ggml_get_rows returns F32 in this tree, so the compact K/V is staged through F32 --
        // twice the bytes a type-preserving gather would write, and the two conts below cost
        // more again. It is still ~7.5x less than the dense path reads at 140k; a typed gather
        // is the first thing milestone 2 removes.
        ggml_tensor * k_g = ggml_get_rows(ctx0, rows_k, idx);   // [hk*n_hkv, n_sel*n_tokens]
        ggml_tensor * v_g = ggml_get_rows(ctx0, rows_v, idx);
        cb(k_g, "qsa_k_gathered", il);
        pxa_qsa_prof_tag(k_g, PXA_QSA_ST_GATHER, il);
        pxa_qsa_prof_tag(v_g, PXA_QSA_ST_GATHER, il);

        // within one cell the layout is head-major, hence [hk, n_hkv, n_sel, n_tokens]
        k_g = ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_4d(ctx0, k_g, hk, n_hkv, n_sel, n_tokens), 0, 2, 1, 3));
        v_g = ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_4d(ctx0, v_g, hv, n_hkv, n_sel, n_tokens), 0, 2, 1, 3));
        pxa_qsa_prof_tag(k_g, PXA_QSA_ST_GATHER, il, /*terminal*/ true);
        pxa_qsa_prof_tag(v_g, PXA_QSA_ST_GATHER, il, /*terminal*/ true);

        // the token axis lives in ne[3] so ONE chain covers every token's own selection; the
        // GQA broadcast is ne[2], n_head % n_head_kv == 0
        ggml_tensor * q_g = ggml_permute(ctx0, Qcur, 0, 2, 3, 1);   // [hk, 1, n_head, n_tokens]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_g, q_g);            // [n_sel, 1, n_head, n_tok]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        pxa_qsa_prof_tag(kq, PXA_QSA_ST_ATTN, il);

        // ggml_soft_max_ext takes a 2D mask only, and the gather mask is 4D (it varies per
        // token, not per row), so it is added explicitly. softmax(x*scale + mask) either way.
        // The mask is what keeps a PADDED slot -- which points at a real cell, because the
        // gather has to stay in bounds -- from being attended to twice.
        kq = ggml_add(ctx0, ggml_scale(ctx0, kq, KQ_scale),
                      llama_kpool_get_inputs(lctx).gather_mask);
        kq = ggml_soft_max(ctx0, kq);
        cb(kq, "qsa_kq_soft_max", il);

        ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_g)); // [n_sel, hv, n_hkv, n_tok]
        pxa_qsa_prof_tag(v_t, PXA_QSA_ST_ATTN, il);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v_t, kq);                // [hv, 1, n_head, n_tok]
        cb(kqv, "qsa_kqv", il);
        pxa_qsa_prof_tag(kqv, PXA_QSA_ST_ATTN, il);

        out = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));     // [hv, n_head, 1, n_tok]
        out = ggml_reshape_2d(ctx0, out, hv*n_head_l, n_tokens);
    }
    cb(out, "kqv_out", il);

    // the attention output gate qwen4exp packs into wq, then wo: the tail of
    // build_std_attention's gated branch, unchanged
    GGML_ASSERT(gate && "qsa: qwen4exp packs an output gate into wq");
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, gate));
    cb(out, "qkv_gated", il);

    out = llm_build_lora_mm(lctx, ctx0, layer.wo, out);
    cb(out, "attn_out", il);
    // the end of the QSA region: everything after this is the trunk again
    pxa_qsa_prof_tag(out, PXA_QSA_ST_ATTN, il, /*terminal*/ true);

    return out;
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

        // PXA_MTP_BATCH_SLOTS_ROWS_v1: one hidden row per BATCH token, for every MTP op type.
        // The draft-gen graph used to allocate a single [n_embd] row here because every draft decode
        // was a 1-row batch; a multi-row draft step then concatenated it against [n_embd, n_tokens]
        // token embeddings and aborted. See common/pxa-mtp-batch-slots.h. No-op at n_tokens == 1.
        ggml_tensor * hidden_states_from_main_model =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
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

    // PXA_QSA. The k-pool plan for THIS batch: every QSA input is sized from it and
    // llama_kpool_set_inputs() fills those same tensors from the same plan, so it MUST be
    // built before anything below reads a size. Nothing is built and nothing is planned when
    // the lever is off, which is what makes PXA_QSA=0 the shipping graph node for node.
    const bool qsa = llama_qsa_planned(lctx);
    if (qsa) {
        llama_kpool_build_plan(lctx, batch, is_reserve);
    }
    pxa_qsa_prof_begin_graph();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // the recurrent-path inputs the fused delta-net kernel reads (same set qwen35moe builds)
    // PXA_RS_RING: when the ring is armed the tensor carries TWO index vectors: [0, n_tokens) are
        // the write rows (plane 0, absolute seq_id -- today's meaning) and [n_tokens, 2*n_tokens) are
        // the read rows (rs_idx[seq]*plane_rows + seq). Ring off -> the shape is unchanged.
        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens * (lctx.kv_self.rs_ring_armed() ? 2 : 1));
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

    if (qsa) {
        build_qwen4exp_inp_qsa(gf);

        // The DECODE gather path reads the indexer's own gather_mask and never touches the
        // causal mask, so without this ggml-alloc leaves inp_KQ_mask bufferless and
        // llama_set_inputs's unconditional fill trips on it. Expanding the leaf costs one
        // node and keeps that fill honest. (Same fix glm5next needed, same reason.)
        ggml_build_forward_expand(gf, KQ_mask);
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
        } else if (qsa && model.layers[il].index_k_proj != nullptr &&
                   kv_self.idx_l[il] != nullptr) {
            // PXA_QSA: the same block, attending only to the cells this layer's own trained
            // indexer selects. See the header comment above build_qwen4exp_inp_qsa().
            cur = build_qwen4exp_qsa_attention(gf, cur, inp_pos, KQ_mask, KQ_scale, il);
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

    if (pxa_qsa_prof_on()) {
        const llama_kpool_dims pd = qsa ? llama_kpool_get_dims(lctx) : llama_kpool_dims();
        pxa_qsa_prof_seal(gf, (int) n_tokens, (int) n_kv, (int) pd.n_pool, (int) pd.n_top,
                          (int) pd.n_sel, pd.gather);
    }

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
