// PXA_GLM5NEXT: GLM-5.3-Flash ("glm5next") — the graph.
//
// Ported from llama.cpp PR #27773 `src/models/glm5-next.cpp`. Copyright (c) 2023-2026 The ggml
// authors. MIT.
//
// 45 trunk blocks on FOUR parallel residual streams (manifold-constrained hyper-connections,
// hc = 4), interleaved:
//
//   * 34 KDA blocks   — Kimi Delta Attention. A linear-attention delta rule with a FIXED
//                       recurrent state, a short causal conv on each of q/k/v, and a forget
//                       gate that is a VECTOR over the value channel (the one thing that
//                       distinguishes it from our Qwen3-Next / Qwen4Exp Gated DeltaNet, and
//                       the reason GGML_OP_DELTA_NET grew `g_per_channel`).
//   * 11 MLA blocks   — NoPE multi-head latent attention (there is no rope half at all)
//                       over a DeepSeek sparse-attention selection produced by an indexer that
//                       scores POOLS of 4 consecutive tokens.
//
// then 3 leading dense FFN blocks and 42 MoE blocks (288 routed experts, sigmoid noaux_tc
// routing, 1 shared expert, clamped SwiGLU), and finally an UNWEIGHTED MEAN over the four
// residual streams before output_norm.
//
// FIRST-CUT SCOPE, each one asserted rather than silently mis-served:
//   * one sequence per batch (-np 1). The recurrent carry and the pool grid are both
//     per-sequence; multi-sequence batching is the delta_net decomposition plus a multi-row
//     pool layout, and getting either wrong corrupts silently.
//   * -fa off with mla_attn == 1, so the MLA latent has the transposed companion store this
//     path reads for the V side (exactly deepseek2's non-flash MLA arrangement).
//   * no NextN/MTP: block 45 is loaded (with -mtp) but never built. Upstream defers it too.
//
// KNOWN COST, deliberate: expanding the selected pools back into cell indices is a get_rows on
// an I32 table, and this tree's CUDA get_rows has no I32 path (same note as the DSV4
// hash-routed layers in build_deepseek4.cpp), so that ONE node per MLA block falls back to the
// CPU backend and splits the graph 11 times. Correct, not fast; the fix is an I32 get_rows
// kernel, which is a separate change.

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-kv-cache-kpool.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------------------
// the k-pool indexer inputs, shared by every MLA block of the graph
// ---------------------------------------------------------------------------------------

void llm_build_context::build_kpool_inputs(ggml_cgraph * gf) {
    auto & in = llama_kpool_get_inputs(lctx);
    const auto & d = llama_kpool_get_dims(lctx);

    in = llama_kpool_inputs{};

    in.pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, d.n_pool);
    in.pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, d.kpool, d.n_pool);
    in.pool_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d.n_pool, n_tokens);
    in.tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, d.kpool - 1, n_tokens);

    cb(in.pool_cells, "kpool_cells", -1);
    cb(in.pool_idxs,  "kpool_idxs",  -1);
    cb(in.pool_mask,  "kpool_mask",  -1);
    cb(in.tail_idxs,  "kpool_tail",  -1);

    ggml_set_input(in.pool_cells);
    ggml_set_input(in.pool_idxs);
    ggml_set_input(in.pool_mask);
    ggml_set_input(in.tail_idxs);

    // These are filled unconditionally by llama_kpool_set_inputs(), so they must be allocated
    // even in the (impossible) case that no op ends up reading one of them.
    ggml_build_forward_expand(gf, in.pool_cells);
    ggml_build_forward_expand(gf, in.pool_idxs);
    ggml_build_forward_expand(gf, in.pool_mask);
    ggml_build_forward_expand(gf, in.tail_idxs);

    if (d.gather) {
        in.gather_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, d.n_sel, 1, 1, n_tokens);
        cb(in.gather_mask, "kpool_gather_mask", -1);
        ggml_set_input(in.gather_mask);
        ggml_build_forward_expand(gf, in.gather_mask);
    }

    // The new-pool block is built UNCONDITIONALLY and at a fixed width (d.n_new_g >= 1).
    // Building it only when this ubatch happened to complete a pool made the node count
    // alternate 0/1 pools' worth every decode token, and this engine answers a moved node
    // count with a full re-reserve: 62 re-plans in 62 tokens, each one rebuilding the graph
    // twice and reallocating every device's compute buffer. See llama_kpool_dims::n_new_g.
    GGML_ASSERT(d.n_new_g >= 1);
    in.new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, d.kpool, d.n_new_g);
    cb(in.new_pool_idxs, "kpool_new_idxs", -1);
    ggml_set_input(in.new_pool_idxs);
    ggml_build_forward_expand(gf, in.new_pool_idxs);

    if (llama_kpool_get_cache_safe(lctx)) {
        in.new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, d.n_new_g);
        cb(in.new_pool_rep, "kpool_new_rep", -1);
        ggml_set_input(in.new_pool_rep);
        ggml_build_forward_expand(gf, in.new_pool_rep);
    }
}

// ---------------------------------------------------------------------------------------
// KDA: the linear-attention block
// ---------------------------------------------------------------------------------------
//
// With `x` the hc-mixed, RMS-normed input (PORT spec §2.2):
//
//   Q,K,V = SiLU(conv1d_causal(W_{q,k,v} x, kernel 4))     three separate conv states
//   g1    = ssm_f_b (ssm_f_a x) + ssm_dt.bias              [head_dim x n_head]
//   g1    = lower_bound * sigmoid(-(g1 * A))               A = ssm_a = -exp(A_log), per head
//   beta  = ssm_beta x                                     RAW: our op sigmoids it
//   Q,K   = L2normalise(Q), L2normalise(K)                 required by the CUDA path
//   o, s' = delta_net_ext(Q, K, V, g1, beta, s, g_per_channel = 1)
//   out   = W_o ( RMSNorm(o, ssm_norm) * sigmoid(ssm_g_b (ssm_g_a x)) )
//
// The recurrent row reuses this tree's qwen3next state layout exactly — the glm5next hparams
// loader sets ssm_{n_group,d_state,d_inner,dt_rank} so that n_embd_v_s() is
// [ (d_conv-1) * 3*d_inner  |  head_dim*head_dim*n_head ], which is what KDA needs — so the
// three q/k/v convolutions are run as ONE ggml_ssm_conv over a 3*d_inner-wide channel block,
// exactly as the fused qwen3next path does.

ggml_tensor * llm_build_context::build_glm5next_kda(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * state_row_idx,
        ggml_tensor * conv_seq_map,
        ggml_tensor * state_mask,
        int           il) {
    const auto & layer = model.layers[il];

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t n_head_k = n_head;
    const int64_t d_inner  = head_dim * n_head_k;
    const int64_t d_conv   = hparams.ssm_d_conv;

    const int64_t conv_dim       = 3*d_inner;
    const int64_t conv_state_dim = (d_conv - 1) * conv_dim;
    const int64_t ssm_state_dim  = head_dim * head_dim * n_head_k;
    const int64_t state_dim      = conv_state_dim + ssm_state_dim;

    // one sequence per batch (see the scope note at the top)
    const int64_t n_seqs       = 1;
    const int64_t n_seq_tokens = n_tokens;

    GGML_ASSERT((int64_t) hparams.n_embd_v_s() == state_dim &&
                "glm5next: the recurrent row does not match the KDA state layout -- check the "
                "ssm_{n_group,d_state,d_inner,dt_rank} the hparams loader derives");

    ggml_tensor * s_l = kv_self.s_l[il];
    GGML_ASSERT(s_l && "glm5next: no recurrent state allocated for a KDA layer");
    const uint32_t n_slots = llama_kv_qnext_state_slots(kv_self);

    //
    // the recurrent row: gather, reset, split into the conv window and the delta state
    //

    ggml_tensor * state_all = ggml_view_2d(ctx0, s_l, state_dim, n_slots, s_l->nb[1], 0);
    ggml_tensor * state_row = ggml_get_rows(ctx0, state_all, state_row_idx);   // [state_dim, 1]
    // per-sequence reset: 0 zeroes a fresh sequence's carried state, 1 keeps it
    state_row = ggml_mul(ctx0, state_row, state_mask);
    cb(state_row, "kda_state_row", il);

    ggml_tensor * conv_states = ggml_reshape_3d(ctx0,
            ggml_view_2d(ctx0, state_row, conv_state_dim, n_seqs, state_row->nb[1], 0),
            d_conv - 1, conv_dim, n_seqs);
    ggml_tensor * state = ggml_reshape_4d(ctx0,
            ggml_view_2d(ctx0, state_row, ssm_state_dim, n_seqs, state_row->nb[1],
                         conv_state_dim*ggml_element_size(state_row)),
            head_dim, head_dim*n_head_k, 1, n_seqs);
    cb(conv_states, "kda_conv_states", il);
    ggml_build_forward_expand(gf, state);

    //
    // Q|K|V through one fused causal convolution
    //

    ggml_tensor * qkv = ggml_concat(ctx0,
            ggml_concat(ctx0,
                llm_build_lora_mm(lctx, ctx0, layer.wq, cur),
                llm_build_lora_mm(lctx, ctx0, layer.wk, cur), 0),
            llm_build_lora_mm(lctx, ctx0, layer.wv, cur), 0);   // [3*d_inner, n_tokens]
    cb(qkv, "kda_qkv", il);

    // ggml_ssm_conv wants ONE [d_conv, conv_dim] F32 weight; the GGUF carries three
    // [d_conv, 1, d_inner] tensors. They are on the quantiser keep-list, so this concat is a
    // few hundred KiB per block and no dequantisation.
    auto conv_w_2d = [&](ggml_tensor * t) {
        ggml_tensor * w = ggml_reshape_2d(ctx0, t, d_conv, d_inner);
        return w->type == GGML_TYPE_F32 ? w : ggml_cast(ctx0, w, GGML_TYPE_F32);
    };
    ggml_tensor * conv_w = ggml_concat(ctx0,
            ggml_concat(ctx0, conv_w_2d(layer.ssm_conv1d_q), conv_w_2d(layer.ssm_conv1d_k), 1),
            conv_w_2d(layer.ssm_conv1d_v), 1);   // [d_conv, 3*d_inner]

    ggml_tensor * conv_raw = ggml_ssm_conv(ctx0, conv_states, qkv, conv_w, conv_seq_map, nullptr);
    cb(conv_raw, "kda_conv_raw", il);

    ggml_tensor * conv_out = ggml_view_2d(ctx0, conv_raw, conv_dim, n_tokens,
            conv_dim*ggml_element_size(conv_raw), 0);
    conv_out = ggml_silu(ctx0, conv_out);
    cb(conv_out, "kda_conv_silu", il);

    const size_t nb1_qkv = ggml_row_size(conv_out->type, conv_dim);
    auto qkv_part = [&](int64_t which) {
        return ggml_view_4d(ctx0, conv_out, head_dim, n_head_k, n_seq_tokens, n_seqs,
                ggml_row_size(conv_out->type, head_dim), nb1_qkv, nb1_qkv*n_seq_tokens,
                ggml_row_size(conv_out->type, which*d_inner));
    };

    // [head_dim, n_head, n_tokens, n_seqs] -> [head_dim, n_tokens, n_head, n_seqs], which is
    // the layout GGML_OP_DELTA_NET wants. Permuting BEFORE the norm is what makes q/k come out
    // contiguous (the op asserts it): rms_norm reduces over ne[0] either way, so the values are
    // identical to normalising first.
    // named to match the reference build's cb() labels, so an eval-callback dump of the two
    // engines can be diffed node for node
    ggml_tensor * q4 = qkv_part(0), * k4 = qkv_part(1), * v4 = qkv_part(2);
    cb(q4, "kda_q_conv", il);
    cb(k4, "kda_k_conv", il);
    cb(v4, "kda_v_conv", il);

    ggml_tensor * Qcur = ggml_permute(ctx0, q4, 0, 2, 1, 3);
    ggml_tensor * Kcur = ggml_permute(ctx0, k4, 0, 2, 1, 3);
    ggml_tensor * Vcur = ggml_permute(ctx0, v4, 0, 2, 1, 3);

    // FLA's L2 norm: scale(rms_norm(x, eps/S), 1/sqrt(S)) == x / sqrt(sum(x^2) + eps).
    // Written out rather than using ggml_l2_norm so it is bit-comparable with the reference
    // build, and so the CUDA delta-net kernel (which does NOT normalise internally) is fed
    // pre-normalised q/k -- see PORT spec §2 note 2.
    {
        constexpr float l2_eps = 1e-6f;
        const float l2_scale = 1.0f/sqrtf((float) head_dim);
        Qcur = ggml_scale(ctx0, ggml_rms_norm(ctx0, Qcur, l2_eps/(float) head_dim), l2_scale);
        Kcur = ggml_scale(ctx0, ggml_rms_norm(ctx0, Kcur, l2_eps/(float) head_dim), l2_scale);
    }
    cb(Qcur, "kda_q", il);
    cb(Kcur, "kda_k", il);

    //
    // the two gates
    //

    // forget gate. ssm_a already holds -exp(A_log), one per head.
    ggml_tensor * g1 = llm_build_lora_mm(lctx, ctx0, layer.ssm_f_b,
                           llm_build_lora_mm(lctx, ctx0, layer.ssm_f_a, cur));
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_k, 1);
    g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_k, n_tokens);
    if (hparams.kda_gate_lower_bound > -INFINITY) {
        // bounded form: lower_bound * sigmoid(-(g1 * A)), which is what a finite
        // kda.gate_lower_bound (-5.0 here) selects
        g1 = ggml_mul(ctx0, g1, A);
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);
    // g1 is already the LOG decay; the op takes exp() of it.
    // The CUDA delta-net kernel indexes g with PACKED strides ([head_dim, n_tokens, n_head]) and
    // asserts contiguity, so the (head, token) transpose has to be materialised here rather than
    // ridden as a view. The CPU kernel is stride-aware and does not need it; one 16 MiB copy per
    // KDA block at ub=512 is not worth branching on the backend for.
    g1 = ggml_cont(ctx0, ggml_permute(ctx0,
            ggml_reshape_4d(ctx0, g1, head_dim, n_head_k, n_seq_tokens, n_seqs), 0, 2, 1, 3));

    // write strength, RAW -- GGML_OP_DELTA_NET applies the sigmoid internally
    ggml_tensor * beta = llm_build_lora_mm(lctx, ctx0, layer.ssm_beta, cur);   // [n_head, n_tokens]
    beta = ggml_permute(ctx0,
            ggml_reshape_4d(ctx0, beta, 1, n_head_k, n_seq_tokens, n_seqs), 0, 2, 1, 3);
    cb(beta, "kda_beta", il);

    //
    // the delta rule
    //

    ggml_tensor * dn = ggml_delta_net_ext(ctx0, Qcur, Kcur, Vcur, g1, beta, state,
                                          nullptr, /*g_per_channel*/ 1);
    cb(dn, "kda_delta_net", il);

    const int64_t out_size = head_dim*n_head_k*n_tokens*n_seqs;
    ggml_tensor * output = ggml_view_4d(ctx0, dn, head_dim, n_head_k, n_seq_tokens, n_seqs,
            ggml_row_size(dn->type, head_dim),
            ggml_row_size(dn->type, head_dim*n_head_k),
            ggml_row_size(dn->type, head_dim*n_head_k*n_seq_tokens), 0);
    ggml_tensor * new_ssm = ggml_view_2d(ctx0, dn, ssm_state_dim, n_seqs,
            ggml_row_size(dn->type, ssm_state_dim), out_size*ggml_element_size(dn));

    cb(new_ssm, "kda_new_state", il);

    // the next conv window is the LAST (d_conv-1) columns of the state block ssm_conv appended
    // after the convolved output (same view build_qkv uses)
    ggml_tensor * new_conv = ggml_view_3d(ctx0, conv_raw, d_conv - 1, conv_dim, n_seqs,
            d_conv*ggml_element_size(conv_raw),
            d_conv*conv_dim*ggml_element_size(conv_raw),
            (conv_dim*n_tokens + 1)*ggml_element_size(conv_raw));
    new_conv = ggml_reshape_2d(ctx0, ggml_cont(ctx0, new_conv), conv_state_dim, n_seqs);

    // one producer, one scatter: no per-token in-place aliasing for ggml-alloc to trip over
    ggml_tensor * new_row = ggml_concat(ctx0, new_conv, new_ssm, 0);   // [state_dim, n_seqs]
    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, s_l, new_row, state_row_idx));

    //
    // output gate and projection
    //

    ggml_tensor * g2 = llm_build_lora_mm(lctx, ctx0, layer.ssm_g_b,
                           llm_build_lora_mm(lctx, ctx0, layer.ssm_g_a, cur));
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_k, n_tokens);
    cb(g2, "kda_g2", il);

    ggml_tensor * o = ggml_reshape_3d(ctx0, ggml_cont(ctx0, output), head_dim, n_head_k, n_tokens);
    o = llm_build_norm(ctx0, o, hparams, layer.ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(o, "kda_normed", il);

    ggml_tensor * gated = ggml_mul(ctx0, o, ggml_sigmoid(ctx0, g2));
    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);

    cur = llm_build_lora_mm(lctx, ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

// ---------------------------------------------------------------------------------------
// the k-pool indexer: score pools, select the top ones, expand them back into cell indices
// ---------------------------------------------------------------------------------------
//
// Returns EITHER an additive attention mask over the n_kv cells (the scatter path, prefill)
// OR the I32 selected-cell table itself (the gather path, decode) -- `*out_gather` says which.

ggml_tensor * llm_build_context::build_glm5next_kpool_select(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * qr,
        ggml_tensor * kq_mask,
        int           il) {
    const auto & layer = model.layers[il];
    const auto & in    = llama_kpool_get_inputs(lctx);
    const auto & d     = llama_kpool_get_dims(lctx);

    const int64_t n_ih   = hparams.indexer_n_head;      // 32
    const int64_t n_ei   = hparams.indexer_head_size;   // 128
    const int64_t kpool  = d.kpool;
    const int64_t n_pool = d.n_pool;
    const int64_t n_new  = d.n_new_g;   // the GRAPH width: fixed, always >= 1 (see n_new_g)

    ggml_tensor * idx_l = kv_self.idx_l[il];
    GGML_ASSERT(idx_l && idx_l->ne[0] == 3*n_ei &&
                "glm5next: the indexer side cache is missing or the wrong width");

    // per-token indexer query, off the q LoRA residual
    ggml_tensor * iq = llm_build_lora_mm(lctx, ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_ei, n_ih, n_tokens);
    cb(iq, "indexer_q", il);

    // per-token key (a real LayerNorm -- it has a bias) and pool gate, cached together
    ggml_tensor * ik = llm_build_lora_mm(lctx, ctx0, layer.indexer_attn_k, cur);
    ik = llm_build_norm(ctx0, ik, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b,
                        LLM_NORM, cb, il);
    cb(ik, "indexer_k", il);

    ggml_tensor * ig = llm_build_lora_mm(lctx, ctx0, layer.indexer_kpool_gate, cur);
    cb(ig, "indexer_gate", il);

    // A cache row is [ key(128) | gate(128) | pooled(128) ]. The pooled third is zeroed here and
    // (re)written below for the pools this ubatch completed; every other pool's third still
    // holds whatever ubatch last completed it.
    {
        ggml_tensor * pzero = ggml_fill(ctx0,
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ei, n_tokens), 0.0f);
        ggml_tensor * packed = ggml_concat(ctx0, ggml_concat(ctx0, ik, ig, 0), pzero, 0);
        // this tree's cache hands out a CONTIGUOUS run of n_tokens cells from kv_head, so the
        // write is a plain view -- no scatter, no k_idxs input
        ggml_tensor * dst = ggml_view_2d(ctx0, idx_l, 3*n_ei, n_tokens,
                idx_l->nb[1], idx_l->nb[1]*kv_head);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, packed, dst));
    }

    const int64_t n_cells = idx_l->ne[1];
    ggml_tensor * kg_all     = ggml_view_2d(ctx0, idx_l, 2*n_ei, n_cells, idx_l->nb[1], 0);
    ggml_tensor * pooled_all = ggml_view_2d(ctx0, idx_l,   n_ei, n_cells, idx_l->nb[1],
                                            ggml_row_size(idx_l->type, 2*n_ei));

    //
    // pool the entries this ubatch completed:
    //   probs  = softmax over the kpool slots of (gate + ape), per channel
    //   pooled = sum_slot probs[slot] * key[member(slot)]
    //
    ggml_tensor * pooled_new = nullptr;
    {
        ggml_tensor * rows = ggml_get_rows(ctx0, kg_all,
                ggml_reshape_1d(ctx0, in.new_pool_idxs, kpool*n_new));
        rows = ggml_reshape_3d(ctx0, rows, 2*n_ei, kpool, n_new);

        ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_ei, kpool, n_new,
                rows->nb[1], rows->nb[2], 0);
        ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_ei, kpool, n_new,
                rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_ei));

        ggml_tensor * logits = ggml_add(ctx0, pg, layer.indexer_kpool_ape);
        logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3));   // [kpool, n_ei, n_new]
        ggml_tensor * probs = ggml_soft_max(ctx0, logits);

        pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
        pooled_new = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pk));        // [1, n_ei, n_new]
        pooled_new = ggml_reshape_2d(ctx0, pooled_new, n_ei, n_new);
        cb(pooled_new, "indexer_pool_k_new", il);

        if (in.new_pool_rep) {
            // write back BEFORE the gather below, so one get_rows picks up both the pools just
            // completed and the ones some earlier ubatch completed
            ggml_build_forward_expand(gf,
                    ggml_set_rows(ctx0, pooled_all, pooled_new, in.new_pool_rep));
        }
    }

    ggml_tensor * pooled;
    if (llama_kpool_get_cache_safe(lctx)) {
        pooled = ggml_get_rows(ctx0, pooled_all, in.pool_cells);
    } else {
        // cells shared between sequences: the pooled key is sequence-relative and cannot be
        // cached, so every pool was recomputed above and the cache is not read at all
        GGML_ASSERT(n_new <= n_pool);
        ggml_tensor * pad = ggml_fill(ctx0,
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ei, n_pool - n_new), 0.0f);
        // rows [d.n_new, n_new) of pooled_new are the padding: they land on pool indices
        // >= n_pool_real, which pool_mask masks with -inf for every token.
        pooled = ggml_concat(ctx0, pooled_new, pad, 1);
    }
    pooled = ggml_reshape_3d(ctx0, pooled, n_ei, 1, n_pool);
    cb(pooled, "indexer_pool_k", il);

    //
    // score every pool for every token, and take the top ones
    //
    ggml_tensor * sel_idx;
    {
        ggml_tensor * weights = llm_build_lora_mm(lctx, ctx0, layer.indexer_proj, cur);
        weights = ggml_scale(ctx0, weights, 1.0f/sqrtf((float) (n_ei*n_ih)));
        cb(weights, "indexer_weights", il);

        ggml_tensor * q_p = ggml_permute(ctx0, iq,     0, 2, 1, 3);   // [n_ei, n_tokens, n_ih]
        ggml_tensor * k_p = ggml_permute(ctx0, pooled, 0, 2, 1, 3);   // [n_ei, n_pool,   1]

        // PXA_KPOOL_FUSED_SCORE.
        //
        // The unfused chain below is what the -ub ceiling is made of: the cont() at its head is
        // a SECOND live [n_ih, n_tokens, n_pool] f32 tensor sitting next to the mul_mat output,
        // and relu/mul add up to two more unless ggml-alloc happens to make them in-place. At
        // n_kv = 32768 (n_pool 8192) and n_tokens = 512 each of those is 512 MiB. The fused op
        // reads kq in its NATIVE layout and writes only [n_pool, n_tokens], so the peak drops to
        // the mul_mat output alone and five nodes per MLA block disappear.
        //
        // ggml_kpool_score reproduces relu -> mul -> sum_rows' arithmetic exactly, including the
        // order of each backend's sum_rows reduction, so this is a bit-identical substitution
        // rather than an approximation (tests/test-kpool-score.cpp asserts that). See the
        // contract note on ggml_kpool_score().
        static const bool fused_score = [] {
            const char * e = getenv("PXA_KPOOL_FUSED_SCORE");
            return e != nullptr && atoi(e) != 0;
        }();

        // PXA_KPOOL_SCORE_TILE: with the fusion in place, the one remaining full-size tensor in
        // this chain is the mul_mat output itself. Every token's score column is independent of
        // every other, so the score can be built in token tiles and the peak becomes
        // n_pool * TILE * n_ih * 4 bytes regardless of the ubatch. That is what lets -ub grow
        // past the point where n_pool * n_tokens * n_ih stops fitting: with TILE = 64 an
        // n_kv = 32768 / -ub 512 prefill needs 64 MiB here instead of 512 MiB.
        //
        // Unlike the fusion this is NOT claimed bit-identical: it changes the N extent of the
        // mul_mat, and the F32 GEMM behind it is free to pick a different internal blocking for
        // a narrower batch. It is a separate lever for that reason, and it needs the
        // logit-spread gate rather than a hash comparison. 0 (the default) means no tiling.
        static const int64_t score_tile = [] {
            const char * e = getenv("PXA_KPOOL_SCORE_TILE");
            const int64_t v = e != nullptr ? atoll(e) : 0;
            return v > 0 ? v : 0;
        }();

        const bool can_fuse = fused_score && n_ih >= 4 && n_ih <= 32 && (n_ih & (n_ih - 1)) == 0;

        ggml_tensor * score = nullptr;
        if (can_fuse && score_tile > 0 && n_tokens > score_tile) {
            for (int64_t t0 = 0; t0 < n_tokens; t0 += score_tile) {
                const int64_t nt = std::min<int64_t>(score_tile, n_tokens - t0);

                ggml_tensor * q_t = ggml_view_3d(ctx0, q_p, n_ei, nt, n_ih,
                        q_p->nb[1], q_p->nb[2], t0*q_p->nb[1]);
                ggml_tensor * w_t = ggml_view_2d(ctx0, weights, n_ih, nt,
                        weights->nb[1], t0*weights->nb[1]);
                ggml_tensor * m_t = ggml_view_2d(ctx0, in.pool_mask, n_pool, nt,
                        in.pool_mask->nb[1], t0*in.pool_mask->nb[1]);

                ggml_tensor * kq_t = ggml_mul_mat(ctx0, k_p, q_t);    // [n_pool, nt, n_ih]
                ggml_tensor * s_t  = ggml_kpool_score(ctx0, kq_t, w_t, m_t);

                score = score ? ggml_concat(ctx0, score, s_t, 1) : s_t;
            }
            GGML_ASSERT(score->ne[0] == n_pool && score->ne[1] == n_tokens);
            score = ggml_reshape_3d(ctx0, score, n_pool, n_tokens, 1);
        } else if (can_fuse) {
            ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);          // [n_pool, n_tokens, n_ih]
            score = ggml_kpool_score(ctx0, kq, weights, in.pool_mask);
            score = ggml_reshape_3d(ctx0, score, n_pool, n_tokens, 1);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);          // [n_pool, n_tokens, n_ih]
            kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));     // [n_ih, n_tokens, n_pool]

            score = ggml_relu(ctx0, kq);
            score = ggml_mul(ctx0, score, weights);
            score = ggml_sum_rows(ctx0, score);                            // [1, n_tokens, n_pool]
            score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));// [n_pool, n_tokens, 1]
            score = ggml_add(ctx0, score, in.pool_mask);
        }
        cb(score, "indexer_score", il);

        // This tree's ggml_top_k IS ggml_argsort(DESC) viewed to the first k, so the selection
        // comes out ORDERED BY DESCENDING SCORE already. Upstream's top_k is unordered and it
        // has to re-sort with argsort(get_rows(score, top_k)) before the gather mask lines up;
        // that whole dance is unnecessary here (and could not be written anyway -- it needs an
        // I32<->F32 ggml_cast this tree does not implement).
        ggml_tensor * top = ggml_top_k(ctx0, ggml_reshape_2d(ctx0, score, n_pool, n_tokens),
                                       (int) d.n_top);                 // I32 [n_top, n_tokens]
        top = ggml_cont(ctx0, top);
        cb(top, "indexer_top_k", il);

        // expand each selected pool into its kpool member cells.
        // NOTE: I32 get_rows -> CPU-only in this tree (see the header note).
        sel_idx = ggml_get_rows(ctx0, in.pool_idxs,
                ggml_reshape_1d(ctx0, top, d.n_top*n_tokens));          // [kpool, n_top*n_tokens]
        sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*d.n_top, n_tokens);

        if (hparams.indexer_kpool_select_tail) {
            // the newest < kpool tokens belong to no complete pool yet and are always attended
            sel_idx = ggml_concat(ctx0, sel_idx, in.tail_idxs, 0);
        }
    }
    GGML_ASSERT(sel_idx->ne[0] == (int64_t) d.n_sel);

    if (d.gather) {
        cb(sel_idx, "indexer_sel_idx", il);
        return sel_idx;
    }

    //
    // scatter: turn the selection into an additive mask of length n_kv, then fold in causality
    //
    // The mask is built one column wide and (n_kv + 1) rows tall so that ggml_set_rows can
    // scatter along ne[1]; the extra row IS the padding sentinel the host writes into
    // pool_idxs / tail_idxs, so a padded slot lands there and never un-masks a live cell.
    //
    // Upstream seeds the two fill values from sel_idx (ggml_cast(view(sel_idx), F32)) to tie the
    // scratch storage's lifetime to this layer's selection. This tree's ggml_cast is a CPY and
    // its dup has no I32 source path, so the seed is a plain F32 scalar instead; ggml_fill
    // overwrites it whole, so the leaf's contents never matter, and the ordinary
    // fill -> repeat -> set_rows -> soft_max dependency chain is what keeps the storage alive.
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

    // Causality is folded in HERE rather than left to the attention, because the selection is
    // what a shared-indexer layer would reuse and it must already be causal.
    sel = ggml_add(ctx0, sel, ggml_view_2d(ctx0, kq_mask, n_kv, n_tokens, kq_mask->nb[1], 0));
    cb(sel, "indexer_sel", il);

    return sel;
}

// ---------------------------------------------------------------------------------------
// MLA: NoPE latent attention over the indexer's selection
// ---------------------------------------------------------------------------------------

ggml_tensor * llm_build_context::build_glm5next_dsa(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * kq_mask,
        int           il) {
    const auto & layer = model.layers[il];
    const auto & d     = llama_kpool_get_dims(lctx);

    const int64_t kv_lora    = hparams.n_lora_kv;            // 512
    const int64_t head_k_mla = hparams.n_embd_head_k_full;   // 256
    const int64_t head_v_mla = hparams.n_embd_head_v_full;   // 256
    const float   kq_scale   = 1.0f/sqrtf((float) head_k_mla);

    GGML_ASSERT(hparams.n_rot == 0 && "glm5next MLA is nope-only");

    ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, layer.wq_a, cur);
    qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "q_resid", il);

    ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, head_k_mla, n_head, n_tokens);

    ggml_tensor * kv_cmpr = llm_build_lora_mm(lctx, ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = llm_build_norm(ctx0, kv_cmpr, hparams, layer.attn_kv_a_norm, nullptr,
                             LLM_NORM_RMS, cb, il);
    cb(kv_cmpr, "kv_cmpr", il);

    // absorb wk_b into the query so the cache only ever holds the compressed latent
    ggml_tensor * q_abs = ggml_permute(ctx0, q, 0, 2, 1, 3);          // [256, n_tokens, n_head]
    q_abs = ggml_mul_mat(ctx0, layer.wk_b, q_abs);                    // [512, n_tokens, n_head]
    cb(q_abs, "q_absorbed", il);

    // the selection. Built from `cur` and `qr`, so it must come before the cache write is
    // consumed but after the row for this token exists -- which is why the write is expanded
    // inside build_glm5next_kpool_select().
    ggml_tensor * sel = build_glm5next_kpool_select(gf, cur, qr, kq_mask, il);

    // append this ubatch's latents to the cache (contiguous run from kv_head)
    {
        ggml_tensor * k_l = kv_self.k_l[il];
        const size_t row = ggml_row_size(k_l->type, kv_lora);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, kv_cmpr,
                ggml_view_2d(ctx0, k_l, kv_lora, n_tokens, row, row*kv_head)));

        ggml_tensor * v_l = kv_self.v_l[il];
        GGML_ASSERT(v_l && "glm5next needs the transposed latent store: run with -fa off and -mla 1");
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_cmpr),
                ggml_view_2d(ctx0, v_l, n_tokens, kv_lora,
                        ggml_row_size(v_l->type, kv_self.size),
                        ggml_row_size(v_l->type, kv_head))));
    }

    ggml_tensor * out;
    if (d.gather) {
        //
        // decode: attend over ONLY the selected latents, so the cost follows top_k rather than
        // the context length. This is the point of DSA, not an optimisation to defer.
        //
        ggml_tensor * k_l = kv_self.k_l[il];
        ggml_tensor * rows = ggml_view_2d(ctx0, k_l, kv_lora, kv_self.size, k_l->nb[1], 0);

        ggml_tensor * k_g = ggml_get_rows(ctx0, rows,
                ggml_reshape_1d(ctx0, sel, d.n_sel*n_tokens));
        k_g = ggml_reshape_4d(ctx0, k_g, kv_lora, d.n_sel, 1, n_tokens);
        cb(k_g, "kv_gathered", il);

        // q_abs is [kv_lora, n_tokens, n_head] here (upstream carries it as
        // [kv_lora, n_head, n_tokens] and permutes 0,2,3,1); the gather needs the token axis in
        // ne[3] so one mul_mat covers every token's own selection.
        ggml_tensor * q_g = ggml_permute(ctx0, q_abs, 0, 3, 2, 1);   // [kv_lora, 1, n_head, n_tok]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_g, q_g);             // [n_sel, 1, n_head, n_tokens]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        // This tree's ggml_soft_max_ext takes a 2D mask only, so the gather mask (which is 4D:
        // it varies per token, not per row) is added explicitly. softmax(x*scale + mask) either
        // way.
        kq = ggml_add(ctx0, ggml_scale(ctx0, kq, kq_scale),
                      llama_kpool_get_inputs(lctx).gather_mask);
        kq = ggml_soft_max(ctx0, kq);
        cb(kq, "kq_soft_max_gathered", il);

        ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, k_g)); // [n_sel, 512, 1, n_tok]
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v_t, kq);                // [512, 1, n_head, n_tok]
        kqv = ggml_mul_mat(ctx0, layer.wv_b, kqv);                      // [256, 1, n_head, n_tok]
        cb(kqv, "kqv_gathered", il);

        out = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));     // [256, n_head, 1, n_tok]
        out = ggml_reshape_2d(ctx0, out, head_v_mla*n_head, n_tokens);
    } else {
        //
        // prefill: an ordinary dense MLA whose mask is the indexer selection (which already
        // carries causality)
        //
        ggml_tensor * k_l = kv_self.k_l[il];
        ggml_tensor * k = ggml_view_2d(ctx0, k_l, kv_lora, n_kv, k_l->nb[1], 0);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q_abs);              // [n_kv, n_tokens, n_head]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx0, kq, sel, kq_scale, 0.0f);
        cb(kq, "kq_soft_max", il);

        ggml_tensor * v_l = kv_self.v_l[il];
        ggml_tensor * v = ggml_view_2d(ctx0, v_l, n_kv, kv_lora,
                ggml_row_size(v_l->type, kv_self.size), 0);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);                // [512, n_tokens, n_head]
        kqv = ggml_mul_mat(ctx0, layer.wv_b, kqv);                    // [256, n_tokens, n_head]
        cb(kqv, "kqv", il);

        out = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));   // [256, n_head, n_tokens]
        out = ggml_reshape_2d(ctx0, out, head_v_mla*n_head, n_tokens);
    }
    cb(out, "kqv_out", il);

    out = llm_build_lora_mm(lctx, ctx0, layer.wo, out);
    cb(out, "attn_out", il);

    return out;
}

// ---------------------------------------------------------------------------------------
// the trunk
// ---------------------------------------------------------------------------------------

ggml_cgraph * llm_build_context::build_glm5next() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc      = hparams.dsv4_hc_mult;
    const int     n_trunk = n_layer - (int) hparams.nextn_predict_layers;

    GGML_ASSERT(hc == 4 && "glm5next: only hyper_connection.count == 4 is implemented");
    GGML_ASSERT(cparams.mtp_op_type == MTP_OP_NONE &&
                "glm5next: the NextN/MTP head is loaded but not built yet");

    // One sequence per batch: both the recurrent carry and the pool grid are per-sequence, and
    // a wrong multi-sequence layout corrupts silently rather than failing.
    {
        bool same = true;
        for (int32_t i = 1; i < batch.n_tokens && same; ++i) {
            const llama_seq_id a = batch.seq_id && batch.seq_id[0] ? batch.seq_id[0][0] : 0;
            const llama_seq_id b = batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][0] : 0;
            same = (a == b) && (!batch.n_seq_id || batch.n_seq_id[i] == 1);
        }
        GGML_ASSERT(same && "glm5next: multi-sequence batches are not implemented yet -- use -np 1");
    }

    // The k-pool plan for THIS batch. Every k-pool input is sized from it, and
    // llama_kpool_set_inputs() fills those same tensors from the same plan, so it MUST be
    // built before anything below reads a size.
    llama_kpool_build_plan(lctx, batch, is_reserve);

    ggml_tensor * inp = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * kq_mask = build_inp_KQ_mask();
    // The DECODE gather path reads the indexer's own gather_mask and never touches the causal
    // mask, so without this ggml-alloc leaves inp_KQ_mask bufferless and llama_set_inputs's
    // unconditional fill trips. Expanding the leaf costs nothing and keeps the fill honest.
    ggml_build_forward_expand(gf, kq_mask);

    // the recurrent-state routing inputs, filled generically by llama_set_inputs()
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

    ggml_tensor * state_row_idx = ggml_view_1d(ctx0, lctx.inp_s_seq_qnext, 1, 0);
    ggml_tensor * conv_seq_map  = ggml_view_2d(ctx0, lctx.inp_conv_seq_map, 1, n_tokens,
                                               lctx.inp_conv_seq_map->nb[1], 0);
    ggml_tensor * state_mask    = ggml_view_2d(ctx0, lctx.inp_qnext_state_mask, 1, 1,
                                               lctx.inp_qnext_state_mask->nb[1], 0);

    build_kpool_inputs(gf);

    // widen the residual to hc parallel streams: [n_embd, hc, n_tokens]
    ggml_tensor * inpL = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_trunk; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_dsv4_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_glm5next_kda(gf, cur, state_row_idx, conv_seq_map, state_mask, il);
        } else {
            cur = build_glm5next_dsa(gf, cur, kq_mask, il);
        }

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_dsv4_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if (il < (int) hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false);
            cb(cur, "ffn_out", il);
        } else {
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
                    cb, il, gf, /*add_input*/ false);
            cb(moe_out, "ffn_moe_out", il);

            // The shared expert's SwiGLU clamps the GATE to [-inf, limit] and the UP to
            // [-limit, limit] and only then multiplies -- it is NOT silu-then-clamp. Built
            // inline (exactly as build_deepseek4.cpp does) so llm_build_ffn's fused SILU path
            // stays byte-untouched for every other architecture.
            ggml_tensor * shexp;
            {
                ggml_tensor * up   = llm_build_lora_mm(lctx, ctx0, layer.ffn_up_shexp,   cur);
                ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_shexp, cur);

                const float limit = hparams.swiglu_limits_shared[il];
                if (limit > 1e-6f) {
                    up   = ggml_clamp(ctx0, up,   -limit,    limit);
                    gate = ggml_clamp(ctx0, gate, -INFINITY, limit);
                }
                ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
                shexp = llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp, act);
            }
            cb(shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_out", il);
    }

    // narrow to the output rows BEFORE collapsing the streams: that saves an hc-wide gather
    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    // GLM5-Next collapses the hc streams with the UNWEIGHTED MEAN -- it has no learned head
    // mixer, unlike DeepSeek-V4's build_dsv4_hc_head().
    {
        ggml_tensor * acc = ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[2], inpL->nb[2], 0);
        for (int64_t s = 1; s < hc; ++s) {
            acc = ggml_add(ctx0, acc,
                    ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[2], inpL->nb[2], s*inpL->nb[1]));
        }
        cur = ggml_scale(ctx0, acc, 1.0f/(float) hc);
    }
    cb(cur, "hc_head", -1);

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
