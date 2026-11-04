// PXA LFM port: the short-conv operator of LiquidAI LFM2 / LFM2-MoE / LFM3.
//
// Math follows mainline llama.cpp src/models/lfm-base.cpp build_lfm_shortconv (MIT, see NOTICE):
//   [b | c | x] = in_proj @ h;  y = c * causal_conv(b * x);  out = out_proj @ y
// The depthwise causal conv has l_cache taps, so the recurrent state is the last (l_cache - 1) values of b*x per
// channel. It lives in this fork's hybrid storage: kv_self.s_l[il] holds one row per sequence, gathered and
// scattered with the same row indices, conv seq-map and reset mask the DeltaNet / Mamba-2 layers use, so every
// host-side input (llama_set_inputs) is shared unchanged. The conv is GGML_OP_SSM_CONV (it also emits the new
// window), exactly as the unfused Mamba-2 path uses it.
#include "llama-delta-net.h"
#include "llama-model.h"
#include "llama-context.h"

ggml_tensor * delta_net::build_layer_shortconv(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * cur, int il, const llm_build_cb & cb) const {
    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    const auto & layer = model.layers[il];

    GGML_ASSERT(hparams.n_shortconv_l_cache >= 2);
    GGML_ASSERT(hparams.is_recurrent(il));
    GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr && lctx.inp_conv_seq_map != nullptr && lctx.inp_qnext_state_mask != nullptr);
    GGML_ASSERT(!kv_self.rs_ring_armed() && "PXA_RS_RING is not wired for short-conv layers");

    const int64_t n_cols    = hparams.n_shortconv_l_cache;     // conv taps
    const int64_t d_conv    = n_cols - 1;                      // window kept per channel
    const int64_t n_embd    = hparams.n_embd;
    const int64_t state_dim = d_conv * n_embd;
    GGML_ASSERT(hparams.n_embd_v_s() == (uint32_t) state_dim);

    const int64_t n_tok = batch.n_tokens;

    // per-sequence routing, exactly as build_layer_mamba2 chooses it
    int64_t n_seqs = 1;
    if (!all_same_seq) {
        GGML_ASSERT(seq_decomp.ok && "short-conv mixed batch is not decomposable (interleaved sequences)");
        GGML_ASSERT(seq_decomp.uniform_tok && "short-conv batched conv requires a uniform tokens-per-sequence count");
        n_seqs = seq_decomp.n_seqs;
        GGML_ASSERT(n_seqs >= 1 && n_tok % n_seqs == 0);
    }
    // one set of routing views per graph (a model has either Mamba-2 or short-conv layers, never both, so the
    // Mamba-2 view cache is reused): fewer scheduler split inputs, same data
    ggml_tensor * state_row_idx = nullptr;
    ggml_tensor * conv_seq_map  = nullptr;
    ggml_tensor * state_mask    = nullptr;
    if (m2_view_ctx == ctx0 && m2_view_n_seqs == n_seqs && m2_view_n_tok == n_tok) {
        state_row_idx = m2_state_row;
        conv_seq_map  = m2_conv_map;
        state_mask    = m2_state_mask;
    } else {
        state_row_idx = ggml_view_1d(ctx0, lctx.inp_s_seq_qnext, n_seqs, 0);
        conv_seq_map  = ggml_view_2d(ctx0, lctx.inp_conv_seq_map, n_seqs, n_tok, lctx.inp_conv_seq_map->nb[1], 0);
        state_mask    = ggml_view_2d(ctx0, lctx.inp_qnext_state_mask, 1, n_seqs, lctx.inp_qnext_state_mask->nb[1], 0);
        m2_view_ctx = ctx0; m2_view_n_seqs = n_seqs; m2_view_n_tok = n_tok;
        m2_state_row = state_row_idx; m2_conv_map = conv_seq_map; m2_state_mask = state_mask;
    }

    ggml_tensor * state_all = kv_self.s_l[il];
    GGML_ASSERT(state_all != nullptr && state_all->type == GGML_TYPE_F32);

    // gather this batch's windows and zero the ones whose sequence starts here
    ggml_tensor * state = ggml_get_rows(ctx0, state_all, state_row_idx);   // [state_dim, n_seqs]
    state = ggml_mul(ctx0, state, state_mask);
    cb(state, "shortconv_state_in", il);
    ggml_tensor * conv_states = ggml_reshape_3d(ctx0, state, d_conv, n_embd, n_seqs);

    // {n_embd, 3*n_embd} @ {n_embd, n_tok} => {3*n_embd, n_tok} = [b | c | x]
    cur = llm_build_context::build_pxqn_rht(ctx0, lctx.model.hparams, cur, il, PXQN_SITE_ATTN_IN);   // PXQN attn_in (in_proj; identity unless the file rotates the site)
    ggml_tensor * bcx = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_in, cur);
    cb(bcx, "shortconv_in_proj", il);
    const size_t eb = ggml_element_size(bcx);
    ggml_tensor * b = ggml_view_2d(ctx0, bcx, n_embd, n_tok, bcx->nb[1], 0 * n_embd * eb);
    ggml_tensor * c = ggml_view_2d(ctx0, bcx, n_embd, n_tok, bcx->nb[1], 1 * n_embd * eb);
    ggml_tensor * x = ggml_view_2d(ctx0, bcx, n_embd, n_tok, bcx->nb[1], 2 * n_embd * eb);

    ggml_tensor * bx = ggml_mul(ctx0, b, x);                                // contiguous [n_embd, n_tok]
    cb(bx, "shortconv_bx", il);

    ggml_tensor * conv_w = layer.ssm_conv1d;                                // {l_cache, n_embd}
    if (conv_w->type != GGML_TYPE_F32) {
        conv_w = ggml_cast(ctx0, conv_w, GGML_TYPE_F32);
    }
    // conv over the (state window | new tokens) sequence; the op also emits the new window
    ggml_tensor * conv_raw = ggml_ssm_conv(ctx0, conv_states, bx, conv_w, conv_seq_map, nullptr);
    cb(conv_raw, "shortconv_conv_raw", il);

    ggml_tensor * conv_out = ggml_view_2d(ctx0, conv_raw, n_embd, n_tok, n_embd * ggml_element_size(conv_raw), 0);
    cb(conv_out, "shortconv_conv", il);

    // the op writes n_cols floats per (channel, sequence) after the outputs; the window is the last d_conv of them
    ggml_tensor * new_conv = ggml_view_3d(ctx0, conv_raw, d_conv, n_embd, n_seqs,
            n_cols * ggml_element_size(conv_raw),
            n_cols * n_embd * ggml_element_size(conv_raw),
            (n_embd * n_tok + 1) * ggml_element_size(conv_raw));
    ggml_tensor * new_row = ggml_reshape_2d(ctx0, ggml_cont(ctx0, new_conv), state_dim, n_seqs);
    ggml_tensor * state_cpy = ggml_set_rows(ctx0, state_all, new_row, state_row_idx);
    cb(state_cpy, "shortconv_state_cpy", il);
    ggml_build_forward_expand(gf, state_cpy);

    ggml_tensor * y = ggml_mul(ctx0, c, conv_out);
    cb(y, "shortconv_y", il);

    y = llm_build_context::build_pxqn_rht(ctx0, lctx.model.hparams, y, il, PXQN_SITE_OUT_IN);       // PXQN out_in (out_proj; identity unless the file rotates the site)
    cur = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_out, y);
    cb(cur, "shortconv_out_proj", il);
    return cur;
}
