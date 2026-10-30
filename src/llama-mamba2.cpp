// PXA Nemotron port: the Mamba-2 mixer of Nemotron-H / Nemotron 3 (dense and MoE hybrids).
//
// Math follows mainline llama.cpp src/models/mamba-base.cpp build_mamba2_layer (MIT, see NOTICE):
//   zxBCdt = ssm_in @ x;  xBC = silu(conv1d(xBC) + b);  y = scan(x, dt + dt_b, A, B, C) + D*x;
//   y = rms_norm_grouped(y * silu(z)) * ssm_norm;  out = ssm_out @ y
// The recurrent state lives in this fork's hybrid storage: kv_self.s_l[il] holds one row per sequence,
// [conv window (d_conv-1) x (d_inner + 2*n_group*d_state) | scan state d_state x d_inner], gathered and
// scattered with the same row indices, conv seq-map and reset mask the DeltaNet layers use, so every
// host-side input (llama_set_inputs) is shared unchanged. The scan itself is GGML_OP_SSM_SCAN2.
#include "pxa-env-flag.h"
#include "llama-delta-net.h"
#include "llama-model.h"
#include "llama-context.h"

#include <cstdlib>
#include <cstring>

// PXA_MAMBA2_FUSE (default 1; 0 = the unfused graph): build the mixer from the two in-place ops
// ggml_mamba2_conv / ggml_mamba2_scan (ggml.h). Taken only for a layer whose recurrent state rows and small
// mixer weights live in the same device (non-host) buffer type: the ops write the rows in place, so they must
// run where the rows live, never on a scheduler copy. CPU layers keep the unfused graph.
static bool pxa_mamba2_fuse_env() {
    static const bool on = [] { const char * e = getenv("PXA_MAMBA2_FUSE"); return !(e && pxa_env_value_off(e)); }();
    return on;
}

static bool pxa_mamba2_same_device(const ggml_tensor * t, const ggml_tensor * ref) {
    if (!t || !t->buffer || !ref->buffer) return false;
    if (ggml_backend_buffer_is_host(t->buffer)) return false;
    ggml_backend_buffer_type_t a = ggml_backend_buffer_get_type(t->buffer);
    ggml_backend_buffer_type_t b = ggml_backend_buffer_get_type(ref->buffer);
    return a == b || strcmp(ggml_backend_buft_name(a), ggml_backend_buft_name(b)) == 0;
}

static bool pxa_mamba2_fuse_ok(const llama_layer & layer, const ggml_tensor * state_all, int64_t d_conv, int64_t d_state) {
    if (!pxa_mamba2_fuse_env()) return false;
    if (d_conv != 4 || !(d_state == 64 || d_state == 128 || d_state == 256)) return false;
    if (!state_all->buffer || ggml_backend_buffer_is_host(state_all->buffer)) return false;
    const ggml_tensor * need[] = { layer.ssm_conv1d, layer.ssm_a, layer.ssm_d, layer.ssm_dt_b };
    for (const ggml_tensor * t : need) {
        if (!t || t->type != GGML_TYPE_F32 || !pxa_mamba2_same_device(t, state_all)) return false;
    }
    if (layer.ssm_conv1d_b && (layer.ssm_conv1d_b->type != GGML_TYPE_F32 || !pxa_mamba2_same_device(layer.ssm_conv1d_b, state_all))) {
        return false;
    }
    return true;
}

// PXA_MAMBA2_SHARE_INPUTS (2026-10-21, ledger rig-n4-decode-host-gap): every
// Mamba-2 layer used to make three fresh views of the routing inputs (inp_s_seq_qnext,
// inp_conv_seq_map, inp_qnext_state_mask). The scheduler treats each view as its own split input,
// so a Nemotron-H decode graph carried ~45 inputs, overflowed GGML_SCHED_MAX_SPLIT_INPUTS (32) and
// was cut into two CUDA splits = two graph launches and two input-copy rounds per token. One set
// of views per graph (cached on this delta_net, which is built per graph) holds the same data, so
// the result is bit-identical; only the split shape changes.
// Default: follows PXA_DN_SHARE_INPUTS (AUTO = on for a single-sequence context, n_seq_max <= 1,
// the boundary that lever was measured correct across). PXA_MAMBA2_SHARE_INPUTS=0/1 forces it.
static bool pxa_mamba2_share_inputs(const llama_context & lctx) {
    static const int m = [] { const char * e = getenv("PXA_MAMBA2_SHARE_INPUTS"); return e == nullptr ? -1 : (atoi(e) != 0 ? 1 : 0); }();
    if (m >= 0) return m != 0;
    extern bool pxa_dn_share_inputs_for(uint32_t n_seq_max);
    return pxa_dn_share_inputs_for(lctx.cparams.n_seq_max);
}

ggml_tensor * delta_net::build_layer_mamba2(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * cur, int il, const llm_build_cb & cb) const {
    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    const auto & layer = model.layers[il];

    GGML_ASSERT(hparams.ssm_mamba2);
    GGML_ASSERT(hparams.is_recurrent(il));
    GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr && lctx.inp_conv_seq_map != nullptr && lctx.inp_qnext_state_mask != nullptr);
    GGML_ASSERT(!kv_self.rs_ring_armed() && "PXA_RS_RING is not wired for Mamba-2 layers yet");

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t n_head   = hparams.ssm_dt_rank;
    const int64_t head_dim = d_inner / n_head;
    const int64_t n_group  = hparams.ssm_n_group;
    const int64_t conv_dim = d_inner + 2*n_group*d_state;
    const int64_t conv_state_dim = (d_conv - 1) * conv_dim;
    const int64_t ssm_state_dim  = d_state * d_inner;
    GGML_ASSERT(d_inner % n_head == 0 && d_inner % n_group == 0);
    GGML_ASSERT(hparams.n_embd_v_s() == (uint32_t) (conv_state_dim + ssm_state_dim));

    const int64_t n_tok = batch.n_tokens;

    // per-sequence routing, exactly as build_layer_attn_linear chooses it
    int64_t n_seqs = 1;
    ggml_tensor * state_row_idx = nullptr;
    ggml_tensor * conv_seq_map  = nullptr;
    ggml_tensor * state_mask    = nullptr;
    if (!all_same_seq) {
        GGML_ASSERT(seq_decomp.ok && "Mamba-2 mixed batch is not decomposable (interleaved sequences)");
        GGML_ASSERT(seq_decomp.uniform_tok && "Mamba-2 batched scan requires a uniform tokens-per-sequence count");
        n_seqs = seq_decomp.n_seqs;
        GGML_ASSERT(n_seqs >= 1 && n_tok % n_seqs == 0);
    }
    const bool share = pxa_mamba2_share_inputs(lctx);
    if (share && m2_view_ctx == ctx0 && m2_view_n_seqs == n_seqs && m2_view_n_tok == n_tok) {
        state_row_idx = m2_state_row;
        conv_seq_map  = m2_conv_map;
        state_mask    = m2_state_mask;
    } else {
        state_row_idx = ggml_view_1d(ctx0, lctx.inp_s_seq_qnext, n_seqs, 0);
        conv_seq_map  = ggml_view_2d(ctx0, lctx.inp_conv_seq_map, n_seqs, n_tok, lctx.inp_conv_seq_map->nb[1], 0);
        state_mask    = ggml_view_2d(ctx0, lctx.inp_qnext_state_mask, 1, n_seqs, lctx.inp_qnext_state_mask->nb[1], 0);
        if (share) {
            m2_view_ctx = ctx0; m2_view_n_seqs = n_seqs; m2_view_n_tok = n_tok;
            m2_state_row = state_row_idx; m2_conv_map = conv_seq_map; m2_state_mask = state_mask;
        }
    }
    const int64_t n_seq_tokens = n_tok / n_seqs;

    ggml_tensor * state_all = kv_self.s_l[il];
    GGML_ASSERT(state_all != nullptr && state_all->type == GGML_TYPE_F32);

    if (pxa_mamba2_fuse_ok(layer, state_all, d_conv, d_state)) {
        // in-place path: conv window and scan state are read (masked) and written back on the rows directly
        ggml_tensor * zxBCdt = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_in, cur);
        cb(zxBCdt, "mamba2_in_proj", il);
        const size_t ei   = ggml_element_size(zxBCdt);
        const size_t nbti = zxBCdt->nb[1];                 // token stride of the in-projection
        const size_t nbsi = nbti * n_seq_tokens;           // sequence stride

        ggml_tensor * xBC = ggml_view_2d(ctx0, zxBCdt, conv_dim, n_tok, nbti, d_inner * ei);
        ggml_tensor * xBC_c = ggml_mamba2_conv(ctx0, state_all, xBC, layer.ssm_conv1d, layer.ssm_conv1d_b, state_row_idx, state_mask);
        cb(xBC_c, "mamba2_conv_out", il);

        const size_t es  = ggml_element_size(xBC_c);
        const size_t nbt = conv_dim * es;
        const size_t nbs = nbt * n_seq_tokens;
        ggml_tensor * x = ggml_view_4d(ctx0, xBC_c, head_dim, n_head,  n_seq_tokens, n_seqs, head_dim * es, nbt, nbs, 0);
        ggml_tensor * B = ggml_view_4d(ctx0, xBC_c, d_state,  n_group, n_seq_tokens, n_seqs, d_state  * es, nbt, nbs, d_inner * es);
        ggml_tensor * C = ggml_view_4d(ctx0, xBC_c, d_state,  n_group, n_seq_tokens, n_seqs, d_state  * es, nbt, nbs, (d_inner + n_group*d_state) * es);
        ggml_tensor * z = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * ei, nbti, nbsi, 0);
        ggml_tensor * dt = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, nbti, nbsi, (2*d_inner + 2*n_group*d_state) * ei);

        ggml_tensor * y = ggml_mamba2_scan(ctx0, state_all, x, dt, layer.ssm_a, B, C, layer.ssm_dt_b, layer.ssm_d, z,
                                           state_row_idx, state_mask, conv_state_dim);
        cb(y, "mamba2_scan_gated", il);

        if (layer.ssm_norm) {
            y = ggml_reshape_3d(ctx0, y, d_inner / n_group, n_group, n_tok);
            y = llm_build_context::llm_build_norm(ctx0, y, hparams, layer.ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
        }
        y = ggml_reshape_2d(ctx0, y, d_inner, n_tok);

        cur = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_out, y);
        cb(cur, "mamba_out", il);
        return cur;
    }

    // gather this batch's state rows and zero the ones whose sequence starts here
    ggml_tensor * state = ggml_get_rows(ctx0, state_all, state_row_idx);   // [state_dim, n_seqs]
    state = ggml_mul(ctx0, state, state_mask);
    cb(state, "mamba2_state_in", il);

    ggml_tensor * conv_state_flat = ggml_view_2d(ctx0, state, conv_state_dim, n_seqs, state->nb[1], 0);
    ggml_tensor * ssm_state_flat  = ggml_view_2d(ctx0, state, ssm_state_dim,  n_seqs, state->nb[1],
                                                 conv_state_dim * ggml_element_size(state));
    if (n_seqs > 1) {
        conv_state_flat = ggml_cont(ctx0, conv_state_flat);
        ssm_state_flat  = ggml_cont(ctx0, ssm_state_flat);
    }
    ggml_tensor * conv_states = ggml_reshape_3d(ctx0, conv_state_flat, d_conv - 1, conv_dim, n_seqs);

    // {n_embd, d_in_proj} @ {n_embd, n_tok} => {d_in_proj, n_tok}, d_in_proj = 2*d_inner + 2*n_group*d_state + n_head
    ggml_tensor * zxBCdt = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_in, cur);
    cb(zxBCdt, "mamba2_in_proj", il);

    ggml_tensor * z   = ggml_view_2d(ctx0, zxBCdt, d_inner,  n_tok, zxBCdt->nb[1], 0);
    ggml_tensor * xBC = ggml_view_2d(ctx0, zxBCdt, conv_dim, n_tok, zxBCdt->nb[1], d_inner * ggml_element_size(zxBCdt));
    ggml_tensor * dt  = ggml_view_2d(ctx0, zxBCdt, n_head,   n_tok, zxBCdt->nb[1], (2*d_inner + 2*n_group*d_state) * ggml_element_size(zxBCdt));

    // conv over the (state window | new tokens) sequence; the op also emits the new window
    ggml_tensor * conv_w = layer.ssm_conv1d;
    if (conv_w->type != GGML_TYPE_F32) {
        conv_w = ggml_cast(ctx0, conv_w, GGML_TYPE_F32);
    }
    ggml_tensor * conv_raw = ggml_ssm_conv(ctx0, conv_states, ggml_cont(ctx0, xBC), conv_w, conv_seq_map, nullptr);
    cb(conv_raw, "mamba2_conv_raw", il);

    ggml_tensor * xBC_c = ggml_view_2d(ctx0, conv_raw, conv_dim, n_tok, conv_dim * ggml_element_size(conv_raw), 0);
    if (layer.ssm_conv1d_b) {
        xBC_c = ggml_add(ctx0, xBC_c, layer.ssm_conv1d_b);
    }
    xBC_c = ggml_silu(ctx0, xBC_c);
    cb(xBC_c, "mamba2_conv_out", il);

    ggml_tensor * new_conv = ggml_view_3d(ctx0, conv_raw, d_conv - 1, conv_dim, n_seqs,
            d_conv * ggml_element_size(conv_raw),
            d_conv * conv_dim * ggml_element_size(conv_raw),
            (conv_dim * n_tok + 1) * ggml_element_size(conv_raw));
    ggml_tensor * new_conv_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, new_conv), conv_state_dim, n_seqs);

    // scan: x / B / C are views into the conv output (tokens of a sequence are contiguous)
    const size_t es  = ggml_element_size(xBC_c);
    const size_t nbt = conv_dim * es;               // token stride
    const size_t nbs = nbt * n_seq_tokens;          // sequence stride
    ggml_tensor * x = ggml_view_4d(ctx0, xBC_c, head_dim, n_head,  n_seq_tokens, n_seqs, head_dim * es, nbt, nbs, 0);
    ggml_tensor * B = ggml_view_4d(ctx0, xBC_c, d_state,  n_group, n_seq_tokens, n_seqs, d_state  * es, nbt, nbs, d_inner * es);
    ggml_tensor * C = ggml_view_4d(ctx0, xBC_c, d_state,  n_group, n_seq_tokens, n_seqs, d_state  * es, nbt, nbs, (d_inner + n_group*d_state) * es);

    dt = ggml_add(ctx0, ggml_cont(ctx0, dt), layer.ssm_dt_b);
    dt = ggml_reshape_3d(ctx0, dt, n_head, n_seq_tokens, n_seqs);

    ggml_tensor * s4 = ggml_reshape_4d(ctx0, ssm_state_flat, d_state, head_dim, n_head, n_seqs);
    ggml_tensor * y_s = ggml_ssm_scan2(ctx0, s4, x, dt, layer.ssm_a, B, C);
    cb(y_s, "mamba2_scan", il);

    const int64_t n_y = head_dim * n_head * n_tok;
    ggml_tensor * new_ssm_flat = ggml_view_2d(ctx0, y_s, ssm_state_dim, n_seqs, ssm_state_dim * ggml_element_size(y_s),
                                              n_y * ggml_element_size(y_s));

    // write the new rows back (conv window | scan state), one scatter per layer
    ggml_tensor * new_row = ggml_concat(ctx0, new_conv_flat, new_ssm_flat, 0);   // [state_dim, n_seqs]
    ggml_tensor * state_cpy = ggml_set_rows(ctx0, state_all, new_row, state_row_idx);
    cb(state_cpy, "mamba2_state_cpy", il);
    ggml_build_forward_expand(gf, state_cpy);

    // y {head_dim, n_head, n_tok}; + D*x; gate with silu(z); grouped RMS norm; out proj
    ggml_tensor * y = ggml_view_3d(ctx0, y_s, head_dim, n_head, n_tok,
                                   head_dim * ggml_element_size(y_s), head_dim * n_head * ggml_element_size(y_s), 0);
    ggml_tensor * x3 = ggml_view_3d(ctx0, xBC_c, head_dim, n_head, n_tok, head_dim * es, nbt, 0);
    y = ggml_add(ctx0, y, ggml_mul(ctx0, x3, layer.ssm_d));
    cb(y, "mamba2_y_add_d", il);

    ggml_tensor * z3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, z), head_dim, n_head, n_tok);
    y = ggml_mul(ctx0, y, ggml_silu(ctx0, z3));

    if (layer.ssm_norm) {
        y = ggml_reshape_3d(ctx0, y, d_inner / n_group, n_group, n_tok);
        y = llm_build_context::llm_build_norm(ctx0, y, hparams, layer.ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    }
    y = ggml_reshape_2d(ctx0, y, d_inner, n_tok);

    cur = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_out, y);
    cb(cur, "mamba_out", il);
    return cur;
}
