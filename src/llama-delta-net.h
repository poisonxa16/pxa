#pragma once

#include "llama-build-context.h"
#include "pxa-seq-decomp.h" // PXA_LLAMA_MTP_FIX: shared distinct-seq decomposition

#include <utility>

// PXA_DN_SHARE_INPUTS: -1 = auto (on iff n_seq_max == 1), 0 = forced off, 1 = forced on.
// Defined in llama-delta-net.cpp; llama.cpp uses these to log the decision at context init.
int  pxa_dn_share_inputs_mode();
bool pxa_dn_share_inputs_for(uint32_t n_seq_max);

struct delta_net {
    delta_net(llama_context & lctx, const llama_batch & batch);
    ~delta_net();

    // Used for speculative decoding to enable per-step state checkpoint restoration.
    bool save_per_step_states = false;

    static std::pair<ggml_tensor *, ggml_tensor *> build_fused_delta_net(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      int il, const llm_build_cb & cb, int repeat_type,
                      ggml_tensor * per_step_ckpt = nullptr);

    // PXA_LLAMA_FIX_v4: takes n_seqs + per-request runtime tensors (state_row_idx, conv_seq_map, state_mask) so the
    // mixed (concurrent) path runs ONE batched delta-net (n_seqs=n_tok) instead of a per-token subgraph loop.
    ggml_tensor * build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * state_row_idx, ggml_tensor * conv_seq_map, ggml_tensor * state_mask, ggml_tensor * inp_out_ids,
            int64_t n_seqs, bool reset_state_local, int il, const llm_build_cb & cb, int64_t pxa_static_slot = -1,
            bool hc_mode = false) const;

    ggml_tensor * build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb,
            // hc_mode: the caller (qwen4exp) has already normed the input with its hyper-connection
            // mixer and will scatter the block output back into the wide residual itself, so the
            // layer norm and the residual add are both skipped, and the output gate is sigmoid.
            bool hc_mode = false) const;

private:

    llama_context     & lctx;
    const llama_batch & batch;
    std::vector<llama_seq_id> token_seq_ids;
    bool all_same_seq;
    bool has_unique_seq_ids;
    // PXA_LLAMA_MTP_FIX: distinct-sequence decomposition (n_seqs, seq_slot[], n_seq_tokens, ...)
    // computed once in the constructor and shared with the builder. Generalizes the v4
    // "n_seqs == n_tok" assumption so an MTP verify batch (n_seq_tokens>1 per seq) is handled.
    pxa_seq_decomp seq_decomp;

    // PXA_DN_SHARE_INPUTS: the state_row_idx / conv_seq_map / state_mask views handed to each
    // delta-net layer are built from graph-level constants only (n_tok, n_seqs), so every layer
    // was building a byte-identical view of the same host tensor. The scheduler keys its split
    // inputs on the tensor pointer, so 48 identical views became 48 separate split inputs, each
    // with its own host->device copy of the WHOLE viewed span (conv_seq_map spans n_tok*n_batch
    // int32 = 16 MiB at n_batch 2048), and 32 of them filled a split (GGML_SCHED_MAX_SPLIT_INPUTS)
    // and forced an extra split. One delta_net instance is constructed per graph build, so
    // caching the three views here is exactly per-graph. Keyed on ctx0 as a cheap guard.
    mutable ggml_context * shared_view_ctx    = nullptr;
    mutable ggml_tensor  * shared_state_row   = nullptr;
    mutable ggml_tensor  * shared_conv_map    = nullptr;
    mutable ggml_tensor  * shared_state_mask  = nullptr;
    mutable int64_t        shared_view_n_seqs = -1;
    mutable int64_t        shared_view_n_tok  = -1;

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * input, int il, const llm_build_cb & cb,
            ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input, int il,
            const llm_build_cb & cb);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input,
            int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_beta_gate(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * ssm_beta_alpha, ggml_tensor * ssm_beta, ggml_tensor * ssm_alpha,
            ggml_tensor * ssm_dt, ggml_tensor * ssm_a, int64_t num_k_heads, int64_t num_v_heads, int64_t n_seqs,
            ggml_tensor * cur, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static ggml_tensor * build_qkv(ggml_context * ctx0, ggml_tensor * state_storage, ggml_tensor * ssm_conv1d,
            ggml_tensor * qkv_mixed, ggml_tensor * state_row_idx, ggml_tensor * conv_seq_map, ggml_tensor * state_mask, ggml_tensor * beta, ggml_tensor * gate,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, int64_t ssm_d_conv,
            int64_t n_seqs_in, uint32_t qnext_state_slots, bool reset_state_local,
            float eps_norm, int repeat_type, int il, const llm_build_cb & cb, ggml_cgraph * gf,
            ggml_tensor * per_step_ssm = nullptr, ggml_tensor * per_step_conv = nullptr, int64_t pxa_static_slot = -1);

    static ggml_tensor * build_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out,
            ggml_tensor * output, ggml_tensor * z, int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb, bool sigmoid_gate = false);
};
