//
// DeepSeek-V4 (deepseek_v4 / DeepseekV4ForCausalLM) graph.
//
// Adapted from llama.cpp `src/models/deepseek4.cpp` @ upstream commit 82dbc4f01
// (originating PR #24162 and follow-ups #25202, #25370, #25521, #25588, #25702,
// #25787, #25325, #25945, #25871).
// Copyright (c) 2023-2026 The ggml authors. MIT license.
// SPDX-License-Identifier: MIT
//
// Upstream carries no per-file copyright header on src/models/deepseek4.cpp; the
// project relies on its root LICENSE. This block is the explicit provenance note
// required for a file taken substantially from that source. Follows the local
// precedent in src/graphs/build_hy3.cpp.
//
// ---------------------------------------------------------------------------
// STRUCTURAL DIFFERENCES vs upstream (deliberate; each one is load-bearing)
// ---------------------------------------------------------------------------
//  1. Upstream is a `llm_graph_context` subclass with an input-object system
//     (llm_graph_input_dsv4 / _dsv4_raw). This tree has neither. The graph is a
//     method on llm_build_context; the DS4 input tensors are created here and
//     their pointers stashed in `llama_dsv4_get_inputs(lctx)` for the host-side
//     filler `llama_dsv4_set_inputs()` (owned by the memory chunk).
//  2. All four fused ops are OFF (LIGHTNING_INDEXER, DSV4_HC_{PRE,COMB,POST}).
//     Only upstream's primitive fallback branches are ported. There is no
//     `res->add_fused_node()` equivalent here, so those calls are dropped.
//  3. `ggml_rope_ext_back` -> `ggml_rope_back` (upstream renamed the symbol; the
//     parameter list is byte-identical). Used in the FORWARD graph to de-rope the
//     attention output before the o-LoRA.
//  4. `llm_graph_context::build_attn_mha` has no equivalent here (our
//     llm_build_kqv is welded to the POD llama_kv_cache), so it is ported as the
//     private member build_dsv4_attn_mha() below, restricted to the two arguments
//     DS4 actually uses (kq_b == nullptr, v_mla == nullptr).
//  5. ALL DS4 mask inputs are GGML_TYPE_F32 and UNPADDED [n_kv, n_tokens, 1, 1].
//     Upstream builds the CSA/HCA masks as F16 under -fa. We cannot: this tree's
//     `ggml_fill` asserts F32 (ggml.c) and `ggml_set_rows` asserts a F32 source,
//     and build_top_k_mask needs both. The graph pads to GGML_KQ_MASK_PAD and
//     casts to F16 itself, immediately before handing a mask to flash-attention.
//     -> The memory chunk must write FLOATS into all four mask tensors.
//  6. `dsv4_with_zero_dep` uses ggml_sum_rows over one row instead of ggml_sum:
//     GGML_OP_SUM has no CUDA backend in this tree and would split the graph to
//     the CPU on every compressed layer. The node exists purely to order the
//     persistent-state overwrite after the state-backed commit has read it; the
//     substitute carries the same dependency edge.
//  7. Single stream only (`-np 1`). This tree's ggml_flash_attn_ext asserts
//     mask->ne[2] == 1 && mask->ne[3] == 1, so a multi-stream mask cannot even be
//     expressed here. Asserted explicitly below rather than silently mis-shaped.
//
// ---------------------------------------------------------------------------
// INTERFACE CONTRACT — declarations this file requires from
// `src/llama-kv-cache-dsv4.h` (owned by the memory chunk).
// Reproduced verbatim so the owner can paste it; if any name changes, this file
// must change with it.
// ---------------------------------------------------------------------------
//
//   enum llama_dsv4_stream {
//       LLAMA_DSV4_CSA = 0,   // compress ratio 4  (+ lightning indexer)
//       LLAMA_DSV4_HCA = 1,   // compress ratio 128
//       LLAMA_DSV4_LID = 2,   // lightning-indexer key stream (ratio 4)
//   };
//
//   // verbatim from upstream llama_kv_cache_dsv4_context::comp_plan
//   struct llama_dsv4_comp_plan {
//       std::vector<int32_t> state_pos;
//       std::vector<int32_t> state_persist_src_idxs;
//       std::vector<int32_t> state_persist_dst_idxs;
//       std::vector<int32_t> state_read_idxs;
//       std::vector<int64_t> state_write_idxs;
//       std::vector<int32_t> state_write_pos;
//       std::vector<int32_t> n_visible;
//       int64_t n_stream = 1;
//       int64_t n_kv     = 0;
//   };
//
//   struct llama_dsv4_comp_inputs {
//       ggml_tensor * state_pos              = nullptr; // I32 [n_state]
//       ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_persist_src]
//       ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_persist_dst]
//       ggml_tensor * state_read_idxs        = nullptr; // I32 [ratio*n_write] (overlap: 2*ratio*n_write)
//       ggml_tensor * state_write_idxs       = nullptr; // I64 [n_write]
//       ggml_tensor * state_write_pos        = nullptr; // I32 [n_write]
//       ggml_tensor * kq_mask                = nullptr; // F32 [n_kv, n_tokens, 1, 1]
//       ggml_tensor * k_rot                  = nullptr; // F32 [nrot, nrot] or null
//   };
//
//   struct llama_dsv4_inputs {
//       ggml_tensor * raw_k_idxs  = nullptr;            // I64 [n_tokens]
//       ggml_tensor * raw_kq_mask = nullptr;            // F32 [n_kv_raw, n_tokens, 1, 1]
//       ggml_tensor * raw_k_rot   = nullptr;            // F32 [nrot, nrot] or null
//       llama_dsv4_comp_inputs comp[3];                 // indexed by llama_dsv4_stream
//   };
//
//   llama_dsv4_inputs & llama_dsv4_get_inputs(llama_context & lctx);
//   const llama_dsv4_comp_plan & llama_dsv4_get_plan(const llama_context & lctx, llama_dsv4_stream s);
//   uint32_t llama_dsv4_get_raw_n_kv(const llama_context & lctx);
//   int64_t  llama_dsv4_get_raw_nrot(const llama_context & lctx);  // 0 => no raw k_rot
//   int64_t  llama_dsv4_get_comp_nrot(const llama_context & lctx, llama_dsv4_stream s);
//
//   ggml_tensor * llama_dsv4_get_raw_k(const llama_context & lctx, ggml_context * ctx, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_raw_k(const llama_context & lctx, ggml_context * ctx,
//                                      ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il);
//   ggml_tensor * llama_dsv4_get_comp_k(const llama_context & lctx, ggml_context * ctx,
//                                       llama_dsv4_stream s, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_comp_k(const llama_context & lctx, ggml_context * ctx,
//                                       llama_dsv4_stream s, ggml_tensor * k_cur,
//                                       ggml_tensor * k_idxs, int32_t il);
//   ggml_tensor * llama_dsv4_get_state_kv(const llama_context & lctx, ggml_context * ctx,
//                                         llama_dsv4_stream s, int32_t il);
//   ggml_tensor * llama_dsv4_get_state_score(const llama_context & lctx, ggml_context * ctx,
//                                            llama_dsv4_stream s, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_state_kv(const llama_context & lctx, ggml_context * ctx,
//                                         llama_dsv4_stream s, ggml_tensor * cur,
//                                         ggml_tensor * idxs, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_state_score(const llama_context & lctx, ggml_context * ctx,
//                                            llama_dsv4_stream s, ggml_tensor * cur,
//                                            ggml_tensor * idxs, int32_t il);
//
// ---------------------------------------------------------------------------
// hparams / tensor fields required from the arch chunk (spec §1.2, §1.6):
//   hparams: dsv4_hc_mult, dsv4_hc_sinkhorn_iters, dsv4_hc_eps, dsv4_hash_layer_count,
//            dsv4_o_group_count, dsv4_o_lora_rank, dsv4_compress_rope_base,
//            dsv4_compress_ratios[LLAMA_MAX_LAYERS],
//            swiglu_limits[LLAMA_MAX_LAYERS], swiglu_limits_shared[LLAMA_MAX_LAYERS]
//   model:   hc_head_fn, hc_head_base, hc_head_scale
//   layer:   wq_b, wo_a, wo_b, hc_attn_{fn,base,scale}, hc_ffn_{fn,base,scale},
//            attn_comp_{wkv,wgate,ape,norm}, indexer_comp_{wkv,wgate,ape,norm},
//            ffn_gate_tid2eid
//            (attn_norm, attn_sinks, wq_a, attn_q_a_norm, wkv, attn_kv_a_norm,
//             indexer_proj, indexer_attn_q_b, ffn_norm, ffn_gate_inp,
//             ffn_exp_probs_b, ffn_*_exps, ffn_*_shexp already exist here)
//   NOTE: this tree calls the compressed-KV norm `attn_kv_a_norm`, upstream calls
//   it `attn_kv_norm`. We use the local name -- in BOTH the graph and the loader.
//

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-kv-cache-dsv4.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

// compression ratios that select the three per-layer attention regimes
static constexpr int64_t DSV4_CSA_RATIO = 4;    // compressed sparse attn + lightning indexer
static constexpr int64_t DSV4_HCA_RATIO = 128;  // hyper-compressed attn

// upstream: dsv4_rope_attn_factor()
static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

// PXA_DS4_ROPE_FLIPPED (default on; set to 0 to restore the pre-port graph).
//
// DS4 stores every roped row as [nope | pe]. The classic shape is
//   nope = view(x, 0)            pe = view(x, off)
//   pe   = rope(pe)              x  = concat(nope, pe, 0)
// The concat reads a NON-contiguous source and writes a full-width destination.
// At n_embd_head=512 / n_head=64 / nt=383 the q and attn-derope concats are
// ~50 MB per layer each -- that is the cost this removes.
//
// Flipped RoPE collapses the triple to one node: rotate the TRAILING n_dims
// channels where they already sit and pass the nope half through in the same
// kernel. Two views and the non-contiguous concat disappear. The rotation
// itself is untouched -- only the memory index moves.
//
// ⚠ NOT in-place, deliberately, and this is a DEVIATION from upstream.
//
// Upstream uses ggml_rope_ext_inplace here, which also drops the destination
// allocation. That is unsafe on this hardware, and it was measured, not
// reasoned about: with in-place ropes, `-fa on` + a ~2600-token prompt aborts
// inside cublasSgemm with "the function failed to launch" (a sticky fault
// raised by an earlier kernel), while the same binary with the switch off, and
// the same binary at `-fa off`, both complete.
//
// Cause: on sm_70 head dim 512 has no CUDA flash-attention kernel, so most DS4
// attention nodes end up on the CPU backend and their outputs live in host
// memory. Measured with GGML_SCHED_DEBUG=2 at this fill: FLASH_ATTN CUDA=43
// CPU=86, while ROPE_BACK is CUDA=129 CPU=0 -- every de-rope assigned to CUDA0,
// 86 of them viewing a host-resident tensor. An in-place rope is a VIEW, and
// ggml_backend_cuda_supports_op returns true unconditionally for
// ROPE/ROPE_BACK, so the scheduler is free to assign the view to CUDA. A view
// cannot be satisfied by inserting a copy -- it must write into its view_src's
// actual buffer -- so the kernel writes to a host pointer. Upstream does not
// hit this because their DS4 attention runs on the GPU (Ampere+).
//
// The concat -- the part that actually showed up in the profile -- is removed
// either way. What in-place would additionally save is one destination
// allocation and the nope-half copy, and the classic path was copying that
// same data through the concat anyway.
static bool dsv4_rope_flipped_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXA_DS4_ROPE_FLIPPED");
        return !(e && e[0] == '0');
    }();
    return v;
}

// Rotate the trailing n_dims channels of `cur`, passing the leading channels
// through. Fresh destination: no aliasing, so this is safe under any backend
// split.
static ggml_tensor * dsv4_rope_flipped(
        ggml_context * ctx, ggml_tensor * cur, ggml_tensor * pos, int n_dims, int rope_type,
        int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(n_dims <= cur->ne[0]);

    ggml_tensor * out = ggml_rope_ext(ctx, cur, pos, nullptr, n_dims, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    out->op_params[15] = 1; // PXA_ROPE_FLIPPED_v1

    // Structural guard, not decoration: an in-place rope here is a VIEW, and a
    // view must write into its view_src's buffer, which the scheduler is free to
    // put on another backend. That is exactly the sm_70 `-fa on` abort
    // (ROPE_BACK -> illegal memory access). If anyone switches these back to the
    // _inplace constructors, fail at graph build rather than at a random kernel.
    GGML_ASSERT(out->view_src == nullptr &&
            "DS4 flipped rope must own its destination");

    return out;
}

// Inverse rotation, same layout. Uses ggml_rope_back directly, so there is no
// need for upstream's trick of retyping a ROPE node to ROPE_BACK.
static ggml_tensor * dsv4_rope_back_flipped(
        ggml_context * ctx, ggml_tensor * cur, ggml_tensor * pos, int n_dims, int rope_type,
        int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(n_dims <= cur->ne[0]);

    ggml_tensor * out = ggml_rope_back(ctx, cur, pos, nullptr, n_dims, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    out->op_params[15] = 1; // PXA_ROPE_FLIPPED_v1

    // Structural guard, not decoration: an in-place rope here is a VIEW, and a
    // view must write into its view_src's buffer, which the scheduler is free to
    // put on another backend. That is exactly the sm_70 `-fa on` abort
    // (ROPE_BACK -> illegal memory access). If anyone switches these back to the
    // _inplace constructors, fail at graph build rather than at a random kernel.
    GGML_ASSERT(out->view_src == nullptr &&
            "DS4 flipped rope must own its destination");

    return out;
}

// upstream: dsv4_append_zero_row(). Adds one synthetic trailing row that
// state_read_idxs may address for the first block's "previous" half.
static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

// upstream: dsv4_with_zero_dep(), but built on sum_rows (see note 6 at the top).
// Adds a numerically-zero edge t <- dep so the persistent-state overwrite cannot
// be scheduled before the state-backed commit that reads the old state.
static ggml_tensor * dsv4_with_zero_dep(ggml_context * ctx, ggml_tensor * t, ggml_tensor * dep) {
    if (dep == nullptr) {
        return t;
    }

    ggml_tensor * row  = ggml_reshape_2d(ctx, ggml_view_1d(ctx, dep, dep->ne[0], 0), dep->ne[0], 1);
    ggml_tensor * zero = ggml_scale(ctx, ggml_sum_rows(ctx, row), 0.0f); // [1,1]

    return ggml_add(ctx, t, zero);
}

static ggml_tensor * dsv4_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

// upstream: llama_mul_mat_hadamard() from src/llama-impl.h. `rot` is an explicit
// [n,n] rotation matrix supplied by the cache; this tree's ggml_hadamard() is a
// different (matrix-free) op and is NOT a substitute.
static ggml_tensor * dsv4_mul_mat_rot(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];

    ggml_tensor * res;
    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

// Pad a DS4 mask along dim 1 to GGML_KQ_MASK_PAD (flash-attention requirement in
// this tree) and cast it to the type the attention kernel wants. All DS4 masks
// arrive here as F32 [n_kv, n_tokens, 1, 1] - see note 5 at the top.
static ggml_tensor * dsv4_finalize_kq_mask(ggml_context * ctx, ggml_tensor * m, bool flash_attn) {
    GGML_ASSERT(m->ne[2] == 1 && m->ne[3] == 1 && "DSV4 first cut is single-stream (-np 1)");

    const int64_t n_pad = GGML_PAD(m->ne[1], GGML_KQ_MASK_PAD);
    if (n_pad > m->ne[1]) {
        ggml_tensor * pad = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m->ne[0], n_pad - m->ne[1], 1, 1);
        pad = ggml_fill(ctx, pad, -INFINITY);
        m   = ggml_concat(ctx, m, pad, 1);
    }

    if (flash_attn && m->type != GGML_TYPE_F16) {
        m = ggml_cast(ctx, m, GGML_TYPE_F16);
    }

    return m;
}

static ggml_tensor * dsv4_build_input_1d(ggml_context * ctx, ggml_type type, int64_t n, const char * name) {
    if (n <= 0) {
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, type, n);
    ggml_set_input(t);
    ggml_set_name(t, name);

    return t;
}

static ggml_tensor * dsv4_build_input_mask(ggml_context * ctx, int64_t n_kv, int64_t n_tokens, const char * name) {
    if (n_kv <= 0) {
        return nullptr;
    }

    // always F32, always unpadded - see note 5 at the top of this file
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens, 1, 1);
    ggml_set_input(t);
    ggml_set_name(t, name);

    return t;
}

static ggml_tensor * dsv4_build_input_rot(ggml_context * ctx, int64_t nrot, const char * name) {
    if (nrot <= 0) {
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
    ggml_set_input(t);
    ggml_set_name(t, name);

    return t;
}

//
// input construction (upstream: llm_graph_context::build_inp_dsv4 +
// dsv4_build_comp_inputs + dsv4_build_raw_kq_mask in llama-graph.cpp)
//

void llm_build_context::build_dsv4_inputs() {
    // Derive this batch's compression plans FIRST: every plan-sized tensor below must be
    // built from the same plan llama_dsv4_set_inputs() will fill it from.
    llama_dsv4_build_plans(lctx, batch);

    llama_dsv4_inputs & inp = llama_dsv4_get_inputs(lctx);

    inp = llama_dsv4_inputs();

    const bool    raw_win  = llama_dsv4_raw_window_enabled(lctx);
    const int64_t n_kv_raw = llama_dsv4_raw_window_mode(lctx) == 1
                                     ? (int64_t) llama_dsv4_raw_window_width(lctx, n_tokens)
                                     : (int64_t) llama_dsv4_get_raw_n_kv(lctx);

    inp.raw_k_idxs  = dsv4_build_input_1d(ctx0, GGML_TYPE_I64, n_tokens, "dsv4_raw_k_idxs");
    inp.raw_kq_mask = dsv4_build_input_mask(ctx0, n_kv_raw, n_tokens, "dsv4_raw_kq_mask");
    if (raw_win) {
        inp.raw_win_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, n_kv_raw, "dsv4_raw_win_idxs");
    }
    inp.raw_k_rot   = dsv4_build_input_rot(ctx0, llama_dsv4_get_raw_nrot(lctx), "dsv4_raw_k_rot");

    static const char * const s_name[3] = { "csa", "hca", "lid" };

    for (int s = 0; s < 3; ++s) {
        const llama_dsv4_stream    st   = (llama_dsv4_stream) s;
        const llama_dsv4_comp_plan & pl = llama_dsv4_get_plan(lctx, st);
        llama_dsv4_comp_inputs     & ci = inp.comp[s];

        GGML_ASSERT(pl.n_stream == 1 && "DSV4 first cut is single-stream (-np 1)");

        char name[64];

        snprintf(name, sizeof(name), "dsv4_%s_state_pos", s_name[s]);
        ci.state_pos = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_pos.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_persist_src_idxs", s_name[s]);
        ci.state_persist_src_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_persist_src_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_persist_dst_idxs", s_name[s]);
        ci.state_persist_dst_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_persist_dst_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_read_idxs", s_name[s]);
        ci.state_read_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_read_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_write_idxs", s_name[s]);
        ci.state_write_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I64, (int64_t) pl.state_write_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_write_pos", s_name[s]);
        ci.state_write_pos = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_write_pos.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_kq_mask", s_name[s]);
        ci.kq_mask = dsv4_build_input_mask(ctx0, pl.n_kv, n_tokens, name);

        snprintf(name, sizeof(name), "dsv4_%s_k_rot", s_name[s]);
        ci.k_rot = dsv4_build_input_rot(ctx0, llama_dsv4_get_comp_nrot(lctx, st), name);
    }
}

//
// hyper-connections
//

// The primitive-op decompositions that used to live here (build_dsv4_hc_mix,
// build_dsv4_hc_sinkhorn, and the per-branch loops in hc_pre/hc_post) are
// replaced by the fused ops GGML_OP_DSV4_HC_{SPLIT_SINKHORN,WEIGHTED_SUM,EXPAND},
// mirroring upstream @ 44c7b01de. The decomposition was ~19,000 of our 32,077
// graph nodes per forward (43 layers x 2 hc passes x 20 Sinkhorn iterations of
// permute/sum_rows/add/div on a 4x4) and was pure launch overhead at bs=1 --
// prefill was at parity while decode ran 2.3-2.6x behind upstream.

// upstream: build_hc_pre(x, hc_fn, hc_scale, hc_base, &post, &comb, il).
// Produces the layer input, plus the post weights and the Sinkhorn-normalised

// ---------------------------------------------------------------------------
// DSpark target-side capture (M2).
//
// DSpark consumes the mean of the target's FOUR hyper-connection streams at the end of
// each of its three capture blocks (ds4.c:15977-15985 sets the weights to 1/4 and
// ds4.c:25874-25897 takes the sum; the call site ds4.c:26214 fires AFTER the FFN
// hc_post, i.e. at the very end of the block). Reproduced here with the SAME fused op
// the DS4 body already uses, so no new ggml op appears.
//
// The 1/hc weight tensor is materialised in-graph from a [hc, nt]-shaped view of the
// stream itself (x*0 + 1/hc). That avoids adding a hand-filled graph input purely to
// carry a constant - the drafter will need one for the non-causal block mask, but the
// capture does not.
//
// LAZY BY CONSTRUCTION: the capture is taken from the SAME out_ids gather the output
// head uses, so a prompt chunk captures n_outputs rows (=1), not n_tokens. That is what
// keeps M2 off the `n_outputs_embd` landmine (llama.cpp:5491) that cost the 35B 40% of
// its cold prefill - we never widen n_outputs, so there is nothing to widen.
//
// Off unless PXA_DSPARK_CAPTURE=1. Default path is byte-unchanged.
// ---------------------------------------------------------------------------
static bool dspark_capture_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXA_DSPARK_CAPTURE");
        return e && *e && *e != '0';
    }();
    return v;
}

// Which target layers to tap. Default = the last three, matching
// dspark.target_layer_ids = [40,41,42] on the 0731 checkpoint. M4 takes these from the
// bound drafter instead of the environment.
static bool dspark_is_capture_layer(int il, int n_layer, int * slot_out) {
    static const char * spec = getenv("PXA_DSPARK_CAPTURE_LAYERS");
    if (spec && *spec) {
        int slot = 0;
        const char * s = spec;
        while (*s) {
            char * end = nullptr;
            long v = strtol(s, &end, 10);
            if (end == s) break;
            if ((int) v == il) { if (slot_out) *slot_out = slot; return true; }
            ++slot;
            s = (*end == ',') ? end + 1 : end;
        }
        return false;
    }
    const int first = n_layer - 3;
    if (il >= first && il < n_layer) {
        if (slot_out) *slot_out = il - first;
        return true;
    }
    return false;
}

// hc x hc combination matrix consumed by build_dsv4_hc_post().
ggml_tensor * llm_build_context::build_dsv4_hc_pre(
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int            il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    // One fused node computes the pre/post affine+sigmoid AND the 20-iteration
    // Sinkhorn on the comb block (upstream: ggml_dsv4_hc_split_sinkhorn).
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx0, mixes, hc_scale, hc_base,
            (int) hc, (int) hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);
    cb(split, "hc_split", il);

    ggml_tensor * pre = ggml_view_2d(ctx0, split, hc, nt, split->nb[1], 0);
    *post = ggml_view_2d(ctx0, split, hc, nt, split->nb[1], hc*split->nb[0]);
    ggml_tensor * comb_flat = ggml_view_2d(ctx0, split, hc*hc, nt, split->nb[1], 2*hc*split->nb[0]);
    if (nt != 1) {
        pre       = ggml_cont(ctx0, pre);
        *post     = ggml_cont(ctx0, *post);
        comb_flat = ggml_cont(ctx0, comb_flat);
    }
    cb(pre, "hc_pre", il);
    cb(*post, "hc_post", il);

    *comb = ggml_reshape_3d(ctx0, comb_flat, hc, hc, nt);
    cb(*comb, "hc_comb", il);

    return ggml_dsv4_hc_weighted_sum(ctx0, x, pre);
}

// upstream: build_hc_post(). Scatters the sublayer output back across the hc
// branches and mixes in the (Sinkhorn-weighted) incoming residual.
ggml_tensor * llm_build_context::build_dsv4_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == (int64_t) hparams.dsv4_hc_mult);

    GGML_UNUSED(il);

    return ggml_dsv4_hc_expand(ctx0, x, residual, post, comb);
}

// upstream: build_hc_head(). Final collapse of the hc-wide stream to n_embd.
ggml_tensor * llm_build_context::build_dsv4_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return ggml_dsv4_hc_weighted_sum(ctx0, x, pre);
}

//
// attention core (upstream: llm_graph_context::build_attn_mha, restricted to
// kq_b == nullptr and v_mla == nullptr, which is all DS4 uses)
//

ggml_tensor * llm_build_context::build_dsv4_attn_mha(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il,
        int64_t       n_visible_max) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // Captured before the permutes: ggml_permute() returns a fresh view struct each
    // call, so after permuting, k and v are never pointer-equal even when the caller
    // passed the same tensor twice (all three DS4 regimes do).
    const bool v_aliases_k = (v == k) && !v_trans;

    const int64_t n_stream = k->ne[3];
    GGML_ASSERT(n_stream == 1 && "DSV4 first cut is single-stream (-np 1)");

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
            q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    if (cparams.flash_attn) {
        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }
        if (v_aliases_k) {
            // Keep V the *same tensor* as K rather than casting it a second time.
            // Two ggml_cast() calls produce two distinct F16 buffers, which doubles the
            // F16 conversion and defeats the DSA kernel's v_is_k_view fast path (it
            // would then gather the value rows a second time).
            v = k;
        } else if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        // Sparse attention: when the mask is mostly -inf, hand the kernel the list of
        // rows each query can actually see instead of a full-width score matrix.
        // n_visible_max must bound the visible count for EVERY query row; the padded
        // list is only built when it is genuinely narrower than the dense mask.
        // PXA_DSA_ATTN=0 must reproduce the PRE-PORT graph exactly, not merely stop the
        // kernel from firing. Measured: with the switch honoured only in the kernel, the
        // off arm still built 43 mask_to_idx nodes per decode graph that nothing consumed
        // -- dead work charged to the baseline, which would bias any A/B in DSA's favour.
        // Mirrors the kernel-side switch in ggml-cuda/dsa_attn.cu, INCLUDING its default:
        // off unless PXA_DSA_ATTN=1. If these two disagree the graph and the kernel
        // disagree about which engine is running, which is worse than either default.
        static const bool dsa_enabled = [] {
            const char * v = getenv("PXA_DSA_ATTN");
            return v && v[0] == '1';
        }();

        ggml_tensor * selected = nullptr;
        if (dsa_enabled && n_visible_max > 0) {
            // The 256-round-up is required by the kernel's launch geometry, and it
            // doubles as slack: up to 255 rows more than the caller's bound still fit.
            const int64_t n_sel = GGML_PAD(n_visible_max, 256);
            const int64_t n_kv  = kq_mask->ne[0];
            const int64_t nt    = q->ne[1];   // post-permute: q is [n_embd_head, n_tokens, n_head]

            // Mirror of the "worth it?" block in ggml_cuda_dsa_attn_supported(). The
            // gather costs n_sel rows per query, so a barely-narrower list is a loss.
            // Applying the same rule here means we never spend a mask_to_idx node
            // building a list the kernel would then decline to use. The kernel's copy
            // is authoritative; disagreeing here only wastes a node, never miscomputes.
            const int64_t min_ratio = [] {
                const char * v = getenv("PXA_DSA_MIN_RATIO");
                const int r = v ? atoi(v) : 4;
                return (int64_t) (r > 0 ? r : 4);
            }();
            const bool worth_it = nt <= 16 ? n_sel < n_kv : n_kv >= min_ratio*n_sel;

            if (worth_it) {
                selected = ggml_mask_to_index(ctx0, kq_mask, n_sel);
                cb(selected, "dsa_mask_to_idx", il);
                ggml_build_forward_expand(gf, selected);
            }
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        if (selected) {
            // Backends that do not implement the sparse path ignore src[5] and compute
            // the dense result, so this stays correct everywhere; it only ever changes
            // which kernel runs.
            cur->src[5] = selected;
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // DS4 scores need the F32 accumulator (long context, sinks, -inf masks)
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

//
// compressed-KV reconstruction from the compressor ring state
//

// upstream: build_hca_compressed_kv_from_state(). ratio 128, no overlap.
ggml_tensor * llm_build_context::build_dsv4_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t       n_embd_head,
        const char *  name,
        int           il) const {
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv,    1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = llm_build_norm(ctx0, comp, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(comp, name, il);

    if (dsv4_rope_flipped_enabled()) {
        // comp is already [n_embd_head, 1, n_blocks] here (sum_rows -> permute ->
        // cont -> norm), which is exactly the shape the rope needs: ne[2] must
        // equal comp_pos->ne[0]. Assert instead of inserting a no-op reshape.
        GGML_ASSERT(comp->ne[0] == n_embd_head && comp->ne[1] == 1 && comp->ne[2] == n_blocks);
        comp = dsv4_rope_flipped(ctx0, comp, comp_pos, n_embd_head_rope, rope_type, n_ctx_orig,
                hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        cb(comp, name, il);
    } else {
        ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
                ggml_row_size(comp->type, n_embd_head),
                ggml_row_size(comp->type, n_embd_head),
                0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
                ggml_row_size(comp->type, n_embd_head),
                ggml_row_size(comp->type, n_embd_head),
                ggml_row_size(comp->type, n_embd_head_nope));

        comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
                hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        cb(comp_pe, name, il);

        comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
        cb(comp, name, il);
    }

    return comp;
}

// upstream: build_overlap_compressed_kv_from_state(). ratio 4 with the
// previous/current overlapped halves (CSA and LID).
ggml_tensor * llm_build_context::build_dsv4_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t       ratio,
        int64_t       n_embd_head,
        const char *  name,
        int           il) const {
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0]    == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    const int64_t n_read = ratio*n_blocks;

    ggml_tensor * kv_rows    = ggml_get_rows(ctx0, kv_state,    state_read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

    ggml_tensor * kv_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev,    kv_cur,    1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = llm_build_norm(ctx0, comp, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(comp, name, il);

    if (dsv4_rope_flipped_enabled()) {
        // comp is already [n_embd_head, 1, n_blocks] here (sum_rows -> permute ->
        // cont -> norm), which is exactly the shape the rope needs: ne[2] must
        // equal comp_pos->ne[0]. Assert instead of inserting a no-op reshape.
        GGML_ASSERT(comp->ne[0] == n_embd_head && comp->ne[1] == 1 && comp->ne[2] == n_blocks);
        comp = dsv4_rope_flipped(ctx0, comp, comp_pos, n_embd_head_rope, rope_type, n_ctx_orig,
                hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        cb(comp, name, il);
    } else {
        ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
                ggml_row_size(comp->type, n_embd_head),
                ggml_row_size(comp->type, n_embd_head),
                0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
                ggml_row_size(comp->type, n_embd_head),
                ggml_row_size(comp->type, n_embd_head),
                ggml_row_size(comp->type, n_embd_head_nope));

        comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
                hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        cb(comp_pe, name, il);

        comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
        cb(comp, name, il);
    }

    return comp;
}

//
// lightning indexer
//

// upstream: build_lid_top_k(). Fused path (ggml_lightning_indexer) not ported.
ggml_tensor * llm_build_context::build_dsv4_lid_top_k(
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int           il) const {
    const auto & layer   = model.layers[il];
    const auto & inp_lid = llama_dsv4_get_inputs(lctx).comp[LLAMA_DSV4_LID];

    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot;
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = llm_build_lora_mm(lctx, ctx0, layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    if (dsv4_rope_flipped_enabled()) {
        indexer_q = dsv4_rope_flipped(ctx0, indexer_q, inp_pos, n_embd_indexer_head_rope,
                rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
                ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        cb(indexer_q, "lid_q_pe", il);
    } else {
        ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, nt,
                ggml_row_size(indexer_q->type, n_embd_indexer_head),
                ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
                0);
        ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, nt,
                ggml_row_size(indexer_q->type, n_embd_indexer_head),
                ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
                ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

        indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
                rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
                ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        cb(indexer_q_pe, "lid_q_pe", il);

        indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    }
    if (inp_lid.k_rot) {
        indexer_q = dsv4_mul_mat_rot(ctx0, indexer_q, inp_lid.k_rot);
        cb(indexer_q, "lid_q_rot", il);
    }

    ggml_tensor * indexer_weights = llm_build_lora_mm(lctx, ctx0, layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = llama_dsv4_get_comp_k(lctx, ctx0, LLAMA_DSV4_LID, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    GGML_ASSERT(n_stream == 1 && "DSV4 first cut is single-stream (-np 1)");

    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    cb(indexer_q, "lid_q", il);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
    cb(indexer_k, "lid_k", il);

    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
    cb(indexer_kq, "lid_kq", il);

    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
    cb(indexer_kq, "lid_kq", il);

    ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
    indexer_score = ggml_sum_rows(ctx0, indexer_score);
    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
    cb(indexer_score, "lid_score", il);

    indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
    cb(indexer_score, "lid_score_masked", il);

    const int64_t n_top_k = indexer_score->ne[0] < (int64_t) hparams.indexer_top_k
        ? indexer_score->ne[0] : (int64_t) hparams.indexer_top_k;

    // NOTE: ggml_top_k() returns a STRIDED VIEW in this tree (argsort + view_4d),
    // where upstream returns a contiguous I32 tensor. The ggml_cont() below is
    // what makes the downstream set_rows legal - do not "optimize" it away.
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, (int) n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

// upstream: build_top_k_mask(). Turns the top-k index list into an additive
// -inf/0 mask and folds in the base visibility mask.
ggml_tensor * llm_build_context::build_dsv4_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char *  name,
        int           il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);
    GGML_ASSERT(kq_mask->type == GGML_TYPE_F32 && "DSV4 comp masks are F32 in this tree (ggml_fill/ggml_set_rows)");

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // upstream picks F16 here under -fa; this tree's ggml_set_rows asserts a F32
    // source, and the CUDA backend supports the F32 -> F16/F32 store, so F32 it is.
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

//
// the three per-layer attention regimes
//

// upstream: build_csa_lid_attention(). raw SWA window + top-k(512) sparse
// compressed rows, selected by the lightning indexer.
ggml_tensor * llm_build_context::build_dsv4_csa_lid_attention(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const llama_dsv4_inputs & inp     = llama_dsv4_get_inputs(lctx);
    const auto              & inp_csa = inp.comp[LLAMA_DSV4_CSA];

    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_dsv4_lid_top_k(qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp.raw_k_rot;
    if (k_rot) {
        q  = dsv4_mul_mat_rot(ctx0, q,  k_rot);
        kv = dsv4_mul_mat_rot(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    ggml_tensor * raw_ring_w = llama_dsv4_cpy_raw_k(lctx, ctx0, kv, inp.raw_k_idxs, il);
    ggml_build_forward_expand(gf, raw_ring_w);

    ggml_tensor * raw_k = llama_dsv4_raw_window_enabled(lctx)
        ? llama_dsv4_get_raw_win_k(lctx, ctx0, raw_ring_w, inp.raw_win_idxs, il)
        : llama_dsv4_get_raw_k(lctx, ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = llama_dsv4_get_comp_k(lctx, ctx0, LLAMA_DSV4_CSA, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    ggml_tensor * raw_mask = inp.raw_kq_mask;
    ggml_tensor * csa_mask = build_dsv4_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    kq_mask = dsv4_finalize_kq_mask(ctx0, kq_mask, cparams.flash_attn);
    cb(kq_mask, "csa_lid_kq_mask", il);

    // Visible-row bound for the sparse path. kq_mask is [raw | csa]:
    //   raw  - the ring/window mask admits a key only when 0 <= p - pos < n_swa
    //          (llama-kv-cache-dsv4.cpp), so at most n_swa rows per query, however
    //          wide the padded ring is;
    //   csa  - build_dsv4_top_k_mask() starts all -inf and zeroes exactly the
    //          top_k->ne[0] selected rows before adding the base mask, so at most
    //          top_k->ne[0] rows.
    // Their sum is therefore a true upper bound, and it does not grow with context
    // while the dense mask does -- which is where the win comes from.
    const int64_t n_visible_max = (int64_t) llama_dsv4_get_raw_swa(lctx) + top_k->ne[0];

    ggml_tensor * out = build_dsv4_attn_mha(gf, q, k_all, k_all, kq_mask, sinks, kq_scale, il, n_visible_max);
    if (k_rot) {
        out = dsv4_mul_mat_rot(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

// upstream: build_hca_attention(). raw SWA window + every completed ratio-128
// compressed row (no sparsity).
ggml_tensor * llm_build_context::build_dsv4_hca_attention(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const llama_dsv4_inputs & inp     = llama_dsv4_get_inputs(lctx);
    const auto              & inp_hca = inp.comp[LLAMA_DSV4_HCA];

    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp.raw_k_rot;
    if (k_rot) {
        q  = dsv4_mul_mat_rot(ctx0, q,  k_rot);
        kv = dsv4_mul_mat_rot(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    ggml_tensor * raw_ring_w = llama_dsv4_cpy_raw_k(lctx, ctx0, kv, inp.raw_k_idxs, il);
    ggml_build_forward_expand(gf, raw_ring_w);

    ggml_tensor * raw_k = llama_dsv4_raw_window_enabled(lctx)
        ? llama_dsv4_get_raw_win_k(lctx, ctx0, raw_ring_w, inp.raw_win_idxs, il)
        : llama_dsv4_get_raw_k(lctx, ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = llama_dsv4_get_comp_k(lctx, ctx0, LLAMA_DSV4_HCA, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, inp.raw_kq_mask, inp_hca.kq_mask, 0);
    kq_mask = dsv4_finalize_kq_mask(ctx0, kq_mask, cparams.flash_attn);
    cb(kq_mask, "hca_kq_mask", il);

    // As in the CSA regime, but HCA applies no top-k sparsity: every one of the n_hca
    // compressed rows may be visible. The bound is still below the dense mask width
    // whenever the padded raw ring is wider than n_swa, which it always is here
    // (raw_size = PAD(n_ubatch + n_swa + 1, 256)).
    const int64_t n_visible_max = (int64_t) llama_dsv4_get_raw_swa(lctx) + n_hca;

    ggml_tensor * out = build_dsv4_attn_mha(gf, q, k_all, k_all, kq_mask, sinks, kq_scale, il, n_visible_max);
    if (k_rot) {
        out = dsv4_mul_mat_rot(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

// upstream: build_raw_attention(). ratio 0 layers: sliding window only.
ggml_tensor * llm_build_context::build_dsv4_raw_attention(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const llama_dsv4_inputs & inp = llama_dsv4_get_inputs(lctx);

    ggml_tensor * k_rot = inp.raw_k_rot;
    if (k_rot) {
        q  = dsv4_mul_mat_rot(ctx0, q,  k_rot);
        kv = dsv4_mul_mat_rot(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    ggml_tensor * raw_ring_w = llama_dsv4_cpy_raw_k(lctx, ctx0, kv, inp.raw_k_idxs, il);
    ggml_build_forward_expand(gf, raw_ring_w);

    ggml_tensor * k = llama_dsv4_raw_window_enabled(lctx)
        ? llama_dsv4_get_raw_win_k(lctx, ctx0, raw_ring_w, inp.raw_win_idxs, il)
        : llama_dsv4_get_raw_k(lctx, ctx0, il);

    ggml_tensor * kq_mask = dsv4_finalize_kq_mask(ctx0, inp.raw_kq_mask, cparams.flash_attn);

    // Same n_swa bound as the raw half of the other two regimes. There is no compressed
    // stream here, so this only pays when the padded ring is much wider than the window
    // -- which is exactly the ring-mode decode case, where raw_size is
    // PAD(n_ubatch + n_swa + 1, 256) but a query still sees at most n_swa rows.
    // In window mode the raw view is already tight and the worth-it test declines.
    // (ik passes -1 here because their raw path is per-ubatch and already narrow; ours
    // is a modular ring, so the same argument does not carry over.)
    const int64_t n_visible_max = (int64_t) llama_dsv4_get_raw_swa(lctx);

    ggml_tensor * out = build_dsv4_attn_mha(gf, q, k, k, kq_mask, sinks, kq_scale, il, n_visible_max);
    if (k_rot) {
        out = dsv4_mul_mat_rot(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

//
// full per-layer attention (upstream: build_attention)
//

ggml_tensor * llm_build_context::build_dsv4_attention(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int           il) const {
    const auto & layer = model.layers[il];
    llama_dsv4_inputs & inp = llama_dsv4_get_inputs(lctx);

    const int64_t n_embd_head      = hparams.n_embd_head_k_full;
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == (int64_t) hparams.n_embd_head_v_full);
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];

    const bool    use_compress_rope = ratio != 0;
    const float   freq_base_l       = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l      = use_compress_rope ? freq_scale : 1.0f;
    const float   ext_factor_l      = use_compress_rope ? ext_factor : 0.0f;
    const float   attn_factor_l     = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float   beta_fast_l       = use_compress_rope ? beta_fast : 0.0f;
    const float   beta_slow_l       = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l      = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    if (dsv4_rope_flipped_enabled()) {
        q = dsv4_rope_flipped(ctx0, q, inp_pos, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(q, "q", il);
    } else {
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
                ggml_row_size(q->type, n_embd_head),
                ggml_row_size(q->type, n_embd_head)*n_head,
                0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
                ggml_row_size(q->type, n_embd_head),
                ggml_row_size(q->type, n_embd_head)*n_head,
                ggml_row_size(q->type, n_embd_head_nope));
        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(q_pe, "q_pe", il);
        q = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(q, "q", il);
    }

    ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, layer.wkv, cur);
    kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    if (dsv4_rope_flipped_enabled()) {
        kv = dsv4_rope_flipped(ctx0, kv, inp_pos, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(kv, "kv", il);
    } else {
        ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                0);
        ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head_nope));
        kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(kv_pe, "kv_pe", il);
        kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
        cb(kv, "kv", il);
    }

    auto & inp_csa = inp.comp[LLAMA_DSV4_CSA];
    auto & inp_hca = inp.comp[LLAMA_DSV4_HCA];
    auto & inp_lid = inp.comp[LLAMA_DSV4_LID];

    // ---- HCA (ratio 128) compressor-state projections ----
    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_hca.state_pos) {
        hca_state_kv = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, layer.attn_comp_ape, inp_hca.state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);
    }

    // ---- CSA + LID (ratio 4) compressor-state update and commit ----
    if (ratio == DSV4_CSA_RATIO && inp_csa.state_pos) {
        ggml_tensor * csa_state_kv = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, layer.attn_comp_ape, inp_csa.state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_csa.state_write_idxs);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0,
                llama_dsv4_get_state_kv(lctx, ctx0, LLAMA_DSV4_CSA, il), csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0,
                llama_dsv4_get_state_score(lctx, ctx0, LLAMA_DSV4_CSA, il), csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_dsv4_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_csa.state_read_idxs,
                inp_csa.state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_csa.k_rot) {
            kv_comp_csa_state = dsv4_mul_mat_rot(ctx0, kv_comp_csa_state, inp_csa.k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, llama_dsv4_cpy_comp_k(lctx, ctx0, LLAMA_DSV4_CSA,
                    kv_comp_csa_state, inp_csa.state_write_idxs, il));

        csa_state_kv    = dsv4_with_zero_dep(ctx0, csa_state_kv,    kv_comp_csa_state);
        csa_state_score = dsv4_with_zero_dep(ctx0, csa_state_score, kv_comp_csa_state);

        ggml_tensor * csa_persist_kv    = ggml_get_rows(ctx0, csa_state_kv,    inp_csa.state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_csa.state_persist_src_idxs);

        csa_state_kv = llama_dsv4_cpy_state_kv(lctx, ctx0, LLAMA_DSV4_CSA,
                csa_persist_kv, inp_csa.state_persist_dst_idxs, il);
        csa_state_score = llama_dsv4_cpy_state_score(lctx, ctx0, LLAMA_DSV4_CSA,
                csa_persist_score, inp_csa.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = llm_build_lora_mm(lctx, ctx0, layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = llm_build_lora_mm(lctx, ctx0, layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, layer.indexer_comp_ape, inp_lid.state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_lid.state_write_idxs);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0,
                llama_dsv4_get_state_kv(lctx, ctx0, LLAMA_DSV4_LID, il), lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0,
                llama_dsv4_get_state_score(lctx, ctx0, LLAMA_DSV4_LID, il), lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_dsv4_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_lid.state_read_idxs,
                inp_lid.state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_lid.k_rot) {
            kv_comp_lid_state = dsv4_mul_mat_rot(ctx0, kv_comp_lid_state, inp_lid.k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, llama_dsv4_cpy_comp_k(lctx, ctx0, LLAMA_DSV4_LID,
                    kv_comp_lid_state, inp_lid.state_write_idxs, il));

        lid_state_kv    = dsv4_with_zero_dep(ctx0, lid_state_kv,    kv_comp_lid_state);
        lid_state_score = dsv4_with_zero_dep(ctx0, lid_state_score, kv_comp_lid_state);

        ggml_tensor * lid_persist_kv    = ggml_get_rows(ctx0, lid_state_kv,    inp_lid.state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_lid.state_persist_src_idxs);

        lid_state_kv = llama_dsv4_cpy_state_kv(lctx, ctx0, LLAMA_DSV4_LID,
                lid_persist_kv, inp_lid.state_persist_dst_idxs, il);
        lid_state_score = llama_dsv4_cpy_state_score(lctx, ctx0, LLAMA_DSV4_LID,
                lid_persist_score, inp_lid.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    // ---- HCA commit ----
    ggml_tensor * hca_state_dep = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_hca.state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        ggml_tensor * hca_source_kv = ggml_concat(ctx0,
                llama_dsv4_get_state_kv(lctx, ctx0, LLAMA_DSV4_HCA, il), hca_state_kv, 1);
        ggml_tensor * hca_source_score = ggml_concat(ctx0,
                llama_dsv4_get_state_score(lctx, ctx0, LLAMA_DSV4_HCA, il), hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_dsv4_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_hca.state_read_idxs,
                inp_hca.state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_hca.k_rot) {
            kv_comp_hca = dsv4_mul_mat_rot(ctx0, kv_comp_hca, inp_hca.k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, llama_dsv4_cpy_comp_k(lctx, ctx0, LLAMA_DSV4_HCA,
                    kv_comp_hca, inp_hca.state_write_idxs, il));
        hca_state_dep = kv_comp_hca;
    }

    if (ratio == DSV4_HCA_RATIO && inp_hca.state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state_kv    = dsv4_with_zero_dep(ctx0, hca_state_kv,    hca_state_dep);
        hca_state_score = dsv4_with_zero_dep(ctx0, hca_state_score, hca_state_dep);

        ggml_tensor * hca_persist_kv    = ggml_get_rows(ctx0, hca_state_kv,    inp_hca.state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_hca.state_persist_src_idxs);

        hca_state_kv = llama_dsv4_cpy_state_kv(lctx, ctx0, LLAMA_DSV4_HCA,
                hca_persist_kv, inp_hca.state_persist_dst_idxs, il);
        hca_state_score = llama_dsv4_cpy_state_score(lctx, ctx0, LLAMA_DSV4_HCA,
                hca_persist_score, inp_hca.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    // ---- the attention itself ----
    // NOTE (deviation from upstream): upstream additionally requires
    // `inp_lid.k_rot != nullptr` to take the CSA/LID branch, because for its LID
    // cache attn_rot_k is force-enabled (llama-kv-cache.cpp: the DEEPSEEK4 /
    // GLM_DSA indexer special case), so it is never null there. Requiring it here
    // would silently demote every ratio-4 layer to plain SWA if the memory chunk
    // does not build a rotation matrix. The rotation is orthogonal and applied to
    // both indexer q and indexer k, so the scores - and therefore the top-k - are
    // identical with or without it; it exists to condition quantized indexer-K
    // storage. So: select on the ratio and the masks, and rotate only if k_rot
    // was actually provided.
    ggml_tensor * out = nullptr;
    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    if (ratio == DSV4_CSA_RATIO && inp_csa.kq_mask && inp_lid.kq_mask) {
        out = build_dsv4_csa_lid_attention(gf, q, kv, qr, cur, inp_pos, layer.attn_sinks, kq_scale, il);
    } else if (ratio == DSV4_HCA_RATIO && inp_hca.kq_mask) {
        out = build_dsv4_hca_attention(gf, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        out = build_dsv4_raw_attention(gf, q, kv, layer.attn_sinks, kq_scale, il);
    }

    // ---- de-rope, then the grouped output LoRA ----
    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    if (dsv4_rope_flipped_enabled()) {
        out = dsv4_rope_back_flipped(ctx0, out, inp_pos, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(out, "attn_derope", il);
    } else {
        ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
                ggml_row_size(out->type, n_embd_head),
                ggml_row_size(out->type, n_embd_head)*n_head,
                0);
        ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
                ggml_row_size(out->type, n_embd_head),
                ggml_row_size(out->type, n_embd_head)*n_head,
                ggml_row_size(out->type, n_embd_head_nope));

        // upstream calls this ggml_rope_ext_back; this tree still has the original
        // name ggml_rope_back with a byte-identical parameter list.
        out_pe = ggml_rope_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        out = ggml_concat(ctx0, out_nope, out_pe, 0);
        cb(out, "attn_derope", il);
    }

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0,
            ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

//
// the graph
//

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hc == 4 && "DSV4 loader/graph assume hyper_connection.count == 4");
    GGML_ASSERT(hparams.n_embd_head_k_full == hparams.n_embd_head_v_full);

    ggml_tensor * cur;

    ggml_tensor * inp         = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_tokens  = lctx.inp_tokens;   // needed by the hash-routed layers
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    build_dsv4_inputs();

    const llama_dsv4_inputs & dsv4 = llama_dsv4_get_inputs(lctx);
    if (dsv4.raw_kq_mask) {
        ggml_build_forward_expand(gf, dsv4.raw_kq_mask);
    }

    // widen the residual stream to hc branches: [n_embd, hc, n_tokens]
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_dsv4_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        cur = build_dsv4_attention(gf, cur, inp_pos, il);

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

        // The first dsv4_hash_layer_count blocks route by a token-id -> expert-id hash
        // table instead of by the router top-k. exp_probs_b does not exist there.
        // Skipped on the warm-up graph: that build forces n_expert_used = n_expert so
        // every expert tensor is touched, and the hash table is only n_expert_used wide
        // (upstream guards this the same way, with !cparams.warmup).
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b      = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count && !is_warmup) {
            GGML_ASSERT(layer.ffn_gate_tid2eid);
            GGML_ASSERT(inp_tokens && "DSV4 hash-routed layers need token ids (embedding input is unsupported)");
            // NOTE: ffn_gate_tid2eid is I32; this tree's CUDA get_rows has no I32
            // path, so this one node falls back to the CPU backend (correct, but
            // it splits the graph on the first dsv4_hash_layer_count blocks).
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            cb(selected_experts, "ffn_moe_topk_hash", il);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                hparams.expert_weights_norm,
                hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, /*add_input*/ false,
                /*up_gate_exps*/ nullptr, /*up_gate_exps_b*/ nullptr,
                /*input_logits*/ nullptr, /*down_exps_s*/ nullptr,
                /*selected_experts_in*/ selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        // shared expert. DS4's SwiGLU clamps the GATE to [-inf, limit] and the UP
        // to [-limit, limit] and only then multiplies - it is NOT silu-then-clamp.
        // Built inline instead of through llm_build_ffn() so the shared helper's
        // fused SILU path stays byte-untouched for every other arch.
        ggml_tensor * ffn_shexp;
        {
            ggml_tensor * up   = llm_build_lora_mm(lctx, ctx0, layer.ffn_up_shexp,   cur);
            cb(up, "ffn_up_shexp", il);
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_shexp, cur);
            cb(gate, "ffn_gate_shexp", il);

            const float limit = hparams.swiglu_limits_shared[il];
            if (limit > 1e-6f) {
                up   = ggml_clamp(ctx0, up,   -limit,     limit);
                gate = ggml_clamp(ctx0, gate, -INFINITY,  limit);
                cb(gate, "ffn_gate_shexp_clamped", il);
            }

            ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
            cb(act, "ffn_swiglu_shexp", il);

            ffn_shexp = llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp, act);
        }
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_last", il);

        int dspark_slot = -1;
        if (dspark_capture_enabled() && dspark_is_capture_layer(il, n_layer, &dspark_slot)) {
            // Take the capture through the same out_ids gather the head uses, so a prompt
            // chunk captures n_outputs rows rather than n_tokens.
            ggml_tensor * src  = inpL;                       // [n_embd, hc, n_tokens]
            ggml_tensor * flat = ggml_reshape_2d(ctx0, src, n_embd*hc, n_tokens);
            int64_t nt = n_tokens;
            if (inp_out_ids && n_tokens > 1) {
                flat = ggml_get_rows(ctx0, flat, inp_out_ids);
                nt   = n_outputs;
                src  = ggml_reshape_3d(ctx0, flat, n_embd, hc, nt);
            }
            // uniform 1/hc weights, shaped [hc, nt]: any [hc, nt] view scaled to zero
            // and biased. Cheap (hc*nt elements) and needs no new graph input.
            ggml_tensor * w = ggml_view_2d(ctx0, flat, hc, nt, flat->nb[1], 0);
            w = ggml_scale_bias(ctx0, ggml_cont(ctx0, w), 0.0f, 1.0f/(float) hc);

            ggml_tensor * capr = ggml_dsv4_hc_weighted_sum(ctx0, src, w);
            ggml_format_name(capr, "dspark_cap_%d", dspark_slot);
            // M5: these nodes have NO consumer, so ggml-alloc is free to hand their
            // storage to the next allocation the instant they are computed
            // (ggml-alloc.c:500,547). Without the OUTPUT flag the host read back float
            // bit patterns of whatever landed there next - the same trap M4 hit twice.
            capr->flags |= GGML_TENSOR_FLAG_OUTPUT;
            ggml_build_forward_expand(gf, capr);
        }
    }

    if (inp_out_ids && n_tokens > 1) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_dsv4_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
