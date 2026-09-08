"""namemap_qwen4exp.py -- ggml -> HF name map for the qwen4exp (Flash-Next) architecture.

STATUS: MAPPING TABLE ONLY, not yet wired into convert.py, and not yet gated. Every entry
below was read out of the fork at <box path>
rather than inferred from the qwen35moe map -- the two architectures share a GDN module and a
prefix convention and differ in almost everything else.

WHAT THE FORK EXPECTS (model.py:659-673)
    checkpoint keys are ``model.language_model.*``; hf_to_vllm_mapper rewrites that prefix to
    ``model.``. Shared experts arrive under ``mlp.shared_expert`` and are fused by
    maybe_fuse_shared_experts. packed_modules_mapping fuses:
        qkv_proj      <- q_proj, k_proj, v_proj
        gate_up_proj  <- gate_proj, up_proj
        in_proj_qkvz  <- in_proj_qkv, in_proj_z
        in_proj_ba    <- in_proj_b, in_proj_a
        input_mix_weight_down_block_inject
                      <- input_mix_weight_down, block_inject_weight, _input_mix_padding

THREE THINGS THAT WOULD OTHERWISE COST A BOOT EACH, found by reading rather than by trying:

 1. hc_*_inject IS **NOT** DISCARDED -- I had this backwards and the correction matters.
    skip_substrs names ``hyper_connection_mixer.block_inject_weight``, which is the
    OUTPUT-level mixer's column ONLY (model.py:499 builds hyper_connection_mixer at model
    level). The PER-LAYER ones are required: _HC_WEIGHTS_MAPPER (model.py:161-172) maps
    checkpoint ``hyper_connection.input_mix_weight_down.weight`` to shard 0 and
    ``hyper_connection.block_inject_weight.weight`` to shard 1 of one stacked
    ``input_mix_weight_down_block_inject``. Emit neither and shard 1 is uninitialised, which
    is the silent-garbage failure and not a load error. So blk.N.hc_attn_inject and
    hc_ffn_inject MUST be emitted, and only output_hc's inject (which the GGUF does not
    carry anyway) is skipped.

 2. THE INDEXER'S q AND k ARE ONE TENSOR ON THE vLLM SIDE. The GGUF has separate
    indexer.q_proj (N=512) and indexer.k_proj (N=128); the fork builds ONE ReplicatedLinear
    ``index_qk_proj`` of (index_n_heads + index_kv_heads) * index_head_dim = (4 + 1) * 128 =
    640 rows -- and it is NOT in packed_modules_mapping, so vLLM will not fuse them for us.
    The converter must CONCATENATE q over k on the output axis and emit one
    ``...self_attn.indexer.index_qk_proj.weight`` of [640, hidden]. Emitting them separately
    would not raise: they would be swallowed as unexpected suffixes and the layer would run
    on uninitialised weights.

 3. THE PLE TABLE IS FP8 WITH ONE GLOBAL SCALE. Qwen4ExpPinnedHostEmbedding raises
    NotImplementedError unless the quant method is Qwen4ExpPLEFp8EmbeddingMethod. Our source
    is q8_0 (int8 + per-32-block fp16 scale). Measured safe: across the 16 n-gram head tables
    the absmax spread is 1.36x and the rms spread 1.05x, so one global e4m3 scale keeps every
    head on 328-448 of 448 levels at ~0.4% rms error. That requantization is blocker 2.

GEOMETRY, read from the GGUF's own KVs (all present; nothing is inferred):
    hidden 2560, 48 blocks, 512 experts top-10, expert_ffn 640
    hyper_connection.count 4, .low_rank 320       -> hc dim 4*2560 = 10240
    attention.indexer.head_count 4, .key_length 128, .top_k 2048
    ple.layers [1], .ngram_size 3, .heads_per_ngram 8, .conv_kernel 4
    embedding_length_per_layer_input 160
    ssm.group_count 16, .time_step_rank 48, .state_size 128, .inner_size 6144
      -> GDN n_k_heads 16, n_v_heads 48, head_dim 128, repeat factor 3
         (qwen35moe is 16/32/128, factor 2 -- SAME module class, different repeat, so the
          v-head permutation logic transfers and the NUMBERS do not. Gate it, do not assume.)
"""

from __future__ import annotations

#: ggml suffix -> HF suffix, relative to ``model.language_model.layers.<i>.``.
#: ``None`` means DO NOT EMIT, with the reason in the comment.
GGML_TO_HF: dict[str, str | None] = {
    # ---- attention (full-attention layers; every 4th per full_attention_interval) --------
    "attn_q.weight":            "self_attn.q_proj.weight",
    "attn_k.weight":            "self_attn.k_proj.weight",
    "attn_v.weight":            "self_attn.v_proj.weight",
    "attn_output.weight":       "self_attn.o_proj.weight",
    "attn_q_norm.weight":       "self_attn.q_norm.weight",
    "attn_k_norm.weight":       "self_attn.k_norm.weight",
    # attn_qkv / attn_gate are the GDN input projection, not the attention one
    "attn_qkv.weight":          "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight":         "linear_attn.in_proj_z.weight",

    # ---- QSA indexer: q and k are ONE tensor on the vLLM side, see note 2 ----------------
    "indexer.q_proj.weight":    "@CONCAT:self_attn.indexer.index_qk_proj.weight:0",
    "indexer.k_proj.weight":    "@CONCAT:self_attn.indexer.index_qk_proj.weight:1",
    "indexer.q_norm.weight":    "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight":    "self_attn.indexer.k_layernorm.weight",

    # ---- GDN linear attention -----------------------------------------------------------
    "ssm_out.weight":           "linear_attn.out_proj.weight",
    "ssm_norm.weight":          "linear_attn.norm.weight",
    "ssm_conv1d.weight":        "linear_attn.conv1d.weight",
    "ssm_a":                    "linear_attn.A_log",
    "ssm_dt.bias":              "linear_attn.dt_bias",
    "ssm_alpha.weight":         "linear_attn.in_proj_b.weight",
    "ssm_beta.weight":          "linear_attn.in_proj_a.weight",

    # ---- MoE ----------------------------------------------------------------------------
    "ffn_gate_inp.weight":      "mlp.gate.weight",
    "ffn_gate_exps.weight":     "mlp.experts.{e}.gate_proj.weight",
    "ffn_up_exps.weight":       "mlp.experts.{e}.up_proj.weight",
    "ffn_down_exps.weight":     "mlp.experts.{e}.down_proj.weight",
    "ffn_gate_shexp.weight":    "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight":      "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight":    "mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",

    # ---- hyperconnection ----------------------------------------------------------------
    "hc_attn_down.weight":      "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_up.weight":        "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_attn_norm.weight":      "attn_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight":       "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_up.weight":         "mlp_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_norm.weight":       "mlp_hyper_connection.hc_norm.weight",
    # See note 1: REQUIRED as shard 1 of the stacked down+inject parameter.
    "hc_attn_inject.weight":    "attn_hyper_connection.block_inject_weight.weight",
    "hc_ffn_inject.weight":     "mlp_hyper_connection.block_inject_weight.weight",

    # ---- PLE stack (only on the layers in ple.layers, here [1]) --------------------------
    "ple_key.weight":           "ple.key_proj.weight",
    "ple_value.weight":         "ple.value_proj.weight",
    "ple_norm_key.weight":      "ple.norm_key.weight",
    "ple_norm_query.weight":    "ple.norm_query.weight",
    "ple_norm_conv.weight":     "ple.norm_conv.weight",
    "ple_conv1d.weight":        "ple.conv1d.weight",

    # ---- norms: THERE ARE NONE, and that is the finding ---------------------------------
    # I first wrote attn_norm -> input_layernorm and post_attention_norm ->
    # post_attention_layernorm here, by analogy with qwen35moe. The coverage gate caught them
    # on its first run: THIS MODEL HAS NO attn_norm, NO post_attention_norm AND NO
    # output_norm TENSOR AT ALL. The hyperconnection carries the normalisation instead --
    # hc_attn_norm and hc_ffn_norm per branch per layer, output_hc_norm at the head -- which
    # is consistent with hc_per_branch_norm=True being passed to HyperConnectionConfig.
    # Left as a comment rather than deleted, because "the obvious norm mapping is wrong here"
    # is the single most likely thing for the next reader to re-add from memory.
}

#: Global (non-layer) tensors.
GGML_TO_HF_GLOBAL: dict[str, str] = {
    "token_embd.weight":            "model.language_model.embed_tokens.weight",
    "output.weight":                "lm_head.weight",
    "per_layer_token_embd.weight":  "model.language_model.ple.ngram_embedding.weight",
    "output_hc_down.weight":        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up.weight":          "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    "output_hc_norm.weight":        "model.language_model.hyper_connection_mixer.hc_norm.weight",
}

#: Everything above is a HYPOTHESIS until the key-set gate runs: build the model under a real
#: VllmConfig, enumerate the parameters it registers, and require this map's image to equal
#: that set. Three of the entries are the ones most likely to be wrong and are flagged so the
#: gate's failure is legible rather than a wall of names:
#: RESOLVED since the first draft, by reading rather than inferring:
#:   hc_*_norm  -> hc_norm. GatedResidual holds hc_norm, input_mix_weight_down and
#:                 input_mix_weight_up DIRECTLY (hyperconnection.py:88,112,121); there is no
#:                 intermediate module with a plain "norm".
#:   output_hc_* -> hyper_connection_mixer, the model-level GatedResidual (model.py:499),
#:                 which is also the name skip_substrs uses.
#:   ple_conv1d -> ple.conv1d (ple_layer.py:923, an nn.Conv1d). My guess was right.
#:
#: RESOLVED, by reading GatedResidual.__init__ rather than the mapper table:
#: THERE IS NO "hyper_connection" MODULE. The mapper's pattern
#: "hyper_connection.input_mix_weight_down.weight" matches as a SUFFIX of
#: "attn_hyper_connection.input_mix_weight_down.weight" -- the attribute names
#: attn_hyper_connection / mlp_hyper_connection / hyper_connection_mixer all END in
#: "hyper_connection", so one pattern catches all three and no intermediate module exists.
#: Every hc_* key is therefore <layer>.<attn|mlp>_hyper_connection.<leaf>.weight.
#:
#: AND THAT EXPLAINS skip_substrs, which had looked arbitrary. GatedResidual takes
#: use_combine (hyperconnection.py:19, default True): with it TRUE -- the per-layer case,
#: since model.py:308/312 pass nothing -- it builds ONE MergedColumnParallelLinear called
#: input_mix_weight_down_block_inject whose two shards are down and inject. With it FALSE --
#: the output-level hyper_connection_mixer, model.py:499 -- it builds a plain ReplicatedLinear
#: input_mix_weight_down and NO inject parameter at all. So the per-layer injects are
#: required (shard 1) and the mixer's must be skipped because there is nowhere to put it,
#: which is exactly the one name skip_substrs lists.
#:
#: THE KEY-SET GATE HAS NOW RUN and confirms the above by ENUMERATION rather than reading.
#: GatedResidual was constructed under a real VllmConfig with this model's geometry
#: (hc_count 4, hidden 2560, hc_lowrank 320, per-branch norm) and asked for its parameters:
#:
#:   use_combine=True  (per layer)   hc_norm.weight                            (10240,)
#:                                   input_mix_weight_down_block_inject.weight (336, 10240)
#:                                   input_mix_weight_up.weight                (10240, 320)
#:   use_combine=False (model level) hc_norm.weight                            (10240,)
#:                                   input_mix_weight_down.weight              (320, 10240)
#:                                   input_mix_weight_up.weight                (10240, 320)
#:
#: No "hyper_connection" child exists in either case -- settled by enumeration, not inference.
#:
#: ONE FACT THE GATE ADDED THAT NO AMOUNT OF READING GAVE ME: the merged parameter is 336
#: rows, not 324. pad_size = (-(hc_lowrank + hc_count)) mod 16 = (-(320 + 4)) mod 16 = 12, so
#: the layout is 320 rows of down, 4 of inject, and 12 of PADDING -- the third element of
#: packed_modules_mapping's "_input_mix_padding". The converter supplies nothing for the pad;
#: it exists so the merged matrix is a multiple of 16. Anyone sizing that tensor from
#: hc_lowrank + hc_count alone would be 12 rows short and would find out either as a
#: load-time shape error or as a silent misalignment of shard 1.
#:
#: GGUF shapes agree with all of it: hc_attn_down (10240, 320) -> N=320 = shard 0,
#: hc_attn_inject (10240, 4) -> N=4 = shard 1, hc_attn_up (320, 10240) -> N=10240, and
#: output_hc_down (10240, 320) -> N=320 with no inject, exactly as use_combine=False builds.
UNVERIFIED = ()
