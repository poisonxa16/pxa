"""namemap_qwen4exp.py -- ggml -> HF name map for the qwen4exp (Flash-Next) architecture.

STATUS: WIRED into convert.py (arch dispatch in namemap.GGML_TO_HF / value_transform /
POLICY_MODULES). Every entry below was read out of the vLLM fork's qwen4_exp package
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
    # CORRECTED 2026-09-06 : the first draft of this table had these
    # two CROSSED (alpha -> in_proj_b, beta -> in_proj_a). Three independent readings agree on
    # the pairing below and none supports the crossed one:
    #   * llama-delta-net.cpp:284-288 -- the un-fused branch computes
    #       beta  = ssm_beta  @ x      alpha = ssm_alpha @ x
    #     and the FUSED branch (:258-278) views b as the FIRST half of ssm_beta_alpha and a as
    #     the second, so ggml's own names are b=beta, a=alpha.
    #   * the fork's create_ba_proj (qwen_gdn_linear_attn.py:2723-2745) builds in_proj_ba as
    #     MergedColumnParallelLinear(output_sizes=[num_v_heads]*2) with shard 0 = in_proj_b and
    #     shard 1 = in_proj_a (model.py:185-186), i.e. b first -- the same order.
    #   * namemap.py's _GDN_MAP records the MEASURED qwen35 pairing against a real reference
    #     checkpoint: alpha -> in_proj_a (per-row rel 0.33) and beta -> in_proj_b (0.34), with
    #     both cross-pairings at 1.15-2.77 (uncorrelated). Same GDN module class, same answer.
    # Crossing them is silent: both tensors are [2560, 48] q8_0, both land in one merged
    # parameter, nothing changes shape, and the model generates fluent garbage.
    "ssm_beta.weight":          "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight":         "linear_attn.in_proj_a.weight",

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


# =============================================================================================
# WIRING (2026-09-06). Everything above is the table; everything below
# is what convert.py needs to USE it. This module stays free of imports from namemap.py so the
# arch dispatch can live there without a cycle.
# =============================================================================================
import re as _re

ARCH = "qwen4exp"
HF_LM = "model.language_model"

_BLK = _re.compile(r"^blk\.(\d+)\.(.+)$")

#: ggml tensor -> why convert.py does NOT emit it. A skip is a DECISION and has to be written
#: down; an unmapped tensor raises instead.
SKIP: dict[str, str] = {
    # The 51 GB n-gram table. It is not a linear and not a panel tier: it is requantized q8_0
    # -> FP8 e4m3 with one global scale by gguf_to_vllm.ple_fp8 and shipped as its own 512
    # loader shards, because Qwen4ExpPinnedHostEmbedding refuses anything but
    # Qwen4ExpPLEFp8EmbeddingMethod. Emitting it here as fp16 would be a 102 GB tensor.
    "per_layer_token_embd.weight": "PLE n-gram table: gguf_to_vllm.ple_fp8 owns it (FP8, 512 shards)",
}

#: ggml suffixes whose HF weight is stored ZERO-CENTRED while ggml stores it offset by +1.
#:
#: MEASURED on this artifact rather than assumed (the sibling arch's list does not transfer:
#: this model has no attn_norm / post_attention_norm / output_norm at all). Every norm below
#: is read by a fork module whose forward is ``normalized * (1.0 + weight)``:
#:   hc_*_norm, output_hc_norm  GroupedGemmaRMSNorm   (common/hyperconnection.py:73-87)
#:   attn_q_norm, attn_k_norm   GemmaRMSNorm          (qwen3_next.py:546-547, imported as
#:                                                     Qwen3NextRMSNorm)
#:   indexer.q_norm/.k_norm     GemmaRMSNorm          (indexer_qsa.py:139-147)
#:   ple_norm_key/query/conv    Qwen4ExpPLEGroupedNorm(ple_layer.py:215-227)
#: and the GGUF's own values are centred on 1.0, not 0.0 (blk.0.hc_attn_norm mean +0.937,
#: blk.3.attn_q_norm mean +1.283 with 98% of entries within 0.35 of 1, ple_norm_key mean
#: +0.893 with 99.9% within 0.35 of 1). Shipping them verbatim scales every one of those norms
#: by ~2x: the activations stay bounded because each later norm renormalises, so the model
#: loads, runs, and babbles -- the same failure the sibling arch hit on 2026-08-19.
#:
#: ssm_norm is DELIBERATELY ABSENT. linear_attn.norm is RMSNormGated, which multiplies by the
#: PLAIN weight, and namemap.py's qwen35 measurement against a real reference says the same.
GEMMA_NORM_SUFFIXES: tuple[str, ...] = (
    "hc_attn_norm.weight",
    "hc_ffn_norm.weight",
    "output_hc_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
    "ple_norm_key.weight",
    "ple_norm_query.weight",
    "ple_norm_conv.weight",
)

#: ggml suffix carrying A rather than log(-A); emitted as A_log = log(-A). Same as the sibling.
A_LOG_SUFFIXES: tuple[str, ...] = ("ssm_a",)

# ---------------------------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------------------------
#: PANEL GEOMETRY DECIDES THIS LIST, not appetite. A module is served as a native panel tier
#: only if (a) every ggml tensor that lands in the same vLLM PARAMETER carries the same tier,
#: and (b) the parameter's per-rank shard stays a whole number of 64-row panels (column
#: parallel) or 32-column slabs (row parallel).
#:
#: What that excludes here, and why -- both are load-bearing:
#:
#:  self_attn.qkv_proj    attn_q is pxq4 on disk but attn_k and attn_v are q8_0, and
#:                        QKVParallelLinear is one parameter (qsa.py:256). A pxq4 q beside a
#:                        q8_0 k/v is the §3.1 violation the converter refuses, so the whole
#:                        fused module is decoded to fp16: 12 layers x 12288x2560 = 0.75 GB.
#:
#:  mlp.shared_expert.gate_up_proj
#:                        MergedColumnParallelLinear over two 640-row halves. 640 rows is 10
#:                        panels, so at TP=4 each rank would get 160 rows = 2.5 panels. Served
#:                        dense fp16 instead (48 layers x 2 x 640 x 2560 x 2 B = 0.31 GB).
#:                        down_proj is row parallel on K=640 -> 160 columns = 5 slabs/rank, so
#:                        it stays a panel tier.
#:
#: THE ROUTED EXPERTS ARE THE SAME 640 AND ARE **NOT** SERVED BY TENSOR PARALLEL AT TP=4.
#: FusedMoE's w13 is [E, 2*I_p, H]; at TP=4 I_p = 160 and the gate/up boundary lands mid-panel.
#: pxq4_vllm.moe._place computes ``per = data.shape[1] // 2`` and would SILENTLY take 2 panels
#: of the 5 it needs. The converter asserts the expert shard arithmetic at TP=(1, 2) only, and
#: a 4-way shape has to be expert parallel (experts whole, I_p = 640 = 10 panels) or pipeline
#: parallel. That is a geometry fact about a 640-wide expert, not a bug in this checkpoint.
PXQ_MODULES_F1: frozenset[str] = frozenset({
    "mlp.experts",
    "mlp.shared_expert.down_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj",
})

#: Expert stacks shard on the intermediate axis, which is 640 = 10 panels: 2-way is 5 panels,
#: 4-way is 2.5. Checked at (1, 2) so the checkpoint is honest about which shapes it serves.
EXPERT_TP_SIZES: tuple[int, ...] = (1, 2)

#: Every linear the fork builds WITH a quant_config, so anything the policy does not serve has
#: to be named in ``ignore`` or get_quant_method would route it nowhere. The two MoE gates
#: (mlp.gate, mlp.shared_expert_gate) and every hyperconnection projection are built with
#: quant_config=None and are therefore never asked about; they are not listed.
ALL_LINEAR_MODULES: tuple[str, ...] = (
    "mlp.experts",
    "mlp.shared_expert.gate_up_proj",
    "mlp.shared_expert.down_proj",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "indexer.index_qk_proj",
    "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj",
    "ple.key_proj",
    "ple.value_proj",
    "lm_head",
)


# ---------------------------------------------------------------------------------------------
# name mapping
# ---------------------------------------------------------------------------------------------
def ple_ggml_layer(tensor_names) -> int | None:
    """The ggml BLOCK index that actually carries the PLE stack, read off the tensor directory.

    The GGUF's ``qwen4exp.ple.layers`` KV is [1] and the fork's ple_layer_ids are ONE-based
    (config.py:123-127 validates 1 <= id <= num_hidden_layers), while ggml block indices are
    ZERO-based -- so the KV alone cannot say which decoder layer owns the stack, and being one
    layer out is silent. The tensors can: exactly one block carries ``ple_key.weight``, and a
    llama.cpp converter writes HF layer L to block L. So the block index IS the zero-based HF
    layer, and config.json must declare ``ple_layer_ids = [block + 1]``.
    """
    for n in tensor_names:
        m = _BLK.match(n)
        if m and m.group(2) == "ple_key.weight":
            return int(m.group(1))
    return None


def ggml_to_hf(name: str, kv: dict) -> str | None:
    """ggml tensor name -> HF tensor name, ``None`` for a deliberate skip.

    A ``@CONCAT:<hf name>:<part>`` result asks the caller to build ONE output tensor by
    concatenating the parts in index order on the output axis; see note 2 in the module
    docstring for why the indexer needs it.
    """
    if name in SKIP:
        return None
    if name in GGML_TO_HF_GLOBAL:
        return GGML_TO_HF_GLOBAL[name]
    m = _BLK.match(name)
    if m is None:
        raise KeyError(f"namemap_qwen4exp: {name!r} is neither a block tensor nor a known "
                       f"global. Refusing to guess -- an unmapped tensor is a silently "
                       f"incomplete model.")
    layer, suffix = int(m.group(1)), m.group(2)
    if suffix not in GGML_TO_HF:
        raise KeyError(f"namemap_qwen4exp: no HF name for ggml suffix {suffix!r} "
                       f"(tensor {name!r}).")
    hf = GGML_TO_HF[suffix]
    if hf is None:
        return None
    prefix = f"{HF_LM}.layers.{layer}"
    if hf.startswith("@CONCAT:"):
        _, target, part = hf.split(":")
        return f"@CONCAT:{prefix}.{target}:{part}"
    return f"{prefix}.{hf}"
