"""config_qwen4exp.py -- build a Qwen4Exp config.json from the GGUF's own KVs.

EVERY FIELD IS TRACED TO A SOURCE KEY. Nothing here is a default I liked the look of: if the
GGUF does not name it, it is not set, and the fork's own dataclass default applies. That
matters because a config.json is the one artifact in this conversion that is AUTHORED rather
than derived, and an authored number that is wrong produces a model that loads and is subtly
mis-shaped -- the failure mode with no error message.

THE GATE IS THE CLASS ITSELF. build() returns a dict; validate() feeds it to the fork's real
Qwen4ExpConfig, whose _validate_ple_config / _validate_ple_layer_ids / _validate_qsa_config
run on construction. If the map is wrong in a way the class can see, it raises there rather
than at model load.
"""

from __future__ import annotations

from typing import Any

#: GGUF KV suffix (under the arch namespace) -> config.json field. One line per field so the
#: provenance of every value is greppable.
_TEXT_FROM_KV: dict[str, str] = {
    "block_count":                        "num_hidden_layers",
    "embedding_length":                   "hidden_size",
    "attention.head_count":               "num_attention_heads",
    "attention.head_count_kv":            "num_key_value_heads",
    "attention.key_length":               "head_dim",
    "attention.layer_norm_rms_epsilon":   "rms_norm_eps",
    "expert_count":                       "num_experts",
    "expert_used_count":                  "num_experts_per_tok",
    "expert_feed_forward_length":         "moe_intermediate_size",
    "expert_shared_feed_forward_length":  "shared_expert_intermediate_size",
    "context_length":                     "max_position_embeddings",
    "full_attention_interval":            "full_attention_interval",
    # hyperconnection
    "hyper_connection.count":             "hc_count",
    "hyper_connection.low_rank":          "hc_lowrank",
    # per-layer embeddings
    "ple.ngram_size":                     "ngram_size",
    "ple.heads_per_ngram":                "heads_per_ngram",
    "ple.conv_kernel":                    "ple_conv_kernel_size",
    # NB: embedding_length_per_layer_input is PER NGRAM HEAD and ple_embed_dim is the
    # TOTAL over the heads -- derived in build(), NOT copied. See the note there.
    # GDN
    "ssm.conv_kernel":                    "linear_conv_kernel_dim",
    "ssm.state_size":                     "linear_key_head_dim",
    "ssm.group_count":                    "linear_num_key_heads",
    "ssm.time_step_rank":                 "linear_num_value_heads",
}

#: Fields whose GGUF value needs a transform rather than a copy, with the reason.
def _ple_layer_ids(kv: dict, arch: str, ple_ggml_layer: int | None = None) -> list[int] | None:
    """The fork's ONE-BASED ple_layer_ids (config.py:123-127 validates 1 <= id <= n_layers).

    THE KV ALONE CANNOT ANSWER THIS AND THE DIFFERENCE IS SILENT. ``qwen4exp.ple.layers`` is
    [1] in this artifact, but ggml block indices are ZERO-based, so [1] read as one-based puts
    the PLE stack on decoder layer 0 while the file's ``blk.1.ple_*`` tensors say layer 1. One
    of those is a whole layer out, and a PLE attached to the wrong layer loads cleanly.

    The tensor directory settles it: a llama.cpp converter writes HF layer L to block L, so the
    block that carries ple_key.weight IS the zero-based layer, and the one-based id is
    block + 1. When the caller supplies that block (``ple_ggml_layer``) it wins over the KV and
    the KV is reported when the two disagree. With no block supplied we fall back to the KV and
    say so, because refusing would block a header-only dry run.
    """
    kv_ids = kv.get(f"{arch}.ple.layers")
    kv_ids = [int(x) for x in kv_ids] if kv_ids is not None else None
    if ple_ggml_layer is None:
        return kv_ids
    ids = [int(ple_ggml_layer) + 1]
    if kv_ids is not None and kv_ids != ids:
        print(f"config_qwen4exp: {arch}.ple.layers says {kv_ids} but the PLE tensors are on "
              f"ggml block {ple_ggml_layer}, i.e. one-based layer {ids[0]}. Using the tensors: "
              f"the KV is ambiguous about its base, the directory is not.")
    return ids


def build(kv: dict[str, Any], ple_ggml_layer: int | None = None) -> dict[str, Any]:
    """GGUF KVs -> the text half of config.json. Fields absent from the file are OMITTED."""
    arch = kv.get("general.architecture")
    if arch != "qwen4exp":
        raise ValueError(f"config_qwen4exp: this builder is for qwen4exp, got {arch!r}")

    text: dict[str, Any] = {"model_type": "qwen4_exp_text"}
    missing: list[str] = []
    for suffix, field in _TEXT_FROM_KV.items():
        key = f"{arch}.{suffix}"
        if key in kv:
            text[field] = kv[key]
        else:
            missing.append(key)

    ids = _ple_layer_ids(kv, arch, ple_ggml_layer)
    if ids is not None:
        text["ple_layer_ids"] = ids
    else:
        missing.append(f"{arch}.ple.layers")

    # linear_value_head_dim is not a KV: ggml gives ssm.inner_size = n_v_heads * head_dim, so
    # the head dim is derived, and the derivation is asserted rather than assumed.
    inner = kv.get(f"{arch}.ssm.inner_size")
    nv = kv.get(f"{arch}.ssm.time_step_rank")
    if inner is not None and nv:
        if int(inner) % int(nv):
            raise ValueError(f"ssm.inner_size {inner} is not divisible by n_v_heads {nv}")
        text["linear_value_head_dim"] = int(inner) // int(nv)
    else:
        missing.append(f"{arch}.ssm.inner_size")

    # ple_embed_dim is NOT the GGUF's embedding_length_per_layer_input. That KV is the width of
    # ONE ngram head's row (160 here, and it is exactly the width of every emitted PLE shard);
    # the fork's ple_embed_dim is the TOTAL over the heads, because Qwen4ExpNGramEmbedding
    # (ple_layer.py:474-480) divides it back down: ngram_heads = (ngram_size - 1) *
    # heads_per_ngram, head_dim = ple_embed_dim // ngram_heads, and the ngram table is built
    # [padded_vocab, head_dim]. Copying the KV straight through set ple_embed_dim=160, which
    # gave head_dim=10 and two hard failures the boot could not survive:
    #   Shape mismatch for PLE embedding shard 0: expected (625003, 10), got (625003, 160)
    #   Error loading weight 'layers.1.ple.key_proj.weight' with checkpoint shape
    #     (10240, 2560) into parameter shape (10240, 160)
    # Both agree on the same number: key_proj is ReplicatedLinear(ple_embed_dim, hc_hidden) and
    # value_proj is ReplicatedLinear(ple_embed_dim, hidden), and this checkpoint's are
    # [10240, 2560] and [2560, 2560] -- so ple_embed_dim = 2560 = 160 * 16 heads.
    per_head = kv.get(f"{arch}.embedding_length_per_layer_input")
    ngram_size = text.get("ngram_size")
    heads_per_ngram = text.get("heads_per_ngram")
    if per_head is not None and ngram_size and heads_per_ngram:
        ngram_heads = (int(ngram_size) - 1) * int(heads_per_ngram)
        if ngram_heads <= 0:
            raise ValueError(f"ngram_heads must be positive, got {ngram_heads}")
        text["ple_embed_dim"] = int(per_head) * ngram_heads
    elif per_head is None:
        missing.append(f"{arch}.embedding_length_per_layer_input")
    else:
        raise ValueError("cannot derive ple_embed_dim without ngram_size and heads_per_ngram")

    # vocab: the tokeniser's own table is the authority, not a KV.
    toks = kv.get("tokenizer.ggml.tokens")
    if toks is not None:
        text["vocab_size"] = len(toks)
    else:
        missing.append("tokenizer.ggml.tokens")

    # rope. dimension_sections is the mrope layout; ggml carries a trailing 0 that HF omits.
    rope: dict[str, Any] = {}
    if f"{arch}.rope.freq_base" in kv:
        rope["rope_theta"] = kv[f"{arch}.rope.freq_base"]
    secs = kv.get(f"{arch}.rope.dimension_sections")
    if secs is not None:
        rope["mrope_section"] = [int(x) for x in secs if int(x) != 0]
        rope["mrope_interleaved"] = True
    if f"{arch}.rope.dimension_count" in kv and "head_dim" in text:
        # partial_rotary_factor = rope dims / head dim, the fraction of each head that rotates
        text["partial_rotary_factor"] = float(kv[f"{arch}.rope.dimension_count"]) / float(text["head_dim"])
    if rope:
        rope["rope_type"] = "default"
        text["rope_parameters"] = rope

    # layer_types, derived from full_attention_interval exactly as the sibling arch does.
    n = text.get("num_hidden_layers")
    iv = text.get("full_attention_interval")
    if n and iv:
        text["layer_types"] = ["full_attention" if (i + 1) % int(iv) == 0 else "linear_attention"
                               for i in range(int(n))]

    for k_kv, k_cfg in (("tokenizer.ggml.bos_token_id", "bos_token_id"),
                        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
                        ("tokenizer.ggml.padding_token_id", "pad_token_id")):
        if k_kv in kv:
            text[k_cfg] = int(kv[k_kv])

    # QSA indexer. THE FIELD NAMES ARE indexer_*, NOT index_*: _QSA_CONFIG_FIELDS
    # (qwen4_exp/config.py:14-20) is ("indexer_n_heads", "indexer_kv_heads",
    # "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"), and model.py:264 turns
    # the indexer ON with `getattr(config, "indexer_n_heads", None) is not None`. Emitting the
    # index_* spelling is not a typo that raises -- _validate_qsa_config sees all five as None,
    # returns happy, the layer is built as a plain Qwen3NextAttention with NO indexer, and the
    # checkpoint's 24 indexer tensors become orphans. Every full-attention layer would then run
    # dense attention over a model trained for sparse: it loads and it is wrong.
    for suffix, field in (("attention.indexer.head_count", "indexer_n_heads"),
                          ("attention.indexer.key_length", "indexer_head_dim"),
                          ("attention.indexer.top_k", "indexer_budget")):
        key = f"{arch}.{suffix}"
        if key in kv:
            text[field] = int(kv[key])
    if "indexer_n_heads" in text:
        # Not KVs, and both are checked by the fork rather than trusted:
        #  * indexer_kv_heads: the fork REQUIRES 1 ("the QSA MQA operators require
        #    indexer_kv_heads=1", config.py:146) and the file agrees -- indexer.k_proj is 128
        #    rows = 1 x indexer_head_dim, against indexer.q_proj's 512 = 4 x 128. The 640-row
        #    index_qk_proj the converter concatenates is (4 + 1) * 128.
        #  * indexer_compress_ratio: ggml carries it per layer as attention.compress_ratios,
        #    0 on the GDN layers and the same value on every full-attention layer. Taking the
        #    unique non-zero rather than element 0 makes a per-layer-varying file fail loudly.
        text["indexer_kv_heads"] = 1
        ratios = kv.get(f"{arch}.attention.compress_ratios")
        if ratios is not None:
            nz = sorted({int(x) for x in ratios if int(x)})
            if len(nz) != 1:
                raise ValueError(f"{arch}.attention.compress_ratios holds {nz} distinct "
                                 f"non-zero values; the fork has ONE indexer_compress_ratio")
            text["indexer_compress_ratio"] = nz[0]
        else:
            missing.append(f"{arch}.attention.compress_ratios")

    text["_missing_kvs"] = missing          # stripped by emit(); kept for the gate to report
    return text


def emit(kv: dict[str, Any], ple_fp8_offload: bool = False,
         ple_ggml_layer: int | None = None) -> tuple[dict[str, Any], list[str]]:
    """(config.json dict, list of KVs that were expected and absent).

    ple_fp8_offload=True adds the two RUNTIME fields the fork's PLE layer reads (ple_layer.py:315-336,
    2026-09-06): text_config.ple_embedding_dtype = "float8_e4m3fn" selects Qwen4ExpPLEFp8EmbeddingMethod
    (force_fp8_storage) for the q8_0->FP8 table gguf_to_vllm.ple_fp8 writes, and
    text_config.ple_offload_embedding = true pins that table in host memory (the fork auto-offloads only on
    sm_70; the P100s are sm_60, so without the explicit flag the 51 GB table would be placed on the cards).
    Neither is derived from the GGUF: they describe the checkpoint we WROTE, so they are set here and only
    when asked for."""
    text = build(kv, ple_ggml_layer)
    missing = text.pop("_missing_kvs")
    if ple_fp8_offload:
        text["ple_embedding_dtype"] = "float8_e4m3fn"
        text["ple_offload_embedding"] = True
    # Qwen4ExpForCausalLM, not ...ForConditionalGeneration. Both are registered
    # (registry.py:120 and :574) and both read the same text_config, but the conditional
    # generation class is the vision-tower variant and this GGUF carries no vision tensors and
    # no vision_config -- there is nothing for it to build a Qwen4ExpVisionConfig from. The
    # text-only class is what the checkpoint actually is.
    cfg = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp",
        "dtype": "float16",
        "tie_word_embeddings": False,
        "text_config": text,
    }
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if k in text:
            cfg[k] = text[k]
    for suffix, field in (("ple.image_token_id", "image_token_id"),):
        key = f"{kv.get('general.architecture')}.{suffix}"
        if key in kv:
            cfg[field] = int(kv[key])
    return cfg, missing
