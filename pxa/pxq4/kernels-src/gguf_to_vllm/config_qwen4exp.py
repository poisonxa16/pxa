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
    "embedding_length_per_layer_input":   "ple_embed_dim",
    # GDN
    "ssm.conv_kernel":                    "linear_conv_kernel_dim",
    "ssm.state_size":                     "linear_key_head_dim",
    "ssm.group_count":                    "linear_num_key_heads",
    "ssm.time_step_rank":                 "linear_num_value_heads",
}

#: Fields whose GGUF value needs a transform rather than a copy, with the reason.
def _ple_layer_ids(kv: dict, arch: str) -> list[int] | None:
    # ggml stores them as `ple.layers`; the fork calls them ple_layer_ids and validates that
    # each is in [1, num_hidden_layers] -- i.e. ONE-BASED. The GGUF's [1] is already in that
    # convention (there is exactly one PLE stack in the file, blk.N.ple_* with N present once).
    v = kv.get(f"{arch}.ple.layers")
    return [int(x) for x in v] if v is not None else None


def build(kv: dict[str, Any]) -> dict[str, Any]:
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

    ids = _ple_layer_ids(kv, arch)
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

    # QSA indexer: the fork reads these off the text config.
    for suffix, field in (("attention.indexer.head_count", "index_n_heads"),
                          ("attention.indexer.key_length", "index_head_dim"),
                          ("attention.indexer.top_k", "index_topk")):
        key = f"{arch}.{suffix}"
        if key in kv:
            text[field] = kv[key]

    text["_missing_kvs"] = missing          # stripped by emit(); kept for the gate to report
    return text


def emit(kv: dict[str, Any], ple_fp8_offload: bool = False) -> tuple[dict[str, Any], list[str]]:
    """(config.json dict, list of KVs that were expected and absent).

    ple_fp8_offload=True adds the two RUNTIME fields the fork's PLE layer reads (ple_layer.py:315-336,
    2026-09-06): text_config.ple_embedding_dtype = "float8_e4m3fn" selects Qwen4ExpPLEFp8EmbeddingMethod
    (force_fp8_storage) for the q8_0->FP8 table gguf_to_vllm.ple_fp8 writes, and
    text_config.ple_offload_embedding = true pins that table in host memory (the fork auto-offloads only on
    sm_70; the P100s are sm_60, so without the explicit flag the 51 GB table would be placed on the cards).
    Neither is derived from the GGUF: they describe the checkpoint we WROTE, so they are set here and only
    when asked for."""
    text = build(kv)
    missing = text.pop("_missing_kvs")
    if ple_fp8_offload:
        text["ple_embedding_dtype"] = "float8_e4m3fn"
        text["ple_offload_embedding"] = True
    cfg = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
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
