#!/usr/bin/env python3
"""Tiny random-weight Hugging Face checkpoints of the vision-language models the converter handles (converter-vl-archs, FR#16):

    qwen25vl    Qwen2_5_VLForConditionalGeneration   (transformers < 4.52 layout: model.layers.*, visual.*)
    qwen3vl     Qwen3VLForConditionalGeneration      (model.language_model.*, model.visual.*, deepstack mergers)
    qwen3vlmoe  Qwen3VLMoeForConditionalGeneration   (packed, transposed experts: gate_up_proj / down_proj)
    smolvlm     SmolVLMForConditionalGeneration      (Idefics3: model.text_model.*, model.vision_model.*, model.connector.*)
    internvl    InternVLChatModel                    (language_model.* = Qwen2, vision_model.*, mlp1.*)

Every tensor-name pattern of the real checkpoints is present, at toy sizes, so the converter's name mapping, the fused-QKV and
Conv3D splits and the expert repack are all exercised on a CPU in seconds. The tokenizer is the one of
tests/fixtures/glm4-moe-lite-tiny (a real BPE tokenizer the converter knows); the vocabulary size follows it.

Usage:  python3 tests/gen_tiny_vl.py OUT_DIR [kind ...]      (default: all kinds; OUT_DIR/<kind>/ each)
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.path.join(HERE, "fixtures", "glm4-moe-lite-tiny")

H, NH, NKV, HD, FF, NL = 64, 4, 2, 16, 128, 2          # text: hidden, heads, kv heads, head dim, ffn, layers
VH, VNH, VFF, VD = 32, 2, 64, 4                          # vision: hidden, heads, ffn, depth (>= 4: mlp1.3 / merger.mlp.2 are block-indexed)
E, EFF = 4, 32                                           # experts, expert ffn (qwen3vlmoe)


def _vocab():
    with open(os.path.join(TOK_DIR, "config.json")) as f:
        return json.load(f)["vocab_size"]


def _save_safetensors(path, tensors):
    """safetensors without the safetensors package: header JSON + raw little-endian data (F32 only)."""
    hdr, off, blobs = {}, 0, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = a.tobytes()
        hdr[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in blobs:
            f.write(b)


def _text_llama_like(rng, prefix, vocab, qk_norm=False, bias=False, moe=False):
    t = {prefix + "embed_tokens.weight": rng.standard_normal((vocab, H)) * 0.02, prefix + "norm.weight": np.ones(H)}
    for i in range(NL):
        p = "%slayers.%d." % (prefix, i)
        t[p + "input_layernorm.weight"] = np.ones(H)
        t[p + "post_attention_layernorm.weight"] = np.ones(H)
        t[p + "self_attn.q_proj.weight"] = rng.standard_normal((NH * HD, H)) * 0.02
        t[p + "self_attn.k_proj.weight"] = rng.standard_normal((NKV * HD, H)) * 0.02
        t[p + "self_attn.v_proj.weight"] = rng.standard_normal((NKV * HD, H)) * 0.02
        t[p + "self_attn.o_proj.weight"] = rng.standard_normal((H, NH * HD)) * 0.02
        if bias:
            for x, n in (("q", NH), ("k", NKV), ("v", NKV)):
                t[p + "self_attn.%s_proj.bias" % x] = rng.standard_normal(n * HD) * 0.02
        if qk_norm:
            t[p + "self_attn.q_norm.weight"] = np.ones(HD)
            t[p + "self_attn.k_norm.weight"] = np.ones(HD)
        if moe:
            t[p + "mlp.gate.weight"] = rng.standard_normal((E, H)) * 0.02
            t[p + "mlp.experts.gate_up_proj"] = rng.standard_normal((E, H, 2 * EFF)) * 0.02
            t[p + "mlp.experts.down_proj"] = rng.standard_normal((E, EFF, H)) * 0.02
        else:
            t[p + "mlp.gate_proj.weight"] = rng.standard_normal((FF, H)) * 0.02
            t[p + "mlp.up_proj.weight"] = rng.standard_normal((FF, H)) * 0.02
            t[p + "mlp.down_proj.weight"] = rng.standard_normal((H, FF)) * 0.02
    return t


def _text_cfg(vocab, **kw):
    c = {"hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": FF,
         "num_hidden_layers": NL, "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "max_position_embeddings": 4096,
         "tie_word_embeddings": False}
    c.update(kw)
    return c


def _qwen_vision(rng, p, qwen3):
    t = {p + "patch_embed.proj.weight": rng.standard_normal((VH, 3, 2, 16 if qwen3 else 14, 16 if qwen3 else 14)) * 0.02}
    if qwen3:
        t[p + "patch_embed.proj.bias"] = rng.standard_normal(VH) * 0.02
        t[p + "pos_embed.weight"] = rng.standard_normal((16, VH)) * 0.02
    for i in range(VD):
        b = "%sblocks.%d." % (p, i)
        t[b + "norm1.weight"] = np.ones(VH)
        t[b + "norm2.weight"] = np.ones(VH)
        if qwen3:
            t[b + "norm1.bias"] = np.zeros(VH)
            t[b + "norm2.bias"] = np.zeros(VH)
        t[b + "attn.qkv.weight"] = rng.standard_normal((3 * VH, VH)) * 0.02
        t[b + "attn.qkv.bias"] = rng.standard_normal(3 * VH) * 0.02
        t[b + "attn.proj.weight"] = rng.standard_normal((VH, VH)) * 0.02
        t[b + "attn.proj.bias"] = np.zeros(VH)
        if qwen3:
            t[b + "mlp.linear_fc1.weight"] = rng.standard_normal((VFF, VH)) * 0.02
            t[b + "mlp.linear_fc1.bias"] = np.zeros(VFF)
            t[b + "mlp.linear_fc2.weight"] = rng.standard_normal((VH, VFF)) * 0.02
            t[b + "mlp.linear_fc2.bias"] = np.zeros(VH)
        else:
            for n, shp in (("gate_proj", (VFF, VH)), ("up_proj", (VFF, VH)), ("down_proj", (VH, VFF))):
                t[b + "mlp.%s.weight" % n] = rng.standard_normal(shp) * 0.02
                t[b + "mlp.%s.bias" % n] = np.zeros(shp[0])
    m = VH * 4
    if qwen3:
        t[p + "merger.norm.weight"] = np.ones(VH)
        t[p + "merger.norm.bias"] = np.zeros(VH)
        t[p + "merger.linear_fc1.weight"] = rng.standard_normal((m, m)) * 0.02
        t[p + "merger.linear_fc1.bias"] = np.zeros(m)
        t[p + "merger.linear_fc2.weight"] = rng.standard_normal((H, m)) * 0.02
        t[p + "merger.linear_fc2.bias"] = np.zeros(H)
        d = p + "deepstack_merger_list.0."
        t[d + "norm.weight"] = np.ones(m)
        t[d + "norm.bias"] = np.zeros(m)
        t[d + "linear_fc1.weight"] = rng.standard_normal((m, m)) * 0.02
        t[d + "linear_fc1.bias"] = np.zeros(m)
        t[d + "linear_fc2.weight"] = rng.standard_normal((H, m)) * 0.02
        t[d + "linear_fc2.bias"] = np.zeros(H)
    else:
        t[p + "merger.ln_q.weight"] = np.ones(VH)
        t[p + "merger.mlp.0.weight"] = rng.standard_normal((m, m)) * 0.02
        t[p + "merger.mlp.0.bias"] = np.zeros(m)
        t[p + "merger.mlp.2.weight"] = rng.standard_normal((H, m)) * 0.02
        t[p + "merger.mlp.2.bias"] = np.zeros(H)
    return t


def _siglip(rng, p, patch=16, n_pos=16, cls=False, intern=False):
    t = {p + "embeddings.patch_embedding.weight": rng.standard_normal((VH, 3, patch, patch)) * 0.02,
         p + "embeddings.patch_embedding.bias": np.zeros(VH)}
    if intern:
        t[p + "embeddings.class_embedding"] = rng.standard_normal((1, 1, VH)) * 0.02
        t[p + "embeddings.position_embedding"] = rng.standard_normal((1, n_pos + 1, VH)) * 0.02
    else:
        t[p + "embeddings.position_embedding.weight"] = rng.standard_normal((n_pos, VH)) * 0.02
        t[p + "post_layernorm.weight"] = np.ones(VH)
        t[p + "post_layernorm.bias"] = np.zeros(VH)
    for i in range(VD):
        b = "%sencoder.layers.%d." % (p, i)
        if intern:
            t[b + "attn.qkv.weight"] = rng.standard_normal((3 * VH, VH)) * 0.02
            t[b + "attn.qkv.bias"] = rng.standard_normal(3 * VH) * 0.02
            t[b + "attn.proj.weight"] = rng.standard_normal((VH, VH)) * 0.02
            t[b + "attn.proj.bias"] = np.zeros(VH)
            for n in ("norm1", "norm2"):
                t[b + n + ".weight"] = np.ones(VH)
                t[b + n + ".bias"] = np.zeros(VH)
            t[b + "ls1"] = np.full(VH, 0.1)
            t[b + "ls2"] = np.full(VH, 0.1)
        else:
            for x in ("q", "k", "v", "out"):
                t[b + "self_attn.%s_proj.weight" % x] = rng.standard_normal((VH, VH)) * 0.02
                t[b + "self_attn.%s_proj.bias" % x] = np.zeros(VH)
            for n in ("layer_norm1", "layer_norm2"):
                t[b + n + ".weight"] = np.ones(VH)
                t[b + n + ".bias"] = np.zeros(VH)
        t[b + "mlp.fc1.weight"] = rng.standard_normal((VFF, VH)) * 0.02
        t[b + "mlp.fc1.bias"] = np.zeros(VFF)
        t[b + "mlp.fc2.weight"] = rng.standard_normal((VH, VFF)) * 0.02
        t[b + "mlp.fc2.bias"] = np.zeros(VH)
    return t


def build(kind, out, seed=1234):
    rng = np.random.default_rng(seed)
    vocab = _vocab()
    pre = {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
    if kind == "qwen25vl":
        cfg = dict(_text_cfg(vocab, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}), architectures=["Qwen2_5_VLForConditionalGeneration"],
                   model_type="qwen2_5_vl", vision_config={"depth": VD, "hidden_size": VH, "intermediate_size": VFF, "num_heads": VNH,
                                                            "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                                                            "fullatt_block_indexes": [1, 3], "window_size": 112, "out_hidden_size": H})
        del cfg["head_dim"]
        cfg["hidden_size"] = H
        t = _text_llama_like(rng, "model.", vocab, bias=True)
        t["lm_head.weight"] = rng.standard_normal((vocab, H)) * 0.02
        t.update(_qwen_vision(rng, "visual.", qwen3=False))
    elif kind in ("qwen3vl", "qwen3vlmoe"):
        moe = kind == "qwen3vlmoe"
        tc = _text_cfg(vocab, rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 3, 3], "rope_type": "default"})
        if moe:
            tc.update(num_experts=E, num_experts_per_tok=2, moe_intermediate_size=EFF, decoder_sparse_step=1, norm_topk_prob=True)
        cfg = {"architectures": ["Qwen3VLMoeForConditionalGeneration" if moe else "Qwen3VLForConditionalGeneration"],
               "model_type": "qwen3_vl_moe" if moe else "qwen3_vl", "text_config": tc, "tie_word_embeddings": False,
               "vision_config": {"depth": VD, "hidden_size": VH, "intermediate_size": VFF, "num_heads": VNH, "patch_size": 16,
                                 "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 16, "out_hidden_size": H,
                                 "deepstack_visual_indexes": [0]}}
        t = _text_llama_like(rng, "model.language_model.", vocab, qk_norm=True, moe=moe)
        t["lm_head.weight"] = rng.standard_normal((vocab, H)) * 0.02
        t.update(_qwen_vision(rng, "model.visual.", qwen3=True))
    elif kind == "smolvlm":
        cfg = {"architectures": ["SmolVLMForConditionalGeneration"], "model_type": "smolvlm", "scale_factor": 2,
               "text_config": dict(_text_cfg(vocab), architectures=["VLlama3ForCausalLM"], model_type="llama"),
               "vision_config": {"model_type": "smolvlm_vision", "hidden_size": VH, "intermediate_size": VFF, "num_attention_heads": VNH,
                                 "num_hidden_layers": VD, "patch_size": 16, "image_size": 64, "layer_norm_eps": 1e-6}}
        del cfg["text_config"]["head_dim"]
        cfg["text_config"]["num_attention_heads"] = NH
        t = _text_llama_like(rng, "model.text_model.", vocab)
        t["lm_head.weight"] = rng.standard_normal((vocab, H)) * 0.02
        t.update(_siglip(rng, "model.vision_model."))
        t["model.connector.modality_projection.proj.weight"] = rng.standard_normal((H, VH * 4)) * 0.02
        pre["size"] = {"longest_edge": 128}
    elif kind == "internvl":
        lc = dict(_text_cfg(vocab), architectures=["Qwen2ForCausalLM"], model_type="qwen2")
        del lc["head_dim"]
        cfg = {"architectures": ["InternVLChatModel"], "model_type": "internvl_chat", "downsample_ratio": 0.5,
               "min_dynamic_patch": 1, "max_dynamic_patch": 6, "llm_config": lc,
               "vision_config": {"architectures": ["InternVisionModel"], "hidden_size": VH, "intermediate_size": VFF,
                                 "num_attention_heads": VNH, "num_hidden_layers": VD, "patch_size": 16, "image_size": 64,
                                 "layer_norm_eps": 1e-6, "hidden_act": "gelu", "qk_normalization": False}}
        t = _text_llama_like(rng, "language_model.model.", vocab, bias=True)
        t["language_model.lm_head.weight"] = rng.standard_normal((vocab, H)) * 0.02
        t.update(_siglip(rng, "vision_model.", cls=True, intern=True))
        m = VH * 4
        t["mlp1.0.weight"] = np.ones(m)
        t["mlp1.0.bias"] = np.zeros(m)
        t["mlp1.1.weight"] = rng.standard_normal((H, m)) * 0.02
        t["mlp1.1.bias"] = np.zeros(H)
        t["mlp1.3.weight"] = rng.standard_normal((H, H)) * 0.02
        t["mlp1.3.bias"] = np.zeros(H)
    else:
        raise SystemExit("unknown kind %r" % kind)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "preprocessor_config.json"), "w") as f:
        json.dump(pre, f)
    for n in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(TOK_DIR, n), os.path.join(out, n))
    _save_safetensors(os.path.join(out, "model.safetensors"), t)
    return {k: np.asarray(v, dtype=np.float32) for k, v in t.items()}


KINDS = ("qwen25vl", "qwen3vl", "qwen3vlmoe", "smolvlm", "internvl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for k in sys.argv[2:] or KINDS:
        build(k, os.path.join(sys.argv[1], k))
        print("wrote", os.path.join(sys.argv[1], k))
