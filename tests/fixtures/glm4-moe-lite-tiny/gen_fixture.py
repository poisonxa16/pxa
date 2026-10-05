#!/usr/bin/env python3
"""Generate random-weight bf16 safetensors for the glm4-moe-lite-tiny fixture.

Writes one tensor for every tensor-name pattern present in the real
zai-org/GLM-4.7-Flash model.safetensors.index.json (dense layer-0 FFN,
MoE layers, MLA attention, NextN/MTP tail), with shapes derived from the
shrunk config.json sitting next to this script. The point of the fixture
is to exercise the converter's tensor-name/shape handling, NOT to be a
usable model — values are random.

Layer layout produced (config: num_hidden_layers=2, first_k_dense_replace=1,
num_nextn_predict_layers=1):
  blk 0  -> dense FFN + full MLA attention   (kept by the converter)
  blk 1  -> MoE FFN + full MLA attention     (kept by the converter)
  blk 2  -> NextN/MTP tail (eh_proj/embed_tokens/enorm/hnorm/shared_head.*
            + a full attn/MoE block, same shape family as blk 1)  -- MUST
            be dropped by the converter (index >= num_hidden_layers).
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def bf16(*shape):
    return torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)


def f32(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default=HERE,
        help="directory holding config.json to read, and to write "
        "model.safetensors into (default: this script's own directory). "
        "Point this at a scratch copy of the fixture dir so test runs "
        "never leave a generated safetensors blob in the git-tracked fixture.",
    )
    args = ap.parse_args()

    with open(os.path.join(args.dir, "config.json")) as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    MI = cfg["moe_intermediate_size"]
    Q = cfg["q_lora_rank"]
    KVR = cfg["kv_lora_rank"]
    NOPE = cfg["qk_nope_head_dim"]
    ROPE = cfg["qk_rope_head_dim"]
    VDIM = cfg["v_head_dim"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    E = cfg["n_routed_experts"]
    VOCAB = cfg["vocab_size"]
    N_LAYERS = cfg["num_hidden_layers"]
    N_DENSE = cfg["first_k_dense_replace"]
    N_NEXTN = cfg["num_nextn_predict_layers"]

    assert "scoring_func" not in cfg, (
        "fixture config.json must OMIT scoring_func (that omission is exactly "
        "what this fixture proves survivable via the noaux_tc/sigmoid default)"
    )

    tensors: dict[str, torch.Tensor] = {}

    # --- top-level ---
    tensors["model.embed_tokens.weight"] = bf16(VOCAB, H)
    tensors["lm_head.weight"] = bf16(VOCAB, H)  # tie_word_embeddings=false
    tensors["model.norm.weight"] = bf16(H)

    def add_attn(prefix):
        # MLA attention, identical shape family on every layer (dense or MoE)
        tensors[f"{prefix}.self_attn.q_a_proj.weight"] = bf16(Q, H)
        tensors[f"{prefix}.self_attn.q_a_layernorm.weight"] = bf16(Q)
        tensors[f"{prefix}.self_attn.q_b_proj.weight"] = bf16(NH * (NOPE + ROPE), Q)
        tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = bf16(KVR + ROPE, H)
        tensors[f"{prefix}.self_attn.kv_a_layernorm.weight"] = bf16(KVR)
        tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = bf16(NKV * (NOPE + VDIM), KVR)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = bf16(H, NH * VDIM)
        tensors[f"{prefix}.input_layernorm.weight"] = bf16(H)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = bf16(H)

    def add_dense_ffn(prefix):
        tensors[f"{prefix}.mlp.gate_proj.weight"] = bf16(I, H)
        tensors[f"{prefix}.mlp.up_proj.weight"] = bf16(I, H)
        tensors[f"{prefix}.mlp.down_proj.weight"] = bf16(H, I)

    def add_moe_ffn(prefix):
        tensors[f"{prefix}.mlp.gate.weight"] = bf16(E, H)
        tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = f32(E)
        for xid in range(E):
            tensors[f"{prefix}.mlp.experts.{xid}.gate_proj.weight"] = bf16(MI, H)
            tensors[f"{prefix}.mlp.experts.{xid}.up_proj.weight"] = bf16(MI, H)
            tensors[f"{prefix}.mlp.experts.{xid}.down_proj.weight"] = bf16(H, MI)
        tensors[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = bf16(MI, H)
        tensors[f"{prefix}.mlp.shared_experts.up_proj.weight"] = bf16(MI, H)
        tensors[f"{prefix}.mlp.shared_experts.down_proj.weight"] = bf16(H, MI)

    # --- main stack: blk 0..N_LAYERS-1 ---
    for i in range(N_LAYERS):
        prefix = f"model.layers.{i}"
        add_attn(prefix)
        if i < N_DENSE:
            add_dense_ffn(prefix)
        else:
            add_moe_ffn(prefix)

    # --- NextN / MTP tail: blk N_LAYERS .. N_LAYERS+N_NEXTN-1 ---
    # Real GLM-4.7-Flash layer 47 (index == num_hidden_layers) carries a FULL
    # attn+MoE block (same shape family as a normal MoE layer, since 47 >=
    # first_k_dense_replace) PLUS the NextN-only tensors below. The converter
    # MUST drop the entire block (index >= num_hidden_layers) -- that's the
    # thing this fixture is built to prove.
    for j in range(N_NEXTN):
        idx = N_LAYERS + j
        prefix = f"model.layers.{idx}"
        tensors[f"{prefix}.embed_tokens.weight"] = bf16(VOCAB, H)
        tensors[f"{prefix}.enorm.weight"] = bf16(H)
        tensors[f"{prefix}.hnorm.weight"] = bf16(H)
        tensors[f"{prefix}.eh_proj.weight"] = bf16(H, 2 * H)
        tensors[f"{prefix}.shared_head.norm.weight"] = bf16(H)
        tensors[f"{prefix}.shared_head.head.weight"] = bf16(VOCAB, H)
        add_attn(prefix)
        add_moe_ffn(prefix)

    out_path = os.path.join(args.dir, "model.safetensors")
    save_file(tensors, out_path, metadata={"format": "pt"})
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total_bytes / 1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
