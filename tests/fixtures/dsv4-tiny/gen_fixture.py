#!/usr/bin/env python3
"""Generate a tiny synthetic DeepSeek-V4 checkpoint for the conversion test.

Reproduces the REAL 0731 checkpoint layout exactly (verified against
`model.safetensors.index.json` of DeepSeek-V4-Flash), shrunk:

  * non-transformers tensor naming (`layers.N.attn.wq_a.weight`, `embed.weight`,
    `hc_head_fn`, ...)
  * fp8 e4m3 dense weights with a sidecar `<name>.scale` in **ue8m0**, one scale
    per 128x128 block  (NOT V3's `_scale_inv`)
  * routed experts as MXFP4: int8 nibble pairs + one ue8m0 byte per 32 values
  * `layers.N.ffn.gate.tid2eid` as I64 [vocab, n_experts_per_tok], only on the
    first `num_hash_layers` blocks
  * per-layer conditional tensors driven by `compress_ratios[il]`
    (0 = raw SWA, 4 = CSA + lightning indexer, 128 = hyper-compressed)
  * an `mtp.0.*` block that the converter MUST drop

It also writes `reference.npz` holding INDEPENDENTLY computed f32 expectations
for one fp8 tensor and one routed-expert stack, so verify_fixture.py can prove
the dequant math (nibble order, scale exponent, block expansion) rather than
just the tensor-name set. Values are random -- this is not a usable model.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))

# e2m1 code points in ggml nibble order
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

rng = np.random.default_rng(20260731)


def bf16(*shape):
    return torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)


def f32(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def fp8_pair(rows, cols):
    """fp8 e4m3 weight + ue8m0 128x128 block scale, plus the f32 reference."""
    w = torch.randn(rows, cols, dtype=torch.float32).to(torch.float8_e4m3fn)
    sr, sc = (rows + 127) // 128, (cols + 127) // 128
    # keep the exponents mild so the reference stays in a comfortable range
    sbytes = rng.integers(120, 135, size=(sr, sc), dtype=np.uint8)
    s = torch.from_numpy(sbytes).view(torch.float8_e8m0fnu)

    # reference, computed here and NOT by the converter's code path
    scale_f = np.exp2(sbytes.astype(np.float32) - 127.0)
    scale_f = np.repeat(scale_f, 128, axis=0)[:rows]
    scale_f = np.repeat(scale_f, 128, axis=1)[:, :cols]
    ref = w.float().numpy() * scale_f
    return w, s, ref


def mxfp4_expert(rows, logical_cols):
    """MXFP4 nibble-packed int8 weight + ue8m0 per-32 scale, plus the f32 reference."""
    assert logical_cols % 32 == 0
    n_blocks = logical_cols // 32
    vals = rng.integers(0, 16, size=(rows, logical_cols), dtype=np.uint8)
    # safetensors packs adjacent values as the low/high nibble of one byte
    packed = (vals[:, 0::2] | (vals[:, 1::2] << 4)).astype(np.uint8)
    sbytes = rng.integers(120, 135, size=(rows, n_blocks), dtype=np.uint8)

    ref = E2M1[vals] * np.repeat(np.exp2(sbytes.astype(np.float32) - 127.0), 32, axis=1)

    w = torch.from_numpy(packed).view(torch.int8)
    s = torch.from_numpy(sbytes).view(torch.float8_e8m0fnu)
    return w, s, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE,
                    help="directory holding config.json to read and to write "
                         "model.safetensors / reference.npz into (point this at a "
                         "scratch copy so the git-tracked fixture stays binary-free)")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "config.json")) as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    HD = cfg["head_dim"]
    NH = cfg["num_attention_heads"]
    Q = cfg["q_lora_rank"]
    OG = cfg["o_groups"]
    OR = cfg["o_lora_rank"]
    MI = cfg["moe_intermediate_size"]
    E = cfg["n_routed_experts"]
    TOPK = cfg["num_experts_per_tok"]
    VOCAB = cfg["vocab_size"]
    NL = cfg["num_hidden_layers"]
    NHASH = cfg["num_hash_layers"]
    IH = cfg["index_n_heads"]
    ID = cfg["index_head_dim"]
    HC = cfg["hc_mult"]
    RATIOS = cfg["compress_ratios"]

    hc_dim = HC * H
    hc_mix = (2 + HC) * HC

    tensors: dict[str, torch.Tensor] = {}
    refs: dict[str, np.ndarray] = {}

    # ---- root ----
    tensors["embed.weight"] = bf16(VOCAB, H)
    tensors["head.weight"] = bf16(VOCAB, H)
    tensors["norm.weight"] = bf16(H)
    tensors["hc_head_fn"] = f32(HC, hc_dim)
    tensors["hc_head_base"] = f32(HC)
    tensors["hc_head_scale"] = f32(1)          # <- the 1-element tensor

    def add_layer(prefix: str, il: int, ratio: int, hashed: bool, want_refs: bool):
        tensors[f"{prefix}.attn_norm.weight"] = bf16(H)
        tensors[f"{prefix}.ffn_norm.weight"] = bf16(H)
        for tag in ("attn", "ffn"):
            tensors[f"{prefix}.hc_{tag}_fn"] = f32(hc_mix, hc_dim)
            tensors[f"{prefix}.hc_{tag}_base"] = f32(hc_mix)
            tensors[f"{prefix}.hc_{tag}_scale"] = f32(3)

        tensors[f"{prefix}.attn.attn_sink"] = f32(NH)
        tensors[f"{prefix}.attn.q_norm.weight"] = bf16(Q)
        tensors[f"{prefix}.attn.kv_norm.weight"] = bf16(HD)

        for nm, (r, c) in {
            "wq_a": (Q, H),
            "wq_b": (NH * HD, Q),
            "wkv": (HD, H),
            "wo_a": (OG * OR, H),
            "wo_b": (H, OG * OR),
        }.items():
            w, s, ref = fp8_pair(r, c)
            tensors[f"{prefix}.attn.{nm}.weight"] = w
            tensors[f"{prefix}.attn.{nm}.scale"] = s
            if want_refs and nm == "wq_a":
                refs["fp8_wq_a"] = ref

        if ratio != 0:
            coff = 2 if ratio == 4 else 1
            tensors[f"{prefix}.attn.compressor.wkv.weight"] = bf16(coff * HD, H)
            tensors[f"{prefix}.attn.compressor.wgate.weight"] = bf16(coff * HD, H)
            tensors[f"{prefix}.attn.compressor.ape"] = f32(ratio, coff * HD)
            tensors[f"{prefix}.attn.compressor.norm.weight"] = bf16(HD)

        if ratio == 4:
            tensors[f"{prefix}.attn.indexer.weights_proj.weight"] = bf16(IH, H)
            w, s, _ = fp8_pair(IH * ID, Q)
            tensors[f"{prefix}.attn.indexer.wq_b.weight"] = w
            tensors[f"{prefix}.attn.indexer.wq_b.scale"] = s
            tensors[f"{prefix}.attn.indexer.compressor.wkv.weight"] = bf16(2 * ID, H)
            tensors[f"{prefix}.attn.indexer.compressor.wgate.weight"] = bf16(2 * ID, H)
            tensors[f"{prefix}.attn.indexer.compressor.ape"] = f32(4, 2 * ID)
            tensors[f"{prefix}.attn.indexer.compressor.norm.weight"] = bf16(ID)

        # routing
        tensors[f"{prefix}.ffn.gate.weight"] = bf16(E, H)
        if hashed:
            tid = rng.integers(0, E, size=(VOCAB, TOPK)).astype(np.int64)
            tensors[f"{prefix}.ffn.gate.tid2eid"] = torch.from_numpy(tid)
            if want_refs:
                refs["tid2eid"] = tid.astype(np.int32)
        else:
            tensors[f"{prefix}.ffn.gate.bias"] = f32(E)

        # shared expert (fp8)
        for nm, (r, c) in {"w1": (MI, H), "w2": (H, MI), "w3": (MI, H)}.items():
            w, s, _ = fp8_pair(r, c)
            tensors[f"{prefix}.ffn.shared_experts.{nm}.weight"] = w
            tensors[f"{prefix}.ffn.shared_experts.{nm}.scale"] = s

        # routed experts (MXFP4)
        for nm, (r, c) in {"w1": (MI, H), "w2": (H, MI), "w3": (MI, H)}.items():
            stack = np.empty((E, r, c), dtype=np.float32)
            for eid in range(E):
                w, s, ref = mxfp4_expert(r, c)
                tensors[f"{prefix}.ffn.experts.{eid}.{nm}.weight"] = w
                tensors[f"{prefix}.ffn.experts.{eid}.{nm}.scale"] = s
                stack[eid] = ref
            if want_refs and nm == "w1":
                refs["exp_gate"] = stack

    for il in range(NL):
        add_layer(f"layers.{il}", il, RATIOS[il], il < NHASH, want_refs=(il == 1))

    # ---- MTP tail: MUST be dropped by the converter ----
    tensors["mtp.0.attn_norm.weight"] = bf16(H)
    tensors["mtp.0.norm.weight"] = bf16(H)
    tensors["mtp.0.main_norm.weight"] = bf16(H)
    tensors["mtp.0.markov_head.markov_w1.weight"] = bf16(H, H)
    tensors["mtp.0.markov_head.markov_w2.weight"] = bf16(H, H)
    tensors["mtp.0.confidence_head.proj.weight"] = bf16(1, H)
    w, s, _ = fp8_pair(Q, H)
    tensors["mtp.0.attn.wq_a.weight"] = w
    tensors["mtp.0.attn.wq_a.scale"] = s

    out_path = os.path.join(args.dir, "model.safetensors")
    save_file(tensors, out_path, metadata={"format": "pt"})
    np.savez(os.path.join(args.dir, "reference.npz"), **refs)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1e6:.1f} MB -> {out_path}")
    print(f"wrote {len(refs)} reference array(s) -> reference.npz")


if __name__ == "__main__":
    sys.exit(main())
