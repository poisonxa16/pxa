#!/usr/bin/env python3
"""Assert a converted glm4-moe-lite-tiny GGUF is exactly what it should be.

Usage: verify_fixture.py path/to/out.gguf

Checks (all must hold, or this exits non-zero with a specific message):
  - general.architecture == deepseek2
  - deepseek2.block_count == 2
  - the tensor-name SET is EXACTLY the golden 36 -- no more, no less. A
    converter that emits an unexpected/missing tensor name FAILS here
    (refusal over silent garbage is the point of this fixture).
  - nothing from layer 2 (the NextN/MTP tail) made it into the GGUF.
  - KV metadata: expert_gating_func==SIGMOID(2), expert_weights_scale==1.8,
    expert_weights_norm==true, leading_dense_block_count==1, q_lora_rank,
    kv_lora_rank, key_length, value_length, rope.dimension_count all match
    the shrunk fixture config.
  - tokenizer.ggml.pre == 'glm4', bos == 154822, eot == 154827.
"""
import json
import os
import sys

import gguf

# The exact tensor-name set a correct conversion of the fixture must produce.
# Derived from the real converter's own output against this fixture (blk.0 =
# dense FFN + full MLA attn, blk.1 = MoE FFN + full MLA attn, both kept;
# blk.2 = NextN tail, entirely absent). Any diff here is a converter bug.
GOLDEN_TENSORS = {
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
} | {
    f"blk.0.{t}"
    for t in [
        "attn_norm.weight",
        "ffn_down.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_norm.weight",
        "attn_kv_a_norm.weight",
        "attn_kv_a_mqa.weight",
        "attn_kv_b.weight",
        "attn_k_b.weight",
        "attn_v_b.weight",
        "attn_output.weight",
        "attn_q_a_norm.weight",
        "attn_q_a.weight",
        "attn_q_b.weight",
    ]
} | {
    f"blk.1.{t}"
    for t in [
        "attn_norm.weight",
        "ffn_down_exps.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "exp_probs_b.bias",
        "ffn_gate_inp.weight",
        "ffn_down_shexp.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_norm.weight",
        "attn_kv_a_norm.weight",
        "attn_kv_a_mqa.weight",
        "attn_kv_b.weight",
        "attn_k_b.weight",
        "attn_v_b.weight",
        "attn_output.weight",
        "attn_q_a_norm.weight",
        "attn_q_a.weight",
        "attn_q_b.weight",
    ]
}
assert len(GOLDEN_TENSORS) == 36, len(GOLDEN_TENSORS)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def kv(reader, key):
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    val = f.parts[f.data[0]]
    # GGUFReader hands back a 1-elem numpy array for scalars; unwrap it.
    try:
        return val[0] if hasattr(val, "__len__") and len(val) == 1 else val
    except TypeError:
        return val


def kv_str(reader, key):
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    parts = [f.parts[i] for i in f.data]
    return b"".join(bytes(p) for p in parts).decode("utf-8")


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <gguf-path> <fixture-config.json>", file=sys.stderr)
        return 2
    gguf_path, cfg_path = sys.argv[1], sys.argv[2]

    with open(cfg_path) as f:
        cfg = json.load(f)

    r = gguf.GGUFReader(gguf_path)

    arch = kv_str(r, "general.architecture")
    if arch != "deepseek2":
        fail(f"general.architecture == {arch!r}, expected 'deepseek2'")

    block_count = int(kv(r, "deepseek2.block_count"))
    if block_count != cfg["num_hidden_layers"]:
        fail(f"deepseek2.block_count == {block_count}, expected {cfg['num_hidden_layers']}")

    # --- exact tensor-name set match ---
    got_tensors = {t.name for t in r.tensors}
    missing = GOLDEN_TENSORS - got_tensors
    extra = got_tensors - GOLDEN_TENSORS
    if missing:
        fail(f"missing {len(missing)} expected tensor(s): {sorted(missing)}")
    if extra:
        fail(f"got {len(extra)} UNEXPECTED tensor(s) (silent garbage / NextN leak?): {sorted(extra)}")

    # --- NextN leak check (redundant with the set-equality above, kept
    #     explicit because "nothing from layer 2" is a named requirement) ---
    leaked = [n for n in got_tensors if n.startswith("blk.2.")]
    if leaked:
        fail(f"NextN/MTP layer 2 tensors leaked into the GGUF: {sorted(leaked)}")

    # --- gating / MoE KVs ---
    checks = [
        ("deepseek2.expert_gating_func", int(gguf.ExpertGatingFuncType.SIGMOID), "expert_gating_func"),
        ("deepseek2.expert_weights_scale", cfg["routed_scaling_factor"], "expert_weights_scale"),
        ("deepseek2.expert_weights_norm", cfg["norm_topk_prob"], "expert_weights_norm"),
        ("deepseek2.leading_dense_block_count", cfg["first_k_dense_replace"], "leading_dense_block_count"),
        ("deepseek2.attention.q_lora_rank", cfg["q_lora_rank"], "q_lora_rank"),
        ("deepseek2.attention.kv_lora_rank", cfg["kv_lora_rank"], "kv_lora_rank"),
        ("deepseek2.attention.key_length", cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"], "key_length"),
        ("deepseek2.attention.value_length", cfg["v_head_dim"], "value_length"),
        ("deepseek2.rope.dimension_count", cfg["qk_rope_head_dim"], "rope.dimension_count"),
    ]
    for key, expected, label in checks:
        got = kv(r, key)
        if got is None:
            fail(f"{label} ({key}) missing from GGUF")
        mismatch = (bool(got) != bool(expected)) if isinstance(expected, bool) else (got != expected)
        if mismatch:
            fail(f"{label} == {got!r}, expected {expected!r}")

    # --- tokenizer ---
    pre = kv_str(r, "tokenizer.ggml.pre")
    if pre != "glm4":
        fail(f"tokenizer.ggml.pre == {pre!r}, expected 'glm4'")
    bos = int(kv(r, "tokenizer.ggml.bos_token_id"))
    if bos != 154822:
        fail(f"bos_token_id == {bos}, expected 154822 ([gMASK])")
    eot = int(kv(r, "tokenizer.ggml.eot_token_id"))
    if eot != 154827:
        fail(f"eot_token_id == {eot}, expected 154827 (<|user|>)")

    print(f"PASS: {len(got_tensors)} tensors match golden set exactly; "
          f"arch=deepseek2 block_count={block_count} gating=sigmoid "
          f"scale={cfg['routed_scaling_factor']} pre=glm4 bos={bos} eot={eot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
