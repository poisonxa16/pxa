#!/usr/bin/env python3
"""Assert a converted dsv4-tiny GGUF is exactly what it should be.

Usage: verify_fixture.py path/to/out.gguf path/to/config.json path/to/reference.npz

Checks (all must hold, or this exits non-zero with a specific message):
  - general.architecture == deepseek4
  - the tensor-name SET is EXACTLY the golden set -- nothing missing, nothing
    extra, and no `mtp.*` leak.
  - the per-layer conditional tensors follow compress_ratios (0 -> no
    compressor, 4 -> compressor + indexer, 128 -> compressor only).
  - `blk.N.ffn_gate_tid2eid.weight` exists for the first num_hash_layers only,
    is I32, and matches the source table EXACTLY.
  - NUMERIC: the fp8 e4m3 x ue8m0-128x128 dequant of blk.1.attn_q_a.weight
    matches an independently computed reference (Q8_0 tolerance).
  - NUMERIC: the MXFP4 -> f32 unpack of blk.1.ffn_gate_exps.weight matches an
    independently computed reference EXACTLY. e2m1 code points scaled by powers
    of two are exactly representable in bf16, so a wrong nibble order or a wrong
    scale exponent cannot hide behind a tolerance.
  - the routed experts are NOT stored as MXFP4 (that is the whole point of the
    fork's dequant path: a raw-MXFP4 expert makes `llama-quantize --outtype
    PXQ2` silently emit a 4.25-bpw file).
  - KV metadata: arch hparams, sqrtsoftplus gating, indexer, o-LoRA,
    compress_ratios (written verbatim, including the MTP tail), hyper-connection
    and hash-layer keys.
"""
import json
import os
import sys

import numpy as np

# use the repo's gguf-py, not whatever is installed site-wide (the DEEPSEEK4
# arch and ExpertGatingFuncType.SQRTSOFTPLUS only exist in this tree)
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "gguf-py"))
import gguf  # noqa: E402


def fail(msg: str):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def kv(reader, key):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def kv_array(reader, key):
    f = reader.get_field(key)
    if f is None:
        return None
    return [f.contents(i) for i in range(len(f.data))]


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u16 = raw.view(np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def q8_0_to_f32(raw: np.ndarray, k: int) -> np.ndarray:
    """raw is byte-shaped (..., k//32*34)."""
    nb = k // 32
    flat = raw.reshape(-1, nb, 34)
    d = flat[:, :, :2].copy().view(np.float16).astype(np.float32)      # (rows, nb, 1)
    qs = flat[:, :, 2:].view(np.int8).astype(np.float32)               # (rows, nb, 32)
    return (qs * d).reshape(*raw.shape[:-1], k)


def main():
    if len(sys.argv) != 4:
        fail("usage: verify_fixture.py out.gguf config.json reference.npz")
    gguf_path, cfg_path, ref_path = sys.argv[1:4]

    with open(cfg_path) as f:
        cfg = json.load(f)
    refs = np.load(ref_path)

    NL = cfg["num_hidden_layers"]
    NHASH = cfg["num_hash_layers"]
    RATIOS = cfg["compress_ratios"]

    r = gguf.GGUFReader(gguf_path, "r")

    arch = kv(r, "general.architecture")
    if isinstance(arch, (bytes, bytearray, memoryview)):
        arch = bytes(arch).decode()
    if arch != "deepseek4":
        fail(f"general.architecture == {arch!r}, expected 'deepseek4'")

    # ---------------- golden tensor-name set ----------------
    golden = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight",
    }
    for il in range(NL):
        golden |= {f"blk.{il}.{t}" for t in (
            "attn_norm.weight", "ffn_norm.weight",
            "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
            "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
            "attn_sinks.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
            "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
            "attn_output_a.weight", "attn_output_b.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight",
            "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight",
        )}
        if RATIOS[il] != 0:
            golden |= {f"blk.{il}.{t}" for t in (
                "attn_compressor_kv.weight", "attn_compressor_gate.weight",
                "attn_compressor_ape.weight", "attn_compressor_norm.weight",
            )}
        if RATIOS[il] == 4:
            golden |= {f"blk.{il}.{t}" for t in (
                "indexer.proj.weight", "indexer.attn_q_b.weight",
                "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
                "indexer_compressor_ape.weight", "indexer_compressor_norm.weight",
            )}
        if il < NHASH:
            golden.add(f"blk.{il}.ffn_gate_tid2eid.weight")
        else:
            golden.add(f"blk.{il}.exp_probs_b.bias")

    got = {t.name for t in r.tensors}
    missing = golden - got
    extra = got - golden
    if missing:
        fail(f"missing {len(missing)} expected tensor(s): {sorted(missing)}")
    if extra:
        fail(f"got {len(extra)} UNEXPECTED tensor(s) (mtp leak / silent garbage?): {sorted(extra)}")

    leaked = [n for n in got if n.startswith("mtp") or ".mtp." in n]
    if leaked:
        fail(f"MTP tensors leaked into the GGUF: {sorted(leaked)}")

    by_name = {t.name: t for t in r.tensors}

    # ---------------- hash routing table ----------------
    # the reference table is the one generated for layer 1 (gen_fixture.py
    # emits references for il == 1, which is both hashed and ratio-4)
    tid = by_name["blk.1.ffn_gate_tid2eid.weight"]
    if tid.tensor_type != gguf.GGMLQuantizationType.I32:
        fail(f"ffn_gate_tid2eid is {tid.tensor_type.name}, expected I32 "
             f"(a quantized hash routing table is fluent garbage)")
    if not np.array_equal(np.asarray(tid.data).reshape(refs["tid2eid"].shape), refs["tid2eid"]):
        fail("ffn_gate_tid2eid values do not match the source table")

    # ---------------- routed experts: type + exact dequant ----------------
    exp = by_name["blk.1.ffn_gate_exps.weight"]
    if exp.tensor_type == gguf.GGMLQuantizationType.MXFP4:
        fail("routed experts were written as raw MXFP4. A later `--outtype PXQ2` "
             "run will short-circuit on tensor->type == new_type and silently emit "
             "a 4.25-bpw file. (Was PXA_DSV4_EXPERTS=mxfp4 set?)")
    if exp.tensor_type != gguf.GGMLQuantizationType.BF16:
        fail(f"blk.1.ffn_gate_exps.weight is {exp.tensor_type.name}, expected BF16 "
             f"(this test converts with --outtype bf16)")
    ref_exp = refs["exp_gate"]
    got_exp = bf16_to_f32(np.asarray(exp.data)).reshape(ref_exp.shape)
    if not np.array_equal(got_exp, ref_exp):
        bad = int(np.count_nonzero(got_exp != ref_exp))
        idx = np.argwhere(got_exp != ref_exp)[:3]
        fail(f"MXFP4 expert dequant is WRONG: {bad}/{ref_exp.size} values differ "
             f"(first: {[tuple(int(x) for x in i) for i in idx]}; "
             f"got {[float(got_exp[tuple(i)]) for i in idx]} vs "
             f"{[float(ref_exp[tuple(i)]) for i in idx]}). "
             f"Check the nibble order (low = even logical index) and the scale "
             f"exponent (2**(byte-127)).")

    # ---------------- fp8 backbone: dequant within Q8_0 tolerance ----------------
    wq = by_name["blk.1.attn_q_a.weight"]
    if wq.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        fail(f"blk.1.attn_q_a.weight is {wq.tensor_type.name}, expected Q8_0 "
             f"(dequantized-fp8 tensors are pinned to Q8_0)")
    ref_wq = refs["fp8_wq_a"]
    got_wq = q8_0_to_f32(np.asarray(wq.data), ref_wq.shape[1]).reshape(ref_wq.shape)
    denom = np.linalg.norm(ref_wq)
    rel = float(np.linalg.norm(got_wq - ref_wq) / denom) if denom else 0.0
    if rel > 0.02:
        fail(f"fp8 e4m3 x ue8m0 dequant is WRONG: relative error {rel:.4f} > 0.02. "
             f"Check the 128x128 block expansion, the MULTIPLY (not divide), and "
             f"that the sidecar is `<w>.scale` (ue8m0), not V3's `_scale_inv`.")

    # ---------------- KV metadata ----------------
    checks = [
        ("deepseek4.block_count", NL),
        ("deepseek4.embedding_length", cfg["hidden_size"]),
        ("deepseek4.attention.head_count", cfg["num_attention_heads"]),
        ("deepseek4.attention.head_count_kv", cfg["num_key_value_heads"]),
        ("deepseek4.attention.key_length", cfg["head_dim"]),
        ("deepseek4.attention.value_length", cfg["head_dim"]),
        ("deepseek4.attention.q_lora_rank", cfg["q_lora_rank"]),
        ("deepseek4.attention.sliding_window", cfg["sliding_window"]),
        ("deepseek4.attention.output_group_count", cfg["o_groups"]),
        ("deepseek4.attention.output_lora_rank", cfg["o_lora_rank"]),
        ("deepseek4.attention.compress_rope_freq_base", cfg["compress_rope_theta"]),
        ("deepseek4.attention.indexer.head_count", cfg["index_n_heads"]),
        ("deepseek4.attention.indexer.key_length", cfg["index_head_dim"]),
        ("deepseek4.attention.indexer.top_k", cfg["index_topk"]),
        ("deepseek4.rope.dimension_count", cfg["qk_rope_head_dim"]),
        ("deepseek4.rope.freq_base", cfg["rope_theta"]),
        ("deepseek4.expert_count", cfg["n_routed_experts"]),
        ("deepseek4.expert_used_count", cfg["num_experts_per_tok"]),
        ("deepseek4.expert_shared_count", cfg["n_shared_experts"]),
        ("deepseek4.expert_feed_forward_length", cfg["moe_intermediate_size"]),
        ("deepseek4.expert_weights_scale", cfg["routed_scaling_factor"]),
        ("deepseek4.expert_weights_norm", cfg["norm_topk_prob"]),
        ("deepseek4.expert_gating_func", int(gguf.ExpertGatingFuncType.SQRTSOFTPLUS)),
        ("deepseek4.hyper_connection.count", cfg["hc_mult"]),
        ("deepseek4.hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"]),
        ("deepseek4.hash_layer_count", cfg["num_hash_layers"]),
        ("deepseek4.vocab_size", cfg["vocab_size"]),
    ]
    for key, expected in checks:
        got_v = kv(r, key)
        if got_v is None:
            fail(f"{key} missing from GGUF")
        if isinstance(expected, bool):
            ok = bool(got_v) == expected
        elif isinstance(expected, float):
            ok = abs(float(got_v) - expected) < 1e-4
        else:
            ok = int(got_v) == int(expected)
        if not ok:
            fail(f"{key} == {got_v!r}, expected {expected!r}")

    ratios = kv_array(r, "deepseek4.attention.compress_ratios")
    if ratios is None:
        fail("deepseek4.attention.compress_ratios missing")
    if [int(x) for x in ratios] != [int(x) for x in RATIOS]:
        fail(f"compress_ratios == {ratios}, expected {RATIOS} (written VERBATIM, "
             f"including the tail entries that belong to the dropped MTP blocks)")

    clamp = kv_array(r, "deepseek4.swiglu_clamp_exp")
    if clamp is None or len(clamp) != NL or abs(float(clamp[0]) - cfg["swiglu_limit"]) > 1e-4:
        fail(f"swiglu_clamp_exp == {clamp}, expected {NL} x {cfg['swiglu_limit']}")

    eps = kv(r, "deepseek4.hyper_connection.epsilon")
    if eps is None or abs(float(eps) - cfg["hc_eps"]) > 1e-9:
        fail(f"hyper_connection.epsilon == {eps}, expected {cfg['hc_eps']}")

    print(f"PASS: {len(got)} tensors match the golden set exactly; arch=deepseek4 "
          f"layers={NL} experts={cfg['n_routed_experts']} gating=sqrtsoftplus; "
          f"MXFP4 expert dequant EXACT; fp8/ue8m0 dequant rel-err {rel:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
