#!/usr/bin/env python3
import sys
import os, pathlib
sys.path.insert(0, str(pathlib.Path(os.environ.get("GGUF_PY", pathlib.Path(__file__).resolve().parents[1] / "gguf-py"))))
import gguf

path = sys.argv[1]
r = gguf.GGUFReader(path)

counts = {}
exps_types = set()
attn_kb_types = set()
attn_vb_types = set()
attn_kvamqa_types = set()
attn_kvb_types = set()
router_types = set()
bias_types = set()
norm_types = set()
shexp_types = set()
output_type = None
embd_type = None

for t in r.tensors:
    name = t.name
    ty = t.tensor_type.name
    counts[ty] = counts.get(ty, 0) + 1
    if name.endswith("_exps.weight"):
        exps_types.add(ty)
    if name.endswith("attn_k_b.weight"):
        attn_kb_types.add(ty)
    if name.endswith("attn_v_b.weight"):
        attn_vb_types.add(ty)
    if name.endswith("attn_kv_a_mqa.weight"):
        attn_kvamqa_types.add(ty)
    if name.endswith("attn_kv_b.weight"):
        attn_kvb_types.add(ty)
    if name.endswith("ffn_gate_inp.weight"):
        router_types.add(ty)
    if name.endswith("exp_probs_b.bias"):
        bias_types.add(ty)
    if name.endswith("_norm.weight"):
        norm_types.add(ty)
    if "_shexp.weight" in name:
        shexp_types.add(ty)
    if name == "output.weight":
        output_type = ty
    if name == "token_embd.weight":
        embd_type = ty

n_exps_tensors = sum(1 for t in r.tensors if t.name.endswith("_exps.weight"))

print("=== ", path)
print("total tensors:", len(r.tensors))
print("type histogram:", counts)
print("_exps.weight tensor count:", n_exps_tensors)
print("_exps.weight types (should be single PXQ type):", exps_types)
print("attn_k_b types:", attn_kb_types)
print("attn_v_b types:", attn_vb_types)
print("attn_kv_a_mqa types:", attn_kvamqa_types)
print("attn_kv_b types:", attn_kvb_types)
print("ffn_gate_inp (router) types (must be F32):", router_types)
print("exp_probs_b.bias types (must be F32):", bias_types)
print("*_norm.weight types (must be F32):", norm_types)
print("*_shexp.weight types:", shexp_types)
print("output.weight type:", output_type)
print("token_embd.weight type:", embd_type)
