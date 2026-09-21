#!/usr/bin/env python3
"""
G3 fixture builder — synthetic GLM-4.7-Flash-shaped GGUF (deepseek2 arch) for
llama-quantize PXQ4/PXQ6 tensor-type dispatch verification only.

NOT a real model: no tokenizer, random weights, most non-expert tensors are
shrunk to a tiny placeholder shape (their exact size is irrelevant to the
name+shape-driven PXQ eligibility check). Tensors whose exact shape DOES
matter for the G3 spec are kept literal:
  - ffn_gate_exps / ffn_up_exps : ne=[2048,1536,E]   (E shrunk 64->2, doesn't
    affect eligibility which only checks ne[0]/ne[1])
  - ffn_down_exps               : ne=[1536,2048,E]
  - ffn_gate_inp (router)       : ne=[2048,64]        (exact per spec)
  - attn_k_b                    : ne[0]=192            (critical MLA gotcha:
    not divisible by 256)
  - exp_probs_b.bias            : 1D bias, must stay F32
  - all *_norm.weight           : must stay F32
"""
import sys
import numpy as np

import os, pathlib
sys.path.insert(0, str(pathlib.Path(os.environ.get("GGUF_PY", pathlib.Path(__file__).resolve().parents[1] / "gguf-py"))))
import gguf  # noqa: E402

N_LAYER = 47          # blk.0 dense, blk.1..46 MoE  (real GLM-4.7-Flash)
HIDDEN = 2048
MOE_FF = 1536
N_ROUTED_EXPERTS = 64  # router width (ffn_gate_inp ne[1]) -- exact per spec
E_TEST = 2             # stacked-expert dim for _exps tensors (shrunk from 64;
                       # eligibility only checks ne[0]/ne[1], not ne[2])
VOCAB = 32
TINY = 64

rng = np.random.default_rng(1234)


def f16(shape):
    return (rng.standard_normal(shape) * 0.02).astype(np.float16)


def f32(shape):
    return (rng.standard_normal(shape) * 0.02).astype(np.float32)


OUT = os.environ.get("GLM_FIXTURE_OUT", "glm47-fixture.gguf")

w = gguf.GGUFWriter(OUT, "deepseek2", endianess=gguf.GGUFEndian.LITTLE)

# ---- minimal KV so llm_load_arch/llm_load_hparams parse cleanly (both are
# wrapped in try/catch in llama-quantize.cpp anyway, but keep it honest) ----
w.add_name("glm47-fixture-G3")
w.add_block_count(N_LAYER)
w.add_embedding_length(HIDDEN)
w.add_feed_forward_length(MOE_FF)
w.add_head_count(20)
w.add_head_count_kv(20)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(64)
w.add_rope_freq_base(1000000.0)
w.add_expert_count(N_ROUTED_EXPERTS)
w.add_expert_used_count(4)
w.add_expert_shared_count(1)
w.add_expert_feed_forward_length(MOE_FF)
w.add_expert_weights_scale(1.8)
w.add_expert_weights_norm(True)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
w.add_leading_dense_block_count(1)
w.add_q_lora_rank(768)
w.add_kv_lora_rank(512)
w.add_key_length(576)
w.add_value_length(512)
w.add_context_length(202752)
w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

# ---- tensors ----
w.add_tensor("token_embd.weight", f16((VOCAB, HIDDEN)))
w.add_tensor("output_norm.weight", f32((HIDDEN,)))
w.add_tensor("output.weight", f16((VOCAB, HIDDEN)))

n_exps_tensors = 0

for i in range(N_LAYER):
    p = f"blk.{i}"
    w.add_tensor(f"{p}.attn_norm.weight", f32((HIDDEN,)))
    w.add_tensor(f"{p}.attn_q_a_norm.weight", f32((TINY,)))
    w.add_tensor(f"{p}.attn_kv_a_norm.weight", f32((TINY,)))
    w.add_tensor(f"{p}.attn_q_a.weight", f16((TINY, TINY)))
    w.add_tensor(f"{p}.attn_q_b.weight", f16((TINY, TINY)))
    w.add_tensor(f"{p}.attn_kv_a_mqa.weight", f16((TINY, TINY)))
    w.add_tensor(f"{p}.attn_kv_b.weight", f16((TINY, TINY)))
    # CRITICAL MLA gotcha tensor: ne[0] MUST be 192 (not 256-divisible)
    w.add_tensor(f"{p}.attn_k_b.weight", f16((TINY, 192)))
    w.add_tensor(f"{p}.attn_v_b.weight", f16((TINY, TINY)))
    w.add_tensor(f"{p}.attn_output.weight", f16((TINY, TINY)))
    w.add_tensor(f"{p}.ffn_norm.weight", f32((HIDDEN,)))

    if i == 0:
        # first_k_dense_replace=1 -> layer 0 is dense, NOT _exps
        w.add_tensor(f"{p}.ffn_gate.weight", f16((TINY, TINY)))
        w.add_tensor(f"{p}.ffn_up.weight", f16((TINY, TINY)))
        w.add_tensor(f"{p}.ffn_down.weight", f16((TINY, TINY)))
    else:
        # router: exact shape per spec, F32 (must stay F32)
        w.add_tensor(f"{p}.ffn_gate_inp.weight", f32((N_ROUTED_EXPERTS, HIDDEN)))
        # score-correction bias: 1D, F32 (must stay F32; excluded anyway since
        # it doesn't end in "weight")
        w.add_tensor(f"{p}.exp_probs_b.bias", f32((N_ROUTED_EXPERTS,)))
        # ELIGIBLE expert tensors -- exact ne[0]/ne[1] per spec
        w.add_tensor(f"{p}.ffn_gate_exps.weight", f16((E_TEST, MOE_FF, HIDDEN)))
        w.add_tensor(f"{p}.ffn_up_exps.weight", f16((E_TEST, MOE_FF, HIDDEN)))
        w.add_tensor(f"{p}.ffn_down_exps.weight", f16((E_TEST, HIDDEN, MOE_FF)))
        n_exps_tensors += 3
        # shared expert (not _exps suffix -> not PXQ-eligible, rides MXFP4 rules)
        w.add_tensor(f"{p}.ffn_gate_shexp.weight", f16((TINY, TINY)))
        w.add_tensor(f"{p}.ffn_up_shexp.weight", f16((TINY, TINY)))
        w.add_tensor(f"{p}.ffn_down_shexp.weight", f16((TINY, TINY)))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()

print(f"wrote {OUT}")
print(f"expected PXQ-eligible _exps tensors: {n_exps_tensors} (should be 138 for 46 MoE layers x 3)")
