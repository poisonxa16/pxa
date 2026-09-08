#!/usr/bin/env python3
"""Integration smoke: a whole engine boots with the pack armed and answers.

The unit gate proves the kernels, and bench/pascal_ops_census.py proves the dispatch moves
on real module instances. Neither proves that an ENGINE -- with its profile_run, its
weight loading, its graph capture and its sampler -- comes up with the seams installed.
That is what this does, on the 1080 Ti, with a small stock model of the same architecture
family as the 35B (Qwen3_5: GemmaRMSNorm layer norms, RMSNormGated GDN layers, SwiGLU),
so every seam in the pack is exercised by a real forward pass.

Run it twice, PXA_OPS_FUSED=0 and =all, and diff the greedy completions. The two runs are
NOT required to agree here -- a divergence at 20 ppm per norm is expected and the real
verdict is the 20-prompt gate on a P100 pair -- but a run that agrees is evidence, and a
run that produces garbage is a defect this catches before a window is spent on it.
"""
import os
import sys

MODEL = os.environ.get("SMOKE_MODEL", "/models/awq-ref")

PROMPTS = [
    "The capital of France is",
    "def add(a, b):",
    "One two three four five six seven",
]


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=float(os.environ.get("SMOKE_GMU", "0.85")),
        enforce_eager=os.environ.get("SMOKE_EAGER", "0") == "1",
        trust_remote_code=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": [1, 2, 4, 8]},
    )
    out = llm.generate(
        PROMPTS,
        SamplingParams(temperature=0.0, max_tokens=24, seed=0),
    )
    print("\n===== COMPLETIONS (PXA_OPS_FUSED=%s) ====="
          % os.environ.get("PXA_OPS_FUSED", "unset"))
    for o in out:
        print(repr(o.prompt), "->", repr(o.outputs[0].text))
        print("   token ids:", list(o.outputs[0].token_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
