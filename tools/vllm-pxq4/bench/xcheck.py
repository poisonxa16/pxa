#!/usr/bin/env python3
"""Greedy-identity check: vLLM vs llama.cpp on the same GGUF.

Stronger than the 17x23 spot check -- it catches a decode path that is subtly
wrong only at longer contexts (e.g. a cudagraph that baked in a stale shape).
"""
import argparse, json, sys, requests

ap = argparse.ArgumentParser()
ap.add_argument("--vllm-port", type=int, default=8423)
ap.add_argument("--lcpp-port", type=int, default=8243)
ap.add_argument("--model", default="m")
ap.add_argument("--n", type=int, default=96)
a = ap.parse_args()

PROMPTS = [
    "What is 17 times 23? Answer with just the number.",
    "The history of the steam engine begins in the first century AD, when",
    "List the planets of the solar system with one interesting fact each:\n",
]

def gen(port, model, prompt, n):
    r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                      json={"model": model, "prompt": prompt, "max_tokens": n,
                            "temperature": 0.0, "top_k": 1, "seed": 0,
                            "n_probs": 0, "cache_prompt": False},
                      timeout=1800)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]

bad = 0
for p in PROMPTS:
    v = gen(a.vllm_port, a.model, p, a.n)
    l = gen(a.lcpp_port, "/m/Qwen38-27B-Unc-PXQ4.gguf", p, a.n)
    # first divergence position
    k = 0
    while k < min(len(v), len(l)) and v[k] == l[k]:
        k += 1
    same = (v == l)
    print(f"--- prompt: {p[:50]!r}")
    print(f"    identical={same}  common_prefix_chars={k}/{min(len(v),len(l))}")
    if not same:
        bad += 1
        print(f"    vllm : {v[:300]!r}")
        print(f"    lcpp : {l[:300]!r}")
    else:
        print(f"    text : {v[:160]!r}")
sys.exit(1 if bad else 0)
