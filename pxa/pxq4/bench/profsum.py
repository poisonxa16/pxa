# SPDX-License-Identifier: Apache-2.0
"""Turn a worker chrome trace into the per-token decode breakdown.

  profsum.py <trace_dir_or_file> [n_steps]

Reports, per profiled decode step: total GPU kernel time, kernel count, and the split
into the categories that a fix would target - collectives, the routed MoE, PXQ4 dense
linears, fp16 GEMV, attention (full and GDN/linear), and the elementwise/cast/norm glue.
Also reports the host side: the wall span of the trace against the summed GPU time, which
is the gap that no kernel fix can touch.

The categories are matched on kernel NAME and every unmatched kernel is listed under
"unclassified" with its own line, so nothing is quietly folded into a bucket it does not
belong in - an unclassified row that is large is a finding, not an inconvenience.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

CATS = [
    ("collective",  r"nccl|AllReduce|all_reduce|cross_device_reduce|custom_ar|oneshot|twoshot"),
    ("moe_routed",  r"pxq4_moe|moe_mmv|k_pxq4_moe"),
    ("pxq4_linear", r"pxq4_mmv|k_pxq4_(?!moe)|pxq4_dequant|pxq4_linear"),
    ("f16_gemv",    r"gemv|f16_mmv|sgemm|hgemm|cutlass|turing_|volta_|maxwell_|dot_kernel|gemmk"),
    ("attn_gdn",    r"gdn|delta|chunk_|recurrent|conv1d|causal_conv|fla_|mamba|ssm|state_passing"),
    ("attn_full",   r"attention|attn|flash|sdpa|paged|softmax_kernel|reshape_and_cache"),
    ("norm",        r"rms_norm|layer_norm|layernorm|rmsnorm"),
    ("topk_route",  r"topk|moeSoftmax|softmax_kernel|argsort|radix"),
    ("elementwise", r"elementwise|vectorized|unrolled|reduce_kernel|fill_|copy|cat_|index|scatter|gather|silu|mul_|add_"),
    ("sample",      r"sample|multinomial|argmax|logits|penal|repetition"),
]
COMPILED = [(n, re.compile(p, re.I)) for n, p in CATS]


def classify(name: str) -> str:
    for n, rx in COMPILED:
        if rx.search(name):
            return n
    return "unclassified"


def load(path: str) -> list[dict]:
    files = []
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.endswith(".json") or f.endswith(".json.gz"):
                files.append(os.path.join(path, f))
    else:
        files = [path]
    if not files:
        print("no traces under", path)
        return []
    return files


def summarize(fn: str, n_steps: int) -> None:
    opener = open
    if fn.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(fn, "rt") as fh:
        tr = json.load(fh)
    ev = tr.get("traceEvents", tr if isinstance(tr, list) else [])
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")]
    if not kern:
        # Some builds tag device events as "gpu_user_annotation"/"cuda_runtime";
        # fall back to anything on a device (pid) track with a dur.
        kern = [e for e in ev if e.get("ph") == "X" and e.get("cat", "").lower()
                in ("gpu_memcpy", "gpu_memset", "kernel")]
    if not kern:
        print(f"  {os.path.basename(fn)}: no kernel events (CUPTI may have lumped a "
              f"captured graph; take the enforce-eager arm for the census)")
        return
    tmin = min(e["ts"] for e in kern)
    tmax = max(e["ts"] + e.get("dur", 0) for e in kern)
    by_cat_t: dict[str, float] = defaultdict(float)
    by_cat_n: dict[str, int] = defaultdict(int)
    unclassified: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for e in kern:
        c = classify(e.get("name", ""))
        by_cat_t[c] += e.get("dur", 0)
        by_cat_n[c] += 1
        if c == "unclassified":
            u = unclassified[e["name"][:70]]
            u[0] += 1
            u[1] += e.get("dur", 0)
    tot_t = sum(by_cat_t.values())
    tot_n = sum(by_cat_n.values())
    span = tmax - tmin
    print(f"\n  === {os.path.basename(fn)} ===")
    print(f"  {n_steps} profiled steps | trace span {span/1000:.2f} ms | "
          f"summed GPU kernel time {tot_t/1000:.2f} ms | {tot_n} kernels")
    print(f"  PER STEP: {span/1000/n_steps:.2f} ms wall, {tot_t/1000/n_steps:.2f} ms GPU, "
          f"{tot_n/n_steps:.0f} kernels, "
          f"GPU busy {100*tot_t/max(span,1):.0f}% of the span")
    print(f"  {'category':<16}{'kernels/step':>14}{'ms/step':>10}{'share':>8}")
    for c, t in sorted(by_cat_t.items(), key=lambda x: -x[1]):
        print(f"  {c:<16}{by_cat_n[c]/n_steps:>14.1f}{t/1000/n_steps:>10.3f}"
              f"{100*t/max(tot_t,1):>7.1f}%")
    if unclassified:
        print("  unclassified kernels, largest first (each is a finding, not a rounding):")
        for name, (n, t) in sorted(unclassified.items(), key=lambda x: -x[1][1])[:12]:
            print(f"    {name:<70}{n/n_steps:>8.1f}{t/1000/n_steps:>10.3f} ms")


if __name__ == "__main__":
    p = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    for f in load(p):
        try:
            summarize(f, n)
        except Exception as e:
            print(f"  {os.path.basename(f)}: {type(e).__name__}: {str(e)[:120]}")
