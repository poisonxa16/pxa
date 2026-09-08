#!/usr/bin/env python3
"""Aggregate CUDA kernel time from a torch profiler chrome trace.
usage: parse_trace.py <trace.json[.gz]> [--steps N] [--window-us A B]
Groups kernels by name, reports total time, count, and per-decode-step cost.
Also derives step boundaries from a marker kernel if --per-step given.
"""
import json, sys, gzip, re, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--steps", type=int, default=None,
                help="divide totals by N steps for per-step cost")
args = ap.parse_args()

op = gzip.open if args.trace.endswith(".gz") else open
with op(args.trace, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data

kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        and e.get("ph") == "X"]
if not kern:
    # newer naming
    kern = [e for e in ev if e.get("cat") in ("cuda_runtime",) and False]
print(f"kernel events: {len(kern)}", file=sys.stderr)

# steady-state window: drop first/last 10% by time
ts = sorted(e["ts"] for e in kern)
t0 = ts[0] + (ts[-1] - ts[0]) * 0.05
t1 = ts[0] + (ts[-1] - ts[0]) * 0.95
kern = [e for e in kern if t0 <= e["ts"] <= t1]
span_ms = (t1 - t0) / 1000.0

def bucket(name):
    n = name
    if "pxq4" in n or "PXQ4" in n: return "pxq4_linear"
    if "marlin" in n.lower(): return "marlin_quant_gemm"
    if "turbomind" in n or "s884" in n or "sm70_f16" in n or "GemmUniversal" in n: return "turbomind_gemm"
    if "fused_sigmoid" in n or "cumsum" in n or "l2norm" in n: return "gdn_linear_attn"
    if "nccl" in n or "ncclKernel" in n or "AllReduce" in n or "cross_device_reduce" in n or "all_reduce" in n: return "allreduce"
    if "gemm" in n.lower() or "gemv" in n.lower() or "cutlass" in n or "s16816" in n or "hmma" in n or "884" in n: return "dense_gemm"
    if "flash" in n.lower() or "attn" in n.lower() or "attention" in n.lower(): return "attention"
    if "chunk_" in n or "fused_recurrent" in n or "gdn" in n.lower() or "kda" in n or "wy_fast" in n or "solve_tril" in n or "kkt" in n or "delta" in n.lower(): return "gdn_linear_attn"
    if "norm" in n.lower(): return "norm"
    if "elementwise" in n or "vectorized" in n or "silu" in n.lower() or "mul" in n.lower() or "add" in n.lower() or "copy" in n.lower() or "cat" in n.lower(): return "elementwise/copy"
    if "topk" in n.lower() or "sample" in n.lower() or "argmax" in n.lower() or "softmax" in n.lower() or "sort" in n.lower(): return "sampling"
    if "Memcpy" in n or "Memset" in n: return "memcpy/set"
    if "rope" in n.lower() or "rotary" in n.lower(): return "rope"
    if "conv" in n.lower(): return "conv1d"
    return "other"

per_gpu = collections.defaultdict(lambda: collections.defaultdict(float))
byname = collections.defaultdict(lambda: [0.0, 0])
bybucket = collections.defaultdict(float)
for e in kern:
    d = e.get("dur", 0)
    nm = e["name"]
    byname[nm][0] += d
    byname[nm][1] += 1
    bybucket[bucket(nm)] += d

tot = sum(v for v in bybucket.values())
print(f"\nwindow {span_ms:.1f} ms wall; total GPU kernel time {tot/1000:.1f} ms")
if args.steps:
    print(f"per-step ({args.steps} steps):")
print("\n=== BUCKETS ===")
for b, v in sorted(bybucket.items(), key=lambda x: -x[1]):
    line = f"{b:22s} {v/1000:9.2f} ms  {100*v/tot:5.1f}%"
    if args.steps: line += f"  {v/1000/args.steps:7.3f} ms/step"
    print(line)
print(f"\n=== TOP {args.top} KERNELS ===")
for nm, (v, c) in sorted(byname.items(), key=lambda x: -x[1][0])[:args.top]:
    line = f"{v/1000:9.2f} ms {c:6d}x  [{bucket(nm)}] {nm[:110]}"
    print(line)
