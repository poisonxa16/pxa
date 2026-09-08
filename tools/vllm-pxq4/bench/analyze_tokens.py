#!/usr/bin/env python3
"""Analyze per-token arrival times from bench_client results JSON.
For each single-stream run: inter-token gap stats, and where slow gaps cluster.
"""
import json, sys, statistics

d = json.load(open(sys.argv[1]))
runs = d["single"]["runs"]
print(f"tag={d['tag']} n={len(runs)}")
all_gaps = []
for r in runs:
    ts = r.get("times")
    if not ts or len(ts) < 3:
        continue
    gaps = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    gaps_ms = [g*1000 for g in gaps]
    med = statistics.median(gaps_ms)
    p90 = sorted(gaps_ms)[int(0.9*len(gaps_ms))]
    p99 = sorted(gaps_ms)[int(0.99*len(gaps_ms))]
    mx = max(gaps_ms)
    # count of gaps > 2x median and their total excess time
    slow = [g for g in gaps_ms if g > 2*med]
    excess = sum(g - med for g in slow)/1000
    # where are the slow ones (token indices)
    slow_idx = [i for i,g in enumerate(gaps_ms) if g > 2*med][:12]
    print(f"run{r['i']:2d} tps={r['e2e_tps']:6.2f} med={med:6.2f}ms p90={p90:6.2f} "
          f"p99={p99:7.2f} max={mx:8.2f} nslow={len(slow):3d} excess={excess:5.2f}s "
          f"slow_at={slow_idx}")
    all_gaps.extend(gaps_ms)
if all_gaps:
    s = sorted(all_gaps)
    n = len(s)
    print(f"ALL gaps n={n} med={s[n//2]:.2f} p10={s[n//10]:.2f} p90={s[int(0.9*n)]:.2f} "
          f"p99={s[int(0.99*n)]:.2f} max={s[-1]:.2f}")
    # histogram
    import collections
    buckets = collections.Counter()
    for g in all_gaps:
        if g < 10: b = "<10ms"
        elif g < 15: b = "10-15"
        elif g < 20: b = "15-20"
        elif g < 25: b = "20-25"
        elif g < 30: b = "25-30"
        elif g < 50: b = "30-50"
        elif g < 100: b = "50-100"
        else: b = ">=100ms"
        buckets[b] += 1
    for b in ["<10ms","10-15","15-20","20-25","25-30","30-50","50-100",">=100ms"]:
        if buckets[b]: print(f"  {b:8s} {buckets[b]}")
