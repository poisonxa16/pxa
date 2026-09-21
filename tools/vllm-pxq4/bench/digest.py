#!/usr/bin/env python3
"""Compact one-file digest of a benchmark result JSON."""
import json, sys
d = json.load(open(sys.argv[1]))
print("  tag:", d.get("tag"), " model:", str(d.get("model"))[-46:])
for key in ("single", "nostream"):
    s = d.get(key)
    if isinstance(s, dict):
        print("  %-9s n=%s median=%.2f mean=%.2f sd=%.2f min=%.2f max=%.2f" % (
            key, s.get("n"), s.get("median", 0), s.get("mean", 0),
            s.get("sd", 0), s.get("min", 0), s.get("max", 0)))
if "median_tokps" in d:
    print("  step      n=%s median=%.2f mean=%.2f sd=%.2f" % (
        d.get("n"), d["median_tokps"], d.get("mean_tokps", 0), d.get("stdev", 0)))
c = d.get("conc")
if isinstance(c, dict):
    for k in sorted(c, key=lambda x: int(x)):
        v = c[k]
        print("  conc%-3s agg=%.2f per_stream=%.2f ok=%s" % (
            k, v.get("aggregate_tps", 0), v.get("per_stream_mean", 0), v.get("streams_ok")))
for key in ("prefill", "ftl", "singleprefill"):
    p = d.get(key)
    if isinstance(p, dict):
        print("  %s: %s" % (key, json.dumps(p)[:240]))
