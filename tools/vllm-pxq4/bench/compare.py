#!/usr/bin/env python3
import json, sys, statistics
R = "/mnt/models/pxa-hth/results/"
def load(t):
    try: return json.load(open(R+t+".json"))
    except Exception: return None
tags = sys.argv[1:] or ["awq-owner","pxq4-owner","pxq4v6-owner","awq-owner-pf"]
rows = {}
for t in tags:
    d = load(t)
    if not d: print(f"[{t}: missing]"); continue
    r = {}
    s = d.get("single")
    if s:
        tps=[x["e2e_tps"] for x in s["runs"]]; dec=[x["decode_tps"] for x in s["runs"]]
        ftl=[x["ftl"] for x in s["runs"]]
        r["single"]=f"{statistics.median(tps):.2f} [{min(tps):.1f}-{max(tps):.1f}] n={len(tps)}"
        r["decode"]=f"{statistics.median(dec):.2f}"
        r["ftl"]=f"med {statistics.median(ftl):.3f} max {max(ftl):.3f} spikes>{0.5}: {sum(1 for f in ftl if f>0.5)}"
    ns=d.get("nostream")
    if ns: r["nostream"]=f"{ns['median']:.2f}"
    p=d.get("prefill")
    if p: r["prefill"]=f"{p['median']:.1f} [{p['min']:.1f}-{p['max']:.1f}] n={p['n']}"
    for c,v in (d.get("conc") or {}).items():
        r[f"conc{c}"]=f"agg {v['aggregate_tps']:.2f} per {v['per_stream_mean']:.2f} [{v['per_stream_min']:.2f}-{v['per_stream_max']:.2f}]"
    rows[t]=r
keys=[]
for r in rows.values():
    for k in r:
        if k not in keys: keys.append(k)
for k in keys:
    print(f"--- {k}")
    for t in rows:
        print(f"   {t:16s} {rows[t].get(k,'-')}")
