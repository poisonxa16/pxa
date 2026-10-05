#!/usr/bin/env python3
"""Coherence-gated decode benchmark.

Refuses to report a throughput number unless the model answers 17x23 -> 391
BEFORE and AFTER the timed runs.  (ignore_eos benchmarks are coherence-blind:
a model that babbles benchmarks perfectly.)
"""
import argparse, json, statistics, sys, time, threading
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="m")
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--conc", default="")
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default=None)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/completions"

PROMPTS = [
 "The history of the steam engine begins in the first century AD, when",
 "def quicksort(arr):\n    \"\"\"Sort the array using quicksort.\"\"\"\n",
 "Explain, step by step, why the sky is blue:\n1.",
 "Translate to French: 'The quick brown fox jumps over the lazy dog.' Then explain each word choice.\n",
 "A train leaves Chicago at 60 mph and another leaves New York at 80 mph. If the cities are 790 miles apart,",
 "Write a short story that begins: The lighthouse keeper had not spoken to anyone in three years, until",
 "List the planets of the solar system with one interesting fact each:\n",
 "SELECT statement to find the top 5 customers by total order value from tables customers(id,name) and orders(id,customer_id,total):\n",
 "The main differences between TCP and UDP are as follows. First,",
 "import numpy as np\n# compute the eigenvalues of a 3x3 matrix\n",
 "Summarize the causes of World War I in a concise paragraph:\n",
 "The recipe for a perfect sourdough loaf starts with the starter.",
]

def coherence():
    r = requests.post(URL, json={"model": a.model,
        "prompt": "What is 17 times 23? Answer with just the number.",
        "max_tokens": 24, "temperature": 0.0}, timeout=300)
    txt = r.json()["choices"][0]["text"]
    return ("391" in txt), txt.strip()[:80]

def stream_one(prompt, ntok):
    body = {"model": a.model, "prompt": prompt, "max_tokens": ntok,
            "temperature": 0.0, "stream": True, "ignore_eos": True, "seed": 12345}
    t0 = time.time(); times = []; n = 0
    with requests.post(URL, json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "): continue
            p = line[6:]
            if p == b"[DONE]": break
            ch = json.loads(p).get("choices", [])
            if ch and ch[0].get("text") is not None:
                n += 1; times.append(time.time())
    t1 = time.time()
    ftl = times[0] - t0 if times else None
    # decode tok/s excludes the prefill/first-token latency
    dec = (n - 1) / (times[-1] - times[0]) if n > 1 else 0.0
    return {"ntok": n, "secs": t1 - t0, "ftl": ftl,
            "e2e_tps": n / (t1 - t0), "decode_tps": dec}

def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median": statistics.median(xs),
            "mean": statistics.mean(xs),
            "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": xs[0], "max": xs[-1],
            "p25": xs[max(0, int(0.25*(len(xs)-1)))],
            "p75": xs[min(len(xs)-1, int(0.75*(len(xs)-1)+0.999))]}

res = {"tag": a.tag, "port": a.port, "tokens": a.tokens}
ok, txt = coherence()
res["coherence_before"] = {"ok": ok, "text": txt}
print(f"coherence BEFORE: {'PASS' if ok else 'FAIL'}  {txt!r}", flush=True)
if not ok:
    print("ABORT: model is not coherent; any throughput number would be meaningless.")
    sys.exit(2)

stream_one(PROMPTS[0], 32)  # warmup

if a.n:
    runs = [stream_one(PROMPTS[i % len(PROMPTS)], a.tokens) for i in range(a.n)]
    res["single"] = {"runs": runs,
                     "e2e": stats([r["e2e_tps"] for r in runs]),
                     "decode": stats([r["decode_tps"] for r in runs]),
                     "ftl": stats([r["ftl"] for r in runs])}
    print("single e2e   ", json.dumps(res["single"]["e2e"]), flush=True)
    print("single decode", json.dumps(res["single"]["decode"]), flush=True)

if a.conc:
    res["conc"] = {}
    for c in [int(x) for x in a.conc.split(",") if x]:
        out = [None]*c
        def w(i):
            out[i] = stream_one(PROMPTS[i % len(PROMPTS)], a.tokens)
        ts = [threading.Thread(target=w, args=(i,)) for i in range(c)]
        t0 = time.time()
        for t in ts: t.start()
        for t in ts: t.join()
        wall = time.time() - t0
        tot = sum(r["ntok"] for r in out)
        res["conc"][str(c)] = {"agg_tps": tot/wall, "wall": wall, "tot_tok": tot,
                               "per_stream": stats([r["e2e_tps"] for r in out]),
                               "runs": out}
        print(f"conc {c}: aggregate {tot/wall:.2f} tok/s, per-stream median "
              f"{res['conc'][str(c)]['per_stream']['median']:.2f}", flush=True)

ok, txt = coherence()
res["coherence_after"] = {"ok": ok, "text": txt}
print(f"coherence AFTER: {'PASS' if ok else 'FAIL'}  {txt!r}", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
if not ok:
    sys.exit(2)
