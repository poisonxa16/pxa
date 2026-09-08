#!/usr/bin/env python3
"""Coherence-gated decode benchmark that also reports spec-decode acceptance.

With correct rejection sampling a *randomly initialised* draft head still
produces correct text -- it just accepts nothing. So for MTP the 391 check is
necessary but nowhere near sufficient: the acceptance rate is what says whether
the draft head carries real weights.
"""
import argparse, json, re, statistics, sys, time, requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="m")
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default=None)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

PROMPTS = [
 "The history of the steam engine begins in the first century AD, when",
 "def quicksort(arr):\n    \"\"\"Sort the array using quicksort.\"\"\"\n",
 "Explain, step by step, why the sky is blue:\n1.",
 "Translate to French: 'The quick brown fox jumps over the lazy dog.' Then explain each word choice.\n",
 "A train leaves Chicago at 60 mph and another leaves New York at 80 mph. If the cities are 790 miles apart,",
 "Write a short story that begins: The lighthouse keeper had not spoken to anyone in three years, until",
 "List the planets of the solar system with one interesting fact each:\n",
 "The main differences between TCP and UDP are as follows. First,",
]

def coherence():
    r = requests.post(f"{BASE}/v1/completions", json={"model": a.model,
        "prompt": "What is 17 times 23? Answer with just the number.",
        "max_tokens": 24, "temperature": 0.0}, timeout=600)
    t = r.json()["choices"][0]["text"]
    return ("391" in t), t.strip()[:80]

def metrics():
    """Return the spec-decode counters vLLM exposes, if any."""
    try:
        txt = requests.get(f"{BASE}/metrics", timeout=30).text
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or "spec" not in line and "draft" not in line and "accept" not in line:
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        out[name + labels] = out.get(name + labels, 0.0) + val
    return out

def stream_one(prompt, ntok):
    body = {"model": a.model, "prompt": prompt, "max_tokens": ntok,
            "temperature": 0.0, "stream": True, "ignore_eos": True, "seed": 12345}
    t0 = time.time(); times = []; n = 0
    with requests.post(f"{BASE}/v1/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "): continue
            p = line[6:]
            if p == b"[DONE]": break
            ch = json.loads(p).get("choices", [])
            if ch and ch[0].get("text") is not None:
                n += 1; times.append(time.time())
    t1 = time.time()
    return {"ntok": n, "secs": t1 - t0, "ftl": times[0]-t0 if times else None,
            "e2e_tps": n/(t1-t0),
            "decode_tps": (n-1)/(times[-1]-times[0]) if n > 1 else 0.0}

def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median": statistics.median(xs), "mean": statistics.mean(xs),
            "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0, "min": xs[0], "max": xs[-1]}

res = {"tag": a.tag, "port": a.port, "tokens": a.tokens}
ok, txt = coherence()
res["coherence_before"] = {"ok": ok, "text": txt}
print(f"coherence BEFORE: {'PASS' if ok else 'FAIL'}  {txt!r}", flush=True)
if not ok:
    print("ABORT: incoherent; any number would be meaningless."); sys.exit(2)

stream_one(PROMPTS[0], 32)
m0 = metrics()
runs = [stream_one(PROMPTS[i % len(PROMPTS)], a.tokens) for i in range(a.n)]
m1 = metrics()

res["single"] = {"runs": runs,
                 "e2e": stats([r["e2e_tps"] for r in runs]),
                 "decode": stats([r["decode_tps"] for r in runs])}
print("single e2e   ", json.dumps(res["single"]["e2e"]), flush=True)
print("single decode", json.dumps(res["single"]["decode"]), flush=True)

delta = {k: m1[k] - m0.get(k, 0.0) for k in m1 if m1[k] - m0.get(k, 0.0) != 0}
res["spec_metrics_delta"] = delta
if delta:
    print("spec-decode counters over the timed runs:")
    for k in sorted(delta): print(f"   {k} = {delta[k]:g}")
    def pick(*must):
        return next((v for k, v in delta.items()
                     if all(m in k for m in must) and "per_pos" not in k), None)
    acc = pick("num_accepted_tokens_total")
    dtok = pick("num_draft_tokens_total")     # total speculative tokens proposed
    ndraft = pick("num_drafts_total")         # number of draft *attempts*
    if acc is not None and dtok:
        # Acceptance rate is accepted / drafted TOKENS -- not / drafts. Dividing
        # by the number of draft attempts instead inflates it by the number of
        # speculative tokens per draft (4 for MTP4) and is not comparable to any
        # published figure.
        res["acceptance_rate"] = acc / dtok
        print(f"   acceptance rate      = {acc/dtok:.4f}  "
              f"({acc:g} accepted / {dtok:g} drafted tokens)")
    if acc is not None and ndraft:
        res["mean_accept_length"] = 1 + acc / ndraft
        print(f"   mean accept length   = {1 + acc/ndraft:.3f}  "
              f"(1.0 means nothing speculative was ever accepted)")
else:
    print("spec-decode counters: NONE exposed (no speculative decoding active, "
          "or metrics disabled)")

ok, txt = coherence()
res["coherence_after"] = {"ok": ok, "text": txt}
print(f"coherence AFTER: {'PASS' if ok else 'FAIL'}  {txt!r}", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)
