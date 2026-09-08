#!/usr/bin/env python3
"""Measure a live vLLM endpoint. No bc on this box, so all timing math is here.
Reports single-stream decode (two-point, prefill cancelled -- same method as the
gate, so numbers compare to it) plus aggregate at N concurrent."""
import json, sys, time, urllib.request, concurrent.futures as cf

PORT = int(sys.argv[1])
BASE = "http://127.0.0.1:%d" % PORT
PROMPT = "Explain in detail how a mixture-of-experts layer routes tokens."

def post(n, prompt=PROMPT, timeout=600):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": n,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    return time.time() - t0, j

out = {}
# correctness FIRST -- a fast wrong answer is worthless
try:
    _, j = post(5, "The capital of France is", timeout=180)
    txt = j["choices"][0]["text"].strip()
    out["correct_text"] = txt[:40]
    out["correct"] = "paris" in txt.lower()
except Exception as e:
    out["correct"] = False; out["correct_text"] = "ERR %s" % str(e)[:60]

if out["correct"]:
    try:
        t16, _ = post(16); t144, _ = post(144)
        out["t16"] = round(t16, 3); out["t144"] = round(t144, 3)
        out["decode_tps"] = round(128.0 / (t144 - t16), 2)
        out["ms_per_tok"] = round((t144 - t16) * 1000.0 / 128.0, 2)
    except Exception as e:
        out["decode_err"] = str(e)[:80]
    for conc in (4, 8):
        try:
            t0 = time.time()
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [ex.submit(post, 128, PROMPT + " " * i) for i in range(conc)]
                [f.result() for f in futs]
            el = time.time() - t0
            out["agg%d_tps" % conc] = round(conc * 128.0 / el, 2)
            out["per_stream%d" % conc] = round(128.0 / el, 2)
        except Exception as e:
            out["agg%d_err" % conc] = str(e)[:60]
print(json.dumps(out))
