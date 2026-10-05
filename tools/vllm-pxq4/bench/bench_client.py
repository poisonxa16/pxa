#!/usr/bin/env python3
"""Benchmark client for an OpenAI-compatible vLLM server on localhost.
Runs INSIDE the server container via docker exec (or anywhere with `requests`).

Modes:
  single:  n sequential single-stream runs, 512 tokens each, streaming with
           per-token arrival timestamps saved.
  conc:    for each c in --conc list, fire c simultaneous streaming requests
           (threads), record per-stream tok/s and aggregate tok/s.

Output: JSON to --out.
"""
import argparse, json, threading, time, statistics, sys
import requests

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, required=True)
p.add_argument("--model", default="qwen3.8-27b")
p.add_argument("--tokens", type=int, default=512)
p.add_argument("--n", type=int, default=12)
p.add_argument("--conc", default="2,4,8,16")
p.add_argument("--nostream-n", type=int, default=6)
p.add_argument("--prefill-n", type=int, default=8)
p.add_argument("--mode", default="all", choices=["single", "conc", "all", "prefill", "singleprefill"])
p.add_argument("--out", required=True)
p.add_argument("--tag", default="run")
args = p.parse_args()

URL = f"http://127.0.0.1:{args.port}/v1/completions"

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

def stream_one(prompt, max_tokens, record_times=False):
    """Returns (completion_tokens, elapsed, first_token_latency, times list)."""
    body = {"model": args.model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True,
            "ignore_eos": True, "seed": 12345}
    t0 = time.time()
    times = []
    ntok = 0
    with requests.post(URL, json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            ch = d.get("choices", [])
            if ch:
                # count every chunk with text as >=1 token; vLLM sends one token per chunk
                if ch[0].get("text") is not None:
                    ntok += 1
                    times.append(time.time())
    t1 = time.time()
    ftl = (times[0] - t0) if times else None
    out_times = [round(x - t0, 5) for x in times] if record_times else None
    return ntok, t1 - t0, ftl, out_times

result = {"tag": args.tag, "port": args.port, "tokens": args.tokens}

# warmup
stream_one(PROMPTS[0], 32)

if args.mode in ("single", "all", "singleprefill"):
    runs = []
    for i in range(args.n):
        pr = PROMPTS[i % len(PROMPTS)]
        ntok, dt, ftl, times = stream_one(pr, args.tokens, record_times=True)
        # decode tok/s excluding first-token (prefill) latency
        decode_s = dt - ftl if ftl else dt
        decode_tps = (ntok - 1) / decode_s if ntok > 1 and decode_s > 0 else 0
        runs.append({"i": i, "ntok": ntok, "secs": dt, "ftl": ftl,
                     "e2e_tps": ntok / dt, "decode_tps": decode_tps,
                     "times": times})
        print(f"[single {i}] {ntok} tok in {dt:.2f}s = {ntok/dt:.2f} tok/s "
              f"(decode {decode_tps:.2f}, ftl {ftl:.3f}s)", flush=True)
    tps = [r["e2e_tps"] for r in runs]
    result["single"] = {
        "n": len(tps), "median": statistics.median(tps),
        "mean": statistics.mean(tps),
        "sd": statistics.stdev(tps) if len(tps) > 1 else 0,
        "min": min(tps), "max": max(tps), "runs": runs}
    print(f"SINGLE: median {statistics.median(tps):.2f} mean {statistics.mean(tps):.2f} "
          f"sd {statistics.stdev(tps):.2f} min {min(tps):.2f} max {max(tps):.2f} n={len(tps)}",
          flush=True)


if args.mode in ("single", "all", "singleprefill") and args.nostream_n > 0:
    ns = []
    for i in range(args.nostream_n):
        pr = PROMPTS[i % len(PROMPTS)]
        body = {"model": args.model, "prompt": pr, "max_tokens": args.tokens,
                "temperature": 0.0, "stream": False, "ignore_eos": True, "seed": 12345}
        t0 = time.time()
        r = requests.post(URL, json=body, timeout=600)
        dt = time.time() - t0
        u = r.json().get("usage", {})
        ct = u.get("completion_tokens", 0)
        ns.append({"i": i, "ntok": ct, "secs": dt, "tps": ct / dt})
        print(f"[nostream {i}] {ct} tok in {dt:.2f}s = {ct/dt:.2f} tok/s", flush=True)
    tps2 = [x["tps"] for x in ns]
    result["nostream"] = {"n": len(tps2), "median": statistics.median(tps2),
        "mean": statistics.mean(tps2),
        "sd": statistics.stdev(tps2) if len(tps2) > 1 else 0,
        "min": min(tps2), "max": max(tps2), "runs": ns}
    print(f"NOSTREAM: median {statistics.median(tps2):.2f} mean {statistics.mean(tps2):.2f} "
          f"sd {statistics.stdev(tps2):.2f} min {min(tps2):.2f} max {max(tps2):.2f} n={len(tps2)}", flush=True)


def tok_count(text):
    r = requests.post(f"http://127.0.0.1:{args.port}/tokenize",
                      json={"model": args.model, "prompt": text}, timeout=60)
    return r.json().get("count") or len(r.json().get("tokens", []))

if args.mode in ("all", "prefill", "singleprefill") and args.prefill_n > 0:
    import random
    random.seed(7)
    WORDS = ("system kernel matrix tensor stream buffer cache thread block warp "
             "quantum ledger harbor granite meadow copper falcon timber orchid velvet").split()
    def make_prompt(n_words, salt):
        return f"[doc {salt}] " + " ".join(random.choice(WORDS) for _ in range(n_words)) + " Summarize:"
    # shape sweep for ftl-spike hunt: two repeats of each length
    sweep = []
    for L in (400, 1200, 2400):
        for rep in range(2):
            pr = make_prompt(L, f"s{L}r{rep}")
            nt = tok_count(pr)
            ntok, dt, ftl, _ = stream_one(pr, 4)
            sweep.append({"len_tokens": nt, "rep": rep, "ftl": ftl, "prefill_tps": nt/ftl})
            print(f"[sweep len={nt} rep={rep}] ftl={ftl:.3f}s prefill={nt/ftl:.1f} tok/s", flush=True)
    # main prefill: n distinct ~2400-word prompts
    pf = []
    for i in range(args.prefill_n):
        pr = make_prompt(2400, f"main{i}")
        nt = tok_count(pr)
        ntok, dt, ftl, _ = stream_one(pr, 4)
        pf.append({"i": i, "prompt_tokens": nt, "ftl": ftl, "prefill_tps": nt/ftl})
        print(f"[prefill {i}] {nt} tok prompt, ftl={ftl:.3f}s = {nt/ftl:.1f} tok/s", flush=True)
    ptps = [x["prefill_tps"] for x in pf]
    result["prefill"] = {"n": len(ptps), "median": statistics.median(ptps),
        "mean": statistics.mean(ptps), "sd": statistics.stdev(ptps) if len(ptps)>1 else 0,
        "min": min(ptps), "max": max(ptps), "runs": pf, "sweep": sweep}
    print(f"PREFILL: median {statistics.median(ptps):.1f} mean {statistics.mean(ptps):.1f} "
          f"sd {statistics.stdev(ptps):.1f} min {min(ptps):.1f} max {max(ptps):.1f} n={len(ptps)}", flush=True)

if args.mode in ("conc", "all"):
    result["conc"] = {}
    for c in [int(x) for x in args.conc.split(",")]:
        out = [None] * c
        def worker(k):
            pr = PROMPTS[k % len(PROMPTS)] + f" (variant {k})"
            try:
                ntok, dt, ftl, _ = stream_one(pr, args.tokens)
                out[k] = {"ntok": ntok, "secs": dt, "ftl": ftl, "tps": ntok / dt}
            except Exception as e:
                out[k] = {"error": str(e)}
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(c)]
        t0 = time.time()
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.time() - t0
        ok = [o for o in out if o and "error" not in o]
        total_tok = sum(o["ntok"] for o in ok)
        agg = total_tok / wall
        per = [o["tps"] for o in ok]
        result["conc"][str(c)] = {
            "wall": wall, "total_tokens": total_tok, "aggregate_tps": agg,
            "per_stream_mean": statistics.mean(per) if per else 0,
            "per_stream_min": min(per) if per else 0,
            "per_stream_max": max(per) if per else 0,
            "streams_ok": len(ok), "streams": out}
        print(f"CONC {c}: total {total_tok} tok in {wall:.2f}s = {agg:.2f} agg tok/s; "
              f"per-stream mean {statistics.mean(per):.2f} "
              f"[{min(per):.2f}-{max(per):.2f}] ok={len(ok)}/{c}", flush=True)

with open(args.out, "w") as f:
    json.dump(result, f)
print("CLIENT_DONE", flush=True)
