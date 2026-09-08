"""Sidecar gate harness: short-prefill probe + N-prompt greedy capture/compare.

  gate.py probe   <port> <model>              raw ptok=1 / ptok=5 / ptok=13 completions
  gate.py capture <port> <model> <tag> [n]    n greedy 128-token completions -> JSON
  gate.py compare <tagA> <tagB>               byte-identity + first-token agreement
"""
import json, sys, os, random, urllib.request

OUT = os.environ.get("PXA_PASCALOPS_RESULTS", "./results")
os.makedirs(OUT, exist_ok=True)


def post(port, body, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def probe(port, model):
    # The three prompt lengths that bracket the captured-graph short-prefill defect.
    # ptok<=8 replays a captured decode graph with zero computed context; before the
    # guard, ptok=1 returned "!!!!" and ptok=5 returned token salad, while ptok=13
    # (which runs eager) was always clean. All three must be clean now.
    cases = [("Hello", 1), ("The capital of France is", 5),
             ("Write one sentence about the history of the printing press in Europe", 13)]
    bad = 0
    for text, approx in cases:
        j = post(port, {"model": model, "prompt": text, "max_tokens": 24,
                        "temperature": 0})
        out = j["choices"][0]["text"]
        ptok = j["usage"]["prompt_tokens"]
        flag = ""
        if out.strip().strip("!").strip() == "" or set(out.strip()) <= set("!"):
            flag = "  <-- DEGENERATE"
            bad += 1
        print(f"ptok={ptok} (~{approx})  {out!r}{flag}", flush=True)
    print("PROBE " + ("FAIL" if bad else "PASS"), flush=True)
    return 1 if bad else 0


def prompts(n):
    random.seed(20260905)
    topics = ["distributed consensus", "cellular respiration", "the French Revolution",
              "garbage collection in runtimes", "plate tectonics", "option pricing",
              "birdsong acquisition", "the CAP theorem", "photolithography",
              "medieval trade routes"]
    seeds = ["Explain {} in detail, covering history, mechanisms, and open problems. ",
             "Write a comprehensive tutorial about {}. Start from first principles. "]
    out = []
    for i, t in enumerate(topics):
        for j, s in enumerate(seeds):
            out.append(f"[q{i}-{j}] " + s.format(t) + "\nNow begin your answer:\n")
    return out[:n]


def capture(port, model, tag, n=20):
    res = {}
    for i, p in enumerate(prompts(n)):
        j = post(port, {"model": model, "prompt": p, "max_tokens": 128,
                        "temperature": 0})
        res[str(i)] = j["choices"][0]["text"]
        if i % 5 == 0:
            print(f"  {i}/{n}", flush=True)
    json.dump(res, open(f"{OUT}/{tag}.json", "w"))
    print(f"captured {n} -> {tag}.json", flush=True)


def capture_np(port, model, tag, n=20, np_=2):
    """Same 20 prompts, issued np_ at a time.

    np=2 is not a duplicate of np=1: with two sequences in flight the scheduler batches
    them into one forward pass, so the norms see M=2 rows instead of M=1 and land on a
    different captured graph. A fused kernel whose fold order is a function of the shape
    therefore has to be checked at BOTH, which is why the campaign gate says np1 AND np2.
    """
    import concurrent.futures as cf
    ps = prompts(n)
    res = {}
    with cf.ThreadPoolExecutor(max_workers=np_) as ex:
        futs = {ex.submit(post, port, {"model": model, "prompt": p,
                                       "max_tokens": 128, "temperature": 0}): i
                for i, p in enumerate(ps)}
        for f in cf.as_completed(futs):
            res[str(futs[f])] = f.result()["choices"][0]["text"]
    json.dump(res, open(f"{OUT}/{tag}.json", "w"))
    print(f"captured {n} at np={np_} -> {tag}.json", flush=True)


def speed(port, model, label, cell, ptok=None, ntok=128, n=3, conc=1):
    """Relative decode / prefill cells. Reports the median of n runs.

    These are RELATIVE numbers for an A0-vs-A1 comparison inside one window on one pair,
    which is all the acceptance needs; they are not board numbers.
    """
    import statistics as st
    import time
    import concurrent.futures as cf
    prompt = ("Explain distributed consensus in detail. " * (ptok // 6)) if ptok \
        else "Explain distributed consensus in detail.\nNow begin your answer:\n"
    vals = []
    for _ in range(n):
        t0 = time.time()
        if conc == 1:
            j = post(port, {"model": model, "prompt": prompt, "max_tokens": ntok,
                            "temperature": 0})
            got = j["usage"]["completion_tokens"]
        else:
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                fs = [ex.submit(post, port, {"model": model, "prompt": prompt,
                                             "max_tokens": ntok, "temperature": 0})
                      for _ in range(conc)]
                got = sum(f.result()["usage"]["completion_tokens"] for f in fs)
        vals.append(got / (time.time() - t0))
    m = st.median(vals)
    print(f"CELL {label} {cell} median {m:.2f} tok/s  runs "
          + " ".join(f"{v:.2f}" for v in vals), flush=True)
    return m


def compare(a, b):
    A = json.load(open(f"{OUT}/{a}.json"))
    B = json.load(open(f"{OUT}/{b}.json"))
    keys = sorted(set(A) & set(B), key=int)
    exact = sum(A[k] == B[k] for k in keys)
    first = sum(bool(A[k].split()) and bool(B[k].split())
                and A[k].split()[0] == B[k].split()[0] for k in keys)
    print(f"{a} vs {b}: byte-identical {exact}/{len(keys)}, "
          f"same first token {first}/{len(keys)}")
    for k in keys:
        if A[k] != B[k]:
            print(f"  DIFF q{k}: A={A[k][:70]!r}\n            B={B[k][:70]!r}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "probe":
        sys.exit(probe(sys.argv[2], sys.argv[3]))
    if c == "capture_np":
        capture_np(sys.argv[2], sys.argv[3], sys.argv[4],
                   int(sys.argv[5]), int(sys.argv[6]))
    elif c == "speed":
        speed(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
              ptok=int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None,
              ntok=int(sys.argv[7]) if len(sys.argv) > 7 else 128,
              n=int(sys.argv[8]) if len(sys.argv) > 8 else 3,
              conc=int(sys.argv[9]) if len(sys.argv) > 9 else 1)
    elif c == "capture":
        capture(sys.argv[2], sys.argv[3], sys.argv[4],
                int(sys.argv[5]) if len(sys.argv) > 5 else 20)
    elif c == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
