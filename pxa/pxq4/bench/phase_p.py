#!/usr/bin/env python3
"""Phase P promotion gate for the fused Pascal small-op pack (main's #1075 ruling).

WHY THIS IS NOT A BYTE GATE. The pack changes the ORDER of an fp32 reduction. The order it
replaces -- torch's `x.pow(2).mean(-1)` -- is itself one arbitrary accumulation order chosen
per shape by TensorIterator, not a reference answer, so demanding byte-identity against it
demands agreement with an arbitrary choice rather than correctness. A probe over 366
candidate layouts established that no single kernel geometry reproduces torch at more than
one shape and that one decode shape is reproduced by none, so that demand is also
unsatisfiable. Byte-identity is therefore RECORDED here as a quality row and is not what
decides promotion.

What decides promotion, all against the FUSED=0 control on the same lib in one quiet cell
(2026-09-06, Board C15: see PROMOTION_CRITERIA below -- byte identity arm-vs-control at matched
np is GATING; np=2 run-to-run determinism is RECORDED, not gating, because the control itself is
2/12 on the P100 pair, bug sm60-vllm-np2-batch-invariance):

  1. RUN-TO-RUN DETERMINISM, 12/12 at np=1 (gating) and at np=2 (recorded). The fused arm must return
     byte-identical output for the same prompt every time. This is the check that catches a
     race, an uninitialised buffer, or a fold whose result depends on block scheduling --
     the failure modes that a single A-vs-B comparison cannot see at all, because both
     sides of it would be wrong in the same run.
  2. DECISIVE-PROMPT TOP-TOKEN AGREEMENT. For every prompt whose top-2 logprob gap exceeds
     DECISIVE_NATS (default 0.5), the arms must choose the same token, 100% of the time. A
     wider margin than the perturbation means a disagreement there is not rounding, it is a
     defect. Prompts below the threshold are reported and excluded, because a near-tie
     flipping is exactly what a last-bit change is expected to do and gating on it would be
     gating on noise.
  3. NEEDLE RECALL at ~6.5k tokens: both planted identifiers recovered, in both arms.
  4. SHORT-PREFILL PROBE: ptok 1 / 5 / 13 clean in both arms (the captured-graph
     short-prefill defect class).

Usage:
  phase_p.py determinism <port> <model> <tag> [reps] [np]
  phase_p.py decisive    <port> <model> <tag> [n]
  phase_p.py needle      <port> <model> <tag>
  phase_p.py verdict     <tagA> <tagB>
"""
import json
import math
import os
import sys
import urllib.request

OUT = os.environ.get("PXA_PASCALOPS_RESULTS", "./results")
os.makedirs(OUT, exist_ok=True)
DECISIVE_NATS = float(os.environ.get("DECISIVE_NATS", "0.5"))


def post(port, body, timeout=1200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def save(tag, obj):
    json.dump(obj, open(f"{OUT}/{tag}.json", "w"))
    print(f"wrote {tag}.json", flush=True)


# ------------------------------------------------------------------ check 1 ---
def determinism(port, model, tag, reps=12, np_=1):
    """The same prompt `reps` times; every output must be byte-identical.

    At np=2 the requests are issued two at a time so the scheduler batches them, which puts
    the norms on a different captured graph (M=2 rather than M=1) -- a fold whose result
    depended on block scheduling would show up here and nowhere else.
    """
    import concurrent.futures as cf
    prompt = ("Explain, in careful detail, how a write-after-read race inside a fused "
              "kernel can produce a stable greedy output while still changing the logits "
              "on every run.\nAnswer:\n")
    body = {"model": model, "prompt": prompt, "max_tokens": 96, "temperature": 0}
    outs = []
    if np_ == 1:
        for _ in range(reps):
            outs.append(post(port, body)["choices"][0]["text"])
    else:
        with cf.ThreadPoolExecutor(max_workers=np_) as ex:
            for i in range(0, reps, np_):
                fs = [ex.submit(post, port, body) for _ in range(min(np_, reps - i))]
                outs += [f.result()["choices"][0]["text"] for f in fs]
    same = sum(o == outs[0] for o in outs)
    print(f"DETERMINISM np={np_}: {same}/{len(outs)} identical "
          f"{'PASS' if same == len(outs) else 'FAIL'}", flush=True)
    save(f"{tag}-det-np{np_}", {"reps": outs, "same": same, "n": len(outs)})
    return same == len(outs)


# ------------------------------------------------------------------ check 2 ---
def decisive(port, model, tag, n=20):
    """Token 0 with its top-2 logprobs, so agreement can be judged against the margin."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gate import prompts
    rows = {}
    for i, p in enumerate(prompts(n)):
        j = post(port, {"model": model, "prompt": p, "max_tokens": 1,
                        "temperature": 0, "logprobs": 5})
        ch = j["choices"][0]
        lp = ch.get("logprobs") or {}
        top = (lp.get("top_logprobs") or [{}])[0]
        tok = (lp.get("tokens") or [ch["text"]])[0]
        ordered = sorted(top.values(), reverse=True) if top else []
        gap = (ordered[0] - ordered[1]) if len(ordered) >= 2 else float("inf")
        rows[str(i)] = {"token": tok, "gap": gap}
    save(f"{tag}-decisive", rows)
    dec = sum(1 for r in rows.values() if r["gap"] > DECISIVE_NATS)
    print(f"decisive prompts (top-2 gap > {DECISIVE_NATS} nats): {dec}/{len(rows)}",
          flush=True)
    return rows


# ------------------------------------------------------------------ check 3 ---
def server_max_len(port):
    """vLLM: /v1/models max_model_len; llama.cpp: /props n_ctx; None when neither answers."""
    for url, pick in ((f"http://127.0.0.1:{port}/v1/models",
                       lambda j: (j.get("data") or [{}])[0].get("max_model_len")),
                      (f"http://127.0.0.1:{port}/props",
                       lambda j: (j.get("default_generation_settings") or {}).get("n_ctx"))):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as r:
                v = pick(json.loads(r.read()))
            if v:
                return int(v)
        except Exception:
            continue
    return None


def needle(port, model, tag, target=6500):
    """Two identifiers planted near the start and middle of a prompt sized UNDER the server's
    max_model_len: min(target, max_model_len - 512) tokens (order #1288-5 / Board C10). The old
    fixed filler*220*2 body was ~9.7k tokens against an 8192 server and came back HTTP 400."""
    a, b = "ZEPHYR-40213", "QUARTZ-88571"
    filler = ("The archive records routine maintenance of the northern relay stations. "
              "Nothing of consequence occurred during this interval. ")
    mml = server_max_len(port)
    want = min(target, mml - 512) if mml else target
    # one cheap 1-token request measures the filler's token cost on THIS tokenizer
    probe = post(port, {"model": model, "prompt": filler * 20, "max_tokens": 1, "temperature": 0})
    per_rep = max(1.0, probe["usage"]["prompt_tokens"] / 20.0)
    reps = max(4, int((want - 60) / per_rep / 2))
    body = (f"Document begins. Access identifier {a} was issued at intake.\n"
            + filler * reps
            + f"\nMid-document note: the secondary identifier is {b}.\n"
            + filler * reps
            + "\nQuestion: list both identifiers that appear in this document.\nAnswer:\n")
    print(f"NEEDLE sizing: max_model_len={mml} target={want} tokens -> filler x{reps} x2 "
          f"({per_rep:.1f} tok/rep)", flush=True)
    j = post(port, {"model": model, "prompt": body, "max_tokens": 48, "temperature": 0})
    out = j["choices"][0]["text"]
    ptok = j["usage"]["prompt_tokens"]
    ok = (a in out) and (b in out)
    print(f"NEEDLE ptok={ptok}: both recalled = {ok}  {'PASS' if ok else 'FAIL'}  "
          f"{out[:120]!r}", flush=True)
    save(f"{tag}-needle", {"ptok": ptok, "out": out, "pass": ok})
    return ok


# ------------------------------------------------------------------ verdict ---
PROMOTION_CRITERIA = """\
PROMOTION CRITERIA (Board C15, order #1288-1, written down 2026-09-06):
  A vLLM pack is promoted to a default ONLY on, all in one quiet cell, same lib, arm vs the
  =0 control:
    1. BYTE IDENTITY arm-vs-control at MATCHED np: 20/20 at np=1 AND 20/20 at np=2
       (gate.py capture / capture_np -> <tag>-np1.json / <tag>-np2.json)      -- GATING
    2. DECISIVE TOP-TOKEN agreement 100% on prompts with top-2 gap > DECISIVE_NATS -- GATING
    3. NEEDLE recall (both identifiers) in BOTH arms, sized under max_model_len   -- GATING
    4. RUN-TO-RUN DETERMINISM at np=1: 12/12 in BOTH arms                          -- GATING
    5. RUN-TO-RUN DETERMINISM at np=2: RECORDED, NOT GATING. The control itself is 2/12 on
       the P100 pair (bug sm60-vllm-np2-batch-invariance: a pre-existing engine property of
       batched decode on sm_60, tracked on its own). Gating the pack on it would gate on the
       engine, not the pack; the arm-vs-control byte gate at np=2 (criterion 1) is what
       proves the pack adds nothing to it.
  A zero-arithmetic change that moves a greedy sha is a bug, not noise.
"""


def verdict(a, b):
    def load(t):
        try:
            return json.load(open(f"{OUT}/{t}.json"))
        except FileNotFoundError:
            return None

    print(f"\n===== PHASE P VERDICT: {a} (control) vs {b} (fused) =====")
    print(PROMOTION_CRITERIA)
    ok = True

    # 1. byte identity arm vs control at matched np (gate.py captures)
    for np_ in (1, 2):
        A, B = load(f"{a}-np{np_}"), load(f"{b}-np{np_}")
        if A is None or B is None:
            print(f"  byte gate np={np_} {a} vs {b}: SKIP (no capture)"); ok = False; continue
        keys = sorted(set(A) & set(B), key=int)
        exact = sum(A[k] == B[k] for k in keys)
        p = len(keys) > 0 and exact == len(keys)
        print(f"  byte gate np={np_} {a} vs {b}: {exact}/{len(keys)} byte-identical "
              f"{'PASS' if p else 'FAIL'}  [GATING]")
        ok &= p

    # 4/5. run-to-run determinism: np=1 gating, np=2 recorded
    for np_ in (1, 2):
        for t in (a, b):
            d = load(f"{t}-det-np{np_}")
            tagline = "[GATING]" if np_ == 1 else "[RECORDED, not gating: bug sm60-vllm-np2-batch-invariance]"
            if d is None:
                print(f"  determinism np={np_} {t}: SKIP (no data) {tagline}")
                if np_ == 1:
                    ok = False
                continue
            p = d["same"] == d["n"]
            print(f"  determinism np={np_} {t}: {d['same']}/{d['n']} "
                  f"{'PASS' if p else 'FAIL'}  {tagline}")
            if np_ == 1:
                ok &= p

    ra, rb = load(f"{a}-decisive"), load(f"{b}-decisive")
    if ra and rb:
        keys = sorted(set(ra) & set(rb), key=int)
        dec = [k for k in keys if min(ra[k]["gap"], rb[k]["gap"]) > DECISIVE_NATS]
        agree = sum(ra[k]["token"] == rb[k]["token"] for k in dec)
        near = [k for k in keys if k not in dec]
        p = agree == len(dec) and len(dec) > 0
        print(f"  decisive top-token: {agree}/{len(dec)} agree "
              f"(margin > {DECISIVE_NATS} nats)  {'PASS' if p else 'FAIL'}")
        print(f"    excluded as near-ties ({len(near)}): "
              + ", ".join(f"q{k}:{min(ra[k]['gap'], rb[k]['gap']):.3f}" for k in near))
        for k in dec:
            if ra[k]["token"] != rb[k]["token"]:
                print(f"    DISAGREE q{k} margin {min(ra[k]['gap'], rb[k]['gap']):.3f}: "
                      f"{ra[k]['token']!r} vs {rb[k]['token']!r}")
        ok &= p
    else:
        print("  decisive top-token: SKIP (no data)"); ok = False

    for t in (a, b):
        nd = load(f"{t}-needle")
        if nd is None:
            print(f"  needle {t}: SKIP"); ok = False; continue
        print(f"  needle {t} (ptok {nd['ptok']}): "
              f"{'PASS' if nd['pass'] else 'FAIL'}")
        ok &= nd["pass"]

    print(f"\n  PHASE P: {'PASS -- promotable to default ON for sm_60' if ok else 'FAIL'}"
          f"  (criteria: byte gate np1+np2, decisive top-token, needle, np1 determinism)")
    return 0 if ok else 1


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "determinism":
        determinism(sys.argv[2], sys.argv[3], sys.argv[4],
                    int(sys.argv[5]) if len(sys.argv) > 5 else 12,
                    int(sys.argv[6]) if len(sys.argv) > 6 else 1)
    elif c == "decisive":
        decisive(sys.argv[2], sys.argv[3], sys.argv[4],
                 int(sys.argv[5]) if len(sys.argv) > 5 else 20)
    elif c == "needle":
        needle(sys.argv[2], sys.argv[3], sys.argv[4])
    elif c == "verdict":
        sys.exit(verdict(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
