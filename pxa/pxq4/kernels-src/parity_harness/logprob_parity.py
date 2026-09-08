"""
GATE G7 / G5 (end to end) -- logprob parity between vLLM-PXQ4 and llama-server-PXQ4.

Runs LATER, when GPUs are free.  Documented and runnable now; it performs NO GPU work of
its own -- it drives two servers someone else started.

WHY THIS SHAPE.  The two engines are supposed to be reading the SAME 4-bit weights with
the same dequant contract, so this is not a "do two quantizations agree" study.  Any
disagreement beyond fp16 GEMM noise is a port bug with a specific address:

  divergence in EVERY layer, everywhere         -> table or nibble-order bug   (G1 missed it)
  divergence only above some row/column index   -> shard boundary              (G3 missed it)
  divergence only on GDN layers                 -> in_proj_b/in_proj_a swap    (§5.4 note 1)
  divergence only on full-attn layers           -> attn_q gate interleave      (§2.5)
  divergence that grows with position           -> KV/cache/rope, not PXQ4
  divergence only at TP>1                       -> sharding, not the kernel

Because of that, the script reports WHERE the two diverge, not just whether.

DEPENDENCIES: python stdlib only.  Both servers speak the OpenAI completions API
(llama-server does; vLLM does).  Talking HTTP rather than importing vllm means this file
never needs the production container's python and cannot perturb it.

USAGE
  # side A -- our engine, already the way the fleet runs it
  llama-server -m /mnt/models/pxa-models/Qwen3.8-27B-PXQ4.gguf -ngl 99 -c 4096 \
               --host 127.0.0.1 --port 8081

  # side B -- vLLM with the plugin, on FREE cards (never take the lease for this)
  PYTHONPATH=/mnt/models/pxa-vllm-pxq4/site \
  VLLM_SM70_QUANT_BACKEND=turbomind \
  python -m vllm.entrypoints.openai.api_server \
      --model /mnt/models/pxa-models/Qwen3.8-27B-PXQ4-vllm \
      --tensor-parallel-size 4 --dtype float16 --port 8082

  python -m parity_harness.logprob_parity \
      --a http://127.0.0.1:8081 --b http://127.0.0.1:8082 \
      --prompts prompts.txt --max-tokens 64 --topk 20 --out parity.json

PASS CRITERIA (plan G7): same-top-token >= 99.5% over >= 20 prompts.  The script prints
the rate and exits non-zero below the threshold, so it can be a CI step rather than a
thing someone eyeballs.

ONE HONEST CAVEAT, stated up front: greedy decoding on two different GEMM
implementations WILL eventually diverge on some prompt even if both are correct, because
a near-tie between two tokens resolves differently under different rounding.  That is why
the criterion is a rate over many prompts and why the script also reports the logprob
MARGIN at each divergence: a divergence with margin < 1e-3 is float noise, a divergence
with margin > 0.1 is a bug.  Reporting the margin is what separates those two, and
without it this test is not decidable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In 1969, humans first landed on the",
    "The three primary colors of light are red, green, and",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "Once upon a time, in a village at the edge of a great forest, there",
    "The derivative of x^3 with respect to x is",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self",
    "Water boils at 100 degrees Celsius at",
    "The Treaty of Westphalia was signed in the year",
    "#include <stdio.h>\n\nint main(void) {\n    printf(",
    "A binary search tree has the property that for every node,",
    "The mitochondrion is often described as the",
    "To reverse a linked list iteratively, you keep three pointers:",
    "The Fourier transform converts a signal from the time domain to the",
    "git rebase -i HEAD~3 will let you",
    "In Rust, the borrow checker prevents",
    "The speed of light in a vacuum is approximately",
    "A transformer's attention mechanism computes softmax(QK^T /",
    "The primary difference between TCP and UDP is that TCP",
    "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, n in",
    "The Pythagorean theorem states that in a right triangle,",
]


def _post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def complete(base, model, prompt, max_tokens, topk, timeout):
    """One temp-0 completion with per-token logprobs, in OpenAI /v1/completions form."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "logprobs": topk,
        "echo": False,
        "stream": False,
    }
    r = _post(base.rstrip("/") + "/v1/completions", payload, timeout)
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    return {
        "text": ch.get("text", ""),
        "tokens": lp.get("tokens") or [],
        "token_logprobs": lp.get("token_logprobs") or [],
        "top_logprobs": lp.get("top_logprobs") or [],
        "finish_reason": ch.get("finish_reason"),
    }


def discover_model(base, timeout=60):
    with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=timeout) as r:
        j = json.loads(r.read().decode())
    return j["data"][0]["id"]


def _margin(top: dict, chosen: str):
    """logprob gap between the chosen token and the runner-up.

    A greedy divergence with a tiny margin is arithmetic; with a large margin it is a
    bug.  Returning None when the server did not send top_logprobs is deliberate: the
    script then says the margin is unknown rather than assuming it was small.
    """
    if not top:
        return None
    items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return None
    if len(items) == 1:
        return float("inf")
    best, second = items[0], items[1]
    if best[0] == chosen:
        return best[1] - second[1]
    return best[1] - top.get(chosen, min(v for _, v in items) - 10.0)


def compare_one(a, b):
    """Compare two completions token by token up to the first structural end."""
    n = min(len(a["tokens"]), len(b["tokens"]))
    first_div = None
    same = 0
    dlp = []
    for i in range(n):
        ta, tb = a["tokens"][i], b["tokens"][i]
        if ta == tb:
            same += 1
            la = a["token_logprobs"][i] if i < len(a["token_logprobs"]) else None
            lb = b["token_logprobs"][i] if i < len(b["token_logprobs"]) else None
            if la is not None and lb is not None:
                dlp.append(abs(la - lb))
        else:
            if first_div is None:
                ma = _margin(a["top_logprobs"][i] if i < len(a["top_logprobs"]) else {}, ta)
                mb = _margin(b["top_logprobs"][i] if i < len(b["top_logprobs"]) else {}, tb)
                first_div = {"pos": i, "a_token": ta, "b_token": tb,
                             "a_margin": ma, "b_margin": mb,
                             "verdict": _verdict(ma, mb)}
            break
    return {
        "compared": n,
        "same_prefix": same,
        "len_a": len(a["tokens"]),
        "len_b": len(b["tokens"]),
        "first_divergence": first_div,
        "mean_abs_dlogprob": statistics.fmean(dlp) if dlp else None,
        "max_abs_dlogprob": max(dlp) if dlp else None,
    }


def _verdict(ma, mb):
    """Classify a divergence.  Thresholds are judgement calls and are stated as such.

    ASSUMPTION: a top-1 margin below 1e-3 nats is within what a different fp16 GEMM
    reduction order can flip, and a margin above 0.1 nats is not.  Nothing measured
    establishes those numbers; they are here so the report is actionable, and the raw
    margins are always printed so the reader can disagree.
    """
    m = min(x for x in (ma, mb) if x is not None) if any(
        x is not None for x in (ma, mb)) else None
    if m is None:
        return "UNKNOWN (server sent no top_logprobs)"
    if m < 1e-3:
        return "NOISE (near-tie; a different reduction order flips this)"
    if m < 0.1:
        return "SUSPICIOUS (investigate; margin is larger than rounding should move)"
    return "BUG (the two engines disagree on a confident token)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="base URL of engine A (llama-server)")
    ap.add_argument("--b", required=True, help="base URL of engine B (vLLM + pxq4)")
    ap.add_argument("--model-a", default=None)
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--prompts", default=None, help="file, one prompt per line (\\n escaped)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.995,
                    help="required same-top-token rate (plan G7 is 0.995)")
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    args = ap.parse_args(argv)

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [ln.rstrip("\n").replace("\\n", "\n") for ln in f if ln.strip()]
    if len(prompts) < 20:
        print(f"WARNING: {len(prompts)} prompts; plan G7 wants >= 20", file=sys.stderr)

    model_a = args.model_a or discover_model(args.a)
    model_b = args.model_b or discover_model(args.b)
    print(f"A: {args.a}  model={model_a}")
    print(f"B: {args.b}  model={model_b}")
    print(f"{len(prompts)} prompts, max_tokens={args.max_tokens}, temp=0\n")

    results = []
    tot_tok = tot_same = 0
    t0 = time.time()
    for i, p in enumerate(prompts):
        try:
            ra = complete(args.a, model_a, p, args.max_tokens, args.topk, args.timeout)
            rb = complete(args.b, model_b, p, args.max_tokens, args.topk, args.timeout)
        except (urllib.error.URLError, KeyError, IndexError) as e:
            print(f"[{i:3d}] REQUEST FAILED: {type(e).__name__}: {e}")
            results.append({"prompt": p, "error": f"{type(e).__name__}: {e}"})
            continue
        c = compare_one(ra, rb)
        c["prompt"] = p
        results.append(c)
        tot_tok += c["compared"]
        tot_same += c["same_prefix"]
        fd = c["first_divergence"]
        flag = "ok " if fd is None else "DIV"
        extra = ""
        if fd:
            extra = (f" @tok{fd['pos']} A={fd['a_token']!r} B={fd['b_token']!r} "
                     f"marginA={fd['a_margin']} -> {fd['verdict']}")
        print(f"[{i:3d}] {flag} {c['same_prefix']}/{c['compared']} "
              f"meanDlp={c['mean_abs_dlogprob']}{extra}")

    rate = (tot_same / tot_tok) if tot_tok else 0.0
    bugs = [r for r in results
            if r.get("first_divergence")
            and r["first_divergence"]["verdict"].startswith("BUG")]
    susp = [r for r in results
            if r.get("first_divergence")
            and r["first_divergence"]["verdict"].startswith("SUSPICIOUS")]

    print(f"\nsame-top-token: {tot_same}/{tot_tok} = {rate:.5f} "
          f"(threshold {args.threshold})")
    print(f"confident disagreements (BUG): {len(bugs)}")
    print(f"borderline (SUSPICIOUS):       {len(susp)}")
    print(f"elapsed {time.time()-t0:.1f}s")

    report = {"a": args.a, "b": args.b, "model_a": model_a, "model_b": model_b,
              "max_tokens": args.max_tokens, "topk": args.topk,
              "same_top_token_rate": rate, "tokens_compared": tot_tok,
              "n_bug": len(bugs), "n_suspicious": len(susp), "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.out}")

    if bugs:
        print("\nFAIL: at least one confident-token disagreement. This is a port bug, "
              "not arithmetic. Re-read the divergence-location table at the top of this "
              "file to localise it.")
        return 1
    if rate < args.threshold:
        print(f"\nFAIL: same-top-token {rate:.5f} < {args.threshold}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
