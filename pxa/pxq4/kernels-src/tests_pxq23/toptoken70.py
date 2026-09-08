"""toptoken70.py -- same-top-token gate ACROSS ENGINES, plus an 8k needle.

The existing qualgate harness compares two captures from one engine and hardcodes the model
name, which is right for an arm-vs-arm gate on one seat and wrong here: the question this test
has to answer is whether the vLLM sidecar decoding PXQ2/PXQ3 agrees with the llama.cpp engine
decoding the SAME GGUF. So the model name is a parameter, both engines are captured the same
way, and the comparison is between the two captures.

WHAT IT MEASURES, and what a number here does and does not mean:
  first    the greedy FIRST token is identical. This is the number that matters: two engines
           with the same weights and the same greedy rule must pick the same argmax, and a
           decode bug shows up here immediately.
  prefix8  the first 8 tokens are identical. Divergence after token 1 is usually a tie broken
           differently, not a wrong weight.
  exact    all 32 tokens identical. Expected to be BELOW 100%% even for a correct port: the two
           engines use different attention kernels, different fp16 reduction orders and
           different KV layouts, so long greedy chains legitimately diverge at a near-tie.
           Reporting exact alone would understate a correct port; reporting first alone would
           miss a slow drift. Both are printed, and the gate is on `first`.

The needle is a separate, harder question: a long-context prompt whose answer is a token that
appeared once, 8k tokens earlier. It catches corrupted recurrent/DeltaNet state across ubatch
boundaries, which a top-token gate on 1k prompts cannot see.

Usage:
    python3 toptoken70.py capture --port 8261 --model fusion2-35b --tag llama-pxq2
    python3 toptoken70.py capture --port 8570 --model fusion2-35b-pxq2 --tag vllm-pxq2
    python3 toptoken70.py compare llama-pxq2 vllm-pxq2
    python3 toptoken70.py needle  --port 8570 --model fusion2-35b-pxq2 --fill 8192
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request

OUT_DIR = os.environ.get("PXQ23_RESULTS", os.environ.get("PXA_CAMPAIGN", "./speed-campaign") + "/logs/pxq23")


def gen_prompts(n: int = 70) -> list[str]:
    """70 deterministic prompts: 10 topics x 7 framings, ~900 filler words each so the prompt
    is long enough to exercise prefill and the linear-attention path, and seeded so the two
    engines see byte-identical input."""
    random.seed(20260905)
    topics = ["distributed consensus", "cellular respiration", "the French Revolution",
              "garbage collection in runtimes", "plate tectonics", "option pricing",
              "birdsong acquisition", "the CAP theorem", "photolithography",
              "medieval trade routes"]
    seeds = ["Explain {} in detail, covering history, mechanisms, and open problems. ",
             "Summarize the key debates about {} and give your own assessment. ",
             "Write an exam-style set of questions and worked answers on {}. ",
             "Describe common misconceptions about {} and correct each one. ",
             "Write a comprehensive tutorial about {}. Start from first principles. ",
             "Compare and contrast two schools of thought on {}. Be specific. ",
             "Draft detailed lecture notes on {} for graduate students. "]
    filler = ("context detail nuance mechanism structure dynamics analysis framework "
              "perspective evidence").split()
    out = []
    for i, t in enumerate(topics):
        for j, s in enumerate(seeds):
            pad = " ".join(random.choice(filler) for _ in range(900))
            out.append(f"[q{i}-{j}] " + s.format(t) + "Background notes: " + pad +
                       "\nNow begin your answer:\n")
    return out[:n]


def post(port: int, body: dict, host: str = "127.0.0.1", timeout: int = 900) -> dict:
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def top2_gap(choice) -> "float | None":
    """Top-1 minus top-2 logprob at the FIRST generated token, in nats, or None.

    This is what says whether a prompt is DECISIVE. A prompt where the model is 0.2 nats from
    changing its mind tells you nothing about two engines' decode fidelity -- it tells you how
    they round. Tolerant of shape differences between the two servers' logprobs blocks, because
    the alternative is a gate that silently degrades to "no gap available" on one engine.
    """
    lp = (choice or {}).get("logprobs") or {}
    tops = lp.get("top_logprobs")
    if not tops:
        # llama.cpp's OpenAI-compatible server (examples/server/server-task.cpp:135-141) returns
        # the CHAT-completions logprobs shape on /v1/completions as well:
        #   {"content": [{"token", "logprob", "top_logprobs": [{"token", "logprob"}, ...]}, ...]}
        # Until 2026-09-06 this function returned None for every reference capture, so the gate
        # sourced decisiveness from the CANDIDATE and printed "gap taken from the reference on 0"
        # (Board C13). The reference is the authority on whether a prompt is decisive.
        content = lp.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            tops = [content[0].get("top_logprobs") or []]
    if not tops or not tops[0]:
        return None
    first = tops[0]
    if isinstance(first, dict):
        vals = sorted(first.values(), reverse=True)
    elif isinstance(first, list):          # [{token, logprob}, ...]
        vals = sorted((e.get("logprob") for e in first if e.get("logprob") is not None),
                      reverse=True)
    else:
        return None
    return float(vals[0] - vals[1]) if len(vals) >= 2 else None


def first_tops(choice) -> "list | None":
    """The raw first-position top_logprobs, normalised to [{token, logprob}, ...], so a capture
    can be RE-SCORED later (rescore) if the decisiveness rule changes; None when absent."""
    lp = (choice or {}).get("logprobs") or {}
    tops = lp.get("top_logprobs")
    if not tops:
        content = lp.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            tops = [content[0].get("top_logprobs") or []]
    if not tops or not tops[0]:
        return None
    first = tops[0]
    if isinstance(first, dict):
        return [{"token": k, "logprob": v} for k, v in first.items()]
    if isinstance(first, list):
        return [{"token": e.get("token"), "logprob": e.get("logprob")} for e in first
                if isinstance(e, dict) and e.get("logprob") is not None]
    return None


def rescore(args) -> int:
    """Recompute every gap in a capture from its stored raw top_logprobs (CPU only)."""
    path = os.path.join(OUT_DIR, f"tt70-{args.tag}.json")
    d = json.load(open(path))
    n_re, n_missing = 0, 0
    for k, r in d["res"].items():
        tops = r.get("tops")
        if not tops:
            n_missing += 1
            continue
        vals = sorted((e["logprob"] for e in tops if e.get("logprob") is not None), reverse=True)
        r["gap"] = float(vals[0] - vals[1]) if len(vals) >= 2 else None
        n_re += 1
    json.dump(d, open(path, "w"))
    print(f"rescored {n_re} prompts in {path}; {n_missing} carried no raw logprobs "
          f"(captured before 2026-09-06 -- re-capture to get a reference-sourced gap)")
    return 0 if n_missing == 0 else 1


def load_prompts(args) -> list[str]:
    """Rendered prompts from a file if given, else the raw generator.

    A file is the supported path for the chat-template gate: both engines must be fed BYTE-
    IDENTICAL text, so the rendering happens once, offline (render_chat_prompts.py), and is
    not re-derived per engine.
    """
    if getattr(args, "prompts", None):
        d = json.load(open(args.prompts))
        return list(d["prompts"])[: args.n]
    return gen_prompts(args.n)


def capture(args) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {}
    prompts = load_prompts(args)
    for i, p in enumerate(prompts):
        body = {"model": args.model, "prompt": p, "max_tokens": args.max_tokens,
                "temperature": 0, "top_p": 1, "seed": 0}
        if args.logprobs:
            body["logprobs"] = args.logprobs
        j = post(args.port, body, args.host)
        c = j["choices"][0]
        res[str(i)] = {"text": c["text"], "ptok": j.get("usage", {}).get("prompt_tokens"),
                       "gap": top2_gap(c), "tops": first_tops(c)}
        if i % 10 == 0:
            print(f"  {i}/{len(prompts)}", flush=True)
    path = os.path.join(OUT_DIR, f"tt70-{args.tag}.json")
    json.dump({"model": args.model, "port": args.port, "n": len(prompts),
               "max_tokens": args.max_tokens,
               "prompts_file": getattr(args, "prompts", None), "res": res}, open(path, "w"))
    print(f"wrote {path}")
    return 0


def compare(a: str, b: str) -> int:
    A = json.load(open(os.path.join(OUT_DIR, f"tt70-{a}.json")))
    B = json.load(open(os.path.join(OUT_DIR, f"tt70-{b}.json")))
    ra, rb = A["res"], B["res"]
    keys = sorted(set(ra) & set(rb), key=int)
    if len(keys) != len(ra) or len(keys) != len(rb):
        print(f"WARNING: captures differ in size ({len(ra)} vs {len(rb)}); comparing "
              f"{len(keys)} common prompts")
    first = pre8 = exact = 0
    misses = []
    for k in keys:
        ta, tb = ra[k]["text"], rb[k]["text"]
        wa, wb = ta.split(), tb.split()
        f = bool(wa) and bool(wb) and wa[0] == wb[0]
        first += f
        pre8 += wa[:8] == wb[:8]
        exact += ta == tb
        if not f:
            misses.append((k, (wa[:6]), (wb[:6])))
    n = len(keys)
    print(f"{a} vs {b}  over {n} prompts")
    print(f"  first token   {first}/{n}  ({100.0*first/n:.1f}%)")
    print(f"  prefix8       {pre8}/{n}  ({100.0*pre8/n:.1f}%)")
    print(f"  exact 32tok   {exact}/{n}  ({100.0*exact/n:.1f}%)")
    for k, wa, wb in misses[:8]:
        print(f"  MISS q{k}: {a}={wa} | {b}={wb}")
    return 0 if first == n else 1


NEEDLE_PRE = ("You are reading a long technical log. Two identifiers appear exactly once each. "
              "Remember them.\nIDENTIFIER ALPHA IS {a}.\n")
NEEDLE_MID = "IDENTIFIER BETA IS {b}.\n"
NEEDLE_Q = ("\nEnd of log.\nQuestion: repeat IDENTIFIER ALPHA and IDENTIFIER BETA exactly, "
            "in that order, and write nothing else.\nAnswer:")


def needle(args) -> int:
    """One identifier at the very start and one in the middle of a ~fill-token prompt.

    Placed at BOTH positions on purpose: a start-only needle is survivable by a model whose
    recurrent state is corrupted only in later ubatches, and a middle-only needle is
    survivable by one that never carries state past the first chunk. Requiring both catches
    each failure. The filler is deterministic word salad, which is why the check is a substring
    test on the two identifiers and not a hash of the answer.
    """
    random.seed(4242)
    a, b = "QX7431KM", "ZR9028WD"
    filler = ("record entry timestamp module handler buffer segment pointer offset checksum "
              "retry latency throughput backlog partition replica quorum leader follower").split()
    words_per_tok = 0.75          # rough; the log prints the server's own prompt_tokens
    n_words = int(args.fill * words_per_tok)
    half = n_words // 2
    body = (NEEDLE_PRE.format(a=a)
            + " ".join(random.choice(filler) for _ in range(half)) + "\n"
            + NEEDLE_MID.format(b=b)
            + " ".join(random.choice(filler) for _ in range(n_words - half)) + "\n"
            + NEEDLE_Q)
    # MAX_TOKENS IS NOT ARBITRARY. This model emits a <think> block before answering, and at
    # 48 tokens the whole budget went to reasoning -- the reference engine "failed" a needle it
    # had plainly understood. A needle that never reaches the answer measures the token budget,
    # not the context. 320 clears the think block with room to spare, and both engines get the
    # identical budget so the comparison is unaffected.
    j = post(args.port, {"model": args.model, "prompt": body, "max_tokens": args.max_tokens,
                         "temperature": 0, "top_p": 1, "seed": 0}, args.host)
    text = j["choices"][0]["text"]
    # The identifiers may land after </think>; searching the whole completion is deliberate.
    answer = text.split("</think>")[-1]
    ptok = j.get("usage", {}).get("prompt_tokens")
    ok_a, ok_b = a in text, b in text
    ok_order = (a in answer and b in answer
                and answer.index(a) < answer.index(b)) if (ok_a and ok_b) else False
    print(f"needle fill~{args.fill} (server counted {ptok} prompt tokens)")
    print(f"  ALPHA {a}: {'FOUND' if ok_a else 'MISSING'}")
    print(f"  BETA  {b}: {'FOUND' if ok_b else 'MISSING'}")
    print(f"  order ALPHA-before-BETA in the answer: {ok_order}")
    print(f"  answer: {answer.strip()[:200]!r}")
    return 0 if (ok_a and ok_b) else 1


#: THE PRE-REGISTERED GATE (main, a pre-registered gate, fixed 2026-09-05 BEFORE the re-capture).
#: Written here rather than in a mail so it cannot drift between registration and reporting.
#:
#:   DECISIVE  a prompt whose top-2 logprob gap at the first generated token exceeds
#:             GATE_GAP_NATS. Below that the model is close to changing its mind, and which
#:             way two engines round is not a fact about their decode.
#:   GATE A    top-1 agreement on EVERY decisive prompt. 100%, no allowance.
#:   GATE B    of the prompts where both engines agree on token 1, at least GATE_B_N must have
#:             byte-identical FULL outputs. Agreement on one token is cheap; agreeing for 32
#:             tokens is not, and it is what says the two engines are running the same weights.
#:
#: Both must pass. Gap is taken from the REFERENCE capture where it has one (it is the
#: authority on whether a prompt is decisive) and from the candidate otherwise; which was used
#: is printed per prompt count, never silently.
GATE_GAP_NATS = 0.5
GATE_B_N = 20


def gate(a: str, b: str, gap_nats: float = GATE_GAP_NATS, need_exact: int = GATE_B_N) -> int:
    A = json.load(open(os.path.join(OUT_DIR, f"tt70-{a}.json")))
    B = json.load(open(os.path.join(OUT_DIR, f"tt70-{b}.json")))
    ra, rb = A["res"], B["res"]
    keys = sorted(set(ra) & set(rb), key=int)

    if A.get("prompts_file") != B.get("prompts_file"):
        print(f"REFUSING TO GATE: the two captures used different prompt files "
              f"({A.get('prompts_file')!r} vs {B.get('prompts_file')!r}). The whole point of "
              f"rendering once is that both engines see byte-identical text.")
        return 2
    ptok_mismatch = [k for k in keys
                     if ra[k].get("ptok") is not None and rb[k].get("ptok") is not None
                     and ra[k]["ptok"] != rb[k]["ptok"]]
    if ptok_mismatch:
        print(f"REFUSING TO GATE: prompt_tokens differ on {len(ptok_mismatch)} prompts "
              f"(e.g. q{ptok_mismatch[0]}). The engines are not tokenizing the same text, so "
              f"any output comparison is meaningless.")
        return 2

    def tok1(t: str):
        w = t.split()
        return w[0] if w else None

    decisive, from_ref, undecided, nogap = [], 0, [], 0
    for k in keys:
        g = ra[k].get("gap")
        src_ref = g is not None
        if g is None:
            g = rb[k].get("gap")
        if g is None:
            nogap += 1
            continue
        (decisive if g > gap_nats else undecided).append(k)
        from_ref += src_ref and g is not None

    agree1 = [k for k in keys if tok1(ra[k]["text"]) == tok1(rb[k]["text"])]
    exact = [k for k in agree1 if ra[k]["text"] == rb[k]["text"]]
    dec_agree = [k for k in decisive if tok1(ra[k]["text"]) == tok1(rb[k]["text"])]

    print(f"PRE-REGISTERED GATE  {a} vs {b}   ({len(keys)} prompts)")
    print(f"  prompts with no logprobs from either engine : {nogap}")
    print(f"  decisive (top-2 gap > {gap_nats} nats)      : {len(decisive)}"
          f"   (gap taken from the reference on {from_ref})")
    print(f"  near-tie, excluded by the gate              : {len(undecided)}")
    print(f"  GATE A  top-1 agreement on decisive prompts : {len(dec_agree)}/{len(decisive)}")
    print(f"  token-1 agreement overall (not gated)       : {len(agree1)}/{len(keys)}")
    print(f"  GATE B  byte-identical full outputs among   : {len(exact)} "
          f"(need >= {need_exact})")
    ok_a = len(decisive) > 0 and len(dec_agree) == len(decisive)
    ok_b = len(exact) >= need_exact
    if not ok_a:
        for k in [x for x in decisive if x not in dec_agree][:8]:
            print(f"    GATE A MISS q{k} (gap {ra[k].get('gap') or rb[k].get('gap'):.3f}): "
                  f"{a}={tok1(ra[k]['text'])!r} {b}={tok1(rb[k]['text'])!r}")
    if len(decisive) == 0:
        print("  GATE A CANNOT BE EVALUATED: no prompt was decisive. That is a defect in the "
              "prompt set, not a pass -- do not report it as one.")
    print(f"\n  GATE A: {'PASS' if ok_a else 'FAIL'}    GATE B: {'PASS' if ok_b else 'FAIL'}")
    return 0 if (ok_a and ok_b) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--n", type=int, default=70)
    c.add_argument("--max-tokens", type=int, default=32)
    c.add_argument("--prompts", default=None,
                   help="JSON file from render_chat_prompts.py; both engines MUST be given "
                        "the same file so they see byte-identical text")
    c.add_argument("--logprobs", type=int, default=5,
                   help="record the top-N logprobs so the gate can tell a decisive prompt "
                        "from a coin flip (0 disables)")
    c.set_defaults(fn=capture)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.set_defaults(fn=lambda a: compare(a.a, a.b))
    g = sub.add_parser("gate", help="the pre-registered gate (a pre-registered gate)")
    g.add_argument("a", help="reference capture tag (the llama engine)")
    g.add_argument("b", help="candidate capture tag (the vLLM sidecar)")
    g.add_argument("--gap-nats", type=float, default=GATE_GAP_NATS)
    g.add_argument("--need-exact", type=int, default=GATE_B_N)
    g.set_defaults(fn=lambda a: gate(a.a, a.b, a.gap_nats, a.need_exact))
    d = sub.add_parser("needle")
    r = sub.add_parser("rescore", help="recompute gaps from the stored raw logprobs (CPU only)")
    r.add_argument("--tag", required=True)
    d.add_argument("--port", type=int, required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--fill", type=int, default=8192)
    d.add_argument("--max-tokens", type=int, default=320)
    d.set_defaults(fn=needle)
    r.set_defaults(fn=rescore)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
