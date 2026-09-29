#!/usr/bin/env python3
"""Speculative serving at -np 2: server-level regression checks (bugs #212, #232, #234).

Each check boots the llama-server binary it is given, drives it over HTTP and decides from what the
server returns (text, per-token probabilities, draft counters) or logs. Nothing here re-implements
server code, so the same script run against an rc-base binary and a fixed binary is the A/B.

  logprobs   #212  n_probs / post_sampling_probs on speculative steps. Any hybrid GGUF works; no MTP
                   head needed (default spec: ngram_map_k), so Qwen3.5-0.8B on the CPU is enough.
                   (a) two CONCURRENT greedy requests, both asking n_probs: each emitted token's own
                       probability must match a no-spec reference run of the same pair. rc-base read
                       verify row i instead of the slot's own row i_batch_dft[i], so the slot whose
                       rows come second in the batch reported the other slot's distribution.
                   (b) the same pair under --recurrent-ckpt-mode cpu: a rejected draft restores by
                       re-decoding, which replaces the verify logits; rc-base read them after that
                       (another row's logits, or NULL -> crash). The server must survive, and every
                       greedy token must be the top-1 of the row reported for it (a check that needs
                       no reference: the greedy text in this mode may leave the no-spec reference).
                   (a) and (b) also apply the top-1 rule to every row. (a) bites only on steps where
                       both slots verify a draft together: an MTP chain does that every step, an n-gram
                       chain only sometimes (Qwen3.5-0.8B CPU, ngram_map_k: rc-base 3 bad rows).
                   (c) post_sampling_probs=true at temp 0.7: every emitted token must sit inside its
                       own post-chain window (rc-base gave every token the last position's window).
  cascade    #232  a composite chain (ngram_map_k + mtp) needs an MTP model. Slot B (short prompt) is
                   generating when slot A arrives with a long prompt. rc-base shared ONE speculative
                   object across slots, so A's begin() set the n-gram map's size_last_begin to A's
                   length and B's next draft scanned B's shorter token list past its end (OOB read;
                   ngram-map logs 'map.size_last_begin > cur_len'). Fixed: per-slot objects, no such
                   line, and both outputs equal a no-spec reference at temp 0.
  plainsync  #234  MTP-only chain, MTP model. A (speculative.n_max = 0: never drafts) generates, and B
                   (long, greedy) joins, so the uniform-batch equalizer keeps B on plain decodes and
                   the np>1 plain-decode sync alone keeps B's MTP companion cache in step for A's
                   lifetime. When A ends B drafts again from that cache. rc-base's sync wrote the
                   unshifted pair (h_p, x_p) at p, so B's post-overlap drafts read a misaligned cache.
                   Decision: B's acceptance on the tail (its only drafted part; the streams give the
                   boundary) against a one-slot reference that drafts the SAME tail from a cache built
                   by prompt warm-up over B's prompt plus B's overlap tokens (PXA_MTP_LAZY_WARMUP=0).

usage: spec-server-np2-test.py <logprobs|cascade|plainsync> --server BIN --model GGUF
            [--port 18300] [--out DIR] [--ngl 0] [--threads 8] [--delay 1.0] [--extra "<more server flags>"]
Exit 0 = pass, 1 = fail, 2 = could not run (boot failure, no drafts, wrong model).
Env for the server is inherited; PXA_SPEC_RELAXED is forced to 0 (exact verification) so that
greedy outputs cannot depend on the drafts.
"""
import argparse, json, os, shlex, signal, subprocess, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("check", choices=["logprobs", "cascade", "plainsync"])
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=18300)
ap.add_argument("--out", default="spec-np2-out")
ap.add_argument("--ngl", type=int, default=0)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--extra", default="")
ap.add_argument("--delay", type=float, default=1.0, help="seconds between the first and second request (cascade)")
ap.add_argument("--spec", default=None, help="override the spec flags of the check (logprobs only)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
URL = f"http://127.0.0.1:{args.port}"
RESULT = {"check": args.check, "server": args.server, "model": os.path.basename(args.model), "arms": {}}


def log(*a):
    print(*a, flush=True)


class Server:
    def __init__(self, label, flags, extra_env=None):
        self.label = label
        self.logpath = os.path.join(args.out, f"server-{args.check}-{label}.log")
        cmd = [args.server, "-m", args.model, "-ngl", str(args.ngl), "-t", str(args.threads), "-c", str(args.ctx),
               "-np", "2", "--host", "127.0.0.1", "--port", str(args.port), "--no-context-shift"] + flags + shlex.split(args.extra)
        env = dict(os.environ, PXA_SPEC_RELAXED="0", **(extra_env or {}))
        self.logf = open(self.logpath, "w")
        log(f"[{label}] {' '.join(cmd)}")
        self.p = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        for _ in range(900):
            if self.p.poll() is not None:
                raise RuntimeError(f"[{label}] server exited during boot (rc {self.p.returncode}), see {self.logpath}")
            try:
                urllib.request.urlopen(URL + "/health", timeout=2).read()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"[{label}] server did not come up, see {self.logpath}")

    def alive(self):
        time.sleep(1)
        if self.p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(URL + "/health", timeout=5).read()
            return True
        except Exception:
            return False

    def stop(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        self.logf.close()

    def logtext(self):
        with open(self.logpath, errors="replace") as f:
            return f.read()


def post(body, timeout=3600):
    req = urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run_concurrent(bodies, delays=None):
    """Send the bodies at once (or after per-body delays, seconds); return responses in order."""
    out = [None] * len(bodies)

    def go(i):
        if delays:
            time.sleep(delays[i])
        try:
            out[i] = post(bodies[i])
        except Exception as e:
            out[i] = {"error": repr(e)}
    th = [threading.Thread(target=go, args=(i,)) for i in range(len(bodies))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def rows(r):
    return r.get("completion_probabilities") or []


def own_prob(row):
    for x in row.get("probs") or row.get("top_probs") or []:
        if x.get("tok_str") == row.get("content"):
            return x.get("prob")
    return None


def draft_counts(r):
    t = r.get("timings", {})
    return t.get("draft_n") or 0, t.get("draft_n_accepted") or 0


def code_prompt(tag, n):
    body = "".join(f"def {tag}_{i}(req):\n    value = req.get('{tag}_field_{i}')\n    if value is None:\n"
                   f"        return error('missing {tag}_field_{i}')\n    return ok(value)\n\n" for i in range(n))
    return f"# {tag} handlers\n\n" + body + f"def {tag}_{n}(req):\n"


# ------------------------------------------------------------------------------------------ #212
def check_logprobs():
    spec = shlex.split(args.spec) if args.spec else ["--spec-type", "ngram_map_k"]
    npred = 96
    pair = [{"prompt": code_prompt("alpha", 6), "n_predict": npred, "temperature": 0.0, "n_probs": 5, "cache_prompt": False},
            {"prompt": code_prompt("beta", 8), "n_predict": npred, "temperature": 0.0, "n_probs": 5, "cache_prompt": False}]
    post_pair = [dict(b, temperature=0.7, top_k=20, top_p=1.0, min_p=0.0, seed=5 + i, post_sampling_probs=True)
                 for i, b in enumerate(pair)]
    ok = True
    # reference: same pair, no speculation
    s = Server("ref", ["--spec-type", "none"])
    try:
        ref = run_concurrent(pair)
    finally:
        s.stop()
    if any("error" in r for r in ref):
        raise RuntimeError(f"reference failed: {ref}")

    def compare(label, got):
        nonlocal ok
        arm = {}
        for k, (g, rf) in enumerate(zip(got, ref)):
            if "error" in g:
                arm[f"req{k}"] = {"error": g["error"][:300]}
                ok = False
                continue
            gr, rr = rows(g), rows(rf)
            n = same = bad = missing = 0
            maxd = 0.0
            for a, b in zip(gr, rr):
                if a.get("content") != b.get("content"):
                    break
                n += 1
                va, vb = own_prob(a), own_prob(b)
                if va is None:
                    missing += 1
                    continue
                if vb is None:
                    continue
                d = abs(va - vb)
                maxd = max(maxd, d)
                if d <= 0.05:
                    same += 1
                else:
                    bad += 1
            # greedy: every emitted token must be the top-1 of the distribution reported for it (one
            # row of slack for an exact tie). This needs no reference, so it covers all rows, also
            # past a point where the text diverges.
            not_top1 = 0
            for a in gr:
                ps = a.get("probs") or []
                if not ps or ps[0].get("tok_str") != a.get("content"):
                    not_top1 += 1
            dn, da = draft_counts(g)
            rec = {"n_rows": len(gr), "n_tokens": g.get("timings", {}).get("predicted_n"), "common_prefix": n,
                   "match": same, "mismatch": bad, "own_token_not_in_window": missing, "max_abs_diff": round(maxd, 4),
                   "emitted_not_top1": not_top1, "draft_n": dn, "draft_accepted": da}
            arm[f"req{k}"] = rec
            if bad + missing > max(2, n // 10) or not_top1 > 1 or len(gr) != rec["n_tokens"] or \
                    (label == "a" and n < 8):
                ok = False
                rec["FAIL"] = True
        return arm

    # (a) + (c): default checkpoint mode
    s = Server("spec", spec)
    try:
        got = run_concurrent(pair)
        RESULT["arms"]["a_row_logprobs"] = compare("a", got)
        got_p = run_concurrent(post_pair)
        arm = {}
        for k, g in enumerate(got_p):
            if "error" in g:
                arm[f"req{k}"] = {"error": g["error"][:300]}
                ok = False
                continue
            rs = rows(g)
            notin = sum(1 for e in rs if not (own_prob(e) or 0) > 0)
            dn, da = draft_counts(g)
            arm[f"req{k}"] = {"n_rows": len(rs), "emitted_not_in_own_window": notin, "draft_n": dn, "draft_accepted": da}
            if notin or not rs:
                ok = False
                arm[f"req{k}"]["FAIL"] = True
        RESULT["arms"]["c_post_sampling"] = arm
        if not s.alive():
            ok = False
            RESULT["arms"]["spec_server"] = "DIED"
    finally:
        s.stop()
    # (b): checkpoint restore by re-decode
    s = Server("ckpt-cpu", spec + ["--recurrent-ckpt-mode", "cpu"])
    try:
        got = run_concurrent(pair)
        RESULT["arms"]["b_ckpt_cpu"] = compare("b", got)
        RESULT["arms"]["b_ckpt_cpu"]["server_alive_after"] = s.alive()
        if not s.alive():
            ok = False
    finally:
        s.stop()
    drafted = sum(RESULT["arms"]["a_row_logprobs"][k].get("draft_n", 0) for k in ("req0", "req1")
                  if isinstance(RESULT["arms"]["a_row_logprobs"].get(k), dict))
    if drafted == 0:
        RESULT["note"] = "no drafts were verified: the check tested nothing"
        return 2
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ #232
def check_cascade():
    spec = ["--spec-type", "ngram_map_k", "--spec-type", "mtp:n_max=3,p_min=0.0"]
    b_body = {"prompt": code_prompt("beta", 3), "n_predict": 160, "temperature": 0.0, "cache_prompt": False}
    a_body = {"prompt": code_prompt("alpha", 60), "n_predict": 48, "temperature": 0.0, "cache_prompt": False}
    s = Server("ref", ["--spec-type", "none"])
    try:
        ref = run_concurrent([b_body, a_body], delays=[0, args.delay])
    finally:
        s.stop()
    s = Server("spec", spec)
    try:
        got = run_concurrent([b_body, a_body], delays=[0, args.delay])
        alive = s.alive()
    finally:
        s.stop()
    text = s.logtext()
    if "MTP" not in text and "mtp" not in text:
        RESULT["note"] = "server log shows no MTP stage: the model has no MTP head?"
        return 2
    oob = text.count("map.size_last_begin > cur_len")
    arm = {"server_alive_after": alive, "size_last_begin_gt_cur_len_lines": oob}
    ok = alive and oob == 0
    for k, (g, rf) in enumerate(zip(got, ref)):
        name = ["B_short_first", "A_long_second"][k]
        if "error" in g:
            arm[name] = {"error": g["error"][:300]}
            ok = False
            continue
        same = g.get("content") == rf.get("content")
        dn, da = draft_counts(g)
        arm[name] = {"prompt_n": g.get("timings", {}).get("prompt_n"), "same_text_as_ref": same, "draft_n": dn, "draft_accepted": da}
        ok = ok and same
    RESULT["arms"]["cascade"] = arm
    if sum(v.get("draft_n", 0) for v in arm.values() if isinstance(v, dict)) == 0:
        RESULT["note"] = "no drafts: the check tested nothing"
        return 2
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ #234
def tokenize(text):
    req = urllib.request.Request(URL + "/tokenize", json.dumps({"content": text, "add_special": False}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["tokens"]


def stream(body, events, tag):
    """POST a streamed completion; append (tag, content, stop) per chunk to the shared events list."""
    req = urllib.request.Request(URL + "/completion", json.dumps(dict(body, stream=True)).encode(),
                                 {"Content-Type": "application/json"})
    final = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            events.append((tag, ev.get("content", ""), bool(ev.get("stop"))))
            if ev.get("stop"):
                final = ev
    return final


def check_plainsync():
    spec = ["--spec-type", "mtp:n_max=3,p_min=0.0"]
    nb = int(os.environ.get("NP2_B_NPRED", 900))
    na = int(os.environ.get("NP2_A_NPRED", 400))
    b_prompt = code_prompt("beta", 4)
    b_body = {"prompt": b_prompt, "n_predict": nb, "temperature": 0.0, "cache_prompt": False}
    a_body = {"prompt": "Write a short story about a lighthouse keeper.", "n_predict": na, "temperature": 0.0,
              "cache_prompt": False, "speculative.n_max": 0}
    # PXA_MTP_LAZY_WARMUP=0: the companion is warmed over every prompt, so the reference below (whose
    # prompt is B's prompt + B's overlap tokens) gets a correctly built cache for exactly the tokens
    # B's cache got from its own prompt warm-up + the plain-decode syncs. (The default lazy warm-up
    # would leave the reference's history EMPTY, and an empty history drafts about as well as a
    # misaligned one.)
    s = Server("spec", spec, {"PXA_MTP_LAZY_WARMUP": "0"})
    try:
        # A (never drafts) starts first; B joins once A is generating. From then until A ends the
        # uniform-batch equalizer keeps B on plain decodes (the np>1 plain-decode sync), so B drafts
        # only after A ends: B's draft counters cover exactly that tail. The streams give the boundary.
        events, res = [], {}
        ta = threading.Thread(target=lambda: res.__setitem__("A", stream(a_body, events, "A")))
        ta.start()
        while not any(t == "A" for t, _, _ in events):
            time.sleep(0.05)
        res["B"] = stream(b_body, events, "B")
        ta.join()
        a_end = next(i for i, (t, _, stop) in enumerate(events) if t == "A" and stop)
        b_head = "".join(c for t, c, _ in events[:a_end] if t == "B")
        b_text = "".join(c for t, c, _ in events if t == "B")
        # reference: the SAME tail drafted from a cache built the clean way (prompt warm-up over B's
        # prompt plus the tokens B generated while A ran), one slot, same binary
        p_toks = tokenize(b_prompt + b_head)
        tail_n = len(tokenize(b_prompt + b_text)) - len(p_toks)
        n_overlap = len(p_toks) - len(tokenize(b_prompt))
        ref = post({"prompt": p_toks, "n_predict": tail_n, "temperature": 0.0, "cache_prompt": False})
        alive = s.alive()
    finally:
        s.stop()
    if "mtp" not in s.logtext().lower():
        RESULT["note"] = "server log shows no MTP stage: the model has no MTP head?"
        return 2
    b, a = res["B"], res["A"]
    odn, oda = draft_counts(b)
    rdn, rda = draft_counts(ref)
    arm = {"server_alive_after": alive, "A_predicted_n": a.get("timings", {}).get("predicted_n"),
           "A_draft_n": draft_counts(a)[0], "B_tokens_during_overlap": n_overlap,
           "B_tail_after_overlap": {"draft_n": odn, "draft_accepted": oda, "accept": round(oda / odn, 4) if odn else None},
           "reference_same_tail": {"draft_n": rdn, "draft_accepted": rda, "accept": round(rda / rdn, 4) if rdn else None,
                                   "predicted_n": ref.get("timings", {}).get("predicted_n"),
                                   "same_text_as_B_tail": b_text[len(b_head):] == ref.get("content")}}
    RESULT["arms"]["plainsync"] = arm
    if not odn or not rdn:
        RESULT["note"] = "no drafts on one side: the check tested nothing"
        return 2
    if arm["A_draft_n"]:
        RESULT["note"] = "A drafted: the per-request n_max=0 did not keep it plain"
        return 2
    # Pass rule: B's drafts after the overlap come from the cache the plain-decode sync kept, and
    # must accept like drafts from a cleanly built cache over the same tokens (1.5 points of slack).
    # Ornith-9B PXQ4 on one P100, 2 runs each: fixed 431/431 and 432/432 vs reference 431/431 and
    # 432/434; rc-base 429/443 and 428/440 vs reference 432/434 and 431/431 (-2.7 points).
    ok = alive and (oda / odn) >= (rda / rdn) - 0.015
    return 0 if ok else 1


rc = 2
try:
    rc = {"logprobs": check_logprobs, "cascade": check_cascade, "plainsync": check_plainsync}[args.check]()
except Exception as e:
    import traceback
    traceback.print_exc()
    RESULT["error"] = repr(e)
    rc = 2
RESULT["verdict"] = {0: "PASS", 1: "FAIL", 2: "NOT-RUN"}[rc]
path = os.path.join(args.out, f"{args.check}.json")
with open(path, "w") as f:
    json.dump(RESULT, f, indent=1)
log(json.dumps(RESULT, indent=1))
sys.exit(rc)
