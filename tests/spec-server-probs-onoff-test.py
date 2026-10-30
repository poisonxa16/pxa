#!/usr/bin/env python3
"""Bug #308: n_probs / post_sampling_probs with speculation ON must equal speculation OFF at every token position.

The server reports a speculative step's per-token probabilities from the verify row that token was sampled from
(slot.i_batch_dft[k], read in the pre-pass before any restore re-decode: bug #212). The row semantics are the
same as a one-token decode -- row k holds the distribution the k-th emitted token was drawn from -- so on a
backend whose rows do not depend on the batch width (the CPU) the reported numbers must be BIT-EQUAL to a
--spec-type none run of the same requests. This boots the given llama-server once per arm (np 1, temp 0, the
same prompts), and compares, per position: the token, its own probability and the full top-N list (ids as
pieces and probabilities, exact float equality as the JSON carries them).

Arms: a self-draft (-md = the target, n_max 8: every draft accepted, so every step verifies 9 rows and every
emitted token past the first comes from a multi-row verify), the same at n_max 3, and ngram_map_k. Each arm
runs pre-sampling probs (n_probs) and post-sampling probs (post_sampling_probs=true).

usage: spec-server-probs-onoff-test.py --server BIN --model GGUF [--threads 4] [--npred 32] [--port 18308]
Exit 0 = every arm bit-equal, 1 = a difference (printed), 77 = skipped (no server binary or model).
"""
import argparse, json, os, shlex, subprocess, sys, tempfile, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=18308)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--npred", type=int, default=32)
ap.add_argument("--nprobs", type=int, default=5)
ap.add_argument("--out", default=None)
args = ap.parse_args()

if not (os.path.isfile(args.server) and os.access(args.server, os.X_OK)) or not os.path.isfile(args.model):
    print(f"SKIP: server {args.server} or model {args.model} missing")
    sys.exit(77)
out = args.out or tempfile.mkdtemp(prefix="spec-probs-onoff-")
os.makedirs(out, exist_ok=True)


def code_prompt(tag, n):
    body = "".join(f"def {tag}_{i}(req):\n    value = req.get('{tag}_field_{i}')\n    if value is None:\n"
                   f"        return error('{tag}_field_{i} missing')\n    return ok(value)\n\n" for i in range(n))
    return f"# {tag} handlers\n\n" + body + f"def {tag}_{n}(req):\n"


PROMPTS = [code_prompt("alpha", 6), "The history of the printing press begins"]
ARMS = [
    ("draft8", ["-md", args.model, "--spec-type", "draft:n_max=8,p_min=0.0"]),
    ("draft3", ["-md", args.model, "--spec-type", "draft:n_max=3,p_min=0.0"]),
    ("ngram", ["--spec-type", "ngram_map_k"]),
]


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as h:
        return json.loads(h.read())


def run(label, flags):
    cmd = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port), "-ngl", "0",
           "-t", str(args.threads), "-c", str(args.ctx), "-np", "1", "--no-warmup"] + flags
    env = dict(os.environ, PXA_SPEC_RELAXED="0")   # exact verification: greedy text cannot depend on the drafts
    with open(os.path.join(out, f"{label}.srv.log"), "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(900):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as h:
                        if h.status == 200:
                            break
                except Exception:
                    pass
                if p.poll() is not None:
                    raise RuntimeError(f"{label}: server exited rc={p.returncode} (log {out}/{label}.srv.log)")
                time.sleep(1)
            res = {}
            for post_sampling in (False, True):
                for i, pr in enumerate(PROMPTS):
                    body = {"prompt": pr, "n_predict": args.npred, "temperature": 0.0, "n_probs": args.nprobs,
                            "cache_prompt": False, "seed": 7, "post_sampling_probs": post_sampling}
                    res[(post_sampling, i)] = post(body)
            return res
        finally:
            p.terminate()
            try:
                p.wait(30)
            except Exception:
                p.kill()


def rows(r):
    return r.get("completion_probabilities") or []


def key(row):
    ps = row.get("probs") or row.get("top_logprobs") or []
    return [(x.get("tok_str", x.get("token")), x.get("prob", x.get("logprob"))) for x in ps]


try:
    ref = run("ref", ["--spec-type", "none"])
except Exception as e:
    print(f"could not run the reference: {e}")
    sys.exit(1)
ok = True
for name, flags in ARMS:
    got = run(name, flags)
    for (post_sampling, i), g in got.items():
        r = ref[(post_sampling, i)]
        gr, rr = rows(g), rows(r)
        diffs = []
        for pos, (x, y) in enumerate(zip(gr, rr)):
            if x.get("content") != y.get("content"):
                diffs.append((pos, "token", x.get("content"), y.get("content")))
                break
            if key(x) != key(y):
                diffs.append((pos, "probs", key(x)[:2], key(y)[:2]))
        t = g.get("timings", {})
        drafted, accepted = t.get("draft_n", 0) or 0, t.get("draft_n_accepted", 0) or 0
        tag = f"{name} {'post' if post_sampling else 'pre '} prompt{i}"
        print(f"{tag}: rows {len(gr)}/{len(rr)} differing {len(diffs)} draft {accepted}/{drafted}")
        if not gr or len(gr) != len(rr) or diffs:
            ok = False
            for d in diffs[:3]:
                print(f"  first differences: {d}")
        if name.startswith("draft") and accepted == 0:
            print(f"{tag}: the self-draft accepted nothing -- the arm did not exercise multi-row verify")
            ok = False
print("RESULT", "BIT-EQUAL" if ok else "DIFFERENT", f"(logs {out})")
sys.exit(0 if ok else 1)
