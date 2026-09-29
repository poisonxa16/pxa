#!/usr/bin/env python3
"""PXA hot swap, end to end on one server process (examples/server/pxa-hotswap.h).

    tests/hotswap-e2e-test.py BUILD_DIR MODEL_A MODEL_B [PORT]

Starts llama-server with MODEL_A as -m and MODEL_B registered with --hot-model, and checks:
  1. /v1/models lists both, MODEL_A on the cards at open;
  2. greedy output of each model through the hot-swap server equals a single-model server of that
     model (before any swap, after a swap in, after a swap back, and once more);
  3. an unknown model name gets 404; an unnamed request is served by the model on the cards;
  4. requests for both models fired concurrently are all served, each by the model it named;
  5. a conversation on A resumes after a round trip through B without re-prefilling its history
     (prompt tokens evaluated on turn 2 << the turn-1 history);
  6. the router's counters in /props agree with what was sent.
Runs on a CPU build (park/unpark move nothing there; routing, queue pausing and slot state are the
same code) and on a CUDA build (then it also exercises the residency groups). Exit 0 on pass.
HS_FLAGS overrides the shared server flags (default: -c 4096 -np 2 -t 4).
"""
import http.client, json, os, subprocess, sys, tempfile, threading, time

BD, MA, MB = sys.argv[1], sys.argv[2], sys.argv[3]
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 18499
SRV = os.path.join(BD, "bin", "llama-server")
FLAGS = os.environ.get("HS_FLAGS", "-c 4096 -np 2 -t 4").split()
LOGDIR = tempfile.mkdtemp(prefix="hotswap-e2e-")
fails = []


def fail(msg):
    print("FAIL:", msg, flush=True)
    fails.append(msg)


def req(method, path, body=None, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None,
              {"Content-Type": "application/json"} if body is not None else {})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except Exception:
        return r.status, data


def start(args, name):
    log = open(os.path.join(LOGDIR, name + ".log"), "w")
    p = subprocess.Popen([SRV] + args + FLAGS + ["--host", "127.0.0.1", "--port", str(PORT)], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(3000):
        if p.poll() is not None:
            return None
        try:
            st, _ = req("GET", "/health", timeout=2)
            if st == 200:
                return p
        except Exception:
            pass
        time.sleep(0.1)
    p.kill()
    return None


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()


P1 = "The three primary colours are"
P2 = "Write one sentence about the sea:"


def gen(model, prompt, n=48):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False, "seed": 1}
    if model is not None:
        body["model"] = model
    st, r = req("POST", "/completion", body)
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {r}")
    return r["content"]


# ---- reference outputs from single-model servers
refs = {}
for path, key in ((MA, "A"), (MB, "B")):
    p = start(["-m", path], "ref-" + key)
    if not p:
        fail(f"reference server for {path} did not start")
        sys.exit(1)
    refs[key] = (gen("x", P1), gen("x", P2))
    stop(p)

# ---- the hot-swap server
p = start(["-m", MA, "-a", "alpha", "--hot-model", f"beta={MB}"], "hotswap")
if not p:
    fail("hot-swap server did not start")
    print(open(os.path.join(LOGDIR, "hotswap.log")).read()[-4000:])
    sys.exit(1)

try:
    st, m = req("GET", "/v1/models")
    ids = [d["id"] for d in m["data"]]
    act = [d["id"] for d in m["data"] if d["pxa_hot_swap"]["active"]]
    if ids != ["alpha", "beta"] or act != ["alpha"]:
        fail(f"/v1/models: ids {ids} active {act}")

    seq = [("alpha", "A"), ("beta", "B"), ("alpha", "A"), ("beta", "B")]
    outs = []
    for name, key in seq:
        o = (gen(name, P1), gen(name, P2))
        outs.append(o)
        if o != refs[key]:
            fail(f"{name} differs from its single-model server: {o!r} vs {refs[key]!r}")
    if outs[0] == outs[1]:
        fail("the two models answered identically -- routing is not happening")

    st, r = req("POST", "/completion", {"model": "gpt-4o", "prompt": "hi", "n_predict": 4})
    if st != 404:
        fail(f"unknown model: HTTP {st} {r}")
    st, props = req("GET", "/props")
    active = props["hot_swap"]["active"]
    if gen(None, P1) != refs["A" if active == "alpha" else "B"][0]:
        fail(f"an unnamed request was not served by the model on the cards ({active})")

    # concurrency: 8 requests alternating models, all at once
    res = {}
    def one(i):
        name = "alpha" if i % 2 == 0 else "beta"
        try:
            res[i] = (name, gen(name, P1))
        except Exception as e:
            res[i] = (name, f"ERROR {e}")
    th = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in th: t.start()
    for t in th: t.join()
    for i, (name, out) in sorted(res.items()):
        want = refs["A" if name == "alpha" else "B"][0]
        if out != want:
            fail(f"concurrent request {i} for {name} got another answer: {out!r}")

    # conversation resume: history on alpha survives a trip through beta
    hist = " ".join(f"Item {i} is a line of the long history that the model must keep." for i in range(120))
    def chat(model, msgs):
        st, r = req("POST", "/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 16,
                                                      "temperature": 0, "cache_prompt": True})
        if st != 200:
            raise RuntimeError(f"HTTP {st}: {r}")
        return r["timings"]["prompt_n"], r["choices"][0]["message"]["content"]
    m1 = [{"role": "user", "content": hist + " Summarise."}]
    n1, t1 = chat("alpha", m1)
    gen("beta", P2, 8)
    n2, _ = chat("alpha", m1 + [{"role": "assistant", "content": t1}, {"role": "user", "content": "Shorter."}])
    print(f"resume: turn 1 evaluated {n1} prompt tokens, turn 2 after the round trip evaluated {n2}", flush=True)
    if not n2 < n1 / 4:
        fail(f"turn 2 re-evaluated {n2} prompt tokens of a {n1}-token history (KV was not kept)")

    st, props = req("GET", "/props")
    h = props["hot_swap"]
    ms = {x["name"]: x for x in h["models"]}
    if not (h["enabled"] and h["n_swaps"] >= 4 and ms["alpha"]["n_swaps_in"] >= 2 and ms["beta"]["n_swaps_in"] >= 2
            and h["last_swap"]["ok"]):
        fail(f"/props hot_swap counters: {json.dumps(h)[:600]}")
    else:
        ls = h["last_swap"]
        print(f"router: {h['n_swaps']} swaps, last {ls['from']} -> {ls['to']} in {ls['total_ms']:.0f} ms", flush=True)
finally:
    stop(p)

hs_lines = [l.rstrip() for l in open(os.path.join(LOGDIR, "hotswap.log"), errors="replace") if "hot swap" in l]
for l in hs_lines[-6:]:
    print("  log:", l[:300])
if fails:
    print(f"hotswap-e2e: {len(fails)} FAILED (logs in {LOGDIR})")
    sys.exit(1)
print("hotswap-e2e: all checks passed")
