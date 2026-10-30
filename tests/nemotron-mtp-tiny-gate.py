#!/usr/bin/env python3
"""PXA_NEMOTRON_MTP: CPU identity gate for the Nemotron NextN head (tiny model from gen_tiny_nemotron_h_moe.py).

Arms (one server each, greedy, -np 1, CPU):
  PLAIN      trunk only                                   -> the reference text
  ENVONLY    PXA_MTP_HEAD_FILE set, no MTP spec           -> must equal PLAIN (head attached but skipped)
  MTPk       PXA_MTP_HEAD_FILE + --spec-type mtp:n_max=k  -> must equal PLAIN, k in --nmax (default 1,2,4)
  NEGk       (--neg) MTPk + PXA_SPEC_CKPT_TEST_NO_RESTORE=1 -> must DIFFER from PLAIN (negative control: the
             rollback is skipped, so a gate that cannot see a wrong recurrent state fails here)
PASS = every arm's text on every prompt is byte-identical to PLAIN, no server died, and each MTP arm both
accepted and rejected drafts (so the accept AND the rollback path ran); with --neg the NEG arms must differ.

usage: nemotron-mtp-tiny-gate.py SERVER_BIN TRUNK.gguf HEAD.gguf --out DIR [--nmax 1,2,4] [--n-predict 96]
       [--extra-env K=V ...] [--tag NAME]
"""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.request

PROMPTS = ["the quick brown fox jumps over the lazy dog ", "hello world, this is a tiny model test ",
           "0123456789 abcdefghij klmnop ", "Once upon a time, in a small village, "]


def ldpath(b):
    d = os.path.dirname(os.path.abspath(b)); r = os.path.dirname(d)
    return ":".join([d, os.path.join(r, "src"), os.path.join(r, "ggml", "src"), os.path.join(r, "examples", "mtmd")])


def run_arm(a, tag, envs, spec, port):
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = ldpath(a.server)
    env.update(envs)
    log = os.path.join(a.out, tag + ".log")
    cmd = ["taskset", "-c", a.cpus, "nice", "-n", "19", a.server, "-m", a.trunk, "-ngl", "0", "-c", "2048", "-np", "1",
           "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(port), "--no-warmup", "--cache-ram", "0"] + spec
    with open(log, "w") as lf:
        lf.write("# " + " ".join(f"{k}={v}" for k, v in envs.items()) + " " + " ".join(cmd) + "\n"); lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    ok = False
    for _ in range(int(a.boot_wait * 2)):
        if p.poll() is not None:
            break
        try:
            if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read():
                ok = True; break
        except Exception:
            pass
        time.sleep(0.5)
    res = {"tag": tag, "boot": ok, "texts": [], "draft_n": 0, "draft_acc": 0, "died": False}
    if ok:
        for pr in PROMPTS:
            body = json.dumps({"prompt": pr, "n_predict": a.n_predict, "temperature": 0, "top_k": 1, "seed": 1,
                               "cache_prompt": False}).encode()
            try:
                r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
                d = json.load(urllib.request.urlopen(r, timeout=600))
                t = d.get("timings", {})
                res["texts"].append(d.get("content", ""))
                res["draft_n"] += int(t.get("draft_n") or 0); res["draft_acc"] += int(t.get("draft_n_accepted") or 0)
            except Exception as e:
                res["texts"].append(f"<ERROR {type(e).__name__}: {e}>")
    res["died"] = p.poll() is not None
    if not res["died"]:
        p.terminate()
        try: p.wait(20)
        except Exception: p.kill()
    txt = open(log, errors="replace").read()
    res["assert"] = ("GGML_ASSERT" in txt) or ("Aborted" in txt) or ("AddressSanitizer" in txt)
    res["sha"] = hashlib.sha256("\x00".join(res["texts"]).encode()).hexdigest()[:16]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server"); ap.add_argument("trunk"); ap.add_argument("head")
    ap.add_argument("--out", required=True); ap.add_argument("--nmax", default="1,2,4")
    ap.add_argument("--n-predict", type=int, default=96); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpus", default="36-53"); ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--extra-env", nargs="*", default=[]); ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--boot-wait", type=float, default=90.0)     # seconds; a 61 GiB file needs minutes
    ap.add_argument("--mtp-args", default="")                     # extra server args for MTP/NEG arms, e.g. "--recurrent-ckpt-mode cpu"
    ap.add_argument("--neg", action="store_true")               # add the NO_RESTORE negative-control arms
    ap.add_argument("--any-acc", action="store_true")            # real model: all-accepted is fine, only draft_n > 0
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    extra = dict(kv.split("=", 1) for kv in a.extra_env)
    head = {"PXA_MTP_HEAD_FILE": a.head}
    arms = [("PLAIN", dict(extra), []), ("ENVONLY", {**extra, **head}, [])]
    for k in [int(x) for x in a.nmax.split(",") if x]:
        arms.append((f"MTP{k}", {**extra, **head}, ["--spec-type", f"mtp:n_max={k},p_min=0.0"] + a.mtp_args.split()))
    if a.neg:
        for k in [int(x) for x in a.nmax.split(",") if x]:
            arms.append((f"NEG{k}", {**extra, **head, "PXA_SPEC_CKPT_TEST_NO_RESTORE": "1"},
                         ["--spec-type", f"mtp:n_max={k},p_min=0.0"] + a.mtp_args.split()))
    if a.arms:
        keep = set(a.arms.split(",")); arms = [x for x in arms if x[0] in keep]
    rows = []
    for i, (tag, envs, spec) in enumerate(arms):
        r = run_arm(a, (a.tag + "-" if a.tag else "") + tag, envs, spec, a.port + i)
        rows.append(r)
        acc = (r["draft_acc"] / r["draft_n"]) if r["draft_n"] else 0.0
        print(f"{r['tag']:16s} boot={r['boot']} died={r['died']} assert={r['assert']} sha={r['sha']} "
              f"drafts={r['draft_n']} acc={r['draft_acc']} ({acc:.0%})", flush=True)
    ref = rows[0]
    ok = ref["boot"] and not any(t.startswith("<ERROR") for t in ref["texts"])
    for r in rows:
        if r["tag"].split("-")[-1].startswith("NEG"):
            ok &= r["boot"] and r["sha"] != ref["sha"] and r["draft_n"] > r["draft_acc"]
            continue
        ok &= r["boot"] and not r["died"] and not r["assert"] and r["sha"] == ref["sha"]
        if r["tag"].split("-")[-1].startswith("MTP"):
            ok &= (r["draft_n"] > 0) if a.any_acc else (r["draft_n"] > r["draft_acc"] > 0)
            # Mamba-2 phase A: per-step capture must be refused and a whole-state mode selected
            lg = open(os.path.join(a.out, r["tag"] + ".log"), errors="replace").read()
            r["refused"] = "pxa_per_step_refused_mamba2" in lg
            r["ckpt_mode"] = next((ln.rsplit("=", 1)[1].strip() for ln in lg.splitlines()
                                   if "fixed recurrent checkpoint mode" in ln), "?")
            print(f"{r['tag']:16s} mamba2-per-step-refused={r['refused']} ckpt-mode={r['ckpt_mode']}", flush=True)
            ok &= (r["refused"] or "--recurrent-ckpt-mode" in a.mtp_args) and r["ckpt_mode"] in ("gpu-fallback", "cpu")
    json.dump(rows, open(os.path.join(a.out, (a.tag + "-" if a.tag else "") + "gate.json"), "w"), indent=1)
    print("GATE", "PASS" if ok else "FAIL", "ref text[0]:", repr(ref["texts"][0][:120]) if ref["texts"] else "-")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
