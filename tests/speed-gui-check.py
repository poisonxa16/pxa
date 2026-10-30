#!/usr/bin/env python3
"""Headless-browser check of PXA Control's Speed tab, now called "Benchmarks" (see speed-gui-check.js for what it checks).

It starts a hermetic environment (tests/live_env.py: a fake nvidia-smi and fake llama-servers that answer /completion like the
engine does, so "Benchmark my rig" really runs through Control's own code; 45 seeded benchmark rows for the history table) and runs
the check in Chromium inside the playwright container for every theme and width, saving screenshots. Every theme x width gets a
fresh environment because the check runs a benchmark and stops a server.

    python3 tests/speed-gui-check.py [--themes dark,light] [--widths 1280,390] [--phases refresh,table,strip,label] [--out DIR]
                                     [--image mcr.microsoft.com/playwright/mcp:latest]

Run it pinned (taskset -c ...) and the browser's container gets the same cores. Needs docker and the playwright image. No GPU, no
network beyond 127.0.0.1. Exit 0 = every check passed at every theme and width. Skipped (exit 0, says so) when docker or the image is
missing, unless --require is given."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SERVERS = [
    dict(name="Rig 1", model_file="Qwen3.8-27B-Uncensored-pxqn5.gguf", gpus=[0, 1], slots=2, dec_tps=36, pre_tps=900, spec=True, seed=3,
         quiet=True, comp_delay=0.5),
    dict(name="Rig 2", model_file="Flash-Next-PXQ4.gguf", gpus=[2, 3], slots=1, dec_tps=22, pre_tps=500, seed=5, quiet=True, comp_delay=0.5),
]
MODELS = ["Qwen3.8-27B-Uncensored-pxqn5.gguf", "Flash-Next-PXQ4.gguf", "gemma-4-12b-it-Q8_0.gguf"]


def seed_history(env, n=45):
    """n finished benchmark rows (oldest first) in the config dir, the shape Control writes; the newest ones carry the MTP and split fields,
    the older ones do not (a row from before they were stored shows a dash)"""
    now = time.time()
    with open(os.path.join(env.tmp, "cfg", "bench.jsonl"), "w") as f:
        for i in range(n):
            ts = now - (n - i) * 3600 * 5
            r = {"ts": ts, "model": "/models/" + MODELS[i % 3], "cards": [0, 1] if i % 2 else [2], "port": 1, "reps": 3, "warmup_s": 10,
                 "classes": [{"class": c, "decode_tps": 20 + (i % 9) * 2.5 + k, "prefill_tps": 600 + 11 * i + 40 * k, "n_prompt": 100 * (k + 1)} for k, c in enumerate(("prose", "edit", "long"))],
                 "greedy512_sha": "%012x" % (0xabc000000000 + i * 7919) + "0" * 52, "telemetry": {"cards": [], "samples": 0, "warnings": []}}
            if i >= n - 30:
                r["mtp"] = bool(i % 3 == 0)
                r["split_mode"] = ("tensor", "layer", "plain")[i % 3]
            f.write(json.dumps(r) + "\n")


def fake_launcher(env):
    """Start cannot launch a real engine here, so it does what a launch ends in: the seat gets a running process and its request
    (the fake llama-server of the same port keeps answering, as the engine would once up). Stop, save and the page are the real ones."""
    app = env.app
    import subprocess

    def start(body):
        seat = app.get_seat(body["sid"], create=True)
        seat.cmd = [sys.executable, "-c", "import time; time.sleep(86400)"]
        seat.proc = subprocess.Popen(seat.cmd)
        env.seats_procs.append(seat.proc)
        seat.req = {"port": (seat.req or {}).get("port") or body.get("port"), "gpus": list(body.get("gpus") or []), "model": body.get("model")}
        return {"ok": True}
    app.start = start


def cpuset():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []
    return [] if len(cpus) >= (os.cpu_count() or 0) else ["--cpuset-cpus", ",".join(map(str, cpus))]


def have_docker(image):
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1280,390")
    ap.add_argument("--phases", default="refresh,table,strip,label")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-speed-shots-")
    os.makedirs(out, exist_ok=True)
    import live_env as E
    rc, fails = 0, 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            env = E.Env(cards=4, servers=[dict(s) for s in SERVERS], library=True).start()
            try:
                seed_history(env)
                fake_launcher(env)
                cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                       "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/speed-gui-check.js", env.url, width, "/out", theme, a.phases]
                print("== %s %s px" % (theme, width), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                sys.stdout.write(r.stdout)
                if r.returncode != 0:
                    sys.stdout.write(r.stderr[-2000:])
                    rc = 1
                fails += r.stdout.count("  FAIL ")
            finally:
                env.close()
    print("screenshots in", out)
    print("TOTAL FAIL lines:", fails)
    return rc


if __name__ == "__main__":
    sys.exit(main())
