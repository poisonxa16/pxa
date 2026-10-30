#!/usr/bin/env python3
"""Headless-browser check of the tokens-per-watt column and chart on PXA Control's Benchmarks tab (see watt-gui-check.js).

Hermetic environment (tests/live_env.py): a fake nvidia-smi whose power.draw follows the load of the fake servers (a benchmark request in
flight = a busy card), fake llama-servers that answer /completion like the engine, and 45 seeded benchmark rows: the newest 30 carry a
power summary, the 15 oldest do not (stored before the feature), and one carries an empty one (no card reported a draw). Two
environments per theme x width: power readable (phases table,chart,run) and power N/A (phase na: a host whose nvidia-smi says [N/A]).

    python3 tests/watt-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR] [--require]
                                    [--image mcr.microsoft.com/playwright/mcp:latest]

Run it pinned (taskset -c ...) and the browser's container gets the same cores. Needs docker and the playwright image. No GPU, no network
beyond 127.0.0.1. Exit 0 = every check passed. Skipped (exit 0, says so) when docker or the image is missing, unless --require."""
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
N = 45
NO_POWER_FROM = 15          # rows 0..14 (the oldest) have no "power" key at all
EMPTY_POWER_ROW = 20        # this one has a power summary without any reading (a card that said N/A)


def seed_history(env, n=N):
    now = time.time()
    with open(os.path.join(env.tmp, "cfg", "bench.jsonl"), "w") as f:
        for i in range(n):
            ts = now - (n - i) * 3600 * 5
            prose = 20 + (i % 9) * 2.5
            r = {"ts": ts, "model": "/models/" + MODELS[i % 3], "cards": [0, 1] if i % 2 else [2], "port": 1, "reps": 3, "warmup_s": 10,
                 "classes": [{"class": c, "decode_tps": prose + k, "prefill_tps": 600 + 11 * i + 40 * k, "n_prompt": 100 * (k + 1)} for k, c in enumerate(("prose", "edit", "long"))],
                 "greedy512_sha": "%012x" % (0xabc000000000 + i * 7919) + "0" * 52, "telemetry": {"cards": [], "samples": 0, "warnings": []}}
            if i >= n - 30:
                r["mtp"] = bool(i % 3 == 0)
                r["split_mode"] = ("tensor", "layer", "plain")[i % 3]
            if i >= NO_POWER_FROM:
                if i == EMPTY_POWER_ROW:
                    r["power"] = {"samples": 0, "interval_s": 1.0, "cards": [], "avg_w": None, "peak_w": None, "decode_tps": prose, "tps_per_w": None}
                else:
                    avg = 150.0 + (i % 7) * 25
                    r["power"] = {"samples": 30, "interval_s": 1.0, "avg_w": avg, "peak_w": avg + 40, "decode_tps": prose,
                                  "tps_per_w": round(prose / avg, 3),
                                  "cards": [{"index": 0, "avg_w": avg / 2, "peak_w": avg / 2 + 20, "limit_w": 250.0, "samples": 30},
                                            {"index": 1, "avg_w": avg / 2, "peak_w": avg / 2 + 20, "limit_w": 250.0, "samples": 30}]}
            f.write(json.dumps(r) + "\n")


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-watt-shots-")
    os.makedirs(out, exist_ok=True)
    import live_env as E
    rc, fails = 0, 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            for mode, phases in (("ok", "table,chart,run"), ("na", "na")):
                env = E.Env(cards=4, servers=[dict(s) for s in SERVERS], library=True).start()
                try:
                    env.set_power_mode(mode)
                    seed_history(env)
                    cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                           "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/watt-gui-check.js", env.url, width, "/out", theme, phases]
                    print("== %s %s px power=%s" % (theme, width, mode), flush=True)
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
