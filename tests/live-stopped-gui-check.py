#!/usr/bin/env python3
"""Headless-browser check of the Live tab's "Running only" switch (see live-stopped-gui-check.js for what it checks).
Starts a hermetic environment (tests/live_env.py: a fake nvidia-smi and three fake llama-servers) per theme x width (the check
STOPS servers, so every run gets a fresh environment) and runs the check in Chromium inside the playwright container.

    python3 tests/live-stopped-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR] [--image IMG] [--require]

Run it pinned (taskset -c ...): the browser's container gets the same cores. Needs docker and the playwright image
(skipped, exit 0, without them unless --require). No GPU, no network beyond 127.0.0.1."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SERVERS = [
    dict(name="Rig 1", model_file="Qwen3.8-27B-Uncensored-pxqn5.gguf", gpus=[0, 1], slots=2, dec_tps=36, pre_tps=900, spec=True, seed=3),
    dict(name="Rig 2", model_file="Flash-Next-PXQ4.gguf", gpus=[2, 3], slots=1, dec_tps=22, pre_tps=500, seed=5),
    dict(name="Rig 3", model_file="gemma-4-12b-it-Q8_0.gguf", gpus=[2], slots=1, dec_tps=31, pre_tps=800, seed=7),
]


def cpuset():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []
    return [] if len(cpus) >= (os.cpu_count() or 0) else ["--cpuset-cpus", ",".join(map(str, cpus))]


def have_docker(image):
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


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
    out = a.out or tempfile.mkdtemp(prefix="pxa-live-stopped-shots-")
    os.makedirs(out, exist_ok=True)
    import live_env as E
    rc = 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            env = E.Env(cards=4, servers=SERVERS).start()
            try:
                cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                       "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/live-stopped-gui-check.js", env.url, width, "/out", theme]
                print("== %s %s px" % (theme, width), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                sys.stdout.write(r.stdout)
                if r.returncode != 0:
                    sys.stdout.write(r.stderr[-2000:])
                    rc = 1
            finally:
                env.close()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
