#!/usr/bin/env python3
"""Headless-browser check of the header row that names every running server (see header-gui-check.js for what it checks).

It starts a hermetic environment (tests/live_env.py: a fake nvidia-smi and three fake llama-servers, so the header has something
to name) and runs the check in Chromium inside the playwright container for every theme and width, saving screenshots.

    python3 tests/header-gui-check.py [--themes dark,light] [--widths 2560,1280,390] [--out DIR] [--base URL]
                                      [--image mcr.microsoft.com/playwright/mcp:latest]

The widths cover the layout's two states on purpose: 2560 px has room for one chip per server, 1280 px and a 390 px phone have
room for the one summary chip, whose dropdown lists them.

After the theme x width matrix, a stop phase checks the header while servers stop (two of three, then the last): the
header shows only what runs and says "no servers" when nothing does. It stops servers, so it gets a fresh environment
per width, and it is skipped with --base.

--base URL checks an environment that is already running (python3 tests/live_env.py) instead of starting one.
Run it pinned (taskset -c ...) and the browser's container gets the same cores.
Needs docker and the playwright image. No GPU, no network beyond 127.0.0.1. Exit 0 = every check passed at every theme and width.
Skipped (exit 0, says so) when docker or the image is missing, unless --require is given."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# three servers that read as three different machines: the whole point is that the header names each of them
SERVERS = [
    dict(name="Rig 1", model_file="Qwen3.8-27B-Uncensored-pxqn5.gguf", gpus=[0, 1], slots=2, dec_tps=36, pre_tps=900, spec=True, seed=3),
    dict(name="Rig 2", model_file="Flash-Next-PXQ4.gguf", gpus=[2, 3], slots=1, dec_tps=22, pre_tps=500, seed=5),
    dict(name="Rig 3", model_file="gemma-4-12b-it-Q8_0.gguf", gpus=[2], slots=1, dec_tps=31, pre_tps=800, seed=7),
]


def cpuset():
    """the cores this run was pinned to (taskset), handed on to the browser's container: a pinned check stays pinned"""
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
    ap.add_argument("--widths", default="2560,1280,390")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-header-shots-")
    os.makedirs(out, exist_ok=True)

    def check(base, theme, width, phase=None):
        cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
               "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/header-gui-check.js", base, width, "/out", theme] + ([phase] if phase else [])
        print("== %s %s px%s" % (theme, width, " " + phase if phase else ""), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-2000:])
        return r.returncode == 0

    env = None
    base = a.base
    if not base:
        import live_env as E
        env = E.Env(cards=4, servers=SERVERS).start()
        base = env.url
    rc = 0
    try:
        for theme in a.themes.split(","):
            for width in a.widths.split(","):
                if not check(base, theme, width):
                    rc = 1
    finally:
        if env:
            env.close()
    if a.base:
        print("stop phase skipped: --base is an environment this check does not own, and the phase stops its servers")
    else:
        for width in a.widths.split(","):
            env = E.Env(cards=4, servers=SERVERS).start()
            try:
                if not check(env.url, a.themes.split(",")[0], width, "stop"):
                    rc = 1
            finally:
                env.close()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
