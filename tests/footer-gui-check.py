#!/usr/bin/env python3
"""Headless-browser check of FR#15: the app footer (the Discord link, the Ko-fi link, the PXA version and the Update button) shows
on every page of PXA Control, Home first, not only on Servers (see footer-gui-check.js for what it checks per page).

It starts the hermetic environment of the other GUI checks (tests/live_env.py: a fake nvidia-smi and two fake llama-servers,
Control in this process) with a fake pxa-update that reports an update waiting (so the footer's Update button shows), then runs
the check in Chromium inside the playwright container for every theme and width, saving screenshots of Home, Models and Servers
scrolled to the bottom.

    python3 tests/footer-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR] [--require]

Needs docker and the playwright image. No GPU, no network beyond 127.0.0.1. Run it pinned (taskset -c ...): the browser's
container gets the same cores. Exit 0 = every check passed at every theme and width. Skipped (exit 0, says so) when docker or
the image is missing, unless --require is given."""
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
]

FAKE_UPDATER = r"""#!/bin/sh
case "$1" in
  check)    echo "current=4.0-beta.5 latest=4.0-beta.6 update=yes channel=stable" ;;
  channels) echo "channels=stable beta=no current=stable" ;;
  notes)    printf '{"version":"4.0-beta.6","build_id":"x","notes":"notes"}\n' ;;
  *)        exit 2 ;;
esac
"""


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
    out = a.out or tempfile.mkdtemp(prefix="pxa-footer-shots-")
    os.makedirs(out, exist_ok=True)
    upd = tempfile.mkdtemp(prefix="pxa-fake-upd-")
    b = os.path.join(upd, "pxa-update")
    with open(b, "w") as f:
        f.write(FAKE_UPDATER)
    os.chmod(b, 0o755)
    os.environ.update(PXA_UPDATE_BIN=b, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    import live_env as E
    env = E.Env(cards=4, servers=SERVERS).start()
    rc, fails = 0, 0
    try:
        for theme in a.themes.split(","):
            for width in a.widths.split(","):
                cmd = ["docker", "run", "--rm", "--network", "host", "--cpu-shares=2"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules",
                       "-v", HERE + ":/t:ro", "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/footer-gui-check.js", env.url, width,
                       "/out", theme]
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
