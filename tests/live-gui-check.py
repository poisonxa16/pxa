#!/usr/bin/env python3
"""Headless-browser check of PXA Control's Live tab (see live-gui-check.js for what it checks).

It starts the hermetic environment of tests/live_env.py (a fake nvidia-smi, two fake llama-servers running a scripted workload, and PXA
Control itself), lets the history build for --warm seconds, and runs the check in Chromium inside the playwright container for every
theme and width, saving screenshots.

    python3 tests/live-gui-check.py [--themes dark,light] [--widths 1280,390] [--warm 150] [--out DIR] [--base URL]
                                    [--image mcr.microsoft.com/playwright/mcp:latest]

--base URL checks an environment that is already running (python3 tests/live_env.py) instead of starting one.
Needs docker and the playwright image. No GPU, no network beyond 127.0.0.1. Exit 0 = every check passed at every theme and width.
Skipped (exit 0, says so) when docker or the image is missing, unless --require is given."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def have_docker(image):
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1280,390")
    ap.add_argument("--warm", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-live-shots-")
    os.makedirs(out, exist_ok=True)
    env = None
    base = a.base
    if not base:
        import live_env as E
        env = E.Env().start()
        base = env.url
        # the page is what keeps the sampler running: look at it while the history builds
        t_end = time.time() + a.warm
        import urllib.request
        while time.time() < t_end:
            try:
                urllib.request.urlopen(base + "/api/live?since=9999999999", timeout=10).read()
            except Exception:
                pass
            time.sleep(2)
    rc = 0
    try:
        for theme in a.themes.split(","):
            for width in a.widths.split(","):
                cmd = ["docker", "run", "--rm", "--network", "host", "-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro", "-v", out + ":/out",
                       "--entrypoint", "node", a.image, "/t/live-gui-check.js", base, width, "/out", theme]
                print("== %s %s px" % (theme, width), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                sys.stdout.write(r.stdout)
                if r.returncode != 0:
                    sys.stdout.write(r.stderr[-2000:])
                    rc = 1
    finally:
        if env:
            env.close()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
