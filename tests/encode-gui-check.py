#!/usr/bin/env python3
"""Headless-browser check of PXA Control's Encode tab (see encode-gui-check.js for what it clicks).

It starts the hermetic environment of tests/encode_env.py (fake GPUs, fake encoder packages, fake Hugging Face, fake licence server,
fake engine, PXA Control itself) and runs the click-through in Chromium inside the playwright container, once at desktop width and
once at 390 px phone width, saving a screenshot per step.

    python3 tests/encode-gui-check.py [--widths 1280,390] [--out DIR] [--image mcr.microsoft.com/playwright/mcp:latest]

Needs docker and the playwright image (playwright-core + Chromium). No GPU, no network beyond 127.0.0.1. Exit 0 = every check passed
at every width with zero page errors. Skipped (exit 0, says so) when docker or the image is missing, unless --require is given."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encode_env as E  # noqa: E402

CHROME = "/ms-playwright/chromium-1229/chrome-linux64/chrome"


def have_docker(image):
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return r.returncode == 0


def run_width(width, out, image):
    env = E.Env().start_fakes().start_control()
    cp = env.start_ctl()
    try:
        denv = dict(os.environ)
        denv["PATH"] = env.orig_path
        cmd = ["docker", "run", "--rm", "--network", "host", "-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro", "-v", out + ":/out",
               "--entrypoint", "node", image, "/t/encode-gui-check.js", env.url, str(width), "/out", str(cp)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, env=denv)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-3000:])
        return r.returncode
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1280,390")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-encode-shots-")
    os.makedirs(out, exist_ok=True)
    rc = 0
    for w in [x for x in a.widths.split(",") if x]:
        print("=== viewport width %s ===" % w, flush=True)
        r = run_width(int(w), out, a.image)
        rc = rc or r
    print("screenshots in", out)
    print("OVERALL:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
