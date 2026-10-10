#!/usr/bin/env python3
"""Headless-browser check that the Expert map card shows the server's own disclaimer (see
tests/emap-disclaimer-ui-check.js for what it compares).

It starts the hermetic environment of tests/encode_env.py (fake GPUs, a fake engine folder, PXA Control
itself) and runs the check in Chromium inside the playwright container against that Control, so the string
on screen is compared with the string the running server serves.

    python3 tests/emap-disclaimer-ui-check.py [--out DIR] [--image mcr.microsoft.com/playwright/mcp:latest]

Needs docker and the playwright image. No GPU, no network beyond 127.0.0.1. Exit 0 = the card shows the
served disclaimer. Skipped (exit 0, says so) when docker or the image is missing, unless --require is given."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def have_docker(image):
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def run(out, image):
    import encode_env as E
    env = E.Env().start_fakes().start_control()
    try:
        denv = dict(os.environ)
        denv["PATH"] = env.orig_path
        cmd = ["docker", "run", "--rm", "--network", "host", "-e", "NODE_PATH=/app/node_modules",
               "-v", HERE + ":/t:ro", "-v", out + ":/out",
               "--entrypoint", "node", image, "/t/emap-disclaimer-ui-check.js", env.url, "/out"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=denv)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-3000:])
        return r.returncode
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-emap-shots-")
    os.makedirs(out, exist_ok=True)
    # the playwright image runs as its own user, so a private temp dir is not writable from inside it
    try:
        os.chmod(out, 0o777)
    except OSError:
        pass
    rc = run(out, a.image)
    print("screenshots in", out)
    print("OVERALL:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
