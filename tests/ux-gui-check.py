#!/usr/bin/env python3
"""Headless-browser check of the PXA Control UX revamp (2026-10): Home in its three states, the grouped navigation, the
theme (system + toggle), every section opening without a page error or a failed request, the Plugins and Store
sections (Anibal hero, consent, activation, locked states, offline grace), at desktop and phone widths, both themes.
See ux-gui-check.js for the checks; every step saves a screenshot.

    python3 tests/ux-gui-check.py [--modes first,ready,serving] [--themes dark,light] [--widths 1440,390] [--out DIR]
                                  [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

Each mode is its own hermetic environment (tests/live_env.py: a fake nvidia-smi, fake llama-servers):
  first   - no model folder, no model, nothing running: Home is the three-step first run;
  ready   - one model, nothing running: Home is "Run a model" with one Start;
  serving - two fake servers running: Home leads with Chat.
Needs docker and the playwright image; no GPU, no network beyond 127.0.0.1 (huggingface.co may be unreachable: the
first-run page must then say so and offer the folder path). Skipped (exit 0, says so) without docker unless --require."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def have_docker(image):
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Origin": base})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="first,ready,serving")
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1440,390")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-ux-shots-")
    os.makedirs(out, exist_ok=True)
    import live_env as E
    rc = 0
    for mode in [m for m in a.modes.split(",") if m]:
        env = E.Env(cards=4, servers=None if mode == "serving" else []).start()
        try:
            if mode == "first":                       # no folder, no model: the real first run
                os.remove(os.path.join(env.models, "tiny.gguf"))
                post(env.url, "/api/models/dirs", {"model_dirs": []})
            elif mode == "ready":                     # two small models the library lists (blk.* tensors): a free PXQ and a supporter PXQN
                from fake_encode_tools import write_gguf
                write_gguf(os.path.join(env.models, "Qwen3-4B-PXQ4.gguf"), 4, 252)
                write_gguf(os.path.join(env.models, "Qwen3-4B-PXQN3.gguf"), 4, 257)
                urllib.request.urlopen(env.url + "/api/models?force=1", timeout=120).read()
            cmd = ["docker", "run", "--rm", "--network", "host", "-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro", "-v", out + ":/out",
                   "--entrypoint", "node", a.image, "/t/ux-gui-check.js", env.url, mode, a.themes, a.widths, "/out"]
            print("== mode %s" % mode, flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
