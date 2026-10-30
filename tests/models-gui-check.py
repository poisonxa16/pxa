#!/usr/bin/env python3
"""Headless-browser check of the Models tab: the folders card above the Library, and the PXA list read on demand
(see models-gui-check.js for the checks). It starts a hermetic PXA Control (tests/live_env.py, no servers) pointed at a
FAKE huggingface.co (tests/fake_hf.py: no real network), seeds the remembered list the way a past visit left it, and runs
the check in Chromium inside the playwright container.

    python3 tests/models-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR]
                                      [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

Phases (each on a fresh page): main (every theme x width), then at the first theme/width: stale (list 30 h old),
nomodel (no model in the folders), nocache (nothing remembered). Run it pinned (taskset -c ...): the browser container gets
the same cores. Needs docker and the playwright image; skipped (exit 0) without them unless --require."""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def cpuset():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []
    return [] if len(cpus) >= (os.cpu_count() or 0) else ["--cpuset-cpus", ",".join(map(str, cpus))]


def have_docker(image):
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def write_gguf(path, size=8 << 20):
    """a header + tensor directory the library scan accepts (a dense 'llama' with one block), sparse to `size` bytes"""
    def s_(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, "llama"), ("general.name", 8, "Local Test Model")]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)))
        for k, t, v in kv:
            f.write(s_(k) + struct.pack("<I", t) + s_(v))
        f.write(s_("blk.0.ffn_down.weight") + struct.pack("<I", 1) + struct.pack("<Q", 1 << 20) + struct.pack("<I", 1) + struct.pack("<Q", 0))
        f.write(b"\0" * (32 - f.tell() % 32))
        f.truncate(f.tell() + size)
    return path


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
    out = a.out or tempfile.mkdtemp(prefix="pxa-models-shots-")
    os.makedirs(out, exist_ok=True)
    import fake_hf
    import live_env as E
    origin = fake_hf.Origin()
    env = E.Env(cards=2, servers=[]).start()
    C = env.C
    saved = C.HF.HF
    C.HF.HF = origin.url                                   # Control reads this origin, never huggingface.co
    app = env.app
    cache = os.path.join(C.config_dir(), "hf-catalog.json")
    tiny = write_gguf(os.path.join(env.models, "Local-Test-PXQ4.gguf"))
    keep = tiny + ".away"

    def seed(age_s, model=True):
        """leave the state a past visit would have: a remembered list read age_s ago on disk, nothing in memory"""
        if os.path.exists(cache):
            os.unlink(cache)
        app.hf.data, app.hf.at, app._hf_saved_ts = None, 0.0, None
        if age_s is not None:
            app.hf_available(force=True)
            with open(cache) as f:
                d = json.load(f)
            d["ts"] = time.time() - age_s
            with open(cache, "w") as f:
                json.dump(d, f)
            app.hf.data, app.hf.at, app._hf_saved_ts = None, 0.0, None
        if model and os.path.exists(keep):
            os.rename(keep, tiny)
        if not model and os.path.exists(tiny):
            os.rename(tiny, keep)
        app.models(force=True)

    def check(theme, width, phase):
        cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
               "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/models-gui-check.js", env.url, origin.url, width, "/out", theme, phase]
        print("== %s %s px %s" % (theme, width, phase), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-2000:])
        return r.returncode == 0

    rc = 0
    try:
        for theme in a.themes.split(","):
            for width in a.widths.split(","):
                seed(40 * 60)
                if not check(theme, width, "main"):
                    rc = 1
        t0, w0 = a.themes.split(",")[0], a.widths.split(",")[0]
        for phase, age, model in (("stale", 30 * 3600, True), ("nomodel", 10 * 60, False), ("nocache", None, True)):
            seed(age, model)
            if not check(t0, w0, phase):
                rc = 1
    finally:
        C.HF.HF = saved
        env.close()
        origin.close()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
