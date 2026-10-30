#!/usr/bin/env python3
"""Headless-browser check of two owed chat-page items (todo control-chat-hardening): the BROKEN-CHAT ROW (a corrupt session file is
listed marked "Broken" with Export raw + Delete in view; opening it shows a friendly card, not an error; delete works) and the
bad_image TOAST (the server's 400 bad_image reason is shown as a visible notice and the typed words come back). See
tests/chat-owed-gui-check.js. Same hermetic environment as tests/chat-ui-gui-check.py (scratch Control + FAKE llama-server, no GPU).

    python3 tests/chat-owed-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR] [--tree DIR] [--phases broken,image]
                                         [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

Run it pinned (taskset -c 36-53 ...). Exit 0 = every check passed at every theme and width. Skipped (exit 0) when docker or the image is missing, unless --require."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = '{"id": "c_corrupt0001", "title": "broken one", "turns": ['


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
    ap.add_argument("--phases", default="broken,image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tree", default=os.path.dirname(HERE))
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    sys.path.insert(0, os.path.join(a.tree, "tools", "pxa_control", "tests"))
    import harness_suite as H
    out = a.out or tempfile.mkdtemp(prefix="pxa-chatowed-shots-")
    os.makedirs(out, exist_ok=True)
    rc = 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            for phase in a.phases.split(","):
                work = tempfile.mkdtemp(prefix="pxa-chatowed-")
                ctl = H.Control(work, work)
                fake = H.Fake("A", work, "fake-agent-8b", ctx=8192, slots=2).start()
                os.makedirs(ctl.cfg, mode=0o700, exist_ok=True)
                with open(os.path.join(ctl.cfg, "control.json"), "w") as f:
                    json.dump({"attach_port": fake.port}, f)
                if phase == "broken":       # a damaged chat file, as a crash mid-write leaves it
                    sd = os.path.join(ctl.cfg, "chat", "sessions")
                    os.makedirs(sd, mode=0o700, exist_ok=True)
                    with open(os.path.join(sd, "c_corrupt0001.json"), "w") as f:
                        f.write(RAW)
                ctl.start()
                try:
                    base = "http://127.0.0.1:%d/?token=%s" % (ctl.port, ctl.token)
                    cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                           "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/chat-owed-gui-check.js", base, width, "/out", theme, phase,
                           RAW]
                    print("== %s %s px %s" % (theme, width, phase), flush=True)
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                    sys.stdout.write(r.stdout)
                    if r.returncode != 0:
                        sys.stdout.write(r.stderr[-2000:])
                        rc = 1
                    if phase == "broken" and os.path.exists(os.path.join(ctl.cfg, "chat", "sessions", "c_corrupt0001.json")):
                        print("  FAIL the damaged file is still on disk after Delete")
                        rc = 1
                finally:
                    ctl.stop()
                    fake.stop()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
