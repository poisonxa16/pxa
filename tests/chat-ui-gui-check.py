#!/usr/bin/env python3
"""Headless-browser check of the chat Assistant page's hardening: bidi controls (U06), a cut text paste (C15), a 1 MB reply
(S21, prints render numbers), and a Control restart mid-stream (E07: reconnect with backoff, resume with no repeated text,
"signed out" with a sign-in action on a 401). See tests/chat-ui-gui-check.js for what each phase clicks.

It starts the hermetic environment of tools/pxa_control/tests/harness_suite.py (a scratch Control from the checkout + a FAKE
llama-server, no GPU, no model) and a small helper the browser script calls to crash and restart that Control, then runs
the check in Chromium inside the playwright container for every theme and width.

    python3 tests/chat-ui-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR] [--tree DIR] [--phases plain,restart,newtoken]
                                       [--long-down 70] [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

--tree DIR runs the same check against another checkout (a base tree, to measure "before"); --phases picks the phases (longdown only
with --long-down SECONDS, run once at the first theme and width). Run it pinned (taskset -c 36-53 ...): the browser's container
gets the same cores. Needs docker and the playwright image. Exit 0 = every check passed at every theme and width.
Skipped (exit 0, says so) when docker or the image is missing, unless --require is given."""
import argparse
import http.server
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


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


class Helper(object):
    """GET /crash (kill -9 Control) and GET /start?token=same|new (start it again, new token if asked) for the browser script."""

    def __init__(self, ctl):
        self.ctl = ctl
        me = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query)
                if u.path == "/crash":
                    me.ctl.crash()
                    out = {"ok": True}
                elif u.path == "/start":
                    if q.get("token", ["same"])[0] == "new":
                        me.ctl.token = secrets.token_urlsafe(24)
                    me.ctl.start()
                    out = {"ok": True, "url": "http://127.0.0.1:%d/?token=%s" % (me.ctl.port, me.ctl.token)}
                else:
                    out = {"ok": False}
                b = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1280,390")
    ap.add_argument("--phases", default="plain,restart,newtoken")
    ap.add_argument("--long-down", type=int, default=0)
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
    out = a.out or tempfile.mkdtemp(prefix="pxa-chatui-shots-")
    os.makedirs(out, exist_ok=True)
    rc = 0
    first = True
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            phases = a.phases.split(",") + (["longdown"] if a.long_down and first else [])
            first = False
            for phase in phases:
                work = tempfile.mkdtemp(prefix="pxa-chatui-")
                ctl = H.Control(work, work)
                fake = H.Fake("A", work, "fake-agent-8b", ctx=8192, slots=2).start()
                os.makedirs(ctl.cfg, mode=0o700, exist_ok=True)
                with open(os.path.join(ctl.cfg, "control.json"), "w") as f:
                    json.dump({"attach_port": fake.port}, f)
                ctl.start()
                helper = Helper(ctl)
                try:
                    base = "http://127.0.0.1:%d/?token=%s" % (ctl.port, ctl.token)
                    cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                           "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/chat-ui-gui-check.js", base, width, "/out", theme, phase,
                           helper.url, str(a.long_down)]
                    print("== %s %s px %s" % (theme, width, phase), flush=True)
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
                    sys.stdout.write(r.stdout)
                    if r.returncode != 0:
                        sys.stdout.write(r.stderr[-2000:])
                        rc = 1
                finally:
                    helper.close()
                    ctl.stop()
                    fake.stop()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
