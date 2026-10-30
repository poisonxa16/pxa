#!/usr/bin/env python3
"""Headless-browser check of the rig picker's offline screen (tools/pxa_control_ui/hosts.html; see hosts-offline-gui-check.js).

It serves tools/pxa_control_ui on a loopback port (the picker) and keeps a second port (the rig) closed until the browser
asks for it, then starts a tiny "rig" there (it answers /api/health and /), so the check sees the page reopen the rig by
itself. Runs Chromium inside the playwright container.

    python3 tests/hosts-offline-gui-check.py [--out DIR] [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

Run it pinned (taskset -c ...) and the browser's container gets the same cores. No GPU, no network beyond 127.0.0.1.
Exit 0 = every check passed. Skipped (exit 0, says so) when docker or the image is missing, unless --require is given."""
import argparse
import functools
import http.server
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(os.path.dirname(HERE), "tools", "pxa_control_ui")


def cpuset():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []
    return [] if len(cpus) >= (os.cpu_count() or 0) else ["--cpuset-cpus", ",".join(map(str, cpus))]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class Rig(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok":true}' if self.path.startswith("/api/health") else b"<!doctype html><title>rig</title><p>rig page</p>"
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if self.path.startswith("/api/") else "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not shutil.which("docker") or subprocess.run(["docker", "image", "inspect", a.image], capture_output=True).returncode:
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-hosts-offline-")
    os.makedirs(out, exist_ok=True)
    flag = os.path.join(out, "OPEN-RIG")
    if os.path.exists(flag):
        os.unlink(flag)
    picker = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=UI))
    threading.Thread(target=picker.serve_forever, daemon=True).start()
    rig_port = free_port()
    base, rig = "http://127.0.0.1:%d" % picker.server_address[1], "http://127.0.0.1:%d" % rig_port
    cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
           "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/hosts-offline-gui-check.js", base, rig, "/out"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rig_srv = None
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < 300:
        if rig_srv is None and os.path.exists(flag):
            rig_srv = http.server.ThreadingHTTPServer(("127.0.0.1", rig_port), Rig)
            threading.Thread(target=rig_srv.serve_forever, daemon=True).start()
        time.sleep(0.2)
    if proc.poll() is None:
        proc.kill()
    sys.stdout.write(proc.stdout.read())
    picker.shutdown()
    if rig_srv:
        rig_srv.shutdown()
    print("screenshots in", out)
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
