#!/usr/bin/env python3
"""Headless-browser check of two PXA Control fixes (see update-hs-gui-check.js for what it checks):
  update  FR#14: the Update dialog asks again on every open and says "Stop the running servers first" only when an update waits AND a
          server runs (it used to read a count taken at page load, and to show the stop line with nothing to install)
  hs      FR#13: Submit closes the "New high score" frame like Not now, the board's answer shows as a toast, and a result already sent
          never brings the frame or the Check button back (not after a reload either); a new run hides an open frame

It starts the hermetic environment of speed-gui-check.py (tests/live_env.py: fake nvidia-smi, fake llama-servers, Control in this
process), a fake pxa-update (PXA_UPDATE_BIN: stable = up to date, beta = 4.0-beta.1 waiting) and a fake high-score board on 127.0.0.1
(PXA_BUG_URL points Control's board calls at it), then runs the check in Chromium inside the playwright container for every theme and
width, saving screenshots. No GPU, no network beyond 127.0.0.1.

    python3 tests/update-hs-gui-check.py [--themes dark,light] [--widths 1280,390] [--phases hs,update] [--out DIR] [--require]

Exit 0 = every check passed at every theme and width. Skipped (exit 0, says so) when docker or the image is missing, unless --require."""
import argparse
import http.server
import importlib.util
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("speed_gui_check", os.path.join(HERE, "speed-gui-check.py"))
SG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SG)

FAKE_UPDATER = r"""#!/bin/sh
d="$PXA_FAKE_UPD_DIR"
echo "$*" >> "$d/calls"
case "$1" in
  check)    ch=$(cat "$d/channel" 2>/dev/null || echo stable)
            if [ "$ch" = beta ]; then echo "current=4.0-beta.5 latest=4.0-beta.6 update=yes channel=beta";
            else echo "current=4.0-beta.5 latest=4.0-beta.5 update=no channel=stable"; fi ;;
  channels) echo "channels=stable,beta beta=yes current=stable" ;;
  channel)  echo "$2" > "$d/channel"; echo "channel=$2" ;;
  notes)    printf '{"version":"4.0-beta.6","build_id":"x","notes":"notes"}\n' ;;
  apply)    echo "installed" ;;
  *)        exit 2 ;;
esac
"""


class Board(object):
    """the community board's three calls, on 127.0.0.1: no record yet for any bracket; a submit is taken (rank 1), the next is held
    for review (flagged). Records every call."""

    def __init__(self):
        self.calls = []
        self.answers = [{"ok": True, "id": 11, "rank": 1, "status": "ok"}, {"ok": True, "id": 12, "rank": None, "status": "flagged"}]
        board = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                board.calls.append(("GET", self.path))
                self._send({"record": None, "top": []})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                board.calls.append(("POST", self.path))
                self._send(board.answers.pop(0) if board.answers else {"ok": True, "rank": 2, "status": "ok"})

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        self.srv = S(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1280,390")
    ap.add_argument("--phases", default="hs,update")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not SG.have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    out = a.out or tempfile.mkdtemp(prefix="pxa-update-hs-shots-")
    os.makedirs(out, exist_ok=True)
    import live_env as E
    rc, fails = 0, 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            upd = tempfile.mkdtemp(prefix="pxa-fake-upd-")
            b = os.path.join(upd, "pxa-update")
            with open(b, "w") as f:
                f.write(FAKE_UPDATER)
            os.chmod(b, 0o755)
            board = Board()
            os.environ.update(PXA_UPDATE_BIN=b, PXA_FAKE_UPD_DIR=upd, PXA_BUG_URL="http://127.0.0.1:%d/v1/report" % board.port,
                              NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
            sys.modules["pxa_control.routes"]._ENGINE_CH.update(t=0.0, beta=False, ttl=0, last="stable") if "pxa_control.routes" in sys.modules else None
            env = E.Env(cards=4, servers=[dict(s) for s in SG.SERVERS], library=True).start()
            try:
                SG.fake_launcher(env)
                cmd = ["docker", "run", "--rm", "--network", "host", "--cpu-shares=2"] + SG.cpuset() + ["-e", "NODE_PATH=/app/node_modules",
                       "-v", HERE + ":/t:ro", "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/update-hs-gui-check.js", env.url, width,
                       "/out", theme, a.phases]
                print("== %s %s px" % (theme, width), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                sys.stdout.write(r.stdout)
                if r.returncode != 0:
                    sys.stdout.write(r.stderr[-2000:])
                    rc = 1
                fails += r.stdout.count("  FAIL ")
                posts = [c for c in board.calls if c[0] == "POST"]
                ok = len(posts) <= 2
                print("  %s   the fake board saw %d submit(s) (one per sent result)" % ("ok" if ok else "FAIL", len(posts)))
                if not ok:
                    rc, fails = 1, fails + 1
            finally:
                env.close()
                board.close()
    print("screenshots in", out)
    print("TOTAL FAIL lines:", fails)
    return rc


if __name__ == "__main__":
    sys.exit(main())
