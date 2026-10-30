#!/usr/bin/env python3
"""Real-browser check of a plugin's OWN top-level tab in PXA Control (manifest ui.tab.nav = true), with the real Wayhouse plugin:
the signed .pxp is built from tools/pxa-private/plugins/wayhouse, installed and allowed in a hermetic Control (tests/live_env.py) that
watches one tiny OpenAI-compatible server; the browser (playwright in docker) then checks: the "Wayhouse" button sits in Control's tab
bar before Plugins, the tab opens with "Turn on", the page runs in a sealed frame filling the tab, a project + a chat answer work
through the host, a deep link (#plug-wayhouse) opens the tab, "Manage in Plugins" goes back, and turning it off shows the off panel.
dark + light x 1280 + 390 px; no horizontal page scroll; coins / supporter wording only.

    python3 tests/plugin-nav-gui-check.py [--out DIR] [--require]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
SRC = os.path.join(ROOT, "tools", "pxa-private", "plugins", "wayhouse")
ANSWER = "The Tesla P100 has **16 GB** of HBM2 memory."


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


def tiny_model():
    """a loopback OpenAI-compatible chat server with a scripted answer (the seat Control watches)."""
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            return

        def _send(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._send(200, {"status": "ok"})
            if self.path.startswith("/v1/models"):
                return self._send(200, {"data": [{"id": "tiny-chat.gguf"}]})
            if self.path.startswith("/props"):
                return self._send(200, {"default_generation_settings": {"n_ctx": 8192}})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            if not self.path.startswith("/v1/chat/completions"):
                return self._send(404, {"error": "not found"})
            self._send(200, {"model": "tiny-chat.gguf", "choices": [{"message": {"role": "assistant", "content": ANSWER},
                                                                     "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 40, "completion_tokens": 12},
                             "timings": {"predicted_per_second": 41.5, "prompt_per_second": 900.0}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Cell(object):
    """one hermetic Control with the real Wayhouse plugin installed and allowed (not yet running)."""

    def __init__(self):
        import live_env as E
        import plugin_ui_env as PE
        self.tmp = tempfile.mkdtemp(prefix="pxa-plugin-nav-gui-")
        self.model = tiny_model()
        self.env = E.Env(cards=1, servers=[], real=[{"name": "tiny-chat", "port": self.model.server_address[1], "gpus": [0]}])
        from pxa_control import store as S
        from pxa_control import plugins as P
        from pxa_control.plugins import host as PH, pxp
        cfg = os.environ["PXA_CONTROL_CONFIG_DIR"]
        path, _body = PE._build_mod().build(SRC, PE.PLUG_SEED, PE.PLUG_KID, os.path.join(self.tmp, "out"))
        pxp.install(path, cfg, PE.plugin_keys())
        m = pxp.read_manifest(path, PE.plugin_keys())
        P.ConsentStore(cfg).grant(m, m["permissions"], by="gui-check")
        self.env.start()
        self.app = self.env.app
        self.app.store_cache = S.EntitlementCache(cfg)
        self.app.plugin_host = self.host = PH.PluginHost(cfg, cache=self.app.store_cache, pubkeys=PE.plugin_keys(),
                                                         model_chat=PH._seat_chat(self.app), model_list=PH._seat_list(self.app))

    @property
    def url(self):
        return self.env.url

    def close(self):
        for f in (self.host.close, self.env.close, self.model.shutdown):
            try:
                f()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


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
    out = a.out or tempfile.mkdtemp(prefix="pxa-plugin-nav-shots-")
    os.makedirs(out, exist_ok=True)
    rc = 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            cell = Cell()
            try:
                cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                       "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/plugin-nav-gui-check.js", cell.url, width, "/out", theme]
                print("== %s %s px" % (theme, width), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                sys.stdout.write(r.stdout)
                if r.returncode != 0:
                    sys.stdout.write(r.stderr[-2000:])
                    rc = 1
            finally:
                cell.close()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
