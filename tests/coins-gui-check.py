#!/usr/bin/env python3
"""Real-browser click-through of the PXA Coins panel (Store tab) against a LOCAL licd (see coins-gui-check.js for what it checks).

It starts, in this process, a real licd (tools/pxa-licd, PXA_LICD_DIR) on 127.0.0.1 with a scratch sqlite db and
throwaway keys generated here (never the store's), PXA Control (tests/live_env.py, no fake servers) pointed at it, and a tiny loopback
"hooks" server the browser script calls for what only the owner can do on the licd side (a chargeback, reading the saved statement).
Then it runs the check in Chromium inside the playwright container for every theme and width, saving screenshots.

    PXA_LICD_DIR=<licd-tree>/tools/pxa-licd python3 tests/coins-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR]
                                                                           [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

A fresh licd + Control + wallet per theme x width (the flow spends coins). Run it pinned (taskset -c ...): the browser container gets the
same cores. Needs docker and the playwright image. No GPU, no network beyond 127.0.0.1. Exit 0 = every check passed at every theme and
width. Skipped (exit 0, says so) when docker, the image or the licd tree is missing, unless --require is given."""
import argparse
import json
import glob
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
# PXA_LICD_DIR names the licd tree; else a sibling checkout of this tree that carries tools/pxa-licd
CANDIDATES = [os.environ.get("PXA_LICD_DIR") or ""] + sorted(glob.glob(os.path.join(os.path.dirname(ROOT), "*", "tools", "pxa-licd")))
LICD_DIR = next((p for p in CANDIDATES if p and os.path.exists(os.path.join(p, "wallet.py"))), "")
KOFI = "kofi-test-token-gui"
TRIAGE = "triage-token-for-the-gui-test-0123456789"


def cpuset():
    """the cores this run was pinned to (taskset), handed on to the browser's container: a pinned check stays pinned"""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []
    return [] if len(cpus) >= (os.cpu_count() or 0) else ["--cpuset-cpus", ",".join(map(str, cpus))]


def have_docker(image):
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


class Cell(object):
    """one licd + one Control + the hooks server: a fresh wallet world for one theme x width"""

    def __init__(self):
        import licd
        import store as licd_store
        import live_env as E
        from pxa_control import store as _S
        _S.VIEW_CACHE_S = 0     # the browser check polls every 3 s and changes licd behind Control (review R1 cache off)
        self.tmp = tempfile.mkdtemp(prefix="pxa-coins-gui-")
        d = os.path.join(self.tmp, "licd")
        keys = licd_store.keys_init(os.path.join(d, "secrets"), "ent-gui", "plug-gui")
        for name, val in (("kofi", KOFI), ("svc-triage.token", TRIAGE)):
            with open(os.path.join(d, name), "w") as f:
                f.write(val + "\n")
        os.makedirs(os.path.join(d, "files"))
        env = {"PXA_LIC_DB": os.path.join(d, "db.sqlite"), "PXA_LIC_PORT": "0", "PXA_LIC_STORE_SEED": keys["ent_seeds"]["ent-gui"],
               "PXA_LIC_STORE_KID": "ent-gui", "PXA_LIC_KOFI_TOKEN": os.path.join(d, "kofi"), "PXA_LIC_STORE_FILES_DIR": os.path.join(d, "files"),
               "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": "", "PXA_LIC_SVC_TRIAGE": os.path.join(d, "svc-triage.token")}
        self.lsvc = licd.Service(licd.Config(env))
        self.lsrv, _ = licd.serve(licd.Config(env), self.lsvc)
        threading.Thread(target=self.lsrv.serve_forever, daemon=True).start()
        self.lurl = "http://127.0.0.1:%d" % self.lsrv.server_address[1]
        self.env = E.Env(cards=1, servers=[])
        os.environ["PXA_STORE_URL"] = self.lurl          # live_env says "" (fixture); this world has a real licd
        from pxa_control import store as S
        self.S = S
        S.GRANT_PUBKEYS.clear()
        S.load_pubkeys(keys["pubkeys_file"])
        S.STMT_GAP_S = 1.0                               # the statement follows a change within a second (a test, not a release value)
        S._STMT["at"] = -1e18
        S._EARN["rows"], S._EARN["at"] = None, -1e18
        self.env.start()
        cell = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def _send(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.startswith("/statement"):
                    st = S.load_statement(os.environ["PXA_CONTROL_CONFIG_DIR"])
                    return self._send({"statement": st and {"wallet": st.get("wallet"), "coins": st.get("coins"), "owned": st.get("owned")}})
                if self.path.startswith("/gift-rows"):
                    return self._send({"n": cell.lsvc.db.q1("SELECT COUNT(*) n FROM store_ledger WHERE kind='gift'")["n"]})
                if self.path.startswith("/spend-rows"):
                    return self._send({"n": cell.lsvc.db.q1("SELECT COUNT(*) n FROM store_ledger WHERE kind='spend'")["n"]})
                self._send({"error": "no"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if self.path.startswith("/chargeback"):
                    r = cell.lsvc.store.deposit("admin", body["wallet"], -abs(int(body["coins"])), "chargeback (gui check)")
                    cell.lsvc._rl_tab = {}
                    return self._send({"ok": True, "r": str(r)[:80]})
                if self.path.startswith("/other-wallet"):           # a second supporter's wallet (another install key) to gift to
                    import ed25519
                    w = cell.lsvc.store.wallet_create("machine", ed25519.public_key(os.urandom(32)).hex(), "10.9.9.9")
                    return self._send({"wallet": w["wallet_id"]})
                if self.path.startswith("/voucher"):               # a voucher PXA made (the voucher era, or one sent by hand)
                    v = cell.lsvc.store.new_vouchers("gui", int(body.get("coins") or 25), 1)[0]
                    return self._send({"code": v["code"]})
                if self.path.startswith("/offline"):               # point Control at a closed port, or back at licd
                    os.environ["PXA_STORE_URL"] = "http://127.0.0.1:1" if body.get("on") else cell.lurl
                    return self._send({"ok": True})
                self._send({"error": "no"})
        self.hooks = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.hooks.serve_forever, daemon=True).start()
        self.hurl = "http://127.0.0.1:%d" % self.hooks.server_address[1]

    @property
    def url(self):
        return self.env.url

    def close(self):
        os.environ["PXA_STORE_URL"] = ""
        for s in (self.hooks, self.lsrv):
            try:
                s.shutdown()
                s.server_close()
            except Exception:
                pass
        self.env.close()
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
    if not LICD_DIR:
        print("SKIP: the licd tree with the coins routes is not found (PXA_LICD_DIR)")
        return 1 if a.require else 0
    sys.path.insert(0, LICD_DIR)
    out = a.out or tempfile.mkdtemp(prefix="pxa-coins-shots-")
    os.makedirs(out, exist_ok=True)
    rc = 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            cell = Cell()
            try:
                cmd = ["docker", "run", "--rm", "--network", "host"] + cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                       "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/coins-gui-check.js", cell.url, width, "/out", theme,
                       cell.hurl, cell.lurl, KOFI, TRIAGE]
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
