#!/usr/bin/env python3
"""Start-up of a PXA Control build: seconds from exec to GET /api/health = 200 (the server is up and the browser
would open), and to the signed-in page (GET /?token= -> 303 -> GET / = 200). Run 1 is the cold start (fresh HOME and
cache: a onefile build unpacks), the next runs are warm. Fake cards, no browser, no tray.
    measure-start.py <binary or 'python3 -m pxa_control'> [runs]"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request


def wait_ok(url, t0, limit=180):
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(0.02)
    return None


def main():
    cmd = shlex.split(sys.argv[1])
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    home = tempfile.mkdtemp(prefix="pxa-start-")
    env = dict(os.environ, HOME=home, XDG_CACHE_HOME=os.path.join(home, ".cache"), PXA_CONTROL_CONFIG_DIR=os.path.join(home, "cfg"),
               PXA_LAUNCH_FAKE_GPUS="1x600", PXA_CONTROL_TELEMETRY="0", PXA_CONTROL_DISCOVER="0", PXA_CONTROL_TRAY="0")
    out = []
    for i in range(runs):
        port = 17700 + i
        t0 = time.time()
        p = subprocess.Popen(cmd + ["--no-browser", "--no-tray", "--port", str(port)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ready = wait_ok(f"http://127.0.0.1:{port}/api/health", t0)
        page = None
        try:
            run = os.path.join(home, "cfg", "run")
            tok = next(json.load(open(os.path.join(run, n)))["token"] for n in os.listdir(run) if n.endswith(".json"))
            opener = urllib.request.build_opener()          # follows the 303 with the cookie it set
            from http.cookiejar import CookieJar
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
            with opener.open(f"http://127.0.0.1:{port}/?token={tok}", timeout=20) as r:
                if r.status == 200 and b"<html" in r.read(4096).lower():
                    page = time.time() - t0
        except Exception as e:      # noqa: BLE001
            page = f"error {e.__class__.__name__}: {e}"
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
        out.append({"run": i + 1, "ready_s": round(ready, 3) if ready else None,
                    "page_s": round(page, 3) if isinstance(page, float) else page})
    print(json.dumps({"cmd": sys.argv[1], "runs": out}))


if __name__ == "__main__":
    main()
