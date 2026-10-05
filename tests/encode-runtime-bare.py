#!/usr/bin/env python3
"""Control's GPU-runtime flow on a BARE machine (run inside a plain ubuntu container that has only the NVIDIA driver: no CUDA toolkit, no cuBLAS, no cuSOLVER).

It drives a running PXA Control over HTTP, the way the Encode tab does:
  1. nothing installed -> "Download Free": signed statement + sha256 checked, installed.
  2. "Use a Pro encoder I already downloaded": the real stamped Pro package (an older wrapper) and the same package with the new wrapper (pxqe_runtime.py):
     both say, in `info --json` runtime{}, that the CUDA libraries cuBLAS / cuSOLVER are missing (the new wrapper names both libraries).
  3. "Download the GPU runtime (one time, N)": GET /api/encode/runtime (the size), POST /api/encode/runtime/get, progress, verified, unpacked next to the encoders.
  4. both Pro encoders now load (the older one through the folder Control puts on its loader path, the new one by itself), `pxqe info --json` run by hand
     finds the pack next to the encoder with no environment at all, and every stage the Pro encoder runs on a card is `ready`.
Exit 0 only when every step passed. The licence server is a scratch one (tests/encode-runtime-scratch-licd.py) that serves the real pack and the real Free build.

    encode-runtime-bare.py --control URL --pro-old DIR --pro-new DIR --enchome DIR [--summary FILE]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def need(ok, msg):
    if not ok:
        say("FAIL " + msg)
        raise SystemExit(1)
    say("ok   " + msg)


class Api(object):
    def __init__(self, base):
        self.base = base.rstrip("/")

    def call(self, method, path, body=None, timeout=120):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {"error": "non-json answer"}


def pxqe_info(d, env_extra=None, strip=("PXQE_RUNTIME_DIR", "PXQE_CUDA_LIBS", "PXQE_ENGINE", "LD_LIBRARY_PATH")):
    env = {k: v for k, v in os.environ.items() if k not in strip}
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, os.path.join(d, "pxqe.py"), "info", "--json"], capture_output=True, text=True, env=env, timeout=60)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="http://127.0.0.1:7814")
    ap.add_argument("--pro-old", required=True)
    ap.add_argument("--pro-new", required=True)
    ap.add_argument("--enchome", required=True)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    api = Api(a.control)
    summary = {}

    # ---- 0. the machine is bare
    libs = subprocess.run("ldconfig -p | grep -E 'libcublas|libcusolver|libcusparse|libnvJitLink|libcudart' || true", shell=True, capture_output=True, text=True).stdout.strip()
    need(libs == "" and not [p for p in os.listdir("/usr/local") if p.startswith("cuda")], "bare machine: no CUDA library in ldconfig, no /usr/local/cuda*")
    need(subprocess.run(["sh", "-c", "ldconfig -p | grep -q 'libcuda.so.1'"]).returncode == 0, "the NVIDIA driver library (libcuda.so.1) is there: the container has the driver only")

    # ---- 1. Free
    st = api.call("GET", "/api/encode/state")[1]
    need(st["encoders"] == [] and st["runtime"]["applies"] is False, "Control starts empty; the GPU runtime does not apply (no Pro encoder)")
    code, r = api.call("POST", "/api/encode/get", {"edition": "free"})
    need(code == 200, "Download Free started")
    t0 = time.time()
    while time.time() - t0 < 300:
        st = api.call("GET", "/api/encode/state")[1]
        if not st["pkg"]["running"]:
            break
        time.sleep(0.5)
    need(st["pkg"]["phase"] == "done" and st["edition"] == "free", "Free installed (signature + sha256 verified): %s" % st["pkg"]["message"])
    summary["free"] = st["pkg"]["message"]

    # ---- 2. the Pro encoders (the real stamped package c; older wrapper and new wrapper) cannot load: what is missing is named
    seen = {}
    for label, d in (("old wrapper", a.pro_old), ("new wrapper", a.pro_new)):
        code, st = api.call("POST", "/api/encode/add-encoder", {"path": d})
        need(code == 200, "Use a Pro encoder I already downloaded: %s (%s)" % (label, d))
        enc = next(e for e in st["encoders"] if os.path.realpath(e["path"]).startswith(os.path.realpath(d) + os.sep))
        rt = enc["runtime"]
        need(enc["ok"] and enc["edition"] == "pro" and rt["lib"] == "unloadable" and rt["fix"] == "runtime-pack",
             "%s: the real libpxqe.so cannot load; the fix is the GPU runtime. Missing: %s | the loader said: %s" % (label, ", ".join(rt["missing"]), rt["detail"]))
        seen[label] = enc["path"]
        summary["before_" + label.split()[0]] = {"missing": rt["missing"], "detail": rt["detail"], "resolver": rt["resolver"]}
    need(summary["before_new"]["missing"] == ["libcublas.so.12", "libcusolver.so.11"] and summary["before_new"]["resolver"] == 1, "the new wrapper names both missing libraries exactly")
    code, st = api.call("POST", "/api/encode/use", {"path": seen["new wrapper"]})
    rt = st["runtime"]
    need(rt["need"] and "libcublas.so.12, libcusolver.so.11" in rt["text"], "Control says: " + rt["text"])
    code, v = api.call("GET", "/api/encode/runtime")
    need(code == 200 and v["offer"], "the offer: GPU runtime %s, %s to download, unpacks to %s (CUDA %s)" % (v["offer"]["id"], v["offer"]["size_h"], v["offer"]["unpacked_h"], v["offer"]["cuda"]))
    summary["offer"] = v["offer"]

    # ---- 3. the one button
    code, r = api.call("POST", "/api/encode/runtime/get", {})
    need(code == 200, "Download the GPU runtime (one time, %s) pressed" % v["offer"]["size_h"])
    t0, last = time.time(), None
    while time.time() - t0 < 1800:
        st = api.call("GET", "/api/encode/state")[1]
        j = st["runtime"]["job"]
        msg = "%s %d%%" % (j["phase"], round(100 * (j["pct"] or 0)))
        if int(time.time() - t0) % 10 == 0 and msg != last:
            say("  runtime: " + msg)
            last = msg
        if not j["running"]:
            break
        time.sleep(1.0)
    need(j["phase"] == "done", "the runtime downloaded, its signed statement + sha256 + file list checked, unpacked: %s" % j["message"])
    inst = st["runtime"]["installed"]
    need(inst and inst["id"] == v["offer"]["id"], "installed: %s (%s, CUDA %s) in %s" % (inst["id"], inst["size_h"], inst["cuda"], inst["dir"]))
    need(st["runtime"]["ok"] and st["runtime"]["source"] == "pack" and not st["runtime"]["need"], "the selected Pro encoder (new wrapper) loads: GPU libraries from the pack")
    summary["installed"] = inst
    summary["download_seconds"] = round(time.time() - t0)
    files = sorted(os.listdir(os.path.join(inst["dir"], "lib")))
    need(files == ["libcublas.so.12", "libcublasLt.so.12", "libcusolver.so.11", "libcusparse.so.12", "libnvJitLink.so.12"], "the pack's lib folder holds exactly the five libraries: " + ", ".join(files))
    need(os.path.isfile(os.path.join(inst["dir"], "LICENSE-NVIDIA-CUDA-EULA.txt")) and os.path.isfile(os.path.join(inst["dir"], "NOTICE.txt")), "the NVIDIA licence text and the notice are in the pack")

    # ---- 4. everything that needs the library now works
    code, st = api.call("POST", "/api/encode/rescan", {})
    for label, path in seen.items():
        enc = next(e for e in st["encoders"] if e["path"] == path)
        need(enc["runtime"]["lib"] == "loadable", "%s: libpxqe.so loads (source: %s, resolver %s)" % (label, enc["runtime"]["source"] or "-", enc["runtime"]["resolver"]))
        stages = {s["name"]: s["ready"] for s in enc["stages"]}
        need(all(stages.get(n) for n in ("skeleton", "hess", "encode", "splice")), "%s: the stages that run on a card are ready: %s" % (label, ", ".join(n for n, r in stages.items() if r)))
    # by hand, no Control, no environment: the pack sits next to the encoder (<root>/pro/<build> and <root>/runtime/<id>)
    root = a.enchome
    mine = os.path.join(root, "pro", "pxqe-by-hand")
    os.makedirs(os.path.dirname(mine), exist_ok=True)
    shutil.copytree(a.pro_new, mine, symlinks=True)
    info = pxqe_info(mine)
    need(info["runtime"]["lib"] == "loadable" and info["runtime"]["source"] == "pack", "`pxqe info --json` by hand, no environment: runtime %s from the %s (%s)" % (info["runtime"]["lib"], info["runtime"]["source"], info["runtime"].get("dir")))
    old = pxqe_info(a.pro_old, {"LD_LIBRARY_PATH": os.path.join(inst["dir"], "lib")})
    need(old["runtime"]["lib"] == "loadable", "the older wrapper loads through the loader path alone (what Control sets for it)")
    off = pxqe_info(mine, {"PXQE_CUDA_LIBS": "/nonexistent"})
    need(off["runtime"]["lib"] == "loadable", "an explicit PXQE_CUDA_LIBS that has nothing falls back to the pack instead of failing")
    shutil.rmtree(mine)
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump(summary, f, indent=1)
    say("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
