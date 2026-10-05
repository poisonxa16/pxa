#!/usr/bin/env python3
"""LIVE end-to-end of PXA Control's Encode tab against the real licence server (https://lic.pxanetwork.com) and Hugging Face.

Not part of CTest: it needs the internet, a running PXA Control (python3 tools/pxa-launch.py --gui --port 7813 --no-browser, with its own
PXA_CONTROL_CONFIG_DIR / PXA_ENCODE_HOME / PXA_ENCODER_HOME) and, for the Pro half, a licence key. It drives the same HTTP API the Encode tab
calls and prints one line per step. It never prints a key: the key is read from a FILE in code and handed to Control, which stores it 0600.

  encode-e2e-live.py --edition free [--tier pxq4]                         Download Free, one classic encode
  encode-e2e-live.py --edition pro --key-file /path/to/keyfile [--tier pxqn4] [--card N]
                                                                         "I have a key" -> the user's stamped Pro package from the licence server ->
                                                                         signature checked -> installed -> Pro detected -> one real PXQN encode
  common: --control http://127.0.0.1:7813 --source Qwen/Qwen3-1.7B --work DIR --out DIR --summary FILE.json

The Pro encode runs on ONE card (--card): take that card's burst lock (and announce the window) BEFORE running this, and release it after.
Exit 0 only when the job ended `done` and the file's sha256 recomputed here equals the one the encoder reported."""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import pxa_encode_pkg as PK  # noqa: E402


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


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def need(ok, msg):
    if not ok:
        say("FAIL " + msg)
        raise SystemExit(1)
    say("ok   " + msg)


def key_status(server, key, base=None):
    """The licence server's own count for this key (the numbers the owner's dashboard shows): POST /v1/key/status. The key is sent, never printed."""
    r = urllib.request.Request("%s/v1/key/status" % (base.rstrip("/") if base else "https://%s" % server), data=json.dumps({"key": key}).encode(),
                               headers={"Content-Type": "application/json", "User-Agent": "pxa-control-e2e"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def verify_installed(folder):
    """Independent re-check of an installed package: MANIFEST.sig is Ed25519 over b'pxqe-manifest|' + MANIFEST.json bytes, with the key built into Control;
    then every file's sha256 against the manifest. -> (signature_ok, files_checked, package.json)."""
    with open(os.path.join(folder, "MANIFEST.json"), "rb") as f:
        man = f.read()
    sig_ok = None                                                   # the Free package ships no MANIFEST.sig: its trust is the signed statement of the download answer
    if os.path.isfile(os.path.join(folder, "MANIFEST.sig")):
        with open(os.path.join(folder, "MANIFEST.sig")) as f:
            sig = f.read().strip()
        pub = PK.pubkey_bytes("pkg-2026-10")                        # the key built into Control (PXA_PACKAGE_PUBKEY overrides it for a scratch licence server)
        raw = base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))
        sig_ok = PK.ed25519_verify(pub, b"pxqe-manifest|" + man, raw)
    files = json.loads(man).get("files") or {}
    for rel, want in files.items():
        got = sha256_file(os.path.join(folder, rel))
        if got != want:
            raise SystemExit("FAIL installed file differs from its manifest: " + rel)
    with open(os.path.join(folder, "package.json")) as f:
        return sig_ok, len(files), json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="http://127.0.0.1:7813")
    ap.add_argument("--edition", choices=("free", "pro"), required=True)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--source", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--card", type=int, default=None, help="the ONE card the encode runs on (Pro): hold its burst lock first")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--until", choices=("runtime",), default=None, help="stop after the GPU runtime step (a rehearsal against a scratch licence server: no encode)")
    ap.add_argument("--licence-url", default=os.environ.get("PXA_LICENCE_URL"), help="the licence server's base URL (default https://<the one Control reports>)")
    a = ap.parse_args()
    tier = a.tier or ("pxqn4" if a.edition == "pro" else "pxq4")
    api = Api(a.control)
    summary = {"edition": a.edition, "source": a.source, "tier": tier}

    st = api.call("GET", "/api/encode/state")[1]
    say("Control answers; encoders found: %d" % len(st.get("encoders") or []))

    # ---- step 1: get the encoder from the licence server ---------------------------------------------------------------
    body = {"edition": a.edition}
    if a.edition == "pro":
        need(a.key_file and os.path.isfile(a.key_file), "a key file was given")
        with open(a.key_file) as f:
            key = f.read().strip()
        code, r = api.call("POST", "/api/encode/key", {"key": key})              # the "I have a key" box
        need(code == 200 and r.get("key_set"), "Control accepted the key (stored masked as %s)" % r.get("key_masked"))
    code, r = api.call("POST", "/api/encode/get", body)
    need(code == 200, "%s download started (%s)" % (a.edition.title(), r.get("error") or "ok"))
    t0 = time.time()
    last = None
    while True:
        st = api.call("GET", "/api/encode/state")[1]
        p = st.get("pkg") or {}
        if p.get("message") != last:
            last = p.get("message")
            say("  encoder download: %s %s" % (p.get("phase"), last))
        if p.get("phase") in ("done", "failed"):
            break
        need(time.time() - t0 < 600, "the encoder download finished in time")
        time.sleep(1.0)
    need(p.get("phase") == "done", "the encoder was downloaded, its signature and sha256 checked and it was installed: %s" % p.get("message"))
    sel = next(e for e in st["encoders"] if e["selected"])
    folder = os.path.dirname(sel["path"])
    sig_ok, nfiles, pj = verify_installed(folder)
    if a.edition == "pro":
        need(sig_ok, "independent check: MANIFEST.sig verifies against the key built into Control (kid pkg-2026-10); %d files match the manifest" % nfiles)
    else:
        say("ok   independent check: %d files match the manifest (the Free package is trusted through the signed download statement Control verified: Ed25519, kid pkg-2026-10)" % nfiles)
    need(sel["edition"] == a.edition, "Control detected the %s edition (build %s, cli %s)" % (sel["edition"].title(), sel["build_id"], sel.get("cli")))
    summary.update({"build_id": sel["build_id"], "manifest_signature_ok": sig_ok, "download_statement_verified_by_control": True, "files": nfiles, "installed": folder,
                    "licensee": {k: v for k, v in (pj.get("licensee") or {}).items() if k in ("name", "key_id", "package", "issued")}})
    if a.edition == "pro":
        code, r = api.call("POST", "/api/encode/licence", {})
        lic = (next(e for e in r["encoders"] if e["selected"]).get("licence")) or {}
        say("  licence as the server reports it: state=%s user=%s key_id=%s encodes_left=%s unlimited=%s expires=%s" % (
            lic.get("state"), lic.get("user"), lic.get("key_id"), lic.get("encodes_left"), lic.get("unlimited"), lic.get("expires")))
        need(lic.get("state") == "valid", "the licence server says the key is valid")
        summary["licence"] = {k: lic.get(k) for k in ("state", "user", "key_id", "encodes_left", "unlimited", "expires")}

    # ---- step 1b: the GPU runtime. The Pro encoder library links cuBLAS and cuSOLVER; a computer with only the NVIDIA driver gets them as ONE signed download -------------------
    if a.edition == "pro":
        st = api.call("GET", "/api/encode/state")[1]
        rt = st["runtime"]
        say("  GPU libraries: lib=%s source=%s missing=%s fix=%s" % (rt.get("lib"), rt.get("source") or "-", ",".join(rt.get("missing") or []) or "-", rt.get("fix") or "-"))
        summary["gpu_runtime"] = {"before": {k: rt.get(k) for k in ("lib", "source", "missing", "fix")}}
        if rt.get("need"):
            code, v = api.call("GET", "/api/encode/runtime", timeout=60)
            need(code == 200 and v.get("offer"), "the licence server offers the GPU runtime (%s)" % (v.get("offer_error") or (v.get("offer") or {}).get("size_h")))
            say("  Download the GPU runtime (one time, %s; unpacks to %s; CUDA %s)" % (v["offer"]["size_h"], v["offer"]["unpacked_h"], v["offer"]["cuda"]))
            code, r = api.call("POST", "/api/encode/runtime/get", {})
            need(code == 200, "the GPU runtime download started")
            t1, last = time.time(), None
            while True:
                st = api.call("GET", "/api/encode/state")[1]
                j = st["runtime"]["job"]
                msg = "%s %d%%" % (j.get("phase"), round(100 * (j.get("pct") or 0)))
                if msg != last and int(time.time() - t1) % 15 == 0:
                    say("  GPU runtime: " + msg)
                    last = msg
                if not j.get("running"):
                    break
                if time.time() - t1 > 3600:
                    need(False, "the GPU runtime download finished in time")
                time.sleep(1.0)
            need(j.get("phase") == "done", "the GPU runtime was downloaded, its signature and sha256 checked and it was installed: %s" % j.get("message"))
            summary["gpu_runtime"]["download_seconds"] = round(time.time() - t1)
            summary["gpu_runtime"]["installed"] = st["runtime"].get("installed")
            rt = st["runtime"]
        need(rt.get("ok"), "the Pro encoder loads here: GPU libraries from %s" % (rt.get("source") or ("the GPU runtime pack (loader path: an older wrapper)" if rt.get("installed") else "the system")))
        summary["gpu_runtime"]["after"] = {k: rt.get(k) for k in ("lib", "source", "ok")}
        if a.until == "runtime":
            if a.summary:
                with open(a.summary, "w") as f:
                    json.dump(summary, f, indent=1)
            say("PASS (until the GPU runtime: no encode was started)")
            return 0

    # ---- step 2: the wizard calls ---------------------------------------------------------------------------------------
    code, ins = api.call("POST", "/api/encode/inspect", {"source": a.source}, 90)
    need(code == 200, "inspect %s: %s, %s layers, licence %s" % (a.source, ins.get("arch"), ins.get("layers"), (ins.get("licence") or {}).get("level")))
    cards = [a.card] if a.card is not None else [0]
    req = {"source": a.source, "cards": cards, "tier": tier, "work_dir": a.work, "out_dir": a.out, "licence_ack": True}
    if a.edition == "pro":
        need(a.card is not None, "a card was chosen for the Pro encode (--card)")
        req["encode_card"] = a.card
    code, pl = api.call("POST", "/api/encode/plan", req, 90)
    need(code == 200 and any(r["key"] == tier and r["available"] for r in pl["tiers"]), "the plan offers %s on this encoder" % tier)
    code, ck = api.call("POST", "/api/encode/checks", req, 180)
    for c in ck.get("checks", []):
        say("  check %-12s %-4s %s" % (c["id"], c["status"], c["text"][:150]))
    need(code == 200 and ck.get("can_start"), "the checks pass (%s)" % (ck.get("refusal") or "can start"))

    # ---- step 3: the encode ---------------------------------------------------------------------------------------------
    before = None
    if a.edition == "pro":
        before = key_status(st["licence_server"], key, a.licence_url)
        say("  licence server before the encode: status=%s tier=%s used=%s of allowed=%s remaining=%s" % (before.get("status"), before.get("tier"), before.get("used"),
                                                                                                     before.get("allowed"), before.get("remaining")))
    code, j = api.call("POST", "/api/encode/start", req, 180)
    need(code == 200, "the encode job started: %s" % (j.get("id") or j.get("error")))
    jid = j["id"]
    t0 = time.time()
    seen = {}
    while True:
        j = api.call("GET", "/api/encode/job?id=" + jid)[1]
        for s in j["stages"]:
            if seen.get(s["id"]) != s["status"]:
                seen[s["id"]] = s["status"]
                say("  stage %-10s %s" % (s["id"], s["status"]))
        if j["status"] not in ("running", "queued", "paused"):
            break
        say("  job %s %.1f%% eta %ss  %s" % (j["status"], 100 * j["pct"], j.get("eta_s"), next((s["detail"] for s in j["stages"] if s["status"] == "running"), "")[:70])) \
            if int(time.time() - t0) % 30 == 0 else None
        if time.time() - t0 > a.timeout:
            api.call("POST", "/api/encode/cancel", {"id": jid})
            need(False, "the encode finished within %d s" % a.timeout)
        time.sleep(1.0)
    log = api.call("GET", "/api/encode/log?id=" + jid + "&since=0")[1].get("lines") or []
    for ln in log[-12:]:
        say("  log: " + ln[:200])
    need(j["status"] == "done", "the job is done (%s)" % (j.get("error") or {}).get("message"))
    res = j["result"]
    got = sha256_file(res["path"])
    need(got == res["sha256"], "sha256 recomputed here equals the encoder's: %s" % got)
    say("  output %s (%d bytes), %d tensors of tier %s, types %s" % (res["path"], res["size"], res["tier_tensors"], res["tier"], res["types"]))
    summary.update({"job": jid, "status": j["status"], "licence_jid": j.get("licence_jid"), "out": res["path"], "sha256": got, "size": res["size"],
                    "tier_tensors": res["tier_tensors"], "types": res["types"], "seconds": round(time.time() - t0)})
    if a.edition == "pro":
        after = key_status(st["licence_server"], key, a.licence_url)
        say("  licence server after the encode:  status=%s used=%s of allowed=%s remaining=%s" % (after.get("status"), after.get("used"), after.get("allowed"), after.get("remaining")))
        need(after.get("used") == (before.get("used") or 0) + 1, "the licence server counts exactly ONE more encode used (%s -> %s)" % (before.get("used"), after.get("used")))
        summary["licence_after"] = {k: after.get(k) for k in ("status", "tier", "used", "allowed", "remaining", "period_end")}
        summary["licence_before"] = {k: before.get(k) for k in ("status", "tier", "used", "allowed", "remaining", "period_end")}
        job_lines = [ln for ln in log if ln.startswith("licence: job")]
        summary["licence_log"] = job_lines
        for ln in job_lines:
            say("  " + ln)
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump(summary, f, indent=1)
    say("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
