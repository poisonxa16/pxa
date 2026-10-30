#!/usr/bin/env python3
"""tools/pxa_engine_update.py + `pxa-update --channel`: the engine beta channel for Valued Supporters.

Hermetic: a fake licence server on loopback serves signed engine releases (the tests/encode_fakes.py test seed, trusted through
PXA_PACKAGE_PUBKEY) and a fake GitHub serves the stable release. The install is a real tree: `current` -> `pxa-v3.1`. The C side
(tools/pxa-update.c) is the real program: PXA_UPDATE_BIN when set, else built here with cc when a compiler exists; without either the
hand-over tests skip and say so. No GPU, no network.

Wired into CTest as test-pxa-engine-update; run by hand as `python3 tests/test-pxa-engine-update.py`."""
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import pxa_encode_pkg as PK          # noqa: E402
import pxa_engine_update as EU       # noqa: E402
import encode_fakes as F             # noqa: E402

KEY = F.GOOD_KEY
_BIN = {}


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def tarball(top, version, extra=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [("VERSION", ("tag: %s\n" % version).encode()), ("bin/llama-server.txt", b"engine " + top.encode())] + list(extra or []):
            ti = tarfile.TarInfo("%s/%s" % (top, name))
            ti.size = len(data)
            ti.mtime = 1760000000
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def release(bid, version, channel, top, notes=None, extra=None, sm="sm60_61_70"):
    tname = "%s-linux-x86_64-cuda12.8-%s.tar.gz" % (top, sm)
    return {"build_id": bid, "version": version, "channel": channel, "platform": "linux-x86_64", "cuda_major": 12, "sm": [60, 61, 70],
            "files": {tname: tarball(top, version, extra),
                      "NOTES.md": notes or ("# PXA %s\nThis is a beta.\n\n## How to report\nDiscord #support, say beta and %s.\n" % (version, version)).encode()}}


BETA1 = release("e4.0-beta.1-aaaaaaaaaa", "4.0-beta.1", "beta", "pxa-v4.0-beta.1")
BETA2 = release("e4.0-beta.2-bbbbbbbbbb", "4.0-beta.2", "beta", "pxa-v4.0-beta.2")
BETA3 = release("e4.0-beta.3-dddddddddd", "4.0-beta.3", "beta", "pxa-v4.0-beta.3", sm="sm60_61_70_75_80_86_89")   # the arch-true name


class FakeEngineLicence(F._Base):
    """POST /v1/engine/latest, POST /v1/engine/channels, GET /v1/engine/download/<ticket>, as pxa-licd answers them."""

    def __init__(self, packs, latest, valued=True, mode="ok", corrupt=None, modify=None, offsite=False, cut=0, busy=0):
        F._Base.__init__(self)
        self.packs, self.latest, self.valued, self.mode = packs, latest, valued, mode
        self.corrupt, self.modify, self.offsite = corrupt, modify, offsite
        self.tickets = {}
        self.cut, self.ranges = cut, []        # cut: tarball streams to break half way (a tunnel drop); ranges: Range headers seen
        self.busy = busy                       # busy: tarball requests answered 503 first (the stream cap, or a tunnel 502/504)

    def _cut(self, h, data):
        """send the headers for the whole file, then half the bytes, then drop the connection."""
        start = int((h.headers.get("Range") or "bytes=0-")[6:].split("-")[0] or 0)
        body = data[start:]
        h.send_response(206 if start else 200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Content-Length", str(len(body)))
        if start:
            h.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
        h.end_headers()
        h.wfile.write(body[:max(1, len(body) // 2)])
        h.wfile.flush()
        h.close_connection = True

    def _statement(self, p):
        files = sorted([{"name": n, "role": "tarball" if n.endswith(".tar.gz") else "notes", "sha256": hashlib.sha256(b).hexdigest(),
                         "size": len(b)} for n, b in p["files"].items()], key=lambda e: e["name"])
        return canon({"edition": "engine", "build_id": p["build_id"], "version": p["version"], "channel": p["channel"],
                      "platform": p["platform"], "cuda_major": p["cuda_major"], "sm": p["sm"], "files": files, "kid": F.KID})

    def handle(self, h, method, body):
        path = urllib.parse.urlparse(h.path).path
        if path.startswith("/v1/engine/download/"):
            t = self.tickets.get(path.rsplit("/", 1)[1])
            if self.mode == "expired" or not t:
                return self.reply(h, 410, {"error": "this download link has expired or been used up"})
            data = bytearray(self.packs[t[0]]["files"][t[1]])
            if self.corrupt == t[1]:
                data[len(data) // 2] ^= 0x01
            if t[1].endswith(".tar.gz"):
                self.ranges.append(h.headers.get("Range") or "")
                if self.busy > 0:
                    self.busy -= 1
                    return self.reply(h, 503, {"error": "the server is busy with other downloads; try again in a minute"})
                if self.cut > 0:
                    self.cut -= 1
                    return self._cut(h, bytes(data))
            return self.serve_range(h, bytes(data))
        try:
            b = json.loads(body.decode() or "{}")
        except ValueError:
            return self.reply(h, 400, {"error": "bad json"})
        if b.get("key") != KEY:
            return self.reply(h, 401, {"error": "invalid key"})
        if path == "/v1/engine/channels":
            return self.reply(h, 200, {"channels": ["stable", "beta"] if self.valued else ["stable"], "beta": self.valued})
        if path != "/v1/engine/latest":
            return self.reply(h, 404, {"error": "not found"})
        ch = b.get("channel")
        if ch == "beta" and not self.valued:
            return self.reply(h, 403, {"error": getattr(self, "deny", "the beta engine channel is for Valued Supporters; your key is not one")})
        p = self.packs.get(self.latest.get(ch) or "")
        if not p:
            return self.reply(h, 404, {"error": "no engine release on that channel yet"})
        st = self._statement(p)
        seed = F.OTHER_SEED if self.mode == "bad_signature" else F.SEED
        files = []
        for n, data in sorted(p["files"].items()):
            tok = "dle_" + hashlib.sha256(("%s|%s|%d" % (p["build_id"], n, time.time_ns())).encode()).hexdigest()[:40]
            self.tickets[tok] = (p["build_id"], n)
            url = ("https://files.example.invalid/" if self.offsite else self.url + "/v1/engine/download/") + tok
            files.append({"name": n, "role": "tarball" if n.endswith(".tar.gz") else "notes", "sha256": hashlib.sha256(data).hexdigest(),
                          "size": len(data), "url": url})
        ans = {"edition": "engine", "build_id": p["build_id"], "version": p["version"], "channel": p["channel"], "files": files,
               "manifest": st, "signature": F._b64u(F.ed_sign(seed, b"pxa-engine|" + st.encode())), "kid": F.KID,
               "expires": int(time.time()) + 900, "update_available": True, "notice": "beta" if ch == "beta" else ""}
        return self.reply(h, 200, self.modify(ans) if self.modify else ans)


class FakeGitHub(F._Base):
    """The stable path: /repos/poisonxa16/pxa/releases/latest + the tarball and its .sha256 sidecar."""

    def __init__(self, tag, top, sms=("sm60_61_70",)):
        F._Base.__init__(self)
        self.tag = tag
        self.tgz = tarball(top, tag)
        self.names = ["%s-linux-x86_64-cuda12.8-%s.tar.gz" % (top, sm) for sm in sms] \
            + ["%s-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz" % top]
        self.got = []                          # the asset paths actually downloaded

    def handle(self, h, method, body):
        path = urllib.parse.urlparse(h.path).path
        if path == "/repos/poisonxa16/pxa/releases/latest":
            assets = []
            for n in self.names:
                assets += [{"browser_download_url": "%s/%s" % (self.url, n)}, {"browser_download_url": "%s/%s.sha256" % (self.url, n)}]
            return self.reply(h, 200, {"tag_name": self.tag, "assets": assets})
        name = path.lstrip("/")
        self.got.append(name)
        if name in self.names:
            return self.reply(h, 200, self.tgz, ctype="application/gzip")
        if name.endswith(".sha256") and name[:-7] in self.names:
            return self.reply(h, 200, ("%s  %s\n" % (hashlib.sha256(self.tgz).hexdigest(), name[:-7])).encode(), ctype="text/plain")
        return self.reply(h, 404, {"error": "not found"})


def update_bin():
    """The real tools/pxa-update.c: PXA_UPDATE_BIN, else built with cc into a temp dir once; '' when neither is possible."""
    if "bin" in _BIN:
        return _BIN["bin"]
    b = os.environ.get("PXA_UPDATE_BIN") or ""
    if not b:
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc:
            d = tempfile.mkdtemp(prefix="pxa-engupd-bin-")
            out = os.path.join(d, "pxa-update")
            if subprocess.run([cc, "-O2", "-o", out, os.path.join(TOOLS, "pxa-update.c")], capture_output=True).returncode == 0:
                b = out
    _BIN["bin"] = b
    return b


class EngineBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-engupd-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for k, v in (("PXA_PACKAGE_PUBKEY", F.PUB_HEX), ("PXA_CONTROL_CONFIG_DIR", os.path.join(self.tmp, "cfg")),
                     ("HOME", os.path.join(self.tmp, "home"))):
            old = os.environ.get(k)
            self.addCleanup(lambda k=k, old=old: os.environ.__setitem__(k, old) if old is not None else os.environ.pop(k, None))
            os.environ[k] = v
        for k in ("PXA_LICENCE_KEY", "PXA_INSTALL_DIR", "PXA_LICENCE_URL", "PXA_UPDATE_BIN_UNUSED"):
            old = os.environ.pop(k, None)
            self.addCleanup(lambda k=k, old=old: os.environ.__setitem__(k, old) if old is not None else None)
        os.makedirs(os.path.join(self.tmp, "cfg"))
        os.makedirs(os.path.join(self.tmp, "home"))
        self.ins = os.path.join(self.tmp, "install")
        v = os.path.join(self.ins, "pxa-v3.1")
        os.makedirs(os.path.join(v, "bin"))
        with open(os.path.join(v, "VERSION"), "w") as f:
            f.write("tag: v3.1\n")
        os.symlink("pxa-v3.1", os.path.join(self.ins, "current"))
        os.makedirs(os.path.join(self.ins, "models"))
        with open(os.path.join(self.ins, "models", "w.gguf"), "w") as f:
            f.write("keep-me")
        self.lic = None

    def serve(self, latest=None, **kw):
        self.lic = FakeEngineLicence({p["build_id"]: p for p in (BETA1, BETA2)}, latest or {"beta": BETA1["build_id"]}, **kw).start()
        self.addCleanup(self.lic.stop)
        return self.lic

    def need_bin(self):
        b = update_bin()
        if not b:
            self.skipTest("no PXA_UPDATE_BIN and no C compiler: the C hand-over is not exercised here")
        return b

    def need_curl(self):
        # the C tool's stable path downloads with curl (as install.sh does); a build container without curl cannot run it
        if not shutil.which("curl"):
            self.skipTest("no curl on PATH: the C tool's stable (GitHub) path is not exercised here")

    def current(self):
        return os.readlink(os.path.join(self.ins, "current"))

    def assertUntouched(self, why=""):
        self.assertEqual(self.current(), "pxa-v3.1", why)
        self.assertFalse(os.path.exists(os.path.join(self.ins, "engine-channel")), why)
        self.assertFalse(os.path.exists(os.path.join(self.ins, "engine-state.json")), why)
        self.assertEqual([n for n in os.listdir(self.ins) if n.startswith(".pxa-")], [], why)
        with open(os.path.join(self.ins, "models", "w.gguf")) as f:
            self.assertEqual(f.read(), "keep-me")

    def apply(self, **kw):
        kw.setdefault("base", self.lic.url)
        kw.setdefault("key", KEY)
        kw.setdefault("channel", "beta")
        kw.setdefault("installer", self.need_bin())
        return EU.apply(self.ins, **kw)

    def cli(self, *args, env=None):
        b = self.need_bin()
        tools = os.path.join(self.ins, "current", "tools")
        os.makedirs(tools, exist_ok=True)
        for f in ("pxa_engine_update.py", "pxa_lib_update.py", "pxa_encode_pkg.py", "pxa_encode_adapter.py"):
            shutil.copy(os.path.join(TOOLS, f), os.path.join(tools, f))
        e = dict(os.environ, PXA_ENGINE_UPDATE=os.path.join(TOOLS, "pxa_engine_update.py"))
        e.update(env or {})
        return subprocess.run([b] + list(args) + ["--dir", self.ins], capture_output=True, text=True, timeout=120, env=e)


class TestBetaChannel(EngineBase):
    def test_valued_installs_the_signed_beta_and_its_notes(self):
        self.serve()
        self.assertEqual(EU.channels(self.lic.url, KEY), {"channels": ["stable", "beta"], "beta": True})
        rel, _, cur, same = EU.check(self.ins, self.lic.url, KEY, "beta")
        self.assertEqual((rel["version"], cur, same), ("4.0-beta.1", "v3.1", False))
        done, rel, before = self.apply()
        self.assertEqual((done, rel["build_id"], before), (True, BETA1["build_id"], "v3.1"))
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        self.assertEqual(os.readlink(os.path.join(self.ins, "previous")), "pxa-v3.1")
        self.assertEqual(EU.read_channel(self.ins), "beta")
        st = EU.load_state(self.ins)
        self.assertEqual((st["channel"], st["version"], st["top"]), ("beta", "4.0-beta.1", "pxa-v4.0-beta.1"))
        self.assertNotIn(KEY, json.dumps(st))
        text, ver, bid = EU.notes(self.ins)
        self.assertEqual((ver, bid), ("4.0-beta.1", BETA1["build_id"]))
        self.assertIn("## How to report", text)
        self.assertEqual([n for n in os.listdir(self.ins) if n.startswith(".pxa-")], [], "no download left behind")
        # the same beta again is not reinstalled; the next one is
        self.assertFalse(self.apply()[0])
        self.lic.latest["beta"] = BETA2["build_id"]
        self.assertTrue(self.apply()[0])
        self.assertEqual((self.current(), os.readlink(os.path.join(self.ins, "previous"))), ("pxa-v4.0-beta.2", "pxa-v4.0-beta.1"))

    def test_supporter_gets_no_beta_and_nothing_changes(self):
        self.serve(valued=False)
        self.assertEqual(EU.channels(self.lic.url, KEY), {"channels": ["stable"], "beta": False})
        for call in (lambda: EU.set_channel(self.ins, self.lic.url, KEY, "beta"),
                     lambda: EU.check(self.ins, self.lic.url, KEY, "beta"),
                     lambda: EU.apply(self.ins, self.lic.url, KEY, "beta", installer="/bin/false")):
            with self.assertRaises(PK.PackageError) as e:
                call()
            self.assertEqual(e.exception.code, "not_valued")
            self.assertIn("Valued Supporters", str(e.exception))
        self.assertUntouched("a supporter key changes nothing")
        self.assertEqual(EU.channels(self.lic.url, ""), {"channels": ["stable"], "beta": False}, "no key = stable only, no request")

    def test_a_bad_signature_or_a_corrupt_download_installs_nothing(self):
        self.serve(mode="bad_signature")
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "bad_signature")
        self.assertUntouched()
        self.lic.mode = "ok"
        self.lic.corrupt = [n for n in BETA1["files"] if n.endswith(".tar.gz")][0]
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "bad_download")
        self.assertUntouched("a tarball one bit off the signed hash never reaches the installer")

    def test_an_expired_ticket_is_refused_plainly(self):
        self.serve(mode="expired")
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "expired_link")
        self.assertUntouched()

    def test_the_unsigned_envelope_cannot_steer_the_install(self):
        self.serve(offsite=True)
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "bad_manifest", "a download link off the licence server is not followed")
        self.lic.offsite = False
        self.lic.modify = lambda a: dict(a, version="9.9-beta.9", build_id="evil")
        done, rel, _ = self.apply()
        self.assertEqual((done, rel["version"], rel["build_id"]), (True, "4.0-beta.1", BETA1["build_id"]), "the signed values win")
        # a stable statement offered to a beta request is refused even though it verifies
        self.lic.modify = None
        self.lic.latest["beta"] = "x"
        self.lic.packs["x"] = dict(release("x", "3.9", "stable", "pxa-v3.9"), build_id="x")
        with self.assertRaises(PK.PackageError) as e:
            EU.check(self.ins, self.lic.url, KEY, "beta")
        self.assertEqual(e.exception.code, "bad_statement")

    def test_a_dropped_download_resumes_in_the_same_apply_and_across_applies(self):
        # a tunnel drop half way: the same ticket is retried with Range, inside one apply
        os.environ["PXA_ENGINE_DL_BACKOFF"] = "0"
        self.addCleanup(os.environ.pop, "PXA_ENGINE_DL_BACKOFF", None)
        EU.DL_BACKOFF = 0
        lic = self.serve(cut=2)
        changed, rel, _ = self.apply()
        self.assertTrue(changed)
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        self.assertEqual(lic.ranges[0], "")
        self.assertTrue(all(r.startswith("bytes=") for r in lic.ranges[1:]) and len(lic.ranges) == 3, lic.ranges)
        self.assertEqual([n for n in os.listdir(self.ins) if n.startswith(".pxa-update-dl-")], [])

    def test_a_download_that_keeps_dropping_is_kept_for_the_next_apply(self):
        EU.DL_BACKOFF = 0
        lic = self.serve(cut=EU.DL_TRIES)
        with self.assertRaises(EU.PackageError) as cm:
            self.apply()
        self.assertEqual(cm.exception.code, "offline")
        self.assertEqual(self.current(), "pxa-v3.1")
        part = [n for n in os.listdir(self.ins) if n.startswith(".pxa-update-dl-")]
        self.assertEqual(part, [".pxa-update-dl-" + BETA1["build_id"]])
        # the next apply gets a fresh ticket and resumes from the .part instead of starting over
        lic.cut = 0
        changed, rel, _ = self.apply()
        self.assertTrue(changed)
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        self.assertTrue(lic.ranges[-1].startswith("bytes=") and lic.ranges[-1] != "bytes=0-", lic.ranges)
        self.assertEqual([n for n in os.listdir(self.ins) if n.startswith(".pxa-update-dl-")], [])

    def test_a_busy_server_is_retried_like_a_drop(self):
        EU.DL_BACKOFF = 0
        lic = self.serve(busy=1, cut=1)
        changed, rel, _ = self.apply()
        self.assertTrue(changed)
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        self.assertEqual(len(lic.ranges), 3, lic.ranges)   # 503, a drop half way, then the resume
        self.assertTrue(lic.ranges[-1].startswith("bytes="), lic.ranges)

    def test_the_first_beta_from_a_tarball_already_on_disk(self):
        # an install whose updater predates the channel: the tester downloaded the tarball once; the tool from inside it installs
        # that file after checking it against the SIGNED sha256, and downloads only the notes
        lic = self.serve()
        tname = [n for n in BETA1["files"] if n.endswith(".tar.gz")][0]
        own = os.path.join(self.tmp, tname)
        with open(own, "wb") as f:
            f.write(bytearray(BETA1["files"][tname]))
        bad = os.path.join(self.tmp, "bad.tar.gz")
        with open(bad, "wb") as f:
            f.write(BETA2["files"][[n for n in BETA2["files"] if n.endswith(".tar.gz")][0]])
        with self.assertRaises(EU.PackageError) as cm:
            self.apply(from_file=bad)
        self.assertEqual(cm.exception.code, "bad_download")
        self.assertUntouched("a tarball that is not the signed one installs nothing")
        changed, rel, _ = self.apply(from_file=own)
        self.assertTrue(changed)
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        self.assertEqual(lic.ranges, [])                    # the tarball was not downloaded again
        self.assertTrue(os.path.isfile(own) and os.path.isfile(bad))
        self.assertEqual(EU.load_state(self.ins)["channel"], "beta")

    def test_notes_on_disk_that_do_not_match_the_signature_are_not_shown(self):
        self.serve()
        self.apply()
        p = os.path.join(self.ins, "engine-notes", BETA1["build_id"] + ".md")
        with open(p, "a") as f:
            f.write("\nclick here\n")
        with self.assertRaises(PK.PackageError) as e:
            EU.notes(self.ins)
        self.assertEqual(e.exception.code, "bad_notes")


class FakeWalletLicence(FakeEngineLicence):
    """adds licd's PXA Coins wallet door: POST /v1/store/nonce and the install-key signed /v1/wallet/engine/{channels,latest}."""

    def __init__(self, *a, redeemed=True, **kw):
        FakeEngineLicence.__init__(self, *a, **kw)
        self.redeemed, self.nonces, self.wallet_calls = redeemed, {}, []

    def handle(self, h, method, body):
        path = urllib.parse.urlparse(h.path).path
        if path == "/v1/store/nonce":
            b = json.loads(body.decode() or "{}")
            n = hashlib.sha256(os.urandom(16)).hexdigest()[:32]
            self.nonces[n] = (b.get("install_pub"), b.get("op"))
            return self.reply(h, 200, {"nonce": n})
        if not path.startswith("/v1/wallet/engine/"):
            return FakeEngineLicence.handle(self, h, method, body)
        what = path.rsplit("/", 1)[1]
        b = json.loads(body.decode() or "{}")
        args = {k: v for k, v in b.items() if k not in ("install_pub", "nonce", "sig")}
        pub, op = self.nonces.pop(b.get("nonce"), (None, None))
        sig = base64.urlsafe_b64decode(str(b.get("sig") or "") + "==")
        msg = b"pxa-plugin-install|" + canon({"op": "engine-" + what, "nonce": b.get("nonce"), "args": args}).encode()
        if pub != b.get("install_pub") or op != "engine-" + what or not PK.ed25519_verify(bytes.fromhex(pub or "00" * 32), msg, sig):
            return self.reply(h, 401, {"error": "bad install signature"})
        self.wallet_calls.append((what, args))
        if what == "channels":
            return self.reply(h, 200, {"beta": self.redeemed, "release": "4.0-beta.1", "wallet_id": "PXW-K7M3-Q9TD-4HXR"})
        if not self.redeemed:
            return self.reply(h, 403, getattr(self, "wallet_deny", {"error": "redeem beta access for 4.0-beta.1 in Store > Wallet (50 coins)",
                                                                      "code": "beta_not_redeemed"}))
        h.path, valued = "/v1/engine/latest", self.valued       # the same signed answer and tickets as the key path
        self.valued = True
        try:
            return FakeEngineLicence.handle(self, h, method, json.dumps({"key": KEY, "channel": "beta"}).encode())
        finally:
            self.valued = valued


class TestBetaByWallet(EngineBase):
    """a supporter without a Valued key: beta access redeemed with coins in the PXA wallet, Control's install key signs."""

    def wallet(self, redeemed=True, valued=False, key_file=True):
        self.lic = FakeWalletLicence({p["build_id"]: p for p in (BETA1, BETA2)}, {"beta": BETA1["build_id"]}, valued=valued, redeemed=redeemed).start()
        self.addCleanup(self.lic.stop)
        if key_file:
            d = os.path.join(self.tmp, "cfg", "plugins")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "install.key"), "wb") as f:
                f.write(hashlib.sha256(b"test-install-key").digest())
        return self.lic

    def test_supporter_key_plus_redeemed_wallet_installs_the_beta(self):
        self.wallet()
        self.assertEqual(EU.channels(self.lic.url, KEY), {"channels": ["stable", "beta"], "beta": True, "via": "wallet"})
        self.assertEqual(EU.set_channel(self.ins, self.lic.url, KEY, "beta"), "beta")
        os.remove(os.path.join(self.ins, "engine-channel"))
        done, rel, before = self.apply()
        self.assertEqual((done, rel["build_id"], before, self.current()), (True, BETA1["build_id"], "v3.1", "pxa-v4.0-beta.1"))
        self.assertEqual([w for w, _ in self.lic.wallet_calls], ["channels", "channels", "latest"])
        self.assertEqual(self.lic.wallet_calls[-1][1], {"channel": "beta", "platform": "linux-x86_64", "cuda_major": "12", "current": "v3.1"})
        st = EU.load_state(self.ins)
        self.assertEqual((st["channel"], st["version"]), ("beta", "4.0-beta.1"))
        self.assertNotIn(KEY, json.dumps(st))

    def test_no_licence_key_wallet_alone_is_enough(self):
        self.wallet()
        rel, _, cur, same = EU.check(self.ins, self.lic.url, "", "beta")
        self.assertEqual((rel["version"], cur, same), ("4.0-beta.1", "v3.1", False))
        self.assertEqual(EU.channels(self.lic.url, "")["beta"], True)

    def test_not_redeemed_says_so_and_changes_nothing(self):
        self.wallet(redeemed=False)
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "beta_not_redeemed")
        self.assertIn("Store > Wallet", str(e.exception))
        self.assertUntouched("not redeemed")
        self.assertEqual(EU.channels(self.lic.url, KEY), {"channels": ["stable"], "beta": False})

    def test_valued_key_never_touches_the_wallet(self):
        self.wallet(valued=True)
        self.apply()
        self.assertEqual(self.lic.wallet_calls, [])

    def test_no_install_key_is_never_made_and_falls_back_to_the_old_answers(self):
        self.wallet(key_file=False)
        with self.assertRaises(PK.PackageError) as e:
            EU.check(self.ins, self.lic.url, "", "beta")
        self.assertEqual(e.exception.code, "no_key")
        with self.assertRaises(PK.PackageError) as e:
            EU.check(self.ins, self.lic.url, KEY, "beta")
        self.assertEqual(e.exception.code, "not_valued")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "cfg", "plugins", "install.key")))
        d = os.path.join(self.tmp, "cfg", "plugins")
        os.makedirs(d)
        with open(os.path.join(d, "install.key"), "wb") as f:
            f.write(b"short")
        self.assertIsNone(EU.wallet_key())
        self.assertEqual(self.lic.wallet_calls, [])

    def test_signature_matches_the_store_client(self):
        sys.path.insert(0, TOOLS)
        from pxa_control import store as S
        seed = hashlib.sha256(b"x").digest()
        self.assertEqual(EU._ed_sign(seed, b"m"), S.ed25519_sign(seed, b"m"))
        self.assertTrue(PK.ed25519_verify(S.ed25519_public(seed), b"m", EU._ed_sign(seed, b"m")))


class TestBetaFilesList(EngineBase):
    """licd's personal-links answer: names, sizes and sha256 come only from the list signed with the package key under its own prefix."""

    def envelope(self, seed=None, prefix=EU.BETA_FILES_PREFIX, typ="beta-files", urls=None):
        files = [{"name": "pxa-v4.0-beta.2.tar.gz", "role": "tarball", "size": 1000, "sha256": "a" * 64},
                 {"name": "docker.tar", "role": "docker", "size": 2000, "sha256": "b" * 64}]
        st = canon({"type": typ, "build_id": "e4.0-beta.2-cccccccccc", "version": "4.0-beta.2", "files": files, "kid": F.KID})
        out = [dict(f, url=(urls or {}).get(f["name"], "http://127.0.0.1:9/g/" + f["name"]), size=1) for f in files]  # envelope numbers are ignored
        out.append({"name": "extra.bin", "role": "docker", "size": 5, "sha256": "c" * 64, "url": "http://127.0.0.1:9/g/extra.bin"})
        return {"manifest": st, "signature": F._b64u(F.ed_sign(seed or F.SEED, prefix + st.encode())), "kid": F.KID, "files": out, "expires": 123}

    def test_good_list_and_only_signed_numbers(self):
        d = EU.verify_beta_files(self.envelope(urls={"docker.tar": "http://example.org/docker.tar"}))
        self.assertEqual((d["build_id"], d["version"], sorted(d["files"]), d["expires"]), ("e4.0-beta.2-cccccccccc", "4.0-beta.2", ["docker.tar", "pxa-v4.0-beta.2.tar.gz"], 123))
        self.assertEqual((d["files"]["pxa-v4.0-beta.2.tar.gz"]["size"], d["files"]["docker.tar"]["sha256"]), (1000, "b" * 64))
        self.assertEqual(d["files"]["docker.tar"]["url"], "", "a plain-http link off localhost is not followed")

    def test_refusals(self):
        for env, code in ((self.envelope(seed=F.OTHER_SEED), "bad_signature"),
                          (self.envelope(prefix=b"pxa-engine|"), "bad_signature"),     # a statement of another kind cannot pass as a file list
                          (self.envelope(typ="engine"), "bad_statement"),
                          (dict(self.envelope(), signature=None), "bad_manifest")):
            with self.assertRaises(PK.PackageError) as e:
                EU.verify_beta_files(env)
            self.assertEqual(e.exception.code, code)
        o = self.envelope()
        o["manifest"] = o["manifest"].replace('"size":1000', '"size":999')
        with self.assertRaises(PK.PackageError) as e:
            EU.verify_beta_files(o)
        self.assertEqual(e.exception.code, "bad_signature")


class TestArchTrueName(EngineBase):
    def test_a_signed_beta_under_the_arch_true_name_installs(self):
        self.lic = FakeEngineLicence({BETA3["build_id"]: BETA3}, {"beta": BETA3["build_id"]}).start()
        self.addCleanup(self.lic.stop)
        done, rel, before = self.apply()
        self.assertEqual((done, rel["tarball"]["name"], before), (True, "pxa-v4.0-beta.3-linux-x86_64-cuda12.8-sm60_61_70_75_80_86_89.tar.gz", "v3.1"))
        self.assertEqual(self.current(), "pxa-v4.0-beta.3")


class TestCHandOver(EngineBase):
    """tools/pxa-update.c: the sticky channel, the hand-over to the module, apply-file, and the way back to stable."""

    def test_stable_is_unchanged_but_for_the_channel_token(self):
        self.need_curl()
        gh = FakeGitHub("v3.1.2", "pxa-v3.1.2").start()
        self.addCleanup(gh.stop)
        out = self.cli("check", "--base-url", gh.url)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "current=v3.1 latest=v3.1.2 update=yes channel=stable")
        self.assertFalse(os.path.exists(os.path.join(self.ins, "engine-channel")), "a stable user gets no new file")
        # an OLDER stable is never installed over a stable install (no downgrade off the beta path)
        gh2 = FakeGitHub("v3.0", "pxa-v3.0").start()
        self.addCleanup(gh2.stop)
        out = self.cli("apply", "--base-url", gh2.url)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "current=v3.1 latest=v3.0 update=no channel=stable"))
        self.assertUntouched()

    def test_stable_takes_the_arch_true_tarball_over_the_legacy_alias(self):
        """a release carrying both names (same bytes): only the arch-true tarball and its own sidecar are downloaded."""
        self.need_curl()
        legacy, true = "sm60_61_70", "sm60_61_70_75_80_86_89"
        for sms in ((legacy, true), (true, legacy), (true,)):
            gh = FakeGitHub("v3.1.2", "pxa-v3.1.2", sms=sms).start()
            self.addCleanup(gh.stop)
            os.remove(os.path.join(self.ins, "current"))
            os.symlink("pxa-v3.1", os.path.join(self.ins, "current"))
            out = self.cli("apply", "--base-url", gh.url)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(self.current(), "pxa-v3.1.2", sms)
            want = "pxa-v3.1.2-linux-x86_64-cuda12.8-%s.tar.gz" % true
            self.assertEqual(sorted(n for n in gh.got if ".tar.gz" in n), [want, want + ".sha256"], sms)

    def test_beta_round_trip_through_the_c_tool_and_back_to_stable(self):
        self.serve()
        out = self.cli("channels", "--licence-url", self.lic.url, "--key", KEY)
        self.assertEqual(out.stdout.strip(), "channels=stable,beta beta=yes current=stable", out.stderr)
        out = self.cli("check", "--channel", "beta", "--licence-url", self.lic.url, "--key", KEY)
        self.assertEqual(out.stdout.strip(), "current=v3.1 latest=4.0-beta.1 update=yes channel=beta", out.stderr)
        out = self.cli("apply", "--channel", "beta", "--licence-url", self.lic.url, "--key", KEY)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("installed pxa-v4.0-beta.1 channel=beta version=4.0-beta.1", out.stdout)
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        # sticky: a plain check now asks the beta channel
        out = self.cli("check", "--licence-url", self.lic.url, "--key", KEY)
        self.assertEqual(out.stdout.strip(), "current=4.0-beta.1 latest=4.0-beta.1 update=no channel=beta", out.stderr)
        out = self.cli("notes")
        self.assertIn("How to report", out.stdout)
        # back to stable: v3.1.2 is OLDER than 4.0-beta.1, and still installs, because the install is leaving the beta
        self.need_curl()
        gh = FakeGitHub("v3.1.2", "pxa-v3.1.2").start()
        self.addCleanup(gh.stop)
        out = self.cli("channel", "stable")
        self.assertEqual(out.returncode, 0, out.stderr)
        out = self.cli("check", "--base-url", gh.url)
        self.assertEqual(out.stdout.strip(), "current=4.0-beta.1 latest=v3.1.2 update=yes channel=stable", out.stderr)
        out = self.cli("apply", "--base-url", gh.url)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual((self.current(), os.readlink(os.path.join(self.ins, "previous"))), ("pxa-v3.1.2", "pxa-v4.0-beta.1"))
        self.assertEqual(EU.read_channel(self.ins), "stable")
        out = self.cli("check", "--base-url", gh.url)
        self.assertEqual(out.stdout.strip(), "current=v3.1.2 latest=v3.1.2 update=no channel=stable")
        # rollback works across channels
        out = self.cli("rollback")
        self.assertEqual((out.returncode, self.current()), (0, "pxa-v4.0-beta.1"))

    def test_a_supporter_through_the_c_tool_stays_on_stable(self):
        self.serve(valued=False)
        for args in (("channel", "beta"), ("check", "--channel", "beta"), ("apply", "--channel", "beta")):
            out = self.cli(*args, "--licence-url", self.lic.url, "--key", KEY)
            self.assertEqual(out.returncode, 1, (args, out.stdout))
            self.assertIn("Valued Supporters", out.stderr)
            self.assertNotIn(KEY, out.stderr + out.stdout)
        self.assertUntouched()

    def test_apply_file_keeps_the_archive_rules(self):
        b = self.need_bin()
        good = os.path.join(self.tmp, "pxa-v4.0-beta.1-linux-x86_64.tar.gz")
        with open(good, "wb") as f:
            f.write(tarball("pxa-v4.0-beta.1", "v4.0-beta.1"))
        out = subprocess.run([b, "apply-file", good, "--sha256", "0" * 64, "--dir", self.ins], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("checksum mismatch", out.stderr)
        bad = os.path.join(self.tmp, "pxa-v4.0-beta.9-linux-x86_64.tar.gz")
        with open(bad, "wb") as f:
            f.write(tarball("pxa-v4.0-beta.9", "v4.0-beta.9", extra=[("models/x.gguf", b"weights")]))
        out = subprocess.run([b, "apply-file", bad, "--dir", self.ins], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("models", out.stderr)
        self.assertUntouched()
        with open(good, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        out = subprocess.run([b, "apply-file", good, "--sha256", sha, "--dir", self.ins], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.current(), "pxa-v4.0-beta.1")
        self.assertTrue(os.path.isfile(good), "apply-file leaves the caller's file to the caller")


LINKS_HOSTED = "this beta downloads through PXA Control's Store > Wallet"     # licd's key path for a links-hosted beta (beta 2)
NOT_WORDS = ("your key is not one", "/key", "paid", "sell", "buy", "price", "purchase")


class TestBetaWording(EngineBase):
    """A Valued member on v3.1 / beta 1 read 'your key is not one'. The refusals say what is true today: the beta is
    for Valued Supporters (no coins), a Valued Supporter installs it from PXA Control's Beta card, v3.1 / beta 1 ask in #support."""

    def clean(self, text):
        for w in NOT_WORDS:
            self.assertNotIn(w, text.lower(), text)

    def test_not_valued_key_says_valued_supporters_and_the_handover(self):
        self.serve(valued=False)
        for call in (lambda: EU.set_channel(self.ins, self.lic.url, KEY, "beta"), lambda: EU.check(self.ins, self.lic.url, KEY, "beta")):
            with self.assertRaises(PK.PackageError) as e:
                call()
            t = str(e.exception)
            self.assertEqual(e.exception.code, "not_valued")
            self.assertIn("The beta channel is for Valued Supporters", t)
            self.clean(t)
        self.assertUntouched()

    def test_not_valued_key_path_message_names_the_handover(self):
        self.serve(valued=False)
        with self.assertRaises(PK.PackageError) as e:
            EU._call(self.lic.url, "/v1/engine/latest", {"key": KEY, "channel": "beta", "platform": "linux-x86_64", "cuda_major": 12})
        t = str(e.exception)
        self.assertIn(EU.HANDOVER, t)
        self.assertIn("/beta", t)
        self.assertIn("Valued Supporters", t)      # PXA Control's routes map a pxa-update refusal with these words to 403 not_valued
        self.clean(t)

    def test_links_hosted_beta_on_the_key_path_points_at_control_not_at_the_key(self):
        self.serve(valued=False).deny = LINKS_HOSTED      # the fake answers every beta key call with licd's links-hosted refusal
        with self.assertRaises(PK.PackageError) as e:
            EU._call(self.lic.url, "/v1/engine/latest", {"key": KEY, "channel": "beta", "platform": "linux-x86_64", "cuda_major": 12})
        t = str(e.exception)
        self.assertEqual(e.exception.code, "not_valued")
        self.assertIn("does not install through this key", t)
        self.assertIn("/beta", t)
        self.assertIn(EU.HANDOVER, t)
        self.assertIn("Valued Supporters", t)
        self.clean(t)
        with self.assertRaises(PK.PackageError) as e:
            EU.check(self.ins, self.lic.url, KEY, "beta")           # no wallet on this computer
        t = str(e.exception)
        self.assertEqual(e.exception.code, "not_valued")
        self.assertIn("/beta", t)
        self.assertIn("no coins", t)
        self.clean(t)
        self.assertUntouched()

    def test_wallet_refusal_under_the_valued_only_rule_says_link_the_key_not_redeem(self):
        lic = TestBetaByWallet.wallet(self, redeemed=False)
        lic.wallet_deny = {"error": "the beta channel is for Valued Supporters", "code": "beta_not_redeemed"}
        with self.assertRaises(PK.PackageError) as e:
            EU.check(self.ins, self.lic.url, "", "beta")
        t = str(e.exception)
        self.assertEqual(e.exception.code, "beta_not_redeemed")
        self.assertIn("Valued Supporters", t)
        self.assertIn("/beta", t)
        self.assertIn("no coins", t)
        self.assertNotIn("Redeem", t)
        self.clean(t)
        self.assertUntouched()

    def test_no_key_no_wallet_names_the_valued_route(self):
        self.serve(valued=False)
        with self.assertRaises(PK.PackageError) as e:
            EU.check(self.ins, self.lic.url, "", "beta")
        self.assertEqual(e.exception.code, "no_key")
        self.assertIn("Valued Supporters", str(e.exception))
        self.clean(str(e.exception))


if __name__ == "__main__":
    unittest.main(verbosity=1)
