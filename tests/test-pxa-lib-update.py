#!/usr/bin/env python3
"""tools/pxa_lib_update.py: the licensed PXQN library as its own signed, updatable artifact - check, apply, rollback.

Hermetic: a fake licence server on loopback serves the signed library releases, signed with the test seed (the same
tests/encode_fakes.py signer the package tests use) and trusted by the client through PXA_PACKAGE_PUBKEY. No GPU, no
network, no real encoder. The engine side is a real install tree: `current` -> `pxa-v3`, with the two copies of
libggml-pxqn.so HARD-LINKED to each other exactly as scripts/make-release-tarball.sh produces them.

Wired into CTest as test-pxa-lib-update; run by hand as `python3 tests/test-pxa-lib-update.py`."""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
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
import pxa_lib_update as LU          # noqa: E402
import encode_fakes as F             # noqa: E402

LIB = LU.LIB_NAME
COPIES = ["lib/" + LIB, "lib-compat/" + LIB]
KEY = F.GOOD_KEY


def canon(o):
    """The same canonical JSON pxa-licd signs: sorted keys, no spaces, ASCII."""
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class FakeLibLicence(F._Base):
    """The licence server's library endpoints: POST /v1/lib/latest + GET /v1/lib/download/<ticket>.

    `packs` is lib_id -> {version, channel, platform, cuda_major, min_engine, sm, lib_id, files{name: bytes}}; `latest`
    picks which one a channel answers with. `mode` is ok | revoked | invalid_key | bad_signature | server_error;
    `valued` is whether the key's role carries the beta channel; `corrupt` is a (lib_id, name) whose bytes are served
    one bit off the signed checksum; `modify` mutates the answer envelope (an unsigned field) after signing."""

    def __init__(self, packs, latest=None, mode="ok", valued=True, corrupt=None, modify=None, drop=None, offsite=False):
        F._Base.__init__(self)
        self.packs, self.mode, self.valued, self.corrupt, self.modify = packs, mode, valued, corrupt, modify
        self.drop, self.offsite = drop, offsite
        self.latest = latest or {}
        self.tickets = {}
        self.notice = ""

    # -- signing
    def _seed(self):
        return F.OTHER_SEED if self.mode == "bad_signature" else F.SEED

    def _statement(self, p):
        files = [{"name": n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)} for n, b in sorted(p["files"].items())]
        return canon({"edition": "lib", "lib": LIB, "lib_id": p["lib_id"], "version": p["version"], "channel": p["channel"],
                      "platform": p["platform"], "cuda_major": p["cuda_major"], "min_engine": p.get("min_engine", ""),
                      "sm": list(p.get("sm") or []), "kid": F.KID, "files": files})

    def _answer(self, p, current):
        st = self._statement(p)
        files = []
        for n, b in sorted(p["files"].items()):
            if self.drop == n:
                continue
            tok = "dl9_" + hashlib.sha256(("%s|%s|%d" % (p["lib_id"], n, time.time_ns())).encode()).hexdigest()[:40]
            self.tickets[tok] = (p["lib_id"], n)
            url = ("https://files.example.invalid/x" if self.offsite else self.url + "/v1/lib/download/") + tok
            files.append({"name": n, "lib_id": p["lib_id"], "sha256": hashlib.sha256(b).hexdigest(), "size": len(b), "url": url})
        ans = {"edition": "lib", "lib": LIB, "lib_id": p["lib_id"], "version": p["version"], "channel": p["channel"],
               "update_available": p["version"] != current, "current": current, "platform": p["platform"],
               "cuda_major": p["cuda_major"], "min_engine": p.get("min_engine", ""), "sm": list(p.get("sm") or []),
               "files": files, "manifest": st, "signature": F._b64u(F.ed_sign(self._seed(), b"pxqn-lib|" + st.encode())),
               "kid": F.KID, "expires": int(time.time()) + 900, "notice": self.notice}
        return self.modify(ans) if self.modify else ans

    # -- serving
    def handle(self, h, method, body):
        path = urllib.parse.urlparse(h.path).path
        if path.startswith("/v1/lib/download/"):
            t = self.tickets.get(path.rsplit("/", 1)[1])
            if not t:
                return self.reply(h, 404, {"error": "unknown or expired download link"})
            lib_id, name = t
            data = bytearray(self.packs[lib_id]["files"][name])
            if self.corrupt == (lib_id, name) and data:
                data[0] ^= 0x01              # same length, one bit off: a transfer that was corrupted on the way
            return self.serve_range(h, bytes(data))
        if path != "/v1/lib/latest" or method != "POST":
            return self.reply(h, 404, {"error": "not found"})
        if self.mode == "server_error":
            return self.reply(h, 503, {"error": "maintenance"})
        try:
            b = json.loads(body.decode() or "{}")
        except ValueError:
            return self.reply(h, 400, {"error": "bad json"})
        if b.get("key") != KEY or self.mode == "invalid_key":
            return self.reply(h, 401, {"error": "invalid key"})
        if self.mode == "revoked":
            return self.reply(h, 403, {"error": "key revoked"})
        ch = b.get("channel")
        if ch == "beta" and not self.valued:
            return self.reply(h, 403, {"error": "the beta library channel is for Valued Supporters; your key is not one"})
        lid = self.latest.get(ch)
        p = self.packs.get(lid) if lid else None
        if not p or p["platform"] != b.get("platform") or p["cuda_major"] != b.get("cuda_major"):
            return self.reply(h, 404, {"error": "no library release for %s / %s CUDA %s yet" % (ch, b.get("platform"), b.get("cuda_major"))})
        return self.reply(h, 200, self._answer(p, str(b.get("current") or "")))


def pack(lib_id, version, channel, tag, min_engine="v3.1", platform="linux-x86_64", cuda_major=12, sm=(60, 61, 70)):
    """One library release: the two copies of libggml-pxqn.so the engine ships, with distinct bytes per release."""
    files = {}
    for i, n in enumerate(COPIES):
        files[n] = b"\x7fELF pxqn library " + ("%s|%s|%d|" % (lib_id, n, i)).encode() * 60 + tag
    return {"lib_id": lib_id, "version": version, "channel": channel, "platform": platform, "cuda_major": cuda_major,
            "min_engine": min_engine, "sm": list(sm), "files": files}


REL1 = pack("pxqn-2026.10.1", "v3.1.1", "stable", b"one")
REL2 = pack("pxqn-2026.10.2", "v3.1.2", "stable", b"two")
BETA = pack("pxqn-2026.10.12", "v3.1.12", "beta", b"beta")
OLD = pack("pxqn-2026.10.1", "v3.1.1", "stable", b"one", min_engine="v9")


class LibBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-libupd-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for k, v in (("PXA_PACKAGE_PUBKEY", F.PUB_HEX), ("PXA_CONTROL_CONFIG_DIR", os.path.join(self.tmp, "cfg")),
                     ("HOME", os.path.join(self.tmp, "home"))):
            self._set(k, v)
        for k in ("PXA_LICENCE_KEY", "PXA_INSTALL_DIR", "PXA_LICENCE_URL"):
            self._unset(k)
        os.makedirs(os.path.join(self.tmp, "cfg"))
        os.makedirs(os.path.join(self.tmp, "home"))
        self.ins, self.engine = self._make_install()
        self.lic = None

    # -- environment
    def _set(self, k, v):
        old = os.environ.get(k)
        self.addCleanup(lambda: os.environ.__setitem__(k, old) if old is not None else os.environ.pop(k, None))
        os.environ[k] = v

    def _unset(self, k):
        old = os.environ.pop(k, None)
        self.addCleanup(lambda: os.environ.__setitem__(k, old) if old is not None else None)

    def serve(self, packs, latest=None, **kw):
        self.lic = FakeLibLicence(packs, latest or {"stable": REL1["lib_id"]}, **kw).start()
        self.addCleanup(self.lic.stop)
        return self.lic

    # -- the engine install
    def _make_install(self, base=None):
        ins = os.path.join(base or self.tmp, "install")
        v = os.path.join(ins, "pxa-v3")
        for sub in ("lib", "lib-compat", "bin"):
            os.makedirs(os.path.join(v, sub))
        with open(os.path.join(v, "VERSION"), "w") as f:
            f.write("v3.1\n")
        engine = b"\x7fELF the library the engine release shipped\n" * 40
        with open(os.path.join(v, COPIES[0]), "wb") as f:
            f.write(engine)
        os.link(os.path.join(v, COPIES[0]), os.path.join(v, COPIES[1]))     # the tarball hard-links the second copy
        os.symlink("pxa-v3", os.path.join(ins, "current"))
        return ins, engine

    # -- readers
    def dest(self, name):
        return os.path.join(self.ins, "current", name)

    def content(self, name):
        with open(self.dest(name), "rb") as f:
            return f.read()

    def state(self):
        return LU.load_state(self.ins)

    def assertPristine(self, why=""):
        """The install still runs the bytes the engine shipped: no link into lib/current, nothing installed."""
        self.assertEqual(LU.installed_links(self.ins), ["shipped:" + n for n in COPIES], why)
        for n in COPIES:
            self.assertEqual(self.content(n), self.engine, why)
        self.assertEqual(LU.installed_version(self.ins), "", why)
        self.assertFalse(os.path.exists(os.path.join(self.ins, "lib", "current")), why)

    # -- actions
    def check(self, **kw):
        kw.setdefault("base", self.lic.url)
        kw.setdefault("key", KEY)
        return LU.check(self.ins, **kw)

    def apply(self, **kw):
        kw.setdefault("base", self.lic.url)
        kw.setdefault("key", KEY)
        return LU.apply(self.ins, **kw)


class TestLibraryUpdate(LibBase):
    def test_check_reports_the_release_and_changes_nothing_but_the_stamp(self):
        self.serve({REL1["lib_id"]: REL1})
        rel, raw, cur = self.check()
        self.assertEqual((rel["version"], cur, rel["lib_id"]), ("v3.1.1", "", REL1["lib_id"]))
        self.assertEqual(sorted(f["name"] for f in rel["files"]), sorted(COPIES))
        self.assertEqual({f["name"]: f["sha256"] for f in rel["files"]},
                         {n: hashlib.sha256(REL1["files"][n]).hexdigest() for n in COPIES})
        self.assertPristine("check installs nothing")
        st = self.state()
        self.assertEqual(sorted(st), ["channel", "checked"], "state after a check holds the channel and the stamp, nothing else")
        self.assertEqual(st["channel"], "stable")

    def test_the_key_travels_in_the_body_and_never_in_a_url(self):
        self.serve({REL1["lib_id"]: REL1})
        self.check()
        been = [r for r in self.lic.requests if r["path"] == "/v1/lib/latest"]
        self.assertEqual(len(been), 1)
        self.assertEqual(json.loads(been[0]["body"])["key"], KEY)
        for r in self.lic.requests:
            self.assertNotIn(KEY, r["path"], "the licence key must not appear in a URL")
            self.assertNotIn(KEY, json.dumps(r["headers"]))
        self.assertNotIn(KEY, json.dumps(self.state()), "the licence key is never written to the client state")

    def test_apply_installs_both_copies_and_keeps_the_engines_own(self):
        self.serve({REL1["lib_id"]: REL1})
        applied, rel, cur = self.apply()
        self.assertTrue(applied)
        self.assertEqual((rel["version"], cur), ("v3.1.1", ""))
        self.assertEqual(LU.installed_links(self.ins), COPIES, "both copies are links into lib/current, not just one")
        for n in COPIES:
            self.assertEqual(self.content(n), REL1["files"][n], "the loader now reads the installed library")
            with open(os.path.join(self.ins, "lib", "shipped", n), "rb") as f:
                self.assertEqual(f.read(), self.engine, "the engine's own bytes are kept for a rollback")
        self.assertEqual(os.readlink(os.path.join(self.ins, "lib", "current")), REL1["lib_id"])
        self.assertEqual(os.readlink(os.path.join(self.ins, "lib", "previous")), "shipped")
        st = self.state()
        self.assertEqual((st["version"], st["lib_id"], st["prev"], st.get("prev_version", "")), ("v3.1.1", REL1["lib_id"], "shipped", ""))
        self.assertEqual(st["sha256"][COPIES[0]], hashlib.sha256(REL1["files"][COPIES[0]]).hexdigest())

    def test_a_second_check_says_up_to_date_and_a_second_apply_does_nothing(self):
        self.serve({REL1["lib_id"]: REL1})
        self.apply()
        rel, _, cur = self.check()
        self.assertEqual(cur, "v3.1.1")
        applied, r2, cur2 = self.apply()
        self.assertFalse(applied)
        self.assertIsNone(r2)
        self.assertEqual(cur2, "v3.1.1")
        self.assertEqual(self.content(COPIES[0]), REL1["files"][COPIES[0]], "still the installed library")

    def test_a_newer_release_on_the_channel_updates_and_keeps_the_previous_one(self):
        self.serve({REL1["lib_id"]: REL1, REL2["lib_id"]: REL2}, {"stable": REL1["lib_id"]})
        self.apply()
        self.lic.latest["stable"] = REL2["lib_id"]
        applied, rel, cur = self.apply()
        self.assertTrue(applied)
        self.assertEqual((rel["version"], cur), ("v3.1.2", "v3.1.1"))
        self.assertEqual(self.content(COPIES[1]), REL2["files"][COPIES[1]])
        self.assertEqual(os.readlink(os.path.join(self.ins, "lib", "previous")), REL1["lib_id"])
        self.assertEqual(self.state()["prev_version"], "v3.1.1")

    def test_a_tampered_download_is_refused_and_nothing_is_swapped(self):
        self.serve({REL1["lib_id"]: REL1}, corrupt=(REL1["lib_id"], COPIES[0]))
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "bad_download")
        self.assertPristine("a file that does not match its signed checksum installs nothing")
        self.assertFalse(os.path.exists(os.path.join(self.ins, "lib", REL1["lib_id"])), "no half release is left staged")
        self.assertEqual([d for d in os.listdir(os.path.join(self.ins, "lib")) if d.endswith(".partial")], [],
                         "the partial directory is cleaned up")

    def test_a_bad_signature_is_refused(self):
        self.serve({REL1["lib_id"]: REL1}, mode="bad_signature")
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "bad_signature")
        self.assertPristine()

    def test_a_release_we_have_no_key_for_is_refused(self):
        self.serve({REL1["lib_id"]: REL1})
        self._set("PXA_PACKAGE_PUBKEY", F.ed_pub(F.OTHER_SEED).hex())
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "bad_signature")
        self.assertPristine()

    def test_the_signed_statement_is_the_only_thing_believed(self):
        # the envelope (unsigned) claims another version and another channel; the signed statement says stable v3.1.1
        def meddle(ans):
            ans.update({"version": "v9.9.9", "channel": "beta", "min_engine": "v9", "lib_id": "pxqn-9999"})
            return ans
        self.serve({REL1["lib_id"]: REL1}, modify=meddle)
        applied, rel, _ = self.apply()
        self.assertTrue(applied)
        self.assertEqual((rel["version"], rel["lib_id"], rel["channel"]), ("v3.1.1", REL1["lib_id"], "stable"))
        self.assertEqual(self.state()["version"], "v3.1.1", "the version in the state is the signed one")

    def test_an_incomplete_or_offsite_envelope_is_refused(self):
        cases = (("a manifest file with no download link", {"drop": COPIES[1]}),
                 ("a download link that is not on the licence server", {"offsite": True}),
                 ("no signature at all", {"modify": lambda a: {k: v for k, v in a.items() if k != "signature"}}),
                 ("no manifest at all", {"modify": lambda a: {k: v for k, v in a.items() if k != "manifest"}}))
        for why, kw in cases:
            with self.subTest(why):
                fresh = tempfile.mkdtemp(prefix="pxa-libupd-case-")
                self.addCleanup(shutil.rmtree, fresh, ignore_errors=True)
                keep, keep_engine = self.ins, self.engine
                self.ins, self.engine = self._make_install(fresh)
                self.serve({REL1["lib_id"]: REL1}, **kw)
                try:
                    with self.assertRaises(PK.PackageError) as e:
                        self.apply()
                    self.assertEqual(e.exception.code, "bad_manifest")
                    self.assertPristine()
                finally:
                    self.lic.stop()
                    self.ins, self.engine = keep, keep_engine

    def test_beta_needs_a_valued_key_and_stable_still_works(self):
        self.serve({REL1["lib_id"]: REL1, BETA["lib_id"]: BETA}, {"stable": REL1["lib_id"], "beta": BETA["lib_id"]}, valued=False)
        with self.assertRaises(PK.PackageError) as e:
            self.check(channel="beta")
        self.assertEqual(e.exception.code, "not_valued")
        self.assertIn("Valued Supporters", str(e.exception))
        self.assertPristine()
        rel, _, _ = self.check()                    # the same key, on the channel it does have
        self.assertEqual(rel["channel"], "stable")
        self.lic.valued = True
        rel, _, _ = self.check(channel="beta")
        self.assertEqual((rel["channel"], rel["version"]), ("beta", "v3.1.12"))

    def test_an_unknown_channel_never_reaches_the_server(self):
        self.serve({REL1["lib_id"]: REL1})
        with self.assertRaises(PK.PackageError) as e:
            self.check(channel="nightly")
        self.assertEqual(e.exception.code, "bad_channel")
        self.assertEqual([r for r in self.lic.requests if r["path"] == "/v1/lib/latest"], [])

    def test_a_revoked_key_gets_nothing_and_says_nothing_changed(self):
        self.serve({REL1["lib_id"]: REL1}, mode="revoked")
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "revoked")
        self.assertIn("nothing was changed", str(e.exception))
        self.assertPristine()

    def test_an_engine_that_is_too_old_is_refused_before_anything_is_downloaded(self):
        self.serve({OLD["lib_id"]: OLD})
        with self.assertRaises(PK.PackageError) as e:
            self.apply()
        self.assertEqual(e.exception.code, "engine_too_old")
        self.assertIn("engine", str(e.exception))
        self.assertPristine()
        self.assertEqual([r for r in self.lic.requests if r["path"].startswith("/v1/lib/download/")], [],
                         "the release is not downloaded when the engine cannot run it")

    def test_a_running_server_stops_the_swap(self):
        self.serve({REL1["lib_id"]: REL1})
        exe = os.path.join(self.ins, "current", "bin", "llama-server")
        shutil.copy(shutil.which("sleep") or "/bin/sleep", exe)
        os.chmod(exe, stat.S_IRWXU)
        p = subprocess.Popen([exe, "30"])
        self.addCleanup(p.kill)
        try:
            time.sleep(0.3)
            with self.assertRaises(PK.PackageError) as e:
                self.apply()
            self.assertEqual(e.exception.code, "server_running")
            self.assertPristine("a running server keeps the library it has open")
        finally:
            p.kill()
            p.wait()

    def test_rollback_returns_the_previous_release(self):
        self.serve({REL1["lib_id"]: REL1, REL2["lib_id"]: REL2})
        self.apply()
        self.lic.latest["stable"] = REL2["lib_id"]
        self.apply()
        ver, was = LU.rollback(self.ins)
        self.assertEqual((ver, was), ("v3.1.1", "v3.1.2"))
        for n in COPIES:
            self.assertEqual(self.content(n), REL1["files"][n], "the loader reads the release we rolled back to")
        self.assertEqual(self.state()["lib_id"], REL1["lib_id"])
        ver, was = LU.rollback(self.ins)                    # previous flips: this returns to the release we just left
        self.assertEqual((ver, was), ("v3.1.2", "v3.1.1"))
        self.assertEqual(self.content(COPIES[0]), REL2["files"][COPIES[0]])

    def test_rollback_returns_the_engines_own_library(self):
        self.serve({REL1["lib_id"]: REL1})
        self.apply()
        ver, was = LU.rollback(self.ins)
        self.assertEqual((ver, was), ("", "v3.1.1"))
        self.assertPristine("a rollback to `shipped` puts the engine's own bytes back")
        for n in COPIES:
            self.assertFalse(os.path.islink(self.dest(n)), "the engine's own file is a real file again, not a link")
        self.assertFalse(os.path.exists(os.path.join(self.ins, "lib", "current")))
        with self.assertRaises(PK.PackageError) as e:
            LU.rollback(self.ins)
        self.assertEqual(e.exception.code, "no_previous")

    def test_the_cli_prints_the_update_line(self):
        self.serve({REL1["lib_id"]: REL1})
        env = dict(os.environ)
        out = subprocess.run([sys.executable, os.path.join(TOOLS, "pxa_lib_update.py"), "check",
                              "--dir", self.ins, "--base-url", self.lic.url, "--key", KEY],
                             capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().split(), ["current=shipped", "latest=v3.1.1", "update=yes", "channel=stable"])

    @unittest.skipUnless(os.environ.get("PXA_UPDATE_BIN"), "set PXA_UPDATE_BIN to a built bin/pxa-update to check the C hand-over")
    def test_the_c_tool_hands_over_to_this_module(self):
        """`pxa-update lib ...` (tools/pxa-update.c) finds tools/pxa_lib_update.py inside the install and execs it with the
        same arguments; the three subcommands must reach this module unchanged."""
        self.serve({REL1["lib_id"]: REL1})
        tools = os.path.join(self.ins, "current", "tools")
        os.makedirs(tools, exist_ok=True)
        # the runtime chain the release tarball must carry for `pxa-update lib` to run: the updater, the package verifier, and
        # the adapter the verifier imports. Copying fewer than these proves the hand-over but not that a shipped tree works.
        for f in ("pxa_lib_update.py", "pxa_encode_pkg.py", "pxa_encode_adapter.py"):
            shutil.copy(os.path.join(TOOLS, f), os.path.join(tools, f))
        binary = os.path.join(self.ins, "current", "bin", "pxa-update")
        shutil.copy(os.environ["PXA_UPDATE_BIN"], binary)
        os.chmod(binary, 0o755)

        def run(sub, *extra):
            return subprocess.run([binary, "lib", sub, "--dir", self.ins, "--base-url", self.lic.url, "--key", KEY] + list(extra),
                                  capture_output=True, text=True, timeout=120)

        out = run("check")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().split(), ["current=shipped", "latest=v3.1.1", "update=yes", "channel=stable"])
        self.assertIn("latest", run("check", "--json").stdout)
        out = run("apply")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("installed library v3.1.1", out.stdout)
        self.assertEqual(LU.installed_links(self.ins), COPIES, "the C hand-over installed both copies")
        out = run("rollback")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertPristine("the C hand-over rolled back to the engine's own library")


class TestLibraryLayout(unittest.TestCase):
    """The version rule the CLI and the C tool must agree on, and the version the engine reports."""

    def test_a_date_tag_is_the_old_line_and_sorts_below_v3(self):
        self.assertLess(LU._vtuple("v2026.10.1"), LU._vtuple("v3.1"))
        self.assertGreater(LU._vtuple("v3.1.12"), LU._vtuple("v3.1.9"))
        self.assertEqual(LU._vtuple("3.1.1"), LU._vtuple("v3.1.1"))
        self.assertTrue(LU.min_engine_ok("v3.1.2", "v3.1"))
        self.assertFalse(LU.min_engine_ok("v2026.10.1", "v3.1"))
        self.assertFalse(LU.min_engine_ok("", "v3.1"))
        self.assertTrue(LU.min_engine_ok("", ""), "a release that names no engine runs on anything")

    def test_the_engine_version_is_read_from_the_version_directory(self):
        tmp = tempfile.mkdtemp(prefix="pxa-libupd-test-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        os.makedirs(os.path.join(tmp, "pxa-v3.1.4"))
        os.symlink("pxa-v3.1.4", os.path.join(tmp, "current"))
        self.assertEqual(LU.engine_version(tmp), "pxa-v3.1.4", "with no VERSION file, the directory names the release")
        with open(os.path.join(tmp, "pxa-v3.1.4", "VERSION"), "w") as f:
            f.write("tag: v3.1.4\n")
        self.assertEqual(LU.engine_version(tmp), "v3.1.4")


if __name__ == "__main__":
    unittest.main(verbosity=1)
