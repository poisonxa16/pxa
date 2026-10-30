#!/usr/bin/env python3
"""Store plugin delivery, engine side: one regression test per finding of the 10-21 review (REVIEW-eng F1-F7, F9, F10, F13).
Each test fails on 45aaa49627 and passes after the fix.

    python3 tests/test-pxa-store-plugin-review.py

Loopback only: stub plugins built and signed with the throwaway keys of tests/test-pxa-plugin-host.py. No GPU, no internet.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import unittest.mock
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HT = _load("pxa_host_test_helpers", os.path.join(ROOT, "tests", "test-pxa-plugin-host.py"))   # sets the config dir, keys, stubs
CFG, TMP = HT.CFG, HT.TMP
from pxa_control import store as S  # noqa: E402
from pxa_control import plugins as P  # noqa: E402
from pxa_control.plugins import gate as G, host as H, pxp  # noqa: E402

RS = _load("pxa_release_scan_review_t", os.path.join(ROOT, "scripts", "release-scan.py"))


def build(pid, ver="0.1.0", entitlement=None, edition="free", out=None):
    """a signed stub .pxp (not installed) -> path"""
    src = os.path.join(TMP, "rsrc", "%s-%s" % (pid, ver))
    shutil.rmtree(src, ignore_errors=True)
    for sub in ("bin", "ui", "lib"):
        os.makedirs(os.path.join(src, sub))
    with open(os.path.join(src, "bin", "stub"), "w") as f:
        f.write(HT.STUB)
    with open(os.path.join(src, "ui", "index.html"), "w") as f:
        f.write("<p>stub</p>\n" * 64)
    shutil.copy(os.path.join(ROOT, "tools", "pxa_control", "plugins", "child.py"), os.path.join(src, "lib", "pxa_plugin_child.py"))
    spec = {"schema": 1, "plugin_id": pid, "name": pid.title(), "version": ver, "publisher": "pxa", "edition": edition,
            "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/stub", "serve"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/%s/" % pid], "ui": {"tab": {"title": pid.title(), "page": "ui/index.html"}},
            "permissions": [], "entitlement": entitlement, "pack": ["bin", "ui", "lib"]}
    with open(os.path.join(src, "plugin.src.json"), "w") as f:
        json.dump(spec, f)
    path, _body = HT._build_mod().build(src, HT.PLUG_SEED, HT.PLUG_KID, out or os.path.join(TMP, "rout", "%s-%s" % (pid, ver)))
    return path


class Install(unittest.TestCase):
    def test_f1_failed_same_version_reinstall_keeps_the_old_install(self):
        path = build("revone")
        pxp.install(path, CFG, HT.PLUG_KEYS)
        before = pxp.folder(CFG, "revone")
        real = os.replace

        def flaky(src, dst):
            if os.path.basename(str(src)).startswith(".stage-"):
                raise PermissionError(13, "in use (test)")
            return real(src, dst)

        with unittest.mock.patch.object(pxp.os, "replace", side_effect=flaky):
            with self.assertRaises(P.PluginError) as cm:
                pxp.install(path, CFG, HT.PLUG_KEYS)
        self.assertEqual((cm.exception.code, cm.exception.status), ("install_failed", 500))
        self.assertEqual(pxp.folder(CFG, "revone"), before)                         # still installed, same folder
        P.load_manifest(before, HT.PLUG_KEYS)                                       # and whole
        base = os.path.dirname(before)
        self.assertEqual(sorted(x for x in os.listdir(base) if x.startswith(".")), [])   # no .old-*, no .stage-*

    def test_f5_corrupt_member_is_a_plugin_error_not_a_500(self):
        path = build("revcrc")
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("ui/index.html")
        with open(path, "r+b") as f:
            f.seek(info.header_offset + 26)
            n, e = int.from_bytes(f.read(2), "little"), int.from_bytes(f.read(2), "little")
            f.seek(info.header_offset + 30 + n + e + info.compress_size // 2)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(P.PluginError) as cm:
            pxp.install(path, CFG, HT.PLUG_KEYS)
        self.assertEqual((cm.exception.code, cm.exception.status), ("bad_package", 400))
        self.assertNotIn("CRC", str(cm.exception))
        self.assertIsNone(pxp.folder(CFG, "revcrc"))


class HostStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = S.EntitlementCache(CFG, HT.ENT_KEYS)
        cls.host = H.PluginHost(CFG, cache=cls.cache, pubkeys=HT.PLUG_KEYS, health_s=3600, backoff=(0, 0, 0, 0, 0), tick_s=3600,
                                hello_timeout_s=10)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()

    def test_f2_deactivate_remove_reinstall_does_not_autostart(self):
        path = build("revauto")
        pxp.install(path, CFG, HT.PLUG_KEYS)
        m, _d = self.host.manifest("revauto")
        self.host.consent.grant(m, m["permissions"])
        self.host.set_disabled("revauto", True)                                     # Deactivate
        self.host.uninstall("revauto")                                              # Remove (consent stays)
        pxp.install(path, CFG, HT.PLUG_KEYS)                                        # Download & install again
        try:
            self.assertNotIn("revauto", self.host.autostart())
            self.assertIn("revauto", self.host.disabled())
            self.host.uninstall("revauto")                                          # Remove without a Deactivate first
            pxp.install(path, CFG, HT.PLUG_KEYS)
            self.assertNotIn("revauto", self.host.autostart())
        finally:
            self.host.stop_all("test")

    def test_f6_unlisted_file_under_the_plugin_folder_refuses_start(self):
        pxp.install(build("revextra"), CFG, HT.PLUG_KEYS)
        d = pxp.folder(CFG, "revextra")
        self.host.manifest("revextra")
        with open(os.path.join(d, "bin", "json.py"), "w") as f:
            f.write("raise SystemExit('shadowed')\n")
        with self.assertRaises(H.HostError) as cm:
            self.host.manifest("revextra")
        self.assertEqual(cm.exception.code, "bad_file")
        self.assertIn("bin/json.py", str(cm.exception))

    def refused(self, pid, why):
        with self.assertRaises((H.HostError, P.PluginError)) as cm:
            self.host.manifest(pid)
        self.assertEqual(cm.exception.code, "bad_file")
        self.assertIn(why, str(cm.exception))
        with self.assertRaises(P.PluginError):
            pxp.load(CFG, pid, HT.PLUG_KEYS)                                         # the in-process loader too

    def test_n1_planted_unchecked_hash_pyc_refuses_start(self):
        import py_compile
        pxp.install(build("revpyc"), CFG, HT.PLUG_KEYS)
        d = pxp.folder(CFG, "revpyc")
        src = os.path.join(TMP, "evil_child.py")
        with open(src, "w") as f:
            f.write("raise SystemExit('planted bytecode')\n")
        tag = sys.implementation.cache_tag
        cfile = os.path.join(d, "lib", "__pycache__", "pxa_plugin_child.%s.pyc" % tag)
        py_compile.compile(src, cfile=cfile, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH, doraise=True)
        self.refused("revpyc", "__pycache__")
        shutil.rmtree(os.path.join(d, "lib", "__pycache__"))
        self.host.manifest("revpyc")
        py_compile.compile(src, cfile=os.path.join(d, "lib", "helper.pyc"), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                           doraise=True)                                             # a sourceless .pyc beside the sources
        self.refused("revpyc", "helper.pyc")

    def test_n1_symlinked_folder_or_file_refuses_start(self):
        pxp.install(build("revlink"), CFG, HT.PLUG_KEYS)
        d = pxp.folder(CFG, "revlink")
        elsewhere = os.path.join(TMP, "elsewhere-lib")
        shutil.copytree(os.path.join(d, "lib"), elsewhere)                          # same bytes, so every sha still matches
        shutil.rmtree(os.path.join(d, "lib"))
        os.symlink(elsewhere, os.path.join(d, "lib"))
        self.refused("revlink", "lib")
        os.unlink(os.path.join(d, "lib"))
        shutil.copytree(elsewhere, os.path.join(d, "lib2"))                          # a link INSIDE the plugin tree
        os.symlink("lib2", os.path.join(d, "lib"))
        self.refused("revlink", "lib")
        os.unlink(os.path.join(d, "lib"))
        shutil.rmtree(os.path.join(d, "lib2"))
        shutil.copytree(elsewhere, os.path.join(d, "lib"))
        self.host.manifest("revlink")
        f = os.path.join(d, "ui", "index.html")
        shutil.copy(f, f + ".real")
        os.replace(f + ".real", os.path.join(TMP, "index.real"))
        os.unlink(f)
        os.symlink(os.path.join(TMP, "index.real"), f)                               # a symlinked file
        self.refused("revlink", "ui/index.html")
        os.unlink(f)
        shutil.copy(os.path.join(TMP, "index.real"), f)
        base = os.path.dirname(d)                                                    # the version folder itself a symlink
        moved = os.path.join(TMP, "moved-version")
        shutil.move(d, moved)
        os.symlink(moved, d)
        self.refused("revlink", "install path")

    def test_n2_os_metadata_files_do_not_block_start_but_code_and_links_still_do(self):
        pxp.install(build("revmeta"), CFG, HT.PLUG_KEYS)
        d = pxp.folder(CFG, "revmeta")
        planted = [".DS_Store", "ui/Thumbs.db", "lib/desktop.ini", "ui/._index.html", "._.DS_Store", "lib/THUMBS.DB"]
        for rel in planted:                                                          # what Finder / Explorer leave behind
            with open(os.path.join(d, *rel.split("/")), "wb") as f:
                f.write(b"\0\x05\x16\x07 metadata")
        self.host.manifest("revmeta")                                                # start is not refused
        self.assertIsNone(pxp.unlisted_any(d, self.host.manifest("revmeta")[0], CFG))
        for rel, why in (("lib/._evil.py", "._evil.py"), ("lib/._evil.so", "._evil.so"), ("lib/._x.pyc", "._x.pyc"),
                         ("lib/notes.txt", "notes.txt")):
            p = os.path.join(d, *rel.split("/"))
            with open(p, "w") as f:
                f.write("x\n")
            self.refused("revmeta", why)                                             # code under a metadata-like name stays refused
            os.unlink(p)
        os.unlink(os.path.join(d, ".DS_Store"))
        os.symlink(os.path.join(d, "plugin.json"), os.path.join(d, ".DS_Store"))     # a metadata name as a symlink stays refused
        self.refused("revmeta", ".DS_Store")
        os.unlink(os.path.join(d, ".DS_Store"))
        self.host.manifest("revmeta")

    def test_f6_python_entries_spawn_isolated(self):
        with open(os.path.join(ROOT, "tools", "pxa_control", "plugins", "host.py")) as f:
            src = f.read()
        self.assertIn('[self.python, "-I", "-B", prog]', src)                      # no bytecode is ever written there
        self.assertIn('"PYTHONDONTWRITEBYTECODE": "1"', src)


class Keys(unittest.TestCase):
    def test_f7_env_pubkeys_never_replace_the_embedded_ones(self):
        evil = os.path.join(TMP, "evil-pubkeys.json")
        with open(evil, "w") as f:
            json.dump({"GRANT_PUBKEYS": {"evil-g": "ab" * 32}, "PLUGIN_PUBKEYS": {"evil-p": "cd" * 32}}, f)
        g0, p0 = dict(S.GRANT_PUBKEYS), dict(P.PLUGIN_PUBKEYS)
        try:
            with unittest.mock.patch.dict(os.environ, {"PXA_STORE_PUBKEYS": evil}):
                S.load_pubkeys()
                self.assertNotIn("evil-g", S.GRANT_PUBKEYS)
                self.assertNotIn("evil-p", P.PLUGIN_PUBKEYS)
                embedded = os.path.join(os.path.dirname(os.path.abspath(S.__file__)), "pubkeys.json")
                real = os.path.isfile
                with unittest.mock.patch.object(S.os.path, "isfile", side_effect=lambda p: False if p == embedded else real(p)):
                    S.load_pubkeys()                                                # a dev build without an embedded file
                self.assertIn("evil-g", S.GRANT_PUBKEYS)
        finally:
            S.GRANT_PUBKEYS.clear()
            S.GRANT_PUBKEYS.update(g0)
            P.PLUGIN_PUBKEYS.clear()
            P.PLUGIN_PUBKEYS.update(p0)


class ScanCase(unittest.TestCase):
    def test_f9_names_match_in_any_case(self):
        for rel in ("x/Plugin.json", "x/PLUGIN.SRC.JSON", "share/ANIBAL.PXP", "tools/Anibal/core.py", "lib/py/Anibal/__init__.py",
                    "tools/PXA_PLUGINS/a.py"):
            self.assertTrue(RS.plugin_payload_name(rel, rel.rsplit("/", 1)[-1]), rel)
        self.assertFalse(RS.plugin_payload_name("tools/pxa_control/core.py", "core.py"))
        z = os.path.join(TMP, "case.bin")
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("Plugin.JSON", "{}")
        self.assertTrue(RS.zip_has_plugin(z))


class PathRule(unittest.TestCase):
    def base(self):
        d = pxp.folder(CFG, "revpath") or pxp.install(build("revpath"), CFG, HT.PLUG_KEYS)["dir"]
        with open(os.path.join(d, "plugin.json")) as f:
            m = json.load(f)
        m.pop("signature", None)
        return m

    def test_f13_windows_and_mac_unsafe_names_are_invalid(self):
        m = self.base()
        self.assertEqual([x for x in P.validate_manifest(m) if x.startswith("files")], [])
        sha = next(iter(m["files"].values()))
        for bad in ("aux.py", "lib/con.txt", "lib/COM1", "lpt9.log", "nul", "lib/a.", "a//b", "./a", "lib/./a", "a\\b", "C:/x", "a b"):
            with self.subTest(bad):
                mm = dict(m, files=dict(m["files"], **{bad: sha}))
                self.assertTrue([x for x in P.validate_manifest(mm) if x.startswith("files")], bad)
        mm = dict(m, files=dict(m["files"], **{"UI/index.html": sha}))
        self.assertTrue([x for x in P.validate_manifest(mm) if "case" in x])
        for ok in ("lib/console.py", "lib/auxiliary.py", "lib/.keep", "lib/com10.txt"):
            with self.subTest(ok):
                self.assertTrue(P.rel_ok(ok), ok)


class StoreInstall(unittest.TestCase):
    """install_plugin against a crafted SIGNED-catalogue answer (LicdClient.catalog patched); downloads must not happen."""
    def setUp(self):
        self.routes = {}
        S.register(self.routes, lambda: CFG)
        self.app = types.SimpleNamespace(store_cache=S.EntitlementCache(CFG, HT.ENT_KEYS))
        self.p0 = dict(P.PLUGIN_PUBKEYS)
        P.PLUGIN_PUBKEYS[HT.PLUG_KID] = HT.PLUG_KEYS[HT.PLUG_KID]

    def tearDown(self):
        P.PLUGIN_PUBKEYS.clear()
        P.PLUGIN_PUBKEYS.update(self.p0)

    def install(self, item, cat):
        def no_dl(*a, **k):
            raise AssertionError("downloaded")
        with unittest.mock.patch.dict(os.environ, {"PXA_STORE_URL": "http://127.0.0.1:9"}), \
                unittest.mock.patch.object(G, "require", return_value={"state": "active"}), \
                unittest.mock.patch.object(S.LicdClient, "catalog", return_value=cat), \
                unittest.mock.patch.object(S.LicdClient, "manifest", side_effect=no_dl), \
                unittest.mock.patch.object(S.LicdClient, "download", side_effect=no_dl):
            try:
                self.routes[("POST", "/api/plugins/install")](self.app, {"item_id": item}, {})
            except Exception as e:  # noqa: BLE001
                return getattr(e, "status", None), getattr(e, "code", e.__class__.__name__)
        return 200, None

    def bundle(self, **kw):
        b = {"version": "0.0.9", "kid": HT.PLUG_KID, "min_host": None,
             "platforms": {"any": {"file_id": "itemx-0.0.9.pxp", "sha256": "ab" * 32, "size": 1000}}}
        b.update(kw)
        return b

    def test_f3_malformed_bundle_or_unlisted_item_is_refused(self):
        self.assertEqual(self.install("itemx", {"items": []}), (404, "no_package"))
        for bad in (self.bundle(kid="bad kid!"), self.bundle(min_host="new"),
                    self.bundle(platforms={"any": {"file_id": "itemx-0.0.9.pxp", "sha256": "ab" * 32, "size": 1000},
                                           "windows-x64": {"file_id": "../x", "sha256": "zz", "size": -1}}),
                    {"version": "0.0.9"}):
            with self.subTest(bad):
                self.assertEqual(self.install("itemx", {"items": [{"id": "itemx", "kind": "plugin", "bundle": bad}]}), (502, "bad_catalog"))

    def test_f8_a_second_install_of_the_same_item_while_one_runs_is_busy(self):
        import threading
        started, go, out = threading.Event(), threading.Event(), {}
        cat = {"items": [{"id": "itembusy", "kind": "plugin", "bundle": self.bundle(platforms={"any": {
            "file_id": "itembusy-0.0.9.pxp", "sha256": "ab" * 32, "size": 1000}})}]}

        def slow_dl(*a, **k):
            started.set()
            go.wait(10)
            raise RuntimeError("stopped by the test")

        def call(key):
            try:
                self.routes[("POST", "/api/plugins/install")](self.app, {"item_id": "itembusy"}, {})
                out[key] = (200, None)
            except Exception as e:  # noqa: BLE001
                out[key] = (getattr(e, "status", None), getattr(e, "code", e.__class__.__name__))
        with unittest.mock.patch.dict(os.environ, {"PXA_STORE_URL": "http://127.0.0.1:9"}), \
                unittest.mock.patch.object(G, "require", return_value={"state": "active"}), \
                unittest.mock.patch.object(S.LicdClient, "catalog", return_value=cat), \
                unittest.mock.patch.object(S.LicdClient, "download", side_effect=slow_dl):
            t = threading.Thread(target=call, args=("first",))
            t.start()
            self.assertTrue(started.wait(10))
            go_later = threading.Timer(3, go.set)                                    # old code: the second call blocks here too
            go_later.start()
            call("second")
            go.set()
            t.join(10)
            go_later.cancel()
        self.assertEqual(out["second"], (409, "install_busy"))
        self.assertEqual(out["first"][1], "RuntimeError")                            # the first one ran its own course

    def test_f11_reinstall_stops_a_running_plugin_first_and_starts_it_after(self):
        path = build("itemrun", ver="0.2.0")
        pxp.install(path, CFG, HT.PLUG_KEYS)
        ev = []

        class Host(object):
            runs = {"itemrun": types.SimpleNamespace(state="running")}

            def stop(self, pid, reason="page"):
                ev.append(("stop", pid, pxp.current_version(CFG, pid)))

            def start(self, pid, by="page"):
                ev.append(("start", pid, pxp.current_version(CFG, pid)))
        real = pxp.install

        def spy(*a, **k):
            ev.append(("install",))
            return real(*a, **k)
        dl = os.path.join(TMP, "dl-itemrun.pxp")
        shutil.copy(path, dl)
        self.app.plugin_host = Host()
        cat = {"items": [{"id": "itemrun", "kind": "plugin", "bundle": self.bundle(version="0.2.0", platforms={"any": {
            "file_id": "itemrun-0.2.0.pxp", "sha256": "ab" * 32, "size": 1000}})}]}
        try:
            with unittest.mock.patch.dict(os.environ, {"PXA_STORE_URL": "http://127.0.0.1:9"}), \
                    unittest.mock.patch.object(G, "require", return_value={"state": "active"}), \
                    unittest.mock.patch.object(S.LicdClient, "catalog", return_value=cat), \
                    unittest.mock.patch.object(S.LicdClient, "download", return_value={"path": dl}), \
                    unittest.mock.patch.object(pxp, "install", side_effect=spy):
                self.routes[("POST", "/api/plugins/install")](self.app, {"item_id": "itemrun"}, {})
        finally:
            del self.app.plugin_host
        self.assertEqual(ev, [("stop", "itemrun", "0.2.0"), ("install",), ("start", "itemrun", "0.2.0")])

    def test_f10_downgrade_check_keys_on_the_plugin_id(self):
        pxp.install(build("revpid", ver="0.1.0", entitlement="itemx", edition="supporter"), CFG, HT.PLUG_KEYS)
        self.assertEqual(self.install("itemx", {"items": [{"id": "itemx", "kind": "plugin", "bundle": self.bundle()}]}), (409, "downgrade"))


class InstallLock(unittest.TestCase):
    def test_f8_remove_waits_for_an_install_in_progress(self):
        import threading
        path = build("revlock")
        pxp.install(path, CFG, HT.PLUG_KEYS)
        inside, go, done = threading.Event(), threading.Event(), threading.Event()
        real = pxp.load_manifest

        def slow(d, *a, **k):
            if os.path.basename(d).startswith(".stage-"):
                inside.set()
                go.wait(10)
            return real(d, *a, **k)
        with unittest.mock.patch.object(pxp, "load_manifest", side_effect=slow):
            t = threading.Thread(target=pxp.install, args=(path, CFG, HT.PLUG_KEYS))
            t.start()
            self.assertTrue(inside.wait(10))
            r = threading.Thread(target=lambda: (pxp.uninstall(CFG, "revlock"), done.set()))
            r.start()
            self.assertFalse(done.wait(1.0))                                         # Remove waits; it never cuts an install in half
            go.set()
            t.join(10)
            r.join(10)
        self.assertTrue(done.is_set())
        self.assertIsNone(pxp.folder(CFG, "revlock"))


class Unlocked(unittest.TestCase):
    def test_f12_the_unlock_map_comes_from_signed_data_only(self):
        d = os.path.join(TMP, "f12cfg")
        os.makedirs(os.path.join(d, "store"), exist_ok=True)
        item = {"id": "f12item", "kind": "feature", "name": "X", "price": 1, "unlocks": "feature:f12x"}
        with open(os.path.join(d, "store", "catalog.json"), "w") as f:              # the page's derived view, unsigned
            json.dump({"items": [item]}, f)
        allow = lambda cfg, i, kind, cache=None: i == "f12item"                    # noqa: E731 - a grant runs for f12item only
        with unittest.mock.patch.object(G, "run_allowed", side_effect=allow):
            self.assertFalse(S.unlocked("feature:f12x", d, pubkeys=HT.ENT_KEYS))     # an edited catalog.json unlocks nothing
            st = {"v": 1, "kid": HT.ENT_KID, "items": [item]}
            env = {"catalog": st, "signature": S.b64u(S.ed25519_sign(HT.ENT_SEED, S.CATALOG_PREFIX + S.canonical_ascii(st)))}
            p = os.path.join(d, "store", S.SIGNED_CATALOG)
            with open(p, "w") as f:
                json.dump(env, f)
            self.assertTrue(S.unlocked("feature:f12x", d, pubkeys=HT.ENT_KEYS))      # the signed catalogue does
            with open(p, "w") as f:
                json.dump(dict(env, catalog=dict(st, items=[dict(item, id="f12item", unlocks="feature:f12y")])), f)
            self.assertFalse(S.unlocked("feature:f12y", d, pubkeys=HT.ENT_KEYS))     # an edit after signing breaks it: not unlocked
            self.assertFalse(S.unlocked("feature:f12x", d, pubkeys=HT.ENT_KEYS))


class Download(unittest.TestCase):
    def test_f4_longer_than_announced_is_fatal_and_not_retried(self):
        d = tempfile.mkdtemp(prefix="dl-", dir=TMP)
        c = S.LicdClient.__new__(S.LicdClient)
        calls = []
        c.call = lambda op, body: calls.append(op) or {"token": "t", "url": "", "name": "x.pxp", "size": 10, "sha256": "00" * 32}

        class R(object):
            status = 200

            def __init__(self):
                self.left = [b"y" * (2 << 20)]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                return self.left.pop() if self.left else b""

        c._req = lambda *a, **k: R()
        with self.assertRaises(Exception) as cm:
            c.download("itemx", "x.pxp", d)
        self.assertEqual(getattr(cm.exception, "code", None), "size_mismatch")
        self.assertEqual(calls, ["download-token"])                                # one pass, no retries
        self.assertFalse(os.path.exists(os.path.join(d, "x.pxp.pxa-part")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
