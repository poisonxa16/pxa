"""Store plugins end to end (CONTRACT.md): a REAL licd (tools/pxa-licd of the store repo, test keys, 127.0.0.1)
and PXA Control's own server. No package ships a plugin: a plugin reaches a computer only as redeem (coins) -> grant-bound download
-> verify (signed catalogue: version, kid, sha256, size; manifest signature with the pinned key; every file's sha) -> staged
install -> the user's consent -> the host starts it. A plugin published later through the publish path appears and installs with
no Control change. Every tamper case refuses with its code.

    PXA_LICD_DIR=<licd tree> python3 tests/test-pxa-control-store-plugins.py
    (PXA_E2E_OUT=<file.json>: also write the timings and sizes)
Skips when no licd tree with the plugin bundle block is found."""
import hashlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="pxa-store-plugins-")
LICD_DIR = next((p for p in (os.environ.get("PXA_LICD_DIR") or "",)
                 if p and os.path.isfile(os.path.join(p, "pxp_publish.py"))), "")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
os.environ["PXA_SPEECH_AUTOSTART"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
HAVE = NAMED = False
BY = "ctl-test"
if LICD_DIR:
    sys.path.insert(0, LICD_DIR)
    import licd  # noqa: E402
    import pxa_lic  # noqa: E402
    import pxp_publish  # noqa: E402
    import store as licd_store  # noqa: E402
    HAVE = hasattr(licd_store.StoreLogic, "bundle")
    NAMED = "by" in inspect.signature(pxp_publish.Pub).parameters   # licd since 10-24: a pxa-lic change names who runs it
import pxa_control as C  # noqa: E402
from pxa_control import core as CORE, store as S  # noqa: E402
from pxa_control import plugins as P  # noqa: E402
from pxa_control.plugins import host as H, pxp as X  # noqa: E402
import plugin_ui_env as E  # noqa: E402

NUMBERS = {}


def build(work, pid, name, seed, kid, version="0.1.0", edition="supporter", min_host="3.1", extra_files=None):
    """a signed stub plugin (.pxp) the way build-pxp.py makes one; -> path."""
    src = os.path.join(work, "src-%s-%s-%s" % (pid, version, kid))
    for sub in ("bin", "ui", "lib"):
        os.makedirs(os.path.join(src, sub), exist_ok=True)
    with open(os.path.join(src, "bin", "stub"), "w") as f:
        f.write(E.STUB)
    with open(os.path.join(src, "ui", "index.html"), "w") as f:
        f.write(E.PAGE % {"pid": pid, "name": name})
    for rel, data in (extra_files or {}).items():
        os.makedirs(os.path.dirname(os.path.join(src, rel)), exist_ok=True)
        with open(os.path.join(src, rel), "wb") as f:
            f.write(data)
    shutil.copy(os.path.join(ROOT, "tools", "pxa_control", "plugins", "child.py"), os.path.join(src, "lib", "pxa_plugin_child.py"))
    spec = {"schema": 1, "plugin_id": pid, "name": name, "version": version, "publisher": "pxa", "edition": edition,
            "min_host": min_host, "entry": {"kind": "process", "argv": ["bin/stub", "serve"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/%s/" % pid], "ui": {"tab": {"title": name, "page": "ui/index.html"}},
            "permissions": [], "entitlement": pid if edition == "supporter" else None, "pack": ["bin", "ui", "lib"] + sorted({r.split("/")[0] for r in (extra_files or {})})}
    with open(os.path.join(src, "plugin.src.json"), "w") as f:
        json.dump(spec, f)
    out = os.path.join(work, "out-%s-%s-%s" % (pid, version, kid))
    path, _m = E._build_mod().build(src, seed, kid, out)
    return path


def rezip(src, dst, change=None, add=None, symlink=None):
    """copy a .pxp, replacing member bytes (change: {name: bytes}), adding members (add: {name: bytes}), or a symlink member."""
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as b:
        for i in a.infolist():
            b.writestr(i, (change or {}).get(i.filename, a.read(i.filename)))
        for k, v in (add or {}).items():
            b.writestr(k, v)
        if symlink:
            zi = zipfile.ZipInfo(symlink[0])
            zi.external_attr = (0o120777 << 16)
            b.writestr(zi, symlink[1])
    return dst


@unittest.skipUnless(HAVE, "licd tree with the plugin bundle block not found (PXA_LICD_DIR)")
class StorePlugins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = cls.d = os.path.join(TMP, "licd")
        keys = cls.keys = licd_store.keys_init(os.path.join(d, "secrets"), "ent-e2e", "plug-e2e")
        os.makedirs(os.path.join(d, "files"))
        with open(os.path.join(d, "kofi"), "w") as f:
            f.write("kofi-test")
        env = {"PXA_LIC_DB": os.path.join(d, "db.sqlite"), "PXA_LIC_PORT": "0", "PXA_LIC_STORE_SEED": keys["ent_seeds"]["ent-e2e"],
               "PXA_LIC_STORE_KID": "ent-e2e", "PXA_LIC_KOFI_TOKEN": os.path.join(d, "kofi"), "PXA_LIC_STORE_FILES_DIR": os.path.join(d, "files"),
               "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": "",
               "PXA_LIC_PLUGIN_PUBKEYS": keys["pubkeys_file"]}       # licd verifies a package before it lists it (licd review F6)
        cls.lsvc = licd.Service(licd.Config(env))
        cls.lsrv, _ = licd.serve(licd.Config(env), cls.lsvc)
        threading.Thread(target=cls.lsrv.serve_forever, daemon=True).start()
        cls.lurl = "http://127.0.0.1:%d" % cls.lsrv.server_address[1]
        os.environ["PXA_STORE_URL"] = cls.lurl
        cls.old_keys = (dict(S.GRANT_PUBKEYS), dict(P.PLUGIN_PUBKEYS))
        S.GRANT_PUBKEYS.clear()
        P.PLUGIN_PUBKEYS.clear()
        S.load_pubkeys(keys["pubkeys_file"])                   # what a release embeds as store/pubkeys.json (test keys here)
        assert list(P.PLUGIN_PUBKEYS) == ["plug-e2e"]
        cls.seed = licd_store.read_seed(keys["plugin_seeds"]["plug-e2e"])
        cls.work = os.path.join(TMP, "work")
        cls.app = C.App(C.desktop.load_launcher(), port=7777, models_dirs=[os.path.join(TMP, "models")])
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]
        st, w = cls.ctl(cls, "POST", "/api/store/wallet/create", {})
        assert st == 200, w
        cls.wid = w["wallet_id"]
        cls.lsvc.store.deposit("admin", cls.wid, 1000, "e2e coins", confirm=True)
        cls.ctl(cls, "POST", "/api/store/refresh")

    @classmethod
    def tearDownClass(cls):
        try:
            H.host_of(cls.app).stop_all("test")
        except Exception:  # noqa: BLE001
            pass
        os.environ.pop("PXA_STORE_URL", None)
        for s in (cls.srv, cls.lsrv):
            s.shutdown()
            s.server_close()
        S.GRANT_PUBKEYS.clear()
        S.GRANT_PUBKEYS.update(cls.old_keys[0])
        P.PLUGIN_PUBKEYS.clear()
        P.PLUGIN_PUBKEYS.update(cls.old_keys[1])
        out = os.environ.get("PXA_E2E_OUT")
        if out and NUMBERS:
            with open(out, "w") as f:
                json.dump(NUMBERS, f, indent=1, sort_keys=True)
        print("\nE2E NUMBERS " + json.dumps(NUMBERS, sort_keys=True))

    # ---- helpers -------------------------------------------------------------------------------------------------
    def ctl(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data, method=method,
                                   headers={"X-PXA-Token": self.app.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                b = resp.read()
                try:
                    return resp.status, json.loads(b)
                except ValueError:
                    return resp.status, b
        except urllib.error.HTTPError as e:
            b = e.read() or b"{}"
            try:
                return e.code, json.loads(b)
            except ValueError:
                return e.code, {"raw": b[:200]}

    def lic(self, *argv):
        argv = list(argv)
        if NAMED and not any(a == "--by" or a.startswith("--by=") for a in argv):   # publish runs pass their own --by=
            argv = ["--by=" + BY] + argv
        out, err = io.StringIO(), io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            rc = pxa_lic.run(list(argv), svc=self.lsvc, out=out)
        finally:
            sys.stderr = old
        return rc, out.getvalue() + err.getvalue()

    def lic_ok(self, *argv):
        rc, t = self.lic(*argv)
        self.assertEqual(rc, 0, "%s: %s" % (argv, t))
        return t

    def publish(self, path, item, create=None):
        """THE publish path (pxp_publish.py --go, local target): verify offline with the pinned keys, stage, files add, list."""
        sha = pxp_publish.sha256_file(path)
        p = pxp_publish.Pub(path, item, self.keys["pubkeys_file"], "local", True, sha, files_dir=os.path.join(self.d, "files"),
                            url=self.lurl, lic=lambda argv: self.lic(*argv), out=io.StringIO(), create=create,
                            **({"by": BY} if NAMED else {}))
        out = p.out if hasattr(p, "out") else None
        self.assertEqual(p.run(), 0, out.getvalue() if out else "")
        return sha

    def raw_list(self, path, item, price=1):
        """what a compromised store would serve: licd's own package check is switched off here (it refuses these packages since
        licd review F6), so Control's checks are the ones under test."""
        if not self.lsvc.db.q1("SELECT id FROM store_items WHERE id=?", (item,)):
            self.lic_ok("catalog", "add", item, "--kind", "plugin", "--name", item, "--price", str(price), "--version", "0.1.0",
                        "--status", "active")
        rel = "plugins/" + os.path.basename(path)
        dest = os.path.join(self.d, "files", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(path, dest)
        unverified = licd_store.pxp_meta
        with unittest.mock.patch.object(licd_store, "pxp_meta", lambda p, pubkeys_path=None, item=None: unverified(p)):
            self.lic_ok("files", "add", item, rel)
        self.lic_ok("catalog", "edit", item, "--status", "active")
        return dest

    def redeem(self, item):
        st, j = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": item})
        self.assertEqual(st, 200, j)
        return j

    def catalog(self):
        st, c = self.ctl("GET", "/api/store/catalog")
        self.assertEqual((st, c["source"]), (200, "licd"))
        return {i["id"]: i for i in c["items"]}

    def card(self, pid):
        return {r["plugin_id"]: r for r in self.ctl("GET", "/api/plugins/host/cards")[1]["plugins"]}.get(pid)

    def install(self, item):
        return self.ctl("POST", "/api/plugins/install", {"item_id": item})

    # ---- the happy path, the second plugin, Update / Deactivate / Remove ------------------------------------------
    def test_1_happy_path_redeem_install_consent_activate(self):
        path = build(self.work, "hello", "Hello", self.seed, "plug-e2e")
        NUMBERS["hello_bundle_bytes"] = os.path.getsize(path)
        self.assertNotIn("hello", self.catalog())
        sha = self.publish(path, "hello", create={"price": 5, "tagline": "A small test reward", "description": "Says hello.", "perm_version": 1})
        c = self.catalog()["hello"]                                              # a NEW plugin, no Control change
        self.assertEqual((c["kind"], c["bundle"]["version"], c["bundle"]["kid"]), ("plugin", "0.1.0", "plug-e2e"))
        self.assertEqual(c["bundle"]["platforms"]["any"], {"file_id": "hello-0.1.0.pxp", "sha256": sha, "size": os.path.getsize(path)})
        self.assertIsNone(self.card("hello"))
        st, j = self.install("hello")
        self.assertEqual((st, j.get("code")), (403, "locked"))                    # no grant: no download at all
        t0 = time.time()
        j = self.redeem("hello")                                                  # redeem -> grant -> download -> verify -> install
        NUMBERS["hello_redeem_download_install_s"] = round(time.time() - t0, 3)
        self.assertEqual(j["plugin"]["plugin_id"], "hello", j)
        self.assertEqual(j["plugin"]["version"], "0.1.0")
        self.assertEqual(os.listdir(os.path.join(C.config_dir(), "plugins", "incoming")), [])
        row = self.card("hello")
        self.assertEqual((row["state"], row["consent"]["status"], row["store"]["update"]), ("stopped", "none", False))
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "hello"})
        self.assertEqual((st, j.get("code")), (409, "needs_consent"))              # nothing runs before the user's OK
        st, j = self.ctl("POST", "/api/plugins/host/consent", {"plugin_id": "hello", "permissions": []})
        self.assertEqual(st, 200, j)
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "hello"})
        self.assertEqual((st, j["state"]), (200, "running"), j)
        st, j = self.ctl("POST", "/api/plugins/hello/echo", {"hi": 1})            # its tab's route answers
        self.assertEqual((st, j["path"]), (200, "/api/plugins/hello/echo"), j)

        # Deactivate persists: a new host (a Control restart) does not autostart it; Activate again clears it
        st, j = self.ctl("POST", "/api/plugins/host/stop", {"plugin_id": "hello"})
        self.assertEqual(j["state"], "stopped")
        h2 = H.PluginHost(C.config_dir())
        self.assertEqual((h2.disabled(), h2.autostart()), (["hello"], []))
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "hello"})
        self.assertEqual(j["state"], "running")
        self.assertEqual(H.PluginHost(C.config_dir()).disabled(), [])

        # Update: the publish path lists 0.2.0; the card offers it; Install = Update (consent stays: same permissions)
        p2 = build(self.work, "hello", "Hello", self.seed, "plug-e2e", version="0.2.0")
        self.publish(p2, "hello")
        self.catalog()                                                            # the Store tab's read refreshes the cache
        row = self.card("hello")
        self.assertEqual((row["store"]["latest"], row["store"]["update"]), ("0.2.0", True))
        t0 = time.time()
        st, j = self.install("hello")
        NUMBERS["hello_update_s"] = round(time.time() - t0, 3)
        self.assertEqual((st, j["installed"]["version"]), (200, "0.2.0"), j)
        self.assertEqual(self.card("hello")["store"]["update"], False)

        # downgrade: the Store's newest listed version drops back to 0.1.0 (0.2.0 retired) -> refused, 0.2.0 stays
        self.lic_ok("files", "retire", "hello-0.2.0.pxp")
        st, j = self.install("hello")
        self.assertEqual((st, j.get("code")), (409, "downgrade"), j)
        self.assertEqual(X.current_version(C.config_dir(), "hello"), "0.2.0")

        # Remove: stops it, the files go; the unlock stays, so Download & install works again
        st, j = self.ctl("POST", "/api/plugins/uninstall", {"plugin_id": "hello"})
        self.assertEqual((st, j["removed"]), (200, True), j)
        self.assertIsNone(X.folder(C.config_dir(), "hello"))
        self.assertIsNone(self.card("hello"))
        self.lic_ok("files", "add", "hello", "plugins/hello-0.2.0.pxp")           # relist 0.2.0
        st, j = self.install("hello")
        self.assertEqual((st, j["installed"]["version"]), (200, "0.2.0"), j)

    def test_2_anibal_bundle_when_present(self):
        """the real Anibal plugin: its source folder (PXA_ANIBAL_SRC, kept outside the open tree) built and signed here with
        THIS run's test key, the same build-pxp the release bundle comes from; else skipped."""
        src = os.environ.get("PXA_ANIBAL_SRC")
        if not src or not os.path.isfile(os.path.join(src, "plugin.src.json")):
            self.skipTest("no Anibal plugin source (PXA_ANIBAL_SRC)")
        if not os.path.isfile(os.path.join(src, "lib", "pxa_plugin_child.py")):
            self.fail("PXA_ANIBAL_SRC is the plugin's source folder, not its staged copy: run "
                      "tools/pxa-private/plugins/anibal/stage.sh <new dir> and point PXA_ANIBAL_SRC there")
        t0 = time.time()
        p, _m = E._build_mod().build(src, self.seed, "plug-e2e", os.path.join(self.work, "out-anibal"))
        NUMBERS["anibal_build_s"] = round(time.time() - t0, 3)
        NUMBERS["anibal_bundle_bytes"] = os.path.getsize(p)
        with zipfile.ZipFile(p) as z:
            NUMBERS["anibal_bundle_files"] = len(z.namelist())
            NUMBERS["anibal_unpacked_bytes"] = sum(i.file_size for i in z.infolist())
        if self.lsvc.db.q1("SELECT id FROM store_items WHERE id=?", ("anibal",)):
            self.publish(p, "anibal")
        else:
            self.publish(p, "anibal", create={"price": 100, "tagline": "Your local agent", "description": "A local agent.", "perm_version": 1})
        with unittest.mock.patch.object(CORE, "CONTROL_VERSION", "v3.1"):
            j = self.redeem("anibal")                                             # min_host 4.0: the grant stands, no install
        self.assertEqual(j["plugin"]["error"], "host_too_old", j)
        self.assertNotIn("anibal", [r["plugin_id"] for r in X.installed(C.config_dir())])
        with unittest.mock.patch.object(CORE, "CONTROL_VERSION", "v4.0.0"):
            t0 = time.time()
            st, j = self.install("anibal")                                        # the same grant, on a 4.0 Control
            NUMBERS["anibal_download_install_s"] = round(time.time() - t0, 3)
        self.assertEqual((st, j["installed"]["plugin_id"]), (200, "anibal"), j)
        m, _d = H.host_of(self.app).manifest("anibal")
        st, j = self.ctl("POST", "/api/plugins/host/consent", {"plugin_id": "anibal", "permissions": list(m["permissions"])})
        self.assertEqual(st, 200, j)
        t0 = time.time()
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "anibal"})
        self.assertEqual((st, j.get("state")), (200, "running"), j)
        for _ in range(100):
            st, j = self.ctl("GET", "/api/plugins/anibal/status")
            if st == 200 and isinstance(j, dict) and j.get("anibal") == "up":
                break
            time.sleep(0.2)
        NUMBERS["anibal_start_to_answer_s"] = round(time.time() - t0, 3)
        self.assertEqual((st, j.get("anibal")), (200, "up"), j)                  # the Anibal tab answers
        self.assertTrue(str(j.get("address") or "").startswith("http://127.0.0.1:"), j)   # loopback only, never a wildcard
        for _ in range(50):
            st, h = self.ctl("GET", "/api/plugins/anibal/health")
            if st == 200:
                break
            time.sleep(0.2)
        self.assertEqual((st, h.get("ok")), (200, True), h)
        self.ctl("POST", "/api/plugins/host/stop", {"plugin_id": "anibal"})
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "anibal"})   # a second start re-verifies every file:
        self.assertEqual((st, j.get("state")), (200, "running"), j)                    # Anibal wrote nothing into its install
        self.ctl("POST", "/api/plugins/host/stop", {"plugin_id": "anibal"})

    # ---- tamper cases ---------------------------------------------------------------------------------------------
    def tamper(self, item, path, code, status=None):
        self.raw_list(path, item)
        j = self.redeem(item)
        got = j.get("plugin") or {}
        self.assertEqual(got.get("error"), code, j)                               # the unlock stands; nothing is installed
        self.assertIsNone(X.folder(C.config_dir(), item))
        st, j = self.install(item)
        self.assertEqual(j.get("code"), code, j)
        if status:
            self.assertEqual(st, status)
        self.assertEqual([n for n in os.listdir(os.path.join(C.config_dir(), "plugins", "incoming")) if not n.endswith(".pxa-part")], [])

    def test_3_bad_bundle_sha(self):
        path = build(self.work, "t-sha", "T", self.seed, "plug-e2e")
        dest = self.raw_list(path, "t-sha")
        with open(dest, "r+b") as f:                                              # bytes on the store disk change after listing
            f.seek(40)
            b = f.read(1)
            f.seek(40)
            f.write(bytes([b[0] ^ 1]))
        j = self.redeem("t-sha")
        self.assertEqual(j["plugin"]["error"], "sha_mismatch", j)
        self.assertEqual(self.install("t-sha")[1]["code"], "sha_mismatch")
        self.assertIsNone(X.folder(C.config_dir(), "t-sha"))

    def test_4_bad_manifest_signature(self):
        self.tamper("t-sig", build(self.work, "t-sig", "T", bytes(range(1, 33)), "plug-e2e"), "bad_signature", 400)

    def test_5_wrong_kid(self):
        self.tamper("t-kid", build(self.work, "t-kid", "T", bytes(range(1, 33)), "plug-other"), "unknown_key", 400)

    def test_6_file_sha_mismatch(self):
        good = build(self.work, "t-file", "T", self.seed, "plug-e2e")
        bad = rezip(good, os.path.join(self.work, "t-file-bad.pxp"), change={"ui/index.html": b"<p>swapped</p>"})
        self.tamper("t-file", bad, "bad_file")

    def test_7_archive_slip_and_symlink(self):
        good = build(self.work, "t-slip", "T", self.seed, "plug-e2e")
        self.tamper("t-slip", rezip(good, os.path.join(self.work, "t-slip-bad.pxp"), add={"../evil.sh": b"x"}), "bad_package")
        good = build(self.work, "t-link", "T", self.seed, "plug-e2e")
        self.tamper("t-link", rezip(good, os.path.join(self.work, "t-link-bad.pxp"), symlink=("ui/x", "/etc/passwd")), "bad_package")

    def test_8_oversize_and_bomb(self):
        bomb = build(self.work, "t-bomb", "T", self.seed, "plug-e2e", extra_files={"data/zero.bin": b"\0" * (4 << 20)})
        self.tamper("t-bomb", bomb, "bad_package")                               # 4 MiB of zeros inflates > 200x
        big = build(self.work, "t-big", "T", self.seed, "plug-e2e", extra_files={"data/r.bin": os.urandom(300000)})
        old = X.MAX_BYTES
        X.MAX_BYTES = 200000
        try:
            self.tamper("t-big", big, "bad_package")
        finally:
            X.MAX_BYTES = old

    def test_9_no_grant_revoked_other_wallet_replay_copied_folder(self):
        path = build(self.work, "t-rev", "T", self.seed, "plug-e2e")
        self.publish(path, "t-rev", create={"price": 1})
        self.assertEqual(self.install("t-rev")[1]["code"], "locked")              # no grant
        self.redeem("t-rev")
        self.ctl("POST", "/api/plugins/host/consent", {"plugin_id": "t-rev", "permissions": []})
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "t-rev"})
        self.assertEqual(j.get("state"), "running", j)

        # a copied folder on another computer (other install key, even with this computer's grants file): locked at start
        cfg2 = os.path.join(TMP, "cfg-copy")
        shutil.copytree(os.path.join(C.config_dir(), "plugins"), os.path.join(cfg2, "plugins"),
                        ignore=shutil.ignore_patterns("*.sock", "run", "audit.jsonl", "install.key"))   # the plugin, not the machine key
        os.makedirs(os.path.join(cfg2, "store"), exist_ok=True)
        shutil.copy(os.path.join(C.config_dir(), "store", "grants.json"), os.path.join(cfg2, "store", "grants.json"))
        h2 = H.PluginHost(cfg2)
        with self.assertRaises(P.PluginError) as e:
            h2.start("t-rev")
        self.assertEqual(e.exception.code, "locked")
        self.assertEqual(h2.autostart(), [])

        # replay: a used download token is refused by licd (410 used)
        _seed, pub = S.install_key(C.config_dir())
        c = S.LicdClient(self.lurl, C.config_dir(), None)
        tok = c.call("download-token", {"item_id": "t-rev", "file_id": "t-rev-0.1.0.pxp"})
        u = self.lurl + "/v1/store/download/" + tok["token"]
        with urllib.request.urlopen(u, timeout=20) as r:
            self.assertEqual(r.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(u, timeout=20)
        self.assertEqual(e.exception.code, 410)

        # revocation: the owner revokes, the sync applies it, the running plugin stops and stays off
        ent = [g for g in self.lsvc.store.grants(self.wid) if g["item"] == "t-rev"][0]
        self.lic_ok("revoke", ent["grant"], "--reason", "e2e")
        self.ctl("POST", "/api/store/refresh")
        H.host_of(self.app).recheck("test")
        row = self.card("t-rev")
        self.assertIn(row["state"], ("locked", "stopped"), row)
        st, j = self.ctl("POST", "/api/plugins/host/start", {"plugin_id": "t-rev"})
        self.assertEqual((st, j.get("code")), (403, "locked"), j)
        self.assertEqual(self.install("t-rev")[0], 403)

    def test_unlocked_extra_and_route(self):
        st, j = self.ctl("GET", "/api/store/unlocked?u=feature:nothing-here")
        self.assertEqual((st, j["active"]), (200, False))
        self.assertFalse(S.unlocked("bogus"))
        self.assertFalse(S.unlocked("feature:../x"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
