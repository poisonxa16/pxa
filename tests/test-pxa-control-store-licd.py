#!/usr/bin/env python3
"""PXA Control's store routes against a REAL licd (tools/pxa-licd from the companion tree) on a local
port: fake Ko-fi webhook -> voucher -> redeem -> grant -> supporter-model download -> owner revoke -> refresh.
Keys are generated in the test. Skips when the licd tree is not there (PXA_LICD_DIR, PXA_LICD_REPO).

    PXA_LICD_DIR=<licd-tree> python3 tests/test-pxa-control-store-licd.py
    PXA_LICD_REPO=<store repo checkout> [PXA_LICD_REF=<ref>] python3 tests/test-pxa-control-store-licd.py

BetaCard (PXA Coins wallet, m2/wallet): the Update page's Beta card against the same real licd plus the gated download server
(tools/pxa-static-range next to the licd tree): redeem, personal links, resume, a sha mismatch refused, re-download without paying,
the 3-a-day cap, back to stable. Needs PXA_UPDATE_BIN (the pxa-update binary) for the install step.
"""
import glob
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_licd():
    """A licd tree WITH the wallet / Ko-fi inbox code (wallet.py): a donation with no wallet is 'unmatched' there; a voucher-era
    tree answers 'voucher', so a tree without wallet.py is never used. Order: PXA_LICD_DIR; else, when PXA_LICD_REPO names the
    store repo checkout, the licd of a ref there (PXA_LICD_REF, then licd/deploy-1019b, then proj/licd) unpacked with git archive
    into the test's scratch dir (read-only on that repo). Nothing set: "" and the test skips."""
    d = os.environ.get("PXA_LICD_DIR")
    if d:
        return d
    repo = os.environ.get("PXA_LICD_REPO", "")
    if not repo:
        return ""
    for ref in [r for r in (os.environ.get("PXA_LICD_REF"), "licd/deploy-1019b", "proj/licd") if r]:
        dest = os.path.join(TMP, "licd-src")
        try:
            os.makedirs(dest, exist_ok=True)
            tar = subprocess.run(["git", "-C", repo, "archive", ref, "tools/pxa-licd", "tools/pxa-static-range"], capture_output=True, timeout=120)
            if tar.returncode != 0:
                continue
            subprocess.run(["tar", "-x", "-C", dest], input=tar.stdout, check=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if os.path.exists(os.path.join(dest, "tools", "pxa-licd", "wallet.py")):
            return os.path.join(dest, "tools", "pxa-licd")
    return os.path.join(repo, "tools", "pxa-licd")

TMP = tempfile.mkdtemp(prefix="pxa-store-licd-")
LICD_DIR = _find_licd()
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
sys.path.insert(0, os.path.join(ROOT, "tools"))
HAVE_LICD = os.path.exists(os.path.join(LICD_DIR, "store.py")) and os.path.exists(os.path.join(LICD_DIR, "wallet.py"))
NAMED = False
if HAVE_LICD:
    sys.path.insert(0, LICD_DIR)
    import licd  # noqa: E402
    import pxa_lic  # noqa: E402
    import store as licd_store  # noqa: E402
    NAMED = "--by" in pxa_lic.build_parser().format_usage()   # licd since 10-24: a pxa-lic change names who runs it
import pxa_control as C  # noqa: E402
from pxa_control import store as S  # noqa: E402

KOFI = "kofi-test-token-xyz"


@unittest.skipUnless(HAVE_LICD, "licd tree not found (PXA_LICD_DIR or PXA_LICD_REPO)")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the whole path starts on a new, unlinked computer: its own Control config dir. BetaCard (run first) links this
        # computer to its own licd, and a sync there keeps that account view in the shared dir (v5 sync applies the
        # account answer even when the reply's revocation list names a key this run does not know), so it would leak in.
        cls.cfg_old = os.environ.get("PXA_CONTROL_CONFIG_DIR")
        os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg-e2e")
        d = cls.d = os.path.join(TMP, "licd")
        keys = licd_store.keys_init(os.path.join(d, "secrets"), "ent-e2e", "plug-e2e")
        with open(os.path.join(d, "kofi"), "w") as f:
            f.write(KOFI)
        os.makedirs(os.path.join(d, "files"))
        cls.blob = os.urandom(2_500_000)
        with open(os.path.join(d, "files", "Qwen3-1.7B-PXQN4.gguf"), "wb") as f:
            f.write(cls.blob)
        env = {"PXA_LIC_DB": os.path.join(d, "db.sqlite"), "PXA_LIC_PORT": "0", "PXA_LIC_STORE_SEED": keys["ent_seeds"]["ent-e2e"],
               "PXA_LIC_STORE_KID": "ent-e2e", "PXA_LIC_KOFI_TOKEN": os.path.join(d, "kofi"), "PXA_LIC_STORE_FILES_DIR": os.path.join(d, "files"),
               "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": ""}
        cls.lsvc = licd.Service(licd.Config(env))
        cls.lsrv, _ = licd.serve(licd.Config(env), cls.lsvc)
        threading.Thread(target=cls.lsrv.serve_forever, daemon=True).start()
        cls.lurl = "http://127.0.0.1:%d" % cls.lsrv.server_address[1]
        os.environ["PXA_STORE_URL"] = cls.lurl
        S.GRANT_PUBKEYS.clear()
        S.load_pubkeys(keys["pubkeys_file"])                # what a release embeds as store/pubkeys.json
        assert S.GRANT_PUBKEYS == keys["GRANT_PUBKEYS"]
        cls.models = os.path.join(TMP, "models")
        os.makedirs(cls.models)
        cls.app = C.App(C.desktop.load_launcher(), port=7777, models_dirs=[cls.models])
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PXA_STORE_URL", None)
        for s in (cls.srv, cls.lsrv):
            s.shutdown()
            s.server_close()
        os.environ["PXA_CONTROL_CONFIG_DIR"] = cls.cfg_old

    def ctl(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data, method=method,
                                   headers={"X-PXA-Token": self.app.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def kofi(self, txn, amount="10.00", message=""):
        d = {"verification_token": KOFI, "kofi_transaction_id": txn, "type": "Donation", "amount": amount, "currency": "USD",
             "email": "fan@example.org", "from_name": "Fan", "message": message}
        req = urllib.request.Request(self.lurl + "/v1/hooks/kofi", data=urllib.parse.urlencode({"data": json.dumps(d)}).encode(),
                                     method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def owner(self, *argv):
        out = io.StringIO()
        rc = pxa_lic.run((["--by=ctl-test"] if NAMED else []) + list(argv), svc=self.lsvc, out=out)
        self.assertEqual(rc, 0, argv)
        return out.getvalue()

    def test_the_whole_path(self):
        st, info = self.ctl("GET", "/api/info")
        self.assertEqual(info["store"]["redeem"], "/api/store/redeem")              # live routes announced once a store URL is set
        st, cat = self.ctl("GET", "/api/store/catalog")
        self.assertEqual((st, cat["source"]), (200, "licd"))
        ids = {i["id"] for i in cat["items"]}                       # licd 20cdffca stopped seeding the Overdrive presets item;
        self.assertTrue({"anibal", "pxqn"} <= ids <= {"anibal", "pxqn", "overdrive-presets", "supporter-models"}, ids)   # newer licd seeds supporter-models
        st, e = self.ctl("GET", "/api/store/entitlements")
        self.assertEqual((e["account"], e["credits"], e["items"]), (None, None, {}))
        st, e = self.ctl("POST", "/api/store/redeem", {"item_id": "pxqn"})
        self.assertEqual((st, e["code"]), (403, "not_linked"))                       # licd's refusal reaches the page as is

        # PXA Coins (m2/wallet): a donation names no wallet -> it waits; nothing is matched on a name or guessed
        self.assertEqual(self.kofi("e2e-1")["result"], "unmatched")                  # 1. a donation, no wallet yet
        st, w = self.ctl("POST", "/api/store/wallet/create", {})                       # 2. the supporter makes a wallet in Control
        self.assertEqual(st, 200, w)
        wid = w["wallet_id"]
        self.owner("kofi", "assign", "e2e-1", wid, "--reason", "e2e")                   #    the owner's queue hands the waiting event over
        self.assertEqual(self.kofi("e2e-2", "5.00", "thanks! " + wid)["result"], "credited")   # 3. the wallet ID in the message lands directly
        self.assertEqual(self.kofi("e2e-3", "1.00")["result"], "unmatched")            #    that event only: a message never links the payer
        self.owner("kofi", "assign", "e2e-3", wid, "--reason", "e2e")
        st, e = self.ctl("POST", "/api/store/refresh")
        self.assertEqual(e["credits"], 160)
        cost = int({i["id"]: i for i in cat["items"]}["pxqn"]["price"])

        st, e = self.ctl("POST", "/api/store/redeem", {"item_id": "pxqn"})           # 4. redeem the PXQN tier
        self.assertEqual((st, e["credits"], e["items"]["pxqn"]["status"], e["items"]["pxqn"]["level"]), (200, 160 - cost, "ok", "pxqn4"))
        self.assertTrue(S.EntitlementCache(C.config_dir()).has("pxqn"))
        self.lsvc.store.deposit("admin", wid, -(160 - cost), "e2e: spend the rest")
        self.ctl("POST", "/api/store/refresh")
        st, e = self.ctl("POST", "/api/store/redeem", {"item_id": "anibal"})
        self.assertEqual((st, e["code"]), (402, "not_enough_coins"))

        self.owner("files", "add", "pxqn", "Qwen3-1.7B-PXQN4.gguf", "--model", "Qwen3-1.7B", "--tier", "pxqn4", "--id", "q17-pxqn4")
        st, m = self.ctl("GET", "/api/store/models")                                  # 5. supporter models, in-app
        f = m["items"][0]["files"][0]
        self.assertEqual((m["items"][0]["owned"], f["file_id"], f["size"], f["local"]), (True, "q17-pxqn4", len(self.blob), False))
        part = os.path.join(self.models, "Qwen3-1.7B-PXQN4.gguf.pxa-part")
        with open(part, "wb") as fh:                                                   # an earlier, interrupted download
            fh.write(self.blob[:700_000])
        st, j = self.ctl("POST", "/api/store/download", {"item_id": "pxqn", "file_id": "q17-pxqn4"})
        for _ in range(200):
            d = self.ctl("GET", "/api/store/downloads")[1]["downloads"][0]
            if d["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(d["state"], "done", d)
        with open(os.path.join(self.models, "Qwen3-1.7B-PXQN4.gguf"), "rb") as fh:
            self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), hashlib.sha256(self.blob).hexdigest())
        self.assertFalse(os.path.exists(part))
        self.assertEqual([r["start"] for r in self.lsvc.store.downloads()], [700_000])   # it resumed, it did not start over
        self.assertTrue(self.ctl("GET", "/api/store/models")[1]["items"][0]["files"][0]["local"])

        grant = self.lsvc.store.grants(wid)[0]["grant"]                 # 6. a leak: the owner revokes
        self.assertIn("revoked 1 grant", self.owner("revoke", grant, "--reason", "leaked file"))
        st, e = self.ctl("POST", "/api/store/refresh")
        self.assertNotIn("pxqn", e["items"])
        self.assertFalse(S.EntitlementCache(C.config_dir()).has("pxqn"))
        st, e = self.ctl("POST", "/api/store/download", {"item_id": "pxqn", "file_id": "q17-pxqn4"})
        for _ in range(200):
            d = [x for x in self.ctl("GET", "/api/store/downloads")[1]["downloads"] if x["id"] == e["id"]][0]
            if d["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(d["state"], "failed")
        self.assertIn("does not own", d["error"])

        st, e = self.ctl("POST", "/api/store/unlink")                                  # 7. the supporter frees this computer
        self.assertEqual((st, e["account"]), (200, None))
        self.assertIn("store.machine.unlink", self.owner("audit", "--since", "24h"))

    def test_a_forged_catalogue_falls_back(self):
        S.GRANT_PUBKEYS["ent-e2e"], real = "00" * 32, S.GRANT_PUBKEYS["ent-e2e"]
        try:
            st, cat = self.ctl("GET", "/api/store/catalog")
            self.assertIn(cat["source"], ("cache", "fixture"))
        finally:
            S.GRANT_PUBKEYS["ent-e2e"] = real



def _tarball(top, version):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("VERSION", ("tag: %s\n" % version).encode()), ("bin/llama-server.txt", b"engine " + top.encode())):
            ti = tarfile.TarInfo("%s/%s" % (top, name))
            ti.size, ti.mtime = len(data), 1760000000
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


UPDATE_BIN = os.environ.get("PXA_UPDATE_BIN", "")
STATIC_SERVER = os.path.join(os.path.dirname(LICD_DIR), "pxa-static-range", "pxa-static-range.py")


@unittest.skipUnless(HAVE_LICD and os.path.exists(STATIC_SERVER), "licd / pxa-static-range tree not found (PXA_LICD_DIR or PXA_LICD_REPO)")
@unittest.skipUnless(UPDATE_BIN and os.access(UPDATE_BIN, os.X_OK), "no PXA_UPDATE_BIN: the install step needs the pxa-update binary")
class BetaCard(unittest.TestCase):
    BID, VER, TOP = "e4.0-beta.2-cccccccccc", "4.0-beta.2", "pxa-v4.0-beta.2"

    @classmethod
    def setUpClass(cls):
        d = cls.d = os.path.join(TMP, "beta")
        cls.env_old = {k: os.environ.get(k) for k in ("PXA_STORE_URL", "PXA_LICENCE_URL", "PXA_PACKAGE_PUBKEY", "PXA_UPDATE_BIN", "PXA_ENGINE_UPDATE", "HOME")}
        keys = licd_store.keys_init(os.path.join(d, "secrets"), "ent-b", "plug-b")
        with open(os.path.join(d, "kofi"), "w") as f:
            f.write(KOFI)
        pkg_seed = secrets.token_bytes(32)
        with open(os.path.join(d, "pkg.seed"), "w") as f:
            f.write(pkg_seed.hex())
        link_key = secrets.token_hex(32)
        with open(os.path.join(d, "link.key"), "w") as f:
            f.write(link_key)
        # the release: engine pair + Docker image archive, in the gated folder of the download server; licd holds only the record
        tname = "%s-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" % cls.TOP
        cls.blobs = {tname: _tarball(cls.TOP, cls.VER), "NOTES.md": b"# PXA 4.0-beta.2\nA beta.\n", "pxa-4.0-beta.2-docker-image.tar": os.urandom(2_500_000)}
        cls.dname = "pxa-4.0-beta.2-docker-image.tar"
        role = {tname: "tarball", "NOTES.md": "notes", cls.dname: "docker"}
        gate = "g-" + secrets.token_hex(4)
        os.makedirs(os.path.join(d, "www", gate, cls.BID))
        os.makedirs(os.path.join(d, "www", "pub"))
        for n, b in cls.blobs.items():
            with open(os.path.join(d, "www", gate, cls.BID, n), "wb") as f:
                f.write(b)
        os.makedirs(os.path.join(d, "engines", cls.BID))
        with open(os.path.join(d, "engines", cls.BID, "engine.json"), "w") as f:
            json.dump({"build_id": cls.BID, "edition": "engine", "version": cls.VER, "channel": "beta", "platform": "linux-x86_64", "cuda_major": 12,
                       "sm": [60, 61, 70], "published": "2026-10-20T00:00:00Z", "hosted": "links",
                       "files": [{"name": n, "role": role[n], "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)} for n, b in cls.blobs.items()]}, f)
        with open(os.path.join(d, "gate.txt"), "w") as f:
            f.write("/%s/\n" % gate)
        cls.gate = gate
        cls.sport = _free_port()
        cls.slog = open(os.path.join(d, "static.log"), "w")
        cls.static = subprocess.Popen([sys.executable, STATIC_SERVER, str(cls.sport), "--bind", "127.0.0.1", "--directory", os.path.join(d, "www"),
                                       "--gate-prefix-file", os.path.join(d, "gate.txt"), "--link-key-file", os.path.join(d, "link.key")],
                                      stdout=cls.slog, stderr=subprocess.STDOUT)
        cls.surl = "http://127.0.0.1:%d" % cls.sport
        env = {"PXA_LIC_DB": os.path.join(d, "db.sqlite"), "PXA_LIC_PORT": "0", "PXA_LIC_STORE_SEED": keys["ent_seeds"]["ent-b"],
               "PXA_LIC_STORE_KID": "ent-b", "PXA_LIC_KOFI_TOKEN": os.path.join(d, "kofi"), "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": "",
               "PXA_LIC_ENGINE_DIR": os.path.join(d, "engines"), "PXA_LIC_PKG_KID": "pkg-b", "PXA_LIC_PKG_SEED": os.path.join(d, "pkg.seed"),
               "PXA_LIC_LINK_KEY": os.path.join(d, "link.key"), "PXA_LIC_LINK_BASE": cls.surl, "PXA_LIC_LINK_PREFIX": "/" + gate + "/",
               "PXA_LIC_MAILER": "file:" + os.path.join(d, "mail")}
        os.makedirs(os.path.join(d, "mail"))
        cls.lsvc = licd.Service(licd.Config(env))
        cls.lsrv, _ = licd.serve(licd.Config(env), cls.lsvc)
        threading.Thread(target=cls.lsrv.serve_forever, daemon=True).start()
        cls.lurl = "http://127.0.0.1:%d" % cls.lsrv.server_address[1]
        import ed25519 as licd_ed
        # an install of v3.1 with pxa-update inside it (the binary finds its install from its own path, as in a release)
        cls.ins = os.path.join(d, "install")
        v = os.path.join(cls.ins, "pxa-v3.1")
        os.makedirs(os.path.join(v, "bin"))
        with open(os.path.join(v, "VERSION"), "w") as f:
            f.write("tag: v3.1\n")
        shutil.copy(UPDATE_BIN, os.path.join(v, "bin", "pxa-update"))
        os.symlink("pxa-v3.1", os.path.join(cls.ins, "current"))
        os.makedirs(os.path.join(d, "home"))
        os.environ.update({"PXA_STORE_URL": cls.lurl, "PXA_LICENCE_URL": cls.lurl, "PXA_PACKAGE_PUBKEY": licd_ed.public_key(pkg_seed).hex(),
                           "PXA_UPDATE_BIN": os.path.join(v, "bin", "pxa-update"), "PXA_ENGINE_UPDATE": os.path.join(ROOT, "tools", "pxa_engine_update.py"),
                           "HOME": os.path.join(d, "home")})
        cls.app = C.App(C.desktop.load_launcher(), port=7777, models_dirs=[os.path.join(TMP, "models")])   # Control saves its folders in the shared config dir
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.env_old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for s in (cls.srv, cls.lsrv):
            s.shutdown()
            s.server_close()
        cls.static.terminate()
        cls.static.wait(5)
        cls.slog.close()

    ctl = EndToEnd.ctl

    def docker(self):
        st, j = self.ctl("POST", "/api/update/beta/docker", {})
        self.assertEqual(st, 200, j)
        for _ in range(400):
            dl = self.ctl("GET", "/api/update/beta")[1]["docker"]
            if dl["state"] != "running":
                return dl
            time.sleep(0.05)
        self.fail("the docker download did not finish")

    def issues(self):
        return self.lsvc.db.q1("SELECT COUNT(*) n FROM store_link_issues")["n"]

    def test_email_join_goes_through_control_signed(self):
        """recovery on a new computer: Control asks licd for a mailed code, then joins the wallet that PROVED that email (none here)."""
        st, j = self.ctl("POST", "/api/store/wallet/email-join-start", {"email": "me@example.org"})
        self.assertEqual((st, j.get("sent")), (200, True), j)
        fs = sorted(glob.glob(os.path.join(self.d, "mail", "*.json")))
        with open(fs[-1]) as f:
            code = re.search(r"\b([0-9]{6})\b", json.load(f)["subject"]).group(1)
        st, j = self.ctl("POST", "/api/store/wallet/email-join", {"email": "me@example.org", "code": code})
        self.assertEqual((st, j.get("code")), (404, "no_wallet"), j)                              # signed, reached licd, nobody proved it

    def test_redeem_install_resume_redownload_cap_and_back_to_stable(self):
        from pxa_control import routes as R
        st, b = self.ctl("GET", "/api/update/beta")
        self.assertEqual((st, b["store"], b["wallet"]), (200, True, None))                     # no wallet yet: the card says how to make one
        st, w = self.ctl("POST", "/api/store/wallet/create", {})
        self.assertEqual(st, 200, w)
        wid = w["wallet_id"]
        # licd's default (PXA_LIC_BETA_VALUED_ONLY unset, as deployed since 596159bc): the beta is for Valued Supporters only, so a
        # plain wallet is not told the release, has no coin price, cannot unlock it and gets no links. (Before this the test assumed the old 50-coin path.)
        orig = getattr(self.lsvc.store, "beta_valued_only", None)
        if orig is None:
            self.skipTest("this licd tree predates the Valued-only beta (596159bc)")
        self.assertTrue(orig)
        st, b = self.ctl("GET", "/api/update/beta")
        self.assertEqual((b["wallet"], b["release"], b["cost"], b["coins"], b["short_by"], b["access"]), (wid, None, 0, 0, 0, False))
        st, e = self.ctl("POST", "/api/update/beta/redeem", {})
        self.assertEqual((st, e.get("code") if isinstance(e, dict) else e), (403, "valued_only"))
        st, e = self.ctl("POST", "/api/store/wallet/links", {})
        self.assertEqual((st, e.get("code") if isinstance(e, dict) else e), (403, "no_beta_access"))
        # the coin path stays in licd behind PXA_LIC_BETA_VALUED_ONLY=0 (test_wallet covers it); the rest of this test runs that path end to end
        self.lsvc.store.beta_valued_only = False
        self.addCleanup(setattr, self.lsvc.store, "beta_valued_only", orig)
        self.addCleanup(R._BETA_VIEW.__setitem__, "at", -1e18)                                    # the next card read is fresh, not this test's coin-path card
        R._BETA_VIEW["at"] = -1e18                                                                # the card's minute has passed
        st, b = self.ctl("GET", "/api/update/beta")
        self.assertEqual((b["wallet"], b["release"], b["cost"], b["coins"], b["short_by"], b["access"]), (wid, self.VER, 50, 0, 50, False))
        st, e = self.ctl("POST", "/api/update/beta/redeem", {})
        self.assertEqual((st, e["code"]), (402, "not_enough_coins"))                             # short of coins: nothing spent, nothing granted
        st, e = self.ctl("POST", "/api/store/wallet/links", {})
        self.assertEqual((st, e["code"]), (403, "no_beta_access"))
        self.lsvc.store.deposit("admin", wid, 60, "staging test")
        st, b = self.ctl("POST", "/api/update/beta/redeem", {})                                # 1. one press: coins out, beta in
        self.assertEqual((st, b["redeemed"], b["access"], b["coins"], b["short_by"]), (200, True, True, 10, 0))
        st, j = self.ctl("POST", "/api/update/channel", {"channel": "beta"})                     # 2. the page records the channel...
        self.assertEqual((st, j["channel"]), (200, "beta"), j)
        st, j = self.ctl("POST", "/api/update/apply", {})                                        # ...and pxa-update installs it in its own folder
        self.assertEqual(st, 200, j)
        self.assertEqual(os.readlink(os.path.join(self.ins, "current")), self.TOP)
        self.assertTrue(os.path.isdir(os.path.join(self.ins, "pxa-v3.1")))
        with open(os.path.join(self.ins, "engine-state.json")) as f:
            self.assertEqual(json.load(f)["version"], self.VER)
        self.assertEqual(self.issues(), 1)
        # 3. the Docker image archive: a corrupted partial download resumes, fails the signed sha256 and is deleted
        dest = os.path.join(os.environ["PXA_CONTROL_CONFIG_DIR"], "downloads", self.dname)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".part", "wb") as f:
            f.write(b"\0" * 700_000)
        dl = self.docker()
        self.assertEqual(dl["state"], "failed", dl)
        self.assertIn("does not match the signed checksum", dl["error"])
        self.assertFalse(os.path.exists(dest))
        with open(dest + ".part", "wb") as f:                                                    # 4. a real partial download resumes
            f.write(self.blobs[self.dname][:900_000])
        dl = self.docker()
        self.assertEqual((dl["state"], dl["path"]), ("done", dest), dl)
        with open(dest, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), hashlib.sha256(self.blobs[self.dname]).hexdigest())
        with open(os.path.join(self.d, "static.log")) as f:
            self.assertGreaterEqual(f.read().count('" 206 '), 2)                                  # both presses asked for the rest only
        self.assertEqual(self.issues(), 1)                                                        # one issue of links served all of it
        # 5. the 3-a-day cap, then a re-download from the links already made, without spending coins
        self.assertEqual(self.ctl("POST", "/api/store/wallet/links", {})[0], 200)
        self.assertEqual(self.ctl("POST", "/api/store/wallet/links", {})[0], 200)
        st, e = self.ctl("POST", "/api/store/wallet/links", {})
        self.assertEqual((st, e["code"]), (429, "link_limit"))
        os.remove(dest)
        self.assertEqual(self.docker()["state"], "done")
        st, b = self.ctl("GET", "/api/update/beta")
        self.assertEqual((b["access"], b["coins"]), (True, 10))
        self.assertEqual(self.ctl("POST", "/api/update/beta/redeem", {})[1]["redeemed"], False)  # redeeming again spends nothing
        self.assertEqual(self.ctl("GET", "/api/update/beta")[1]["coins"], 10)
        # 6. without a link the gated folder is closed
        for path in ("/%s/%s/%s" % (self.gate, self.BID, self.dname), "/%s/" % self.gate):
            try:
                urllib.request.urlopen(self.surl + path, timeout=10)
                self.fail("gated path served without a link: " + path)
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 403)
        # 7. back to stable: the channel changes, the beta folder stays until stable installs over it
        st, j = self.ctl("POST", "/api/update/channel", {"channel": "stable"})
        self.assertEqual((st, j["channel"]), (200, "stable"))
        with open(os.path.join(self.ins, "engine-channel")) as f:
            self.assertEqual(f.read().strip(), "stable")

    def test_valued_key_links_to_the_wallet_once(self):
        """SF4: a Valued Supporter's licence key is proven to this computer's wallet by pxa-update's channel check (licd key-link,
        install-key signed); the card then says "Yours as a Valued Supporter" and a marker skips the call next time."""
        import pxa_engine_update as EU
        from pxa_control import routes as R
        st, w = self.ctl("POST", "/api/store/wallet/create", {})                                 # (an existing wallet answers as well)
        self.assertIn(st, (200, 409), w)
        key = self.lsvc.issue_key("test", "998877665544332266", "valued", role="valued")["key"]
        self.assertTrue(EU.channels(self.lurl, key)["beta"])
        mark = os.path.join(os.environ["PXA_CONTROL_CONFIG_DIR"], "plugins", "wallet-key-linked")
        self.assertTrue(os.path.isfile(mark))
        with open(mark) as f:
            self.assertNotIn(key, f.read())                                                       # a hash, never the key
        n = self.lsvc.db.q1("SELECT COUNT(*) n FROM audit WHERE action='store.wallet.key_linked'")["n"]
        self.assertEqual(n, 1)
        self.assertTrue(EU._link_valued_key(self.lurl, key))                                      # the marker: no second call
        R._BETA_VIEW["at"] = -1e18                                                                # the card's minute has passed
        st, b = self.ctl("GET", "/api/update/beta")
        self.assertEqual((st, b["access"], b["how"], b["valued_supporter"], b["cost"]), (200, True, "valued-supporter", True, 0), b)


if __name__ == "__main__":
    unittest.main()
