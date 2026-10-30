#!/usr/bin/env python3
"""PXA Store and plugins, the local side (tools/pxa_control/store, tools/pxa_control/plugins; PLUGINS-STORE-DESIGN.md).
Keys are generated IN the test (a licd test key, a plugin test key); nothing here knows a production key. No network.

    python3 tests/test-pxa-control-store.py
"""
import hashlib
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-store-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""          # hermetic: the fixture store, never the live licd (the release default)
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
sys.path.insert(0, TOOLS)
import pxa_control as C  # noqa: E402
import pxa_encode_pkg as PK  # noqa: E402
from pxa_control import plugins as P, store as S  # noqa: E402

LICD_SEED = os.urandom(32)                      # the test's own licd key pair
LICD_KEYS = {"ent-test": S.ed25519_public(LICD_SEED).hex()}
PLUG_SEED = os.urandom(32)
PLUG_KEYS = {"plug-test": S.ed25519_public(PLUG_SEED).hex()}
NOW = 1_760_400_000


def grant(cfg, kind="quant-tier", item="pxqn", issued=NOW, ref=86400, life=180 * 86400, ent_id=None, seed=LICD_SEED,
          kid="ent-test", prefix=S.ENT_PREFIX, **over):
    _s, pub = S.install_key(cfg)
    st = {"v": 1, "iss": "licd", "kid": kid, "ent_id": ent_id or os.urandom(16).hex(), "kind": kind, "item_id": item,
          "level": "pxqn4" if kind == "quant-tier" else None, "sub": hashlib.sha256(b"acct").hexdigest()[:32],
          "install_pub": pub, "issued": issued, "refresh_after": issued + ref, "not_after": issued + life}
    st.update(over)
    return {"statement": st, "signature": S.b64u(S.ed25519_sign(seed, prefix + S.canonical(st)))}


class Ed25519(unittest.TestCase):
    def test_rfc8032_vector_1_and_the_shipped_verifier_agree(self):
        seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
        pub = S.ed25519_public(seed)
        self.assertEqual(pub.hex(), "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        sig = S.ed25519_sign(seed, b"")
        self.assertEqual(sig.hex(), "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e3970"
                                    "1cf9b46bd25bf5f0595bbe24655141438e7a100b")
        self.assertTrue(PK.ed25519_verify(pub, b"", sig))
        self.assertFalse(PK.ed25519_verify(pub, b"x", sig))

    def test_install_key_is_private_and_stable(self):
        cfg = tempfile.mkdtemp(dir=TMP)
        s1, p1 = S.install_key(cfg)
        s2, p2 = S.install_key(cfg)
        self.assertEqual((s1, p1), (s2, p2))
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(cfg, "plugins", "install.key")).st_mode), 0o600)
        ch = S.refresh_challenge(cfg, "n0nce")
        self.assertTrue(PK.ed25519_verify(bytes.fromhex(ch["install_pub"]), b"pxa-plugin-install|n0nce", S._b64d(ch["sig"])))


class Grants(unittest.TestCase):
    def setUp(self):
        self.cfg = tempfile.mkdtemp(dir=TMP)
        self.pub = S.install_key(self.cfg)[1]

    def v(self, env, now=NOW + 10):
        return S.verify_grant(env, LICD_KEYS, now, self.pub)

    def test_valid_refresh_due_expired(self):
        g = grant(self.cfg)
        self.assertEqual(self.v(g)[1], "ok")
        self.assertEqual(self.v(g, NOW + 2 * 86400)[1], "refresh_due")             # offline: still counts
        with self.assertRaises(S.GrantError) as e:
            self.v(g, NOW + 181 * 86400)
        self.assertEqual(e.exception.code, "expired")

    def test_refusals(self):
        cases = {
            "bad_signature": [dict(prefix=b"pxa-other|"), dict(seed=os.urandom(32))],    # domain separation; another key
            "unknown_key": [dict(kid="ent-nope")],
            "bad_entitlement": [dict(kind="weapon"), dict(item="Bad Item")],
            "bad_issuer": [dict(iss="me")],
            "bad_shape": [dict(ref=-5), dict(life=500 * 86400), dict(install_pub="zz")],
            "not_yet": [dict(issued=NOW + 3600)],
        }
        for code, variants in cases.items():
            for kw in variants:
                g = grant(self.cfg, **kw)
                with self.assertRaises(S.GrantError, msg=kw) as e:
                    self.v(g)
                self.assertEqual(e.exception.code, code, kw)
        g = grant(self.cfg)
        g["statement"]["level"] = "pxqn5"                                           # tamper after signing
        with self.assertRaises(S.GrantError) as e:
            self.v(g)
        self.assertEqual(e.exception.code, "bad_signature")
        other = tempfile.mkdtemp(dir=TMP)                                           # copied to another computer
        with self.assertRaises(S.GrantError) as e:
            S.verify_grant(grant(self.cfg), LICD_KEYS, NOW + 10, S.install_key(other)[1])
        self.assertEqual(e.exception.code, "wrong_machine")

    def test_cache_replay_revocation_rollback(self):
        clock = [NOW + 10]
        cache = S.EntitlementCache(self.cfg, LICD_KEYS, clock=lambda: clock[0])
        old = grant(self.cfg, ent_id="ab" * 16, issued=NOW)
        new = grant(self.cfg, ent_id="ab" * 16, issued=NOW + 5)
        cache.add(new)
        with self.assertRaises(S.GrantError) as e:
            cache.add(old)                                                          # a backup of an older copy
        self.assertEqual(e.exception.code, "replayed")
        self.assertTrue(cache.has("pxqn", "quant-tier"))
        self.assertEqual(stat.S_IMODE(os.stat(cache.path).st_mode), 0o600)
        cache.add(grant(self.cfg, kind="plugin", item="anibal", ent_id="cd" * 16))
        self.assertEqual(sorted(st["item_id"] for st, _s in cache.active()), ["anibal", "pxqn"])
        n = cache.apply_revocations(S.sign_revocations(LICD_SEED, "ent-test", ["ab" * 16], NOW + 20))
        self.assertEqual(n, 1)
        self.assertEqual([st["item_id"] for st, _s in cache.active()], ["anibal"])
        with self.assertRaises(S.GrantError) as e:
            cache.add(grant(self.cfg, ent_id="ab" * 16, issued=NOW + 30))
        self.assertEqual(e.exception.code, "revoked")
        bad = S.sign_revocations(os.urandom(32), "ent-test", ["cd" * 16], NOW)       # forged list
        with self.assertRaises(S.GrantError):
            cache.apply_revocations(bad)
        clock[0] = NOW + 10 * 86400
        self.assertEqual(cache.active()[0][1], "refresh_due")
        cache.set_locks({})                    # reads never write (F303 contract): `seen` moves on a write (every sync writes)
        clock[0] = NOW + 10 * 86400 - 3 * 86400                                     # clock set back 3 days
        self.assertEqual(cache.active(), [])
        for g in (grant(self.cfg, ent_id="ef" * 16, issued=NOW, seed=os.urandom(32)),    # not licd's signature
                  grant(self.cfg, ent_id="cd" * 16, kind="plugin", item="anibal")):    # nothing newer from licd
            with self.assertRaises(S.GrantError) as e:
                cache.add(g)
            self.assertEqual(e.exception.code, "clock_rollback")
        # F310 (b): a VERIFIED licd grant, newer than any copy of its ent_id, moves the fast `seen` back to licd's time
        # (max of its issued and every other grant's newest), never further: the clock is then within 48 h again
        cache.add(grant(self.cfg, ent_id="ef" * 16, issued=NOW + 6))
        self.assertEqual(json.load(open(cache.path))["seen"], NOW + 6)
        self.assertEqual(sorted(st["item_id"] for st, _s in cache.active()), ["anibal", "pxqn"])


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(C.desktop.load_launcher(), port=7777)
        cls.app.store_cache = S.EntitlementCache(C.config_dir(), LICD_KEYS, clock=lambda: NOW + 10)
        cls.app.store_cache.add(grant(C.config_dir(), kind="preset", item="overdrive-presets", ent_id="99" * 16))
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def get(self, path, token=True):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers={"X-PXA-Token": self.app.token} if token else {})
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_catalog_shape(self):
        """the shape the UX lane's store.js renders (the Control UX pass store.js header)."""
        st, d = self.get("/api/store/catalog")
        self.assertEqual((st, d["source"], d["version"]), (200, "fixture", 1))
        self.assertEqual(set(d["currency"]), {"name", "source", "rate_note"})
        self.assertEqual({i["kind"] for i in d["items"]}, set(S.ENT_KINDS))
        for i in d["items"]:
            self.assertTrue({"id", "kind", "name", "tagline", "description", "price", "permissions"} <= set(i), i["id"])
            for p in i["permissions"]:
                self.assertEqual(set(p), {"id", "group", "label"})
                self.assertIn(p["id"], P.PERMISSIONS)
        self.assertEqual(self.get("/api/store/catalog", token=False)[0], 401)
        st, info = self.get("/api/info")
        self.assertEqual(info["store"], {"catalog": "/api/store/catalog", "entitlements": "/api/store/entitlements"})

    def test_entitlements_shape(self):
        st, d = self.get("/api/store/entitlements")
        self.assertEqual(st, 200)
        self.assertEqual(set(d), {"account", "credits", "items", "grants", "receipts", "locks", "hard_lock", "sync", "machine", "cache"})
        self.assertEqual(d["cache"], {"state": "ok", "hard_lock": None, "corrupt_copy": None, "since": None, "clock": None,
                                      "why": None})
        self.assertEqual((d["account"], d["credits"], d["receipts"], d["grants"]), (None, None, [], ["overdrive-presets"]))
        self.assertEqual(d["items"], {"overdrive-presets": {"state": "owned", "since": NOW, "expires": NOW + 180 * 86400,
                                                            "kind": "preset", "level": None, "grant": "99" * 6, "status": "ok",
                                                            "refresh_after": NOW + 86400}})
        self.assertEqual(d["sync"], {"online": False, "checked": None, "grace_days": 180, "grace_until": NOW + 180 * 86400})
        self.assertEqual(len(d["machine"]), 16)
        self.assertEqual(self.get("/api/store/entitlements", token=False)[0], 401)
        seed = S.install_key(C.config_dir())[0]
        self.assertNotIn(seed.hex(), json.dumps(d))


def anibal_manifest(root, seed=PLUG_SEED, kid="plug-test", **over):
    os.makedirs(os.path.join(root, "bin"), exist_ok=True)
    os.makedirs(os.path.join(root, "ui"), exist_ok=True)
    files = {"bin/anibal-plugin": b"#!/bin/sh\necho anibal\n", "ui/index.html": b"<!doctype html><title>Anibal</title>"}
    for rel, b in files.items():
        with open(os.path.join(root, rel), "wb") as f:
            f.write(b)
    m = {"schema": 1, "plugin_id": "anibal", "name": "Anibal", "version": "0.1.0", "publisher": "pxa", "edition": "free",
         "min_host": "3.1", "entry": {"kind": "process", "argv": ["bin/anibal-plugin"], "protocol": "pxa-plugin/1"},
         "routes": ["/api/plugins/anibal/"], "ui": {"tab": {"title": "Anibal", "page": "ui/index.html"}},
         "permissions": ["device.mic", "device.speaker", "model.chat", "net.loopback"], "entitlement": None,
         "files": {rel: hashlib.sha256(b).hexdigest() for rel, b in files.items()}}
    m.update(over)
    m["signature"] = {"kid": kid, "sig": S.b64u(S.ed25519_sign(seed, P.MANIFEST_PREFIX + P.canonical(m)))}
    with open(os.path.join(root, "plugin.json"), "w") as f:
        json.dump(m, f)
    return m


class Plugins(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=TMP)

    def test_anibal_loads_and_verifies(self):
        anibal_manifest(self.root)
        m = P.load_manifest(self.root, PLUG_KEYS)
        self.assertEqual(m["plugin_id"], "anibal")

    def test_manifest_refusals(self):
        bad = [dict(publisher="acme"), dict(routes=["/api/start"]), dict(permissions=["device.mic", "root"]),
               dict(entry={"kind": "process", "argv": ["../bin/x"], "protocol": "pxa-plugin/1"}), dict(schema=2),
               dict(plugin_id="Anibal!"), dict(version="1"), dict(edition="gold"), dict(files={"../x": "0" * 64})]
        for kw in bad:
            m = anibal_manifest(self.root, **kw)
            self.assertTrue(P.validate_manifest(m), kw)
            with self.assertRaises(P.PluginError, msg=kw):
                P.verify_manifest(m, PLUG_KEYS)

    def test_signature_files_and_links(self):
        m = anibal_manifest(self.root, seed=os.urandom(32))                 # signed by someone else
        with self.assertRaises(P.PluginError) as e:
            P.verify_manifest(m, PLUG_KEYS)
        self.assertEqual(e.exception.code, "bad_signature")
        m = anibal_manifest(self.root, kid="plug-unknown")
        with self.assertRaises(P.PluginError) as e:
            P.verify_manifest(m, PLUG_KEYS)
        self.assertEqual(e.exception.code, "unknown_key")
        m = anibal_manifest(self.root)
        with open(os.path.join(self.root, "bin", "anibal-plugin"), "ab") as f:
            f.write(b"evil\n")
        with self.assertRaises(P.PluginError) as e:
            P.load_manifest(self.root, PLUG_KEYS)
        self.assertEqual(e.exception.code, "bad_file")
        m = anibal_manifest(self.root)
        os.remove(os.path.join(self.root, "ui", "index.html"))
        os.symlink("/etc/hostname", os.path.join(self.root, "ui", "index.html"))
        with self.assertRaises(P.PluginError) as e:
            P.verify_files(m, self.root)
        self.assertEqual(e.exception.code, "bad_file")

    def test_consent(self):
        cfg = tempfile.mkdtemp(dir=TMP)
        cs = P.ConsentStore(cfg, clock=lambda: NOW)
        m = anibal_manifest(self.root)
        self.assertEqual(cs.state(m)["status"], "none")
        with self.assertRaises(P.PluginError):
            cs.grant(m, ["device.mic", "host.shell"])                       # more than it asked for
        cs.grant(m, ["device.mic", "model.chat"])
        st = cs.state(m)
        self.assertEqual((st["status"], st["missing"]), ("partial", ["device.speaker", "net.loopback"]))
        self.assertTrue(cs.allowed(m, "device.mic"))
        self.assertFalse(cs.allowed(m, "net.loopback"))
        self.assertEqual(stat.S_IMODE(os.stat(cs.path).st_mode), 0o600)
        m2 = anibal_manifest(self.root, version="0.2.0", permissions=m["permissions"] + ["home.assistant"])
        self.assertEqual(cs.state(m2)["status"], "needs_reconsent")         # an update asking for more
        self.assertFalse(cs.allowed(m2, "home.assistant"))
        self.assertTrue(cs.revoke("anibal"))
        self.assertEqual(cs.state(m)["status"], "revoked")
        self.assertFalse(cs.allowed(m, "device.mic"))


class BetaCardPolling(unittest.TestCase):
    """SF3 (REVIEW-CODE.md round 2): the page polls every 2 s during a 20-minute Docker download. licd allows a computer 120 wallet
    calls an hour (wallet-pc) and an address 120 nonces per 10 minutes. The poll must never trip that, and a 429 must neither hide
    the card nor lock the wallet: the last good card stays and Control backs off until licd's Retry-After."""
    class FakeLicd(object):
        def __init__(self, clock):
            self.clock, self.calls, self.limited, self.hits = clock, [], 0, []

        def wallet(self, action, args=None):
            t = self.clock[0]
            self.hits = [x for x in self.hits if x > t - 3600] + [t]        # licd's sliding window: 120 an hour per install key
            self.calls.append(t)
            if len(self.hits) > 120:
                self.limited += 1
                e = C.Invalid("store: too many requests", status=429, code="store_error")
                e.retry_after = 600
                raise e
            return {"wallet_id": "PXW-AAAA-BBBB-CCCC", "coins": 70, "beta": {"release": "4.0-beta.2", "access": True, "how": "redeemed"}}

    def setUp(self):
        from pxa_control import routes as R
        self.R, self.clock = R, [1000.0]
        self.fake = self.FakeLicd(self.clock)
        self.saved = (R._beta_client, R._beta_clock, S.store_url, dict(R._BETA_VIEW), dict(R._BETA_DL))
        R._beta_client, R._beta_clock = (lambda: self.fake), (lambda: self.clock[0])
        S.store_url = lambda: "http://licd.test"
        R._BETA_VIEW.update(card=None, at=-1e18, epoch=-1, until=0.0)
        R._BETA_DL.update(state="running", done=0, size=2_000_000_000)

    def tearDown(self):
        R = self.R
        R._beta_client, R._beta_clock, S.store_url = self.saved[:3]
        R._BETA_VIEW.clear(); R._BETA_VIEW.update(self.saved[3])
        R._BETA_DL.clear(); R._BETA_DL.update(self.saved[4])

    def poll(self, q):
        return self.R.r_update_beta(None, None, q)

    def test_twenty_minute_download_polled_every_2s_never_trips_the_limiter(self):
        card = self.poll({})                                        # the dialog opens: one wallet read
        self.assertEqual((card["release"], card["access"]), ("4.0-beta.2", True))
        for i in range(600):                                        # 20 minutes, a tick every 2 s: progress is local
            self.clock[0] += 2
            self.R._BETA_DL["done"] = i * 3_000_000
            j = self.poll({"dl": ["1"]})
            self.assertEqual(j["docker"]["done"], i * 3_000_000)
        self.assertEqual((len(self.fake.calls), self.fake.limited), (1, 0))
        for i in range(600):                                        # worst case: the full card every 2 s (an old page, two tabs)
            self.clock[0] += 2
            j = self.poll({})
            self.assertTrue(j["store"] and j["release"] == "4.0-beta.2")
        self.assertLessEqual(len(self.fake.calls), 1 + 21)          # at most one wallet read a minute
        self.assertEqual(self.fake.limited, 0)

    def test_a_429_keeps_the_card_and_backs_off_until_retry_after(self):
        self.poll({})
        self.fake.hits = [self.clock[0]] * 120                      # something else spent the hour's budget
        self.clock[0] += 61
        j = self.poll({})
        self.assertEqual((j["release"], j["access"], j["busy"]), ("4.0-beta.2", True, True))   # the card stays
        self.assertEqual(j["retry_in"], 601)                         # the page looks again by itself after the back-off (D2-4)
        n = len(self.fake.calls)
        for _ in range(250):                                        # 500 s of 2 s polls inside Retry-After: no licd call at all
            self.clock[0] += 2
            j = self.poll({})
            self.assertTrue(j["access"])
        self.assertEqual(len(self.fake.calls), n)
        self.clock[0] += 120
        self.fake.hits = []
        self.assertFalse(self.poll({}).get("busy"))                 # after Retry-After the wallet reads again, nothing locked
        self.assertEqual(len(self.fake.calls), n + 1)

    def test_a_wallet_change_refreshes_the_card_at_once(self):
        self.poll({})
        self.poll({})
        self.assertEqual(len(self.fake.calls), 1)
        S.WALLET_CHANGES[0] += 1                                    # e.g. the Store tab redeemed or joined
        self.poll({})
        self.assertEqual(len(self.fake.calls), 2)


class ValuedKeyLinkMemo(unittest.TestCase):
    """D2-6: with no writable marker the key-link is still made once per process; a refusal is not retried for 10 minutes."""
    def test_unwritable_marker_and_refusal_do_not_repeat_licd_calls(self):
        import pxa_engine_update as EU
        from unittest import mock
        calls = []
        def call(base, path, body, *a, **k):
            calls.append(path); return {"beta": True}
        def wcall(base, what, args, cfg_dir=None, op=None, path=None):
            calls.append(what)
            if args["key"].endswith("used"):
                raise EU.PackageError("this licence key is already linked to another PXA wallet", "key_used")
            return {"valued_supporter": True}
        d = tempfile.mkdtemp()                                        # no plugins/ dir: the marker cannot be written
        now = [1000000.0]
        with mock.patch.object(EU, "wallet_key", lambda cfg_dir=None: (b"s" * 32, "ab" * 32)), \
             mock.patch.object(EU.pkg, "valid_key", lambda k: True), mock.patch.object(EU, "_call", call), \
             mock.patch.object(EU, "_wallet_call", wcall), mock.patch.object(EU.time, "time", lambda: now[0]):
            self.assertTrue(EU._link_valued_key("https://lic.test", "pxk1-ok", d))
            self.assertTrue(EU._link_valued_key("https://lic.test", "pxk1-ok", d))
            self.assertEqual(calls, ["/v1/engine/channels", "key-link"])
            self.assertFalse(EU._link_valued_key("https://lic.test", "pxk1-used", d))
            self.assertFalse(EU._link_valued_key("https://lic.test", "pxk1-used", d))
            self.assertEqual(len(calls), 4)
            now[0] += 601
            self.assertFalse(EU._link_valued_key("https://lic.test", "pxk1-used", d))
            self.assertEqual(len(calls), 6)


if __name__ == "__main__":
    unittest.main(verbosity=1)
