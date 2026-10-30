#!/usr/bin/env python3
"""PXA Control's store client (tools/pxa_control/store LicdClient + EntitlementCache) on the spec readings Core's gate
(anibal/a-store-gate b0115812b6, tools/vigil/entitlement.py) and the plugin host (plugins/v1-store 03294342ea,
tools/pxa_plugins/store_backend.py) already share (fixtures/entitlement/PROVENANCE.md):

  * https only; plain http only on loopback (the store URL and a download token's URL);
  * a garbled or cut-off HTTP reply (http.client.HTTPException) is OFFLINE; a 2xx that is not one JSON object is
    MALFORMED; neither changes the cache;
  * licd's account.owned[]: an entry without a string item_id and status (or owned / grants not a list) makes the whole
    reply malformed (nothing applied, not even its revocations); a present status other than "active" locks the item;
  * the shared 8-vector fixture (sha256 pinned on all three sides), verify layer and cache layer.

A fake licd on loopback (raw sockets for the garbled replies), throwaway keys made here. No GPU, no network.

    python3 tests/test-pxa-control-store-verify.py          (wired into CTest as test-pxa-control-store-verify)
"""
import hashlib
import http.server
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-store-verify-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
sys.path.insert(0, TOOLS)
import pxa_control as C  # noqa: E402
from pxa_control import store as S  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "anibal-entitlement-vectors.json")
FIXTURE_SHA256 = "b84c41b391a4369018b592e3adc94f333ffa615de296da18c12027b6fecca880"   # = Core VectorProvenance + Plugins TestCoreReadings
VECTOR_LABEL = b"pxa-anibal-entitlement-vectors|DEV-ONLY|v1"                           # pxa_plugins/store_vectors.LABEL
SEED = os.urandom(32)
KID = "ent-test"
KEYS = {KID: S.ed25519_public(SEED).hex()}


def grant(cfg, item="anibal", kind="plugin", ent_id=None, issued=None, seed=SEED):
    _s, pub = S.install_key(cfg)
    t = int(time.time()) if issued is None else issued
    st = {"v": 1, "iss": "licd", "kid": KID, "ent_id": ent_id or os.urandom(16).hex(), "kind": kind, "item_id": item,
          "level": None, "sub": "cd" * 16, "install_pub": pub, "issued": t, "refresh_after": t + 86400, "not_after": t + 180 * 86400}
    return {"statement": st, "signature": S.b64u(S.ed25519_sign(seed, S.ENT_PREFIX + S.canonical(st)))}


def revocations(ent_ids, issued=None):
    return S.sign_revocations(SEED, KID, list(ent_ids), int(time.time()) if issued is None else issued)


# ---------------------------------------------------------------------------------------------------------- fake licd
class FakeLicd(http.server.BaseHTTPRequestHandler):
    """script[path] = callable(body) -> (status, obj | bytes) or the string "garbage" | "cut" for a broken HTTP reply."""
    script = {}
    calls = []
    date_header = None          # None: the real Date header; "": no Date header; any other string: sent as is

    def log_message(self, *a):
        pass

    def send_header(self, k, v):
        if k.lower() == "date" and FakeLicd.date_header is not None:
            if FakeLicd.date_header == "":
                return
            v = FakeLicd.date_header
        super().send_header(k, v)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        self._answer(body)

    def do_GET(self):
        self._answer(None)

    def _answer(self, body):
        FakeLicd.calls.append(self.path)
        fn = FakeLicd.script.get(self.path)
        if fn is None and self.path.startswith("/v1/store/download/"):
            fn = FakeLicd.script.get("/v1/store/download/")
        if fn is None:
            return self._send(404, {"error": "no such route", "code": "not_found"})
        r = fn(body)
        if r == "garbage":                          # not HTTP at all: a bad status line
            self.wfile.write(b"GARBAGE \x00\xff NOT HTTP\r\n\r\n")
            self.close_connection = True
            return
        if r == "cut":                              # promises 400 bytes, sends 10, hangs up: IncompleteRead
            self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 400\r\n\r\n{\"grants\":")
            self.wfile.flush()
            self.close_connection = True
            return
        self._send(*r)

    def _send(self, code, obj, headers=None):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


def http_date(t):
    import email.utils
    return email.utils.formatdate(t, usegmt=True)


def nonce_ok(_b):
    return 200, {"nonce": os.urandom(8).hex()}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLicd)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.cfg = tempfile.mkdtemp(dir=TMP)
        FakeLicd.script = {"/v1/store/nonce": nonce_ok}
        FakeLicd.calls = []
        FakeLicd.date_header = None
        self.client = S.LicdClient(self.url, self.cfg, pubkeys=KEYS, timeout=10)
        self.cache = S.EntitlementCache(self.cfg, pubkeys=KEYS)

    def refresh_reply(self, grants=(), owned=None, revs=None, credits=7):
        r = {"grants": list(grants), "account": {"credits": credits, "coins": credits,
                                                 "owned": owned if owned is not None else
                                                 [{"item_id": g["statement"]["item_id"], "status": "active"} for g in grants]}}
        if revs is not None:
            r["revocations"] = revs
        return r

    def on_refresh(self, reply):
        FakeLicd.script["/v1/store/refresh"] = (lambda _b: reply) if isinstance(reply, str) else (lambda _b: (200, reply))

    def code(self, fn, *a):
        with self.assertRaises(C.Invalid) as e:
            fn(*a)
        return getattr(e.exception, "status", 400), getattr(e.exception, "code", "invalid")

    def primed(self, item="anibal"):
        """a cache holding one good grant (via a good sync), like any linked computer."""
        g = grant(self.cfg, item)
        self.on_refresh(self.refresh_reply([g]))
        self.assertEqual(self.client.sync(self.cache)["grants"], 1)
        self.assertTrue(self.cache.has(item))
        return g

    def snapshot(self):
        with open(self.cache.path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------------------------------------- (1) https
class HttpsOnly(Base):
    def test_check_url_accepts_https_and_loopback_http(self):
        for u in ("https://lic.pxanetwork.com", "https://lic.pxanetwork.com/", "http://127.0.0.1:8799", "http://localhost:1",
                  "http://[::1]:8799", " https://lic.pxanetwork.com "):
            self.assertEqual(S.check_url(u), u.strip().rstrip("/"), u)

    def test_check_url_refuses_the_rest(self):
        for u in ("http://lic.pxanetwork.com", "http://192.168.1.5:8799", "http://10.0.0.2", "http://127.0.0.1.evil.com",
                  "http://localhost.evil.com", "ftp://127.0.0.1", "file:///etc/passwd", "lic.pxanetwork.com", "https://", "",
                  None, "javascript:alert(1)", "http://[::ffff:8.8.8.8]"):
            self.assertEqual(self.code(S.check_url, u), (400, "bad_url"), u)

    def test_client_refuses_a_plain_http_store(self):
        self.assertEqual(self.code(S.LicdClient, "http://lic.pxanetwork.com", self.cfg), (400, "bad_url"))
        self.assertEqual(S.LicdClient("https://lic.pxanetwork.com/", self.cfg).base, "https://lic.pxanetwork.com")
        self.assertEqual(S.DEFAULT_STORE_URL[:8], "https://")
        S.LicdClient(S.DEFAULT_STORE_URL, self.cfg)                                 # the release default passes

    def test_absolute_urls_in_requests_are_checked_too(self):
        self.assertEqual(self.code(self.client._req, "GET", "http://cdn.example.com/x"), (400, "bad_url"))
        self.assertEqual(FakeLicd.calls, [])

    def test_download_token_url_must_be_https_off_loopback_and_is_not_retried(self):
        n = [0]

        def tok(_b):
            n[0] += 1
            return 200, {"name": "m.gguf", "size": 4, "sha256": "00" * 32, "token": "t", "url": "http://cdn.example.com/v1/x"}
        FakeLicd.script["/v1/store/download-token"] = tok
        dest = tempfile.mkdtemp(dir=TMP)
        self.assertEqual(self.code(self.client.download, "pxqn", "f1", dest), (400, "bad_url"))
        self.assertEqual(n[0], 1)
        self.assertEqual(os.listdir(dest), [])

    def test_loopback_download_still_works(self):
        blob = b"abcd"
        FakeLicd.script["/v1/store/download-token"] = lambda _b: (200, {"name": "m.gguf", "size": 4, "token": "t1",
                                                                        "sha256": hashlib.sha256(blob).hexdigest()})
        FakeLicd.script["/v1/store/download/"] = lambda _b: (200, blob)
        dest = tempfile.mkdtemp(dir=TMP)
        r = self.client.download("pxqn", "f1", dest)
        self.assertEqual((r["size"], open(r["path"], "rb").read()), (4, blob))

    def test_the_routes_refuse_a_plain_http_store_url(self):
        old = os.environ.get("PXA_STORE_URL")
        os.environ["PXA_STORE_URL"] = "http://lic.pxanetwork.com"
        try:
            app = types.SimpleNamespace()
            for key in (("POST", "/api/store/refresh"), ("POST", "/api/store/unlink")):
                self.assertEqual(self.code(C.ROUTES[key], app, {}, {}), (400, "bad_url"), key)
            self.assertEqual(self.code(C.routes._beta_client), (400, "bad_url"))
        finally:
            os.environ["PXA_STORE_URL"] = old if old is not None else ""


# ---------------------------------------------------------------------------------------------------------- (2) transport
class Transport(Base):
    def test_garbled_http_is_offline(self):
        FakeLicd.script["/v1/store/nonce"] = lambda _b: "garbage"
        self.assertEqual(self.code(self.client.call, "refresh"), (503, "store_offline"))

    def test_cut_off_reply_is_offline(self):
        FakeLicd.script["/v1/store/catalog"] = lambda _b: "cut"
        self.assertEqual(self.code(self.client._req, "GET", "/v1/store/catalog"), (503, "store_offline"))

    def test_nobody_listening_is_offline(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        c = S.LicdClient("http://127.0.0.1:%d" % port, self.cfg, timeout=5)
        self.assertEqual(self.code(c._req, "GET", "/v1/store/catalog"), (503, "store_offline"))

    def test_2xx_that_is_not_a_json_object_is_malformed(self):
        for body in (b"<html>cloudflare says hi</html>", b"[1, 2]", b"\"x\"", b"null", b"\xff\xfe"):
            FakeLicd.script["/v1/store/revocations"] = lambda _b, body=body: (200, body)
            self.assertEqual(self.code(self.client.revocations), (503, "store_malformed"), body)
        FakeLicd.script["/v1/store/revocations"] = lambda _b: (200, b"")
        self.assertEqual(self.client.revocations(), {})                               # empty body = {} as before

    def test_oversized_reply_is_malformed(self):
        old = S.MAX_REPLY
        S.MAX_REPLY = 32
        try:
            FakeLicd.script["/v1/store/revocations"] = lambda _b: (200, {"pad": "x" * 100})
            self.assertEqual(self.code(self.client.revocations), (503, "store_malformed"))
        finally:
            S.MAX_REPLY = old

    def test_nonce_reply_without_a_nonce_is_malformed_not_a_500(self):
        for reply in ({}, {"nonce": None}, {"nonce": 5}, {"nonce": ""}):
            FakeLicd.script["/v1/store/nonce"] = lambda _b, reply=reply: (200, reply)
            self.assertEqual(self.code(self.client.call, "refresh"), (503, "store_malformed"), reply)
            self.assertEqual(self.code(self.client.wallet, "view"), (503, "store_malformed"), reply)

    def test_licd_4xx_and_5xx_keep_their_status_code_and_retry_after(self):
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (402, {"error": "not enough coins", "code": "not_enough_coins"})
        self.assertEqual(self.code(self.client.call, "refresh"), (402, "not_enough_coins"))
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (429, {"error": "slow down", "code": "rate_limited"}, {"Retry-After": "42"})
        with self.assertRaises(C.Invalid) as e:
            self.client.call("refresh")
        self.assertEqual((e.exception.status, e.exception.code, e.exception.retry_after), (429, "rate_limited", 42))

    def test_error_bodies_that_are_garbage_never_crash(self):
        for body in (b"<html>502</html>", b"[\"code\"]", b"{\"code\": 5, \"error\": [1]}", b"\xff"):
            FakeLicd.script["/v1/store/refresh"] = lambda _b, body=body: (403, body)
            self.assertEqual(self.code(self.client.call, "refresh"), (403, "store_error"), body)
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (503, b"busy")
        self.assertEqual(self.code(self.client.call, "refresh"), (503, "store_error"))

    def test_offline_and_malformed_syncs_change_nothing(self):
        g = self.primed()
        before = self.snapshot()
        for reply in ("garbage", "cut", (200, b"not json"), (200, b"[]")):
            FakeLicd.script["/v1/store/refresh"] = lambda _b, reply=reply: reply
            st, code = self.code(self.client.sync, self.cache)
            self.assertEqual(st, 503, reply)
            self.assertIn(code, ("store_offline", "store_malformed"), reply)
            after = self.snapshot()
            self.assertEqual((after["grants"], after["revoked"], after["locks"]), (before["grants"], before["revoked"], before["locks"]))
            self.assertTrue(self.cache.has("anibal"), reply)
        self.assertIn(g["statement"]["ent_id"], self.snapshot()["grants"])


# ---------------------------------------------------------------------------------------------------------- (3) owned[] locks
class OwnedLocks(Base):
    def test_status_not_active_locks_and_drops_the_grant(self):
        g = self.primed()
        self.on_refresh(self.refresh_reply([], owned=[{"item_id": "anibal", "status": "revoked"}]))
        r = self.client.sync(self.cache)
        self.assertEqual(r["locked"], ["anibal"])
        self.assertFalse(self.cache.has("anibal"))
        d = self.snapshot()
        self.assertEqual(d["locks"], {"anibal": "revoked"})
        self.assertNotIn(g["statement"]["ent_id"], d["grants"])
        self.assertNotIn("anibal", S.entitlements_view(self.cache)["items"])

    def test_a_locked_items_grant_in_the_same_reply_does_not_unlock_it(self):
        self.primed()
        g2 = grant(self.cfg, "anibal")
        self.on_refresh(self.refresh_reply([g2], owned=[{"item_id": "anibal", "status": "suspended"}]))
        self.assertEqual(self.client.sync(self.cache)["locked"], ["anibal"])
        self.assertFalse(self.cache.has("anibal"))

    def test_active_again_unlocks_with_the_next_grant(self):
        self.primed()
        self.on_refresh(self.refresh_reply([], owned=[{"item_id": "anibal", "status": "revoked"}]))
        self.client.sync(self.cache)
        self.assertFalse(self.cache.has("anibal"))
        self.on_refresh(self.refresh_reply([grant(self.cfg, "anibal")]))
        r = self.client.sync(self.cache)
        self.assertEqual(r["locked"], [])
        self.assertTrue(self.cache.has("anibal"))
        self.assertEqual(self.snapshot()["locks"], {})

    def test_only_the_non_active_item_locks(self):
        self.primed("anibal")
        gp = grant(self.cfg, "pxqn", kind="quant-tier")
        self.on_refresh(self.refresh_reply([gp], owned=[{"item_id": "anibal", "status": "chargeback"},
                                                        {"item_id": "pxqn", "status": "active"}]))
        self.assertEqual(self.client.sync(self.cache)["locked"], ["anibal"])
        self.assertTrue(self.cache.has("pxqn"))
        self.assertFalse(self.cache.has("anibal"))

    def test_owned_without_status_is_malformed_and_changes_nothing_even_revocations(self):
        g = self.primed()
        before = self.snapshot()
        reply = self.refresh_reply([], owned=[{"item_id": "anibal"}], revs=revocations([g["statement"]["ent_id"]]))
        self.on_refresh(reply)
        self.assertEqual(self.code(self.client.sync, self.cache), (503, "store_malformed"))
        after = self.snapshot()
        self.assertEqual((after["grants"], after["revoked"], after["locks"]), (before["grants"], before["revoked"], before["locks"]))
        self.assertTrue(self.cache.has("anibal"))

    def test_other_malformed_account_answers_change_nothing(self):
        self.primed()
        good = self.refresh_reply([])
        muts = (lambda r: r["account"]["owned"].append({"item_id": "anibal", "status": None}),
                lambda r: r["account"]["owned"].append({"status": "revoked"}),
                lambda r: r["account"]["owned"].append({"item_id": 7, "status": "revoked"}),
                lambda r: r["account"]["owned"].append("anibal"),
                lambda r: r["account"].__setitem__("owned", "anibal"),
                lambda r: r["account"].pop("owned"),
                lambda r: r.pop("account"),
                lambda r: r.__setitem__("account", []),
                lambda r: r.__setitem__("grants", {"x": 1}),
                lambda r: r.pop("grants"))
        for i, mut in enumerate(muts):
            r = json.loads(json.dumps(good))
            mut(r)
            self.on_refresh(r)
            self.assertEqual(self.code(self.client.sync, self.cache), (503, "store_malformed"), i)
            self.assertEqual(self.snapshot()["locks"], {}, i)
            self.assertTrue(self.cache.has("anibal"), i)

    def test_revocations_apply_before_grants(self):
        self.primed()
        g2 = grant(self.cfg, "pxqn", kind="quant-tier")
        self.on_refresh(self.refresh_reply([g2], revs=revocations([g2["statement"]["ent_id"]])))
        r = self.client.sync(self.cache)
        self.assertIn("revoked", r["refused"])
        self.assertFalse(self.cache.has("pxqn"))

    def test_a_forged_revocation_list_is_refused_not_a_crash(self):
        self.primed()
        st = {"v": 1, "iss": "licd", "kid": KID, "type": "revocations", "ent_ids": ["ab" * 16], "issued": 1}
        forged = {"statement": st, "signature": S.b64u(S.ed25519_sign(os.urandom(32), S.REVOKE_PREFIX + S.canonical(st)))}
        self.on_refresh(self.refresh_reply([], owned=[{"item_id": "anibal", "status": "active"}], revs=forged))
        r = self.client.sync(self.cache)
        self.assertEqual(r["refused"], ["revocations:bad_signature"])
        self.assertTrue(self.cache.has("anibal"))

    def test_locks_the_plugin_host_wrote_are_honoured(self):
        self.primed()
        d = self.snapshot()
        d["locks"] = {"anibal": "account_disabled"}
        d["last_sync"] = 1
        self.cache._save(d)
        self.assertFalse(self.cache.has("anibal"))
        self.assertEqual(self.cache.locks(), {"anibal": "account_disabled"})
        self.assertEqual(self.snapshot()["last_sync"], 1)                             # Control keeps the host's own keys

    def test_old_cache_files_without_locks_still_load(self):
        os.makedirs(self.cache.dir, exist_ok=True)
        with open(self.cache.path, "w") as f:
            json.dump({"grants": {}, "revoked": [], "newest": {}, "seen": 0}, f)
        self.assertEqual(self.cache.locks(), {})
        with open(self.cache.path, "w") as f:      # a known key of the wrong type: fail closed (F305-P), not "no locks"
            json.dump({"grants": {}, "revoked": [], "newest": {}, "seen": 0, "locks": ["x"]}, f)
        self.assertEqual(self.cache.locks(), {"*": "corrupt"})

    def test_the_account_snapshot_is_kept_on_a_good_sync_only(self):
        self.primed()
        self.assertEqual(S.load_snapshot(self.cfg)["credits"], 7)
        self.on_refresh(self.refresh_reply([], owned=[{"item_id": "anibal"}], credits=99))
        self.code(self.client.sync, self.cache)
        self.assertEqual(S.load_snapshot(self.cfg)["credits"], 7)


# ---------------------------------------------------------------------------------------------------------- (4) 403 hard-lock
class HardLock403(Base):
    """licd 403 revoked | not_linked | account_disabled locks everything (Core's gate b0115812b6, Plugins 03294342ea);
    only a fresh grant that verifies unlocks an item; any other refusal or trouble changes nothing."""

    def deny(self, code, status=403):
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (status, {"error": "no", "code": code})

    def two_items(self):
        ga, gp = grant(self.cfg, "anibal"), grant(self.cfg, "pxqn", kind="quant-tier")
        self.on_refresh(self.refresh_reply([ga, gp], owned=[{"item_id": "anibal", "status": "active"},
                                                            {"item_id": "pxqn", "status": "active"},
                                                            {"item_id": "preset-x", "status": "active"}]))
        self.client.sync(self.cache)
        self.assertTrue(self.cache.has("anibal") and self.cache.has("pxqn"))
        return ga, gp

    def test_each_hard_lock_code_locks_every_item_and_drops_every_grant(self):
        for code in S.HARD_LOCK_CODES:
            with self.subTest(code=code):
                self.setUp()
                ga, gp = self.two_items()
                self.deny(code)
                r = self.client.sync(self.cache)
                self.assertEqual((r["hard_lock"], r["locked"]), (code, ["anibal", "preset-x", "pxqn"]))
                d = self.snapshot()
                self.assertEqual(d["grants"], {})
                self.assertEqual(d["locks"], {"anibal": code, "preset-x": code, "pxqn": code})   # what Core / the plugin host read
                self.assertEqual(d["hard_locks"], d["locks"])
                self.assertEqual(self.cache.active(), [])
                self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))
                v = S.entitlements_view(self.cache, S.load_snapshot(self.cfg))
                self.assertEqual((v["items"], v["grants"], v["hard_lock"]), ({}, [], code))
                self.assertEqual(set(v["locks"]), {"anibal", "preset-x", "pxqn"})
                self.assertIn(ga["statement"]["ent_id"], d["newest"])                       # anti-replay records stay

    def test_other_refusals_and_trouble_never_lock(self):
        self.two_items()
        before = self.snapshot()
        for status, code in ((403, "requires_item"), (403, "bad_signature"), (401, "revoked"), (404, "not_linked"),
                             (500, "account_disabled"), (429, "rate_limited"), (403, "")):
            self.deny(code, status)
            self.assertEqual(self.code(self.client.sync, self.cache)[0], status)
            after = self.snapshot()
            self.assertEqual((after["grants"], after["locks"], after["hard_locks"]), (before["grants"], before["locks"], {}), (status, code))
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (403, b"<html>Cloudflare</html>")     # not licd's JSON: store_error
        self.code(self.client.sync, self.cache)
        self.assertTrue(self.cache.has("anibal"))
        self.assertNotIn("hard_lock", S.load_snapshot(self.cfg))

    def test_a_403_on_the_nonce_step_locks_too(self):
        self.two_items()
        FakeLicd.script["/v1/store/nonce"] = lambda _b: (403, {"error": "disabled", "code": "account_disabled"})
        self.assertEqual(self.client.sync(self.cache)["hard_lock"], "account_disabled")
        self.assertEqual(self.cache.active(), [])

    def locked(self, code="revoked"):
        ga, gp = self.two_items()
        self.deny(code)
        self.client.sync(self.cache)
        return ga, gp

    def refused(self, g):
        with self.assertRaises(S.GrantError) as e:
            self.cache.add(g)
        return e.exception.code

    def test_the_lock_time_and_the_grant_ids_seen_are_recorded(self):
        t0 = time.time()
        ga, gp = self.locked()
        d = self.snapshot()
        self.assertTrue(int(t0) <= d["hard_lock_at"] <= time.time())      # licd's Date header (1 s resolution)
        self.assertEqual(d["hard_lock_clock"], "server")
        self.assertTrue({ga["statement"]["ent_id"], gp["statement"]["ent_id"]} <= set(d["hard_lock_seen"]))

    def test_a_second_403_moves_the_lock_time_later(self):
        self.locked()
        at1 = self.snapshot()["hard_lock_at"]
        FakeLicd.date_header = http_date(time.time() + 60)
        self.deny("not_linked")
        self.client.sync(self.cache)
        self.assertGreater(self.snapshot()["hard_lock_at"], at1)
        FakeLicd.date_header = http_date(time.time() - 3600)              # an earlier one never moves it back
        self.client.sync(self.cache)
        self.assertGreater(self.snapshot()["hard_lock_at"], at1)

    # ---- the lock time is licd's clock (PXANetwork ruling 2026-10-19 04:33)
    def lock_with(self, date=None, body_time=None):
        self.two_items()
        if date is not None:
            FakeLicd.date_header = date
        body = {"error": "no", "code": "revoked"}
        if body_time is not None:
            body["server_time"] = body_time
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (403, body)
        self.client.sync(self.cache)
        FakeLicd.date_header = None
        return self.snapshot()

    def unlocks(self, issued):
        try:
            self.cache.add(grant(self.cfg, "anibal", issued=int(issued)))
        except S.GrantError as e:
            self.assertEqual(e.code, "pre_lock")
            return False
        return self.cache.has("anibal")

    def test_server_date_ahead_of_the_local_clock(self):
        now = time.time()
        d = self.lock_with(http_date(now + 200))
        self.assertEqual((d["hard_lock_clock"], d["hard_lock_at"]), ("server", float(int(now + 200))))
        self.assertFalse(self.unlocks(now + 150))           # before licd's lock time (the local clock would have let it in)
        self.assertTrue(self.unlocks(now + 260))            # after it

    def test_server_date_behind_the_local_clock(self):
        now = time.time()
        d = self.lock_with(http_date(now - 3600))
        self.assertEqual((d["hard_lock_clock"], d["hard_lock_at"]), ("server", float(int(now - 3600))))
        self.assertFalse(self.unlocks(now - 4000))
        self.assertTrue(self.unlocks(now - 3000))           # after licd's lock time (the local clock would have refused it)

    def test_no_date_falls_back_to_local_minus_5_minutes(self):
        t0 = time.time()
        d = self.lock_with("")
        self.assertEqual(d["hard_lock_clock"], "local")
        self.assertTrue(t0 - 300 <= d["hard_lock_at"] <= time.time() - 300)
        self.assertFalse(self.unlocks(t0 - 400))
        self.assertTrue(self.unlocks(t0 - 200))

    def test_unparseable_date_falls_back_the_same_way(self):
        for bad in ("yesterday at noon", "Mon, 99 Foo 2026 25:61:61 GMT", "Thu, 01 Jan 1970 00:00:00 GMT"):
            with self.subTest(date=bad):
                self.setUp()
                t0 = time.time()
                d = self.lock_with(bad)
                self.assertEqual(d["hard_lock_clock"], "local")
                self.assertTrue(t0 - 300 <= d["hard_lock_at"] <= time.time() - 300)
                self.assertFalse(self.unlocks(t0 - 400))
                self.assertTrue(self.unlocks(t0 - 200))

    def test_a_server_time_in_the_403_body_wins_over_the_date_header(self):
        now = time.time()
        d = self.lock_with(http_date(now), body_time=int(now) - 1000)
        self.assertEqual((d["hard_lock_clock"], d["hard_lock_at"]), ("server", float(int(now) - 1000)))
        self.assertTrue(self.unlocks(now - 900))
        self.setUp()
        d = self.lock_with(http_date(now - 50), body_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 2000)))
        self.assertEqual(d["hard_lock_at"], float(int(now - 2000)))         # ISO 8601 too
        self.setUp()
        d = self.lock_with(http_date(now - 50), body_time="soon")          # unreadable body time: the Date header
        self.assertEqual((d["hard_lock_clock"], d["hard_lock_at"]), ("server", float(int(now - 50))))

    # ---- a locked computer: offline, garbled and malformed replies change nothing
    def test_locked_then_offline_stays_locked_with_no_state_change(self):
        self.locked()
        before = self.snapshot()
        dead = socket.socket()
        dead.bind(("127.0.0.1", 0))
        port = dead.getsockname()[1]
        dead.close()                                       # nobody listens there now
        off = S.LicdClient("http://127.0.0.1:%d" % port, self.cfg, pubkeys=KEYS, timeout=5)
        self.assertEqual(self.code(off.sync, self.cache), (503, "store_offline"))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.cache.active(), [])
        self.assertEqual(S.load_snapshot(self.cfg).get("hard_lock"), "revoked")

    def test_locked_then_malformed_or_garbled_stays_locked(self):
        self.locked()
        before = self.snapshot()
        g = grant(self.cfg, "anibal", issued=int(time.time()) + 2)
        for reply in ("garbage", "cut", (200, b"not json"), (200, b"[]"),
                      (200, {"grants": [g], "account": {"owned": [{"item_id": "anibal"}]}}),   # owned[] without status
                      (200, {"grants": {"x": g}, "account": {"owned": []}})):
            FakeLicd.script["/v1/store/refresh"] = lambda _b, reply=reply: reply
            self.assertEqual(self.code(self.client.sync, self.cache), (503, "store_offline" if reply in ("garbage", "cut")
                                                                       else "store_malformed"), reply)
            after = self.snapshot()
            after.pop("seen", None)                        # (has() below records the time it read the cache)
            self.assertEqual(after, {k: v for k, v in before.items() if k != "seen"}, reply)
            self.assertFalse(self.cache.has("anibal"), reply)

    def test_resending_the_old_grant_never_unlocks(self):
        ga, _gp = self.locked()
        self.assertEqual(self.refused(ga), "pre_lock")
        self.on_refresh(self.refresh_reply([ga]))                      # nor when licd's reply carries it
        self.assertEqual(self.client.sync(self.cache)["refused"], ["pre_lock"])
        self.assertFalse(self.cache.has("anibal"))

    def test_an_older_copy_of_the_old_grant_is_a_replay(self):
        ga, _gp = self.locked()
        older = grant(self.cfg, "anibal", issued=int(ga["statement"]["issued"]) - 3600, ent_id=ga["statement"]["ent_id"])
        self.assertEqual(self.refused(older), "replayed")
        self.assertFalse(self.cache.has("anibal"))

    # (v4, PXANetwork 05:18: licd re-signs the SAME ent_id on refresh; the old "seen id never unlocks" rule is gone)
    def test_the_old_grant_id_with_the_same_issued_never_unlocks(self):
        ga, _gp = self.locked()
        again = grant(self.cfg, "anibal", issued=int(ga["statement"]["issued"]), ent_id=ga["statement"]["ent_id"])
        self.assertEqual(self.refused(again), "pre_lock")
        self.assertFalse(self.cache.has("anibal"))

    def test_a_new_grant_id_issued_before_the_lock_never_unlocks(self):
        self.locked()
        self.assertEqual(self.refused(grant(self.cfg, "anibal", issued=int(time.time()) - 600)), "pre_lock")
        self.assertFalse(self.cache.has("anibal"))

    def test_a_refresh_marking_items_active_without_a_grant_never_unlocks(self):
        self.locked()
        self.on_refresh(self.refresh_reply([], owned=[{"item_id": "anibal", "status": "active"}, {"item_id": "pxqn", "status": "active"}]))
        r = self.client.sync(self.cache)
        self.assertEqual(r["locked"], [])
        self.assertNotIn("hard_lock", S.load_snapshot(self.cfg))
        self.assertEqual(self.cache.active(), [])
        self.assertEqual(set(self.cache.locks()), {"anibal", "preset-x", "pxqn"})

    def test_a_grant_that_does_not_verify_never_unlocks(self):
        self.locked()
        bad = json.loads(json.dumps(grant(self.cfg, "anibal", issued=int(time.time()) + 2)))
        bad["statement"]["level"] = "forged"
        self.on_refresh(self.refresh_reply([bad], owned=[{"item_id": "anibal", "status": "active"}]))
        self.assertTrue(self.client.sync(self.cache)["refused"])
        self.assertFalse(self.cache.has("anibal"))

    def test_a_revoked_fresh_grant_never_unlocks(self):
        self.locked()
        g = grant(self.cfg, "anibal", issued=int(time.time()) + 2)
        self.on_refresh(self.refresh_reply([g], revs=revocations([g["statement"]["ent_id"]])))
        self.client.sync(self.cache)
        self.assertFalse(self.cache.has("anibal"))

    def test_a_fresh_grant_unlocks_only_its_item(self):
        self.locked()
        self.on_refresh(self.refresh_reply([grant(self.cfg, "anibal", issued=int(time.time()) + 2)]))
        self.assertEqual(self.client.sync(self.cache)["refused"], [])
        self.assertTrue(self.cache.has("anibal"))
        self.assertFalse(self.cache.has("pxqn"))
        self.assertEqual(set(self.cache.locks()), {"preset-x", "pxqn"})
        self.assertIn("hard_lock_at", self.snapshot())                 # still in force for the rest

    def test_the_lock_record_goes_once_every_item_is_unlocked(self):
        self.locked()
        t = int(time.time()) + 2
        self.on_refresh(self.refresh_reply([grant(self.cfg, i, issued=t) for i in ("anibal", "pxqn", "preset-x")]))
        self.client.sync(self.cache)
        d = self.snapshot()
        self.assertEqual((d["hard_locks"], d["locks"]), ({}, {}))
        self.assertNotIn("hard_lock_at", d)
        self.assertNotIn("hard_lock_seen", d)

    def test_while_locked_an_old_grant_for_an_unlisted_item_is_refused_too(self):
        g_old = self.primed("other-item")
        self.cache.drop_item("other-item")                            # gone from the cache, its id still seen
        S.save_snapshot(self.cfg, {"owned": []}, ok=True)             # and no longer in the account answer
        self.deny("account_disabled")
        self.client.sync(self.cache)
        self.assertNotIn("other-item", self.cache.locks())
        self.assertEqual(self.refused(g_old), "pre_lock")

    def test_the_auto_refresh_in_the_entitlements_route_applies_the_lock(self):
        self.two_items()
        self.deny("not_linked")
        r = self.client.sync(self.cache)            # the same sync() r_store_entitlements / r_refresh call
        self.assertEqual(r["grants"], 0)
        self.assertEqual(S.entitlements_view(self.cache, S.load_snapshot(self.cfg))["items"], {})


class HardLockV4(HardLock403):
    """hard-lock v4 (PXANetwork 05:18 from licd store.py; Plugins confirmed + install_pub): while locked a grant that
    verifies for THIS machine and is not revoked counts when issued after hard_lock_at (licd's clock) and either its
    ent_id was known at lock time with a strictly newer issued than recorded then (J_LOCK_NEWEST), or the ent_id is new
    (a relink). Anti-replay = newest issued per ent_id (also journalled) + licd's server time. No nonces."""

    def renewed(self, g, issued):
        return grant(self.cfg, g["statement"]["item_id"], kind=g["statement"]["kind"], issued=int(issued),
                     ent_id=g["statement"]["ent_id"])

    def test_a_reenabled_account_same_ent_id_newer_issued_unlocks(self):
        ga, gp = self.locked()
        self.assertEqual(self.snapshot()[S.J_LOCK_NEWEST][ga["statement"]["ent_id"]], int(ga["statement"]["issued"]))
        time.sleep(1.1)                                             # licd's refresh after the account is re-enabled
        g2 = self.renewed(ga, time.time() + 1)
        self.on_refresh(self.refresh_reply([g2]))
        self.assertEqual(self.client.sync(self.cache)["refused"], [])
        self.assertTrue(self.cache.has("anibal"))
        self.assertNotIn("anibal", self.cache.locks())              # = Core: the item's lock is gone, not just masked
        self.assertFalse(self.cache.has("pxqn"))
        self.assertEqual(self.snapshot()["newest"][ga["statement"]["ent_id"]], int(g2["statement"]["issued"]))

    def test_a_reenabled_account_same_ent_id_equal_or_older_issued_stays_locked(self):
        """= Core b86d489152 test_a_reenabled_account_same_ent_id_equal_or_older_issued_stays_locked: after the account is
        re-enabled, the same ent_id re-signed with an equal issued (pre_lock) or an older one (replayed) unlocks nothing."""
        now = time.time()
        ga, _gp = self.two_items()
        FakeLicd.date_header = http_date(now - 3600)                # lock time an hour back: an older issue is still after it
        self.deny("revoked")
        self.client.sync(self.cache)
        FakeLicd.date_header = None
        self.assertEqual(self.refused(self.renewed(ga, ga["statement"]["issued"])), "pre_lock")          # equal
        self.assertEqual(self.refused(self.renewed(ga, ga["statement"]["issued"] - 60)), "replayed")     # older
        self.on_refresh(self.refresh_reply([self.renewed(ga, ga["statement"]["issued"])]))               # via a sync too
        self.assertEqual(self.client.sync(self.cache)["refused"], ["pre_lock"])
        self.assertFalse(self.cache.has("anibal"))
        self.assertEqual(self.cache.locks().get("anibal"), "revoked")

    def test_a_newer_issue_before_the_lock_time_is_refused(self):
        ga, _gp = self.two_items()
        FakeLicd.date_header = http_date(time.time() + 200)          # licd locked at now+200 (its clock)
        self.deny("revoked")
        self.client.sync(self.cache)
        FakeLicd.date_header = None
        self.assertEqual(self.refused(self.renewed(ga, ga["statement"]["issued"] + 100)), "pre_lock")

    def test_a_new_ent_id_after_relink_unlocks(self):
        self.locked()
        time.sleep(1.1)
        g = grant(self.cfg, "anibal", issued=int(time.time()) + 1)  # a new ent_id: licd issues one only after a relink
        self.on_refresh(self.refresh_reply([g]))
        self.assertEqual(self.client.sync(self.cache)["refused"], [])
        self.assertTrue(self.cache.has("anibal"))

    def test_a_revoked_grant_never_unlocks(self):
        ga, _gp = self.locked()
        self.cache.apply_revocations(revocations([ga["statement"]["ent_id"]]))
        time.sleep(1.1)
        self.assertEqual(self.refused(self.renewed(ga, time.time() + 1)), "revoked")     # even re-signed newer
        g = grant(self.cfg, "anibal", issued=int(time.time()) + 2)
        self.on_refresh(self.refresh_reply([g], revs=revocations([g["statement"]["ent_id"]])))
        self.client.sync(self.cache)                                 # a new id revoked in the same reply
        self.assertFalse(self.cache.has("anibal"))

    def test_a_replay_of_the_old_grant_is_refused(self):
        ga, gp = self.locked()
        for g in (ga, gp):
            self.assertEqual(self.refused(g), "pre_lock")
        self.on_refresh(self.refresh_reply([ga, gp]))
        self.assertEqual(self.client.sync(self.cache)["refused"], ["pre_lock", "pre_lock"])
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))

    def test_a_newer_issue_for_another_machine_is_refused(self):
        ga, _gp = self.locked()
        other = tempfile.mkdtemp(dir=TMP)
        g = grant(other, "anibal", issued=int(time.time()) + 2, ent_id=ga["statement"]["ent_id"])   # another install_pub
        self.assertEqual(self.refused(g), "wrong_machine")
        self.assertFalse(self.cache.has("anibal"))

    def test_the_lock_time_newest_survives_a_corrupt_file(self):
        ga, _gp = self.locked()
        j = json.load(open(os.path.join(self.cfg, S.EntitlementCache.JOURNALS[0])))
        self.assertEqual(j[S.J_LOCK_NEWEST][ga["statement"]["ent_id"]], int(ga["statement"]["issued"]))
        self.assertEqual(j[S.J_NEWEST][ga["statement"]["ent_id"]], int(ga["statement"]["issued"]))
        self.assertGreater(j[S.J_LICD_SEEN], 0)
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        self.assertEqual(self.refused(self.renewed(ga, ga["statement"]["issued"])), "pre_lock")
        self.assertEqual(self.refused(self.renewed(ga, ga["statement"]["issued"] - 60)), "replayed")   # newest restored
        time.sleep(1.1)
        self.cache.add(self.renewed(ga, time.time() + 1))
        self.assertTrue(self.cache.has("anibal"))

    def test_a_missing_lock_time_newest_record_fails_closed_for_known_ids(self):
        ga, _gp = self.locked()
        d = self.snapshot()
        d.pop(S.J_LOCK_NEWEST)                                      # a lock recorded before v4 (or the record stripped)
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        for name in S.EntitlementCache.JOURNALS:
            p = os.path.join(self.cfg, name)
            j = json.load(open(p))
            j.pop(S.J_LOCK_NEWEST, None)
            with open(p, "w") as f:
                json.dump(j, f)
        time.sleep(1.1)
        self.assertEqual(self.refused(self.renewed(ga, time.time() + 1)), "pre_lock")   # known id: no proof it is newer
        self.cache.add(grant(self.cfg, "anibal", issued=int(time.time()) + 2))         # a new id (relink) still unlocks
        self.assertTrue(self.cache.has("anibal"))

    def test_corrupt_clock_fast_locks_on_licd_time(self):
        """QA F310 corrupt_clock_fast on Control: a computer whose clock runs a month ahead finds grants.json corrupt; the
        corruption lock is on licd's clock (journal licd_seen), so licd's fresh grant is accepted."""
        ga, _gp = self.two_items()
        real = time.time()
        t = [real + 30 * 86400]
        fast = S.EntitlementCache(self.cfg, pubkeys=KEYS, clock=lambda: t[0])
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        fast.locks()
        d = self.snapshot()
        self.assertEqual(d["hard_lock_clock"], "server")
        self.assertLess(d["hard_lock_at"], real + 60)
        fast.add(self.renewed(ga, real + 60))                       # licd's refresh of the same ent_id, on licd's clock
        g = grant(self.cfg, "pxqn", kind="quant-tier", issued=int(real) + 61)
        fast.add(g)
        self.assertTrue(fast.has("anibal") and fast.has("pxqn"))
        with self.assertRaises(S.GrantError) as e:
            fast.add(ga)                                            # the pre-corruption copy is still refused
        self.assertEqual(e.exception.code, "replayed")             # (newest per ent_id, restored from the journal)

    def test_without_licd_time_on_record_a_corruption_locks_at_the_local_clock(self):
        g = grant(self.cfg, "anibal")
        self.cache.add(g)                                           # no reply, so no licd time on record
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        self.cache.locks()
        self.assertEqual(self.snapshot()["hard_lock_clock"], "local")


# ------------------------------------------- RC3.6 item 4, the Control halves (anibal-finish CONTRACT-CONTROL-HALVES.md)
def _hold_lock(path, ready, release_after):
    """a child process that holds grants.json.lock for release_after seconds (a stopped / hung writer)."""
    with S.file_lock(path):
        ready.set()
        time.sleep(release_after)


def _stale_writer(path, cfg, ready, pause):
    """another writer (a plugin host mid-transaction): takes the lock, loads grants.json and the journal, sleeps, then
    writes back what it loaded with gen + 1, as a writer that read the journal before a 403 landed would."""
    with S.file_lock(path):
        with open(path) as f:
            d = json.load(f)
        jp = [os.path.join(cfg, n) for n in S.EntitlementCache.JOURNALS]
        js = [json.load(open(p)) for p in jp]
        ready.set()
        time.sleep(pause)
        d["gen"] = int(d.get("gen") or 0) + 1
        S.write_atomic(path, json.dumps(d).encode())
        for p, j in zip(jp, js):
            j["gen"] = d["gen"]
            S.write_atomic(p, json.dumps(j).encode())


class Item4Control(Base):
    """RC3.6 item 4 (store hard-lock journal hardening), the Control halves: shared journal keys and merge rules, F311,
    F312, F314, F317, F318, F310 (b), the hard_lock_source enum and the process-only "_" keys. = Plugins plugins/item4-rc36
    0d675a4b51 (store_backend.GrantCache) and Core b86d489152 (anibal/entitlement.GrantCache)."""
    deny, two_items, locked, refused = HardLock403.deny, HardLock403.two_items, HardLock403.locked, HardLock403.refused

    def journals(self):
        out = []
        for n in S.EntitlementCache.JOURNALS:
            with open(os.path.join(self.cfg, n)) as f:
                out.append(json.load(f))
        return out

    def file_bytes(self):
        with open(self.cache.path, "rb") as f:
            return f.read()

    def mp(self):
        import multiprocessing
        return multiprocessing.get_context("fork")

    # ---- Low notes: process-only "_" keys, install.key never through a symlink
    def test_underscore_keys_are_stripped_before_every_write_and_on_every_read(self):
        self.two_items()
        d = self.snapshot()
        d.update(_star_cleared=True, _unsaved=True, _seen_reset=True, _rebuilt=True)
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        for p in [os.path.join(self.cfg, n) for n in S.EntitlementCache.JOURNALS]:
            j = json.load(open(p))
            j.update(star=True, _star_cleared=True)
            with open(p, "w") as f:
                json.dump(j, f)
        self.assertEqual(self.cache.locks().get("*"), "corrupt")    # a planted _star_cleared never suppresses the "*"
        self.cache.set_locks({"x": "suspended"})                    # a planted _unsaved never suppresses a save
        d = self.snapshot()
        self.assertEqual(d["locks"].get("x"), "suspended")
        self.assertEqual([k for k in d if k.startswith("_")], [])
        for j in self.journals():
            self.assertEqual([k for k in j if k.startswith("_")], [])

    @unittest.skipIf(not hasattr(os, "O_NOFOLLOW"), "POSIX O_NOFOLLOW")
    def test_read_seed_never_follows_a_symlinked_install_key(self):
        d = os.path.join(self.cfg, "plugins")
        os.makedirs(d, exist_ok=True)
        target = os.path.join(self.cfg, "planted.key")
        with open(target, "wb") as f:
            f.write(b"k" * 32)
        os.symlink(target, os.path.join(d, "install.key"))
        self.assertIsNone(S._read_seed(os.path.join(d, "install.key")))
        seed, pub = S.install_key(self.cfg)
        self.assertNotEqual(seed, b"k" * 32)
        self.assertFalse(os.path.islink(os.path.join(d, "install.key")))
        self.assertEqual(open(target, "rb").read(), b"k" * 32)       # the link's target is never touched

    # ---- F318: `revoked` in both journals, grow-only, merged before any sync result, kept by every rebuild
    def test_f318_revoked_ids_survive_a_corruption_and_a_sync_that_clears_the_star(self):
        ga, gp = self.two_items()
        eid = ga["statement"]["ent_id"]
        self.cache.apply_revocations(revocations([eid]))
        for j in self.journals():
            self.assertIn(eid, j.get("revoked") or [])
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        self.assertEqual(self.cache.locks().get("*"), "corrupt")
        self.assertIn(eid, self.snapshot()["revoked"])              # the rebuild keeps it
        time.sleep(1.1)
        renewed = grant(self.cfg, "anibal", issued=int(time.time()) + 1, ent_id=eid)
        fresh = grant(self.cfg, "pxqn", kind="quant-tier", issued=int(time.time()) + 1)
        self.on_refresh(self.refresh_reply([renewed, fresh]))
        self.assertEqual(self.client.sync(self.cache)["refused"], ["revoked"])
        self.assertTrue(self.cache.has("pxqn"))
        # another writer (the plugin host's online_ok) clears "*" and writes a file that lost the revoked list
        d = self.snapshot()
        d["hard_locks"].pop("*", None)
        d["revoked"], d["gen"] = [], int(d["gen"]) + 1
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        for p in [os.path.join(self.cfg, n) for n in S.EntitlementCache.JOURNALS]:
            j = json.load(open(p))
            j.update(star=False, gen=d["gen"])
            with open(p, "w") as f:
                json.dump(j, f)
        self.assertEqual(self.refused(ga), "revoked")
        self.assertEqual(self.refused(renewed), "revoked")
        self.on_refresh(self.refresh_reply([renewed]))
        self.assertEqual(self.client.sync(self.cache)["refused"], ["revoked"])
        self.assertFalse(self.cache.has("anibal"))
        self.assertIn(eid, self.snapshot()["revoked"])
        for j in self.journals():
            self.assertIn(eid, j["revoked"])

    # ---- F317: the journal may only add; a rebuild takes the stricter value of the file and the journal
    def test_f317_a_forged_higher_gen_journal_cannot_lower_an_intact_lock(self):
        ga, _gp = self.two_items()
        now = int(time.time())
        FakeLicd.date_header = http_date(now)
        self.deny("revoked")
        self.client.sync(self.cache)
        FakeLicd.date_header = None
        locked = self.snapshot()
        self.assertGreaterEqual(locked["hard_lock_at"], now - 1)
        for p in [os.path.join(self.cfg, n) for n in S.EntitlementCache.JOURNALS]:
            j = json.load(open(p))                                   # grants.json intact; only the journal is edited
            j.update({"licd_high_water": 1.6e9, "hard_lock_at": 1.6e9, "gen": int(j["gen"]) + 5, "hard_locks": {},
                      "hard_lock_seen": [], "ids": [], "newest": {}, "hard_lock_newest": {}, "seen": 0})
            with open(p, "w") as f:
                json.dump(j, f)
        pre = grant(self.cfg, "anibal", issued=now - 60)            # captured before the 403, new ent_id, never added
        self.assertEqual(self.refused(pre), "pre_lock")
        self.assertEqual(self.refused(ga), "pre_lock")              # the file's seen ids / hard_lock_newest are kept
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))
        d = self.snapshot()
        self.assertGreaterEqual(d["hard_lock_at"], locked["hard_lock_at"])
        self.assertGreaterEqual(d["licd_high_water"], locked["licd_high_water"])
        self.assertEqual({i: d["hard_locks"][i] for i in ("anibal", "pxqn")}, {"anibal": "revoked", "pxqn": "revoked"})
        self.assertIn(ga["statement"]["ent_id"], d["hard_lock_seen"])
        self.assertEqual(d["hard_lock_newest"][ga["statement"]["ent_id"]], ga["statement"]["issued"])
        self.assertEqual(d["corrupt_why"], "rolled_back")

    # ---- F314: a bad or symlinked journal never throws away an intact grants.json lock record
    def test_f314_a_bad_or_symlinked_journal_keeps_an_intact_lock_record(self):
        for how in ("corrupt", "symlink"):
            with self.subTest(how=how):
                self.setUp()
                ga, _gp = self.two_items()
                FakeLicd.date_header = http_date(time.time())
                self.deny("revoked")
                self.client.sync(self.cache)
                FakeLicd.date_header = None
                locked = self.snapshot()
                planted = []
                for n in S.EntitlementCache.JOURNALS:
                    p = os.path.join(self.cfg, n)
                    if how == "corrupt":
                        with open(p, "w") as f:
                            f.write("{garbage")
                    else:                                   # a symlink to an empty, valid-looking journal
                        with open(p + ".x", "w") as f:
                            json.dump({"v": 1, "gen": 10 ** 6, "hard_locks": {}}, f)
                        planted.append(p + ".x")
                        os.unlink(p)
                        os.symlink(p + ".x", p)
                t = [ga["statement"]["issued"] - 299.0]     # clock set back just inside the old grant's skew
                back = S.EntitlementCache(self.cfg, pubkeys=KEYS, clock=lambda: t[0])
                with self.assertRaises(S.GrantError) as e:
                    back.add(ga)
                self.assertEqual(e.exception.code, "pre_lock")
                self.assertFalse(back.has("anibal") or self.cache.has("pxqn"))
                d = self.snapshot()
                self.assertEqual(d["hard_locks"].get("*"), "corrupt")
                self.assertEqual({i: d["hard_locks"].get(i) for i in ("anibal", "pxqn")}, {"anibal": "revoked", "pxqn": "revoked"})
                self.assertGreaterEqual(d["hard_lock_at"], locked["hard_lock_at"])
                self.assertIn(ga["statement"]["ent_id"], d["hard_lock_seen"])
                self.assertEqual(d["corrupt_why"], "journal")
                store = os.listdir(os.path.join(self.cfg, "store"))
                self.assertIn("grants.json", store)            # the intact file is never moved aside
                self.assertTrue(any(n.startswith("grants.journal.json.corrupt-") for n in store), store)
                for n in S.EntitlementCache.JOURNALS:          # rebuilt as regular files from the file's record
                    p = os.path.join(self.cfg, n)
                    self.assertFalse(os.path.islink(p))
                    j = json.load(open(p))
                    self.assertTrue(j["star"])
                    self.assertEqual(set(j["hard_locks"]) >= {"anibal", "pxqn"}, True)
                    self.assertIn(ga["statement"]["ent_id"], j["hard_lock_seen"])
                for x in planted:                              # a link's target is never written
                    self.assertEqual(json.load(open(x)), {"v": 1, "gen": 10 ** 6, "hard_locks": {}})

    # ---- F312: Windows sharing violation on the replace: journal first, retry up to 5 s, then StoreBusy
    def _busy_replace(self, fails):
        real, calls, n = os.replace, [], [0]

        def busy(src, dst, *a, **kw):
            calls.append((os.path.basename(dst), time.monotonic()))
            if dst == self.cache.path and (fails is None or n[0] < fails):
                n[0] += 1
                raise PermissionError(13, "The process cannot access the file because it is being used by another process", dst)
            return real(src, dst, *a, **kw)
        return real, busy, calls, n

    def test_f312_replace_permission_error_retries_5s_after_the_journal(self):
        # a few sharing violations: retried, the 403 lands in the file; the journal was written before the first try
        self.two_items()
        real, busy, calls, n = self._busy_replace(3)
        os.replace = busy
        try:
            self.cache.hard_lock("revoked")
        finally:
            os.replace = real
        names = [c[0] for c in calls]
        self.assertEqual(n[0], 3)
        self.assertLess(max(names.index(j) for j in ("grants.journal.json", "store-lock-journal.json")),
                        names.index("grants.json"), names)
        self.assertEqual(self.snapshot()["hard_locks"], {"anibal": "revoked", "pxqn": "revoked"})
        # the file stays busy: gives up after 5 s with StoreBusy; grants.json unchanged, the lock is in the journal
        self.setUp()
        self.two_items()
        before = self.file_bytes()
        real, busy, calls, n = self._busy_replace(None)
        os.replace = busy
        t0 = time.monotonic()
        try:
            with self.assertRaises(S.StoreBusy):
                self.cache.hard_lock("revoked")
        finally:
            os.replace = real
        took = time.monotonic() - t0
        self.assertGreaterEqual(took, S.REPLACE_RETRY_S - 0.05)
        self.assertLess(took, S.REPLACE_RETRY_S + 2)
        self.assertEqual(S.REPLACE_RETRY_S, 5.0)                      # = Core
        first = [c[1] for c in calls if c[0] == "grants.json"][0]
        self.assertTrue(all(c[1] <= first for c in calls if c[0] != "grants.json"))
        self.assertEqual(self.file_bytes(), before)
        for j in self.journals():
            self.assertEqual(set(j["hard_locks"]), {"anibal", "pxqn"})
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))
        self.assertEqual(set(self.cache.locks()) >= {"anibal", "pxqn"}, True)

    # ---- F311: past LOCK_TIMEOUT_S Control never writes grants.json without the lock: journal first, then StoreBusy
    def _timeouts(self, wait, retry):
        saved = (S.LOCK_TIMEOUT_S, getattr(S, "LOCK_RETRY_S", None))
        S.LOCK_TIMEOUT_S, S.LOCK_RETRY_S = wait, retry
        self.addCleanup(lambda: (setattr(S, "LOCK_TIMEOUT_S", saved[0]), setattr(S, "LOCK_RETRY_S", saved[1])))

    def test_f311_lock_timeout_writes_the_journal_only_and_never_grants_json(self):
        self.two_items()
        before = self.file_bytes()
        ctx = self.mp()
        ready = ctx.Event()
        p = ctx.Process(target=_hold_lock, args=(self.cache.path, ready, 4.0))
        p.start()
        self.addCleanup(p.join, 20)
        self.assertTrue(ready.wait(10))
        self._timeouts(0.3, 0.3)
        written = []
        real = S.write_atomic
        S.write_atomic = lambda path, data: (written.append(os.path.basename(path)), real(path, data))[1]
        try:
            with self.assertRaises(S.StoreBusy):
                self.cache.hard_lock("revoked")
        finally:
            S.write_atomic = real
        self.assertTrue(p.is_alive())                                # the holder still has it: nothing waited it out
        self.assertNotIn("grants.json", written)
        self.assertEqual(self.file_bytes(), before)                  # grants.json bytes unchanged
        for j in self.journals():                                    # the lock is in the journal (fail closed)
            self.assertEqual(j["hard_locks"], {"anibal": "revoked", "pxqn": "revoked"})
            self.assertGreater(j["hard_lock_at"], 0)
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))   # a read while busy: locked, writes nothing
        self.assertEqual(self.file_bytes(), before)
        p.join(20)
        self.cache.set_locks({})                                     # the next write under the lock persists it
        d = self.snapshot()
        self.assertEqual({i: d["hard_locks"].get(i) for i in ("anibal", "pxqn")}, {"anibal": "revoked", "pxqn": "revoked"})
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))

    def test_f311_a_writer_holding_the_lock_past_the_timeout_cannot_lose_the_403(self):
        """QA proc_plugins_write_vs_ctl_lock_timeout: another writer that loaded before the 403 holds the lock past
        LOCK_TIMEOUT_S and then writes back what it loaded (overwriting the journal too); Control's second wait
        (LOCK_RETRY_S) takes the lock after it and writes the 403 under the lock."""
        self.two_items()
        ctx = self.mp()
        ready = ctx.Event()
        p = ctx.Process(target=_stale_writer, args=(self.cache.path, self.cfg, ready, 1.5))
        p.start()
        self.addCleanup(p.join, 20)
        self.assertTrue(ready.wait(10))
        self._timeouts(0.3, 5.0)
        t0 = time.monotonic()
        self.assertEqual(self.cache.hard_lock("revoked"), ["anibal", "pxqn"])
        self.assertGreater(time.monotonic() - t0, 1.0)
        p.join(20)
        self.assertEqual(p.exitcode, 0)
        d = self.snapshot()
        self.assertEqual({i: d["hard_locks"].get(i) for i in ("anibal", "pxqn")}, {"anibal": "revoked", "pxqn": "revoked"})
        self.assertEqual(d["grants"], {})
        for j in self.journals():
            self.assertEqual(set(j["hard_locks"]), {"anibal", "pxqn"})
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))

    def test_f311_reads_and_revocations_while_busy_never_write_grants_json(self):
        ga, _gp = self.two_items()
        before = self.file_bytes()
        ctx = self.mp()
        ready = ctx.Event()
        p = ctx.Process(target=_hold_lock, args=(self.cache.path, ready, 4.0))
        p.start()
        self.addCleanup(p.join, 20)
        self.assertTrue(ready.wait(10))
        self._timeouts(0.3, 0.3)
        with open(self.cache.path, "w") as f:                       # even a corrupt file is not moved or rewritten
            f.write("{garbage")
        self.assertEqual(self.cache.locks().get("*"), "corrupt")     # fail closed in memory
        self.assertEqual(open(self.cache.path).read(), "{garbage")
        self.assertFalse([n for n in os.listdir(os.path.join(self.cfg, "store")) if ".corrupt-" in n])
        with open(self.cache.path, "wb") as f:
            f.write(before)
        with self.assertRaises(S.StoreBusy):
            self.cache.apply_revocations(revocations([ga["statement"]["ent_id"]]))
        with self.assertRaises(S.StoreBusy):
            self.cache.drop_item("anibal")
        self.assertEqual(self.file_bytes(), before)
        for j in self.journals():                                    # the revocation is journalled first
            self.assertIn(ga["statement"]["ent_id"], j["revoked"])
        self.assertFalse(self.cache.has("anibal"))
        self.assertTrue(self.cache.has("pxqn"))

    # ---- F310 (b): only a verified licd grant moves a fast `seen` back, never below licd_high_water, never an unlock
    def _fast_seen(self, days=30):
        t = [time.time() + days * 86400]
        S.EntitlementCache(self.cfg, pubkeys=KEYS, clock=lambda: t[0]).set_locks({})     # a write on the fast clock
        self.assertGreater(self.snapshot()["seen"], time.time() + (days - 1) * 86400)
        return self.snapshot()["seen"]

    def test_f310b_only_a_verified_licd_grant_moves_a_fast_seen_back(self):
        self.locked()                                               # a 403 is in force: the reset must not lift it
        fast = self._fast_seen()
        self.assertEqual(self.cache.active(), [])                   # the clock is fixed: every read is a rollback
        real = time.time()
        self.on_refresh({"grants": [], "account": {"owned": [], "credits": 0}, "server_time": int(real)})
        self.client.sync(self.cache)                                # an unsigned server_time never moves it
        self.assertEqual(self.snapshot()["seen"], fast)
        bad = grant(self.cfg, "anibal", issued=int(real) + 1)
        bad["signature"] = bad["signature"][:-4] + ("AAAA" if not bad["signature"].endswith("AAAA") else "BBBB")
        self.on_refresh(self.refresh_reply([bad]))
        self.assertEqual(self.client.sync(self.cache)["refused"], ["clock_rollback"])   # a bad signature never does
        self.assertEqual(self.snapshot()["seen"], fast)
        time.sleep(1.1)
        good = grant(self.cfg, "anibal", issued=int(time.time()) + 1)   # licd's fresh grant (a relink: new ent_id)
        self.on_refresh(self.refresh_reply([good]))
        self.assertEqual(self.client.sync(self.cache)["refused"], [])
        d = self.snapshot()
        self.assertLess(d["seen"], time.time() + 60)                 # moved back to licd's signed time
        for j in self.journals():
            self.assertLess(j["seen"], time.time() + 60)
        self.assertTrue(self.cache.has("anibal"))                   # the fresh grant unlocks its own item ...
        self.assertFalse(self.cache.has("pxqn"))                    # ... the reset unlocks nothing by itself
        self.assertEqual(self.cache.locks().get("pxqn"), "revoked")

    def test_f310b_the_reset_never_goes_below_licd_high_water(self):
        self.two_items()
        now = time.time()
        hw = float(int(now) + 20 * 3600)                            # licd's time on record, 20 h ahead of this clock
        self.cache.add(grant(self.cfg, "preset-x", kind="preset", issued=int(now)), server_now=hw)
        self.assertEqual(self.snapshot()["licd_high_water"], hw)
        self._fast_seen()
        time.sleep(1.1)
        g = grant(self.cfg, "fleet-view", kind="feature", issued=int(time.time()) + 1)
        self.cache.add(g)                                           # verified, newer: seen goes back, to hw at most
        d = self.snapshot()
        self.assertEqual(d["seen"], hw)
        for j in self.journals():
            self.assertEqual(j["seen"], hw)
        self.assertTrue(self.cache.has("fleet-view") and self.cache.has("anibal"))
        with self.assertRaises(S.GrantError) as e:                  # an older copy never resets anything
            self.cache.add(grant(self.cfg, "fleet-view", kind="feature", issued=int(now) - 10,
                                 ent_id=g["statement"]["ent_id"]))
        self.assertEqual(e.exception.code, "replayed")
        self.assertEqual(self.snapshot()["seen"], hw)

    # ---- hard_lock_source: body | date | local | licd_high_water | licd_reset; legacy "server" reads as body
    def test_hard_lock_source_legacy_server_reads_as_body_and_writes_enum(self):
        self.locked()
        for legacy, want in (("server", "body"), ("server_time", "local"), ("body", "body"), ("date", "date"),
                             ("licd_high_water", "licd_high_water"), ("licd_reset", "licd_reset")):
            with self.subTest(legacy=legacy):
                d = self.snapshot()
                d["hard_lock_source"] = legacy
                with open(self.cache.path, "w") as f:
                    json.dump(d, f)
                self.assertEqual(self.cache.status()["state"], "hard_locked")   # a legacy value is no corruption
                self.cache.set_locks({})                                         # read, then written back
                self.assertEqual(self.snapshot()["hard_lock_source"], want)
        self.assertEqual(S.HARD_LOCK_SOURCES, ("body", "date", "local", "licd_high_water", "licd_reset"))
        self.setUp()
        self.two_items()
        self.cache.hard_lock("revoked", at=time.time(), source="server")         # a caller's legacy name -> enum
        self.assertEqual(self.snapshot()["hard_lock_source"], "body")
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        self.cache.locks()                                                       # a rebuild: licd_high_water | local
        self.assertIn(self.snapshot()["hard_lock_source"], ("licd_high_water", "local"))
        for v in ({"x": 1}, 7):                                                  # not a string: the wrong shape
            with self.subTest(bad=v):
                d = self.snapshot()
                d["hard_lock_source"] = v
                with open(self.cache.path, "w") as f:
                    json.dump(d, f)
                self.assertEqual(self.cache.status()["state"], "corrupt")


class Item4ReviewFixes(Base):
    """the fix pass on the item 4 review (the item 4 review notes): licd_high_water grow-only (A, Minor 5),
    non-finite numbers (B), a 403 kept in memory after StoreBusy (Minor 1), one replace deadline per save (Minor 2), the
    busy path journals only the change (Minor 3)."""
    deny, two_items, locked, refused = HardLock403.deny, HardLock403.two_items, HardLock403.locked, HardLock403.refused
    journals, file_bytes, mp, _timeouts = Item4Control.journals, Item4Control.file_bytes, Item4Control.mp, Item4Control._timeouts
    DAY = 86400.0

    def at(self, clock):
        return S.EntitlementCache(self.cfg, pubkeys=KEYS, clock=lambda: clock[0])

    def jhw(self):
        return [j.get("licd_high_water") for j in self.journals()]

    def hw_cache(self, now, issued):
        """a cache whose licd time on record is `now` (a reply's server_time), holding one grant."""
        clock = [now]
        c = self.at(clock)
        c.add(grant(self.cfg, "anibal", issued=int(issued)), server_now=now)
        self.assertEqual(self.jhw(), [now, now])
        return clock, c

    # ---- A + Minor 5: licd_high_water only grows, whatever the local clock says
    def test_licd_high_water_stays_grow_only_under_a_clock_set_back(self):
        now = float(int(time.time()))
        # A: a corruption rebuilt under a clock set back 30 days, then the clock fixed and a save
        clock, c = self.hw_cache(now, now - 10)
        clock[0] = now - 30 * self.DAY
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        c.locks()
        self.assertEqual(self.jhw(), [now, now])
        clock[0] = now
        c.set_locks({})
        self.assertEqual(self.jhw(), [now, now])
        # B: grants.json lost its licd_high_water (an old writer / a one-file edit, same gen); a save, clock 30 d back
        self.setUp()
        clock, c = self.hw_cache(now, now - 40 * self.DAY)
        d = self.snapshot()
        d.pop("licd_high_water")
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        clock[0] = now - 30 * self.DAY
        c.set_locks({})
        self.assertEqual(self.jhw(), [now, now])
        self.assertEqual(self.snapshot()["licd_high_water"], now)
        # C: as B, then one verified captured grant (new ent_id) issued at the set-back time: the F310 (b) floor holds
        self.setUp()
        clock, c = self.hw_cache(now, now - 40 * self.DAY)
        d = self.snapshot()
        d.pop("licd_high_water")
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        clock[0] = now - 30 * self.DAY
        with self.assertRaises(S.GrantError) as e:
            c.add(grant(self.cfg, "pxqn", kind="quant-tier", issued=int(now - 30 * self.DAY - 100)))
        self.assertEqual(e.exception.code, "clock_rollback")
        self.assertEqual(self.jhw(), [now, now])
        self.assertGreaterEqual(self.snapshot()["seen"], now)

    def test_a_rebuild_with_both_journals_bad_stores_licd_high_water_uncapped(self):
        """Minor 5 (review/probe_hw_d.py): both journals bad and the clock 30 d back: the rebuild keeps the file's
        licd_high_water as it is (the lock time is capped, the stored high water is not)."""
        now = float(int(time.time()))
        clock, c = self.hw_cache(now, now - 10)
        for n in S.EntitlementCache.JOURNALS:
            with open(os.path.join(self.cfg, n), "w") as f:
                f.write("{garbage")
        clock[0] = now - 30 * self.DAY
        c.locks()
        d = self.snapshot()
        self.assertEqual(d["licd_high_water"], now)
        self.assertLessEqual(d["hard_lock_at"], clock[0] + S.LOCK_AHEAD_CAP_S)     # the lock time stays capped
        self.assertEqual(self.jhw(), [now, now])

    # ---- B: a non-finite number (NaN, inf) in grants.json or a journal is the wrong shape, never a weaker lock
    def test_a_non_finite_hard_lock_at_in_grants_json_never_unlocks_a_pre_403_grant(self):
        for bad in (float("nan"), float("-inf"), float("inf")):
            with self.subTest(bad=bad):
                self.setUp()
                now = float(int(time.time()))
                clock = [now]
                c = self.at(clock)
                c.add(grant(self.cfg, "anibal", issued=int(now - 3600)))
                pre = grant(self.cfg, "anibal", issued=int(now - 600))   # signed before the 403, captured, never added
                clock[0] = now + 60
                c.hard_lock("revoked", ["anibal"], at=now, source="body")
                with self.assertRaises(S.GrantError) as e:
                    c.add(pre)
                self.assertEqual(e.exception.code, "pre_lock")
                d = self.snapshot()
                d["hard_lock_at"] = bad                                   # a same-gen edit of grants.json only
                with open(self.cache.path, "w") as f:
                    f.write(json.dumps(d))
                with self.assertRaises(S.GrantError) as e:
                    c.add(pre)
                self.assertEqual(e.exception.code, "pre_lock")
                self.assertFalse(c.has("anibal"))
                for j in self.journals():
                    self.assertTrue(math.isfinite(j["hard_lock_at"]))
        self.assertFalse(S.EntitlementCache._fresh({"issued": 10 ** 9, "ent_id": "x"}, float("nan"), {}, set()))
        self.assertIsNone(S._num(float("nan")))
        self.assertIsNone(S._num(float("inf")))

    def test_a_non_finite_seen_never_switches_off_the_rollback_guard(self):
        now = float(int(time.time()))
        clock = [now]
        c = self.at(clock)
        c.add(grant(self.cfg, "anibal", issued=int(now - 40 * self.DAY)))
        c.set_locks({})
        clock[0] = now - 30 * self.DAY
        self.assertEqual(c.active(), [])
        for bad in (float("nan"), float("-inf")):
            with self.subTest(bad=bad):
                d = self.snapshot()
                d["seen"] = bad
                with open(self.cache.path, "w") as f:
                    f.write(json.dumps(d))
                self.assertEqual(c.active(), [])
                with self.assertRaises(S.GrantError):
                    c.add(grant(self.cfg, "pxqn", kind="quant-tier", issued=int(now - 31 * self.DAY)))
                c.set_locks({})
                for j in self.journals():                             # the anchor is never poisoned
                    self.assertTrue(math.isfinite(j["seen"]) and j["seen"] >= now, j["seen"])
                self.assertTrue(math.isfinite(self.snapshot()["seen"]))

    def test_a_non_finite_value_in_a_journal_is_a_bad_journal(self):
        for key, bad in (("hard_lock_at", float("nan")), ("seen", float("nan")), ("licd_high_water", float("inf")),
                         ("newest", "nan-map"), ("hard_lock_newest", "nan-map")):
            with self.subTest(key=key):
                self.setUp()
                ga, _gp = self.locked()
                for n in S.EntitlementCache.JOURNALS:
                    p = os.path.join(self.cfg, n)
                    j = json.load(open(p))
                    j[key] = {ga["statement"]["ent_id"]: float("nan")} if bad == "nan-map" else bad
                    with open(p, "w") as f:
                        f.write(json.dumps(j))
                self.assertEqual(self.refused(ga), "pre_lock")
                d = self.snapshot()
                self.assertEqual(d["corrupt_why"], "journal")             # the F314 way: the file's lock record kept
                self.assertEqual(d["hard_locks"].get("anibal"), "revoked")
                for j in self.journals():
                    self.assertNotIn("NaN", json.dumps(j))
                    self.assertNotIn("Infinity", json.dumps(j))

    # ---- Minor 2: the 5 s replace budget is per save (contract: 5 s TOTAL), not per file
    def _all_busy(self):
        real = os.replace
        root = os.path.realpath(self.cfg)

        def busy(src, dst, *a, **kw):
            if os.path.realpath(dst).startswith(root):
                raise PermissionError(13, "The process cannot access the file because it is being used by another process", dst)
            return real(src, dst, *a, **kw)
        os.replace = busy
        self.addCleanup(setattr, os, "replace", real)

    def test_f312_one_replace_deadline_per_save_bounds_a_busy_hard_lock(self):
        self.two_items()
        self._all_busy()                                   # every replace in the config dir stays busy
        t0 = time.monotonic()
        with self.assertRaises(S.StoreBusy):
            self.cache.hard_lock("revoked")
        took = time.monotonic() - t0
        self.assertLess(took, S.REPLACE_RETRY_S + 1.5, took)   # 2 journals + grants.json + 2 journals share 5 s
        self.assertGreaterEqual(took, S.REPLACE_RETRY_S - 0.1)
        # busy store lock too: LOCK_TIMEOUT_S + the journal-only write + LOCK_RETRY_S share one replace budget
        ctx = self.mp()
        ready = ctx.Event()
        p = ctx.Process(target=_hold_lock, args=(self.cache.path, ready, 20.0))
        p.start()
        self.addCleanup(p.terminate)
        self.assertTrue(ready.wait(10))
        self._timeouts(0.3, 0.3)
        t0 = time.monotonic()
        with self.assertRaises(S.StoreBusy):
            self.cache.hard_lock("revoked")
        took = time.monotonic() - t0
        self.assertLess(took, 0.3 + 0.3 + S.REPLACE_RETRY_S + 1.5, took)

    # ---- Minor 3: on the busy path only the change itself is journalled; a racing read never writes "*"
    def test_f311_busy_path_journals_only_the_change_never_a_false_star(self):
        self.two_items()
        ctx = self.mp()
        ready = ctx.Event()
        p = ctx.Process(target=_hold_lock, args=(self.cache.path, ready, 3.0))
        p.start()
        self.addCleanup(p.join, 20)
        self.assertTrue(ready.wait(10))
        # the holder (Core order: journal at gen + 1 first) is between its journal and its grants.json write
        d = self.snapshot()
        ahead = int(d["gen"]) + 1
        for n in S.EntitlementCache.JOURNALS:
            pth = os.path.join(self.cfg, n)
            j = json.load(open(pth))
            j["gen"] = ahead
            with open(pth, "w") as f:
                json.dump(j, f)
        self._timeouts(0.3, 0.3)
        with self.assertRaises(S.StoreBusy):
            self.cache.hard_lock("revoked")
        for j in self.journals():
            self.assertFalse(j["star"])                               # no "*" from the read that raced the writer
            self.assertEqual(set(j["hard_locks"]), {"anibal", "pxqn"})
            self.assertEqual(j["gen"], ahead)                         # gen untouched: the file is not written
            self.assertGreater(j["hard_lock_at"], 0)
        d["gen"] = ahead                                              # the holder finishes its grants.json write
        S.write_atomic(self.cache.path, json.dumps(d).encode())
        p.join(20)
        locks = S.EntitlementCache(self.cfg, pubkeys=KEYS).locks()
        self.assertNotIn("*", locks)
        self.assertEqual({i: locks.get(i) for i in ("anibal", "pxqn")}, {"anibal": "revoked", "pxqn": "revoked"})

    # ---- Minor 1: after StoreBusy the 403 is held in memory until a locked save lands (= Core _mem)
    def test_f311_a_403_dropped_from_the_journal_after_storebusy_is_held_in_memory(self):
        """a writer that read the journal before Control's journal-only write holds the lock past LOCK_TIMEOUT_S +
        LOCK_RETRY_S, then writes back what it loaded (file AND journals): the 403 is gone from disk, but this Control
        keeps it in memory, fails closed on every read and writes it with its next locked save."""
        self.two_items()
        ctx = self.mp()
        ready = ctx.Event()
        p = ctx.Process(target=_stale_writer, args=(self.cache.path, self.cfg, ready, 2.0))
        p.start()
        self.addCleanup(p.join, 20)
        self.assertTrue(ready.wait(10))
        self._timeouts(0.3, 0.3)
        with self.assertRaises(S.StoreBusy):
            self.cache.hard_lock("revoked")
        self.assertTrue(p.is_alive())                                  # deterministic: the overwrite comes after
        p.join(20)
        self.assertEqual(p.exitcode, 0)
        for j in self.journals():
            self.assertEqual(j["hard_locks"], {})                      # the stale writer dropped it from disk ...
        other = S.EntitlementCache(self.cfg, pubkeys=KEYS)
        self.assertTrue(other.has("anibal"))                          # ... (another process sees it open) ...
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))   # ... this one holds it in memory
        self.assertEqual(self.cache.locks().get("anibal"), "revoked")
        self.cache.set_locks({})                                       # the next locked save writes it
        self.assertFalse(other.has("anibal") or other.has("pxqn"))
        self.assertEqual({i: self.snapshot()["hard_locks"].get(i) for i in ("anibal", "pxqn")},
                         {"anibal": "revoked", "pxqn": "revoked"})
        self.assertIsNone(self.cache._pending)                         # landed: nothing held any more
        # a revocation that hits StoreBusy is held the same way
        self.setUp()
        ga, _gp = self.two_items()
        ready = ctx.Event()
        p = ctx.Process(target=_stale_writer, args=(self.cache.path, self.cfg, ready, 2.0))
        p.start()
        self.addCleanup(p.join, 20)
        self.assertTrue(ready.wait(10))
        with self.assertRaises(S.StoreBusy):
            self.cache.apply_revocations(revocations([ga["statement"]["ent_id"]]))
        p.join(20)
        self.assertEqual(self.refused(ga), "revoked")
        self.cache.set_locks({})
        self.assertIn(ga["statement"]["ent_id"], self.snapshot()["revoked"])
        self.assertIsNone(self.cache._pending)


_ABSENT = object()


class Item4SecRead(Base):
    """security read on bdd21afcba (probes
    probe_junk_grant.py, probe_junk_list.py): one hand-edited entry in grants.json must never stop a 403 hard lock."""
    deny, two_items, locked, refused = HardLock403.deny, HardLock403.two_items, HardLock403.locked, HardLock403.refused
    journals = Item4Control.journals

    def edit(self, fn):
        d = self.snapshot()
        fn(d)
        with open(self.cache.path, "w") as f:
            f.write(json.dumps(d))

    def lock_still_lands(self, ga):
        self.deny("revoked")
        r = self.client.sync(self.cache)                            # the 403: never lost to a junk entry
        self.assertEqual(r["hard_lock"], "revoked")
        self.assertEqual(self.refused(ga), "pre_lock")              # the real grant is refused
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))
        locks = self.cache.locks()
        self.assertTrue(locks.get("anibal") in ("revoked", "corrupt") and locks.get("pxqn") in ("revoked", "corrupt"), locks)
        for j in self.journals():
            self.assertTrue(j["star"] or {"anibal", "pxqn"} <= set(j["hard_locks"]), j)

    # ---- NEW-1: a stored grant that is not {statement: {..., issued: finite int}} is the wrong shape
    def test_a_junk_grant_entry_never_stops_a_403(self):
        junk = {"issued-nan": {"statement": {"item_id": "anibal", "issued": float("nan")}},
                "issued-text": {"statement": {"item_id": "anibal", "issued": "soon"}},
                "issued-missing": {"statement": {"item_id": "anibal"}},
                "statement-list": {"statement": [1]},
                "statement-missing": {"signature": "x"},
                "entry-list": [1], "entry-text": "grant"}
        for name, entry in junk.items():
            with self.subTest(junk=name):
                self.setUp()
                ga, _gp = self.two_items()
                self.edit(lambda d: d["grants"].__setitem__("zz-junk", entry))
                self.assertEqual(self.cache.status()["state"], "corrupt")   # fail closed: the "*" rebuild
                self.lock_still_lands(ga)
        record = S.EntitlementCache._lock_record({"grants": {"a": {"statement": [1]}, "b": [1], "c": {"statement":
                                                  {"item_id": "x", "issued": "soon"}}, "d": {"statement": {"item_id": "y",
                                                  "ent_id": "ab" * 16, "issued": 1760400000}}}, "newest": {}, "revoked": []})
        self.assertEqual(record["held"], {"y"})                     # the safe statement helper skips junk, never raises
        self.assertIsNone(S._statement({"statement": {"item_id": "x", "issued": float("nan")}}))
        self.assertIsNone(S._statement([1]))

    # ---- NEW-1 list variant: a non-string element in an id list is the wrong shape (file) / a bad journal
    def test_a_non_string_id_in_a_list_never_stops_a_403_or_a_read(self):
        for key in ("revoked", "hard_lock_seen", "fresh", "star_held"):
            for elem in ([1], 7, None):
                with self.subTest(file_key=key, elem=elem):
                    self.setUp()
                    ga, _gp = self.two_items()
                    self.edit(lambda d: d.__setitem__(key, list(d.get(key) or []) + [elem]))
                    self.assertEqual(self.cache.status()["state"], "corrupt")   # fail closed, never a TypeError
                    self.assertEqual(self.cache.active(), [])
                    self.lock_still_lands(ga)
        for key in ("ids", "held", "revoked", "hard_lock_seen", "fresh"):
            with self.subTest(journal_key=key):
                self.setUp()
                ga, _gp = self.two_items()
                for n in S.EntitlementCache.JOURNALS:
                    p = os.path.join(self.cfg, n)
                    j = json.load(open(p))
                    j[key] = list(j.get(key) or []) + [[1]]
                    with open(p, "w") as f:
                        json.dump(j, f)
                self.assertEqual(self.cache.status()["why"], "journal")           # the F314 way
                self.lock_still_lands(ga)

    # ---- LOW: a non-finite last_sync (the plugin host's key) is the wrong shape (= Core)
    def test_a_non_finite_last_sync_is_the_wrong_shape(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                self.setUp()
                ga, _gp = self.two_items()
                self.edit(lambda d: d.__setitem__("last_sync", bad))
                self.assertEqual(self.cache.status()["state"], "corrupt")
                self.assertNotIn("last_sync", self.snapshot())            # never written back as NaN / Infinity
                self.lock_still_lands(ga)
        self.setUp()
        self.two_items()
        self.edit(lambda d: d.__setitem__("last_sync", 1760400000))      # a finite one is fine
        self.assertEqual(self.cache.status()["state"], "ok")

    # ---- MED, NEW-1 snapshot variant: a junk store/account.json never stops the 403 path (LicdClient._hard_lock)
    def test_a_junk_account_snapshot_never_stops_a_403(self):
        snaps = {"owned-int": json.dumps({"owned": 1}), "owned-true": json.dumps({"owned": True}),
                 "owned-text": json.dumps({"owned": "anibal"}), "owned-junk-list": json.dumps({"owned": [1, [2], {"item_id": 5}]}),
                 "deep": "[" * 20000 + "]" * 20000, "deep-dict": '{"owned": ' + "[" * 20000 + "]" * 20000 + "}",
                 "not-json": "{garbage", "list": "[]"}
        for name, raw in snaps.items():
            with self.subTest(snapshot=name):
                self.setUp()
                ga, _gp = self.two_items()
                with open(os.path.join(self.cfg, "store", "account.json"), "w") as f:
                    f.write(raw)
                self.assertIsInstance(S.load_snapshot(self.cfg), dict)
                self.lock_still_lands(ga)
                self.assertEqual(S.load_snapshot(self.cfg).get("hard_lock"), "revoked")

    # ---- LOW parity: grant times are ints only (= Core / Plugins): a float, a numeric string or a bool is bad_shape
    def resigned(self, field, value):
        g = grant(self.cfg, "anibal")
        g["statement"][field] = value
        g["signature"] = S.b64u(S.ed25519_sign(SEED, S.ENT_PREFIX + S.canonical(g["statement"])))
        return g

    def test_grant_times_are_ints_only(self):
        now = int(time.time())
        for field, value in (("issued", float(now)), ("issued", str(now)), ("refresh_after", float(now + 86400)),
                             ("not_after", str(now + 86400)), ("issued", True), ("refresh_after", True)):
            with self.subTest(field=field, value=value):
                g = self.resigned(field, value)
                with self.assertRaises(S.GrantError) as e:
                    S.verify_grant(g, KEYS)
                self.assertEqual(e.exception.code, "bad_shape")
                self.assertEqual(self.refused(g), "bad_shape")       # never stored, so never a float in grants.json
        self.assertEqual(S.verify_grant(grant(self.cfg, "anibal"), KEYS)[1], "ok")

    # ---- NIT: a stored grant's issued is an int only (= Core / Plugins)
    def test_a_stored_grant_with_a_non_int_issued_is_the_wrong_shape(self):
        self.assertIsNone(S._statement({"statement": {"item_id": "x", "issued": 1760400000.0}}))
        self.assertIsNone(S._statement({"statement": {"item_id": "x", "issued": True}}))
        self.assertEqual(S._issued({"statement": {"item_id": "x", "ent_id": "ab" * 16, "issued": 1760400000}}), 1760400000)
        for value in (1760400000.0, 1760400000.5, True):
            with self.subTest(issued=value):
                self.setUp()
                ga, _gp = self.two_items()
                self.edit(lambda d: d["grants"][ga["statement"]["ent_id"]]["statement"].__setitem__("issued", value))
                self.assertEqual(self.cache.status()["state"], "corrupt")
                self.lock_still_lands(ga)

    # ---- NIT: a junk `checked` in store/account.json never makes the entitlements page answer 500
    def test_a_junk_checked_in_the_snapshot_never_breaks_the_entitlements_page(self):
        self.two_items()
        routes = S.register({}, lambda: self.cfg, KEYS)
        app = types.SimpleNamespace(store_cache=self.cache)
        old = os.environ.get("PXA_STORE_URL")
        os.environ["PXA_STORE_URL"] = self.url
        self.addCleanup(lambda: os.environ.__setitem__("PXA_STORE_URL", old) if old is not None
                        else os.environ.pop("PXA_STORE_URL", None))
        for checked in ("soon", 10 ** 400, float("nan"), float("inf"), [1], {"a": 1}, True):
            with self.subTest(checked=checked):
                with open(os.path.join(self.cfg, "store", "account.json"), "w") as f:
                    f.write(json.dumps({"account": {"id": "a"}, "checked": checked, "ok": True, "owned": []}))
                v = routes[("GET", "/api/store/entitlements")](app, None, {})
                self.assertIn("anibal", v["items"])                   # a bad field reads as due: the page answers

    # ---- NEW-1c: a stored grant whose item_id is not a string (unhashable list / dict) is the wrong shape
    def test_a_stored_grant_with_a_non_string_item_id_never_stops_a_403(self):
        for name, item_id in (("list", [1]), ("dict", {"a": 1}), ("int", 7), ("none", None)):
            with self.subTest(item_id=name):
                self.setUp()
                ga, _gp = self.two_items()
                self.edit(lambda d: d["grants"].__setitem__("zz-junk", {"statement": {"item_id": item_id,
                                                                                        "issued": 1760400000}}))
                self.assertEqual(self.cache.status()["state"], "corrupt")   # fail closed: the "*" rebuild
                self.lock_still_lands(ga)
        self.assertIsNone(S._statement({"statement": {"item_id": [1], "issued": 1760400000}}))

    # ---- NEW-1d (security read): the journal merge asks _lifted about a lock the file lost; _lifted reads RAW grants
    def held_two(self):
        """anibal + pxqn held, straight into the cache (the table's 280 rows skip the fake licd round trips)"""
        ga, gp = grant(self.cfg, "anibal"), grant(self.cfg, "pxqn", kind="quant-tier")
        self.cache.add(ga)
        self.cache.add(gp)
        self.assertTrue(self.cache.has("anibal") and self.cache.has("pxqn"))
        return ga, gp

    def lifted_setup(self, fast=False):
        """a 403 locked every item; the file then lost its own lock records (the journal keeps them). -> (ga, lock time)"""
        if fast:
            ga, _gp = self.held_two()
            self.cache.hard_lock("revoked")
        else:
            ga, _gp = self.locked()
        at = self.snapshot()["hard_lock_at"]
        self.edit(lambda d: [d.__setitem__(k, {}) for k in ("hard_locks", "locks", "revoked_items")])
        return ga, at

    def fresh_for(self, item, at, field=None, value=None, envelope=False):
        """a grant for item issued after the lock, new ent_id, signed with the test key AFTER the junk went in (so the
        junk reaches every check past the signature too); field None = a good one."""
        g = grant(self.cfg, item, issued=int(at) + 5)
        st = g["statement"]
        if envelope:
            g[field] = value
        elif field is not None:
            if value is _ABSENT:
                st.pop(field)
            else:
                st[field] = value
            if isinstance(st, dict):
                g["signature"] = S.b64u(S.ed25519_sign(SEED, S.ENT_PREFIX + S.canonical(st)))
        key = st.get("ent_id") if isinstance(st, dict) and isinstance(st.get("ent_id"), str) and len(st["ent_id"]) <= 128 \
            else "ef" * 16
        return key, g

    def reads(self):
        """every read the page and the plugin gate make; none may raise. -> the active item ids"""
        self.cache.status()
        self.cache.locks()
        self.cache.items()
        self.cache.has("anibal")
        S.entitlements_view(self.cache, S.load_snapshot(self.cfg))
        return {st["item_id"] for st, _s in self.cache.active()}

    def test_new_1d_a_fresh_grant_with_an_unhashable_ent_id_never_breaks_the_lifted_lock_merge(self):
        for name, ent in (("list", [1]), ("dict", {"a": 1})):
            with self.subTest(ent_id=name):
                self.setUp()
                _ga, at = self.lifted_setup()
                self.edit(lambda d: d["grants"].__setitem__("cd" * 16, {"statement": {"item_id": "anibal", "ent_id": ent,
                                                                                        "issued": int(at) + 5}}))
                self.assertNotIn("anibal", self.reads())
                self.cache.hard_lock("revoked", ["other-item"])                # a 403 for another item still writes
                locks = self.cache.locks()
                self.assertTrue("*" in locks or {"anibal", "other-item"} <= set(locks), locks)
        self.assertIsNone(S._statement({"statement": {"item_id": "x", "ent_id": [1], "issued": 1760400000}}))

    def test_new_1d_fresh_is_false_for_a_non_string_ent_id(self):
        # defence in depth: _lifted reads raw grants, so _fresh itself never hashes an ent_id that is not a str
        for ent in ([1], {"a": 1}, None, 7, 1.5, True):
            with self.subTest(ent_id=repr(ent)):
                self.assertFalse(S.EntitlementCache._fresh({"issued": 10 ** 10, "ent_id": ent}, 0, {"ab": 1}, {"ab"}))
                self.assertFalse(S.EntitlementCache._fresh({"issued": 10 ** 10, "ent_id": ent}, 0, None, set()))

    # ---- the table: every field any path reads, every junk value, plain read and lifted lock
    JUNK = (("list", [1]), ("dict", {}), ("none", None), ("float", 1.5), ("true", True), ("huge", 10 ** 400),
            ("nan-text", "NaN"), ("nan", float("nan")), ("64k", "x" * 65536), ("absent", None))
    STATEMENT_FIELDS = ("v", "iss", "kid", "ent_id", "kind", "item_id", "level", "sub", "install_pub", "issued",
                        "refresh_after", "not_after")
    ENVELOPE_FIELDS = ("statement", "signature")
    LEGIT = {("level", "none"), ("level", "nan-text"), ("level", "absent")}     # level: "pxqn4" | null (licd)

    def table_rows(self):
        for f in self.STATEMENT_FIELDS:
            for vname, v in self.JUNK:
                yield f, vname, (_ABSENT if vname == "absent" else v), False
        for f in self.ENVELOPE_FIELDS:
            for vname, v in self.JUNK:
                yield f, vname, (_ABSENT if vname == "absent" else v), True

    def test_table_junk_in_any_stored_grant_field_never_raises_never_activates_never_stops_a_403(self):
        rows = 0
        for path in ("plain", "lifted"):
            # the good row first: the lifted path really lifts (so the junk rows reach _lifted / _fresh / verify_grant)
            self.setUp()
            _ga, at = self.lifted_setup(fast=True)
            key, g = self.fresh_for("anibal", at)
            self.edit(lambda d: d["grants"].__setitem__(key, g))
            self.assertIn("anibal", self.reads())
            for field, vname, value, envelope in self.table_rows():
                with self.subTest(path=path, field=field, value=vname):
                    rows += 1
                    self.setUp()
                    if path == "plain":
                        ga, _gp = self.held_two()
                        at, item = time.time() - 600, "junk-item"
                    else:
                        ga, at = self.lifted_setup(fast=True)
                        item = "anibal"
                    if envelope and value is _ABSENT:
                        key, g = self.fresh_for(item, at)
                        g.pop(field)
                    else:
                        key, g = self.fresh_for(item, at, field, value, envelope)
                    self.edit(lambda d: d["grants"].__setitem__(key, g))
                    active = self.reads()                                     # no read raises
                    if (field, vname) not in self.LEGIT:
                        self.assertNotIn(item, active)                        # the junk grant never makes an item active
                    self.assertLessEqual(active - {item}, {"anibal", "pxqn"} if path == "plain" else set())
                    self.cache.hard_lock("revoked", ["other-item"])           # a 403 for another item still writes
                    locks = self.cache.locks()
                    self.assertTrue("*" in locks or {"anibal", "pxqn", "other-item"} <= set(locks), locks)
                    self.assertEqual(self.reads() - {"*"}, set())             # and holds after
        self.assertEqual(rows, 2 * (len(self.STATEMENT_FIELDS) + len(self.ENVELOPE_FIELDS)) * len(self.JUNK))


# ---------------------------------------------------------------------------------------------------------- the shared fixture
def _load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def vector_cache_has(vec, pubkeys):
    """Control's EntitlementCache primed exactly as the vector says (the same priming as the Plugins cross-check)."""
    cfg = tempfile.mkdtemp(dir=TMP)
    try:
        os.makedirs(os.path.join(cfg, "plugins"))
        with open(os.path.join(cfg, "plugins", "install.key"), "wb") as f:
            f.write(hashlib.sha256(VECTOR_LABEL + b"|" + b"install-this").digest())
        os.makedirs(os.path.join(cfg, "store"))
        st = vec["grant"]["statement"]
        with open(os.path.join(cfg, "store", "grants.json"), "w") as f:
            json.dump({"grants": {st["ent_id"]: vec["grant"]}, "revoked": [], "newest": {st["ent_id"]: st["issued"]},
                       "seen": vec["seen"]}, f)
        cache = S.EntitlementCache(cfg, pubkeys=pubkeys, clock=lambda: vec["now"])
        if vec.get("revocations"):
            cache.apply_revocations(vec["revocations"])
        return cache.has(st["item_id"])
    finally:
        shutil.rmtree(cfg, ignore_errors=True)


def vector_verify(vec, pubkeys):
    try:
        return S.verify_grant(vec["grant"], pubkeys, vec["now"], vec["install_pub"])[1]
    except S.GrantError as e:
        return e.code


def run_fixture():
    """-> [(name, ok, got, want)] over both layers; the number the three clients report (Plugins 8/8)."""
    fx = _load_fixture()
    out = []
    for vec in fx["vectors"]:
        got = (vector_verify(vec, fx["pubkeys"]), vector_cache_has(vec, fx["pubkeys"]))
        want = (vec["expect"]["verify"], vec["expect"]["extras"])
        out.append((vec["name"], got == want, got, want))
    return out


# ------------------------------------------------- (F305-P) a corrupt grants.json fails closed, kept aside, never rewritten
class CorruptFailsClosed(Base):
    """= the plugin host's GrantCache rule (plugins/v1-store-rc36 store_backend._load/_corrupt): unreadable, cut off, not
    an object or a known key of the wrong type -> the file goes to grants.json.corrupt-<t> and every item is hard-locked
    ("*": "corrupt") until licd issues a fresh grant per item."""

    def two(self):
        ga, gp = grant(self.cfg, "anibal"), grant(self.cfg, "pxqn", kind="quant-tier")
        self.on_refresh(self.refresh_reply([ga, gp]))
        self.client.sync(self.cache)
        self.assertTrue(self.cache.has("anibal") and self.cache.has("pxqn"))
        return ga, gp

    def copies(self):
        return sorted(n for n in os.listdir(self.cache.dir) if n.startswith("grants.json.corrupt-"))

    def corrupt_with(self, raw):
        with open(self.cache.path, "wb") as f:
            f.write(raw)

    def assert_fails_closed(self, raw, ga):
        self.assertEqual(self.cache.locks(), {"*": "corrupt"})
        self.assertFalse(self.cache.has("anibal") or self.cache.has("pxqn"))
        self.assertEqual(self.cache.active(), [])
        names = self.copies()
        self.assertEqual(len(names), 1, names)
        with open(os.path.join(self.cache.dir, names[0]), "rb") as f:
            self.assertEqual(f.read(), raw)                         # the bad bytes are kept, byte for byte
        d = self.snapshot()                                          # grants.json is a NEW file: the lock, not the bad bytes
        self.assertEqual((d["hard_locks"], d["hard_lock"], d["grants"], d["corrupt_copy"]),
                         ({"*": "corrupt"}, "corrupt", {}, names[0]))
        with self.assertRaises(S.GrantError) as e:                   # the grant held before is from before the lock
            self.cache.add(ga)
        self.assertEqual(e.exception.code, "pre_lock")
        v = S.entitlements_view(self.cache, S.load_snapshot(self.cfg))
        self.assertEqual((v["items"], v["hard_lock"], v["cache"]["state"], v["cache"]["corrupt_copy"]),
                         ({}, "corrupt", "corrupt", names[0]))
        self.assertEqual(self.copies(), names)                       # no second copy: the new file reads fine

    def test_corrupt_json_fails_closed_and_is_kept_aside(self):
        ga, _gp = self.two()
        self.corrupt_with(b"{garbage")
        self.assert_fails_closed(b"{garbage", ga)

    def test_truncated_file_fails_closed(self):
        ga, _gp = self.two()
        with open(self.cache.path, "rb") as f:
            raw = f.read()
        self.corrupt_with(raw[: len(raw) // 2])
        self.assert_fails_closed(raw[: len(raw) // 2], ga)

    def test_non_object_and_wrong_types_fail_closed(self):
        bad = [b"[]", b'"grants"', b"42", b"null", b"", b"\xff\xfe\x00", b'{"grants": []}', b'{"revoked": {}}',
               b'{"newest": 1}', b'{"seen": "x"}', b'{"seen": true}', b'{"hard_locks": ["anibal"]}',
               b'{"hard_lock_at": "soon", "hard_lock_seen": []}', b'{"hard_lock_at": 5}', b'{"last_sync": "x"}',
               b'{"revoked_items": []}', b'{"licd_time": []}', b"[" * 100000 + b"]" * 100000]
        for raw in bad:
            with self.subTest(raw=raw[:40]):
                self.setUp()
                ga, _gp = self.two()
                self.corrupt_with(raw)
                self.assert_fails_closed(raw, ga)

    def test_every_entry_point_fails_closed(self):
        for name, call in (("add", lambda c, g: c.add(g)), ("set_locks", lambda c, g: c.set_locks({"x": "y"})),
                           ("items", lambda c, g: c.items()), ("drop_item", lambda c, g: c.drop_item("pxqn")),
                           ("hard_lock", lambda c, g: c.hard_lock("revoked")), ("active", lambda c, g: c.active()),
                           ("apply_revocations", lambda c, g: c.apply_revocations(revocations(["ab" * 16]))),
                           ("status", lambda c, g: c.status())):
            with self.subTest(entry=name):
                self.setUp()
                ga, _gp = self.two()
                self.corrupt_with(b"{garbage")
                try:
                    call(self.cache, ga)
                except S.GrantError:
                    pass
                self.assertEqual(len(self.copies()), 1)
                self.assertIn("*", self.cache.locks())
                self.assertFalse(self.cache.has("anibal"))

    def test_a_file_that_cannot_be_kept_aside_is_never_overwritten(self):
        ga, _gp = self.two()
        self.corrupt_with(b"{garbage")
        saved = S._aside
        S._aside = lambda path: None
        try:
            for fn in (self.cache.locks, self.cache.active, lambda: self.cache.set_locks({"a": "b"}),
                       lambda: self.cache.hard_lock("revoked"), lambda: self.cache.add(grant(self.cfg, "anibal", issued=int(time.time()) + 5))):
                try:
                    fn()
                except S.GrantError:
                    pass
                with open(self.cache.path, "rb") as f:
                    self.assertEqual(f.read(), b"{garbage")
            self.assertEqual(self.cache.locks(), {"*": "corrupt"})
            self.assertFalse(self.cache.has("anibal"))
            self.assertEqual(self.cache.status()["state"], "corrupt")
        finally:
            S._aside = saved

    def test_a_fresh_grant_lifts_the_corruption_lock_for_its_item_only(self):
        self.two()
        self.corrupt_with(b"{garbage")
        self.cache.locks()
        self.cache.add(grant(self.cfg, "anibal", issued=int(time.time()) + 2))
        self.assertTrue(self.cache.has("anibal"))
        self.assertFalse(self.cache.has("pxqn"))
        self.assertEqual(self.snapshot()["fresh"], ["anibal"])

    def test_two_corruptions_never_overwrite_an_earlier_copy(self):
        self.two()
        self.corrupt_with(b"{one")
        self.cache.locks()
        self.corrupt_with(b"{two")
        self.cache.locks()
        got = set()
        for n in self.copies():
            with open(os.path.join(self.cache.dir, n), "rb") as f:
                got.add(f.read())
        self.assertEqual(got, {b"{one", b"{two"})

    def test_a_plugin_host_corruption_lock_is_honoured(self):
        g = self.primed()
        d = self.snapshot()
        d.update(hard_locks={"*": "corrupt"}, hard_lock_at=float(time.time()), hard_lock_seen=[], hard_lock="corrupt",
                 fresh=[], last_sync=0, revoked_items={})          # what GrantCache._corrupt writes (no licd_time, no clock)
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        self.assertFalse(self.cache.has("anibal"))
        self.assertEqual(self.cache.status()["state"], "corrupt")
        self.assertEqual(self.cache.items(), ["anibal"])

    def test_a_symlinked_file_fails_closed_and_its_target_is_untouched(self):
        ga, _gp = self.two()
        real = os.path.join(self.cfg, "restored-backup.json")
        shutil.copy(self.cache.path, real)
        os.unlink(self.cache.path)
        os.symlink(real, self.cache.path)                           # QA F304 symlink_rollback shape
        self.assertEqual(self.cache.locks(), {"*": "corrupt"})
        self.assertFalse(self.cache.has("anibal"))
        self.assertEqual(self.cache.status()["why"], "link")
        names = self.copies()
        self.assertEqual(len(names), 1)
        self.assertTrue(os.path.islink(os.path.join(self.cache.dir, names[0])))   # the link itself was moved aside
        self.assertFalse(os.path.islink(self.cache.path))
        with open(real) as f:
            self.assertIn(ga["statement"]["ent_id"], f.read())      # the file it pointed at is not written

    def test_a_missing_file_with_a_lock_journal_fails_closed(self):
        for j in S.EntitlementCache.JOURNALS:
            with self.subTest(journal=j):
                self.setUp()
                self.two()
                os.makedirs(os.path.dirname(os.path.join(self.cfg, j)), exist_ok=True)
                with open(os.path.join(self.cfg, j), "w") as f:
                    json.dump({"v": 1, "gen": 3, "hard_locks": {"anibal": "revoked"}}, f)
                os.unlink(self.cache.path)
                self.assertEqual(self.cache.locks(), {"*": "corrupt"})
                self.assertFalse(self.cache.has("anibal"))
                self.assertEqual(self.cache.status()["why"], "missing")

    def test_a_missing_file_is_empty_not_corrupt(self):
        self.assertEqual((self.cache.locks(), self.cache.status()["state"], self.copies() if os.path.isdir(self.cache.dir) else []),
                         ({}, "ok", []))


# ------------------------------------------------ (F303-P) one cross-process lock around every load-modify-save
def _child_add(cfg, env, out):
    try:
        S.EntitlementCache(cfg, pubkeys=KEYS).add(env)
        out.put("ok")
    except Exception as e:  # noqa: BLE001
        out.put("err %r" % e)


def _child_hard_lock(cfg, started):
    started.set()
    S.EntitlementCache(cfg, pubkeys=KEYS).hard_lock("revoked", items=["anibal"])


def _child_crash_on_replace(cfg, env):
    real = os.replace

    def crash(src, dst):
        if dst.endswith("grants.json"):
            os._exit(9)                                       # the process dies between the fsync'd temp file and the rename
        return real(src, dst)
    os.replace = crash
    S.EntitlementCache(cfg, pubkeys=KEYS).add(env)


class OneWriterAtATime(Base):
    def mp(self):
        import multiprocessing
        return multiprocessing.get_context("fork")

    def test_concurrent_writers_in_processes_lose_no_update(self):
        ctx = self.mp()
        envs = [grant(self.cfg, "item-%02d" % i, kind="feature") for i in range(24)]
        out = ctx.Queue()
        procs = [ctx.Process(target=_child_add, args=(self.cfg, e, out)) for e in envs]
        for p in procs:
            p.start()
        for p in procs:
            p.join(60)
        self.assertEqual(sorted(out.get(timeout=5) for _ in procs), ["ok"] * len(envs))
        d = self.snapshot()
        self.assertEqual(set(d["grants"]), {e["statement"]["ent_id"] for e in envs})
        self.assertEqual(len(self.cache.active()), len(envs))

    def test_concurrent_writers_in_threads_lose_no_update(self):
        envs = [grant(self.cfg, "item-%02d" % i, kind="feature") for i in range(16)]
        caches = [S.EntitlementCache(self.cfg, pubkeys=KEYS) for _ in range(4)]
        bar = threading.Barrier(len(envs))
        errs = []

        def run(i, e):
            bar.wait()
            try:
                caches[i % 4].add(e)
                caches[(i + 1) % 4].set_locks({"lock-%02d" % i: "suspended"})
            except Exception as x:  # noqa: BLE001
                errs.append(x)
        ts = [threading.Thread(target=run, args=(i, e)) for i, e in enumerate(envs)]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual(errs, [])
        d = self.snapshot()
        self.assertEqual(len(d["grants"]), len(envs))
        self.assertEqual(set(d["locks"]), {"lock-%02d" % i for i in range(len(envs))})

    def test_add_racing_a_403_in_another_process_keeps_the_lock(self):
        """QA F303-P repro (vigil/h-qa-plugins-review 88bcf376a3 control_lost_update), with a real second process: add()
        is held between load and save; the 403 hard_lock in another process must wait for it, then lock on top."""
        ga = self.primed("anibal")
        entered, go = threading.Event(), threading.Event()
        real = S.verify_grant

        def slow(*a, **k):
            r = real(*a, **k)
            entered.set()
            go.wait(20)
            return r
        S.verify_grant = slow
        ctx = self.mp()
        started = ctx.Event()
        t = threading.Thread(target=lambda: self.cache.add(grant(self.cfg, "preset-x", kind="preset")))
        try:
            t.start()
            self.assertTrue(entered.wait(10))
            p = ctx.Process(target=_child_hard_lock, args=(self.cfg, started))
            p.start()
            self.assertTrue(started.wait(10))
            p.join(0.5)
            self.assertTrue(p.is_alive(), "the 403 wrote while add() held the store lock")
        finally:
            go.set()
            t.join(20)
            S.verify_grant = real
        p.join(30)
        self.assertEqual(p.exitcode, 0)
        locks = self.cache.locks()
        self.assertEqual((locks.get("anibal"), locks.get("preset-x")), ("revoked", "revoked"))
        self.assertFalse(self.cache.has("anibal") or self.cache.has("preset-x"))
        with self.assertRaises(S.GrantError):
            self.cache.add(ga)

    def test_a_403_landing_inside_add_like_qa_repro_keeps_the_lock(self):
        """the exact QA shape: the 403 fires from add()'s clock call (another cache on the same file). The clock is read
        before the store lock is taken, so the 403 lands first and add() works on the locked file."""
        self.primed("anibal")
        other = S.EntitlementCache(self.cfg, pubkeys=KEYS)
        fired = []

        def hook():
            if not fired:
                fired.append(1)
                other.hard_lock("revoked", items=["anibal", "pxqn"])
            return time.time()
        A = S.EntitlementCache(self.cfg, pubkeys=KEYS, clock=hook)
        A.add(grant(self.cfg, "preset-x", kind="preset", issued=int(time.time()) + 2))
        self.assertTrue(fired)
        self.assertEqual(self.cache.locks().get("anibal"), "revoked")
        self.assertFalse(self.cache.has("anibal"))

    def test_add_waits_for_a_403_in_progress_and_is_refused(self):
        g_old = self.primed("anibal")
        entered, go = threading.Event(), threading.Event()
        real = S.write_atomic
        calls = []

        def slow(path, data):
            if path.endswith("grants.json") and not calls:
                calls.append(1)
                entered.set()
                go.wait(20)
            return real(path, data)
        S.write_atomic = slow
        try:
            t = threading.Thread(target=lambda: self.cache.hard_lock("revoked"))
            t.start()
            self.assertTrue(entered.wait(10))
            res = []
            u = threading.Thread(target=lambda: res.append(self._try_add(g_old)))
            u.start()
            u.join(0.5)
            self.assertTrue(u.is_alive(), "add() ran while hard_lock() held the store lock")
            go.set()
            t.join(20)
            u.join(20)
        finally:
            go.set()
            S.write_atomic = real
        self.assertEqual(res, ["pre_lock"])
        self.assertFalse(self.cache.has("anibal"))

    def _try_add(self, g):
        try:
            S.EntitlementCache(self.cfg, pubkeys=KEYS).add(g)
            return "accepted"
        except S.GrantError as e:
            return e.code

    def test_the_lock_file_is_shared_and_kept(self):
        self.primed()
        self.assertTrue(os.path.exists(self.cache.path + ".lock"))
        with S.file_lock(self.cache.path):
            with S.file_lock(self.cache.path):              # re-entrant in one thread
                pass
            ctx = self.mp()
            out = ctx.Queue()
            p = ctx.Process(target=_child_add, args=(self.cfg, grant(self.cfg, "pxqn", kind="quant-tier"), out))
            p.start()
            p.join(0.5)
            self.assertTrue(p.is_alive(), "another process wrote while this one held grants.json.lock")
        p.join(30)
        self.assertEqual(out.get(timeout=5), "ok")
        self.assertTrue(self.cache.has("pxqn"))

    def test_a_busy_lock_refuses_an_unlock_but_a_403_is_still_written(self):
        self.primed()
        before = self.snapshot()
        ctx = self.mp()
        hold, ready = ctx.Event(), ctx.Event()

        def holder(path):
            with S.file_lock(path):
                ready.set()
                hold.wait(20)
        p = ctx.Process(target=holder, args=(self.cache.path,))
        p.start()
        try:
            self.assertTrue(ready.wait(10))
            saved = S.LOCK_TIMEOUT_S, S.LOCK_RETRY_S
            S.LOCK_TIMEOUT_S, S.LOCK_RETRY_S = 0.3, 0.3
            try:
                for fn in (lambda: self.cache.add(grant(self.cfg, "pxqn", kind="quant-tier")),
                           lambda: self.cache.set_locks({}, replace=True)):
                    with self.assertRaises(S.StoreBusy):           # an unlock never runs without the lock
                        fn()
                    self.assertEqual(self.snapshot(), before)
                with self.assertRaises(S.StoreBusy):               # F311: a lock goes to the journal (contract rule 8),
                    self.cache.hard_lock("revoked")                 # never to grants.json without the lock
                self.assertEqual(self.snapshot(), before)
                self.assertFalse(self.cache.has("anibal"))
            finally:
                S.LOCK_TIMEOUT_S, S.LOCK_RETRY_S = saved
        finally:
            hold.set()
            p.join(20)
        self.cache.set_locks({})                                    # the next locked write puts it in the file
        self.assertEqual(self.snapshot()["hard_locks"], {"anibal": "revoked"})
        self.assertFalse(self.cache.has("anibal"))

    def test_a_symlinked_lock_file_is_never_followed(self):
        self.primed()
        target = os.path.join(self.cfg, "elsewhere")
        with open(target, "w") as f:
            f.write("keep")
        os.unlink(self.cache.path + ".lock")
        os.symlink(target, self.cache.path + ".lock")
        with self.assertRaises(OSError):                            # O_NOFOLLOW: refused, not locked through the link
            self.cache.add(grant(self.cfg, "pxqn", kind="quant-tier"))
        self.cache.hard_lock("revoked")                             # a 403 still lands
        self.assertFalse(self.cache.has("anibal"))
        with open(target) as f:
            self.assertEqual(f.read(), "keep")

    def test_reads_never_write(self):
        self.primed()
        with open(self.cache.path, "rb") as f:
            raw = f.read()
        st = os.stat(self.cache.path)
        for fn in (self.cache.active, lambda: self.cache.has("anibal"), self.cache.items, self.cache.locks,
                   self.cache.status, lambda: S.entitlements_view(self.cache, {})):
            fn()
        with open(self.cache.path, "rb") as f:
            self.assertEqual(f.read(), raw)
        self.assertEqual(os.stat(self.cache.path).st_ino, st.st_ino)   # not even replaced with the same bytes

    def test_the_plugin_hosts_keys_are_kept(self):
        self.primed()
        d = self.snapshot()
        extra = {"star_held": ["anibal"], "hard_lock_why": "x", "last_sync": 5, "revoked_items": {},
                 "future_key": {"a": [1]}}
        d["gen"] = d.get("gen", 0) + 5                              # the plugin host saved since (gen moves on, never back)
        d.update(extra)
        with open(self.cache.path, "w") as f:
            json.dump(d, f)
        self.cache.add(grant(self.cfg, "pxqn", kind="quant-tier"))
        self.cache.set_locks({"z": "suspended"})
        self.cache.hard_lock("revoked")
        after = self.snapshot()
        self.assertEqual({k: after.get(k) for k in extra}, extra)
        self.assertEqual(after["gen"], d["gen"] + 3)                # one bump per save


class CrashMidWrite(Base):
    def files(self):
        return sorted(os.listdir(self.cache.dir))

    def test_a_failure_before_the_rename_leaves_the_old_file_and_no_temp(self):
        self.primed()
        with open(self.cache.path, "rb") as f:
            old = f.read()
        for name in ("fsync", "replace"):
            with self.subTest(fail=name):
                real = getattr(os, name)

                def boom(*a, **k):
                    raise OSError("disk full")
                setattr(os, name, boom)
                try:
                    with self.assertRaises(OSError):
                        self.cache.add(grant(self.cfg, "pxqn", kind="quant-tier"))
                finally:
                    setattr(os, name, real)
                with open(self.cache.path, "rb") as f:
                    self.assertEqual(f.read(), old)
                self.assertFalse([n for n in self.files() if ".tmp" in n], self.files())
                self.assertTrue(self.cache.has("anibal") and not self.cache.has("pxqn"))
                with open(self.cache.path, "rb") as f:
                    old = f.read()                   # has() saved `seen`

    def test_a_process_killed_mid_write_leaves_the_old_file_intact(self):
        self.primed()
        with open(self.cache.path, "rb") as f:
            old = json.loads(f.read())
        import multiprocessing
        p = multiprocessing.get_context("fork").Process(target=_child_crash_on_replace,
                                                        args=(self.cfg, grant(self.cfg, "pxqn", kind="quant-tier")))
        p.start()
        p.join(30)
        self.assertEqual(p.exitcode, 9)
        self.assertEqual(self.snapshot(), old)                      # the old file, whole: not corrupt, not half new
        self.assertTrue(self.cache.has("anibal"))
        self.assertFalse(self.cache.has("pxqn"))
        self.assertEqual(self.cache.status()["state"], "ok")
        self.cache.add(grant(self.cfg, "pxqn", kind="quant-tier"))  # the lock the dead process held is free again
        self.assertTrue(self.cache.has("pxqn"))

    def test_saves_are_0600_and_go_through_one_rename(self):
        self.primed()
        import stat
        self.assertEqual(stat.S_IMODE(os.stat(self.cache.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.cache.path + ".lock").st_mode), 0o600)
        seen = []
        real = os.replace
        os.replace = lambda s, d: (seen.append((os.path.basename(s), os.path.basename(d))), real(s, d))[1]
        try:
            self.cache.set_locks({"x": "suspended"})
        finally:
            os.replace = real
        self.assertEqual([x[1] for x in seen], ["grants.journal.json", "store-lock-journal.json", "grants.json",
                                                "grants.journal.json", "store-lock-journal.json"])   # F312: journal, file, journal
        self.assertTrue(all(x[0].startswith("." + x[1] + ".tmp-") for x in seen), seen)


# ------------------------------------------- the not-yet check runs on licd's clock (server_time, else Date; 5 min)
class LicdClockNotYet(Base):
    def sync_with(self, g, date=None, body_time=None):
        reply = self.refresh_reply([g])
        if body_time is not None:
            reply["server_time"] = body_time
        self.on_refresh(reply)
        FakeLicd.date_header = date
        try:
            return self.client.sync(self.cache)
        finally:
            FakeLicd.date_header = None

    def test_licd_20_minutes_ahead_accepts_a_fresh_grant(self):
        now = time.time()
        g = grant(self.cfg, "anibal", issued=int(now + 1200))
        r = self.sync_with(g, date=http_date(now + 1200))
        self.assertEqual((r["grants"], r["refused"]), (1, []))
        self.assertTrue(self.cache.has("anibal"))                   # and it keeps counting on the local clock afterwards
        v = S.entitlements_view(self.cache, S.load_snapshot(self.cfg))
        self.assertEqual(v["items"]["anibal"]["state"], "owned")
        self.setUp()
        g = grant(self.cfg, "anibal", issued=int(now + 1200))
        self.assertEqual(self.sync_with(g, date=http_date(now + 1100), body_time=int(now + 1200))["refused"], [])  # body wins
        self.assertTrue(self.cache.has("anibal"))                   # (Date within 5 min of it)

    def test_a_grant_more_than_5_minutes_past_licd_time_is_refused(self):
        now = time.time()
        for server, issued in ((now + 1200, now + 1200 + 400), (now - 1200, now), (now, now + 400)):
            with self.subTest(server=int(server - now), issued=int(issued - now)):
                self.setUp()
                g = grant(self.cfg, "anibal", issued=int(issued))
                r = self.sync_with(g, date=http_date(server))
                self.assertEqual(r["refused"], ["not_yet"])
                self.assertFalse(self.cache.has("anibal"))
        self.setUp()                                                # within the 5 minutes: accepted
        g = grant(self.cfg, "anibal", issued=int(now - 1200 + 250))
        self.assertEqual(self.sync_with(g, date=http_date(now - 1200))["refused"], [])
        self.assertTrue(self.cache.has("anibal"))

    def test_without_licd_time_the_local_clock_applies(self):
        now = time.time()
        for date in ("", "yesterday at noon", "Thu, 01 Jan 1970 00:00:00 GMT"):
            with self.subTest(date=date):
                self.setUp()
                r = self.sync_with(grant(self.cfg, "anibal", issued=int(now + 1200)), date=date, body_time="soon")
                self.assertEqual(r["refused"], ["not_yet"])
                self.setUp()
                r = self.sync_with(grant(self.cfg, "anibal", issued=int(now + 200)), date=date)
                self.assertEqual(r["refused"], [])
                self.assertTrue(self.cache.has("anibal"))

    def test_the_reply_carries_licd_time_and_add_takes_it(self):
        now = time.time()
        FakeLicd.script["/v1/store/revocations"] = lambda _b: (200, {"server_time": int(now + 900)})
        FakeLicd.date_header = http_date(now + 900)
        r = self.client.revocations()
        FakeLicd.date_header = None
        self.assertEqual((r.server_time, r.server_source), (float(int(now + 900)), "body"))
        g = grant(self.cfg, "anibal", issued=int(now + 900))
        with self.assertRaises(S.GrantError) as e:
            self.cache.add(g)                                       # no licd time: local clock, 15 min ahead -> not_yet
        self.assertEqual(e.exception.code, "not_yet")
        self.cache.add(g, server_now=now + 900)
        self.assertTrue(self.cache.has("anibal"))
        self.assertEqual(self.snapshot()["licd_time"], {g["statement"]["ent_id"]: float(now + 900)})


# ------------------------- (F309-P) the lock journal outside grants.json: locks, ids, gen and a high-water time
class JournalF309(Base):
    """= Plugins GrantCache (plugins/v1-store-rc36 9351f0a99c / f20def198b, DESIGN.md s31.3): every save bumps `gen` and
    writes <config>/store/grants.journal.json and <config>/plugins/store-lock-journal.json; a corrupt or missing file is
    rebuilt from them; `seen` there is the rollback high-water time."""

    def deny(self, code="revoked"):
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (403, {"error": "no", "code": code})

    def two(self):
        ga, gp = grant(self.cfg, "anibal"), grant(self.cfg, "pxqn", kind="quant-tier")
        self.on_refresh(self.refresh_reply([ga, gp]))
        self.client.sync(self.cache)
        return ga, gp

    def journals(self):
        out = []
        for j in S.EntitlementCache.JOURNALS:
            with open(os.path.join(self.cfg, j)) as f:
                out.append(json.load(f))
        return out

    def at(self, t):
        return S.EntitlementCache(self.cfg, pubkeys=KEYS, clock=lambda: t[0])

    def code_of(self, cache, g):
        try:
            cache.add(g)
            return "accepted"
        except S.GrantError as e:
            return e.code

    def test_qa_repro_clock_back_into_the_skew_then_corrupt(self):
        """QA F309-P (vigil/h-qa-plugins-review c3f16587a1, probe corrupt_clock_back) on Control: 403 lock, clock set back
        to just inside the 5-minute skew of the old grant, grants.json made unparseable: the old grant stays out."""
        ga, gp = self.two()
        self.deny()
        self.client.sync(self.cache)
        t = [float(ga["statement"]["issued"]) - 299]
        back = self.at(t)
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        back.locks()
        self.assertEqual(self.code_of(back, ga), "pre_lock")
        self.assertEqual(self.code_of(back, gp), "pre_lock")
        t[0] = time.time()                                          # the clock put right again
        self.assertFalse(back.has("anibal") or back.has("pxqn"))
        self.assertFalse(self.cache.has("anibal"))

    def test_without_a_403_an_old_held_grant_never_lifts_a_corruption_lock(self):
        ga, _gp = self.two()
        t = [float(ga["statement"]["issued"]) - 299]
        back = self.at(t)
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        self.assertEqual(self.code_of(back, ga), "pre_lock")        # its id is in the journal's ids
        self.assertFalse(back.has("anibal"))

    def test_the_journal_holds_gen_ids_locks_and_the_high_water_time(self):
        ga, gp = self.two()
        d = self.snapshot()
        j1, j2 = self.journals()
        self.assertEqual(j1, j2)
        self.assertEqual(j1["gen"], d["gen"])
        self.assertTrue({ga["statement"]["ent_id"], gp["statement"]["ent_id"]} <= set(j1["ids"]))
        self.assertEqual(set(j1["held"]), {"anibal", "pxqn"})
        self.assertAlmostEqual(j1["seen"], d["seen"], delta=1)
        self.deny()
        self.client.sync(self.cache)
        j = self.journals()[0]
        self.assertEqual(set(j["hard_locks"]), {"anibal", "pxqn"})
        self.assertEqual(j["hard_lock_at"], self.snapshot()["hard_lock_at"])
        self.assertGreater(j["gen"], j1["gen"])

    def test_a_corrupt_or_missing_file_is_rebuilt_from_the_journal(self):
        for how in ("corrupt", "missing", "truncated"):
            with self.subTest(how=how):
                self.setUp()
                ga, gp = self.two()
                self.deny()
                self.client.sync(self.cache)
                lock_at = self.snapshot()["hard_lock_at"]
                if how == "missing":
                    os.unlink(self.cache.path)
                else:
                    with open(self.cache.path, "rb+") as f:
                        raw = f.read()
                        f.seek(0)
                        f.truncate()
                        f.write(b"{garbage" if how == "corrupt" else raw[: len(raw) // 2])
                d = (self.cache.locks(), self.snapshot())[1]
                self.assertEqual(d["hard_locks"].get("*"), "corrupt")
                self.assertGreaterEqual(d["hard_lock_at"], lock_at)
                self.assertTrue({ga["statement"]["ent_id"], gp["statement"]["ent_id"]} <= set(d["hard_lock_seen"]))
                self.assertEqual(self.code_of(self.cache, ga), "pre_lock")
                time.sleep(1.1)
                self.cache.add(grant(self.cfg, "anibal", issued=int(time.time()) + 1))   # a fresh grant lifts anibal only
                self.assertTrue(self.cache.has("anibal"))
                self.assertFalse(self.cache.has("pxqn"))
                self.assertIn("pxqn", self.journals()[0]["hard_locks"])                  # the 403 lock is still journalled

    def test_a_restored_older_copy_is_rolled_back_and_fails_closed(self):
        ga, _gp = self.two()
        shutil.copy(self.cache.path, self.cache.path + ".bak")
        self.deny()
        self.client.sync(self.cache)
        shutil.copy(self.cache.path + ".bak", self.cache.path)   # QA F304 "rollback": the pre-lock file, gen older
        self.assertFalse(self.cache.has("anibal"))
        self.assertEqual(self.cache.status()["why"], "rolled_back")
        self.assertEqual(self.code_of(self.cache, ga), "pre_lock")

    def test_the_high_water_time_survives_a_corruption(self):
        t = [time.time() + 10 * 86400]
        fwd = self.at(t)
        fwd.add(grant(self.cfg, "anibal", issued=int(time.time())))   # F310 (b): a verified grant resets a fast seen ...
        fwd.set_locks({})                                             # ... any other write records the fast clock
        with open(self.cache.path, "w") as f:
            f.write("{garbage")
        t[0] = time.time() + 7 * 86400 - 3600                     # more than 48 h behind the high-water time
        self.assertEqual(fwd.active(), [])
        # F310 (b): a verified grant gets past the rollback guard, then the corruption lock (on the fast local clock: no
        # licd time on record) refuses it; nothing moved `seen` back
        self.assertEqual(self.code_of(fwd, grant(self.cfg, "pxqn", kind="quant-tier", issued=int(time.time()))),
                         "pre_lock")
        bad = grant(self.cfg, "pxqn", kind="quant-tier", issued=int(time.time()))
        bad["signature"] = bad["signature"][:-4] + ("AAAA" if not bad["signature"].endswith("AAAA") else "BBBB")
        self.assertEqual(self.code_of(fwd, bad), "clock_rollback")
        self.assertGreaterEqual(self.journals()[0]["seen"], time.time() + 10 * 86400 - 60)

    def test_a_garbled_journal_fails_closed_and_one_journal_is_enough(self):
        self.two()
        os.unlink(os.path.join(self.cfg, S.EntitlementCache.JOURNALS[1]))
        self.deny()
        self.client.sync(self.cache)
        os.unlink(self.cache.path)
        self.assertFalse(self.cache.has("anibal"))                  # the store journal alone keeps the lock
        self.setUp()
        self.two()
        with open(os.path.join(self.cfg, S.EntitlementCache.JOURNALS[0]), "w") as f:
            f.write("[not a journal")
        self.assertFalse(self.cache.has("anibal"))
        self.assertEqual(self.cache.status()["why"], "journal")

    def test_a_plugin_host_journal_lock_is_merged_in(self):
        self.two()
        for name in S.EntitlementCache.JOURNALS:                    # what the plugin host writes (both copies, same gen)
            p = os.path.join(self.cfg, name)
            with open(p) as f:
                j = json.load(f)
            j.update(hard_locks={"anibal": "revoked"}, hard_lock_at=float(time.time()), hard_lock_seen=[])
            with open(p, "w") as f:
                json.dump(j, f)
        self.assertFalse(self.cache.has("anibal"))                  # a lock another writer journalled counts
        self.assertTrue(self.cache.has("pxqn"))


# ---------------------------------------------- (F310-P) the lock time is bounded: earlier of body/Date, now + 24 h cap
class LockTimeF310(Base):
    def lock(self, date=None, body_time=None):
        ga = grant(self.cfg, "anibal")
        self.on_refresh(self.refresh_reply([ga]))
        self.client.sync(self.cache)
        body = {"error": "no", "code": "revoked"}
        if body_time is not None:
            body["server_time"] = body_time
        FakeLicd.script["/v1/store/refresh"] = lambda _b: (403, body)
        FakeLicd.date_header = date
        try:
            self.client.sync(self.cache)
        finally:
            FakeLicd.date_header = None
        return self.snapshot()

    def test_server_time_checked(self):
        now = time.time()
        H = lambda t: {"Date": http_date(t)}                       # noqa: E731
        self.assertEqual(S.server_time_checked({"server_time": int(now)}, H(now + 200)), (float(int(now)), "body"))
        self.assertEqual(S.server_time_checked({"server_time": int(now + 1000)}, H(now)), (float(int(now)), "date"))
        self.assertEqual(S.server_time_checked({"server_time": int(now - 1000)}, H(now)), (float(int(now - 1000)), "body"))
        self.assertEqual(S.server_time_checked({}, H(now)), (float(int(now)), "date"))
        self.assertEqual(S.server_time_checked({"server_time": int(now)}, {}), (float(int(now)), "body"))
        self.assertEqual(S.server_time_checked({}, {}), (None, None))

    def test_a_body_years_ahead_of_a_right_date_locks_at_the_date(self):
        now = time.time()
        d = self.lock(date=http_date(now), body_time=int(now + 70 * 365 * 86400))
        self.assertEqual((d["hard_lock_at"], d["hard_lock_source"], d["hard_lock_clock"]), (float(int(now)), "date", "server"))
        self.assertNotIn("hard_lock_capped", d)
        time.sleep(1.1)
        self.cache.add(grant(self.cfg, "anibal", issued=int(time.time()) + 1))
        self.assertTrue(self.cache.has("anibal"))

    def test_a_date_ahead_of_the_body_by_more_than_5_minutes_locks_at_the_body(self):
        now = time.time()
        d = self.lock(date=http_date(now + 3600), body_time=int(now))
        self.assertEqual((d["hard_lock_at"], d["hard_lock_source"]), (float(int(now)), "body"))

    def test_agreeing_body_and_date_keep_the_body(self):
        now = time.time()
        d = self.lock(date=http_date(now), body_time=int(now) - 200)
        self.assertEqual((d["hard_lock_at"], d["hard_lock_source"]), (float(int(now) - 200), "body"))

    def test_a_lock_time_is_capped_at_now_plus_24_hours_until_a_fresh_grant(self):
        now = time.time()
        d = self.lock(date=http_date(now + 30 * 86400))           # only a far-future Date (no body time)
        self.assertTrue(d["hard_lock_capped"])
        self.assertEqual(d["hard_lock_source"], "date")
        self.assertTrue(now + 86400 - 5 <= d["hard_lock_at"] <= time.time() + 86400)
        cap = d["hard_lock_at"]
        self.lock(date=http_date(now + 60 * 86400))                 # a second one never moves past the cap's rule
        self.assertLessEqual(self.snapshot()["hard_lock_at"], time.time() + 86400)
        cap = self.snapshot()["hard_lock_at"]
        g = grant(self.cfg, "anibal", issued=int(cap) + 60)         # licd's fresh grant after the capped time, on licd's clock
        self.on_refresh(self.refresh_reply([g]))
        FakeLicd.date_header = http_date(cap + 120)
        try:
            r = self.client.sync(self.cache)
        finally:
            FakeLicd.date_header = None
        self.assertEqual(r["refused"], [])
        self.assertTrue(self.cache.has("anibal"))
        d = self.snapshot()
        self.assertNotIn("hard_lock_capped", d)
        self.assertNotIn("hard_lock_at", d)

    def test_direct_hard_lock_is_capped_too(self):
        now = time.time()
        self.cache.hard_lock("revoked", ["anibal"], at=now + 5 * 86400, source="body")
        d = self.snapshot()
        self.assertTrue(d["hard_lock_capped"])
        self.assertLessEqual(d["hard_lock_at"], time.time() + 86400)

    def test_qa_insane_times_are_none(self):
        bad = [{"server_time": float("inf")}, {"server_time": float("nan")}, {"server_time": 10 ** 400},
               {"server_time": True}, {"server_time": "9999-12-31T00:00:00Z"}, {"server_time": "2026-10-19T08:00:00+99:00"},
               {"server_time": ["x"]}, {"server_time": "0001-01-01T00:00:00+14:00"}, {"server_time": -1e308}]
        dates = ["garbage", "Mon, 19 Oct 99999 08:00:00 GMT", "Mon, 19 Oct 1900 08:00:00 GMT", "Fri, 31 Dec 9999 23:59:59 +2359",
                 "Mon, 19 Oct 2026 08:00:00 +9999", "", "\x00"]
        got = [(repr(b)[:40], d, S.server_time(b, {"Date": d})) for b in bad for d in dates]
        self.assertEqual([x for x in got if x[2] is not None], [])
        self.assertEqual([x for x in ((b, d, S.server_time_checked(b, {"Date": d})) for b in bad for d in dates)
                          if x[2] != (None, None)], [])


# ------------------------------------------------------------ install.key: created once, under a lock, never overwritten
def _child_install_key(cfg, go, out):
    go.wait(10)
    out.put(S.install_key(cfg)[1])


class InstallKeyOnce(unittest.TestCase):
    def test_processes_starting_at_once_get_the_same_key(self):
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        for _round in range(5):
            cfg = tempfile.mkdtemp(dir=TMP)
            go, out = ctx.Event(), ctx.Queue()
            ps = [ctx.Process(target=_child_install_key, args=(cfg, go, out)) for _ in range(16)]
            [p.start() for p in ps]
            go.set()
            pubs = {out.get(timeout=30) for _ in ps}
            [p.join(30) for p in ps]
            self.assertEqual(len(pubs), 1, pubs)
            self.assertEqual(pubs, {S.install_key(cfg)[1]})
            d = os.path.join(cfg, "plugins")
            self.assertEqual(os.path.getsize(os.path.join(d, "install.key")), 32)
            self.assertEqual(sorted(n for n in os.listdir(d) if not n.startswith("install.key")), [])   # no temp left
            import stat
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(d, "install.key")).st_mode), 0o600)

    def test_threads_starting_at_once_get_the_same_key(self):
        cfg = tempfile.mkdtemp(dir=TMP)
        bar, pubs = threading.Barrier(12), []
        ts = [threading.Thread(target=lambda: (bar.wait(), pubs.append(S.install_key(cfg)[1]))) for _ in range(12)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(len(set(pubs)), 1)
        self.assertEqual(len(pubs), 12)

    def test_a_key_published_by_another_writer_wins_and_is_never_overwritten(self):
        cfg = tempfile.mkdtemp(dir=TMP)
        d = os.path.join(cfg, "plugins")
        os.makedirs(d)
        theirs = os.urandom(32)
        real = S._read_seed
        calls = []

        def racing(p):                                  # the plugin host writes its key between our check and our publish
            calls.append(p)
            if len(calls) == 2:
                with open(p, "wb") as f:
                    f.write(theirs)
            return real(p)
        S._read_seed = racing
        try:
            seed, _pub = S.install_key(cfg)
        finally:
            S._read_seed = real
        self.assertEqual(seed, theirs)
        with open(os.path.join(d, "install.key"), "rb") as f:
            self.assertEqual(f.read(), theirs)

    def test_a_wrong_length_key_is_replaced_once(self):
        cfg = tempfile.mkdtemp(dir=TMP)
        os.makedirs(os.path.join(cfg, "plugins"))
        with open(os.path.join(cfg, "plugins", "install.key"), "wb") as f:
            f.write(b"short")
        a = S.install_key(cfg)
        self.assertEqual(S.install_key(cfg), a)
        self.assertEqual(os.path.getsize(os.path.join(cfg, "plugins", "install.key")), 32)


class SharedFixture(unittest.TestCase):
    def test_fixture_sha256_pinned_same_as_core_and_plugins(self):
        with open(FIXTURE, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), FIXTURE_SHA256)   # change all three sides together

    def test_eight_named_vectors(self):
        fx = _load_fixture()
        self.assertEqual(fx["schema"], "pxa-entitlement-vectors/1")
        self.assertEqual([v["name"] for v in fx["vectors"]],
                         ["ok", "refresh_due", "expired", "wrong_machine", "bad_sig", "revoked", "unknown_kid", "clock"])
        self.assertNotIn(fx["kid"], S.GRANT_PUBKEYS)                                 # the DEV vector key is never pinned

    def test_vector_install_key_is_the_one_the_vectors_name(self):
        seed = hashlib.sha256(VECTOR_LABEL + b"|" + b"install-this").digest()
        self.assertEqual(S.ed25519_public(seed).hex(), _load_fixture()["vectors"][0]["install_pub"])

    def test_control_agrees_8_of_8_both_layers(self):
        res = run_fixture()
        bad = [(n, got, want) for n, ok, got, want in res if not ok]
        self.assertEqual(bad, [])
        self.assertEqual(sum(1 for r in res if r[1]), 8)
        print("\nshared fixture: Control %d/%d (verify + cache layers)" % (sum(1 for r in res if r[1]), len(res)), file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
