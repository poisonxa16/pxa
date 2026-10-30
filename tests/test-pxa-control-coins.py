#!/usr/bin/env python3
"""PXA Coins wallet panel in Control, against a fake licd (no network, no production store).

The fake speaks the wallet API Control already signs: POST /v1/store/nonce and POST /v1/wallet/<action>
(view, create, statement, redeem). The panel's page files are checked for the balance split, history,
ways to earn, redeem, both themes (token colours only) and a phone width.

    python3 tests/test-pxa-control-coins.py
"""
import json
import os
import re
import stat
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
UI = os.path.join(ROOT, "tools", "pxa_control_ui")
TMP = tempfile.mkdtemp(prefix="pxa-coins-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
sys.path.insert(0, TOOLS)
import pxa_control as C  # noqa: E402
from pxa_control import store as S  # noqa: E402

ITEMS = {
    "anibal": {"id": "anibal", "kind": "plugin", "name": "Anibal", "tagline": "A supporter plugin",
               "description": "Unlock Anibal.", "price": 100, "permissions": []},
    "pxqn": {"id": "pxqn", "kind": "quant-tier", "name": "Own PXQN encodes", "tagline": "Make your own PXQN files",
             "description": "A supporter reward.", "price": 100, "permissions": []},
}
BANNED = re.compile(r"\b(paid|sell|buy|price|PXA_[A-Z0-9_]+)\b")
EARN_ROWS = [{"what": "Donation", "coins": "12 per 1 US dollar", "note": "No monthly limit."},
             {"what": "Confirmed bug report, small", "coins": 11, "note": "The first report only."},
             {"what": "Monthly earn limit", "coins": 180, "note": "Resets on the 1st (UTC)."}]


LICD_ARGS = {"claim": ("code",)}   # licd's own signed fields (WALLET_ARGS), so the fake does not just echo Control's list
AUTH_KEYS = ("install_pub", "nonce", "sig")


def balances(rows):
    k = {}
    for r in rows:
        k[r["kind"]] = k.get(r["kind"], 0) + int(r["delta"])
    return {"coins": sum(k.values()), "from_donations": k.get("donation", 0) + k.get("migration", 0),
            "earned": k.get("earn", 0), "redeemed": -k.get("spend", 0), "reversed": -k.get("reversal", 0),
            "adjusted": k.get("admin", 0)}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self.server.box.handle(self, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except ValueError:
            body = {}
        self.server.box.handle(self, body if isinstance(body, dict) else {})


class FakeLicd(object):
    """the slice of licd the wallet panel calls. Signatures are checked; amounts live in memory."""

    def __init__(self):
        self.lock = threading.Lock()
        self.nonces = {}
        self.by_pub = {}
        self.seed = os.urandom(32)
        self.kid = "coins-test"
        self.pub = S.ed25519_public(self.seed).hex()
        self.claims = []                # the body of every signed claim, minus the signature fields

    def serve(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        httpd.box = self
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.httpd = httpd
        return "http://127.0.0.1:%d" % httpd.server_address[1]

    def stop(self):
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        self.httpd.shutdown()
        self.httpd.server_close()

    def credit(self, wallet_id, delta, kind, what):
        with self.lock:
            w = next(v for v in self.by_pub.values() if v["wallet_id"] == wallet_id)
            w["rows"].append({"ts": 1_760_500_000 + len(w["rows"]), "delta": int(delta), "kind": kind, "what": what})

    def _send(self, h, code, obj):
        raw = json.dumps(obj).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)

    def _sign(self, prefix, st):
        return S.b64u(S.ed25519_sign(self.seed, prefix + S.canonical_ascii(st)))

    def _view(self, w):
        rows = list(reversed(w["rows"]))
        b = balances(w["rows"])
        return {"wallet_id": w["wallet_id"], **b,
                "history": [{"ts": r["ts"], "delta": r["delta"], "what": r["what"]} for r in rows],
                "owned": [{"item_id": i, "name": ITEMS[i]["name"], "kind": ITEMS[i]["kind"], "since": 1, "status": "active"} for i in w["owned"]],
                "kofi_linked": False, "discord_linked": False, "computers": 1,
                "valued_supporter": False,
                "beta": {"release": "4.0-beta.2", "access": w["beta"], "how": "redeemed" if w["beta"] else None,
                         "item_id": "beta-4.0-beta.2", "coins": 0 if w["beta"] else 50},
                "door": "machine"}

    def _auth(self, body, action):
        op = "wallet-" + action
        args = {k: str(body.get(k) or "").strip() for k in LICD_ARGS.get(action) or S.LicdClient.WALLET_ACTIONS[action]}
        pub, nonce = str(body.get("install_pub") or ""), str(body.get("nonce") or "")
        sig = S._b64d(body.get("sig"))
        msg = S.INSTALL_PREFIX + S.canonical_ascii({"op": op, "nonce": nonce, "args": args})
        if not re.fullmatch(r"[0-9a-fA-F]{64}", pub) or len(sig) != 64 or not S.ed25519_verify(bytes.fromhex(pub), msg, sig):
            return None
        with self.lock:
            got = self.nonces.pop(nonce, None)
        if got != (pub, op):
            return None
        return pub, args

    def handle(self, h, body):
        path = h.path.split("?", 1)[0]
        if h.command == "GET" and path == "/v1/store/catalog":
            st = {"v": 1, "iss": "licd", "kid": self.kid, "type": "catalog", "issued": 1_760_500_000,
                  "currency": {"name": "coins", "source": "https://ko-fi.com/shatteredrealms1",
                               "rate_note": "thank-you coins: a one-off donation gives 10 per US dollar"},
                  "items": list(ITEMS.values())}
            self._send(h, 200, {"catalog": st, "signature": self._sign(S.CATALOG_PREFIX, st)})
            return
        if h.command == "GET" and path == "/v1/wallet/earn-table":
            self._send(h, 200, {"v": 1, "rows": EARN_ROWS})
            return
        if h.command == "POST" and path == "/v1/store/nonce":
            nonce = os.urandom(16).hex()
            with self.lock:
                self.nonces[nonce] = (str(body.get("install_pub") or ""), str(body.get("op") or ""))
            self._send(h, 200, {"nonce": nonce})
            return
        if h.command == "POST" and path.startswith("/v1/wallet/"):
            action = path[len("/v1/wallet/"):]
            if action not in S.LicdClient.WALLET_ACTIONS and action not in LICD_ARGS:
                self._send(h, 404, {"error": "not found", "code": "not_found"})
                return
            who = self._auth(body, action)
            if who is None:
                self._send(h, 401, {"error": "bad signature", "code": "bad_signature"})
                return
            pub, args = who
            if action == "claim":
                with self.lock:
                    self.claims.append({k: v for k, v in body.items() if k not in AUTH_KEYS})
            self._send(h, *self._action(pub, action, args))
            return
        self._send(h, 404, {"error": "not found", "code": "not_found"})

    def _action(self, pub, action, args):
        with self.lock:
            w = self.by_pub.get(pub)
            if action == "create":
                made = w is None
                if made:
                    w = {"wallet_id": "PXW-K7M3-Q9TD-4HXR", "rows": [], "owned": [], "beta": False}
                    self.by_pub[pub] = w
                return 200, dict(self._view(w), created=made)
            if action == "claim":                         # licd: the code is checked first, then a machine without a wallet gets one
                if args["code"] != "PXK-AB12-CD34-EF56":
                    return 404, {"error": "that code is not valid or was already used", "code": "bad_code"}
                made = w is None
                if made:
                    w = {"wallet_id": "PXW-K7M3-Q9TD-4HXR", "rows": [], "owned": [], "beta": False}
                    self.by_pub[pub] = w
                for i in (1, 2):
                    w["rows"].append({"ts": 1_760_500_000 + i, "delta": 120, "kind": "donation", "what": "Ko-fi donation"})
                return 200, {"claimed": True, "credited": 2, "review": 1, "created": made, "wallet": self._view(w)}
            if w is None:
                return 404, {"error": "no wallet yet: create one first", "code": "no_wallet"}
            if action == "view":
                return 200, self._view(w)
            if action == "statement":
                b = balances(w["rows"])
                st = {"v": 1, "iss": "licd", "kid": self.kid, "type": "wallet-statement", "wallet": w["wallet_id"],
                      "issued": 1_760_500_100, **b, "last_id": len(w["rows"]), "last_hash": "0" * 64,
                      "owned": sorted(w["owned"])}
                return 200, {"statement": st, "signature": self._sign(S.STMT_PREFIX, st)}
            if action == "redeem":
                it = ITEMS.get(args.get("item_id") or "")
                if not it:
                    return 404, {"error": "no such reward", "code": "not_found"}
                if args["item_id"] in w["owned"]:
                    return 200, {"item_id": it["id"], "redeemed": False, "coins_used": 0, "wallet": self._view(w)}
                if balances(w["rows"])["coins"] < it["price"]:
                    return 402, {"error": "not enough coins", "code": "not_enough_coins"}
                w["rows"].append({"ts": 1_760_500_050, "delta": -it["price"], "kind": "spend", "what": "Redeemed: " + it["name"]})
                w["owned"].append(it["id"])
                return 200, {"item_id": it["id"], "redeemed": True, "coins_used": it["price"], "wallet": self._view(w)}
            if action == "beta":
                return 200, {"release": "4.0-beta.2", "redeemed": False, "wallet": self._view(w)}
            if action == "report-link":
                return 200, {"code": "PXR-AB12-CD34", "expires": 1_760_500_000 + 7 * 86400, "wallet_id": w["wallet_id"]}
            return 200, self._view(w)


S.VIEW_CACHE_S = 0          # these tests change the fake licd behind Control's back and read the view at once (review R1 cache off; ZReviewFixes turns it on)


class WalletPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeLicd()
        os.environ["PXA_STORE_URL"] = cls.fake.serve()
        S.GRANT_PUBKEYS[cls.fake.kid] = cls.fake.pub
        cls.app = C.App(C.desktop.load_launcher(), port=7777, models_dirs=[os.path.join(TMP, "models")])
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        os.environ["PXA_STORE_URL"] = ""
        try:
            cls.srv.shutdown()
            cls.srv.server_close()
        except Exception:
            pass
        try:
            cls.fake.stop()
        except Exception:
            pass

    def ctl(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data, method=method,
                                     headers={"X-PXA-Token": self.app.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_balance_history_redeem_and_the_signed_statement(self):
        st, w = self.ctl("GET", "/api/store/wallet")
        self.assertEqual((st, w["wallet"], w["offline"]), (200, None, False))
        self.assertEqual(w["earn"], EARN_ROWS)                        # the live earn table (licd's rules), not Control's copy
        st, cat = self.ctl("GET", "/api/store/catalog")
        self.assertEqual((st, cat["source"], cat["currency"]["name"]), (200, "licd", "coins"))
        self.assertEqual({i["id"]: i["price"] for i in cat["items"]}, {"anibal": 100, "pxqn": 100})

        st, made = self.ctl("POST", "/api/store/wallet/create", {})
        self.assertEqual(st, 200, made)
        self.assertTrue(made["created"])
        wid = made["wallet_id"]
        self.assertEqual(wid, "PXW-K7M3-Q9TD-4HXR")
        self.assertEqual(made["coins"], 0)
        S.STMT_GAP_S = 0.0                                              # N3: the statement follows the wallet without anyone pressing Save
        cfg = os.environ["PXA_CONTROL_CONFIG_DIR"]
        self.assertEqual((S.load_statement(cfg) or {}).get("coins"), 0)  # written when the wallet was created

        self.fake.credit(wid, 100, "donation", "Ko-fi donation")
        self.fake.credit(wid, 30, "earn", "Bug report confirmed")
        st, view = self.ctl("GET", "/api/store/wallet")
        w = view["wallet"]
        self.assertEqual(S.load_statement(cfg)["coins"], 130)           # N3: a live read that differs from the saved statement refreshes it
        self.assertEqual((w["coins"], w["from_donations"], w["earned"], w["redeemed"]), (130, 100, 30, 0))
        self.assertEqual([(r["delta"], r["what"]) for r in w["history"]], [(30, "Bug report confirmed"), (100, "Ko-fi donation")])

        st, r = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "anibal"})
        self.assertEqual((st, r["redeemed"], r["coins_used"], r["wallet"]["coins"], r["wallet"]["redeemed"]), (200, True, 100, 30, 100))
        self.assertEqual(r["wallet"]["owned"][0]["item_id"], "anibal")
        saved = S.load_statement(cfg)                                    # N3: a redeem refreshes the statement at once
        self.assertEqual((saved["coins"], saved["owned"]), (30, ["anibal"]))
        st, again = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "anibal"})
        self.assertEqual((st, again["redeemed"], again["wallet"]["coins"]), (200, False, 30))
        st, short = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "pxqn"})
        self.assertEqual((st, short["code"]), (402, "not_enough_coins"))

        st, link = self.ctl("POST", "/api/store/wallet/report-link", {})
        self.assertEqual((st, link["code"], link["wallet_id"]), (200, "PXR-AB12-CD34", wid))
        st, stmt = self.ctl("POST", "/api/store/wallet/statement", {})
        self.assertEqual(st, 200, stmt)
        checked = S.verify_statement(stmt)
        self.assertEqual((checked["wallet"], checked["coins"], checked["from_donations"], checked["earned"], checked["redeemed"]),
                         (wid, 30, 100, 30, 100))
        path = os.path.join(os.environ["PXA_CONTROL_CONFIG_DIR"], "store", "statement.json")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        bad = dict(stmt, signature=("A" if not stmt["signature"].startswith("A") else "B") + stmt["signature"][1:])   # first char: 6 real bits (the last one can be padding only: flaky)
        with self.assertRaises(C.Invalid):
            S.verify_statement(bad)

        # N3: a statement of another wallet never stands in for this one: it is dropped when the live wallet is a different one
        class Down(object):
            def wallet(self, action, args=None):
                raise OSError("store down")
        self.assertFalse(S.keep_statement(Down(), cfg, view={"wallet_id": "PXW-OTHER-WALLET", "coins": 5, "owned": []}, force=True))
        self.assertIsNone(S.load_statement(cfg))
        S.save_statement(cfg, stmt)                                      # (put it back for the offline checks below)
        # the same wallet, same numbers: nothing to read; a different balance with the store down keeps the old copy
        self.assertFalse(S.keep_statement(Down(), cfg, view={"wallet_id": wid, "coins": 30, "owned": [{"item_id": "anibal", "status": "active"}]}))
        self.assertFalse(S.keep_statement(Down(), cfg, view={"wallet_id": wid, "coins": 31, "owned": [{"item_id": "anibal", "status": "active"}]}, force=True))
        self.assertEqual(S.load_statement(cfg)["coins"], 30)
        self.fake.stop()
        st, off = self.ctl("GET", "/api/store/wallet")
        self.assertEqual(st, 200, off)
        self.assertTrue(off["offline"])
        self.assertEqual((off["wallet"]["coins"], off["wallet"]["from_donations"], off["wallet"]["earned"], off["wallet"]["redeemed"], off["wallet"]["history"]),
                         (30, 100, 30, 100, []))
        self.assertEqual(off["as_of"], 1_760_500_100)
        self.assertEqual(off["earn"], EARN_ROWS)                      # the last good table stands while the store is down
        with open(path) as f:
            saved = json.load(f)
        saved["signature"] = "A" * 80
        with open(path, "w") as f:
            json.dump(saved, f)
        st, gone = self.ctl("GET", "/api/store/wallet")
        self.assertEqual((st, gone["wallet"], gone["offline"]), (200, None, True))

    def test_the_page_has_the_panel_for_both_themes_and_a_phone(self):
        with open(os.path.join(UI, "wallet.js"), encoding="utf-8") as f:
            js = f.read()
        with open(os.path.join(UI, "wallet.css"), encoding="utf-8") as f:
            css = f.read()
        with open(os.path.join(UI, "index.html"), encoding="utf-8") as f:
            page = f.read()
        for needle in ("PXA Coins", "From donations", "Ways to earn", "Unlock for ", "Save a signed statement",
                       "Create my wallet", "10 coins per $1", "150 coins", "thank-you points", "Get a report code", "earnRows()",
                       "Ko-fi code", "Code from the Ko-fi email", "Add my coins", 'post("claim", {code: c})',
                       "Later Ko-fi donations arrive by themselves."):
            self.assertIn(needle, js, needle)
        self.assertIsNone(BANNED.search(js), BANNED.search(js))
        self.assertIsNone(BANNED.search(css), BANNED.search(css))
        self.assertNotIn("credit", js.lower())
        self.assertIn("@media (max-width:820px)", css)
        self.assertIn("var(--text)", css)
        self.assertIn("var(--ok)", css)
        self.assertIsNone(re.search(r"#[0-9A-Fa-f]{3,8}\b", css))
        self.assertIn('id="wl-root"', page)
        self.assertIn('href="/wallet.css"', page)
        self.assertIn('src="/wallet.js"', page)
        self.assertIn("[data-theme=light]", page)
        code, body = self._static("/wallet.js")
        self.assertEqual(code, 200)
        self.assertIn(b"PXA Coins", body)
        code, body = self._static("/wallet.css")
        self.assertEqual(code, 200)
        self.assertIn(b"max-width:820px", body)

    def _static(self, path):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


class ZKofiCode(unittest.TestCase):
    """PXA first (KOFI-PXA-FIRST Task 8): the private code from the Ko-fi email, typed into Control. Own fake licd and Control (runs
    after WalletPanel); this install has no wallet in the new fake, so the claim makes one."""
    setUpClass = classmethod(WalletPanel.setUpClass.__func__)
    tearDownClass = classmethod(WalletPanel.tearDownClass.__func__)
    ctl = WalletPanel.ctl

    def test_claim_signs_code_and_posts(self):
        st, bad = self.ctl("POST", "/api/store/wallet/claim", {"code": "PXK-ZZZZ-ZZZZ-ZZZZ"})
        self.assertEqual((st, bad.get("code")), (404, "bad_code"), bad)
        st, r = self.ctl("POST", "/api/store/wallet/claim", {"code": " PXK-AB12-CD34-EF56 ", "email": "someone@example.com"})
        self.assertEqual(st, 200, r)
        self.assertEqual((r["claimed"], r["payments"], r["review"], r["created"]), (True, 2, 1, True))
        self.assertEqual(self.fake.claims, [{"code": "PXK-ZZZZ-ZZZZ-ZZZZ"}, {"code": "PXK-AB12-CD34-EF56"}])   # signed; only the code leaves
        st, w = self.ctl("GET", "/api/store/wallet")
        self.assertEqual((st, w["wallet"]["wallet_id"], w["wallet"]["coins"]), (200, "PXW-K7M3-Q9TD-4HXR", 240))
        stmt = S.load_statement(os.environ["PXA_CONTROL_CONFIG_DIR"])  # the offline balance follows the claim
        self.assertEqual((stmt["wallet"], stmt["coins"]), ("PXW-K7M3-Q9TD-4HXR", 240))


class ZReviewFixes(unittest.TestCase):
    """review of the coins delta before beta 2: R1 one licd read for every tab, R7 a statement error never
    breaks the view, R8 a failed join keeps the saved statement. Own fake licd and Control (runs after WalletPanel)."""
    setUpClass = classmethod(WalletPanel.setUpClass.__func__)
    tearDownClass = classmethod(WalletPanel.tearDownClass.__func__)
    ctl = WalletPanel.ctl

    def test_review_r1_r7_r8(self):
        from unittest import mock
        cfg = os.environ["PXA_CONTROL_CONFIG_DIR"]
        S._VIEW.clear()
        S.VIEW_CACHE_S = 20
        try:
            st, made = self.ctl("POST", "/api/store/wallet/create", {})
            self.assertEqual(st, 200, made)
            calls = []
            orig = S.LicdClient.wallet

            def counted(cl, action, args=None):
                calls.append(action)
                return orig(cl, action, args)
            with mock.patch.object(S.LicdClient, "wallet", counted):
                for _ in range(3):                                            # three tabs polling
                    st, v = self.ctl("GET", "/api/store/wallet")
                    self.assertEqual((st, v["wallet"]["wallet_id"]), (200, made["wallet_id"]))
                self.assertEqual(calls.count("view"), 1)
                self.assertEqual(self.ctl("POST", "/api/store/wallet/statement", {})[0], 200)   # an action clears the cache
                self.ctl("GET", "/api/store/wallet")
                self.assertEqual(calls.count("view"), 2)
            S._VIEW.clear()
            with mock.patch.object(S, "keep_statement", side_effect=RuntimeError("tampered file")):
                st, v = self.ctl("GET", "/api/store/wallet")                  # R7
                self.assertEqual((st, v["wallet"]["wallet_id"]), (200, made["wallet_id"]))
            self.assertIsNotNone(S.load_statement(cfg))
            def wrong_code(cl, action, args=None):
                raise S._invalid("that join code is not valid", 400, "bad_code")
            with mock.patch.object(S.LicdClient, "wallet", wrong_code):
                st, _j = self.ctl("POST", "/api/store/wallet/join", {"code": "PXJ-WRONG"})
            self.assertEqual(st, 400)
            self.assertIsNotNone(S.load_statement(cfg))                       # R8: the failed join left the offline balance alone
        finally:
            S.VIEW_CACHE_S = 0
            S._VIEW.clear()


if __name__ == "__main__":
    unittest.main()
