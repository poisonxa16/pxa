#!/usr/bin/env python3
"""Coin gifts, both halves together: PXA Control's gift routes against a REAL licd (tools/pxa-licd of
the store service tree) on 127.0.0.1 with a scratch sqlite db and keys generated here. Skips when that licd tree is
not there (PXA_LICD_DIR).

    PXA_LICD_DIR=<store tree>/tools/pxa-licd python3 tests/test-pxa-control-gift-licd.py

Covers: preview moves nothing, the confirmed gift writes both rows, the panel's split/history/statement show it, a second
confirm is refused, self-gift refused, the daily cap, a gift by email shows a masked wallet, a forged signature moves nothing.
"""
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = [os.environ.get("PXA_LICD_DIR") or "",
              os.path.join(os.path.dirname(ROOT), "pxa-licd")]   # the store service tree is not part of this repo
LICD_DIR = next((p for p in CANDIDATES if p and os.path.exists(os.path.join(p, "wallet.py"))), "")
TMP = tempfile.mkdtemp(prefix="pxa-gift-joint-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
sys.path.insert(0, os.path.join(ROOT, "tools"))
HAVE_LICD = False
if LICD_DIR:
    sys.path.insert(0, LICD_DIR)
    import licd  # noqa: E402
    import store as licd_store  # noqa: E402
    HAVE_LICD = hasattr(licd_store.StoreLogic, "gift_confirm")      # the gift routes
import pxa_control as C  # noqa: E402
from pxa_control import store as S  # noqa: E402
S.VIEW_CACHE_S = 0      # the joint test changes licd behind Control and reads the view at once (review R1 cache off)

KOFI = "kofi-test-token-joint"
TRIAGE = "triage-token-for-the-joint-test-0123456789"


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(HAVE_LICD, "licd tree with the gift routes not found (PXA_LICD_DIR)")
class GiftJoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = cls.d = os.path.join(TMP, "licd")
        keys = licd_store.keys_init(os.path.join(d, "secrets"), "ent-joint", "plug-joint")
        for name, val in (("kofi", KOFI), ("svc-triage.token", TRIAGE)):
            with open(os.path.join(d, name), "w") as f:
                f.write(val + "\n")
        os.makedirs(os.path.join(d, "files"))
        env = {"PXA_LIC_DB": os.path.join(d, "db.sqlite"), "PXA_LIC_PORT": "0", "PXA_LIC_STORE_SEED": keys["ent_seeds"]["ent-joint"],
               "PXA_LIC_STORE_KID": "ent-joint", "PXA_LIC_KOFI_TOKEN": os.path.join(d, "kofi"), "PXA_LIC_STORE_FILES_DIR": os.path.join(d, "files"),
               "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": "", "PXA_LIC_SVC_TRIAGE": os.path.join(d, "svc-triage.token")}
        cls.lsvc = licd.Service(licd.Config(env))
        cls.lsrv, _ = licd.serve(licd.Config(env), cls.lsvc)
        threading.Thread(target=cls.lsrv.serve_forever, daemon=True).start()
        cls.lurl = "http://127.0.0.1:%d" % cls.lsrv.server_address[1]
        os.environ["PXA_STORE_URL"] = cls.lurl
        S.GRANT_PUBKEYS.clear()
        S.load_pubkeys(keys["pubkeys_file"])
        cls.app = C.App(C.desktop.load_launcher(), port=7777, models_dirs=[os.path.join(TMP, "models")])
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PXA_STORE_URL", None)
        for s in (cls.srv, cls.lsrv):
            s.shutdown()
            s.server_close()

    def ctl(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data, method=method,
                                   headers={"X-PXA-Token": self.app.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def licd(self, method, path, body=None, headers=None, raw=None, ctype="application/json"):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        r = urllib.request.Request(self.lurl + path, data=data, method=method, headers=dict(headers or {}))
        if data is not None:
            r.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def kofi(self, txn, amount, message):
        d = {"verification_token": KOFI, "kofi_transaction_id": txn, "message_id": "m-" + txn, "type": "Donation", "amount": amount,
             "currency": "USD", "email": "fan@example.org", "from_name": "Fan", "message": message}
        return self.licd("POST", "/v1/hooks/kofi", raw=urllib.parse.urlencode({"data": json.dumps(d)}).encode(),
                         ctype="application/x-www-form-urlencoded")

    def award(self, body):
        return self.licd("POST", "/v1/svc/award", body, headers={"Authorization": "Bearer " + TRIAGE})

    def spend_rows(self, wid):
        return self.lsvc.db.q1("SELECT COUNT(*) n FROM store_ledger l JOIN store_accounts a ON a.id=l.account_id "
                               "WHERE a.wallet_id=? AND l.kind='spend'", (wid,))["n"]

    def other_wallet(self):
        """a second supporter's wallet, made through licd's own machine door (another install key, not this Control)."""
        import ed25519
        seed = os.urandom(32)
        pub = ed25519.public_key(seed).hex()
        return self.lsvc.store.wallet_create("machine", pub, "10.9.9.9")["wallet_id"]

    def rows(self, kind):
        return [tuple(r) for r in self.lsvc.db.q("SELECT a.wallet_id, l.delta FROM store_ledger l JOIN store_accounts a ON a.id=l.account_id WHERE l.kind=? ORDER BY l.id", (kind,))]

    def test_gift_path(self):
        st, made = self.ctl("POST", "/api/store/wallet/create", {})
        wid = made["wallet_id"]
        other = self.other_wallet()
        # not a supporter yet: no donation, no Valued key
        st, j = self.ctl("POST", "/api/store/wallet/gift-start", {"to": other, "coins": "10"})
        self.assertEqual((st, j.get("code")), (403, "gift_supporters"), j)
        st, k = self.kofi("txn-gift-1", "20.00", "for " + wid)
        self.assertEqual((st, k.get("coins")), (200, 200), k)

        # 1. preview: nothing moves
        st, p = self.ctl("POST", "/api/store/wallet/gift-start", {"to": other, "coins": "30"})
        self.assertEqual((st, p["to"], p["coins"], p["balance_after"], p["cap"], p["used_today"]), (200, other, 30, 170, 500, 0), p)
        self.assertRegex(p["gift"], r"^PXG-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}$")
        self.assertEqual(self.rows("gift"), [])
        # 2. confirm: both rows in one go; the panel shows the gift, the statement signs it
        st, r = self.ctl("POST", "/api/store/wallet/gift-confirm", {"gift": p["gift"]})
        self.assertEqual((st, r["sent"], r["coins"], r["wallet"]["coins"], r["wallet"]["gifts"]), (200, True, 30, 170, -30), r)
        self.assertEqual(self.rows("gift"), [(wid, -30), (other, 30)])
        st, v = self.ctl("GET", "/api/store/wallet")
        self.assertEqual((v["wallet"]["coins"], v["wallet"]["history"][0]["what"], v["wallet"]["history"][0]["delta"]), (170, "Gift to " + other, -30))
        st, stmt = self.ctl("POST", "/api/store/wallet/statement", {})
        self.assertEqual((S.verify_statement(stmt)["coins"], stmt["statement"]["gifts"]), (170, -30))
        self.assertEqual(S.wallet_from_statement(stmt["statement"])["gifts"], -30)                 # the offline view shows the gift too
        # 3. the same code again moves nothing
        st, again = self.ctl("POST", "/api/store/wallet/gift-confirm", {"gift": p["gift"]})
        self.assertEqual((st, again["code"]), (409, "gift_done"))
        self.assertEqual(len(self.rows("gift")), 2)
        # 4. own wallet, bad input, the daily cap
        self.assertEqual(self.ctl("POST", "/api/store/wallet/gift-start", {"to": wid, "coins": "5"})[1]["code"], "gift_self")
        self.assertEqual(self.ctl("POST", "/api/store/wallet/gift-start", {"to": "PXW-0000-0000-000X", "coins": "5"})[1]["code"], "gift_to")
        self.lsvc.store.set_rules("admin", {"gift_day_cap": 50})
        st, cap = self.ctl("POST", "/api/store/wallet/gift-start", {"to": other, "coins": "21"})
        self.assertEqual((st, cap["code"]), (429, "gift_cap"))
        # 5. by email: the wallet linked to that Ko-fi email, shown masked
        st_ = self.lsvc.store
        with self.lsvc.db.tx():
            st_._bind_payer(st_.wallet_by_id(other), st_.email_hmac("friend@example.org"), "email-code", "test")
        st, pe = self.ctl("POST", "/api/store/wallet/gift-start", {"to": "friend@example.org", "coins": "20"})
        self.assertEqual((st, pe["to"], pe["to_how"]), (200, "PXW-....-....-" + other[-4:], "email"), pe)
        # 6. a forged signature on the confirm moves nothing
        real = S.store_auth

        def forged(cfg_dir, op, nonce, args):
            a = real(cfg_dir, op, nonce, args)
            sig = bytearray(S._b64d(a["sig"]))
            sig[0] ^= 0x01
            return dict(a, sig=S.b64u(bytes(sig)))
        S.store_auth = forged
        try:
            st, bad = self.ctl("POST", "/api/store/wallet/gift-confirm", {"gift": pe["gift"]})
        finally:
            S.store_auth = real
        self.assertEqual((st, bad["code"]), (401, "bad_signature"))
        self.assertEqual(len(self.rows("gift")), 2)
        st, ok = self.ctl("POST", "/api/store/wallet/gift-confirm", {"gift": pe["gift"]})
        self.assertEqual((st, ok["to"], ok["wallet"]["coins"]), (200, "PXW-....-....-" + other[-4:], 150), ok)
        self.assertEqual(st_.balances(st_.wallet_by_id(other)["id"])["coins"], 50)
        self.assertTrue(st_.ledger_verify()["ok"])


if __name__ == "__main__":
    unittest.main()
