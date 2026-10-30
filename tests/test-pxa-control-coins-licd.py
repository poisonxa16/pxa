#!/usr/bin/env python3
"""PXA Coins, both halves together: PXA Control's wallet routes against a REAL licd (tools/pxa-licd from the companion tree
licd/coins) on 127.0.0.1 with a scratch sqlite db and keys generated here (never the store's). Skips when the licd tree is
not there (PXA_LICD_DIR).

    PXA_LICD_DIR=/path/to/licd/tools/pxa-licd python3 tests/test-pxa-control-coins-licd.py

Covers: earn (Ko-fi donation and a confirmed report through the award route, a report code made in Control), the balance split
and history, redeem a reward (one spend row), a second redeem is not a second spend, over-balance refused, a bad install-key
signature refused, a replayed nonce refused, the per-wallet redeem limit, the signed statement offline, a chargeback hold, and
PXA first: the private Ko-fi code licd mails (its file mailer here) claimed in Control.
"""
import json
import glob
import os
import re
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PXA_LICD_DIR names the licd tree; else a sibling checkout of this tree that carries tools/pxa-licd
CANDIDATES = [os.environ.get("PXA_LICD_DIR") or ""] + sorted(glob.glob(os.path.join(os.path.dirname(ROOT), "*", "tools", "pxa-licd")))
LICD_DIR = next((p for p in CANDIDATES if p and os.path.exists(os.path.join(p, "wallet.py"))), "")
TMP = tempfile.mkdtemp(prefix="pxa-coins-joint-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
sys.path.insert(0, os.path.join(ROOT, "tools"))
HAVE_LICD = HAVE_CLAIM = False
if LICD_DIR:
    sys.path.insert(0, LICD_DIR)
    import licd  # noqa: E402
    import store as licd_store  # noqa: E402
    HAVE_LICD = hasattr(licd_store.StoreLogic, "award")      # the coins award routes (licd/coins)
    HAVE_CLAIM = hasattr(licd_store.StoreLogic, "claim")     # the Ko-fi codes (KOFI-PXA-FIRST, proj/licd)
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


@unittest.skipUnless(HAVE_LICD, "licd tree with the coins award routes not found (PXA_LICD_DIR)")
class CoinsJoint(unittest.TestCase):
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
               "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": "", "PXA_LIC_SVC_TRIAGE": os.path.join(d, "svc-triage.token"),
               "PXA_LIC_MAILER": "file:" + os.path.join(d, "mail")}
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

    def kofi(self, txn, amount, message, email="fan@example.org"):
        d = {"verification_token": KOFI, "kofi_transaction_id": txn, "message_id": "m-" + txn, "type": "Donation", "amount": amount,
             "currency": "USD", "email": email, "from_name": "Fan", "message": message}
        return self.licd("POST", "/v1/hooks/kofi", raw=urllib.parse.urlencode({"data": json.dumps(d)}).encode(),
                         ctype="application/x-www-form-urlencoded")

    def award(self, body):
        return self.licd("POST", "/v1/svc/award", body, headers={"Authorization": "Bearer " + TRIAGE})

    def spend_rows(self, wid):
        return self.lsvc.db.q1("SELECT COUNT(*) n FROM store_ledger l JOIN store_accounts a ON a.id=l.account_id "
                               "WHERE a.wallet_id=? AND l.kind='spend'", (wid,))["n"]

    def test_the_whole_coins_path(self):
        # 1. no wallet yet; the earn table is licd's, not Control's copy
        st, w = self.ctl("GET", "/api/store/wallet")
        self.assertEqual((st, w["wallet"], w["offline"]), (200, None, False))
        rows = {r["what"]: r["coins"] for r in w["earn"]}
        self.assertEqual((rows["Donation"], rows["Monthly earn limit"]), ("10 per 1 US dollar", 150))
        st, cat = self.ctl("GET", "/api/store/catalog")
        self.assertEqual((st, cat["source"], cat["currency"]["name"]), (200, "licd", "coins"))
        cost = {i["id"]: int(i["price"]) for i in cat["items"]}
        self.assertEqual((cost["anibal"], cost["pxqn"]), (100, 100))

        # 2. create, then earn: a Ko-fi donation with the wallet id in the message, and a confirmed report through the award route
        st, made = self.ctl("POST", "/api/store/wallet/create", {})
        self.assertEqual((st, made["created"], made["coins"]), (200, True, 0), made)
        wid = made["wallet_id"]
        self.assertRegex(wid, r"^PXW-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}$")
        st, k = self.kofi("txn-joint-1", "10.00", "thanks " + wid)
        self.assertEqual((st, k.get("result"), k.get("coins")), (200, "credited", 100), k)
        st, aw = self.award({"source": "bug", "class": "correctness", "bug_id": "joint-1", "wallet": wid})
        self.assertEqual((st, aw["status"], aw["granted"], aw["new"]), (200, "credited", 30, True), aw)
        st, v = self.ctl("GET", "/api/store/wallet")
        w = v["wallet"]
        self.assertEqual((w["wallet_id"], w["coins"], w["from_donations"], w["earned"], w["redeemed"]), (wid, 130, 100, 30, 0))
        self.assertEqual([r["delta"] for r in w["history"]], [30, 100])          # newest first
        self.assertTrue(all(r["what"] and r["ts"] for r in w["history"]))

        # 3. a report code made in Control names the wallet on a report; the award lands without a wallet id
        st, link = self.ctl("POST", "/api/store/wallet/report-link", {})
        self.assertEqual(st, 200, link)
        self.assertRegex(link["code"], r"^PXR-[0-9A-Z]{4}-[0-9A-Z]{4}$")
        st, aw2 = self.award({"source": "bug", "class": "small", "bug_id": "joint-2", "link_code": link["code"]})
        self.assertEqual((aw2["wallet"], aw2["granted"], aw2["balance"]), (wid, 10, 140), aw2)
        st, used = self.award({"source": "bug", "class": "small", "bug_id": "joint-3", "link_code": link["code"]})
        self.assertEqual((st, used["code"]), (400, "bad_code"))                   # one use

        # 4. redeem a reward: one spend row, a grant for this computer; a second redeem is not a second spend
        st, r = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "anibal"})
        self.assertEqual((st, r["redeemed"], r["coins_used"], r["wallet"]["coins"], r["wallet"]["redeemed"]), (200, True, 100, 40, 100), r)
        self.assertEqual(r["synced"]["grants"], 1)
        self.assertEqual(self.spend_rows(wid), 1)
        st, again = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "anibal"})
        self.assertEqual((st, again["redeemed"], again["coins_used"], again["wallet"]["coins"]), (200, False, 0, 40))
        self.assertEqual(self.spend_rows(wid), 1)
        # 5. over the balance
        st, short = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "pxqn"})
        self.assertEqual((st, short["code"]), (402, "not_enough_coins"))
        self.assertEqual((self.spend_rows(wid), self.ctl("GET", "/api/store/wallet")[1]["wallet"]["coins"]), (1, 40))
        self.assertTrue(self.lsvc.store.ledger_verify()["ok"])

        # 6. a bad install-key signature is refused by licd and nothing moves
        real = S.store_auth

        def forged(cfg_dir, op, nonce, args):
            a = real(cfg_dir, op, nonce, args)
            sig = bytearray(S._b64d(a["sig"]))
            sig[0] ^= 0x01
            return dict(a, sig=S.b64u(bytes(sig)))
        S.store_auth = forged
        try:
            st, bad = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "pxqn"})
        finally:
            S.store_auth = real
        self.assertEqual((st, bad["code"]), (401, "bad_signature"), bad)
        # 7. a replayed call (same nonce, same signature) is refused
        _seed, pub = S.install_key(os.environ["PXA_CONTROL_CONFIG_DIR"])
        st, n = self.licd("POST", "/v1/store/nonce", {"install_pub": pub, "op": "wallet-view"})
        body = dict(S.store_auth(os.environ["PXA_CONTROL_CONFIG_DIR"], "wallet-view", n["nonce"], {}))
        self.assertEqual(self.licd("POST", "/v1/wallet/view", body)[0], 200)
        st, rep = self.licd("POST", "/v1/wallet/view", body)
        self.assertEqual((st, rep["code"]), (401, "bad_nonce"))

        # 8. the signed statement: verified, kept 0600, and the balance the panel shows with the store down
        st, stmt = self.ctl("POST", "/api/store/wallet/statement", {})
        self.assertEqual(st, 200, stmt)
        checked = S.verify_statement(stmt)
        self.assertEqual((checked["wallet"], checked["coins"], checked["earned"], checked["redeemed"], checked["owned"]), (wid, 40, 40, 100, ["anibal"]))
        path = os.path.join(os.environ["PXA_CONTROL_CONFIG_DIR"], "store", "statement.json")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        os.environ["PXA_STORE_URL"] = "http://127.0.0.1:%d" % _closed_port()
        try:
            st, off = self.ctl("GET", "/api/store/wallet")
        finally:
            os.environ["PXA_STORE_URL"] = self.lurl
        self.assertEqual((st, off["offline"], off["wallet"]["coins"], off["wallet"]["history"], off["as_of"]), (200, True, 40, [], checked["issued"]))
        self.assertEqual({r["what"] for r in off["earn"]} >= {"Donation", "Monthly earn limit"}, True)

        # 9. the per-wallet redeem limit reaches Control as a 429 (30 an hour; 4 redeem calls so far)
        codes = []
        for _ in range(30):
            st, j = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "anibal"})
            codes.append(st)
            if st == 429:
                break
        self.assertIn(429, codes, codes)
        self.assertEqual(codes.count(429), 1)
        self.assertEqual(self.spend_rows(wid), 1)

        # 10. a chargeback below zero holds the wallet: no new unlock, the grant and the view stay
        self.lsvc.store.deposit("admin", wid, -100, "chargeback (joint test)")
        self.lsvc._rl_tab = {}                                                    # the limiter, not the hold, is what step 9 tested
        st, held = self.ctl("GET", "/api/store/wallet")
        self.assertEqual((st, held["wallet"]["coins"], held["wallet"]["reversed"]), (200, -60, 100))
        st, no = self.ctl("POST", "/api/store/wallet/redeem", {"item_id": "pxqn"})
        self.assertEqual((st, no["code"]), (402, "below_zero"))
        st, ref = self.ctl("POST", "/api/store/refresh", {})
        self.assertEqual((st, ref["synced"]["grants"]), (200, 1), ref)             # unlocks already made stay


    @unittest.skipUnless(HAVE_CLAIM, "licd tree without the Ko-fi codes")
    def test_zkofi_code_from_the_email_claims_in_control(self):
        """PXA first (KOFI-PXA-FIRST Task 8): a payment with no wallet ID waits and mails its payer a private code; Control signs
        that code with the install key and the payment lands in this computer's wallet. Runs after the whole path (a wallet here)."""
        maildir = os.path.join(self.d, "mail")
        st, k = self.kofi("txn-joint-code", "5.00", "no wallet here", email="payer2@example.org")
        self.assertEqual((st, k.get("result")), (200, "unmatched"), k)
        for _ in range(100):                                          # the code mail goes out after the commit, on its own thread
            mails = sorted(os.listdir(maildir)) if os.path.isdir(maildir) else []
            if mails:
                break
            time.sleep(0.05)
        self.assertEqual(len(mails), 1)
        with open(os.path.join(maildir, mails[0])) as f:
            mail = json.load(f)
        self.assertEqual(mail["to"], "payer2@example.org")
        code = re.search(r"PXK-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}", mail["text"]).group(0)
        before = self.ctl("GET", "/api/store/wallet")[1]["wallet"]
        st, r = self.ctl("POST", "/api/store/wallet/claim", {"code": " " + code.lower() + " "})   # as people type it
        self.assertEqual(st, 200, r)
        self.assertEqual((r["claimed"], r["payments"], r["review"], r["created"], r["wallet"]["wallet_id"]),
                         (True, 1, 0, False, before["wallet_id"]))
        after = self.ctl("GET", "/api/store/wallet")[1]["wallet"]
        self.assertEqual((after["coins"], after["from_donations"]), (before["coins"] + 50, before["from_donations"] + 50))
        st, again = self.ctl("POST", "/api/store/wallet/claim", {"code": code})                      # one use
        self.assertEqual((st, again.get("code")), (404, "bad_code"), again)


if __name__ == "__main__":
    unittest.main()
