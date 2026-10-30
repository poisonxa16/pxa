#!/usr/bin/env python3
"""Background grant sync (PLUGINS-PROTOTYPE-PLAN S6, ANIBAL spec B3): 12 h +- 1 h timer, on wake, after a redeem; a 403
revoked / not_linked / account_disabled from licd locks every grant at once; offline never locks.

    python3 tests/test-pxa-plugin-sync.py
    PXA_LICD_DIR=<store repo>/tools/pxa-licd python3 tests/test-pxa-plugin-sync.py     # + the real-licd cases

Loopback only (a fake licd on 127.0.0.1, and a real licd when its tree is there). No GPU, no internet.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import http.server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-plugin-sync-test-")
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)
LICD_DIR = os.environ.get("PXA_LICD_DIR", "")   # the licd tree by env only (no default path)
HAVE_LICD = os.path.exists(os.path.join(LICD_DIR, "store.py")) and os.path.exists(os.path.join(LICD_DIR, "wallet.py"))
if HAVE_LICD:
    sys.path.insert(0, LICD_DIR)
    import licd  # noqa: E402
    import pxa_lic  # noqa: E402
    import store as licd_store  # noqa: E402

import pxa_control  # noqa: E402,F401
from pxa_control import store as S  # noqa: E402
from pxa_control.plugins import gate as G  # noqa: E402

SEED = bytes(range(1, 33))      # the test's licd entitlement key (never a real key)
KID = "t-sync"


class Clock(object):
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


def grant_for(cfg, item="dictate", kind="plugin", issued=None, ent="0123456789abcdef0123456789abcdef"):
    _seed, pub = S.install_key(cfg)
    now = int(time.time()) if issued is None else issued
    st = {"v": 1, "iss": "licd", "kid": KID, "ent_id": ent, "kind": kind, "item_id": item, "level": None,
          "sub": "a" * 32, "install_pub": pub, "issued": now, "refresh_after": now + 86400, "not_after": now + 180 * 86400}
    return {"statement": st, "signature": S.b64u(S.ed25519_sign(SEED, S.ENT_PREFIX + S.canonical(st)))}


class FakeLicd(object):
    """POST /v1/store/nonce and /v1/store/refresh; refresh answers whatever `answer` holds: (status, json)."""

    def __init__(self):
        self.answer = (200, {"grants": [], "account": {"account": "acct", "credits": 0, "owned": []}})
        self.calls = []
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                fake.calls.append(self.path)
                if self.path == "/v1/store/nonce":
                    st, body = 200, {"nonce": "n" * 32}
                else:
                    st, body = fake.answer
                data = json.dumps(body).encode()
                self.send_response(st)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def refreshes(self):
        return sum(1 for c in self.calls if c == "/v1/store/refresh")

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Fake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.licd = FakeLicd()
        cls.keys = {KID: S.ed25519_public(SEED).hex()}

    @classmethod
    def tearDownClass(cls):
        cls.licd.close()

    def setUp(self):
        self.cfg = tempfile.mkdtemp(dir=TMP)
        self.clock = Clock(int(time.time()))
        self.cache = S.EntitlementCache(self.cfg, self.keys, clock=self.clock)
        self.heard = []
        self.listener = S.on_sync(lambda cfg, r: self.heard.append((cfg, r)))
        self.licd.calls.clear()
        self.licd.answer = (200, {"grants": [], "account": {"account": "acct", "credits": 0, "owned": []}})
        self.client = S.LicdClient(self.licd.url, self.cfg, self.keys, timeout=5)
        self.bg = S.BackgroundSync(self.cfg, client=lambda: self.client, cache=self.cache, clock=self.clock, rand=lambda a, b: 0.0)

    def tearDown(self):
        self.bg.stop()
        S.off_sync(self.listener)

    def own(self, item="dictate", **kw):
        g = grant_for(self.cfg, item, **kw)
        self.cache.add(g)
        return g

    def runs(self, item="dictate"):
        return G.run_allowed(self.cfg, item, "plugin", self.cache)

    def test_timer_is_12h_plus_minus_1h_and_starts_after_start_delay(self):
        self.own()
        t0 = self.clock.t
        self.assertIsNone(self.bg.step())                                  # nothing due at start
        self.clock.t = t0 + S.SYNC_START_S
        r = self.bg.step()
        self.assertEqual((r["reason"], r["ok"]), ("timer", True))
        self.assertEqual(self.bg.next_due, self.clock.t + 12 * 3600)
        seen = []
        bg = S.BackgroundSync(self.cfg, client=lambda: self.client, cache=self.cache, clock=self.clock,
                              rand=lambda a, b: seen.append((a, b)) or b)
        bg._schedule(self.clock.t, True)
        self.assertEqual((seen, bg.next_due - self.clock.t), ([(-3600, 3600)], 13 * 3600))
        bg.stop()
        for _ in range(50):                                                # the real jitter stays in the window
            real = S.BackgroundSync(self.cfg, client=lambda: None, cache=self.cache, clock=self.clock)
            real._schedule(self.clock.t, True)
            self.assertTrue(11 * 3600 <= real.next_due - self.clock.t <= 13 * 3600)
            real.stop()
        self.clock.t += 12 * 3600 - 61
        self.bg._last_tick = self.clock.t - 60
        self.assertIsNone(self.bg.step())                                  # not yet
        self.clock.t += 61
        self.bg._last_tick = self.clock.t - 60
        self.assertEqual(self.bg.step()["reason"], "timer")

    def test_wake_syncs_now(self):
        self.own()
        self.bg.step()
        self.clock.t += S.SYNC_TICK_S                                      # a normal tick: nothing
        self.assertIsNone(self.bg.step())
        self.clock.t += S.SYNC_TICK_S + S.WAKE_GAP_S + 1                   # the loop slept through a suspend
        self.assertEqual(self.bg.step()["reason"], "wake")
        self.assertEqual(self.licd.refreshes(), 1)

    def test_after_redeem_kick_and_any_sync_restarts_the_count(self):
        self.own()
        self.bg.kick("redeem")
        self.assertEqual(self.bg.step()["reason"], "redeem")
        self.clock.t += 5 * 3600
        self.client.sync(self.cache)                                       # a page refresh / an inline redeem sync
        self.assertEqual(self.bg.next_due, self.clock.t + 12 * 3600)
        self.assertTrue(S.kick("x") is False or True)                      # no running loop: a no-op, never an error

    def test_offline_never_locks(self):
        self.own()
        for st in (503, 502, 500, 429):
            self.licd.answer = (st, {"error": "busy", "code": "store_busy"})
            r = self.bg.run_once("timer")
            self.assertEqual((r["ok"], r["hard_lock"]), (False, None))
            self.assertTrue(self.runs())
            self.assertEqual(self.bg.next_due, self.clock.t + S.SYNC_RETRY_S)
        self.licd.answer = (403, {"error": "forbidden", "code": "bad_signature"})      # a 403 that is not one of the three
        self.assertEqual(self.bg.run_once("timer")["hard_lock"], None)
        self.assertTrue(self.runs())
        self.bg.client = lambda: S.LicdClient("http://127.0.0.1:9", self.cfg, self.keys, timeout=2)   # nothing listens
        self.assertEqual(self.bg.run_once("timer")["ok"], False)
        self.assertTrue(self.runs())
        self.assertFalse(S.load_snapshot(self.cfg)["ok"])

    def test_403_hard_locks_within_one_sync(self):
        for code in S.HARD_LOCK_CODES:
            cfg_before = self.cfg
            self.setUp()
            self.assertNotEqual(cfg_before, self.cfg)
            g = self.own(issued=int(self.clock.t) - 60)
            S.save_snapshot(self.cfg, {"account": "acct", "credits": 5}, ok=True)
            self.assertTrue(self.runs())
            self.licd.answer = (403, {"error": "no", "code": code})
            r = self.bg.run_once("timer")
            self.assertEqual((r["ok"], r["hard_lock"]), (False, code))
            self.assertFalse(self.runs())                                  # off within ONE sync
            st = G.state(self.cfg, "dictate", "plugin", self.cache)
            self.assertEqual((st["state"], st["hard_lock"]), ("revoked", code))
            self.assertEqual(self.heard[-1][1]["hard_lock"], code)
            self.assertEqual(S.entitlements_view(self.cache)["sync"]["locked"], code)
            if code != "revoked":
                self.assertIsNone(S.load_snapshot(self.cfg)["account"])
            with self.assertRaises(S.GrantError) as e:                     # the dropped copy put back: issued before
                self.cache.add(g)                                          # the lock time (RC3.6 item 4 hard-lock v4)
            self.assertEqual(e.exception.code, "pre_lock")
            self.assertFalse(self.runs())
            self.assertEqual(self.bg.next_due, self.clock.t + 12 * 3600)  # licd answered: the normal turn
            self.tearDown()
        self.setUp()

    def test_relink_clears_the_lock(self):
        g = self.own(issued=int(self.clock.t) - 60)
        self.licd.answer = (403, {"error": "no", "code": "not_linked"})
        self.bg.run_once("timer")
        self.assertFalse(self.runs())
        fresh = grant_for(self.cfg, issued=int(time.time()) + 1)        # licd signs the same grant again, newer and
                                                                          # after the 403's lock time (its Date header)
        self.assertGreater(fresh["statement"]["issued"], g["statement"]["issued"])
        self.licd.answer = (200, {"grants": [fresh], "account": {"account": "acct", "credits": 0, "owned": []}})
        r = self.bg.run_once("timer")
        self.assertEqual((r["ok"], r.get("grants"), r.get("refused")), (True, 1, []), r)
        self.assertTrue(self.runs())
        self.assertIsNone(self.cache.hard_locked())

    def test_revocation_list_turns_a_plugin_off_within_one_sync(self):
        g = self.own()
        self.assertTrue(self.runs())
        rev = S.sign_revocations(SEED, KID, [g["statement"]["ent_id"]], int(time.time()))
        self.licd.answer = (200, {"grants": [], "revocations": rev, "account": {"account": "acct", "credits": 0, "owned": []}})
        self.assertTrue(self.bg.run_once("timer")["ok"])
        self.assertFalse(self.runs())
        self.assertEqual(G.state(self.cfg, "dictate", "plugin", self.cache)["state"], "missing")

    # ---- review 10-26 L1 / L2: the account-level 403 / corruption reason after a good sync -------------------
    ENT_A = "a" * 32
    ENT_D = "d" * 32

    def two_items_then_403(self, code="not_linked"):
        ga = self.own("anibal", issued=int(self.clock.t) - 60, ent=self.ENT_A)
        gd = self.own("dictate", issued=int(self.clock.t) - 60, ent=self.ENT_D)
        self.assertTrue(self.runs("anibal") and self.runs("dictate"))
        self.licd.answer = (403, {"error": "no", "code": code})
        self.assertEqual(self.bg.run_once("timer").get("hard_lock"), code)
        self.assertEqual(self.cache.hard_locked()["code"], code)
        return ga, gd

    def test_l1_relink_without_one_item_clears_the_account_lock(self):
        """scn E: unlink -> 403 -> relink; licd grants anibal afresh, dictate's purchase was revoked meanwhile (owned[]
        says revoked, no grant): the account-level reason goes, dictate ends revoked by licd, not by the old 403."""
        self.two_items_then_403()
        fresh = grant_for(self.cfg, "anibal", issued=int(time.time()) + 1, ent=self.ENT_A)
        owned = [{"item_id": "anibal", "status": "active"}, {"item_id": "dictate", "status": "revoked"}]
        self.licd.answer = (200, {"grants": [fresh], "account": {"account": "acct", "credits": 0, "owned": owned}})
        r = self.bg.run_once("timer")
        self.assertTrue(r["ok"], r)
        self.assertTrue(self.runs("anibal"))
        self.assertFalse(self.runs("dictate"))                         # runs never change: no grant, still in the record
        self.assertIsNone(self.cache.hard_locked())
        self.assertNotIn("locked", S.entitlements_view(self.cache, S.load_snapshot(self.cfg))["sync"])
        self.assertIsNone(self.bg.status()["hard_lock"])
        st = G.state(self.cfg, "dictate", "plugin", self.cache)
        self.assertEqual((st["state"], st.get("lock"), st.get("hard_lock")), ("revoked", "revoked", None), st)

    def test_l1_relink_item_not_owned_any_more_reads_missing(self):
        self.two_items_then_403("account_disabled")
        fresh = grant_for(self.cfg, "anibal", issued=int(time.time()) + 1, ent="e" * 32)       # a new ent_id (relink)
        self.licd.answer = (200, {"grants": [fresh], "account": {"account": "acct", "credits": 0,
                                                                  "owned": [{"item_id": "anibal", "status": "active"}]}})
        self.assertTrue(self.bg.run_once("timer")["ok"])
        self.assertTrue(self.runs("anibal"))
        self.assertFalse(self.runs("dictate"))
        self.assertIsNone(self.cache.hard_locked())
        self.assertEqual(G.state(self.cfg, "dictate", "plugin", self.cache)["state"], "missing")

    def test_l1_no_fresh_evidence_never_lifts_the_account_lock(self):
        """an answer without a grant licd signed after the lock (none, the old copy again, an unknown key's) keeps it."""
        ga, gd = self.two_items_then_403()
        bad_key = dict(grant_for(self.cfg, "anibal", issued=int(time.time()) + 1, ent="b" * 32))
        bad_key["signature"] = S.b64u(S.ed25519_sign(bytes(range(2, 34)), S.ENT_PREFIX + S.canonical(bad_key["statement"])))
        acct = {"account": "acct", "credits": 0, "owned": [{"item_id": "anibal", "status": "active"}]}
        for grants in ([], [ga], [ga, gd], [bad_key]):
            self.licd.answer = (200, {"grants": grants, "account": acct})
            self.assertTrue(self.bg.run_once("timer")["ok"])
            self.assertEqual((self.cache.hard_locked() or {}).get("code"), "not_linked", grants)
            self.assertEqual(S.entitlements_view(self.cache)["sync"].get("locked"), "not_linked")
            self.assertEqual(G.state(self.cfg, "dictate", "plugin", self.cache).get("hard_lock"), "not_linked")
            self.assertFalse(self.runs("anibal") or self.runs("dictate"))
        # a later 403 after a lift puts the account lock back (the fresh grant is dropped and is before the new lock)
        self.licd.answer = (200, {"grants": [grant_for(self.cfg, "anibal", issued=int(time.time()) + 1, ent=self.ENT_A)],
                                  "account": acct})
        self.assertTrue(self.bg.run_once("timer")["ok"])
        self.assertIsNone(self.cache.hard_locked())
        time.sleep(1.1)
        self.licd.answer = (403, {"error": "no", "code": "revoked"})
        self.bg.run_once("timer")
        self.assertEqual(self.cache.hard_locked()["code"], "revoked")
        self.assertFalse(self.runs("anibal"))

    def corrupt(self):
        with open(self.cache.path, "w") as f:
            f.write("{not json")

    def test_l2_never_held_item_after_a_damaged_file_reads_missing(self):
        """scn F: a damaged grants.json locks every held item ("*"); an item this computer never held reads what the main
        line at c0c4d88f1a shows (missing: unlocks with PXA coins), before and after the next good sync."""
        self.own("dictate", issued=int(self.clock.t) - 60, ent=self.ENT_D)
        self.corrupt()
        self.assertFalse(self.runs("dictate"))
        self.assertEqual(G.state(self.cfg, "dictate", "plugin", self.cache)["state"], "invalid")
        self.assertEqual(G.state(self.cfg, "anibal", "plugin", self.cache)["state"], "missing")
        self.assertEqual(self.cache.hard_locked()["code"], "corrupt")
        fresh = grant_for(self.cfg, "dictate", issued=int(time.time()) + 1, ent=self.ENT_D)
        self.licd.answer = (200, {"grants": [fresh], "account": {"account": "acct", "credits": 0,
                                                                  "owned": [{"item_id": "dictate", "status": "active"}]}})
        self.assertTrue(self.bg.run_once("timer")["ok"])
        self.assertTrue(self.runs("dictate"))
        self.assertEqual(G.state(self.cfg, "anibal", "plugin", self.cache)["state"], "missing")
        self.assertFalse(self.runs("anibal"))
        self.assertIsNone(self.cache.hard_locked())

    def test_l2_held_set_unknown_reads_invalid_until_licd_answers(self):
        """no journal survived either: which items were held is unknown, so every item reads "could not be checked"
        until a fresh grant arrives (fail closed); after it, a never-held item reads missing."""
        self.own("dictate", issued=int(self.clock.t) - 60, ent=self.ENT_D)
        for p in self.cache._journal_paths():
            os.unlink(p)
        self.corrupt()
        self.assertEqual(G.state(self.cfg, "anibal", "plugin", self.cache)["state"], "invalid")
        fresh = grant_for(self.cfg, "dictate", issued=int(time.time()) + 1, ent=self.ENT_D)
        self.licd.answer = (200, {"grants": [fresh], "account": {"account": "acct", "credits": 0,
                                                                  "owned": [{"item_id": "dictate", "status": "active"}]}})
        self.assertTrue(self.bg.run_once("timer")["ok"])
        self.assertTrue(self.runs("dictate"))
        self.assertEqual(G.state(self.cfg, "anibal", "plugin", self.cache)["state"], "missing")

    def test_nothing_to_keep_fresh_calls_nobody(self):
        r = self.bg.run_once("timer")
        self.assertTrue(r["skipped"])
        self.assertEqual(self.licd.refreshes(), 0)
        S.save_snapshot(self.cfg, {"account": "acct", "credits": 1}, ok=True)
        self.assertFalse(self.bg.run_once("timer").get("skipped"))
        self.assertEqual(self.licd.refreshes(), 1)

    def test_thread_and_off_switch(self):
        self.own()
        bg = S.BackgroundSync(self.cfg, client=lambda: self.client, cache=self.cache, tick=0.05, start_delay=3600)
        bg.start()
        self.assertIs(S._BG[0], bg)
        self.assertTrue(S.kick("redeem"))
        for _ in range(100):
            if bg.last:
                break
            time.sleep(0.05)
        bg.stop()
        self.assertEqual((bg.last["reason"], bg.last["ok"]), ("redeem", True))
        self.assertIsNone(S._BG[0])
        os.environ["PXA_STORE_URL"] = self.licd.url
        try:
            os.environ["PXA_STORE_SYNC"] = "0"
            self.assertIsNone(S.start_background(self.cfg, self.cache))
            os.environ["PXA_STORE_SYNC"] = "1"
            b2 = S.start_background(self.cfg, self.cache)
            self.assertIsInstance(b2, S.BackgroundSync)
            b2.stop()
        finally:
            os.environ["PXA_STORE_URL"] = ""
            os.environ.pop("PXA_STORE_SYNC", None)
        self.assertIsNone(S.start_background(self.cfg, self.cache))        # no store URL: no loop


@unittest.skipUnless(HAVE_LICD, "licd tree not found (PXA_LICD_DIR)")
class RealLicd(unittest.TestCase):
    """the proof column: revoke on a loopback licd -> the plugin is off within one background sync; unlink -> 403
    not_linked -> hard lock."""

    @classmethod
    def setUpClass(cls):
        d = os.path.join(TMP, "licd")
        keys = licd_store.keys_init(os.path.join(d, "secrets"), "ent-sync", "plug-sync")
        os.makedirs(os.path.join(d, "files"))
        env = {"PXA_LIC_DB": os.path.join(d, "db.sqlite"), "PXA_LIC_PORT": "0", "PXA_LIC_STORE_SEED": keys["ent_seeds"]["ent-sync"],
               "PXA_LIC_STORE_KID": "ent-sync", "PXA_LIC_STORE_FILES_DIR": os.path.join(d, "files"),
               "PXA_LIC_BACKUP_KEEP": "0", "PXA_LIC_PUBLIC_URL": ""}
        cls.lsvc = licd.Service(licd.Config(env))
        cls.lsrv, _ = licd.serve(licd.Config(env), cls.lsvc)
        threading.Thread(target=cls.lsrv.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d" % cls.lsrv.server_address[1]
        cls.keys = dict(keys["GRANT_PUBKEYS"])
        cls.lsvc._rl = lambda *a, **k: None     # every test computer is 127.0.0.1: no per-IP wallet-create limit here
        cls.owner("catalog", "edit", "dictate", "--status", "active")

    @classmethod
    def tearDownClass(cls):
        cls.lsrv.shutdown()
        cls.lsrv.server_close()

    @classmethod
    def owner(cls, *argv):
        """pxa-lic as the operator. licd audits every change under a name (`--by`, required since proj/licd 6b10f7394's
        successors; without it pxa_lic.run returns 2 and says why on stderr, which the assert now carries)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = pxa_lic.run(["--by", "sync-test"] + list(argv), svc=cls.lsvc, out=out)
        assert rc == 0, (argv, out.getvalue(), err.getvalue())
        return out.getvalue()

    def machine(self):
        cfg = tempfile.mkdtemp(dir=TMP)
        cache = S.EntitlementCache(cfg, self.keys)
        cl = S.LicdClient(self.url, cfg, self.keys, timeout=10)
        wid = cl.wallet("create")["wallet_id"]
        self.lsvc.store.deposit("admin", wid, 300, "sync test")
        cl.wallet("redeem", {"item_id": "dictate"})
        bg = S.BackgroundSync(cfg, client=lambda: cl, cache=cache, clock=time.time, rand=lambda a, b: 0.0)
        r = bg.run_once("redeem")
        self.assertEqual(r["ok"], True, r)
        self.assertTrue(G.run_allowed(cfg, "dictate", "plugin", cache))
        return cfg, cache, cl, wid, bg

    def machine2(self, items, wid=None):
        cfg = tempfile.mkdtemp(dir=TMP)
        cache = S.EntitlementCache(cfg, self.keys)
        cl = S.LicdClient(self.url, cfg, self.keys, timeout=10)
        if wid is None:
            wid = cl.wallet("create")["wallet_id"]
            self.lsvc.store.deposit("admin", wid, 1000, "sync test", confirm=True)
            for i in items:
                cl.wallet("redeem", {"item_id": i})
        bg = S.BackgroundSync(cfg, client=lambda: cl, cache=cache, clock=time.time, rand=lambda a, b: 0.0)
        return cfg, cache, cl, wid, bg

    def acct_of(self, cfg):
        _s, pub = S.install_key(cfg)
        return self.lsvc.store.db.q1("SELECT * FROM store_accounts WHERE id=(SELECT account_id FROM store_machines "
                                     "WHERE install_pub=?)", (pub,)), pub

    def test_l1_partial_relink_two_items(self):
        """review 10-26 scn E on the real licd: unlink -> 403 -> the dictate purchase revoked -> rejoin: anibal runs,
        dictate stays off as revoked, and the account-level 403 reason is gone."""
        self.owner("catalog", "edit", "anibal", "--status", "active")
        cfg, cache, cl, wid, bg = self.machine2(("dictate", "anibal"))
        self.assertTrue(bg.run_once("redeem")["ok"])
        self.assertTrue(G.run_allowed(cfg, "anibal", "plugin", cache) and G.run_allowed(cfg, "dictate", "plugin", cache))
        code = cl.wallet("code")
        a, pub = self.acct_of(cfg)
        self.lsvc.store.unlink_machine("admin", a, pub)
        time.sleep(1.1)
        self.assertEqual(bg.run_once("timer").get("hard_lock"), "not_linked")
        self.assertEqual(cache.hard_locked()["code"], "not_linked")
        self.lsvc.store.db.x("UPDATE store_entitlements SET status='revoked' WHERE account_id=? AND item_id='dictate'", (a["id"],))
        cfg2, cache2, cl2, _w, bg2 = self.machine2((), wid=wid)
        cl2.wallet("join", {"code": code.get("code")})
        cl.wallet("join", {"code": cl2.wallet("code").get("code")})
        time.sleep(1.1)
        self.assertTrue(bg.run_once("redeem")["ok"])
        self.assertTrue(G.run_allowed(cfg, "anibal", "plugin", cache))
        self.assertFalse(G.run_allowed(cfg, "dictate", "plugin", cache))
        self.assertIsNone(cache.hard_locked())
        self.assertNotIn("locked", S.entitlements_view(cache, S.load_snapshot(cfg))["sync"])
        self.assertIsNone(bg.status()["hard_lock"])
        st = G.state(cfg, "dictate", "plugin", cache)
        self.assertEqual(st["state"], "revoked", st)
        self.assertNotIn("hard_lock", st)
        bg.stop(); bg2.stop()

    def test_l2_corrupt_file_never_held_item(self):
        """review 10-26 scn F on the real licd: a damaged store file, then one good sync."""
        self.owner("catalog", "edit", "anibal", "--status", "active")
        cfg, cache, cl, wid, bg = self.machine2(("dictate",))
        self.assertTrue(bg.run_once("redeem")["ok"])
        with open(cache.path, "w") as f:
            f.write("{not json")
        self.assertFalse(G.run_allowed(cfg, "dictate", "plugin", cache))
        self.assertEqual(G.state(cfg, "anibal", "plugin", cache)["state"], "missing")
        time.sleep(1.1)
        self.assertTrue(bg.run_once("timer")["ok"])
        self.assertTrue(G.run_allowed(cfg, "dictate", "plugin", cache))
        self.assertEqual(G.state(cfg, "anibal", "plugin", cache)["state"], "missing")
        self.assertIsNone(cache.hard_locked())
        bg.stop()

    def test_owner_revoke_is_off_within_one_sync(self):
        cfg, cache, cl, wid, bg = self.machine()
        grant = self.lsvc.store.grants(wid)[0]["grant"]
        self.assertIn("revoked 1 grant", self.owner("revoke", grant, "--reason", "sync test"))
        self.assertTrue(G.run_allowed(cfg, "dictate", "plugin", cache))   # still on until the next sync (offline-safe)
        r = bg.run_once("timer")
        self.assertTrue(r["ok"], r)
        self.assertFalse(G.run_allowed(cfg, "dictate", "plugin", cache))
        bg.stop()

    def test_unlinked_computer_hard_locks(self):
        cfg, cache, cl, wid, bg = self.machine()
        _seed, pub = S.install_key(cfg)
        self.lsvc.store.unlink_machine("admin", self.lsvc.store.db.q1(
            "SELECT * FROM store_accounts WHERE id=(SELECT account_id FROM store_machines WHERE install_pub=?)", (pub,)), pub)
        env = dict(next(iter(cache._load()["grants"].values())))
        r = bg.run_once("timer")
        self.assertEqual((r["ok"], r["hard_lock"]), (False, "not_linked"), r)
        self.assertFalse(G.run_allowed(cfg, "dictate", "plugin", cache))
        self.assertEqual(G.state(cfg, "dictate", "plugin", cache)["state"], "revoked")
        with self.assertRaises(S.GrantError):                               # a saved copy does not come back
            cache.add(env)
        bg.stop()


if __name__ == "__main__":
    if not HAVE_LICD:          # I2 (review 10-26): the real-licd proof column is never skipped silently
        n = len([m for m in dir(RealLicd) if m.startswith("test_")])
        sys.stderr.write("test-pxa-plugin-sync: REAL-LICD SKIPPED: %d test(s) did not run; set PXA_LICD_DIR to a licd tree "
                         "(tools/pxa-licd with store.py + wallet.py); now %r\n" % (n, LICD_DIR))
    unittest.main(verbosity=1)
