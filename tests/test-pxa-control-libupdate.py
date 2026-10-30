#!/usr/bin/env python3
"""PXA Control's licensed-library panel, ROUTE level (todo libupdate-ui-test; impl pxqn-lib-autoupdate's named next lever).

tests/test-pxa-lib-update.py covers the updater MODULE (tools/pxa_lib_update.py). This file covers what the page talks to:
a real C.App(L, ...) served by a real Control on loopback, with the module's LU swapped for a fake (C.LU = fake: class
LibUpdate reads the module global at call time), through the four routes

    GET  /api/lib/update            status payload shape; ?check=1 forces a check; the cached answer otherwise
    POST /api/lib/update/apply      200 + payload; up to date; the updater's refusal as a 4xx sentence
    POST /api/lib/update/rollback   200 + payload
    POST /api/lib/update/settings   channel / auto validation (a bad channel, a non-bool auto: 400), saved + echoed

plus a busy updater answering the 'already running' sentence, an unreachable server as a sentence (never a 500), the
auto-apply rule, the token guard, and a Control without the updater. No network, no GPU.

    python3 tests/test-pxa-control-libupdate.py          (wired into CTest as test-pxa-control-libupdate)
"""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-libupdate-ui-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN",
           "PXA_MODELS_DIR", "PXA_LIB_URL", "PXA_LIB_KEY"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402
import pxa_encode_pkg as PK  # noqa: E402

ENGINE_DIR = os.path.join(TMP, "engine")
os.makedirs(ENGINE_DIR, exist_ok=True)
STATUS_KEYS = {"available", "installed", "channel", "auto", "beta", "running", "busy", "error", "latest", "lib_id", "update",
               "notice", "min_engine"}
INSTALLED_KEYS = {"version", "lib_id", "prev", "prev_version", "applied", "checked", "links"}


class FakeLU(object):
    """the four calls LibUpdate makes, scripted; every call is recorded with the install dir and channel it got."""

    def __init__(self):
        self.calls = []
        self.state = {"version": "1.2.0", "lib_id": "pxqn-lib-1.2.0", "prev": None, "prev_version": "", "applied": 1760000000,
                      "checked": None, "channel": ""}
        self.latest = {"version": "1.3.0", "lib_id": "pxqn-lib-1.3.0", "min_engine": "4.0"}
        self.raw = {"notice": "faster MoE routing"}
        self.running = False
        self.check_exc = None
        self.apply_exc = None
        self.apply_done = True
        self.gate = None                 # a threading.Event: apply waits on it (the busy test)

    def load_state(self, d=None):
        self.calls.append(("load_state", d))
        return dict(self.state)

    def installed_links(self, d=None):
        return [{"name": "libpxqn.so", "target": "libpxqn.so.1.2.0"}]

    def server_running(self, d=None):
        self.calls.append(("server_running", d))
        return self.running

    def check(self, d=None, channel=None, **kw):
        self.calls.append(("check", d, channel))
        if self.check_exc:
            raise self.check_exc
        return dict(self.latest), dict(self.raw), self.state["version"]

    def apply(self, d=None, channel=None, **kw):
        self.calls.append(("apply", d, channel))
        if self.gate is not None:
            self.gate.wait(10)
        if self.apply_exc:
            raise self.apply_exc
        if not self.apply_done:
            return False, dict(self.latest), self.state["version"]
        self.state.update(prev_version=self.state["version"], version=self.latest["version"], lib_id=self.latest["lib_id"])
        return True, dict(self.latest), self.state["prev_version"]

    def rollback(self, d=None):
        self.calls.append(("rollback", d))
        ver, was = self.state["prev_version"] or "", self.state["version"]
        self.state.update(version=ver, prev_version="")
        return ver, was

    def n(self, what):
        return sum(1 for c in self.calls if c[0] == what)


TOKENS = {}


def serve_app(app):
    srv = C.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    TOKENS[srv.server_address[1]] = app.token
    return srv, srv.server_address[1]


def req(port, path, method="GET", body=None, auth=True):
    h = {}
    if auth and TOKENS.get(port):
        h["X-PXA-Token"] = TOKENS[port]
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


class LibUpdateRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_lu = C.core.LU
        cls.fake = FakeLU()
        C.LU = cls.fake
        cls.app = C.App(L, port=7777)
        cls.app.rig_static = lambda *a, **k: {"engine_dir": ENGINE_DIR}
        cls.app.lib.stop_ev.set()             # no background poller: every check in this file is one the test asked for
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        C.LU = cls.real_lu

    def setUp(self):
        C.LU = self.fake
        f = self.fake
        f.__init__()
        lib = self.app.lib
        lib._cache, lib._ask, lib._busy, lib._beta = None, 0.0, "", None
        C.save_config({})

    # ---- wiring --------------------------------------------------------------------------------
    def test_all_four_routes_are_wired(self):
        for k, fn in ((("GET", "/api/lib/update"), C.routes.r_lib_update),
                      (("POST", "/api/lib/update/apply"), C.routes.r_lib_update_apply),
                      (("POST", "/api/lib/update/rollback"), C.routes.r_lib_update_rollback),
                      (("POST", "/api/lib/update/settings"), C.routes.r_lib_update_settings)):
            self.assertIs(C.ROUTES.get(k), fn, k)

    def test_the_fake_reaches_the_class(self):
        self.assertIs(C.core.LU, self.fake)
        self.assertIs(C.routes.LU, self.fake)

    def test_wrong_methods_are_405(self):
        self.assertEqual(req(self.port, "/api/lib/update", "POST", {})[0], 405)
        self.assertEqual(req(self.port, "/api/lib/update/apply")[0], 405)
        self.assertEqual(req(self.port, "/api/lib/update/settings")[0], 405)

    def test_every_route_needs_the_token(self):
        for path, method in (("/api/lib/update", "GET"), ("/api/lib/update/apply", "POST"),
                             ("/api/lib/update/rollback", "POST"), ("/api/lib/update/settings", "POST")):
            code, _b = req(self.port, path, method, {} if method == "POST" else None, auth=False)
            self.assertEqual(code, 401, path)
        self.assertEqual(self.fake.n("apply") + self.fake.n("rollback"), 0)

    # ---- GET /api/lib/update -------------------------------------------------------------------
    def test_status_payload_shape_before_any_check(self):
        code, j = req(self.port, "/api/lib/update")
        self.assertEqual(code, 200)
        self.assertEqual(set(j), STATUS_KEYS)
        self.assertEqual(set(j["installed"]), INSTALLED_KEYS)
        self.assertIs(j["available"], True)
        self.assertEqual(j["installed"]["version"], "1.2.0")
        self.assertEqual(j["installed"]["links"][0]["name"], "libpxqn.so")
        self.assertEqual((j["channel"], j["auto"], j["running"], j["busy"], j["update"], j["latest"]),
                         ("stable", False, False, "", False, ""))
        self.assertEqual(self.fake.n("check"), 0)                       # status never asks the server
        self.assertIn(("load_state", ENGINE_DIR), self.fake.calls)      # the install Control runs from

    def test_check_1_forces_a_check_and_the_payload_carries_it(self):
        code, j = req(self.port, "/api/lib/update?check=1")
        self.assertEqual(code, 200)
        self.assertEqual(self.fake.n("check"), 1)
        self.assertEqual(self.fake.calls[[c[0] for c in self.fake.calls].index("check")], ("check", ENGINE_DIR, "stable"))
        self.assertEqual((j["latest"], j["lib_id"], j["update"], j["notice"], j["min_engine"]),
                         ("1.3.0", "pxqn-lib-1.3.0", True, "faster MoE routing", "4.0"))
        self.assertIsInstance(j["installed"]["checked"], int)
        for v in ("true", "yes", "on", "1"):
            req(self.port, "/api/lib/update?check=" + v)
        self.assertEqual(self.fake.n("check"), 5)                       # a forced check is never served from cache

    def test_without_check_the_cached_answer_is_returned(self):
        req(self.port, "/api/lib/update?check=1")
        code, j = req(self.port, "/api/lib/update")
        self.assertEqual(code, 200)
        self.assertEqual(self.fake.n("check"), 1)
        self.assertEqual((j["latest"], j["update"]), ("1.3.0", True))
        req(self.port, "/api/lib/update?check=0")
        self.assertEqual(self.fake.n("check"), 1)

    def test_same_version_is_no_update(self):
        self.fake.latest["version"] = "1.2.0"
        _c, j = req(self.port, "/api/lib/update?check=1")
        self.assertIs(j["update"], False)

    def test_unreachable_server_is_a_sentence_not_a_500(self):
        self.fake.check_exc = PK.PackageError("the licence server did not answer", code="offline")
        code, j = req(self.port, "/api/lib/update?check=1")
        self.assertEqual(code, 200)
        self.assertEqual(j["error"], {"code": "offline", "message": "the licence server did not answer"})
        self.assertIs(j["update"], False)
        self.fake.check_exc = OSError("boom")
        code, j = req(self.port, "/api/lib/update?check=1")
        self.assertEqual(code, 200)
        self.assertEqual(j["error"]["code"], "failed")
        self.assertIn("OSError: boom", j["error"]["message"])

    def test_beta_not_valued_marks_beta_refused(self):
        C.save_config({"lib": {"channel": "beta"}})
        self.fake.check_exc = PK.PackageError("beta needs the valued role", code="not_valued")
        _c, j = req(self.port, "/api/lib/update?check=1")
        self.assertIs(j["beta"], False)
        self.fake.check_exc = None
        _c, j = req(self.port, "/api/lib/update?check=1")
        self.assertIs(j["beta"], True)
        self.assertIn(("check", ENGINE_DIR, "beta"), self.fake.calls)

    def test_running_server_is_shown(self):
        self.fake.running = True
        _c, j = req(self.port, "/api/lib/update")
        self.assertIs(j["running"], True)

    # ---- POST /api/lib/update/apply ------------------------------------------------------------
    def test_apply_200_and_payload(self):
        code, j = req(self.port, "/api/lib/update/apply", "POST", {})
        self.assertEqual(code, 200)
        self.assertEqual(j["applied"], True)
        self.assertEqual((j["version"], j["lib_id"]), ("1.3.0", "pxqn-lib-1.3.0"))
        self.assertIn("Library 1.3.0 installed", j["message"])
        self.assertIn(("apply", ENGINE_DIR, "stable"), self.fake.calls)
        self.assertEqual(self.fake.n("check"), 1)                       # the page's answer is refreshed after
        self.assertEqual(self.app.lib._busy, "")
        _c, s = req(self.port, "/api/lib/update")
        self.assertEqual(s["installed"]["version"], "1.3.0")

    def test_apply_on_the_channel_the_body_names(self):
        req(self.port, "/api/lib/update/apply", "POST", {"channel": "beta"})
        self.assertIn(("apply", ENGINE_DIR, "beta"), self.fake.calls)

    def test_apply_up_to_date(self):
        self.fake.apply_done = False
        code, j = req(self.port, "/api/lib/update/apply", "POST", {})
        self.assertEqual(code, 200)
        self.assertEqual((j["applied"], j["version"]), (False, "1.2.0"))
        self.assertEqual(j["message"], "The library is already up to date.")

    def test_apply_refusal_is_a_4xx_sentence_with_the_code(self):
        self.fake.apply_exc = PK.PackageError("Stop the server on port 8091 first: it has the library open.", code="server_running")
        code, j = req(self.port, "/api/lib/update/apply", "POST", {})
        self.assertEqual(code, 400)
        self.assertEqual(j["code"], "server_running")
        self.assertIn("Stop the server", j["error"])
        self.assertEqual(self.app.lib._busy, "")                       # a refusal never leaves the panel stuck busy

    def test_busy_updater_says_already_running(self):
        self.fake.gate = threading.Event()
        out = {}
        t = threading.Thread(target=lambda: out.update(r=req(self.port, "/api/lib/update/apply", "POST", {})))
        t.start()
        for _ in range(200):
            if self.app.lib._busy:
                break
            threading.Event().wait(0.02)
        self.assertEqual(self.app.lib._busy, "apply")
        _c, s = req(self.port, "/api/lib/update")
        self.assertEqual(s["busy"], "apply")
        for path in ("/api/lib/update/apply", "/api/lib/update/rollback"):
            code, j = req(self.port, path, "POST", {})
            self.assertEqual(code, 400, path)
            self.assertEqual(j["error"], "A library apply is already running; wait for it to finish.")
        self.fake.gate.set()
        t.join(15)
        self.assertEqual(out["r"][0], 200)
        self.assertEqual(self.fake.n("apply"), 1)
        self.assertEqual(self.fake.n("rollback"), 0)

    def test_busy_check_is_named_too(self):
        # since 10-19 a click waits up to LIB_CHECK_WAIT_S for an in-flight check (it only reads the server);
        # a check that never ends is still refused by name once the wait runs out
        saved = C.core.LIB_CHECK_WAIT_S
        C.core.LIB_CHECK_WAIT_S = 0.3
        self.addCleanup(setattr, C.core, "LIB_CHECK_WAIT_S", saved)
        self.app.lib._busy = "check"
        code, j = req(self.port, "/api/lib/update/rollback", "POST", {})
        self.assertEqual(code, 400)
        self.assertIn("already running", j["error"])
        self.assertIn("check", j["error"])

    # ---- POST /api/lib/update/rollback ---------------------------------------------------------
    def test_rollback_200_and_payload(self):
        req(self.port, "/api/lib/update/apply", "POST", {})
        code, j = req(self.port, "/api/lib/update/rollback", "POST", {})
        self.assertEqual(code, 200)
        self.assertEqual(j, {"applied": True, "version": "1.2.0", "was": "1.3.0", "message": "Rolled back to 1.2.0."})
        self.assertIn(("rollback", ENGINE_DIR), self.fake.calls)

    def test_rollback_to_the_shipped_library(self):
        code, j = req(self.port, "/api/lib/update/rollback", "POST", {})
        self.assertEqual(code, 200)
        self.assertEqual(j["version"], "")
        self.assertEqual(j["message"], "Rolled back to the library the engine shipped.")

    # ---- POST /api/lib/update/settings ---------------------------------------------------------
    def test_settings_saved_and_echoed(self):
        code, j = req(self.port, "/api/lib/update/settings", "POST", {"channel": "beta", "auto": True})
        self.assertEqual(code, 200)
        self.assertEqual(set(j), STATUS_KEYS)
        self.assertEqual((j["channel"], j["auto"]), ("beta", True))
        self.assertEqual(C.load_config()["lib"], {"channel": "beta", "auto": True})
        code, j = req(self.port, "/api/lib/update/settings", "POST", {"auto": False})
        self.assertEqual((code, j["channel"], j["auto"]), (200, "beta", False))       # an absent field is left alone
        code, j = req(self.port, "/api/lib/update/settings", "POST", {"auto": True})
        code, j = req(self.port, "/api/lib/update/settings", "POST", {"channel": "stable"})
        self.assertEqual((code, j["channel"], j["auto"]), (200, "stable", True))      # the page sends one field at a time
        self.assertEqual(C.load_config()["lib"], {"channel": "stable", "auto": True})

    def test_setting_beats_the_state_files_channel(self):
        self.fake.state["channel"] = "beta"
        _c, j = req(self.port, "/api/lib/update")
        self.assertEqual(j["channel"], "beta")
        _c, j = req(self.port, "/api/lib/update/settings", "POST", {"channel": "stable"})
        self.assertEqual(j["channel"], "stable")

    def test_bad_channel_400(self):
        for ch in ("nightly", "", "STABLE", 1, True, ["stable"]):
            code, j = req(self.port, "/api/lib/update/settings", "POST", {"channel": ch})
            self.assertEqual(code, 400, ch)
            self.assertEqual(j["error"], "channel must be one of: stable, beta")
        self.assertNotIn("lib", C.load_config())

    def test_non_bool_auto_400(self):
        for a in ("true", 1, 0, "yes", [], {}):
            code, j = req(self.port, "/api/lib/update/settings", "POST", {"auto": a})
            self.assertEqual(code, 400, a)
            self.assertEqual(j["error"], "auto must be true or false")
        self.assertNotIn("lib", C.load_config())

    def test_settings_body_must_be_an_object(self):
        h = {"X-PXA-Token": TOKENS[self.port], "Content-Type": "application/json"}
        r = urllib.request.Request("http://127.0.0.1:%d/api/lib/update/settings" % self.port, data=b"[1]", headers=h,
                                   method="POST")
        try:
            urllib.request.urlopen(r, timeout=10)
            self.fail("a list body was accepted")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    # ---- the auto rule -------------------------------------------------------------------------
    def test_auto_off_never_applies(self):
        req(self.port, "/api/lib/update?check=1")
        self.assertEqual(self.fake.n("apply"), 0)

    def test_auto_on_applies_a_newer_release_after_a_check(self):
        req(self.port, "/api/lib/update/settings", "POST", {"auto": True})
        _c, j = req(self.port, "/api/lib/update?check=1")
        self.assertEqual(self.fake.n("apply"), 1)
        self.assertEqual(j["installed"]["version"], "1.3.0")

    def test_auto_on_with_a_refusing_updater_stays_a_200(self):
        req(self.port, "/api/lib/update/settings", "POST", {"auto": True})
        self.fake.apply_exc = PK.PackageError("a server is running", code="server_running")
        code, _j = req(self.port, "/api/lib/update?check=1")
        self.assertEqual(code, 200)


class NoUpdater(unittest.TestCase):
    """a Control built without tools/pxa_lib_update.py: the panel says so, nothing 500s."""

    def setUp(self):
        self.saved = C.core.LU
        C.LU = None
        self.app = C.App(L, port=7777)
        self.app.lib.stop_ev.set()
        self.srv, self.port = serve_app(self.app)

    def tearDown(self):
        self.srv.shutdown()
        C.LU = self.saved

    def test_status_and_actions(self):
        code, j = req(self.port, "/api/lib/update")
        self.assertEqual(code, 200)
        self.assertIs(j["available"], False)
        self.assertEqual(j["error"]["code"], "unavailable")
        code, j = req(self.port, "/api/lib/update?check=1")
        self.assertEqual((code, j["available"]), (200, False))
        for path in ("/api/lib/update/apply", "/api/lib/update/rollback"):
            code, j = req(self.port, path, "POST", {})
            self.assertEqual(code, 400, path)
            self.assertIn("no library updater", j["error"])


class Poller(unittest.TestCase):
    def test_a_page_view_starts_one_poller_that_checks_once(self):
        saved = C.core.LU
        fake = FakeLU()
        C.LU = fake
        try:
            app = C.App(L, port=7777)
            app.rig_static = lambda *a, **k: {"engine_dir": ENGINE_DIR}
            srv, port = serve_app(app)
            try:
                req(port, "/api/lib/update")
                req(port, "/api/lib/update")
                for _ in range(250):
                    if fake.n("check"):
                        break
                    threading.Event().wait(0.02)
                self.assertEqual(fake.n("check"), 1)
                self.assertEqual(sum(1 for t in threading.enumerate() if t.name == "pxa-lib" and t.is_alive()), 1)
            finally:
                app.lib.close()
                srv.shutdown()
        finally:
            C.LU = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
