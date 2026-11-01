#!/usr/bin/env python3
"""The hook door (tools/pxa_control/plugins/hookdoor.py): POST /plugins/<id>/hook/<trigger> from a program on this computer
reaches a running plugin as the host op "hook", only when its signed manifest says hook_door and the user turned the door on
in Control. One or more tests per door rule (the rule's number is in the test's name):

    python3 tests/test-pxa-plugin-hookdoor.py

Loopback only: stub plugins built and signed here with a throwaway key (build-pxp.py). Every trigger key is a throwaway made
here and handed to the stub through its own data folder (Control never holds one), and the Control token is this test's own.
Two Control servers on 127.0.0.1: one with this computer's rules, one with the --lan rules. It starts and stops plugin
processes. No GPU, no internet.
"""
import hashlib
import hmac
import http.client
import importlib.util
import io
import json
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-hookdoor-test-")
CFG = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_CONFIG_DIR"] = CFG
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _k in ("PXA_CONTROL_TOKEN", "PXA_CONTROL_ACCESS_LOG", "PXA_CONTROL_HF_TOKEN", "HF_TOKEN"):
    os.environ.pop(_k, None)
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)

import pxa_control as C  # noqa: E402
from pxa_control import plugins as P, server as SV, store as S  # noqa: E402
from pxa_control.plugins import child as CH, hookdoor as HD, host as H, overlay as OVL, pxp, ui as PUI  # noqa: E402

PLUG_SEED = bytes(range(41, 73))        # throwaway plugin publisher key (never a real key)
PLUG_KID = "plug-test"
PLUG_KEYS = {PLUG_KID: S.ed25519_public(PLUG_SEED).hex()}
TOKEN = "this-test-control-token-" + secrets.token_hex(12)      # the test Controls' own token (never a real one)
STUBS = ("stubhook", "stubhook2", "stubnohook", "stubgone")
DATA_FILES = ("ops.jsonl", "http.jsonl", "entered.jsonl", "release", "plan.json")
# every header the server may read for a door request: the Host check, then the door's own
DOOR_READS = {"host", "origin", "sec-fetch-site", "transfer-encoding", "content-length", "content-type", "x-hive-time",
              "x-hive-sign"}

STUB = r'''#!/usr/bin/env python3
import hashlib, hmac, json, os, sys, threading, time
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))
from pxa_plugin_child import Plugin
DATA = os.environ.get("PXA_PLUGIN_DATA", ".")
LOCK = threading.Lock()
SEEN = set()


def note(name, row):
    with LOCK:
        with open(os.path.join(DATA, name), "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def plan():
    try:
        with open(os.path.join(DATA, "plan.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def handle(method, path, query, ctype, body):
    note("http.jsonl", {"method": method, "path": path})
    if path.endswith("/hello"):
        return 200, "application/json", PL.hello
    if path.endswith("/call"):
        j = json.loads(body or b"{}")
        return 200, "application/json", PL.call(j.get("name"), j.get("args") or {})
    return 404, "application/json", {"error": "no such stub route"}


def hook(trigger, ts, sig, ctype, body):
    p = plan()
    key = (p.get("keys") or {}).get(trigger)
    if key is None:
        return 404, "application/json", {"ok": False, "code": "no_trigger"}
    want = hmac.new(key.encode("ascii"), ts.encode("ascii") + b"." + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want.encode("ascii"), sig.lower().encode("ascii")):
        return 401, "application/json", {"ok": False, "code": "bad_signature"}
    with LOCK:
        if (trigger, sig.lower()) in SEEN:
            return 409, "application/json", {"ok": False, "code": "replayed"}
        SEEN.add((trigger, sig.lower()))
    if trigger in (p.get("paused") or []):
        return 409, "application/json", {"ok": False, "code": "trigger_paused"}
    if trigger in (p.get("html") or []):
        return 200, "text/html", "<script>alert(1)</script>"
    if trigger in (p.get("slow") or []):
        note("entered.jsonl", {"t": time.time()})
        end = time.time() + 20
        while not os.path.exists(os.path.join(DATA, "release")) and time.time() < end:
            time.sleep(0.02)
    return 202, "application/json", {"ok": True, "bytes": len(body), "ctype": ctype}


class Stub(Plugin):
    def _on_request(self, msg):
        if msg.get("op") == "hook":
            note("ops.jsonl", msg)                  # the hook op exactly as it crossed (the test reads what reached the plugin)
        return Plugin._on_request(self, msg)


PL = Stub(handle, on_hook=hook)
sys.exit(PL.run())
'''


def _build_mod():
    spec = importlib.util.spec_from_file_location("build_pxp", os.path.join(TOOLS, "pxa-private", "plugins", "build-pxp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_stub(pid, hook_door=True):
    src = os.path.join(TMP, "src", pid)
    for d in ("bin", "ui", "lib"):
        os.makedirs(os.path.join(src, d))
    with open(os.path.join(src, "bin", "stub"), "w") as f:
        f.write(STUB)
    with open(os.path.join(src, "ui", "index.html"), "w") as f:
        f.write("<p>stub</p>\n")
    shutil.copy(os.path.join(TOOLS, "pxa_control", "plugins", "child.py"), os.path.join(src, "lib", "pxa_plugin_child.py"))
    spec = {"schema": 1, "plugin_id": pid, "name": pid.title(), "version": "0.1.0", "publisher": "pxa", "edition": "free",
            "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/stub", "serve"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/%s/" % pid], "ui": {"tab": {"title": pid.title(), "page": "ui/index.html"}},
            "permissions": [], "entitlement": None, "pack": ["bin", "ui", "lib"]}
    if hook_door is not None:
        spec["hook_door"] = hook_door
    with open(os.path.join(src, "plugin.src.json"), "w") as f:
        json.dump(spec, f)
    path, _body = _build_mod().build(src, PLUG_SEED, PLUG_KID, os.path.join(TMP, "out"))
    return pxp.install(path, CFG, PLUG_KEYS)


def sign(key, ts, body):
    return hmac.new(key.encode("ascii"), ts.encode("ascii") + b"." + body, hashlib.sha256).hexdigest()


def until(fn, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    return fn()


def jload(data):
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def data_path(pid, name):
    return os.path.join(CFG, "plugins", "data", pid, name)


def rows(pid, name):
    try:
        with open(data_path(pid, name)) as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


def audit_rows():
    try:
        with open(os.path.join(CFG, "plugins", "audit.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


def read_text(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ---- the test servers: a stand-in peer address, and a record of every header name the server reads ----------------------------
FAKE = {"peer": None}
READS = []          # (request path, set of lower-case header names read after the request was parsed)


class Recorder(object):
    """a request's headers that note every name asked for; any other use of them (items, keys, iteration) counts as all."""

    def __init__(self, msg, sink):
        self._m, self._sink = msg, sink

    def get(self, name, default=None):
        self._sink.add(name.lower())
        return self._m.get(name, default)

    def get_all(self, name, failobj=None):
        self._sink.add(name.lower())
        return self._m.get_all(name, failobj)

    def __getitem__(self, name):
        self._sink.add(name.lower())
        return self._m[name]

    def __contains__(self, name):
        self._sink.add(name.lower())
        return name in self._m

    def __iter__(self):
        self._sink.add("*all")
        return iter(self._m)

    def __len__(self):
        self._sink.add("*all")
        return len(self._m)

    def __getattr__(self, attr):
        self._sink.add("*" + attr)
        return getattr(self._m, attr)


def serve_app(app):
    srv = C.make_server(app, "127.0.0.1", 0)
    base = srv.RequestHandlerClass

    class Probe(base):
        def setup(self):
            base.setup(self)
            if FAKE["peer"]:
                self.client_address = (FAKE["peer"], 40000)

        def parse_request(self):
            ok = base.parse_request(self)
            if ok:
                sink = set()
                READS.append((self.path, sink))
                self.headers = Recorder(self.headers, sink)
            return ok

    srv.RequestHandlerClass = Probe
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def req(srv, path, body=b"", headers=(), method="POST", host=None, length=True, send_body=True, timeout=10):
    """one raw request: the headers exactly as given (a list of pairs, so a header can repeat) -> (status, headers, bytes)."""
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    hs = [] if host is False else [("Host", host or "127.0.0.1:%d" % port)]
    if length and body is not None:
        hs.append(("Content-Length", str(len(body))))
    for k, v in list(hs) + list(headers):
        c.putheader(k, v)
    c.endheaders(body if (send_body and body) else None)     # one send: a refusal never leaves the body unread mid-way
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, r.headers, data


def listening():
    """the sockets this process listens on (TCP and unix), from /proc."""
    inodes = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink("/proc/self/fd/" + fd)
        except OSError:
            continue
        if t.startswith("socket:["):
            inodes.add(t[8:-1])
    out = set()
    for proto in ("tcp", "tcp6"):
        for row in read_text("/proc/self/net/" + proto).splitlines()[1:]:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A" and cols[9] in inodes:
                out.add((proto, cols[1]))
    for row in read_text("/proc/self/net/unix").splitlines()[1:]:
        cols = row.split()
        if len(cols) > 6 and cols[5] == "01" and cols[6] in inodes:
            out.add(("unix", cols[6]))
    return out


class CountingReader(io.BytesIO):
    pass


class FakeHandler(object):
    """enough of the request handler for hookdoor.serve, with any peer and raw header lines."""

    def __init__(self, peer, raw, body=b"", command="POST"):
        self.client_address = (peer, 5555)
        self.headers = http.client.parse_headers(io.BytesIO(raw + b"\r\n"))
        self.rfile = CountingReader(body)
        self.wfile = io.BytesIO()
        self.command = command
        self.code = None
        self.sent = []
        self.close_connection = False

    def send_response(self, code):
        self.code = code

    def send_header(self, k, v):
        self.sent.append((k, v))

    def end_headers(self):
        pass


class DoorBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_stub("stubhook")
        make_stub("stubhook2")
        make_stub("stubnohook", hook_door=None)
        make_stub("stubgone")
        cls.now = [1000.0]
        cls.host = H.PluginHost(CFG, cache=S.EntitlementCache(CFG), pubkeys=PLUG_KEYS, health_s=3600, tick_s=3600,
                                backoff=(0,) * 5, hello_timeout_s=10)
        cls.host.door.clock = lambda: cls.now[0]
        models = os.path.join(TMP, "models")
        cls.app = C.App(C.desktop.load_launcher(), port=7785, token=TOKEN, models_dirs=[models])
        cls.app.plugin_host = cls.host
        cls.srv = serve_app(cls.app)
        cls.lan_app = C.App(C.desktop.load_launcher(), port=7786, lan=True, token=TOKEN, models_dirs=[models])
        cls.lan_app.plugin_host = cls.host
        cls.lan_srv = serve_app(cls.lan_app)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        for s in (cls.srv, cls.lan_srv):
            s.shutdown()
            s.server_close()

    def setUp(self):
        FAKE["peer"] = None
        self.now[0] = 1000.0
        for pid in STUBS:
            self.host.stop(pid, "test")
            self.host.door.set_on(pid, False)
            for name in DATA_FILES:
                try:
                    os.unlink(data_path(pid, name))
                except OSError:
                    pass
        with self.host.door.lock:
            self.host.door.fails.clear()
            self.host.door.closed.clear()
            self.host.door.busy.clear()
        with self.host.consent.lock:
            self.host.consent._save({})

    # ---- helpers ---------------------------------------------------------------------------------------------------------
    def consent(self, pid):
        m, _d = self.host.manifest(pid)
        self.host.consent.grant(m, m["permissions"])
        return m

    def plan(self, pid, **kw):
        os.makedirs(os.path.dirname(data_path(pid, "plan.json")), exist_ok=True)
        with open(data_path(pid, "plan.json"), "w") as f:
            json.dump(kw, f)

    def open_door(self, pid="stubhook", **plan):
        """consent, the user's switch on, the plugin running, one trigger: -> (trigger id, its key)."""
        self.consent(pid)
        self.host.door.set_on(pid, True)
        self.host.start(pid)
        tid, key = secrets.token_hex(16), "hk_" + secrets.token_hex(24)
        keys = dict(plan.pop("keys", {}), **{tid: key})
        self.plan(pid, keys=keys, **plan)
        return tid, key

    def signed(self, key, body, ts=None, ctype="text/plain"):
        ts = str(int(time.time())) if ts is None else ts
        return [("X-Hive-Time", ts), ("X-Hive-Sign", sign(key, ts, body)), ("Content-Type", ctype)]

    def call(self, tid, key, body=b"build 41 passed", pid="stubhook", srv=None, headers=(), path=None, **kw):
        return req(srv or self.srv, path or "/plugins/%s/hook/%s" % (pid, tid), body, self.signed(key, body) + list(headers), **kw)

    def ops(self, pid="stubhook"):
        return rows(pid, "ops.jsonl")

    def hello(self, pid):
        st, _ct, data = self.host.forward(pid, "GET", "/api/plugins/%s/hello" % pid, "", "", b"")
        self.assertEqual(st, 200)
        return json.loads(data)

    def ctl(self, path, body, srv=None, token=True, headers=()):
        h = [("Content-Type", "application/json")] + ([("X-PXA-Token", TOKEN)] if token else []) + list(headers)
        st, hd, data = req(srv or self.srv, path, json.dumps(body).encode(), h)
        return st, jload(data)


class DoorTest(DoorBase):
    # ---- the happy path ----------------------------------------------------------------------------------------------------
    def test_a_signed_call_reaches_the_plugin_and_its_answer_comes_back_as_json(self):
        tid, key = self.open_door()
        st, hd, data = self.call(tid, key, b"build 41 passed")
        self.assertEqual((st, jload(data)), (202, {"ok": True, "bytes": 15, "ctype": "text/plain"}))
        self.assertEqual(hd["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(hd["Cache-Control"], "no-store")
        self.assertEqual(hd["Content-Security-Policy"], H.PLUGIN_CSP)
        self.assertEqual(hd["X-Content-Type-Options"], "nosniff")
        self.assertEqual(len(self.ops()), 1)
        # an answer the plugin labels HTML is never served as HTML on Control's origin
        tid2, key2 = secrets.token_hex(16), "hk_" + secrets.token_hex(24)
        self.plan("stubhook", keys={tid: key, tid2: key2}, html=[tid2])
        st, hd, data = self.call(tid2, key2, b"x")
        self.assertEqual(st, 200)
        self.assertEqual(hd["Content-Type"], "application/octet-stream")

    # ---- rule 1: the route ---------------------------------------------------------------------------------------------------
    def test_rule1_only_the_door_shape_goes_to_the_door_and_every_other_plugins_path_is_unchanged(self):
        tid, key = self.open_door()
        self.assertEqual(HD.match("/plugins/stubhook/hook/" + tid), ("stubhook", tid))
        others = ["/plugins/stubhook/hook/" + tid.upper(), "/plugins/stubhook/hook/" + tid[:-1], "/plugins/stubhook/hook/" + tid + "0",
                  "/plugins/stubhook/hook/" + tid + "/", "/plugins/stubhook/hook/" + tid + "/x", "/plugins/stubhook/hooks/" + tid,
                  "/plugins/stubhook/hook", "/plugins/Stubhook/hook/" + tid, "/plugins/stubhook/overlay",
                  "/plugins/stub_hook/hook/" + tid, "/plugins//hook/" + tid]
        for p in others:
            self.assertIsNone(HD.match(p), p)
        with mock.patch.object(HD, "serve", side_effect=AssertionError("the door took another path")), \
                mock.patch.object(OVL, "serve", wraps=OVL.serve) as ov:
            for p in others:
                body = b"x"
                st, _hd, data = req(self.srv, p, body, self.signed(key, body))
                self.assertIn(st, (403, 404, 405), p)
            self.assertEqual(ov.call_count, len(others))                 # overlay.serve answered each, as before the door
            st, _hd, _d = req(self.srv, "/plugins/stubhook/overlay?k=x", None, method="GET")
            self.assertEqual(st, 404)                                    # the overlay route itself: unchanged (no overlay here)
        self.assertEqual(self.ops(), [])

    def test_rule1_after_the_host_check_and_before_the_origin_and_token_checks(self):
        tid, key = self.open_door()
        st, _hd, data = self.call(tid, key, host="evil.example")      # the Host check runs first (this computer's bind)
        self.assertEqual((st, jload(data)["code"]), (421, "bad_host"))
        self.assertEqual(self.ops(), [])
        with mock.patch.object(SV.Handler, "_needs_token", side_effect=AssertionError("token check ran")), \
                mock.patch.object(SV.Handler, "_auth", side_effect=AssertionError("token read")), \
                mock.patch.object(SV, "fetch_site_ok", side_effect=AssertionError("fetch-metadata check ran")), \
                mock.patch.object(SV.Handler, "_origin_ok", side_effect=AssertionError("Origin check ran")):
            st, _hd, data = self.call(tid, key, b"no token at all")    # no token: the door answers, not the token check
            self.assertEqual(st, 202)
            st, _hd, data = self.call(tid, key, b"from a page", headers=[("Origin", "http://evil.example")])
            self.assertEqual((st, jload(data)["code"]), (403, "from_a_page"))     # the door's own refusal, not H2's
        self.assertEqual(len(self.ops()), 1)

    # ---- rule 2: loopback only -------------------------------------------------------------------------------------------
    def test_rule2_a_peer_that_is_not_this_computer_is_refused_on_every_bind(self):
        tid, key = self.open_door()
        port, lan_port = self.srv.server_address[1], self.lan_srv.server_address[1]
        for peer in ("192.168.1.50", "10.0.0.7", "::ffff:192.168.1.50", "127.0.0.2", "fe80::1"):
            FAKE["peer"] = peer
            for srv, host in ((self.srv, "127.0.0.1:%d" % port), (self.lan_srv, "127.0.0.1:%d" % lan_port),
                              (self.lan_srv, "192.168.1.10:%d" % lan_port)):
                st, _hd, data = self.call(tid, key, srv=srv, host=host)
                self.assertEqual((st, jload(data)["code"]), (403, "not_local"), (peer, host))
        FAKE["peer"] = None
        self.assertEqual(self.ops(), [])
        self.assertEqual(self.call(tid, key, b"from here", srv=self.lan_srv)[0], 202)
        for peer, host in (("::1", "[::1]:8080"), ("::ffff:127.0.0.1", "localhost:8080"), ("127.0.0.1", "LOCALHOST")):
            h = FakeHandler(peer, ("Host: %s\r\nContent-Length: 0\r\n" % host).encode())
            self.assertTrue(HD._local(h), (peer, host))

    def test_rule2_a_loopback_peer_with_a_host_that_is_not_this_computer_is_refused(self):
        tid, key = self.open_door()
        lan_port = self.lan_srv.server_address[1]
        for host in ("192.168.1.10:%d" % lan_port, "evil.example", "evil.example:%d" % lan_port, "localhost.evil.example",
                     "127.0.0.1.nip.io", "[::2]:%d" % lan_port):          # DNS rebinding: the --lan bind lets any Host through
            st, _hd, data = self.call(tid, key, srv=self.lan_srv, host=host)
            self.assertEqual((st, jload(data)["code"]), (403, "not_local"), host)
        st, _hd, data = self.call(tid, key, srv=self.lan_srv, headers=[("Host", "evil.example")])   # a second Host header
        self.assertEqual((st, jload(data)["code"]), (403, "not_local"))
        self.assertEqual(self.ops(), [])
        self.assertEqual(self.call(tid, key, b"named right", srv=self.lan_srv, host="localhost:%d" % lan_port)[0], 202)

    # ---- rule 3: POST only, no Origin, at most 16384 bytes ----------------------------------------------------------------
    def test_rule3_post_only(self):
        tid, key = self.open_door()
        path = "/plugins/stubhook/hook/" + tid
        for method in ("GET", "DELETE"):
            st, hd, data = req(self.srv, path, None, self.signed(key, b""), method=method)
            self.assertEqual((st, jload(data)["code"], hd["Allow"]), (405, "method_not_allowed", "POST"), method)
        st, _hd, _data = req(self.srv, path, None, method="HEAD")
        self.assertEqual(st, 405)
        for method in ("PUT", "PATCH", "OPTIONS"):                       # never reach Control's dispatch at all
            st, _hd, _data = req(self.srv, path, b"x", self.signed(key, b"x"), method=method)
            self.assertEqual(st, 501, method)
        self.assertEqual(self.ops(), [])

    def test_rule3_any_request_with_an_origin_header_is_refused(self):
        tid, key = self.open_door()
        port = self.srv.server_address[1]
        for origin in ("http://127.0.0.1:%d" % port, "http://localhost:%d" % port, "null", "http://evil.example", ""):
            st, _hd, data = self.call(tid, key, headers=[("Origin", origin)])
            self.assertEqual((st, jload(data)["code"]), (403, "from_a_page"), origin)
        for site in ("same-origin", "same-site", "cross-site"):
            st, _hd, data = self.call(tid, key, headers=[("Sec-Fetch-Site", site)])
            self.assertEqual((st, jload(data)["code"]), (403, "from_a_page"), site)
        self.assertEqual(self.ops(), [])
        self.assertEqual(self.call(tid, key, headers=[("Sec-Fetch-Site", "none")])[0], 202)

    def test_rule3_16384_bytes_pass_and_16385_are_refused_without_reading_past_the_cap(self):
        tid, key = self.open_door()
        body = b"a" * 16384
        st, _hd, data = self.call(tid, key, body)
        self.assertEqual((st, jload(data)["bytes"]), (202, 16384))
        self.assertEqual(len(CH.b64d(self.ops()[-1]["b64"])), 16384)
        big = b"b" * 16385
        t0 = time.monotonic()                     # the headers only: a door that waited for the body would hang here
        st, _hd, data = req(self.srv, "/plugins/stubhook/hook/" + tid, big, self.signed(key, big), send_body=False)
        self.assertEqual((st, jload(data)["code"]), (413, "too_large"))
        self.assertLess(time.monotonic() - t0, 5.0)
        h = FakeHandler("127.0.0.1", b"Host: 127.0.0.1:1\r\nContent-Length: 16385\r\n", big)
        HD.serve(h, self.host, "POST", "stubhook", tid)
        self.assertEqual((h.code, h.rfile.tell()), (413, 0))           # not one byte of the body was read
        self.assertEqual(len(self.ops()), 1)

    def test_rule3_the_length_must_be_given_once_as_digits_and_the_body_read_exactly(self):
        tid, key = self.open_door()
        path = "/plugins/stubhook/hook/" + tid
        st, _hd, data = req(self.srv, path, b"x", self.signed(key, b"x"), length=False, send_body=False)
        self.assertEqual((st, jload(data)["code"]), (411, "length_required"))
        st, _hd, data = req(self.srv, path, b"1\r\nx\r\n0\r\n\r\n", self.signed(key, b"x") + [("Transfer-Encoding", "chunked")],
                            length=False, send_body=False)
        self.assertEqual((st, jload(data)["code"]), (411, "length_required"))
        for bad in ("abc", "-1", "+5", "1e3", "5 5", "²", "0x10", "1234567890"):
            st, _hd, data = req(self.srv, path, b"", self.signed(key, b"") + [("Content-Length", bad)], length=False)
            self.assertEqual((st, jload(data)["code"]), (400, "bad_length"), bad)
        st, _hd, data = req(self.srv, path, b"x", self.signed(key, b"x") + [("Content-Length", "1")])     # two lengths
        self.assertEqual((st, jload(data)["code"]), (400, "bad_length"))
        # a body that ends before its length: refused, never forwarded short
        port = self.srv.server_address[1]
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        hs = "".join("%s: %s\r\n" % kv for kv in self.signed(key, b"y" * 100))
        s.sendall(("POST %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Length: 100\r\n%s\r\n" % (path, port, hs)).encode() + b"y" * 50)
        s.shutdown(socket.SHUT_WR)
        got = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            got += chunk
        s.close()
        self.assertTrue(got.startswith(b"HTTP/1.0 400"), got[:40])
        self.assertIn(b'"short_body"', got)
        self.assertEqual(self.ops(), [])

    # ---- rule 4: Cookie, X-PXA-Token and Authorization are never read or forwarded; the token never opens the door ---------
    def test_rule4_cookie_token_and_authorization_are_never_read_or_forwarded(self):
        tid, key = self.open_door()
        cookie = "%s=%s; pxa_control_token=%s" % (self.app.cookie_name, TOKEN, TOKEN)
        extra = [("Cookie", cookie), ("Authorization", "Bearer " + TOKEN), ("X-PXA-Token", TOKEN)]
        path = "/plugins/stubhook/hook/%s?token=%s" % (tid, TOKEN)
        del READS[:]
        st, _hd, data = self.call(tid, key, b"with every token", headers=extra, path=path)
        self.assertEqual(st, 202)
        door_reads = [sink for (p, sink) in READS if p == path]
        self.assertEqual(len(door_reads), 1)
        self.assertLessEqual(door_reads[0], DOOR_READS)                  # never Cookie, X-PXA-Token, Authorization, nor all
        op = self.ops()[-1]
        self.assertEqual(sorted(op), ["b64", "ctype", "id", "op", "sig", "t", "trigger", "ts"])
        self.assertEqual(op["op"], "hook")
        for word in (TOKEN, "Bearer", "pxa_control_token", "Cookie", "Authorization"):
            self.assertNotIn(word, json.dumps(op))
        # the recorder sees what the server reads: an /api call reads the token header
        del READS[:]
        self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": True})
        self.assertIn("x-pxa-token", READS[-1][1])

    def test_rule4_the_control_token_in_place_of_a_signature_is_refused_like_any_bad_signature(self):
        tid, key = self.open_door()
        path = "/plugins/stubhook/hook/" + tid
        body, ts = b"let me in", str(int(time.time()))
        tokens = [("Cookie", "%s=%s" % (self.app.cookie_name, TOKEN)), ("Authorization", "Bearer " + TOKEN),
                  ("X-PXA-Token", TOKEN)]
        # the token as the signature (the wrong shape): the same answer as any malformed signature
        st, _hd, as_sig = req(self.srv, path, body, [("X-Hive-Time", ts), ("X-Hive-Sign", TOKEN)] + tokens)
        st2, _hd, junk = req(self.srv, path, body, [("X-Hive-Time", ts), ("X-Hive-Sign", "zz")])
        self.assertEqual((st, as_sig), (st2, junk))
        self.assertEqual((st, jload(as_sig)["code"]), (401, "bad_signature"))
        # the token used as an HMAC key, and a well-formed wrong signature with every token beside it: the plugin's 401
        st, _hd, with_tok = req(self.srv, path, body, [("X-Hive-Time", ts), ("X-Hive-Sign", sign(TOKEN, ts, body))] + tokens)
        st2, _hd, without = req(self.srv, path, body, [("X-Hive-Time", ts), ("X-Hive-Sign", "0" * 64)])
        self.assertEqual((st, with_tok), (st2, without))
        self.assertEqual((st, jload(with_tok)["code"]), (401, "bad_signature"))
        # no signature headers at all, only the token
        st, _hd, data = req(self.srv, path, body, tokens)
        self.assertEqual((st, jload(data)["code"]), (401, "bad_signature"))
        self.assertEqual(len(self.host.door.fails["stubhook"]), 5)         # every one counted toward the failure limit
        self.assertEqual(len(self.ops()), 2)                               # only the two well-formed ones reached the plugin

    # ---- rule 5: the "hook" op, never "http", never an /api/plugins route; the door never sees a key ---------------------
    def test_rule5_the_call_reaches_the_plugin_as_the_hook_op_and_never_as_http(self):
        tid, key = self.open_door()
        body = b'{"build": 41}'
        with mock.patch.object(self.host, "forward", side_effect=AssertionError("the http op")):
            st, _hd, _data = self.call(tid, key, body, headers=[])
        self.assertEqual(st, 202)
        op = self.ops()[-1]
        self.assertEqual((op["op"], op["trigger"], CH.b64d(op["b64"]), op["ctype"]), ("hook", tid, body, "text/plain"))
        self.assertTrue(re.match(r"^[0-9]{1,12}$", op["ts"]) and re.match(r"^[0-9a-f]{64}$", op["sig"]))
        self.assertEqual(rows("stubhook", "http.jsonl"), [])                # the plugin's http handler never ran
        # the same call on an /api/plugins path is an ordinary plugin route: the token is asked for, the hook handler never runs
        api = "/api/plugins/stubhook/hook/" + tid
        st, _hd, _data = req(self.srv, api, body, self.signed(key, body))
        self.assertEqual(st, 401)
        st, _hd, _data = req(self.srv, api, body, self.signed(key, body) + [("X-PXA-Token", TOKEN)])
        self.assertEqual(st, 404)                                           # the stub's http handler: no such route
        self.assertEqual(len(self.ops()), 1)
        self.assertEqual([r["path"] for r in rows("stubhook", "http.jsonl")], [api])

    def test_rule5_the_plugin_side_hook_branch(self):
        got = []

        def hook(*a):
            got.append(a)
            return 202, "application/json", {"ok": True}

        def http(*a):
            raise AssertionError("the http handler ran for a hook op")

        pl = CH.Plugin(http, on_hook=hook)
        out = pl._on_request({"op": "hook", "trigger": "a" * 32, "ts": "1700000000", "sig": "b" * 64, "ctype": "text/plain",
                              "b64": CH.b64e(b"hello")})
        self.assertEqual(got, [("a" * 32, "1700000000", "b" * 64, "text/plain", b"hello")])
        self.assertEqual((out["status"], out["ctype"], json.loads(CH.b64d(out["b64"]))), (202, "application/json", {"ok": True}))
        self.assertEqual(CH.Plugin(http)._on_request({"op": "hook", "trigger": "a" * 32}), {"status": 404, "code": "no_hook"})

    def test_rule5_the_door_never_sees_a_key(self):
        tid, key = self.open_door()
        for body in (b"one", b"two"):
            self.assertEqual(self.call(tid, key, body)[0], 202)
        self.assertEqual(self.call(tid, "hk_wrong", b"three")[0], 401)
        src = read_text(HD.__file__)
        for word in ("hmac", "compare_digest", "hashlib", "secrets", "keystore"):
            self.assertNotIn(word, src)                                  # the door has nothing to check a signature with
        self.assertNotIn(key, repr(vars(self.host.door)))
        for path in (os.path.join(CFG, "plugins", "audit.jsonl"), os.path.join(CFG, "plugins", HD.SETTING)):
            self.assertNotIn(key, read_text(path))

    # ---- rule 6: no key, signature or body in a log line, an error text or an answer ---------------------------------------
    def test_rule6_no_key_signature_or_body_in_control_or_plugin_logs_errors_or_answers(self):
        tid, key = self.open_door()
        marks = ["BODYMARK%d%s" % (i, secrets.token_hex(4)) for i in range(7)]
        secrets_seen, answers = [key], []
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"PXA_CONTROL_ACCESS_LOG": "1"}), mock.patch.object(sys, "stderr", err):
            for i, extra in enumerate(([], [("Origin", "null")])):
                body = marks[i].encode()
                hs = self.signed(key, body)
                secrets_seen.append(hs[1][1])
                answers.append(req(self.srv, "/plugins/stubhook/hook/" + tid, body, hs + extra))
            body = marks[2].encode()
            hs = self.signed(key, body)
            secrets_seen.append(hs[1][1])
            answers.append(req(self.srv, "/plugins/stubhook/hook/" + tid, body, hs))           # the replay of a fresh one
            answers.append(req(self.srv, "/plugins/stubhook/hook/" + tid, body, hs))
            body = marks[3].encode()
            bad = [("X-Hive-Time", "1700000000"), ("X-Hive-Sign", "c" * 64)]
            secrets_seen.append("c" * 64)
            answers.append(req(self.srv, "/plugins/stubhook/hook/" + tid, body, bad))           # a wrong signature
            answers.append(req(self.srv, "/plugins/stubhook/hook/" + "d" * 32, body, bad))      # an unknown id
            big = marks[4].encode() + b"z" * 16400
            answers.append(req(self.srv, "/plugins/stubhook/hook/" + tid, big, self.signed(key, big), send_body=False))
            body = marks[5].encode()
            answers.append(req(self.srv, "/plugins/stubnohook/hook/" + tid, body, self.signed(key, body)))
            body = marks[6].encode()
            with mock.patch.object(self.host, "hook", side_effect=ValueError(marks[6] + " " + key)):
                answers.append(req(self.srv, "/plugins/stubhook/hook/" + tid, body, self.signed(key, body)))
        self.assertEqual([a[0] for a in answers], [202, 403, 202, 409, 401, 404, 413, 404, 500])
        self.assertIn("the hook door failed (ValueError)", err.getvalue())
        self.assertIn("/plugins/stubhook/hook/", err.getvalue())              # the access log did run
        logs = {"control stderr": err.getvalue(), "audit": read_text(os.path.join(CFG, "plugins", "audit.jsonl")),
                "plugin log": read_text(os.path.join(CFG, "plugins", "logs", "stubhook.log")),
                "door setting": read_text(os.path.join(CFG, "plugins", HD.SETTING)),
                "answers": "".join(a[2].decode("utf-8", "replace") + str(dict(a[1])) for a in answers)}
        for label, text in logs.items():
            for s in secrets_seen + marks:
                self.assertNotIn(s, text, label)
        # the hook op itself carried them: that is the one place they go
        self.assertIn(marks[0], "".join(CH.b64d(o["b64"]).decode() for o in self.ops()))

    # ---- rule 7: the failure limit -------------------------------------------------------------------------------------------
    def test_rule7_ten_refusals_in_a_minute_close_the_door_for_five_minutes(self):
        tid, key = self.open_door()
        for i in range(10):
            st, _hd, data = self.call(tid, "hk_guess%d" % i, b"guess %d" % i)
            self.assertEqual((st, jload(data)["code"]), (401, "bad_signature"), i)
            self.now[0] += 1
        self.assertEqual(len(self.ops()), 10)
        st, hd, data = self.call(tid, key, b"the 11th, signed right")       # the 11th call in the minute: closed
        self.assertEqual((st, jload(data)["code"], hd["Retry-After"]), (429, "door_closed", "299"))   # closed at 1009 + 300
        self.assertEqual(len(self.ops()), 10)                              # nothing forwarded while it is closed
        closed = [r for r in audit_rows() if r["plugin"] == "stubhook" and r["event"] == "hook_door_closed"]
        self.assertEqual(len(closed), 1)
        for at in (1011.0, 1150.0, 1308.9):                                # still closed inside the 5 minutes
            self.now[0] = at
            st, hd, data = self.call(tid, key, b"still closed %r" % at)
            self.assertEqual(st, 429, at)
            self.assertEqual(int(hd["Retry-After"]), max(1, int(1309 - at + 0.999)), at)
        self.assertEqual(len(self.ops()), 10)
        self.now[0] = 1309.0                                               # 300 s after the 10th refusal: open again
        self.assertEqual(self.call(tid, key, b"open again")[0], 202)
        self.assertEqual(len(self.ops()), 11)

    def test_rule7_what_counts_and_for_whom(self):
        tid, key = self.open_door()
        paused = secrets.token_hex(16)
        self.plan("stubhook", keys={tid: key, paused: key}, paused=[paused])
        for i in range(12):                                                # a paused trigger is no guess: never counted
            self.assertEqual(self.call(paused, key, b"paused %d" % i)[0], 409)
        for i in range(12):                                                # nor a right call
            self.assertEqual(self.call(tid, key, b"right %d" % i)[0], 202)
        self.assertNotIn("stubhook", self.host.door.fails)
        body = b"replay me"
        hs = self.signed(key, body)
        self.assertEqual(req(self.srv, "/plugins/stubhook/hook/" + tid, body, hs)[0], 202)
        self.assertEqual(req(self.srv, "/plugins/stubhook/hook/" + tid, body, hs)[0], 409)     # replay: counted
        self.assertEqual(self.call("e" * 32, key)[0], 404)                                     # unknown id: counted
        self.assertEqual(req(self.srv, "/plugins/stubhook/hook/" + tid, b"x", [("X-Hive-Time", "soon")])[0], 401)   # malformed
        self.assertEqual(len(self.host.door.fails["stubhook"]), 3)
        self.now[0] += 60                                                  # refusals older than a minute drop out
        for i in range(9):
            self.assertEqual(self.call(tid, "hk_x", b"late %d" % i)[0], 401)
        self.assertEqual(self.call(tid, key, b"9 inside the minute")[0], 202)
        self.assertEqual(self.call(tid, "hk_x", b"the 10th")[0], 401)
        self.assertEqual(self.call(tid, key, b"closed now")[0], 429)
        # the limit is per plugin: another plugin's door stays open
        tid2, key2 = self.open_door("stubhook2")
        self.assertEqual(self.call(tid2, key2, pid="stubhook2")[0], 202)

    def test_rule7_at_most_four_calls_in_flight_per_plugin(self):
        tid, key = self.open_door()
        slow = secrets.token_hex(16)
        self.plan("stubhook", keys={tid: key, slow: key}, slow=[slow])
        results = []
        threads = [threading.Thread(target=lambda i=i: results.append(self.call(slow, key, b"slow %d" % i)[0])) for i in range(4)]
        for t in threads:
            t.start()
        self.assertTrue(until(lambda: len(rows("stubhook", "entered.jsonl")) == 4))
        st, hd, data = self.call(tid, key, b"the fifth")
        self.assertEqual((st, jload(data)["code"], hd["Retry-After"]), (503, "busy", "1"))
        with open(data_path("stubhook", "release"), "w"):
            pass
        for t in threads:
            t.join(30)
        self.assertEqual(sorted(results), [202] * 4)
        self.assertEqual(len(self.ops()), 4)
        self.assertEqual(self.host.door.busy, {})
        self.assertEqual(self.call(tid, key, b"after")[0], 202)

    # ---- rule 8: off by default; the manifest AND the user's switch in Control --------------------------------------------
    def test_rule8_off_until_the_user_turns_it_on_in_control(self):
        self.consent("stubhook")
        self.host.start("stubhook")
        tid, key = secrets.token_hex(16), "hk_" + secrets.token_hex(24)
        self.plan("stubhook", keys={tid: key})
        self.assertFalse(self.host.door.is_on("stubhook"))                  # the default
        st, _hd, data = self.call(tid, key)
        self.assertEqual((st, jload(data)["code"]), (404, "not_found"))
        self.assertIs(self.hello("stubhook")["hook_door"], False)
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": True}, token=False)
        self.assertEqual(st, 401)                                         # Control's own page, signed in
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": True},
                         headers=[("Origin", "http://evil.example")])
        self.assertEqual(st, 403)
        pid_before = self.host.runs["stubhook"].proc.pid
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": True})
        self.assertEqual((st, j["hook_door"], j["state"]), (200, {"declared": True, "on": True}, "running"))
        self.assertNotEqual(self.host.runs["stubhook"].proc.pid, pid_before)   # restarted: its hello says the new state
        self.assertIs(self.hello("stubhook")["hook_door"], True)
        self.assertEqual(self.call(tid, key, b"on")[0], 202)
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": False})
        self.assertEqual((st, j["hook_door"]["on"]), (200, False))
        self.assertIs(self.hello("stubhook")["hook_door"], False)
        self.assertEqual(self.call(tid, key, b"off again")[0], 404)
        for bad in ("yes", 1, None):
            self.assertEqual(self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": bad})[0], 400, bad)
        events = [(r["event"], r.get("on")) for r in audit_rows() if r["plugin"] == "stubhook" and r["event"] == "hook_door"]
        self.assertEqual(events, [("hook_door", True), ("hook_door", False)])

    def test_rule8_a_manifest_without_hook_door_never_gets_a_door(self):
        self.consent("stubnohook")
        self.host.start("stubnohook")
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubnohook", "on": True})
        self.assertEqual((st, j["code"]), (409, "no_hook_door"))
        self.assertFalse(self.host.door.is_on("stubnohook"))
        self.host.door.set_on("stubnohook", True)                          # even a switch left on from somewhere
        tid, key = secrets.token_hex(16), "hk_" + secrets.token_hex(24)
        self.plan("stubnohook", keys={tid: key})
        st, _hd, data = self.call(tid, key, pid="stubnohook")
        self.assertEqual((st, jload(data)["code"]), (404, "not_found"))
        self.assertEqual(self.ops("stubnohook"), [])
        self.host.stop("stubnohook", "test")
        self.host.start("stubnohook")
        self.assertIs(self.hello("stubnohook")["hook_door"], False)
        # the manifest field is a plain true or false
        m, _d = self.host.manifest("stubhook")
        base = {k: v for k, v in m.items() if k != "hook_door"}
        for ok in (True, False):
            self.assertEqual(P.validate_manifest(dict(base, hook_door=ok)), [])
        for bad in ("yes", 1, None, {"on": True}):
            self.assertTrue(any(p.startswith("hook_door:") for p in P.validate_manifest(dict(base, hook_door=bad))), bad)

    def test_rule8_only_the_user_turns_it_on_never_the_plugin(self):
        self.consent("stubhook")
        self.host.start("stubhook")
        self.assertEqual([n for n in H.HOST_CALLS if "hook" in n or "door" in n], [])
        for name in ("hookdoor.on", "hook_door", "door.open"):
            st, _ct, data = self.host.forward("stubhook", "POST", "/api/plugins/stubhook/call", "", "application/json",
                                              json.dumps({"name": name, "args": {"on": True}}).encode())
            self.assertEqual(json.loads(data)["code"], "unknown_call", name)
        self.assertFalse(self.host.door.is_on("stubhook"))
        # on --lan, a page on another computer may turn it off but never on (like consent); a page here may
        FAKE["peer"] = "192.168.1.50"
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": True}, srv=self.lan_srv)
        self.assertEqual((st, j["code"]), (403, "hook_door_local_only"))
        self.assertFalse(self.host.door.is_on("stubhook"))
        FAKE["peer"] = None
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": True}, srv=self.lan_srv)
        self.assertEqual(st, 200)
        FAKE["peer"] = "192.168.1.50"
        st, j = self.ctl("/api/plugins/host/hookdoor", {"plugin_id": "stubhook", "on": False}, srv=self.lan_srv)
        self.assertEqual((st, j["hook_door"]["on"]), (200, False))

    def test_rule8_a_stopped_plugin_withdrawn_consent_or_an_uninstall_shut_the_door(self):
        tid, key = self.open_door()
        self.host.stop("stubhook", "test")
        self.assertEqual(self.call(tid, key)[0], 404)
        self.assertEqual(self.host.runs["stubhook"].state, "stopped")        # an outside call never starts a plugin
        self.host.start("stubhook")
        self.assertEqual(self.call(tid, key, b"running again")[0], 202)
        self.host.consent.revoke("stubhook")                               # before the host's own recheck stops it
        self.assertEqual(self.call(tid, key, b"no consent")[0], 404)
        self.assertEqual(len(self.ops()), 1)
        gtid, gkey = self.open_door("stubgone")
        self.assertTrue(self.host.door.is_on("stubgone"))
        self.host.uninstall("stubgone")
        self.assertFalse(self.host.door.is_on("stubgone"))                 # a reinstall starts with the door shut
        self.assertEqual(self.call(gtid, gkey, pid="stubgone")[0], 404)

    def test_rule8_the_card_and_the_consent_lines_show_the_door(self):
        self.consent("stubhook")
        self.host.door.set_on("stubhook", True)
        rows_ = {r["plugin_id"]: r for r in PUI.cards(self.host, self.app)["plugins"]}
        self.assertEqual(rows_["stubhook"]["hook_door"], {"declared": True, "on": True})
        self.assertEqual(rows_["stubnohook"]["hook_door"], {"declared": False, "on": False})
        m, _d = self.host.manifest("stubhook")
        line = [x for x in PUI.lines(m) if x["key"] == "hook_door"]
        self.assertEqual(len(line), 1)
        self.assertFalse(line[0]["ask"])                                   # not a tick box: the card's own switch decides
        m2, _d = self.host.manifest("stubnohook")
        self.assertEqual([x for x in PUI.lines(m2) if x["key"] == "hook_door"], [])

    # ---- rule 9: no new port, no listener, no thread ------------------------------------------------------------------------
    def test_rule9_no_new_port_listener_or_thread(self):
        src = read_text(HD.__file__)
        for pat in (r"\bimport socket\b", r"socket\.socket", r"\.listen\(", r"\.bind\(", r"threading\.Thread", r"serve_forever",
                    r"subprocess", r"asyncio", r"selectors"):
            self.assertIsNone(re.search(pat, src), pat)
        tid, key = self.open_door()
        self.assertEqual(self.call(tid, key, b"warm")[0], 202)
        before_l = listening()
        before_t = set(threading.enumerate())
        self.assertTrue(before_l)
        for i in range(6):
            self.assertIn(self.call(tid, key, b"call %d" % i)[0], (202,))
            self.assertEqual(self.call(tid, "hk_x", b"bad %d" % i)[0], 401)
        self.assertEqual(listening(), before_l)
        self.assertTrue(until(lambda: set(threading.enumerate()) <= before_t),
                        "a thread outlived its request")


if __name__ == "__main__":
    unittest.main()
