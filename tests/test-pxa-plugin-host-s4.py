#!/usr/bin/env python3
"""PXA Control's plugin host, S4 (PLUGINS-PROTOTYPE-PLAN S4): the overlay route Control serves (/plugins/<id>/overlay, loopback peers,
its own key, sandboxed page, SSE + RAM audio fed by overlay.* host calls), the key store (secrets.*, per plugin, by name, never
printed) and the egress proxy (CONNECT only, consented scoped hosts, TLS ports, public addresses; sealed in a network namespace on
Linux when scoped hosts are a plugin's only network).

    python3 tests/test-pxa-plugin-host-s4.py

Loopback only: stub plugins built and signed here with a throwaway key, a Control server on 127.0.0.1, an echo server standing in
for an internet host (the proxy's resolver maps the scoped test name to it). No GPU, no internet.
"""
import http.client
import importlib.util
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-plugin-s4-test-")
CFG = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_CONFIG_DIR"] = CFG
os.environ["PXA_STORE_URL"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)

import pxa_control as C  # noqa: E402
from pxa_control import store as S  # noqa: E402
from pxa_control.plugins import egress as EG, host as H, keystore as KS, overlay as OVL, pxp, ui as PUI  # noqa: E402

PLUG_SEED = bytes(range(41, 73))        # throwaway plugin publisher key (never a real key)
PLUG_KID = "plug-test"
PLUG_KEYS = {PLUG_KID: S.ed25519_public(PLUG_SEED).hex()}
FAKE_HOST = "chat.example-stream.tv"    # a scoped host name; the proxy's resolver maps it to the local echo server

STUB = r'''#!/usr/bin/env python3
import json, os, socket, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.environ["PXA_CONTROL_TOOLS"])
from pxa_plugin_child import Plugin
from pxa_control.plugins.egress import open_tunnel


def handle(method, path, query, ctype, body):
    j = json.loads(body or b"{}") if body else {}
    if path.endswith("/env"):
        return 200, "application/json", {k: os.environ.get(k) for k in ("PXA_PLUGIN_EGRESS", "PXA_PLUGIN_EGRESS_MODE", "PXA_PLUGIN_OVERLAY")}
    if path.endswith("/call"):
        return 200, "application/json", PL.call(j.get("name"), j.get("args") or {})
    if path.endswith("/tunnel"):
        try:
            s = open_tunnel(os.environ["PXA_PLUGIN_EGRESS"], j["host"], j["port"], os.environ.get("PXA_PLUGIN_EGRESS_TOKEN"), 5)
            s.sendall(b"ping\n")
            got = s.recv(64)
            s.close()
            return 200, "application/json", {"ok": True, "echo": got.decode()}
        except OSError as e:
            return 200, "application/json", {"ok": False, "error": str(e)}
    if path.endswith("/hold"):
        s = open_tunnel(os.environ["PXA_PLUGIN_EGRESS"], j["host"], j["port"], None, 5)
        HELD.append(s)
        return 200, "application/json", {"ok": True}
    if path.endswith("/held"):
        out = []
        for s in HELD:
            try:
                s.settimeout(2)
                s.sendall(b"x\n")
                out.append(s.recv(8).decode() or "eof")
            except OSError as e:
                out.append("closed")
        return 200, "application/json", {"held": out}
    if path.endswith("/direct"):
        try:
            s = socket.create_connection(("127.0.0.1", int(j["port"])), timeout=3)
            s.close()
            return 200, "application/json", {"reached": True}
        except OSError as e:
            return 200, "application/json", {"reached": False, "error": e.__class__.__name__}
    return 404, "application/json", {"error": "no such stub route"}


HELD = []
PL = Plugin(handle)
sys.exit(PL.run())
'''


def _build_mod():
    spec = importlib.util.spec_from_file_location("build_pxp", os.path.join(TOOLS, "pxa-private", "plugins", "build-pxp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_stub(pid, perms, overlay=False):
    src = os.path.join(TMP, "src", pid)
    for d in ("bin", "ui", "lib", "overlay"):
        os.makedirs(os.path.join(src, d))
    with open(os.path.join(src, "bin", "stub"), "w") as f:
        f.write(STUB)
    with open(os.path.join(src, "ui", "index.html"), "w") as f:
        f.write("<p>stub</p>\n")
    with open(os.path.join(src, "overlay", "index.html"), "w") as f:
        f.write("<!doctype html><title>ov</title><p id=t></p><script>new EventSource('overlay/events'+location.search)</script>\n")
    shutil.copy(os.path.join(TOOLS, "pxa_control", "plugins", "child.py"), os.path.join(src, "lib", "pxa_plugin_child.py"))
    spec = {"schema": 1, "plugin_id": pid, "name": pid.title(), "version": "0.1.0", "publisher": "pxa", "edition": "free",
            "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/stub", "serve"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/%s/" % pid], "ui": {"tab": {"title": pid.title(), "page": "ui/index.html"}},
            "permissions": perms, "entitlement": None, "pack": ["bin", "ui", "lib"] + (["overlay"] if overlay else [])}
    if overlay:
        spec["overlay"] = {"page": "overlay/index.html", "scope": "loopback"}
    with open(os.path.join(src, "plugin.src.json"), "w") as f:
        json.dump(spec, f)
    path, _body = _build_mod().build(src, PLUG_SEED, PLUG_KID, os.path.join(TMP, "out"))
    return pxp.install(path, CFG, PLUG_KEYS)


class Echo(object):
    """a line echo server on 127.0.0.1 standing in for an internet host."""

    def __init__(self):
        self.s = socket.socket()
        self.s.bind(("127.0.0.1", 0))
        self.s.listen(16)
        self.port = self.s.getsockname()[1]
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                c, _a = self.s.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(c,), daemon=True).start()

    def _one(self, c):
        try:
            while True:
                b = c.recv(64)
                if not b:
                    break
                c.sendall(b"echo:" + b)
        except OSError:
            pass
        c.close()


def until(fn, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    return fn()


def audit_rows():
    try:
        with open(os.path.join(CFG, "plugins", "audit.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


class FakeHandler(object):
    """enough of BaseHTTPRequestHandler for overlay.serve, with any peer address."""

    def __init__(self, peer, host):
        self.client_address = (peer, 5555)
        self.headers = {"Host": host}
        self.command = "GET"
        self.code = None
        self.out = []
        import io
        self.wfile = io.BytesIO()

    def send_response(self, code):
        self.code = code

    def send_header(self, k, v):
        self.out.append((k, v))

    def end_headers(self):
        pass


class S4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.echo = Echo()
        make_stub("stubnet", ["net.internet:" + FAKE_HOST, "model.chat"], overlay=True)
        make_stub("stubplain", ["model.chat"])
        make_stub("stublan", ["net.internet:" + FAKE_HOST, "net.lan"])
        resolver = lambda h, p: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", cls.echo.port))]
        cls.opts = {"resolver": resolver, "public_only": False, "ports": (cls.echo.port,)}
        cls.host = H.PluginHost(CFG, cache=S.EntitlementCache(CFG), pubkeys=PLUG_KEYS, health_s=3600, tick_s=3600, backoff=(0,) * 5,
                                hello_timeout_s=10, egress_opts=cls.opts)
        cls.app = C.App(C.desktop.load_launcher(), port=7781, models_dirs=[os.path.join(TMP, "models")])
        cls.app.plugin_host = cls.host
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]
        cls.host.overlays.base = "http://127.0.0.1:%d" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        for pid in ("stubnet", "stubplain", "stublan"):
            self.host.stop(pid, "test")
            m, _d = self.host.manifest(pid)
            self.host.consent.grant(m, m["permissions"])

    def fwd(self, pid, sub, body=None):
        st, _ct, data = self.host.forward(pid, "POST" if body is not None else "GET", "/api/plugins/%s/%s" % (pid, sub), "",
                                          "application/json", json.dumps(body).encode() if body is not None else b"")
        return st, json.loads(data.decode() or "{}")

    def call(self, pid, name, args=None):
        return self.fwd(pid, "call", {"name": name, "args": args or {}})[1]

    def get(self, path, host=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": host or "127.0.0.1:%d" % self.port}
        h.update(headers or {})
        c.request("GET", path, headers=h)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r.headers, body

    # ---- overlay ------------------------------------------------------------------------------------------------------
    def test_overlay_route_serves_the_signed_page_sandboxed_with_its_own_key(self):
        self.host.start("stubnet")
        u = self.call("stubnet", "overlay.url")["result"]
        self.assertTrue(u["url"].startswith("http://127.0.0.1:%d/plugins/stubnet/overlay?k=" % self.port))
        k = u["url"].split("k=")[1]
        self.assertGreaterEqual(len(k), 22)                                                 # 128 bits
        st, hd, body = self.get(u["path"])
        self.assertEqual(st, 200)
        self.assertTrue(hd["Content-Security-Policy"].startswith("sandbox allow-scripts"))
        self.assertIn(b"EventSource", body)
        for bad in ("/plugins/stubnet/overlay", "/plugins/stubnet/overlay?k=", "/plugins/stubnet/overlay?k=" + k[:-1] + "x",
                    "/plugins/stubnet/overlay/events?k=nope"):
            self.assertEqual(self.get(bad)[0], 403, bad)
        self.assertEqual(self.get("/plugins/stubplain/overlay?k=" + k)[0], 404)          # no overlay in its manifest
        self.assertEqual(self.get("/plugins/nobody/overlay?k=" + k)[0], 404)
        self.assertEqual(self.get("/plugins/stubnet/overlay/../ui/index.html?k=" + k)[0], 404)
        self.assertEqual(self.get("/plugins/stubnet/overlay?k=" + k, host="evil.example.com")[0], 421)       # DNS rebinding
        self.host.start("stubplain")
        self.assertEqual(self.call("stubplain", "overlay.url")["code"], "no_overlay")
        mode = stat.S_IMODE(os.stat(os.path.join(CFG, "plugins", "overlay-keys.json")).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(OVL.Overlays(CFG).key("stubnet"), k)                                # survives a Control restart
        self.assertNotIn(k, json.dumps(audit_rows()))

    def test_overlay_refuses_a_lan_peer_even_on_a_lan_bind(self):
        self.host.start("stubnet")
        k = self.call("stubnet", "overlay.url")["result"]["url"].split("k=")[1]
        for peer, host, want in (("192.168.1.50", "192.168.1.10:8080", 403), ("192.168.1.50", "127.0.0.1:8080", 403),
                                 ("127.0.0.1", "192.168.1.10:8080", 403), ("127.0.0.1", "127.0.0.1:8080", 200),
                                 ("::1", "[::1]:8080", 200)):
            h = FakeHandler(peer, host)
            OVL.serve(h, self.host, "GET", "/plugins/stubnet/overlay", {"k": [k]})
            self.assertEqual(h.code, want, (peer, host))
        h = FakeHandler("127.0.0.1", "127.0.0.1:8080")
        OVL.serve(h, self.host, "POST", "/plugins/stubnet/overlay", {"k": [k]})
        self.assertEqual(h.code, 405)

    def test_overlay_events_and_audio_flow_from_host_calls_then_idle_on_stop(self):
        self.host.start("stubnet")
        k = self.call("stubnet", "overlay.url")["result"]["url"].split("k=")[1]
        wav = b"RIFF" + b"\x00" * 60
        self.assertEqual(self.call("stubnet", "overlay.audio", {"name": "a1-0", "b64": __import__("base64").b64encode(wav).decode()})["status"], 200)
        self.assertEqual(self.call("stubnet", "overlay.emit", {"event": {"type": "caption", "id": "a1", "text": "hi <b>"}})["status"], 200)
        self.assertEqual(self.call("stubnet", "overlay.emit", {"event": {"type": "x", "pad": "y" * 9000}})["code"], "too_large")
        self.assertEqual(self.call("stubnet", "overlay.emit", {"event": "nope"})["code"], "invalid")
        self.assertEqual(self.call("stubnet", "overlay.audio", {"name": "../x", "b64": "UklGRg=="})["code"], "invalid")
        self.assertEqual(self.call("stubnet", "overlay.audio", {"name": "ok", "b64": "aGVsbG8="})["code"], "invalid")    # not a WAV
        st, hd, body = self.get("/plugins/stubnet/overlay/audio/a1-0.wav?k=" + k)
        self.assertEqual((st, hd["Content-Type"], body, hd["Access-Control-Allow-Origin"]), (200, "audio/wav", wav, "*"))
        self.assertEqual(self.get("/plugins/stubnet/overlay/audio/zz.wav?k=" + k)[0], 404)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", "/plugins/stubnet/overlay/events?k=" + k)
        r = c.getresponse()
        self.assertEqual((r.status, r.headers["Content-Type"], r.headers["Access-Control-Allow-Origin"]), (200, "text/event-stream", "*"))
        buf, end = b"", time.time() + 5
        while b'"caption"' not in buf and time.time() < end:
            buf += r.read1(65536)
        self.assertIn(b'"text": "hi <b>"', buf)
        threading.Timer(0.3, lambda: self.host.stop("stubnet", "grant")).start()           # a lost grant: the overlay shows nothing
        end = time.time() + 5
        while b'"idle"' not in buf and time.time() < end:
            buf += r.read1(65536)
        c.close()
        self.assertIn(b'"type": "idle"', buf)
        self.assertEqual(self.get("/plugins/stubnet/overlay/audio/a1-0.wav?k=" + k)[0], 404)  # the audio is dropped with it

    def test_rotate_drops_open_streams_and_the_old_key(self):
        self.host.start("stubnet")
        old = self.call("stubnet", "overlay.url")["result"]["url"].split("k=")[1]
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", "/plugins/stubnet/overlay/events?k=" + old)
        r = c.getresponse()
        r.read1(100)
        new = self.call("stubnet", "overlay.rotate")["result"]["url"].split("k=")[1]
        self.assertNotEqual(old, new)
        end, data = time.time() + 5, b"x"
        while data and time.time() < end:
            data = r.read1(65536)
        self.assertEqual(data, b"")                                                         # the stream ended
        c.close()
        self.assertEqual(self.get("/plugins/stubnet/overlay?k=" + old)[0], 403)
        self.assertEqual(self.get("/plugins/stubnet/overlay?k=" + new)[0], 200)

    # ---- key store ----------------------------------------------------------------------------------------------------
    def test_key_store_is_per_plugin_by_name_and_never_printed(self):
        self.host.start("stubnet")
        self.host.start("stubplain")
        val = "refresh-token-THIS-IS-SECRET-42"
        r = self.call("stubnet", "secrets.put", {"name": "youtube", "value": val})
        self.assertEqual(r["result"], {"ref": "secret:youtube", "bytes": len(val)})
        self.assertEqual(self.call("stubnet", "secrets.get", {"name": "secret:youtube"})["result"]["value"], val)
        self.assertEqual(self.call("stubplain", "secrets.get", {"name": "youtube"})["code"], "not_found")   # another plugin's names: none
        self.assertEqual(self.call("stubplain", "secrets.names")["result"]["names"], [])
        self.assertEqual(self.call("stubnet", "secrets.names")["result"]["names"], ["youtube"])
        self.assertEqual(self.call("stubnet", "secrets.put", {"name": "../x", "value": "v"})["code"], "invalid")
        self.assertEqual(self.call("stubnet", "secrets.put", {"name": "big", "value": "v" * 9000})["code"], "invalid")
        for i in range(15):
            self.call("stubnet", "secrets.put", {"name": "k%d" % i, "value": "v"})
        self.assertEqual(self.call("stubnet", "secrets.put", {"name": "one-more", "value": "v"})["code"], "too_many")
        p = os.path.join(CFG, "plugins", "secrets", "stubnet.json")
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode), 0o700)
        with open(os.path.join(CFG, "plugins", "audit.jsonl")) as f:
            self.assertNotIn(val, f.read())
        self.assertIn("youtube", [r.get("key") for r in audit_rows() if r["event"] == "secrets_put"])
        self.assertNotIn(val, repr(self.host.keys) + json.dumps(self.host.status()))
        card = {r["plugin_id"]: r for r in PUI.cards(self.host, self.app)["plugins"]}
        self.assertEqual((card["stubnet"]["keys"], card["stubplain"]["keys"]), (16, 0))     # the card shows a count, never a name
        self.assertNotIn("youtube", json.dumps(card["stubnet"]))
        st, j, _h = self.ctl("POST", "/api/plugins/host/keys/forget", {"plugin_id": "stubnet"})
        self.assertEqual((st, j["forgotten"]), (200, 16))
        self.assertEqual(self.call("stubnet", "secrets.names")["result"]["names"], [])
        self.assertTrue(KS.is_ref("secret:youtube") and not KS.is_ref("AIza-something"))

    def ctl(self, method, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, json.dumps(body), {"Content-Type": "application/json", "X-PXA-Token": self.app.token,
                                                   "Host": "127.0.0.1:%d" % self.port})
        r = c.getresponse()
        b = r.read()
        c.close()
        return r.status, json.loads(b or b"{}"), r.headers

    # ---- egress -------------------------------------------------------------------------------------------------------
    def test_egress_lets_the_consented_host_through_and_nothing_else(self):
        self.host.start("stubnet")
        env = self.fwd("stubnet", "env")[1]
        self.assertTrue(env["PXA_PLUGIN_EGRESS"].startswith("unix:"))
        self.assertEqual(env["PXA_PLUGIN_OVERLAY"], "host")
        ok = self.fwd("stubnet", "tunnel", {"host": FAKE_HOST, "port": self.echo.port})[1]
        self.assertEqual(ok, {"ok": True, "echo": "echo:ping\n"})
        ok = self.fwd("stubnet", "tunnel", {"host": FAKE_HOST.upper() + ".", "port": self.echo.port})[1]
        self.assertTrue(ok["ok"])                                                           # case and a trailing dot are the same name
        for host, port, code in (("example.org", self.echo.port, "not_allowed"), (FAKE_HOST, 22, "port"),
                                 ("localhost", self.echo.port, "not_allowed"), ("127.0.0.1", self.echo.port, "not_allowed")):
            r = self.fwd("stubnet", "tunnel", {"host": host, "port": port})[1]
            self.assertFalse(r["ok"], host)
            self.assertIn(code, r["error"], host)
        rows = [r for r in audit_rows() if r["event"] == "egress_denied"]
        self.assertIn("example.org", [r.get("host") for r in rows])
        run = self.host.runs["stubnet"]
        run.egress.public_only = True                                                       # the real rule: the name resolves to 127.0.0.1
        r = self.fwd("stubnet", "tunnel", {"host": FAKE_HOST, "port": self.echo.port})[1]
        run.egress.public_only = False
        self.assertIn("private_address", r["error"])
        st = self.host.status_of("stubnet")["egress"]
        self.assertGreaterEqual(st["tunnels"], 2)
        self.assertGreaterEqual(st["refused"], 5)

    def test_egress_refuses_plain_requests_and_strangers(self):
        self.host.start("stubnet")
        addr = self.host.runs["stubnet"].egress.address
        with self.assertRaises(OSError) as e:                                               # this test process is not the plugin
            EG.open_tunnel(addr, FAKE_HOST, self.echo.port, timeout=5)
        self.assertIn("bad_peer", str(e.exception))
        self.host.runs["stubnet"].egress.pgid = os.getpgid(0)                              # pretend we are: then a GET is still refused
        s = socket.socket(socket.AF_UNIX)
        s.connect(addr[5:])
        s.sendall(b"GET http://%s/ HTTP/1.1\r\nHost: %s\r\n\r\n" % (FAKE_HOST.encode(), FAKE_HOST.encode()))
        self.assertIn(b"405", s.recv(200))
        s.close()

    def test_withdrawn_consent_closes_the_door_at_once_and_stop_cuts_tunnels(self):
        self.host.start("stubnet")
        self.assertTrue(self.fwd("stubnet", "hold", {"host": FAKE_HOST, "port": self.echo.port})[1]["ok"])
        self.assertEqual(self.fwd("stubnet", "held")[1]["held"], ["echo:x\n"])
        m, _d = self.host.manifest("stubnet")
        self.host.consent.grant(m, ["model.chat"])                                          # the scoped host unticked
        r = self.fwd("stubnet", "tunnel", {"host": FAKE_HOST, "port": self.echo.port})[1]
        self.assertIn("not_allowed", r["error"])
        self.host.consent.grant(m, m["permissions"])
        eg = self.host.runs["stubnet"].egress
        self.host.stop("stubnet", "test")
        self.assertTrue(eg.closed.is_set())
        self.assertEqual(eg.tunnels, set())
        self.assertFalse(os.path.exists(eg.address[5:]))

    def test_sealed_on_linux_when_scoped_hosts_are_the_only_network(self):
        self.host.start("stubnet")
        env = self.fwd("stubnet", "env")[1]
        d = self.fwd("stubnet", "direct", {"port": self.echo.port})[1]
        if EG.netns_prefix():
            self.assertEqual(env["PXA_PLUGIN_EGRESS_MODE"], "sealed")
            self.assertFalse(d["reached"])                                                  # no loopback, no LAN, no internet: only the door
            self.assertEqual(self.host.status_of("stubnet")["egress"]["mode"], "sealed")
        else:
            self.assertEqual(env["PXA_PLUGIN_EGRESS_MODE"], "proxy")
        self.assertTrue(self.fwd("stubnet", "tunnel", {"host": FAKE_HOST, "port": self.echo.port})[1]["ok"])   # the door still works
        self.host.start("stublan")                                                          # net.lan: the proxy, never sealed
        env = self.fwd("stublan", "env")[1]
        self.assertEqual(env["PXA_PLUGIN_EGRESS_MODE"], "proxy")
        self.assertTrue(self.fwd("stublan", "direct", {"port": self.echo.port})[1]["reached"])
        self.host.start("stubplain")                                                        # no scoped host: no proxy, no namespace (as before S4)
        env = self.fwd("stubplain", "env")[1]
        self.assertEqual((env["PXA_PLUGIN_EGRESS"], env["PXA_PLUGIN_EGRESS_MODE"], env["PXA_PLUGIN_OVERLAY"]), (None, None, None))
        self.assertNotIn("egress", self.host.status_of("stubplain"))

    def test_netns_off_switch(self):
        saved = dict(EG._NETNS)
        os.environ["PXA_PLUGIN_NETNS"] = "0"
        EG._NETNS.clear()
        try:
            self.assertIsNone(EG.netns_prefix())
            self.host.start("stubnet")
            self.assertEqual(self.fwd("stubnet", "env")[1]["PXA_PLUGIN_EGRESS_MODE"], "proxy")
            self.assertTrue(self.fwd("stubnet", "direct", {"port": self.echo.port})[1]["reached"])
        finally:
            os.environ.pop("PXA_PLUGIN_NETNS", None)
            EG._NETNS.clear()
            EG._NETNS.update(saved)
            self.host.stop("stubnet", "test")

    def test_host_web_fetch_keeps_the_seal(self):
        """Review item 3: net.fetch / net.search run INSIDE the host (HOST_CALLS -> net.http), so asking for them gives the
        child no network of its own: a scoped-host plugin that adds net.http stays sealed."""
        self.assertEqual({H.HOST_CALLS["net.fetch"], H.HOST_CALLS["net.search"]}, {"net.http"})
        self.assertNotIn("net.http", EG.DIRECT_NET)
        self.assertTrue(EG.sealable({}, ["net.internet:" + FAKE_HOST, "net.http"]))
        self.assertTrue(EG.sealable({"listen": None}, ["net.internet:" + FAKE_HOST, "net.http", "model.chat"]))
        for direct in ("net.loopback", "net.lan", "net.internet"):                        # the child's own network: no seal
            self.assertFalse(EG.sealable({}, ["net.internet:" + FAKE_HOST, "net.http", direct]), direct)
        self.assertFalse(EG.sealable({}, ["net.http"]))                                    # no scoped host: no proxy at all

    def test_hello_names_the_host_calls(self):
        self.assertIn("overlay.emit", H.HOST_CALLS)
        self.assertIsNone(H.HOST_CALLS["secrets.get"])                                     # its own state: no permission to ask for
        self.assertTrue(EG.sealable({"listen": None}, ["net.internet:" + FAKE_HOST]))
        self.assertFalse(EG.sealable({"listen": [{"port": 1}]}, ["net.internet:" + FAKE_HOST]))
        self.assertFalse(EG.sealable({}, ["net.internet:" + FAKE_HOST, "net.loopback"]))
        self.assertFalse(EG.wants_proxy({}, ["net.internet", "model.chat"]))
        v6 = lambda a: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (a, 443, 0, 0))]
        self.assertFalse(EG._public(v6("::ffff:127.0.0.1")))                                # an IPv4-mapped private address stays private
        self.assertFalse(EG._public(v6("::ffff:10.0.0.1")))
        self.assertFalse(EG._public([(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.9", 443))]))
        self.assertTrue(EG._public([(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]))
        for a in ("64:ff9b::a00:1", "64:ff9b:1::a00:1", "2002:c0a8:109::1", "2002:7f00:1::1"):   # NAT64 / 6to4 wrapping a home address
            self.assertFalse(EG._public(v6(a)), a)
        self.assertTrue(EG._public(v6("2606:4700:4700::1111")))


class TunnelRefusedAtOnce(unittest.TestCase):
    """a proxy that refuses on accept (bad_peer) answers and closes before reading the CONNECT: the client's write can then
    fail (EPIPE) but the refusal code is still read and raised (a busy box made test_egress_refuses_... flaky)."""

    def test_the_refusal_code_survives_a_closed_pipe(self):
        d = tempfile.mkdtemp(prefix="pxa-eg-")
        path = os.path.join(d, "eg.sock")
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(path)
        srv.listen(1)
        closed = threading.Event()

        def serve():
            c, _a = srv.accept()
            c.sendall(b"HTTP/1.1 403 Forbidden\r\nX-Pxa-Egress: bad_peer\r\nContent-Length: 0\r\n\r\n")
            c.close()
            closed.set()
        threading.Thread(target=serve, daemon=True).start()

        class Late(socket.socket):                     # the client writes only after the proxy has closed
            def connect(self, a):
                super().connect(a)
                closed.wait(5)
                time.sleep(0.05)
        try:
            with mock.patch.object(EG.socket, "socket", Late):
                with self.assertRaises(OSError) as e:
                    EG.open_tunnel("unix:" + path, "example.org", 443, timeout=5)
            self.assertIn("bad_peer", str(e.exception))
        finally:
            srv.close()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
