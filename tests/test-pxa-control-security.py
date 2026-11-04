#!/usr/bin/env python3
"""PXA Control hardening (the Control hardening pass; threat model THREAT-MODEL.md, fixes H1-H10).

One test class per fix: the per-session token on loopback (H1), CSRF / DNS rebinding (H2), one cookie per port
(H3), the drafter rule in extra args (H4), docker names (H5), download names (H6), limits (H7), secrets in logs,
banners and answers (H8), security headers and the CSP (H9), query-string tokens (H10). No GPU, no network beyond
127.0.0.1.

    python3 tests/test-pxa-control-security.py
"""
import base64
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
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
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-sec-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""          # hermetic: the fixture store, never the live licd (the release default)
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN",
           "PXA_MODELS_DIR", "PXA_CONTROL_ACCESS_LOG", "PXA_CONTROL_HF_TOKEN", "PXA_CONTROL_MQTT_PASSWORD"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402
import pxa_hf as HF  # noqa: E402

MODELS = os.path.join(TMP, "models")
OUTSIDE = os.path.join(TMP, "outside")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTSIDE, exist_ok=True)
GGUF = b"GGUF\x03\x00\x00\x00" + b"\x00" * 16
for _p in (os.path.join(MODELS, "tiny.gguf"), os.path.join(MODELS, "draft.gguf"), os.path.join(OUTSIDE, "evil.gguf")):
    with open(_p, "wb") as _f:
        _f.write(GGUF)

# four secrets the process may hold; none may ever come back in an answer (H8)
HF_SECRET = "hf_" + "Q" * 34
MQTT_SECRET = "mqtt-pass-" + "Z" * 20
LIC_SECRET = "pxk1.pro.ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"


def serve(app, max_connections=None, timeout=None):
    srv = C.make_server(app, "127.0.0.1", 0)
    if max_connections is not None:
        srv._slots = threading.BoundedSemaphore(max_connections)
    if timeout is not None:
        srv.RequestHandlerClass.timeout = timeout
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def req(port, path, method="GET", body=None, headers=None, raw=None):
    h = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h, method=method)
    host = h.pop("Host", None)
    if host:
        r.add_unredirected_header("Host", host)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    try:
        with urllib.request.build_opener(NoRedirect).open(r, timeout=20) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def tok(app):
    return {"X-PXA-Token": app.token}


class Base(unittest.TestCase):
    lan = False

    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=7777, lan=cls.lan, token=("lan-token-" + "x" * 20) if cls.lan else None,
                        models_dirs=[MODELS])
        cls.app.auth_throttle = C.AuthThrottle(limit=10 ** 6)          # the crawl below is many refusals
        cls.srv, cls.port = serve(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()


# ---- H1 ---------------------------------------------------------------------------------------------------
class SessionToken(Base):
    def test_a_loopback_control_has_a_fresh_unsaved_token(self):
        self.assertGreaterEqual(len(self.app.token), 32)
        self.assertNotEqual(C.App(L, port=7777).token, C.App(L, port=7777).token)
        self.assertFalse(os.path.exists(os.path.join(C.config_dir(), "token")),
                         "the per-session token must not be written where the LAN token lives")

    def test_every_api_route_needs_it_except_health_and_metrics(self):
        routes = sorted(C.ROUTES) + [("GET", "/api/engine/health"), ("POST", "/api/engine/v1/chat/completions"),
                                      ("GET", "/api/log/stream")]
        open_ = []
        for method, path in routes:
            st, _h, _b = req(self.port, path, method, body={} if method != "GET" else None)
            if st != 401:
                open_.append((method, path, st))
        # public code under /api/ (the Vigil page/HUD shells and their fixed assets; review #53657) is open like STATIC
        pub = sorted(C.PUBLIC_GET)
        self.assertTrue(all(p in ("/api/chat/vigil/page", "/api/chat/vigil/hud") or p.startswith("/api/chat/vigil/ui/")
                            for p in pub), pub)
        self.assertEqual(sorted((m, p) for m, p, _st in open_ if p not in C.PUBLIC_GET),
                         [("GET", "/api/health"), ("GET", "/metrics")])
        self.assertEqual(dict(((m, p), st) for m, p, st in open_)[("GET", "/api/health")], 200)   # /metrics: 404 when off
        for method, path in routes[:40]:                               # and the token opens them (not 401)
            st, _h, _b = req(self.port, path, method, body={} if method != "GET" else None, headers=tok(self.app))
            self.assertNotEqual(st, 401, (method, path))

    def test_the_page_shell_is_open_the_token_signs_the_tab_in(self):
        # control-cookie-port-scope: a navigation cannot carry the header, so the shell (public code) loads without
        # the token and signs the tab in from sessionStorage; a ?token= visit moves the token to the fragment
        st, h, b = req(self.port, "/")
        self.assertEqual(st, 200)
        self.assertIn(b'<script src="/auth.js"></script>', b)
        self.assertEqual(req(self.port, "/chat.js")[0], 200)
        self.assertEqual(req(self.port, "/mark.png")[0], 200)
        st, h, _b = req(self.port, "/?token=" + self.app.token)
        self.assertEqual(st, 303)
        self.assertEqual(h["Location"], "/#token=" + self.app.token)
        self.assertNotIn("Set-Cookie", h)
        st, _h, b = req(self.port, "/?token=wrong" + "x" * 30)
        self.assertEqual(st, 401)
        self.assertIn(b"run <code>pxa</code> again", b)                 # the loopback sign-in hint
        st, _h, b = req(self.port, "/login")
        self.assertEqual(st, 200)
        self.assertIn(b"run <code>pxa</code> again", b)

    def test_every_html_page_signs_in_through_auth_js(self):
        # review 10-18 night: /share.html joined STATIC without joining PAGE_PATHS; every page that calls the API must
        # load /auth.js first and take the ?token= redirect. /hosts.html is the exception: the PWA's offline host
        # picker (test_pxa_control_pwa), no API, no user data
        pages = sorted(p for p, (_fn, ct) in C.STATIC.items() if ct.startswith("text/html") and p != "/hosts.html")
        self.assertIn("/share.html", pages)
        for p in pages:
            st, h, b = req(self.port, p)
            self.assertEqual(st, 200, p)
            self.assertIn(b'<script src="/auth.js"></script>', b.split(b"</head>")[0], p)
            st, h, _b = req(self.port, p + "?token=" + self.app.token)
            self.assertEqual(st, 303, p)
            self.assertEqual(h["Location"], p + "#token=" + self.app.token)

    def test_wrong_tokens_and_bearer(self):
        for h in ({"X-PXA-Token": "nope"}, {"X-PXA-Token": self.app.token[:-1]}, {"Authorization": "Bearer x" * 3},
                  {"Cookie": "pxa_control_token_7777=" + self.app.token + "x"}, {"X-PXA-Token": "é" * 32}):
            self.assertEqual(req(self.port, "/api/info", headers=h)[0], 401, h)
        self.assertEqual(req(self.port, "/api/info", headers={"Authorization": "Bearer " + self.app.token})[0], 200)
        info = json.loads(req(self.port, "/api/info", headers=tok(self.app))[2])
        self.assertTrue(info["token_required"])

    def test_run_record_keeps_the_token_private_and_reopens_signed_in(self):
        p = C.write_run_record(self.port, False, False, self.app.token)
        try:
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode), 0o700)
            r = C.find_running(port=self.port)
            self.assertEqual(r["url"], f"http://127.0.0.1:{self.port}/?token={self.app.token}")
            self.assertTrue(C.control_answers(self.port))                # /api/health: no token needed to see it is up
        finally:
            os.unlink(p)
        self.assertEqual(C.control_url(7777), "http://127.0.0.1:7777/")
        self.assertEqual(C.control_url(7777, token="abc"), "http://127.0.0.1:7777/?token=abc")

    def test_explicit_no_token_mode_is_test_only_and_open(self):
        a = C.App(L, port=7777, token="")
        self.assertIsNone(a.token)
        srv, port = serve(a)
        try:
            self.assertEqual(req(port, "/api/info")[0], 200)
        finally:
            srv.shutdown()
            srv.server_close()


class LanToken(Base):
    lan = True

    def test_only_the_public_code_is_open_on_the_lan(self):
        for path in ("/api/health", "/metrics", "/api/info"):
            self.assertEqual(req(self.port, path)[0], 401, path)
        for path in ("/chat.js", "/", "/auth.js", "/login"):           # the shell signs in from sessionStorage
            self.assertEqual(req(self.port, path)[0], 200, path)
        self.assertEqual(req(self.port, "/api/health", headers=tok(self.app))[0], 200)
        self.assertEqual(req(self.port, "/api/health", headers={"Host": "rig.lan:7777", **tok(self.app)})[0], 200)


# ---- H2 ---------------------------------------------------------------------------------------------------
class CrossOrigin(Base):
    def test_fetch_metadata(self):
        for site, want in (("cross-site", 403), ("same-site", 403), ("same-origin", 200), ("none", 200), ("", 200)):
            h = dict(tok(self.app))
            if site:
                h["Sec-Fetch-Site"] = site
            self.assertEqual(req(self.port, "/api/info", headers=h)[0], want, site)
            self.assertEqual(req(self.port, "/api/stop", "POST", {}, headers=h)[0], want, site)
        # a navigation from another site to the page is fine (it gets the public shell, never the API)
        self.assertEqual(req(self.port, "/", headers={"Sec-Fetch-Site": "cross-site"})[0], 200)
        self.assertEqual(req(self.port, "/api/info", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)

    def test_origin_must_be_exactly_this_host(self):
        me = f"http://127.0.0.1:{self.port}"
        cases = (("http://evil.example", 403), ("null", 403), (f"{me}/x", 403), (f"ftp://127.0.0.1:{self.port}", 403),
                 (f"http://127.0.0.1:{self.port + 1}", 403), (f"http://localhost:{self.port}", 403), (me, 200))
        for origin, want in cases:
            self.assertEqual(req(self.port, "/api/stop", "POST", {}, headers={"Origin": origin, **tok(self.app)})[0],
                             want, origin)
            self.assertEqual(req(self.port, "/api/info", headers={"Origin": origin, **tok(self.app)})[0], want, origin)

    def test_dns_rebinding_host_refused(self):
        for host in ("evil.example", f"evil.example:{self.port}", "127.0.0.1.evil.example", ""):
            st = req(self.port, "/api/health", headers={"Host": host} if host else {"Host": " "})[0]
            self.assertEqual(st, 421, host)
        self.assertEqual(req(self.port, "/api/health", headers={"Host": f"localhost:{self.port}"})[0], 200)


# ---- H3 (control-cookie-port-scope) -------------------------------------------------------
# Browsers send a host's cookies to EVERY port of it, so any other listener on 127.0.0.1 (or on the rig, --lan) got
# Control's token cookie and could replay it: owner access. A per-port NAME does not help (still sent everywhere).
# The token lives in the tab (sessionStorage) and rides the X-PXA-Token header; no answer sets a cookie, and a cookie
# never signs anything in. Navigations that cannot carry a header (frames, downloads, script tags, new tabs) use a
# short-lived single-use ticket bound to one path, minted through the header.
def ticket(app, port, path):
    st, _h, b = req(port, "/api/auth/ticket", "POST", {"path": path}, headers=tok(app))
    assert st == 200, (st, b)
    return json.loads(b)["ticket"]


class TokenNeverInACookie(Base):
    def visits(self):
        t = self.app.token
        return [("GET", "/", {}), ("GET", "/?token=" + t, {}), ("GET", "/share.html?token=" + t, {}),
                ("GET", "/?token=wrong" + "x" * 30, {}), ("GET", "/login", {}), ("GET", "/chat.js", {}),
                ("GET", "/api/info", tok(self.app)), ("GET", "/api/info?token=" + t, {}), ("GET", "/api/info", {}),
                ("GET", "/api/health", {}), ("GET", "/api/info", {"Cookie": f"{self.app.cookie_name}={t}"}),
                ("POST", "/api/auth/ticket", tok(self.app))]

    def test_a_no_answer_ever_sets_a_cookie(self):
        for method, path, h in self.visits():
            st, hd, body = req(self.port, path, method, body={"path": "/api/info"} if method == "POST" else None, headers=h)
            cookies = [v for k, v in hd.items() if k.lower() == "set-cookie"]
            self.assertEqual(cookies, [], (method, path, st))
            self.assertNotIn(self.app.token.encode(), body if path != "/api/auth/ticket" else b"", (method, path))

    def test_b_a_cookie_alone_is_refused(self):
        t = self.app.token
        for ck in (f"{self.app.cookie_name}={t}", f"pxa_control_token={t}", f"a=b; {self.app.cookie_name}={t}; pxa_control_token={t}"):
            self.assertEqual(req(self.port, "/api/info", headers={"Cookie": ck})[0], 401, ck)
            self.assertEqual(req(self.port, "/api/stop", "POST", {}, headers={"Cookie": ck})[0], 401, ck)
            self.assertEqual(req(self.port, "/api/auth/ticket", "POST", {"path": "/api/info"}, headers={"Cookie": ck})[0], 401, ck)
        self.assertFalse(hasattr(self.app, "cookie_tokens"))

    def test_c_the_header_works_a_bad_header_does_not(self):
        self.assertEqual(req(self.port, "/api/info", headers=tok(self.app))[0], 200)
        self.assertEqual(req(self.port, "/api/info", headers={**tok(self.app), "Cookie": "pxa_control_token=junk"})[0], 200)
        for bad in ("nope", self.app.token[:-1], self.app.token + "x", "é" * 32, ""):
            self.assertEqual(req(self.port, "/api/info", headers={"X-PXA-Token": bad})[0], 401, bad)
            self.assertEqual(req(self.port, "/api/info", headers={"X-PXA-Token": bad,
                                                                  "Cookie": f"{self.app.cookie_name}={self.app.token}"})[0], 401, bad)

    def test_d_a_first_visit_redirects_to_an_address_without_the_token(self):
        t = self.app.token
        for p, want in (("/?token=" + t, "/"), ("/share.html?token=" + t, "/share.html"), ("/index.html?token=" + t, "/index.html"),
                        ("/?theme=dark&token=" + t, "/?theme=dark")):
            st, h, _b = req(self.port, p)
            self.assertEqual(st, 303, p)
            loc = h["Location"]
            server_part, _sep, frag = loc.partition("#")
            self.assertEqual(server_part, want, p)               # what reaches a server again has no token in it
            self.assertNotIn(t, server_part)
            self.assertEqual(frag, "token=" + urllib.parse.quote(t, safe=""), p)   # the tab reads it once, then drops it
        st, _h, b = req(self.port, "/?token=wrong" + "x" * 30)
        self.assertEqual(st, 401)
        self.assertIn(b"<form", b)

    def test_the_page_shell_is_public_code_the_api_is_not(self):
        st, _h, b = req(self.port, "/")
        self.assertEqual(st, 200)
        head = b.split(b"</head>")[0]
        first = re.search(rb"<script\b[^>]*\bsrc=\"?([^\s\">]+)", head)
        self.assertEqual(first.group(1), b"/auth.js")        # signs the tab in before any other script asks the API
        self.assertEqual(req(self.port, "/auth.js")[0], 200)
        st, _h, b = req(self.port, "/login")
        self.assertEqual(st, 200)
        self.assertIn(b"name=token", b)
        self.assertEqual(req(self.port, "/api/info")[0], 401)

    def test_tickets_are_single_use_path_bound_and_short_lived(self):
        t = ticket(self.app, self.port, "/api/info")
        self.assertNotEqual(t, self.app.token)
        self.assertEqual(req(self.port, "/api/info?ticket=" + t)[0], 200)
        self.assertEqual(req(self.port, "/api/info?ticket=" + t)[0], 401)          # used up
        t = ticket(self.app, self.port, "/api/info")
        self.assertEqual(req(self.port, "/api/status?ticket=" + t)[0], 401)        # another path
        self.assertEqual(req(self.port, "/api/info?ticket=" + t)[0], 401)          # a refused try also spends it
        t = ticket(self.app, self.port, "/api/stop")
        self.assertEqual(req(self.port, "/api/stop?ticket=" + t, "POST", {})[0], 401)   # GET only
        t = ticket(self.app, self.port, "/api/info")
        self.app.tickets.age(t, 10 ** 6)
        self.assertEqual(req(self.port, "/api/info?ticket=" + t)[0], 401)          # expired
        self.assertEqual(req(self.port, "/api/auth/ticket", "POST", {"path": "/api/info"})[0], 401)   # minting needs the header
        for bad in ("/", "api/info", "/api/info?x=1", "//evil/api", "/api/../x", 7, ""):
            self.assertEqual(req(self.port, "/api/auth/ticket", "POST", {"path": bad}, headers=tok(self.app))[0], 400, bad)
        self.assertEqual(req(self.port, "/api/info?ticket=" + "x" * 32)[0], 401)

    def test_the_ui_never_uses_a_cookie_and_sends_the_header(self):
        ui = os.path.join(TOOLS, "pxa_control_ui")
        with open(os.path.join(ui, "auth.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("sessionStorage", js)
        self.assertIn("X-PXA-Token", js)
        for d in (ui, os.path.join(TOOLS, "pxa_chat", "ui")):
            for n in sorted(os.listdir(d)):
                if n.endswith((".js", ".html")):
                    with open(os.path.join(d, n), encoding="utf-8", errors="replace") as f:
                        self.assertNotIn("document.cookie", f.read(), n)
        for page in ("index.html", "share.html"):
            with open(os.path.join(ui, page), encoding="utf-8") as f:
                head = f.read().split("</head>")[0]
            self.assertIn('<script src="/auth.js"></script>', head, page)


class TokenNeverInACookieLan(TokenNeverInACookie):
    lan = True

    def test_the_page_shell_is_public_code_the_api_is_not(self):
        super().test_the_page_shell_is_public_code_the_api_is_not()
        for path in ("/api/health", "/metrics", "/api/info"):          # on the LAN only the code is public
            self.assertEqual(req(self.port, path)[0], 401, path)


class SignInEverywhere(Base):
    """review of c25f0e952c (#53657): the Vigil pages Control serves, exports, frame tickets, HEAD /login."""

    def vigil(self):
        if ("GET", "/api/chat/vigil/page") not in C.ROUTES:
            self.skipTest("the chat agent is not importable here")
        # MED-1 (27796c4076): every /api/chat/vigil/* route answers 403 locked while Control holds no supporter pass for
        # Anibal (pxa_chat/test_vigil_pass.py). These tests are about signing in, so they run with a pass, made the
        # test-only way (anibal/tests/supporter_pass.py) in Control's own config folder, which the hub reads.
        try:
            from anibal.tests import supporter_pass
        except ImportError:
            self.skipTest("the Anibal plugin is not in this tree")
        supporter_pass.unlock(self, cfg_dir=C.config_dir())

    def test_vigil_pages_sign_in_through_auth_js_and_their_code_loads(self):
        self.vigil()
        for path in ("/api/chat/vigil/page", "/api/chat/vigil/hud"):
            for h in ({}, tok(self.app), None):
                if h is None:                                   # the way Control opens it: a ticket
                    st, hd, b = req(self.port, path + "?ticket=" + ticket(self.app, self.port, path))
                else:
                    st, hd, b = req(self.port, path, headers=h)
                self.assertEqual(st, 200, (path, h))
                html = b.decode()
                head = html.split("</head>")[0]
                first = re.search(r"<script\b[^>]*\bsrc=\"([^\"]+)\"", head)
                self.assertEqual(first.group(1), "/auth.js", path)  # before every other script
                m = re.search(r"script-src 'nonce-([^']+)'", hd.get("Content-Security-Policy", ""))
                if m:                                           # the HUD: only nonce'd scripts run
                    self.assertIn('<script nonce="%s" src="/auth.js"></script>' % m.group(1), html)
                self.assertNotIn(self.app.token, html)
            srcs = sorted(set(re.findall(r'(?:src|href)="(/api/chat/vigil/ui/[^"?]+)"', html)))
            self.assertTrue(srcs, path)
            for s_ in srcs:                                     # its code: public, like Control's own scripts
                self.assertEqual(req(self.port, s_)[0], 200, s_)
                self.assertEqual(req(self.port, s_, headers=tok(self.app))[0], 200, s_)
        for path in ("/api/chat/vigil/pane/state", "/api/chat/vigil/events", "/api/chat/vigil/pane/hud"):
            self.assertEqual(req(self.port, path)[0], 401, path)          # its data: the token, as ever

    def test_tickets_bind_the_query_too(self):
        st, _h, b = req(self.port, "/api/auth/ticket", "POST", {"path": "/api/info", "query": "x=1"}, headers=tok(self.app))
        self.assertEqual(st, 200, b)
        t = json.loads(b)["ticket"]
        self.assertEqual(req(self.port, "/api/info?x=2&ticket=" + t)[0], 401)          # another query
        t = json.loads(req(self.port, "/api/auth/ticket", "POST", {"path": "/api/info", "query": "x=1"}, headers=tok(self.app))[2])["ticket"]
        self.assertEqual(req(self.port, "/api/info?ticket=" + t + "&x=1")[0], 200)     # same query, any order
        t = ticket(self.app, self.port, "/api/info")
        self.assertEqual(req(self.port, "/api/info?x=1&ticket=" + t)[0], 401)          # no query bound: none allowed
        fr = "/api/plugins/host/frame"
        t = json.loads(req(self.port, "/api/auth/ticket", "POST", {"path": fr, "query": "plugin_id=a"}, headers=tok(self.app))[2])["ticket"]
        self.assertEqual(req(self.port, fr + "?plugin_id=b&ticket=" + t)[0], 401)       # a frame ticket opens only its plugin
        t = json.loads(req(self.port, "/api/auth/ticket", "POST", {"path": fr, "query": "plugin_id=a"}, headers=tok(self.app))[2])["ticket"]
        self.assertNotEqual(req(self.port, fr + "?plugin_id=a&ticket=" + t)[0], 401)
        for bad in (7, "a" * 2000, "x=1#y"):
            self.assertEqual(req(self.port, "/api/auth/ticket", "POST", {"path": "/api/info", "query": bad}, headers=tok(self.app))[0], 400, bad)

    def test_tickets_keep_blank_query_values(self):
        # ctl-cookie-lows L1: a query with an empty value (?q=&x=1) mints and redeems as the same pairs
        mint = lambda q: json.loads(req(self.port, "/api/auth/ticket", "POST", {"path": "/api/info", "query": q},
                                        headers=tok(self.app))[2])["ticket"]
        self.assertEqual(req(self.port, "/api/info?q=&x=1&ticket=" + mint("q=&x=1"))[0], 200)
        self.assertEqual(req(self.port, "/api/info?x=1&ticket=" + mint("q=&x=1"))[0], 401)        # the blank pair is bound
        self.assertEqual(req(self.port, "/api/info?q=&x=1&ticket=" + mint("x=1"))[0], 401)        # and not added for free

    def test_login_answers_head(self):
        st, h, b = req(self.port, "/login", "HEAD")
        self.assertEqual(st, 200)
        self.assertTrue(h["Content-Type"].startswith("text/html"))
        self.assertEqual(b, b"")

    def test_no_ui_navigation_to_the_api_without_a_ticket(self):
        # location.href / location.assign / window.open of an /api/ address cannot carry the header: it must go
        # through PXA_AUTH.url (a ticket). The Vigil page and HUD are public shells that sign in from the tab.
        nav = re.compile(r"""(location\.href\s*=|location\.assign\(|location\.replace\(|window\.open\(|root\.open\()\s*[`"'](/api/[^`"'?]*)""")
        hits = []
        for d in (os.path.join(TOOLS, "pxa_control_ui"), os.path.join(TOOLS, "pxa_chat", "ui"), os.path.join(TOOLS, "anibal", "ui")):
            if not os.path.isdir(d):
                continue
            for n in sorted(os.listdir(d)):
                if n.endswith((".js", ".html")):
                    with open(os.path.join(d, n), encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            for _k, path in nav.findall(line):
                                if path not in ("/api/chat/vigil/page", "/api/chat/vigil/hud"):
                                    hits.append(f"{n}:{i}: {path}")
        self.assertEqual(hits, [])


class SignInEverywhereLan(SignInEverywhere):
    lan = True


# ---- H4 ---------------------------------------------------------------------------------------------------
class DrafterInExtraArgs(unittest.TestCase):
    def v(self, extra):
        return C.validate_launch({"gpus": [0], "model": os.path.join(MODELS, "tiny.gguf"), "port": 18080,
                                  "extra_args": extra}, [0, 1], set(), [MODELS])

    def test_drafter_flag_obeys_the_model_folder_rule(self):
        for bad in (["-md", os.path.join(OUTSIDE, "evil.gguf")], ["--model-draft=" + os.path.join(OUTSIDE, "evil.gguf")],
                    ["-md", "/etc/passwd"], ["-md", os.path.join(MODELS, "..", "outside", "evil.gguf")]):
            with self.assertRaises(C.Invalid, msg=bad):
                self.v(bad)
        ok = self.v(["-md", os.path.join(MODELS, "draft.gguf"), "-c", "4096"])
        self.assertIn(os.path.join(MODELS, "draft.gguf"), ok["extra_args"])

    def test_no_shell_metacharacters_or_unknown_flags(self):
        for bad in ("--log-file /tmp/x", "-c 4096; rm -rf /", "--alias $(id)", "--alias `id`", "--slot-save-path /",
                    ["--alias", "a b"], "--chat-template-file /etc/shadow"):
            with self.assertRaises(C.Invalid, msg=bad):
                C.validate_extra_args(bad)


# ---- H5 ---------------------------------------------------------------------------------------------------
class DockerNames(unittest.TestCase):
    def test_a_name_that_reads_as_a_flag_never_reaches_docker(self):
        app = C.App(L, port=7777)
        calls = []
        orig = C.subprocess.run
        C.subprocess.run = lambda *a, **k: calls.append(a) or orig(["true"])
        try:
            for name in ("--help", "-H", "a b", ""):
                x = {"key": "d:" + name, "kind": "docker", "adopted": True, "control": True, "container": name,
                     "name": name, "origin": "docker"}
                app._instance = lambda key, x=x: x
                with self.assertRaises(C.Invalid):
                    app.external_control("d:" + name, "stop", name)
                with self.assertRaises(C.Invalid):
                    app.external_log("d:" + name)
        finally:
            C.subprocess.run = orig
        self.assertEqual(calls, [])


# ---- H6 ---------------------------------------------------------------------------------------------------
class DownloadNames(unittest.TestCase):
    def test_names(self):
        good = ("m.gguf", "M-Q4.GGUF", "m-00001-of-00002.gguf", "m.gguf.expert-counts.csv")
        bad = ("../m.gguf", "a/m.gguf", "a\\m.gguf", "..", ".", ".m.gguf", "m.gguf\x00", "m.sh", "m.gguf.exe",
               "x" * 300 + ".gguf", "", None)
        for n in good:
            self.assertTrue(HF.safe_file_name(n), n)
        for n in bad:
            self.assertFalse(HF.safe_file_name(n), n)

    def test_a_bad_name_is_refused_before_any_byte_is_fetched(self):
        d = HF.Download("pxa/x", [{"path": "sub/../../evil.gguf", "name": "../evil.gguf", "size": 1}], MODELS)
        orig = HF._open
        HF._open = lambda *a, **k: self.fail("fetched a file with a path in its name")
        try:
            with self.assertRaises(HF.HFError):
                d._one(d.items[0])
        finally:
            HF._open = orig
        self.assertFalse(os.path.exists(os.path.join(TMP, "evil.gguf.part")))


# ---- H7 ---------------------------------------------------------------------------------------------------
class Limits(unittest.TestCase):
    def test_oversized_body_is_413(self):
        app = C.App(L, port=7777)
        srv, port = serve(app)
        try:
            # The server refuses on the Content-Length header, before it reads the body, and closes. Streaming the whole
            # oversized body from the client raced that close (BrokenPipe under a parallel ctest, pass 8e): claim the
            # oversized length but send a short body, so the write always completes, then read the 413.
            hdrs = {"Content-Type": "application/json", "Content-Length": str(C.MAX_BODY + 10), **tok(app)}
            with socket.create_connection(("127.0.0.1", port), timeout=20) as sk:
                sk.sendall(("POST /api/stop HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n" % port).encode() +
                           "".join("%s: %s\r\n" % kv for kv in hdrs.items()).encode() + b"\r\n" + b"{" + b" " * 1024)
                resp = b""
                while True:
                    chunk = sk.recv(65536)
                    if not chunk: break
                    resp += chunk
                    if b"\r\n\r\n" in resp:
                        head, _, rest = resp.partition(b"\r\n\r\n")
                        m = re.search(rb"(?i)content-length:\s*(\d+)", head)
                        if m and len(rest) >= int(m.group(1)): break
            head, _, b = resp.partition(b"\r\n\r\n")
            st = int(head.split(b" ", 2)[1])
            self.assertEqual((st, json.loads(b)["code"]), (413, "too_large"))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_wrong_token_flood_gets_429_and_the_owner_still_gets_in(self):
        app = C.App(L, port=7777)
        app.auth_throttle = C.AuthThrottle(limit=5, window=60)
        srv, port = serve(app)
        try:
            codes = [req(port, "/api/info", headers={"X-PXA-Token": "guess%d" % i})[0] for i in range(8)]
            self.assertEqual(codes, [401] * 5 + [429] * 3)
            st, h, _b = req(port, "/api/info", headers={"X-PXA-Token": "guess"})
            self.assertEqual((st, h.get("Retry-After")), (429, "60"))
            self.assertEqual(req(port, "/api/info", headers=tok(app))[0], 200)     # never a lock-out
        finally:
            srv.shutdown()
            srv.server_close()
        t = C.AuthThrottle(limit=2, window=10)
        t.fail("a", now=0)
        t.fail("a", now=1)
        self.assertTrue(t.blocked("a", now=2))
        self.assertFalse(t.blocked("a", now=12))
        self.assertFalse(t.blocked("b", now=2))

    def test_connection_cap_and_stalled_clients(self):
        app = C.App(L, port=7777)
        srv, port = serve(app, max_connections=2, timeout=1.0)
        held = []
        try:
            for _ in range(2):                                     # two clients that never send a request
                s = socket.create_connection(("127.0.0.1", port))
                held.append(s)
            time.sleep(0.2)
            s3 = socket.create_connection(("127.0.0.1", port))
            s3.settimeout(5)
            s3.sendall(b"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            self.assertIn(b" 503 ", s3.recv(4096))
            s3.close()
            t0 = time.time()                                        # the stalled ones are cut at the timeout
            for s in held:
                s.settimeout(10)
                self.assertEqual(s.recv(4096), b"")
            self.assertLess(time.time() - t0, 8)
            time.sleep(0.2)
            self.assertEqual(req(port, "/api/health")[0], 200)       # and their slots came back
        finally:
            for s in held:
                s.close()
            srv.RequestHandlerClass.timeout = C.REQUEST_TIMEOUT_S
            srv.shutdown()
            srv.server_close()


# ---- H8 ---------------------------------------------------------------------------------------------------
class Secrets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with C._cfg_lock:
            c = C.load_config()
            c["hf_token"] = HF_SECRET
            c["mqtt"] = {"enabled": False, "host": "broker.invalid", "user": "u", "password": MQTT_SECRET}
            c["encode"] = {"licence_key": LIC_SECRET}
            C.save_config(c)
        cls.app = C.App(L, port=7777, models_dirs=[MODELS])
        cls.app.auth_throttle = C.AuthThrottle(limit=10 ** 6)
        cls.srv, cls.port = serve(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_no_get_route_answers_a_secret(self):
        secrets_ = [s.encode() for s in (self.app.token, HF_SECRET, MQTT_SECRET, LIC_SECRET)]
        skip = {"/api/bench", "/api/models/available", "/api/update", "/api/lib/update", "/api/score/check",
                "/api/encode/runtime"}                       # these call out to the network; nothing secret in them
        n = 0
        for method, path in sorted(C.ROUTES):
            if method != "GET" or path in skip or path.startswith("/api/chat/events"):
                continue
            st, h, b = req(self.port, path, headers=tok(self.app))
            blob = b + json.dumps({k: v for k, v in h.items()}).encode()
            for s in secrets_:
                self.assertNotIn(s, blob, (path, s[:6]))
            n += 1
        self.assertGreater(n, 40)

    def test_access_log_masks_the_token(self):
        err = io.StringIO()
        os.environ["PXA_CONTROL_ACCESS_LOG"] = "1"
        try:
            with contextlib.redirect_stderr(err):
                req(self.port, "/?token=" + self.app.token)
                req(self.port, "/api/info?token=" + self.app.token)
                time.sleep(0.2)
        finally:
            os.environ.pop("PXA_CONTROL_ACCESS_LOG", None)
        self.assertIn("token=<redacted>", err.getvalue())
        self.assertNotIn(self.app.token, err.getvalue())

    def test_a_500_never_echoes_a_secret(self):
        def boom(app, body, query):
            raise RuntimeError(f"broke with {app.token} and hf_token={HF_SECRET}")
        C.ROUTES[("GET", "/api/test-boom")] = boom
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                st, _h, b = req(self.port, "/api/test-boom", headers=tok(self.app))
        finally:
            C.ROUTES.pop(("GET", "/api/test-boom"), None)
        self.assertEqual(st, 500)
        self.assertNotIn(self.app.token.encode(), b)
        self.assertNotIn(HF_SECRET.encode(), b)
        self.assertIn(b"RuntimeError", b)

    def test_a_background_banner_has_no_token(self):
        # The port comes from the kernel (bind to 0), never from a fixed range: copies of this suite run side by side (CI, a
        # box under load) and a shared start port made two of them listen for the same Control, so one read the OTHER copy's
        # answer and run record and then found nothing of its own in C.SERVING (KeyError: 17990, a flake found 2026-10-19).
        # Our own listener is waited for by name (C.SERVING[port]); a lost race for the port (rc 2) just takes another port.
        out = io.StringIO()
        for _attempt in range(5):
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()
            holder = {}
            with contextlib.redirect_stdout(out):
                th = threading.Thread(target=lambda: holder.update(rc=C.serve(L, port=port, open_browser=False,
                                                                              models_dirs=[MODELS])), daemon=True)
                th.start()
                for _ in range(200):
                    if port in C.SERVING or "rc" in holder:
                        break
                    time.sleep(0.05)
                if port not in C.SERVING:                            # lost the race for the port: try another
                    th.join(5)
                    out.truncate(0)
                    out.seek(0)
                    continue
                for _ in range(100):
                    if C.control_answers(port):
                        break
                    time.sleep(0.1)
                rec = C.find_running(port=port)
                C.SERVING[port].shutdown()
                th.join(10)
            break
        else:
            self.fail("no free port in 5 tries")
        self.assertTrue(rec and rec.get("token"))
        self.assertNotIn(rec["token"], out.getvalue())               # stdout is not a terminal here
        self.assertIn(f"PXA Control: http://127.0.0.1:{port}/", out.getvalue())
        self.assertIn("sign in: run `pxa` again", out.getvalue())

    def test_scrub(self):
        s = C.scrub_secrets("GET /?token=abcdef123456&x=1 tok=SEKRETSEKRET hf_token=" + HF_SECRET, ("SEKRETSEKRET",))
        self.assertNotIn("abcdef123456", s)
        self.assertNotIn("SEKRETSEKRET", s)
        self.assertNotIn(HF_SECRET, s)
        o = C.scrub_secrets('"GET /plugins/cohost/overlay/events?k=0123456789abcdef0123456789abcdef HTTP/1.1" 200 -')
        self.assertNotIn("0123456789abcdef0123456789abcdef", o)                     # a plugin overlay key never reaches the access log


# ---- H9 ---------------------------------------------------------------------------------------------------
class _HtmlEngine(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b"<!doctype html><script>fetch('/api/stop',{method:'POST'})</script>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


class Headers(Base):
    def test_every_answer_carries_the_headers(self):
        for path, h in (("/api/info", tok(self.app)), ("/api/health", {}), ("/nope", tok(self.app)), ("/chat.js", {})):
            _st, hd, _b = req(self.port, path, headers=h)
            self.assertEqual(hd.get("X-Frame-Options"), "DENY", path)
            self.assertEqual(hd.get("X-Content-Type-Options"), "nosniff", path)
            self.assertEqual(hd.get("Cross-Origin-Resource-Policy"), "same-origin", path)
            self.assertEqual(hd.get("Cross-Origin-Opener-Policy"), "same-origin", path)
            self.assertIn("camera=()", hd.get("Permissions-Policy", ""))
            self.assertEqual(hd.get("Referrer-Policy"), "no-referrer", path)

    def test_page_csp_hashes_the_inline_scripts_and_forbids_inline_eval(self):
        st, h, body = req(self.port, "/")
        self.assertEqual(st, 200)
        csp = h["Content-Security-Policy"]
        script = re.search(r"script-src ([^;]*)", csp).group(1)
        self.assertNotIn("unsafe-inline", script)
        self.assertNotIn("unsafe-eval", script)
        for d in ("object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'", "form-action 'self'"):
            self.assertIn(d, csp)
        blocks = re.findall(rb"<script\b([^>]*)>(.*?)</script\s*>", body, re.I | re.S)
        inline = [b for a, b in blocks if b"src" not in a]
        self.assertTrue(inline, "the page has its inline script")
        for b in inline:
            want = "'sha256-" + base64.b64encode(hashlib.sha256(b.replace(b"\r\n", b"\n")).digest()).decode() + "'"
            self.assertIn(want, script)

    def test_hash_vectors(self):
        h = C.inline_script_hashes(b"<p><script>alert(1)</script><script src=/x.js></script><SCRIPT type=module>\r\nx\r\n</SCRIPT>")
        self.assertEqual(h, ["'sha256-" + base64.b64encode(hashlib.sha256(b"alert(1)").digest()).decode() + "'",
                             "'sha256-" + base64.b64encode(hashlib.sha256(b"\nx\n").digest()).decode() + "'"])

    def test_engine_html_runs_sandboxed(self):
        eng = http.server.HTTPServer(("127.0.0.1", 0), _HtmlEngine)
        threading.Thread(target=eng.serve_forever, daemon=True).start()
        try:
            self.app.set_attach_port(eng.server_address[1])
            st, h, _b = req(self.port, "/api/engine/pxa/speed", headers=tok(self.app))
            self.assertEqual(st, 200)
            self.assertEqual(h["Content-Security-Policy"], C.SANDBOX_CSP)
            self.assertTrue(h["Content-Security-Policy"].startswith("sandbox allow-scripts;"))
        finally:
            self.app.set_attach_port(0)
            eng.shutdown()
            eng.server_close()

    def test_the_ui_keeps_the_csp_contract(self):
        """inline <script> blocks are hashed; inline handlers, eval and javascript: URLs would be blocked."""
        bad = re.compile(r"""(<[a-z][^>]*\son[a-z]+\s*=|\beval\s*\(|new\s+Function\s*\(|javascript:|setTimeout\(\s*["'`]|setInterval\(\s*["'`])""", re.I)
        hits = []
        for d in (os.path.join(TOOLS, "pxa_control_ui"), os.path.join(TOOLS, "pxa_chat", "ui")):
            for n in sorted(os.listdir(d)):
                if n.endswith((".html", ".js")):
                    with open(os.path.join(d, n), encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if bad.search(line):
                                hits.append(f"{n}:{i}: {line.strip()[:100]}")
        self.assertEqual(hits, [])


# ---- H10 --------------------------------------------------------------------------------------------------
class QueryToken(Base):
    def test_query_token_only_on_get(self):
        st, h, _b = req(self.port, "/api/info?token=" + self.app.token)        # a GET link: answered, no cookie
        self.assertEqual((st, h.get("Location"), h.get("Set-Cookie")), (200, None, None))
        self.assertEqual(req(self.port, "/api/stop?token=" + self.app.token, "POST", {})[0], 401)
        self.assertEqual(req(self.port, "/api/stop", "POST", {}, headers=tok(self.app))[0], 200)
        self.assertEqual(req(self.port, "/api/presets?token=" + self.app.token, "DELETE", {"name": "x"})[0], 401)


if __name__ == "__main__":
    unittest.main(verbosity=1)
