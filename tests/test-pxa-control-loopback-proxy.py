#!/usr/bin/env python3
"""L4: Control's 127.0.0.1 requests (the plugin host's model.chat to the local seat, and core's probes of the local server)
never go through http_proxy / the Windows system proxy: the prompt must not reach a proxy, and a proxy's error page
(internal hostnames, user names) must never come back to a plugin. A fake proxy and a fake seat on loopback; no GPU."""
import http.server
import json
import os
import socket
import sys
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pxa_control as C  # noqa: E402
from pxa_control.plugins import host as H  # noqa: E402
import urllib.request  # noqa: E402

PROXY_PAGE = b'{"error":"squid at gw-internal.corp.example, user alice@CORP, denied"}'


def serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Proxy(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _any(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        Proxy.seen.append((self.command, self.path))
        self.send_response(502)
        self.send_header("Content-Length", str(len(PROXY_PAGE)))
        self.end_headers()
        self.wfile.write(PROXY_PAGE)
    do_GET = do_POST = _any


class Seat(http.server.BaseHTTPRequestHandler):
    seen = []
    tokens = []

    def log_message(self, *a):
        pass

    def _any(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        Seat.seen.append((self.command, self.path))
        if self.headers.get("X-PXA-Token") is not None:
            Seat.tokens.append(self.headers.get("X-PXA-Token"))
        out = json.dumps({"ok": True, "path": self.path, "data": [], "status": "ok", "control_version": "test"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
    do_GET = do_POST = _any


class LoopbackNoProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy, cls.seat = serve(Proxy), serve(Seat)
        cls.pport, cls.sport = cls.proxy.server_address[1], cls.seat.server_address[1]
        url = "http://127.0.0.1:%d" % cls.pport
        cls.env = mock.patch.dict(os.environ, {"http_proxy": url, "HTTP_PROXY": url, "https_proxy": url, "HTTPS_PROXY": url,
                                               "no_proxy": "", "NO_PROXY": ""})
        cls.env.start()
        # the Windows system proxy is read through getproxies() (the registry); make that say the same
        cls.gp = mock.patch.object(urllib.request, "getproxies", lambda: {"http": url, "https": url})
        cls.gp.start()

    @classmethod
    def tearDownClass(cls):
        cls.gp.stop()
        cls.env.stop()
        for s in (cls.proxy, cls.seat):
            s.shutdown()
            s.server_close()

    def setUp(self):
        Proxy.seen.clear()
        Seat.seen.clear()
        Seat.tokens.clear()

    def test_the_harness_does_route_a_plain_urlopen_through_the_proxy(self):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/x" % self.sport, timeout=3)
        except OSError:
            pass
        self.assertTrue(Proxy.seen, "the fake proxy is not in the way, so the tests below prove nothing")

    def chat(self, port):
        seat = type("S", (), {"req": True, "running": lambda s: True, "port": lambda s: port})()
        return H._seat_chat(type("App", (), {"seats": {"a": seat}})())({"messages": [{"role": "user", "content": "secret prompt"}]})

    def test_model_chat_reaches_the_seat_not_the_proxy(self):
        r = self.chat(self.sport)
        self.assertEqual(r.get("path"), "/v1/chat/completions")
        self.assertEqual(Seat.seen, [("POST", "/v1/chat/completions")])
        self.assertEqual(Proxy.seen, [])

    def test_model_chat_down_never_shows_the_proxy_error(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        r = self.chat(dead)
        self.assertEqual(r["code"], "seat_down")
        self.assertEqual(Proxy.seen, [])
        text = json.dumps(r)
        for w in ("gw-internal", "alice", "squid", "denied"):
            self.assertNotIn(w, text)

    def test_core_probes_reach_the_server_not_the_proxy(self):
        self.assertEqual(C._http_json(self.sport, "/v1/models")["path"], "/v1/models")
        self.assertEqual(C.probe_health(self.sport)[0], "ok")
        self.assertEqual(Proxy.seen, [])
        self.assertEqual([p for _, p in Seat.seen], ["/v1/models", "/health"])

    def test_core_probe_of_a_dead_port_is_down_not_a_proxy_page(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        self.assertEqual(C.probe_health(dead, timeout=1)[0], "down")
        with self.assertRaises(OSError):
            C._http_json(dead, "/v1/models", timeout=1)
        self.assertEqual(Proxy.seen, [])

    def test_control_answers_sends_the_token_to_control_not_the_proxy(self):
        # security review 4.0 M3: the session token (X-PXA-Token) must never reach an http_proxy
        self.assertTrue(C.control_answers(self.sport, token="tok-not-for-the-proxy", timeout=3))
        self.assertEqual(Proxy.seen, [])
        self.assertEqual(Seat.seen, [("GET", "/api/health")])
        self.assertEqual(Seat.tokens, ["tok-not-for-the-proxy"])

    def test_control_answers_on_a_dead_port_is_false_and_never_proxied(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        self.assertFalse(C.control_answers(dead, token="tok-not-for-the-proxy", timeout=1))
        self.assertEqual(Proxy.seen, [])


if __name__ == "__main__":
    unittest.main()
