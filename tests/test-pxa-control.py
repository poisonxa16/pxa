#!/usr/bin/env python3
"""PXA Control (pxa-launch --gui, tools/pxa_control.py): HTTP handlers, token auth, lever validation,
preset persistence, the engine proxy and the seat's process handling. No GPU, no model, no network
beyond 127.0.0.1: cards come from PXA_LAUNCH_FAKE_GPUS and the "engine" is a stub HTTP server.

    python3 tests/test-pxa-control.py          (wired into CTest as test-pxa-control)
"""
import contextlib
import http.server
import io
import importlib.util
import json
import os
import shutil
import socket
import socketserver
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
TMP = tempfile.mkdtemp(prefix="pxa-control-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_MODELS_DIR", None)
os.environ["PXA_CONTROL_DISCOVER"] = "0"      # hermetic: never list the host's real servers
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN"):
    os.environ.pop(_k, None)                  # no browser opens, and the auto-start rule starts from its defaults
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402

MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)
FAKE_MODEL = os.path.join(MODELS, "tiny.gguf")
with open(FAKE_MODEL, "wb") as f:
    f.write(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)       # a header with 0 tensors, 0 KVs
CATALOG = {"PXA_ENHANCE", "PXA_CUDA_GRAPH_LRU", "PXQ_TEST"}


def serve_app(app):
    srv = C.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def req(port, path, method="GET", body=None, headers=None, host=None):
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h, method=method)
    if host:
        r.add_unredirected_header("Host", host)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    op = urllib.request.build_opener(NoRedirect)
    try:
        with op.open(r, timeout=20) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class LeverValidation(unittest.TestCase):
    def test_known_names_and_values(self):
        ok, errs = C.validate_levers({"PXA_ENHANCE": "1", "PXA_CUDA_GRAPH_LRU": 4, "PXQ_TEST": True}, CATALOG)
        self.assertEqual(errs, [])
        self.assertEqual(ok, {"PXA_ENHANCE": "1", "PXA_CUDA_GRAPH_LRU": "4", "PXQ_TEST": "1"})

    def test_unknown_and_foreign_names_refused(self):
        for name in ("LD_PRELOAD", "PATH", "PXA_NOT_IN_CATALOG", "pxa_enhance", "PXA_ENHANCE;rm"):
            _ok, errs = C.validate_levers({name: "1"}, CATALOG)
            self.assertTrue(errs, name)

    def test_bad_values_refused(self):
        for v in ("1; rm -rf /", "$(id)", "a b", "x\ny", "`id`", "a" * 201, "'q'", {"x": 1}):
            _ok, errs = C.validate_levers({"PXA_ENHANCE": v}, CATALOG)
            self.assertTrue(errs, repr(v))

    def test_empty_value_means_default(self):
        ok, errs = C.validate_levers({"PXA_ENHANCE": ""}, CATALOG)
        self.assertEqual((ok, errs), ({}, []))

    def test_shape(self):
        self.assertTrue(C.validate_levers(["PXA_ENHANCE"], CATALOG)[1])
        self.assertTrue(C.validate_levers({f"PXA_X{i}": "1" for i in range(65)}, CATALOG)[1])

    def test_catalog_parser(self):
        p = os.path.join(TMP, "cat.inc")
        with open(p, "w") as f:
            f.write('// comment\n{ "PXA_A", "on", "any", "default-on", "", "house lever" },\n'
                    '{ "PXA_B", "off (=1 on)", "any", "lever-off", "ev", "a \\"quoted\\" rule" },\n'
                    '{ "not_a_lever", "x", "y", "z", "", "" },\n')
        rows, src = C.load_catalog(p)
        self.assertEqual([r["name"] for r in rows], ["PXA_A", "PXA_B"])
        self.assertEqual(rows[1]["rule"], 'a "quoted" rule')
        self.assertEqual(src, os.path.abspath(p))

    def test_real_catalog_loads(self):
        rows, src = C.load_catalog()
        if src is None:
            self.skipTest("no catalog in this tree")
        self.assertGreater(len(rows), 50)
        self.assertIn("PXA_ENHANCE", {r["name"] for r in rows})


class BuiltinLevers(unittest.TestCase):
    """A release catalog drops the levers the closed library reads and keeps only their hashes: those names are
    accepted (no refusal, no lint warning), labelled built-in, and a typo is still refused."""
    def setUp(self):
        self.p = os.path.join(TMP, "cat-release.inc")
        h = C.lever_hash
        with open(self.p, "w") as f:
            f.write('{ "PXA_ENHANCE", "level 2", "any", "default-on", "", "x" },\n')
            f.write("// built-in: %s %s %s\n" % (h("PXA_HIDDEN_A"), h("PXA_LOCKY"), h("PXA_FAM*")))
            f.write("// built-in-preset: %s %s\n" % (h("PXA_HIDDEN_A"), h("PXA_FAM*")))
        rows, src = C.load_catalog(self.p)
        self.rows = rows
        self.names = C.LeverNames({r["name"] for r in rows}, *C.load_builtin(src))

    def test_accepted_and_labelled(self):
        ok, errs = C.validate_levers({"PXA_HIDDEN_A": "1", "PXA_FAM_X": "2", "PXA_LOCKY": "1", "PXA_ENHANCE": "1"}, self.names)
        self.assertEqual(errs, [])
        self.assertEqual(ok, {"PXA_HIDDEN_A": "1", "PXA_FAM_X": "2", "PXA_LOCKY": "1", "PXA_ENHANCE": "1"})
        self.assertEqual(self.names.builtin("PXA_HIDDEN_A"), "preset")
        self.assertEqual(self.names.builtin("PXA_FAM_X"), "preset")
        self.assertEqual(self.names.builtin("PXA_LOCKY"), "builtin")
        self.assertIsNone(self.names.builtin("PXA_ENHANCE"))
        self.assertEqual(C.lint_levers({"PXA_HIDDEN_A": "1"}, self.rows), [])

    def test_typo_still_refused(self):
        _ok, errs = C.validate_levers({"PXA_HIDDEN_B": "1"}, self.names)
        self.assertTrue(errs)
        _ok, errs = C.validate_levers({"LD_PRELOAD": "x"}, self.names)
        self.assertTrue(errs)

    def test_full_catalog_has_no_builtin(self):
        self.assertEqual(C.load_builtin(os.path.join(TMP, "nope.inc")), (set(), set()))

    def test_hash_matches_catalog_script(self):
        self.assertEqual(C.lever_hash("PXA_ENHANCE"), "%016x" % self._fnv(b"PXA_ENHANCE"))

    @staticmethod
    def _fnv(b):
        h = 0xcbf29ce484222325
        for x in b:
            h = ((h ^ x) * 0x100000001b3) % (1 << 64)
        return h


class LaunchValidation(unittest.TestCase):
    def v(self, **kw):
        body = {"gpus": [0], "model": FAKE_MODEL}
        body.update(kw)
        return C.validate_launch(body, {0, 1}, CATALOG, [MODELS], gui_port=7777)

    def test_defaults(self):
        r = self.v()
        self.assertEqual((r["gpus"], r["kv"], r["sm"], r["fa"], r["ctx"], r["np"]),
                         ([0], "auto", "auto", "auto", 0, 0))
        # port auto = the first FREE port from 8080 up (8080 itself is often taken)
        self.assertGreaterEqual(r["port"], 8080)
        self.assertFalse(C.port_in_use(r["port"]))

    def test_port_busy_and_content_length(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        try:
            self.assertTrue(C.port_in_use(busy))
            self.assertNotEqual(C.free_port(busy), busy)
        finally:
            s.close()
        self.assertEqual(C.content_length({}, 10), 0)
        self.assertEqual(C.content_length({"Content-Length": "7"}, 10), 7)
        for bad in ("-1", "abc", "11", " 1e3"):
            with self.assertRaises(C.Invalid):
                C.content_length({"Content-Length": bad}, 10)

    def test_refusals(self):
        bad = [dict(gpus=[]), dict(gpus=[7]), dict(gpus=["x"]), dict(model=""), dict(model="/etc/passwd"),
               dict(model=os.path.join(TMP, "nope.gguf")), dict(kv="q1_0"), dict(sm="graph"), dict(fa="maybe"),
               dict(port=80), dict(port=7777), dict(port="8080; id"), dict(ctx=-1), dict(ctx=1 << 22),
               dict(np=65), dict(np=True), dict(levers={"LD_PRELOAD": "x"})]
        for b in bad:
            with self.assertRaises(C.Invalid, msg=repr(b)):
                self.v(**b)

    def test_model_must_be_under_a_folder(self):
        other = os.path.join(TMP, "elsewhere.gguf")
        with open(other, "wb") as f:
            f.write(b"GGUF")
        with self.assertRaises(C.Invalid):
            self.v(model=other)
        # a link in the folder to a real GGUF elsewhere (an offloaded model) is allowed (2026-10-06);
        # a link out of the folder to anything that is not a GGUF is still refused (SymlinkedModels)
        link = os.path.join(MODELS, "link.gguf")
        if not os.path.lexists(link):
            os.symlink(other, link)
        self.assertEqual(self.v(model=link)["model"], link)
        notg = os.path.join(TMP, "elsewhere-text.gguf")
        with open(notg, "wb") as f:
            f.write(b"text")
        badlink = os.path.join(MODELS, "badlink.gguf")
        if not os.path.lexists(badlink):
            os.symlink(notg, badlink)
        with self.assertRaises(C.Invalid):
            self.v(model=badlink)

    def test_argv(self):
        r = self.v(gpus="1,0", ctx=16384, np=2, kv="q8_0", sm="tensor", fa="on", mtp=True, port=8123,
                   expose=True, accept_unmeasured=True)
        a = C.launcher_argv(r)
        self.assertEqual(a[:4], ["--gpus", "1,0", "--model", FAKE_MODEL])
        for want in (["--ctx", "16384"], ["--np", "2"], ["--ctk", "q8_0"], ["--ctv", "q8_0"], ["--sm", "tensor"],
                     ["--workload", "serve"], ["--spec", "mtp"], ["--host", "0.0.0.0"], ["--port", "8123"]):
            i = a.index(want[0])
            self.assertEqual(a[i:i + 2], want)
        self.assertIn("--accept-unmeasured", a)
        self.assertNotIn("--allow-busy", a)
        # kv auto (the default) passes no --ctk/--ctv: the engine registry picks the KV type
        a2 = C.launcher_argv(self.v())
        self.assertNotIn("--ctk", a2)
        self.assertNotIn("--ctv", a2)
        ns2 = L.build_parser().parse_args(a2)
        self.assertEqual((ns2.ctk, ns2.ctv), ("auto", "auto"))
        # the launcher's own parser accepts every argv the GUI builds
        ns = L.build_parser().parse_args(a)
        self.assertEqual((ns.gpus, ns.np, ns.workload, ns.spec), ("1,0", 2, "serve", "mtp"))
        self.assertEqual(C.launcher_argv(self.v(fa="off"))[C.launcher_argv(self.v(fa="off")).index("--workload") + 1],
                         "longdoc")
        self.assertNotIn("--workload", C.launcher_argv(self.v()))

    def test_cli_line_quotes(self):
        r = self.v(levers={"PXA_ENHANCE": "0"})
        line = C.cli_line(r)
        self.assertTrue(line.startswith("PXA_ENHANCE=0 pxa-launch --gpus 0"))


class Presets(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        self.app.catalog_names = CATALOG

    def test_roundtrip_and_persistence(self):
        self.app.save_preset("1080ti chat", {"gpus": [1], "model": FAKE_MODEL, "kv": "q8_0",
                                             "levers": {"PXA_ENHANCE": "1"}})
        again = C.App(L, port=7777)
        p = again.presets()["1080ti chat"]
        self.assertEqual((p["gpus"], p["kv"], p["levers"]), ([1], "q8_0", {"PXA_ENHANCE": "1"}))
        mode = os.stat(C.config_path()).st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.assertTrue(again.delete_preset("1080ti chat"))
        self.assertNotIn("1080ti chat", C.App(L, port=7777).presets())

    def test_bad_names_and_bodies(self):
        for n in ("", "../x", "a" * 49, "semi;colon", None):
            with self.assertRaises(C.Invalid):
                self.app.save_preset(n, {"gpus": [0], "model": FAKE_MODEL})
        with self.assertRaises(C.Invalid):
            self.app.save_preset("ok", {"gpus": [0], "model": FAKE_MODEL, "levers": {"PATH": "/tmp"}})

    def test_model_dirs(self):
        self.assertEqual(self.app.set_model_dirs([MODELS]), [MODELS])
        self.assertEqual(C.load_config()["model_dirs"], [MODELS])
        with self.assertRaises(C.Invalid):
            self.app.set_model_dirs(["/definitely/not/here"])
        with self.assertRaises(C.Invalid):
            self.app.set_model_dirs("notalist")


class Handlers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        cls.app = C.App(L, port=7777, models_dirs=[MODELS])
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_static_and_info(self):
        code, h, body = req(self.port, "/")
        self.assertEqual(code, 200)
        self.assertIn(b"PXA Control", body)
        self.assertIn("frame-ancestors 'none'", h.get("Content-Security-Policy", ""))
        code, _, body = req(self.port, "/mark.png")
        self.assertEqual((code, body[:4]), (200, b"\x89PNG"))
        code, _, body = req(self.port, "/api/info")
        d = json.loads(body)
        self.assertEqual((code, d["lan"], d["kv_types"][0]), (200, False, "auto"))

    def test_status_rig_presets(self):
        self.assertEqual(json.loads(req(self.port, "/api/status")[2])["running"], False)
        live = json.loads(req(self.port, "/api/rig/live")[2])
        self.assertEqual([c["index"] for c in live["cards"]], [0, 1])
        self.assertEqual(live["cards"][0]["sm"], 60)
        code, _, body = req(self.port, "/api/presets", "POST",
                            {"name": "p1", "settings": {"gpus": [0, 1], "model": FAKE_MODEL, "sm": "tensor"}})
        self.assertEqual(code, 200, body)
        self.assertEqual(json.loads(req(self.port, "/api/presets")[2])["p1"]["sm"], "tensor")
        code, _, body = req(self.port, "/api/presets", "DELETE", {"name": "p1"})
        self.assertEqual((code, json.loads(body)["deleted"]), (200, True))

    def test_validation_is_400(self):
        code, _, body = req(self.port, "/api/plan", "POST", {"gpus": [0], "model": FAKE_MODEL,
                                                             "levers": {"LD_PRELOAD": "/x.so"}})
        self.assertEqual(code, 400)
        self.assertIn("LD_PRELOAD", json.loads(body)["error"])
        code, _, _ = req(self.port, "/api/start", "POST", {"gpus": [9], "model": FAKE_MODEL})
        self.assertEqual(code, 400)
        code, _, _ = req(self.port, "/api/plan", "POST", None, headers={"Content-Length": "0"})
        self.assertEqual(code, 400)

    def test_guards(self):
        self.assertEqual(req(self.port, "/api/info", host="evil.example:7777")[0], 421)
        self.assertEqual(req(self.port, "/api/info", host=f"localhost:{self.port}")[0], 200)
        code, _, _ = req(self.port, "/api/stop", "POST", {}, headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, _, _ = req(self.port, "/api/stop", "POST", {}, headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(code, 200)
        self.assertEqual(req(self.port, "/api/nope")[0], 404)
        self.assertEqual(req(self.port, "/api/engine/slots/save", "POST", {})[0], 404)   # not proxied
        self.assertEqual(req(self.port, "/api/engine/../../etc/passwd")[0], 404)

    def test_engine_proxy_without_server(self):
        C.save_config(dict(C.load_config(), attach_port=0))
        self.assertEqual(req(self.port, "/api/engine/health")[0], 503)
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": 7777})[0], 400)   # its own port
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": 70000})[0], 400)
        self.assertEqual(req(self.port, "/api/bench", "POST", {})[0], 400)                 # no server


class StubEngine(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            b = b'{"status":"ok"}'
            self.send_response(200)
        elif self.path.startswith("/pxa/stats"):
            b = json.dumps({"count": 1, "records": [{"ts": time.time(), "model": "m", "decode_tps": 50.0,
                                                     "prefill_tps": 900.0, "n_prompt": 100}]}).encode()
            self.send_response(200)
        else:
            b = b'{"error":"nf"}'
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in ("hel", "lo"):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": w}}]}).encode() + b"\n\n")
            self.wfile.flush()
        last = {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "stub-model",
                "timings": {"prompt_n": 7, "predicted_n": 2, "prompt_per_second": 321.0,
                            "predicted_per_second": 42.5}, "echo_stream": body.get("stream")}
        self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")


class EngineProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        cls.stub = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StubEngine)
        cls.stub.daemon_threads = True
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.app = C.App(L, port=7777)
        cls.srv, cls.port = serve_app(cls.app)
        code, _, body = req(cls.port, "/api/attach", "POST", {"port": cls.stub.server_address[1]})
        assert code == 200, body

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.stub.shutdown()

    def test_health_and_stats(self):
        st = json.loads(req(self.port, "/api/status")[2])
        self.assertEqual((st["health"], st["attached"]), ("ok", True))
        code, _, body = req(self.port, "/api/engine/pxa/stats?since=0")
        self.assertEqual((code, json.loads(body)["records"][0]["decode_tps"]), (200, 50.0))
        self.assertEqual(req(self.port, "/api/engine/props")[0], 404)     # passed through from the stub

    def test_stream_and_history(self):
        code, h, body = req(self.port, "/api/engine/v1/chat/completions", "POST",
                            {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("event-stream", h.get("Content-Type", ""))
        text = body.decode()
        self.assertIn('"hel"', text)
        self.assertIn("[DONE]", text)
        hist = json.loads(req(self.port, "/api/history?since=0")[2])["records"]
        self.assertTrue(any(r.get("decode_tps") == 42.5 and r.get("model") == "stub-model" for r in hist), hist)


class AttachedSeatInFleet(unittest.TestCase):
    """d7121574b5 (TucsonJohn, #general 2026-10-05): a seat ATTACHED to a server the launcher started (attach_port) serves, so the fleet must
    carry attached + its port and the Speed/Chat/Report pickers must not call it '(stopped)'. The fix shipped without a test."""

    @classmethod
    def setUpClass(cls):
        cls.stub = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StubEngine)
        cls.stub.daemon_threads = True
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.app = C.App(L, port=7777)
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        req(cls.port, "/api/attach", "POST", {"port": 0})        # leave no attach_port behind for later classes
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.stub.shutdown()

    def main_row(self):
        code, _, body = req(self.port, "/api/fleet?force=1")
        self.assertEqual(code, 200, body)
        rows = [x for x in json.loads(body)["instances"] if x.get("key") == "m:main"]
        self.assertEqual(len(rows), 1, body)
        return rows[0]

    def test_attached_seat_is_attached_with_the_launcher_port(self):
        sport = self.stub.server_address[1]
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": sport})[0], 200)
        row = self.main_row()
        self.assertFalse(row["running"])                  # Control did not start it ...
        self.assertTrue(row["attached"])                  # ... but it is attached, so not "stopped"
        self.assertEqual(row["port"], sport)
        st = json.loads(req(self.port, "/api/status")[2])
        self.assertEqual((st["attached"], st["engine_port"]), (True, sport))   # the header and the pickers agree

    def test_detached_seat_is_not_attached(self):
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": 0})[0], 200)
        row = self.main_row()
        self.assertFalse(row["running"])
        self.assertFalse(row["attached"])

    def test_the_pickers_label_an_attached_seat_attached_not_stopped(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertIn('x.attached ? " (attached)" : " (stopped)"', html)               # Speed / Report picker
        self.assertIn('x.attached ? " :" + x.port + " (attached)" : " (stopped)"', html)  # Chat targets
        self.assertIn("x.running || x.attached", html)                                    # an attached seat is picked as live


class TokenAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=7777, lan=True, token="s3cret-token")
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_random_token_when_lan(self):
        a = C.App(L, port=7777, lan=True)
        self.assertGreaterEqual(len(a.token), 20)
        self.assertIsNone(C.App(L, port=7777).token)

    def test_refused_without_token(self):
        self.assertEqual(req(self.port, "/api/info")[0], 401)
        code, _, body = req(self.port, "/")
        self.assertEqual(code, 401)
        self.assertIn(b"token", body)
        self.assertEqual(req(self.port, "/api/info", headers={"X-PXA-Token": "wrong"})[0], 401)
        self.assertEqual(req(self.port, "/api/info", headers={"Cookie": "pxa_control_token=nope"})[0], 401)

    def test_header_query_cookie(self):
        self.assertEqual(req(self.port, "/api/info", headers={"X-PXA-Token": "s3cret-token"})[0], 200)
        code, h, _ = req(self.port, "/?token=s3cret-token&x=1")
        self.assertEqual(code, 303)
        self.assertEqual(h.get("Location"), "/?x=1")
        self.assertIn("pxa_control_token=s3cret-token", h.get("Set-Cookie", ""))
        self.assertIn("HttpOnly", h.get("Set-Cookie", ""))
        self.assertEqual(req(self.port, "/api/info", headers={"Cookie": "a=b; pxa_control_token=s3cret-token"})[0], 200)
        # any Host is fine on a LAN bind (the token is the guard), cross-origin POST still refused
        self.assertEqual(req(self.port, "/api/info", host="192.168.1.5:7777",
                             headers={"X-PXA-Token": "s3cret-token"})[0], 200)
        self.assertEqual(req(self.port, "/api/stop", "POST", {}, headers={"X-PXA-Token": "s3cret-token",
                                                                         "Origin": "http://evil.example"})[0], 403)


class SeatProcess(unittest.TestCase):
    def built(self, code):
        plan = L.Plan()
        return (plan, [sys.executable, "-u", "-c", code], {"PXA_X": "1"}, "0", {}, 4096)

    def test_log_phase_exit(self):
        s = C.Seat(L)
        s.start(self.built("import os;print('llama_model_loader: x');print('server is listening');"
                           "print('env', os.environ.get('PXA_X'), os.environ.get('CUDA_VISIBLE_DEVICES'), "
                           "os.environ.get('PXA_ENHANCE'))"),
                {"port": 8999, "model": "m", "gpus": [0]}, {"PXA_ENHANCE": "0"})
        s.proc.wait(timeout=20)
        time.sleep(0.3)
        lines = [x[1] for x in s.lines_since(0)]
        self.assertTrue(lines[0].startswith("$ PXA_ENHANCE=0 "), lines[0])
        self.assertIn("env 1 0 0", lines)
        self.assertTrue(any("exited with code 0" in x for x in lines))
        self.assertEqual(s.status()["running"], False)
        self.assertEqual(s.lines_since(len(lines)), [])

    def test_stop_and_single_seat(self):
        s = C.Seat(L)
        s.start(self.built("import time\nprint('up', flush=True)\ntime.sleep(60)"),
                {"port": 8999, "model": "m", "gpus": [0]}, {})
        self.assertTrue(s.running())
        with self.assertRaises(C.Invalid):
            s.start(self.built("pass"), {"port": 8999, "model": "m", "gpus": [0]}, {})
        pid = s.proc.pid
        self.assertTrue(s.stop())
        self.assertFalse(s.running())
        self.assertFalse(os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z")
        self.assertFalse(s.stop())


class Redaction(unittest.TestCase):
    """the bug-report bundle leaves the machine only after redact_obj()."""
    H, U = {"rigbox"}, {"alice"}

    def r(self, s):
        return C.redact_text(s, self.H, self.U)

    def test_home_paths(self):
        home = os.path.expanduser("~")
        self.assertEqual(self.r(home + "/models/x.gguf"), "~/models/x.gguf")
        self.assertEqual(self.r("/home/bob/pxa/x"), "~/pxa/x")
        self.assertEqual(self.r("/Users/carol/Library/x"), "~/Library/x")
        self.assertEqual(self.r("C:\\Users\\dave\\m.gguf"), "~\\m.gguf")
        self.assertEqual(self.r("/root/m.gguf"), "~/m.gguf")

    def test_ips_hosts_users_mail(self):
        out = self.r("server 192.168.1.50:8080 from 10.0.0.7 and 127.0.0.1, host rigbox user alice mail a.b@example.com")
        for bad in ("192.168.1.50", "10.0.0.7", "rigbox", "alice", "example.com"):
            self.assertNotIn(bad, out)
        self.assertIn("127.0.0.1", out)
        self.assertIn("<ip>:8080", out)
        self.assertNotIn("fe80", self.r("addr fe80::1c2b:3aff:fe4d:5e6f end"))
        self.assertNotIn("aa:bb:cc:dd:ee:ff", self.r("mac aa:bb:cc:dd:ee:ff"))

    def test_secrets(self):
        cases = ["--api-key sk-abcdefghijklmnop1234", "--api-key=hunter2hunter2", "Authorization: Bearer abcdef0123456789xyz",
                 "HF_TOKEN=hf_" + "a" * 30, "token: abcdEFGH12345678", "password = 'p4ssw0rd!'",
                 "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc123def456",
                 "ghp_" + "Z" * 36, "MTIzNDU2Nzg5MDEyMzQ1Njc4" + "." + "GabcdE" + "." + "abcdefghijklmnopqrstuvwxyz0123",
                 "key AbCdEf0123456789AbCdEf0123456789xyz"]
        for c in cases:
            out = self.r(c)
            self.assertIn("<redacted", out, c)
            for frag in ("hunter2", "sk-abc", "abcdef0123456789xyz", "hf_aaaa", "abcdEFGH", "p4ssw0rd", "eyJhbGci", "ghp_ZZ",
                         "GabcdE", "AbCdEf0123456789AbCdEf"):
                self.assertNotIn(frag, out, c)

    def test_keeps_useful_diagnostics(self):
        for keep in ("PXA_TSPLIT: mode=tensor devs=2", "prompt eval time = 812.4 ms / 2048 tokens", "n_tokens = 512",
                     "llama_model_loader: loaded meta data with 41 key-value pairs", "CUDA error: out of memory",
                     "greedy512 sha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "Tesla V100-SXM2-16GB cc 70",
                     "driver 535.104.05 CUDA 12.2"):
            self.assertEqual(self.r(keep), keep)

    def test_idempotent_and_recursive(self):
        o = {"a": ["/home/bob/x", {"k": "10.1.2.3"}], "n": 5, "log": ["--api-key abcdefgh12345"]}
        once = C.redact_obj(o, self.H, self.U)
        self.assertEqual(once, C.redact_obj(once, self.H, self.U))
        self.assertNotIn("bob", json.dumps(once))
        self.assertNotIn("10.1.2.3", json.dumps(once))
        self.assertEqual(once["n"], 5)


class ReportBundle(unittest.TestCase):
    def setUp(self):
        self.app = C.App(L, port=0, models_dirs=[MODELS])

    def test_gpu_telemetry_in_bundle(self):
        line = ("0, 83, 544, 1328, 715, 715, 121.50, 250.00, 0x0000000000000068\n"
                "1, 41, 1328, 1328, 715, 715, 30.00, 250.00, 0x1\n")
        real = self.app.L._run
        self.app.L._run = lambda argv, timeout=10: line if argv and argv[0] == "nvidia-smi" else real(argv, timeout=timeout)
        fake = os.environ.pop("PXA_LAUNCH_FAKE_GPUS")
        try:
            t = self.app.gpu_telemetry()
        finally:
            os.environ["PXA_LAUNCH_FAKE_GPUS"] = fake
            self.app.L._run = real
        self.assertEqual([r["index"] for r in t], [0, 1])
        self.assertEqual((t[0]["temp_c"], t[0]["sm_clock_mhz"], t[0]["power_limit_w"]), (83.0, 544.0, 250.0))
        self.assertEqual(sorted(t[0]["throttle_reasons"]), ["hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown"])
        self.assertEqual(t[1]["throttle_reasons"], [])   # idle only is not a throttle
        # the fake rig asks no nvidia-smi
        self.assertEqual(self.app.gpu_telemetry(), [])

    def test_heat_warning(self):
        w = C.App.heat_warning(2, 84.0, ["hw_thermal_slowdown"])
        self.assertEqual(w, "Card 2 is running hot (84 \u00b0C, thermal throttling): expect much lower speed. "
                            "Passive Tesla cards need forced airflow.")
        self.assertIn("80 \u00b0C", C.App.heat_warning(0, 80.0, []))
        self.assertIn("power throttling", C.App.heat_warning(1, None, ["sw_power_cap"]))
        self.assertIsNone(C.App.heat_warning(0, 79.0, []))
        self.assertIsNone(C.App.heat_warning(0, 40.0, ["gpu_idle"]))
        self.assertIn("heat_warning", self.app.rig_live()["cards"][0])

    def test_bundle_shape_and_send_failure_is_reported(self):
        self.app.seat._append("PXA_REGISTRY: card 0 sm 60 class=p100")
        self.app.seat._append("build: 4711 (abc1234)")
        self.app.seat._append("using key sk-abcdefghijklmnop1234 at /home/bob/x")
        b = self.app.report_bundle()
        self.assertEqual((b["app"], b["version"]), ("pxa-control", C.CONTROL_VERSION))
        bu = b["bundle"]
        self.assertTrue(any("PXA_REGISTRY" in x for x in bu["banner"]))
        self.assertEqual(bu["engine"]["version"], "4711 (abc1234)")
        self.assertEqual(len(bu["gpus"]), 2)
        self.assertIn("gpu_telemetry", bu)
        self.assertNotIn("sk-abc", json.dumps(b))
        self.assertNotIn("/home/bob", json.dumps(b))
        old = os.environ.get("PXA_BUG_URL")
        os.environ["PXA_BUG_URL"] = "http://127.0.0.1:1/v1/report"     # nothing listens
        try:
            r = self.app.report_send({"payload": b})
        finally:
            os.environ.pop("PXA_BUG_URL") if old is None else os.environ.__setitem__("PXA_BUG_URL", old)
        self.assertFalse(r["ok"])
        self.assertIn("could not reach", r["error"])
        with self.assertRaises(C.Invalid):
            self.app.report_send({"payload": "{not json"})
        with self.assertRaises(C.Invalid):
            self.app.report_send({"payload": {"x": "y" * (C.REPORT_MAX + 10)}})


    def test_engine_version_survives_the_log_ring(self):
        seat = self.app.seat
        seat._append("build: 4711 (abc1234)")
        for i in range(seat.log.maxlen + 50):          # a chatty server pushes the build line out of the ring
            seat._append(f"slot update_slots: id 0 | task {i} | n_past = {i}")
        self.assertFalse(any("build:" in x[1] for x in seat.log))
        self.assertEqual(self.app.report_bundle()["bundle"]["engine"]["version"], "4711 (abc1234)")
        seat.log.clear()
        seat.build_lines = []

    def test_engine_version_from_the_binary_when_the_log_has_none(self):
        d = tempfile.mkdtemp(prefix="pxa-engver-")
        exe = os.path.join(d, "bin", "llama-server")
        os.makedirs(os.path.dirname(exe))
        with open(exe, "w") as f:
            f.write("#!/bin/sh\necho 'version: 9999 (deadbee)' >&2\necho 'built with cc for x86_64' >&2\n")
        os.chmod(exe, 0o755)
        seat = self.app.seat
        old = (seat.cmd, seat.req, list(seat.log), list(seat.build_lines))
        try:
            seat.cmd, seat.req = [exe, "-m", "x.gguf"], None
            seat.log.clear()
            seat.build_lines = []
            v = self.app.report_bundle()["bundle"]["engine"]["version"]
            self.assertEqual(v, "9999 (deadbee) (from --version)")
            self.assertIn("9999 (deadbee)", self.app._engine_ver_cache.values())     # cached per binary + mtime
        finally:
            seat.cmd, seat.req = old[0], old[1]
            seat.log.clear()
            seat.log.extend(old[2])
            seat.build_lines = old[3]
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(C.App.parse_engine_version("main: x\nbuild: 7046 (1a2b3c4) with gcc"), "7046 (1a2b3c4) with gcc")
        self.assertIsNone(C.App.parse_engine_version("no version here"))


class ScoreBoardClient(unittest.TestCase):
    def test_name_filter_matches_board(self):
        for bad in ("http://x.example", "see example.com", "@everyone", "f u c k", "sh1t", "<b>", ""):
            self.assertEqual(C.clean_board_name(bad), "", bad)
        self.assertEqual(C.clean_board_name("  Pat O'Neil "), "Pat O'Neil")
        self.assertEqual(len(C.clean_board_name("y" * 90)), 32)

    def test_payload_shape_and_redaction(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "-sm", "tensor", "--api-key", "sk-abcdefghijklmnop1234"]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0, 1], "port": 8999}
        res = {"ts": 1.0, "model": FAKE_MODEL, "cards": [0, 1], "reps": 3, "greedy512_sha": "ab" * 32,
               "classes": [{"class": "prose", "decode_tps": 41.5, "prefill_tps": 0, "n_prompt": 20, "decode_all": [41, 42], "prefill_all": []},
                           {"class": "long", "decode_tps": 30.0, "prefill_tps": 900.0, "n_prompt": 8000, "decode_all": [], "prefill_all": [900]}]}
        p = app.score_payload(res, "Zed")
        self.assertEqual((p["bracket"]["mode"], p["bracket"]["gpu_count"], p["name"]), ("tensor", 2, "Zed"))
        self.assertEqual((p["metrics"]["decode_tps"], p["metrics"]["prefill_tps"]), (41.5, 900.0))
        self.assertNotIn("sk-abc", json.dumps(p))
        self.assertRegex(p["bracket"]["model_sha8"], r"^[0-9a-f]{8}$")
        self.assertEqual(app.score_check()["available"], False)          # no benchmark run on this App yet

    def test_params_block_has_the_facts_and_no_secrets(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "--host", "192.168.7.9", "-sm", "tensor", "-c", "16384", "-ctk", "q4_0", "-ctv", "q4_0",
                        "-b", "2048", "-ub", "512", "-fa", "on", "--spec-type", "mtp:n_max=3", "--api-key", "sk-abcdefghijklmnop1234"]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0, 1], "port": 8999}
        res = {"ts": 1_780_000_000.0, "model": FAKE_MODEL, "cards": [0, 1], "reps": 3, "greedy512_sha": "ab" * 32,
               "classes": [{"class": "prose", "decode_tps": 41.5, "prefill_tps": 0, "n_prompt": 20, "decode_all": [41], "prefill_all": []}]}
        p = app.score_payload(res, "Zed")
        q = p["params"]
        self.assertEqual((q["card_count"], q["split_mode"], q["ctx"], q["kv_k"], q["batch"], q["ubatch"], q["flash_attn"], q["n_max"], q["mtp"]),
                         (2, "tensor", 16384, "q4_0", 2048, 512, "on", 3, True))
        self.assertEqual((q["reps"], q["greedy512_sha"], q["prompt_class"], q["model_file"]), (3, "ab" * 32, "prose", os.path.basename(FAKE_MODEL)))
        self.assertTrue(q["cards"] and q["cards"][0]["vram_mb"] > 0)
        blob = json.dumps(p)
        for bad in ("sk-abc", "192.168", os.path.dirname(FAKE_MODEL)):
            self.assertNotIn(bad, blob)
        self.assertNotIn("--host", p["cmd"])

    def test_name_is_saved_and_reused(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        app = C.App(L, port=0, models_dirs=[MODELS])
        self.assertEqual(app.user_name()["name"], "")
        self.assertEqual(app.set_user_name({"name": "  Pat  "})["name"], "Pat")
        self.assertEqual(C.App(L, port=0).user_name()["name"], "Pat")         # survives a restart (control.json)
        with self.assertRaises(C.Invalid):
            app.set_user_name({"name": "http://evil.example"})
        self.assertEqual(app.user_name()["name"], "Pat")                      # a refused name changes nothing
        self.assertEqual(app.set_user_name({"name": ""})["name"], "")         # forget
        # a submit saves the name too (first entry)
        real = C.urllib.request.urlopen
        C.urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        try:
            app.score_send({"payload": {"name": "Zed", "bracket": {}}})
        finally:
            C.urllib.request.urlopen = real
        self.assertEqual(app.user_name()["name"], "Zed")
        self.assertEqual(json.load(open(C.config_path()))["user_name"], "Zed")

    def test_detail_validates_id(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        for bad in ("x", "0", "-4", "99999999999", None):
            with self.assertRaises(C.Invalid):
                app.score_detail(bad)


# ---------------------------------------------------------------------------------------------
# 2026.10.2: several servers per GUI, the fleet, extra args, lever hints, telemetry, bounded report
# ---------------------------------------------------------------------------------------------
def fake_running_seat(app, sid, port, gpus):
    """a seat whose process is a real sleeping child (running() is true) holding port/gpus."""
    import subprocess
    seat = app.get_seat(sid, create=True)
    seat.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    seat.req = {"port": port, "gpus": list(gpus), "model": FAKE_MODEL}
    return seat


class Servers(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        self.app.catalog_names = CATALOG
        self.kids = []

    def tearDown(self):
        for s in self.app.seats.values():
            if s.proc and s.proc.poll() is None:
                s.proc.kill()
                s.proc.wait()

    def test_profiles_crud_and_persistence(self):
        self.assertEqual(list(self.app.profiles()), ["main"])
        sid, ent = self.app.save_profile(None, "V100 box", {"gpus": [1], "model": FAKE_MODEL, "port": 0})
        self.assertEqual(sid, "v100-box")
        sid2, _ = self.app.save_profile(None, "V100 box", {})
        self.assertEqual(sid2, "v100-box-2")                       # names may repeat, ids may not
        again = C.App(L, port=7777)
        self.assertEqual(again.profiles()["v100-box"]["settings"]["gpus"], [1])
        self.assertEqual(again.profiles()["v100-box"]["settings"]["port"], 0)   # auto stays auto
        with self.assertRaises(C.Invalid):
            self.app.delete_profile("main")
        for bad in ("", "x" * 49, "semi;colon", None):
            with self.assertRaises(C.Invalid):
                self.app.save_profile(None, bad, {})
        with self.assertRaises(C.Invalid):
            self.app.get_seat("../etc")
        with self.assertRaises(C.Invalid):
            self.app.get_seat("nope")                                 # unknown and not created
        self.assertTrue(self.app.delete_profile("v100-box-2"))
        self.assertNotIn("v100-box-2", C.App(L, port=7777).profiles())

    def test_delete_running_refused(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        with self.assertRaises(C.Invalid):
            self.app.delete_profile("b")

    def test_port_and_card_conflicts(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        req = self.app.validate({"gpus": [1], "model": FAKE_MODEL, "port": 18555}, resolve_port=False)
        c = self.app.conflicts(req, "main")
        self.assertTrue(any("port 18555" in e and "'B'" in e for e in c["errors"]), c)
        self.assertTrue(any(e.startswith("card 1 is in use by server 'B'") for e in c["errors"]), c)
        req = self.app.validate({"gpus": [1], "model": FAKE_MODEL, "port": 18556, "allow_busy": True}, resolve_port=False)
        c = self.app.conflicts(req, "main")
        self.assertEqual(c["errors"], [])
        self.assertTrue(any("allowed" in w for w in c["warnings"]))
        req = self.app.validate({"gpus": [0], "model": FAKE_MODEL, "port": 18556}, resolve_port=False)
        self.assertEqual(self.app.conflicts(req, "main"), {"errors": [], "warnings": []})
        # a server's own port/cards are not a conflict with itself
        req = self.app.validate({"gpus": [1], "model": FAKE_MODEL, "port": 18555}, resolve_port=False)
        self.assertEqual(self.app.conflicts(req, "b")["errors"], [])
        # auto port skips the ports other servers hold
        self.assertNotIn(self.app.validate({"gpus": [0], "model": FAKE_MODEL})["port"], self.app.held_ports())

    def test_stop_then_start_on_the_same_port(self):
        # regression (TucsonJohn, 2026-10-07): the engine listens with SO_REUSEPORT only (cpp-httplib), so
        # the TIME_WAIT sockets its health polls leave on its port refused Control's SO_REUSEADDR probe for
        # up to a minute after Stop: "port N is already in use by another program" until the port changed
        engine = ("import socket, sys\n"
                  "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)\n"
                  "s.bind(('127.0.0.1', 0)); s.listen(8); print(s.getsockname()[1], flush=True)\n"
                  "while True:\n"
                  "    c, _ = s.accept(); c.recv(1024); c.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: 0\\r\\n\\r\\n'); c.close()\n")
        seat = self.app.get_seat("main")
        seat.start((L.Plan(), [sys.executable, "-u", "-c", engine], {}, "0", {}, 4096),
                   {"port": 0, "model": FAKE_MODEL, "gpus": [0]}, {})
        t0 = time.time()
        while not any(x[1].strip().isdigit() for x in seat.lines_since(0)) and time.time() - t0 < 20:
            time.sleep(0.05)
        port = int(next(x[1] for x in seat.lines_since(0) if x[1].strip().isdigit()))
        seat.req["port"] = port
        for _ in range(3):                      # health polls: the server closes first -> TIME_WAIT on its port
            with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
                c.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                c.recv(1024)
                time.sleep(0.05)
        self.assertTrue(C.port_in_use(port))    # a live listener is still refused
        self.assertTrue(self.app.stop("main"))
        self.assertFalse(C.port_in_use(port))
        req = self.app.validate({"gpus": [0], "model": FAKE_MODEL, "port": port}, resolve_port=False)
        self.assertEqual(self.app.conflicts(req, "main")["errors"], [])
        s2 = socket.socket()                    # and the next engine really binds there
        s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            s2.bind(("127.0.0.1", port))
            s2.listen(1)
        finally:
            s2.close()

    def test_first_plan_sees_discovered_servers(self):
        # regression: right after start the fleet cache was empty and the first plan saw no clash
        snap = {"instances": [{"key": "d:seat", "kind": "docker", "name": "seat", "container": "seat",
                               "running": True, "port": 18700, "gpus": [0], "vram": {}}], "cards": []}
        calls = []

        def fake_fleet(max_age=2.5):
            calls.append(max_age)
            self.app._fleet_cache = (time.time(), snap)
            return snap
        self.app.fleet = fake_fleet
        d = self.app.plan({"gpus": [0], "model": FAKE_MODEL, "port": 18700})
        self.assertTrue(calls)
        errs = " ".join(d["conflicts"]["errors"])
        self.assertIn("card 0 is in use by docker seat", errs)
        self.assertIn("port 18700", errs)

    def test_target_port_only_known_servers(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        self.assertEqual(self.app.target_port({"sid": ["b"]}), 18555)
        self.assertEqual(self.app.target_port({"port": ["18555"]}), 18555)
        for bad in ("22", "631", "abc"):
            with self.assertRaises(C.Invalid):
                self.app.target_port({"port": [bad]})

    def test_fleet_lists_managed_and_holders(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        f = self.app.fleet(max_age=0)
        keys = {x["key"]: x for x in f["instances"]}
        self.assertIn("m:main", keys)
        self.assertNotIn("delete", keys["m:main"]["actions"])     # the first server cannot be deleted
        self.assertEqual((keys["m:b"]["running"], keys["m:b"]["port"], keys["m:b"]["gpus"]), (True, 18555, [1]))
        self.assertEqual(keys["m:b"]["health"], "down")            # nothing answers on that port
        card1 = [c for c in f["cards"] if c["index"] == 1][0]
        self.assertEqual([u["sid"] for u in card1["users"]], ["b"])

    def test_external_control_needs_adopt_and_confirm(self):
        snap = {"instances": [{"key": "d:pxa-seat", "kind": "docker", "name": "pxa-seat", "container": "pxa-seat",
                               "running": True, "port": 18600, "gpus": [0], "vram": {}, "adopted": False, "control": False}],
                "cards": []}
        self.app._fleet_cache = (time.time(), snap)
        self.app.fleet = lambda max_age=2.5: snap
        with self.assertRaises(C.Invalid) as e:
            self.app.external_control("d:pxa-seat", "stop", "pxa-seat")
        self.assertIn("read-only", str(e.exception))
        snap["instances"][0].update(adopted=True, control=True)
        with self.assertRaises(C.Invalid) as e:
            self.app.external_control("d:pxa-seat", "stop", "pxa-sea")
        self.assertIn("confirm", str(e.exception))
        with self.assertRaises(C.Invalid):
            self.app.external_control("d:pxa-seat", "rm", "pxa-seat")
        with self.assertRaises(C.Invalid):
            self.app.adopt("p:1234")                                   # a bare process cannot be adopted
        # a discovered server's card is a conflict for a new launch, its port is known to the proxy
        req = self.app.validate({"gpus": [0], "model": FAKE_MODEL, "port": 18601}, resolve_port=False)
        self.assertTrue(any("docker pxa-seat" in e for e in self.app.conflicts(req, "main")["errors"]))
        self.assertEqual(self.app.target_port({"port": ["18600"]}), 18600)

    def test_handlers(self):
        srv, port = serve_app(self.app)
        try:
            code, _, body = req(port, "/api/servers", "POST", {"name": "Second", "settings": {}})
            self.assertEqual(code, 200, body)
            sid = json.loads(body)["sid"]
            d = json.loads(req(port, "/api/servers")[2])
            self.assertEqual([x["sid"] for x in d["servers"]], ["main", sid])
            self.assertEqual(json.loads(req(port, f"/api/status?sid={sid}")[2])["running"], False)
            self.assertEqual(req(port, "/api/status?sid=nope")[0], 400)
            f = json.loads(req(port, "/api/fleet?force=1")[2])
            self.assertEqual(len(f["instances"]), 2)
            code, _, body = req(port, "/api/plan", "POST", {"sid": sid, "gpus": [0], "model": FAKE_MODEL,
                                                             "extra_args": "--path /etc"})
            self.assertEqual(code, 400)
            self.assertIn("--path", json.loads(body)["error"])
            code, _, body = req(port, "/api/levers/lint", "POST", {"levers": {"PXA_ENHANCE": "1"}})
            self.assertEqual(code, 200, body)
            self.assertIn("warnings", json.loads(body))
            self.assertEqual(req(port, "/api/servers", "DELETE", {"sid": sid})[0], 200)
            self.assertEqual(req(port, "/api/servers", "DELETE", {"sid": "main"})[0], 400)
            self.assertEqual(req(port, "/api/engine/health?port=22")[0], 400)
        finally:
            srv.shutdown()
            srv.server_close()


class ExtraArgsAndLevers(unittest.TestCase):
    def test_allow_list(self):
        self.assertEqual(C.validate_extra_args("--alias my-model -t 8 --metrics --temp=0.6"),
                         ["--alias", "my-model", "-t", "8", "--metrics", "--temp", "0.6"])
        self.assertEqual(C.validate_extra_args(["-ot", "blk\\.[0-9]+\\.ffn_.*=CPU"]), ["-ot", "blk\\.[0-9]+\\.ffn_.*=CPU"])
        self.assertEqual(C.validate_extra_args(""), [])
        for bad in ("--path /etc", "--log-file /tmp/x", "--host 0.0.0.0", "--api-key k", "-m /x.gguf",
                    "--slot-save-path /tmp", "-t", "-t -c", "--alias 'a b'", "--alias $(id)", "--alias a;b",
                    "--metrics=1", 42):
            with self.assertRaises(C.Invalid, msg=repr(bad)):
                C.validate_extra_args(bad)

    def test_extra_args_reach_the_command(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        r = app.validate({"gpus": [0], "model": FAKE_MODEL, "extra_args": "--alias zed"}, resolve_port=False)
        self.assertEqual(r["extra_args"], ["--alias", "zed"])

    def test_lever_kind_default_and_lint(self):
        rows = [{"name": "PXA_A", "default": "off (=1 on)", "rule": "", "status": "measured"},
                {"name": "PXA_B", "default": "16", "rule": "", "status": "measured"},
                {"name": "PXA_C", "default": "see site rule", "rule": "", "status": "diagnostic"}]
        self.assertEqual([C.lever_kind(r) for r in rows], ["bool", "int", "text"])
        self.assertEqual([C.lever_default_state(r) for r in rows], ["off", "value", "site"])
        w = C.lint_levers({"PXA_A": "yes please", "PXA_B": "x", "PXA_C": "1", "PXA_UNKNOWN": "1"}, rows)
        self.assertTrue(any("on/off" in x for x in w), w)
        self.assertTrue(any("whole number" in x for x in w), w)
        self.assertTrue(any("diagnostic" in x for x in w), w)
        self.assertTrue(any("already the default" in x for x in C.lint_levers({"PXA_A": "0"}, rows)))
        self.assertEqual(C.lint_levers({"PXA_A": "1", "PXA_B": "32"}, rows), [])

    def test_shell_line(self):
        self.assertEqual(C.shell_line({"B": "x y", "A": "1"}, ["/opt/llama-server", "-m", "/m/a b.gguf"]),
                         "A=1 B='x y' /opt/llama-server -m '/m/a b.gguf'")

    def test_preset_keeps_auto_port(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        app = C.App(L, port=7777, models_dirs=[MODELS])
        app.save_preset("auto", {"gpus": [0], "model": FAKE_MODEL})
        self.assertEqual(app.presets()["auto"]["port"], 0)


class TokenPersistence(unittest.TestCase):
    def test_token_survives_restart(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        old = os.environ.pop("PXA_CONTROL_TOKEN", None)
        try:
            t1 = C.load_or_make_token()
            self.assertGreaterEqual(len(t1), 16)
            self.assertEqual(C.load_or_make_token(), t1)
            self.assertEqual(os.stat(os.path.join(C.config_dir(), "token")).st_mode & 0o777, 0o600)
            os.environ["PXA_CONTROL_TOKEN"] = "env-token-0123456789"
            self.assertEqual(C.load_or_make_token(), "env-token-0123456789")
            os.environ["PXA_CONTROL_TOKEN"] = "short"
            with self.assertRaises(SystemExit):
                C.load_or_make_token()
        finally:
            os.environ.pop("PXA_CONTROL_TOKEN", None)
            if old is not None:
                os.environ["PXA_CONTROL_TOKEN"] = old


class BenchTelemetry(unittest.TestCase):
    def row(self, i, temp, clk, reasons=(), limit=250.0):
        return {"index": i, "temp_c": temp, "sm_clock_mhz": clk, "sm_clock_max_mhz": 1480.0, "power_w": 200.0,
                "power_limit_w": limit, "throttle_reasons": list(reasons)}

    def test_summary_and_warnings(self):
        samples = [[self.row(0, 60, 1400), self.row(1, 70, 1300)],
                   [self.row(0, 84, 800, ["hw_thermal_slowdown", "gpu_idle"]), self.row(1, 72, 1310)]]
        t = C.App.summarize_telemetry(samples, cards=[0], stock={0: 250.0})
        self.assertEqual([c["index"] for c in t["cards"]], [0])        # only the benchmarked card
        c = t["cards"][0]
        self.assertEqual((c["temp_max"], c["sm_clock_min"], c["samples"]), (84, 800, 2))
        self.assertIn("hw_thermal_slowdown", c["throttle"])
        self.assertTrue(any("84" in w for w in t["warnings"]))
        self.assertTrue(any("throttled" in w for w in t["warnings"]))
        low = C.App.summarize_telemetry([[self.row(0, 50, 1400, limit=150.0)]], stock={0: 250.0})
        self.assertTrue(any("below its stock 250 W" in w for w in low["warnings"]), low)
        slow = C.App.summarize_telemetry([[self.row(0, 50, 405)]], stock={})
        self.assertTrue(any("never clocked above 405" in w for w in slow["warnings"]), slow)
        self.assertEqual(C.App.summarize_telemetry([], cards=[0])["warnings"], [])


class StuckRigReport(unittest.TestCase):
    def test_report_comes_back_when_everything_hangs(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        hang = lambda *a, **k: time.sleep(30)      # noqa: E731
        app.gpus, app.gpu_telemetry, app.rig_static = hang, hang, hang
        app._rig_cache = (0.0, None)
        app.REPORT_BUDGET_S = {"gpus": 0.5, "rig": 0.5, "telemetry": 0.5}
        app.seat._append("llama_model_loader: loading model")
        t0 = time.time()
        b = app.report_bundle("main")
        self.assertLess(time.time() - t0, 5)
        notes = " ".join(b["bundle"]["collection_notes"])
        self.assertIn("nvidia-smi did not answer", notes)
        self.assertIn("telemetry timed out", notes)

    def test_bounded(self):
        self.assertEqual(C.bounded(lambda: 7, 1), 7)
        self.assertEqual(C.bounded(lambda: time.sleep(5), 0.2, "late"), "late")
        self.assertEqual(C.bounded(lambda: 1 / 0, 1, "err"), "err")


class GpuFallback(unittest.TestCase):
    def test_probe_never_raises(self):
        rows, src, err = C.probe_gpus_fallback(timeout=20)
        self.assertIsInstance(rows, list)
        self.assertTrue(rows or err)

    def test_used_when_nvidia_smi_is_missing(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        fake = os.environ.pop("PXA_LAUNCH_FAKE_GPUS")
        real_table, real_probe = app.L.gpu_table, C.probe_gpus_fallback
        app.L.gpu_table = lambda: ([], "nvidia-smi not found")
        C.probe_gpus_fallback = lambda timeout=15: ([(0, "Tesla V100-SXM2-16GB", 70, 16384, 0, "GPU-x")], "nvml", None)
        try:
            rows, err = app.gpus(max_age=0)
        finally:
            os.environ["PXA_LAUNCH_FAKE_GPUS"] = fake
            app.L.gpu_table, C.probe_gpus_fallback = real_table, real_probe
        self.assertEqual(rows[0][:3], (0, "Tesla V100-SXM2-16GB", 70))
        self.assertIn("cards read from nvml", err)


class EngineSearch(unittest.TestCase):
    def test_normalize_and_find(self):
        root = os.path.join(TMP, "src")
        for b in ("build", "build-cuda12"):
            os.makedirs(os.path.join(root, b, "bin"), exist_ok=True)
            with open(os.path.join(root, b, "bin", "llama-server"), "w") as f:
                f.write("#!/bin/sh\n")
        os.utime(os.path.join(root, "build", "bin", "llama-server"), (1, 1))
        b = os.path.join(root, "build")
        for given in (b, b + "/bin", b + "/bin/llama-server", b + "/"):
            self.assertEqual(L.normalize_engine_dir(given), b, given)
        self.assertEqual(L.find_source_builds([root]), [os.path.join(root, "build-cuda12"), b])
        old = os.environ.get("PXA_ENGINE_DIR")
        os.environ["PXA_ENGINE_DIR"] = b + "/bin"
        try:
            got = L.resolve_engine_dir()
        finally:
            os.environ.pop("PXA_ENGINE_DIR") if old is None else os.environ.__setitem__("PXA_ENGINE_DIR", old)
        self.assertEqual((got[0] if isinstance(got, tuple) else got), b)


# ---------------------------------------------------------------------------------------------
# the Encode tab: routes, auth, plain errors, the licence key never leaves its place
# ---------------------------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "tests"))
import encode_fakes as EF  # noqa: E402

ENC_SECRET = "A1b2C3d4A1b2C3d4"


class EncodeRedaction(unittest.TestCase):
    def test_licence_key_forms_are_redacted_in_every_text_that_leaves(self):
        key = EF.GOOD_KEY
        for text in ("key %s here" % key, "PXQE_KEY=%s" % key, "PXQE_KEY: %s" % key, "licence_key=%s" % key, "export PXQE_KEY='%s'" % key):
            out = C.redact_text(text, {"rigbox"}, {"alice"})
            self.assertNotIn(ENC_SECRET, out, text)
            self.assertIn("<redacted", out, text)
        o = C.redact_obj({"cmd": "pxqe run PXQE_KEY=%s" % key, "log": ["token %s" % key]}, {"rigbox"}, {"alice"})
        self.assertNotIn(ENC_SECRET, json.dumps(o))

    def test_a_key_in_the_server_log_never_reaches_the_report_bundle(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat._append("env PXQE_KEY=%s" % EF.GOOD_KEY)
        app.seat._append("plain %s line" % EF.GOOD_KEY)
        b = app.report_bundle()
        self.assertNotIn(ENC_SECRET, json.dumps(b))

    def test_score_payload_redacts_a_key_in_the_command(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "--alias", EF.GOOD_KEY, "-c", "4096"]
        app.seat.req = {"port": 8999, "model": FAKE_MODEL, "gpus": [0]}
        res = {"model": FAKE_MODEL, "port": 8999, "cards": [0], "classes": [{"class": "prose", "decode_tps": 40.0}, {"class": "long", "prefill_tps": 900.0}], "reps": 3, "greedy512_sha": "x"}
        p = app.score_payload(res)
        self.assertNotIn(ENC_SECRET, json.dumps(p))


class EncodeRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pxa-encode-routes-")
        cls.saved = {k: os.environ.get(k) for k in ("PXA_ENCODE_HOME", "PXA_ENCODER_HOME", "PXA_CONVERT_MODULES", "PXA_PACKAGE_PUBKEY", "HF_ENDPOINT",
                                                     "PXA_LICENCE_URL", "PXA_ENGINE_DIR", "PXA_MODELS_DIR", "PXA_CONTROL_CONFIG_DIR")}
        os.environ["PXA_ENCODE_HOME"] = os.path.join(cls.tmp, "data")
        os.environ["PXA_ENCODER_HOME"] = os.path.join(cls.tmp, "enchome")
        os.environ["PXA_CONVERT_MODULES"] = ""
        os.environ["PXA_PACKAGE_PUBKEY"] = EF.PUB_HEX
        os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(cls.tmp, "cfg")
        cls.models = os.path.join(cls.tmp, "models")
        os.makedirs(cls.models)
        os.environ["PXA_MODELS_DIR"] = cls.models
        cls.engine = EF.make_engine_dir(os.path.join(cls.tmp, "engine"))
        os.environ["PXA_ENGINE_DIR"] = cls.engine
        cls.hf = EF.FakeHF({"tiny/Tiny-Qwen3": EF.tiny_repo()}).start()
        os.environ["HF_ENDPOINT"] = cls.hf.url
        cls.lic = EF.FakeLicence(os.path.join(cls.tmp, "pkg")).start()
        os.environ["PXA_LICENCE_URL"] = cls.lic.url
        cls.app = C.App(L, port=0)
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.app.encode_shutdown()
        cls.hf.stop()
        cls.lic.stop()
        for k, v in cls.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, path, headers=None):
        code, _, body = req(self.port, path, headers=headers)
        return code, json.loads(body)

    def post(self, path, body=None):
        code, _, raw = req(self.port, path, "POST", body if body is not None else {})
        return code, json.loads(raw)

    def wait_pkg(self):
        t0 = time.time()
        while time.time() - t0 < 60:
            st = self.get("/api/encode/state")[1]
            if not st["pkg"]["running"]:
                return st
            time.sleep(0.1)
        self.fail("package install still running")

    def test_01_static_files_and_the_tab_are_served(self):
        for path, ctype in (("/encode.js", "application/javascript"), ("/encode.css", "text/css")):
            code, h, body = req(self.port, path)
            self.assertEqual(code, 200)
            self.assertTrue(h["Content-Type"].startswith(ctype))
            self.assertGreater(len(body), 1000)
        page = req(self.port, "/")[2].decode()
        self.assertIn('data-tab="encode"', page)
        self.assertIn('id="p-encode"', page)
        self.assertIn('src="/encode.js"', page)
        self.assertIn('href="/encode.css"', page)

    def test_02_empty_start_and_plain_errors(self):
        st = self.get("/api/encode/state")[1]
        self.assertEqual((st["encoders"], st["edition"], st["key_set"]), ([], None, False))
        self.assertEqual(len(st["catalog"]), len(__import__("pxa_encode_plan").TIERS))
        self.assertIn("ko-fi.com", st["how_to_get_pro"]["kofi"])
        for path, body, frag in (("/api/encode/inspect", {}, "Paste a Hugging Face model"), ("/api/encode/inspect", {"source": "justaword"}, "neither a Hugging Face"),
                                 ("/api/encode/plan", {"source": "tiny/Tiny-Qwen3"}, "Pick the cards"), ("/api/encode/start", {}, "Paste a Hugging Face model"),
                                 ("/api/encode/key", {"key": "nope"}, "starts with pxk1."), ("/api/encode/get", {"edition": "gold"}, "Free or Pro"),
                                 ("/api/encode/get", {"edition": "pro"}, "Paste your key first"), ("/api/encode/add-encoder", {"path": "/nope"}, "does not exist"),
                                 ("/api/encode/pause", {"id": "x"}, "no such encode job"), ("/api/encode/test", {"id": "x"}, "no such encode job")):
            code, j = self.post(path, body)
            self.assertEqual(code, 400, (path, j))
            self.assertIn(frag, j["error"])
            self.assertNotIn("Traceback", j["error"])
        code, j = self.get("/api/encode/job?id=zzz")
        self.assertEqual(code, 400)
        code, j = self.get("/api/encode/browse?path=" + urllib_quote(self.tmp))
        self.assertEqual(code, 200)
        self.assertIn("models", [e["name"] for e in j["entries"]])
        # a body that is not an object is ignored, not a 500
        code, _, raw = req(self.port, "/api/encode/inspect", "POST", [1, 2])
        self.assertEqual(code, 400)

    def test_03_free_download_then_a_whole_encode_through_the_routes(self):
        self.assertEqual(self.post("/api/encode/get", {"edition": "free"})[0], 200)
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        self.assertEqual((st["edition"], len(st["encoders"])), ("free", 1))
        code, p = self.post("/api/encode/inspect", {"source": "tiny/Tiny-Qwen3"})
        self.assertEqual((code, p["arch"]), (200, "Qwen3ForCausalLM"))
        code, pl = self.post("/api/encode/plan", {"source": "tiny/Tiny-Qwen3", "cards": [0, 1]})
        self.assertEqual(code, 200)
        self.assertTrue(pl["recommended"])
        self.assertEqual(len(pl["locked"]), 7)
        body = {"source": "tiny/Tiny-Qwen3", "cards": [0, 1], "tier": pl["recommended"][0], "work_dir": os.path.join(self.tmp, "work"), "out_dir": self.models}
        code, ck = self.post("/api/encode/checks", body)
        self.assertEqual((code, ck["can_start"]), (200, True), ck.get("refusal"))
        code, j = self.post("/api/encode/start", body)
        self.assertEqual(code, 200)
        t0 = time.time()
        while time.time() - t0 < 90:
            j = self.get("/api/encode/job?id=" + j["id"])[1]
            if j["status"] not in ("running", "queued"):
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertTrue(os.path.isfile(j["result"]["path"]))
        lg = self.get("/api/encode/log?id=" + j["id"])[1]
        self.assertTrue(any(ln.startswith("pxqe make:") for ln in lg["lines"]))      # the one-command flow: pxqe make (its human lines, not its @pxqe progress lines)
        self.assertFalse(any(ln.startswith("@pxqe") for ln in lg["lines"]))
        self.encoded = j
        type(self).job = j

    def test_04_test_it_uses_the_existing_server_flow(self):
        j = getattr(type(self), "job", None)
        if not j:
            self.skipTest("needs the encode of the previous step")
        calls = []
        real = self.app.start
        self.app.start = lambda b: (calls.append(b), {"ok": True, "text": "stub", "sid": b["sid"]})[1]
        try:
            code, r = self.post("/api/encode/test", {"id": j["id"]})
        finally:
            self.app.start = real
        self.assertEqual(code, 200, r)
        self.assertEqual(r["sid"], "encode-test")
        self.assertEqual(calls[0]["model"], j["result"]["path"])
        self.assertEqual(calls[0]["gpus"], [0, 1])         # the cards it was encoded for
        self.assertIsNone(r["note"])
        self.app.start = lambda b: (calls.append(b), {"ok": True, "text": "stub", "sid": b["sid"]})[1]
        try:
            self.post("/api/encode/test", {"id": j["id"], "cards": [1]})
        finally:
            self.app.start = real
        self.assertEqual(calls[1]["gpus"], [1])
        self.assertIn(self.models, load_cfg_model_dirs())
        self.assertEqual(self.app.profiles()["encode-test"]["name"], "Encode test")
        code, r = self.post("/api/encode/test", {"id": "nope"})
        self.assertEqual(code, 400)

    def test_05_key_is_stored_0600_masked_and_never_returned(self):
        code, st = self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        self.assertEqual(code, 200)
        self.assertTrue(st["key_set"])
        self.assertNotIn(ENC_SECRET, json.dumps(st))
        self.assertTrue(st["key_masked"].startswith("pxk1.K-TEST0001."))
        mode = stat_mode(C.config_path())
        self.assertEqual(mode, 0o600)
        with open(C.config_path()) as f:
            self.assertEqual(json.load(f)["encode"]["licence_key"], EF.GOOD_KEY)
        for path in ("/api/encode/state", "/api/info", "/api/rig", "/api/presets", "/api/user", "/api/servers"):
            code, _, raw = req(self.port, path)
            self.assertNotIn(ENC_SECRET.encode(), raw, path)
        self.assertNotIn(ENC_SECRET, json.dumps(self.app.report_bundle()))
        code, st = self.post("/api/encode/key", {"key": ""})
        self.assertFalse(st["key_set"])
        self.assertNotIn(ENC_SECRET, open(C.config_path()).read())

    def test_06_the_key_only_reaches_the_licence_server_and_only_in_the_body(self):
        self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        self.lic.requests.clear()
        self.hf.requests.clear()
        self.post("/api/encode/get", {"edition": "pro"})
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        self.assertEqual(st["edition"], "pro")
        posts = [r for r in self.lic.requests if r["method"] == "POST"]
        self.assertTrue(posts)
        for r in self.lic.requests:
            self.assertNotIn(ENC_SECRET, r["path"])
            self.assertNotIn(ENC_SECRET, json.dumps(r["headers"]))
            if r["method"] != "POST":
                self.assertNotIn(ENC_SECRET, r["body"])
        self.assertFalse(any(ENC_SECRET in json.dumps(r) for r in self.hf.requests))
        code, lic = self.post("/api/encode/licence", {})
        self.assertEqual(code, 200)
        self.assertEqual(lic["encoders"][0]["licence"]["state"], "valid")

    def test_07_encode_tab_unavailable_is_a_plain_400_not_a_500(self):
        real = C.ENC
        C.ENC = None
        a = C.App(L, port=0)
        srv, port = serve_app(a)
        try:
            code, _, raw = req(port, "/api/encode/state")
            self.assertEqual(code, 400)
            self.assertIn("not available in this install", json.loads(raw)["error"])
        finally:
            C.ENC = real
            srv.shutdown()
            srv.server_close()

    def test_08_the_gpu_runtime_routes(self):
        """A Pro encoder on a computer without the CUDA libraries: state names what is missing, GET runtime fetches the offer, POST runtime/get downloads +
        verifies + installs it, and the encoder loads afterwards. Public data only: no key goes to /v1/runtime/*."""
        self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        self.lic.pro_fake = {"needs_runtime": True}
        self.lic.rebuild()
        self.post("/api/encode/get", {"edition": "pro", "update": True})
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        rt = st["runtime"]
        self.assertTrue(rt["applies"] and rt["need"], rt)
        self.assertEqual(rt["missing"], ["libcublas.so.12", "libcusolver.so.11"])
        self.assertIsNone(rt["offer"])
        code, v = self.get("/api/encode/runtime")
        self.assertEqual(code, 200)
        self.assertEqual((v["offer"]["id"], v["need"]), ("cuda12.8.1-r1", True))
        self.assertIn("B", v["offer"]["size_h"])
        self.lic.requests.clear()
        self.assertEqual(self.post("/api/encode/runtime/get", {})[0], 200)
        t0 = time.time()
        while time.time() - t0 < 60:
            st = self.get("/api/encode/state")[1]
            if not st["runtime"]["job"]["running"]:
                break
            time.sleep(0.1)
        self.assertEqual(st["runtime"]["job"]["phase"], "done", st["runtime"]["job"])
        self.assertTrue(st["runtime"]["ok"] and not st["runtime"]["need"])
        self.assertEqual(st["runtime"]["source"], "pack")
        self.assertEqual(st["encoders"][0]["runtime"]["lib"], "loadable")
        self.assertTrue(all(r["method"] == "GET" and r["body"] == "" for r in self.lic.requests if "/v1/runtime/" in r["path"]))
        self.assertEqual(self.post("/api/encode/runtime/cancel", {})[0], 200)
        self.assertEqual(self.post("/api/encode/runtime/get", {})[0], 200)                  # already installed: it is fetched again, harmlessly
        self.wait_pkg()
        t0 = time.time()
        while self.get("/api/encode/state")[1]["runtime"]["job"]["running"] and time.time() - t0 < 60:
            time.sleep(0.1)
        self.lic.pro_fake = {}
        self.lic.rebuild()

    def test_09_who_can_load_this_file_through_the_routes(self):
        """The lock choice end to end through the HTTP routes: the plan carries what the key's tier may choose, a choice the plan does not allow is refused in
        one sentence before anything runs, an allowed one reaches the encoder, and the finished job says who can load the file. The server's switch off:
        the choice is disabled and the file is written open."""
        self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        on = {"enabled": True, "allowed_modes": ["personal", "supporters"], "default_mode": "personal", "epoch": "2026-10"}
        self.lic.pro_fake = {"lock_support": True, "lock_server": on}
        self.lic.rebuild()
        self.post("/api/encode/get", {"edition": "pro", "update": True})
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        self.assertEqual(self.post("/api/encode/rescan", {})[0], 200)
        base = {"source": "tiny/Tiny-Qwen3", "cards": [0, 1], "work_dir": os.path.join(self.tmp, "work-lock"), "out_dir": self.models}
        code, pl = self.post("/api/encode/plan", base)
        self.assertEqual(code, 200)
        self.assertEqual((pl["lock"]["state"], [c["label"] for c in pl["lock"]["choices"]], pl["lock"]["default"]),
                         ("on", ["Only me (recommended)", "Any PXA supporter"], "personal"))
        code, ck = self.post("/api/encode/checks", dict(base, tier="pxqn4", lock="open"))
        self.assertEqual(code, 200)
        self.assertFalse(ck["can_start"])
        self.assertIn("Your plan does not allow a file that anyone can load (no lock). It allows: only me, any PXA supporter.", ck["refusal"])
        code, err = self.post("/api/encode/start", dict(base, tier="pxqn4", lock="open"))
        self.assertEqual(code, 400)
        self.assertIn("It allows: only me, any PXA supporter.", err["error"])
        code, err = self.post("/api/encode/start", dict(base, tier="pxqn4", lock="everyone"))
        self.assertEqual(code, 400)
        self.assertIn("only you, any PXA supporter, or anyone", err["error"])
        code, j = self.post("/api/encode/start", dict(base, tier="pxqn4", lock="supporters"))
        self.assertEqual(code, 200)
        t0 = time.time()
        while time.time() - t0 < 90:
            j = self.get("/api/encode/job?id=" + j["id"])[1]
            if j["status"] not in ("running", "queued"):
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertEqual(j["params"]["lock"], "supporters")
        lock = j["result"]["lock"]
        self.assertEqual((lock["mode"], lock["locked"], lock["title"]), ("supporters", True, "Locked to PXA supporters"))
        self.assertIn("loads in PXA v3 or newer", lock["needs"])
        self.assertEqual(L.gguf_header(j["result"]["path"])["kv"].get("pxa.lock.mode"), "supporters")
        # the licence server's switch goes off: the page reads it again on a Rescan; the choice is disabled and the next file is open
        self.lic.pro_fake = {"lock_support": True, "lock_server": {"enabled": False, "allowed_modes": ["open"], "default_mode": "open", "epoch": "2026-10"}}
        self.lic.rebuild()
        self.post("/api/encode/get", {"edition": "pro", "update": True})
        self.assertEqual(self.wait_pkg()["pkg"]["phase"], "done")
        self.post("/api/encode/rescan", {})
        pl = self.post("/api/encode/plan", base)[1]
        self.assertEqual((pl["lock"]["state"], pl["lock"]["choices"], pl["lock"]["text"]), ("off", [], "File locking turns on with the next encoder update"))
        code, j = self.post("/api/encode/start", dict(base, tier="pxqn3", lock="personal"))
        self.assertEqual(code, 200)
        t0 = time.time()
        while time.time() - t0 < 90:
            j = self.get("/api/encode/job?id=" + j["id"])[1]
            if j["status"] not in ("running", "queued"):
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertEqual((j["params"]["lock"], j["result"]["lock"]["locked"], j["result"]["lock"]["title"]), ("open", False, "Not locked"))
        self.lic.pro_fake = {}
        self.lic.rebuild()


class EncodeAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pxa-encode-auth-")
        cls.saved = os.environ.get("PXA_ENCODE_HOME")
        os.environ["PXA_ENCODE_HOME"] = os.path.join(cls.tmp, "data")
        cls.app = C.App(L, port=7777, lan=True, token="s3cret-token")
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        if cls.saved is None:
            os.environ.pop("PXA_ENCODE_HOME", None)
        else:
            os.environ["PXA_ENCODE_HOME"] = cls.saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_encode_route_needs_the_token_and_a_same_origin_post(self):
        for path in ("/api/encode/state", "/api/encode/job?id=x", "/api/encode/log?id=x", "/api/encode/browse", "/api/encode/runtime"):
            self.assertEqual(req(self.port, path)[0], 401, path)
        for path in ("/api/encode/start", "/api/encode/key", "/api/encode/get", "/api/encode/add-encoder", "/api/encode/test", "/api/encode/cancel", "/api/encode/runtime/get", "/api/encode/runtime/cancel"):
            self.assertEqual(req(self.port, path, "POST", {})[0], 401, path)
            self.assertEqual(req(self.port, path, "POST", {}, headers={"X-PXA-Token": "s3cret-token", "Origin": "http://evil.example"})[0], 403, path)
        self.assertEqual(req(self.port, "/api/encode/state", headers={"X-PXA-Token": "s3cret-token"})[0], 200)


# ---------------------------------------------------------------------------------------------
# the Live tab: rolling history of the cards and the servers
# ---------------------------------------------------------------------------------------------
class ScriptedEngine(http.server.BaseHTTPRequestHandler):
    """what a PXA llama-server answers, from a dict the test edits between samples."""
    state = {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        st, p = ScriptedEngine.state, self.path.split("?")[0]
        if p == "/health":
            return self._send(200, {"status": "ok"})
        if p == "/slots":
            if st.get("no_slots"):
                return self._send(501, {"error": "no slots"})
            return self._send(200, st["slots"])
        if p == "/props":
            return self._send(200, st["props"])
        if p == "/metrics":
            if not st.get("metrics"):
                return self._send(501, {"error": "no metrics"})
            return self._send(200, "".join("llamacpp:%s %s\n" % kv for kv in st["metrics"].items()).encode(), "text/plain")
        if p == "/pxa/stats":
            return self._send(200, {"records": st.get("records", [])})
        self._send(404, {"error": "nf"})


def slot(i, task, busy, n_decoded, remain=-1, dtot=0, dacc=0, ema=0.7):
    return {"id": i, "id_task": task, "state": 1 if busy else 0, "prompt": "SECRET PROMPT TEXT " * 50,
            "next_token": {"n_decoded": n_decoded, "n_remain": remain}, "n_draft_total": dtot, "n_draft_accepted": dacc,
            "spec_accept_ema": ema}


class LiveHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ScriptedEngine)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.eport = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        snap = {"instances": [{"key": "m:main", "kind": "managed", "name": "Server 1", "label": "Server 1", "running": True,
                               "port": self.eport, "gpus": [0, 1], "health": "ok", "vram": {}, "model_file": "tiny.gguf"}], "cards": []}
        self.app.fleet = lambda max_age=2.5: snap
        self.snap = snap
        ScriptedEngine.state = {"slots": [slot(0, -1, False, 0)], "props": {"n_ctx": 1000, "kv_cache_used_cells": 100, "total_slots": 1}}

    def tearDown(self):
        self.app.live.close()

    def rows(self):
        return self.app.live.snapshot(since=0)["servers"][0]["rows"]

    def col(self, name):
        return C.SRV_COLS.index(name)

    def test_decode_rate_from_slot_progress(self):
        lv = self.app.live
        t = 10000.0
        lv.tick(now=t)                                                   # the first sample has nothing to difference
        self.assertIsNone(self.rows()[-1][self.col("dec")])
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 0, 100)]
        lv.tick(now=t + 2)
        r = self.rows()[-1]
        self.assertEqual((r[self.col("busy")], r[self.col("dec")]), (1, None))       # reading the prompt: nothing written yet, no speed
        self.assertEqual(lv.snapshot(0)["servers"][0]["phase"], "prefill")
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 70, 30)]
        lv.tick(now=t + 4)
        self.assertAlmostEqual(self.rows()[-1][self.col("dec")], 35.0, places=1)
        self.assertEqual(lv.snapshot(0)["servers"][0]["phase"], "decode")
        self.assertEqual(lv.snapshot(0)["servers"][0]["source"], "slot progress")
        ScriptedEngine.state["slots"] = [slot(0, -1, False, 100)]
        lv.tick(now=t + 6)
        self.assertIsNone(self.rows()[-1][self.col("dec")])                          # idle is a gap in the line, not a 0 t/s

    def test_a_slot_that_stalls_while_writing_is_zero(self):
        lv = self.app.live
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 50, 50)]
        lv.tick(now=25000.0)
        lv.tick(now=25002.0)                                                         # same counter two seconds later: it is stuck
        self.assertEqual(self.rows()[-1][self.col("dec")], 0.0)

    def test_a_new_task_does_not_count_the_old_counter(self):
        lv = self.app.live
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 500, 10)]
        lv.tick(now=20000.0)
        ScriptedEngine.state["slots"] = [slot(0, 8, True, 20, 100)]                  # another task on the same slot: counter restarted
        lv.tick(now=20002.0)
        self.assertIsNone(self.rows()[-1][self.col("dec")])

    def test_speculation_and_expert_cache_rates(self):
        lv = self.app.live
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 10, 90, dtot=20, dacc=10, ema=0.55)]
        ScriptedEngine.state["props"].update(pxa_xcache={"hits": 1000, "misses": 100, "swaps_started": 5})
        lv.tick(now=30000.0)
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 60, 40, dtot=60, dacc=40, ema=0.6)]
        ScriptedEngine.state["props"].update(pxa_xcache={"hits": 1090, "misses": 110, "swaps_started": 9})
        lv.tick(now=30002.0)
        r = self.rows()[-1]
        self.assertAlmostEqual(r[self.col("acc")], 30 / 40)
        self.assertAlmostEqual(r[self.col("ema")], 0.6)
        self.assertAlmostEqual(r[self.col("xhit")], 0.9)
        self.assertAlmostEqual(r[self.col("xswap")], 2.0)
        self.assertAlmostEqual(r[self.col("ctx")], 0.1)
        meta = lv.snapshot(0)["servers"][0]
        self.assertTrue(meta["has_spec"] and meta["has_xcache"])
        # counters going backwards (a restarted server) are a gap, never a negative rate
        ScriptedEngine.state["props"].update(pxa_xcache={"hits": 5, "misses": 1, "swaps_started": 0})
        lv.tick(now=30004.0)
        self.assertIsNone(self.rows()[-1][self.col("xhit")])

    def test_exact_counters_when_the_server_has_metrics(self):
        lv = self.app.live
        m = {"tokens_predicted_total": 1000, "tokens_predicted_seconds_total": 30.0, "prompt_tokens_total": 5000, "prompt_seconds_total": 6.0}
        ScriptedEngine.state.update(metrics=dict(m), slots=[slot(0, 7, True, 5, 90)])
        lv.tick(now=40000.0)
        ScriptedEngine.state["metrics"] = {"tokens_predicted_total": 1100, "tokens_predicted_seconds_total": 32.0,
                                           "prompt_tokens_total": 6000, "prompt_seconds_total": 7.0}
        lv.tick(now=40002.0)
        r = self.rows()[-1]
        self.assertAlmostEqual(r[self.col("dec")], 50.0)             # 100 tokens in 2 busy seconds
        self.assertAlmostEqual(r[self.col("pre")], 1000.0)           # 1000 prompt tokens in 1 busy second
        self.assertEqual(lv.snapshot(0)["servers"][0]["source"], "engine counters")

    def test_no_slots_endpoint_is_a_gap_not_a_crash(self):
        ScriptedEngine.state["no_slots"] = True
        self.app.live.tick(now=50000.0)
        self.app.live.tick(now=50002.0)
        r = self.rows()[-1]
        self.assertEqual((r[self.col("dec")], r[self.col("busy")]), (None, None))
        self.assertAlmostEqual(r[self.col("ctx")], 0.1)               # /props still answers

    def test_prompt_text_never_leaves_the_server_object(self):
        ScriptedEngine.state["props"]["model_alias"] = "/home/someone/models/secret-dir/tiny.gguf"
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 5, 90)]
        self.app.live.tick(now=60000.0)
        self.app.live.tick(now=60002.0)
        snap = json.dumps(self.app.live.snapshot(0))
        self.assertNotIn("SECRET", snap)
        self.assertNotIn("secret-dir", snap)                          # a model is named by its file, never by its path
        self.assertEqual(self.app.live.snapshot(0)["servers"][0]["model"], "tiny.gguf")

    def test_finished_requests_are_kept_once(self):
        lv = self.app.live
        rec = {"ts": 70010.0, "slot": 0, "n_prompt": 900, "n_cached": 100, "prompt_n": 800, "prompt_ms": 1000.0, "prefill_tps": 800.0,
               "n_gen": 100, "gen_ms": 3000.0, "decode_tps": 33.3, "draft_n": 40, "draft_acc": 30}
        ScriptedEngine.state["records"] = [rec]
        lv.tick(now=70012.0)
        lv.tick(now=70020.0)                                         # same record offered again: not added twice
        reqs = lv.snapshot(0)["servers"][0]["reqs"]
        self.assertEqual(len(reqs), 1)
        self.assertEqual(dict(zip(C.REQ_COLS, reqs[0]))["decode_tps"], 33.3)
        ScriptedEngine.state["records"] = [rec, dict(rec, ts=70030.0, slot=1)]
        lv.tick(now=70032.0)
        lv.tick(now=70040.0)
        self.assertEqual(len(lv.snapshot(0)["servers"][0]["reqs"]), 2)

    def test_since_step_and_pruning(self):
        lv = self.app.live
        now = time.time()
        for i in range(10):
            lv.tick(now=now - 100 + i * 10)
        s = lv.snapshot(since=now - 45)
        self.assertEqual(len(s["servers"][0]["rows"]), 4)            # rows newer than `since`, nothing older
        b = lv.snapshot(since=0, step=30)
        self.assertLess(len(b["servers"][0]["rows"]), 10)
        lv.tick(now=now + C.LIVE_KEEP_S + 100)                       # everything older than the keep time goes
        s = lv.snapshot(since=0)
        self.assertEqual(len(s["servers"][0]["rows"]), 1)
        self.assertEqual(len(s["cards"][0]["rows"]), 1)

    def test_colour_slot_is_first_seen_and_never_repainted(self):
        lv = self.app.live
        a, b, c = lv.slot_for("m:main"), lv.slot_for("d:seat"), lv.slot_for("p:1")
        self.assertEqual((a, b, c), (0, 1, 2))
        self.assertEqual(lv.slot_for("d:seat"), 1)
        self.assertEqual(lv.slot_for("m:main"), 0)
        self.assertEqual([lv.slot_for("x%d" % i) for i in range(6)], [3, 4, 5, 6, 7, 0])     # eight colours, then they repeat

    def test_stopped_server_keeps_its_history_until_it_ages_out(self):
        lv = self.app.live
        lv.tick(now=time.time())
        self.snap["instances"][0]["running"] = False
        lv.tick(now=time.time() + 2)
        s = lv.snapshot(since=0)["servers"][0]
        self.assertFalse(s["up"])
        self.assertTrue(s["rows"])

    def test_route_and_the_first_look(self):
        srv, port = serve_app(self.app)
        try:
            code, _, body = req(port, "/api/live")
            self.assertEqual(code, 200, body)
            d = json.loads(body)
            self.assertEqual(d["cols"]["card"], C.CARD_COLS)
            self.assertGreaterEqual(d["ticks"], 1)                   # the first look takes a sample right away
            self.assertEqual(len(d["cards"]), 2)                     # the fake rig has two cards
            self.assertEqual([s["key"] for s in d["servers"]], ["m:main"])
            self.assertEqual(json.loads(req(port, "/api/live?since=abc&step=zzz")[2])["since"] > 0, True)
            self.assertEqual(req(port, "/live.js")[0], 200)
            self.assertEqual(req(port, "/live.css")[0], 200)
            page = req(port, "/")[2].decode()
            self.assertIn('data-tab="live"', page)
            self.assertIn('src="/live.js"', page)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_gpu_live_one_call_and_throttle_field_discovery(self):
        calls = []
        old_which, old_fake = C.shutil_which, os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
        try:
            C.shutil_which = lambda x: "/usr/bin/" + x
            self.app.gpus = lambda max_age=1.5: ([(0, "NVIDIA Tesla V100-PCIE-16GB", 70, 16384, 0, "u0"), (1, "NVIDIA GeForce GTX 1080 Ti", 61, 11264, 0, "u1")], None)

            def run(cmd, timeout=20):
                calls.append(cmd)
                q = [a for a in cmd if a.startswith("--query-gpu=")][0]
                if "clocks_throttle_reasons.active" in q:
                    return None                                       # a newer driver renamed it
                if "clocks_event_reasons.active" in q:
                    return "0, 9000, 71, 64, 172.5, 250, 1380, 0x4\n1, [N/A], 0, 40, [N/A], 250, 1500, 0x0"
                return None
            self.app.L = type("LL", (), {"_run": staticmethod(run), "CARD_CLASS": L.CARD_CLASS})
            rows = self.app.gpu_live()
            self.assertEqual([r["index"] for r in rows], [0, 1])
            self.assertEqual((rows[0]["mem"], rows[0]["util"], rows[0]["temp"], rows[0]["power"], rows[0]["throttle"]), (9000.0, 71.0, 64.0, 172.5, ["sw_power_cap"]))
            self.assertEqual((rows[1]["mem"], rows[1]["power"], rows[1]["throttle"]), (0, None, []))     # [N/A] is None, memory falls back to the card table
            self.assertEqual(self.app._thr_field if hasattr(self.app, "_thr_field") else self.app.live._thr_field, "clocks_event_reasons.active")
            n = len(calls)
            self.app.gpu_live()
            self.assertEqual(len(calls) - n, 1)                       # the working field is remembered: one call per reading
        finally:
            C.shutil_which = old_which
            if old_fake is not None:
                os.environ["PXA_LAUNCH_FAKE_GPUS"] = old_fake

    @unittest.skipUnless(os.path.exists("/proc/meminfo") and os.path.exists("/proc/stat"), "needs /proc")
    def test_host_memory_and_cpu(self):
        lv = self.app.live
        now = time.time()
        lv.tick(now=now)
        lv.tick(now=now + 2)
        h = lv.snapshot(0)["host"]
        self.assertEqual(len(h["rows"]), 2)
        t, ram, cpu, swap = h["rows"][-1]
        self.assertTrue(0 < ram <= h["ram_total_mib"])
        self.assertTrue(cpu is not None and 0 <= cpu <= 100)               # the first sample has nothing to difference, the second does
        self.assertIsNone(h["rows"][0][2])

    def test_bucket_rows(self):
        rows = [(0.0, 1.0, None), (1.0, 3.0, None), (2.0, None, 4.0), (10.0, 5.0, 6.0), (11.0, 7.0, None), (12.0, 9.0, None)]
        out = C.bucket_rows(rows, 5)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1:], [2.0, 4.0])
        self.assertEqual(out[1][1:], [7.0, 6.0])
        self.assertEqual(C.bucket_rows(rows, 0), [list(r) for r in rows])


# ---------------------------------------------------------------------------------------------
# v3.1 telemetry history: the Live sampler keeps running in the background and fills telemetry.db
# ---------------------------------------------------------------------------------------------
class TelemetryHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ScriptedEngine)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.eport = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        for k in ("PXA_CONTROL_TELEMETRY", "PXA_CONTROL_METRICS", "PXA_CONTROL_TELEMETRY_DB"):
            os.environ.pop(k, None)
        self.apps = []
        ScriptedEngine.state = {"slots": [slot(0, -1, False, 0)], "props": {"n_ctx": 1000, "kv_cache_used_cells": 250, "total_slots": 1}}

    def tearDown(self):
        for app, srv in self.apps:
            if srv is not None:
                srv.shutdown()
                srv.server_close()
            app.live.close()
            app.stop_telemetry()
        for k in ("PXA_CONTROL_TELEMETRY", "PXA_CONTROL_METRICS", "PXA_CONTROL_TELEMETRY_DB"):
            os.environ.pop(k, None)

    def make(self, key="m:main", lan=False, token=None, http=True):
        app = C.App(L, port=7777, models_dirs=[MODELS], lan=lan, token=token)
        snap = {"instances": [{"key": key, "kind": "managed" if key.startswith("m:") else "process", "name": "Server 1",
                               "label": "Server 1", "running": True, "port": self.eport, "gpus": [0, 1], "health": "ok",
                               "vram": {}, "model_file": "tiny.gguf"}], "cards": []}
        app.fleet = lambda max_age=2.5: snap
        srv = port = None
        if http:
            srv, port = serve_app(app)
        self.apps.append((app, srv))
        return app, port

    def drive(self, app, n=4, every=10.0):
        """n samples, `every` seconds apart, ending now; decoding at 50 t/s with two finished requests."""
        t0 = time.time() - n * every
        for i in range(n):
            ScriptedEngine.state["slots"] = [slot(0, 7, True, 1 + int(50 * every * i), 100000)]
            ScriptedEngine.state["records"] = [
                {"ts": t0 + 1, "slot": 0, "n_prompt": 900, "n_cached": 100, "prompt_n": 800, "prompt_ms": 1000.0, "prefill_tps": 800.0,
                 "n_gen": 100, "gen_ms": 2000.0, "decode_tps": 50.0, "draft_n": 0, "draft_acc": 0},
                {"ts": t0 + every * i + 2, "slot": 0, "n_prompt": 50, "n_cached": 0, "prompt_n": 50, "prompt_ms": 100.0, "prefill_tps": 500.0,
                 "n_gen": 10, "gen_ms": 200.0, "decode_tps": 50.0, "draft_n": 0, "draft_acc": 0}]
            app.live.tick(now=t0 + every * i, wait=True)
        app.telemetry.flush(force=True)
        self.assertTrue(app.telemetry.sync())

    def test_history_is_recorded_and_served(self):
        app, port = self.make()
        self.assertIsNotNone(app.start_telemetry())
        self.assertTrue(app.telemetry.is_writer)
        self.drive(app)
        code, d = jreq(port, "/api/telemetry/series?kind=server&since=%d&step=10" % (time.time() - 3600))
        self.assertEqual(code, 200)
        self.assertEqual(d["source"], "raw")
        s = d["series"][0]
        self.assertEqual((s["key"], s["label"], s["info"]["model_file"]), ("m:main", "Server 1", "tiny.gguf"))
        col = d["cols"].index
        decs = [r[col("dec")] for r in s["rows"] if r[col("dec")] is not None]
        self.assertTrue(decs and all(abs(v - 50.0) < 0.5 for v in decs), decs)
        self.assertTrue(all(abs(r[col("ctx")] - 0.25) < 1e-6 for r in s["rows"]))
        # the old record back-filled by the first poll is kept as a request but not counted into an interval
        self.assertLessEqual(sum(r[col("req")] or 0 for r in s["rows"]), 4)
        code, cards = jreq(port, "/api/telemetry/series?kind=card&since=%d" % (time.time() - 3600))
        self.assertEqual(sorted(x["key"] for x in cards["series"]), ["0", "1"])
        code, r = jreq(port, "/api/telemetry/requests?since=%d" % (time.time() - 3600))
        self.assertGreaterEqual(len(r["rows"]), 2)
        code, h = jreq(port, "/api/telemetry/history?since=%d" % (time.time() - 3600))
        self.assertEqual(code, 200)
        self.assertEqual(set(h["cols"]), {"card", "server", "host"})
        self.assertEqual(h["servers"][0]["key"], "m:main")
        code, st = jreq(port, "/api/telemetry")
        self.assertTrue(st["recording"])
        self.assertGreater(st["raw_rows"], 0)
        self.assertIn("server", {x["kind"] for x in st["series_list"]})

    def test_prompt_text_and_paths_never_reach_the_file(self):
        app, port = self.make()
        app.start_telemetry()
        ScriptedEngine.state["props"]["model_alias"] = "/home/someone/secret-dir/tiny.gguf"
        self.drive(app)
        app.stop_telemetry()
        blob = b""
        for p in glob_db(C.config_dir()):
            with open(p, "rb") as f:
                blob += f.read()
        self.assertNotIn(b"SECRET", blob)
        self.assertNotIn(b"secret-dir", blob)

    def test_csv_download(self):
        app, port = self.make()
        app.start_telemetry()
        self.drive(app)
        u = "http://127.0.0.1:%d/api/telemetry/csv?kind=card&since=%d&step=10" % (port, time.time() - 3600)
        with urllib.request.urlopen(u, timeout=10) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/csv"))
            self.assertIn("attachment;", r.headers["Content-Disposition"])
            lines = r.read().decode().splitlines()
        self.assertTrue(lines[0].startswith("time_utc,unix_s,card,label,mem"))
        self.assertGreaterEqual(len(lines), 1 + 2)
        self.assertEqual(req(port, "/api/telemetry/csv?kind=disk")[0], 400)

    def test_off_by_env_and_by_setting(self):
        os.environ["PXA_CONTROL_TELEMETRY"] = "0"
        app, port = self.make()
        self.assertIsNone(app.start_telemetry())
        code, d = jreq(port, "/api/telemetry/series?kind=card")
        self.assertEqual(code, 400)
        self.assertIn("off", d["error"])
        self.assertFalse(app.live.background)
        os.environ.pop("PXA_CONTROL_TELEMETRY")
        code, st = jreq(port, "/api/telemetry/settings", "POST", {"enabled": True, "raw_days": 3})
        self.assertEqual(code, 200)
        self.assertTrue(st["recording"])
        self.assertEqual(st["settings"]["raw_days"], 3.0)
        self.assertEqual(C.load_config()["telemetry"]["raw_days"], 3.0)
        code, st = jreq(port, "/api/telemetry/settings", "POST", {"enabled": False})
        self.assertFalse(st["recording"])
        self.assertIsNone(app.telemetry)
        self.assertEqual(req(port, "/api/telemetry/settings", "POST", {"colour": "red"})[0], 400)

    def test_background_sampler_runs_with_nobody_looking(self):
        app, port = self.make(http=False)
        app.start_telemetry()
        self.assertTrue(app.live.background)
        app.live.bg_every = 0.25                     # test pace; the shipping floor is LIVE_FAST_S
        app.live.last_any = time.time() - C.LIVE_IDLE_S - 60     # the last viewer left long ago
        t = time.time()
        while app.live.ticks < 3 and time.time() - t < 15:
            time.sleep(0.1)
        self.assertGreaterEqual(app.live.ticks, 3)
        self.assertTrue(app.live.thread.is_alive())

    def test_without_history_the_sampler_still_stops_when_idle(self):
        app, port = self.make(http=False)
        app.live.touch()
        app.live.last_any = time.time() - C.LIVE_IDLE_S - 60
        app.live.last_fast = 0
        t = time.time()
        while app.live.thread is not None and app.live.thread.is_alive() and time.time() - t < 15:
            time.sleep(0.2)
        self.assertFalse(app.live.thread is not None and app.live.thread.is_alive())

    def test_second_control_reads_but_does_not_record(self):
        a, _p = self.make(http=False)
        a.start_telemetry()
        b, port_b = self.make()
        st = b.start_telemetry()
        self.assertFalse(st.is_writer)
        self.assertFalse(b.live.background)
        self.drive(a)
        code, d = jreq(port_b, "/api/telemetry/series?kind=card&since=%d" % (time.time() - 3600))
        self.assertEqual(code, 200)
        self.assertEqual(len(d["series"]), 2)
        self.assertFalse(jreq(port_b, "/api/telemetry")[1]["recording"])

    def test_bare_process_is_filed_under_its_port(self):
        app, port = self.make(key="p:4242")
        app.start_telemetry()
        self.drive(app, n=2)
        keys = [x["key"] for x in app.telemetry.list_series("server")]
        self.assertEqual(keys, ["p@%d" % self.eport])

    def test_metrics_is_off_until_asked_then_relabels_engine_counters(self):
        app, port = self.make()
        app.start_telemetry()
        ScriptedEngine.state["metrics"] = {"tokens_predicted_total": 1000, "tokens_predicted_seconds_total": 30.0,
                                           "prompt_tokens_total": 5000, "prompt_seconds_total": 6.0}
        self.drive(app, n=2)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen("http://127.0.0.1:%d/metrics" % port, timeout=10)
        self.assertEqual(cm.exception.code, 404)
        jreq(port, "/api/telemetry/settings", "POST", {"prometheus": True})
        with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % port, timeout=10) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/plain"))
            txt = r.read().decode()
        self.assertIn('pxa_card_memory_used_mib{card="0"', txt)
        self.assertIn('pxa_server_kv_used_ratio{server="m:main",name="Server 1"} 0.25', txt)
        self.assertIn('llamacpp:tokens_predicted_total{server="m:main"} 1000', txt)
        self.assertIn("pxa_server_requests_total", txt)

    def test_metrics_on_a_lan_bind_needs_the_token_as_bearer(self):
        os.environ["PXA_CONTROL_METRICS"] = "1"
        app, port = self.make(lan=True, token="s3cret-token-0123456789")
        app.start_telemetry()
        self.drive(app, n=2)
        code, body = raw_get(port, "/metrics")
        self.assertEqual(code, 401)
        self.assertIn(b"Bearer", body)
        code, body = raw_get(port, "/metrics", {"Authorization": "Bearer s3cret-token-0123456789"})
        self.assertEqual(code, 200)
        self.assertIn(b"pxa_card_", body)
        self.assertEqual(raw_get(port, "/metrics", {"Authorization": "Bearer wrong-token-0123456789"})[0], 401)


def jreq(port, path, method="GET", body=None, headers=None):
    code, _h, raw = req(port, path, method, body, headers)
    try:
        return code, json.loads(raw.decode() or "null")
    except ValueError:
        return code, raw


def glob_db(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.startswith("telemetry.db")] if os.path.isdir(d) else []


def raw_get(port, path, headers=None):
    rq = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), headers=dict(headers or {}, Host="127.0.0.1:%d" % port))
    try:
        with urllib.request.urlopen(rq, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---------------------------------------------------------------------------------------------
# PXA Control opens by itself (2026-10-05): the on/off rule, reuse of a running Control, the port
# fallback, adoption of a server the launcher started (and Stop), and the idle exit of a Control
# that started in the background. Real processes (sleeping children) and loopback sockets only.
# ---------------------------------------------------------------------------------------------
def sleeper(*args, secs=60):
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({secs})"] + list(args))
    # under load /proc/<pid>/cmdline can still show the parent (pre-exec) for a moment: wait for our argv
    want = (list(args) or [f"time.sleep({secs})"])[-1].encode()
    end = time.time() + 10
    while time.time() < end:
        try:
            with open(f"/proc/{p.pid}/cmdline", "rb") as f:
                if want in f.read():
                    break
        except OSError:
            break
        time.sleep(0.02)
    return p


def wait_until(fn, timeout=15.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(step)
    return fn()


class ControlFirst(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.kids, self.srvs = [], []

    def tearDown(self):
        for k in self.kids:
            if k.poll() is None:
                k.kill()
                k.wait()
        for srv in self.srvs:
            srv.shutdown()
            srv.server_close()
        for rec in C.running_controls():            # a background Control a failed test left behind
            if rec["pid"] != os.getpid():
                try:
                    os.kill(rec["pid"], 15)
                except OSError:
                    pass

    def kid(self, *args, secs=60):
        k = sleeper(*args, secs=secs)
        self.kids.append(k)
        return k

    def up(self, lan=False):
        """an in-process Control with its run record, as a running `pxa` would leave it."""
        app = C.App(L, port=0, lan=lan, token="t" * 20 if lan else None, models_dirs=[MODELS])
        srv, port = serve_app(app)
        self.srvs.append(srv)
        app.port = port
        C.write_run_record(port, lan, companion=False)
        return app, port

    def ns(self, *argv):
        return L.build_parser().parse_args(list(argv))

    # ---- the rule --------------------------------------------------------------------------
    def test_autostart_rule(self):
        d = C.autostart_decision
        self.assertEqual(d(False, {}, tty=True, container=None), (True, "a terminal"))
        self.assertFalse(d(False, {}, tty=False, container=None)[0])            # service, script, pipe
        self.assertIn("PXA_CONTROL=1", d(False, {}, tty=False, container=None)[1])
        self.assertFalse(d(False, {}, tty=True, container="docker")[0])         # container: opt-in only
        self.assertIn("-p 7777:7777", d(False, {}, tty=True, container="docker")[1])
        for v in ("1", "on", "yes", "true", "ON"):
            self.assertTrue(d(False, {"PXA_CONTROL": v}, tty=False, container="docker")[0], v)
        for v in ("0", "off", "no", "false"):
            self.assertEqual(d(False, {"PXA_CONTROL": v}, tty=True, container=None), (False, None), v)
        self.assertEqual(d(True, {"PXA_CONTROL": "1"}, tty=True, container=None), (False, None))   # --no-control wins
        self.assertFalse(d(False, {"PXA_CONTROL": "1", "PXA_CONTROL_SPAWNED": "1"}, tty=True)[0])  # never a Control in a Control
        self.assertEqual(C.control_env_mode({"PXA_CONTROL": "auto"}), "auto")

    def test_bare_invocation(self):
        ap = L.build_parser()
        bare = lambda *argv: L.bare_invocation(ap, ap.parse_args(list(argv)))      # noqa: E731
        self.assertTrue(bare())
        self.assertTrue(bare("--models-dir", MODELS, "--no-browser", "--lan", "--control-port", "7800"))
        for argv in (["-m", FAKE_MODEL], ["--gpus", "0"], ["--workload", "chat"], ["--doctor"], ["--explain"],
                     ["--no-control"], ["--no-tui"], ["--port", "8081"], ["--yes"], ["--selftest"]):
            self.assertFalse(bare(*argv), argv)

    def test_launcher_decision_and_off_paths(self):
        a = self.ns("--no-control")
        self.assertEqual(L.control_decision(a, tty=True), (False, None))
        a = self.ns("-m", FAKE_MODEL, "--gpus", "0")
        on, why = L.control_decision(a, tty=False)
        self.assertFalse(on)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIsNone(L.control_alongside(a, "0", tty=False))
        self.assertIn("PXA Control: off (not started from a terminal", buf.getvalue())
        self.assertEqual(C.launched_servers(), {})                              # nothing recorded when off
        os.environ["PXA_CONTROL"] = "0"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertIsNone(L.control_alongside(a, "0", tty=True))
            self.assertEqual(buf.getvalue(), "")                                # your own off switch: silent
        finally:
            os.environ.pop("PXA_CONTROL", None)

    # ---- records ---------------------------------------------------------------------------
    def test_run_records_and_reuse(self):
        self.assertIsNone(C.find_running())
        app, port = self.up()
        r = C.find_running()
        self.assertEqual((r["port"], r["pid"], r["url"]), (port, os.getpid(), f"http://127.0.0.1:{port}/"))
        self.assertIsNone(C.find_running(port=port + 1))                        # asked for another port
        self.assertIsNone(C.find_running(lan=True))                             # a LAN one was asked for
        # a record whose process is gone is dropped; one whose port is not a Control is ignored
        dead = self.kid(secs=0)
        dead.wait()
        rd = os.path.join(C.config_dir(), "run")
        with open(os.path.join(rd, f"control-{dead.pid}.json"), "w") as f:
            json.dump({"pid": dead.pid, "start": 1, "port": port, "lan": False}, f)
        stub = http.server.HTTPServer(("127.0.0.1", 0), StubEngine)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        try:
            other = self.kid()
            with open(os.path.join(rd, f"control-{other.pid}.json"), "w") as f:
                json.dump({"pid": other.pid, "start": C.proc_start_ticks(other.pid), "port": stub.server_address[1],
                           "lan": False, "companion": False, "since": 0}, f)
            self.assertEqual(C.find_running()["port"], port)
        finally:
            stub.shutdown()
            stub.server_close()
        self.assertFalse(os.path.exists(os.path.join(rd, f"control-{dead.pid}.json")))

    def test_lan_control_is_found_with_its_token(self):
        app, port = self.up(lan=True)
        os.environ["PXA_CONTROL_TOKEN"] = "t" * 20
        try:
            r = C.find_running(lan=True)
            self.assertEqual(r["url"], f"http://127.0.0.1:{port}/?token=" + "t" * 20)
        finally:
            os.environ.pop("PXA_CONTROL_TOKEN", None)

    def test_launched_records_need_the_same_process(self):
        k = self.kid()
        C.record_launched(k.pid, port=18123, model=FAKE_MODEL, gpus="0,1", by="pxa-launch")
        rec = C.launched_servers()[k.pid]
        self.assertEqual((rec["port"], rec["gpus"], rec["by"]), (18123, [0, 1], "pxa-launch"))
        self.assertEqual(stat_mode(os.path.join(C.config_dir(), "launched")), 0o700)
        self.assertEqual(stat_mode(os.path.join(C.config_dir(), "launched", f"{k.pid}.json")), 0o600)
        # the same pid with another start time is another process (a recycled pid): never matched
        p = os.path.join(C.config_dir(), "launched", f"{k.pid}.json")
        with open(p) as f:
            d = json.load(f)
        d["start"] = (d["start"] or 0) + 12345
        with open(p, "w") as f:
            json.dump(d, f)
        self.assertNotIn(k.pid, C.launched_servers())
        self.assertFalse(os.path.exists(p))
        C.record_launched(k.pid, port=18123)
        k.kill()
        k.wait()
        self.assertNotIn(k.pid, C.launched_servers())                           # gone: dropped

    # ---- adoption + Stop --------------------------------------------------------------------
    def test_fleet_adopts_the_launched_server_and_stops_it(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        k = self.kid("--port", "18124", "-m", FAKE_MODEL)
        C.record_launched(k.pid, port=18124, model=FAKE_MODEL, gpus="1", by="run-server.sh")
        x = [i for i in app.fleet(max_age=0)["instances"] if i["key"] == f"p:{k.pid}"]
        self.assertEqual(len(x), 1, "listed even with discovery off")
        x = x[0]
        self.assertEqual((x["kind"], x["origin"], x["adopted"], x["control"], x["port"], x["gpus"], x["model_file"]),
                         ("process", "launcher", True, True, 18124, [1], "tiny.gguf"))
        self.assertEqual(x["actions"], ["log", "stop"])
        self.assertIn("run-server.sh", x["label"])
        with self.assertRaises(C.Invalid):
            app.external_control(x["key"], "restart", "stop")                  # Stop only
        with self.assertRaises(C.Invalid):
            app.external_control(x["key"], "stop", "yes")                      # the typed word
        # a Control the launcher started ALSO watches it on the Live tab (same fleet)
        self.assertIn(x["key"], {i["key"] for i in app.fleet(max_age=0)["instances"] if i.get("running") and i.get("port")})
        r = app.external_control(x["key"], "stop", "stop")
        self.assertTrue(r["ok"])
        self.assertTrue(wait_until(lambda: k.poll() is not None, 10))
        self.assertNotIn(k.pid, C.launched_servers())
        self.assertNotIn(f"p:{k.pid}", {i["key"] for i in app.fleet(max_age=0)["instances"]})

    def test_a_new_launched_server_skips_the_fleet_cache(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        self.assertEqual([i for i in app.fleet(max_age=60)["instances"] if i["kind"] == "process"], [])
        k = self.kid()
        C.record_launched(k.pid, port=18130)
        self.assertIn(f"p:{k.pid}", [i["key"] for i in app.fleet(max_age=60)["instances"]])   # not the 60 s cache

    def test_stop_route_through_http(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        srv, port = serve_app(app)
        self.srvs.append(srv)
        k = self.kid()
        C.record_launched(k.pid, port=18125, model=FAKE_MODEL)
        st, _h, b = req(port, "/api/fleet")
        self.assertEqual(st, 200)
        self.assertIn(f"p:{k.pid}", [i["key"] for i in json.loads(b)["instances"]])
        st, _h, b = req(port, "/api/fleet/control", "POST", {"key": f"p:{k.pid}", "action": "stop", "confirm": "stop"})
        self.assertEqual(st, 200, b)
        self.assertTrue(wait_until(lambda: k.poll() is not None, 10))
        st, _h, b = req(port, "/api/info")
        self.assertFalse(json.loads(b)["companion"])

    # ---- the idle rule ----------------------------------------------------------------------
    def test_idle_clock(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        now = time.time()
        app.last_request = app.last_busy = now - 1000
        self.assertGreaterEqual(app.idle_seconds(now), 999)
        k = self.kid()
        C.record_launched(k.pid, port=18126)
        self.assertEqual(app.idle_seconds(now), 0.0)                            # a launched server runs
        self.assertEqual(app.last_busy, now)
        k.kill()
        k.wait()
        self.assertAlmostEqual(app.idle_seconds(now + 30), 30, delta=1)
        app.last_request = now + 20                                              # a page asked something
        self.assertAlmostEqual(app.idle_seconds(now + 30), 10, delta=1)
        seat = fake_running_seat(app, "main", 18127, [0])
        try:
            self.assertEqual(app.idle_seconds(now + 30), 0.0)                   # a server it started runs
        finally:
            seat.proc.kill()
            seat.proc.wait()

    def test_requests_reset_the_clock(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        srv, port = serve_app(app)
        self.srvs.append(srv)
        app.last_request = 0.0
        req(port, "/api/info")
        self.assertGreater(app.last_request, time.time() - 5)

    def test_companion_closes_itself_when_idle(self):
        out = {}
        port = C.free_port(17950)

        def run():
            out["rc"] = C.serve(L, port=port, open_browser=False, models_dirs=[MODELS], companion=True, idle_s=0.8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            th = threading.Thread(target=run, daemon=True)
            th.start()
            self.assertTrue(wait_until(lambda: C.control_answers(port), 10))
            st, _h, b = req(port, "/api/info")
            info = json.loads(b)
            self.assertEqual((info["companion"], info["idle_exit_s"]), (True, 0.8))
            th.join(20)
        self.assertFalse(th.is_alive(), "a background Control with nothing to do must exit")
        self.assertEqual(out.get("rc"), 0)
        self.assertIn("closing", buf.getvalue())
        self.assertEqual(C.running_controls(), [])                              # its run record is gone

    def test_user_opened_control_does_not_idle_out(self):
        port = C.free_port(17960)
        holder = {}

        def run():
            holder["rc"] = C.serve(L, port=port, open_browser=False, models_dirs=[MODELS], idle_s=0.3)
        with contextlib.redirect_stdout(io.StringIO()):
            th = threading.Thread(target=run, daemon=True)
            th.start()
            self.assertTrue(wait_until(lambda: C.control_answers(port), 10))
            time.sleep(1.5)
            self.assertTrue(th.is_alive())                                       # `pxa` / --gui: until Ctrl-C
            rec = C.find_running(port=port)
            self.assertFalse(rec["companion"])
            C.SERVING[port].shutdown()                                           # what Ctrl-C ends in
            th.join(10)
        self.assertEqual(holder.get("rc"), 0)
        self.assertEqual(C.running_controls(), [])

    # ---- start or reuse ---------------------------------------------------------------------
    def test_ensure_control_reuses_the_running_one(self):
        app, port = self.up()
        r = C.ensure_control("/nonexistent/pxa-launch.py", port=port + 1)
        self.assertEqual((r.get("reused"), r.get("port")), (True, port))

    def test_ensure_control_starts_one_in_the_background_on_the_next_free_port(self):
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        want = busy.getsockname()[1]
        k = self.kid()
        C.record_launched(k.pid, port=18128, model=FAKE_MODEL)
        os.environ["PXA_CONTROL_IDLE_S"] = "1"
        try:
            r = C.ensure_control(os.path.join(TOOLS, "pxa-launch.py"), port=want, avoid={want + 1})
        finally:
            os.environ.pop("PXA_CONTROL_IDLE_S", None)
            busy.close()
        self.assertFalse(r.get("error"), r)
        self.assertFalse(r["reused"])
        self.assertNotIn(r["port"], (want, want + 1))                           # busy, and the server's own
        self.assertEqual(r["url"], f"http://127.0.0.1:{r['port']}/")
        rec = C.find_running()
        self.assertTrue(rec["companion"])
        self.assertNotEqual(rec["pid"], os.getpid())
        st, _h, b = req(r["port"], "/api/fleet")
        keys = [i["key"] for i in json.loads(b)["instances"]]
        self.assertIn(f"p:{k.pid}", keys)                                       # adopted at once
        # a second launch reuses it instead of starting another
        r2 = C.ensure_control(os.path.join(TOOLS, "pxa-launch.py"), port=want)
        self.assertEqual((r2["reused"], r2["port"]), (True, r["port"]))
        # the server goes; nobody is looking; the background Control closes itself
        k.kill()
        k.wait()
        self.assertTrue(wait_until(lambda: C.running_controls() == [], 30), "background Control did not exit")
        with open(C.control_log_path()) as f:
            self.assertIn("closing", f.read())

    def test_run_control_reuse_and_port_fallback(self):
        app, port = self.up()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = L.run_control(self.ns("--no-browser"), [], L=L)
        self.assertEqual(rc, 0)
        self.assertIn(f"PXA Control: http://127.0.0.1:{port}/", buf.getvalue())
        self.assertIn("already running", buf.getvalue())
        # --gui --port N names another port: that one is not reused (a second Control is asked for)
        self.assertIsNone(C.find_running(port=port + 1))
        # nothing running: the default port when free, else the next free one; --control-port is taken as given
        shutil.rmtree(os.path.join(C.config_dir(), "run"), ignore_errors=True)
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        busy_port = busy.getsockname()[1]
        got, saved = [], (C.serve, C.DEFAULT_PORT)
        C.serve = lambda L_, **kw: got.append(kw) or 0
        C.DEFAULT_PORT = busy_port
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(L.run_control(self.ns("--no-browser"), [], L=L), 0)
                self.assertEqual(L.run_control(self.ns("--gui", "--control-port", "17971"), [], L=L), 0)
                self.assertEqual(L.run_control(self.ns("--gui", "--port", "17972"), ["--gui", "--port", "17972"], L=L), 0)
        finally:
            C.serve, C.DEFAULT_PORT = saved
            busy.close()
        self.assertGreater(got[0]["port"], busy_port)                           # 7777 taken -> the next free one
        self.assertEqual([g["port"] for g in got[1:]], [17971, 17972])
        self.assertFalse(got[0]["lan"])
        self.assertFalse(got[0].get("companion", False))

    def test_control_for_pid_reads_the_server_line(self):
        app, port = self.up()                                                   # reused: nothing is spawned
        k = self.kid("--port", "18129", "-m", FAKE_MODEL)
        os.environ["PXA_CONTROL"] = "1"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(L.control_for_pid(self.ns("--control-for-pid", str(k.pid))), 0)
                rec = C.launched_servers()[k.pid]
                self.assertEqual((rec["port"], rec["model"], rec["by"]), (18129, FAKE_MODEL, "run-server.sh"))
                self.assertIn(f"PXA Control: http://127.0.0.1:{port}/", buf.getvalue())
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):                          # the entrypoint did it already
                    self.assertEqual(L.control_for_pid(self.ns("--control-for-pid", str(k.pid))), 0)
                self.assertEqual(buf2.getvalue(), "")
        finally:
            os.environ.pop("PXA_CONTROL", None)

    def test_control_model_dirs_seed_only_an_empty_library(self):
        a = self.ns("--models-dir", MODELS)
        self.assertEqual(L.control_model_dirs(a, C, FAKE_MODEL), [MODELS])
        other = os.path.join(TMP, "other-models")
        os.makedirs(other, exist_ok=True)
        m2 = os.path.join(other, "x.gguf")
        open(m2, "wb").close()
        self.assertEqual(L.control_model_dirs(self.ns(), C, m2), [other])       # first time: where the model is
        C.save_config(dict(C.load_config(), model_dirs=[MODELS]))
        self.assertEqual(L.control_model_dirs(self.ns(), C, m2), [])            # a library exists: left alone

    def test_bare_launcher_without_a_terminal_opens_nothing(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(TOOLS, "pxa-launch.py")], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)                   # the old answer: no model
        self.assertIn("no model", r.stderr)
        self.assertNotIn("PXA Control:", r.stdout)
        self.assertEqual(C.running_controls(), [])

    def test_page_offers_stop_for_command_line_servers(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html")) as f:
            page = f.read()
        self.assertIn('x.origin === "launcher"', page)
        self.assertIn('confirm: "stop"', page)
        self.assertIn("S.info.companion", page)


def load_cfg_model_dirs():
    with open(C.config_path()) as f:
        return json.load(f).get("model_dirs", []) + [os.environ.get("PXA_MODELS_DIR", "")]


# ---------------------------------------------------------------------------------------------
# PXA v3 launcher fixes (2026-10-05): the expert-cache route, measured PXQN files,
# no spurious --accept-unmeasured on GGUF plans, the Control version text.
# ---------------------------------------------------------------------------------------------
GIB = 1 << 30


def fake_gguf(path, arch="qwen35moe", n_expert=128, size_gib=20.0, ple_gib=0.0, mtp=False,
              types=(1,), kvs=()):
    """A SPARSE synthetic GGUF: a real header and tensor directory, whose last tensor (the routed
    experts, or a dense FFN when n_expert=0) runs to a sparse end of file, so it can be 20+ GiB on
    disk without using the space. Tensor bytes come from the directory's offsets, as in a real file."""
    import struct

    def s_(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, arch)] + list(kvs)
    if n_expert:
        kv.append((f"{arch}.expert_count", 4, n_expert))
    tensors = []                                          # (name, type, offset)
    off = 0
    small = 64 << 20                                      # 64 MiB of non-expert weights
    for i, t in enumerate(types):
        tensors.append((f"blk.{i}.attn_q.weight", t, off)); off += small
    if ple_gib:
        tensors.append(("per_layer_token_embd.weight", 8, off)); off += int(ple_gib * GIB)
    if mtp:
        tensors.append(("blk.1.nextn.eh_proj.weight", 8, off)); off += small
    tensors.append(("blk.0.ffn_gate_exps.weight" if n_expert else "blk.0.ffn_down.weight", types[0], off))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for k, t, v in kv:
            f.write(s_(k) + struct.pack("<I", t) + (s_(v) if t == 8 else struct.pack("<I", v)))
        for nm, t, o in tensors:
            ne = 1 << 20 if not nm.startswith("per_layer") else int(ple_gib * GIB / 1.0625)
            f.write(s_(nm) + struct.pack("<I", 1) + struct.pack("<Q", ne) + struct.pack("<I", t) + struct.pack("<Q", o))
        f.write(b"\0" * (32 - f.tell() % 32))
        f.truncate(f.tell() + off + int(size_gib * GIB))     # every tensor above, then the last one
    return path


def meminfo(path, total_gib, avail_gib):
    with open(path, "w") as f:
        f.write(f"MemTotal: {int(total_gib * 1048576)} kB\nMemAvailable: {int(avail_gib * 1048576)} kB\n")
    return path


class V3Plans(unittest.TestCase):
    P100 = (0, "Tesla P100-PCIE-16GB", 60, 16384, 0, "GPU-t-0")
    V100 = (0, "Tesla V100-PCIE-16GB", 70, 16384, 0, "GPU-t-0")

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="v3plans-", dir=TMP)
        cls.moe = fake_gguf(os.path.join(cls.td, "moe.gguf"), size_gib=24.0, ple_gib=0.5, mtp=True)
        cls.dense = fake_gguf(os.path.join(cls.td, "dense.gguf"), arch="llama", n_expert=0, size_gib=24.0)
        # a MoE whose NON-expert part alone is bigger than one card: experts are a small slice
        cls.fat = fake_gguf(os.path.join(cls.td, "fat.gguf"), size_gib=2.0, types=(1,) * 300)
        cls.ram_big = meminfo(os.path.join(cls.td, "mem128"), 128, 120)
        cls.ram_small = meminfo(os.path.join(cls.td, "mem24"), 24, 22)
        cls.ram_busy = meminfo(os.path.join(cls.td, "mem64busy"), 64, 12)

    def plan(self, model, gpus, mem, extra=()):
        env = {"PXA_MEMINFO": mem, "PXA_LAUNCH_TABLES_ONLY": "1"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            a = L.build_parser().parse_args(["--gpus", ",".join(str(g[0]) for g in gpus), "--model", model,
                                             "--yes"] + list(extra))
            a.explain = True
            return L._Capture().run(L.plan_and_build, a, list(gpus))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_profile_reads_expert_bytes_from_offsets(self):
        p = L.model_profile(self.moe, "gguf")
        self.assertTrue(p["is_moe"])
        self.assertGreater(p["expert_bytes"], 23 * GIB)
        self.assertTrue(p["typemap_fp"] and len(p["typemap_fp"]) == 16)

    def test_moe_bigger_than_card_with_ram_goes_expert_cache(self):
        cap = self.plan(self.moe, [self.P100], self.ram_big)
        self.assertEqual(cap.code, 0, cap.text[-3000:])
        plan, cmd, env, cv, prof, ctx = cap.value
        self.assertEqual(plan.xcache["verdict"], "ok")
        self.assertNotIn("R-17B", cap.text)
        self.assertIn("EXPERT CACHE", cap.text)
        self.assertIn("bigger than your card", plan.notes[0])
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "999")              # every layer asked for: the engine plans the cache
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "none")      # MTP head in the file: kept unloaded
        self.assertEqual(cmd[cmd.index("--cache-ram") + 1], "0")
        self.assertEqual(cmd[cmd.index("-sm") + 1], "layer")
        self.assertIn(r"per_layer_token_embd\.weight=CPU", cmd)          # the PLE table on the CPU

    def test_expert_cache_keeps_a_users_spec_and_partial_ngl_is_not_the_route(self):
        cap = self.plan(self.moe, [self.P100], self.ram_big, ["--spec", "ngram-mod:n_max=4,n_min=2"])
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertEqual(cap.value[1][cap.value[1].index("--spec-type") + 1], "ngram-mod:n_max=4,n_min=2")
        cap = self.plan(self.moe, [self.P100], self.ram_big, ["--ngl", "20"])
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertIsNone(cap.value[0].xcache)
        self.assertNotIn("--cache-ram", cap.value[1])

    def test_expert_cache_on_two_cards_is_a_layer_split(self):
        p2 = (1, "Tesla P100-PCIE-16GB", 60, 16384, 0, "GPU-t-1")
        big = fake_gguf(os.path.join(self.td, "moe40.gguf"), size_gib=40.0)
        cap = self.plan(big, [self.P100, p2], self.ram_big)
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertEqual(cap.value[1][cap.value[1].index("-sm") + 1], "layer")
        self.assertIn("bigger than your cards", cap.text)

    def test_moe_without_enough_ram_is_refused_in_plain_words(self):
        cap = self.plan(self.moe, [self.P100], self.ram_small)
        self.assertEqual(cap.code, 2)
        self.assertIn("[R-17C]", cap.text)
        self.assertRegex(cap.text, r"needs about \d+ GB of system RAM and this machine has 24 GB")

    def test_busy_ram_warns_without_refusing(self):
        cap = self.plan(self.moe, [self.P100], self.ram_busy)
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertIn("RIGHT NOW only 12 GB of RAM is free", cap.text)

    def test_dense_bigger_than_card_is_still_refused(self):
        cap = self.plan(self.dense, [self.P100], self.ram_big)
        self.assertEqual(cap.code, 2)
        self.assertIn("[R-17B]", cap.text)
        self.assertNotIn("EXPERT CACHE", cap.text)

    def test_moe_whose_non_expert_part_does_not_fit_is_refused(self):
        # 300 x 64 MiB = 18.75 GiB of non-expert weights on a 16 GiB card
        cap = self.plan(self.fat, [self.P100], self.ram_big)
        self.assertEqual(cap.code, 2)
        self.assertIn("[R-17D]", cap.text)

    def test_xcache_plan_measured_cell_quotes_twenty_tps_only_there(self):
        prof = {"is_moe": True, "arch": "qwen4exp", "ple_bytes": 50 * GIB, "expert_bytes": 24 * GIB,
                "kv_bytes_tok": 0}
        mem = (128 * GIB, 120 * GIB)
        x = L.xcache_plan([self.P100], 77 * GIB, prof, 8192, True, mem=mem)
        self.assertEqual(x["verdict"], "ok")
        self.assertIn("expect about 20 t/s on one P100", x["lines"][0])
        x = L.xcache_plan([self.V100], 77 * GIB, prof, 8192, True, mem=mem)
        self.assertIn("not measured by us", x["lines"][0])
        self.assertIsNone(L.xcache_plan([self.P100], 77 * GIB, dict(prof, is_moe=False), 8192, True, mem=mem))
        self.assertIsNone(L.xcache_plan([self.P100], 60 * GIB, prof, 8192, True, mem=mem))   # GPU part fits

    def test_flash_next_one_p100_row_is_the_gate_line(self):
        sel = [self.P100]
        prof = {"arch": "qwen4exp", "is_moe": True, "n_expert": 512, "deltanet": True, "tier": "PXQ_UNIVERSAL",
                "xcache_route": True}
        r, st, _e, _n = L.recipe_for(sel, prof, "chat")
        self.assertEqual((r.key, st, r.b, r.ub, r.ctx, r.threads), ("1xp100-flashnext-xcache", "MEASURED", 2048, 2048, 8192, 16))
        # the 35B PXQU row was measured on a file that FITS one card: never handed to an expert-cache file
        prof35 = {"arch": "qwen35moe", "is_moe": True, "n_expert": 256, "tier": "PXQ_UNIVERSAL"}
        self.assertEqual(L.recipe_for([self.V100], prof35, "chat")[0].key, "1xv100-pxqu16")
        self.assertNotEqual(L.recipe_for([self.V100], dict(prof35, xcache_route=True), "chat")[1], "MEASURED")
        r4 = [r for r in L.RECIPES if r.key == "4xp100-flashnext"][0]
        self.assertEqual((r4.ts, r4.ctx, r4.np), ("0.95,1.01,1.01,1.03", 32768, 1))

    def test_pxqn_mixed_file_needs_no_ack_classic_universal_still_does(self):
        uni = (257, 259)                                   # PXQN3 + PXQN4 tensors: a mixed (UNIVERSAL) tier map
        pxqn = fake_gguf(os.path.join(self.td, "bal.gguf"), arch="qwen35", n_expert=0, size_gib=1.0, types=uni,
                         kvs=[("pxa.pxqn.encoder", 8, "pxqn-ladder-test"), ("pxa.pxqn.alloc", 8, "0123456789abcdef")])
        classic = fake_gguf(os.path.join(self.td, "uni.gguf"), arch="qwen35", n_expert=0, size_gib=1.0, types=(252, 255))
        for path, want_ack in ((pxqn, False), (classic, True)):
            prof = L.model_profile(path, "gguf")
            self.assertEqual(prof["tier"], "PXQ_UNIVERSAL")
            p = L.decide([self.V100], "gguf", path, None, prof, 1, "chat", set(), None, [])
            self.assertEqual(bool([x for x in p.needs_ack if x.startswith("PXQ_UNIVERSAL")]), want_ack, p.needs_ack)
        cap = self.plan(pxqn, [self.V100], self.ram_big)
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertIn("no release-gate result on record", cap.text)

    def test_onecard_cells_are_measured_by_identity_not_by_name(self):
        oc = {"typemap_fp": "050b79b16e836341", "pxqn_alloc": "ea306a5d31393142", "tier": "PXQ_UNIVERSAL",
              "pxqn_encoder": "pxqn-ladder-cc44672e5a"}
        for card, word in ((self.V100, "42.5"), (self.P100, "27.8")):
            st, text = L.measured_file_cell(oc, [card])
            self.assertEqual(st, "MEASURED")
            self.assertIn("12/12", text)
            self.assertIn("125,000", text)
            self.assertIn(word, text)
        # same tier map, another encode (a fine-tune): not claimed as gated
        st, text = L.measured_file_cell(dict(oc, pxqn_alloc="353c1391120bb694"), [self.V100])
        self.assertEqual(st, "NOTE")
        self.assertIn("not checked by us", text)
        # a card set the gate did not run: said, not refused
        st, text = L.measured_file_cell(oc, [(0, "x", 61, 11264, 0, "u")])
        self.assertEqual(st, "NOTE")
        # PXQN3bal is in the table too
        self.assertEqual(L.measured_file_cell({"typemap_fp": "579c89d9674e8ff8", "pxqn_alloc": "aabe9387a002b210"},
                                              [self.P100])[0], "MEASURED")

    @unittest.skipUnless(os.path.isfile(os.environ.get("PXA_TEST_ONECARD_GGUF", "")),
                         "the real one-card file is not on this machine")
    def test_real_onecard_file_plans_without_accept_unmeasured(self):
        path = os.environ.get("PXA_TEST_ONECARD_GGUF", "")
        self.assertEqual(L.model_profile(path, "gguf")["typemap_fp"], "050b79b16e836341")
        for card in (self.V100, self.P100):
            cap = self.plan(path, [card], self.ram_big)
            self.assertEqual(cap.code, 0, cap.text[-2000:])
            self.assertIn("MEASURED file: this is our Qwen3.8-27B one-card mix", cap.text)
            self.assertNotIn("--accept-unmeasured to proceed", cap.text)

    def test_gguf_plans_need_no_ack_for_card_count_or_unmeasured_sets(self):
        prof = {"tier": "PXQN4", "arch": "qwen35", "is_moe": False, "deltanet": True, "n_expert": 0}
        quad = [(i, "Tesla P100-PCIE-16GB", 60, 16384, 0, f"u{i}") for i in range(4)]
        mixed = [(0, "Tesla P100-PCIE-16GB", 60, 16384, 0, "u0"), (1, "Tesla V100-PCIE-16GB", 70, 16384, 0, "u1")]
        three = quad[:3]
        for sel in (quad, mixed, three):
            p = L.decide(sel, "gguf", "/x/m.gguf", None, prof, 1, "chat", set(), None, [])
            self.assertEqual((p.engine, p.needs_ack), ("llama", []), (len(sel), p.needs_ack))
        # the engine-choice table still asks when it CHOSE the engine (a converted vLLM dir, both eligible)
        pv = L.decide(quad, "vllm_dir", "/x/coder35", None, dict(prof, tier="PXQ4", is_moe=True, n_expert=256), 8,
                      "serve", {60}, "pxa-sm60-dev", [])
        self.assertTrue(pv.engine_from_table)
        self.assertTrue(any("table is 2-card only" in x for x in pv.needs_ack))

    def test_control_fit_badge_says_ram_for_the_expert_cache(self):
        e = {"size": 24 * GIB, "n_expert": 128, "expert_bytes": 23 * GIB, "ple_bytes": 0, "kv_bytes_tok": 0}
        old = os.environ.get("PXA_MEMINFO")
        os.environ["PXA_MEMINFO"] = self.ram_big
        try:
            app = C.App(L, port=7777, models_dirs=[MODELS])
            self.assertEqual(app.fits(e, [self.P100])["verdict"], "ram")
            self.assertEqual(app.fits(dict(e, n_expert=0, expert_bytes=0), [self.P100])["verdict"], "no")
            os.environ["PXA_MEMINFO"] = self.ram_small
            self.assertEqual(app.fits(e, [self.P100])["verdict"], "no")
        finally:
            os.environ.pop("PXA_MEMINFO", None) if old is None else os.environ.__setitem__("PXA_MEMINFO", old)

    def test_control_version_comes_from_the_package(self):
        with open(os.path.join(TOOLS, "pxa_control.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn('CONTROL_VERSION = "v3"', src)
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            tok = C.version_token(f.read())
        self.assertEqual(tok, "v3.1")
        self.assertEqual(C.version_token("tag:              v3.1\ncommit: abc\n"), "v3.1")
        self.assertEqual(C.version_token("v3.1.4\n"), "v3.1.4")
        self.assertEqual(C.version_token(""), "")
        self.assertEqual(C.CONTROL_VERSION, tok)
        d = tempfile.mkdtemp(dir=TMP)
        self.assertEqual(C.version_from_dir(d), "")
        os.makedirs(os.path.join(d, "pxa-v3.1.4"))
        os.symlink("pxa-v3.1.4", os.path.join(d, "current"))
        self.assertEqual(C.version_from_dir(d), "v3.1.4")
        with open(os.path.join(d, "VERSION"), "w", encoding="utf-8") as f:
            f.write("tag: v9.9\n")
        self.assertEqual(C.version_from_dir(d), "v9.9")
        with open(os.path.join(d, "pxa-v3.1.4", "VERSION"), "w", encoding="utf-8") as f:
            f.write("tag: v8.8\n")
        self.assertEqual(C.version_from_dir(d), "v8.8")
        app = C.App(L, port=7777, models_dirs=[MODELS])
        srv, port = serve_app(app)
        try:
            info = json.loads(req(port, "/api/info")[2])
            self.assertEqual(info["control_version"], tok)
        finally:
            srv.shutdown()
            srv.server_close()


class QuantizerTag(unittest.TestCase):
    """_extra_facts exposes the pxa.quantizer.* KVs (prefix stripped) or None."""

    def _facts(self, kv):
        d = tempfile.mkdtemp(dir=TMP)
        p = os.path.join(d, "m.gguf")
        open(p, "wb").write(b"x")
        fake = type("A", (), {})()
        fake._prof_cache = {}
        fake.L = type("LL", (), {"gguf_header": staticmethod(lambda path: {"kv": kv, "tensors": []}),
                                  "PXQ_GGML_TYPE": {}, "NON_PXQ_GGML_TYPE": {}})
        return C.App._extra_facts(fake, p)

    def test_tagged_and_untagged(self):
        q = self._facts({"general.name": "n", "pxa.quantizer.edition": "pro", "pxa.quantizer.licensee": "T",
                         "pxa.quantizer.license_expires": 1791244800})["quantizer"]
        self.assertEqual(q, {"edition": "pro", "licensee": "T", "license_expires": 1791244800})
        self.assertIsNone(self._facts({"general.name": "n"})["quantizer"])


def urllib_quote(p):
    import urllib.parse
    return urllib.parse.quote(p, safe="")


def stat_mode(p):
    import stat as _st
    return _st.S_IMODE(os.stat(p).st_mode)


class SymlinkedModels(unittest.TestCase):
    """A model offloaded to another disk and linked back into a model folder can be planned and
    started; a link to anything that is not a regular GGUF, or a path that walks out, cannot."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pxa-models-root-")
        self.outside = tempfile.mkdtemp(prefix="pxa-models-outside-")
        def w(name, data):
            p = os.path.join(self.outside, name)
            with open(p, "wb") as f:
                f.write(data)
            return p
        self.real = w("real.gguf", b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)
        notg = w("secret.gguf", b"not a gguf at all")
        plain = w("passwd", b"root:x:0:0::/root:/bin/sh\n")
        os.symlink(self.real, os.path.join(self.root, "linked.gguf"))
        os.symlink(notg, os.path.join(self.root, "fake.gguf"))
        os.symlink(plain, os.path.join(self.root, "plain.gguf"))
        os.symlink(os.path.join(self.outside, "gone.gguf"), os.path.join(self.root, "dangling.gguf"))
        os.symlink(self.outside, os.path.join(self.root, "offload"))        # a linked folder
        os.symlink(self.outside, os.path.join(self.root, "offload.gguf"))   # a directory named .gguf

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.outside, ignore_errors=True)

    def check(self, model):
        return C.validate_launch({"gpus": [0], "model": model}, {0, 1}, CATALOG, [self.root], resolve_port=False)

    def test_link_to_a_gguf_outside_the_folders_is_allowed(self):
        for m in (os.path.join(self.root, "linked.gguf"), os.path.join(self.root, "offload", "real.gguf")):
            self.assertEqual(self.check(m)["model"], m)
            self.assertTrue(C.model_path_ok(m, [self.root]))

    def test_everything_else_outside_is_refused(self):
        bad = [os.path.join(self.root, "fake.gguf"),             # a link to a file without the GGUF magic
               os.path.join(self.root, "plain.gguf"),            # a link to a text file
               os.path.join(self.root, "dangling.gguf"),         # a link to nothing
               os.path.join(self.root, "offload.gguf"),          # a link to a directory
               self.real,                                        # the target itself, named directly
               os.path.join(self.root, "..", os.path.basename(self.outside), "real.gguf")]   # walking out
        for m in bad:
            with self.assertRaises(C.Invalid, msg=m):
                self.check(m)
        self.assertFalse(C.model_path_ok(self.real, [self.root]))

    def test_api_thinking_follows_the_same_rule(self):
        app = C.App(L, port=7777)
        app.model_roots = lambda: [self.root]
        prof, _st, src, model = app.thinking_target(urllib.parse.parse_qs("model=" + urllib.parse.quote(os.path.join(self.root, "linked.gguf"))))
        self.assertEqual(model, os.path.join(self.root, "linked.gguf"))
        with self.assertRaises(C.Invalid):
            app.thinking_target(urllib.parse.parse_qs("model=" + urllib.parse.quote(os.path.join(self.root, "fake.gguf"))))


class LaunchCtxAndLog(unittest.TestCase):
    """The plan header's context is the -c on the command, and the log pane loads the backlog."""

    def _args(self, ctx):
        return type("A", (), {
            "ctx": ctx, "sm": "layer", "model": FAKE_MODEL, "host": "127.0.0.1", "port": 9,
            "ngl": 99, "np": 1, "ctk": "f16", "ctv": "f16", "threads": 2, "emit_threads": False,
            "workload": "chat", "ub": 0, "b": 0, "ts": "", "spec": None, "draft_model": None,
            "hot_model": [], "no_mmap": False,
        })()

    def _plan(self, picked):
        plan = L.Plan()
        plan.fa = "on"
        plan.engine_ac = {"picks": {"c": {"value": str(picked), "status": "measured", "why": "test cell"}}}
        return plan

    def test_auto_ctx_returned_is_the_command_c(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd, env, ctx = L.build_llama_cmd(self._plan(32768), self._args(0),
                                              [(0, "Tesla P100", 60, 16384, 0, "u")], {}, 4096, ["n/a"], None,
                                              explain=True)
        self.assertEqual(cmd[cmd.index("-c") + 1], "32768")
        self.assertEqual(ctx, 32768)
        self.assertIn("was -c 4096", buf.getvalue())
        with open(os.path.join(TOOLS, "pxa-launch.py"), encoding="utf-8") as f:
            self.assertIn("cmd, env, ctx = build_llama_cmd", f.read())

    def test_a_hand_set_ctx_is_not_replaced(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd, env, ctx = L.build_llama_cmd(self._plan(32768), self._args(16384),
                                              [(0, "Tesla P100", 60, 16384, 0, "u")], {}, 16384, ["n/a"], None,
                                              explain=True)
        self.assertEqual((cmd[cmd.index("-c") + 1], ctx), ("16384", 16384))
        self.assertNotIn("from the engine's registry", buf.getvalue())

    def test_plan_dict_says_auto_only_when_the_request_left_it(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        plan = L.Plan()
        plan.engine = "llama"

        class Cap(object):
            code = 0
            text = "plan"
            thinking = None
        cap = Cap()
        cap.args = type("A", (), {"sm": "layer", "np": 1, "workload": "chat"})()
        cap.value = (plan, ["llama-server", "-m", FAKE_MODEL, "-c", "32768"], {}, "0", {}, 32768)
        auto = app.validate({"gpus": [0], "model": FAKE_MODEL, "ctx": 0}, resolve_port=False)
        d = app._plan_dict(auto, cap, "main", {"errors": [], "warnings": []})
        self.assertEqual(d["ctx"], 32768)
        self.assertTrue(d["ctx_auto"])
        self.assertIn("-c 32768", d["command"])
        cap.value = (plan, ["llama-server", "-m", FAKE_MODEL, "-c", "16384"], {}, "0", {}, 16384)
        hand = app.validate({"gpus": [0], "model": FAKE_MODEL, "ctx": 16384}, resolve_port=False)
        d2 = app._plan_dict(hand, cap, "main", {"errors": [], "warnings": []})
        self.assertEqual((d2["ctx"], d2["ctx_auto"]), (16384, False))

    def test_page_loads_the_log_before_the_serving_line(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            page = f.read()
        self.assertIn('ctx ${d.ctx_auto ? "auto " : ""}${d.ctx}', page)
        poll = page[page.find("async function pollStatus"):page.find("function logRender")]
        self.assertLess(poll.find("await loadLogBacklog()"), poll.find('badge("ok", "serving")'))
        opened = page[page.find("function openLog"):page.find("async function pollLog")]
        self.assertLess(opened.find("loadLogBacklog()"), opened.find("new EventSource"))
        reset = page[page.find("function resetLog"):page.find("function loadLogBacklog")]
        self.assertIn("clearTimeout(logAppend._t)", reset)
        self.assertIn("pre.dataset.live", page)

    def test_log_lines_written_before_attach_are_readable(self):
        seat = C.Seat(L)
        seat._append("$ llama-server -c 32768")
        seat._append("llama_model_loader: mock")
        items = seat.lines_since(0)
        self.assertEqual([x[1] for x in items][0], "$ llama-server -c 32768")
        self.assertEqual(items[-1][0], seat.seq)
        self.assertEqual(seat.lines_since(items[0][0])[0][1], "llama_model_loader: mock")


# ---------------------------------------------------------------------------------------------
# The licensed-library routes (/api/lib/update) on the SAME stub licence server the updater's own
# test uses (tests/test-pxa-lib-update.py).  That test covers tools/pxa_lib_update.py as a module;
# this covers the four HTTP handlers and the settings validation that exists only in Control, driven
# end to end: route -> LibUpdate -> tools/pxa_lib_update.py -> the fake licence server on loopback.
# ---------------------------------------------------------------------------------------------
_LUT_PATH = os.path.join(ROOT, "tests", "test-pxa-lib-update.py")
_LUT = None
if os.path.isfile(_LUT_PATH):
    _lut_spec = importlib.util.spec_from_file_location("pxa_lib_update_under_test", _LUT_PATH)
    _LUT = importlib.util.module_from_spec(_lut_spec)
    _lut_spec.loader.exec_module(_LUT)


@unittest.skipUnless(_LUT is not None, "tests/test-pxa-lib-update.py not present")
class LibUpdateRoutes(unittest.TestCase):
    """The library-update page's seams: status, a forced check, apply, rollback, and settings."""

    def setUp(self):
        self.ltmp = tempfile.mkdtemp(prefix="pxa-lib-routes-")
        self.addCleanup(shutil.rmtree, self.ltmp, ignore_errors=True)
        self.saved = {k: os.environ.get(k) for k in
                      ("PXA_PACKAGE_PUBKEY", "PXA_CONTROL_CONFIG_DIR", "PXA_LICENCE_KEY",
                       "PXA_LICENCE_URL", "PXA_ENGINE_DIR", "HOME")}
        self.addCleanup(self._restore_env)
        for k in ("PXA_LICENCE_URL", "PXA_LICENCE_KEY", "PXA_ENGINE_DIR"):
            os.environ.pop(k, None)
        os.environ["PXA_PACKAGE_PUBKEY"] = _LUT.F.PUB_HEX
        os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(self.ltmp, "cfg")
        os.environ["HOME"] = os.path.join(self.ltmp, "home")
        os.makedirs(os.environ["PXA_CONTROL_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["HOME"], exist_ok=True)
        self.ins, self.engine = self._make_install()
        self.lic = _LUT.FakeLibLicence({_LUT.REL1["lib_id"]: _LUT.REL1},
                                       {"stable": _LUT.REL1["lib_id"]}).start()
        self.addCleanup(self.lic.stop)
        os.environ["PXA_LICENCE_URL"] = self.lic.url
        os.environ["PXA_LICENCE_KEY"] = _LUT.KEY
        self.app = C.App(L, port=0, models_dirs=[MODELS])
        # Pin the install under test to THIS temp tree.  On a real box rig_static()'s engine_dir is a
        # live install, and a route test must never apply or roll back a library on someone's engine.
        self.app.rig_static = lambda force=False: {"engine_dir": self.ins}
        self.srv, self.port = serve_app(self.app)
        self.addCleanup(self._stop)

    def _restore_env(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _stop(self):
        try:
            self.srv.shutdown()
            self.srv.server_close()
        finally:
            self.app.lib.close()

    def _make_install(self):
        """The engine's install tree exactly as the tarball lays it out: `current` -> pxa-v3, and the two
        copies of libggml-pxqn.so hard-linked to each other."""
        ins = os.path.join(self.ltmp, "install")
        v = os.path.join(ins, "pxa-v3")
        for sub in ("lib", "lib-compat", "bin"):
            os.makedirs(os.path.join(v, sub))
        with open(os.path.join(v, "VERSION"), "w") as f:
            f.write("v3.1\n")
        engine = b"\x7fELF the library the engine release shipped\n" * 40
        with open(os.path.join(v, _LUT.COPIES[0]), "wb") as f:
            f.write(engine)
        os.link(os.path.join(v, _LUT.COPIES[0]), os.path.join(v, _LUT.COPIES[1]))
        os.symlink("pxa-v3", os.path.join(ins, "current"))
        return ins, engine

    def _content(self, name):
        with open(os.path.join(self.ins, "current", name), "rb") as f:
            return f.read()

    def get(self, path):
        code, _, body = req(self.port, path)
        return code, json.loads(body)

    def post(self, path, body):
        code, _, raw = req(self.port, path, "POST", body)
        return code, json.loads(raw)

    def test_status_is_pristine_then_a_forced_check_reports_the_release(self):
        # A GET touches the page, which starts the background poller; it may already have refreshed
        # the cache, so `update` on the first read is not pinned. What must hold: the install is
        # pristine and the install's settings are the defaults.
        code, d = self.get("/api/lib/update")
        self.assertEqual(code, 200, d)
        self.assertTrue(d["available"], d)
        self.assertEqual(d["installed"]["version"], "", d)
        self.assertEqual(d["installed"]["lib_id"], None, d)
        self.assertEqual(d["channel"], "stable", d)
        self.assertFalse(d["auto"], d)
        code, d = self.get("/api/lib/update?check=1")
        self.assertEqual(code, 200, d)
        self.assertEqual(d["latest"], _LUT.REL1["version"], d)
        self.assertEqual(d["lib_id"], _LUT.REL1["lib_id"], d)
        self.assertTrue(d["update"], d)
        self.assertIsNone(d["error"], d)

    def test_apply_installs_the_release_then_rollback_restores_the_shipped_library(self):
        code, d = self.get("/api/lib/update?check=1")
        self.assertTrue(d["update"], d)
        code, a = self.post("/api/lib/update/apply", {})
        self.assertEqual(code, 200, a)
        self.assertTrue(a["applied"], a)
        self.assertEqual(a["version"], _LUT.REL1["version"], a)
        code, d = self.get("/api/lib/update")
        self.assertEqual(d["installed"]["version"], _LUT.REL1["version"], d)
        self.assertNotEqual(self._content(_LUT.COPIES[0]), self.engine, "the shipped bytes were swapped")
        with open(os.path.join(self.ins, "lib", "lib-state.json")) as f:
            self.assertNotIn(_LUT.KEY, f.read(), "the licence key is never written to the state")
        code, r = self.post("/api/lib/update/rollback", {})
        self.assertEqual(code, 200, r)
        self.assertTrue(r["applied"], r)
        self.assertEqual(self._content(_LUT.COPIES[0]), self.engine, "rollback did not restore the shipped library")
        self.assertEqual(_LUT.LU.installed_version(self.ins), "", "the install is not back to shipped")

    def test_settings_reject_a_bad_channel_and_a_non_boolean_auto(self):
        code, d = self.post("/api/lib/update/settings", {"channel": "gamma"})
        self.assertEqual(code, 400, d)
        self.assertIn("channel must be one of", d["error"], d)
        code, d = self.post("/api/lib/update/settings", {"auto": "yes"})
        self.assertEqual(code, 400, d)
        self.assertIn("auto must be true or false", d["error"], d)
        code, d = self.post("/api/lib/update/settings", {"channel": "beta", "auto": True})
        self.assertEqual(code, 200, d)
        self.assertEqual((d["channel"], d["auto"]), ("beta", True), d)
        code, d = self.get("/api/lib/update")
        self.assertEqual((d["channel"], d["auto"]), ("beta", True), "settings did not persist")

    def test_apply_refuses_with_a_sentence_while_a_server_is_running(self):
        """The page is promised a 4xx and a plain sentence.  A refusal that reaches the page as a 500
        is the one shape a browser cannot render, so the status is asserted, not only the text."""
        saved = _LUT.LU.server_running
        _LUT.LU.server_running = lambda d=None: True
        self.addCleanup(lambda: setattr(_LUT.LU, "server_running", saved))
        code, d = self.post("/api/lib/update/apply", {})
        self.assertEqual(code, 400, d)
        self.assertIn("running", d["error"].lower(), d)

    def test_a_key_without_the_valued_role_is_refused_on_beta(self):
        self.lic.valued = False
        code, d = self.post("/api/lib/update/settings", {"channel": "beta"})
        self.assertEqual(code, 200, d)
        code, d = self.get("/api/lib/update?check=1")
        self.assertEqual(code, 200, d)
        self.assertEqual((d["error"] or {}).get("code"), "not_valued", d)
        self.assertFalse(d["beta"], d)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
