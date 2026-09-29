#!/usr/bin/env python3
"""PXA Control (pxa-launch --gui, tools/pxa_control.py): HTTP handlers, token auth, lever validation,
preset persistence, the engine proxy and the seat's process handling. No GPU, no model, no network
beyond 127.0.0.1: cards come from PXA_LAUNCH_FAKE_GPUS and the "engine" is a stub HTTP server.

    python3 tests/test-pxa-control.py          (wired into CTest as test-pxa-control)
"""
import http.server
import importlib.util
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_MODELS_DIR", None)
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


class LaunchValidation(unittest.TestCase):
    def v(self, **kw):
        body = {"gpus": [0], "model": FAKE_MODEL}
        body.update(kw)
        return C.validate_launch(body, {0, 1}, CATALOG, [MODELS], gui_port=7777)

    def test_defaults(self):
        r = self.v()
        self.assertEqual((r["gpus"], r["kv"], r["sm"], r["fa"], r["ctx"], r["np"]),
                         ([0], "f16", "auto", "auto", 0, 0))
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
        # symlink escape out of the folder is refused too
        link = os.path.join(MODELS, "link.gguf")
        if not os.path.lexists(link):
            os.symlink(other, link)
        with self.assertRaises(C.Invalid):
            self.v(model=link)

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
        self.assertEqual((code, d["lan"], d["kv_types"][0]), (200, False, "f16"))

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


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
