#!/usr/bin/env python3
"""PXA Control fixes from the Valued Supporters' beta 2 feedback (2026-10-19). No GPU, no model, loopback only.

  1. Benchmarks: a KV cache type option (default = as served), the KV the run used in the result and the history row,
     the served type read from the engine's /props (cache_type_k/v) with the command line and the saved setting as
     fallbacks; a different type restarts a server Control started, a server started by hand is refused with a sentence.
  2. Report a problem: a server started by hand (a process or container the Servers list found, a llama-server
     answering on a port of this computer, or an address the user types) can be reported, with version, KV, model
     file and flags; addresses are limited to this computer and private LAN IPs.
  3. Model download: the sha256 is computed while the bytes arrive (no second read of an 80 GB file), a resume hashes
     the part on disk first with progress, and the status carries bytes hashed / total / rate / time left.

    python3 tests/test-pxa-control-beta2.py          (wired into CTest as test-pxa-control-beta2)
"""
import hashlib
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
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-beta2-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_MODELS_DIR", None)
os.environ["PXA_CONTROL_DISCOVER"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402
import pxa_control.core as CORE  # noqa: E402
import pxa_control.routes as R  # noqa: E402
import pxa_hf as HF  # noqa: E402

MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)
FAKE_MODEL = os.path.join(MODELS, "tiny.gguf")
with open(FAKE_MODEL, "wb") as f:
    f.write(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)


def slurp(p):
    with open(p, "rb") as f:
        return f.read()


class _TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StubEngine(object):
    """a llama-server's /props and /health on loopback; props is editable per test."""

    def __init__(self, props=None, health=200):
        self.props = props if props is not None else {}
        self.health = health
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/props":
                    return self._send(200, json.dumps(outer.props).encode())
                if p == "/health":
                    return self._send(outer.health, b'{"status":"ok"}' if outer.health == 200 else b'{"error":"loading"}')
                return self._send(404, b"{}")

            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.srv = _TS(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def closed_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeBench(object):
    CLASS_PROMPTS = {c: {"prompt": c + " prompt", "n_predict": 8} for c in ("prose", "edit", "long")}

    @staticmethod
    def _get_json(url, path, timeout=10):
        return {"model_path": FAKE_MODEL}

    @staticmethod
    def measure_once(url, prompt, n, timeout=0):
        return {}

    @staticmethod
    def _post_json(url, path, body, timeout=0):
        return {"timings": {"predicted_per_second": 40.0, "prompt_per_second": 900.0, "prompt_n": 20}}

    @staticmethod
    def greedy512(url):
        return {"sha256": "cd" * 32, "empty": False}


# ---- 1. the bench's KV cache type ------------------------------------------------------------------------------------
class ServedKV(unittest.TestCase):
    def test_engine_props_first(self):
        e = StubEngine({"cache_type_k": "q8_0", "cache_type_v": "q8_0"})
        try:
            kv = C.served_kv(e.port, ["llama-server", "-ctk", "q4_0"], {"kv": "f16"})
            self.assertEqual((kv["label"], kv["source"]), ("q8_0", "server"))   # the engine's own answer beats flags/settings
            e.props = {"cache_type_k": "q8_0", "cache_type_v": "q4_0"}
            self.assertEqual(C.served_kv(e.port)["label"], "q8_0/q4_0")
        finally:
            e.close()

    def test_flags_then_settings_then_none(self):
        dead = closed_port()
        kv = C.served_kv(dead, ["llama-server", "-m", "x.gguf", "-ctk", "q4_0", "-ctv", "q4_0"])
        self.assertEqual((kv["label"], kv["source"]), ("q4_0", "flags"))
        kv = C.served_kv(None, ["llama-server", "--cache-type-k=q8_0"])
        self.assertEqual(kv["label"], "q8_0/f16")                  # the engine default for the side not given
        self.assertEqual(C.served_kv(None, None, {"kv": "q8_0"})["source"], "settings")
        self.assertIsNone(C.served_kv(None, None, {"kv": "auto"}))   # auto = the engine picks: unknown without /props
        self.assertIsNone(C.served_kv(dead, ["llama-server"]))
        e = StubEngine({"cache_type_k": "q8_0; rm -rf /"})          # a hostile answer cannot inject text
        try:
            self.assertEqual(C.served_kv(e.port)["k"], "q8_0rmrf")
        finally:
            e.close()


class BenchKV(unittest.TestCase):
    def setUp(self):
        self.e = StubEngine({"cache_type_k": "q8_0", "cache_type_v": "q8_0", "model_path": FAKE_MODEL})
        self.app = C.App(L, port=0, models_dirs=[MODELS])
        self.calls = []
        self.app._bench_module = lambda: FakeBench
        self.app._bench_run = lambda *a: self.calls.append(a)          # the thread does nothing here

    def tearDown(self):
        self.e.close()

    def _run_seat(self):
        seat = self.app.seat
        seat.cmd = ["llama-server", "-m", FAKE_MODEL, "--port", str(self.e.port)]
        seat.req = {"model": FAKE_MODEL, "gpus": [0], "port": self.e.port, "kv": "auto"}
        seat.running = lambda: True
        return seat

    def test_result_and_history_carry_the_kv(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0], "port": self.e.port}
        app.bench = {"running": True, "lines": [], "result": None, "error": None}
        app._bench_run(FakeBench, self.e.port, 1, 0, [0])
        self.assertIsNone(app.bench["error"], app.bench["error"])
        res = app.bench["result"]
        self.assertEqual((res["kv"], res["kv_source"]), ("q8_0", "server"))
        self.assertTrue(any(ln.endswith("KV cache: q8_0") for ln in app.bench["lines"]), app.bench["lines"])
        row = [r for r in R.r_bench(app, {}, {})["history"] if r.get("ts") == res["ts"]][0]
        self.assertEqual(row["kv"], "q8_0")
        p = app.score_payload(res)                     # the board payload names the KV too (auto setting, no -ctk)
        self.assertEqual((p["params"]["kv_k"], p["params"]["kv_v"]), ("q8_0", "q8_0"))

    def test_as_served_is_the_default_and_never_restarts(self):
        self._run_seat()
        for kv in (None, "", "served", "auto"):
            body = {"reps": 1, "warmup": 0}
            if kv is not None:
                body["kv"] = kv
            r = self.app.start_bench(body)
            self.assertEqual((r["ok"], r["restart"]), (True, None), kv)
            self.app.bench["running"] = False
        self.assertEqual([c[5] for c in self.calls], [None] * 4)

    def test_same_type_no_restart_other_type_restarts(self):
        self._run_seat()
        self.assertIsNone(self.app.start_bench({"kv": "q8_0"})["restart"])   # already q8_0 (from /props)
        self.app.bench["running"] = False
        r = self.app.start_bench({"kv": "q4_0"})
        self.assertEqual(r["restart"], {"sid": "main", "kv": "q4_0", "was": "q8_0"})
        self.assertEqual(self.calls[-1][5], r["restart"])

    def test_bad_type_and_a_server_started_by_hand(self):
        self._run_seat()
        with self.assertRaises(C.Invalid):
            self.app.start_bench({"kv": "q3_x"})
        app = C.App(L, port=0, models_dirs=[MODELS])
        app._bench_module = lambda: FakeBench
        app._bench_run = lambda *a: None
        app.set_attach_port(self.e.port)                 # attached to a server Control did not start
        try:
            with self.assertRaises(C.Invalid) as cm:
                app.start_bench({"kv": "q4_0"})
            self.assertIn("not started by PXA Control", str(cm.exception))
            self.assertIn("-ctk q4_0", str(cm.exception))
            self.assertEqual(app.start_bench({})["restart"], None)      # as served still works
        finally:
            app.set_attach_port(0)

    def test_restart_waits_for_health_and_says_so(self):
        self._run_seat()
        seen = {}

        def fake_restart(body, sid=None):
            seen["body"], seen["sid"] = body, sid
            return {"ok": True}
        self.app.restart = fake_restart
        lines = []
        port = self.app._bench_restart_kv({"sid": "main", "kv": "q4_0", "was": "q8_0"}, lines.append)
        self.assertEqual(port, self.e.port)
        self.assertEqual((seen["body"]["kv"], seen["sid"]), ("q4_0", "main"))
        self.assertEqual(seen["body"]["model"], FAKE_MODEL)                 # the rest of the server's settings kept
        self.assertTrue(lines[0].startswith("restarting") and "q4_0" in lines[0] and "q8_0" in lines[0], lines)
        self.app.restart = lambda body, sid=None: {"ok": False, "text": "refused: does not fit"}
        with self.assertRaises(RuntimeError):
            self.app._bench_restart_kv({"sid": "main", "kv": "f16", "was": "q8_0"}, lines.append)

    def test_page_has_the_option_and_the_column(self):
        html = slurp(os.path.join(TOOLS, "pxa_control_ui", "index.html")).decode("utf-8")
        self.assertIn('id="bn-kv"', html)
        for v in ('value=""', 'value="f16"', 'value="q8_0"', 'value="q4_0"'):
            self.assertIn(v, html[html.index('id="bn-kv"'):html.index('id="bn-kv"') + 700])
        self.assertIn('kv: $("#bn-kv").value', html)
        self.assertIn('class: "bn-kv mono"', html)
        self.assertIn(">KV</th>", html)


# ---- 2. Report a problem for a server started by hand ---------------------------------------------------------------
class ReportByHand(unittest.TestCase):
    def setUp(self):
        self.e = StubEngine({"model_path": "/home/bob/models/Hand-Model-PXQN4.gguf", "n_ctx": 32768, "total_slots": 1,
                             "build_info": "b4711-abc1234", "cache_type_k": "q8_0", "cache_type_v": "q8_0"})
        self.app = C.App(L, port=0, models_dirs=[MODELS])

    def tearDown(self):
        self.e.close()

    def test_address_rules(self):
        P = C.parse_report_address
        self.assertEqual(P("8080"), ("http", "127.0.0.1", 8080))
        self.assertEqual(P("localhost:9000"), ("http", "127.0.0.1", 9000))
        self.assertEqual(P("http://192.168.1.20:8081"), ("http", "192.168.1.20", 8081))
        self.assertEqual(P("http://[::1]:8080/"), ("http", "::1", 8080))
        for bad in ("example.com:80", "http://8.8.8.8:80", "169.254.169.254:80", "0.0.0.0:80", "127.0.0.1",
                    "127.0.0.1:99999", "http://127.0.0.1:8080/props?x=1", "file:///etc/passwd", "", "a b"):
            with self.assertRaises(C.Invalid, msg=bad):
                P(bad)

    def test_found_on_a_listening_port(self):
        dead = closed_port()
        d = self.app.report_servers(ports=[self.e.port, dead])
        ids = [x["id"] for x in d["servers"]]
        self.assertIn("url:http://127.0.0.1:%d" % self.e.port, ids)
        self.assertNotIn("url:http://127.0.0.1:%d" % dead, ids)
        x = [x for x in d["servers"] if x["kind"] == "port"][0]
        self.assertIn("Hand-Model-PXQN4.gguf", x["label"])
        self.assertTrue(any(x["kind"] == "managed" for x in d["servers"]))      # our own still listed
        self.assertEqual(R.r_report_servers(self.app, {}, {})["servers"][0]["kind"], "managed")   # route, discovery off

    def test_bundle_by_address(self):
        b = self.app.report_bundle(url="127.0.0.1:%d" % self.e.port)
        bu = b["bundle"]
        self.assertEqual(bu["engine"]["version"], "b4711-abc1234")
        self.assertEqual((bu["server"]["kv"], bu["server"]["n_ctx"], bu["server"]["health"]), ("q8_0", 32768, "ok"))
        self.assertEqual(bu["model"]["filename"], "Hand-Model-PXQN4.gguf")
        self.assertEqual(bu["phase"], "serving")
        self.assertTrue(any("only the address is known" in n for n in bu["collection_notes"]), bu["collection_notes"])
        self.assertNotIn("/home/bob", json.dumps(b))
        r = R.r_report_bundle(self.app, {}, {"url": ["http://127.0.0.1:%d" % self.e.port]})   # the route
        self.assertEqual(r["bundle"]["server"]["port"], self.e.port)
        with self.assertRaises(C.Invalid):
            R.r_report_bundle(self.app, {}, {"url": ["http://8.8.8.8:80"]})
        # a server that is down still gives a report the user can send
        b = self.app.report_bundle(url=str(closed_port()))
        self.assertEqual(b["bundle"]["phase"], "down")
        self.assertTrue(b["bundle"]["engine"]["version"].startswith("unknown"))

    def test_bundle_by_key_has_flags_and_log(self):
        inst = {"key": "p:4242", "kind": "process", "port": self.e.port, "pid": 4242, "gpus": [0],
                "model": "/home/bob/models/Hand-Model-PXQN4.gguf",
                "cmd": "llama-server -m Hand-Model-PXQN4.gguf -ctk q8_0 -ctv q8_0 -c 32768 --api-key sk-abcdefghijklmnop1234"}
        self.app._instance = lambda key: inst if key == "p:4242" else (_ for _ in ()).throw(C.Invalid("no such server"))
        self.app.external_log = lambda key, tail=300, q="": {"key": key, "src": "/tmp/x.log",
                                                              "lines": ["build: 4711 (abc1234)", "PXA_REGISTRY: card 0 sm 70",
                                                                        "slot ready"]}
        bu = self.app.report_bundle(key="p:4242")["bundle"]
        self.assertIn("-ctk q8_0", bu["flags"])
        self.assertNotIn("sk-abcdefghijklmnop1234", json.dumps(bu))              # redacted
        self.assertEqual(bu["banner"], ["PXA_REGISTRY: card 0 sm 70"])
        self.assertEqual(bu["server"]["kind"], "process")
        self.assertEqual(bu["server"]["gpus"], [0])
        with self.assertRaises(C.Invalid):
            self.app.report_bundle(key="p:../../etc")
        with self.assertRaises(C.Invalid):
            self.app.report_bundle(key="p:999")

    def test_our_own_server_is_unchanged(self):
        self.app.seat._append("build: 4711 (abc1234)")
        bu = self.app.report_bundle("main")["bundle"]
        self.assertNotIn("server", bu)
        self.assertEqual(bu["engine"]["version"], "4711 (abc1234)")

    def test_page_lists_more_servers_and_an_address_box(self):
        html = slurp(os.path.join(TOOLS, "pxa_control_ui", "index.html")).decode("utf-8")
        self.assertIn('id="rp-addr"', html)
        self.assertIn('"/api/report/servers"', html)
        self.assertIn('Another server: type its address', html)


# ---- 3. the download's sha256: hashed as it arrives, with progress ---------------------------------------------------
class _Files(object):
    data = {}
    status_range = True


class FileHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        name = urllib.parse.unquote(self.path.rsplit("/", 1)[-1])
        d = _Files.data.get(name)
        if d is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range") or ""
        start = int(rng[6:].split("-")[0]) if (rng.startswith("bytes=") and _Files.status_range) else 0
        self.send_response(206 if start else 200)
        body = d[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DownloadHash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = _TS(("127.0.0.1", 0), FileHandler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.saved = HF.HF
        HF.HF = "http://127.0.0.1:%d" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        HF.HF = cls.saved
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="dl-", dir=TMP)
        self.data = os.urandom(3 * 1024 * 1024 + 123)
        _Files.data = {"m.gguf": self.data}
        _Files.status_range = True
        self.reads = []
        self.real = HF._sha256

        def counting(path, stop=None, progress=None, h=None, limit=None):
            self.reads.append(limit)
            return self.real(path, stop, progress, h=h, limit=limit)
        HF._sha256 = counting

    def tearDown(self):
        HF._sha256 = self.real
        shutil.rmtree(self.dir, ignore_errors=True)

    def item(self, sha=None):
        return {"name": "m.gguf", "path": "m.gguf", "size": len(self.data),
                "sha256": sha if sha is not None else hashlib.sha256(self.data).hexdigest()}

    def run_dl(self, sha=None):
        d = HF.Download("poisonxa/X", [self.item(sha)], self.dir)
        d.run()
        return d

    def test_fresh_download_never_reads_the_file_again(self):
        d = self.run_dl()
        self.assertEqual(d.state, "done", d.error)
        self.assertEqual(slurp(os.path.join(self.dir, "m.gguf")), self.data)
        self.assertEqual(self.reads, [], "the sha256 was computed from the bytes as they arrived")
        st = d.status()
        for k in ("hash_phase", "hash_done", "hash_total", "hash_rate", "hash_eta_s"):
            self.assertIn(k, st)

    def test_mismatch_still_throws_the_file_away(self):
        d = self.run_dl(sha="0" * 64)
        self.assertEqual(d.state, "failed")
        self.assertIn("sha256", d.error)
        self.assertEqual(os.listdir(self.dir), [])

    def test_resume_hashes_only_the_part_on_disk_with_progress(self):
        have = 1024 * 1024 + 7
        with open(os.path.join(self.dir, "m.gguf.part"), "wb") as f:
            f.write(self.data[:have])
        seen = []
        d = HF.Download("poisonxa/X", [self.item()], self.dir)
        real_prog = d._hash_progress

        def prog(n):
            real_prog(n)
            st = d.status()
            seen.append((st["state"], st["hash_phase"], st["hash_done"], st["hash_total"]))
        d._hash_progress = prog
        d.run()
        self.assertEqual(d.state, "done", d.error)
        self.assertEqual(slurp(os.path.join(self.dir, "m.gguf")), self.data)
        self.assertEqual(self.reads, [have], "only the resumed prefix is read back")
        self.assertTrue(seen and all(s[0] == "checking" and s[1] == "resume" and s[3] == have for s in seen), seen[:3])
        self.assertEqual(seen[-1][2], have)
        self.assertEqual([s[2] for s in seen], sorted(s[2] for s in seen))         # monotone

    def test_resume_with_a_bad_prefix_is_caught(self):
        with open(os.path.join(self.dir, "m.gguf.part"), "wb") as f:
            f.write(b"\xff" * 4096)                                                 # not the file's first bytes
        d = self.run_dl()
        self.assertEqual(d.state, "failed")
        self.assertIn("sha256", d.error)

    def test_server_ignoring_the_range_starts_over_and_still_matches(self):
        with open(os.path.join(self.dir, "m.gguf.part"), "wb") as f:
            f.write(self.data[:5000])
        _Files.status_range = False
        d = self.run_dl()
        self.assertEqual(d.state, "done", d.error)
        self.assertEqual(self.reads, [], "a restart from zero hashes as it arrives")

    def test_eta_and_rate_while_checking(self):
        d = HF.Download("poisonxa/X", [self.item()], self.dir)
        d.state, d.hash_phase, d.hash_total, d.hash_started = "checking", "resume", 1000, time.time() - 2.0
        d._hash_progress(250)
        st = d.status()
        self.assertAlmostEqual(st["hash_rate"], 125.0, delta=10)
        self.assertAlmostEqual(st["hash_eta_s"], 6.0, delta=0.6)

    def test_page_shows_bytes_and_time_left(self):
        html = slurp(os.path.join(TOOLS, "pxa_control_ui", "index.html")).decode("utf-8")
        self.assertIn("function hfHashLine(dl)", html)
        self.assertIn("left", html[html.index("function hfHashLine(dl)"):html.index("function hfHashLine(dl)") + 900])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
