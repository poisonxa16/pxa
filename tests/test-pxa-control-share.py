#!/usr/bin/env python3
"""PXA Control's FREE shareable bench card (tools/pxa_control/share.py, tools/pxa_control_ui/share.*; todo
control-share-report): the card built from a bench-history row, the public text rules (what + how much, never how; no
rival engine names; no paths, flags, ports or hashes), the self-contained SVG, the three /api/share routes served by a
real Control, the token guard, and that nothing on the path asks the Store or an entitlement (free, never gated).
Fake cards, loopback only, no GPU, no network.

    python3 tests/test-pxa-control-share.py          (wired into CTest as test-pxa-control-share)
"""
import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-share-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN",
           "PXA_MODELS_DIR"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402
from pxa_control import share as SH  # noqa: E402

CFG = os.environ["PXA_CONTROL_CONFIG_DIR"]
os.makedirs(CFG, exist_ok=True)
GPUS = [(0, "NVIDIA GeForce GTX 1080 Ti", "6.1", 11264), (1, "NVIDIA GeForce GTX 1080 Ti", "6.1", 11264),
        (2, "NVIDIA GeForce RTX 3090", "8.6", 24576)]
RIVALS = ("llama.cpp", "llama-cpp", "ik_llama.cpp", "llama-server", "ollama", "vllm", "exllamav2", "lm studio", "lmstudio",
          "koboldcpp", "sglang", "tabbyapi", "tensorrt-llm", "mlc-llm")
FORBIDDEN_KEYS = ("cmd", "flags", "levers", "split_mode", "mtp", "port", "greedy512_sha", "telemetry", "params", "raw",
                  "model_sha8", "path")


def rec(ts, model="/srv/models/secret-dir/Qwen3-30B-A3B-Q4_K_M.gguf", dec=41.23, pre=812.4, cards=None, **kw):
    r = {"ts": ts, "model": model, "cards": cards if cards is not None else [0, 1], "port": 8091, "reps": 3, "warmup_s": 5,
         "classes": [{"class": "prose", "decode_tps": dec, "prefill_tps": 300.0, "reps": 3},
                     {"class": "edit", "decode_tps": dec * 0.9, "prefill_tps": 400.0, "reps": 3},
                     {"class": "long", "decode_tps": dec * 0.8, "prefill_tps": pre, "reps": 3}],
         "split_mode": "tensor", "mtp": True, "greedy512_sha": "ab" * 32,
         "telemetry": {"cards": [], "samples": 0, "warnings": []}}
    r.update(kw)
    return r


def write_bench(rows):
    with open(os.path.join(CFG, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class FakeApp(object):
    def __init__(self, rows=GPUS):
        self.rows = rows

    def gpus(self):
        return list(self.rows), None


def walk_strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from walk_strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from walk_strings(v)
    elif o is not None:
        yield str(o)


class Labels(unittest.TestCase):
    def test_rig_label_groups_cards_in_order(self):
        self.assertEqual(SH.rig_label(["NVIDIA GeForce GTX 1080 Ti", "NVIDIA GeForce GTX 1080 Ti", "NVIDIA GeForce RTX 3090"]),
                         "2x GTX 1080 Ti + RTX 3090")
        self.assertEqual(SH.rig_label(["Tesla V100-PCIE-32GB"]), "Tesla V100-PCIE-32GB")
        self.assertEqual(SH.rig_label([]), "this rig")

    def test_model_label_drops_folder_extension_split_and_quant(self):
        self.assertEqual(SH._model_label("/home/pat/models/Qwen3-30B-A3B-Q4_K_M.gguf"), ("Qwen3-30B-A3B", "Q4_K_M"))
        self.assertEqual(SH._model_label("C:\\Users\\pat\\GLM-4.6-PXQN4-00001-of-00003.gguf"), ("GLM-4.6", "PXQN4"))
        self.assertEqual(SH._model_label(""), ("a local model", ""))
        self.assertEqual(SH._model_label(None), ("a local model", ""))

    def test_rival_engine_names_never_survive(self):
        for r in RIVALS:
            for s in (r, r.upper(), "model-for-" + r + "-Q4_K_M", "%s fork" % r):
                out = SH.scrub(s)
                self.assertIsNone(SH.RIVAL_RE.search(out), (s, out))
                self.assertNotIn(r.lower().replace(" ", ""), out.lower().replace(" ", ""), (s, out))

    def test_model_names_that_merely_contain_llama_stay(self):
        self.assertEqual(SH._model_label("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")[0], "Meta-Llama-3.1-8B-Instruct")

    def test_scrub_bounds_and_strips_control_chars(self):
        self.assertEqual(SH.scrub("a\x00b\nc"), "a b c")
        self.assertEqual(len(SH.scrub("x" * 500)), 80)

    def test_speeds_pick_prose_decode_and_long_prefill(self):
        self.assertEqual(SH._speeds(rec(1, dec=41.26, pre=812.4)), (41.3, 812.0))
        self.assertEqual(SH._speeds({"classes": [{"class": "edit", "decode_tps": 9.0}]}), (9.0, 0.0))
        self.assertEqual(SH._speeds({"classes": [{"class": "prose", "decode_tps": "nan"}]}), (0.0, 0.0))
        self.assertEqual(SH._speeds({}), (0.0, 0.0))


class Card(unittest.TestCase):
    def setUp(self):
        C.save_config({})

    def test_card_shape_and_values(self):
        c = SH.card(FakeApp(), rec(1760800000.5))
        for k in ("free", "id", "when", "rig", "gpu_count", "vram_gb", "model", "quant", "decode_tps", "prefill_tps", "name",
                  "headline", "text", "mark", "links"):
            self.assertIn(k, c)
        self.assertIs(c["free"], True)
        self.assertEqual(c["rig"], "2x GTX 1080 Ti")
        self.assertEqual(c["gpu_count"], 2)
        self.assertEqual(c["vram_gb"], 22.0)
        self.assertEqual((c["model"], c["quant"]), ("Qwen3-30B-A3B", "Q4_K_M"))
        self.assertEqual(c["decode_tps"], 41.2)
        self.assertEqual(c["headline"], "2x GTX 1080 Ti runs Qwen3-30B-A3B Q4_K_M at 41.2 tokens/s")
        self.assertEqual(c["mark"], "Made with PXA")
        self.assertEqual(c["links"]["discord"], "https://discord.pxanetwork.com")
        self.assertIn("store", c["links"])
        self.assertIn("discord.pxanetwork.com", c["text"])
        self.assertIn("pxanetwork.com", c["text"])

    def test_never_how_no_paths_flags_ports_or_hashes(self):
        c = SH.card(FakeApp(), rec(1760800000.5, extra_args="--n-cpu-moe 30 -ot exps=CPU"))
        for k in FORBIDDEN_KEYS:
            self.assertNotIn(k, c, k)
        blob = " ".join(walk_strings(c))
        for bad in ("/srv/models", "secret-dir", ".gguf", "8091", "ab" * 32, "tensor", "--n-cpu-moe", "exps=CPU"):
            self.assertNotIn(bad, blob, bad)

    def test_rival_named_model_file_is_scrubbed_everywhere(self):
        c = SH.card(FakeApp(), rec(1, model="/m/ollama-vllm-llama.cpp-Mistral-7B-Q4_0.gguf"))
        blob = " ".join(walk_strings(c)) + SH.svg(c)
        self.assertIsNone(SH.RIVAL_RE.search(blob), blob)
        self.assertIn("Mistral-7B", c["model"])

    def test_mixed_rig_old_card_next_to_new_is_named_as_it_is(self):
        c = SH.card(FakeApp(), rec(1, cards=[0, 1, 2]))
        self.assertEqual(c["rig"], "2x GTX 1080 Ti + RTX 3090")
        self.assertEqual(c["gpu_count"], 3)

    def test_no_cards_recorded_means_every_card(self):
        self.assertEqual(SH.card(FakeApp(), rec(1, cards=[]))["gpu_count"], 3)

    def test_unreadable_card_list_is_this_rig_not_a_crash(self):
        class Bad(object):
            def gpus(self):
                raise RuntimeError("nvidia-smi gone")
        c = SH.card(Bad(), rec(1))
        self.assertEqual(c["rig"], "this rig")

    def test_board_name_only_when_asked(self):
        C.save_config({"user_name": "Pat"})
        self.assertEqual(SH.card(FakeApp(), rec(1))["name"], "")
        self.assertEqual(SH.card(FakeApp(), rec(1), with_name=True)["name"], "Pat")
        C.save_config({"user_name": "visit www.spam.com"})
        self.assertEqual(SH.card(FakeApp(), rec(1), with_name=True)["name"], "")     # the board's own name rules

    def test_big_numbers_drop_the_decimal(self):
        c = SH.card(FakeApp(), rec(1, dec=123.24))
        self.assertIn("at 123 tokens/s", c["headline"])


class Svg(unittest.TestCase):
    def test_svg_is_wellformed_self_contained_1200x630(self):
        s = SH.svg(SH.card(FakeApp(), rec(1, pre=812.4)))
        root = ET.fromstring(s)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertEqual((root.get("width"), root.get("height")), ("1200", "630"))
        low = s.lower().replace('xmlns="http://www.w3.org/2000/svg"', "")
        for bad in ("<script", "onload", "href=", "<image", "<foreignobject", "http://", "url(http"):
            self.assertNotIn(bad, low, bad)
        for need in ("Made with PXA", "discord.pxanetwork.com", "Store", "41.2", "812", "2x GTX 1080 Ti"):
            self.assertIn(need, s, need)

    def test_svg_escapes_hostile_text(self):
        c = SH.card(FakeApp([(0, "<script>alert(1)</script> & \"x\"", "8.6", 1024)]), rec(1, cards=[0],
                    model="/m/<img src=x onerror=alert(1)>-Q4_0.gguf"))
        s = SH.svg(c)
        ET.fromstring(s)
        self.assertNotIn("<script", s)
        self.assertNotIn("<img", s)

    def test_board_name_on_the_image_only_when_asked(self):
        C.save_config({"user_name": "Pat"})
        self.assertNotIn("by Pat", SH.svg(SH.card(FakeApp(), rec(1))))
        self.assertIn("by Pat", SH.svg(SH.card(FakeApp(), rec(1), with_name=True)))

    def test_long_rig_names_get_a_smaller_font(self):
        short = SH.svg(SH.card(FakeApp(), rec(1, cards=[0])))
        long_ = SH.svg(SH.card(FakeApp(), rec(1, cards=[0, 1, 2])))
        self.assertIn('font-size="64"', short)
        self.assertNotIn('font-size="64" font-weight="800"', long_)

    def test_no_prefill_no_prefill_block(self):
        r = rec(1)
        r["classes"] = [{"class": "prose", "decode_tps": 10.0}]
        s = SH.svg(SH.card(FakeApp(), r))
        self.assertNotIn("reading the prompt", s)


TOKENS = {}


def serve_app(app):
    srv = C.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    TOKENS[srv.server_address[1]] = app.token
    return srv, srv.server_address[1]


def req(port, path, auth=True):
    h = {}
    if auth and TOKENS.get(port):
        h["X-PXA-Token"] = TOKENS[port]
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), headers=h)
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=7777)
        cls.app.gpus = lambda *a, **k: (list(GPUS), None)
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        C.save_config({"user_name": "Pat"})
        write_bench([rec(1760700000.25, dec=30.0), rec(1760800000.5, dec=41.23),
                     {"ts": 1760900000.0, "model": "x.gguf", "classes": []}])      # a failed run: no number, no card

    def test_routes_are_registered_and_do_not_shadow_existing_ones(self):
        for k in (("GET", "/api/share/runs"), ("GET", "/api/share/card"), ("GET", "/api/share/card.svg")):
            self.assertIn(k, C.ROUTES)
        self.assertIn(("GET", "/api/bench"), C.ROUTES)                    # the Benchmarks tab's own route is untouched
        self.assertIs(C.ROUTES[("GET", "/api/bench")], C.routes.r_bench)

    def test_runs_newest_first_without_numberless_runs(self):
        code, _h, b = req(self.port, "/api/share/runs")
        self.assertEqual(code, 200)
        j = json.loads(b)
        self.assertIs(j["free"], True)
        self.assertEqual([r["id"] for r in j["runs"]], [1760800000.5, 1760700000.25])
        self.assertEqual(j["runs"][0]["model"], "Qwen3-30B-A3B")
        self.assertEqual(j["runs"][0]["decode_tps"], 41.2)
        self.assertNotIn("/srv/models", b.decode())

    def test_card_defaults_to_the_newest_run(self):
        code, _h, b = req(self.port, "/api/share/card")
        self.assertEqual(code, 200)
        j = json.loads(b)
        self.assertEqual(j["id"], 1760800000.5)
        self.assertEqual(j["rig"], "2x GTX 1080 Ti")
        self.assertEqual(j["name"], "")

    def test_card_by_id_and_with_name(self):
        code, _h, b = req(self.port, "/api/share/card?id=1760700000.25&name=1")
        self.assertEqual(code, 200)
        j = json.loads(b)
        self.assertEqual((j["decode_tps"], j["name"]), (30.0, "Pat"))

    def test_bad_and_unknown_ids(self):
        code, _h, b = req(self.port, "/api/share/card?id=abc")
        self.assertEqual(code, 400)
        self.assertEqual(json.loads(b)["code"], "bad_id")
        code, _h, b = req(self.port, "/api/share/card.svg?id=12345")
        self.assertEqual(code, 404)
        self.assertEqual(json.loads(b)["code"], "not_found")

    def test_no_runs_is_a_sentence(self):
        write_bench([])
        code, _h, b = req(self.port, "/api/share/card")
        self.assertEqual(code, 404)
        self.assertEqual(json.loads(b)["code"], "no_runs")
        self.assertIn("run a benchmark first", json.loads(b)["error"])
        code, _h, b = req(self.port, "/api/share/runs")
        self.assertEqual((code, json.loads(b)["runs"]), (200, []))

    def test_svg_route(self):
        code, h, b = req(self.port, "/api/share/card.svg")
        self.assertEqual(code, 200)
        self.assertTrue(h["Content-Type"].startswith("image/svg+xml"), h["Content-Type"])
        self.assertIn("default-src 'none'", h.get("Content-Security-Policy", ""))
        self.assertEqual(h.get("X-Content-Type-Options"), "nosniff")
        ET.fromstring(b)
        self.assertIn(b"41.2", b)

    def test_api_needs_the_token(self):
        for p in ("/api/share/runs", "/api/share/card", "/api/share/card.svg"):
            code, _h, _b = req(self.port, p, auth=False)
            self.assertEqual(code, 401, p)

    def test_free_never_asks_the_store_or_an_entitlement(self):
        """the store is configured to explode: the card must not care (free, never gated)."""
        from pxa_control import store as ST
        calls = []
        saved = {}
        for n in ("store_url", "LicdClient", "EntitlementCache", "catalog_view"):
            if hasattr(ST, n):
                saved[n] = getattr(ST, n)
                setattr(ST, n, lambda *a, _n=n, **k: calls.append(_n) or (_ for _ in ()).throw(AssertionError(_n)))
        try:
            for p in ("/api/share/runs", "/api/share/card", "/api/share/card.svg"):
                code, _h, _b = req(self.port, p)
                self.assertEqual(code, 200, p)
        finally:
            for n, v in saved.items():
                setattr(ST, n, v)
        self.assertEqual(calls, [])

    def test_share_module_imports_no_gate(self):
        tree = ast.parse(open(SH.__file__).read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mods.add((node.module or "") + ":" + ",".join(a.name for a in node.names))
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        blob = " ".join(mods)
        for bad in ("store", "plugins", "urllib", "http", "socket", "entitle"):
            self.assertNotIn(bad, blob, bad)
        self.assertIs(SH.FREE, True)


class Page(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=7777)
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_page_and_assets_are_served(self):
        for p, ct in (("/share.html", "text/html"), ("/share.js", "application/javascript"), ("/share.css", "text/css")):
            code, h, b = req(self.port, p)
            self.assertEqual(code, 200, p)
            self.assertTrue(h["Content-Type"].startswith(ct), (p, h["Content-Type"]))
            self.assertTrue(b, p)

    def test_page_has_no_inline_script_and_a_strict_csp(self):
        code, h, b = req(self.port, "/share.html")
        self.assertEqual(code, 200)
        self.assertEqual(C.inline_script_hashes(b), [])
        csp = h.get("Content-Security-Policy", "")
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-inline' ; script", csp)
        html = b.decode()
        self.assertIn('src="/share.js"', html)
        for need in ("discord.pxanetwork.com", "/#store", "Free"):
            self.assertIn(need, html)

    def test_page_js_only_talks_to_the_share_routes(self):
        js = open(os.path.join(C.UI_DIR, "share.js")).read()
        paths = set(re.findall(r'"(/api/[a-z/.]+)', js))
        self.assertTrue(paths)
        self.assertTrue(all(p.startswith("/api/share/") for p in paths), paths)
        self.assertNotIn("eval(", js)
        self.assertNotIn("innerHTML", js)

    def test_control_menu_links_the_share_page(self):
        html = open(os.path.join(C.UI_DIR, "index.html")).read()
        self.assertIn('href="/share.html"', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
