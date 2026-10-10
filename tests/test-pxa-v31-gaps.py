#!/usr/bin/env python3
"""Gaps that the v3.1 tickets asked for and the tree did not already have.

Gemma 4 assistant drafter matching (the same rule as pxa_find_gemma4_assistant), the launch
allow-list for -md, --hot-model argv, the Pascal hide for hot swap, and /v1/models parsing for
the Servers tab. No GPU. The engine hot-swap router and /v1/models listing already exist; this
does not retest them.

    python3 tests/test-pxa-v31-gaps.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-v31-gaps-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
sys.path.insert(0, TOOLS)

import importlib.util
_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402


def write_gguf(path, kv):
    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", len(kv))
    for key, kind, val in kv:
        kb = key.encode()
        buf += struct.pack("<Q", len(kb)) + kb
        if kind == "str":
            buf += struct.pack("<I", 8)
            vb = val.encode()
            buf += struct.pack("<Q", len(vb)) + vb
        else:
            buf += struct.pack("<I", 4)
            buf += struct.pack("<I", int(val))
    with open(path, "wb") as f:
        f.write(buf)


class GemmaDraft(unittest.TestCase):
    def test_match_is_the_one_assistant_with_the_same_width(self):
        target = {"general.architecture": "gemma4", "gemma4.embedding_length": 256}
        cands = [
            ("/m/other.gguf", {"general.architecture": "qwen3"}, 100),
            ("/m/a.gguf", {"general.architecture": "gemma4_mtp", "gemma4-assistant.embedding_length_out": 256}, 1000),
            ("/m/wide.gguf", {"general.architecture": "gemma4-assistant", "gemma4-assistant.embedding_length_out": 512}, 1000),
            ("/m/huge.gguf", {"general.architecture": "gemma4_mtp", "gemma4-assistant.embedding_length_out": 256}, (4 << 30) + 1),
        ]
        self.assertEqual(L.gemma4_assistant_match(target, cands), "/m/a.gguf")

    def test_two_matches_is_not_a_guess(self):
        target = {"general.architecture": "gemma4", "gemma4.embedding_length": 256}
        cands = [
            ("/m/a.gguf", {"general.architecture": "gemma4_mtp", "gemma4-assistant.embedding_length_out": 256}, 10),
            ("/m/b.gguf", {"general.architecture": "gemma4-assistant", "gemma4-assistant.embedding_length_out": 256}, 10),
        ]
        self.assertEqual(L.gemma4_assistant_match(target, cands), "")

    def test_not_gemma4(self):
        self.assertEqual(L.gemma4_assistant_match({"general.architecture": "qwen3"}, []), "")

    def test_sibling_on_disk(self):
        d = tempfile.mkdtemp(prefix="gemma-sib-")
        target = os.path.join(d, "gemma.gguf")
        draft = os.path.join(d, "assistant.gguf")
        write_gguf(target, [("general.architecture", "str", "gemma4"), ("gemma4.embedding_length", "u32", 128)])
        write_gguf(draft, [("general.architecture", "str", "gemma4_mtp"),
                           ("gemma4-assistant.embedding_length_out", "u32", 128)])
        self.assertEqual(os.path.abspath(L.gemma4_assistant_siblings(target)), os.path.abspath(draft))
        self.assertTrue(L.gemma4_draft_measured(target, draft))
        self.assertFalse(L.gemma4_draft_measured(target, target))


class LaunchSurface(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="pxa-launch-surface-")
        self.model = os.path.join(self.d, "model.gguf")
        self.other = os.path.join(self.d, "other.gguf")
        write_gguf(self.model, [("general.architecture", "str", "qwen3")])
        write_gguf(self.other, [("general.architecture", "str", "qwen3")])
        self.roots = [self.d]

    def body(self, **kw):
        b = {"gpus": [0], "model": self.model, "port": 8080}
        b.update(kw)
        return b

    def test_md_is_on_the_allow_list(self):
        self.assertEqual(C.validate_extra_args("-md " + self.other), ["-md", self.other])
        self.assertEqual(C.validate_extra_args("--model-draft=" + self.other), ["--model-draft", self.other])

    def test_draft_and_hot_models_round_trip_into_argv(self):
        req = C.validate_launch(self.body(draft_model=self.other, hot_models=[self.other]),
                                {0}, set(), self.roots, check_model=True, resolve_port=False)
        argv = C.launcher_argv(req)
        self.assertIn("--draft-model", argv)
        self.assertIn(self.other, argv)
        spec = "other=" + self.other
        self.assertIn(spec, argv)
        parsed = L.build_parser().parse_args(argv)
        self.assertEqual(parsed.draft_model, self.other)
        self.assertEqual(parsed.hot_model, [spec])

    def test_hot_model_rejects_a_space_and_the_same_file(self):
        with self.assertRaises(C.Invalid):
            C.validate_launch(self.body(hot_models=[os.path.join(self.d, "has space.gguf")]),
                              {0}, set(), self.roots, check_model=False, resolve_port=False)
        with self.assertRaises(C.Invalid):
            C.validate_launch(self.body(hot_models=[self.model]),
                              {0}, set(), self.roots, check_model=True, resolve_port=False)

    def test_pascal_is_not_offered_volta_fake_cards_are(self):
        pascal = [(0, "P100", 60, 16384, 0, "GPU-fake-0")]
        volta = [(1, "V100", 70, 16384, 0, "GPU-fake-1")]
        self.assertFalse(L.vmm_offered(pascal))
        self.assertFalse(L.vmm_offered([]))
        self.assertTrue(L.vmm_offered(volta))

    def test_engine_refuses_hot_swap_on_pascal(self):
        # The launcher hide above is not on the engine-argument path. The server
        # gate has to refuse a card below Volta itself. ggml stores cc as
        # 100*major+10*minor, so the cut is 700, not the launcher's 70.
        with open(os.path.join(ROOT, "examples", "server", "server.cpp"),
                  encoding="utf-8", errors="replace") as f:
            text = f.read()
        i = text.find("hot swap: --hot-model needs CUDA virtual memory")
        self.assertGreater(i, 0)
        window = text[i:i + 900]
        self.assertIn("cc < 700", window)
        self.assertIn("Pascal", window)

    def test_launcher_refuses_sliding_window_hot_swap(self):
        qwen = os.path.join(self.d, "qwen.gguf")
        gemma = os.path.join(self.d, "gemma.gguf")
        write_gguf(qwen, [("general.architecture", "str", "qwen3")])
        write_gguf(gemma, [("general.architecture", "str", "gemma4"),
                           ("gemma4.attention.sliding_window", "u32", 512)])
        self.assertFalse(L.model_profile(qwen, "gguf").get("swa"))
        self.assertTrue(L.model_profile(gemma, "gguf").get("swa"))
        phrase = "hot swap supports models without sliding-window attention for now; Gemma support is coming"
        env = os.environ.copy()
        env["PXA_LAUNCH_FAKE_GPUS"] = "2x700"
        launch = os.path.join(TOOLS, "pxa-launch.py")

        def explain(model, other):
            r = subprocess.run(
                [sys.executable, launch, "--explain", "--no-interactive", "--no-tui", "--yes",
                 "--gpus", "0", "--model", model, "--hot-model", "beta=" + other],
                env=env, capture_output=True, text=True, timeout=90)
            return r.returncode, r.stdout + r.stderr

        rc, out = explain(qwen, gemma)
        self.assertNotEqual(rc, 0)
        self.assertIn("R-33", out)
        self.assertIn(phrase, out)
        rc, out = explain(gemma, qwen)
        self.assertNotEqual(rc, 0)
        self.assertIn("R-33", out)
        rc, out = explain(qwen, self.other)
        self.assertNotIn("R-33", out)

    def test_engine_refuses_sliding_window_next_to_the_pascal_gate(self):
        with open(os.path.join(ROOT, "examples", "server", "server.cpp"),
                  encoding="utf-8", errors="replace") as f:
            text = f.read()
        i = text.find("cc < 700")
        self.assertGreater(i, 0)
        window = text[i:i + 1800]
        self.assertIn("pxa_gguf_sliding_window", window)
        self.assertIn("hot swap supports models without sliding-window attention for now; Gemma support is coming", window)


class RegisteredModels(unittest.TestCase):
    def test_hot_swap_list_is_kept_and_a_plain_list_is_not(self):
        bodies = {}

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                raw = bodies[self.path].encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                return

        bodies["/v1/models"] = json.dumps({"object": "list", "data": [
            {"id": "main", "pxa_hot_swap": {"active": True}},
            {"id": "other", "pxa_hot_swap": {"active": False}},
        ]})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            rows = C.probe_registered_models(port)
            self.assertEqual(rows, [{"id": "main", "active": True}, {"id": "other", "active": False}])
            bodies["/v1/models"] = json.dumps({"object": "list", "data": [{"id": "only"}]})
            self.assertIsNone(C.probe_registered_models(port))
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
