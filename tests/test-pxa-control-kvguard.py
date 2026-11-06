"""PXA Control and the engine's K-cache guard (PXA_KV_K_GUARD, engine 4.0.4, bug #378). No GPU, no model, loopback only.

Qwen2-family models (GGUF arch qwen2 / qwen2vl / qwen2moe) answer garbage with a 4-bit K cache, so the engine raises such a
K to q8_0 for them (V stays as asked; PXA_KV_K_GUARD=0 keeps the 4-bit K). Control keeps passing the KV the user picked and
says what runs:
  1. kv_k_guard_note(): the rule (arch x KV pick x the =0 lever) and its words;
  2. /api/info names the archs and types (the Launch KV picker's note is computed from them in the page);
  3. the model's arch comes from the file header (_extra_facts), and the launch plan carries the note;
  4. the bench board payload names the K the engine reports in use (/props), not the -ctk text;
  5. the Launch page has the note element and the same words as core.py.

    python3 tests/test-pxa-control-kvguard.py          (wired into CTest as test-pxa-control-kvguard)
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-kvguard-")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402
import pxa_control.core as CORE  # noqa: E402
import pxa_control.routes as R  # noqa: E402

MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)


def tiny_gguf(name, arch):
    """a header-only GGUF (no tensors) whose general.architecture is `arch`"""
    import gguf  # gguf-py from this tree
    p = os.path.join(MODELS, name)
    w = gguf.GGUFWriter(p, arch)
    w.add_name(name)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    return p


QWEN2VL = tiny_gguf("Qwen2.5-VL-7B-Instruct-PXQN5.gguf", "qwen2vl")
QWEN3 = tiny_gguf("Qwen3-8B-PXQ4.gguf", "qwen3")


class Note(unittest.TestCase):
    def test_qwen2_family_with_a_4bit_kv(self):
        for arch in ("qwen2", "qwen2vl", "qwen2moe"):
            n = C.kv_k_guard_note(arch, "q4_0")
            self.assertIsNotNone(n, arch)
            self.assertIn("K cache at q8_0", n)
            self.assertIn("V stays q4_0", n)
            self.assertIn(arch, n)
            self.assertIn("PXA_KV_K_GUARD=0", n)
        self.assertIn("V stays iq4_nl", C.kv_k_guard_note("qwen2", "iq4_nl"))

    def test_no_note_elsewhere(self):
        for arch, kv in (("qwen3", "q4_0"), ("llama", "q4_0"), ("qwen3vl", "q4_0"), ("qwen2vl", "q8_0"),
                         ("qwen2vl", "q5_0"), ("qwen2vl", "f16"), ("qwen2vl", "auto"), (None, "q4_0"), ("qwen2", None)):
            self.assertIsNone(C.kv_k_guard_note(arch, kv), (arch, kv))

    def test_off_switch_says_so(self):
        n = C.kv_k_guard_note("qwen2vl", "q4_0", {"PXA_KV_K_GUARD": "0"})
        self.assertIn("PXA_KV_K_GUARD=0", n)
        self.assertIn("keeps a q4_0 K cache", n)
        self.assertIn("garbage", n)
        self.assertIn("K cache at q8_0", C.kv_k_guard_note("qwen2vl", "q4_0", {"PXA_KV_K_GUARD": "1"}))

    def test_types_are_kv_cache_types(self):
        self.assertEqual(set(CORE.KV_K_GUARD_ARCHS), {"qwen2", "qwen2vl", "qwen2moe"})
        self.assertIn("q4_0", CORE.KV_K_GUARD_TYPES)
        self.assertTrue(set(CORE.KV_K_GUARD_TYPES).isdisjoint({"f16", "q8_0", "q6_0", "q5_0", "auto"}))
        self.assertEqual([k for k in CORE.KV_TYPES if k in CORE.KV_K_GUARD_TYPES], ["q4_0"])   # the picker's one 4-bit choice


class App(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=0, models_dirs=[MODELS])

    def test_info_names_archs_and_types(self):
        d = R.r_info(self.app, {}, {})
        self.assertEqual(d["kv_k_guard"]["archs"], ["qwen2", "qwen2vl", "qwen2moe"])
        self.assertIn("q4_0", d["kv_k_guard"]["types"])

    def test_arch_from_the_header(self):
        self.assertEqual(self.app._extra_facts(QWEN2VL).get("arch"), "qwen2vl")
        self.assertEqual(self.app._extra_facts(QWEN3).get("arch"), "qwen3")

    def _plan(self, model, kv, levers=None):
        req = {"gpus": [0], "model": model, "ctx": 16384, "np": 1, "kv": kv, "sm": "auto", "fa": "on", "port": 18999,
               "mtp": False, "ngram": "auto", "expose": False, "accept_unmeasured": False, "allow_busy": False,
               "levers": levers or {}, "extra_args": [], "engine_dir": None, "thinking": None}
        plan = types.SimpleNamespace(engine="host", notes=["a plan note"], blockers=[])
        cap = types.SimpleNamespace(code=0, value=(plan, ["llama-server", "-m", model], {}, [0], None, 16384), text="",
                                    args=types.SimpleNamespace(sm="layer", np=1, workload="chat"))
        self.app.fit_now = lambda req: {}
        return self.app._plan_dict(req, cap, None)

    def test_plan_carries_the_note(self):
        d = self._plan(QWEN2VL, "q4_0")
        self.assertTrue(d["ok"], d)
        self.assertEqual(d["notes"][0], "a plan note")
        self.assertEqual(d["notes"][-1], C.kv_k_guard_note("qwen2vl", "q4_0"))
        d = self._plan(QWEN2VL, "q4_0", {"PXA_KV_K_GUARD": "0"})
        self.assertIn("keeps a q4_0 K cache", d["notes"][-1])
        for model, kv in ((QWEN3, "q4_0"), (QWEN2VL, "q8_0"), (QWEN2VL, "auto")):
            d = self._plan(model, kv)
            self.assertEqual(d["notes"], ["a plan note"], (model, kv))

    def test_board_payload_names_the_k_in_use(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", QWEN2VL, "-c", "16384", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on"]
        app.seat.req = {"model": QWEN2VL, "gpus": [0], "port": 8999}
        res = {"ts": 1_780_000_000.0, "model": QWEN2VL, "cards": [0], "reps": 3, "greedy512_sha": "ab" * 32, "kv": "q8_0/q4_0",
               "classes": [{"class": "prose", "decode_tps": 30.0, "prefill_tps": 0, "n_prompt": 20, "decode_all": [30],
                            "prefill_all": []}]}
        q = app.score_payload(res, "Zed")["params"]
        self.assertEqual((q["kv_k"], q["kv_v"]), ("q8_0", "q4_0"))     # /props (the guard's q8_0 K), not the -ctk text
        res.pop("kv")
        q = app.score_payload(res, "Zed")["params"]
        self.assertEqual((q["kv_k"], q["kv_v"]), ("q4_0", "q4_0"))     # an engine without /props KV: the flags, as before


class Page(unittest.TestCase):
    def test_launch_page_has_the_note(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            page = f.read()
        self.assertIn('id="l-kv-note"', page)
        self.assertIn("function kvGuardNote()", page)
        self.assertIn('$("#l-kv").addEventListener("change", kvGuardNote);', page)
        # the same words as core.py kv_k_guard_note
        self.assertIn("Qwen2-family model (${m.arch}): the engine keeps the K cache at q8_0, because a ${kv} K breaks these "
                      "models; V stays ${kv}. PXA_KV_K_GUARD=0 turns this off.", page)
        self.assertEqual(C.kv_k_guard_note("qwen2vl", "q4_0"),
                         "Qwen2-family model (qwen2vl): the engine keeps the K cache at q8_0, because a q4_0 K breaks these "
                         "models; V stays q4_0. PXA_KV_K_GUARD=0 turns this off.")


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
