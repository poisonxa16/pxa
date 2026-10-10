#!/usr/bin/env python3
"""Per-model thinking switch (tools/pxa_thinking.py + tools/pxa_thinking_profiles.json).

  * detection: every family from a real template (models/templates/*.jinja shipped with the engine,
    tests/fixtures/thinking/*.jinja copied from the GGUFs on the PXA box), from architecture + name
    alone (families with no template here), and the unknown-model fallback;
  * rendering (needs jinja2; skipped without it): the switch on and off through apply_request(),
    then the model's own template, the way the engine calls it (enable_thinking always passed):
    the expected tokens appear, or do not;
  * launch flags (launch_args) and the request rewrite (budget, guard, soft switches);
  * PXA Control wiring: settings validation, per-model overrides, plan flags, the engine proxy.

No GPU, no model, no network beyond 127.0.0.1.   python3 tests/test-pxa-thinking.py
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
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
ENGINE_T = os.path.join(ROOT, "models", "templates")
FIX = os.path.join(ROOT, "tests", "fixtures", "thinking")
sys.path.insert(0, TOOLS)
import pxa_thinking as TH  # noqa: E402

try:
    import jinja2  # noqa: F401
    HAVE_JINJA = True
except ImportError:
    HAVE_JINJA = False


def tpl(name):
    for d in (FIX, ENGINE_T):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    raise unittest.SkipTest(f"template {name} not in this tree")


USER = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
SYS_USER = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"}]}

# (template file, arch, general.name, expected family)
TEMPLATE_CASES = [
    ("qwen38-swift-flashnext-hemlock.jinja", "qwen4exp", "Swift1.5-Qwen3.8-Flash-Next", "qwen38-effort"),
    ("qwen38-swift-flashnext-hemlock.jinja", "qwen4exp", "PXA Hemlock 124B-A5B", "qwen38-effort"),
    ("qwen38-swift-flashnext-hemlock.jinja", "qwen35", "Qwen3.8-27B", "qwen38-effort"),
    ("pxa-coder-agent-froggeric.jinja", "qwen35moe", "PXA Coder", "pxa-froggeric"),
    ("gemma4-12b-26b.jinja", "gemma4", "Gemma 4 26B A4B It", "gemma4"),
    ("google-gemma-4-31B-it.jinja", "gemma4", "Gemma 4 31B", "gemma4"),
    ("openai-gpt-oss-120b.jinja", "gpt-oss", "gpt-oss-120b", "gpt-oss"),
    ("muse-glimmer-30b.jinja", "muse-glimmer", "Muse-Glimmer-30B", "muse-glimmer"),
    ("tencent-Hunyuan-V3.jinja", "hy_v3", "Hunyuan V3", "hunyuan-v3"),
    ("glm53-flash.jinja", "glm5next", "GLM 5.3 Flash", "glm5-effort"),
    ("Cohere2MoE.jinja", "cohere2_moe", "Cohere2 MoE", "cohere-reasoning"),
    ("CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja", "cohere2", "c4ai-command-r7b", "command-r7b"),
    ("HuggingFaceTB-SmolLM3-3B.jinja", "smollm3", "SmolLM3-3B", "smollm3"),
    ("deepseek-ai-DeepSeek-V3.1.jinja", "deepseek2", "DeepSeek-V3.1", "deepseek-hybrid"),
    ("deepseek-ai-DeepSeek-V3.2.jinja", "deepseek2", "DeepSeek-V3.2", "deepseek-hybrid"),
    ("deepseek-v4-flash.jinja", "deepseek4", "DeepSeek V4 Flash", "deepseek-hybrid"),
    ("Kimi-K2-Thinking.jinja", "deepseek2", "Kimi-K2-Thinking", "kimi-thinking"),
    ("Apriel-1.6-15b-Thinker-fixed.jinja", "llama", "Apriel-1.6-15b-Thinker", "apriel-thinker"),
    ("mistralai-Ministral-3-14B-Reasoning-2512.jinja", "mistral3", "Ministral-3-14B-Reasoning", "magistral"),
    ("ibm-granite-granite-3.3-2B-Instruct.jinja", "granite", "granite-3.3-2b-instruct", "granite-thinking"),
    ("llama31-nemotron-nano-8b-v1.jinja", "llama", "Llama 3.1 Nemotron Nano 8B v1", "nemotron-detailed"),
    ("GLM-4.6.jinja", "glm4moe", "GLM-4.6", "glm-hybrid"),
    ("GLM-4.7-Flash.jinja", "glm4moe", "GLM-4.7-Flash", "glm-hybrid"),
    ("Qwen-Qwen3-0.6B.jinja", "qwen3", "Qwen3-0.6B", "qwen-hybrid"),
    ("Qwen3.5-4B.jinja", "qwen35", "Qwen3.5-4B", "qwen-hybrid"),
    ("qwen35-0.8b.jinja", "qwen35", "Qwen3.5-0.8B", "qwen-hybrid"),
    ("qwen36-fusion2.jinja", "qwen35moe", "PXA-Fusion2-35B", "qwen-hybrid"),
    ("ornith-1.5-35b.jinja", "qwen35moe", "Ornith-1.5-35B", "qwen-hybrid"),
    ("laguna.jinja", "laguna", "Laguna Abliterated", "qwen-hybrid"),
    ("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja", "nemotron_h", "Nemotron-3-Nano-30B-A3B", "qwen-hybrid"),
    ("MiroThinker.jinja", "qwen3moe", "MiroThinker", "qwen-hybrid"),
    ("deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja", "llama", "DeepSeek-R1-Distill-Llama-8B", "deepseek-r1"),
    ("llama-cpp-deepseek-r1.jinja", "deepseek2", "DeepSeek-R1", "deepseek-r1"),
    ("stepfun-ai-Step-3.5-Flash.jinja", "step35", "Step-3.5-Flash", "deepseek-r1"),
    ("Apertus-8B-Instruct.jinja", "apertus", "Apertus-8B-Instruct", "generic-kwarg"),
    ("stepfun-ai-Step-3.5-Flash.jinja", None, None, "generic-always"),
    # no thinking: the toggle is hidden
    ("qwen3-30b-a3b-instruct-2507.jinja", "qwen3moe", "Qwen3 30B A3B Instruct 2507", "none"),
    ("qwen3-coder-30b-a3b.jinja", "qwen3moe", "Qwen3 Coder 30B A3B Instruct", "none"),
    ("Qwen3-Coder.jinja", "qwen3moe", "Qwen3-Coder-480B-A35B-Instruct", "none"),
    ("Qwen-Qwen2.5-7B-Instruct.jinja", "qwen2", "Qwen2.5-7B-Instruct", "none"),
    ("meta-llama-Llama-3.1-8B-Instruct.jinja", "llama", "Llama-3.1-8B-Instruct", "none"),
    ("meta-llama-Llama-3.3-70B-Instruct.jinja", "llama", "Llama-3.3-70B-Instruct", "none"),
    ("command-r.jinja", "command-r", "command-r", "none"),
    ("Mistral-Small-3.2-24B-Instruct-2506.jinja", "mistral3", "Mistral-Small-3.2-24B-Instruct-2506", "none"),
    ("unsloth-mistral-Devstral-Small-2507.jinja", "llama", "Devstral-Small-2507", "none"),
    ("moonshotai-Kimi-K2.jinja", "deepseek2", "Kimi-K2-Instruct", "none"),
    ("microsoft-Phi-3.5-mini-instruct.jinja", "phi3", "Phi-3.5-mini-instruct", "none"),
    ("LFM2-8B-A1B.jinja", "lfm2", "LFM2-8B-A1B", "none"),
    ("GigaChat3-10B-A1.8B.jinja", None, None, "none"),
]

# (arch, general.name, expected family): families with no template on this box
HEURISTIC_CASES = [
    ("seed_oss", "Seed-OSS-36B-Instruct", "seed-oss"),
    ("phi3", "Phi-4-reasoning-plus", "phi4-reasoning"),
    ("phi3", "Phi-4-mini-reasoning", "phi4-reasoning"),
    ("phi3", "phi-4", "none"),
    ("minimax-m2", "MiniMax-M2", "minimax-m2"),
    ("mistral3", "Magistral-Small-2509", "magistral"),
    ("llama", "Magistral-Small-2506", "magistral"),
    ("llama", "Hermes-4-70B", "hermes4"),
    ("llama", "Hermes-3-Llama-3.1-8B", "none"),
    ("llama", "Llama-3_3-Nemotron-Super-49B-v1", "nemotron-detailed"),
    ("llama", "Llama-3_3-Nemotron-Super-49B-v1.5", "nemotron-soft"),
    ("nemotron_h", "NVIDIA-Nemotron-Nano-9B-v2", "nemotron-soft"),
    ("qwen3moe", "Qwen3-30B-A3B-Thinking-2507", "deepseek-r1"),
    ("qwen3moe", "Qwen3-235B-A22B", "qwen-hybrid"),
    ("qwen3moe", "Qwen3-30B-A3B-Instruct-2507", "none"),
    ("qwen3", "Qwen3-8B", "qwen-hybrid"),
    ("qwen3next", "Qwen3-Next-80B-A3B-Instruct", "none"),
    ("qwen3next", "Qwen3-Next-80B-A3B-Thinking", "deepseek-r1"),
    ("qwen3vlmoe", "Qwen3-VL-30B-A3B-Instruct", "none"),
    ("qwen2", "QwQ-32B", "deepseek-r1"),
    ("qwen2", "DeepSeek-R1-Distill-Qwen-14B", "deepseek-r1"),
    ("qwen2", "Qwen2.5-Coder-32B-Instruct", "none"),
    ("llama", "DeepSeek-R1-Distill-Llama-70B", "deepseek-r1"),
    ("deepseek2", "DeepSeek-R1-0528", "deepseek-r1"),
    ("deepseek2", "DeepSeek-V3-0324", "none"),
    ("deepseek2", "DeepSeek-V3.1-Terminus", "deepseek-hybrid"),
    ("deepseek2", "Kimi-K2-Thinking", "kimi-thinking"),
    ("deepseek2", "Kimi-K2-Instruct-0905", "none"),
    ("gpt-oss", "gpt-oss-20b", "gpt-oss"),
    ("gemma4", "Gemma 4 12B It", "gemma4"),
    ("gemma4_mtp", "Gemma 4 26B A4B It Assistant", "gemma4"),
    ("gemma3", "gemma-3-27b-it", "none"),
    ("gemma2", "gemma-2-9b-it", "none"),
    ("llama4", "Llama-4-Scout-17B-16E-Instruct", "none"),
    ("llama", "Meta-Llama-3.1-8B-Instruct", "none"),
    ("glm4moe", "GLM-4.5-Air", "glm-hybrid"),
    ("glm4", "GLM-Z1-32B-0414", "deepseek-r1"),
    ("glm4", "GLM-4-32B-0414", "none"),
    ("glm5next", "GLM 5.3 Flash", "glm5-effort"),
    ("granite", "granite-3.3-8b-instruct", "granite-thinking"),
    ("granitemoe", "granite-4.0-h-small", "none"),
    ("command-r", "c4ai-command-r-plus-08-2024", "none"),
    ("cohere2", "c4ai-command-r7b-12-2024", "command-r7b"),
    ("smollm3", "SmolLM3-3B", "smollm3"),
    ("step35", "Step-3.5-Flash", "deepseek-r1"),
    ("ernie4_5-moe", "ERNIE-4.5-21B-A3B-Thinking", "deepseek-r1"),
    ("ernie4_5-moe", "ERNIE-4.5-21B-A3B-PT", "none"),
    ("bailingmoe2", "Ring-mini-2.0", "deepseek-r1"),
    ("bailingmoe2", "Ling-mini-2.0", "none"),
    ("hunyuan-moe", "Hunyuan-A13B-Instruct", "qwen-hybrid"),
    ("hy_v3", "Hunyuan-V3", "hunyuan-v3"),
    ("muse-glimmer", "Muse-Glimmer-30B", "muse-glimmer"),
    ("qwen35", "ThinkingCap-Qwen3.8-27B", "qwen38-effort"),
    ("qwen35", "Swift-1.5-Qwen3.8-27B", "qwen38-effort"),
    ("qwen4exp", "Qwen3.8 Flash Next", "qwen38-effort"),
    ("qwen4exp", "PXA Hemlock 124B-A5B", "qwen38-effort"),
    ("qwen35moe", "Ornith-1.5-35B", "qwen-hybrid"),
    ("qwen35moe", "PXA-Fusion4-35B", "qwen-hybrid"),
    ("qwen35", "PXA Agent", "qwen-hybrid"),
    ("laguna", "Laguna", "qwen-hybrid"),
    ("deepseek4", "DeepSeek V4 Flash", "deepseek-hybrid"),
    ("refact", "Refact-1.6B", "unknown"),
    (None, None, "unknown"),
]


class Table(unittest.TestCase):
    def test_table_is_well_formed(self):
        t = TH.load_table()
        ids = [f["id"] for f in t["families"]]
        self.assertEqual(len(ids), len(set(ids)))
        for f in t["families"]:
            self.assertIn(f["mechanism"], ("kwarg", "kwarg+effort", "effort", "system", "budget-kwarg", "always",
                                           "none", "unknown"), f["id"])
            self.assertIn(f["budget"]["enforce"], ("engine", "template", "guard", "none"), f["id"])
            if f["mechanism"] not in ("none", "unknown"):
                self.assertGreater(f["budget"]["suggested"], 0, f["id"])
                self.assertIsNotNone(f["on"], f["id"])
            if f.get("levels"):
                self.assertIn(f["levels"]["default"], f["levels"]["values"], f["id"])
        self.assertIn("SUGGESTIONS", t["note"])

    def test_every_engine_arch_resolves(self):
        # every architecture the engine registers (src/llama-arch.cpp) gets a profile without an error
        src = os.path.join(ROOT, "src", "llama-arch.cpp")
        if not os.path.isfile(src):
            self.skipTest("src/llama-arch.cpp not in this tree")
        import re
        with open(src, encoding="utf-8") as f:
            archs = re.findall(r'\{\s*LLM_ARCH_[A-Z0-9_]+,\s*"([^"]+)"\s*\}', f.read())
        self.assertGreater(len(archs), 50)
        for a in archs:
            p = TH.detect(arch=a, name=a)
            self.assertIn("family", p)


class Detection(unittest.TestCase):
    def test_template_families(self):
        bad = []
        for fn, arch, name, want in TEMPLATE_CASES:
            try:
                t = tpl(fn)
            except unittest.SkipTest:
                continue
            got = TH.detect(arch=arch, name=name, template=t)["family"]
            if got != want:
                bad.append(f"{fn} ({arch}, {name}): {got} != {want}")
        self.assertEqual(bad, [])

    def test_heuristic_families(self):
        bad = []
        for arch, name, want in HEURISTIC_CASES:
            p = TH.detect(arch=arch, name=name)
            if p["family"] != want:
                bad.append(f"({arch}, {name}): {p['family']} != {want}")
        self.assertEqual(bad, [])

    def test_order_template_beats_heuristic(self):
        # a qwen35 file whose template has no switch at all: the name says Qwen3.8, the template wins
        p = TH.detect(arch="qwen35", name="Qwen3.8-27B", template=tpl("Qwen-Qwen2.5-7B-Instruct.jinja"))
        self.assertEqual(p["family"], "none")
        self.assertTrue(any("does not read enable_thinking" in n for n in p["notes"]), p["notes"])

    def test_unknown_arch_with_thinking_template_is_inspected(self):
        p = TH.detect(arch="llama", name="Some-Custom-Merge", template=tpl("Qwen-Qwen3-0.6B.jinja"))
        self.assertEqual(p["family"], "qwen-hybrid")          # template signature first
        p = TH.detect(arch="brand-new-arch", name="X", template=tpl("Apertus-8B-Instruct.jinja"))
        self.assertEqual((p["family"], p["detected_by"], p["supported"]), ("generic-kwarg", "template inspection", True))
        p = TH.detect(arch="brand-new-arch", name="X")
        self.assertEqual((p["family"], p["supported"]), ("unknown", False))

    def test_generic_inspection(self):
        cases = {
            "{% if enable_thinking %}[REASON]{% endif %}": "generic-kwarg",
            "{% if thinking is defined and thinking %}x{% endif %}{{ add_generation_prompt }}": "generic-thinking-kwarg",
            "{% for m in messages %}{{ m.thinking }}{% endfor %}": "none",            # a message field, not a switch
            "Reasoning: {{ reasoning_effort }}{% if reasoning_effort in ['lo', 'hi'] %}{% endif %}": "generic-effort",
            "{% if add_generation_prompt %}<|assistant|><think>\n{% endif %}": "generic-always",
            "{% for m in messages %}{{ m.content.split('</think>')[-1] }}{% endfor %}{% if add_generation_prompt %}A:{% endif %}": "none",
        }
        for t, want in cases.items():
            self.assertEqual(TH.detect(template=t)["family"], want, t)
        p = TH.detect(template="Reasoning: {{ reasoning_effort }}{% if reasoning_effort in ['lo', 'hi'] %}{% endif %}")
        self.assertEqual(p["levels"]["values"], ["lo", "hi"])

    def test_budget_suggestions(self):
        p = TH.detect(arch="qwen35", name="Qwen3.8-27B", size_label="27B")
        self.assertEqual(p["budget"]["suggested"], 8192)
        self.assertTrue(p["budget"]["is_suggestion"])
        self.assertEqual(TH.detect(arch="qwen3", name="Qwen3-4B", size_label="4B")["budget"]["suggested"], 2048)
        self.assertEqual(TH.detect(arch="qwen3", name="Qwen3-4B", params=4.0e9)["budget"]["suggested"], 2048)
        self.assertEqual(TH.detect(arch="gpt-oss", name="gpt-oss-120b", size_label="120B")["budget"]["enforce"], "guard")
        self.assertEqual(TH.params_b(size_label="124B-A5B"), 124)
        self.assertEqual(TH.params_b(size_label="8x7B"), 56)
        self.assertEqual(TH.params_b(size_label="600M"), 0.6)


@unittest.skipUnless(HAVE_JINJA, "jinja2 not installed: rendering tests skipped")
class Rendering(unittest.TestCase):
    """the switch, through apply_request(), through the model's own template."""

    def prompt(self, fn, mode, body=USER, level=None, fallback=False, arch=None, name=None, budget=-1):
        t = tpl(fn)
        p = TH.detect(arch=arch, name=name, template=t)
        b, notes = TH.apply_request(body, p, mode, budget=budget, level=level, use_fallback=fallback)
        return TH.render_prompt(t, b), b, p

    def test_qwen38(self):
        on, _, _ = self.prompt("qwen38-swift-flashnext-hemlock.jinja", "on")
        off, _, _ = self.prompt("qwen38-swift-flashnext-hemlock.jinja", "off")
        low, b, _ = self.prompt("qwen38-swift-flashnext-hemlock.jinja", "on", level="low")
        self.assertTrue(on.endswith("<|im_start|>assistant\n<think>\n"), on[-60:])
        self.assertIn("Reasoning effort is set to xhigh", on)
        self.assertTrue(off.endswith("<think>\n\n</think>\n\n"), off[-60:])
        self.assertNotIn("Reasoning effort", off)
        self.assertIn("Reasoning effort is set to low", low)
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "low"})

    def test_qwen_hybrid_kwarg_and_soft_switch(self):
        on, _, _ = self.prompt("Qwen-Qwen3-0.6B.jinja", "on")
        off, _, _ = self.prompt("Qwen-Qwen3-0.6B.jinja", "off")
        self.assertFalse(on.rstrip().endswith("</think>"), on[-40:])
        self.assertTrue(off.endswith("<think>\n\n</think>\n\n"), off[-40:])
        soft, b, _ = self.prompt("Qwen-Qwen3-0.6B.jinja", "off", fallback=True)
        self.assertTrue(b["messages"][-1]["content"].endswith(" /no_think"))
        self.assertIn("/no_think<|im_end|>", soft)

    def test_qwen35_family_and_derivatives(self):
        for fn in ("Qwen3.5-4B.jinja", "qwen35-0.8b.jinja", "qwen36-fusion2.jinja", "ornith-1.5-35b.jinja"):
            on, _, _ = self.prompt(fn, "on")
            off, _, _ = self.prompt(fn, "off")
            self.assertTrue(on.endswith("<think>\n"), (fn, on[-40:]))
            self.assertTrue(off.rstrip().endswith("</think>"), (fn, off[-40:]))

    def test_pxa_froggeric(self):
        on, _, _ = self.prompt("pxa-coder-agent-froggeric.jinja", "on")
        off, _, _ = self.prompt("pxa-coder-agent-froggeric.jinja", "off")
        self.assertTrue(on.endswith("<think>\n"), on[-40:])
        self.assertTrue(off.endswith("<think>\n\n</think>\n\n"), off[-40:])
        tag, b, _ = self.prompt("pxa-coder-agent-froggeric.jinja", "off", body=SYS_USER, fallback=True)
        self.assertTrue(b["messages"][0]["content"].startswith("<|think_off|>"))
        self.assertNotIn("<|think_off|>", tag)       # the template consumes the tag
        self.assertTrue(tag.endswith("<think>\n\n</think>\n\n"), tag[-40:])

    def test_gemma4(self):
        for fn in ("gemma4-12b-26b.jinja", "google-gemma-4-31B-it.jinja"):
            on, _, _ = self.prompt(fn, "on")
            off, _, _ = self.prompt(fn, "off")
            self.assertIn("<|think|>", on)
            self.assertFalse(on.endswith("<|channel>thought\n<channel|>"), on[-50:])
            self.assertNotIn("<|think|>", off)
            self.assertTrue(off.endswith("<|channel>thought\n<channel|>"), off[-50:])

    def test_gpt_oss_effort(self):
        on, _, p = self.prompt("openai-gpt-oss-120b.jinja", "on")
        hi, _, _ = self.prompt("openai-gpt-oss-120b.jinja", "on", level="high")
        off, _, _ = self.prompt("openai-gpt-oss-120b.jinja", "off")
        self.assertIn("Reasoning: medium", on)
        self.assertIn("Reasoning: high", hi)
        self.assertIn("Reasoning: low", off)
        self.assertFalse(p["can_disable"] and p["mechanism"] != "effort")

    def test_muse_glimmer_strength(self):
        on, _, _ = self.prompt("muse-glimmer-30b.jinja", "on", level="medium")
        off, _, _ = self.prompt("muse-glimmer-30b.jinja", "off")
        self.assertIn("Reasoning strength: medium.", on)
        self.assertIn("Reasoning strength: low.", off)

    def test_hunyuan_v3(self):
        on, _, _ = self.prompt("tencent-Hunyuan-V3.jinja", "on")
        off, _, _ = self.prompt("tencent-Hunyuan-V3.jinja", "off")
        self.assertTrue(on.endswith("<think:opensource>"), on[-50:])
        self.assertTrue(off.endswith("<think:opensource></think:opensource>"), off[-50:])

    def test_glm5_effort(self):
        on, _, _ = self.prompt("glm53-flash.jinja", "on")
        low, _, _ = self.prompt("glm53-flash.jinja", "on", level="low")
        off, b, _ = self.prompt("glm53-flash.jinja", "off")
        self.assertIn("Reasoning Effort: Max", on)
        self.assertIn("Reasoning Effort: Low", low)
        self.assertIn("Reasoning Effort: Low", off)
        self.assertTrue(on.endswith("<think>"))

    def test_glm_hybrid(self):
        on, _, _ = self.prompt("GLM-4.6.jinja", "on")
        off, _, _ = self.prompt("GLM-4.6.jinja", "off")
        self.assertNotIn("<think></think>", on[-30:])
        self.assertIn("<think></think>", off[-30:])

    def test_cohere(self):
        on, _, _ = self.prompt("Cohere2MoE.jinja", "on")
        off, _, _ = self.prompt("Cohere2MoE.jinja", "off")
        self.assertTrue(on.endswith("<|START_THINKING|>"), on[-40:])
        self.assertTrue(off.endswith("<|START_THINKING|><|END_THINKING|>"), off[-40:])
        on, _, _ = self.prompt("CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja", "on")
        off, _, _ = self.prompt("CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja", "off")
        self.assertFalse(on.endswith("<|END_THINKING|>"))
        self.assertTrue(off.endswith("<|START_THINKING|><|END_THINKING|>"))

    def test_smollm3(self):
        on, _, _ = self.prompt("HuggingFaceTB-SmolLM3-3B.jinja", "on")
        off, _, _ = self.prompt("HuggingFaceTB-SmolLM3-3B.jinja", "off")
        self.assertIn("Reasoning Mode: /think", on)
        self.assertIn("Reasoning Mode: /no_think", off)
        self.assertTrue(off.endswith("<think>\n\n</think>\n"))

    def test_deepseek_hybrid(self):
        for fn, arch, name in (("deepseek-ai-DeepSeek-V3.1.jinja", "deepseek2", "DeepSeek-V3.1"),
                               ("deepseek-ai-DeepSeek-V3.2.jinja", "deepseek2", "DeepSeek-V3.2"),
                               ("deepseek-ai-DeepSeek-V4.jinja", "deepseek4", "DeepSeek-V4"),
                               ("deepseek-v4-flash.jinja", "deepseek4", "DeepSeek V4 Flash")):
            on, _, _ = self.prompt(fn, "on", arch=arch, name=name)
            off, _, _ = self.prompt(fn, "off", arch=arch, name=name)
            self.assertTrue(on.endswith("<think>"), (fn, on[-30:]))
            self.assertTrue(off.endswith("</think>"), (fn, off[-30:]))

    def test_always_think_off_is_an_empty_think_prefill(self):
        # (template, the prompt must end with) - only the close when the generation prompt opened it already
        cases = (("deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja", "<｜Assistant｜><think>\n\n</think>\n\n"),
                 ("llama-cpp-deepseek-r1.jinja", "<｜Assistant｜><think>\n\n</think>\n\n"),
                 ("stepfun-ai-Step-3.5-Flash.jinja", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                 ("Kimi-K2-Thinking.jinja", "<|im_assistant|>assistant<|im_middle|><think></think>"),
                 ("mistralai-Ministral-3-14B-Reasoning-2512.jinja", "[/INST][THINK][/THINK]"),
                 ("Apriel-1.6-15b-Thinker-fixed.jinja", "Here are my reasoning steps:\n\n[BEGIN FINAL RESPONSE]\n"),
                 ("glm53-flash.jinja", "<|assistant|><think></think>"),
                 ("openai-gpt-oss-120b.jinja",
                  "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"))
        for fn, tail in cases:
            on, bon, p = self.prompt(fn, "on")
            off, boff, _ = self.prompt(fn, "off")
            self.assertEqual(p["off_method"], "empty-think-prefill", fn)
            self.assertTrue(off.endswith(tail), (fn, off[-80:]))
            self.assertEqual(boff["messages"][-1]["role"], "assistant", fn)
            self.assertIs(boff["chat_template_kwargs"]["enable_thinking"], False, fn)   # the engine needs it
            self.assertNotIn("thinking_budget_tokens", boff, fn)
            self.assertNotEqual(on, off, fn)
            self.assertNotIn("thinking_budget_tokens", bon, fn)  # budget -1 = unlimited: nothing sent
            self.assertTrue(p["partial_off"], fn)
            self.assertTrue(TH.verify_render(tpl(fn), p)["off_closes_thought"], fn)

    def test_prefill_fallback_is_the_reasoning_budget(self):
        for fn in ("deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja", "Kimi-K2-Thinking.jinja",
                   "mistralai-Ministral-3-14B-Reasoning-2512.jinja"):
            off, b, p = self.prompt(fn, "off", fallback=True)
            self.assertEqual(b.get("thinking_budget_tokens"), 0, fn)
            self.assertEqual(b["messages"][-1]["role"], "user", fn)
        # a request that already ends with an assistant message cannot take a prefill: the next method is used
        p = TH.detect(template=tpl("deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja"))
        body = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Sure,"}]}
        b, notes = TH.apply_request(body, p, "off")
        self.assertEqual((b["thinking_budget_tokens"], b["messages"]), (0, body["messages"]))
        # gpt-oss: the fallback is the lowest effort, no prefill
        off, b, _ = self.prompt("openai-gpt-oss-120b.jinja", "off", fallback=True)
        self.assertIn("Reasoning: low", off)
        self.assertTrue(off.endswith("<|start|>assistant"), off[-40:])

    def test_soft_tags(self):
        # the kwarg is preferred; the fallback is the soft tag in the last user message
        off, b, p = self.prompt("Qwen-Qwen3-0.6B.jinja", "off", fallback=True)
        self.assertEqual(p["methods"]["off"], ["template-kwarg", "soft-tag"])   # no prefill: the kwarg closes it
        self.assertNotIn("chat_template_kwargs", b)
        self.assertTrue(b["messages"][-1]["content"].endswith(" /no_think"))
        on, b, _ = self.prompt("Qwen-Qwen3-0.6B.jinja", "on", fallback=True)
        self.assertTrue(b["messages"][-1]["content"].endswith(" /think"))
        off, b, p = self.prompt("GLM-4.6.jinja", "off", fallback=True)
        self.assertTrue(b["messages"][-1]["content"].endswith(" /nothink"))
        off, b, _ = self.prompt("HuggingFaceTB-SmolLM3-3B.jinja", "off", fallback=True)
        self.assertTrue(b["messages"][0]["content"].startswith("/no_think"))
        # a family switched only by a tag: Nemotron v1.5 (no template here: by name)
        p = TH.detect(arch="llama", name="Llama-3_3-Nemotron-Super-49B-v1.5")
        b, _ = TH.apply_request(USER, p, "off")
        self.assertEqual(b["messages"][0], {"role": "system", "content": "/no_think"})
        self.assertEqual(p["methods"]["off"], ["soft-tag", "empty-think-prefill"])

    def test_methods_recorded_and_ordered(self):
        order = ["template-kwarg", "soft-tag", "empty-think-prefill", "reasoning-budget", "effort-level"]
        for f in TH.load_table()["families"]:
            ms = f["methods"]["off"]
            idx = [order.index(m) for m in ms]
            self.assertEqual(idx, sorted(idx), f["id"])     # kwarg, then soft tag, then prefill, then budget 0
            if f["mechanism"] not in ("none", "unknown"):
                self.assertTrue(ms, f["id"])
                self.assertTrue(f["methods"]["on"], f["id"])
            if "empty-think-prefill" in ms:
                self.assertTrue(f["prefill"]["close"], f["id"])
            if "reasoning-budget" in ms:
                self.assertIn("empty-think-prefill", ms, f["id"])   # budget 0 is never the first choice

    def test_magistral_reasoning_prompt(self):
        on, _, _ = self.prompt("mistralai-Ministral-3-14B-Reasoning-2512.jinja", "on")
        self.assertIn("[THINK]", on)

    def test_granite_kwarg(self):
        on, _, _ = self.prompt("ibm-granite-granite-3.3-2B-Instruct.jinja", "on")
        off, _, _ = self.prompt("ibm-granite-granite-3.3-2B-Instruct.jinja", "off")
        self.assertIn("thought process", on)
        self.assertNotIn("thought process", off)

    def test_nemotron_v1_system_line(self):
        on, b, _ = self.prompt("llama31-nemotron-nano-8b-v1.jinja", "on", arch="llama", name="Llama 3.1 Nemotron Nano 8B v1")
        off, _, _ = self.prompt("llama31-nemotron-nano-8b-v1.jinja", "off", body=SYS_USER, arch="llama",
                                name="Llama 3.1 Nemotron Nano 8B v1")
        self.assertIn("system<|end_header_id|>\n\ndetailed thinking on<|eot_id|>", on)
        self.assertIn("detailed thinking off\n\nBe brief.", off)
        self.assertNotIn("chat_template_kwargs", b)

    def test_laguna_generation_tag(self):
        on, _, _ = self.prompt("laguna.jinja", "on")
        off, _, _ = self.prompt("laguna.jinja", "off")
        self.assertTrue(on.endswith("<think>"), on[-30:])
        self.assertFalse(off.endswith("<think>"), off[-30:])

    def test_no_thinking_models_unchanged(self):
        for fn in ("meta-llama-Llama-3.1-8B-Instruct.jinja", "Qwen-Qwen2.5-7B-Instruct.jinja",
                   "qwen3-30b-a3b-instruct-2507.jinja", "qwen3-coder-30b-a3b.jinja", "command-r.jinja"):
            t = tpl(fn)
            p = TH.detect(template=t)
            b, notes = TH.apply_request(USER, p, "off", budget=4096)
            self.assertEqual(b, USER, fn)
            self.assertFalse(p["supported"])

    def test_verify_render(self):
        for fn, differs in (("qwen38-swift-flashnext-hemlock.jinja", True), ("gemma4-12b-26b.jinja", True),
                            ("openai-gpt-oss-120b.jinja", True), ("deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja", True)):
            t = tpl(fn)
            v = TH.verify_render(t, TH.detect(template=t))
            self.assertTrue(v["ok"], (fn, v))
            self.assertEqual(v["differs"], differs, fn)


class RequestRewrite(unittest.TestCase):
    def test_budget_engine_and_answer_room(self):
        p = TH.detect(arch="qwen3", name="Qwen3-32B")
        b, notes = TH.apply_request({"messages": [], "max_tokens": 512}, p, "on", budget=2048)
        self.assertEqual(b["thinking_budget_tokens"], 2048)
        self.assertEqual(b["max_tokens"], 2048 + 1024)
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True})
        b, _ = TH.apply_request({"messages": [], "max_tokens": 9000}, p, "on", budget=2048)
        self.assertEqual(b["max_tokens"], 9000)
        b, _ = TH.apply_request({"messages": []}, p, "on", budget=-1)       # -1 = no cap: the server's own
        self.assertNotIn("thinking_budget_tokens", b)                        # default stays in force
        b, _ = TH.apply_request({"messages": []}, p, "auto")            # auto: nothing forced
        self.assertNotIn("thinking_budget_tokens", b)
        self.assertNotIn("chat_template_kwargs", b)

    def test_guard_only_family(self):
        p = TH.detect(arch="gpt-oss", name="gpt-oss-20b")
        b, notes = TH.apply_request({"messages": []}, p, "on", budget=2048)
        self.assertEqual(b["max_tokens"], 3072)
        self.assertNotIn("thinking_budget_tokens", b)
        self.assertTrue(any("capped" in n for n in notes))

    def test_seed_oss_native_budget(self):
        p = TH.detect(arch="seed_oss", name="Seed-OSS-36B-Instruct")
        b, _ = TH.apply_request({"messages": []}, p, "on", budget=1024)
        self.assertEqual(b["chat_template_kwargs"]["thinking_budget"], 1024)
        b, _ = TH.apply_request({"messages": []}, p, "off")
        self.assertEqual(b["chat_template_kwargs"]["thinking_budget"], 0)

    def test_user_kwargs_kept_and_bad_level(self):
        p = TH.detect(arch="qwen4exp", name="Qwen3.8 Flash Next")
        b, notes = TH.apply_request({"messages": [], "chat_template_kwargs": {"x": 1}}, p, "on", level="ultra")
        self.assertEqual(b["chat_template_kwargs"], {"x": 1, "enable_thinking": True})
        self.assertTrue(any("ignored" in n for n in notes))

    def test_settings_validation(self):
        self.assertEqual(TH.clean_settings(None), {"mode": "auto", "budget": None, "level": None})
        self.assertEqual(TH.clean_settings({"mode": "on", "budget": "2048"})["budget"], 2048)
        for bad in ({"mode": "maybe"}, {"budget": -2}, {"budget": 10 ** 7}, {"budget": True}, {"level": "x y"}, "on"):
            with self.assertRaises(TH.Invalid, msg=repr(bad)):
                TH.clean_settings(bad)


class LaunchArgs(unittest.TestCase):
    def la(self, arch, name, **st):
        return TH.launch_args(TH.detect(arch=arch, name=name), st)

    def test_flags(self):
        self.assertEqual(self.la("qwen3", "Qwen3-8B", mode="off")[0], ["--reasoning", "off"])
        self.assertEqual(self.la("qwen3", "Qwen3-8B", mode="on", budget=2048)[0],
                         ["--reasoning", "on", "--reasoning-budget", "2048"])
        self.assertEqual(self.la("qwen3", "Qwen3-8B", mode="auto")[0], [])
        self.assertEqual(self.la("gemma4", "Gemma 4 12B It", mode="on")[0], ["--reasoning", "on"])
        self.assertEqual(self.la("qwen4exp", "Qwen3.8 Flash Next", mode="on", level="medium")[0],
                         ["--reasoning", "on", "--chat-template-kwargs", '{"reasoning_effort":"medium"}'])
        self.assertEqual(self.la("gpt-oss", "gpt-oss-120b", mode="on", level="high")[0],
                         ["--chat-template-kwargs", '{"reasoning_effort":"high"}'])
        self.assertEqual(self.la("gpt-oss", "gpt-oss-120b", mode="off")[0],
                         ["--chat-template-kwargs", '{"reasoning_effort":"low"}'])
        self.assertEqual(self.la("deepseek2", "DeepSeek-V3.1", mode="off")[0],
                         ["--reasoning", "off", "--chat-template-kwargs", '{"thinking":false}'])
        self.assertEqual(self.la("deepseek2", "DeepSeek-R1-0528", mode="off")[0], ["--reasoning-budget", "0"])
        self.assertEqual(self.la("seed_oss", "Seed-OSS-36B", mode="on", budget=1024)[0],
                         ["--chat-template-kwargs", '{"thinking_budget":1024}'])
        args, notes = self.la("llama", "Llama-3_3-Nemotron-Super-49B-v1", mode="on")
        self.assertEqual(args, [])
        self.assertTrue(any("only works per request" in n for n in notes))
        args, notes = self.la("deepseek2", "DeepSeek-R1-0528", mode="off")
        self.assertTrue(any("not yet measured live" in n for n in notes))
        self.assertEqual(self.la("gpt-oss", "gpt-oss-120b", mode="off", level="high")[0],
                         ["--chat-template-kwargs", '{"reasoning_effort":"low"}'])      # off ignores the level
        self.assertEqual(self.la("llama", "Llama-3.1-8B-Instruct", mode="on")[0], [])


# ---------------------------------------------------------------------------------------------
# PXA Control wiring
# ---------------------------------------------------------------------------------------------
TMP = tempfile.mkdtemp(prefix="pxa-thinking-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_MODELS_DIR", None)
os.environ["PXA_CONTROL_DISCOVER"] = "0"
MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)


def write_gguf(path, kv):
    """a GGUF with only a KV block (no tensors): what the launcher's header reader needs."""
    import struct
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, v in kv.items():
        out += s(k) + struct.pack("<I", 8) + s(v)
    with open(path, "wb") as f:
        f.write(out)


class StubEngine(http.server.BaseHTTPRequestHandler):
    seen = []
    props = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        b = json.dumps(self.props).encode() if self.path == "/props" else b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        StubEngine.seen.append(json.loads(self.rfile.read(n)))
        b = b'{"choices":[{"message":{"content":"4"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class ControlWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
        cls.L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.L)
        import pxa_control
        cls.C = pxa_control
        shutil.rmtree(cls.C.config_dir(), ignore_errors=True)
        cls.qwen = os.path.join(MODELS, "Qwen3-8B-test.gguf")
        write_gguf(cls.qwen, {"general.architecture": "qwen3", "general.name": "Qwen3-8B",
                              "tokenizer.chat_template": tpl("Qwen-Qwen3-0.6B.jinja")})
        cls.gemma = os.path.join(MODELS, "gemma4-test.gguf")
        write_gguf(cls.gemma, {"general.architecture": "gemma4", "general.name": "Gemma 4 12B It",
                               "tokenizer.chat_template": tpl("gemma4-12b-26b.jinja")})
        cls.app = cls.C.App(cls.L, port=7777, models_dirs=[MODELS])
        cls.stub = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StubEngine)
        cls.stub.daemon_threads = True
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.srv = cls.C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.stub.shutdown()
        cls.stub.server_close()
        shutil.rmtree(TMP, ignore_errors=True)

    def call(self, path, method="GET", body=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
                                   headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_validate_and_models(self):
        r = self.app.validate({"gpus": [0], "model": self.qwen, "thinking": {"mode": "off", "budget": 1024}},
                              resolve_port=False)
        self.assertEqual(r["thinking"], {"mode": "off", "budget": 1024, "level": None})
        with self.assertRaises(self.C.Invalid):
            self.app.validate({"gpus": [0], "model": self.qwen, "thinking": {"mode": "loud"}}, resolve_port=False)
        # the Models tab column (the stub files have no tensors, so read the facts directly)
        self.assertEqual(self.app._extra_facts(self.qwen)["thinking"]["family"], "qwen-hybrid")
        self.assertEqual(self.app._extra_facts(self.gemma)["thinking"]["family"], "gemma4")

    def test_launch_flags_and_override(self):
        req = self.app.validate({"gpus": [0], "model": self.gemma}, resolve_port=False)
        self.assertEqual(self.app.thinking_launch(req)["args"], [])                  # auto: nothing
        self.app.set_thinking_override(self.gemma, {"mode": "on", "budget": 3000})    # per model, persisted
        self.assertEqual(self.C.load_config()["thinking_models"][self.gemma]["budget"], 3000)
        th = self.app.thinking_launch(req)
        self.assertEqual((th["args"], th["source"]), (["--reasoning", "on", "--reasoning-budget", "3000"], "model"))
        req2 = self.app.validate({"gpus": [0], "model": self.gemma, "thinking": {"mode": "off"}}, resolve_port=False)
        th = self.app.thinking_launch(req2)                                           # the server's own setting wins
        self.assertEqual((th["args"], th["source"]), (["--reasoning", "off"], "server"))
        self.app.set_thinking_override(self.gemma, None)
        self.assertNotIn(self.gemma, self.C.load_config()["thinking_models"])

    def test_build_appends_flags_before_extra_args(self):
        class Cap(object):                       # the launcher's planner, stubbed: a successful plan
            def run(self_, fn, a, rows):
                self_.code, self_.text = 0, ""
                self_.value = (None, ["llama-server", "-m", "x"], {}, "0", None, 4096)
                return self_
        orig = self.L._Capture
        self.L._Capture = Cap
        try:
            req = self.app.validate({"gpus": [0], "model": self.gemma, "thinking": {"mode": "off"},
                                     "extra_args": "--reasoning-format none"}, resolve_port=False)
            cap = self.app._build(req, False)
        finally:
            self.L._Capture = orig
        self.assertEqual(cap.value[1], ["llama-server", "-m", "x", "--reasoning", "off", "--reasoning-format", "none"])
        self.assertEqual(cap.thinking["family"], "gemma4")

    def test_plan_text_shows_the_flags_passed(self):
        text = ("  setting              value                                        source\n"
                "  --jinja              ON                                           MEASURED\n"
                "  --reasoning-format   not passed -> engine default 'deepseek'      MEASURED\n"
                "  --reasoning-budget   not passed (unlimited)                       [INFERRED]\n"
                "  --temp               not passed -> llama-server's built-in def... [INFERRED]\n")
        class Cap(object):
            def run(self_, fn, a, rows):
                self_.code, self_.text = 0, text
                self_.value = (None, ["llama-server"], {}, "0", None, 4096)
                return self_
        orig = self.L._Capture
        self.L._Capture = Cap
        try:
            req = self.app.validate({"gpus": [0], "model": self.qwen, "thinking": {"mode": "on", "budget": 2048},
                                     "extra_args": "--temp 0.6"}, resolve_port=False)
            cap = self.app._build(req, False)
        finally:
            self.L._Capture = orig
        self.assertNotIn("not passed (unlimited)", cap.text)
        self.assertRegex(cap.text, r"\n  --reasoning-budget   2048 +PXA Control \(thinking\)\n")
        self.assertRegex(cap.text, r"\n  --reasoning-budget   .*\n  --reasoning +on +PXA Control \(thinking\)\n")
        self.assertRegex(cap.text, r"\n  --temp +0\.6 +YOURS \(extra args\)\n")
        self.assertIn("--reasoning-format   not passed", cap.text)        # untouched: nobody passes it
        self.assertEqual(cap.value[1], ["llama-server", "--reasoning", "on", "--reasoning-budget", "2048", "--temp", "0.6"])
        # the last one of a flag wins, as on the engine's command line; a table without rows gets a block
        t = self.C.annotate_plan_text(text, [("--reasoning-budget", "4096", "A"), ("--reasoning-budget", "0", "B")])
        self.assertRegex(t, r"--reasoning-budget   0 +B\n")
        t = self.C.annotate_plan_text("no table here", [("--reasoning", "off", "PXA Control (thinking)")])
        self.assertIn("added by PXA Control after the plan", t)
        self.assertIn("--reasoning", t)
        self.assertEqual(self.C._flag_pairs(["--reasoning-budget", "-1", "--no-warmup", "--temp", "0.6"], "s"),
                         [("--reasoning-budget", "-1", "s"), ("--no-warmup", "on", "s"), ("--temp", "0.6", "s")])

    def test_proxy_prefill_for_an_always_think_server(self):
        StubEngine.props = {"chat_template": tpl("deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja")}
        self.app._prof_cache.clear()
        self.app.set_attach_port(self.stub.server_address[1])
        code, h, b = self.call("/api/engine/v1/chat/completions", "POST",
                               {"messages": [{"role": "user", "content": "hi"}], "pxa_thinking": {"mode": "off"}})
        self.assertEqual(code, 200, b)
        sent = StubEngine.seen[-1]
        self.assertEqual(sent["messages"][-1], {"role": "assistant", "content": "\n</think>\n\n"})
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(json.loads(h.get("X-PXA-Thinking"))["family"], "deepseek-r1")
        self.call("/api/engine/v1/chat/completions", "POST",
                  {"messages": [{"role": "user", "content": "hi"}], "pxa_thinking": {"mode": "off", "fallback": True}})
        self.assertEqual((StubEngine.seen[-1]["thinking_budget_tokens"], len(StubEngine.seen[-1]["messages"])), (0, 1))
        self.app._prof_cache.clear()

    def test_api_thinking_for_a_model(self):
        code, _, b = self.call("/api/thinking?model=" + urllib.request.quote(self.qwen))
        self.assertEqual(code, 200, b)
        d = json.loads(b)
        self.assertEqual(d["profile"]["family"], "qwen-hybrid")
        self.assertEqual(d["settings"]["mode"], "auto")
        if HAVE_JINJA:
            self.assertTrue(d["profile"]["verify"]["differs"], d["profile"]["verify"])
        code, _, b = self.call("/api/thinking?model=/etc/passwd.gguf")
        self.assertEqual(code, 400)
        code, _, b = self.call("/api/thinking/override", "POST", {"model": self.qwen, "thinking": {"mode": "off"}})
        self.assertEqual((code, json.loads(b)["saved"]["mode"]), (200, "off"))
        d = json.loads(self.call("/api/thinking?model=" + urllib.request.quote(self.qwen))[2])
        self.assertEqual((d["settings"]["mode"], d["source"]), ("off", "model"))
        self.assertEqual(json.loads(self.call("/api/thinking/table")[2])["families"][0]["id"], "qwen38-effort")

    def test_proxy_rewrites_the_chat_body(self):
        StubEngine.props = {"chat_template": tpl("gemma4-12b-26b.jinja"), "model_path": "/nonexistent/x.gguf"}
        self.app.set_attach_port(self.stub.server_address[1])
        code, h, b = self.call("/api/engine/v1/chat/completions", "POST",
                               {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100,
                                "pxa_thinking": {"mode": "on", "budget": 2048}})
        self.assertEqual(code, 200, b)
        sent = StubEngine.seen[-1]
        self.assertNotIn("pxa_thinking", sent)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual((sent["thinking_budget_tokens"], sent["max_tokens"]), (2048, 3072))
        info = json.loads(h.get("X-PXA-Thinking"))
        self.assertEqual(info["family"], "gemma4")
        # a body without pxa_thinking passes through byte for byte
        self.call("/api/engine/v1/chat/completions", "POST", {"messages": [], "chat_template_kwargs": {"z": 1}})
        self.assertEqual(StubEngine.seen[-1], {"messages": [], "chat_template_kwargs": {"z": 1}})
        # an unknown server (no template): toggle hidden, body unchanged apart from the removed key
        StubEngine.props = {}
        self.app._prof_cache.clear()
        self.call("/api/engine/v1/chat/completions", "POST", {"messages": [], "pxa_thinking": {"mode": "off"}})
        self.assertEqual(StubEngine.seen[-1], {"messages": []})


    def test_proxy_effort_path(self):
        StubEngine.props = {"chat_template": tpl("Qwen-Qwen3-0.6B.jinja"), "model_path": self.qwen}
        self.app._prof_cache.clear()
        self.app.set_attach_port(self.stub.server_address[1])
        url = "/api/engine/v1/chat/completions"
        msg = [{"role": "user", "content": "hi"}]
        # top-level reasoning_effort (the engine ignores it) -> a budget
        self.call(url, "POST", {"messages": msg, "reasoning_effort": "medium"})
        self.assertEqual(StubEngine.seen[-1].get("thinking_budget_tokens"), 4096)
        self.assertNotIn("reasoning_effort", StubEngine.seen[-1])
        orig = self.app.thinking_target
        try:
            prof = self.app.thinking_for_model(self.qwen)
            self.app.thinking_target = lambda q, verify=False: (prof, {"mode": "auto", "budget": None,
                                                                       "level": None, "effort": "low"}, "model", self.qwen)
            code, h, b = self.call(url, "POST", {"messages": msg})
            self.assertEqual(code, 200, b)
            self.assertEqual(StubEngine.seen[-1]["thinking_budget_tokens"], 1024)
            self.assertIn("effort low", json.loads(h.get("X-PXA-Thinking"))["notes"][0])
            # the client's own setting wins
            self.call(url, "POST", {"messages": msg, "chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(StubEngine.seen[-1], {"messages": msg, "chat_template_kwargs": {"enable_thinking": False}})
            # ... unless the admin locks it
            self.app.thinking_target = lambda q, verify=False: (prof, {"mode": "auto", "budget": None, "level": None,
                                                                       "effort": "high", "lock": True}, "model", self.qwen)
            self.call(url, "POST", {"messages": msg, "chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual((StubEngine.seen[-1]["thinking_budget_tokens"],
                              StubEngine.seen[-1]["chat_template_kwargs"]["enable_thinking"]), (16384, True))
        finally:
            self.app.thinking_target = orig
            self.app._prof_cache.clear()


class EffortSelector(unittest.TestCase):
    def setUp(self):
        self.p = TH.detect(arch="qwen3", name="Qwen3-32B")

    def test_default_off_unchanged(self):
        b = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(TH.apply_effort(dict(b), self.p, TH.clean_settings(None)), (b, []))
        self.assertNotIn("effort", TH.clean_settings({"mode": "on"}))

    def test_levels_map_to_budgets(self):
        for eff, want in (("low", 1024), ("medium", 4096), ("high", 16384)):
            b, _ = TH.apply_effort({"messages": [{"role": "user", "content": "hi"}]}, self.p, {"effort": eff})
            self.assertEqual(b["thinking_budget_tokens"], want)
            self.assertEqual(b["chat_template_kwargs"]["enable_thinking"], True)
        b, _ = TH.apply_effort({"messages": [{"role": "user", "content": "hi"}]}, self.p, {"effort": "max"})
        self.assertNotIn("thinking_budget_tokens", b)            # -1 = unlimited
        b, _ = TH.apply_effort({"messages": [{"role": "user", "content": "hi"}]}, self.p, {"effort": "off"})
        self.assertEqual(b["chat_template_kwargs"]["enable_thinking"], False)

    def test_request_wins(self):
        for body in ({"messages": [], "chat_template_kwargs": {"enable_thinking": True}},
                     {"messages": [], "reasoning_effort": "high"},
                     {"messages": [], "thinking_budget_tokens": 50},
                     {"messages": [{"role": "user", "content": "hi /no_think"}]}):
            out, notes = TH.apply_effort(dict(body), self.p, {"effort": "low"})
            self.assertEqual(out, body)
            self.assertIn("not applied", notes[0])

    def test_lock_wins(self):
        body = {"messages": [{"role": "user", "content": "hi /think"}], "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": True, "z": 1}}
        out, notes = TH.apply_effort(body, self.p, {"effort": "off", "lock": True})
        self.assertNotIn("reasoning_effort", out)
        self.assertEqual(out["chat_template_kwargs"], {"z": 1, "enable_thinking": False})
        self.assertEqual(out["messages"][0]["content"], "hi")
        self.assertIn("locked", notes[0])

    def test_effort_level_passed_when_family_has_it(self):
        p = TH.detect(arch="qwen4exp", name="Qwen3.8 Flash Next")
        b, _ = TH.apply_effort({"messages": [{"role": "user", "content": "x"}]}, p, {"effort": "medium"})
        self.assertEqual(b["chat_template_kwargs"].get("reasoning_effort"), "medium")
        self.assertEqual(b["thinking_budget_tokens"], 4096)

    def test_validation(self):
        self.assertEqual(TH.clean_settings({"effort": "high", "lock": True})["effort"], "high")
        for bad in ({"effort": "ultra"}, {"lock": "yes"}):
            with self.assertRaises(TH.Invalid):
                TH.clean_settings(bad)


class ClientEffort(unittest.TestCase):
    def test_top_level_reasoning_effort_maps(self):
        p = TH.detect(arch="qwen3", name="Qwen3-32B")
        b, _ = TH.map_client_effort({"messages": [], "reasoning_effort": "low"}, p)
        self.assertEqual((b["thinking_budget_tokens"], "reasoning_effort" in b), (1024, False))
        b, _ = TH.map_client_effort({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"}, p)
        self.assertEqual(b["chat_template_kwargs"]["enable_thinking"], False)
        b, _ = TH.map_client_effort({"messages": [], "reasoning_effort": "high", "thinking_budget_tokens": 77}, p)
        self.assertEqual(b["thinking_budget_tokens"], 77)
        b, n = TH.map_client_effort({"messages": [], "reasoning_effort": "weird"}, p)
        self.assertIn("not mapped", n[0])
        self.assertEqual(TH.map_client_effort({"messages": []}, p), ({"messages": []}, []))


if __name__ == "__main__":
    unittest.main(verbosity=1)
