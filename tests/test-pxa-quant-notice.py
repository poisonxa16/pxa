#!/usr/bin/env python3
"""Detection: a PXA quant gets no notice, a plain Q4_K_M gets the one line.

No GPU and no GGUF on disk. The function takes a filename plus the same
signals the header reader already returns (tensor types, provenance keys, tier).

    python3 tests/test-pxa-quant-notice.py
"""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

NOTICE = L.STANDARD_GGUF_NOTICE
HREF = "https://huggingface.co/poisonxa"


class NoticeTests(unittest.TestCase):
    def test_sentence_is_the_one_liner(self):
        self.assertEqual(
            NOTICE,
            "This is a standard GGUF quant. PXA runs it, but its speed work targets PXA quants. "
            "Get the PXA version: " + HREF)
        low = NOTICE.lower()
        self.assertNotIn("horrib", low)
        self.assertNotIn("broken", low)

    def test_pxa_filename_no_notice(self):
        for name in ("Qwen3.8-27B-PXQN2.gguf", "model-PXQ4.gguf", "flash-PXQU.gguf",
                     "mix-PXQ_UNIVERSAL.gguf", "weights-PXA4.gguf"):
            self.assertEqual(L.standard_gguf_notice(name, kv={}, tensors=[("blk.0.attn_q.weight", 12, 1)],
                                                    inspected=True), "", name)

    def test_pxa_tensor_type_beats_a_plain_name(self):
        # 260 is PXQN2 in PXQ_GGML_TYPE. The name says Q4_K_M; the tensors are ours.
        self.assertEqual(L.standard_gguf_notice("foo-Q4_K_M.gguf", tensors=[("blk.0.ffn.weight", 260, 1)]), "")
        self.assertTrue(L.is_pxa_quant("plain.gguf", tensors=[("w", 257, 1)]))  # PXQN3

    def test_provenance_kv_no_notice(self):
        self.assertEqual(L.standard_gguf_notice("plain.gguf", kv={"pxa.pxq2.version": 1}), "")
        self.assertEqual(L.standard_gguf_notice("plain.gguf", kv={"pxa.pxqn.encoder": "pxqn"}), "")
        self.assertEqual(L.standard_gguf_notice("plain.gguf", tier="PXQ4"), "")
        self.assertEqual(L.standard_gguf_notice("plain.gguf", tier_kv="PXQ_UNIVERSAL"), "")

    def test_plain_q4_k_m_notices(self):
        n = L.standard_gguf_notice(
            "foo-Q4_K_M.gguf", kv={}, tensors=[("blk.0.attn_q.weight", 12, 1)], inspected=True)
        self.assertEqual(n, NOTICE)
        for name in ("gemma-Q8_0.gguf", "gemma-q4_0.gguf", "model-IQ4_XS.gguf", "x-MXFP4.gguf"):
            self.assertEqual(L.standard_gguf_notice(name), NOTICE, name)

    def test_quantizer_identity_alone_is_not_a_tier(self):
        # pxa.quantizer.* says who wrote the file, not that the weights are a PXA tier.
        n = L.standard_gguf_notice("foo-Q4_K_M.gguf", kv={"pxa.quantizer.name": "pxa"}, inspected=True)
        self.assertEqual(n, NOTICE)

    def test_uninspected_plain_name_stays_quiet(self):
        self.assertEqual(L.standard_gguf_notice("model.gguf"), "")
        self.assertEqual(L.standard_gguf_notice(""), "")

    def test_readme_and_ui_carry_the_same_idea(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        start = readme.index("## Use PXA quants")
        end = readme.index("## Why it is fast", start)
        section = readme[start:end]
        self.assertIn(HREF, section)
        self.assertIn("loads and runs", section)
        self.assertNotIn("horrib", section.lower())
        self.assertNotIn("broken", section.lower())
        self.assertIn("[Use PXA quants](#use-pxa-quants)", readme)
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertIn("function quantNotice", html)
        self.assertIn("x.quant_notice", html)
        self.assertIn("x.notice", html)
        self.assertIn("l-model-note", html)
        self.assertIn('badge("warn", "standard GGUF")', html)
        self.assertIn("stdBadge(x.notice)", html)
        self.assertIn('id="c-qnote"', html)
        self.assertIn("function chatQuantNote", html)
        self.assertIn("function noteForTarget", html)
        self.assertIn("function selectedModelNotice", html)
        self.assertIn("return selectedModelNotice()", html)
        self.assertIn('$("#l-model") && $("#l-model").value', html)
        self.assertIn("#l-think.think,#c-think.think{white-space:normal", html)
        self.assertIn("#l-think.note-only,#c-think.note-only{margin-top:6px;min-height:0", html)
        self.assertIn('box.classList.toggle("note-only", bare)', html)
        self.assertIn("s.quant_notice", html)
        self.assertIn('$("#ag-qnote")', html)
        with open(os.path.join(ROOT, "tools", "pxa_chat", "ui", "agent.js"), encoding="utf-8") as f:
            agent = f.read()
        self.assertIn('id: "ag-qnote"', agent)
        self.assertIn("chatQuantNote()", agent)
        with open(os.path.join(TOOLS, "pxa_control.py"), encoding="utf-8") as f:
            ctl = f.read()
        self.assertIn("standard_gguf_notice", ctl)
        self.assertIn('"notice": notice', ctl)
        self.assertIn("quant_notice", ctl)
        with open(os.path.join(ROOT, "tools", "pxa_chat", "picker.py"), encoding="utf-8") as f:
            pick = f.read()
        self.assertIn("def quant_notice_for", pick)
        self.assertIn('"quant_notice": quant_notice_for', pick)


if __name__ == "__main__":
    unittest.main(verbosity=2)
