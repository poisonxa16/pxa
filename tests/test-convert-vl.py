#!/usr/bin/env python3
"""The converter's vision-language models (converter-vl-archs, FR#16): Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, SmolVLM2 (Idefics3) and
InternVL, each converted twice from a tiny random checkpoint (tests/gen_tiny_vl.py): the text model, and with --mmproj the vision
projector the engine's mtmd clip loader reads.

Checked: the architecture and the keys the engine reads (M-RoPE sections, deepstack count, projector type, window-attention pattern,
scale factor), the exact tensor-name set of every file, and the numbers of the tensors the converter rearranges (the fused QKV
split, the Conv3D patch embedding split in two, the packed and transposed Qwen3-VL-MoE experts). The mmproj files of these
fixtures were checked byte-identical to the upstream llama.cpp converter's output when this was written.

CPU only, no network, about 5 min (the text conversions read a 155k-token tokenizer).      python3 tests/test-convert-vl.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:                                    # the converter's own dependencies; without them there is nothing to test (CTest: SKIP)
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import safetensors  # noqa: F401
except ImportError as e:
    print("SKIP: the converter needs torch, transformers and safetensors (%s)" % e)
    sys.exit(77)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
import gen_tiny_vl as G  # noqa: E402
import gguf  # noqa: E402

TMP = tempfile.mkdtemp(prefix="pxa-convert-vl-")
SRC, OUT = {}, {}


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def convert(kind, mmproj):
    key = (kind, mmproj)
    if key in OUT:
        return OUT[key]
    if kind not in SRC:
        SRC[kind] = G.build(kind, os.path.join(TMP, "src", kind))
    out = os.path.join(TMP, "%s-%s.gguf" % (kind, "mmproj" if mmproj else "text"))
    argv = [sys.executable, os.path.join(ROOT, "convert_hf_to_gguf.py"), os.path.join(TMP, "src", kind), "--outtype", "f16", "--outfile", out]
    if mmproj:
        argv.append("--mmproj")
    p = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    if p.returncode != 0:
        raise AssertionError("convert %s%s failed:\n%s" % (kind, " --mmproj" if mmproj else "", (p.stdout + p.stderr)[-3000:]))
    r = gguf.GGUFReader(out)
    kv = {k: f.contents() for k, f in r.fields.items() if not k.startswith("GGUF.")}
    t = {x.name: x for x in r.tensors}
    OUT[key] = (kv, t)
    return OUT[key]


def arr(t):
    """GGUF tensor -> numpy in torch (HF) order, as float32."""
    return np.asarray(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)])


def text_names(n_layers, bias=False, qk_norm=False, moe=False):
    s = {"token_embd.weight", "output_norm.weight", "output.weight"}
    for i in range(n_layers):
        p = "blk.%d." % i
        s |= {p + x for x in ("attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")}
        if bias:
            s |= {p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias"}
        if qk_norm:
            s |= {p + "attn_q_norm.weight", p + "attn_k_norm.weight"}
        s |= ({p + "ffn_gate_inp.weight", p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight", p + "ffn_down_exps.weight"} if moe else
              {p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"})
    return s


class TextModels(unittest.TestCase):
    def check_text(self, kind, arch, **names):
        kv, t = convert(kind, False)
        self.assertEqual(kv["general.architecture"], arch)
        self.assertEqual(set(t), text_names(G.NL, **names))
        self.assertFalse([n for n in t if n.startswith(("v.", "mm."))])            # the vision tower stays out of the text model
        self.assertEqual(kv["%s.block_count" % arch], G.NL)
        self.assertEqual(kv["%s.embedding_length" % arch], G.H)
        return kv, t

    def test_qwen25vl(self):
        kv, _ = self.check_text("qwen25vl", "qwen2vl", bias=True)
        self.assertEqual(list(kv["qwen2vl.rope.dimension_sections"]), [2, 3, 3, 0])

    def test_qwen3vl(self):
        kv, _ = self.check_text("qwen3vl", "qwen3vl", qk_norm=True)
        self.assertEqual(list(kv["qwen3vl.rope.dimension_sections"]), [2, 3, 3, 0])
        self.assertEqual(kv["qwen3vl.n_deepstack_layers"], 1)

    def test_qwen3vlmoe_experts_are_split_and_transposed(self):
        kv, t = self.check_text("qwen3vlmoe", "qwen3vlmoe", qk_norm=True, moe=True)
        self.assertEqual(kv["qwen3vlmoe.expert_count"], G.E)
        self.assertEqual(kv["qwen3vlmoe.expert_feed_forward_length"], G.EFF)
        self.assertEqual(kv["qwen3vlmoe.n_deepstack_layers"], 1)
        src = SRC["qwen3vlmoe"]
        gu = src["model.language_model.layers.1.mlp.experts.gate_up_proj"]          # {E, H, 2*EFF}
        dn = src["model.language_model.layers.1.mlp.experts.down_proj"]             # {E, EFF, H}
        for name, want in (("blk.1.ffn_gate_exps.weight", gu[..., :G.EFF].transpose(0, 2, 1)),
                           ("blk.1.ffn_up_exps.weight", gu[..., G.EFF:].transpose(0, 2, 1)),
                           ("blk.1.ffn_down_exps.weight", dn.transpose(0, 2, 1))):
            self.assertEqual(t[name].tensor_type, gguf.GGMLQuantizationType.F16, name)     # weights, not forced to F32
            np.testing.assert_allclose(arr(t[name]), want.astype(np.float16).astype(np.float32), err_msg=name)

    def test_smolvlm(self):
        self.check_text("smolvlm", "llama")

    def test_internvl_text_is_its_language_model(self):
        kv, t = self.check_text("internvl", "qwen2", bias=True)
        want = SRC["internvl"]["language_model.model.layers.0.self_attn.v_proj.weight"]
        np.testing.assert_allclose(arr(t["blk.0.attn_v.weight"]), want.astype(np.float16).astype(np.float32))


def vision_names(depth, qkv_fused=False, gate=False, ln_bias=True, ls=False):
    s = set()
    for i in range(depth):
        p = "v.blk.%d." % i
        s |= {p + "attn_qkv.weight", p + "attn_qkv.bias"} if qkv_fused else {p + "attn_%s.%s" % (x, b) for x in "qkv" for b in ("weight", "bias")}
        s |= {p + "attn_out.weight", p + "attn_out.bias", p + "ln1.weight", p + "ln2.weight",
              p + "ffn_up.weight", p + "ffn_up.bias", p + "ffn_down.weight", p + "ffn_down.bias"}
        if ln_bias:
            s |= {p + "ln1.bias", p + "ln2.bias"}
        if gate:
            s |= {p + "ffn_gate.weight", p + "ffn_gate.bias"}
        if ls:
            s |= {p + "ls1.weight", p + "ls2.weight"}
    return s


class Projectors(unittest.TestCase):
    def check_common(self, kind, ptype):
        kv, t = convert(kind, True)
        self.assertEqual(kv["general.architecture"], "clip")
        self.assertEqual(kv["general.type"], "mmproj")
        self.assertEqual(kv["clip.projector_type"], ptype)
        self.assertTrue(kv["clip.has_vision_encoder"])
        self.assertEqual(kv["clip.vision.projection_dim"], G.H)
        self.assertEqual(kv["clip.vision.embedding_length"], G.VH)
        self.assertEqual(kv["clip.vision.block_count"], G.VD)
        self.assertEqual(kv["clip.vision.attention.head_count"], G.VNH)
        self.assertEqual(list(kv["clip.vision.image_mean"]), [0.5, 0.5, 0.5])
        self.assertFalse([n for n in t if n.startswith("blk.") or n.startswith("token_embd")])   # no text tensors
        return kv, t

    def test_qwen25vl(self):
        kv, t = self.check_common("qwen25vl", "qwen2.5vl_merger")
        self.assertTrue(kv["clip.use_silu"])
        self.assertEqual(kv["clip.vision.n_wa_pattern"], 2)
        self.assertEqual(set(t), vision_names(G.VD, gate=True, ln_bias=False) | {
            "v.patch_embd.weight", "v.patch_embd.weight.1", "v.post_ln.weight", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"})
        src = SRC["qwen25vl"]
        qkv = src["visual.blocks.1.attn.qkv.weight"]
        f16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
        np.testing.assert_allclose(arr(t["v.blk.1.attn_k.weight"]), f16(qkv[G.VH:2 * G.VH]))
        pe = src["visual.patch_embed.proj.weight"]                                   # Conv3D {VH, 3, 2, 14, 14} -> two Conv2Ds
        np.testing.assert_allclose(arr(t["v.patch_embd.weight"]), f16(pe[:, :, 0]))
        np.testing.assert_allclose(arr(t["v.patch_embd.weight.1"]), f16(pe[:, :, 1]))

    def test_qwen3vl_and_moe(self):
        for kind in ("qwen3vl", "qwen3vlmoe"):
            kv, t = self.check_common(kind, "qwen3vl_merger")
            self.assertTrue(kv["clip.use_gelu"])
            self.assertEqual(kv["clip.vision.spatial_merge_size"], 2)
            self.assertEqual(kv["clip.vision.image_size"], 64)                     # sqrt(num_position_embeddings) * patch
            self.assertEqual(list(kv["clip.vision.is_deepstack_layers"]), [True] + [False] * (G.VD - 1))
            ds = {"v.deepstack.0.%s.%s" % (a, b) for a in ("norm", "fc1", "fc2") for b in ("weight", "bias")}
            self.assertEqual(set(t), vision_names(G.VD, qkv_fused=True) | ds | {
                "v.patch_embd.weight", "v.patch_embd.weight.1", "v.patch_embd.bias", "v.position_embd.weight",
                "v.post_ln.weight", "v.post_ln.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"})
            self.assertEqual(t["v.position_embd.weight"].tensor_type, gguf.GGMLQuantizationType.F32)

    def test_smolvlm(self):
        kv, t = self.check_common("smolvlm", "idefics3")
        self.assertEqual(kv["clip.vision.projector.scale_factor"], 2)
        self.assertEqual(kv["clip.vision.preproc_image_size"], 128)
        self.assertEqual(set(t), vision_names(G.VD) | {"v.patch_embd.weight", "v.patch_embd.bias", "v.position_embd.weight",
                                                       "v.post_ln.weight", "v.post_ln.bias", "mm.model.fc.weight"})

    def test_internvl(self):
        kv, t = self.check_common("internvl", "internvl")
        self.assertEqual(kv["clip.vision.projector.scale_factor"], 2)
        self.assertEqual((kv["clip.vision.preproc_min_tiles"], kv["clip.vision.preproc_max_tiles"]), (1, 6))
        self.assertEqual(set(t), vision_names(G.VD, ls=True) | {
            "v.class_embd", "v.patch_embd.weight", "v.patch_embd.bias", "v.position_embd.weight",
            "mm.model.mlp.0.weight", "mm.model.mlp.0.bias", "mm.model.mlp.1.weight", "mm.model.mlp.1.bias", "mm.model.mlp.3.weight", "mm.model.mlp.3.bias"})
        self.assertEqual([int(d) for d in t["v.class_embd"].shape], [G.VH, 1, 1])     # as the vision tower has it: not squeezed


class Refusals(unittest.TestCase):
    def test_a_text_only_model_has_nothing_for_mmproj(self):
        d = os.path.join(TMP, "textonly")
        G.build("qwen3vl", d)
        import json
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        cfg = dict(cfg["text_config"], architectures=["Qwen3ForCausalLM"])
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "convert_hf_to_gguf.py"), d, "--mmproj", "--outfile", os.path.join(TMP, "x.gguf")],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
        self.assertEqual(p.returncode, 1)
        self.assertIn("no vision tower", p.stdout + p.stderr)
        self.assertFalse(os.path.exists(os.path.join(TMP, "x.gguf")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
