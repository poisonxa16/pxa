"""gguf_to_vllm_test.py — CPU-only test suite for the converter. No GPU, no DGX, no lease.

Run:  python3 gguf_to_vllm_test.py            (from the impl/ directory)

Fixtures live in ``fixtures/`` and are REAL BYTES pulled from the real artifact:

    hdr.bin            first 10,997,184 B of Qwen3.8-27B-PXQ4.gguf — the complete header, KV
                       table and 866-entry tensor directory, with no data section.
    fx_pxq4_gate       2 whole panels of blk.0.attn_gate.weight  (N=128, K=5120)
    fx_pxq4_down       1 whole panel  of blk.0.ffn_down.weight   (N=64,  K=17408)
    fx_pxq4_q          2 whole panels of blk.3.attn_q.weight     (N=128, K=5120)
    fx_q8_0_k          8 rows of blk.3.attn_k.weight             (q8_0)
    fx_q6k_embd        4 rows of token_embd.weight               (q6_K)
    fx_mxfp4_out       8 rows of blk.0.ssm_out.weight            (mxfp4)
    fx_f32_conv1d      all of blk.0.ssm_conv1d.weight            (f32)
    *.f32              the matching output of ``fixtures/oracle``, the standalone build of
                       gguf_to_vllm_oracle.c, which carries the production ggml C VERBATIM.
    refhf/             a header-only stub of the AWQ twin (real safetensors header + real
                       index + real config.json) so the name-map diff runs offline.

The gates that are actually load-bearing are G1 (bit-exactness against the engine's own
decoder) and G3 (the TP repack is a permutation). Both run here, on any machine.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures")
sys.path.insert(0, HERE)

from gguf_to_vllm import dequant_ref as D          # noqa: E402
from gguf_to_vllm import gguf_raw as G             # noqa: E402
from gguf_to_vllm import layout as L               # noqa: E402
from gguf_to_vllm import namemap as NM             # noqa: E402
from gguf_to_vllm import reference as R            # noqa: E402
from gguf_to_vllm import safetensors_io as ST      # noqa: E402

GGUF_TRUE_SIZE = 15_719_771_584
PXQ4_FIXTURES = [("fx_pxq4_gate", 128, 5120), ("fx_pxq4_down", 64, 17408),
                 ("fx_pxq4_q", 128, 5120)]


def blob(tag: str) -> bytes:
    with open(os.path.join(FX, tag), "rb") as f:
        return f.read()


def oracle_f32(tag: str, N: int, K: int) -> np.ndarray:
    return np.fromfile(os.path.join(FX, tag + ".f32"), dtype="<f4").reshape(N, K)


_HDR_CACHE: list = []


def header() -> G.GGUFHeaderOnly:
    """The real 866-tensor directory, parsed once and reused: opening it 20 times in a test run
    is 20 mmaps of a 10 MB file for no benefit."""
    if not _HDR_CACHE:
        _HDR_CACHE.append(G.GGUFHeaderOnly(os.path.join(FX, "hdr.bin"), GGUF_TRUE_SIZE))
    return _HDR_CACHE[0]


# =============================================================================================
class TestLayout(unittest.TestCase):
    def test_panel_and_tensor_bytes_match_ggml_row_size(self):
        # ggml charges row_meta + type_size*K/32 per row; the panel formula must agree exactly,
        # otherwise every offset past the first tensor is wrong.
        for N, K in [(64, 32), (128, 5120), (6144, 5120), (5120, 17408), (12288, 5120)]:
            self.assertEqual(L.tensor_bytes(N, K), N * (2 + 17 * K // 32))
            self.assertEqual(L.panel_bytes(K), 128 + (K // 32) * 1088)

    def test_bits_per_weight_matches_documented_formula(self):
        self.assertAlmostEqual(L.bits_per_weight(5120), 4.25 + 16 / 5120, places=12)
        self.assertAlmostEqual(L.tensor_bytes(64, 5120) * 8 / (64 * 5120),
                               L.bits_per_weight(5120), places=12)

    def test_geometry_gate_rejects_misalignment(self):
        with self.assertRaises(ValueError):
            L.assert_geometry(63, 32)          # partial panel: no valid anchor header
        with self.assertRaises(ValueError):
            L.assert_geometry(64, 33)          # partial slab: no valid sub-scale byte
        L.assert_geometry(64, 32)

    def test_split_join_roundtrip_on_real_bytes(self):
        for tag, N, K in PXQ4_FIXTURES:
            b = blob(tag)
            self.assertEqual(len(b), L.tensor_bytes(N, K), tag)
            slabs, anchor = L.split_blob(b, N, K)
            self.assertEqual(slabs.shape, L.slab_shape(N, K))
            self.assertEqual(anchor.shape, L.anchor_shape(N))
            self.assertEqual(slabs.dtype, np.uint8)
            self.assertEqual(anchor.dtype, np.float16)
            self.assertEqual(L.join_blob(slabs, anchor), b, f"{tag}: G2 round-trip")

    def test_split_is_a_partition_not_a_transform(self):
        # Every byte of the blob must appear exactly once across the two outputs, in order.
        b = blob("fx_pxq4_gate")
        slabs, anchor = L.split_blob(b, 128, 5120)
        hdr = anchor.view(np.uint8).reshape(2, 128)
        for p in range(2):
            beg = p * L.panel_bytes(5120)
            self.assertEqual(bytes(hdr[p]), b[beg:beg + 128])
            self.assertEqual(bytes(slabs[p].reshape(-1)),
                             b[beg + 128: beg + L.panel_bytes(5120)])

    def test_shard_helpers_reject_misaligned_boundaries(self):
        slabs, anchor = L.split_blob(blob("fx_pxq4_gate"), 128, 5120)
        with self.assertRaises(ValueError):
            L.shard_columns(slabs, anchor, 0, 32)       # not a whole panel
        with self.assertRaises(ValueError):
            L.shard_k(slabs, anchor, 0, 16)             # not a whole slab

    def test_expert_stack_split(self):
        # No expert tensor exists in this model (verified: zero *_exps in 866 tensors), but the
        # 3-D addressing is a documented outer dimension and must not be silently wrong if a
        # future MoE artifact appears.
        b = blob("fx_pxq4_gate")                     # 2 panels = 1 "expert" of N=64 twice
        s3, a3 = L.split_blob_3d(b, 2, 64, 5120)
        self.assertEqual(s3.shape, (2, 1, 160, 1088))
        self.assertEqual(a3.shape, (2, 1, 64))
        s2, a2 = L.split_blob(b, 128, 5120)
        self.assertTrue(np.array_equal(s3.reshape(2, 160, 1088), s2))
        self.assertTrue(np.array_equal(a3.reshape(2, 64), a2))


# =============================================================================================
class TestTables(unittest.TestCase):
    def test_invariants(self):
        R.check_tables()

    def test_tables_equal_the_files_own_kvs(self):
        # PXA_PXQ6_BOOK / PXA_PXQ6_SUB can override the compiled-in tables at build time, so
        # the file records what it was actually quantized with. They must agree, or every
        # weight decodes wrong.
        gg = header()
        self.assertTrue(np.array_equal(np.asarray(gg.kv["pxa.pxq6.book"], np.float32), R.BOOK))
        self.assertTrue(np.array_equal(np.asarray(gg.kv["pxa.pxq6.sub"], np.float32), R.SUB))

    def test_book_is_fp16_exact_and_normalised(self):
        self.assertEqual(R.BOOK[7], 0.0)
        self.assertEqual(R.BOOK[15], 1.0)
        self.assertTrue(np.array_equal(R.BOOK.astype(np.float16).astype(np.float32), R.BOOK))


# =============================================================================================
class TestReferenceDequant(unittest.TestCase):
    """GATE G1. Bit-exact against the engine's own CPU decoder, on real weights."""

    def test_g1_bit_exact_vs_verbatim_c_oracle(self):
        for tag, N, K in PXQ4_FIXTURES:
            got = R.dequant_blob(blob(tag), N, K)
            want = oracle_f32(tag, N, K)
            self.assertEqual(got.shape, (N, K))
            self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)),
                            f"{tag}: not bit-exact vs pxa_deq_row_pxq6")

    def test_vectorised_matches_scalar_transcription(self):
        for tag, N, K in PXQ4_FIXTURES:
            slabs, anchor = L.split_blob(blob(tag), N, K)
            a = R.dequant(slabs[:1], anchor[:1])
            b = R.dequant_scalar(slabs[:1], anchor[:1])
            self.assertTrue(np.array_equal(a.view(np.uint32), b.view(np.uint32)), tag)

    def test_zero_anchor_row_decodes_to_exact_zero(self):
        slabs, anchor = L.split_blob(blob("fx_pxq4_gate"), 128, 5120)
        anchor = anchor.copy()
        anchor[0, 5] = np.float16(0.0)
        w = R.dequant(slabs, anchor)
        self.assertTrue(np.array_equal(w[5], np.zeros(5120, dtype=np.float32)))

    def test_multiply_order_is_safe_ONLY_because_the_tables_are_fp16_exact(self):
        """FINDING, measured here, that corrects an assumption in the plan.

        The plan states the multiply order ``(anchor*sub)*book`` is load-bearing for
        bit-exactness and must not be reassociated. On THIS table pair it is not: across all
        4,849,664 products in the three real fixtures, and across a synthetic sweep of 4096
        anchors x 16 subs x 16 book entries, ``(a*s)*b`` and ``a*(s*b)`` are bit-identical in
        every single case.

        The reason is structural, not luck. BOOK and SUB are fp16-snapped by construction
        (ggml-pxq6-tables.h:32,39) and the anchor is stored fp16, so every operand carries at
        most an 11-bit significand. Any product of two of them is exact in float32 (22 bits <=
        24), so both groupings round the SAME exact real value exactly once.

        This is worth knowing because it means the CUDA kernel has freedom the plan denied it.
        It is NOT a licence to reassociate: the moment a table override (PXA_PXQ6_BOOK /
        PXA_PXQ6_SUB, honoured at pxq-cpu.c:80-82) introduces an entry that is not fp16-exact,
        the two orders diverge — as the second half of this test demonstrates. reference.py
        keeps the C's order so it stays correct for a table it was not written against, and
        check_tables() refuses a non-fp16-exact table outright.
        """
        ndiff = 0
        total = 0
        for tag, N, K in PXQ4_FIXTURES:
            slabs, anchor = L.split_blob(blob(tag), N, K)
            anch = anchor.astype(np.float32)
            sb = slabs[:, :, :64]
            codes = slabs[:, :, 64:].reshape(*slabs.shape[:2], 64, 16)
            for idx in (sb & 0x0F, sb >> 4):
                for c in (codes & 0x0F, codes >> 4):
                    good = (anch[:, None, :] * R.SUB[idx])[..., None] * R.BOOK[c]
                    alt = anch[:, None, :, None] * (R.SUB[idx][..., None] * R.BOOK[c])
                    total += good.size
                    ndiff += int(np.count_nonzero(good.view(np.uint32) != alt.view(np.uint32)))
        self.assertGreater(total, 4_000_000)
        self.assertEqual(ndiff, 0, "the frozen tables are fp16-exact, so reassociation must "
                                   "be a no-op; a difference means a table changed")

        # ...and the same reassociation is NOT safe once an operand is not fp16-exact.
        rng = np.random.default_rng(0)
        a = np.float16(rng.uniform(0.01, 1, 20000)).astype(np.float32)
        s_ = np.float32(rng.uniform(0.2, 1, 20000))
        b_ = np.float32(rng.uniform(-1, 1, 20000))
        self.assertGreater(
            int(np.count_nonzero(((a * s_) * b_).view(np.uint32) != (a * (s_ * b_)).view(np.uint32))),
            0)

    def test_check_tables_rejects_a_non_fp16_exact_table(self):
        bad = R.BOOK.copy()
        bad[3] = np.float32(bad[3]) * np.float32(1.0000001)
        with self.assertRaises(ValueError):
            R.check_tables(bad, R.SUB)

    def test_dequant_rejects_shape_mismatch(self):
        slabs, anchor = L.split_blob(blob("fx_pxq4_gate"), 128, 5120)
        with self.assertRaises(ValueError):
            R.dequant(slabs, anchor[:1])
        with self.assertRaises(ValueError):
            R.dequant(slabs.astype(np.int8), anchor)


# =============================================================================================
class TestShardCommutation(unittest.TestCase):
    """GATE G3. dequant(shard(x)) == shard(dequant(x)), bit-exact, both axes, TP 2 and 4."""

    def _check(self, tag, N, K, tps=(2, 4)):
        slabs, anchor = L.split_blob(blob(tag), N, K)
        full = R.dequant(slabs, anchor)
        for tp in tps:
            if N % tp == 0 and (N // tp) % 64 == 0:
                per = N // tp
                for r in range(tp):
                    s, a = L.shard_columns(slabs, anchor, r * per, (r + 1) * per)
                    got = R.dequant(s, a)
                    self.assertTrue(
                        np.array_equal(got.view(np.uint32),
                                       full[r * per:(r + 1) * per].view(np.uint32)),
                        f"{tag} column tp={tp} rank={r}")
            if K % tp == 0 and (K // tp) % 32 == 0:
                per = K // tp
                for r in range(tp):
                    s, a = L.shard_k(slabs, anchor, r * per, (r + 1) * per)
                    got = R.dequant(s, a)
                    self.assertTrue(
                        np.array_equal(got.view(np.uint32),
                                       full[:, r * per:(r + 1) * per].view(np.uint32)),
                        f"{tag} row tp={tp} rank={r}")

    def test_g3_on_every_fixture_shape(self):
        for tag, N, K in PXQ4_FIXTURES:
            self._check(tag, N, K)

    def test_k_shard_duplicates_the_header_verbatim(self):
        # The row-parallel shard's cost and its correctness are the same fact: the 128 B anchor
        # header has no cross-K coupling, so it is copied, not recomputed.
        slabs, anchor = L.split_blob(blob("fx_pxq4_down"), 64, 17408)
        _, a = L.shard_k(slabs, anchor, 0, 4352)
        self.assertTrue(np.array_equal(a.view(np.uint16), anchor.view(np.uint16)))

    def test_column_shard_is_a_memcpy_of_whole_panels(self):
        b = blob("fx_pxq4_gate")
        slabs, anchor = L.split_blob(b, 128, 5120)
        s, a = L.shard_columns(slabs, anchor, 64, 128)
        self.assertEqual(L.join_blob(s, a), b[L.panel_bytes(5120):])

    def test_arbitrary_panel_boundaries_all_commute(self):
        slabs, anchor = L.split_blob(blob("fx_pxq4_q"), 128, 5120)
        full = R.dequant(slabs, anchor)
        for beg, end in [(0, 64), (64, 128), (0, 128)]:
            s, a = L.shard_columns(slabs, anchor, beg, end)
            self.assertTrue(np.array_equal(R.dequant(s, a).view(np.uint32),
                                           full[beg:end].view(np.uint32)))

    def test_real_model_shard_sizes_are_panel_aligned(self):
        # The real output_sizes the fork builds. A boundary that is not a multiple of 64 is the
        # silent-truncation bug (parameter.py:605-610 does round(size // packed_factor)).
        cases = {
            "in_proj_qkvz": [2048, 2048, 6144, 6144],
            "gate_up_proj": [17408, 17408],
            "qkv_proj": [12288, 1024, 1024],
        }
        for tp in (1, 2, 4):
            for label, sizes in cases.items():
                off = 0
                for i, sz in enumerate(sizes):
                    self.assertEqual(sz % tp, 0, f"{label} tp={tp} shard {i}")
                    self.assertEqual((off // tp) % 64, 0, f"{label} tp={tp} offset {i}")
                    self.assertEqual((sz // tp) % 64, 0, f"{label} tp={tp} size {i}")
                    off += sz
        for label, K in (("down_proj", 17408), ("o_proj", 6144), ("out_proj", 6144)):
            for tp in (1, 2, 4):
                self.assertEqual(K % tp, 0, label)
                self.assertEqual((K // tp) % 32, 0, f"{label} tp={tp}")


# =============================================================================================
class TestDenseDecoders(unittest.TestCase):
    """Every non-PXQ4 type in the artifact, pinned against verbatim ggml C."""

    def test_q8_0_bit_exact(self):
        got = D.dequant_q8_0(blob("fx_q8_0_k"), 8, 5120)
        self.assertTrue(np.array_equal(got.view(np.uint32),
                                       oracle_f32("fx_q8_0_k", 8, 5120).view(np.uint32)))

    def test_q6_K_bit_exact(self):
        got = D.dequant_q6_K(blob("fx_q6k_embd"), 4, 5120)
        self.assertTrue(np.array_equal(got.view(np.uint32),
                                       oracle_f32("fx_q6k_embd", 4, 5120).view(np.uint32)))

    def test_mxfp4_bit_exact(self):
        # ssm_out on all 48 GDN layers is MXFP4, not PXQ4 — the type the brief did not expect.
        got = D.dequant_mxfp4(blob("fx_mxfp4_out"), 8, 6144)
        self.assertTrue(np.array_equal(got.view(np.uint32),
                                       oracle_f32("fx_mxfp4_out", 8, 6144).view(np.uint32)))

    def test_mxfp4_is_split_halves_not_interleaved_pairs(self):
        # If this layout were the PXQ4 (2b, 2b+1) pairing, ssm_out would be a plausible
        # permutation of itself on 48 layers and nothing would raise.
        raw = np.zeros(17, dtype=np.uint8)
        raw[0] = 127 + 1                      # e8m0 -> d = 1.0 via the /2 in the half variant
        raw[1] = 0x21                         # low nibble 1, high nibble 2
        w = D.dequant_mxfp4(raw.tobytes(), 1, 32)[0]
        self.assertEqual(w[0], 1.0)           # element 0  <- low nibble of byte 0
        self.assertEqual(w[16], 2.0)          # element 16 <- high nibble of byte 0
        self.assertEqual(w[1], 0.0)

    def test_e8m0_special_cases(self):
        # e=0 and e=1 do NOT follow 2**(e-128); they are hard-coded subnormal patterns
        # (ggml-impl.h:41). Getting them wrong scales a whole 32-element block.
        e = np.array([0, 1, 2, 128, 255], dtype=np.uint8)
        d = D._e8m0_to_fp32_half(e)
        self.assertEqual(d[0].view(np.uint32) if hasattr(d[0], "view")
                         else np.float32(d[0]).view(np.uint32), np.uint32(0x00400000))
        self.assertEqual(np.float32(d[1]).view(np.uint32), np.uint32(0x00200000))
        self.assertEqual(np.float32(d[3]), np.float32(2.0 ** (128 - 1 - 127)))

    def test_f32_passthrough(self):
        w = D.dequant_f32(blob("fx_f32_conv1d"), 10240, 4)
        self.assertEqual(w.shape, (10240, 4))
        self.assertTrue(np.all(np.isfinite(w)))

    def test_dequant_any_reverses_ne(self):
        w = D.dequant_any(blob("fx_f32_conv1d"), G.GGML_F32, (4, 10240))
        self.assertEqual(w.shape, (10240, 4))

    def test_pxq4_has_no_per_row_decoder(self):
        with self.assertRaises(ValueError):
            D.dequant_any(b"", G.GGML_PXQ4, (32, 64))


# =============================================================================================
class TestGGUFReader(unittest.TestCase):
    def test_reads_the_real_header(self):
        gg = header()
        self.assertEqual(gg.version, 3)
        self.assertEqual(len(gg.tensors), 866)
        self.assertEqual(gg.kv["general.architecture"], "qwen35")
        self.assertEqual(gg.data_start, 10_997_184)

    def test_type_histogram_is_the_five_types_we_handle(self):
        hist = header().type_histogram()
        self.assertEqual({k: v[0] for k, v in hist.items()},
                         {"f32": 360, "mxfp4": 48, "pxq4": 325, "q6_K": 1, "q8_0": 132})
        self.assertEqual(hist["pxq4"][1], 12_231_950_336)

    def test_all_types_supported(self):
        header().assert_all_supported()

    def test_derived_sizes_agree_with_the_panel_formula(self):
        # This is the strongest available confirmation of the layout short of decoding: the
        # reader derives each tensor's length from neighbouring offsets, then the constructor
        # cross-checks it against row_size(). All 866 pass, so the data section is dense and
        # the PXQ4 panel formula is exactly what the writer used.
        gg = header()
        for n, t in gg.tensors.items():
            if t.type_id == G.GGML_PXQ4:
                self.assertEqual(t.nbytes, L.tensor_bytes(t.ne1, t.ne0), n)

    def test_no_expert_tensors_exist(self):
        gg = header()
        self.assertEqual([n for n in gg.tensors if "_exps" in n], [])
        self.assertEqual([k for k in gg.kv if "expert" in k], [])

    def test_every_pxq4_tensor_passes_the_geometry_gate(self):
        gg = header()
        for n, t in gg.tensors.items():
            if t.type_id == G.GGML_PXQ4:
                L.assert_geometry(t.ne1, t.ne0)


# =============================================================================================
class TestNameMap(unittest.TestCase):
    def test_every_ggml_tensor_maps_or_is_deliberately_skipped(self):
        gg = header()
        mtp = NM.mtp_block_range(gg.kv)
        self.assertEqual(list(mtp), [64])
        skipped = 0
        for n in gg.order:
            hf = NM.GGML_TO_HF(n, gg.kv)          # raises on an unmapped suffix
            if hf is None:
                skipped += 1
                self.assertTrue(n.startswith("blk.64."), n)
        self.assertEqual(skipped, 15)

    def test_keyset_matches_the_reference_checkpoint_exactly(self):
        """GATE G4."""
        from gguf_to_vllm import convert as C
        gg = header()
        plan = C.build_plan(gg, "p1", os.path.join(FX, "refhf"), have_encoder=False)
        rep = C.keyset_diff(plan, os.path.join(FX, "refhf"))
        self.assertEqual(rep["missing"], [])
        self.assertEqual(rep["extra"], [])
        self.assertEqual(rep["n_ours"], rep["n_ref"])

    def test_fused_module_resolution(self):
        f = NM.HF_MODULE_OF
        self.assertEqual(f("model.language_model.layers.0.mlp.gate_proj.weight"),
                         "model.language_model.layers.0.mlp.gate_up_proj")
        self.assertEqual(f("model.language_model.layers.0.mlp.up_proj.weight"),
                         "model.language_model.layers.0.mlp.gate_up_proj")
        self.assertEqual(f("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
                         "model.language_model.layers.0.linear_attn.in_proj_qkvz")
        self.assertEqual(f("model.language_model.layers.0.linear_attn.in_proj_z.weight"),
                         "model.language_model.layers.0.linear_attn.in_proj_qkvz")
        self.assertEqual(f("model.language_model.layers.3.self_attn.k_proj.weight"),
                         "model.language_model.layers.3.self_attn.qkv_proj")
        self.assertEqual(f("model.language_model.layers.0.linear_attn.in_proj_b.weight"),
                         "model.language_model.layers.0.linear_attn.in_proj_ba")
        self.assertEqual(f("lm_head.weight"), "lm_head")
        self.assertEqual(f("model.language_model.layers.0.linear_attn.A_log"),
                         "model.language_model.layers.0.linear_attn")

    def test_ignore_list_always_carries_the_gdn_split_key(self):
        # Without these two entries _uses_split_gdn_input_projections returns False, the 48-row
        # b/a fold into in_proj_qkvz, and TP=4 gives 12 rows/rank — silently truncated.
        for pol in ("p1", "p2a", "p2b", "p2c"):
            ig = NM.ignore_list(pol)
            self.assertIn("linear_attn.in_proj_a", ig, pol)
            self.assertIn("linear_attn.in_proj_b", ig, pol)

    def test_policies_are_monotone(self):
        p = NM.POLICY_MODULES
        self.assertTrue(p["p1"] < p["p2a"] <= p["p2b"] < p["p2c"])
        self.assertIn("self_attn.qkv_proj", p["p2c"])
        self.assertNotIn("self_attn.qkv_proj", p["p2b"])

    def test_no_policy_serves_an_unservable_module(self):
        # THE REGRESSION TEST. p2b/p2c used to carry "lm_head", which
        # convert.py:295 copied verbatim into quantization_config; the engine
        # rejects it (pxq4_vllm.config.UNSERVABLE_PXQ4_LEAF_MODULES) because
        # ParallelLMHead forces the v1 vocab weight_loader. Every p2b/p2c run
        # died during model construction.
        for pol, mods in NM.POLICY_MODULES.items():
            for m in mods:
                self.assertNotIn(m.rsplit(".", 1)[-1], NM.UNSERVABLE_PXQ4_MODULES,
                                 f"{pol} serves {m}")
            # lm_head is a linear-shaped module the dispatcher WILL be asked about, so it
            # must be explicitly ignored. embed_tokens is a VocabParallelEmbedding and never
            # reaches the LinearBase branch, so it needs no entry.
            self.assertIn("lm_head", NM.ignore_list(pol), pol)

    def test_blocked_policy_is_refused_by_name(self):
        # p2b minus the head is p2a. Refuse it instead of silently aliasing it,
        # so a benchmark cannot be reported against a policy that never ran.
        with self.assertRaises(SystemExit):
            NM.assert_policy_supported("p2b")
        for pol in ("p1", "p2a", "p2c"):
            NM.assert_policy_supported(pol)

    def test_unknown_suffix_raises_rather_than_guessing(self):
        gg = header()
        with self.assertRaises(KeyError):
            NM.GGML_TO_HF("blk.0.some_new_thing.weight", gg.kv)


# =============================================================================================
class TestPlan(unittest.TestCase):
    def setUp(self):
        from gguf_to_vllm import convert as C
        self.C = C
        self.gg = header()
        self.ref = os.path.join(FX, "refhf")

    def test_p1_serves_exactly_the_uniformly_pxq4_modules(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        srcs = {e.src for e in plan.emits if e.kind == "pxq4"}
        suffixes = {s.split(".", 2)[-1] for s in srcs}
        self.assertEqual(suffixes, {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                                    "attn_output.weight", "attn_qkv.weight",
                                    "attn_gate.weight"})
        # attn_q is PXQ4 on disk but must NOT be served in P1: its module (qkv_proj) also
        # holds q8_0 k/v, and a mixed module violates the §3.1 uniformity invariant.
        self.assertNotIn("blk.3.attn_q.weight", srcs)
        self.assertEqual(len(srcs), 304)

    def test_p1_needs_no_encoder(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        self.assertEqual(plan.reencode, [])

    def test_p2_policies_require_an_encoder(self):
        for pol in ("p2a", "p2b", "p2c"):
            with self.assertRaises(SystemExit, msg=pol):
                self.C.build_plan(self.gg, pol, self.ref, have_encoder=False)

    def test_p2c_reencodes_exactly_ssm_out_and_kv(self):
        plan = self.C.build_plan(self.gg, "p2c", self.ref, have_encoder=True)
        by = {}
        for n in plan.reencode:
            by[n.split(".")[-2] if n.startswith("blk.") else n] = by.get(
                n.split(".")[-2] if n.startswith("blk.") else n, 0) + 1
        # output.weight is NOT here: the LM head stays fp16 (namemap
        # PXQ4_MODULES_P2B), because the engine cannot load a PXQ4 one.
        self.assertEqual(by, {"ssm_out": 48, "attn_k": 16, "attn_v": 16})
        self.assertNotIn("output.weight", plan.reencode)
        # attn_q is already PXQ4 and must be passed through, never re-encoded.
        self.assertNotIn("blk.3.attn_q.weight", plan.reencode)

    def test_lm_head_is_emitted_dense_for_every_policy(self):
        for pol in ("p1", "p2a", "p2b", "p2c"):
            plan = self.C.build_plan(self.gg, pol, self.ref, have_encoder=True)
            heads = [e for e in plan.emits if e.name == "lm_head.weight"]
            self.assertEqual(len(heads), 1, pol)
            self.assertEqual(heads[0].kind, "dense", pol)

    def test_every_module_is_uniformly_one_type(self):
        for pol in ("p1", "p2a", "p2b", "p2c"):
            plan = self.C.build_plan(self.gg, pol, self.ref, have_encoder=True)
            for m, kinds in plan.module_types.items():
                self.assertEqual(len(kinds), 1, f"{pol}: {m} is {kinds}")

    def test_pxq4_emits_two_tensors_and_no_weight(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        names = {e.name for e in plan.emits}
        stem = "model.language_model.layers.0.mlp.gate_proj"
        self.assertIn(stem + ".pxq4_slabs", names)
        self.assertIn(stem + ".pxq4_anchor", names)
        self.assertNotIn(stem + ".weight", names)

    def test_emitted_pxq4_shapes_are_the_contract_shapes(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        by = {e.name: e for e in plan.emits}
        s = by["model.language_model.layers.0.mlp.gate_proj.pxq4_slabs"]
        a = by["model.language_model.layers.0.mlp.gate_proj.pxq4_anchor"]
        self.assertEqual(s.shape, (17408 // 64, 5120 // 32, 1088))   # [N/64, K/32, 1088]
        self.assertEqual(s.dtype, "U8")
        self.assertEqual(a.shape, (17408 // 64, 64))                 # [N/64, 64]
        self.assertEqual(a.dtype, "F16")

    def test_conv1d_is_reshaped_to_hf_orientation(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        e = next(x for x in plan.emits if x.name.endswith("layers.0.linear_attn.conv1d.weight"))
        self.assertEqual(e.shape, (10240, 1, 4))
        # ...and it matches the reference checkpoint's shape exactly.
        hdr = ST.read_header(os.path.join(FX, "refhf", "model.safetensors"))
        self.assertEqual(
            hdr["model.language_model.layers.0.linear_attn.conv1d.weight"]["shape"],
            [10240, 1, 4])

    def test_emitted_dense_shapes_match_the_reference_checkpoint(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        hdr = ST.read_header(os.path.join(FX, "refhf", "model.safetensors"))
        checked = 0
        for e in plan.emits:
            if e.kind == "dense" and e.name in hdr:
                self.assertEqual(list(e.shape), hdr[e.name]["shape"], e.name)
                checked += 1
        self.assertGreater(checked, 400)

    def test_pxq4_logical_shapes_match_the_reference_checkpoint(self):
        # For a PXQ4 module the reference stores weight_shape / weight_packed; compare against
        # weight_scale's row count, which is the true out_features.
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        hdr = ST.read_header(os.path.join(FX, "refhf", "model.safetensors"))
        checked = 0
        for e in plan.emits:
            if not e.name.endswith(".pxq4_slabs"):
                continue
            stem = e.name[: -len(".pxq4_slabs")]
            ref = hdr.get(stem + ".weight_scale")
            if ref is None:
                continue
            N = e.shape[0] * 64
            K = e.shape[1] * 32
            self.assertEqual(N, ref["shape"][0], stem)
            self.assertEqual(K, ref["shape"][1] * 128, stem)   # AWQ group_size 128
            checked += 1
        self.assertEqual(checked, 304)

    def test_vision_tower_is_copied_verbatim(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        vis = [e for e in plan.emits if e.kind == "copy"]
        self.assertEqual(len(vis), 333)
        self.assertTrue(all(e.dtype == "BF16" for e in vis))
        self.assertEqual(sum(e.nbytes for e in vis), 921_460_192)

    def test_quantization_config_is_well_formed(self):
        cfg = self.C.build_quantization_config(self.gg, "p1")
        self.assertEqual(cfg["quant_method"], "pxq4")
        self.assertEqual(cfg["type_id"], 252)
        self.assertEqual((cfg["panel_rows"], cfg["slab_cols"], cfg["slab_bytes"],
                          cfg["header_bytes"]), (64, 32, 1088, 128))
        self.assertEqual(len(cfg["book"]), 16)
        self.assertEqual(len(cfg["sub"]), 16)
        self.assertIn("linear_attn.in_proj_a", cfg["ignore"])
        self.assertIn("linear_attn.in_proj_b", cfg["ignore"])
        self.assertIn("model.visual", cfg["ignore"])
        self.assertEqual(cfg["backbone_rev"], 2)
        self.assertEqual(sorted(cfg["pxq4_modules"]),
                         ["linear_attn.in_proj_qkvz", "mlp.down_proj", "mlp.gate_up_proj",
                          "self_attn.o_proj"])
        # P1 does not serve these, so the runtime must be told to leave them unquantized.
        for m in ("self_attn.qkv_proj", "linear_attn.out_proj", "lm_head"):
            self.assertIn(m, cfg["ignore"])


# =============================================================================================
class TestSafetensorsWriter(unittest.TestCase):
    def test_roundtrip_including_a_bf16_passthrough(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            a = np.arange(12, dtype=np.uint8).reshape(3, 4)
            b = np.linspace(-1, 1, 8).astype(np.float16)
            raw_bf16 = np.array([0x3F80, 0xBF80, 0x0000], dtype="<u2").tobytes()
            p = os.path.join(d, "m.safetensors")
            ST.write_file(p, [ST.Tensor.from_numpy("a", a),
                              ST.Tensor.from_numpy("b", b),
                              ST.Tensor("v", "BF16", (3,), raw_bf16)])
            hdr = ST.read_header(p)
            self.assertEqual(hdr["a"]["dtype"], "U8")
            self.assertEqual(hdr["b"]["dtype"], "F16")
            self.assertEqual(hdr["v"]["dtype"], "BF16")
            self.assertEqual(ST.read_tensor_bytes(p, "a")[2], a.tobytes())
            self.assertEqual(ST.read_tensor_bytes(p, "v")[2], raw_bf16)

    def test_offsets_are_eight_aligned(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.safetensors")
            ST.write_file(p, [ST.Tensor.from_numpy(f"t{i}", np.zeros(i + 1, np.uint8))
                              for i in range(6)])
            for k, v in ST.read_header(p).items():
                self.assertEqual(v["data_offsets"][0] % 8, 0, k)

    def test_streaming_tensor_writes_and_is_length_checked(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.safetensors")
            payload = np.arange(1000, dtype=np.float16)

            def w(f):
                for beg in range(0, 1000, 250):
                    f.write(payload[beg:beg + 250].tobytes())

            ST.write_file(p, [ST.Tensor("big", "F16", (1000,), w, streaming=True)])
            self.assertEqual(ST.read_tensor_bytes(p, "big")[2], payload.tobytes())

            bad = ST.Tensor("b", "F16", (1000,), lambda f: f.write(b"x" * 10), streaming=True)
            with self.assertRaises(ValueError):
                ST.write_file(os.path.join(d, "n.safetensors"), [bad])

    def test_declared_size_mismatch_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            t = ST.Tensor("x", "U8", (10,), lambda: b"short")
            with self.assertRaises(ValueError):
                ST.write_file(os.path.join(d, "m.safetensors"), [t])

    def test_sharding_and_index(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            w = ST.ShardWriter(d, max_bytes=1024)
            for i in range(6):
                w.add(ST.Tensor.from_numpy(f"t{i}", np.zeros(400, np.uint8)))
            idx = w.finish()
            self.assertEqual(len(set(idx["weight_map"].values())), 3)
            self.assertTrue(os.path.exists(os.path.join(d, "model.safetensors.index.json")))


# =============================================================================================
class TestEndToEndOnRealBytes(unittest.TestCase):
    """A full miniature conversion: real PXQ4 + real q8_0 bytes through the writer and back."""

    def test_emit_and_reload_reproduces_the_decoded_weights(self):
        import tempfile
        from gguf_to_vllm import convert as C
        b = blob("fx_pxq4_gate")
        slabs, anchor = L.split_blob(b, 128, 5120)
        kraw = blob("fx_q8_0_k")
        kdense = D.dequant_q8_0(kraw, 8, 5120).astype(np.float16)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model.safetensors")
            ST.write_file(p, [
                ST.Tensor.from_numpy("m.mlp.gate_proj.pxq4_slabs", slabs),
                ST.Tensor.from_numpy("m.mlp.gate_proj.pxq4_anchor", anchor),
                ST.Tensor.from_numpy("m.self_attn.k_proj.weight", kdense),
            ])
            hdr = ST.read_header(p)
            self.assertEqual(hdr["m.mlp.gate_proj.pxq4_slabs"]["shape"], [2, 160, 1088])
            self.assertEqual(hdr["m.mlp.gate_proj.pxq4_anchor"]["shape"], [2, 64])

            _, sh, sb = ST.read_tensor_bytes(p, "m.mlp.gate_proj.pxq4_slabs")
            _, ah, ab = ST.read_tensor_bytes(p, "m.mlp.gate_proj.pxq4_anchor")
            s2 = np.frombuffer(sb, np.uint8).reshape(sh)
            a2 = np.frombuffer(ab, "<f2").reshape(ah)
            # Bytes survived the container...
            self.assertEqual(L.join_blob(s2, a2), b)
            # ...and decode to exactly what the engine's own C decoder produced.
            self.assertTrue(np.array_equal(R.dequant(s2, a2).view(np.uint32),
                                           oracle_f32("fx_pxq4_gate", 128, 5120).view(np.uint32)))

            _, kh, kb = ST.read_tensor_bytes(p, "m.self_attn.k_proj.weight")
            k2 = np.frombuffer(kb, "<f2").reshape(kh)
            self.assertTrue(np.array_equal(k2, kdense))

    def test_verify_gates_on_the_fixture(self):
        # Exercise verify.py's gate functions against a hand-built one-tensor "file" so the
        # module itself is covered, not just the algorithms it calls.
        from gguf_to_vllm import verify as V
        b = blob("fx_pxq4_down")
        slabs, anchor = L.split_blob(b, 64, 17408)
        full = R.dequant(slabs, anchor)
        for tp in (2, 4):
            per = 17408 // tp
            for r in range(tp):
                s, a = L.shard_k(slabs, anchor, r * per, (r + 1) * per)
                self.assertTrue(np.array_equal(
                    R.dequant(s, a).view(np.uint32),
                    full[:, r * per:(r + 1) * per].view(np.uint32)))
        gg = header()
        self.assertEqual(V.gate_real_shards(gg, "p2c"), [])



# =============================================================================================
# GDN v-head ORDER. The defect this class exists for: ggml orders the 48 value-heads
# repeat-major (i = 16*r + k) and HF orders them k-head-major (j = 3*k + r), so a converter
# that only renames emits a model that loads, shards, passes every byte gate and generates
# fluent garbage. Every number asserted here was MEASURED on the real artifacts:
#
#   blk.0.ssm_dt.bias vs the AWQ twin's dt_bias : max|diff| 0.000000 permuted, 22.5625 identity
#   blk.0.ssm_a       vs A_log                  : log(-ggml) == A_log permuted (<5e-7), 3.19
#                                                 identity; ggml == -exp(A_log) exactly
#   blk.0.ssm_conv1d  vs conv1d (all 40960)     : EXACT 0.0 permuted + taps forward;
#                                                 0.61 identity; 0.75 taps reversed
#   blk.0.ssm_alpha   vs in_proj_a rows         : rel 0.33 permuted vs 1.24-2.29 identity
# =============================================================================================
class TestGdnHeadOrder(unittest.TestCase):
    def setUp(self):
        self.geom = NM.gdn_geometry(header().kv)

    def test_geometry_comes_from_the_files_own_kvs(self):
        g = self.geom
        self.assertEqual((g.n_k_heads, g.n_v_heads, g.head_dim, g.value_dim),
                         (16, 48, 128, 6144))
        self.assertEqual(g.repeats, 3)
        self.assertEqual(g.key_dim, 2048)
        self.assertEqual(g.qkv_rows, 10240)

    def test_gather_matches_the_measured_witness(self):
        g = NM.v_head_gather(self.geom)
        self.assertEqual(sorted(g), list(range(48)))
        # measured: ggml[i] lands at hf[3*(i%16) + i//16]
        for i in range(48):
            self.assertEqual(g[3 * (i % 16) + i // 16], i)
        self.assertEqual(g[:6], [0, 16, 32, 1, 17, 33])

    def test_identity_is_not_the_answer(self):
        # the whole point: the gather must actually move something
        self.assertNotEqual(NM.v_head_gather(self.geom), list(range(48)))

    def test_spec_covers_every_gdn_tensor_exactly_once(self):
        covered = set(NM.GDN_PERM_SPEC) | set(NM.GDN_NO_PERM)
        self.assertEqual(covered, set(NM._GDN_MAP))
        self.assertFalse(set(NM.GDN_PERM_SPEC) & set(NM.GDN_NO_PERM))

    def test_q_and_k_head_axes_are_left_alone(self):
        # attn_qkv rows 0:4096 are the 16-way q and k head axes, identical in both files
        axis, gather = NM.gdn_permutation("attn_qkv.weight", self.geom, 10240)
        self.assertEqual(axis, 0)
        self.assertEqual(gather[:4096], list(range(4096)))
        self.assertNotEqual(gather[4096:], list(range(4096, 10240)))
        self.assertEqual(sorted(gather), list(range(10240)))
        # first HF v-head block comes from ggml v-head 0, the second from ggml v-head 16
        self.assertEqual(gather[4096], 4096)
        self.assertEqual(gather[4224], 4096 + 16 * 128)

    def test_out_proj_permutes_the_contraction_axis(self):
        axis, gather = NM.gdn_permutation("ssm_out.weight", self.geom, 6144)
        self.assertEqual(axis, 1)
        self.assertEqual(gather[128], 16 * 128)

    def test_shape_mismatch_refuses_rather_than_guesses(self):
        with self.assertRaises(SystemExit):
            NM.gdn_permutation("attn_gate.weight", self.geom, 6145)

    def test_a_log_transform_inverts_the_measured_relation(self):
        fn = NM.VALUE_TRANSFORMS["ssm_a"][0]
        a_log = np.array([-3.203125, -2.65625, -4.65625], dtype=np.float32)
        a = -np.exp(a_log).astype(np.float32)
        self.assertTrue(np.allclose(fn(a), a_log, atol=1e-6))
        with self.assertRaises(SystemExit):
            fn(np.array([0.5], dtype=np.float32))   # a positive A is not this format

    # --- the byte move, on REAL pxq4 panel bytes -----------------------------------------
    def _fixture_pair(self, tag, N, K):
        return L.split_blob(blob(tag), N, K)

    def test_panel_gather_equals_row_gather_after_dequant(self):
        # fx_pxq4_gate is 2 whole panels of the real blk.0.attn_gate; treat the 128 rows as
        # one head block and build a 2-block swap on a doubled copy, so the test exercises a
        # head-block permutation on real bytes at the real 128-row width.
        slabs, anchor = self._fixture_pair("fx_pxq4_gate", 128, 5120)
        slabs2 = np.concatenate([slabs, slabs[::-1]], axis=0)
        anchor2 = np.concatenate([anchor, anchor[::-1]], axis=0)
        rows = np.concatenate([np.arange(128, 256), np.arange(0, 128)])   # swap the 2 blocks
        pidx = L.block_gather_to_panels(rows)
        self.assertTrue(np.array_equal(pidx, np.array([2, 3, 0, 1])))
        ps, pa = L.gather_panels(slabs2, anchor2, pidx)
        want = R.dequant(slabs2, anchor2)[rows]
        got = R.dequant(ps, pa)
        self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)),
                        "panel gather is not bit-identical to a post-dequant row gather")

    def test_slab_gather_equals_column_gather_after_dequant(self):
        slabs, anchor = self._fixture_pair("fx_pxq4_down", 64, 17408)
        S = slabs.shape[1]
        blocks = S // 4                                   # 128 columns = 4 slabs
        order = np.concatenate([np.arange(blocks // 2, blocks), np.arange(0, blocks // 2)])
        cols = np.concatenate([np.arange(b * 128, (b + 1) * 128) for b in order])
        sidx = L.col_gather_to_slabs(cols)
        got = R.dequant(L.gather_slabs(slabs, sidx), anchor)
        want = R.dequant(slabs, anchor)[:, cols]
        self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)))

    def test_permutation_is_undoable_byte_for_byte(self):
        from gguf_to_vllm import convert as C
        raw = blob("fx_pxq4_gate")
        slabs, anchor = L.split_blob(raw, 128, 5120)
        perm = (0, list(range(64, 128)) + list(range(0, 64)))   # swap the two panels
        ps, pa = C._apply_perm_pxq4(slabs, anchor, perm)
        self.assertNotEqual(L.join_blob(ps, pa), raw)
        us, ua = C._unapply_perm_pxq4(ps, pa, perm)
        self.assertEqual(L.join_blob(us, ua), raw)

    def test_misaligned_permutation_is_refused_not_truncated(self):
        with self.assertRaises(ValueError):
            L.block_gather_to_panels(np.concatenate(
                [np.arange(32, 96), np.arange(0, 32), np.arange(96, 128)]))
        with self.assertRaises(ValueError):
            L.col_gather_to_slabs(np.concatenate([np.arange(16, 48), np.arange(0, 16)]))
        with self.assertRaises(ValueError):
            L.assert_permutation(np.array([0, 0, 2, 3]), 4)    # duplicated head


class TestGdnPlanGate(unittest.TestCase):
    """_check_plan must REFUSE a plan that forgot the reorder — the regression guard."""

    def setUp(self):
        from gguf_to_vllm import convert as C
        self.C = C
        self.gg = header()
        self.ref = os.path.join(FX, "refhf")

    def test_every_gdn_tensor_is_marked_permuted(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        seen = {}
        for e in plan.emits:
            if e.kind == "copy":
                continue
            suf = NM.ggml_suffix(e.src)
            if suf in NM._GDN_MAP:
                seen.setdefault(suf, set()).add(bool(e.perm))
        self.assertEqual(seen["attn_qkv.weight"], {True})
        self.assertEqual(seen["attn_gate.weight"], {True})
        self.assertEqual(seen["ssm_out.weight"], {True})
        self.assertEqual(seen["ssm_conv1d.weight"], {True})
        self.assertEqual(seen["ssm_a"], {True})
        self.assertEqual(seen["ssm_dt.bias"], {True})
        self.assertEqual(seen["ssm_alpha.weight"], {True})
        self.assertEqual(seen["ssm_beta.weight"], {True})
        # per-head_v_dim, no v-head axis, and it says so in GDN_NO_PERM
        self.assertEqual(seen["ssm_norm.weight"], {False})

    def test_plan_with_an_unpermuted_gdn_tensor_is_rejected(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        for e in plan.emits:
            if NM.ggml_suffix(e.src) == "ssm_dt.bias":
                e.perm = ""
        with self.assertRaises(SystemExit) as cm:
            self.C._check_plan(plan, "p1")
        self.assertIn("v-head", str(cm.exception))

    def test_a_new_gdn_tensor_must_be_declared(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        try:
            NM._GDN_MAP["ssm_newthing.weight"] = "linear_attn.newthing.weight"
            plan.emits.append(self.C.Emit("x", "dense", "F16", (4,), 8,
                                          "blk.0.ssm_newthing.weight"))
            with self.assertRaises(SystemExit) as cm:
                self.C._check_plan(plan, "p1")
            self.assertIn("GDN_PERM_SPEC", str(cm.exception))
        finally:
            NM._GDN_MAP.pop("ssm_newthing.weight", None)

    def test_lm_head_and_ffn_are_never_permuted(self):
        plan = self.C.build_plan(self.gg, "p1", self.ref, have_encoder=False)
        for e in plan.emits:
            if e.kind == "copy":
                continue
            if NM.ggml_suffix(e.src) not in NM._GDN_MAP:
                self.assertEqual(e.perm, "", e.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
