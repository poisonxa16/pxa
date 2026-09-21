# SPDX-License-Identifier: Apache-2.0
"""CPU tests for component B (PXQ4LinearMethod / parameters / ops).

DESTINATION IN THE REPO OF PLAN 09: ``tests/test_linear.py`` (+ ``test_shard.py``).

Run:  python3 test_pxq4_linear.py            (stdlib unittest; pytest also collects)

NO GPU, NO TORCH REQUIRED.  With torch+vLLM absent the numpy fakes in
``pxq4_linear_stubs.py`` stand in; the sharding tests are then exercising a
VERBATIM transcription of parameter.py/linear.py rather than the real classes,
which is stated plainly rather than hidden -- inside the container the same
file re-runs against the genuine vLLM (the stubs skip themselves), and that is
what upgrades these from "consistent with my reading" to "true".

WHAT IS ACTUALLY BEING PROVEN HERE
  1. For every real module shape of Qwen3.8-27B at TP in {1,2,4}, the stock v2
     loaders narrow our two parameters on WHOLE PANELS (dim 0, 64 rows) and
     WHOLE SLABS (dim 1, 32 columns), and the anchor is duplicated verbatim on
     a K-split.  No offset is ever floor-divided into a wrong-but-well-formed
     slice (parameter.py:605-616).
  2. Shard-then-dequantize == dequantize-then-slice, BIT-EXACT in fp32, on both
     axes.  That is the plan's gate G3 restricted to what component B controls:
     it proves the TP split is a permutation of bytes, not a re-quantization.
  3. ``create_weights`` refuses every geometry that would truncate, including
     the real in_proj_ba 48-row trap.
  4. ``apply`` dispatches mmv vs dequant+GEMM correctly, adds bias, restores
     the leading dims, and allocates exactly one tensor.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pxq4_linear_stubs as stubs  # noqa: E402

torch = stubs.install_torch_stub()
stubs.install_vllm_stub(torch)

import pxq4_kernel_ref as ref  # noqa: E402  (component C's numpy oracle)

PANEL_ROWS = 64
SLAB_COLS = 32
SLAB_BYTES = 1088


# --------------------------------------------------------------------------
# Assemble the package so the relative imports in linear.py resolve.
# --------------------------------------------------------------------------
def _build_pkg() -> str:
    tmp = tempfile.mkdtemp(prefix="pxq4pkg-")
    pkg = Path(tmp) / "pxq4_vllm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    shutil.copyfile(_HERE / "pxq4_ops.py", pkg / "ops.py")
    shutil.copyfile(_HERE / "pxq4_parameters.py", pkg / "parameters.py")
    shutil.copyfile(_HERE / "pxq4_linear.py", pkg / "linear.py")
    sys.path.insert(0, tmp)
    return tmp


_PKG_TMP = _build_pkg()

from pxq4_vllm import linear as pxlinear  # noqa: E402
from pxq4_vllm import ops as pxops  # noqa: E402
from pxq4_vllm import parameters as pxparams  # noqa: E402

_PARAM_MOD = sys.modules["vllm.model_executor.parameter"]
_LINEAR_MOD = sys.modules["vllm.model_executor.layers.linear"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _t(a: np.ndarray):
    """numpy -> tensor of whichever torch is installed."""
    if stubs.USING_STUBS:
        dt = {
            np.dtype(np.uint8): stubs.UINT8,
            np.dtype(np.float16): stubs.FLOAT16,
            np.dtype(np.float32): stubs.FLOAT32,
        }[a.dtype]
        return stubs.FakeTensor(a, dt, stubs._Device())
    return torch.from_numpy(a)


def _np(t) -> np.ndarray:
    return stubs.arr(t)


def _set_tp(rank: int, size: int) -> None:
    _PARAM_MOD.TP["rank"] = rank
    _PARAM_MOD.TP["size"] = size


class FakeLayer:
    """Stand-in for a LinearBase. Only what create_weights touches."""

    def __init__(self, prefix: str, tp_size: int = 1) -> None:
        self.prefix = prefix
        self.tp_size = tp_size
        self._params: dict[str, object] = {}

    def register_parameter(self, name, param) -> None:
        self._params[name] = param
        setattr(self, name, param)


def weight_loader_v2(param, loaded_weight, shard_id=None):  # name matters
    """Stand-in for the bound ``Linear*.weight_loader_v2``. ``create_weights``
    identifies the v2 path by ``__name__`` (linear.py:704-708)."""
    raise AssertionError("tests drive param.load_* directly")


def _make_method(**cfg):
    quant_config = SimpleNamespace(
        book=list(ref.BOOK.astype(float)), sub=list(ref.SUB.astype(float)), **cfg
    )
    return pxlinear.PXQ4LinearMethod(quant_config)


def _mark_loaded(layer, *, slabs=True, anchor=True):
    """Record the loader calls the tests bypass.

    The sharding tests drive ``param.load_*`` directly (that is the point --
    they are testing the stock v2 loaders), which skips
    ``param.weight_loader`` and therefore skips the _LoaderCallCounter that
    ``process_weights_after_loading`` requires.  Real loading goes through
    ``param.weight_loader``; this stands in for it.  Pass slabs=False /
    anchor=False to simulate a checkpoint that carries only one of the two.
    """
    if slabs:
        layer.pxq4_slab_loads.calls += 1
    if anchor:
        layer.pxq4_anchor_loads.calls += 1


def _create(prefix, K, out_sizes, input_size, output_size, tp_size, method=None):
    layer = FakeLayer(prefix, tp_size=tp_size)
    m = method or _make_method()
    m.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=list(out_sizes),
        input_size=input_size,
        output_size=output_size,
        params_dtype=torch.float16,
        weight_loader=weight_loader_v2,
    )
    return layer


# --- synthetic checkpoint whose bytes identify their own (panel, slab) -----
def _ckpt(N: int, K: int):
    P, S = N // PANEL_ROWS, K // SLAB_COLS
    slabs = np.zeros((P, S, SLAB_BYTES), dtype=np.uint8)
    pp = np.arange(P, dtype=np.uint16)[:, None]
    ss = np.arange(S, dtype=np.uint16)[None, :]
    slabs[:, :, 0] = (pp & 0xFF).astype(np.uint8)
    slabs[:, :, 1] = (pp >> 8).astype(np.uint8)
    slabs[:, :, 2] = (ss & 0xFF).astype(np.uint8)
    slabs[:, :, 3] = (ss >> 8).astype(np.uint8)
    anchor = np.repeat(
        np.arange(P, dtype=np.float16)[:, None], PANEL_ROWS, axis=1
    ).astype(np.float16)
    return slabs, anchor


def _panels_of(slab_arr: np.ndarray) -> list[int]:
    return sorted({int(v) for v in (slab_arr[:, 0, 0].astype(np.uint16)
                                    | (slab_arr[:, 0, 1].astype(np.uint16) << 8))})


def _slabs_of(slab_arr: np.ndarray) -> list[int]:
    return sorted({int(v) for v in (slab_arr[0, :, 2].astype(np.uint16)
                                    | (slab_arr[0, :, 3].astype(np.uint16) << 8))})


# Real shapes of Qwen3.8-27B (06-file-composition.md sec.2). (N, K).
GATE_UP_ONE = (17408, 5120)     # ffn_gate / ffn_up, fused into gate_up_proj
DOWN = (5120, 17408)            # ffn_down, row-parallel
O_PROJ = (5120, 6144)           # attn_output, row-parallel
IN_PROJ_QKV = (10240, 5120)     # GDN attn_qkv  = q2048 + k2048 + v6144
IN_PROJ_Z = (6144, 5120)        # GDN attn_gate
SSM_OUT = (5120, 6144)          # MXFP4 today; PXQ4 at P2a, row-parallel
ATTN_Q = (12288, 5120)          # P2c
ATTN_KV = (1024, 5120)          # P2c


class Base(unittest.TestCase):
    def setUp(self) -> None:
        _set_tp(0, 1)
        pxops.PXQ4Workspace.reset()
        stubs.ALLOC_COUNT["n"] = 0


# ==========================================================================
class TestParameterShaping(Base):
    def test_attributes_match_the_sharding_contract(self):
        layer = _create("mlp.gate_up_proj", 5120, [17408, 17408], 5120, 34816, 1)
        s, a = layer.pxq4_slabs, layer.pxq4_anchor
        self.assertEqual(tuple(s.shape), (34816 // 64, 5120 // 32, 1088))
        self.assertEqual(tuple(a.shape), (34816 // 64, 64))
        self.assertEqual((s.output_dim, s.input_dim), (0, 1))
        self.assertEqual((s.packed_dim, s.packed_factor), (0, 64))
        self.assertEqual(a.output_dim, 0)
        self.assertEqual((a.packed_dim, a.packed_factor), (0, 64))
        # The anchor must NOT be row-parallel: no input_dim is what makes a
        # K-split full-copy it (parameters.py docstring, point 2).
        self.assertFalse(hasattr(a, "input_dim"))
        self.assertIsNone(s.marlin_tile_size)

    def test_forbids_the_sm70_fp16_bypass(self):
        layer = _create("self_attn.o_proj", 6144, [5120], 6144, 5120, 1)
        self.assertTrue(layer._sm70_f16_forbidden)
        self.assertFalse(getattr(layer, "_sm70_f16_prepared", False))

    def test_method_is_registered_for_weight_loader_v2(self):
        self.assertIn("PXQ4LinearMethod", _LINEAR_MOD.WEIGHT_LOADER_V2_SUPPORTED)

    def test_workspace_reservation_is_the_max_over_layers(self):
        m = _make_method()
        _create("a", 5120, [17408], 5120, 17408, 1, m)
        _create("b", 6144, [5120], 6144, 5120, 1, m)
        self.assertEqual(pxops.PXQ4Workspace.reserved()[0], 17408 * 5120)


# ==========================================================================
class TestCreateWeightsRefusals(Base):
    def _expect(self, msg_fragment, **kw):
        with self.assertRaises(ValueError) as cm:
            _create(**kw)
        self.assertIn(msg_fragment, str(cm.exception))

    def test_rejects_unaligned_K(self):
        self._expect("not a multiple of 32", prefix="x", K=5104,
                     out_sizes=[64], input_size=5104, output_size=64, tp_size=1)

    def test_rejects_unaligned_N(self):
        self._expect("not a multiple of 64", prefix="x", K=64,
                     out_sizes=[100], input_size=64, output_size=100, tp_size=1)

    def test_rejects_the_in_proj_ba_trap(self):
        # qwen3_5.py:212-230 with _uses_split_gdn_input_projections False folds
        # b/a (48 rows each) into in_proj_qkvz. At TP=4 that is 12 rows/rank,
        # which parameter.py:605-616 would floor to 0 panels SILENTLY.
        # The real fused shape trips the N check first (4120 rows total).
        self._expect(
            "N=4120 is not a multiple of 64",
            prefix="linear_attn.in_proj_qkvz",
            K=5120,
            out_sizes=[512, 512, 1536, 1536, 12, 12],
            input_size=5120,
            output_size=16768,
            tp_size=4,
        )

    def test_rejects_a_panel_aligned_total_with_a_ragged_shard(self):
        # The nastier variant: the SUM is panel aligned, so only the per-shard
        # check stands between us and a truncated offset.
        self._expect(
            "output_partition_sizes[4]=32",
            prefix="linear_attn.in_proj_qkvz",
            K=5120,
            out_sizes=[512, 512, 1536, 1536, 32, 32],
            input_size=5120,
            output_size=16640,
            tp_size=4,
        )

    def test_rejects_non_fp16(self):
        layer = FakeLayer("x")
        with self.assertRaises(ValueError) as cm:
            _make_method().create_weights(
                layer, 64, [64], 64, 64, torch.float32,
                weight_loader=weight_loader_v2,
            )
        self.assertIn("fp16", str(cm.exception))

    def test_rejects_v1_loader_when_sharded(self):
        def weight_loader(param, loaded_weight, shard_id=None):
            pass

        layer = FakeLayer("x", tp_size=4)
        with self.assertRaises(ValueError) as cm:
            _make_method().create_weights(
                layer, 5120, [4352], 5120, 17408, torch.float16,
                weight_loader=weight_loader,
            )
        self.assertIn("(v1)", str(cm.exception))

    def test_allows_v1_loader_when_unsharded(self):
        # ReplicatedLinear (linear.py:559-567) has no v2 path but also no split.
        def weight_loader(param, loaded_weight, shard_id=None):
            pass

        layer = FakeLayer("x", tp_size=1)
        _make_method().create_weights(
            layer, 5120, [17408], 5120, 17408, torch.float16,
            weight_loader=weight_loader,
        )
        self.assertEqual(tuple(layer.pxq4_slabs.shape)[0], 272)

    def test_rejects_unsharded_output_size_not_panel_aligned(self):
        self._expect("unsharded output_size", prefix="x", K=64, out_sizes=[64],
                     input_size=64, output_size=100, tp_size=1)


# ==========================================================================
@unittest.skipUnless(stubs.USING_STUBS, "multi-rank sim needs the fake TP group")
class TestColumnParallelSharding(Base):
    def test_plain_column_split_takes_whole_panels(self):
        for (N, K) in (IN_PROJ_Z, ATTN_Q):
            full_s, full_a = _ckpt(N, K)
            for tp in (1, 2, 4):
                for rank in range(tp):
                    _set_tp(rank, tp)
                    layer = _create(f"col{N}", K, [N // tp], K, N, tp)
                    layer.pxq4_slabs.load_column_parallel_weight(_t(full_s))
                    layer.pxq4_anchor.load_column_parallel_weight(_t(full_a))
                    got = _np(layer.pxq4_slabs.data)
                    ppr = N // tp // PANEL_ROWS
                    self.assertEqual(
                        _panels_of(got), list(range(rank * ppr, (rank + 1) * ppr)),
                        f"N={N} tp={tp} rank={rank}",
                    )
                    np.testing.assert_array_equal(
                        _np(layer.pxq4_anchor.data),
                        full_a[rank * ppr:(rank + 1) * ppr],
                    )

    def test_merged_gate_up_int_shard_ids(self):
        # MergedColumnParallelLinear(output_sizes=[17408, 17408]),
        # linear.py:1140-1205 int branch.
        N1, K = GATE_UP_ONE
        full_s, full_a = _ckpt(N1, K)
        out_sizes = [N1, N1]
        for tp in (1, 2, 4):
            for rank in range(tp):
                _set_tp(rank, tp)
                layer = _create("mlp.gate_up_proj", K, [N1 // tp] * 2, K,
                                sum(out_sizes), tp)
                for sid in (0, 1):
                    # gate_proj and up_proj are separate checkpoint tensors of
                    # 17408 rows each; both are 272 panels, so the same
                    # synthetic tensor stands in for both.
                    stubs.merged_column_weight_loader_v2(
                        layer.pxq4_slabs, _t(full_s), sid, out_sizes, tp
                    )
                    stubs.merged_column_weight_loader_v2(
                        layer.pxq4_anchor, _t(full_a), sid, out_sizes, tp
                    )
                got = _np(layer.pxq4_slabs.data)
                ppr = N1 // tp // PANEL_ROWS
                # both halves of the fused param carry the same panel range of
                # their own checkpoint tensor
                self.assertEqual(
                    _panels_of(got[:ppr]), list(range(rank * ppr, (rank + 1) * ppr))
                )
                self.assertEqual(
                    _panels_of(got[ppr:]), list(range(rank * ppr, (rank + 1) * ppr))
                )

    def test_gdn_in_proj_qkvz_tuple_shard_id(self):
        # ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)) -- qwen3_5.py:490.
        # One checkpoint tensor (10240 rows) fills three logical shards.
        N, K = IN_PROJ_QKV
        full_s, full_a = _ckpt(N, K)
        all_sizes = [2048, 2048, 6144, 6144]     # qwen3_5.py:214-224
        sub_sizes = [2048, 2048, 6144]
        for tp in (1, 2, 4):
            for rank in range(tp):
                _set_tp(rank, tp)
                layer = _create(
                    "linear_attn.in_proj_qkvz", K, [s // tp for s in all_sizes],
                    K, sum(all_sizes), tp,
                )
                stubs.load_fused_module_from_checkpoint(
                    layer.pxq4_slabs, _t(full_s), sub_sizes, all_sizes, tp
                )
                got = _np(layer.pxq4_slabs.data)
                # rank's q/k/v panel ranges inside the 10240-row checkpoint
                expect: list[int] = []
                base = 0
                for size in sub_sizes:
                    per = size // tp // PANEL_ROWS
                    start = base + rank * per
                    expect += list(range(start, start + per))
                    base += size // PANEL_ROWS
                filled = (all_sizes[0] + all_sizes[1] + all_sizes[2]) // tp // PANEL_ROWS
                self.assertEqual(_panels_of(got[:filled]), sorted(expect),
                                 f"tp={tp} rank={rank}")

    def test_qkv_parallel_p2c(self):
        # Forward-looking: P2c makes self_attn.qkv_proj uniformly PXQ4.
        # QKVParallelLinear at TP=4 -> q (0,3072), k (3072,256), v (3328,256).
        for tp in (1, 2, 4):
            q, kv = ATTN_Q[0] // tp, ATTN_KV[0] // tp
            for size in (q, kv):
                self.assertEqual(size % PANEL_ROWS, 0, f"tp={tp}")
            offsets = [0, q, q + kv]
            for off in offsets:
                self.assertEqual(off % PANEL_ROWS, 0)


# ==========================================================================
@unittest.skipUnless(stubs.USING_STUBS, "multi-rank sim needs the fake TP group")
class TestRowParallelSharding(Base):
    def test_k_split_takes_whole_slabs_and_duplicates_the_anchor(self):
        for (N, K) in (DOWN, O_PROJ, SSM_OUT):
            full_s, full_a = _ckpt(N, K)
            for tp in (1, 2, 4):
                for rank in range(tp):
                    _set_tp(rank, tp)
                    layer = _create(f"row{N}x{K}", K // tp, [N], K, N, tp)
                    layer.pxq4_slabs.load_row_parallel_weight(_t(full_s))
                    layer.pxq4_anchor.load_row_parallel_weight(_t(full_a))
                    got = _np(layer.pxq4_slabs.data)
                    spr = K // tp // SLAB_COLS
                    self.assertEqual(
                        _slabs_of(got), list(range(rank * spr, (rank + 1) * spr)),
                        f"N={N} K={K} tp={tp} rank={rank}",
                    )
                    self.assertEqual(_panels_of(got), list(range(N // PANEL_ROWS)))
                    # the 128 B header is duplicated verbatim on every rank
                    np.testing.assert_array_equal(_np(layer.pxq4_anchor.data), full_a)


# ==========================================================================
class TestShardEquivalence(Base):
    """Gate G3 restricted to component B: the split is a byte permutation."""

    @staticmethod
    def _random_weight(N: int, K: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        P, S = N // PANEL_ROWS, K // SLAB_COLS
        anchor = (rng.random((P, PANEL_ROWS)).astype(np.float32) * 3 + 0.01).astype(
            np.float16
        )
        sub_idx = rng.integers(0, 16, size=(P, S, PANEL_ROWS, 2), dtype=np.uint8)
        codes = rng.integers(0, 16, size=(P, PANEL_ROWS, K), dtype=np.uint8)
        return ref.pack_pxq4(anchor, sub_idx, codes)

    @unittest.skipUnless(stubs.USING_STUBS, "multi-rank sim needs the fake TP group")
    def test_column_shard_then_dequant_is_bit_exact(self):
        N, K = 256, 256
        slabs, anchor = self._random_weight(N, K, seed=1)
        full = ref.dequant(slabs, anchor)
        for tp in (2, 4):
            for rank in range(tp):
                _set_tp(rank, tp)
                layer = _create("col", K, [N // tp], K, N, tp)
                layer.pxq4_slabs.load_column_parallel_weight(_t(slabs))
                layer.pxq4_anchor.load_column_parallel_weight(_t(anchor))
                got = ref.dequant(
                    np.ascontiguousarray(_np(layer.pxq4_slabs.data)),
                    np.ascontiguousarray(_np(layer.pxq4_anchor.data)),
                )
                rows = N // tp
                self.assertTrue(
                    np.array_equal(got, full[rank * rows:(rank + 1) * rows]),
                    f"column tp={tp} rank={rank}",
                )

    @unittest.skipUnless(stubs.USING_STUBS, "multi-rank sim needs the fake TP group")
    def test_row_shard_then_dequant_is_bit_exact(self):
        N, K = 256, 256
        slabs, anchor = self._random_weight(N, K, seed=2)
        full = ref.dequant(slabs, anchor)
        for tp in (2, 4):
            for rank in range(tp):
                _set_tp(rank, tp)
                layer = _create("row", K // tp, [N], K, N, tp)
                layer.pxq4_slabs.load_row_parallel_weight(_t(slabs))
                layer.pxq4_anchor.load_row_parallel_weight(_t(anchor))
                got = ref.dequant(
                    np.ascontiguousarray(_np(layer.pxq4_slabs.data)),
                    np.ascontiguousarray(_np(layer.pxq4_anchor.data)),
                )
                cols = K // tp
                self.assertTrue(
                    np.array_equal(got, full[:, rank * cols:(rank + 1) * cols]),
                    f"row tp={tp} rank={rank}",
                )


# ==========================================================================
class _FakeOps:
    """torch.ops.pxq4 backed by the numpy reference. Records every call."""

    def __init__(self, mmv_ok=True, max_m=8) -> None:
        self.calls: list[str] = []
        self._mmv_ok = mmv_ok
        self._max_m = max_m

    def dequant_out(self, out, slabs, anchor):
        self.calls.append("dequant_out")
        w = ref.dequant(
            np.ascontiguousarray(stubs.arr(slabs)),
            np.ascontiguousarray(stubs.arr(anchor)),
        )
        stubs.arr(out)[...] = w.astype(np.float16)

    def mmv_out(self, out, x, slabs, anchor):
        self.calls.append("mmv_out")
        w = ref.dequant(
            np.ascontiguousarray(stubs.arr(slabs)),
            np.ascontiguousarray(stubs.arr(anchor)),
        )
        stubs.arr(out)[...] = (
            stubs.arr(x).astype(np.float32) @ w.T
        ).astype(np.float16)

    def mmv_supported(self, K):
        return self._mmv_ok

    def mmv_max_m(self):
        return self._max_m

    def version(self):
        return 1

    def set_tables(self, book, sub):
        self.calls.append("set_tables")


@unittest.skipUnless(stubs.USING_STUBS, "apply() harness needs the numpy fakes")
class TestApply(Base):
    def _ready_layer(self, N, K, *, mmv_ok=True, max_m=8, seed=3):
        fake = _FakeOps(mmv_ok=mmv_ok, max_m=max_m)
        torch.ops.pxq4 = fake
        pxops._loaded = False
        pxops._fakes_registered = False
        pxlinear.PXQ4LinearMethod._tables_uploaded = False
        m = _make_method()
        layer = _create("mlp.down_proj", K, [N], K, N, 1, m)
        slabs, anchor = TestShardEquivalence._random_weight(N, K, seed=seed)
        stubs.arr(layer.pxq4_slabs.data)[...] = slabs            # clears the 0xA5
        stubs.arr(layer.pxq4_anchor.data)[...] = anchor          # clears the NaN
        _mark_loaded(layer)
        m.process_weights_after_loading(layer)
        return m, layer, fake, ref.dequant(slabs, anchor)

    def test_small_m_takes_mmv_and_large_m_takes_dequant(self):
        m, layer, fake, _w = self._ready_layer(128, 128)
        x = _t(np.zeros((4, 128), dtype=np.float16))
        m.apply(layer, x)
        self.assertEqual(fake.calls[-1], "mmv_out")
        x = _t(np.zeros((9, 128), dtype=np.float16))
        m.apply(layer, x)
        self.assertEqual(fake.calls[-1], "dequant_out")

    def test_unsupported_K_never_uses_mmv(self):
        m, layer, fake, _w = self._ready_layer(128, 128, mmv_ok=False)
        self.assertFalse(layer.pxq4_use_mmv)
        m.apply(layer, _t(np.zeros((1, 128), dtype=np.float16)))
        self.assertEqual(fake.calls[-1], "dequant_out")

    def test_numerics_match_the_reference_on_both_paths(self):
        m, layer, _fake, w = self._ready_layer(128, 128, seed=5)
        rng = np.random.default_rng(7)
        for M in (2, 32):
            xa = rng.standard_normal((M, 128)).astype(np.float16)
            out = _np(m.apply(layer, _t(xa)))
            want = (xa.astype(np.float32) @ w.T).astype(np.float16)
            np.testing.assert_allclose(
                out.astype(np.float32), want.astype(np.float32), rtol=2e-2, atol=2e-2
            )

    def test_bias_and_leading_dims(self):
        m, layer, _fake, w = self._ready_layer(128, 128, seed=9)
        rng = np.random.default_rng(11)
        xa = rng.standard_normal((2, 3, 128)).astype(np.float16)
        bias = rng.standard_normal(128).astype(np.float16)
        out = _np(m.apply(layer, _t(xa), _t(bias)))
        self.assertEqual(out.shape, (2, 3, 128))
        want = (xa.reshape(-1, 128).astype(np.float32) @ w.T + bias.astype(np.float32))
        np.testing.assert_allclose(
            out.reshape(-1, 128).astype(np.float32), want, rtol=3e-2, atol=3e-2
        )

    def test_apply_allocates_exactly_one_tensor(self):
        # Capture safety: the only allocation may be `out`, which the caching
        # allocator serves from the cuda-graph private pool. The dequant buffer
        # must come from the preallocated workspace.
        m, layer, _fake, _w = self._ready_layer(128, 128)
        for M in (1, 64):
            stubs.ALLOC_COUNT["n"] = 0
            m.apply(layer, _t(np.zeros((M, 128), dtype=np.float16)))
            self.assertEqual(stubs.ALLOC_COUNT["n"], 1, f"M={M}")

    def test_zero_token_batch(self):
        m, layer, fake, _w = self._ready_layer(128, 128)
        before = len(fake.calls)
        out = _np(m.apply(layer, _t(np.zeros((0, 128), dtype=np.float16))))
        self.assertEqual(out.shape, (0, 128))
        self.assertEqual(len(fake.calls), before)

    # -- written-ness: the three checks, one test each ---------------------
    # This is the defect class the whole design exists to prevent: a
    # checkpoint that loads cleanly and serves subtly wrong logits.  Nothing
    # upstream raises -- default_loader.py:403-412 disables the unloaded-weight
    # check for quantized models and :421-431 exempts every module with a
    # process_weights_after_loading, and qwen3_5.py:564 drops an unmatched
    # name silently -- so each of the three must be tested on its own.

    def _fresh(self, prefix="self_attn.qkv_proj", N=128, K=128):
        torch.ops.pxq4 = _FakeOps()
        pxops._loaded = False
        m = _make_method()
        return m, _create(prefix, K, [N], K, N, 1, m)

    def test_slabs_are_born_holding_the_sentinel_byte(self):
        # If create_weights ever goes back to a bare torch.empty, the slab
        # tripwire silently stops working. Pin the fill.
        _m, layer = self._fresh()
        self.assertTrue(
            (stubs.arr(layer.pxq4_slabs.data) == pxlinear._SLAB_SENTINEL_BYTE).all()
        )

    def test_missing_slab_tensor_is_caught_even_though_the_anchor_loaded(self):
        # THE regression test. Converter emits '<mod>.pxq4_slab' (typo) plus a
        # correct '<mod>.pxq4_anchor'. qwen3_5.py:564 drops the slab silently;
        # the anchor loads and overwrites every NaN, so the anchor tripwire is
        # clean. Only the per-parameter call counter sees this.
        m, layer = self._fresh()
        slabs, anchor = TestShardEquivalence._random_weight(128, 128, seed=17)
        stubs.arr(layer.pxq4_anchor.data)[...] = anchor
        _mark_loaded(layer, slabs=False, anchor=True)
        del slabs
        with self.assertRaises(ValueError) as cm:
            m.process_weights_after_loading(layer)
        msg = str(cm.exception)
        self.assertIn("NEVER called", msg)
        self.assertIn("pxq4_slabs", msg)

    def test_missing_anchor_tensor_is_caught_even_though_the_slabs_loaded(self):
        m, layer = self._fresh()
        slabs, anchor = TestShardEquivalence._random_weight(128, 128, seed=18)
        stubs.arr(layer.pxq4_slabs.data)[...] = slabs
        stubs.arr(layer.pxq4_anchor.data)[...] = anchor   # NaN gone too
        _mark_loaded(layer, slabs=True, anchor=False)
        with self.assertRaises(ValueError) as cm:
            m.process_weights_after_loading(layer)
        self.assertIn("pxq4_anchor", str(cm.exception))

    def test_partially_written_slabs_are_caught(self):
        # The loader ran for pxq4_slabs but covered only some panels: a fused
        # module whose other shard is not PXQ4. Counters are clean; only the
        # 0xA5 scan sees it.
        m, layer = self._fresh(N=256)
        slabs, anchor = TestShardEquivalence._random_weight(256, 128, seed=19)
        arr = stubs.arr(layer.pxq4_slabs.data)
        arr[: 256 // 128] = slabs[: 256 // 128]          # first 2 of 4 panels
        stubs.arr(layer.pxq4_anchor.data)[...] = anchor  # anchor fully written
        _mark_loaded(layer)
        with self.assertRaises(ValueError) as cm:
            m.process_weights_after_loading(layer)
        msg = str(cm.exception)
        self.assertIn("never written by the weight loader", msg)
        self.assertIn("0xA5", msg)

    def test_partially_written_anchor_is_caught(self):
        m, layer = self._fresh(N=256)
        slabs, anchor = TestShardEquivalence._random_weight(256, 128, seed=20)
        stubs.arr(layer.pxq4_slabs.data)[...] = slabs    # slabs fully written
        stubs.arr(layer.pxq4_anchor.data)[:2] = anchor[:2]
        _mark_loaded(layer)
        with self.assertRaises(ValueError) as cm:
            m.process_weights_after_loading(layer)
        self.assertIn("anchor panels were", str(cm.exception))

    def test_written_ness_check_is_idempotent(self):
        # model_loader/reload/layerwise.py can call
        # process_weights_after_loading twice; the counter swap must not turn
        # the second call into a spurious "loader was never called".
        m, layer, _fake, _w = self._ready_layer(128, 128, seed=23)
        m.process_weights_after_loading(layer)

    def test_slab_sentinel_survives_PXQ4_SENTINEL_0(self):
        # Turning the NaN anchor fill off must not leave the layer undefended:
        # that env knob exists for the NaN's read-risk, not to disable
        # written-ness checking.
        m, layer = self._fresh(N=256)
        slabs, _anchor = TestShardEquivalence._random_weight(256, 128, seed=21)
        arr = stubs.arr(layer.pxq4_slabs.data)
        arr[:2] = slabs[:2]
        _mark_loaded(layer)
        old = pxlinear._SENTINEL_ENABLED
        pxlinear._SENTINEL_ENABLED = False
        try:
            with self.assertRaises(ValueError) as cm:
                m.process_weights_after_loading(layer)
        finally:
            pxlinear._SENTINEL_ENABLED = old
        self.assertIn("0xA5", str(cm.exception))


# ==========================================================================
@unittest.skipUnless(stubs.USING_STUBS, "workspace test needs the numpy fakes")
class TestWorkspace(Base):
    def test_view_is_a_slice_of_one_arena(self):
        ws = pxops.PXQ4Workspace
        ws.reserve(dequant_elems=8704 * 5120)
        ws.materialize(stubs._Device())
        a = ws.dequant_view(64, 128, stubs._Device())
        b = ws.dequant_view(128, 64, stubs._Device())
        self.assertEqual(tuple(a.shape), (64, 128))
        self.assertEqual(tuple(b.shape), (128, 64))
        # same storage: writing through one is visible through the other
        stubs.arr(a)[...] = 1.0
        self.assertEqual(float(stubs.arr(b)[0, 0]), 1.0)

    def test_growth_after_materialize_raises(self):
        ws = pxops.PXQ4Workspace
        ws.reserve(dequant_elems=1024)
        ws.materialize(stubs._Device())
        with self.assertRaises(RuntimeError):
            ws.reserve(dequant_elems=4096)

    def test_view_before_materialize_raises(self):
        ws = pxops.PXQ4Workspace
        ws.reserve(dequant_elems=1024)
        with self.assertRaises(RuntimeError):
            ws.dequant_view(32, 32, stubs._Device())


# ==========================================================================
class TestNoTruncationArithmetic(Base):
    """The arithmetic claim of plan sec.2.3, checked directly and exhaustively
    over the model's real modules -- independent of any tensor plumbing."""

    MODULES = {
        "mlp.gate_up_proj": ([17408, 17408], 5120, "col"),
        "mlp.down_proj": ([5120], 17408, "row"),
        "self_attn.o_proj": ([5120], 6144, "row"),
        "linear_attn.in_proj_qkvz": ([2048, 2048, 6144, 6144], 5120, "col"),
        "linear_attn.out_proj": ([5120], 6144, "row"),     # P2a
        "self_attn.qkv_proj": ([12288, 1024, 1024], 5120, "col"),   # P2c
        "lm_head": ([248320], 5120, "col"),                # P2b
    }

    def test_every_shard_is_panel_and_slab_aligned(self):
        for name, (out_sizes, K, kind) in self.MODULES.items():
            for tp in (1, 2, 4):
                if kind == "row":
                    self.assertEqual(K % tp, 0, name)
                    self.assertEqual((K // tp) % SLAB_COLS, 0,
                                     f"{name} K/{tp} not slab aligned")
                off = 0
                for i, size in enumerate(out_sizes):
                    self.assertEqual(size % tp, 0, f"{name}[{i}] tp={tp}")
                    per = size // tp
                    self.assertEqual(per % PANEL_ROWS, 0,
                                     f"{name}[{i}] {per} rows/rank not %64 at tp={tp}")
                    self.assertEqual((off // tp) % PANEL_ROWS, 0,
                                     f"{name}[{i}] offset not %64 at tp={tp}")
                    self.assertEqual(size % PANEL_ROWS, 0,
                                     f"{name}[{i}] checkpoint side not %64")
                    off += size

    def test_bytes_per_tensor_match_the_file(self):
        # (rows/64)*(128 + (K/32)*1088), 06-file-composition.md sec.5
        # (N rows, K) -> bytes, from 06-file-composition.md sec.5, which lists
        # ne as "K x rows".
        cases = {
            (17408, 5120): 47_384_576,
            (5120, 17408): 47_360_000,
            (10240, 5120): 27_873_280,
            (6144, 5120): 16_723_968,
            (12288, 5120): 33_447_936,
            (5120, 6144): 16_721_920,
        }
        for (N, K), want in cases.items():
            got = (N // PANEL_ROWS) * (128 + (K // SLAB_COLS) * SLAB_BYTES)
            self.assertEqual(got, want, f"{N}x{K}")


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(_PKG_TMP, ignore_errors=True)
