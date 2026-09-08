# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the PXQ4 quantization config.

DESTINATION IN THE REPO OF PLAN 09: ``tests/test_config.py``.

No GPU, no torch, no vLLM required -- ``pxq4_config_stubs.install_stubs()``
supplies fakes when the real packages are absent, and steps aside when they are
present (so the same file runs inside the vLLM container as a stronger test).

The highest-value case here is ``test_gdn_split_probe_agrees``: it embeds a
*verbatim copy* of ``_uses_split_gdn_input_projections`` from
qwen3_5.py:127-157 and asserts our config object drives it to True.  That
probe returning False is the failure mode that loads cleanly and generates
subtly wrong tokens (parameter.py:605-610 truncates the 12-row shard without
raising), so it is worth pinning against a copy of the real code rather than
against a paraphrase.

Run:  python3 -m pytest test_pxq4_config.py -q
  or: python3 test_pxq4_config.py        (no pytest needed)
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pxq4_config_stubs as stubs  # noqa: E402

stubs.install_stubs()

_TMP = Path(tempfile.mkdtemp(prefix="pxq4cfg-"))
sys.path.insert(0, stubs.build_package(_TMP, _HERE / "pxq4_config.py"))

from pxq4_vllm import config as cfg  # noqa: E402
from pxq4_vllm.linear import PXQ4LinearMethod  # noqa: E402

# Snapshot the registration made by importing cfg. Later tests build extra
# copies of the package under other names, and each import re-runs the
# decorator, so the registry would otherwise show the last copy.
try:
    from vllm.model_executor.layers.quantization import (  # noqa: E402
        get_quantization_config as _get_quantization_config,
    )

    _REGISTERED_AT_IMPORT = _get_quantization_config("pxq4")
except (ImportError, AttributeError, ValueError):
    _REGISTERED_AT_IMPORT = getattr(stubs, "REGISTERED", {}).get("pxq4")
from vllm.model_executor.layers.linear import (  # noqa: E402
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# ggml/include/ggml-pxq6-tables.h:32-45, PXQ6_BOOK_INIT / PXQ6_SUB16_INIT,
# hex-float literals evaluated to fp32.  These are what the converter must copy
# out of the gguf KVs pxa.pxq6.book / pxa.pxq6.sub.
BOOK = [
    -0.98779296875, -0.7353515625, -0.55859375, -0.419677734375,
    -0.301025390625, -0.1944580078125, -0.09552001953125, 0.0,
    0.084716796875, 0.1712646484375, 0.261962890625, 0.360595703125,
    0.47119140625, 0.6005859375, 0.765625, 1.0,
]
SUB = [
    0.2147216796875, 0.303466796875, 0.362060546875, 0.408935546875,
    0.449951171875, 0.4873046875, 0.52294921875, 0.55859375,
    0.59423828125, 0.6318359375, 0.671875, 0.71630859375,
    0.7666015625, 0.82470703125, 0.89599609375, 0.98779296875,
]

BACKBONE_MAP = (
    "attn_q,attn_qkv,attn_output,attn_gate_ch,shexp,ffn_dense=tier+1;"
    "attn_k,attn_v=q8_0;attn_gate_head=f16;token_embd=q6_k;output=q8_0"
)


def p1_config_dict() -> dict:
    """Plan 09 sec.5.5, plus the ``modules_to_not_convert`` recommendation."""
    return {
        "quant_method": "pxq4",
        "pxq4_version": 1,
        "tier": "core",
        "type_id": 252,
        "panel_rows": 64,
        "slab_cols": 32,
        "slab_bytes": 1088,
        "header_bytes": 128,
        "book": list(BOOK),
        "sub": list(SUB),
        "backbone_rev": 2,
        "backbone_map": BACKBONE_MAP,
        "pxq4_modules": [
            "mlp.gate_up_proj",
            "mlp.down_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkvz",
        ],
        "ignore": [
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_ba",
            "linear_attn.out_proj",
            "self_attn.qkv_proj",
            "lm_head",
            "model.visual",
        ],
        "modules_to_not_convert": ["mtp"],
    }


def p2c_config_dict() -> dict:
    d = p1_config_dict()
    d["pxq4_modules"] = [
        "mlp.gate_up_proj",
        "mlp.down_proj",
        "self_attn.o_proj",
        "self_attn.qkv_proj",
        "linear_attn.in_proj_qkvz",
        "linear_attn.out_proj",
    ]
    d["ignore"] = [
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_ba",
        "lm_head",
        "model.visual",
    ]
    return d


class ParallelLMHead:
    """Dispatch keys on the class *name* (to avoid importing
    vocab_parallel_embedding at plugin-load time), so the stand-in must carry
    the real name. It is deliberately NOT a LinearBase."""


class Attention:
    pass


L = 3  # a full-attention block (L % 4 == 3); GDN blocks are the others
GDN_PREFIX = "language_model.model.layers.2.linear_attn"
ATTN_PREFIX = "language_model.model.layers.3.self_attn"
MLP_PREFIX = "language_model.model.layers.3.mlp"
VIS_PREFIX = "visual.blocks.7"


_FAKE_CACHE: dict[type, type] = {}


def _fake_of(base: type) -> type:
    """A real ``isinstance(x, LinearBase)`` object that skips LinearBase's
    __init__.  The real __init__ (linear.py:470-500) would itself call
    ``quant_config.get_quant_method`` -- the very thing under test -- and needs
    input_size/output_size/tp state we do not have.  Bypassing it keeps the
    isinstance dispatch honest while staying a pure unit test."""
    if base not in _FAKE_CACHE:

        class _Fake(base):  # type: ignore[misc, valid-type]
            def __init__(self, prefix: str = "") -> None:
                object.__setattr__(self, "prefix", prefix)

        _Fake.__name__ = f"Fake{base.__name__}"
        _FAKE_CACHE[base] = _Fake
    return _FAKE_CACHE[base]


def mk(prefix: str, cls=None):
    return _fake_of(cls or LinearBase)(prefix=prefix)


# --------------------------------------------------------------------------
# _matches
# --------------------------------------------------------------------------

def test_matches_exact_suffix_substring():
    assert cfg._matches("mlp.down_proj", ["mlp.down_proj"])
    assert cfg._matches("a.b.mlp.down_proj", ["mlp.down_proj"])
    assert cfg._matches("a.b.mlp.down_proj.extra", ["mlp.down_proj"])  # substring clause
    assert not cfg._matches("a.b.mlp.up_proj", ["mlp.down_proj"])


def test_matches_does_not_catch_vision_tower():
    """The substring clause in _matches is only safe because the vision tower
    uses linear_fc1/linear_fc2/attn.qkv/attn.proj (qwen3_vl.py:395-410,
    :496-510).  Pin that, so a future model swap fails here rather than by
    demanding PXQ4 weights for an fp16 vision linear."""
    pats = p1_config_dict()["pxq4_modules"]
    for name in (
        f"{VIS_PREFIX}.mlp.linear_fc1",
        f"{VIS_PREFIX}.mlp.linear_fc2",
        f"{VIS_PREFIX}.attn.qkv",
        f"{VIS_PREFIX}.attn.proj",
        "visual.merger.linear_fc1",
        "visual.deepstack_merger_list.0.linear_fc2",
    ):
        assert not cfg._matches(name, pats), name


# --------------------------------------------------------------------------
# from_config
# --------------------------------------------------------------------------

def test_from_config_populates_all_ignore_surfaces():
    """qwen3_5.py:127-157 unions four different attributes; all must be set."""
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    for entry in cfg.REQUIRED_IGNORE_ENTRIES:
        assert entry in c.ignore
        assert entry in c.ignored_layers
        assert entry in c.config["ignore"]
    assert c.modules_to_not_convert == ["mtp"]
    assert c.backbone_rev == 2
    assert c.tier == "core"
    assert c.book == tuple(BOOK)
    assert c.sub == tuple(SUB)
    # base_config.py:72-76 must have run
    assert isinstance(c.packed_modules_mapping, dict)


def test_from_config_rejects_missing_gdn_ignore_entries():
    d = p1_config_dict()
    d["ignore"] = ["lm_head"]
    try:
        cfg.PXQ4Config.from_config(d)
    except ValueError as e:
        assert "in_proj_a" in str(e) and "in_proj_b" in str(e)
    else:
        raise AssertionError("expected ValueError for missing GDN ignore entries")


def test_from_config_rejects_foreign_geometry():
    for key, bad in (
        ("type_id", 250),        # the RETIRED MXFP4-repack id documented in pxq4.cuh
        ("panel_rows", 32),
        ("slab_cols", 16),
        ("slab_bytes", 1152),    # the PXQ6HQ slab size
        ("header_bytes", 64),
    ):
        d = p1_config_dict()
        d[key] = bad
        try:
            cfg.PXQ4Config.from_config(d)
        except ValueError as e:
            assert key in str(e)
        else:
            raise AssertionError(f"expected ValueError for {key}={bad}")


def test_from_config_rejects_bad_version_and_tables():
    d = p1_config_dict(); d["pxq4_version"] = 2
    try:
        cfg.PXQ4Config.from_config(d)
    except ValueError as e:
        assert "pxq4_version" in str(e)
    else:
        raise AssertionError("expected ValueError for pxq4_version=2")

    d = p1_config_dict(); del d["book"]
    try:
        cfg.PXQ4Config.from_config(d)
    except ValueError as e:
        assert "book" in str(e)
    else:
        raise AssertionError("expected ValueError for missing book")

    d = p1_config_dict(); d["sub"] = SUB[:8]
    try:
        cfg.PXQ4Config.from_config(d)
    except ValueError as e:
        assert "16 entries" in str(e)
    else:
        raise AssertionError("expected ValueError for short sub table")


def test_from_config_rejects_empty_module_list():
    d = p1_config_dict(); d["pxq4_modules"] = []
    # empty list is not the same as absent: absent falls back to P1 with a
    # warning, explicitly-empty is a contradiction with quant_method=pxq4.
    del d["pxq4_modules"]
    c = cfg.PXQ4Config.from_config(d)
    assert list(c.pxq4_modules) == list(cfg.P1_PXQ4_MODULES)


def test_from_config_rejects_foreign_quant_method():
    d = p1_config_dict(); d["quant_method"] = "compressed-tensors"
    try:
        cfg.PXQ4Config.from_config(d)
    except ValueError as e:
        assert "compressed-tensors" in str(e)
    else:
        raise AssertionError("expected ValueError for foreign quant_method")


# --------------------------------------------------------------------------
# The GDN split probe, verbatim from the fork
# --------------------------------------------------------------------------

def _uses_split_gdn_input_projections(quant_config) -> bool:
    """VERBATIM COPY of vllm/model_executor/models/qwen3_5.py:127-157
    (fork 1Cat-vLLM @ 2ceb15066).  Do not "clean up" -- its value is that it is
    identical to the code that will actually run."""
    ignored_modules: list[str] = []

    def add_ignored_modules(value: object) -> None:
        if not value:
            return
        if isinstance(value, str):
            ignored_modules.append(value)
            return
        try:
            ignored_modules.extend(str(module) for module in value)
        except TypeError:
            return

    for attr_name in ("modules_to_not_convert", "ignored_layers", "ignore"):
        add_ignored_modules(getattr(quant_config, attr_name, None))
    raw_config = getattr(quant_config, "config", None)
    if isinstance(raw_config, dict):
        add_ignored_modules(raw_config.get("ignore"))
    if not ignored_modules:
        return False
    return any(
        module_name == "linear_attn"
        or module_name.endswith(".linear_attn")
        or ("linear_attn.in_proj_a" in module_name)
        or ("linear_attn.in_proj_b" in module_name)
        for module_name in ignored_modules
    )


def test_gdn_split_probe_agrees():
    """If this fails, in_proj_qkvz gains two 48-row shards -> 12 rows/rank at
    TP=4 -> silent truncation in parameter.py:605-610."""
    assert _uses_split_gdn_input_projections(cfg.PXQ4Config.from_config(p1_config_dict()))
    assert _uses_split_gdn_input_projections(cfg.PXQ4Config.from_config(p2c_config_dict()))


def test_mtp_probe_agrees():
    """qwen3_5_mtp.py:447-470 -- verbatim shape of the check."""
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    quant_cfg = c.config
    mtc = quant_cfg.get("modules_to_not_convert")
    assert mtc and any(str(m) in ("mtp", "model.mtp") for m in mtc)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

P1_EXPECT_PXQ4 = [
    f"{MLP_PREFIX}.gate_up_proj",
    f"{MLP_PREFIX}.down_proj",
    f"{ATTN_PREFIX}.o_proj",
    f"{GDN_PREFIX}.in_proj_qkvz",
]

P1_EXPECT_FP16 = [
    f"{ATTN_PREFIX}.qkv_proj",
    f"{GDN_PREFIX}.in_proj_ba",
    f"{GDN_PREFIX}.out_proj",          # ggml ssm_out, MXFP4 on disk -> fp16
    f"{VIS_PREFIX}.mlp.linear_fc1",
    f"{VIS_PREFIX}.mlp.linear_fc2",
    f"{VIS_PREFIX}.attn.qkv",
    f"{VIS_PREFIX}.attn.proj",
    "visual.merger.linear_fc1",
]


def test_dispatch_p1():
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    for prefix in P1_EXPECT_PXQ4:
        m = c.get_quant_method(mk(prefix), prefix)
        assert isinstance(m, PXQ4LinearMethod), f"{prefix} -> {type(m).__name__}"
        assert m.quant_config is c
    for prefix in P1_EXPECT_FP16:
        m = c.get_quant_method(mk(prefix), prefix)
        assert isinstance(m, UnquantizedLinearMethod), f"{prefix} -> {type(m).__name__}"


def test_dispatch_p2c_promotes_qkv_and_out_proj():
    c = cfg.PXQ4Config.from_config(p2c_config_dict())
    for prefix in (*P1_EXPECT_PXQ4, f"{ATTN_PREFIX}.qkv_proj", f"{GDN_PREFIX}.out_proj"):
        assert isinstance(c.get_quant_method(mk(prefix), prefix), PXQ4LinearMethod), prefix
    # b/a must STILL be unquantized in P2c -- 48 rows can never be panel-aligned.
    m = c.get_quant_method(mk(f"{GDN_PREFIX}.in_proj_ba"), f"{GDN_PREFIX}.in_proj_ba")
    assert isinstance(m, UnquantizedLinearMethod)


def test_never_returns_none_for_a_linear():
    """linear.py:492-495 raises ValueError on a falsy return."""
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    for prefix in [*P1_EXPECT_PXQ4, *P1_EXPECT_FP16, "some.unknown.linear", ""]:
        assert c.get_quant_method(mk(prefix), prefix) is not None, prefix
    # concrete LinearBase subclasses used by this model
    for cls in (MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear):
        prefix = f"{MLP_PREFIX}.gate_up_proj"
        assert c.get_quant_method(mk(prefix, cls), prefix) is not None


def test_non_linear_layers_return_none():
    """vocab_parallel_embedding.py:479-482 / attention.py:159 accept None."""
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    assert c.get_quant_method(ParallelLMHead(), "lm_head") is None
    assert c.get_quant_method(Attention(), f"{ATTN_PREFIX}.attn") is None


def test_pxq4_lm_head_is_rejected_at_parse_time():
    """THE REGRESSION TEST for the P2b/P2c load failure.

    A converter that lists ``lm_head`` in ``pxq4_modules`` produces a checkpoint
    this build cannot load at all -- ParallelLMHead forces the v1 vocab
    weight_loader (vocab_parallel_embedding.py:520-527, :633-682), which cannot
    slice the 64-row panel layout at TP>1 and cannot fill both pxq4_slabs and
    pxq4_anchor.  Before this test the disagreement surfaced only deep in model
    construction, after the engine had started and the weights were open.  It
    must now fail at from_config, i.e. at engine start.
    """
    for name in ("lm_head", "model.language_model.lm_head", "model.embed_tokens"):
        d = p1_config_dict()
        d["pxq4_modules"] = [*d["pxq4_modules"], name]
        d["ignore"] = [x for x in d["ignore"] if x != "lm_head"]
        try:
            cfg.PXQ4Config.from_config(d)
        except ValueError as e:
            assert name in str(e), (name, str(e))
            assert "P2b" in str(e), str(e)
        else:
            raise AssertionError(f"expected ValueError for a PXQ4 {name}")


def test_pxq4_lm_head_backstop_still_raises_in_dispatch():
    """Defence in depth: the get_quant_method guard must survive a config that
    got past _validate (direct construction, or post-construction mutation)."""
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    c.pxq4_modules = [*c.pxq4_modules, "lm_head"]
    c._ignore_for_dispatch = tuple(
        x for x in c._ignore_for_dispatch if x != "lm_head"
    )
    try:
        c.get_quant_method(ParallelLMHead(), "lm_head")
    except NotImplementedError as e:
        assert "P2b" in str(e)
    else:
        raise AssertionError("expected NotImplementedError for a PXQ4 lm_head")


def test_namemap_policies_are_servable():
    """CROSS-COMPONENT CONTRACT. The defect this pins: component A's converter
    wrote ``sorted(POLICY_MODULES[policy])`` verbatim into
    ``quantization_config['pxq4_modules']`` (convert.py:295) while its p2b/p2c
    entries contained ``lm_head``, which this config rejects -- so every p2b/p2c
    run died during model construction and nothing caught it earlier.

    Skips (does not fail) when the converter is not importable, because the two
    components ship to different directories in the plan-09 repo.
    """
    try:
        from gguf_to_vllm import namemap as NM  # noqa: PLC0415
    except ImportError:
        print("   (skip: converter namemap not importable here)")
        return
    for policy, modules in sorted(NM.POLICY_MODULES.items()):
        bad = cfg._unservable_entries(sorted(modules))
        assert not bad, (
            f"namemap.POLICY_MODULES[{policy!r}] serves {bad}, which "
            f"PXQ4Config._validate rejects. The two components disagree about "
            f"what {policy} means; fix the converter, not the config."
        )
        # And the real thing: the exact dict convert.py would emit must parse.
        d = p1_config_dict()
        d["pxq4_modules"] = sorted(modules)
        d["ignore"] = NM.ignore_list(policy)
        c = cfg.PXQ4Config.from_config(d)
        assert set(c.pxq4_modules) == set(modules), policy


def test_ignore_wins_over_allow_list():
    d = p1_config_dict()
    d["pxq4_modules"] = [*d["pxq4_modules"], "self_attn.qkv_proj"]  # also in ignore
    c = cfg.PXQ4Config.from_config(d)
    prefix = f"{ATTN_PREFIX}.qkv_proj"
    assert isinstance(c.get_quant_method(mk(prefix), prefix), UnquantizedLinearMethod)


def test_modules_to_not_convert_also_routes_to_fp16():
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    prefix = "mtp.layers.0.mlp.down_proj"
    assert isinstance(c.get_quant_method(mk(prefix), prefix), UnquantizedLinearMethod)


# --------------------------------------------------------------------------
# sec.3.1 invariant
# --------------------------------------------------------------------------

def test_fused_uniformity_violation_raises():
    """A fused module cannot be half PXQ4: create_weights allocates one slab
    tensor for all output_partition_sizes."""
    d = p1_config_dict()
    d["ignore"] = [*d["ignore"], "in_proj_z"]
    c = cfg.PXQ4Config.from_config(d)
    c.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    prefix = f"{GDN_PREFIX}.in_proj_qkvz"
    try:
        c.get_quant_method(mk(prefix), prefix)
    except ValueError as e:
        assert "uniformly PXQ4" in str(e)
    else:
        raise AssertionError("expected ValueError for a half-ignored fused module")


def test_fused_uniformity_passes_for_real_mapping():
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    c.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    c.validate_fused_uniformity()  # must not raise


# --------------------------------------------------------------------------
# Registration / identity / override
# --------------------------------------------------------------------------

def test_abc_surface():
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    import torch

    assert c.get_name() == "pxq4"
    assert c.get_supported_act_dtypes() == [torch.float16]
    assert torch.bfloat16 not in c.get_supported_act_dtypes()  # no bf16 on sm_70
    assert cfg.PXQ4Config.get_min_capability() == 70
    assert cfg.PXQ4Config.get_config_filenames() == []


def test_registration_happened_on_import():
    assert _REGISTERED_AT_IMPORT is cfg.PXQ4Config, _REGISTERED_AT_IMPORT
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    assert "pxq4" in QUANTIZATION_METHODS


def test_override_quantization_method():
    assert (
        cfg.PXQ4Config.override_quantization_method(p1_config_dict(), None) == "pxq4"
    )
    # must not hijack anybody else's checkpoint: config/model.py:1045-1067 calls
    # this classmethod with every checkpoint's quant config.
    for foreign in (
        {"quant_method": "compressed-tensors", "ignore": []},
        {"quant_method": "awq"},
        {},
        None,
        "not-a-dict",
    ):
        assert cfg.PXQ4Config.override_quantization_method(foreign, None) is None


def test_maybe_update_config_warns_about_mtp(caplog=None):
    class _TextCfg:
        mtp_num_hidden_layers = 1

    class _HFCfg:
        text_config = _TextCfg()

    d = p1_config_dict()
    d["modules_to_not_convert"] = []
    c = cfg.PXQ4Config.from_config(d)

    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lg = logging.getLogger(cfg.__name__)
    h = _H()
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    try:
        c.maybe_update_config("m", hf_config=_HFCfg())
    finally:
        lg.removeHandler(h)
    assert any("mtp" in r for r in records), records

    # and it must stay quiet when the opt-out is present
    c2 = cfg.PXQ4Config.from_config(p1_config_dict())
    records.clear()
    lg.addHandler(h)
    try:
        c2.maybe_update_config("m", hf_config=_HFCfg())
    finally:
        lg.removeHandler(h)
    assert not records


# --------------------------------------------------------------------------
# Codebook cross-check against component A's reference tables
# --------------------------------------------------------------------------

def _load_variant(pkg_name: str, reference_tables):
    """Import a second copy of config.py under *pkg_name*, carrying its own
    reference.py. Keeps the shared pxq4_vllm package loaded so class identities
    used by the dispatch tests stay stable."""
    import importlib

    tmp = Path(tempfile.mkdtemp(prefix=f"{pkg_name}-"))
    root = stubs.build_package(
        tmp, _HERE / "pxq4_config.py", pkg_name=pkg_name,
        reference_tables=reference_tables,
    )
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"{pkg_name}.config")


def test_book_mismatch_against_reference_is_fatal():
    """The CUDA kernels read the compiled-in __device__ pxq6_book_g
    (pxq6.cuh:79), never the checkpoint.  A file quantized with a different
    PXA_PXQ6_BOOK would dequantize wrongly with no other symptom, so the
    mismatch must be caught at load."""
    bad_book = list(BOOK)
    bad_book[3] = -0.5
    cfg2 = _load_variant("pxq4_badbook", (bad_book, list(SUB)))
    try:
        cfg2.PXQ4Config.from_config(p1_config_dict())
    except ValueError as e:
        assert "book" in str(e)
    else:
        raise AssertionError("expected ValueError on book mismatch")


def test_book_match_against_reference_is_accepted():
    cfg3 = _load_variant("pxq4_goodbook", (list(BOOK), list(SUB)))
    c = cfg3.PXQ4Config.from_config(p1_config_dict())
    assert c.book == tuple(BOOK)


# --------------------------------------------------------------------------

def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
