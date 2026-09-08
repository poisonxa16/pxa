# SPDX-License-Identifier: Apache-2.0
"""PXQ4 quantization config for vLLM (sm_70 / Volta).

DESTINATION IN THE REPO OF PLAN 09: ``src/pxq4_vllm/config.py``.

This is component "quant-config" of the PXQ4-in-vLLM port described in
``09-chosen-design.md``.  It owns exactly one thing: telling vLLM that a
checkpoint whose ``config.json`` carries ``quantization_config.quant_method
== "pxq4"`` should be served by our linear method, and which modules of the
model that applies to.

It deliberately owns *no* numerics, *no* parameter shapes and *no* sharding
logic.  Those live in ``linear.py`` / ``parameters.py`` (component B) and
``csrc/`` (component C).

--------------------------------------------------------------------------
Verified against the fork at /opt/1Cat-vLLM, git 2ceb15066 (v0.1.dev1+g2ceb15066).
Every file:line below was read in that tree.

  * ``QuantizationConfig`` ABC and the six abstract members:
        base_config.py:69-163  (get_name :78, get_supported_act_dtypes :83,
        get_min_capability :88 @classmethod, get_config_filenames :99
        @staticmethod, from_config :105 @classmethod, get_quant_method :150)
    ``__init__`` at base_config.py:72-76 seeds ``packed_modules_mapping``;
    ``super().__init__()`` is mandatory because model_loader/utils.py:290
    overwrites that attribute *by reference* and would otherwise AttributeError
    on a config that never created it.
  * ``register_quantization_config`` : quantization/__init__.py:57-104.
    Appends to the runtime ``QUANTIZATION_METHODS`` list (:46/:92) and stores
    the class in ``_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`` (:101), which is merged
    *last* into the lookup table at :232 and therefore overrides built-ins.
  * Checkpoint self-selection: config/model.py:1002-1090 ``_verify_quantization``
    reads ``quant_cfg["quant_method"]`` (:1011) and then probes every registered
    method's ``override_quantization_method`` (:1048).  A custom name that is in
    ``QUANTIZATION_METHODS`` but not in ``get_args(QuantizationMethods)`` is
    exempt from the "override not in the overrides list" raise at :1055-1064.
  * Instantiation: model_loader/weight_utils.py:263-318 -> ``from_config`` is
    handed the raw ``config.json["quantization_config"]`` dict verbatim.
    ``get_config_filenames()`` is only consulted on the file-based fallback
    path further down, so returning ``[]`` is correct.
  * Capability + dtype gate: config/vllm.py:600-628.  ``get_min_capability()``
    is compared against the device capability, and ``model_config.dtype`` must
    be in ``get_supported_act_dtypes()`` or the engine raises.
  * ``get_quant_method`` consumers and their None-handling:
        linear.py:492   ``elif quant_method := ...`` -> falsy raises ValueError
                        "All linear layers should support quant method."
                        => we must NEVER return None for a LinearBase.
        vocab_parallel_embedding.py:479-482 -> None means UnquantizedEmbeddingMethod.
        attention/attention.py:159 and mla_attention.py:928 -> None is fine.
        fused_moe/layer.py:349-364 -> None means UnquantizedFusedMoEMethod.
  * The GDN split trap: qwen3_5.py:127-157 ``_uses_split_gdn_input_projections``
    unions ``modules_to_not_convert`` / ``ignored_layers`` / ``ignore`` off the
    config object plus ``config["ignore"]`` if ``.config`` is a dict, and
    returns True iff some entry is ``linear_attn``, ends with ``.linear_attn``,
    or *contains* ``linear_attn.in_proj_a`` / ``linear_attn.in_proj_b``.
    We therefore expose all four surfaces.
  * The MTP branch: qwen3_5_mtp.py:447-505 reads
    ``quantization_config["modules_to_not_convert"]`` (NOT ``ignore``) and
    disables quantization for the whole draft model when it contains
    ``"mtp"`` or ``"model.mtp"``.
  * Prefix renaming cannot break suffix matching: the only WeightsMapper in
    play (qwen3_vl.py:1635-1641) rewrites *prefixes* only
    (``model.visual.``->``visual.``, ``model.language_model.``->
    ``language_model.model.``), never leaf module names.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm.model_executor.layers.quantization import QuantizationMethods
else:  # the fork aliases this to ``str`` outside TYPE_CHECKING (base_config.py:15)
    QuantizationMethods = str

logger = init_logger(__name__)

QUANT_METHOD_NAME = "pxq4"

# --------------------------------------------------------------------------
# On-disk geometry.  These are NOT tunables: they are the ggml type-252 layout,
# fixed by ggml-pxq6-tables.h:21-27 (QK=32, BM=64, SLAB_BYTES=1088,
# HDR_BYTES=128, ROW_META=2) and consumed by pxq6.cuh:317-346 / :520-526.
# The config's job here is to refuse a checkpoint that claims a *different*
# geometry, because such a file would load cleanly and dequantize to garbage.
# Single source of truth is layout.py (component A); we import it when the
# package is complete and cross-check, and fall back to literals so that this
# module stays importable and unit-testable on its own.
# --------------------------------------------------------------------------
PXQ4_TYPE_ID = 252
PXQ4_PANEL_ROWS = 64
PXQ4_SLAB_COLS = 32
PXQ4_SLAB_BYTES = 1088
PXQ4_HEADER_BYTES = 128
PXQ4_BOOK_LEN = 16
PXQ4_SUB_LEN = 16

try:  # pragma: no cover - exercised only inside the assembled package
    from . import layout as _layout
except Exception:  # noqa: BLE001 - standalone use, tests, partial checkouts
    _layout = None
else:
    # Drift detector.  If component A ever changes a constant, fail at import
    # rather than at token 4000 of a generation.
    assert _layout.TYPE_ID == PXQ4_TYPE_ID
    assert _layout.PANEL_ROWS == PXQ4_PANEL_ROWS
    assert _layout.SLAB_COLS == PXQ4_SLAB_COLS
    assert _layout.SLAB_BYTES == PXQ4_SLAB_BYTES
    assert _layout.HEADER_BYTES == PXQ4_HEADER_BYTES

# Entries that MUST stay in the ignore list.  Dropping either one flips
# ``_uses_split_gdn_input_projections`` (qwen3_5.py:127-157) to False, which
# folds the 48-row b/a projections into the fused, quantized ``in_proj_qkvz``.
# At TP=4 that is 12 rows per rank -- not a multiple of the 64-row panel -- and
# ``_adjust_shard_indexes_for_packing`` (parameter.py:605-610) does
# ``round(shard_size // packed_factor)``, i.e. it TRUNCATES TO ZERO WITHOUT
# RAISING.  The model would load and produce subtly wrong logits.  This is the
# single most dangerous misconfiguration available to this backend, so it is a
# hard error at config-parse time rather than a warning.
REQUIRED_IGNORE_ENTRIES = (
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
)

# Recommended contents of ``modules_to_not_convert``.  qwen3_5_mtp.py:465-505
# keys on exactly the strings "mtp" / "model.mtp" in this list (and *only* this
# list -- it does not look at ``ignore``) to build the speculative draft model
# with ``quant_config = None``.  Without it, a run started with
# ``--speculative-config`` would construct PXQ4 linears for mtp.* and then fail
# to find the weights, because the P1/P2 converter does not emit blk.64.
DEFAULT_MODULES_TO_NOT_CONVERT = ("mtp",)

# The P1 allow-list from plan 09 sec.3.  Kept here as documentation and as the
# fallback when a checkpoint omits ``pxq4_modules``; the checkpoint is
# authoritative when it supplies one.
P1_PXQ4_MODULES = (
    "mlp.gate_up_proj",
    "mlp.down_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkvz",
)

# Leaf module names this backend CANNOT serve as PXQ4, whatever a checkpoint
# asks for.  THIS LIST IS THE OWNER OF THE P2b DECISION: the LM head is fp16 in
# this build, and the converter (tools/pxq4_gguf/namemap.py POLICY_MODULES) must
# agree -- it must leave `lm_head` OUT of `pxq4_modules` and IN `ignore` for
# every policy.  ``test_namemap_policies_are_servable`` pins the two together.
#
# Why it cannot be served, verified in the fork this session -- TWO independent
# blocks, either of which is fatal on its own:
#
#   1. ``VocabParallelEmbedding.__init__`` calls create_weights with
#      ``weight_loader=self.weight_loader`` (vocab_parallel_embedding.py:520-527),
#      forcing its own bespoke v1 loader (:633-682).  That loader asserts
#      ``loaded_weight.shape[output_dim] == org_vocab_size // param.packed_factor``,
#      narrows by vocab shard indices and then does ``param[:n].copy_()`` /
#      ``param[n:].fill_(0)``.  PXQ4LinearMethod.create_weights explicitly
#      REFUSES a non-v2 loader whenever tp_size > 1 (linear.py of this package,
#      the ``wl_name != "weight_loader_v2"`` guard), because the v1 loader
#      cannot slice the 64-row panel layout.  At TP=4 that is a hard error.
#   2. One checkpoint tensor -> one ``weight_loader`` call, but a PXQ4 module
#      owns TWO parameters (``pxq4_slabs`` [P, S, 1088] and ``pxq4_anchor``
#      [P, 64]) with different first-dim semantics.  The vocab loader has no
#      way to express that; only the v2 parameter protocol does.
#
# ``embedding()`` (base_config.py:44, method_has_implemented_embedding :58-67)
# is NOT the missing piece for the LM head -- the fork only requires it when
# ``type(self) is VocabParallelEmbedding`` (vocab_parallel_embedding.py:488-497),
# and ParallelLMHead is a subclass.  Serving a 4-bit head needs a dedicated
# PXQ4 embedding method that reimplements the vocab-sharded loader over panels.
# That is plan-09 P3 work, not a decorator tweak.  ``embed_tokens`` is listed
# too: it is a real VocabParallelEmbedding, so it needs all of the above AND
# ``embedding()``, and the backbone table pins it to q6_k anyway.
UNSERVABLE_PXQ4_LEAF_MODULES = ("lm_head", "embed_tokens")


def _unservable_entries(modules: "list[str] | tuple[str, ...]") -> list[str]:
    """Entries of ``pxq4_modules`` naming a vocab-parallel module.

    Matches on the final dotted component so that both ``lm_head`` and
    ``model.language_model.lm_head`` are caught, while a hypothetical
    ``foo.lm_head_proj`` (a LinearBase) is not.
    """
    return [m for m in modules if m.rsplit(".", 1)[-1] in UNSERVABLE_PXQ4_LEAF_MODULES]


def _matches(prefix: str, pats: "list[str] | tuple[str, ...]") -> bool:
    """Plan 09 sec.6.5, verbatim.

    The three clauses are redundant (``p in prefix`` subsumes the other two);
    they are kept because the plan froze this predicate and components B and D
    test against it.

    The substring clause is the sharp edge.  It is safe for this model only
    because the vision tower's linears are named ``linear_fc1`` / ``linear_fc2``
    / ``attn.qkv`` / ``attn.proj`` (qwen3_vl.py:395-410, :496-510), none of
    which contains any allow-list pattern.  Verify this again before reusing
    the config for a different architecture.
    """
    return any(p == prefix or prefix.endswith("." + p) or p in prefix for p in pats)


def _as_str_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"pxq4: '{field}' must be a string or a sequence of strings") from exc


def _as_float_table(value: Any, field: str, length: int) -> tuple[float, ...]:
    if value is None:
        raise ValueError(
            f"pxq4: '{field}' is missing from quantization_config. The converter "
            f"must copy it verbatim from the gguf KV 'pxa.pxq6.{field}'; "
            f"PXA_PXQ6_BOOK / PXA_PXQ6_SUB can be overridden at engine build "
            f"time, so it is not safe to assume the compiled-in default."
        )
    table = tuple(float(v) for v in value)
    if len(table) != length:
        raise ValueError(f"pxq4: '{field}' must have exactly {length} entries, got {len(table)}")
    return table


@register_quantization_config(QUANT_METHOD_NAME)
class PXQ4Config(QuantizationConfig):
    """Serving config for a PXQ4 (ggml type id 252) checkpoint.

    A PXQ4 *file* is never uniformly PXQ4 -- ``pxa.pxq.backbone_rev`` 2 demotes
    attn_k/attn_v to q8_0, token_embd to q6_k, output to q8_0 and leaves
    ssm_out at MXFP4.  The offline converter dequantizes every non-PXQ4 class
    to fp16, so at *serving* time the split is binary: a module is either
    listed in ``pxq4_modules`` (two uint8/fp16 parameters, PXQ4LinearMethod) or
    it is an ordinary fp16 ``.weight`` (UnquantizedLinearMethod).

    Plan 09 sec.3.1 invariant: a module in ``pxq4_modules`` is *uniformly* PXQ4
    across all of its ``output_partition_sizes``.  There is no mixed-precision
    fused module.  ``validate_fused_uniformity()`` below enforces the config
    half of that; ``create_weights`` enforces the shape half.
    """

    def __init__(
        self,
        *,
        pxq4_modules: list[str],
        ignore: list[str],
        book: list[float],
        sub: list[float],
        pxq4_version: int = 1,
        tier: str = "core",
        backbone_rev: int | None = None,
        backbone_map: str | None = None,
        modules_to_not_convert: list[str] | None = None,
        raw_config: dict[str, Any] | None = None,
    ) -> None:
        # base_config.py:72-76 -- creates ``packed_modules_mapping``, which
        # model_loader/utils.py:290 later rebinds to the model class's mapping.
        super().__init__()

        self.pxq4_modules: list[str] = list(pxq4_modules)
        self.ignore: list[str] = list(ignore)
        self.book: tuple[float, ...] = tuple(book)
        self.sub: tuple[float, ...] = tuple(sub)
        self.pxq4_version = int(pxq4_version)
        self.tier = str(tier)
        self.backbone_rev = backbone_rev
        self.backbone_map = backbone_map
        self.modules_to_not_convert: list[str] = list(
            DEFAULT_MODULES_TO_NOT_CONVERT if modules_to_not_convert is None else modules_to_not_convert
        )

        # qwen3_5.py:127-157 probes four different surfaces for the ignore
        # list.  Populate all of them with the same content so the probe cannot
        # miss it regardless of which one a future fork revision reads.
        #   - self.ignore                 (list, above)
        #   - self.modules_to_not_convert (list, above)
        #   - self.ignored_layers         (list, alias)
        #   - self.config["ignore"]       (dict)
        self.ignored_layers: list[str] = self.ignore
        self.config: dict[str, Any] = dict(raw_config or {})
        self.config.setdefault("ignore", list(self.ignore))
        self.config.setdefault("modules_to_not_convert", list(self.modules_to_not_convert))

        # Union used for dispatch.  ``modules_to_not_convert`` carries "mtp",
        # which must also route to fp16 if an MTP module is ever constructed in
        # a process that shares this config object.
        self._ignore_for_dispatch: tuple[str, ...] = tuple(
            dict.fromkeys([*self.ignore, *self.modules_to_not_convert])
        )

        self._log_dispatch = os.getenv("PXQ4_LOG_DISPATCH", "0") == "1"
        self._fused_checked = False

        self._validate()

    # ---------------------------------------------------------------- checks

    def _validate(self) -> None:
        if self.pxq4_version != 1:
            raise ValueError(
                f"pxq4: unsupported pxq4_version={self.pxq4_version}. This build "
                f"implements version 1 (ggml type id {PXQ4_TYPE_ID}, 64-row "
                f"panel / 32-column slab)."
            )
        if not self.pxq4_modules:
            raise ValueError(
                "pxq4: 'pxq4_modules' is empty. A checkpoint with no PXQ4-served "
                "module should not declare quant_method='pxq4' at all."
            )

        # Fail at engine start, not two thousand modules into model
        # construction.  A P2b/P2c checkpoint whose converter listed `lm_head`
        # in pxq4_modules is unloadable, and the only useful moment to say so
        # is before any weight is touched.
        unservable = _unservable_entries(self.pxq4_modules)
        if unservable:
            raise ValueError(
                f"pxq4: {unservable} cannot be served by this build and must not "
                "appear in quantization_config['pxq4_modules'].\n"
                "This is the plan-09 P2b LM-head lever, and it is NOT implemented: "
                "ParallelLMHead/VocabParallelEmbedding force their own v1 vocab "
                "weight_loader (vocab_parallel_embedding.py:520-527, :633-682), "
                "which cannot slice a 64-row panel layout and cannot fill the two "
                "parameters (pxq4_slabs, pxq4_anchor) a PXQ4 module owns. See "
                "UNSERVABLE_PXQ4_LEAF_MODULES.\n"
                "FIX THE CHECKPOINT, not this list: the converter must leave these "
                "names out of 'pxq4_modules' and put them in 'ignore', and must "
                "emit the tensor as dense fp16 "
                "(tools/pxq4_gguf/namemap.py POLICY_MODULES). A config that lists "
                "lm_head describes a 4-bit head the engine will never read, so the "
                "weights would be missing, not merely slow."
            )

        missing = [e for e in REQUIRED_IGNORE_ENTRIES if e not in self.ignore]
        if missing:
            raise ValueError(
                "pxq4: quantization_config['ignore'] must contain "
                f"{list(REQUIRED_IGNORE_ENTRIES)} (missing: {missing}).\n"
                "Without them, vllm/model_executor/models/qwen3_5.py:127-157 "
                "folds the two 48-row GDN b/a projections into the quantized "
                "fused in_proj_qkvz. At TP=4 that shard is 12 rows -- not a "
                "multiple of the 64-row PXQ4 panel -- and parameter.py:605-610 "
                "truncates it silently instead of raising. The model would load "
                "and generate subtly wrong tokens."
            )

        # Cross-check the checkpoint's codebook against the one compiled into
        # the CUDA extension.  The kernels do NOT read book/sub from the file
        # (pxq6.cuh:79 uploads __device__ pxq6_book_g / pxq6_sub16_g once at
        # library init), so a checkpoint quantized with a different
        # PXA_PXQ6_BOOK would dequantize wrongly with no other symptom.
        self._check_tables_against_reference()

    def _check_tables_against_reference(self) -> None:
        try:
            from . import reference as _reference  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - reference.py belongs to component A
            logger.debug(
                "pxq4: reference tables unavailable; skipping book/sub cross-check"
            )
            return

        for field, got, want in (
            ("book", self.book, tuple(float(v) for v in _reference.BOOK)),
            ("sub", self.sub, tuple(float(v) for v in _reference.SUB)),
        ):
            if got != want:
                raise ValueError(
                    f"pxq4: checkpoint '{field}' table does not match the table "
                    f"compiled into this backend.\n  checkpoint: {list(got)}\n"
                    f"  backend:    {list(want)}\n"
                    "The CUDA kernels use the compiled-in table, so serving this "
                    "checkpoint would silently produce wrong values. Rebuild the "
                    "extension with the matching PXA_PXQ6_BOOK/PXA_PXQ6_SUB, or "
                    "requantize."
                )

    def validate_fused_uniformity(self) -> None:
        """Enforce the config half of the plan 09 sec.3.1 invariant.

        ``packed_modules_mapping`` is bound by reference from the model class
        (model_loader/utils.py:291), so it is only populated after the model
        class is resolved -- hence the lazy call from ``get_quant_method``
        rather than from ``__init__``.

        A fused vLLM module (``gate_up_proj``, ``in_proj_qkvz``, ``qkv_proj``)
        is loaded shard by shard from separate checkpoint tensors.  If one shard
        were PXQ4 and another fp16, ``PXQ4LinearMethod.create_weights`` would
        allocate one uniform slab tensor and the loader would write fp16 bytes
        into part of it -- a well-formed, wrong model.  Catch the config that
        expresses that intent.
        """
        if self._fused_checked:
            return
        self._fused_checked = True

        mapping = getattr(self, "packed_modules_mapping", None) or {}
        for fused_name, shard_names in mapping.items():
            fused_selected = _matches(fused_name, self.pxq4_modules) or any(
                p.endswith("." + fused_name) or p == fused_name for p in self.pxq4_modules
            )
            if not fused_selected:
                continue
            bad = [s for s in shard_names if _matches(s, self._ignore_for_dispatch)]
            if bad:
                raise ValueError(
                    f"pxq4: module '{fused_name}' is in pxq4_modules but its "
                    f"shard(s) {bad} are in the ignore list. A fused module must "
                    "be uniformly PXQ4 (plan 09 sec.3.1) -- there is no "
                    "per-shard dispatch and the loader cannot express one."
                )

    # -------------------------------------------------------- ABC: identity

    def get_name(self) -> QuantizationMethods:
        return QUANT_METHOD_NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # fp16 only, and deliberately not bf16: sm_70 has no bf16 arithmetic,
        # and the vendored kernels are fp16/fp32 throughout (pxq6.cuh:920-923).
        # config/vllm.py:622-627 turns this into a clear startup error instead
        # of a mysterious dtype mismatch deep in create_weights.
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Volta.  Unconditional -- unlike awq.py:177-184 / compressed_tensors_
        # wNa16.py:80-88 in this fork, which return 70 only when the turbomind
        # knob is on, our kernels are sm_70-native by construction.
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Everything lives in config.json["quantization_config"], which
        # weight_utils.py:302-318 hands straight to from_config().
        return []

    # ------------------------------------------------------- ABC: construction

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PXQ4Config":
        quant_method = config.get("quant_method")
        if quant_method not in (None, QUANT_METHOD_NAME):
            raise ValueError(
                f"pxq4: PXQ4Config.from_config called on a "
                f"quant_method={quant_method!r} checkpoint"
            )

        # Refuse a file whose declared geometry is not the one this backend
        # implements.  These keys are optional (older converter output may omit
        # them); when present they must agree exactly.
        for key, expected in (
            ("type_id", PXQ4_TYPE_ID),
            ("panel_rows", PXQ4_PANEL_ROWS),
            ("slab_cols", PXQ4_SLAB_COLS),
            ("slab_bytes", PXQ4_SLAB_BYTES),
            ("header_bytes", PXQ4_HEADER_BYTES),
        ):
            if key in config and int(config[key]) != expected:
                raise ValueError(
                    f"pxq4: checkpoint declares {key}={config[key]}, this backend "
                    f"implements {key}={expected}. Refusing to load: the panel/slab "
                    "addressing in pxq6.cuh:317-346 and :520-526 is compiled in."
                )

        pxq4_modules = _as_str_list(config.get("pxq4_modules"), "pxq4_modules")
        if not pxq4_modules:
            logger.warning(
                "pxq4: checkpoint does not declare 'pxq4_modules'; falling back "
                "to the plan-09 P1 allow-list %s. This is a guess -- prefer an "
                "explicit list in config.json.",
                list(P1_PXQ4_MODULES),
            )
            pxq4_modules = list(P1_PXQ4_MODULES)

        return cls(
            pxq4_modules=pxq4_modules,
            ignore=_as_str_list(config.get("ignore"), "ignore"),
            book=list(_as_float_table(config.get("book"), "book", PXQ4_BOOK_LEN)),
            sub=list(_as_float_table(config.get("sub"), "sub", PXQ4_SUB_LEN)),
            pxq4_version=int(config.get("pxq4_version", 1)),
            tier=str(config.get("tier", "core")),
            backbone_rev=config.get("backbone_rev"),
            backbone_map=config.get("backbone_map"),
            modules_to_not_convert=(
                _as_str_list(config["modules_to_not_convert"], "modules_to_not_convert")
                if "modules_to_not_convert" in config
                else None
            ),
            raw_config=config,
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        """Let the checkpoint self-select without a ``--quantization`` flag.

        Strictly speaking this is belt-and-braces: config/model.py:1011 already
        reads ``quant_cfg["quant_method"]`` and, finding no override, assigns it
        to ``self.quantization`` at :1071-1072, which then passes the
        ``supported_quantization`` membership test at :1082 because
        ``register_quantization_config`` appended "pxq4" to
        ``QUANTIZATION_METHODS``.  Implementing it also makes an explicit
        ``--quantization pxq4`` agree instead of raising at :1073-1079.

        This classmethod is called with *every other* checkpoint's config too
        (the probe loop at :1045-1067), so it must return None for anything
        that is not ours.
        """
        del user_quant, hf_config
        if not isinstance(hf_quant_cfg, dict):
            return None
        if hf_quant_cfg.get("quant_method") != QUANT_METHOD_NAME:
            return None
        return QUANT_METHOD_NAME

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any = None,
        revision: str | None = None,
    ) -> None:
        """Warn about the one config omission that fails late instead of early.

        base_config.py:181-198 hook, called from config/vllm.py:630-633 right
        after construction, i.e. before any module is built.
        """
        del model_name, revision
        if hf_config is None:
            return
        text_cfg = getattr(hf_config, "text_config", hf_config)
        n_mtp = getattr(text_cfg, "mtp_num_hidden_layers", 0) or 0
        has_mtp_optout = any(
            m in ("mtp", "model.mtp") for m in self.modules_to_not_convert
        )
        if n_mtp and not has_mtp_optout:
            logger.warning(
                "pxq4: model declares mtp_num_hidden_layers=%d but "
                "quantization_config['modules_to_not_convert'] does not contain "
                "'mtp'. qwen3_5_mtp.py:465-505 uses exactly that key to build "
                "the speculative draft branch unquantized. Running with "
                "--speculative-config will construct PXQ4 linears for mtp.* and "
                "fail to find weights, because the converter does not emit "
                "blk.64 in P1/P2.",
                n_mtp,
            )

    # --------------------------------------------------------- ABC: dispatch

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Imported here, not at module scope. This module is imported by the
        # vllm.general_plugins entry point, which runs at arg_utils.py:749 --
        # before ModelConfig exists. quantization/__init__.py:108 comments that
        # it defers these imports "to avoid triggering torch.compile too early";
        # we follow the same discipline.
        from vllm.model_executor.layers.linear import (  # noqa: PLC0415
            LinearBase,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            self.validate_fused_uniformity()

            # Ignore is checked FIRST (plan 09 sec.6.5). The two lists overlap
            # by construction in P2 policies, where a module moves from one to
            # the other; ignore-wins makes a half-edited config fall back to
            # fp16 rather than to a missing-weight crash.
            if _matches(prefix, self._ignore_for_dispatch):
                return self._dispatch(prefix, "fp16 (ignored)", UnquantizedLinearMethod())

            if _matches(prefix, self.pxq4_modules):
                from .linear import PXQ4LinearMethod  # noqa: PLC0415

                return self._dispatch(prefix, "pxq4", PXQ4LinearMethod(self))

            # Never None for a LinearBase: linear.py:492-495 raises on falsy.
            # This is the branch that carries the 333 fp16 vision-tower linears
            # and every fp16 tensor class of the mixed-type backbone.
            return self._dispatch(prefix, "fp16 (default)", UnquantizedLinearMethod())

        # ParallelLMHead / VocabParallelEmbedding: None -> UnquantizedEmbeddingMethod
        # (vocab_parallel_embedding.py:479-483).  Backstop only: _validate()
        # already rejected an lm_head/embed_tokens entry in pxq4_modules at
        # from_config time, so this branch is unreachable through the normal
        # path.  It stays for the two ways round that gate -- a directly
        # constructed PXQ4Config, or pxq4_modules mutated after construction --
        # because silently returning None here would serve fp16 weights that a
        # P2b checkpoint does not contain.
        if _matches(prefix, self.pxq4_modules) and type(layer).__name__ in (
            "ParallelLMHead",
            "VocabParallelEmbedding",
        ):
            raise NotImplementedError(
                f"pxq4: '{prefix}' is listed in pxq4_modules but {type(layer).__name__} "
                "is not a LinearBase and no PXQ4 embedding method exists. "
                "Serving a 4-bit lm_head or embedding is plan-09 phase P2b/P3 and "
                "is not implemented in this build. This should have been caught by "
                "PXQ4Config._validate() -- see UNSERVABLE_PXQ4_LEAF_MODULES."
            )

        # Attention (attention.py:159, mla_attention.py:928) and FusedMoE
        # (fused_moe/layer.py:349-364) both accept None. This model has no MoE
        # (866 tensors, zero *_exps) and we do not quantize the KV cache.
        return None

    def _dispatch(self, prefix: str, kind: str, method: QuantizeMethodBase) -> QuantizeMethodBase:
        if self._log_dispatch:
            logger.info("pxq4 dispatch: %-64s -> %s", prefix, kind)
        return method

    # ------------------------------------------------------------------ misc

    def get_cache_scale(self, name: str) -> str | None:
        # No KV-cache quantization in this backend.
        return None

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        # base_config.py:200-214 uses this to pre-round hidden_size for MXFP4
        # MoE. PXQ4 is not MXFP4 and this model has no MoE.
        del prefix, layer
        return False

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"PXQ4Config(version={self.pxq4_version}, tier={self.tier!r}, "
            f"backbone_rev={self.backbone_rev}, "
            f"pxq4_modules={self.pxq4_modules}, "
            f"ignore={self.ignore}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )
