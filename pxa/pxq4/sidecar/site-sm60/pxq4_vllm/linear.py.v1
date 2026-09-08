# SPDX-License-Identifier: Apache-2.0
"""PXQ4LinearMethod -- weight declaration, TP sharding contract, and apply().

DESTINATION IN THE REPO OF PLAN 09: ``src/pxq4_vllm/linear.py``.

Component B ("runtime"), plan 09 sec.6.6.  This is where TP correctness lives.

==============================================================================
WHAT THIS CLASS IS RESPONSIBLE FOR
==============================================================================
1. Declaring two parameters whose shapes and ``output_dim``/``input_dim``/
   ``packed_dim``/``packed_factor`` attributes make vLLM's *stock* v2 weight
   loaders shard a 64-row panel-interleaved format correctly, in both
   directions, with no custom loader code.  See ``parameters.py`` for the
   layout argument; this file owns the *asserts* that make a misaligned split
   impossible.
2. Refusing, loudly and early, anything it cannot serve.
3. ``apply()``: a small-M mmv kernel and a large-M dequant+cuBLAS path, both
   allocation-free and CUDA-graph-capture safe.

==============================================================================
THE MIXED-TYPE CHECKPOINT, AND WHY THIS FILE HAS NO q8_0 / q6_k / MXFP4 PATH
==============================================================================
A "PXQ4" GGUF is not uniformly PXQ4.  Backbone rev 2 (docs/LEVERS.md; gguf KV
``pxa.pxq.backbone_map``) demotes by tensor class, and the artifact really
contains five types: PXQ4(252) 325 tensors, Q8_0 132, Q6_K 1, MXFP4(39) 48,
F32 360 -- attn_k/attn_v are q8_0, token_embd is q6_k, output(lm_head) is
q8_0, ssm_out is MXFP4, norms are f32.

The chosen design (plan sec.5.3) resolves that OFFLINE: the converter emits
PXQ4 modules as ``<module>.pxq4_slabs`` + ``<module>.pxq4_anchor`` and emits
EVERY other class as a plain fp16 ``<module>.weight``.  So at serving time the
split is binary and this file only ever sees the PXQ4 half:

    config.get_quant_method(layer, prefix)
        prefix in ignore        -> UnquantizedLinearMethod   (fp16 .weight)
        prefix in pxq4_modules  -> PXQ4LinearMethod          (this file)
        otherwise               -> UnquantizedLinearMethod   (fp16 .weight)

Adding a q8_0 or MXFP4 path *here* would be strictly worse than the offline
dequant: it would need its own parameter classes, its own sharding proof, and
-- for a fused module such as ``qkv_proj``, where attn_q is PXQ4 but attn_k/v
are q8_0 -- a per-shard dispatch inside one parameter, which is the single
most likely source of a silently mis-sharded, cleanly-loading, subtly-wrong
model (plan sec.3.1).  P2c dissolves that case by re-encoding k/v instead.

What this file DOES owe the mixed-type reality is a loud failure when the
config claims a module is PXQ4 and the checkpoint disagrees.  That is the
three-part written-ness proof below -- ``_SENTINEL`` (anchor),
``_SLAB_SENTINEL_BYTE`` (slabs) and ``_LoaderCallCounter`` (whole parameter).
BOTH parameters carry a value sentinel: an anchor-only sentinel is not a proof
that the module is usable, because the slab tensor is the other half of the
weight and a checkpoint can carry one without the other (a converter namemap
typo is enough -- see the docstring of ``_LoaderCallCounter``).

Note the deliberate asymmetry: we set ``_sm70_f16_forbidden`` on PXQ4 layers
only.  fp16 layers keep the fork's TurboMind sm70 fp16 GEMM fast path
(linear.py:56-96, armed by UnquantizedLinearMethod at :408), which is faster
than torch.mm on Volta.  We must never arm it for ourselves: it reads
``layer.weight``, which a PXQ4 layer does not have.

==============================================================================
VERIFIED FACTS THIS FILE DEPENDS ON (all read in /opt/1Cat-vLLM, git 2ceb15066)
==============================================================================
* ``LinearMethodBase.create_weights`` signature: linear.py:290-313;
  ``apply``: linear.py:316-325.
* Loader version is selected BY CLASS NAME:
  ``self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED``
  at ColumnParallelLinear (linear.py:697-709) and RowParallelLinear
  (linear.py:1696-1708).  The list is linear.py:193-209 and the fork exposes a
  public decorator ``register_weight_loader_v2_supported_method``
  (linear.py:212-215) -- used below, so we opt in with zero patches.
  ReplicatedLinear (linear.py:559-567) always passes the v1 loader; its v1
  body is a size assert plus a full copy (linear.py:600-604), which is
  correct for an unsharded layer, so that case is allowed explicitly.
* Column split: ``_ColumnvLLMParameter.load_column_parallel_weight``
  parameter.py:145-151 (narrow by ``self.data.shape[output_dim]``).
* Merged column split: ``load_merged_column_weight`` parameter.py:153-173,
  driven by ``MergedColumnParallelLinear.weight_loader_v2`` linear.py:1140-1205
  (``shard_offset = sum(output_sizes[:id]) // tp_size``) and, for tuple shard
  ids such as ``("in_proj_qkvz", "in_proj_qkv", (0, 1, 2))``
  (qwen3_5.py:487-493), by ``_load_fused_module_from_checkpoint``
  linear.py:1100-1138, which packing-adjusts the CHECKPOINT-side narrow too.
* QKV split: ``load_qkv_weight`` parameter.py:175-201.
* Row split: ``RowvLLMParameter.load_row_parallel_weight`` parameter.py:220-230
  -- narrows ``input_dim`` and never consults ``packed_factor``.
* Full copy for a param with no ``input_dim``: ``BasevLLMParameter.
  _assert_and_load`` parameter.py:92-103.
* THE SILENT TRUNCATION: ``_adjust_shard_indexes_for_packing``
  parameter.py:605-616 -- ``round(x // packed_factor)``, no raise.
* sm70 fp16 bypass ordering: ``_maybe_sm70_dense_forward`` linear.py:56-96,
  called BEFORE ``quant_method.apply`` at linear.py:612, :805, :1794.
  Gate order: ``_sm70_f16_forbidden`` (:61) then ``_sm70_f16_prepared`` (:63).
* ``set_weight_attrs`` asserts the attribute does not already exist
  (utils.py:28-29) -- hence the filtered copy below.
* ``process_weights_after_loading`` runs for every module after the whole model
  is loaded and before compile/capture (model_loader/utils.py:99-111), inside
  ``device_loading_context``, so params are on the target device there.
* THE LOADER DOES NOT BACKSTOP US.  An earlier revision of this file claimed
  "unloaded whole parameters are already caught by the loader
  (default_loader.py:432-437)".  That is FALSE for this deployment, twice over,
  and every check in this file must be read as the ONLY line of defence:
    - default_loader.py:403-412 arms the strict check only for UNQUANTIZED
      models: ``default_enable_weights_track = (model_config.quantization is
      None and loaded_weights is not None)``.  We run ``quantization="pxq4"``,
      so ``track_weights_loading`` is never called at all.
    - even forced on via ``enable_weights_track``, track_weights_loading
      (default_loader.py:421-431) does
      ``has_postprocess_quant = getattr(quant_method,
      "process_weights_after_loading", None)`` and, when truthy, ADDS every
      parameter of that module to ``loaded_weights`` -- i.e. exempts it.
      ``QuantizeMethodBase.process_weights_after_loading`` is defined on the
      base class (base_config.py:50-55), so the attribute is truthy for EVERY
      quant method, ours included.  Every linear layer is exempt by
      construction.
  Nor does the model backstop us: qwen3_5.py:564 (``if name not in
  params_dict: continue``, inside the stacked_params_mapping loop) drops a
  mis-named checkpoint tensor, and the fallthrough at qwen3_5.py:641-646 only
  ``logger.warning_once``.  Nothing raises.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

# NOTE: import the *module* for PXQ4_MMV_MAX_M, never the value. load_library()
# reconciles that constant with the .so's compiled-in default, and a
# ``from .ops import PXQ4_MMV_MAX_M`` would freeze the pre-load value of 8 in
# this module's namespace.
from . import ops as _ops
from .ops import PXQ4Workspace, load_library, mmv_supported, upload_tables
from .parameters import (
    PANEL_ROWS,
    SLAB_BYTES,
    SLAB_COLS,
    PXQ4AnchorParameter,
    PXQ4SlabParameter,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import PXQ4Config

logger = init_logger(__name__)

# --------------------------------------------------------------------------
# v2 weight-loader opt-in.
#
# WHY THIS IS NOT OPTIONAL: the v1 loader (linear.py:753-788, :924-1098,
# :1728-1761) reasons about 2-D [out, in] weights and calls
# ``adjust_marlin_shard`` / ``_adjust_shard_indexes_for_packing`` in a
# different order.  Our slab parameter is 3-D and its dim 1 is in slab units;
# only the v2 path (parameter.py:145-230) narrows exactly one declared dim and
# leaves ``packed_factor`` out of the row split, which is the whole reason the
# format shards for free.  If neither hook exists we refuse at import time
# rather than silently taking v1 and mis-sharding.
# --------------------------------------------------------------------------
# TIMING: the membership test happens inside ColumnParallelLinear.__init__
# (linear.py:706) / RowParallelLinear.__init__ (:1705), which run AFTER
# LinearBase.__init__ has already called ``quant_config.get_quant_method``
# (linear.py:492) -- and that call is what imports this module (config.py does
# ``from .linear import PXQ4LinearMethod`` lazily). So the registration is
# always in place before the first layer consults the list.
def _optin_weight_loader_v2(cls: type) -> type:
    try:
        from vllm.model_executor.layers.linear import (  # noqa: PLC0415
            register_weight_loader_v2_supported_method as _reg,
        )
    except ImportError:
        _reg = None
    if _reg is not None:
        return _reg(cls)
    try:
        from vllm.model_executor.layers.linear import (  # noqa: PLC0415
            WEIGHT_LOADER_V2_SUPPORTED,
        )
    except ImportError as exc:  # pragma: no cover - not reachable in this fork
        raise ImportError(
            "pxq4: this vLLM build exposes neither "
            "register_weight_loader_v2_supported_method nor "
            "WEIGHT_LOADER_V2_SUPPORTED; PXQ4 cannot opt into the v2 weight "
            "loader and the v1 loader would mis-shard the panel layout."
        ) from exc
    if cls.__name__ not in WEIGHT_LOADER_V2_SUPPORTED:
        WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
    return cls


def _require(cond: bool, msg: str) -> None:
    """Hard check that survives ``python -O``.

    Plain ``assert`` is stripped by -O.  Every check in ``create_weights``
    stands between us and a model that loads cleanly and produces subtly wrong
    logits (parameter.py:605-616 truncates instead of raising), so none of them
    may be optimisable away.
    """
    if not cond:
        raise ValueError(msg)


# An fp16 anchor in a real checkpoint is ``fp16(row absmax)``: finite and >= 0
# (pxq6-quantize.inc.cpp:263-284).  NaN is therefore unreachable from data,
# which makes it a deterministic "this parameter was never written" marker.
# Costs one fp16 fill of N/64*64 = N elements per layer at init and one
# ``isnan().any()`` per layer at load.
_SENTINEL = float("nan")
# PXQ4_SENTINEL=0 disables ONLY the NaN anchor fill, because NaN is the one
# sentinel with a risk attached: some loader path we have not read could scan
# the anchor for finiteness.  It does NOT disable the slab sentinel or the
# call counter -- turning the anchor tripwire off must not leave the layer
# with zero written-ness coverage.
_SENTINEL_ENABLED = os.getenv("PXQ4_SENTINEL", "1") != "0"

# The slab sentinel.  0xA5 is an inert byte in a uint8 tensor -- no NaN
# semantics, nothing can trip on it, no loader can be upset by it -- so unlike
# the anchor sentinel this one is UNCONDITIONAL.
#
# WHY A WHOLE SLAB CANNOT BE UNIFORMLY 0xA5 IN REAL DATA: a slab is 1088 B =
# 64 B of SoA sub-scales (one frozen SUB16 4-bit sub-scale per row, packed two
# per byte) + 1024 B of 4-bit codes (64 rows x 16 B).  Uniform 0xA5 means every
# row of the slab picked sub-scale index 10 for its low nibble and 5 for its
# high nibble AND every one of its 2048 codes alternates book entry 5 / 10.
# That is not reachable from an imatrix-calibrated quantizer on real weights;
# it is reachable from exactly one thing, which is nobody having written here.
# Cost: one uint8 memset of N/64*K/32*1088 B per layer at init (the tensor is
# allocated either way) and one equality pass per layer at load, pre-capture.
_SLAB_SENTINEL_BYTE = 0xA5

# Bound on the transient bool tensor the slab scan materialises, in bytes.
# lm_head at TP=4 is 593 panels x 160 slabs x 1088 B = 103 MB; chunking keeps
# the scan's peak footprint flat regardless of layer size.
_SLAB_SCAN_CHUNK_BYTES = 4 << 20


class _LoaderCallCounter:
    """Transparent wrapper around the ``weight_loader`` vLLM handed us.

    SOLE PURPOSE: prove that the checkpoint contained *a* tensor addressed to
    this parameter.  This is the whole-parameter check that the module
    docstring above shows the loader does NOT perform for a quantized model.

    THE FAILURE IT CATCHES: the converter emits ``<mod>.gate_proj.pxq4_slab``
    (typo, or a stale namemap) alongside a correct
    ``<mod>.gate_proj.pxq4_anchor``.  qwen3_5.py:564 drops the slab tensor
    silently; the anchor loads and overwrites every NaN, so the anchor
    tripwire is clean; ``track_weights_loading`` never runs.  Without this
    counter the server starts and every PXQ4 layer multiplies by whatever the
    caching allocator last left in that buffer.

    It counts the OUTER call only.  A merged/QKV module's per-shard recursion
    happens inside ``weight_loader_v2`` (linear.py:1100-1138), below this
    wrapper, so ``calls`` is "how many checkpoint tensors were routed here",
    not "how many shards were written".  Shard-level coverage is the value
    sentinels' job; this is the coarse "did anything arrive at all" gate.

    Kept deliberately dumb: no narrow bookkeeping, no reimplementation of the
    v2 loaders' offset arithmetic.  Re-deriving that logic here would put a
    second, drifting copy of parameter.py:145-230 in the blast radius of the
    very bug it is meant to catch.
    """

    def __init__(self, inner, prefix: str, param_name: str) -> None:
        self._inner = inner
        self.prefix = prefix
        self.param_name = param_name
        self.calls = 0
        # Keep the wrapper indistinguishable from the bound method for the
        # two things the fork introspects on a weight loader:
        # reload/layerwise.py:139 reads ``__name__``, :148 reads
        # ``inspect.signature``.  ``__signature__`` is honoured by inspect.
        self.__name__ = getattr(inner, "__name__", "weight_loader")
        self.__qualname__ = getattr(inner, "__qualname__", self.__name__)
        self.__doc__ = getattr(inner, "__doc__", None)
        try:  # pragma: no cover - introspection only
            import inspect as _inspect  # noqa: PLC0415

            self.__signature__ = _inspect.signature(inner)
        except (TypeError, ValueError):  # pragma: no cover
            pass

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._inner(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"_LoaderCallCounter({self.prefix}.{self.param_name}, "
            f"inner={self.__name__}, calls={self.calls})"
        )


class _AlreadyVerified:
    """Left on the layer in place of a _LoaderCallCounter once written-ness has
    been proven, so that a second ``process_weights_after_loading`` (layerwise
    reload) sees a satisfied check instead of a missing one, while the bound
    ``weight_loader`` -- and the layer reference cycle it carries -- is
    dropped at the same point the parameters are rebound."""

    calls = 1

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return "_AlreadyVerified()"


_VERIFIED = _AlreadyVerified()


def _count_sentinel_slabs(slabs: torch.Tensor) -> int:
    """How many whole 1088-byte slabs are still uniformly the sentinel byte.

    Chunked over panels so the transient bool tensor stays bounded by
    ``_SLAB_SCAN_CHUNK_BYTES``.  Runs once per layer inside
    ``process_weights_after_loading`` -- eager, pre-capture, so the
    device-to-host read it implies is safe here and only here.
    """
    panels, kslabs, slab_bytes = (int(v) for v in slabs.shape)
    per_panel = max(1, kslabs * slab_bytes)
    step = max(1, _SLAB_SCAN_CHUNK_BYTES // per_panel)
    survivors = 0
    for start in range(0, panels, step):
        chunk = slabs.narrow(0, start, min(step, panels - start))
        survivors += int(chunk.eq(_SLAB_SENTINEL_BYTE).all(dim=2).sum())
    return survivors


@_optin_weight_loader_v2
class PXQ4LinearMethod(LinearMethodBase):
    """Serve one uniformly-PXQ4 linear module.

    Plan sec.3.1 invariant: uniformly PXQ4 across ALL of
    ``output_partition_sizes``.  There is no mixed-precision fused module and
    this class must never grow one.
    """

    # Tables are per-device global state in the .so; upload them once.
    _tables_uploaded: bool = False

    def __init__(self, quant_config: "PXQ4Config") -> None:
        self.quant_config = quant_config
        # Loading the .so here rather than in apply(): apply() runs inside a
        # captured region, dlopen does not.
        load_library()

    # ------------------------------------------------------------------
    # create_weights
    # ------------------------------------------------------------------
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        prefix = getattr(layer, "prefix", None) or type(layer).__name__
        K = int(input_size_per_partition)
        N = int(sum(int(o) for o in output_partition_sizes))
        tp_size = int(getattr(layer, "tp_size", 1) or 1)

        # -- 1. dtype -------------------------------------------------------
        # sm_70 has no bf16 and no fp8; the vendored kernels are fp16-in /
        # fp16-out.  Refusing here is better than a wrong-dtype kernel launch:
        # the engine-level gate (config/vllm.py:600-628) already checks
        # get_supported_act_dtypes(), so reaching this branch means something
        # bypassed it.
        _require(
            params_dtype is torch.float16,
            f"pxq4 [{prefix}]: requires fp16 activations on sm_70, got {params_dtype}",
        )

        # -- 2. geometry, per rank -----------------------------------------
        # These four are the entire defence against parameter.py:605-616
        # truncating a misaligned shard into a well-formed wrong slice.
        _require(
            K % SLAB_COLS == 0,
            f"pxq4 [{prefix}]: K={K} is not a multiple of {SLAB_COLS}. A K-split "
            "must fall on a slab boundary; a partial slab cannot be addressed.",
        )
        _require(
            N % PANEL_ROWS == 0,
            f"pxq4 [{prefix}]: N={N} is not a multiple of {PANEL_ROWS}. A row-split "
            "must fall on a panel boundary; panels carry their own anchor header.",
        )
        for i, o in enumerate(output_partition_sizes):
            _require(
                int(o) % PANEL_ROWS == 0,
                f"pxq4 [{prefix}]: output_partition_sizes[{i}]={o} is not a multiple "
                f"of {PANEL_ROWS}. This shard offset would be floor-divided by "
                "packed_factor and SILENTLY truncated (parameter.py:605-616). "
                "Known trigger: the GDN in_proj_ba (48 rows) folded into "
                "in_proj_qkvz -- the quant config must list "
                "'linear_attn.in_proj_a'/'linear_attn.in_proj_b' in `ignore` so "
                "qwen3_5.py:127-157 splits them out.",
            )

        # -- 3. geometry, unsharded (the checkpoint side) -------------------
        # _load_fused_module_from_checkpoint (linear.py:1100-1138) narrows the
        # FULL checkpoint tensor by full-size offsets before the per-rank
        # narrow, so the unsharded sizes must be panel/slab aligned too.
        _require(
            int(output_size) % PANEL_ROWS == 0,
            f"pxq4 [{prefix}]: unsharded output_size={output_size} is not a multiple "
            f"of {PANEL_ROWS}; the checkpoint tensor is not a whole number of panels.",
        )
        _require(
            int(input_size) % SLAB_COLS == 0,
            f"pxq4 [{prefix}]: unsharded input_size={input_size} is not a multiple "
            f"of {SLAB_COLS}.",
        )
        _require(
            int(input_size) in (K, K * tp_size),
            f"pxq4 [{prefix}]: input_size={input_size} is neither K={K} "
            f"(column-parallel) nor K*tp={K * tp_size} (row-parallel); the layer's "
            "partitioning is not one this method understands.",
        )

        # -- 4. loader version ---------------------------------------------
        # If the layer handed us the v1 loader, our 3-D slab parameter would be
        # sliced by 2-D logic.  Detect it by the bound method's name: v2 call
        # sites pass ``self.weight_loader_v2`` (linear.py:704-708, :1703-1707).
        # ASSUMPTION: qwen3_5.load_weights' NON-stacked branch calls
        # ``weight_loader(param, loaded_weight)`` (two positional args), which
        # is what ColumnParallelLinear/RowParallelLinear.weight_loader_v2
        # expect. The stacked branch was read (qwen3_5.py:487-508 ->
        # linear.py:1140-1143, three args) and matches; the two-arg convention
        # is standard across vLLM models but was not read line by line here.
        weight_loader = extra_weight_attrs.get("weight_loader", None)
        _require(weight_loader is not None, f"pxq4 [{prefix}]: no weight_loader passed")
        wl_name = getattr(weight_loader, "__name__", "")
        if wl_name != "weight_loader_v2":
            # ReplicatedLinear (linear.py:559-567) has no v2 path at all, but it
            # also does no sharding: its v1 body is a size assert plus a full
            # copy (linear.py:600-604), which is correct for both our params.
            _require(
                tp_size == 1,
                f"pxq4 [{prefix}]: got weight loader '{wl_name}' (v1) on a layer with "
                f"tp_size={tp_size}. The v1 loader cannot shard the panel layout. "
                "PXQ4LinearMethod must be in WEIGHT_LOADER_V2_SUPPORTED "
                "(linear.py:193-215) -- check that this module was imported "
                "before the model was built.",
            )

        panels = N // PANEL_ROWS
        kslabs = K // SLAB_COLS

        # -- 5. the parameters ---------------------------------------------
        # No device= : vLLM constructs the model inside a default-device
        # context, exactly as UnquantizedLinearMethod does (linear.py:330-346).
        #
        # BOTH parameters get a value sentinel and a call counter. Neither the
        # loader nor the model raises on a missing tensor here (see the module
        # docstring), so an undefended parameter is an undefended parameter --
        # and the slab tensor is 99.98% of the layer's bytes.
        slab_loads = _LoaderCallCounter(weight_loader, prefix, "pxq4_slabs")
        anchor_loads = _LoaderCallCounter(weight_loader, prefix, "pxq4_anchor")

        slab_data = torch.empty(panels, kslabs, SLAB_BYTES, dtype=torch.uint8)
        # Unconditional: an inert byte pattern, checked whole-slab at load.
        slab_data.fill_(_SLAB_SENTINEL_BYTE)
        slabs = PXQ4SlabParameter(
            data=slab_data,
            output_dim=0,
            input_dim=1,       # in SLAB units -- see parameters.py
            packed_dim=0,
            packed_factor=PANEL_ROWS,
            weight_loader=slab_loads,
        )
        anchor_data = torch.empty(panels, PANEL_ROWS, dtype=torch.float16)
        if _SENTINEL_ENABLED:
            # ASSUMPTION: nothing between create_weights and
            # process_weights_after_loading READS anchor values -- no finiteness
            # scan, no dtype-probe that would trip on NaN, no constant folding.
            # Not verified against every loader path in the fork (cpu-offload,
            # sleep/wake, layerwise reload). If one is found, PXQ4_SENTINEL=0
            # disables this; the slab sentinel and both call counters stay on.
            anchor_data.fill_(_SENTINEL)
        anchor = PXQ4AnchorParameter(
            data=anchor_data,
            output_dim=0,
            # NO input_dim: on a RowParallelLinear this falls through to
            # BasevLLMParameter._assert_and_load = full copy, which is the
            # header duplication a K-split requires (parameters.py, point 2).
            packed_dim=0,
            packed_factor=PANEL_ROWS,
            weight_loader=anchor_loads,
        )

        layer.register_parameter("pxq4_slabs", slabs)
        layer.register_parameter("pxq4_anchor", anchor)
        # Read back in process_weights_after_loading. Held on the layer, not on
        # the parameter: process_weights_after_loading rebinds both parameters
        # to plain torch.nn.Parameter and would drop anything attached there.
        layer.pxq4_slab_loads = slab_loads
        layer.pxq4_anchor_loads = anchor_loads

        # Forward any other loader hints (is_sharded_weight, etc.).
        # set_weight_attrs asserts the attribute is absent (utils.py:28-29),
        # so filter what the parameter classes already own.
        extras = {
            k: v
            for k, v in extra_weight_attrs.items()
            if k != "weight_loader" and not hasattr(slabs, k)
        }
        if extras:
            set_weight_attrs(slabs, extras)
            set_weight_attrs(anchor, {k: v for k, v in extras.items()
                                      if not hasattr(anchor, k)})

        layer.pxq4_N = N
        layer.pxq4_K = K
        layer.pxq4_panels = panels
        layer.pxq4_kslabs = kslabs
        layer.pxq4_prefix = prefix
        # Decided in process_weights_after_loading, once the .so can answer.
        layer.pxq4_use_mmv = False
        layer.pxq4_mmv_max_m = 0

        # -- 6. never arm the fp16 fast path -------------------------------
        # _maybe_sm70_dense_forward (linear.py:56-96) runs BEFORE
        # quant_method.apply and would try to read layer.weight, which does not
        # exist here.  ``_sm70_f16_prepared`` is set only by
        # UnquantizedLinearMethod (linear.py:408) so it cannot become true for
        # us, but _mark_default_sm70_dense_modules (qwen3_5.py:167-177) sets
        # ``_sm70_f16_force_enable`` on every qkv_proj/out_proj by name -- set
        # the explicit forbid flag so no future revision of that gate can pick
        # us up.
        layer._sm70_f16_forbidden = True

        # -- 7. workspace budget -------------------------------------------
        # Sized by the largest single dequantized weight on this rank; the
        # arena is shared across layers and materialized once, before capture.
        PXQ4Workspace.reserve(dequant_elems=N * K, act_elems=0)

    # ------------------------------------------------------------------
    # process_weights_after_loading
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Validate, freeze, and size the runtime. No repacking.

        The on-disk layout IS the device layout -- the converter's split of the
        GGUF blob into [panels, kslabs, 1088] + [panels, 64] reorders no byte
        (plan sec.5.3) -- so unlike the TurboMind template there is nothing to
        repack here.  Everything below is a check or a one-time setup.
        """
        prefix = getattr(layer, "pxq4_prefix", "<unknown>")
        slabs: torch.Tensor = layer.pxq4_slabs
        anchor: torch.Tensor = layer.pxq4_anchor

        _require(
            slabs.dtype is torch.uint8 and anchor.dtype is torch.float16,
            f"pxq4 [{prefix}]: bad param dtypes {slabs.dtype}/{anchor.dtype}",
        )
        _require(
            tuple(slabs.shape) == (layer.pxq4_panels, layer.pxq4_kslabs, SLAB_BYTES),
            f"pxq4 [{prefix}]: slab shape {tuple(slabs.shape)} != "
            f"{(layer.pxq4_panels, layer.pxq4_kslabs, SLAB_BYTES)}",
        )
        _require(
            tuple(anchor.shape) == (layer.pxq4_panels, PANEL_ROWS),
            f"pxq4 [{prefix}]: anchor shape {tuple(anchor.shape)} != "
            f"{(layer.pxq4_panels, PANEL_ROWS)}",
        )
        # The kernel does a 16-byte aligned load at slab_base + 64 + 16*row.
        # Only a contiguous, naturally-strided tensor guarantees that.
        _require(
            slabs.is_contiguous() and anchor.is_contiguous(),
            f"pxq4 [{prefix}]: parameters must be contiguous after loading",
        )
        _require(
            slabs.device == anchor.device,
            f"pxq4 [{prefix}]: slabs on {slabs.device}, anchor on {anchor.device}",
        )

        # ==================================================================
        # WRITTEN-NESS.  NOTHING ELSE CHECKS THIS.
        # ==================================================================
        # There is no loader backstop for a quantized model (module docstring:
        # default_loader.py:403-412 never arms the check, and :421-431 exempts
        # every module that has process_weights_after_loading, which is all of
        # them, base_config.py:50-55) and no model backstop either
        # (qwen3_5.py:564 drops a mis-named tensor silently, :641-646 only
        # warns).  So this block is the entire defence, and it must cover BOTH
        # parameters -- the anchor is 0.02% of the layer's bytes.
        #
        # Three checks, coarse to fine:
        #   (a) counter == 0  -> no checkpoint tensor was ever routed here.
        #   (b) slab sentinel -> a (panel, kslab) range was never written.
        #   (c) anchor NaN    -> a panel's anchors were never written.
        # (a) is the one that catches a converter namemap typo on ONE of the
        # two parameters, which is the case where the other parameter's value
        # sentinel is cleanly overwritten and reports nothing wrong.

        # -- (a) whole-parameter: was the loader ever called? ---------------
        for pname, counter in (
            ("pxq4_slabs", getattr(layer, "pxq4_slab_loads", None)),
            ("pxq4_anchor", getattr(layer, "pxq4_anchor_loads", None)),
        ):
            _require(
                counter is not None,
                f"pxq4 [{prefix}]: no load counter for {pname}; "
                "process_weights_after_loading ran on a layer that "
                "create_weights did not build.",
            )
            _require(
                counter.calls > 0,
                f"pxq4 [{prefix}]: the weight loader was NEVER called for "
                f"{prefix}.{pname} -- the checkpoint carries no tensor by that "
                "name. Nothing upstream raises on this: for a quantized model "
                "default_loader.py:403-412 disables the unloaded-weight check "
                "outright, and qwen3_5.py:564 drops an unmatched name without "
                "logging. Check the converter's output names against "
                f"'<module>.pxq4_slabs' / '<module>.pxq4_anchor' (a stale "
                "namemap emitting e.g. '.pxq4_slab' produces exactly this).",
            )

        # -- (b) slab sentinel: any 1088-byte slab never written -------------
        sentinel_slabs = _count_sentinel_slabs(slabs)
        if sentinel_slabs:
            total_slabs = layer.pxq4_panels * layer.pxq4_kslabs
            raise ValueError(
                f"pxq4 [{prefix}]: {sentinel_slabs}/{total_slabs} slabs were "
                "never written by the weight loader (still uniformly "
                f"0x{_SLAB_SENTINEL_BYTE:02X}). The loader was called for "
                "pxq4_slabs but did not cover the whole tensor: a fused module "
                "where one logical shard came from a checkpoint tensor that is "
                "NOT PXQ4 (attn_k/attn_v q8_0 next to a PXQ4 attn_q, ssm_out "
                "MXFP4, ...), i.e. the plan sec.3.1 uniformity invariant "
                "violated by a bad `pxq4_modules` list."
            )

        # -- (c) anchor sentinel --------------------------------------------
        # A surviving NaN means some panel range of this module was never
        # written -- same fused-module cause as (b), caught on the other
        # parameter.
        if _SENTINEL_ENABLED and bool(torch.isnan(anchor).any()):
            bad = int(torch.isnan(anchor).any(dim=1).sum())
            raise ValueError(
                f"pxq4 [{prefix}]: {bad}/{layer.pxq4_panels} anchor panels were "
                "never written by the weight loader. The checkpoint does not "
                "carry PXQ4 tensors for all shards of this module. Either the "
                "converter emitted it as fp16 .weight, or `pxq4_modules` lists a "
                "module that is not uniformly PXQ4 (plan sec.3.1) -- e.g. "
                "self_attn.qkv_proj before P2c re-encodes attn_k/attn_v."
            )

        # Both parameters are proven written. Swap the counters for the
        # already-verified marker: drops the bound weight_loader (and the
        # layer->counter->layer cycle it carries) without turning a second
        # process_weights_after_loading into a spurious failure.
        layer.pxq4_slab_loads = _VERIFIED
        layer.pxq4_anchor_loads = _VERIFIED

        # ASSUMPTION: nothing downstream reads the vLLM parameter attributes
        # (output_dim / packed_factor / weight_loader) AFTER loading. True for
        # the loaders and for apply(); NOT verified for LoRA, the layerwise
        # reload path (model_loader/reload/layerwise.py) or sleep/wake, none of
        # which this deployment uses. Dropping the rebind is a safe fallback.
        # Rebind to plain Parameters.  WHY: BasevLLMParameter is a Tensor
        # subclass with a __torch_function__ override (parameter.py:122-127) and
        # holds a reference to the layer's weight_loader.  Neither is wanted
        # once loading is done, and a plain Parameter is what torch.compile
        # traces most predictably.  Same storage -- this copies nothing.
        # ASSUMPTION: torch.nn.Parameter accepts a uint8 tensor when
        # requires_grad=False (it rejects only autograd-tracking integer
        # params). Standard torch behaviour; not executed in this workflow
        # because no torch is installed on the workflow host.
        layer.pxq4_slabs = Parameter(slabs.data, requires_grad=False)
        layer.pxq4_anchor = Parameter(anchor.data, requires_grad=False)

        # DO NOT set layer._sm70_f16_prepared (linear.py:63). Ever.
        _require(
            not getattr(layer, "_sm70_f16_prepared", False),
            f"pxq4 [{prefix}]: _sm70_f16_prepared is set on a PXQ4 layer; the sm70 "
            "fp16 fast path would bypass apply() and read a nonexistent "
            "layer.weight (linear.py:56-96).",
        )
        layer._sm70_f16_forbidden = True

        # One-time device setup, eager, pre-capture.
        if not PXQ4LinearMethod._tables_uploaded:
            book = getattr(self.quant_config, "book", None)
            sub = getattr(self.quant_config, "sub", None)
            if book and sub:
                upload_tables(book, sub)
            PXQ4LinearMethod._tables_uploaded = True

        # mmv eligibility is a function of K only (shared-memory staging), so
        # ask the .so once per layer and cache a Python bool -- apply() must
        # never call into the op registry to make a control decision.
        layer.pxq4_use_mmv = mmv_supported(layer.pxq4_K)
        # Snapshot the ceiling as a plain Python int on the layer: apply() must
        # read a constant, not a module global that could be rebound between
        # graph capture and replay.
        layer.pxq4_mmv_max_m = _ops.mmv_max_m()
        if not layer.pxq4_use_mmv:
            logger.info(
                "pxq4 [%s]: K=%d has no mmv kernel; using dequant+GEMM for all M",
                prefix,
                layer.pxq4_K,
            )

        PXQ4Workspace.materialize(slabs.device)

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """out = x @ W^T (+ bias), W being the PXQ4 weight of this rank.

        CAPTURE SAFETY (plan risk 3): no allocation other than ``out`` (served
        from the graph-private pool), no host-side read of a device value, no
        ``.item()``, no stream sync, and every control decision is a Python
        constant fixed at load time.  The two ops mutate their first argument
        and are declared ``Tensor(a!)`` so functionalization keeps the ordering.
        """
        N = layer.pxq4_N
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]

        out = torch.empty((M, N), dtype=torch.float16, device=x2.device)
        if M == 0:
            # A zero-token batch happens on some warmup/dummy-run paths; both
            # kernels would launch an empty grid, which is legal but pointless.
            return out.reshape(*x.shape[:-1], N)

        if layer.pxq4_use_mmv and M <= layer.pxq4_mmv_max_m:
            # Decode path. One block per (panel, token): reads each weight byte
            # exactly once per token and never materializes the dequantized
            # weight, which is the entire bandwidth argument for this port.
            torch.ops.pxq4.mmv_out(out, x2, layer.pxq4_slabs, layer.pxq4_anchor)
        else:
            # Prefill / large-batch path. Coalesced dequant into fp16 followed
            # by cuBLAS HMMA beats the fused 4-bit tile on sm_70 -- measured at
            # -18.6% for the fused shape in our own engine
            # (ggml-cuda.cu:4436-4444) -- and it is also far less code.
            # ASSUMPTION: one arena shared by every PXQ4 layer is safe because
            # the model runs as one ordered op stream on one stream, and the
            # ``Tensor(a!)`` mutation annotation keeps inductor from hoisting a
            # read of `w` above the dequant that fills it. Neither half was
            # tested on a GPU in this workflow -- that is gate G8. Fallback:
            # PXQ4_DEQUANT_ALLOC=torch allocates per call instead.
            w = PXQ4Workspace.dequant_view(N, layer.pxq4_K, x2.device)
            torch.ops.pxq4.dequant_out(w, layer.pxq4_slabs, layer.pxq4_anchor)
            # ASSUMPTION: torch.mm(out=) does not allocate under cuda-graph
            # capture. cuBLAS workspaces are per-stream and preallocated by
            # torch, so this should hold; verify at G8 with a capture-time
            # allocator counter.
            torch.mm(x2, w.t(), out=out)

        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], N)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"PXQ4LinearMethod(mmv_max_m={_ops.mmv_max_m()})"
