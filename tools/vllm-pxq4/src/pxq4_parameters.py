# SPDX-License-Identifier: Apache-2.0
"""PXQ4 parameter classes -- the whole of the TP sharding contract.

DESTINATION IN THE REPO OF PLAN 09: ``src/pxq4_vllm/parameters.py``.

Component B ("runtime"), plan 09 sec.6.4.

------------------------------------------------------------------------------
WHY TWO PARAMETERS AND NOT ONE BLOB
------------------------------------------------------------------------------
A PXQ4 tensor on disk is one flat byte blob of

    (N/64) panels, each = 128 B fp16 anchor header + (K/32) slabs of 1088 B

(ggml-pxq6-tables.h:21-27 BM=64 / SLAB_BYTES=1088 / HDR_BYTES=128; addressing
pxq6.cuh:520-526).  A *weight row* is therefore NOT a contiguous byte range:
row r of panel p has its 4-bit codes scattered one 16-byte run per slab, at
slab_base + 64 + 16*r.  Every generic sharder in vLLM (and in the vendored
``gguf`` package) slices rows assuming row-contiguity, which is exactly why the
GGUF loader path is unusable for this format and why we convert offline.

vLLM's v2 weight loaders do exactly one thing to a parameter:
``tensor.narrow(dim, offset, size)`` along ONE declared dim, with the offset
optionally divided by ``packed_factor`` (parameter.py:145-230, :605-616).
That is a perfect match for this format *if* the two things that shard on
different axes are separate tensors:

    pxq4_slabs  uint8  [N/64, K/32, 1088]   dim0 = panels, dim1 = slabs
    pxq4_anchor fp16   [N/64, 64]           dim0 = panels

  * column-parallel (split output rows N): narrow(dim 0) on BOTH tensors with
    ``packed_factor=64`` so a logical row offset becomes a panel offset.  A
    panel is a self-contained contiguous byte range, so this is a pure
    whole-panel memcpy -- byte-identical to the unsharded file.
  * row-parallel (split K): narrow(dim 1) of the slab tensor only.  Slabs are
    per-32-columns and independent, and the fp16 row anchor has no cross-K
    coupling (eff = anchor[row] * sub[block], pxq6.cuh:326-331), so a K-split
    at a multiple of 32 is bit-identical numerics, not a re-quantization.
    The anchor must be DUPLICATED verbatim on every rank -- see below.

------------------------------------------------------------------------------
THE TWO PROPERTIES THAT MAKE THIS WORK, AND WHERE THEY WERE READ
------------------------------------------------------------------------------
1. ``RowvLLMParameter.load_row_parallel_weight`` (parameter.py:220-230) narrows
   ``input_dim`` using ``self.data.shape[input_dim]`` and NEVER consults
   ``packed_factor``.  With dim 1 already counted in slabs, the K split lands
   on slab boundaries for free -- no packing adjustment wanted, none applied.
2. A parameter that declares ``output_dim`` but NO ``input_dim`` falls through
   to ``BasevLLMParameter.load_row_parallel_weight`` -> ``_assert_and_load``
   (parameter.py:92-103) = full copy.  That is *precisely* the 128 B header
   duplication the K-split needs, and it costs zero lines of code.
   Hence ``PXQ4AnchorParameter`` derives from ``PackedColumnParameter``
   (column-only) and NOT from ``PackedvLLMParameter`` (column+row).
   Deriving it from the latter would make every rank narrow the anchor along
   a nonexistent K axis and blow up -- or worse, silently take a slice.

------------------------------------------------------------------------------
THE SILENT FAILURE MODE (read this before changing anything here)
------------------------------------------------------------------------------
``_adjust_shard_indexes_for_packing`` (parameter.py:605-616) is

    shard_size   = round(shard_size   // packed_factor)
    shard_offset = round(shard_offset // packed_factor)

Integer floor division.  A misaligned offset does NOT raise: it truncates,
yielding a well-formed slice of the wrong panels.  The model then loads
cleanly and produces subtly wrong logits.  Nothing downstream can detect it.
The ONLY defence is the hard %64 / %32 asserts in
``PXQ4LinearMethod.create_weights`` (linear.py of this package) plus gate G3.

------------------------------------------------------------------------------
NEITHER CLASS OVERRIDES ANY load_* METHOD -- BY DESIGN
------------------------------------------------------------------------------
Plan 09 sec.3.1 invariant: every vLLM linear module served by PXQ4 is
*uniformly* PXQ4 across all of its ``output_partition_sizes``.  Under that
invariant the stock loaders are sufficient.  If an implementer finds
themselves writing ``load_qkv_weight`` or ``load_merged_column_weight`` here,
the invariant has been broken -- stop and re-read sec.3.1.  (The concrete
consequence in P1: ``self_attn.qkv_proj`` stays fp16 because attn_k/attn_v are
q8_0 in the artifact; P2c re-encodes them instead of adding a per-shard
dispatch parameter class.)
"""

from __future__ import annotations

import torch

from vllm.model_executor.parameter import (
    PackedColumnParameter,
    PackedvLLMParameter,
)

# Layout constants.  Single source of truth is ``layout.py`` (component A);
# imported when the package is assembled, with literals as the standalone
# fallback so this module stays importable on its own.
try:  # pragma: no cover - exercised only inside the assembled package
    from . import layout as _layout

    PANEL_ROWS = _layout.PANEL_ROWS
    SLAB_COLS = _layout.SLAB_COLS
    SLAB_BYTES = _layout.SLAB_BYTES
    HEADER_BYTES = _layout.HEADER_BYTES
except Exception:  # noqa: BLE001 - standalone use / partial checkout / tests
    PANEL_ROWS = 64
    SLAB_COLS = 32
    SLAB_BYTES = 1088
    HEADER_BYTES = 128

assert HEADER_BYTES == PANEL_ROWS * 2, "one fp16 anchor per row of the panel"


class PXQ4SlabParameter(PackedvLLMParameter):
    """uint8 ``[N/64, K/32, 1088]`` -- the codes and the sub-scales.

    Constructed with ``output_dim=0, input_dim=1, packed_dim=0,
    packed_factor=64``.

    ``packed_dim == output_dim`` is what arms the packing adjustment in
    ``load_merged_column_weight`` (parameter.py:153-173),
    ``load_qkv_weight`` (:175-201) and
    ``MergedColumnParallelLinear._load_fused_module_from_checkpoint``
    (linear.py:1100-1138), turning logical row offsets into panel offsets on
    both the parameter side and the checkpoint side.

    ``input_dim=1`` is in SLAB units, deliberately: the row loader divides by
    nothing, so declaring the dim in slabs is what makes a K-split land on a
    slab boundary.  Do not "fix" this by declaring K in elements.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.data.dim() == 3, "pxq4_slabs must be [panels, kslabs, 1088]"
        assert self.data.shape[2] == SLAB_BYTES, (
            f"pxq4_slabs last dim must be {SLAB_BYTES}, got {self.data.shape[2]}"
        )
        assert self.data.dtype is torch.uint8, "pxq4_slabs must be uint8"
        assert self.output_dim == 0 and self.input_dim == 1
        assert self.packed_dim == 0 and self.packed_factor == PANEL_ROWS
        # marlin_tile_size must stay None: it would multiply the adjusted
        # offsets again (parameter.py:600-604) and silently re-scale our
        # already-correct panel indices.
        assert self.marlin_tile_size is None

    @property
    def pxq4_panels(self) -> int:
        return int(self.data.shape[0])

    @property
    def pxq4_kslabs(self) -> int:
        return int(self.data.shape[1])

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        p, s, _ = tuple(self.data.shape)
        return f"PXQ4SlabParameter(panels={p}, kslabs={s}, N={p * 64}, K={s * 32})"


class PXQ4AnchorParameter(PackedColumnParameter):
    """fp16 ``[N/64, 64]`` -- one fp16 row anchor per weight row, panel-major.

    Column-parallel only, ON PURPOSE (see the module docstring, point 2): with
    no ``input_dim`` a row-parallel layer full-copies it, which is the header
    duplication a K-split requires.  ``packed_factor=64`` keeps its panel
    offsets in lockstep with the slab parameter on merged/QKV column splits.

    Element [p, r] is the anchor of global weight row ``p*64 + r``; on disk it
    is the first 128 bytes of panel p, read as 64 little-endian fp16.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.data.dim() == 2, "pxq4_anchor must be [panels, 64]"
        assert self.data.shape[1] == PANEL_ROWS, (
            f"pxq4_anchor row must be {PANEL_ROWS}, got {self.data.shape[1]}"
        )
        assert self.data.dtype is torch.float16, "pxq4_anchor must be fp16"
        assert self.output_dim == 0
        assert self.packed_dim == 0 and self.packed_factor == PANEL_ROWS
        assert self.marlin_tile_size is None
        assert not hasattr(self, "_input_dim"), (
            "PXQ4AnchorParameter must NOT declare input_dim -- see parameters.py "
            "docstring point 2; declaring it turns the required full copy into "
            "a slice along an axis the anchor does not have."
        )

    @property
    def pxq4_panels(self) -> int:
        return int(self.data.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        p = int(self.data.shape[0])
        return f"PXQ4AnchorParameter(panels={p}, N={p * 64})"
