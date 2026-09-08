# SPDX-License-Identifier: Apache-2.0
"""Runtime tier table for the PXQ panel formats: pxq2 (254), pxq3 (255), pxq4 (252).

WHAT A "TIER" IS, and what it is not. All three formats share ONE layout -- 64-row panels,
128 B fp16 anchor header, 32-column slabs, a 64 B sub-scale SoA per slab, the shared SUB16
LUT and the parity-locked contract ``eff = fp32(anchor) * SUB16[nibble]; w = eff * book[code]``.
They differ in exactly two things: how many bits a code is (2 / 3 / 4, hence 8 / 12 / 16 code
bytes per row per slab, hence 576 / 832 / 1088 slab bytes) and how many entries the book has
(4 / 8 / 16). Everything else -- sharding rules, parameter shapes, loader semantics, the
capture-safety argument -- is identical, which is why this is a table and not three code paths.

WHY THE ON-DISK KEYS STILL SAY ``pxq4``. ``<module>.pxq4_slabs`` / ``.pxq4_anchor`` is a
WIRE-FORMAT name, like the ``pxq4`` torch namespace and the ``pxq4`` quant_method. The tier
travels in ``config.json`` (``quantization_config.pxq_tiers``), never in a key, so:
  * every PXQ4 checkpoint already in the field keeps loading with no config change;
  * vLLM's FusedMoE weight_name matching and the two vLLM parameter classes are untouched;
  * a mixed-tier model is expressible, because the tier is a property of the MODULE.
Renaming the keys would break every existing checkpoint for zero functional gain.

THE ONE THING THAT IS CHECKED AND NEVER INFERRED is the slab stride. A pxq2 tensor read with
the pxq4 stride is a well-formed array of the wrong bytes: it loads, shards, passes every
shape assertion and generates fluent garbage. So ``slab_bytes`` is asserted against the
parameter's own last dimension at create_weights time, and the C++ ops assert it again.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Tier", "TIERS", "BY_NAME", "tier_of", "tier_by_name", "default_tier",
           "upload_tables", "selftest", "ops_available", "fused_available"]

PANEL_ROWS = 64
SLAB_COLS = 32
HEADER_BYTES = 128


@dataclass(frozen=True)
class Tier:
    type_id: int
    name: str
    slab_bytes: int
    book_n: int
    #: torch.ops.pxq4 op names for this tier. pxq4's are the FROZEN v12b ops and are not
    #: renamed; pxq2/pxq3 get their own so that ABI is not touched (pxq23_torch.cpp).
    op_dequant: str
    op_mmv: str
    op_linear: str
    op_moe_mmv: str
    #: Fused MoE decode block (pxq_moe_fused.cuh). Optional: a v13 library built before the
    #: fused kernels landed has the tier's other ops and not these, and ``fused_available``
    #: below is what the MoE method asks rather than assuming from the library file name.
    op_moe_gateup: str = ""
    op_moe_down_fold: str = ""
    op_moe_down_part: str = ""

    @property
    def is_pxq4(self) -> bool:
        return self.type_id == 252


TIERS: dict[int, Tier] = {
    254: Tier(254, "pxq2", 576, 4,
              "pxq2_dequant_out", "pxq2_mmv_out", "pxq2_linear_out", "pxq2_moe_mmv_out",
              "pxq2_moe_gateup_glu_out", "pxq2_moe_down_fold_out", "pxq2_moe_down_part_out"),
    255: Tier(255, "pxq3", 832, 8,
              "pxq3_dequant_out", "pxq3_mmv_out", "pxq3_linear_out", "pxq3_moe_mmv_out",
              "pxq3_moe_gateup_glu_out", "pxq3_moe_down_fold_out", "pxq3_moe_down_part_out"),
    252: Tier(252, "pxq4", 1088, 16,
              "dequant_out", "mmv_out", "linear_out", "moe_mmv_out",
              "moe_gateup_glu_out", "moe_down_fold_out", "moe_down_part_out"),
}
BY_NAME: dict[str, Tier] = {t.name: t for t in TIERS.values()}


def tier_of(type_id: int) -> Tier:
    t = TIERS.get(int(type_id))
    if t is None:
        raise ValueError(f"pxq: ggml type id {type_id} is not a PXQ panel tier "
                         f"(known: {sorted(TIERS)})")
    return t


def tier_by_name(name: str) -> Tier:
    t = BY_NAME.get(str(name).lower())
    if t is None:
        raise ValueError(f"pxq: unknown tier {name!r} (known: {sorted(BY_NAME)})")
    return t


def default_tier() -> Tier:
    """What a checkpoint that declares no tiers means. Every checkpoint written before tiers
    existed is uniformly pxq4, so that is the only safe default -- and it is a DEFAULT, not a
    fallback: a checkpoint that declares a tier this build cannot serve raises."""
    return TIERS[252]


def ops_available(tier: Tier) -> bool:
    """Is this tier's op family present in the loaded library?

    A pxq4-only library (v12b and earlier) has no pxq2_* ops. Asking the library rather than
    the file name is what makes a wrong PXQ4_LIB a clear error at load time instead of an
    AttributeError inside the first forward.
    """
    if not hasattr(torch.ops, "pxq4"):
        return False
    return all(hasattr(torch.ops.pxq4, op) for op in
               (tier.op_dequant, tier.op_mmv, tier.op_linear, tier.op_moe_mmv))


def fused_available(tier: Tier) -> bool:
    """Does the loaded library carry the FUSED MoE decode block for this tier?

    Asked, never inferred, for the same reason ``ops_available`` is: a v13 library built
    before the fused kernels landed carries every other op of this tier, so a file-name check
    would pass and the first forward would die on an AttributeError inside a captured graph.
    ``moe_slot_fold_out`` is tier-independent but is checked here too, because form B needs it
    and a library with the per-tier ops but not the fold pass is a broken build, not a choice.
    """
    if not hasattr(torch.ops, "pxq4"):
        return False
    names = (tier.op_moe_gateup, tier.op_moe_down_fold, tier.op_moe_down_part)
    if not all(names):
        return False
    return (all(hasattr(torch.ops.pxq4, n) for n in names)
            and hasattr(torch.ops.pxq4, "moe_slot_fold_out"))


def op(tier: Tier, which: str):
    """Resolve one of this tier's ops, with a message that says what to fix."""
    name = getattr(tier, "op_" + which)
    if not hasattr(torch.ops, "pxq4") or not hasattr(torch.ops.pxq4, name):
        raise RuntimeError(
            f"pxq: the loaded kernel library has no torch.ops.pxq4.{name}, so tier "
            f"{tier.name} cannot be served. PXQ4_LIB must point at a v13 or later library "
            f"(libpxq_sm60_v13.so / libpxq_sm70_v13.so); the v12b and earlier libraries are "
            f"PXQ4-only.")
    return getattr(torch.ops.pxq4, name)


def upload_tables(tier: Tier, book, sub=None) -> None:
    """Push this tier's book (and, once, the shared SUB16 LUT) to the device.

    EAGER ONLY -- these are cudaMemcpyToSymbol calls and must run before any graph capture.

    WHY IT IS NOT OPTIONAL. The quantizer's tables can be overridden at build time
    (PXA_PXQ2_BOOK, PXA_PXQ2_V3, PXA_PXQ_CEIL_V2, PXA_PXQ6_SUB ...), and the GGUF records what
    was actually used in ``pxa.pxq2.book`` / ``pxa.pxq3.book`` / ``.sub``, which the converter
    copies into config.json. A checkpoint is only self-describing if we honour what it
    recorded; decoding a v3-book file with the v1 table is silent, uniform weight error.

    The SUB16 LUT is SHARED by all three tiers in the kernel TU (the engine shares PXQ6's LUT
    across every code width), so uploading it for one tier uploads it for all of them. That is
    correct as long as every tier in one checkpoint records the SAME sub -- which the converter
    asserts, because they come from one quantizer run.
    """
    if tier.is_pxq4:
        # Delegate: ops.upload_tables carries the sm_60 / tensor-core-TU hazard note and the
        # no-op-when-unchanged check, and there must be exactly one place that knows about it.
        if sub is not None:
            from . import ops as _ops
            _ops.upload_tables(book, sub)
        return
    if not hasattr(torch.ops.pxq4, "pxq_set_book"):
        raise RuntimeError("pxq: the loaded kernel library has no pxq_set_book; it predates "
                           "tier support (need libpxq_*_v13.so or later)")
    b = torch.as_tensor(list(book), dtype=torch.float32)
    if b.numel() != tier.book_n:
        raise ValueError(f"pxq: tier {tier.name} wants a {tier.book_n}-entry book, "
                         f"got {b.numel()}")
    torch.ops.pxq4.pxq_set_book(tier.type_id, b)
    if sub is not None:
        s = torch.as_tensor(list(sub), dtype=torch.float32)
        torch.ops.pxq4.pxq_set_sub(s)


def selftest(tier: Tier) -> int:
    """Run the library's built-in kernel self-test for this tier. 0 == pass.

    This decodes a deterministic synthetic panel set with a host oracle written from the
    format spec and requires the device kernels to agree BIT-EXACTLY, so it proves the
    kernels before a 10 GB model is loaded. Cheap (a few hundred KB, one launch).
    """
    if not hasattr(torch.ops.pxq4, "pxq_selftest"):
        raise RuntimeError("pxq: the loaded kernel library has no pxq_selftest "
                           "(need libpxq_*_v13.so or later)")
    return int(torch.ops.pxq4.pxq_selftest(tier.type_id))
