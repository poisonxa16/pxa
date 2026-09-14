# SPDX-License-Identifier: Apache-2.0
"""Runtime tier table for the PXQ panel formats: pxq2 (254), pxq3 (255), pxq4 (252),
pxq4hq (253).

WHAT A "TIER" IS, and what it is not. All four formats share ONE layout -- 64-row panels,
128 B fp16 anchor header, 32-column slabs, a sub-scale SoA at the head of each slab, and the
parity-locked contract ``eff = fp32(anchor) * SUB[nibble]; w = eff * book[code]``.
They differ in how many bits a code is (2 / 3 / 4, hence 8 / 12 / 16 code bytes per row per
slab), how many entries the book has (4 / 8 / 16), and -- for pxq4hq alone -- how FINE the
sub-scale is. Everything else -- sharding rules, parameter shapes, loader semantics, the
capture-safety argument -- is identical, which is why this is a table and not four code paths.

PXQ4HQ IS THE ONE THAT BREAKS TWO OTHERWISE-UNIVERSAL ASSUMPTIONS, so they are stated here
rather than discovered later:
  * ITS SUB TABLE IS NOT THE SHARED ONE. pxq2/pxq3/pxq4/pxq6 all index the same 16-entry SUB16
    LUT, one sub index per 16 elements, and one upload moves all of them. pxq4hq spends one
    sub index per 8 elements against its own SUB8 fit (128 B of scale SoA per slab, not 64,
    hence 1152 slab bytes and 4.50 bpw). Uploading a pxq4 sub for it is silent, uniform weight
    error, so it carries ``own_sub`` and its own upload op.
  * IT HAS NO MoE OP FAMILY. It exists to serve the ATTENTION block of a promoted tier profile
    -- the tier a policy buys attention UP to -- so its modules are dense LINEAR modules. An
    expert tensor at this tier is refused by the offline converter, naming the missing op, so
    the decision is visible at conversion time and not as a silent dequant at serving time.
    Its ``op_moe_mmv`` is therefore the empty string, and ``ops_available`` asks only for the
    ops a tier declares.

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
    #: Empty for a tier with no MoE op family (pxq4hq). ``ops_available`` skips empty names,
    #: so declaring none is how a tier says "LINEAR only" rather than "library too old".
    op_moe_mmv: str
    #: Fused MoE decode block (pxq_moe_fused.cuh). Optional: a v13 library built before the
    #: fused kernels landed has the tier's other ops and not these, and ``fused_available``
    #: below is what the MoE method asks rather than assuming from the library file name.
    op_moe_gateup: str = ""
    op_moe_down_fold: str = ""
    op_moe_down_part: str = ""
    #: True when this tier does NOT index the shared SUB16 LUT and must be given its own sub
    #: table. Only pxq4hq. See the module docstring.
    own_sub: bool = False
    #: The op that uploads BOTH tables for a tier that owns its sub. Empty for every tier that
    #: goes through the shared pxq_set_book / pxq_set_sub pair.
    op_set_tables: str = ""
    #: The op that runs this tier's built-in kernel self-test, when it is not the shared
    #: ``pxq_selftest(tier)``.
    op_selftest: str = ""

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
    # pxq4hq: pxq4's book and code packing, one sub index per 8 elements instead of 16, hence
    # 128 B of scale SoA per slab and a 1152 B stride. LINEAR only, own SUB8 table.
    253: Tier(253, "pxq4hq", 1152, 16,
              "pxq4hq_dequant_out", "pxq4hq_mmv_out", "pxq4hq_linear_out", "",
              own_sub=True, op_set_tables="pxq4hq_set_tables",
              op_selftest="pxq4hq_selftest"),
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
    # An EMPTY name means the tier does not have that op family at all (pxq4hq has no MoE), as
    # opposed to a library too old to carry it. Asking for it would make a correct build look
    # unserviceable, so only declared ops are required.
    wanted = [op for op in (tier.op_dequant, tier.op_mmv, tier.op_linear, tier.op_moe_mmv) if op]
    if tier.op_set_tables:
        wanted.append(tier.op_set_tables)
    return all(hasattr(torch.ops.pxq4, op) for op in wanted)


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
    if not name:
        raise RuntimeError(
            f"pxq: tier {tier.name} has no {which} op family in this package. pxq4hq serves "
            f"LINEAR modules only -- it is the tier a policy buys the ATTENTION block up to, "
            f"and an expert tensor at this tier is refused by the offline converter rather "
            f"than dequantised silently at serving time. If a checkpoint reached here with a "
            f"{which} module at pxq4hq, the file was written by a converter that predates that "
            f"refusal; re-run the conversion.")
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
    if tier.own_sub:
        # ONE ENTRY POINT, BOTH TABLES. This tier's sub is not the shared SUB16, so it must not
        # travel through pxq_set_sub (which would overwrite every other tier's LUT with it) and
        # its book must not be applied without it: a book uploaded against the wrong scale
        # ladder is uniform weight error across the module, not a partial failure.
        if not hasattr(torch.ops.pxq4, tier.op_set_tables):
            raise RuntimeError(
                f"pxq: the loaded kernel library has no torch.ops.pxq4.{tier.op_set_tables}, so "
                f"tier {tier.name} cannot be served. PXQ4_LIB must point at a v17 or later "
                f"library (libpxq_sm60_v18.so / libpxq_sm70_v18.so); v14/v15/v16 carry "
                f"pxq2/pxq3/pxq4 only.")
        if sub is None:
            raise ValueError(
                f"pxq: tier {tier.name} owns its sub-scale table and cannot be uploaded without "
                f"one. The checkpoint records it in quantization_config.tier_subs[{tier.name!r}]; "
                f"a file that does not carry it was written by a converter that predates this "
                f"tier and must be re-converted.")
        b = torch.as_tensor(list(book), dtype=torch.float32)
        s = torch.as_tensor(list(sub), dtype=torch.float32)
        if b.numel() != tier.book_n:
            raise ValueError(f"pxq: tier {tier.name} wants a {tier.book_n}-entry book, "
                             f"got {b.numel()}")
        if s.numel() != 16:
            raise ValueError(f"pxq: tier {tier.name} wants a 16-entry sub table, got {s.numel()}")
        getattr(torch.ops.pxq4, tier.op_set_tables)(b, s)
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
    if tier.op_selftest:
        if not hasattr(torch.ops.pxq4, tier.op_selftest):
            raise RuntimeError(f"pxq: the loaded kernel library has no {tier.op_selftest} "
                               f"(need libpxq_*_v18.so or later for tier {tier.name})")
        return int(getattr(torch.ops.pxq4, tier.op_selftest)())
    if not hasattr(torch.ops.pxq4, "pxq_selftest"):
        raise RuntimeError("pxq: the loaded kernel library has no pxq_selftest "
                           "(need libpxq_*_v13.so or later)")
    return int(torch.ops.pxq4.pxq_selftest(tier.type_id))
