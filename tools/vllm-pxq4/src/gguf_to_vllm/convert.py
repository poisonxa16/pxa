"""convert.py — offline GGUF -> vLLM-loadable safetensors converter for a PXQ4 checkpoint.

Plan §5. Pure Python + numpy: no torch, no CUDA, no vLLM, no GPU. Everything except
the final byte-writing is exercised by ``--dry-run``, which plans the entire conversion from
the GGUF header alone and runs every structural self-check.

    python -m gguf_to_vllm.convert \
      --gguf   $PXA_MODELS_DIR/pxa-models/Qwen3.8-27B-PXQ4.gguf \
      --ref-hf $PXA_MODELS_DIR/hf/philbert440/Qwen3.8-27B-Uncensored-Cyber-W4A16-AWQ \
      --out    $PXA_MODELS_DIR/pxa-models/Qwen3.8-27B-PXQ4-vllm \
      --policy p1 [--encoder .../pxq4_encode.so] [--shard-size-gb 4] [--dry-run]

WHAT COMES OUT

  For every module the policy serves as PXQ4, TWO tensors and NO ``.weight``:

      <module>.pxq4_slabs   uint8   [N/64, K/32, 1088]   C-contiguous
      <module>.pxq4_anchor  float16 [N/64, 64]           C-contiguous

  derived from the GGUF blob by a PURE SPLIT — the header bytes and the slab bytes of each
  panel, reinterpreted, with no value recomputed (layout.split_blob). That is why the emitted
  checkpoint can be proven equal to the GGUF by a byte comparison rather than a numeric
  tolerance, and why ``--verify`` round-trips every tensor.

  Everything else is decoded to float16 ``<module>.weight``, and the 333 BF16 ``model.visual.*``
  tensors are copied byte-for-byte from ``--ref-hf`` so the vision tower is bit-identical to
  what the incumbent already serves.

THE ONE PLACE BYTES MOVE: GDN HEAD ORDER. The two checkpoints do not agree on the order of the
48 GDN value-heads — ggml is repeat-major, HF is k-head-major (namemap module docstring) — so
every per-v-head axis is gathered into HF order on the way out. It stays a byte move, because
a 128-row head block is exactly 2 panels and a 128-column head block exactly 4 slabs, so no
nibble, sub-scale or anchor value is touched and ``--verify`` still compares BYTES (it undoes
the gather first). ``ssm_a`` additionally takes ``A_log = log(-A)``. Both are enforced: a
GDN tensor emitted without its reorder fails ``_check_plan``, and with ``--ref-hf`` the
reorder is proved exactly, per layer, against the reference checkpoint before anything is
written (``gate_gdn_head_order``). An unpermuted GDN checkpoint loads, shards, passes every
byte gate and generates fluent garbage; that is why both checks are fatal rather than warnings.

WHY NOT vLLM'S GGUF LOADER. ``gguf.GGMLQuantizationType(252)`` raises inside
``GGUFReader._build_tensors``, killing the file open before any tensor is yielded; and vLLM's
generic GGUF sharder slices rows assuming per-row-contiguous blocks, which 64-row panel
interleave violates. Neither is patchable without forking three packages.

THE INVARIANT THIS FILE ENFORCES. Every vLLM linear module served by
``PXQ4LinearMethod`` is UNIFORMLY PXQ4 across all of its ``output_partition_sizes``. There is
no mixed-precision fused module, ever. That is why P1 leaves ``self_attn.qkv_proj`` in fp16
even though ``attn_q`` is already PXQ4 on disk: ``QKVParallelLinear`` is hard-wired in
``Qwen3NextAttention`` (qwen3_next.py:505) with no split seam, so a PXQ4 ``q`` beside an fp16
``k``/``v`` would need custom ``load_qkv_weight`` overrides — the single most likely source of
a silently mis-sharded, cleanly-loading, subtly-wrong model. P2c dissolves it instead.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import dequant_ref as D
from . import gguf_raw as G
from . import layout as L
from . import tiers as TT
from . import namemap as NM
from . import reference as R
from . import safetensors_io as ST

#: Files copied verbatim from --ref-hf. config.json is copied and then has ONLY its
#: quantization_config rewritten: keeping every architectural field byte-identical to what the
#: incumbent already runs removes a whole class of "the fork reads a field we did not think
#: about" failure.
COPY_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "chat_template.json",
)

VISUAL_PREFIX = "model.visual."


@dataclass
class Emit:
    """One planned output tensor."""
    name: str
    kind: str                  # "pxq4" | "dense" | "copy"
    dtype: str                 # safetensors dtype string
    shape: tuple[int, ...]
    nbytes: int
    src: str                   # ggml tensor name, or "<ref-hf>" for a verbatim copy
    note: str = ""
    #: Non-empty iff a GDN v-head reorder was applied on the way out. ``_check_plan`` REQUIRES
    #: it on every tensor with a v-head axis: an unpermuted GDN checkpoint loads cleanly and
    #: generates fluent garbage, so "we forgot" has to be a hard failure, not a silence.
    perm: str = ""
    #: >=0 iff this emit is one expert's slice of a 3-D ggml expert stack. The writer needs it
    #: to know WHICH sub-tensor of ``src`` to cut, since E emits share one source name.
    expert: int = -1
    #: ggml type id of the PANEL TIER this emit carries (252 pxq4 / 254 pxq2 / 255 pxq3), or 0
    #: for a dense/copy emit. It is what decides the slab stride, and the stride is the ONLY
    #: thing that distinguishes the three formats on disk -- reading a pxq2 tensor at the pxq4
    #: stride yields a well-formed array of the wrong bytes -- so it is carried explicitly
    #: from the GGUF directory all the way into config.json rather than being re-derived.
    tier: int = 0


@dataclass
class Plan:
    emits: list[Emit] = field(default_factory=list)
    module_types: dict[str, set[str]] = field(default_factory=dict)
    reencode: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    gdn_geometry: Any = None
    #: modules whose fused vLLM parameter --fuse-uniform-pxq4 promoted to a single pxq4 tier.
    #: Recorded so the run PRINTS which tensors paid a second quantization pass -- a silent
    #: extra lossy step is exactly what this package refuses to do anywhere else.
    fused_uniform: list[str] = field(default_factory=list)
    #: module -> (tensor count, bytes) for tensors that ARE a native PXQ panel tier on disk
    #: (pxq2/pxq3/pxq4) but landed in the dense f16 branch because this policy's module list
    #: does not declare their module servable. Populated in ``build_plan``'s dense-emit branch;
    #: printed as a warning in ``main`` (never fatal -- some of this is intentional, e.g. a
    #: fused qkv_proj whose k/v shards are q8_0). See namemap.py POLICY_MODULES: a policy
    #: written for one architecture's module names (e.g. MoE ``mlp.experts``) silently forces
    #: EVERY tensor of a different architecture (e.g. dense ``mlp.gate_proj``) dense, even when
    #: those tensors are natively PXQ2/PXQ3 on disk -- this is exactly the mistake that sent
    #: policy m2 against a dense qwen35-PXQ3 source (2026-09-08): 41.7 GB of native pxq3
    #: ffn_gate/up/down went to fp16 because m2's module list only names MoE leaf names, and
    #: nothing printed said so. This field exists so the next run SEES it instead of finding
    #: out from a checkpoint that only boots with --cpu-offload-gb.
    native_dense: dict[str, tuple[int, int]] = field(default_factory=dict)

    def total_bytes(self) -> int:
        return sum(e.nbytes for e in self.emits)


# ---------------------------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------------------------
def _hf_of(ggml_name: str, kv: dict) -> str | None:
    return NM.GGML_TO_HF(ggml_name, kv)


def _gdn_shapes(gg) -> dict[str, tuple[int, ...]]:
    """Bare ggml suffix -> ne, taken from the first block that actually has a GDN stack.

    Used only to cross-check the geometry the KVs claim against the tensors that exist.
    """
    for name in gg.order:
        m = NM._BLK.match(name)
        if not m or m.group(2) != "attn_gate.weight":
            continue
        layer = m.group(1)
        return {NM.ggml_suffix(n): gg.tensors[n].dims
                for n in gg.order if n.startswith(f"blk.{layer}.")}
    return {}


def gdn_perm_for(ggml_name: str, ti, geom) -> tuple[int, list[int]] | None:
    """(axis of the emitted tensor, element gather) for one ggml tensor, or None.

    The axis is expressed against ``logical_shape`` (reversed ne, i.e. torch order), which is
    the axis the emitted safetensors tensor has, so a caller never converts axes twice. For a
    PXQ4-served tensor axis 0 is the panel axis and axis 1 the slab axis.
    """
    suffix = NM.ggml_suffix(ggml_name)
    spec = NM.GDN_PERM_SPEC.get(suffix)
    if spec is None:
        return None
    axis = spec[0]
    shape = ti.logical_shape
    if axis >= len(shape):
        raise SystemExit(f"{ggml_name}: GDN permutation wants axis {axis} of a "
                         f"{len(shape)}-D tensor")
    return NM.gdn_permutation(suffix, geom, shape[axis])


def _apply_perm_pxq4(slabs, anchor, perm):
    """Apply a v-head reorder to an already-split PXQ4 pair. Pure byte move (layout.py)."""
    axis, gather = perm
    if axis == 0:
        pidx = L.block_gather_to_panels(np.asarray(gather, dtype=np.int64))
        return L.gather_panels(slabs, anchor, pidx)
    sidx = L.col_gather_to_slabs(np.asarray(gather, dtype=np.int64))
    return L.gather_slabs(slabs, sidx), anchor


def _unapply_perm_pxq4(slabs, anchor, perm):
    """Inverse of ``_apply_perm_pxq4``, so the byte round-trip gate can still compare to the
    GGUF after the emitted bytes have been reordered."""
    axis, gather = perm
    if axis == 0:
        pidx = L.block_gather_to_panels(np.asarray(gather, dtype=np.int64))
        return L.gather_panels(slabs, anchor, L.unpermute_index(pidx))
    sidx = L.col_gather_to_slabs(np.asarray(gather, dtype=np.int64))
    return L.gather_slabs(slabs, L.unpermute_index(sidx)), anchor


def _pxq4_emit_names(hf_weight_name: str) -> tuple[str, str]:
    """``...mlp.gate_proj.weight`` -> (``...mlp.gate_proj.pxq4_slabs``, ``....pxq4_anchor``).

    The stem keeps the ON-DISK module name, not the fused one: vLLM's ``load_weights`` rewrites
    ``gate_proj`` -> ``gate_up_proj`` itself via ``packed_modules_mapping`` and then looks the
    result up in ``params_dict``, so ``...gate_up_proj.pxq4_slabs`` is found with stock loaders
    and no custom weight_loader. Pre-fusing here would break that rewrite.
    """
    stem = hf_weight_name[: -len(".weight")] if hf_weight_name.endswith(".weight") else hf_weight_name
    return stem + ".pxq4_slabs", stem + ".pxq4_anchor"


def _moe_param_of(hf_name: str) -> str:
    """Routed-expert projection -> the fused FusedMoE parameter that will hold it.

    gate_proj and up_proj are concatenated into ``w13`` on the output axis; down_proj is
    ``w2``. The uniformity invariant applies per PARAMETER, not per module: two tensors that
    end up in one tensor must be one tier, two that do not need not be. Returns a ``#``-
    prefixed suffix so the key can never collide with a real module name.
    """
    stem = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else hf_name
    leaf = stem.rsplit(".", 1)[-1]
    if leaf in ("gate_proj", "up_proj"):
        return "#w13"
    if leaf == "down_proj":
        return "#w2"
    raise SystemExit(f"unexpected routed-expert projection {hf_name!r}: this converter knows "
                     f"gate_proj/up_proj (-> w13) and down_proj (-> w2) only")


def _mixed_tier_modules(gg, policy: str, limit_layers: int = 0) -> set[str]:
    """Modules this policy serves whose ONE vLLM parameter would hold more than one tier.

    A fused parameter (``self_attn.qkv_proj`` = ggml ``attn_q`` + ``attn_k`` + ``attn_v``) has
    one slab stride and one book, so every shard in it must be the same tier. When the shards
    disagree on disk -- q native pxq3, k/v q8_0, which is what the rev-2 backbone table emits
    for a dense 27B -- the plan is unwritable and the whole module falls back to fp16. This
    names those modules so ``--fuse-uniform-pxq4`` can re-encode the panel shards up to pxq4
    and keep the parameter servable.

    Expert stacks are deliberately not considered: their uniformity is keyed on the fused
    FusedMoE parameter (``w13``/``w2``) and re-encoding a 256-expert stack is not offered.
    """
    kv = gg.kv
    seen: dict[str, set[str]] = {}
    for name in gg.order:
        if limit_layers:
            m = NM._BLK.match(name)
            if m and int(m.group(1)) >= limit_layers:
                continue
        hf = _hf_of(name, kv)
        if hf is None or hf.startswith("@CONCAT:") or "{e}" in hf:
            continue
        module = NM.HF_MODULE_OF(hf)
        if module.rsplit(".", 1)[-1] in NM.POLICY_Q8.get(policy, frozenset()):
            continue
        if not NM.is_pxq4_module(module, policy):
            continue
        ti = gg.tensors[name]
        seen.setdefault(module, set()).add(
            TT.tier_of(ti.type_id).name if TT.is_pxq(ti.type_id) else "pxq4")
    return {m for m, tiers in seen.items() if len(tiers) > 1}


def build_plan(gg: G.GGUFFile | G.GGUFHeaderOnly, policy: str, ref_hf: str | None,
               have_encoder: bool, limit_layers: int = 0, with_visual: bool = True,
               fuse_uniform: bool = False) -> Plan:
    """``limit_layers`` and ``with_visual`` exist for smoke tests only: they produce a
    checkpoint that is NOT servable, and the caller is expected to say so. They let the whole
    emission path — decode, split, shard-writing, config — run against real bytes in a minute
    instead of against 23 GB in an hour."""
    plan = Plan()
    kv = gg.kv
    #: hf target -> {part index: ggml tensor name}. See the "@CONCAT" branch below.
    concat: dict[str, dict[int, str]] = {}
    # The GDN v-head order differs between the two checkpoints (namemap module docstring), so
    # the geometry is needed before a single tensor is planned. Derived from the file's KVs and
    # cross-checked against the shapes the file actually has: permuting head blocks on guessed
    # geometry would be worse than not permuting at all.
    geom = NM.gdn_geometry(kv)
    geom.check_against_tensors(_gdn_shapes(gg))
    plan.gdn_geometry = geom

    #: modules whose fused vLLM parameter would hold MORE THAN ONE tier if every shard kept
    #: its on-disk format, and which --fuse-uniform-pxq4 therefore re-encodes to a uniform
    #: pxq4. Empty unless the flag is given: the re-encode costs one extra quantization pass
    #: on a tensor that is already quantized, so it never happens by default.
    force_pxq4 = _mixed_tier_modules(gg, policy, limit_layers) if fuse_uniform else set()
    plan.fused_uniform = sorted(force_pxq4)

    for name in gg.order:
        if limit_layers:
            m = NM._BLK.match(name)
            if m and int(m.group(1)) >= limit_layers:
                plan.skipped.append((name, f"--limit-layers {limit_layers}"))
                continue
        ti = gg.tensors[name]
        hf = _hf_of(name, kv)
        if hf is None:
            why = getattr(NM.Q4, "SKIP", {}).get(name) if NM.arch_of(kv) == NM.Q4.ARCH else None
            plan.skipped.append((name, why or "MTP / not mapped in P1-P2 (plan §3, P3 work)"))
            continue

        # --- @CONCAT: N ggml tensors that vLLM keeps as ONE ------------------------------
        # The QSA indexer's q (N=512) and k (N=128) are separate on disk and one
        # ReplicatedLinear ``index_qk_proj`` of 640 rows in the fork (indexer_qsa.py:133-138),
        # and it is NOT in packed_modules_mapping -- so nothing downstream will fuse them for
        # us. Emitting them separately does not raise: AutoWeightsLoader swallows the two
        # unexpected suffixes and the layer runs on uninitialised weights. Collected here and
        # emitted as one tensor after the loop, when every part is known.
        if hf.startswith("@CONCAT:"):
            _, target, part = hf.split(":")
            slot = concat.setdefault(target, {})
            if int(part) in slot:
                raise SystemExit(f"{name}: @CONCAT part {part} of {target!r} is already "
                                 f"claimed by {slot[int(part)]!r}")
            slot[int(part)] = name
            continue

        # --- 3-D expert stacks fan out to E per-expert emits -------------------------------
        if "{e}" in hf:
            n_exp = NM.n_experts(kv)
            if n_exp <= 0:
                raise SystemExit(
                    f"{name} is an expert stack but the GGUF declares no expert_count under "
                    f"arch {kv.get('general.architecture')!r}. Refusing to guess E.")
            if len(ti.dims) != 3 or ti.dims[2] != n_exp:
                raise SystemExit(
                    f"{name}: expert stack expected ne=(K, N, {n_exp}) but the file says "
                    f"ne={tuple(ti.dims)}.")
            module = NM.HF_MODULE_OF(hf.format(e=0))
            if not NM.is_pxq4_module(module, policy):
                raise SystemExit(
                    f"policy {policy} does not serve {module!r} as PXQ4, so the {n_exp} experts "
                    f"of {name} would be emitted DENSE as fp16. For this model that is the "
                    f"3.4x-over-VRAM failure (dense fp16 experts do not fit) -- "
                    f"refusing rather than writing a checkpoint that cannot be loaded. Use a "
                    f"policy whose module list contains {NM.module_suffix(module)!r}.")
            if not TT.is_pxq(ti.type_id):
                raise SystemExit(
                    f"{name} is ggml type {ti.type}. Expert stacks are served as a BYTE MOVE "
                    f"of a native panel tier (pxq2 254 / pxq3 255 / pxq4 252 / pxq4hq 253); "
                    f"re-encoding a 256-expert stack is not offered, and decoding one to fp16 "
                    f"is the over-VRAM failure this policy exists to avoid.")
            if ti.type_id == TT.PXQ4HQ:
                # THE ONE TIER THAT IS LINEAR-ONLY, refused HERE rather than at serving time.
                # The vLLM kernel package gives pxq4hq a dequant, an mmv, a tensor-core arena
                # path and a linear dispatcher -- and no expert-indexed mmv, because the tier
                # exists to carry the ATTENTION block a policy buys up, not the experts. A
                # checkpoint that reached the runtime with pxq4hq experts would die at layer
                # construction on a missing torch.ops.pxq4.pxq4hq_moe_mmv_out, hours into a
                # load, so the file is refused at conversion time with the op named.
                raise SystemExit(
                    f"{name} is a pxq4hq expert stack. This converter serves pxq4hq for "
                    f"LINEAR modules only: the vLLM kernel package has no "
                    f"torch.ops.pxq4.pxq4hq_moe_mmv_out, so those experts have no decode path "
                    f"and would fail at layer construction. Two ways forward: quantize the "
                    f"experts at pxq2/pxq3/pxq4 and let the POLICY buy only the attention "
                    f"block up to pxq4hq (which is what --pxq-policy balanced/attn4 does), or "
                    f"serve this file with llama-server, which runs every tier on every "
                    f"expert.")
            tier = TT.tier_of(ti.type_id)
            N, K = ti.ne1, ti.ne0
            TT.assert_geometry(N, K)
            P_, S_ = N // 64, K // 32
            for e_ in range(n_exp):
                sl_name, an_name = _pxq4_emit_names(hf.format(e=e_))
                note = f"native {tier.name} expert slice"
                plan.emits.append(Emit(sl_name, "pxq4", "U8", (P_, S_, tier.slab_bytes),
                                       P_ * S_ * tier.slab_bytes, name, note,
                                       expert=e_, tier=tier.type_id))
                plan.emits.append(Emit(an_name, "pxq4", "F16", (P_, 64), P_ * 64 * 2,
                                       name, note, expert=e_, tier=tier.type_id))
            # UNIFORMITY AT THE RIGHT GRANULARITY. vLLM's FusedMoE keeps gate+up in ONE
            # tensor (w13) and down in another (w2), so gate and up must share a tier but
            # down need not. Keying the uniformity check on the fused PARAMETER rather than
            # on the module is what makes a pxq2-gate/up + pxq3-down file (what the
            # Flash-Next quantizer emits) expressible without weakening the invariant.
            plan.module_types.setdefault(module + _moe_param_of(hf), set()).add(tier.name)
            continue
        # -----------------------------------------------------------------------------------

        module = NM.HF_MODULE_OF(hf)
        # ---- int8 head ---------------------------------------------------------------
        # Checked before the pxq4/dense split because it is neither: two tensors, not one,
        # and a format the panel machinery knows nothing about.
        if module.rsplit(".", 1)[-1] in NM.POLICY_Q8.get(policy, frozenset()):
            N_, K_ = ti.ne1, ti.ne0
            stem = hf[: -len(".weight")] if hf.endswith(".weight") else hf
            src_note = ("q8 head from the reference checkpoint's unquantized copy"
                        if ref_hf else f"q8 head re-quantized from ggml {ti.type}")
            plan.emits.append(Emit(stem + ".weight", "q8", "I8", (N_, K_), N_ * K_,
                                   name, src_note))
            plan.emits.append(Emit(stem + ".weight_scale", "q8", "F16", (N_, 1), N_ * 2,
                                   name, src_note))
            plan.module_types.setdefault(module, set()).add("q8")
            continue
        want_pxq4 = NM.is_pxq4_module(module, policy)
        perm = gdn_perm_for(name, ti, geom)
        perm_note = ""
        if perm is not None:
            perm_note = (f"gdn v-head reorder on axis {perm[0]} "
                         f"({geom.n_v_heads} heads, {geom.repeats}x{geom.n_k_heads})")
        xform = NM.value_transform(NM.ggml_suffix(name), kv)

        if want_pxq4:
            N, K = ti.ne1, ti.ne0
            L.assert_geometry(N, K)
            slab_name, anch_name = _pxq4_emit_names(hf)
            if xform is not None:
                raise SystemExit(
                    f"{name} needs the value transform {xform[1]!r}, which cannot be expressed "
                    f"as a byte move — it must not be served as PXQ4.")
            forced = (module in force_pxq4 and TT.is_pxq(ti.type_id)
                      and ti.type_id != TT.PXQ4)
            if TT.is_pxq(ti.type_id) and not forced:
                tname = TT.tier_of(ti.type_id).name
                note = (f"native {tname}, panel-permuted byte move" if perm
                        else f"native {tname}, pure byte split")
            elif forced:
                # --fuse-uniform-pxq4: this tensor IS a native panel tier and could have been
                # a byte move, but a sibling shard of the same fused vLLM parameter is not,
                # and one parameter is one stride and one book. Re-encoding the
                # panel shard up to pxq4 is what makes the whole parameter servable instead of
                # dense. It costs a SECOND quantization on this tensor -- pxq3 -> fp32 -> pxq4
                # -- and the wrel printed at encode time is the honest measure of it.
                note = f"RE-ENCODE {ti.type} -> pxq4 (uniform fused parameter)"
                plan.reencode.append(name)
                if not have_encoder:
                    raise SystemExit(
                        f"--fuse-uniform-pxq4 has to re-encode {name} ({ti.type} -> pxq4) so "
                        f"{module} is one tier, but no --encoder was given. Refusing to fall "
                        f"back silently.")
            else:
                note = f"RE-ENCODE {ti.type} -> pxq4"
                plan.reencode.append(name)
                if not have_encoder:
                    raise SystemExit(
                        f"policy {policy} requires re-encoding {name} ({ti.type} -> pxq4) but "
                        f"no --encoder was given. Refusing to silently fall back to fp16: that "
                        f"would produce a checkpoint that loads, runs, and is quietly slower "
                        f"than the policy claims.")
            # A tensor that is already a panel tier keeps that tier (byte move); anything
            # else is re-encoded, and the encoder only produces pxq4. A forced tensor is
            # re-encoded too, so it lands at pxq4 like the rest of its parameter.
            out_tier = (TT.tier_of(ti.type_id) if TT.is_pxq(ti.type_id) and not forced
                        else TT.TIERS[TT.PXQ4])
            P, S = N // 64, K // 32
            plan.emits.append(Emit(slab_name, "pxq4", "U8", (P, S, out_tier.slab_bytes),
                                   P * S * out_tier.slab_bytes, name, note, perm_note,
                                   tier=out_tier.type_id))
            plan.emits.append(Emit(anch_name, "pxq4", "F16", (P, 64), P * 64 * 2, name, note,
                                   perm_note, tier=out_tier.type_id))
            plan.module_types.setdefault(module, set()).add(out_tier.name)
        else:
            shape = ti.logical_shape
            if hf.endswith("shared_expert_gate.weight"):
                # ggml stores the shared-expert gate as a 1-D vector, ne=(2048,), because it
                # is a single output row. vLLM builds it as ReplicatedLinear(hidden_size, 1)
                # whose weight is 2-D [1, hidden] (confirmed against the reference checkpoint:
                # shape [1, 2048]). Emitting the bare 1-D vector gives
                #   AssertionError: Tried to load weights of size torch.Size([2048])
                #                   to a parameter of size torch.Size([1, 2048])
                # at default_weight_loader. This is a pure reshape -- no values move.
                shape = (1, int(shape[0]))
            elif hf.endswith("conv1d.weight"):
                # ggml ne=(4, 10240) -> HF [10240, 1, 4]. The middle axis is the depthwise
                # conv's in-channels-per-group of 1; HF stores conv1d weights as
                # [out_channels, in_channels/groups, kernel]. See the ASSUMPTION in namemap.py.
                shape = (ti.ne1, 1, ti.ne0)
            n = 2
            for d in shape:
                n *= d
            note = f"{ti.type} -> f16"
            if xform is not None:
                note += f"; {xform[1]}"
                perm_note = (perm_note + "; value transform") if perm_note else "value transform"
            plan.emits.append(Emit(hf, "dense", "F16", tuple(shape), n, name, note, perm_note))
            plan.module_types.setdefault(module, set()).add("dense")
            # This tensor is ALREADY a native PXQ panel tier on disk (pxq2/pxq3/pxq4) but is
            # going out dense anyway, because `policy` does not declare `module` servable --
            # not because the bytes need re-encoding. Sometimes that is deliberate (a fused
            # module whose other shard is a non-PXQ type, e.g. self_attn.qkv_proj beside q8_0
            # k/v); sometimes it is this policy's module-name table simply not matching this
            # architecture (see the Plan.native_dense docstring). Either way it is worth a
            # human seeing the byte total before it turns into an OOM or a --cpu-offload-gb
            # surprise three steps downstream.
            if TT.is_pxq(ti.type_id):
                c, b = plan.native_dense.get(module, (0, 0))
                plan.native_dense[module] = (c + 1, b + n)

    for target, parts in sorted(concat.items()):
        idx = sorted(parts)
        if idx != list(range(len(idx))):
            raise SystemExit(f"@CONCAT {target!r}: parts {idx} are not 0..{len(idx)-1}. A "
                             f"missing part would be a silently short tensor.")
        srcs = [parts[i] for i in idx]
        tis = [gg.tensors[s_] for s_ in srcs]
        K = tis[0].ne0
        for s_, t_ in zip(srcs, tis):
            if t_.ne0 != K:
                raise SystemExit(f"@CONCAT {target!r}: {s_} has K={t_.ne0}, expected {K}")
            if NM.ggml_suffix(s_) in NM.GDN_PERM_SPEC:
                raise SystemExit(f"@CONCAT {target!r}: {s_} carries a GDN v-head axis; "
                                 f"concatenating a permuted tensor is not implemented and "
                                 f"would be silent.")
        N = sum(t_.ne1 for t_ in tis)
        note = "concat " + " || ".join(f"{s_}[{t_.ne1}]" for s_, t_ in zip(srcs, tis))
        plan.emits.append(Emit(target, "concat", "F16", (N, K), N * K * 2,
                               ",".join(srcs), note))
        plan.module_types.setdefault(NM.HF_MODULE_OF(target), set()).add("dense")

    if ref_hf and with_visual:
        for e in _plan_visual(ref_hf):
            plan.emits.append(e)

    _check_plan(plan, policy, arch=NM.arch_of(kv))
    return plan


def _plan_visual(ref_hf: str) -> list[Emit]:
    """The 333 BF16 vision tensors, copied verbatim.

    They cost ~0.21 GiB/GPU resident and ZERO decode bandwidth, and dropping them is not the
    two-line win it looks like: ``Qwen3_5ForCausalLM`` exists (qwen3_5.py:772) but is not
    registered (registry.py:560) and does not declare ``IsHybrid`` (qwen3_5.py:658-664 vs :819),
    so registering it would silently lose the hybrid mamba-state cache config that
    ``ModelConfig.is_hybrid`` drives (config/model.py:1630-1631, :1764). Copying is correct and
    cheap; dropping is a P3 experiment.
    """
    idx_path = os.path.join(ref_hf, "model.safetensors.index.json")
    out: list[Emit] = []
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            wm = json.load(f)["weight_map"]
        shards = {}
        for k, fn in wm.items():
            if k.startswith(VISUAL_PREFIX):
                shards.setdefault(fn, []).append(k)
        for fn, keys in shards.items():
            hdr = ST.read_header(os.path.join(ref_hf, fn))
            for k in keys:
                e = hdr[k]
                beg, end = e["data_offsets"]
                out.append(Emit(k, "copy", e["dtype"], tuple(e["shape"]), end - beg,
                                f"<ref-hf>/{fn}", "verbatim vision tower"))
    else:
        for fn in ("model.safetensors",):
            p = os.path.join(ref_hf, fn)
            if not os.path.exists(p):
                continue
            hdr = ST.read_header(p)
            for k, e in hdr.items():
                if k == "__metadata__" or not k.startswith(VISUAL_PREFIX):
                    continue
                beg, end = e["data_offsets"]
                out.append(Emit(k, "copy", e["dtype"], tuple(e["shape"]), end - beg,
                                f"<ref-hf>/{fn}", "verbatim vision tower"))
    return sorted(out, key=lambda e: e.name)


def _check_plan(plan: Plan, policy: str, arch: str = "") -> None:
    """Plan §5.6 checks 1, 5 and 6. All of these fail the run; none of them warn."""
    # (6) §3.1 uniformity, restated for tiers: everything that ends up in ONE vLLM parameter
    # must be ONE type -- all pxq2, or all pxq3, or all pxq4, or all dense. Mixing tiers
    # ACROSS parameters is fine and is the whole point (experts pxq2, backbone pxq4); mixing
    # them INSIDE one is the silent mis-shard we refuse to write, because a fused parameter
    # has a single slab stride and a single book.
    mixed = {m: sorted(t) for m, t in plan.module_types.items() if len(t) > 1}
    if mixed:
        raise SystemExit(
            f"policy {policy} violates the §3.1 uniformity invariant — these vLLM parameters "
            f"would hold more than one type at once, which needs a custom per-shard weight "
            f"loader and is exactly the silent mis-shard we refuse to write: {mixed}. "
            f"(A '#w13' / '#w2' suffix names a fused FusedMoE parameter, not a module.)")

    # (5) shard arithmetic at every TP degree we intend to serve.
    for e in plan.emits:
        if e.kind != "pxq4" or not e.name.endswith(".pxq4_slabs"):
            continue
        P, S, sb = e.shape
        if e.tier and sb != TT.slab_bytes(e.tier):
            raise SystemExit(
                f"{e.name}: planned slab stride {sb} does not match tier "
                f"{TT.tier_of(e.tier).name}'s {TT.slab_bytes(e.tier)}. This is a converter "
                f"bug and it would produce a checkpoint that loads and decodes garbage.")
        N, K = P * 64, S * 32
        row_parallel = any(e.name.endswith(s + ".pxq4_slabs")
                           for s in ("down_proj", "o_proj", "out_proj"))
        # WHICH TP DEGREES THIS TENSOR CLAIMS TO SERVE. A routed expert's output axis is the
        # per-expert intermediate size, and a model whose experts are 640 wide (10 panels) is
        # shardable 2 ways and not 4 -- at TP=4 FusedMoE's w13 gate/up boundary lands mid-panel
        # and pxq4_vllm.moe._place silently takes 2 of the 5 panels it needs. Declaring the
        # narrower set here is what makes the checkpoint honest about it instead of the engine
        # discovering it as fluent garbage; a 4-way shape for this model is expert parallel or
        # pipeline parallel, where the expert tensors stay whole.
        tps = (1, 2, 4)
        if arch == NM.Q4.ARCH and ".experts." in e.name:
            tps = NM.Q4.EXPERT_TP_SIZES
        L.assert_shardable(N, K, tps, row_parallel=row_parallel, name=e.name)

    names = [e.name for e in plan.emits]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"duplicate output tensor names: {dup}")

    # (7) THE GDN HEAD-ORDER GATE. Every emitted tensor with a v-head axis must carry the
    # reorder, and every GDN tensor without one must have said so out loud in GDN_NO_PERM.
    # This exists because the failure mode is invisible: an unpermuted GDN checkpoint loads,
    # shards, passes every byte gate, and generates fluent garbage. "Forgot to permute" and
    # "a new ssm_* tensor appeared" both have to be run-stopping, not silent.
    unpermuted, undeclared = [], []
    for e in plan.emits:
        if e.kind == "copy":
            continue
        suf = NM.ggml_suffix(e.src)
        if suf not in NM._GDN_MAP:
            continue
        if suf in NM.GDN_PERM_SPEC:
            if not e.perm:
                unpermuted.append(e.name)
        elif suf not in NM.GDN_NO_PERM:
            undeclared.append(f"{e.name} (ggml {suf})")
    if unpermuted:
        raise SystemExit(
            f"{len(unpermuted)} GDN tensors would be emitted WITHOUT the v-head reorder, e.g. "
            f"{unpermuted[:4]}. ggml orders the "
            f"{getattr(plan.gdn_geometry, 'n_v_heads', 48)}-way value-head "
            f"axis repeat-major and HF orders it k-head-major (namemap module docstring); "
            f"shipping them unpermuted produces a model that loads and generates fluent "
            f"garbage. This is a converter bug, not a policy choice.")
    if undeclared:
        raise SystemExit(
            f"GDN tensors with no entry in namemap.GDN_PERM_SPEC and no entry in GDN_NO_PERM: "
            f"{undeclared[:8]}. Decide explicitly whether each has a v-head axis — defaulting "
            f"to 'no permutation' is exactly the bug this gate exists for.")


# ---------------------------------------------------------------------------------------------
# quantization_config
# ---------------------------------------------------------------------------------------------
def build_quantization_config(gg, policy: str, plan: "Plan | None" = None,
                             books_from_defaults: bool = False) -> dict:
    """The ``quantization_config`` block of the emitted config.json.

    TIERS. A pre-2026-09-05 checkpoint was uniformly pxq4 and recorded one book and one sub.
    A tiered checkpoint records, per tier actually present in the plan, the table that tier's
    tensors were quantized WITH -- read out of the GGUF's own ``pxa.pxq2.book`` /
    ``pxa.pxq3.book`` / ``pxa.pxq6.book`` KVs, never assumed. That is not bureaucracy: the
    quantizer's books are overridable at build time (PXA_PXQ2_V3, PXA_PXQ_CEIL_V2,
    PXA_PXQ*_BOOK) and decoding a v3-book file with the v1 table is a silent, uniform weight
    error across every expert in the model, with no shape, checksum or load-time symptom.

    ``quant_method`` becomes "pxq" as soon as more than one tier is present, and stays "pxq4"
    for a single-tier pxq4 file so that every checkpoint already in the field, and every
    launcher passing --quantization pxq4, is bit-for-bit unaffected.
    """
    kv = gg.kv
    NM.assert_policy_supported(policy)

    # Which tiers does this checkpoint actually contain? Read off the PLAN, not off the file:
    # a file may hold tiers that no served module uses, and declaring a tier we do not serve
    # would make the runtime demand a book for nothing.
    tiers_present: set[int] = set()
    if plan is not None:
        tiers_present = {e.tier for e in plan.emits if e.tier}
    if not tiers_present:
        tiers_present = {TT.PXQ4}

    # The SUB16 LUT is SHARED by every tier -- the PXQ2 and PXQ3 engine headers both reuse
    # PXQ6's verbatim, and the shipped files agree byte for byte. So there is ONE sub, and if
    # two tiers in one file disagreed about it that would mean two quantizer runs with
    # different overrides were spliced together, which is a corrupt artifact.
    subs: dict[str, np.ndarray] = {}
    books: dict[str, list[float]] = {}
    #: tiers whose tables --books-from-defaults supplied because the file did not record them
    assumed: list[str] = []
    for tid in sorted(tiers_present):
        t = TT.tier_of(tid)
        book = kv.get(f"pxa.{t.name}.book")
        sub = kv.get(f"pxa.{t.name}.sub")
        if t.type_id == TT.PXQ4HQ:
            # PXQ4HQ'S TABLES, AND THE FORMAT HOLE THEY HAVE TO SURVIVE.
            #
            # The tier shares the frozen PX16 book with pxq4 and has its OWN sub (SUB8). The
            # quantizer stamps pxa.pxq4hq.book / pxa.pxq4hq.sub for it -- but only since this
            # change. Before it, the 4-bit family recorded ONE pair of KVs, pxa.pxq6.book and
            # pxa.pxq6.sub, whose meaning was switched by the string pxa.pxq6.tier: "core"
            # means pxa.pxq6.sub IS SUB16, "hq" means it is SUB8. A mixed pxq4 + pxq4hq file
            # therefore recorded the HQ sub ONLY, and its pxq4 tensors' SUB16 is not in the
            # file at all -- the frozen default is the only source for them, which is sound
            # because pxq4's tables are the frozen ones and are cross-checked below.
            #
            # So: prefer the tier's own KVs; fall back to the tier-string reading; and if
            # neither is there, the frozen SUB8 is correct for the same reason the frozen
            # PXQ4 tables are -- both are checked against what the file recorded whenever it
            # recorded anything.
            if book is None:
                book = kv.get("pxa.pxq6.book")
            if sub is None and str(kv.get("pxa.pxq6.tier", "")) == "hq":
                sub = kv.get("pxa.pxq6.sub")
            if book is None or sub is None:
                book = R.BOOK.tolist() if book is None else book
                if sub is None:
                    sub = TT.sub_of(TT.PXQ4HQ).tolist()
                print("    pxq4hq tables: the file records no pxa.pxq4hq.book/sub and its "
                      "pxa.pxq6.tier is not 'hq'; recording the frozen PX16 book and SUB8 "
                      "table. This is right for any file the stock quantizer produced and "
                      "wrong only for one built with a table override, which no released "
                      "quantizer offers for this tier.", file=sys.stderr)
            b = np.asarray(book, dtype=np.float32)
            sv = np.asarray(sub, dtype=np.float32)
            TT.check_book(b, t.type_id)
            TT.check_sub(sv)
            if not np.array_equal(b, R.BOOK):
                raise SystemExit(
                    "the pxq4hq book in this file differs from the frozen PX16 book. pxq4hq "
                    "shares that book with pxq4 bit for bit, so a difference means a table "
                    "override the vendored CUDA header would decode every hq weight against "
                    "wrongly.")
            if np.array_equal(sv, R.SUB):
                raise SystemExit(
                    "the pxq4hq sub table in this file is the SHARED SUB16, not a bs8 table. "
                    "pxq4hq spends one sub index per 8 elements and its levels are a "
                    "different fit; SUB16 here means the file's pxa.pxq6.tier said 'core' "
                    "while it carries hq tensors, i.e. it was written by a quantizer whose "
                    "stamping predates mixed-tier files. Re-stamp it with "
                    "tools/pxq-stamp-books.py before converting.")
        elif t.type_id == TT.PXQ4:
            # PXQ4 records itself under the pxq6 namespace (it is the PXQ6 core tier). If the
            # file has no pxq4 tensors at all -- our pxq4 modules were RE-ENCODED by us -- then
            # the encoder's compiled-in frozen tables are the truth, and reference.py is the
            # single place they are written down.
            book = kv.get("pxa.pxq6.book", book)
            sub = kv.get("pxa.pxq6.sub", sub)
            if book is None or sub is None:
                book, sub = R.BOOK.tolist(), R.SUB.tolist()
                print("    pxq4 tables: no pxa.pxq6.book/sub in the file (its pxq4 tensors are "
                      "re-encoded by us); recording the frozen PXQ6 tables", file=sys.stderr)
            b = np.asarray(book, dtype=np.float32)
            sv = np.asarray(sub, dtype=np.float32)
            R.check_tables(b, sv)
            if not np.array_equal(b, R.BOOK) or not np.array_equal(sv, R.SUB):
                raise SystemExit(
                    "pxa.pxq6.book/sub in the file differ from the frozen PXQ6 tables. This "
                    "file was built with a table override; the vendored CUDA header would "
                    "decode every pxq4 weight in it wrong.")
        else:
            if (book is None or sub is None) and books_from_defaults:
                # --books-from-defaults. The quantizer before rc3 stamped book/sub only for
                # the ftype that was REQUESTED, so a mixed file (say a pxq2 request that also
                # emitted pxq3 tensors) reaches here with one tier undocumented, even though
                # llama.cpp decodes it correctly with the compiled-in tables it was built
                # with. Substituting those tables is right for such a file and WRONG for a
                # file built with an override, and nothing in the file distinguishes the two
                # -- which is why this is a flag, why it says out loud what it assumed, and
                # why the assumption is stamped into the output's own metadata below.
                book = TT.book_of(t.type_id).tolist() if book is None else book
                sub = R.SUB.tolist() if sub is None else sub
                assumed.append(t.name)
                print(f"    --books-from-defaults: no pxa.{t.name}.book/sub in the file; "
                      f"assuming the engine's compiled-in {t.name} tables. This is correct "
                      f"ONLY if the file was built by the stock quantizer with no "
                      f"PXA_{t.name.upper()}_BOOK / PXA_PXQ2_V3 / PXA_PXQ_CEIL_V2 override. "
                      f"Recorded in the checkpoint as quantization_config.assumed_books.",
                      file=sys.stderr)
            if book is None or sub is None:
                raise SystemExit(
                    f"the GGUF carries {t.name} tensors but no pxa.{t.name}.book / "
                    f"pxa.{t.name}.sub. Those KVs are the file's own record of the tables it "
                    f"was quantized with, and PXA_{t.name.upper()}_BOOK / PXA_PXQ2_V3 / "
                    f"PXA_PXQ_CEIL_V2 can override the compiled-in defaults at build time, so "
                    f"this converter will not assume them. If the file came from the stock "
                    f"quantizer with no such override -- which is the case for every PXQ file "
                    f"published before rc3, whose quantizer stamped only the requested tier -- "
                    f"re-run with --books-from-defaults, which substitutes the compiled-in "
                    f"tables, says which tiers it assumed, and stamps the assumption into the "
                    f"output checkpoint.")
            b = np.asarray(book, dtype=np.float32)
            sv = np.asarray(sub, dtype=np.float32)
            TT.check_book(b, t.type_id)
            TT.check_sub(sv)
        books[t.name] = [float(x) for x in b]
        subs[t.name] = sv

    # THE CROSS-CHECK APPLIES ONLY TO THE TIERS THAT SHARE. pxq2/pxq3/pxq4 all index one
    # SUB16 LUT, so two of them disagreeing means two quantizer runs were spliced together.
    # pxq4hq indexes its own SUB8 and is expected to differ; folding it into this check would
    # reject every correct mixed file, which is exactly the bug the check is meant to catch.
    shared = {n: sv for n, sv in subs.items() if not TT.BY_NAME[n].own_sub}
    own = {n: sv for n, sv in subs.items() if TT.BY_NAME[n].own_sub}
    ref_sub = next(iter(shared.values())) if shared else R.SUB
    for name, sv in shared.items():
        if not np.array_equal(sv, ref_sub):
            raise SystemExit(
                f"tier {name} records a different SUB16 LUT from the other tiers in this "
                f"file. The LUT is shared by every code width, so this artifact was spliced "
                f"from two quantizer runs and cannot be served as one checkpoint.")

    ignore = NM.ignore_list(policy, NM.arch_of(kv))
    _q8 = set(NM.q8_list(policy))
    # THE HEAD IS EITHER IGNORED OR SERVED AS int8, AND EXACTLY ONE OF THOSE. The original
    # check demanded it always be ignored, which was right while fp16 was the only option it
    # had: get_quant_method consults ignore first and wins, so an lm_head that is neither
    # ignored nor served would be routed nowhere. Now there is a second right answer, and the
    # invariant is the exclusive-or rather than the constant.
    if "lm_head" in _q8:
        if "lm_head" in ignore:
            raise SystemExit(
                "quantization_config lists lm_head in BOTH q8_modules and ignore. ignore is "
                "checked first and wins, so the engine would build an fp16 head and then look "
                "for an fp16 lm_head.weight this checkpoint does not contain.")
    elif "lm_head" not in ignore:
        raise SystemExit(
            "quantization_config.ignore must contain 'lm_head': pxq4_vllm.config rejects a "
            "checkpoint that lists it in pxq4_modules (UNSERVABLE_PXQ4_LEAF_MODULES), so the "
            "engine builds the head as fp16 and expects an fp16 lm_head.weight in the file. "
            "(embed_tokens needs no entry: it is a VocabParallelEmbedding, never a linear, so "
            "get_quant_method is never asked about it as a LinearBase.)")
    for must in NM.BASE_IGNORE[:2]:
        if must not in ignore:
            raise SystemExit(f"quantization_config.ignore must contain {must!r}: "
                             f"_uses_split_gdn_input_projections (qwen3_5.py:127-157) keys off "
                             f"it, and without it the 48-row b/a fold into in_proj_qkvz and "
                             f"give 12 rows/rank at TP=4 — silently truncated, not an error.")

    # The per-module tier map the runtime resolves with (longest declared suffix wins). Keys
    # are the same suffix form as pxq4_modules; a fused FusedMoE parameter is named by
    # appending ".w13" / ".w2", which is how gate/up and down can differ.
    pxq_tiers: dict[str, str] = {}
    if plan is not None:
        # PER-LAYER TIERS NEED PER-LAYER KEYS. module_suffix collapses the layer index, which
        # is right while every layer serves a module at the same tier and WRONG the moment one
        # does not: Flash-Next is pxq2 gate/up on 44 layers and pxq3 on the last four, and a
        # single "mlp.experts.w13" key would record whichever layer happened to be written
        # last. That is not silent -- pxq4_vllm.moe._place asserts the on-disk slab stride
        # against the parameter it was built for -- but it is a boot that dies for a reason
        # nobody can read. So a key whose members disagree is re-emitted layer-qualified
        # ("layers.44.mlp.experts.w13"), which tier_for resolves by longest-suffix-wins.
        entries: list[tuple[str, str, str]] = []      # (short key, long key, tier)
        for mod, types in sorted(plan.module_types.items()):
            tname = next(iter(types))
            if tname == "dense":
                continue
            base = mod.split("#")[0]
            short = NM.module_suffix(base)
            long_ = NM.layer_qualified_suffix(base)
            if "#" in mod:
                part = "." + mod.split("#")[1]
                short, long_ = short + part, long_ + part
            entries.append((short, long_, tname))
        ambiguous = {k for k in {e[0] for e in entries}
                     if len({t for s_, _, t in entries if s_ == k}) > 1}
        for short, long_, tname in entries:
            pxq_tiers[long_ if short in ambiguous else short] = tname

    multi = len(tiers_present) > 1 or tiers_present != {TT.PXQ4}
    qcfg = {
        "quant_method": "pxq" if multi else "pxq4",
        "pxq4_version": 1,
        "tier": str(kv.get("pxa.pxq6.tier", "core")),
        "type_id": L.TYPE_ID,
        "panel_rows": L.PANEL_ROWS,
        "slab_cols": L.SLAB_COLS,
        "slab_bytes": L.SLAB_BYTES,
        "header_bytes": L.HEADER_BYTES,
        "book": books.get("pxq4", R.BOOK.tolist()),
        "sub": [float(x) for x in ref_sub],
        "backbone_rev": int(kv.get("pxa.pxq.backbone_rev", 0)) or None,
        "backbone_map": kv.get("pxa.pxq.backbone_map"),
        "pxq4_modules": sorted(NM.POLICY_MODULES[policy]),
        "ignore": ignore,
    }
    q8 = NM.q8_list(policy)
    if q8:
        # The runtime keys its ParallelLMHead dispatch off this, and its presence is also what
        # tells a reader that `ignore` deliberately does NOT contain lm_head.
        qcfg["q8_modules"] = q8
        qcfg["q8_format"] = {
            "weights": "int8", "scale": "float16", "scale_axis": "row",
            "scale_rule": "absmax/127", "symmetric": True, "zero_point": False,
            "reconstruction": "w[n,k] = float(q[n,k]) * float(scale[n])",
        }
    if multi:
        qcfg["pxq_tiers"] = pxq_tiers
        qcfg["tier_books"] = books
        qcfg["tier_sub"] = [float(x) for x in ref_sub]
        if own:
            # Per-tier sub tables, for the tiers whose LUT is not the shared one. Recorded
            # separately from ``tier_sub`` rather than replacing it, so a runtime that predates
            # pxq4hq reads a config it understands and simply has no tier to apply it to.
            qcfg["tier_subs"] = {n: [float(x) for x in sv] for n, sv in own.items()}
        # ``slab_bytes``/``type_id`` describe the DEFAULT tier only on a tiered file; the
        # runtime skips those two checks when pxq_tiers is present, and says so.
    if assumed:
        # The checkpoint carries its own caveat. An operator reading config.json six months
        # from now should not have to remember which flag was passed at conversion time.
        qcfg["assumed_books"] = sorted(assumed)
        qcfg["assumed_books_note"] = (
            "--books-from-defaults was used: the GGUF recorded no pxa.<tier>.book/sub for "
            + ", ".join(sorted(assumed)) + ", so the engine's compiled-in v1 tables were "
            "substituted. That is correct for a file built by the stock quantizer and WRONG "
            "for one built with PXA_PXQ*_BOOK / PXA_PXQ2_V3 / PXA_PXQ_CEIL_V2 set. If this "
            "checkpoint decodes to fluent nonsense, this is the first thing to check.")
        qcfg["tier_note"] = ("type_id and slab_bytes describe the pxq4 default tier only; "
                             "this checkpoint holds more than one tier, see pxq_tiers")
    return qcfg


# ---------------------------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------------------------
#: Rows decoded per chunk. token_embd is 248320 x 5120 and its q6_K decoder materialises a
#: float32 [N, K/256, 256] intermediate, which is ~5 GB in one shot — bigger than the tensor.
#: Chunking keeps peak RSS bounded regardless of vocab size.
_DENSE_CHUNK_ROWS = 8192


def _ref_weight_map(ref_hf: str) -> dict[str, str]:
    idx = os.path.join(ref_hf, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    p = os.path.join(ref_hf, "model.safetensors")
    if not os.path.exists(p):
        return {}
    return {k: "model.safetensors" for k in ST.read_header(p) if k != "__metadata__"}


def _ref_tensor_f32(ref_hf: str, key: str, wm: dict[str, str] | None = None
                    ) -> np.ndarray | None:
    """One reference-checkpoint tensor as float32, or None if it is not there.

    BF16 is widened by a shift (``encoder.bf16_to_f32``), so this never rounds and never needs
    torch. Used both for the LM head and for the GDN head-order gate.
    """
    if not ref_hf:
        return None
    from .encoder import bf16_to_f32
    wm = _ref_weight_map(ref_hf) if wm is None else wm
    fname = wm.get(key)
    if fname is None:
        return None
    path = os.path.join(ref_hf, fname)
    if not os.path.exists(path):
        return None
    dtype, shape, raw = ST.read_tensor_bytes(path, key)
    if dtype == "BF16":
        return bf16_to_f32(raw, tuple(shape))
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)
    raise SystemExit(f"{key} in {path} has dtype {dtype}, which this converter cannot read; "
                     f"refusing to guess")


#: Relative half-ULP of each storage dtype a reference checkpoint may use. This is the floor on
#: how well ANY correct converter can reproduce a reference tensor: the reference itself only
#: preserves this many bits, so a residual at this scale is the reference's rounding, not ours.
#:
#:   F32  2^-24 = 5.96e-08     F16  2^-11 = 4.88e-04     BF16 2^-8 = 3.91e-03
#:
#: BF16 is COARSER than F16 despite the wider exponent -- 8 mantissa bits against 11 -- which is
#: why this is a table and not a "16-bit vs 32-bit" branch.
_REF_HALF_ULP: dict[str, float] = {"F32": 2.0 ** -24, "F16": 2.0 ** -11, "BF16": 2.0 ** -8}


def _ref_dtype(ref_hf: str, key: str, wm: dict[str, str]) -> str | None:
    """The on-disk dtype string of one reference tensor, or None if it is not there."""
    fname = wm.get(key)
    if fname is None:
        return None
    path = os.path.join(ref_hf, fname)
    if not os.path.exists(path):
        return None
    hdr = ST.read_header(path)
    ent = hdr.get(key)
    if not isinstance(ent, dict):
        return None
    return ent.get("dtype")


# ---------------------------------------------------------------------------------------------
# THE GDN HEAD-ORDER GATE (the reviewer's G5, promoted to a run-stopping check)
# ---------------------------------------------------------------------------------------------
def gate_gdn_head_order(gg, ref_hf: str, geom, layers: int = 0) -> list[str]:
    """Prove, per GDN layer and against the reference checkpoint, that the gather is the right
    one and is applied in the right direction.

    The two 48-entry vectors are the cheapest possible witnesses and they are EXACT ones, so
    this needs no correlation threshold to argue about and reads ~100 KB for the whole model:

      * ``ssm_dt.bias`` vs ``dt_bias``   — identical values, permuted order. Under the gather
        the two agree bit-for-bit; under identity they do not.
      * ``ssm_a``       vs ``A_log``     — ``log(-ssm_a)`` under the gather. This witnesses the
        value transform at the same time as the order, which is why both live in one gate.

    Any layer where identity fits at least as well as the permutation fails the run: that is
    the signature of a model whose head order we have mis-read.
    """
    problems: list[str] = []
    wm = _ref_weight_map(ref_hf)
    if not wm:
        return ["gdn head-order gate: --ref-hf has no readable safetensors index"]
    gather = np.asarray(NM.v_head_gather(geom), dtype=np.int64)
    checked = 0
    for name in gg.order:
        m = NM._BLK.match(name)
        if not m or m.group(2) != "ssm_dt.bias":
            continue
        layer = int(m.group(1))
        if layers and checked >= layers:
            break
        hf_pref = f"{NM.HF_LM}.layers.{layer}.linear_attn."
        for suffix, hf_key, fn in (("ssm_dt.bias", "dt_bias", lambda x: x),
                                   ("ssm_a", "A_log", None)):
            gname = f"blk.{layer}.{suffix}"
            if gname not in gg.tensors:
                continue
            ti = gg.tensors[gname]
            ours = D.dequant_any(gg.raw(gname), ti.type_id, ti.dims).reshape(-1)
            theirs = _ref_tensor_f32(ref_hf, hf_pref + hf_key, wm)
            ref_dt = _ref_dtype(ref_hf, hf_pref + hf_key, wm)
            if theirs is None:
                problems.append(f"layer {layer}: reference has no {hf_pref + hf_key}")
                continue
            theirs = np.asarray(theirs, dtype=np.float32).reshape(-1)
            if ours.size != geom.n_v_heads or theirs.size != geom.n_v_heads:
                problems.append(f"layer {layer} {suffix}: sizes {ours.size}/{theirs.size}, "
                                f"expected {geom.n_v_heads}")
                continue
            ours_t = fn(ours) if fn is not None else NM.VALUE_TRANSFORMS["ssm_a"][0](ours)
            d_perm = float(np.abs(ours_t[gather] - theirs).max())
            d_ident = float(np.abs(ours_t - theirs).max())
            # The tolerance is set by the REFERENCE's storage precision, not by a constant.
            # The old fixed 1e-5 silently assumed an F32 reference (which the dense 27B twin
            # had, and where log(-ssm_a) reproduced A_log to 5e-7). The qwen35moe references
            # store A_log/dt_bias as F16, whose half-ULP is 2.44e-4 RELATIVE -- 24x the old
            # tolerance -- so a bit-perfect converter fails a fixed 1e-5 on them. Measured on
            # PXA-Coder-35B-v2 layer 0: relative residual 1.64e-4, i.e. UNDER one F16 half-ULP,
            # while the wrong (identity) order sits at 7.64 absolute. The discriminator that
            # actually catches a mis-read head order is ``d_perm < d_ident``, and it has four
            # orders of magnitude of headroom here; the tolerance only guards against both
            # orders being wrong together.
            rel = _REF_HALF_ULP.get(ref_dt or "F32", _REF_HALF_ULP["F32"])
            tol = max(1e-5, 4.0 * rel) * max(1.0, float(np.abs(theirs).max()))
            if not (d_perm < d_ident and d_perm <= tol):
                problems.append(
                    f"layer {layer} {suffix} -> {hf_key}: max|diff| permuted {d_perm:.6g} vs "
                    f"identity {d_ident:.6g} (tol {tol:.2g}, reference dtype {ref_dt}) — the "
                    f"v-head gather does not reproduce the reference checkpoint")
        checked += 1
    if not checked:
        problems.append("gdn head-order gate: no GDN layers found (no blk.*.ssm_dt.bias)")
    return problems


def _lm_head_from_ref(ref_hf: str) -> np.ndarray | None:
    """The AWQ twin's ``lm_head.weight`` as float32, or None if unavailable.

    CURRENTLY UNREACHABLE and deliberately kept: no policy serves ``lm_head`` as PXQ4 (see
    namemap.PXQ4_MODULES_P2B), so ``output.weight`` never reaches ``_pxq4_payload``. This is
    the entry point the P3 head will need once the engine can load one; deleting it would
    lose the "source the head from their BF16, not from our q8_0" decision with it.

    P2b's LM head must come from HERE, not from our own ``output.weight``. Their ``lm_head`` is
    in their 311-entry ignore list and is therefore stored UNQUANTIZED BF16 — encoding it to
    PXQ4 is one quantization step. Encoding our q8_0 copy would be two, and the second one
    would be quantizing an already-quantized grid.
    """
    if not ref_hf:
        return None
    from .encoder import bf16_to_f32
    idx = os.path.join(ref_hf, "model.safetensors.index.json")
    fname = "model.safetensors"
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        if "lm_head.weight" not in wm:
            return None
        fname = wm["lm_head.weight"]
    path = os.path.join(ref_hf, fname)
    if not os.path.exists(path):
        return None
    dtype, shape, raw = ST.read_tensor_bytes(path, "lm_head.weight")
    if dtype == "BF16":
        return bf16_to_f32(raw, tuple(shape))
    if dtype == "F16":
        # The coder35 twin stores its head as F16, not BF16. Still UNQUANTIZED, which is the
        # property that matters: quantizing it is one step, whereas quantizing our own q8_0
        # copy of the same head would be two, the second one quantizing an already-quantized
        # grid.
        return np.frombuffer(raw, dtype=np.float16).reshape(tuple(shape)).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(tuple(shape)).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)
    raise SystemExit(f"lm_head.weight in {path} has dtype {dtype}, which this converter cannot "
                     f"read; refusing to guess")


def _native_pxq4_pair(gg, ggml_name: str, N: int, K: int, perm=None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Split a native PXQ4 tensor and apply the GDN v-head reorder if it has one.

    Still a pure byte move even when ``perm`` is set: a 128-row head block is 2 whole panels
    and a 128-column head block is 4 whole slabs, so the reorder gathers complete addressable
    units and never touches a nibble, a sub-scale or an anchor value. See layout.py.
    """
    ti = gg.tensors[ggml_name]
    slabs, anchor = TT.split_blob(gg.raw(ggml_name), ti.type_id, N, K)
    if perm is not None:
        slabs, anchor = _apply_perm_pxq4(slabs, anchor, perm)
    return slabs, anchor


def _native_pxq4_expert(gg, ggml_name: str, e: int, N: int, K: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Split expert ``e`` out of a 3-D PXQ4 expert stack.

    Experts are the SLOWEST-varying ggml axis and each expert slice is a complete,
    independently addressable PXQ4 tensor (pxq6.cuh:520-526 addresses a panel as
    ``W + (e*panels + p)*panel_bytes``), so this is the 2-D split applied to one expert's
    byte range -- not a gather, not a re-encode, and it touches only that expert's bytes.

    Deliberately NOT ``L.split_blob_3d``: that materialises all E experts at once, and at
    E=256 the largest stack here is 143 MB which the ShardWriter would then pin until its
    next flush. Cutting one expert at a time keeps peak extra memory at one expert.
    """
    tid = gg.tensors[ggml_name].type_id
    per = TT.tensor_bytes(tid, N, K)
    blob = gg.raw(ggml_name)
    if len(blob) % per:
        raise SystemExit(f"{ggml_name}: {len(blob)} B is not a whole multiple of the "
                         f"{per} B per-expert {TT.tier_of(tid).name} size for N={N} K={K}")
    return TT.split_blob(blob[e * per:(e + 1) * per], tid, N, K)


def _pxq4_payload(gg, ggml_name: str, N: int, K: int, enc, ti,
                  ref_hf: str | None = None, perm=None,
                  force_reencode: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    """(slabs, anchor, source-description) for one PXQ4-served module.

    ``force_reencode`` is set only by ``--fuse-uniform-pxq4``, for a native pxq2/pxq3 shard of
    a fused parameter whose other shards cannot be that tier. It decodes the panel and encodes
    it again at pxq4 -- a second lossy pass, which is why nothing sets it implicitly.
    """
    if TT.is_pxq(ti.type_id) and not force_reencode:
        # The whole point of P1: the bytes already on disk ARE the answer. No decode, no
        # re-encode, no value touched — just a partition of the panel into its header and its
        # slabs, plus (for GDN tensors) a gather of whole panels/slabs into HF head order.
        # verify_pxq4_roundtrip proves the partition, and the gather's inverse, are exact.
        _tn = TT.tier_of(ti.type_id).name
        desc = (f"native {_tn} (byte split, panel-permuted)" if perm
                else f"native {_tn} (byte split)")
        return (*_native_pxq4_pair(gg, ggml_name, N, K, perm), desc)

    from .encoder import encode_and_check
    src_desc = (f"RE-ENCODED from native ggml {ti.type} for a uniform fused parameter "
                f"(second quantization pass)" if force_reencode
                else f"re-encoded from ggml {ti.type}")
    w = None
    if ggml_name == "output.weight":
        w = _lm_head_from_ref(ref_hf)
        if w is not None:
            src_desc = "re-encoded from the reference checkpoint's UNQUANTIZED lm_head"
            if w.shape != (N, K):
                raise SystemExit(f"reference lm_head is {w.shape}, expected {(N, K)}")
        else:
            print("    WARNING: no reference lm_head available; falling back to our q8_0 copy, "
                  "which double-quantizes. Prefer --ref-hf.", file=sys.stderr)
    if w is None:
        w = D.dequant_any(gg.raw(ggml_name), ti.type_id, ti.dims).reshape(N, K)

    if perm is not None:
        # Permute BEFORE encoding, not after: a K-axis reorder changes which values share a
        # 32-column sub-scale, so encoding the ggml order and then moving slabs would give a
        # different (and wrong) grouping. Encoding the HF order is the whole point.
        axis, gather = perm
        w = np.take(w, np.asarray(gather, dtype=np.int64), axis=axis)
        src_desc += ", gdn v-head reordered before encode"

    blob, stats = encode_and_check(enc, w, ggml_name)
    print(f"    {ggml_name}: {src_desc}, wrel={stats['wrel']:.4f}", file=sys.stderr)
    return (*TT.split_blob(blob, TT.PXQ4, N, K), src_desc)


def _dense_stream(gg, ggml_name: str, ti, f) -> None:
    """Decode a non-panel-tier tensor to fp16 and write it straight to the output file, one row
    chunk at a time. Peak extra memory is one chunk, not one tensor.

    Each chunk goes through ``dequant_any`` (never a direct ``DECODERS[ti.type_id]`` lookup): a
    type this dispatch cannot decode then raises dequant_any's own clear error instead of a bare
    ``KeyError``, and no future tier can silently bypass the panel-vs-dense dispatch again the
    way PXQ2/PXQ3 did here (gguf_to_vllm bug fixed 2026-09-08, see PXQ23-VLLM-2026-09-08.md).
    """
    K = ti.ne0
    N = 1
    for d in ti.dims[1:]:
        N *= d
    blob = gg.raw(ggml_name)
    rowb = G.row_size(ti.type_id, K)
    for beg in range(0, N, _DENSE_CHUNK_ROWS):
        end = min(beg + _DENSE_CHUNK_ROWS, N)
        chunk = D.dequant_any(blob[beg * rowb:end * rowb], ti.type_id, (K, end - beg))
        f.write(chunk.astype(np.float16).tobytes())


def _dense_streamable(ti, perm=None, xform=None) -> bool:
    """True when the tensor decodes row-by-row from a contiguous byte range.

    A GDN reorder or a value transform disqualifies it: both need the whole tensor in hand
    (a row gather is not a chunk-local operation), and every tensor they apply to is small
    enough that the one-shot decode costs nothing worth a second code path — the largest is
    ``ssm_out`` at 5120 x 6144, a 126 MB float32 intermediate.

    Every PXQ panel tier (PXQ2/PXQ3/PXQ4, ``tiers.is_pxq``) is excluded because its rows are
    interleaved across a 64-row panel, so a row range is not a byte range — chunking it would
    have to be a panel loop, and at the largest real dense-served shape (12288 x 5120) the
    whole-tensor decode is only a 252 MB intermediate. (PXQ2/PXQ3 were missing from this
    exclusion until 2026-09-08 — see PXQ23-VLLM-2026-09-08.md — which let a large PXQ3 tensor
    reach the chunked path and crash on a flat ``DECODERS`` lookup that never had panel tiers in
    it.) F32 is excluded because it is a memcpy already and never large here (the biggest is
    ssm_conv1d at 160 KB).
    """
    if TT.is_pxq(ti.type_id) or ti.type_id == G.GGML_F32:
        return False
    if perm is not None or xform is not None:
        return False
    N = 1
    for d in ti.dims[1:]:
        N *= d
    return N > _DENSE_CHUNK_ROWS


def _dense_payload(gg, ggml_name: str, ti, shape, perm=None, xform=None) -> bytes:
    """Decode any tensor to fp16 bytes, chunked along the slow axis.

    A PXQ4 SOURCE REACHES HERE ROUTINELY, and must not be treated as an error: in P1 the 17
    ``attn_q`` tensors are PXQ4 on disk but their module (``self_attn.qkv_proj``) also holds
    q8_0 k/v, so the §3.1 uniformity invariant forces the whole module to fp16. Decoding PXQ4
    to fp16 is the deliberate cost of deferring that module to P2c — 0.366 GiB/GPU during P1
    only — and is exactly what the ``dense`` branch of the plan asked for.

    The chunked path below cannot be used for any PXQ panel tier (PXQ2/PXQ3/PXQ4): rows are
    interleaved across a 64-row panel, so a row range is not a byte range. Chunking it would
    mean chunking by PANEL, which is a different loop; at the largest real shape (12288 x 5120)
    the whole-tensor decode is a 252 MB float32 intermediate, so it is not worth a second code
    path. PXQ4 is short-circuited above via ``reference.dequant_blob``; PXQ2/PXQ3 fall through
    to the ``N <= _DENSE_CHUNK_ROWS`` whole-tensor branch below because ``_dense_streamable``
    (this function's caller in ``_dense_stream``'s selection) now excludes them from the chunked
    path the same way it excludes PXQ4 -- but this function's own chunked branch is guarded
    identically here too, via ``dequant_any``, in case something ever calls it directly.
    """
    def finish(w):
        """Reorder heads, then transform values, then narrow to fp16 — in that order.

        Order matters: ``VALUE_TRANSFORMS`` are elementwise, so they commute with the gather,
        but doing them in fp32 before the narrowing does not lose the precision the log costs.
        """
        if perm is not None:
            axis, gather = perm
            w = np.take(w, np.asarray(gather, dtype=np.int64), axis=axis)
        if xform is not None:
            w = xform[0](w)
        return w.reshape(shape).astype(np.float16).tobytes()

    if ti.type_id == G.GGML_PXQ4:
        return finish(R.dequant_blob(gg.raw(ggml_name), ti.ne1, ti.ne0))
    K = ti.ne0
    N = 1
    for d in ti.dims[1:]:
        N *= d
    if N <= _DENSE_CHUNK_ROWS or ti.type_id == G.GGML_F32 or perm is not None or xform is not None:
        return finish(D.dequant_any(gg.raw(ggml_name), ti.type_id, ti.dims))
    blob = gg.raw(ggml_name)
    rowb = G.row_size(ti.type_id, K)
    out = bytearray()
    for beg in range(0, N, _DENSE_CHUNK_ROWS):
        end = min(beg + _DENSE_CHUNK_ROWS, N)
        chunk = D.dequant_any(blob[beg * rowb:end * rowb], ti.type_id, (K, end - beg))
        out += chunk.astype(np.float16).tobytes()
    return bytes(out)


def run_convert(args) -> int:
    # --assume-file-size lets the whole planning path run against a truncated header slice of
    # the artifact, so gates G4/G5-prep and every §5.6 structural check execute on a laptop
    # with no GPU access. It is refused for a real conversion: emitting tensor data from a
    # file whose length we had to be told is not something to do quietly.
    if args.assume_file_size:
        if not args.dry_run:
            raise SystemExit("--assume-file-size is only valid with --dry-run")
        gg = G.GGUFHeaderOnly(args.gguf, args.assume_file_size)
    else:
        gg = G.GGUFFile(args.gguf)
    try:
        gg.assert_all_supported()
        hist = gg.type_histogram()
        print(f"gguf: {len(gg.tensors)} tensors, {len(gg.kv)} KVs, "
              f"arch={gg.kv.get('general.architecture')}", file=sys.stderr)
        for t, (c, b) in hist.items():
            print(f"  {t:7s} {c:4d} tensors {b:>14,} B", file=sys.stderr)

        # The .so is loaded lazily, at first use, so that a P2 policy can be PLANNED (and its
        # shard arithmetic and uniformity checked) on a machine where the encoder has not been
        # built yet. It is still required before any byte is written.
        enc = None
        if args.encoder and not args.dry_run:
            from .encoder import NativeEncoder
            enc = NativeEncoder(args.encoder)

        plan = build_plan(gg, args.policy, args.ref_hf, have_encoder=bool(args.encoder),
                          limit_layers=args.limit_layers, with_visual=not args.no_visual,
                          fuse_uniform=args.fuse_uniform_pxq4)
        if args.limit_layers or args.no_visual:
            print("WARNING: --limit-layers / --no-visual produce a NON-SERVABLE checkpoint. "
                  "Smoke test only.", file=sys.stderr)
        qcfg = build_quantization_config(gg, args.policy, plan,
                                        books_from_defaults=args.books_from_defaults)

        print(f"\nplan (policy={args.policy}): {len(plan.emits)} output tensors, "
              f"{plan.total_bytes():,} B", file=sys.stderr)
        kinds = {}
        for e in plan.emits:
            kinds[e.kind] = kinds.get(e.kind, [0, 0])
            kinds[e.kind][0] += 1
            kinds[e.kind][1] += e.nbytes
        for k, (c, b) in sorted(kinds.items()):
            print(f"  {k:6s} {c:5d} tensors {b:>15,} B", file=sys.stderr)
        if plan.reencode:
            print(f"  re-encode: {len(plan.reencode)} ggml tensors", file=sys.stderr)
        reasons: dict[str, int] = {}
        for _, why in plan.skipped:
            reasons[why] = reasons.get(why, 0) + 1
        print(f"  skipped:   {len(plan.skipped)} ggml tensors "
              f"({', '.join(f'{v}x {k}' for k, v in sorted(reasons.items()))})",
              file=sys.stderr)
        if plan.fused_uniform:
            print(f"  --fuse-uniform-pxq4: promoted {len(plan.fused_uniform)} fused "
                  f"parameter(s) to a single pxq4 tier. Every native pxq2/pxq3 shard of these "
                  f"is quantized a SECOND time (panel -> fp32 -> pxq4); the alternative is the "
                  f"whole parameter in fp16. Per-tensor wrel is printed as each one encodes:",
                  file=sys.stderr)
            for mod in plan.fused_uniform:
                print(f"    {mod}", file=sys.stderr)
        if plan.native_dense:
            nd_bytes = sum(b for _, b in plan.native_dense.values())
            print(f"  WARNING: {nd_bytes:,} B ({nd_bytes / 2**30:.2f} GiB) of this plan is "
                  f"native PXQ on disk but served DENSE f16 because policy {args.policy} does "
                  f"not name the module -- check namemap.POLICY_MODULES[{args.policy!r}] "
                  f"against this file's actual module names before assuming the size is "
                  f"unavoidable:", file=sys.stderr)
            for mod, (c, b) in sorted(plan.native_dense.items(), key=lambda kv: -kv[1][1]):
                print(f"    {mod:40s} {c:4d} tensors {b:>14,} B", file=sys.stderr)

        if not (args.limit_layers or args.no_visual):
            print_bandwidth(plan)

        # The GDN head-order gate runs BEFORE anything is written and also on a --dry-run, so
        # the cheapest possible run catches the highest-consequence mistake this converter can
        # make. It needs real tensor data (~100 KB), so it is skipped only when the file itself
        # is a truncated header slice.
        if args.ref_hf and not args.assume_file_size:
            probs = gate_gdn_head_order(gg, args.ref_hf, plan.gdn_geometry,
                                        layers=getattr(args, 'gdn_gate_layers', 0))
            n_gdn = sum(1 for n in gg.order if n.endswith(".ssm_dt.bias"))
            print(f"\nGDN v-head order gate ({getattr(args, 'gdn_gate_layers', 0) or n_gdn} layers, exact vs "
                  f"reference): {'PASS' if not probs else str(len(probs)) + ' PROBLEMS'}",
                  file=sys.stderr)
            for p_ in probs[:10]:
                print("  ", p_, file=sys.stderr)
            if probs:
                raise SystemExit(
                    "the GDN v-head permutation does not reproduce the reference checkpoint. "
                    "Emitting anyway would produce a model that loads, shards and generates "
                    "fluent garbage — the exact failure this gate exists for.")
        elif not args.ref_hf:
            print("\nWARNING: no --ref-hf, so the GDN v-head order gate did NOT run. The "
                  "permutation is applied unverified.", file=sys.stderr)

        if args.ref_hf and not (args.limit_layers or args.no_visual):
            report = keyset_diff(plan, args.ref_hf)
            print_keyset_diff(report)
            if report["unexpected_missing"] or report["unexpected_extra"]:
                if not args.allow_key_diff:
                    raise SystemExit(
                        "key-set diff against the reference checkpoint found differences that "
                        "are not the intended PXQ4/MTP substitutions (see above). Pass "
                        "--allow-key-diff only if you have read every line of that list.")

        if args.dry_run:
            if args.emit_plan:
                with open(args.emit_plan, "w") as f:
                    json.dump({"policy": args.policy,
                               "quantization_config": qcfg,
                               "emits": [e.__dict__ for e in plan.emits],
                               "skipped": plan.skipped}, f, indent=1, default=list)
                print(f"wrote plan to {args.emit_plan}", file=sys.stderr)
            print("dry run: nothing written", file=sys.stderr)
            return 0

        os.makedirs(args.out, exist_ok=True)
        writer = ST.ShardWriter(args.out, int(args.shard_size_gb * (1 << 30)),
                                metadata={"format": "pt", "pxq4_policy": args.policy})

        done_pxq4: set[str] = set()
        done_expert: set[tuple[str, int]] = set()
        _q8_cache: dict = {}
        for e in plan.emits:
            if e.kind == "copy":
                src_file = e.src.split("/", 1)[1]
                path = os.path.join(args.ref_hf, src_file)
                name = e.name
                writer.add(ST.Tensor(name, e.dtype, e.shape,
                                     lambda p=path, n=name: ST.read_tensor_bytes(p, n)[2]))
                # (the vision tensors top out at ~28 MB each, so a plain read is fine)
            elif e.kind == "q8":
                # The head is quantized ONCE and both emits are served from the cache: the
                # two tensors are two views of one quantization and re-running it per emit
                # would be a second, independent quantization of the same weights.
                if _q8_cache.get("src") != e.src:
                    _q8_cache.clear()
                    ti = gg.tensors[e.src]
                    N_, K_ = ti.ne1, ti.ne0
                    head = _lm_head_from_ref(args.head_ref or args.ref_hf)
                    if head is not None:
                        if head.shape != (N_, K_):
                            raise SystemExit(
                                f"reference lm_head is {head.shape}, the GGUF says {(N_, K_)}")
                        src_desc = "the reference checkpoint's UNQUANTIZED head"
                    else:
                        # No unquantized twin: decode ours and say so. This head is then
                        # TWICE quantized, and the README has to carry that.
                        print(f"    {e.src}: no unquantized reference head available; "
                              f"re-quantizing our own {ti.type} copy -- this head is twice "
                              f"quantized and the checkpoint README must say so",
                              file=sys.stderr)
                        head = D.dequant_any(gg.raw(e.src), ti.type_id, ti.dims).reshape(N_, K_)
                        src_desc = f"our own {ti.type} copy (twice quantized)"
                    from .q8head import quantize_head
                    q, sc, st = quantize_head(head)
                    del head
                    print(f"    {e.src}: q8 head from {src_desc}, wrel={st['wrel']:.5f}, "
                          f"{st['zero_rows']} zero rows, {st['bpw']:.4f} bpw", file=sys.stderr)
                    _q8_cache.update(src=e.src, q=q, scale=sc, stats=st)
                if e.name.endswith(".weight_scale"):
                    writer.add(ST.Tensor.from_numpy(e.name, _q8_cache["scale"]))
                else:
                    writer.add(ST.Tensor.from_numpy(e.name, _q8_cache["q"]))
            elif e.kind == "concat":
                # One output tensor from N ggml sources, concatenated on the OUTPUT axis in
                # part order. Decoded whole rather than streamed: the largest of these is the
                # indexer's 640 x 2560, a 6 MB fp16 tensor.
                def _concat_bytes(srcs=e.src, shape=e.shape):
                    parts = []
                    for s_ in srcs.split(","):
                        t_ = gg.tensors[s_]
                        parts.append(D.dequant_any(gg.raw(s_), t_.type_id, t_.dims)
                                     .reshape(t_.ne1, t_.ne0))
                    return np.concatenate(parts, axis=0).reshape(shape).astype(
                        np.float16).tobytes()
                writer.add(ST.Tensor(e.name, "F16", e.shape, _concat_bytes))
            elif e.kind == "dense":
                ti = gg.tensors[e.src]
                pm = gdn_perm_for(e.src, ti, plan.gdn_geometry)
                xf = NM.value_transform(NM.ggml_suffix(e.src), gg.kv)
                if _dense_streamable(ti, pm, xf):
                    writer.add(ST.Tensor(e.name, "F16", e.shape,
                                         lambda f, t=ti: _dense_stream(gg, t.name, t, f),
                                         streaming=True))
                else:
                    writer.add(ST.Tensor(e.name, "F16", e.shape,
                                         lambda t=ti, s=e.shape, p=pm, x=xf:
                                         _dense_payload(gg, t.name, t, s, p, x)))
            elif e.expert >= 0:
                # One expert's slice of a 3-D stack. Both emits of a (slabs, anchor) pair carry
                # the same (src, expert), so dedupe on the pair, not on src alone.
                key = (e.src, e.expert)
                if key in done_expert:
                    continue
                done_expert.add(key)
                ti = gg.tensors[e.src]
                N, K = ti.ne1, ti.ne0
                hf_t = _hf_of(e.src, gg.kv)
                sl_name, an_name = _pxq4_emit_names(hf_t.format(e=e.expert))
                if args.verify and e.expert == 0:
                    # Verifying all 256 experts of all 40 layers would re-read the whole file
                    # ~10x for no new information: the split is the same code on every expert
                    # and differs only in the byte offset. Expert 0 of every stack proves the
                    # geometry; a mis-sliced expert 7 would be an offset bug, which
                    # _expert_offsets_gate below checks directly and cheaply.
                    sl0, an0 = _native_pxq4_expert(gg, e.src, 0, N, K)
                    _verify_expert_slice(e.src, gg, ti, sl0, an0, 0, N, K)
                P_, S_ = N // 64, K // 32
                writer.add(ST.Tensor(
                    sl_name, "U8", (P_, S_, TT.slab_bytes(ti.type_id)),
                    lambda src=e.src, x=e.expert, n=N, k=K:
                    _native_pxq4_expert(gg, src, x, n, k)[0].tobytes()))
                writer.add(ST.Tensor(
                    an_name, "F16", (P_, 64),
                    lambda src=e.src, x=e.expert, n=N, k=K:
                    _native_pxq4_expert(gg, src, x, n, k)[1].tobytes()))
            else:
                if e.src in done_pxq4:
                    continue
                done_pxq4.add(e.src)
                ti = gg.tensors[e.src]
                N, K = ti.ne1, ti.ne0
                pm = gdn_perm_for(e.src, ti, plan.gdn_geometry)
                sl_name, an_name = _pxq4_emit_names(_hf_of(e.src, gg.kv))
                # The PLAN decides, not the source type: a native panel tensor whose emit
                # carries a DIFFERENT tier was promoted by --fuse-uniform-pxq4 and has to be
                # re-encoded like any non-panel source. Branching on ti.type_id alone would
                # silently byte-move it and put two strides in one parameter.
                if TT.is_pxq(ti.type_id) and e.tier == ti.type_id:
                    # Native panel tier: verify the split here, then let the WRITER re-do it
                    # lazily.
                    # The ShardWriter batches up to --shard-size-gb of tensors before flushing,
                    # so holding each panel blob as materialised bytes until the flush would
                    # pin a whole shard in RAM; a closure over the mmap pins nothing.
                    slabs, anchor = _native_pxq4_pair(gg, e.src, N, K, pm)
                    if args.verify:
                        verify_pxq4_roundtrip(e.src, gg, ti, slabs, anchor, pm)
                    P, S = N // 64, K // 32
                    sb = TT.slab_bytes(ti.type_id)
                    del slabs, anchor
                    writer.add(ST.Tensor(
                        sl_name, "U8", (P, S, sb),
                        lambda src=e.src, n=N, k=K, p=pm:
                        _native_pxq4_pair(gg, src, n, k, p)[0].tobytes()))
                    writer.add(ST.Tensor(
                        an_name, "F16", (P, 64),
                        lambda src=e.src, n=N, k=K, p=pm:
                        _native_pxq4_pair(gg, src, n, k, p)[1].tobytes()))
                else:
                    # Re-encoded: the encode is expensive and non-deterministic to repeat
                    # lazily inside a writer callback, so it is done once, here.
                    slabs, anchor, _desc = _pxq4_payload(
                        gg, e.src, N, K, enc, ti, args.ref_hf, pm,
                        force_reencode=TT.is_pxq(ti.type_id))
                    writer.add(ST.Tensor.from_numpy(sl_name, slabs))
                    writer.add(ST.Tensor.from_numpy(an_name, anchor))

        index = writer.finish()
        print(f"wrote {len(set(index['weight_map'].values()))} shard(s), "
              f"{index['metadata']['total_size']:,} B", file=sys.stderr)

        write_config(args.out, args.ref_hf, qcfg, gg=gg)
        return 0
    finally:
        gg.close()


def _verify_expert_slice(ggml_name: str, gg, ti, slabs: np.ndarray, anchor: np.ndarray,
                         e: int, N: int, K: int) -> None:
    """The expert-stack twin of ``verify_pxq4_roundtrip``: rejoining expert ``e``'s split must
    reproduce exactly that expert's byte range from the file. Proves the split AND the offset.
    """
    per = TT.tensor_bytes(ti.type_id, N, K)
    want = np.frombuffer(gg.raw(ggml_name), dtype=np.uint8)[e * per:(e + 1) * per]
    got = np.frombuffer(TT.join_blob(slabs, anchor), dtype=np.uint8)
    if got.size != want.size or not np.array_equal(got, want):
        raise SystemExit(
            f"{ggml_name} expert {e}: the {TT.tier_of(ti.type_id).name} split does not rejoin "
            f"to the original bytes. "
            f"An expert-stack offset or panel-geometry error -- refusing to write.")


def verify_pxq4_roundtrip(ggml_name: str, gg, ti, slabs: np.ndarray,
                          anchor: np.ndarray, perm=None) -> None:
    """Plan §5.6 check 2. Only meaningful for a NATIVE pxq4 source: the split must rejoin to
    the original bytes exactly. A re-encoded tensor has no original bytes to compare against —
    ``encoder.encode_and_check`` covers that case instead.

    For a GDN tensor the emitted panels are in HF head order, so the reorder is undone first.
    That keeps this a BYTE comparison — it now proves two things at once: the split is a
    partition of the file, and the head reorder is a lossless permutation of whole panels
    rather than a slice that dropped or duplicated one."""
    if not TT.is_pxq(ti.type_id):
        return
    if perm is not None:
        slabs, anchor = _unapply_perm_pxq4(slabs, anchor, perm)
    if TT.join_blob(slabs, anchor) != bytes(gg.raw(ggml_name)):
        raise SystemExit(f"{ggml_name}: split -> join did not reproduce the original bytes. "
                         f"The panel arithmetic is wrong; nothing downstream can be trusted.")


def write_config(out_dir: str, ref_hf: str | None, qcfg: dict, gg=None) -> None:
    if not ref_hf:
        cfg: dict = {"quantization_config": qcfg}
        # An arch with no reference checkpoint on the box has to AUTHOR its config, and the
        # only honest source is the GGUF's own KVs. config_qwen4exp.emit traces every field to
        # a source key and the fork's own config class validates the result.
        if gg is not None and NM.arch_of(gg.kv) == NM.Q4.ARCH:
            from .config_qwen4exp import emit as _emit_q4
            built, missing = _emit_q4(gg.kv, ple_fp8_offload=True,
                                      ple_ggml_layer=NM.Q4.ple_ggml_layer(gg.order))
            if missing:
                print(f"    config: {len(missing)} expected KVs absent: {missing}",
                      file=sys.stderr)
            built["quantization_config"] = qcfg
            cfg = built
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        return
    for fn in COPY_FILES:
        src = os.path.join(ref_hf, fn)
        if os.path.exists(src) and fn != "config.json":
            shutil.copy2(src, os.path.join(out_dir, fn))
    with open(os.path.join(ref_hf, "config.json")) as f:
        cfg = json.load(f)
    # ONLY quantization_config is rewritten. Everything else — architectures, text_config,
    # vision_config, rope, the head counts — stays byte-identical to the config the incumbent
    # is serving from right now.
    cfg["quantization_config"] = qcfg
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


# ---------------------------------------------------------------------------------------------
# decode-bandwidth accounting, computed from the ACTUAL emission plan
# ---------------------------------------------------------------------------------------------
#: Suffixes that are TP-sharded linears. Everything else in a layer (norms, A_log, dt_bias,
#: conv1d) is either replicated or negligible; the whole non-linear remainder is under 0.1% of
#: a layer's bytes, so it is counted as replicated rather than modelled precisely.
_SHARDED = ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d",
            "lm_head")


def bandwidth_report(plan: Plan, tp: int) -> dict:
    """Weight bytes read per GPU per decoded token, from the emitted tensor list.

    This is an independent recomputation of the number the whole project's economics rest on,
    from the artifact we actually produce rather than from a spreadsheet. It is NOT a
    throughput measurement and must never be quoted as tok/s: turning it into a rate requires
    assuming our kernels sustain the same effective HBM bandwidth as the incumbent's
    tensor-core GEMM, which is exactly the assumption the plan flags as optimistic.

    COUNTED:     every language-model layer weight, plus lm_head (read in full every step).
    NOT COUNTED: embed_tokens (a gather of one row, not a read of the table), the vision tower
                 (not in the decode path), MTP (not emitted), and the KV cache (unchanged by
                 this project).
    """
    per_gpu = 0
    detail: dict[str, int] = {}
    for e in plan.emits:
        if e.kind == "copy":
            continue
        n = e.name
        if "embed_tokens" in n:
            continue
        if not (".layers." in n or n.startswith("lm_head")):
            continue
        sharded = any(s in n for s in _SHARDED)
        b = e.nbytes // tp if sharded else e.nbytes
        per_gpu += b
        cls = n.split(".")[-2] if "." in n else n
        detail[cls] = detail.get(cls, 0) + b
    return {"tp": tp, "bytes_per_gpu": per_gpu,
            "gib_per_gpu": per_gpu / (1 << 30),
            "by_class": dict(sorted(detail.items(), key=lambda kv: -kv[1]))}


def print_bandwidth(plan: Plan, tps=(2, 4)) -> None:
    print("\ndecode weight bytes read per GPU per token (PROJECTION INPUT, not a measurement):",
          file=sys.stderr)
    for tp in tps:
        r = bandwidth_report(plan, tp)
        print(f"  TP={tp}: {r['gib_per_gpu']:.3f} GiB/GPU  ({r['bytes_per_gpu']:,} B)",
              file=sys.stderr)
        for cls, b in list(r["by_class"].items())[:8]:
            print(f"       {cls:24s} {b / (1 << 30):7.3f} GiB", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# gate G4 — key-set diff against the reference checkpoint
# ---------------------------------------------------------------------------------------------
_AWQ_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_zero_point", ".weight_shape",
                 ".weight_g_idx")


def _collapse_awq(name: str) -> str:
    for s in _AWQ_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)] + ".weight"
    return name


#: ``...experts.7.gate_proj.weight`` -> ``(...experts, gate)``.
_EXPERT_KEY = re.compile(r"^(.*\.experts)\.\d+\.(gate|up|down)_proj\.weight$")


def _collapse_experts(name: str) -> str:
    """Fold our per-expert key back onto the reference's stacked spelling.

    The reference checkpoint stores each layer's experts as TWO stacked tensors,
    ``experts.gate_up_proj`` [E, 2I, H] and ``experts.down_proj`` [E, I, H] (verified: 7 keys
    under ``layers.0.mlp``). We deliberately emit E*3 separate per-expert tensors instead --
    see the ``_MOE_EXPERT_MAP`` note in namemap.py: the per-expert spelling is the one
    ``FusedMoE.make_expert_params_mapping`` rewrites by a pure ``name.replace``, whereas the
    stacked spelling takes vLLM's ``is_fused_expert`` branch whose ``chunk(2, dim=-2)`` would
    cut a panel-major PXQ4 slab array along its K-slab axis and silently corrupt every expert.

    So the two key sets differ BY DESIGN, and this collapse is what lets the gate still check
    the thing it exists to check -- that no expert is missing, duplicated or misnamed -- rather
    than being switched off wholesale with --allow-key-diff.
    """
    m = _EXPERT_KEY.match(name)
    if not m:
        return name
    stem, which = m.group(1), m.group(2)
    return f"{stem}.down_proj" if which == "down" else f"{stem}.gate_up_proj"


def keyset_diff(plan: Plan, ref_hf: str) -> dict:
    """Compare our emitted key set to the reference checkpoint's, collapsing AWQ's four-tensor
    encoding to a single ``.weight`` and our two-tensor PXQ4 encoding likewise.

    This is the gate that catches a name-mapping mistake as a *set* difference rather than as a
    mysterious KeyError at load. It cannot catch a mapping that is wrong but well-formed (e.g.
    in_proj_b/in_proj_a swapped) — that is G5's job, and those three cases are flagged as
    ASSUMPTIONs in namemap.py.
    """
    idx_path = os.path.join(ref_hf, "model.safetensors.index.json")
    ref_keys: set[str] = set()
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            ref_keys = set(json.load(f)["weight_map"])
    else:
        hdr = ST.read_header(os.path.join(ref_hf, "model.safetensors"))
        ref_keys = {k for k in hdr if k != "__metadata__"}

    ref_logical = {_collapse_awq(k) for k in ref_keys}
    ref_logical = {k for k in ref_logical if not k.startswith("mtp.")}

    ours: set[str] = set()
    n_expert_emits = 0
    for e in plan.emits:
        if e.name.endswith(".pxq4_slabs"):
            logical = e.name[: -len(".pxq4_slabs")] + ".weight"
        elif e.name.endswith(".pxq4_anchor"):
            continue
        else:
            logical = e.name
        collapsed = _collapse_experts(logical)
        if collapsed is not logical and collapsed != logical:
            n_expert_emits += 1
        ours.add(collapsed)

    missing = sorted(ref_logical - ours)
    extra = sorted(ours - ref_logical)
    if n_expert_emits:
        print(f"  (collapsed {n_expert_emits} per-expert emits onto the reference's stacked "
              f"experts.gate_up_proj / experts.down_proj spelling)", file=sys.stderr)
    return {
        "n_ref": len(ref_logical), "n_ours": len(ours),
        "missing": missing, "extra": extra,
        # Nothing is expected to be missing or extra once MTP is excluded on both sides: the
        # PXQ4 substitution is name-preserving under the collapse above.
        "unexpected_missing": missing, "unexpected_extra": extra,
    }


def print_keyset_diff(rep: dict) -> None:
    print(f"\nkey-set vs reference: ours={rep['n_ours']} ref={rep['n_ref']} "
          f"missing={len(rep['missing'])} extra={len(rep['extra'])}", file=sys.stderr)
    for k in rep["missing"][:40]:
        print(f"  MISSING {k}", file=sys.stderr)
    if len(rep["missing"]) > 40:
        print(f"  ... and {len(rep['missing']) - 40} more", file=sys.stderr)
    for k in rep["extra"][:40]:
        print(f"  EXTRA   {k}", file=sys.stderr)
    if len(rep["extra"]) > 40:
        print(f"  ... and {len(rep['extra']) - 40} more", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gguf_to_vllm.convert",
                                 description="PXQ4 GGUF -> vLLM safetensors converter")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref-hf", default=None,
                    help="AWQ twin's model dir: source of config/tokenizer/vision tower and "
                         "the key-set diff. Effectively mandatory for a servable output.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--policy", default="p1", choices=sorted(NM.POLICY_MODULES))
    ap.add_argument("--head-ref", default=None,
                    help="checkpoint dir holding an UNQUANTIZED lm_head.weight, used only for "
                         "the q8 head. Defaults to --ref-hf. It is a separate flag because the "
                         "dir that has the right config, tokenizer and vision tower is not "
                         "always the one that has an unquantized head, and quantizing an "
                         "already-quantized head is two steps where one will do.")
    ap.add_argument("--encoder", default=None,
                    help="path to pxq4_encode.so; required by the p2 policies")
    ap.add_argument("--books-from-defaults", action="store_true",
                    help="when the GGUF records no pxa.<tier>.book / pxa.<tier>.sub for a "
                         "tier it contains, substitute the engine's compiled-in v1 tables "
                         "instead of refusing. Every PXQ file published before rc3 needs this "
                         "-- that quantizer stamped only the requested tier's tables. Correct "
                         "ONLY for a file from the stock quantizer with no PXA_PXQ*_BOOK / "
                         "PXA_PXQ2_V3 / PXA_PXQ_CEIL_V2 override; the run names every tier it "
                         "assumed and the assumption is stamped into the output config.json.")
    ap.add_argument("--fuse-uniform-pxq4", action="store_true",
                    help="re-encode a native pxq2/pxq3 shard of a FUSED vLLM parameter up to "
                         "pxq4 when its sibling shards cannot be that tier, so the parameter "
                         "is servable instead of falling back to fp16. Costs a second "
                         "quantization pass on the tensors it promotes (each one's wrel is "
                         "printed); off by default. Needs --encoder.")

    ap.add_argument("--shard-size-gb", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and run every structural check without reading tensor data")
    ap.add_argument("--emit-plan", default=None, help="write the plan as JSON")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="round-trip every native PXQ4 tensor (default on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--allow-key-diff", action="store_true")
    ap.add_argument("--gdn-gate-layers", type=int, default=0,
                    help="check the GDN v-head order on only the first N GDN layers "
                         "(0 = all of them; the gate reads ~100 KB total, so 0 is the "
                         "right answer unless you are debugging)")
    ap.add_argument("--assume-file-size", type=int, default=0,
                    help="dry-run only: treat a truncated header slice as a file of this size")
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="SMOKE TEST ONLY: emit only ggml blocks [0, N). Not servable.")
    ap.add_argument("--no-visual", action="store_true",
                    help="SMOKE TEST ONLY: skip the vision tower copy. Not servable.")
    args = ap.parse_args(argv)
    # Before anything reads the 23 GB artifact: refuse a policy that cannot produce a
    # loadable checkpoint (namemap.BLOCKED_POLICIES).
    NM.assert_policy_supported(args.policy)
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")
    return run_convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
