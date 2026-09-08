"""convert.py — offline GGUF -> vLLM-loadable safetensors converter for a PXQ4 checkpoint.

Plan §5. Pure Python + numpy: no torch, no CUDA, no vLLM, no GPU, no lease. Everything except
the final byte-writing is exercised by ``--dry-run``, which plans the entire conversion from
the GGUF header alone and runs every structural self-check.

    python -m gguf_to_vllm.convert \
      --gguf   /path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf \
      --ref-hf /path/to/hf/<reference-hf-model> \
      --out    /path/to/models/pxa-models/Qwen3.8-27B-PXQ4-vllm \
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

THE INVARIANT THIS FILE ENFORCES (plan §3.1). Every vLLM linear module served by
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
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import dequant_ref as D
from . import gguf_raw as G
from . import layout as L
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


@dataclass
class Plan:
    emits: list[Emit] = field(default_factory=list)
    module_types: dict[str, set[str]] = field(default_factory=dict)
    reencode: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    gdn_geometry: Any = None

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


def build_plan(gg: G.GGUFFile | G.GGUFHeaderOnly, policy: str, ref_hf: str | None,
               have_encoder: bool, limit_layers: int = 0, with_visual: bool = True) -> Plan:
    """``limit_layers`` and ``with_visual`` exist for smoke tests only: they produce a
    checkpoint that is NOT servable, and the caller is expected to say so. They let the whole
    emission path — decode, split, shard-writing, config — run against real bytes in a minute
    instead of against 23 GB in an hour."""
    plan = Plan()
    kv = gg.kv
    # The GDN v-head order differs between the two checkpoints (namemap module docstring), so
    # the geometry is needed before a single tensor is planned. Derived from the file's KVs and
    # cross-checked against the shapes the file actually has: permuting head blocks on guessed
    # geometry would be worse than not permuting at all.
    geom = NM.gdn_geometry(kv)
    geom.check_against_tensors(_gdn_shapes(gg))
    plan.gdn_geometry = geom

    for name in gg.order:
        if limit_layers:
            m = NM._BLK.match(name)
            if m and int(m.group(1)) >= limit_layers:
                plan.skipped.append((name, f"--limit-layers {limit_layers}"))
                continue
        ti = gg.tensors[name]
        hf = _hf_of(name, kv)
        if hf is None:
            plan.skipped.append((name, "MTP / not mapped in P1-P2 (plan §3, P3 work)"))
            continue

        module = NM.HF_MODULE_OF(hf)
        want_pxq4 = NM.is_pxq4_module(module, policy)
        perm = gdn_perm_for(name, ti, geom)
        perm_note = ""
        if perm is not None:
            perm_note = (f"gdn v-head reorder on axis {perm[0]} "
                         f"({geom.n_v_heads} heads, {geom.repeats}x{geom.n_k_heads})")
        xform = NM.VALUE_TRANSFORMS.get(NM.ggml_suffix(name))

        if want_pxq4:
            N, K = ti.ne1, ti.ne0
            L.assert_geometry(N, K)
            slab_name, anch_name = _pxq4_emit_names(hf)
            if xform is not None:
                raise SystemExit(
                    f"{name} needs the value transform {xform[1]!r}, which cannot be expressed "
                    f"as a byte move — it must not be served as PXQ4.")
            if ti.type_id == G.GGML_PXQ4:
                note = ("native pxq4, panel-permuted byte move" if perm
                        else "native pxq4, pure byte split")
            else:
                note = f"RE-ENCODE {ti.type} -> pxq4"
                plan.reencode.append(name)
                if not have_encoder:
                    raise SystemExit(
                        f"policy {policy} requires re-encoding {name} ({ti.type} -> pxq4) but "
                        f"no --encoder was given. Refusing to silently fall back to fp16: that "
                        f"would produce a checkpoint that loads, runs, and is quietly slower "
                        f"than the policy claims.")
            P, S = N // 64, K // 32
            plan.emits.append(Emit(slab_name, "pxq4", "U8", (P, S, 1088), P * S * 1088,
                                   name, note, perm_note))
            plan.emits.append(Emit(anch_name, "pxq4", "F16", (P, 64), P * 64 * 2, name, note,
                                   perm_note))
            plan.module_types.setdefault(module, set()).add("pxq4")
        else:
            shape = ti.logical_shape
            if hf.endswith("conv1d.weight"):
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

    if ref_hf and with_visual:
        for e in _plan_visual(ref_hf):
            plan.emits.append(e)

    _check_plan(plan, policy)
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


def _check_plan(plan: Plan, policy: str) -> None:
    """Plan §5.6 checks 1, 5 and 6. All of these fail the run; none of them warn."""
    # (6) §3.1 uniformity: a module must be entirely pxq4 or entirely dense.
    mixed = {m: sorted(t) for m, t in plan.module_types.items() if len(t) > 1}
    if mixed:
        raise SystemExit(
            f"policy {policy} violates the §3.1 uniformity invariant — these vLLM modules "
            f"would be part-PXQ4 and part-fp16, which needs a custom per-shard weight loader "
            f"and is exactly the silent mis-shard we refuse to write: {mixed}")

    # (5) shard arithmetic at every TP degree we intend to serve.
    for e in plan.emits:
        if e.kind != "pxq4" or not e.name.endswith(".pxq4_slabs"):
            continue
        P, S, _ = e.shape
        N, K = P * 64, S * 32
        row_parallel = any(e.name.endswith(s + ".pxq4_slabs")
                           for s in ("down_proj", "o_proj", "out_proj"))
        L.assert_shardable(N, K, (1, 2, 4), row_parallel=row_parallel, name=e.name)

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
def build_quantization_config(gg, policy: str) -> dict:
    kv = gg.kv
    book = kv.get("pxa.pxq6.book")
    sub = kv.get("pxa.pxq6.sub")
    if book is None or sub is None:
        raise SystemExit(
            "the GGUF does not carry pxa.pxq6.book / pxa.pxq6.sub. Those KVs are the file's "
            "own record of the tables it was quantized with, and PXA_PXQ6_BOOK / PXA_PXQ6_SUB "
            "can override the compiled-in defaults at build time — so we will not assume.")

    # Plan §5.6 check 3: the file's tables must equal the ones our reference and the vendored
    # CUDA header use. If they ever differ, every weight in the file decodes wrong.
    R.check_tables(np.asarray(book, dtype=np.float32), np.asarray(sub, dtype=np.float32))
    if not np.array_equal(np.asarray(book, dtype=np.float32), R.BOOK):
        raise SystemExit("pxa.pxq6.book in the file differs from ggml-pxq6-tables.h "
                         "PXQ6_BOOK_INIT — this file was built with a table override.")
    if not np.array_equal(np.asarray(sub, dtype=np.float32), R.SUB):
        raise SystemExit("pxa.pxq6.sub in the file differs from ggml-pxq6-tables.h "
                         "PXQ6_SUB16_INIT — this file was built with a table override.")

    NM.assert_policy_supported(policy)
    ignore = NM.ignore_list(policy)
    if "lm_head" not in ignore:
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

    return {
        "quant_method": "pxq4",
        "pxq4_version": 1,
        "tier": str(kv.get("pxa.pxq6.tier", "core")),
        "type_id": L.TYPE_ID,
        "panel_rows": L.PANEL_ROWS,
        "slab_cols": L.SLAB_COLS,
        "slab_bytes": L.SLAB_BYTES,
        "header_bytes": L.HEADER_BYTES,
        "book": [float(x) for x in book],
        "sub": [float(x) for x in sub],
        "backbone_rev": int(kv.get("pxa.pxq.backbone_rev", 0)) or None,
        "backbone_map": kv.get("pxa.pxq.backbone_map"),
        "pxq4_modules": sorted(NM.POLICY_MODULES[policy]),
        "ignore": ignore,
    }


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
            tol = 1e-5 * max(1.0, float(np.abs(theirs).max()))
            if not (d_perm < d_ident and d_perm <= tol):
                problems.append(
                    f"layer {layer} {suffix} -> {hf_key}: max|diff| permuted {d_perm:.6g} vs "
                    f"identity {d_ident:.6g} (tol {tol:.2g}) — the v-head gather does not "
                    f"reproduce the reference checkpoint")
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
    slabs, anchor = L.split_blob(gg.raw(ggml_name), N, K)
    if perm is not None:
        slabs, anchor = _apply_perm_pxq4(slabs, anchor, perm)
    return slabs, anchor


def _pxq4_payload(gg, ggml_name: str, N: int, K: int, enc, ti,
                  ref_hf: str | None = None, perm=None) -> tuple[np.ndarray, np.ndarray, str]:
    """(slabs, anchor, source-description) for one PXQ4-served module."""
    if ti.type_id == G.GGML_PXQ4:
        # The whole point of P1: the bytes already on disk ARE the answer. No decode, no
        # re-encode, no value touched — just a partition of the panel into its header and its
        # slabs, plus (for GDN tensors) a gather of whole panels/slabs into HF head order.
        # verify_pxq4_roundtrip proves the partition, and the gather's inverse, are exact.
        desc = "native pxq4 (byte split, panel-permuted)" if perm else "native pxq4 (byte split)"
        return (*_native_pxq4_pair(gg, ggml_name, N, K, perm), desc)

    from .encoder import encode_and_check
    src_desc = f"re-encoded from ggml {ti.type}"
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
    return (*L.split_blob(blob, N, K), src_desc)


def _dense_stream(gg, ggml_name: str, ti, f) -> None:
    """Decode a non-PXQ4 tensor to fp16 and write it straight to the output file, one row
    chunk at a time. Peak extra memory is one chunk, not one tensor."""
    K = ti.ne0
    N = 1
    for d in ti.dims[1:]:
        N *= d
    blob = gg.raw(ggml_name)
    rowb = G.row_size(ti.type_id, K)
    fn = D.DECODERS[ti.type_id]
    for beg in range(0, N, _DENSE_CHUNK_ROWS):
        end = min(beg + _DENSE_CHUNK_ROWS, N)
        f.write(fn(blob[beg * rowb:end * rowb], end - beg, K).astype(np.float16).tobytes())


def _dense_streamable(ti, perm=None, xform=None) -> bool:
    """True when the tensor decodes row-by-row from a contiguous byte range.

    A GDN reorder or a value transform disqualifies it: both need the whole tensor in hand
    (a row gather is not a chunk-local operation), and every tensor they apply to is small
    enough that the one-shot decode costs nothing worth a second code path — the largest is
    ``ssm_out`` at 5120 x 6144, a 126 MB float32 intermediate.

    PXQ4 is excluded because its rows are interleaved across a 64-row panel, so a row range is
    not a byte range — chunking it would have to be a panel loop, and at the largest real
    dense-served shape (12288 x 5120) the whole-tensor decode is only a 252 MB intermediate.
    F32 is excluded because it is a memcpy already and never large here (the biggest is
    ssm_conv1d at 160 KB).
    """
    if ti.type_id in (G.GGML_PXQ4, G.GGML_F32):
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

    The chunked path below cannot be used for PXQ4: rows are interleaved across a 64-row panel,
    so a row range is not a byte range. Chunking it would mean chunking by PANEL, which is a
    different loop; at the largest real shape (12288 x 5120) the whole-tensor decode is a
    252 MB float32 intermediate, so it is not worth a second code path.
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
    fn = D.DECODERS[ti.type_id]
    for beg in range(0, N, _DENSE_CHUNK_ROWS):
        end = min(beg + _DENSE_CHUNK_ROWS, N)
        chunk = fn(blob[beg * rowb:end * rowb], end - beg, K)
        out += chunk.astype(np.float16).tobytes()
    return bytes(out)


def run_convert(args) -> int:
    # --assume-file-size lets the whole planning path run against a truncated header slice of
    # the artifact, so gates G4/G5-prep and every §5.6 structural check execute on a laptop
    # with no DGX access. It is refused for a real conversion: emitting tensor data from a
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
                          limit_layers=args.limit_layers, with_visual=not args.no_visual)
        if args.limit_layers or args.no_visual:
            print("WARNING: --limit-layers / --no-visual produce a NON-SERVABLE checkpoint. "
                  "Smoke test only.", file=sys.stderr)
        qcfg = build_quantization_config(gg, args.policy)

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
        for e in plan.emits:
            if e.kind == "copy":
                src_file = e.src.split("/", 1)[1]
                path = os.path.join(args.ref_hf, src_file)
                name = e.name
                writer.add(ST.Tensor(name, e.dtype, e.shape,
                                     lambda p=path, n=name: ST.read_tensor_bytes(p, n)[2]))
                # (the vision tensors top out at ~28 MB each, so a plain read is fine)
            elif e.kind == "dense":
                ti = gg.tensors[e.src]
                pm = gdn_perm_for(e.src, ti, plan.gdn_geometry)
                xf = NM.VALUE_TRANSFORMS.get(NM.ggml_suffix(e.src))
                if _dense_streamable(ti, pm, xf):
                    writer.add(ST.Tensor(e.name, "F16", e.shape,
                                         lambda f, t=ti: _dense_stream(gg, t.name, t, f),
                                         streaming=True))
                else:
                    writer.add(ST.Tensor(e.name, "F16", e.shape,
                                         lambda t=ti, s=e.shape, p=pm, x=xf:
                                         _dense_payload(gg, t.name, t, s, p, x)))
            else:
                if e.src in done_pxq4:
                    continue
                done_pxq4.add(e.src)
                ti = gg.tensors[e.src]
                N, K = ti.ne1, ti.ne0
                pm = gdn_perm_for(e.src, ti, plan.gdn_geometry)
                sl_name, an_name = _pxq4_emit_names(_hf_of(e.src, gg.kv))
                if ti.type_id == G.GGML_PXQ4:
                    # Native PXQ4: verify the split here, then let the WRITER re-do it lazily.
                    # The ShardWriter batches up to --shard-size-gb of tensors before flushing,
                    # so holding each panel blob as materialised bytes until the flush would
                    # pin a whole shard in RAM; a closure over the mmap pins nothing.
                    slabs, anchor = _native_pxq4_pair(gg, e.src, N, K, pm)
                    if args.verify:
                        verify_pxq4_roundtrip(e.src, gg, ti, slabs, anchor, pm)
                    P, S = N // 64, K // 32
                    del slabs, anchor
                    writer.add(ST.Tensor(
                        sl_name, "U8", (P, S, 1088),
                        lambda src=e.src, n=N, k=K, p=pm:
                        _native_pxq4_pair(gg, src, n, k, p)[0].tobytes()))
                    writer.add(ST.Tensor(
                        an_name, "F16", (P, 64),
                        lambda src=e.src, n=N, k=K, p=pm:
                        _native_pxq4_pair(gg, src, n, k, p)[1].tobytes()))
                else:
                    # Re-encoded: the encode is expensive and non-deterministic to repeat
                    # lazily inside a writer callback, so it is done once, here.
                    slabs, anchor, _desc = _pxq4_payload(gg, e.src, N, K, enc, ti, args.ref_hf,
                                                         pm)
                    writer.add(ST.Tensor.from_numpy(sl_name, slabs))
                    writer.add(ST.Tensor.from_numpy(an_name, anchor))

        index = writer.finish()
        print(f"wrote {len(set(index['weight_map'].values()))} shard(s), "
              f"{index['metadata']['total_size']:,} B", file=sys.stderr)

        write_config(args.out, args.ref_hf, qcfg)
        return 0
    finally:
        gg.close()


def verify_pxq4_roundtrip(ggml_name: str, gg, ti, slabs: np.ndarray,
                          anchor: np.ndarray, perm=None) -> None:
    """Plan §5.6 check 2. Only meaningful for a NATIVE pxq4 source: the split must rejoin to
    the original bytes exactly. A re-encoded tensor has no original bytes to compare against —
    ``encoder.encode_and_check`` covers that case instead.

    For a GDN tensor the emitted panels are in HF head order, so the reorder is undone first.
    That keeps this a BYTE comparison — it now proves two things at once: the split is a
    partition of the file, and the head reorder is a lossless permutation of whole panels
    rather than a slice that dropped or duplicated one."""
    if ti.type_id != G.GGML_PXQ4:
        return
    if perm is not None:
        slabs, anchor = _unapply_perm_pxq4(slabs, anchor, perm)
    if L.join_blob(slabs, anchor) != bytes(gg.raw(ggml_name)):
        raise SystemExit(f"{ggml_name}: split -> join did not reproduce the original bytes. "
                         f"The panel arithmetic is wrong; nothing downstream can be trusted.")


def write_config(out_dir: str, ref_hf: str | None, qcfg: dict) -> None:
    if not ref_hf:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"quantization_config": qcfg}, f, indent=1)
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
    for e in plan.emits:
        if e.name.endswith(".pxq4_slabs"):
            ours.add(e.name[: -len(".pxq4_slabs")] + ".weight")
        elif e.name.endswith(".pxq4_anchor"):
            continue
        else:
            ours.add(e.name)

    missing = sorted(ref_logical - ours)
    extra = sorted(ours - ref_logical)
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
    ap.add_argument("--encoder", default=None,
                    help="path to pxq4_encode.so; required by p2a/p2b/p2c")
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
