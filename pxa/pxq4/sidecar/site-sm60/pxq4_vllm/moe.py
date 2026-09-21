# SPDX-License-Identifier: Apache-2.0
"""pxq4_moe.py -- PXQ FusedMoE quantization method (tiers pxq2 / pxq3 / pxq4).

TIERS (2026-09-05). This class used to be PXQ4-only. It now serves any of the three PXQ panel
tiers, and the two stacked FusedMoE parameters may be DIFFERENT tiers: vLLM keeps w13
(gate+up, column parallel) and w2 (down, row parallel) as separate tensors, so a checkpoint
whose gate/up experts are pxq2 and whose down experts are pxq3 -- which is exactly what the
Flash-Next quantizer emits -- is expressible without breaking the uniformity invariant. What
is NOT allowed is a mixed tier INSIDE one parameter: gate and up share w13, so they must
match, and the converter refuses a file where they do not.

Only three things vary with the tier (see tiers.py): the slab stride (576 / 832 / 1088), the
op family, and the book length. Sharding, loader semantics, the capture-safety argument and
the arithmetic contract are identical, which is why the tier is a lookup and not a branch.
The slab stride is ASSERTED against the parameter's own shape rather than inferred: a pxq2
tensor read with the pxq4 stride loads, shards, passes every structural gate and generates
fluent garbage.

Before this class existed,
``PXQ4Config.get_quant_method`` returned ``None`` for every ``FusedMoE`` layer, which
``fused_moe/layer.py:357-358`` turns into ``UnquantizedFusedMoEMethod`` -- i.e. fp16 expert
weights. For the 122B that is 216.0 GiB of expert weight against 63.55 GiB of P100 VRAM, a
3.40x overshoot; for the 35B it is 33.8 GiB against 31.8 GiB. Both are unloadable, so the
whole point of this file is that the experts stay PXQ4 in memory and are decoded per use.

WEIGHT LAYOUT
-------------
vLLM's FusedMoE contract is two stacked parameters per layer:

    w13   [E, 2*I_p, H]     gate and up, concatenated on the output axis (column parallel)
    w2    [E, H,     I_p]   down                                          (row parallel)

``I_p`` is ALREADY the per-rank intermediate size -- vLLM hands ``create_weights`` the sharded
value. Each of those becomes a PXQ4 (slabs, anchor) pair with the expert as a new slowest axis:

    w13_pxq4_slabs  uint8   [E, 2*I_p/64, H/32,   1088]
    w13_pxq4_anchor float16 [E, 2*I_p/64, 64]
    w2_pxq4_slabs   uint8   [E, H/64,     I_p/32, 1088]
    w2_pxq4_anchor  float16 [E, H/64,     64]

SHARDING, AND WHY BOTH DIRECTIONS ARE LEGAL BYTE MOVES
------------------------------------------------------
Column parallel (w13) cuts the OUTPUT axis. A PXQ4 panel is 64 output rows, so the cut is a
whole-panel slice as long as ``I_p % 64 == 0``; the anchor (one fp16 per row) is cut the same
way. Nothing inside a panel is touched.

Row parallel (w2) cuts the CONTRACTION axis. A slab is 64 rows x 32 columns and carries its own
sub-scale, so a K-cut is a whole-slab slice as long as ``I_p % 32 == 0``, and the anchor is
NOT cut -- every rank keeps the full per-output-row anchor and the partial products are summed
by the all-reduce afterwards. That duplication is correct because the anchor is a linear
per-row scale: ``sum_r scale*partial_r == scale * sum_r partial_r``.

Both conditions are asserted in ``create_weights`` rather than assumed, because a silent
truncation here produces a model that loads and generates fluent garbage.

COMPUTE
-------
``apply`` is a per-expert loop, NOT a grouped kernel. For each expert actually routed to in
this batch it gathers that expert's tokens, runs the existing 2-D PXQ4 ops on them, and
scatters the weighted result back. This is deliberately the simple correct thing:

  * it reuses ``pxq4::mmv_out`` / ``pxq4::dequant_out``, which are already validated bit-exact
    against the numpy oracle, so nothing new has to be trusted numerically;
  * it keeps the experts PXQ4 in memory, which is the entire memory argument;
  * it is SLOW -- one kernel launch per routed expert per projection per layer, and a
    host-side ``.tolist()`` of the routed expert set, which makes it CUDA-graph hostile.

The fast path is the grouped kernel family that already exists on the llama.cpp side
(``ggml/src/ggml-cuda/pxq6.cuh``: ``moe-gateup-split`` / ``moe-down`` / ``k_pxq6_gemm_grouped``,
with pxq2/3/6 sharing one policy-templated family). Porting that off ggml's ``MUL_MAT_ID``
onto this contract is the follow-on; this file is what makes that port testable end to end.
"""

from __future__ import annotations

import os
import sys

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs

from . import tiers as _tiers

PANEL_ROWS = 64
SLAB_COLS = 32
#: Legacy name, kept because out-of-tree code imports it. It is the PXQ4 stride and is NOT
#: the stride of an arbitrary layer any more -- read ``layer.pxq_tier13.slab_bytes`` instead.
SLAB_BYTES = 1088

# v8 indexed MoE path (device-resident topk ids -> capture-safe). PXQ4_MOE_INDEXED=0 restores
# the host-sync per-expert loop unconditionally (eager-only, the pre-v8 behaviour).
_INDEXED_ENABLED = os.getenv("PXQ4_MOE_INDEXED", "1") != "0"

# ---------------------------------------------------------------------------------------------
# FUSED MoE decode block. PXQ_MOE_FUSED selects the routed-expert path:
#
#   0    (default) the v8 indexed path below, unchanged. Off until it is measured.
#   1    FORM A: two launches -- gate+up+SwiGLU, then down + the router-weighted top_k fold.
#   2    FORM B: three launches -- the same gate+up, then per-slot down partials on the old
#        grid, then the fold pass. Bit-identical to form A by construction; it keeps the block
#        count form A trades away, and only a measurement can say which side of that trade
#        this rig is on.
#   ref  REFERENCE: the v8 ops, but with the slot fold written as an explicit ascending fp32
#        loop instead of torch's sum(dim=1). This is the thing forms A and B are required to
#        equal BIT-EXACTLY, and it is a real gate rather than a tolerance -- see the unit test.
#
# The default is 0 and stays 0 until the window says otherwise. Nothing about the shipped path
# changes while it is 0: the branch is taken on an env read and a shape, never on data.
_FUSED_MODE = (os.getenv("PXQ_MOE_FUSED", "0") or "0").strip().lower()

# Token ceiling for the fused path. Above it the GEMV shape stops being the right one for the
# same reason PXQ4_MMV_MAX_M exists (grid.y is the row axis, so each block re-reads its panel
# once per row) and the existing branches take over. Decode is M <= 8 on this stack.
_FUSED_MAX_M = int(os.getenv("PXQ_MOE_FUSED_MAX_M", "8"))


# ---------------------------------------------------------------------------------------------
# Per-device scratch for the fused path: `act` [S, Ip] and, for form B, `dn` [S, H].
#
# WHY A FROZEN PER-DEVICE BUFFER AND NOT A per-call torch.empty. A CUDA graph records RAW
# DEVICE ADDRESSES. The caching allocator does make torch.empty legal inside a capture, but the
# v12b use-after-free note in pxq4_kernel_torch.cpp is about the other half of that rule: a
# buffer that is REPLACED after a capture hands the old block back while captured graphs still
# write into it. A single buffer allocated at its maximum size before any capture, and only
# ever sliced afterwards, cannot move -- so there is no seal, no counter, and no arena to grow.
# It is also shared across LAYERS rather than per-layer, because layers run sequentially and a
# per-layer copy would be 40x the memory for no benefit.
#
# The cap is small precisely because the fused path is gated at M <= 8: at the 35B's shapes
# (H 2048, Ip 512 at TP=1, top_k 8) that is 64 rows -> 64 KB of act and 256 KB of dn.
_FUSED_SCRATCH: dict[tuple[int, str], torch.Tensor] = {}

# PHANTOM-LEVER DISCIPLINE, borrowed from the engine (ggml-cuda.cu's *_split_log family): an
# armed lever that never prints FIRING did not engage, which makes the A/B VOID rather than
# null. So the fused path announces its first firing and its first decline per process, with
# the reason and the shape, exactly once each -- enough for a window to prove from the log that
# it measured what it thinks it measured, and quiet enough to leave on in production.
_FUSED_LOGGED: set[str] = set()


def _fused_log(what: str, msg: str) -> None:
    if what in _FUSED_LOGGED:
        return
    _FUSED_LOGGED.add(what)
    print(f"PXQ_MOE_FUSED {what}: {msg}", file=sys.stderr, flush=True)


def _fused_scratch(dev: torch.device, kind: str, need: int) -> torch.Tensor:
    key = (int(dev.index if dev.index is not None else 0), kind)
    buf = _FUSED_SCRATCH.get(key)
    if buf is not None and buf.numel() >= need:
        return buf
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"pxq moe fused: the {kind} scratch on {dev} would have to grow to {need} fp16 "
            f"elements during CUDA graph capture (have "
            f"{0 if buf is None else buf.numel()}). Every shape a captured graph replays must "
            f"have run once eagerly first -- process_weights_after_loading sizes this buffer, "
            f"so reaching here means a capture size the warmup did not cover.")
    buf = torch.empty((need,), dtype=torch.float16, device=dev)
    _FUSED_SCRATCH[key] = buf
    return buf


# v11 epilogue . "legacy" is the shipped, byte-gated arithmetic and is the DEFAULT.
# Measured on GPU 3 (bench/moe_census.py) and confirmed on the P100 pair: one MoE layer at
# decode width M=1 costs 12 CUDA kernels and 98.6 us at the TP=2 shard, of which only TWO are
# moe_mmv. The other ten -- the expand, the slice/silu/mul, four casts, the fp32 multiply and
# the reduce -- are 40.8 us, 41% of the layer, and that share does not shrink with the TP
# width because none of it scales with the shard. "fused" replaces the slice+silu+mul with
# vLLM's silu_and_mul where available and the cast/mul/sum/cast fold with one batched fp32
# GEMM, taking the layer to 10 kernels and 86.6 us.
#
# It is OFF by default because both substitutions change ROUNDING, not results: silu_and_mul
# evaluates silu in fp32 where torch evaluates it in fp16, and a bmm reduces the top_k slots
# in a different order than sum(). Gated on GPU 3 at four widths: bit-identical at M=1 and
# M=2, 2e-6 apart at M=4 and M=8. A last bit is a different token, so: legacy.
_EPILOGUE = os.getenv("PXQ4_MOE_EPILOGUE", "legacy").strip().lower()


def _geom(N: int, K: int, what: str) -> tuple[int, int]:
    if N % PANEL_ROWS:
        raise ValueError(
            f"pxq4 moe: {what} output size {N} is not a multiple of the {PANEL_ROWS}-row "
            f"panel. At this TP size the shard would cut a panel in half and the packed "
            f"arithmetic would truncate silently.")
    if K % SLAB_COLS:
        raise ValueError(
            f"pxq4 moe: {what} contraction size {K} is not a multiple of the {SLAB_COLS}-column "
            f"slab. The shard would cut a slab and its sub-scale apart.")
    return N // PANEL_ROWS, K // SLAB_COLS


def _max_capture_tokens(default: int = 8) -> int:
    """Largest token count a captured decode graph can replay.

    The arenas behind ``moe_mmv_out`` refuse to grow under stream capture, so
    the eager pre-capture sweep in ``process_weights_after_loading`` has to
    cover every S = tokens * top_k a captured graph can ask for. That bound
    used to be hardcoded at 8 tokens, which silently matched the shipped
    ladder [1,2,4,8] and silently did NOT match any wider one: raise
    ``max_num_seqs`` past 8 and the ladder grows to 16, the sweep still stops
    at 8 * top_k, and the first captured 16-token replay hits the arena's
    growth check mid-capture. Read the ladder the engine was actually given
    instead of assuming it.
    """
    try:
        from vllm.config import get_current_vllm_config

        sizes = get_current_vllm_config().compilation_config.cudagraph_capture_sizes
        if sizes:
            return max(int(s) for s in sizes)
    except Exception:
        pass
    return default


def _glu(gu, I: int):
    """silu(gate) * up over the concatenated [gate | up] projection.

    legacy: two torch kernels over two non-contiguous slices.
    fused:  one launch of vLLM's silu_and_mul, which reads the concatenated tensor and
            writes the halved one. NOT bit-identical: it evaluates silu in fp32 where the
            torch path evaluates it in fp16. Falls back silently if the op is absent --
            on the sm_60 image torch.ops._C carries only ten ops and this is not one of
            them, so on Pascal this branch is a no-op today and the fold below is the
            whole of the "fused" saving.
    """
    if _EPILOGUE == "fused":
        try:
            out = torch.empty((gu.shape[0], I), dtype=gu.dtype, device=gu.device)
            torch.ops._C.silu_and_mul(out, gu)
            return out
        except Exception:
            pass
    act = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    return act if act.is_contiguous() else act.contiguous()


def _fold(dn, topk_weights, M: int, top_k: int, H: int):
    """Weighted sum of the top_k expert outputs per token.

    legacy: cast the weights to fp32, cast dn to fp32, multiply into an [M, top_k, H]
            fp32 temporary, reduce it, cast back -- five kernels and a temporary eight
            times the size of the answer at decode width.
    fused:  one cast plus one batched fp32 GEMM. Same arithmetic in fp32, different
            REDUCTION ORDER, so it is not promised bit-identical and must clear the
            20-prompt byte gate on the pair before it could ever be a default.
    """
    if _EPILOGUE == "fused":
        try:
            w = topk_weights.to(torch.float32).view(M, 1, top_k)
            r = torch.bmm(w, dn.view(M, top_k, H).to(torch.float32))
            return r.view(M, H).to(torch.float16)
        except Exception:
            pass
    wts = topk_weights.to(torch.float32).reshape(M, top_k, 1)
    folded = (dn.view(M, top_k, H).to(torch.float32) * wts).sum(dim=1)
    return folded.to(torch.float16)


class PXQ4MoEMethod(FusedMoEMethodBase):
    """FusedMoE method that keeps routed experts in PXQ4 and decodes them per use."""

    def __init__(self, quant_config, moe, prefix: str = "") -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        # Resolved once, here, so that a checkpoint declaring a tier this build cannot serve
        # fails at layer construction with a clear message rather than at the first forward.
        resolve = getattr(quant_config, "tier_for", None)
        if resolve is None:
            self.tier13 = self.tier2 = _tiers.default_tier()
        else:
            self.tier13 = resolve(prefix, "w13")
            self.tier2 = resolve(prefix, "w2")
        for t in {self.tier13, self.tier2}:
            if not _tiers.ops_available(t):
                raise RuntimeError(
                    f"pxq moe: this checkpoint serves {prefix or '<moe>'} at tier {t.name}, "
                    f"but the loaded kernel library does not carry the {t.name} ops. Point "
                    f"PXQ4_LIB at libpxq_sm60_v13.so / libpxq_sm70_v13.so; every earlier "
                    f"library is PXQ4-only.")

    # ------------------------------------------------------------------ weights
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        E = num_experts
        H = hidden_size
        I = intermediate_size_per_partition
        n13 = 2 * I if getattr(self.moe, "is_act_and_mul", True) else I

        p13, s13 = _geom(n13, H, "w13 (gate_up)")
        p2, s2 = _geom(H, I, "w2 (down)")

        # The stride is the tier. Allocating w13 at the wrong stride is undetectable
        # downstream, so it comes from the resolved tier and the ops re-assert it.
        sb13, sb2 = self.tier13.slab_bytes, self.tier2.slab_bytes
        dev = torch.cuda.current_device()
        specs = {
            "w13_pxq4_slabs": (torch.empty(E, p13, s13, sb13, dtype=torch.uint8, device=dev)),
            "w13_pxq4_anchor": (torch.empty(E, p13, PANEL_ROWS, dtype=torch.float16, device=dev)),
            "w2_pxq4_slabs": (torch.empty(E, p2, s2, sb2, dtype=torch.uint8, device=dev)),
            "w2_pxq4_anchor": (torch.empty(E, p2, PANEL_ROWS, dtype=torch.float16, device=dev)),
        }
        # FusedMoE puts its own ``weight_loader`` in extra_weight_attrs, and
        # ``set_weight_attrs`` ASSERTS rather than overwrites (model_executor/utils.py:29),
        # so ours has to replace it in the dict -- not be applied as a second call.
        # The stock loader slices dense [E, N, K] tensors and would index our 1088-byte slab
        # axis as if it were K, so it must not survive.
        attrs = dict(extra_weight_attrs)
        attrs["weight_loader"] = self._weight_loader
        for name, t in specs.items():
            param = torch.nn.Parameter(t, requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, attrs)

        layer.pxq4_E = E
        layer.pxq4_H = H
        layer.pxq4_I = I
        layer.pxq4_n13 = n13
        layer.pxq_tier13 = self.tier13
        layer.pxq_tier2 = self.tier2

    # ------------------------------------------------------------------ loading
    def _weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ):
        """Place ONE expert's on-disk PXQ4 tensor into the stacked parameter.

        ``loaded_weight`` is the FULL (unsharded) tensor for that expert as the converter wrote
        it, so this does the TP cut as well as the placement. ``shard_id`` is vLLM's
        w1 = gate, w3 = up, w2 = down.
        """
        ok = self._place(param, loaded_weight, weight_name, shard_id, expert_id)
        return ok if return_success else None

    def _place(self, param, loaded, weight_name, shard_id, expert_id) -> bool:
        try:
            tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        except Exception:
            tp = (0, 1)
        rank, world = tp
        data = param.data
        is_anchor = weight_name.endswith("_pxq4_anchor")

        # TIER CHECK AT LOAD. The on-disk key says nothing about the tier (it is a wire-format
        # name), so the only place a tier mismatch between checkpoint and config can be caught
        # is here, where both shapes are in hand. Without it a pxq3 tensor copied into a pxq2
        # parameter is a shape error at best and a silent truncation at worst.
        if not is_anchor and loaded.dim() >= 1 and loaded.shape[-1] != data.shape[-1]:
            raise ValueError(
                f"pxq moe: {weight_name} expert {expert_id} has slab stride "
                f"{loaded.shape[-1]} on disk but the layer was built for "
                f"{data.shape[-1]} (tier "
                f"{(self.tier13 if shard_id in ('w1', 'w3') else self.tier2).name}). The "
                f"checkpoint's quantization_config.pxq_tiers disagrees with its own tensors.")

        if shard_id in ("w1", "w3"):
            # Column parallel on the output axis == the PANEL axis of both slabs and anchor.
            per = data.shape[1] // 2          # panels per half (gate or up)
            beg = 0 if shard_id == "w1" else per
            src = loaded.narrow(0, rank * per, per)
            data[expert_id].narrow(0, beg, per).copy_(src)
            return True

        if shard_id == "w2":
            if is_anchor:
                # Row parallel does NOT cut the output axis, so the anchor is replicated whole.
                data[expert_id].copy_(loaded)
            else:
                # Cut the K-slab axis; each slab keeps its own sub-scale.
                per = data.shape[2]
                data[expert_id].copy_(loaded.narrow(1, rank * per, per))
            return True

        raise ValueError(f"pxq4 moe: unexpected shard_id {shard_id!r} for {weight_name!r}")

    # ------------------------------------------------------------------ warmup
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Size the shared mmv partials/counter arenas for every S the indexed MoE path can
        see inside a CUDA graph, eagerly and pre-capture.

        The arenas (pxq4_kernel_torch.cpp) refuse to grow under stream capture, so every
        (S, shape) the captured graphs can replay must have run once eagerly first. vLLM's
        pre-capture warmup runs the model at each capture size, but the dispatch inside
        moe_mmv_out switches mono/fused-split on S*panels, so we sweep S = 1..S_max here
        rather than trusting the warmup batch shapes to cover every branch. Dummy data;
        ~2*S_max tiny launches, once per layer, at load time.
        """
        from . import ops as _ops  # noqa: F401  (registers torch.ops.pxq4)

        # TABLES FIRST, and eagerly. These are cudaMemcpyToSymbol calls, so they must land
        # before any graph capture -- and before the sweep below, which would otherwise warm
        # the arenas with one book and run with another. The checkpoint's own book is used,
        # never the compiled-in default: the quantizer's tables are overridable and the file
        # is the only record of which were actually used (config.book_for raises rather than
        # guessing). The SUB16 LUT is shared by every tier, so it is uploaded once.
        cfg = self.quant_config
        if hasattr(cfg, "book_for"):
            _sub = getattr(cfg, "tier_sub", None)
            for _t in {self.tier13, self.tier2}:
                _tiers.upload_tables(_t, cfg.book_for(_t), _sub)

        if not (_tiers.ops_available(self.tier13) and _tiers.ops_available(self.tier2)):
            layer.pxq4_moe_indexed_ok = False
            return

        def _supported(tier, K: int) -> bool:
            # The library is the authority on whether a K fits the mmv's shared-memory
            # budget; there is no python-side formula guaranteed to track the kernel's
            # chunking, and an over-optimistic guess routes a layer into a kernel that
            # cannot launch.
            if tier.is_pxq4:
                return bool(torch.ops.pxq4.mmv_supported(int(K)))
            return bool(torch.ops.pxq4.pxq_supported(tier.type_id, int(K)))

        h_ok = _supported(self.tier13, layer.pxq4_H)
        i_ok = _supported(self.tier2, layer.pxq4_I)
        layer.pxq4_moe_indexed_ok = h_ok and i_ok
        if not layer.pxq4_moe_indexed_ok:
            return

        top_k = int(getattr(self.moe, "experts_per_token", 0) or
                    getattr(self.moe, "top_k", 0) or 8)
        s_max = int(os.getenv("PXQ4_MOE_INDEXED_MAX_S",
                              str(_max_capture_tokens() * top_k)))
        layer.pxq4_moe_smax = s_max

        dev = layer.w13_pxq4_slabs.device
        ids = torch.zeros((s_max,), dtype=torch.int32, device=dev)
        x13 = torch.zeros((s_max, layer.pxq4_H), dtype=torch.float16, device=dev)
        o13 = torch.empty((s_max, layer.pxq4_n13), dtype=torch.float16, device=dev)
        x2 = torch.zeros((s_max, layer.pxq4_I), dtype=torch.float16, device=dev)
        o2 = torch.empty((s_max, layer.pxq4_H), dtype=torch.float16, device=dev)
        moe13 = _tiers.op(self.tier13, "moe_mmv")
        moe2 = _tiers.op(self.tier2, "moe_mmv")
        s = 1
        while True:
            moe13(o13[:s], x13[:s], ids[:s],
                  layer.w13_pxq4_slabs, layer.w13_pxq4_anchor)
            moe2(o2[:s], x2[:s], ids[:s],
                 layer.w2_pxq4_slabs, layer.w2_pxq4_anchor)
            if s >= s_max:
                break
            s = min(s + 1, s_max)
        del ids, x13, o13, x2, o2
        # ---- fused MoE decode block ------------------------------------------------------
        # Availability is ASKED of the library, not inferred from its file name: a v13 built
        # before the fused kernels landed carries every other op of this tier, so a name check
        # would pass and the first captured forward would die on an AttributeError.
        # The n13 == 2*I test is NOT redundant with the panel check, and getting it wrong is a
        # silent wrong answer rather than an error. create_weights sets n13 = 2*I only when the
        # layer is act_and_mul; for a plain (non-gated) MoE it is n13 == I, and w13 then holds
        # ONE matrix rather than a gate half followed by an up half. The fused gateup kernel
        # walks panel p and panel p + panels/2 as a gate/up pair, so on such a layer it would
        # happily multiply the first half of the output by the second half and return plausible
        # garbage. Qwen3_5Moe is act_and_mul so this does not bite today, which is exactly why
        # it has to be checked rather than assumed.
        layer.pxq_moe_fused_ok = bool(
            _tiers.fused_available(self.tier13) and _tiers.fused_available(self.tier2)
            and layer.pxq4_n13 == 2 * layer.pxq4_I
            and layer.pxq4_I % PANEL_ROWS == 0
            and layer.pxq4_H % PANEL_ROWS == 0)
        if not layer.pxq_moe_fused_ok or _FUSED_MODE in ("0", "", "off", "ref"):
            return

        # Size the shared scratch for the largest S the fused path can ever be handed, EAGERLY
        # and once, then never let it move. _FUSED_MAX_M is the gate in apply(), so this is the
        # true maximum and not a guess.
        s_fused = max(1, _FUSED_MAX_M * top_k)
        _fused_scratch(dev, "act", s_fused * layer.pxq4_I)
        _fused_scratch(dev, "dn", s_fused * layer.pxq4_H)

        # One eager firing per shape the captured graphs can replay. The kernels allocate
        # nothing, so this is not arena warmup -- it is a launch-geometry and shared-memory
        # check that fails HERE, at load time, with a shape in the message, instead of inside
        # a capture where the error is unrecoverable.
        f_ids = torch.zeros((s_fused,), dtype=torch.int32, device=dev)
        f_wts = torch.zeros((s_fused,), dtype=torch.float32, device=dev)
        f_x = torch.zeros((_FUSED_MAX_M, layer.pxq4_H), dtype=torch.float16, device=dev)
        f_act = _fused_scratch(dev, "act", s_fused * layer.pxq4_I)
        f_dn = _fused_scratch(dev, "dn", s_fused * layer.pxq4_H)
        f_out = torch.empty((_FUSED_MAX_M, layer.pxq4_H), dtype=torch.float16, device=dev)
        gu_op = _tiers.op(self.tier13, "moe_gateup")
        dnf_op = _tiers.op(self.tier2, "moe_down_fold")
        dnp_op = _tiers.op(self.tier2, "moe_down_part")
        for m in range(1, _FUSED_MAX_M + 1):
            sr = m * top_k
            a = f_act[:sr * layer.pxq4_I].view(sr, layer.pxq4_I)
            gu_op(a, f_x[:m], f_ids[:sr], layer.w13_pxq4_slabs, layer.w13_pxq4_anchor, top_k)
            dnf_op(f_out[:m], a, f_ids[:sr], f_wts[:sr],
                   layer.w2_pxq4_slabs, layer.w2_pxq4_anchor, top_k)
            d = f_dn[:sr * layer.pxq4_H].view(sr, layer.pxq4_H)
            dnp_op(d, a, f_ids[:sr], layer.w2_pxq4_slabs, layer.w2_pxq4_anchor)
            torch.ops.pxq4.moe_slot_fold_out(f_out[:m], d, f_wts[:sr], top_k)
        del f_ids, f_wts, f_x, f_out

    # ------------------------------------------------------------------ compute
    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        from . import ops as _ops  # noqa: F401  (registers torch.ops.pxq4)

        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M, H = x2.shape
        # out is the accumulator for the HOST-SYNC fallback loop below and nothing else:
        # the indexed branch returns its own tensor and never reads it. Allocating and
        # zeroing it here cost one memset kernel per layer per token on the path actually
        # taken -- 40 dead kernels a token. Created in the fallback now. This is a deletion
        # of unreachable work and cannot move an output bit, so it is not behind the switch.

        w13_s, w13_a = layer.w13_pxq4_slabs, layer.w13_pxq4_anchor
        w2_s, w2_a = layer.w2_pxq4_slabs, layer.w2_pxq4_anchor
        I = layer.pxq4_I

        # ---- FUSED decode block ----------------------------------------------------------
        # Two launches (form A) or three (form B) for the whole routed-expert MLP, against the
        # ten-to-thirteen the indexed path below spends. What disappears is not only launches:
        # the `xg` expand (the kernel reads x per TOKEN, so there is nothing to materialise),
        # the `gu` and `dn` intermediates, the S*H fp16->fp32 expansion, the broadcast multiply,
        # the sum over dim 1, and the final cast. (The dead `out = torch.zeros(M, H)` that used
        # to sit at the top of this function is gone too, for every path -- see above.)
        #
        # The gate is shape-and-env only, never data, so a captured graph always replays the
        # branch it captured.
        _fused_ok = getattr(layer, "pxq_moe_fused_ok", False)
        if _FUSED_MODE in ("1", "2") and not (_fused_ok and M <= _FUSED_MAX_M):
            _fused_log("DECLINED", (
                f"mode={_FUSED_MODE} but "
                + ("the library has no fused ops for this layer's tiers"
                   if not _fused_ok else
                   f"M={M} exceeds PXQ_MOE_FUSED_MAX_M={_FUSED_MAX_M}")
                + f" (H={H} I={I} top_k={top_k}) -> falling back to the indexed path"))
        if _FUSED_MODE in ("1", "2") and _fused_ok and M <= _FUSED_MAX_M:
            _fused_log("FIRING", f"form {'A' if _FUSED_MODE == '1' else 'B'} "
                                 f"(M={M} top_k={top_k} S={s_rows} H={H} I={I})")
            ids = topk_ids.reshape(-1).to(torch.int32)
            wts = topk_weights.reshape(-1).to(torch.float32).contiguous()
            act = _fused_scratch(x2.device, "act", s_rows * I)[:s_rows * I].view(s_rows, I)
            _tiers.op(self.tier13, "moe_gateup")(act, x2, ids, w13_s, w13_a, top_k)
            fused = torch.empty((M, H), dtype=torch.float16, device=x2.device)
            if _FUSED_MODE == "1":
                _tiers.op(self.tier2, "moe_down_fold")(fused, act, ids, wts, w2_s, w2_a, top_k)
            else:
                dn = _fused_scratch(x2.device, "dn",
                                    s_rows * H)[:s_rows * H].view(s_rows, H)
                _tiers.op(self.tier2, "moe_down_part")(dn, act, ids, w2_s, w2_a)
                torch.ops.pxq4.moe_slot_fold_out(fused, dn, wts, top_k)
            return fused.reshape(*x.shape[:-1], H)

        # ---- v8 capture-safe indexed path -------------------------------------------------
        # S = M*topk is a STATIC shape; the only data-dependent quantity is which expert
        # serves each row, and moe_mmv_out reads that from device memory itself. No host
        # sync, so this branch is legal under CUDA-graph capture — which is the whole point:
        # the host-sync loop below forced --enforce-eager, measured at ~3.4x slower decode
        # on this stack (docs/12). The M gate keeps the mmv (one full weight re-read per row)
        # off large prefill batches, where the per-expert dequant+GEMM loop below wins; that
        # loop only ever runs eagerly (capture sizes are <= 8 tokens), so keeping it is safe.
        # torch.cuda.is_current_stream_capturing() backstops the gate: if a capture ever runs
        # a larger S than expected we take the indexed path anyway (correct, if slower)
        # rather than crashing capture with a host sync.
        top_k = int(topk_ids.shape[-1])
        s_rows = M * top_k
        indexed_ok = (_INDEXED_ENABLED and getattr(layer, "pxq4_moe_indexed_ok", False))
        if indexed_ok and (s_rows <= getattr(layer, "pxq4_moe_smax", 0)
                           or torch.cuda.is_current_stream_capturing()):
            ids = topk_ids.reshape(-1).to(torch.int32)
            xg = (x2.unsqueeze(1).expand(M, top_k, H).reshape(s_rows, H).contiguous())
            gu = torch.empty((s_rows, layer.pxq4_n13), dtype=torch.float16, device=x2.device)
            _tiers.op(self.tier13, "moe_mmv")(gu, xg, ids, w13_s, w13_a)
            act = _glu(gu, I)
            dn = torch.empty((s_rows, H), dtype=torch.float16, device=x2.device)
            _tiers.op(self.tier2, "moe_mmv")(dn, act, ids, w2_s, w2_a)
            # Fold the topk slots per token. A padded slot's id is < 0 and its dn row is
            # zeros by the kernel contract, so no mask is needed either way.
            return _fold(dn, topk_weights, M, top_k, H).reshape(*x.shape[:-1], H)

        # HOST SYNC. topk_ids has to come to the CPU to drive a per-expert Python loop, which
        # is precisely why this path cannot be captured into a CUDA graph. Documented, not
        # hidden: the grouped-kernel port removes it.
        out = torch.zeros((M, H), dtype=torch.float16, device=x2.device)
        ids = topk_ids.to("cpu")
        wts = topk_weights.to(torch.float32).to("cpu")

        for e in sorted(set(ids.reshape(-1).tolist())):
            if e < 0:
                continue
            sel = (ids == e).nonzero(as_tuple=False)
            if sel.numel() == 0:
                continue
            rows = sel[:, 0].to(x2.device)
            scale = wts[sel[:, 0], sel[:, 1]].to(x2.device, torch.float16).unsqueeze(1)

            xe = x2.index_select(0, rows).contiguous()
            m = xe.shape[0]

            gu = torch.empty((m, layer.pxq4_n13), dtype=torch.float16, device=x2.device)
            _tiers.op(self.tier13, "linear")(gu, xe, w13_s[e], w13_a[e])
            gate, up = gu[:, :I], gu[:, I:]
            act = torch.nn.functional.silu(gate) * up

            dn = torch.empty((m, H), dtype=torch.float16, device=x2.device)
            _tiers.op(self.tier2, "linear")(dn, act.contiguous(), w2_s[e], w2_a[e])
            out.index_add_(0, rows, (dn * scale).to(torch.float16))

        # DO NOT TOUCH ``shared_experts`` HERE. The MoE runner reads it itself --
        # ``moe_runner.py:782`` evaluates ``self._shared_experts.output`` and passes the
        # result down -- and ``SharedExperts.output`` is a CONSUMING property: it returns
        # the tensor and clears the slot (shared_experts.py:162-167). Reading it from the
        # quant method steals it, and the runner's own read then trips
        # ``assert self._output[self._output_idx] is not None`` at shared_experts.py:163.
        # Calling ``shared_experts.apply()`` instead is equally wrong: its first line asserts
        # the slot is EMPTY, and the runner has already filled it. The shared expert is the
        # runner's business; ours is the routed experts only. Both parameters are accepted
        # and deliberately unused, matching GGUFMoEMethod.apply (gguf.py:643-667).

        return out.reshape(*x.shape[:-1], H)


try:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
except Exception:  # pragma: no cover - import shape differs across forks
    def get_tensor_model_parallel_rank():
        return 0

    def get_tensor_model_parallel_world_size():
        return 1
