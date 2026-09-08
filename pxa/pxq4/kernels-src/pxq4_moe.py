# SPDX-License-Identifier: Apache-2.0
"""pxq4_moe.py -- PXQ4 FusedMoE quantization method.

This is the engine half of blocker C in FABLE-122B-VLLM-FINDINGS.md. Before it,
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

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs

PANEL_ROWS = 64
SLAB_COLS = 32
SLAB_BYTES = 1088

# v8 indexed MoE path (device-resident topk ids -> capture-safe). PXQ4_MOE_INDEXED=0 restores
# the host-sync per-expert loop unconditionally (eager-only, the pre-v8 behaviour).
_INDEXED_ENABLED = os.getenv("PXQ4_MOE_INDEXED", "1") != "0"


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


class PXQ4MoEMethod(FusedMoEMethodBase):
    """FusedMoE method that keeps routed experts in PXQ4 and decodes them per use."""

    def __init__(self, quant_config, moe) -> None:
        super().__init__(moe)
        self.quant_config = quant_config

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

        dev = torch.cuda.current_device()
        specs = {
            "w13_pxq4_slabs": (torch.empty(E, p13, s13, SLAB_BYTES, dtype=torch.uint8, device=dev)),
            "w13_pxq4_anchor": (torch.empty(E, p13, PANEL_ROWS, dtype=torch.float16, device=dev)),
            "w2_pxq4_slabs": (torch.empty(E, p2, s2, SLAB_BYTES, dtype=torch.uint8, device=dev)),
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

        if not (hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, "moe_mmv_out")):
            layer.pxq4_moe_indexed_ok = False
            return
        h_ok = bool(torch.ops.pxq4.mmv_supported(int(layer.pxq4_H)))
        i_ok = bool(torch.ops.pxq4.mmv_supported(int(layer.pxq4_I)))
        layer.pxq4_moe_indexed_ok = h_ok and i_ok
        if not layer.pxq4_moe_indexed_ok:
            return

        top_k = int(getattr(self.moe, "experts_per_token", 0) or
                    getattr(self.moe, "top_k", 0) or 8)
        s_max = int(os.getenv("PXQ4_MOE_INDEXED_MAX_S", str(8 * top_k)))
        layer.pxq4_moe_smax = s_max

        dev = layer.w13_pxq4_slabs.device
        ids = torch.zeros((s_max,), dtype=torch.int32, device=dev)
        x13 = torch.zeros((s_max, layer.pxq4_H), dtype=torch.float16, device=dev)
        o13 = torch.empty((s_max, layer.pxq4_n13), dtype=torch.float16, device=dev)
        x2 = torch.zeros((s_max, layer.pxq4_I), dtype=torch.float16, device=dev)
        o2 = torch.empty((s_max, layer.pxq4_H), dtype=torch.float16, device=dev)
        s = 1
        while True:
            torch.ops.pxq4.moe_mmv_out(o13[:s], x13[:s], ids[:s],
                                       layer.w13_pxq4_slabs, layer.w13_pxq4_anchor)
            torch.ops.pxq4.moe_mmv_out(o2[:s], x2[:s], ids[:s],
                                       layer.w2_pxq4_slabs, layer.w2_pxq4_anchor)
            if s >= s_max:
                break
            s = min(s + 1, s_max)
        del ids, x13, o13, x2, o2

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
        from . import pxq4_ops as _ops  # noqa: F401  (registers torch.ops.pxq4)

        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M, H = x2.shape
        out = torch.zeros((M, H), dtype=torch.float16, device=x2.device)

        w13_s, w13_a = layer.w13_pxq4_slabs, layer.w13_pxq4_anchor
        w2_s, w2_a = layer.w2_pxq4_slabs, layer.w2_pxq4_anchor
        I = layer.pxq4_I

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
            torch.ops.pxq4.moe_mmv_out(gu, xg, ids, w13_s, w13_a)
            act = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
            dn = torch.empty((s_rows, H), dtype=torch.float16, device=x2.device)
            torch.ops.pxq4.moe_mmv_out(dn, act.contiguous(), ids, w2_s, w2_a)
            # fold the topk slots per token in fp32 (a padded slot's id is < 0; its dn row is
            # zeros by the kernel contract, so no mask is needed)
            wts = topk_weights.to(torch.float32).reshape(M, top_k, 1)
            folded = (dn.view(M, top_k, H).to(torch.float32) * wts).sum(dim=1)
            return folded.to(torch.float16).reshape(*x.shape[:-1], H)

        # HOST SYNC. topk_ids has to come to the CPU to drive a per-expert Python loop, which
        # is precisely why this path cannot be captured into a CUDA graph. Documented, not
        # hidden: the grouped-kernel port removes it.
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
            torch.ops.pxq4.linear_out(gu, xe, w13_s[e], w13_a[e])
            gate, up = gu[:, :I], gu[:, I:]
            act = torch.nn.functional.silu(gate) * up

            dn = torch.empty((m, H), dtype=torch.float16, device=x2.device)
            torch.ops.pxq4.linear_out(dn, act.contiguous(), w2_s[e], w2_a[e])
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
