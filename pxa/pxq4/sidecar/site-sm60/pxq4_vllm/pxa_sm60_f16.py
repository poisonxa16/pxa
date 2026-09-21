# SPDX-License-Identifier: Apache-2.0
"""sm_60 fp16 dense decode fast path — plugin-side, no fork edits.

MECHANISM. On P100 the model's plain-fp16 linears (in_proj_a/b on 48 GDN layers,
q/k/v_proj on 16 attn layers, out_proj, lm_head) run through cuBLAS gemv2T at decode:
profiled at 22.7% of the decode step's self-CUDA time (10,736 launches / 96 tokens =
~112 per step @ 118 us avg). pxq4::f16_mmv_out (op v9) is a warp-per-row fp32-accum
GEMV/MT-GEMM measured 1.6-4.9x faster than torch.mm on the real shapes at M<=8.

This module arms it by patching, from the plugin entry point (runs in every vLLM
process), two seams the fork already provides:
  * UnquantizedLinearMethod.process_weights_after_loading — after the original, mark
    eligible fp16 layers (`layer._pxa60_f16 = True`).
  * linear._maybe_sm70_dense_forward — the fork's own pre-apply fast-path hook, called
    unconditionally at each linear forward; our wrapper serves armed layers at M<=8
    and defers to the original (and then cuBLAS) otherwise.

Capture safety: one preallocated-out kernel launch, no host reads, no allocation other
than `out` (graph-pool under capture) — same contract as pxq4::mmv_out.

Env:
  PXA_SM60_F16_MMV=0        kill switch (default on when the op + an sm_60 device exist)
  PXA_SM60_F16_SUFFIXES     comma list overriding the target suffixes
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# The fork's own sm70 target list covers exactly this model's fp16 modules
# (in_proj_ba x48, out_proj x48, qkv_proj x16 = the profiled 112 gemvs/step);
# _eligible()'s fp16-weight check naturally excludes any of these that are PXQ4.
_DEFAULT_SUFFIXES = {
    "gate_up_proj", "down_proj", "in_proj_ba", "in_proj_qkvz",
    "qkv_proj", "o_proj", "out_proj",
}

_patched = False


def maybe_patch() -> None:
    global _patched
    if _patched:
        return
    if os.getenv("PXA_SM60_F16_MMV", "1") == "0":
        return
    _patched = True

    suffixes = _DEFAULT_SUFFIXES
    env_sfx = os.getenv("PXA_SM60_F16_SUFFIXES")
    if env_sfx:
        suffixes = {s.strip() for s in env_sfx.split(",") if s.strip()}

    from vllm.model_executor.layers import linear as _lin

    orig_pwal = _lin.UnquantizedLinearMethod.process_weights_after_loading
    orig_hook = _lin._maybe_sm70_dense_forward

    def _eligible(layer: torch.nn.Module) -> bool:
        w = getattr(layer, "weight", None)
        if w is None or not isinstance(w, torch.Tensor):
            return False
        if w.dtype != torch.float16 or not w.is_cuda or w.dim() != 2:
            return False
        if not w.is_contiguous() or (w.shape[1] % 8) != 0:
            return False
        cc = torch.cuda.get_device_capability(w.device)
        if cc not in ((6, 0), (7, 0)):
            return False
        if getattr(layer, "_sm70_f16_prepared", False):
            return False  # the fork's own TurboMind path is armed; keep it
        prefix = getattr(layer, "prefix", "") or ""
        if prefix.rsplit(".", 1)[-1] not in suffixes:
            return False
        if not (hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, "f16_mmv_out")):
            return False
        return True

    def pwal(self, layer: torch.nn.Module) -> None:  # noqa: ANN001
        orig_pwal(self, layer)
        try:
            if _eligible(layer):
                layer._pxa60_f16 = True
                logger.info_once(
                    "pxa sm60 fp16 mmv fast path armed (first layer: %s)",
                    getattr(layer, "prefix", "?"))
        except Exception:  # pragma: no cover — arming must never break loading
            logger.exception("pxa sm60 f16 arming failed; layer left on cuBLAS")

    def hook(layer, x, bias):  # noqa: ANN001
        if getattr(layer, "_pxa60_f16", False):
            # NO M branch here: the mmv-vs-cuBLAS policy runs inside the C++ op, per
            # call — a Python branch is baked by torch.compile for the whole range
            # (this exact bug killed the first V100 arm at M=16).
            x2 = x.reshape(-1, x.shape[-1])
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            w = layer.weight
            out = torch.empty((x2.shape[0], w.shape[0]), dtype=torch.float16,
                              device=x2.device)
            torch.ops.pxq4.f16_mmv_out(out, x2, w)
            if bias is not None:
                out = out + bias
            return out.reshape(*x.shape[:-1], w.shape[0])
        return orig_hook(layer, x, bias)

    _lin.UnquantizedLinearMethod.process_weights_after_loading = pwal
    _lin._maybe_sm70_dense_forward = hook

    # lm_head logits gemv (2.1 ms/step at M=1 through gemv2T on the 75968x5120 shard).
    # UnquantizedEmbeddingMethod.apply is the logits matmul; embedding lookup is separate.
    try:
        from vllm.model_executor.layers import vocab_parallel_embedding as _vpe

        orig_emb_apply = _vpe.UnquantizedEmbeddingMethod.apply

        def emb_apply(self, layer, x, bias=None):  # noqa: ANN001
            w = getattr(layer, "weight", None)
            if (
                isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.is_cuda
                and w.dim() == 2 and w.is_contiguous() and (w.shape[1] % 8) == 0
                and hasattr(torch.ops, "pxq4")
                and hasattr(torch.ops.pxq4, "f16_mmv_out")
                and torch.cuda.get_device_capability(w.device) in ((6, 0), (7, 0))
            ):
                x2 = x.reshape(-1, x.shape[-1])
                if not x2.is_contiguous():
                    x2 = x2.contiguous()
                out = torch.empty((x2.shape[0], w.shape[0]), dtype=torch.float16,
                                  device=x2.device)
                torch.ops.pxq4.f16_mmv_out(out, x2, w)
                if bias is not None:
                    out = out + bias
                return out.reshape(*x.shape[:-1], w.shape[0])
            return orig_emb_apply(self, layer, x, bias)

        _vpe.UnquantizedEmbeddingMethod.apply = emb_apply
    except Exception:  # pragma: no cover
        logger.exception("pxa sm60 lm_head hook failed; cuBLAS kept for logits")

    logger.info_once("pxa sm60 fp16 mmv fast path patch installed")
