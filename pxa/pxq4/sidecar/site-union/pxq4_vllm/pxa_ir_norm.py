# SPDX-License-Identifier: Apache-2.0
"""Plug a fused Pascal/Volta norm into vLLM's ir op registry, from the plugin side.

THE PROBLEM THIS EXISTS FOR. Modern vLLM does not ship fused CUDA kernels for its small
ops; it ships DECOMPOSITIONS and relies on Inductor to fuse them back. ``vllm/ir/ops/
layernorm.py`` defines ``rms_norm`` and ``fused_add_rms_norm`` as plain torch expression
trees, and ``RMSNorm.forward_cuda`` in this fork returns ``forward_native`` unconditionally
(layernorm.py:347-359). On sm_60 there is no Inductor at all -- ``TORCHDYNAMO_DISABLE=1``
is load-bearing, because without it ``profile_run`` compiles the language model and dies
with ``GPUTooOldForTriton``. So every norm in the model runs as its eager decomposition.

Measured on the real shape (vllm-pp, bench/norm_census.py, GPU 3, each op in its own
captured graph): ``fused_add_rms_norm`` is 11 kernels and 25.7 us, invoked 80 times per
decode token; ``rms_norm`` is 8 kernels and 20.1 us, invoked 51 times. That is 1288
kernels and 3.1 ms of a 33.5 ms token -- against 440 kernels and 4.2 ms for every routed
expert in the model. There is no Python-level fix: ``torch.nn.functional.rms_norm`` in
torch 2.7 is itself a decomposition and measures identically, and the only cheaper form
does the residual add in fp16 and is not bit-exact.

THE SEAM. ``vllm.ir`` gives every op a provider registry and a priority list
(``ir/op.py:241`` register_impl, ``:409`` set_default), and with no priority set it says
so out loud in the log: "Priority not set for op fused_add_rms_norm, using native
implementation." So a plugin can register a fused provider and take the dispatch without
touching the fork -- the same shape of hook as ``pxa_sm60_f16`` on the dense linear path.

ARITHMETIC ORDER IS THE WHOLE GATE. A fused kernel is only allowed to become the default
if it reproduces the native order bit for bit: cast to fp32, add the residual in fp32,
write the fp16 residual out, square, mean over the last dim, rsqrt(var + eps), multiply,
cast to the weight dtype, multiply by the weight, cast back. Anything else is a different
last bit, and a different last bit is a different token.

Env:
  PXA_FUSED_NORM=auto      (default) arm only if torch.ops.pxq4 provides the kernel;
                           with no kernel present this module changes nothing at all
  PXA_FUSED_NORM=0         never arm
  PXA_FUSED_NORM=1         arm, and log loudly if the kernel is missing
  PXA_FUSED_NORM=selftest  register a torch-implemented provider identical to native and
                           take the dispatch with it. Proves the seam end to end without
                           a kernel: if the model still passes its gates under selftest,
                           the only thing left to trust when a real kernel lands is the
                           kernel itself.
"""

from __future__ import annotations

import os

import torch

_ARMED = False

_OP_FUSED = "fused_add_rms_norm_out"
_OP_PLAIN = "rms_norm_out"


def _have(name: str) -> bool:
    return hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, name)


def maybe_patch() -> None:
    global _ARMED
    mode = os.getenv("PXA_FUSED_NORM", "auto").strip().lower()
    if _ARMED or mode == "0":
        return

    from vllm import ir
    from vllm.logger import init_logger

    logger = init_logger("pxq4_vllm.ir_norm")

    have_fused, have_plain = _have(_OP_FUSED), _have(_OP_PLAIN)
    selftest = mode == "selftest"
    if not selftest and not (have_fused or have_plain):
        if mode == "1":
            logger.warning(
                "PXA_FUSED_NORM=1 but torch.ops.pxq4.%s / %s are absent; the norms will "
                "keep running as %d-kernel torch decompositions. Load a lib that "
                "provides them, or use PXA_FUSED_NORM=selftest to exercise the seam.",
                _OP_FUSED, _OP_PLAIN, 11)
        _ARMED = True
        return

    # ---- capability gate: only the shapes and dtypes the kernel is written for ----
    def _ok_common(x: torch.Tensor, weight, variance_size) -> bool:
        """WIDENED after pascal-ops #900 and confirmed against a live trace.

        The first draft required fp16 2-D x and would have fired on NOTHING for the
        models this stack actually serves. Qwen3_5 builds every norm from GemmaRMSNorm
        (qwen3_5.py:41), whose forward_native does the residual add itself and then calls
        ir.ops.rms_norm with x ALREADY IN FP32 (the residual is carried in fp32 between
        layers) and a weight that is fp32 because it is recomputed as weight.float()+1.0
        on every call. The 20 q/k norms are additionally 3-D, [T,16,256] and [T,2,256].
        So: accept fp16 or fp32 x, allow the weight dtype to differ from x, and allow
        dim > 2 since a contiguous [..., H] flattens exactly. Verified from the ARM A
        trace: rsqrt_kernel_cuda fires exactly 101.0 times per decode step, which is
        precisely the 101 GemmaRMSNorm invocations, so this gate now covers all of them.

        variance_size is still declined - a partial-variance norm is a different op and
        this model never uses one - and so is weight=None, which the kernel would have to
        special-case for no gain."""
        return (
            variance_size is None
            and weight is not None
            and x.dtype in (torch.float16, torch.float32)
            and x.is_cuda
            and x.is_contiguous()
            and x.dim() >= 2
            and x.shape[-1] % 8 == 0
        )

    def _supports_fused(x, x_residual, weight, epsilon, variance_size=None) -> bool:
        return (
            _ok_common(x, weight, variance_size)
            and x_residual.shape == x.shape
            and x_residual.is_contiguous()
        )

    def _supports_plain(x, weight, epsilon, variance_size=None) -> bool:
        return _ok_common(x, weight, variance_size)

    # ---- implementations -------------------------------------------------------
    def _fused_kernel(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        res = torch.empty_like(x_residual)
        torch.ops.pxq4.fused_add_rms_norm_out(out, res, x, x_residual, weight, epsilon)
        return out, res

    def _plain_kernel(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        torch.ops.pxq4.rms_norm_out(out, x, weight, epsilon)
        return out

    def _fused_selftest(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Deliberately the native arithmetic, in the native order. This provider must be
        # bit-identical to native by construction; it exists to prove the dispatch, not
        # to be fast.
        o = x.dtype
        f = x.to(torch.float32) + x_residual.to(torch.float32)
        res = f.to(o)
        var = f.pow(2).mean(dim=-1, keepdim=True)
        f = f * torch.rsqrt(var + epsilon)
        f = f.to(weight.dtype) * weight
        return f.to(o), res

    def _plain_selftest(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> torch.Tensor:
        o = x.dtype
        f = x.to(torch.float32)
        var = f.pow(2).mean(dim=-1, keepdim=True)
        f = f * torch.rsqrt(var + epsilon)
        f = f.to(weight.dtype) * weight
        return f.to(o)

    armed = []
    try:
        if selftest or have_fused:
            ir.ops.fused_add_rms_norm.register_impl(
                "pxa", supported=True, supports_args=_supports_fused,
            )(_fused_selftest if selftest else _fused_kernel)
            ir.ops.fused_add_rms_norm.set_default(["pxa", "native"])
            armed.append("fused_add_rms_norm")
        if selftest or have_plain:
            ir.ops.rms_norm.register_impl(
                "pxa", supported=True, supports_args=_supports_plain,
            )(_plain_selftest if selftest else _plain_kernel)
            ir.ops.rms_norm.set_default(["pxa", "native"])
            armed.append("rms_norm")
    except Exception:
        logger.exception("pxa ir norm provider registration failed; native path kept")
        _ARMED = True
        return

    _ARMED = True
    logger.info("pxa ir norm provider armed (mode=%s) on: %s",
                mode, ", ".join(armed) or "nothing")
