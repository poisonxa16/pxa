"""Rotary embedding for a vLLM build whose _C extension does not have the kernel.

This image's `torch.ops._C` carries 16 ops -- the Marlin/Machete/mxfp8 GEMM set --
and none of the elementwise ones. The Pascal port already supplies its own
providers for the norms and activations (pxa_pascal_ops, pxa_ir_norm); rope had no
provider because nothing served here reached it. Qwen4Exp does, in QSA:

    vllm/models/qwen4_exp/nvidia/qsa.py:385   query, key = self.rotary_emb(...)
    vllm/model_executor/layers/rotary_embedding/base.py:244  forward_cuda
    vllm/_custom_ops.py:302                   torch.ops._C.rotary_embedding(...)
    AttributeError: '_OpNamespace' '_C' object has no attribute 'rotary_embedding'

vLLM's rotary classes are CustomOps and every one of them ships a `forward_native`
that is the reference implementation of the very same math -- the CUDA path exists
to fuse it, not to change it. So when the op is absent, point `forward_cuda` at
`forward_native` for every rotary class that has both.

This must happen before any layer is constructed: CustomOp binds `_forward_method`
to the bound `forward_cuda` in `__init__`, so a later patch would not be seen by
layers that already exist. sitecustomize imports this at process start.

PXA_ROPE_NATIVE=0 disables; =force patches even when the op is present;
=strict raises if nothing could be patched.
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_ROPE_NATIVE", "auto").strip().lower()


def _install() -> None:
    if _MODE in ("0", "off", "no", "false"):
        return

    import importlib
    import inspect
    import pkgutil

    import torch

    have_op = hasattr(torch.ops._C, "rotary_embedding")
    if have_op and _MODE != "force":
        log.info("rope_native_fallback: torch.ops._C.rotary_embedding is present; "
                 "leaving the fused path alone")
        return

    pkg = importlib.import_module("vllm.model_executor.layers.rotary_embedding")
    mods = [pkg]
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            mods.append(importlib.import_module(f"{pkg.__name__}.{info.name}"))
        except Exception as exc:
            log.debug("rope_native_fallback: skipping %s (%s)", info.name, exc)

    patched, seen = [], set()
    for mod in mods:
        for name, obj in vars(mod).items():
            if not inspect.isclass(obj) or obj in seen:
                continue
            seen.add(obj)
            # only rotary classes: CustomOp itself, and the norms/activations that
            # subclass it, are imported into these namespaces and must NOT be
            # dragged onto the native path -- the Pascal port has fused providers
            # for them.
            if not getattr(obj, "__module__", "").startswith(pkg.__name__):
                continue
            fwd_cuda = obj.__dict__.get("forward_cuda")
            if fwd_cuda is None or not hasattr(obj, "forward_native"):
                continue
            obj.forward_cuda = obj.forward_native
            patched.append(f"{obj.__module__.rsplit('.', 1)[-1]}.{obj.__name__}")

    if not patched:
        msg = "rope_native_fallback: found no rotary class with both forward_cuda and forward_native"
        if _MODE == "strict":
            raise RuntimeError(msg)
        log.warning(msg)
        return

    log.info("rope_native_fallback: torch.ops._C.rotary_embedding is MISSING "
             "(_C has %d ops); forward_cuda -> forward_native on %d class(es): %s",
             len(dir(torch.ops._C)), len(patched), ", ".join(sorted(patched)))


try:
    _install()
except Exception as exc:  # pragma: no cover
    if _MODE == "strict":
        raise
    log.warning("rope_native_fallback: not installed (%s)", exc)
