"""QSA attention-metadata builder on Pascal: use the fork's own torch builder, not the Triton kernel.

CUDA-graph capture on the four P100s died in

    vllm/models/qwen4_exp/common/qsa_cache.py:421  build_qsa_metadata_triton
      -> _build_qsa_metadata_kernel      Triton 3.3.1 make_ttgir: PassManager::run failed (capability 60)

reached via mamba_hybrid.prepare_attn -> attn_utils.build_attn_metadata ->
backend.build_for_cudagraph_capture (boot-pp4.dockerlog:1150-1162, 2026-09-06). Triton cannot lower
that kernel for sm_60 (it never could; eager boots never reached capture, so nobody saw it).

The fork ALREADY carries the reference implementation of the same computation --
`_build_qsa_metadata_torch` in the same file, selected at import time by

    build_qsa_metadata = build_qsa_metadata_triton if HAS_TRITON else _build_qsa_metadata_torch

HAS_TRITON is true in this image (Triton is present and other kernels need it), so the Triton
builder is chosen and fails on this hardware. This provider re-points the module-level name at the
torch builder, before any model is constructed, so capture and every later step use the reference
math. It is integer metadata (token->request search, logical positions, slot mapping, work items):
the two builders are two implementations of one specification, and the torch one is the one the
fork itself falls back to when Triton is absent.

PXA_QSA_METADATA=torch   force the torch builder (the sm60 default set by sitecustomize)
PXA_QSA_METADATA=triton  leave the fork's choice alone
PXA_QSA_METADATA=auto    torch when the current device's compute capability is < 7.0, else triton
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_QSA_METADATA", "auto").strip().lower()


def _want_torch() -> bool:
    if _MODE in ("torch", "1", "on", "force"):
        return True
    if _MODE in ("triton", "0", "off"):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return True
        major, _minor = torch.cuda.get_device_capability()
        return major < 7
    except Exception:
        return True


def _install() -> None:
    if not _want_torch():
        log.info("qsa_metadata_torch: leaving the fork's Triton QSA metadata builder in place (mode=%s)", _MODE)
        return
    try:
        from vllm.models.qwen4_exp.common import qsa_cache
    except Exception as exc:  # the module is only present in the qwen4_exp fork
        log.debug("qsa_metadata_torch: qsa_cache not importable (%s); nothing to do", exc)
        return
    torch_builder = getattr(qsa_cache, "_build_qsa_metadata_torch", None)
    if torch_builder is None:
        log.warning("qsa_metadata_torch: the fork has no _build_qsa_metadata_torch; cannot install")
        return
    if getattr(qsa_cache, "build_qsa_metadata", None) is torch_builder:
        return
    qsa_cache.build_qsa_metadata = torch_builder
    # anyone who did `from ...qsa_cache import build_qsa_metadata` before us would hold the old name;
    # the fork resolves it at module level and the one call site (qsa_cache.py:690) reads the module
    # global, so this is the only binding that matters -- but say so if that ever changes.
    import sys
    stale = [m for m, mod in list(sys.modules.items())
             if mod is not None and m != qsa_cache.__name__
             and getattr(mod, "build_qsa_metadata", None) is getattr(qsa_cache, "build_qsa_metadata_triton", None)]
    for m in stale:
        setattr(sys.modules[m], "build_qsa_metadata", torch_builder)
    log.info("qsa_metadata_torch: QSA metadata builder -> torch reference (%s)%s", _MODE,
             (" also rebound in " + ", ".join(stale)) if stale else "")


_install()
