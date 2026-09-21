# SPDX-License-Identifier: Apache-2.0
"""PXQ4 plugin entry point.

DESTINATION IN THE REPO OF PLAN 09: ``src/pxq4_vllm/__init__.py``.

Plan 09 sec.9 assigns ``__init__.py`` to component B (runtime).  This file is
the *registration* half of it, which belongs to the quant-config component;
merge it into B's ``__init__.py`` rather than shipping both.

How the hook works (all read in the vLLM fork checkout, git 2ceb15066):

  * ``vllm/plugins/__init__.py:14`` declares the group name
    ``vllm.general_plugins`` and ``:28-68`` enumerates it with
    ``importlib.metadata.entry_points(group=...)``.  Because that API scans
    ``.dist-info`` directories found on ``sys.path``, a hand-written
    ``pxq4_vllm-0.1.0.dist-info/entry_points.txt`` under
    ``$PXA_MODELS_DIR/pxa-vllm-pxq4/site`` plus ``PYTHONPATH`` is enough -- nothing
    needs to be pip-installed into the container image, whose ``/`` is 100%
    full.
  * ``load_general_plugins()`` runs in every process that builds a config or a
    model: ``arg_utils.py:749`` (API server), ``v1/engine/core.py:108``
    (engine core) and ``v1/worker/worker_base.py:247`` (each TP worker).
    ``plugins_loaded`` (``plugins/__init__.py:25``) makes it once-per-process.
  * ``VLLM_PLUGINS`` can restrict which plugins load
    (``plugins/__init__.py:31,57``); if it is set for any reason, "pxq4" must
    be in it.

Nothing here patches vLLM.  The only side effect is the
``@register_quantization_config("pxq4")`` decorator firing on import of
``.config``.
"""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """``vllm.general_plugins`` entry point.

    Importing ``.config`` runs the ``@register_quantization_config("pxq4")``
    decorator, which appends "pxq4" to the runtime ``QUANTIZATION_METHODS``
    list and stores the class in ``_CUSTOMIZED_METHOD_TO_QUANT_CONFIG``
    (quantization/__init__.py:92-101).

    Idempotent by three independent mechanisms, because this runs once in the
    engine-core process and once in every TP worker:
      1. ``plugins_loaded`` in vllm/plugins/__init__.py:25;
      2. Python's module cache -- the decorator only fires on first import;
      3. the ``_REGISTERED`` flag here, for direct callers.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Import for side effect (the decorator). Deliberately not re-exported at
    # module scope: this module is imported very early, before ModelConfig
    # exists, and pulling torch/vllm layer modules in at that point is what
    # quantization/__init__.py:108 explicitly avoids.
    from . import config as _config  # noqa: F401,PLC0415

    # sm_60 fp16 dense decode fast path (plugin-side patch; PXA_SM60_F16_MMV=0 disables).
    # Guarded: a failure here must never take registration down with it.
    try:
        from . import pxa_sm60_f16 as _f16  # noqa: PLC0415
        _f16.maybe_patch()
    except Exception:  # pragma: no cover
        import logging
        logging.getLogger("pxq4_vllm").exception("pxa_sm60_f16 patch failed; cuBLAS path kept")

    # Tiled online-softmax prefill SDPA for the Pascal backend (kills the
    # O(chunk x ctx) fp32 scores OOM class; PXA_SDPA_TILED=0 disables). Guarded.
    try:
        from . import pxa_sdpa_tiled as _sdpat  # noqa: PLC0415
        _sdpat.maybe_patch()
    except Exception:  # pragma: no cover
        import logging
        logging.getLogger("pxq4_vllm").exception("pxa_sdpa_tiled patch failed; one-shot kept")

    # Hand-fused Pascal/Volta norms and activations (PXA_OPS_FUSED; default 0 = off).
    # vLLM ships these ops as decompositions and expects Inductor to fuse them; on sm_60
    # there is no Inductor, so ~1600 kernels of every decode token go on small-op
    # launches. Guarded: a failure here must never take registration down with it.
    try:
        from . import pxa_pascal_ops as _pops  # noqa: PLC0415
        _pops.maybe_patch()
    except Exception:  # pragma: no cover
        import logging
        logging.getLogger("pxq4_vllm").exception(
            "pxa_pascal_ops patch failed; the torch decompositions are kept")

    # Per-step decode instrument: host wall vs GPU busy vs gap . OFF unless
    # PXA_STEP_TIMER=1, so this is inert on every shipped boot. Guarded.
    try:
        from . import pxa_step_timer as _stept  # noqa: PLC0415
        _stept.maybe_patch()
    except Exception:  # pragma: no cover
        import logging
        logging.getLogger("pxq4_vllm").exception("pxa_step_timer patch failed; ignored")

    # rms_norm / fused_add_rms_norm provider for the vllm.ir priority registry .
    # Inert unless a lib supplies the kernel (PXA_FUSED_NORM=auto is the default), and it
    # declines any arg shape it was not written for rather than guessing. Guarded.
    try:
        from . import pxa_ir_norm as _irn  # noqa: PLC0415
        _irn.maybe_patch()
    except Exception:  # pragma: no cover
        import logging
        logging.getLogger("pxq4_vllm").exception("pxa_ir_norm patch failed; native norms kept")

    _REGISTERED = True


__all__ = ["register"]
