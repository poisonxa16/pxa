# SPDX-License-Identifier: Apache-2.0
"""PXQ4 plugin entry point.

DESTINATION IN THE REPO OF PLAN 09: ``src/pxq4_vllm/__init__.py``.

Plan 09 sec.9 assigns ``__init__.py`` to component B (runtime).  This file is
the *registration* half of it, which belongs to the quant-config component;
merge it into B's ``__init__.py`` rather than shipping both.

How the hook works (all read in /opt/1Cat-vLLM, git 2ceb15066):

  * ``vllm/plugins/__init__.py:14`` declares the group name
    ``vllm.general_plugins`` and ``:28-68`` enumerates it with
    ``importlib.metadata.entry_points(group=...)``.  Because that API scans
    ``.dist-info`` directories found on ``sys.path``, a hand-written
    ``pxq4_vllm-0.1.0.dist-info/entry_points.txt`` under
    ``/mnt/models/pxa-vllm-pxq4/site`` plus ``PYTHONPATH`` is enough -- nothing
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

    _REGISTERED = True


__all__ = ["register"]
