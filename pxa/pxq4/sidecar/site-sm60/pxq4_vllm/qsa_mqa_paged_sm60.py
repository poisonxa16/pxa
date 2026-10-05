"""QSA paged MQA scorer on Pascal: compile the decode shape the way the prefill shape compiled.

Boot 6 of Flash-Next under vLLM on the four P100s (2026-09-06 22:27) got past the QSA metadata
builder (qsa_metadata_torch) and died one kernel later, at CUDA-graph capture:

    vllm/models/qwen4_exp/nvidia/ops/qsa.py:1235 qsa_select_paged_tokens
      -> ops/qsa.py:914 qsa_mqa_paged -> _qsa_mqa_paged_kernel
      Triton 3.3.1 make_ttgir: PassManager::run failed (capability 60)

The SAME kernel compiled and ran during the profile run (boot 4 completed a full forward pass),
where the wrapper picks `tiles_per_program = 8` (rows > 32). Capture runs the decode shape,
rows <= 32, where the wrapper picks `tiles_per_program = 1` -- a different constexpr set, and the
one Triton cannot lower for sm_60. The kernel loops over `tiles_per_program` column tiles per
program; running 8 tiles per program on a small batch is legal (the grid shrinks, the results are
identical), it just changes the launch geometry. So on capability < 7.0 use the prefill tiling for
every row count.

Implemented the way the package's earlier providers did: the wrapper's own source is re-compiled with
one textual edit and rebound in the module, so there is no second implementation to drift.

PXA_QSA_MQA_TILES=<n>   force tiles_per_program for all row counts (default 8 on sm<70, off elsewhere)
PXA_QSA_MQA_TILES=off   leave the fork's choice alone
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_QSA_MQA_TILES", "auto").strip().lower()
_NEEDLE = "tiles_per_program = 1 if q.shape[0] <= 32 else 8"


def _tiles() -> int | None:
    if _MODE in ("off", "0", "no", "triton"):
        return None
    if _MODE.isdigit():
        return int(_MODE)
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, _minor = torch.cuda.get_device_capability()
        return 8 if major < 7 else None
    except Exception:
        return None


def _install() -> None:
    tiles = _tiles()
    if tiles is None:
        log.info("qsa_mqa_paged_sm60: leaving the fork's tiling alone (mode=%s)", _MODE)
        return
    try:
        import inspect
        from vllm.models.qwen4_exp.nvidia.ops import qsa
    except Exception as exc:
        log.debug("qsa_mqa_paged_sm60: ops.qsa not importable (%s)", exc)
        return
    fn = getattr(qsa, "qsa_mqa_paged", None)
    if fn is None:
        log.warning("qsa_mqa_paged_sm60: no qsa_mqa_paged in ops.qsa")
        return
    try:
        src = inspect.getsource(fn)
    except Exception as exc:
        log.warning("qsa_mqa_paged_sm60: cannot read the wrapper source (%s); not patched", exc)
        return
    if _NEEDLE not in src:
        log.warning("qsa_mqa_paged_sm60: the tiling line moved (needle not found); not patched -- "
                    "update _NEEDLE against ops/qsa.py")
        return
    import textwrap
    patched = textwrap.dedent(src).replace(_NEEDLE, "tiles_per_program = %d" % tiles, 1)
    ns = qsa.__dict__
    exec(compile(patched, getattr(qsa, "__file__", "ops/qsa.py") + "#pxa-sm60-tiles", "exec"), ns)
    qsa.qsa_mqa_paged = ns["qsa_mqa_paged"]
    # callers inside the same module read the module global; anyone who imported the name
    # elsewhere gets rebound too
    import sys
    for m, mod in list(sys.modules.items()):
        if mod is not None and m != qsa.__name__ and getattr(mod, "qsa_mqa_paged", None) is fn:
            setattr(mod, "qsa_mqa_paged", qsa.qsa_mqa_paged)
    log.info("qsa_mqa_paged_sm60: paged MQA scorer uses tiles_per_program=%d for every row count (sm<70)", tiles)


_install()
