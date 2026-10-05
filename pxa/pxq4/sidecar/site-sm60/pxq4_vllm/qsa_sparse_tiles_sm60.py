"""Sparse QSA attention tiles that fit Pascal's 48 KB of shared memory.

Boot 14 of Flash-Next on the four P100s (2026-09-07 01:57) served /health and died on the FIRST request:

    _qsa_sparse_paged_gqa_splitk_kernel: triton.runtime.errors.OutOfResources: out of resource:
    shared memory, Required: 86016, Hardware limit: 49152

ops/qsa.py _qsa_sparse_launch_profile() picks BLOCK_N=64 for prefill-sized program counts (tuned on
GB300, 228 KB smem; V100 has 96 KB). With HEAD_DIM=256 fp16 a 64-column K tile plus V tile is ~64 KB
before Q and partials. sm_6x has 48 KB. Cap BLOCK_N on capability < 7.0 (default 16: K+V tiles ~16 KB;
PXA_QSA_SPARSE_BLOCK_N=32 is the next thing to try once serving is proven) and use four warps like the
SM70 route. Results are identical: BLOCK_N is a tiling constant of a split-K reduction with fp32 partials.

PXA_QSA_SPARSE_BLOCK_N=<n>|off
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_QSA_SPARSE_BLOCK_N", "auto").strip().lower()


def _cap() -> int | None:
    if _MODE in ("off", "0", "no"):
        return None
    if _MODE.isdigit():
        return int(_MODE)
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, _minor = torch.cuda.get_device_capability()
        return 16 if major < 7 else None
    except Exception:
        return None


def _install() -> None:
    cap = _cap()
    if cap is None:
        log.info("qsa_sparse_tiles_sm60: leaving the fork's sparse-attention tiles alone (mode=%s)", _MODE)
        return
    try:
        from vllm.models.qwen4_exp.nvidia.ops import qsa
    except Exception as exc:
        log.debug("qsa_sparse_tiles_sm60: ops.qsa not importable (%s)", exc)
        return
    orig = getattr(qsa, "_qsa_sparse_launch_profile", None)
    if orig is None or getattr(orig, "_pxa_sm60_tiles", False):
        return

    def profile(base_programs, block_m, is_sm70):
        block_n, target_splits, partial_warps = orig(base_programs, block_m, is_sm70)
        if block_n > cap:
            block_n = cap
            partial_warps = max(partial_warps, 4)
        return block_n, target_splits, partial_warps

    profile._pxa_sm60_tiles = True
    profile.__wrapped__ = orig
    qsa._qsa_sparse_launch_profile = profile
    log.info("qsa_sparse_tiles_sm60: sparse QSA BLOCK_N capped at %d (48 KB smem) on sm<70", cap)


_install()
