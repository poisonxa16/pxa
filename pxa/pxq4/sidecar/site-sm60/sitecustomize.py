# Attaches a stderr handler to the pxq4_vllm logger tree so its INFO lines are
# visible in container logs (vLLM's logging config only handles the "vllm" namespace).
import logging, sys
_lg = logging.getLogger("pxq4_vllm")
if not _lg.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("PXQ4LOG %(levelname)s %(name)s: %(message)s"))
    _lg.addHandler(_h)
    _lg.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# PASCAL PORT: Triton 3.3.1 has no Hopper programmatic-dependent-launch builtins
# (tl.extra.cuda.gdc_wait / gdc_launch_dependents), which the fork's Qwen4Exp
# hyper-connection, QSA-cache, GDN state and fused-QK-RMSNorm kernels call. The
# AttributeError is raised by Triton's dependency AST walk, before the
# `if launch_pdl:` guard is ever evaluated, so it is fatal on sm_60 regardless.
# Installed FIRST, before anything can compile a kernel.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.triton_gdc_shim  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: this image's torch.ops._C carries only the Marlin/Machete/mxfp8
# GEMMs -- no rotary_embedding. Qwen4Exp's QSA calls it. Point every rotary
# CustomOp's forward_cuda at its own forward_native BEFORE any layer is built
# (CustomOp binds _forward_method in __init__).
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.rope_native_fallback  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: the QSA attention-metadata Triton kernel cannot be lowered for
# sm_60 (PassManager::run failed at CUDA-graph capture, 2026-09-06). The fork
# has its own torch builder for the same metadata; select it on capability<7.0
# (PXA_QSA_METADATA=torch|triton|auto). Must run before any model is built.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.qsa_metadata_torch  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: the QSA paged MQA scorer's DECODE tiling (tiles_per_program=1)
# does not lower for sm_60 while the prefill tiling (8) does; use 8 for every
# row count on capability<7.0 (PXA_QSA_MQA_TILES=<n>|off). Before any model.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.qsa_mqa_paged_sm60  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: the sparse QSA attention kernel's prefill tile (BLOCK_N=64, D=256) needs 84 KB of
# shared memory; sm_6x has 48 KB (boot 14 2026-09-07 died on the first request). Cap BLOCK_N
# (PXA_QSA_SPARSE_BLOCK_N=<n>|off, default 16). Before any model.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.qsa_sparse_tiles_sm60  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: Triton 3.3.1 cannot lower an fp16 tl.dot for sm_60 (no MMA -> FMA promotion via
# tt.fp_to_fp f16->f32, which the NVIDIA lowering rejects: "Unsupported conversion from f16 to
# f16", boot 7 2026-09-06). Promote the operands to fp32 in the front end instead, for every
# kernel in the process (PXA_TRITON_DOT_FMA=auto|on|off). Before any model.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.triton_dot_fma_sm60  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: ptxas rejects atom/red .sem/.scope qualifiers and fence below sm_70 (boot 11: the
# sampler bincount's atomic_add). Rewrite Triton's PTX text for capability < 70
# (PXA_TRITON_PTX_SM60=auto|on|off). Before any kernel compiles.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.triton_ptx_sm60  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: this image's torch.ops._C has none of the QSA block top-k selectors
# (persistent_topk / cooperative_topk / qsa_lexicographic_topk; boot 9 2026-09-06). Install a
# torch stable-sort top-k under those names on sm<70 (PXA_QSA_TOPK=torch|off). Before any model.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.qsa_topk_torch  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: no _C_cache_ops.reshape_and_cache_flash in this image (boot 10 2026-09-07, KV write
# at capture). Torch two-index scatter under the same name on sm<70 (PXA_CACHE_OPS=torch|off).
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.cache_ops_torch  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# PASCAL PORT: torch 2.7 compatibility shims for torch.accelerator APIs that
# the upstream fork (written against torch 2.10) calls. Loaded in every vllm
# process via PYTHONPATH. Each shim maps to the torch.cuda equivalent.
# ---------------------------------------------------------------------------
try:
    import torch as _pxa_torch

    _acc = _pxa_torch.accelerator
    if not hasattr(_acc, "empty_cache"):
        _acc.empty_cache = _pxa_torch.cuda.empty_cache
    if not hasattr(_acc, "device_index"):
        _acc.device_index = _pxa_torch.cuda.device
    if not hasattr(_acc, "reset_peak_memory_stats"):
        _acc.reset_peak_memory_stats = _pxa_torch.cuda.reset_peak_memory_stats
    if not hasattr(_acc, "max_memory_allocated"):
        _acc.max_memory_allocated = _pxa_torch.cuda.max_memory_allocated
    if not hasattr(_acc, "memory_allocated"):
        _acc.memory_allocated = _pxa_torch.cuda.memory_allocated
    if not hasattr(_acc, "memory_reserved"):
        _acc.memory_reserved = _pxa_torch.cuda.memory_reserved
    for _name in (
        "memory_stats", "memory_summary", "mem_get_info", "memory_snapshot",
        "max_memory_reserved", "reset_accumulated_memory_stats",
        "reset_max_memory_allocated", "synchronize",
    ):
        if not hasattr(_acc, _name) and hasattr(_pxa_torch.cuda, _name):
            setattr(_acc, _name, getattr(_pxa_torch.cuda, _name))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Flash-Next checkpoint/model agreement audit. Off unless
# PXA_FLASHNEXT_AUDIT is set; the module itself decides and never raises here.
# ---------------------------------------------------------------------------
try:
    import os as _pxa_os
    if _pxa_os.environ.get("PXA_FLASHNEXT_AUDIT"):
        import pxq4_vllm.flashnext_audit  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# Qwen4Exp pipeline-parallel repair (fork defect #1545: hyper_connection_mixer
# is the literal None on non-last PP ranks, which AutoWeightsLoader cannot skip).
# Imported AFTER the audit so the repair runs OUTSIDE it -- i.e. before the load
# the audit measures. PXA_QWEN4EXP_PP_FIX=0 disables; the module never raises here.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.qwen4exp_pp_fix  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# safetensors get_tensor diagnosis + independent re-read (#1574: every worker
# dies at 0/76 with "could not determine the shape of object type
# 'torch.storage.UntypedStorage'" and the traceback never names the tensor).
# PXA_ST_FALLBACK=0 disables, =strict logs and re-raises.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.safetensors_fallback  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# FusedMoE: one pxq4 expert's slab tensor is 3-D, and vLLM reads rank 3 as
# "this is the fused all-experts stack" -- IndexError at layer 0, or half the
# panels loaded silently. PXA_MOE_PEREXPERT_FIX=0 disables.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.fused_moe_perexpert_fix  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# Qwen4Exp n-gram PLE at pipeline_parallel_size > 1 (fork wall #1595: every rank
# raises "N-gram PLE embedding currently requires pipeline_parallel_size=1").
# PP is the only legal 4-card shape for this checkpoint, so this lifts the wall
# by running the fork's OWN post-assertion statements. PXA_QWEN4EXP_PLE_PP=0
# disables; =strict raises if the fork source no longer matches.
# ---------------------------------------------------------------------------
try:
    import pxq4_vllm.qwen4exp_ple_pp  # noqa: F401
except Exception:
    pass
