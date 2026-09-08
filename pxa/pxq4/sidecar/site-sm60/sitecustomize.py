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
# PASCAL PORT: torch 2.7 compatibility shims for torch.accelerator APIs that
# the 1cat fork (written against torch 2.10) calls. Loaded in every vllm
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
