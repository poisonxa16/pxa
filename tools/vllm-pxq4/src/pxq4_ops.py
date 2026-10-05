# SPDX-License-Identifier: Apache-2.0
"""torch.ops.pxq4 wrappers, fake kernels, and the preallocated workspace.

DESTINATION IN THE REPO OF PLAN 09: ``src/pxq4_vllm/ops.py``.

Component B ("runtime"), plan 09 sec.6.7.  The frozen op ABI is plan sec.7.1:

    pxq4::dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()
        out    fp16 [N, K] contiguous cuda
        slabs  uint8 [N/64, K/32, 1088] contiguous
        anchor fp16  [N/64, 64] contiguous
    pxq4::mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()
        x      fp16 [M, K] contiguous
        out    fp16 [M, N] contiguous
    pxq4::version() -> int

Namespace is ``pxq4``, never ``_C``: the host fork already owns
``torch.ops._C`` with 54 registered sm70 ops and a second TORCH_LIBRARY(_C)
in one process is a hard registration conflict.

------------------------------------------------------------------------------
DIVERGENCE FROM PLAN sec.7.3, RESOLVED IN FAVOUR OF THE FROZEN ABI
------------------------------------------------------------------------------
Plan sec.7.3 anticipated that ``k_pxq6_mmv`` consumes fp32 activations and
writes fp32 (pxq6.cuh:920-923, 968-969) and therefore that this module would
own fp32 staging buffers around the call.  Component C's delivered binding
takes fp16 x and fp16 out directly (pxq4_kernel_torch.cpp mmv_out: both
arguments are checked ``at::kHalf``), i.e. it matches the FROZEN sec.7.1 ABI
and does the conversion inside the kernel.  The staging views below are
therefore NOT used by ``linear.apply`` and are never materialized unless a
caller asks for them; ``reserve(act_elems=...)`` is accepted and recorded so
that the sec.6.7 signature is honoured, but it allocates nothing.
Keeping the API is deliberate: if the mmv is ever re-vendored closer to the
llama.cpp original (which feeds f32), the buffers are already specified here
rather than being invented inside a capture-sensitive code path.

------------------------------------------------------------------------------
WHY A WORKSPACE AT ALL
------------------------------------------------------------------------------
The prefill path is ``dequant_out`` into an fp16 [N, K] buffer followed by
``torch.mm``.  At TP=4 the largest per-rank buffer is ``mlp.gate_up_proj``
N=8704 x K=5120 fp16 = 85 MiB (TP=2: 170 MiB).  Allocating that inside
``apply()`` is legal under CUDA-graph capture (the caching allocator serves it
from the graph-private pool) but it makes the resident footprint a function of
capture order, and every captured graph pins its own block.  One arena sized
by the max over all layers, materialized once before capture, is predictable
and is what sec.6.7 mandates.

The arena is SHARED by all PXQ4 layers.  That is safe because the model
executes as one ordered stream of ops on one stream: each ``apply()`` fills the
arena and consumes it in the immediately following ``mm`` before any other
layer runs.  It is NOT safe if two PXQ4 linears are ever run concurrently on
different streams -- vLLM does not do that today, and ``PXQ4_DEQUANT_ALLOC=torch``
is the escape hatch if that ever changes.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "PXQ4_MMV_MAX_M",
    "PXQ4Workspace",
    "have_ops",
    "load_library",
    "mmv_max_m",
    "mmv_supported",
    "ops_version",
]

# --------------------------------------------------------------------------
# Library loading
# --------------------------------------------------------------------------
_LIB_ENV = "PXQ4_LIB"
_LIB_NAME = "libpxq4_sm70.so"

_lock = threading.Lock()
_loaded = False
_load_error: Exception | None = None


def _candidate_paths() -> list[Path]:
    env = os.getenv(_LIB_ENV)
    out: list[Path] = []
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    out.append(here / "_lib" / _LIB_NAME)
    out.append(here / _LIB_NAME)
    return out


def load_library(*, required: bool = False) -> bool:
    """Load ``libpxq4_sm70.so`` and register the fake (meta) kernels.

    Idempotent, thread-safe, and safe to call from every TP worker process:
    the plugin entry point runs once per process (arg_utils.py:749,
    v1/engine/core.py:108, v1/worker/worker_base.py:247) and each of those is a
    distinct interpreter, so there is no cross-process double registration.
    Within one process the ``_loaded`` flag is what prevents a second
    ``torch.ops.load_library`` (which would raise on duplicate schema defs).

    Returns True if ``torch.ops.pxq4`` is usable.  With ``required=False`` a
    missing .so is not fatal -- CPU-only unit tests and the offline converter
    import this module without a GPU or a build.
    """
    global _loaded, _load_error
    with _lock:
        if _loaded:
            return True
        if hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, "dequant_out"):
            # Someone (a test harness, or a previous partial import) already
            # registered the schema. Do not load again.
            _loaded = True
            _register_fakes()
            _reconcile_mmv_max_m()
            return True

        tried: list[str] = []
        for path in _candidate_paths():
            tried.append(str(path))
            if not path.is_file():
                continue
            torch.ops.load_library(str(path))
            _loaded = True
            _load_error = None
            _register_fakes()
            _reconcile_mmv_max_m()
            logger.info("pxq4: loaded %s (op version %d)", path, ops_version())
            return True

        _load_error = FileNotFoundError(
            f"pxq4: could not find {_LIB_NAME}. Set {_LIB_ENV} or place it in "
            f"src/pxq4_vllm/_lib/. Tried: {tried}"
        )
        if required:
            raise _load_error
        logger.warning("pxq4: %s", _load_error)
        return False


def have_ops() -> bool:
    return _loaded and hasattr(torch.ops, "pxq4")


_fakes_registered = False


def _register_fakes() -> None:
    """Register meta implementations for the two mutating ops.

    MANDATORY, not optional: vLLM compiles the model with
    ``cudagraph_mode=FULL_AND_PIECEWISE`` and the piecewise splitting_ops list
    (compilation.py:764-773) means our ops are traced by dynamo/inductor.  A
    custom op with no fake kernel fails tracing before capture is even
    attempted.  The ops mutate argument 0 and return nothing, so the fakes are
    pure shape checks -- the ``Tensor(a!)`` annotation in the C++ schema is
    what tells functionalization that ``out`` is written in place.
    """
    global _fakes_registered
    if _fakes_registered:
        return
    if not hasattr(torch.library, "register_fake"):  # torch < 2.4
        logger.warning("pxq4: torch.library.register_fake missing; torch.compile "
                       "tracing of pxq4 ops will fail")
        return

    @torch.library.register_fake("pxq4::dequant_out")
    def _dequant_out_meta(out, slabs, anchor):  # noqa: ANN001, ANN202
        torch._check(slabs.dim() == 3 and anchor.dim() == 2)
        torch._check(slabs.shape[0] == anchor.shape[0])
        torch._check(out.dim() == 2)
        torch._check(out.shape[0] == slabs.shape[0] * 64)
        torch._check(out.shape[1] == slabs.shape[1] * 32)
        return None

    @torch.library.register_fake("pxq4::mmv_out")
    def _mmv_out_meta(out, x, slabs, anchor):  # noqa: ANN001, ANN202
        torch._check(slabs.dim() == 3 and anchor.dim() == 2)
        torch._check(x.dim() == 2 and out.dim() == 2)
        torch._check(x.shape[1] == slabs.shape[1] * 32)
        torch._check(out.shape[0] == x.shape[0])
        torch._check(out.shape[1] == slabs.shape[0] * 64)
        return None

    if hasattr(torch.ops.pxq4, "linear_out"):
        @torch.library.register_fake("pxq4::linear_out")
        def _linear_out_meta(out, x, slabs, anchor):  # noqa: ANN001, ANN202
            torch._check(slabs.dim() == 3 and anchor.dim() == 2)
            torch._check(x.dim() == 2 and out.dim() == 2)
            torch._check(x.shape[1] == slabs.shape[1] * 32)
            torch._check(out.shape[0] == x.shape[0])
            torch._check(out.shape[1] == slabs.shape[0] * 64)
            return None

    _fakes_registered = True


def ops_version() -> int:
    if not have_ops():
        return 0
    return int(torch.ops.pxq4.version())


# --------------------------------------------------------------------------
# mmv policy
# --------------------------------------------------------------------------
# Token-count ceiling for the mmv (matrix x vector-batch) path.  Above it we
# dequantize and hand the GEMM to cuBLAS.  Default 8 mirrors the engine's
# PXA_PXQ4_2D_MAX_NY (ggml-cuda.cu:4019-4021), which is the shape the kernel
# was tuned for: grid.y is the token axis, so each block re-reads its whole
# panel once per token and the weight-traffic advantage evaporates as M grows.
# Plan risk 4 is precisely that the real crossover may sit below vLLM's median
# decode batch; the env var exists so G8/G9 can sweep it without a rebuild.
def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


PXQ4_MMV_MAX_M: int = _env_int("PXQ4_MMV_MAX_M", 8) or 8
_mmv_max_m_from_env = os.getenv("PXQ4_MMV_MAX_M") not in (None, "")


def _reconcile_mmv_max_m() -> None:
    """Adopt the .so's compiled-in default unless the operator overrode it."""
    global PXQ4_MMV_MAX_M
    if _mmv_max_m_from_env or not have_ops():
        return
    if hasattr(torch.ops.pxq4, "mmv_max_m"):
        PXQ4_MMV_MAX_M = int(torch.ops.pxq4.mmv_max_m())


def mmv_max_m() -> int:
    return PXQ4_MMV_MAX_M


def mmv_supported(K: int) -> bool:
    """Can the mmv kernel serve this K on this device?

    The kernel stages activations in dynamic shared memory; component C stays
    inside the 48 KiB sm_70 no-opt-in budget by chunking
    (pxq4_kernel.cu:24-35).  The authority is the .so, so we ask it.

    With no .so loaded we return False rather than guessing: an over-optimistic
    guess would route a layer into a kernel that cannot launch, and there is no
    Python-side formula that is guaranteed to track the kernel's chunking.
    """
    if K <= 0 or K % 32 != 0:
        return False
    if not have_ops() or not hasattr(torch.ops.pxq4, "mmv_supported"):
        return False
    return bool(torch.ops.pxq4.mmv_supported(int(K)))


def upload_tables(book, sub) -> None:
    """Push the checkpoint's own book/sub tables to the device.

    EAGER ONLY -- cudaMemcpyToSymbol; must run before any graph capture.
    Called from ``PXQ4LinearMethod.process_weights_after_loading`` on the first
    layer processed, i.e. after loading and before the compile/capture phase.

    WHY it is not optional: PXA_PXQ6_BOOK / PXA_PXQ6_SUB can override the
    frozen literals at quantize time, and the file records what was used
    (gguf KVs pxa.pxq6.book / pxa.pxq6.sub -> config.json).  A checkpoint is
    only self-describing if we honour what it recorded.
    """
    if not have_ops() or not hasattr(torch.ops.pxq4, "set_tables"):
        return
    b = torch.as_tensor(list(book), dtype=torch.float32)
    s = torch.as_tensor(list(sub), dtype=torch.float32)
    torch.ops.pxq4.set_tables(b, s)


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
class _DeviceArena:
    __slots__ = ("dequant", "act_f32", "out_f32")

    def __init__(self) -> None:
        self.dequant: torch.Tensor | None = None
        self.act_f32: torch.Tensor | None = None
        self.out_f32: torch.Tensor | None = None


class PXQ4Workspace:
    """Per-device scratch, sized in ``create_weights`` and allocated once.

    Contract (plan sec.6.7):
      * ``reserve()``   -- called from every ``create_weights``; takes the max.
      * ``materialize()`` -- called once per device from
        ``process_weights_after_loading``, i.e. after all weights are loaded and
        before torch.compile / cudagraph capture.
      * ``dequant_view()`` -- returns an fp16 [N, K] view; allocates nothing.

    Growing the arena after the first view would invalidate any captured graph
    that baked in the old pointer, so ``reserve`` raises once frozen.
    """

    _reserved_dequant: int = 0
    _reserved_act: int = 0
    _frozen: bool = False
    _arenas: dict[str, _DeviceArena] = {}
    _mode: str = os.getenv("PXQ4_DEQUANT_ALLOC", "workspace")

    # ---------------------------------------------------------------- sizing
    @classmethod
    def reserve(cls, *, dequant_elems: int, act_elems: int = 0) -> None:
        if dequant_elems < 0 or act_elems < 0:
            raise ValueError("pxq4: negative workspace reservation")
        if cls._frozen and (
            dequant_elems > cls._reserved_dequant or act_elems > cls._reserved_act
        ):
            raise RuntimeError(
                "pxq4: workspace reservation grew after materialization "
                f"(want dequant={dequant_elems}, have {cls._reserved_dequant}). "
                "All create_weights calls must complete before the first "
                "process_weights_after_loading; see ops.py PXQ4Workspace."
            )
        cls._reserved_dequant = max(cls._reserved_dequant, int(dequant_elems))
        cls._reserved_act = max(cls._reserved_act, int(act_elems))

    @classmethod
    def reserved(cls) -> tuple[int, int]:
        return cls._reserved_dequant, cls._reserved_act

    # ----------------------------------------------------------- allocation
    @classmethod
    def _key(cls, device: torch.device) -> str:
        return f"{device.type}:{device.index if device.index is not None else 0}"

    @classmethod
    def materialize(cls, device: torch.device) -> None:
        if cls._mode == "torch":
            # Escape hatch: allocate per call from the caching allocator.
            cls._frozen = True
            return
        key = cls._key(device)
        arena = cls._arenas.get(key)
        if arena is None:
            arena = _DeviceArena()
            cls._arenas[key] = arena
        if arena.dequant is None and cls._reserved_dequant > 0:
            arena.dequant = torch.empty(
                cls._reserved_dequant, dtype=torch.float16, device=device
            )
            logger.info(
                "pxq4: dequant workspace %.1f MiB on %s",
                cls._reserved_dequant * 2 / (1 << 20),
                key,
            )
        cls._frozen = True

    @classmethod
    def _arena_for(cls, device: torch.device) -> _DeviceArena:
        key = cls._key(device)
        arena = cls._arenas.get(key)
        if arena is None:
            arena = _DeviceArena()
            cls._arenas[key] = arena
        return arena

    # ---------------------------------------------------------------- views
    @classmethod
    def _resolve_device(cls, device: "torch.device | None") -> torch.device:
        # Plan sec.6.6 calls dequant_view(N, K) with no device; a TP worker owns
        # exactly one device, so the current device is the right default. The
        # explicit argument exists so a test (or a future multi-device host) can
        # be unambiguous.
        if device is not None:
            return device
        return torch.device("cuda", torch.cuda.current_device())

    @classmethod
    def dequant_view(cls, N: int, K: int, device: "torch.device | None" = None) -> torch.Tensor:
        """fp16 [N, K] contiguous scratch for ``dequant_out``.

        No allocation on the workspace path -- the arena was sized by the max
        over all layers in ``create_weights``.
        """
        need = int(N) * int(K)
        device = cls._resolve_device(device)
        if cls._mode == "torch":
            return torch.empty((N, K), dtype=torch.float16, device=device)
        arena = cls._arena_for(device)
        if arena.dequant is None or arena.dequant.numel() < need:
            raise RuntimeError(
                f"pxq4: dequant workspace not materialized or too small "
                f"(need {need} fp16 elems, have "
                f"{0 if arena.dequant is None else arena.dequant.numel()}). "
                "PXQ4LinearMethod.process_weights_after_loading must run before "
                "the first forward."
            )
        return arena.dequant[:need].view(N, K)

    @classmethod
    def act_f32_view(cls, M: int, K: int, device: "torch.device | None" = None) -> torch.Tensor:
        """fp32 [M, K] staging. Unused by the fp16 ABI (see module docstring);
        allocated lazily and therefore EAGER-ONLY."""
        device = cls._resolve_device(device)
        return cls._staging(cls._arena_for(device), "act_f32", M, K, device)

    @classmethod
    def out_f32_view(cls, M: int, N: int, device: "torch.device | None" = None) -> torch.Tensor:
        """fp32 [M, N] staging. Unused by the fp16 ABI; EAGER-ONLY."""
        device = cls._resolve_device(device)
        return cls._staging(cls._arena_for(device), "out_f32", M, N, device)

    @classmethod
    def _staging(
        cls, arena: _DeviceArena, slot: str, a: int, b: int, device: torch.device
    ) -> torch.Tensor:
        need = int(a) * int(b)
        buf = getattr(arena, slot)
        if buf is None or buf.numel() < need:
            buf = torch.empty(max(need, cls._reserved_act), dtype=torch.float32,
                              device=device)
            setattr(arena, slot, buf)
        return buf[:need].view(a, b)

    # ------------------------------------------------------------- teardown
    @classmethod
    def reset(cls) -> None:
        """Drop every arena. Tests only -- never call on a live engine."""
        cls._arenas.clear()
        cls._reserved_dequant = 0
        cls._reserved_act = 0
        cls._frozen = False
