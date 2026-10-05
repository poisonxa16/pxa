"""Rewrite Triton's PTX for sm_6x: drop memory-ordering and scope qualifiers Pascal cannot assemble.

Boot 11 of Flash-Next on the four P100s (2026-09-07 00:22) captured CUDA graphs and then the last PP
rank died in ptxas: "Feature '.acq_rel' requires .target sm_70 or higher" (sampler penalty bincount,
tl.atomic_add). Forcing sem="relaxed" in the front end is not enough: ".relaxed requires sm_70" too.
PTX ISA: the .sem and .scope qualifiers on atom/red, and the fence instruction, all need sm_70. On
sm_6x the plain forms are the only valid ones and mean exactly what Pascal hardware does (monotonic
atomics, membar fences).

So post-process the PTX text Triton produces, only when the target is below sm_70:
  atom.<sem>.<scope>.<rest>  ->  atom.<rest>        red.<sem>.<scope>.<rest> -> red.<rest>
  fence.<sem>.gpu;           ->  membar.gl;          fence.<sem>.cta; -> membar.cta;  fence.<sem>.sys; -> membar.sys;
  nanosleep.u32 N;           ->  (dropped; sm_70+ only, spin-wait hint)

Installed by wrapping CUDABackend.make_ptx (triton/backends/nvidia/compiler.py). PXA_TRITON_PTX_SM60=auto|on|off
"""

import logging
import os
import re

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_TRITON_PTX_SM60", "auto").strip().lower()

_SEM = {"relaxed", "acquire", "release", "acq_rel", "sc"}
_SCOPE = {"cta", "gpu", "sys", "cluster"}
# the whole mnemonic of an atom/red instruction: "atom.global.gpu.relaxed.add.u32" (any qualifier order)
_MNEMONIC = re.compile(r"\b(atom|red)((?:\.[A-Za-z0-9_]+)+)")
_FENCE = re.compile(r"\bfence(?:\.(?:relaxed|acquire|release|acq_rel|sc))?\.(cta|gpu|sys|cluster)\s*;")
_NANOSLEEP = re.compile(r"\bnanosleep\.u32\s+[^;]+;")
_MEMBAR = {"cta": "membar.cta;", "gpu": "membar.gl;", "sys": "membar.sys;", "cluster": "membar.gl;"}


def rewrite_ptx(ptx: str) -> tuple[str, int]:
    n = 0

    def strip(m):
        nonlocal n
        parts = m.group(2).split(".")[1:]
        kept = [q for q in parts if q not in _SEM and q not in _SCOPE]
        if len(kept) != len(parts):
            n += 1
        return m.group(1) + "." + ".".join(kept)

    out = _MNEMONIC.sub(strip, ptx)
    out, k = _FENCE.subn(lambda m: _MEMBAR[m.group(1)], out)
    n += k
    out, k = _NANOSLEEP.subn("", out)
    n += k
    return out, n


def _wanted() -> bool:
    if _MODE in ("off", "0", "no"):
        return False
    if _MODE in ("on", "1", "yes", "force"):
        return True
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major < 7
    except Exception:
        return False


def _install() -> None:
    if not _wanted():
        log.info("triton_ptx_sm60: not active (mode=%s)", _MODE)
        return
    # Triton loads backends/nvidia/compiler.py through spec_from_file_location WITHOUT registering
    # it in sys.modules, so `import triton.backends.nvidia.compiler` yields a SECOND copy of the class
    # that the compiler never calls. Patch the class the registry actually uses.
    classes = []
    try:
        from triton.backends import backends
        classes.append(backends["nvidia"].compiler)
    except Exception as exc:
        log.debug("triton_ptx_sm60: triton backend registry not available (%s)", exc)
    try:
        from triton.backends.nvidia.compiler import CUDABackend
        if CUDABackend not in classes:
            classes.append(CUDABackend)
    except Exception:
        pass
    if not classes:
        return
    for CUDABackend in classes:
        _patch(CUDABackend)


def _patch(CUDABackend) -> None:
    orig = CUDABackend.make_ptx
    if getattr(orig, "_pxa_sm60_ptx", False):
        return

    def make_ptx(self, src, metadata, opt, capability):
        ptx = orig(self, src, metadata, opt, capability)
        if capability >= 70:
            return ptx
        out, n = rewrite_ptx(ptx)
        if n:
            log.debug("triton_ptx_sm60: make_ptx capability=%s rewrote %d qualifier(s)", capability, n)
        return out

    make_ptx._pxa_sm60_ptx = True
    make_ptx.__wrapped__ = orig
    CUDABackend.make_ptx = make_ptx
    log.info("triton_ptx_sm60: PTX rewrite armed on %s.%s (atom/red sem+scope, fence->membar, nanosleep) for capability < 70",
             CUDABackend.__module__, CUDABackend.__name__)


_install()
