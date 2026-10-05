"""Programmatic-dependent-launch builtins for a Triton that does not have them.

The fork's Qwen4Exp kernels (hyper-connection ops, the QSA cache, the mamba/GDN
state gather-scatter, fused QK RMSNorm, fused-MoE utils) call

    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()

which are Hopper *programmatic dependent launch* (PDL) primitives.  Triton 3.3.1 --
the version in this image -- has neither, so every one of those kernels dies with

    AttributeError: module 'triton.language.extra.cuda' has no attribute 'gdc_wait'

and it dies at *hash* time, inside Triton's DependenciesFinder AST walk
(triton/runtime/jit.py visit_Attribute -> getattr), which visits every attribute
node in the kernel body.  That is why the ``if launch_pdl:`` guard around each call
does not save us: the guard is evaluated at codegen, the AST walk happens before it.

PDL is a *scheduling* facility: gdc_launch_dependents lets a dependent grid start
before this one retires, and gdc_wait blocks until the producer grid's stores are
visible.  With PDL off -- and it does not exist at all below sm_90; these are P100s
at sm_60 -- consecutive kernels on one stream are already serialised by the stream,
so both primitives are semantically no-ops.  This installs them as no-op
``@triton.jit`` device functions, which satisfies the dependency walk and inlines to
nothing in the taken branch.

PXA_TRITON_GDC_SHIM=0 disables; =strict raises if the attributes cannot be installed.
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_TRITON_GDC_SHIM", "auto").strip().lower()

_NAMES = ("gdc_wait", "gdc_launch_dependents")


def _install() -> None:
    if _MODE in ("0", "off", "no", "false"):
        return

    import triton
    import triton.language.extra.cuda as _cuda

    missing = [n for n in _NAMES if not hasattr(_cuda, n)]
    if not missing:
        log.info("triton_gdc_shim: triton %s already has %s; nothing to do",
                 triton.__version__, ", ".join(_NAMES))
        return

    @triton.jit
    def gdc_wait():
        # PDL producer-visibility wait.  No PDL below sm_90; the stream orders us.
        pass

    @triton.jit
    def gdc_launch_dependents():
        # PDL dependent-grid release.  No PDL below sm_90; the stream orders us.
        pass

    impls = {"gdc_wait": gdc_wait, "gdc_launch_dependents": gdc_launch_dependents}
    for name in missing:
        setattr(_cuda, name, impls[name])

    still = [n for n in _NAMES if not hasattr(_cuda, n)]
    if still:
        raise RuntimeError(f"triton_gdc_shim: could not install {still}")

    log.info("triton_gdc_shim: installed no-op %s on triton.language.extra.cuda "
             "(triton %s, mode=%s)", ", ".join(missing), triton.__version__, _MODE)


try:
    _install()
except Exception as exc:  # pragma: no cover
    if _MODE == "strict":
        raise
    log.warning("triton_gdc_shim: not installed (%s)", exc)
