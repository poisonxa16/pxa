"""
adapters.py -- optional bindings to the OTHER agents' components.

The harness must be useful before agents A, B and C have landed anything, so every
foreign component is discovered at runtime and its absence downgrades a gate to SKIP,
never to PASS.  A gate that cannot see the thing it tests must say so.

  * `pxq4_vllm.reference` / `pxq4_vllm.layout` (agent A)  -- plan §6.2, §6.3
  * `libpxq4_sm70.so` -> torch.ops.pxq4.*     (agent C)   -- plan §7.1
"""

from __future__ import annotations

import importlib
import os

_MISSING = object()


class Missing:
    def __init__(self, what: str, why: str):
        self.what = what
        self.why = why

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<missing {self.what}: {self.why}>"


# The plan (§4) names the runtime package `pxq4_vllm`, but the early prototypes landed
# under flatter names while the repo layout was still being settled.
# Search both, in plan order, so the gates keep working either way and nothing is skipped
# merely because a directory got renamed.
_REF_CANDIDATES = ("pxq4_vllm.reference", "gguf_to_vllm.reference", "pxq4_kernel_ref")
_LAYOUT_CANDIDATES = ("pxq4_vllm.layout", "gguf_to_vllm.layout", "pxq4_kernel_ref")


def _first_importable(names, what):
    errs = []
    for n in names:
        try:
            return importlib.import_module(n)
        except Exception as e:        # ImportError, or a broken module mid-development
            errs.append(f"{n}: {type(e).__name__}: {e}")
    return Missing(what, "; ".join(errs))


def ref_module():
    """The reference dequant under test.  Plan §6.3 names it pxq4_vllm.reference."""
    return _first_importable(_REF_CANDIDATES, "reference.dequant")


def all_ref_modules():
    """Every reference implementation that is importable, as {name: module}.

    More than one is normal here: agent A ships one for the converter and agent C ships
    one as the kernel's numpy twin.  They are INDEPENDENT transcriptions of the same C,
    so cross-checking all of them against each other and against cref is strictly more
    evidence than checking one.
    """
    out = {}
    for n in _REF_CANDIDATES:
        try:
            m = importlib.import_module(n)
        except Exception:
            continue
        if hasattr(m, "dequant"):
            out[n] = m
    return out


def layout_module():
    return _first_importable(_LAYOUT_CANDIDATES, "layout")


def torch_module():
    try:
        return importlib.import_module("torch")
    except Exception as e:
        return Missing("torch", f"{type(e).__name__}: {e}")


def cuda_available():
    t = torch_module()
    if not t:
        return False
    try:
        return t.cuda.is_available()
    except Exception:
        return False


def pxq4_ops():
    """torch.ops.pxq4 with dequant_out/mmv_out loaded, or Missing.

    Resolution order:
      1. pxq4_vllm.ops.load_library()  -- the supported path once agent B has landed
      2. $PXQ4_LIB                     -- direct .so path, for testing agent C standalone
      3. already-loaded torch.ops.pxq4 -- someone else loaded it
    """
    t = torch_module()
    if not t:
        return Missing("torch.ops.pxq4", "torch is not importable")
    try:
        ops = importlib.import_module("pxq4_vllm.ops")
        ops.load_library()
    except Exception:
        lib = os.environ.get("PXQ4_LIB")
        if lib:
            if not os.path.exists(lib):
                return Missing("torch.ops.pxq4", f"$PXQ4_LIB={lib} does not exist")
            try:
                t.ops.load_library(lib)
            except Exception as e:
                return Missing("torch.ops.pxq4", f"load_library({lib}) failed: {e}")
    try:
        ns = t.ops.pxq4
        # Touch both ops so a partially-registered library fails here, loudly, rather
        # than at the first call inside a test.
        _ = ns.dequant_out
        _ = ns.mmv_out
        return ns
    except Exception as e:
        return Missing("torch.ops.pxq4",
                       f"not registered ({type(e).__name__}: {e}); set $PXQ4_LIB or "
                       f"install pxq4_vllm")
