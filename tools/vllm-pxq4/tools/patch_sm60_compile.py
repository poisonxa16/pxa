#!/usr/bin/env python3
"""Lift the sm_70 gates that block torch.compile / inductor on sm_60 (P100).

None of them reflects a real capability limit of the device:

 1. torch/utils/_triton.py::has_triton() hard-codes `device major >= 7`, so
    inductor's scheduler raises GPUTooOldForTriton before it ever tries to
    compile anything.  Triton 3.3 itself compiles and runs ordinary
    pointwise/reduction kernels on sm_60 correctly (verified) -- only `tl.dot`
    (MMA) is genuinely unavailable, and inductor does not lower matmuls to
    triton unless max_autotune is on.

 2. Inductor's generated triton passes eviction_policy='evict_first'/'evict_last'
    on nearly every tl.load.  Those lower to the PTX `.L1::evict_*` modifiers,
    which ptxas rejects with "requires .target sm_70 or higher".  The modifier
    is a pure L1/L2 cache *hint*: dropping it changes performance, never
    semantics.  Stripped from the PTX text in the nvidia backend's make_ptx(),
    which is the only place that both sees the finished PTX and knows the
    target capability -- inductor's compile-worker subprocesses have no CUDA
    context, so probing the active driver at the triton-semantic layer silently
    no-ops there.  A second, belt-and-braces suppression is applied at
    semantic._str_to_eviction_policy for the main process and for hand-written
    kernels (fla, vllm) that pass the hint explicitly.

Idempotent; --revert undoes it.  Run with the target venv's python:

    venv/bin/python patch_sm60_compile.py [--revert]
"""
import argparse
import pathlib
import re

MARK = "# --- pxa sm_60 compile enablement ---"

TRITON_SHIM = MARK + '''
def _pxa_sm60_no_evict():
    """True when the active CUDA target predates sm_70 (no .evict_* in PTX).

    Best-effort only: inductor compile workers have no CUDA context and fall
    through to False here.  The authoritative strip is in the nvidia backend's
    make_ptx(), which is handed the real target capability.
    """
    global _PXA_SM60
    try:
        return _PXA_SM60
    except NameError:
        pass
    _PXA_SM60 = False
    try:
        from triton.runtime import driver
        tgt = driver.active.get_current_target()
        if getattr(tgt, "backend", None) == "cuda" and isinstance(tgt.arch, int):
            _PXA_SM60 = tgt.arch < 70
    except Exception:
        pass
    return _PXA_SM60
# --- end pxa sm_60 compile enablement ---
'''

SEM_OLD = ("    eviction = ir.EVICTION_POLICY.NORMAL  # default\n"
           "    if eviction_policy:\n")
SEM_NEW = ("    eviction = ir.EVICTION_POLICY.NORMAL  # default\n"
           "    if eviction_policy and not _pxa_sm60_no_evict():\n")


def patch_triton_semantic(root, revert):
    p = root / "triton" / "language" / "semantic.py"
    src = p.read_text()
    if revert:
        if MARK not in src:
            return False
        src = re.sub(MARK + r".*?# --- end pxa sm_60 compile enablement ---\n\n\n",
                     "", src, flags=re.S)
        src = src.replace(SEM_NEW, SEM_OLD, 1)
        p.write_text(src)
        return True
    if MARK in src:
        return False
    anchor = "def _str_to_eviction_policy(eviction_policy):\n"
    assert anchor in src, "triton semantic.py: anchor not found"
    assert SEM_OLD in src, "triton semantic.py: eviction body not found"
    src = src.replace(anchor, TRITON_SHIM + "\n\n" + anchor, 1)
    src = src.replace(SEM_OLD, SEM_NEW, 1)
    p.write_text(src)
    return True


PTX_MARK = "# pxa sm_60: .L1::evict_* are cache hints and need sm_70+; drop them"
PTX_OLD = '        if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":\n'
PTX_NEW = (
    "        " + PTX_MARK + "\n"
    "        if capability < 70:\n"
    "            ret = re.sub(r'\\.L1::evict_(?:first|last)', '', ret)\n"
    + PTX_OLD
)


def patch_triton_ptx(root, revert):
    p = root / "triton" / "backends" / "nvidia" / "compiler.py"
    src = p.read_text()
    if revert:
        if PTX_MARK not in src:
            return False
        p.write_text(src.replace(PTX_NEW, PTX_OLD, 1))
        return True
    if PTX_MARK in src:
        return False
    assert src.count(PTX_OLD) == 1, "triton nvidia compiler.py: make_ptx anchor not unique"
    p.write_text(src.replace(PTX_OLD, PTX_NEW, 1))
    return True


TORCH_OLD = "        return device_interface.Worker.get_device_properties().major >= 7\n"
TORCH_NEW = (
    "        # pxa: sm_60 runs everything inductor emits (it never needs tl.dot).\n"
    "        # The >= 7 gate is upstream policy, not a device capability.\n"
    "        import os\n"
    "        if os.environ.get('PXA_TRITON_ALLOW_SM60', '1') != '0':\n"
    "            return device_interface.Worker.get_device_properties().major >= 6\n"
    "        return device_interface.Worker.get_device_properties().major >= 7\n"
)


def patch_torch(root, revert):
    p = root / "torch" / "utils" / "_triton.py"
    src = p.read_text()
    if revert:
        if TORCH_NEW not in src:
            return False
        p.write_text(src.replace(TORCH_NEW, TORCH_OLD, 1))
        return True
    if TORCH_NEW in src:
        return False
    assert TORCH_OLD in src, "torch _triton.py: anchor not found"
    p.write_text(src.replace(TORCH_OLD, TORCH_NEW, 1))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None,
                    help="site-packages dir (default: derived from the running interpreter)")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.site:
        root = pathlib.Path(a.site)
    else:
        import triton
        root = pathlib.Path(triton.__file__).resolve().parent.parent
    for name, fn in (("triton.semantic", patch_triton_semantic),
                     ("triton.ptx", patch_triton_ptx),
                     ("torch.has_triton", patch_torch)):
        changed = fn(root, a.revert)
        state = "reverted" if (a.revert and changed) else "patched" if changed else "already ok"
        print(f"{name}: {state}")


if __name__ == "__main__":
    main()
