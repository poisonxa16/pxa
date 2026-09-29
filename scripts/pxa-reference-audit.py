#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Bug #281 guard: PXA_REFERENCE=1 must turn off every default-on lever it claims to.

The REFERENCE banner (pxa-enhance.cuh, pxa_enhance_log_startup) says "PXA levers OFF" and names
what stays on. This scan finds every CUDA-side lever whose unset default is a bare ON
(`!(e && atoi(e) == 0)`) - i.e. a lever that ignores the config level - and fails unless it is on
STAY_ON, the list the banner names. Gate a new default-on lever with pxa_gate_default(true), or add
it here AND to the banner string in the same commit.

  scripts/pxa-reference-audit.py      exit 1 on an unlisted level-blind default-on lever
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = ["ggml/src/ggml-cuda.cu", "ggml/src/ggml-cuda"]

# Named in the REFERENCE banner: the codec kernels (how a PXQ tensor is read at all), the sm_70
# GEMM dispatch that also fixes an ldc overrun, the MXFP4 dequant form, and switches that are
# not speed levers (a cache inside an opt-in arm, safety clamps).
STAY_ON = {
    "PXA_PXQ6", "PXA_PXQ6R", "PXA_PXQ1", "PXA_PXQ2", "PXA_PXQ3", "PXA_PXQ4_2D",
    "PXA_PXQN_DQ_RT", "PXA_PXQN_GEMM_KS", "PXA_PXQN_ROT_ROUTE", "PXA_PXQN_MMVQ", "PXA_PXQN_Q8SC",
    "PXA_PXQN_GU_PREFILL", "PXA_PXQN_GU_FUSE", "PXA_PXQN_MOE_GEMM_GUFUSE", "PXA_PXQN_MOE_GEMM_XFOLD",
    "PXA_PXQN_RHT_FUSE", "PXA_PXQN_DQ_CO",
    "PXA_VOLTA_F16_GEMM", "PXA_MXFP4_DEQ_V2",
    "PXA_PXQ4_MMQ_ACACHE",          # cache inside the default-OFF PXQ4_MMQ arm, cannot fire alone
    "PXA_PXQ_DENSE_GATEUP_SPLIT",   # sub-switch of PXQ_DENSE_GATEUP, which REFERENCE turns off
    "PXA_P2P_SELFTEST",             # startup correctness check (corrupting peer links), not a lever
}

GETENV = re.compile(r'getenv\("(PXA_[A-Z0-9_]+)"\)')
BLIND = re.compile(r'!\(\s*e\s*&&\s*atoi\(e\)\s*==\s*0\s*\)')


def files():
    for s in SCAN:
        p = os.path.join(ROOT, s)
        if os.path.isfile(p):
            yield p
            continue
        for d, _, fs in os.walk(p):
            for f in fs:
                if f.endswith((".cu", ".cuh", ".h")):
                    yield os.path.join(d, f)


def main():
    bad = []
    for path in files():
        lines = open(path, errors="replace").read().split("\n")
        for i, ln in enumerate(lines):
            for m in GETENV.finditer(ln):
                win = "\n".join(lines[i:i + 3])
                if BLIND.search(win) and m.group(1) not in STAY_ON:
                    bad.append(f"{os.path.relpath(path, ROOT)}:{i+1}: {m.group(1)}")
    if bad:
        print("pxa-reference-audit: level-blind default-on levers not named by the REFERENCE banner:")
        print("\n".join("  " + b for b in bad))
        print("fix: default through pxa_gate_default(true), or list it in STAY_ON and the banner")
        return 1
    print("pxa-reference-audit: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
