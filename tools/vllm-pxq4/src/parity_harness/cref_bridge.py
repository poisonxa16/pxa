"""
cref_bridge.py -- run the SHIPPING engine's own CPU dequant and get its fp32 back.

This is what makes gate G1 a source-of-truth test rather than a self-consistency test:
the bytes go through ggml/src/pxq-cpu.c compiled from a verbatim copy of the production
tree (see cref/VENDOR_PROVENANCE.txt), not through a Python transcription of it.

Builds on first use (cc + libm, ~1 s, no CUDA, no ggml build system).  If no compiler is
available the harness degrades to skipping G1's C leg and says so loudly -- it does not
quietly pass.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CREF_DIR = os.path.join(_HERE, "cref")
CREF_BIN = os.path.join(CREF_DIR, "pxq4_cref")

# pxa_pxq_ensure_tables() (pxq-cpu.c:75-106) lets these replace the frozen tables at
# runtime.  Note PXA_PXQ2_SUB and PXA_PXQ3_SUB overwrite the SAME sub16 array PXQ4 uses,
# which is easy to miss.  The C tool refuses to run with any of them set; we scrub them
# from the child environment as well so a stray export in the operator's shell cannot
# turn a real mismatch into a confusing error.
_TABLE_ENV = ("PXA_PXQ6_BOOK", "PXA_PXQ6_SUB", "PXA_PXQ2_SUB", "PXA_PXQ3_SUB")


class CRefUnavailable(RuntimeError):
    pass


def available() -> bool:
    if os.path.exists(CREF_BIN):
        return True
    try:
        build()
        return True
    except Exception:
        return False


def build(force: bool = False) -> str:
    if os.path.exists(CREF_BIN) and not force:
        return CREF_BIN
    script = os.path.join(CREF_DIR, "build.sh")
    if not os.path.exists(script):
        raise CRefUnavailable(f"missing {script}")
    r = subprocess.run(["sh", script], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(CREF_BIN):
        raise CRefUnavailable(f"cref build failed:\n{r.stdout}\n{r.stderr}")
    return CREF_BIN


def dequant(blob, N: int, K: int) -> np.ndarray:
    """blob (bytes) -> float32[N,K], computed by the production C."""
    build()
    env = {k: v for k, v in os.environ.items() if k not in _TABLE_ENV}
    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, "in.bin")
        fout = os.path.join(td, "out.f32")
        with open(fin, "wb") as f:
            f.write(blob if isinstance(blob, (bytes, bytearray)) else bytes(blob))
        r = subprocess.run([CREF_BIN, str(N), str(K), fin, fout],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise CRefUnavailable(f"pxq4_cref exited {r.returncode}: {r.stderr.strip()}")
        a = np.fromfile(fout, dtype="<f4")
    if a.size != N * K:
        raise CRefUnavailable(f"pxq4_cref produced {a.size} floats, expected {N*K}")
    return a.reshape(N, K)
