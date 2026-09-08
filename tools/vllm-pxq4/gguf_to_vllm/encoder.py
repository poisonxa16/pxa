"""encoder.py — optional binding to the native PXQ4 encoder, used only by the P2 policies.

P1 NEEDS NOTHING HERE. Everything P1 serves as PXQ4 is already PXQ4 on disk, and the converter
moves those bytes without touching a value. This module exists for P2, where three tensor
classes must be re-encoded because they are NOT PXQ4 in the artifact:

    P2a  linear_attn.out_proj   ggml MXFP4 (id 39), 48 tensors      -0.516 GiB/GPU at TP=4
    P2b  lm_head                ggml q8_0 -> source the AWQ twin's  -0.435 GiB/GPU
                                UNQUANTIZED BF16 lm_head.weight instead, to avoid encoding
                                an already-quantized tensor a second time
    P2c  self_attn.{k,v}_proj   ggml q8_0, 34 tensors               -0.427 GiB/GPU

The backbone table pins k/v and the LM head away from 4 bits deliberately (docs/LEVERS.md,
recorded in the file as ``pxa.pxq.backbone_map``), so each of these is a quality decision, not
just a bandwidth one — gate G10 exists for exactly that and this module will happily encode
something that should not ship.

THE ABI, frozen by plan §P2 (``tools/pxq4_encode/``, a pybind/C target that ``#include``s
``/path/to/engine-repo/src/pxq6-quantize.inc.cpp`` read-only so the file-static
``pxq6_quantize_expert(src, dst, R, K, imx, tier, row0)`` becomes visible in its TU):

    int pxq4_encode(const float * src, uint8_t * dst, int R, int K, const float * imx_or_null)

    src : row-major float32 [R, K]
    dst : R/64 panels of (128 + (K/32)*1088) bytes
    imx : optional importance vector of length K, or NULL
    ret : 0 on success

ALWAYS ENCODE A WHOLE TENSOR FROM ROW 0. ``row0`` seeds the deterministic tie-break
``pxq_tie_take_hi`` (pxq6-quantize.inc.cpp:49, used :230, warned :416-418), so encoding a
64-row-aligned SUB-RANGE starting at a nonzero row yields DIFFERENT BYTES for identical
weights. The binding below has no row0 parameter for that reason: there is no legitimate
caller for a partial encode.

The extension is not built by this component (plan §9: agent C owns ``csrc/``). If
``--encoder`` is not given, a P2 policy fails loudly at planning time rather than silently
falling back to fp16 — a silent fallback would produce a checkpoint that loads, runs, and is
simply slower than advertised, which is the hardest kind of bug to notice.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from . import layout as L
from . import reference as R


class NativeEncoder:
    """ctypes binding to ``libpxq4_encode.so`` / ``pxq4_encode.so``."""

    def __init__(self, so_path: str) -> None:
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"pxq4 encoder shared object not found: {so_path}")
        self.path = so_path
        self._lib = ctypes.CDLL(so_path)
        fn = self._lib.pxq4_encode
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                       ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self._fn = fn

    def encode(self, w: np.ndarray, imatrix: np.ndarray | None = None) -> bytes:
        """float32 [N, K] -> PXQ4 panel blob. Whole tensors only."""
        if w.ndim != 2:
            raise ValueError(f"pxq4 encode: expected 2-D weight, got {w.shape}")
        N, K = w.shape
        L.assert_geometry(N, K)
        src = np.ascontiguousarray(w, dtype=np.float32)
        dst = np.empty(L.tensor_bytes(N, K), dtype=np.uint8)
        imx_p = None
        if imatrix is not None:
            imx = np.ascontiguousarray(imatrix, dtype=np.float32)
            if imx.shape != (K,):
                raise ValueError(f"pxq4 encode: imatrix must be [{K}], got {imx.shape}")
            imx_p = imx.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._fn(src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                      ctypes.c_int(N), ctypes.c_int(K), imx_p)
        if rc != 0:
            raise RuntimeError(f"pxq4_encode returned {rc} for [{N},{K}]")
        return dst.tobytes()


def encode_and_check(enc: NativeEncoder, w: np.ndarray, name: str,
                     imatrix: np.ndarray | None = None,
                     max_rel_err: float = 0.35) -> tuple[bytes, dict]:
    """Encode, then immediately decode with our own reference and measure the error.

    This is not a numerical gate on quality — that is G10's job on real evals — it is a
    TRIPWIRE. If the encoder's output layout ever diverges from what ``reference.dequant``
    expects (a different tier, an HQ slab size, a table override picked up from the
    environment), the reconstruction error explodes to order-1 and this raises, instead of the
    error being discovered three days later as "the model got dumber". The default bound is
    deliberately loose: the published PXQ6-core figure is wrel 0.068 on the calibration
    protocol, so 0.35 catches a broken layout without failing an unusual tensor.
    """
    blob = enc.encode(w, imatrix)
    N, K = w.shape
    back = R.dequant_blob(blob, N, K)
    num = float(np.linalg.norm(back - w))
    den = float(np.linalg.norm(w)) or 1.0
    wrel = num / den
    if not np.isfinite(wrel) or wrel > max_rel_err:
        raise RuntimeError(
            f"{name}: pxq4 re-encode round-trip wrel={wrel:.4f} exceeds {max_rel_err} — the "
            f"encoder's layout or tables do not match reference.dequant. Refusing to emit.")
    return blob, {"wrel": wrel, "absmax": float(np.abs(w).max())}


def bf16_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Widen raw little-endian bfloat16 bytes to float32.

    numpy has no bfloat16 dtype, and we must read the AWQ twin's BF16 ``lm_head.weight``
    without dragging torch into the converter. bfloat16 IS the top 16 bits of a float32, so
    the widening is a shift with no rounding and no special cases — inf and NaN patterns carry
    across unchanged.
    """
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).reshape(shape)
