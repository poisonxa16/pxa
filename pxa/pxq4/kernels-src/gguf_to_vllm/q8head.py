"""q8head.py -- quantize an LM head to int8 with a per-row fp16 scale.

The format is defined by the kernel that consumes it (kernels-src/pxq_q8.cuh); this module is
the other half of that contract and deliberately restates it rather than referring to it:

    weight        int8    [V, H]   row-major
    weight_scale  float16 [V, 1]   scale[n] = absmax(row n) / 127
    reconstruction  w[n, k] = float(q[n, k]) * float(scale[n])

PER-ROW, NOT PER-TENSOR, and the reason is arithmetic rather than taste. A vocabulary head has
248,320 rows whose absmax spans orders of magnitude -- a handful of frequent tokens carry large
weights and the long tail does not. One scale for the whole tensor is set by the largest row
and quantizes every other row into a few levels around zero. Per-row also costs nothing to
shard: the vocab axis is exactly the axis vLLM's vocab loader narrows.

SYMMETRIC, NO ZERO POINT. An asymmetric scheme would buy about half a bit on a distribution
that is already near-symmetric around zero, and would cost a second per-row tensor plus a
correction term in the kernel's inner loop. Not worth it here.

ROUNDING IS ROUND-HALF-TO-EVEN, via numpy's rint, and clipping is to [-127, 127] rather than
[-128, 127]: the asymmetric endpoint is unusable under a symmetric scale (nothing maps to
-128 unless a value exceeds absmax) and excluding it keeps the negation of a representable
value representable, which matters if anything downstream ever folds a sign.

A ZERO ROW gets scale 0 and codes 0. That is exact -- 0 * 0 == 0 -- and it avoids a division
by zero that would otherwise produce NaN codes for a padding row, of which a padded vocabulary
has hundreds.
"""

from __future__ import annotations

import numpy as np


def quantize_head(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """float [V, H] -> (int8 [V, H], float16 [V, 1], stats).

    Returns stats rather than printing them: the caller decides what to report, and a
    conversion that silently degraded a head is exactly what a stats dict is for.
    """
    if w.ndim != 2:
        raise ValueError(f"q8head: expected a 2-D head, got shape {w.shape}")
    w32 = np.ascontiguousarray(w, dtype=np.float32)
    if not np.all(np.isfinite(w32)):
        raise ValueError("q8head: the head contains non-finite values; refusing to quantize "
                         "-- a NaN row would quantize to a valid-looking row of zeros")

    absmax = np.abs(w32).max(axis=1)                       # [V]
    scale = absmax / 127.0
    zero_rows = int((absmax == 0).sum())
    # Exact for a zero row, and no division by zero for the hundreds of padding rows a padded
    # vocabulary carries.
    safe = np.where(scale > 0, scale, 1.0)
    q = np.rint(w32 / safe[:, None]).clip(-127, 127).astype(np.int8)
    q[absmax == 0] = 0

    scale16 = scale.astype(np.float16).reshape(-1, 1)
    # The scale is STORED as fp16, so the error that matters is measured against the fp16
    # scale the kernel will actually use, not against the fp32 one we computed with.
    deq = q.astype(np.float32) * scale16.astype(np.float32)
    err = deq - w32
    denom = float(np.linalg.norm(w32))
    stats = {
        "rows": int(w32.shape[0]),
        "cols": int(w32.shape[1]),
        "zero_rows": zero_rows,
        "wrel": float(np.linalg.norm(err) / denom) if denom > 0 else 0.0,
        "max_abs_err": float(np.abs(err).max()),
        "scale_min": float(scale16.min()),
        "scale_max": float(scale16.max()),
        "bpw": 8.0 + 16.0 / float(w32.shape[1]),
    }
    # A per-row symmetric int8 quantizer should land near 1/(127*sqrt(3)) ~ 0.0045 relative on
    # a well-behaved head. An order of magnitude worse means the input was not what we think
    # -- an already-quantized head re-quantized, or a transposed tensor -- and that is worth
    # stopping for rather than shipping.
    if stats["wrel"] > 0.05:
        raise ValueError(
            f"q8head: relative error {stats['wrel']:.4f} is far above what per-row int8 should "
            f"give (~0.005). The input is probably not a raw head -- check for a transpose, or "
            f"for a head that was already quantized once.")
    return q, scale16, stats
