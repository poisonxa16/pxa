"""
compare.py -- comparison primitives with the right strictness for each claim.

The distinction that matters here: `np.array_equal` is the WRONG tool for a bit-exactness
gate.  It reports 0.0 == -0.0 as equal and NaN == NaN as unequal.  PXQ4 produces signed
zeros routinely (book[7] is exactly 0, so every code-7 element is +/-0 depending on the
sign of its anchor), so a sign-of-zero bug -- e.g. computing eff as |anchor|*sub, or
reassociating to anchor*(sub*book) -- would slip past array_equal on the very elements
where it is easiest to introduce.  Everything claiming "bit-exact" here compares the
raw bit patterns.
"""

from __future__ import annotations

import numpy as np


# numpy >= 2 promotes a 0-d array to shape (1,) under `.view()`, so the shape has to be
# restored explicitly or a scalar goes in and a length-1 vector comes out -- which then
# fails `int()` with "only 0-dimensional arrays can be converted to Python scalars".
# That is not hypothetical: it broke bit_diff_report's per-element lines, i.e. the
# diagnostic every bit-exactness gate prints at the exact moment it is needed.
_BITS_VIEW = {np.dtype(np.float16): np.uint16,
              np.dtype(np.float32): np.uint32,
              np.dtype(np.float64): np.uint64}


def bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    u = _BITS_VIEW.get(a.dtype)
    if u is None:
        return a
    return a.view(u).reshape(a.shape)


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(bits(a), bits(b)))


def bit_diff_report(a: np.ndarray, b: np.ndarray, name_a="A", name_b="B", limit=8) -> str:
    """Human-readable first-N mismatches, with the bit patterns, because a fp32 decimal
    print of a 1-ULP difference looks identical on both sides."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {name_a}={a.shape} vs {name_b}={b.shape}"
    if a.dtype != b.dtype:
        return f"dtype {name_a}={a.dtype} vs {name_b}={b.dtype}"
    ba, bb = bits(a), bits(b)
    idx = np.argwhere(ba != bb)
    if idx.size == 0:
        return "identical"
    lines = [f"{idx.shape[0]} / {a.size} elements differ ({100.0*idx.shape[0]/a.size:.4f}%)"]
    ulp = np.abs(ba.astype(np.int64) - bb.astype(np.int64))
    lines.append(f"  max |bitpattern delta| = {int(ulp.max())} "
                 f"(1 == adjacent representable value, i.e. 1 ULP)")
    for row in idx[:limit]:
        t = tuple(int(v) for v in row)
        # Index the bit arrays that were already computed rather than re-viewing a
        # scalar: fewer conversions, and no 0-d view to get wrong.
        lines.append(f"  {t}: {name_a}={a[t]!r} (0x{int(ba[t]):x})  "
                     f"{name_b}={b[t]!r} (0x{int(bb[t]):x})")
    return "\n".join(lines)


def err_stats(got: np.ndarray, ref: np.ndarray) -> dict:
    """Error of `got` against a high-precision `ref`, in float64."""
    g = np.asarray(got, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    d = np.abs(g - r)
    denom = np.maximum(np.abs(r), np.finfo(np.float64).tiny)
    rel = d / denom
    # A relative error on an output that is itself ~0 is meaningless; report the
    # normalised error against the row's own scale as well, which is what actually
    # governs whether a downstream softmax notices.
    scale = np.maximum(np.abs(r).max(), np.finfo(np.float64).tiny)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float(rel.max()),
        "median_rel": float(np.median(rel)),
        "max_abs_over_absmax": float(d.max() / scale),
        "rms": float(np.sqrt(np.mean((g - r) ** 2))),
        "ref_absmax": float(scale),
    }
