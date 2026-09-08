"""
fixtures.py -- test data for the PXQ4 parity gates.

Two sources, and the harness runs both:

SYNTHETIC (default, no external files).  Random panel bytes are a STRICTLY STRONGER
layout test than real weights: every one of the 16 book entries and all 16 sub levels
appear with equal probability in every row, so a transposed nibble, a swapped lo/hi
sub, or an off-by-one in the 64 B scale SoA cannot hide in an unused table entry.  Real
quantized weights cluster hard around book[7]=0 and would mask exactly those bugs.

REAL (opt-in via extract.py).  Proves the fixture generator is not itself the thing being
tested, and is the only way to check the per-file `pxa.pxq6.book`/`sub` KVs.

Anchors deliberately include the awkward fp16 values -- signed zeros, subnormals, 65504 --
because the comparison is BITWISE.  -0.0 * book[7] is -0.0 in IEEE, and np.array_equal
would call that equal to +0.0 while a bit comparison would not; the harness uses the bit
comparison, so the generator has to produce the case.
"""

from __future__ import annotations

import os

import numpy as np

from . import oracle as O

# The six PXQ4 shapes actually present in Qwen3.8-27B-PXQ4.gguf, as (ggml ne0=K, ne1=rows)
# -> (N, K) in vLLM/torch orientation.  06-file-composition.md §5.
REAL_SHAPES = {
    "attn_gate":  (6144, 5120),      # GDN in_proj_z,        48 tensors
    "attn_qkv":   (10240, 5120),     # GDN in_proj_qkv,      48 tensors
    "attn_q":     (12288, 5120),     # full-attn q+gate,     17 tensors
    "ffn_gate":   (17408, 5120),     # also ffn_up,         130 tensors
    "attn_output": (5120, 6144),     # row-parallel o_proj,  17 tensors
    "ffn_down":   (5120, 17408),     # row-parallel,         65 tensors
}

# Small stand-ins with the same divisibility structure, so the CPU gates finish in
# seconds while still covering: N/64 not a power of two, K/32 not a power of two, K
# small enough that canon_nfix() saturates below CMAX, and K large enough that it does
# not.  canon_nfix is the one piece of mmv arithmetic that depends on K.
SMALL_SHAPES = {
    "tiny":       (64, 32),          # exactly one panel, one slab -> nfix == 1
    "one_panel":  (64, 256),         # nfix = 2
    "narrowK":    (128, 512),        # kslabs=16  -> lim=4  -> nfix = 4
    "wideK":      (192, 2048),       # kslabs=64  -> lim=16 -> nfix = 16 (saturated)
    "oddpanels":  (320, 1024),       # 5 panels: exercises non-power-of-two panel counts
    "shardable4": (256, 4096),       # N/4 = 64 and K/4 = 1024 -> legal at TP=4 both ways
}

# Awkward fp16 anchor values, split by whether they are safe to put in a fixture that
# will also be fed through a matmul.
#   "extreme" exercises the widest fp16 exponent range, including 65504, and is used by
#   the BIT-EXACTNESS gates, which do not care about output magnitude.
#   "realistic" keeps signed zeros and subnormals -- the cases that actually distinguish
#   a correct dequant from a sloppy one -- but drops the huge magnitudes, so the numeric
#   gates can assert something meaningful about fp16 output range.
_ANCHOR_EDGE_EXTREME = np.array(
    [0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0,
     np.float32(2.0) ** -14,          # smallest fp16 normal
     np.float32(2.0) ** -24,          # smallest fp16 subnormal
     -(np.float32(2.0) ** -24)],
    dtype=np.float32).astype(np.float16)

_ANCHOR_EDGE_REALISTIC = np.array(
    [0.0, -0.0,
     np.float32(2.0) ** -14,
     np.float32(2.0) ** -24,
     -(np.float32(2.0) ** -24)],
    dtype=np.float32).astype(np.float16)

# Row absmax ranges.  "realistic" brackets what a per-row absmax of a transformer weight
# matrix actually is (order 1e-2); the quantizer snaps it to fp16 either way.
_ANCHOR_RANGE = {"extreme": (1e-3, 2.0), "realistic": (5e-3, 8e-2)}


def synth_parts(N: int, K: int, seed: int = 0, profile: str = "extreme"):
    """Deterministic synthetic PXQ4 tensor in EMITTED form (plan §5.3).

    Returns (slabs uint8[P,S,1088], anchor float16[P,64]).

    `profile` picks the anchor magnitude distribution, NOT the layout: the slab bytes are
    uniform random in both cases, so format coverage is identical.  Only the numeric
    range differs, which matters for the gates that assert fp16 output headroom.
    """
    O.assert_geometry(N, K)
    if profile not in _ANCHOR_RANGE:
        raise ValueError(f"unknown anchor profile {profile!r}")
    P, S = N // O.PANEL_ROWS, K // O.SLAB_COLS
    rng = np.random.default_rng(seed)

    # Every byte of the slab is a valid PXQ4 byte: the low 64 are two 4-bit sub indices
    # per row, the rest are two 4-bit book codes per byte.  No reserved bits exist, so
    # uniform random bytes are in-format by construction -- and, unlike real weights,
    # they hit all 16 book entries and all 16 sub levels in every row, which is what a
    # layout test needs.
    slabs = rng.integers(0, 256, size=(P, S, O.SLAB_BYTES), dtype=np.uint8)

    lo, hi = _ANCHOR_RANGE[profile]
    mag = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(P, O.PANEL_ROWS)))
    sign = rng.choice([-1.0, 1.0], size=(P, O.PANEL_ROWS))
    anchor = (mag * sign).astype(np.float32).astype(np.float16)

    edges = _ANCHOR_EDGE_EXTREME if profile == "extreme" else _ANCHOR_EDGE_REALISTIC
    flat = anchor.reshape(-1)
    n = min(flat.size, edges.size)
    # Placed at the FRONT of panel 0 so the unsharded gates always see the edge cases,
    # and a shard test that keeps only later panels still sees ordinary values.
    flat[:n] = edges[:n]
    return slabs, anchor


def synth_blob(N: int, K: int, seed: int = 0, profile: str = "extreme") -> np.ndarray:
    slabs, anchor = synth_parts(N, K, seed, profile)
    return O.join_blob(slabs, anchor)


def synth_activations(M: int, K: int, seed: int = 0, scale: str = "unit") -> np.ndarray:
    """fp16-representable activations.

    vLLM hands the linear method fp16 (plan §6.6 asserts params_dtype is torch.float16)
    and the kernel consumes fp32 (pxq6.cuh:920), so the value must round-trip
    fp16->fp32 exactly or the CPU model and the GPU would be comparing different inputs.
    Generating in fp16 and widening guarantees that.

    scale="normalized" divides by sqrt(K), which is what a post-RMSNorm activation
    entering a K-wide reduction actually looks like; it keeps y = xW^T inside fp16 range
    so the output-range gate tests the kernel rather than the fixture.
    """
    rng = np.random.default_rng(0x5EED ^ seed)
    x = rng.normal(0.0, 1.0, size=(M, K))
    if scale == "normalized":
        x = x / np.sqrt(K)
    elif scale != "unit":
        raise ValueError(f"unknown activation scale {scale!r}")
    return x.astype(np.float16)


def all_cpu_cases(include_real=None):
    """Yield (label, N, K, slabs, anchor) for the CPU gates."""
    for label, (N, K) in SMALL_SHAPES.items():
        slabs, anchor = synth_parts(N, K, seed=abs(hash(label)) % (2 ** 31),
                                    profile="extreme")
        yield (f"synth:{label}", N, K, slabs, anchor)
    if include_real:
        for label, (N, K, slabs, anchor) in include_real.items():
            yield (f"real:{label}", N, K, slabs, anchor)


def load_real(path: str):
    """Load fixtures produced by `python -m parity_harness.extract`.

    Returns {label: (N, K, slabs, anchor)} plus the file's KV dict under key '__kv__'
    if it was saved.
    """
    z = np.load(path, allow_pickle=True)
    out = {}
    meta = {}
    for key in z.files:
        if key.startswith("__"):
            meta[key] = z[key]
            continue
        if not key.endswith("|blob"):
            continue
        label = key[:-len("|blob")]
        N = int(z[f"{label}|N"])
        K = int(z[f"{label}|K"])
        blob = z[key].tobytes()
        slabs, anchor = O.split_blob(blob, N, K)
        out[label] = (N, K, slabs, anchor)
    if "__kv__" in meta:
        out["__kv__"] = meta["__kv__"].item()
    return out


def load_raw_dir(path: str):
    """Load a fixture directory written by extract_raw.py (stdlib-only extractor).

    Returns {label: (N, K, slabs, anchor)} with the file's pxa.* KVs under '__kv__',
    matching load_real()'s shape so the gates do not care which extractor produced them.
    """
    import json
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    out = {}
    for t in man["tensors"]:
        with open(os.path.join(path, t["label"] + ".bin"), "rb") as f:
            blob = f.read()
        N, K = int(t["N"]), int(t["K"])
        if len(blob) != O.tensor_bytes(N, K):
            raise ValueError(f"{t['label']}: {len(blob)} B != geometry {O.tensor_bytes(N, K)} B")
        out[t["label"]] = (N, K) + O.split_blob(blob, N, K)
    out["__kv__"] = man.get("kv", {})
    return out
