"""
oracle.py -- INDEPENDENT numpy model of the PXQ4 (ggml type id 252) device format.

Why this file exists at all, given the plan already assigns `src/pxq4_vllm/reference.py`
to Agent A: a parity harness that only compares Agent A's reference against Agent C's
kernel proves the two agree, not that either is right. This module is a SECOND,
independently-written model, transcribed directly from the C

    pxa_deq_row_pxq6()            ggml/src/pxq-cpu.c:135-158
    pxa_deq_pairs16()             ggml/src/pxq-cpu.c:126-133

and from the CUDA

    struct pxq6_pol_p6            ggml/src/ggml-cuda/pxq6.cuh:317-346
    pxq6_dot32                    ggml/src/ggml-cuda/pxq6.cuh:634-674
    pxq6_acc2                     ggml/src/ggml-cuda/pxq6.cuh:603-609
    pxq6_canon_nfix               ggml/src/ggml-cuda/pxq6.cuh:826-834
    k_pxq6_dequant_matrix         ggml/src/ggml-cuda/pxq6.cuh:680-726
    k_pxq6_mmv                    ggml/src/ggml-cuda/pxq6.cuh:914-971

so that oracle == pxq4_vllm.reference == libpxq4_sm70.so is a three-way agreement, and
`cref/` (which compiles the ACTUAL mgv-wt C) makes it four-way with the shipping engine.

No torch. No vllm. No CUDA. numpy only. Runs on any machine, no GPU, in milliseconds.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------
# Frozen geometry.  ggml/include/ggml-pxq6-tables.h:21-27.
# ---------------------------------------------------------------------------------------
PANEL_ROWS = 64          # PXQ6_BM
SLAB_COLS = 32           # PXQ6_QK
SLAB_BYTES = 1088        # PXQ6_SLAB_BYTES  = 64 scale bytes + 64 rows * 16 code bytes
HEADER_BYTES = 128       # PXQ6_HDR_BYTES   = 64 rows * fp16 anchor
CODE_OFF = 64            # pxq6_pol_p6::CODE_OFF -- scale SoA occupies bytes [0,64)
CODE_BYTES = 16          # pxq6_pol_p6::CODE_BYTES -- one 16 B nibble row per weight row
NEFF = 2                 # pxq6_pol_p6::NEFF -- two eff scales per 32-element block
TYPE_ID = 252            # GGML_TYPE_PXQ4

# k_pxq6_mmv block decomposition.  PXQ4_MMV_KSEG = 4 (ggml/src/ggml-cuda/pxq4.cuh:114);
# PXQ6_MMV_SPLIT_MAX = PXQ6_CANON_CMAX = 16 (pxq6.cuh:818-822).
MMV_KSEG = 4
MMV_CANON_CMAX = 16

# ---------------------------------------------------------------------------------------
# Frozen tables, transcribed as C99 hex-float literals so the transcription is provably
# lossless -- a decimal transcription would need 9 significant digits and be unreviewable.
# ggml/include/ggml-pxq6-tables.h:33-44.  These are ALSO stored per-file in the GGUF as
# `pxa.pxq6.book` / `pxa.pxq6.sub`; the converter must read those (PXA_PXQ6_BOOK /
# PXA_PXQ6_SUB can override them at build time) and `check_tables_against_gguf` below
# is the gate that catches a mismatch.
# ---------------------------------------------------------------------------------------
_BOOK_HEX = [
    "-0x1.f9c0p-1", "-0x1.7880p-1", "-0x1.1e00p-1", "-0x1.adc0p-2",
    "-0x1.3440p-2", "-0x1.8e40p-3", "-0x1.8740p-4", "0x0.0p+0",
    "0x1.5b00p-4", "0x1.5ec0p-3", "0x1.0c40p-2", "0x1.7140p-2",
    "0x1.e280p-2", "0x1.3380p-1", "0x1.8800p-1", "0x1.0p+0",
]
_SUB16_HEX = [
    "0x1.b7c0p-3", "0x1.36c0p-2", "0x1.72c0p-2", "0x1.a2c0p-2",
    "0x1.ccc0p-2", "0x1.f300p-2", "0x1.0bc0p-1", "0x1.1e00p-1",
    "0x1.3040p-1", "0x1.4380p-1", "0x1.5800p-1", "0x1.6ec0p-1",
    "0x1.8880p-1", "0x1.a640p-1", "0x1.cac0p-1", "0x1.f9c0p-1",
]

BOOK = np.array([float.fromhex(h) for h in _BOOK_HEX], dtype=np.float32)
SUB = np.array([float.fromhex(h) for h in _SUB16_HEX], dtype=np.float32)

# Both tables are documented as fp16-snapped.  Asserting it here means a future table
# regeneration that breaks the property trips a test instead of silently changing which
# arithmetic is exact.  (It matters: the CUDA PRMT/SHFL book modes at pxq6.cuh:538 are
# only bit-identical to the smem-table modes BECAUSE half->float is exact.)
assert np.array_equal(BOOK.astype(np.float16).astype(np.float32), BOOK)
assert np.array_equal(SUB.astype(np.float16).astype(np.float32), SUB)
assert BOOK[7] == 0.0 and BOOK[15] == 1.0
assert np.all(np.diff(BOOK) > 0) and np.all(np.diff(SUB) > 0)


# ---------------------------------------------------------------------------------------
# Layout arithmetic.  Mirrors the plan's src/pxq4_vllm/layout.py contract (plan §6.2) so
# the harness can be pointed at either implementation.
# ---------------------------------------------------------------------------------------
def panel_bytes(K: int) -> int:
    """One 64-row panel: 128 B anchor header + K/32 slabs of 1088 B.  pxq6.cuh:520-522."""
    return HEADER_BYTES + (K // SLAB_COLS) * SLAB_BYTES


def tensor_bytes(N: int, K: int) -> int:
    """Total on-disk bytes.  Equals ggml_row_size(PXQ4,K)*N = (2 + 17*K/32)*N exactly,
    i.e. 4.25 + 16/K bpw (ggml.h:465-467).  There is no inter-tensor padding."""
    return (N // PANEL_ROWS) * panel_bytes(K)


def slab_shape(N: int, K: int) -> tuple:
    return (N // PANEL_ROWS, K // SLAB_COLS, SLAB_BYTES)


def anchor_shape(N: int) -> tuple:
    return (N // PANEL_ROWS, PANEL_ROWS)


def assert_geometry(N: int, K: int) -> None:
    """The quantizer's own eligibility gate (llama-quantize.cpp:1399-1401 demotes to q8_0
    on failure), and the only thing standing between us and a silently truncated shard:
    `round(shard_size // packed_factor)` in parameter.py:605-610 does NOT raise."""
    if N % PANEL_ROWS != 0:
        raise ValueError(f"pxq4: N={N} is not a multiple of {PANEL_ROWS} rows")
    if K % SLAB_COLS != 0:
        raise ValueError(f"pxq4: K={K} is not a multiple of {SLAB_COLS} columns")


def split_blob(blob, N: int, K: int):
    """Raw GGUF tensor bytes -> (slabs uint8[P,S,1088], anchor float16[P,64]).

    This is THE cross-component contract (plan §5.3).  It is a pure split: the header of
    each panel becomes one anchor row, the remainder becomes that panel's slab stack.
    Not one byte is reordered and not one value is recomputed -- which is why `join_blob`
    below must reproduce the input exactly, and why gate G2 is a byte comparison rather
    than a numeric one.
    """
    assert_geometry(N, K)
    P, S = N // PANEL_ROWS, K // SLAB_COLS
    a = np.frombuffer(blob, dtype=np.uint8)
    exp = tensor_bytes(N, K)
    if a.size != exp:
        raise ValueError(f"pxq4: blob is {a.size} B, geometry says {exp} B (N={N},K={K})")
    a = a.reshape(P, panel_bytes(K))
    anchor = a[:, :HEADER_BYTES].copy().view("<f2")
    slabs = a[:, HEADER_BYTES:].copy().reshape(P, S, SLAB_BYTES)
    return slabs, anchor


def join_blob(slabs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Inverse of split_blob.  Byte-exact round-trip is gate G2."""
    P, S, sb = slabs.shape
    assert sb == SLAB_BYTES and anchor.shape == (P, PANEL_ROWS)
    assert anchor.dtype == np.float16
    hdr = np.ascontiguousarray(anchor).view(np.uint8).reshape(P, HEADER_BYTES)
    body = np.ascontiguousarray(slabs).reshape(P, S * SLAB_BYTES)
    return np.concatenate([hdr, body], axis=1).reshape(-1)


# ---------------------------------------------------------------------------------------
# Dequant.  THE parity-locked contract (pxq-cpu.h:16-18): the dequant is bit-exact, the
# GEMM kernels explicitly are not (they snap products to fp16 inside the MMA).
# ---------------------------------------------------------------------------------------
def dequant(slabs: np.ndarray, anchor: np.ndarray, *, book=BOOK, sub=SUB) -> np.ndarray:
    """slabs uint8[P,S,1088], anchor float16[P,64] -> float32[P*64, S*32].

    Exact model of pxa_deq_row_pxq6 (pxq-cpu.c:135-158):

        anch    = fp32(anchor[p, r])                       # fp16->fp32, always exact
        sb      = slabs[p, kb, r]                          # the row's scale byte
        eff_lo  = anch * sub[sb & 0xF]                     # elements  0..15
        eff_hi  = anch * sub[sb >> 4]                      # elements 16..31
        byte    = slabs[p, kb, 64 + 16*r + b]              # b = 0..15
        w[2b]   = eff * book[byte & 0xF]
        w[2b+1] = eff * book[byte >> 4]

    The multiply order is (anchor * sub) * book -- two fp32 roundings, in that
    association.  Reassociating to anchor * (sub * book) changes the low bit on real
    data and would break parity with the shipping engine.  There is no fused
    multiply-add anywhere in the dequant path, which is why this is exactly
    reproducible in numpy while the mmv (below) is not.
    """
    P, S, sb_ = slabs.shape
    assert sb_ == SLAB_BYTES
    assert anchor.shape == (P, PANEL_ROWS) and anchor.dtype == np.float16
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)

    anch = anchor.astype(np.float32)                            # [P,64], exact widening

    scale_bytes = slabs[:, :, :CODE_OFF]                        # [P,S,64], row-indexed SoA
    lo_idx = (scale_bytes & 0x0F).astype(np.intp)
    hi_idx = (scale_bytes >> 4).astype(np.intp)
    # anch broadcasts [P,1,64] against [P,S,64]: eff is per (panel, kblock, row).
    eff_lo = (anch[:, None, :] * sub[lo_idx]).astype(np.float32)
    eff_hi = (anch[:, None, :] * sub[hi_idx]).astype(np.float32)

    codes = slabs[:, :, CODE_OFF:].reshape(P, S, PANEL_ROWS, CODE_BYTES)
    # Nibble unpack: byte b carries element 2b in the low nibble, 2b+1 in the high nibble
    # (pxq-cpu.c:126-133).  Interleaving them recovers natural element order within the
    # 32-element block -- the PXQ4 permutation lives at the row/panel level only, never
    # inside a block.
    q = np.empty((P, S, PANEL_ROWS, SLAB_COLS), dtype=np.float32)
    q[..., 0::2] = book[(codes & 0x0F).astype(np.intp)]
    q[..., 1::2] = book[(codes >> 4).astype(np.intp)]

    eff = np.empty((P, S, PANEL_ROWS, SLAB_COLS), dtype=np.float32)
    eff[..., :16] = eff_lo[..., None]
    eff[..., 16:] = eff_hi[..., None]

    w = (eff * q).astype(np.float32)                            # [P,S,64,32]
    # -> [row, k]:  row = p*64 + r, column = kb*32 + element
    return np.ascontiguousarray(w.transpose(0, 2, 1, 3).reshape(P * PANEL_ROWS, S * SLAB_COLS))


def dequant_blob(blob, N: int, K: int) -> np.ndarray:
    slabs, anchor = split_blob(blob, N, K)
    return dequant(slabs, anchor)


# ---------------------------------------------------------------------------------------
# mmv:  a bit-exact model of k_pxq6_mmv's fp32 accumulation ORDER.
#
# This is not "a matmul".  The kernel commits to a specific fold and the comments call it
# PXQ_CANON_v1 "bit-identical to the K1/S-split forms" (pxq6.cuh:947).  Reproducing the
# order is what lets a mismatch be attributed to a layout bug rather than to float noise.
# ---------------------------------------------------------------------------------------
def canon_nfix(kslabs: int, cmax: int = MMV_CANON_CMAX) -> int:
    """pxq6_canon_nfix, pxq6.cuh:826-834.  Largest power of two <= min(kslabs/KSEG, cmax)."""
    lim = kslabs // MMV_KSEG
    if lim < 1:
        lim = 1
    if lim > cmax:
        lim = cmax
    n = 1
    while n * 2 <= lim:
        n *= 2
    return n


def _f32(x):
    return np.float32(x)


def _fma32(a, b, c):
    """Correctly-rounded binary32 fused multiply-add, emulated in binary64.

    Safe because binary64 has 53 >= 2*24+2 bits, which is the classical sufficient
    condition for double rounding of a binary32 FMA to be innocuous.  numpy exposes no
    fp32 FMA primitive, and the whole point of this module is bit-exactness, so the
    emulation is not optional.
    """
    a = np.asarray(a, dtype=np.float32).astype(np.float64)
    b = np.asarray(b, dtype=np.float32).astype(np.float64)
    c = np.asarray(c, dtype=np.float32).astype(np.float64)
    return (a * b + c).astype(np.float32)


# Which fp32 contraction nvcc actually emitted for
#     pxq6_acc2:  acc + (a0*x0 + a1*x1)                    (pxq6.cuh:603-609, CANON_V2=0)
# is a codegen decision, not a source fact.  With the default -fmad=true exactly one of
# the two multiplies can fuse into the inner add; the outer add has no adjacent multiply
# and cannot fuse.  So there are three candidate semantics, and the honest thing is to
# model all three and let the harness REPORT which one the built .so matches rather than
# assert a guess.  ASSUMPTION: the harness assumes nvcc does not reassociate across the
# parentheses (it may not: FP reassociation is not permitted by contraction alone).
ACC_VARIANTS = ("none", "fma_a0", "fma_a1")
# Same question for the tail reduction  eff[0]*t[0] + eff[1]*t[1]  (pxq6.cuh:672).
TAIL_VARIANTS = ("none", "fma_e0", "fma_e1")


def _acc2(acc, a0, x0, a1, x1, variant: str):
    if variant == "none":
        return (acc + (_f32(a0) * _f32(x0) + _f32(a1) * _f32(x1))).astype(np.float32)
    if variant == "fma_a0":
        return (acc + _fma32(a0, x0, (_f32(a1) * _f32(x1)))).astype(np.float32)
    if variant == "fma_a1":
        return (acc + _fma32(a1, x1, (_f32(a0) * _f32(x0)))).astype(np.float32)
    raise ValueError(variant)


def _tail(e0, t0, e1, t1, variant: str):
    if variant == "none":
        return (_f32(e0) * t0 + _f32(e1) * t1).astype(np.float32)
    if variant == "fma_e0":
        return _fma32(e0, t0, (_f32(e1) * t1))
    if variant == "fma_e1":
        return _fma32(e1, t1, (_f32(e0) * t0))
    raise ValueError(variant)


def _dot32_all(slabs, anch, x_f32, book, sub, acc_variant, tail_variant):
    """Per-(panel,kblock,row) 32-element dot product, in pxq6_dot32's exact order.

    Vectorised across the 64 rows of a panel and across panels; the b-loop stays
    sequential because its accumulation order is the thing under test.
    """
    P, S, _ = slabs.shape
    sb = slabs[:, :, :CODE_OFF]
    eff0 = (anch[:, None, :] * sub[(sb & 0x0F).astype(np.intp)]).astype(np.float32)
    eff1 = (anch[:, None, :] * sub[(sb >> 4).astype(np.intp)]).astype(np.float32)

    codes = slabs[:, :, CODE_OFF:].reshape(P, S, PANEL_ROWS, CODE_BYTES)
    lo = book[(codes & 0x0F).astype(np.intp)]                  # [P,S,64,16] = element 2b
    hi = book[(codes >> 4).astype(np.intp)]                    # [P,S,64,16] = element 2b+1

    xk = x_f32.reshape(S, SLAB_COLS)
    x_even = xk[:, 0::2]                                       # [S,16] elements 2b
    x_odd = xk[:, 1::2]                                        # [S,16] elements 2b+1

    t0 = np.zeros((P, S, PANEL_ROWS), dtype=np.float32)
    t1 = np.zeros((P, S, PANEL_ROWS), dtype=np.float32)
    for b in range(16):
        # t index = (b*NEFF)>>4  ->  b<8 accumulates into t[0], b>=8 into t[1]
        # (pxq6.cuh:668).  eff is applied once, after the loop -- NOT per element.
        acc = t0 if b < 8 else t1
        new = _acc2(acc,
                    lo[:, :, :, b], x_even[None, :, None, b],
                    hi[:, :, :, b], x_odd[None, :, None, b],
                    acc_variant)
        if b < 8:
            t0 = new
        else:
            t1 = new
    return _tail(eff0, t0, eff1, t1, tail_variant)             # [P,S,64]


def mmv(x, slabs, anchor, *, book=BOOK, sub=SUB,
        acc_variant: str = "none", tail_variant: str = "none") -> np.ndarray:
    """Bit-exact model of k_pxq6_mmv (pxq6.cuh:914-971) for the dense, single-expert case.

    x: float32[M,K] (the kernel consumes fp32 activations and emits fp32 -- pxq6.cuh:920,
    :968 -- which is why the plan stages fp16<->fp32 around the call, §7.3).
    Returns float32[M,N].

    The fold, reproduced exactly:
      * the block owns one 64-row panel; thread (kseg, row) with kseg in 0..KSEG-1
      * kslabs is chopped into `nfix` chunks with boundaries (kslabs*c)//nfix
      * within a chunk, lane kseg walks kb = b0+kseg, b0+kseg+KSEG, ... ASCENDING
      * per-chunk partial `t` starts at zero and is added into the lane total `su`
      * the cross-lane reduction sums s = 0..KSEG-1 in ascending order
    Change any of those and the result moves in the low bits.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if x.ndim == 1:
        x = x[None, :]
    M, K = x.shape
    P, S, _ = slabs.shape
    assert S * SLAB_COLS == K, f"x has K={K}, weights have K={S*SLAB_COLS}"
    N = P * PANEL_ROWS
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    anch = anchor.astype(np.float32)

    nfix = canon_nfix(S)
    bounds = [(S * c) // nfix for c in range(nfix + 1)]

    out = np.empty((M, N), dtype=np.float32)
    for m in range(M):
        d = _dot32_all(slabs, anch, x[m], book, sub, acc_variant, tail_variant)  # [P,S,64]
        su = np.zeros((MMV_KSEG, P, PANEL_ROWS), dtype=np.float32)
        for kseg in range(MMV_KSEG):
            acc = np.zeros((P, PANEL_ROWS), dtype=np.float32)
            for c in range(nfix):
                t = np.zeros((P, PANEL_ROWS), dtype=np.float32)
                for kb in range(bounds[c] + kseg, bounds[c + 1], MMV_KSEG):
                    t = (t + d[:, kb, :]).astype(np.float32)
                acc = (acc + t).astype(np.float32)
            su[kseg] = acc
        u = np.zeros((P, PANEL_ROWS), dtype=np.float32)
        for s in range(MMV_KSEG):
            u = (u + su[s]).astype(np.float32)
        out[m] = u.reshape(N)
    return out


# ---------------------------------------------------------------------------------------
# Sharding, expressed in the ONLY two units that are legal: whole panels and whole slabs.
# ---------------------------------------------------------------------------------------
def shard_column(slabs, anchor, row_offset: int, row_size: int):
    """Column-parallel (output rows) split.  A panel is a self-contained contiguous byte
    range, so this is a pure whole-panel selection: no header fixup, no renumbering.

    row_offset/row_size are in WEIGHT ROWS and must both be multiples of 64.  vLLM's
    loader will divide them by packed_factor=64 itself
    (_adjust_shard_indexes_for_packing, parameter.py:605-610) -- and it uses
    `round(x // 64)`, which truncates a misaligned value WITHOUT raising.  That is the
    bug this function refuses to reproduce.
    """
    if row_offset % PANEL_ROWS or row_size % PANEL_ROWS:
        raise ValueError(
            f"column shard [{row_offset},{row_offset+row_size}) is not panel-aligned; "
            f"vLLM would silently truncate it to panel {row_offset // PANEL_ROWS}")
    p0, p1 = row_offset // PANEL_ROWS, (row_offset + row_size) // PANEL_ROWS
    if p1 > slabs.shape[0]:
        raise ValueError("column shard runs past the end of the tensor")
    return (np.ascontiguousarray(slabs[p0:p1]),
            np.ascontiguousarray(anchor[p0:p1]))


def shard_row(slabs, anchor, k_offset: int, k_size: int):
    """Row-parallel (input K) split.  Slabs are per-32-column and independent, and the
    fp16 row anchor has NO cross-K coupling, so a K split is a slab subrange plus a
    VERBATIM duplication of the 128 B header -- a byte gather, not a re-quantization.
    The numerics of each shard are bit-identical to the corresponding columns of the
    unsharded dequant; gate G3 proves it.

    The duplicated header is also free at load time: PXQ4AnchorParameter deliberately
    declares no `input_dim`, so RowvLLMParameter's loader falls through to
    BasevLLMParameter._assert_and_load (parameter.py:93-103) = full copy.
    """
    if k_offset % SLAB_COLS or k_size % SLAB_COLS:
        raise ValueError(
            f"row shard [{k_offset},{k_offset+k_size}) is not slab-aligned (32 columns)")
    s0, s1 = k_offset // SLAB_COLS, (k_offset + k_size) // SLAB_COLS
    if s1 > slabs.shape[1]:
        raise ValueError("row shard runs past the end of the tensor")
    return (np.ascontiguousarray(slabs[:, s0:s1, :]),
            np.ascontiguousarray(anchor))          # header duplicated, unchanged


def shard_bytes_overhead(N: int, K: int, tp: int) -> int:
    """Extra bytes across all ranks caused by duplicating the header on a K split."""
    return (tp - 1) * (N // PANEL_ROWS) * HEADER_BYTES


# ---------------------------------------------------------------------------------------
# vLLM loader arithmetic, modelled so the alignment gates can run with no vLLM installed.
# ---------------------------------------------------------------------------------------
def packed_shard_indices(shard_offset: int, shard_size: int, packed_factor: int = PANEL_ROWS):
    """_adjust_shard_indexes_for_packing, parameter.py:605-610 / linear.py:1053-1058.

    Returns (offset_units, size_units, exact) where `exact` is False iff the real vLLM
    code would have TRUNCATED.  vLLM raises nothing in that case; it produces a
    well-formed wrong slice.  Every caller here must assert exact is True.
    """
    off = shard_offset // packed_factor
    size = shard_size // packed_factor
    exact = (shard_offset % packed_factor == 0) and (shard_size % packed_factor == 0)
    return off, size, exact


def merged_column_shards(output_sizes, tp_size: int, tp_rank: int):
    """MergedColumnParallelLinear.weight_loader_v2, linear.py:1140-1205.

    For each fused sub-shard i:  shard_offset = sum(output_sizes[:i]) // tp_size
                                 shard_size   = output_sizes[i]      // tp_size
    Note the offset is the sum of the ALREADY-DIVIDED sizes, so a per-shard remainder
    poisons every later offset too.  Yields (i, shard_offset, shard_size).
    """
    out = []
    off = 0
    for i, o in enumerate(output_sizes):
        sz = o // tp_size
        out.append((i, off, sz))
        off += sz
    return out


def check_module_shardable(name: str, output_sizes, K: int, tp_sizes=(1, 2, 4),
                           row_parallel: bool = False):
    """Converter/config self-check #5 (plan §5.6).  Returns a list of failure strings."""
    fails = []
    for tp in tp_sizes:
        for i, off, sz in merged_column_shards(output_sizes, tp, 0):
            if output_sizes[i] % tp:
                fails.append(f"{name}: TP={tp} sub-shard {i} size {output_sizes[i]} not divisible by tp")
            if sz % PANEL_ROWS:
                fails.append(f"{name}: TP={tp} sub-shard {i} rows/rank={sz} not %64")
            _, _, exact = packed_shard_indices(off, sz)
            if not exact:
                fails.append(f"{name}: TP={tp} sub-shard {i} offset {off} would be TRUNCATED by vLLM")
        if row_parallel:
            if K % tp:
                fails.append(f"{name}: TP={tp} K={K} not divisible by tp")
            elif (K // tp) % SLAB_COLS:
                fails.append(f"{name}: TP={tp} K/rank={K // tp} not %32")
    return fails


def check_tables_against_gguf(kv: dict):
    """Gate: the file records the tables it was built with.  PXA_PXQ6_BOOK /
    PXA_PXQ6_SUB env overrides at quantize time would make the compiled-in tables wrong
    for THIS file, and the failure mode is a plausible-looking model, so this is checked
    rather than trusted.  llama-quantize.cpp:1980-1983 writes the KVs."""
    problems = []
    for key, ours in (("pxa.pxq6.book", BOOK), ("pxa.pxq6.sub", SUB)):
        got = kv.get(key)
        if got is None:
            problems.append(f"{key}: MISSING from the GGUF")
            continue
        got = np.asarray(got, dtype=np.float32)
        if got.shape != ours.shape:
            problems.append(f"{key}: length {got.shape} != {ours.shape}")
        elif not np.array_equal(got, ours):
            bad = [(i, float(a), float(b)) for i, (a, b) in enumerate(zip(got, ours)) if a != b]
            problems.append(f"{key}: differs from the compiled-in table at {bad}")
    return problems


def match_mmv_variants(got_f16, x, slabs, anchor, *, first_only: bool = True):
    """Find which (acc_variant, tail_variant) foldings of `mmv` reproduce `got_f16`.

    Returns (hits, n_tried).  `first_only` stops at the first match, which is the normal
    case and keeps the search from costing 9 full model evaluations on a 17408-wide
    tensor; pass False when the COUNT matters (it measures how discriminating the test
    is -- 9 hits means fp16 rounding absorbed the contraction difference and the test
    could not tell the variants apart).
    """
    import numpy as _np
    hits, tried = [], 0
    for av in ACC_VARIANTS:
        for tv in TAIL_VARIANTS:
            tried += 1
            ref = mmv(_np.asarray(x, dtype=_np.float32), slabs, anchor,
                      acc_variant=av, tail_variant=tv).astype(_np.float16)
            if _np.array_equal(ref.view(_np.uint16), _np.asarray(got_f16).view(_np.uint16)):
                hits.append((av, tv))
                if first_only:
                    return hits, tried
    return hits, tried
