"""layout.py — PXQ4 (ggml type id 252) panel/slab arithmetic. Plan §6.2.

SHARED CONTRACT SURFACE. This module is imported by the converter, by the runtime package's
parameter shaping, and by the tests. It must never import torch, vllm or CUDA — gates G1-G4
run on a laptop, before the GPU lease is ever taken.

THE LAYOUT, and why every function here is index arithmetic and never a value computation:

    tensor  = P panels, row-major in P                     P = N/64
    panel   = 128 B anchor header + S slabs, K-major       S = K/32
    header  = 64 x fp16 row anchors, anchor[r] at byte 2*r
    slab    = 64 B sub-scale SoA (byte r is row r's scale byte for THIS 32-column block:
                low nibble -> elements 0..15, high nibble -> elements 16..31)
            + 64 code rows of 16 B (row r at slab[64 + 16*r];
                byte b holds code(2b) in its low nibble and code(2b+1) in its high nibble)

    ggml/src/pxq-cpu.h:1-17, ggml/src/ggml-cuda/pxq6.cuh:8-18, :317-346, :520-526.
    Addressing verified byte-exactly against all 866 tensors of the real artifact: the
    derived on-disk length equals (N/64)*(128 + (K/32)*1088) for every one of the six PXQ4
    shapes present, with no inter-tensor padding.

    FILENAME TRAP: the kernels for id 252 live in pxq6.cuh. ggml-cuda/pxq4.cuh documents the
    RETIRED id-250 MXFP4-repack format and is NOT this layout. Do not port from it.

WHY THE SHARDING RULES FALL OUT OF THE ADDRESSING, WITH NO REQUANTIZATION:

  * COLUMN-parallel (split output rows N). A panel is a self-contained contiguous byte range
    holding its own anchors and all its codes. Any boundary at a multiple of 64 rows is a
    whole number of panels, so the shard is a memcpy — byte-identical, and itself a valid
    PXQ4 tensor.
  * ROW-parallel (split the contraction dim K). Slabs are per-32-column and carry their own
    sub-scale nibbles; the fp16 row anchor has NO cross-K coupling (pxq-cpu.c:143-156 reads
    it once per row, outside the kb loop). A K-split at a multiple of 32 is therefore a
    byte-gather: take the same slab subrange out of every panel and duplicate the 128 B
    header verbatim. Bit-identical numerics, +16/K' bpw for the duplicated header.
  * ILLEGAL: any row boundary not a multiple of 64, any K boundary not a multiple of 32, or
    treating a weight row as a contiguous byte run. The last is why vLLM's generic GGUF
    sharder cannot be used at all, and why ggml's own to_float is NULL for these types
    (ggml.c:1407-1414).

  The alignment rule is enforced by assertion rather than trusted, because the failure is
  SILENT: vLLM's `_adjust_shard_indexes_for_packing` does `round(shard_size // packed_factor)`
  (parameter.py:605-610), so a misaligned offset truncates without raising and yields a
  well-formed wrong slice — a model that loads cleanly and produces subtly wrong logits.
"""

from __future__ import annotations

import numpy as np

TYPE_ID = 252

PANEL_ROWS = 64          # PXQ6_BM              (ggml-pxq6-tables.h:24)
SLAB_COLS = 32           # PXQ6_QK              (ggml-pxq6-tables.h:22)
SLAB_BYTES = 1088        # PXQ6_SLAB_BYTES      (ggml-pxq6-tables.h:25)
HEADER_BYTES = 128       # PXQ6_HDR_BYTES       (ggml-pxq6-tables.h:26)
ROW_META = 2             # PXQ6_ROW_META        (ggml-pxq6-tables.h:27)
CODE_OFF = 64            # scale SoA occupies slab[0:64]; code rows start here
CODE_BYTES = 16          # 16 B of nibbles per row per 32-column block
TYPE_SIZE = 17           # PXQ6_TYPE_SIZE: bytes per 32 elements, excluding row meta


def panel_bytes(K: int) -> int:
    """Bytes in one 64-row panel covering K columns."""
    if K % SLAB_COLS:
        raise ValueError(f"pxq4: K={K} is not a multiple of {SLAB_COLS}")
    return HEADER_BYTES + (K // SLAB_COLS) * SLAB_BYTES


def tensor_bytes(N: int, K: int) -> int:
    """On-disk size of a whole [N, K] PXQ4 tensor.

    Equals ggml's own accounting, ``N * ggml_row_size(PXQ4, K)`` = ``N * (2 + 17*K/32)``,
    which is where the documented ``4.25 + 16/K`` bpw comes from (ggml.h:465-467).
    """
    assert_geometry(N, K)
    return (N // PANEL_ROWS) * panel_bytes(K)


def bits_per_weight(K: int) -> float:
    return 4.25 + 16.0 / K


def slab_shape(N: int, K: int) -> tuple[int, int, int]:
    assert_geometry(N, K)
    return (N // PANEL_ROWS, K // SLAB_COLS, SLAB_BYTES)


def anchor_shape(N: int) -> tuple[int, int]:
    if N % PANEL_ROWS:
        raise ValueError(f"pxq4: N={N} is not a multiple of {PANEL_ROWS}")
    return (N // PANEL_ROWS, PANEL_ROWS)


def assert_geometry(N: int, K: int) -> None:
    """The quantizer's eligibility gate, restated as a load-time invariant.

    ``pxq*_tensor_eligible`` requires ``ne[1] % 64 == 0 && ne[0] % 32 == 0`` and demotes the
    tensor to q8_0 otherwise; the CUDA dequant kernels hard-abort on the same condition
    (pxq-cpu.h:44-47). Anything reaching this function has already claimed to be PXQ4, so a
    violation means a corrupt file or a bad shard boundary, not a demotion.
    """
    if N <= 0 or K <= 0:
        raise ValueError(f"pxq4: non-positive geometry N={N} K={K}")
    if N % PANEL_ROWS:
        raise ValueError(
            f"pxq4: N={N} is not a multiple of {PANEL_ROWS} — a partial panel has no valid "
            f"anchor header and cannot be addressed")
    if K % SLAB_COLS:
        raise ValueError(
            f"pxq4: K={K} is not a multiple of {SLAB_COLS} — a partial slab has no valid "
            f"sub-scale byte")


def assert_shardable(N: int, K: int, tp_sizes=(1, 2, 4), *, row_parallel: bool = False,
                     name: str = "<tensor>") -> None:
    """Refuse a tensor that cannot be split at every TP degree we intend to serve.

    Checked at CONVERSION time, not at load time, because a converter that emits an
    unshardable tensor has produced a checkpoint that will fail (or worse, silently truncate)
    on a machine we may not have. The column check is the one that matters for both axes:
    every ``output_partition_size`` must stay a whole number of panels, and for a
    RowParallelLinear the per-rank K must stay a whole number of slabs.
    """
    for tp in tp_sizes:
        if N % tp == 0 and (N // tp) % PANEL_ROWS:
            raise ValueError(
                f"{name}: column shard at TP={tp} gives {N // tp} rows/rank, not a multiple "
                f"of {PANEL_ROWS}")
        if row_parallel:
            if K % tp:
                raise ValueError(f"{name}: K={K} not divisible by TP={tp}")
            if (K // tp) % SLAB_COLS:
                raise ValueError(
                    f"{name}: row shard at TP={tp} gives K={K // tp}/rank, not a multiple "
                    f"of {SLAB_COLS}")


# ---------------------------------------------------------------------------------------------
# blob <-> (slabs, anchor). Plan §5.3: this is a PURE SPLIT. No byte is reordered and no value
# is recomputed — the two emitted tensors are the header bytes and the slab bytes of each
# panel, reinterpreted. That is what makes the emitted checkpoint provably the same weights as
# the GGUF, verifiable by a byte comparison rather than by a numeric tolerance.
# ---------------------------------------------------------------------------------------------
def split_blob(blob, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF panel blob -> (slabs uint8[P,S,1088], anchor float16[P,64])."""
    assert_geometry(N, K)
    P, S = N // PANEL_ROWS, K // SLAB_COLS
    need = tensor_bytes(N, K)
    a = np.frombuffer(blob, dtype=np.uint8)
    if a.size != need:
        raise ValueError(f"pxq4: blob is {a.size} B, expected {need} B for N={N} K={K}")
    a = a.reshape(P, HEADER_BYTES + S * SLAB_BYTES)
    # .copy() is deliberate: the source is a read-only mmap of the GGUF, and safetensors
    # writing plus the shard tests both want owned, C-contiguous arrays.
    anchor = a[:, :HEADER_BYTES].copy().view("<f2")
    slabs = a[:, HEADER_BYTES:].copy().reshape(P, S, SLAB_BYTES)
    if anchor.shape != (P, PANEL_ROWS):
        raise AssertionError(f"pxq4: anchor reinterpret gave {anchor.shape}")
    return slabs, anchor


def join_blob(slabs: np.ndarray, anchor: np.ndarray) -> bytes:
    """(slabs, anchor) -> the original GGUF panel blob. Exact inverse of ``split_blob``."""
    if slabs.dtype != np.uint8 or slabs.ndim != 3 or slabs.shape[2] != SLAB_BYTES:
        raise ValueError(f"pxq4: slabs must be uint8 [P,S,{SLAB_BYTES}], got "
                         f"{slabs.dtype} {slabs.shape}")
    P, S, _ = slabs.shape
    if anchor.dtype != np.float16 or anchor.shape != (P, PANEL_ROWS):
        raise ValueError(f"pxq4: anchor must be float16 [{P},{PANEL_ROWS}], got "
                         f"{anchor.dtype} {anchor.shape}")
    hdr = np.ascontiguousarray(anchor).view(np.uint8).reshape(P, HEADER_BYTES)
    return np.concatenate([hdr, slabs.reshape(P, S * SLAB_BYTES)], axis=1).tobytes()


# ---------------------------------------------------------------------------------------------
# shard helpers — the reference implementations of what vLLM's stock loaders do to our two
# parameters. The tests use these to prove shard-then-dequant == dequant-then-shard (gate G3).
# ---------------------------------------------------------------------------------------------
def shard_columns(slabs: np.ndarray, anchor: np.ndarray, row_beg: int, row_end: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Column-parallel shard: keep output rows [row_beg, row_end).

    This is exactly ``_ColumnvLLMParameter.load_column_parallel_weight`` narrowing dim 0 after
    ``_adjust_shard_indexes_for_packing`` divides by ``packed_factor=64`` — i.e. a narrow in
    PANEL units. The anchor rides along on the same axis, which is why the header travels
    with its panel for free.
    """
    if row_beg % PANEL_ROWS or row_end % PANEL_ROWS:
        raise ValueError(f"pxq4: column shard [{row_beg},{row_end}) is not panel-aligned "
                         f"(multiples of {PANEL_ROWS})")
    p0, p1 = row_beg // PANEL_ROWS, row_end // PANEL_ROWS
    return slabs[p0:p1].copy(), anchor[p0:p1].copy()


def shard_k(slabs: np.ndarray, anchor: np.ndarray, k_beg: int, k_end: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Row-parallel shard: keep input columns [k_beg, k_end).

    ``RowvLLMParameter.load_row_parallel_weight`` narrows dim ``input_dim`` and never consults
    ``packed_factor`` (parameter.py:220-230) — so with dim 1 already in SLAB units the split
    lands on slab boundaries for free. The anchor parameter deliberately declares no
    ``input_dim``, so it falls through to ``BasevLLMParameter._assert_and_load`` and is
    full-copied to every rank: that full copy IS the required header duplication.
    """
    if k_beg % SLAB_COLS or k_end % SLAB_COLS:
        raise ValueError(f"pxq4: K shard [{k_beg},{k_end}) is not slab-aligned "
                         f"(multiples of {SLAB_COLS})")
    s0, s1 = k_beg // SLAB_COLS, k_end // SLAB_COLS
    return slabs[:, s0:s1].copy(), anchor.copy()


# ---------------------------------------------------------------------------------------------
# HEAD-ORDER PERMUTATION (see namemap.gdn_permutation).
#
# ggml and HF order the 48 GDN value-heads differently, so the converter has to reorder whole
# head blocks on its way out. Both directions stay a PURE BYTE MOVE, for the same reason the
# TP shards do:
#
#   * a 128-row head block is exactly 2 panels, and every GDN block boundary that needs
#     moving (row 4096 of attn_qkv, row 0 of attn_gate) is panel-aligned, so an output-axis
#     permutation is a gather on the PANEL axis of ``slabs``/``anchor``;
#   * a 128-column head block is exactly 4 slabs, so a contraction-axis permutation is a
#     gather on the SLAB axis of ``slabs`` and leaves ``anchor`` untouched (the fp16 row
#     anchor has no cross-K coupling — same property that makes the K shard free).
#
# Neither touches a nibble, a sub-scale or an anchor VALUE, so the permuted tensor is still
# bit-identical to the source weights and still a valid PXQ4 tensor. ``unpermute_index``
# exists so the byte round-trip gate can undo the move and still compare against the file.
# ---------------------------------------------------------------------------------------------
def assert_permutation(index: np.ndarray, n: int, what: str = "index") -> np.ndarray:
    """Refuse anything that is not a bijection of ``range(n)``.

    A gather that merely *looks* plausible — a duplicated head, a dropped head — produces a
    checkpoint that loads and generates fluent garbage, which is precisely the failure this
    whole permutation exists to fix. So it is checked, not trusted.
    """
    idx = np.asarray(index, dtype=np.int64)
    if idx.ndim != 1 or idx.size != n:
        raise ValueError(f"{what}: expected a 1-D index of length {n}, got shape {idx.shape}")
    if not np.array_equal(np.sort(idx), np.arange(n, dtype=np.int64)):
        raise ValueError(f"{what}: not a permutation of range({n}) — it drops or duplicates "
                         f"entries, which would silently corrupt weights")
    return idx


def unpermute_index(index: np.ndarray) -> np.ndarray:
    """The inverse gather. ``x[index][unpermute_index(index)] == x``."""
    idx = np.asarray(index, dtype=np.int64)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(idx.size, dtype=np.int64)
    return inv


def block_gather_to_panels(row_gather: np.ndarray) -> np.ndarray:
    """Row-block gather (units of ``PANEL_ROWS``) -> panel-axis gather.

    ``row_gather`` is indexed in whole rows and must be constant-per-panel; this expands it to
    the panel index array that ``gather_panels`` wants. Any row permutation that is not
    panel-aligned is rejected here rather than at load, because vLLM would not raise.
    """
    g = np.asarray(row_gather, dtype=np.int64)
    n = g.size
    if n % PANEL_ROWS:
        raise ValueError(f"pxq4: row gather of length {n} is not a multiple of {PANEL_ROWS}")
    g = g.reshape(n // PANEL_ROWS, PANEL_ROWS)
    base = g[:, :1]
    if not np.array_equal(g, base + np.arange(PANEL_ROWS, dtype=np.int64)):
        raise ValueError(
            "pxq4: this row permutation is not panel-aligned — it would move rows within a "
            "64-row panel, which is not a byte move and would need a re-quantization")
    if np.any(base % PANEL_ROWS):
        raise ValueError("pxq4: row permutation sources a panel at a non-panel-aligned row")
    return assert_permutation((base[:, 0] // PANEL_ROWS), n // PANEL_ROWS, "panel gather")


def col_gather_to_slabs(col_gather: np.ndarray) -> np.ndarray:
    """Column gather (units of single columns) -> slab-axis gather. Same contract, K axis."""
    g = np.asarray(col_gather, dtype=np.int64)
    n = g.size
    if n % SLAB_COLS:
        raise ValueError(f"pxq4: column gather of length {n} is not a multiple of {SLAB_COLS}")
    g = g.reshape(n // SLAB_COLS, SLAB_COLS)
    base = g[:, :1]
    if not np.array_equal(g, base + np.arange(SLAB_COLS, dtype=np.int64)):
        raise ValueError(
            "pxq4: this column permutation is not slab-aligned — it would move columns inside "
            "a 32-column block, which changes which values share a sub-scale")
    if np.any(base % SLAB_COLS):
        raise ValueError("pxq4: column permutation sources a slab at a non-slab-aligned column")
    return assert_permutation((base[:, 0] // SLAB_COLS), n // SLAB_COLS, "slab gather")


def gather_panels(slabs: np.ndarray, anchor: np.ndarray, panel_index: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Reorder whole panels. Output-axis permutation; the header travels with its panel."""
    idx = assert_permutation(panel_index, slabs.shape[0], "panel gather")
    return np.ascontiguousarray(slabs[idx]), np.ascontiguousarray(anchor[idx])


def gather_slabs(slabs: np.ndarray, slab_index: np.ndarray) -> np.ndarray:
    """Reorder whole slabs within every panel. Contraction-axis permutation; anchor unchanged."""
    idx = assert_permutation(slab_index, slabs.shape[1], "slab gather")
    return np.ascontiguousarray(slabs[:, idx])


# ---------------------------------------------------------------------------------------------
# 3D (expert-stacked) support.
#
# THIS MODEL HAS NO EXPERTS. Verified against the artifact itself, not assumed: 866 tensors,
# zero ``*_exps``, zero expert KVs, ffn_gate/up/down dense on all 65 blocks. The functions
# below exist because the PXQ4 addressing for an expert stack is a documented outer dimension
# (``panel = W + (e*panels + p)*panel_bytes``, pxq6.cuh:520-526) and a converter that silently
# mis-handled a 3D tensor would be worse than one that refuses; convert.py refuses by default.
# ---------------------------------------------------------------------------------------------
def expert_stack_bytes(E: int, N: int, K: int) -> int:
    return E * tensor_bytes(N, K)


def split_blob_3d(blob, E: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Expert-stacked blob -> (slabs uint8[E,P,S,1088], anchor float16[E,P,64]).

    Experts are the SLOWEST-varying axis and each expert slice is a complete, independently
    addressable PXQ4 tensor, so the split is the 2D split applied per expert.
    """
    a = np.frombuffer(blob, dtype=np.uint8)
    need = expert_stack_bytes(E, N, K)
    if a.size != need:
        raise ValueError(f"pxq4: expert blob is {a.size} B, expected {need} B "
                         f"for E={E} N={N} K={K}")
    per = tensor_bytes(N, K)
    sl, an = [], []
    for e in range(E):
        s, h = split_blob(a[e * per:(e + 1) * per], N, K)
        sl.append(s)
        an.append(h)
    return np.stack(sl), np.stack(an)
