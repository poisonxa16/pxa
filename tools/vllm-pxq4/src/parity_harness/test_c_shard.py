"""
GATE G3 -- SHARDED parity.  The one that catches the worst bug class.

A PXQ4 weight ROW is not a contiguous byte range: its 16-byte code rows are scattered one
per slab across the whole 64-row panel (pxq-cpu.h:5-9, which is also why ggml's own
to_float/vec_dot traits are NULL for these types).  Every generic row-slicing sharder in
existence -- including vLLM's GGUF one -- therefore produces a WELL-FORMED WRONG SLICE.
Well-formed is the problem: the model loads cleanly, every shape checks out, nothing
raises, and the logits are subtly wrong.  vLLM's own packing adjustment makes it worse:
`round(shard_size // packed_factor)` (parameter.py:605-610, linear.py:1053-1058) truncates
a misaligned offset in silence.

So the invariant under test is stronger than "close enough":

    dequant(shard(W))  ==  shard(dequant(W))      BITWISE, on both axes, at TP in {1,2,4}

If that holds, the TP repack is a permutation of bytes and not a re-quantization, and
every downstream numeric question is a normal float question.  If it does not, no amount
of end-to-end eval will localise the bug.

The gate is run in BOTH representations, because they are the two things a converter bug
can disagree about:
  * emitted space -- slabs[P,S,1088] / anchor[P,64], what vLLM's narrow() sees
  * blob space    -- raw GGUF bytes, what the byte-gather repack produces
"""

from __future__ import annotations

import numpy as np

from . import compare, fixtures
from . import oracle as O
from .test_a_dequant import Skip

TP_SIZES = (1, 2, 4)

# Every real PXQ4-served module, with the axis vLLM splits it on.
# (module, output_sizes on disk, K, row_parallel)  -- plan §2.3, §5.4.
REAL_MODULES = [
    ("mlp.gate_up_proj",          [17408, 17408], 5120,  False),
    ("mlp.down_proj",             [5120],         17408, True),
    ("self_attn.o_proj",          [5120],         6144,  True),
    ("linear_attn.in_proj_qkvz",  [2048, 2048, 6144, 6144], 5120, False),
    ("linear_attn.out_proj",      [5120],         6144,  True),   # P2a
    ("self_attn.qkv_proj",        [12288, 1024, 1024], 5120, False),  # P2c
    ("lm_head",                   [248320],       5120,  False),  # P2b
]

# The GDN b/a projection MUST stay out of the fused quantized module.
# _uses_split_gdn_input_projections (qwen3_5.py:127-157) only splits it out if the quant
# config's `ignore` names linear_attn.in_proj_a / in_proj_b.  Left fused, in_proj_qkvz
# gains two 48-row shards -> 12 rows/rank at TP=4 -> not panel-aligned -> silent
# truncation.  This is the fused variant the config must never produce.
FUSED_QKVZBA_TRAP = ("linear_attn.in_proj_qkvzba", [2048, 2048, 6144, 6144, 48, 48], 5120)


def _cases(real=None):
    for label in ("narrowK", "wideK", "oddpanels", "shardable4"):
        N, K = fixtures.SMALL_SHAPES[label]
        slabs, anchor = fixtures.synth_parts(N, K, seed=0xC0 ^ (abs(hash(label)) % 7919),
                                            profile="extreme")
        yield (f"synth:{label}", N, K, slabs, anchor)
    if real:
        for label, v in real.items():
            if label.startswith("__"):
                continue
            N, K, slabs, anchor = v
            yield (f"real:{label}", N, K, slabs, anchor)


# ---------------------------------------------------------------------------------------
# G3a -- column-parallel (split output rows)
# ---------------------------------------------------------------------------------------
def test_g3_column_shard_bitexact(real=None):
    for label, N, K, slabs, anchor in _cases(real):
        full = O.dequant(slabs, anchor)
        for tp in TP_SIZES:
            if N % tp or (N // tp) % O.PANEL_ROWS:
                continue                      # not a legal split for this shape
            rows = N // tp
            pieces = []
            for r in range(tp):
                s, a = O.shard_column(slabs, anchor, r * rows, rows)
                # A column shard is a whole-panel selection, so the shard is itself a
                # valid standalone PXQ4 tensor with the SAME K -- assert that, because
                # it is the property the converter relies on to slice by memcpy.
                assert s.shape == (rows // O.PANEL_ROWS, K // O.SLAB_COLS, O.SLAB_BYTES)
                assert a.shape == (rows // O.PANEL_ROWS, O.PANEL_ROWS)
                d = O.dequant(s, a)
                want = full[r * rows:(r + 1) * rows]
                assert compare.bitwise_equal(d, want), (
                    f"[{label}] TP={tp} rank={r} column shard is not a slice of the "
                    f"unsharded dequant\n" + compare.bit_diff_report(want, d, "full[slice]", "shard"))
                pieces.append(d)
            assert compare.bitwise_equal(np.concatenate(pieces, axis=0), full), (
                f"[{label}] TP={tp} concatenated column shards != full")


# ---------------------------------------------------------------------------------------
# G3b -- row-parallel (split K)
# ---------------------------------------------------------------------------------------
def test_g3_row_shard_bitexact(real=None):
    for label, N, K, slabs, anchor in _cases(real):
        full = O.dequant(slabs, anchor)
        for tp in TP_SIZES:
            if K % tp or (K // tp) % O.SLAB_COLS:
                continue
            kk = K // tp
            pieces = []
            for r in range(tp):
                s, a = O.shard_row(slabs, anchor, r * kk, kk)
                # The anchor is duplicated VERBATIM -- there is no cross-K coupling in
                # the fp16 row anchor, so the shard's own dequant reproduces the parent's
                # columns exactly.  This is the entire argument for "byte gather, not
                # re-quantization".
                assert compare.bitwise_equal(a, anchor), (
                    f"[{label}] row shard modified the anchor header; it must be copied "
                    f"unchanged to every rank")
                d = O.dequant(s, a)
                want = full[:, r * kk:(r + 1) * kk]
                assert compare.bitwise_equal(d, want), (
                    f"[{label}] TP={tp} rank={r} row shard is not a column-slice of the "
                    f"unsharded dequant\n" + compare.bit_diff_report(want, d, "full[:,slice]", "shard"))
                pieces.append(d)
            assert compare.bitwise_equal(np.concatenate(pieces, axis=1), full), (
                f"[{label}] TP={tp} concatenated row shards != full")


def test_g3_row_shard_header_overhead(real=None, report=None):
    """The K split duplicates the 128 B header per panel per rank.  Quantify it so the
    "+0.60 MiB/rank at TP=4" claim in the sharding verdict stays checkable rather than
    remembered."""
    total = 0
    for name, sizes, K, row_par in REAL_MODULES:
        if not row_par:
            continue
        N = sum(sizes)
        total += O.shard_bytes_overhead(N, K, 4)
    if report is not None:
        report.append(("row-shard header overhead TP=4, per-layer row-parallel modules",
                       {"bytes": total, "MiB": total / 2 ** 20}))
    assert total > 0


# ---------------------------------------------------------------------------------------
# G3c -- the two representations must agree
# ---------------------------------------------------------------------------------------
def _blob_row_shard(blob, N, K, k_off, k_size):
    """Byte-gather a K shard directly out of the raw GGUF bytes: per panel, take the
    128 B header verbatim, then the contiguous run of slabs [k_off/32, (k_off+k_size)/32).
    This is what the converter does; the emitted-space path is what vLLM's loader does.
    They must produce the same bytes."""
    P = N // O.PANEL_ROWS
    pb = O.panel_bytes(K)
    s0, s1 = k_off // O.SLAB_COLS, (k_off + k_size) // O.SLAB_COLS
    a = np.frombuffer(blob, dtype=np.uint8).reshape(P, pb)
    hdr = a[:, :O.HEADER_BYTES]
    body = a[:, O.HEADER_BYTES:].reshape(P, K // O.SLAB_COLS, O.SLAB_BYTES)[:, s0:s1, :]
    out = np.concatenate([hdr, body.reshape(P, (s1 - s0) * O.SLAB_BYTES)], axis=1)
    return np.ascontiguousarray(out).reshape(-1)


def _blob_column_shard(blob, N, K, row_off, row_size):
    """Column shard in blob space: a contiguous byte range of whole panels.  Nothing
    else.  If this ever needs more than a slice, the format claim is wrong."""
    pb = O.panel_bytes(K)
    p0 = row_off // O.PANEL_ROWS
    p1 = (row_off + row_size) // O.PANEL_ROWS
    a = np.frombuffer(blob, dtype=np.uint8)
    return np.ascontiguousarray(a[p0 * pb:p1 * pb])


def test_g3_blob_and_emitted_shards_agree(real=None):
    for label, N, K, slabs, anchor in _cases(real):
        blob = O.join_blob(slabs, anchor)
        for tp in TP_SIZES:
            if N % tp == 0 and (N // tp) % O.PANEL_ROWS == 0:
                rows = N // tp
                for r in range(tp):
                    bs = _blob_column_shard(blob, N, K, r * rows, rows)
                    es = O.join_blob(*O.shard_column(slabs, anchor, r * rows, rows))
                    assert np.array_equal(bs, es), (
                        f"[{label}] TP={tp} rank={r}: blob-space and emitted-space "
                        f"column shards differ")
                    # And the blob shard must still parse as a standalone PXQ4 tensor.
                    s2, a2 = O.split_blob(bs, rows, K)
                    assert compare.bitwise_equal(O.dequant(s2, a2),
                                                 O.dequant(slabs, anchor)[r * rows:(r + 1) * rows])
            if K % tp == 0 and (K // tp) % O.SLAB_COLS == 0:
                kk = K // tp
                for r in range(tp):
                    bs = _blob_row_shard(blob, N, K, r * kk, kk)
                    es = O.join_blob(*O.shard_row(slabs, anchor, r * kk, kk))
                    assert np.array_equal(bs, es), (
                        f"[{label}] TP={tp} rank={r}: blob-space and emitted-space "
                        f"row shards differ")
                    assert bs.size == O.tensor_bytes(N, kk), (
                        f"[{label}] a K-shard must itself be a valid PXQ4 tensor at K={kk}")


# ---------------------------------------------------------------------------------------
# G3d -- the fused (merged-column) modules, assembled exactly as vLLM assembles them
# ---------------------------------------------------------------------------------------
def test_g3_merged_column_assembly(real=None):
    """gate_up_proj and in_proj_qkvz are fused in vLLM but SEPARATE on disk (plan §5.4:
    "the converter emits them separately; do not pre-fuse").  Each disk tensor is
    column-sharded independently and dropped at its own offset in the fused parameter.
    Model the whole assembly and check every rank's fused weight against the fused
    ground truth."""
    K = 512
    # Sub-tensor row counts mirroring in_proj_qkvz's structure [2048,2048,6144,6144]
    # divided by 8: two small, two large, and -- the property that matters -- every one
    # still a whole number of 64-row panels after division by TP=4.  The real module
    # gives 512/512/1536/1536 rows per rank at TP=4; this gives 64/64/192/192.
    output_sizes = [256, 256, 768, 768]
    parts = []
    for i, n in enumerate(output_sizes):
        parts.append(fixtures.synth_parts(n, K, seed=0xD0 + i))
    full_deq = [O.dequant(s, a) for s, a in parts]
    fused_full = np.concatenate(full_deq, axis=0)

    for tp in TP_SIZES:
        rank_weights = []
        for rank in range(tp):
            chunks = []
            for i, (n, (s, a)) in enumerate(zip(output_sizes, parts)):
                sz = n // tp
                off, size_u, exact = O.packed_shard_indices(rank * sz, sz)
                assert exact, (
                    f"TP={tp} sub-shard {i}: offset {rank*sz} size {sz} is not a whole "
                    f"number of 64-row panels; vLLM would TRUNCATE this silently "
                    f"(parameter.py:605-610)")
                ss, aa = O.shard_column(s, a, rank * sz, sz)
                chunks.append(O.dequant(ss, aa))
            rank_weights.append(np.concatenate(chunks, axis=0))
        # Reassembling the ranks is NOT a plain concatenation of rank blocks: vLLM
        # interleaves by sub-shard.  Rebuild the fused matrix the way an all-gather
        # would and compare to the fused ground truth.
        rebuilt = []
        cursor = [0] * len(output_sizes)
        for i, n in enumerate(output_sizes):
            per = n // tp
            for rank in range(tp):
                off = sum(output_sizes[j] // tp for j in range(i))
                rebuilt.append(rank_weights[rank][off:off + per])
            cursor[i] += n
        assert compare.bitwise_equal(np.concatenate(rebuilt, axis=0), fused_full), (
            f"TP={tp} merged-column reassembly != fused ground truth")


def test_g3_qkv_shard_arithmetic():
    """QKVParallelLinear splits q, k and v as three independent shard ids
    (linear.py:1538-1546, :1596-1600), so the fused [q;k;v] never has to be cut at a
    non-panel boundary as long as each of q, k, v is individually panel-aligned per rank.
    Check the real numbers: q 12288, k 1024, v 1024 at TP in {1,2,4}."""
    for tp in TP_SIZES:
        for name, n in (("q", 12288), ("k", 1024), ("v", 1024)):
            assert n % tp == 0, f"{name}={n} not divisible by TP={tp}"
            per = n // tp
            assert per % O.PANEL_ROWS == 0, (
                f"TP={tp} {name}: {per} rows/rank is not %64 -- qkv_proj cannot be PXQ4")
    # attn_q is gate-fused: 12288 = 2 * (24 heads * 256 head_dim), stored per-head
    # interleaved [q_h(256) | gate_h(256)] (llama-build-context.cpp:2003-2007 ==
    # qwen3_next.py:564-571).  A TP shard must therefore contain whole 512-row head
    # PAIRS or the interleave is cut in half.
    for tp in TP_SIZES:
        per = 12288 // tp
        assert per % 512 == 0, (
            f"TP={tp}: attn_q shard of {per} rows cuts a (q,gate) head pair in half")
        assert per % O.PANEL_ROWS == 0
        # 512 is a multiple of 64, so panel alignment and head-pair alignment can never
        # disagree here.  Recording it means a future head_dim change trips this test.
    assert 512 % O.PANEL_ROWS == 0


def test_g3_real_module_alignment():
    """Every PXQ4-served module, at TP 1/2/4, on the axis vLLM actually splits."""
    problems = []
    for name, sizes, K, row_par in REAL_MODULES:
        problems += O.check_module_shardable(name, sizes, K, TP_SIZES, row_parallel=row_par)
    assert not problems, "shard-alignment failures:\n  " + "\n  ".join(problems)


def test_g3_fused_gdn_ba_is_a_trap():
    """Prove the trap is real rather than asserting it in a comment.

    If the quant config fails to list linear_attn.in_proj_a / in_proj_b in `ignore`,
    _uses_split_gdn_input_projections (qwen3_5.py:127-157) returns False and the 48-row
    b and a projections are folded into the fused, quantized in_proj_qkvz.  At TP=4 that
    is 12 rows per rank.  This test asserts the failure IS detected by
    check_module_shardable -- i.e. that our own gate would catch the misconfiguration.
    """
    name, sizes, K = FUSED_QKVZBA_TRAP
    problems = O.check_module_shardable(name, sizes, K, (2, 4))
    assert problems, (
        "check_module_shardable did NOT flag the fused-ba layout. The gate is broken: "
        "48 rows / 4 ranks = 12 is not a whole panel and must be rejected.")
    assert any("12" in p or "24" in p for p in problems), problems


def test_g3_misalignment_is_refused_not_truncated():
    """The failure mode, demonstrated.

    vLLM computes `shard_size // packed_factor` and does not check the remainder, so a
    12-row shard of a 64-row-packed parameter becomes 0 panels: an empty, well-formed,
    completely wrong slice.  oracle.shard_column raises instead.  Both behaviours are
    pinned here so the difference cannot quietly disappear.
    """
    N, K = 256, 512
    slabs, anchor = fixtures.synth_parts(N, K, seed=9)

    for bad_off, bad_size in ((0, 12), (32, 64), (64, 100), (0, 63)):
        try:
            O.shard_column(slabs, anchor, bad_off, bad_size)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shard_column accepted misaligned ({bad_off},{bad_size})")
        off_u, size_u, exact = O.packed_shard_indices(bad_off, bad_size)
        assert not exact, f"packed_shard_indices called ({bad_off},{bad_size}) exact"
    # ... and the specific, dangerous case: it truncates to nothing without complaint.
    off_u, size_u, exact = O.packed_shard_indices(0, 12)
    assert (off_u, size_u, exact) == (0, 0, False), (off_u, size_u, exact)

    # (32, 32) is LEGAL -- 32 columns is exactly one slab.  The illegal cases are the
    # ones whose offset or size is not a whole number of 32-column slabs.
    for bad_off, bad_size in ((0, 16), (16, 32), (32, 48), (8, 32), (0, 33)):
        try:
            O.shard_row(slabs, anchor, bad_off, bad_size)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shard_row accepted non-slab ({bad_off},{bad_size})")
    O.shard_row(slabs, anchor, 32, 32)          # one whole slab: must be accepted
    O.shard_row(slabs, anchor, 0, K)            # the degenerate TP=1 case


# ---------------------------------------------------------------------------------------
# G3e -- the numeric consequence of a K split (all-reduce), and why it is NOT bit-exact
# ---------------------------------------------------------------------------------------
def test_g3_row_parallel_allreduce_tolerance(real=None, report=None):
    """A row-parallel linear computes sum_r (x_r @ W_r^T) across ranks.

    The dequantized WEIGHTS are bit-identical to the unsharded ones (proved above), but
    the SUM is not, for two independent and entirely expected reasons:
      1. the all-reduce adds tp partial sums in an order the unsharded kernel never uses;
      2. k_pxq6_mmv's canonical fold depends on kslabs, and a K-shard has fewer slabs, so
         canon_nfix(K/tp/32) may differ from canon_nfix(K/32) -- pxq6.cuh:826-834.
    Reason 2 is the one that will otherwise be mistaken for a bug. It is recorded here
    with the actual nfix values so nobody spends a day on it.
    """
    for label, N, K, slabs, anchor in _cases(real):
        x16 = fixtures.synth_activations(1, K, seed=77, scale="normalized")
        x32 = x16.astype(np.float32)
        full = O.mmv(x32, slabs, anchor)
        w = O.dequant(slabs, anchor)
        exact = x32.astype(np.float64) @ w.astype(np.float64).T
        for tp in TP_SIZES:
            if K % tp or (K // tp) % O.SLAB_COLS:
                continue
            kk = K // tp
            acc = np.zeros_like(full)
            for r in range(tp):
                s, a = O.shard_row(slabs, anchor, r * kk, kk)
                acc = (acc + O.mmv(x32[:, r * kk:(r + 1) * kk], s, a)).astype(np.float32)
            st_split = compare.err_stats(acc, exact)
            st_full = compare.err_stats(full, exact)
            if report is not None:
                report.append((f"allreduce {label} TP={tp}", {
                    "nfix_full": O.canon_nfix(K // 32),
                    "nfix_shard": O.canon_nfix(kk // 32),
                    "split_err": st_split["max_abs_over_absmax"],
                    "full_err": st_full["max_abs_over_absmax"],
                    "bitexact_vs_full": compare.bitwise_equal(acc, full),
                }))
            # Both must be accurate; neither is required to equal the other.
            assert st_split["max_abs_over_absmax"] < 2 ** -19, (
                f"[{label}] TP={tp} all-reduced row-parallel result drifts by "
                f"{st_split['max_abs_over_absmax']:.3e} of absmax -- too much for a "
                f"reassociation; check the K-shard slab range, not the float order")
