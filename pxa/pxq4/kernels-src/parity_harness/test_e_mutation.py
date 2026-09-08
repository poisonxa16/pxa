"""
NEGATIVE CONTROLS -- proof that the gates actually catch the bugs they claim to catch.

A parity harness that has never failed is indistinguishable from one that cannot fail.
Every check below deliberately introduces a plausible port bug and asserts that the
corresponding gate REJECTS it.  If a mutation slips through, the gate above it is
decorative and must be fixed before anyone trusts a PASS.

The mutations are not invented: each one is a specific thing a person writing this port
would actually do.
"""

from __future__ import annotations

import numpy as np

from . import compare, fixtures
from . import oracle as O

_N, _K = 192, 512


def _tensor(seed=0xE0):
    return fixtures.synth_parts(_N, _K, seed=seed, profile="extreme")


def _expect_differs(name, mutant, truth):
    assert not compare.bitwise_equal(mutant, truth), (
        f"NEGATIVE CONTROL FAILED: the mutation {name!r} produced bit-identical output "
        f"to the correct implementation. The gate that is supposed to catch it is "
        f"therefore useless. Either the mutation is a no-op (fix the test) or the "
        f"comparison is too weak (fix the gate).")


# ---------------------------------------------------------------------------------------
# Dequant mutations -- these are what G1 exists to catch.
# ---------------------------------------------------------------------------------------
def _dequant_mut(slabs, anchor, mutation):
    """A deliberately-wrong dequant, parameterised by which mistake to make."""
    P, S, _ = slabs.shape
    book, sub = O.BOOK, O.SUB
    anch = anchor.astype(np.float32)
    sbytes = slabs[:, :, :O.CODE_OFF]

    lo_i = (sbytes & 0x0F).astype(np.intp)
    hi_i = (sbytes >> 4).astype(np.intp)
    if mutation == "sub_nibbles_swapped":
        lo_i, hi_i = hi_i, lo_i

    if mutation == "anchor_per_panel":
        # Treats the 128 B header as ONE fp16 broadcast to the panel instead of 64
        # per-row anchors.  A reader that mis-reads row_meta_size would do this.
        anch = np.repeat(anch[:, :1], O.PANEL_ROWS, axis=1)
    if mutation == "scale_soa_is_row_major":
        # Reads the scale byte at slab[16*r] instead of slab[r]: i.e. assumes the scale
        # travels with the code row rather than living in a 64-byte SoA plane.
        idx = (np.arange(O.PANEL_ROWS) * 16) % O.CODE_OFF
        lo_i = (sbytes[:, :, idx] & 0x0F).astype(np.intp)
        hi_i = (sbytes[:, :, idx] >> 4).astype(np.intp)

    if mutation == "reassociated":
        # anchor * (sub * book) instead of (anchor * sub) * book.  Mathematically equal,
        # NOT bit-equal, and it is the single most natural "cleanup" someone would make.
        eff_lo = sub[lo_i]
        eff_hi = sub[hi_i]
    else:
        eff_lo = (anch[:, None, :] * sub[lo_i]).astype(np.float32)
        eff_hi = (anch[:, None, :] * sub[hi_i]).astype(np.float32)

    code_off = O.CODE_OFF
    if mutation == "code_off_by_one":
        code_off = O.CODE_OFF + 1
    codes = slabs[:, :, code_off:code_off + O.PANEL_ROWS * O.CODE_BYTES]
    if codes.shape[-1] < O.PANEL_ROWS * O.CODE_BYTES:
        pad = O.PANEL_ROWS * O.CODE_BYTES - codes.shape[-1]
        codes = np.concatenate([codes, np.zeros(codes.shape[:-1] + (pad,), np.uint8)], -1)
    codes = codes.reshape(P, S, O.PANEL_ROWS, O.CODE_BYTES)

    q = np.empty((P, S, O.PANEL_ROWS, O.SLAB_COLS), dtype=np.float32)
    lo_c = book[(codes & 0x0F).astype(np.intp)]
    hi_c = book[(codes >> 4).astype(np.intp)]
    if mutation == "code_nibbles_swapped":
        lo_c, hi_c = hi_c, lo_c
    q[..., 0::2] = lo_c
    q[..., 1::2] = hi_c

    eff = np.empty((P, S, O.PANEL_ROWS, O.SLAB_COLS), dtype=np.float32)
    eff[..., :16] = eff_lo[..., None]
    eff[..., 16:] = eff_hi[..., None]

    if mutation == "reassociated":
        w = (anch[:, None, :, None] * (eff * q).astype(np.float32)).astype(np.float32)
    else:
        w = (eff * q).astype(np.float32)

    out = w.transpose(0, 2, 1, 3).reshape(P * O.PANEL_ROWS, S * O.SLAB_COLS)
    if mutation == "panel_major_transposed":
        # Emits row = r*P + p instead of row = p*64 + r: the classic "I reshaped the
        # panel axis in the wrong order" bug.  Same values, wrong rows.
        out = w.transpose(2, 0, 1, 3).reshape(P * O.PANEL_ROWS, S * O.SLAB_COLS)
    return np.ascontiguousarray(out)


# NOTE "reassociated" is deliberately NOT in this list: it is provably a no-op for this
# format.  See test_reassociation_is_provably_a_noop below -- that is a CORRECTION to the
# plan, not an omission.
MUTATIONS = ("sub_nibbles_swapped", "code_nibbles_swapped", "code_off_by_one",
             "anchor_per_panel", "scale_soa_is_row_major",
             "panel_major_transposed")


def test_negative_control_dequant_mutations():
    slabs, anchor = _tensor()
    truth = O.dequant(slabs, anchor)
    for m in MUTATIONS:
        _expect_differs(m, _dequant_mut(slabs, anchor, m), truth)


def test_reassociation_is_provably_a_noop():
    """CORRECTION TO THE PLAN (§6.3) -- exhaustively verified here, not argued.

    The plan says: "The multiply order (anchor * sub) * book is load-bearing for
    bit-exactness -- do not reassociate."  For PXQ4 that is false, and this test proves
    it by exhaustion rather than by asserting the opposite claim on faith.

    Reason: all three factors are fp16-snapped, so each has at most 11 significand bits.
    Any pairwise product needs at most 22 bits and is therefore EXACT in fp32 (24 bits).
    Both associations are consequently a single correctly-rounded rounding of the same
    exact triple product, and must agree bit for bit.

    Why keep the test rather than delete the claim: the property depends entirely on the
    tables staying fp16-snapped.  ggml-pxq6-tables.h says they are, and oracle.py asserts
    it at import, but a future table regeneration could break it -- at which point this
    test fires and the plan's warning becomes live again.  The 5.1 M-combination sweep
    below is cheap insurance on a property that would otherwise be silently lost.

    Practical consequence for agent C: the CUDA kernel is free to fold `anchor * sub`
    into a precomputed `eff` (which pxq6_pol_p6::row_effs already does, pxq6.cuh:337-341)
    without any bit-exactness risk.  That is the arrangement the port inherits anyway.
    """
    book, sub = O.BOOK.astype(np.float32), O.SUB.astype(np.float32)
    rng = np.random.default_rng(0)
    anchors = np.concatenate([
        fixtures._ANCHOR_EDGE_EXTREME.astype(np.float32),
        np.float16(np.exp(rng.uniform(np.log(2.0 ** -24), np.log(65504.0), 20000))
                   ).astype(np.float32),
        -np.float16(np.exp(rng.uniform(np.log(2.0 ** -24), np.log(65504.0), 20000))
                    ).astype(np.float32),
    ])
    mism = 0
    for si in range(16):
        for bi in range(16):
            left = ((anchors * sub[si]).astype(np.float32) * book[bi]).astype(np.float32)
            right = (anchors * (sub[si] * book[bi]).astype(np.float32)).astype(np.float32)
            mism += int((compare.bits(left) != compare.bits(right)).sum())
    assert mism == 0, (
        f"{mism} bit mismatches: the tables are no longer fp16-exact, so multiply order "
        f"IS load-bearing again and the plan's §6.3 warning must be reinstated")

    # The pairwise-exactness premise, checked directly rather than inferred.
    for si in range(16):
        for bi in range(16):
            prod = np.float64(sub[si]) * np.float64(book[bi])
            assert np.float32(prod) == prod, (si, bi, "sub*book is not exact in fp32")


# ---------------------------------------------------------------------------------------
# Shard mutations -- these are what G3 exists to catch, and the reason the project has
# a dedicated sharding gate at all.
# ---------------------------------------------------------------------------------------
def _naive_row_contiguous_shard(blob, N, K, k_off, k_size):
    """The bug: treat each weight row as `2 + 17*K/32` contiguous bytes and slice the K
    range out of it.  That IS ggml's row_size, so the arithmetic looks right, and this is
    precisely what vLLM's generic GGUF sharder does.  For PXQ4 it is nonsense: a row's
    bytes are spread one 16-byte code row per slab across the entire 64-row panel."""
    row_size = 2 + 17 * K // 32
    a = np.frombuffer(blob, dtype=np.uint8).reshape(N, row_size)  # <- the false premise
    b0 = 2 + 17 * k_off // 32
    b1 = 2 + 17 * (k_off + k_size) // 32
    return np.ascontiguousarray(np.concatenate([a[:, :2], a[:, b0:b1]], axis=1)).reshape(-1)


def test_negative_control_naive_row_shard_is_wrong():
    """The headline failure mode: it produces a blob of exactly the RIGHT SIZE."""
    slabs, anchor = _tensor()
    blob = O.join_blob(slabs, anchor)
    k_off, k_size = _K // 2, _K // 2
    naive = _naive_row_contiguous_shard(blob, _N, _K, k_off, k_size)
    correct = O.join_blob(*O.shard_row(slabs, anchor, k_off, k_size))
    assert naive.size == correct.size == O.tensor_bytes(_N, k_size), (
        "the naive shard should be the same SIZE as the correct one -- that is what makes "
        "it dangerous; if the sizes differ the test is not exercising the real bug")
    assert not np.array_equal(naive, correct), (
        "NEGATIVE CONTROL FAILED: a row-contiguous K shard produced the same bytes as a "
        "slab-aligned one. Either K is degenerate here or the format is not what we think.")
    # And it still parses and dequantizes cleanly -- silently wrong, never raising.
    # Reinterpreting slab bytes as the fp16 anchor header can produce inf/NaN anchors,
    # so numpy emits invalid-value warnings here.  That is a PROPERTY of the corrupt
    # fixture, not a defect, and silencing it locally keeps a real warning elsewhere
    # visible instead of teaching the operator to ignore this one.
    s2, a2 = O.split_blob(naive, _N, k_size)
    with np.errstate(invalid="ignore", over="ignore"):
        d_bad = O.dequant(s2, a2)
    d_good = O.dequant(*O.shard_row(slabs, anchor, k_off, k_size))
    assert d_bad.shape == d_good.shape
    _expect_differs("naive row-contiguous K shard", d_bad, d_good)


def test_negative_control_anchor_sharded_on_k():
    """A very likely mistake: giving PXQ4AnchorParameter an `input_dim` so vLLM narrows
    it on a K split.  The anchor has no K axis -- it is per (panel,row) -- so narrowing
    it either changes shape or silently drops rows."""
    slabs, anchor = _tensor()
    correct_s, correct_a = O.shard_row(slabs, anchor, 0, _K // 2)
    # what a wrongly-declared anchor param would receive: half the columns of [P,64]
    wrong_a = np.ascontiguousarray(anchor[:, :O.PANEL_ROWS // 2])
    try:
        O.dequant(correct_s, wrong_a)
    except AssertionError:
        return                      # shape mismatch caught -- the desired outcome
    raise AssertionError(
        "NEGATIVE CONTROL FAILED: dequant accepted a K-narrowed anchor. The shape "
        "assertion in dequant() is the thing that would catch a wrongly-declared "
        "PXQ4AnchorParameter, and it did not fire.")


def test_negative_control_offset_by_one_panel():
    """An off-by-one in the panel offset -- the most likely arithmetic slip in a
    converter -- must not survive the column-shard gate."""
    slabs, anchor = _tensor()
    full = O.dequant(slabs, anchor)
    # _N = 192 rows = 3 panels.  Take the LAST panel, then take the one before it.
    rows = O.PANEL_ROWS
    off = _N - rows
    s, a = O.shard_column(slabs, anchor, off, rows)
    good = O.dequant(s, a)
    assert compare.bitwise_equal(good, full[off:off + rows])
    p = off // O.PANEL_ROWS
    bad = O.dequant(np.ascontiguousarray(slabs[p - 1:p]),
                    np.ascontiguousarray(anchor[p - 1:p]))
    _expect_differs("column shard off by one panel", bad, good)


def test_negative_control_mmv_fold_matters():
    """The mmv model claims its ACCUMULATION ORDER is load-bearing.  Show that: a
    left-to-right sum over kb -- the obvious implementation -- differs from the kernel's
    KSEG/nfix fold.  If it did not, modelling the fold would be wasted effort and the
    CUDA comparison could use a plain dot product."""
    slabs, anchor = fixtures.synth_parts(_N, 2048, seed=5, profile="realistic")
    x = fixtures.synth_activations(1, 2048, seed=5, scale="normalized").astype(np.float32)
    kernel_order = O.mmv(x, slabs, anchor)
    w = O.dequant(slabs, anchor)
    # naive: fp32 left-to-right
    naive = np.zeros((1, w.shape[0]), dtype=np.float32)
    acc = np.zeros(w.shape[0], dtype=np.float32)
    for k in range(w.shape[1]):
        acc = (acc + w[:, k] * np.float32(x[0, k])).astype(np.float32)
    naive[0] = acc
    _expect_differs("naive left-to-right fp32 dot", naive, kernel_order)
    # ...but both are close to exact, which is the point: only a bit comparison sees it.
    exact = x.astype(np.float64) @ w.astype(np.float64).T
    for name, v in (("kernel", kernel_order), ("naive", naive)):
        st = compare.err_stats(v, exact)
        assert st["max_abs_over_absmax"] < 2 ** -18, (name, st)
