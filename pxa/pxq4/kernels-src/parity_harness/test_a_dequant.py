"""
GATE G1 (+G2) -- bit-exact dequant parity.

The plan calls G1 and G3 "the project": between them they retire >90% of the format risk
with no GPU, no lease and no container.  This module is G1 and G2.

What G1 actually proves, in order of how expensive the bug would have been:
  * the 128 B panel header is 64 fp16 anchors indexed by row, not by anything else
  * the 64 B scale SoA is row-indexed, low nibble = elements 0..15, high = 16..31
  * the 16 B code row lives at slab+64+16*row and byte b carries elements (2b, 2b+1)
  * the book and sub tables are the ones the artifact was built with
  * the multiply order is (anchor*sub)*book, in fp32, unfused

Four implementations are compared pairwise:
  C     -- production ggml/src/pxq-cpu.c, compiled here (cref/)          [ground truth]
  O     -- this harness's independent numpy transcription (oracle.py)
  A     -- agent A's src/pxq4_vllm/reference.py                          [skipped if absent]
  CUDA  -- agent C's torch.ops.pxq4.dequant_out                          [needs a GPU]

C vs O is runnable right now and is the load-bearing link: it is the only comparison in
the whole project that touches the shipping engine's own code.
"""

from __future__ import annotations

import numpy as np

from . import adapters, compare, cref_bridge, fixtures
from . import oracle as O


class Skip(Exception):
    """Raised to mark a gate as SKIPPED rather than passed."""


def _cases(real=None):
    return list(fixtures.all_cpu_cases(include_real=real))


# ---------------------------------------------------------------------------------------
# G1a  C  <->  O
# ---------------------------------------------------------------------------------------
def test_g1_c_vs_oracle(real=None):
    if not cref_bridge.available():
        raise Skip("no C compiler: cannot build cref/pxq4_cref from the vendored "
                   "production pxq-cpu.c. This is the ground-truth leg -- fix the "
                   "toolchain rather than trusting the remaining legs.")
    for label, N, K, slabs, anchor in _cases(real):
        blob = O.join_blob(slabs, anchor)
        c = cref_bridge.dequant(blob, N, K)
        o = O.dequant(slabs, anchor)
        assert compare.bitwise_equal(c, o), (
            f"[{label}] N={N} K={K} oracle.dequant != production pxa_deq_row_pxq6\n"
            + compare.bit_diff_report(c, o, "C", "oracle"))


# ---------------------------------------------------------------------------------------
# G1b  O  <->  A   (the reference the converter and every other gate will call)
# ---------------------------------------------------------------------------------------
def test_g1_oracle_vs_agent_a(real=None):
    """Check EVERY importable reference implementation, not just the first.

    Agent A ships one for the converter and agent C ships one as the kernel's numpy twin.
    They are independent transcriptions of the same C, so checking all of them against the
    oracle (which is a third independent transcription, itself pinned to the production C
    by G1a) is strictly more evidence than checking one and calling it done."""
    mods = adapters.all_ref_modules()
    if not mods:
        raise Skip(str(adapters.ref_module()))

    cases = _cases(real)
    for modname, ref in sorted(mods.items()):
        # The tables are part of the contract: each module hard-codes them from
        # ggml-pxq6-tables.h and the converter cross-checks them against the GGUF KVs.
        for name, ours in (("BOOK", O.BOOK), ("SUB", O.SUB)):
            theirs = getattr(ref, name, None)
            assert theirs is not None, f"{modname}.{name} is missing (plan §6.3)"
            theirs = np.asarray(theirs, dtype=np.float32)
            assert compare.bitwise_equal(theirs, ours), (
                f"{modname}.{name} differs from ggml-pxq6-tables.h\n"
                + compare.bit_diff_report(ours, theirs, "tables.h", modname))

        for label, N, K, slabs, anchor in cases:
            a = np.asarray(ref.dequant(slabs, anchor))
            assert a.dtype == np.float32, (
                f"[{modname}/{label}] dequant returned {a.dtype}; plan §6.3 specifies "
                f"float32 (the fp16 cast belongs to the op, not the reference)")
            assert a.shape == (N, K), f"[{modname}/{label}] shape {a.shape} != {(N, K)}"
            o = O.dequant(slabs, anchor)
            assert compare.bitwise_equal(a, o), (
                f"[{modname}/{label}] N={N} K={K} dequant != harness oracle\n"
                + compare.bit_diff_report(o, a, "oracle", modname))


# ---------------------------------------------------------------------------------------
# G2  split <-> join round-trip (converter layout)
# ---------------------------------------------------------------------------------------
def test_g2_split_join_roundtrip(real=None):
    """The emitted-tensor contract (plan §5.3) is a PURE SPLIT: no byte is reordered, no
    value recomputed.  Therefore rejoining must reproduce the original bytes exactly, and
    that is a byte comparison, not a numeric one."""
    for label, N, K, slabs, anchor in _cases(real):
        blob = O.join_blob(slabs, anchor)
        s2, a2 = O.split_blob(blob, N, K)
        assert compare.bitwise_equal(s2, slabs), f"[{label}] slab bytes changed by split/join"
        assert compare.bitwise_equal(a2, anchor), f"[{label}] anchors changed by split/join"
        assert np.array_equal(O.join_blob(s2, a2), blob), f"[{label}] blob round-trip failed"
        # Size is where a wrong panel_bytes shows up first and cheapest.
        assert blob.size == O.tensor_bytes(N, K), (
            f"[{label}] {blob.size} B != geometry {O.tensor_bytes(N, K)} B")
        # Equivalent formulation from ggml_row_size (ggml.c:4903-4906): 2 + 17*K/32 per row.
        assert blob.size == N * (2 + 17 * K // 32)


def test_g2_geometry_gate_rejects_misalignment():
    """assert_geometry must REFUSE what vLLM would silently truncate."""
    for N, K in ((63, 64), (64, 33), (100, 64), (64, 48)):
        try:
            O.assert_geometry(N, K)
        except ValueError:
            continue
        raise AssertionError(f"assert_geometry({N},{K}) should have raised")
    O.assert_geometry(64, 32)
    O.assert_geometry(17408, 5120)


def test_g2_real_shapes_byte_sizes():
    """The six real PXQ4 shapes, against the sizes measured in the artifact
    (06-file-composition.md §5).  If panel_bytes were wrong by one slab this fails."""
    expect = {
        (6144, 5120): 6144 // 64 * (128 + 5120 // 32 * 1088),
        (10240, 5120): 10240 // 64 * (128 + 5120 // 32 * 1088),
        (12288, 5120): 12288 // 64 * (128 + 5120 // 32 * 1088),
        (17408, 5120): 47384576,     # stated per-tensor in the plan §0; 4.2540 bpw
        (5120, 6144): 5120 // 64 * (128 + 6144 // 32 * 1088),
        (5120, 17408): 5120 // 64 * (128 + 17408 // 32 * 1088),
    }
    for (N, K), want in expect.items():
        got = O.tensor_bytes(N, K)
        assert got == want, f"tensor_bytes({N},{K}) = {got}, expected {want}"
        bpw = got * 8.0 / (N * K)
        assert abs(bpw - (4.25 + 16.0 / K)) < 1e-9, f"bpw {bpw} != 4.25+16/{K}"


# ---------------------------------------------------------------------------------------
# G6  CUDA dequant_out  <->  oracle          (GPU; plan gate G6)
# ---------------------------------------------------------------------------------------
def test_g6_cuda_dequant(real=None):
    ops = adapters.pxq4_ops()
    if not ops:
        raise Skip(str(ops))
    if not adapters.cuda_available():
        raise Skip("no CUDA device")
    torch = adapters.torch_module()

    for label, N, K, slabs, anchor in _cases(real):
        dev = torch.device("cuda")
        ts = torch.from_numpy(np.ascontiguousarray(slabs)).to(dev)
        ta = torch.from_numpy(np.ascontiguousarray(anchor)).to(dev)
        out = torch.empty((N, K), dtype=torch.float16, device=dev)
        ops.dequant_out(out, ts, ta)
        torch.cuda.synchronize()
        got = out.cpu().numpy()

        # The op writes fp16 (plan §7.1) while the parity-locked contract is fp32
        # (pxq-cpu.h:16-18).  The kernel computes the fp32 product then stores
        # (dst_t)(e*v) -- k_pxq6_dequant_matrix, pxq6.cuh:716-718 -- and CUDA's
        # float->half is round-to-nearest-even, identical to numpy's astype.  So the
        # correct comparison is fp32-oracle-then-round, and it must be BIT-exact: a
        # tolerance here would hide exactly the layout bugs this gate exists for.
        want = O.dequant(slabs, anchor).astype(np.float16)
        assert compare.bitwise_equal(got, want), (
            f"[{label}] N={N} K={K} torch.ops.pxq4.dequant_out != oracle (fp16-rounded)\n"
            + compare.bit_diff_report(want, got, "oracle_f16", "cuda"))


def test_g6_cuda_dequant_abi(real=None):
    """The op is declared `dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()`
    (plan §7.1): it must write into the caller's buffer and allocate nothing.  A version
    that returns a fresh tensor would work in eager mode and then fail under
    FULL_AND_PIECEWISE capture, which is the worst possible place to discover it."""
    ops = adapters.pxq4_ops()
    if not ops:
        raise Skip(str(ops))
    if not adapters.cuda_available():
        raise Skip("no CUDA device")
    torch = adapters.torch_module()
    N, K = 128, 512
    slabs, anchor = fixtures.synth_parts(N, K, seed=7)
    dev = torch.device("cuda")
    ts = torch.from_numpy(slabs).to(dev)
    ta = torch.from_numpy(anchor).to(dev)
    out = torch.empty((N, K), dtype=torch.float16, device=dev)
    ptr_before = out.data_ptr()
    r = ops.dequant_out(out, ts, ta)
    torch.cuda.synchronize()
    assert r is None, "dequant_out must return () -- it is an out-variant"
    assert out.data_ptr() == ptr_before, "dequant_out replaced the output storage"
    assert torch.isfinite(out).all(), "dequant_out produced non-finite values"
