"""test_pxq4_kernel_ref.py — GPU-free gates for the PXQ4 CUDA extension.

Two things are under test and they check each other:
  * pxq4_kernel_ref.py, the numpy reference (the oracle other components validate against);
  * pxq4_kernel.cuh, the actual vendored device source, executed on the CPU through
    pxq4_kernel_hostsim.cpp (see that file for how, and why it is trustworthy).

Nothing here needs a GPU, a lease, a container, or the checkpoint. Run:
    bash build_hostsim.sh && python3 -m pytest -q test_pxq4_kernel_ref.py
or plain `python3 test_pxq4_kernel_ref.py` for a pytest-free run.

WHAT THIS DOES NOT COVER, stated plainly:
  * that the frozen tables in pxq4_kernel_tables.h match the tables recorded in the real
    artifact's gguf KVs pxa.pxq6.book / pxa.pxq6.sub. That comparison needs the file and is
    plan §5.6 check 3, owned by the converter.
  * that nvcc emits for sm_70 what g++ emits here. Compilers are free to contract a*b+c into
    an FMA; the host build is compiled without -ffast-math and the CUDA build must be compiled
    without -use_fast_math for the same reason, but only gate G6 (on the device) proves it.
"""

from __future__ import annotations

import ctypes


class _HostsimUnavailable(RuntimeError):
    """Raised when the CPU simulator was never compiled - a skip, not a failure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pxq4_kernel_ref as ref  # noqa: E402

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpxq4_hostsim.so")


def _load_hostsim():
    if not os.path.exists(_LIB_PATH):
        return None
    if not os.path.exists(_LIB_PATH):
        # The host simulator is a BUILD ARTIFACT, not a checked-in file. Its absence
        # means "no compiler here" (Unraid has no g++), not "the kernel is wrong".
        # Reporting 8 hard failures for a missing toolchain is a misleading signal -
        # the same class of defect this suite exists to catch. Skip loudly instead.
        raise _HostsimUnavailable(
            f"{os.path.basename(_LIB_PATH)} not built.\n"
            f"    Build it with a C++ compiler:\n"
            f"      g++ -O2 -std=c++17 -shared -fPIC -Ihostsim -I. -pthread \\\n"
            f"          pxq4_kernel_hostsim.cpp -o libpxq4_hostsim.so\n"
            f"    (this box has no g++; build on a dev host or in a CUDA container)")
    lib = ctypes.CDLL(_LIB_PATH)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    f32 = ctypes.POINTER(ctypes.c_float)
    lib.pxq4_hostsim_dequant_f32.argtypes = [u8, u16, f32, ctypes.c_int, ctypes.c_int]
    lib.pxq4_hostsim_dequant_f16.argtypes = [u8, u16, u16, ctypes.c_int, ctypes.c_int]
    lib.pxq4_hostsim_mmv_f16.argtypes = [u8, u16, u16, u16, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int]
    lib.pxq4_hostsim_canon_nfix.argtypes = [ctypes.c_int]
    lib.pxq4_hostsim_canon_nfix.restype = ctypes.c_int
    lib.pxq4_hostsim_canon_max_chunk.argtypes = [ctypes.c_int]
    lib.pxq4_hostsim_canon_max_chunk.restype = ctypes.c_int
    lib.pxq4_hostsim_builtin_tables.argtypes = [f32, f32]
    return lib


LIB = _load_hostsim()


def _p(a, ct):
    return a.ctypes.data_as(ctypes.POINTER(ct))


def sim_dequant_f32(slabs, anchor):
    P, S, _ = slabs.shape
    out = np.zeros((P * 64, S * 32), dtype=np.float32)
    LIB.pxq4_hostsim_dequant_f32(_p(np.ascontiguousarray(slabs), ctypes.c_uint8),
                                 _p(np.ascontiguousarray(anchor).view(np.uint16), ctypes.c_uint16),
                                 _p(out, ctypes.c_float), P, S)
    return out


def sim_dequant_f16(slabs, anchor):
    P, S, _ = slabs.shape
    out = np.zeros((P * 64, S * 32), dtype=np.float16)
    LIB.pxq4_hostsim_dequant_f16(_p(np.ascontiguousarray(slabs), ctypes.c_uint8),
                                 _p(np.ascontiguousarray(anchor).view(np.uint16), ctypes.c_uint16),
                                 _p(out.view(np.uint16), ctypes.c_uint16), P, S)
    return out


def sim_mmv_f16(slabs, anchor, x, vecx=True):
    P, S, _ = slabs.shape
    M = x.shape[0]
    out = np.zeros((M, P * 64), dtype=np.float16)
    LIB.pxq4_hostsim_mmv_f16(_p(np.ascontiguousarray(slabs), ctypes.c_uint8),
                             _p(np.ascontiguousarray(anchor).view(np.uint16), ctypes.c_uint16),
                             _p(np.ascontiguousarray(x).view(np.uint16), ctypes.c_uint16),
                             _p(out.view(np.uint16), ctypes.c_uint16),
                             M, P, S, 1 if vecx else 0)
    return out


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------
def make_random_weight(N, K, seed=0):
    """Random PXQ4 bit patterns, including the pathological ones: a zero row (anchor +0, which
    the format represents exactly) and a row whose anchor is subnormal in fp16."""
    rng = np.random.default_rng(seed)
    P, S = N // 64, K // 32
    anchor = (rng.random((P, 64), dtype=np.float32) * 0.5 + 0.01).astype(np.float16)
    anchor[0, 0] = np.float16(0.0)
    if P * 64 > 1:
        anchor[0, 1] = np.float16(6e-8)          # fp16 subnormal
    sub_idx = rng.integers(0, 16, size=(P, S, 64, 2), dtype=np.uint8)
    codes = rng.integers(0, 16, size=(P, 64, K), dtype=np.uint8)
    return ref.pack_pxq4(anchor, sub_idx, codes)


SHAPES = [(64, 32), (64, 256), (128, 128), (192, 512), (128, 1024)]


# ---------------------------------------------------------------------------------------------
# T1-T2: tables
# ---------------------------------------------------------------------------------------------
def test_table_invariants():
    ref.check_tables()


def test_tables_match_c_literals():
    assert LIB is not None, "build libpxq4_hostsim.so first (bash build_hostsim.sh)"
    book = np.zeros(16, dtype=np.float32)
    sub = np.zeros(16, dtype=np.float32)
    LIB.pxq4_hostsim_builtin_tables(_p(book, ctypes.c_float), _p(sub, ctypes.c_float))
    # bit-exact, not approximate: a decimal-rounded transcription would fail here
    assert np.array_equal(book, ref.BOOK), (book, ref.BOOK)
    assert np.array_equal(sub, ref.SUB), (sub, ref.SUB)


# ---------------------------------------------------------------------------------------------
# T3: layout round trip (gate G2 in miniature)
# ---------------------------------------------------------------------------------------------
def test_split_join_round_trip():
    for N, K in SHAPES:
        slabs, anchor = make_random_weight(N, K, seed=N + K)
        blob = ref.join_blob(slabs, anchor)
        assert len(blob) == ref.tensor_bytes(N, K)
        s2, a2 = ref.split_blob(blob, N, K)
        assert np.array_equal(s2, slabs)
        assert np.array_equal(a2.view(np.uint16), anchor.view(np.uint16))
        assert ref.join_blob(s2, a2) == blob


def test_tensor_bytes_matches_ggml_row_size():
    # ggml stores 2 B of row meta + 17 B per 32 elements per row (ggml.c:4903-4906), i.e.
    # 4.25 + 16/K bits per weight. The panel formula must agree exactly.
    for N, K in [(64, 32), (17408, 5120), (5120, 17408), (12288, 5120), (10240, 5120), (6144, 5120)]:
        assert ref.tensor_bytes(N, K) == N * (2 + 17 * K // 32)


# ---------------------------------------------------------------------------------------------
# T4-T5: numpy reference vs the real kernel source
# ---------------------------------------------------------------------------------------------
def test_reference_vs_naive_transcription():
    for N, K in [(64, 32), (128, 128), (64, 256)]:
        slabs, anchor = make_random_weight(N, K, seed=1234 + N)
        a = ref.dequant(slabs, anchor)
        b = ref.dequant_naive(slabs, anchor)
        assert np.array_equal(a, b), f"vectorised and scalar references disagree at [{N},{K}]"


def test_kernel_dequant_bit_exact_fp32():
    assert LIB is not None, "build libpxq4_hostsim.so first (bash build_hostsim.sh)"
    for N, K in SHAPES:
        slabs, anchor = make_random_weight(N, K, seed=N * 7 + K)
        want = ref.dequant(slabs, anchor)
        got = sim_dequant_f32(slabs, anchor)
        assert np.array_equal(got, want), f"k_pxq4_dequant_matrix != reference at [{N},{K}]"


def test_kernel_dequant_fp16_is_reference_rounded():
    """The shipped op writes fp16. That is the fp32 parity contract with ONE extra
    round-to-nearest-even, so gate G6 must compare against reference.dequant().astype(f16) and
    NOT against the fp32 array. This test pins that relationship so nobody has to rediscover
    it on the GPU."""
    assert LIB is not None
    for N, K in SHAPES:
        slabs, anchor = make_random_weight(N, K, seed=N + 3 * K)
        want = ref.dequant(slabs, anchor).astype(np.float16)
        got = sim_dequant_f16(slabs, anchor)
        assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), f"[{N},{K}]"


# ---------------------------------------------------------------------------------------------
# T6: sharding is a permutation (gate G3), on both axes, at TP = 2 and 4
# ---------------------------------------------------------------------------------------------
def test_column_shard_is_whole_panels():
    """Column-parallel = narrow(dim 0) in whole panels. Dequantising a panel subrange must give
    exactly the corresponding row band of the full dequant — no re-quantisation, no drift."""
    N, K = 256, 256
    slabs, anchor = make_random_weight(N, K, seed=99)
    full = ref.dequant(slabs, anchor)
    for tp in (2, 4):
        P = N // 64
        assert P % tp == 0
        per = P // tp
        for r in range(tp):
            sub_s = slabs[r * per:(r + 1) * per]
            sub_a = anchor[r * per:(r + 1) * per]
            got = ref.dequant(sub_s, sub_a)
            want = full[r * per * 64:(r + 1) * per * 64, :]
            assert np.array_equal(got, want), f"TP={tp} rank={r}"


def test_row_shard_is_whole_slabs_with_duplicated_header():
    """Row-parallel = narrow(dim 1) in whole slabs, with the 128 B anchor header duplicated to
    every rank. The anchor has no cross-K coupling, so the K split is bit-identical — this is
    the claim the whole TP story rests on, and it is cheap to check."""
    N, K = 128, 512
    slabs, anchor = make_random_weight(N, K, seed=1001)
    full = ref.dequant(slabs, anchor)
    for tp in (2, 4):
        S = K // 32
        assert S % tp == 0
        per = S // tp
        for r in range(tp):
            got = ref.dequant(slabs[:, r * per:(r + 1) * per, :], anchor)   # anchor copied whole
            want = full[:, r * per * 32:(r + 1) * per * 32]
            assert np.array_equal(got, want), f"TP={tp} rank={r}"


def test_real_model_shapes_shard_cleanly():
    """Every PXQ4-served module of Qwen3.8-27B, at TP in {1,2,4}. A shard that violates
    %64 / %32 does not raise in vLLM — `round(shard_size // packed_factor)` truncates silently
    (parameter.py:605-610) and the model loads cleanly with wrong logits. This is the cheapest
    place to catch it."""
    col = {                       # module -> (N per full tensor, K)
        "mlp.gate_proj": (17408, 5120),
        "mlp.up_proj": (17408, 5120),
        "linear_attn.in_proj_qkv": (10240, 5120),
        "linear_attn.in_proj_z": (6144, 5120),
        "self_attn.q_proj": (12288, 5120),
        "self_attn.k_proj": (1024, 5120),
        "self_attn.v_proj": (1024, 5120),
    }
    row = {                       # split on K instead
        "mlp.down_proj": (5120, 17408),
        "self_attn.o_proj": (5120, 6144),
        "linear_attn.out_proj": (5120, 6144),
    }
    for tp in (1, 2, 4):
        for name, (N, K) in col.items():
            ref.assert_geometry(N, K)
            assert (N // tp) % 64 == 0, f"{name} column shard at TP={tp}: {N // tp} rows"
            assert K % 32 == 0
        for name, (N, K) in row.items():
            ref.assert_geometry(N, K)
            assert N % 64 == 0
            assert (K // tp) % 32 == 0, f"{name} row shard at TP={tp}: K={K // tp}"
    # the GDN in_proj_qkvz fusion the fork builds: output_sizes [2048, 2048, 6144, 6144]
    # (qwen3_5.py:212-230). Every shard must land on a panel boundary at TP=4 or the packed
    # loader truncates.
    for size in (2048, 2048, 6144, 6144):
        for tp in (1, 2, 4):
            assert (size // tp) % 64 == 0, f"in_proj_qkvz shard {size} at TP={tp}"
    # and the b/a rows that MUST be split out (48 each): they are NOT panel-aligned, which is
    # exactly why plan §2.4 requires the config's `ignore` to name in_proj_a / in_proj_b.
    assert 48 % 64 != 0 and (48 // 4) % 64 != 0


# ---------------------------------------------------------------------------------------------
# T7-T9: the mmv
# ---------------------------------------------------------------------------------------------
def test_canon_nfix_matches_c():
    assert LIB is not None
    for kslabs in list(range(1, 130)) + [136, 160, 192, 544]:
        assert ref.canon_nfix(kslabs) == LIB.pxq4_hostsim_canon_nfix(kslabs), kslabs


def test_mmv_smem_stays_small():
    """The reason the port does not need the 96 KiB Volta shared-memory opt-in: chunked
    staging caps the mmv's dynamic shared memory at ceil(kslabs/nfix)*128 B. Checked here on
    every real K this model uses, including the K=17408 ffn_down that broke the engine's own
    whole-K staging (ggml-cuda.cu:4249-4258)."""
    assert LIB is not None
    for K in (1024, 4352, 5120, 6144, 17408):
        kslabs = K // 32
        chunk = LIB.pxq4_hostsim_canon_max_chunk(kslabs)
        nfix = ref.canon_nfix(kslabs)
        assert chunk == -(-kslabs // nfix), (K, chunk, nfix)
        smem = chunk * 32 * 4
        assert smem <= 8 * 1024, f"K={K} needs {smem} B"


def test_mmv_matches_fold_reference():
    """k_pxq4_mmv against an independent float32 replication of its fold order. Bit-exact in
    fp16 output. Small shapes only: mmv_fold is a pure-python triple loop."""
    assert LIB is not None
    for N, K, M in [(64, 128, 1), (128, 256, 2), (64, 512, 3)]:
        slabs, anchor = make_random_weight(N, K, seed=N + K + M)
        rng = np.random.default_rng(7)
        x = (rng.standard_normal((M, K)) * 0.3).astype(np.float16)
        want = ref.mmv_fold(slabs, anchor, x).astype(np.float16)
        got = sim_mmv_f16(slabs, anchor, x)
        assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), f"[{N},{K}] M={M}"


def test_mmv_vecx_equals_scalar():
    """VECX only widens the activation load; the multiply/accumulate order is identical, so the
    two instantiations must agree bit-for-bit. A disagreement means the float4 path is reading
    the wrong lane, which is otherwise very hard to see."""
    assert LIB is not None
    slabs, anchor = make_random_weight(128, 256, seed=42)
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((2, 256)) * 0.5).astype(np.float16)
    a = sim_mmv_f16(slabs, anchor, x, vecx=True)
    b = sim_mmv_f16(slabs, anchor, x, vecx=False)
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


def test_mmv_close_to_dequant_gemm():
    """Sanity, not exactness: the mmv folds in a different order from dequant + GEMM, so they
    are close but never bit-identical. This is the tolerance relationship gate G8 checks on the
    device; pinning it here stops someone asserting equality there."""
    assert LIB is not None
    N, K, M = 128, 512, 2
    slabs, anchor = make_random_weight(N, K, seed=5)
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((M, K)) * 0.4).astype(np.float16)
    w = ref.dequant(slabs, anchor)
    want = x.astype(np.float32) @ w.T
    got = sim_mmv_f16(slabs, anchor, x).astype(np.float32)
    denom = np.maximum(np.abs(want).max(), 1e-6)
    assert np.abs(got - want).max() / denom < 5e-3


def test_zero_row_is_exactly_zero():
    """anchor == fp16(+0) must dequantise to exact zeros regardless of codes or sub indices —
    the format's representation of a dead row, and a case that a sign or subnormal bug in the
    anchor path would break."""
    slabs, anchor = make_random_weight(64, 128, seed=8)
    anchor = anchor.copy()
    anchor[0, 5] = np.float16(0.0)
    w = ref.dequant(slabs, anchor)
    assert np.all(w[5] == 0.0)
    if LIB is not None:
        assert np.all(sim_dequant_f32(slabs, anchor)[5] == 0.0)


# ---------------------------------------------------------------------------------------------
# T10: real artifact bytes
# ---------------------------------------------------------------------------------------------
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixture_blk0_attn_gate_panel0.txt")


def test_real_artifact_panel():
    """Panel 0 of blk.0.attn_gate.weight, lifted verbatim out of
    /mnt/models/pxa-models/Qwen3.8-27B-PXQ4.gguf (ggml type 252, K=5120, R=6144) by
    dump_one_panel.py. A panel is a self-contained byte range, so 174208 bytes is a complete,
    legal PXQ4 tensor of 64 rows -- which is itself the column-parallel sharding claim.

    The load-bearing assertion is the last one. book[15] == 1.0 and sub[15] == 0.98779296875,
    and the quantizer anchors each row on its fp16 absmax, so a correctly interpreted row's
    largest magnitude must be anchor * sub[s] * 1.0 with s usually 15, i.e. the ratio
    rowmax/anchor must sit just under 0.9878 and NEVER exceed it. A transposed nibble order, a
    byte-swapped anchor, or a wrong sub table all break that relation immediately -- on
    production data, not on synthetic bytes we generated ourselves."""
    if not os.path.exists(_FIXTURE):
        return
    import base64
    meta, b64 = open(_FIXTURE).read().split("\n")[:2]
    assert meta.split()[3] == "252", meta
    blob = base64.b64decode(b64.split(" ", 1)[1])
    K = 5120
    assert len(blob) == ref.panel_bytes(K) == 174208

    slabs, anchor = ref.split_blob(blob, 64, K)
    assert ref.join_blob(slabs, anchor) == blob

    w = ref.dequant(slabs, anchor)
    assert w.shape == (64, K) and np.isfinite(w).all()

    a32 = anchor.astype(np.float32).reshape(-1)
    rowmax = np.abs(w).max(axis=1)
    assert np.all(a32 > 0)
    ratio = rowmax / a32
    assert np.all(ratio <= np.float32(ref.SUB[15] * ref.BOOK[15]) + 1e-7), ratio.max()
    assert np.median(ratio) > 0.9, np.median(ratio)

    if LIB is not None:
        assert np.array_equal(sim_dequant_f32(slabs, anchor), w)
        assert np.array_equal(sim_dequant_f16(slabs, anchor).view(np.uint16),
                              w.astype(np.float16).view(np.uint16))


def _main():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:                        # noqa: BLE001
            bad += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
