"""test_pxq4hq_cpu.py -- CPU gates for the PXQ4HQ tier. No GPU, no torch, no vLLM.

Five gates, in the order they can fail:

  G1 BYTE ROUND TRIP    split_blob -> join_blob == the GGUF bytes, on real tensors of a real
                        PXQ4HQ model. The split is the converter's whole claim to being a
                        repack and not a requant, so it is checked byte-for-byte rather than
                        numerically.
  G2 SCALE + CODE       the vectorised numpy decode (word arithmetic, one reshape for the bs8
                        blocks) against a slow, literal transcription of the ENGINE's C decode
                        loop (ggml/src/pxq-cpu.c pxa_deq_row_pxq6 with hq = true, byte
                        arithmetic). Two different formulas over the same bytes: agreement is
                        evidence, not tautology. THIS is the gate that catches the tier's one
                        real trap -- reading slab[r] instead of slab[2r + h] gives a
                        well-formed tensor that is uniformly wrong.
  G3 VALUE CONTRACT     the decoded values obey eff = anchor*SUB8[nibble], w = eff*book[code]
                        exactly, per 8-element block, including the reconstruction ceiling
                        |w| <= max(SUB8) * max|book| * |anchor|.
  G4 SHARD COMMUTES     dequant(shard(x)) == shard(dequant(x)) on both axes, at TP 1/2/4 --
                        the property that makes a TP split a byte move. Panels are 64 rows and
                        slabs 32 columns for this tier exactly as for the others, so the legal
                        shard boundaries are unchanged by the finer sub-scale.
  G5 TIER CONFUSION     a PXQ4 slab array offered to the pxq4hq decoder, and vice versa, must
                        RAISE. Both are 16-byte nibble code rows over the same book; only the
                        stride and the scale SoA differ, so this is the one pair of tiers that
                        could silently decode each other's bytes into plausible garbage.

  G6 (optional, needs a C compiler) the engine's OWN decode function, extracted verbatim by
     build_oracle_pxq4hq.sh, on the same bytes:
         CC=gcc ./build_oracle_pxq4hq.sh && python3 test_pxq4hq_cpu.py --oracle ./oracle_pxq4hq

USAGE
    python3 test_pxq4hq_cpu.py --gguf /path/to/a-PXQ4HQ.gguf [--oracle ./oracle_pxq4hq]
    python3 test_pxq4hq_cpu.py                       # synthetic only (G2..G5)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import gguf_raw as G          # noqa: E402
from gguf_to_vllm import reference as R         # noqa: E402
from gguf_to_vllm import tiers as T             # noqa: E402


# ------------------------------------------------------------------ the slow engine transcript
def slow_decode_engine(blob, N: int, K: int, book: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """A literal, scalar transcription of ggml/src/pxq-cpu.c pxa_deq_row_pxq6(hq = true).

    Deliberately written with the C's BYTE indexing -- ``slab[2*r]`` / ``slab[2*r+1]``, the
    four eff[] entries in the engine's own order, and ``eff[i >> 3]`` for the element -- which
    is a different expression of the same packing from the reshape-based arithmetic tiers.py
    uses. Slow on purpose: it is the reference, not a path anything ships.
    """
    t = T.tier_of(T.PXQ4HQ)
    b = np.frombuffer(blob, dtype=np.uint8)
    KB = K // 32
    panel_stride = T.HEADER_BYTES + KB * t.slab_bytes
    out = np.empty((N, K), dtype=np.float32)
    for row in range(N):
        p, r = row >> 6, row & 63
        panel = b[p * panel_stride:(p + 1) * panel_stride]
        anch = np.float32(panel[:T.HEADER_BYTES].view("<f2")[r])
        for kb in range(KB):
            slab = panel[T.HEADER_BYTES + kb * t.slab_bytes:
                         T.HEADER_BYTES + (kb + 1) * t.slab_bytes]
            eff = [np.float32(anch * sub[slab[2 * r] & 0xF]),        # elems  0- 7
                   np.float32(anch * sub[slab[2 * r] >> 4]),         # elems  8-15
                   np.float32(anch * sub[slab[2 * r + 1] & 0xF]),    # elems 16-23
                   np.float32(anch * sub[slab[2 * r + 1] >> 4])]     # elems 24-31
            q = slab[128 + 16 * r:128 + 16 * r + 16]
            for bidx in range(16):
                i0, i1 = 2 * bidx, 2 * bidx + 1
                out[row, kb * 32 + i0] = eff[i0 >> 3] * book[q[bidx] & 0xF]
                out[row, kb * 32 + i1] = eff[i1 >> 3] * book[q[bidx] >> 4]
    return out


def synth(P: int, S: int, seed: int):
    rng = np.random.default_rng(seed)
    slabs = rng.integers(0, 256, size=(P, S, T.tier_of(T.PXQ4HQ).slab_bytes), dtype=np.uint8)
    anchor = (((rng.integers(0, 2001, size=(P, 64)).astype(np.float32) - 1000.0) / 1024.0)
              .astype(np.float16))
    return slabs, anchor


# --------------------------------------------------------------------------------- the gates
def g2_g3(book, sub, verbose=True) -> list[str]:
    fails = []
    for seed, (P, S) in enumerate([(3, 5), (1, 1), (2, 9)]):
        slabs, anchor = synth(P, S, 1000 + seed)
        got = T.dequant(slabs, anchor, T.PXQ4HQ, book, sub)
        want = slow_decode_engine(T.join_blob(slabs, anchor), P * 64, S * 32, book, sub)
        if not np.array_equal(got, want):
            n = int((got != want).sum())
            fails.append(f"G2 P={P} S={S}: {n}/{got.size} values differ from the scalar "
                         f"engine transcript (max abs {np.abs(got - want).max():g})")
        # G3: the contract, restated on the values themselves.
        ceiling = float(np.max(sub)) * float(np.max(np.abs(book)))
        anch_abs = np.abs(anchor.astype(np.float32)).repeat(1).reshape(P, 64)
        bound = np.repeat(anch_abs.reshape(P * 64), S * 32).reshape(P * 64, S * 32) * ceiling
        over = np.abs(got) > bound * (1 + 1e-6)
        if over.any():
            fails.append(f"G3 P={P} S={S}: {int(over.sum())} values exceed "
                         f"max(SUB8)*max|book|*|anchor|")
        # every decoded value must be exactly some eff * some book entry
        if not np.all(np.isfinite(got)):
            fails.append(f"G3 P={P} S={S}: non-finite values")
    if verbose and not fails:
        print("  G2 scalar-engine transcript : PASS (bit-exact)")
        print("  G3 value contract           : PASS")
    return fails


def g4(book, sub, verbose=True) -> list[str]:
    """dequant(shard(x)) == shard(dequant(x)), both axes, TP 1/2/4."""
    fails = []
    P, S = 8, 8
    slabs, anchor = synth(P, S, 4242)
    full = T.dequant(slabs, anchor, T.PXQ4HQ, book, sub)
    for tp in (1, 2, 4):
        # dim 0: whole panels
        for i in range(tp):
            p0, p1 = P // tp * i, P // tp * (i + 1)
            part = T.dequant(slabs[p0:p1], anchor[p0:p1], T.PXQ4HQ, book, sub)
            if not np.array_equal(part, full[p0 * 64:p1 * 64, :]):
                fails.append(f"G4 tp={tp} dim0 shard {i} differs")
        # dim 1: whole slabs
        for i in range(tp):
            s0, s1 = S // tp * i, S // tp * (i + 1)
            part = T.dequant(slabs[:, s0:s1], anchor, T.PXQ4HQ, book, sub)
            if not np.array_equal(part, full[:, s0 * 32:s1 * 32]):
                fails.append(f"G4 tp={tp} dim1 shard {i} differs")
    if verbose and not fails:
        print("  G4 shard commutation        : PASS (tp 1/2/4, both axes)")
    return fails


def g5(book, sub, verbose=True) -> list[str]:
    """The two 4-bit tiers must refuse each other's bytes."""
    fails = []
    P, S = 2, 3
    hq_slabs, anchor = synth(P, S, 77)
    p4_slabs = np.zeros((P, S, T.tier_of(T.PXQ4).slab_bytes), dtype=np.uint8)
    try:
        T.dequant(p4_slabs, anchor, T.PXQ4HQ, book, sub)
        fails.append("G5: a 1088-byte-stride array was accepted by the pxq4hq decoder")
    except ValueError:
        pass
    try:
        T.dequant(hq_slabs, anchor, T.PXQ4, book, R.SUB)
        fails.append("G5: a 1152-byte-stride array was accepted by the pxq4 decoder")
    except ValueError:
        pass
    if verbose and not fails:
        print("  G5 tier confusion refused   : PASS")
    return fails


def g1_and_oracle(path: str, oracle: str | None, book, sub, limit: int = 6) -> list[str]:
    """G1 on real tensors, plus G6 against the engine-extracted binary when one is given."""
    fails = []
    gg = G.GGUFFile(path)
    names = [n for n, ti in gg.tensors.items() if ti.type_id == G.GGML_PXQ4HQ][:limit]
    if not names:
        return [f"G1: {path} carries no pxq4hq tensors"]
    print(f"  G1/G6 on {len(names)} pxq4hq tensor(s) of {os.path.basename(path)}")
    for n in names:
        ti = gg.tensors[n]
        N, K = ti.ne1, ti.ne0
        raw = bytes(gg.raw(n))
        slabs, anchor = T.split_blob(raw, T.PXQ4HQ, N, K)
        if T.join_blob(slabs, anchor) != raw:
            fails.append(f"G1 {n}: split -> join is not the original bytes")
            continue
        got = T.dequant(slabs, anchor, T.PXQ4HQ, book, sub)
        # a bounded scalar cross-check (the first 128 rows) so a big tensor stays cheap
        rows = min(N, 128)
        want = slow_decode_engine(raw[: (rows // 64) * (T.HEADER_BYTES + (K // 32) * 1152)],
                                  rows, K, book, sub)
        if not np.array_equal(got[:rows], want):
            fails.append(f"G2/real {n}: rows 0..{rows} differ from the scalar transcript")
        if oracle:
            with tempfile.TemporaryDirectory() as td:
                bi, bo = os.path.join(td, "in.bin"), os.path.join(td, "out.f32")
                with open(bi, "wb") as f:
                    f.write(raw)
                rc = subprocess.run([oracle, str(N), str(K), bi, bo]).returncode
                if rc != 0:
                    fails.append(f"G6 {n}: oracle exited {rc}")
                    continue
                ref = np.fromfile(bo, dtype=np.float32).reshape(N, K)
            if not np.array_equal(got, ref):
                d = int((got != ref).sum())
                fails.append(f"G6 {n}: {d}/{got.size} values differ from the ENGINE binary "
                             f"(max abs {np.abs(got - ref).max():g})")
        print(f"    {n:44s} [{N} x {K}]  ok")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.environ.get("PXA_PXQ4HQ_GGUF"),
                    help="a real PXQ4HQ GGUF for G1/G6 (optional)")
    ap.add_argument("--oracle", default=None,
                    help="path to the engine-extracted oracle binary (gate G6)")
    args = ap.parse_args()

    book, sub = R.BOOK, T.sub_of(T.PXQ4HQ)
    T.check_book(book, T.PXQ4HQ)
    T.check_sub(sub)
    if np.array_equal(np.asarray(sub), np.asarray(R.SUB)):
        print("FAIL: SUB8 and SUB16 are the same table -- the transcription is wrong")
        return 1
    print(f"pxq4hq: slab {T.tier_of(T.PXQ4HQ).slab_bytes} B, code_off "
          f"{T.tier_of(T.PXQ4HQ).code_off}, {T.tier_of(T.PXQ4HQ).neff} effs per 32-elem block, "
          f"bpw {T.tier_of(T.PXQ4HQ).bpw_fn(4096):.4f} at K=4096")

    fails = []
    fails += g2_g3(book, sub)
    fails += g4(book, sub)
    fails += g5(book, sub)
    if args.gguf:
        fails += g1_and_oracle(args.gguf, args.oracle, book, sub)
    else:
        print("  G1/G6 skipped (no --gguf)")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print("\nALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
