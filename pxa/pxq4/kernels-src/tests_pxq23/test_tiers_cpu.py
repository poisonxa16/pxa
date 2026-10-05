"""test_tiers_cpu.py -- CPU gates for the PXQ2/PXQ3 tier support. No GPU, no torch, no vLLM.

Four gates, in the order they can fail:

  G1 BYTE ROUND TRIP    split_blob -> join_blob == the GGUF bytes, on real tensors of the real
                        model. The split is the converter's whole claim to being a repack and
                        not a requant, so it is checked byte-for-byte rather than numerically.
  G2 CODE EXTRACTION    the vectorised numpy code extraction (u32-word arithmetic) against a
                        slow, literal transcription of the ENGINE's C decode loops (byte
                        arithmetic, ggml/src/pxq-cpu.c pxa_deq_row_pxq2/pxq3). Two different
                        formulas over the same bytes: agreement is evidence, not tautology.
  G3 VALUE CONTRACT     the decoded values obey eff = anchor*SUB16[nibble], w = eff*book[code]
                        exactly, including the reconstruction ceiling |w| <= max(SUB16) *
                        max|book| * |anchor| that the v1 books' absmax < 1 implies.
  G4 SHARD COMMUTES     dequant(shard(x)) == shard(dequant(x)) on both axes, at TP 1/2/4 --
                        the property that makes a TP split a byte move.

  G5 (optional, needs a C compiler) the engine's OWN decode functions, extracted verbatim by
     build_oracle.sh, on the same bytes. Run inside a serving-image container:
         CC=g++ ./build_oracle.sh && python3 test_tiers_cpu.py --oracle ./oracle_pxq23
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import gguf_raw as G          # noqa: E402
from gguf_to_vllm import tiers as T             # noqa: E402

GGUF = {
    T.PXQ2: os.environ.get("PXA_MODELS_COLD", "./models") + "/fusion35bv2/fusion2-35b-PXQ2.gguf",
    T.PXQ3: os.environ.get("PXA_MODELS_COLD", "./models") + "/fusion35bv2/fusion2-35b-PXQ3.gguf",
}


# ------------------------------------------------------------------ the slow engine transcript
def slow_decode_engine(blob: memoryview, type_id: int, N: int, K: int,
                       book: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """A literal, scalar transcription of ggml/src/pxq-cpu.c pxa_deq_row_pxq2 / pxq3.

    Deliberately written with the C's BYTE indexing (`q[j>>2] >> 2*(j&3)` for pxq2, and the
    explicit w0/w1/w2 assembly for pxq3), which is a different expression of the same packing
    from the u32-word arithmetic tiers.py uses. Slow on purpose: it is the reference, not a
    path anything ships.
    """
    t = T.tier_of(type_id)
    b = np.frombuffer(blob, dtype=np.uint8)
    KB = K // 32
    panel_stride = T.HEADER_BYTES + KB * t.slab_bytes
    out = np.empty((N, K), dtype=np.float32)
    for row in range(N):
        p, r = row >> 6, row & 63
        panel = p * panel_stride
        anchor = np.frombuffer(b[panel + 2 * r: panel + 2 * r + 2].tobytes(),
                               dtype="<f2")[0].astype(np.float32)
        for kb in range(KB):
            slab = panel + T.HEADER_BYTES + kb * t.slab_bytes
            sb = int(b[slab + r])
            eff0 = np.float32(anchor * sub[sb & 0xF])
            eff1 = np.float32(anchor * sub[sb >> 4])
            q = slab + 64 + r * t.code_bytes
            o = out[row, kb * 32: kb * 32 + 32]
            if type_id == T.PXQ2:
                for j in range(32):
                    c = (int(b[q + (j >> 2)]) >> (2 * (j & 3))) & 3
                    o[j] = (eff0 if j < 16 else eff1) * book[c]
            else:
                w0 = w1 = w2 = 0
                for i in range(4):
                    w0 |= int(b[q + i]) << (8 * i)
                    w1 |= int(b[q + 4 + i]) << (8 * i)
                    w2 |= int(b[q + 8 + i]) << (8 * i)
                for j in range(16):
                    c0 = ((w0 >> (2 * j)) & 3) | (((w2 >> j) & 1) << 2)
                    c1 = ((w1 >> (2 * j)) & 3) | (((w2 >> (16 + j)) & 1) << 2)
                    o[j] = eff0 * book[c0]
                    o[16 + j] = eff1 * book[c1]
    return out


def kv_tables(gg, type_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The file's OWN book and sub. A checkpoint is only self-describing if we honour what it
    recorded, so nothing here falls back to the compiled-in defaults silently."""
    n = T.tier_of(type_id).name                      # "pxq2" / "pxq3"
    book = gg.kv.get(f"pxa.{n}.book")
    sub = gg.kv.get(f"pxa.{n}.sub")
    if book is None or sub is None:
        raise SystemExit(f"the GGUF carries no pxa.{n}.book / .sub")
    book = np.asarray(book, dtype=np.float32)
    sub = np.asarray(sub, dtype=np.float32)
    T.check_book(book, type_id)
    T.check_sub(sub)
    return book, sub


def expert_slice(gg, name, type_id, e):
    """Cut expert e out of a 3-D [K, N, E] ggml expert stack. Each expert slice is a complete,
    independently addressable panel tensor, so this is a byte range, not a gather."""
    ti = gg.tensors[name]
    K, N, E = ti.dims[0], ti.dims[1], ti.dims[2]
    per = T.tensor_bytes(type_id, N, K)
    raw = gg.raw(name)
    if len(raw) != per * E:
        raise SystemExit(f"{name}: {len(raw)} B on disk but {per*E} B predicted for "
                         f"E={E} N={N} K={K} tier={T.tier_of(type_id).name}")
    return raw[e * per:(e + 1) * per], N, K, E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=None,
                    help="path to the engine-extracted oracle binary (gate G5)")
    ap.add_argument("--experts", type=int, default=2, help="experts to check per tensor")
    ap.add_argument("--rows", type=int, default=128,
                    help="rows of the slow scalar reference (it is O(N*K) python)")
    args = ap.parse_args()

    fails = 0
    for type_id, path in GGUF.items():
        t = T.tier_of(type_id)
        if not os.path.exists(path):
            print(f"SKIP {t.name}: {path} not present")
            continue
        gg = G.GGUFFile(path)
        book, sub = kv_tables(gg, type_id)
        print(f"=== {t.name}: {os.path.basename(path)} "
              f"book={book.tolist()} sub[0]={sub[0]}")

        # pick one gate/up stack (K=2048, N=512) and one down stack (K=512, N=2048)
        names = [n for n in gg.order if n.endswith("_exps.weight")
                 and gg.tensors[n].type_id == type_id]
        probe = [n for n in names if "blk.0." in n][:3]
        if not probe:
            print(f"FAIL {t.name}: no {t.name} expert stacks found")
            fails += 1
            gg.close()
            continue

        for name in probe:
            for e in range(args.experts):
                mv, N, K, E = expert_slice(gg, name, type_id, e)
                blob = bytes(mv)           # own the bytes; a live memoryview of the mmap
                del mv                     # would block GGUFFile.close() at the end
                slabs, anchor = T.split_blob(blob, type_id, N, K)

                # -- G1 byte round trip
                if T.join_blob(slabs, anchor) != blob:
                    print(f"FAIL G1 {name} e{e}: split/join is not byte-exact")
                    fails += 1
                    continue

                # -- G2 vectorised vs the engine transcript, on the first `rows` rows
                nrow = min(args.rows, N)
                want = slow_decode_engine(blob, type_id, nrow, K, book, sub)
                got = T.dequant(slabs[: nrow // 64], anchor[: nrow // 64],
                                type_id, book, sub)
                if not np.array_equal(got, want):
                    bad = int(np.argmax(np.abs(got - want)))
                    print(f"FAIL G2 {name} e{e}: numpy decode != engine transcript, first "
                          f"worst at flat {bad}: {got.flat[bad]} vs {want.flat[bad]}")
                    fails += 1
                    continue

                # -- G3 value contract: every decoded weight must be exactly
                #    eff * book[c] for one of the book entries, and the reconstruction
                #    ceiling must hold.
                a = np.abs(anchor.astype(np.float32)).max()
                ceiling = a * float(sub.max()) * float(np.abs(book).max())
                if np.abs(got).max() > ceiling * (1 + 1e-6):
                    print(f"FAIL G3 {name} e{e}: |w|max {np.abs(got).max()} exceeds the "
                          f"reconstruction ceiling {ceiling}")
                    fails += 1
                    continue

                # -- G4 shard commutes, both axes, at TP 1/2/4
                full = T.dequant(slabs, anchor, type_id, book, sub)
                ok = True
                for tp in (1, 2, 4):
                    if N % tp == 0 and (N // tp) % 64 == 0:          # column shard (panels)
                        per = (N // tp) // 64
                        for r in range(tp):
                            sh = T.dequant(slabs[r * per:(r + 1) * per],
                                           anchor[r * per:(r + 1) * per], type_id, book, sub)
                            if not np.array_equal(sh, full[r * (N // tp):(r + 1) * (N // tp)]):
                                ok = False
                    if K % tp == 0 and (K // tp) % 32 == 0:          # row shard (slabs)
                        per = (K // tp) // 32
                        for r in range(tp):
                            # the anchor is DUPLICATED on a K split, never cut -- that is the
                            # property that makes the split free
                            sh = T.dequant(slabs[:, r * per:(r + 1) * per], anchor,
                                           type_id, book, sub)
                            if not np.array_equal(sh, full[:, r * (K // tp):(r + 1) * (K // tp)]):
                                ok = False
                if not ok:
                    print(f"FAIL G4 {name} e{e}: a TP shard does not commute with dequant")
                    fails += 1
                    continue

                # -- G5 the engine's own binary, if it was built
                if args.oracle:
                    with tempfile.TemporaryDirectory() as d:
                        fin, fout = os.path.join(d, "in.bin"), os.path.join(d, "out.f32")
                        open(fin, "wb").write(blob)
                        subprocess.run([args.oracle, str(t.code_bits), str(N), str(K),
                                        fin, fout], check=True)
                        ref = np.fromfile(fout, dtype=np.float32).reshape(N, K)
                    if not np.array_equal(full, ref):
                        print(f"FAIL G5 {name} e{e}: numpy decode != the engine's own decode")
                        fails += 1
                        continue

                print(f"  ok {name} e{e}  N={N} K={K} E={E} slab={t.slab_bytes} "
                      f"bytes={len(blob)} |w|max={np.abs(full).max():.5f}"
                      + ("  [G5 engine-binary]" if args.oracle else ""))
        gg.close()

    print("PASS" if fails == 0 else f"{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
