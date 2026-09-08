"""test_moe_shard_cpu.py -- the FusedMoE TP placement, checked on CPU against real tensors.

the sidecar's own warning, in its own words: `PXQ4MoEMethod.create_weights` asserts N % 64 == 0 and
K % 32 == 0, and a silent truncation there "loads, shards, passes every structural gate and
generates fluent garbage". That is the failure this file exists to make impossible to reach in
a window, because a window is a bad place to discover it.

It reimplements exactly what `PXQ4MoEMethod._place` does -- the same narrows, the same
per-rank sizes, the same anchor-is-replicated-not-cut rule -- and then checks the only thing
that actually matters: that the CONCATENATION of what the ranks hold decodes to the same
weights as the unsharded tensor. Numerically, on real bytes out of the converted checkpoint,
not structurally.

  S1 geometry     the panel/slab divisibility the method asserts, at every TP degree, for both
                  fused parameters and both real shapes
  S2 w13 column   gate goes to panels [0, I_p/64), up to [I_p/64, 2*I_p/64), each rank taking
                  its own slice of each -- and the reassembled [2I, H] equals the unsharded one
  S3 w2 row       the K-slab axis is cut and the ANCHOR IS REPLICATED WHOLE. The replication is
                  the load-bearing part: the anchor is a per-output-row linear scale, so
                  sum_r scale*partial_r == scale*sum_r partial_r; cutting it would be silently
                  wrong at TP>1 and correct at TP=1, i.e. invisible in single-card testing.
  S4 stride       a tensor of the wrong tier is refused rather than reshaped

Run: python3 test_moe_shard_cpu.py --checkpoint <box path>
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import tiers as T          # noqa: E402

PANEL_ROWS, SLAB_COLS = 64, 32


def read_tensor(ckpt: str, idx: dict, key: str):
    with open(os.path.join(ckpt, idx[key]), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        info = hdr[key]
        beg, end = info["data_offsets"]
        f.seek(8 + n + beg)
        raw = f.read(end - beg)
    dt = {"U8": np.uint8, "F16": np.float16, "F32": np.float32}[info["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(info["shape"]).copy()


def place_w13(slabs_g, anch_g, slabs_u, anch_u, tp, rank):
    """_place for shard_id w1 (gate) and w3 (up), verbatim in numpy.

    The parameter is [2*I_p/64, S, SLAB]; `per` is panels per half; gate lands at panel 0 and
    up at panel `per`. Each rank narrows the SAME range out of both loaded halves.
    """
    per = slabs_g.shape[0] // tp                      # panels per half per rank
    s = np.concatenate([slabs_g[rank * per:(rank + 1) * per],
                        slabs_u[rank * per:(rank + 1) * per]], axis=0)
    a = np.concatenate([anch_g[rank * per:(rank + 1) * per],
                        anch_u[rank * per:(rank + 1) * per]], axis=0)
    return s, a


def place_w2(slabs, anchor, tp, rank):
    """_place for shard_id w2 (down): cut the K-slab axis, REPLICATE the anchor."""
    per = slabs.shape[1] // tp
    return slabs[:, rank * per:(rank + 1) * per], anchor.copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--expert", type=int, default=3)
    ap.add_argument("--tps", default="1,2,4")
    args = ap.parse_args()
    tps = [int(x) for x in args.tps.split(",")]

    idx = json.load(open(os.path.join(args.checkpoint,
                                      "model.safetensors.index.json")))["weight_map"]
    qc = json.load(open(os.path.join(args.checkpoint, "config.json")))["quantization_config"]
    stem = (f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}")
    tier13 = T.BY_NAME[qc["pxq_tiers"]["mlp.experts.w13"]]
    tier2 = T.BY_NAME[qc["pxq_tiers"]["mlp.experts.w2"]]
    book13 = np.asarray(qc["tier_books"][tier13.name], dtype=np.float32)
    book2 = np.asarray(qc["tier_books"][tier2.name], dtype=np.float32)
    sub = np.asarray(qc.get("tier_sub") or qc["sub"], dtype=np.float32)

    g_s = read_tensor(args.checkpoint, idx, stem + ".gate_proj.pxq4_slabs")
    g_a = read_tensor(args.checkpoint, idx, stem + ".gate_proj.pxq4_anchor")
    u_s = read_tensor(args.checkpoint, idx, stem + ".up_proj.pxq4_slabs")
    u_a = read_tensor(args.checkpoint, idx, stem + ".up_proj.pxq4_anchor")
    d_s = read_tensor(args.checkpoint, idx, stem + ".down_proj.pxq4_slabs")
    d_a = read_tensor(args.checkpoint, idx, stem + ".down_proj.pxq4_anchor")

    I = g_s.shape[0] * PANEL_ROWS
    H = g_s.shape[1] * SLAB_COLS
    print(f"expert {args.expert} of layer {args.layer}: I={I} H={H}  "
          f"w13 tier {tier13.name}/{tier13.slab_bytes}B  w2 tier {tier2.name}/{tier2.slab_bytes}B")
    fails = 0

    # ---- S4 stride is the tier, and is checked not reshaped
    for name, arr, tier in (("w13", g_s, tier13), ("w2", d_s, tier2)):
        ok = arr.shape[-1] == tier.slab_bytes
        print(f"S4 {name} slab stride {arr.shape[-1]} == tier {tier.name}'s "
              f"{tier.slab_bytes}: {'ok' if ok else 'FAIL'}")
        fails += not ok

    full13 = np.concatenate([T.dequant(g_s, g_a, tier13.type_id, book13, sub),
                             T.dequant(u_s, u_a, tier13.type_id, book13, sub)], axis=0)
    full2 = T.dequant(d_s, d_a, tier2.type_id, book2, sub)

    for tp in tps:
        # ---- S1 geometry, exactly the assertions PXQ4MoEMethod.create_weights makes
        I_p = I // tp
        n13, K13 = 2 * I_p, H
        N2, K2 = H, I_p
        bad = []
        if I % tp: bad.append(f"I={I} not divisible by TP={tp}")
        if n13 % PANEL_ROWS: bad.append(f"w13 N={n13} not a multiple of {PANEL_ROWS}")
        if K13 % SLAB_COLS: bad.append(f"w13 K={K13} not a multiple of {SLAB_COLS}")
        if N2 % PANEL_ROWS: bad.append(f"w2 N={N2} not a multiple of {PANEL_ROWS}")
        if K2 % SLAB_COLS: bad.append(f"w2 K={K2} not a multiple of {SLAB_COLS}")
        print(f"S1 TP={tp}: w13 [{n13} x {K13}] w2 [{N2} x {K2}] "
              f"{'ok' if not bad else 'REFUSED: ' + '; '.join(bad)}")
        if bad:
            continue          # a TP this shape cannot take is a legitimate refusal, not a fail

        # ---- S2 w13 column parallel: rank slices reassemble to the unsharded tensor
        parts = []
        for r in range(tp):
            s, a = place_w13(g_s, g_a, u_s, u_a, tp, r)
            if s.shape[0] != n13 // PANEL_ROWS:
                print(f"S2 TP={tp} rank {r}: got {s.shape[0]} panels, want {n13 // PANEL_ROWS}")
                fails += 1
            parts.append(T.dequant(s, a, tier13.type_id, book13, sub))
        # rank r holds gate rows [r*I_p, (r+1)*I_p) then up rows [r*I_p, (r+1)*I_p), so the
        # reassembly interleaves halves rather than concatenating ranks end to end
        gate = np.concatenate([p[:I_p] for p in parts], axis=0)
        up = np.concatenate([p[I_p:] for p in parts], axis=0)
        ok = np.array_equal(np.concatenate([gate, up], axis=0), full13)
        print(f"S2 TP={tp} w13 column shard reassembles exactly: {'ok' if ok else 'FAIL'}")
        fails += not ok

        # ---- S3 w2 row parallel: the K pieces reassemble, and the anchor is replicated
        cols = []
        for r in range(tp):
            s, a = place_w2(d_s, d_a, tp, r)
            if not np.array_equal(a, d_a):
                print(f"S3 TP={tp} rank {r}: the anchor was CUT, not replicated")
                fails += 1
            cols.append(T.dequant(s, a, tier2.type_id, book2, sub))
        ok = np.array_equal(np.concatenate(cols, axis=1), full2)
        print(f"S3 TP={tp} w2 row shard reassembles exactly (anchor replicated): "
              f"{'ok' if ok else 'FAIL'}")
        fails += not ok

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
