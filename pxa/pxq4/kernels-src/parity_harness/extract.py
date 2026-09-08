"""
extract.py -- pull real PXQ4 tensors out of the GGUF into a small, portable .npz fixture.

Run it where the 14.6 GiB artifact lives (the DGX), copy the ~few-MB .npz to wherever the
gates run.  A panel subrange of a PXQ4 tensor is itself a valid PXQ4 tensor -- that is
exactly the column-shard argument -- so `--panels 4` gives a real fixture at 1/272 the
size of ffn_gate with no loss of test coverage.

    python -m parity_harness.extract \
        --gguf /mnt/models/pxa-models/Qwen3.8-27B-PXQ4.gguf \
        --out  /mnt/models/pxa-fixtures/pxq4_real.npz \
        --panels 4

By default it takes one tensor of each of the six distinct PXQ4 shapes, plus the FULL
first ffn_down (K=17408, the widest K in the model and the one whose mmv dynamic smem
exceeds 48 KiB at TP<=2 -- plan §7.3) when --full-widest is given.

It also saves `pxa.pxq6.book` / `pxa.pxq6.sub` / `pxa.pxq.backbone_map` from the file's
KV table, because the compiled-in tables are only correct for this file if they match
(PXA_PXQ6_BOOK can override them at quantize time) and the gates check that.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import oracle as O
from .gguf_raw import GGUFRaw

# The wanted-shape list is (ne0=K, ne1=rows) as ggml stores it.
_WANT = [
    ("attn_gate",   "blk.0.attn_gate.weight"),
    ("attn_qkv",    "blk.0.attn_qkv.weight"),
    ("attn_q",      "blk.3.attn_q.weight"),
    ("attn_output", "blk.3.attn_output.weight"),
    ("ffn_gate",    "blk.0.ffn_gate.weight"),
    ("ffn_down",    "blk.0.ffn_down.weight"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", type=int, default=4,
                    help="panels (64 rows each) to take per tensor; -1 = whole tensor")
    ap.add_argument("--panel0", type=int, default=0,
                    help="first panel to take; use >0 to prove the extract is not "
                         "accidentally always reading the start of the data section")
    ap.add_argument("--full-widest", action="store_true",
                    help="also store blk.0.ffn_down.weight in full (K=17408)")
    ap.add_argument("--tensor", action="append", default=[],
                    help="extra tensor to extract, as label=ggml.name")
    args = ap.parse_args(argv)

    save = {}
    manifest = []
    with GGUFRaw(args.gguf) as g:
        kv_keep = {k: v for k, v in g.kv.items()
                   if k.startswith("pxa.") or k in ("general.architecture",
                                                    "general.alignment")}
        wanted = list(_WANT)
        for spec in args.tensor:
            label, _, name = spec.partition("=")
            wanted.append((label, name))

        for label, name in wanted:
            t = g.tensors.get(name)
            if t is None:
                print(f"  skip {label}: {name} not in file", file=sys.stderr)
                continue
            if t.type_id != O.TYPE_ID:
                print(f"  skip {label}: {name} is {t.type_name}, not PXQ4", file=sys.stderr)
                continue
            K, rows = t.K, t.rows
            O.assert_geometry(rows, K)
            pb = O.panel_bytes(K)
            # This is the first place the on-disk size is checked against the geometry.
            # If it disagrees, every downstream conclusion is void, so fail hard here.
            if t.nbytes != O.tensor_bytes(rows, K):
                raise SystemExit(f"{name}: on-disk {t.nbytes} B != geometry "
                                 f"{O.tensor_bytes(rows, K)} B -- layout assumption is WRONG")
            npan = rows // O.PANEL_ROWS if args.panels < 0 else min(args.panels,
                                                                   rows // O.PANEL_ROWS - args.panel0)
            blob = g.raw(name, panel0=args.panel0, npanels=npan, panel_bytes=pb)
            N_sub = npan * O.PANEL_ROWS
            save[f"{label}|blob"] = np.frombuffer(blob, dtype=np.uint8)
            save[f"{label}|N"] = np.int64(N_sub)
            save[f"{label}|K"] = np.int64(K)
            manifest.append({"label": label, "ggml_name": name, "N": N_sub, "K": K,
                             "panel0": args.panel0, "panels": npan,
                             "full_rows": rows, "bytes": len(blob)})
            print(f"  {label:12s} {name:32s} N={N_sub:6d} K={K:6d} {len(blob):>10d} B")

        if args.full_widest:
            name = "blk.0.ffn_down.weight"
            t = g.tensors.get(name)
            if t is not None and t.type_id == O.TYPE_ID:
                blob = g.raw(name)
                save["ffn_down_full|blob"] = np.frombuffer(blob, dtype=np.uint8)
                save["ffn_down_full|N"] = np.int64(t.rows)
                save["ffn_down_full|K"] = np.int64(t.K)
                manifest.append({"label": "ffn_down_full", "ggml_name": name,
                                 "N": t.rows, "K": t.K, "bytes": len(blob)})
                print(f"  ffn_down_full {name} N={t.rows} K={t.K} {len(blob)} B")

    save["__kv__"] = np.array(kv_keep, dtype=object)
    save["__manifest__"] = np.array(json.dumps(manifest, indent=1), dtype=object)
    np.savez_compressed(args.out, **save)
    print(f"wrote {args.out}")
    # Print the tables so they can be eyeballed against ggml-pxq6-tables.h without numpy.
    for k in ("pxa.pxq6.book", "pxa.pxq6.sub", "pxa.pxq.backbone_map", "pxa.pxq.backbone_rev"):
        if k in kv_keep:
            print(f"  KV {k} = {kv_keep[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
