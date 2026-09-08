"""
extract_raw.py -- STDLIB-ONLY fixture extractor.

Same job as extract.py, but with zero third-party imports, because the box that holds the
14.6 GiB artifact does not necessarily have numpy (the DGX host does not), and the only
python that does live inside a production container that must not be disturbed.

Writes a directory:
    <out>/manifest.json          labels, N, K, ggml names, byte counts, and the pxa.* KVs
    <out>/<label>.bin            raw PXQ4 panel bytes, directly copyable

Load it back with parity_harness.fixtures.load_raw_dir().

    python3 extract_raw.py --gguf /mnt/models/pxa-models/Qwen3.8-27B-PXQ4.gguf \
                          --out  /mnt/models/pxa-fixtures/raw --panels 4

Run it with `python3 extract_raw.py` directly (no package import needed): it inlines the
GGUF parse so the file can be scp'd anywhere on its own.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys

PANEL_ROWS, SLAB_COLS, SLAB_BYTES, HEADER_BYTES, TYPE_ID = 64, 32, 1088, 128, 252

(T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL,
 T_STR, T_ARR, T_U64, T_I64, T_F64) = range(13)
_FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
        T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
_SZ = {k: struct.calcsize(v) for k, v in _FMT.items()}

DEFAULT_WANT = [
    ("attn_gate",   "blk.0.attn_gate.weight"),
    ("attn_qkv",    "blk.0.attn_qkv.weight"),
    ("attn_q",      "blk.3.attn_q.weight"),
    ("attn_output", "blk.3.attn_output.weight"),
    ("ffn_gate",    "blk.0.ffn_gate.weight"),
    ("ffn_down",    "blk.0.ffn_down.weight"),
]


def panel_bytes(K):
    return HEADER_BYTES + (K // SLAB_COLS) * SLAB_BYTES


def tensor_bytes(N, K):
    return (N // PANEL_ROWS) * panel_bytes(K)


class _Reader:
    def __init__(self, f):
        self.f = f
        self.pos = 0

    def rd(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.rd(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.rd(8))[0]

    def s(self):
        return self.rd(self.u64()).decode("utf-8", "replace")

    def val(self, t):
        if t == T_ARR:
            et, n = self.u32(), self.u64()
            if et == T_STR:
                return [self.s() for _ in range(n)]
            raw = self.rd(_SZ[et] * n)
            return list(struct.unpack("<%d%s" % (n, _FMT[et][1]), raw))
        if t == T_STR:
            return self.s()
        return struct.unpack(_FMT[t], self.rd(_SZ[t]))[0]


def parse(path):
    f = open(path, "rb")
    r = _Reader(f)
    if r.rd(4) != b"GGUF":
        raise SystemExit(f"{path}: not GGUF")
    r.u32()
    nt, nkv = r.u64(), r.u64()
    kv = {}
    for _ in range(nkv):
        k = r.s()
        kv[k] = r.val(r.u32())
    tensors = []
    for _ in range(nt):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        tid = r.u32()
        off = r.u64()
        tensors.append({"name": name, "dims": dims, "type_id": tid, "offset": off})
    align = int(kv.get("general.alignment", 32))
    data_start = (r.pos + align - 1) // align * align
    fsize = os.path.getsize(path)
    order = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(order):
        nxt = order[i + 1]["offset"] if i + 1 < len(order) else fsize - data_start
        t["nbytes"] = nxt - t["offset"]
    return f, kv, {t["name"]: t for t in tensors}, data_start


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", type=int, default=4)
    ap.add_argument("--panel0", type=int, default=0)
    ap.add_argument("--tensor", action="append", default=[])
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    f, kv, tensors, data_start = parse(args.gguf)
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    want = list(DEFAULT_WANT)
    for spec in args.tensor:
        label, _, name = spec.partition("=")
        want.append((label, name))

    manifest = {"gguf": args.gguf, "tensors": [], "kv": {}}
    for k, v in kv.items():
        if k.startswith("pxa.") or k in ("general.architecture", "general.alignment"):
            manifest["kv"][k] = v

    for label, name in want:
        t = tensors.get(name)
        if t is None:
            print(f"  skip {label}: {name} absent", file=sys.stderr)
            continue
        if t["type_id"] != TYPE_ID:
            print(f"  skip {label}: {name} type {t['type_id']} != PXQ4", file=sys.stderr)
            continue
        K = t["dims"][0]
        rows = t["dims"][1] if len(t["dims"]) > 1 else 1
        if rows % PANEL_ROWS or K % SLAB_COLS:
            raise SystemExit(f"{name}: geometry rows={rows} K={K} violates %64/%32")
        # Hard check: if the on-disk length disagrees with the panel formula, every
        # conclusion downstream is void.  Fail here, not three components later.
        if t["nbytes"] != tensor_bytes(rows, K):
            raise SystemExit(f"{name}: on-disk {t['nbytes']} B != geometry "
                             f"{tensor_bytes(rows, K)} B -- LAYOUT ASSUMPTION IS WRONG")
        pb = panel_bytes(K)
        maxp = rows // PANEL_ROWS - args.panel0
        npan = maxp if args.panels < 0 else min(args.panels, maxp)
        if npan <= 0:
            print(f"  skip {label}: panel0={args.panel0} past end", file=sys.stderr)
            continue
        base = data_start + t["offset"] + args.panel0 * pb
        blob = mm[base:base + npan * pb]
        outp = os.path.join(args.out, f"{label}.bin")
        with open(outp, "wb") as g:
            g.write(blob)
        manifest["tensors"].append({
            "label": label, "ggml_name": name, "N": npan * PANEL_ROWS, "K": K,
            "panel0": args.panel0, "panels": npan, "bytes": len(blob),
            "full_rows": rows, "full_bytes": t["nbytes"],
        })
        print(f"  {label:12s} {name:30s} N={npan*PANEL_ROWS:6d} K={K:6d} {len(blob):>9d} B")

    with open(os.path.join(args.out, "manifest.json"), "w") as g:
        json.dump(manifest, g, indent=1)
    mm.close()
    f.close()
    print(f"wrote {args.out}/manifest.json")
    for k in ("pxa.pxq.backbone_rev", "pxa.pxq.backbone_map", "pxa.pxq6.tier"):
        if k in manifest["kv"]:
            print(f"  KV {k} = {manifest['kv'][k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
