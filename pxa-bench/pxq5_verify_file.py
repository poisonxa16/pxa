#!/usr/bin/env python3
# pxq5_verify_file.py — G3 gate A: file-level independent verification of a PXQ5 GGUF.
#
# Given the PXQ5 file and (optionally) its Q8_0 source, for N sampled experts of every
# PXQ5 tensor: dequant the PXQ5 slabs with the REFERENCE tables (pxq5_quantize.py — the
# same frozen numerics the quantizer used) and, when the source is given, dequant the
# matching Q8_0 expert and report rel_l2. PASS band (from the CPU PoC on real weights):
# per-tensor rel_l2 ~0.06-0.11. A layout/quantizer misread shows up as rel_l2 >> 0.2
# or non-finite values. This is Python-vs-Python + Python-vs-source — the on-device
# kernel gate is pxq5-g3-parity.sh (gate B).
#
# Usage: pxq5_verify_file.py PXQ5.gguf [--src Q8_0.gguf] [--experts 3] [--max-tensors 12]
import argparse, mmap, struct, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxq5_quantize import dequant_tensor as pxq5_dequant, QK, BM

PXQ5_ID, Q8_0_ID = 251, 8

def parse_gguf(path):
    """Minimal read-only GGUF v2/v3 parser -> {name: (dtype, dims, abs_off, nbytes)}, mmap."""
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    def rd(fmt, off):
        sz = struct.calcsize(fmt)
        return struct.unpack_from(fmt, mm, off)[0], off + sz
    assert mm[:4] == b"GGUF", "not a GGUF"
    ver, off = rd("<I", 4)
    assert ver in (2, 3), f"gguf v{ver}"
    n_tensors, off = rd("<Q", off)
    n_kv, off = rd("<Q", off)
    SC = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}
    def rd_str(off):
        n, off = rd("<Q", off)
        return mm[off:off+n].decode("utf-8", "replace"), off + n
    alignment = 32
    for _ in range(n_kv):
        key, off = rd_str(off)
        vt, off = rd("<I", off)
        if vt in SC:
            v, off = rd(SC[vt], off)
            if key == "general.alignment": alignment = v
        elif vt == 8:
            _, off = rd_str(off)
        elif vt == 9:
            et, off = rd("<I", off); cnt, off = rd("<Q", off)
            if et in SC: off += struct.calcsize(SC[et]) * cnt
            elif et == 8:
                for _ in range(cnt): _, off = rd_str(off)
            else: raise ValueError(f"kv arr type {et}")
        else: raise ValueError(f"kv type {vt}")
    tens = []
    for _ in range(n_tensors):
        name, off = rd_str(off)
        nd, off = rd("<I", off)
        dims = []
        for _ in range(nd):
            d, off = rd("<Q", off); dims.append(d)
        dt, off = rd("<I", off)
        rel, off = rd("<Q", off)
        tens.append([name, dt, dims, rel])
    data0 = (off + alignment - 1) // alignment * alignment
    out = {}
    for name, dt, dims, rel in tens:
        out[name] = (dt, dims, data0 + rel)
    return out, mm

def q8_dequant_rows(mm, base, K, row0, nrows):
    nb = K // 32
    raw = np.frombuffer(mm, dtype=np.uint8, count=nrows*nb*34, offset=base + row0*nb*34)
    blk = raw.reshape(nrows*nb, 34)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = blk[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(nrows, K)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pxq5_file")
    ap.add_argument("--src", help="Q8_0 source GGUF for rel_l2")
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--max-tensors", type=int, default=12)
    args = ap.parse_args()

    t5, mm5 = parse_gguf(args.pxq5_file)
    pxq5 = {n: v for n, v in t5.items() if v[0] == PXQ5_ID}
    print(f"{len(pxq5)} PXQ5 tensors in {os.path.basename(args.pxq5_file)}")
    if not pxq5:
        print("VERIFY FAIL: no PXQ5 tensors found"); sys.exit(1)
    tsrc = msrc = None
    if args.src:
        tsrc, msrc = parse_gguf(args.src)

    rng = np.random.default_rng(0)
    fails = 0
    for i, (name, (dt, dims, off)) in enumerate(sorted(pxq5.items())):
        if i >= args.max_tensors: break
        K, R, E = dims[0], dims[1], (dims[2] if len(dims) > 2 else 1)
        exp_bytes = R * (K // 32) * 17
        picks = sorted(rng.choice(E, min(args.experts, E), replace=False))
        rels = []
        for e in picks:
            buf = np.frombuffer(mm5, dtype=np.uint8, count=exp_bytes, offset=off + e*exp_bytes)
            W5 = pxq5_dequant(buf, R, K)
            if not np.isfinite(W5).all():
                print(f"  {name} e{e}: NON-FINITE dequant — FAIL"); fails += 1; continue
            if tsrc and name in tsrc and tsrc[name][0] == Q8_0_ID:
                so = tsrc[name][2] + e * R * (K//32) * 34
                Ws = q8_dequant_rows(msrc, tsrc[name][2], K, e*R, R) if False else \
                     q8_dequant_rows(msrc, so - 0, K, 0, R)  # so already includes expert offset
                rel = float(np.linalg.norm(W5 - Ws) / np.linalg.norm(Ws))
                rels.append(rel)
                if not (0.03 < rel < 0.15):
                    print(f"  {name} e{e}: rel_l2={rel:.4f} OUT OF BAND — FAIL"); fails += 1
        band = f" rel_l2={min(rels):.4f}..{max(rels):.4f}" if rels else " (no src compare)"
        print(f"  {name} [{K}x{R}x{E}] experts {picks}:{band}")
    print("VERIFY " + ("PASS" if fails == 0 else f"FAIL ({fails} failures)"))
    sys.exit(0 if fails == 0 else 1)

if __name__ == "__main__":
    main()
