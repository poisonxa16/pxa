#!/usr/bin/env python3
# pxq4_repack.py — lossless MXFP4 -> PXQ4-LEGACY GGUF repack (PXA-native slab type, id 250).
#
# ⚠ RETIRED 2026-07-21: type id 250 was REMOVED from the fork (enum/traits/kernels/quantize
# path all gone) — the engine now refuses id-250 files with a clean error. Do NOT use this
# tool to PRODUCE id-250 files anymore. It is kept ONLY for --reverse (id-250 -> MXFP4), the
# migration path for old files: reverse-repack, then llama-quantize PXQ4 from a real source.
#
# NOTE (2026-07-19 re-ladder): this LEGACY type was previously displayed "PXQ4"; that name now
# belongs to the 4-bit quality tier (id 252, formerly PXQ6). As of 2026-07-21 llama-quantize no
# longer has any id-250 target at all (the type is retired; see the RETIRED note above).
#
# PXQ4-LEGACY keeps MXFP4's numerics bit-for-bit (E2M1 codes x E8M0 scale, 32-elem blocks, 4.25 bpw,
# 17 B per 32 elems) and only PERMUTES the bits into the fused-kernel slab layout:
#   slab = 64 rows x 32 K-values = 1088 B: [64 scale bytes][64 x 16 B nibble rows,
#   byte b of a row = code(k=2b) | code(k=2b+1) << 4]; slabs K-major within a 64-row panel;
#   panels row-major; experts outermost.
# Because type_size(17)/blck_size(32) are identical, every tensor offset in the file is
# UNCHANGED: the repack is copy + in-place permute + dtype-field patch (39 -> 250).
#
# Selection: tensors named *_exps.weight, dtype MXFP4, ne1 % 64 == 0, ne0 % 32 == 0.
# Everything else stays byte-identical.
#
# Usage:
#   pxq4_repack.py --selftest
#   pxq4_repack.py --src in.gguf --dst out.gguf [--verify-experts 2] [--reverse]
import argparse, os, shutil, struct, sys
import numpy as np

MXFP4_ID, PXQ4_ID = 39, 250
KVAL = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.float32)

# ---------------- GGUF header parse (records the file offset of each tensor dtype field) ----------------
class T:
    __slots__ = ("name","dims","dtype","rel_off","dtype_fpos","nbytes","abs_off")

def _rd(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))

def _rd_str(f):
    (n,) = _rd(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")

_KV_SCALAR = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}

def _skip_kv_value(f, vt):
    if vt in _KV_SCALAR:
        f.read(struct.calcsize(_KV_SCALAR[vt])); return None
    if vt == 8:
        return _rd_str(f)
    if vt == 9:
        (et,) = _rd(f, "<I"); (cnt,) = _rd(f, "<Q")
        if et in _KV_SCALAR:
            f.read(struct.calcsize(_KV_SCALAR[et]) * cnt)
        elif et == 8:
            for _ in range(cnt): _rd_str(f)
        else:
            raise ValueError(f"nested array kv type {et}")
        return None
    raise ValueError(f"unknown kv type {vt}")

def parse_gguf(path):
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", f"not a GGUF: {magic!r}"
    (version,) = _rd(f, "<I")
    assert version in (2, 3), f"gguf version {version}"
    (n_tensors,) = _rd(f, "<Q")
    (n_kv,) = _rd(f, "<Q")
    alignment = 32
    for _ in range(n_kv):
        key = _rd_str(f)
        (vt,) = _rd(f, "<I")
        if key == "general.alignment" and vt == 4:
            (alignment,) = _rd(f, "<I")
        else:
            _skip_kv_value(f, vt)
    tensors = []
    for _ in range(n_tensors):
        t = T()
        t.name = _rd_str(f)
        (nd,) = _rd(f, "<I")
        t.dims = list(_rd(f, f"<{nd}Q"))
        t.dtype_fpos = f.tell()
        (t.dtype,) = _rd(f, "<I")
        (t.rel_off,) = _rd(f, "<Q")
        tensors.append(t)
    header_end = f.tell()
    data_start = (header_end + alignment - 1) // alignment * alignment
    f.close()
    # nbytes for the types we care about (MXFP4/PXQ4): ne0/32*17 * prod(rest)
    for t in tensors:
        if t.dtype in (MXFP4_ID, PXQ4_ID):
            n = t.dims[0] // 32 * 17
            for d in t.dims[1:]: n *= d
            t.nbytes = int(n)
        else:
            t.nbytes = None
        t.abs_off = data_start + t.rel_off
    return tensors, data_start, alignment

# ---------------- the permutation (vectorized, per expert-chunk) ----------------
def mxfp4_to_pxq4(raw, R, K):
    """raw: uint8 array (E_chunk * R * K/32 * 17). Returns permuted bytes, same length."""
    KB = K // 32
    P = R // 64
    blk = raw.reshape(-1, R, KB, 17)
    E = blk.shape[0]
    scales = blk[..., 0]                                  # (E,R,KB)
    qs = blk[..., 1:]                                     # (E,R,KB,16)
    lo = qs & 0x0F                                        # k = 0..15
    hi = qs >> 4                                          # k = 16..31
    codes = np.concatenate([lo, hi], axis=-1)             # (E,R,KB,32) indexed by k
    pairs = codes[..., 0::2] | (codes[..., 1::2] << 4)    # (E,R,KB,16)
    scales_t = np.ascontiguousarray(scales.reshape(E, P, 64, KB).transpose(0, 1, 3, 2))          # (E,P,KB,64)
    pairs_t  = np.ascontiguousarray(pairs.reshape(E, P, 64, KB, 16).transpose(0, 1, 3, 2, 4))    # (E,P,KB,64,16)
    slab = np.concatenate([scales_t, pairs_t.reshape(E, P, KB, 1024)], axis=-1)                  # (E,P,KB,1088)
    return np.ascontiguousarray(slab).reshape(-1)

def pxq4_to_mxfp4(raw, R, K):
    """exact inverse of mxfp4_to_pxq4"""
    KB = K // 32
    P = R // 64
    slab = raw.reshape(-1, P, KB, 1088)
    E = slab.shape[0]
    scales_t = slab[..., :64]                                             # (E,P,KB,64)
    pairs_t  = slab[..., 64:].reshape(E, P, KB, 64, 16)
    scales = scales_t.transpose(0, 1, 3, 2).reshape(E, R, KB)
    pairs  = pairs_t.transpose(0, 1, 3, 2, 4).reshape(E, R, KB, 16)
    codes = np.empty((E, R, KB, 32), dtype=np.uint8)
    codes[..., 0::2] = pairs & 0x0F
    codes[..., 1::2] = pairs >> 4
    lo, hi = codes[..., :16], codes[..., 16:]
    blk = np.empty((E, R, KB, 17), dtype=np.uint8)
    blk[..., 0] = scales
    blk[..., 1:] = lo | (hi << 4)
    return blk.reshape(-1)

# ---------------- independent dequants (for the bit-exactness proof) ----------------
def _e8m0(e):
    e = e.astype(np.uint32)
    u = np.where(e >= 2, (e - 1) << 23, np.where(e == 1, 0x00400000, 0x00200000)).astype(np.uint32)
    return u.view(np.float32)

def dequant_mxfp4(raw, R, K):
    KB = K // 32
    blk = raw.reshape(R, KB, 17)
    d = _e8m0(blk[..., 0])                                # (R,KB)
    qs = blk[..., 1:]
    out = np.empty((R, KB, 32), dtype=np.float32)
    out[..., :16] = KVAL[qs & 0x0F]
    out[..., 16:] = KVAL[qs >> 4]
    out *= d[..., None]
    return out.reshape(R, K)

def dequant_pxq4(raw, R, K):
    """dequant straight from the slab layout with independent index math."""
    KB = K // 32
    P = R // 64
    slab = raw.reshape(P, KB, 1088)
    d = _e8m0(slab[..., :64])                             # (P,KB,64) scale per (panel,kb,row)
    pairs = slab[..., 64:].reshape(P, KB, 64, 16)
    vals = np.empty((P, KB, 64, 32), dtype=np.float32)
    vals[..., 0::2] = KVAL[pairs & 0x0F]
    vals[..., 1::2] = KVAL[pairs >> 4]
    vals *= d[..., None]                                  # d is (P,KB,64) -> broadcast over 32 k
    # arrange to (R, K): row = p*64 + r, k = kb*32 + j
    return vals.transpose(0, 2, 1, 3).reshape(R, K)

def selftest():
    rng = np.random.default_rng(42)
    E, R, K = 3, 192, 160
    raw = rng.integers(0, 256, size=E * R * (K // 32) * 17, dtype=np.uint8)
    px = mxfp4_to_pxq4(raw, R, K)
    assert px.shape == raw.shape
    back = pxq4_to_mxfp4(px, R, K)
    assert np.array_equal(raw, back), "roundtrip FAILED"
    per_e = R * (K // 32) * 17
    for e in range(E):
        a = dequant_mxfp4(raw[e*per_e:(e+1)*per_e], R, K)
        b = dequant_pxq4(px[e*per_e:(e+1)*per_e], R, K)
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), f"dequant mismatch expert {e}"
    # non-trivial permutation sanity: bytes actually moved
    assert not np.array_equal(raw, px)
    print("SELFTEST PASS: roundtrip byte-exact + dequant bit-exact (fp32 views equal) on synthetic tensor")

def eligible(t, reverse=False):
    want_src = PXQ4_ID if reverse else MXFP4_ID
    return (t.dtype == want_src and t.name.endswith("_exps.weight")
            and len(t.dims) >= 2 and t.dims[1] % 64 == 0 and t.dims[0] % 32 == 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src"); ap.add_argument("--dst")
    ap.add_argument("--verify-experts", type=int, default=2,
                    help="per repacked tensor, dequant-verify this many experts (0=off)")
    ap.add_argument("--reverse", action="store_true", help="PXQ4 -> MXFP4 (inverse)")
    ap.add_argument("--chunk-experts", type=int, default=8)
    args = ap.parse_args()

    if args.selftest:
        selftest(); return

    assert args.src and args.dst and args.src != args.dst
    tensors, data_start, alignment = parse_gguf(args.src)
    todo = [t for t in tensors if eligible(t, args.reverse)]
    tot = sum(t.nbytes for t in todo)
    print(f"{len(tensors)} tensors, data@{data_start}, align {alignment}; repacking {len(todo)} "
          f"expert tensors ({tot/1e9:.2f} GB) {'PXQ4->MXFP4' if args.reverse else 'MXFP4->PXQ4'}")
    for t in todo[:3]:
        print(f"  e.g. {t.name} dims={t.dims} off={t.abs_off}")

    if not os.path.exists(args.dst) or os.path.getsize(args.dst) != os.path.getsize(args.src):
        print(f"copying {args.src} -> {args.dst} ...")
        shutil.copyfile(args.src, args.dst)
    else:
        print("dst exists with matching size; reusing (will overwrite headers+expert regions)")

    fwd = pxq4_to_mxfp4 if args.reverse else mxfp4_to_pxq4
    new_id = MXFP4_ID if args.reverse else PXQ4_ID

    src = open(args.src, "rb")
    dst = open(args.dst, "r+b")
    done = 0
    for n, t in enumerate(todo):
        K, R = t.dims[0], t.dims[1]
        E = 1
        for d in t.dims[2:]: E *= d
        per_e = t.nbytes // E
        assert per_e * E == t.nbytes
        # patch dtype field
        dst.seek(t.dtype_fpos); dst.write(struct.pack("<I", new_id))
        # permute data per expert chunk
        for e0 in range(0, E, args.chunk_experts):
            ec = min(args.chunk_experts, E - e0)
            src.seek(t.abs_off + e0 * per_e)
            raw = np.frombuffer(src.read(ec * per_e), dtype=np.uint8)
            out = fwd(raw, R, K)
            dst.seek(t.abs_off + e0 * per_e)
            dst.write(out.tobytes())
        done += t.nbytes
        if (n + 1) % 12 == 0 or n + 1 == len(todo):
            print(f"  [{n+1}/{len(todo)}] {done/1e9:.1f}/{tot/1e9:.1f} GB", flush=True)
    dst.flush()

    # verification: independent dequant equality src-vs-dst on sample experts of every tensor
    if args.verify_experts > 0 and not args.reverse:
        print("verifying (independent dequant, bit-exact fp32 compare) ...")
        for n, t in enumerate(todo):
            K, R = t.dims[0], t.dims[1]
            E = 1
            for d in t.dims[2:]: E *= d
            per_e = t.nbytes // E
            picks = sorted({0, E - 1, (E // 2)})[:max(1, args.verify_experts)]
            for e in picks:
                src.seek(t.abs_off + e * per_e); a_raw = np.frombuffer(src.read(per_e), np.uint8)
                dst.seek(t.abs_off + e * per_e); b_raw = np.frombuffer(dst.read(per_e), np.uint8)
                a = dequant_mxfp4(a_raw, R, K)
                b = dequant_pxq4(b_raw, R, K)
                if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
                    print(f"*** VERIFY FAIL: {t.name} expert {e}"); sys.exit(1)
                rt = pxq4_to_mxfp4(b_raw, R, K)
                if not np.array_equal(rt, a_raw):
                    print(f"*** ROUNDTRIP FAIL: {t.name} expert {e}"); sys.exit(1)
            if (n + 1) % 24 == 0 or n + 1 == len(todo):
                print(f"  verified [{n+1}/{len(todo)}]", flush=True)
        print("VERIFY PASS: dequant bit-exact + roundtrip byte-exact on sampled experts of all tensors")
    src.close(); dst.close()
    print("done.")

if __name__ == "__main__":
    main()
