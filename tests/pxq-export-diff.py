#!/usr/bin/env python3
"""Tensor-level diff of a PXQ GGUF against its llama-pxq-export output.

Asserts:
  * the two files carry the SAME tensor names in the SAME order;
  * every tensor whose input type is NOT a PXQ slab type is byte-identical -- same type,
    same shape, same bytes;
  * every PXQ tensor became F16 (or F32) with the same shape and the expected byte count.

Deliberately does not use gguf-py's GGUFReader for the tensor payloads: the PXQ types carry a
2 B/row anchor header that the GGML_QUANT_SIZES table cannot express, so the reader's slice for
a PXQ tensor is short. Offsets and shapes come from a minimal parser here; sizes for the
non-PXQ types come from the same table ggml.c uses.
"""
import mmap
import struct
import sys

PXQ_TYPES = {248: "pxq1", 252: "pxq4", 253: "pxq4hq", 254: "pxq2", 255: "pxq3", 256: "pxq6"}

# (blck_size, type_size, row_meta_size) -- only what a real model file can contain.
TYPES = {
    0:  ("f32",    1,  4, 0),
    1:  ("f16",    1,  2, 0),
    2:  ("q4_0",  32, 18, 0),
    3:  ("q4_1",  32, 20, 0),
    6:  ("q5_0",  32, 22, 0),
    7:  ("q5_1",  32, 24, 0),
    8:  ("q8_0",  32, 34, 0),
    9:  ("q8_1",  32, 36, 0),
    10: ("q2_K", 256, 84, 0),
    11: ("q3_K", 256,110, 0),
    12: ("q4_K", 256,144, 0),
    13: ("q5_K", 256,176, 0),
    14: ("q6_K", 256,210, 0),
    15: ("q8_K", 256,292, 0),
    16: ("iq2_xxs",256,66,0),
    17: ("iq2_xs", 256,74,0),
    18: ("iq3_xxs",256,98,0),
    19: ("iq1_s",  256,50,0),
    20: ("iq4_nl",  32,18,0),
    21: ("iq3_s",  256,110,0),
    22: ("iq2_s",  256,82,0),
    23: ("iq4_xs", 256,136,0),
    24: ("i8",      1, 1, 0),
    25: ("i16",     1, 2, 0),
    26: ("i32",     1, 4, 0),
    27: ("i64",     1, 8, 0),
    28: ("f64",     1, 8, 0),
    29: ("iq1_m",  256,56, 0),
    30: ("bf16",    1, 2, 0),
    39: ("mxfp4",  32,17, 0),
    248: ("pxq1",  32, 5, 2),
    252: ("pxq4",  32,17, 2),
    253: ("pxq4hq",32,18, 2),
    254: ("pxq2",  32, 9, 2),
    255: ("pxq3",  32,13, 2),
    256: ("pxq6",  32,21, 2),
}

GT_U8, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR, GT_ARR, GT_U64, GT_I64, GT_F64 = range(13)
FIXED = {GT_U8: 1, GT_I8: 1, GT_U16: 2, GT_I16: 2, GT_U32: 4, GT_I32: 4,
         GT_F32: 4, GT_BOOL: 1, GT_U64: 8, GT_I64: 8, GT_F64: 8}


class Cur:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v

    def i64(self):
        v = struct.unpack_from("<q", self.b, self.o)[0]; self.o += 8; return v

    def s(self):
        n = self.u64(); v = self.b[self.o:self.o + n].decode("utf-8", "replace"); self.o += n; return v

    def skip_val(self, t):
        if t in FIXED:
            self.o += FIXED[t]
        elif t == GT_STR:
            # NB: n must be read BEFORE the +=; `self.o += self.u64()` loads self.o first and
            # loses the 8 bytes u64() consumed.
            n = self.u64(); self.o += n
        elif t == GT_ARR:
            et = self.u32(); n = self.u64()
            if et in FIXED:
                self.o += FIXED[et] * n
            elif et == GT_STR:
                for _ in range(n):
                    ln = self.u64(); self.o += ln
            else:
                raise ValueError(f"array of type {et}")
        else:
            raise ValueError(f"kv type {t}")


def parse(path):
    with open(path, "rb") as f:
        # mmap the whole file: the metadata block can be tens of MiB (a big vocab array) and a
        # fixed-size head read silently truncates it. mmap costs nothing -- only the metadata
        # pages are ever touched.
        head = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, n_tensors, n_kv = struct.unpack_from("<IIQQ", head, 0)
        assert magic == 0x46554747, f"{path}: not a GGUF"
        c = Cur(head); c.o = 24
        align = 32
        for _ in range(n_kv):
            k = c.s()
            t = c.u32()
            if k == "general.alignment":
                assert t == GT_U32
                align = struct.unpack_from("<I", c.b, c.o)[0]
            c.skip_val(t)
        tensors = []
        for _ in range(n_tensors):
            name = c.s()
            nd = c.u32()
            ne = [c.u64() for _ in range(nd)] + [1] * (4 - nd)
            ttype = c.u32()
            off = c.u64()
            tensors.append((name, ne, ttype, off))
        data_off = (c.o + align - 1) // align * align
        head.close()
    return tensors, data_off, align


def nbytes(ne, ttype):
    if ttype not in TYPES:
        raise SystemExit(f"unknown ggml type id {ttype} -- add it to TYPES")
    _, blck, tsz, meta = TYPES[ttype]
    rows = ne[1] * ne[2] * ne[3]
    return rows * (meta + tsz * ne[0] // blck)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    ta, off_a, _ = parse(a_path)
    tb, off_b, _ = parse(b_path)

    if [t[0] for t in ta] != [t[0] for t in tb]:
        sa, sb = set(t[0] for t in ta), set(t[0] for t in tb)
        raise SystemExit(f"tensor sets/order differ: only in input {sorted(sa - sb)[:5]}, "
                         f"only in export {sorted(sb - sa)[:5]}")

    n_copy = n_copy_ok = n_pxq = 0
    fa = open(a_path, "rb"); fb = open(b_path, "rb")
    for (na, nea, tya, oa), (nb_, neb, tyb, ob) in zip(ta, tb):
        if nea != neb:
            raise SystemExit(f"{na}: shape changed {nea} -> {neb}")
        if tya in PXQ_TYPES:
            n_pxq += 1
            if tyb not in (0, 1):
                raise SystemExit(f"{na}: {PXQ_TYPES[tya]} did not become f16/f32 (type id {tyb})")
            want = nbytes(neb, tyb)
            fb.seek(off_b + ob)
            if len(fb.read(want)) != want:
                raise SystemExit(f"{na}: exported tensor is truncated")
            continue
        n_copy += 1
        if tya != tyb:
            raise SystemExit(f"{na}: non-PXQ tensor changed type {tya} -> {tyb}")
        n = nbytes(nea, tya)
        fa.seek(off_a + oa); fb.seek(off_b + ob)
        left = n
        while left:
            k = min(left, 1 << 24)
            if fa.read(k) != fb.read(k):
                raise SystemExit(f"{na}: non-PXQ tensor bytes differ")
            left -= k
        n_copy_ok += 1
    fa.close(); fb.close()

    if n_pxq == 0:
        raise SystemExit("no PXQ tensors in the input -- this diff proves nothing")
    print(f"   {n_copy_ok}/{n_copy} non-PXQ tensors byte-identical, "
          f"{n_pxq} PXQ tensors dequantized, {len(ta)} tensors total")


if __name__ == "__main__":
    main()
