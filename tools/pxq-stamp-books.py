#!/usr/bin/env python3
"""Add the missing pxa.<tier>.book / .sub KVs to a PXQ GGUF, without touching one weight byte.

WHY THIS EXISTS
---------------
Until 2026-09-09 llama-quantize chose which codebook KVs to write from the REQUESTED FTYPE
rather than from the tiers it actually emitted. Under BACKBONE_REV 2 a PXQ2 target on a MoE
model emits pxq2 routed experts AND a pxq4 backbone, and stamped only `pxa.pxq2.*` — so a large
share of everything published before that date contains a tier whose codebook is not in the
file. This engine does not notice (its CUDA side decodes from compiled-in tables), but any
reader that takes the KVs at their word cannot decode those tensors, and the vLLM converter is
exactly such a reader.

Re-quantizing to fix a metadata bug would be hours per file and would produce different bytes.
This tool copies the tensor data verbatim and adds the missing keys.

WHEN IT REFUSES
---------------
Only default codebooks can be re-stamped, because only they can be reconstructed. A file built
with `PXA_PXQ_CEIL_V2=1` or `PXA_PXQ2_V3=1` used a different book, and its version KV says so.
If a tier's version KV is present and is not 1, and the file does not already carry that tier's
book, then the correct table is unknowable from the file alone — writing the v1 default there
would produce a file that is confidently, silently wrong, which is worse than the file we
started with. Those refuse. Re-quantize them instead.

USAGE
    python3 tools/pxq-stamp-books.py --check  model.gguf            # say what is missing
    python3 tools/pxq-stamp-books.py model.gguf out.gguf            # write the fixed copy
    python3 tools/pxq-stamp-books.py --verify model.gguf out.gguf   # per-tensor sha256 proof

The books are read out of this repo's own headers (ggml/include/ggml-pxq{2,3,6}-tables.h) at
run time, not copied into this file, so they cannot drift away from the codec.
"""
import argparse, hashlib, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC  = os.path.join(ROOT, 'ggml', 'include')

# ------------------------------------------------------------------------------------------
# The default tables, parsed from the C headers. The values are C99 hex floats
# (-0x1.694p-1f), which float.fromhex understands once the f suffix is stripped.
# ------------------------------------------------------------------------------------------
def read_macro(path, macro, want):
    src = open(path).read()
    m = re.search(r'#define\s+' + re.escape(macro) + r'\s*\{(.*?)\}', src, re.S)
    if not m:
        raise SystemExit(f'{macro} not found in {path} — the codec headers moved; fix this tool')
    vals = [float.fromhex(v.strip().rstrip('f'))
            for v in m.group(1).replace('\\\n', ' ').replace('\\', ' ').split(',')
            if v.strip()]
    if len(vals) != want:
        raise SystemExit(f'{macro}: expected {want} entries, parsed {len(vals)}')
    return vals

def default_tables():
    sub16 = read_macro(os.path.join(INC, 'ggml-pxq6-tables.h'), 'PXQ6_SUB16_INIT', 16)
    sub8  = read_macro(os.path.join(INC, 'ggml-pxq6-tables.h'), 'PXQ6_SUB8_INIT', 16)
    book16 = read_macro(os.path.join(INC, 'ggml-pxq6-tables.h'), 'PXQ6_BOOK_INIT', 16)
    return {
        # key prefix -> (book, sub). pxq4/pxq6-slab tensors read the pxq6 family.
        'pxa.pxq2': (read_macro(os.path.join(INC, 'ggml-pxq2-tables.h'), 'PXQ2_BOOK_INIT', 4), sub16),
        'pxa.pxq3': (read_macro(os.path.join(INC, 'ggml-pxq3-tables.h'), 'PXQ3_BOOK_INIT', 8), sub16),
        'pxa.pxq6': (book16, sub16),
        # pxq4hq shares the PX16 book with pxq4 and has its OWN bs8 sub table. It gets its own
        # key prefix because a file may hold BOTH tiers, and the pxq6 pair can only describe
        # one of them: the quantizer writes SUB8 there when any hq tensor is present, which
        # leaves the pxq4 tensors' SUB16 unrecorded. See the pxa.pxq4hq block in
        # src/llama-quantize.cpp.
        'pxa.pxq4hq': (book16, sub8),
    }

# ------------------------------------------------------------------------------------------
# Minimal GGUF reader/writer. Stdlib only: the dev image has no numpy, and this tool has to be
# runnable wherever a model file is.
# ------------------------------------------------------------------------------------------
T_U32, T_F32, T_STR, T_ARR, T_U64 = 4, 6, 8, 9, 10
VSZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
GGML_PXQ = {248:'PXQ1', 252:'PXQ4', 253:'PXQ4HQ', 254:'PXQ2', 255:'PXQ3', 256:'PXQ6'}
# A tier may need MORE THAN ONE key prefix. PXQ4HQ needs both: pxa.pxq6.* is what this
# engine's own loader warns about when it is absent, and pxa.pxq4hq.* is what tells an external
# reader which of the two sub tables it is looking at when the file is mixed.
NEED = {'PXQ2':('pxa.pxq2',), 'PXQ3':('pxa.pxq3',),
        'PXQ4':('pxa.pxq6',), 'PXQ4HQ':('pxa.pxq6', 'pxa.pxq4hq'), 'PXQ6':('pxa.pxq6',)}

class Gguf:
    def __init__(self, path):
        self.path = path
        f = self.f = open(path, 'rb')
        if f.read(4) != b'GGUF':
            raise SystemExit(f'{path}: not a GGUF')
        self.version, n_t, n_kv = struct.unpack('<IQQ', f.read(20))
        self.kv = []                      # [(key, type, raw_value_bytes)]
        for _ in range(n_kv):
            k = self._str()
            (t,) = struct.unpack('<I', f.read(4))
            start = f.tell(); self._skip(t)
            end = f.tell(); f.seek(start)
            self.kv.append((k, t, f.read(end - start)))
        self.tensors = []                 # [(name, dims, type, offset, raw_info_bytes)]
        for _ in range(n_t):
            start = f.tell()
            name = self._str()
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
            (ty,) = struct.unpack('<I', f.read(4))
            (off,) = struct.unpack('<Q', f.read(8))
            end = f.tell(); f.seek(start)
            self.tensors.append((name, dims, ty, off, f.read(end - start)))
        self.align = 32
        for k, t, v in self.kv:
            if k == 'general.alignment' and t == T_U32:
                self.align = struct.unpack('<I', v)[0]
        pad = -f.tell() % self.align
        f.seek(pad, 1)
        self.data_start = f.tell()
        self.keys = {k for k, _, _ in self.kv}

    def _str(self):
        (n,) = struct.unpack('<Q', self.f.read(8))
        return self.f.read(n).decode('utf-8', 'replace')

    def _skip(self, t):
        f = self.f
        if t == T_STR:
            (n,) = struct.unpack('<Q', f.read(8)); f.seek(n, 1)
        elif t == T_ARR:
            (et,) = struct.unpack('<I', f.read(4)); (n,) = struct.unpack('<Q', f.read(8))
            if et == T_STR:
                for _ in range(n):
                    (m,) = struct.unpack('<Q', f.read(8)); f.seek(m, 1)
            elif et == T_ARR:
                for _ in range(n): self._skip(T_ARR)
            else:
                f.seek(VSZ[et] * n, 1)
        else:
            f.seek(VSZ[t], 1)

    def val_u32(self, key):
        for k, t, v in self.kv:
            if k == key and t == T_U32:
                return struct.unpack('<I', v)[0]
        return None

    def tiers(self):
        return sorted({GGML_PXQ[t] for _, _, t, _, _ in self.tensors if t in GGML_PXQ} & set(NEED))

    def tensor_bytes_span(self, i):
        """(absolute start, length) of tensor i's data. Length comes from the NEXT tensor's
        offset (or EOF), which is exact without needing every type's row arithmetic."""
        offs = sorted(t[3] for t in self.tensors)
        off = self.tensors[i][3]
        nxt = min([o for o in offs if o > off], default=None)
        end = self.data_start + nxt if nxt is not None else os.path.getsize(self.path)
        return self.data_start + off, end - (self.data_start + off)

def enc_str(s):
    b = s.encode('utf-8'); return struct.pack('<Q', len(b)) + b

def enc_f32_arr(vals):
    return struct.pack('<IQ', T_F32, len(vals)) + b''.join(struct.pack('<f', v) for v in vals)

# ------------------------------------------------------------------------------------------
def plan(g):
    """-> (missing_prefixes, refusals). A prefix is missing when the file HAS tensors of a tier
    that reads it and does NOT already carry its book."""
    missing, refuse = [], []
    for tier in g.tiers():
        for pre in NEED[tier]:
            if pre + '.book' in g.keys and pre + '.sub' in g.keys:
                continue
            # a non-default book cannot be reconstructed; refuse rather than guess
            vkey = pre + '.version'
            ver = g.val_u32(vkey)
            if ver is not None and ver != 1:
                refuse.append(f'{tier}: {vkey} = {ver} (a non-default book) and no {pre}.book in the '
                              f'file — the table it was built with is unknowable. Re-quantize.')
            elif pre not in missing:
                missing.append(pre)
    return missing, refuse

def do_check(g):
    print(f'{g.path}')
    print(f'  PXQ tiers present : {", ".join(g.tiers()) or "(none)"}')
    print(f'  codebook KVs      : {", ".join(sorted(k for k in g.keys if ".book" in k or ".sub" in k)) or "(none)"}')
    missing, refuse = plan(g)
    for r in refuse:
        print('  REFUSE ' + r)
    if missing:
        print(f'  MISSING           : {", ".join(p + ".{book,sub}" for p in missing)}')
    elif not refuse:
        print('  OK — every tier in this file carries its codebook')
    return 2 if refuse else (1 if missing else 0)

def do_stamp(g, out):
    missing, refuse = plan(g)
    if refuse:
        for r in refuse:
            sys.stderr.write('REFUSE ' + r + '\n')
        return 2
    if not missing:
        print('nothing to add; not writing an output')
        return 0
    tables = default_tables()
    add = []
    for pre in missing:
        book, sub = tables[pre]
        add.append((pre + '.book', enc_f32_arr(book)))
        add.append((pre + '.sub',  enc_f32_arr(sub)))
    add.append(('pxa.pxq.books_restamped',
                struct.pack('<I', T_STR)[0:0] + enc_str('pxq-stamp-books.py: default tables, '
                                                        'tensor bytes copied verbatim')))
    # header
    hdr = bytearray()
    hdr += b'GGUF' + struct.pack('<IQQ', g.version, len(g.tensors), len(g.kv) + len(add))
    for k, t, v in g.kv:
        hdr += enc_str(k) + struct.pack('<I', t) + v
    for k, v in add[:-1]:
        hdr += enc_str(k) + struct.pack('<I', T_ARR) + v
    k, v = add[-1]
    hdr += enc_str(k) + struct.pack('<I', T_STR) + v
    for name, dims, ty, off, raw in g.tensors:
        hdr += raw                       # offsets are relative to data_start: unchanged
    hdr += b'\x00' * (-len(hdr) % g.align)
    with open(out, 'wb') as o:
        o.write(hdr)
        g.f.seek(g.data_start)
        while True:
            chunk = g.f.read(1 << 22)
            if not chunk: break
            o.write(chunk)
    print(f'wrote {out}: +{len(add)} KVs ({", ".join(k for k, _ in add)})')
    return 0

def do_verify(a, b):
    ga, gb = Gguf(a), Gguf(b)
    if len(ga.tensors) != len(gb.tensors):
        print('FAIL tensor count differs'); return 1
    bad = 0
    for i, (ta, tb) in enumerate(zip(ga.tensors, gb.tensors)):
        if ta[0] != tb[0] or ta[1] != tb[1] or ta[2] != tb[2]:
            print(f'FAIL tensor {i} descriptor differs'); bad += 1; continue
        (oa, la), (ob, lb) = ga.tensor_bytes_span(i), gb.tensor_bytes_span(i)
        if la != lb:
            print(f'FAIL {ta[0]}: length {la} != {lb}'); bad += 1; continue
        ha, hb = hashlib.sha256(), hashlib.sha256()
        ga.f.seek(oa); gb.f.seek(ob)
        left = la
        while left:
            n = min(left, 1 << 22)
            ha.update(ga.f.read(n)); hb.update(gb.f.read(n)); left -= n
        if ha.digest() != hb.digest():
            print(f'FAIL {ta[0]}: sha256 differs'); bad += 1
    print(f'{len(ga.tensors)} tensors, {bad} mismatch(es) — '
          + ('BYTE-IDENTICAL' if bad == 0 else 'NOT IDENTICAL'))
    print(f'  new KVs: {sorted(gb.keys - ga.keys)}')
    return 1 if bad else 0

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inp'); ap.add_argument('out', nargs='?')
    ap.add_argument('--check',  action='store_true', help='report only, write nothing')
    ap.add_argument('--verify', action='store_true', help='per-tensor sha256 of inp vs out')
    a = ap.parse_args()
    if a.verify:
        if not a.out: ap.error('--verify needs both files')
        return do_verify(a.inp, a.out)
    g = Gguf(a.inp)
    if a.check or not a.out:
        return do_check(g)
    return do_stamp(g, a.out)

if __name__ == '__main__':
    sys.exit(main())
