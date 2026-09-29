#!/usr/bin/env python3
"""mix-merge.py -- build a mixed-tier GGUF by BYTE-COPYING whole tensors out of several GGUFs
of the SAME model that differ only in per-tensor quant types.

  mix-merge.py OUT BASE --take SRC=REGEX [--take SRC=REGEX ...] [--kv KEY=STR ...] [--dry]

BASE supplies the header KVs and every tensor no --take matches. For each tensor the LAST
matching --take (regex, re.search on the tensor name) names the source file. Tensor byte length =
next tensor's offset minus its own (the offsets partition the data block), so no quant-size
table is needed -- that is what makes it safe for PXQ tiers gguf-py cannot size.
Asserted before writing: same tensor names/order/shapes in every source, same alignment, same
PXQ book/version KVs (the engine decodes with compiled-in tables, but a mismatch would mean
the sources were cut with different tables). Output goes to OUT.partial, renamed on success.
Why: a tier-sensitivity arm then costs one sequential copy instead of a 10-minute quantize,
and the encoder is per-tensor deterministic, so the merged tensor IS the tensor the quantizer
would emit for that rule (verified by tensor sha against a direct --custom-q quantize).
"""
import sys, re, struct, os, argparse, hashlib

FIX = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

class G:
    def __init__(s, path):
        s.path = path; f = s.f = open(path, 'rb')
        assert f.read(4) == b'GGUF', path
        s.ver, = struct.unpack('<I', f.read(4)); assert s.ver == 3
        nt, nkv = struct.unpack('<QQ', f.read(16))
        s.kv = []  # (key, raw_bytes_of_type_and_value, parsed)
        for _ in range(nkv):
            k = s._str(); st = f.tell(); t, = struct.unpack('<I', f.read(4)); v = s._val(t)
            en = f.tell(); f.seek(st); raw = f.read(en - st); s.kv.append((k, raw, v))
        s.kvd = {k: v for k, _, v in s.kv}
        s.align = s.kvd.get('general.alignment', 32)
        s.t = []
        for _ in range(nt):
            n = s._str(); nd, = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd)); ty, off = struct.unpack('<IQ', f.read(12))
            s.t.append([n, dims, ty, off, 0])
        s.data = (f.tell() + s.align - 1) // s.align * s.align
        end = os.path.getsize(path) - s.data
        offs = sorted(x[3] for x in s.t) + [end]
        nxt = {offs[i]: offs[i + 1] for i in range(len(offs) - 1)}
        for x in s.t: x[4] = nxt[x[3]] - x[3]
        s.byname = {x[0]: x for x in s.t}
    def _str(s):
        n, = struct.unpack('<Q', s.f.read(8)); return s.f.read(n).decode('utf-8', 'replace')
    def _val(s, t):
        f = s.f
        if t in FIX: return f.read(FIX[t]) if t not in (4, 5, 10, 11) else struct.unpack({4:'<I',5:'<i',10:'<Q',11:'<q'}[t], f.read(FIX[t]))[0]
        if t == 8: return s._str()
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            if et in FIX: return f.read(FIX[et] * n)
            return [s._val(et) for _ in range(n)]
        raise ValueError(t)

def pstr(b): b = b.encode(); return struct.pack('<Q', len(b)) + b

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('base')
ap.add_argument('--take', action='append', default=[])
ap.add_argument('--kv', action='append', default=[])
ap.add_argument('--dry', action='store_true')
a = ap.parse_args()

base = G(a.base); srcs = {a.base: base}; rules = []
for tk in a.take:
    p, rx = tk.split('=', 1)
    if p not in srcs: srcs[p] = G(p)
    rules.append((re.compile(rx), p))
for p, g in srcs.items():
    assert [x[0] for x in g.t] == [x[0] for x in base.t], f'tensor list differs: {p}'
    assert all(x[1] == base.byname[x[0]][1] for x in g.t), f'shape differs: {p}'
    assert g.align == base.align, p
    for k in ('pxa.pxq3.book', 'pxa.pxq6.book', 'pxa.pxq2.book', 'pxa.pxq3.version', 'pxa.pxq6.version', 'pxa.pxq2.version'):
        if k in g.kvd and k in base.kvd: assert g.kvd[k] == base.kvd[k], f'{k} differs in {p}'

plan = []
for n, dims, ty, off, sz in base.t:
    src = a.base
    for rx, p in rules:
        if rx.search(n): src = p
    x = srcs[src].byname[n]; plan.append((n, dims, x[2], x[4], src, x[3]))

TN = {0:'f32',1:'f16',8:'q8_0',14:'q6_K',39:'mxfp4',252:'pxq4',253:'pxq4hq',254:'pxq2',255:'pxq3',248:'pxq1',256:'pxq6'}
from collections import Counter
cnt = Counter(); byt = Counter()
for n, d, ty, sz, src, o in plan: cnt[ty] += 1; byt[ty] += sz
tot = sum(byt.values()); embd = [p for p in plan if p[0] == 'token_embd.weight'][0][3]
print('census:', ', '.join(f'{TN.get(t,t)} n={cnt[t]} {byt[t]/1e9:.3f}GB' for t in sorted(byt, key=lambda t: -byt[t])))
print(f'tensor bytes {tot} (GPU-resident excl token_embd {tot-embd}, {(tot-embd)/2**20:.1f} MiB)')
if a.dry: sys.exit(0)

kvs = [(k, raw) for k, raw, v in base.kv]
over = dict(x.split('=', 1) for x in a.kv)
kvs = [(k, raw) for k, raw in kvs if k not in over] + [(k, struct.pack('<I', 8) + pstr(v)) for k, v in over.items()]
hdr = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(kvs))
for k, raw in kvs: hdr += pstr(k) + raw
off = 0; infos = b''
for n, d, ty, sz, src, o in plan:
    infos += pstr(n) + struct.pack('<I', len(d)) + struct.pack('<%dQ' % len(d), *d) + struct.pack('<IQ', ty, off)
    off += (sz + base.align - 1) // base.align * base.align
hdr += infos
pad = (-len(hdr)) % base.align
tmp = a.out + '.partial'
with open(tmp, 'wb') as w:
    w.write(hdr + b'\0' * pad); pos = 0
    for n, d, ty, sz, src, o in plan:
        g = srcs[src]; g.f.seek(g.data + o); rem = sz
        while rem:
            b = g.f.read(min(rem, 64 << 20)); assert b, n; w.write(b); rem -= len(b)
        p2 = (-sz) % base.align; w.write(b'\0' * p2)
os.rename(tmp, a.out)
print('wrote', a.out, os.path.getsize(a.out))
