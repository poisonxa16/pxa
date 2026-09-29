#!/usr/bin/env python3
"""verify-identity.py A B -- tensor-for-tensor sha256 of the data of two GGUFs (names, types, shapes, bytes)."""
import sys, hashlib
sys.argv, args = sys.argv[:1] + ['/dev/null', '/dev/null'], sys.argv[1:]
exec(open(__file__.replace('verify-identity.py', 'mix-merge.py')).read().split("ap = argparse")[0])
a, b = G(args[0]), G(args[1]); same = diff = 0
for x in a.t:
    y = b.byname.get(x[0])
    ok = y is not None and x[1] == y[1] and x[2] == y[2] and x[4] == y[4]
    if ok:
        a.f.seek(a.data + x[3]); b.f.seek(b.data + y[3])
        ok = hashlib.sha256(a.f.read(x[4])).digest() == hashlib.sha256(b.f.read(y[4])).digest()
    same += ok; diff += not ok
    if not ok: print('DIFF', x[0])
print(f'{same} identical / {diff} differing of {len(a.t)} tensors ->', 'TENSOR-FOR-TENSOR IDENTICAL' if diff == 0 and len(a.t) == len(b.t) else 'NOT IDENTICAL')
