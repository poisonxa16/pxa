#!/usr/bin/env python3
"""Regression check: gguf-py reads PXQN/PXA4 GGUFs (ggml types 257-264) with the right per-tensor byte size.

These types carry a 2 B/row fp16 anchor header (128 B per 64-row panel, row_meta_size 2 in ggml.c) that the
(block, type_size) table cannot express; the reader adds it in gguf_reader.py (_pxa_rm) and quants.py (_PXA_ROW_META).

  python3 -I gguf-py/tests/test_pxqn_reader.py [FILE.gguf ...]     exit 0 = pass, 1 = fail

Per file: (1) the reader opens it; (2) every tensor's byte span (next offset - offset, last: EOF - offset) equals
align_up(n_bytes); (3) the last tensor ends at EOF (up to alignment padding); (4) every PXQN tensor's n_bytes equals
(rows//64) * (128 + (K//qk) * slab) * n_expert (slab/qk from ggml-pxqn.h), independent of the reader's table."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gguf  # noqa: E402

_one = os.environ.get("GGUF_READER_FILE", "")
DEFAULT_FILES = [_one] if _one else []
# type -> (qk, slab bytes per 64-row panel per K//qk slab), ggml-pxqn.h *_QK / *_SLAB_BYTES
QK_SLAB = {257: (128, 3328), 258: (128, 3584), 259: (32, 1088), 260: (128, 2304), 261: (128, 1280), 262: (32, 1152),
           263: (128, 5376), 264: (128, 81 * 64)}


def align_up(n, a):
    return (n + a - 1) // a * a


def check(path):
    fails = []
    r = gguf.GGUFReader(path)
    a = int(r.alignment)
    size = os.path.getsize(path)
    ts = sorted(r.tensors, key=lambda t: int(t.data_offset))
    n_pxqn = 0
    for k, t in enumerate(ts):
        off = int(t.data_offset)
        end = int(ts[k + 1].data_offset) if k + 1 < len(ts) else size
        nb = int(t.n_bytes)
        if end - off != align_up(nb, a):
            fails.append(f"{t.name}: type {int(t.tensor_type)} span {end - off} != align_up(n_bytes {nb})")
        tt = int(t.tensor_type)
        if tt in QK_SLAB:
            n_pxqn += 1
            qk, slab = QK_SLAB[tt]
            K, R = int(t.shape[0]), int(t.shape[1])
            ne = int(t.shape[2]) if len(t.shape) > 2 else 1
            want = (R // 64) * (128 + (K // qk) * slab) * ne
            if nb != want:
                fails.append(f"{t.name}: type {tt} n_bytes {nb} != panel formula {want}")
    last = ts[-1]
    if int(last.data_offset) + align_up(int(last.n_bytes), a) != size:
        fails.append(f"last tensor {last.name} ends at {int(last.data_offset) + int(last.n_bytes)}, file size {size}")
    print(f"{path}: {len(ts)} tensors, {n_pxqn} PXQN/PXA4, {len(fails)} failures")
    for f in fails[:10]:
        print("  FAIL", f)
    return not fails


if __name__ == "__main__":
    files = sys.argv[1:] or DEFAULT_FILES
    ok = True
    for p in files:
        try:
            ok = check(p) and ok
        except Exception as e:  # the pre-fix reader crashes while reshaping the truncated blob
            print(f"{p}: FAIL reader raised {type(e).__name__}: {e}")
            ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
