#!/usr/bin/env python3
"""Checks _pxa_swa_split from swa-keep-hashed-block-pool.patch without vLLM."""
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
patch = open(os.path.join(HERE, "swa-keep-hashed-block-pool.patch")).read()
src = "\n".join(l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
m = re.search(
    r"def _pxa_swa_split\(freed, reuse_first, keep, caching\):\n(.*?)\n    return pre, post\n",
    src,
    re.S,
)
assert m, "helper not found in the patch"
code = "def _pxa_swa_split(freed, reuse_first, keep, caching):\n" + m.group(1) + "\n    return pre, post\n"
ns = {}
exec(code, ns)
split = ns["_pxa_swa_split"]


def B(hashed):
    return types.SimpleNamespace(block_hash=("h" if hashed else None))


def ids(blocks):
    return [b.block_hash for b in blocks]


H, U = B(True), B(False)
freed = [H, U, H]
bad = 0


def check(name, got, want):
    global bad
    ok = got == want
    bad += not ok
    print("ok  " if ok else "FAIL", name, got)


check("off keeps caller order in append", split(freed, True, False, True), ([], freed))
check("caching off keeps caller order", split(freed, False, True, False), ([], freed))
pre, post = split(freed, True, True, True)
check("on prepends only unhashed", ids(pre), [None])
check("on appends hashed in order", ids(post), ["h", "h"])
check("on ignores reuse_first for hashed", split(freed, True, True, True)[1][0] is H, True)
check("empty", split([], False, True, True), ([], []))
sys.exit(1 if bad else 0)
