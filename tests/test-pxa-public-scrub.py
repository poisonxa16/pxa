#!/usr/bin/env python3
"""scripts/pxa-public-scrub.py on a copy of this tree's four files: every evidence id it owns is emptied, the rest of
each file is unchanged line for line, a second run changes nothing, and a git checkout is refused (exit 2).

    python3 tests/test-pxa-public-scrub.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRUB = os.path.join(ROOT, "scripts", "pxa-public-scrub.py")
FILES = ("common/pxa-registry.cpp", "common/pxa-lever-catalog.inc", "scripts/pxa-lever-catalog.py",
         "tests/test-pxa-autoconfig.cpp", "scripts/pxa-release-scrub.py", "scripts/pxa-public-scrub.py")
fails = 0


def check(ok, what):
    global fails
    print("%-5s %s" % ("ok" if ok else "FAIL", what))
    fails += 0 if ok else 1


def run(root):
    return subprocess.run([sys.executable, "-B", SCRUB, root], capture_output=True, text=True)


with tempfile.TemporaryDirectory(prefix="pxa-public-scrub-") as t:
    for rel in FILES:
        os.makedirs(os.path.dirname(os.path.join(t, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(t, rel))
    before = {rel: open(os.path.join(t, rel), encoding="utf-8").read() for rel in FILES}
    r = run(t)
    check(r.returncode == 0, "first pass exit 0 (%s)" % r.stdout.strip().replace("\n", "; "))
    after = {rel: open(os.path.join(t, rel), encoding="utf-8").read() for rel in FILES}
    for rel in FILES[:4]:
        check(before[rel].count("\n") == after[rel].count("\n"), "%s keeps its line count" % rel)
    inc = after["common/pxa-lever-catalog.inc"]
    left = [m for m in re.findall(r'^\{ "[^"]*", "(?:[^"\\]|\\.)*", "[^"]*", "[^"]*", "([^"]*)"', inc, re.M) if m]
    check(not left, "catalog .inc: no evidence id left (%d)" % len(left))
    check(not re.search(r'ev_is\([^,]+,\s*"[^"]+"\)', after["tests/test-pxa-autoconfig.cpp"]),
          "autoconfig test: every ev_is() id empty")
    check(not re.search(r'evidence\s*=\s*"[^"]+"', after["common/pxa-registry.cpp"]), "registry: no evidence literal left")
    r2 = run(t)
    again = {rel: open(os.path.join(t, rel), encoding="utf-8").read() for rel in FILES}
    check(r2.returncode == 0 and again == after, "second pass changes nothing")
    check(not os.path.exists(os.path.join(t, "scripts", "__pycache__")), "no __pycache__ written into the tree")
    os.makedirs(os.path.join(t, ".git"))
    r3 = run(t)
    check(r3.returncode == 2, "a git checkout is refused (exit %d)" % r3.returncode)

print("test-pxa-public-scrub: %s" % ("PASS" if not fails else "FAIL %d" % fails))
sys.exit(1 if fails else 0)
