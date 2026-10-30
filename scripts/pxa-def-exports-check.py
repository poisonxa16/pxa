#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Windows export check for ggml-pxqn.dll: every name ggml/src/win-compat/ggml-pxqn.def exports must be DEFINED in a source
file the MSVC build compiles (ggml/src/CMakeLists.txt: the _pxqn_srcs list minus its MSVC list(REMOVE_ITEM ...)). A name
defined only in a Linux-only file links on Linux and fails on Windows with LNK2001 / LNK1120.

  scripts/pxa-def-exports-check.py [repo-root]     exit 0 = every export has a definition, 1 = list of the missing ones
"""
import os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = os.path.join(root, "ggml", "src")
cm = open(os.path.join(src, "CMakeLists.txt"), encoding="utf-8").read()
deff = os.path.join(src, "win-compat", "ggml-pxqn.def")
if not os.path.exists(deff):
    print("pxa-def-exports-check: no %s (public tree): nothing to check" % deff); sys.exit(0)

m = re.search(r"set\(_pxqn_srcs\s+([^)]*)\)", cm)
if not m: print("pxa-def-exports-check: no set(_pxqn_srcs ...) in ggml/src/CMakeLists.txt"); sys.exit(1)
srcs = m.group(1).split()
for r in re.finditer(r"if\s*\(\s*MSVC\s*\)(.*?)endif\(\)", cm, re.S):
    for rm in re.finditer(r"list\(REMOVE_ITEM\s+_pxqn_srcs\s+([^)]*)\)", r.group(1)):
        srcs = [s for s in srcs if s not in rm.group(1).split()]
# the CUDA half of the dll (GGML_SOURCES_PXQN_CUDA) builds on Windows too
cuda = []
for d, _, fs in os.walk(os.path.join(src, "ggml-cuda", "pxa")):
    cuda += [os.path.relpath(os.path.join(d, f), src) for f in fs if f.endswith(".cu")]
text = ""
for s in srcs + cuda:
    p = os.path.join(src, s)
    if os.path.exists(p): text += open(p, encoding="utf-8", errors="replace").read() + "\n"

names = []
for line in open(deff, encoding="utf-8"):
    line = line.split(";")[0].strip()
    if not line or line.upper().startswith(("LIBRARY", "EXPORTS")): continue
    names.append(line.split()[0])

def defined(n):
    # a definition: the name at file scope followed by a parameter list and a body (not a call, not a prototype)
    return re.search(r"(?m)^[A-Za-z_][^;{}\n]*?\b%s\s*\([^;{}]*\)\s*(?:noexcept\s*)?\{" % re.escape(n), text) is not None

missing = [n for n in names if not defined(n)]
if missing:
    print("pxa-def-exports-check: %d export(s) in ggml-pxqn.def have no definition in the MSVC sources (%s):"
          % (len(missing), " ".join(srcs)))
    for n in missing: print("   " + n)
    sys.exit(1)
print("pxa-def-exports-check: %d exports, all defined in the MSVC build's sources" % len(names))
