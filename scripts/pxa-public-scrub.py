#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Public source export: empty the internal evidence ids (ledger record ids) in an exported tree.

  scripts/pxa-public-scrub.py EXPORT_ROOT

The engine keeps, next to each measured default, the id of the ledger record that is its evidence. Those ids are
internal: no build ships them (common/CMakeLists.txt compiles the registry from a scrubbed copy, the embedded lever
catalog drops the evidence column), and the public source tree must not carry them either. This step makes the
exported SOURCE match what every build already compiles:

  common/pxa-registry.cpp          evidence literals emptied, the same rule scripts/pxa-release-scrub.py applies
  common/pxa-lever-catalog.inc     the evidence column (5th field of a row) emptied
  scripts/pxa-lever-catalog.py     the evidence field of every CURATED row emptied (the 4th tuple element), and the
                                   id the Overdrive loop appends; the regenerated table then matches the .inc above
  tests/test-pxa-autoconfig.cpp    ev_is(<pick>, "<id>") -> ev_is(<pick>, ""): ev_is() checks for an empty id in
                                   every build without -DPXA_REGISTRY_EVIDENCE=ON, which the public tree cannot set
                                   to anything but "" anyway

Only string literals in those exact shapes change; everything else is copied byte for byte, and a second run
changes nothing. It runs on an exported tree only (scripts/pxa-lever-catalog.py --public-filter calls it): in a git
checkout it refuses (exit 2), because emptying the ids there would lose them. Exit 1 when a file it must scrub is
missing or an id is left over afterwards.
"""
import ast
import importlib.util
import os
import re
import sys
sys.dont_write_bytecode = True   # nothing written into the export tree but the scrubbed files

HERE = os.path.dirname(os.path.abspath(__file__))

ROW = re.compile(r'^(\{ (?:"(?:[^"\\]|\\.)*", ){4})"(?:[^"\\]|\\.)*"(, "(?:[^"\\]|\\.)*" \},?\s*)$')
EV_IS = re.compile(r'(\bev_is\(\s*[^,()]*(?:\([^()]*\)[^,()]*)*,\s*)"(?:[^"\\\n]|\\.)*"(\s*\))')


def _release_scrub():
    spec = importlib.util.spec_from_file_location("pxa_release_scrub", os.path.join(HERE, "pxa-release-scrub.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.scrub


def scrub_inc(s):
    out, n = [], 0
    for l in s.splitlines(True):
        m = ROW.match(l)
        if m:
            l2 = m.group(1) + '""' + m.group(2)
            n += l2 != l
            l = l2
        out.append(l)
    return "".join(out), n


def _curated_evidence_nodes(tree):
    nodes = []

    def take(d):
        if isinstance(d, ast.Dict):
            for v in d.values:
                if isinstance(v, ast.Tuple) and len(v.elts) == 5 and isinstance(v.elts[3], ast.Constant) \
                        and isinstance(v.elts[3].value, str):
                    nodes.append(v.elts[3])
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CURATED" for t in n.targets):
            take(n.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update" \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "CURATED":
            for a in n.args:
                take(a)
        # CURATED[_n] = (..., (_ev + ", " if _ev else "") + "<id>", ...): the appended id
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript) \
                and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "CURATED" \
                and isinstance(n.value, ast.Tuple) and len(n.value.elts) == 5:
            ev = n.value.elts[3]
            if isinstance(ev, ast.BinOp) and isinstance(ev.right, ast.Constant) and isinstance(ev.right.value, str):
                nodes.append(ev.right)
    return nodes


def scrub_catalog_py(s):
    nodes = [c for c in _curated_evidence_nodes(ast.parse(s)) if c.value]
    starts = [0]
    for l in s.splitlines(True):
        starts.append(starts[-1] + len(l))
    b = s
    for c in sorted(nodes, key=lambda c: (c.lineno, c.col_offset), reverse=True):
        # ast offsets are UTF-8 byte columns; convert within the line
        line = s.splitlines(True)[c.lineno - 1]
        a = starts[c.lineno - 1] + len(line.encode("utf-8")[:c.col_offset].decode("utf-8"))
        eline = s.splitlines(True)[c.end_lineno - 1]
        e = starts[c.end_lineno - 1] + len(eline.encode("utf-8")[:c.end_col_offset].decode("utf-8"))
        b = b[:a] + '""' + b[e:]
    return b, len(nodes)


def scrub_ev_is(s):
    return EV_IS.subn(lambda m: m.group(1) + '""' + m.group(2), s)


def main(argv):
    if len(argv) != 2:
        sys.stderr.write(__doc__)
        return 2
    root = os.path.abspath(argv[1])
    if os.path.exists(os.path.join(root, ".git")):
        sys.stderr.write("pxa-public-scrub: %s is a git checkout: this step runs on an exported tree only\n" % root)
        return 2
    rel_scrub = _release_scrub()
    jobs = [("common/pxa-registry.cpp", lambda s: (lambda r: (r[0], r[1] + r[2]))(rel_scrub(s))),
            ("common/pxa-lever-catalog.inc", scrub_inc),
            ("scripts/pxa-lever-catalog.py", scrub_catalog_py),
            ("tests/test-pxa-autoconfig.cpp", scrub_ev_is)]
    bad = 0
    for rel, fn in jobs:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            print("pxa-public-scrub: missing %s" % rel)
            bad += 1
            continue
        s = open(p, encoding="utf-8").read()
        out, n = fn(s)
        if out != s:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(out)
            os.replace(tmp, p)
        again, n2 = fn(out)
        if again != out:
            print("pxa-public-scrub: %s not stable after one pass" % rel)
            bad += 1
        print("pxa-public-scrub: %s: %d id(s) emptied" % (rel, n if out != s else 0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
