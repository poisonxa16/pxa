#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Write the release copy of a registry source: every evidence string (an internal record id) becomes "".

  scripts/pxa-release-scrub.py IN.cpp OUT.cpp

Every build (common/CMakeLists.txt; a release build included) compiles OUT instead of IN, so no binary carries
record ids; only a dev build configured with -DPXA_REGISTRY_EVIDENCE=ON compiles IN. Only string literals change,
and only these two shapes (an evidence id written any other way, e.g. a ternary, is not emptied: write it as one
plain assignment per branch; tests/test-pxa-release-strings.py finds the ids it reads here in the binaries):
  <x>.evidence = "<id>"                       (any number of chained assignments)
  costs.push_back({ <what>, <evidence>, ... })  (the second element of a cost row, when it is not itself a {...})
Everything else is copied byte for byte. Exit 1 if IN cannot be read or OUT cannot be written.
"""
import os
import re
import sys

LIT = re.compile(r'"(?:[^"\\\n]|\\.)*"')
EVID = re.compile(r'(\bevidence\s*=\s*)"(?:[^"\\\n]|\\.)*"')
COST = re.compile(r'\bcosts\.push_back\(\s*\{')


def second_element_span(s, start):
    """[a, b) of the second top-level element of the brace list that opens just before `start`."""
    depth, i, commas, a = 0, start, 0, None
    while i < len(s):
        c = s[i]
        if c == '"':
            m = LIT.match(s, i)
            i = m.end() if m else i + 1
            continue
        if c in "({[":
            depth += 1
        elif c in ")}]":
            if depth == 0:
                return (a, i) if a is not None else None
            depth -= 1
        elif c == "," and depth == 0:
            commas += 1
            if commas == 1:
                a = i + 1
            elif commas == 2:
                return a, i
        i += 1
    return None


def scrub(s):
    s, n = EVID.subn(lambda m: m.group(1) + '""', s)
    out, last, k = [], 0, 0
    for m in COST.finditer(s):
        sp = second_element_span(s, m.end())
        if not sp or sp[0] < last:
            continue
        a, b = sp
        if "{" in s[a:b]:                 # a nested initializer (a JSON pair, not a cost row): left alone
            continue
        seg, c = LIT.subn('""', s[a:b])
        out.append(s[last:a]); out.append(seg); last = b; k += c
    out.append(s[last:])
    return "".join(out), n, k


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    try:
        src = open(argv[1], encoding="utf-8").read()
    except OSError as e:
        sys.stderr.write("pxa-release-scrub: %s\n" % e)
        return 1
    out, n, k = scrub(src)
    os.makedirs(os.path.dirname(os.path.abspath(argv[2])), exist_ok=True)
    tmp = argv[2] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(out)
    os.replace(tmp, argv[2])
    print("pxa-release-scrub: %s: %d evidence assignment(s), %d cost evidence literal(s) emptied"
          % (os.path.basename(argv[1]), n, k))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
