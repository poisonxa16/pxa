#!/usr/bin/env python3
"""pxa-env-flag-sweep: list (and optionally rewrite) the `atoi(e) == 0` env-switch sites (bug #332 review).

The idiom `!(e && *e && atoi(e) == 0)` reads every NON-numeric value as 0, so PXA_X=on / =true turns a
default-on lever OFF. ggml/include/pxa-env-flag.h holds the one rule (0/off/false/no off, unset/empty
default, anything else on; numbers exactly as atoi). This script classifies every site:
  A  default-on switch, `atoi(e) == 0` means "explicitly off"   -> mechanical: atoi(e) == 0 -> pxa_env_value_off(e)
  B  default-off switch `!e || atoi(e) == 0` (opt-in; =on/=true read OFF) -> mechanical: -> !pxa_env_value_flag(e, 0)
  C  numeric / multi-valued / mixed: hand review, never rewritten by --apply
Usage:  scripts/pxa-env-flag-sweep.py [--md] [--apply]   (run at the repo root; --apply edits A and B sites in
place and adds #include "pxa-env-flag.h" where missing; numeric values keep their exact meaning)
"""
import re, subprocess, sys

PAT = r'atoi\(e\) *== *0'
ROOTS = ['src', 'common', 'ggml/src']
NAME_RE = re.compile(r'getenv\("([A-Z0-9_]+)"\)')

def sites():
    out = subprocess.run(['git', 'grep', '-nE', PAT, '--'] + ROOTS, capture_output=True, text=True).stdout
    for ln in out.splitlines():
        f, n, txt = ln.split(':', 2)
        yield f, int(n), txt

def env_name(f, n, txt):
    m = NAME_RE.findall(txt)
    if m:
        return m[-1]
    lines = open(f, encoding='utf-8', errors='replace').read().split('\n')
    for k in range(n - 2, max(0, n - 8), -1):
        m = NAME_RE.findall(lines[k])
        if m:
            return m[-1]
    return '?'

B_PATTERNS = [r'!e \|\| atoi\(e\) *== *0']          # default-off: `if (!e || atoi(e) == 0) return false;`
C_PATTERNS = [r'atoi\(e\) *[<>]']                     # multi-valued numbers (atoi(e) > 0 ? atoi(e) : ...)

def classify(txt):
    # every other `atoi(e) == 0` in the tree is "=0 is the off switch" of a default-on switch (checked by
    # hand on d75bf451d8: !(e && ...), (e && ...) ? 0 : 1, if (e && ...) return false / {off block},
    # labels "PXA_X=0", PXA_ENHANCE's level 1/2, PXA_NUMA_BIND's gate-default ternary)
    for p in B_PATTERNS:
        if re.search(p, txt):
            return 'B'
    for p in C_PATTERNS:
        if re.search(p, txt):
            return 'C'
    return 'A'

def rewrite(txt, cls):
    if cls == 'A':
        txt = re.sub(r'(std::)?atoi\(e\) *== *0', 'pxa_env_value_off(e)', txt)
        txt = re.sub(r'atoi\((r|m)\) *== *0', r'pxa_env_value_off(\1)', txt)
    elif cls == 'B':
        txt = re.sub(r'!e \|\| atoi\(e\) *== *0', '!pxa_env_value_flag(e, 0)', txt)
    return txt

def include_slot(lines):
    """index before the first #include outside any #if block (an include guard does not count)"""
    base, depth, guard = 0, 0, None
    first = [i for i, l in enumerate(lines) if l.lstrip().startswith('#')][:2]
    if len(first) == 2:
        a, b = lines[first[0]].split(), lines[first[1]].split()
        if a[:1] == ['#ifndef'] and b[:1] == ['#define'] and a[1:2] == b[1:2]:
            guard, base = first[0], 1
    for i, l in enumerate(lines):
        t = l.lstrip()
        if i == guard:
            depth += 1
            continue
        if re.match(r'#\s*if', t):
            depth += 1
        elif re.match(r'#\s*endif', t):
            depth -= 1
        elif t.startswith('#include') and depth == base:
            return i
    return 0

def main():
    md, apply = '--md' in sys.argv, '--apply' in sys.argv
    rows = [(f, n, env_name(f, n, t), classify(t), t.strip()) for f, n, t in sites()]
    if md:
        print('| # | class | file:line | switch | code |')
        print('|---|---|---|---|---|')
        for i, (f, n, nm, c, t) in enumerate(rows, 1):
            print(f'| {i} | {c} | {f}:{n} | {nm} | `{t[:110].replace("|", "\\|")}` |')
    else:
        for f, n, nm, c, t in rows:
            print(f'{c} {f}:{n} {nm}')
    cnt = {k: sum(1 for r in rows if r[3] == k) for k in 'ABC'}
    print(f'\ntotal {len(rows)}: A {cnt["A"]} (default-on, =0 off), B {cnt["B"]} (default-off, opt-in), C {cnt["C"]} (hand review)', file=sys.stderr)
    if apply:
        byfile = {}
        for f, n, nm, c, t in rows:
            if c in 'AB':
                byfile.setdefault(f, []).append((n, c))
        for f, items in byfile.items():
            lines = open(f, encoding='utf-8').read().split('\n')
            for n, c in items:
                lines[n - 1] = rewrite(lines[n - 1], c)
            src = '\n'.join(lines)
            if 'pxa-env-flag.h' not in src:
                lines.insert(include_slot(lines), '#include "pxa-env-flag.h"')
                src = '\n'.join(lines)
            open(f, 'w', encoding='utf-8').write(src)
        print(f'rewrote {sum(len(v) for v in byfile.values())} sites in {len(byfile)} files', file=sys.stderr)

if __name__ == '__main__':
    main()
