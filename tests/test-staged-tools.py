#!/usr/bin/env python3
"""Two silent-omission checks over the release's explicit file lists (no build, no GPU):

  1. IMPORTS.  Every module a shipped python file imports must itself be staged, across the
     launcher (tools/pxa-launch.py), PXA Control (tools/pxa_control.py), the encoder front end
     (tools/pxa_encode*.py), the updater (tools/pxa_lib_update.py) and the three tool packages
     (tools/pxa_ctl, tools/pxa_chat, tools/pxa_control_ui).

  2. DOC LINKS.  Every relative link target a shipped .md names, that exists in the tree, must
     itself be staged -- the class that produced 25 dead links in the rc3 package, where the
     README named fourteen relative targets and the packer staged six.

Both defects have the same shape: a package is assembled from explicit lists
(scripts/make-release-tarball.sh) and explicit COPY lines (docker/Dockerfile), and both skip a
file that was never named -- in silence.  Found this way on 2026-10-09: tools/pxa_expert_map.py,
imported by the Expert map routes in tools/pxa_control.py, is named in neither list, so the Expert
map card would answer every request with ModuleNotFoundError.

The packer also refuses to TAR a package with a dead link (its own guard, reading the stage).
This test is the earlier, static form: it tells a contributor before a build, and it reads the
LISTS, so a name added to one list and not the other fails here.

    python3 tests/test-staged-tools.py [--root DIR]
"""
import argparse
import os
import re
import sys

# The check's own subjects.  Each has to be in the staged set, or the check is not covering the
# area it claims to (a refactor that moves one out of the package must fail here, not pass).
SUBJECTS = {
    "tools/pxa-launch.py": "the launcher",
    "tools/pxa_control.py": "PXA Control",
    "tools/pxa_encode.py": "the encoder front end",
    "tools/pxa_encode_pkg.py": "the encoder's package/verifier",
    "tools/pxa_lib_update.py": "the updater (pxa-update lib)",
}
# Third-party / optional imports that no package carries.  vllm_pxq4 is the vLLM sidecar plugin a
# user installs themselves (tools/vllm-pxq4/) and pxa-launch.py probes it inside a try/except.
EXTERNAL = {"vllm_pxq4", "jinja2", "numpy", "yaml", "requests", "cryptography", "nacl", "PIL",
            "safetensors", "torch", "transformers", "tqdm", "flask", "aiohttp", "websockets"}


def packer_text(root):
    p = os.path.join(root, "scripts", "make-release-tarball.sh")
    return open(p, encoding="utf-8", errors="replace").read() if os.path.isfile(p) else ""


def staged_from_packer(text):
    """repo-relative paths the packer stages: explicit copies, its `for f in` loops, keep()/keep_doc()"""
    names = set()
    for m in re.finditer(r'cp -aL?\s+"\$EXPORT/([A-Za-z0-9_./-]+)"', text):
        names.add(m.group(1))
    for m in re.finditer(r'\$EXPORT/tools/\$f', text):
        # a `for f in a b c; do cp -a "$EXPORT/tools/$f" ...` loop: take the words of that loop
        start = text.rfind("for f in", 0, m.start())
        end = text.find("done", m.start())
        if start != -1 and end != -1:
            words = text[start:end].split("\n")[0].replace("for f in", "").strip().split()
            names.update("tools/" + w for w in words)
    for m in re.finditer(r'^\s*(?:keep|keep_doc)\s+([A-Za-z0-9_./-]+)', text, re.M):
        names.add(m.group(1))
    if "$RELNOTES" in text:
        # the release notes are staged through a variable ($RELNOTES, $(basename "$RELNOTES")), so
        # no literal name appears in any cp line: every RELEASE-NOTES-*.md at the tree root is a
        # candidate the picker may name, and all of them are staged at the root or in docs/.
        names.add("RELEASE-NOTES-$VARYING.md")
    return names


def staged_from_dockerfile(text):
    names = set()
    for line in text.splitlines():
        if line.startswith("COPY ") and "tools/" in line:
            for tok in line.split()[1:-1]:
                if tok.startswith("tools/"):
                    names.add(tok)
    return names


def delinked_targets(packer):
    """Link targets the packer rewrites or de-links inside the stage (an artifact it withholds on
    purpose, e.g. bench/gate/LAST-RUN.md).  Read out of the packer's own sed lines, so a new
    deliberate withholding is picked up the moment its sed line lands."""
    out = set()
    for m in re.finditer(r's#\\?\]\\?\(([^)]+)\\?\)#', packer):
        out.add(m.group(1).replace("\\.", "."))
    return out


def tools_modules(root):
    """module name -> path of the sibling module or package, under tools/ and gguf-py/"""
    found = {}
    for base in ("tools", "gguf-py"):
        d = os.path.join(root, base)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith(".py"):
                found.setdefault(f[:-3], "%s/%s" % (base, f))
            elif os.path.isdir(p) and os.path.exists(os.path.join(p, "__init__.py")):
                found.setdefault(f, "%s/%s" % (base, f))
    return found


def imports_in(path):
    """top-level module names imported anywhere in the file (import a.b / from a.b import c)"""
    text = open(path, encoding="utf-8", errors="replace").read()
    names = set()
    for m in re.finditer(r'^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)', text, re.M):
        names.add(m.group(1).split(".")[0])
    return names


def doc_links(path):
    """relative link/image targets a markdown file names"""
    text = open(path, encoding="utf-8", errors="replace").read()
    out = set()
    for m in re.finditer(r'\[[^\]]*\]\(([^)\s]+)\)|<img[^>]*\ssrc="([^"]+)"', text):
        t = (m.group(1) or m.group(2) or "").split("#")[0].strip()
        if t and "://" not in t and not t.startswith(("mailto:", "/", "#")):
            out.add(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    root = a.root
    if not os.path.isdir(os.path.join(root, "tools")):
        print("SKIP: no tools/ under", root)
        return 0
    packer = packer_text(root)
    dockerfile = os.path.join(root, "docker", "Dockerfile")
    staged = set()
    if packer:
        staged |= staged_from_packer(packer)
    if os.path.isfile(dockerfile):
        staged |= staged_from_dockerfile(open(dockerfile, encoding="utf-8", errors="replace").read())
    fails = []

    absent_subjects = [p for p in SUBJECTS if p not in staged]
    if absent_subjects:
        for p in absent_subjects:
            fails.append("the check no longer covers %s: %s is not staged by any list" % (SUBJECTS[p], p))

    def is_staged(target):
        """Is this repo-relative path carried into the package by some staging line?"""
        if "RELEASE-NOTES-$VARYING.md" in staged and re.match(r'^RELEASE-NOTES-.*\.md$', target):
            return True
        return target in staged or any(s.startswith(target) for s in staged)

    # ---- 1. imports -------------------------------------------------------------------------
    tools = os.path.join(root, "tools")
    known = tools_modules(root)
    have = set(known)  # a module is carried if its file/package is staged
    carried = set()
    for mod, rel in known.items():
        if is_staged(rel):
            carried.add(mod)
    # package dirs staged whole (tools/pxa_ctl/*.py, tools/pxa_chat, tools/pxa_control_ui)
    for pkg in ("pxa_ctl", "pxa_chat", "pxa_control_ui"):
        if any(s.startswith("tools/" + pkg) for s in staged):
            carried.add(pkg)
    n_tools = 0
    for rel in sorted(staged):
        if not rel.endswith(".py"):
            continue
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continues = None
            fails.append("staging names %s but the tree has no such file" % rel)
            continue
        n_tools += 1
        for name in sorted(imports_in(p)):
            if name.startswith("_") or name in EXTERNAL:
                continue
            if name in known and name not in carried:
                fails.append("%s imports %s -- in the tree at %s, staged by neither list"
                             % (rel, name, known[name]))
    # files under a staged package dir count as staged too
    for pkg in ("pxa_ctl", "pxa_chat", "pxa_control_ui"):
        d = os.path.join(tools, pkg)
        if not os.path.isdir(d) or pkg not in carried:
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                n_tools += 1
                for name in sorted(imports_in(os.path.join(d, f))):
                    if name.startswith("_") or name in EXTERNAL:
                        continue
                    if name in known and name not in carried and name != pkg:
                        fails.append("tools/%s/%s imports %s -- in the tree at %s, staged by neither list"
                                     % (pkg, f, name, known[name]))

    # ---- 2. doc links -----------------------------------------------------------------------
    skip = delinked_targets(packer)
    n_docs = n_links = 0
    for rel in sorted(staged):
        if not rel.endswith(".md"):
            continue
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        n_docs += 1
        d = os.path.dirname(rel)
        for t in sorted(doc_links(p)):
            if t in skip:
                continue                       # the packer de-links this one inside the stage
            target = os.path.normpath(os.path.join(d, t))
            if os.path.isabs(target) or target.startswith(".."):
                continue                       # outside the package: not this check's business
            n_links += 1
            if os.path.exists(os.path.join(root, target)) and not is_staged(target):
                fails.append("%s links %s -- present in the tree, staged by neither list" % (rel, t))

    print("staged: %d entries; %d python files, %d docs, %d relative links checked"
          % (len(staged), n_tools, n_docs, n_links))
    if fails:
        print("FAIL: %d finding(s)" % len(fails))
        for f in sorted(set(fails)):
            print("  " + f)
        return 1
    print("PASS: every sibling module and every tree-resolvable doc link a shipped file names is staged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
