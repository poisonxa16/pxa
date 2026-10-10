#!/usr/bin/env python3
"""Refuse to ship a package whose own python files import a module the package does not carry.

    python3 scripts/pxa-staged-imports.py <stage-dir> --tree <export-tree>
    python3 scripts/pxa-staged-imports.py --image <docker/Dockerfile> --tree <export-tree>

Same shape as the dead-link and leak guards in scripts/make-release-tarball.sh: they read what is
about to ship, not what was asked for.  A release is assembled from explicit lists
(scripts/make-release-tarball.sh) and explicit COPY lines (docker/Dockerfile), and BOTH skip a
file that was never named -- in silence.  The package builds, and the tool breaks at its call.

There are TWO packages here and they fail independently: the tarball, and the container image.
The image ships its own copy of PXA Control under /usr/local/bin.

Found this way on 2026-10-09: tools/pxa_expert_map.py, imported UNGUARDED by the Expert map routes
in tools/pxa_control.py (r_expert_map / r_expert_map_rebuild / r_expert_map_reset), is named in
neither list, so the Expert map card -- a headline v3.1 feature with measured numbers in the
release notes -- answers every request with ModuleNotFoundError.

Guarded vs unguarded: tools/pxa_control.py wraps its optional tabs in try/except and says why
("without it, that one panel is lost").  An absent module there costs a feature, so it is a NOTE.
An absent module behind an unguarded import raises at the call site, so it is a FAILURE.

Exit 0 pass, 1 failure, 4 refused (the thing named is not a package, so a pass would mean nothing).
"""
import argparse
import ast
import os
import re
import sys

# Names that are not ours and never ship: the stdlib is handled by sys.stdlib_module_names; these
# are the third-party/optional imports the tools make on purpose.  vllm_pxq4 is the vLLM sidecar
# plugin a user installs themselves (tools/vllm-pxq4/) and pxa-launch.py probes it in a try.
EXTERNAL = {
    "vllm_pxq4", "jinja2", "numpy", "yaml", "requests", "cryptography", "nacl", "PIL",
    "safetensors", "torch", "transformers", "tqdm", "flask", "aiohttp", "websockets",
    "huggingface_hub", "sentencepiece", "google", "protobuf",
}
# The check's own subjects: one per area it claims to cover.  Missing here = the check is not
# covering what it says (a refactor that moves one out must fail here, not pass).  The image is a
# RUNTIME image -- it has no install folder, so no pxa-update and no pxa_lib_update.py; its copy of
# PXA Control guards that panel ("without it, that one panel is lost"), which is why the image list
# is the four the image really must run.
SUBJECTS_STAGE = [
    "pxa-launch.py",      # the launcher
    "pxa_control.py",     # PXA Control
    "pxa_encode.py",      # the encoder front end
    "pxa_encode_pkg.py",  # the encoder's package/verifier
    "pxa_lib_update.py",  # the updater (pxa-update lib)
]
SUBJECTS_IMAGE = [
    "pxa-launch.py",
    "pxa_control.py",
    "pxa_encode.py",
    "pxa_encode_pkg.py",
]


def is_stdlib(name):
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return name in names
    return name in {"os", "sys", "re", "json", "time", "math", "argparse", "typing", "pathlib"}


def imports_of(path):
    """(unguarded, guarded): top-level module names this file imports.

    Guarded = inside a try whose handlers could catch an ImportError (bare except, Exception,
    BaseException, ImportError, ModuleNotFoundError).
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except (SyntaxError, OSError):
        return set(), set()
    CATCHES_IMPORT = {"", "Exception", "BaseException", "ImportError", "ModuleNotFoundError"}
    guarded, plain = set(), set()

    def names_of(node):
        out = set()
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
        return out

    def catches_import(try_node):
        for h in try_node.handlers:
            if h.type is None:
                return True
            for e in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]):
                if isinstance(e, ast.Name) and e.id in CATCHES_IMPORT:
                    return True
        return False

    def visit(node, in_guard):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                (guarded if in_guard else plain).update(names_of(child))
                continue
            visit(child, in_guard or (isinstance(child, ast.Try) and catches_import(child)))

    visit(tree, False)
    return plain, guarded


def tree_modules(tree):
    """module name -> the file the tree keeps it in (tools/ and gguf-py/, one level of packages)."""
    out = {}
    for base in ("tools", "gguf-py"):
        d = os.path.join(tree, base)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith(".py"):
                out.setdefault(f[:-3], p)
            elif os.path.isdir(p) and os.path.exists(os.path.join(p, "__init__.py")):
                out.setdefault(f, os.path.join(p, "__init__.py"))
    return out


def view_stage(stage):
    """module name -> path, for the python files a staged directory carries."""
    out = {}
    for dp, _dn, fn in os.walk(stage):
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            out[f[:-3] if f != "__init__.py" else os.path.basename(dp)] = p
    return out


def view_image(dockerfile, tree):
    """module name -> path, for the tools/ python files the image COPYs (flat /usr/local/bin)."""
    text = open(dockerfile, encoding="utf-8", errors="replace").read()
    out = {}
    for line in text.splitlines():
        if not line.startswith("COPY ") or "tools/" not in line:
            continue
        for tok in line.split()[1:-1]:
            if not tok.startswith("tools/"):
                continue
            src = os.path.join(tree, tok)
            if tok.endswith(".py"):
                out[os.path.basename(tok)[:-3]] = src
            elif os.path.isdir(src):                       # a package copied whole
                out[os.path.basename(tok)] = os.path.join(src, "__init__.py")
                for f in sorted(os.listdir(src)):
                    if f.endswith(".py"):
                        out[f[:-3]] = os.path.join(src, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", help="a staged package directory")
    ap.add_argument("--image", help="a docker/Dockerfile, to check the image's own COPY list")
    ap.add_argument("--tree", required=True, help="the export tree the package was cut from")
    a = ap.parse_args()
    tree = a.tree
    if a.image:
        view, what = view_image(a.image, tree), "the container image (docker/Dockerfile)"
        subjects = SUBJECTS_IMAGE
        if not view:
            print("REFUSED: %s copies no tools/ python at all -- wrong file?" % a.image)
            return 4
    elif a.stage:
        view, what = view_stage(a.stage), "the staged package (%s)" % a.stage
        subjects = SUBJECTS_STAGE
        if len(view) < 20:
            print("REFUSED: %s carries only %d python files -- not a package (a pass would mean "
                  "nothing)" % (a.stage, len(view)))
            return 4
    else:
        print("REFUSED: name a stage directory or --image")
        return 4
    absent = [s for s in subjects if s[:-3] not in view]
    if absent:
        print("REFUSED: %s does not carry the check's own subjects, so it cannot judge them:" % what)
        for s in absent:
            print("  tools/%s" % s)
        return 4
    known = tree_modules(tree)
    bad, notes = [], []
    for name, path in sorted(view.items()):
        plain, guarded = imports_of(path)
        for imp in sorted(plain | guarded):
            if imp.startswith("_") or is_stdlib(imp) or imp in EXTERNAL:
                continue
            if imp in known and imp not in view:
                (bad if imp in plain else notes).append((os.path.relpath(path, tree), imp))
    for f, imp in sorted(set(notes)):
        print("note: %s imports %s inside a try -- the package omits it and loses that feature, "
              "not the program" % (f, imp))
    if bad:
        print("FAIL: %s carries a file that imports a module the package does not carry:" % what)
        for f, imp in sorted(set(bad)):
            print("  %s imports %s -- the tree keeps it at %s; name it in the tool lists in "
                  "scripts/make-release-tarball.sh AND on the docker/Dockerfile COPY line"
                  % (f, imp, os.path.relpath(known[imp], tree)))
        return 1
    print("PASS: %s -- %d python files, every unguarded import the tree can satisfy is carried"
          % (what, len(view)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
