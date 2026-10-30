#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""The clean-package scan rules over the binaries this build made (ctest test-pxa-release-strings).

    tests/test-pxa-release-strings.py --source-root SRC --build-dir BUILD [--release] [--evidence-ids]

What a package ships is scanned by scripts/release-scan.py before any upload. This test runs the same rules on the
built binaries every time the tests run, so a log, help or error string that names an internal record cannot land
unnoticed between two packages:
  1. in-repo rule (always): no evidence id of the registry (common/pxa-registry.cpp: the literals that
     scripts/pxa-release-scrub.py empties) and no evidence id of the lever catalog (common/pxa-lever-catalog.inc,
     the column the release filter empties) appears among the printable strings of a shipped ELF file;
  2. block list rules (when PXA_RELEASE_BLOCKLIST names the private block list, from the environment or the CMake
     cache): every binary-scope category of that list, the way release-scan.py applies it. A dev build carries its
     build paths (__FILE__), so the path categories (engine-dir, host-path) apply to a release build (--release,
     -DPXA_RELEASE_BUILD=ON) only.
--evidence-ids (-DPXA_REGISTRY_EVIDENCE=ON, a dev build that is never a release) compiles the registry ids in on
purpose: they are then masked instead of reported. The secret checks stay in release-scan.py (they need the
secrets directory): this test never reads a secret.
The shipped files mirror scripts/make-release-tarball.sh: bin/ llama-* executables plus the self-tests it lists,
and every lib*.so within three directory levels of the build directory. Exit 0 clean, 1 hits, 2 cannot run.
"""
import argparse
import importlib.util
import os
import re
import sys

LIT = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
ROW_FIELDS = re.compile(r'^\{ ((?:"(?:[^"\\]|\\.)*", ){5})"((?:[^"\\]|\\.)*)" \},?\s*$')
TOKEN_FULL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*[A-Za-z0-9]")
PATH_CATS = ("engine-dir", "host-path")
MIN_ID = 6          # shorter literals are not record ids


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def registry_ids(src_root, scrub):
    """Every literal the release scrub empties in the registry source (what a scrubbed build must not carry)."""
    s = open(os.path.join(src_root, "common", "pxa-registry.cpp"), encoding="utf-8").read()
    ids = set()
    # every literal of every statement that assigns an evidence field: also the shapes the scrub does not empty
    # (a ternary), which then show up here as a hit instead of shipping quietly
    for m in re.finditer(r'\bevidence\s*=(?!=)', s):
        end = s.find(";", m.end())
        ids.update(x.group(1) for x in LIT.finditer(s[m.end():end if end > 0 else len(s)]))
    # and a comparison against an id (the literal right after "evidence ==" / "!=")
    ids.update(re.findall(r'\bevidence\s*[!=]=\s*"((?:[^"\\\n]|\\.)*)"', s))
    for m in scrub.EVID.finditer(s):
        lm = LIT.search(m.group(0), len(m.group(1)))
        if lm:
            ids.add(lm.group(1))
    for m in scrub.COST.finditer(s):
        sp = scrub.second_element_span(s, m.end())
        if sp and "{" not in s[sp[0]:sp[1]]:
            ids.update(x.group(1) for x in LIT.finditer(s[sp[0]:sp[1]]))
    return ids


def catalog_ids(src_root):
    ids = set()
    for l in open(os.path.join(src_root, "common", "pxa-lever-catalog.inc"), encoding="utf-8"):
        m = ROW_FIELDS.match(l)
        if m:
            ev = LIT.findall(m.group(1))[4]
            ids.update(x.strip() for x in ev.split(","))
    return ids


SRC_EXT = (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".inc")
TAG_LIT = re.compile(r'"([a-z0-9][a-z0-9-]{4,}[a-z0-9])["(:]')


def component_tags(src_root):
    """Source file stems and literal log tags ("dn-convfuse", "pxqn-cpu-mmv: ...") in the code. A catalog evidence id
    that is also one of these names a component, not a record, and is not a hit of the in-repo rule (the block
    list still judges it)."""
    tags = set()
    for top in ("ggml", "src", "common", "tools", "examples"):
        for dp, dns, fs in os.walk(os.path.join(src_root, top)):
            dns[:] = [d for d in dns if not d.startswith(".")]
            for f in fs:
                stem, ext = os.path.splitext(f)
                if ext not in SRC_EXT or f in ("pxa-registry.cpp", "pxa-lever-catalog.inc"):
                    continue
                tags.add(stem)
                try:
                    tags.update(TAG_LIT.findall(open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()))
                except OSError:
                    pass
    return tags


def catalog_switches(src_root):
    rx = re.compile(r'^\{ "([A-Z][A-Z0-9_]*)"', re.M)
    return set(rx.findall(open(os.path.join(src_root, "common", "pxa-lever-catalog.inc"), encoding="utf-8").read()))


def shipped_files(src_root, build_dir):
    """bin/llama-*, the self-tests the packager stages, and lib*.so (maxdepth 3), ELF only."""
    names = set()
    try:
        mk = open(os.path.join(src_root, "scripts", "make-release-tarball.sh"), encoding="utf-8").read()
        for var in ("REQUIRED_BIN", "REQUIRED_ONDEMAND_BIN", "TEST_BIN", "OPTIONAL_BIN"):
            m = re.search(r'^%s="([^"]*)"' % var, mk, re.M)
            if m:
                names.update(m.group(1).split())
    except OSError:
        pass
    out = []
    bindir = os.path.join(build_dir, "bin")
    if os.path.isdir(bindir):
        for f in sorted(os.listdir(bindir)):
            if f.startswith("llama-") or f in names:
                out.append(os.path.join(bindir, f))
    base = build_dir.rstrip(os.sep).count(os.sep)
    for dp, dns, fs in os.walk(build_dir):
        if dp.rstrip(os.sep).count(os.sep) - base >= 3:
            dns[:] = []
        dns[:] = [d for d in dns if d != "CMakeFiles" and d != "bin"]
        for f in sorted(fs):
            if f.startswith("lib") and (f.endswith(".so") or ".so." in f):
                out.append(os.path.join(dp, f))
    res, seen = [], set()
    for p in out:
        rp = os.path.realpath(p)
        if rp in seen or not os.path.isfile(rp):
            continue
        with open(rp, "rb") as fh:
            if fh.read(4) != b"\x7fELF":
                continue
        seen.add(rp)
        res.append(p)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--release", action="store_true", help="a PXA_RELEASE_BUILD=ON build: the path categories apply too")
    ap.add_argument("--evidence-ids", action="store_true", help="a PXA_REGISTRY_EVIDENCE=ON dev build: registry ids are expected")
    ap.add_argument("--blocklist", default="", help="default: $PXA_RELEASE_BLOCKLIST")
    ap.add_argument("files", nargs="*", help="ELF files to scan instead of the build's shipped set")
    a = ap.parse_args(argv)
    if a.release and a.evidence_ids:
        print("test-pxa-release-strings: --release and --evidence-ids exclude each other")
        return 2
    src = os.path.abspath(a.source_root)
    try:
        rs = load(os.path.join(src, "scripts", "release-scan.py"), "pxa_release_scan")
        scrub = load(os.path.join(src, "scripts", "pxa-release-scrub.py"), "pxa_release_scrub")
        reg = {i for i in registry_ids(src, scrub) if len(i) >= MIN_ID}
        cat = {i for i in catalog_ids(src) if len(i) >= MIN_ID} - component_tags(src)
    except (OSError, AttributeError) as e:
        print("test-pxa-release-strings: CANNOT RUN: %s" % e)
        return 2
    if not reg:
        print("test-pxa-release-strings: CANNOT RUN: no registry evidence id found (the source shape changed?)")
        return 2
    ids = cat | (set() if a.evidence_ids else reg)
    mask_ids = sorted(reg, key=len, reverse=True) if a.evidence_ids else []
    id_tokens = {i.lower() for i in ids if TOKEN_FULL.fullmatch(i)}
    id_phrases = sorted(i for i in ids if not TOKEN_FULL.fullmatch(i))

    blist = a.blocklist or os.environ.get("PXA_RELEASE_BLOCKLIST", "")
    rules = None
    if blist:
        if not os.path.isfile(blist):
            print("test-pxa-release-strings: CANNOT RUN: PXA_RELEASE_BLOCKLIST is set but the file does not exist")
            return 2
        try:
            rules = rs.load_rules(blist)
        except rs.GateError as e:
            print("test-pxa-release-strings: CANNOT RUN: block list: %s" % e)
            return 2
    switches = catalog_switches(src) | (rules.allow_switch if rules else set())

    build = os.path.abspath(a.build_dir)
    files = [os.path.abspath(f) for f in a.files] or shipped_files(src, build)
    if not files:
        print("test-pxa-release-strings: CANNOT RUN: no shipped ELF file under %s" % a.build_dir)
        return 2
    print("test-pxa-release-strings: %d ELF file(s); in-repo ids %d (registry %s, catalog %d); block list %s; mode %s"
          % (len(files), len(ids), "masked" if a.evidence_ids else len(reg), len(cat),
             "%d categories" % len(rules.cats) if rules else "not set (in-repo rule only)",
             "release" if a.release else ("dev+evidence-ids" if a.evidence_ids else "dev")))
    bad = 0
    # the build tree's RUNPATH (directories of this build and the CUDA toolkit, ':'-joined): the packager rewrites it to
    # $ORIGIN/../lib with patchelf (scripts/make-release-tarball.sh), so it never ships and is not a hit here
    run_dirs = (build, os.path.realpath(build), "/usr/local/cuda")

    def runpath_line(l):
        parts = [x for x in l.split(":") if x]
        return bool(parts) and all(any(x == d or x.startswith(d) for d in run_dirs) for x in parts)

    for f in files:
        text = "\n".join(l for l in rs.elf_strings(f).split("\n") if not runpath_line(l))
        for mid in mask_ids:
            text = text.replace(mid, " " * len(mid))
        hits = {}
        t = text
        if rules:
            t = rs.masked(text, rules, switches)
            for c in rules.cats:
                if "binary" not in c["scope"] or (not a.release and c["id"] in PATH_CATS):
                    continue
                if c["kind"] == "regex":
                    found = [m.group(0) for m in c["rx"].finditer(t)] if c["rx"] is not None else []
                else:
                    found = [m.group(0) for m in rs.TOKEN_SPLIT.finditer(t) if m.group(0).lower() in c["ids"]]
                for x in found:
                    hits.setdefault(c["id"], set()).add(x[:80])
        for m in rs.TOKEN_SPLIT.finditer(t):
            if m.group(0).lower() in id_tokens:
                hits.setdefault("evidence-id", set()).add(m.group(0))
        for p in id_phrases:
            if p in t:
                hits.setdefault("evidence-id", set()).add(p[:80])
        rel = os.path.relpath(f, build) if f.startswith(build + os.sep) else f
        if hits:
            bad += 1
            for c in sorted(hits):
                xs = sorted(hits[c])
                print("  FAIL %-14s %-36s %d: %s" % (c, rel, len(xs), ", ".join(repr(x) for x in xs[:6]) + (" ..." if len(xs) > 6 else "")))
    if bad:
        print("test-pxa-release-strings: FAIL - %d file(s) carry internal strings (word them neutrally in the source)" % bad)
        return 1
    print("test-pxa-release-strings: PASS - 0 hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
