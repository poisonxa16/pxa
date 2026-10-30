#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Decisions for scripts/make-windows-release.sh. Stdlib only.

The Windows release is three files: pxa-control-win64.exe, pxa-engine-win64-cuda12.zip
(engine programs, the engine DLLs, the CUDA 12 runtime, the Microsoft C++ runtime), and
ggml-pxqn.dll. The PXQN library is licensed. It is published beside the zip. It is placed
inside the zip only when the packager passes --bundle-pxqn. A DLL whose export table is
the old all-symbols build is refused. The release DLL exports exactly the .def names (13 since
ggml_pxqn_get_overdrive_api): the three gu_* test hooks are not dllexported from ggml-pxqn.dll on
Windows since 2026-10-19 (ggml-cuda.h GGML_PXQN_HOOK_API); a DLL that still exports them is a
pre-fix or a test build.

The VM build tree is C:\\pxa\\src. This runs on the host that holds the staged copies
(default: the directory PXA_WIN_STAGE names). It does not publish anything.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_TOP = "pxa-engine-win64-cuda12"
CONTROL_NAME = "pxa-control-win64.exe"
PXQN_NAME = "ggml-pxqn.dll"
ZIP_NAME = "pxa-engine-win64-cuda12.zip"
NOTES_NAME = "RELEASE-NOTES-windows.md"
DEF_REL = os.path.join("ggml", "src", "win-compat", "ggml-pxqn.def")
NOTES_REL = os.path.join("docs", "windows-release-notes.md")
THIRD_PARTY_REL = os.path.join("docs", "windows-third-party-notices.txt")
PXQN_TERMS_REL = os.path.join("docs", "LICENSE-ggml-pxqn.txt")
PXQN_TERMS_NAME = "LICENSE-ggml-pxqn.txt"
WORK_MARKER = ".pxa-win-release-work"
# gu_* test hooks: exported from ggml-pxqn.dll only by a pre-2026-10-19 build or a PXA_PXQN_EXPORT_TEST_HOOKS test
# build. The release check refuses them (allow_hooks=False); nothing outside the .def list may be exported.
PXQN_HOOKS = (
    "ggml_backend_cuda_pxqn_gu_counters_reset",
    "ggml_backend_cuda_pxqn_gu_decode_count",
    "ggml_backend_cuda_pxqn_gu_prefill_count",
)
# the build host's stage, search root and output directory come from the environment only (no default path: an
# unset output fails closed in build_plan)
DEFAULT_STAGED = os.environ.get("PXA_WIN_STAGE") or ""
DEFAULT_SEARCH = os.environ.get("PXA_WIN_SEARCH") or ""
DEFAULT_OUT = os.environ.get("PXA_WIN_OUT") or ""
MAX_EXPORTS = 32
MAX_DLL_BYTES = 512 << 20

REQUIRED_EXACT = ("llama-server.exe", "ggml.dll", "vcruntime140.dll")
REQUIRED_GLOBS = ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
REFUSE_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".cu", ".cuh", ".inc", ".py", ".map"}
SKIP_EXT = {".pdb", ".lib", ".exp", ".ilk", ".obj", ".idb", ".pch"}

# The sentence in docs/windows-release-notes.md. --bundle-pxqn rewrites this one sentence.
PXQN_SEPARATE = (
    "The PXQN library is ggml-pxqn.dll. It is published beside the zip, not inside it. "
    "Copy it into engine\\bin next to ggml.dll to run a PXQN model. Without that file, "
    "PXQ models and other GGUF models still run, and a PXQN model is refused with one message."
)
PXQN_BUNDLE = (
    "The PXQN library is ggml-pxqn.dll. This zip includes it in bin\\ next to ggml.dll, "
    "and the same file is published beside the zip. Without that file, PXQ models and "
    "other GGUF models still run, and a PXQN model is refused with one message."
)

BANNED = (
    ("switch", re.compile(r"\bPXA_[A-Z0-9_]+\b")),
    ("switch", re.compile(r"\bPXQ_[A-Z0-9_]+\b")),
    ("store-word", re.compile(r"\b(?:paid|price|sell|buy)\b", re.I)),
    ("lane", re.compile(r"\bm2-[a-z0-9-]+\b")),
    ("lane", re.compile(r"\bmain[0-9]\b")),
    ("lane", re.compile(r"\bgrok-[a-z0-9-]+\b")),
    ("path", re.compile(r"/(?:mnt|boot)/")),
)

ROOTS_REFUSED = {"/", "/mnt", "/media", "/usr", "/boot", "/root", "/etc", "/bin", "/home"}
MOUNT_PARENTS = ("/mnt", "/media")      # a mount root directly under these (a whole disk or share) is refused too


class GuardError(Exception):
    pass


def _need(data, off, n, what):
    if off < 0 or n < 0 or off + n > len(data):
        raise GuardError("PE export table is truncated (%s)" % what)


def def_export_names(path):
    """Names under EXPORTS in a .def file. Comment text after ';' is ignored."""
    names = []
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        line = line.split(";", 1)[0].strip()
        if not line or line.upper().startswith("LIBRARY") or line.upper().startswith("EXPORTS"):
            continue
        name = line.split()[0].split("=")[0]
        if name:
            names.append(name)
    return names


def pe_export_names(path):
    """Export names of a PE32 or PE32+ DLL. Reads the file once. Refuses a non-PE or a truncated table."""
    size = os.path.getsize(path)
    if size <= 0 or size > MAX_DLL_BYTES:
        raise GuardError("PE file is empty or larger than the library cap")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 0x40 or not data.startswith(b"MZ"):
        raise GuardError("not a PE file")
    e = struct.unpack_from("<I", data, 0x3C)[0]
    _need(data, e, 4 + 20, "PE header")
    if data[e:e + 4] != b"PE\0\0":
        raise GuardError("not a PE file")
    coff = e + 4
    nsections, size_opt = struct.unpack_from("<H", data, coff + 2)[0], struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    _need(data, opt, 2, "optional header")
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic == 0x20B:
        dd_off = 112
    elif magic == 0x10B:
        dd_off = 96
    else:
        raise GuardError("not a PE32 or PE32+ image")
    if size_opt < dd_off + 8:
        return []
    _need(data, opt, size_opt, "optional header")
    exp_rva, _exp_size = struct.unpack_from("<II", data, opt + dd_off)
    if exp_rva == 0:
        return []
    if nsections < 1 or nsections > 96:
        raise GuardError("PE section count is implausible")
    sec_off = opt + size_opt
    _need(data, sec_off, nsections * 40, "section table")
    sections = []
    for i in range(nsections):
        off = sec_off + i * 40
        vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((va, max(vsize, rawsize), rawptr, rawsize))

    def rva_to_off(rva):
        for va, span, rawptr, rawsize in sections:
            if va <= rva < va + span:
                delta = rva - va
                if delta >= rawsize:
                    raise GuardError("PE export RVA is outside the section's file bytes")
                return rawptr + delta
        raise GuardError("PE export RVA is not in any section")

    eo = rva_to_off(exp_rva)
    _need(data, eo, 40, "export directory")
    nnames = struct.unpack_from("<I", data, eo + 24)[0]
    names_rva = struct.unpack_from("<I", data, eo + 32)[0]
    if nnames > 100000:
        raise GuardError("export table is implausible (%d names)" % nnames)
    if nnames == 0:
        return []
    no = rva_to_off(names_rva)
    _need(data, no, nnames * 4, "export names")
    names = []
    for i in range(nnames):
        nrva = struct.unpack_from("<I", data, no + i * 4)[0]
        so = rva_to_off(nrva)
        end = data.find(b"\0", so, so + 4096)
        if end < 0:
            raise GuardError("export name is not a C string")
        names.append(data[so:end].decode("ascii", "replace"))
    return names


def check_exports(names, required, max_exports=MAX_EXPORTS, allow_hooks=False):
    """The release DLL must export every .def name and must not be the all-symbols build."""
    if not required:
        raise GuardError("export list is empty")
    missing = [n for n in required if n not in names]
    if missing:
        raise GuardError("missing %d required export(s)" % len(missing))
    if len(names) > max_exports:
        raise GuardError("export-count %d" % len(names))
    allowed = set(required) | (set(PXQN_HOOKS) if allow_hooks else set())
    extra = [n for n in names if n not in allowed]
    if extra:
        hooks = sum(1 for n in extra if n in PXQN_HOOKS)
        raise GuardError("unexpected %d export(s) outside the .def list (%d gu_* test hook(s): rebuild without "
                         "PXA_PXQN_EXPORT_TEST_HOOKS)" % (len(extra), hooks))
    return len(names)


def is_old_pxqn(path):
    return ".OLD" in os.path.basename(path)


def is_driver_dll(name):
    n = name.lower()
    return n.startswith("nvcuda") or n.startswith("libcuda") or (n.endswith(".dll") and "nvidia" in n)


def banned_public_text(text):
    """Rule ids for public wording that must not ship. The matched text is not returned."""
    hits = []
    for kind, rx in BANNED:
        if rx.search(text):
            hits.append(kind)
    return hits


def check_notes_text(text):
    hits = banned_public_text(text)
    if "SmartScreen" not in text:
        hits.append("missing-smartscreen")
    if "Run anyway" not in text:
        hits.append("missing-run-anyway")
    return hits


def render_notes(template, bundle):
    if bundle:
        if PXQN_SEPARATE not in template:
            raise GuardError("notes template has no PXQN paragraph to rewrite")
        return template.replace(PXQN_SEPARATE, PXQN_BUNDLE, 1)
    return template


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_status(path):
    """None if there is no sibling .sha256. 'ok', 'mismatch', or 'malformed' otherwise.
    The hash itself is not returned."""
    side = path + ".sha256"
    if not os.path.isfile(side):
        return None
    raw = open(side, encoding="utf-8", errors="replace").read().strip().split()
    if not raw or not re.fullmatch(r"[0-9a-fA-F]{64}", raw[0]):
        return "malformed"
    return "ok" if sha256_file(path) == raw[0].lower() else "mismatch"


def find_pxqn(root, max_depth=4):
    """Newest file whose name is exactly ggml-pxqn.dll. *.OLD* copies are reported, not chosen."""
    root = os.path.abspath(root)
    best, best_m, old = "", -1.0, []
    if not os.path.isdir(root):
        return best, old
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.startswith(PXQN_NAME):
                continue
            path = os.path.join(dirpath, fn)
            if fn != PXQN_NAME:
                if ".OLD" in fn:
                    old.append(path)
                continue
            try:
                m = os.path.getmtime(path)
            except OSError:
                continue
            if m >= best_m:
                best, best_m = path, m
    return best, old


def zip_names(path):
    names = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/").lstrip("/")
            if name and not name.endswith("/"):
                names.append(name)
    return names


def basenames(names):
    found = {}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        found.setdefault(base, name)
    return found


def is_plugin_payload(base):
    """A plugin package or manifest: plugins ship only as signed .pxp Store downloads, never inside the package
    (owner 2026-10-21). release-scan.py's built-in plugin-payload rule checks the whole stage and the zip too."""
    b = base.lower()
    return b.endswith(".pxp") or b in ("plugin.json", "plugin.src.json")


def member_problems(found):
    """(missing labels, bad (basename, reason)). found maps basename -> any path."""
    missing, bad = [], []
    bases = list(found)
    for name in REQUIRED_EXACT:
        if name not in found:
            missing.append(name)
    for pattern in REQUIRED_GLOBS:
        if not any(fnmatch.fnmatch(b, pattern) for b in bases):
            missing.append(pattern)
    for base in bases:
        ext = os.path.splitext(base)[1].lower()
        if is_plugin_payload(base):
            bad.append((base, "plugin-payload"))
        elif ext in REFUSE_EXT:
            bad.append((base, "source-file"))
        elif is_driver_dll(base):
            bad.append((base, "driver-dll"))
    return missing, bad


def _refuse_root(path, what):
    ap = os.path.abspath(path)
    if ap in ROOTS_REFUSED or os.path.dirname(ap) in MOUNT_PARENTS:
        raise GuardError("refusing %s directory" % what)


def _nested(a, b):
    a, b = os.path.abspath(a), os.path.abspath(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def _nonempty(path):
    # a file with bytes, or a directory with at least one entry (--engine-bin / --cuda-bin / --redist are directories)
    if os.path.isdir(path):
        return any(True for _ in os.scandir(path))
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _parent_writable(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent
    return os.path.isdir(p) and os.access(p, os.W_OK)


def build_plan(opts):
    """Resolve every input path. Missing files stay in the plan so a dry-run can list them all."""
    notes = opts.notes or os.path.join(REPO, NOTES_REL)
    out = opts.out or DEFAULT_OUT
    if not out:
        raise GuardError("no output directory: pass --out or set PXA_WIN_OUT")
    work = opts.work or (os.path.abspath(out) + ".work")
    _refuse_root(out, "output")
    _refuse_root(work, "work")
    if _nested(out, work):
        raise GuardError("output and work directories must be different")
    # The packager removes the work directory at the end: never one it did not create.
    if os.path.isdir(work) and os.listdir(work) and not os.path.isfile(os.path.join(work, WORK_MARKER)):
        raise GuardError("work directory exists and was not made by this packager (no %s)" % WORK_MARKER)
    staged = opts.staged
    if not staged and not opts.control and not opts.engine_bin and not opts.engine_zip and not opts.pxqn:
        if DEFAULT_STAGED and os.path.isfile(os.path.join(DEFAULT_STAGED, PXQN_NAME)):
            staged = DEFAULT_STAGED
        elif (opts.search_root or DEFAULT_SEARCH) and os.path.isdir(opts.search_root or DEFAULT_SEARCH):
            dll, _old = find_pxqn(opts.search_root or DEFAULT_SEARCH)
            staged = os.path.dirname(dll) if dll else ""
    found_dll, old = ("", [])
    if staged and os.path.isdir(staged):
        found_dll, old = find_pxqn(staged, max_depth=3)
    control = opts.control or ""
    engine_zip = opts.engine_zip or ""
    engine_bin = opts.engine_bin or ""
    pxqn = opts.pxqn or ""
    if staged and os.path.isdir(staged):
        base = os.path.dirname(found_dll) if found_dll else staged
        if not control:
            cand = os.path.join(base, CONTROL_NAME)
            if os.path.isfile(cand):
                control = cand
            elif os.path.isfile(os.path.join(staged, CONTROL_NAME)):
                control = os.path.join(staged, CONTROL_NAME)
        if not engine_zip and not engine_bin:
            for directory in (base, staged):
                cand = os.path.join(directory, ZIP_NAME)
                if os.path.isfile(cand):
                    engine_zip = cand
                    break
        if not pxqn:
            pxqn = found_dll
    if opts.no_pxqn and opts.bundle_pxqn:
        raise GuardError("--bundle-pxqn and --no-pxqn cannot be used together")
    if opts.no_pxqn:
        disposition = "absent"
    elif opts.bundle_pxqn:
        disposition = "bundle"
    else:
        disposition = "separate"
    blocklist = opts.blocklist or os.environ.get("PXA_RELEASE_BLOCKLIST", "")
    plan = {
        "repo": REPO,
        "staged": staged or "",
        "control": control,
        "engine_zip": engine_zip,
        "engine_bin": engine_bin,
        "cuda_bin": opts.cuda_bin or "",
        "redist": opts.redist or "",
        "pxqn": pxqn,
        "pxqn_old": old,
        "notes": notes,
        "blocklist": blocklist,
        "bundle": bool(opts.bundle_pxqn),
        "no_pxqn": bool(opts.no_pxqn),
        "disposition": disposition,
        "out": os.path.abspath(out),
        "work": os.path.abspath(work),
        "closed_inc": opts.closed_inc or "",
        "verify_hashes": not opts.no_verify_hashes,
        "def_file": os.path.join(REPO, DEF_REL),
        "release_scan": os.path.join(REPO, "scripts", "release-scan.py"),
        "lever_catalog": os.path.join(REPO, "scripts", "pxa-lever-catalog.py"),
        "closed_levers": os.path.join(REPO, "scripts", "pxa-closed-levers.txt"),
        "stage": "",
        "engine_dir": "",
        "zip_path": "",
        "clean_files": [],
    }
    return plan


class Lines(object):
    def __init__(self):
        self.rows = []
        self.bad = 0

    def ok(self, kind, path):
        self.rows.append("OK %s %s" % (kind, path))

    def miss(self, kind, path):
        self.bad += 1
        self.rows.append("MISSING %s %s" % (kind, path))

    def badline(self, kind, path, why):
        self.bad += 1
        self.rows.append("BAD %s %s %s" % (kind, path, why))

    def skip(self, kind, path):
        self.rows.append("SKIP %s %s" % (kind, path))


def _hash_line(lines, kind, path, verify):
    if not verify or not os.path.isfile(path):
        return
    status = sidecar_status(path)
    if status is None:
        return
    if status == "ok":
        lines.ok("sha256", os.path.basename(path))
    else:
        lines.badline("sha256", os.path.basename(path), status)


def _payload_names(plan):
    """Basenames that would be packed, from a zip listing or a loose bin directory. Does not extract."""
    names = []
    if plan["engine_bin"] and os.path.isdir(plan["engine_bin"]):
        names.extend(_dir_payload_names(plan["engine_bin"]))
    elif plan["engine_zip"] and os.path.isfile(plan["engine_zip"]):
        names.extend(os.path.basename(n) for n in zip_names(plan["engine_zip"]))
    for extra, pattern in ((plan["cuda_bin"], "*.dll"), (plan["redist"], "*.dll")):
        if extra and os.path.isdir(extra):
            for fn in os.listdir(extra):
                if fnmatch.fnmatch(fn, pattern) and os.path.isfile(os.path.join(extra, fn)):
                    names.append(fn)
    if plan["bundle"] and plan["pxqn"]:
        names.append(PXQN_NAME)
    # The licensed library is not part of the engine payload unless bundling.
    if not plan["bundle"]:
        names = [n for n in names if n.lower() != PXQN_NAME.lower()]
    return basenames(names)


def _dir_payload_names(directory):
    out = []
    for fn in os.listdir(directory):
        path = os.path.join(directory, fn)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in SKIP_EXT:
            continue
        if fn.lower().endswith(".exe") or fn.lower().endswith(".dll") or ext in REFUSE_EXT or is_plugin_payload(fn):
            out.append(fn)
    return out


def evaluate(plan):
    """Check every input path. Exit-style code is lines.bad (0 = ready)."""
    lines = Lines()
    lines.ok("disposition", plan["disposition"])
    for kind, path in (
        ("control", plan["control"]),
        ("engine-zip", plan["engine_zip"]),
        ("engine-bin", plan["engine_bin"]),
        ("cuda-bin", plan["cuda_bin"]),
        ("redist", plan["redist"]),
        ("notes", plan["notes"]),
        ("blocklist", plan["blocklist"]),
        ("release-scan", plan["release_scan"]),
        ("lever-catalog", plan["lever_catalog"]),
        ("def-file", plan["def_file"]),
    ):
        if kind in ("engine-zip", "engine-bin", "cuda-bin", "redist") and not path:
            continue
        if kind == "blocklist" and not path:
            lines.miss(kind, "PXA_RELEASE_BLOCKLIST")
            continue
        if _nonempty(path):
            lines.ok(kind, path)
        else:
            lines.miss(kind, path or kind)
    repo = plan.get("repo") or REPO
    for kind, rel in (("license", "LICENSE"), ("notice", "NOTICE"), ("third-party", THIRD_PARTY_REL),
                      ("pxqn-terms", PXQN_TERMS_REL)):
        if kind == "pxqn-terms" and plan["no_pxqn"]:
            continue
        path = os.path.join(repo, rel)
        if _nonempty(path):
            lines.ok(kind, path)
        else:
            lines.miss(kind, path)
    if plan["engine_zip"] and plan["engine_bin"]:
        lines.badline("engine", plan["engine_bin"], "both-zip-and-bin")
    if not plan["engine_zip"] and not plan["engine_bin"]:
        lines.miss("engine", "engine-zip-or-bin")
    if not _parent_writable(plan["out"]):
        lines.miss("out", plan["out"])
    else:
        lines.ok("out", plan["out"])
    for path in plan["pxqn_old"]:
        lines.skip("old-pxqn", path)
    if plan["no_pxqn"]:
        lines.ok("pxqn", "absent")
    elif not _nonempty(plan["pxqn"]):
        lines.miss("pxqn", plan["pxqn"] or PXQN_NAME)
    elif is_old_pxqn(plan["pxqn"]):
        lines.badline("pxqn", plan["pxqn"], "old-dll")
    else:
        lines.ok("pxqn", plan["pxqn"])
        try:
            required = def_export_names(plan["def_file"]) if os.path.isfile(plan["def_file"]) else []
            count = check_exports(pe_export_names(plan["pxqn"]), required)
            lines.ok("pxqn-exports", str(count))
        except GuardError as e:
            why = str(e)
            lines.badline("pxqn-exports", os.path.basename(plan["pxqn"]), why.split()[0] if why else "exports")
    if os.path.isfile(plan["notes"]):
        try:
            rendered = render_notes(open(plan["notes"], encoding="utf-8").read(), plan["bundle"])
        except GuardError as e:
            lines.badline("notes-text", plan["notes"], "template")
            rendered = ""
        for kind in check_notes_text(rendered):
            lines.badline("notes-text", os.path.basename(plan["notes"]), kind)
    found = _payload_names(plan)
    if plan["engine_zip"] or plan["engine_bin"]:
        missing, bad = member_problems(found)
        for name in missing:
            lines.miss("member", name)
        for base, why in bad:
            lines.badline("member", base, why)
        for name in list(REQUIRED_EXACT) + [b for b in found if any(fnmatch.fnmatch(b, g) for g in REQUIRED_GLOBS)]:
            if name in found and name not in missing:
                lines.ok("member", name)
    if plan["verify_hashes"]:
        for path in (plan["control"], plan["engine_zip"], plan["pxqn"]):
            if path:
                _hash_line(lines, "sha256", path, True)
    if os.path.isfile(plan["closed_levers"]):
        lines.ok("closed-levers", plan["closed_levers"])
    else:
        lines.skip("closed-levers", plan["closed_levers"])
    return lines


def _extract_zip(path, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/").lstrip("/")
            if not name or name.endswith("/"):
                continue
            if name.startswith("../") or "/../" in "/" + name or re.match(r"^[A-Za-z]:", name):
                raise GuardError("engine zip member escapes the extract directory")
            target = os.path.realpath(os.path.join(dest, name))
            if not (target == os.path.realpath(dest) or target.startswith(os.path.realpath(dest) + os.sep)):
                raise GuardError("engine zip member escapes the extract directory")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = z.open(info, "r")
            try:
                with open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            finally:
                src.close()


def _find_bin(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "llama-server.exe" in files:
            found.append(dirpath)
    if not found:
        raise GuardError("engine input has no llama-server.exe")
    found.sort(key=lambda p: (0 if os.path.basename(p).lower() == "bin" else 1, len(p)))
    return found[0]


def _copy_payload(src_dir, dest_bin, bundle):
    os.makedirs(dest_bin, exist_ok=True)
    copied = []
    for fn in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, fn)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(fn)[1].lower()
        low = fn.lower()
        if ext in SKIP_EXT:
            continue
        if ext in REFUSE_EXT or is_driver_dll(fn):
            raise GuardError("refusing %s" % fn)
        if low == PXQN_NAME.lower() and not bundle:
            continue
        if low.endswith(".exe") or low.endswith(".dll"):
            shutil.copyfile(src, os.path.join(dest_bin, fn))
            copied.append(fn)
    return copied


def _merge_extra(directory, dest_bin, patterns):
    if not directory:
        return
    if not os.path.isdir(directory):
        raise GuardError("extra dll directory is missing")
    have = set(os.listdir(dest_bin))
    for fn in sorted(os.listdir(directory)):
        if fn in have or not os.path.isfile(os.path.join(directory, fn)):
            continue
        if is_driver_dll(fn):
            raise GuardError("refusing %s" % fn)
        if any(fnmatch.fnmatch(fn, p) for p in patterns):
            shutil.copyfile(os.path.join(directory, fn), os.path.join(dest_bin, fn))


def assemble(plan):
    lines = evaluate(plan)
    if lines.bad:
        sys.stdout.write("\n".join(lines.rows) + "\n")
        raise GuardError("inputs are not ready")
    work, out = plan["work"], plan["out"]
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, WORK_MARKER), "w", encoding="utf-8") as f:
        f.write("made by scripts/make-windows-release.sh\n")
    stage = os.path.join(work, "stage")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    engine_dir = os.path.join(stage, ENGINE_TOP)
    dest_bin = os.path.join(engine_dir, "bin")
    os.makedirs(dest_bin, exist_ok=True)
    if plan["engine_bin"]:
        src_bin = plan["engine_bin"]
    else:
        unpack = os.path.join(work, "unpack")
        if os.path.isdir(unpack):
            shutil.rmtree(unpack)
        _extract_zip(plan["engine_zip"], unpack)
        src_bin = _find_bin(unpack)
    _copy_payload(src_bin, dest_bin, plan["bundle"])
    _merge_extra(plan["cuda_bin"], dest_bin, REQUIRED_GLOBS)
    _merge_extra(plan["redist"], dest_bin, ("*.dll",))
    if plan["bundle"]:
        shutil.copyfile(plan["pxqn"], os.path.join(dest_bin, PXQN_NAME))
    elif os.path.exists(os.path.join(dest_bin, PXQN_NAME)):
        raise GuardError("the licensed library reached the engine directory")
    rendered = render_notes(open(plan["notes"], encoding="utf-8").read(), plan["bundle"])
    notes_path = os.path.join(stage, NOTES_NAME)
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(rendered if rendered.endswith("\n") else rendered + "\n")
    with open(os.path.join(engine_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(rendered if rendered.endswith("\n") else rendered + "\n")
    # MIT requires the copyright notice in every copy; the CUDA and MSVC runtimes need their own notice.
    repo = plan.get("repo") or REPO
    shutil.copyfile(os.path.join(repo, "LICENSE"), os.path.join(engine_dir, "LICENSE"))
    with open(os.path.join(repo, "NOTICE"), encoding="utf-8") as f:
        notice = f.read()
    with open(os.path.join(repo, THIRD_PARTY_REL), encoding="utf-8") as f:
        notice = notice.rstrip("\n") + "\n" + f.read()
    with open(os.path.join(engine_dir, "NOTICE"), "w", encoding="utf-8") as f:
        f.write(notice)
    shutil.copyfile(plan["control"], os.path.join(stage, CONTROL_NAME))
    if not plan["no_pxqn"]:
        shutil.copyfile(plan["pxqn"], os.path.join(stage, PXQN_NAME))
        # the library's own terms: beside the DLL download and in the engine folder it is copied into
        shutil.copyfile(os.path.join(repo, PXQN_TERMS_REL), os.path.join(stage, PXQN_TERMS_NAME))
        shutil.copyfile(os.path.join(repo, PXQN_TERMS_REL), os.path.join(engine_dir, PXQN_TERMS_NAME))
    found = basenames(_dir_payload_names(dest_bin))
    missing, bad = member_problems(found)
    if missing or bad:
        raise GuardError("staged engine is incomplete")
    clean = []
    for dirpath, _dirs, files in os.walk(stage):
        for fn in sorted(files):
            if fn.lower() == PXQN_NAME.lower():
                continue
            clean.append(os.path.join(dirpath, fn))
    plan["stage"] = stage
    plan["engine_dir"] = engine_dir
    plan["zip_path"] = os.path.join(work, ZIP_NAME)
    plan["clean_files"] = clean
    return plan


def zip_engine(plan):
    engine_dir = plan["engine_dir"]
    zip_path = plan["zip_path"]
    if not engine_dir or not os.path.isdir(engine_dir):
        raise GuardError("nothing to zip")
    dll = os.path.join(engine_dir, "bin", PXQN_NAME)
    if os.path.exists(dll) and not plan["bundle"]:
        raise GuardError("the licensed library reached the engine zip")
    if plan["bundle"] and not os.path.isfile(dll):
        raise GuardError("bundle requested but the library is not in the engine directory")
    parent = os.path.dirname(os.path.abspath(engine_dir))
    tmp = zip_path + ".partial"
    if os.path.exists(tmp):
        os.remove(tmp)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        paths = []
        for dirpath, _dirs, files in os.walk(engine_dir):
            for fn in files:
                paths.append(os.path.join(dirpath, fn))
        for full in sorted(paths):
            rel = os.path.relpath(full, parent).replace(os.sep, "/")
            z.write(full, rel)
    os.replace(tmp, zip_path)
    return zip_path


def save_plan(plan, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_plan(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parser():
    ap = argparse.ArgumentParser(prog="windows-release-guards", add_help=False)
    ap.add_argument("--staged", default="")
    ap.add_argument("--search-root", default=DEFAULT_SEARCH)
    ap.add_argument("--control", default="")
    ap.add_argument("--engine-bin", default="")
    ap.add_argument("--engine-zip", default="")
    ap.add_argument("--cuda-bin", default="")
    ap.add_argument("--redist", default="")
    ap.add_argument("--pxqn", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--closed-inc", default="")
    ap.add_argument("--blocklist", default="")
    ap.add_argument("--bundle-pxqn", action="store_true")
    ap.add_argument("--no-pxqn", action="store_true")
    ap.add_argument("--no-verify-hashes", action="store_true")
    return ap


def cmd_plan(argv):
    ap = _parser()
    ap.add_argument("--json", required=True)
    opts = ap.parse_args(argv)
    plan = build_plan(opts)
    save_plan(plan, opts.json)
    return 0


def cmd_check(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plan", required=True)
    a = ap.parse_args(argv)
    lines = evaluate(load_plan(a.plan))
    sys.stdout.write("\n".join(lines.rows) + "\n")
    return 1 if lines.bad else 0


def cmd_assemble(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plan", required=True)
    a = ap.parse_args(argv)
    plan = load_plan(a.plan)
    plan = assemble(plan)
    save_plan(plan, a.plan)
    print("STAGE %s" % plan["stage"])
    return 0


def cmd_zip(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plan", required=True)
    a = ap.parse_args(argv)
    plan = load_plan(a.plan)
    path = zip_engine(plan)
    print("ZIP %s" % path)
    return 0


def cmd_show(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plan", required=True)
    ap.add_argument("key")
    a = ap.parse_args(argv)
    value = load_plan(a.plan)[a.key]
    if isinstance(value, list):
        sys.stdout.write("\n".join(value) + ("\n" if value else ""))
    elif isinstance(value, bool):
        print("1" if value else "0")
    else:
        print(value)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: windows-release-guards.py plan|check|assemble|zip-engine|show ...", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "plan":
            return cmd_plan(rest)
        if cmd == "check":
            return cmd_check(rest)
        if cmd == "assemble":
            return cmd_assemble(rest)
        if cmd == "zip-engine":
            return cmd_zip(rest)
        if cmd == "show":
            return cmd_show(rest)
    except GuardError as e:
        print("windows-release-guards: %s" % e, file=sys.stderr)
        return 1
    print("windows-release-guards: unknown command %s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
