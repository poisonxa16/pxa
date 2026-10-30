#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Acceptance checks for the Windows release package. Read-only on the inputs. Stdlib only.

    scripts/check-win-package.sh <pxa-engine-win64-cuda12.zip | release-dir> [options]

The zip's siblings (pxa-control-win64.exe, ggml-pxqn.dll, RELEASE-NOTES-windows.md, SHA256SUMS) are found next
to it unless given. Every check prints one line:  PASS|FAIL|WARN|INFO|MANUAL <check-id> <detail>.
Exit 0 = no FAIL, 1 = at least one FAIL, 3 = the checker could not run (unreadable zip, bad arguments).
Hits print the file name, the rule id and a count, never the matched text (the private block list and any
secret stay unprinted). The checklist this implements: WIN-PACKAGE-REVIEW.md (release docs).
"""
import argparse
import fnmatch
import hashlib
import importlib.util
import json
import marshal
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_TOP = "pxa-engine-win64-cuda12"
ZIP_NAME = ENGINE_TOP + ".zip"
CONTROL_NAME = "pxa-control-win64.exe"
PXQN_NAME = "ggml-pxqn.dll"
NOTES_NAME = "RELEASE-NOTES-windows.md"
PXQN_TERMS_NAME = "LICENSE-ggml-pxqn.txt"
SUMS_NAME = "SHA256SUMS"
DEF_REL = os.path.join("ggml", "src", "win-compat", "ggml-pxqn.def")
# GGML_API test hooks (ggml-cuda.h). Since 2026-10-19 ggml-pxqn.dll does not export them on Windows, so the release
# DLL exports exactly the .def names (read from the .def: 13 since ggml_pxqn_get_overdrive_api). --allow-test-hooks
# accepts an older build or a test build (.def + 3).
PXQN_HOOKS = ("ggml_backend_cuda_pxqn_gu_counters_reset", "ggml_backend_cuda_pxqn_gu_decode_count",
              "ggml_backend_cuda_pxqn_gu_prefill_count")
DEFAULT_ARCHS = (60, 61, 70, 75, 80, 86, 89)   # SASS: the release floor (sm 60;61;70;75) + 80;86;89
DEFAULT_PTX = (89,)                            # PTX for the newest arch only: GPUs newer than sm_89 JIT it
DEFAULT_BLOCKLIST = ""   # the private block list comes from PXA_RELEASE_BLOCKLIST or --blocklist
CUOBJDUMP_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"

# NVIDIA's own redistributables (cuBLAS, cudart ...): their arch lists are NVIDIA's. A cubin for sm_XY runs on every sm_XZ
# with Z >= Y (same major), so a required arch counts as covered by the lib's own SASS for it OR an older SASS of the same
# major (cuBLAS 12.8 ships sm_86 + sm_90 and no sm_89: an RTX 40xx runs the sm_86 cubins natively, no JIT). Ours stay exact.
VENDOR_PREFIXES = ("cublas", "cudart", "cusparse", "cufft", "curand", "nvrtc", "nvjit")
VENDOR_FATBINS = set()      # extracted fatbin file names of vendor libs (for the cuobjdump cross-check)


def is_vendor(label):
    return os.path.basename(label).lower().startswith(VENDOR_PREFIXES)


def arch_lack(archs, sass, vendor):
    if not vendor:
        return [a for a in archs if a not in sass]
    return [a for a in archs if not any(s // 10 == a // 10 and s <= a for s in sass)]


REQUIRED_BIN = ("llama-server.exe", "ggml.dll", "llama.dll", "cudart64_12.dll", "cublas64_12.dll",
                "cublasLt64_12.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
TOP_FILES_REQUIRED = ("README.txt", "LICENSE", "NOTICE")
REFUSE_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".cu", ".cuh", ".inc", ".py", ".map", ".def",
              ".pdb", ".lib", ".exp", ".ilk", ".obj", ".idb", ".pch", ".sh", ".bat", ".ps1"}
SECRET_NAMES = ("*.pem", "*.key", "*.token", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_ecdsa*", ".env",
                "*.env", "*secret*", "*credential*", "*.kdbx", ".netrc", ".git-credentials")
# Windows DLLs every supported Windows has. Anything else a PE imports must ship in bin\ (or be the driver).
SYSTEM_DLLS = {
    "kernel32", "kernelbase", "ntdll", "user32", "gdi32", "advapi32", "shell32", "ole32", "oleaut32", "ws2_32",
    "bcrypt", "crypt32", "dbghelp", "shlwapi", "version", "winmm", "iphlpapi", "secur32", "rpcrt4", "comdlg32",
    "comctl32", "psapi", "setupapi", "userenv", "wintrust", "imm32", "dwmapi", "uxtheme", "mswsock", "dnsapi",
    "netapi32", "normaliz", "winhttp", "wininet", "powrprof", "cfgmgr32", "ucrtbase", "msvcrt", "dxgi", "d3d11",
    "d3d12", "opengl32", "glu32", "winspool.drv", "ncrypt", "mpr", "wtsapi32", "pdh", "netutils", "cabinet",
    "shcore", "propsys", "msi", "wldap32", "dbgeng", "ntdll", "authz", "credui", "urlmon", "oleacc", "hid",
    "winusb", "wevtapi", "tdh", "fltlib", "virtdisk", "wsock32", "msimg32", "usp10", "gdiplus", "windowscodecs",
    "mfplat", "mf", "mfreadwrite", "avrt", "ksuser", "dsound", "xinput1_4", "comsvcs", "esent", "srvcli",
}
DRIVER_DLLS = {"nvcuda", "nvml", "nvapi64", "nvrtc-builtins64"}
SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-")

# Generic shapes only: the private words live in the block list (release-scan), never in this file.
STRING_RULES = (
    ("lane-name", re.compile(rb"\b(?:m[23]-[a-z][a-z0-9]{2,}(?:-[a-z0-9]+)*|main[23]\b|grok-[a-z]{3,}[a-z0-9-]*)")),
    ("private-path", re.compile(rb"/(?:mnt)/(?:disk\d+|cache\w*|user)/|/(?:boot)/(?:config)/|/root/\.(?:claude|ssh)")),
    ("private-repo", re.compile(rb"github\.com[/:][A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+-private\b")),
    ("secret-shape", re.compile(rb"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-ant-[A-Za-z0-9_-]{20,}|"
                                rb"xox[abps]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
NOTES_STORE = re.compile(r"\b(?:paid|price|prices|pricing|sell|sells|selling|buy|buying|purchase)\b", re.I)
ASCII_RUN = re.compile(rb"[\x20-\x7e]{4,}")
UTF16_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")
FATBIN_MAGIC = 0xBA55ED50
PYI_MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"


class Result(object):
    def __init__(self):
        self.rows = []

    def add(self, status, check, detail):
        self.rows.append({"status": status, "check": check, "detail": detail})
        print("%-6s %-22s %s" % (status, check, detail), flush=True)

    def failed(self):
        return sum(1 for r in self.rows if r["status"] == "FAIL")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- PE parsing
class PE(object):
    """Just enough of a PE32/PE32+ image: machine, sections, imports, delay imports, exports."""

    def __init__(self, data):
        if len(data) < 0x40 or not data.startswith(b"MZ"):
            raise ValueError("not a PE file")
        e = struct.unpack_from("<I", data, 0x3C)[0]
        if e + 24 > len(data) or data[e:e + 4] != b"PE\0\0":
            raise ValueError("not a PE file")
        self.data = data
        coff = e + 4
        self.machine, nsec = struct.unpack_from("<HH", data, coff)
        size_opt = struct.unpack_from("<H", data, coff + 16)[0]
        opt = coff + 20
        self.magic = struct.unpack_from("<H", data, opt)[0]
        dd = {0x20B: 112, 0x10B: 96}.get(self.magic)
        if dd is None:
            raise ValueError("not a PE32 or PE32+ image")
        ndd = struct.unpack_from("<I", data, opt + dd - 4)[0] if size_opt >= dd else 0
        self.dirs = []
        for i in range(min(ndd, 16)):
            if opt + dd + 8 * i + 8 > opt + size_opt:
                break
            self.dirs.append(struct.unpack_from("<II", data, opt + dd + 8 * i))
        if nsec > 96:
            raise ValueError("implausible section count")
        self.sections = []
        so = opt + size_opt
        for i in range(nsec):
            o = so + 40 * i
            name = data[o:o + 8].rstrip(b"\0").decode("ascii", "replace")
            vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, o + 8)
            self.sections.append((name, va, max(vsize, rawsize), rawptr, rawsize))

    def off(self, rva):
        for _n, va, span, rawptr, rawsize in self.sections:
            if va <= rva < va + span:
                if rva - va >= rawsize:
                    raise ValueError("RVA outside file bytes")
                return rawptr + rva - va
        raise ValueError("RVA not in a section")

    def cstr(self, rva):
        o = self.off(rva)
        end = self.data.find(b"\0", o, o + 512)
        if end < 0:
            raise ValueError("unterminated name")
        return self.data[o:end].decode("ascii", "replace")

    def section_span(self, name):
        for n, _va, _span, rawptr, rawsize in self.sections:
            if n == name:
                return (rawptr, rawptr + rawsize)
        return None

    def section(self, name):
        for n, _va, _span, rawptr, rawsize in self.sections:
            if n == name:
                return self.data[rawptr:rawptr + rawsize]
        return None

    def _dir(self, i):
        return self.dirs[i] if i < len(self.dirs) else (0, 0)

    def imports(self):
        names = []
        rva, _size = self._dir(1)
        if rva:
            o = self.off(rva)
            for _ in range(4096):
                if o + 20 > len(self.data):
                    break
                ilt, _t, _f, name_rva, iat = struct.unpack_from("<IIIII", self.data, o)
                if not (ilt or name_rva or iat):
                    break
                names.append(self.cstr(name_rva))
                o += 20
        rva, _size = self._dir(13)          # delay-load imports
        if rva:
            o = self.off(rva)
            for _ in range(4096):
                if o + 32 > len(self.data):
                    break
                attrs, name_rva = struct.unpack_from("<II", self.data, o)
                if not name_rva:
                    break
                names.append(self.cstr(name_rva))
                o += 32
        return names

    def exports(self):
        rva, _size = self._dir(0)
        if not rva:
            return []
        o = self.off(rva)
        nnames = struct.unpack_from("<I", self.data, o + 24)[0]
        names_rva = struct.unpack_from("<I", self.data, o + 32)[0]
        if nnames > 100000:
            raise ValueError("implausible export count")
        no = self.off(names_rva) if nnames else 0
        return [self.cstr(struct.unpack_from("<I", self.data, no + 4 * i)[0]) for i in range(nnames)]


def fatbin_entries(blob):
    """[(fatbin_index, kind, sm)] for every fatbin in a .nv_fatb section. kind 1 = PTX, 2 = SASS (ELF)."""
    out, off, idx = [], 0, 0
    n = len(blob)
    while off + 16 <= n:
        magic, _ver, hsz, fsz = struct.unpack_from("<IHHQ", blob, off)
        if magic != FATBIN_MAGIC:
            off += 8
            continue
        if hsz < 16 or off + hsz + fsz > n:
            raise ValueError("truncated fatbin")
        p, end = off + hsz, off + hsz + fsz
        while p + 32 <= end:
            kind, _ev, ehs, pad = struct.unpack_from("<HHII", blob, p)
            sm = struct.unpack_from("<I", blob, p + 28)[0]
            if ehs < 32:
                raise ValueError("bad fatbin entry")
            out.append((idx, kind, sm))
            p += ehs + pad
        idx += 1
        off = end
    return out


def compressed_payload_spans(blob):
    """[(start, end)] of the fatbin entry payloads in a .nv_fatb section that are compressed (GGML_CUDA_COMPRESSION_MODE):
    a SASS payload that does not start with an ELF header or a PTX payload that is not text. Those bytes are random
    to a strings scan (a 7-char run like "q7-abcd" turns up by chance in hundreds of MB), so the string rules skip them;
    uncompressed device ELF / PTX stays scanned. The release scan gate still reads every byte."""
    spans, off, n = [], 0, len(blob)
    while off + 16 <= n:
        magic, _ver, hsz, fsz = struct.unpack_from("<IHHQ", blob, off)
        if magic != FATBIN_MAGIC:
            off += 8
            continue
        if hsz < 16 or off + hsz + fsz > n:
            break
        p, end = off + hsz, off + hsz + fsz
        while p + 32 <= end:
            kind, _ev, ehs, pad = struct.unpack_from("<HHII", blob, p)
            if ehs < 32:
                break
            a, b = p + ehs, min(p + ehs + pad, end)
            head = blob[a:a + 64]
            if kind == 2:
                packed = not head.startswith(b"\x7fELF")
            else:
                packed = bool(head) and sum(1 for c in head if 9 <= c <= 13 or 32 <= c < 127) < len(head) * 0.9
            if packed and b > a:
                spans.append((a, b))
            p += ehs + pad
        off = end
    return spans


def first_fatbin(blob):
    off = 0
    while off + 16 <= len(blob):
        magic, _ver, hsz, fsz = struct.unpack_from("<IHHQ", blob, off)
        if magic == FATBIN_MAGIC:
            return blob[off:off + hsz + fsz]
        off += 8
    return b""


def rule_hits(data):
    """{rule-id: count} over the ASCII and UTF-16LE strings of a blob."""
    hits = {}
    runs = ASCII_RUN.findall(data)
    runs += [r.decode("utf-16-le", "replace").encode("ascii", "replace") for r in UTF16_RUN.findall(data)]
    joined = b"\n".join(runs)
    for rid, rx in STRING_RULES:
        c = len(rx.findall(joined))
        if c:
            hits[rid] = c
    return hits, joined


# ---------------------------------------------------------------- PyInstaller (Control exe)
def pyinstaller_extract(data, dest):
    """Unpack a PyInstaller one-file exe: CArchive entries and the PYZ modules (zlib, marshal TOC).
    Writes every entry's bytes under dest (names sanitised). Returns [(name, path, ours)] or None if not PyInstaller."""
    c = data.rfind(PYI_MAGIC)
    if c < 0:
        return None
    if c + 88 <= len(data):
        _m, plen, toff, tlen, _pyv, _lib = struct.unpack_from("!8sIIII64s", data, c)
        cookie = 88
    else:
        _m, plen, toff, tlen, _pyv = struct.unpack_from("!8sIIII", data, c)
        cookie = 24
    pkg = c + cookie - plen
    if pkg < 0:
        raise ValueError("bad PyInstaller cookie")
    out = []
    t, end = pkg + toff, pkg + toff + tlen
    entries = []
    while t < end:
        es, dp, cl, ul, cf, ty = struct.unpack_from("!IIIIBc", data, t)
        if es < 18:
            raise ValueError("bad PyInstaller TOC")
        name = data[t + 18:t + es].rstrip(b"\0").decode("utf-8", "replace")
        entries.append((name, dp, cl, ul, cf, ty))
        t += es

    def safe(name):
        name = name.replace("\\", "/").lstrip("/")
        name = "/".join(p for p in name.split("/") if p not in ("", ".", ".."))
        return name or "_"

    def ours(name):
        base = name.replace("\\", "/").split("/")[0].lower()
        return base.startswith("pxa") or base.startswith("pxq")

    def put(rel, blob, mine):
        # third-party entries are scanned for content only: the suffix keeps the block list's file-name rules
        # (meant for files a user sees, e.g. stray *.pyc) off the stdlib inside the exe
        path = os.path.join(dest, "ours" if mine else "third", safe(rel) + ("" if mine else ".entry"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(blob)
        out.append((rel, path, mine))

    for name, dp, cl, _ul, cf, ty in entries:
        blob = data[pkg + dp:pkg + dp + cl]
        if cf:
            blob = zlib.decompress(blob)
        if ty in (b"z", b"Z") and blob[:4] == b"PYZ\0":
            toc_pos = struct.unpack("!i", blob[8:12])[0]
            toc = marshal.loads(blob[toc_pos:])
            items = toc.items() if isinstance(toc, dict) else toc
            for mod, meta in items:
                _typ, pos, length = meta
                try:
                    code = zlib.decompress(blob[pos:pos + length])
                except zlib.error:
                    code = blob[pos:pos + length]
                put("PYZ/" + mod.replace(".", "/") + ".pyc-data", code, ours(mod) or ours(mod.split(".")[0]))
            continue
        mine = ours(name) or ty == b"s" and name.startswith("pxa")
        if name.lower().endswith(".zip") and blob[:2] == b"PK":
            import io
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    for info in z.infolist():
                        if not info.filename.endswith("/"):
                            put(name + "/" + info.filename, z.read(info), mine)
                continue
            except zipfile.BadZipFile:
                pass
        put(name, blob, mine)
    return out


# ---------------------------------------------------------------- checks
def check_zip_layout(res, zpath, bundle, no_pxqn=False):
    try:
        z = zipfile.ZipFile(zpath)
    except (OSError, zipfile.BadZipFile) as e:
        res.add("FAIL", "zip-open", "%s: %s" % (os.path.basename(zpath), e.__class__.__name__))
        return None
    bad = z.testzip()
    if bad:
        res.add("FAIL", "zip-crc", "first bad member %s" % bad)
    else:
        res.add("PASS", "zip-crc", "%d members, CRC ok" % len(z.infolist()))
    names = [i.filename for i in z.infolist() if not i.filename.endswith("/")]
    back = [n for n in names if "\\" in n]
    if back:
        res.add("FAIL", "zip-separators", "%d member name(s) use backslashes (zip spec: '/'); e.g. %s" % (len(back), back[0]))
    else:
        res.add("PASS", "zip-separators", "all member names use '/'")
    norm = [n.replace("\\", "/") for n in names]
    escape = [n for n in norm if n.startswith("/") or n.startswith("../") or "/../" in "/" + n or re.match(r"^[A-Za-z]:", n)]
    if escape:
        res.add("FAIL", "zip-paths", "%d member(s) escape the extract dir" % len(escape))
    tops = sorted(set(n.split("/", 1)[0] for n in norm))
    if tops != [ENGINE_TOP]:
        res.add("FAIL", "zip-top", "top-level entries %s, expected only %s/" % (tops[:5], ENGINE_TOP))
    else:
        res.add("PASS", "zip-top", "single top folder %s/" % ENGINE_TOP)
    rel = [n.split("/", 1)[1] if "/" in n else n for n in norm]
    top_files = set(r for r in rel if "/" not in r)
    for need in TOP_FILES_REQUIRED:
        if need in top_files:
            res.add("PASS", "zip-member", "%s/%s" % (ENGINE_TOP, need))
        else:
            res.add("FAIL", "zip-member", "%s/%s missing (MIT notice and third-party notices must ship in the zip)" % (ENGINE_TOP, need))
    if not no_pxqn:
        ok = PXQN_TERMS_NAME in top_files
        res.add("PASS" if ok else "FAIL", "zip-member", "%s/%s%s" % (ENGINE_TOP, PXQN_TERMS_NAME,
                "" if ok else " missing (the PXQN library terms go next to LICENSE)"))
    bins = set(r[4:] for r in rel if r.startswith("bin/") and "/" not in r[4:])
    for need in REQUIRED_BIN:
        res.add("PASS" if need in bins else "FAIL", "zip-member", "bin/%s%s" % (need, "" if need in bins else " missing"))
    others = [r for r in rel if "/" in r and not r.startswith("bin/")]
    if others:
        res.add("WARN", "zip-layout", "%d file(s) outside bin/ and the top files, e.g. %s" % (len(others), others[0]))
    for r in rel:
        base = r.rsplit("/", 1)[-1]
        low = base.lower()
        ext = os.path.splitext(low)[1]
        if ext in REFUSE_EXT:
            res.add("FAIL", "zip-refused", "%s (source/build/debug file)" % r)
        if os.path.splitext(low)[0] in DRIVER_DLLS or (low.endswith(".dll") and "nvidia" in low):
            res.add("FAIL", "zip-refused", "%s (NVIDIA driver file must not ship)" % r)
        if ".old" in low:
            res.add("FAIL", "zip-refused", "%s (old copy)" % r)
        if any(fnmatch.fnmatch(low, g) for g in SECRET_NAMES):
            res.add("FAIL", "zip-refused", "%s (secret-like file name)" % r)
        if low == PXQN_NAME and not bundle:
            res.add("FAIL", "zip-refused", "%s inside the zip without --bundle-pxqn" % r)
    if bundle and PXQN_NAME not in bins:
        res.add("FAIL", "zip-member", "bin/%s missing (--bundle-pxqn)" % PXQN_NAME)
    return z


def notice_text_ok(res, text, where):
    low = text.lower()
    need = (("nvidia", "the CUDA runtime DLLs (NVIDIA CUDA EULA)"), ("microsoft", "the Visual C++ runtime DLLs"))
    for word, what in need:
        if word in low:
            res.add("PASS", "notice", "%s names %s" % (where, what))
        else:
            res.add("FAIL", "notice", "%s does not name %s, which the zip redistributes" % (where, what))


def check_notes(res, guards, text, where):
    hits = guards.check_notes_text(text)
    store = len(NOTES_STORE.findall(text))
    if store:
        hits.append("store-word")
    if hits:
        res.add("FAIL", "notes-text", "%s: %s" % (where, ",".join(sorted(set(hits)))))
    else:
        res.add("PASS", "notes-text", "%s: SmartScreen + Run anyway present; no switch/lane/path/store words" % where)
    if SUMS_NAME not in text:
        res.add("WARN", "notes-text", "%s does not mention %s" % (where, SUMS_NAME))


def check_pe_file(res, label, data, bin_names, archs, want_fatbin, fat_dir, ptx_req=()):
    """Machine, import closure, fatbin archs, string rules. Returns PE or None."""
    try:
        pe = PE(data)
    except (ValueError, struct.error) as e:
        res.add("FAIL", "pe-format", "%s: %s" % (label, e))
        return None
    if pe.machine != 0x8664 or pe.magic != 0x20B:
        res.add("FAIL", "pe-format", "%s: machine 0x%x magic 0x%x (want x64 PE32+)" % (label, pe.machine, pe.magic))
    try:
        imps = pe.imports()
    except (ValueError, struct.error) as e:
        res.add("FAIL", "pe-imports", "%s: unreadable import table (%s)" % (label, e))
        imps = []
    if bin_names is not None:
        missing = []
        for dll in imps:
            low = dll.lower()
            stem = low[:-4] if low.endswith(".dll") else low
            if low in bin_names or stem in SYSTEM_DLLS or stem in DRIVER_DLLS or low.startswith(SYSTEM_PREFIXES):
                continue
            missing.append(dll)
        if missing:
            res.add("FAIL", "pe-imports", "%s imports %s, not in bin/ and not a Windows system DLL" % (label, ",".join(sorted(set(missing)))))
    blob = pe.section(".nv_fatb")
    if blob is not None:
        try:
            ents = fatbin_entries(blob)
        except (ValueError, struct.error) as e:
            res.add("FAIL", "cuda-arch", "%s: unreadable fatbin (%s)" % (label, e))
            ents = []
        sass = sorted(set(sm for _i, k, sm in ents if k == 2))
        ptx = sorted(set(sm for _i, k, sm in ents if k == 1))
        nfat = len(set(i for i, _k, _s in ents))
        vendor = is_vendor(label)
        lack = arch_lack(archs, sass, vendor)
        if lack:
            res.add("FAIL", "cuda-arch", "%s: SASS %s, missing sm_%s%s" % (label, sass, ",sm_".join(map(str, lack)),
                                                                       " (NVIDIA lib: no same-major SASS either)" if vendor else ""))
        elif vendor and [a for a in archs if a not in sass]:
            res.add("PASS", "cuda-arch", "%s: NVIDIA lib, SASS sm_%s; sm_%s run its same-major older SASS natively" % (
                label, ",sm_".join(map(str, sass)), ",sm_".join(str(a) for a in archs if a not in sass)))
        else:
            res.add("PASS", "cuda-arch", "%s: %d fatbins, SASS sm_%s, PTX %s" % (label, nfat, ",sm_".join(map(str, sass)),
                                                                         ",".join("compute_%d" % p for p in ptx) or "none"))
        per = {}
        for i, k, sm in ents:
            if k == 2:
                per.setdefault(i, set()).add(sm)
        thin = [i for i in range(nfat) if not set(archs) <= per.get(i, set())]
        if thin and not vendor:
            res.add("WARN", "cuda-arch", "%s: %d of %d fatbins lack SASS for some of sm_%s" % (label, len(thin), nfat, ",sm_".join(map(str, archs))))
        nojit = [p for p in ptx_req if p not in ptx]
        if nojit and not vendor:
            res.add("FAIL", "cuda-arch", "%s: PTX %s, missing compute_%s (GPUs newer than sm_%d cannot JIT it)"
                    % (label, ",".join("compute_%d" % p for p in ptx) or "none", ",compute_".join(map(str, nojit)), max(archs)))
        elif not ptx and not vendor:
            res.add("WARN", "cuda-arch", "%s: no PTX; GPUs newer than sm_%d cannot run it (notes must say which cards)" % (label, max(archs)))
        if fat_dir is not None:
            fb = first_fatbin(blob)
            if fb:
                if vendor:
                    VENDOR_FATBINS.add(label.replace("/", "_") + ".fatbin")
                with open(os.path.join(fat_dir, label.replace("/", "_") + ".fatbin"), "wb") as f:
                    f.write(fb)
    elif want_fatbin:
        res.add("FAIL", "cuda-arch", "%s: no .nv_fatb section (no device code)" % label)
    scan = data
    span = pe.section_span(".nv_fatb")
    if span is not None:
        packed = compressed_payload_spans(data[span[0]:span[1]])
        if packed:
            buf = bytearray(data)
            for a, b in packed:
                buf[span[0] + a:span[0] + b] = bytes(b - a)
            scan = bytes(buf)
    hits, _ = rule_hits(scan)
    if hits:
        res.add("FAIL", "strings", "%s: %s" % (label, ", ".join("%s x%d" % kv for kv in sorted(hits.items()))))
    return pe


def check_exports(res, label, pe, def_names, expect, allow_hooks=False):
    try:
        names = pe.exports()
    except (ValueError, struct.error) as e:
        res.add("FAIL", "pxqn-exports", "%s: unreadable export table (%s)" % (label, e))
        return
    allowed = set(def_names) | (set(PXQN_HOOKS) if allow_hooks else set())
    missing = [n for n in def_names if n not in names]
    extra = [n for n in names if n not in allowed]
    if missing or extra or len(names) != expect:
        hooks = sum(1 for n in extra if n in PXQN_HOOKS)
        res.add("FAIL", "pxqn-exports", "%s: %d exports (want %d), %d .def name(s) missing, %d unexpected%s" % (
            label, len(names), expect, len(missing), len(extra),
            " (%d gu_* test hook(s): a pre-fix or test build)" % hooks if hooks else ""))
    else:
        res.add("PASS", "pxqn-exports", "%s: exactly %d exports (.def %d + hooks %d)" % (
            label, len(names), len(def_names), len(names) - len(def_names)))


def check_sums(res, base_dir, sums, shipped):
    if not sums or not os.path.isfile(sums):
        res.add("FAIL", "sha256sums", "%s missing next to the downloads" % SUMS_NAME)
        return
    listed = {}
    for line in read_text(sums).splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if not m:
            res.add("FAIL", "sha256sums", "malformed line in %s" % SUMS_NAME)
            continue
        listed[m.group(2)] = m.group(1).lower()
    for name, want in sorted(listed.items()):
        path = os.path.join(os.path.dirname(sums), name)
        if not os.path.isfile(path):
            res.add("FAIL", "sha256sums", "%s listed but not present" % name)
        elif sha256_file(path) != want:
            res.add("FAIL", "sha256sums", "%s does not match %s" % (name, SUMS_NAME))
        else:
            res.add("PASS", "sha256sums", "%s matches" % name)
    for path in shipped:
        if os.path.basename(path) not in listed:
            res.add("FAIL", "sha256sums", "%s ships but is not listed" % os.path.basename(path))


def run_release_scan(res, label, target, blocklist, work, jobs):
    scan = os.path.join(REPO, "scripts", "release-scan.py")
    os.makedirs(work, exist_ok=True)
    out_json = os.path.join(work, "scan-%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", label))
    cmd = [sys.executable, "-I", scan, "--blocklist", blocklist, "--work", work, "--jobs", str(jobs),
           "--json", out_json, target]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = [l for l in p.stdout.splitlines() if l.strip()]
    parts = []
    try:
        with open(out_json, encoding="utf-8") as f:
            rep = list(json.load(f).values())[0]
        for cat, v in sorted(rep.get("blocks", {}).items(), key=lambda kv: -kv[1].get("hits", 0)):
            files = sorted(v.get("by_file", {}).items(), key=lambda kv: -kv[1])
            parts.append("%s %d hit(s) in %s" % (cat, v.get("hits", 0), ", ".join("%s x%d" % fv for fv in files[:4]) +
                                                 (" +%d more" % (len(files) - 4) if len(files) > 4 else "")))
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    summary = "; ".join(parts) or "; ".join(l.strip() for l in tail if "PASS" in l or "FAIL" in l)[:300]
    if p.returncode == 0:
        res.add("PASS", "scan-gate", "%s: %s" % (label, summary or "clean"))
    elif p.returncode == 2:
        res.add("FAIL", "scan-gate", "%s: %s" % (label, summary))
    else:
        res.add("FAIL", "scan-gate", "%s: gate could not run (rc %d): %s" % (label, p.returncode, (tail[-1] if tail else "")[:200]))


def cuobjdump_crosscheck(res, fat_dir, archs, image):
    files = sorted(f for f in os.listdir(fat_dir) if f.endswith(".fatbin"))
    if not files:
        return
    if not shutil.which("docker"):
        res.add("MANUAL", "cuobjdump", "docker not found; run cuobjdump --list-elf on the extracted fatbins in %s" % fat_dir)
        return
    for f in files:
        cmd = ["docker", "run", "--rm", "--network", "none", "--cpuset-cpus", os.environ.get("CHECK_WIN_CPUS", "36-53"),
               "-v", "%s:/w:ro" % fat_dir, image, "cuobjdump", "--list-elf", "/w/" + f]
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            res.add("WARN", "cuobjdump", "%s: timed out" % f)
            continue
        got = sorted(set(int(x) for x in re.findall(r"\.sm_(\d+)\.cubin", p.stdout)))
        lack = arch_lack(archs, got, f in VENDOR_FATBINS)
        res.add("FAIL" if lack else "PASS", "cuobjdump", "%s (first fatbin): cuobjdump lists SASS sm_%s%s" % (
            f, ",sm_".join(map(str, got)) or "none", " (NVIDIA lib: same-major rule)" if f in VENDOR_FATBINS else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check-win-package")
    ap.add_argument("package", help="pxa-engine-win64-cuda12.zip or the release directory that holds it")
    ap.add_argument("--control", default="")
    ap.add_argument("--pxqn", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--sums", default="")
    ap.add_argument("--no-pxqn", action="store_true", help="PXQ-only package: no library beside the zip")
    ap.add_argument("--bundle-pxqn", action="store_true", help="the library is also inside the zip")
    ap.add_argument("--expect-exports", type=int, default=0, help="ggml-pxqn.dll export count (default: the .def names)")
    ap.add_argument("--allow-test-hooks", action="store_true", help="accept the 3 gu_* test-hook exports (old or test build)")
    ap.add_argument("--archs", default=",".join(map(str, DEFAULT_ARCHS)))
    ap.add_argument("--ptx", default=",".join(map(str, DEFAULT_PTX)), help="required PTX archs (comma list; \"\" = none)")
    ap.add_argument("--blocklist", default="")
    ap.add_argument("--ctest-log", default="", help="the Windows ctest output (100%% tests passed line)")
    ap.add_argument("--cuobjdump", action="store_true", help="cross-check the fatbins with cuobjdump in docker")
    ap.add_argument("--cuobjdump-image", default=CUOBJDUMP_IMAGE)
    ap.add_argument("--work", default="")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    pkg = os.path.abspath(a.package)
    if os.path.isdir(pkg):
        zpath = os.path.join(pkg, ZIP_NAME)
    else:
        zpath = pkg
    if not os.path.isfile(zpath):
        print("check-win-package: no engine zip at %s" % zpath, file=sys.stderr)
        return 3
    base = os.path.dirname(zpath)
    control = a.control or os.path.join(base, CONTROL_NAME)
    pxqn = "" if a.no_pxqn else (a.pxqn or os.path.join(base, PXQN_NAME))
    notes = a.notes or os.path.join(base, NOTES_NAME)
    sums = a.sums or os.path.join(base, SUMS_NAME)
    terms = "" if a.no_pxqn else os.path.join(base, PXQN_TERMS_NAME)
    archs = tuple(int(x) for x in a.archs.split(",") if x.strip())
    ptx_req = tuple(int(x) for x in a.ptx.split(",") if x.strip())
    blocklist = a.blocklist or os.environ.get("PXA_RELEASE_BLOCKLIST", "") or (DEFAULT_BLOCKLIST if os.path.isfile(DEFAULT_BLOCKLIST) else "")
    guards = load_module(os.path.join(REPO, "scripts", "windows-release-guards.py"), "win_guards")
    def_names = guards.def_export_names(os.path.join(REPO, DEF_REL))
    expect = a.expect_exports or (len(def_names) + (len(PXQN_HOOKS) if a.allow_test_hooks else 0))

    work_root = a.work or tempfile.gettempdir()
    os.makedirs(work_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="check-win-", dir=work_root)
    fat_dir = os.path.join(work, "fatbins")
    os.makedirs(fat_dir)
    res = Result()
    try:
        res.add("INFO", "input", "zip=%s control=%s pxqn=%s" % (zpath, control if os.path.isfile(control) else "-",
                                                               pxqn if pxqn and os.path.isfile(pxqn) else "-"))
        # sha256 list first: the exact files every later line is about
        shipped = [p for p in (control, zpath, pxqn, terms, notes) if p and os.path.isfile(p)]
        if terms and not os.path.isfile(terms):
            res.add("FAIL", "pxqn-terms", "%s missing beside %s" % (PXQN_TERMS_NAME, PXQN_NAME))
        elif terms:
            hits = NOTES_STORE.findall(read_text(terms))
            res.add("FAIL" if hits else "PASS", "pxqn-terms", "%s beside the library%s" % (
                PXQN_TERMS_NAME, ": store words %d" % len(hits) if hits else ""))
        for p in shipped:
            res.add("INFO", "sha256", "%s  %s" % (sha256_file(p), os.path.basename(p)))
        for p in shipped:
            side = p + ".sha256"
            if os.path.isfile(side):
                want = (read_text(side).split() or [""])[0].lower()
                res.add("PASS" if want == sha256_file(p) else "FAIL", "sha256-sidecar", "%s vs %s" % (os.path.basename(p), os.path.basename(side)))

        z = check_zip_layout(res, zpath, a.bundle_pxqn, a.no_pxqn)
        if z is None:
            return 1
        names = {}
        for info in z.infolist():
            n = info.filename.replace("\\", "/")
            if not n.endswith("/"):
                names[n] = info
        bin_names = set(n.rsplit("/", 1)[-1].lower() for n in names if "/bin/" in "/" + n)
        top_text = {}
        for n, info in names.items():
            rel = n.split("/", 1)[1] if "/" in n else n
            if rel in TOP_FILES_REQUIRED:
                top_text[rel] = z.read(info).decode("utf-8", "replace")
        if "NOTICE" in top_text:
            notice_text_ok(res, top_text["NOTICE"], "zip NOTICE")
        if "README.txt" in top_text:
            check_notes(res, guards, top_text["README.txt"], "zip README.txt")

        cuda_dlls = {"ggml.dll", "ggml-cuda.dll"}
        seen_cuda = False
        for n in sorted(names):
            low = n.lower()
            if not (low.endswith(".exe") or low.endswith(".dll")):
                continue
            data = z.read(names[n])
            base_l = low.rsplit("/", 1)[-1]
            want = base_l in ("ggml-cuda.dll",)
            pe = check_pe_file(res, n, data, bin_names, archs, want, fat_dir, ptx_req)
            if pe is not None and pe.section(".nv_fatb") is not None and base_l in cuda_dlls:
                seen_cuda = True
            if pe is not None and base_l == PXQN_NAME:
                check_exports(res, n, pe, def_names, expect, a.allow_test_hooks)
            del data
        if not seen_cuda:
            res.add("FAIL", "cuda-arch", "neither ggml.dll nor ggml-cuda.dll in the zip carries device code")
        res.add("PASS" if res.failed() == 0 else "INFO", "pe-scan", "%d PE files in the zip checked (imports, arch, strings)" %
                sum(1 for n in names if n.lower().endswith((".exe", ".dll"))))

        # the library beside the zip
        if pxqn:
            if not os.path.isfile(pxqn):
                res.add("FAIL", "pxqn", "%s missing beside the zip (use --no-pxqn for a PXQ-only package)" % PXQN_NAME)
            else:
                data = read_bytes(pxqn)
                pe = check_pe_file(res, PXQN_NAME, data, bin_names | {PXQN_NAME}, archs, True, fat_dir, ptx_req)
                if pe is not None:
                    check_exports(res, PXQN_NAME, pe, def_names, expect, a.allow_test_hooks)
                del data

        # Control exe: the PE itself, then its PyInstaller payload (compressed: a plain strings scan cannot see it)
        ctl_dir = os.path.join(work, "control")
        if not os.path.isfile(control):
            res.add("FAIL", "control", "%s missing" % CONTROL_NAME)
        else:
            data = read_bytes(control)
            check_pe_file(res, CONTROL_NAME, data, None, archs, False, None)
            try:
                ents = pyinstaller_extract(data, ctl_dir)
            except (ValueError, struct.error, zlib.error, EOFError, TypeError) as e:
                res.add("FAIL", "control-payload", "PyInstaller payload unreadable (%s)" % e)
                ents = []
            if ents is None:
                res.add("INFO", "control-payload", "not a PyInstaller exe; the PE strings scan above covers it")
            elif ents:
                ours = sum(1 for e in ents if e[2])
                res.add("INFO", "control-payload", "%d entries unpacked (%d ours)" % (len(ents), ours))
                bad = {}
                for rel, path, _mine in ents:
                    with open(path, "rb") as f:
                        hits, _ = rule_hits(f.read())
                    for rid, cnt in hits.items():
                        bad.setdefault(rid, []).append(rel)
                for rid, rels in sorted(bad.items()):
                    res.add("FAIL", "control-strings", "%s in %d bundled file(s), e.g. %s" % (rid, len(rels), rels[0]))
                if not bad:
                    res.add("PASS", "control-strings", "no lane/private-path/private-repo/secret shapes in the bundled files")
                # the block list itself runs on the exe through release-scan (beside-the-zip below), which unpacks
                # the same payload (check 4 of release-scan.py)
            del data

        if not os.path.isfile(notes):
            res.add("FAIL", "notes", "%s missing" % NOTES_NAME)
        else:
            check_notes(res, guards, read_text(notes), NOTES_NAME)
            if "README.txt" in top_text and top_text["README.txt"].strip() != read_text(notes).strip():
                res.add("WARN", "notes", "zip README.txt differs from %s" % NOTES_NAME)

        check_sums(res, base, sums, shipped)

        # clean-package gate (block list: public text, secret names, strings of every ELF/PE) on the exact files
        if not blocklist or not os.path.isfile(blocklist):
            res.add("FAIL", "scan-gate", "no block list (PXA_RELEASE_BLOCKLIST / --blocklist): the gate fails closed")
        else:
            run_release_scan(res, "engine-zip", zpath, blocklist, os.path.join(work, "scan"), a.jobs)
            side = os.path.join(work, "beside")
            os.makedirs(side)
            for p in (control, pxqn, terms, notes, sums):
                if p and os.path.isfile(p):
                    os.link(p, os.path.join(side, os.path.basename(p))) if os.stat(p).st_dev == os.stat(side).st_dev \
                        else shutil.copyfile(p, os.path.join(side, os.path.basename(p)))
            run_release_scan(res, "beside-the-zip", side, blocklist, os.path.join(work, "scan"), a.jobs)

        if a.cuobjdump:
            cuobjdump_crosscheck(res, fat_dir, archs, a.cuobjdump_image)
        else:
            res.add("MANUAL", "cuobjdump", "optional cross-check: rerun with --cuobjdump (docker, no GPU)")

        if a.ctest_log:
            log = read_text(a.ctest_log) if os.path.isfile(a.ctest_log) else ""
            m = re.search(r"(\d+)% tests passed, (\d+) tests? failed out of (\d+)", log)
            if m and m.group(1) == "100" and m.group(2) == "0":
                res.add("PASS", "ctest", "100%% of %s tests passed" % m.group(3))
            else:
                res.add("FAIL", "ctest", "no '100%% tests passed, 0 tests failed' line in %s" % a.ctest_log)
        else:
            res.add("MANUAL", "ctest", "Windows ctest must be 100%: pass the log with --ctest-log")
        res.add("MANUAL", "smoke", "clean Windows box: unzip as engine\\, copy ggml-pxqn.dll to engine\\bin, Control boots the engine, one PXQN chat")
        res.add("MANUAL", "smartscreen", "downloaded copy (Mark of the Web) shows 'Windows protected your PC' -> More info -> Run anyway, as the notes say")
        res.add("MANUAL", "pxqn-terms", "%s wording approved by the maintainers (first version 2026-10-19)" % PXQN_TERMS_NAME)
    finally:
        if a.keep:
            print("check-win-package: work kept at %s" % work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    nf = res.failed()
    print("check-win-package: %s (%d FAIL, %d WARN, %d MANUAL)" % ("PASS" if nf == 0 else "FAIL", nf,
          sum(1 for r in res.rows if r["status"] == "WARN"), sum(1 for r in res.rows if r["status"] == "MANUAL")))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"verdict": "PASS" if nf == 0 else "FAIL", "rows": res.rows}, f, indent=1)
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
