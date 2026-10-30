#!/usr/bin/env python3
"""release-scan.py - the release package scan gate: nothing ships until this passes on the exact files.

    PXA_RELEASE_BLOCKLIST=/path/to/blocklist  scripts/release-scan.py [--json OUT] [--work DIR] INPUT...

INPUT is an unpacked package directory, a package tarball (.tar.gz / .tgz / .tar), a zip archive, a
`docker save` archive (plain or gzipped), or one bare file (a program, a .dll, a data file). Exit 0 = clean; 2 = at least one BLOCK hit (no package, no upload); 3 = the gate could not
run (no block list, unreadable input, self-test failed). The gate FAILS CLOSED: an unset or missing block list
is exit 3, never a pass.

Three checks, all on the exact files that would ship:
  1. text: every text file against the block list's categories (regexes and literal id lists);
  2. secrets: token shapes on all text and binary strings, plus the literal contents of the secret files the
     block list names (directory given there, never here). A secret hit prints the FILE NAME and the rule id
     only, never the matched value;
  3. binaries: the printable strings (runs of >= 4 bytes, what `strings -a` prints) of every ELF file and
     every PE file (Windows .exe / .dll, MZ + PE signature), and of every other binary data file, against the
     categories marked scope=binary (build paths, private names).
  4. onefile payloads: a binary that carries a compressed program inside it (a PyInstaller archive, a Nuitka
     onefile zstd payload on Windows / macOS / Linux, a zip or 7z archive appended to an executable or
     self-extractor, any zip found by its magic bytes whatever its name, a .zst file) is
     unpacked into --work and every file inside is scanned with the same rules, named `<binary>!/<entry>`;
     inside a payload, binary entries (compiled Python, data blobs) get the strings pass too and nested zips
     are opened (depth 3); there, compiled Python whose zip signatures all sit in bytes constants of its code and
     that does not open as a zip with members (the stdlib's zipimport.pyc) is scanned as bytes, not unpacked. A payload that
     is found but cannot be unpacked (corrupt, no `zstd` tool, over the size budget) stops the gate (exit 3): fail
     closed. Mach-O files (macOS) get the strings pass like ELF/PE.
plus file rules (file-name patterns that must not ship at all, e.g. compiled-language sources); inside a
payload the compiled-Python names (*.pyc, *.pyo) a onefile build always carries are exempt from the file rules.

The block list is private and lives outside this repository: its words are exactly what must not be public.
Its format (one directive per line, `#` comments):
  @category <id> [scope=text,binary] [flags=i] [ext=.md,.txt] [allow-files=GLOB,GLOB]   then regex lines
  @ids <id> [scope=text,binary] [allow-files=...]          then literal tokens, one per line (exact token match)
  @files <id>                                               then fnmatch globs (package-relative paths)
  @allow-phrase <text>        masked before matching (a public name that contains a blocked word)
  @allow-switch <NAME>        a documented switch name (also: every row name of a shipped lever catalog)
  @secrets-dir <dir>          the secret files whose literal contents must not appear (names-only report)
  @secrets-skip <regex>       secret files (relative name) that are public by nature (public keys)
  @secrets-allow <regex> GLOB[,GLOB]   that secret file's literals are allowed in these package files only (a record
                              of text that was already published, e.g. a provenance record of shipped tables)
  @archive-allow GLOB[,GLOB]  a binary archive at this package path is a known upstream file (e.g. a build wrapper jar,
                              a prebuilt web bundle): reported as a warning instead of a block (its inside is not read)
  @image-skip <prefix>        container paths never scanned (toolkits the base image installs)
  @selftest <category> <sample>   a sample the gate MUST catch, or it fails closed
  @include <path>             another block list file (relative to this one)
Hits inside a documented switch name (PXA_*, PXQ_*, GGML_*, LLAMA_*) are not hits: the switch list a user reads
is public by design. Stdlib only.
"""
import argparse, bz2, fnmatch, gzip, io, lzma, json, marshal, mmap, os, re, shutil, stat, struct, subprocess, sys, tarfile, \
    tempfile, zipfile, zlib

TEXT_LIMIT = 16 << 20
ELF_MAGIC = b"\x7fELF"
MACHO_MAGICS = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce")
FAT_MAGIC = b"\xca\xfe\xba\xbe"
PYI_COOKIE = b"MEI\x0c\x0b\x0a\x0b\x0e"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
NUITKA_ZSTD = b"KAY" + ZSTD_MAGIC          # Nuitka onefile: "KA" + "Y" (zstd-compressed) + the zstd frame
SEVENZ_MAGIC = b"7z\xbc\xaf\x27\x1c"        # 7z archive or a 7z self-extractor (signature after the stub)
ZIP_SIGNATURES = (b"PK\x01\x02", b"PK\x03\x04", b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07")
MARSHAL_BYTES4 = (b"s\x04\0\0\0", b"\xf3\x04\0\0\0")    # compiled Python: a 4-byte bytes constant follows (plain, FLAG_REF)
PAYLOAD_DEPTH = 3
PAYLOAD_BUDGET = 8 << 30                    # bytes unpacked per input, all payloads together
PAYLOAD_SKIP_FILE_RULES = ("*.pyc", "*.pyo")
_PAYLOAD = {"work": None, "bytes": 0}
# Plugin payloads never ship inside a package (owner 2026-10-21: beta 3 ships no plugin, Anibal included; plugins reach a
# computer only as a signed .pxp through the grant-bound Store download). Built in, so it holds without the block list.
PLUGIN_CAT = "plugin-payload"
PLUGIN_NAME_GLOBS = ("*.pxp", "plugin.src.json", "plugin.json")
PLUGIN_PATH_RX = re.compile(r"(?:^|/|!/)(?:tools/(?:anibal|vigil|pxa_plugins|pxa-private)/"
                            r"|(?:PYZ/)?(?:anibal|pxa_plugins)/(?:__init__\.py[co]?$|.*\.py[co]?$))", re.I)
PLUGIN_MODULE_RX = re.compile(r"\b(?:anibal|pxa_plugins)\.(?:core|server|voice|llm|memory|fastpath|laya|ops|remote|tools|ui|"
                              r"bench|host|delivery|entitlement|consent|bridge_http|community|packs|vigil|sdk|cli|common|handoff)\b")
ZIP_MAGIC = b"PK\x03\x04"


def plugin_payload_name(rel, f):
    """The file name or path of a plugin payload (a package, a plugin manifest, Anibal / plugin framework code)."""
    if any(fnmatch.fnmatchcase(f.lower(), g) for g in PLUGIN_NAME_GLOBS):      # any case: Plugin.json, ANIBAL.PXP
        return f
    m = PLUGIN_PATH_RX.search(rel)
    return m.group(0).lstrip("!/")[:80] if m else None


def zip_has_plugin(path):
    """A zip of any name whose members carry a plugin manifest (a .pxp under another name)."""
    import zipfile
    try:
        with open(path, "rb") as f:
            if f.read(4) != ZIP_MAGIC:
                return False
        with zipfile.ZipFile(path) as z:
            return any(os.path.basename(n).lower() in ("plugin.json", "plugin.src.json") for n in z.namelist())
    except (OSError, zipfile.BadZipFile, ValueError):
        return False
STRINGS_RX = re.compile(rb"[\x20-\x7e\t]{4,}")
SWITCH_RX = re.compile(r"\b(?:PXA|PXQ|GGML|LLAMA)_[A-Z0-9_]*[A-Z0-9]\b")
CATALOG_ROW = re.compile(r'^\{ "([A-Z][A-Z0-9_]*)"', re.M)
CATALOG_NAMES = ("pxa-lever-catalog.inc",)
TOKEN_SPLIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*[A-Za-z0-9]")
SECRET_SHAPES = [
    ("secret-shape-github", r"\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"),
    ("secret-shape-apikey", r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{30,}|\bhf_[A-Za-z0-9]{30,}|\bnvapi-[A-Za-z0-9_-]{30,}"),
    ("secret-shape-aws", r"\bAKIA[0-9A-Z]{16}\b"),
    ("secret-shape-pem", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    ("secret-shape-slack", r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    ("secret-shape-jwt", r"\beyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"),
    ("secret-shape-discord", r"\b[MN][A-Za-z\d]{23,25}\.[\w-]{6}\.[\w-]{27,}"),
]
SECRET_SHAPES = [(n, re.compile(p)) for n, p in SECRET_SHAPES]


class GateError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ block list
class Rules:
    def __init__(self):
        self.cats = []          # dict(id, kind=regex|ids, rx|set, scope, ext, allow)
        self.files = []         # (id, [globs])
        self.allow_phrases = []
        self.allow_switch = set()
        self.secrets_dir = None
        self.secrets_skip = []
        self.secrets_allow = []  # (secret-name regex, [file globs])
        self.archive_allow = []  # package-path globs of known upstream binary archives
        self.image_skip = []
        self.selftests = []


def load_rules(path, rules=None, seen=None):
    rules = rules or Rules()
    seen = seen or set()
    path = os.path.abspath(path)
    if path in seen:
        return rules
    seen.add(path)
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:
        raise GateError("block list unreadable: %s (%s)" % (os.path.basename(path), e.strerror))
    cur = None
    for n, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("@"):
            head, _, rest = line.partition(" ")
            rest = rest.strip()
            cur = None
            if head in ("@category", "@ids"):
                parts = rest.split()
                if not parts:
                    raise GateError("%s:%d: %s needs an id" % (os.path.basename(path), n, head))
                opts = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
                cur = {"id": parts[0], "kind": "regex" if head == "@category" else "ids",
                       "scope": set((opts.get("scope") or "text").split(",")),
                       "flags": re.I if "i" in opts.get("flags", "") else 0,
                       "ext": tuple(e for e in opts.get("ext", "").split(",") if e),
                       "allow": [g for g in opts.get("allow-files", "").split(",") if g],
                       "pats": [], "ids": set()}
                rules.cats.append(cur)
            elif head == "@files":
                cur = {"id": rest.split()[0], "kind": "files", "globs": []}
                rules.files.append(cur)
            elif head == "@allow-phrase":
                rules.allow_phrases.append(rest)
            elif head == "@allow-switch":
                rules.allow_switch.update(rest.split())
            elif head == "@secrets-dir":
                rules.secrets_dir = rest
            elif head == "@secrets-skip":
                rules.secrets_skip.append(re.compile(rest))
            elif head == "@secrets-allow":
                rx, _, globs = rest.partition(" ")
                globs = [g for g in globs.strip().split(",") if g]
                if not rx or not globs:
                    raise GateError("%s:%d: @secrets-allow needs a secret-name regex and file globs" % (os.path.basename(path), n))
                rules.secrets_allow.append((re.compile(rx), globs))
            elif head == "@archive-allow":
                globs = [g for g in rest.split(",") if g]
                if not globs:
                    raise GateError("%s:%d: @archive-allow needs file globs" % (os.path.basename(path), n))
                rules.archive_allow.extend(globs)
            elif head == "@image-skip":
                rules.image_skip.append(rest.rstrip("/"))
            elif head == "@selftest":
                cat, _, sample = rest.partition(" ")
                rules.selftests.append((cat, sample.encode().decode("unicode_escape") if sample.isascii() else sample))
            elif head == "@include":
                load_rules(os.path.join(os.path.dirname(path), rest), rules, seen)
            else:
                raise GateError("%s:%d: unknown directive %s" % (os.path.basename(path), n, head))
            continue
        if cur is None:
            raise GateError("%s:%d: a pattern line outside any section" % (os.path.basename(path), n))
        if cur["kind"] == "regex":
            try:
                cur["pats"].append(re.compile(line.strip(), cur["flags"]))
            except re.error as e:
                raise GateError("%s:%d: bad regex (%s)" % (os.path.basename(path), n, e))
        elif cur["kind"] == "ids":
            cur["ids"].add(line.strip().lower())
        else:
            cur["globs"].append(line.strip())
    for c in rules.cats:
        if c["kind"] == "regex":
            c["rx"] = re.compile("|".join("(?:%s)" % p.pattern for p in c["pats"]), c["flags"]) if c["pats"] else None
    return rules


# ------------------------------------------------------------------------------------------------ secrets
def secret_tokens(rules):
    """{token: secret file name}. Tokens = 20..400-char runs without spaces, as whole lines and split on quote / = / :."""
    out = {}
    d = rules.secrets_dir
    if not d:
        return out
    if not os.path.isdir(d):
        raise GateError("the secrets directory the block list names is not readable here")
    for dp, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, d)
            if any(r.search(rel) for r in rules.secrets_skip):
                continue
            try:
                if os.path.getsize(p) > 65536:
                    continue
                data = open(p, "rb").read().decode("utf-8", "ignore")
            except OSError:
                continue
            for line in data.splitlines():
                for tok in [line.strip()] + re.split(r"[\s\"'=:,;]+", line):
                    tok = tok.strip()
                    if not (20 <= len(tok) <= 400) or " " in tok or tok.startswith("-----"):
                        continue
                    if re.fullmatch(r"[0-9.,eE+\-x()fFuUlL{}]+", tok):        # numeric tables
                        continue
                    if re.fullmatch(r"(?:https?://)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:[/:][^\s]*)?", tok) and not re.search(r"[A-Za-z0-9]{24,}", tok):
                        continue                                                 # a host name / URL without a key in it
                    out.setdefault(tok, rel)
    return out


def secret_hits(text, toks):
    if not toks:
        return []
    hits = []
    cands = set(re.split(r"[\s\"'`=:,;()\[\]{}<>]+", text))
    cands.update(text.splitlines())
    for c in cands:
        c = c.strip()
        if c in toks:
            hits.append(toks[c])
    return hits


# ------------------------------------------------------------------------------------------------ scanning
class Report:
    def __init__(self, label):
        self.label = label
        self.n_text = self.n_elf = self.n_pe = self.n_other = self.n_strings = 0
        self.n_macho = self.n_payloads = self.n_payload_files = 0
        self.hits = {}          # cat -> {file: [count, examples]}
        self.warn = {}

    def add(self, cat, f, example, secret=False, warn=False):
        book = self.warn if warn else self.hits
        e = book.setdefault(cat, {}).setdefault(f, [0, []])
        e[0] += 1
        if not secret and example is not None and len(e[1]) < 3 and example not in e[1]:
            e[1].append(example)

    def total(self):
        return sum(c for files in self.hits.values() for c, _ in files.values())


def masked(text, rules, switches):
    for ph in rules.allow_phrases:
        if ph in text:
            text = text.replace(ph, " " * len(ph))
    if switches:
        text = SWITCH_RX.sub(lambda m: " " * len(m.group(0)) if m.group(0) in switches else m.group(0), text)
    return text


def scan_blob(rep, rel, text, kind, rules, switches, stoks):
    """kind = 'text' or 'binary' (the joined strings of an ELF or PE file)."""
    t = masked(text, rules, switches)
    base_ext = os.path.splitext(rel)[1]
    for c in rules.cats:
        if kind not in c["scope"]:
            continue
        if c["ext"] and base_ext not in c["ext"]:
            continue
        if any(fnmatch.fnmatch(rel, g) for g in c["allow"]):
            continue
        if c["kind"] == "regex":
            if c["rx"] is None:
                continue
            for m in c["rx"].finditer(t):
                rep.add(c["id"], rel, m.group(0)[:80])
        else:
            for m in TOKEN_SPLIT.finditer(t):
                tok = m.group(0).lower()
                if tok in c["ids"]:
                    rep.add(c["id"], rel, tok)
    for m in PLUGIN_MODULE_RX.finditer(text):
        rep.add(PLUGIN_CAT, rel, m.group(0))
    for name, rx in SECRET_SHAPES:
        for _ in rx.finditer(text):
            rep.add(name, rel, None, secret=True)
    for src in secret_hits(text, stoks):
        if any(rx.search(src) and any(fnmatch.fnmatch(rel, g) for g in gl) for rx, gl in rules.secrets_allow):
            continue
        rep.add("secret-literal", rel, None, secret=True)
        rep.add("secret-literal-source", "%s <- %s" % (rel, src), None, secret=True, warn=True)
    for m in SWITCH_RX.finditer(text):
        if kind == "text" and switches and m.group(0) not in switches:
            rep.add("undocumented-switch", rel, m.group(0), warn=True)


FATBIN_MAGIC = b"\x50\xed\x55\xba"          # CUDA fatbin container (0xBA55ED50, little endian)
FATBIN_UNPACK_LIMIT = 8 << 30                 # decoded device code per file; more fails closed


ZSTD_SKIPPABLE = range(0x184D2A50, 0x184D2A60)   # zstd skippable frames: magic, u32 size, opaque bytes


def zstd_frame_len(data, a, b):
    """Exact length of the zstd frame starting at data[a] (RFC 8878 frame header + block headers + optional checksum),
    walked without decoding; None when the bytes in [a, b) are not one complete frame."""
    if data[a:a + 4] != ZSTD_MAGIC or a + 5 > b:
        return None
    fhd = data[a + 4]
    fcs_flag, single, has_ck, did_flag = fhd >> 6, (fhd >> 5) & 1, (fhd >> 2) & 1, fhd & 3
    if fhd & 0x08:                                 # reserved bit must be zero
        return None
    p = a + 5 + (0 if single else 1) + (0, 1, 2, 4)[did_flag] + ((1 if single else 0), 2, 4, 8)[fcs_flag]
    while True:
        if p + 3 > b:
            return None
        h = data[p] | data[p + 1] << 8 | data[p + 2] << 16
        last, btype, bsize = h & 1, (h >> 1) & 3, h >> 3
        if btype == 3:
            return None
        p += 3 + (1 if btype == 1 else bsize)
        if p > b:
            return None
        if last:
            break
    p += 4 if has_ck else 0
    return p - a if p <= b else None


def zstd_entry_frames(data, a, b):
    """Every zstd frame inside one fatbin entry payload [a, b): [(start, end)]. Bytes between and after frames (padding,
    anything else) are NOT frames and stay in place for the plain strings scan; a zstd magic that does not start a
    complete frame fails closed (never 'decode the first frame and trust the rest')."""
    frames, p = [], a
    while p < b:
        if data[p:p + 4] == ZSTD_MAGIC:
            n = zstd_frame_len(data, p, b)
            if n is None:
                raise GateError("compressed CUDA fatbin entry holds a zstd magic that is not a complete frame: the gate fails closed")
            frames.append((p, p + n))
            p += n
            continue
        if p + 8 <= b and struct.unpack_from("<I", data, p)[0] in ZSTD_SKIPPABLE:
            p += 8 + struct.unpack_from("<I", data, p + 4)[0]   # opaque bytes: left for the plain scan
            continue
        q = data.find(ZSTD_MAGIC, p + 1, b)
        p = b if q < 0 else q
    return frames


def fatbin_unpack(data):
    """CUDA fatbins whose entries are zstd-compressed (GGML_CUDA_COMPRESSION_MODE): returns (data with every
    zstd FRAME zeroed, the set of distinct printable runs of the decoded frames). The compressed bytes are noise to a strings
    scan (a lane-shaped run turns up by chance in hundreds of MB), so they are replaced by what they decode to: the device ELF /
    PTX is scanned as itself. EVERY frame of every entry is located exactly (zstd_entry_frames) and decoded on its own with a
    strict zstd exit; a frame that does not decode fails closed. Bytes of a compressed entry outside its frames (padding, a
    trailer, a second payload behind junk) are not zeroed: the plain strings pass reads them. Uncompressed entries are left as they are."""
    spans, decoded, total = [], set(), 0
    off = data.find(FATBIN_MAGIC)
    tool = None
    while off >= 0:
        ok = off + 16 <= len(data)
        if ok:
            _m, ver, hsz, fsz = struct.unpack_from("<IHHQ", data, off)
            ok = ver == 1 and hsz >= 16 and off + hsz + fsz <= len(data)
        if not ok:
            off = data.find(FATBIN_MAGIC, off + 4)
            continue
        p, end = off + hsz, off + hsz + fsz
        while p + 32 <= end:
            _kind, _ev, ehs, padded = struct.unpack_from("<HHII", data, p)
            if ehs < 32 or p + ehs > end:
                break
            a, b = p + ehs, min(p + ehs + padded, end)
            if data[a:a + 4] == ZSTD_MAGIC:
                for fa, fb in zstd_entry_frames(data, a, b):
                    tool = tool or shutil.which("zstd")
                    if not tool:
                        raise GateError("a compressed CUDA fatbin needs the zstd tool, which is missing: the gate fails closed")
                    r = subprocess.run([tool, "-dc", "-q", "--no-progress", "-"], input=data[fa:fb],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if r.returncode != 0 or not r.stdout:
                        raise GateError("compressed CUDA fatbin frame does not decode (%s): the gate fails closed"
                                        % r.stderr.decode("utf-8", "replace").strip()[:120])
                    total += len(r.stdout)
                    if total > FATBIN_UNPACK_LIMIT:
                        raise GateError("CUDA fatbins decode to more than %d GiB: the gate fails closed" % (FATBIN_UNPACK_LIMIT >> 30))
                    decoded.update(STRINGS_RX.findall(r.stdout))   # distinct runs only: GBs of SASS repeat the same names
                    spans.append((fa, fb))
            p += ehs + padded
        off = data.find(FATBIN_MAGIC, end)
    if not spans:
        return data, decoded
    buf = bytearray(data)
    for a, b in spans:
        buf[a:b] = bytes(b - a)
    return bytes(buf), decoded


BIG_DATA = 256 << 20         # a data file over this size gets the plain strings pass in chunks (memory), no fatbin decode
CHUNK = 64 << 20


def chunked_strings(path, chunk=CHUNK):
    """What STRINGS_RX finds in the whole file, read `chunk` bytes at a time; a run cut by a chunk edge is carried."""
    runs, carry = [], b""
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            data = carry + block
            m = re.search(rb"[\x20-\x7e\t]*\Z", data)
            carry, data = data[m.start():], data[:m.start()]
            runs += STRINGS_RX.findall(data)
    if len(carry) >= 4:
        runs.append(carry)
    return "\n".join(s.decode("ascii", "replace") for s in runs)


def elf_strings(path):
    with open(path, "rb") as f:
        data = f.read()
    data, decoded = fatbin_unpack(data)
    runs = STRINGS_RX.findall(data)
    runs += sorted(decoded)
    return "\n".join(s.decode("ascii", "replace") for s in runs)


def is_pe_bytes(head):
    """MZ plus a PE signature at e_lfanew. A text file that merely starts with the letters MZ is not a PE."""
    if len(head) < 0x40 or not head.startswith(b"MZ"):
        return False
    e = int.from_bytes(head[0x3C:0x40], "little")
    if e < 0x40 or e + 4 > len(head):
        return False
    return head[e:e + 4] == b"PE\0\0"


def classify(path):
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
            if head.startswith(b"MZ") and len(head) >= 0x40:
                e = int.from_bytes(head[0x3C:0x40], "little")
                if 0x40 <= e < (1 << 20) and e + 4 > len(head):
                    f.seek(0)
                    head = f.read(e + 4)
    except OSError:
        return "unreadable"
    if head.startswith(ELF_MAGIC):
        return "elf"
    if is_pe_bytes(head):
        return "pe"
    if head[:4] in MACHO_MAGICS or (head[:4] == FAT_MAGIC and 0 < int.from_bytes(head[4:8], "big") <= 20):
        return "macho"          # a Java class file also starts CAFEBABE, but its next word is >= 45
    if b"\0" in head[:8192]:
        return "other"
    return "text"


def documented_switches(root, rules):
    sw = set(rules.allow_switch)
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f in CATALOG_NAMES:
                try:
                    sw.update(CATALOG_ROW.findall(open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()))
                except OSError:
                    pass
    return sw


_JOB = {}
JOBS = 4


def _elf_job(item):
    """One ELF file in a worker process: its printable strings against the binary-scope rules."""
    p, rel = item
    r = Report(rel)
    s = chunked_strings(p) if os.path.getsize(p) > BIG_DATA and classify(p) == "other" else elf_strings(p)
    r.n_strings = s.count("\n") + 1
    scan_blob(r, rel, s, "binary", _JOB["rules"], _JOB["switches"], _JOB["stoks"])
    return rel, r.hits, r.warn, r.n_strings


def _set_jobs(n):
    global JOBS
    JOBS = max(1, n)


def _merge(rep, rel, hits, warn, alias=None):
    for book, src in ((rep.hits, hits), (rep.warn, warn)):
        for cat, files in src.items():
            for f, e in files.items():
                key = alias if (alias is not None and f == rel) else f
                book.setdefault(cat, {})[key] = [e[0], list(e[1])]


def scan_tree(root, rep, rules, stoks, only=None, prefix="", depth=0, outer_switches=None):
    """only: optional predicate(rel) -> bool (image mode: the files the base OS did not install).
    depth > 0: the tree is an unpacked onefile payload (see the module doc, check 4)."""
    switches = documented_switches(root, rules) | set(outer_switches or ())
    elfs, aliases, inodes, payloads = [], [], {}, []
    in_payload = depth > 0
    for dp, dns, fs in os.walk(root):
        dns.sort()
        for f in sorted(fs):
            p = os.path.join(dp, f)
            rel = prefix + os.path.relpath(p, root)
            if only is not None and not only(rel):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if in_payload:
                rep.n_payload_files += 1
            if not (in_payload and any(fnmatch.fnmatch(f, g) for g in PAYLOAD_SKIP_FILE_RULES)):
                for fr in rules.files:
                    if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(f, g) for g in fr["globs"]):
                        rep.add(fr["id"], rel, f)
            pn = plugin_payload_name(rel, f)
            if pn:
                rep.add(PLUGIN_CAT, rel, pn)
            k = classify(p)
            if k != "text" and zip_has_plugin(p):
                rep.add(PLUGIN_CAT, rel, "zip with a plugin manifest")
            if k != "text":
                found = detect_payloads(p, in_payload)
                if found and depth >= PAYLOAD_DEPTH:      # security review P10 L3: a payload past the depth is a block, never a silent skip
                    rep.add("payload-too-deep", rel, f)
                elif found:
                    payloads.append((p, rel, found))
            if k in ("elf", "pe", "macho", "other"):
                if k == "elf":
                    rep.n_elf += 1
                elif k == "pe":
                    rep.n_pe += 1
                elif k == "macho":
                    rep.n_macho += 1
                else:
                    rep.n_other += 1
                    if not in_payload and re.search(r"\.(?:tar|tgz|gz|zip|whl|xz|bz2|7z|jar)$", f, re.I):
                        rep.add("nested-archive", rel, f, warn=any(fnmatch.fnmatch(rel, g) for g in rules.archive_allow))
                ino = (st.st_dev, st.st_ino)
                if ino in inodes:                 # a hard link ships the same bytes under a second name
                    aliases.append((rel, inodes[ino]))
                else:
                    inodes[ino] = rel
                    elfs.append((p, rel))
            elif k == "text":
                if st.st_size > TEXT_LIMIT:
                    rep.add("oversize-text", rel, None, warn=True)
                    continue
                rep.n_text += 1
                scan_blob(rep, rel, open(p, "rb").read().decode("utf-8", "replace"), "text", rules, switches, stoks)
                if re.search(r"\.(?:tar|tgz|gz|zip|whl|xz|bz2|7z)$", f):
                    rep.add("nested-archive", rel, f, warn=True)
            else:
                rep.n_other += 1
                if re.search(r"\.(?:tar|tgz|gz|zip|whl|xz|bz2|7z|jar)$", f, re.I):
                    rep.add("nested-archive", rel, f, warn=any(fnmatch.fnmatch(rel, g) for g in rules.archive_allow))
    for p, rel, found in payloads:
        scan_payloads(p, rel, found, rep, rules, stoks, depth, switches)
    if not elfs:
        return
    _JOB.update(rules=rules, switches=switches, stoks=stoks)
    elfs.sort(key=lambda it: -os.path.getsize(it[0]))       # the biggest first: the pool finishes together
    if JOBS > 1 and len(elfs) > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(min(JOBS, len(elfs))) as pool:
            results = pool.map(_elf_job, elfs, chunksize=1)
    else:
        results = [_elf_job(it) for it in elfs]
    done = {}
    for rel, hits, warn, n in results:
        rep.n_strings += n
        _merge(rep, rel, hits, warn)
        done[rel] = (hits, warn, n)
    for rel, first in aliases:
        hits, warn, n = done[first]
        rep.n_strings += n
        _merge(rep, first, hits, warn, alias=rel)


# ------------------------------------------------------------------------------------------------ onefile payloads
def detect_payloads(path, in_payload=False):
    """[(kind, offset)] of compressed payloads inside one file. Cheap: mmap + find, nothing is unpacked here."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    if size < 16:                 # a tiny gzip or zip still carries a payload
        return []
    found = []
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            return []
        try:
            c = mm.rfind(PYI_COOKIE)
            if c >= 0 and c + 24 <= size:
                found.append(("pyinstaller", c))
            i = mm.find(NUITKA_ZSTD)
            while i >= 0 and len(found) < 16:
                found.append(("nuitka", i))
                i = mm.find(NUITKA_ZSTD, i + 1)
            i, n7 = mm.find(SEVENZ_MAGIC), 0
            while i >= 0 and n7 < 4:
                if mm[i + 6:i + 7] == b"\0" and mm[i + 7:i + 8] in (b"\x02", b"\x03", b"\x04"):
                    found.append(("7z", i))
                    n7 += 1
                i = mm.find(SEVENZ_MAGIC, i + 1)
            head = mm[:10]
        finally:
            mm.close()
    if head[:4] == ZSTD_MAGIC:
        found.append(("zstd", 0))
    elif head[:3] == b"\x1f\x8b\x08":
        found.append(("gzip", 0))
    elif head[:6] == b"\xfd7zXZ\0":
        found.append(("xz", 0))
    elif head[:3] == b"BZh" and head[3:4] in b"123456789" and head[4:10] == b"1AY&SY":
        found.append(("bz2", 0))
    else:
        # a zip appended to a program, or any zip by its magic bytes whatever its name (base_library.zip, bundled
        # data, a renamed .dat, an allowed upstream jar): its members get the same rules
        try:
            if zipfile.is_zipfile(path) and not (in_payload and _pyc_not_zip(path, head)):
                found.append(("zip", 0))
        except OSError:
            pass
    return found


def _pyc_not_zip(path, head):
    """True for compiled Python that only looks like a zip: the stdlib's zipimport.pyc keeps the zip signatures as
    bytes constants, so is_zipfile() says yes, yet nothing can be unpacked from it. Only when every zip signature in
    the file sits in such a constant and ZipFile() refuses the file as not a zip or opens it with no members (Python
    3.10's zipfile reads its own zipimport.pyc as an empty zip; 3.13's refuses it); asked inside a payload only, where
    its bytes still get the strings pass. A .pyc that opens as a zip with members, or carries a signature outside a
    constant (a zip, damaged or not, empty or not), stays a payload. A PYZ member of a PyInstaller program is the same
    compiled Python as bare marshal data, with no pyc header (10-26)."""
    pyz_raw = (os.sep + "PYZ" + os.sep) in path and head[:1] in (b"c", b"\xe3")    # marshal code object, plain or FLAG_REF
    if not path.endswith((".pyc", ".pyo")) or not (head[2:4] == b"\r\n" or pyz_raw):
        return False
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for sig in ZIP_SIGNATURES:
            i = mm.find(sig)
            while i >= 0:
                if mm[max(0, i - 5):i] not in MARSHAL_BYTES4:
                    return False
                i = mm.find(sig, i + 1)
    try:
        with zipfile.ZipFile(path) as z:
            return not z.infolist()     # no members: nothing to unpack, the strings pass reads its bytes
    except zipfile.BadZipFile:
        return True
    except Exception:
        return False        # a zip that opens wrong: the payload path opens it again and stops the gate (exit 3)


def _budget(n):
    _PAYLOAD["bytes"] += n
    if _PAYLOAD["bytes"] > PAYLOAD_BUDGET:
        raise GateError("onefile payloads unpack to more than %d GiB: the gate fails closed" % (PAYLOAD_BUDGET >> 30))


def _safe_member(name):
    name = name.replace("\\", "/")
    parts = [x for x in name.split("/") if x not in ("", ".", "..")]
    if parts and re.match(r"^[A-Za-z]:$", parts[0]):
        parts = parts[1:]
    return "/".join(parts) or "_"


def _put(dest, name, blob):
    _budget(len(blob))
    target = os.path.join(dest, _safe_member(name))
    if os.path.exists(target):                       # two entries with one name: keep both
        target += ".dup%d" % len(os.listdir(os.path.dirname(target)))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as out:
        out.write(blob)


def unpack_pyinstaller(data, cookie, dest):
    """CArchive entries (zlib when flagged) and every module of a PYZ archive (zlib'd marshal data)."""
    if cookie + 88 <= len(data):
        _m, plen, toff, tlen, _pyv = struct.unpack_from("!8sIIII", data, cookie)
        end = cookie + 88
    else:
        _m, plen, toff, tlen, _pyv = struct.unpack_from("!8sIIII", data, cookie)
        end = cookie + 24
    pkg = end - plen
    if pkg < 0 or pkg + toff + tlen > len(data):
        end, pkg = cookie + 24, cookie + 24 - plen      # PyInstaller 2.x cookie without the library name
        if pkg < 0 or pkg + toff + tlen > len(data):
            raise GateError("PyInstaller archive header is out of range")
    t, stop, n = pkg + toff, pkg + toff + tlen, 0
    while t < stop:
        es, dp, cl, _ul, cf, ty = struct.unpack_from("!IIIIBc", data, t)
        if es < 18 or t + es > stop or pkg + dp + cl > len(data):
            raise GateError("PyInstaller table of contents is corrupt")
        name = data[t + 18:t + es].rstrip(b"\0").decode("utf-8", "replace")
        blob = data[pkg + dp:pkg + dp + cl]
        if cf:
            blob = zlib.decompress(blob)
        if ty in (b"z", b"Z") and blob[:4] == b"PYZ\0":
            toc = marshal.loads(blob[struct.unpack("!i", blob[8:12])[0]:])
            for mod, meta in (toc.items() if isinstance(toc, dict) else toc):
                pos, length = meta[-2], meta[-1]
                code = blob[pos:pos + length]
                try:
                    code = zlib.decompress(code)
                except zlib.error:
                    pass
                _put(dest, "PYZ/" + str(mod).replace(".", "/") + ".pyc", code)
                n += 1
        else:
            _put(dest, name, blob)
            n += 1
        t += es
    return n


def _zstd_decompress(data, out_path):
    """The zstd tool decodes the first frame(s); bytes after the payload (padding, signature) end the stream with
    'unsupported format', which is expected. Anything else, or no output, fails closed."""
    tool = shutil.which("zstd")
    if not tool:
        raise GateError("a zstd-compressed payload needs the zstd tool, which is missing: the gate fails closed")
    with open(out_path, "wb") as out:
        p = subprocess.run([tool, "-dc", "-q", "--no-progress", "-"], input=data, stdout=out, stderr=subprocess.PIPE)
    got = os.path.getsize(out_path)
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode != 0 and not (got > 0 and re.search(r"unsupported format|unknown header|Unknown frame descriptor", err)):
        raise GateError("zstd payload does not decode (%s): the gate fails closed" % err.strip()[:120])
    if got == 0:
        raise GateError("zstd payload decodes to nothing: the gate fails closed")
    _budget(got)
    return got


def _nuitka_entries(o):
    """Nuitka onefile file table: name (UTF-8, or UTF-16LE on Windows) NUL, [1 flag byte on POSIX], size u64,
    [crc32 u32], bytes; an empty name ends it. The layout variant is the one that parses to the end exactly."""
    for wide in (False, True):
        for flags in (1, 0):
            for crc in (4, 0):
                p, out, ok = 0, [], False
                try:
                    while p < len(o):
                        if wide:
                            e = p
                            while True:
                                e = o.find(b"\0\0", e)
                                if e < 0 or (e - p) % 2 == 0:
                                    break
                                e += 1
                            if e < 0:
                                break
                            name = o[p:e].decode("utf-16-le")
                            p = e + 2
                        else:
                            e = o.find(b"\0", p, p + 4096)
                            if e < 0:
                                break
                            name = o[p:e].decode("utf-8")
                            p = e + 1
                        if not name:
                            ok = not o[p:].strip(b"\0")
                            break
                        p += flags
                        size = struct.unpack_from("<Q", o, p)[0]
                        p += 8 + crc
                        if size > len(o) - p:
                            break
                        out.append((name, p, size))
                        p += size
                except (UnicodeDecodeError, struct.error):
                    ok = False
                if ok and out:
                    return out
    return None


def unpack_nuitka(path, offset, dest):
    with open(path, "rb") as f:
        f.seek(offset + 3)
        data = f.read()
    raw = os.path.join(dest, ".nuitka-payload.raw")
    _zstd_decompress(data, raw)
    del data
    with open(raw, "rb") as f:
        o = f.read()
    os.remove(raw)
    ents = _nuitka_entries(o)
    if ents is None:            # unknown table layout: the whole stream is still scanned, as one binary
        _put(dest, "nuitka-payload.bin", o)
        return 1
    for name, p, size in ents:
        _put(dest, name, o[p:p + size])
    return len(ents)


def unpack_7z(path, offset, dest):
    """7z (or the 7z part of a self-extractor) with bsdtar or 7z; no tool = fail closed."""
    tool = shutil.which("bsdtar") or shutil.which("7zz") or shutil.which("7z") or shutil.which("7za")
    if not tool:
        raise GateError("a 7z payload needs bsdtar or 7z, neither is installed: the gate fails closed")
    arc = dest.rstrip(os.sep) + ".7z"
    with open(path, "rb") as f, open(arc, "wb") as out:
        f.seek(offset)
        shutil.copyfileobj(f, out, 1 << 20)
    tree = dest
    if os.path.basename(tool) == "bsdtar":
        cmd = [tool, "-x", "-f", arc, "-C", tree, "--no-same-owner", "--no-same-permissions"]
    else:
        cmd = [tool, "x", "-y", "-bd", "-o" + tree, arc]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(arc)
    if p.returncode != 0:
        raise GateError("7z payload does not unpack (%s): the gate fails closed" % p.stderr.decode("utf-8", "replace").strip()[:120])
    n = 0
    for dp, _dn, fs in os.walk(tree):
        for fn in fs:
            fp = os.path.join(dp, fn)
            if os.path.islink(fp) or not os.path.isfile(fp):
                continue
            real = os.path.realpath(fp)
            if not real.startswith(os.path.realpath(tree) + os.sep):
                raise GateError("7z payload member escapes the extraction directory")
            _budget(os.path.getsize(fp))
            n += 1
    return n


DECOMPRESSORS = {"gzip": gzip.open, "xz": lzma.open, "bz2": bz2.open}


def unpack_stream(path, kind, dest):
    """A gzip / xz / bz2 file (any name): decoded into one file; trailing garbage or a bad stream fails closed."""
    out_path = os.path.join(dest, os.path.basename(path) + "." + kind + "-data")
    got = 0
    with DECOMPRESSORS[kind](path, "rb") as src, open(out_path, "wb") as out:
        while True:
            block = src.read(CHUNK)
            if not block:
                break
            got += len(block)
            _budget(len(block))
            out.write(block)
    if got == 0:
        raise GateError("%s payload decodes to nothing: the gate fails closed" % kind)
    return 1


def unpack_zip_payload(path, dest):
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.filename.endswith("/") or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                continue
            if info.file_size > PAYLOAD_BUDGET:
                raise GateError("zip payload member is over the size budget")
            _put(dest, info.filename, z.read(info))
            n += 1
    return n


def scan_payloads(path, rel, found, rep, rules, stoks, depth, switches):
    """Unpack each payload of one file into --work and scan it as a tree named <rel>!/..."""
    work = _PAYLOAD["work"] or tempfile.gettempdir()
    for kind, offset in found:
        sub = tempfile.mkdtemp(prefix="release-scan-payload-", dir=work)
        try:
            try:
                if kind == "pyinstaller":
                    with open(path, "rb") as f:
                        data = f.read()
                    n = unpack_pyinstaller(data, offset, sub)
                    del data
                elif kind == "nuitka":
                    n = unpack_nuitka(path, offset, sub)
                elif kind == "zstd":
                    with open(path, "rb") as f:
                        data = f.read()
                    _zstd_decompress(data, os.path.join(sub, os.path.basename(path) + ".unzst"))
                    del data
                    n = 1
                elif kind == "7z":
                    n = unpack_7z(path, offset, sub)
                elif kind in DECOMPRESSORS:
                    n = unpack_stream(path, kind, sub)
                else:
                    n = unpack_zip_payload(path, sub)
            except (struct.error, zlib.error, ValueError, EOFError, TypeError, IndexError, zipfile.BadZipFile, OSError,
                    lzma.LZMAError, RuntimeError, NotImplementedError) as e:    # RuntimeError: an encrypted zip member
                raise GateError("cannot unpack the %s payload of %s (%s): the gate fails closed" % (kind, rel, e))
            rep.n_payloads += 1
            tag = "" if len(found) == 1 else "@%s%d" % (kind, offset)
            scan_tree(sub, rep, rules, stoks, prefix="%s%s!/" % (rel, tag), depth=depth + 1, outer_switches=switches)
        finally:
            shutil.rmtree(sub, ignore_errors=True)


def make_pyinstaller_sample(modules, scripts=(), stub=b""):
    """A minimal PyInstaller one-file layout (tests, self-test): stub + CArchive (zlib'd scripts, one PYZ of zlib'd
    module blobs) + the 88-byte cookie."""
    body, toc, pos = b"", [], 12
    for name, text in modules:
        blob = zlib.compress(b"\xe3" + text.encode("utf-8"))
        toc.append((name, (0, pos, len(blob))))
        body += blob
        pos += len(blob)
    pyz = b"PYZ\0" + b"\xcb\x0d\x0d\x0a" + struct.pack("!i", 12 + len(body)) + body + marshal.dumps(toc)
    data, ents = b"", []
    for name, text in scripts:
        blob = zlib.compress(text.encode("utf-8"))
        ents.append((name, len(data), len(blob), len(text), 1, b"s"))
        data += blob
    ents.append(("PYZ.pyz", len(data), len(pyz), len(pyz), 0, b"z"))
    data += pyz
    tocb = b""
    for name, dp, cl, ul, cf, ty in ents:
        nb = name.encode("utf-8") + b"\0"
        nb = nb.ljust((len(nb) + 18 + 15) // 16 * 16 - 18, b"\0")
        tocb += struct.pack("!IIIIBc", 18 + len(nb), dp, cl, ul, cf, ty) + nb
    plen = len(data) + len(tocb) + 88
    return stub + data + tocb + struct.pack("!8sIIII64s", PYI_COOKIE, plen, len(data), len(tocb), 312, b"python312.dll")


def make_nuitka_sample(files, stub=b"", tail=b"\0" * 64):
    """A minimal Nuitka onefile layout (POSIX table: name NUL, flag byte, u64 size, crc32, bytes), zstd-compressed
    with the zstd tool. Returns None when the tool is missing."""
    tool = shutil.which("zstd")
    if not tool:
        return None
    table = b""
    for name, blob in files:
        table += name.encode("utf-8") + b"\0" + b"\x01" + struct.pack("<Q", len(blob)) + \
            struct.pack("<I", zlib.crc32(blob)) + blob
    table += b"\0"
    p = subprocess.run([tool, "-q", "-c", "-3", "-"], input=table, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None
    return stub + b"KAY" + p.stdout + tail


def safe_extract(tf, dest, members=None):
    """Regular files and hard links only (a hard link ships the same bytes under a second name, so it is
    linked here too and scanned under its own name); symlinks carry no bytes and are skipped."""
    dest = os.path.realpath(dest)

    def inside(name):
        t = os.path.realpath(os.path.join(dest, name.lstrip("/")))
        if not (t == dest or t.startswith(dest + os.sep)):
            raise GateError("archive member escapes the extraction directory: %s" % name)
        return t

    for m in (tf if members is None else members):
        target = inside(m.name)
        if m.issym():
            continue
        if m.isdir():
            os.makedirs(target, exist_ok=True)
            continue
        if not (m.isreg() or m.islnk()):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target):
            os.unlink(target)
        if m.islnk():
            src = inside(m.linkname)
            if os.path.isfile(src):
                os.link(src, target)
                continue
        data = tf.extractfile(m)
        if data is None:
            continue
        with open(target, "wb") as out:
            shutil.copyfileobj(data, out, 1 << 20)


def is_zip(path):
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
    except OSError:
        return False
    return sig[:2] == b"PK" and sig[2:4] in (b"\x03\x04", b"\x05\x06", b"\x07\x08")


def safe_extract_zip(path, dest):
    """Regular files only. Symlinks carry no bytes of their own and are skipped. A member that
    escapes the extraction directory (absolute path, drive letter, or '..') refuses the archive."""
    dest = os.path.realpath(dest)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/").lstrip("/")
            if not name or name.endswith("/"):
                continue
            if name.startswith("../") or "/../" in "/" + name or re.match(r"^[A-Za-z]:", name) or name.startswith("//"):
                raise GateError("archive member escapes the extraction directory: %s" % info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                continue
            target = os.path.realpath(os.path.join(dest, name))
            if not (target == dest or target.startswith(dest + os.sep)):
                raise GateError("archive member escapes the extraction directory: %s" % info.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = z.open(info, "r")
            try:
                with open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            finally:
                src.close()


def open_tar(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    return tarfile.open(path, "r:gz" if magic == b"\x1f\x8b" else "r:")


def is_docker_archive(path):
    try:
        with open_tar(path) as tf:
            for m in tf:
                if m.name in ("manifest.json", "./manifest.json"):
                    return True
                if m.name.count("/") > 3:
                    return False
    except (tarfile.TarError, OSError, EOFError):
        return False
    return False


def scan_docker(path, rep, rules, stoks, work):
    outer = os.path.join(work, "image")
    os.makedirs(outer)
    with open_tar(path) as tf:
        safe_extract(tf, outer)
    man = json.load(open(os.path.join(outer, "manifest.json")))
    if not man:
        raise GateError("empty docker manifest")
    rootfs = os.path.join(work, "rootfs")
    os.makedirs(rootfs)
    cfg_name = man[0].get("Config")
    for layer in man[0].get("Layers", []):
        lp = os.path.join(outer, layer)
        with open_tar(lp) as lt:
            members = lt.getmembers()
            for m in members:                         # whiteouts first: they delete what lower layers put there
                b = os.path.basename(m.name)
                if b == ".wh..wh..opq":
                    d = os.path.join(rootfs, os.path.dirname(m.name))
                    if os.path.isdir(d):
                        for x in os.listdir(d):
                            q = os.path.join(d, x)
                            shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.unlink(q)
                elif b.startswith(".wh."):
                    q = os.path.join(rootfs, os.path.dirname(m.name), b[4:])
                    if os.path.lexists(q):
                        shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.unlink(q)
            keep = [m for m in members if not os.path.basename(m.name).startswith(".wh.")]
            for m in keep:
                q = os.path.join(rootfs, m.name.lstrip("/"))
                if m.isreg() and os.path.lexists(q) and (os.path.islink(q) or not os.path.isfile(q)):
                    shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.unlink(q)
            safe_extract(lt, rootfs, keep)
    owned = set()
    info = os.path.join(rootfs, "var/lib/dpkg/info")
    if os.path.isdir(info):
        for f in os.listdir(info):
            if f.endswith(".list"):
                for l in open(os.path.join(info, f), encoding="utf-8", errors="ignore"):
                    owned.add(l.strip().lstrip("/"))
    skip = tuple(s.lstrip("/") for s in rules.image_skip) + ("var/lib/dpkg/", "var/lib/apt/", "var/cache/", "var/log/",
                                                             "usr/share/doc/", "usr/share/man/", "usr/share/locale/", "etc/")
    def only(rel):
        r = rel[len("image:/"):]
        return r not in owned and not r.startswith(skip)
    scan_tree(rootfs, rep, rules, stoks, only=only, prefix="image:/")
    # the image config carries the build history (RUN lines), labels and the environment: public text too
    if cfg_name and os.path.isfile(os.path.join(outer, cfg_name)):
        cfg = open(os.path.join(outer, cfg_name), encoding="utf-8", errors="replace").read()
        rep.n_text += 1
        scan_blob(rep, "image-config.json", cfg, "text", rules, documented_switches(rootfs, rules), stoks)


def looks_like_tar(path):
    """A tarball by its name or its gzip magic (a corrupt one must still stop the gate), or by tarfile's own check."""
    if re.search(r"\.(?:tar|tgz|tar\.gz|gz)$", path, re.I):
        return True
    with open(path, "rb") as f:
        head = f.read(262)
    return head[:2] == b"\x1f\x8b" or head[257:262] == b"ustar"    # never tarfile.is_tarfile: 512 zero bytes pass it


def scan_input(path, rules, stoks, work):
    rep = Report(os.path.basename(path.rstrip("/")))
    if os.path.isdir(path):
        scan_tree(path, rep, rules, stoks)
        return rep
    if not os.path.isfile(path):
        raise GateError("input not found: %s" % path)
    sub = tempfile.mkdtemp(prefix="release-scan-", dir=work)
    try:
        if is_zip(path):
            safe_extract_zip(path, sub)
            scan_tree(sub, rep, rules, stoks)
        elif is_docker_archive(path):
            rep.label += " (container image)"
            scan_docker(path, rep, rules, stoks, sub)
        elif not looks_like_tar(path):
            one = os.path.join(sub, os.path.basename(path))
            try:
                os.link(path, one)
            except OSError:
                shutil.copy2(path, one)
            scan_tree(sub, rep, rules, stoks)
        else:
            with open_tar(path) as tf:
                safe_extract(tf, sub)
                if not tf.getmembers():
                    raise GateError("tarball %s holds no files: the gate fails closed" % os.path.basename(path))
                while True:                               # past the end blocks only zero padding may follow
                    block = tf.fileobj.read(1 << 20)
                    if not block:
                        break
                    if block.strip(b"\0"):
                        raise GateError("tarball %s has bytes after its end: the gate fails closed" % os.path.basename(path))
            scan_tree(sub, rep, rules, stoks)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, ValueError, OSError) as e:
        raise GateError("cannot read archive %s: %s" % (os.path.basename(path), e))
    finally:
        shutil.rmtree(sub, ignore_errors=True)
    return rep


# ------------------------------------------------------------------------------------------------ self-test
def self_test(rules, work):
    """Poisoned samples must be caught, in a text file and in an ELF file, or the gate is broken: fail closed."""
    d = tempfile.mkdtemp(prefix="release-scan-selftest-", dir=work)
    try:
        samples = list(rules.selftests) + [("secret-shape-github", "ghp_" + "Q7" * 18), ("secret-shape-pem", "-----BEGIN " + "PRIVATE KEY-----")]
        cats = {}
        for c in rules.cats:                    # one id may span sections (a text rule and a binary rule)
            m = cats.setdefault(c["id"], {"scope": set(), "ext": (), "allow": []})
            m["scope"] |= c["scope"]
            m["ext"] = m["ext"] or c["ext"]
            m["allow"] = list(dict.fromkeys(list(m["allow"]) + list(c["allow"])))
        names = []
        for i, (cat, s) in enumerate(samples):
            c = cats.get(cat)
            ext = c["ext"][0] if c is not None and c.get("ext") else ".md"
            names.append("t%03d%s" % (i, ext))
            open(os.path.join(d, names[-1]), "w", encoding="utf-8").write("x %s y\n" % s)
        open(os.path.join(d, "e.so"), "wb").write(ELF_MAGIC + b"\x02\x01\x01" + b"\0" * 9 + b"\0" * 64 +
                                                  b"\0".join(s.encode() for _, s in samples) + b"\0")
        pe = bytearray(0x80)
        pe[0:2] = b"MZ"
        pe[0x3C:0x40] = (0x80).to_bytes(4, "little")
        pe = bytes(pe) + b"PE\0\0" + b"\0".join(s.encode() for _, s in samples) + b"\0"
        open(os.path.join(d, "e.dll"), "wb").write(pe)
        # onefile payloads: the samples compressed inside a program must be caught too (check 4)
        blob = "\n".join(s for _, s in samples)
        open(os.path.join(d, "o1.exe"), "wb").write(make_pyinstaller_sample([("pkg.mod", blob)], stub=pe))
        onefile = ["o1.exe"]
        nk = make_nuitka_sample([("data.bin", b"\0" + blob.encode("utf-8") + b"\0")], stub=ELF_MAGIC + b"\0" * 60)
        if nk is not None:
            open(os.path.join(d, "o2"), "wb").write(nk)
            onefile.append("o2")
        rep = Report("selftest")
        scan_tree(d, rep, rules, {})
        bad = []
        for i, (cat, s) in enumerate(samples):
            c = cats.get(cat)
            if (c is None or "text" in c["scope"]) and names[i] not in rep.hits.get(cat, {}):
                bad.append("%s (text)" % cat)
            if (c is None or "binary" in c["scope"]) and "e.so" not in rep.hits.get(cat, {}):
                bad.append("%s (binary)" % cat)
            # A PE file is scanned the same way as an ELF file. An extension-limited rule (ext=.so) does
            # not have to hit e.dll; every other binary-scope sample does, or the gate is blind to Windows.
            ext = () if c is None else (c.get("ext") or ())
            allowed = c is not None and any(fnmatch.fnmatch("e.dll", g) for g in (c.get("allow") or ()))
            if (not allowed) and (c is None or "binary" in c["scope"]) and (not ext or ".dll" in ext) \
                    and "e.dll" not in rep.hits.get(cat, {}):
                bad.append("%s (pe)" % cat)
            if (c is None or "binary" in c["scope"]) and not ext:
                for o in onefile:
                    if not any(f.startswith((o + "!/", o + "@")) for f in rep.hits.get(cat, {})):
                        bad.append("%s (onefile %s)" % (cat, o))
        bad += plugin_self_test(d, rules, pe)
        if bad:
            raise GateError("self-test: the gate missed its own poisoned samples: " + ", ".join(bad))
        return len(samples)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def plugin_self_test(work, rules, pe_stub):
    """Every plugin payload shape is caught (names, paths, a renamed .pxp, a onefile module, binary module names) and
    the Control's own mentions of the Store item (an id, a voice folder name) are not."""
    import zipfile
    d = tempfile.mkdtemp(prefix="plugin-selftest-", dir=work)
    try:
        def put(rel, data=b"x\n"):
            fp = os.path.join(d, rel)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "wb").write(data)
        want = ["a/anibal-0.1.0.pxp", "src/plugin.src.json", "x/hello/plugin.json", "tools/anibal/core/brain.py",
                "tools/vigil/run.py", "tools/pxa_plugins/host.py", "tools/pxa-private/plugins/k.md", "lib/anibal/__init__.py",
                "renamed.bin", "o3.exe", "mod.so"]
        for rel in want[:8]:
            put(rel)
        zb = os.path.join(d, "renamed.bin")
        with zipfile.ZipFile(zb, "w") as z:
            z.writestr("plugin.json", "{}")
            z.writestr("bin/run", "x")
        open(os.path.join(d, "o3.exe"), "wb").write(make_pyinstaller_sample([("anibal" ".core.brain", "x = 1")], stub=pe_stub))
        open(os.path.join(d, "mod.so"), "wb").write(ELF_MAGIC + b"\x02\x01\x01" + b"\0" * 73 + b"anibal" b".voice.bridge\0")
        clean = ["tools/pxa_control/speech/tts.py", "tools/pxa_control_ui/plugins.js", "docs/anibal.md"]
        put(clean[0], b"VOICES = 'anibal/voice/models'\n")
        put(clean[1], b"if (it.id === 'anibal') show(anibal.name);\n")
        put(clean[2], b"Anibal is a supporter reward in the Store.\n")
        rep = Report("plugin-selftest")
        scan_tree(d, rep, rules, {})
        got = rep.hits.get(PLUGIN_CAT, {})
        bad = ["%s (plugin payload missed)" % w for w in want if not any(k == w or k.startswith(w + "!/") for k in got)]
        bad += ["%s (plugin payload false hit)" % c for c in clean if c in got]
        return bad
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ main
SECRET_CATS = ("secret-",)


def print_report(rep, show):
    print("release-scan: %s: %d text files, %d ELF files, %d PE files, %d Mach-O files (%d strings), %d other%s" % (
        rep.label, rep.n_text, rep.n_elf, rep.n_pe, rep.n_macho, rep.n_strings, rep.n_other,
        "; %d onefile payload(s) unpacked, %d file(s) inside" % (rep.n_payloads, rep.n_payload_files) if rep.n_payloads else ""))
    for cat in sorted(rep.hits, key=lambda c: -sum(v[0] for v in rep.hits[c].values())):
        files = rep.hits[cat]
        print("  BLOCK %-22s %6d hit(s) in %d file(s)" % (cat, sum(v[0] for v in files.values()), len(files)))
        for f, (cnt, ex) in sorted(files.items(), key=lambda kv: -kv[1][0])[:show]:
            if cat.startswith(SECRET_CATS):
                print("      %5d  %s  [value not shown]" % (cnt, f))
            else:
                print("      %5d  %s  %s" % (cnt, f, ", ".join(repr(e) for e in ex)))
        if len(files) > show:
            print("      ... %d more file(s)" % (len(files) - show))
    for cat in sorted(rep.warn):
        files = rep.warn[cat]
        if cat == "secret-literal-source":
            for f in sorted(files):
                print("  NOTE  secret-literal  %s  [value not shown]" % f)
            continue
        print("  WARN  %-22s %6d in %d file(s)" % (cat, sum(v[0] for v in files.values()), len(files)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="release package scan gate (exit 0 clean, 2 blocked, 3 cannot run)")
    ap.add_argument("inputs", nargs="+", help="package directory, package tarball, or docker save archive")
    ap.add_argument("--blocklist", default=os.environ.get("PXA_RELEASE_BLOCKLIST", ""), help="default: $PXA_RELEASE_BLOCKLIST")
    ap.add_argument("--json", help="write the counts (no matched values) to this file")
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"), help="where archives are unpacked (needs ~3x their size)")
    ap.add_argument("--show", type=int, default=8, help="files listed per category")
    ap.add_argument("--jobs", type=int, default=JOBS, help="worker processes for the binaries (default %(default)s)")
    a = ap.parse_args(argv)
    _set_jobs(a.jobs)
    try:
        if not a.blocklist:
            raise GateError("PXA_RELEASE_BLOCKLIST is not set: the gate fails closed (no block list, no package)")
        if not os.path.isfile(a.blocklist):
            raise GateError("the block list file does not exist: the gate fails closed")
        rules = load_rules(a.blocklist)
        if not rules.cats:
            raise GateError("the block list has no categories: the gate fails closed")
        os.makedirs(a.work, exist_ok=True)
        _PAYLOAD["work"] = a.work
        n = self_test(rules, a.work)
        stoks = secret_tokens(rules)
        print("release-scan: block list %d categories, %d file rules, %d secret literals, self-test %d samples OK"
              % (len(rules.cats), len(rules.files), len(stoks), n))
        reps = []
        for p in a.inputs:
            _PAYLOAD["bytes"] = 0
            reps.append(scan_input(p, rules, stoks, a.work))
    except GateError as e:
        print("release-scan: CANNOT RUN: %s" % e, file=sys.stderr)
        return 3
    total = 0
    for rep in reps:
        print_report(rep, a.show)
        total += rep.total()
    if a.json:
        out = {r.label: {"blocks": {c: {"hits": sum(v[0] for v in f.values()), "files": len(f),
                                        "by_file": {k: v[0] for k, v in f.items()}} for c, f in r.hits.items()},
                         "warns": {c: {"hits": sum(v[0] for v in f.values()), "files": len(f)} for c, f in r.warn.items()
                                   if c != "secret-literal-source"},
                         "text_files": r.n_text, "elf_files": r.n_elf, "pe_files": r.n_pe, "macho_files": r.n_macho,
                         "payloads": r.n_payloads, "payload_files": r.n_payload_files} for r in reps}
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    if total:
        print("release-scan: FAIL - %d blocked hit(s): no package, no upload" % total)
        return 2
    print("release-scan: PASS - 0 blocked hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
