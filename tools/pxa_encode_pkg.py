"""pxa_encode_pkg.py - "Get the encoder": download, verify and install the PXA Quantizer package for PXA Control.

The quantizer is distributed as a signed package from the PXA licence server (shape: tools/pxa-licd/PACKAGE.md there):
  * Free  - GET  <licence>/v1/free/latest?platform=&cuda_major=                    (anyone, no key)
  * Pro   - POST <licence>/v1/package {key, platform, cuda_major, build_id?}        (a supporter key from the Discord bot)
  * check - POST <licence>/v1/package/latest {key, build_id}                         (is there a newer Pro build?)
  * GPU runtime - GET <licence>/v1/runtime/latest?platform=&cuda_major=              (anyone, no key: the NVIDIA CUDA libraries the Pro encoder needs,
                    the same pack for every user, signed like the Free package; installed once next to the encoders, see "the GPU runtime" below)
Every answer carries `statement` (canonical JSON: build_id, cuda_major, edition, kid, pkg, platform, sha256, size) and `signature`
(base64url Ed25519 over b"pxqe-pkg|" + statement) plus a download `url` (a path is resolved against the server).

SAFETY (this code downloads something Control will later run, so it is strict on purpose):
  * the statement's Ed25519 signature must verify against the public key built into Control, the statement must say what was asked
    for (edition, platform, CUDA major, build), and the downloaded file's size and sha256 must equal the statement's; a package that
    fails any of these is deleted and refused with one plain sentence; nothing is extracted first;
  * the package's own MANIFEST.json (sha256 of every file) must carry a valid MANIFEST.sig, and every extracted file must match it;
  * extraction refuses absolute paths, `..`, links and special files; it extracts into a temp folder and renames;
  * after install the command line must answer `info --json` with the manifest's own edition and build id;
  * the licence KEY is sent only to the licence server URL (a constant, plus a config/env override for tests) and only in the body
    of the two POSTs above. It is never put in a URL, a log line, an error message, a score, a bug report or a telemetry record,
    and the page only ever sees a masked form of it. The download itself carries no key.
"""

import base64
import binascii
import hashlib
import json
import os
import platform as _platform
import re
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request

from pxa_encode_adapter import AdapterError, install_root, probe

# The licence server (tools/pxa-licd/deploy/ROUTING.md: the planned public hostname; it answers once the owner applies the routing).
# Override in tests or on a private setup with PXA_LICENCE_URL or the licence_url field of control.json.
LICENCE_URL_DEFAULT = "https://lic.pxanetwork.com"
# The Ed25519 public keys (64 hex chars) packages are signed with, by key id: the trust anchor, shipped inside Control.
# A package that names a key id not listed here is refused (fail closed).
PACKAGE_PUBKEYS = {"pkg-2026-10": "de9a21ae8508a373452f422fe110ea08e68ac72569019cb2214442facfdf7af4"}
SIGN_PREFIX = b"pxqe-pkg|"
MANIFEST_PREFIX = b"pxqe-manifest|"
DEFAULT_PLATFORM = "linux-x86_64"
DEFAULT_CUDA = 12              # the packages are CUDA 12 builds
USER_AGENT = "pxa-control-encode"
KEY_RE = re.compile(r"^pxk1\.[A-Za-z0-9\-]{3,24}\.[A-Za-z0-9_\-]{16,64}$")
MAX_MANIFEST = 64 * 1024
MAX_PACKAGE = 4 << 30           # 4 GiB: far above any real package; a sanity bound on a hostile Content-Length


class PackageError(Exception):
    """A plain sentence for the user, plus a short code the page can branch on (offline, revoked, bad_signature ...)."""

    def __init__(self, msg, code="failed"):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------------------------------------
# the key: validate, mask, never leak
# ---------------------------------------------------------------------------------------------
def valid_key(key):
    return isinstance(key, str) and KEY_RE.match(key.strip()) is not None


def mask_key(key):
    """pxk1.K-CBFAB757.XXXXXXXX... -> 'pxk1.K-CBFAB757.••••••••' (the id part is the support handle, the secret is hidden)."""
    if not isinstance(key, str) or not key:
        return ""
    parts = key.strip().split(".")
    if len(parts) == 3:
        return "%s.%s.%s" % (parts[0], parts[1], "•" * 8)
    return "•" * 8


def licence_url(cfg=None):
    u = os.environ.get("PXA_LICENCE_URL") or ((cfg or {}).get("licence_url") or "") or LICENCE_URL_DEFAULT
    return u.rstrip("/")


def _scrub(text, key=None):
    """Anything that is about to be shown or logged goes through here: the key, in any spelling, becomes <key>."""
    s = str(text)
    if key:
        s = s.replace(key, "<key>")
    return re.sub(r"pxk1\.[A-Za-z0-9\-]{3,24}\.[A-Za-z0-9_\-]{8,}", "<key>", s)


# ---------------------------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------------------------
def detect_platform(cuda_version=None):
    """-> (platform, cuda_major): 'linux-x86_64' and the CUDA major version to ask for. The packages are CUDA 12 builds: a driver that
    supports CUDA 12 or newer asks for 12, a machine whose CUDA version cannot be read is asked for 12, and an older driver asks for its
    own (the server then says plainly that there is no package for it)."""
    sysname = _platform.system().lower() or "linux"
    mach = _platform.machine().lower().replace("amd64", "x86_64") or "x86_64"
    m = re.match(r"^(\d+)", str(cuda_version or ""))
    major = int(m.group(1)) if m else DEFAULT_CUDA
    return "%s-%s" % (sysname, mach), min(major, DEFAULT_CUDA)           # a newer driver (CUDA 13) runs the CUDA 12 build; an older one asks for its own


# ---------------------------------------------------------------------------------------------
# Ed25519 verification (RFC 8032, reference algorithm). Public data only, so no constant-time worries.
# ---------------------------------------------------------------------------------------------
_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _recover_x(y, sign):
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _add(a, b):
    A = (a[1] - a[0]) * (b[1] - b[0]) % _P
    B = (a[1] + a[0]) * (b[1] + b[0]) % _P
    C = 2 * a[3] * b[3] * _D % _P
    D = 2 * a[2] * b[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s, pt):
    r = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            r = _add(r, pt)
        pt = _add(pt, pt)
        s >>= 1
    return r


def _eq(a, b):
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _decompress(s):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def ed25519_verify(pub, msg, sig):
    """True when `sig` (64 bytes) is a valid Ed25519 signature of `msg` under `pub` (32 bytes)."""
    if len(pub) != 32 or len(sig) != 64:
        return False
    A = _decompress(pub)
    R = _decompress(sig[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _Q:
        return False
    h = int.from_bytes(hashlib.sha512(sig[:32] + pub + msg).digest(), "little") % _Q
    return _eq(_mul(s, _G), _add(R, _mul(h, A)))


def _decode_sig(text):
    t = str(text or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{128}", t):
        return bytes.fromhex(t)
    try:
        return base64.urlsafe_b64decode(t.replace("+", "-").replace("/", "_") + "=" * (-len(t) % 4))
    except (binascii.Error, ValueError):
        return b""


def pubkey_bytes(kid=None):
    """The trust anchor for a key id. PXA_PACKAGE_PUBKEY (64 hex chars) overrides it for tests only."""
    h = (os.environ.get("PXA_PACKAGE_PUBKEY") or PACKAGE_PUBKEYS.get(str(kid or ""), "") or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", h):
        return b""
    return bytes.fromhex(h)


def check_statement(o, want_edition, platform, cuda_major, want_build=None):
    """The answer of the licence server -> the manifest this module installs from. Raises PackageError unless the statement is signed
    by a key Control trusts AND says what was asked for."""
    need = ("build_id", "edition", "statement", "signature", "url")
    if not isinstance(o, dict) or not all(isinstance(o.get(k), str) and o.get(k) for k in need):
        raise PackageError("The licence server's package description is incomplete.", "bad_manifest")
    pub = pubkey_bytes(o.get("kid"))
    if not pub:
        raise PackageError("This build of PXA Control does not know the key (%s) this package is signed with, so it will not install it. "
                           "Update PXA Control and try again." % str(o.get("kid") or "none")[:32], "no_pubkey")
    if not ed25519_verify(pub, SIGN_PREFIX + o["statement"].encode(), _decode_sig(o["signature"])):
        raise PackageError("The package signature is not valid, so it was NOT installed. If this keeps happening, tell us in the PXA Network Discord.", "bad_signature")
    try:
        st = json.loads(o["statement"])
    except ValueError:
        raise PackageError("The package statement cannot be read, so it was refused.", "bad_manifest")
    if not isinstance(st, dict):
        raise PackageError("The package statement cannot be read, so it was refused.", "bad_manifest")
    ok = (st.get("edition") == want_edition == o["edition"] and st.get("platform") == platform and st.get("cuda_major") == cuda_major
          and st.get("build_id") == o["build_id"] and (not want_build or st.get("build_id") == want_build))
    if not ok:
        raise PackageError("The signed package description does not match what was asked for (edition, platform, CUDA version or build), so it was refused.", "bad_statement")
    if not re.fullmatch(r"[A-Za-z0-9._\-]{1,64}", str(st.get("build_id"))) or not re.fullmatch(r"[0-9a-f]{64}", str(st.get("sha256") or "")):
        raise PackageError("The signed package description has no valid build id or checksum, so it was refused.", "bad_manifest")
    size = st.get("size")
    if not (isinstance(size, int) and 0 < size <= MAX_PACKAGE):
        raise PackageError("The signed package description has no valid size, so it was refused.", "bad_manifest")
    return {"edition": o["edition"], "build_id": st["build_id"], "version": str(o.get("version") or ("%s-%s" % (o["edition"], st["build_id"])))[:40],
            "sha256": st["sha256"], "size": size, "url": o["url"], "statement": o["statement"], "signature": o["signature"],
            "kid": str(o.get("kid") or ""), "expires": o.get("expires"), "tiers": o.get("tiers") if isinstance(o.get("tiers"), list) else None,
            "platform": platform, "cuda_major": cuda_major}


def verify_package(path, manifest):
    """Raise PackageError unless the file at `path` has the statement's size and sha256 (the statement's signature is checked again)."""
    pub = pubkey_bytes(manifest.get("kid"))
    if not pub:
        raise PackageError("This build of PXA Control does not know the key this package is signed with, so it will not install it. "
                           "Update PXA Control and try again.", "no_pubkey")
    if not ed25519_verify(pub, SIGN_PREFIX + str(manifest.get("statement") or "").encode(), _decode_sig(manifest.get("signature"))):
        raise PackageError("The package signature is not valid, so it was NOT installed. It was deleted.", "bad_signature")
    want = str(manifest.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", want):
        raise PackageError("The package description has no valid checksum, so it was refused.", "bad_manifest")
    if manifest.get("size") is not None and os.path.getsize(path) != int(manifest["size"]):
        raise PackageError("The downloaded package is the wrong size (damaged or cut short). It was deleted: try again.", "bad_checksum")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != want:
        raise PackageError("The downloaded package is damaged (its checksum does not match). It was deleted: try again.", "bad_checksum")


# ---------------------------------------------------------------------------------------------
# the licence server (JSON over HTTPS; the key travels only in these POST bodies)
# ---------------------------------------------------------------------------------------------
def _check_url(url):
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("https", "http") or not u.hostname:
        raise PackageError("The licence server address is not valid.", "bad_url")
    if u.scheme == "http" and u.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise PackageError("Refusing to talk to the licence server over plain http.", "bad_url")
    return u


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _call(url, body=None, key=None, timeout=20):
    _check_url(url)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as r:
            raw = r.read(MAX_MANIFEST + 1)
    except urllib.error.HTTPError as e:
        try:
            msg = str((json.loads(e.read(4096).decode("utf-8", "replace")) or {}).get("error") or "")
        except (ValueError, AttributeError):
            msg = ""
        raise _http_error(e.code, _scrub(msg, key))
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise PackageError("Cannot reach the licence server. Check your internet connection and try again, "
                           "or use the Free encoder, which needs no key.", "offline")
    if len(raw) > MAX_MANIFEST:
        raise PackageError("The licence server sent an answer that is too large.", "bad_manifest")
    try:
        o = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise PackageError("The licence server sent an answer Control cannot read.", "bad_manifest")
    if not isinstance(o, dict):
        raise PackageError("The licence server sent an answer Control cannot read.", "bad_manifest")
    return o


def _http_error(code, msg):
    tail = (" (" + msg[:160] + ")") if msg else ""
    if code == 401:
        return PackageError("The licence server does not recognise this key." + tail + " Check the key you pasted, or get a fresh one from the "
                            "PXA Network Discord (use /encoder).", "bad_key")
    if code == 403 and re.search(r"looks? shared|is paused|paused because", msg or "", re.I):
        return PackageError("This key is paused because it was used from too many networks in one day. Ask an admin in the PXA Network "
                            "Discord to resume it; a new key does not help while this one is paused.", "paused")
    if code == 403 and re.search(r"already in use on \d+ machines?|different machine", msg or "", re.I):
        return PackageError("This key is already registered to other machines (two at most). Move it to this computer with /encoder reset "
                            "in the PXA Network Discord (once every 30 days), or ask an admin there.", "device_limit")
    if code == 403:
        return PackageError("The licence server refused this key: it is revoked, expired or suspended." + tail
                            + " Get a fresh key from the PXA Network Discord (use /encoder).", "revoked")
    if code == 402:
        return PackageError("This key has no encodes left." + tail, "no_quota")
    if code == 404:
        return PackageError("There is no package for this machine yet." + tail, "no_package")
    if code == 409:
        return PackageError("The licence server does not accept that build any more." + tail, "old_build")
    if code == 410:
        return PackageError("The download link has expired (it works for 15 minutes). Press Download again.", "expired_link")
    if code == 429:
        return PackageError("The licence server is busy or you asked too often. Wait a few minutes and try again.", "busy")
    if code >= 500:
        return PackageError("The licence server is having trouble (%d). Try again later, or use the Free encoder." % code, "server")
    return PackageError("The licence server said %d.%s" % (code, tail), "failed")


def _resolve(base, url):
    return url if re.match(r"^https?://", url) else base.rstrip("/") + "/" + url.lstrip("/")


def free_latest(base, platform=DEFAULT_PLATFORM, cuda_major=DEFAULT_CUDA):
    q = urllib.parse.urlencode({"platform": platform, "cuda_major": cuda_major})
    m = check_statement(_call(base + "/v1/free/latest?" + q), "free", platform, cuda_major)
    m["url"] = _resolve(base, m["url"])
    return m


def request_package(base, key, platform=DEFAULT_PLATFORM, cuda_major=DEFAULT_CUDA, build_id=None):
    if not valid_key(key):
        raise PackageError("That does not look like a PXA Quantizer key. It starts with pxk1. and comes from the PXA Network Discord (/encoder).", "bad_key")
    body = {"key": key.strip(), "platform": platform, "cuda_major": cuda_major}
    if build_id:
        body["build_id"] = build_id
    m = check_statement(_call(base + "/v1/package", body, key.strip()), "pro", platform, cuda_major, build_id)
    m["url"] = _resolve(base, m["url"])
    return m


def latest_pro(base, key, build_id):
    """Is there a Pro build newer than `build_id`? -> {"latest": build id, "update_available": bool}."""
    if not valid_key(key):
        raise PackageError("No key to check updates with.", "bad_key")
    o = _call(base + "/v1/package/latest", {"key": key.strip(), "build_id": build_id}, key.strip())
    latest = o.get("latest")
    if isinstance(latest, dict):
        latest = latest.get("build_id")
    if not isinstance(latest, str) or not re.fullmatch(r"[A-Za-z0-9._\-]{1,64}", latest):
        raise PackageError("The licence server's update answer cannot be read.", "bad_manifest")
    return {"latest": latest, "update_available": o.get("update_available") is True, "edition": "pro"}


# ---------------------------------------------------------------------------------------------
# the GPU runtime: cuBLAS / cuSOLVER (and what they load), shared by every user, installed once
# ---------------------------------------------------------------------------------------------
# The Pro encoder library links the NVIDIA CUDA 12 libraries cuBLAS and cuSOLVER. A normal computer has the NVIDIA driver only, so the licence
# server publishes them as a signed "runtime pack" (NVIDIA redistributables, the licence text is inside; the same bytes for every user, NOT stamped).
# Control downloads it once, verifies it exactly like a package (signed statement, sha256, MANIFEST) and unpacks it to <install root>/runtime/<id>/lib.
# The encoder's wrapper (pxqe_runtime.py) looks there first (PXQE_RUNTIME_DIR, or the runtime folder next to the encoders); an older wrapper is given the
# folder through LD_LIBRARY_PATH by pxa_encode_adapter.runtime_env().
RUNTIME_EDITION = "runtime"
RUNTIME_DIR = "runtime"
RUNTIME_OK = ".pxqe-runtime-ok"                          # written last: a folder without it is not an installed pack
RUNTIME_NEEDS = ("libcublas.so.12", "libcusolver.so.11")        # what the encoder library links (the pack carries their dependencies next to them)
MAX_RUNTIME = 6 << 30


def runtime_root(root=None):
    return os.path.join(root or install_root(), RUNTIME_DIR)


def runtime_latest(base, platform=DEFAULT_PLATFORM, cuda_major=DEFAULT_CUDA, timeout=10):
    """The newest GPU runtime pack the licence server offers (no key). -> the verified manifest, plus libs / unpacked / cuda / notice."""
    q = urllib.parse.urlencode({"platform": platform, "cuda_major": cuda_major})
    o = _call(base + "/v1/runtime/latest?" + q, timeout=timeout)
    m = check_statement(o, RUNTIME_EDITION, platform, cuda_major)
    m["url"] = _resolve(base, m["url"])
    un = o.get("unpacked")
    m["unpacked"] = un if isinstance(un, int) and not isinstance(un, bool) and 0 < un <= MAX_RUNTIME else None
    m["libs"] = [str(x)[:40] for x in o.get("libs") or [] if isinstance(x, str)][:12]
    m["cuda"] = str(o.get("cuda") or "")[:16]
    m["notice"] = str(o.get("notice") or "")[:300]
    if m["size"] > MAX_RUNTIME:
        raise PackageError("The GPU runtime the server describes is larger than expected, so it was refused.", "bad_manifest")
    return m


def installed_runtime(root=None):
    """The newest installed GPU runtime pack -> {id, dir, lib_dir, libs, size, cuda}, or None. Cheap (existence + sizes of the files runtime.json lists):
    the sha256 of every file was checked when it was installed."""
    base = runtime_root(root)
    try:
        names = sorted((n for n in os.listdir(base) if not n.startswith(".")), reverse=True)
    except OSError:
        return None
    for name in names:
        d = os.path.join(base, name)
        if not os.path.isfile(os.path.join(d, RUNTIME_OK)):
            continue
        try:
            with open(os.path.join(d, "runtime.json")) as f:
                meta = json.load(f)
            sizes = meta["sizes"]
            ok = isinstance(sizes, dict) and sizes and all(os.path.isfile(os.path.join(d, k)) and os.path.getsize(os.path.join(d, k)) == v for k, v in sizes.items())
            libs = [str(x) for x in meta.get("libs") or []]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if ok and all(n in libs and os.path.isfile(os.path.join(d, "lib", n)) for n in RUNTIME_NEEDS):
            return {"id": str(meta.get("runtime_id") or name), "dir": d, "lib_dir": os.path.join(d, "lib"), "libs": libs, "size": sum(sizes.values()), "cuda": str(meta.get("cuda") or "")}
    return None


def install_runtime(archive, manifest, root=None):
    """Verify (signed statement, size, sha256), extract, check the pack's own MANIFEST and runtime.json, move into <root>/runtime/<id>/. -> installed_runtime()."""
    verify_package(archive, manifest)
    base = runtime_root(root)
    rid = str(manifest["build_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-]{0,63}", rid) or ".." in rid:
        raise PackageError("The GPU runtime has an id Control will not use as a folder name, so it was refused.", "bad_manifest")
    final = os.path.join(base, rid)
    tmp = os.path.join(base, "." + rid + ".installing")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        _safe_extract_stream(archive, tmp)
        verify_manifest_files(tmp)
        try:
            with open(os.path.join(tmp, "runtime.json")) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            raise PackageError("The GPU runtime does not describe itself (runtime.json is missing), so it was refused.", "bad_package")
        if not isinstance(meta, dict) or meta.get("runtime_id") != rid or not isinstance(meta.get("sizes"), dict):
            raise PackageError("The GPU runtime is not what the licence server described (its id differs), so it was refused.", "bad_package")
        for rel, size in meta["sizes"].items():
            p = os.path.join(tmp, rel)
            if not os.path.isfile(p) or os.path.getsize(p) != size:
                raise PackageError("A file of the GPU runtime is missing or the wrong size (%s), so it was refused." % str(rel)[:60], "bad_package")
        missing = [n for n in RUNTIME_NEEDS if not os.path.isfile(os.path.join(tmp, "lib", n))]
        if missing:
            raise PackageError("The GPU runtime does not contain %s, so it was refused." % ", ".join(missing), "bad_package")
        with open(os.path.join(tmp, RUNTIME_OK), "w") as f:
            f.write(rid + "\n")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    for n in os.listdir(base):                       # older packs are of no use any more (a run that has one open keeps it until it exits)
        if n != rid and not n.startswith("."):
            shutil.rmtree(os.path.join(base, n), ignore_errors=True)
    info = installed_runtime(root)
    if not info:
        raise PackageError("The GPU runtime was unpacked but is not complete.", "broken")
    return info


def _need_disk(path, need, what):
    """Refuse a download that cannot fit, with the numbers: PackageError('disk_full')."""
    p = path
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        free = shutil.disk_usage(p or ".").free
    except OSError:
        return
    if free < need:
        raise PackageError("Not enough disk space for the %s: it needs about %.1f GB free (the download and what it unpacks to), and %.1f GB is free in %s. "
                           "Free some space, or set PXA_ENCODER_HOME to a bigger drive." % (what, need / 1e9, free / 1e9, p), "disk_full")


def install_runtime_from_manifest(manifest, root=None, work=None, progress=None, cancelled=None):
    """Download the runtime pack into `work`, then verify + install it. Deletes the archive on any failure except an interrupted download (it resumes)."""
    root = root or install_root()
    work = work or os.path.join(root, "_download")
    os.makedirs(work, exist_ok=True)
    _need_disk(runtime_root(root), int((manifest.get("size") or 0) + (manifest.get("unpacked") or manifest.get("size") or 0) * 1.05), "GPU runtime")
    dest = os.path.join(work, "pxqe-runtime-%s.pack" % manifest["build_id"])          # an archive (xz today): the extractor reads whatever compression it has
    try:
        download(manifest["url"], dest, manifest.get("size"), progress, cancelled)
        info = install_runtime(dest, manifest, root)
    except PackageError as e:
        if e.code not in ("offline", "cancelled", "download"):
            for p in (dest, dest + ".part"):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        raise
    try:
        os.unlink(dest)
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------------------------
# download + install
# ---------------------------------------------------------------------------------------------
def download(url, dest, size=None, progress=None, cancelled=None):
    """Stream `url` to `dest` (resuming a .part file with Range when the server supports it). The key is never sent here."""
    _check_url(url)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    hdr = {"User-Agent": USER_AGENT}
    if have:
        hdr["Range"] = "bytes=%d-" % have
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:       # the .part is already the whole file
            r = None
        elif e.code == 410:
            raise _http_error(410, "")
        else:
            raise PackageError("The download failed (the server said %d)." % e.code, "download")
    except (urllib.error.URLError, OSError, TimeoutError):
        raise PackageError("The download was interrupted: cannot reach the server. Try again (it resumes where it stopped).", "offline")
    if r is not None:
        total = int(r.headers.get("Content-Length") or 0)
        if r.status == 200 and have:
            have = 0                       # server ignored Range: start over
        if total + have > MAX_PACKAGE:
            raise PackageError("The package is larger than expected, so it was not downloaded.", "bad_manifest")
        mode = "ab" if have and r.status == 206 else "wb"
        done = have if mode == "ab" else 0
        try:
            with r, open(part, mode) as f:
                while True:
                    if cancelled and cancelled():
                        raise PackageError("Download cancelled. It resumes where it stopped.", "cancelled")
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    if progress:
                        progress(done, size or (total + (have if mode == "ab" else 0)) or None)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise PackageError("The download was interrupted. Try again (it resumes where it stopped).", "offline")
    os.replace(part, dest)
    return dest


def _safe_extract(archive, into):
    """tar(.gz) -> `into`, refusing anything that is not a plain file or folder under `into`. A single top-level folder (the
    packages use `pxqe/`) is stripped, so the command line lands at <into>/pxqe whatever the archive's top folder is called."""
    base = os.path.realpath(into)
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        if len(members) > 5000:
            raise PackageError("The package has too many files, so it was refused.", "bad_package")
        for m in members:
            name = m.name
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                raise PackageError("The package contains an unsafe path, so it was refused.", "bad_package")
            if not (m.isfile() or m.isdir()):
                raise PackageError("The package contains a link or special file, so it was refused.", "bad_package")
        tops = {m.name.strip("/").split("/")[0] for m in members if m.name.strip("/")}
        strip = len(tops) == 1 and all(m.isdir() if m.name.strip("/") == list(tops)[0] else True for m in members)
        top = list(tops)[0] if strip else None
        for m in members:
            rel = m.name.strip("/")
            if top is not None:
                rel = rel[len(top):].lstrip("/")
            if not rel:
                continue
            tgt = os.path.realpath(os.path.join(base, rel))
            if tgt != base and not tgt.startswith(base + os.sep):
                raise PackageError("The package contains an unsafe path, so it was refused.", "bad_package")
            if m.isdir():
                os.makedirs(tgt, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            src = tf.extractfile(m)
            with open(tgt, "wb") as out:
                shutil.copyfileobj(src, out)
            os.chmod(tgt, 0o755 if (m.mode & 0o111 or re.match(r"^pxqe$", os.path.basename(rel))) else 0o644)


def _safe_extract_stream(archive, into, max_files=5000):
    """Like _safe_extract for ONE top-level folder, but in a single pass (tarfile stream mode): a compressed archive of 1.5 GB is read once, not twice (listing every member first
    would decompress all of it just to read the headers). Every member is checked BEFORE it is written, and the whole thing lands in a temp folder the caller deletes on failure."""
    base = os.path.realpath(into)
    top, count = None, 0
    with tarfile.open(archive, "r|*") as tf:
        for m in tf:
            count += 1
            if count > max_files:
                raise PackageError("The package has too many files, so it was refused.", "bad_package")
            name = m.name
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                raise PackageError("The package contains an unsafe path, so it was refused.", "bad_package")
            if not (m.isfile() or m.isdir()):
                raise PackageError("The package contains a link or special file, so it was refused.", "bad_package")
            parts = name.strip("/").split("/")
            if not parts or not parts[0]:
                continue
            if top is None:
                top = parts[0]
            if parts[0] != top:
                raise PackageError("The package has more than one top-level folder, so it was refused.", "bad_package")
            rel = "/".join(parts[1:])
            if not rel:
                continue
            tgt = os.path.realpath(os.path.join(base, rel))
            if tgt != base and not tgt.startswith(base + os.sep):
                raise PackageError("The package contains an unsafe path, so it was refused.", "bad_package")
            if m.isdir():
                os.makedirs(tgt, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            with open(tgt, "wb") as out:
                shutil.copyfileobj(tf.extractfile(m), out, 1 << 20)
            os.chmod(tgt, 0o755 if (m.mode & 0o111) else 0o644)


def verify_manifest_files(root):
    """The package's own MANIFEST.json (sha256 of every file): when MANIFEST.sig is there it must verify, and in every case each listed
    file must match its checksum. A package with no manifest at all is accepted: the archive itself was already verified against the
    signed statement (the Free package ships a manifest the server may not have signed)."""
    mj, ms = os.path.join(root, "MANIFEST.json"), os.path.join(root, "MANIFEST.sig")
    if not os.path.exists(mj) and not os.path.exists(ms):
        return
    try:
        with open(mj, "rb") as f:
            raw = f.read()
        man = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        raise PackageError("The package's file list is missing or unreadable, so it was refused.", "bad_package")
    if os.path.exists(ms):
        try:
            with open(ms) as f:
                sig = f.read().strip()
        except OSError:
            raise PackageError("The package's file list signature is unreadable, so it was refused.", "bad_package")
        pubs = [pubkey_bytes(k) for k in list(PACKAGE_PUBKEYS) + [None]]
        if not any(p and ed25519_verify(p, MANIFEST_PREFIX + raw, _decode_sig(sig)) for p in pubs):
            raise PackageError("The package's file list is not signed correctly, so it was refused.", "bad_signature")
    files = man.get("files") if isinstance(man, dict) else None
    if not isinstance(files, dict):
        raise PackageError("The package's file list is not valid, so it was refused.", "bad_package")
    for rel, want in files.items():
        p = os.path.realpath(os.path.join(root, rel))
        if not p.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(p):
            raise PackageError("A file the package lists is missing (%s), so it was refused." % str(rel)[:60], "bad_package")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != want:
            raise PackageError("A file in the package does not match its checksum (%s), so it was refused." % str(rel)[:60], "bad_package")


def install(archive, manifest, root=None):
    """Verify, extract, check and move into <root>/<edition>/<build_id>/. -> the probe() dict of the installed CLI."""
    verify_package(archive, manifest)
    root = root or install_root()
    final = os.path.join(root, manifest["edition"], manifest["build_id"])
    tmp = final + ".installing"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        _safe_extract(archive, tmp)
        verify_manifest_files(tmp)
        from pxa_encode_adapter import cli_in_dir
        cli = cli_in_dir(tmp)
        if not cli:
            raise PackageError("The package does not contain the encoder command line, so it was refused.", "bad_package")
        info = probe(cli)
        if not info["ok"]:
            raise PackageError("The package installed, but the encoder will not start: %s" % (info["error"] or "no answer"), "broken")
        if info["edition"] != manifest["edition"] or info["build_id"] != manifest["build_id"]:
            raise PackageError("The package is not what the licence server described (edition or build differ), so it was refused.", "bad_package")
        shutil.rmtree(final, ignore_errors=True)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return probe(cli_in_dir(final))


def install_from_manifest(manifest, root=None, work=None, progress=None, cancelled=None):
    """Download the manifest's package into `work`, then verify + install it. Deletes the archive on any failure."""
    work = work or os.path.join(root or install_root(), "_download")
    os.makedirs(work, exist_ok=True)
    dest = os.path.join(work, "pxqe-%s-%s.tar.gz" % (manifest["edition"], manifest["build_id"]))
    try:
        download(manifest["url"], dest, manifest.get("size"), progress, cancelled)
        info = install(dest, manifest, root)
    except PackageError as e:
        if e.code not in ("offline", "cancelled", "download"):      # keep a half download for Range resume
            for p in (dest, dest + ".part"):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        raise
    try:
        os.unlink(dest)
    except OSError:
        pass
    return info


def newer(installed_builds, manifest):
    """True when the manifest's build is not among the builds already installed for that edition."""
    return bool(manifest) and manifest["build_id"] not in set(installed_builds or [])
