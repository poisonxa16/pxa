"""pxa_lib_update.py - the licensed PXQN library (`libggml-pxqn.so`) as its own signed, updatable artifact.

A supporter install carries the closed library inside the engine release, so a new library used to mean a new engine tarball. This module
updates the library on its own, from the licence server, without touching the engine:

  * `check`  - POST <licence>/v1/lib/latest {key, channel, platform, cuda_major, current} -> the newest release for that channel.
  * `apply`  - download every file, verify it, stage it beside the engine versions, and switch `lib/current` to it in one rename.
  * `rollback` - switch `lib/current` back to the previous release, or to what the engine shipped before the first update.

The layout (everything below `<install>/lib/`, next to the version directories `current` points at):

    <install>/lib/<lib_id>/<name>     the release, exactly the file names the signed manifest lists
    <install>/lib/current -> <lib_id> the active release
    <install>/lib/previous -> <lib_id>|shipped   the last good one (rollback always has a target)
    <install>/lib/shipped/<name>      the files the engine release shipped, kept the first time one is replaced
    <install>/lib/lib-state.json      channel, installed version, last check (never the licence key)

and inside the version directory each manifest name (`lib/libggml-pxqn.so`, `lib-compat/libggml-pxqn.so` - a v3.1 install carries two
copies of the same library) becomes a relative symlink into `../../lib/current/`. That is what makes the swap atomic for BOTH copies at
once: the loader keeps looking where it always looked, and one rename of `lib/current` moves the whole install to the new library.

A manifest that arrives over the network is only a proposal until its Ed25519 signature verifies against the key id it names, using the
same trust anchor and the same verifier as the quantizer packages (`pxa_encode_pkg`). Every trusted value used here - version, channel,
min_engine, file names, sha256s, sizes - is read from the SIGNED statement, never from the unsigned envelope around it.

Why this is Python and not C: `pxa-update.c` owns the engine releases and has no crypto in it. Verifying here means one implementation of
the trust decision, shared with the package path, instead of a second Ed25519 in C. `pxa-update lib ...` execs this module.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time

import pxa_encode_pkg as pkg
from pxa_encode_pkg import PackageError

SIGN_PREFIX = b"pxqn-lib|"                              # a library statement can never pass for a package or a runtime one
LIB_NAME = "libggml-pxqn.so"
CHANNELS = ("stable", "beta")
STATE_NAME = "lib-state.json"
MAX_FILE = 512 << 20                                    # a library is tens of MB; this is a sanity bound on a hostile Content-Length
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")   # relative, no `..`, no leading slash, no empty segment
LIB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,48}$")
VERSION_MAX = 40


# -------------------------------------------------------------------------------------------- paths
def install_root(d=None):
    """The directory that holds the engine version directories (and so `current`). `--dir` wins, then the same env the CLI uses, then
    the per-user default - the library lives beside the versions, which is where the loader's own directory points through the links."""
    if d:
        return os.path.abspath(d)
    return os.path.abspath(os.environ.get("PXA_INSTALL_DIR") or os.path.join(os.path.expanduser("~"), ".local", "share", "pxa"))


def lib_dir(d=None):
    return os.path.join(install_root(d), "lib")


def state_path(d=None):
    return os.path.join(lib_dir(d), STATE_NAME)


def load_state(d=None):
    try:
        with open(state_path(d)) as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(d=None, **kw):
    """Merge the given fields into the state file and write it atomically. The licence key is never a field here."""
    p = state_path(d)
    s = load_state(d)
    s.update({k: v for k, v in kw.items() if v is not None})
    d0 = os.path.dirname(p)
    if d0:
        os.makedirs(d0, exist_ok=True)
    # A per-writer temp name: the Control page's poller and a request can both check at once, and a
    # fixed ".new" lets one rename the other's file out from under it (spurious ENOENT on the state).
    tmp = "%s.new.%d.%d" % (p, os.getpid(), threading.get_ident())
    with open(tmp, "w") as f:
        json.dump(s, f, sort_keys=True)
    os.replace(tmp, p)
    return s


def installed_version(d=None):
    """The library version this install is running: what we last installed, or '' when nothing has been installed over the engine's own."""
    return str(load_state(d).get("version") or "")


# -------------------------------------------------------------------------------------------- the key
def licence_key(d=None, key=None):
    """The key from an explicit argument, then the environment, then the same control.json field the encoder uses. Never written to state."""
    if key:
        return key.strip()
    k = os.environ.get("PXA_LICENCE_KEY")
    if k and pkg.valid_key(k):
        return k.strip()
    try:
        import pxa_control
        cfg = pxa_control.load_config()
    except Exception:  # noqa: BLE001  (Control may not be importable: the CLI still works with --key / the env)
        cfg = {}
    enc = cfg.get("encode") if isinstance(cfg.get("encode"), dict) else {}
    k = (enc or {}).get("licence_key") or cfg.get("licence_key") or ""
    return k.strip() if isinstance(k, str) else ""


# -------------------------------------------------------------------------------------------- the server
def _call(base, path, body=None, timeout=30):
    """One request to the licence server. The key travels only in a POST body, never in a URL (a URL ends up in proxy logs)."""
    url = base.rstrip("/") + path
    pkg._check_url(url)
    data = json.dumps(body).encode() if body is not None else None
    req = pkg.urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json", "User-Agent": pkg.USER_AGENT, "Accept": "application/json"})
    try:
        with pkg.urllib.request.build_opener(pkg._NoRedirect()).open(req, timeout=timeout) as r:
            raw = r.read(pkg.MAX_MANIFEST + 1)
    except pkg.urllib.error.HTTPError as e:
        try:
            msg = str((json.loads(e.read(4096).decode("utf-8", "replace")) or {}).get("error") or "")
        except (ValueError, AttributeError):
            msg = ""
        raise _http_error(e.code, pkg._scrub(msg, body.get("key") if isinstance(body, dict) else None))
    except (pkg.urllib.error.URLError, OSError, TimeoutError):
        raise PackageError("Cannot reach the licence server. Check your internet connection and try again.", "offline")
    if len(raw) > pkg.MAX_MANIFEST:
        raise PackageError("The licence server sent an answer that is too large.", "bad_manifest")
    try:
        o = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise PackageError("The licence server sent an answer that cannot be read.", "bad_manifest")
    if not isinstance(o, dict):
        raise PackageError("The licence server sent an answer that cannot be read.", "bad_manifest")
    return o


def _http_error(code, msg):
    tail = (" (" + msg[:160] + ")") if msg else ""
    if code == 401:
        return PackageError("The licence server does not recognise this key." + tail + " Check the key you pasted, or get a fresh one "
                            "from the PXA Network Discord (use /encoder).", "bad_key")
    if code == 403 and re.search(r"beta", msg or "", re.I):
        return PackageError("The beta library channel is for Valued Supporters. Nothing was changed; the stable channel is unaffected.", "not_valued")
    if code == 403:
        return PackageError("Updates are paused: the licence server refused this key - it is revoked, expired or suspended." + tail
                            + " The library already installed keeps working; nothing was changed.", "revoked")
    if code == 404:
        return PackageError("There is no library release for this machine or channel yet." + tail, "no_release")
    if code == 410:
        return PackageError("The download link has expired (it works for 15 minutes). Press Check again.", "expired_link")
    if code == 429:
        return PackageError("The licence server is busy or you asked too often. Wait a few minutes and try again.", "busy")
    if code >= 500:
        return PackageError("The licence server is having trouble (%d). Try again later." % code, "server")
    return PackageError("The licence server said %d.%s" % (code, tail), "failed")


# -------------------------------------------------------------------------------------------- the manifest
def verify_manifest(o, base, channel, platform, cuda_major):
    """The server's answer -> the release to install. Raises PackageError unless the statement is signed by a key we trust AND the
    signed content says what was asked for. Everything returned comes from the signed statement."""
    if not isinstance(o, dict) or not isinstance(o.get("manifest"), str) or not o["manifest"] or not isinstance(o.get("signature"), str):
        raise PackageError("The licence server's library description is incomplete, so nothing was changed.", "bad_manifest")
    pub = pkg.pubkey_bytes(o.get("kid"))
    if not pub:
        raise PackageError("This build of PXA does not know the key (%s) this library is signed with, so it will not install it. "
                           "Update the engine and try again." % str(o.get("kid") or "none")[:32], "no_pubkey")
    if not pkg.ed25519_verify(pub, SIGN_PREFIX + o["manifest"].encode(), pkg._decode_sig(o["signature"])):
        raise PackageError("The library release signature is not valid, so nothing was installed. If this keeps happening, tell us in the "
                           "PXA Network Discord.", "bad_signature")
    try:
        st = json.loads(o["manifest"])
    except ValueError:
        raise PackageError("The signed library description cannot be read, so it was refused.", "bad_manifest")
    if not isinstance(st, dict):
        raise PackageError("The signed library description cannot be read, so it was refused.", "bad_manifest")
    if st.get("edition") != "lib" or str(st.get("lib") or "") != LIB_NAME:
        raise PackageError("The signed description is not a PXQN library release, so it was refused.", "bad_statement")
    if str(st.get("channel") or "") != channel or str(st.get("platform") or "") != platform or st.get("cuda_major") != cuda_major:
        raise PackageError("The signed library description does not match what was asked for (channel, platform or CUDA version), so it was refused.",
                           "bad_statement")
    lib_id = str(st.get("lib_id") or "")
    version = str(st.get("version") or "")
    if not LIB_ID_RE.match(lib_id) or not version or len(version) > VERSION_MAX:
        raise PackageError("The signed library description has no valid release id or version, so it was refused.", "bad_manifest")
    files = st.get("files")
    if not isinstance(files, list) or not files:
        raise PackageError("The signed library description lists no files, so it was refused.", "bad_manifest")
    out_files = []
    for e in files:
        if not isinstance(e, dict):
            raise PackageError("The signed library description is malformed, so it was refused.", "bad_manifest")
        name = str(e.get("name") or "")
        sha = str(e.get("sha256") or "")
        size = e.get("size")
        if not name or len(name) > 120 or not NAME_RE.match(name) or ".." in name.split("/"):
            raise PackageError("The signed library description names a file outside the release, so it was refused.", "bad_manifest")
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or not isinstance(size, int) or not (0 < size <= MAX_FILE):
            raise PackageError("The signed library description has no valid checksum or size for %s, so it was refused." % name, "bad_manifest")
        out_files.append({"name": name, "sha256": sha, "size": int(size)})
    return {"lib_id": lib_id, "version": version, "channel": channel, "platform": platform, "cuda_major": cuda_major,
            "min_engine": str(st.get("min_engine") or ""), "sm": [int(x) for x in st.get("sm") or [] if isinstance(x, int)],
            "files": out_files, "kid": str(o.get("kid") or ""), "manifest": o["manifest"], "signature": o["signature"],
            "base": base.rstrip("/")}


def _tickets(o, rel):
    """The download URLs from the unsigned envelope, checked against the one place a library file may come from: this server's own path.
    A URL for a name the signed manifest does not list is ignored, so a hostile envelope cannot add a file."""
    out = {}
    for e in (o.get("files") if isinstance(o.get("files"), list) else []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "")
        url = str(e.get("url") or "")
        if name in [f["name"] for f in rel["files"]] and url.startswith(rel["base"] + "/v1/lib/download/"):
            out[name] = url
    missing = [f["name"] for f in rel["files"] if f["name"] not in out]
    if missing:
        raise PackageError("The licence server did not give a download link for %s, so nothing was changed." % missing[0], "bad_manifest")
    return out


def lib_latest(base, key, channel="stable", platform=None, cuda_major=None, current="", timeout=30):
    """Ask the licence server for the newest library release on a channel. Returns (release, raw answer)."""
    if channel not in CHANNELS:
        raise PackageError("Unknown channel %r: use stable or beta." % str(channel)[:20], "bad_channel")
    if not pkg.valid_key(key):
        raise PackageError("No licence key is set. Put one in PXA Control (or PXA_LICENCE_KEY) and try again.", "no_key")
    if platform is None or cuda_major is None:
        p, c = pkg.detect_platform()
        platform = platform or p
        cuda_major = c if cuda_major is None else cuda_major
    body = {"key": key, "channel": channel, "platform": platform, "cuda_major": int(cuda_major), "current": current}
    o = _call(base, "/v1/lib/latest", body, timeout=timeout)
    return verify_manifest(o, base, channel, platform, int(cuda_major)), o


# -------------------------------------------------------------------------------------------- versions
def _vtuple(v):
    """'v3.1.1' -> (1, 3, 1, 1) and a date tag 'v2026.10.1' -> (0, 2026, 10, 1): the month/year line is the OLD line and sorts
    below every v3 release, which is the rule pxa-update.c uses, so the CLI and this module never disagree about which of two
    versions is newer. The leading tag keeps tuples of both shapes comparable; (-1,) is a version that cannot be read."""
    s = str(v or "").strip()
    if s[:1] in ("v", "V"):
        s = s[1:]
    nums = re.findall(r"\d+", s)
    if not nums:
        return (-1,)
    if int(nums[0]) >= 2000:
        return (0,) + tuple(int(x) for x in nums)
    return (1,) + tuple(int(x) for x in nums)


def min_engine_ok(installed, need):
    """True when the installed engine is at least `need` (an empty `need` means the release does not care)."""
    if not need:
        return True
    i, n = _vtuple(installed), _vtuple(need)
    if i == (-1,) or n == (-1,):
        return False                    # a version we cannot read is not a version we may assume is new enough
    return (i[0] == 0) == (n[0] == 0) and i >= n


def engine_version(d=None):
    """The engine version at <dir>/current/VERSION (or the version directory's own name), "" when there is none."""
    root = install_root(d)
    p = os.path.join(root, "current", "VERSION")
    try:
        with open(p) as f:
            head = f.readline().strip()
    except OSError:
        head = ""
    if head:
        return head[4:].strip() if head.startswith("tag:") else head.split()[0]
    try:
        return os.path.basename(os.path.realpath(os.path.join(root, "current")))
    except OSError:
        return ""


# -------------------------------------------------------------------------------------------- layout
def _symlink(from_path, to_path):
    """Point to_path at from_path with a rename, so a reader either sees the old link or the new one and never a half-made one."""
    tmp = to_path + ".pxa-new"
    if os.path.islink(tmp) or os.path.exists(tmp):
        os.remove(tmp)
    os.symlink(from_path, tmp)
    os.replace(tmp, to_path)


def _rel_link(path):
    try:
        t = os.readlink(path)
    except OSError:
        return ""
    return t if t and "/" not in t else ""


def server_running(d=None):
    """True when a process from this install is running: the same /proc scan pxa-update.c does for engine releases. A library is swapped
    under a live server, so the check has to pass first. Falls back to a `current`-under-this-root test when /proc is not readable."""
    root = os.path.realpath(install_root(d))
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return False
    for pid in pids:
        try:
            exe = os.readlink("/proc/%s/exe" % pid)
        except OSError:
            continue
        if not (exe == root or exe.startswith(root + os.sep)):
            continue
        if "llama-server" in exe or "pxa-launch" in exe:
            return True
    return False


def link_files(d, names):
    """Make each manifest name inside the version directory a relative symlink into ../../lib/current/. A file the engine shipped is kept
    first (under lib/shipped/) so a rollback can put those exact bytes back. Idempotent: an existing correct link is left alone."""
    root = install_root(d)
    cur = os.path.join(root, "current")
    moved = []
    for name in names:
        target = os.path.join(cur, name)
        tdir = os.path.dirname(target)
        if tdir:
            os.makedirs(tdir, exist_ok=True)
        want = os.path.join("..", "..", "lib", "current", name)
        if os.path.islink(target) and os.readlink(target) == want:
            continue
        if os.path.exists(target) and not os.path.islink(target):
            keep = os.path.join(lib_dir(d), "shipped", name)
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            if not os.path.exists(keep):
                shutil.move(target, keep)           # the engine's own copy, kept byte-for-byte for the rollback to "shipped"
                moved.append(name)
            else:
                os.remove(target)
        _symlink(want, target)
    return moved


def unlink_files(d, names):
    """Put the engine's shipped files back in place of the links, for a rollback to `shipped`."""
    root = install_root(d)
    cur = os.path.join(root, "current")
    for name in names:
        target = os.path.join(cur, name)
        keep = os.path.join(lib_dir(d), "shipped", name)
        if os.path.exists(keep):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.islink(target) or os.path.exists(target):
                os.remove(target)
            shutil.move(keep, target)


def release_names(d, lib_id):
    """Every file name of a staged release, from its own directory tree."""
    base = os.path.join(lib_dir(d), lib_id)
    out = []
    for r, _, fs in os.walk(base):
        for fn in fs:
            out.append(os.path.relpath(os.path.join(r, fn), base).replace(os.sep, "/"))
    return sorted(out)


def installed_links(d=None):
    """The names the version directory currently links into lib/current (what the loader will actually open)."""
    root = install_root(d)
    cur = os.path.join(root, "current")
    out = []
    for name in ("lib", "lib-compat"):
        p = os.path.join(cur, name, LIB_NAME)
        if os.path.islink(p) and os.readlink(p) == os.path.join("..", "..", "lib", "current", name + "/" + LIB_NAME):
            out.append("%s/%s" % (name, LIB_NAME))
        elif os.path.exists(p):
            out.append("shipped:" + name + "/" + LIB_NAME)
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stage(d, rel, tickets, progress=None):
    """Download and verify every file of a release into lib/<lib_id>.partial/, then rename it into lib/<lib_id>. Nothing outside that
    directory is touched, so a failure anywhere leaves the installed library exactly as it was."""
    lib = lib_dir(d)
    os.makedirs(lib, exist_ok=True)
    partial = os.path.join(lib, rel["lib_id"] + ".partial")
    if os.path.isdir(partial):
        shutil.rmtree(partial)
    os.makedirs(partial)
    try:
        for ent in rel["files"]:
            dest = os.path.join(partial, ent["name"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            pkg.download(tickets[ent["name"]], dest, size=ent["size"], progress=progress)
            if os.path.getsize(dest) != ent["size"] or _sha256(dest) != ent["sha256"]:
                raise PackageError("The downloaded library file %s does not match the signed checksum, so nothing was installed." % ent["name"],
                                   "bad_download")
        final = os.path.join(lib, rel["lib_id"])
        if os.path.isdir(final):
            shutil.rmtree(final)
        os.replace(partial, final)
        return final
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _channel(d, channel):
    return (channel or load_state(d).get("channel") or "stable")


def check(d=None, base=None, key=None, channel=None, platform=None, cuda_major=None):
    """Ask the server what is available. Changes nothing on disk except the state's last-check stamp: no download, no swap.
    -> (release, answer, installed version). The caller compares versions; the server's own `update_available` field is not trusted."""
    d = install_root(d)
    base = base or pkg.licence_url(None)
    channel = _channel(d, channel)
    key = licence_key(d, key)
    cur = installed_version(d)
    rel, raw = lib_latest(base, key, channel, platform, cuda_major, current=cur)
    save_state(d, channel=channel, checked=int(time.time()))
    return rel, raw, cur


def apply(d=None, base=None, key=None, channel=None, platform=None, cuda_major=None, progress=None, force=False):
    """Fetch the newest release on the channel and install it. Refuses while a server from this install is running.
    -> (applied, release|None, installed version before)."""
    d = install_root(d)
    base = base or pkg.licence_url(None)
    channel = _channel(d, channel)
    key = licence_key(d, key)
    cur = installed_version(d)
    rel, raw = lib_latest(base, key, channel, platform, cuda_major, current=cur)
    if rel["version"] == cur and not force:
        save_state(d, channel=channel, checked=int(time.time()))
        return False, None, cur
    eng = engine_version(d)
    if not min_engine_ok(eng, rel["min_engine"]):
        raise PackageError("This library release needs engine %s or newer and this install is %s. Update the engine first."
                           % (rel["min_engine"], eng or "unknown"), "engine_too_old")
    if server_running(d):
        raise PackageError("A server is running from this install. Stop it and try again - the library is swapped under it, "
                           "and a running server has the old one open.", "server_running")
    st = load_state(d)
    prev = st.get("lib_id") or ("shipped" if os.path.isdir(os.path.join(lib_dir(d), "shipped")) else "")
    if prev == rel["lib_id"]:
        prev = st.get("prev") or ""
    tickets = _tickets(raw, rel)
    stage(d, rel, tickets, progress=progress)
    _symlink(rel["lib_id"], os.path.join(lib_dir(d), "current"))
    try:
        moved = link_files(d, [f["name"] for f in rel["files"]])
    except BaseException:
        # the verified release stays staged; the links go back to what they were, so the install keeps running what it ran
        if prev and prev != "shipped":
            _symlink(prev, os.path.join(lib_dir(d), "current"))
            link_files(d, release_names(d, prev))
        raise
    if not prev and moved:
        prev = "shipped"        # the first update keeps the engine's own bytes in lib/shipped: that is what a rollback returns to
    if prev:
        _symlink(prev, os.path.join(lib_dir(d), "previous"))
    save_state(d, channel=channel, version=rel["version"], lib_id=rel["lib_id"], prev=prev or None, prev_version=cur or None,
               sha256={f["name"]: f["sha256"] for f in rel["files"]}, applied=int(time.time()), checked=int(time.time()))
    return True, rel, cur


def rollback(d=None):
    """Switch lib/current back to the previous release, or (when the previous is `shipped`) to the files the engine release brought.
    -> (restored version, what it was)."""
    d = install_root(d)
    lib = lib_dir(d)
    if server_running(d):
        raise PackageError("A server is running from this install. Stop it and try again.", "server_running")
    prev = _rel_link(os.path.join(lib, "previous"))
    if not prev:
        raise PackageError("There is nothing to roll back to.", "no_previous")
    st = load_state(d)
    was, was_ver = st.get("lib_id"), str(st.get("version") or "")
    if prev == "shipped":
        names = release_names(d, was) or ["lib/" + LIB_NAME, "lib-compat/" + LIB_NAME]
        unlink_files(d, names)                       # puts the engine's own bytes back where the links were
        for gone in ("current", "previous"):         # nothing is installed and there is nothing left to roll back to
            try:
                os.remove(os.path.join(lib, gone))
            except OSError:
                pass
        save_state(d, version="", lib_id=None, prev=None, prev_version=None, sha256=None)
        return "", was_ver
    if not os.path.isdir(os.path.join(lib, prev)):
        raise PackageError("The previous library release is gone from this install.", "no_previous")
    _symlink(prev, os.path.join(lib, "current"))
    names = release_names(d, prev)
    link_files(d, names)
    if was and was != prev:
        _symlink(was, os.path.join(lib, "previous"))
    ver = str(st.get("prev_version") or "") or prev
    shas = {n: _sha256(os.path.join(lib, prev, n)) for n in names if os.path.isfile(os.path.join(lib, prev, n))}
    save_state(d, version=ver, lib_id=prev, prev=was or None, prev_version=was_ver or None, sha256=shas or None)
    return ver, was_ver


# -------------------------------------------------------------------------------------------- CLI
def _usage():
    return ("usage: pxa-update lib check|apply|rollback [--channel stable|beta] [--dir DIR] [--base-url URL] [--key KEY] [--json]\n"
            "       pxa_lib_update.py check|apply|rollback [same options]")


def main(argv):
    cmd, channel, d, base, key, as_json = None, None, None, None, None, False
    i = 0
    args = list(argv)
    if args and args[0] in ("check", "apply", "rollback"):
        cmd = args.pop(0)
    while args:
        a = args.pop(0)
        if a in ("--channel", "--dir", "--base-url", "--key") and args:
            v = args.pop(0)
            if a == "--channel":
                channel = v
            elif a == "--dir":
                d = v
            elif a == "--base-url":
                base = v
            else:
                key = v
        elif a == "--json":
            as_json = True
        elif a in ("-h", "--help"):
            print(_usage())
            return 0
        else:
            print("pxa-update: %s" % _usage(), file=sys.stderr)
            return 1
    if cmd not in ("check", "apply", "rollback"):
        print("pxa-update: %s" % _usage(), file=sys.stderr)
        return 1
    d = install_root(d)
    try:
        if cmd == "check":
            rel, raw, cur = check(d, base, key, channel)
            ch = _channel(d, channel)
            newer = bool(rel["version"]) and rel["version"] != cur
            print("current=%s latest=%s update=%s channel=%s" % (cur or "shipped", rel["version"], "yes" if newer else "no", ch))
            if as_json:
                print(json.dumps({"current": cur, "latest": rel["version"], "update": newer, "channel": ch,
                                  "lib_id": rel["lib_id"], "linked": installed_links(d), "notice": raw.get("notice") or ""}))
            return 0
        if cmd == "apply":
            done, rel, cur = apply(d, base, key, channel)
            ch = _channel(d, channel)
            if not done:
                print("current=%s latest=%s update=no channel=%s" % (cur or "shipped", cur or "none", ch))
                return 0
            print("installed library %s (%s)" % (rel["version"], rel["lib_id"]))
            print("next %s/lib/current" % d)
            return 0
        ver, was = rollback(d)
        print("rolled back to %s" % (ver or "the engine's own library"))
        print("next %s/lib/current" % d)
        return 0
    except PackageError as e:
        print("pxa-update: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
