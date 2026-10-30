"""pxa_engine_update.py - the engine beta channel: new engine builds for Valued Supporters before everyone else.

Stable engine releases come from GitHub through `pxa-update check|apply` (tools/pxa-update.c) and nothing here touches that path. When an
install is on the beta channel (`<install>/engine-channel` says `beta`, or `--channel beta` is given) pxa-update hands the call to this
module:

  * `check`    - POST <licence>/v1/engine/latest {key, channel, platform, cuda_major, current} -> the newest beta for this machine.
  * `apply`    - verify the signed release, download the tarball and its notes, check every byte against the signature, then call back
                 `pxa-update apply-file <tarball> --sha256 <hex>`: the C installer keeps the one archive rule set (one top directory, no
                 models / settings / expert counts) and the atomic `current` / `previous` swap. `pxa-update rollback` works across channels.
  * `channel`  - `channel beta` (only for a Valued key: the server says so) or `channel stable`; it records the choice, it installs nothing.
  * `channels` - what this key may use: `channels=stable,beta beta=yes`.

A supporter without a Valued key gets the beta through the PXA Coins wallet: when the key is missing or not Valued and this computer has
PXA Control's install key (<control config>/plugins/install.key, made by Store > Wallet), the same calls go to licd's
/v1/wallet/engine/{channels,latest}, signed with that install key; licd answers with the same signed statement and keyed tickets when the
wallet redeemed beta access for that release. The install key is only read here, never made.
  * `notes`    - the release notes of the installed (or, with nothing installed, the offered) beta, shown only after their sha256 matches the
                 signed statement again.

Files it writes, all beside the version directories:

    <install>/engine-channel                one word, stable or beta (sticky)
    <install>/engine-state.json             channel, version, build_id, top, sha256s, the signed statement (never the licence key)
    <install>/engine-notes/<build_id>.md    the notes of each beta installed here

Every trusted value - version, channel, platform, file names, sha256s, sizes - is read from the SIGNED statement (`pxa-engine|` + canonical
JSON, Ed25519, the package trust anchor in pxa_encode_pkg), never from the unsigned envelope around it. Nothing is collected or sent back: a
beta tester reports in the PXA Network Discord #support, saying "beta" and the version (the notes say so).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import base64

import pxa_encode_pkg as pkg
from pxa_encode_pkg import PackageError
import pxa_lib_update as lu

BETA_FILES_PREFIX = b"pxa-beta-files|"                   # licd's signed list of a beta's files (engine pair + the Docker image archive)
LINKS_MARGIN_S = 600                                      # a cached personal link is reused while it has 10 minutes left
SIGN_PREFIX = b"pxa-engine|"                            # an engine statement never passes for a library, package or runtime one
CHANNELS = ("stable", "beta")
STATE_NAME = "engine-state.json"
CHANNEL_NAME = "engine-channel"
NOTES_DIR = "engine-notes"
NOTES_MAX = 256 * 1024
BUILD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TARBALL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,159}\.tar\.gz$")
VERSION_MAX = 40
DL_TRIES = 3                                               # stream attempts per file in one apply (a dle_ ticket allows 3 uses)
DL_BACKOFF = float(os.environ.get("PXA_ENGINE_DL_BACKOFF", "5"))   # seconds, x attempt number


# -------------------------------------------------------------------------------------------- local state
def install_root(d=None):
    return lu.install_root(d)


def read_channel(d=None):
    try:
        with open(os.path.join(install_root(d), CHANNEL_NAME)) as f:
            w = (f.read(32).split() or [""])[0]
    except OSError:
        return ""
    return w if w in CHANNELS else ""


def write_channel(d, channel):
    if channel not in CHANNELS:
        raise PackageError("Unknown channel %r: use stable or beta." % str(channel)[:20], "bad_channel")
    root = install_root(d)
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, CHANNEL_NAME)
    tmp = "%s.new.%d.%d" % (p, os.getpid(), threading.get_ident())
    with open(tmp, "w") as f:
        f.write(channel + "\n")
    os.replace(tmp, p)


def load_state(d=None):
    try:
        with open(os.path.join(install_root(d), STATE_NAME)) as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(d, s):
    """Write the whole state atomically. Flat keys only: pxa-update.c reads `channel` and `top` from it with a plain string scan."""
    p = os.path.join(install_root(d), STATE_NAME)
    tmp = "%s.new.%d.%d" % (p, os.getpid(), threading.get_ident())
    with open(tmp, "w") as f:
        json.dump(s, f, sort_keys=True)
    os.replace(tmp, p)


def current_top(d=None):
    try:
        t = os.readlink(os.path.join(install_root(d), "current"))
    except OSError:
        return ""
    return t if t and "/" not in t else ""


def installed_beta(d=None):
    """The beta release `current` points at, from the state this module wrote; {} when current is not a beta we installed."""
    s = load_state(d)
    if s.get("channel") == "beta" and s.get("top") and s.get("top") == current_top(d):
        return s
    return {}


def effective_channel(d=None, channel=None):
    return channel or read_channel(d) or "stable"


# -------------------------------------------------------------------------------------------- the PXA Coins wallet (Control's install key)
# v3.1 and beta 1 have no Beta card: /beta in the PXA Discord (licd re-checks Valued, the bot DMs personal links) works on any
# client
HANDOVER = "v3.1 / beta 1 users: run /beta in Discord."
DOOR = "run /beta in the PXA Discord: the bot sends you personal download links, no coins"


def _wallet_hosted(e):
    """licd's key path answered 'this beta downloads through PXA Control's Store > Wallet': a links-hosted beta, the key may be Valued"""
    return bool(re.search(r"Store > Wallet|through PXA Control", str(getattr(e, "server_msg", "") or ""), re.I))


def _not_valued_text(e):
    if _wallet_hosted(e):
        return ("This beta is for Valued Supporters and does not install through this key: " + DOOR + ". " + HANDOVER
                + " You are on stable and nothing changed.")
    return ("The beta channel is for Valued Supporters. If you are one, " + DOOR + ". " + HANDOVER + " You are on stable and nothing changed.")


INSTALL_PREFIX = b"pxa-plugin-install|"                  # licd's machine-signature prefix (the store's, see pxa_control/store)


def control_config_dir():
    d = os.environ.get("PXA_CONTROL_CONFIG_DIR")
    if d:
        return d
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"), "pxa")


def _ed_point(pt):
    zi = pow(pt[2], pkg._P - 2, pkg._P)
    x, y = pt[0] * zi % pkg._P, pt[1] * zi % pkg._P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _ed_sign(seed, msg):
    """RFC 8032 Ed25519 over the shipped verifier's curve arithmetic (the same code as pxa_control/store)."""
    h = hashlib.sha512(seed).digest()
    a = (int.from_bytes(h[:32], "little") & ((1 << 254) - 8)) | (1 << 254)
    pub = _ed_point(pkg._mul(a, pkg._G))
    r = int.from_bytes(hashlib.sha512(h[32:] + msg).digest(), "little") % pkg._Q
    R = _ed_point(pkg._mul(r, pkg._G))
    k = int.from_bytes(hashlib.sha512(R + pub + msg).digest(), "little") % pkg._Q
    return R + ((r + k * a) % pkg._Q).to_bytes(32, "little")


def wallet_key(cfg_dir=None):
    """-> (seed, install_pub hex) when PXA Control made an install key on this computer, else None. Never creates one."""
    try:
        with open(os.path.join(cfg_dir or control_config_dir(), "plugins", "install.key"), "rb") as f:
            seed = f.read(33)
    except OSError:
        return None
    if len(seed) != 32:
        return None
    h = hashlib.sha512(seed).digest()
    a = (int.from_bytes(h[:32], "little") & ((1 << 254) - 8)) | (1 << 254)
    return seed, _ed_point(pkg._mul(a, pkg._G)).hex()


def _wallet_call(base, what, args, cfg_dir=None, op=None, path=None):
    """one signed call to licd's /v1/wallet/engine/<what> (op engine-<what>), the install key over prefix + canon({op, nonce, args})."""
    k = wallet_key(cfg_dir)
    if not k:
        raise PackageError("This computer has no PXA wallet yet: open PXA Control, Store > Wallet.", "no_wallet")
    seed, pub = k
    op = op or "engine-" + what
    try:
        n = str(lu._call(base, "/v1/store/nonce", {"install_pub": pub, "op": op}).get("nonce") or "")
        canon = json.dumps({"op": op, "nonce": n, "args": args}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        sig = base64.urlsafe_b64encode(_ed_sign(seed, INSTALL_PREFIX + canon)).rstrip(b"=").decode("ascii")
        return lu._call(base, path or "/v1/wallet/engine/" + what, dict(args, install_pub=pub, nonce=n, sig=sig))
    except PackageError as e:
        text, code = str(e), getattr(e, "code", "")
        if code == "busy" and op == "wallet-links":
            raise PackageError("This wallet made its 3 download links for today. The links made earlier keep working for 24 hours; "
                               "try again tomorrow.", "link_limit")
        if re.search(r"no wallet", text, re.I):
            raise PackageError("This computer has no PXA wallet yet: open PXA Control, Store > Wallet.", "no_wallet")
        if code == "not_valued" and re.search(r"Valued Supporters", str(getattr(e, "server_msg", "") or "")):    # the Valued-only beta
            raise PackageError("The beta channel is for Valued Supporters, and this wallet is not linked to one. If you are one, "
                               + DOOR + ". You are on stable and nothing changed.", "beta_not_redeemed")
        if code == "not_valued":                       # licd's 403 beta_not_redeemed (lu maps every beta 403 to not_valued)
            raise PackageError("Redeem beta access for this release in PXA Control (Store > Wallet) with your thank-you coins. You are on stable "
                               "and nothing changed.", "beta_not_redeemed")
        if code == "revoked":
            raise PackageError("This computer is no longer linked to that PXA wallet. Nothing changed.", "no_wallet")
        raise


def verify_beta_files(o):
    """licd's personal-links answer -> {build_id, version, files: {name: {role, size, sha256, url}}}. Names, sizes and sha256 come from the
    signed list (package key, its own prefix); only the URLs come from the envelope, and a URL can only deliver bytes that match."""
    if not isinstance(o, dict) or not isinstance(o.get("manifest"), str) or not isinstance(o.get("signature"), str):
        raise PackageError("The download links came without their signed file list, so nothing was downloaded.", "bad_manifest")
    pub = pkg.pubkey_bytes(o.get("kid"))
    if not pub or not pkg.ed25519_verify(pub, BETA_FILES_PREFIX + o["manifest"].encode(), pkg._decode_sig(o["signature"])):
        raise PackageError("The beta's file list is not signed by PXA, so nothing was downloaded.", "bad_signature")
    try:
        st = json.loads(o["manifest"])
    except ValueError:
        st = None
    if not isinstance(st, dict) or st.get("type") != "beta-files" or not BUILD_RE.match(str(st.get("build_id") or "")):
        raise PackageError("The signed file list is not a beta release, so it was refused.", "bad_statement")
    urls = {}
    for e in o.get("files") if isinstance(o.get("files"), list) else []:
        if isinstance(e, dict) and isinstance(e.get("url"), str):
            try:
                pkg._check_url(e["url"])
            except PackageError:
                continue
            urls[str(e.get("name") or "")] = e["url"]
    files = {}
    for e in st.get("files") if isinstance(st.get("files"), list) else []:
        name, sha, size, role = str(e.get("name") or ""), str(e.get("sha256") or ""), e.get("size"), str(e.get("role") or "")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,159}", name) or not re.fullmatch(r"[0-9a-f]{64}", sha) or not isinstance(size, int)
                or size <= 0 or size > pkg.MAX_PACKAGE or role not in ("tarball", "notes", "docker") or name in files):
            raise PackageError("The signed file list names an invalid file, so it was refused.", "bad_manifest")
        files[name] = {"name": name, "role": role, "size": size, "sha256": sha, "url": urls.get(name, "")}
    try:
        exp = int(o.get("expires") or 0)
    except (TypeError, ValueError):
        exp = 0
    return {"build_id": str(st["build_id"]), "version": str(st.get("version") or ""), "files": files, "expires": exp}


def _links_cache(cfg_dir=None):
    """beside the install key (owner-only), so pxa-update and PXA Control share one issue of links."""
    return os.path.join(cfg_dir or control_config_dir(), "store", "beta-links.json")


def personal_links(base, want_build=None, cfg_dir=None, want_version=None):
    """The current beta's personal links (licd op wallet-links: 3 issues a day, 24 h each). A still-valid answer is kept beside the
    install key and reused, so a resumed or repeated download does not spend another issue."""
    p = _links_cache(cfg_dir)
    o = None
    try:
        with open(p) as f:
            o = json.load(f)
        v = verify_beta_files(o)
        if v["expires"] - time.time() < LINKS_MARGIN_S or (want_build and v["build_id"] != want_build) or o.get("_base") != base.rstrip("/") \
                or (want_version and v["version"] != want_version):
            o = None
    except (OSError, ValueError, PackageError):
        o = None
    if o is None:
        o = _wallet_call(base, "links", {}, cfg_dir, op="wallet-links", path="/v1/wallet/links")
        v = verify_beta_files(o)
        os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
        tmp = "%s.new.%d" % (p, os.getpid())
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)     # the links open the files for 24 h: owner-only
        with os.fdopen(fd, "w") as f:
            json.dump(dict(o, _base=base.rstrip("/")), f)
        os.replace(tmp, p)
    v = verify_beta_files(o)
    if want_build and v["build_id"] != want_build:
        raise PackageError("The beta on the download server (%s) is not the one this wallet installs (%s). Check again in a minute."
                           % (v["build_id"][:64], want_build[:64]), "bad_manifest")
    if want_version and v["version"] != want_version:
        raise PackageError("The beta on the download server (%s) is not the current one (%s). Check again in a minute."
                           % (v["version"][:64], want_version[:64]), "bad_manifest")
    return v


def link_urls(d, base, rel):
    """the engine pair's URLs for a links-hosted beta, after checking that the signed file list names the very bytes the signed engine
    statement names."""
    v = personal_links(base, rel["build_id"])
    out = {}
    for role in ("tarball", "notes"):
        f = v["files"].get(rel[role]["name"])
        if not f or f["role"] != role or f["sha256"] != rel[role]["sha256"] or f["size"] != rel[role]["size"] or not f["url"]:
            raise PackageError("The beta's file list does not match its signed release, so nothing was downloaded.", "bad_manifest")
        out[f["name"]] = f["url"]
    return out


def fetch_file(role, dest_dir, base=None, progress=None, cancelled=None, cfg_dir=None, want_version=None):
    """Download one more file of the current beta (role 'docker': the Docker image archive, download only) through a personal link,
    resuming a .part, and keep it only if its size and sha256 match the signed list. want_version (the release the wallet shows) drops
    a still-valid links cache of an older beta. -> (path, entry)."""
    base = base or pkg.licence_url(None)
    v = personal_links(base, cfg_dir=cfg_dir, want_version=want_version)
    f = [x for x in v["files"].values() if x["role"] == role]
    if not f:
        raise PackageError("This beta has no %s file." % role, "no_file")
    f = f[0]
    if not f["url"]:
        raise PackageError("The download server gave no link for %s." % f["name"], "bad_manifest")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f["name"])
    if os.path.isfile(dest) and os.path.getsize(dest) == f["size"] and _sha256(dest) == f["sha256"]:
        return dest, f
    pkg.download(f["url"], dest, size=f["size"], progress=progress, cancelled=cancelled)
    if os.path.getsize(dest) != f["size"] or _sha256(dest) != f["sha256"]:
        os.remove(dest)
        raise PackageError("The downloaded %s does not match the signed checksum, so it was deleted." % f["name"], "bad_download")
    return dest, f


# -------------------------------------------------------------------------------------------- the server
def _call(base, path, body):
    try:
        return lu._call(base, path, body)
    except PackageError as e:
        if getattr(e, "code", "") == "not_valued" or (getattr(e, "code", "") == "revoked" and re.search(r"beta", str(e), re.I)):
            err = PackageError(_not_valued_text(e), "not_valued")
            err.server_msg = getattr(e, "server_msg", "")
            raise err
        if getattr(e, "code", "") == "no_release":
            raise PackageError("There is no engine release on that channel for this machine yet. Nothing changed.", "no_release")
        raise


def verify_release(o, base, channel, platform, cuda_major):
    """The server's answer -> the release to install, or PackageError. Everything returned comes from the signed statement."""
    if not isinstance(o, dict) or not isinstance(o.get("manifest"), str) or not o["manifest"] or not isinstance(o.get("signature"), str):
        raise PackageError("The licence server's engine release description is incomplete, so nothing was changed.", "bad_manifest")
    pub = pkg.pubkey_bytes(o.get("kid"))
    if not pub:
        raise PackageError("This build of PXA does not know the key (%s) this engine release is signed with, so it will not install it."
                           % str(o.get("kid") or "none")[:32], "no_pubkey")
    if not pkg.ed25519_verify(pub, SIGN_PREFIX + o["manifest"].encode(), pkg._decode_sig(o["signature"])):
        raise PackageError("The engine release signature is not valid, so nothing was installed. If this keeps happening, tell us in the "
                           "PXA Network Discord #support.", "bad_signature")
    try:
        st = json.loads(o["manifest"])
    except ValueError:
        st = None
    if not isinstance(st, dict) or st.get("edition") != "engine":
        raise PackageError("The signed description is not an engine release, so it was refused.", "bad_statement")
    if str(st.get("channel") or "") != channel or str(st.get("platform") or "") != platform or st.get("cuda_major") != cuda_major:
        raise PackageError("The signed engine release does not match what was asked for (channel, platform or CUDA version), so it was refused.",
                           "bad_statement")
    bid, version = str(st.get("build_id") or ""), str(st.get("version") or "")
    if not BUILD_RE.match(bid) or not version or len(version) > VERSION_MAX:
        raise PackageError("The signed engine release has no valid id or version, so it was refused.", "bad_manifest")
    files = st.get("files") if isinstance(st.get("files"), list) else []
    by_role = {}
    for e in files:
        if not isinstance(e, dict):
            continue
        name, sha, size, role = str(e.get("name") or ""), str(e.get("sha256") or ""), e.get("size"), str(e.get("role") or "")
        if role not in ("tarball", "notes") or role in by_role:
            raise PackageError("The signed engine release lists an unexpected file, so it was refused.", "bad_manifest")
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or not isinstance(size, int) or size <= 0:
            raise PackageError("The signed engine release has no valid checksum or size for %s, so it was refused." % role, "bad_manifest")
        if role == "tarball" and (not TARBALL_RE.match(name) or size > pkg.MAX_PACKAGE):
            raise PackageError("The signed engine release names an invalid tarball, so it was refused.", "bad_manifest")
        if role == "notes" and (name != "NOTES.md" or size > NOTES_MAX):
            raise PackageError("The signed engine release has invalid notes, so it was refused.", "bad_manifest")
        by_role[role] = {"name": name, "sha256": sha, "size": int(size), "role": role}
    if sorted(by_role) != ["notes", "tarball"]:
        raise PackageError("The signed engine release does not list a tarball and its notes, so it was refused.", "bad_manifest")
    return {"build_id": bid, "version": version, "channel": channel, "platform": platform, "cuda_major": cuda_major,
            "sm": [int(x) for x in st.get("sm") or [] if isinstance(x, int)], "tarball": by_role["tarball"], "notes": by_role["notes"],
            "kid": str(o.get("kid") or ""), "manifest": o["manifest"], "signature": o["signature"], "base": base.rstrip("/"),
            "notice": str(o.get("notice") or "")[:400]}


def tickets(o, rel):
    """Download URLs from the unsigned envelope, accepted only on this server's own engine download path and only for signed names."""
    out = {}
    for e in (o.get("files") if isinstance(o.get("files"), list) else []):
        if not isinstance(e, dict):
            continue
        name, url = str(e.get("name") or ""), str(e.get("url") or "")
        if name in (rel["tarball"]["name"], rel["notes"]["name"]) and url.startswith(rel["base"] + "/v1/engine/download/"):
            out[name] = url
    for role in ("tarball", "notes"):
        if rel[role]["name"] not in out:
            raise PackageError("The licence server did not give a download link for the %s, so nothing was changed." % role, "bad_manifest")
    return out


def latest(d, base, key, channel, platform=None, cuda_major=None):
    if channel not in CHANNELS:
        raise PackageError("Unknown channel %r: use stable or beta." % str(channel)[:20], "bad_channel")
    by_wallet = channel == "beta" and wallet_key() is not None
    if not pkg.valid_key(key) and not by_wallet:
        raise PackageError("No licence key is set and this computer has no PXA wallet. The beta channel is for Valued Supporters: "
                           "if you are one, " + DOOR + ".", "no_key")
    if platform is None or cuda_major is None:
        p, c = pkg.detect_platform()
        platform = platform or p
        cuda_major = c if cuda_major is None else cuda_major
    cur = installed_beta(d).get("version") or lu.engine_version(d)
    if pkg.valid_key(key):
        body = {"key": key, "channel": channel, "platform": platform, "cuda_major": int(cuda_major), "current": cur}
        try:
            o = _call(base, "/v1/engine/latest", body)
            return verify_release(o, base, channel, platform, int(cuda_major)), o
        except PackageError as e:
            if by_wallet and getattr(e, "code", "") == "not_valued":
                _link_valued_key(base, key)       # a links-hosted beta: a Valued key reaches it through the wallet, without coins
            if getattr(e, "code", "") == "not_valued" and not by_wallet:
                if _wallet_hosted(e):      # the key path steps aside for a links-hosted beta: the wallet is the way in
                    raise PackageError("This beta is for Valued Supporters and does not install through this key: " + DOOR
                                       + ". You are on stable and nothing changed.", "not_valued")
                raise PackageError("The beta channel is for Valued Supporters, and the licence server does not see that role on "
                                   "this key. If you are one, " + DOOR + ". You are on stable and nothing changed.", "not_valued")
            if not (by_wallet and getattr(e, "code", "") == "not_valued"):
                raise
    o = _wallet_call(base, "latest", {"channel": "beta", "platform": str(platform), "cuda_major": str(int(cuda_major)), "current": str(cur or "")})
    return verify_release(o, base, channel, platform, int(cuda_major)), o


_KEY_LINK_SEEN = {}     # tag -> (linked?, when): this process remembers a link even when the marker cannot be written, and a refusal for 10 minutes


def _link_valued_key(base, key, cfg_dir=None):
    """A Valued Supporter's licence key, proven once to this computer's wallet (licd wallet action key-link, install-key signed): the
    wallet then gets every beta, links-hosted ones included, without coins. Only a key licd calls Valued is linked; a marker beside
    the install key (a hash, never the key) skips the call after the first success. Best effort: a refusal changes nothing."""
    k = wallet_key(cfg_dir)
    if not (pkg.valid_key(key) and k):
        return False
    mark = os.path.join(cfg_dir or control_config_dir(), "plugins", "wallet-key-linked")
    tag = hashlib.sha256(("%s|%s|%s" % (base.rstrip("/"), k[1], key)).encode()).hexdigest()
    seen = _KEY_LINK_SEEN.get(tag)
    if seen and (seen[0] or time.time() - seen[1] < 600):
        return seen[0]
    try:
        with open(mark) as f:
            if f.read().strip() == tag:
                return True
    except OSError:
        pass
    try:
        if not _call(base, "/v1/engine/channels", {"key": key}).get("beta"):
            _KEY_LINK_SEEN[tag] = (False, time.time())
            return False
        if not _wallet_call(base, "key-link", {"key": key}, cfg_dir, op="wallet-key-link", path="/v1/wallet/key-link").get("valued_supporter"):
            _KEY_LINK_SEEN[tag] = (False, time.time())
            return False
    except PackageError:
        _KEY_LINK_SEEN[tag] = (False, time.time())     # key_used / key_other / refused / offline: not again for 10 minutes
        return False
    _KEY_LINK_SEEN[tag] = (True, time.time())
    try:
        fd = os.open(mark, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(tag + "\n")
    except OSError:
        pass
    return True


def channels(base, key):
    out = {"channels": ["stable"], "beta": False}
    if pkg.valid_key(key):
        o = _call(base, "/v1/engine/channels", {"key": key})
        ch = [c for c in (o.get("channels") or []) if c in CHANNELS] or ["stable"]
        out = {"channels": ch, "beta": bool(o.get("beta")) and "beta" in ch}
        if out["beta"] and wallet_key() is not None:
            _link_valued_key(base, key)           # once: the wallet then shows "Yours as a Valued Supporter" and gets the personal links
    if not out["beta"] and wallet_key() is not None:
        try:
            w = _wallet_call(base, "channels", {})
        except PackageError:
            w = {}
        if w.get("beta") is True:
            out = {"channels": ["stable", "beta"], "beta": True, "via": "wallet"}
    return out


# -------------------------------------------------------------------------------------------- verbs
def check(d=None, base=None, key=None, channel=None):
    """-> (release, answer, installed beta version or the engine version). Changes nothing on disk."""
    d = install_root(d)
    base = base or pkg.licence_url(None)
    rel, raw = latest(d, base, lu.licence_key(d, key), effective_channel(d, channel))
    have = installed_beta(d)
    return rel, raw, (have.get("version") if have else lu.engine_version(d)), bool(have) and have.get("build_id") == rel["build_id"]


def _sha256(path):
    return lu._sha256(path)


def apply(d=None, base=None, key=None, channel=None, installer=None, progress=None, force=False, from_file=None):
    """Install the newest beta. -> (applied, release, version before). Nothing changes on disk until every byte matches the signature.
    from_file: a tarball already on disk (the first beta on an install whose updater predates the channel): it is checked against the
    SIGNED size and sha256 exactly like a download, and never moved or deleted."""
    d = install_root(d)
    base = base or pkg.licence_url(None)
    channel = effective_channel(d, channel)
    if channel != "beta":
        raise PackageError("This tool installs the beta channel; stable comes from `pxa-update apply`.", "bad_channel")
    key = lu.licence_key(d, key)
    rel, raw = latest(d, base, key, channel)
    have = installed_beta(d)
    before = have.get("version") if have else lu.engine_version(d)
    if have and have.get("build_id") == rel["build_id"] and not force:
        write_channel(d, "beta")
        return False, rel, before
    if lu.server_running(d):
        raise PackageError("A server is running from this install. Stop it and try again.", "server_running")
    urls = link_urls(d, base, rel) if raw.get("links") is True else tickets(raw, rel)
    installer = installer or os.environ.get("PXA_UPDATE_BIN") or shutil.which("pxa-update")
    if not installer or not os.access(installer, os.X_OK):
        raise PackageError("Cannot find the pxa-update program of this install.", "no_installer")
    os.makedirs(d, exist_ok=True)
    # One partial folder per release, on the install filesystem (/tmp can be small or noexec). A 1.5 GB beta comes through the
    # licence server's tunnel: an interrupted download keeps its .part here, and the next apply resumes it with Range (a fresh
    # ticket). Partial folders of OTHER releases are dropped; this one is removed on success and on any non-network failure.
    dl = os.path.join(d, ".pxa-update-dl-" + rel["build_id"])
    for n in os.listdir(d):
        if n.startswith(".pxa-update-dl-") and n != os.path.basename(dl):
            shutil.rmtree(os.path.join(d, n), ignore_errors=True)
    os.makedirs(dl, exist_ok=True)
    keep = False
    try:
        got = {}
        for role in ("notes", "tarball"):
            f = rel[role]
            dest = os.path.join(dl, f["name"])
            local = bool(role == "tarball" and from_file)
            if local:
                dest = os.path.abspath(from_file)
                if not os.path.isfile(dest):
                    raise PackageError("The tarball %s is not there, so nothing was installed." % from_file, "bad_download")
            for attempt in range(0 if local else DL_TRIES):          # a ticket allows 3 uses: retry an interrupted stream on the same ticket
                try:
                    pkg.download(urls[f["name"]], dest, size=f["size"], progress=progress if role == "tarball" else None)
                    break
                except PackageError as e:
                    if getattr(e, "code", "") != "offline":
                        raise
                    if attempt == DL_TRIES - 1:
                        keep = True                  # the next apply resumes from the .part
                        raise
                    time.sleep(DL_BACKOFF * (attempt + 1))
            if os.path.getsize(dest) != f["size"] or _sha256(dest) != f["sha256"]:
                raise PackageError("The downloaded %s does not match the signed checksum, so nothing was installed." % role, "bad_download")
            got[role] = dest
        r = subprocess.run([installer, "apply-file", got["tarball"], "--sha256", rel["tarball"]["sha256"], "--dir", d],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise PackageError((r.stderr or r.stdout or "the installer failed").strip().splitlines()[-1][:300], "install_failed")
        top = ""
        for line in r.stdout.splitlines():
            if line.startswith("installed "):
                top = line[len("installed "):].strip()
        if not top or top != current_top(d):
            raise PackageError("The installer did not switch to the new version.", "install_failed")
        nd = os.path.join(d, NOTES_DIR)
        os.makedirs(nd, exist_ok=True)
        shutil.copyfile(got["notes"], os.path.join(nd, rel["build_id"] + ".md"))
    finally:
        if not keep:
            shutil.rmtree(dl, ignore_errors=True)
    save_state(d, {"channel": "beta", "version": rel["version"], "build_id": rel["build_id"], "top": top,
                   "tarball_sha256": rel["tarball"]["sha256"], "notes_sha256": rel["notes"]["sha256"], "kid": rel["kid"],
                   "manifest": rel["manifest"], "signature": rel["signature"], "previous_version": before or "",
                   "installed": int(time.time())})
    write_channel(d, "beta")
    return True, rel, before


def set_channel(d=None, base=None, key=None, channel=None):
    """Record the channel choice. Beta only for a key the server calls Valued; stable always (that is how a tester leaves)."""
    d = install_root(d)
    if channel not in CHANNELS:
        raise PackageError("Say which channel: stable or beta.", "bad_channel")
    if channel == "beta":
        c = channels(base or pkg.licence_url(None), lu.licence_key(d, key))
        if not c["beta"]:
            raise PackageError("The beta channel is for Valued Supporters. If you are one, " + DOOR + ". " + HANDOVER
                               + " You are on stable and nothing changed.", "not_valued")
    write_channel(d, channel)
    return channel


def notes(d=None, base=None, key=None):
    """-> (text, version, build_id) of the installed beta's notes, re-verified against the signed statement kept in the state; with no beta
    installed, the offered beta's notes (downloaded, verified, not kept)."""
    d = install_root(d)
    s = installed_beta(d)
    if s:
        pub = pkg.pubkey_bytes(s.get("kid"))
        if not pub or not pkg.ed25519_verify(pub, SIGN_PREFIX + str(s.get("manifest") or "").encode(), pkg._decode_sig(str(s.get("signature") or ""))):
            raise PackageError("The saved release statement no longer verifies, so the notes are not shown.", "bad_signature")
        st = json.loads(s["manifest"])
        want = [f for f in st.get("files") or [] if isinstance(f, dict) and f.get("role") == "notes"]
        p = os.path.join(d, NOTES_DIR, str(s.get("build_id")) + ".md")
        if not want or not os.path.isfile(p) or os.path.getsize(p) > NOTES_MAX or _sha256(p) != want[0].get("sha256"):
            raise PackageError("The saved release notes do not match the signed release, so they are not shown.", "bad_notes")
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read(), str(st.get("version") or ""), str(st.get("build_id") or "")
    base = base or pkg.licence_url(None)
    rel, raw = latest(d, base, lu.licence_key(d, key), "beta")
    url = (link_urls(d, base, rel) if raw.get("links") is True else tickets(raw, rel))[rel["notes"]["name"]]
    with tempfile.TemporaryDirectory(prefix=".pxa-notes-", dir=d if os.path.isdir(d) else None) as tmp:
        p = os.path.join(tmp, "NOTES.md")
        pkg.download(url, p, size=rel["notes"]["size"])
        if _sha256(p) != rel["notes"]["sha256"]:
            raise PackageError("The release notes do not match the signed release, so they are not shown.", "bad_notes")
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read(), rel["version"], rel["build_id"]


# -------------------------------------------------------------------------------------------- CLI
USAGE = ("usage: pxa-update check|apply --channel beta | channel stable|beta | channels | notes [--dir DIR] [--licence-url URL] "
         "[--key KEY] [--json]   (apply also takes --from-file TGZ: a beta tarball already downloaded, checked against the signature)")


def main(argv):
    args = list(argv)
    cmd = args.pop(0) if args and not args[0].startswith("-") else None
    arg = args.pop(0) if cmd == "channel" and args and not args[0].startswith("-") else None
    opts = {"--channel": None, "--dir": None, "--base-url": None, "--key": None, "--installer": None, "--from-file": None}
    as_json = False
    while args:
        a = args.pop(0)
        if a in opts and args:
            opts[a] = args.pop(0)
        elif a == "--json":
            as_json = True
        else:
            print("pxa-update: " + USAGE, file=sys.stderr)
            return 1
    if cmd not in ("check", "apply", "channel", "channels", "notes"):
        print("pxa-update: " + USAGE, file=sys.stderr)
        return 1
    d, base, key = install_root(opts["--dir"]), opts["--base-url"], opts["--key"]
    try:
        if cmd == "check":
            rel, raw, cur, same = check(d, base, key, opts["--channel"])
            print("current=%s latest=%s update=%s channel=%s" % (cur or "0", rel["version"], "no" if same else "yes", rel["channel"]))
            if as_json:
                print(json.dumps({"current": cur, "latest": rel["version"], "update": not same, "channel": rel["channel"],
                                  "build_id": rel["build_id"], "notice": rel["notice"]}))
            return 0
        if cmd == "apply":
            done, rel, before = apply(d, base, key, opts["--channel"], installer=opts["--installer"], from_file=opts["--from-file"])
            if not done:
                print("current=%s latest=%s update=no channel=beta" % (before or "0", rel["version"]))
                return 0
            print("installed %s channel=beta version=%s" % (current_top(d), rel["version"]))
            print("next %s/current" % d)
            return 0
        if cmd == "channel":
            ch = set_channel(d, base, key, arg)
            print("channel=%s" % ch)
            print("next pxa-update apply" + ("" if ch == "beta" else "   (installs the latest stable release, older than a beta if need be)"))
            return 0
        if cmd == "channels":
            c = channels(base or pkg.licence_url(None), lu.licence_key(d, key))
            print("channels=%s beta=%s current=%s" % (",".join(c["channels"]), "yes" if c["beta"] else "no", effective_channel(d)))
            if as_json:
                print(json.dumps(dict(c, current=effective_channel(d))))
            return 0
        text, ver, bid = notes(d, base, key)
        if as_json:
            print(json.dumps({"version": ver, "build_id": bid, "notes": text}))
        else:
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return 0
    except PackageError as e:
        print("pxa-update: %s" % pkg._scrub(str(e), key), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
