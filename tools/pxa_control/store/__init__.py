"""PXA Store, the LOCAL side (stubs; design: PLUGINS-STORE-DESIGN.md, built on plugins/p-platform DESIGN.md §6-7).
No payment code, no credit issuance: credits, codes and grants are issued by the owner-run licensing daemon (licd).
This module only

  * verifies an entitlement GRANT licd signed (Ed25519 over a domain prefix + canonical JSON, the pure-Python verifier
    tools/pxa_encode_pkg.py already ships; prefix shared with p-platform so licd has ONE entitlement signer),
  * keeps grants in an entitlement cache (config dir, 0600) with offline grace, refresh, revocation, anti-replay and a
    clock-rollback guard,
  * answers GET /api/store/catalog (fixture items until licd serves the catalog) and GET /api/store/entitlements.

A grant, as licd sends it and the cache stores it (one grant per item):

    {"statement": {"v": 1, "iss": "licd", "kid": "ent-2026-10",
                   "ent_id": "<128-bit hex>",                 grant id: anti-replay, revocation, the per-grant watermark key
                   "kind": "plugin|quant-tier|preset|feature", "item_id": "pxqn", "level": "pxqn4" | null,
                   "sub": "<account hash>",                   never the Discord id or e-mail
                   "install_pub": "<64 hex>",                 this machine's install key (plugins/install.key, 0600)
                   "issued": 1760400000, "refresh_after": 1760486400, "not_after": 1775952000},
     "signature": "<base64url Ed25519 over b'pxa-plugin-ent|' + canonical(statement)>"}

refresh_after is short (licd: 24 h): an online Control renews daily, so a revoked or downgraded account loses its grant
within a day. not_after is the offline grace (owner, 2026-10-11: 180 days). Nothing in the open engine checks any of
this (owner, 2026-10-14): a quant-tier grant unlocks DISTRIBUTION (signed downloads bound to the grant, watermarked per
grant), presets/features unlock Control-side settings.
"""
import base64
import contextlib
import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time

ENT_PREFIX = b"pxa-plugin-ent|"                 # = p-platform common.ENT_PREFIX: one entitlement signer in licd
REVOKE_PREFIX = b"pxa-store-revocations|"
GRANT_PUBKEYS = {}          # kid -> 64 hex: licd entitlement public key, filled from store/pubkeys.json at release
DEFAULT_STORE_URL = "https://lic.pxanetwork.com"

def _load_pubkeys():
    path = os.path.join(os.path.dirname(__file__), "pubkeys.json")
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return {}
    keys = data.get("GRANT_PUBKEYS") or {}
    return {str(k): str(v) for k, v in keys.items() if isinstance(v, str)}

GRANT_PUBKEYS.update(_load_pubkeys())
ENT_KINDS = ("plugin", "quant-tier", "preset", "feature")
CLOCK_SKEW_S = 300
HARD_LOCK_CODES = ("revoked", "not_linked", "account_disabled")   # spec s4: the only licd answers (403) that lock everything
LOCAL_LOCK_SLACK_S = 300   # no licd time on a 403: the lock time is the local clock minus 5 minutes (the seen-id rule still holds)
BODY_DATE_WINDOW_S = 300    # F310: body server_time and Date further apart than this -> the EARLIER one is licd's time
LOCK_AHEAD_CAP_S = 24 * 3600  # F310: a lock time is never later than this computer's clock + 24 h
# hard-lock v4 (PXANetwork 05:18, licd store.py: a refresh re-signs the SAME ent_id with a newer `issued`; a new ent_id only
# after unlink + relink). Lock-journal keys, shared by both journals (<cfg>/store/grants.journal.json and
# <cfg>/plugins/store-lock-journal.json) and the same names in Plugins (plugins/item4-rc36) and Core (RC3.6 item 4):
# hard_lock_at latest wins | hard_locks + star union / OR (star cleared only by a v4-fresh grant) | seen max (except the
# F310 (b) verified reset) | licd_high_water max (capped now + 24 h) | newest, hard_lock_newest per-key max | revoked
# union | gen max, then + 1 on save. Keys starting with "_" are process-only: never written, never read (_file_keys).
J_NEWEST = "newest"                 # {ent_id: newest issued ever accepted}: anti-replay, also in the journal
J_LOCK_NEWEST = "hard_lock_newest"  # {ent_id: newest issued at lock time}: a known id unlocks only past this
J_LICD_SEEN = "licd_high_water"     # = Plugins LICD_HIGH_WATER: latest licd time seen (never local), capped now + 24 h
J_REVOKED = "revoked"               # = Plugins / Core J_REVOKED: every revoked ent_id ever seen (grow-only, F318)
HARD_LOCK_SOURCES = ("body", "date", "local", "licd_high_water", "licd_reset")   # hard_lock_source (= Plugins / Core):
#   body / date: the 403 reply's server_time / Date (server_time_checked); local: this clock; licd_high_water: a rebuild on
#   the newest licd time on record; licd_reset: reserved (the lock time is never moved back today).


def _lock_source(v):
    """a hard_lock_source as read or given -> an enum value: legacy "server" (Control before RC3.6 item 4) -> body; any
    other unknown value -> local, the strictest reading."""
    if v in HARD_LOCK_SOURCES:
        return v
    return "body" if v == "server" else "local"


ROLLBACK_S = 48 * 3600      # p-platform §6.3: a clock more than 48 h behind the latest time seen counts as expired
MAX_LIFE_S = 400 * 86400
_ITEM_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ITEM_LOCKS = {}                                   # item -> Lock: one Download & install per item (eng review F8)
_ITEM_LOCKS_GUARD = threading.Lock()
_HEX_RE = re.compile(r"^[0-9a-f]{16,128}$")


class GrantError(ValueError):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code
        self.status = 403


def canonical(obj):
    """the exact bytes licd signs (after the prefix): sorted keys, no spaces, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def b64u(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(text):
    t = str(text or "").strip()
    try:
        return base64.urlsafe_b64decode(t + "=" * (-len(t) % 4))
    except (ValueError, TypeError):
        return b""


# ---- Ed25519: verify with the shipped verifier; keygen/sign for this machine's install key (RFC 8032) -----------
def _pk():
    import pxa_encode_pkg as PK
    return PK


def _encode_point(pt):
    PK = _pk()
    zi = pow(pt[2], PK._P - 2, PK._P)
    x, y = pt[0] * zi % PK._P, pt[1] * zi % PK._P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _expand(seed):
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def ed25519_public(seed):
    a, _ = _expand(seed)
    return _encode_point(_pk()._mul(a, _pk()._G))


def ed25519_sign(seed, msg):
    PK = _pk()
    a, prefix = _expand(seed)
    pub = _encode_point(PK._mul(a, PK._G))
    r = int.from_bytes(hashlib.sha512(prefix + msg).digest(), "little") % PK._Q
    R = _encode_point(PK._mul(r, PK._G))
    k = int.from_bytes(hashlib.sha512(R + pub + msg).digest(), "little") % PK._Q
    return R + ((r + k * a) % PK._Q).to_bytes(32, "little")


def ed25519_verify(pub, msg, sig):
    return _pk().ed25519_verify(pub, msg, sig)


def _read_seed(p):
    """the 32-byte install key, never through a symlink (lstat + O_NOFOLLOW, = _read_regular): a planted link reads as
    no key, and the create-once path then replaces the link itself, never what it points at."""
    try:
        if os.path.islink(p):
            return None
        fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as f:
            seed = f.read(33)
    except OSError:
        return None
    return seed if len(seed) == 32 else None


def _publish_seed(p):
    """a new 32-byte seed published whole and at most once: written to a temp file (fsync, 0600) and hard-linked to
    install.key, which fails if another process got there first (then its key wins and is returned). Only a key file of
    the wrong length is replaced. Falls back to O_CREAT|O_EXCL where hard links are not available."""
    d = os.path.dirname(p)
    seed = secrets.token_bytes(32)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".install.key.tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(seed)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        try:
            os.link(tmp, p)
            return seed
        except FileExistsError:
            have = _read_seed(p)
            if have is not None:
                return have                       # another process made it first: everyone ends up with that key
            os.replace(tmp, p)                    # a key file of the wrong length (under the install.key lock)
            return seed
        except OSError:                           # no hard links here: exclusive create
            try:
                xfd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            except FileExistsError:
                have = _read_seed(p)
                if have is not None:
                    return have
                os.replace(tmp, p)
                return seed
            with os.fdopen(xfd, "wb") as f:
                f.write(seed)
                f.flush()
                os.fsync(f.fileno())
            return seed
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def install_key(cfg_dir):
    """-> (seed, install_pub hex). <config_dir>/plugins/install.key (0600), p-platform's path: made once per machine;
    licd binds grants to its public half and the refresh challenge is signed with it, so a copied grant file alone
    does not work on another machine. The first-time create runs under <plugins>/install.key.lock and publishes by
    hard link (no overwrite), so two processes starting at once end up with the same key."""
    d = os.path.join(cfg_dir, "plugins")
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, "install.key")
    seed = _read_seed(p)
    if seed is None:
        with file_lock(p):
            seed = _read_seed(p) or _publish_seed(p)
    return seed, ed25519_public(seed).hex()


def refresh_challenge(cfg_dir, nonce):
    """the body Control sends licd's /v1/plugins/refresh: licd's nonce signed with the install key (anti-replay of
    the refresh itself: a recorded refresh cannot be played back with another nonce)."""
    seed, pub = install_key(cfg_dir)
    msg = b"pxa-plugin-install|" + str(nonce).encode()
    return {"install_pub": pub, "nonce": str(nonce), "sig": b64u(ed25519_sign(seed, msg))}


# ---- grants ---------------------------------------------------------------------------------------------------
def verify_grant(env, pubkeys=None, now=None, install_pub=None, issued_ref=None):
    """-> (statement, status) with status ok | refresh_due, else GrantError .code in bad_shape, unknown_key,
    bad_signature, bad_issuer, not_yet, expired, wrong_machine, bad_entitlement.
    issued_ref: licd's clock for the not-yet check (server_time in the reply that carried the grant, else its Date
    header; docs/ANIBAL-HARD-LOCK-SPEC.md s5): a grant licd issued after that + CLOCK_SKEW_S is `not_yet`. None -> the
    local clock `now` (no server time). Expiry and refresh_due stay on the local clock."""
    now = time.time() if now is None else now
    licd_now = now if issued_ref is None else float(issued_ref)
    keys = GRANT_PUBKEYS if pubkeys is None else pubkeys
    if not isinstance(env, dict) or not isinstance(env.get("statement"), dict):
        raise GrantError("not a grant", "bad_shape")
    st = env["statement"]
    kid = str(st.get("kid") or "")
    hexkey = keys.get(kid) or ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", hexkey):
        raise GrantError(f"unknown grant key {kid[:32]!r}: update PXA Control", "unknown_key")
    sig = _b64d(env.get("signature"))
    if len(sig) != 64 or not ed25519_verify(bytes.fromhex(hexkey), ENT_PREFIX + canonical(st), sig):
        raise GrantError("the grant's signature does not verify", "bad_signature")
    if st.get("v") != 1 or isinstance(st.get("v"), bool) or st.get("iss") != "licd":   # True == 1 in Python (the junk-value table)
        raise GrantError("not a licd grant", "bad_issuer")
    if st.get("kind") not in ENT_KINDS or not _ITEM_RE.match(str(st.get("item_id") or "")):
        raise GrantError("the grant names an unknown item", "bad_entitlement")
    lv = st.get("level")
    if lv is not None and not (isinstance(lv, str) and len(lv) <= 64):
        # "pxqn4" | null (licd); the view hands level to the page as is, so a list / number / NaN / 64 KB text never counts
        raise GrantError("the grant names an unknown level", "bad_entitlement")
    if not (_HEX_RE.match(str(st.get("ent_id") or "")) and _HEX_RE.match(str(st.get("sub") or ""))
            and re.fullmatch(r"[0-9a-f]{64}", str(st.get("install_pub") or ""))):
        raise GrantError("grant id, account or machine missing", "bad_shape")
    issued, ref, until = st.get("issued"), st.get("refresh_after"), st.get("not_after")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (issued, ref, until)):
        # ints only (= Core / Plugins): a float, a numeric string or a bool would be stored as is and make the shared
        # grants.json the wrong shape for the other caches
        raise GrantError("grant times missing", "bad_shape")
    if not issued <= ref <= until or until - issued > MAX_LIFE_S:
        raise GrantError("grant times out of order", "bad_shape")
    if issued > licd_now + CLOCK_SKEW_S:
        raise GrantError("grant not valid yet (check the clock)", "not_yet")
    if now > until + CLOCK_SKEW_S:
        raise GrantError("grant expired: reconnect once and PXA Control renews it", "expired")
    if install_pub is not None and st["install_pub"] != install_pub:
        raise GrantError("this grant belongs to another computer", "wrong_machine")
    return st, ("refresh_due" if now >= ref else "ok")


def sign_revocations(seed, kid, ent_ids, issued):
    """what licd does (here for tests and the fake): a signed list of revoked grant ids."""
    st = {"v": 1, "iss": "licd", "kid": kid, "type": "revocations", "issued": int(issued), "ent_ids": sorted(ent_ids)}
    return {"statement": st, "signature": b64u(ed25519_sign(seed, REVOKE_PREFIX + canonical(st)))}


def verify_revocations(env, pubkeys=None):
    keys = GRANT_PUBKEYS if pubkeys is None else pubkeys
    st = env.get("statement") if isinstance(env, dict) else None
    if not isinstance(st, dict) or st.get("type") != "revocations" or not isinstance(st.get("ent_ids"), list):
        raise GrantError("not a revocation list", "bad_shape")
    hexkey = keys.get(str(st.get("kid") or "")) or ""
    sig = _b64d(env.get("signature"))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", hexkey) or len(sig) != 64 or \
            not ed25519_verify(bytes.fromhex(hexkey), REVOKE_PREFIX + canonical(st), sig):
        raise GrantError("the revocation list's signature does not verify", "bad_signature")
    return st


# ---- the shared store files: one cross-process lock, atomic replace (F303-P / F305-P, RC3.6) --------------------------
LOCK_SUFFIX = ".lock"         # <store>/grants.json.lock: the lock Control, Core's gate and the plugin host all take
LOCK_TIMEOUT_S = 30
LOCK_RETRY_S = 5.0            # F311: a write that only adds locks, busy past LOCK_TIMEOUT_S, journals first and waits this long more
_LOCK_TLS = threading.local()


class StoreBusy(OSError):
    """another process held the store lock for LOCK_TIMEOUT_S: nothing was read or written."""


def _lock_fd(fd, timeout):
    end = time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > end:
                    raise StoreBusy("the store file is busy (another PXA process holds its lock)")
                time.sleep(0.01)
    import fcntl
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (BlockingIOError, PermissionError):
            if time.monotonic() > end:
                raise StoreBusy("the store file is busy (another PXA process holds its lock)")
            time.sleep(0.01)


def _unlock_fd(fd):
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(path, timeout=None):
    """exclusive lock on <path>.lock across processes and threads (POSIX flock / Windows msvcrt.locking on byte 0 of
    the lock file; the lock file itself is never deleted). Re-entrant within one thread, so a mutator may call another.
    Every read-modify-save of a store file runs inside it: load, change, write to a temp file, fsync, os.replace."""
    lp = os.path.abspath(path) + LOCK_SUFFIX
    held = getattr(_LOCK_TLS, "held", None)
    if held is None or getattr(_LOCK_TLS, "pid", None) != os.getpid():   # a forked child holds nothing it inherited
        held = _LOCK_TLS.held = {}
        _LOCK_TLS.pid = os.getpid()
    if held.get(lp):
        held[lp] += 1
        try:
            yield
        finally:
            held[lp] -= 1
        return
    os.makedirs(os.path.dirname(lp), mode=0o700, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0), 0o600)
    try:
        _lock_fd(fd, LOCK_TIMEOUT_S if timeout is None else timeout)
        held[lp] = 1
        try:
            yield
        finally:
            held.pop(lp, None)
            _unlock_fd(fd)
    finally:
        os.close(fd)


REPLACE_RETRY_S = 5.0     # F312 (= Core): a sharing violation (AV scanner, indexer, backup) on the replace is retried this long


_REPLACE_TLS = threading.local()


@contextlib.contextmanager
def _replace_budget():
    """one REPLACE_RETRY_S budget for every replace inside (review Minor 2: the contract's 5 s is per save, not per
    file): it starts at the first replace and nested scopes share it, so a store save of 2 journals + grants.json + 2
    journals (and a busy hard_lock's journal-only write plus its save) retries for 5 s in all."""
    if getattr(_REPLACE_TLS, "active", False):
        yield
        return
    _REPLACE_TLS.active, _REPLACE_TLS.until = True, None
    try:
        yield
    finally:
        _REPLACE_TLS.active, _REPLACE_TLS.until = False, None


def _replace_retry(src, dst, budget=None):
    """os.replace, retried on a transient PermissionError (Windows sharing violation / ERROR_ACCESS_DENIED while another
    process has dst open without FILE_SHARE_DELETE) with backoff for up to REPLACE_RETRY_S in all (= Core
    _replace_retry; inside _replace_budget() the deadline is shared). Anything else raises at once; the last
    PermissionError raises when the time is up (at least one try is always made)."""
    if budget is None and getattr(_REPLACE_TLS, "active", False):
        if _REPLACE_TLS.until is None:
            _REPLACE_TLS.until = time.monotonic() + REPLACE_RETRY_S
        deadline = _REPLACE_TLS.until
    else:
        deadline = time.monotonic() + (REPLACE_RETRY_S if budget is None else budget)
    delay = 0.01
    while True:
        try:
            return os.replace(src, dst)
        except PermissionError:
            left = deadline - time.monotonic()
            if left <= 0:
                raise
            time.sleep(min(delay, left))
            delay = min(delay * 2, 0.25)


def write_atomic(path, data):
    """data (bytes) -> path: a temp file in the same directory, flush + fsync, chmod 0600 (os.chmod, not os.fchmod:
    Windows Python 3.12 has none, F307-P), os.replace, then fsync the directory (POSIX). A crash at any point leaves
    either the old file or the new one, never a part; the temp file is removed on failure. A PermissionError on the
    replace (Windows: the target open elsewhere) is retried for REPLACE_RETRY_S (F312)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        _replace_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if os.name != "nt":
        try:
            dfd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass


def _aside(path):
    """keep a corrupt file for diagnosis: move it to <path>.corrupt-<unix time>[-n] (never over an earlier copy); if the
    move fails, copy it. -> the copy's name, or None when neither worked (then the caller must not write)."""
    base = "%s.corrupt-%d" % (path, int(time.time()))
    dest, n = base, 1
    while os.path.lexists(dest):
        dest, n = "%s-%d" % (base, n), n + 1
    try:
        os.rename(path, dest)                     # a symlink moves as the link itself, never what it points at
        return os.path.basename(dest)
    except OSError:
        pass
    try:
        if os.path.islink(path):
            return None
        shutil.copyfile(path, dest)
        return os.path.basename(dest)
    except OSError:
        return None


def _read_regular(path, limit):
    """-> ("missing" | "ok" | "bad" | "link", obj). Never follows a symlink: lstat first, then O_NOFOLLOW (= Plugins)."""
    import stat as _stat
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "bad", None
    if _stat.S_ISLNK(st.st_mode) or not _stat.S_ISREG(st.st_mode):
        return "link", None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as f:
            raw = f.read(limit + 1)
    except OSError:
        return "bad", None
    if len(raw) > limit:
        return "bad", None
    try:
        return "ok", json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        return "bad", None


def _num(x):
    """a finite number as float, else None (bool, NaN, +-inf: Python's json reads and writes NaN / Infinity, and a NaN
    defeats every max() and comparison the stricter-value merge and the rollback guard rely on; review B)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return float(x) if math.isfinite(x) else None
        except OverflowError:
            return None
    return None


def _nonfinite(x):
    """True for a float NaN / +-inf (or an int too large for a float): the wrong shape in a store file or a journal."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and _num(x) is None


def _finite_map(m):
    """a {key: number} map (newest, hard_lock_newest) without a non-finite value."""
    return not isinstance(m, dict) or not any(_nonfinite(v) for v in m.values())


def _statement(env):
    """a stored grant's statement, or None when the entry is not {statement: {item_id: a str, ent_id: a str, issued: an
    int, ...}} (SEC read NEW-1d: an unhashable ent_id made the journal merge's _lifted / _fresh raise on every read;
    NEW-1c: an unhashable item_id made hard_lock's set of items raise; security
    read NEW-1: a hand-edited entry must never raise inside a 403 hard_lock or a read). The one helper for every place
    that reads stored grants (_held, items, hard_lock, drop_item, apply_revocations, _lock_record)."""
    st = env.get("statement") if isinstance(env, dict) else None
    if not isinstance(st, dict) or not isinstance(st.get("issued"), int) or isinstance(st.get("issued"), bool) \
            or not isinstance(st.get("item_id"), str) or not isinstance(st.get("ent_id"), str):
        return None                                     # issued an int only (= Core / Plugins), item_id, ent_id strings
    return st


def _str_list(x):
    """an id list (revoked, hard_lock_seen, fresh, star_held; journal ids / held / ...): None or a list of strings only
    (SEC read NEW-1 list variant: a [1] element made every set() / sorted() over it raise TypeError)."""
    return x is None or (isinstance(x, list) and all(isinstance(e, str) for e in x))


def _issued(env):
    st = _statement(env)
    return int(st["issued"]) if st is not None else None


def _file_keys(d):
    """d without its process-only keys (those that start with "_": _star_cleared, _unsaved, _seen_reset, _rebuilt ...):
    stripped before every write and on every read of grants.json and the journals (= Plugins _file_keys, Core F324)."""
    return {k: v for k, v in d.items() if not (isinstance(k, str) and k.startswith("_"))}


class _Reply(dict):
    """licd's JSON reply, plus .server_time: licd's clock (server_time_checked: body server_time, else the Date header,
    the earlier of the two when they disagree by more than 5 min; None if neither) and .server_source (body | date)."""
    server_time = None
    server_source = None


class EntitlementCache(object):
    """verified grants on disk (<config_dir>/store/grants.json, 0600), shared with the plugin host's GrantCache and Core's
    gate (same file, same keys, the same lock: <store>/grants.json.lock).
    * anti-replay: per ent_id the newest `issued` is kept; an older copy of the same grant (a backup restored after a
      downgrade or a revocation) is refused (`replayed`); a revoked ent_id never comes back;
    * clock rollback: the latest time seen is kept; a clock more than 48 h behind it makes every grant `expired`;
    * offline: a grant counts until not_after (180 days) and says `refresh_due` after refresh_after (24 h);
    * one writer at a time (F303-P): every load-modify-save runs under file_lock(grants.json) and saves with
      write_atomic; the clock is read BEFORE the lock is taken, so nothing outside runs inside it;
    * fail closed (F305-P, = Plugins' GrantCache rule): a file that does not parse, is cut off, or has the wrong shape is
      kept aside as grants.json.corrupt-<t>, never rewritten in place, and every item is hard-locked ("*": "corrupt",
      hard_lock "corrupt") until licd issues a fresh grant per item; missing -> empty."""

    _SHAPE = (("grants", dict), ("revoked", list), ("newest", dict), ("seen", (int, float)), ("locks", dict),
              ("hard_locks", dict))
    _PLUGIN_SHAPE = (("last_sync", (int, float)), ("revoked_items", dict))     # the plugin host's keys, when present

    def __init__(self, cfg_dir, pubkeys=None, clock=time.time):
        self.cfg_dir = cfg_dir
        self.dir = os.path.join(cfg_dir, "store")
        self.path = os.path.join(self.dir, "grants.json")
        self.pubkeys = pubkeys
        self.clock = clock
        self.lock = threading.RLock()
        self._nowrite = False     # F311: set while the store lock is busy past LOCK_TIMEOUT_S: nothing writes grants.json
        self._pending = None      # review Minor 1 (= Core _mem): a lock-adding change that hit StoreBusy, held in memory

    @contextlib.contextmanager
    def _locked(self, lock_only=False):
        """the store lock; yields True when it was BUSY past LOCK_TIMEOUT_S (lock_only only). A busy or unopenable lock
        file is never an unlock: StoreBusy / OSError refuses add() and set_locks(). A call that only reads or only ADDS
        locks (lock_only) still runs under the in-process lock (F303 contract rule 8), but F311: when the lock is busy
        nothing in it writes or moves grants.json (reads fail closed in memory; hard_lock and apply_revocations journal
        first and wait LOCK_RETRY_S more, see _adding_write; drop_item raises StoreBusy). An unopenable lock file (a
        planted symlink: nobody can hold it) runs as before, with writes."""
        with self.lock:
            busy = False
            try:
                cm = file_lock(self.path)
                cm.__enter__()
            except StoreBusy:
                if not lock_only:
                    raise
                cm, busy = None, True
            except OSError:
                if not lock_only:
                    raise
                cm = None
            prev, self._nowrite = self._nowrite, busy
            try:
                yield busy
            finally:
                self._nowrite = prev
                if cm is not None:
                    cm.__exit__(None, None, None)

    def _adding_write(self, now, change):
        """F311 for a write that only ADDS locks (hard_lock, apply_revocations): change(d) -> (result, delta) on the loaded
        record, saved under the store lock; delta = only what the change adds (item locks + lock time / seen ids /
        hard_lock_newest / licd_high_water, revoked ids). Busy past LOCK_TIMEOUT_S: the DELTA alone goes into the
        current lock journal record FIRST (_journal_delta: fail closed, grants.json is never written without the lock,
        and a read that raced another writer never journals a rebuilt "*"), then the whole load-change-save is redone
        under the lock if it frees within LOCK_RETRY_S (a writer that loaded before the change and overwrote the journal
        is merged then); else StoreBusy, with the change in the journal."""
        held = []
        try:
            with _replace_budget():                  # review Minor 2: one replace budget for the whole call
                return self._adding_write_in_budget(now, change, held)
        except OSError:
            if held:                                 # review Minor 1: StoreBusy (or a failed save) never drops the
                with self.lock:                      # change: it holds in memory, fails closed on every load here and
                    self._pending = self._delta_into(self._pending or {}, held[-1])   # goes out with the next save
                    self._pending.setdefault("deltas", []).append(held[-1])
            raise

    def _adding_write_in_budget(self, now, change, held):
        with self._locked(lock_only=True) as busy:
            d = self._load(now)
            out, delta = change(d)
            held.append(delta)
            if not busy:
                self._save(d)
                return out
            self._journal_delta(delta)
        with self.lock, file_lock(self.path, timeout=LOCK_RETRY_S):
            d = self._load(now)
            out, delta = change(d)
            held.append(delta)
            self._save(d)
        return out

    def _apply_pending(self, d, now):
        """review Minor 1: a lock-adding change that hit StoreBusy (held in memory) applies to every record loaded or
        saved here until a locked save lands: its item locks (unless a fresh verified grant in d lifts one since),
        lock time / seen ids / hard_lock_newest, revoked ids, licd_high_water. Grow-only."""
        m = self._pending
        if not m:
            return d
        at = _num(m.get("hard_lock_at")) or 0.0
        ln = self._max_merge(d.get(J_LOCK_NEWEST), m.get(J_LOCK_NEWEST))
        seen = set(d.get("hard_lock_seen") or ()) | set(m.get("hard_lock_seen") or ())
        hl = d.setdefault("hard_locks", {})
        locked = False
        for i, c in (m.get("hard_locks") or {}).items():
            if i in hl:
                locked = True
                continue
            if self._lifted(d, i, at, ln, seen, now):
                continue                                 # licd granted it fresh since: the lock is lifted for real
            hl[i] = c
            d.setdefault("locks", {})[i] = c
            locked = True
        if locked:
            d["hard_lock_at"] = max(_num(d.get("hard_lock_at")) or 0.0, at)
            d["hard_lock_seen"] = sorted(x for x in seen if isinstance(x, str))
            d[J_LOCK_NEWEST] = ln
            d.setdefault("hard_lock_clock", m.get("hard_lock_clock") or "local")
            codes = [x.get("code") for x in m.get("deltas") or () if x.get("code")]
            if codes:
                d.setdefault("hard_lock", codes[-1])
        if m.get(J_REVOKED):
            d["revoked"] = sorted({x for x in d.get("revoked") or () if isinstance(x, str)} | set(m[J_REVOKED]))
            for e in list(d.get("grants") or {}):
                if e in d["revoked"]:
                    d["grants"].pop(e, None)
        if _num(m.get(J_LICD_SEEN)):
            d[J_LICD_SEEN] = max(_num(d.get(J_LICD_SEEN)) or 0.0, _num(m[J_LICD_SEEN]))
        return d

    @staticmethod
    def _delta_into(rec, delta):
        """a change's additions (see _adding_write) into a journal record (or a loaded record: same keys), grow-only."""
        items = {i: c for i, c in (delta.get("items") or {}).items() if isinstance(i, str) and i != "*"}
        hl = rec.setdefault("hard_locks", {})
        for i, c in items.items():
            hl.setdefault(i, c)
        if items:
            rec["hard_lock_at"] = max(_num(rec.get("hard_lock_at")) or 0.0, _num(delta.get("at")) or 0.0)
            rec["hard_lock_seen"] = sorted({x for x in rec.get("hard_lock_seen") or () if isinstance(x, str)}
                                           | set(delta.get("seen_ids") or ()))
            rec[J_LOCK_NEWEST] = EntitlementCache._max_merge(rec.get(J_LOCK_NEWEST), delta.get("lock_newest"))
            rec.setdefault("hard_lock_clock", delta.get("clock") or "local")
        if delta.get("revoked"):
            rec[J_REVOKED] = sorted({x for x in rec.get(J_REVOKED) or () if isinstance(x, str)} | set(delta["revoked"]))
        if _num(delta.get("hw")):
            rec[J_LICD_SEEN] = max(_num(rec.get(J_LICD_SEEN)) or 0.0, _num(delta["hw"]))
        return rec

    def _journal_delta(self, delta):
        """F311 busy path (review Minor 3): merge ONLY the change's additions into the current lock journal record and
        write it back at its own gen (no bump: grants.json is not written), so every reader merges it into the file it
        finds. A journal that is bad is left for the next locked load (F314), never overwritten here."""
        j, _bad = self._journal()
        bad = set(getattr(self, "_bad_journals", ()))
        if j is not None:
            rec = json.loads(json.dumps(j))
        else:
            kind, f = _read_regular(self.path, 64 << 20)
            gen = f.get("gen") if kind == "ok" and isinstance(f, dict) else 0
            rec = {"v": 1, "gen": gen if isinstance(gen, int) and not isinstance(gen, bool) else 0, "hard_locks": {},
                   "star": False, "fresh": [], "held": [], "ids": [], "seen": 0, J_NEWEST: {}, J_REVOKED: []}
        self._delta_into(rec, delta)
        rec["held"] = sorted({x for x in rec.get("held") or () if isinstance(x, str)} | set(delta.get("items") or {}))
        rec["ids"] = sorted({x for x in rec.get("ids") or () if isinstance(x, str)} | set(delta.get("seen_ids") or ())
                            | set(delta.get("revoked") or ()))
        for p in self._journal_paths():
            if p in bad:
                continue
            try:
                write_atomic(p, json.dumps(_file_keys(rec), sort_keys=True).encode())
            except OSError:
                pass                                    # one copy is enough

    JOURNALS = (os.path.join("store", "grants.journal.json"), os.path.join("plugins", "store-lock-journal.json"))
    MAX_HELD = 4096

    @staticmethod
    def _empty():
        return {"grants": {}, "revoked": [], "newest": {}, "seen": 0, "locks": {}, "hard_locks": {}}

    # ---- the lock journal (F304/F309, = Plugins GrantCache 9351f0a99c..b5507ccf83, DESIGN.md s31.3) ---------------------
    def _journal_paths(self):
        return [os.path.join(self.cfg_dir, j) for j in self.JOURNALS]

    def _journal(self):
        """-> (the newest readable journal or None, bad). bad: one is there but unreadable, wrongly shaped or a symlink.
        Format {v, gen, hard_locks, star, fresh, held, ids, hard_lock_at?, hard_lock_seen?, hard_lock_clock?, seen?,
        J_NEWEST, J_LOCK_NEWEST?, J_LICD_SEEN?, J_REVOKED}."""
        best, bad = None, False
        self._bad_journals = []                         # F314: the bad ones (unreadable, wrong shape, a symlink)
        for p in self._journal_paths():
            kind, j = _read_regular(p, 4 << 20)         # lstat + O_NOFOLLOW: a symlinked journal is never followed
            if kind == "missing":
                continue
            if kind == "ok" and isinstance(j, dict):
                j = _file_keys(j)                       # process-only keys are never read from a file
            ok = (kind == "ok" and isinstance(j, dict) and isinstance(j.get("gen"), int) and not isinstance(j["gen"], bool)
                  and isinstance(j.get("hard_locks", {}), dict) and isinstance(j.get("hard_lock_seen", []), list)
                  and isinstance(j.get("held", []), list) and isinstance(j.get("ids", []), list)
                  and isinstance(j.get("hard_lock_at") or 0, (int, float)) and isinstance(j.get("seen") or 0, (int, float))
                  and isinstance(j.get("fresh", []), list) and isinstance(j.get(J_NEWEST, {}), dict)
                  and isinstance(j.get(J_LOCK_NEWEST, {}), dict) and isinstance(j.get(J_LICD_SEEN) or 0, (int, float))
                  and isinstance(j.get(J_REVOKED, []), list)
                  # SEC read NEW-1: id lists hold strings only; review B: no non-finite number (bad journal, F314 way)
                  and all(_str_list(j.get(k)) for k in ("ids", "held", J_REVOKED, "hard_lock_seen", "fresh"))
                  and not any(_nonfinite(j.get(k)) for k in ("hard_lock_at", "seen", J_LICD_SEEN))
                  and _finite_map(j.get(J_NEWEST)) and _finite_map(j.get(J_LOCK_NEWEST)))
            if not ok:
                bad = True
                self._bad_journals.append(p)
                continue
            if best is None or j["gen"] > best["gen"]:
                best = j
        return best, bad

    @staticmethod
    def _fresh(st, at, lock_newest, known):
        """hard-lock v4: while locked, a grant (already verified for this machine, not revoked) is fresh when its `issued`
        is after the lock time AND either its ent_id was known at lock time and `issued` is strictly newer than the
        newest issued recorded for it then (licd re-signs the same ent_id on refresh), or its ent_id is new (a relink).
        A known id whose lock-time newest is not on record fails closed. Nonces are not used."""
        try:
            issued = int(st.get("issued") or 0)
        except (TypeError, ValueError, OverflowError):
            return False
        if not issued > float(at):              # NaN-safe: a lock time that compares as nothing is never passed
            return False
        e = st.get("ent_id")
        if not isinstance(e, str):              # SEC read NEW-1d: _lifted reads raw grants; never hash a list / dict id
            return False
        if isinstance(lock_newest, dict) and e in lock_newest:
            try:
                return issued > int(lock_newest[e] or 0)
            except (TypeError, ValueError, OverflowError):
                return False
        return e not in known                   # known at lock time but no newest on record: never fresh

    @staticmethod
    def _max_merge(*maps):
        out = {}
        for m in maps:
            for k, v in (m or {}).items() if isinstance(m, dict) else ():
                if isinstance(k, str) and _num(v) is not None:
                    out[k] = max(int(v), out.get(k, int(v)))
        return out

    def _lifted(self, d, item_id, at, lock_newest, known, now):
        """a fresh grant for item_id (v4 _fresh) that verifies is in the file."""
        pub = install_key(self.cfg_dir)[1]
        lt = d.get("licd_time") or {}
        for e, env in (d.get("grants") or {}).items():
            st = env.get("statement") if isinstance(env, dict) else None
            if not isinstance(st, dict) or st.get("item_id") != item_id or not self._fresh(st, at, lock_newest, known):
                continue
            try:
                ref = lt.get(e)
                verify_grant(env, self.pubkeys, now, pub,
                             issued_ref=max(now, float(ref)) if isinstance(ref, (int, float)) else None)
                return True
            except (GrantError, ValueError, TypeError, KeyError):
                continue
        return False

    def _merge(self, d, j, now):
        """the journal's lock record into d; it only ever adds (a recorded lock goes only with a fresh verified grant in
        d). The journal's high-water time (`seen`) is the rollback anchor even when the file lost it."""
        if not j:
            return d
        if not d.get("_seen_reset"):                # F310 (b): a verified licd reset is written as is, under this lock
            d["seen"] = max(float(d.get("seen") or 0), float(j.get("seen") or 0))
        d["newest"] = self._max_merge(d.get("newest"), j.get(J_NEWEST))      # anti-replay survives a lost file
        # F318: revoked ids only grow, whatever another writer did to grants.json (merged before any sync result)
        d["revoked"] = sorted(set(d.get("revoked") or ()) | {x for x in j.get(J_REVOKED) or () if isinstance(x, str)})
        if j.get(J_LICD_SEEN) or d.get(J_LICD_SEEN):          # grow-only: plain max of file and journal (= Plugins);
            d[J_LICD_SEEN] = max(float(d.get(J_LICD_SEEN) or 0),  # the now + 24 h cap applies where a licd time is first
                                 float(j.get(J_LICD_SEEN) or 0))  # recorded and on lock times, never to a stored value
        ln = self._max_merge(d.get(J_LOCK_NEWEST), j.get(J_LOCK_NEWEST)) if (J_LOCK_NEWEST in d or J_LOCK_NEWEST in j) else None
        jat = float(j.get("hard_lock_at") or 0)
        at = max(float(d.get("hard_lock_at") or 0), jat)
        seen = set(d.get("hard_lock_seen") or ()) | set(j.get("hard_lock_seen") or ())
        hl = d["hard_locks"]
        if j.get("star") and "*" not in hl and not d.get("_star_cleared"):
            hl["*"] = "corrupt"
            d["fresh"] = sorted(set(d.get("fresh") or ()) | set(j.get("fresh") or ()))
            d.setdefault("star_held", sorted(set(j.get("held") or ()) - {"*"}))
        for i, code in sorted((j.get("hard_locks") or {}).items()):
            if not isinstance(i, str) or i == "*" or i in hl:
                continue
            if "*" in hl and i not in (d.get("fresh") or ()):
                continue                        # covered by "*" (kept in the journal until a fresh grant lifts it)
            if not self._lifted(d, i, jat, ln, seen, now):
                hl[i] = str(code)[:64]
        if hl:
            d["hard_lock_at"] = at
            d["hard_lock_seen"] = sorted(seen)
            if ln is not None:
                d[J_LOCK_NEWEST] = ln
            d.setdefault("hard_lock_clock", j.get("hard_lock_clock") or "local")
        return d

    def _held(self, d):
        held = {(_statement(env) or {}).get("item_id") for env in d["grants"].values()}
        return {x for x in held | set(d["locks"]) | set(d["hard_locks"]) | set(d.get("revoked_items") or {})
                | set(d.get("fresh") or ()) if isinstance(x, str)} - {"*"}

    @staticmethod
    def _ids(d):
        """every grant id this file knows (F309: known at a corruption lock; v4 uses no nonces)."""
        ids = set(d["newest"]) | set(d["grants"]) | set(d["revoked"])
        return {x for x in ids if isinstance(x, str)}

    def _load(self, now=None):
        """call with the lock held. The file merged with the lock journal (journal read first: an honest writer saves the
        file, then the journal). Missing with no journal -> empty (first start). FAIL CLOSED -> _corrupt: the file
        unreadable / cut off / not an object / a known key of the wrong type / a symlink or not a regular file /
        missing while a journal exists / older than the journal (gen) / a journal that is there but unreadable."""
        now = float(time.time() if now is None else now)
        j, jbad = self._journal()
        kind, d = _read_regular(self.path, 64 << 20)
        if kind == "link":
            return self._corrupt(now, "link", j)
        if kind == "bad" or (kind == "ok" and not self._shape_ok(d)):
            return self._corrupt(now, "corrupt", j)
        if kind == "ok":
            d = _file_keys(d)                           # process-only keys are never read from the file
            if "hard_lock_source" in d:                 # legacy "server" -> body, unknown -> local; written as the enum
                d["hard_lock_source"] = _lock_source(d["hard_lock_source"])
        if kind == "missing":
            if j is not None or jbad:
                return self._corrupt(now, "missing", j)
            return self._apply_pending(self._empty(), now)
        if j is not None and int(d.get("gen") or 0) < j["gen"]:
            # a restored copy, a stale writer that skipped the lock, or a journal edited ahead (gen + n): the file still
            # parses, so its lock record is kept and merged stricter-value with the journal's (F317: the journal may only add)
            return self._corrupt(now, "rolled_back", j, f=d)
        if jbad:
            # F314: only a journal is bad; grants.json parses, so it stays where it is and its lock record (item locks, lock
            # time, seen ids, newest, revoked) is kept under the "*"; the bad journals are moved aside and rebuilt
            return self._corrupt(now, "journal", j, f=d)
        return self._apply_pending(self._merge(d, j, now), now)

    def _shape_ok(self, d):
        if not isinstance(d, dict):
            return False
        for k, typ in self._SHAPE:
            d.setdefault(k, typ[0]() if isinstance(typ, tuple) else typ())
            if not isinstance(d[k], typ) or isinstance(d[k], bool):
                return False
        for k, typ in self._PLUGIN_SHAPE:
            if k in d and (not isinstance(d[k], typ) or isinstance(d[k], bool)):
                return False
        if d.get("hard_lock_at") is not None and (not isinstance(d["hard_lock_at"], (int, float)) or
                                                  isinstance(d["hard_lock_at"], bool) or
                                                  not isinstance(d.get("hard_lock_seen"), list)):
            return False
        if not isinstance(d.get("gen", 0), int) or isinstance(d.get("gen", 0), bool):
            return False
        if "hard_lock_source" in d and not isinstance(d["hard_lock_source"], str):
            return False                                # the enum, legacy "server" or an unknown name (read as local)
        if d.get(J_LICD_SEEN) is not None and _num(d.get(J_LICD_SEEN)) is None:
            return False
        # SEC read NEW-1: id lists hold strings only, every stored grant is {statement: {item_id: a str, ent_id: a
        # str, issued: an int, ...}} (fail closed, rebuilt)
        if not all(_str_list(d.get(k)) for k in ("revoked", "hard_lock_seen", "fresh", "star_held")):
            return False
        if any(_statement(env) is None for env in d["grants"].values()):
            return False
        # review B: a non-finite lock time / seen / issued map value is the wrong shape (fail closed, rebuilt)
        if _nonfinite(d.get("hard_lock_at")) or _nonfinite(d.get("seen")) or _nonfinite(d.get("last_sync")) \
                or not _finite_map(d.get("newest")) \
                or not _finite_map(d.get(J_LOCK_NEWEST)) or not _finite_map(d.get("licd_time")):
            return False
        return isinstance(d.get("licd_time", {}), dict) and isinstance(d.get("fresh", []), list)

    @staticmethod
    def _lock_record(f):
        """F317 (= Plugins _lock_record P:528): the lock record of a grants.json that still parses (shape-checked), for
        the grow-only rebuild merge. -> None for anything else."""
        if not isinstance(f, dict):
            return None
        hl = f.get("hard_locks") if isinstance(f.get("hard_locks"), dict) else {}
        grants = f.get("grants") if isinstance(f.get("grants"), dict) else {}
        held = {(_statement(env) or {}).get("item_id") for env in grants.values()}
        for k in ("locks", "revoked_items"):
            held |= set(f.get(k) or {}) if isinstance(f.get(k), dict) else set()
        newest = EntitlementCache._max_merge(f.get("newest") if isinstance(f.get("newest"), dict) else {})
        revoked = f.get("revoked") if isinstance(f.get("revoked"), list) else []
        seen_ids = f.get("hard_lock_seen") if isinstance(f.get("hard_lock_seen"), list) else []
        ids = set(seen_ids) | set(newest) | set(grants) | set(revoked)
        local = f.get("hard_lock_clock") == "local"
        at = _num(f.get("hard_lock_at")) or 0.0
        ln = f.get(J_LOCK_NEWEST) if isinstance(f.get(J_LOCK_NEWEST), dict) else {}
        gen = f.get("gen")
        return {"at": 0.0 if local else at, "at_local": at if local else 0.0, "hw": _num(f.get(J_LICD_SEEN)) or 0.0,
                "ids": {x for x in ids if isinstance(x, str)}, "newest": newest,
                "lock_newest": EntitlementCache._max_merge(ln), "revoked": {x for x in revoked if isinstance(x, str)},
                "seen": _num(f.get("seen")) or 0.0, "gen": gen if isinstance(gen, int) and not isinstance(gen, bool) else 0,
                "items": {i: str(c)[:64] for i, c in hl.items() if isinstance(i, str) and i != "*"},
                "held": {x for x in held | set(hl) if isinstance(x, str)} - {"*"}}

    def _newest_copy(self):
        """the newest grants.json.corrupt-<t>[-n] copy that still parses with the right shape (= Plugins _newest_copy
        P:548), or None: its seen ids, newest, revoked and seen survive a rebuild too."""
        try:
            names = [n for n in os.listdir(self.dir) if n.startswith(os.path.basename(self.path) + ".corrupt-")]
        except OSError:
            return None

        def key(n):
            try:
                return tuple(int(x) for x in n.rsplit(".corrupt-", 1)[1].split("-"))
            except ValueError:
                return (-1,)
        for n in sorted(names, key=key, reverse=True)[:8]:
            kind, c = _read_regular(os.path.join(self.dir, n), 64 << 20)
            if kind == "ok" and self._shape_ok(c):
                return _file_keys(c)
        return None

    def _corrupt(self, now=None, why="corrupt", j=None, f=None):
        """the file is kept aside (moved, else copied) and a corruption lock on every item starts. The lock record is
        REBUILT from the journal (F309): its per-item locks (merged on save), every grant id it knew (`ids`,
        hard_lock_seen), the newest issued per ent_id (J_NEWEST -> both `newest` and J_LOCK_NEWEST, so an old grant is
        never strictly newer), the revoked ids (F318) and the rollback high-water `seen`. F317 (the journal may only add,
        = Plugins P:568): f, a grants.json that still parses (rolled_back), keeps its own lock record, merged with the
        journal's taking the stricter value everywhere: lock time / licd high water = max, item locks, seen ids and revoked
        ids = union, newest / hard_lock_newest = per-key max; the newest .corrupt-<t> copy that parses adds its ids.
        The lock time is on licd's clock (corrupt_clock_fast): the latest licd time on record, capped at now +
        LOCK_AHEAD_CAP_S; only with no licd time on record, the local clock. If the file cannot be kept aside, nothing is
        written and the lock holds in memory. why: corrupt | link | missing | rolled_back | journal."""
        now = float(time.time() if now is None else now)
        if self._nowrite:                                # F311: no store lock: nothing is moved or written (in memory)
            kept = None
        elif why == "journal":                           # F314: the file is good; the bad journals (or links) go aside
            kept = None
            for p in getattr(self, "_bad_journals", ()):
                kept = _aside(p) or kept
        else:
            kept = None if why == "missing" else _aside(self.path)
        jj = j or {}
        fr = [r for r in (self._lock_record(f),) if r]
        cr = self._lock_record(self._newest_copy())
        if cr:                                           # its ids, not its item locks (those may have been lifted since)
            fr.append(dict(cr, items={}, held=set()))
        hw = max([_num(jj.get(J_LICD_SEEN)) or 0.0] + [r["hw"] for r in fr])
        licd = max([_num(jj.get("hard_lock_at")) or 0.0, hw] + [r["at"] for r in fr])
        local = max([r["at_local"] for r in fr] + [0.0])
        if licd:
            at, clock, src = min(max(licd, local), now + LOCK_AHEAD_CAP_S), "server", "licd_high_water"
        elif local:
            at, clock, src = min(local, now + LOCK_AHEAD_CAP_S), "local", "local"
        else:
            at, clock, src = now, "local", "local"
        seen = {x for x in set(jj.get("hard_lock_seen") or ()) | set(jj.get("ids") or ()) if isinstance(x, str)}
        seen.update(*[r["ids"] for r in fr])
        newest = self._max_merge(jj.get(J_NEWEST), *[r["newest"] for r in fr])
        items = {}
        for r in fr:
            items.update(r["items"])
        held = (set(jj.get("held") or ()) | set().union(*[r["held"] for r in fr])) if (j or fr) else None
        revoked = {x for x in jj.get(J_REVOKED) or () if isinstance(x, str)}.union(*[r["revoked"] for r in fr])
        d = self._empty()
        d.update(hard_locks=dict(items, **{"*": "corrupt"}), hard_lock_at=at, hard_lock_seen=sorted(seen),
                 hard_lock="corrupt", hard_lock_clock=clock, hard_lock_source=src, fresh=[], newest=newest,
                 **{J_LOCK_NEWEST: self._max_merge(jj.get(J_LOCK_NEWEST), newest, *[r["lock_newest"] for r in fr])},
                 corrupt_copy=kept, corrupt_why=why, hard_lock_why=why,
                 seen=max([_num(jj.get("seen")) or 0.0] + [r["seen"] for r in fr]),
                 star_held=sorted({x for x in held if isinstance(x, str)} - {"*"}) if held is not None else None,
                 revoked=sorted(revoked))                # F318: survives a rebuild
        d["locks"] = dict(items)
        gens = [int(jj["gen"])] if jj.get("gen") is not None else []
        gens += [r["gen"] for r in fr[:1] if f is not None]
        if gens:
            d["gen"] = max(gens)
        if hw:
            d[J_LICD_SEEN] = hw                          # stored as it is: grow-only (the lock time above is capped)
        if self._nowrite or (why not in ("missing", "journal") and kept is None):
            d["_unsaved"] = True                         # in memory only: never written, so the bad file is never overwritten
        try:
            self._save(d, now)
        except OSError:
            pass                                         # the lock holds in memory (fail closed)
        return d

    @staticmethod
    def _hard_locked(d, item_id):
        hl = d.get("hard_locks") or {}
        return item_id in hl or ("*" in hl and item_id not in (d.get("fresh") or ()))

    def _save(self, d, now=None):
        with _replace_budget():
            return self._save_in_budget(d, now)

    def _save_in_budget(self, d, now=None):
        self._apply_pending(d, float(self.clock() if now is None else now))   # review Minor 1: never saved without it
        pending = self._pending
        self._save_core(d, now)
        if pending is not None and self._pending is pending and not d.get("_unsaved") and not self._nowrite \
                and d.get("_saved"):
            self._pending = None                         # landed in grants.json (and the journal): nothing held now
        d.pop("_saved", None)

    def _save_core(self, d, now=None):
        """under the store lock: merge what another writer journalled since our load, bump gen, write the journal in both
        places FIRST (atomic replace, fsync; one copy is enough) as a pre-record at the file's current gen, then grants.json,
        then the journal at the new gen (F312): a crash or a replace still busy after REPLACE_RETRY_S (StoreBusy) leaves
        the change in the journal and the old file whole, merged on the next load. The journal only grows: locks union (an
        item leaves only with a fresh verified grant in the file), ids / held / hard_lock_seen / revoked union, lock time
        and high-water time max. Never weakens a lock (F303, F304, F309)."""
        if d.get("_unsaved") or self._nowrite:
            return                                      # a corrupt file that could not be kept aside is never overwritten;
        now = float(self.clock() if now is None else now)   # F311: nothing writes grants.json without the store lock
        j, _bad = self._journal()
        self._merge(d, j, now)
        d.pop("_star_cleared", None)
        reset = d.pop("_seen_reset", False)             # F310 (b): the journal's `seen` follows the verified reset
        if isinstance(d.get("licd_time"), dict):        # licd's clock per grant held, only for grants still held
            d["licd_time"] = {k: v for k, v in d["licd_time"].items() if k in d["grants"]}
        base = max(int(d.get("gen") or 0), int((j or {}).get("gen") or 0))
        d["gen"] = base + 1
        rec = self._record(d, j, now, reset)
        # F312: the journal FIRST, as a pre-record at the gen grants.json has now (its locks / ids / revoked / newest only
        # add, so a reader merges it into the file it finds); then grants.json; then the journal at the new gen. A crash
        # or a replace that stays busy past REPLACE_RETRY_S in between leaves the lock in the journal and the old file
        # whole, with no false rolled_back; the busy replace raises StoreBusy.
        self._write_journals(dict(rec, gen=base))
        try:
            write_atomic(self.path, json.dumps(_file_keys(d)).encode())     # never a process-only "_" key
        except PermissionError as e:                    # still busy after REPLACE_RETRY_S: the lock is in the journal
            raise StoreBusy("grants.json stays busy (another program has it open); the change is in the lock journal") from e
        self._write_journals(rec)
        d["_saved"] = True

    def _write_journals(self, rec):
        for p in self._journal_paths():
            try:
                write_atomic(p, json.dumps(rec, sort_keys=True).encode())
            except OSError:
                pass                                    # one copy is enough; the other may be on a read-only share

    def _record(self, d, j, now, reset=False):
        """the lock journal record for d (merged with j, d["gen"] set): grow-only (see _save)."""
        pending = {}
        if j:                                           # journal locks still covered by "*" stay in the journal
            seen = set(d.get("hard_lock_seen") or ()) | set(j.get("hard_lock_seen") or ())
            ln = self._max_merge(d.get(J_LOCK_NEWEST), j.get(J_LOCK_NEWEST)) if (J_LOCK_NEWEST in d or J_LOCK_NEWEST in j) else None
            for i, code in (j.get("hard_locks") or {}).items():
                if isinstance(i, str) and i != "*" and i not in d["hard_locks"] and "*" in d["hard_locks"] \
                        and not self._lifted(d, i, float(j.get("hard_lock_at") or 0), ln, seen, now):
                    pending[i] = code
        items = dict(pending, **{i: c for i, c in d["hard_locks"].items() if i != "*"})
        rec = {"v": 1, "gen": d["gen"], "hard_locks": items, "star": "*" in d["hard_locks"],
               "fresh": sorted(d.get("fresh") or ()),
               "held": sorted(set((j or {}).get("held") or ()) | self._held(d))[-self.MAX_HELD:],
               "ids": sorted(set((j or {}).get("ids") or ()) | self._ids(d))[-20000:],
               "seen": (float(d.get("seen") or 0) if reset
                        else max(float(d.get("seen") or 0), float((j or {}).get("seen") or 0))),
               J_NEWEST: dict(sorted(self._max_merge((j or {}).get(J_NEWEST), d.get("newest")).items(),
                                     key=lambda kv: kv[1])[-20000:]),
               J_REVOKED: sorted({x for x in set((j or {}).get(J_REVOKED) or ()) | set(d.get("revoked") or ())
                                  if isinstance(x, str)})[-20000:]}                             # F318: grow-only
        licd = max(float(d.get(J_LICD_SEEN) or 0), float((j or {}).get(J_LICD_SEEN) or 0))   # grow-only
        if licd:
            rec[J_LICD_SEEN] = licd
        if items or rec["star"]:
            rec.update(hard_lock_at=max(float(d.get("hard_lock_at") or 0), float((j or {}).get("hard_lock_at") or 0)),
                       hard_lock_seen=sorted(set(d.get("hard_lock_seen") or ()) | set((j or {}).get("hard_lock_seen") or ())),
                       hard_lock_clock=d.get("hard_lock_clock") or (j or {}).get("hard_lock_clock") or "local")
            if J_LOCK_NEWEST in d or J_LOCK_NEWEST in (j or {}):
                rec[J_LOCK_NEWEST] = self._max_merge(d.get(J_LOCK_NEWEST), (j or {}).get(J_LOCK_NEWEST))
        return rec

    def _now(self, d, now):
        if now < float(d.get("seen") or 0) - ROLLBACK_S:
            return now, True
        d["seen"] = max(float(d.get("seen") or 0), now)
        return now, False

    def add(self, env, server_now=None):
        """verify and keep a grant licd sent; -> its statement. GrantError adds codes revoked, replayed, clock_rollback,
        pre_lock. server_now: licd's clock from the reply that carried the grant (_Reply.server_time); the not-yet
        check uses it (CLOCK_SKEW_S either way), the local clock only when it is None.
        F310 (b) (ruling (b), = Plugins add / _seen_back P:862/P:921): with the clock more than 48 h behind the last time
        seen, a grant is still VERIFIED (signature, machine, issuer, not revoked, issued strictly newer than any copy of its
        ent_id) and only then may move a `seen` that ran ahead of licd back to licd's signed time (_seen_back); an unsigned
        server_time / Date or a bad signature never moves it (clock_rollback). The reset never unlocks anything itself."""
        _seed, pub = install_key(self.cfg_dir)
        now = self.clock()
        if server_now is not None:
            server_now = _sane_time(server_now)
        with self._locked():
            d = self._load(now)
            now, rolled = self._now(d, now)
            rollback = GrantError("this computer's clock is behind the last time PXA Control saw", "clock_rollback")
            try:
                st, _status = verify_grant(env, self.pubkeys, now, pub, issued_ref=server_now if not rolled else None)
            except GrantError:
                if rolled:
                    raise rollback
                raise
            if st["ent_id"] in d["revoked"]:
                raise GrantError("this grant was revoked", "revoked")
            prev = int(d["newest"].get(st["ent_id"], 0))
            if int(st["issued"]) < prev:
                raise GrantError("an older copy of a grant already renewed (replay)", "replayed")
            if rolled and int(st["issued"]) <= prev:
                raise rollback                       # nothing newer from licd: no reset, the clock is still behind
            if d.get("hard_lock_at") is not None:
                # while a 403 (or corruption) hard-lock is in force, only a grant licd issued after it, under a grant id
                # (and nonce) never seen before it, counts (docs/ANIBAL-HARD-LOCK-SPEC.md): an old grant never unlocks
                if not self._fresh(st, d["hard_lock_at"], d.get(J_LOCK_NEWEST), set(d.get("hard_lock_seen") or ())):
                    raise GrantError("a grant from before the account was locked", "pre_lock")
            d["grants"][st["ent_id"]] = env
            d["newest"][st["ent_id"]] = int(st["issued"])
            if server_now is not None:
                d.setdefault("licd_time", {})[st["ent_id"]] = float(server_now)
                d[J_LICD_SEEN] = max(float(d.get(J_LICD_SEEN) or 0), min(float(server_now), now + LOCK_AHEAD_CAP_S))
            elif isinstance(d.get("licd_time"), dict):
                d["licd_time"].pop(st["ent_id"], None)
            if int(st["issued"]) > prev:              # F310 (b): after this reply's licd time is on record (the floor)
                self._seen_back(d, float(st["issued"]), st["ent_id"])
            if rolled and now < float(d.get("seen") or 0) - ROLLBACK_S:
                raise rollback                       # licd's signed time does not reach this clock: nothing changes
            d["locks"].pop(st["item_id"], None)       # a new grant licd signed for the item is licd saying it is active again
            if isinstance(d.get("revoked_items"), dict):
                d["revoked_items"].pop(st["item_id"], None)
            if d.get("hard_lock_at") is not None:
                d["hard_locks"].pop(st["item_id"], None)   # a 403 / corruption hard-lock lifts only this way, item by item
                if "*" in d["hard_locks"]:
                    d["fresh"] = sorted(set(d.get("fresh") or ()) | {st["item_id"]})
                if not d["hard_locks"]:
                    for k in ("hard_lock_at", "hard_lock_seen", "hard_lock_clock", "fresh", "corrupt_copy", "corrupt_why",
                              "star_held", "hard_lock_why", "hard_lock_source", "hard_lock_capped", J_LOCK_NEWEST):
                        d.pop(k, None)
            self._save(d)
        return st

    @staticmethod
    def _seen_back(d, signed, ent_id=None):
        """F310 (b): a verified licd grant (signed `issued`, strictly newer than newest[ent_id]: checked by add) moves a
        last-seen that is ahead of licd back to target = max(its issued, licd_high_water (file and journal, merged), every
        other grant's newest issued), only when seen > target + CLOCK_SKEW_S. Written as is under the store lock (the
        process-only _seen_reset tells _merge / _save not to max it with the journal's)."""
        floor = max([_num(d.get(J_LICD_SEEN)) or 0.0]
                    + [float(v) for e, v in (d.get("newest") or {}).items() if e != ent_id and _num(v) is not None])
        target = max(float(signed), floor)
        if float(d.get("seen") or 0) > target + CLOCK_SKEW_S:
            d["seen"], d["_seen_reset"] = target, True

    def set_locks(self, locks, replace=False):
        """{item_id: reason} from licd's signed account answer (owned[].status != active, spec s4 hard-lock). A locked item
        counts for nothing until licd grants it again; the plugin host writes the same "locks" key in this file."""
        now = self.clock()
        with self._locked():
            d = self._load(now)
            self._now(d, now)                      # every write records the latest time seen (rollback guard)
            d["locks"] = dict(locks) if replace else dict(d["locks"], **locks)
            self._save(d)

    def items(self):
        """every item this computer holds or held (grants, locks, hard-locks): what a 403 hard-lock covers."""
        now = self.clock()
        with self._locked(lock_only=True):
            d = self._load(now)
        held = {(_statement(env) or {}).get("item_id") for env in d["grants"].values()}
        return sorted(x for x in held | set(d["locks"]) | set(d["hard_locks"]) if isinstance(x, str) and x != "*")

    def hard_lock(self, code, items=(), at=None, source=None):
        """licd answered 403 revoked | not_linked | account_disabled (spec s4, Core's gate b0115812b6, Plugins 03294342ea):
        every item locks and every grant is dropped. The locks go in "locks" (what Core and the plugin host read) and in
        "hard_locks", which only a fresh grant for that item clears (add()); a later good sync that rewrites "locks" from
        owned[] does not lift them. Revocation and anti-replay records stay, so an old copy of a grant cannot come back.
        at: licd's time of the 403 (server_time in its body, else its Date header), compared with grants' server-issued
        `issued`; None -> the local clock minus LOCAL_LOCK_SLACK_S. hard_lock_clock records which (server | local),
        hard_lock_source which reply field (body | date | local; server_time_checked picks the EARLIER when they disagree
        by more than 5 min, F310). Any lock time is capped at the local clock + LOCK_AHEAD_CAP_S (hard_lock_capped).
        The whole load-modify-save runs under the store lock, so a concurrent add() cannot write the pre-lock file back."""
        code = str(code)[:64]
        local = float(self.clock())

        def change(d):
            self._now(d, local)                      # every write records the latest time seen (rollback guard)
            held = {(_statement(env) or {}).get("item_id") for env in d["grants"].values()}
            every = sorted(x[:64] for x in (held | set(d["locks"]) | set(d["hard_locks"]) | set(d.get("revoked_items") or {})
                                            | set(d.get("star_held") or ())      # held before a "*" rebuild too
                                            | set(items or ())) - {"*"} if isinstance(x, str) and x)
            for i in every:
                d["locks"][i] = code
                d["hard_locks"][i] = code
            # v4: every grant id known at lock time and the newest issued per ent_id then (J_LOCK_NEWEST, also journalled):
            # a known id unlocks only with a strictly newer issue after the lock time; a new id (relink) after it
            held_issued = {e: _issued(env) for e, env in d["grants"].items() if _issued(env) is not None}
            d[J_LOCK_NEWEST] = self._max_merge(d.get(J_LOCK_NEWEST), d["newest"], held_issued)
            d["hard_lock_seen"] = sorted(set(d.get("hard_lock_seen") or ()) | self._ids(d))
            lat = _sane_time(at) if at is not None else None
            if lat is not None:                                 # licd's time, capped at now + 24 h (= Core)
                d[J_LICD_SEEN] = max(float(d.get(J_LICD_SEEN) or 0), min(float(lat), local + LOCK_AHEAD_CAP_S))
            new_at, src = (float(lat), "server") if lat is not None else (local - LOCAL_LOCK_SLACK_S, "local")
            capped = new_at > local + LOCK_AHEAD_CAP_S
            if capped:                                          # F310: never later than this computer's clock + 24 h
                new_at = local + LOCK_AHEAD_CAP_S
            if new_at > float(d.get("hard_lock_at") or 0):      # a second 403 moves the lock time later, never earlier
                d["hard_lock_at"], d["hard_lock_clock"] = new_at, src
                d["hard_lock_source"] = _lock_source(source or "body") if lat is not None else "local"
                if capped:
                    d["hard_lock_capped"] = True
                else:
                    d.pop("hard_lock_capped", None)
            d["hard_lock"] = code
            d["grants"] = {}
            return every, {"items": {i: code for i in every}, "at": d.get("hard_lock_at"),
                           "clock": d.get("hard_lock_clock"), "seen_ids": list(d.get("hard_lock_seen") or ()),
                           "lock_newest": dict(d.get(J_LOCK_NEWEST) or {}), "hw": d.get(J_LICD_SEEN), "code": code,
                           "source": d.get("hard_lock_source")}
        # the whole load-modify-save runs under the store lock; busy past LOCK_TIMEOUT_S -> the journal first, then a
        # second wait, then StoreBusy (F311: grants.json is never written without the lock)
        every = self._adding_write(local, change)
        return every

    def drop_item(self, item_id):
        """forget every grant for one item (a hard-lock). Revocation and anti-replay records stay."""
        now = self.clock()
        with self._locked(lock_only=True) as busy:
            if busy:                               # F311: never written without the lock (the lock itself is set_locks')
                raise StoreBusy("the store file is busy (another PXA process holds its lock)")
            d = self._load(now)
            self._now(d, now)                      # every write records the latest time seen (rollback guard)
            for e, env in list(d["grants"].items()):
                if (_statement(env) or {}).get("item_id") == item_id:
                    d["grants"].pop(e, None)
            self._save(d)

    def locks(self):
        """{item_id: reason} of every lock in force (owned[] status, the plugin host's, and 403 / corruption hard-locks;
        "*" = every item without a fresh grant)."""
        now = self.clock()
        with self._locked(lock_only=True):
            d = self._load(now)
        return dict(d["locks"], **d["hard_locks"])

    def status(self):
        """the cache's own state for the entitlements view: {state: ok | hard_locked | corrupt, hard_lock: code | None,
        corrupt_copy: file name | None, since: lock time | None, clock: server | local | None,
        why: corrupt | link | missing | None}."""
        now = self.clock()
        with self._locked(lock_only=True):
            d = self._load(now)
        hl = d.get("hard_locks") or {}
        state = "corrupt" if hl.get("*") == "corrupt" else ("hard_locked" if hl else "ok")
        return {"state": state, "hard_lock": d.get("hard_lock") if hl else None,
                "corrupt_copy": d.get("corrupt_copy") if state == "corrupt" else None,
                "since": d.get("hard_lock_at") if hl else None, "clock": d.get("hard_lock_clock") if hl else None,
                "why": d.get("corrupt_why", "corrupt") if state == "corrupt" else None}

    def apply_revocations(self, env):
        st = verify_revocations(env, self.pubkeys)
        now = self.clock()

        def change(d):
            self._now(d, now)                      # every write records the latest time seen (rollback guard)
            d["revoked"] = sorted({x for x in d["revoked"] if isinstance(x, str)} | {str(x) for x in st["ent_ids"]})[-20000:]
            for e in list(d["grants"]):
                if e in d["revoked"]:
                    env2 = d["grants"].pop(e, None) or {}
                    item = (_statement(env2) or {}).get("item_id")
                    ri = d.setdefault("revoked_items", {}) if item else None   # = Plugins: {item_id: ent_id}, made when
                    if isinstance(ri, dict):              # absent (the Plugins tab says "no longer unlocked", not "locked")
                        ri[item] = e                      # the plugin host's view keeps the item `revoked`
            return len(d["revoked"]), {"revoked": sorted({str(x) for x in st["ent_ids"]})}
        return self._adding_write(now, change)    # F311: busy -> the journal first, StoreBusy if it stays busy

    def active(self):
        """[(statement, status)] of every grant that counts now (status ok | refresh_due). A grant licd issued ahead of
        the local clock still counts once accepted on licd's clock (licd_time: licd's time when it was accepted)."""
        _seed, pub = install_key(self.cfg_dir)
        out = []
        now = self.clock()
        with self._locked(lock_only=True):
            d = self._load(now)                         # reads never write (F303 contract rule 3); writes move `seen`
            now, rolled = self._now(d, now)
        if rolled:
            return []
        revoked = set(x for x in d["revoked"] if isinstance(x, str))
        lt = d.get("licd_time") or {}
        for ent_id, env in sorted(d["grants"].items()):
            if ent_id in revoked:
                continue
            try:
                ref = lt.get(ent_id)
                st, status = verify_grant(env, self.pubkeys, now, pub,
                                          issued_ref=max(now, float(ref)) if isinstance(ref, (int, float)) else None)
            except GrantError:
                continue
            if st["ent_id"] in revoked:     # the signed id decides, not the file's key (a hand-edited key never beats a revoke)
                continue
            item = st.get("item_id")
            if self._hard_locked(d, item) or item in d["locks"]:   # a locked item counts for nothing
                continue
            out.append((st, status))
        return out

    def has(self, item_id, kind=None):
        return any(st["item_id"] == item_id and (kind is None or st["kind"] == kind) for st, _s in self.active())

    def record(self):
        """the store record as it reads now (grants.json merged with the lock journal), under the store lock, for the
        gate's reasons and the Plugins tab. Read-only for the caller: what runs is active(), never this."""
        now = self.clock()
        with self._locked(lock_only=True):
            return self._load(now)

    def account_lock_lifted(self, d, now=None):
        """review 10-26 L1: True when licd has signed this computer a fresh grant since the 403 / corruption lock
        in `d` (a record() snapshot) - the evidence v5 already takes to lift an item's lock (add() / _fresh, spec
        docs/ANIBAL-HARD-LOCK-SPEC.md): verified for this machine with a known key, not revoked by key or signed id,
        issued after hard_lock_at and a new ent_id or strictly newer than its lock-time newest. Then the account-level
        reason (hard_locked(), sync.locked, the gate's per-item 403 reason) is over: licd serves this computer again.
        Items that got no such grant stay in hard_locks and never run; only the words they get change (gate.state).
        An unsigned, stale or replayed reply carries no fresh grant, and a later 403 drops every grant and moves the
        lock time, so neither lifts it. Reads only; active() (what runs) never calls this."""
        at = _num(d.get("hard_lock_at")) if isinstance(d, dict) else None
        if at is None or not (d.get("hard_locks") or {}):
            return False
        now = float(self.clock() if now is None else now)
        if now < float(_num(d.get("seen")) or 0) - ROLLBACK_S:
            return False                                 # the clock went back: nothing is judged on it
        revoked = {x for x in d.get("revoked") or () if isinstance(x, str)}
        known = {x for x in d.get("hard_lock_seen") or () if isinstance(x, str)}
        ln = d.get(J_LOCK_NEWEST) if isinstance(d.get(J_LOCK_NEWEST), dict) else None
        items = set()
        for e, env in (d.get("grants") or {}).items():
            st = _statement(env)
            if st and e not in revoked and st.get("ent_id") not in revoked and isinstance(st.get("item_id"), str):
                items.add(st["item_id"])
        live = dict(d, grants={e: env for e, env in (d.get("grants") or {}).items()
                               if e not in revoked and (_statement(env) or {}).get("ent_id") not in revoked})
        return any(self._lifted(live, i, float(at), ln, known, now) for i in sorted(items))

    def hard_locked(self):
        """-> {"code", "at"} while a 403 / corruption hard-lock is in force on the account (hard_locks, = Core / Plugins),
        else None. Only a fresh grant licd signs lifts it (add() item by item; account_lock_lifted for the account-level
        reason); a good sync alone never does."""
        d = self.record()
        hl = d.get("hard_locks") or {}
        if not hl or self.account_lock_lifted(d):
            return None
        code = d.get("hard_lock") if isinstance(d.get("hard_lock"), str) and d.get("hard_lock") else \
            ("corrupt" if "*" in hl else str(sorted(hl.values())[0]))
        return {"code": str(code)[:64], "at": int(_num(d.get("hard_lock_at")) or 0)}


# ---- the two routes the page's store.js adapter calls (fixture catalog until licd serves it) ------------------
# Shapes = the ones store.js renders (store.js header comment), so the page needs no
# second adapter: catalog {version, updated, currency, items[]}, entitlements {account, credits, items{}, grants,
# receipts, sync}. account / credits / receipts come from licd (owner-run); locally they are null / [] .
def _perm(pid):
    from .. import plugins as _plugins
    return {"id": pid, "group": pid.split(".")[0], "label": _plugins.PERMISSIONS.get(pid, pid)}


CATALOG_COST_KEY = "price"     # the live catalogue API field for an item's coins (never shown to the user as a word)
CATALOG_FIXTURE = [
    {"id": "anibal", "kind": "plugin", "name": "Anibal", "tagline": "Voice assistant for the models you run here",
     "description": "First-party plugin; runs in its own process with only the permissions you allow.", "price": 10,
     "version": "0.1.0", "perm_version": 1, "permissions": ["device.mic", "device.speaker", "model.chat", "net.loopback"]},
    {"id": "dictate", "kind": "plugin", "name": "Dictate",
     "tagline": "Hold a key, speak, and text appears on this computer",
     "description": "Unlocks with PXA coins. Local speech-to-text on this computer. Speech stays on this computer.",
     "price": 100, "version": "0.2.0", "perm_version": 2,
     "permissions": ["device.mic", "speech.listen", "input.type"]},
    {"id": "pxqn", "kind": "quant-tier", "name": "PXQN quants", "tagline": "The supporter quant tier (PXQ stays free)",
     "description": "PXQN files and the Pro encoder's PXQN tiers, delivered as signed downloads bound to your grant.",
     "price": 10, "level": "pxqn4", "permissions": []},
    {"id": "overdrive-presets", "kind": "preset", "name": "Overdrive presets", "tagline": "Measured lever presets per card class",
     "description": "Settings, not code: the presets PXA measured for your cards, applied at launch.", "price": 5,
     "version": "2026.10", "grants": ["overdrive-presets"], "permissions": []},
    {"id": "fleet-view", "kind": "feature", "name": "Fleet view", "tagline": "Several rigs on one page",
     "description": "Not available yet.", "price": 5, "grants": ["fleet-view"], "permissions": ["net.loopback"]},
]


def catalog_view():
    """GET /api/store/catalog -> {version, updated, source, currency{name, source, rate_note}, items[{id, kind, name,
    tagline, description, price, version?, level?, grants?, perm_version?, permissions[{id, group, label}]}]}
    ("price" is the live API field name for the credits an item needs; it is never shown to the user as a word:
    the store is donation-based and the UI says "Free" or "N credits".)"""
    items = []
    for x in CATALOG_FIXTURE:
        it = dict(x)
        it["permissions"] = [_perm(pid) for pid in x.get("permissions") or []]
        items.append(it)
    return {"version": 1, "updated": "2026-10-14T00:00:00Z", "source": "fixture",
            "currency": {"name": "credits", "source": "https://ko-fi.com/shatteredrealms1",
                         "rate_note": "Fixture: the store backend (licd) sets how many credits a donation earns."},
            "items": items}


def entitlements_view(cache, snap=None):
    """GET /api/store/entitlements -> {account, credits, items: {<item_id>: {state, since, expires, kind, level, grant,
    status, refresh_after}}, grants: [feature/preset ids], receipts, sync: {online, checked, grace_days, grace_until},
    machine}. account / credits / receipts come from licd's last answer (`snap`, <config>/store/account.json); without
    licd they are null / null / []. machine = a short prefix of the public install key, never the private key."""
    _seed, pub = install_key(cache.cfg_dir)
    items, grants, until = {}, [], []
    for st, status in cache.active():
        items[st["item_id"]] = {"state": "owned", "since": int(st["issued"]), "expires": int(st["not_after"]),
                                "kind": st["kind"], "level": st.get("level"), "grant": st["ent_id"][:12],
                                "status": status, "refresh_after": int(st["refresh_after"])}
        if st["kind"] in ("preset", "feature"):
            grants.append(st["item_id"])
        until.append(int(st["not_after"]))
    snap = snap or {}
    cs = cache.status()          # the cache file's own state: ok | hard_locked | corrupt (F305-P: fail closed, kept aside)
    out = {"account": snap.get("account"), "credits": snap.get("credits"), "items": items, "grants": sorted(grants),
           "receipts": snap.get("receipts") or [], "locks": cache.locks(),
           "hard_lock": snap.get("hard_lock") or ("corrupt" if cs["state"] == "corrupt" else None), "cache": cs,
           "sync": {"online": bool(snap.get("ok")), "checked": snap.get("checked"), "grace_days": 180,
                    "grace_until": min(until) if until else None},
           "machine": pub[:16]}
    locked = cache.hard_locked()
    if locked:                                  # licd's 403 at the last sync (S6): the page says why everything is off
        out["sync"]["locked"] = locked.get("code")
    return out


# ---- licd, the real store backend (licd tools/pxa-licd/store.py; PLUGINS-STORE-DESIGN.md sections 5-6) -----------
INSTALL_PREFIX = b"pxa-plugin-install|"
CATALOG_PREFIX = b"pxa-store-catalog|"
STMT_PREFIX = b"pxa-wallet-stmt|"          # licd's signed wallet statement (balance split + owned items)
DEFAULT_STORE_URL = "https://lic.pxanetwork.com"   # licd is live (2026-10-14); PXA_STORE_URL / control.json store_url override, "" = fixture
SYNC_EVERY_S = 12 * 3600    # an online Control refreshes its grants twice a day (grants say refresh_after = 24 h)
SYNC_JITTER_S = 3600        # background sync (plan S6): every 12 h +- 1 h, so a fleet of Controls never hits licd at once
SYNC_TICK_S = 60            # the background loop looks at the wall clock this often
WAKE_GAP_S = 300            # a tick that took this much longer than SYNC_TICK_S = the computer slept: sync on wake
SYNC_RETRY_S = 3600         # offline (network, 5xx, 429): try again in an hour; offline never locks anything
SYNC_START_S = 120          # first background sync this long after Control starts (a start is a wake too)
# HARD_LOCK_CODES (licd's refresh answers that lock every item at once) is defined once, at the top (spec s4)
_SYNC_LISTENERS = []        # fn(cfg_dir, result) after every sync answer (the plugin host stops what lost its grant)
_BG = [None]                # the running BackgroundSync (kick() after a redeem whose inline sync failed)


def canonical_ascii(obj):
    """licd's canon(): sorted keys, no spaces, ASCII escapes. Equal to canonical() for ASCII-only statements (grants,
    revocations); the catalogue can carry non-ASCII text, so it is verified with this form."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def load_pubkeys(path=None):
    """the public keys a release embeds: tools/pxa_control/store/pubkeys.json, written by `pxa-lic keys init` (public
    halves only). Fills GRANT_PUBKEYS and plugins.PLUGIN_PUBKEYS. PXA_STORE_PUBKEYS names another file only where no
    pubkeys.json is embedded (a dev build): it never replaces the keys a shipped Control pins (review F7)."""
    embedded = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pubkeys.json")
    p = path or (os.environ.get("PXA_STORE_PUBKEYS") if not os.path.isfile(embedded) else None) or embedded
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    ok = lambda m: {str(k): str(v).lower() for k, v in (m or {}).items() if re.fullmatch(r"[0-9a-fA-F]{64}", str(v))}
    GRANT_PUBKEYS.update(ok(d.get("GRANT_PUBKEYS")))
    try:
        from .. import plugins as _plugins
        _plugins.PLUGIN_PUBKEYS.update(ok(d.get("PLUGIN_PUBKEYS")))
    except Exception:  # noqa: BLE001 - the plugin keys are optional here
        pass
    return d


def store_url():
    """licd's origin: PXA_STORE_URL, else control.json "store_url", else the release default ("" = fixture mode)."""
    v = os.environ.get("PXA_STORE_URL")
    if v is None:
        try:
            from .. import core as _core
            v = _core.load_config().get("store_url")
        except Exception:  # noqa: BLE001
            v = None
    return str(v if v is not None else DEFAULT_STORE_URL).rstrip("/")


_LOOPBACK = ("127.0.0.1", "localhost", "::1")
MAX_REPLY = 8 << 20                # a licd JSON answer is a few KB; anything this big is not one


def check_url(url):
    """spec s2: the store is spoken to over https only; plain http only on loopback (tests, a local licd). The same rule
    as Core's gate (tools/vigil/entitlement._check_url) and the plugin host (pxa_plugins.store_backend.check_url).
    -> the url without a trailing slash; core.Invalid 400 bad_url otherwise."""
    import urllib.parse
    u = str(url or "").strip().rstrip("/")
    try:
        p = urllib.parse.urlsplit(u)
        host = p.hostname
    except ValueError:
        host, p = None, None
    if not host or p.scheme not in ("https", "http") or (p.scheme == "http" and host not in _LOOPBACK):
        raise _invalid("the store address must be https (plain http only on this computer)", 400, "bad_url")
    return u


def store_auth(cfg_dir, op, nonce, args):
    """the machine signature licd checks on every store call: install key over prefix + canon({op, nonce, args})."""
    seed, pub = install_key(cfg_dir)
    return {"install_pub": pub, "nonce": str(nonce),
            "sig": b64u(ed25519_sign(seed, INSTALL_PREFIX + canonical_ascii({"op": op, "nonce": str(nonce), "args": args})))}


def _sane_time(t):
    try:
        return t if isinstance(t, (int, float)) and not isinstance(t, bool) and 1.5e9 < t < 4.1e9 else None
    except (OverflowError, TypeError):
        return None


def server_time(body, headers):
    """licd's clock from a reply: the JSON body's server_time (epoch seconds, or an ISO 8601 UTC string) first, else
    the HTTP Date header. -> epoch seconds, or None when there is none or it does not parse."""
    import email.utils
    from datetime import datetime, timezone
    t = (body or {}).get("server_time") if isinstance(body, dict) else None
    if isinstance(t, str):
        try:
            dt = datetime.fromisoformat(t.strip().replace("Z", "+00:00"))
            t = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
        except (ValueError, OverflowError, OSError, TypeError):
            t = None
    t = _sane_time(t)
    if t is not None:
        return float(t)
    try:
        d = (headers or {}).get("Date")
        dt = email.utils.parsedate_to_datetime(str(d)) if d else None
    except (TypeError, ValueError, IndexError, OverflowError):
        dt = None
    if dt is None:
        return None
    try:
        t = _sane_time((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp())
    except (ValueError, OverflowError, OSError):
        t = None
    return float(t) if t is not None else None


def server_time_checked(body, headers):
    """F310 (= Plugins b5507ccf83): licd's clock with a cross-check. -> (epoch seconds or None, "body" | "date" | None).
    The body's server_time wins when the reply's Date agrees within BODY_DATE_WINDOW_S or is missing; when both parse
    and disagree by more, the EARLIER of the two (a body years ahead would otherwise lock for good)."""
    b = server_time(body, {})
    dt = server_time({}, headers)
    if b is not None and dt is not None and abs(b - dt) > BODY_DATE_WINDOW_S:
        return (b, "body") if b < dt else (dt, "date")
    if b is not None:
        return b, "body"
    return (dt, "date") if dt is not None else (None, None)


def _invalid(msg, status=400, code="invalid"):
    from .. import core as _core
    return _core.Invalid(msg, status=status, code=code)


def _user_agent():
    try:
        from .. import core as _core
        v = re.sub(r"[^A-Za-z0-9._+-]", "", str(_core.CONTROL_VERSION or ""))[:40]
    except Exception:  # noqa: BLE001 - the version is decoration
        v = ""
    return "PXA-Control/" + (v or "dev")


WALLET_CHANGES = [0]      # bumped by every wallet action that can change the wallet (create, join, redeem, beta, Ko-fi link...)
VIEW_CACHE_S = 20         # review R1: one licd wallet read per this many seconds for all tabs (GET /api/store/wallet)
_VIEW = {}
_VIEW_LOCK = threading.Lock()


class LicdClient(object):
    """talks to licd's /v1/store/* (stdlib urllib). Errors become core.Invalid with licd's status and code, so a
    Control route passes "not enough coins" (402 not_enough_coins) straight to the page."""

    def __init__(self, base, cfg_dir, pubkeys=None, timeout=20):
        self.base, self.cfg_dir, self.timeout = check_url(base), cfg_dir, timeout
        self.pubkeys = GRANT_PUBKEYS if pubkeys is None else pubkeys

    def _req(self, method, path, body=None, headers=None, raw=False):
        """-> licd's JSON object (raw=True: the open response, for a download). core.Invalid otherwise, with the code the
        page and sync() branch on: licd's own code + status for a 4xx/5xx it answered; "store_offline" (503) for anything
        that is not an HTTP answer at all (network, DNS, TLS, a garbled or cut-off reply: http.client.HTTPException);
        "store_malformed" (503) for a 2xx whose body is not one JSON object. Neither of the last two changes anything."""
        import http.client
        import urllib.error
        import urllib.request
        url = check_url(path) if path.startswith("http") else self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
        if data is not None:
            req.add_header("Content-Type", "application/json")
        # lic.pxanetwork.com sits behind Cloudflare, which answers urllib's default "Python-urllib/x.y" with 403
        req.add_header("User-Agent", _user_agent())
        try:
            r = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            try:
                j = json.loads(e.read(65536) or b"{}")
            except (ValueError, OSError, http.client.HTTPException):
                j = {}
            finally:
                e.close()
            j = j if isinstance(j, dict) else {}
            msg = j.get("error") if isinstance(j.get("error"), str) else "HTTP %d" % e.code
            code = j.get("code") if isinstance(j.get("code"), str) and j.get("code") else "store_error"
            x = _invalid("store: " + msg[:300], e.code if 400 <= e.code < 600 else 502, code[:64])
            x.server_time, x.server_source = server_time_checked(j, e.headers)   # licd's clock: a 403's lock time
            ra = str((e.headers or {}).get("Retry-After") or "").strip()
            x.retry_after = int(ra) if ra.isdigit() else None      # licd's limiter says when the window frees (a 429 or a busy 503)
            raise x
        except (OSError, ValueError, http.client.HTTPException) as e:     # incl. a bad status line, a cut-off or garbled reply
            raise _invalid("the store is not reachable (%s)" % e.__class__.__name__, 503, "store_offline")
        if raw:
            return r
        try:
            with r:
                data = r.read(MAX_REPLY + 1)
                short = len(data) <= MAX_REPLY and bool(getattr(r, "length", None))   # fewer bytes than Content-Length said
        except (OSError, ValueError, http.client.HTTPException) as e:     # IncompleteRead mid-body: offline, never a change
            raise _invalid("the store is not reachable (%s)" % e.__class__.__name__, 503, "store_offline")
        if len(data) > MAX_REPLY:
            raise _invalid("the store's reply is too large", 503, "store_malformed")
        if short:                                       # a cut-off reply is a network failure, not a reply to read
            raise _invalid("the store is not reachable (IncompleteRead)", 503, "store_offline")
        try:
            j = json.loads(data or b"{}")
        except ValueError:
            j = None
        if not isinstance(j, dict):
            raise _invalid("the store's reply is not readable", 503, "store_malformed")
        j = _Reply(j)
        j.server_time, j.server_source = server_time_checked(j, getattr(r, "headers", None))   # the not-yet check
        return j

    def catalog(self):
        j = self._req("GET", "/v1/store/catalog")
        st = verify_catalog(j, self.pubkeys)
        self.signed_catalog = {"catalog": st, "signature": j.get("signature")}   # kept as signed, re-verified on read (F12)
        return st

    def revocations(self):
        return self._req("GET", "/v1/store/revocations")

    def manifest(self, item_id):
        if not _ITEM_RE.match(str(item_id or "")):
            raise _invalid("bad item id")
        return self._req("GET", "/v1/store/manifest/" + item_id)

    def _nonce(self, pub, op):
        n = self._req("POST", "/v1/store/nonce", {"install_pub": pub, "op": op}).get("nonce")
        if not isinstance(n, str) or not n:
            raise _invalid("the store's reply is not readable", 503, "store_malformed")
        return n

    def call(self, op, args=None, extra=None):
        args = dict(args or {})
        _seed, pub = install_key(self.cfg_dir)
        n = self._nonce(pub, op)
        body = dict(args)
        body.update(store_auth(self.cfg_dir, op, n, args))
        body.update(extra or {})
        return self._req("POST", "/v1/store/" + op, body)

    WALLET_ACTIONS = {"view": (), "create": (), "statement": (), "code": (), "join": ("code",), "kofi-email": ("email",),
                      "kofi-confirm": ("code",), "redeem": ("item_id",), "beta": (), "links": (),
                      "email-join-start": ("email",), "email-join": ("email", "code"),   # a new computer joins the wallet its proven email holds
                      "report-link": (),                  # a one-time code that names this wallet on a GitHub or Discord report (DESIGN.md 7)
                      "claim": ("code",),                 # PXA first: the private code from the Ko-fi email; a computer without a wallet gets one
                      "gift-start": ("to", "coins"),      # gift coins to another wallet : a preview + a 10-minute code ...
                      "gift-confirm": ("gift",)}          # ... and the transfer, after the person confirmed the preview

    def earn_table(self):
        """GET /v1/wallet/earn-table (public, unsigned): what a person can earn and how many coins, as licd's rules say now."""
        j = self._req("GET", "/v1/wallet/earn-table")
        rows = j.get("rows") if isinstance(j, dict) else None
        if not isinstance(rows, list):
            raise _invalid("the store sent no earn table", 502, "store_error")
        return [{"what": str(r.get("what") or ""), "coins": r.get("coins") if isinstance(r.get("coins"), int) else str(r.get("coins") or ""),
                 "note": str(r.get("note") or "")} for r in rows if isinstance(r, dict) and r.get("what")]

    def wallet(self, action, args=None):
        """the PXA Coins wallet this computer holds: POST /v1/wallet/<action>, signed like every store call (op 'wallet-<action>').
        Only the fields licd signs for that action are sent."""
        if action not in self.WALLET_ACTIONS:
            raise _invalid("unknown wallet action", 404, "not_found")
        a = {k: str((args or {}).get(k) or "").strip() for k in self.WALLET_ACTIONS[action]}
        op = "wallet-" + action
        _seed, pub = install_key(self.cfg_dir)
        n = self._nonce(pub, op)
        body = dict(a)
        body.update(store_auth(self.cfg_dir, op, n, a))
        r = self._req("POST", "/v1/wallet/" + action, body)
        if action not in ("view", "statement", "links"):
            WALLET_CHANGES[0] += 1          # the Beta card's cached view (routes.py) is stale after a change
        return r

    def sync(self, cache):
        """refresh every grant for this computer, apply the signed revocation list, lock what licd's account answer says is
        no longer active, keep that answer. The spec readings Core's gate and the plugin host share (anibal/a-store-gate
        fixtures/entitlement/PROVENANCE.md): a reply whose grants is not a list, whose account.owned is not a list, or
        with an owned[] entry without a string item_id AND a string status is MALFORMED: nothing in it is applied (not even
        its revocations) and it raises store_malformed (503, like offline). Only a present status other than "active"
        locks (s4 hard-lock); a lock drops that item's grants until licd grants it again."""
        try:
            r = self.call("refresh")
        except Exception as e:  # noqa: BLE001 - core.Invalid; anything else is re-raised unchanged
            if getattr(e, "status", None) == 403 and getattr(e, "code", None) in HARD_LOCK_CODES:
                return self._hard_lock(cache, e.code, getattr(e, "server_time", None), getattr(e, "server_source", None))
            raise                                       # offline / 5xx / 429 / malformed / any other 4xx: never a lock
        grants, acct = r.get("grants"), r.get("account")
        owned = acct.get("owned") if isinstance(acct, dict) else None
        if not isinstance(grants, list) or not isinstance(owned, list) or any(
                not isinstance(o, dict) or not isinstance(o.get("item_id"), str) or not isinstance(o.get("status"), str)
                for o in owned):
            raise _invalid("the store's reply is not readable; nothing was changed", 503, "store_malformed")
        bad = []
        if r.get("revocations") is not None:           # revocations first: a grant revoked in this reply never lands
            try:
                cache.apply_revocations(r["revocations"])
            except GrantError as e:
                bad.append("revocations:" + e.code)
        t = getattr(r, "server_time", None)            # licd's clock in this reply (body server_time, else Date)
        for g in grants:
            try:
                cache.add(g, server_now=t)
            except GrantError as e:
                bad.append(e.code)
        locks = {o["item_id"][:64]: o["status"][:64] for o in owned if o["status"] != "active"}
        cache.set_locks(locks, replace=True)
        for i in locks:
            cache.drop_item(i)
        save_snapshot(self.cfg_dir, acct, ok=True, hard_lock=None)
        out = {"grants": len(grants), "refused": bad, "locked": sorted(locks), "credits": acct.get("credits")}
        _notify(self.cfg_dir, dict(out, ok=True))      # S6: the plugin host rechecks what runs (on_sync)
        return out

    def _hard_lock(self, cache, code, at=None, source=None):
        """403 revoked | not_linked | account_disabled: lock every item this computer holds, held, or the last account
        answer listed, and drop every grant. Only a fresh grant licd signs later unlocks an item."""
        # SEC read NEW-1 (snapshot): store/account.json is a local file anyone can edit; nothing read from it may raise
        # before cache.hard_lock (owned must be a list of {item_id: str}; anything else names no extra items)
        try:
            owned = load_snapshot(self.cfg_dir).get("owned")
            known = [o["item_id"] for o in owned if isinstance(o, dict) and isinstance(o.get("item_id"), str)] \
                if isinstance(owned, list) else []
        except Exception:  # noqa: BLE001 - never lose a 403 to a bad snapshot
            known = []
        items = cache.hard_lock(code, known, at=at, source=source)
        # not linked / disabled: the account view this computer kept is gone too (S6); revoked keeps it. Written after
        # cache.hard_lock, which already read owned[] above
        gone = {"account": None, "credits": None, "receipts": [], "owned": [], "machines": None} if code != "revoked" else None
        try:
            save_snapshot(self.cfg_dir, gone, ok=True, hard_lock=code)
        except OSError:
            pass                                        # the lock is in the store; the snapshot is only the page's view
        out = {"grants": 0, "refused": [], "locked": items, "credits": None, "hard_lock": code}
        _notify(self.cfg_dir, {"ok": False, "hard_lock": code, "dropped": len(items)})   # S6: running plugins stop now
        return out


    def download(self, item_id, file_id, dest_dir, progress=None, tries=5, expect=None):
        """a supporter model into dest_dir: a fresh 10-minute token per request, `Range: bytes=<have>-` resume into
        <name>.pxa-part, sha256 of the whole file checked against the token's (this grant's stamped copy), then an
        atomic rename next to the free models. -> {path, size, sha256}"""
        tok, have, err = None, 0, None
        os.makedirs(dest_dir, exist_ok=True)
        for _ in range(tries):
            tok = self.call("download-token", {"item_id": item_id, "file_id": file_id})
            if expect is not None:                           # the signed catalogue's sha + size: a token for other bytes is refused
                if str(tok.get("sha256") or "") != expect["sha256"]:
                    raise _invalid("the Store offered other bytes than its signed catalogue lists", 502, "sha_mismatch")
                if int(tok.get("size") or -1) != int(expect["size"]):
                    raise _invalid("the Store offered another size than its signed catalogue lists", 502, "size_mismatch")
            name = os.path.basename(str(tok.get("name") or ""))
            if not name or name.startswith("."):
                raise _invalid("the store sent a bad file name", 502, "store_error")
            part = os.path.join(dest_dir, name + ".pxa-part")
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > int(tok["size"]):
                os.unlink(part)
                have = 0
            if have == int(tok["size"]):
                break
            # a token URL off this store must be https too (spec s2): refused outright, not retried as a cut connection
            url = check_url(tok["url"]) if str(tok.get("url") or "").startswith("http") else "/v1/store/download/" + tok["token"]
            try:
                r = self._req("GET", url, headers={"Range": "bytes=%d-" % have} if have else None, raw=True)
                with r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                    if r.status != 206:
                        have = 0
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if have > int(tok["size"]):              # never more bytes than announced
                            raise _invalid("the download is longer than announced", 502, "size_mismatch")
                        if progress:
                            progress(have, int(tok["size"]))
                err = None
            except Exception as e:  # noqa: BLE001 - a cut connection: the next pass asks a new token and resumes
                if getattr(e, "code", None) == "size_mismatch":   # more bytes than announced: fatal, never retried (review F4)
                    try:
                        os.unlink(part)
                    except OSError:
                        pass
                    raise
                err = e
            if have == int(tok["size"]):
                break
        if tok is None or have != int(tok["size"]):
            raise _invalid("the download did not finish (%s); try again, it resumes" % (err.__class__.__name__ if err else "short"), 503, "download_incomplete")
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != tok["sha256"]:
            os.unlink(part)
            raise _invalid("the downloaded file does not match its sha256; it was deleted, try again", 502, "sha_mismatch")
        dest = os.path.join(dest_dir, name)
        os.replace(part, dest)
        return {"path": dest, "size": have, "sha256": tok["sha256"]}



def on_sync(fn):
    """fn(cfg_dir, result) after every sync answer: {ok: True, grants, refused, credits} or {ok: False, hard_lock, dropped}."""
    if fn not in _SYNC_LISTENERS:
        _SYNC_LISTENERS.append(fn)
    return fn


def off_sync(fn):
    try:
        _SYNC_LISTENERS.remove(fn)
    except ValueError:
        pass


def _notify(cfg_dir, result):
    for fn in list(_SYNC_LISTENERS):
        try:
            fn(cfg_dir, result)
        except Exception:  # noqa: BLE001 - a listener never breaks a sync
            pass


def kick(reason="redeem"):
    """ask the running background sync to sync now (after a redeem whose inline sync failed); no-op without one."""
    bg = _BG[0]
    if bg is not None:
        bg.kick(reason)
        return True
    return False


class BackgroundSync(object):
    """plan S6 / ANIBAL spec B3: grants refresh on their own, not only when the page asks.
    * timer: every SYNC_EVERY_S +- SYNC_JITTER_S (12 h +- 1 h); any other successful sync (a page refresh, a redeem)
      restarts the count;
    * wake: a loop tick that took WAKE_GAP_S longer than it should = the computer slept -> sync now; a start counts too
      (first sync SYNC_START_S after start);
    * after a redeem: kick("redeem") when the inline sync did not get through;
    * only when there is something to keep fresh (a linked account or a grant on file) and a store URL;
    * 403 revoked / not_linked / account_disabled -> LicdClient.sync hard-locks; offline -> retry in SYNC_RETRY_S, nothing locks.
    clock/rand/client are injectable for tests; step() is one loop turn."""

    def __init__(self, cfg_dir, client=None, cache=None, every=SYNC_EVERY_S, jitter=SYNC_JITTER_S, tick=SYNC_TICK_S,
                 wake_gap=WAKE_GAP_S, retry=SYNC_RETRY_S, start_delay=SYNC_START_S, clock=time.time, rand=random.uniform):
        self.cfg_dir = cfg_dir
        self.client = client or (lambda: LicdClient(store_url(), cfg_dir, timeout=20) if store_url() else None)
        self.cache = cache if cache is not None else EntitlementCache(cfg_dir)
        self.every, self.jitter, self.tick, self.wake_gap, self.retry = every, jitter, tick, wake_gap, retry
        self.clock, self.rand = clock, rand
        self.next_due = clock() + start_delay
        self.last = None
        self._last_tick = None
        self._kicks = []
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._busy = False
        on_sync(self._heard)

    def _schedule(self, now, ok):
        self.next_due = now + (self.every + self.rand(-self.jitter, self.jitter) if ok else self.retry)

    def _heard(self, cfg_dir, result):
        if cfg_dir == self.cfg_dir and not self._busy and (result.get("ok") or result.get("hard_lock")):
            self._schedule(self.clock(), True)      # a page sync or a redeem counts as this turn's sync

    def wanted(self):
        if load_snapshot(self.cfg_dir).get("account") or load_statement(self.cfg_dir):
            return True
        d = self.cache.record()
        return bool(d.get("grants") or d.get("hard_locks"))          # a locked computer keeps asking (relink, re-enable)

    def run_once(self, reason):
        now = self.clock()
        cl = self.client()
        if cl is None or (reason in ("timer", "wake") and not self.wanted()):     # a kick (redeem) always asks
            self._schedule(now, True)
            self.last = {"ok": None, "skipped": True, "reason": reason, "at": int(now)}
            return self.last
        self._busy = True
        try:
            res = cl.sync(self.cache)
            if res.get("hard_lock"):                 # a 403 revoked / not_linked / account_disabled: sync() locked every
                hl = res["hard_lock"]                # item and RETURNS that (it is licd's answer, not a failure)
                res = {"ok": False, "code": hl, "hard_lock": hl, "locked": res.get("locked")}
            else:
                res = dict(res, ok=True)
            ok = True
        except Exception as e:  # noqa: BLE001
            code = str(getattr(e, "code", "") or e.__class__.__name__)
            hard = getattr(e, "status", 0) == 403 and code in HARD_LOCK_CODES
            if not hard:
                save_snapshot(self.cfg_dir, None, ok=False)
            res = {"ok": False, "code": code, "hard_lock": code if hard else None}
            ok = hard                                # licd answered: the normal turn; offline: retry sooner
        finally:
            self._busy = False
        self._schedule(now, ok)
        self.last = dict(res, reason=reason, at=int(now))
        return self.last

    def kick(self, reason="redeem"):
        with self._lock:
            self._kicks.append(reason)
        self._ev.set()

    def step(self):
        """one loop turn -> the sync result, or None when nothing was due."""
        now = self.clock()
        with self._lock:
            kicks, self._kicks = self._kicks, []
        reason = None
        if kicks:
            reason = kicks[0]
        elif self._last_tick is not None and now - self._last_tick > self.tick + self.wake_gap:
            reason = "wake"
        elif now >= self.next_due:
            reason = "timer"
        self._last_tick = now
        return self.run_once(reason) if reason else None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # noqa: BLE001 - the loop outlives any one failure
                pass
            self._ev.wait(self.tick)
            self._ev.clear()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="pxa-store-sync")
            self._thread.start()
            _BG[0] = self
        return self

    def stop(self):
        self._stop.set()
        self._ev.set()
        off_sync(self._heard)
        if _BG[0] is self:
            _BG[0] = None

    def status(self):
        return {"next_due": int(self.next_due), "last": self.last, "hard_lock": self.cache.hard_locked()}


def start_background(cfg_dir, cache=None):
    """PXA Control's serve(): the background sync, unless PXA_STORE_SYNC=0 (off switch) or no store URL is set."""
    if os.environ.get("PXA_STORE_SYNC", "1").strip() == "0" or not store_url():
        return None
    return BackgroundSync(cfg_dir, cache=cache).start()


def _snap_path(cfg_dir):
    return os.path.join(cfg_dir, "store", "account.json")


def load_snapshot(cfg_dir):
    """licd's last account answer, or {} when it is missing, unreadable, not an object or nested too deep to parse
    (RecursionError: a hand-edited file must never stop the 403 path, SEC read NEW-1)."""
    try:
        with open(_snap_path(cfg_dir)) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def save_snapshot(cfg_dir, account_view, ok=True, hard_lock=False):
    """licd's last account answer (credits, receipts, owned items) in <config>/store/account.json (0600).
    hard_lock: a 403 code to record, None to clear it (a good refresh), False to leave it as it is."""
    with file_lock(_snap_path(cfg_dir)):
        return _save_snapshot_locked(cfg_dir, account_view, ok, hard_lock)


def _save_snapshot_locked(cfg_dir, account_view, ok, hard_lock):
    d = load_snapshot(cfg_dir)
    if hard_lock is None:
        d.pop("hard_lock", None)
    elif hard_lock:
        d["hard_lock"] = str(hard_lock)[:64]
    if ok and account_view:
        d.update({"account": account_view.get("account"), "credits": account_view.get("credits"),
                  "receipts": account_view.get("receipts") or [], "owned": account_view.get("owned") or [],
                  "machines": account_view.get("machines")})
    d.update({"ok": bool(ok), "checked": int(time.time()) if ok else d.get("checked")})
    write_atomic(_snap_path(cfg_dir), json.dumps(d).encode())
    return d


def verify_statement(payload):
    """licd's signed wallet statement (prefix pxa-wallet-stmt|). A bad signature never becomes a balance on screen."""
    if not isinstance(payload, dict):
        raise _invalid("the store sent no wallet statement", 502, "bad_signature")
    st = payload.get("statement")
    sig = _b64d(payload.get("signature"))
    kid = str((st or {}).get("kid") or "") if isinstance(st, dict) else ""
    hexkey = GRANT_PUBKEYS.get(kid) or ""
    ok = (isinstance(st, dict) and st.get("type") == "wallet-statement" and st.get("iss") == "licd"
          and re.fullmatch(r"[0-9a-fA-F]{64}", hexkey) and len(sig) == 64
          and ed25519_verify(bytes.fromhex(hexkey), STMT_PREFIX + canonical_ascii(st), sig))
    if not ok:
        raise _invalid("the wallet statement's signature does not verify", 502, "bad_signature")
    return st


def _statement_path(cfg_dir):
    return os.path.join(cfg_dir, "store", "statement.json")


def save_statement(cfg_dir, payload):
    """keep the newest genuine statement (0600) so the wallet can show a balance with the store offline."""
    st = verify_statement(payload)
    path = _statement_path(cfg_dir)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"statement": st, "signature": payload.get("signature")}, f)
    os.replace(tmp, path)
    return st


def load_statement(cfg_dir):
    """the saved statement, or None when there is none or it no longer verifies."""
    try:
        with open(_statement_path(cfg_dir)) as f:
            payload = json.load(f)
        return verify_statement(payload)
    except (OSError, ValueError, GrantError):
        return None
    except Exception as e:  # noqa: BLE001 - a bad file is "no statement", not a crash of the wallet page
        if getattr(e, "code", "") == "bad_signature":
            return None
        raise


def drop_statement(cfg_dir):
    """forget the saved statement (it belonged to another wallet: this install joined a different one)."""
    try:
        os.remove(_statement_path(cfg_dir))
    except OSError:
        pass


STMT_GAP_S = 120.0             # at most one statement read per 2 minutes (a statement read costs two of licd's wallet-read budget)
_STMT = {"at": -1e18}
_STMT_LOCK = threading.Lock()


def _statement_differs(st, view):
    """True when the saved statement no longer matches the live wallet view (other wallet, balance or unlocks moved)."""
    if not st or not isinstance(view, dict):
        return True
    owned = sorted(o.get("item_id") for o in (view.get("owned") or []) if isinstance(o, dict) and o.get("status") not in ("revoked", "removed"))
    return (st.get("wallet") != view.get("wallet_id") or int(st.get("coins") or 0) != int(view.get("coins") or 0)
            or sorted(st.get("owned") or []) != owned)


def keep_statement(cl, cfg_dir, view=None, force=False, now=None):
    """N3: keep the saved signed statement (the offline balance) in step with the wallet this install holds, so a person who
    never pressed "Save a signed statement" still has one, and it is never another wallet's or a stale balance's.
    Best effort: a store that is down leaves the old copy standing (unless it is another wallet's: then it is dropped).
    view = the live wallet view just read, or None after a wallet action (always refresh). Rate-limited by STMT_GAP_S
    unless force. -> True when a fresh statement was written."""
    now = time.monotonic() if now is None else now
    st = load_statement(cfg_dir)
    if view is not None and not force and not _statement_differs(st, view):
        return False
    if view is not None and st and st.get("wallet") != view.get("wallet_id"):
        drop_statement(cfg_dir)                     # another wallet's balance must never stand in for this one
    with _STMT_LOCK:
        if not force and now - _STMT["at"] < STMT_GAP_S:
            return False
        _STMT["at"] = now
    try:
        save_statement(cfg_dir, cl.wallet("statement"))
        return True
    except Exception:  # noqa: BLE001 - the wallet answer stands; the next read tries again
        return False


EARN_TTL_S = 3600.0
_EARN = {"rows": None, "at": -1e18}
_EARN_LOCK = threading.Lock()


def earn_rows(cl, now=None):
    """licd's earn table, read at most once an hour; the last good copy stands while the store is unreachable, None before the first."""
    now = time.monotonic() if now is None else now
    with _EARN_LOCK:
        if _EARN["rows"] is not None and now - _EARN["at"] < EARN_TTL_S:
            return _EARN["rows"]
    try:
        rows = cl.earn_table()
    except Exception:  # noqa: BLE001 - the panel keeps its built-in list
        with _EARN_LOCK:
            return _EARN["rows"]
    with _EARN_LOCK:
        _EARN["rows"], _EARN["at"] = rows, now
    return rows


def wallet_from_statement(st):
    """the offline wallet the page draws: the signed balance split, no history (that needs a live read)."""
    owned = [{"item_id": i, "name": i, "status": "active"} for i in (st.get("owned") or []) if isinstance(i, str)]
    return {"wallet_id": st.get("wallet"), "coins": st.get("coins") or 0,
            "from_donations": st.get("from_donations") or 0, "earned": st.get("earned") or 0,
            "redeemed": st.get("redeemed") or 0, "reversed": st.get("reversed") or 0,
            "adjusted": st.get("adjusted") or 0, "gifts": st.get("gifts") or 0, "owned": owned, "history": [],
            "beta": {"release": None, "access": False, "how": None}}


PLUGIN_PLATFORMS = ("any", "linux-x86_64", "linux-aarch64", "windows-x64", "macos-arm64", "macos-x86_64")
_BUNDLE_VER_RE = re.compile(r"^\d{1,6}\.\d{1,6}\.\d{1,6}$")
_FILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def bundle_of(x):
    """a catalogue item's plugin `bundle` block, shape-checked ({version, kid, min_host, platforms: {p: {file_id, sha256,
    size}}}), or None. The block is covered by the catalogue signature; this refuses a shape no installer should read, whole:
    one bad platform entry, kid or min_host makes the block invalid (review F3), it never drops to a looser path."""
    b = x.get("bundle") if isinstance(x, dict) else None
    if not isinstance(b, dict) or not _BUNDLE_VER_RE.match(str(b.get("version") or "")) or not isinstance(b.get("platforms"), dict):
        return None
    plats = {}
    for k, v in b["platforms"].items():
        if not (k in PLUGIN_PLATFORMS and isinstance(v, dict) and _FILE_ID_RE.match(str(v.get("file_id") or ""))
                and re.fullmatch(r"[0-9a-f]{64}", str(v.get("sha256") or "")) and type(v.get("size")) is int and 0 < v["size"] <= (64 << 20)):
            return None
        plats[k] = {"file_id": v["file_id"], "sha256": v["sha256"], "size": v["size"]}
    if not plats:
        return None
    kid = b.get("kid")
    mh = b.get("min_host")
    if kid is not None and not re.fullmatch(r"[A-Za-z0-9._-]{1,32}", str(kid)):
        return None
    if mh is not None and not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", str(mh)):
        return None
    return {"version": b["version"], "kid": kid or None, "min_host": mh or None, "platforms": plats}


def this_platform():
    import platform as _pf
    mach = _pf.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(mach, mach)
    if sys.platform.startswith("win"):
        return "windows-x64" if arch in ("x64", "x86_64") else "windows-" + arch
    if sys.platform == "darwin":
        return "macos-arm64" if arch == "aarch64" else "macos-x86_64"
    return "linux-" + ("x86_64" if arch in ("x64", "x86_64") else arch)


def pick_platform(bundle):
    """this computer's package from a bundle block: its own platform first, then `any`. -> (platform, row) or (None, None)."""
    for k in (this_platform(), "any"):
        if k in (bundle or {}).get("platforms", {}):
            return k, bundle["platforms"][k]
    return None, None


def host_too_old(min_host, version=None):
    """True when this Control's version is known and older than min_host (a dev tree or an odd tag never blocks)."""
    from .. import core as _core
    v = str(version if version is not None else (_core.CONTROL_VERSION or ""))
    m, n = re.match(r"^v?(\d+)\.(\d+)", v), re.match(r"^(\d+)\.(\d+)", str(min_host or ""))
    if not m or not n:
        return False
    return (int(m.group(1)), int(m.group(2))) < (int(n.group(1)), int(n.group(2)))


def verify_catalog(j, pubkeys=None):
    """{catalog, signature} as licd sends it -> the statement, or GrantError bad_signature (the pinned licd keys only)."""
    keys = GRANT_PUBKEYS if pubkeys is None else pubkeys
    st = j.get("catalog") if isinstance(j, dict) else None
    hexkey = keys.get(str((st or {}).get("kid") or "")) or "" if isinstance(st, dict) else ""
    sig = _b64d(j.get("signature")) if isinstance(j, dict) else b""
    if not isinstance(st, dict) or not re.fullmatch(r"[0-9a-fA-F]{64}", hexkey) or len(sig) != 64 or \
            not ed25519_verify(bytes.fromhex(hexkey), CATALOG_PREFIX + canonical_ascii(st), sig):
        raise GrantError("the store catalogue's signature does not verify", "bad_signature")
    return st


SIGNED_CATALOG = "catalog-signed.json"            # <cfg>/store/: licd's catalogue as signed (catalog.json is the page's view)


def signed_catalog_view(cfg_dir, pubkeys=None):
    """the last catalogue licd signed, re-verified now with the pinned keys -> the catalog_from_licd shape, or None."""
    try:
        with open(os.path.join(cfg_dir, "store", SIGNED_CATALOG)) as f:
            return catalog_from_licd(verify_catalog(json.load(f), pubkeys))
    except (OSError, ValueError, GrantError, TypeError, AttributeError):
        return None


def unlocked(unlock, cfg_dir=None, cache=None, catalog=None, pubkeys=None):
    """THE generic check for a small extra that ships inside Control or the library: is "<kind>:<id>" (a preset or a feature)
    unlocked on this computer now? True only while a grant for an item that unlocks it runs (gate.RUNS: active, or refresh
    due inside its offline grace). Never put an engine speed win or a default-ON lever behind this: extras only."""
    from ..plugins import gate as _gate
    m = re.fullmatch(r"(preset|feature|plugin):([a-z0-9][a-z0-9._-]{0,63})", str(unlock or ""))
    if not m:
        return False
    kind, xid = m.group(1), m.group(2)
    if cfg_dir is None:
        from .. import core as _core
        cfg_dir = _core.config_dir()
    cat = catalog
    if cat is None:                                # the unlock map comes from signed data only: licd's catalogue, re-verified
        cat = signed_catalog_view(cfg_dir, pubkeys) or catalog_view()   # on every read, else the built-in list (review F12)
    items = [it for it in (cat.get("items") if isinstance(cat, dict) else cat) or [] if isinstance(it, dict)]
    ids = {xid} | {it["id"] for it in items if it.get("kind") == kind and (it.get("unlocks") == unlock or xid in (it.get("grants") or []))}
    return any(_gate.run_allowed(cfg_dir, i, kind, cache) for i in sorted(ids))


def catalog_from_licd(st):
    """licd's signed catalogue statement -> the store.js shape (same as catalog_view, source "licd")."""
    items = []
    for x in st.get("items") or []:
        it = {k: x[k] for k in ("id", "kind", "name", "tagline", "description", "price") if k in x}
        for k in ("version", "level", "grants", "perm_version", "unlocks"):
            if x.get(k) is not None:
                it[k] = x[k]
        b = bundle_of(x) if x.get("kind") == "plugin" else None
        if b:
            it["bundle"] = b
        it["permissions"] = [_perm(pid) for pid in x.get("permissions") or []]
        items.append(it)
    cur = st.get("currency") or {}
    return {"version": 1, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(st.get("issued") or 0))), "source": "licd",
            "currency": {"name": cur.get("name", "credits"), "source": cur.get("source", ""), "rate_note": cur.get("rate_note", "")},
            "items": items}


STORE_URLS = {"catalog": "/api/store/catalog", "entitlements": "/api/store/entitlements"}   # announced in GET /api/info
STORE_URLS_LIVE = {"link": "/api/store/link", "voucher": "/api/store/voucher", "redeem": "/api/store/redeem",
                   "refresh": "/api/store/refresh", "unlink": "/api/store/unlink", "models": "/api/store/models",
                   "download": "/api/store/download", "downloads": "/api/store/downloads", "wallet": "/api/store/wallet"}


def announced():
    """what GET /api/info announces: the two read routes always; the licd actions only when a store URL is set."""
    d = dict(STORE_URLS)
    if store_url():
        d.update(STORE_URLS_LIVE)
    return d


def register(ROUTES, config_dir, pubkeys=None):
    """the store routes (token-guarded like every /api route: server.Handler). Without a store URL they serve the
    fixture catalogue and the local grants, as before; with one they speak to licd."""
    load_pubkeys()

    def cache_of(app):
        cache = getattr(app, "store_cache", None)
        if cache is None:
            cache = app.store_cache = EntitlementCache(config_dir(), pubkeys)
        return cache

    def client():
        base = store_url()
        if not base:
            raise _invalid("the store is not configured on this computer (PXA_STORE_URL)", 409, "store_off")
        return LicdClient(base, config_dir(), pubkeys)

    def r_store_catalog(app, body, query):
        if not store_url():
            return catalog_view()
        p = os.path.join(config_dir(), "store", "catalog.json")
        try:
            c = client()
            v = catalog_from_licd(c.catalog())
            for path, data in ((os.path.join(os.path.dirname(p), SIGNED_CATALOG), c.signed_catalog), (p, v)):
                with file_lock(path):
                    write_atomic(path, json.dumps(data).encode())
            return v
        except Exception:  # noqa: BLE001 - offline or a bad signature: the last good catalogue, else the fixture
            try:
                with open(p) as f:
                    v = json.load(f)
                v["source"] = "cache"
                return v
            except (OSError, ValueError):
                return catalog_view()

    def r_store_entitlements(app, body, query):
        cache = cache_of(app)
        snap = load_snapshot(config_dir())
        # a `checked` that is not a finite number (a hand edit) reads as never checked: due, never a 500
        if store_url() and snap.get("account") and time.time() - (_num(snap.get("checked")) or 0.0) > SYNC_EVERY_S:
            try:
                LicdClient(store_url(), config_dir(), pubkeys, timeout=5).sync(cache)
            except Exception:  # noqa: BLE001 - offline: the grants keep counting until not_after
                save_snapshot(config_dir(), None, ok=False)
            snap = load_snapshot(config_dir())
        return entitlements_view(cache, snap)

    def _after(app, r):
        """link / voucher / redeem answers carry licd's account view; then pull this computer's grants."""
        acct = r.get("account") if isinstance(r.get("account"), dict) and "credits" in r.get("account") else r
        save_snapshot(config_dir(), acct, ok=True)
        if r.get("grant"):
            cache_of(app).add(r["grant"], server_now=getattr(r, "server_time", None))
        try:
            s = client().sync(cache_of(app))
        except Exception:  # noqa: BLE001 - the page hears licd; the background sync tries again (S6 "after redeem")
            kick("redeem")
            raise
        return dict(entitlements_view(cache_of(app), load_snapshot(config_dir())), synced=s)

    def r_link(app, body, query):
        key = str((body or {}).get("key") or "").strip()
        if not key.startswith("pxk1."):
            raise _invalid("enter your quantizer key (pxk1....), the one PXA Control's encoder uses")
        return _after(app, client().call("link", extra={"key": key}))

    def r_voucher(app, body, query):
        code = str((body or {}).get("code") or "").strip()
        if not code.startswith("pxd1."):
            raise _invalid("a voucher code starts with pxd1.")
        return _after(app, client().call("voucher", extra={"code": code}))

    def r_redeem(app, body, query):
        item = str((body or {}).get("item_id") or "")
        if not _ITEM_RE.match(item):
            raise _invalid("bad item id")
        return _after(app, client().call("redeem", {"item_id": item}))

    def r_refresh(app, body, query):
        s = client().sync(cache_of(app))
        return dict(entitlements_view(cache_of(app), load_snapshot(config_dir())), synced=s)

    def r_unlink(app, body, query):
        r = client().call("unlink")
        if r.get("revocations"):
            cache_of(app).apply_revocations(r["revocations"])
        save_snapshot(config_dir(), {"account": None, "credits": None, "receipts": [], "owned": [], "machines": None}, ok=True)
        return entitlements_view(cache_of(app), load_snapshot(config_dir()))

    def r_models(app, body, query):
        """GET /api/store/models: per owned quant-tier item, licd's manifest (name, size, sha256, model, tier) and
        whether the file is already in the models folder."""
        c, cache = client(), cache_of(app)
        owned = {st["item_id"] for st, _s in cache.active() if st["kind"] == "quant-tier"}
        roots = _model_roots(app)
        out = []
        for it in c.catalog().get("items") or []:
            if it.get("kind") != "quant-tier":
                continue
            m = c.manifest(it["id"])
            for f in m.get("files") or []:
                f["local"] = any(os.path.exists(os.path.join(r, f["name"])) for r in roots)
            out.append({"item_id": it["id"], "name": it.get("name"), "price": it.get("price"), "owned": it["id"] in owned,
                        "files": m.get("files") or []})
        return {"items": out, "dest": roots[0] if roots else None}

    def r_download(app, body, query):
        b = body or {}
        item, fid = str(b.get("item_id") or ""), str(b.get("file_id") or "")
        if not _ITEM_RE.match(item) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", fid):
            raise _invalid("item_id and file_id please")
        roots = _model_roots(app)
        if not roots:
            raise _invalid("no models folder is set", 409, "no_models_dir")
        jobs = app.__dict__.setdefault("store_downloads", {})
        jid = secrets.token_hex(6)
        job = jobs[jid] = {"id": jid, "item_id": item, "file_id": fid, "state": "running", "have": 0, "size": None, "error": None, "path": None}

        def run():
            try:
                r = client().download(item, fid, roots[0], progress=lambda h, n: job.update(have=h, size=n))
                job.update(state="done", path=r["path"], sha256=r["sha256"])
            except Exception as e:  # noqa: BLE001
                job.update(state="failed", error=str(e)[:200])
        threading.Thread(target=run, daemon=True).start()
        return {"id": jid}

    def r_downloads(app, body, query):
        return {"downloads": list(getattr(app, "store_downloads", {}).values())}

    def r_wallet_view(app, body, query):
        """GET /api/store/wallet: the coins wallet this computer holds ({"wallet": null} before one is made or joined).
        Offline, the last genuine signed statement stands in for the balance (history stays empty).
        Review R1: one licd read serves every tab and window for VIEW_CACHE_S (a wallet action clears it), so polling tabs cannot use up
        the install key's hourly wallet-read / wallet-pc budget that redeem, beta and links share."""
        cl = client()
        key = (cl.base, WALLET_CHANGES[0])
        with _VIEW_LOCK:                                 # single flight: a second tab waits for the first read, then takes it
            hit = _VIEW.get("out")
            if hit is not None and _VIEW.get("key") == key and time.time() - _VIEW.get("t", 0) < VIEW_CACHE_S:
                return hit
            try:
                view = cl.wallet("view")
                try:
                    keep_statement(cl, config_dir(), view)   # N3: the offline balance follows the live one
                except Exception:  # noqa: BLE001 - review R7: the saved statement is a convenience; the live answer stands
                    pass
                out = {"wallet": view, "offline": False, "earn": earn_rows(cl)}
                if WALLET_CHANGES[0] == key[1]:          # an action that landed meanwhile makes this read stale: do not keep it
                    _VIEW.update(key=key, t=time.time(), out=out)
                return out
            except Exception as e:  # noqa: BLE001
                if getattr(e, "code", "") == "no_wallet":
                    return {"wallet": None, "offline": False, "earn": earn_rows(cl)}
                if getattr(e, "code", "") == "store_offline" or getattr(e, "status", 0) in (502, 503):
                    st = load_statement(config_dir())
                    earn = _EARN["rows"]                       # the last good table, never a new call while the store is down
                    if st:
                        return {"wallet": wallet_from_statement(st), "offline": True, "as_of": st.get("issued"), "earn": earn}
                    return {"wallet": None, "offline": True, "earn": earn}
                raise
    
    def wallet_action(action):
        def r(app, body, query):
            out = client().wallet(action, body if isinstance(body, dict) else {})
            _VIEW.pop("out", None)                      # review R1: the next view reads licd again
            if action in ("join", "email-join"):
                drop_statement(config_dir())            # N3: a joined wallet is another wallet (review R8: only after the join worked)
            if action == "statement":
                save_statement(config_dir(), out)       # checked before it is kept or shown
            elif action in ("create", "join", "email-join", "redeem", "beta", "kofi-confirm", "gift-confirm", "claim"):
                try:
                    keep_statement(client(), config_dir(), force=True)    # N3: the offline balance follows every wallet change
                except Exception:  # noqa: BLE001 - review R7: coins already moved; the wallet answer stands
                    pass
            if action == "claim":                       # licd counts the Ko-fi payments the code added; the panel words them as payments
                out = dict(out, payments=int(out.get("credited") or 0))
            if action in ("create", "join", "email-join", "redeem", "beta", "kofi-confirm", "claim"):
                try:                                    # a redeemed plugin or tier arrives as a grant for this computer
                    out = dict(out, synced=client().sync(cache_of(app)))
                except Exception:  # noqa: BLE001 - the wallet answer stands; the background sync tries again now
                    kick(action)
            return out
        return r

    # ---- supporter plugin packages (.pxp): installed only through the grant-bound download, loaded only on the gate
    def _plugins_dir():
        return os.path.join(config_dir(), "plugins", "incoming")

    def install_plugin(app, item):
        """one Download & install per item at a time (eng review F8): a second click, tab or Update while one runs is a
        409 install_busy, never a second writer of the same download file."""
        with _ITEM_LOCKS_GUARD:
            lk = _ITEM_LOCKS.setdefault(item, threading.Lock())
        if not lk.acquire(blocking=False):
            raise _invalid("%s is being installed already; wait for it to finish" % item, 409, "install_busy")
        try:
            return _install_plugin(app, item)
        finally:
            lk.release()

    def _install_plugin(app, item):
        """Download & install (and Update): item's signed .pxp, as the SIGNED catalogue lists it (bundle: version, signing kid,
        this platform's file id + sha256 + size), through licd's download-token (only for a computer holding the grant), then
        pxp.install (manifest signature with the pinned key, every file's sha, staged, atomic rename). CONTRACT.md 3-5."""
        from ..plugins import gate as _gate, pxp as _pxp, PLUGIN_PUBKEYS as _PK
        _gate.require(config_dir(), item, "plugin", cache_of(app))
        c = client()
        try:
            cat = c.catalog()                              # signature checked against the pinned licd key, every time
        except GrantError as e:
            raise _invalid("the Store catalogue could not be verified (%s); nothing was installed" % e, 502, "bad_catalog")
        row = next((x for x in cat.get("items") or [] if isinstance(x, dict) and x.get("id") == item), None)
        if row is None or row.get("kind") != "plugin":     # only what the signed catalogue lists as a plugin (review F3)
            raise _invalid("the Store has no package for %s" % item, 404, "no_package")
        b = bundle_of(row)
        if b is None and "bundle" in row:                  # present but malformed: refused, never the looser path (review F3)
            raise _invalid("the Store catalogue lists a package for %s that this PXA Control cannot read; nothing was installed" % item,
                           502, "bad_catalog")
        if b is None:                                      # a licd from before the bundle block: its newest listed .pxp
            files = [f for f in (c.manifest(item).get("files") or []) if str(f.get("name") or "").endswith(_pxp.PXP_EXT)]
            if not files:
                raise _invalid("the Store has no package for %s yet" % item, 404, "no_package")
            files.sort(key=lambda f: (_pxp.ver_key(f.get("release")), str(f.get("name") or "")))
            fid, expect, want_ver, want_kid = files[-1]["file_id"], None, None, None
        else:
            plat, pr = pick_platform(b)
            if pr is None:
                raise _invalid("the Store has no package of %s for this computer (%s)" % (item, this_platform()), 404, "no_package")
            if not b["kid"] or b["kid"] not in _PK:         # the pinned plugin keys only (pubkeys.json); never a key from the Store
                raise _invalid("the package is signed with a key this PXA Control does not trust", 400, "unknown_key")
            if b["min_host"] and host_too_old(b["min_host"]):
                raise _invalid("%s needs a newer PXA Control (%s or later)" % (row.get("name") or item, b["min_host"]), 409, "host_too_old")
            pid = next((r["plugin_id"] for r in _pxp.installed(config_dir(), cache=cache_of(app))
                        if (r.get("entitlement") or r["plugin_id"]) == item), item)   # keyed on the plugin id (review F10)
            have = _pxp.current_version(config_dir(), pid)
            if have and _pxp.ver_key(b["version"]) < _pxp.ver_key(have):
                raise _invalid("%s %s is installed; the Store lists the older %s" % (row.get("name") or item, have, b["version"]), 409, "downgrade")
            fid, expect, want_ver, want_kid = pr["file_id"], pr, b["version"], b["kid"]
        got = c.download(item, fid, _plugins_dir(), expect=expect) if expect is not None else c.download(item, fid, _plugins_dir())
        rid = next((r["plugin_id"] for r in _pxp.installed(config_dir(), cache=cache_of(app))
                    if (r.get("entitlement") or r["plugin_id"]) == item), None)
        h = getattr(app, "plugin_host", None)              # only a host that already runs; never made here
        was_running = h is not None and rid is not None and getattr(h.runs.get(rid), "state", "") == "running"
        if was_running:                                    # a reinstall or Update never swaps a running plugin's folder (review F11)
            h.stop(rid, "update")
        try:
            return _pxp.install(got["path"], config_dir(), item=item, expect_version=want_ver, expect_kid=want_kid,
                                refuse_older=True)
        finally:
            try:
                os.unlink(got["path"])
            except OSError:
                pass
            if was_running:                                # back on: the new version, or the old one when the install failed;
                try:                                       # start asks for consent again itself when the permissions grew
                    h.start(rid, by="update")
                except Exception:  # noqa: BLE001 - it stays off; its card says why
                    pass

    def r_plugins_installed(app, body, query):
        from ..plugins import pxp as _pxp
        return {"plugins": _pxp.installed(config_dir(), cache=cache_of(app))}

    def r_plugin_install(app, body, query):
        item = str((body or {}).get("item_id") or "")
        if not _ITEM_RE.match(item):
            raise _invalid("bad item id")
        from ..plugins import PluginError as _PE
        try:
            return {"installed": install_plugin(app, item)}
        except _PE as e:                                  # locked 403, a bad package 400: the page shows the words
            raise _invalid(str(e), getattr(e, "status", 400), e.code)

    _wallet_redeem = wallet_action("redeem")

    def r_wallet_redeem(app, body, query):
        """the coins panel's redeem; a supporter plugin whose grant just arrived is then installed from its .pxp."""
        out = _wallet_redeem(app, body, query)
        item = str((body or {}).get("item_id") or "") if isinstance(body, dict) else ""
        from ..plugins import gate as _gate
        if isinstance(out, dict) and _ITEM_RE.match(item) and _gate.run_allowed(config_dir(), item, "plugin", cache_of(app)):
            try:
                out = dict(out, plugin=install_plugin(app, item))
            except Exception as e:  # noqa: BLE001 - the unlock stands; Plugins offers the install again
                out = dict(out, plugin={"error": getattr(e, "code", "install_failed"), "message": str(e)[:200]})
        return out

    ROUTES[("GET", "/api/store/wallet")] = r_wallet_view
    for _a in LicdClient.WALLET_ACTIONS:
        if _a != "view":
            ROUTES[("POST", "/api/store/wallet/" + _a)] = wallet_action(_a)
    def r_store_unlocked(app, body, query):
        u = str(((query or {}).get("u") or [""])[0])[:100]
        return {"unlock": u, "active": unlocked(u, config_dir(), cache_of(app), pubkeys=pubkeys)}

    ROUTES[("GET", "/api/store/unlocked")] = r_store_unlocked
    ROUTES[("GET", "/api/store/catalog")] = r_store_catalog
    ROUTES[("GET", "/api/store/entitlements")] = r_store_entitlements
    ROUTES[("POST", "/api/store/link")] = r_link
    ROUTES[("POST", "/api/store/voucher")] = r_voucher
    ROUTES[("POST", "/api/store/redeem")] = r_redeem
    ROUTES[("POST", "/api/store/refresh")] = r_refresh
    ROUTES[("POST", "/api/store/unlink")] = r_unlink
    ROUTES[("GET", "/api/store/models")] = r_models
    ROUTES[("POST", "/api/store/download")] = r_download
    ROUTES[("GET", "/api/store/downloads")] = r_downloads
    ROUTES[("POST", "/api/store/wallet/redeem")] = r_wallet_redeem
    ROUTES[("GET", "/api/plugins/installed")] = r_plugins_installed
    ROUTES[("POST", "/api/plugins/install")] = r_plugin_install
    return ROUTES


def _model_roots(app):
    try:
        roots = list(app.model_roots())
    except Exception:  # noqa: BLE001
        roots = []
    if not roots:
        from .. import desktop as _desktop
        roots = [_desktop.default_models_dir()]
    return roots
