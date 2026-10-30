"""PXA Control plugins, the LOCAL side (stubs; design: PLUGINS-STORE-DESIGN.md). FIRST-PARTY ONLY (owner, 2026-10-14):
every plugin is built by PXA from Discord requests and signed with the one PXA plugin key; there is no third-party
publishing path. Anibal (tools/vigil) is the first plugin.

This module: the manifest loader + validator, the manifest signature check, and the consent record. The host (a child
process per plugin, pxa-plugin/1, its routes forwarded under /api/plugins/<plugin_id>/) is plugins/host.py + child.py;
its page in the Plugins tab is Control's UI (plan S3). Field names, permission names and the signing prefix follow
plugins/p-platform (tools/pxa_plugins/manifest.py, common.py) so one manifest serves both.

Manifest (plugin.json, in the plugin's folder):

    {"schema": 1, "plugin_id": "anibal", "name": "Anibal", "version": "0.1.0", "publisher": "pxa", "edition": "free",
     "min_host": "3.1",
     "entry": {"kind": "process", "argv": ["bin/anibal-plugin"], "protocol": "pxa-plugin/1"},
     "routes": ["/api/plugins/anibal/"],
     "ui": {"tab": {"title": "Anibal", "page": "ui/index.html", "icon": "ui/icon.svg"}},
     "permissions": ["device.mic", "device.speaker", "model.chat", "net.loopback"],   # + "net.internet:<host>" (one host each)
     "listen": [{"port": 10310, "proto": "wyoming", "scope": "lan"}],     # optional: lan (needs net.lan) or loopback, never public
     "overlay": {"page": "overlay/index.html", "scope": "loopback"},      # optional: a read-only page for OBS, 127.0.0.1 only
     "entitlement": null,                                  # or a store item id (kind "plugin") a grant must carry
     "python": {"package": "anibal", "path": "lib"},       # optional: an in-process package (pxp.load), until the host runs it
     "files": {"bin/anibal-plugin": "<sha256>", "ui/index.html": "<sha256>", ...},
     "signature": {"kid": "plug-2026-10", "sig": "<base64url Ed25519 over b'pxa-plugin-manifest|' + canonical(rest)>"}}
"""
import base64
import json
import os
import re
import threading
import time

MANIFEST_VERSION = 1
MANIFEST_PREFIX = b"pxa-plugin-manifest|"      # = p-platform common.MANIFEST_PREFIX (no cross-domain signature replay)
PLUGIN_PUBKEYS = {}         # kid -> 64 hex chars: the ONE offline first-party plugin key (filled at release; none here)
PUBLISHERS = ("pxa",)       # first-party only (owner, 2026-10-14): no publisher certificates, no third-party path
EDITIONS = ("free", "supporter")
# every permission a plugin can ask for and the consent card's words. The first twelve are p-platform's
# (manifest.PERMISSIONS, same names); device.*, home.assistant, gpu.read, engine.control are new for Anibal and Control.
# input.type is Dictate (hold a key, type the words): the focused window, only after the user allows it.
PERMISSIONS = {
    "net.loopback": "talk to programs on this computer",
    "net.lan": "accept connections from your home network",
    "net.internet": "reach the internet",
    "net.http": "fetch web pages",
    "fs.config": "keep its settings in the PXA config folder",
    "fs.read": "read files you point it at",
    "fs.write": "write files you point it at",
    "fs.workspace.read": "read its workspace folder",
    "fs.workspace.write": "write its workspace folder",
    "proc.spawn": "start helper programs",
    "model.chat": "send prompts to the models you run here",
    "speech.listen": "turn speech into text on this computer (nothing leaves it)",
    "speech.speak": "turn text into speech on this computer",
    "host.shell": "run commands on this computer (each one asks you first)",
    "plugin.tools": "offer tools to the chat agent",
    "device.mic": "use your microphone",
    "input.type": "type text into the window you are using",
    "device.speaker": "play sound",
    "device.audio-capture": "hear what this computer plays (a call, a video)",
    "device.camera": "use your camera",
    "home.assistant": "control Home Assistant devices",
    "gpu.read": "read card telemetry",
    "engine.control": "start and stop servers",
}
# scoped internet: "net.internet:<host>" = this one host only (the consent card lists every host; the host's egress proxy
# lets nothing else out). Never a wildcard, an address or a home-network name: those are net.lan / net.loopback.
NET_SCOPED = "net.internet:"
# ui.tab.glyph: the icon of a plugin's own tab in Control's bar, one of Control's own drawings (a plugin never ships markup for it)
NAV_GLYPHS = ("plug", "chat", "folder", "models", "spark", "play", "servers")
_HOST_RE = re.compile(r"^(?=.{4,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal", ".intranet", ".corp", ".localhost", ".localdomain", ".home.arpa", ".arpa")
MAX_SCOPED_HOSTS = 16
# what a plugin may listen on (manifest "listen"): its own ports, on the home network or this computer, never public.
# "lan" needs the net.lan permission; the host binds the address itself (a plugin never names one).
LISTEN_PROTOS = ("http", "ws", "tcp", "udp", "wyoming")
LISTEN_SCOPES = ("lan", "loopback")
MAX_LISTEN = 4
# an overlay (manifest "overlay"): one read-only page served on 127.0.0.1 only, for a capture program (OBS) on this computer
OVERLAY_SCOPES = ("loopback",)
ENTRY_KINDS = ("process",)
_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_VER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]{1,32})?$")
_REL_RE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._/-]{1,200}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
# a component Windows maps to a device whatever its extension (con.txt, aux.py, com1); review F13
_WIN_RESERVED = re.compile(r"^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])$", re.I)


def rel_ok(rel):
    """a package path that means the same file on Linux, Windows and macOS: relative, '/'-separated, ASCII letters, digits and
    ._-; no empty, '.' or '..' component, no component ending in '.', no Windows device name (review F13). The charset already
    rules out a backslash, ':' and spaces."""
    if not isinstance(rel, str) or not _REL_RE.match(rel):
        return False
    for c in rel.split("/"):
        if c in ("", ".", "..") or c.endswith(".") or _WIN_RESERVED.match(c.split(".")[0]):
            return False
    return True


class PluginError(ValueError):
    def __init__(self, msg, code="invalid_manifest"):
        super().__init__(msg)
        self.code = code
        self.status = 400


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def scoped_host(perm):
    """'net.internet:<host>' -> host when the host is a public DNS name, else None (not scoped, or refused)."""
    if not (isinstance(perm, str) and perm.startswith(NET_SCOPED)):
        return None
    h = perm[len(NET_SCOPED):]
    if not _HOST_RE.match(h) or h.endswith(_LOCAL_SUFFIXES):
        return None
    return h


def permission_known(perm):
    return perm in PERMISSIONS or scoped_host(perm) is not None


def permission_words(perm):
    """the consent card's words for one permission (a scoped host is named)."""
    h = scoped_host(perm)
    if h is not None:
        return "reach %s on the internet (nothing else)" % h
    return PERMISSIONS.get(perm, perm)


def consent_lines(m):
    """[(key, words)] for the consent card: every permission, then each port it listens on and its overlay."""
    out = [(x, permission_words(x)) for x in (m.get("permissions") or [])]
    for ln in m.get("listen") or []:
        if ln["port"] == 0:                     # loopback only (validated): the system picks a free port at start
            out.append(("listen:0", "accept %s connections from programs on this computer, on a free port" % ln["proto"]))
            continue
        where = "your home network" if ln["scope"] == "lan" else "this computer"
        out.append(("listen:%d" % ln["port"], "accept %s connections from %s on port %d" % (ln["proto"], where, ln["port"])))
    if m.get("overlay"):
        out.append(("overlay", "show a read-only overlay page to programs on this computer (OBS)"))
    return out


def _listen_problems(m, perms):
    ls = m.get("listen")
    if ls is None:
        return []
    if not isinstance(ls, list) or not 0 < len(ls) <= MAX_LISTEN:
        return ["listen: a list of 1-%d {port, proto, scope}" % MAX_LISTEN]
    p, seen = [], set()
    for ln in ls:
        if not isinstance(ln, dict) or set(ln) != {"port", "proto", "scope"}:
            p.append("listen: each entry is exactly {port, proto, scope} (the host picks the address)")
            continue
        port = ln["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not (port == 0 or 1024 <= port <= 65535):
            p.append("listen: port 1024-65535, or 0 (a free port) with scope loopback")
        elif port == 0 and ln["scope"] != "loopback":
            p.append("listen: port 0 only with scope loopback (a free port is never opened off this computer)")
        if ln["proto"] not in LISTEN_PROTOS:
            p.append("listen: proto one of " + ", ".join(LISTEN_PROTOS))
        if ln["scope"] not in LISTEN_SCOPES:
            p.append("listen: scope lan or loopback (a plugin never listens on the internet)")
        elif ln["scope"] == "lan" and "net.lan" not in perms:
            p.append("listen: scope lan needs the net.lan permission")
        key = (port, "udp" if ln["proto"] == "udp" else "tcp")
        if key in seen:
            p.append("listen: port %s twice" % port)
        seen.add(key)
    return p


def _overlay_problems(m):
    ov = m.get("overlay")
    if ov is None:
        return []
    files = m.get("files") if isinstance(m.get("files"), dict) else {}
    if not (isinstance(ov, dict) and set(ov) == {"page", "scope"} and rel_ok(str(ov.get("page") or ""))):
        return ["overlay: exactly {page (a path inside the plugin), scope: loopback}"]
    p = []
    if ov["scope"] not in OVERLAY_SCOPES:
        p.append("overlay: scope loopback only (an overlay is never served off this computer)")
    if files and ov["page"] not in files:
        p.append("overlay: the page must be one of the listed files")
    return p


def validate_manifest(m):
    """-> [problems] (empty = valid). Shape only; the signature is verify_manifest's job."""
    p = []
    if not isinstance(m, dict):
        return ["the manifest is not a JSON object"]
    if m.get("schema") != MANIFEST_VERSION:
        p.append(f"schema must be {MANIFEST_VERSION}")
    pid = str(m.get("plugin_id") or "")
    if not _ID_RE.match(pid):
        p.append("plugin_id: 2-32 of a-z 0-9 -, starting with a letter")
    if not isinstance(m.get("name"), str) or not 0 < len(m["name"]) <= 48:
        p.append("name: 1-48 characters")
    if not _VER_RE.match(str(m.get("version") or "")):
        p.append("version: semver x.y.z")
    if m.get("publisher") not in PUBLISHERS:
        p.append("publisher: first-party plugins only (pxa)")
    if m.get("edition") not in EDITIONS:
        p.append("edition: free or supporter")
    if not re.match(r"^\d+\.\d+(?:\.\d+)?$", str(m.get("min_host") or "")):
        p.append("min_host: the oldest PXA Control it runs in, x.y")
    e = m.get("entry")
    if not (isinstance(e, dict) and e.get("kind") in ENTRY_KINDS and isinstance(e.get("argv"), list) and e["argv"]
            and all(isinstance(a, str) and a for a in e["argv"]) and rel_ok(e["argv"][0])
            and e.get("protocol") == "pxa-plugin/1"):
        p.append("entry: {kind: process, argv: [a path inside the plugin, args...], protocol: pxa-plugin/1}")
    routes = m.get("routes") or []
    if not isinstance(routes, list) or any(not isinstance(r, str) or not r.startswith(f"/api/plugins/{pid}/") for r in routes):
        p.append(f"routes: only under /api/plugins/{pid}/")
    perms = m.get("permissions") or []
    if not isinstance(perms, list) or any(not permission_known(x) for x in perms) or len(set(map(str, perms))) != len(perms):
        p.append("permissions: unknown or repeated (known: " + ", ".join(sorted(PERMISSIONS)) + ", " + NET_SCOPED + "<host>)")
        perms = [x for x in perms if isinstance(x, str)] if isinstance(perms, list) else []
    elif sum(1 for x in perms if scoped_host(x)) > MAX_SCOPED_HOSTS:
        p.append("permissions: at most %d scoped internet hosts" % MAX_SCOPED_HOSTS)
    p += _listen_problems(m, perms)
    p += _overlay_problems(m)
    ui = m.get("ui")
    if ui is not None:
        tab = ui.get("tab") if isinstance(ui, dict) else None
        if not (isinstance(tab, dict) and isinstance(tab.get("title"), str) and 0 < len(tab["title"]) <= 32
                and rel_ok(str(tab.get("page") or ""))):
            p.append("ui.tab: {title (1-32), page (a path inside the plugin)}")
        elif not (isinstance(tab.get("nav", False), bool) and tab.get("glyph", "plug") in NAV_GLYPHS):
            p.append("ui.tab: nav (true: its own tab in Control's bar) and glyph (one of %s)" % ", ".join(NAV_GLYPHS))
    files = m.get("files")
    if not (isinstance(files, dict) and files and all(rel_ok(k) and _SHA_RE.match(str(v)) for k, v in files.items())):
        p.append("files: {relative path: sha256} for every file the plugin ships")
    elif len({k.lower() for k in files}) != len(files):
        p.append("files: two listed paths differ only by case (one file on Windows and macOS)")
    elif isinstance(e, dict) and isinstance(e.get("argv"), list) and e["argv"] and e["argv"][0] not in files:
        p.append("entry: the program must be one of the listed files")
    py = m.get("python")
    if py is not None:
        ok = (isinstance(py, dict) and re.match(r"^[a-z][a-z0-9_]{1,31}$", str(py.get("package") or ""))
              and rel_ok(str(py.get("path") or "")))
        if not ok:
            p.append("python: {package (a-z 0-9 _), path (a folder inside the plugin)}")
        elif isinstance(files, dict) and "%s/%s/__init__.py" % (py["path"], py["package"]) not in files:
            p.append("python: the package's __init__.py must be one of the listed files")
    if m.get("edition") == "supporter" and not m.get("entitlement"):
        p.append("entitlement: a supporter plugin names the store item whose grant runs it")
    ent = m.get("entitlement")
    if ent is not None and not (isinstance(ent, str) and re.match(r"^[a-z0-9][a-z0-9._-]{0,63}$", ent)):
        p.append("entitlement: a store item id or null")
    sig = m.get("signature")
    if not (isinstance(sig, dict) and isinstance(sig.get("kid"), str) and isinstance(sig.get("sig"), str)):
        p.append("signature: {kid, sig} (every plugin is signed)")
    return p


def verify_manifest(m, pubkeys=None):
    """raise PluginError unless the manifest is valid AND signed by the first-party plugin key."""
    probs = validate_manifest(m)
    if probs:
        raise PluginError("; ".join(probs))
    keys = PLUGIN_PUBKEYS if pubkeys is None else pubkeys
    hexkey = keys.get(m["signature"]["kid"]) or ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", hexkey):
        raise PluginError("signed with a key this Control does not know", "unknown_key")
    body = {k: v for k, v in m.items() if k != "signature"}
    t = m["signature"]["sig"]
    try:
        sig = base64.urlsafe_b64decode(t + "=" * (-len(t) % 4))
    except (ValueError, TypeError):
        sig = b""
    import pxa_encode_pkg as PK
    if len(sig) != 64 or not PK.ed25519_verify(bytes.fromhex(hexkey), MANIFEST_PREFIX + canonical(body), sig):
        raise PluginError("the plugin's signature does not verify", "bad_signature")
    return m


def verify_files(m, root):
    """every listed file has the listed sha256, and nothing is a link out of the plugin folder."""
    import hashlib
    root_r = os.path.realpath(root)
    for rel, want in m["files"].items():
        p = os.path.realpath(os.path.join(root, rel))
        if not (p == root_r or p.startswith(root_r + os.sep)):
            raise PluginError(f"{rel} points outside the plugin", "bad_file")
        h = hashlib.sha256()
        try:
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
        except OSError:
            raise PluginError(f"{rel} is missing", "bad_file")
        if h.hexdigest() != want:
            raise PluginError(f"{rel} does not match the signed manifest", "bad_file")
    return True


def load_manifest(folder, pubkeys=None, check_files=True):
    """plugin.json from folder -> the verified manifest (signature, then file hashes)."""
    try:
        with open(os.path.join(folder, "plugin.json"), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        raise PluginError(f"plugin.json unreadable: {e.__class__.__name__}")
    verify_manifest(m, pubkeys)
    if check_files:
        verify_files(m, folder)
    return m


class ConsentStore(object):
    """what the user allowed each plugin (config dir plugins/consent.json, 0600). Consent is per plugin and per
    permission; a new version asking for MORE permissions needs a new consent (the old one does not stretch); a
    revoked consent stops the plugin at its next call."""

    def __init__(self, cfg_dir, clock=time.time):
        self.dir = os.path.join(cfg_dir, "plugins")
        self.path = os.path.join(self.dir, "consent.json")
        self.clock = clock
        self.lock = threading.Lock()

    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, d):
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(d, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def grant(self, manifest, permissions, by="page"):
        """the user ticked these permissions on the consent card for this manifest; -> the record."""
        asked = set(manifest.get("permissions") or [])
        perms = sorted(set(permissions))
        if not set(perms) <= asked:
            raise PluginError("consent can only cover what the plugin asked for", "bad_consent")
        rec = {"plugin": manifest["plugin_id"], "version": manifest["version"], "permissions": perms,
               "asked": sorted(asked), "granted_at": int(self.clock()), "by": str(by)[:32], "revoked_at": None}
        with self.lock:
            d = self._load()
            d[manifest["plugin_id"]] = rec
            self._save(d)
        return rec

    def revoke(self, plugin_id):
        with self.lock:
            d = self._load()
            if plugin_id in d and d[plugin_id].get("revoked_at") is None:
                d[plugin_id]["revoked_at"] = int(self.clock())
                self._save(d)
                return True
        return False

    def state(self, manifest):
        """-> {"status": none|granted|partial|needs_reconsent|revoked, "permissions": [...], "missing": [...]}"""
        rec = self._load().get(manifest["plugin_id"])
        asked = set(manifest.get("permissions") or [])
        if not rec:
            return {"status": "none", "permissions": [], "missing": sorted(asked)}
        if rec.get("revoked_at"):
            return {"status": "revoked", "permissions": [], "missing": sorted(asked)}
        have = set(rec.get("permissions") or [])
        new = asked - set(rec.get("asked") or [])
        if new:
            return {"status": "needs_reconsent", "permissions": sorted(have & asked), "missing": sorted(asked - have)}
        return {"status": "granted" if asked <= have else "partial", "permissions": sorted(have & asked),
                "missing": sorted(asked - have)}

    def allowed(self, manifest, permission):
        st = self.state(manifest)
        return st["status"] in ("granted", "partial") and permission in st["permissions"]
