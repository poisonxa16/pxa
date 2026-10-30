"""PXA Control's plugin host (PLUGINS-PROTOTYPE-PLAN S2; PLUGINS-STORE-DESIGN s2, s3, s7).

* One child process per installed plugin, started from the signed manifest's entry.argv (argv only, never a shell), cwd =
  its own data folder (<config>/plugins/data/<id>, 0700), environment = an allow-list (no Control token, no HF token, no
  store or licence keys) plus PXA_PLUGIN_* (socket, one-time key, id, data folder, the consented permissions, the ports it
  may bind). The installed folder is re-verified (signature + every sha256) before every start.
* pxa-plugin/1 (plugins/child.py) on a per-plugin unix socket (POSIX: 0700 folder, peer uid + process group checked) or
  a named pipe (Windows), HMAC challenge with the one-time key both ways.
* /api/plugins/<id>/* is forwarded by server.py AFTER Control's own Host, Origin/fetch-metadata and token checks; only the
  method, path, query, Content-Type and body go across; the answer's Content-Type is limited (never HTML: a plugin cannot
  put a page on Control's origin) and carries a sandbox CSP.
* Host calls (plugin -> host): each name maps to a permission; a call the consent record does not hold answers 403
  permission_denied and writes an audit row. A revoked consent or a lost grant stops the plugin. speech.stt/vad
  (speech.listen) and speech.tts (speech.speak) go to the shared speech service (pxa_control.speech), imported lazily.
  net.fetch/net.search (net.http) share one per-plugin token bucket (20 at once, then 30 a minute; PXA_PLUGIN_NET_RATE,
  0 = off): over it the call answers 429 rate_limited with retry_after_s and an audit row, and nothing is fetched.
* Gate: a supporter plugin starts only with an active machine-bound grant (gate.require, the one gate); after every
  store sync (store.on_sync, S6) and every health tick, running supporter plugins without one are stopped.
* Lifecycle: health ping every 15 s (5 s to answer), a dead or silent plugin restarts with backoff 1-16 s, at most 5
  restarts in 10 min, then "failed" until someone starts it again; stop on Control exit, consent revoke, grant loss, or
  from the tab.
* <config>/plugins/audit.jsonl (0600): start, stop, exit, restart, failed, locked, denied; names and codes only, never a
  body, an argument or a token.
* S4 (plugins/overlay.py, keystore.py, egress.py): a manifest `overlay` is served by Control at /plugins/<id>/overlay?k=<key>
  (loopback peers only, the plugin feeds it with overlay.* calls); secrets.* keep a plugin's sign-in keys in Control's key store
  (by name, never in plugin files); `net.internet:<host>` plugins get their own CONNECT-only egress proxy (consented hosts, TLS
  ports, public addresses) and, on Linux when that is their only network, start in a network namespace with no other way out.

Honest limits (v1): no OS sandbox yet (p-platform's bwrap/landlock ladder, Windows job object): the permissions decide what
the HOST does for a plugin; first-party signed code is what runs. Windows named-pipe path is written, not yet run on the
Windows VM.
"""
import collections
import json
import os
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

from . import PluginError, load_manifest, ConsentStore, _ID_RE
from . import gate, pxp, sdk
from .. import pyfind
from . import egress as _egress
from .keystore import KeyStore, KeyStoreError
from .overlay import Overlays, OverlayError
from .child import Channel, ChannelClosed, PROTOCOL, b64d, b64e

HEALTH_S = 15.0
HEALTH_TIMEOUT_S = 5.0
RESTARTS_MAX = 5
RESTART_WINDOW_S = 600.0
BACKOFF_S = (1, 2, 4, 8, 16)
HELLO_TIMEOUT_S = 10.0
STOP_GRACE_S = 3.0
FORWARD_TIMEOUT_S = 60.0
MAX_RESPONSE = 16 << 20
TICK_S = 1.0
WINDOWS = os.name == "nt"
TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# what a plugin process inherits from Control's environment (nothing else; never a token or key)
ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
            "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "XDG_RUNTIME_DIR", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
            "PULSE_SERVER", "PIPEWIRE_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY")
# what a plugin answer may be served as on Control's origin (anything else -> application/octet-stream)
RESPONSE_TYPES = ("application/json", "text/plain", "text/event-stream", "text/markdown", "text/csv", "application/octet-stream",
                  "audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "image/png", "image/jpeg", "image/webp")
PLUGIN_CSP = "default-src 'none'; sandbox"
# host calls: name -> the permission it needs (None: compute only, or the plugin's own state)
HOST_CALLS = {
    "model.chat": "model.chat",
    "model.list": "model.chat",         # the running servers a plugin may pick with model.chat {seat} (names, never paths)
    "net.fetch": "net.http",            # one public web page as text, fetched BY THE HOST (private addresses always refused)
    "net.search": "net.http",           # a web search through the Assistant's search settings (titles, links, snippets)
    "gpu.read": "gpu.read",
    "notify": None,
    "entitlement.state": None,
    "speech.stt": "speech.listen",      # the shared speech service (plan S5, pxa_control.speech, pxa-speech/1)
    "speech.vad": "speech.listen",
    "speech.tts": "speech.speak",
    "overlay.emit": None,               # S4: the plugin's own overlay (needs a manifest overlay), plugins/overlay.py
    "overlay.audio": None,
    "overlay.url": None,
    "overlay.rotate": None,
    "secrets.put": None,                # S4: the plugin's own keys in Control's key store, plugins/keystore.py
    "secrets.get": None,
    "secrets.delete": None,
    "secrets.names": None,
    "speech.info": ("speech.listen", "speech.speak"),   # a tuple = any one of them: which voices / hearing model are there
}
MAX_SPEECH_AUDIO = 120 * 16000 * 2      # = the speech service's 120 s of 16 kHz s16le per call
RESERVED_IDS = ("install", "installed", "host")
AUDIT_MAX_BYTES = 4 << 20
NET_RATE_PER_MIN = 30     # net.fetch + net.search per plugin, sustained, a minute (PXA_PLUGIN_NET_RATE=<n>; 0 = no limit)
NET_BURST = 20            # calls one plugin may make back to back before the rate applies
SPEECH_WAIT_S = 15.0      # how long a speech.* call waits for the speech service the host just started
SPEECH_RETRY_S = 30.0     # a failed start is not retried sooner than this
SPEECH_STATUS_S = 15.0    # the speech setup's status from another Python is asked again after this (the tab refreshes every 5 s)
# what the speech setup's downloads need from Control's environment as well: proxies and certificate settings
NET_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE")
NO_WINDOW = 0x08000000    # Windows CREATE_NO_WINDOW: no console window for a Python child of a windowed Control


class HostError(PluginError):
    def __init__(self, msg, code, status):
        super().__init__(msg, code)
        self.status = status


class _Run(object):
    """one plugin's process and channel."""

    def __init__(self, pid):
        self.pid = pid
        self.manifest = None
        self.dir = None
        self.proc = None
        self.channel = None
        self.sock = None
        self.sock_dir = None
        self.address = None
        self.key = None
        self.state = "stopped"          # starting | running | restarting | stopped | failed | locked
        self.why = None
        self.restarts = collections.deque()
        self.restart_at = None
        self.last_health = 0.0
        self.started_at = None
        self.hello = threading.Event()
        self.lock = threading.RLock()
        self.log = None
        self.egress = None              # S4: the plugin's EgressProxy while it runs
        self.sealed = False             # started in its own network namespace (no way out but the proxy)


def _is_python(path):
    try:
        with open(path, "rb") as f:
            first = f.read(256).split(b"\n", 1)[0]
    except OSError:
        return False
    return first.startswith(b"#!") and b"python" in first


def _ctype(ct):
    """the plugin's Content-Type, re-written from an allow-list (never echoed: no header injection, never HTML)."""
    base = str(ct or "").split(";", 1)[0].strip().lower()
    if base in RESPONSE_TYPES:
        return base + "; charset=utf-8" if base.startswith("text/") or base == "application/json" else base
    return "application/octet-stream"


class RateLimit(object):
    """A token bucket per key: `burst` calls at once, then `per_min` a minute. take(key) -> 0.0 when the call may go, else the
    seconds until the next one may. per_min <= 0 turns it off. Keys are plugin ids (bounded by what is installed)."""

    def __init__(self, per_min, burst, clock=time.monotonic):
        self.per_min, self.burst, self.clock = float(per_min), float(max(1, int(burst))), clock
        self._b = {}
        self._lock = threading.Lock()

    def take(self, key):
        if self.per_min <= 0:
            return 0.0
        rate = self.per_min / 60.0
        with self._lock:
            now = self.clock()
            tokens, last = self._b.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + max(0.0, now - last) * rate)
            if tokens >= 1.0:
                self._b[key] = (tokens - 1.0, now)
                return 0.0
            self._b[key] = (tokens, now)
            return (1.0 - tokens) / rate


def net_rate_per_min():
    """PXA_PLUGIN_NET_RATE (calls a minute per plugin; 0 = no limit), else NET_RATE_PER_MIN."""
    try:
        return max(0, int(os.environ.get("PXA_PLUGIN_NET_RATE", NET_RATE_PER_MIN)))
    except ValueError:
        return NET_RATE_PER_MIN


class PluginHost(object):
    def __init__(self, cfg_dir, cache=None, consent=None, pubkeys=None, clock=time.time, health_s=HEALTH_S,
                 health_timeout_s=HEALTH_TIMEOUT_S, restarts_max=RESTARTS_MAX, restart_window_s=RESTART_WINDOW_S,
                 backoff=BACKOFF_S, tick_s=TICK_S, model_chat=None, gpu_read=None, python=None, tools_dir=None,
                 hello_timeout_s=HELLO_TIMEOUT_S, speech=None, model_list=None, web_fetch=None, web_search=None, net_limit=None,
                 egress_opts=None):
        from .. import store as S
        self.cfg_dir = cfg_dir
        self.cache = cache if cache is not None else S.EntitlementCache(cfg_dir)
        self.consent = consent if consent is not None else ConsentStore(cfg_dir)
        self.pubkeys = pubkeys
        self.clock = clock
        self.health_s, self.health_timeout_s = health_s, health_timeout_s
        self.restarts_max, self.restart_window_s, self.backoff = restarts_max, restart_window_s, tuple(backoff)
        self.tick_s = tick_s
        self.hello_timeout_s = hello_timeout_s
        self._last_recheck = 0.0
        self.speech = speech or _speech_client      # () -> a SpeechClient; raises when the service is not there
        # the shared speech service is started by the host on the first speech.* call that finds it down (PXA_SPEECH_AUTOSTART=0:
        # never; someone runs `python3 -m pxa_control.speech serve` by hand). Only with the real client, never with a test stand-in.
        self.speech_autostart = speech is None and os.environ.get("PXA_SPEECH_AUTOSTART", "1") != "0"
        self._speech_proc, self._speech_t, self._speech_lock = None, 0.0, threading.Lock()
        self._setup_proc, self._setup_lock, self._speech_st = None, threading.Lock(), None    # the speech setup (Plugins tab)
        self.model_chat, self.gpu_read = model_chat, gpu_read
        self.model_list = model_list                # () -> {"models": [...]}: the running servers (model.list)
        self.web_fetch = web_fetch or net_fetch     # (args) -> {url, ctype, title, text, truncated} (net.fetch)
        self.web_search = web_search or net_search  # (args) -> {query, results, note} (net.search)
        self.net_limit = net_limit or RateLimit(net_rate_per_min(), NET_BURST)   # per plugin, net.fetch + net.search together
        self.python, self.tools_dir = python, tools_dir   # None: found when first needed (the properties below)
        self.runs = {}
        self.notices = collections.deque(maxlen=50)
        self._lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.audit_path = os.path.join(cfg_dir, "plugins", "audit.jsonl")
        self.egress_opts = dict(egress_opts or {})     # tests only: resolver / public_only / ports for a loopback stand-in
        self.overlays = Overlays(cfg_dir)
        self.keys = KeyStore(cfg_dir)
        S.on_sync(self._on_sync)

    # ---- turned off by the user (Deactivate): autostart skips it until the user turns it on again ----------------
    def _disabled_path(self):
        return os.path.join(self.cfg_dir, "plugins", "disabled.json")

    def disabled(self):
        try:
            with open(self._disabled_path()) as f:
                d = json.load(f)
            return sorted(x for x in d.get("off", []) if isinstance(x, str) and _ID_RE.match(x)) if isinstance(d, dict) else []
        except (OSError, ValueError):
            return []

    def set_disabled(self, pid, off):
        with self._lock:
            cur = set(self.disabled())
            if (pid in cur) == bool(off):
                return
            cur.add(pid) if off else cur.discard(pid)
            p = self._disabled_path()
            os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"off": sorted(cur)}, f)
            os.replace(tmp, p)

    def uninstall(self, pid, by="page"):
        """Remove: stop it, then its files go (consent and saved keys stay until the user withdraws or forgets them)."""
        self.stop(pid, "removed")
        out = pxp.uninstall(self.cfg_dir, pid)
        with self._lock:
            self.runs.pop(pid, None)
        self.set_disabled(pid, True)               # a reinstall stays switched off until the user turns it on (review F2)
        self.audit(pid, "uninstall", by=by)
        return out

    # ---- audit -----------------------------------------------------------------------------------------------
    def audit(self, plugin, event, **kw):
        row = {"t": int(time.time()), "plugin": plugin, "event": event}
        for k, v in kw.items():
            if v is not None:
                row[k] = v if isinstance(v, (int, float, bool)) else str(v)[:200]
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._audit_lock:
            os.makedirs(os.path.dirname(self.audit_path), mode=0o700, exist_ok=True)
            try:
                if os.path.getsize(self.audit_path) > AUDIT_MAX_BYTES:
                    os.replace(self.audit_path, self.audit_path + ".1")
            except OSError:
                pass
            fd = os.open(self.audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(line)
        return row

    # ---- what is installed -----------------------------------------------------------------------------------
    @property
    def python(self):
        """the Python that runs Python plugins and the speech service: the one given, else pyfind's (a source run's own
        interpreter; a compiled Control's bundled or a found one, None when there is none)."""
        return self._python or pyfind.find(self.cfg_dir)

    @python.setter
    def python(self, v):
        self._python = v

    @property
    def tools_dir(self):
        """where plugins and the speech service import pxa_control from: the one given, else the tools folder (a source run)
        or the plugin kit unpacked from a compiled Control's bundle (None when it has none)."""
        return self._tools_dir or sdk.tools_dir(self.cfg_dir)

    @tools_dir.setter
    def tools_dir(self, v):
        self._tools_dir = v

    def runtime_missing(self):
        """why a Python plugin or the speech service cannot start in this Control, in plain words, or None."""
        if not self.python:
            return pyfind.missing_words()
        if not self.tools_dir:
            return "This copy of PXA Control is missing the part plugins run on. Reinstall PXA Control."
        return None

    def _run(self, pid):
        with self._lock:
            r = self.runs.get(pid)
            if r is None:
                r = self.runs[pid] = _Run(pid)
            return r

    def knows(self, pid):
        return bool(_ID_RE.match(pid or "")) and pid not in RESERVED_IDS and (pid in self.runs or pxp.folder(self.cfg_dir, pid) is not None)

    def manifest(self, pid):
        """the installed, verified manifest (signature + every file hash, now) and its folder."""
        if pid in RESERVED_IDS or not _ID_RE.match(pid or ""):
            raise HostError("no such plugin", "not_installed", 404)
        d = pxp.folder(self.cfg_dir, pid)
        if not d:
            raise HostError("%s is not installed" % pid, "not_installed", 404)
        m = load_manifest(d, self.pubkeys)
        if m["plugin_id"] != pid:
            raise HostError("the installed folder holds another plugin", "bad_file", 400)
        extra = pxp.unlisted_any(d, m, self.cfg_dir)
        if extra:
            raise HostError("%s is not part of the signed plugin; reinstall it" % extra, "bad_file", 400)
        return m, d

    def _check(self, m):
        """may this plugin run now? Locked (403) without a grant, 409 without consent."""
        if m["edition"] == "supporter":
            gate.require(self.cfg_dir, m["entitlement"], "plugin", self.cache, name=m["name"])
        st = self.consent.state(m)
        if st["status"] not in ("granted", "partial"):
            raise HostError("%s needs your consent first (%s)" % (m["name"], st["status"]), "needs_consent", 409)
        return st

    # ---- start / stop ----------------------------------------------------------------------------------------
    def start(self, pid, by="page"):
        r = self._run(pid)
        with r.lock:
            if r.state == "running" and r.proc is not None and r.proc.poll() is None:
                return self.status_of(pid)
            m, d = self.manifest(pid)
            try:
                self._check(m)
            except gate.Locked:
                r.state, r.why = "locked", "grant"
                self.audit(pid, "locked", by=by)
                raise
            if by != "restart":
                r.restarts.clear()
            self._spawn(r, m, d)
            self.audit(pid, "start", version=m["version"], by=by, pid=r.proc.pid)
        self._ensure_loop()
        return self.status_of(pid)

    def _binds(self, m, perms):
        out = []
        for ln in m.get("listen") or []:
            if ln["scope"] == "lan" and "net.lan" not in perms:
                continue                                   # not consented: the plugin degrades (no LAN port)
            out.append({"port": ln["port"], "proto": ln["proto"], "host": "0.0.0.0" if ln["scope"] == "lan" else "127.0.0.1"})
        return out

    def _listen(self, r):
        r.key = secrets.token_bytes(32)
        if WINDOWS:
            from multiprocessing.connection import Listener
            r.address = r"\\.\pipe\pxa-plugin-%s-%s" % (r.pid, secrets.token_hex(8))
            r.sock = Listener(r.address, family="AF_PIPE", authkey=r.key)
            return
        base = os.path.join(self.cfg_dir, "plugins", "run")
        if len(base) > 80:                                 # sun_path is ~104-108 bytes
            base = tempfile.mkdtemp(prefix="pxa-plug-")
        os.makedirs(base, mode=0o700, exist_ok=True)
        os.chmod(base, 0o700)
        r.sock_dir = base
        r.address = os.path.join(base, "%s-%s.sock" % (r.pid, secrets.token_hex(4)))
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(r.address)
        os.chmod(r.address, 0o600)
        s.listen(1)
        r.sock = s

    def _accept(self, r):
        from multiprocessing import connection as mpc
        if WINDOWS:
            box = {}
            t = threading.Thread(target=lambda: box.setdefault("c", r.sock.accept()), daemon=True)
            t.start()
            t.join(self.hello_timeout_s)
            if "c" not in box:
                raise HostError("the plugin did not connect", "plugin_failed_start", 502)
            return box["c"]
        deadline = time.monotonic() + self.hello_timeout_s
        r.sock.settimeout(0.1)
        while True:
            try:
                c, _addr = r.sock.accept()
                break
            except socket.timeout:
                pass
            except OSError:
                raise HostError("the plugin did not connect", "plugin_failed_start", 502)
            if r.proc.poll() is not None:
                raise HostError("the plugin exited at start (%s)" % r.proc.returncode, "plugin_failed_start", 502)
            if time.monotonic() > deadline:
                raise HostError("the plugin did not connect", "plugin_failed_start", 502)
        c.settimeout(None)
        if hasattr(socket, "SO_PEERCRED"):
            ppid, puid, _pgid = struct.unpack("3i", c.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
            try:
                same_group = os.getpgid(ppid) == r.proc.pid
            except OSError:
                same_group = False
            if puid != os.getuid() or not (ppid == r.proc.pid or same_group):
                c.close()
                raise HostError("another process tried the plugin's socket", "bad_peer", 502)
        conn = mpc.Connection(c.detach())
        try:
            mpc.deliver_challenge(conn, r.key)
            mpc.answer_challenge(conn, r.key)
        except (mpc.AuthenticationError, OSError, EOFError):
            conn.close()
            raise HostError("the plugin's key did not match", "bad_peer", 502)
        return conn

    def _spawn(self, r, m, d):
        argv = list(m["entry"]["argv"])
        prog = os.path.join(d, *argv[0].split("/"))
        is_py = _is_python(prog)
        why = self.runtime_missing() if is_py else None
        if why:
            raise HostError(why, "no_python", 409)
        perms = list(self.consent.state(m)["permissions"])
        data = os.path.join(self.cfg_dir, "plugins", "data", r.pid)
        os.makedirs(data, mode=0o700, exist_ok=True)
        logs = os.path.join(self.cfg_dir, "plugins", "logs")
        os.makedirs(logs, mode=0o700, exist_ok=True)
        self._close(r)
        self._listen(r)
        env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
        env.update({"PXA_PLUGIN_SOCKET": r.address, "PXA_PLUGIN_KEY": r.key.hex(), "PXA_PLUGIN_ID": r.pid,
                    "PXA_PLUGIN_VERSION": m["version"], "PXA_PLUGIN_PROTOCOL": PROTOCOL, "PXA_PLUGIN_DATA": data,
                    "PXA_PLUGIN_PERMISSIONS": json.dumps(perms), "PXA_PLUGIN_LISTEN": json.dumps(self._binds(m, perms)),
                    "PXA_CONTROL_TOOLS": self.tools_dir or "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
        if m.get("overlay"):
            env["PXA_PLUGIN_OVERLAY"] = "host"            # S4: Control serves the overlay; the plugin feeds it with overlay.*
        r.sealed = False
        if _egress.wants_proxy(m, perms):
            r.egress = _egress.EgressProxy(r.pid, lambda _r=r: self._egress_allowed(_r), audit=lambda ev, **kw: self.audit(r.pid, ev, **kw),
                                           run_dir=r.sock_dir, **self.egress_opts).start()
            env.update(r.egress.env())
        # -I: the script's folder is not on sys.path and PYTHON* variables are ignored (the entry adds its own lib/); review F6
        argv = ([self.python, "-I", "-B", prog] if is_py else [prog]) + argv[1:]
        if r.egress is not None and _egress.sealable(m, perms):
            pre = _egress.netns_prefix()
            if pre:
                argv, r.sealed = pre + argv, True
        if r.egress is not None:
            env["PXA_PLUGIN_EGRESS_MODE"] = "sealed" if r.sealed else "proxy"
        logp = os.path.join(logs, r.pid + ".log")
        try:
            if os.path.getsize(logp) > (2 << 20):
                os.replace(logp, logp + ".1")
        except OSError:
            pass
        r.log = open(os.open(logp, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab")
        kw = {"creationflags": 0x00000200 | 0x08000000} if WINDOWS else {"start_new_session": True}
        r.hello.clear()
        r.manifest, r.dir, r.state, r.why = m, d, "starting", None
        try:
            r.proc = subprocess.Popen(argv, cwd=data, env=env, stdin=subprocess.DEVNULL, stdout=r.log, stderr=subprocess.STDOUT,
                                      close_fds=True, **kw)
            if r.egress is not None:
                r.egress.pgid = r.proc.pid
            conn = self._accept(r)
            r.channel = Channel(conn, on_request=lambda msg, _r=r: self._from_plugin(_r, msg)).start()
            if not r.hello.wait(self.hello_timeout_s):
                raise HostError("the plugin did not say hello", "plugin_failed_start", 502)
        except Exception:
            self._kill(r)
            self._close(r)
            r.state = "stopped"
            raise
        r.state, r.started_at, r.last_health = "running", self.clock(), self.clock()

    def _kill(self, r):
        p = r.proc
        if p is None or p.poll() is not None:
            return
        try:
            if WINDOWS:
                p.terminate()
            else:
                os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            p.wait(STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            try:
                if WINDOWS:
                    p.kill()
                else:
                    os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                p.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                pass

    def _close(self, r):
        if r.egress is not None:
            r.egress.close()
            r.egress = None
        if r.channel is not None:
            r.channel.close()
            r.channel = None
        if r.sock is not None:
            try:
                r.sock.close()
            except OSError:
                pass
            r.sock = None
        if r.address and not WINDOWS:
            try:
                os.unlink(r.address)
            except OSError:
                pass
        if r.sock_dir and r.sock_dir.startswith(tempfile.gettempdir()) and os.path.basename(r.sock_dir).startswith("pxa-plug-"):
            shutil.rmtree(r.sock_dir, ignore_errors=True)
            r.sock_dir = None
        r.address = None
        if r.log is not None:
            try:
                r.log.close()
            except OSError:
                pass
            r.log = None

    def stop(self, pid, reason="page"):
        r = self.runs.get(pid)
        if r is None:
            return {"plugin_id": pid, "state": "stopped"}
        with r.lock:
            was = r.state
            if r.channel is not None and r.proc is not None and r.proc.poll() is None:
                try:
                    r.channel.request({"op": "stop"}, STOP_GRACE_S)
                    r.proc.wait(STOP_GRACE_S)
                except (ChannelClosed, TimeoutError, subprocess.TimeoutExpired, OSError):
                    pass
            self._kill(r)
            self._close(r)
            r.state = "locked" if reason == "grant" else ("failed" if reason == "failed" else "stopped")
            r.why, r.restart_at = reason, None
            self.overlays.stopped(pid)
            if was in ("running", "starting", "restarting"):
                self.audit(pid, "stop", reason=reason)
        return self.status_of(pid)

    def autostart(self):
        """at Control start: every installed plugin with consent (and, for a supporter plugin, an active grant). -> ids"""
        out = []
        off = set(self.disabled())
        for it in pxp.installed(self.cfg_dir, self.pubkeys, self.cache):
            if not it.get("ok") or (it.get("edition") == "supporter" and it.get("gate") not in gate.RUNS) or it["plugin_id"] in off:
                continue
            try:
                m, _d = self.manifest(it["plugin_id"])
                if self.consent.state(m)["status"] not in ("granted", "partial"):
                    continue
                self.start(it["plugin_id"], by="control-start")
                out.append(it["plugin_id"])
            except Exception as e:  # noqa: BLE001 - one plugin never stops Control from starting
                self.audit(it["plugin_id"], "failed", code=getattr(e, "code", e.__class__.__name__))
        return out

    def stop_all(self, reason="exit"):
        for pid in list(self.runs):
            self.stop(pid, reason)

    def close(self):
        self._stop.set()
        self.stop_all("exit")
        self._speech_stop()
        from .. import store as S
        S.off_sync(self._on_sync)

    # ---- traffic ---------------------------------------------------------------------------------------------
    def forward(self, pid, method, path, query, ctype, body):
        """a request that passed Control's checks -> (status, content-type, bytes). Starts the plugin on first use."""
        r = self.runs.get(pid)
        if r is None or r.state != "running":
            self.start(pid, by="use")
            r = self.runs[pid]
        m = r.manifest
        if not any(path.startswith(x) for x in (m.get("routes") or [])):
            raise HostError("not a route of %s" % pid, "not_found", 404)
        try:
            res = r.channel.request({"op": "http", "method": method, "path": path, "query": query or "",
                                     "ctype": str(ctype or "")[:200], "b64": b64e(body or b"")}, FORWARD_TIMEOUT_S)
        except (ChannelClosed, TimeoutError, AttributeError):
            raise HostError("%s did not answer" % m["name"], "plugin_unavailable", 502)
        status = res.get("status")
        if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status <= 599 or status in (204, 304):
            status = 502                                   # always a status that carries the body Control sends
        data = b64d(res.get("b64"))
        if len(data) > MAX_RESPONSE:
            raise HostError("%s answered too much" % m["name"], "plugin_too_large", 502)
        return status, _ctype(res.get("ctype")), data

    def _from_plugin(self, r, msg):
        op = msg.get("op")
        m = r.manifest
        if op == "hello":
            if msg.get("protocol") != PROTOCOL or msg.get("plugin") != r.pid:
                return {"status": 400, "code": "bad_hello", "error": "pxa-plugin/1 and the plugin's own id please"}
            perms = self.consent.state(m)["permissions"]
            r.hello.set()
            return {"status": 200, "protocol": PROTOCOL, "permissions": perms, "listen": self._binds(m, perms),
                    "host_calls": sorted(HOST_CALLS), "overlay": bool(m.get("overlay")),
                    "egress": ("sealed" if r.sealed else "proxy") if r.egress is not None else None}
        if op != "call":
            return {"status": 404, "code": "unknown_op"}
        name = str(msg.get("name") or "")[:64]
        if name not in HOST_CALLS:
            self.audit(r.pid, "denied", call=name, code="unknown_call")
            return {"status": 404, "code": "unknown_call", "error": "no such host call"}
        st = self.consent.state(m)
        if st["status"] not in ("granted", "partial"):
            self.audit(r.pid, "denied", call=name, code="consent_" + st["status"])
            threading.Thread(target=self.stop, args=(r.pid, "consent"), daemon=True).start()
            return {"status": 403, "code": "consent_revoked", "error": "the consent for this plugin was withdrawn"}
        perm = HOST_CALLS[name]
        if isinstance(perm, tuple):
            perm = next((p for p in perm if p in st["permissions"]), perm[0])
        if perm and perm not in st["permissions"]:
            self.audit(r.pid, "denied", call=name, permission=perm, code="permission_denied")
            return {"status": 403, "code": "permission_denied", "error": "this plugin was not allowed to " + perm}
        if m["edition"] == "supporter" and not gate.run_allowed(self.cfg_dir, m["entitlement"], "plugin", self.cache):
            self.audit(r.pid, "locked", call=name)
            threading.Thread(target=self.stop, args=(r.pid, "grant"), daemon=True).start()
            return {"status": 403, "code": "locked", "error": gate.MESSAGES["revoked"].format(name=m["name"])}
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        if name == "entitlement.state":
            if m["edition"] != "supporter":
                return {"status": 200, "result": {"state": "free"}}
            s = gate.state(self.cfg_dir, m["entitlement"], "plugin", self.cache)
            return {"status": 200, "result": {"state": s["state"]}}
        if name == "notify":
            n = {"plugin": r.pid, "text": str(args.get("text") or "")[:300], "at": int(time.time())}
            self.notices.append(n)
            return {"status": 200, "result": {"ok": True}}
        if name == "model.chat":
            if self.model_chat is None:
                return {"status": 503, "code": "no_model", "error": "no model is running here"}
            return {"status": 200, "result": self.model_chat(args)}
        if name == "model.list":
            if self.model_list is None:
                return {"status": 200, "result": {"models": []}}
            return {"status": 200, "result": self.model_list()}
        if name in ("net.fetch", "net.search"):
            wait = self.net_limit.take(r.pid)
            if wait:
                self.audit(r.pid, "denied", call=name, code="rate_limited")
                return {"status": 429, "code": "rate_limited", "retry_after_s": max(1, int(wait + 0.999)),
                        "error": "too many web calls from this plugin; try again in a moment"}
            try:
                res = (self.web_fetch if name == "net.fetch" else self.web_search)(args)
            except (ValueError, OSError) as e:
                self.audit(r.pid, "net", call=name, ok=False)
                return {"status": 502 if isinstance(e, OSError) else 400, "code": "net_refused" if not isinstance(e, OSError) else "net_error",
                        "error": str(e)[:300] or "the fetch failed"}
            self.audit(r.pid, "net", call=name, ok=True)                 # never the address or the text: the audit log keeps no content
            return {"status": 200, "result": res}
        if name == "gpu.read":
            if self.gpu_read is None:
                return {"status": 503, "code": "no_gpu_read"}
            return {"status": 200, "result": self.gpu_read()}
        if name.startswith("speech."):
            return self._speech(name, args)
        if name.startswith("overlay."):
            if not m.get("overlay"):
                return {"status": 404, "code": "no_overlay", "error": "this plugin has no overlay in its manifest"}
            try:
                return {"status": 200, "result": self.overlays.call(r.pid, name, args)}
            except OverlayError as e:
                return {"status": e.status, "code": e.code, "error": str(e)[:200]}
        if name.startswith("secrets."):
            try:
                res = self.keys.call(r.pid, name, args)
            except KeyStoreError as e:
                return {"status": e.status, "code": e.code, "error": str(e)[:200]}
            if name in ("secrets.put", "secrets.delete"):          # the name only: never a value, never its length class
                self.audit(r.pid, name.replace(".", "_"), key=str(args.get("name") or "")[:48])
            return {"status": 200, "result": res}
        return {"status": 501, "code": "not_available", "error": name + " is not in this PXA Control yet"}

    def _egress_allowed(self, r):
        """the hosts this plugin may reach NOW (consent read again, grant checked): [] closes the door."""
        m = r.manifest
        if m is None or r.state not in ("running", "starting"):
            return []
        st = self.consent.state(m)
        if st["status"] not in ("granted", "partial"):
            return []
        if m["edition"] == "supporter" and not gate.run_allowed(self.cfg_dir, m["entitlement"], "plugin", self.cache):
            return []
        return _egress.scoped_hosts(st["permissions"])

    def _speech_down(self, e):
        return e.__class__.__name__ == "SpeechUnavailable" or isinstance(e, OSError)

    def _speech_up(self):
        """the speech service answers its health check now."""
        try:
            self.speech().health()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _speech_start(self):
        """start the shared speech service (lazy engines, no downloads) once, as Control's own child, and wait up to
        SPEECH_WAIT_S for it to answer. True when it answers."""
        if not self.speech_autostart or self.runtime_missing():
            return False
        from ..speech import setup as SS
        with self._speech_lock:
            # never on engines a setup replaces: its child alive (setup.json may not say running yet) or its state file;
            # _setup_lock: a setup that is starting has stopped the service and is about to start its child
            with self._setup_lock:
                if self._setup_alive() or self._setup_running():
                    return False
                p = self._speech_proc
                if p is None or p.poll() is not None:
                    if time.monotonic() - self._speech_t < SPEECH_RETRY_S and self._speech_t:
                        return self._speech_up()
                    env = self._speech_env()
                    log = os.path.join(self.cfg_dir, "plugins", "speech.log")
                    os.makedirs(os.path.dirname(log), mode=0o700, exist_ok=True)
                    self._speech_t = time.monotonic()
                    try:
                        with open(log, "ab") as lf:
                            os.chmod(log, 0o600)
                            self._speech_proc = p = subprocess.Popen(
                                SS.speech_argv(self.python, "serve", "--lazy", tools=self.tools_dir), cwd=self.tools_dir,
                                env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=lf, **_detached())
                    except OSError:
                        self.audit("host", "speech", ok=False)
                        return False
                    self.audit("host", "speech", ok=True)
            t_end = time.monotonic() + SPEECH_WAIT_S
            while time.monotonic() < t_end and p.poll() is None:
                if self._speech_up():
                    return True
                time.sleep(0.25)
            return self._speech_up()

    def _speech_env(self, net=False):
        """a speech process's environment: ENV_KEEP, the tools, this config and the speech service's own knobs; net: what
        the setup's downloads need too (NET_ENV). Never a token or key."""
        env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
        env.update(PYTHONPATH=self.tools_dir or "", PXA_CONTROL_CONFIG_DIR=self.cfg_dir, PYTHONUNBUFFERED="1")
        env.update({k: v for k, v in os.environ.items() if k.startswith("PXA_SPEECH_") or k == "HF_HOME"})   # its own knobs
        if net:
            env.update({k: v for k, v in os.environ.items() if k.upper() in NET_ENV})
        return env

    # ---- "Speech on this computer" (Plugins tab): the engines and voice models the speech plugins need ---------------------
    def _setup_alive(self):
        """the setup child this Control started has not exited (it may not have written setup.json yet)."""
        p = self._setup_proc
        return p is not None and p.poll() is None

    def _setup_running(self):
        from ..speech import setup as SS
        return (SS.read_job(self.cfg_dir) or {}).get("state") == "running"

    def _speech_status_of(self):
        """`setup --status` in the Python that runs speech (not Control's own interpreter)."""
        from ..speech import setup as SS
        why = self.runtime_missing()
        if not why:
            try:
                out = subprocess.run(SS.speech_argv(self.python, "setup", "--status", tools=self.tools_dir), cwd=self.tools_dir,
                                     env=self._speech_env(), stdin=subprocess.DEVNULL, capture_output=True, timeout=60,
                                     **_no_window())
                st = json.loads(out.stdout.decode("utf-8", "replace"))
                if out.returncode == 0 and isinstance(st, dict) and "supported" in st:
                    return st
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        st = SS.status(self.cfg_dir)                              # the models are this config's; the engines are not known
        st.update(supported=False, why=why or "PXA Control could not start Python for speech.", python=None, engines="missing",
                  ready=False)
        return st

    def speech_setup_status(self):
        """speech.setup.status() as the Python that runs speech sees it: this process when that is Control's own
        interpreter, else `setup --status` in that Python, kept SPEECH_STATUS_S or until a setup ends. The job is read fresh."""
        from ..speech import setup as SS
        if _same_python(self.python):
            return SS.status(self.cfg_dir)
        job = SS.read_job(self.cfg_dir) or {}
        key = (job.get("state"), None if job.get("state") == "running" else job.get("updated"))
        c = self._speech_st
        if c is None or c[1] != key or time.monotonic() - c[0] > SPEECH_STATUS_S:
            c = self._speech_st = (time.monotonic(), key, self._speech_status_of())
        return dict(c[2], job=job or None)

    def speech_setup_start(self, languages=False, force=False):
        """the one button: `pxa_control.speech setup [--languages] [--force]` (setup.speech_argv) as Control's detached child (it
        finishes even if Control closes; speech/setup.json says how it goes, its console goes to setup-console.log). The
        host's speech service is stopped first and not started while the setup runs: Windows cannot replace engines a
        process has open."""
        from ..speech import setup as SS
        with self._setup_lock:
            if self._setup_running() or self._setup_alive():
                raise HostError("a speech setup is already running", "busy", 409)
            st = self.speech_setup_status()
            if not st.get("supported"):
                raise HostError(st.get("why") or "speech cannot run on this computer", "speech_unsupported", 409)
            self._speech_stop()
            sd = SS.speech_dir(self.cfg_dir)
            os.makedirs(sd, mode=0o700, exist_ok=True)
            log = os.path.join(sd, "setup-console.log")
            argv = SS.speech_argv(self.python, "setup", *(["--languages"] if languages else []) + (["--force"] if force else []),
                                  tools=self.tools_dir)
            try:
                with os.fdopen(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as lf:
                    os.chmod(log, 0o600)
                    self._setup_proc = p = subprocess.Popen(argv, cwd=self.tools_dir, env=self._speech_env(net=True),
                                                            stdin=subprocess.DEVNULL, stdout=lf, stderr=lf, **_detached())
            except OSError as e:
                self.audit("host", "speech_setup", ok=False)
                raise HostError("the speech setup did not start: %s" % e, "setup_failed", 500)
            self.audit("host", "speech_setup", ok=True, languages=bool(languages), force=bool(force))
            t_end = time.monotonic() + 5.0                        # its state file, so the tab shows it running at once
            while time.monotonic() < t_end and p.poll() is None and not self._setup_running():
                time.sleep(0.05)
        return {"started": True, "speech": self.speech_setup_status()}

    def _speech_stop(self):
        p, self._speech_proc = self._speech_proc, None
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(STOP_GRACE_S)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except OSError:
                    pass

    def _speech(self, name, args, _again=True):
        """speech.stt {b64: 16 kHz mono s16le, language?: "es"|"auto"} -> {text, ms[, language]}; speech.vad {b64} ->
        {segments, speech_end_s}; speech.tts {text, voice?, speed?} -> {b64, rate}; speech.info {} -> {voices, stt:
        {model, multilingual}}. A service that is down is started once (_speech_start) and the call retried; 503
        speech_unavailable when it is not installed or still not running (the plugin degrades), 502 speech_error for
        the service's own refusal."""
        try:
            c = self.speech()
            if name == "speech.info":
                d = c.info() or {}
                stt, tts = d.get("stt") or {}, d.get("tts") or {}
                voices = [str(v)[:64] for v in (tts.get("voices") or [])][:200]
                return {"status": 200, "result": {"voices": voices, "rate": int(tts.get("rate") or 0) or None,
                                                  "stt": {"model": str(stt.get("model") or "")[:64],
                                                          "multilingual": bool(stt.get("multilingual"))}}}
            if name == "speech.tts":
                text = str(args.get("text") or "")
                if not text.strip() or len(text) > 4000:
                    return {"status": 400, "code": "invalid", "error": "text: 1-4000 characters"}
                speed = args.get("speed")
                pcm = c.tts(text, voice=str(args["voice"]) if args.get("voice") else None,
                            speed=float(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) else None)
                return {"status": 200, "result": {"b64": b64e(pcm), "rate": int(getattr(c, "tts_rate", 24000) or 24000)}}
            pcm = b64d(args.get("b64"))
            if not pcm or len(pcm) > MAX_SPEECH_AUDIO or len(pcm) % 2:
                return {"status": 400, "code": "invalid", "error": "b64: 16 kHz mono 16-bit audio, up to 120 s"}
            if name == "speech.stt":
                lang = args.get("language")
                if lang is not None and not (isinstance(lang, str) and re.fullmatch(r"[a-z]{2,3}|auto", lang)):
                    return {"status": 400, "code": "invalid", "error": "language: an ISO 639-1 code such as 'es', or 'auto'"}
                res = c.transcribe_full(pcm, lang) if lang else c.transcribe_full(pcm)
            else:
                res = c.vad(pcm)
            return {"status": 200, "result": res}
        except ImportError:
            return {"status": 503, "code": "speech_unavailable", "error": "the shared speech service is not in this PXA Control"}
        except Exception as e:  # noqa: BLE001 - SpeechUnavailable / SpeechClientError / the socket
            if self._speech_down(e):
                if _again and self._speech_start():
                    return self._speech(name, args, _again=False)
                return {"status": 503, "code": "speech_unavailable", "error": "the shared speech service is not running"}
            return {"status": 502, "code": "speech_error", "error": str(e)[:200]}

    # ---- watching ----------------------------------------------------------------------------------------------
    def _on_sync(self, cfg_dir, result):
        if cfg_dir == self.cfg_dir:
            self.recheck("sync")

    def recheck(self, why="tick"):
        """stop every running plugin whose grant or consent is gone. -> [stopped ids]"""
        out = []
        for pid, r in list(self.runs.items()):
            m = r.manifest
            if r.state not in ("running", "starting", "restarting") or m is None:
                continue
            if m["edition"] == "supporter" and not gate.run_allowed(self.cfg_dir, m["entitlement"], "plugin", self.cache):
                self.audit(pid, "locked", why=why)
                self.stop(pid, "grant")
                out.append(pid)
            elif self.consent.state(m)["status"] not in ("granted", "partial"):
                self.stop(pid, "consent")
                out.append(pid)
        return out

    def _died(self, r, why):
        now = self.clock()
        self._kill(r)
        self._close(r)
        self.overlays.stopped(r.pid)
        self.audit(r.pid, "exit", why=why)
        while r.restarts and now - r.restarts[0] > self.restart_window_s:
            r.restarts.popleft()
        if len(r.restarts) >= self.restarts_max:
            r.state, r.why, r.restart_at = "failed", why, None
            self.audit(r.pid, "failed", restarts=len(r.restarts))
            return
        r.restarts.append(now)
        r.state, r.why = "restarting", why
        r.restart_at = now + self.backoff[min(len(r.restarts), len(self.backoff)) - 1]

    def tick(self):
        """one watch turn (the loop runs it every tick_s; tests call it)."""
        now = self.clock()
        if now - self._last_recheck >= min(self.health_s, 60.0):     # the grant file is read (and its clock record written)
            self._last_recheck = now
            self.recheck("tick")
        for pid, r in list(self.runs.items()):
            with r.lock:
                if r.state == "running":
                    if r.proc is None or r.proc.poll() is not None:
                        self._died(r, "exited %s" % (r.proc.returncode if r.proc else "?"))
                    elif now - r.last_health >= self.health_s:
                        try:
                            res = r.channel.request({"op": "health"}, self.health_timeout_s)
                            ok = res.get("status") == 200
                        except (ChannelClosed, TimeoutError, AttributeError):
                            ok = False
                        if ok:
                            r.last_health = self.clock()
                        else:
                            self._died(r, "no health answer")
                elif r.state == "restarting" and r.restart_at is not None and now >= r.restart_at:
                    try:
                        m, d = self.manifest(pid)
                        self._check(m)
                        self._spawn(r, m, d)
                        self.audit(pid, "restart", n=len(r.restarts), pid=r.proc.pid)
                    except gate.Locked:
                        r.state, r.why, r.restart_at = "locked", "grant", None
                        self.audit(pid, "locked", why="restart")
                    except PluginError as e:
                        r.state, r.why, r.restart_at = "failed", e.code, None
                        self.audit(pid, "failed", code=e.code)
                    except Exception as e:  # noqa: BLE001
                        self._died(r, e.__class__.__name__)

    def _loop(self):
        while not self._stop.wait(self.tick_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the watcher outlives any one failure
                pass

    def _ensure_loop(self):
        with self._lock:
            if self._thread is None and not self._stop.is_set():
                self._thread = threading.Thread(target=self._loop, daemon=True, name="pxa-plugin-host")
                self._thread.start()

    # ---- what the tab shows -------------------------------------------------------------------------------------
    def status_of(self, pid):
        r = self.runs.get(pid)
        row = {"plugin_id": pid, "state": r.state if r else "stopped", "why": r.why if r else None,
               "pid": r.proc.pid if r and r.proc is not None and r.proc.poll() is None else None,
               "restarts": len(r.restarts) if r else 0}
        eg = r.egress if r is not None else None                 # read once: a stop clears it under r.lock meanwhile
        if eg is not None:
            row["egress"] = dict(eg.view(), mode="sealed" if r.sealed else "proxy")
        return row

    def status(self):
        rows = []
        for it in pxp.installed(self.cfg_dir, self.pubkeys, self.cache):
            row = dict(it, **self.status_of(it["plugin_id"]))
            if it.get("ok"):
                try:
                    m, _d = self.manifest(it["plugin_id"])
                    row["consent"] = self.consent.state(m)
                except PluginError as e:
                    row.update(ok=False, error=e.code)
            rows.append(row)
        return {"plugins": rows, "notices": list(self.notices)}


# ---- the web, for plugins with net.http: the host fetches, the plugin never opens a socket of its own -------------------------
NET_FETCH_S = 30.0
NET_MAX_CHARS = 60000
_TEXTUAL = ("text/", "application/json", "application/xhtml", "application/xml", "application/rss", "application/atom")


def _bounded(v, lo, hi, default):
    """an int in [lo, hi] from a JSON number (bools, NaN, Infinity and anything else -> default)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return default
    return max(lo, min(int(v), hi))


def net_fetch(args):
    """net.fetch {url, max_chars?} -> {url, ctype, title, text, truncated}. http(s) only, no user:password@, never a loopback /
    home-network / link-local / reserved address (checked on the name, on every address it resolves to, pinned for the connect,
    and again on every redirect: pxa_chat's fetch opener), one deadline for the whole page, 2 MiB at most, text pages only.
    The text is screened for injected instructions the way the Assistant's own web_fetch screens it."""
    import types
    import urllib.parse
    import urllib.request
    from pxa_chat import guards as G
    from pxa_chat import tools as T
    url = str(args.get("url") or "").strip()
    if len(url) > 2000 or not url.lower().startswith(("http://", "https://")) or any(c in url for c in " \t\r\n\\"):
        raise ValueError("only http:// and https:// addresses can be opened")
    u = urllib.parse.urlsplit(url)
    if u.username or u.password or "@" in u.netloc:
        raise ValueError("an address with a user name or password is refused")
    if not u.hostname or G.private_host(u.hostname):
        raise ValueError("that address is on this computer or your home network; a plugin cannot open it")
    ctx = types.SimpleNamespace(approved_private=False, resolver=None, interrupted=False, interrupts=[])
    req = urllib.request.Request(url, headers={"User-Agent": "PXA-Control-Plugin/1", "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"})
    ctype, raw = T._fetch_body(req, ctx, NET_FETCH_S)
    low = (ctype or "").lower()
    head = raw[:200].lstrip().lower()
    if low and not low.startswith(_TEXTUAL):
        raise ValueError("that address is not a text page (%s)" % low.split(";")[0][:60])
    if not low and b"\x00" in raw[:4096]:                               # no content type and binary bytes: not a page
        raise ValueError("that address is not a text page")
    cs = "utf-8"
    for part in low.split(";")[1:]:
        k, _s, v = part.strip().partition("=")
        if k == "charset" and v.strip():
            cs = v.strip().strip('"')[:32]
    try:
        text = raw.decode(cs, "replace")
    except LookupError:
        text = raw.decode("utf-8", "replace")
    title = ""
    if "html" in low or head.startswith((b"<!doctype", b"<html")):
        import html as _html
        import re as _re
        m = _re.search(r"(?is)<title[^>]*>(.*?)</title>", text[:200000])
        title = _re.sub(r"\s+", " ", _html.unescape(m.group(1))).strip() if m else ""
        text = T.html_text(text)
    mx = _bounded(args.get("max_chars"), 1000, NET_MAX_CHARS, 20000)
    return {"url": url, "ctype": low.split(";")[0][:80], "title": G.screen_injected(title[:300]), "text": G.screen_injected(text[:mx]),
            "truncated": len(text) > mx}


def _search_cfg():
    """the Assistant's web-search settings (Settings > Web search) when its hub is up, else the built-in defaults."""
    from pxa_chat import search as WS
    try:
        import pxa_chat
        for h in list(getattr(pxa_chat, "_hubs", {}).values()):
            return h.search_cfg()
    except Exception:  # noqa: BLE001
        pass
    return WS.config(None)


def net_search(args):
    """net.search {query, n?} -> {query, results: [{title, url, snippet}], note}: the same providers and settings as the Assistant."""
    from pxa_chat import guards as G
    from pxa_chat import search as WS
    q = " ".join(str(args.get("query") or "").split())[:300]
    if not q:
        raise ValueError("query: required")
    n = _bounded(args.get("n"), 1, 10, 5)
    res, note = WS.search(q, n, _search_cfg())
    rows = []
    for x in (res or [])[:n]:
        link = str(x.get("url") or "")
        if not link.lower().startswith(("http://", "https://")) or len(link) > 2000:
            continue
        rows.append({"title": G.screen_injected(str(x.get("title") or "")[:300]), "url": link,
                     "snippet": G.screen_injected(str(x.get("snippet") or "")[:400])})
    return {"query": q, "results": rows, "note": str(note or "")[:300]}


def _speech_client():
    """the shared speech service's client, imported only when a plugin asks (works without it: ImportError -> 503)."""
    from ..speech.client import SpeechClient
    return SpeechClient()


def _same_python(p):
    """p is the interpreter running Control (a compiled Control's sys.executable is Control itself)."""
    try:
        return os.path.samefile(p, sys.executable)
    except (OSError, TypeError, ValueError):
        return False


def _no_window():
    return {"creationflags": NO_WINDOW} if WINDOWS else {}


def _detached():
    """Popen options for a child that outlives the request: its own session or process group, no console window."""
    if WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | NO_WINDOW}
    return {"start_new_session": True}


# ---- Control wiring ------------------------------------------------------------------------------------------------
def _running_seats(app):
    """[(sid, seat)] of the seats with a running server, in Control's order."""
    out = []
    for sid, seat in list(getattr(app, "seats", {}).items()):
        try:
            if seat.req and seat.running() and seat.port():
                out.append((str(getattr(seat, "sid", sid) or sid), seat))
        except Exception:  # noqa: BLE001
            continue
    return out


def _seat_list(app):
    """model.list -> {"models": [{seat, model, gpus, ctx}]}: what a plugin may pick with model.chat {seat}. The model's file name only
    (never its folder), the cards it runs on and its context; no port, no command line, no environment."""
    def models():
        out = []
        for sid, seat in _running_seats(app):
            req = seat.req or {}
            name = os.path.basename(str(req.get("model") or "")) or sid
            ctx = req.get("ctx") or req.get("ctx_size") or req.get("c")
            out.append({"seat": sid, "model": name[:200], "gpus": req.get("gpus"),
                        "ctx": int(ctx) if isinstance(ctx, (int, float)) and not isinstance(ctx, bool) else None})
        return {"models": out}
    return models


# what a model.chat call may set on the seat's request (sampling, tools for a home-assistant style agent, and the prompt-cache
# knobs: a plugin that resends one long prompt every turn keeps it cached in one pinned slot). Nothing else crosses.
CHAT_ARGS = ("max_tokens", "temperature", "top_p", "stop", "seed", "tools", "tool_choice", "cache_prompt", "id_slot")
SEAT_CHAT_TIMEOUT_S = 300
SEAT_ERROR_MAX = 200
_CTX_WORDS = ("exceed_context_size", "context_length_exceeded", "context size", "context length")


def _seat_error_text(raw):
    """the server's own words for a failed request: the JSON body's error.message, else the raw body; <= 200 characters,
    never a header."""
    t = (raw or b"").decode("utf-8", "replace")
    typ = ""
    try:
        j = json.loads(t)
        e = j.get("error") if isinstance(j, dict) else None
        if isinstance(e, dict):
            typ = str(e.get("type") or "")
            if e.get("message"):
                t = str(e["message"])
        elif isinstance(e, str) and e:
            t = e
    except ValueError:
        pass
    return " ".join(t.split())[:SEAT_ERROR_MAX], typ


def _seat_failure(e):
    """an exception from the seat request -> {error, code} (seat_loading / too_long / seat_error / seat_down / seat_slow),
    the shape no_seat / no_model already use."""
    import http.client
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        try:
            raw = e.read(65536)
        except (OSError, http.client.HTTPException):
            raw = b""
        text, typ = _seat_error_text(raw)
        text = text or ("the server answered HTTP %d" % e.code)
        if e.code == 503:
            return {"error": text, "code": "seat_loading"}
        if e.code in (400, 500) and any(w in (text + " " + typ).lower() for w in _CTX_WORDS):
            return {"error": text, "code": "too_long"}
        return {"error": text, "code": "seat_error"}
    reason = e.reason if isinstance(e, urllib.error.URLError) else e
    if isinstance(reason, TimeoutError):            # socket.timeout is TimeoutError
        return {"error": "the server did not answer within %g s" % SEAT_CHAT_TIMEOUT_S, "code": "seat_slow"}
    if isinstance(e, http.client.HTTPException):
        return {"error": ("the server's answer was broken (%s)" % e.__class__.__name__)[:SEAT_ERROR_MAX], "code": "seat_error"}
    return {"error": ("the server is not reachable (%s)" % (getattr(reason, "strerror", None) or reason.__class__.__name__))
            [:SEAT_ERROR_MAX], "code": "seat_down"}


def _seat_chat(app):
    """model.chat -> a running seat's /v1/chat/completions (loopback), non-streaming: args.seat (one model.list named) or the first."""
    def chat(args):
        import http.client
        import urllib.request
        from ..core import _NO_PROXY_OPENER        # loopback: never through http_proxy / the system proxy
        port = None
        want = args.get("seat")
        for sid, seat in _running_seats(app):
            if want is None or str(want) == sid:
                port = seat.port()
                break
        if not port:
            if want is not None:
                return {"error": "that server is not running", "code": "no_seat"}
            return {"error": "no model is running here", "code": "no_model"}
        body = {"messages": args.get("messages") or [], "stream": False}
        for k in CHAT_ARGS:
            if k in args:
                body[k] = args[k]
        rf = args.get("response_format")      # structured output (Scribe's notes): a JSON object or schema, nothing else
        if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema") and len(json.dumps(rf)) <= 16384:
            body["response_format"] = rf
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % int(port), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with _NO_PROXY_OPENER.open(req, timeout=SEAT_CHAT_TIMEOUT_S) as r:
                raw = r.read()
        except (OSError, http.client.HTTPException) as e:   # HTTPError and URLError are OSErrors
            return _seat_failure(e)
        return json.loads(raw or b"{}")
    return chat


_HOST_LOCK = threading.Lock()


def host_of(app):
    """the app's plugin host (made on first use, once: two first requests at the same moment get the same host)."""
    h = getattr(app, "plugin_host", None)
    if h is None:
        with _HOST_LOCK:
            h = getattr(app, "plugin_host", None)
            if h is None:
                h = _make_host(app)
    if h.overlays.base is None and isinstance(getattr(app, "port", None), int):
        h.overlays.base = "http://127.0.0.1:%d" % app.port
    return h


def _make_host(app):
    """a new PluginHost wired to this Control (host_of holds _HOST_LOCK)."""
    from ..core import config_dir
    from .. import store as S
    cache = getattr(app, "store_cache", None)
    if cache is None:
        cache = app.store_cache = S.EntitlementCache(config_dir())
    gpu = getattr(app, "gpu_state", None)
    h = app.plugin_host = PluginHost(config_dir(), cache=cache, model_chat=_seat_chat(app), gpu_read=gpu,
                                     model_list=_seat_list(app))
    return h


def _who_local():
    try:
        from ..core import _REQ
        who = str(getattr(_REQ, "who", "") or "")
    except Exception:  # noqa: BLE001
        who = ""
    return who.endswith("127.0.0.1") or who.endswith("::1")


def register(ROUTES):
    from ..core import Invalid

    def wrap(fn):
        def r(app, body, query):
            try:
                return fn(host_of(app), app, body if isinstance(body, dict) else {}, query)
            except PluginError as e:
                raise Invalid(str(e), status=getattr(e, "status", 400), code=e.code)
        return r

    def pid_of(body):
        pid = str(body.get("plugin_id") or "")
        if not _ID_RE.match(pid) or pid in RESERVED_IDS:
            raise HostError("plugin_id please", "invalid", 400)
        return pid

    def r_status(h, app, body, query):
        return h.status()

    def r_start(h, app, body, query):
        pid = pid_of(body)
        out = h.start(pid)
        h.set_disabled(pid, False)                 # the user turned it on: it starts with Control again
        return out

    def r_stop(h, app, body, query):
        pid = pid_of(body)
        out = h.stop(pid)
        if h.knows(pid):
            h.set_disabled(pid, True)              # Deactivate stays off across Control restarts
        return out

    def r_uninstall(h, app, body, query):
        return h.uninstall(pid_of(body))

    def r_consent(h, app, body, query):
        """the consent card's answer. On --lan only a page on this computer may grant (design s3)."""
        if getattr(app, "lan", False) and not _who_local():
            raise HostError("consent is given at the computer itself", "consent_local_only", 403)
        pid = pid_of(body)
        m, _d = h.manifest(pid)
        perms = body.get("permissions")
        if not isinstance(perms, list):
            raise HostError("permissions: the list you allow", "invalid", 400)
        rec = h.consent.grant(m, [str(x) for x in perms], by="page")
        h.audit(pid, "consent", permissions=",".join(rec["permissions"]))
        return {"consent": h.consent.state(m)}

    def r_consent_revoke(h, app, body, query):
        pid = pid_of(body)
        h.consent.revoke(pid)
        h.audit(pid, "consent_revoked")
        return h.stop(pid, "consent")

    def r_speech(h, app, body, query):
        return h.speech_setup_status()

    def r_speech_setup(h, app, body, query):
        """"Speech on this computer": it downloads and installs on this computer, so on --lan only a page on this computer
        may start it (like consent)."""
        if getattr(app, "lan", False) and not _who_local():
            raise HostError("the speech setup is started at the computer itself", "setup_local_only", 403)
        return h.speech_setup_start(languages=bool(body.get("languages")), force=bool(body.get("force")))

    ROUTES[("GET", "/api/plugins/host")] = wrap(r_status)
    ROUTES[("POST", "/api/plugins/host/start")] = wrap(r_start)
    ROUTES[("POST", "/api/plugins/host/stop")] = wrap(r_stop)
    ROUTES[("POST", "/api/plugins/uninstall")] = wrap(r_uninstall)
    ROUTES[("POST", "/api/plugins/host/consent")] = wrap(r_consent)
    def r_keys_forget(h, app, body, query):
        """the user's Forget keys: every key this plugin kept in Control's key store (names only in the audit)."""
        pid = pid_of(body)
        names = h.keys.names(pid)
        h.keys.forget(pid)
        h.audit(pid, "keys_forgotten", n=len(names))
        return {"plugin_id": pid, "forgotten": len(names)}

    ROUTES[("POST", "/api/plugins/host/consent/revoke")] = wrap(r_consent_revoke)
    ROUTES[("POST", "/api/plugins/host/keys/forget")] = wrap(r_keys_forget)
    ROUTES[("GET", "/api/plugins/host/speech")] = wrap(r_speech)
    ROUTES[("POST", "/api/plugins/host/speech/setup")] = wrap(r_speech_setup)
    from . import ui as _ui                    # the Plugins tab: GET .../cards and the sandboxed page .../frame (S3)
    _ui.register(ROUTES, wrap)
    return ROUTES
