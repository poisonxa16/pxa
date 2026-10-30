"""A plugin's OBS overlay, served by PXA Control itself (PLUGINS-PROTOTYPE-PLAN S4; Co-host design s5).

OBS cannot send Control's token, so an overlay is a READ-ONLY capability route, not an /api route:

    GET /plugins/<id>/overlay?k=<key>               the manifest's overlay page (a signed, hashed file of the plugin)
    GET /plugins/<id>/overlay/events?k=<key>        server-sent events the plugin emitted (captions, audio announcements)
    GET /plugins/<id>/overlay/audio/<name>.wav?k=   audio the plugin handed over (RAM only, a small ring)

Rules (server.py checks the first three before anything else; this module the rest):
* the peer is this computer (127.0.0.1 / ::1) and the Host header names it, on EVERY bind: a --lan Control refuses a LAN peer (403);
* the key is per plugin, 128 bits, compared in constant time; it rotates from the plugin (host call overlay.rotate), which drops every
  open stream; it is kept in <config>/plugins/overlay-keys.json (0600) so the OBS URL survives a restart; it is never logged;
* the page runs sandboxed (CSP sandbox allow-scripts: an opaque origin, so it can never use Control's cookie or reach /api),
  events and audio answer it with Access-Control-Allow-Origin: * (the key is the capability, no cookie is read);
* the plugin feeds it with host calls overlay.emit {event} / overlay.audio {name, b64}; when the plugin stops (grant lost, consent
  withdrawn, Stop) the host emits {"type": "idle"} and drops the audio: the overlay shows nothing, no nag on stream.
"""
import collections
import hmac
import json
import os
import re
import secrets
import threading
import time

RING = 24                       # events a late OBS gets on connect
AUDIO_ITEMS, AUDIO_BYTES = 24, 12 << 20
MAX_AUDIO = 4 << 20             # one clip
MAX_EVENT = 8 << 10             # one event, as JSON
MAX_CLIENTS = 8                 # open streams per plugin
HEARTBEAT_S = 15.0
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PAGE_CSP = ("sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; "
            "font-src data:; media-src *; connect-src *; base-uri 'none'; form-action 'none'")
LOOPBACK_PEERS = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


class OverlayError(ValueError):
    def __init__(self, msg, code="invalid", status=400):
        ValueError.__init__(self, msg)
        self.code, self.status = code, status


class Feed(object):
    """one plugin's events (ring + live) and RAM audio."""

    def __init__(self):
        self.cv = threading.Condition()
        self.ring = collections.deque(maxlen=RING)
        self.seq = 0
        self.generation = 0          # bumps on rotate and on close: open streams end
        self.audio = collections.OrderedDict()
        self.audio_bytes = 0
        self.clients = 0

    def emit(self, ev):
        with self.cv:
            self.seq += 1
            ev = dict(ev, n=self.seq)
            self.ring.append(ev)
            self.cv.notify_all()
            return ev

    def put_audio(self, name, data):
        with self.cv:
            old = self.audio.pop(name, None)
            if old is not None:
                self.audio_bytes -= len(old)
            self.audio[name] = data
            self.audio_bytes += len(data)
            while len(self.audio) > AUDIO_ITEMS or self.audio_bytes > AUDIO_BYTES:
                _k, v = self.audio.popitem(last=False)
                self.audio_bytes -= len(v)

    def get_audio(self, name):
        with self.cv:
            return self.audio.get(name)

    def drop(self):
        """the plugin stopped: tell open overlays to show nothing, forget the audio."""
        with self.cv:
            self.audio.clear()
            self.audio_bytes = 0
        self.emit({"type": "idle"})

    def bump(self):
        with self.cv:
            self.generation += 1
            self.cv.notify_all()


class Overlays(object):
    """every plugin's overlay key and feed (the host owns one)."""

    def __init__(self, cfg_dir):
        self.path = os.path.join(cfg_dir, "plugins", "overlay-keys.json")
        self.lock = threading.Lock()
        self.feeds = {}
        self.base = None                 # "http://127.0.0.1:<Control's port>", set by host_of

    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, d):
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(d, f, sort_keys=True)
        os.replace(tmp, self.path)

    def key(self, pid):
        with self.lock:
            d = self._load()
            k = d.get(pid)
            if not (isinstance(k, str) and len(k) >= 22):
                k = d[pid] = secrets.token_urlsafe(16)          # 128 bits
                self._save(d)
            return k

    def rotate(self, pid):
        with self.lock:
            d = self._load()
            d[pid] = secrets.token_urlsafe(16)
            self._save(d)
        self.feed(pid).bump()
        return d[pid]

    def check(self, pid, k):
        want = self.key(pid)
        return bool(k) and hmac.compare_digest(str(k).encode("utf-8", "replace"), want.encode())

    def feed(self, pid):
        with self.lock:
            f = self.feeds.get(pid)
            if f is None:
                f = self.feeds[pid] = Feed()
            return f

    def url(self, pid):
        path = "/plugins/%s/overlay?k=%s" % (pid, self.key(pid))
        return {"url": (self.base or "") + path, "path": path, "scope": "loopback"}

    def stopped(self, pid):
        f = self.feeds.get(pid)
        if f is not None:
            f.drop()

    # ---- host calls ---------------------------------------------------------------------------------------------------
    def call(self, pid, name, args):
        """overlay.emit {event} / overlay.audio {name, b64} / overlay.url / overlay.rotate -> result dict."""
        from .child import b64d
        if name == "overlay.emit":
            ev = args.get("event")
            if not isinstance(ev, dict) or not isinstance(ev.get("type"), str):
                raise OverlayError("event: an object with a type")
            if len(json.dumps(ev)) > MAX_EVENT:
                raise OverlayError("event: at most %d bytes" % MAX_EVENT, "too_large", 413)
            ev = {k: v for k, v in ev.items() if k != "n"}
            return {"n": self.feed(pid).emit(ev)["n"]}
        if name == "overlay.audio":
            nm = str(args.get("name") or "")
            if not NAME_RE.match(nm):
                raise OverlayError("name: 1-64 of A-Z a-z 0-9 _ -")
            data = b64d(args.get("b64"))
            if not data or len(data) > MAX_AUDIO or data[:4] != b"RIFF":
                raise OverlayError("b64: one WAV clip up to %d bytes" % MAX_AUDIO, "too_large" if len(data) > MAX_AUDIO else "invalid")
            self.feed(pid).put_audio(nm, data)
            return {"ok": True}
        if name == "overlay.url":
            return self.url(pid)
        if name == "overlay.rotate":
            self.rotate(pid)
            return self.url(pid)
        raise OverlayError("no such overlay call", "unknown_call", 404)


def peer_is_local(handler):
    a = handler.client_address[0] if handler.client_address else ""
    return a in LOOPBACK_PEERS


def host_is_local(handler):
    h = (handler.headers.get("Host") or "").strip().lower()
    if h.startswith("["):
        name = h.split("]")[0] + "]"
    else:
        name = h.rsplit(":", 1)[0]
    return name in ("127.0.0.1", "localhost", "[::1]")


def _plain(handler, code, text):
    body = (text + "\n").encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


def serve(handler, ph, method, path, query):
    """one /plugins/<id>/overlay* request (server.py calls this before the token check: the key is the capability)."""
    if method not in ("GET", "HEAD"):
        return _plain(handler, 405, "read only")
    if not peer_is_local(handler) or not host_is_local(handler):
        return _plain(handler, 403, "an overlay is served to this computer only")
    parts = path.split("/")                     # ["", "plugins", id, "overlay", ...]
    pid = parts[2] if len(parts) > 3 else ""
    if len(parts) < 4 or parts[3] != "overlay" or not ph.knows(pid):
        return _plain(handler, 404, "not found")
    try:
        m, d = ph.manifest(pid)
    except Exception:  # noqa: BLE001 - not installed / does not verify: nothing to show
        return _plain(handler, 404, "not found")
    ov = m.get("overlay")
    if not ov:
        return _plain(handler, 404, "not found")
    k = (query.get("k") or [""])[0]
    if not ph.overlays.check(pid, k):
        return _plain(handler, 403, "forbidden")
    rest = "/".join(parts[4:])
    if rest == "":
        return _page(handler, m, d, ov)
    if rest == "events" and method == "GET":
        return _events(handler, ph.overlays, pid)
    if rest.startswith("audio/") and rest.endswith(".wav") and rest.count("/") == 1:
        nm = rest[6:-4]
        data = ph.overlays.feed(pid).get_audio(nm) if NAME_RE.match(nm) else None
        if data is None:
            return _plain(handler, 404, "gone")
        return _send(handler, 200, "audio/wav", data)
    return _plain(handler, 404, "not found")


def _send(handler, code, ctype, body, extra=()):
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    for a, b in extra:
        handler.send_header(a, b)
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


def _page(handler, m, d, ov):
    page = ov["page"]
    if page not in (m.get("files") or {}):
        return _plain(handler, 404, "not found")
    p = os.path.realpath(os.path.join(d, *page.split("/")))
    if not p.startswith(os.path.realpath(d) + os.sep):
        return _plain(handler, 404, "not found")
    try:
        with open(p, "rb") as f:
            body = f.read(1 << 20)
    except OSError:
        return _plain(handler, 404, "not found")
    return _send(handler, 200, "text/html; charset=utf-8", body, [("Content-Security-Policy", PAGE_CSP)])


def _events(handler, overlays, pid):
    feed = overlays.feed(pid)
    with feed.cv:
        if feed.clients >= MAX_CLIENTS:
            busy = True
        else:
            busy = False
            feed.clients += 1
        gen = feed.generation
    if busy:
        return _plain(handler, 503, "too many overlays open")
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("X-Accel-Buffering", "no")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        try:
            last = int(handler.headers.get("Last-Event-ID") or 0)
        except ValueError:
            last = 0
        handler.wfile.write(b"retry: 2000\n\n")
        handler.wfile.flush()
        beat = time.monotonic()
        while True:
            with feed.cv:
                pending = [e for e in feed.ring if e["n"] > last]
                if not pending and feed.generation == gen:
                    feed.cv.wait(1.0)
                    pending = [e for e in feed.ring if e["n"] > last]
                gone = feed.generation != gen
            if gone:
                return
            for e in pending:
                handler.wfile.write(("id: %d\ndata: %s\n\n" % (e["n"], json.dumps(e))).encode("utf-8"))
                last = e["n"]
            if pending:
                handler.wfile.flush()
                beat = time.monotonic()
            elif time.monotonic() - beat > HEARTBEAT_S:
                handler.wfile.write(b": ping\n\n")
                handler.wfile.flush()
                beat = time.monotonic()
    except (OSError, ValueError):
        return
    finally:
        with feed.cv:
            feed.clients -= 1
