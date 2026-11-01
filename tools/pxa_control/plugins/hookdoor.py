"""The hook door: a program on this computer wakes a plugin with one signed call, served by PXA Control itself.

    POST /plugins/<id>/hook/<trigger id: 32 lowercase hex>      headers X-Hive-Time and X-Hive-Sign, body at most 16384 bytes

A script cannot hold Control's token, so this is not an /api route. server.py sends this one path shape here from inside its
/plugins/ branch: AFTER its Host check, BEFORE its Origin/fetch-metadata and token checks; every other /plugins/ path still goes
to overlay.serve. The rules, in the order serve() checks them (tests/test-pxa-plugin-hookdoor.py has a test for each):

* loopback only: the peer is this computer AND the one Host header names it, on every bind (a --lan Control too). A loopback
  peer with any other Host is DNS rebinding and is refused (403);
* POST only (405; PUT and the rest never reach Control's dispatch); a request that carries an Origin header, or a browser's
  Sec-Fetch-Site other than "none", is refused (403): no web page can call the door;
* Content-Length is required (411; a chunked body is refused) and at most 16384 (413, answered before a byte of the body is
  read); the body is read to exactly that length (400 when it ends early);
* off by default: 404 unless the plugin is running, its signed manifest says "hook_door": true, the user turned the door on for
  it in Control (<config>/plugins/hook-door.json, POST /api/plugins/host/hookdoor: never the plugin), and its consent (and, for
  a supporter plugin, its grant) is in force. An outside call never starts a plugin;
* the failure limit: the door counts refusals per plugin (the plugin's: bad signature, unknown id, replay; and its own for a
  malformed signature); FAIL_MAX of them inside FAIL_WINDOW_S close that plugin's door for CLOSED_S: 429 with Retry-After,
  nothing forwarded;
* the two signature headers must each be there once and well formed (a time of 1-12 digits, 64 hex characters), else 401 like
  any bad signature. Cookie, X-PXA-Token, Authorization and the query string are never read and never forwarded: Control's
  token never opens the door. The trigger's signature is the only credential and only the plugin checks it (it holds the
  keys); the door never sees a key;
* at most IN_FLIGHT calls per plugin are with the plugin at once (503 busy beyond that), so a burst cannot outrun the count;
* the call goes to the plugin as the host op "hook" {trigger, ts, sig, ctype, b64} (PluginHost.hook), never the "http" op and
  never an /api/plugins route; the plugin's answer comes back with no-store and a sandbox CSP, never as HTML;
* no new port, no listener, no thread: the door runs on the request's own thread.
Nothing here writes a signature, a body or a key to a log line, an audit row or an error text.
"""
import collections
import json
import os
import re
import sys
import threading
import time

from . import PluginError, _ID_RE
from .overlay import host_is_local, peer_is_local

PATH_RE = re.compile(r"^/plugins/([a-z][a-z0-9-]{1,31})/hook/([0-9a-f]{32})\Z")
MAX_BODY = 16384
TIME_HEADER = "X-Hive-Time"
SIGN_HEADER = "X-Hive-Sign"
TIME_RE = re.compile(r"^[0-9]{1,12}\Z")
SIGN_RE = re.compile(r"^[0-9a-fA-F]{64}\Z")
LENGTH_RE = re.compile(r"^[0-9]{1,9}\Z")
FAIL_MAX = 10               # refusals by the plugin ...
FAIL_WINDOW_S = 60.0        # ... inside this many seconds close the plugin's door ...
CLOSED_S = 300.0            # ... for this long
IN_FLIGHT = 4               # calls of one plugin with the plugin at once
SETTING = "hook-door.json"  # in <config>/plugins: {"on": [plugin ids the user turned the door on for]}


def match(path):
    """-> (plugin id, trigger id) when path is the door's one shape, else None (the path is not the door's)."""
    m = PATH_RE.match(path or "")
    return (m.group(1), m.group(2)) if m else None


def declared(manifest):
    """the plugin's signed manifest asks for the door."""
    return isinstance(manifest, dict) and manifest.get("hook_door") is True


class Door(object):
    """the user's door setting per plugin, and each plugin's refusal count, closed time and calls in flight (in memory)."""

    def __init__(self, cfg_dir, clock=time.monotonic):
        self.path = os.path.join(cfg_dir, "plugins", SETTING)
        self.clock = clock
        self.lock = threading.Lock()
        self.fails = {}             # plugin id -> deque of refusal times (clock)
        self.closed = {}            # plugin id -> clock time the door opens again
        self.busy = {}              # plugin id -> calls with the plugin now

    def __repr__(self):
        return "<Door>"

    # ---- the user's setting (default off) ------------------------------------------------------------------------------
    def on_ids(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return []
        ids = d.get("on") if isinstance(d, dict) else None
        return sorted(x for x in ids if isinstance(x, str) and _ID_RE.match(x)) if isinstance(ids, list) else []

    def is_on(self, pid):
        return pid in self.on_ids()

    def set_on(self, pid, on):
        """-> True when the setting changed."""
        with self.lock:
            cur = set(self.on_ids())
            if (pid in cur) == bool(on):
                return False
            cur.add(pid) if on else cur.discard(pid)
            os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
            tmp = self.path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"on": sorted(cur)}, f)
            os.replace(tmp, self.path)
            return True

    def serves(self, pid, manifest):
        """the manifest declares the door and the user turned it on (PluginHost.hook_serves adds: running, consent, grant)."""
        return declared(manifest) and self.is_on(pid)

    # ---- the failure limit -----------------------------------------------------------------------------------------------
    def wait(self, pid):
        """-> whole seconds until the plugin's door opens again; 0 when it is open."""
        with self.lock:
            until = self.closed.get(pid)
            if until is None:
                return 0
            left = until - self.clock()
            if left <= 0:
                self.closed.pop(pid, None)
                return 0
            return max(1, int(left + 0.999))

    def refused(self, pid):
        """the plugin refused a call (or the door did, for a malformed signature). -> True when this one closed the door."""
        with self.lock:
            now = self.clock()
            q = self.fails.get(pid)
            if q is None:
                q = self.fails[pid] = collections.deque()
            while q and now - q[0] >= FAIL_WINDOW_S:
                q.popleft()
            q.append(now)
            if len(q) < FAIL_MAX:
                return False
            q.clear()
            self.closed[pid] = now + CLOSED_S
            return True

    def enter(self, pid):
        with self.lock:
            n = self.busy.get(pid, 0)
            if n >= IN_FLIGHT:
                return False
            self.busy[pid] = n + 1
            return True

    def leave(self, pid):
        with self.lock:
            n = self.busy.get(pid, 0) - 1
            if n > 0:
                self.busy[pid] = n
            else:
                self.busy.pop(pid, None)


def _one(headers, name):
    """the header's value when it is there exactly once, else None (two copies of a header are refused, never picked from)."""
    vals = headers.get_all(name) or []
    return vals[0].strip() if len(vals) == 1 and isinstance(vals[0], str) else None


def _local(handler):
    """the peer is this computer and the one Host header names it (overlay.py's two checks, plus: exactly one Host)."""
    return _one(handler.headers, "Host") is not None and peer_is_local(handler) and host_is_local(handler)


def _length(headers):
    """-> (n, None) or (None, (status, words, code)). Content-Length only: exactly one, plain digits, at most MAX_BODY."""
    if headers.get("Transfer-Encoding") is not None:
        return None, (411, "send the body with a Content-Length (no chunked transfer)", "length_required")
    vals = headers.get_all("Content-Length") or []
    if not vals:
        return None, (411, "a Content-Length is required", "length_required")
    raw = _one(headers, "Content-Length") or ""
    if not LENGTH_RE.match(raw):
        return None, (400, "bad Content-Length", "bad_length")
    n = int(raw)
    if n > MAX_BODY:
        return None, (413, "a call carries at most %d bytes" % MAX_BODY, "too_large")
    return n, None


def counted(status, data):
    """the plugin's refusals the failure limit counts: 401 (bad signature, a time outside the window), 404 (unknown id) and
    409 replayed. A paused trigger, a full queue (429) or a note that could not be left are not the caller guessing."""
    if status in (401, 404):
        return True
    if status != 409:
        return False
    try:
        return json.loads(bytes(data or b"").decode("utf-8")).get("code") == "replayed"
    except (ValueError, AttributeError, TypeError):
        return False


def _send(handler, status, ctype, body, extra=()):
    from .host import PLUGIN_CSP
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Security-Policy", PLUGIN_CSP)
    for k, v in extra:
        handler.send_header(k, v)
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


def _refuse(handler, status, words, code, extra=()):
    body = json.dumps({"error": words, "code": code, "status": status}).encode()
    return _send(handler, status, "application/json; charset=utf-8", body, extra)


def serve(handler, ph, method, pid, trigger):
    """one request on the door's path (server.py: after the Host check, before the Origin and token checks)."""
    handler.close_connection = True               # one call per connection: an unread body is never read as a request
    try:
        return _serve(handler, ph, method, pid, trigger)
    except (BrokenPipeError, ConnectionResetError):
        raise
    except Exception as e:  # noqa: BLE001 - the class name only: an exception text could quote what the caller sent
        sys.stderr.write("pxa-control: the hook door failed (%s)\n" % e.__class__.__name__)
        try:
            return _refuse(handler, 500, "the door failed", "internal")
        except OSError:
            return None


def _serve(handler, ph, method, pid, trigger):
    hd = handler.headers
    if not _local(handler):
        return _refuse(handler, 403, "outside calls are taken from programs on this computer only", "not_local")
    if method != "POST":
        return _refuse(handler, 405, "POST only", "method_not_allowed", (("Allow", "POST"),))
    if hd.get("Origin") is not None or (hd.get("Sec-Fetch-Site") or "none").strip().lower() != "none":
        return _refuse(handler, 403, "a web page cannot call this", "from_a_page")
    n, why = _length(hd)
    if why is not None:
        return _refuse(handler, *why)
    try:
        body = handler.rfile.read(n) if n else b""
    except OSError:                               # the caller stalled past the socket timeout or went away: nobody to answer
        return None
    if len(body) != n:
        return _refuse(handler, 400, "the body ended before its Content-Length", "short_body")
    if not ph.hook_serves(pid):
        return _refuse(handler, 404, "not found", "not_found")
    door = ph.door
    wait = door.wait(pid)
    if wait:
        return _refuse(handler, 429, "too many refused calls: this door is closed for a while", "door_closed",
                       (("Retry-After", str(wait)),))
    ts, sig = _one(hd, TIME_HEADER), _one(hd, SIGN_HEADER)
    if not (ts and TIME_RE.match(ts) and sig and SIGN_RE.match(sig)):
        if door.refused(pid):
            ph.audit(pid, "hook_door_closed", refusals=FAIL_MAX, seconds=int(CLOSED_S))
        return _refuse(handler, 401, "the call is not signed: send %s and %s" % (TIME_HEADER, SIGN_HEADER), "bad_signature")
    if not door.enter(pid):
        return _refuse(handler, 503, "busy: try again in a moment", "busy", (("Retry-After", "1"),))
    try:
        status, ctype, data = ph.hook(pid, trigger, ts, sig, hd.get("Content-Type"), body)
    except PluginError as e:
        return _refuse(handler, getattr(e, "status", 502), str(e), e.code)
    finally:
        door.leave(pid)
    if counted(status, data) and door.refused(pid):
        ph.audit(pid, "hook_door_closed", refusals=FAIL_MAX, seconds=int(CLOSED_S))
    return _send(handler, status, ctype, data)
