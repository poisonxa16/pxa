"""pxa-plugin/1: the wire between PXA Control's plugin host and one plugin process (PLUGINS-PROTOTYPE-PLAN S2).
Standard library only, no import from pxa_control: a plugin copies this file into its own package (lib/) and the host
imports the same Channel, so both sides speak the same frames.

Transport: the host listens on a per-plugin unix socket (POSIX: inside a 0700 folder, the peer's uid and process group
checked) or a named pipe (Windows), passed to the child as PXA_PLUGIN_SOCKET. The child connects with
multiprocessing.connection.Client and the one-time 32-byte key in PXA_PLUGIN_KEY (HMAC challenge both ways), so only the
process the host started can talk on it. One connection carries both directions.

Frames: one JSON object per message (multiprocessing.connection length-prefixed bytes, <= MAX_FRAME):
    {"t": "req", "id": <int>, "op": ..., ...}      a request (either side)
    {"t": "res", "id": <same>, "status": <http-like int>, ...}
host -> plugin ops: "health" -> {status 200}; "stop" -> {status 200}, then the plugin exits;
                    "http" {method, path, query, ctype, b64} -> {status, ctype, b64} (a /api/plugins/<id>/* request that
                    already passed Control's token and Origin checks).
plugin -> host ops: "hello" {protocol, plugin} (first, within 10 s) -> {status 200, permissions, listen};
                    "call" {name, args} -> {status 200, result} or {status 403, code permission_denied, error} ...
Bodies travel as base64 ("b64"). The host never sends Control's token, cookies or other headers.
"""
import base64
import json
import os
import threading

PROTOCOL = "pxa-plugin/1"
MAX_FRAME = 32 << 20


class ChannelClosed(Exception):
    pass


def b64e(data):
    if data is None:
        return ""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.b64encode(data).decode("ascii")


def b64d(text):
    try:
        return base64.b64decode(text or "", validate=False)
    except (ValueError, TypeError):
        return b""


class Channel(object):
    """request/response over one multiprocessing Connection, both directions; requests from the other side run on
    worker threads (at most `workers` at once; more get 503 busy)."""

    def __init__(self, conn, on_request=None, workers=8):
        self.conn = conn
        self.on_request = on_request
        self._send_lock = threading.Lock()
        self._pending = {}
        self._plock = threading.Lock()
        self._next = 0
        self._sem = threading.BoundedSemaphore(workers)
        self.closed = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._reader, daemon=True, name="pxa-plugin-channel")
        self._thread.start()
        return self

    def send(self, msg):
        data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_FRAME:
            raise ValueError("frame too large")
        with self._send_lock:
            if self.closed.is_set():
                raise ChannelClosed("closed")
            try:
                self.conn.send_bytes(data)
            except (OSError, EOFError, ValueError) as e:
                self.close()
                raise ChannelClosed(str(e))

    def request(self, msg, timeout=60.0):
        ev = threading.Event()
        with self._plock:
            self._next += 1
            rid = self._next
            self._pending[rid] = [ev, None]
        try:
            self.send(dict(msg, t="req", id=rid))
            if not ev.wait(timeout):
                raise TimeoutError("no answer in %.0f s" % timeout)
            res = self._pending.get(rid, [None, None])[1]
            if res is None:
                raise ChannelClosed("closed")
            return res
        finally:
            with self._plock:
                self._pending.pop(rid, None)

    def _reader(self):
        while not self.closed.is_set():
            try:
                raw = self.conn.recv_bytes(MAX_FRAME)
                msg = json.loads(raw.decode("utf-8"))
            except (EOFError, OSError, ValueError, TypeError):
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("t") == "res":
                with self._plock:
                    p = self._pending.get(msg.get("id"))
                if p is not None:
                    p[1] = msg
                    p[0].set()
            elif msg.get("t") == "req":
                if not self._sem.acquire(blocking=False):
                    self._reply(msg, {"status": 503, "code": "busy", "error": "too many requests at once"})
                    continue
                threading.Thread(target=self._serve, args=(msg,), daemon=True).start()
        self.close()

    def _reply(self, msg, out):
        try:
            self.send(dict(out, t="res", id=msg.get("id")))
        except (ChannelClosed, ValueError):
            pass

    def _serve(self, msg):
        try:
            out = self.on_request(msg) if self.on_request else {"status": 404, "code": "unknown_op"}
        except Exception as e:  # noqa: BLE001 - one bad request never kills the channel
            out = {"status": 500, "code": "internal", "error": "%s" % e.__class__.__name__}
        finally:
            self._sem.release()
        self._reply(msg, out if isinstance(out, dict) else {"status": 500, "code": "internal"})

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.conn.close()
        except OSError:
            pass
        with self._plock:
            for p in self._pending.values():
                p[0].set()


def connect(address=None, key=None):
    """the child's connection to its host (PXA_PLUGIN_SOCKET + PXA_PLUGIN_KEY)."""
    from multiprocessing.connection import Client
    address = address or os.environ["PXA_PLUGIN_SOCKET"]
    key = key if key is not None else bytes.fromhex(os.environ["PXA_PLUGIN_KEY"])
    family = "AF_PIPE" if address.startswith("\\\\.\\pipe\\") else "AF_UNIX"
    return Client(address, family=family, authkey=key)


class Plugin(object):
    """the plugin side in a few lines:

        def handle(method, path, query, ctype, body):        # path = /api/plugins/<id>/...
            return 200, "application/json", {"ok": True}       # body: dict (JSON), str or bytes
        p = Plugin(handle); p.run()                            # returns when the host says stop or goes away
        p.call("model.chat", {...})                            # from any thread while running -> the host's answer
    """

    def __init__(self, handler, on_stop=None):
        self.handler = handler
        self.on_stop = on_stop
        self.channel = None
        self.hello = None
        self._stop = threading.Event()

    def run(self, address=None, key=None):
        self.channel = Channel(connect(address, key), self._on_request).start()
        self.hello = self.channel.request({"op": "hello", "protocol": PROTOCOL, "plugin": os.environ.get("PXA_PLUGIN_ID", "")}, 10.0)
        if self.hello.get("status") != 200:
            self.channel.close()
            return 3
        while not self._stop.is_set() and not self.channel.closed.is_set():
            self._stop.wait(0.5)
        self.channel.close()
        return 0

    def call(self, name, args=None, timeout=120.0):
        return self.channel.request({"op": "call", "name": name, "args": args or {}}, timeout)

    def _on_request(self, msg):
        op = msg.get("op")
        if op == "health":
            return {"status": 200}
        if op == "stop":
            if self.on_stop:
                try:
                    self.on_stop()
                except Exception:  # noqa: BLE001
                    pass
            threading.Timer(0.05, self._stop.set).start()
            return {"status": 200}
        if op == "http":
            status, ctype, body = self.handler(msg.get("method"), msg.get("path"), msg.get("query") or "",
                                               msg.get("ctype") or "", b64d(msg.get("b64")))
            if isinstance(body, (dict, list)):
                body, ctype = json.dumps(body).encode("utf-8"), ctype or "application/json"
            return {"status": int(status), "ctype": ctype or "application/octet-stream", "b64": b64e(body)}
        return {"status": 404, "code": "unknown_op"}
