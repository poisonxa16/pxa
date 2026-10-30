"""The local speech API (pxa-speech/1): plain HTTP on loopback, one warm SpeechService behind it.

Every request except /health carries ``Authorization: Bearer <token>``; the token lives in a 0600 file
(<config>/speech/token) and is read BY PATH by clients, never printed or logged. Requests are refused when the peer
is not loopback, the Host header is not a loopback name (DNS rebinding) or an Origin header is present (browsers).
Audio in = 16 kHz mono s16le, raw (``audio/L16``) or a WAV with exactly that format.

  GET    /v1/speech/health                        -> {"ok": true, "api": "pxa-speech/1"}
  GET    /v1/speech/info                          -> engines, rates, voices, load/warm times, counters
  POST   /v1/speech/stt[?language=es]   <audio>   -> {"text", "ms"[, "language"]}          batch, one utterance
  POST   /v1/speech/stt/sessions        {json}    -> {"id"}   json: {"partial_s": 0, "hang_ms": 300, "language"?}
  POST   /v1/speech/stt/sessions/<id>   <audio>   -> {"events": ["start"|"end"...], "partial"}
  POST   /v1/speech/stt/sessions/<id>/finish      -> {"text", "reused", "ms", "audio_s"}  (closes the session)
  DELETE /v1/speech/stt/sessions/<id>             -> {"closed": bool}
  POST   /v1/speech/tts                 {json}    -> 200 chunked audio/L16, one HTTP chunk per clause, FIRST CLAUSE
                                                     FIRST; json: {"text", "voice"?, "speed"?}; X-Speech-Rate header
  POST   /v1/speech/vad                 <audio>   -> {"segments": [[start_s, end_s]...], "speech_end_s"}
"""
import hmac
import io
import json
import os
import secrets
import socket
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import registry
from .service import MAX_AUDIO_S, SpeechError
from .tts import RATE

API = "pxa-speech/1"
LOOPBACK = ("127.0.0.1", "::1", "localhost")
MAX_BODY = MAX_AUDIO_S * RATE * 2 + 4096


def speech_dir():
    return os.path.join(registry.config_dir(), "speech")


def token_path():
    return os.environ.get("PXA_SPEECH_TOKEN_FILE") or os.path.join(speech_dir(), "token")


def discovery_path():
    return os.path.join(speech_dir(), "service.json")


def _write_private(path, data):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def ensure_token(path=None):
    """-> the token file path (created 0600 with a fresh random token when missing)."""
    path = path or token_path()
    if not os.path.isfile(path):
        _write_private(path, secrets.token_urlsafe(32))
    return path


def read_token(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def pcm_from_body(body, ctype):
    """raw L16 or a WAV (16 kHz mono 16-bit only) -> PCM bytes."""
    ctype = (ctype or "").split(";")[0].strip().lower()
    if ctype in ("audio/wav", "audio/x-wav", "audio/wave") or body[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(body), "rb") as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
                    raise SpeechError("WAV must be 16 kHz mono 16-bit")
                return w.readframes(w.getnframes())
        except (wave.Error, EOFError):
            raise SpeechError("not a readable WAV")
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pxa-speech"
    sys_version = ""

    def log_message(self, fmt, *args):      # no request logging (audio/text never reach a log)
        pass

    # ---- helpers -----------------------------------------------------------------------------------------
    def _json(self, code, obj):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg):
        self._json(code, {"error": msg})

    def _allowed(self, need_auth=True):
        if self.client_address[0] not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            self._err(403, "loopback only")
            return False
        host = (self.headers.get("Host") or "").strip().lower()
        hname = host.rsplit(":", 1)[0].strip("[]") if not host.startswith("[") else host[1:host.find("]")]
        if hname not in LOOPBACK:
            self._err(403, "bad host")
            return False
        if self.headers.get("Origin"):
            self._err(403, "browser requests are not accepted")
            return False
        if need_auth:
            got = self.headers.get("Authorization") or ""
            want = "Bearer " + self.server.token
            if not hmac.compare_digest(got.encode("utf-8"), want.encode("utf-8")):
                self._err(401, "missing or wrong token")
                return False
        return True

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            raise SpeechError("body too large or no Content-Length")
        return self.rfile.read(n) if n else b""

    def _jbody(self):
        b = self._body()
        if not b:
            return {}
        try:
            d = json.loads(b.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise SpeechError("bad JSON")
        if not isinstance(d, dict):
            raise SpeechError("JSON object expected")
        return d

    def _query(self):
        """?language=es -> {"language": "es"} (last value wins; values are checked by the service)."""
        from urllib.parse import parse_qsl
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        return dict(parse_qsl(q, keep_blank_values=False)[:8])

    # ---- routes ------------------------------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/speech/health":
            if self._allowed(need_auth=False):
                self._json(200, {"ok": True, "api": API})
            return
        if not self._allowed():
            return
        if path == "/v1/speech/info":
            return self._json(200, self.server.svc.info())
        self._err(404, "not found")

    def do_DELETE(self):
        if not self._allowed():
            return
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if len(parts) == 5 and parts[:4] == ["v1", "speech", "stt", "sessions"]:
            return self._json(200, {"closed": self.server.svc.close_session(parts[4])})
        self._err(404, "not found")

    def do_POST(self):
        if not self._allowed():
            return
        svc = self.server.svc
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        try:
            if parts[:2] != ["v1", "speech"]:
                raise _NotFound()
            rest = parts[2:]
            if rest == ["stt"]:
                return self._json(200, svc.transcribe(pcm_from_body(self._body(), self.headers.get("Content-Type")),
                                                      language=self._query().get("language")))
            if rest == ["stt", "sessions"]:
                d = self._jbody()
                return self._json(200, {"id": svc.open_session(d.get("partial_s", 0), d.get("hang_ms", 300),
                                                               language=d.get("language"))})
            if len(rest) == 3 and rest[:2] == ["stt", "sessions"]:
                return self._json(200, svc.feed(rest[2], pcm_from_body(self._body(),
                                                                        self.headers.get("Content-Type"))))
            if len(rest) == 4 and rest[:2] == ["stt", "sessions"] and rest[3] == "finish":
                self._body()
                return self._json(200, svc.finish(rest[2]))
            if rest == ["vad"]:
                return self._json(200, svc.vad(pcm_from_body(self._body(), self.headers.get("Content-Type"))))
            if rest == ["tts"]:
                return self._tts(self._jbody())
            raise _NotFound()
        except _NotFound:
            self._err(404, "not found")
        except SpeechError as e:
            self._err(getattr(e, "status", 400), str(e))
        except RuntimeError as e:          # engine missing / not fetched
            self._err(503, str(e))

    def _tts(self, d):
        svc = self.server.svc
        it = svc.tts_stream(d.get("text"), voice=d.get("voice"), speed=d.get("speed"))
        try:
            first = next(it)              # errors before the first byte still get a JSON error
        except StopIteration:
            first = None
        rate = svc.tts.rate
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16; rate=%d; channels=1" % rate)
        self.send_header("X-Speech-Rate", str(rate))
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            if first is not None:
                self._chunk(first[1])
                for _clause, pcm in it:
                    self._chunk(pcm)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            close = getattr(it, "close", None)
            if close:
                close()                    # the client hung up (barge-in): stop synthesising
            self.close_connection = True

    def _chunk(self, b):
        if b:
            self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
            self.wfile.flush()


class _NotFound(Exception):
    pass


class SpeechServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, svc, host="127.0.0.1", port=0, token_file=None):
        if host not in ("127.0.0.1", "::1"):
            raise ValueError("the speech service listens on loopback only")
        if host == "::1":
            self.address_family = socket.AF_INET6
        ThreadingHTTPServer.__init__(self, (host, port), Handler)
        self.svc = svc
        self.token_file = ensure_token(token_file)
        self.token = read_token(self.token_file)

    @property
    def url(self):
        h, p = self.server_address[:2]
        return "http://%s:%d" % ("[%s]" % h if ":" in h else h, p)

    def write_discovery(self, path=None):
        path = path or discovery_path()
        _write_private(path, json.dumps({"api": API, "url": self.url, "token_file": self.token_file,
                                         "pid": os.getpid()}))
        return path

    def start_background(self):
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t
