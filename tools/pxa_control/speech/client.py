"""Python client for the local speech API (pxa-speech/1) + the adapters the plugins plug in. Stdlib only.

    c = SpeechClient()                     # finds the service via <config>/speech/service.json (or url=, token_file=)
    c.transcribe(pcm) -> str
    s = c.session(partial_s=0); s.feed(pcm) -> {"events", "partial"}; s.finish() -> {"text", "reused", ...}
    for pcm in c.tts_stream("Sure. It is sunny."): play(pcm)          # first clause arrives first
    c.vad(pcm) -> {"segments", "speech_end_s"}

Adapters (duck-typed: they import nothing from the plugins, so no plugin depends on this module at import time):
  * ScribeSTT        Scribe's STT interface: transcribe(pcm, channel) -> str, close().
                     Today, with no Scribe change: ``--stt "exec:python3 -m pxa_control.speech stt-pipe"``.
  * SatelliteSpeech  Satellite's Stt + Tts backends: start(rate, width, channels, language) -> session with
                     feed/partial/finish, synth(sentence, voice) -> PCM at (rate, width, channels); voices; languages.
  * DictateRecognizer  Dictate's model.recognize(pcm, vocab=None) -> text.
"""
import http.client
import json
import os
import struct
import urllib.parse

RATE = 16000


class SpeechUnavailable(RuntimeError):
    pass


class SpeechClientError(RuntimeError):
    def __init__(self, status, msg):
        RuntimeError.__init__(self, "speech service HTTP %d: %s" % (status, msg))
        self.status = status


def _discover():
    from . import server
    p = server.discovery_path()
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("url"), d.get("token_file")
    except (OSError, ValueError):
        return None, None


class SpeechClient(object):
    def __init__(self, url=None, token_file=None, timeout=120.0):
        if url is None:
            url = os.environ.get("PXA_SPEECH_URL")
            token_file = token_file or os.environ.get("PXA_SPEECH_TOKEN_FILE")
            if url is None:
                url, tf = _discover()
                token_file = token_file or tf
        if not url:
            raise SpeechUnavailable("the shared speech service is not running (python3 -m pxa_control.speech serve)")
        u = urllib.parse.urlsplit(url)
        if u.scheme != "http" or u.hostname not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("the speech service URL must be http on loopback")
        self.host, self.port, self.timeout = u.hostname, u.port or 80, float(timeout)
        self.token_file = token_file

    def _headers(self, ctype=None):
        h = {}
        if self.token_file:
            with open(self.token_file, encoding="utf-8") as f:     # by path, never logged
                h["Authorization"] = "Bearer " + f.read().strip()
        if ctype:
            h["Content-Type"] = ctype
        return h

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _call(self, method, path, body=None, ctype=None):
        conn = self._conn()
        try:
            try:
                conn.request(method, path, body=body, headers=self._headers(ctype))
                r = conn.getresponse()
                data = r.read(16 << 20)
            except OSError as e:
                raise SpeechUnavailable("the shared speech service is not reachable: %s" % type(e).__name__)
            try:
                d = json.loads(data.decode("utf-8")) if data else {}
            except ValueError:
                d = {"error": "bad reply"}
            if r.status != 200:
                raise SpeechClientError(r.status, d.get("error", ""))
            return d
        finally:
            conn.close()

    def health(self):
        return self._call("GET", "/v1/speech/health")

    def info(self):
        return self._call("GET", "/v1/speech/info")

    @staticmethod
    def _stt_path(language):
        if not language:
            return "/v1/speech/stt"
        from urllib.parse import quote
        return "/v1/speech/stt?language=" + quote(str(language), safe="")

    def transcribe(self, pcm, language=None):
        return self.transcribe_full(pcm, language)["text"]

    def transcribe_full(self, pcm, language=None):
        """-> {"text", "ms"[, "language"]}. language: an ISO 639-1 hint ("es") or "auto"; English-only models ignore it."""
        return self._call("POST", self._stt_path(language), bytes(pcm), "audio/L16; rate=16000; channels=1")

    def session(self, partial_s=0.0, hang_ms=300, language=None):
        body = {"partial_s": partial_s, "hang_ms": hang_ms}
        if language:
            body["language"] = language
        sid = self._call("POST", "/v1/speech/stt/sessions", json.dumps(body).encode(), "application/json")["id"]
        return ClientSession(self, sid)

    def vad(self, pcm, hang_ms=300):
        return self._call("POST", "/v1/speech/vad", bytes(pcm), "audio/L16; rate=16000; channels=1")

    def tts_stream(self, text, voice=None, speed=None, read_size=640):
        """-> generator of PCM pieces (s16le mono at self.tts_rate, set once the reply headers arrive)."""
        body = {"text": text}
        if voice:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed
        conn = self._conn()
        try:
            conn.request("POST", "/v1/speech/tts", json.dumps(body).encode("utf-8"), self._headers("application/json"))
            r = conn.getresponse()
        except OSError as e:
            conn.close()
            raise SpeechUnavailable("the shared speech service is not reachable: %s" % type(e).__name__)
        if r.status != 200:
            data = r.read(65536)
            conn.close()
            try:
                msg = json.loads(data.decode("utf-8")).get("error", "")
            except ValueError:
                msg = ""
            raise SpeechClientError(r.status, msg)
        self.tts_rate = int(r.getheader("X-Speech-Rate") or RATE)
        return self._pieces(conn, r, read_size)

    @staticmethod
    def _pieces(conn, r, read_size):
        try:
            odd = b""
            while True:
                b = r.read1(read_size) if hasattr(r, "read1") else r.read(read_size)
                if not b:
                    break
                b = odd + b
                cut = len(b) - len(b) % 2
                odd = b[cut:]
                if cut:
                    yield b[:cut]
        finally:
            conn.close()

    def tts(self, text, voice=None, speed=None):
        return b"".join(self.tts_stream(text, voice, speed))


class ClientSession(object):
    def __init__(self, client, sid):
        self.client, self.id = client, sid
        self.last_partial = ""
        self.done = False

    def feed(self, pcm):
        d = self.client._call("POST", "/v1/speech/stt/sessions/" + self.id, bytes(pcm),
                              "audio/L16; rate=16000; channels=1")
        self.last_partial = d.get("partial") or self.last_partial
        return d

    def partial(self):
        return self.last_partial

    def finish(self):
        self.done = True
        return self.client._call("POST", "/v1/speech/stt/sessions/%s/finish" % self.id)

    def close(self):
        if not self.done:
            self.done = True
            try:
                self.client._call("DELETE", "/v1/speech/stt/sessions/" + self.id)
            except (SpeechUnavailable, SpeechClientError):
                pass


# ---- audio helpers (stdlib) ------------------------------------------------------------------------------------

def to_mono16k(pcm, rate, width=2, channels=1):
    """s16le at any rate/channel count -> 16 kHz mono s16le (channel mean, linear interpolation)."""
    if width != 2:
        raise ValueError("only 16-bit audio is supported")
    n = len(pcm) // (2 * channels)
    if not n:
        return b""
    s = struct.unpack("<%dh" % (n * channels), pcm[: n * 2 * channels])
    mono = s if channels == 1 else [sum(s[i * channels:(i + 1) * channels]) / channels for i in range(n)]
    if rate == RATE:
        return pcm[: n * 2] if channels == 1 else struct.pack("<%dh" % n, *(int(x) for x in mono))
    m = int(n * RATE / rate)
    out = []
    step = rate / float(RATE)
    for j in range(m):
        x = j * step
        i = int(x)
        f = x - i
        a = mono[i]
        b = mono[i + 1] if i + 1 < n else a
        out.append(int(round(a + (b - a) * f)))
    return struct.pack("<%dh" % m, *out)


# ---- adapters --------------------------------------------------------------------------------------------------

class ScribeSTT(object):
    """Scribe's STT backend (scribe.stt.STT protocol). channel ("mic"|"playback") is Scribe's own label."""
    name = "speech"

    def __init__(self, client=None):
        self.client = client or SpeechClient()

    def transcribe(self, pcm, channel):
        return self.client.transcribe(pcm)

    def close(self):
        pass


class _SatSession(object):
    def __init__(self, client, rate, width, channels, partial_s):
        self.rate, self.width, self.channels = rate, width, channels
        self.s = client.session(partial_s=partial_s)

    def feed(self, pcm):
        self.s.feed(to_mono16k(pcm, self.rate, self.width, self.channels))

    def partial(self):
        return self.s.partial()

    def finish(self):
        return self.s.finish()["text"]


class SatelliteSpeech(object):
    """Satellite's Stt + Tts backends (satellite.backends protocol) on the shared service."""
    name = "pxa-speech"
    languages = ("en",)
    width, channels = 2, 1

    def __init__(self, client=None, partial_s=0.0, voice=None):
        self.client, self.partial_s, self.voice = client or SpeechClient(), partial_s, voice
        self.rate = RATE
        try:
            tts = self.client.info().get("tts") or {}
            self.rate = int(tts.get("rate") or RATE)
            self.voices = tuple((v, ("en",)) for v in tts.get("voices") or ()) or (("default", ("en",)),)
        except (SpeechUnavailable, SpeechClientError):
            self.voices = (("default", ("en",)),)

    def start(self, rate, width, channels, language=None):
        return _SatSession(self.client, rate, width, channels, self.partial_s)

    def synth(self, sentence, voice=None):
        v = voice if voice and voice != "default" else self.voice
        return self.client.tts(sentence, voice=v)


class DictateRecognizer(object):
    """Dictate's recognizer (dictate.model.recognize(pcm, vocab=None)) on the shared service: open vocabulary."""

    def __init__(self, client=None):
        self.client = client or SpeechClient()

    def recognize(self, pcm, vocab=None):
        if not isinstance(pcm, (bytes, bytearray)):     # Dictate also passes float samples
            pcm = struct.pack("<%dh" % len(pcm), *(int(max(-1.0, min(1.0, float(x))) * 32767) for x in pcm))
        return self.client.transcribe(pcm)
