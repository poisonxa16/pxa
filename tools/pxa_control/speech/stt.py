"""Speech-to-text engines + the incremental session (vendored from Anibal voice stt.py).

Engine protocol: ``transcribe(pcm, language=None) -> str`` for one utterance (16 kHz mono s16le bytes). Engines keep
their model warm and serialise calls on one lock, so every plugin shares one loaded model. ``language`` is a hint (ISO
639-1 such as "es", or "auto") that only a multilingual engine uses; an English-only model ignores it.

  * WhisperLibSTT     whisper.cpp (MIT) in-process via pywhispercpp (MIT, prebuilt CPU wheel); model stays loaded.
  * WhisperServerSTT  a warm whisper.cpp ``whisper-server`` on loopback (what vigil's bench used).
  * FakeSTT           tests: fixed text after an optional delay.

``StreamSession`` = ``speech.stt`` incremental: feed audio of any size as it arrives; the session cuts 20 ms frames,
runs the VAD, and on a pause (80 ms) starts a speculative transcription of the audio so far in the background, so
end-of-speech -> text costs ~0 when nothing more is said (``reused``). Optional partials every ``partial_s``.
"""
import math
import os
import re
import threading
import time
import wave
import io

from .vad import EnergyVAD, FRAME_BYTES, RATE

_NOISE = re.compile(r"\s*[\[(](?:BLANK_AUDIO|blank_audio|MUSIC|Music|music|NOISE|noise|SILENCE|silence|"
                    r"inaudible|INAUDIBLE|no speech)[^\])]*[\])]\s*")


def clean(text):
    """whisper's non-speech markers ([BLANK_AUDIO], (music) ...) -> ''; whitespace collapsed."""
    return " ".join(_NOISE.sub(" ", str(text or "")).split())


def default_threads():
    try:
        n = int(os.environ.get("PXA_SPEECH_THREADS") or 0)
    except ValueError:
        n = 0
    if n > 0:
        return n
    try:
        avail = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        avail = os.cpu_count() or 4
    return max(1, min(4, avail))     # 4: fastest for whisper tiny/base and Kokoro on a shared CPU (bench 10-19)


AUDIO_CTX_MARGIN = 192               # encoder frames (20 ms each) beyond the audio: 3.84 s of slack


def audio_ctx_for(n_samples, setting=None):
    """whisper encoder context for an utterance. whisper.cpp encodes a full 30 s window (1500 frames) even for a
    1.8 s command; 'auto' (default) sizes it to the audio + AUDIO_CTX_MARGIN, rounded up to 64, capped at 1500.
    PXA_SPEECH_AUDIO_CTX=0 = the full window (off switch); a number = that fixed size. Not bit-identical to the
    full window; check 2026-10-19 on 11 clips (tiny.en): margin 192 gives the full-window text on 6/11, the rest
    differ in punctuation/case or in names the full window also mishears, and the full window itself changed
    text between two runs on 2 clips (the model's own run-to-run band). Decode 3-6x faster on short commands.
    """
    if setting is None:
        setting = os.environ.get("PXA_SPEECH_AUDIO_CTX", "auto")
    setting = str(setting).strip().lower()
    if setting in ("", "auto"):
        frames = n_samples / float(RATE) * 50 + AUDIO_CTX_MARGIN
        return int(min(1500, -(-int(math.ceil(frames)) // 64) * 64))
    try:
        v = int(setting)
    except ValueError:
        return 0
    return max(0, min(1500, v))


_LANG_RE = re.compile(r"^(?:[a-z]{2,3}|auto)$")


def check_language(language):
    """'' / None -> None; 'es' / 'ES' / 'es-MX' / 'auto' -> 'es' / 'es' / 'es' / 'auto'; anything else ValueError."""
    if language is None or str(language).strip() == "":
        return None
    c = str(language).strip().lower().replace("_", "-").split("-")[0]
    if not _LANG_RE.match(c):
        raise ValueError("language must be an ISO 639-1 code such as 'es', or 'auto'")
    return c


def is_english_only(model):
    """whisper ggml files named *.en.bin (tiny.en, base.en, small.en) know English only."""
    return ".en." in os.path.basename(str(model or "")) or str(model or "").endswith(".en")


def wav_bytes(pcm, rate=RATE):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


class STTEngine(object):
    name = "stt"
    rate = RATE
    multilingual = False      # True: the engine uses a per-call language hint (self.call_language inside _transcribe)

    def __init__(self):
        self.lock = threading.Lock()
        self.calls, self.busy_s = 0, 0.0
        self.call_language = None

    def transcribe(self, pcm, language=None):
        if not pcm:
            return ""
        language = check_language(language)
        with self.lock:
            t = time.monotonic()
            self.call_language = language if self.multilingual else None
            try:
                return clean(self._transcribe(bytes(pcm)))
            finally:
                self.calls += 1
                self.busy_s += time.monotonic() - t

    def _transcribe(self, pcm):
        raise NotImplementedError

    def warm(self):
        self.transcribe(b"\0\0" * (RATE // 2))

    def info(self):
        return {"engine": self.name, "rate": self.rate, "calls": self.calls, "busy_s": round(self.busy_s, 3),
                "multilingual": bool(self.multilingual)}


class FakeSTT(STTEngine):
    name = "fake"

    def __init__(self, text="", delay=0.0, by_len=None):
        STTEngine.__init__(self)
        self.text, self.delay, self.by_len, self.seen = text, float(delay), by_len, []

    def _transcribe(self, pcm):
        self.seen.append(len(pcm))
        if self.delay:
            time.sleep(self.delay)
        return self.by_len(pcm) if self.by_len else self.text


class WhisperLibSTT(STTEngine):
    """whisper.cpp in this process (pywhispercpp). model = path to a ggml .bin from the pinned registry."""
    name = "whisper.cpp"

    def __init__(self, model, threads=None, language=None, model_id="", audio_ctx=None):
        STTEngine.__init__(self)
        if not os.path.isfile(model):
            raise RuntimeError("whisper model not found (fetch it first: python3 -m pxa_control.speech fetch)")
        try:
            from pywhispercpp.model import Model
            import numpy as np
        except ImportError:
            raise RuntimeError("pip install pywhispercpp numpy (optional extra) for whisper.cpp in-process")
        self.np, self.threads, self.model_id = np, threads or default_threads(), model_id
        self.audio_ctx_setting = audio_ctx
        # a multilingual model (ggml-base.bin, ggml-small.bin) gets a language on EVERY call: the hint, else the
        # service default, else "auto" (whisper.cpp's own default is "en", which makes a multilingual model write
        # Spanish speech down in English). English-only models keep whisper.cpp's default.
        self.multilingual = not (is_english_only(model) or is_english_only(model_id))
        self.language = check_language(language) or ("auto" if self.multilingual else None)
        kw = dict(n_threads=self.threads, print_progress=False, print_realtime=False, print_timestamps=False,
                  no_context=True, redirect_whispercpp_logs_to=None)
        if self.language:
            kw["language"] = self.language
        self.m = Model(model, **kw)
        self.last_language = None

    def _transcribe(self, pcm):
        a = self.np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(self.np.float32) / 32768.0
        ac = audio_ctx_for(len(a), self.audio_ctx_setting)
        self.last_audio_ctx = ac
        kw = {"audio_ctx": ac}
        if self.multilingual:
            kw["language"] = self.last_language = self.call_language or self.language
        return " ".join(s.text.strip() for s in self.m.transcribe(a, **kw) if s.text.strip())

    def info(self):
        d = STTEngine.info(self)
        d.update(threads=self.threads, model=self.model_id, multilingual=self.multilingual, language=self.language,
                 audio_ctx=str(self.audio_ctx_setting or os.environ.get("PXA_SPEECH_AUDIO_CTX", "auto")))
        return d


class WhisperServerSTT(STTEngine):
    """whisper.cpp's whisper-server on loopback (model warm in that process). Stdlib HTTP only."""
    name = "whisper-server"

    def __init__(self, url="http://127.0.0.1:8178", timeout=30.0):
        STTEngine.__init__(self)
        import urllib.parse
        u = urllib.parse.urlsplit(url)
        if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("whisper-server must be http on loopback")
        self.url, self.timeout = url.rstrip("/"), float(timeout)

    def _transcribe(self, pcm):
        import urllib.request
        bnd = b"pxaspeechboundary7"
        body = (b"--" + bnd + b"\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n"
                b"--" + bnd + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                b"Content-Type: audio/wav\r\n\r\n" + wav_bytes(pcm) + b"\r\n--" + bnd + b"--\r\n")
        req = urllib.request.Request(self.url + "/inference", data=body,
                                     headers={"Content-Type": "multipart/form-data; boundary=" + bnd.decode()})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read(1 << 20).decode("utf-8", "replace").strip()


class IncrementalSTT(object):
    """Speculative transcription on top of an engine's transcribe() (Anibal voice IncrementalSTT, same logic).

    add(pcm, voiced) per 20 ms frame. When a pause reaches ``pause_frames`` (80 ms) a background transcription of
    the audio up to the last voiced frame starts at once, overlapping the VAD hang; result() reuses it when nothing
    voiced came after. Partials every ``partial_s`` of audio go to ``on_partial``.
    """

    def __init__(self, stt, partial_s=0.0, on_partial=None, rate=RATE, pause_frames=4, language=None):
        self.stt, self.on_partial = stt, on_partial
        self.language = check_language(language)
        self.buf = bytearray()
        self.voiced_len = 0
        self.quiet = 0
        self.pause_frames = pause_frames
        self.spec = None          # (length, thread, slot)
        self.partial_bytes = int(partial_s * rate) * 2
        self._last_partial = 0
        self._partial_busy = False
        self.reused = False

    def _tx(self, pcm):
        """the engine's transcribe, with the session's language hint when there is one (duck-typed engines keep
        their one-argument transcribe)."""
        return self.stt.transcribe(pcm, self.language) if self.language else self.stt.transcribe(pcm)

    def _launch(self, pcm):
        slot = {}

        def run():
            try:
                slot["text"] = self._tx(pcm)
            except Exception as e:
                slot["err"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, slot

    def add(self, pcm, voiced):
        self.buf += pcm
        if voiced:
            self.quiet = 0
            self.voiced_len = len(self.buf)
            if self.spec and self.spec[0] < self.voiced_len:
                self.spec = None  # they kept talking: discard the guess
        else:
            self.quiet += 1
            if self.quiet == self.pause_frames and self.voiced_len:
                self.spec = (self.voiced_len,) + self._launch(bytes(self.buf[:self.voiced_len]))
        if (self.on_partial and self.partial_bytes and voiced and not self._partial_busy
                and len(self.buf) - self._last_partial >= self.partial_bytes):
            self._last_partial = len(self.buf)
            snap = bytes(self.buf)
            self._partial_busy = True

            def part():
                try:
                    txt = self._tx(snap)
                    if txt:
                        self.on_partial(txt)
                except Exception:
                    pass
                finally:
                    self._partial_busy = False
            threading.Thread(target=part, daemon=True).start()

    def result(self):
        self.reused = False
        if self.spec and self.spec[0] >= self.voiced_len:
            n, t, slot = self.spec
            t.join()
            if "text" in slot:
                self.reused = True
                return slot["text"]
        return self._tx(bytes(self.buf[:self.voiced_len]) if self.voiced_len else bytes(self.buf))


class StreamSession(object):
    """One utterance for ``speech.stt`` incremental. feed(pcm) any size -> {"events": [...], "partial": str}.
    events: "start" (speech began), "end" (the VAD hang passed: the caller may finish()). finish() -> text."""

    def __init__(self, engine, partial_s=0.0, hang_ms=300, vad=None, language=None):
        self.partial = ""
        self.vad = vad or EnergyVAD(hang_ms=hang_ms)
        self.inc = IncrementalSTT(engine, partial_s=partial_s, on_partial=self._on_partial, language=language)
        self.rest = b""
        self.ended = self.started = False
        self.pre = []
        self.created = self.touched = time.monotonic()
        self.frames = 0
        self.lock = threading.Lock()

    def _on_partial(self, text):
        self.partial = text

    PREROLL = 8      # frames (160 ms) kept before the VAD start so the onset is not cut

    def feed(self, pcm):
        events = []
        with self.lock:
            self.touched = time.monotonic()
            if self.ended:          # one utterance per session: audio after the end is not used
                return {"events": events, "partial": self.partial}
            data = self.rest + bytes(pcm)
            n = len(data) - len(data) % FRAME_BYTES
            for i in range(0, n, FRAME_BYTES):
                fr = data[i:i + FRAME_BYTES]
                ev = self.vad.feed(fr)
                self.frames += 1
                if not self.started:
                    if ev != "start":
                        self.pre.append(fr)
                        del self.pre[:-self.PREROLL]
                        continue
                    self.started = True
                    for p in self.pre:
                        self.inc.add(p, False)
                    self.pre = []
                self.inc.add(fr, self.vad.quiet == 0)       # = Anibal Conversation.listen
                if ev:
                    events.append(ev)
                    if ev == "end":
                        self.ended = True
                        break
            self.rest = b"" if self.ended else data[n:]
        return {"events": events, "partial": self.partial}

    def finish(self):
        with self.lock:
            if not self.started and self.pre:   # never crossed the VAD start: transcribe what there is
                for p in self.pre:
                    self.inc.add(p, False)
                self.pre = []
            if self.rest:
                self.inc.add(self.rest, False)
                self.rest = b""
            t = time.monotonic()
            text = self.inc.result()
            return {"text": text, "reused": self.inc.reused, "ms": round((time.monotonic() - t) * 1000, 1),
                    "audio_s": round(len(self.inc.buf) / 2.0 / RATE, 3)}
