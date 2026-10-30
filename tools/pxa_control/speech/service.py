"""SpeechService: the engines kept warm in ONE process, shared by every plugin (Dictate, Scribe, Satellite, Co-host).

    svc = SpeechService.from_config(stt="tiny.en", tts="kokoro")     # or stt=FakeSTT(...), tts=StubTTS()
    svc.load()                       # load + warm both (startup cost paid once)
    svc.transcribe(pcm) -> {"text", "ms"}
    sid = svc.open_session(); svc.feed(sid, pcm) -> {"events", "partial"}; svc.finish(sid) -> {"text", "reused", "ms"}
    for clause, pcm in svc.tts_stream(text): ...                       # first clause first
    svc.vad(pcm) -> {"segments", "speech_end_s"}

Engines load lazily on the first call when load() was not called. One lock per engine: calls from several plugins
queue on the one model instead of loading a second copy. Sessions expire after ``session_ttl`` idle seconds; at most
``max_sessions`` are open.
"""
import os
import threading
import time
import uuid

from . import registry, vad as vadmod
from .stt import STTEngine, StreamSession, WhisperLibSTT, WhisperServerSTT, check_language, default_threads
from .tts import TTSEngine, KokoroTTS, MultiTTS, PiperTTS, StubTTS, RATE

STT_MODELS = {"tiny.en": "whisper-tiny.en", "base.en": "whisper-base.en", "small.en": "whisper-small.en",
              "base": "whisper-base", "small": "whisper-small"}
TTS_MODELS = {"kokoro": "kokoro-82m-onnx-q8"}
# extra voices for languages Kokoro does not speak, added to the "kokoro" engine when downloaded
# (fetch --extra piper-de); PXA_SPEECH_EXTRAS=0 turns them off.
TTS_EXTRAS = {"piper-de": "piper-de-ramona-low"}
MAX_AUDIO_S = 120                     # one utterance / one batch call
MAX_TEXT = 4000


# the hearing model a service starts with when nobody names one: the best one already downloaded, multilingual first
# (the Interpreter needs one that hears more than English), tiny.en when none is
STT_PREFERENCE = ("small", "base", "small.en", "base.en", "tiny.en")


def best_stt(cache=None):
    try:
        reg = registry.load_registry()
        for k in STT_PREFERENCE:
            e = reg.get(STT_MODELS[k])
            if e and registry.is_fetched(e, cache):
                return k
    except Exception:  # noqa: BLE001 - an unreadable registry: the default below says what to fetch
        pass
    return "tiny.en"


class SpeechError(Exception):
    status = 400


class NotFound(SpeechError):
    status = 404


class Busy(SpeechError):
    status = 429


class SpeechService(object):
    def __init__(self, stt=None, tts=None, max_sessions=8, session_ttl=60.0, clock=time.monotonic):
        self._stt, self._tts = stt, tts      # an engine, or a zero-arg factory
        self.stt_engine = stt if isinstance(stt, STTEngine) else None
        self.tts_engine = tts if isinstance(tts, TTSEngine) else None
        self.max_sessions, self.session_ttl, self.clock = int(max_sessions), float(session_ttl), clock
        self.sessions = {}
        self._lock = threading.Lock()
        self.started = clock()
        self.load_s = {}

    # ---- construction --------------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, stt="tiny.en", tts="kokoro", cache=None, threads=None, language=None, voice=None,
                    tts_rate=RATE, whisper_url=None, fetch=False, log=None, **kw):
        """stt: tiny.en | base.en | small.en | base | server (whisper_url) | an engine. tts: kokoro | stub | an engine.
        fetch=True downloads missing pinned models (one cache for every plugin)."""
        reg = registry.load_registry()
        threads = threads or default_threads()

        def need(model_id):
            entry = reg[model_id]
            if fetch:
                return registry.fetch(entry, cache=cache, log=log)
            if not registry.is_fetched(entry, cache):
                raise SpeechError("speech model %s is not downloaded (python3 -m pxa_control.speech fetch)" % model_id)
            return registry.local_dir(entry, cache)

        if isinstance(stt, STTEngine):
            stt_f = stt
        elif stt == "server":
            stt_f = lambda: WhisperServerSTT(whisper_url or "http://127.0.0.1:8178")  # noqa: E731
        elif stt in STT_MODELS:
            mid = STT_MODELS[stt]
            path = os.path.join(need(mid), reg[mid]["files"][0]["path"])
            stt_f = lambda: WhisperLibSTT(path, threads=threads, language=language, model_id=mid)  # noqa: E731
        else:
            raise SpeechError("unknown stt %r" % (stt,))
        if isinstance(tts, TTSEngine):
            tts_f = tts
        elif tts == "stub":
            tts_f = StubTTS
        elif tts in TTS_MODELS:
            mid = TTS_MODELS[tts]
            root = need(mid)
            extras = []
            if os.environ.get("PXA_SPEECH_EXTRAS", "1") != "0":
                extras = [(xid, registry.local_dir(reg[xid], cache)) for xid in sorted(set(TTS_EXTRAS.values()))
                          if xid in reg and registry.is_fetched(reg[xid], cache)]

            def tts_f():
                k = KokoroTTS(root, voice=voice, threads=threads, rate=tts_rate, model_id=mid)
                if not extras:
                    return k
                return MultiTTS([k] + [PiperTTS(d, threads=threads, rate=tts_rate, model_id=x) for x, d in extras])
        else:
            raise SpeechError("unknown tts %r" % (tts,))
        svc = cls(stt_f, tts_f, **kw)
        if not isinstance(stt, STTEngine) and stt in STT_MODELS:     # what info() can say before a lazy service loads it
            svc.stt_desc = {"engine": "whisper.cpp", "model": STT_MODELS[stt], "multilingual": not stt.endswith(".en"), "loaded": False}
        return svc

    def _get(self, kind):
        with self._lock:
            eng = getattr(self, kind + "_engine")
            if eng is None:
                t = time.monotonic()
                eng = getattr(self, "_" + kind)()
                setattr(self, kind + "_engine", eng)
                self.load_s[kind] = round(time.monotonic() - t, 3)
            return eng

    def load(self, warm=True):
        """Load (and warm) both engines now, so the first real call is fast."""
        for kind in ("stt", "tts"):
            eng = self._get(kind)
            if warm:
                t = time.monotonic()
                eng.warm()
                self.load_s[kind + "_warm"] = round(time.monotonic() - t, 3)
        return self

    @property
    def stt(self):
        return self._get("stt")

    @property
    def tts(self):
        return self._get("tts")

    # ---- speech.stt ----------------------------------------------------------------------------------------
    @staticmethod
    def check_pcm(pcm):
        if len(pcm) % 2:
            raise SpeechError("PCM must be 16-bit samples (even byte count)")
        if len(pcm) > MAX_AUDIO_S * RATE * 2:
            raise SpeechError("audio longer than %d s" % MAX_AUDIO_S)

    @staticmethod
    def check_language(language):
        try:
            return check_language(language)
        except ValueError as e:
            raise SpeechError(str(e))

    def transcribe(self, pcm, language=None):
        """-> {"text", "ms"} (+ "language": the hint the engine used, when it is multilingual)."""
        self.check_pcm(pcm)
        language = self.check_language(language)
        t = time.monotonic()
        eng = self.stt
        text = eng.transcribe(pcm, language) if language else eng.transcribe(pcm)
        out = {"text": text, "ms": round((time.monotonic() - t) * 1000, 1)}
        if getattr(eng, "multilingual", False):
            out["language"] = language or getattr(eng, "language", None) or "auto"
        return out

    def _expire(self):
        now = self.clock()
        for sid in [s for s, v in self.sessions.items() if now - v.touched > self.session_ttl]:
            self.sessions.pop(sid, None)

    def open_session(self, partial_s=0.0, hang_ms=300, language=None):
        eng = self.stt
        language = self.check_language(language)
        with self._lock:
            self._expire()
            if len(self.sessions) >= self.max_sessions:
                raise Busy("too many open speech sessions")
            sid = uuid.uuid4().hex
            try:
                partial_s = max(0.0, min(10.0, float(partial_s or 0)))
                hang_ms = max(100, min(2000, int(hang_ms or 300)))
            except (TypeError, ValueError):
                raise SpeechError("bad session options")
            s = StreamSession(eng, partial_s=partial_s, hang_ms=hang_ms, language=language)
            s.touched = self.clock()
            self.sessions[sid] = s
            return sid

    def _session(self, sid):
        with self._lock:
            s = self.sessions.get(sid)
            if s is None:
                raise NotFound("no such speech session")
            s.touched = self.clock()
            return s

    def feed(self, sid, pcm):
        self.check_pcm(pcm)
        s = self._session(sid)
        if (len(s.inc.buf) + len(pcm)) > MAX_AUDIO_S * RATE * 2:
            raise SpeechError("utterance longer than %d s" % MAX_AUDIO_S)
        return s.feed(pcm)

    def finish(self, sid):
        s = self._session(sid)
        try:
            return s.finish()
        finally:
            self.close_session(sid)

    def close_session(self, sid):
        with self._lock:
            return self.sessions.pop(sid, None) is not None

    # ---- speech.tts ----------------------------------------------------------------------------------------
    def tts_stream(self, text, voice=None, speed=None):
        text = str(text or "").strip()
        if not text:
            raise SpeechError("no text")
        if len(text) > MAX_TEXT:
            raise SpeechError("text longer than %d characters" % MAX_TEXT)
        return self.tts.stream(text, voice=voice, speed=speed)

    def tts_pcm(self, text, voice=None, speed=None):
        return b"".join(pcm for _c, pcm in self.tts_stream(text, voice, speed))

    # ---- vad -----------------------------------------------------------------------------------------------
    def vad(self, pcm, hang_ms=300):
        self.check_pcm(pcm)
        return {"segments": vadmod.segments(pcm, hang_ms=max(100, min(2000, int(hang_ms)))),
                "speech_end_s": round(vadmod.speech_end_s(pcm), 3)}

    def info(self):
        d = {"api": "pxa-speech/1", "rate_in": RATE, "uptime_s": round(self.clock() - self.started, 1),
             "sessions": len(self.sessions), "load_s": dict(self.load_s)}
        if self.stt_engine is not None:
            d["stt"] = self.stt_engine.info()
        elif getattr(self, "stt_desc", None):
            d["stt"] = dict(self.stt_desc)
        if self.tts_engine is not None:
            d["tts"] = self.tts_engine.info()
        return d
