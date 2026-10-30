"""Text-to-speech engines, first clause first (vendored from Anibal voice tts.py).

Engine protocol: ``synth(text, voice=None, speed=None) -> PCM`` (s16le mono at ``engine.rate``) for ONE clause;
``stream(text)`` splits the text into clauses (a short first clause, then whole sentences) and yields
``(clause, pcm)`` in order, synthesising clause N+1 while clause N is being sent (prefetch).

  * KokoroTTS  Kokoro-82M (Apache-2.0 weights) int8 ONNX via kokoro-onnx (MIT) + onnxruntime, from the pinned
               registry snapshot (onnx/model_quantized.onnx + allowlisted voices/*.bin). Licence caveat (as in
               Anibal): kokoro-onnx phonemises with espeak-ng loaded in-process via espeakng-loader (GPL-3.0), the
               user's own pip install, never bundled.
  * PiperTTS   Piper VITS voices as plain ONNX (no piper binary): espeak-ng IPA through the same phonemizer
               kokoro-onnx already loads, ids from the voice's .onnx.json, onnxruntime. Only allowlisted voices
               (PIPER_VOICES: weights trained from scratch on data that allows commercial use). Speaks the
               languages Kokoro has no voice for (German).
  * MultiTTS   one engine per voice family behind one name: the voice picks the engine, output at one rate.
  * StubTTS    tests: a quiet 440 Hz tone, length proportional to the text.
"""
import math
import os
import re
import struct
import threading
import time

RATE = 16000

# Kokoro-82M v1.0 voices (Apache-2.0); "espeak" = the G2P code. Same allowlist as Anibal voices.KOKORO
# (the *_santa novelty voices are not listed).
_LANG = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-br",
         "z": "cmn"}
KOKORO_VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_nova", "af_river", "af_jessica", "af_kore", "af_aoede",
    "af_alloy", "af_sky", "am_michael", "am_fenrir", "am_puck", "am_adam", "am_echo", "am_eric", "am_liam", "am_onyx",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george", "bm_fable", "bm_lewis", "bm_daniel", "ef_dora",
    "em_alex", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha",
    "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "zf_xiaobei", "zf_xiaoni",
    "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang")
DEFAULT_VOICE = "bm_george"           # Anibal's signature voice
SPEED_MIN, SPEED_MAX = 0.5, 2.0
DEFAULT_SPEED = 1.12                  # = Anibal settings.DEFAULT_SPEED (one PXA voice everywhere)
# fixed pronunciations (IPA as Kokoro expects), English voices only
PRONOUNCE = {"anibal": "ɑnˈiːbɑl", "anibal's": "ɑnˈiːbɑlz", "vigil": "vˈɪdʒɪl", "vigil's": "vˈɪdʒɪlz"}


# Piper voices (rhasspy/piper-voices, MIT repo) checked one by one against MODEL_CARD 2026-10-20: only voices trained
# FROM SCRATCH on data that allows commercial use. de_DE-ramona-low: M-AILABS (BSD-style, commercial use allowed,
# keep the notice). Refused: de_DE-thorsten-* (fine-tuned from lessac = Blizzard research-only), kerstin/karlsson
# (fine-tuned from ryan = CC BY-NC-SA), pavoque (CC BY-NC-SA).
PIPER_VOICES = {"de_DE-ramona-low": {"lang": "de", "espeak": "de", "licence": "M-AILABS (BSD-style); trained from scratch"}}


def espeak_lang(voice):
    return _LANG.get(str(voice)[:1], "en-us")


def clamp_speed(v, default=1.0):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(SPEED_MIN, min(SPEED_MAX, v))


def split_clauses(text, first_max=60):
    """Speak the first short clause ASAP (latency), then whole sentences."""
    parts = re.split(r"(?<=[.!?;:])\s+", str(text).strip())
    out = []
    for i, p in enumerate(p for p in parts if p):
        if i == 0 and len(p) > first_max and "," in p[:first_max + 20]:
            a, b = p.split(",", 1)
            out += [a + ",", b.strip()]
        else:
            out.append(p)
    return out


def pause_after(clause, rate=RATE):
    """Trailing silence so clauses don't run together: 260 ms after . ! ?, 110 ms after , ; :, else 40 ms."""
    end = clause.rstrip()[-1:] if clause.strip() else ""
    ms = 260 if end and end in ".!?" else 110 if end and end in ",;:—–" else 40
    return b"\0\0" * int(rate * ms / 1000)


def prefetch(gen, depth=2):
    """Run a producer in a thread so item N+1 is computed while N is consumed. Errors reach the consumer."""
    import queue
    q = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()
    end = object()

    def run():
        try:
            for item in gen:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except Exception as e:
            q.put(e)
        finally:
            q.put(end)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()


class TTSEngine(object):
    """synth() keeps an LRU of the last CACHE_N clauses (text, voice, speed) -> PCM: assistants start many replies
    with the same short clause ("Sure.", "Okay."). Kokoro on onnxruntime CPU is deterministic, so a hit returns the
    same bytes a fresh synthesis would (bit-identical). PXA_SPEECH_TTS_CACHE=0 turns it off."""
    name = "tts"
    rate = RATE
    voices = (DEFAULT_VOICE,)
    CACHE_N = 64
    CACHE_MAX_CHARS = 80

    def __init__(self):
        import collections
        self.lock = threading.Lock()            # the engine (one synthesis at a time)
        self.clock = threading.Lock()           # the cache (a hit never waits for a running synthesis)
        self.calls, self.busy_s, self.hits = 0, 0.0, 0
        self.cache = collections.OrderedDict()
        self.cache_on = os.environ.get("PXA_SPEECH_TTS_CACHE", "1") != "0"

    def synth(self, text, voice=None, speed=None):
        key = (text, voice, speed)
        if self.cache_on and len(text) <= self.CACHE_MAX_CHARS:
            with self.clock:
                pcm = self.cache.get(key)
                if pcm is not None:
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return pcm
            pcm = self._synth_locked(text, voice, speed)
            with self.clock:
                self.cache[key] = pcm
                while len(self.cache) > self.CACHE_N:
                    self.cache.popitem(last=False)
            return pcm
        return self._synth_locked(text, voice, speed)

    def _synth_locked(self, text, voice, speed):
        with self.lock:
            t = time.monotonic()
            try:
                return self._synth(text, voice, speed)
            finally:
                self.calls += 1
                self.busy_s += time.monotonic() - t

    def _synth(self, text, voice, speed):
        raise NotImplementedError

    def stream(self, text, voice=None, speed=None, pauses=True):
        """-> iterator of (clause, pcm), first clause first, next clause synthesised while this one is sent."""
        def gen():
            for clause in split_clauses(text):
                pcm = self.synth(clause, voice, speed)
                yield clause, pcm + (pause_after(clause, self.rate) if pauses else b"")
        return prefetch(gen())

    def warm(self):
        self.synth("Ready.")
        self.synth("Okay, one moment.")

    def info(self):
        return {"engine": self.name, "rate": self.rate, "voices": list(self.voices), "calls": self.calls,
                "busy_s": round(self.busy_s, 3), "cache": {"on": self.cache_on, "hits": self.hits,
                                                           "size": len(self.cache)}}


class StubTTS(TTSEngine):
    name = "stub"

    def __init__(self, ms_per_char=40, min_ms=200, max_ms=8000, amplitude=800, delay=0.0):
        TTSEngine.__init__(self)
        self.ms_per_char, self.min_ms, self.max_ms = ms_per_char, min_ms, max_ms
        self.amplitude, self.delay, self.texts = int(amplitude), float(delay), []

    def _synth(self, text, voice, speed):
        self.texts.append(text)
        if self.delay:
            time.sleep(self.delay)
        ms = max(self.min_ms, min(self.max_ms, len(text.strip()) * self.ms_per_char))
        n = (ms - ms % 20) * self.rate // 1000
        return b"".join(struct.pack("<h", int(self.amplitude * math.sin(2 * math.pi * 440 * i / self.rate)))
                        for i in range(n))


_KANJI = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def japanese_reader():
    """kanji -> katakana reading, or None. espeak-ng has no kanji dictionary: without a reader it says "Chinese letter"
    for every kanji (round trip 2026-10-20: chrF 0.10), so the Japanese Kokoro voices are offered only when a reader is
    installed: pyopenjtalk (MIT; OpenJTalk and its dictionary BSD-3), e.g. pip install pyopenjtalk-plus.
    PXA_SPEECH_JA_READER=0 = off (the Japanese voices then drop out)."""
    if os.environ.get("PXA_SPEECH_JA_READER", "1") == "0":
        return None
    try:
        import pyopenjtalk
    except ImportError:
        return None

    def read(text):
        return pyopenjtalk.g2p(text, kana=True) if _KANJI.search(text) else text
    return read


class KokoroTTS(TTSEngine):
    """Kokoro-82M int8 ONNX from a registry snapshot dir (onnx/model_quantized.onnx + voices/*.bin)."""
    name = "kokoro"

    def __init__(self, root, voice=None, speed=DEFAULT_SPEED, threads=None, rate=RATE, model_id=""):
        TTSEngine.__init__(self)
        try:
            import numpy as np
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except ImportError:
            raise RuntimeError("pip install kokoro-onnx onnxruntime numpy (optional extra) for Kokoro")
        from .stt import default_threads
        self.np, self.rate, self.model_id = np, int(rate), model_id
        if self.rate not in (16000, 22050, 24000):
            raise ValueError("Kokoro output rate must be 16000, 22050 or 24000")
        model = os.path.join(root, "onnx", "model_quantized.onnx")
        if not os.path.isfile(model):
            raise RuntimeError("Kokoro model not found (fetch it first: python3 -m pxa_control.speech fetch)")
        vdir = os.path.join(root, "voices")
        voices = {}
        for f in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
            n = f[:-4]
            if f.endswith(".bin") and n in KOKORO_VOICES:
                a = np.fromfile(os.path.join(vdir, f), dtype="<f4")
                if a.size % 256 or not np.isfinite(a).all():
                    raise RuntimeError("bad Kokoro voice file %s" % f)
                voices[n] = a.reshape(-1, 1, 256)
        self.ja_reader = japanese_reader()
        self.missing = {}
        if self.ja_reader is None and [v for v in voices if v.startswith("j")]:
            voices = {k: v for k, v in voices.items() if not k.startswith("j")}
            self.missing["ja"] = "Japanese voices need a kanji reader: pip install pyopenjtalk-plus"
        if not voices:
            raise RuntimeError("no allowlisted Kokoro voices in %s" % vdir)
        npz = os.path.join(root, "voices.npz")
        np.savez(npz, **voices)
        self.threads = threads or default_threads()
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if os.environ.get("PXA_SPEECH_GPU") == "1":   # opt-in: GPU providers when onnxruntime-gpu has them
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider",
                                     "CoreMLExecutionProvider") if p in avail] + providers
        self.providers = providers
        sess = ort.InferenceSession(model, so, providers=providers)
        self.k = Kokoro.from_session(sess, npz)
        self.voices = tuple(sorted(voices))
        self.voice = voice if voice in voices else (DEFAULT_VOICE if DEFAULT_VOICE in voices else self.voices[0])
        self.speed = clamp_speed(speed)

    def phonemes(self, text, lang):
        if lang == "ja" and self.ja_reader is not None:
            text = self.ja_reader(text)
        parts = re.split(r"(?i)\b(%s)\b" % "|".join(map(re.escape, sorted(PRONOUNCE, key=len, reverse=True))), text)
        out = []
        for p in parts:
            if not p:
                continue
            ov = PRONOUNCE.get(p.lower()) if lang.startswith("en") else None
            out.append(ov if ov else self.k.tokenizer.phonemize(p, lang))
        return " ".join(x.strip() for x in out if x.strip())

    def _synth(self, text, voice, speed):
        np = self.np
        voice = voice if voice in self.voices else self.voice
        lang = espeak_lang(voice)
        ph = self.phonemes(text, lang)
        if not ph:
            return b""
        a, sr = self.k.create(ph, voice=voice, speed=clamp_speed(speed, self.speed), lang=lang, is_phonemes=True)
        if sr != self.rate:
            a = np.interp(np.arange(0, len(a), sr / self.rate), np.arange(len(a)), a)
        return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()

    def info(self):
        d = TTSEngine.info(self)
        d.update(threads=self.threads, model=self.model_id, voice=self.voice, providers=self.providers)
        if self.missing:
            d["missing"] = dict(self.missing)
        return d


def resample(np, pcm, src, dst):
    """s16le mono at src Hz -> dst Hz (linear interpolation, the same as KokoroTTS uses)."""
    if src == dst or not pcm:
        return pcm
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n = int(round(len(a) * dst / float(src)))
    b = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    return np.clip(b, -32768, 32767).astype("<i2").tobytes()


class PiperTTS(TTSEngine):
    """Piper voices from a registry snapshot dir (<voice>.onnx + <voice>.onnx.json), allowlisted names only."""
    name = "piper"

    def __init__(self, root, threads=None, rate=RATE, model_id="", phonemize=None):
        TTSEngine.__init__(self)
        import json
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("pip install onnxruntime numpy (optional extra) for Piper voices")
        from .stt import default_threads
        self.np, self.rate, self.model_id = np, int(rate), model_id
        self.threads = threads or default_threads()
        self.sessions, self.cfg = {}, {}
        found = []
        for d, _dirs, files in sorted(os.walk(root)) if os.path.isdir(root) else []:
            found += [(f[:-5], os.path.join(d, f)) for f in sorted(files) if f.endswith(".onnx")]
        for v, path in found:       # the registry keeps the repo layout (de/de_DE/ramona/low/<voice>.onnx)
            if v not in PIPER_VOICES or v in self.sessions or not os.path.isfile(path + ".json"):
                continue
            with open(path + ".json", encoding="utf-8") as fh:
                cfg = json.load(fh)
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            self.sessions[v] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            self.cfg[v] = cfg
        if not self.sessions:
            raise RuntimeError("no allowlisted Piper voices in %s" % root)
        self.voices = tuple(sorted(self.sessions))
        self.voice = self.voices[0]
        self._phonemize = phonemize or self._espeak

    @staticmethod
    def _espeak(text, lang):
        """espeak-ng IPA with stress and punctuation, through phonemizer + espeakng-loader (the user's own pip install,
        the same pair kokoro-onnx loads; GPL-3.0 espeak-ng is never bundled)."""
        import espeakng_loader
        import phonemizer
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        try:
            from kokoro_onnx.tokenizer import _espeak_lock as lock    # espeak-ng is process-global: share Kokoro's lock
        except ImportError:
            lock = threading.Lock()
        with lock:
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            return phonemizer.phonemize(text, lang, preserve_punctuation=True, with_stress=True)

    def ids(self, phonemes, voice):
        m = self.cfg[voice]["phoneme_id_map"]
        pad, out = m["_"][0], list(m["^"]) + [m["_"][0]]
        for ch in phonemes:
            if ch in m:
                out += list(m[ch]) + [pad]
        return out + list(m["$"])

    def _synth(self, text, voice, speed):
        np = self.np
        voice = voice if voice in self.sessions else self.voice
        cfg = self.cfg[voice]
        ph = self._phonemize(text, cfg.get("espeak", {}).get("voice") or PIPER_VOICES[voice]["espeak"]).strip()
        if not ph:
            return b""
        inf = cfg.get("inference", {})
        ids = np.array([self.ids(ph, voice)], dtype=np.int64)
        scales = np.array([inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0) / clamp_speed(speed, 1.0),
                           inf.get("noise_w", 0.8)], dtype=np.float32)
        feeds = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64), "scales": scales}
        if int(cfg.get("num_speakers", 1)) > 1:
            feeds["sid"] = np.array([0], dtype=np.int64)
        a = self.sessions[voice].run(None, feeds)[0].reshape(-1)
        pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
        return resample(np, pcm, int(cfg.get("audio", {}).get("sample_rate", 22050)), self.rate)

    def warm(self):
        for v in self.voices:
            self.synth("Hallo.", v)

    def info(self):
        d = TTSEngine.info(self)
        d.update(threads=self.threads, model=self.model_id, voice=self.voice)
        return d


class MultiTTS(TTSEngine):
    """Several engines behind one: the voice picks the engine (the first engine is the default), output at the first
    engine's rate. Each engine keeps its own lock; the router keeps the shared clause cache."""
    name = "multi"

    def __init__(self, engines):
        TTSEngine.__init__(self)
        self.engines = [e for e in engines if e is not None]
        if not self.engines:
            raise ValueError("MultiTTS needs at least one engine")
        self.rate = self.engines[0].rate
        self.voice = getattr(self.engines[0], "voice", None)
        seen = []
        for e in self.engines:
            seen += [v for v in e.voices if v not in seen]
        self.voices = tuple(seen)

    def engine_for(self, voice):
        for e in self.engines:
            if voice in e.voices:
                return e
        return self.engines[0]

    def _synth_locked(self, text, voice, speed):
        e = self.engine_for(voice)
        pcm = e._synth_locked(text, voice, speed)
        if e.rate != self.rate:
            import numpy as np
            pcm = resample(np, pcm, e.rate, self.rate)
        self.calls += 1
        return pcm

    def warm(self):
        for e in self.engines:
            e.warm()

    def info(self):
        d = TTSEngine.info(self)
        d["engines"] = [e.info() for e in self.engines]
        return d
