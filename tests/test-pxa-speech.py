#!/usr/bin/env python3
"""Shared speech service (plugin plan S5): registry, VAD, incremental STT, first-clause-first TTS, the local API,
the client and the plugin adapters. Stdlib only, no network, no real models (fake/stub engines).

    python3 tests/test-pxa-speech.py

Opt-in real-engine end to end (wav -> stt -> text -> tts -> wav -> stt): PXA_SPEECH_MODELS=<cache with
whisper-tiny.en + kokoro-82m-onnx-q8> and the optional extras installed.
"""
import http.client
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-speech-test-")
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REAL_MODELS = os.environ.get("PXA_SPEECH_MODELS")
if not REAL_MODELS:
    os.environ["PXA_SPEECH_MODELS"] = os.path.join(TMP, "models")
sys.path.insert(0, TOOLS)

from pxa_control.speech import registry, vad, stt, tts, service, server, client  # noqa: E402

RATE = 16000


def tone(ms, freq=440.0, amp=8000):
    n = RATE * ms // 1000
    return struct.pack("<%dh" % n, *(int(amp * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)))


def silence(ms):
    return b"\0\0" * (RATE * ms // 1000)


def wav(pcm, rate=RATE, ch=1):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


UTT = silence(300) + tone(900) + silence(1200)


class Registry(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp(dir=TMP)
        self.reg = registry.load_registry()

    def test_pinned_entries_and_licences(self):
        for mid in ("whisper-tiny.en", "whisper-base.en", "kokoro-82m-onnx-q8"):
            e = self.reg[mid]
            self.assertRegex(e["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(registry.check_licence(e, cache=self.cache), "allowed")
            for f in e["files"]:
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual((set(service.STT_MODELS.values()) | set(service.TTS_MODELS.values())) - set(self.reg), set())

    def test_blocked_and_unknown_licence_refused_override_recorded(self):
        with self.assertRaises(registry.LicenceRefused):
            registry.check_licence(self.reg["xtts-v2"], cache=self.cache)
        e = dict(self.reg["whisper-tiny.en"], licence="cc-by-nc-4.0")
        with self.assertRaises(registry.LicenceRefused):
            registry.check_licence(e, cache=self.cache)
        with self.assertRaises(registry.LicenceRefused):
            registry.check_licence(e, override={"reason": " "}, cache=self.cache)
        self.assertEqual(registry.check_licence(e, override={"reason": "owner test", "by": "t"}, cache=self.cache),
                         "override")
        with open(os.path.join(self.cache, "overrides.jsonl")) as f:
            self.assertEqual(json.loads(f.readline())["reason"], "owner test")

    def _entry(self, data, path="ggml-x.bin", engine="whisper.cpp"):
        import hashlib
        return {"id": "t-model", "kind": "stt", "engine": engine, "repo": "r/r", "revision": "a" * 40,
                "licence": "mit", "files": [{"path": path, "sha256": hashlib.sha256(data).hexdigest(),
                                             "size": len(data)}]}

    def _dl(self, data, calls):
        def dl(repo, filename, revision, token, local_dir):
            calls.append((repo, filename, revision))
            p = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
            return p
        return dl

    def test_fetch_verifies_and_reuses(self):
        data = b"lmgg" + os.urandom(100)
        e, calls = self._entry(data), []
        self.assertFalse(registry.is_fetched(e, self.cache))
        root = registry.fetch(e, cache=self.cache, downloader=self._dl(data, calls))
        self.assertTrue(registry.is_fetched(e, self.cache, verify=True))
        registry.fetch(e, cache=self.cache, downloader=self._dl(data, calls))
        self.assertEqual(len(calls), 1, "a verified file is not downloaded again (one download for every plugin)")
        self.assertEqual(root, os.path.join(self.cache, "t-model"))

    def test_sha_mismatch_floating_and_pickle_refused(self):
        data = b"lmgg" + os.urandom(64)
        e = self._entry(data)
        with self.assertRaises(registry.RegistryError):
            registry.fetch(e, cache=self.cache, downloader=self._dl(b"lmgg-other", []))
        self.assertFalse(os.path.exists(os.path.join(self.cache, "t-model", "ggml-x.bin")))
        with self.assertRaises(registry.RegistryError):
            registry.fetch(dict(e, revision="main"), cache=self.cache, downloader=self._dl(data, []))
        with self.assertRaises(registry.UnsafeFormat):
            registry.fetch(self._entry(data, "model.pth"), cache=self.cache, downloader=self._dl(data, []))
        pk = b"\x80\x04" + os.urandom(30)
        with self.assertRaises(registry.UnsafeFormat):
            registry.fetch(self._entry(pk, "m.onnx", "x"), cache=self.cache, downloader=self._dl(pk, []))
        with self.assertRaises(registry.UnsafeFormat):     # .bin that is not ggml
            registry.fetch(self._entry(b"zzzz1234", "w.bin"), cache=self.cache, downloader=self._dl(b"zzzz1234", []))

    def test_every_model_pins_its_repo_commit_and_each_file(self):
        for mid, e in self.reg.items():
            if e.get("blocked"):
                continue
            self.assertRegex(e["repo"], r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", mid)
            self.assertRegex(e["revision"], r"^[0-9a-f]{40}$", mid)
            for f in e["files"]:
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$", (mid, f["path"]))
                self.assertGreater(f["size"], 0, (mid, f["path"]))

    def test_a_file_without_a_pinned_sha_is_never_downloaded(self):
        data, calls = b"lmgg" + os.urandom(64), []
        e = self._entry(data)
        for bad in ("", None, "A" * 64, "a" * 63):
            f = dict(e["files"][0], sha256=bad)
            with self.assertRaises(registry.RegistryError):
                registry.fetch(dict(e, files=[f]), cache=self.cache, downloader=self._dl(data, calls))
        self.assertEqual(calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.cache, "t-model", "ggml-x.bin")))

    def test_a_floating_revision_cannot_be_fetched_at_all(self):
        data, calls = b"lmgg" + os.urandom(64), []
        with self.assertRaises((TypeError, registry.RegistryError)):      # no escape hatch for an unpinned revision
            registry.fetch(dict(self._entry(data), revision="main"), cache=self.cache, allow_floating=True,
                           downloader=self._dl(data, calls))
        self.assertEqual(calls, [])

    def test_token_redacted(self):
        os.environ["HF_TOKEN"] = "hf_" + "Q" * 20
        try:
            def boom(*a):
                raise OSError("401 for token hf_" + "Q" * 20)
            with self.assertRaises(registry.RegistryError) as cm:
                registry.fetch(self._entry(b"lmggxx"), cache=self.cache, downloader=boom)
            self.assertNotIn("Q" * 20, str(cm.exception))
        finally:
            del os.environ["HF_TOKEN"]


class Vad(unittest.TestCase):
    def test_segments_and_end(self):
        segs = vad.segments(UTT)
        self.assertEqual(len(segs), 1)
        a, b = segs[0]
        self.assertAlmostEqual(a, 0.3, delta=0.05)
        self.assertAlmostEqual(b, 1.2 + 0.3, delta=0.06)       # end event = last voiced + 300 ms hang
        self.assertAlmostEqual(vad.speech_end_s(UTT), 1.2, delta=0.03)
        self.assertEqual(vad.segments(silence(2000)), [])

    def test_echo_gate(self):
        v = vad.EnergyVAD()
        fr = tone(20, amp=3000)
        self.assertTrue(v.is_voiced(fr))
        v.echo_level = 1.0
        self.assertFalse(v.is_voiced(fr))


class Stt(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(stt.clean(" [BLANK_AUDIO] "), "")
        self.assertEqual(stt.clean("Hello (music)  there"), "Hello there")

    def test_session_speculative_reuse(self):
        eng = stt.FakeSTT("what is the weather like today")
        s = stt.StreamSession(eng)
        evs = []
        for i in range(0, len(UTT), 640):
            evs += s.feed(UTT[i:i + 640])["events"]
        self.assertEqual(evs, ["start", "end"])
        r = s.finish()
        self.assertEqual(r["text"], "what is the weather like today")
        self.assertTrue(r["reused"], "the pause-time transcription is reused: end of speech -> text ~0")
        self.assertEqual(len(eng.seen), 1)
        # the preroll keeps the onset: audio from ~160 ms before the VAD start up to the last voiced frame
        self.assertGreaterEqual(eng.seen[0], len(tone(900)))

    def test_session_odd_chunks_and_continued_speech(self):
        eng = stt.FakeSTT(by_len=lambda pcm: "n%d" % len(pcm))
        s = stt.StreamSession(eng)
        audio = silence(200) + tone(400) + silence(120) + tone(400) + silence(600)
        pos = 0
        for n in (1000, 333, 2047, 640, 7):        # arbitrary sizes are re-cut into 20 ms frames
            s.feed(audio[pos:pos + n])
            pos += n
        while pos < len(audio):
            s.feed(audio[pos:pos + 999])
            pos += 999
        r = s.finish()
        self.assertTrue(r["reused"])
        self.assertGreaterEqual(int(r["text"][1:]), len(tone(800)), "the guess after the 120 ms gap was discarded")

    def test_partials(self):
        class Slow(stt.FakeSTT):
            pass
        eng = Slow("partial words")
        s = stt.StreamSession(eng, partial_s=0.2)
        for i in range(0, len(silence(100) + tone(800)), 640):
            s.feed((silence(100) + tone(800))[i:i + 640])
        deadline = time.time() + 2
        while not s.partial and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(s.partial, "partial words")

    def test_no_speech(self):
        s = stt.StreamSession(stt.FakeSTT("x"))
        self.assertEqual(s.feed(silence(500))["events"], [])
        self.assertEqual(s.finish()["text"], "x")

    def test_audio_ctx(self):
        os.environ.pop("PXA_SPEECH_AUDIO_CTX", None)
        self.assertEqual(stt.audio_ctx_for(int(1.81 * RATE)), 320)        # 90.5 + 192 -> 320
        self.assertEqual(stt.audio_ctx_for(0), 192)
        self.assertEqual(stt.audio_ctx_for(40 * RATE), 1500)               # long audio: the full window
        self.assertEqual(stt.audio_ctx_for(RATE, "0"), 0)                  # off switch = whisper's full window
        self.assertEqual(stt.audio_ctx_for(RATE, "768"), 768)
        self.assertEqual(stt.audio_ctx_for(RATE, "junk"), 0)
        os.environ["PXA_SPEECH_AUDIO_CTX"] = "0"
        try:
            self.assertEqual(stt.audio_ctx_for(RATE), 0)
        finally:
            del os.environ["PXA_SPEECH_AUDIO_CTX"]
        self.assertEqual(stt.audio_ctx_for(RATE) % 64, 0)

    def test_engine_lock_serialises(self):
        eng = stt.FakeSTT("a", delay=0.05)
        ts = [threading.Thread(target=eng.transcribe, args=(tone(100),)) for _ in range(4)]
        t = time.monotonic()
        [x.start() for x in ts]
        [x.join() for x in ts]
        self.assertGreaterEqual(time.monotonic() - t, 0.19)
        self.assertEqual(eng.info()["calls"], 4)


class Tts(unittest.TestCase):
    def test_split_and_pauses(self):
        self.assertEqual(tts.split_clauses("Sure. It's sunny! Bring a coat?"), ["Sure.", "It's sunny!", "Bring a coat?"])
        long_first = "It is twenty one degrees and sunny outside today, with a light breeze from the west. Ok."
        self.assertEqual(tts.split_clauses(long_first)[0], "It is twenty one degrees and sunny outside today,")
        self.assertEqual(len(tts.pause_after("Hi.")), 2 * 16000 * 260 // 1000)
        self.assertEqual(len(tts.pause_after("Hi,")), 2 * 16000 * 110 // 1000)
        self.assertEqual(tts.clamp_speed("nan"), 1.0)
        self.assertEqual(tts.clamp_speed(9), 2.0)

    def test_first_clause_first_with_prefetch(self):
        eng = tts.StubTTS(delay=0.1)
        t = time.monotonic()
        it = eng.stream("Sure. It's twenty one degrees. Bring a coat.")
        c0, p0 = next(it)
        first = time.monotonic() - t
        rest = [c for c, _ in it]
        whole = time.monotonic() - t
        self.assertEqual(c0, "Sure.")
        self.assertEqual(rest, ["It's twenty one degrees.", "Bring a coat."])
        self.assertLess(first, 0.18)
        self.assertGreaterEqual(whole, 0.29)
        self.assertEqual(eng.texts, ["Sure.", "It's twenty one degrees.", "Bring a coat."])

    def test_clause_cache(self):
        eng = tts.StubTTS(delay=0.05)
        a = eng.synth("Sure.")
        t = time.monotonic()
        b = eng.synth("Sure.")
        self.assertLess(time.monotonic() - t, 0.03)
        self.assertEqual(a, b)
        self.assertEqual(eng.texts, ["Sure."], "a hit does not synthesise again")
        self.assertEqual(eng.info()["cache"]["hits"], 1)
        eng.synth("Sure.", voice="bf_emma")                               # other voice/speed = other entry
        self.assertEqual(len(eng.texts), 2)
        long_text = "x" * (eng.CACHE_MAX_CHARS + 1)
        eng.synth(long_text)
        eng.synth(long_text)
        self.assertEqual(eng.texts.count(long_text), 2, "long clauses are not cached")
        for i in range(eng.CACHE_N + 5):
            eng.synth("c%d" % i)
        self.assertEqual(len(eng.cache), eng.CACHE_N)
        os.environ["PXA_SPEECH_TTS_CACHE"] = "0"
        try:
            off = tts.StubTTS()
            off.synth("Sure.")
            off.synth("Sure.")
            self.assertEqual(off.texts, ["Sure.", "Sure."])
        finally:
            del os.environ["PXA_SPEECH_TTS_CACHE"]

    def test_prefetch_error_reaches_consumer(self):
        def gen():
            yield 1
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            list(tts.prefetch(gen()))

    def test_voice_allowlist(self):
        self.assertIn("bm_george", tts.KOKORO_VOICES)
        self.assertNotIn("am_santa", tts.KOKORO_VOICES)
        self.assertEqual(tts.espeak_lang("bm_george"), "en-gb")


class Service(unittest.TestCase):
    def setUp(self):
        self.clock = [100.0]
        self.svc = service.SpeechService(stt.FakeSTT("hello there"), tts.StubTTS(), max_sessions=2, session_ttl=10,
                                         clock=lambda: self.clock[0])

    def test_calls(self):
        self.assertEqual(self.svc.transcribe(tone(200))["text"], "hello there")
        self.assertGreater(len(self.svc.tts_pcm("Hi. There.")), 0)
        self.assertEqual(len(self.svc.vad(UTT)["segments"]), 1)
        self.assertEqual(self.svc.info()["api"], "pxa-speech/1")

    def test_limits(self):
        with self.assertRaises(service.SpeechError):
            self.svc.transcribe(b"\0\0\0")
        with self.assertRaises(service.SpeechError):
            self.svc.transcribe(b"\0\0" * (service.MAX_AUDIO_S * RATE + 1))
        with self.assertRaises(service.SpeechError):
            list(self.svc.tts_stream("   "))
        with self.assertRaises(service.SpeechError):
            list(self.svc.tts_stream("x" * (service.MAX_TEXT + 1)))

    def test_sessions_cap_ttl_notfound(self):
        a = self.svc.open_session()
        self.svc.open_session()
        with self.assertRaises(service.Busy):
            self.svc.open_session()
        self.clock[0] += 11
        c = self.svc.open_session()                  # the two idle ones expired
        self.assertEqual(len(self.svc.sessions), 1)
        with self.assertRaises(service.NotFound):
            self.svc.feed(a, tone(20))
        self.svc.feed(c, UTT)
        self.assertEqual(self.svc.finish(c)["text"], "hello there")
        self.assertEqual(self.svc.sessions, {})

    def test_from_config(self):
        with self.assertRaises(service.SpeechError):
            service.SpeechService.from_config(stt="nope", tts="stub")
        with self.assertRaises(service.SpeechError) as cm:
            service.SpeechService.from_config(stt="tiny.en", tts="stub", cache=os.path.join(TMP, "empty"))
        self.assertIn("not downloaded", str(cm.exception))
        s = service.SpeechService.from_config(stt=stt.FakeSTT("k"), tts="stub")
        self.assertIsNone(s.tts_engine)
        s.load()
        self.assertEqual(s.transcribe(tone(40))["text"], "k")
        self.assertIn("tts_warm", s.load_s)

    def test_lazy_info_and_best_stt(self):
        cache = os.path.join(TMP, "best-stt")
        self.assertEqual(service.best_stt(cache), "tiny.en")                            # nothing downloaded: what fetch gets
        reg = registry.load_registry()
        for mid in ("whisper-base.en", "whisper-base"):                                 # a multilingual model wins over .en
            e = reg[mid]
            for f in e["files"]:
                p = os.path.join(registry.local_dir(e, cache), f["path"])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fh:
                    fh.truncate(f.get("size") or 1)
        self.assertEqual(service.best_stt(cache), "base")
        s = service.SpeechService.from_config(stt="base", tts="stub", cache=cache)     # lazy: nothing loaded
        self.assertIsNone(s.stt_engine)
        self.assertEqual({k: s.info()["stt"][k] for k in ("model", "multilingual")}, {"model": "whisper-base", "multilingual": True})
        s2 = service.SpeechService.from_config(stt="base.en", tts="stub", cache=cache)
        self.assertFalse(s2.info()["stt"]["multilingual"])


class LangFake(stt.FakeSTT):
    """a multilingual fake: records the hint each call got."""
    multilingual = True

    def __init__(self, text=""):
        stt.FakeSTT.__init__(self, text)
        self.langs = []

    def _transcribe(self, pcm):
        self.langs.append(self.call_language)
        return self.text


class Languages(unittest.TestCase):
    def test_check_language(self):
        self.assertEqual([stt.check_language(x) for x in (None, "", "es", "ES", "es-MX", "pt_BR", "auto", "yue")],
                         [None, None, "es", "es", "es", "pt", "auto", "yue"])
        for bad in ("e", "english", "es1", "../x", "es mx"):
            with self.assertRaises(ValueError):
                stt.check_language(bad)
        self.assertTrue(stt.is_english_only("/m/ggml-base.en.bin"))
        self.assertTrue(stt.is_english_only("whisper-tiny.en"))
        self.assertFalse(stt.is_english_only("/m/ggml-small.bin"))

    def test_hint_only_reaches_a_multilingual_engine(self):
        e = LangFake("hola")
        e.transcribe(tone(100), "es")
        e.transcribe(tone(100))
        self.assertEqual(e.langs, ["es", None])
        en = stt.FakeSTT("hi")
        en.transcribe(tone(100), "es")                  # an English-only engine ignores the hint
        self.assertIsNone(en.call_language)
        self.assertEqual((e.info()["multilingual"], en.info()["multilingual"]), (True, False))

    def test_service_and_session_pass_the_hint(self):
        e = LangFake("hola")
        svc = service.SpeechService(e, tts.StubTTS())
        out = svc.transcribe(tone(200), language="es")
        self.assertEqual((out["text"], out["language"], e.langs[-1]), ("hola", "es", "es"))
        out = svc.transcribe(tone(200), language="es-MX")
        self.assertEqual((out["text"], out["language"]), ("hola", "es"))
        self.assertEqual(svc.transcribe(tone(200))["language"], "auto")
        self.assertNotIn("language", service.SpeechService(stt.FakeSTT("x"), tts.StubTTS()).transcribe(tone(200)))
        with self.assertRaises(service.SpeechError):
            svc.transcribe(tone(200), language="klingon")
        with self.assertRaises(service.SpeechError):
            svc.open_session(language="1x")
        sid = svc.open_session(language="de")
        svc.feed(sid, UTT)
        svc.finish(sid)
        self.assertIn("de", e.langs)
        self.assertEqual(set(e.langs) - {"es", None, "de"}, set())

    def test_japanese_reader_only_when_installed_and_on(self):
        import types
        fake = types.ModuleType("pyopenjtalk")
        fake.g2p = lambda text, kana=False: "オミズヲ" if kana else "o m i"
        old = sys.modules.get("pyopenjtalk")
        sys.modules["pyopenjtalk"] = fake
        try:
            r = tts.japanese_reader()
            self.assertEqual((r("お水を"), r("おみずを")), ("オミズヲ", "おみずを"))      # kana-only text is left alone
            os.environ["PXA_SPEECH_JA_READER"] = "0"
            self.assertIsNone(tts.japanese_reader())
        finally:
            os.environ.pop("PXA_SPEECH_JA_READER", None)
            if old is None:
                sys.modules.pop("pyopenjtalk", None)
            else:
                sys.modules["pyopenjtalk"] = old
        sys.modules["pyopenjtalk"] = None                                           # not installed
        try:
            self.assertIsNone(tts.japanese_reader())
        finally:
            if old is None:
                sys.modules.pop("pyopenjtalk", None)
            else:
                sys.modules["pyopenjtalk"] = old

    def test_registry_has_the_language_models_with_licences(self):
        reg = registry.load_registry()
        cache = tempfile.mkdtemp(dir=TMP)
        for mid in ("whisper-base", "whisper-small", "piper-de-ramona-low"):
            self.assertEqual(registry.check_licence(reg[mid], cache=cache), "allowed")
            self.assertTrue(all(len(f["sha256"]) == 64 for f in reg[mid]["files"]))
        self.assertEqual(service.STT_MODELS["small"], "whisper-small")
        self.assertEqual(set(service.TTS_EXTRAS.values()) - set(reg), set())
        names = {f["path"].rsplit("/", 1)[-1] for f in reg["kokoro-82m-onnx-q8"]["files"]}
        for v in ("ef_dora", "ff_siwis", "hf_alpha", "jf_alpha", "zf_xiaoxiao", "bm_george"):
            self.assertIn(v + ".bin", names)
            self.assertIn(v, tts.KOKORO_VOICES)
        self.assertEqual(set(tts.PIPER_VOICES), {"de_DE-ramona-low"})
        self.assertEqual({tts.espeak_lang(v) for v in ("ef_dora", "ff_siwis", "hf_alpha")}, {"es", "fr-fr", "hi"})


class FakeSession(object):
    def __init__(self, seen):
        self.seen = seen

    def run(self, _out, feeds):
        import numpy as np
        self.seen.append(feeds)
        return [np.full((1, 1, 2205), 0.25, dtype=np.float32)]      # 0.1 s at 22.05 kHz


class Piper(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        self.np = np
        self.seen = []
        p = tts.PiperTTS.__new__(tts.PiperTTS)
        tts.TTSEngine.__init__(p)
        p.np, p.rate, p.model_id, p.threads = np, 24000, "piper-de-ramona-low", 1
        p.cfg = {"de_DE-ramona-low": {"phoneme_id_map": {"^": [1], "_": [0], "$": [2], "h": [20], "a": [14], "l": [24], "o": [27]},
                                      "audio": {"sample_rate": 22050}, "espeak": {"voice": "de"},
                                      "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8}}}
        p.sessions = {"de_DE-ramona-low": FakeSession(self.seen)}
        p.voices = ("de_DE-ramona-low",)
        p.voice = p.voices[0]
        p._phonemize = lambda text, lang: (self.seen.append(("ph", text, lang)) or "halo?")
        self.p = p

    def test_ids_frame_and_pad(self):
        self.assertEqual(self.p.ids("hal?o", "de_DE-ramona-low"), [1, 0, 20, 0, 14, 0, 24, 0, 27, 0, 2])

    def test_synth_feeds_scales_and_resamples(self):
        pcm = self.p.synth("Hallo.", "de_DE-ramona-low", speed=2.0)
        self.assertEqual(self.seen[0], ("ph", "Hallo.", "de"))
        f = self.seen[1]
        self.assertEqual(f["input"].tolist(), [[1, 0, 20, 0, 14, 0, 24, 0, 27, 0, 2]])
        self.assertEqual(f["input_lengths"].tolist(), [11])
        self.assertAlmostEqual(float(f["scales"][1]), 1 / tts.clamp_speed(2.0, 1.0), places=5)
        self.assertNotIn("sid", f)
        self.assertEqual(len(pcm) // 2, 2400)               # 2205 samples at 22.05 kHz -> 0.1 s at 24 kHz
        self.assertEqual(len(tts.resample(self.np, b"", 22050, 24000)), 0)

    def test_multi_routes_by_voice(self):
        k = tts.StubTTS()
        k.voices = ("af_heart", "bm_george")
        k.rate = 24000
        m = tts.MultiTTS([k, self.p])
        self.assertEqual(m.voices, ("af_heart", "bm_george", "de_DE-ramona-low"))
        self.assertIs(m.engine_for("de_DE-ramona-low"), self.p)
        self.assertIs(m.engine_for("ef_dora"), k)                # unknown -> the first engine (its own default)
        m.synth("Guten Tag.", "de_DE-ramona-low")
        m.synth("Hello.", "bm_george")
        self.assertEqual(k.texts, ["Hello."])
        self.assertEqual([x[1] for x in self.seen if isinstance(x, tuple)], ["Guten Tag."])
        self.assertEqual([e["engine"] for e in m.info()["engines"]], ["stub", "piper"])
        with self.assertRaises(ValueError):
            tts.MultiTTS([None])


class _ServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = service.SpeechService(stt.FakeSTT("what is the weather like today"), tts.StubTTS(delay=0.08))
        cls.srv = server.SpeechServer(cls.svc)
        cls.srv.start_background()
        cls.disc = cls.srv.write_discovery()
        cls.c = client.SpeechClient(cls.srv.url, cls.srv.token_file)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def raw(self, method, path, body=None, headers=None):
        h = {"Host": "127.0.0.1:%d" % self.srv.server_address[1]}
        h.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=10)
        conn.request(method, path, body=body, headers=h)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data


class Api(_ServerCase):
    def test_private_files(self):
        for p in (self.srv.token_file, self.disc):
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
        with open(self.disc) as f:
            d = json.load(f)
        self.assertEqual(d["url"], self.srv.url)
        self.assertNotIn(self.srv.token, json.dumps(d), "the discovery file names the token file, not the token")

    def test_auth_host_origin(self):
        self.assertEqual(self.raw("GET", "/v1/speech/health")[0], 200)
        self.assertEqual(self.raw("GET", "/v1/speech/info")[0], 401)
        self.assertEqual(self.raw("GET", "/v1/speech/info", headers={"Authorization": "Bearer nope"})[0], 401)
        good = {"Authorization": "Bearer " + self.srv.token}
        self.assertEqual(self.raw("GET", "/v1/speech/info", headers=good)[0], 200)
        self.assertEqual(self.raw("GET", "/v1/speech/info", headers=dict(good, Host="evil.example:80"))[0], 403)
        self.assertEqual(self.raw("GET", "/v1/speech/health", headers={"Host": "rebind.example"})[0], 403)
        self.assertEqual(self.raw("POST", "/v1/speech/stt", tone(40),
                                  headers=dict(good, Origin="http://127.0.0.1:1"))[0], 403)
        self.assertEqual(self.raw("POST", "/v1/speech/nope", b"", headers=good)[0], 404)
        for st, data in (self.raw("GET", "/v1/speech/info"), self.raw("POST", "/v1/speech/tts", b"{}", good)):
            self.assertNotIn(self.srv.token.encode(), data)

    def test_stt_raw_wav_and_bad_wav(self):
        self.assertEqual(self.c.transcribe(tone(300)), "what is the weather like today")
        good = {"Authorization": "Bearer " + self.srv.token, "Content-Type": "audio/wav"}
        st, data = self.raw("POST", "/v1/speech/stt", wav(tone(300)), good)
        self.assertEqual((st, json.loads(data)["text"]), (200, "what is the weather like today"))
        st, data = self.raw("POST", "/v1/speech/stt", wav(tone(300), rate=44100), good)
        self.assertEqual(st, 400)
        self.assertIn("16 kHz", json.loads(data)["error"])
        st, _ = self.raw("POST", "/v1/speech/stt", b"\0\0\0", dict(good, **{"Content-Type": "audio/L16"}))
        self.assertEqual(st, 400)

    def test_session_flow(self):
        s = self.c.session()
        evs = []
        for i in range(0, len(UTT), 3200):
            evs += s.feed(UTT[i:i + 3200])["events"]
        self.assertEqual(evs, ["start", "end"])
        r = s.finish()
        self.assertEqual(r["text"], "what is the weather like today")
        self.assertTrue(r["reused"])
        with self.assertRaises(client.SpeechClientError) as cm:
            s.feed(tone(20))
        self.assertEqual(cm.exception.status, 404)
        s2 = self.c.session()
        s2.close()
        self.assertEqual(self.svc.sessions, {})

    def test_tts_streams_first_clause_first(self):
        t = time.monotonic()
        it = self.c.tts_stream("Sure. It's twenty one degrees. Bring a coat.")
        first = next(it)
        t_first = time.monotonic() - t
        rest = b"".join(it)
        t_all = time.monotonic() - t
        self.assertEqual(self.c.tts_rate, 16000)
        self.assertEqual(len(first) % 2, 0)
        self.assertLess(t_first, 0.2, "the first clause is on the wire before the others are synthesised")
        self.assertGreaterEqual(t_all, 0.23)
        expect = self.svc.tts_pcm("Sure. It's twenty one degrees. Bring a coat.")
        self.assertEqual(first + rest, expect)
        with self.assertRaises(client.SpeechClientError) as cm:
            self.c.tts(" ")
        self.assertEqual(cm.exception.status, 400)

    def test_vad(self):
        d = self.c.vad(UTT)
        self.assertEqual(len(d["segments"]), 1)
        self.assertAlmostEqual(d["speech_end_s"], 1.2, delta=0.03)

    def test_discovery_client_and_stt_pipe(self):
        c = client.SpeechClient()                    # via <config>/speech/service.json
        self.assertTrue(c.health()["ok"])
        env = dict(os.environ, PYTHONPATH=TOOLS)
        r = subprocess.run([sys.executable, "-m", "pxa_control.speech", "stt-pipe"], input=tone(300), env=env,
                           capture_output=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertEqual(r.stdout.decode(), "what is the weather like today")
        out = os.path.join(TMP, "say.wav")
        r = subprocess.run([sys.executable, "-m", "pxa_control.speech", "say", "Hi. There.", "-o", out], env=env,
                           capture_output=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        with wave.open(out) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
            self.assertGreater(w.getnframes(), 1000)

    def test_unavailable(self):
        with self.assertRaises(client.SpeechUnavailable):
            client.SpeechClient("http://127.0.0.1:9", None, timeout=2).transcribe(tone(20))
        with self.assertRaises(ValueError):
            client.SpeechClient("http://example.com:80", None)


class Adapters(_ServerCase):
    """The plugin adapters against the same running service (duck-typed protocols of Scribe/Satellite/Dictate)."""

    def test_scribe(self):
        a = client.ScribeSTT(self.c)
        self.assertEqual(a.transcribe(tone(300), "playback"), "what is the weather like today")
        a.close()

    def test_satellite(self):
        a = client.SatelliteSpeech(self.c)
        self.assertEqual((a.rate, a.width, a.channels), (16000, 2, 1))
        sess = a.start(22050, 2, 2, "en")                   # HA may send other formats: converted to 16k mono
        pcm22 = tone(900)
        n = len(pcm22) // 2
        src = struct.unpack("<%dh" % n, pcm22)
        stereo = struct.pack("<%dh" % (2 * n), *[x for v in src for x in (v, v)])
        sess.feed(stereo)
        self.assertEqual(sess.finish(), "what is the weather like today")
        self.assertGreater(len(a.synth("Turning on the kitchen light.")), 1000)

    def test_dictate(self):
        a = client.DictateRecognizer(self.c)
        self.assertEqual(a.recognize([0.2, -0.2] * 4000), "what is the weather like today")

    def test_to_mono16k(self):
        self.assertEqual(client.to_mono16k(tone(100), 16000), tone(100))
        out = client.to_mono16k(struct.pack("<4h", 100, 300, 100, 300), 16000, 2, 2)
        self.assertEqual(struct.unpack("<2h", out), (200, 200))
        self.assertEqual(len(client.to_mono16k(b"\0\0" * 48000, 48000)), 32000)


from pxa_control.speech import setup as ssetup  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402
from unittest import mock  # noqa: E402

PYZ = b"fake pip zipapp"


class FakeRun(object):
    """the setup's runner: records argv; pip calls fill the --target folder, the rest answer rc."""
    def __init__(self, rc=None):
        self.calls, self.rc = [], dict(rc or {})

    def __call__(self, argv, env, cwd, log):
        self.calls.append(list(argv))
        kind = ("pip" if "install" in argv else "check" if "--check" in argv else "fetch" if "fetch" in argv else "other")
        if kind == "pip":
            tgt = argv[argv.index("--target") + 1]
            reqs = [x for x in argv if "==" in x]
            if "-r" in argv:
                with open(argv[argv.index("-r") + 1], encoding="utf-8") as f:
                    reqs += [ln.split()[0] for ln in f if "==" in ln.split(";")[0] and not ln.startswith((" ", "#"))]
            for req in reqs:
                d = os.path.join(tgt, req.split("==")[0].replace("-", "_"))
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, "__init__.py"), "w").close()
        log.write(("%s ran\n" % kind).encode())
        return self.rc.get(kind, 0)

    def kinds(self):
        return ["pip" if "install" in a else "check" if "--check" in a else "fetch" if "fetch" in a else "other" for a in self.calls]


def fake_download(url, dest):
    with open(dest, "wb") as f:
        f.write(PYZ)


class Setup(unittest.TestCase):
    def setUp(self):
        self.cfg = tempfile.mkdtemp(dir=TMP)
        self.site = ssetup.site_dir(self.cfg)
        p = mock.patch.multiple(ssetup, PIP_SHA256=hashlib.sha256(PYZ).hexdigest(), PIP_SIZE=len(PYZ))
        p.start()
        self.addCleanup(p.stop)

    def run_setup(self, fr=None, **kw):
        fr = fr if fr is not None else FakeRun()
        kw.setdefault("disk_free", lambda path: 50 << 30)
        rc = ssetup.run(self.cfg, runner=fr, downloader=kw.pop("downloader", fake_download), python="/py", **kw)
        return rc, fr, ssetup.read_job(self.cfg)

    def test_pins_follow_the_python_version(self):
        self.assertIn("numpy==2.2.6", ssetup.pins((3, 10)))
        self.assertIn("onnxruntime==1.23.2", ssetup.pins((3, 10)))
        self.assertIn("numpy==2.4.4", ssetup.pins((3, 11)))
        for v in ((3, 11), (3, 12), (3, 13), (3, 14)):
            self.assertIn("onnxruntime==1.31.0", ssetup.pins(v))
        for v in ((3, 12), (3, 13), (3, 14)):
            self.assertIn("numpy==2.5.4", ssetup.pins(v))
        for v in ((3, 10), (3, 14)):
            p = ssetup.pins(v)
            self.assertTrue(set(ssetup.COMMON) <= set(p))
            self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9.]*", x) for x in p), p)
            self.assertNotIn(ssetup.KOKORO, p)                  # kokoro-onnx goes in its own --no-deps call
        self.assertNotEqual(ssetup.pins_id((3, 10)), ssetup.pins_id((3, 13)))

    def test_supported_pythons_platforms_and_compiled_builds(self):
        ok = dict(plat="linux", machine="x86_64", frozen=False)
        self.assertEqual(ssetup.supported((3, 13), **ok), (True, None))
        self.assertTrue(ssetup.supported((3, 10), **ok)[0])
        self.assertTrue(ssetup.supported((3, 14), **ok)[0])
        for v in ((3, 9), (3, 15)):
            good, why = ssetup.supported(v, **ok)
            self.assertFalse(good)
            self.assertIn("3.10", why)
        self.assertTrue(ssetup.supported((3, 12), plat="win32", machine="AMD64", frozen=False)[0])
        self.assertTrue(ssetup.supported((3, 12), plat="darwin", machine="arm64", frozen=False)[0])
        self.assertTrue(ssetup.supported((3, 12), plat="linux", machine="aarch64", frozen=False)[0])
        self.assertFalse(ssetup.supported((3, 12), plat="darwin", machine="x86_64", frozen=False)[0])
        self.assertFalse(ssetup.supported((3, 12), plat="win32", machine="ARM64", frozen=False)[0])
        good, why = ssetup.supported((3, 12), plat="linux", machine="x86_64", frozen=True)
        self.assertFalse(good)
        self.assertTrue(why)

    def test_add_site_only_a_finished_install_ahead_of_site_packages(self):
        sp = ["/tools", "/usr/lib/python313.zip", "/usr/lib/python3.13", "/home/u/.local/lib/python3.13/site-packages",
              "/usr/lib/python3/dist-packages"]
        os.makedirs(self.site)
        self.assertIsNone(ssetup.add_site(self.site, syspath=sp))           # no marker: a half install is never used
        self.assertNotIn(self.site, sp)
        ssetup.write_marker(self.site)
        self.assertEqual(ssetup.add_site(self.site, syspath=sp), self.site)
        self.assertEqual(sp.index(self.site), 3)                            # after the stdlib, before every site-packages
        ssetup.add_site(self.site, syspath=sp)
        self.assertEqual(sp.count(self.site), 1)
        other = os.path.join(self.cfg, "x")
        os.makedirs(other)
        self.assertEqual(ssetup.add_site(other, force=True, syspath=sp), other)   # the check of a new folder

    def test_run_installs_checks_swaps_then_fetches(self):
        rc, fr, job = self.run_setup()
        self.assertEqual(rc, 0, job)
        self.assertEqual(fr.kinds(), ["pip", "pip", "check", "fetch"])
        new = self.site + ".new"
        a = fr.calls[0]
        self.assertEqual(a[:4], ["/py", "-I", os.path.join(self.cfg, "speech", "pip-%s.pyz" % ssetup.PIP_VERSION), "install"])
        for flag in ("--isolated", "--only-binary=:all:", "--no-input", "--disable-pip-version-check"):
            self.assertIn(flag, a)
        self.assertEqual(a[a.index("--target") + 1], new)
        self.assertIn("--require-hashes", a)                                 # every file pip takes is checked by its sha256
        self.assertEqual(a[a.index("-r") + 1], ssetup.ENGINES_LOCK)
        self.assertEqual([x for x in a if "==" in x], [])                    # nothing from outside the lock
        k = fr.calls[1]
        for flag in ("--require-hashes", "--no-deps", "--ignore-requires-python"):
            self.assertIn(flag, k)
        self.assertEqual(k[k.index("-r") + 1], ssetup.KOKORO_LOCK)
        self.assertEqual([x for x in k if "==" in x], [])
        self.assertEqual(fr.calls[2], ssetup.speech_argv("/py", "setup", "--check", new))
        self.assertEqual(fr.calls[3], ssetup.speech_argv("/py", "fetch"))
        self.assertTrue(os.path.isfile(os.path.join(self.site, "numpy", "__init__.py")))
        self.assertEqual(ssetup.read_marker(self.site)["pins"], ssetup.pins_id())
        self.assertFalse(os.path.exists(new) or os.path.exists(self.site + ".old"))
        self.assertEqual((job["state"], job["step"]), ("done", "done"))
        self.assertTrue(os.path.isfile(os.path.join(self.cfg, "speech", "setup.log")))
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.cfg, "speech", "setup.log")).st_mode), 0o600)

    def test_languages_add_the_multilingual_models(self):
        rc, fr, job = self.run_setup(languages=True)
        self.assertEqual(rc, 0)
        self.assertEqual(fr.calls[-1], ssetup.speech_argv("/py", "fetch", "--stt", "small", "--extra", "piper-de"))
        self.assertTrue(job["languages"])

    def test_a_new_set_that_fails_its_check_leaves_the_old_one(self):
        self.run_setup()
        keep = os.path.join(self.site, "keep.txt")
        open(keep, "w").close()
        rc, fr, job = self.run_setup(FakeRun({"check": 1}), force=True)
        self.assertEqual(rc, 1)
        self.assertEqual((job["state"], job["step"]), ("failed", "verify"))
        self.assertTrue(os.path.isfile(keep))
        self.assertFalse(os.path.exists(self.site + ".new"))
        self.assertNotIn("fetch", fr.kinds())

    def test_pip_install_failure_says_so_and_keeps_nothing_half_done(self):
        rc, fr, job = self.run_setup(FakeRun({"pip": 1}))
        self.assertEqual(rc, 1)
        self.assertEqual((job["state"], job["step"]), ("failed", "engines"))
        self.assertFalse(os.path.exists(self.site) or os.path.exists(self.site + ".new"))
        self.assertIn("setup.log", job["message"])

    def test_a_failure_hint_comes_from_this_run_only(self):
        sd = os.path.join(self.cfg, "speech")
        os.makedirs(sd, exist_ok=True)
        with open(os.path.join(sd, "setup.log"), "w") as f:              # an earlier run's failure is still in the log
            f.write("OSError: [Errno 28] No space left on device\n")
        rc, fr, job = self.run_setup(FakeRun({"pip": 1}))
        self.assertEqual(rc, 1)
        self.assertNotIn("disk is full", job["message"])

        class Offline(FakeRun):
            def __call__(self, argv, env, cwd, log):
                log.write(b"socket.gaierror: [Errno -3] Temporary failure in name resolution\n")
                return super().__call__(argv, env, cwd, log)
        rc, fr, job = self.run_setup(Offline({"pip": 1}))
        self.assertIn("could not reach the internet", job["message"])     # this run's own output still gives its hint
        self.assertNotIn("disk is full", job["message"])

    def test_a_failure_hint_sees_the_last_unflushed_write(self):
        """the log is buffered: the installer download's error is written just before failed() and is the hint's only
        evidence, so it has to reach the file before the hint reads it."""
        def offline(url, dest):
            raise OSError("[Errno -3] Temporary failure in name resolution")
        rc, fr, job = self.run_setup(downloader=offline)
        self.assertEqual((rc, job["state"], job["step"]), (1, "failed", "installer"))
        self.assertEqual(fr.calls, [])
        self.assertIn("could not reach the internet", job["message"])

    def test_an_unexpected_error_ends_the_job_failed_with_a_hint(self):
        class Boom(FakeRun):
            def __call__(self, argv, env, cwd, log):
                if "fetch" in argv:
                    raise RuntimeError("the registry said no")
                return super().__call__(argv, env, cwd, log)
        rc, fr, job = self.run_setup(Boom())
        self.assertEqual(rc, 1)
        self.assertEqual((job["state"], job["step"]), ("failed", "models"))      # never left "running" until it goes stale
        self.assertIn("RuntimeError", job["message"])
        self.assertIn("setup.log", job["message"])
        with open(os.path.join(self.cfg, "speech", "setup.log"), encoding="utf-8") as f:
            self.assertIn("the registry said no", f.read())                   # the detail goes to the log (0600)

        def full(path):
            raise OSError(28, "No space left on device")
        rc, fr, job = self.run_setup(disk_free=full)                          # a second run is not refused as busy
        self.assertEqual((rc, job["state"]), (1, "failed"))
        self.assertIn("disk is full", job["message"])

    def test_a_notify_that_raises_never_fails_the_job(self):
        """say() only tells whoever started the setup (the Plugins tab reads setup.json): when it fails, the job still runs
        to its end and the file still says how it went."""
        said = []

        def say(message):
            said.append(message)
            raise RuntimeError("the notify channel is down")
        rc, fr, job = self.run_setup(say=say)
        self.assertEqual((rc, job["state"], job["step"]), (0, "done", "done"), job)
        self.assertEqual(said[-1], "Speech is ready.")                       # it was tried at every step, down to the last
        j = ssetup._Job(self.cfg, False, time.time, say=say)
        try:
            j.save(step="models", message="still written")                  # the file is written, nothing comes out
            self.assertEqual(ssetup.read_job(self.cfg)["message"], "still written")
            self.assertTrue(j.beat.is_alive())                               # and the heartbeat lives on
        finally:
            j.end("failed", "models", "over")
        self.assertEqual(ssetup.read_job(self.cfg)["state"], "failed")

    def test_pip_calls_ignore_every_config_file(self):
        """--isolated still loads the machine-wide pip config (C:\\ProgramData\\pip\\pip.ini, writable by a standard user):
        PIP_CONFIG_FILE=os.devnull makes pip skip every config file."""
        seen = []

        class Env(FakeRun):
            def __call__(self, argv, env, cwd, log):
                seen.append((list(argv), dict(env)))
                return super().__call__(argv, env, cwd, log)
        rc, fr, job = self.run_setup(Env())
        self.assertEqual(rc, 0, job)
        pips = [(a, e) for a, e in seen if "install" in a]
        self.assertEqual(len(pips), 2)
        for a, e in pips:
            self.assertEqual(e.get("PIP_CONFIG_FILE"), os.devnull)
            self.assertIn("--isolated", a)
            self.assertIn("--require-hashes", a)

    @unittest.skipUnless(sys.platform.startswith("linux") and importlib.util.find_spec("pip"), "needs pip on Linux")
    def test_real_pip_ignores_a_machine_wide_config(self):
        """offline: a global pip.conf (XDG_CONFIG_DIRS stands in for the machine-wide folder) adding a hashed requirement
        and its own wheel folder installs nothing beyond the locked set."""
        import base64
        import zipfile
        d = tempfile.mkdtemp(dir=TMP)
        wheels, xdg = os.path.join(d, "wheels"), os.path.join(d, "xdg")
        os.makedirs(wheels)
        os.makedirs(os.path.join(xdg, "pip"))

        def wheel(name, files):
            di = "%s-1.0.dist-info" % name
            files = dict(files, **{di + "/METADATA": ("Metadata-Version: 2.1\nName: %s\nVersion: 1.0\n" % name).encode(),
                                   di + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n"})
            rec = ["%s,sha256=%s,%d" % (k, base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b"=").decode(), len(v))
                   for k, v in files.items()]
            files[di + "/RECORD"] = ("\n".join(rec + [di + "/RECORD,,"]) + "\n").encode()
            w = os.path.join(wheels, "%s-1.0-py3-none-any.whl" % name)
            with zipfile.ZipFile(w, "w") as z:
                for k, v in files.items():
                    z.writestr(k, v)
            with open(w, "rb") as f:
                return "%s==1.0 --hash=sha256:%s\n" % (name, hashlib.sha256(f.read()).hexdigest())
        lock, evil = os.path.join(d, "lock.txt"), os.path.join(d, "evil.txt")
        with open(lock, "w") as f:
            f.write(wheel("pxagoodpkg", {"pxagoodpkg/__init__.py": b""}))
        with open(evil, "w") as f:
            f.write(wheel("pxaevilpkg", {"pxagoodpkg/_distributor_init_local.py": b"print('evil')\n"}))
        with open(os.path.join(xdg, "pip", "pip.conf"), "w") as f:
            f.write("[install]\nrequirement = %s\nfind-links = %s\n" % (evil, wheels))
        tgt = os.path.join(d, "site")
        env = dict(ssetup._env(self.cfg), XDG_CONFIG_DIRS=xdg)
        argv = [sys.executable, "-m", "pip", "install", "--isolated", "--target", tgt, "--only-binary=:all:", "--no-input",
                "--disable-pip-version-check", "--no-index", "--find-links", wheels, "--require-hashes", "-r", lock]
        out = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertTrue(os.path.isfile(os.path.join(tgt, "pxagoodpkg", "__init__.py")))
        self.assertFalse(os.path.exists(os.path.join(tgt, "pxagoodpkg", "_distributor_init_local.py")))

    def test_wrong_installer_fingerprint_installs_nothing(self):
        def bad(url, dest):
            with open(dest, "wb") as f:
                f.write(b"something else")
        rc, fr, job = self.run_setup(downloader=bad)
        self.assertEqual(rc, 1)
        self.assertEqual(fr.calls, [])
        self.assertIn("fingerprint", job["message"])
        self.assertEqual([n for n in os.listdir(os.path.join(self.cfg, "speech")) if n.startswith("pip-")], [])

    def test_installed_engines_are_kept_unless_forced(self):
        self.run_setup()
        rc, fr, job = self.run_setup()
        self.assertEqual((rc, fr.kinds()), (0, ["fetch"]))
        rc, fr, job = self.run_setup(force=True)
        self.assertEqual(fr.kinds(), ["pip", "pip", "check", "fetch"])

    def test_not_enough_disk_space_stops_before_any_download(self):
        got = []
        rc, fr, job = self.run_setup(disk_free=lambda path: 100 << 20, downloader=lambda u, d: got.append(u))
        self.assertEqual(rc, 1)
        self.assertEqual((got, fr.calls), ([], []))
        self.assertIn("disk space", job["message"])

    def test_a_running_job_is_not_started_twice_and_a_silent_one_goes_stale(self):
        now = time.time()
        ssetup.write_job(self.cfg, {"state": "running", "step": "engines", "message": "x", "started": now, "updated": now})
        rc, fr, job = self.run_setup()
        self.assertEqual((rc, fr.calls), (2, []))
        self.assertEqual(job["state"], "running")
        ssetup.write_job(self.cfg, {"state": "running", "step": "engines", "message": "x", "started": now - 600, "updated": now - 600})
        self.assertEqual(ssetup.read_job(self.cfg)["state"], "stopped")
        rc, fr, job = self.run_setup()
        self.assertEqual(rc, 0)

    def test_a_bad_state_file_reads_as_no_job(self):
        p = ssetup.job_path(self.cfg)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for bad in ("x", {"a": 1}, [1], "inf", "nan", 1e309, True):
            with open(p, "w") as f:
                f.write(json.dumps({"state": "running", "step": "engines", "message": "x", "updated": bad}).replace("Infinity", "1e309"))
            self.assertIsNone(ssetup.read_job(self.cfg), bad)
            self.assertIsNone(ssetup.status(self.cfg, find_spec=lambda name: None)["job"])
        self.assertEqual(self.run_setup()[0], 0)                             # and a setup runs over it

    def test_an_updated_that_is_no_epoch_reads_as_no_job(self):
        """`updated` is seconds since 1970: a millisecond stamp (never stale), a negative number, infinity or NaN is a damaged
        file, not a job that can stay "running" for ever."""
        p = ssetup.job_path(self.cfg)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for bad in ("1e12", "1700000000000", "-5", "Infinity", "-Infinity", "NaN"):      # raw JSON, as a hand edit writes it
            with open(p, "w") as f:
                f.write('{"state": "running", "step": "engines", "message": "x", "updated": %s}' % bad)
            self.assertIsNone(ssetup.read_job(self.cfg), bad)
            self.assertIsNone(ssetup.status(self.cfg, find_spec=lambda name: None)["job"], bad)
        with open(p, "w") as f:
            json.dump({"state": "running", "step": "engines", "message": "x", "updated": time.time()}, f)
        self.assertEqual(ssetup.read_job(self.cfg)["state"], "running")                  # a real epoch is still a job

    def test_a_job_stamped_ahead_of_the_clock_goes_stale_too(self):
        """a future stamp, or a clock that jumped back, must not keep a job "running" for ever."""
        now = time.time()
        job = {"state": "running", "step": "engines", "message": "x", "started": now}
        for at in (now + 2 * ssetup.STALE_S, now - 2 * ssetup.STALE_S):
            ssetup.write_job(self.cfg, dict(job, updated=at))
            self.assertEqual(ssetup.read_job(self.cfg, clock=lambda: now)["state"], "stopped", at - now)
        ssetup.write_job(self.cfg, dict(job, updated=now + ssetup.STALE_S / 2))
        self.assertEqual(ssetup.read_job(self.cfg, clock=lambda: now)["state"], "running")   # a small skew is no reason to stop it

    def test_status_engines_models_and_ready(self):
        models = os.path.join(self.cfg, "models", "speech")
        p = mock.patch.dict(os.environ, {"PXA_SPEECH_MODELS": models})
        p.start()
        self.addCleanup(p.stop)
        st = ssetup.status(self.cfg, find_spec=lambda name: None)
        self.assertEqual((st["engines"], st["ready"]), ("missing", False))
        self.assertEqual(st["models"], {"english": False, "languages": False})
        self.assertGreater(st["download_mb"]["english"], 150)
        self.assertGreater(st["download_mb"]["languages"], 400)
        st = ssetup.status(self.cfg, find_spec=lambda name: object())
        self.assertEqual(st["engines"], "system")
        # run() installs Control's own engines whenever they are not this set: the sizes shown say so
        self.assertEqual(st["download_mb"]["english"] - ssetup.ENGINES_MB, ssetup.status(self.cfg, find_spec=lambda name: None)
                         ["download_mb"]["english"] - ssetup.ENGINES_MB)
        self.assertEqual(st["download_mb"]["add_languages"], ssetup.ENGINES_MB + st["download_mb"]["languages"])
        os.makedirs(self.site)
        ssetup.write_marker(self.site)
        self.assertEqual(ssetup.status(self.cfg, find_spec=lambda name: None)["engines"], "ready")
        with open(os.path.join(self.site, ssetup.MARKER), "w") as f:
            json.dump({"pins": "older"}, f)
        self.assertEqual(ssetup.status(self.cfg, find_spec=lambda name: None)["engines"], "update")
        reg = registry.load_registry()
        for mid in ("whisper-tiny.en", "kokoro-82m-onnx-q8"):
            for f in reg[mid]["files"]:
                fp = os.path.join(models, mid, f["path"])
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "wb") as fh:
                    fh.truncate(f["size"])
        st = ssetup.status(self.cfg, find_spec=lambda name: None)
        self.assertEqual(st["models"], {"english": True, "languages": False})
        self.assertTrue(st["ready"])
        self.assertEqual(st["engines"], "update")
        self.assertEqual(st["download_mb"]["english"], ssetup.ENGINES_MB)          # an update reinstalls the engines
        self.assertEqual(st["download_mb"]["add_languages"], ssetup.ENGINES_MB + st["download_mb"]["languages"])
        ssetup.write_marker(self.site)
        st = ssetup.status(self.cfg, find_spec=lambda name: None)
        self.assertEqual((st["engines"], st["download_mb"]["english"]), ("ready", 0))
        self.assertEqual(st["download_mb"]["add_languages"], st["download_mb"]["languages"])

    def test_check_names_what_does_not_import(self):
        d = tempfile.mkdtemp(dir=TMP)
        with open(os.path.join(d, "pxa_fake_engine_ok.py"), "w") as f:
            f.write("__version__ = '1.0'\n")
        sp = list(sys.path)
        try:
            self.assertEqual(ssetup.check(d, modules=("pxa_fake_engine_ok",), deep=False),
                             {"ok": True, "versions": {"pxa_fake_engine_ok": "1.0"}})
            r = ssetup.check(d, modules=("pxa_fake_engine_ok", "pxa_fake_engine_missing"), deep=False)
        finally:
            sys.path[:] = sp
        self.assertFalse(r["ok"])
        self.assertIn("pxa_fake_engine_missing", r["error"])

    def test_hints_for_known_failures(self):
        self.assertIn("Visual C++", ssetup.hint("ImportError: DLL load failed while importing onnxruntime_pybind11_state", plat="win32"))
        self.assertIn("Python", ssetup.hint("ERROR: No matching distribution found for onnxruntime==1.31.0", plat="linux"))
        self.assertEqual(ssetup.hint("something new", plat="linux"), "")

    def test_cli_setup_status_prints_json(self):
        out = subprocess.run([sys.executable, "-m", "pxa_control.speech", "setup", "--status"], cwd=TOOLS,
                             env=dict(os.environ, PXA_CONTROL_CONFIG_DIR=self.cfg), capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        st = json.loads(out.stdout)
        self.assertIn(st["engines"], ("missing", "system", "ready", "update"))
        self.assertIn("supported", st)

    def test_speech_children_start_through_the_bootstrap(self):
        a = ssetup.speech_argv("/py", "fetch", tools="/kit")
        self.assertEqual(a, ["/py", "-B", "-u", "-c", ssetup.BOOT, "/kit", "fetch"])
        self.assertEqual(ssetup.speech_argv("/py", "serve")[5], TOOLS)       # default: the tools this module came from

    def test_the_bootstrap_runs_in_an_isolated_python_with_no_path_help(self):
        # python.org's embeddable Python (a compiled Control's own) ignores PYTHONPATH and the current folder: -I is the same
        argv = ssetup.speech_argv(sys.executable, "setup", "--status", tools=TOOLS)
        env = {k: os.environ[k] for k in ("PATH", "HOME", "SYSTEMROOT") if k in os.environ}
        env["PXA_CONTROL_CONFIG_DIR"] = self.cfg
        out = subprocess.run(argv[:1] + ["-I"] + argv[1:], cwd=tempfile.mkdtemp(dir=TMP), env=env, capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        st = json.loads(out.stdout)
        self.assertIn("supported", st)
        self.assertEqual(st["python"], "%d.%d" % sys.version_info[:2])

    def test_cli_setup_check_imports_from_the_folder_first(self):
        d = tempfile.mkdtemp(dir=TMP)
        os.makedirs(os.path.join(d, "numpy"))
        with open(os.path.join(d, "numpy", "__init__.py"), "w") as f:
            f.write("raise ImportError('broken on purpose')\n")          # shadows any numpy the system has
        out = subprocess.run([sys.executable, "-m", "pxa_control.speech", "setup", "--check", d], cwd=TOOLS,
                             env=dict(os.environ, PXA_CONTROL_CONFIG_DIR=self.cfg), capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 1, out.stderr)
        r = json.loads(out.stdout)
        self.assertFalse(r["ok"])
        self.assertIn("broken on purpose", r["error"])

    def test_cli_uses_the_finished_install_for_every_command_but_setup(self):
        from pxa_control.speech import __main__ as cli
        with mock.patch.object(ssetup, "add_site") as add, mock.patch.dict(os.environ, {"PXA_CONTROL_CONFIG_DIR": self.cfg}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["fetch", "--extra", "nope"])                    # stops after parsing, before any download
            self.assertEqual(add.call_count, 1)
            self.assertEqual(cli.main(["setup", "--status"]), 0)
            self.assertEqual(add.call_count, 1)


def lock_entries(path):
    """a pip requirements file -> [(requirement, [sha256, ...])], one per requirement (continued lines joined)."""
    with open(path, encoding="utf-8") as f:
        text = f.read().replace("\\\n", " ")
    out = []
    for ln in text.splitlines():
        if ln.strip() and not ln.lstrip().startswith("#"):
            out.append((re.sub(r"\s*--hash=\S+", "", ln).strip(), re.findall(r"--hash=sha256:(\S+)", ln)))
    return out


def pin(req):
    """'Huggingface_Hub==2.2.0 ; marker' -> ('huggingface-hub', '2.2.0') (PEP 503 names)."""
    name, ver = req.split(" ; ")[0].split("==")
    return re.sub(r"[-_.]+", "-", name).lower(), ver


class EngineLock(unittest.TestCase):
    """the engines come only from the hash locks: pip checks every file it downloads against its sha256, and the locks
    name everything the pins pull in, so nothing new on PyPI reaches a setup."""

    def test_every_requirement_is_a_pin_with_its_sha256_and_nothing_else(self):
        for path in (ssetup.ENGINES_LOCK, ssetup.KOKORO_LOCK):
            entries = lock_entries(path)
            self.assertTrue(entries, path)
            for req, hashes in entries:
                # no option line (another index, a link page, an editable or nested file) and no URL
                self.assertRegex(req, r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[0-9][^\s;]*( ; [^\s:/][^:/]*)?$")
                self.assertTrue(hashes, req)
                for h in hashes:
                    self.assertRegex(h, r"^[0-9a-f]{64}$")

    def test_the_lock_is_made_from_the_pins(self):
        got, want = {}, {}
        for req, _h in lock_entries(ssetup.ENGINES_LOCK):
            n, v = pin(req)
            got.setdefault(n, set()).add(v)
        for line in ssetup.lock_input().splitlines():
            n, v = pin(line)
            want.setdefault(n, set()).add(v)
        self.assertEqual(set(want), {pin(p)[0] for v in ((3, 10), (3, 11), (3, 14)) for p in ssetup.pins(v)})
        for n, vs in want.items():
            self.assertEqual(got.get(n), vs, n)                              # the pinned versions and no other
        self.assertEqual([r for r, _h in lock_entries(ssetup.KOKORO_LOCK)], [ssetup.KOKORO])

    def test_the_lock_input_gives_each_pin_its_pythons(self):
        text = ssetup.lock_input()
        self.assertIn('numpy==2.2.6 ; python_version == "3.10"\n', text)
        self.assertIn('numpy==2.4.4 ; python_version == "3.11"\n', text)
        self.assertIn('numpy==2.5.4 ; python_version >= "3.12"\n', text)
        for p in ssetup.COMMON:
            self.assertIn("\n%s\n" % p, text)

    def test_a_changed_lock_is_a_new_set(self):
        before = ssetup.pins_id()
        changed = os.path.join(tempfile.mkdtemp(dir=TMP), "engines.lock")
        with open(ssetup.ENGINES_LOCK, "rb") as f, open(changed, "wb") as g:
            g.write(f.read() + b"\n")
        with mock.patch.object(ssetup, "ENGINES_LOCK", changed):
            self.assertNotEqual(ssetup.pins_id(), before)
        self.assertEqual(ssetup.pins_id(), before)


@unittest.skipUnless(REAL_MODELS, "set PXA_SPEECH_MODELS to a fetched model cache for the real end-to-end")
class RealEndToEnd(unittest.TestCase):
    def test_wav_stt_text_tts_wav(self):
        svc = service.SpeechService.from_config(stt="tiny.en", tts="kokoro").load()
        srv = server.SpeechServer(svc, token_file=os.path.join(TMP, "real-token"))
        srv.start_background()
        try:
            c = client.SpeechClient(srv.url, srv.token_file)
            with wave.open(os.path.join(TOOLS, "pxa_control", "speech", "fixtures", "question.wav")) as w:
                q = w.readframes(w.getnframes())
            text = c.transcribe(q)
            self.assertIn("weather", text.lower())
            reply = c.tts("You asked: " + text)
            out = os.path.join(TMP, "e2e.wav")
            with open(out, "wb") as f:
                f.write(wav(reply))
            back = c.transcribe(reply)
            self.assertIn("weather", back.lower())
            print("\n  e2e: heard %r -> spoke %.2f s -> heard back %r (%s)" % (text, len(reply) / 32000.0, back, out))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=1)
