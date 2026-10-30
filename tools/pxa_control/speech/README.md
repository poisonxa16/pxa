# Shared speech service (plugin plan S5)

One warm process gives every voice plugin (Dictate, Scribe, Satellite, Co-host) the same three things:

- `speech.stt`: batch (one utterance) or incremental (a session you feed as audio arrives; the transcription
  of the speech so far starts during the pause, so end of speech to text costs about zero when nothing more is said);
- `speech.tts`: streamed, first clause first (a short first clause, then whole sentences; clause N+1 is synthesised
  while clause N is on the wire);
- VAD: the energy VAD with adaptive noise floor and echo gate.

The models are downloaded once into one cache (`$PXA_SPEECH_MODELS`, else `<Control config>/models/speech`) from the
pinned registry `models.json`: whisper.cpp ggml tiny.en/base.en (MIT) and Kokoro-82M int8 ONNX (Apache-2.0), each
file sha256-checked, licence on the voice allowlist, no pickle formats. The engines and the registry rules are
Anibal's voice stack (PXA's own).

Optional extras (CPU wheels, no build): `pip install pywhispercpp kokoro-onnx onnxruntime numpy huggingface_hub`.
Licence note kept from Anibal: kokoro-onnx phonemises with espeak-ng loaded in-process through `espeakng-loader`
(GPL-3.0): that is the user's own pip install, never bundled in a package.

## Run

    python3 -m pxa_control.speech fetch                    # once: whisper tiny.en + Kokoro int8 (~170 MB)
    python3 -m pxa_control.speech fetch --stt small --extra piper-de   # other languages: multilingual whisper + German voice
    python3 -m pxa_control.speech serve [--stt tiny.en|base.en|base|small] [--threads 8] [--voice bm_george]
    python3 -m pxa_control.speech transcribe in.wav
    python3 -m pxa_control.speech say "Sure. It is sunny." -o out.wav
    python3 -m pxa_control.speech bench --runs 10          # latency vs Anibal's numbers

`serve` binds 127.0.0.1 only, loads and warms both engines, writes `<config>/speech/service.json`
(`{"url", "token_file", "pid"}`, 0600) and a random token in `<config>/speech/token` (0600). Clients read the
token BY PATH; it never appears in a reply, a log or the discovery file. `PXA_SPEECH_GPU=1` lets onnxruntime use
a GPU provider when the installed wheel has one (CPU otherwise).

## Languages (STT hint, voices)

English-only whisper files (`tiny.en`, `base.en`, `small.en`) hear English only. For other languages run a
multilingual file: `fetch --stt base` (142 MB) or `--stt small` (466 MB, clearly better outside English), then
`serve --stt small`. whisper.cpp keeps its language between calls and starts at English, so the service passes a
language on EVERY call: the caller's hint (`?language=es` on `/v1/speech/stt`, `"language"` on a session; ISO 639-1,
region dropped) or `auto` (whisper detects). A known speaker language helps a lot (German on whisper-base: chrF
0.42 with the hint, 0.26 on auto).

Round trip (a voice reads 10 reference sentences, whisper hears them, chrF vs the text; 2026-10-20, CPU, busy box):
whisper-base: en 0.91, es 0.75, fr 0.53, de 0.42; whisper-small: es 0.91, fr 0.81, de 0.62, ja 0.18 (with the reader
below). Hindi fails on whisper-base (chrF 0.00) and is very slow on whisper-small on a CPU;
treat Hindi speech input as unproven and type Hindi instead.

Voices: the registry pins Kokoro voices for English, Spanish (ef_dora, em_alex), French (ff_siwis), Hindi (hf_alpha,
hm_omega), Italian, Portuguese, Japanese (jf_alpha, jm_kumo) and Chinese. Kokoro has no German voice:
`fetch --extra piper-de` adds the Piper voice `de_DE-ramona-low` (trained from scratch on M-AILABS, BSD-style
licence; run as plain ONNX, no piper package). Piper German voices fine-tuned from lessac/ryan checkpoints or trained
on NC data (thorsten, kerstin, karlsson, pavoque) are refused on licence. `PXA_SPEECH_EXTRAS=0` leaves the extras off.

Japanese needs a kanji reader: espeak-ng has no kanji dictionary and reads every kanji as "Chinese letter". With
`pip install pyopenjtalk-plus` (MIT; OpenJTalk BSD-3; SudachiPy and its dictionary Apache-2.0) kanji are read to kana
first; without it the Japanese voices drop out (`info` lists them under `tts.missing`) and a plugin shows Japanese as
text only. `PXA_SPEECH_JA_READER=0` = off. The Kokoro Japanese voices were trained on a different phoneme set
(misaki), so Japanese speech stays rough even with the reader (round trip 0.18); misaki's own Japanese reader did not
install here (mojimoji has no wheel for this Python).

Credits: Kokoro-82M (Apache-2.0) voices ff_siwis (SIWIS corpus) and jm_kumo (Koniwa / Aozora Bunko readings);
Piper voice de_DE-ramona-low (M-AILABS).

## API (pxa-speech/1)

Audio in: 16 kHz mono s16le, raw (`audio/L16`) or a WAV of exactly that format. Every call except `health` needs
`Authorization: Bearer <token>`; requests from a non-loopback peer, with a non-loopback `Host` (DNS rebinding) or with
an `Origin` header (browsers) get 403.

| call | body | reply |
|---|---|---|
| `GET /v1/speech/health` | - | `{"ok": true, "api": "pxa-speech/1"}` |
| `GET /v1/speech/info` | - | engines, rates, voices, load/warm seconds, call counters |
| `POST /v1/speech/stt[?language=es]` | audio | `{"text", "ms"}` (+ `"language"` on a multilingual model) |
| `POST /v1/speech/stt/sessions` | `{"partial_s": 0, "hang_ms": 300, "language"?}` | `{"id"}` (max 8 open, 60 s idle expiry) |
| `POST /v1/speech/stt/sessions/<id>` | audio, any size | `{"events": ["start", "end"], "partial"}` |
| `POST /v1/speech/stt/sessions/<id>/finish` | - | `{"text", "reused", "ms", "audio_s"}` (closes it) |
| `DELETE /v1/speech/stt/sessions/<id>` | - | `{"closed"}` |
| `POST /v1/speech/tts` | `{"text", "voice"?, "speed"?}` | chunked `audio/L16`, one HTTP chunk per clause, `X-Speech-Rate` |
| `POST /v1/speech/vad` | audio | `{"segments": [[start_s, end_s]], "speech_end_s"}` |

Limits: 120 s of audio per call/utterance, 4000 characters per TTS call. Closing the TTS connection mid-stream
(barge-in) stops the synthesis.

## Python client and the plugin adapters (`client.py`, stdlib only)

```python
from pxa_control.speech.client import SpeechClient, ScribeSTT, SatelliteSpeech, DictateRecognizer
c = SpeechClient()                 # finds the running service; or SpeechClient(url, token_file)
c.transcribe(pcm); s = c.session(); s.feed(pcm); s.finish()["text"]
for piece in c.tts_stream("Sure. It is sunny."): play(piece)        # c.tts_rate after the first piece
```

The adapters are duck-typed (they import nothing from a plugin), so each plugin wires them with a few lines:

- **Scribe** (`scribe/lib/scribe/stt.py`): works TODAY with no change through its exec backend:
  `--stt "exec:python3 -m pxa_control.speech stt-pipe"` (PCM on stdin, words on stdout, no file). The direct
  backend is two lines in `make_stt`:

      if spec == "speech":
          from pxa_control.speech.client import ScribeSTT; return ScribeSTT()

  `ScribeSTT.transcribe(pcm, channel)` and `close()` match `scribe.stt.STT`.
- **Satellite** (`satellite/lib/satellite/__main__.py`, where `speech = B.HostSpeech()`):

      try:
          from pxa_control.speech.client import SatelliteSpeech; speech = SatelliteSpeech()
      except Exception:
          speech = B.HostSpeech()          # unchanged message when the service is not there

  `SatelliteSpeech` has `start(rate, width, channels, language)` -> a session with `feed/partial/finish`
  (any Wyoming format is converted to 16 kHz mono), `synth(sentence, voice)` -> PCM at `rate` (16000), `width` 2,
  `channels` 1, and `voices`/`languages`, the `satellite.backends.Stt`/`Tts` protocol used by `server.py`.
  A `SpeechUnavailable`/`SpeechClientError` from a call should map to Satellite's `BackendUnavailable` /
  `BackendError` (one `except` in its `run_blocking` callers, or subclass in `backends.py`).
- **Dictate** (`dictate/lib/dictate/run.py`, `model.recognize(pcm, vocab=vocab)`): `DictateRecognizer().recognize`
  is the open-vocabulary drop-in for the lexicon model.
- **Co-host**: `SpeechClient` directly (sessions for the mic, `tts_stream` for replies).
- **Plugin host (S2)**: map the host calls `speech.stt` / `speech.stt.session.*` / `speech.tts` / `speech.vad` onto
  `SpeechClient` (or an in-process `SpeechService`) and start `python3 -m pxa_control.speech serve` once, on first use.
  Proposed manifest permissions: `speech.listen` (stt + vad) and `speech.speak` (tts); none exist in
  `pxa_control.plugins.PERMISSIONS` yet (S2 owns that table).

## Latency (bench.py, CPU only)

Measured through the HTTP client, the same quantities as Anibal's `bench_latency` (its README "Latency": STT 0.26 s,
TTS first chunk 0.38 s, total 0.87 s on an 8 vCPU KVM Xeon, tiny.en `-t 8`, Kokoro int8 `bm_george`).
This box: Xeon E5-2699 v3, 18 shared logical cores (VMs and builds running, load 29-31, NOT quiet), 10 runs,
whisper tiny.en (encoder context sized to the utterance), Kokoro int8 at 1.12x, 4 threads each, 2026-10-19.

| quantity | service (API) | engines direct (Anibal's code path, same box) | Anibal's README |
|---|---|---|---|
| STT, question.wav 1.8 s (batch after the endpoint) | **0.184 s** (p90 0.187) | 0.183 s | 0.26 s |
| TTS first chunk, "Sure." (cold, clause cache off) | **0.519 s** (p90 0.528) | 0.522 s | 0.38 s |
| TTS first chunk, repeated first clause (cache on) | 0.001 s | - | - |
| total, end of speech -> first TTS audio (300 ms hang, instant agent) | **0.828 s** (p90 0.842) | - | 0.87 s |

Total = endpoint 0.281 + finish 0.001 (the pause-time transcription was reused 10/10) + TTS 0.545. The API adds
about 1 ms to either call. With the full 30 s whisper window and 8 threads (the first build) the same box gave
STT 1.05 s and total 2.47 s. Kokoro's cold first clause is about 35% slower than on Anibal's test machine on this
older, shared CPU; the engine path without the service is the same 0.52 s.
