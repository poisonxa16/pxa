"""PXA shared speech service (plugin plan S5): host-provided ``speech.stt`` (incremental), ``speech.tts`` (first clause
first) and VAD as ONE warm process with ONE model download, shared by Dictate, Scribe, Satellite and Co-host.

Host-agnostic on purpose: a plain local API (server.py, pxa-speech/1 on loopback with a 0600 token file) and a
stdlib Python client (client.py, with the Scribe / Satellite / Dictate adapters). The plugin host (S2) maps its
``speech.*`` host calls onto SpeechClient (or onto an in-process SpeechService) later; nothing here imports the host.

Engines are Anibal's voice stack (PXA's own): whisper.cpp (MIT; in-process via
pywhispercpp, MIT) for STT, Kokoro-82M int8 ONNX (Apache-2.0 weights; kokoro-onnx MIT + onnxruntime MIT) for TTS,
the stdlib energy VAD, and the pinned model registry (licence allowlist, sha256, no pickle). Optional extras:
``pip install pywhispercpp kokoro-onnx onnxruntime numpy huggingface_hub``; without them the service still runs
with the stub/fake engines (tests) and refuses real calls with a clear 503.

Modules: registry (models.json + fetch), vad, stt, tts, service (SpeechService), server (HTTP API), client
(SpeechClient + adapters), bench (latency vs vigil), __main__ (fetch / serve / stt-pipe / transcribe / say / info /
bench). README.md has the API and the exact adapter wiring for each plugin.
"""
API = "pxa-speech/1"
