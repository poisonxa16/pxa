"""python3 -m pxa_control.speech <command>

  setup [--languages] [--force] [--status]    install the speech engines (pinned wheels, into Control's own folder) and
                                              the voice models (setup.py); --languages adds the multilingual ones,
                                              --status prints what is installed and what a setup would download (JSON)
  fetch [--stt tiny.en] [--tts kokoro] [--extra piper-de]
                                              download the pinned models once (licence allowlist + sha256);
                                              --stt base|small = multilingual hearing, --extra piper-de = a German voice
  serve [--stt tiny.en|base.en|server|...] [--tts kokoro|stub] [--port 0] [--threads N] [--voice bm_george]
                                              one warm process on 127.0.0.1; writes <config>/speech/service.json
  stt-pipe                                    PCM (16 kHz mono s16le) on stdin -> text on stdout via the service
                                              (= Scribe's existing exec: backend, no Scribe change)
  transcribe <in.wav>                         -> text
  say <text> -o <out.wav>                     -> a 16 kHz WAV (first clause first, written as it arrives)
  info                                        the running service's engines and counters
  bench [...]                                 latency bench (see bench.py)
"""
import argparse
import json
import os
import signal
import sys
import wave


def _term(*_):
    raise KeyboardInterrupt()


def _serve(a):
    from .server import SpeechServer
    from .service import SpeechService
    from .service import best_stt
    stt = a.stt or os.environ.get("PXA_SPEECH_STT") or best_stt(a.models)     # --stt, else PXA_SPEECH_STT, else the best fetched
    svc = SpeechService.from_config(stt=stt, tts=a.tts, cache=a.models, threads=a.threads, voice=a.voice,
                                    whisper_url=a.whisper_url, fetch=a.fetch,
                                    log=lambda m: print(m, file=sys.stderr))
    srv = SpeechServer(svc, port=a.port, token_file=a.token_file)
    if not a.lazy:
        svc.load()
    path = srv.write_discovery(a.discovery)
    print(json.dumps({"url": srv.url, "discovery": path, "load_s": svc.load_s}), flush=True)
    signal.signal(signal.SIGTERM, _term)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        try:
            with open(path, encoding="utf-8") as f:
                if json.load(f).get("pid") == os.getpid():
                    os.remove(path)          # our own discovery file only
        except (OSError, ValueError):
            pass
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m pxa_control.speech")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("setup")
    g.add_argument("--languages", action="store_true", help="also the multilingual hearing model and the German voice")
    g.add_argument("--force", action="store_true", help="reinstall the engines even when this set is installed")
    g.add_argument("--status", action="store_true", help="print what is installed and what a setup would download (JSON)")
    g.add_argument("--check", default=None, metavar="FOLDER", help=argparse.SUPPRESS)    # setup.run's fresh-process check
    f = sub.add_parser("fetch")
    f.add_argument("--stt", default="tiny.en")
    f.add_argument("--tts", default="kokoro")
    f.add_argument("--extra", action="append", default=[], help="extra voices: piper-de")
    f.add_argument("--models", default=None)
    s = sub.add_parser("serve")
    s.add_argument("--stt", default=None, help="tiny.en|base.en|small.en|base|small|server (default: PXA_SPEECH_STT, else the best downloaded, multilingual first)")
    s.add_argument("--tts", default="kokoro")
    s.add_argument("--models", default=None)
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--voice", default=None)
    s.add_argument("--whisper-url", default=None)
    s.add_argument("--token-file", default=None)
    s.add_argument("--discovery", default=None)
    s.add_argument("--fetch", action="store_true", help="download missing pinned models first")
    s.add_argument("--lazy", action="store_true", help="load the engines on the first call, not at start")
    sub.add_parser("stt-pipe")
    t = sub.add_parser("transcribe")
    t.add_argument("wav")
    y = sub.add_parser("say")
    y.add_argument("text")
    y.add_argument("-o", "--out", required=True)
    y.add_argument("--voice", default=None)
    sub.add_parser("info")
    sub.add_parser("bench", add_help=False)
    a, rest = ap.parse_known_args(argv)

    from . import setup
    if a.cmd != "setup":
        setup.add_site()                     # the engines `setup` installed, ahead of the system's own packages
    if a.cmd == "bench":
        from . import bench
        return bench.main(rest)
    if rest:
        ap.error("unrecognized arguments: %s" % " ".join(rest))
    if a.cmd == "setup":
        if a.check:
            r = setup.check(a.check)
            print(json.dumps(r))
            return 0 if r["ok"] else 1
        if a.status:
            print(json.dumps(setup.status(), indent=1))
            return 0
        rc = setup.run(languages=a.languages, force=a.force, say=lambda m: print(m, file=sys.stderr, flush=True))
        if rc == 2:
            print("A speech setup is already running.", file=sys.stderr)
        return rc
    if a.cmd == "fetch":
        from . import registry
        from .service import STT_MODELS, TTS_MODELS, TTS_EXTRAS
        reg = registry.load_registry()
        bad = [x for x in a.extra if x not in TTS_EXTRAS]
        if bad:
            ap.error("unknown --extra %s (known: %s)" % (", ".join(bad), ", ".join(sorted(TTS_EXTRAS))))
        for mid in [STT_MODELS[a.stt], TTS_MODELS[a.tts]] + [TTS_EXTRAS[x] for x in a.extra]:
            print(registry.fetch(reg[mid], cache=a.models, log=lambda m: print(m, file=sys.stderr)))
        return 0
    if a.cmd == "serve":
        return _serve(a)
    from .client import SpeechClient
    c = SpeechClient()
    if a.cmd == "stt-pipe":
        sys.stdout.write(c.transcribe(sys.stdin.buffer.read()))
        return 0
    if a.cmd == "transcribe":
        with wave.open(a.wav, "rb") as w:
            print(c.transcribe(w.readframes(w.getnframes())))
        return 0
    if a.cmd == "say":
        it = c.tts_stream(a.text, voice=a.voice)
        with wave.open(a.out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(c.tts_rate)
            for pcm in it:
                w.writeframes(pcm)
        print(a.out)
        return 0
    if a.cmd == "info":
        print(json.dumps(c.info(), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
