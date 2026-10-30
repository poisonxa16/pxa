"""Latency bench for the shared speech service, the same measurements as Anibal's bench_latency (vigil README
"Latency", 2026-10-11: STT 0.26 s, TTS first chunk 0.38 s, median total 0.87 s on 8 vCPU, tiny.en -t 8, Kokoro int8
bm_george), taken THROUGH the local API (HTTP client), i.e. the path every plugin uses.

  python3 -m pxa_control.speech bench [--url URL --token-file F | --stt tiny.en --tts kokoro --threads 8]
                                      [--runs 10] [--wav question.wav] [--out result.json]

  stt        POST /v1/speech/stt of the whole utterance (batch, after the endpoint) -> text; wall time.
  tts_first  POST /v1/speech/tts of REPLY -> first PCM piece received (first clause first); wall time.
  tts_first_cached  the same with the clause cache on (a repeated first clause: bit-identical PCM, no synthesis).
  direct     (in-process service only) the same STT/TTS calls on the engines without the API.
  total      the utterance replayed in real time (20 ms frames, 300 ms lead, 1.2 s tail) into an incremental
             session; on the VAD "end" event -> finish -> tts of REPLY; time from the last voiced frame to the
             first TTS piece (= bench_latency's "median total" with an instant stub agent). Parts: endpoint (end
             event after the last voiced frame), finish (text ready; ~0 when the speculative pass is reused), tts.
"""
import argparse
import json
import os
import platform
import statistics
import time
import wave

from .vad import RATE, FRAME_BYTES, speech_end_s

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "question.wav")
REPLY = "Sure. It's twenty one degrees and sunny, with a light breeze from the west."
VIGIL = {"stt_s": 0.26, "tts_first_s": 0.38, "total_median_s": 0.87, "source": "vigil README Latency 2026-10-11"}


def _stats(xs):
    xs = sorted(xs)
    return {"median_s": round(statistics.median(xs), 3), "p90_s": round(xs[int(round(0.9 * (len(xs) - 1)))], 3),
            "min_s": round(xs[0], 3), "n": len(xs)}


def box():
    d = {"machine": platform.machine(), "load": os.getloadavg()}
    try:
        d["affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                d["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    d["quiet"] = "NOT-QUIET" if d["load"][0] > 2 else "quiet"
    return d


def direct(svc, utt, runs=10, reply=REPLY):
    """The engines called in-process, no API (= Anibal's own code path): the service overhead is API - direct."""
    stt, first = [], []
    for _ in range(runs):
        t = time.monotonic()
        svc.stt.transcribe(utt)
        stt.append(time.monotonic() - t)
    for _ in range(runs):
        t = time.monotonic()
        it = svc.tts.stream(reply)
        next(it)
        first.append(time.monotonic() - t)
        for _p in it:
            pass
    return {"stt": _stats(stt), "tts_first": _stats(first)}


def run(client, utt, runs=10, reply=REPLY, hang_ms=300, total=True):
    rep = {}
    text = client.transcribe(utt)                          # warm the path
    b"".join(client.tts_stream(reply))
    stt = []
    for _ in range(runs):
        t = time.monotonic()
        text = client.transcribe(utt)
        stt.append(time.monotonic() - t)
    rep["stt"] = dict(_stats(stt), heard=text)
    first, whole = [], []
    for _ in range(runs):
        t = time.monotonic()
        it = client.tts_stream(reply)
        next(it)
        first.append(time.monotonic() - t)
        for _p in it:
            pass
        whole.append(time.monotonic() - t)
    rep["tts_first"] = _stats(first)
    rep["tts_whole"] = _stats(whole)
    if total:
        lead = b"\0\0" * (RATE * 300 // 1000)
        tail = b"\0\0" * (RATE * 1200 // 1000)
        audio = lead + utt + tail
        end_off = len(lead) / 2.0 / RATE + speech_end_s(utt)
        tot, ends, fins, ttss, reused = [], [], [], [], 0
        for _ in range(runs):
            s = client.session(hang_ms=hang_ms)
            t0 = time.monotonic()
            t_end = None
            for i in range(0, len(audio) - FRAME_BYTES + 1, FRAME_BYTES):
                due = t0 + i / 2.0 / RATE
                d = due - time.monotonic()
                if d > 0:
                    time.sleep(d)
                if "end" in s.feed(audio[i:i + FRAME_BYTES])["events"]:
                    t_end = time.monotonic()
                    break
            if t_end is None:
                t_end = time.monotonic()
            f = s.finish()
            t_fin = time.monotonic()
            it = client.tts_stream(reply)
            next(it)
            t_first = time.monotonic()
            for _p in it:
                pass
            speech_end = t0 + end_off
            tot.append(t_first - speech_end)
            ends.append(t_end - speech_end)
            fins.append(t_fin - t_end)
            ttss.append(t_first - t_fin)
            reused += 1 if f.get("reused") else 0
            heard = f.get("text")
        rep["total"] = dict(_stats(tot), endpoint_median_s=round(statistics.median(ends), 3),
                            finish_median_s=round(statistics.median(fins), 3),
                            tts_median_s=round(statistics.median(ttss), 3), stt_reused="%d/%d" % (reused, runs),
                            heard=heard, hang_ms=hang_ms)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m pxa_control.speech bench")
    ap.add_argument("--url", default=None)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--stt", default="tiny.en")
    ap.add_argument("--tts", default="kokoro")
    ap.add_argument("--models", default=None)
    ap.add_argument("--threads", type=int, default=None, help="default: stt.default_threads() (4)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--wav", default=FIX)
    ap.add_argument("--no-total", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from .client import SpeechClient
    with wave.open(a.wav, "rb") as w:
        utt = w.readframes(w.getnframes())
    srv = None
    rep = {"box_before": box(), "vigil": VIGIL, "runs": a.runs}
    if a.url:
        client = SpeechClient(a.url, a.token_file)
    else:
        import tempfile
        from .server import SpeechServer
        from .service import SpeechService
        svc = SpeechService.from_config(stt=a.stt, tts=a.tts, cache=a.models, threads=a.threads)
        t = time.monotonic()
        svc.load()
        rep["load_and_warm_s"] = round(time.monotonic() - t, 2)
        tok = os.path.join(tempfile.mkdtemp(prefix="pxa-speech-bench-"), "token")
        srv = SpeechServer(svc, token_file=tok)
        srv.start_background()
        client = SpeechClient(srv.url, tok)
    try:
        rep["info_before"] = client.info()
        if srv is not None:
            svc.tts.cache_on = False       # primary arms: every clause synthesised (cold), like Anibal's own bench
        rep.update(run(client, utt, a.runs, total=not a.no_total))
        if srv is not None:
            rep["direct"] = direct(svc, utt, a.runs)
            svc.tts.cache_on = True
            b"".join(client.tts_stream(REPLY))
            cached = []
            for _ in range(a.runs):
                t = time.monotonic()
                it = client.tts_stream(REPLY)
                next(it)
                cached.append(time.monotonic() - t)
                for _p in it:
                    pass
            rep["tts_first_cached"] = _stats(cached)
        rep["info_after"] = client.info()
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
    rep["box_after"] = box()
    rep["vs_vigil"] = {"stt": round(rep["stt"]["median_s"] / VIGIL["stt_s"], 3),
                       "tts_first": round(rep["tts_first"]["median_s"] / VIGIL["tts_first_s"], 3)}
    if "total" in rep:
        rep["vs_vigil"]["total"] = round(rep["total"]["median_s"] / VIGIL["total_median_s"], 3)
    out = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    print(out)
    return 0
