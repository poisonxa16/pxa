"""Stdlib voice-activity detection: energy with an adaptive noise floor (vendored from Anibal voice vad.py).

All audio in the speech service is 16 kHz mono signed 16-bit little-endian PCM; a frame is 20 ms = 640 bytes.
``echo_level``: while TTS plays, set it to the TTS level so speaker bleed does not count as speech (a frame must
beat echo_gate x that level; gating, not echo cancellation).
"""
import array
import math

RATE = 16000
FRAME_MS = 20
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2      # 640


def rms(pcm):
    """RMS of s16le mono PCM, scaled to 0..1."""
    if not pcm:
        return 0.0
    a = array.array("h")
    a.frombytes(bytes(pcm[: len(pcm) - len(pcm) % 2]))
    if not a:
        return 0.0
    return math.sqrt(sum(x * x for x in a) / len(a)) / 32768.0


class EnergyVAD:
    def __init__(self, threshold=0.02, ratio=3.0, start_ms=60, hang_ms=300, frame_ms=FRAME_MS, echo_gate=0.5):
        self.threshold, self.ratio = threshold, ratio
        self.echo_gate = echo_gate
        self.start_n = max(1, start_ms // frame_ms)
        self.hang_n = max(1, hang_ms // frame_ms)
        self.reset()

    def reset(self):
        self.floor, self.run, self.quiet, self.in_speech = 0.002, 0, 0, False
        self.echo_level = 0.0

    def is_voiced(self, pcm):
        lvl = rms(pcm)
        gate = max(self.threshold, self.floor * self.ratio, self.echo_level * self.echo_gate)
        voiced = lvl >= gate
        if not voiced:  # track the noise floor on unvoiced frames only
            self.floor = 0.95 * self.floor + 0.05 * lvl
        return voiced

    def feed(self, pcm):
        """-> 'start' | 'end' | None for this frame."""
        v = self.is_voiced(pcm)
        if not self.in_speech:
            self.run = self.run + 1 if v else 0
            if self.run >= self.start_n:
                self.in_speech, self.quiet = True, 0
                return "start"
            return None
        self.quiet = 0 if v else self.quiet + 1
        if self.quiet >= self.hang_n:
            self.in_speech, self.run = False, 0
            return "end"
        return None


def segments(pcm, **kw):
    """Offline pass -> [(start_s, end_s), ...] of speech (end = the 'end' event time, i.e. after the hang)."""
    v, out, start = EnergyVAD(**kw), [], None
    n = len(pcm) - len(pcm) % FRAME_BYTES
    for i in range(0, n, FRAME_BYTES):
        ev = v.feed(pcm[i:i + FRAME_BYTES])
        t = (i + FRAME_BYTES) / 2.0 / RATE
        if ev == "start":
            start = max(0.0, t - v.start_n * FRAME_MS / 1000.0)
        elif ev == "end" and start is not None:
            out.append((round(start, 3), round(t, 3)))
            start = None
    if start is not None:
        out.append((round(start, 3), round(n / 2.0 / RATE, 3)))
    return out


def speech_end_s(pcm):
    """Time (s) right after the last voiced frame (bench_latency's 'end of speech')."""
    v, last = EnergyVAD(), 0.0
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        if v.is_voiced(pcm[i:i + FRAME_BYTES]):
            last = (i + FRAME_BYTES) / 2.0 / RATE
    return last
