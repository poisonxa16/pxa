# SPDX-License-Identifier: Apache-2.0
"""Per-step decode instrument: host wall vs GPU busy vs host gap.

WHY THIS EXISTS. A chrome trace tells you which kernels ran; it does not tell you,
in one line, whether a 33 ms decode token is 33 ms of GPU work or 12 ms of GPU work
behind 21 ms of Python. That single split decides whether the next fix belongs in a
kernel or in the runner, and every hour spent reading a trace without it is a guess.

MECHANISM. Wraps the v1 model runner's ``execute_model``:
  * ``time.perf_counter()`` either side  -> HOST WALL for the step (what the token rate
    is actually made of);
  * a pair of ``torch.cuda.Event``s recorded on the current stream inside the wrapper
    -> GPU BUSY for the step, including a captured graph's replay (an event recorded
    before and after ``cudaGraphLaunch`` brackets every node in it).
  * GAP = wall - busy: the part of the token that is Python, the scheduler, sampling,
    detokenisation and launch latency rather than arithmetic.

NO SYNCHRONISATION IN THE HOT PATH. ``Event.elapsed_time`` needs the events to have
completed, and asking for that inline would serialise the very pipeline being measured.
So events go into a ring and are read back RING_LAG steps later, by which time the work
is long finished. The instrument therefore perturbs the thing it measures by two event
records per step and nothing else.

OFF BY DEFAULT. ``PXA_STEP_TIMER=1`` arms it; anything else and this module does not
patch. Env:
  PXA_STEP_TIMER=1          arm
  PXA_STEP_TIMER_EVERY=64   report period, in steps
  PXA_STEP_TIMER_WARMUP=16  steps discarded before the first report window
"""

from __future__ import annotations

import os
import time

_ARMED = False


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return s[i]


class _Ring:
    """Fixed-size ring of (start_event, end_event, host_ms, ntok, nreq)."""

    def __init__(self, lag: int) -> None:
        self.lag = lag
        self.buf: list = [None] * lag
        self.i = 0

    def push(self, item):
        out = self.buf[self.i]
        self.buf[self.i] = item
        self.i = (self.i + 1) % self.lag
        return out


def maybe_patch() -> None:
    global _ARMED
    if _ARMED or os.getenv("PXA_STEP_TIMER", "0") != "1":
        return

    import torch
    from vllm.logger import init_logger

    logger = init_logger("pxq4_vllm.step_timer")

    every = int(os.getenv("PXA_STEP_TIMER_EVERY", "64"))
    warmup = int(os.getenv("PXA_STEP_TIMER_WARMUP", "16"))
    ring = _Ring(8)

    state = {"n": 0, "wall": [], "gpu": [], "tok": 0, "batch": []}

    def _report(tag: str) -> None:
        w, g = state["wall"], state["gpu"]
        if not w:
            return
        gaps = [a - b for a, b in zip(w, g)] if len(g) == len(w) else []
        logger.info(
            "STEPTIMER %s n=%d  wall ms p50 %.2f p90 %.2f  gpu ms p50 %.2f p90 %.2f  "
            "gap ms p50 %.2f (%.0f%%)  mean batch tok %.2f  implied single-stream %.2f tok/s",
            tag, len(w), _pct(w, .5), _pct(w, .9), _pct(g, .5), _pct(g, .9),
            _pct(gaps, .5), 100.0 * _pct(gaps, .5) / max(_pct(w, .5), 1e-9),
            (sum(state["batch"]) / len(state["batch"])) if state["batch"] else 0.0,
            1000.0 / max(_pct(w, .5), 1e-9),
        )
        state["wall"], state["gpu"], state["batch"] = [], [], []

    def _wrap(cls) -> bool:
        orig = getattr(cls, "execute_model", None)
        if orig is None or getattr(orig, "_pxa_step_timer", False):
            return False

        def execute_model(self, *a, **kw):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            # Batch width, read off the SchedulerOutput without touching the GPU.
            ntok = 0
            try:
                so = a[0] if a else kw.get("scheduler_output")
                ntok = int(getattr(so, "total_num_scheduled_tokens", 0) or 0)
            except Exception:
                pass
            ev0.record()
            t0 = time.perf_counter()
            out = orig(self, *a, **kw)
            t1 = time.perf_counter()
            ev1.record()

            old = ring.push((ev0, ev1, (t1 - t0) * 1000.0, ntok))
            state["n"] += 1
            if old is not None:
                e0, e1, hostms, otok = old
                if state["n"] > warmup:
                    try:
                        state["gpu"].append(e0.elapsed_time(e1))
                        state["wall"].append(hostms)
                        state["batch"].append(otok)
                    except Exception:
                        pass
            if state["n"] % every == 0:
                _report("step%d" % state["n"])
            return out

        execute_model._pxa_step_timer = True
        cls.execute_model = execute_model
        return True

    hits = []
    for mod, name in (
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
    ):
        try:
            m = __import__(mod, fromlist=[name])
            cls = getattr(m, name, None)
            if cls is not None and _wrap(cls):
                hits.append("%s.%s" % (mod, name))
        except Exception:
            continue

    _ARMED = True
    logger.info("STEPTIMER armed on %s (every=%d warmup=%d)",
                ", ".join(hits) or "NOTHING", every, warmup)
