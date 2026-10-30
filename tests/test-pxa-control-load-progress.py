#!/usr/bin/env python3
"""PXA Control bug #373: a big first load from a slow disk read ~53 GB for 5 min with no log output and was
reported as "won't load". Control now samples the server child's /proc/<pid>/io while the model loads,
shows bytes read / rate / percent / ETA against the GPU buffer sizes the engine printed, and "silent" means
no log output AND no read progress. No GPU, no model: a fake /proc io reader and fed log lines.

    python3 tests/test-pxa-control-load-progress.py   (wired into CTest as test-pxa-control-load-progress)
"""
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-loadp-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_MODELS_DIR", None)
os.environ["PXA_CONTROL_DISCOVER"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402

MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)
FAKE_MODEL = os.path.join(MODELS, "tiny.gguf")
with open(FAKE_MODEL, "wb") as f:
    f.write(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)

MiB = 1024 * 1024
# the four lines of the #373 report (5x P100, 64 GB PXQN file)
BUG_LINES = [
    "llm_load_tensors: offloaded 50/50 layers to GPU",
    "llm_load_tensors:        CPU buffer size = 52779.64 MiB",
    "llm_load_tensors:      CUDA0 buffer size = 12804.12 MiB",
    "llm_load_tensors:      CUDA1 buffer size = 12969.83 MiB",
    "llm_load_tensors:      CUDA2 buffer size = 12972.88 MiB",
    "llm_load_tensors:      CUDA3 buffer size = 14212.62 MiB",
]
BUG_GPU_BYTES = int(round((12804.12 + 12969.83 + 12972.88 + 14212.62) * MiB))


def io_text(read_bytes, rchar):
    return (f"rchar: {rchar}\nwchar: 10\nsyscr: 5\nsyscw: 1\nread_bytes: {read_bytes}\n"
            f"write_bytes: 0\ncancelled_write_bytes: 0\n")


class ProcIo(unittest.TestCase):
    def opener(self, text=None, exc=None):
        def _open(path, *a, **k):
            self.assertEqual(path, "/proc/4242/io")
            if exc:
                raise exc
            return io.StringIO(text)
        return _open

    def test_read_bytes_is_used(self):
        self.assertEqual(C.read_proc_io(4242, opener=self.opener(io_text(5000, 100)), platform="linux"), 5000)

    def test_rchar_when_larger(self):
        # a file already in the page cache read with read(): storage read_bytes stays 0, rchar grows
        self.assertEqual(C.read_proc_io(4242, opener=self.opener(io_text(0, 7000)), platform="linux"), 7000)

    def test_permission_error_and_missing_file_give_none(self):
        self.assertIsNone(C.read_proc_io(4242, opener=self.opener(exc=PermissionError(13, "denied")), platform="linux"))
        self.assertIsNone(C.read_proc_io(4242, opener=self.opener(exc=FileNotFoundError(2, "gone")), platform="linux"))
        self.assertIsNone(C.read_proc_io(4242, opener=self.opener("garbage\n"), platform="linux"))

    def test_other_os_does_nothing(self):
        def boom(*a, **k):
            raise AssertionError("must not touch /proc off Linux")
        for plat in ("win32", "darwin"):
            self.assertIsNone(C.read_proc_io(4242, opener=boom, platform=plat))
        self.assertIsNone(C.read_proc_io(None, opener=boom, platform="linux"))


class LoadProgressMath(unittest.TestCase):
    def test_gpu_buffer_sum_ignores_cpu_and_kv(self):
        lp = C.LoadProgress()
        for x in BUG_LINES + ["llama_kv_cache_init:      CUDA0 KV buffer size =  1024.00 MiB",
                              "llama_new_context_with_model:      CUDA0 compute buffer size =   300.00 MiB"]:
            lp.feed_line(x)
        self.assertEqual(lp.total_bytes, BUG_GPU_BYTES)

    def test_percent_rate_eta(self):
        lp = C.LoadProgress()
        for x in BUG_LINES:
            lp.feed_line(x)
        t0 = 1000.0
        lp.sample(2 * 10 ** 9, t0)                         # baseline: what the child read before the weights
        rate = 180 * 10 ** 6
        for i in range(1, 6):
            lp.sample(2 * 10 ** 9 + rate * 2 * i, t0 + 2 * i)
        s = lp.summary(t0 + 10)
        self.assertEqual(s["read_bytes"], rate * 10)
        self.assertAlmostEqual(s["rate_mb_s"], 180.0, places=1)
        self.assertEqual(s["total_bytes"], BUG_GPU_BYTES)
        self.assertAlmostEqual(s["percent"], 100.0 * rate * 10 / BUG_GPU_BYTES, places=1)
        self.assertAlmostEqual(s["eta_s"], (BUG_GPU_BYTES - rate * 10) / rate, delta=1.0)
        self.assertFalse(s["slow_disk"])
        self.assertEqual(s["text"], "Loading weights onto the cards: 1.8 of 55.5 GB (180 MB/s, about 5 min left)")

    def test_percent_caps_at_100_and_no_eta(self):
        lp = C.LoadProgress()
        lp.feed_line("llm_load_tensors:      CUDA0 buffer size = 100.00 MiB")
        lp.sample(0, 0.0)
        lp.sample(200 * MiB, 2.0)
        s = lp.summary(2.0)
        self.assertEqual(s["percent"], 100.0)
        self.assertIsNone(s["eta_s"])

    def test_no_total_yet(self):
        lp = C.LoadProgress()
        lp.sample(0, 0.0)
        lp.sample(400 * 10 ** 6, 2.0)
        s = lp.summary(2.0)
        self.assertIsNone(s["percent"])
        self.assertIsNone(s["eta_s"])
        self.assertEqual(s["text"], "Reading the model file: 0.4 GB so far (200 MB/s)")

    def test_no_samples_no_summary(self):
        self.assertIsNone(C.LoadProgress().summary(5.0))

    def test_eta_wording(self):
        self.assertEqual(C.eta_words(40), "about 40 s left")
        self.assertEqual(C.eta_words(130), "about 2 min left")
        self.assertEqual(C.eta_words(3 * 3600), "about 3.0 h left")

    def test_slow_disk_hint_after_60_s_under_50_mb_s(self):
        lp = C.LoadProgress()
        for x in BUG_LINES:
            lp.feed_line(x)
        b = 0
        for i in range(0, 31):                             # 60 s at 20 MB/s
            lp.sample(b, 2.0 * i)
            b += 40 * 10 ** 6
        s = lp.summary(60.0)
        self.assertTrue(s["slow_disk"])
        self.assertIn("The model file is on a slow disk", s["text"])
        self.assertIn("page cache", s["text"])
        self.assertNotIn("PXA_", s["text"])
        self.assertNotIn("—", s["text"])

    def test_slow_needs_60_s(self):
        lp = C.LoadProgress()
        for i in range(0, 20):                             # 38 s at 20 MB/s: too soon to say
            lp.sample(i * 40 * 10 ** 6, 2.0 * i)
        self.assertFalse(lp.summary(38.0)["slow_disk"])

    def test_fast_disk_no_hint(self):
        lp = C.LoadProgress()
        for i in range(0, 40):
            lp.sample(i * 400 * 10 ** 6, 2.0 * i)
        s = lp.summary(78.0)
        self.assertFalse(s["slow_disk"])
        self.assertNotIn("slow disk", s["text"])

    def test_last_progress_moves_only_on_growth(self):
        lp = C.LoadProgress()
        lp.sample(1000, 0.0)
        self.assertIsNone(lp.last_progress_ts)             # the baseline is not progress
        lp.sample(1000 + 50 * MiB, 2.0)
        self.assertEqual(lp.last_progress_ts, 2.0)
        lp.sample(1000 + 50 * MiB + 100, 4.0)              # a few bytes of /proc chatter: not progress
        self.assertEqual(lp.last_progress_ts, 2.0)


class SeatSilence(unittest.TestCase):
    def setUp(self):
        self.seat = C.Seat(L)
        self.seat.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.seat.req = {"port": 8999, "gpus": [0], "model": FAKE_MODEL}
        self.seat.started = 0.0
        self.bytes = [0]
        self.seat.io_reader = lambda pid: self.bytes[0]

    def tearDown(self):
        if self.seat.proc.poll() is None:
            self.seat.proc.kill()
            self.seat.proc.wait()

    def feed(self, line, now):
        self.seat._append(line)
        self.seat.last_line_ts = now
        self.seat._load_line(line, now=now)

    def test_silent_does_not_grow_while_reads_grow(self):
        t = 1000.0
        self.feed("llama_model_loader: loaded meta data", t)
        self.assertEqual(self.seat.phase, "loading model")
        for x in BUG_LINES:
            self.feed(x, t)
        self.assertEqual(self.seat.phase, "weights on the GPUs")
        for i in range(1, 157):                            # 312 s of no log output while ~180 MB/s comes off the disk
            self.bytes[0] += 360 * 10 ** 6
            self.seat._sample_load(now=t + 2 * i)
        now = t + 312
        self.assertLessEqual(self.seat.silent_age(now), 2.0)
        st = self.seat.status(now=now)
        self.assertLessEqual(st["silent_age"], 2.0)
        self.assertGreaterEqual(st["last_line_age"], 312 - 1)
        self.assertGreater(st["load_progress"]["percent"], 90)
        self.assertTrue(st["load_progress"]["text"].startswith("Loading weights onto the cards: "))

    def test_silent_grows_when_both_stop(self):
        t = 1000.0
        self.feed("llama_model_loader: loaded meta data", t)
        for x in BUG_LINES:
            self.feed(x, t)
        for i in range(1, 11):
            self.bytes[0] += 360 * 10 ** 6
            self.seat._sample_load(now=t + 2 * i)
        for i in range(11, 160):                           # the reads stop too
            self.seat._sample_load(now=t + 2 * i)
        now = t + 320
        self.assertAlmostEqual(self.seat.silent_age(now), 300.0, delta=0.5)

    def test_no_proc_io_path_is_unchanged(self):
        self.seat.io_reader = lambda pid: None             # other OS / permission error
        t = 1000.0
        self.feed("llama_model_loader: loaded meta data", t)
        for x in BUG_LINES:
            self.feed(x, t)
        for i in range(1, 50):
            self.seat._sample_load(now=t + 2 * i)
        st = self.seat.status(now=t + 300)
        self.assertIsNone(st["load_progress"])
        self.assertAlmostEqual(st["silent_age"], 300.0, delta=0.5)
        self.assertAlmostEqual(st["last_line_age"], 300.0, delta=0.5)

    def test_sampling_only_while_loading(self):
        t = 1000.0
        self.seat._sample_load(now=t)                      # phase "stopped": nothing
        self.assertIsNone(self.seat.loadp.summary(t))
        self.feed("llama_model_loader: loaded meta data", t)
        self.bytes[0] = 10 ** 9
        self.feed("main: server is listening on 127.0.0.1:8999", t + 5)
        self.assertEqual(self.seat.phase, "listening")
        self.seat._sample_load(now=t + 6)
        self.assertNotEqual(self.seat.loadp.summary(t + 6)["read_bytes"], 10 ** 9)


class PumpFeedsProgress(unittest.TestCase):
    def test_child_lines_reach_the_tracker(self):
        s = C.Seat(L)
        code = ("print('llama_model_loader: x', flush=True)\n"
                "print('llm_load_tensors: offloaded 2/2 layers to GPU', flush=True)\n"
                "print('llm_load_tensors:      CUDA0 buffer size = 100.00 MiB', flush=True)\n"
                "print('llm_load_tensors:      CUDA1 buffer size = 50.00 MiB', flush=True)\n")
        s.start((L.Plan(), [sys.executable, "-u", "-c", code], {}, "0", {}, 4096),
                {"port": 8999, "model": "m", "gpus": [0]}, {})
        s.proc.wait(timeout=20)
        deadline = time.time() + 5
        while time.time() < deadline and "exited" not in s.phase:
            time.sleep(0.05)
        self.assertEqual(s.loadp.total_bytes, int(150 * MiB))


class ReportBundle(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        self.app.catalog_names = set()

    def tearDown(self):
        for s in self.app.seats.values():
            if s.proc and s.proc.poll() is None:
                s.proc.kill()
                s.proc.wait()

    def test_bundle_carries_load_progress_and_silent_uses_reads(self):
        seat = self.app.seat
        seat.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        seat.req = {"port": 8999, "gpus": [0], "model": FAKE_MODEL}
        now = time.time()
        seat.started = now - 400
        b = [0]
        seat.io_reader = lambda pid: b[0]
        seat._append("llama_model_loader: loaded meta data")
        for x in BUG_LINES:
            seat._append(x)
            seat._load_line(x, now=now - 312)
        seat.phase = "weights on the GPUs"
        seat.last_line_ts = now - 312
        seat._sample_load(now=now - 312)
        b[0] = 30 * 10 ** 9
        seat._sample_load(now=now - 1)
        bu = self.app.report_bundle()["bundle"]
        self.assertIn("load_progress", bu)
        self.assertEqual(bu["load_progress"]["read_bytes"], 30 * 10 ** 9)
        self.assertLessEqual(bu["silent_s"], 5)

    def test_bundle_without_proc_io(self):
        bu = self.app.report_bundle()["bundle"]
        self.assertIn("load_progress", bu)
        self.assertIsNone(bu["load_progress"])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
