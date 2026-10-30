#!/usr/bin/env python3
"""Windows path rules of PXA Control, checked on any host: the model-folder test (core._path_in) with ntpath semantics,
and the launcher's /dev/shm probe on a platform without os.statvfs (RC2.1 RTX 3080 Ti smoke: /api/rig 500, /api/plan 400)."""
import importlib.util
import ntpath
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pxa_control.core as core  # noqa: E402
import pxa_control.bridge as bridge  # noqa: E402


class WinPaths(unittest.TestCase):
    def win(self):
        fake = types.SimpleNamespace(path=ntpath, sep="\\")
        return mock.patch.object(core, "os", fake)

    def test_inside_folder(self):
        with self.win():
            self.assertTrue(core._path_in(r"D:\pxa\rc21\models\q.gguf", r"D:\pxa\rc21\models"))
            self.assertTrue(core._path_in(r"d:\PXA\rc21\Models\q.gguf", r"D:\pxa\rc21\models\\"))   # case-blind, trailing sep
            self.assertTrue(core._path_in(r"D:\pxa\rc21\models", r"D:\pxa\rc21\models"))
            self.assertFalse(core._path_in(r"D:\pxa\rc21\models", r"D:\pxa\rc21\models", allow_equal=False))

    def test_sibling_prefix_refused(self):
        with self.win():
            self.assertFalse(core._path_in(r"D:\pxa\rc21\models2\q.gguf", r"D:\pxa\rc21\models"))
            self.assertFalse(core._path_in(r"E:\pxa\rc21\models\q.gguf", r"D:\pxa\rc21\models"))

    def test_posix_unchanged(self):
        self.assertTrue(core._path_in("/m/a/q.gguf", "/m/a"))
        self.assertTrue(core._path_in("/m/a/q.gguf", "/m/a/"))
        self.assertTrue(core._path_in("/x/q.gguf", "/"))
        self.assertFalse(core._path_in("/m/ab/q.gguf", "/m/a"))
        self.assertFalse(core._path_in("/M/a/q.gguf", "/m/a"))


class NoStatvfs(unittest.TestCase):
    def test_shm_bytes_without_statvfs(self):
        spec = importlib.util.spec_from_file_location("pxa_launch_t", os.path.join(ROOT, "tools", "pxa-launch.py"))
        L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(L)
        fake = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if k != "statvfs" and not k.startswith("__")})
        with mock.patch.object(L, "os", fake):
            self.assertEqual(L.shm_bytes("/dev/shm"), 0)


class FakeWinApi:
    """The ctypes layer of core._win_proc, faked: pid -> ("live"|"dead"|"denied"|"gone", created FILETIME)."""
    def __init__(self, procs):
        self.procs, self.opened, self.closed = procs, [], []

    def open(self, pid):
        kind, _ = self.procs.get(pid, ("gone", None))
        if kind == "gone":
            return None, 87                 # ERROR_INVALID_PARAMETER: no such pid
        if kind == "denied":
            return None, 5                  # ERROR_ACCESS_DENIED: it exists
        h = 1000 + pid
        self.opened.append(h)
        return h, 0

    def running(self, h):
        kind, _ = self.procs[h - 1000]
        return kind == "live"

    def created(self, h):
        return self.procs[h - 1000][1]

    def close(self, h):
        self.closed.append(h)


class FakeKernel32:
    """kernel32 as _WinApi sees it: records the OpenProcess access mask and every call, so the real _WinApi class runs."""
    def __init__(self, wait, exit_code=259, open_ok=True):
        self.wait, self.exit_code, self.open_ok = wait, exit_code, open_ok
        self.calls, self.access, self.closed = [], None, []
        self.OpenProcess = self._open
        self.WaitForSingleObject = self._wait
        self.GetExitCodeProcess = self._exit
        self.CloseHandle = lambda h: self.closed.append(h) or 1

    def _open(self, access, inherit, pid):
        self.access = access
        self.calls.append("open")
        return 77 if self.open_ok else None

    def _wait(self, h, ms):
        self.calls.append(("wait", ms))
        return self.wait

    def _exit(self, h, ref):
        self.calls.append("exitcode")
        return 1


class WinApiLayer(unittest.TestCase):
    """L1 on the real core._WinApi methods with a faked kernel32 (nothing here can run on real Windows)."""
    def api(self, k):
        a = object.__new__(core._WinApi)
        a.k = k
        return a

    def test_open_asks_for_synchronize_and_query_limited(self):
        k = FakeKernel32(wait=0x102)
        h, err = self.api(k).open(4100)
        self.assertEqual((h, err), (77, 0))
        self.assertEqual(k.access, 0x00100000 | 0x1000)

    def test_still_running_is_wait_timeout_not_exit_code_259(self):
        for wait, want in ((0x102, True), (0, False), (0xFFFFFFFF, False)):
            k = FakeKernel32(wait=wait, exit_code=259)       # a process may exit with code 259: not proof of life
            self.assertIs(self.api(k).running(77), want, hex(wait))
            self.assertIn(("wait", 0), k.calls)
            self.assertNotIn("exitcode", k.calls)

    def test_handle_always_closed_through_win_proc(self):
        k = FakeKernel32(wait=0x102)
        a = self.api(k)
        a.created = lambda h: 1111
        with mock.patch.object(core, "_win_api", lambda: a):
            self.assertEqual(core._win_proc(4100), (True, 1111))
            k.wait = 0
            self.assertEqual(core._win_proc(4100), (False, None))
            a.created = mock.Mock(side_effect=OSError("boom"))
            k.wait = 0x102
            with self.assertRaises(OSError):
                core._win_proc(4100)
        self.assertEqual(k.closed, [77, 77, 77])


class WinStopLaunched(unittest.TestCase):
    """L2: signal.SIGKILL does not exist on Windows; Stop must still forget the record when a server outlives the grace."""
    def run_stop(self, windows, has_sigkill=True):
        import signal
        kills = []
        fake = types.SimpleNamespace(last_busy=0, _fleet_cache=None)
        forgot = []
        saved = getattr(signal, "SIGKILL", None)
        if not has_sigkill:
            del signal.SIGKILL
        try:
            with mock.patch.multiple(core, _WINDOWS=windows,
                                     launched_servers=lambda: {900: {"pid": 900, "start": 5}},
                                     pid_alive=lambda pid, start=None: True,
                                     forget_launched=lambda pid: forgot.append(pid)), \
                    mock.patch.object(core.os, "kill", lambda pid, sig: kills.append((pid, sig))):
                r = core.App.stop_launched(fake, 900, grace_s=0.05)
        finally:
            if saved is not None:
                signal.SIGKILL = saved
        return r, kills, forgot

    def test_windows_skips_the_second_kill_and_forgets_the_record(self):
        import signal
        r, kills, forgot = self.run_stop(True, has_sigkill=False)
        self.assertEqual(forgot, [900])
        self.assertEqual(kills, [(900, signal.SIGTERM)])        # the first kill is already a hard kill there
        self.assertEqual((r["ok"], r["killed"], r["pid"]), (True, False, 900))

    def test_posix_still_escalates_to_sigkill(self):
        import signal
        r, kills, forgot = self.run_stop(False)
        self.assertEqual(kills, [(900, signal.SIGTERM), (900, signal.SIGKILL)])
        self.assertEqual((r["killed"], forgot), (True, [900]))


class WinLiveness(unittest.TestCase):
    """Review item 1: on Windows os.kill(pid, 0) is CTRL_C_EVENT, not a probe; Control asks kernel32 instead."""
    def win(self, procs):
        api = FakeWinApi(procs)
        kill = mock.Mock(side_effect=AssertionError("os.kill must not be called on Windows"))
        return api, mock.patch.multiple(core, _WINDOWS=True, _win_api=lambda: api), mock.patch.object(core.os, "kill", kill)

    def test_live_dead_denied_gone(self):
        api, p1, p2 = self.win({4100: ("live", 1111), 4200: ("dead", 2222), 4: ("denied", None)})
        with p1, p2:
            self.assertTrue(core.pid_alive(4100))
            self.assertFalse(core.pid_alive(4200))          # handle opens, the process has exited
            self.assertTrue(core.pid_alive(4))              # access denied, no start time recorded = it exists
            self.assertFalse(core.pid_alive(4300))          # any other OpenProcess failure = gone
            self.assertFalse(core.pid_alive(0))
            self.assertFalse(core.pid_alive("x"))
        self.assertEqual(sorted(api.opened), sorted(api.closed))   # every handle closed

    def test_start_time_from_process_times(self):
        api, p1, p2 = self.win({4100: ("live", 1111)})
        with p1, p2:
            self.assertEqual(core.proc_start_ticks(4100), 1111)
            self.assertTrue(core.pid_alive(4100, 1111))
            self.assertFalse(core.pid_alive(4100, 9999))     # pid reused by another process
            self.assertTrue(core.pid_alive(4100, None))
            self.assertIsNone(core.proc_start_ticks(4300))

    def test_access_denied_with_a_recorded_start_is_not_ours(self):
        """L1: a record that carries a start time and a PID Windows will not open (another user's process now
        owns the number) must not count as our server: the start-time check cannot run, so it is dead for us."""
        api, p1, p2 = self.win({4: ("denied", None), 4100: ("live", 1111)})
        with p1, p2:
            self.assertFalse(core.pid_alive(4, 1111))
            self.assertTrue(core.pid_alive(4))              # nothing recorded to compare: it exists
            self.assertTrue(core.pid_alive(4100, 1111))
            self.assertIsNone(core.proc_start_ticks(4))     # asking for the start time of a denied pid stays None

    def test_live_records_drop_a_denied_pid_with_a_start(self):
        api, p1, p2 = self.win({4: ("denied", None)})
        import json, tempfile
        d = tempfile.mkdtemp(prefix="pxa-winlive-")
        os.makedirs(os.path.join(d, "runs"))
        with open(os.path.join(d, "runs", "r4.json"), "w") as f:
            json.dump({"pid": 4, "start": 1111}, f)
        with p1, p2, mock.patch.object(core, "config_dir", lambda: d):
            self.assertEqual(core._live_records("runs"), [])

    def test_live_records_use_it(self):
        api, p1, p2 = self.win({4100: ("live", 1111), 4200: ("dead", 2222)})
        import json, tempfile
        d = tempfile.mkdtemp(prefix="pxa-winlive-")
        os.makedirs(os.path.join(d, "runs"))
        for pid, st in ((4100, 1111), (4200, 2222), (4101, None)):
            with open(os.path.join(d, "runs", f"r{pid}.json"), "w") as f:
                json.dump({"pid": pid, "start": st}, f)
        with p1, p2, mock.patch.object(core, "config_dir", lambda: d):
            got = [r["pid"] for _, r in core._live_records("runs")]
        self.assertEqual(got, [4100])

    def test_any_error_reading_a_record_is_stale_and_moved_aside(self):
        """A run record left by taskkill /F made the liveness check raise SystemError on the beta 4 exe; the read
        caught only OSError/ValueError, so Control would not start. Any error on one record now makes it stale: it is
        moved aside to run/stale/ and the other records still count."""
        import json, tempfile
        d = tempfile.mkdtemp(prefix="pxa-winlive-")
        run = os.path.join(d, "run")
        os.makedirs(run)
        for pid in (4100, 4200):
            with open(os.path.join(run, f"control-{pid}.json"), "w") as f:
                json.dump({"pid": pid, "start": 1111, "port": 9100 + pid // 100}, f)

        def alive(pid, start=None):
            if int(pid) == 4100:
                raise SystemError("<built-in function kill> returned a result with an exception set")
            return True
        with mock.patch.object(core, "config_dir", lambda: d), mock.patch.object(core, "pid_alive", alive):
            got = [r["pid"] for r in bridge.running_controls()]        # what start-up asks (desktop: find_running)
        self.assertEqual(got, [4200])
        self.assertFalse(os.path.exists(os.path.join(run, "control-4100.json")))
        self.assertTrue(os.path.isfile(os.path.join(run, "stale", "control-4100.json")))     # kept for a look

    def test_a_record_that_cannot_be_moved_aside_is_skipped(self):
        """the move aside failing as well (a locked file, a read-only folder) still never stops start-up"""
        import json, tempfile
        d = tempfile.mkdtemp(prefix="pxa-winlive-")
        os.makedirs(os.path.join(d, "run"))
        with open(os.path.join(d, "run", "control-4100.json"), "w") as f:
            json.dump({"pid": 4100, "start": 1111, "port": 9141}, f)
        denied = PermissionError(13, "denied")
        with mock.patch.object(core, "config_dir", lambda: d), \
                mock.patch.object(core, "pid_alive", mock.Mock(side_effect=SystemError("boom"))), \
                mock.patch.object(core.os, "replace", side_effect=denied), mock.patch.object(core.os, "unlink", side_effect=denied):
            self.assertEqual(bridge.running_controls(), [])
            self.assertEqual(core._live_records("run", "control-"), [])

    def test_posix_path_untouched(self):
        self.assertFalse(core._WINDOWS)
        self.assertTrue(core.pid_alive(os.getpid(), core.proc_start_ticks(os.getpid())))
        self.assertFalse(core.pid_alive(os.getpid(), -1))


if __name__ == "__main__":
    unittest.main()
