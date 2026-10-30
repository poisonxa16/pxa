#!/usr/bin/env python3
"""P2: the launcher's and the Encode tab's own pid checks (both imported by Control) on Windows, simulated.
os.kill(pid, 0) there sends CTRL_C_EVENT (signal 0 = Ctrl+C), and the Encode backend also used os.killpg / os.getpgid,
which Windows lacks. Both now use Control's kernel32 check; POSIX behaviour is unchanged."""
import importlib.util
import os
import signal
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pxa_control.core as core  # noqa: E402
import pxa_encode as E  # noqa: E402


def load_launcher():
    spec = importlib.util.spec_from_file_location("pxa_launch_p2", os.path.join(ROOT, "tools", "pxa-launch.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L


class FakeApi:
    """pid -> "live" | "dead" | "denied" | "gone" (core._WinApi's four calls)."""
    def __init__(self, procs):
        self.procs, self.closed = procs, []

    def open(self, pid):
        kind = self.procs.get(pid, "gone")
        if kind == "gone":
            return None, 87
        if kind == "denied":
            return None, 5
        return 1000 + pid, 0

    def running(self, h):
        return self.procs[h - 1000] == "live"

    def created(self, h):
        return 1111

    def close(self, h):
        self.closed.append(h)


def no_kill(*a, **k):
    raise AssertionError("os.kill / os.killpg must not be called on Windows for a liveness probe")


class WinPidAlive(unittest.TestCase):
    def check(self, mod):
        api = FakeApi({4100: "live", 4200: "dead", 4: "denied"})
        with mock.patch.multiple(core, _WINDOWS=True, _win_api=lambda: api), \
                mock.patch.object(mod, "_WINDOWS", True), \
                mock.patch.object(os, "kill", no_kill):
            self.assertTrue(mod._pid_alive(4100))
            self.assertFalse(mod._pid_alive(4200))
            self.assertTrue(mod._pid_alive(4))                  # access denied, nothing recorded: it exists
            self.assertFalse(mod._pid_alive(4300))
            self.assertFalse(mod._pid_alive(0))
            self.assertFalse(mod._pid_alive(None))

    def test_encode_pid_alive_on_windows(self):
        self.check(E)

    def test_launcher_pid_alive_on_windows(self):
        self.check(load_launcher())

    def test_posix_unchanged(self):
        L = load_launcher()
        for mod in (E, L):
            self.assertFalse(mod._WINDOWS)
            self.assertTrue(mod._pid_alive(os.getpid()))
            p = subprocess.Popen(["true"])
            p.wait()
            self.assertFalse(mod._pid_alive(p.pid))             # reaped: gone
        z = subprocess.Popen(["true"])                          # a zombie (exited, not yet reaped) is not alive for the launcher
        import time
        for _ in range(100):
            with open("/proc/%d/stat" % z.pid) as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    break
            time.sleep(0.02)
        self.assertFalse(L._pid_alive(z.pid))
        z.wait()


class NoProcessGroups(unittest.TestCase):
    """Windows has no os.killpg / os.getpgid (and no SIGKILL / SIGSTOP / SIGCONT): the Encode backend must neither raise
    AttributeError nor leave the process alone when it is asked to stop it."""
    def without(self, *names):
        saved = {n: getattr(os, n) for n in names if hasattr(os, n)}
        for n in saved:
            delattr(os, n)
        return saved

    def restore(self, saved):
        for n, v in saved.items():
            setattr(os, n, v)

    def test_signal_falls_back_to_the_process(self):
        sent = []
        p = types.SimpleNamespace(pid=555, send_signal=lambda s: sent.append(s))
        svc = E.EncodeService.__new__(E.EncodeService)
        saved = self.without("killpg", "getpgid")
        try:
            svc._signal(p, signal.SIGTERM)
        finally:
            self.restore(saved)
        self.assertEqual(sent, [signal.SIGTERM])

    def test_kill_orphan_on_windows_terminates_the_process(self):
        svc = E.EncodeService.__new__(E.EncodeService)
        job = {"pid": 4100}
        alive = {"v": True}
        killed = []

        def kill(pid, sig):
            killed.append((pid, sig))
            alive["v"] = False
        saved = self.without("killpg", "getpgid")
        try:
            with mock.patch.object(E, "_pid_alive", lambda pid: alive["v"]), mock.patch.object(os, "kill", kill):
                svc._kill_orphan(job)
        finally:
            self.restore(saved)
        self.assertEqual(killed, [(4100, signal.SIGTERM)])
        self.assertIsNone(job["pid"])

    def test_group_signal_without_groups_raises_oserror_and_skips_missing_signals(self):
        saved = self.without("killpg", "getpgid")
        try:
            with self.assertRaises(OSError):
                E._group_signal(555, signal.SIGTERM)
        finally:
            self.restore(saved)
        self.assertIsNone(E._group_signal(555, None))           # a signal the platform lacks is skipped

if __name__ == "__main__":
    unittest.main()
