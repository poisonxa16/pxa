#!/usr/bin/env python3
"""PXA_JIT_FIRST_START_v1: every server the launcher or PXA Control starts gets CUDA_CACHE_MAXSIZE=4 GiB.

A GPU newer than every SASS arch in a release build (H100, B200, RTX 50) runs kernels the driver
JIT-compiles from PTX on first start, then caches them. The driver's default cache can be smaller
than our compiled kernels, so without a bigger cache every start would compile again. The
launcher's device_env() (the one place every child's CUDA_* env comes from: Control's seat start,
the TUI, the exec path, the generated scripts) adds the variable, unless the user exported it.

Checks the launcher function and PXA Control's real Seat.start() with a fake Popen (no server,
no GPU). Exit 0 pass, 1 failure.
"""
import importlib.util
import io
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
FAILS = []


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def load_launcher():
    sys.path.insert(0, TOOLS)
    spec = importlib.util.spec_from_file_location("pxa_launch_t", os.path.join(TOOLS, "pxa-launch.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeProc:
    def __init__(self, cmd, env=None, **_kw):
        FakeProc.last = (list(cmd), dict(env or {}))
        self.stdout = io.StringIO("")
        self.pid = 4242

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def control_start_env(L, user_env):
    """The env PXA Control's Seat.start() hands the server, with os.environ = user_env."""
    from pxa_control import core
    saved = dict(os.environ)
    real_popen = core.subprocess.Popen
    try:
        os.environ.clear()
        os.environ.update(user_env)
        core.subprocess.Popen = FakeProc
        seat = core.Seat(L)
        built = (None, ["/opt/pxa/bin/llama-server", "--port", "18999"], {"PXA_X_TEST": "1"}, "4", None, 4096)
        seat.start(built, {"port": 18999}, {})
        return FakeProc.last[1], seat.env_shown
    finally:
        core.subprocess.Popen = real_popen
        os.environ.clear()
        os.environ.update(saved)


def main():
    L = load_launcher()
    check(L.JIT_CACHE_MAXSIZE == "4294967296", "the cache size is 4 GiB (the driver maximum)")

    check(L.jit_cache_env({}) == {"CUDA_CACHE_MAXSIZE": "4294967296"}, "jit_cache_env: unset -> 4 GiB")
    check(L.jit_cache_env({"CUDA_CACHE_MAXSIZE": "1073741824"}) == {}, "jit_cache_env: the user's value is kept")
    check(L.jit_cache_env({"CUDA_CACHE_MAXSIZE": "  "}) == {"CUDA_CACHE_MAXSIZE": "4294967296"},
          "jit_cache_env: a blank value counts as unset")

    saved = os.environ.pop("CUDA_CACHE_MAXSIZE", None)
    try:
        d, _ = L.device_env("2,4")
        check(d.get("CUDA_CACHE_MAXSIZE") == "4294967296", "device_env (index): carries the cache size")
        d, _ = L.device_env("GPU-aad5ef40-9b80-8fd0-4391-dfe595f42640")
        check(d.get("CUDA_CACHE_MAXSIZE") == "4294967296", "device_env (UUID): carries the cache size")
        check(d.get("CUDA_VISIBLE_DEVICES") == "GPU-aad5ef40-9b80-8fd0-4391-dfe595f42640",
              "device_env: the card selection is unchanged")
        os.environ["CUDA_CACHE_MAXSIZE"] = "536870912"
        d, _ = L.device_env("2,4")
        check("CUDA_CACHE_MAXSIZE" not in d, "device_env: a user-exported CUDA_CACHE_MAXSIZE is not overridden")
    finally:
        os.environ.pop("CUDA_CACHE_MAXSIZE", None)
        if saved is not None:
            os.environ["CUDA_CACHE_MAXSIZE"] = saved

    base = {"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": os.environ.get("HOME", "/root")}
    env, shown = control_start_env(L, base)
    check(env.get("CUDA_CACHE_MAXSIZE") == "4294967296", "Control Seat.start: the server child gets 4 GiB")
    check(shown.get("CUDA_CACHE_MAXSIZE") == "4294967296", "Control Seat.start: the env panel shows it")
    check(env.get("CUDA_VISIBLE_DEVICES") == "4" and env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
          "Control Seat.start: the card selection is unchanged")
    check(env.get("PXA_X_TEST") == "1", "Control Seat.start: the launcher env still reaches the child")

    env, _ = control_start_env(L, dict(base, CUDA_CACHE_MAXSIZE="1073741824"))
    check(env.get("CUDA_CACHE_MAXSIZE") == "1073741824", "Control Seat.start: the user's own value is kept")

    print(f"test-control-jit-cache-env: {'all checks passed' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
