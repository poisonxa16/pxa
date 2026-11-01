"""Shared guard for tests that spawn or signal (F295): true only inside the fresh PID namespace that
tests/run-in-pidns.sh makes. Tests skip otherwise, so they never run on a real host's PID namespace."""
import os
import sys


def in_fresh_pidns():
    if os.environ.get("PXA_FRESH_PIDNS") != "1" or not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/1/comm") as f:
            init = f.read().strip()
    except OSError:
        return False
    return os.getpid() < 50 and init not in ("init", "systemd", "launchd") and len(
        [d for d in os.listdir("/proc") if d.isdigit()]) < 50
