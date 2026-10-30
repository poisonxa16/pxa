#!/usr/bin/env python3
"""LaunchTUI._stop: a seat with no command (cmd None) must still stop its process, and a failure must be
reported, not swallowed. The old code did cmd[0] on None, raised TypeError, and `except Exception: pass`
left the server running.

Cases (real child processes, no GPU, no model):
  none_cmd       Popen(sleep) + cmd None            -> terminated, returns None
  pid_only       bare int pid of a sleep child      -> terminated (zombie-aware), returns None
  ignores_term   child that ignores SIGTERM         -> SIGKILL path, terminated, returns None
  docker_fail    cmd = docker run --name X, fake docker exits 1 -> process still stopped AND the
                 failure line names the docker stop; returns a non-None string
  signal_fail    stub proc whose send_signal raises -> returns "could not stop pid ..." (logged path)
  already_gone   exited Popen + cmd None            -> returns None

usage: test-pxa-launch-stop.py [--launcher tools/pxa-launch.py]    exit 0 = PASS
"""
import argparse, importlib.util, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    spec = importlib.util.spec_from_file_location("pxa_launch_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sleeper():
    return subprocess.Popen(["sleep", "300"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class _SignalFail(object):
    pid = 424242

    def poll(self):
        return None

    def send_signal(self, sig):
        raise PermissionError("denied (test stub)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", default=os.path.join(HERE, "..", "tools", "pxa-launch.py"))
    a = ap.parse_args()
    L = load(os.path.abspath(a.launcher)).LaunchTUI
    fails = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))
        if not cond:
            fails.append(name)

    # none_cmd: a real sleeping child, seat has no cmd
    p = sleeper()
    t0 = time.time()
    err = L._stop(p, None)
    gone = p.poll() is not None
    if not gone:
        p.kill(); p.wait()                       # clean up a failed run; the verdict is `gone`
    check("none_cmd: process terminated by _stop", gone, f"rc={p.returncode}")
    check("none_cmd: returns None on success", err is None, repr(err))
    check("none_cmd: no 10 s wait", time.time() - t0 < 5, f"{time.time() - t0:.1f}s")

    # pid_only: stop by bare pid (the child is not reaped yet, so it is a zombie once killed)
    p = sleeper()
    err = L._stop(p.pid, None)
    try:
        p.wait(timeout=10)
        gone = True
    except subprocess.TimeoutExpired:
        gone = False
        p.kill(); p.wait()                       # clean up a failed run; the verdict is `gone`
    check("pid_only: pid terminated by _stop", gone, f"rc={p.returncode}")
    check("pid_only: returns None", err is None, repr(err))

    # ignores_term: SIGTERM is ignored, so the SIGKILL step must end it
    ig = subprocess.Popen([sys.executable, "-c",
                           "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(0.5)  # let it install the handler before we signal it
    err = L._stop(ig, None)
    rc = ig.poll()
    if rc is None:
        ig.kill(); ig.wait()
    check("ignores_term: SIGKILL ends it", rc == -9, f"rc={rc}")
    check("ignores_term: returns None", err is None, repr(err))

    # docker_fail: fake docker that exits 1; the process must still be stopped and the failure reported
    tmp = tempfile.mkdtemp(prefix="pxa-stop-test-")
    fake = os.path.join(tmp, "docker")
    with open(fake, "w") as f:
        f.write("#!/bin/sh\necho 'no such container (test)' >&2\nexit 1\n")
    os.chmod(fake, 0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = tmp + os.pathsep + old_path
    try:
        p = sleeper()
        err = L._stop(p, ["docker", "run", "--name", "pxa-test-seat", "img"])
    finally:
        os.environ["PATH"] = old_path
    gone = p.poll() is not None
    if not gone:
        p.kill(); p.wait()
    check("docker_fail: process still terminated", gone, f"rc={p.returncode}")
    check("docker_fail: failure is reported", err is not None and "docker stop pxa-test-seat" in err, repr(err))

    # signal_fail: the failure path is returned, not swallowed
    err = L._stop(_SignalFail(), None)
    check("signal_fail: failure is returned", err is not None and "could not stop pid 424242" in err, repr(err))

    # already_gone: nothing to stop, no error
    p = sleeper()
    p.kill(); p.wait(timeout=10)
    err = L._stop(p, None)
    check("already_gone: returns None", err is None, repr(err))

    print("PASS" if not fails else f"FAIL ({len(fails)}): {', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
