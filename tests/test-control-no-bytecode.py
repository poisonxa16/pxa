#!/usr/bin/env python3
"""PXA Control writes no bytecode next to its own sources (review N3).

A copy of tools/ is laid out like the release package (tools/ + the pxa-launch wrapper the packager writes), with no
__pycache__ in it. Control is then started the ways a user starts it -- the shipped pxa-launch wrapper, `python3
tools/pxa-launch.py --gui`, `python3 tools/pxa_control.py` -- with PYTHONDONTWRITEBYTECODE removed from the environment
and no -B on our side, waits until it answers on loopback, is stopped, and the tree must hold no .pyc and no
__pycache__. The other launchers (run-server.sh, the docker entry and image, the compiled binary's entry) are checked
for the switch as text. No GPU (fake cards), loopback only.
"""
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

sys.dont_write_bytecode = True       # this test itself must not write into the repo
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
MK = os.path.join(ROOT, "scripts", "make-release-tarball.sh")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def bytecode_in(tree):
    hits = []
    for d, dirs, files in os.walk(tree):
        hits += [os.path.join(d, x) for x in dirs if x == "__pycache__"]
        hits += [os.path.join(d, f) for f in files if f.endswith((".pyc", ".pyo"))]
    return sorted(hits)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def heredoc(script, target):
    """the body of `cat > "$STAGE/<target>" <<'WRAP' ... WRAP` in the packager"""
    src = read(script)
    m = re.search(r'cat > "\$STAGE/%s" <<\'WRAP\'\n(.*?)\nWRAP\n' % re.escape(target), src, re.S)
    return m.group(1) + "\n" if m else None


class ControlWritesNoBytecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pxa-nobytecode-")
        cls.inst = os.path.join(cls.tmp, "pxa-vT")
        ign = shutil.ignore_patterns("__pycache__", "*.pyc", "pxa-private", "mtp-train")
        shutil.copytree(TOOLS, os.path.join(cls.inst, "tools"), ignore=ign)
        if os.path.isdir(os.path.join(ROOT, "common")):
            shutil.copytree(os.path.join(ROOT, "common"), os.path.join(cls.inst, "common"), ignore=ign)
        w = heredoc(MK, "pxa-launch")
        assert w, "the packager's pxa-launch wrapper was not found"
        with open(os.path.join(cls.inst, "pxa-launch"), "w") as f:
            f.write(w)
        os.chmod(os.path.join(cls.inst, "pxa-launch"), 0o755)
        assert bytecode_in(cls.inst) == [], "the copied tree already holds bytecode"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def env(self, home):
        e = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX", "PYTHONPATH")}
        e.update(HOME=home, XDG_CONFIG_HOME=os.path.join(home, ".config"), XDG_DATA_HOME=os.path.join(home, ".local", "share"),
                 XDG_CACHE_HOME=os.path.join(home, ".cache"), PXA_LAUNCH_FAKE_GPUS="2x700", PXA_ENGINE_DIR=self.inst,
                 PXA_CONTROL_NO_TELEMETRY="1", BROWSER="true", DISPLAY="", WAYLAND_DISPLAY="")
        return e

    def start(self, name, argv, port):
        home = os.path.join(self.tmp, "home-" + name)
        os.makedirs(home)
        log = open(os.path.join(self.tmp, name + ".log"), "w")
        p = subprocess.Popen(argv, cwd=home, env=self.env(home), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        up, t0 = False, time.time()
        while time.time() - t0 < 90 and p.poll() is None:
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=2).read(64)
                up = True
                break
            except Exception as e:                       # an HTTP error page still means Control answered
                if hasattr(e, "code"):
                    up = True
                    break
                time.sleep(0.5)
        time.sleep(1.0)                                   # let the start finish its lazy imports
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=20)
        log.close()
        tail = read(os.path.join(self.tmp, name + ".log"))[-1500:]
        self.assertTrue(up, "%s: Control never answered on port %d; log tail:\n%s" % (name, port, tail))
        self.assertEqual(bytecode_in(self.inst), [], "%s wrote bytecode into the install tree" % name)

    def test_1_shipped_wrapper(self):
        port = free_port()
        self.start("wrapper", [os.path.join(self.inst, "pxa-launch"), "--gui", "--no-browser", "--port", str(port)], port)

    def test_2_launcher_without_switch(self):
        port = free_port()
        self.start("launch-py", [sys.executable, os.path.join(self.inst, "tools", "pxa-launch.py"), "--gui", "--no-browser",
                                 "--port", str(port)], port)

    def test_3_control_shim_without_switch(self):
        port = free_port()
        self.start("shim", [sys.executable, os.path.join(self.inst, "tools", "pxa_control.py"), "--no-browser", "--no-tray",
                            "--port", str(port)], port)

    def test_4_every_launcher_carries_the_switch(self):
        w = heredoc(MK, "pxa-launch")
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", w)
        self.assertRegex(w, r'exec python3 -B "\$HERE/tools/pxa-launch.py"')
        rs = heredoc(MK, "run-server.sh")
        self.assertRegex(rs, r'PYTHONDONTWRITEBYTECODE=1 python3 -B "\$HERE/tools/pxa-launch.py" --control-for-pid')
        ep = read(os.path.join(TOOLS, "pxa-entrypoint.sh"))
        self.assertIn("\nexport PYTHONDONTWRITEBYTECODE=1\n", ep)
        self.assertEqual(re.findall(r"\bpython3 (?!-B)\"", ep), [], "a python3 call in the docker entry lacks -B")
        df = read(os.path.join(ROOT, "docker", "Dockerfile"))
        self.assertRegex(df, r"\n    PYTHONDONTWRITEBYTECODE=1 \\\n")
        for f in ("packaging/control/pxa_control_entry.py", "tools/pxa_control.py", "tools/pxa-launch.py", "tools/pxa_control/__init__.py"):
            self.assertRegex(read(os.path.join(ROOT, f)), r"\n_?sys\.dont_write_bytecode = True\n", f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
