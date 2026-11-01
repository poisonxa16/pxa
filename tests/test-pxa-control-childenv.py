#!/usr/bin/env python3
"""What every program PXA Control starts inherits (tools/pxa_control/childenv.py, 4.0.1).

Windows (the rule is run with windows=True, so it is checked on every OS): every Popen gets CREATE_NO_WINDOW unless the
caller asked for a console itself; PXA_CHILD_CONSOLE=1 is the off switch. Every spawn site in Control's code is
checked to go through subprocess.Popen (no os.system, os.spawn*, CreateProcess outside the plugin sandbox launchers,
which set CREATE_NO_WINDOW themselves), and the hook is installed when the package loads.

Linux: a system program (xdg-open, nvidia-smi, the shell) started with Control's environment gets the package's
library folders taken off LD_LIBRARY_PATH; an engine program and an environment the caller built are left alone.
The packager's pxa-launch wrapper no longer exports lib/ for Python, and run-server.sh keeps it for the engine only.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
MK = os.path.join(ROOT, "scripts", "make-release-tarball.sh")

from pxa_control import childenv as CE  # noqa: E402

NO_WINDOW = 0x08000000


def bound(*a, **kw):
    return CE._SIG.bind(None, *a, **kw).arguments


class WindowsFlags(unittest.TestCase):
    def test_plain_call_gets_no_window(self):
        args = CE.adjust(bound(["llama-server.exe", "--version"], stdout=subprocess.PIPE), windows=True, environ={})
        self.assertEqual(args["creationflags"], NO_WINDOW)

    def test_existing_flags_kept(self):
        args = CE.adjust(bound(["x"], creationflags=0x200), windows=True, environ={})
        self.assertEqual(args["creationflags"], 0x200 | NO_WINDOW)

    def test_positional_creationflags(self):
        # Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env,
        #       universal_newlines, startupinfo, creationflags)
        a = bound(["x"], -1, None, None, None, None, None, True, False, None, None, None, None, 0x4)
        self.assertEqual(CE.adjust(a, windows=True, environ={})["creationflags"], 0x4 | NO_WINDOW)

    def test_console_on_purpose_is_respected(self):
        for f in (0x10, 0x8):        # CREATE_NEW_CONSOLE, DETACHED_PROCESS
            self.assertEqual(CE.adjust(bound(["x"], creationflags=f), windows=True, environ={})["creationflags"], f)

    def test_off_switch(self):
        args = CE.adjust(bound(["x"]), windows=True, environ={"PXA_CHILD_CONSOLE": "1"})
        self.assertNotIn("creationflags", args)

    def test_windows_env_untouched(self):
        env = {"LD_LIBRARY_PATH": "/pkg/lib", "PXA_ENGINE_DIR": "/pkg"}
        self.assertNotIn("env", CE.adjust(bound(["x"]), windows=True, environ=env))


class HookInstalled(unittest.TestCase):
    def test_package_load_installs_hook(self):
        import pxa_control  # noqa: F401
        self.assertIs(subprocess.Popen.__init__, CE._popen_init)
        self.assertFalse(CE.install(), "install() must be idempotent")

    def test_hook_reaches_the_real_popen_on_windows_rule(self):
        """subprocess.run -> Popen.__init__ -> the rule -> the original __init__ with the changed flags."""
        seen = {}
        orig = CE._ORIG_INIT

        def fake(self, *a, **kw):
            seen.update(CE._SIG.bind(self, *a, **kw).arguments)
            raise RuntimeError("stop")
        real_adjust = CE.adjust
        try:
            CE._ORIG_INIT = fake
            CE.adjust = lambda arguments: real_adjust(arguments, windows=True, environ={})
            with self.assertRaises(RuntimeError):
                subprocess.run(["llama-server", "--version"], capture_output=True, text=True, timeout=5)
        finally:
            CE._ORIG_INIT = orig
            CE.adjust = real_adjust
        self.assertEqual(seen["creationflags"], NO_WINDOW)
        self.assertEqual(seen["args"], ["llama-server", "--version"])

    def test_real_child_still_runs(self):
        r = subprocess.run([sys.executable, "-c", "print(41+1)"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.stdout.strip(), "42")


SPAWN_RX = re.compile(r"\bos\.(system|spawn\w*|exec\w+|startfile)\s*\(|_winapi\.CreateProcess|CreateProcessW?\(")
# Files that create processes by hand on purpose: the plugin sandboxes (AppContainer / restricted token launchers, each
# passes CREATE_NO_WINDOW in its own flags, checked below), the Linux sandbox's exec inside its own child (Linux only),
# and the launcher's final exec into the engine (a CLI path).
OWN_LAUNCHERS = {"pxa_plugins/winlaunch.py", "pxa_plugins/sandbox_win.py", "pxa_plugins/sandbox.py", "pxa-launch.py"}


class EverySpawnSite(unittest.TestCase):
    def control_files(self):
        roots = ["pxa_control", "pxa_chat", "pxa_ctl", "pxa_plugins"]
        files = ["pxa-launch.py", "pxa-bench.py", "pxa_telemetry.py", "pxa_engine_update.py", "pxa_lib_update.py",
                 "pxa_hf.py", "pxa_encode.py", "pxa_encode_adapter.py", "pxa_encode_pkg.py", "pxa_mqtt.py",
                 "pxa_expert_map.py", "pxa_thinking.py"]
        for r in roots:
            for d, _, fs in os.walk(os.path.join(TOOLS, r)):
                if os.sep + "tests" in d:
                    continue
                files += [os.path.relpath(os.path.join(d, f), TOOLS) for f in fs if f.endswith(".py")]
        return [f for f in files if os.path.isfile(os.path.join(TOOLS, f))]

    def test_no_spawn_outside_popen(self):
        bad = []
        for f in self.control_files():
            if f.replace(os.sep, "/") in OWN_LAUNCHERS or os.path.basename(f).startswith("test"):
                continue
            with open(os.path.join(TOOLS, f), encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            for n, ln in enumerate(src.splitlines(), 1):
                if ln.lstrip().startswith("#"):
                    continue
                if SPAWN_RX.search(ln):
                    bad.append(f"{f}:{n}: {ln.strip()[:100]}")
        self.assertEqual(bad, [], "a child started outside subprocess.Popen would miss the no-window rule")

    def test_own_launchers_hide_their_window(self):
        for f in ("pxa_plugins/winlaunch.py", "pxa_plugins/sandbox_win.py"):
            with open(os.path.join(TOOLS, f), encoding="utf-8") as fh:
                src = fh.read()
            self.assertIn("CREATE_NO_WINDOW", src, f)
        with open(os.path.join(TOOLS, "pxa-launch.py"), encoding="utf-8") as fh:
            src = fh.read()
        execs = [ln for ln in src.splitlines() if re.search(r"\bos\.exec\w+\(", ln) and not ln.lstrip().startswith("#")]
        self.assertEqual(len(execs), 1, execs)     # the one exec into the engine at the end of a CLI launch


class LinuxEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.E = os.path.join(self.tmp.name, "pxa-v4.0.1")
        for d in ("lib", "lib-compat", "bin"):
            os.makedirs(os.path.join(self.E, d))
        self.exe = os.path.join(self.E, "bin", "llama-server")
        open(self.exe, "w").close()
        self.env = {"PXA_ENGINE_DIR": self.E, "PATH": "/usr/bin:/bin",
                    "LD_LIBRARY_PATH": f"{self.E}/lib-compat:{self.E}/lib:/opt/user/lib"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_system_program_loses_package_libs(self):
        for argv in (["xdg-open", "http://127.0.0.1:7777/"], ["nvidia-smi"], "hostname -I"):
            a = CE.adjust(bound(argv, shell=isinstance(argv, str)), windows=False, environ=self.env)
            self.assertEqual(a["env"]["LD_LIBRARY_PATH"], "/opt/user/lib", argv)
            self.assertEqual(a["env"]["PXA_ENGINE_DIR"], self.E)

    def test_only_package_libs_unset(self):
        env = dict(self.env, LD_LIBRARY_PATH=f"{self.E}/lib")
        a = CE.adjust(bound(["xdg-open", "u"]), windows=False, environ=env)
        self.assertNotIn("LD_LIBRARY_PATH", a["env"])

    def test_engine_program_keeps_them(self):
        a = CE.adjust(bound([self.exe, "--version"]), windows=False, environ=self.env)
        self.assertNotIn("env", a)

    def test_explicit_env_untouched(self):
        mine = {"LD_LIBRARY_PATH": f"{self.E}/lib"}
        a = CE.adjust(bound([self.exe], env=mine), windows=False, environ=self.env)
        self.assertIs(a["env"], mine)

    def test_clean_environment_untouched(self):
        env = dict(self.env, LD_LIBRARY_PATH="/opt/user/lib")
        self.assertNotIn("env", CE.adjust(bound(["xdg-open", "u"]), windows=False, environ=env))
        env.pop("LD_LIBRARY_PATH")
        self.assertNotIn("env", CE.adjust(bound(["xdg-open", "u"]), windows=False, environ=env))

    def test_real_child_sees_clean_path(self):
        old = dict(os.environ)
        try:
            os.environ.update(self.env)
            os.environ["PATH"] = old.get("PATH", "/usr/bin:/bin")
            out = subprocess.run(["sh", "-c", 'printf %s "${LD_LIBRARY_PATH-UNSET}"'], capture_output=True,
                                 text=True, timeout=30).stdout
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(out, "/opt/user/lib")


class ConsoleParent(unittest.TestCase):
    """review M1: a CONSOLE program that imports pxa_control (pxa-launch's CLI, pxa_encode, the bench handoff) keeps its
    children on its own console: CREATE_NO_WINDOW only when the parent has no console (the windowed Control)."""

    class K32(object):
        def __init__(self, hwnd):
            self.hwnd = hwnd

        def GetConsoleWindow(self):
            return self.hwnd

    def with_k32(self, k):
        orig = CE._kernel32
        CE._kernel32 = lambda: k
        self.addCleanup(setattr, CE, "_kernel32", orig)

    def test_console_parent_children_keep_console(self):
        a = CE.adjust(bound(["python.exe", "pxa-bench.py"]), windows=True, environ={}, has_console=True)
        self.assertNotIn("creationflags", a)

    def test_windowed_parent_children_hidden(self):
        a = CE.adjust(bound(["llama-server.exe", "--version"]), windows=True, environ={}, has_console=False)
        self.assertEqual(a["creationflags"], NO_WINDOW)

    def test_get_console_window_decides(self):
        self.with_k32(self.K32(0x1234))
        self.assertTrue(CE.parent_has_console())
        self.assertNotIn("creationflags", CE.adjust(bound(["x"]), windows=True, environ={}))
        self.with_k32(self.K32(0))
        self.assertFalse(CE.parent_has_console())
        self.assertEqual(CE.adjust(bound(["x"]), windows=True, environ={})["creationflags"], NO_WINDOW)

    def test_no_kernel32_means_no_console(self):
        self.with_k32(None)
        self.assertFalse(CE.parent_has_console())

    def test_console_parent_keeps_callers_flags(self):
        a = CE.adjust(bound(["x"], creationflags=0x200), windows=True, environ={}, has_console=True)
        self.assertEqual(a["creationflags"], 0x200)


class DesktopLaunch(unittest.TestCase):
    """review M2: desktop.launch starts a program for the user: a console program gets its own console window."""

    def test_launch_asks_for_a_console(self):
        from unittest import mock
        from pxa_plugins.skills import desktop as DSK
        b = object.__new__(DSK.WinBackend)
        with mock.patch("subprocess.Popen") as P:
            b.launch("C:\\Windows\\System32\\cmd.exe")
        kw = P.call_args.kwargs
        self.assertEqual(P.call_args.args[0], ["C:\\Windows\\System32\\cmd.exe"])
        self.assertTrue(kw.get("creationflags", 0) & 0x10, kw)
        self.assertEqual(CE.windows_flags(kw["creationflags"]), kw["creationflags"], "the hook must leave it as asked")


# subprocess calls whose `exe` is not a package program: a system program found on PATH (shutil.which), or the program
# the user asked desktop.launch to start
SYSTEM_EXE = {("pxa_control/plugins/egress.py", "[exe"), ("pxa_plugins/sandbox.py", "[exe"),
              ("pxa_plugins/skills/desktop.py", "[exe")}


class EngineEnv(unittest.TestCase):
    """review M3: every run of the package's own engine programs gets engine_ld_path (lib-compat on an old CPU, lib/ on
    a package without RUNPATH), not Control's environment."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.E = os.path.join(self.tmp.name, "pkg")
        for d in ("bin", "lib", "lib-compat"):
            os.makedirs(os.path.join(self.E, d))

    def test_engine_build_string_uses_engine_ld_path(self):
        from pxa_control import core
        exe = os.path.join(self.E, "bin", "llama-server")
        with open(exe, "w") as f:
            f.write('#!/bin/sh\necho "version: ${LD_LIBRARY_PATH-UNSET}"\n')
        os.chmod(exe, 0o755)
        seen = []

        class L(object):
            @staticmethod
            def engine_ld_path(E):
                seen.append(E)
                return "%s/lib-compat:%s/lib" % (E, E), []

        class Fake(object):
            _engine_env = core.App._engine_env
        fake = Fake()
        fake.L, fake._build_cache = L, {}
        got = core.App.engine_build_string(fake, exe)
        E = os.path.realpath(self.E)
        self.assertEqual(seen, [E])
        self.assertEqual(got, "%s/lib-compat:%s/lib" % (E, E))

    def test_no_engine_run_without_env(self):
        """no subprocess call in Control's code runs a package program (an exe under bin/, llama-*) with env=None."""
        import ast
        bad = []
        for f in EverySpawnSite().control_files():
            path = os.path.join(TOOLS, f)
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            for n in ast.walk(ast.parse(src)):
                if not isinstance(n, ast.Call) or not n.args:
                    continue
                fn = ast.unparse(n.func)
                if not (fn.startswith("subprocess.") and fn.split(".")[-1] in ("run", "Popen", "check_output", "call")):
                    continue
                if any(k.arg == "env" for k in n.keywords):
                    continue
                a0 = ast.unparse(n.args[0])
                if (f.replace(os.sep, "/"), a0.split(",")[0].split("]")[0]) in SYSTEM_EXE:
                    continue
                if re.search(r"llama-|ENGINE_EXE|\bexe\b|\bE\b.*bin", a0):
                    bad.append("%s:%d: %s" % (f, n.lineno, a0[:80]))
        self.assertEqual(bad, [])

    def expert_map_tries(self, environ):
        import ctypes
        from unittest import mock
        import pxa_expert_map as XM
        tried = []

        def cdll(p, *a, **kw):
            tried.append(p)
            raise OSError("stub")
        keys = ("PXA_ENGINE_DIR", "PXA_PXQN_LIB", "PXA_CPU_LIB_PICKED", "PXA_CPU_LIB", "PXA_CPUINFO")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        clean.update(environ)
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(ctypes, "CDLL", cdll):
            XM._PXQN, XM._PXQN_TRIED = None, False
            try:
                self.assertIsNone(XM._pxqn())
            finally:
                XM._PXQN, XM._PXQN_TRIED = None, False
        return tried

    def cpuinfo(self, flags):
        p = os.path.join(self.tmp.name, "cpuinfo")
        with open(p, "w") as f:
            f.write("processor\t: 0\nflags\t\t: %s\n" % flags)
        return p

    def test_expert_map_old_cpu_tries_lib_compat(self):
        old = self.cpuinfo("fpu sse sse2 avx")
        tried = self.expert_map_tries({"PXA_ENGINE_DIR": self.E, "PXA_CPUINFO": old})
        self.assertEqual(tried[1:], [os.path.join(self.E, "lib-compat", "libggml-pxqn.so"),
                                     os.path.join(self.E, "lib", "libggml-pxqn.so")])

    def test_expert_map_fast_cpu_skips_lib_compat(self):
        new = self.cpuinfo("fpu sse sse2 avx avx2 fma f16c")
        tried = self.expert_map_tries({"PXA_ENGINE_DIR": self.E, "PXA_CPUINFO": new})
        self.assertEqual(tried[1:], [os.path.join(self.E, "lib", "libggml-pxqn.so")])
        forced = self.expert_map_tries({"PXA_ENGINE_DIR": self.E, "PXA_CPUINFO": new, "PXA_CPU_LIB": "compat"})
        self.assertIn(os.path.join(self.E, "lib-compat", "libggml-pxqn.so"), forced)
        picked = self.expert_map_tries({"PXA_ENGINE_DIR": self.E, "PXA_CPUINFO": new, "PXA_CPU_LIB_PICKED": "compat"})
        self.assertIn(os.path.join(self.E, "lib-compat", "libggml-pxqn.so"), picked)

    def test_expert_map_isa_list_matches_launcher(self):
        import pxa_expert_map as XM
        with open(os.path.join(TOOLS, "pxa-launch.py"), encoding="utf-8") as fh:
            m = re.search(r"^CPU_FAST_NEEDS = (\(.*?\))", fh.read(), re.M)
        self.assertEqual(tuple(XM.CPU_FAST_NEEDS), eval(m.group(1)))

    def test_packager_fails_without_patchelf(self):
        with open(MK, encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r"if dimg -- 'command -v patchelf'.*?\nelse\n(.*?)\nfi\n", src, re.S)
        self.assertTrue(m)
        self.assertTrue(re.search(r"^\s*die ", m.group(1), re.M), m.group(1))


def wrapper(target):
    with open(MK, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'cat > "\$STAGE/%s" <<\'WRAP\'\n(.*?)\nWRAP\n' % re.escape(target), src, re.S)
    assert m, target
    return m.group(1) + "\n"


@unittest.skipUnless(sys.platform.startswith("linux"), "the release wrappers are Linux shell scripts")
class Wrappers(unittest.TestCase):
    """the packager's pxa-launch and run-server.sh, run for real on a stub package."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.P = P = os.path.join(self.tmp.name, "pkg")
        for d in ("lib", "lib-compat", "bin", "tools"):
            os.makedirs(os.path.join(P, d))
        self.out = os.path.join(self.tmp.name, "out")
        stub = ('import os,sys\nopen(os.environ["OUT"] + "." + ("control" if "--control-for-pid" in sys.argv else "launch"), "w")'
                '.write(os.environ.get("LD_LIBRARY_PATH", "UNSET"))\n')
        self.write(os.path.join(P, "tools", "pxa-launch.py"), stub)
        self.write(os.path.join(P, "bin", "llama-server"), '#!/bin/sh\nprintf %s "${LD_LIBRARY_PATH-UNSET}" > "$OUT.engine"\n')
        os.chmod(os.path.join(P, "bin", "llama-server"), 0o755)
        for t in ("pxa-launch", "run-server.sh"):
            self.write(os.path.join(P, t), wrapper(t))
            os.chmod(os.path.join(P, t), 0o755)

    @staticmethod
    def write(path, text):
        with open(path, "w") as f:
            f.write(text)

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, argv, ldp, cpu="fast"):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OUT": self.out, "PXA_CPU_LIB": cpu,
               "PXA_CONTROL": "1", "HOME": self.tmp.name}
        if ldp is not None:
            env["LD_LIBRARY_PATH"] = ldp
        subprocess.run(argv, env=env, timeout=60, check=True, stdin=subprocess.DEVNULL)

    def read(self, kind):
        with open(self.out + "." + kind) as f:
            return f.read()

    def compat_pkg(self):
        import shutil
        shutil.copy(os.path.join(TOOLS, "pxa-cpu-check.sh"), os.path.join(self.P, "tools", "pxa-cpu-check.sh"))
        open(os.path.join(self.P, "lib-compat", "libggml.so"), "w").close()

    def test_compat_pxa_launch_keeps_user_path(self):
        """review L3: PXA_CPU_LIB=compat: the check runs (and picks lib-compat), Control still gets the user's value."""
        self.compat_pkg()
        for ldp, want in ((None, "UNSET"), ("/opt/user/lib", "/opt/user/lib")):
            self.run_it([os.path.join(self.P, "pxa-launch")], ldp, cpu="compat")
            self.assertEqual(self.read("launch"), want, ldp)

    def test_compat_run_server_engine_gets_lib_compat_first(self):
        self.compat_pkg()
        self.run_it([os.path.join(self.P, "run-server.sh"), "-m", "x.gguf"], "/opt/user/lib", cpu="compat")
        self.assertEqual(self.read("engine"), f"{self.P}/lib-compat:{self.P}/lib:/opt/user/lib")
        self.assertEqual(self.read("control"), "/opt/user/lib")
        self.run_it([os.path.join(self.P, "run-server.sh"), "-m", "x.gguf"], None, cpu="compat")
        self.assertEqual(self.read("engine"), f"{self.P}/lib-compat:{self.P}/lib")
        self.assertEqual(self.read("control"), "UNSET")

    def test_pxa_launch_keeps_user_path(self):
        for ldp, want in ((None, "UNSET"), ("/opt/user/lib", "/opt/user/lib"), ("", "")):
            self.run_it([os.path.join(self.P, "pxa-launch")], ldp)
            self.assertEqual(self.read("launch"), want, ldp)

    def test_run_server_engine_gets_lib_control_does_not(self):
        self.run_it([os.path.join(self.P, "run-server.sh"), "-m", "x.gguf"], "/opt/user/lib")
        self.assertEqual(self.read("engine"), f"{self.P}/lib:/opt/user/lib")
        self.assertEqual(self.read("control"), "/opt/user/lib")
        self.run_it([os.path.join(self.P, "run-server.sh"), "-m", "x.gguf"], None)
        self.assertEqual(self.read("engine"), f"{self.P}/lib")
        self.assertEqual(self.read("control"), "UNSET")


if __name__ == "__main__":
    unittest.main(verbosity=2)
