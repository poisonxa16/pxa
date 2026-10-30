#!/usr/bin/env python3
"""PXA Control as a package (tools/pxa_control/, the Control hardening pass): the stable API the old single file had, the
compatibility shim tools/pxa_control.py, patch forwarding, the module layering, and the desktop entry the compiled
pxa-control binary runs (engine discovery, first-run models folder, single instance). No GPU, loopback only.

    python3 tests/test-pxa-control-package.py
"""
import ast
import builtins
import contextlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import symtable
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
PKG = os.path.join(TOOLS, "pxa_control")
TMP = tempfile.mkdtemp(prefix="pxa-control-pkg-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_ENGINE_DIR", "PXA_MODELS_DIR", "PXA_BUNDLE_ROOT", "PXA_CONTROL_TOKEN"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)
import pxa_control as C  # noqa: E402
from pxa_control import desktop as D  # noqa: E402

MODULES = ("core", "auth", "bridge", "routes", "server", "desktop")
API = ("App", "Seat", "Live", "make_server", "serve", "ROUTES", "STATIC", "Handler", "Invalid", "Reply", "validate_launch",
       "validate_extra_args", "validate_levers", "load_config", "save_config", "config_dir", "load_or_make_token",
       "ensure_control", "find_running", "control_url", "control_answers", "write_run_record", "running_controls",
       "autostart_decision", "record_launched", "launched_servers", "free_port", "redact_text", "redact_obj",
       "AuthThrottle", "page_csp", "scrub_secrets", "SERVING", "DEFAULT_PORT", "UI_DIR", "CONTROL_VERSION", "headless_hint",
       "open_in_browser", "has_desktop", "start_detached", "content_length", "port_in_use", "median", "shutil_which")


class Package(unittest.TestCase):
    def test_it_is_the_package_with_its_modules(self):
        self.assertTrue(hasattr(C, "__path__"), "import pxa_control must give the package, not the shim")
        for m in MODULES + ("plugins", "store"):
            self.assertTrue(hasattr(C, m), m)
        for name in API:
            self.assertTrue(hasattr(C, name), name)
        self.assertIs(C.App, C.core.App)
        self.assertIs(C.ROUTES, C.routes.ROUTES)
        self.assertIs(C.serve, C.server.serve)
        self.assertEqual(C.HERE, TOOLS)                       # paths still resolve from tools/
        self.assertEqual(C.UI_DIR, os.path.join(TOOLS, "pxa_control_ui"))

    def test_patches_reach_the_code_that_reads_them(self):
        old = C.DEFAULT_PORT
        try:
            C.DEFAULT_PORT = 12345
            for m in (C.core, C.bridge, C.server, C.routes):
                self.assertEqual(m.DEFAULT_PORT, 12345, m.__name__)
        finally:
            C.DEFAULT_PORT = old
        self.assertEqual(C.bridge.DEFAULT_PORT, old)
        calls = []
        orig = C.shutil_which
        C.shutil_which = lambda x: calls.append(x) or None
        try:
            C.routes.r_health(C.App(C.core.__dict__.get("L") or _launcher(), port=7777), None, {})
        finally:
            C.shutil_which = orig
        self.assertIn("nvidia-smi", calls)                    # r_health (routes) saw the patch made on the package

    def test_the_shim_loaded_by_path_gives_the_same_objects(self):
        spec = importlib.util.spec_from_file_location("pxa_control_by_path", os.path.join(TOOLS, "pxa_control.py"))
        shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shim)
        self.assertIs(shim.App, C.App)
        self.assertIs(shim.serve, C.serve)
        self.assertIs(sys.modules["pxa_control"], C)

    def test_entry_points(self):
        env = dict(os.environ, PYTHONPATH=TOOLS)
        for argv in ([sys.executable, "-m", "pxa_control", "--version"], [sys.executable, os.path.join(TOOLS, "pxa_control.py"), "--version"]):
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env, cwd=TMP)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.startswith("PXA Control "), r.stdout)

    def test_layering_is_acyclic(self):
        """core imports none of its siblings at module level; each module imports only the ones before it."""
        order = ["core", "auth", "bridge", "routes", "server", "desktop"]
        for i, m in enumerate(order):
            tree = ast.parse(open(os.path.join(PKG, m + ".py")).read())
            for node in tree.body:                         # module level only (a function may import lazily)
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    dep = (node.module or "").split(".")[0] or [a.name for a in node.names][0]
                    if dep in order:
                        self.assertLess(order.index(dep), i, f"{m} imports {dep} at module level")

    def test_no_undefined_global_in_any_function(self):
        bad = []
        for m in MODULES:
            mod = getattr(C, m)
            st = symtable.symtable(open(mod.__file__).read(), mod.__file__, "exec")

            def walk(t):
                for s in t.get_symbols():
                    if t.get_type() != "module" and s.is_global() and s.is_referenced():
                        n = s.get_name()
                        if n not in vars(mod) and not hasattr(builtins, n):
                            bad.append(f"{m}.{t.get_name()}: {n}")
                for c in t.get_children():
                    walk(c)
            walk(st)
        self.assertEqual(bad, [])


def _launcher():
    return D.load_launcher()


class Desktop(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(dir=TMP)
        self.env_backup = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_engine_inside_the_app_is_found_and_made_runnable(self):
        b = os.path.join(self.home, "bundle")
        os.makedirs(os.path.join(b, "engine", "bin"))
        exe = os.path.join(b, "engine", "bin", "llama-server")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(exe, 0o644)                                   # a onefile payload can drop the x bit
        os.environ["PXA_BUNDLE_ROOT"] = b
        d, why = D.find_engine()
        self.assertEqual((d, why), (os.path.join(b, "engine"), "inside this app"))
        self.assertTrue(os.stat(exe).st_mode & stat.S_IXUSR)
        os.environ["PXA_ENGINE_DIR"] = "/opt/mine"
        self.assertEqual(D.find_engine(), ("/opt/mine", "PXA_ENGINE_DIR"))

    def test_no_engine_leaves_the_search_to_the_launcher(self):
        os.environ["PXA_BUNDLE_ROOT"] = os.path.join(self.home, "empty")
        d, why = D.find_engine()
        self.assertIsNone(d)
        self.assertIn("searches", why)

    def test_first_run_models_folder(self):
        os.environ["HOME"] = self.home
        os.environ["USERPROFILE"] = self.home
        os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(self.home, "cfg")
        want = os.path.join(self.home, "PXA", "models")
        self.assertEqual(D.models_dirs_for_first_run(), [want])
        self.assertTrue(os.path.isdir(want))
        with C._cfg_lock:
            c = C.load_config()
            c["model_dirs"] = ["/data/models"]
            C.save_config(c)
        self.assertEqual(D.models_dirs_for_first_run(), [])     # remembered: never asked again
        os.environ["PXA_MODELS_DIR"] = "/x"
        self.assertEqual(D.models_dirs_for_first_run(), [])

    def test_launcher_loads_and_source_build_is_not_frozen(self):
        L = D.load_launcher()
        self.assertTrue(hasattr(L, "build_parser"))
        self.assertFalse(D.frozen())
        self.assertEqual(D.app_dir(), ROOT)

    def test_second_start_reuses_the_running_control(self):
        opened = []
        orig_find, orig_open = C.bridge.find_running, C.server.open_in_browser
        D._bridge.find_running = lambda **k: {"port": 7790, "url": "http://127.0.0.1:7790/?token=t", "pid": 1}
        D._server.open_in_browser = lambda url, wait=False: opened.append(url) or True
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = D.main([])
        finally:
            D._bridge.find_running, D._server.open_in_browser = orig_find, orig_open
        self.assertEqual(rc, 0)
        self.assertEqual(opened, ["http://127.0.0.1:7790/?token=t"])   # signed in, and no second server
        self.assertNotIn("token=t", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=1)
