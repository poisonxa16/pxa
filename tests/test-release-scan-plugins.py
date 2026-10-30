#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""scripts/release-scan.py: no plugin payload ships inside a package (beta 3 and later). Each payload shape (a .pxp,
a renamed .pxp, a plugin manifest, Anibal or plugin-framework code as files, as a onefile module, as module names in a
binary) must FAIL (exit 2); the Control's own mentions of the Store item must PASS (exit 0); a rule that stops seeing
payloads must stop the gate at its self-test (exit 3). No GPU, no network.

    python3 tests/test-release-scan-plugins.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(REPO, "scripts", "release-scan.py")
BLOCKLIST = "# fixture block list\n@category planted scope=text,binary\nZZPLUGINFIXTUREZZ\n"


def load():
    spec = importlib.util.spec_from_file_location("rscan_plugins_t", SCAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RS = load()
RS._set_jobs(1)          # in-process self-test: no worker pool for a module loaded by path


class PluginPayloads(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rscan-plugins-")
        self.bl = os.path.join(self.d, "bl.txt")
        open(self.bl, "w").write(BLOCKLIST)
        self.pkg = os.path.join(self.d, "pkg")
        self.put("tools/pxa_control/core.py", b"print('control')\n")
        self.put("tools/pxa_control/speech/tts.py", b"VOICES = 'anibal/voice/models'\n")
        self.put("tools/pxa_control_ui/plugins.js", b"if (it.id === 'anibal') show(anibal.name);\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def put(self, rel, data=b"x\n"):
        p = os.path.join(self.pkg, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def scan(self):
        work = os.path.join(self.d, "work")
        os.makedirs(work, exist_ok=True)
        r = subprocess.run([sys.executable, "-I", SCAN, "--blocklist", self.bl, "--work", work, self.pkg],
                           capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr

    def test_clean_control_passes(self):
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)

    def test_each_shape_blocks(self):
        shapes = {
            "pxp": lambda: self.put("share/anibal-0.1.0.pxp"),
            "src manifest": lambda: self.put("tools/x/plugin.src.json", b"{}"),
            "manifest": lambda: self.put("plugins/hello/plugin.json", b'{"id": "hello"}'),
            "anibal tree": lambda: self.put("tools/anibal/core/brain.py"),
            "vigil tree": lambda: self.put("tools/vigil/run.py"),
            "plugin framework": lambda: self.put("tools/pxa_plugins/host.py"),
            "private tree": lambda: self.put("tools/pxa-private/plugins/dictate/README.md"),
            "anibal package": lambda: self.put("lib/python3/site-packages/anibal/__init__.py"),
            "module import": lambda: self.put("tools/pxa_control/extra.py", b"from anibal" b".core import brain\n"),
        }
        for name, make in shapes.items():
            with self.subTest(name):
                shutil.rmtree(self.pkg)
                self.setUp_pkg_only()
                make()
                rc, out = self.scan()
                self.assertEqual(rc, 2, out)
                self.assertIn("plugin-payload", out)

    def setUp_pkg_only(self):
        self.put("tools/pxa_control/core.py", b"print('control')\n")

    def test_renamed_pxp_blocks(self):
        with zipfile.ZipFile(self.put("bin/data.bin", b""), "w") as z:
            z.writestr("plugin.json", "{}")
            z.writestr("bin/run", "x")
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("plugin-payload", out)

    def test_onefile_module_and_binary_names_block(self):
        stub = bytearray(0x80)
        stub[0:2] = b"MZ"
        stub[0x3C:0x40] = (0x80).to_bytes(4, "little")
        stub = bytes(stub) + b"PE\0\0"
        self.put("PXA-Control.exe", RS.make_pyinstaller_sample([("anibal" ".voice.bridges", "x = 1")], stub=stub))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("PXA-Control.exe!/", out)
        os.remove(os.path.join(self.pkg, "PXA-Control.exe"))
        self.put("lib/libx.so", RS.ELF_MAGIC + b"\x02\x01\x01" + b"\0" * 73 + b"pxa_plugins" b".host\0")
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("lib/libx.so", out)

    def test_blind_rule_stops_the_gate(self):
        rules = RS.load_rules(self.bl)
        saved = RS.PLUGIN_PATH_RX
        RS.PLUGIN_PATH_RX = __import__("re").compile(r"(?!x)x")
        try:
            with self.assertRaises(RS.GateError):
                RS.self_test(rules, self.d)
        finally:
            RS.PLUGIN_PATH_RX = saved
        self.assertGreater(RS.self_test(rules, self.d), 0)


class ControlStageGuard(unittest.TestCase):
    def test_stage_refuses_plugin_files(self):
        spec = importlib.util.spec_from_file_location("pxa_control_stage_t", os.path.join(REPO, "packaging", "control", "stage.py"))
        st = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(st)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "pxa_control", "speech"))
            open(os.path.join(d, "pxa_control", "speech", "tts.py"), "w").write("V = 'anibal/voice'\n")
            self.assertEqual(st.plugin_leaks(d), [])
            for rel, body in (("anibal/__init__.py", ""), ("pxa_control/x.pxp", ""), ("pxa_control/y.py", "import pxa_plugins" ".host\n")):
                os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
                open(os.path.join(d, rel), "w").write(body)
            self.assertEqual(st.plugin_leaks(d), ["anibal/__init__.py", "pxa_control/x.pxp", "pxa_control/y.py"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
