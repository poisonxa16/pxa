#!/usr/bin/env python3
"""The plugin kit (pxa_control/plugins/sdk.py): the plain-source part of PXA Control that plugins and the speech service
import through PXA_CONTROL_TOOLS, shipped as plugin-sdk.zip in a compiled Control and unpacked once into the config folder.

    python3 tests/test-pxa-plugin-sdk.py

Checks: the kit is closed (every module imports in an isolated Python that sees only the kit, the grant check runs, nothing
reaches the rest of Control), every plugin's shipped code imports only kit modules, the zip is the same bytes on every
build, and an unpacked kit is used only when every file matches the kit's manifest.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
PLUGINS = os.path.join(TOOLS, "pxa-private", "plugins")
TMP = tempfile.mkdtemp(prefix="pxa-plugin-sdk-test-")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)

from pxa_control.plugins import sdk  # noqa: E402

# a Python that sees only the kit: Control's other modules must never load, the grant check must work
CLOSED = r'''
import importlib, os, sys
kit, cfg = sys.argv[1], sys.argv[2]
sys.path.insert(0, kit)
OUT = ("pxa_control.core", "pxa_control.server", "pxa_control.routes", "pxa_control.desktop", "pxa_control.auth",
       "pxa_control.bridge", "pxa_control.share", "pxa_control.plugins.host", "pxa_control.plugins.ui",
       "pxa_control.plugins.pxp", "pxa_control.plugins.egress", "pxa_control.plugins.overlay", "pxa_control.plugins.keystore")
class Block(object):
    def find_spec(self, name, path=None, target=None):
        if name in OUT:
            raise ImportError("the kit reached " + name)
sys.meta_path.insert(0, Block())
names = [n for n in sys.argv[3:]]
for n in names:
    m = importlib.import_module(n)
    assert m.__file__.startswith(kit), (n, m.__file__)
from pxa_control.plugins import gate, ConsentStore
st = gate.state(cfg, "satellite")
assert st["state"] == "missing", st
ConsentStore(cfg)
from pxa_control.speech import registry, setup
assert registry.load_registry()
assert len(setup.pins_id()) == 16          # the engine locks are in the kit: every speech status reads them
print("closed", len(names))
'''


def kit_modules(names):
    out = []
    for n in names:
        if n.endswith(".py"):
            m = n[:-3].replace("/", ".")
            out.append(m[:-len(".__init__")] if m.endswith(".__init__") else m)
    return sorted(out)


class Kit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zip = os.path.join(TMP, sdk.KIT_ZIP)
        cls.sha = sdk.build(cls.zip)
        with zipfile.ZipFile(cls.zip) as z:
            cls.names = z.namelist()
            cls.manifest = json.loads(z.read(sdk.MANIFEST))
            cls.init = z.read("pxa_control/__init__.py").decode()

    def test_same_bytes_on_every_build(self):
        again = os.path.join(TMP, "again.zip")
        self.assertEqual(sdk.build(again), self.sha)
        with open(self.zip, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), self.sha)

    def test_what_is_in_it(self):
        for n in ("pxa_control/__init__.py", "pxa_control/mdns.py", "pxa_control/plugins/__init__.py",
                  "pxa_control/plugins/gate.py", "pxa_control/plugins/child.py", "pxa_control/store/__init__.py",
                  "pxa_control/speech/__main__.py", "pxa_control/speech/setup.py", "pxa_control/speech/models.json",
                  "pxa_control/speech/engines.lock", "pxa_control/speech/kokoro.lock"):
            self.assertIn(n, self.names)
        for n in self.names:
            self.assertNotRegex(n, r"(^|/)(tests?|fixtures|bench-results|__pycache__)/|\.pyc$|^pxa_control/(core|auth|bridge|routes|"
                                   r"server|share|desktop)\.py$|^pxa_control/plugins/(host|ui|pxp|egress|overlay|keystore)\.py$")
        self.assertNotIn("import", self.init)                       # the stub: importing the kit pulls in nothing else

    def test_manifest_names_every_file_with_its_sha(self):
        with zipfile.ZipFile(self.zip) as z:
            for n in self.names:
                if n != sdk.MANIFEST:
                    self.assertEqual(self.manifest["files"][n], hashlib.sha256(z.read(n)).hexdigest(), n)
        self.assertEqual(set(self.manifest["files"]), set(self.names) - {sdk.MANIFEST})

    def test_closed_in_a_python_that_sees_only_the_kit(self):
        kit = os.path.join(TMP, "closed")
        with zipfile.ZipFile(self.zip) as z:
            z.extractall(kit)
        cfg = tempfile.mkdtemp(dir=TMP)
        mods = kit_modules(self.names)
        out = subprocess.run([sys.executable, "-I", "-B", "-c", CLOSED, kit, cfg] + mods, cwd=tempfile.mkdtemp(dir=TMP),
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("closed %d" % len(mods), out.stdout)

    def test_every_plugin_imports_only_kit_modules(self):
        mods = set(kit_modules(self.names))
        rx = re.compile(r"^\s*(?:from\s+(pxa_control[\w.]*)\s+import\s+([\w, ]+)|import\s+(pxa_control[\w.]*))", re.M)
        seen = 0
        for pid in sorted(os.listdir(PLUGINS)):
            src = os.path.join(PLUGINS, pid, "plugin.src.json")
            if not os.path.isfile(src):
                continue
            with open(src) as f:
                dev = set(json.load(f).get("dev_only") or [])
            for dp, dn, fn in os.walk(os.path.join(PLUGINS, pid)):
                dn[:] = [d for d in dn if d not in ("tests", "test", "__pycache__")]
                for f in fn:
                    rel = os.path.relpath(os.path.join(dp, f), os.path.join(PLUGINS, pid)).replace(os.sep, "/")
                    if not f.endswith(".py") or rel in dev or f.startswith("test"):
                        continue
                    with open(os.path.join(dp, f), encoding="utf-8") as fh:
                        for m in rx.finditer(fh.read()):
                            base = m.group(1) or m.group(3)
                            want = [base] + [base + "." + x.strip() for x in (m.group(2) or "").split(",")
                                             if (base + "." + x.strip()) in mods or x.strip()[:1].islower() and
                                             os.path.exists(os.path.join(TOOLS, *(base + "." + x.strip()).split(".")) + ".py")]
                            for w in want:
                                seen += 1
                                self.assertIn(w, mods, "%s/%s imports %s, which a compiled Control does not ship" % (pid, rel, w))
        self.assertGreater(seen, 10)


class Ensure(unittest.TestCase):
    def setUp(self):
        sdk.forget()
        self.d = tempfile.mkdtemp(dir=TMP)
        self.cfg = os.path.join(self.d, "cfg")
        self.bundle = os.path.join(self.d, "bundle")
        os.makedirs(self.cfg)
        os.makedirs(self.bundle)
        self.sha = sdk.build(os.path.join(self.bundle, sdk.KIT_ZIP))
        self.home = os.path.join(self.cfg, "plugins", "sdk", self.sha[:16])

    def test_unpacked_once_into_the_config_folder(self):
        self.assertEqual(sdk.ensure(self.cfg, self.bundle), self.home)
        self.assertTrue(os.path.isfile(os.path.join(self.home, "pxa_control", "plugins", "gate.py")))
        mark = os.path.join(self.home, "kept")
        open(mark, "w").close()
        sdk.forget()
        self.assertEqual(sdk.ensure(self.cfg, self.bundle), self.home)
        self.assertTrue(os.path.exists(mark))                              # intact: not unpacked again

    def test_a_changed_file_is_put_back(self):
        sdk.ensure(self.cfg, self.bundle)
        gate = os.path.join(self.home, "pxa_control", "plugins", "gate.py")
        with open(gate, "a") as f:
            f.write("\nRAN = True\n")
        sdk.forget()
        self.assertEqual(sdk.ensure(self.cfg, self.bundle), self.home)
        with open(gate) as f:
            self.assertNotIn("RAN = True", f.read())

    def test_no_kit_in_the_bundle(self):
        self.assertIsNone(sdk.ensure(self.cfg, self.d))
        self.assertIsNone(sdk.ensure(self.cfg, ""))

    def test_a_kit_with_unsafe_names_is_refused(self):
        bad = os.path.join(self.d, "bad")
        os.makedirs(bad)
        with zipfile.ZipFile(os.path.join(bad, sdk.KIT_ZIP), "w") as z:
            z.writestr(sdk.MANIFEST, json.dumps({"files": {}}))
            z.writestr("../evil.py", "x = 1\n")
        self.assertIsNone(sdk.ensure(self.cfg, bad))
        self.assertFalse(os.path.exists(os.path.join(self.cfg, "plugins", "evil.py")))

    def test_source_run_uses_the_tree(self):
        self.assertEqual(sdk.tools_dir(self.cfg, frozen_=False), TOOLS)
        self.assertEqual(sdk.tools_dir(self.cfg, frozen_=True, bundle=self.bundle), self.home)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
