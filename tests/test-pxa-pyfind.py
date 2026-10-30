#!/usr/bin/env python3
"""The Python that runs plugins and the speech service (pxa_control/pyfind.py). A compiled Control's sys.executable is
Control itself, so it cannot be that Python: source run = this interpreter; compiled = PXA_CONTROL_PYTHON, then the Python
the bundle carries (Windows: python.org's embeddable package, unpacked once into the config folder), then one found on
this computer (3.10 - 3.14, 64-bit; Microsoft Store aliases and the macOS developer-tools stub are never started).

    python3 tests/test-pxa-pyfind.py

POSIX: the stand-in Pythons are shell scripts that answer the version probe.
"""
import hashlib
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-pyfind-test-")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)

from pxa_control import pyfind as PF  # noqa: E402


def fake(d, name, answer):
    """a stand-in Python: prints `answer` for the probe (whatever the arguments)."""
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write("#!/bin/sh\necho %s\n" % answer)
    os.chmod(p, 0o700)
    return p


class Find(unittest.TestCase):
    def setUp(self):
        PF.forget()
        self.d = tempfile.mkdtemp(dir=TMP)
        self.cfg = os.path.join(self.d, "cfg")
        os.makedirs(self.cfg)

    def find(self, **kw):
        kw.setdefault("frozen_", True)
        kw.setdefault("env", {})
        kw.setdefault("bundle", "")
        kw.setdefault("cands", [])
        return PF.find(self.cfg, **kw)

    def test_a_source_run_uses_its_own_interpreter(self):
        self.assertEqual(PF.find(self.cfg, frozen_=False), sys.executable)

    def test_compiled_override_first_when_it_is_a_usable_python(self):
        good = fake(self.d, "py-override", "3 13 64")
        found = fake(self.d, "py-found", "3 12 64")
        self.assertEqual(self.find(env={"PXA_CONTROL_PYTHON": good}, cands=[found]), good)
        PF.forget()
        old = fake(self.d, "py-old", "3 9 64")
        self.assertEqual(self.find(env={"PXA_CONTROL_PYTHON": old}, cands=[found]), found)

    def test_found_pythons_must_be_310_to_314_and_64_bit(self):
        c = [fake(self.d, "a", "3 9 64"), fake(self.d, "b", "3 15 64"), fake(self.d, "c", "3 12 32"),
             fake(self.d, "d", "Python was not found"), os.path.join(self.d, "no-such-python"), fake(self.d, "e", "3 12 64")]
        self.assertEqual(self.find(cands=c), c[-1])

    def test_nothing_usable_is_none_and_asked_again_after_a_while(self):
        t, c = [100.0], []
        self.assertIsNone(self.find(cands=c, clock=lambda: t[0]))
        c.append(fake(self.d, "late", "3 12 64"))
        self.assertIsNone(self.find(cands=c, clock=lambda: t[0]))           # remembered for a while: no probe on every call
        t[0] += PF.MISS_TTL_S + 1
        self.assertEqual(self.find(cands=c, clock=lambda: t[0]), c[0])     # Python installed later: no restart needed
        c[0] = os.path.join(self.d, "gone")
        t[0] += 10 * PF.MISS_TTL_S
        self.assertEqual(self.find(cands=c, clock=lambda: t[0]), os.path.join(self.d, "late"))   # a hit is kept

    def test_the_probe_gets_no_secret(self):
        p = os.path.join(self.d, "py-env")
        with open(p, "w") as f:
            f.write('#!/bin/sh\n[ -z "$HF_TOKEN$PXA_CONTROL_TOKEN_TEST_MARK" ] && echo 3 12 64\n')
        os.chmod(p, 0o700)
        os.environ.update(HF_TOKEN="hf_this_must_never_reach_a_probe", PXA_CONTROL_TOKEN_TEST_MARK="nor-this")
        try:
            self.assertEqual(PF.probe(p), (3, 12, 64))
        finally:
            os.environ.pop("HF_TOKEN")
            os.environ.pop("PXA_CONTROL_TOKEN_TEST_MARK")

    def test_probe_reads_this_interpreter(self):
        self.assertEqual(PF.probe(sys.executable), (sys.version_info[0], sys.version_info[1], struct.calcsize("P") * 8))


class Bundled(unittest.TestCase):
    """the Python a compiled Windows Control carries: <bundle>/python-embed.zip, unpacked once to <config>/python/<sha16>."""

    def setUp(self):
        PF.forget()
        self.d = tempfile.mkdtemp(dir=TMP)
        self.cfg = os.path.join(self.d, "cfg")
        self.bundle = os.path.join(self.d, "bundle")
        os.makedirs(self.cfg)
        os.makedirs(self.bundle)
        self.zip = os.path.join(self.bundle, PF.EMBED_ZIP)

    def make(self, names=None, answer="3 13 64"):
        with zipfile.ZipFile(self.zip, "w") as z:
            for n in names or (PF.EMBED_EXE, "python313._pth", "python313.zip"):
                z.writestr(n, "#!/bin/sh\necho %s\n" % answer if n == PF.EMBED_EXE else "x\n")
        with open(self.zip, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def find(self, cands=()):
        return PF.find(self.cfg, frozen_=True, env={}, bundle=self.bundle, cands=list(cands))

    def test_unpacked_once_into_the_config_folder(self):
        sha = self.make()
        py = self.find()
        home = os.path.join(self.cfg, "python", sha[:16])
        self.assertEqual(py, os.path.join(home, PF.EMBED_EXE))
        self.assertTrue(os.path.isfile(os.path.join(home, "python313._pth")))
        mark = os.path.join(home, "kept")
        open(mark, "w").close()
        PF.forget()
        self.assertEqual(self.find(), py)
        self.assertTrue(os.path.exists(mark))                              # not unpacked again
        self.assertEqual([n for n in os.listdir(os.path.join(self.cfg, "python")) if n != sha[:16]], [])

    def test_a_half_unpacked_copy_is_replaced(self):
        sha = self.make()
        home = os.path.join(self.cfg, "python", sha[:16])
        os.makedirs(home)
        open(os.path.join(home, "python313._pth"), "w").close()            # an earlier unpack that stopped half way
        self.assertEqual(self.find(), os.path.join(home, PF.EMBED_EXE))

    def test_bundled_comes_before_a_found_python(self):
        self.make()
        found = fake(self.d, "py-found", "3 12 64")
        self.assertNotEqual(self.find([found]), found)

    def test_a_bundled_python_that_does_not_run_falls_back_to_a_found_one(self):
        self.make(answer="3 8 64")
        found = fake(self.d, "py-found", "3 12 64")
        self.assertEqual(self.find([found]), found)

    def test_unsafe_names_are_never_unpacked(self):
        for bad in ("../evil", "/abs/evil", "C:/evil", "a/../../evil"):
            shutil.rmtree(self.cfg)
            os.makedirs(self.cfg)
            PF.forget()
            self.make(names=(PF.EMBED_EXE, bad))
            self.assertIsNone(self.find(), bad)
            self.assertFalse(os.path.exists(os.path.join(self.d, "evil")))
            home = os.path.join(self.cfg, "python")
            self.assertEqual(os.listdir(home) if os.path.isdir(home) else [], [], bad)   # nothing left behind


class Candidates(unittest.TestCase):
    def test_windows_registry_then_path_without_store_aliases_then_usual_folders(self):
        which = {"python": r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe",
                 "python3": r"C:\Tools\Python312\python.exe"}.get
        c = PF.candidates(plat="win32", env={"LOCALAPPDATA": r"C:\Users\u\AppData\Local", "ProgramFiles": r"C:\Program Files"},
                          which=which, registry=lambda: [r"C:\Py313\python.exe"], exists=lambda p: True)
        self.assertEqual(c[:2], [r"C:\Py313\python.exe", r"C:\Tools\Python312\python.exe"])
        self.assertFalse([p for p in c if "WindowsApps" in p])
        self.assertIn(r"C:\Users\u\AppData\Local\Programs\Python\Python313\python.exe", c)
        self.assertIn(r"C:\Program Files\Python314\python.exe", c)
        self.assertEqual(len(c), len(set(c)))

    def test_macos_never_starts_the_developer_tools_stub(self):
        c = PF.candidates(plat="darwin", env={}, which=lambda n: "/usr/bin/python3" if n == "python3" else None,
                          registry=lambda: [], exists=lambda p: True)
        self.assertNotIn("/usr/bin/python3", c)
        self.assertEqual(c[0], "/opt/homebrew/bin/python3")
        self.assertIn("/Library/Frameworks/Python.framework/Versions/3.13/bin/python3", c)

    def test_linux_newest_named_first(self):
        which = {"python3.12": "/usr/bin/python3.12", "python3": "/usr/bin/python3"}.get
        c = PF.candidates(plat="linux", env={}, which=which, registry=lambda: [], exists=lambda p: p.startswith("/usr/bin/"))
        self.assertEqual(c, ["/usr/bin/python3.12", "/usr/bin/python3"])

    def test_a_python_in_the_current_folder_is_never_started(self):
        # Windows' which() looks in the current folder before PATH (a python.exe beside a download); a relative or empty
        # PATH entry does the same anywhere: only absolute paths are candidates
        for plat, which, planted in (
                ("win32", {"python": r".\python.exe", "python3": r"C:\Tools\Python312\python.exe"}.get, r".\python.exe"),
                ("linux", {"python3.12": "python3.12", "python3": "/usr/bin/python3"}.get, "python3.12"),
                ("darwin", {"python3.13": "python3.13"}.get, "python3.13")):
            c = PF.candidates(plat=plat, env={}, which=which, registry=lambda: [], exists=lambda p: True)
            self.assertNotIn(planted, c, plat)
            self.assertTrue(c, plat)                                         # the absolute ones are still there

    def test_registry_tags_skip_32_bit_and_old(self):
        rows = [("3.13", r"C:\Py313"), ("3.12-32", r"C:\Py312-32"), ("3.9", r"C:\Py39"), ("3.14t", r"C:\Py314t"),
                ("3.14", r"C:\Py314")]
        self.assertEqual(PF.pick_registry(rows), [r"C:\Py314\python.exe", r"C:\Py313\python.exe"])


class Words(unittest.TestCase):
    def test_what_to_do_when_there_is_no_python(self):
        self.assertIn("Reinstall PXA Control", PF.missing_words("win32"))
        self.assertIn("python.org", PF.missing_words("darwin"))
        self.assertIn("package manager", PF.missing_words("linux"))
        for p in ("win32", "darwin", "linux"):
            self.assertIn("3.10", PF.missing_words(p))


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
