#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Guard logic for scripts/make-windows-release.sh, on fixtures. No GPU, no publish.

    python3 tests/test-make-windows-release.py
"""
import importlib.util
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE = os.environ.get("PXA_WIN_STAGE") or ""   # the build host's stage (no default path; the DLL cases skip without it)
REAL_DLL = os.environ.get("PXA_WIN_DLL") or (_STAGE and _STAGE + "/ggml-pxqn.dll")
OLD_DLL = os.environ.get("PXA_WIN_OLD_DLL") or (_STAGE and _STAGE + "/ggml-pxqn.dll.OLD-allexports-3a662b56")
BLOCKLIST = """# fixture block list for the Windows packager test
@category win-block scope=text,binary
ZZWINBLOCKTOKEN
@files win-source
*.cu
*.cpp
@selftest win-block ZZWINBLOCKTOKEN
"""


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pe_with_exports(names):
    """A tiny PE32+ DLL whose export table lists `names`. Enough for the packager's parser."""
    n = len(names)
    func_off = 40
    names_off = func_off + 4 * n
    ords_off = names_off + 4 * n
    str_off = (ords_off + 2 * n + 3) & ~3
    raw = b""
    where = []
    cursor = str_off
    for name in names:
        where.append(cursor)
        raw += name.encode("ascii") + b"\0"
        cursor += len(name) + 1
    content = bytearray(str_off + len(raw))
    content[str_off:] = raw
    struct.pack_into("<I", content, 16, 1)
    struct.pack_into("<I", content, 20, n)
    struct.pack_into("<I", content, 24, n)
    struct.pack_into("<I", content, 28, 0x1000 + func_off)
    struct.pack_into("<I", content, 32, 0x1000 + names_off)
    struct.pack_into("<I", content, 36, 0x1000 + ords_off)
    for i, off in enumerate(where):
        struct.pack_into("<I", content, func_off + 4 * i, 0x1000)
        struct.pack_into("<I", content, names_off + 4 * i, 0x1000 + off)
        struct.pack_into("<H", content, ords_off + 2 * i, i)
    dos = bytearray(0x80)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = bytearray(20)
    struct.pack_into("<H", coff, 0, 0x8664)
    struct.pack_into("<H", coff, 2, 1)
    struct.pack_into("<H", coff, 16, 240)
    struct.pack_into("<H", coff, 18, 0x2000)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, 32, 0x1000)
    struct.pack_into("<I", opt, 36, 0x200)
    struct.pack_into("<I", opt, 56, 0x2000)
    struct.pack_into("<I", opt, 60, 0x200)
    struct.pack_into("<H", opt, 68, 3)
    struct.pack_into("<I", opt, 108, 16)
    struct.pack_into("<I", opt, 112, 0x1000)
    struct.pack_into("<I", opt, 116, len(content))
    sec = bytearray(40)
    sec[0:6] = b".rdata"
    rawsize = (len(content) + 0x1FF) & ~0x1FF
    struct.pack_into("<I", sec, 8, len(content))
    struct.pack_into("<I", sec, 12, 0x1000)
    struct.pack_into("<I", sec, 16, rawsize)
    struct.pack_into("<I", sec, 20, 0x200)
    blob = bytes(dos) + b"PE\0\0" + bytes(coff) + bytes(opt) + bytes(sec)
    return blob.ljust(0x200, b"\0") + bytes(content).ljust(rawsize, b"\0")


class WindowsReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load(os.path.join(REPO, "scripts", "windows-release-guards.py"), "win_guards")
        cls.required = cls.g.def_export_names(os.path.join(REPO, cls.g.DEF_REL))
        cls.script = os.path.join(REPO, "scripts", "make-windows-release.sh")

    def test_def_list_and_roundtrip(self):
        self.assertIn("ggml_pxqn_get_api", self.required)
        self.assertGreaterEqual(len(self.required), 12)
        names = self.required + [
            "ggml_backend_cuda_pxqn_gu_counters_reset",
            "ggml_backend_cuda_pxqn_gu_decode_count",
            "ggml_backend_cuda_pxqn_gu_prefill_count",
        ]
        blob = pe_with_exports(names)
        fd, path = tempfile.mkstemp(suffix=".dll")
        os.close(fd)
        try:
            open(path, "wb").write(blob)
            got = self.g.pe_export_names(path)
            self.assertEqual(got, names)
            with self.assertRaises(self.g.GuardError):          # the release DLL exports the .def names only
                self.g.check_exports(got, self.required)
            self.assertEqual(self.g.check_exports(got, self.required, allow_hooks=True), len(names))
            self.assertEqual(self.g.check_exports(self.required, self.required), len(self.required))
        finally:
            os.remove(path)

    def test_export_guards(self):
        with self.assertRaises(self.g.GuardError):
            self.g.check_exports(["ggml_pxqn_get_api"], self.required)
        many = self.required + ["extra%d" % i for i in range(40)]
        with self.assertRaises(self.g.GuardError):
            self.g.check_exports(many, self.required)
        with self.assertRaises(self.g.GuardError):
            self.g.check_exports(self.required + ["ggml_internal_symbol"], self.required)
        self.assertEqual(self.g.check_exports(self.required + list(self.g.PXQN_HOOKS), self.required, allow_hooks=True),
                         len(self.required) + 3)
        self.assertTrue(self.g.is_old_pxqn("/tmp/ggml-pxqn.dll.OLD-allexports-3a662b56"))
        self.assertFalse(self.g.is_old_pxqn("/tmp/ggml-pxqn.dll"))

    def test_notes_wording(self):
        g = self.g
        text = open(os.path.join(REPO, "docs", "windows-release-notes.md"), encoding="utf-8").read()
        self.assertIn(g.PXQN_SEPARATE, text)
        self.assertEqual(g.check_notes_text(text), [])
        bundled = g.render_notes(text, True)
        self.assertIn(g.PXQN_BUNDLE, bundled)
        self.assertNotIn("not inside it", bundled)
        self.assertEqual(g.check_notes_text(bundled), [])
        self.assertIn("switch", g.banned_public_text("set PXA_FOO=1"))
        self.assertIn("store-word", g.banned_public_text("the price is here"))
        self.assertIn("lane", g.banned_public_text("lane grok-win-release did this"))
        self.assertIn("path", g.banned_public_text("built at /mnt/<disk>/x"))
        self.assertIn("missing-smartscreen", g.check_notes_text("Run anyway\n"))

    def test_find_pxqn_skips_old(self):
        d = tempfile.mkdtemp()
        try:
            open(os.path.join(d, "ggml-pxqn.dll.OLD-allexports-1"), "wb").write(b"old")
            good = os.path.join(d, "ggml-pxqn.dll")
            open(good, "wb").write(b"new")
            found, old = self.g.find_pxqn(d)
            self.assertEqual(os.path.realpath(found), os.path.realpath(good))
            self.assertEqual(len(old), 1)
        finally:
            shutil.rmtree(d)

    def test_real_relink_if_present(self):
        if not os.path.isfile(REAL_DLL):
            self.skipTest("staged dll not on this host")
        names = self.g.pe_export_names(REAL_DLL)
        missing = [n for n in self.required if n not in names]
        if missing:                   # a stale staged artefact, not a code fault: say so loudly, do not fail ctest
            with self.assertRaises(self.g.GuardError):
                self.g.check_exports(names, self.required, allow_hooks=True)
            self.skipTest("the staged %s predates this tree's .def (%d name(s) missing): rebuild it" % (REAL_DLL, len(missing)))
        hooks = [n for n in names if n in self.g.PXQN_HOOKS]
        self.assertIn(len(hooks), (0, 3))
        self.assertEqual(len(names), len(self.required) + len(hooks))
        if hooks:                     # built before the hook fix: refused by the release check
            with self.assertRaises(self.g.GuardError):
                self.g.check_exports(names, self.required)
            self.assertEqual(self.g.check_exports(names, self.required, allow_hooks=True), len(names))
        else:
            self.assertEqual(self.g.check_exports(names, self.required), len(self.required))
        self.assertIn("ggml_pxqn_get_api", names)

    def test_old_dll_refused_if_present(self):
        if not os.path.isfile(OLD_DLL):
            self.skipTest("old dll not on this host")
        names = self.g.pe_export_names(OLD_DLL)
        self.assertGreater(len(names), self.g.MAX_EXPORTS)
        with self.assertRaises(self.g.GuardError):
            self.g.check_exports(names, self.required)

    def _fixture(self, root, server=b"llama-server", extra_names=None):
        """A staged directory: control exe, engine zip, relinked dll, block list."""
        g = self.g
        names = list(self.required)
        if extra_names:
            names.extend(extra_names)
        bins = {
            "llama-server.exe": server,
            "ggml.dll": b"ggml",
            "cudart64_12.dll": b"cudart",
            "cublas64_12.dll": b"cublas",
            "cublasLt64_12.dll": b"cublaslt",
            "vcruntime140.dll": b"vcr",
        }
        staged = os.path.join(root, "staged")
        os.makedirs(staged)
        zpath = os.path.join(staged, "pxa-engine-win64-cuda12.zip")
        with zipfile.ZipFile(zpath, "w") as z:
            for fn, data in bins.items():
                z.writestr("pxa-engine-win64-cuda12/bin/" + fn, data)
        open(os.path.join(staged, "pxa-control-win64.exe"), "wb").write(b"control-exe")
        open(os.path.join(staged, "ggml-pxqn.dll"), "wb").write(pe_with_exports(names))
        open(os.path.join(staged, "ggml-pxqn.dll.OLD-allexports-9"), "wb").write(b"old-dll")
        bl = os.path.join(root, "blocklist.txt")
        open(bl, "w", encoding="utf-8").write(BLOCKLIST)
        inc = os.path.join(root, "closed.inc")
        open(inc, "w", encoding="utf-8").write('"NOT_IN_THE_FILES_ZZ",\n')
        return staged, bl, inc

    def _run(self, args, blocklist):
        env = os.environ.copy()
        env["PXA_RELEASE_BLOCKLIST"] = blocklist
        return subprocess.run(
            ["bash", self.script] + args, cwd=REPO, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_dry_run_lists_every_input(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, _inc = self._fixture(root)
            p = self._run(["--dry-run", "--staged", staged, "--out", os.path.join(root, "out"),
                           "--work", os.path.join(root, "work"), "--no-verify-hashes"], bl)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            for kind in ("control", "engine-zip", "notes", "blocklist", "release-scan",
                         "def-file", "pxqn ", "disposition separate", "pxqn-exports"):
                self.assertIn("OK " + kind, p.stdout)
            self.assertIn("SKIP old-pxqn", p.stdout)
            self.assertNotIn(".OLD", p.stdout.split("OK pxqn ", 1)[1].splitlines()[0])
            for member in ("llama-server.exe", "ggml.dll", "vcruntime140.dll",
                           "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
                self.assertIn("OK member " + member, p.stdout)
        finally:
            shutil.rmtree(root)

    def test_engine_bin_directory_is_an_input(self):
        # --engine-bin is a directory (the VM's loose bin\): it used to read as MISSING because only files counted
        root = tempfile.mkdtemp()
        try:
            staged, bl, _inc = self._fixture(root)
            bindir = os.path.join(root, "bin")
            os.makedirs(bindir)
            with zipfile.ZipFile(os.path.join(staged, "pxa-engine-win64-cuda12.zip")) as z:
                for n in z.namelist():
                    open(os.path.join(bindir, n.rsplit("/", 1)[1]), "wb").write(z.read(n))
            out = os.path.join(root, "out")
            p = self._run(["--engine-bin", bindir, "--control", os.path.join(staged, "pxa-control-win64.exe"),
                           "--pxqn", os.path.join(staged, "ggml-pxqn.dll"), "--out", out,
                           "--work", os.path.join(root, "work")], bl)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with zipfile.ZipFile(os.path.join(out, "pxa-engine-win64-cuda12.zip")) as z:
                self.assertIn("pxa-engine-win64-cuda12/bin/llama-server.exe", z.namelist())
            p = self._run(["--dry-run", "--engine-bin", bindir, "--control", os.path.join(staged, "pxa-control-win64.exe"),
                           "--pxqn", os.path.join(staged, "ggml-pxqn.dll"), "--out", os.path.join(root, "out3")], bl)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("OK engine-bin " + bindir, p.stdout)
            empty = os.path.join(root, "empty")
            os.makedirs(empty)
            p = self._run(["--dry-run", "--engine-bin", empty, "--control", os.path.join(staged, "pxa-control-win64.exe"),
                           "--pxqn", os.path.join(staged, "ggml-pxqn.dll"), "--out", os.path.join(root, "out2")], bl)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("MISSING engine-bin", p.stdout)
        finally:
            shutil.rmtree(root)

    def test_dry_run_missing_member(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, _inc = self._fixture(root)
            # Rebuild the zip without the CUDA runtime.
            zpath = os.path.join(staged, "pxa-engine-win64-cuda12.zip")
            os.remove(zpath)
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("pxa-engine-win64-cuda12/bin/llama-server.exe", b"exe")
                z.writestr("pxa-engine-win64-cuda12/bin/ggml.dll", b"ggml")
                z.writestr("pxa-engine-win64-cuda12/bin/vcruntime140.dll", b"vcr")
            p = self._run(["--dry-run", "--staged", staged, "--out", os.path.join(root, "out"),
                           "--no-verify-hashes"], bl)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("MISSING member cudart64_*.dll", p.stdout)
            self.assertIn("MISSING member cublas64_*.dll", p.stdout)
            self.assertIn("MISSING member cublasLt64_*.dll", p.stdout)
        finally:
            shutil.rmtree(root)

    def test_assemble_separate_and_sha256(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, _inc = self._fixture(root)
            out = os.path.join(root, "out")
            p = self._run(["--staged", staged, "--out", out, "--work", os.path.join(root, "work"),
                           "--no-verify-hashes"], bl)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            for name in ("pxa-control-win64.exe", "pxa-engine-win64-cuda12.zip", "ggml-pxqn.dll",
                         "LICENSE-ggml-pxqn.txt", "RELEASE-NOTES-windows.md", "SHA256SUMS"):
                self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
            with zipfile.ZipFile(os.path.join(out, "pxa-engine-win64-cuda12.zip")) as z:
                names = [n.replace("\\", "/") for n in z.namelist()]
            self.assertTrue(any(n.endswith("bin/llama-server.exe") for n in names))
            self.assertFalse(any(n.lower().endswith("ggml-pxqn.dll") for n in names))
            self.assertIn("pxa-engine-win64-cuda12/LICENSE", names)
            self.assertIn("pxa-engine-win64-cuda12/NOTICE", names)
            self.assertIn("pxa-engine-win64-cuda12/LICENSE-ggml-pxqn.txt", names)
            with open(os.path.join(out, "SHA256SUMS")) as f:
                self.assertIn("LICENSE-ggml-pxqn.txt", f.read())
            with zipfile.ZipFile(os.path.join(out, "pxa-engine-win64-cuda12.zip")) as z:
                notice = z.read("pxa-engine-win64-cuda12/NOTICE").decode("utf-8")
            self.assertIn("NVIDIA", notice)
            self.assertIn("Microsoft", notice)
            self.assertFalse(os.path.exists(os.path.join(root, "work")))
            notes = open(os.path.join(out, "RELEASE-NOTES-windows.md"), encoding="utf-8").read()
            self.assertIn("SmartScreen", notes)
            self.assertIn("Run anyway", notes)
            self.assertIn("not inside it", notes)
            c = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=out, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(c.returncode, 0, c.stdout + c.stderr)
        finally:
            shutil.rmtree(root)

    def test_foreign_work_dir_is_refused_and_kept(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, _inc = self._fixture(root)
            work = os.path.join(root, "someone-elses-dir")
            os.makedirs(work)
            keep = os.path.join(work, "keep.txt")
            open(keep, "w").write("x")
            p = self._run(["--staged", staged, "--out", os.path.join(root, "out"), "--work", work,
                           "--no-verify-hashes"], bl)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("not made by this packager", p.stderr + p.stdout)
            self.assertTrue(os.path.isfile(keep))
        finally:
            shutil.rmtree(root)

    def test_bundle_puts_the_library_in_the_zip(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, inc = self._fixture(root)
            out = os.path.join(root, "out")
            p = self._run(["--staged", staged, "--out", out, "--work", os.path.join(root, "work"),
                           "--bundle-pxqn", "--closed-inc", inc, "--no-verify-hashes"], bl)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with zipfile.ZipFile(os.path.join(out, "pxa-engine-win64-cuda12.zip")) as z:
                names = [n.replace("\\", "/") for n in z.namelist()]
            self.assertTrue(any(n.endswith("bin/ggml-pxqn.dll") for n in names))
            notes = open(os.path.join(out, "RELEASE-NOTES-windows.md"), encoding="utf-8").read()
            self.assertIn(self.g.PXQN_BUNDLE, notes)
            self.assertTrue(os.path.isfile(os.path.join(out, "ggml-pxqn.dll")))
        finally:
            shutil.rmtree(root)

    def test_closed_lever_in_an_engine_binary_refuses_the_package(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, inc = self._fixture(root, server=b"prefix " + b"PXA_XCACHE_MTP_N" + b"O_TICK" + b" suffix")
            out = os.path.join(root, "out")
            p = self._run(["--staged", staged, "--out", out, "--work", os.path.join(root, "work"),
                           "--closed-inc", inc, "--no-verify-hashes"], bl)
            self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("CLOSED", p.stdout + p.stderr)
            self.assertFalse(os.path.isfile(os.path.join(out, "pxa-engine-win64-cuda12.zip")))
        finally:
            shutil.rmtree(root)

    def test_scan_gate_refuses_a_blocked_string(self):
        root = tempfile.mkdtemp()
        try:
            staged, bl, inc = self._fixture(root)
            notes = os.path.join(root, "notes.md")
            text = open(os.path.join(REPO, "docs", "windows-release-notes.md"), encoding="utf-8").read()
            open(notes, "w", encoding="utf-8").write(text + "\nZZWINBLOCKTOKEN\n")
            out = os.path.join(root, "out")
            p = self._run(["--staged", staged, "--out", out, "--work", os.path.join(root, "work"),
                           "--notes", notes, "--closed-inc", inc, "--no-verify-hashes"], bl)
            self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("clean-package gate FAILED", p.stderr)
            self.assertFalse(os.path.isfile(os.path.join(out, "SHA256SUMS")))
        finally:
            shutil.rmtree(root)

    def test_source_and_driver_are_refused(self):
        g = self.g
        root = tempfile.mkdtemp()
        try:
            staged, bl, _inc = self._fixture(root)
            zpath = os.path.join(staged, "pxa-engine-win64-cuda12.zip")
            os.remove(zpath)
            with zipfile.ZipFile(zpath, "w") as z:
                for fn in ("llama-server.exe", "ggml.dll", "cudart64_12.dll", "cublas64_12.dll",
                           "cublasLt64_12.dll", "vcruntime140.dll"):
                    z.writestr("bin/" + fn, b"x")
                z.writestr("bin/nvcuda.dll", b"driver")
                z.writestr("bin/secret.cu", b"kernel")
                z.writestr("bin/anibal-0.1.0.pxp", b"PK")
                z.writestr("bin/plugins/x/plugin.json", b"{}")
            opt = g._parser().parse_args(["--staged", staged, "--out", os.path.join(root, "out"),
                                           "--work", os.path.join(root, "work"), "--no-verify-hashes",
                                           "--blocklist", bl])
            lines = g.evaluate(g.build_plan(opt))
            self.assertGreater(lines.bad, 0)
            text = "\n".join(lines.rows)
            self.assertIn("BAD member nvcuda.dll driver-dll", text)
            self.assertIn("BAD member secret.cu source-file", text)
            self.assertIn("BAD member anibal-0.1.0.pxp plugin-payload", text)
            self.assertIn("BAD member plugin.json plugin-payload", text)
        finally:
            shutil.rmtree(root)

    def _scan(self, blocklist, work, target):
        return subprocess.run(
            ["python3", os.path.join(REPO, "scripts", "release-scan.py"),
             "--blocklist", blocklist, "--work", work, "--jobs", "1", target],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_release_scan_pe_and_zip(self):
        root = tempfile.mkdtemp()
        try:
            bl = os.path.join(root, "blocklist.txt")
            open(bl, "w", encoding="utf-8").write(BLOCKLIST)
            clean = os.path.join(root, "clean")
            os.makedirs(clean)
            open(os.path.join(clean, "readme.txt"), "w", encoding="utf-8").write("hello\n")
            open(os.path.join(clean, "a.dll"), "wb").write(pe_with_exports(["ggml_pxqn_get_api"]))
            self.assertEqual(self._scan(bl, os.path.join(root, "w"), clean).returncode, 0)
            dirty = os.path.join(root, "dirty")
            os.makedirs(dirty)
            pe = bytearray(pe_with_exports(["ggml_pxqn_get_api"]))
            pe.extend(b"ZZWINBLOCKTOKEN\0")
            open(os.path.join(dirty, "a.dll"), "wb").write(pe)
            self.assertEqual(self._scan(bl, os.path.join(root, "w"), dirty).returncode, 2)
            zpath = os.path.join(root, "bad.zip")
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("readme.txt", "ZZWINBLOCKTOKEN\n")
            self.assertEqual(self._scan(bl, os.path.join(root, "w"), zpath).returncode, 2)
            slip = os.path.join(root, "slip.zip")
            with zipfile.ZipFile(slip, "w") as z:
                z.writestr(zipfile.ZipInfo("../evil.txt"), b"ZZWINBLOCKTOKEN")
            self.assertEqual(self._scan(bl, os.path.join(root, "w"), slip).returncode, 3)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
