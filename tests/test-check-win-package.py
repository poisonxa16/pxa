#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""scripts/check-win-package.py on synthetic packages (tiny PE files, a fake fatbin, a fake PyInstaller payload).
No GPU, no network, no real package needed.

    python3 tests/test-check-win-package.py
"""
import contextlib
import importlib.util
import io
import json
import marshal
import os
import shutil
import struct
import tempfile
import unittest
import zipfile
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = "pxa-engine-win64-cuda12"
BLOCKLIST = """# fixture block list
@category win-block scope=text,binary
ZZWINBLOCKTOKEN
@selftest win-block ZZWINBLOCKTOKEN
"""
HOOKS = ("ggml_backend_cuda_pxqn_gu_counters_reset", "ggml_backend_cuda_pxqn_gu_decode_count",
         "ggml_backend_cuda_pxqn_gu_prefill_count")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REL_ARCHS = (60, 61, 70, 75, 80, 86, 89)   # the release layout: SASS for each, PTX for 89 only


def fatbin(archs, ptx=True):
    """ptx: True = PTX for every arch, False = none, a tuple = only those archs."""
    entries = b""
    for sm in archs:
        for kind in ((1, 2) if ptx is True or (ptx and sm in ptx) else (2,)):
            payload = b"\x7fELF" + bytes(28) if kind == 2 else b".version 7.0\0" + bytes(19)
            hdr = bytearray(64)
            struct.pack_into("<HHII", hdr, 0, kind, 0x101, 64, len(payload))
            struct.pack_into("<I", hdr, 28, sm)
            entries += bytes(hdr) + payload
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(entries)) + entries


def make_pe(exports=(), imports=(), fat=None, extra=b""):
    """A PE32+ x64 image: one .rdata section with the export and import tables, an optional .nv_fatb section."""
    base = 0x1000
    body = bytearray(0x40)                        # export directory at 0, import descriptors after it
    strings = bytearray()

    def s(text):
        off = 0x400 + len(strings)
        strings.extend(text.encode("ascii") + b"\0")
        return off

    exp_rva = imp_rva = 0
    if exports:
        n = len(exports)
        names_off = len(body)
        body.extend(bytes(4 * n))
        name_offs = [s(x) for x in exports]
        for i, o in enumerate(name_offs):
            struct.pack_into("<I", body, names_off + 4 * i, base + o)
        struct.pack_into("<I", body, 24, n)
        struct.pack_into("<I", body, 32, base + names_off)
        exp_rva = base
    if imports:
        imp_off = len(body)
        body.extend(bytes(20 * (len(imports) + 1)))
        for i, dll in enumerate(imports):
            struct.pack_into("<IIIII", body, imp_off + 20 * i, 0, 0, 0, base + s(dll), base + 0x30)
        imp_rva = base + imp_off
    content = bytes(body).ljust(0x400, b"\0") + bytes(strings) + extra
    sections = [(b".rdata", content)]
    if fat is not None:
        sections.append((b".nv_fatb", fat))
    dos = bytearray(0x80)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = bytearray(20)
    struct.pack_into("<HH", coff, 0, 0x8664, len(sections))
    struct.pack_into("<H", coff, 16, 240)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, 108, 16)
    struct.pack_into("<II", opt, 112, exp_rva, 0x40 if exp_rva else 0)
    struct.pack_into("<II", opt, 120, imp_rva, 20 if imp_rva else 0)
    sec_tab = b""
    raw = b""
    rawptr = 0x400
    va = base
    for name, data in sections:
        rawsize = (len(data) + 0x1FF) & ~0x1FF
        sec = bytearray(40)
        sec[0:len(name)] = name
        struct.pack_into("<IIII", sec, 8, len(data), va, rawsize, rawptr)
        sec_tab += bytes(sec)
        raw += data.ljust(rawsize, b"\0")
        rawptr += rawsize
        va += (rawsize + 0xFFF) & ~0xFFF
    head = bytes(dos) + b"PE\0\0" + bytes(coff) + bytes(opt) + sec_tab
    return head.ljust(0x400, b"\0") + raw


def pyinstaller_payload(modules, scripts):
    """A CArchive with zlib'd scripts and one PYZ of zlib'd module blobs, as PyInstaller lays it out."""
    pyz_body = b""
    toc = []
    pos = 12
    for name, text in modules:
        blob = zlib.compress(b"\xe3" + text.encode())
        toc.append((name, (0, pos, len(blob))))
        pyz_body += blob
        pos += len(blob)
    pyz = b"PYZ\0" + b"\xcb\x0d\x0d\x0a" + struct.pack("!i", 12 + len(pyz_body)) + pyz_body + marshal.dumps(toc)
    data = b""
    ents = []
    for name, text in scripts:
        blob = zlib.compress(text.encode())
        ents.append((name, len(data), len(blob), len(text), 1, b"s"))
        data += blob
    ents.append(("PYZ.pyz", len(data), len(pyz), len(pyz), 0, b"z"))
    data += pyz
    toc_bytes = b""
    for name, dp, cl, ul, cf, ty in ents:
        nb = name.encode() + b"\0"
        nb = nb.ljust((len(nb) + 18 + 15) // 16 * 16 - 18, b"\0")
        toc_bytes += struct.pack("!IIIIBc", 18 + len(nb), dp, cl, ul, cf, ty) + nb
    plen = len(data) + len(toc_bytes) + 88
    cookie = struct.pack("!8sIIII64s", b"MEI\x0c\x0b\x0a\x0b\x0e", plen, len(data), len(toc_bytes), 312,
                         b"python312.dll")
    return data + toc_bytes + cookie


class CheckWinPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load(os.path.join(REPO, "scripts", "check-win-package.py"), "check_win")
        g = load(os.path.join(REPO, "scripts", "windows-release-guards.py"), "win_guards_t")
        cls.defs = g.def_export_names(os.path.join(REPO, g.DEF_REL))
        cls.notes = open(os.path.join(REPO, "docs", "windows-release-notes.md"), encoding="utf-8").read()

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="cwp-test-")
        self.bl = os.path.join(self.root, "blocklist.txt")
        open(self.bl, "w", encoding="utf-8").write(BLOCKLIST)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self, name="rel", members=None, drop=(), backslash=False, pxqn_exports=None, control=None,
              notice="Third-party: NVIDIA CUDA runtime (CUDA EULA); Microsoft Visual C++ runtime.\n", sums=True,
              pxqn_extra=b"", terms="PXQN library terms\n"):
        out = os.path.join(self.root, name)
        os.makedirs(out)
        fat3 = fatbin(REL_ARCHS, ptx=(89,))
        files = {
            "README.txt": self.notes.encode(),
            "LICENSE": b"MIT License\n",
            "NOTICE": notice.encode(),
            "LICENSE-ggml-pxqn.txt": b"PXQN library terms\n",
            "bin/llama-server.exe": make_pe(imports=("ggml.dll", "llama.dll", "KERNEL32.dll", "VCRUNTIME140.dll",
                                                     "api-ms-win-crt-runtime-l1-1-0.dll")),
            "bin/ggml.dll": make_pe(exports=("ggml_init",), imports=("cudart64_12.dll", "cublas64_12.dll", "nvcuda.dll"), fat=fat3),
            "bin/llama.dll": make_pe(exports=("llama_init",), imports=("ggml.dll",)),
            "bin/cudart64_12.dll": make_pe(exports=("cudaMalloc",)),
            "bin/cublas64_12.dll": make_pe(exports=("cublasCreate_v2",), imports=("cublasLt64_12.dll",)),
            "bin/cublasLt64_12.dll": make_pe(exports=("cublasLtCreate",)),
            "bin/vcruntime140.dll": make_pe(exports=("memcpy",)),
            "bin/vcruntime140_1.dll": make_pe(exports=("__CxxFrameHandler4",)),
            "bin/msvcp140.dll": make_pe(exports=("x",)),
        }
        files.update(members or {})
        for d in drop:
            files.pop(d, None)
        zpath = os.path.join(out, TOP + ".zip")
        with zipfile.ZipFile(zpath, "w") as z:
            for rel, data in sorted(files.items()):
                arc = TOP + "/" + rel
                z.writestr(arc.replace("/", "\\") if backslash else arc, data)
        exports = list(self.defs) if pxqn_exports is None else pxqn_exports
        open(os.path.join(out, "ggml-pxqn.dll"), "wb").write(make_pe(exports=exports, fat=fat3, extra=pxqn_extra))
        if control is None:
            control = make_pe(imports=("KERNEL32.dll",)) + pyinstaller_payload(
                [("pxa_control.app", "def main(): return 'hello'"), ("json.decoder", "x")],
                [("pxa_control_entry", "import pxa_control.app")])
        open(os.path.join(out, "pxa-control-win64.exe"), "wb").write(control)
        open(os.path.join(out, "RELEASE-NOTES-windows.md"), "w", encoding="utf-8").write(self.notes)
        if terms:
            with open(os.path.join(out, "LICENSE-ggml-pxqn.txt"), "w") as f:
                f.write(terms)
        if sums:
            lines = []
            for fn in ("pxa-control-win64.exe", TOP + ".zip", "ggml-pxqn.dll", "LICENSE-ggml-pxqn.txt", "RELEASE-NOTES-windows.md"):
                if not os.path.isfile(os.path.join(out, fn)):
                    continue
                lines.append("%s  %s\n" % (self.c.sha256_file(os.path.join(out, fn)), fn))
            open(os.path.join(out, "SHA256SUMS"), "w").write("".join(lines))
        return out

    def run_check(self, target, *extra, blocklist=True):
        js = os.path.join(self.root, "out-%d.json" % len(os.listdir(self.root)))
        argv = [target, "--work", os.path.join(self.root, "work"), "--json", js, "--jobs", "1"]
        if blocklist:
            argv += ["--blocklist", self.bl]
        argv += list(extra)
        env_bl = os.environ.pop("PXA_RELEASE_BLOCKLIST", None)
        saved = self.c.DEFAULT_BLOCKLIST
        self.c.DEFAULT_BLOCKLIST = os.path.join(self.root, "no-such-blocklist")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = self.c.main(argv)
        finally:
            self.c.DEFAULT_BLOCKLIST = saved
            if env_bl is not None:
                os.environ["PXA_RELEASE_BLOCKLIST"] = env_bl
        rows = json.load(open(js))["rows"] if os.path.isfile(js) else []
        fails = [(r["check"], r["detail"]) for r in rows if r["status"] == "FAIL"]
        return rc, fails, buf.getvalue()

    def assertFailCheck(self, fails, check, needle=""):
        self.assertTrue(any(c == check and needle in d for c, d in fails), "%s/%s not in %s" % (check, needle, fails))

    def test_good_package_passes(self):
        rc, fails, out = self.run_check(self.build())
        self.assertEqual(fails, [], out)
        self.assertEqual(rc, 0)
        self.assertIn("exactly %d exports" % len(self.defs), out)   # the .def names (13 since the Overdrive table)
        self.assertIn("SASS sm_60,sm_61,sm_70", out)
        self.assertIn("3 entries unpacked (2 ours)", out)

    def test_release_dir_argument(self):
        out = self.build()
        rc, fails, _ = self.run_check(out)
        self.assertEqual((rc, fails), (0, []))

    def test_missing_arch_fails(self):
        out = self.build(members={"bin/ggml.dll": make_pe(exports=("g",), fat=fatbin((60, 70)))})
        rc, fails, _ = self.run_check(out)
        self.assertEqual(rc, 1)
        self.assertFailCheck(fails, "cuda-arch", "sm_61")

    def test_missing_ptx_fails(self):
        out = self.build(members={"bin/ggml.dll": make_pe(exports=("g",), fat=fatbin(REL_ARCHS, ptx=False))})
        rc, fails, _ = self.run_check(out)
        self.assertEqual(rc, 1)
        self.assertFailCheck(fails, "cuda-arch", "missing compute_89")

    def test_nvidia_lib_same_major_rule(self):
        # cuBLAS 12.8's real list: no sm_89 SASS, sm_86 covers it (same major); ours must stay exact
        cub = fatbin((50, 60, 61, 70, 75, 80, 86, 90, 100, 120), ptx=(120,))
        rc, fails, out = self.run_check(self.build(members={"bin/cublas64_12.dll": make_pe(exports=("c",), imports=("cublasLt64_12.dll",), fat=cub)}))
        self.assertEqual((rc, fails), (0, []), out)
        self.assertIn("sm_89 run its same-major older SASS", out)
        rc, fails, _ = self.run_check(self.build(name="r2", members={"bin/cublas64_12.dll": make_pe(exports=("c",), fat=fatbin((80, 86, 90)))}))
        self.assertEqual(rc, 1)
        self.assertFailCheck(fails, "cuda-arch", "missing sm_60,sm_61,sm_70,sm_75")
        rc, fails, _ = self.run_check(self.build(name="r3", members={"bin/ggml.dll": make_pe(exports=("g",), fat=fatbin((60, 61, 70, 75, 80, 86), ptx=(86,)))}))
        self.assertEqual(rc, 1)
        self.assertFailCheck(fails, "cuda-arch", "missing sm_89")

    def test_no_device_code_fails(self):
        out = self.build(members={"bin/ggml.dll": make_pe(exports=("g",))})
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "cuda-arch", "neither")

    def test_export_count_is_exact(self):
        out = self.build(pxqn_exports=list(self.defs) + ["ggml_internal_thing"])
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "pxqn-exports", "%d exports (want %d)" % (len(self.defs) + 1, len(self.defs)))
        hooked = self.build(name="relh", pxqn_exports=list(self.defs) + list(HOOKS))
        rc, fails, _ = self.run_check(hooked)
        self.assertFailCheck(fails, "pxqn-exports", "3 gu_* test hook(s)")
        rc, fails, _ = self.run_check(hooked, "--allow-test-hooks")
        self.assertEqual(fails, [])
        out2 = os.path.join(self.root, "x")
        os.rename(out, out2)
        out = self.build(name="rel2", pxqn_exports=list(self.defs)[:-1])
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "pxqn-exports", "1 .def name(s) missing")

    def test_unresolved_import_fails(self):
        out = self.build(drop=("bin/cublasLt64_12.dll",))
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "pe-imports", "cublasLt64_12.dll")
        self.assertFailCheck(fails, "zip-member", "cublasLt64_12.dll missing")

    def test_layout_licence_and_separators(self):
        out = self.build(backslash=True, drop=("LICENSE",), notice="pxa\n")
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "zip-separators")
        self.assertFailCheck(fails, "zip-member", "LICENSE missing")
        self.assertFailCheck(fails, "notice", "CUDA runtime")
        self.assertFailCheck(fails, "notice", "Visual C++")

    def test_refused_members(self):
        out = self.build(members={"bin/ggml-pxqn.dll": make_pe(exports=list(self.defs)), "bin/x.pdb": b"x",
                                  "bin/nvcuda.dll": make_pe(), "bin/k.pem": b"x"})
        rc, fails, _ = self.run_check(out)
        for needle in ("ggml-pxqn.dll inside", "x.pdb", "nvcuda.dll", "k.pem"):
            self.assertFailCheck(fails, "zip-refused", needle)

    def test_private_strings_fail_in_dll_and_in_control_payload(self):
        lane = ("m" + "2-" + "wk" + "build").encode()
        out = self.build(pxqn_extra=b"\0" + lane + b"\0" + ("/m" + "nt/" + "disk7/x").encode("utf-16-le") + b"\0\0")
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "strings", "lane-name")
        self.assertFailCheck(fails, "strings", "private-path")
        ctl = make_pe() + pyinstaller_payload([("pxa_control.secret", "TOKEN = 'ghp_" + "A1" * 18 + "'")], [])
        rc, fails, _ = self.run_check(self.build(name="rel2", control=ctl))
        self.assertFailCheck(fails, "control-strings", "secret-shape")

    def test_compressed_fatbin_bytes_are_not_strings(self):
        # GGML_CUDA_COMPRESSION_MODE packs device code: a lane-like run inside a packed payload is chance, not a name.
        # The same run inside an uncompressed device ELF is still a FAIL.
        lane = ("m" + "3-" + "l1mb").encode()

        def fat_with(payload):
            hdr = bytearray(64)
            struct.pack_into("<HHII", hdr, 0, 2, 0x101, 64, len(payload))
            struct.pack_into("<I", hdr, 28, 60)
            body = fatbin(REL_ARCHS, ptx=(89,)) + b""
            ent = bytes(hdr) + payload
            return body + struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(ent)) + ent

        packed = b"\x28\xb5\x2f\xfd" + b"\x9c" + lane + b"\x13" + bytes(20)
        out = self.build(members={"bin/ggml.dll": make_pe(exports=("g",), fat=fat_with(packed))})
        rc, fails, o = self.run_check(out)
        self.assertFalse(any(c == "strings" for c, _d in fails), o)
        plain = b"\x7fELF" + bytes(12) + b"\0" + lane + b"\0" + bytes(12)
        out = self.build(name="rel2", members={"bin/ggml.dll": make_pe(exports=("g",), fat=fat_with(plain))})
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "strings", "lane-name")

    def test_scan_gate_runs_on_zip_and_beside_and_fails_closed(self):
        out = self.build(members={"bin/extra.dll": make_pe(extra=b"ZZWINBLOCKTOKEN\0")})
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "scan-gate", "engine-zip")
        ctl = make_pe() + pyinstaller_payload([("pxa_control.ui", "S = 'ZZWINBLOCKTOKEN'")], [])
        rc, fails, _ = self.run_check(self.build(name="rel2", control=ctl))
        self.assertFailCheck(fails, "scan-gate", "beside-the-zip")
        rc, fails, _ = self.run_check(self.build(name="rel3"), blocklist=False)
        self.assertFailCheck(fails, "scan-gate", "no block list")

    def test_sha256sums(self):
        out = self.build(sums=False)
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "sha256sums", "missing")
        out = self.build(name="rel2")
        with open(os.path.join(out, "RELEASE-NOTES-windows.md"), "a") as f:
            f.write("\n")
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "sha256sums", "does not match")

    def test_notes_wording(self):
        out = self.build()
        open(os.path.join(out, "RELEASE-NOTES-windows.md"), "w").write(self.notes.replace("Run anyway", "Run") + "\nBuy it now.\n")
        rc, fails, _ = self.run_check(out)
        self.assertFailCheck(fails, "notes-text", "missing-run-anyway")
        self.assertFailCheck(fails, "notes-text", "store-word")

    def test_pxqn_terms(self):
        rc, fails, _ = self.run_check(self.build(terms=None, drop=("LICENSE-ggml-pxqn.txt",)))
        self.assertFailCheck(fails, "pxqn-terms", "missing")
        self.assertFailCheck(fails, "zip-member", "LICENSE-ggml-pxqn.txt missing")
        rc, fails, _ = self.run_check(self.build(name="rel2", terms="You may buy it.\n"))
        self.assertFailCheck(fails, "pxqn-terms", "store words")

    def test_ctest_log(self):
        out = self.build()
        good = os.path.join(self.root, "ok.log")
        open(good, "w").write("100% tests passed, 0 tests failed out of 87\n")
        bad = os.path.join(self.root, "bad.log")
        open(bad, "w").write("98% tests passed, 2 tests failed out of 87\n")
        rc, fails, _ = self.run_check(out, "--ctest-log", good)
        self.assertEqual(fails, [])
        rc, fails, _ = self.run_check(out, "--ctest-log", bad)
        self.assertFailCheck(fails, "ctest")

    def test_unreadable_zip_is_rc3(self):
        rc, _f, _o = self.run_check(os.path.join(self.root, "nope.zip"))
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
