#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""scripts/release-scan.py check 4: compressed onefile payloads (PyInstaller, Nuitka zstd, appended zip, 7z,
nested zips inside a payload) and Mach-O strings. A planted blocked word inside a compressed program must FAIL
(exit 2), the same program without it must PASS (exit 0), a payload that cannot be unpacked must stop the gate
(exit 3). No GPU, no network.

    python3 tests/test-release-scan-payloads.py
"""
import importlib.util
import bz2
import gzip
import io
import lzma
import os
import py_compile
import shutil
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import unittest
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(REPO, "scripts", "release-scan.py")
WORD = "ZZPAYLOADBLOCK"
BLOCKLIST = """# fixture block list
@category planted scope=text,binary
%s
@files stale-file
*.pyc
@selftest planted %s
""" % (WORD, WORD)


def load():
    spec = importlib.util.spec_from_file_location("rscan_payload_t", SCAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pe_stub():
    pe = bytearray(0x200)
    pe[0:2] = b"MZ"
    struct.pack_into("<I", pe, 0x3C, 0x80)
    pe[0x80:0x84] = b"PE\0\0"
    return bytes(pe)


ELF_STUB = b"\x7fELF\x02\x01\x01" + b"\0" * 57
MACHO_STUB = b"\xcf\xfa\xed\xfe" + b"\0" * 60


def zipped(name, data):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, data)
    return buf.getvalue()


def in_embed(name, data):
    """name at payload depth 3, where python.org's embeddable Python keeps its stdlib inside the Control exe
    (exe -> python-embed.zip -> python313.zip -> name)"""
    return ELF_STUB + zipped("python-embed.zip", zipped("python313.zip", zipped(name, data)))


PYC_HEAD = b"\xf3\x0d\x0d\x0a" + b"\0" * 12          # 3.13's pyc magic, flags, mtime, size
CONST = b"s\x04\0\0\0"                                 # marshal: a 4-byte bytes constant follows (0xf3 with FLAG_REF)


def fake_pyc(body=b"", rec=None):
    """Compiled-Python bytes shaped like zipimport.pyc: zip signatures inside marshal bytes constants, read as an end
    record over a central directory. The default record is cut short, so is_zipfile() says yes (a check of the first
    directory record passes too) and ZipFile() refuses it; it ends in 0xf3 + 4 bytes, the FLAG_REF form of the
    constant prefix, in front of the end signature."""
    rec = b"PK\x01\x02" + bytes(41) + b"\xf3\x04\0\0\0" if rec is None else rec
    end = b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, 1, 1, len(rec), 0, 0)
    return PYC_HEAD + body + b"c" * 700 + CONST + rec + end + b"z" * 400


class PayloadScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = load()

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rs-payload-")
        self.bl = os.path.join(self.root, "blocklist.txt")
        with open(self.bl, "w", encoding="utf-8") as f:
            f.write(BLOCKLIST)
        self.pkg = os.path.join(self.root, "pkg")
        os.makedirs(self.pkg)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def put(self, name, data):
        path = os.path.join(self.pkg, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def scan(self, target=None):
        p = subprocess.run([sys.executable, "-I", SCAN, "--blocklist", self.bl, "--work", os.path.join(self.root, "w"),
                            "--jobs", "1", "--show", "50", target or self.pkg],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout + p.stderr

    def test_pyinstaller_onefile(self):
        self.put("app.exe", self.rs.make_pyinstaller_sample([("pkg.ui", "title = 'hello'")], [("entry", "x=1")], stub=pe_stub()))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.assertIn("1 onefile payload(s) unpacked", out)
        self.put("app.exe", self.rs.make_pyinstaller_sample([("pkg.ui", "title = '%s'" % WORD)], stub=pe_stub()))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("app.exe!/PYZ/pkg/ui.pyc", out)
        # the planted word is invisible to a plain strings pass: it only shows once the payload is unpacked
        with open(os.path.join(self.pkg, "app.exe"), "rb") as f:
            self.assertNotIn(WORD.encode(), f.read())

    def test_pyz_member_shaped_like_zipimport_is_not_a_payload(self):
        """PyInstaller keeps zipimport in its PYZ as bare marshal data (no pyc header): with the zip signatures inside its
        bytes constants it is compiled Python, scanned as bytes and never unpacked as a zip (10-26: a onefile
        build stopped the gate with exit 3)"""
        rec = b"PK\x01\x02" + bytes(41) + CONST
        body = b"c" * 700 + CONST + rec + b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, 1, 1, len(rec), 0, 0) + b"z" * 400
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(b"\xe3" + body)))
        self.put("app.exe", self.rs.make_pyinstaller_sample([("zipimport", body.decode("latin-1"))], stub=pe_stub()))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        # still scanned: a planted word in that module fails the gate by name
        planted = b"\0" + WORD.encode() + b"\0" + body
        self.put("app.exe", self.rs.make_pyinstaller_sample([("zipimport", planted.decode("latin-1"))], stub=pe_stub()))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("app.exe!/PYZ/zipimport.pyc", out)

    def test_pyinstaller_script_entry_and_nested_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("lib/mod.txt", "a %s b\n" % WORD)
        nested = buf.getvalue()
        sample = bytearray(self.rs.make_pyinstaller_sample([("pkg.m", "x")], [("entry", "print('%s')" % WORD)], stub=ELF_STUB))
        self.put("app", bytes(sample))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("app!/entry", out)
        # a zip inside a payload is opened (depth 2): place one as a data entry via an appended-zip program
        self.put("app", ELF_STUB + nested)
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("app!/lib/mod.txt", out)

    def test_payload_past_the_depth_blocks(self):
        """security review P10 L3: a payload nested past PAYLOAD_DEPTH is a blocking hit, never skipped unscanned"""
        def zipped(name, data):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(name, data)
            return buf.getvalue()
        z = zipped("x.txt", "clean\n")
        for lvl in range(self.rs.PAYLOAD_DEPTH - 1):
            z = zipped("l%d.bin" % lvl, z)
        self.put("app", ELF_STUB + z)                       # opened down to the depth: clean
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.put("app", ELF_STUB + zipped("top.bin", z))    # one level more: the innermost zip is past the depth
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("payload-too-deep", out)

    def test_pyc_holding_a_zip_marker_is_not_a_payload(self):
        """the stdlib's zipimport.pyc keeps the zip signatures as bytes constants, and python.org's embeddable Python
        carries it at the depth (Control exe -> python-embed.zip -> python313.zip): inside a payload, compiled Python whose
        every zip signature sits in a bytes constant and that does not open as a zip is scanned as bytes, never unpacked
        or blocked as too deep; anything else that looks like a zip stays a payload (security review 10-24)"""
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(fake_pyc())))
        self.assertRaises(zipfile.BadZipFile, zipfile.ZipFile, io.BytesIO(fake_pyc()))
        self.put("app", in_embed("zipimport.pyc", fake_pyc()))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        # still scanned: a planted word in that compiled file fails the gate by name
        self.put("app", in_embed("zipimport.pyc", fake_pyc(b"\0" + WORD.encode() + b"\0")))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("python313.zip!/zipimport.pyc", out)
        self.assertNotIn("payload-too-deep", out)
        # each of these stays a payload, so past the depth it blocks
        uname = b"\xff" + CONST                                 # a whole directory record whose name is bad UTF-8
        bad_name = struct.pack("<4s4B4HL2L5H2L", b"PK\x01\x02", 20, 0, 20, 0, 0x800, 0, 0, 0, 0, 0, 0, len(uname),
                               0, 0, 0, 0, 0, 0) + uname
        damaged = bytearray(zipped("x.txt", "x %s y\n" % WORD))  # a real zip whose directory size is off by 4
        e = damaged.rfind(b"PK\x05\x06")
        struct.pack_into("<I", damaged, e + 12, struct.unpack_from("<I", damaged, e + 12)[0] + 4)
        cases = [
            ("a .pyc that opens as a zip", "zipimport.pyc", PYC_HEAD + b"c" * 700 + zipped("x.txt", "clean\n"), None),
            ("a damaged zip whose signatures are not constants", "mod.pyc", PYC_HEAD + bytes(damaged), zipfile.BadZipFile),
            ("an open error other than not-a-zip", "zipimport.pyc", fake_pyc(rec=bad_name), UnicodeDecodeError),
            ("no pyc magic", "zipimport.pyc", b"XXXX" + fake_pyc()[4:], zipfile.BadZipFile),
        ]
        for what, fname, data, refusal in cases:
            with self.subTest(what):
                if refusal:
                    self.assertRaises(refusal, zipfile.ZipFile, io.BytesIO(data))
                self.put("app", in_embed(fname, data))
                rc, out = self.scan()
                self.assertEqual(rc, 2, out)
                self.assertIn("payload-too-deep", out)

    def test_pyc_whose_zip_view_is_empty_is_not_a_payload(self):
        """Python 3.10's zipfile opens its own zipimport.pyc as a zip with no members where 3.13's refuses it, so the
        verdict must not hang on the Python that runs the gate: an empty zip view holds nothing to unpack, and the bytes
        still get the strings pass; a signature outside a constant keeps it a payload, empty or not"""
        def empty_view(body=b"", const=CONST):
            head = PYC_HEAD + body + b"c" * 700 + const
            return head + b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, 0, 0, 0, len(head), 0)
        for const in (CONST, b""):
            with zipfile.ZipFile(io.BytesIO(empty_view(const=const))) as z:
                self.assertEqual(z.infolist(), [])
        self.put("app", in_embed("zipimport.pyc", empty_view(const=b"")))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("payload-too-deep", out)
        self.put("app", in_embed("zipimport.pyc", empty_view()))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.put("app", in_embed("zipimport.pyc", empty_view(b"\0" + WORD.encode() + b"\0")))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("python313.zip!/zipimport.pyc", out)
        self.assertNotIn("payload-too-deep", out)

    def test_pyc_exemption_only_inside_payloads(self):
        """security review 10-24: outside a payload a file like that gets no strings pass, so it stays a payload there
        and the gate stops when it cannot be unpacked"""
        self.put("helper.pyo", fake_pyc(b"\0" + WORD.encode() + b"\0"))
        rc, out = self.scan()
        self.assertEqual(rc, 3, out)
        self.assertIn("cannot unpack", out)

    def test_stdlib_zipimport_pyc_at_the_depth_passes(self):
        """the real trigger: the stdlib's zipimport, compiled by this Python, where the Control exe carries it"""
        src = os.path.join(sysconfig.get_path("stdlib"), "zipimport.py")
        if not os.path.exists(src):
            self.skipTest("no stdlib source to compile")
        pyc = os.path.join(self.root, "zipimport.pyc")
        py_compile.compile(src, cfile=pyc, doraise=True)
        with open(pyc, "rb") as f:
            self.put("app", in_embed("zipimport.pyc", f.read()))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)

    def test_nuitka_onefile_posix_and_macho(self):
        files = [("pxa-control.bin", MACHO_STUB + b"\0const\0"), ("ui/index.html", b"<p>hello</p>\n")]
        sample = self.rs.make_nuitka_sample(files, stub=MACHO_STUB)
        if sample is None:
            self.skipTest("zstd tool not installed")
        self.put("pxa-control", sample)
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.assertIn("1 onefile payload(s) unpacked, 2 file(s) inside", out)
        files[1] = ("ui/index.html", ("<p>%s</p>\n" % WORD).encode())
        self.put("pxa-control", self.rs.make_nuitka_sample(files, stub=ELF_STUB))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("pxa-control!/ui/index.html", out)
        # inside the compiled program blob (binary entry): the strings pass applies there too
        files = [("pxa-control.bin", MACHO_STUB + b"\0" + WORD.encode() + b"\0")]
        self.put("pxa-control", self.rs.make_nuitka_sample(files, stub=pe_stub()))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("pxa-control!/pxa-control.bin", out)

    def test_corrupt_payload_fails_closed(self):
        if shutil.which("zstd") is None:
            self.skipTest("zstd tool not installed")
        self.put("bad", ELF_STUB + b"KAY" + self.rs.ZSTD_MAGIC + b"\x00garbage-not-a-frame" * 4)
        rc, out = self.scan()
        self.assertEqual(rc, 3, out)
        self.assertIn("fails closed", out)

    def _fatbin_pe(self, payload_plain=None, raw=None, tail=b""):
        """A PE with one CUDA fatbin entry whose payload is zstd(payload_plain) (or raw bytes) + tail inside the padding."""
        if raw is None:
            raw = subprocess.run(["zstd", "-q", "-c", "-19", "-"], input=payload_plain, stdout=subprocess.PIPE, check=True).stdout
        body = raw + tail
        body += b"\0" * (-len(body) % 8)
        hdr = bytearray(64)
        struct.pack_into("<HHII", hdr, 0, 2, 0x101, 64, len(body))
        struct.pack_into("<I", hdr, 28, 60)
        ent = bytes(hdr) + body
        return pe_stub() + struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(ent)) + ent

    def test_compressed_cuda_fatbin_is_decoded_and_its_noise_skipped(self):
        if shutil.which("zstd") is None:
            self.skipTest("zstd tool not installed")
        # a blocked word inside the compressed device code is found once decoded (invisible to a plain strings pass)
        # (the word's head appears once before, so the compressor stores the second copy as a back-reference + "LOCK")
        dev = ELF_STUB + b"\0" + WORD.encode()[:-4] + b"\0" + bytes(40) + b"\0kernel_" + WORD.encode() + b"_name\0" + bytes(64)
        data = self._fatbin_pe(dev)
        self.assertNotIn(WORD.encode(), data)
        self.put("ggml.dll", data)
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        # the compressed frame itself is noise and is not string-scanned: a clean device ELF with zero padding passes
        self.put("ggml.dll", self._fatbin_pe(ELF_STUB + b"\0clean_kernel\0" + bytes(64), tail=bytes(24)))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        # bytes of a compressed entry OUTSIDE its zstd frames are real file bytes: plain text there is found (I3)
        self.put("ggml.dll", self._fatbin_pe(ELF_STUB + b"\0clean_kernel\0" + bytes(64), tail=b"\0" + WORD.encode() + b"\0"))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        # a compressed entry that does not decode stops the gate
        self.put("ggml.dll", self._fatbin_pe(raw=self.rs.ZSTD_MAGIC + b"\x00garbage-not-a-frame" * 4))
        rc, out = self.scan()
        self.assertEqual(rc, 3, out)
        self.assertIn("fails closed", out)

    def _zst(self, plain):
        return subprocess.run(["zstd", "-q", "-c", "-19", "-"], input=plain, stdout=subprocess.PIPE, check=True).stdout

    def test_every_zstd_frame_of_a_fatbin_entry_is_decoded(self):
        """I3: a blocked word in the SECOND frame of an entry is found, whether the frames are adjacent or
        separated by bytes that are not a frame (the old decoder stopped at the first non-frame byte and trusted the rest)."""
        if shutil.which("zstd") is None:
            self.skipTest("zstd tool not installed")
        clean = self._zst(ELF_STUB + b"\0clean_kernel\0" + bytes(64))
        dirty = self._zst(ELF_STUB + b"\0" + WORD.encode()[:-4] + b"\0" + bytes(40) + b"\0kernel_" + WORD.encode() + b"_name\0" + bytes(64))
        self.assertNotIn(WORD.encode(), dirty)
        for gap in (b"", bytes(16), b"\x01\x02\x03junk-not-a-frame\x04" * 3):
            self.put("ggml.dll", self._fatbin_pe(raw=clean + gap + dirty))
            rc, out = self.scan()
            self.assertEqual(rc, 2, "gap=%r: %s" % (gap[:8], out))
        # two clean frames with a zero gap pass
        self.put("ggml.dll", self._fatbin_pe(raw=clean + bytes(16) + clean))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        # a second frame that is cut short fails closed
        self.put("ggml.dll", self._fatbin_pe(raw=clean + bytes(8) + dirty[:len(dirty) // 2]))
        rc, out = self.scan()
        self.assertEqual(rc, 3, out)
        self.assertIn("fails closed", out)

    def test_zstd_frame_len_matches_the_encoder(self):
        if shutil.which("zstd") is None:
            self.skipTest("zstd tool not installed")
        for plain in (b"", b"a", bytes(5000), os.urandom(3000), b"kernel_name\0" * 4000):
            for lvl in ("-1", "-19"):
                z = subprocess.run(["zstd", "-q", "-c", lvl, "-"], input=plain, stdout=subprocess.PIPE, check=True).stdout
                self.assertEqual(self.rs.zstd_frame_len(z + b"\0" * 9, 0, len(z) + 9), len(z))
                self.assertIsNone(self.rs.zstd_frame_len(z[:-1], 0, len(z) - 1))

    def test_appended_zip_program(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("data/clean.txt", "nothing here\n")
        self.put("sfx.exe", pe_stub() + buf.getvalue())
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("data/notes.txt", "x %s y\n" % WORD)
        self.put("sfx.exe", pe_stub() + buf.getvalue())
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("sfx.exe!/data/notes.txt", out)

    def test_7z_self_extractor(self):
        tool = shutil.which("bsdtar")
        if tool is None:
            self.skipTest("bsdtar not installed")
        src = os.path.join(self.root, "src")
        os.makedirs(os.path.join(src, "d"))
        arc = os.path.join(self.root, "a.7z")
        for text, want in (("clean\n", 0), ("x %s y\n" % WORD, 2)):
            with open(os.path.join(src, "d", "f.txt"), "w") as f:
                f.write(text + "padding " * 40)
            if os.path.exists(arc):
                os.remove(arc)
            subprocess.run([tool, "--format", "7zip", "-cf", arc, "-C", src, "d"], check=True)
            with open(arc, "rb") as f:
                seven = f.read()
            self.put("setup.exe", pe_stub() + seven)
            rc, out = self.scan()
            self.assertEqual(rc, want, out)
            if want:
                self.assertIn("setup.exe!/d/f.txt", out)

    def test_macho_strings_pass(self):
        self.put("bin/tool", MACHO_STUB + b"\0" + WORD.encode() + b"\0")
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("1 Mach-O files", out)

    def test_pyc_file_rule_only_outside_payloads(self):
        self.put("app.exe", self.rs.make_pyinstaller_sample([("pkg.clean", "x")], stub=pe_stub()))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)                     # app.exe!/PYZ/pkg/clean.pyc is not a stray file
        self.put("stray.pyc", b"\xe3\0\0")
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("stale-file", out)

    def test_zip_input_with_onefile_inside(self):
        exe = self.rs.make_pyinstaller_sample([("pkg.ui", "s = '%s'" % WORD)], stub=pe_stub())
        zpath = os.path.join(self.root, "pkg.zip")
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("top/app.exe", exe)
        rc, out = self.scan(zpath)
        self.assertEqual(rc, 2, out)
        self.assertIn("top/app.exe!/PYZ/pkg/ui.pyc", out)


    # scan-gate-data-renamed-zips (review notes on the scan gate): data files and renamed zips at the package's top level
    def test_top_level_data_file_gets_the_strings_pass(self):
        self.put("share/table.bin", b"\0\x01\x02 clean data \0" * 8)
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.put("share/table.bin", b"\0\x01\x02 %s \0" % WORD.encode() * 8)
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("share/table.bin", out)

    def test_zip_under_a_non_archive_name_is_unpacked(self):
        self.put("share/assets.dat", zipped("inner/notes.txt", "nothing here\n" * 20))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.assertIn("1 onefile payload(s) unpacked", out)
        blob = zipped("inner/notes.txt", ("x %s y\n" % WORD) * 20)
        self.assertNotIn(WORD.encode(), blob)            # deflated: only an unpack finds it
        self.put("share/assets.dat", blob)
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("share/assets.dat!/inner/notes.txt", out)

    def test_allowed_archive_is_still_scanned_inside(self):
        with open(self.bl, "a", encoding="utf-8") as f:
            f.write("@archive-allow tools/wrapper.jar\n")
        self.put("tools/wrapper.jar", zipped("META-INF/x.txt", "clean\n" * 20))
        rc, out = self.scan()
        self.assertEqual(rc, 0, out)
        self.put("tools/wrapper.jar", zipped("META-INF/x.txt", ("x %s y\n" % WORD) * 20))
        rc, out = self.scan()
        self.assertEqual(rc, 2, out)
        self.assertIn("tools/wrapper.jar!/META-INF/x.txt", out)

    def test_bare_program_files_as_input(self):
        for name, stub in (("app.exe", pe_stub()), ("lib.dll", pe_stub()), ("tool", ELF_STUB)):
            path = os.path.join(self.root, name)
            with open(path, "wb") as f:
                f.write(stub + b"\0 clean \0")
            rc, out = self.scan(path)
            self.assertEqual(rc, 0, out)
            with open(path, "wb") as f:
                f.write(stub + b"\0" + WORD.encode() + b"\0")
            rc, out = self.scan(path)
            self.assertEqual(rc, 2, out)
            self.assertIn(name, out)

    def test_corrupt_tarball_input_still_fails_closed(self):
        path = os.path.join(self.root, "pkg.tar.gz")
        with open(path, "wb") as f:
            f.write(b"\x1f\x8b\x08\0 not really gzip" + b"\0" * 64)
        rc, out = self.scan(path)
        self.assertEqual(rc, 3, out)

    # review of 0ea54bf3f3: tar sniffing, compressed data by magic, unreadable zips, big data files
    def test_bare_file_starting_with_zeros_is_scanned(self):
        path = os.path.join(self.root, "zeros.bin")
        with open(path, "wb") as f:
            f.write(b"\0" * 1024 + (" %s " % WORD).encode() * 20)
        rc, out = self.scan(path)
        self.assertEqual(rc, 2, out)

    def tar_with(self, name, tail=b""):
        path = os.path.join(self.root, name)
        with tarfile.open(path, "w") as tf:
            data = b"nothing here\n"
            ti = tarfile.TarInfo("top/clean.txt")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        with open(path, "ab") as f:
            f.write(tail)
        return path

    def test_tar_input_with_bytes_after_its_end_fails_closed(self):
        rc, out = self.scan(self.tar_with("pkg.tar"))
        self.assertEqual(rc, 0, out)
        rc, out = self.scan(self.tar_with("pkg2.tar", (" %s " % WORD).encode() * 20))
        self.assertEqual(rc, 3, out)

    def test_tar_named_input_with_no_files_fails_closed(self):
        path = os.path.join(self.root, "empty.tar")
        with open(path, "wb") as f:
            f.write(b"\0" * 10240)
        rc, out = self.scan(path)
        self.assertEqual(rc, 3, out)

    def test_compressed_data_under_a_non_archive_name_is_unpacked(self):
        for pack in (gzip.compress, lzma.compress, bz2.compress):
            self.put("share/blob.dat", pack(b"nothing here\n" * 40))
            rc, out = self.scan()
            self.assertEqual(rc, 0, out)
            blob = pack(("x %s y\n" % WORD).encode() * 40)
            self.assertNotIn(WORD.encode(), blob)
            self.put("share/blob.dat", blob)
            rc, out = self.scan()
            self.assertEqual(rc, 2, (pack, out))
            self.assertIn("share/blob.dat!/", out)

    def test_zip_with_an_encrypted_member_fails_closed(self):
        blob = bytearray(zipped("inner/notes.txt", "nothing here\n" * 20))
        blob[6] |= 1                                       # local header: general purpose flag bit 0 = encrypted
        c = blob.find(b"PK\x01\x02")
        blob[c + 8] |= 1                                   # central directory: the same flag
        self.put("share/assets.dat", bytes(blob))
        rc, out = self.scan()
        self.assertEqual(rc, 3, out)

    def test_big_data_files_are_read_in_chunks(self):
        path = os.path.join(self.root, "big.bin")
        with open(path, "wb") as f:
            f.write(b"\0" * 13 + WORD.encode() + b"\0" + b"abc" + b"\0" * 7 + b"tail-run")
        got = self.rs.chunked_strings(path, chunk=16).split("\n")
        self.assertEqual(got, [WORD, "tail-run"])

if __name__ == "__main__":
    unittest.main()
