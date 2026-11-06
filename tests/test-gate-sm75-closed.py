#!/usr/bin/env python3
"""scripts/gate-sm75-check.sh: the closed-library rules (2026-11-04 #61684). libggml-pxqn*.so* / ggml-pxqn*.dll must
carry SASS for the closed list and NO PTX entry; the open libraries keep the PTX rule. Synthetic ELF files, header scan only
(--no-cuobjdump), plus the fail-closed case (no cuobjdump anywhere -> exit 3)."""
import os, struct, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "..", "scripts", "gate-sm75-check.sh")
REL = (60, 61, 70, 75, 80, 86, 89)
CLOSED = REL + (100, 120)


def fatbin(archs, ptx=()):
    ents = b""
    for sm in archs:
        for kind in ((2, 1) if sm in ptx else (2,)):
            payload = b"\x7fELF" + bytes(28) if kind == 2 else b".version 8.7\0" + bytes(19)
            hdr = bytearray(64)
            struct.pack_into("<HHIQ", hdr, 0, kind, 0x101, 64, len(payload))
            struct.pack_into("<I", hdr, 28, sm)
            ents += bytes(hdr) + payload
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(ents)) + ents


def elf(*fbs):
    return b"\x7fELF" + bytes(60) + b"".join(fbs) + bytes(16)


class GateClosed(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def put(self, name, data):
        p = os.path.join(self.d, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def gate(self, *args, env=None):
        p = subprocess.run(["bash", GATE, *args], capture_output=True, text=True, env=env or os.environ.copy(), timeout=120)
        return p.returncode, p.stdout + p.stderr

    def test_closed_sass_only_passes(self):
        self.put("libggml.so", elf(fatbin(REL, ptx=(89,)), fatbin(REL, ptx=(89,))))
        self.put("libggml-pxqn.so", elf(fatbin(CLOSED), fatbin(CLOSED)))
        rc, out = self.gate("--no-cuobjdump", "--require-closed", self.d)
        self.assertEqual(rc, 0, out)
        self.assertIn("1 closed-library binaries SASS only", out)

    def test_closed_with_ptx_fails(self):   # the 4.0.3 layout
        self.put("libggml.so", elf(fatbin(REL, ptx=(89,))))
        self.put("libggml-pxqn.so", elf(fatbin(REL, ptx=(89,)), fatbin(REL, ptx=(89,))))
        rc, out = self.gate("--no-cuobjdump", self.d)
        self.assertEqual(rc, 2, out)
        self.assertIn("PTX-IN-CLOSED-LIB(2 of 2 fatbins; compute_89)", out)
        self.assertIn("MISSING sm_100 sm_120", out)
        rc, out = self.gate("--no-cuobjdump", "--no-closed", self.d)   # an old package checked the old way
        self.assertEqual(rc, 0, out)

    def test_one_ptx_entry_fails(self):
        self.put("libggml-pxqn.so", elf(fatbin(CLOSED), fatbin(CLOSED, ptx=(120,))))
        rc, out = self.gate("--no-cuobjdump", self.d)
        self.assertEqual(rc, 2, out)
        self.assertIn("PTX-IN-CLOSED-LIB(1 of 2 fatbins; compute_120)", out)

    def test_closed_thin_fatbin_fails(self):
        self.put("libggml-pxqn.so", elf(fatbin(CLOSED), fatbin(REL)))
        rc, out = self.gate("--no-cuobjdump", self.d)
        self.assertEqual(rc, 2, out)
        self.assertIn("fatbins-missing sm_100x1,sm_120x1", out)
        rc, out = self.gate("--no-cuobjdump", "--closed-archs", " ".join(map(str, REL)), self.d)
        self.assertEqual(rc, 0, out)

    def test_open_lib_rule_unchanged(self):
        self.put("libggml.so", elf(fatbin(REL)))   # open library without PTX 89: still a FAIL
        rc, out = self.gate("--no-cuobjdump", self.d)
        self.assertEqual(rc, 2, out)
        self.assertIn("MISSING compute_89", out)

    def test_require_closed(self):
        self.put("libggml.so", elf(fatbin(REL, ptx=(89,))))
        rc, out = self.gate("--no-cuobjdump", "--require-closed", self.d)
        self.assertEqual(rc, 2, out)
        self.assertIn("no closed-library binary", out)

    def test_dll_name_is_closed(self):
        self.put("ggml-pxqn.dll", b"MZ" + bytes(62) + fatbin(REL, ptx=(89,)))
        rc, out = self.gate("--no-cuobjdump", self.d)
        self.assertEqual(rc, 2, out)
        self.assertIn("PTX-IN-CLOSED-LIB", out)

    def test_fails_closed_without_cuobjdump(self):
        self.put("libggml-pxqn.so", elf(fatbin(CLOSED)))
        env = {"PATH": "/usr/bin:/bin", "PXA_CUOBJDUMP_IMAGE": "pxa-no-such-image:none", "HOME": self.d, "TMPDIR": self.d}
        if os.path.exists("/usr/bin/cuobjdump") or os.path.exists("/bin/cuobjdump"):
            self.skipTest("a cuobjdump on /usr/bin")
        rc, out = self.gate(self.d, env=env)
        self.assertEqual(rc, 3, out)
        self.assertIn("fails closed", out)


if __name__ == "__main__":
    unittest.main()
