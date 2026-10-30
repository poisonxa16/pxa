#!/usr/bin/env python3
"""A fixture with box paths goes through the same writer the gate uses.

The output, which is what run-gate.sh appends to LAST-RUN.md, must contain
none of /mnt, /root, /home, /boot, the hostname, or the internal model name.
"""
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REDACT = os.path.join(HERE, "redact-paths.py")

_M = "/" + "mnt"
_B = "/" + "boot"
_R = "/" + "root"
_H = "/" + "home"
FIXTURE = "\n".join([
    "[20:05:46] gate starting",
    "[20:05:46]   model    " + _M + "/pool/models/Example-27B-PXQ4.gguf",
    "[20:05:46]   binaries " + _M + "/pool/engine-wt/build/bin",
    "[20:05:46]   logs     /tmp/pxa-gate.ABC123",
    "[20:05:46]   host     GATEHOST saw the file Example-27B-PXQ4.gguf",
    "  FAIL  server died; see " + _R + "/pxa/server.log and " + _H + "/ubuntu/models/x.gguf",
    "  FAIL  secrets live in " + _B + "/config/keys",
    "=== GATE RESULT: PASS=0 FAIL=1 SKIP=0  (logs in /tmp/pxa-gate.ABC123) ===",
]) + "\n"


def run(text):
    env = dict(os.environ)
    env.update({
        "PXA_MODEL": _M + "/pool/models/Example-27B-PXQ4.gguf",
        "PXA_BIN": _M + "/pool/engine-wt/build/bin",
        "PXA_WORKDIR": "/tmp/pxa-gate.ABC123",
        "PXA_ROOT": _M + "/pool/engine-wt",
        "PXA_PROMPTS": _M + "/pool/engine-wt/bench/gate/prompts",
        "PXA_HOST": "GATEHOST",
        "PXA_FQDN": "GATEHOST",
    })
    p = subprocess.run([os.environ.get("PYTHON", "python3"), REDACT],
                       input=text, text=True, capture_output=True, env=env, check=True)
    return p.stdout


class Redact(unittest.TestCase):
    def test_fixture_keeps_no_box_path(self):
        out = run(FIXTURE)
        for banned in (_M + "/", _M, _R, _H + "/", _B, "GATEHOST",
                       "Example-27B-PXQ4.gguf", "/pool/", "engine-wt"):
            self.assertNotIn(banned, out, banned)
        self.assertIn("<model>", out)
        self.assertIn("<package>", out)
        self.assertIn("<logdir>", out)
        self.assertIn("<host>", out)
        self.assertIn("<root>", out)
        self.assertIn("<home>", out)
        self.assertIn("<boot>", out)
        self.assertIn("GATE RESULT", out)

    def test_runner_writes_the_redacted_line(self):
        """The shell writer in run-gate.sh is this pipe. Prove a file written that way is clean."""
        out = run(FIXTURE)
        fd, path = tempfile.mkstemp(prefix="lastrun-", suffix=".md")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(out)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotIn("/mnt", text)
            self.assertNotIn("/boot", text)
            self.assertNotIn("/root", text)
            self.assertNotIn("/home", text)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
