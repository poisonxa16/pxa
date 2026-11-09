"""CPU-only regression for the opt-in dual-Volta launch profile."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / "tools/pxa-dual-volta.sh"


@unittest.skipUnless(os.name == "posix", "POSIX shell wrapper")
class LaunchProfileTest(unittest.TestCase):
    def test_default_and_explicit_values_preserve_arguments(self):
        code = "import os,sys,json;print(json.dumps([os.environ['CUDA_SCALE_LAUNCH_QUEUES'],sys.argv[1:]]))"
        for value in (None, "", "1x", "2x", "4x"):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("CUDA_SCALE_LAUNCH_QUEUES", None)
                if value is not None:
                    env["CUDA_SCALE_LAUNCH_QUEUES"] = value
                reply = subprocess.run(
                    ["sh", str(WRAPPER), sys.executable, "-c", code,
                     "two words", "literal;$(echo no)"],
                    env=env, capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(reply.stdout), [
                    "4x" if value is None else value,
                    ["two words", "literal;$(echo no)"]])

    def test_status_and_usage(self):
        reply = subprocess.run(["sh", str(WRAPPER), sys.executable, "-c",
                                "raise SystemExit(23)"], capture_output=True)
        self.assertEqual(reply.returncode, 23)
        reply = subprocess.run(["sh", str(WRAPPER)], capture_output=True)
        self.assertEqual(reply.returncode, 64)
        self.assertIn(b"Usage:", reply.stderr)

    def test_exec_preserves_pid_and_signal_delivery(self):
        code = "import os,time;print(os.getpid(),flush=True);time.sleep(30)"
        process = subprocess.Popen(["sh", str(WRAPPER), sys.executable, "-c", code],
                                   stdout=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0])
            self.assertEqual(int(process.stdout.readline()), process.pid)
            process.terminate()
            self.assertEqual(process.wait(timeout=5), -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
