"""boxenv.py -- load pxa/pxq4/box.env (KEY=VALUE lines, gitignored) into os.environ if present.

Import it first in a harness that runs on one box; the tracked code then names no machine.
    python3 -m gguf_to_vllm.boxenv     # prints the resolved variables
"""
import os
import sys


def load(start: str | None = None) -> str | None:
    d = os.path.abspath(start or os.path.dirname(__file__))
    for _ in range(8):
        p = os.path.join(d, "box.env")
        if os.path.isfile(p):
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
            return p
        d = os.path.dirname(d)
    return None


load()

if __name__ == "__main__":
    p = load()
    print(f"box.env: {p or 'not found'}")
    for k in sorted(k for k in os.environ if k.startswith("PXA_")):
        print(f"{k}={os.environ[k]}")
    sys.exit(0)
