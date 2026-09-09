#!/usr/bin/env python3
"""tests/test-pxa-launch-env.py - the launcher must never hand a child a card INDEX
without the ordering that gives that index its meaning.

WHY THIS TEST EXISTS
  CUDA_VISIBLE_DEVICES=0 is not a card selection on its own. CUDA's default
  CUDA_DEVICE_ORDER is FASTEST_FIRST, which sorts the cards by compute capability,
  while nvidia-smi numbers them by PCI bus id - and tools/pxa-launch.py takes its
  --gpus numbers, and everything it prints about a card, from nvidia-smi. On a box
  holding a P100 and a V100 those two numberings disagree, so a launcher that sets
  only CUDA_VISIBLE_DEVICES starts the model on a card the user did not pick and
  says nothing about it. The fix is that the order travels with the index, every
  time, everywhere - which is what this file checks.

  No GPU, no driver and no nvidia-smi are needed: everything here is environment
  composition and source structure.

RUN
  python3 tests/test-pxa-launch-env.py
  (exit 0 = pass; every check prints its own line)
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(os.path.dirname(HERE), "tools", "pxa-launch.py")

_spec = importlib.util.spec_from_file_location("pxa_launch", LAUNCHER)
if _spec is None or _spec.loader is None:
    print(f"FAIL: cannot load {LAUNCHER}")
    sys.exit(1)
pxa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pxa)          # module level is data only; nothing is started

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def with_order(value, fn):
    """Run fn() with CUDA_DEVICE_ORDER set to `value` (None = unset), then restore."""
    saved = os.environ.pop("CUDA_DEVICE_ORDER", None)
    try:
        if value is not None:
            os.environ["CUDA_DEVICE_ORDER"] = value
        return fn()
    finally:
        os.environ.pop("CUDA_DEVICE_ORDER", None)
        if saved is not None:
            os.environ["CUDA_DEVICE_ORDER"] = saved


print("=== pxa-launch device scoping (I-13) ===")

# 1. The whole point: an index selection carries the order that defines it.
env, note = with_order(None, lambda: pxa.device_env("2,4"))
check("an index selection sets CUDA_VISIBLE_DEVICES",
      env.get("CUDA_VISIBLE_DEVICES") == "2,4", str(env))
check("an index selection also sets CUDA_DEVICE_ORDER=PCI_BUS_ID",
      env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", str(env))
check("nothing is said about an order the user never set", note is None, str(note))
check("the container variable is set alongside, unchanged",
      env.get("NVIDIA_VISIBLE_DEVICES") == "2,4", str(env))

# 2. A single card is the case the first-time reader hits, and the one that bit:
#    'card 0' from nvidia-smi must not become CUDA's fastest card.
env1, _ = with_order(None, lambda: pxa.device_env("0"))
check("the single-card case pins the order too",
      env1.get("CUDA_VISIBLE_DEVICES") == "0"
      and env1.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", str(env1))

# 3. An order the caller exported is RESPECTED, not overwritten - and said out loud.
env2, note2 = with_order("FASTEST_FIRST", lambda: pxa.device_env("2,4"))
check("an exported CUDA_DEVICE_ORDER is kept, not overridden",
      env2.get("CUDA_DEVICE_ORDER") == "FASTEST_FIRST", str(env2))
check("and keeping it produces exactly one note to print",
      isinstance(note2, str) and "FASTEST_FIRST" in note2, str(note2))
env3, note3 = with_order("PCI_BUS_ID", lambda: pxa.device_env("2,4"))
check("an exported PCI_BUS_ID is kept and acknowledged",
      env3.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and isinstance(note3, str), str(note3))

# 4. A UUID selection names the card outright, so no ordering applies to it and the
#    launcher must not imply one.
uenv, unote = with_order(None, lambda: pxa.device_env(
    "GPU-aad5ef40-9b80-8fd0-4391-dfe595f42640"))
check("a UUID selection needs no order and gets none",
      "CUDA_DEVICE_ORDER" not in uenv and unote is None, str((uenv, unote)))

# 5. Composition: merging the device scoping onto a recipe's env must add the two
#    variables without disturbing what the recipe asked for. This is the shape both
#    launch paths use (dict(os.environ) <- recipe env <- device_env).
child = dict(os.environ)
child.update({"PXA_ENHANCE": "1", "GGML_CUDA_FORCE_MMQ": "1"})
child.update(with_order(None, lambda: pxa.device_env("2,4"))[0])
check("the composed child env carries both device variables",
      child.get("CUDA_VISIBLE_DEVICES") == "2,4"
      and child.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID")
check("and the recipe's own env survives the merge",
      child.get("PXA_ENHANCE") == "1" and child.get("GGML_CUDA_FORCE_MMQ") == "1")

# 6. The frozen restart script (--serve-name) is a second way a seat is started, on
#    another day, by someone who did not watch it being decided. It must carry the
#    order in its own text.
with tempfile.TemporaryDirectory() as td:
    path = with_order(None, lambda: pxa.write_serve_script(
        "devorder-test",
        ["/nonexistent/bin/llama-server", "-m", "/models/x.gguf", "-ngl", "99"],
        {"PXA_ENHANCE": "1"}, "2,4", td))
    text = open(path).read()
    check("the restart script exports CUDA_VISIBLE_DEVICES",
          "export CUDA_VISIBLE_DEVICES=2,4" in text)
    check("the restart script exports CUDA_DEVICE_ORDER=PCI_BUS_ID",
          "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in text)
    check("the order is exported before the devices it orders",
          text.index("CUDA_DEVICE_ORDER") < text.index("export CUDA_VISIBLE_DEVICES"))
    check("the restart script is valid shell",
          subprocess.run(["sh", "-n", path], capture_output=True).returncode == 0)

# 7. Structural, so a new call site cannot reintroduce the bug: no line outside
#    device_env() may put CUDA_VISIBLE_DEVICES into an environment or a script by
#    hand. Mirrors standing assertion A10 in `pxa-launch.py --selftest`, and is
#    repeated here so it is checked even when nobody runs the selftest.
SRC = open(LAUNCHER).read().splitlines()
MARKERS = ('e["CUDA_VISIBLE_DEVICES"]', "e['CUDA_VISIBLE_DEVICES']",
           'env["CUDA_VISIBLE_DEVICES"]', "export CUDA_VISIBLE_DEVICES")


def fn_span(name):
    lo = next((i for i, ln in enumerate(SRC) if ln.startswith(f"def {name}(")), None)
    if lo is None:
        return range(0)
    hi = next((i for i in range(lo + 1, len(SRC)) if SRC[i].startswith("def ")), len(SRC))
    return range(lo, hi)


allowed = set(fn_span("device_env")) | set(fn_span("selftest"))
stray = [i + 1 for i, ln in enumerate(SRC) if any(m in ln for m in MARKERS) and i not in allowed]
check("no call site sets CUDA_VISIBLE_DEVICES outside device_env()",
      not stray, f"lines {stray}")
check("device_env() is the function the launch paths call",
      sum(1 for ln in SRC if "device_env(" in ln and "def device_env" not in ln) >= 4)

print(f"=== {'ALL PASS' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)} ===")
sys.exit(1 if FAILURES else 0)
