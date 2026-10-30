#!/usr/bin/env python3
"""Checks the _pxa_lpt helper of solo-prefill-scheduler.patch without vLLM."""
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
patch = open(os.path.join(HERE, "solo-prefill-scheduler.patch")).read()
src = "\n".join(l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
m = re.search(r"(_PXA_LPT_SOLO = .*?\n)\n\ndef _pxa_lpt\(sched\) -> int:\n(.*?)\n    return thr\n", src, re.S)
assert m, "helper not found in the patch"
code = "import os\n" + m.group(1) + "\ndef _pxa_lpt(sched):\n" + m.group(2) + "\n    return thr\n"


def run(env_on, thr, running, waiting, skipped):
    os.environ["PXA_LPT_SOLO"] = "1" if env_on else "0"
    ns = {}
    exec(code, ns)
    sched = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(long_prefill_token_threshold=thr),
                                  running=[0] * running, waiting=[0] * waiting, skipped_waiting=[0] * skipped)
    return ns["_pxa_lpt"](sched)


CASES = [
    ("alone, enabled", run(True, 1648, 1, 0, 0), 0),
    ("alone in waiting, enabled", run(True, 1648, 0, 1, 0), 0),
    ("two requests, enabled", run(True, 1648, 1, 1, 0), 1648),
    ("running + skipped, enabled", run(True, 1648, 1, 0, 1), 1648),
    ("alone, disabled", run(False, 1648, 1, 0, 0), 1648),
    ("threshold 0", run(True, 0, 1, 0, 0), 0),
]
bad = 0
for name, got, want in CASES:
    bad += got != want
    print("ok  " if got == want else "FAIL", name, got)
sys.exit(1 if bad else 0)
