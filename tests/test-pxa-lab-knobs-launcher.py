#!/usr/bin/env python3
"""bug #324, launcher side: mark_launcher_env() names, in PXA_ENV_BY_PXA, the PXA_*/PXQ* variables the launcher
injected and the user did not have; a name the user already had is never listed. usage: test-pxa-lab-knobs-launcher.py"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(HERE, "..", "tools", "pxa-launch.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for k in [k for k in os.environ if k.startswith(("PXA_", "PXQ"))]:
    del os.environ[k]
os.environ["PXA_USER_SET"] = "1"
e = dict(os.environ)
e.update({"PXA_TSPLIT_REDUCE": "fused", "PXA_USER_SET": "2", "PXQ_X": "1", "CUDA_VISIBLE_DEVICES": "0,1"})
m.mark_launcher_env(e)
got = set(e["PXA_ENV_BY_PXA"].split(","))
assert got == {"PXA_TSPLIT_REDUCE", "PXQ_X"}, got           # not the user's PXA_USER_SET, not CUDA_*
e2 = dict(os.environ)
m.mark_launcher_env(e2)
assert "PXA_ENV_BY_PXA" not in e2                            # nothing injected: no marker
os.environ["PXA_ENV_BY_PXA"] = "PXA_PRIOR"
e3 = dict(os.environ); e3["PXA_NEW"] = "1"
m.mark_launcher_env(e3)
assert e3["PXA_ENV_BY_PXA"] == "PXA_PRIOR,PXA_NEW", e3["PXA_ENV_BY_PXA"]   # an outer marker is kept
print("PASS launcher marker")
