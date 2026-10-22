"""Verify the new VRAM hint reaches explain without touching GPU state."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

path = Path(__file__).resolve().parents[1] / "tools/pxa-launch.py"
spec = importlib.util.spec_from_file_location("pxa_launch_sm75_test", path)
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)
launcher.engine_can_explain = lambda _: True
launcher.engine_ld_path = lambda _: ("/engine/lib", "test")
observed = []

def run(command, **kwargs):
    observed.append(kwargs["env"]["PXA_TOPOLOGY"])
    assert kwargs["env"]["PXA_EXPLAIN"] == "1"
    return SimpleNamespace(returncode=0, stdout='PXA_EXPLAIN_JSON {"picks":{"ub":{"value":"fit 2048","status":"FIT"}}}\n')

launcher.subprocess.run = run
a = SimpleNamespace(sm="auto", ub=0, ts="", ngl=999, np=1, ctx=0)
for cards, expected in (
    ([(0, "RTX 2080 Ti", 75, 22528)], "750:22528"),
    ([(0, "RTX 2080 Ti", 75, 22528), (1, "V100", 70, 32768)], "750:22528,700:32768"),
    ([(0, "V100", 70, 32768), (1, "RTX 2080 Ti", 75, 22528)], "700:32768,750:22528"),
    ([(0, "V100", 70, 32768)], "700"),
    ([(0, "P100", 60, 16384), (1, "V100", 70, 32768)], "600,700"),
):
    result, why = launcher.engine_autoconfig("/engine", "/model", cards, a)
    assert observed[-1] == expected, (observed[-1], expected)
    assert launcher.engine_pick(result, "ub")["value"] == "fit 2048"
    assert why == "engine registry"
print("5 topology transport cases passed")
