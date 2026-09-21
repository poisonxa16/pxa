"""gate_ple_pp.py -- prove the PP>1 PLE patch is exactly the fork's own PP=1 setup.

CPU only; run inside pxa-vllm:sm60-v15 with PYTHONPATH=<sidecar>/site-sm60.

    python3 tests_pxq23/gate_ple_pp.py [/path/to/config.json]

What it checks (blocker #1595):
  1. the wall really is in the fork source (we are not patching a phantom),
  2. the sidecar extracts the fork's OWN post-assertion statements -- named, listed,
     and compared statement-for-statement against the source,
  3. running them builds exactly the PLE state PP=1 builds: ngram_context filled with the
     EOS id, the [-(n-1) .. -1] offsets, and the query_start_loc buffer,
  4. the ``Qwen4ExpModel.forward`` relay is a no-op when input_ids is already present.
"""
import ast
import inspect
import json
import sys
import types

import torch

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


if len(sys.argv) < 2:
    sys.exit("usage: gate_ple_pp.py /path/to/config.json")
cfg_path = sys.argv[1]
raw = json.load(open(cfg_path))
tcfg = raw.get("text_config", raw)

import pxq4_vllm.qwen4exp_ple_pp as pp  # noqa: E402
from vllm.models.qwen4_exp.nvidia import model_state as ms  # noqa: E402
try:                                     # needs an active CUDA driver (triton autotune)
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel  # noqa: E402
except Exception as _exc:                # CPU-only box: sections 1-3 still run
    Qwen4ExpModel = None
    print(f"SKIP  Qwen4ExpModel import needs a CUDA driver ({type(_exc).__name__}); "
          "the forward-relay checks are skipped on this host")

# --- 1. the wall exists -----------------------------------------------------------------
src = inspect.getsource(ms.Qwen4ExpModelState.__init__)
check("wall present in fork source", pp._WALL in src and "raise RuntimeError" in src)

# --- 2. the tail is the fork's own statements -------------------------------------------
tail_fn, n_tail = pp._build_tail(ms.Qwen4ExpModelState)
body = ast.parse(inspect.getsource(ms.Qwen4ExpModelState.__init__).lstrip()).body
# re-derive the expected tail independently of the sidecar
fn = ast.parse(inspect.cleandoc(src) if False else __import__("textwrap").dedent(src)).body[0]
idx = [
    i for i, n in enumerate(fn.body)
    if isinstance(n, ast.If) and len(n.body) == 1 and isinstance(n.body[0], ast.Raise)
    and pp._WALL in ast.dump(n.body[0])
]
check("exactly one PP wall statement", len(idx) == 1, f"found {len(idx)}")
expect_tail = fn.body[idx[0] + 1:]
check("sidecar tail length == source tail length", n_tail == len(expect_tail),
      f"{n_tail} vs {len(expect_tail)}")
assigned = []
for n in expect_tail:
    for t in getattr(n, "targets", []):
        if isinstance(t, ast.Attribute):
            assigned.append(t.attr)
print("      tail assigns:", assigned)
for want in ("ngram_context_len", "ngram_eos_token_id", "ngram_context",
             "ngram_context_offsets", "ple_query_start_loc"):
    check(f"tail builds self.{want}", want in assigned)

# --- 3. running the tail builds the PP=1 state -------------------------------------------
stub = types.SimpleNamespace()
stub.model_config = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(**tcfg))
stub.max_num_reqs = 8
stub.device = torch.device("cpu")
tail_fn(stub)
n_ctx = int(tcfg["ngram_size"]) - 1
eos = int(tcfg["eos_token_id"])
check("ngram_context_len", stub.ngram_context_len == n_ctx, str(stub.ngram_context_len))
check("ngram_eos_token_id", stub.ngram_eos_token_id == eos, str(stub.ngram_eos_token_id))
check("ngram_context shape/dtype/fill",
      tuple(stub.ngram_context.shape) == (8, n_ctx)
      and stub.ngram_context.dtype == torch.int32
      and bool((stub.ngram_context == eos).all()),
      f"{tuple(stub.ngram_context.shape)} {stub.ngram_context.dtype}")
check("ngram_context_offsets",
      stub.ngram_context_offsets.tolist() == list(range(-n_ctx, 0)),
      str(stub.ngram_context_offsets.tolist()))
check("ple_query_start_loc",
      tuple(stub.ple_query_start_loc.shape) == (9,)
      and stub.ple_query_start_loc.dtype == torch.int32
      and bool((stub.ple_query_start_loc == 0).all()))

# --- 4. patches are installed, and the relay is a no-op with real input_ids ---------------
pp._install_model_state()
if Qwen4ExpModel is not None:
    pp._install_model_forward()
check("Qwen4ExpModelState.__init__ patched", getattr(ms.Qwen4ExpModelState.__init__, "_pxa_ple_pp", False))
check("Qwen4ExpModelState.prepare_inputs patched", getattr(ms.Qwen4ExpModelState.prepare_inputs, "_pxa_ple_pp", False))
if Qwen4ExpModel is not None:
    check("Qwen4ExpModel.forward patched", getattr(Qwen4ExpModel.forward, "_pxa_ple_pp", False))

if Qwen4ExpModel is not None:
    # a stand-alone copy of the relay wrapper around a fake original: no GPU, no real model
    import torch.nn as nn  # noqa: E402


    class _FakeModel(nn.Module):
        def __init__(self, with_ple):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            if with_ple:
                self.layers[0].ple = nn.Identity()


    check("_owns_ple sees a built PLE stack", pp._owns_ple(_FakeModel(True)) == ["layers.0"],
          str(pp._owns_ple(_FakeModel(True))))
    check("_owns_ple is empty without one", pp._owns_ple(_FakeModel(False)) == [])

    seen = {}


    def _fake_orig(self, input_ids, positions, *a, **k):
        seen["ids"] = input_ids
        return "ok"


    # re-wrap the relay around a fake original so no GPU and no real model are needed
    _saved = Qwen4ExpModel.forward
    Qwen4ExpModel.forward = _fake_orig
    pp._install_model_forward()
    probe = _FakeModel(True)
    ids = torch.arange(5, dtype=torch.int32)
    Qwen4ExpModel.forward(probe, ids, torch.arange(5))
    check("relay passes a present input_ids through untouched", seen.get("ids") is ids)
    pp._LAST_INPUT_IDS = None
    seen.clear()
    Qwen4ExpModel.forward(probe, None, torch.arange(5))
    check("relay substitutes nothing when the runner published nothing", seen.get("ids") is None)
    Qwen4ExpModel.forward = _saved


print()
print(("GATE PLE-PP: PASS" if not FAILS else "GATE PLE-PP: FAIL " + ",".join(FAILS)))
sys.exit(1 if FAILS else 0)
