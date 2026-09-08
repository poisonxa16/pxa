# SPDX-License-Identifier: Apache-2.0
"""Integration checks for the PXQ4 quant config that need the REAL vLLM fork.

DESTINATION IN THE REPO OF PLAN 09: ``tests/test_config_integration.py``.

Still CPU-only -- no GPU, no model, no engine.  Run it inside a *throwaway*
container from the serving image (never the production container):

  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
    -v /mnt/models/pxa-vllm-pxq4/impl:/work -w /work kewaii/vllm:latest \
    /opt/vllm-venv/bin/python test_pxq4_config_integration.py

Every check here is one that a stub cannot honestly make:
  1. the fork's own ``_uses_split_gdn_input_projections`` on our config object;
  2. the fork's own ``get_quantization_config`` lookup table;
  3. the real ``_verify_quantization`` override-probe loop -- confirming no
     built-in method hijacks a ``quant_method: "pxq4"`` checkpoint;
  4. real ``importlib.metadata`` entry-point discovery from a hand-written
     ``.dist-info``, which is how this ships (the image's / is 100% full).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    import vllm  # noqa: F401
except ImportError:
    print("SKIP: real vLLM not importable; run this inside the serving image")
    raise SystemExit(0)

import pxq4_config_stubs as stubs  # noqa: E402

stubs.install_stubs()  # no-op when the real packages exist

_TMP = Path(tempfile.mkdtemp(prefix="pxq4int-"))
sys.path.insert(0, stubs.build_package(_TMP, _HERE / "pxq4_config.py"))

from pxq4_vllm import config as cfg  # noqa: E402
from test_pxq4_config import p1_config_dict, p2c_config_dict  # noqa: E402


def test_real_gdn_split_probe():
    """The fork's own function, not a copy of it."""
    from vllm.model_executor.models.qwen3_5 import _uses_split_gdn_input_projections

    for d in (p1_config_dict(), p2c_config_dict()):
        c = cfg.PXQ4Config.from_config(d)
        assert _uses_split_gdn_input_projections(c), d["pxq4_modules"]

    # And the negative control: strip the two required entries and confirm the
    # probe really would flip. (from_config refuses to build such a config, so
    # mutate after construction.)
    c = cfg.PXQ4Config.from_config(p1_config_dict())
    c.ignore = [x for x in c.ignore if "in_proj_" not in x]
    c.ignored_layers = c.ignore
    c.modules_to_not_convert = []
    c.config = {}
    assert not _uses_split_gdn_input_projections(c)


def test_real_registry_lookup():
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    assert "pxq4" in QUANTIZATION_METHODS
    assert get_quantization_config("pxq4") is cfg.PXQ4Config


def test_no_builtin_method_hijacks_a_pxq4_checkpoint():
    """config/model.py:1045-1067 probes EVERY registered method's
    ``override_quantization_method`` against our checkpoint's quant config and
    breaks on the first non-None.  If some built-in claimed it, our config class
    would never be constructed."""
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    d = p1_config_dict()
    claimers = []
    for name in QUANTIZATION_METHODS:
        try:
            method = get_quantization_config(name)
        except Exception:  # noqa: BLE001 - optional backends (humming, torchao...)
            continue
        try:
            got = method.override_quantization_method(d, None, hf_config=None)
        except TypeError:
            try:
                got = method.override_quantization_method(d, None)
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue
        if got is not None:
            claimers.append((name, got))
    assert claimers == [("pxq4", "pxq4")], claimers


def test_our_override_does_not_hijack_the_incumbent_checkpoint():
    """The same loop runs for every other model served from this image."""
    incumbent = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "quantization_status": "compressed",
    }
    assert cfg.PXQ4Config.override_quantization_method(incumbent, None) is None


def test_entry_point_discovery_from_a_handwritten_dist_info():
    """This is the shipping mechanism: PYTHONPATH + a .dist-info we write by
    hand, because nothing can be installed into the image (its / is 100% full,
    0 bytes available)."""
    import importlib.metadata as md

    site = Path(tempfile.mkdtemp(prefix="pxq4site-"))
    pkg = site / "pxq4_vllm_ep"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "REGISTERED = False\n"
        "def register():\n"
        "    global REGISTERED\n"
        "    REGISTERED = True\n"
    )
    dist = site / "pxq4_vllm_ep-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: pxq4-vllm-ep\nVersion: 0.1.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[vllm.general_plugins]\npxq4 = pxq4_vllm_ep:register\n"
    )

    sys.path.insert(0, str(site))
    try:
        eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "pxq4"]
        assert eps, "entry point not discovered from the hand-written .dist-info"
        fn = eps[0].load()
        fn()
        import pxq4_vllm_ep

        assert pxq4_vllm_ep.REGISTERED is True
    finally:
        sys.path.remove(str(site))


def test_dtype_gate_matches_the_engine_check():
    """config/vllm.py:622-627 raises when model_config.dtype is not in the
    returned list. Volta has no bf16; make sure we would raise rather than
    silently run bf16 activations through fp16 kernels."""
    import torch

    c = cfg.PXQ4Config.from_config(p1_config_dict())
    dtypes = c.get_supported_act_dtypes()
    assert torch.float16 in dtypes
    assert torch.bfloat16 not in dtypes
    assert torch.float32 not in dtypes


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
