"""test_sidecar_import.py -- import the sidecar as a boot would, and parse every checkpoint.

WHY THIS EXISTS. It was written after it found a bug that nothing else in the suite would
have found until a 12 GiB model load: PXQ4Config.__init__ computed _ignore_for_dispatch --
which READS self.q8_modules -- several lines BEFORE self.q8_modules was assigned. Every
config carrying q8_modules, i.e. every m3 checkpoint, raised AttributeError inside __init__.
It stayed latent because every config exercised until then had no q8 modules: the unit tests
used m2, and m3's config had been written by the converter but never loaded.

So the gate is deliberately shaped as "would a boot get this far", not as a unit test:
  * import pxq4_vllm the way the plugin entry point does, with NOTHING armed, so the
    default-off claim of every landed pack is exercised rather than asserted
  * import each optional module by name, so a pack that lands a file that does not import
    is caught here rather than inside a guarded try in __init__ that swallows it
  * run PXQ4Config.from_config over EVERY checkpoint on disk, because the interesting
    configs are the ones a unit test did not think to write

Cheap enough to run after every landing: no GPU, no model weights, a few seconds.

    PYTHONPATH=<pkg>/sidecar/site-sm60 python3 test_sidecar_import.py <ckpt dir> [<ckpt dir> ...]
"""

from __future__ import annotations

import json
import os
import sys


def main(argv: list[str]) -> int:
    ckpts = argv[1:] or [
        os.environ.get("PXA_MODELS", "./models") + "/coder35-moe-pxq4-m3",
        os.environ.get("PXA_MODELS", "./models") + "/coder35-moe-pxq4-m2",
        os.environ.get("PXA_MODELS", "./models") + "/fusion2-35b-pxq2-vllm-m2",
    ]
    for var in ("PXA_OPS_FUSED", "PXA_STEP_TIMER", "PXA_FUSED_NORM", "PXQ4_MOE_EPILOGUE"):
        if os.getenv(var):
            print(f"NOTE: {var}={os.getenv(var)} is set; this gate is about the DEFAULT state")

    fails = 0
    import pxq4_vllm  # noqa: F401
    from pxq4_vllm import config, moe, tiers

    # Every optional module by name. A file that lands but does not import would otherwise be
    # swallowed by the guarded try blocks in __init__ and show up as a missing speedup.
    optional = ("pxa_step_timer", "pxa_ir_norm", "pxa_pascal_ops", "head_q8",
                "pxa_sm60_f16", "pxa_sdpa_tiled", "linear", "ops", "parameters")
    for name in optional:
        try:
            __import__(f"pxq4_vllm.{name}")
            print(f"  import pxq4_vllm.{name}: ok")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  import pxq4_vllm.{name}: FAIL {type(exc).__name__}: {exc}")
            fails += 1

    print(f"  moe._EPILOGUE default: {moe._EPILOGUE!r}")
    if moe._EPILOGUE != "legacy":
        print("  FAIL: the MoE epilogue default is not 'legacy'")
        fails += 1
    print(f"  tiers known: {sorted(tiers.BY_NAME)}")

    for path in ckpts:
        cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(cfg_path):
            print(f"  SKIP {path} (no config.json)")
            continue
        try:
            q = json.load(open(cfg_path))["quantization_config"]
            c = config.PXQ4Config.from_config(q)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {os.path.basename(path)}: FAIL {type(exc).__name__}: {exc}")
            fails += 1
            continue
        q8 = list(getattr(c, "q8_modules", []))
        ign = "lm_head" in c.ignore
        tier = c.tier_for("model.layers.7.mlp.experts", "w13").name
        # THE EXCLUSIVE-OR: the head is either ignored or served as int8, never both and
        # never neither. Both would route it to fp16 and then look for a tensor that is not
        # in the file; neither would route it nowhere.
        head_ok = (("lm_head" in q8) != ign)
        print(f"  {os.path.basename(path)}: q8={q8} lm_head_ignored={ign} experts.w13={tier} "
              f"head_xor={'ok' if head_ok else 'FAIL'}")
        fails += not head_ok

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
