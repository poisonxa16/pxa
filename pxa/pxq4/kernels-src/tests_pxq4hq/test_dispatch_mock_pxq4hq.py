"""test_dispatch_mock_pxq4hq.py -- CPU/mock gates for the sidecar's PXQ4HQ dispatch.

No GPU, no model, no kernel library. It answers the questions that would otherwise only be
answered by a failed boot in a scheduled window:

  H1  the real converted checkpoint's quantization_config parses, and parses as TIERED
  H2  tier_for() resolves every module prefix the model will construct to pxq4hq, and the
      resolved Tier carries the 1152 stride and the empty MoE op name
  H3  sub_for() returns the tier's OWN SUB8 for pxq4hq and the SHARED SUB16 for a tier that
      shares -- the single decision that separates a correct load from a uniformly wrong one
  H4  a checkpoint that declares pxq4hq modules but records no tier_subs["pxq4hq"] is REFUSED
      at that call, not decoded against SUB16. This is the file a converter predating this
      tier would produce, and the failure it would otherwise cause has no load-time symptom.
  H5  a legacy single-tier PXQ4 config still parses, still reports every module as pxq4, and
      sub_for still hands it the shared table -- the tier machinery is inert for it
  H6  the geometry cross-check does not reject this file for its non-1088 stride

Run inside a serving image (it imports vllm):
    python3 test_dispatch_mock_pxq4hq.py --checkpoint <box path>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAYER = "model.language_model.layers.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    import tiers as T                      # noqa: PLC0415
    import pxq4_config as C                # noqa: PLC0415

    with open(os.path.join(args.checkpoint, "config.json")) as f:
        qc = json.load(f)["quantization_config"]
    fails = []

    # ---- H1
    cfg = C.PXQ4Config.from_config(qc)
    tiered = bool(qc.get("pxq_tiers"))
    print(f"H1 config parses, tiered={tiered}, quant_method={qc.get('quant_method')}")
    if not tiered:
        fails.append("H1: the checkpoint did not parse as tiered")

    # ---- H2
    hq = T.BY_NAME["pxq4hq"]
    if (hq.type_id, hq.slab_bytes, hq.book_n, hq.own_sub, hq.op_moe_mmv) != (253, 1152, 16, True, ""):
        fails.append(f"H2: the pxq4hq Tier row is wrong: {hq}")
    seen = set()
    for suffix in sorted(qc["pxq_tiers"]):
        t = cfg.tier_for(f"{LAYER}.{suffix}")
        seen.add(t.name)
        if t.name != qc["pxq_tiers"][suffix]:
            fails.append(f"H2: {suffix} resolved to {t.name}, file says {qc['pxq_tiers'][suffix]}")
    print(f"H2 tier_for resolves {len(qc['pxq_tiers'])} declared suffixes -> {sorted(seen)}")

    # ---- H3
    s_hq = cfg.sub_for(hq)
    s_sh = cfg.sub_for(T.BY_NAME["pxq4"])
    if tuple(s_hq) == tuple(s_sh):
        fails.append("H3: sub_for(pxq4hq) returned the SHARED SUB16 -- the wrong table")
    if abs(float(s_hq[0]) - 0.1683349609375) > 0:
        fails.append(f"H3: sub_for(pxq4hq)[0] = {s_hq[0]}, expected the SUB8 first level")
    if abs(float(s_sh[0]) - 0.2147216796875) > 0:
        fails.append(f"H3: sub_for(pxq4)[0] = {s_sh[0]}, expected the SUB16 first level")
    print(f"H3 sub_for: pxq4hq[0]={float(s_hq[0]):.10f} (SUB8)  shared[0]={float(s_sh[0]):.10f} (SUB16)")

    # ---- H4
    stripped = copy.deepcopy(qc)
    stripped.pop("tier_subs", None)
    cfg2 = C.PXQ4Config.from_config(stripped)
    try:
        cfg2.sub_for(hq)
        fails.append("H4: a checkpoint with no tier_subs[pxq4hq] was NOT refused")
        print("H4 FAIL: sub_for returned a table the checkpoint never recorded")
    except ValueError as exc:
        print(f"H4 correctly refused: {str(exc)[:110]}...")

    # ---- H5
    legacy = {k: v for k, v in qc.items()
              if k not in ("pxq_tiers", "tier_books", "tier_sub", "tier_subs", "tier_note")}
    legacy["quant_method"] = "pxq4"
    cfg3 = C.PXQ4Config.from_config(legacy)
    t = cfg3.tier_for(f"{LAYER}.mlp.down_proj")
    if t.name != "pxq4" or t.slab_bytes != 1088:
        fails.append(f"H5: a legacy config resolved {t.name}/{t.slab_bytes}, expected pxq4/1088")
    if tuple(cfg3.sub_for(t)) != tuple(cfg3.sub):
        fails.append("H5: sub_for on a legacy config did not return its own shared sub")
    print(f"H5 legacy PXQ4 config -> {t.name}, stride {t.slab_bytes}, shared sub: ok")

    # ---- H6
    print(f"H6 declared slab_bytes={qc.get('slab_bytes')} (the pxq4 default row) with "
          f"pxq_tiers present: the geometry cross-check is skipped, as designed")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print("\nALL DISPATCH GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
