"""test_dispatch_mock.py -- CPU/mock gates for the sidecar's tier dispatch.

No GPU, no model, no kernel library. It answers the questions that would otherwise only be
answered by a failed boot in a scheduled window:

  D1  the real checkpoint's quantization_config parses, and parses as a TIERED config
  D2  tier_for() resolves every module prefix the model will actually construct, to the tier
      the converter recorded -- including the w13/w2 split, where longest-suffix-wins is the
      only thing standing between "pxq2 gate/up, pxq3 down" and a silently wrong stride
  D3  book_for() returns the FILE's book for a tiered module, and REFUSES (raises) for a tier
      the checkpoint did not record a book for -- the failure mode that would otherwise be a
      uniform, silent weight error with no load-time symptom
  D4  a legacy PXQ4 checkpoint config still parses, still reports every module as pxq4, and is
      byte-for-byte unaffected by the tier machinery
  D5  the geometry cross-check does not reject a tiered file for having a non-1088 stride,
      and still rejects a single-tier file that declares the wrong one

Run inside a serving image (it imports vllm):
    python3 test_dispatch_mock.py --checkpoint <box path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--site", default=None,
                    help="sidecar site dir to import from (default: infer from PYTHONPATH)")
    args = ap.parse_args()
    if args.site:
        sys.path.insert(0, args.site)

    from pxq4_vllm.config import PXQ4Config
    from pxq4_vllm import tiers as T

    fails = 0
    qc = json.load(open(os.path.join(args.checkpoint, "config.json")))["quantization_config"]

    # ---- D1
    cfg = PXQ4Config.from_config(qc)
    tiered = bool(getattr(cfg, "pxq_tiers", None))
    print(f"D1 parsed: quant_method={qc['quant_method']} tiered={tiered} "
          f"tiers={sorted(set(cfg.pxq_tiers.values())) if tiered else ['pxq4']}")
    if not tiered:
        print("D1 FAIL: this checkpoint declares no pxq_tiers")
        fails += 1

    # ---- D2. These are the prefixes vLLM actually hands get_quant_method for this model.
    LAYER = "model.language_model.layers.7"
    cases = [
        (f"{LAYER}.mlp.experts", "w13", "pxq2"),
        (f"{LAYER}.mlp.experts", "w2", "pxq2"),
        (f"{LAYER}.self_attn.o_proj", None, "pxq4"),
        (f"{LAYER}.linear_attn.in_proj_qkvz", None, "pxq4"),
        (f"{LAYER}.linear_attn.out_proj", None, "pxq4"),
        (f"{LAYER}.mlp.shared_expert.gate_up_proj", None, "pxq4"),
        (f"{LAYER}.mlp.shared_expert.down_proj", None, "pxq4"),
        # a module the file says nothing about must fall back to pxq4, never to a guess
        (f"{LAYER}.self_attn.qkv_proj", None, "pxq4"),
    ]
    for prefix, which, want in cases:
        got = cfg.tier_for(prefix, which)
        ok = got.name == want
        print(f"D2 tier_for({prefix.split('.', 3)[-1]}, {which}) -> {got.name} "
              f"(slab {got.slab_bytes}) {'ok' if ok else 'FAIL want ' + want}")
        fails += not ok

    # ---- D2b. Longest-suffix-wins, on a synthetic mixed-tier map: this is the resolution
    # that makes pxq2 gate/up beside pxq3 down possible, so it is tested even though the file
    # in hand is uniform.
    mixed = dict(qc)
    mixed["pxq_tiers"] = dict(qc["pxq_tiers"])
    mixed["pxq_tiers"]["mlp.experts.w2"] = "pxq3"
    mixed["tier_books"] = dict(qc["tier_books"])
    mixed["tier_books"]["pxq3"] = [-0.90673828125, -0.5478515625, -0.2978515625,
                                   -0.0931396484375, 0.0919189453125, 0.295654296875,
                                   0.54541015625, 0.90576171875]
    cfg2 = PXQ4Config.from_config(mixed)
    a = cfg2.tier_for(f"{LAYER}.mlp.experts", "w13").name
    b = cfg2.tier_for(f"{LAYER}.mlp.experts", "w2").name
    ok = (a, b) == ("pxq2", "pxq3")
    print(f"D2b mixed w13/w2 map -> w13={a} w2={b} {'ok' if ok else 'FAIL'}")
    fails += not ok

    # ---- D3
    b2 = cfg.book_for(T.BY_NAME["pxq2"])
    ok = len(b2) == 4 and b2 == tuple(qc["tier_books"]["pxq2"])
    print(f"D3 book_for(pxq2) -> {len(b2)} entries, from the file: {'ok' if ok else 'FAIL'}")
    fails += not ok
    try:
        cfg.book_for(T.BY_NAME["pxq3"])
        print("D3 FAIL: book_for(pxq3) returned a book the checkpoint never recorded")
        fails += 1
    except ValueError as exc:
        print(f"D3 book_for(pxq3) correctly refused: {str(exc)[:80]}...")

    # ---- D4 legacy
    legacy = {k: v for k, v in qc.items()
              if k not in ("pxq_tiers", "tier_books", "tier_sub", "tier_note")}
    legacy["quant_method"] = "pxq4"
    lc = PXQ4Config.from_config(legacy)
    ok = (not lc.pxq_tiers
          and lc.tier_for(f"{LAYER}.mlp.experts", "w13").name == "pxq4"
          and lc.book_for(T.BY_NAME["pxq4"]) == tuple(legacy["book"]))
    print(f"D4 legacy pxq4 config: uniformly pxq4, book from 'book' {'ok' if ok else 'FAIL'}")
    fails += not ok

    # ---- D5
    bad = dict(legacy)
    bad["slab_bytes"] = 576                     # a single-tier file lying about its stride
    try:
        PXQ4Config.from_config(bad)
        print("D5 FAIL: a single-tier config declaring slab_bytes=576 was accepted")
        fails += 1
    except ValueError:
        print("D5 single-tier config with the wrong slab_bytes correctly refused")
    good = dict(qc)
    good["slab_bytes"] = 576                    # a TIERED file may declare any default stride
    try:
        PXQ4Config.from_config(good)
        print("D5 tiered config not rejected for its default slab_bytes: ok")
    except ValueError as exc:
        print(f"D5 FAIL: tiered config rejected on slab_bytes: {exc}")
        fails += 1

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
