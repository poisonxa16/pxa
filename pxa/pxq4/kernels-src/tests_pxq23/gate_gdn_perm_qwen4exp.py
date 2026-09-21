"""gate_gdn_perm_qwen4exp.py -- the GDN v-head permutation gate for qwen4exp. CPU ONLY.

WHAT IT PROVES, and just as importantly what it does not.

PROVES: at qwen4exp's geometry (16 k-heads, 48 v-heads, head_dim 128, repeat 3 -- against
qwen35moe's 16/32/repeat 2) the converter's own gdn_perm_for() produces a valid bijection for
every tensor GDN_PERM_SPEC covers, and for the three that are in panel format the reorder is
STILL A WHOLE-PANEL OR WHOLE-SLAB GATHER, i.e. still a pure byte move rather than something
that would force a requantization. A 128-row head block is exactly 2 panels and a 128-column
head block exactly 4 slabs, so the alignment holds -- but that is a consequence of this
model's head_dim, not a law, and a model with head_dim 96 would fail here and should.

DOES NOT PROVE: that the permutation is needed, or that its DIRECTION is right. The spec
encodes "ggml orders the v-head axis repeat-major, HF orders it k-head-major". That was
established for qwen35moe against a reference checkpoint. No HF-form qwen4exp exists on this
box, so there is nothing to compare against, and applying a permutation that is not wanted is
just as wrong as omitting one that is. THAT question is settled at first boot by coherence --
a model with the v-heads misordered loads, shards, passes every structural gate and generates
fluent nonsense -- and it must be checked with the needle before any number is quoted.

Run:  python3 gate_gdn_perm_qwen4exp.py [gguf]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import convert as C          # noqa: E402
from gguf_to_vllm import gguf_raw as G         # noqa: E402
from gguf_to_vllm import layout as L           # noqa: E402
from gguf_to_vllm import namemap as NM         # noqa: E402
from gguf_to_vllm import tiers as T            # noqa: E402

DEFAULT = os.environ.get("PXA_CACHEONE", "/tmp") + "/qwen4exp/Qwen3.8-Flash-Next-Uncensored-PXQU-4xP100.gguf"


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT
    gg = G.GGUFFile(path)
    geom = NM.gdn_geometry(gg.kv)
    rep = geom.n_v_heads // geom.n_k_heads
    print(f"geometry {geom}  repeat {rep}")
    if geom.n_v_heads % geom.n_k_heads:
        print("FAIL: v-heads is not a whole multiple of k-heads")
        return 1

    fails = moved = 0
    for suffix in sorted(set(NM.GDN_PERM_SPEC) | set(NM.GDN_NO_PERM)):
        name = f"blk.0.{suffix}"
        if name not in gg.tensors:
            print(f"  {suffix:<20} absent from this architecture")
            continue
        ti = gg.tensors[name]
        # Go through the CONVERTER's entry point, not a re-derivation. My first two attempts
        # at this gate used my own axis-length arithmetic and were wrong twice -- once by
        # asking whether N was a multiple of value_dim (which misses attn_qkv, whose v-heads
        # are a sub-range after q and k), and once by using ne1 for a 1-D tensor (ssm_a is
        # ne=(48,), so ne1 is 1). gdn_perm_for uses logical_shape[axis] and gets both right.
        try:
            got = C.gdn_perm_for(name, ti, geom)
        except SystemExit as exc:
            print(f"  {suffix:<20} REFUSED: {exc}")
            fails += 1
            continue
        if got is None:
            print(f"  {suffix:<20} {ti.type:<5} declared no-perm")
            continue
        axis, gather = got
        g = np.asarray(gather)
        n = ti.logical_shape[axis]
        bij = sorted(g.tolist()) == list(range(n))
        if not bij:
            print(f"  {suffix:<20} NOT A BIJECTION of range({n})")
            fails += 1
            continue
        if not T.is_pxq(ti.type_id):
            print(f"  {suffix:<20} {ti.type:<5} axis {axis} len {n:<6} bijection ok "
                  f"(not panel format, alignment N/A)")
            continue
        try:
            idx = (L.block_gather_to_panels(g) if axis == 0 else L.col_gather_to_slabs(g))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {suffix:<20} NOT ALIGNED -- would need a requantization: {exc}")
            fails += 1
            continue
        unit = "panels" if axis == 0 else "slabs"
        print(f"  {suffix:<20} {ti.type:<5} axis {axis} len {n:<6} bijection ok  "
              f"{idx.size} {unit}, PURE BYTE MOVE")
        moved += 1

    print(f"\nGDN PERMUTATION GATE: {'PASS' if fails == 0 else 'FAIL'}  "
          f"({moved} panel tensors stay byte moves)")
    print("NOTE: this gates APPLICABILITY and alignment, not direction. Whether ggml and HF "
          "actually disagree for qwen4exp is unproven -- no HF-form checkpoint exists to "
          "compare against -- and must be settled by a coherence/needle check at first boot.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
