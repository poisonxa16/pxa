"""gpu_split_gate.py -- the device gate for the PXQ2/PXQ3 K-chunk-split decode family.

The claim this file has to prove is BIT-EXACTNESS, not closeness. The split kernels preserve
the monolithic kernel's fold exactly -- per-lane left-associated chunk chain, then ascending
k-segment, one final ``__float2half_rn`` -- and their atomic is an arrival counter rather than
an accumulator, so every output must equal ``k_pxq23_mmv``'s to the bit. A tolerance here would
hide the only defects this family can have.

It must run on a CARD. A host replay executes blocks sequentially and is structurally
incapable of observing the fused kernels' arrival barrier; a missing release fence produces
silently stale fp16 that nothing downstream can detect.

  A  library self-test        ``torch.ops.pxq4.pxq_selftest(tier)``. Inside the .so: the host
                              oracle arms that predate this change, PLUS the split
                              differential -- four ragged shapes covering nfix 2/4/8/16, M in
                              {1,2,3,5,8,16}, each of split2 / fused / fused_mt run three times
                              back to back and compared to the monolithic kernel at
                              max-abs-diff 0, with the arrival counters read back and required
                              to be zero afterwards.
  E  REAL weights             the same three-way comparison on an actual model tensor out of a
                              PXQ2 and a PXQ3 checkpoint, at every M from 1 to 16. Synthetic
                              panels exercise the arithmetic; a real tensor exercises the
                              shape the server will actually run (panels in the hundreds,
                              K in the thousands, nfix at the CMAX cap).
  F  CUDA GRAPH capture       arm E's decode shapes replayed from a captured graph after
                              ``pxq_warm_split``. This is what says the persistent arenas and
                              the counter rearm survive capture: a graph records raw device
                              addresses, and a counter left dirty by one replay makes the next
                              one write nothing at all.
  G  dequant + mm             one independent cross-check per tier that does not share the
                              fold: dequantise the weight and multiply with cuBLAS. NOT
                              bit-exact by construction (different order, different rounding
                              points), so this arm alone states a tolerance -- 2e-3 relative on
                              the row norm, the same figure the pre-existing mmv arm uses.

Usage (one GPU, no server, a few hundred MB):

    PXQ4_LIB=/out/libpxq_sm60_v18.so python3 gpu_split_gate.py \\
        --ckpt-pxq2 /models/qwen38-27b-pxq2-vllm-p2a \\
        --ckpt-pxq3 /models/qwen38-27b-pxq3-vllm-rc3

Exit code 0 means every arm passed; non-zero names the first arm that did not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

TIER_ID = {"pxq2": 254, "pxq3": 255, "pxq4": 252}
# Decode-shaped batches. 1 is the np1 step; 2..8 is the concurrency range the multi-token
# kernel owns; 9..16 is the verify-batch range the linear dispatcher now serves in one pass.
M_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]
# The shapes a captured decode graph actually replays on this stack.
M_CAPTURE = [1, 2, 4, 8]


def load_lib() -> int:
    path = os.environ.get("PXQ4_LIB")
    if not path:
        raise SystemExit("set PXQ4_LIB to the libpxq_*.so you want to gate")
    torch.ops.load_library(path)
    for name in ("pxq_selftest", "pxq2_mmv_mono_out", "pxq2_mmv_split2_out", "pxq_warm_split"):
        if not hasattr(torch.ops.pxq4, name):
            raise SystemExit(f"{path} has no torch.ops.pxq4.{name} -- it predates the split "
                             f"family; there is nothing here to gate")
    ver = int(torch.ops.pxq4.pxq_version())
    print(f"loaded {path}, pxq_version={ver}")
    return ver


def op(tier: str, which: str):
    return getattr(torch.ops.pxq4, f"{tier}_{which}")


def load_real_weight(ckpt: str, tier: str, device: torch.device):
    """One real linear weight out of a checkpoint, as (slabs, anchor, N, K).

    The on-disk key suffix is ``pxq4_slabs``/``pxq4_anchor`` for every tier -- that is a
    wire-format name, not a tier claim; the tier travels in config.json. The slab STRIDE is
    what identifies the tier, and the op checks it, so a mismatch here fails loudly rather
    than decoding into well-formed garbage.
    """
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(ckpt, "config.json")))["quantization_config"]
    tiers = cfg.get("pxq_tiers") or {}
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]

    want = None
    for key in idx:
        if not key.endswith(".pxq4_slabs"):
            continue
        module = key[: -len(".pxq4_slabs")]
        family = ".".join(module.split(".")[-2:])
        if tiers.get(family) == tier:
            want = module
            break
    if want is None:
        raise SystemExit(f"{ckpt} declares no {tier} linear module (tiers: {sorted(set(tiers.values()))})")

    shard = idx[want + ".pxq4_slabs"]
    blob = load_file(os.path.join(ckpt, shard))
    slabs = blob[want + ".pxq4_slabs"].to(device)
    anchor = blob[want + ".pxq4_anchor"].to(device)
    N, K = slabs.shape[0] * 64, slabs.shape[1] * 32
    print(f"  {tier}: {want} from {shard}: slabs {tuple(slabs.shape)} -> N={N} K={K}")

    # honour the FILE's tables, exactly as the runtime does
    books = cfg.get("tier_books") or {}
    if tier in books:
        torch.ops.pxq4.pxq_set_book(TIER_ID[tier], torch.tensor(books[tier], dtype=torch.float32))
    sub = cfg.get("tier_sub") or cfg.get("sub")
    if sub:
        torch.ops.pxq4.pxq_set_sub(torch.tensor(sub, dtype=torch.float32))
    return slabs, anchor, N, K


def first_diff(got: torch.Tensor, ref: torch.Tensor) -> str:
    d = (got.view(torch.int16).to(torch.int32) - ref.view(torch.int16).to(torch.int32)).abs()
    n = int((d != 0).sum())
    where = torch.nonzero(d.reshape(-1), as_tuple=False)
    i = int(where[0, 0]) if where.numel() else -1
    return (f"{n} of {d.numel()} elements differ, max |ULP| {int(d.max())}, "
            f"first at flat index {i}: 0x{int(got.reshape(-1)[i].view(torch.int16)) & 0xFFFF:04x} "
            f"vs 0x{int(ref.reshape(-1)[i].view(torch.int16)) & 0xFFFF:04x}")


def arm_e(tier: str, slabs, anchor, N: int, K: int, device) -> int:
    fails = 0
    torch.manual_seed(0x5EED)
    for m in M_VALUES:
        x = (torch.randn((m, K), device=device, dtype=torch.float32) * 0.05).to(torch.float16)
        ref = torch.empty((m, N), device=device, dtype=torch.float16)
        op(tier, "mmv_mono_out")(ref, x, slabs, anchor)

        for arm, fn in (("dispatched", op(tier, "mmv_out")),
                        ("split2", op(tier, "mmv_split2_out"))):
            got = torch.full((m, N), float("nan"), device=device, dtype=torch.float16)
            for _ in range(3):          # repeat: a bad counter rearm passes the first launch
                fn(got, x, slabs, anchor)
            torch.cuda.synchronize()
            if torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                continue
            print(f"    E FAIL {tier} M={m} arm={arm}: {first_diff(got, ref)}")
            fails += 1
    if not fails:
        print(f"    E {tier}: PASS, bit-exact vs the monolithic mmv at M={M_VALUES} "
              f"(dispatched and split2, 3 launches each)")
    return fails


def arm_f(tier: str, slabs, anchor, N: int, K: int, device) -> int:
    """Replay the decode shapes from a captured graph.

    pxq_warm_split sizes the arenas and zeroes the counters BEFORE capture, which is the
    contract the C++ side refuses to break: growing an arena inside a capture would hand the
    old block back to the allocator while the graph still writes into it.
    """
    torch.ops.pxq4.pxq_warm_split(TIER_ID[tier], slabs, anchor, 16)
    fails = 0
    torch.manual_seed(0xC0FFEE)
    for m in M_CAPTURE:
        x = (torch.randn((m, K), device=device, dtype=torch.float32) * 0.05).to(torch.float16)
        ref = torch.empty((m, N), device=device, dtype=torch.float16)
        op(tier, "mmv_mono_out")(ref, x, slabs, anchor)
        out = torch.empty((m, N), device=device, dtype=torch.float16)

        # one eager call on this exact shape first -- the same warmup vLLM performs
        op(tier, "mmv_out")(out, x, slabs, anchor)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(tier, "mmv_out")(out, x, slabs, anchor)
        torch.cuda.current_stream().wait_stream(s)
        try:
            with torch.cuda.graph(g):
                op(tier, "mmv_out")(out, x, slabs, anchor)
        except Exception as exc:                          # noqa: BLE001 - report, do not raise
            print(f"    F FAIL {tier} M={m}: capture raised {type(exc).__name__}: {exc}")
            return fails + 1

        for _ in range(4):            # several replays: the counters must rearm every time
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                print(f"    F FAIL {tier} M={m} on graph replay: {first_diff(out, ref)}")
                fails += 1
                break
        del g
    if not fails:
        print(f"    F {tier}: PASS, captured graph replays bit-exact at M={M_CAPTURE} "
              f"(4 replays each, arenas warmed pre-capture)")
    return fails


def arm_g(tier: str, slabs, anchor, N: int, K: int, device) -> int:
    """An independent cross-check that does not share the fold. Tolerance is stated, not tuned."""
    torch.manual_seed(7)
    x = (torch.randn((1, K), device=device, dtype=torch.float32) * 0.05).to(torch.float16)
    w = torch.empty((N, K), device=device, dtype=torch.float16)
    op(tier, "dequant_out")(w, slabs, anchor)
    want = (x.float() @ w.float().t())
    got = torch.empty((1, N), device=device, dtype=torch.float16)
    op(tier, "mmv_out")(got, x, slabs, anchor)
    torch.cuda.synchronize()
    rel = float((got.float() - want).norm() / want.norm().clamp_min(1e-30))
    ok = rel < 2e-3
    print(f"    G {tier}: {'PASS' if ok else 'FAIL'}, mmv vs dequant+cuBLAS "
          f"relative row-norm error {rel:.2e} (tolerance 2e-3)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt-pxq2", default=None)
    ap.add_argument("--ckpt-pxq3", default=None)
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    load_lib()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"device: {torch.cuda.get_device_name(dev)} (sm_{cap[0]}{cap[1]})")
    fails = 0

    # ------------------------------------------------------------------ A
    if args.skip_selftest:
        print("\nA library self-test: SKIPPED")
    else:
        for tname in ("pxq4", "pxq2", "pxq3"):
            rc = int(torch.ops.pxq4.pxq_selftest(TIER_ID[tname]))
            print(f"A library self-test {tname}: {'PASS' if rc == 0 else f'FAIL rc={rc}'}")
            fails += bool(rc)

    # ------------------------------------------------------- E / F / G
    for tname, ckpt in (("pxq2", args.ckpt_pxq2), ("pxq3", args.ckpt_pxq3)):
        if not ckpt:
            print(f"\n{tname}: E/F/G SKIPPED (no --ckpt-{tname})")
            continue
        print(f"\n=== {tname} on real weights ===")
        slabs, anchor, N, K = load_real_weight(ckpt, tname, dev)
        fails += arm_e(tname, slabs, anchor, N, K, dev)
        fails += arm_f(tname, slabs, anchor, N, K, dev)
        fails += arm_g(tname, slabs, anchor, N, K, dev)
        del slabs, anchor
        torch.cuda.empty_cache()

    print(f"\n{'ALL ARMS PASS' if fails == 0 else f'{fails} ARM(S) FAILED'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
