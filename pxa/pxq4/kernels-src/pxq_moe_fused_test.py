#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness gate for the FUSED MoE decode block, on REAL expert tensors.

WHAT IT PROVES, and why each claim needs its own comparison:

  1. FUSED FORM A == THE SHIPPED UNFUSED PATH, BIT-EXACT.  Not "within a tolerance", and not
     against a reference of my own choosing: against ``(dn.float() * w).sum(dim=1)``, the exact
     line the sidecar runs today.  That became possible once the reduce axis was enumerated
     rather than assumed -- torch folds these top_k terms with FOUR accumulators taken
     stride-wise, and the kernel now does the same (see pxq_moe_fused.cuh).  The earlier
     ascending fold differed by 1 ULP on a handful of elements and cost the 20-prompt byte gate
     7/20; this gate is the thing that has to pass for the fused block to be a default.

  2. FUSED FORM B == FUSED FORM A, BIT-EXACT.  Form B keeps the slot axis in the grid and pays
     an extra launch; it exists so the speed window can measure blocks-versus-launches instead
     of guessing.  It must not be a second set of numerics.

  3. FUSED vs THE SHIPPED PATH (torch's sum(dim=1)) is REPORTED, not asserted.  The two differ
     only in fp32 accumulation order over top_k terms, so the honest statement is a measured
     max-abs and max-ULP difference plus the fraction of elements that differ at all -- and the
     ship gate for that difference is greedy byte-identity on the 20-prompt set, which is a
     serving-level test and does not belong in this file.

  4. PADDING SLOTS.  vLLM emits id < 0 for an unrouted slot.  The kernels must write a zero
     contribution and must still WRITE the output row rather than leave it stale, so the test
     deliberately poisons the output buffers and routes some slots to -1.

Deliberately standalone: it needs torch and the .so, NOT a vLLM install and NOT a model boot.
That is what lets it run on a spare card under a short lock instead of inside a seat window.

    PXQ4_LIB=/path/libpxq_sm60_v13.so python3 pxq_moe_fused_test.py \\
        --model <box path> --experts 32 --tp 1 --M 1,2,4,8

Exit code 0 = every gate passed.  Nonzero = the first failure, named.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PANEL_ROWS = 64
SLAB_COLS = 32
TIER_ID = {"pxq2": 254, "pxq3": 255, "pxq4": 252}
TIER_SLAB = {254: 576, 255: 832, 252: 1088}
TIER_OPS = {
    254: dict(moe_mmv="pxq2_moe_mmv_out", gateup="pxq2_moe_gateup_glu_out",
              down_fold="pxq2_moe_down_fold_out", down_part="pxq2_moe_down_part_out"),
    255: dict(moe_mmv="pxq3_moe_mmv_out", gateup="pxq3_moe_gateup_glu_out",
              down_fold="pxq3_moe_down_fold_out", down_part="pxq3_moe_down_part_out"),
    252: dict(moe_mmv="moe_mmv_out", gateup="moe_gateup_glu_out",
              down_fold="moe_down_fold_out", down_part="moe_down_part_out"),
}


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_lib() -> None:
    path = os.getenv("PXQ4_LIB")
    if not path:
        die("set PXQ4_LIB to the libpxq_<arch>_v13.so under test")
    torch.ops.load_library(path)
    if not hasattr(torch.ops, "pxq4"):
        die(f"{path} registered no pxq4 namespace")
    missing = [n for n in ("moe_gateup_glu_out", "moe_down_fold_out", "moe_down_part_out",
                           "moe_slot_fold_out", "moe_fused_version")
               if not hasattr(torch.ops.pxq4, n)]
    if missing:
        die(f"{path} is a v13 library WITHOUT the fused MoE block (missing {missing}). "
            f"Rebuild with build_pxq_v13.sh from this tree.")
    print(f"lib      : {path}")
    print(f"fused v  : {int(torch.ops.pxq4.moe_fused_version())}")


# ------------------------------------------------------------------ checkpoint -> stacked moe
def read_cfg(model: str) -> dict:
    with open(os.path.join(model, "config.json")) as fh:
        return json.load(fh)


def tiers_of(cfg: dict) -> tuple[int, int]:
    """(tier13, tier2) as ggml type ids, from the checkpoint's own declaration."""
    q = cfg.get("quantization_config", {})
    pt = q.get("pxq_tiers") or {}
    t13 = pt.get("mlp.experts.w13") or q.get("tier") or "pxq4"
    t2 = pt.get("mlp.experts.w2") or q.get("tier") or "pxq4"
    # `tier: "core"` in the older configs means the default pxq4 tier.
    t13 = "pxq4" if t13 not in TIER_ID else t13
    t2 = "pxq4" if t2 not in TIER_ID else t2
    return TIER_ID[t13], TIER_ID[t2]


def upload_books(cfg: dict, tiers: set[int]) -> None:
    """Honour the checkpoint's recorded book/sub, exactly as the sidecar does at load time.

    Not optional: the quantizer's tables are overridable at build time and the file is the only
    record of which were actually used. Decoding a custom-book file with the compiled-in
    default is silent, uniform weight error -- and it would make this whole test a comparison
    of two identically-wrong numbers.
    """
    q = cfg.get("quantization_config", {})
    sub = q.get("tier_sub") or q.get("sub")
    books = q.get("tier_books") or {}
    sm70 = torch.cuda.get_device_capability(0)[0] >= 7
    for tid in sorted(tiers):
        name = {v: k for k, v in TIER_ID.items()}[tid]
        book = books.get(name) or (q.get("book") if tid == 252 else None)
        if book is None:
            print(f"note     : checkpoint records no book for {name}; using the compiled-in table")
            continue
        b = torch.as_tensor(list(book), dtype=torch.float32)
        sb = torch.as_tensor(list(sub), dtype=torch.float32) if sub else None
        if tid == 252:
            # PXQ4's table upload goes through set_tables, which ALSO writes the MMA TU's copy
            # of the book -- and pxq4_mma.cu is compiled sm_70 only, so on a Pascal card that
            # cudaMemcpyToSymbol returns "no kernel image is available" and the library's
            # PXQ4_MMA_CHECK calls abort(). The process dies; it is not catchable. So: compare
            # first, upload only if it is actually needed, and refuse loudly rather than
            # crashing if it is needed on an arch that cannot do it.
            have = torch.ops.pxq4.get_tables().flatten().float().cpu()
            same = torch.equal(have[:16], b) and (sb is None or torch.equal(have[16:32], sb))
            if same:
                print(f"tables   : {name} compiled-in book/sub already MATCH the checkpoint "
                      f"(verified against get_tables, no upload needed)")
                continue
            if not sm70:
                die(f"this checkpoint records a custom {name} book that differs from the "
                    f"compiled-in one, and set_tables cannot run on this Pascal card: it "
                    f"unconditionally uploads the sm_70-only MMA table and aborts. Run this "
                    f"arm on a Volta card, or fix set_tables to guard that second upload on "
                    f"pxq4_mma_arch_ok().")
            if sb is None:
                die("pxq4 book present but no sub table: set_tables needs both")
            torch.ops.pxq4.set_tables(b, sb)
        else:
            torch.ops.pxq4.pxq_set_book(tid, b)
            if sb is not None:
                torch.ops.pxq4.pxq_set_sub(sb)
        print(f"tables   : uploaded {name} book ({b.numel()} entries)")


def stack_experts(model: str, layer: int, n_exp: int, rank: int, world: int, dev):
    """Build the stacked w13/w2 exactly as PXQ4MoEMethod._weight_loader does, TP cut included.

    w13 is ONE tensor with gate in the first half of the PANEL axis and up in the second, which
    is the layout the fused gateup kernel walks; getting this wrong is the single easiest way
    to produce a test that passes on garbage.
    """
    from safetensors import safe_open

    idx_path = os.path.join(model, "model.safetensors.index.json")
    with open(idx_path) as fh:
        wmap = json.load(fh)["weight_map"]

    def pfx(e: int, proj: str) -> str:
        for stem in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
            k = f"{stem}.{layer}.mlp.experts.{e}.{proj}_proj"
            if k + ".pxq4_slabs" in wmap:
                return k
        die(f"cannot find layer {layer} expert {e} {proj}_proj in {idx_path}")

    handles: dict[str, object] = {}

    def get(key: str) -> torch.Tensor:
        fn = wmap[key]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(model, fn), framework="pt", device="cpu")
        return handles[fn].get_tensor(key)

    g0 = get(pfx(0, "gate") + ".pxq4_slabs")
    d0 = get(pfx(0, "down") + ".pxq4_slabs")
    p_full, s13, slab13 = g0.shape          # gate panels (full, unsharded), kslabs, slab bytes
    p2_full, s2_full, slab2 = d0.shape

    per13 = p_full // world                 # column parallel: cut the PANEL axis
    per2 = s2_full // world                 # row parallel: cut the K-SLAB axis
    if per13 * world != p_full or per2 * world != s2_full:
        die(f"TP={world} does not divide this layer (gate panels {p_full}, down kslabs {s2_full})")

    w13_s = torch.empty((n_exp, 2 * per13, s13, slab13), dtype=torch.uint8)
    w13_a = torch.empty((n_exp, 2 * per13, PANEL_ROWS), dtype=torch.float16)
    w2_s = torch.empty((n_exp, p2_full, per2, slab2), dtype=torch.uint8)
    w2_a = torch.empty((n_exp, p2_full, PANEL_ROWS), dtype=torch.float16)

    for e in range(n_exp):
        for half, proj in ((0, "gate"), (per13, "up")):
            k = pfx(e, proj)
            w13_s[e, half:half + per13] = get(k + ".pxq4_slabs").narrow(0, rank * per13, per13)
            w13_a[e, half:half + per13] = get(k + ".pxq4_anchor").narrow(0, rank * per13, per13)
        k = pfx(e, "down")
        # The anchor is NOT cut on a row-parallel shard: it is a linear per-output-row scale,
        # so every rank keeps the whole thing and the all-reduce sums the partials.
        w2_s[e] = get(k + ".pxq4_slabs").narrow(1, rank * per2, per2)
        w2_a[e] = get(k + ".pxq4_anchor")

    H = s13 * SLAB_COLS
    Ip = per13 * PANEL_ROWS
    print(f"shapes   : E={n_exp} H={H} Ip={Ip} (TP rank {rank}/{world}) "
          f"w13 {tuple(w13_s.shape)} w2 {tuple(w2_s.shape)}")
    return (w13_s.to(dev), w13_a.to(dev), w2_s.to(dev), w2_a.to(dev), H, Ip)


# ------------------------------------------------------------------------------ the three paths
def reference(t13, t2, x, ids, wts, w13_s, w13_a, w2_s, w2_a, H, Ip, top_k):
    """The shipped unfused ops. Returns (act, torch-fold, ascending-fold).

    `shipped` is moe.py's line verbatim and is the gate. `ascending` is kept only so the report
    can still quote the size of the difference the old order used to make.
    """
    M = x.shape[0]
    S = ids.numel()
    xg = x.unsqueeze(1).expand(M, top_k, H).reshape(S, H).contiguous()
    gu = torch.empty((S, 2 * Ip), dtype=torch.float16, device=x.device)
    getattr(torch.ops.pxq4, TIER_OPS[t13]["moe_mmv"])(gu, xg, ids, w13_s, w13_a)
    act = (torch.nn.functional.silu(gu[:, :Ip]) * gu[:, Ip:]).contiguous()
    dn = torch.empty((S, H), dtype=torch.float16, device=x.device)
    getattr(torch.ops.pxq4, TIER_OPS[t2]["moe_mmv"])(dn, act, ids, w2_s, w2_a)
    d3 = dn.view(M, top_k, H)
    w3 = wts.view(M, top_k)
    folded = torch.zeros((M, H), dtype=torch.float32, device=x.device)
    for j in range(top_k):
        folded = folded + d3[:, j, :].to(torch.float32) * w3[:, j:j + 1]
    shipped = (d3.to(torch.float32) * w3.unsqueeze(-1)).sum(dim=1)
    return act, shipped.to(torch.float16), folded.to(torch.float16)


def fused(t13, t2, x, ids, wts, w13_s, w13_a, w2_s, w2_a, H, Ip, top_k, form):
    M = x.shape[0]
    S = ids.numel()
    # poisoned, not zeroed: a kernel that fails to write a padded row must be caught, not
    # accidentally agree with a zero it never produced.
    act = torch.full((S, Ip), float("nan"), dtype=torch.float16, device=x.device)
    out = torch.full((M, H), float("nan"), dtype=torch.float16, device=x.device)
    getattr(torch.ops.pxq4, TIER_OPS[t13]["gateup"])(act, x, ids, w13_s, w13_a, top_k)
    if form == "A":
        getattr(torch.ops.pxq4, TIER_OPS[t2]["down_fold"])(out, act, ids, wts, w2_s, w2_a, top_k)
    else:
        dn = torch.full((S, H), float("nan"), dtype=torch.float16, device=x.device)
        getattr(torch.ops.pxq4, TIER_OPS[t2]["down_part"])(dn, act, ids, w2_s, w2_a)
        torch.ops.pxq4.moe_slot_fold_out(out, dn, wts, top_k)
    return act, out


# ------------------------------------------------------------------------------------ compare
def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def gate_equal(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if bits(a).equal(bits(b)):
        print(f"  PASS  {name}: bit-identical ({a.numel()} elements)")
        return
    diff = (bits(a) != bits(b))
    n = int(diff.sum())
    af, bf = a.float(), b.float()
    worst = int((af - bf).abs().argmax())
    die(f"{name}: {n}/{a.numel()} elements differ; worst at flat {worst} "
        f"({af.flatten()[worst].item()} vs {bf.flatten()[worst].item()})")


def report_vs_shipped(a: torch.Tensor, b: torch.Tensor) -> None:
    ab, bb = bits(a).to(torch.int32), bits(b).to(torch.int32)
    ne = int((ab != bb).sum())
    ulp = int((ab - bb).abs().max()) if ne else 0
    mad = float((a.float() - b.float()).abs().max())
    print(f"  INFO  vs the OLD ascending fold: {ne}/{a.numel()} elements differ, "
          f"max |delta| {mad:.3e}, max ULP {ulp} "
          f"(this is the difference that cost the 20-prompt gate 7/20 before the fold order "
          f"was matched to torch's)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--M", default="1,2,4,8")
    ap.add_argument("--top-k", type=int, default=0, help="0 = read it from the config")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        die("no CUDA device")
    dev = torch.device("cuda")
    print(f"device   : {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")

    load_lib()
    cfg = read_cfg(args.model)
    t13, t2 = tiers_of(cfg)
    print(f"tiers    : w13 {t13} slab {TIER_SLAB[t13]} | w2 {t2} slab {TIER_SLAB[t2]}")
    upload_books(cfg, {t13, t2})

    txt = cfg.get("text_config", cfg)
    top_k = args.top_k or int(txt.get("num_experts_per_tok", 8))
    w13_s, w13_a, w2_s, w2_a, H, Ip = stack_experts(
        args.model, args.layer, args.experts, args.rank, args.tp, dev)
    E = w13_s.shape[0]

    torch.manual_seed(args.seed)
    failures = 0
    for M in [int(v) for v in args.M.split(",") if v.strip()]:
        S = M * top_k
        print(f"\n--- M={M} top_k={top_k} S={S} ---")
        x = (torch.randn((M, H), device=dev) * 0.5).to(torch.float16)
        ids = torch.randint(0, E, (S,), dtype=torch.int32, device=dev)
        # vLLM emits id < 0 for an unrouted slot; make sure at least one exists, and put it in
        # the middle of a token's slots rather than at the end where an off-by-one would hide.
        if S >= 3:
            ids[1] = -1
        wts = torch.rand((S,), dtype=torch.float32, device=dev)

        ref_act, ref_out, ascending = reference(
            t13, t2, x, ids, wts, w13_s, w13_a, w2_s, w2_a, H, Ip, top_k)
        a_act, a_out = fused(t13, t2, x, ids, wts, w13_s, w13_a, w2_s, w2_a, H, Ip, top_k, "A")
        b_act, b_out = fused(t13, t2, x, ids, wts, w13_s, w13_a, w2_s, w2_a, H, Ip, top_k, "B")
        torch.cuda.synchronize()

        try:
            gate_equal("gate 1  act   (fused gateup+SwiGLU vs unfused mmv+silu+mul)",
                       a_act, ref_act)
            gate_equal("gate 2  out   (form A vs the SHIPPED torch sum(dim=1) fold)", a_out, ref_out)
            gate_equal("gate 3  out   (form B vs form A)", b_out, a_out)
            gate_equal("gate 4  act   (form B gateup vs form A gateup)", b_act, a_act)
            if torch.isnan(a_out).any() or torch.isnan(a_act).any():
                die("a NaN survived: the kernel left a poisoned element unwritten")
            print("  PASS  gate 5  padding: no poisoned element survived (every row written)")
            report_vs_shipped(a_out, ascending)
        except SystemExit:
            failures += 1
            raise

    print("\nALL GATES PASSED" if not failures else "\nFAILURES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
