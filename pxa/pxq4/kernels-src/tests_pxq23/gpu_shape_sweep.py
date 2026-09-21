"""gpu_shape_sweep.py -- the split family against the monolithic mmv on EVERY DISTINCT SHAPE
a given checkpoint actually contains, for one tier.

WHY THIS EXISTS SEPARATELY FROM gpu_split_gate.py. That gate proves the kernels on synthetic
panels and on ONE real tensor. This one asks a narrower and, when a file misbehaves, more
useful question: does any shape in THIS file behave differently? It enumerates the distinct
(panels, kslabs) pairs among the file's tensors of the named tier, loads one representative of
each, and runs the same bit-exact comparison across the whole dispatchable M range plus a
captured-graph replay. A file with five distinct linear shapes is five chances for a
shape-dependent defect that a single-tensor gate would miss.

The comparison is max-abs-diff 0 in the fp16 encoding. It is not a tolerance: the split family
preserves the monolithic fold exactly, so any difference is a defect.

  PXQ4_LIB=/out/libpxq_sm60_v18.so python3 gpu_shape_sweep.py \\
      --ckpt /models/qwen38-27b-pxq2-attn4-vllm-p2a --tier pxq2
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import torch

TIER_ID = {"pxq2": 254, "pxq3": 255}
M_VALUES = list(range(1, 17))
M_CAPTURE = [1, 2, 4, 8]


def op(tier: str, which: str):
    return getattr(torch.ops.pxq4, f"{tier}_{which}")


def first_diff(got, ref):
    d = (got.view(torch.int16).to(torch.int32) - ref.view(torch.int16).to(torch.int32)).abs()
    n = int((d != 0).sum())
    where = torch.nonzero(d.reshape(-1), as_tuple=False)
    i = int(where[0, 0]) if where.numel() else -1
    return (f"{n} of {d.numel()} elements differ, max |ULP| {int(d.max())}, first at flat "
            f"index {i}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tier", default="pxq2", choices=sorted(TIER_ID))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    path = os.environ.get("PXQ4_LIB")
    if not path:
        raise SystemExit("set PXQ4_LIB")
    torch.ops.load_library(path)
    print(f"loaded {path}, pxq_version={int(torch.ops.pxq4.pxq_version())}")
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"device: {torch.cuda.get_device_name(dev)} (sm_{cap[0]}{cap[1]})")

    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))["quantization_config"]
    tiers = cfg.get("pxq_tiers") or {}
    idx = json.load(open(os.path.join(args.ckpt, "model.safetensors.index.json")))["weight_map"]
    books = cfg.get("tier_books") or {}
    if args.tier in books:
        torch.ops.pxq4.pxq_set_book(TIER_ID[args.tier],
                                    torch.tensor(books[args.tier], dtype=torch.float32))
    sub = cfg.get("tier_sub") or cfg.get("sub")
    if sub:
        torch.ops.pxq4.pxq_set_sub(torch.tensor(sub, dtype=torch.float32))

    # SELECT BY SLAB STRIDE, NOT BY NAME. The stride is what identifies a tier -- it is the
    # check the op itself enforces -- and the name map cannot be trusted here: this
    # checkpoint's config declares "mlp.gate_up_proj" while the file stores "mlp.gate_proj"
    # and "mlp.up_proj" separately, because vLLM fuses them at load. A name-based sweep
    # silently skipped 128 of the 192 tensors and reported one shape where there are three.
    from safetensors import safe_open

    STRIDE = {"pxq2": 576, "pxq3": 832}[args.tier]
    by_shard = collections.defaultdict(list)
    for key in idx:
        if key.endswith(".pxq4_slabs"):
            by_shard[idx[key]].append(key[: -len(".pxq4_slabs")])

    shapes: dict[tuple, str] = {}          # (panels, kslabs) -> representative module
    holder: dict[str, str] = {}            # module -> shard
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(args.ckpt, shard), framework="pt") as f:
            for m in names:
                sh = f.get_slice(m + ".pxq4_slabs").get_shape()
                if len(sh) != 3 or int(sh[2]) != STRIDE:
                    continue
                key = (int(sh[0]), int(sh[1]))
                holder[m] = shard
                shapes.setdefault(key, m)
    n_mods = len(holder)
    print(f"{n_mods} {args.tier} tensors (slab stride {STRIDE}) in {args.ckpt}")
    if not n_mods:
        raise SystemExit(f"no {args.tier} tensors in this checkpoint")

    reps: dict[tuple, tuple] = {}
    for key, m in sorted(shapes.items()):
        blob = load_file(os.path.join(args.ckpt, holder[m]))
        reps[key] = (m, blob[m + ".pxq4_slabs"].clone(), blob[m + ".pxq4_anchor"].clone())
        del blob

    # THE SHAPE THE KERNEL ACTUALLY SEES is not always a shape on disk. vLLM concatenates
    # gate_proj and up_proj along the output axis into one gate_up_proj parameter, so the
    # served panel count is the SUM. Synthesise it, because that fused shape is the one every
    # decode step runs and it appears nowhere in the file.
    for a, b in (("gate_proj", "up_proj"), ("q_proj", "k_proj")):
        ma = next((m for m in holder if m.endswith("." + a)), None)
        mb = ma.rsplit(".", 1)[0] + "." + b if ma else None
        if not ma or mb not in holder:
            continue
        ba = load_file(os.path.join(args.ckpt, holder[ma]))
        bb = ba if holder[mb] == holder[ma] else load_file(os.path.join(args.ckpt, holder[mb]))
        sl = torch.cat([ba[ma + ".pxq4_slabs"], bb[mb + ".pxq4_slabs"]], dim=0)
        an = torch.cat([ba[ma + ".pxq4_anchor"], bb[mb + ".pxq4_anchor"]], dim=0)
        key = (int(sl.shape[0]), int(sl.shape[1]))
        reps.setdefault(key, (f"{ma}+{b} (fused at load)", sl.clone(), an.clone()))
        del ba, bb, sl, an

    print(f"{len(reps)} distinct shapes: " +
          ", ".join(f"panels={p} kslabs={k}" for (p, k) in sorted(reps)))

    fails = 0
    for (panels, kslabs), (name, slabs_c, anchor_c) in sorted(reps.items()):
        N, K = panels * 64, kslabs * 32
        nfix = None
        slabs = slabs_c.to(dev)
        anchor = anchor_c.to(dev)
        torch.ops.pxq4.pxq_warm_split(TIER_ID[args.tier], slabs, anchor, 16)
        bad = []
        torch.manual_seed(0x51DE + panels + kslabs)
        for m in M_VALUES:
            x = (torch.randn((m, K), device=dev, dtype=torch.float32) * 0.05).to(torch.float16)
            ref = torch.empty((m, N), device=dev, dtype=torch.float16)
            op(args.tier, "mmv_mono_out")(ref, x, slabs, anchor)
            for arm, fn in (("dispatched", op(args.tier, "mmv_out")),
                            ("split2", op(args.tier, "mmv_split2_out"))):
                got = torch.full((m, N), float("nan"), device=dev, dtype=torch.float16)
                for _ in range(3):
                    fn(got, x, slabs, anchor)
                torch.cuda.synchronize()
                if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    bad.append(f"M={m} {arm}: {first_diff(got, ref)}")

        # captured-graph replay on the decode shapes
        for m in M_CAPTURE:
            x = (torch.randn((m, K), device=dev, dtype=torch.float32) * 0.05).to(torch.float16)
            ref = torch.empty((m, N), device=dev, dtype=torch.float16)
            op(args.tier, "mmv_mono_out")(ref, x, slabs, anchor)
            out = torch.empty((m, N), device=dev, dtype=torch.float16)
            op(args.tier, "mmv_out")(out, x, slabs, anchor)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                op(args.tier, "mmv_out")(out, x, slabs, anchor)
            torch.cuda.current_stream().wait_stream(s)
            try:
                with torch.cuda.graph(g):
                    op(args.tier, "mmv_out")(out, x, slabs, anchor)
                for _ in range(3):
                    out.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                        bad.append(f"M={m} graph: {first_diff(out, ref)}")
                        break
            except Exception as exc:                              # noqa: BLE001
                bad.append(f"M={m} graph capture raised {type(exc).__name__}: {exc}")
            del g

        status = "PASS" if not bad else "FAIL"
        print(f"  panels={panels:>4} kslabs={kslabs:>4} (N={N} K={K}) {status}   [{name}]")
        for b in bad:
            print(f"      {b}")
        fails += len(bad)
        del slabs, anchor
        torch.cuda.empty_cache()

    print(f"\n{'ALL SHAPES BIT-EXACT' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
