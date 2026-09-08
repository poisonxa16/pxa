"""gpu_selftest.py -- prove the PXQ2/PXQ3 kernels on a GPU before any model is loaded.

Run this FIRST in a window. It takes a few hundred KB of device memory and about a second, and
it answers the only question that matters before a 12 GiB load: do the tiered kernels decode
and multiply exactly what the format says they should.

  A  library self-test        torch.ops.pxq4.pxq_selftest(tier). Inside the .so, a HOST oracle
                              written from the format spec decodes a deterministic synthetic
                              panel set and the device dequant must match it BIT-EXACTLY; the
                              mmv is then checked against a host replay of the kernel's exact
                              canonical-chunk fold, again bit-exactly. Tier 252 additionally
                              runs the TEMPLATED pxq4 instantiation against the SHIPPED pxq4
                              kernel -- the transcription gate for the whole header family.
  B  torch-level dequant      the tier's dequant_out against tiers.dequant (the numpy decode
                              already gated bit-exact against the engine's own CPU decode on
                              real model bytes), on REAL expert tensors out of the GGUF.
  C  mmv vs dequant+mm        the decode GEMV against dequant-then-cuBLAS. These are NOT
                              bit-identical by construction -- different fold orders, one fp16
                              rounding each -- so this is a tolerance check, and the tolerance
                              is stated rather than tuned: 2e-3 relative on the row norm.
  D  moe_mmv vs per-expert    the expert-indexed decode GEMV against a loop of single-expert
                              mmv calls, INCLUDING out-of-range ids (vLLM emits -1 padding
                              slots, whose rows must come back zero and never stale).
                              Bit-exact by construction: same fold on the same bytes.

Usage inside the window (one GPU, no server):
    PXQ4_LIB=/work/kernels/libpxq_sm70_v13.so python3 gpu_selftest.py --tier pxq2 --tier pxq3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import gguf_raw as G          # noqa: E402
from gguf_to_vllm import tiers as T             # noqa: E402

GGUF = {"pxq2": os.environ.get("PXA_MODELS_COLD", "./models") + "/fusion35bv2/fusion2-35b-PXQ2.gguf",
        "pxq3": os.environ.get("PXA_MODELS_COLD", "./models") + "/fusion35bv2/fusion2-35b-PXQ3.gguf"}


# The converter-side Tier (gguf_to_vllm.tiers) carries geometry only; the op names live on the
# RUNTIME-side Tier (pxq4_vllm.tiers), which this script does not import because it must run
# without vLLM on the path. Derive them here from the one rule that defines them: pxq4 keeps
# the frozen unprefixed op names, every other tier gets its own prefixed family.
def op_name(tier_name: str, which: str) -> str:
    base = {"dequant": "dequant_out", "mmv": "mmv_out", "moe_mmv": "moe_mmv_out",
            "linear": "linear_out"}[which]
    return base if tier_name == "pxq4" else f"{tier_name}_{base}"


def op(tier_name: str, which: str):
    name = op_name(tier_name, which)
    if not hasattr(torch.ops.pxq4, name):
        raise SystemExit(f"the loaded library has no torch.ops.pxq4.{name}")
    return getattr(torch.ops.pxq4, name)


def load_lib() -> None:
    path = os.environ.get("PXQ4_LIB")
    if not path:
        raise SystemExit("set PXQ4_LIB to the libpxq_*_v13.so you want to test")
    torch.ops.load_library(path)
    if not hasattr(torch.ops.pxq4, "pxq_selftest"):
        raise SystemExit(f"{path} has no pxq_selftest -- it predates tier support")
    print(f"loaded {path}, pxq_version={int(torch.ops.pxq4.pxq_version())}")


def real_tensor(tier_name: str, e: int = 0):
    """One real expert of one real layer, as (slabs, anchor, book, sub, N, K)."""
    tid = T.BY_NAME[tier_name].type_id
    gg = G.GGUFFile(GGUF[tier_name])
    name = "blk.0.ffn_gate_exps.weight"
    ti = gg.tensors[name]
    K, N = ti.dims[0], ti.dims[1]
    per = T.tensor_bytes(tid, N, K)
    blob = bytes(gg.raw(name)[e * per:(e + 1) * per])
    book = np.asarray(gg.kv[f"pxa.{tier_name}.book"], dtype=np.float32)
    sub = np.asarray(gg.kv[f"pxa.{tier_name}.sub"], dtype=np.float32)
    gg.close()
    slabs, anchor = T.split_blob(blob, tid, N, K)
    return slabs, anchor, book, sub, N, K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", action="append", default=None,
                    help="pxq2 / pxq3 / pxq4 (repeatable); default pxq4 pxq2 pxq3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--experts", type=int, default=4, help="experts in the MoE gate (D)")
    ap.add_argument("--skip-selftest", action="store_true",
                    help="skip arm A (the library's built-in host-oracle check) and run the "
                         "device-vs-device arms B/C/D only. Use when arm A is failing for a "
                         "reason you have already isolated to the HOST replay -- B/C/D are "
                         "what say whether the kernels themselves are right.")
    args = ap.parse_args()
    tiers = args.tier or ["pxq4", "pxq2", "pxq3"]

    load_lib()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    print(f"device: {torch.cuda.get_device_name(dev)} "
          f"(sm_{torch.cuda.get_device_capability(dev)[0]}{torch.cuda.get_device_capability(dev)[1]})")
    fails = 0

    for tname in tiers:
        tier = T.BY_NAME[tname]
        print(f"\n=== {tname} (id {tier.type_id}, slab {tier.slab_bytes} B) ===")

        # ---------------------------------------------------------------- A
        if args.skip_selftest:
            print("A library self-test: SKIPPED (--skip-selftest)")
        else:
            rc = int(torch.ops.pxq4.pxq_selftest(tier.type_id))
            print(f"A library self-test: {'PASS' if rc == 0 else f'FAIL rc={rc}'}")
            if rc:
                fails += 1
                continue
        if tname == "pxq4":
            continue          # B-D use the tiered ops, which do not serve 252

        slabs, anchor, book, sub, N, K = real_tensor(tname)
        # honour the FILE's tables, exactly as the runtime does
        torch.ops.pxq4.pxq_set_book(tier.type_id, torch.as_tensor(book))
        torch.ops.pxq4.pxq_set_sub(torch.as_tensor(sub))
        d_slabs = torch.from_numpy(np.ascontiguousarray(slabs)).to(dev)
        d_anchor = torch.from_numpy(np.ascontiguousarray(anchor)).to(dev)

        # ---------------------------------------------------------------- B
        want = T.dequant(slabs, anchor, tier.type_id, book, sub)
        want16 = torch.from_numpy(want).to(torch.float16)
        got = torch.empty((N, K), dtype=torch.float16, device=dev)
        op(tname, 'dequant')(got, d_slabs, d_anchor)
        same = torch.equal(got.cpu(), want16)
        print(f"B dequant_out vs the numpy decode on a REAL expert "
              f"(N={N} K={K}): {'BIT-EXACT' if same else 'MISMATCH'}")
        if not same:
            d = (got.cpu().float() - want16.float()).abs()
            print(f"   max abs diff {d.max().item()} at {int(d.argmax())}")
            fails += 1

        # ---------------------------------------------------------------- C
        for M in (1, 2, 4, 8):
            x = (torch.randn(M, K, device=dev, dtype=torch.float16) * 0.5)
            o_mmv = torch.empty(M, N, dtype=torch.float16, device=dev)
            op(tname, 'mmv')(o_mmv, x, d_slabs, d_anchor)
            w = torch.empty((N, K), dtype=torch.float16, device=dev)
            op(tname, 'dequant')(w, d_slabs, d_anchor)
            o_ref = (x.float() @ w.t().float())
            rel = ((o_mmv.float() - o_ref).norm() / o_ref.norm().clamp_min(1e-9)).item()
            ok = rel < 2e-3
            print(f"C mmv vs dequant+mm, M={M}: rel {rel:.2e} {'ok' if ok else 'FAIL'}")
            if not ok:
                fails += 1

        # ---------------------------------------------------------------- D
        E = args.experts
        stack_s = torch.stack([d_slabs] * E)          # same weights, distinct expert slots:
        stack_a = torch.stack([d_anchor] * E)         # this isolates the INDEXING from the math
        # give each slot a different scale so a wrong id cannot pass unnoticed
        for e in range(E):
            stack_a[e] = d_anchor * float(1 + e)
        S = 12
        ids = torch.tensor([(-1 if i % 5 == 4 else i % E) for i in range(S)],
                           dtype=torch.int32, device=dev)
        x = (torch.randn(S, K, device=dev, dtype=torch.float16) * 0.5)
        o_moe = torch.empty(S, N, dtype=torch.float16, device=dev)
        op(tname, 'moe_mmv')(o_moe, x, ids, stack_s, stack_a)
        o_ref = torch.zeros(S, N, dtype=torch.float16, device=dev)
        for i in range(S):
            e = int(ids[i])
            if e < 0:
                continue                              # padding slot must stay zero
            row = torch.empty(1, N, dtype=torch.float16, device=dev)
            op(tname, 'mmv')(
                row, x[i:i + 1].contiguous(), stack_s[e].contiguous(), stack_a[e].contiguous())
            o_ref[i] = row[0]
        same = torch.equal(o_moe, o_ref)
        npad = int((ids < 0).sum())
        zeros_ok = bool((o_moe[ids < 0] == 0).all())
        print(f"D moe_mmv vs per-expert mmv, S={S} E={E} ({npad} padding slots): "
              f"{'BIT-EXACT' if same else 'MISMATCH'}; padding rows zero: {zeros_ok}")
        if not (same and zeros_ok):
            fails += 1

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
