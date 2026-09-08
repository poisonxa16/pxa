#!/usr/bin/env python3
"""Bit-identity table for the ATen-exact fold, per shape.

The earlier kernel was 0-2 fp16 ULP from the native chain and failed a 20-prompt byte gate.
This one replicates at::native::setReduceConfig's geometry, so the requirement here is not
a tolerance: EVERY element of EVERY output must be equal, at every shape inside the
modelled family. A single differing element is a failure, because the whole argument for
this rewrite is that agreement is achievable rather than approachable.

Shapes outside the family fall back to the deterministic fold and are reported separately
with their ULP distance, so a regression there is visible instead of hidden.
"""
import os
import sys

import torch

torch.ops.load_library(
    os.environ.get("PXQ4_LIB", "/work/pxa/pxq4/kernels/libpxq_sm60_v13.so"))

DEV = "cuda"
EPS = 1e-6
FAIL = []


def gen(shape, dtype, seed, scale=0.05):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
            * scale).to(dtype)


def ref_add(x, res, w, eps):
    orig = x.dtype
    weight = w.float() + 1.0
    xf = x.float() + res.float() if orig is torch.float16 else x + res
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out.to(weight.dtype) * weight
    return out.to(orig), xf


def ref_plain(x, w, eps):
    orig = x.dtype
    weight = w.float() + 1.0
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out.to(weight.dtype) * weight
    return out.to(orig)


def poison(shape, dtype):
    t = torch.empty(shape, dtype=dtype, device=DEV)
    t.fill_(float("nan"))
    return t


def ulp(a, b):
    ia = a.contiguous().view(torch.int16 if a.dtype is torch.float16 else torch.int32)
    ib = b.contiguous().view(torch.int16 if b.dtype is torch.float16 else torch.int32)
    return int((ia.to(torch.int64) - ib.to(torch.int64)).abs().max())


def row(label, got, ref, exact_required, seeds):
    n = got.numel()
    bad = int((got != ref).sum())
    u = ulp(got, ref) if bad else 0
    ok = (bad == 0) if exact_required else True
    print(f"  {label:<44} {'PASS' if ok else 'FAIL'}  "
          f"exact {n - bad}/{n}  maxULP {u}  seeds {seeds}")
    if not ok:
        FAIL.append(f"{label}: {bad}/{n} differ, maxULP {u}")


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    print("pxq4 pascal-ops lib version:", int(torch.ops.pxq4.pascal_ops_version()),
          "(2 = ATen-exact fold present)")
    exact_on = os.environ.get("PXA_OPS_EXACT", "1") != "0"
    print("PXA_OPS_EXACT:", "on" if exact_on else "OFF (fallback fold)")

    # the modelled family: the widths this model normalises over, at the capture ladder
    FAMILY = [(m, h) for h in (2048, 256, 128) for m in (1, 2, 4, 8, 16)]
    OUTSIDE = [(512, 2048), (4096, 2048), (1, 5120), (8, 5120)]
    SEEDS = 6

    print(f"\n== MODELLED FAMILY -- exact equality REQUIRED (PXA_OPS_EXACT={exact_on}) ==")
    for (M, H) in FAMILY:
        gm = go = 0
        gmn = gon = 0
        gmu = gou = 0
        for s in range(SEEDS):
            x = gen((M, H), torch.float16, 100 + s * 13 + M + H)
            w = gen((H,), torch.float32, 200 + s * 13 + H, 0.02)
            res = gen((M, H), torch.float32, 300 + s * 13 + M + H)
            ro, rr = ref_add(x, res, w, EPS)
            o = poison((M, H), torch.float16)
            r2 = poison((M, H), torch.float32)
            torch.ops.pxq4.gemma_add_rms_norm_out(o, r2, x, res, w, EPS)
            gm += int((o != ro).sum()); gmn += o.numel()
            if int((o != ro).sum()):
                gmu = max(gmu, ulp(o, ro))
            xf = gen((M, H), torch.float32, 400 + s * 13 + M + H)
            po = ref_plain(xf, w, EPS)
            o2 = poison((M, H), torch.float32)
            torch.ops.pxq4.gemma_rms_norm_out(o2, xf, w, EPS)
            go += int((o2 != po).sum()); gon += o2.numel()
            if int((o2 != po).sum()):
                gou = max(gou, ulp(o2, po))
        for lbl, bad, n, u in (("gemma_add", gm, gmn, gmu), ("gemma_plain", go, gon, gou)):
            ok = bad == 0
            print(f"  {lbl:<12} [{M:5d}x{H:<5d}] {'PASS' if ok else 'FAIL'}  "
                  f"exact {n - bad}/{n}  maxULP {u}")
            if not ok:
                FAIL.append(f"{lbl} [{M}x{H}]: {bad}/{n} differ, maxULP {u}")

    print("\n== OUTSIDE THE FAMILY -- fallback fold, reported not gated ==")
    for (M, H) in OUTSIDE:
        x = gen((M, H), torch.float16, 900 + M + H)
        w = gen((H,), torch.float32, 950 + H, 0.02)
        res = gen((M, H), torch.float32, 970 + M + H)
        ro, _ = ref_add(x, res, w, EPS)
        o = poison((M, H), torch.float16)
        r2 = poison((M, H), torch.float32)
        torch.ops.pxq4.gemma_add_rms_norm_out(o, r2, x, res, w, EPS)
        bad = int((o != ro).sum())
        print(f"  gemma_add    [{M:5d}x{H:<5d}] exact {o.numel() - bad}/{o.numel()}  "
              f"maxULP {ulp(o, ro) if bad else 0}  ({1e6*bad/o.numel():.1f} ppm)")

    print("\n== ALIASING, DETERMINISM, GRAPH (exact path) ==")
    M, H = 8, 2048
    x = gen((M, H), torch.float16, 1500)
    r0 = gen((M, H), torch.float32, 1501)
    w = gen((H,), torch.float32, 1502, 0.02)
    ro, rr = ref_add(x, r0, w, EPS)
    r1 = r0.clone()
    o1 = torch.empty_like(x)
    torch.ops.pxq4.gemma_add_rms_norm_out(o1, r1, x, r1, w, EPS)
    row("aliased residual: out", o1, ro, True, 1)
    row("aliased residual: residual_out", r1, rr, True, 1)

    base = None
    spread = 0
    for _ in range(32):
        o = poison((M, H), torch.float16)
        rr2 = poison((M, H), torch.float32)
        torch.ops.pxq4.gemma_add_rms_norm_out(o, rr2, x, r0, w, EPS)
        if base is None:
            base = (o.clone(), rr2.clone())
        else:
            spread += int((o != base[0]).sum()) + int((rr2 != base[1]).sum())
    print(f"  {'determinism, 32 runs':<44} "
          f"{'PASS' if spread == 0 else 'FAIL'}  differing {spread}")
    if spread:
        FAIL.append(f"determinism: {spread}")

    o = torch.empty((M, H), dtype=torch.float16, device=DEV)
    rr3 = torch.empty((M, H), dtype=torch.float32, device=DEV)
    torch.ops.pxq4.gemma_add_rms_norm_out(o, rr3, x, r0, w, EPS)
    eager = (o.clone(), rr3.clone())
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.ops.pxq4.gemma_add_rms_norm_out(o, rr3, x, r0, w, EPS)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.pxq4.gemma_add_rms_norm_out(o, rr3, x, r0, w, EPS)
    o.fill_(float("nan")); rr3.fill_(float("nan"))
    g.replay(); torch.cuda.synchronize()
    gok = bool((o == eager[0]).all() and (rr3 == eager[1]).all())
    print(f"  {'cuda graph capture + replay == eager':<44} {'PASS' if gok else 'FAIL'}")
    if not gok:
        FAIL.append("cuda graph replay differs")

    print("\n" + "=" * 76)
    if FAIL:
        print(f"EXACT GATE: {len(FAIL)} FAILURES")
        for f in FAIL:
            print("  -", f)
        return 1
    print("EXACT GATE: ALL PASS -- bit-identical to the native chain on the whole "
          "modelled family")
    return 0


if __name__ == "__main__":
    sys.exit(main())
