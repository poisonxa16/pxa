# SPDX-License-Identifier: Apache-2.0
"""Prove the vllm.ir provider seam end to end, with no kernel in hand.

If this passes, then when a fused sm_60 norm kernel arrives the only thing left to trust
is the kernel: the registration, the schema match, the arg gate, the dispatch and the
fallback are all already known to work. If it fails, we find out now rather than inside
a measurement window.
"""
from __future__ import annotations
import os, sys
import torch

os.environ.setdefault("PXA_FUSED_NORM", "selftest")
sys.path.insert(0, os.environ.get(
    "PXQ4_SITE", os.environ.get("PXA_PP_ROOT", ".") + "/pxa/pxq4/sidecar/site-sm60"))

H = 2048


def main() -> int:
    from vllm import ir
    import pxq4_vllm.pxa_ir_norm as irn

    print("before:", "fused priority", ir.ops.fused_add_rms_norm.get_priority(),
          "| rms priority", ir.ops.rms_norm.get_priority())
    irn.maybe_patch()
    pf = ir.ops.fused_add_rms_norm.get_priority()
    pp = ir.ops.rms_norm.get_priority()
    print("after: ", "fused priority", pf, "| rms priority", pp)
    if pf[:1] != ["pxa"] or pp[:1] != ["pxa"]:
        print("SEAM FAIL: our provider is not first in the priority list")
        return 2

    dev = "cuda"
    w = torch.randn(H, device=dev, dtype=torch.float16)
    fails = 0

    # 1. supported shape must route to us AND match native bit for bit.
    for M in (1, 2, 4, 8):
        x = torch.randn(M, H, device=dev, dtype=torch.float16)
        r = torch.randn(M, H, device=dev, dtype=torch.float16)
        o, res = ir.ops.fused_add_rms_norm(x, r, w, 1e-6, None)
        n_o, n_res = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(x, r, w, 1e-6, None)
        ok = torch.equal(o, n_o) and torch.equal(res, n_res)
        o2 = ir.ops.rms_norm(x, w, 1e-6, None)
        n_o2 = ir.ops.rms_norm.impls["native"].impl_fn(x, w, 1e-6, None)
        ok2 = torch.equal(o2, n_o2)
        print(f"  M={M}: fused_add bit-identical {ok} | rms bit-identical {ok2}")
        fails += (not ok) + (not ok2)

    # 2. an UNSUPPORTED shape must fall through to native rather than break.
    xb = torch.randn(2, H, device=dev, dtype=torch.float32)
    wb = torch.randn(H, device=dev, dtype=torch.float32)
    try:
        ob = ir.ops.rms_norm(xb, wb, 1e-6, None)
        print("  fp32 input falls through to native:", ob.dtype == torch.float32)
    except Exception as e:
        print("  *** fp32 fallthrough FAILED:", str(e)[:120]); fails += 1
    try:
        xv = torch.randn(2, H, device=dev, dtype=torch.float16)
        ov = ir.ops.rms_norm(xv, w, 1e-6, 512)
        print("  variance_size override falls through to native:", tuple(ov.shape) == (2, H))
    except Exception as e:
        print("  *** variance_size fallthrough FAILED:", str(e)[:120]); fails += 1

    # 3. it must survive graph capture, because that is where it will actually run.
    x = torch.randn(1, H, device=dev, dtype=torch.float16)
    r = torch.randn(1, H, device=dev, dtype=torch.float16)
    try:
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                ir.ops.fused_add_rms_norm(x, r, w, 1e-6, None)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ir.ops.fused_add_rms_norm(x, r, w, 1e-6, None)
        g.replay(); torch.cuda.synchronize()
        print("  captures and replays inside a CUDA graph: True")
    except Exception as e:
        print("  *** graph capture FAILED:", str(e)[:160]); fails += 1

    print("\n  SEAM", "PASS" if fails == 0 else "FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
