"""gpu_q8_test.py -- gate the int8 per-row-scale head kernels on a GPU.

Three arms, in increasing realism:

  A  the library's built-in q8_selftest: synthetic weights, a host oracle written from the
     format spec, dequant required BIT-EXACT and the GEMV required within the same 1 ULP
     contraction allowance the PXQ arm carries. N=200 K=300 there, so both the row-block tail
     and the K tail are exercised.
  B  REAL HEAD SHAPE, 248,320 x 2,048 -- the actual LM head of the 35B, quantized by the
     converter's own q8head.quantize_head so the two halves of the format contract are tested
     against each other rather than each against itself. dequant bit-exact against numpy on a
     slice; the GEMV against dequant-then-matmul on the whole thing, as a TOLERANCE check
     because those are deliberately different summation orders.
  C  the M-routing boundary: q8_linear_out must agree with q8_mmv_out below the threshold and
     with dequant+mm above it, so a batch that crosses the boundary does not change answers
     by more than the two paths' inherent difference.

NOT MEASURED HERE: timing. Per vllm-pp, GP102 runs fp16 at 1/64 rate against GP100, so a
1080 Ti number for this kernel would be actively misleading. Everything above is exactness.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm.q8head import quantize_head        # noqa: E402


def main() -> int:
    path = os.environ.get("PXQ4_LIB")
    if not path:
        raise SystemExit("set PXQ4_LIB")
    torch.ops.load_library(path)
    if not hasattr(torch.ops.pxq4, "q8_selftest"):
        raise SystemExit(f"{path} has no q8 ops -- it predates the int8 head pack")
    dev = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(0)
    print(f"loaded {os.path.basename(path)} on {torch.cuda.get_device_name(0)} "
          f"(sm_{cap[0]}{cap[1]})")
    fails = 0

    # ---------------------------------------------------------------- A
    rc = int(torch.ops.pxq4.q8_selftest())
    print(f"A built-in q8 self-test: {'PASS' if rc == 0 else f'FAIL rc={rc}'}")
    fails += rc != 0

    # ---------------------------------------------------------------- B
    V, H = 248320, 2048          # the real head of the 35B
    rng = np.random.default_rng(11)
    # SPLIT DELIBERATELY IN TWO. Quantizing the full [248320, 2048] head on the host means a
    # 2 GB float32 array and minutes of numpy -- and it would be testing numpy, not the
    # kernels. So: the CONVERTER half is validated at real WIDTH on a slice, and the KERNEL
    # half is then exercised at the real full SHAPE (grid.x = 31,040 blocks, 509 MB of int8
    # resident) using codes generated directly, with no host float array at all.
    sl = 16384
    row_gain = np.exp(rng.normal(0.0, 0.8, size=(sl, 1))).astype(np.float32)
    w = (rng.standard_normal((sl, H)).astype(np.float32) * 0.01 * row_gain)
    w[-243:] = 0.0               # the [PAD...] tail this model's tokenizer really carries
    q_sl, s_sl, st = quantize_head(w)
    print(f"B quantize_head at real width, {sl}x{H}: wrel {st['wrel']:.5f}, "
          f"{st['zero_rows']} zero rows, {st['bpw']:.3f} bpw")

    # full real shape, codes generated directly (int8 is 509 MB, no float intermediate)
    q = np.empty((V, H), dtype=np.int8)
    q[:sl] = q_sl
    q[sl:] = rng.integers(-127, 128, size=(V - sl, H), dtype=np.int8)
    q[-243:] = 0                                       # the padded tail, codes zero
    s = np.empty((V, 1), dtype=np.float16)
    s[:sl] = s_sl
    s[sl:] = (0.0005 + rng.random((V - sl, 1)) * 0.02).astype(np.float16)
    s[-243:] = 0                                       # padded tail, scale zero
    print(f"B full head shape {V}x{H}: {q.nbytes/2**20:.0f} MiB of int8, "
          f"grid.x = {(V + 7)//8} blocks")

    d_q = torch.from_numpy(q).to(dev)
    d_s = torch.from_numpy(s.reshape(-1)).to(dev)

    # dequant, bit-exact, on the converter-produced slice: this is the arm where the two
    # halves of the format contract meet, so it is checked against numpy and not a tolerance
    rows = sl
    out = torch.empty((rows, H), dtype=torch.float16, device=dev)
    torch.ops.pxq4.q8_dequant_out(out, d_q[:rows].contiguous(), d_s[:rows].contiguous())
    want = torch.from_numpy((q[:rows].astype(np.float32)
                             * s[:rows].astype(np.float32))).to(torch.float16)
    same = torch.equal(out.cpu(), want)
    print(f"B dequant_out vs numpy on {rows} real-shape rows: "
          f"{'BIT-EXACT' if same else 'MISMATCH'}")
    fails += not same

    # GEMV vs dequant-then-matmul, whole head, tolerance (different summation orders by design)
    for M in (1, 2, 8):
        x = (torch.randn(M, H, device=dev, dtype=torch.float16) * 0.5)
        o = torch.empty((M, V), dtype=torch.float16, device=dev)
        torch.ops.pxq4.q8_mmv_out(o, x, d_q, d_s)
        ref = torch.zeros((M, V), dtype=torch.float32, device=dev)
        chunk = 32768
        for r0 in range(0, V, chunk):
            r1 = min(r0 + chunk, V)
            wd = torch.empty((r1 - r0, H), dtype=torch.float16, device=dev)
            torch.ops.pxq4.q8_dequant_out(wd, d_q[r0:r1].contiguous(), d_s[r0:r1].contiguous())
            ref[:, r0:r1] = x.float() @ wd.t().float()
        rel = ((o.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        ok = rel < 2e-3
        print(f"B mmv vs dequant+mm, M={M}: rel {rel:.2e} {'ok' if ok else 'FAIL'}")
        fails += not ok
        del ref

    # ---------------------------------------------------------------- C
    maxm = int(torch.ops.pxq4.q8_mmv_max_m())
    print(f"C routing threshold PXQ_Q8_MMV_MAX_M = {maxm}")
    for M in (maxm, maxm + 1):
        x = (torch.randn(M, H, device=dev, dtype=torch.float16) * 0.5)
        a = torch.empty((M, V), dtype=torch.float16, device=dev)
        b = torch.empty((M, V), dtype=torch.float16, device=dev)
        torch.ops.pxq4.q8_linear_out(a, x, d_q, d_s)
        torch.ops.pxq4.q8_mmv_out(b, x, d_q, d_s)
        if M <= maxm:
            same = torch.equal(a, b)
            print(f"C linear_out == mmv_out at M={M} (below/at threshold): "
                  f"{'BIT-EXACT' if same else 'MISMATCH'}")
            fails += not same
        else:
            rel = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()
            ok = rel < 2e-3
            print(f"C linear_out (dequant+cuBLAS) vs mmv_out at M={M} (above threshold): "
                  f"rel {rel:.2e} {'ok' if ok else 'FAIL'} -- expected non-zero, the two are "
                  f"different summation orders by design")
            fails += not ok

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
