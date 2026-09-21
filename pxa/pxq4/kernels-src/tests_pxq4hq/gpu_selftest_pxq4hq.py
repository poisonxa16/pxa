"""gpu_selftest_pxq4hq.py -- prove the PXQ4HQ kernels on a GPU before any model is loaded.

Run this FIRST in a window. It takes a few hundred MB of device memory and about a second, and
it answers the only question that matters before a 16 GiB load: do the tier-253 kernels decode
and multiply exactly what the format says they should.

  A  library self-test        torch.ops.pxq4.pxq4hq_selftest(). Inside the .so, a HOST oracle
                              written from the format spec decodes a deterministic synthetic
                              panel set and the device dequant must match it BIT-EXACTLY; the
                              mmv is then checked against a host replay of the kernel's exact
                              canonical-chunk fold, to 1 ULP of fp16 (see the note in
                              pxq4hq_kernel.cu for why that arm is 1 ULP and not bit-exact).
  B  torch-level dequant      pxq4hq_dequant_out against gguf_to_vllm.tiers.dequant on REAL
                              tensors out of a real PXQ4HQ GGUF. The numpy decode is itself
                              already gated BIT-EXACT against the engine's own CPU decode
                              function, extracted verbatim (tests_pxq4hq/build_oracle_pxq4hq.sh),
                              so this closes the chain GGUF bytes -> engine C -> numpy -> CUDA.
                              Bit-exact: the dequant path performs no accumulation.
  C  mmv vs dequant+mm        the decode GEMV against dequant-then-cuBLAS. NOT bit-identical by
                              construction -- different fold orders, one fp16 rounding each --
                              so this is a tolerance check, and the tolerance is stated rather
                              than tuned: 2e-3 relative on the row norm, the same figure the
                              pxq2/pxq3 gate uses.
  D  mma vs mmv               the sm_70 tensor-core arena path against the SIMT mmv at
                              M = 1..16. Also not bit-identical by construction (the arena
                              rounds eff*book to fp16 before the HMMA, the SIMT kernel keeps
                              fp32 to the end), and the same 2e-3 tolerance applies. This is
                              the arm that would catch a wrong scale-byte address in the
                              tensor-core kernel, which no shape check can see.
  E  tier confusion           a PXQ4 (1088-byte stride) weight offered to a pxq4hq op must
                              RAISE, not decode. The two tiers share a book and a code layout,
                              so this is the one pair that could produce plausible garbage.
  F  table round trip         pxq4hq_set_tables -> pxq4hq_get_book / _get_sub returns what was
                              uploaded, and the tier's default sub is NOT the shared SUB16.

Usage inside the window (one GPU, no server):
    PXQ4_LIB=/work/kernels/libpxq_sm70_v15.so python3 gpu_selftest_pxq4hq.py \
        --gguf /path/to/a-PXQ4HQ.gguf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import gguf_raw as G          # noqa: E402
from gguf_to_vllm import reference as R         # noqa: E402
from gguf_to_vllm import tiers as T             # noqa: E402

TOL = 2e-3          # relative on the row norm, for the two arms that are not bit-exact


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.float(), b.float()
    n = torch.linalg.vector_norm(b32, dim=-1).clamp_min(1e-6)
    return float((torch.linalg.vector_norm(a32 - b32, dim=-1) / n).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.environ.get("PXA_PXQ4HQ_GGUF"))
    ap.add_argument("--tensors", type=int, default=3)
    args = ap.parse_args()

    lib = os.environ.get("PXQ4_LIB")
    if not lib:
        print("PXQ4_LIB must name the kernel library explicitly")
        return 2
    torch.ops.load_library(lib)
    dev = torch.device("cuda")
    print(f"library: {lib}")
    print(f"device : {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")

    fails: list[str] = []

    # ---------------------------------------------------------------- A: library self-test
    for op in ("pxq4hq_selftest", "pxq4hq_dequant_out", "pxq4hq_mmv_out", "pxq4hq_mma_out",
               "pxq4hq_linear_out", "pxq4hq_set_tables"):
        if not hasattr(torch.ops.pxq4, op):
            fails.append(f"A: the library has no torch.ops.pxq4.{op}")
    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    rc = int(torch.ops.pxq4.pxq4hq_selftest())
    print(f"  A library self-test         : {'PASS' if rc == 0 else 'FAIL rc=' + str(rc)}")
    if rc != 0:
        fails.append(f"A: pxq4hq_selftest returned {rc}")

    # ---------------------------------------------------------------- F: tables
    book_d = torch.ops.pxq4.pxq4hq_get_book().cpu().numpy()
    sub_d = torch.ops.pxq4.pxq4hq_get_sub().cpu().numpy()
    if not np.array_equal(book_d, np.asarray(R.BOOK)):
        fails.append("F: the library's default pxq4hq book is not the frozen PX16 book")
    if not np.array_equal(sub_d, np.asarray(T.sub_of(T.PXQ4HQ))):
        fails.append("F: the library's default pxq4hq sub is not the frozen SUB8 table")
    if np.array_equal(sub_d, np.asarray(R.SUB)):
        fails.append("F: the library's pxq4hq sub IS the shared SUB16 -- wrong table wired")
    torch.ops.pxq4.pxq4hq_set_tables(torch.tensor(R.BOOK.tolist(), dtype=torch.float32),
                                     torch.tensor(T.sub_of(T.PXQ4HQ).tolist(), dtype=torch.float32))
    if not np.array_equal(torch.ops.pxq4.pxq4hq_get_sub().cpu().numpy(),
                          np.asarray(T.sub_of(T.PXQ4HQ))):
        fails.append("F: set_tables -> get_sub did not round trip")
    print(f"  F table round trip          : {'PASS' if not fails else 'see below'}")

    if not args.gguf:
        print("  B/C/D/E skipped (no --gguf)")
        return 1 if fails else 0

    gg = G.GGUFFile(args.gguf)
    names = [n for n, ti in gg.tensors.items() if ti.type_id == G.GGML_PXQ4HQ][:args.tensors]
    if not names:
        print(f"FAIL {args.gguf} carries no pxq4hq tensors")
        return 1
    book, sub = R.BOOK, T.sub_of(T.PXQ4HQ)

    for n in names:
        ti = gg.tensors[n]
        N, K = ti.ne1, ti.ne0
        slabs_np, anchor_np = T.split_blob(bytes(gg.raw(n)), T.PXQ4HQ, N, K)
        slabs = torch.from_numpy(np.ascontiguousarray(slabs_np)).to(dev)
        anchor = torch.from_numpy(np.ascontiguousarray(anchor_np)).to(dev)

        # ---- B: dequant, bit-exact against the numpy decode
        w = torch.empty((N, K), dtype=torch.float16, device=dev)
        torch.ops.pxq4.pxq4hq_dequant_out(w, slabs, anchor)
        ref = torch.from_numpy(T.dequant(slabs_np, anchor_np, T.PXQ4HQ, book, sub)).to(torch.float16)
        if not torch.equal(w.cpu().view(torch.int16), ref.view(torch.int16)):
            d = int((w.cpu() != ref).sum())
            fails.append(f"B {n}: {d}/{N*K} fp16 values differ from the numpy decode")

        # ---- C: mmv vs dequant + cuBLAS
        x = (torch.randn(8, K, device=dev, dtype=torch.float16) * 0.05).contiguous()
        out = torch.empty((8, N), dtype=torch.float16, device=dev)
        torch.ops.pxq4.pxq4hq_mmv_out(out, x, slabs, anchor)
        gemm = torch.mm(x, w.t())
        e = relerr(out, gemm)
        if e > TOL:
            fails.append(f"C {n}: mmv vs dequant+mm relative error {e:.3e} > {TOL:.0e}")

        # ---- D: mma vs mmv, M = 1..16
        worst = 0.0
        if torch.cuda.get_device_capability(0)[0] >= 7:
            for M in (1, 2, 4, 5, 8, 13, 16):
                xm = x[:1].repeat(M, 1).contiguous() if M > 8 else x[:M].contiguous()
                om = torch.empty((M, N), dtype=torch.float16, device=dev)
                torch.ops.pxq4.pxq4hq_mma_out(om, xm, slabs, anchor)
                worst = max(worst, relerr(om, torch.mm(xm, w.t())))
            if worst > TOL:
                fails.append(f"D {n}: mma vs dequant+mm relative error {worst:.3e} > {TOL:.0e}")

        # ---- E: tier confusion
        bad = torch.zeros((slabs.size(0), slabs.size(1), 1088), dtype=torch.uint8, device=dev)
        try:
            torch.ops.pxq4.pxq4hq_dequant_out(w, bad, anchor)
            fails.append(f"E {n}: a 1088-byte-stride weight was accepted by pxq4hq_dequant_out")
        except RuntimeError:
            pass

        print(f"    {n:44s} [{N} x {K}]  B bit-exact  C {e:.2e}  D {worst:.2e}")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print("\nALL GPU GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
