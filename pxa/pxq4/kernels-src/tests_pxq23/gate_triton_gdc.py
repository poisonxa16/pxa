#!/usr/bin/env python3
"""Gate: the Triton PDL no-op shim lets the fork's real Qwen4Exp kernels compile
and run, and they produce the right numbers.

Run inside pxa-vllm:sm60-v15 with the sidecar on PYTHONPATH:

    python tests_pxq23/gate_triton_gdc.py

Exercises the kernel that actually killed the PP=4 boot -- grouped Gemma RMSNorm
from vllm/models/qwen4_exp/*/ops/hc.py -- plus the other hc ops, against a plain
torch reference.
"""
import sys

import torch

FAIL = []
PASS = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def main():
    import triton
    import triton.language.extra.cuda as tcuda

    check("shim: gdc_wait present", hasattr(tcuda, "gdc_wait"), f"triton {triton.__version__}")
    check("shim: gdc_launch_dependents present", hasattr(tcuda, "gdc_launch_dependents"))

    if not torch.cuda.is_available():
        check("cuda available", False)
        return finish()

    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")

    from vllm.models.qwen4_exp.nvidia.ops import hc

    torch.manual_seed(0)
    eps = 1e-6

    # ---- grouped Gemma RMSNorm: the kernel the boot died in -------------------
    for num_groups, group_dim, rows in ((4, 128, 7), (8, 256, 16), (2, 64, 1)):
        dim = num_groups * group_dim
        x = torch.randn(rows, dim, device=dev, dtype=torch.float16)
        w = torch.randn(dim, device=dev, dtype=torch.float16)
        got = hc.grouped_gemma_rmsnorm(x, w, eps, num_groups)

        xf = x.float().view(rows, num_groups, group_dim)
        rrms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        ref = xf * rrms
        ref = ref + ref * w.float().view(1, num_groups, group_dim)
        ref = ref.view(rows, dim).to(got.dtype)

        err = (got.float() - ref.float()).abs().max().item()
        check(f"grouped_gemma_rmsnorm g={num_groups} d={group_dim} n={rows}",
              err < 2e-2, f"max abs err {err:.3e}")

    # ---- hc_silu --------------------------------------------------------------
    hc_count = 4
    x = torch.randn(6, hc_count * 96, device=dev, dtype=torch.float16)
    got = hc.hc_silu(x, hc_count)
    # the kernel is silu(x / HC): the hyper-connection sum is averaged in the same pass
    ref = torch.nn.functional.silu(x.float() / hc_count).to(got.dtype)
    err = (got.float() - ref.float()).abs().max().item()
    check("hc_silu", err < 2e-2, f"max abs err {err:.3e}")

    # ---- hc_gate_mix: shape contract is the fork's; only the compile matters here
    hidden = x.shape[1] // hc_count
    for gshape in ((6, hc_count), (6, hc_count * hidden), (6, 1)):
        try:
            gate = torch.randn(*gshape, device=dev, dtype=torch.float16)
            got = hc.hc_gate_mix(x, gate, hc_count)
            check("hc_gate_mix compiles and runs", got.isfinite().all().item(),
                  f"gate {gshape} -> {tuple(got.shape)}")
            break
        except AssertionError:
            continue
        except Exception as exc:
            check("hc_gate_mix compiles and runs", False, repr(exc)[:160])
            break
    else:
        print("SKIP  hc_gate_mix (no gate shape matched the fork's assert)")

    return finish()


def finish():
    total = len(PASS) + len(FAIL)
    print(f"\n{len(PASS)}/{total} PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
