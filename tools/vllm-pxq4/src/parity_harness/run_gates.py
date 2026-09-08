"""
run_gates.py -- one command, one verdict.

    python -m parity_harness.run_gates                     # CPU gates only
    python -m parity_harness.run_gates --real fx.npz       # + real tensors from the GGUF
    python -m parity_harness.run_gates --gpu               # also G6/G8 (needs a GPU)

Deliberately does NOT require pytest: these gates have to be runnable on the DGX inside a
throwaway container with nothing installed but numpy, and inside the production container
(read-only, disk 100% full) where installing anything is impossible.  The test modules are
plain functions with asserts, so pytest can collect them too if it happens to be present.

Exit code 0 iff every non-skipped gate passed.  A SKIP is never counted as a pass, and the
summary prints why each skip happened -- a harness that quietly skips its ground-truth leg
is worse than no harness.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from . import adapters, cref_bridge, fixtures
from . import oracle as O
from . import test_a_dequant as TA
from . import test_b_linear as TB
from . import test_c_shard as TC
from . import test_d_ops_abi as TD
from . import test_e_mutation as TE
from . import test_f_hostsim as TF
from .test_a_dequant import Skip

# (gate id, human label, callable, needs_gpu)
GATES = [
    ("G1a", "dequant: production C  ==  harness oracle", TA.test_g1_c_vs_oracle, False),
    ("G1b", "dequant: harness oracle ==  pxq4_vllm.reference", TA.test_g1_oracle_vs_agent_a, False),
    ("G2a", "layout: split/join byte round-trip", TA.test_g2_split_join_roundtrip, False),
    ("G2b", "layout: geometry gate refuses misalignment", TA.test_g2_geometry_gate_rejects_misalignment, False),
    ("G2c", "layout: real shapes reproduce on-disk byte sizes", TA.test_g2_real_shapes_byte_sizes, False),
    ("G3a", "shard: column split is bit-exact", TC.test_g3_column_shard_bitexact, False),
    ("G3b", "shard: row (K) split is bit-exact", TC.test_g3_row_shard_bitexact, False),
    ("G3c", "shard: blob-space == emitted-space", TC.test_g3_blob_and_emitted_shards_agree, False),
    ("G3d", "shard: merged-column assembly", TC.test_g3_merged_column_assembly, False),
    ("G3e", "shard: qkv / attn_q head-pair arithmetic", TC.test_g3_qkv_shard_arithmetic, False),
    ("G3f", "shard: every real module aligns at TP 1/2/4", TC.test_g3_real_module_alignment, False),
    ("G3g", "shard: fused-GDN-ba trap is detected", TC.test_g3_fused_gdn_ba_is_a_trap, False),
    ("G3h", "shard: misalignment refused, not truncated", TC.test_g3_misalignment_is_refused_not_truncated, False),
    ("G3i", "shard: header-duplication overhead", TC.test_g3_row_shard_header_overhead, False),
    ("G3j", "shard: row-parallel all-reduce tolerance", TC.test_g3_row_parallel_allreduce_tolerance, False),
    ("N1", "negative control: dequant mutations are caught", TE.test_negative_control_dequant_mutations, False),
    ("N2", "PROOF: multiply reassociation is a no-op (plan §6.3 correction)", TE.test_reassociation_is_provably_a_noop, False),
    ("N3", "negative control: naive row-contiguous K shard is caught", TE.test_negative_control_naive_row_shard_is_wrong, False),
    ("N4", "negative control: K-narrowed anchor is caught", TE.test_negative_control_anchor_sharded_on_k, False),
    ("N5", "negative control: off-by-one panel is caught", TE.test_negative_control_offset_by_one_panel, False),
    ("N6", "negative control: mmv fold order is load-bearing", TE.test_negative_control_mmv_fold_matters, False),
    ("Gb1", "linear: mmv fold vs exact GEMM", TB.test_b_cpu_mmv_model_vs_exact, False),
    ("Gb2", "linear: canon_nfix fold is pinned", TB.test_b_cpu_fold_is_deterministic, False),
    ("Gb3", "linear: fp16 store dominates the error", TB.test_b_cpu_fp16_output_headroom, False),
    ("H1", "hostsim: kernel TU tables == ggml-pxq6-tables.h", TF.test_h1_hostsim_tables_match, False),
    ("H2", "hostsim: KERNEL dequant f32 == oracle (bit-exact)", TF.test_h2_hostsim_dequant_f32_bitexact, False),
    ("H3", "hostsim: KERNEL dequant f16 is one RNE of the f32", TF.test_h3_hostsim_dequant_f16_rounding, False),
    ("H4", "hostsim: KERNEL shard invariant, both axes, TP 1/2/4", TF.test_h4_hostsim_shard_invariant, False),
    ("H5", "hostsim: KERNEL mmv == bit-exact fold model", TF.test_h5_hostsim_mmv_matches_fold_model, False),
    ("H6", "hostsim: VECX arms are bit-identical", TF.test_h6_hostsim_vecx_is_bit_identical, False),
    ("H7", "hostsim: kernel canon_nfix == model", TF.test_h7_hostsim_canon_nfix_matches_model, False),
    ("G6a", "cuda: dequant_out == oracle (bit-exact)", TA.test_g6_cuda_dequant, True),
    ("G6b", "cuda: dequant_out honours the out-variant ABI", TA.test_g6_cuda_dequant_abi, True),
    ("G8a", "cuda: mmv_out == bit-exact CPU model", TB.test_g8_cuda_mmv_vs_model, True),
    ("G8b", "cuda: mmv_out == dequant+mm within tolerance", TB.test_g8_cuda_mmv_vs_deqmm, True),
    ("G8c", "cuda: apply() shape contract", TB.test_g8_cuda_apply_shape_contract, True),
    ("G8d", "cuda: schema annotates output mutation", TD.test_schema_declares_output_mutation, True),
    ("G8e", "cuda: meta/fake kernels registered", TD.test_meta_kernel_registered, True),
    ("G8f", "cuda: apply() allocates nothing", TD.test_apply_allocates_nothing, True),
    ("G8g", "cuda: CUDA-graph capture + replay", TD.test_cudagraph_capture_and_replay, True),
    ("G8h", "cuda: wide-K dynamic smem opt-in", TD.test_dequant_smem_limit_for_wide_k, True),
]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _call(fn, real, report):
    """Call a gate, passing only the kwargs it declares."""
    import inspect
    kw = {}
    params = inspect.signature(fn).parameters
    if "real" in params:
        kw["real"] = real
    if "report" in params:
        kw["report"] = report
    return fn(**kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", help=".npz written by parity_harness.extract")
    ap.add_argument("--real-dir", help="directory written by parity_harness.extract_raw")
    ap.add_argument("--gpu", action="store_true", help="also run the CUDA gates")
    ap.add_argument("--only", action="append", default=[], help="run only these gate ids")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    real = None
    if args.real and args.real_dir:
        raise SystemExit("pass --real or --real-dir, not both")
    if args.real_dir:
        real = fixtures.load_raw_dir(args.real_dir)
    elif args.real:
        real = fixtures.load_real(args.real)
    if real is not None:
        kv = real.pop("__kv__", None)
        print(f"real fixtures: {sorted(real)}")
        if kv:
            problems = O.check_tables_against_gguf(kv)
            if problems:
                print("\n!!! TABLE MISMATCH -- the artifact was NOT built with the "
                      "compiled-in tables:")
                for p in problems:
                    print("    " + p)
                print("    Every numeric gate below is meaningless until this is "
                      "resolved. See PXA_PXQ6_BOOK / PXA_PXQ6_SUB.\n")
                return 3
            print("  pxa.pxq6.book / pxa.pxq6.sub match ggml-pxq6-tables.h exactly")
            bm = kv.get("pxa.pxq.backbone_map")
            if bm:
                print(f"  backbone_rev={kv.get('pxa.pxq.backbone_rev')} map={bm}")

    print(f"\nenvironment: numpy ok | cref {'yes' if cref_bridge.available() else 'NO'} "
          f"| torch {'yes' if adapters.torch_module() else 'no'} "
          f"| cuda {'yes' if adapters.cuda_available() else 'no'} "
          f"| pxq4_vllm.reference {'yes' if adapters.ref_module() else 'no'} "
          f"| torch.ops.pxq4 {'yes' if adapters.pxq4_ops() else 'no'}\n")

    report = []
    results = []
    for gid, label, fn, needs_gpu in GATES:
        if args.only and gid not in args.only:
            continue
        if needs_gpu and not args.gpu:
            results.append((gid, label, SKIP, "not requested (--gpu)"))
            continue
        t0 = time.time()
        # Print progress as we go: some gates take seconds and a silent run is
        # indistinguishable from a hung one, which is exactly the wrong property for the
        # tool people reach for when something is already wrong.
        print(f"  .. {gid} {label}", end="", flush=True)
        try:
            _call(fn, real, report)
            results.append((gid, label, PASS, f"{time.time()-t0:.2f}s"))
            print(f"   ok {time.time()-t0:.2f}s", flush=True)
        except Skip as e:
            results.append((gid, label, SKIP, str(e)))
            print("   skip", flush=True)
        except AssertionError as e:
            results.append((gid, label, FAIL, str(e)))
            print("   FAIL", flush=True)
        except Exception:
            results.append((gid, label, FAIL, traceback.format_exc()))
            print("   FAIL", flush=True)

    width = max(len(l) for _, l, _, _ in results) if results else 40
    print("=" * (width + 22))
    for gid, label, status, detail in results:
        mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[status]
        print(f"{mark}  {gid:5s} {label:<{width}}  "
              f"{detail if status != FAIL else ''}")
    print("=" * (width + 22))

    for gid, label, status, detail in results:
        if status == FAIL:
            print(f"\n----- {gid} {label} -----\n{detail}")
        elif status == SKIP and args.verbose:
            print(f"\n----- {gid} SKIPPED: {detail}")

    if report:
        print("\nmeasurements (informational, not assertions):")
        for name, data in report:
            print(f"  {name}: {data}")

    npass = sum(1 for r in results if r[2] == PASS)
    nfail = sum(1 for r in results if r[2] == FAIL)
    nskip = sum(1 for r in results if r[2] == SKIP)
    print(f"\n{npass} passed, {nfail} failed, {nskip} skipped")
    if nskip:
        print("NOTE: a skip is not a pass. Skipped gates below are UNVERIFIED:")
        for gid, label, status, detail in results:
            if status == SKIP:
                print(f"  {gid}: {detail.splitlines()[0] if detail else ''}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
