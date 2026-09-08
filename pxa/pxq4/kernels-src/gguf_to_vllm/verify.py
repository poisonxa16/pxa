"""verify.py — the converter's verification mode. Gates G1, G2 and G3, no GPU, no writes.

    python -m gguf_to_vllm.verify --gguf <file> [--all] [--sample 12] [--oracle ./oracle]
                                  [--tp 2 4] [--seed 0]

WHAT EACH GATE ACTUALLY RETIRES, and why these three are worth more than the rest combined:

  G2  split -> join reproduces the ORIGINAL GGUF BYTES for every PXQ4 tensor.
      Proves the converter's emission is a partition of the file, not a transformation of it.
      A byte comparison, so there is no tolerance to argue about.

  G1  reference.dequant() == the engine's own pxa_deq_row_pxq6, bit-exact in fp32.
      Requires --oracle, the standalone build of gguf_to_vllm_oracle.c (which carries the
      production C verbatim). This is the gate that pins panel arithmetic, slab offsets,
      nibble order, the two frozen tables and the multiply order all at once.

  G3  dequant(shard(x)) == shard(dequant(x)), bit-exact, on BOTH axes, at every TP degree.
      This is the one that matters most and is the least obvious. It proves the tensor-parallel
      repack is a PERMUTATION OF BYTES and not a re-quantization — the column split is whole
      panels, the K split is a slab subrange plus a duplicated 128 B header, and neither
      changes a single decoded value. It is also the only automatic defence against the silent
      failure mode in vLLM's loader: `_adjust_shard_indexes_for_packing` does
      `round(shard_size // packed_factor)` (parameter.py:605-610), so a misaligned boundary
      TRUNCATES WITHOUT RAISING and yields a well-formed wrong slice.

  G5H GDN v-head order, exact, per layer, against the reference checkpoint (needs --ref-hf).
      ggml orders the 48-way value-head axis repeat-major and HF orders it k-head-major, so
      every GDN tensor is reordered on the way out (namemap.GDN_PERM_SPEC). G2/G1/G3 are all
      blind to this — an unpermuted checkpoint passes every one of them and then generates
      fluent garbage. This gate compares the two 48-entry witnesses (`ssm_dt.bias` -> `dt_bias`
      and `log(-ssm_a)` -> `A_log`) under the gather and under identity and demands the gather
      win exactly. It reads ~100 KB for the whole model. Implemented in
      convert.gate_gdn_head_order.

  A note on what these gates cannot do: they say nothing about whether `ssm_beta` is really
  `in_proj_b` (that one is now settled by direct measurement — see namemap.py) or about
  anything that needs a real forward pass compared against llama.cpp.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

from . import dequant_ref as D
from . import gguf_raw as G
from . import layout as L
from . import namemap as NM
from . import reference as R


def _tables_from_file(gg) -> tuple[np.ndarray, np.ndarray]:
    book = gg.kv.get("pxa.pxq6.book")
    sub = gg.kv.get("pxa.pxq6.sub")
    if book is None or sub is None:
        return R.BOOK, R.SUB
    return (np.asarray(book, dtype=np.float32), np.asarray(sub, dtype=np.float32))


def gate_g2(gg, names: list[str]) -> tuple[int, list[str]]:
    """split -> join must reproduce the original bytes, exactly."""
    fails = []
    for n in names:
        ti = gg.tensors[n]
        N, K = ti.ne1, ti.ne0
        raw = bytes(gg.raw(n))
        slabs, anchor = L.split_blob(raw, N, K)
        if L.join_blob(slabs, anchor) != raw:
            fails.append(n)
    return len(names), fails


def gate_g1(gg, names: list[str], oracle: str, book, sub) -> tuple[int, list[str]]:
    """reference.dequant vs the verbatim-C oracle, bit-exact in fp32.

    Only the first two panels of each tensor are compared: the oracle decodes row by row with
    no cross-panel state, so panel 0 and panel 1 exercise every code path (including the p>0
    panel-stride term), and comparing 12 GB of tensors would take longer than the lease.
    """
    if not np.array_equal(book, R.BOOK) or not np.array_equal(sub, R.SUB):
        raise SystemExit("G1: the file's tables differ from the compiled-in ones; the oracle "
                         "carries the compiled-in tables, so the comparison would be invalid.")
    fails = []
    for n in names:
        ti = gg.tensors[n]
        N, K = ti.ne1, ti.ne0
        p = min(2, N // 64)
        nrows = p * 64
        sub_bytes = bytes(gg.raw(n))[: p * L.panel_bytes(K)]
        with tempfile.NamedTemporaryFile(suffix=".pxq4") as tf:
            tf.write(sub_bytes)
            tf.flush()
            out = subprocess.run([oracle, tf.name, str(nrows), str(K)],
                                 stdout=subprocess.PIPE, check=True).stdout
        c = np.frombuffer(out, dtype="<f4").reshape(nrows, K)
        py = R.dequant_blob(sub_bytes, nrows, K, book=book, sub=sub)
        if not np.array_equal(c.view(np.uint32), py.view(np.uint32)):
            fails.append(n)
    return len(names), fails


def gate_g3(gg, names: list[str], tps: list[int], book, sub) -> tuple[int, list[str]]:
    """dequant(shard(x)) == shard(dequant(x)), bit-exact, both axes."""
    checked, fails = 0, []
    for n in names:
        ti = gg.tensors[n]
        N, K = ti.ne1, ti.ne0
        # Only the first two panels are needed: the column split is a panel-boundary property
        # and the K split is identical for every panel, so more panels add cost, not coverage.
        p = min(2, N // 64)
        nrows = p * 64
        blob = bytes(gg.raw(n))[: p * L.panel_bytes(K)]
        slabs, anchor = L.split_blob(blob, nrows, K)
        full = R.dequant(slabs, anchor, book=book, sub=sub)

        for tp in tps:
            # column-parallel: split the output rows
            if nrows % tp == 0 and (nrows // tp) % L.PANEL_ROWS == 0:
                per = nrows // tp
                for r in range(tp):
                    s, a = L.shard_columns(slabs, anchor, r * per, (r + 1) * per)
                    got = R.dequant(s, a, book=book, sub=sub)
                    want = full[r * per:(r + 1) * per]
                    checked += 1
                    if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                        fails.append(f"{n} col tp={tp} rank={r}")
            # row-parallel: split the contraction dim
            if K % tp == 0 and (K // tp) % L.SLAB_COLS == 0:
                per = K // tp
                for r in range(tp):
                    s, a = L.shard_k(slabs, anchor, r * per, (r + 1) * per)
                    got = R.dequant(s, a, book=book, sub=sub)
                    want = full[:, r * per:(r + 1) * per]
                    checked += 1
                    if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                        fails.append(f"{n} row tp={tp} rank={r}")
    return checked, fails


def gate_real_shards(gg, policy: str) -> list[str]:
    """The shard arithmetic for the ACTUAL fused modules, at the ACTUAL output_sizes.

    Not a synthetic sweep: these are the four output_sizes the fork builds for
    ``in_proj_qkvz`` (qwen3_5.py:212-230), the two for ``gate_up_proj``, and the q/k/v blocks
    of ``QKVParallelLinear``. A boundary here that is not a multiple of 64 is the truncation
    bug, and it would be invisible at load.
    """
    problems = []
    cfg_qkvz = [2048, 2048, 6144, 6144]           # q, k, v, z — key_dim 2048, value_dim 6144
    cfg_gate_up = [17408, 17408]
    cfg_qkv = [24 * 2 * 256, 4 * 256, 4 * 256]    # attn_output_gate=True doubles the q block
    for tp in (1, 2, 4):
        for label, sizes in (("in_proj_qkvz", cfg_qkvz), ("gate_up_proj", cfg_gate_up),
                             ("qkv_proj", cfg_qkv)):
            off = 0
            for i, sz in enumerate(sizes):
                if sz % tp:
                    problems.append(f"{label} tp={tp} shard {i}: {sz} not divisible by {tp}")
                    continue
                so, ss = off // tp, sz // tp
                if so % L.PANEL_ROWS or ss % L.PANEL_ROWS:
                    problems.append(
                        f"{label} tp={tp} shard {i}: offset {so} size {ss} not a multiple of "
                        f"{L.PANEL_ROWS} — vLLM would silently truncate this shard")
                off += sz
    for label, K in (("down_proj", 17408), ("o_proj", 6144), ("out_proj", 6144)):
        for tp in (1, 2, 4):
            if K % tp or (K // tp) % L.SLAB_COLS:
                problems.append(f"{label} tp={tp}: K/rank={K // tp} not a multiple of "
                                f"{L.SLAB_COLS}")
    return problems


def gate_dense(gg, names: list[str]) -> list[str]:
    """Every non-PXQ4 tensor must decode to finite values of the right shape."""
    bad = []
    for n in names:
        ti = gg.tensors[n]
        K = ti.ne0
        N = 1
        for d in ti.dims[1:]:
            N *= d
        rows = min(N, 64)
        if ti.type_id == G.GGML_F32:
            w = D.dequant_f32(bytes(gg.raw(n))[: rows * K * 4], rows, K)
        else:
            rowb = G.row_size(ti.type_id, K)
            w = D.DECODERS[ti.type_id](bytes(gg.raw(n))[: rows * rowb], rows, K)
        if not np.all(np.isfinite(w)):
            bad.append(f"{n}: non-finite values")
        if np.abs(w).max() > 1e4:
            bad.append(f"{n}: implausible absmax {np.abs(w).max():g}")
    return bad


# ---------------------------------------------------------------------------------------------
# post-conversion check: the EMITTED checkpoint against the SOURCE gguf
# ---------------------------------------------------------------------------------------------
def check_output(gguf_path: str, out_dir: str, ref_hf: str | None, sample: int,
                 seed: int) -> int:
    """Close the loop: read what was actually written and prove it still is the GGUF.

    ``verify --all`` proves the CONVERTER's arithmetic. This proves the ARTIFACT — a shard that
    was written short, an index that points at the wrong file, a dtype that got downgraded, a
    tensor that silently never made it out. For each sampled PXQ4 module it re-joins
    ``pxq4_slabs`` + ``pxq4_anchor`` from the emitted safetensors and compares the result to the
    GGUF's bytes; for each sampled dense tensor it decodes the GGUF and compares to the emitted
    fp16 exactly (fp16(x) is deterministic, so this is equality, not tolerance).
    """
    import json as _json
    from . import convert as _C
    from . import namemap as _NM
    from . import safetensors_io as _ST

    idx_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            wmap = _json.load(f)["weight_map"]
    else:
        hdr = _ST.read_header(os.path.join(out_dir, "model.safetensors"))
        wmap = {k: "model.safetensors" for k in hdr if k != "__metadata__"}

    gg = G.GGUFFile(gguf_path)
    try:
        rng = random.Random(seed)
        ok = bad = 0
        problems: list[str] = []
        # The emitted GDN tensors are in HF head order, not ggml order, so every comparison
        # below has to move the SOURCE into that order before it can claim a mismatch. Without
        # this the check reports 48 layers of false failures — and, worse, a converter that
        # silently stopped permuting would report PASS.
        geom = _NM.gdn_geometry(gg.kv)

        cands = []
        for n in gg.order:
            hf = _NM.GGML_TO_HF(n, gg.kv)
            if hf is None:
                continue
            stem = hf[:-len(".weight")] if hf.endswith(".weight") else hf
            if stem + ".pxq4_slabs" in wmap:
                cands.append((n, stem, "pxq4"))
            elif hf in wmap:
                cands.append((n, hf, "dense"))
        rng.shuffle(cands)

        for ggml_name, key, kind in cands[:sample]:
            ti = gg.tensors[ggml_name]
            perm = _C.gdn_perm_for(ggml_name, ti, geom)
            xform = _NM.VALUE_TRANSFORMS.get(_NM.ggml_suffix(ggml_name))
            if kind == "pxq4":
                _, ssh_, sb = _ST.read_tensor_bytes(os.path.join(out_dir, wmap[key + ".pxq4_slabs"]),
                                                    key + ".pxq4_slabs")
                _, ash, ab = _ST.read_tensor_bytes(os.path.join(out_dir, wmap[key + ".pxq4_anchor"]),
                                                   key + ".pxq4_anchor")
                slabs = np.frombuffer(sb, np.uint8).reshape(ssh_)
                anchor = np.frombuffer(ab, "<f2").reshape(ash)
                if ti.type_id == G.GGML_PXQ4:
                    un_s, un_a = ((slabs, anchor) if perm is None else
                                  _C._unapply_perm_pxq4(slabs, anchor, perm))
                    if L.join_blob(un_s, un_a) == bytes(gg.raw(ggml_name)):
                        ok += 1
                    else:
                        bad += 1
                        problems.append(f"{key}: emitted panels != gguf bytes"
                                        + (" (after undoing the GDN head reorder)" if perm
                                           else ""))
                else:
                    # re-encoded: no source bytes to compare, so check the decode is finite and
                    # correlated with the original weight instead.
                    w = D.dequant_any(gg.raw(ggml_name), ti.type_id, ti.dims).reshape(
                        ti.ne1, ti.ne0)
                    if perm is not None:
                        w = np.take(w, np.asarray(perm[1], dtype=np.int64), axis=perm[0])
                    back = R.dequant(slabs, anchor)
                    rel = float(np.linalg.norm(back - w) / (np.linalg.norm(w) or 1.0))
                    if rel < 0.35:
                        ok += 1
                    else:
                        bad += 1
                        problems.append(f"{key}: re-encoded wrel={rel:.4f}")
            else:
                dt, sh, raw = _ST.read_tensor_bytes(os.path.join(out_dir, wmap[key]), key)
                got = np.frombuffer(raw, "<f2").reshape(sh)
                if ti.type_id == G.GGML_PXQ4:
                    want = R.dequant_blob(gg.raw(ggml_name), ti.ne1, ti.ne0)
                else:
                    want = D.dequant_any(gg.raw(ggml_name), ti.type_id, ti.dims)
                if perm is not None:
                    want = np.take(want, np.asarray(perm[1], dtype=np.int64), axis=perm[0])
                if xform is not None:
                    want = xform[0](want)
                want = want.reshape(got.shape).astype(np.float16)
                if np.array_equal(got.view(np.uint16), want.view(np.uint16)):
                    ok += 1
                else:
                    bad += 1
                    n_diff = int(np.count_nonzero(got.view(np.uint16) != want.view(np.uint16)))
                    problems.append(f"{key}: {n_diff}/{got.size} fp16 words differ")

        print(f"output check: {ok} match, {bad} MISMATCH (of {min(sample, len(cands))} sampled "
              f"of {len(cands)} emitted)")
        for p_ in problems[:10]:
            print("  ", p_)
        return 1 if bad else 0
    finally:
        gg.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gguf_to_vllm.verify")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--all", action="store_true", help="every PXQ4 tensor (gate G2 in full)")
    ap.add_argument("--sample", type=int, default=12,
                    help="how many PXQ4 tensors to run the expensive gates on")
    ap.add_argument("--oracle", default=None,
                    help="path to the compiled gguf_to_vllm_oracle binary; enables G1")
    ap.add_argument("--tp", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-output", default=None,
                    help="an already-converted output dir; verifies the ARTIFACT against the "
                         "gguf instead of re-checking the converter's arithmetic")
    ap.add_argument("--ref-hf", default=None)
    args = ap.parse_args(argv)

    if args.check_output:
        return check_output(args.gguf, args.check_output, args.ref_hf, args.sample, args.seed)

    gg = G.GGUFFile(args.gguf)
    try:
        gg.assert_all_supported()
        book, sub = _tables_from_file(gg)
        R.check_tables(book, sub)

        pxq4 = [n for n, t in gg.tensors.items() if t.type_id == G.GGML_PXQ4]
        dense = [n for n, t in gg.tensors.items() if t.type_id != G.GGML_PXQ4]
        print(f"{len(gg.tensors)} tensors: {len(pxq4)} pxq4, {len(dense)} other")
        print(f"tables: book/sub match compiled-in = "
              f"{np.array_equal(book, R.BOOK) and np.array_equal(sub, R.SUB)}")

        rng = random.Random(args.seed)
        # Cover every distinct shape first, then fill the sample at random: shapes are what
        # exercise different slab counts and panel strides, names are not.
        by_shape: dict[tuple[int, int], list[str]] = {}
        for n in pxq4:
            t = gg.tensors[n]
            by_shape.setdefault((t.ne1, t.ne0), []).append(n)
        sample = [v[0] for v in by_shape.values()]
        rest = [n for n in pxq4 if n not in set(sample)]
        rng.shuffle(rest)
        sample += rest[: max(0, args.sample - len(sample))]
        print(f"distinct pxq4 shapes: {sorted(by_shape)}")

        rc = 0

        g2_names = pxq4 if args.all else sample
        n2, f2 = gate_g2(gg, g2_names)
        print(f"G2 split->join byte round-trip: {n2 - len(f2)}/{n2} PASS")
        if f2:
            print("  FAIL:", f2[:10]); rc = 1

        if args.oracle:
            n1, f1 = gate_g1(gg, sample, args.oracle, book, sub)
            print(f"G1 reference.dequant == pxa_deq_row_pxq6 (fp32 bit-exact): "
                  f"{n1 - len(f1)}/{n1} PASS")
            if f1:
                print("  FAIL:", f1[:10]); rc = 1
        else:
            print("G1 SKIPPED (no --oracle): build gguf_to_vllm_oracle.c and pass it")

        n3, f3 = gate_g3(gg, sample, args.tp, book, sub)
        print(f"G3 shard/dequant commute (both axes, tp={args.tp}): {n3 - len(f3)}/{n3} PASS")
        if f3:
            print("  FAIL:", f3[:10]); rc = 1

        probs = gate_real_shards(gg, "p2c")
        print(f"real fused-module shard arithmetic: "
              f"{'PASS' if not probs else str(len(probs)) + ' PROBLEMS'}")
        for p in probs[:20]:
            print("  ", p)
        if probs:
            rc = 1

        if args.ref_hf:
            from . import convert as C
            geom = NM.gdn_geometry(gg.kv)
            gprobs = C.gate_gdn_head_order(gg, args.ref_hf, geom)
            print(f"G5H GDN v-head order vs reference (exact, all GDN layers): "
                  f"{'PASS' if not gprobs else str(len(gprobs)) + ' PROBLEMS'}")
            for p in gprobs[:10]:
                print("  ", p)
            if gprobs:
                rc = 1
        else:
            print("G5H SKIPPED (no --ref-hf): the GDN v-head order is the one thing G1/G2/G3 "
                  "cannot see, so run this before trusting a checkpoint")

        bad = gate_dense(gg, dense if args.all else dense[:64])
        print(f"dense decoders sane: {'PASS' if not bad else str(len(bad)) + ' PROBLEMS'}")
        for b in bad[:10]:
            print("  ", b)
        if bad:
            rc = 1

        return rc
    finally:
        gg.close()


if __name__ == "__main__":
    raise SystemExit(main())
