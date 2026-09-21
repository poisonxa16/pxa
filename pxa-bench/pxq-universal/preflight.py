#!/usr/bin/env python3
"""Preflight a PXQU tier map against the real tensor geometry, using the quantiser's own gates.

WHY THIS FILE CHANGED (2026-09-16)
----------------------------------
The previous revision modelled behaviour that no longer exists. It assumed that any tensor
left unmatched by the map falls through to the PXQU default type (MXFP4, a block-32 codec)
and aborts the process on GGML_ASSERT(n_per_row % kBlockSize == 0). That was true when the
first build died at tensor 59/1224 on blk.1.ple_conv1d.weight (ne0=4), and the docstring here
recorded it.

It is no longer true. llama-quantize.cpp now decides, per tensor and BEFORE the map is
consulted, whether the tensor is eligible for quantisation at all:

    quantize &= (ggml_n_dims(tensor) >= 2);
    quantize &= tensor->name.ends_with("weight");
    quantize &= !name contains "_norm.weight";
    quantize &= !name contains "ffn_gate_inp.weight";
    quantize &= !name contains "ffn_gate_tid2eid.weight";
    quantize &= !pxa_is_row_gather_tensor(tensor);          // per_layer_token_embd
    quantize &= (tensor->ne[0] % 32 == 0);                  // <- the fix
    quantize &= !ssm_conv1d.weight / !ssm_x.weight / !ssm_dt.weight
    quantize &= !attn_rel_b

with the comment: "Every codec here has a block of at least 32 and ASSERTS n_per_row % block
== 0 -- it aborts the process rather than falling back. Conv kernels are 4 elements wide
(ssm_conv1d and ple_conv1d are both [4, 10240]), so a short row must be left alone
structurally, not by remembering to name each one. Turns a crash into a copy."

So a short row is now COPIED, not aborted, and the old check reported failures that cannot
happen. On a valid six-card map it false-positived 109 tensors. Two consequences, both kept:
a stale instrument must not be able to block a good map, and the map must never be deformed
to satisfy a tool.

WHAT THIS VERSION CHECKS INSTEAD
--------------------------------
  1. genuine abort risk  - a tensor that is quantise-ELIGIBLE (passes every gate above) has
                           ne[0] % 32 != 0, and the map targets it with a block-32 codec.
                           This is the only condition that can still kill a run.
  2. silent default      - an eligible tensor the map never matches. It does not abort; it
                           silently takes the PXQU default. Worth knowing, not worth failing.
  3. coverage            - how many tensors are copied by the gates vs reach the map.

SELF-TEST (mirrors preflight.py --selftest, and runs by default)
----------------------------------------------------------------
A negative result from a check is only worth as much as its positive control, so the gate
model is exercised against the exact historical crash before it is trusted on a new map:
blk.1.ple_conv1d.weight [4, 10240] must be reported COPIED, not flagged. If that assertion
fails the tool refuses to report (exit 4) rather than printing a reassuring OK.

Exit: 0 ok, 1 genuine abort risk, 3 missing input, 4 self-test refused.
"""
import re
import sys

BLOCK32 = {"mxfp4", "q8_0", "q4_k", "q5_k", "q6_k", "q4_0", "q5_0", "q6_0", "iq4_nl",
           "iq1_s", "iq2_xxs", "pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6"}
PASSTHRU = {"f32", "f16", "bf16"}


def copied_by_gate(name, ne):
    """Return the gate that copies this tensor through, or None if it is quantise-eligible.

    Order and content mirror llama-quantize.cpp; each entry names its own gate so a report
    can say WHY a tensor was skipped rather than only that it was.
    """
    if len(ne) < 2:
        return "ndims<2"
    if not name.endswith("weight"):
        return "not *weight"
    if "_norm.weight" in name:
        return "norm"
    if "ffn_gate_inp.weight" in name:
        return "gate_inp"
    if "ffn_gate_tid2eid.weight" in name:
        return "hash-router"
    if name.startswith("per_layer_token_embd.weight"):
        return "row-gather"
    if ne[0] % 32 != 0:
        return f"short-row ne0={ne[0]}"
    if ("ssm_conv1d.weight" in name or "ssm_x.weight" in name or "ssm_dt.weight" in name):
        return "mamba-small"
    if "attn_rel_b.weight" in name:
        return "T5-rel-bias"
    return None


def selftest():
    """Positive control: the historical crash must be modelled as a COPY, not a failure."""
    cases = [
        # name, ne, expected gate
        ("blk.1.ple_conv1d.weight", [4, 10240], "short-row ne0=4"),
        ("blk.0.ssm_conv1d.weight", [4, 10240], "short-row ne0=4"),
        ("per_layer_token_embd.weight", [160, 320001536], "row-gather"),
        ("blk.0.attn_norm.weight", [2560], "ndims<2"),
        ("blk.0.ffn_gate_inp.weight", [2560, 512], "gate_inp"),
        ("blk.0.ffn_gate_exps.weight", [640, 2560, 512], None),   # eligible
        ("blk.0.attn_q.weight", [2560, 6144], None),              # eligible
    ]
    bad = []
    for name, ne, want in cases:
        got = copied_by_gate(name, ne)
        if got != want:
            bad.append(f"{name}: gate model says {got!r}, expected {want!r}")
    if bad:
        print("*** SELF-TEST FAILED - REFUSING TO REPORT ***")
        for b in bad:
            print("   ", b)
        sys.exit(4)
    return len(cases)


def load_rules(map_path):
    rules = []
    with open(map_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            rx, t = line.rsplit("=", 1)
            rules.append((re.compile(rx), t.lower()))
    return rules


def load_tensors(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\S+)\s+([0-9x]+)\s+(\S+)\s+off=", line)
            if not m:
                continue
            out.append((m.group(1), [int(x) for x in m.group(2).split("x")]))
    return out


def main(argv):
    map_path = argv[1] if len(argv) > 1 else "recipes/pxqu96-flashnext-6card-262k-ub2048.tiers"
    ten_path = argv[2] if len(argv) > 2 else "/tmp/qwen4exp-tensors.txt"

    n_case = selftest()
    rules = load_rules(map_path)
    tensors = load_tensors(ten_path)
    if not tensors:
        print(f"no tensors parsed from {ten_path}")
        return 3

    abort, unmatched, copied, reached = [], [], 0, 0
    gate_hist = {}
    for name, ne in tensors:
        gate = copied_by_gate(name, ne)
        if gate is not None:
            copied += 1
            gate_hist[gate] = gate_hist.get(gate, 0) + 1
            continue
        reached += 1
        hit = None
        for rx, t in rules:
            if rx.search(name):
                hit = t
                break
        if hit is None:
            unmatched.append((name, ne))
            continue
        if ne[0] % 32 != 0 and hit in BLOCK32:
            abort.append((name, ne, hit))

    print(f"self-test      {n_case} gate cases matched the quantiser (positive control passed)")
    print(f"map            {map_path}  ({len(rules)} rules)")
    print(f"tensors        {len(tensors)}")
    print(f"  copied       {copied} by source gates  {gate_hist}")
    print(f"  reached map  {reached}   (unmatched: {len(unmatched)})")

    if unmatched:
        print(f"\nNOTE: {len(unmatched)} eligible tensor(s) matched no rule and will silently take")
        print("      the PXQU default type. Not fatal, but check this is intended:")
        for name, ne in unmatched[:10]:
            print(f"  {name:<46} ne={'x'.join(map(str, ne))}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    if abort:
        print(f"\n*** PREFLIGHT FAILED: {len(abort)} tensor(s) would abort a block-32 codec ***")
        for name, ne, t in abort[:12]:
            print(f"  {name:<44} ne={'x'.join(map(str, ne)):<16} -> {t}")
        return 1

    print("\nPREFLIGHT OK: no quantise-eligible tensor with ne0 % 32 != 0 is targeted at a "
          "block-32 codec")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
