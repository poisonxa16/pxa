#!/usr/bin/env python3
"""pxa-thin-pxqn-docs.py <tree> [--allow-missing] -- PXQN ships compiled-only, so the public docs keep what a user can set
and observe, never how the closed kernels work. Row by row, keyed by the lever name; FAILS CLOSED when a keyed row is
not found exactly once (the doc moved -> re-key, never skip silently)."""
import re, sys, pathlib

T = pathlib.Path(sys.argv[1])
ALLOW_MISSING = '--allow-missing' in sys.argv[2:]
REL = '(the closed PXQN library; PXA release tarball and images only)'
ROWS = {
 'docs/LEVERS.md': {
  '`PXA_TSPLIT_EPI`': "| `PXA_TSPLIT_EPI` | **on** with the fused reduce at decode width | Ends each tensor-split phase in ONE launch per card instead of three or four: the residual add, the two-card sum and the next norm run in the reduce kernel itself. Output is byte-identical. The PXQN form of this epilogue ships in the release build " + REL + ". | `PXA_TSPLIT_EPI=0` runs the separate kernels exactly as before. |",
  '`PXA_TSPLIT_EPI_PUSH`': "| `PXA_TSPLIT_EPI_PUSH` | **on** with `PXA_TSPLIT_EPI` | The weight kernel that produces each card's partial also writes it straight into the other card's staging slot, so the card-to-card copy overlaps the matrix work. Byte-identical output. | `PXA_TSPLIT_EPI_PUSH=0` lets the reduce kernel do the copy. |",
  '`PXA_TSPLIT_EPI_NORM`': "| `PXA_TSPLIT_EPI_NORM` | **on** for P100 (sm_60) cards with `PXA_TSPLIT_EPI` | Gives the one-launch tensor-split epilogue to classic files (PXQ4, k-quants, Q8_0): the residual add, the two-card sum and the next RMS norm run in the reduce kernel, and the norm node is skipped. Byte-identical output (the norm kernel's exact operation order). `=2` also arms it on other architectures. | `PXA_TSPLIT_EPI_NORM=0` runs the separate kernels. |",
  '`PXA_PXQ4_RB`': "| `PXA_PXQ4_RB` | **on** (P100 / sm_60, one-token decode of classic PXQ4 weights) in the release build " + REL + " | A faster P100 decode kernel for classic PXQ4 weights, including a one-launch up/gate + SiLU. Deterministic, not byte-identical to the open kernel. A build from source has no such kernel and decodes PXQ4 with the open kernel (correct, about 12% slower). | `PXA_PXQ4_RB=0` restores the open decode kernel. |",
  '`PXA_SPEC_FAST_VERIFY`': "| `PXA_SPEC_FAST_VERIFY` | off (0) | Lets the speculative-decoding verify step (2–8 tokens at once) use cheaper arithmetic in the 4-bit weight kernels. On a V100 the verify kernel gets about 12% faster at 4 tokens. On a P100 it makes no measurable difference. | Leave it off unless you have measured it on your cards. With it on, greedy output with MTP on can differ slightly from MTP off. It is still the same every run. |",
  '`PXA_PXQN_RHT_NODE_NY`': "| `PXA_PXQN_RHT_NODE_NY` | on (1) | PXQN files " + REL + ": on a P100 the verify step (2–8 tokens) of the rotated layers is about 26% faster at 4 tokens; output unchanged bit for bit. | Set 0 only to compare against the old path. |",
 },
 'docs/DEFAULTS.md': {
  'Kernels: quantized-KV narrow attention': "| | Kernels: quantized-KV narrow attention (`PXA_FA_QKV_DIRECT`), the PXQN decode levers of the release build (`PXA_PXQN_RB`, `PXA_PXQN_RB_RHT`, `PXA_PXQN_RHT_NODE_NY`), DeltaNet conv cluster and in-place carry (`PXA_DN_CONVFUSE`, `PXA_DN_INPLACE`, `PXA_DN_CONVFUSE_NY`). | `ctx16k-fa-qkv-split-pv8`, `gemv-bw-pxqn4-rb`, `pxqn-rb-rht-unfuse`, `pxqn-rht-node-ny`, `dn-convfuse`, `dn-inplace-carry`, `ctx16k-dn-convfuse-depth`, `dnny-convfuse-ny-fire` |",
  'ssm_out RHT ranges, concat and q8_1': "| | A PXQN V100 decode lever of the release build (`PXA_PXQN_RR_Q8`). The fused epilogue `PXA_TSPLIT_EPI_Q8` stays off. | `v100-plain-rr-q8` (+3.0% tg128), `v100-plain-epi-q8` (measured loss) |",
 },
}
bad = 0
for rel, rows in ROWS.items():
    p = T / rel
    if not p.exists():
        if ALLOW_MISSING:
            print(f"skip: {rel} not in this tree"); continue
        print(f"FAIL: {rel} missing"); bad = 1; continue
    lines = p.read_text().split('\n')
    for key, new in rows.items():
        hits = [i for i, l in enumerate(lines) if l.startswith('|') and key in l]
        if rel == 'docs/LEVERS.md':
            hits = [i for i in hits if lines[i].startswith('| ' + key + ' |')]
        if len(hits) != 1:
            print(f"FAIL: {rel}: key {key!r} matched {len(hits)} rows (want 1)"); bad = 1; continue
        lines[hits[0]] = new
    p.write_text('\n'.join(lines))
    print(f"ok: {rel}: {len(rows)} rows thinned")
# docs/lab/ is a closed directory. The public-tree omit pass removes it. Leave it removed:
# creating the directory again makes a closed directory entry a no-op.
# A package stage still has the file, because keep_doc copied the lab table in before this
# script runs. Replace that copy with the public lever page so the package does not ship the lab table.
lab, pub = T / 'docs/lab/LEVERS.md', T / 'docs/LEVERS.md'
if not lab.exists() and not lab.is_symlink():
    print("ok: docs/lab/ stays out")
elif pub.exists() and not bad:
    if lab.is_symlink():
        lab.unlink()
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text('<!-- The lab lever table is not part of the public release; this is the public lever page (docs/LEVERS.md). -->\n'
                   + pub.read_text())
    print("ok: docs/lab/LEVERS.md = the public lever page")
sys.exit(bad)
