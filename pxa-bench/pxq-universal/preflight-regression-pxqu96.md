# preflight.py regression case, built from the six-card Flash-Next map

Kept so the stale-instrument failure cannot silently return. Both numbers below are measured
on the same inputs: `recipes/pxqu96-flashnext-6card-262k-ub2048.tiers` (835 rules) against
the real 1224-tensor geometry of the Flash-Next uncensored artifact.

| instrument | verdict on this map |
|---|---|
| pre-2026-09-16 `preflight.py` | **FAILED — 109 tensors** |
| current `preflight.py` | **PREFLIGHT OK** (exit 0) |

Of the 109 the old tool flagged, **0 are genuinely eligible-and-short**. Every one has
`ne0 % 32 != 0` and is copied through by a source gate before the map is ever consulted —
`llama-quantize.cpp` now applies `quantize &= (tensor->ne[0] % 32 == 0)` with the comment
*"Every codec here has a block of at least 32 and ASSERTS n_per_row % block == 0 -- it aborts
the process rather than falling back. Conv kernels are 4 elements wide (ssm_conv1d and
ple_conv1d are both [4, 10240]), so a short row must be left alone structurally, not by
remembering to name each one. Turns a crash into a copy."*

The old tool modelled the pre-fix crash — its docstring still described dying at tensor
59/1224 on `blk.1.ple_conv1d.weight`. It was a correct instrument for a quantiser that no
longer exists, and a false-negative machine for the one that does.

## What the current instrument reports

```
self-test      7 gate cases matched the quantiser (positive control passed)
tensors        1224
  copied       390 by source gates  {ndims<2: 304, row-gather: 1, gate_inp: 48, short-row ne0=4: 37}
  reached map  834   (unmatched: 0)
PREFLIGHT OK
```

`unmatched: 0` is the number that matters for a map: no eligible tensor silently takes the
PXQU default type.

## The rule this case exists to enforce

**Do not deform a map to satisfy a tool.** The old tool reported a failure that could not
happen; the correct response was to correct the instrument, not to add 109 inert rules naming
a codec for tensors that are copied through regardless. A stale check that can block a valid
artifact is a defect in the check.

## Reproduce

```
cd pxa-bench/pxq-universal
python3 preflight.py recipes/pxqu96-flashnext-6card-262k-ub2048.tiers /tmp/qwen4exp-tensors.txt
```

The gate model self-tests on every run against the exact historical crash (`blk.1.ple_conv1d.weight
[4, 10240]` must report COPIED). If that positive control fails, the tool exits 4 and refuses to
report — a negative result is only worth as much as its positive control.
