# PXA Network — PXQ4 sm_60 (Tesla P100) serving

Measured on the box, 2026-08-27, cards 1+6 (2x Tesla P100-PCIE-16GB, TP=2).
Every arm was correctness-gated (Paris / raw-1-token / 17*23=391) AND byte-gated
(20 greedy generations identical to an NCCL reference boot) before its number was
recorded. A fast wrong answer is not a result, and on this arch "wrong" shows up
as `!!!!` on a raw prompt while the plausible prompts still pass -- so the raw
prompt is the one that matters.

## Shipping configuration

    scripts/pxa-serve-sm60.sh

| metric | reference | measured |
|---|---|---|
| single-stream decode | 24.01 | 24.0 - 26.4 |
| aggregate @8 | ~70 | 70.0 - 72.1 |
| aggregate @4 | -- | 45.0 - 51.5 |
| long-doc prefill | ~225 | ~218 |

Prefill sits ~3% under the reference. The 218 figure is stable across every arm;
the reference used a different prompt harness, and `--max-model-len 8192` rejects
the 9k-token probe, so the two are not measured identically. Treated as unresolved
rather than matched.

## Where the 13.2 came from

Nothing was broken in the build. The gate shipped the conservative earlier recipe
and the later tuning never landed in it:

| lever | effect |
|---|---|
| custom-all-reduce OFF -> ON | 13.3 -> 24.0, ~1.8x. The dominant lever. |
| old site -> site-sm60 (fp16 mmv hook, 112 layers) | ~+12% |
| v7 kernel -> v10 | part of the above |
| MNS 4 -> 8 with ladder [1,2,4,8] | required for correctness, see below |

## Traps, each of which cost real time

**`TORCHDYNAMO_DISABLE=1` decides whether the server starts.** Without it,
`profile_run` compiles the LANGUAGE model through Inductor and dies with
`GPUTooOldForTriton` on a capability-6.0 card. The failing frame is
`qwen3_5.py:926 -> language_model.model(...)`, not the vision encoder.

**Do NOT "fix" that by disabling Triton globally.** A capability guard that sets
`HAS_TRITON=False` below sm_70 was tried and REVERTED: the Pascal warmup path
then calls `triton.next_power_of_2` on `TritonPlaceholder`, which does not define
it. Shimming that method only moves the failure to a real Triton kernel launch on
the same path. The predicate was verified correct in both directions and the
image still could not serve -- verifying a predicate is not verifying the image.

**`custom_ops:["none"]` is a correctness workaround, not dead weight.** With
defaults at MNS=4 / ladder[1,2,4], a raw 1-token prompt returns `!!!!` while
Paris and 391 both pass. It clears at MNS=8 / ladder[1,2,4,8]. GMU 0.94
reintroduces it. Any speed harness that only checks a capital city or a
multiplication cannot see this failure.

**`SPLIT_MAX_BLOCKS=150` is bimodal.** Four boots: 24.58, 24.61, 19.75, 9.27.
`300` gave 24.01 / 23.97 / 23.97 across three. A single good 150 sample looks
like a win. The mechanism is not understood; 300 is shipped on stability.

**`PXQ4_MMV_SLICE_MAX` is a dead knob** at serving level (14.43 at 8 vs 14.24 at
16). An earlier op-level note claiming 16-24 is better was superseded the
following day and does not survive at serving level.

**`PXQ4_LIB` must be explicit.** The sidecar bundles an sm70-only `.so` built
against the torch 2.10 ABI; unset, the loader can reach it on this image and die
with `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv` mid-load.

**Count the fp16 hook, do not assume it.** It is worth ~12% and its absence is
silent. The log string is `fp16 mmv fast path armed` -- grepping `f16 mmv`
matches nothing and makes a working hook look absent.

## Measured but NOT shipped

`libpxq4_sm60_v11.so` with `PXQ4_GEMM2D=1` reaches **300.1 tok/s prefill**
(+37%, well past the 225 reference) and FAILS raw-prompt correctness. Decode and
the byte-gate are unaffected. A prefill win that changes what the model says is
not a win. Recorded here so it is not rediscovered as a fresh idea.

`MNS=16` is UNVERIFIED on this arch: its only test ran on the broken image, so
the failure it produced says nothing about whether the 16-token graph is legal.

## Cards

Cards 1 and 6 are the free P100s. Card 0 keeps a smaller split because it is shared
with another production service, and card 3 is the protected 1080 Ti; the launcher
refuses both by explicit check, and the
container boundary excludes everything not listed in `NVIDIA_VISIBLE_DEVICES`.
