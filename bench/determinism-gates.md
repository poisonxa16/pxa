# PXQ determinism & correctness gates (G1/G2/G3)

Every PXQ kernel and quantizer change ships only after passing a fixed gate battery. These are
the gates behind the phrase "bit-exact fast kernels" in the README — publishing them because
kernel speed claims without a determinism story are vibes.

## The primary gate: logit reproducibility

**Run the same prompt twice and compare the probabilities, not the text.** Every other check on
this page compares an *argmax* — a generated token, or a sha over generated tokens. An argmax is a
lossy view of the forward pass: a kernel can wobble the logits on every single run and still emit
the same token wherever the top-2 margin is wider than the wobble. What you then see is not
"deterministic", it is "deterministic except on the prompts where two candidates happen to be
close", which reads as a rare flake on long prompts and under load.

The check: `LOGIT_REPS` (default 6) identical requests at `n_predict 1`, `n_probs 2`,
`temperature 0`, `top_k 1`, `cache_prompt false`, at np=1 and again pinned to slot 1 at np=2.
**PASS only if every returned probability is identical to the last digit** — one distinct value
across all runs, spread exactly `0.000e+00`. It is check 3b in `bench/gate/run-gate.sh`.

Why it is the primary gate, 2026-09-03: the release gate on `rel/v2026.09.03` passed np=1 12/12
byte-identical and then failed needle20801 once in four runs, with the box busy. Under the sha view
that looks like a near-tie worth documenting. Under this view it was not close: the token-0
probability of an *identical* prompt took 46 distinct values in 48 runs (0.4964–0.5339) at 3,121
tokens and 12 distinct values in 12 runs (0.3937–0.4339) at 20,801, in **every** arm including the
ones that passed 12/12. The cause was a cross-block write-after-read race in the DeltaNet out-gate
fusion (`PXA_FUSE_DELTANET` bit 1); the token that flipped had a top-2 margin of 4.9e-04 against a
1.2e-02 run-to-run spread at that index.

**Rule: never call a greedy flip a benign near-tie without the probabilities.** A near-tie in a
deterministic engine resolves the same way every time. If the output moved, the logits moved.

## Quantizer gates

- **Q-G1 — byte-parity:** `pxa-bench/pxq6_ref.cpp` (a standalone build of the *production*
  converter) must BYTE-MATCH the golden numpy implementation (`pxa-bench/pxqu_golden.py`) on
  both the quantized bytes AND the dequantized values, for every tier (PXQ2/PXQ3/PXQ4).
- **Q-G2 — wrel reproduction:** the C quantizer's relative weight error must equal the numpy
  harness (`pxa-bench/pxqu_wrel.py`) on a frozen 36-slice rng-seed-42 protocol to ±1e-4, per
  tier. This pins quant quality to a reference before any ppl run.

## Kernel gates (the `PXA_PXQ6_*` fast paths)

Each env-gated fast kernel is proven **memcmp bit-exact** against the baseline kernel on real
model tensors before it defaults on — same FMA order, same accumulation chains:

| gate | kernel | proof |
|---|---|---|
| K1 `PXA_PXQ6_KSPLIT` | decode gate/up K-split + persistent workspace reducer | memcmp bit-exact, all formats |
| K2 `PXA_PXQ6_PAIRLUT` / `PXA_PXQ6_VECX` | byte-pair LUT / float4 activation loads | memcmp bit-exact |
| K3 `PXA_PXQ6_GUFUSE` / `PXA_PXQ6_SCATFUSE` | fused up+gate GEMM + GLU epilogue / fused MoE scatter | memcmp bit-exact vs the unfused pipeline |
| K4 `PXA_PXQ6_RAGTAIL` | ragged-tail FMA skip | memcmp bit-exact (skipped work was store-masked) |
| K5 `PXA_PXQ6_PIPE` | sm_60 2-stage register prefetch | memcmp bit-exact (identical arithmetic DAG) |
| `PXA_G2_ADDFUSE` | residual-add fusion (ADD+FUSED_RMS_NORM pair, ne0≥256; MUL_MULTI_ADD residual epilogue) | temp-0 sha identical on/off (bitwise-commutative-identical order) |
| `PXA_FUSE_DELTANET` bit 1 | DeltaNet out-gate (FUSED_RMS_NORM + FUSED_MUL_UNARY) | **failed** the logit gate 2026-09-04: cross-block write-after-read on a shifted overlap. Out of the default (mask 53) and guarded by `pxa_g2_addfuse_no_shifted_overlap` when re-armed |
| `PXA_G2_NORMFUSE` / `PXA_G2_QUANTFOLD` | q8_1 sidecar producers (fused rms-norm / DeltaNet out-gate) | temp-0 sha identical on/off (no measured gain — default OFF) |

End-to-end: temp-0 generation SHA over a fixed prompt battery is **identical** with all fast
paths on vs all off, on sm_60 (P100), sm_61 (1080 Ti) and sm_70 (V100).

## G3 — the non-bit-exact paths (declared, gated differently)

- **CPU↔CUDA dequant:** top-20 logprob parity + identical temp-0 generation at `-ngl 0` vs
  `-ngl 99` (bit-exactness across backends is not claimed — parity of outcomes is).
- **`PXA_PXQ6_WMMA` (experimental V100 tensor-core prefill):** deterministic but NOT bit-exact
  (~1e-6 output deltas by design). It stays default-OFF and is ppl-regated separately before
  any future enable. If you benchmark with it on, say so.
- **`PXA_PXQ_INT8_PREFILL` (opt-in sm_61 int8 prefill tile):** G3-gated — flag-off dispatch is
  byte-identical; flag-on passed temp-0 sha-identity on the 5.8k-token continuation battery,
  top-1 logit identity on every spot-check, and the all-tier silicon tile test (max rel err
  3.6e-7 vs an fp64 snapped-book reference). Tail top-5 order can shift at p≈0.015.

## The release gate you can actually run

The gates above are the per-lever proofs. The gate a *build* has to pass before it ships is in the
repository and runs in one command:

```bash
MODEL=/path/to/model.gguf ./bench/gate/run-gate.sh
```

It boots and kills its own `llama-server`, takes no container, hard-codes no path, and depends on
nothing but `bash` and `python3`. It checks greedy determinism over 12 repetitions at `-np 1`; a
coherence completion; needle recall with a stable sha on two long-context prompts (3,121 and
20,801 tokens, shipped under `bench/gate/prompts/`); greedy determinism over 12 repetitions at
`-np 2` on a slot whose neighbour was erased first; and `test-pxq-cpu-dot`, `test-kv-seq-shadow`
and `test-narrow-kernel-parity`. It exits 0 only if every check passed, and prints `SKIP` with a
reason rather than a false `PASS` for anything it could not run honestly.

Two rules are built into it, and `bench/gate/README.md` explains why each was paid for the hard
way:

- **12/12 identical at `-np 1` *and* at `-np 2`.** Six clean runs is not a pass, and one arm is not
  a pass. The failure mode that matters is rare: a fused lever that was identical 11 times in 12 at
  `-np 1`, and another that was 12/12 at `-np 1` while producing three different outputs across 12
  runs at `-np 2`. Both were real defects — a kernel writing into storage it had *inferred* was
  dead. Anything short of 12/12 in either arm is a fail, not a flake.
- **Compare slots only at the same KV placement.** Two slots at different KV offsets legitimately
  produce different bytes from the same prompt: a different `n_kv` changes the attention reduction
  tiling, which changes summation order by a few ULPs, which occasionally lands on a near-tie. In
  the measured case two slots agreed for 12 tokens and then diverged where the top-2 gap was
  0.005 nats; erasing the other slot so both started at offset 0 made them byte-identical. So erase
  the other slots before comparing — which is what `run-gate.sh` does — or compare log-probabilities
  with a tolerance. A sha difference between differently placed slots is evidence of nothing.

## Reproducing

The gate harnesses live in `pxa-bench/`: `pxq6_ref.cpp`, `pxq6_test.cu` (device memcmp
battery), `pxqu_golden.py`, `pxqu_wrel.py`, `pxqu_ref.cpp`. Build notes are at the top of each
file. Run them against any PXQ GGUF tier; a failure of any bit-exact gate is a release-blocking
bug — report it.
