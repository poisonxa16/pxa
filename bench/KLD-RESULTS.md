# PXQ — KL-divergence vs the bf16 reference (the #1 credibility gate)

**Status: MEASURED 2026-07-18.** True bf16 reference (not a proxy). Numbers below are final.

## What this measures

KL-divergence isolates **quantization damage**: it compares each PXQ tier's next-token
distribution against the **bf16 merge source's** distribution, token-for-token, over a fixed eval
corpus. Unlike perplexity (which also moves with model quality), KLD is a direct "how far did the
quant push the logits" number — the measure quant reviewers (ubergarm/bartowski-style tables)
trust most.

## Reference & protocol

- **Reference model (bf16):** `pxa-35b-ornith-siq-bf16.gguf` (34.66 B params, BF16, 64.6 GiB) —
  the exact DELLA merge (Ornith 0.80 / SIQ 0.20) the PXQ tiers were quantized from. This is the
  **true bf16 reference**, not a proxy. **bf16 reference ppl (100 chunks): 7.0816 ± 0.109.**
- **Corpus:** wikitext-2-raw **test** split, n_ctx 512, **100 chunks**.
- **Engine:** pxa `build-pxqu` (the full-tier UNIVERSAL build), `-fa on -ctk f16 -ctv f16`,
  PXQ fast kernels enabled (`PXA_PXQ6/2/3=1` + `KSPLIT/VECX/GUFUSE/SCATFUSE/RAGTAIL`).
- **Pass 1:** dump bf16 reference logits (`--kl-divergence-base`) on 6 Tesla cards (`-sm layer`).
- **Pass 2:** each PXQ tier scored against that base (`--kl-divergence`), on 2× V100 (`-ts 1,1`).
- Hardware/driver: 2× V100 + 4× P100, NVIDIA 580.142, CUDA 12.8.1.

## Results — measured

Lower KLD = closer to bf16. "top-1 agreement" = fraction of tokens where the quant's argmax
matches bf16's argmax. All three tiers scored against the **same** bf16 logit dump.

| tier | bits | wikitext-2 ppl* | **Mean KLD** | Median KLD | 99.9% KLD | top-1 agreement | RMS Δp |
|---|---|---|---|---|---|---|---|
| **PXQ4** (formerly PXQ6) | 4.27 bpw | 7.36 | **0.0560 ± 0.0007** | 0.0308 | 1.292 | **89.4%** | 6.36% |
| **PXQ3** | 3.27 bpw | 7.44 (+1.1%) | **0.0758 ± 0.0009** | 0.0402 | 1.730 | **87.8%** | 7.61% |
| **PXQ2** | 2.27 bpw | 8.39 (+14%) | **0.2054 ± 0.0022** | 0.1092 | 3.869 | **80.3%** | 13.08% |

\* ppl column is the published 200-chunk ladder (`bench/ppl-ladder.sh`); KLD is this 100-chunk
run. Within this KLD run the derived per-tier ppl (bf16 × e^(mean lnΔ)) is PXQ4 7.33 / PXQ3 7.40 /
PXQ2 8.31 — consistent with the ladder.

**Reading it:** KLD is **monotonic** and tracks the ppl ladder. PXQ4 (4-bit) diverges from bf16 by
a mean KLD of just **0.056** with **89.4%** top-1 token agreement — strong for a low-active MoE at
4.27 bpw. PXQ3 (3-bit) costs only ~35% more divergence (0.076) while holding **87.8%** top-1 — the
tier's "essentially flagship at 14.7 GB" story survives the harder metric. PXQ2 (2-bit) is the real
trade: mean KLD ~3.7× the flagship (0.205), top-1 down to 80.3% — coherent, but visibly lossier,
exactly as advertised.

Raw logs: `kld-{PXQ6,PXQ3,PXQ2}-pxqu.log` + the `base.log` bf16 dump from the `kld.sh` run (kept off-repo; regenerate with `kld.sh`).

> **Build note (repro gotcha):** the KLD must be run with the **UNIVERSAL** build (`build-pxqu` /
> the public repo built for all tiers). An older PXQ6-only build (`build-pxq6`) SIGFPEs in
> `llama-perplexity`'s KLD path on the PXQ2/PXQ3 low-bit types (they postdate it). This is a
> tooling-build artifact, **not** a model or inference defect — the low-bit tiers run fine in
> `llama-server` on P100/V100/1080Ti. `bench/kld.sh` builds the full-arch binary, so following it
> reproduces these numbers directly.

## Incumbent comparison

**Pending — a matched-size incumbent (e.g. a bartowski/unsloth Qwen3.x IQ3/Q3 at ~14 GB) was not
available for this run.** Note: a true *KLD* against an incumbent isn't meaningful anyway (KLD
requires the **same** base model's logits, and the incumbent is a different base) — the honest
incumbent comparison is a **perplexity-at-matched-size** run on the same wikitext protocol, which
`bench/ppl-ladder.sh` supports for any GGUF. This is a fast-follow.

## Reproduce

`bench/kld.sh` in the kernel repo is the exact procedure (bf16 base dump, then per-tier scoring).
Point it at the bf16 source + the tier GGUFs and it prints the same numbers.
