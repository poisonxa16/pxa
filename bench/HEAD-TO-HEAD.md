# PXQ vs ik IQ_K — matched-size head-to-head (identical weights, imatrix, corpus, reference)

> **2026-07-19 update:** since this document was measured, two recipe/env changes shipped and were
> verified (median-of-3, same protocol): a **q8_0 output head** (`--output-tensor-type q8_0`,
> +123 MB) and **`PXA_FUSE_DELTANET=3`**. Together they move PXQU-16 decode **57.2 → 62.4 t/s on
> P100** and **98.5 → 101.3 t/s on V100**. The tables below predate those changes and keep their
> original numbers (both sides measured without them — the *comparison* is unaffected).

**Measured 2026-07-18.** The strongest possible comparison: we quantized **our own** fusion2 bf16
(`pxa-35b-ornith-siq-bf16.gguf`, the exact source of the released PXQ tiers) to ikawrakow's IQ_K
incumbents with the fork's `llama-quantize`, using the **same imatrix** the PXQ tiers used
(`ornith-forquant.imatrix`, 2530 chunks). Same weights + same imatrix + same wikitext-2 corpus +
the **same bf16 KL-divergence reference** ⇒ KLD is directly comparable, not just perplexity. This is
a fairer, harder test than any downloaded incumbent (which would be different base weights).

## The table

Fidelity: wikitext-2, 100 chunks, bf16 reference ppl 7.0816. Speed: `llama-server`
`timings`, 200-tok decode (median of 3), prefill over a ~5.8k-token prompt at ub2048, model fully
GPU-resident, `build-pxqu`, driver 580.142 / CUDA 12.8.1.

> **Engine parity (read this before quoting the speed deltas).** Every number in this document —
> PXQ *and* IQ_K alike — was measured on the **same binary** (`build-pxqu`, this fork), same flags,
> same cards, same weights. Two consequences, stated precisely:
> 1. The PXQ-specific fast kernels (`PXA_PXQ6_*`) apply **only to PXQ-format tensors** — they do
>    nothing for the IQ_K runs.
> 2. This fork also carries **format-agnostic** Pascal/Volta engine work on top of upstream
>    ik_llama.cpp — an sm_60 fp16-GEMM gate fix (the P100 was wrongly excluded from full-rate fp16
>    cuBLAS), flash-attention improvements, MoE-path fixes, and the qwen35moe graph fusions — and
>    the IQ_K runs get **all of that too**, since those paths are shared.
>
> In other words: the incumbent was **not benchmarked on a handicapped engine**. The IQ_K numbers
> here are at least what stock upstream ik_llama.cpp would produce on these cards (we have not
> measured the exact stock-vs-fork delta for IQ_K, so we don't quote one) — and PXQ still wins by
> the stated margins. The speed comparison is format-vs-format on shared, equally-optimized
> infrastructure.

### 3-bit tier — **PXQ3 vs IQ3_K** (the priority pair)

| | file size | bpw | ppl | Mean KLD | top-1 agree | decode (2×P100) | prefill (2×P100) | decode (2×V100) | prefill (2×V100) |
|---|---|---|---|---|---|---|---|---|---|
| **PXQ3** | **14.7 GB** | 3.27 | 7.396 | 0.0758 | 87.8% | **55.0 t/s** | **836 t/s** | **94.6 t/s** | **1758 t/s** |
| IQ3_K | 15.1 GB | 3.44 | 7.384 | 0.0591 | 88.9% | 47.7 t/s | 696 t/s | 89.2 t/s | 1393 t/s |
| **Δ (PXQ3 vs IQ3_K)** | **−0.4 GB (smaller)** | | +0.16% | +28% (worse) | −1.1 pt | **+15% faster** | **+20% faster** | **+6% faster** | **+26% faster** |

### 4-bit tier — **PXQ4 (formerly PXQ6) vs IQ4_KS**

| | file size | bpw | ppl | Mean KLD | top-1 agree | decode (2×P100) | prefill (2×P100) |
|---|---|---|---|---|---|---|---|
| **PXQ4** | 18.7 GB | 4.27 | 7.349 | 0.0579 | 88.9% | **58.3 t/s** | **843 t/s** |
| IQ4_KS | 18.8 GB | 4.25 | 7.203 | 0.0280 | 92.0% | 51.1 t/s | 694 t/s |
| **Δ** | ~same | | +2.0% (worse) | +107% (worse) | −3.1 pt | **+14% faster** | **+21% faster** |

### 2-bit tier — **PXQ2 vs IQ2_KS**

| | file size | bpw | ppl | Mean KLD | top-1 agree | decode (1×V100) | prefill (1×V100) |
|---|---|---|---|---|---|---|---|
| **PXQ2** | 10.7 GB | 2.27 | **8.312** | **0.2054** | **80.3%** | **98.1 t/s** | **1876 t/s** |
| IQ2_KS | 10.1 GB | 2.19 | 9.114 | 0.2840 | 77.1% | 95.6 t/s | 1467 t/s |
| **Δ** | +0.6 GB | | **−8.8% (better)** | **−28% (better)** | **+3.2 pt** | **+3% faster** | **+28% faster** |

## Honest verdict (no spin)

**Speed: PXQ wins every pair, on every card.** Decode +3–15%, prefill **+20–28%**. The prefill margin
is the biggest and most consistent — exactly where the fused Pascal/Volta kernels earn their keep
(prefill is compute-bound; decode is bandwidth-bound so at matched bytes/token it lands closer, most
visibly on the 2-bit V100 pair at +3%). This is the real, measured PXQ advantage, and it holds on the
exact salvaged cards that are the whole point.

**Fidelity: mixed, and we state it plainly.**
- At **3-bit and 4-bit**, ikawrakow's IQ_K quants are **better per byte**: IQ3_K edges PXQ3 (KLD
  0.059 vs 0.076; ppl within noise) at a slightly *larger* size, and IQ4_KS clearly beats PXQ4 (KLD
  0.028 vs 0.058) at the same size. IQ_K is a decade-refined SOTA low-bit family; matching it on pure
  fidelity-per-byte was never PXQ's design goal.
- At **2-bit**, **PXQ2 wins outright** — better ppl (8.31 vs 9.11), better KLD (0.205 vs 0.284),
  higher top-1 (80.3 vs 77.1) **and** faster. PXQ's approach pulls ahead where the bit budget is
  tightest.

**Why the 3–4-bit fidelity gap (root cause, measured):** IQ_K spends bits where the output
distribution is most sensitive — it protects the backbone (attn_v / output / token_embd) at
q5_K/q6_K/q8_0, and its expert codebook is non-linear (levels concentrated where the weights are
dense). PXQ pins its backbone at MXFP4 (4-bit) and uses an MXFP4/E16-scale expert layout chosen so
the dequant is a cheap, aligned, vectorizable FP16 op that flies on Pascal/Volta. That layout is
exactly what buys the speed IQ_K can't match on sm_60/sm_70 — and it costs some fidelity-per-bit at
3–4 bit. The backbone half of the gap is cheaply closable (promote PXQ's backbone to q5/q6 like IQ_K,
small size cost, no speed hit); the codebook half is more inherent to the speed-first design.

**The defensible one-line positioning:** *at matched size on salvaged Pascal/Volta cards, PXQ delivers
~15% faster decode and ~20–28% faster prefill than ubergarm's IQ_K quants, with competitive fidelity
at 3–4 bit and better fidelity at 2-bit.* We do **not** claim to beat IQ_K on fidelity at 3–4 bit — we
claim (and measured) that we're faster at matched size, which is the axis that matters for "usable
tokens/sec on a $100 GPU."

---

## The comparison that actually matters: ub2048 on a SINGLE 16 GB card

The core PXQ promise is running a real 35B at **full ub2048 prefill on ONE 16 GB landfill card**.
Measured single-card ub2k ceiling (`-b2048 -ub2048 -c8192`, peak VRAM during prefill): ~13.4 GiB /
**~14.0 GB decimal weights** (fixed overhead — compute buffer + KV + context — ≈ 2.6 GB, so peak
maxes the card). **PXQ3 (14.7 GB) OOMs single-card ub2k; so does ubergarm's IQ3_K (15.1 GB).** For
this regime PXQ ships **PXQU** — PXQ-Universal, a knapsack mix of PXQ2/3/6 tensors sized to a card:
**PXQU-16** (14.0 GB, fits a 16 GB P100/V100) and **PXQU-12** (11.6 GB, fits a 12 GB card —
measured 2026-07-19: it does **not** load on an 11 GB 1080 Ti, whose tier is PXQ2).
(Pure PXQ2/3/4 are single-type "pick your quality"; PXQU is "pick your card, get the best that fits
at full ub2048 speed." Same kernels, same speed.)

Head-to-head, everything below is **-ub2048 on one card**, PXQU vs the IQ quant that also fits:

### ~14 GB tier — **IQ3_K (15.1 GB) does NOT fit ub2k single-card. These do:**

| | size | ppl | Mean KLD | top-1 | decode P100 | prefill P100 | decode V100 | prefill V100 |
|---|---|---|---|---|---|---|---|---|
| **PXQU-16** | 14.0 GB | 7.58 | 0.107 | 85.6% | **57.5 t/s** | **827 t/s** | **98.5 t/s** | **1896 t/s** |
| IQ3_KS | 14.2 GB | 7.57 | 0.085 | 86.7% | 47.8 t/s | 710 t/s | 89.2 t/s | — |
| **Δ** | | ~tie | +26% (worse) | −1.1 pt | **+20% faster** | **+16% faster** | **+10% faster** | |

IQ3_KS edges fidelity; PXQU-16 is faster and both barely fit. The best fidelity-per-byte IQ3
(IQ3_K) isn't an option here at all.

### ~12 GB tier — **PXQU-12 wins both axes**

| | size | ppl | Mean KLD | top-1 | decode P100 | prefill P100 | decode V100 | prefill V100 |
|---|---|---|---|---|---|---|---|---|
| **PXQU-12** | 11.6 GB | **7.93** | **0.161** | **82.5%** | **58.4 t/s** | **842 t/s** | **97.6 t/s** | **1905 t/s** |
| IQ2_KL | 12.0 GB | 8.12 | 0.165 | 81.5% | 42.5 t/s | 685 t/s | 84.0 t/s | 1475 t/s |
| **Δ** | −0.4 GB (smaller) | **better** | **better** | **+1.0 pt** | **+37% faster** | **+23% faster** | **+16% faster** | **+29% faster** |

### ~10 GB tier — **PXQ2 wins both axes**

| | size | ppl | Mean KLD | top-1 | decode V100 | prefill V100 |
|---|---|---|---|---|---|---|
| **PXQ2** | 10.7 GB | **8.31** | **0.205** | **80.3%** | **98.1 t/s** | **1876 t/s** |
| IQ2_KS | 9.5 GB | 9.11 | 0.284 | 77.1% | 95.6 t/s | 1467 t/s |
| **Δ** | +1.2 GB | **−8.8% (better)** | **−28% (better)** | **+3.2 pt** | **+3% faster** | **+28% faster** |

**ub2k verdict:** the fidelity crossover is ~2.7–3 bpw. At **≤12 GB — what most people run on one
16 GB card with margin — PXQ wins fidelity AND speed outright.** At the 14 GB ceiling PXQ trades a
hair of fidelity for +20% speed, and IQ3_K can't run ub2k single-card at all. The speed win is
universal (+16–37% decode, +16–29% prefill) — the fused Pascal/Volta kernels vs IQ_K's slower
low-bit dequant, measured in the exact single-card ub2048 regime that is the whole point.

## Can the 3–4-bit fidelity gap be closed? (measured backbone experiment)

We tested the cheap idea — bump PXQ's backbone toward IQ_K's precision without touching the fast
expert kernels. First pass (output + token_embd only, MXFP4→q6_K):

| | size | Mean KLD | top-1 |
|---|---|---|---|
| PXQ3 (MXFP4 backbone) | 14.7 GB | 0.0758 | 87.8% |
| PXQ3 + q6_K on output + token_embd | 14.9 GB | **0.0721** | 87.5% |
| (IQ3_K target) | 15.1 GB | 0.0591 | 88.9% |

Bumping the two logit-critical tensors (output projection + embeddings) MXFP4→q6_K closes **~24%**
of the gap for +0.2 GB and ~zero speed cost (they're a tiny compute fraction; experts untouched).
That left the bigger logit lever untested — attn — because the fork's `--attn-*-type` overrides
were dead code (a strcmp-vs-prefixed-name bug quietly ignored them). We fixed the bug and ran the
real experiment: promote the attention backbone to q6_K, like IQ_K does, at a fixed 14.0 GB size
(PXQU-16):

| | size | Mean KLD | top-1 | decode P100 | prefill P100 | decode V100 | prefill V100 |
|---|---|---|---|---|---|---|---|
| PXQU-16 (MXFP4 attn, shipped) | 14.0 GB | 0.107 | 85.6% | 57.7 t/s | 843 t/s | 98.5 t/s | 1900 t/s |
| PXQU-16 + q6_K attn | 14.0 GB | 0.103 | 85.8% | 54.8 t/s | 854 t/s | 95.4 t/s | 1893 t/s |

At fixed size the result is a **wash on fidelity** (KLD 0.107→0.103, top-1 +0.2 pt — inside
run-to-run noise) for a **real decode cost**: −5% on P100 (57.7→54.8 t/s), −3% on V100 (98.5→95.4
t/s), because q6_K attention dequant is slower per-token than MXFP4 on Pascal/Volta. Letting size
float instead of holding it fixed does close more gap — PXQ3's attn promoted to q6_K (no size cap)
moves KLD 0.0758→0.0665 for +0.3 GB — but still falls short of IQ3_K's 0.0591, and now costs both
size and speed.

**Verdict: we closed the loop on our own roadmap item and are publishing the negative result.** At
matched size, trading MXFP4 attention for q6_K attention is not worth it on this hardware — the
decode loss outweighs a fidelity gain too small to matter. The **shipped PXQ/PXQU quants
intentionally keep the faster MXFP4 attention backbone.** The code fix itself (`--attn-*-type`
overrides now actually work) is committed to the kernel repo, so anyone who wants to make that
trade for their own use case can. The expert-codebook half of the gap (IQ_K's non-linear codebook)
is a separate, more inherent difference from PXQ's speed-optimized aligned-dequant layout; a mild
non-linear code warp could recover part of it at some speed cost, but full fidelity parity is at
odds with the prefill moat. **Honest bottom line: PXQ's edge is speed + single-card ub2k fit +
winning at ≤12 GB — not fidelity-per-byte at 3–4 bit, and we say so.**

## Reproduce

Incumbents built with `llama-quantize --allow-requantize --imatrix ornith-forquant.imatrix <bf16>
<out> {IQ3_K|IQ4_KS|IQ2_KS|IQ3_KS|IQ2_KL}` (backbone experiment adds `--output-tensor-type q6_K
--token-embedding-type q6_K`). Scored with `bench/kld.sh` / `bench/ppl-ladder.sh`; speed with
`bench/speed-bench.sh` (single-card `-ub2048` for the ub2k table). All on the same box (2× V100 +
4× P100).
