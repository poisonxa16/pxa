# Qwen3.8-27B PXQ mix27: a one-card 131k file that is better than PXQ3-balanced

`Qwen3.8-27B-PXQ-mix27.gguf` is a per-tensor PXQ3/PXQ4 mix of Qwen3.8-27B, with q8_0 and q6_K tensors where
they pay. It is built for ONE 16 GB card at `-c 131072 -ctk q4_0 -ctv q4_0 -ub 256`. Compared with
PXQ3-balanced-ssmout-q8_0 it uses fewer GPU bytes and has a lower KLD and PPL against the Q8_0 source,
measured with the same instrument on the same hardware.

- sha256 `4af1ceedaae527ef177c9a672cab5890fbf22af506d6fb1d4661589f46a70522`, 13,897,297,920 bytes (13.91 GB)
- Lane b-mix27, burst 2026-09-24/25, branch `spd/b-mix27-20260925`
- Ledger rows: `pxq-mix-tensor-sensitivity-27b`, `pxq-head-shrink-27b` (both measured-win), and
  `pxq-imx-27b-moe` (NOT measured, still open: the model is dense, not MoE)

**Short version:** use it plain. It fits and runs at 131k on one V100 or one P100, and it passes the
determinism, logit-spread and 96k-needle gates. `--spec-type mtp` loads and serves short prompts at
131k, but it is not usable at depth yet (it runs out of memory on a ~96k prompt) and it is slower than
plain decode on normal text. See "MTP: what fits and what does not".

## Tier map (mix27 = arm "e20")

The base level is PXQ3 (balanced). There are three overrides, applied by `--custom-q` in one quantizer pass from the Q8_0 source:

| tensors | PXQ3-balanced | mix27 | why (measured, in-dist KLD vs Q8_0) |
|---|---|---|---|
| `ffn_up/gate/down`, blk 22-41 (20 layers) | pxq3 | **pxq4** | The middle layers carry the error. pxq3->pxq4 per 16-layer band: L0-15 -0.0065, L16-31 -0.0339, L32-47 -0.0366, L48-63 -0.0092 |
| `ffn_*`, the other 44 layers + MTP blk 64 | pxq3 | pxq3 | About 0.5e-3 KLD per layer at the edges, against 2.2e-3 in the middle |
| `ssm_out` (48 DeltaNet layers) | q8_0 | **pxq4** | +0.0070 KLD for -764 MiB. This is the cheapest byte source after the head, and the DeltaNet recurrence does not amplify the error |
| `output.weight` (head) | q8_0 | **q6_K** | -293 MiB. A pxq4 head would free 644 MiB for +0.0010, but a PXQ head with `--spec-type mtp` runs out of memory (bug #204), so the head stays a k-quant |
| attention `qkv/gate/q/output` | pxq4 | pxq4 | Dropping these to pxq3 costs +0.0256 for -660 MiB, the worst trade we measured |
| `attn_k/v`, `ssm_alpha/beta`, `nextn.eh_proj` | q8_0 | q8_0 | Small tensors |
| `token_embd` | q6_K | q6_K | The engine keeps it on the CPU (the 994.63 MiB CPU buffer) |

Census: pxq4 has 238 tensors (6.64 GB), pxq3 135 tensors (4.89 GB), q6_K 2 tensors (2.09 GB: the head,
plus token_embd on the CPU) and q8_0 131 tensors (0.27 GB).

### GPU bytes, as the engine reports them

These are the engine's own `CUDA0 buffer size` lines on one card. The MTP block (blk 64, 215.4 MiB) is
loaded only with `--spec-type mtp`, so there are two figures. bpw is GPU bytes divided by the 26.05 B
parameters that live on the GPU (27.321 B minus token_embd).

| file | CUDA0 weights, plain | with MTP | GPU bpw (plain) | engine "model size" line (whole file) | repeating layers |
|---|---|---|---|---|---|
| PXQ3-balanced-ssmout-q8_0 | 12464.19 MiB | 12679.6 MiB | 4.014 | 13.354 GiB, 4.199 BPW | 3.857 BPW |
| **mix27** | **12043.51 MiB** | **12258.87 MiB** | **3.878** | 12.943 GiB, 4.069 BPW | 3.814 BPW |
| mix27-a / mix27-c | 12075.99 / 12330.99 MiB | - | 3.889 / 3.971 | - | - |
| PXQ4 (stock, ssm_out off) | 14504.19 MiB (2 cards: 7433.94 + 7070.25) | - | 4.671 | 15.377 GiB, 4.835 BPW | 4.558 BPW |

The first version of this README used 12259 / 12680 / 14751 MiB, which are the tensor byte sums
including the MTP block. Those match the engine only with MTP loaded. The ninfer-5080 reference
(3.953 "effective bpw") does not say what it counts, so no figure here is directly comparable to it.
The closest analogues are the repeating-layer BPW (3.814) and the whole-file BPW (4.069).

## Quality: same corpus, same instrument, same hardware

Instrument: Qwen3.8's own chat-template corpus (`mkqwenchat.py`, 327k characters from oasst1, dolly,
gsm8k, code, glaive and wikitext-test), `PXA_PPL_PARSE_SPECIAL=1`, `-c 2048` x 24 chunks, `-b/-ub 512`,
`-fa on`, KLD against the logits of the Q8_0 source (the file every 27B PXQ file is cut from, PPL 7.947).
The binary is llama-perplexity @36e10ad954.

**P100 column (all three files on P100s).** PXQ4 does not fit one card with this instrument, so it ran
on a P100 pair. PXQ3-balanced ran on the same pair and mix27 on one P100 (card 1, 2026-09-25 05:55 EDT).

| file | Mean KLD | same top | PPL(Q) |
|---|---|---|---|
| PXQ3-balanced-ssmout-q8_0 | 0.1797 +/- 0.0069 | 90.29% | 8.702 |
| **mix27** | **0.1401 +/- 0.0061** | **91.44%** | **8.099** |
| PXQ4 (stock) | 0.0975 +/- 0.0055 | 93.63% | 7.806 |

**V100 column (one V100).**

| file | Mean KLD | same top | PPL(Q) |
|---|---|---|---|
| PXQ3-balanced-ssmout-q8_0 | 0.1786 +/- 0.0070 | 90.67% | 8.789 |
| **mix27** | **0.1387 +/- 0.0063** | **91.94%** | **8.065** |

On both cards mix27 has 22% lower KLD than PXQ3-balanced and uses 421 MiB less GPU memory. On the P100
column it closes 48% of the KLD gap from PXQ3-balanced to PXQ4. The two cards agree within 0.0014 KLD
for each file. PPL on this corpus is not a fidelity number: the text was written by other people and
models, and several sensitivity arms moved PPL the opposite way to KLD. Read KLD.

Candidates we scored and did not ship (V100 KLD):

| arm | change vs mix27 | KLD | CUDA0 plain | why not |
|---|---|---|---|---|
| mix27-a | head pxq4, FFN pxq4 on L16-47 | 0.1303 | 12075.99 MiB | PXQ head: `--spec-type mtp` runs out of memory at the first request (#204) |
| mix27-c | a + FFN pxq4 on L12-15 and L48-51 | 0.1214 | 12330.99 MiB | Same, and no MTP headroom |
| mix27-b | a + attention edges to pxq3 | 0.1398 | - | Attention bytes are the worst trade |
| mix27-e26 / e32 | q6_K head, FFN pxq4 on 26 / 32 layers | - | +200 / +400 MiB | With MTP at 131k, e26 runs out of memory at runtime and e32 at boot |

**Why e20 ships and not a or c.** Plain, mix27-c is the better file (KLD 0.1214, and it fits one V100
at 131k). mix27 (e20) was chosen because it is the only candidate that also loads `--spec-type mtp` at
131k on one card. Given the MTP results below, **choose mix27-c if you will never turn MTP on.** It is
in `arms/mix27-c.gguf` next to this file; it has not been through the gates.

## Ship gates (run once, on the shipping file, release binary v2026.09.20, one V100, 131k q4_0 ub256)

| gate | result |
|---|---|
| determinism, np 1: 12 greedy-512 runs | **12/12 identical** (sha `7b33b6a3`) PASS |
| determinism, np 2, slot-pinned, slot 1 idle (the established gate) | **12/12 identical**, and the same sha as np 1. PASS |
| logit spread: 10 identical prefills, token-0 top-1 probability, np 1 and np 2 | **spread 0, one distinct value** at both. PASS |
| needle at ~50% depth of a 96,173-token prompt (plain) | **found** (`cobalt-heron-7291`). Prefill 246.2 t/s, decode at depth 8.11 t/s, peak 15507 MiB. PASS |
| extra: np 2 with slot 1 decoding a different prompt at the same time | 3/12 (8 distinct shas). This is **not a mix27 property**: PXQ3-balanced gives 2/12 (9 distinct) in the same test. It is the open engine bug `qwen4exp-np2-kvunified-concurrent-nondeterminism` (batch-width-dependent numerics under concurrent decode) |

## 2-card -sm tensor (mailbox #8210)

Two P100s (cards 5 and 6), build spd/b-dim0-20260925 @4ef8b0a61d (sm_60), plain decode, REPS 3:

| split | PPL (b-dim0 harness) | decode ctrl / rep |
|---|---|---|
| -sm layer | 7.6051 +/- 0.305 | 15.23 / 15.23 t/s |
| -sm tensor (fused) | 7.6099 +/- 0.305 | **20.31 / 20.26 t/s (+33.4%)** |

The file boots and splits correctly on 2 cards. The token-0 distribution matches between layer and
tensor splits ('#' 0.802 vs 0.801). Main's hotfix #206 moves 4-card `--sm auto` to layer, because the
release engine serves wrong tokens on a 4-way tensor split. The 2-card split is unaffected.

## Fit and speed (one card, -c 131072, q4_0 K/V, -ub 256, -fa on, -np 1)

All rows are REPS 3 medians after a 60 s warm-up: greedy 512 tokens in two classes (control and
repetition), then a 22.6k-token prefill. Plain arms set `PXA_AUTO_SPEC=0`, because without it the
server arms an n-gram drafter on its own. VRAM is nvidia-smi on the card. "Peak" is 1 s samples during the run.

| card | build | file | spec | VRAM | decode ctrl / rep (t/s) | prefill 22.6k |
|---|---|---|---|---|---|---|
| V100 | release | PXQ3-balanced | plain | boot 15549 | 30.04 / 29.69 | 511.8 |
| V100 | release | **mix27** | plain | boot 15127 | **32.49 / 32.10** | 511.0 |
| V100 | eng3 ab8c20e42a | **mix27** | plain | peak 15319 | 32.07 / 31.81 | 511.3 |
| V100 | release | mix27 | mtp (gpu-fallback) | boot 15809, peak 16007 | 18.55 / 40.88 (accepted 299/633, 379/396) | 508.1 |
| V100 | eng3 ab8c20e42a | mix27 | mtp (gpu-fallback) | peak 16007 | 18.39 / 40.57 (299/633, 379/396) | 509.3, then **OOM on a ~96k prompt** |
| P100 | release, same bracket | PXQ3-balanced | plain | peak 15573 | 13.08 / 12.98 | 178.4 |
| P100 | release, same bracket | **mix27** | plain | peak 15151 | **13.65 / 13.55** | 179.7 |
| P100 | eng1 e8e1a4530a | mix27 | plain | boot 15061 | 15.29 / 15.27 | 181.0 |
| P100 | eng1, eng2 | mix27 | mtp | boot 15743 / 15593 | - | OOM at the first request |
| P100 | eng3 ab8c20e42a | mix27 | mtp (gpu-fallback) | boot 15743, peak 16003 | **6.67** / 13.41 (305/613, 378/399) | 181.3 |

"Release" is the packaged v2026.09.20 engine. eng1, eng2 and eng3 are 's MTP-memory fix builds.
The P100 same-bracket rows ran at the same time (05:36-05:47 EDT, cards 0 and 1).

### MTP: what fits and what does not

- **Plain serving** fits on one V100 or one P100 at 131k. It served a 96k-token prompt, found the needle,
  and peaked at 15.5 GB. On the same card, mix27 is faster than PXQ3-balanced: +8% decode on V100 and +4% on P100.
- **`--spec-type mtp` loads at 131k and serves short prompts.** On V100 it works on the release binary; on
  P100 it needs b-mtpfit's eng3 build, which slices the head's f16 copy. The peak is 16003-16007 MiB,
  about 140 MiB below the card limit.
- **MTP is not proven at depth.** On V100 eng3, a ~96k-token prompt runs out of memory in the WMMA
  flash-attention path (`flash_attn_ext_wmma_f16_case<256,256,8>` converts the whole quantized KV). This
  was reported to b-mtpfit (#8224); b-mtpfit's own 118k arm hit the same wall (#205).
- **MTP is slower than plain on normal text.** Control class: -43% on V100 and -51% on P100 (on P100
  every verify round dequantizes the 2.4 GiB head through the sliced cuBLAS path). Only the
  repetition class gains, +27% on V100, and on P100 it is flat. Keep MTP off for this file until the
  verify path is cheaper.
- **Greedy output with MTP is not byte-identical to plain on the control class.** The sha is `09fc905f`
  with MTP and `7b33b6a3` plain, on the same card and binary. The repetition class matches. This is the
  known engine property `spec-verify-batch-invariance` / `mtp-verify-batch-shape-not-byte-lossless`:
  the target's logits depend on the verify batch width, so near-tie tokens can flip. MTP output is
  stable across its own runs (3/3 identical).

## Recipe

`make-mix27.sh` is the recipe: one `pxq-quantize` pass with the release quantizer from the Q8_0 source.

    --custom-q 'ssm_out\.weight=pxq4,^output\.weight=q6_K,^blk\.(2[2-9]|3[0-9]|4[01])\.ffn_(up|gate|down)\.weight=pxq4'   base type PXQ3

The measured arms were byte-merged from per-tier source files (`mix-merge.py`, `build-mix.sh`). The
direct quantize output matches that merged file tensor for tensor: 866/866 sha256 (`verify-identity.py`).

Harness:
- `kld-arm.sh` scores one arm.
- `arms.sh` runs the one-group sensitivity arms.
- `speed-arm.sh` does the one-card fit and speed probe: REPS 3, control and repetition classes, greedy512 sha, 22.6k prefill and peak VRAM.
  `GATES=` or `POSTGATES=` run `gate-client.py` (`det`, `detq`, `spread`, `needle`), and `NP=2` sets `-np`.
