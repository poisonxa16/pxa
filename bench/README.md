# bench/ — the PXQ repro pack

Every number we publish for **PXA Fusion2-35B** + **pxa** is reproducible from this
directory. If a number in the model card or a launch post isn't reproducible from here, treat it
as a bug and open an issue.

| claim | where to reproduce |
|---|---|
| Perplexity ladder (PXQ4 7.36 / PXQ3 7.44 / PXQ2 8.39, wikitext-2) | `ppl-ladder.sh` |
| KL-divergence vs the bf16 reference | `kld.sh` |
| Decode/prefill speed per card (V100 101.3 t/s, P100 62.4, 1080 Ti 71.4) | `speed-bench.sh` |
| 0 refusals / 57 adversarial prompts | `refusal-probes/` |
| 9/10 hard reasoning gauntlet, 11/12 factual, ~19/24 coding | `gauntlets/` |
| SWE-bench Lite 19.6% (11/56), official docker grading | `swe/` |
| Bit-exact kernel determinism gates (G1/G2/G3) | `determinism-gates.md` |
| File integrity | `checksums.sha256` |

## Reference hardware (what OUR numbers were measured on)

All published numbers were measured on one homelab box of salvaged cards. Match the config
below when comparing; report your own hardware when they differ (we'll add community numbers
to the compatibility table with credit).

- **GPUs:** Tesla V100-PCIE-16GB (sm_70), Tesla P100-PCIE-16GB (sm_60), GTX 1080 Ti 11GB (sm_61)
- **Driver:** NVIDIA 580.142 · **CUDA:** 12.8.1 (built + run inside `nvidia/cuda:12.8.1-devel-ubuntu24.04`)
- **Build:** `cmake -DCMAKE_CUDA_ARCHITECTURES="60;61;70" -DGGML_CUDA=ON`, targets
  `llama-server llama-cli llama-perplexity llama-quantize`
- **PCIe:** x4 per card (bifurcation risers) — decode is unaffected (single-card residency);
  only multi-card `-sm layer` hand-off touches the bus.
- **Runtime env (the bit-exact fast kernels):**
  `PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1 PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1`
- **2026-07-19 addition:** `PXA_FUSE_DELTANET=3` (DeltaNet decode glue-kernel fusion, bit-exact,
  default off) is now part of the recommended env — measured +3.7% decode on P100, +2.8% combined
  with the q8_0 head on V100. The updated numbers below include it.

## The published numbers, exactly

Perplexity: wikitext-2-raw **test**, `-c 512 --chunks 200 -b 512 -ub 512 -fa on -ctk f16 -ctv f16`
(the standard llama.cpp ppl protocol — comparable to other GGUF ppl tables at n_ctx=512):

| tier | file | wikitext-2 ppl (200 chunks) | Δ vs PXQ4 |
|---|---|---|---|
| PXQ4 (flagship, 4.27 bpw; formerly PXQ6 — the published files keep the `PXQ6` filename) | 18.7 GB | **7.3563 ± 0.0818** | — |
| PXQ3 (3.27 bpw) | 14.7 GB | **7.4407 ± 0.0830** | **+1.1%** |
| PXQ2 (2.27 bpw) | 10.7 GB | **8.3906 ± 0.0961** | **+14.1%** |

The ladder is monotonic. Same corpus, same protocol, same imatrix for every tier — only the
quant type varies. (KLD vs the bf16 merge source: `kld.sh` is the exact procedure; our own run
is queued behind production GPU occupancy and will be published as a follow-up — run it on
your card and beat us to it.)

Speed (measured via `llama-server` `timings.predicted_per_second`, 200-token generations,
median of 3, model fully GPU-resident — see `speed-bench.sh`):

| card | tier | decode | prefill |
|---|---|---|---|
| 1× Tesla V100 16 GB | PXQU-16 (q8_0 head) | **101.3 t/s** | ~1800–1900 t/s @ `-ub 2048` (5.8k-token cold prompt) |
| 1× Tesla P100 16 GB | PXQU-16 (q8_0 head) | **62.4 t/s** | 827–843 t/s @ `-ub 2048` |
| 1× Tesla P100 16 GB | PXQ3 | **55.8 t/s** | — |
| 2× Tesla P100 | PXQ4 (`*-PXQ6.gguf`) | 55.7 t/s | — |
| 1× GTX 1080 Ti 11 GB | PXQ2 | **71.4 t/s** (re-verified 2026-07-19: 70.9, within noise) | 248 t/s @ `-ub 768`; **709 t/s** with `PXA_PXQ_INT8_PREFILL=1` (opt-in, see note) |

Cells we didn't measure are absent, not implied. PRs with measured numbers welcome.

> **The 1080 Ti row is the 2026-07 protocol** (5.8k-token cold prompt, `-ub 768`) and is kept for
> continuity with the corrections below it. The current figures for that card, measured on the
> 2026-09-03 build against a fresh upstream ik_llama.cpp build under one stated protocol
> (n=3, `-b 2048 -ub 768`), are **1,306 t/s cold prefill and 59.2 t/s chat decode** — see the
> GTX 1080 Ti section of [`fair-battle.md`](fair-battle.md). The two are different prompts and
> different sampling, so they are listed separately rather than reconciled into one number.

**Corrections + notes (2026-07-19):**

- **Correction (withdrawn claim):** an earlier version of this table published "1× V100 + PXQ6 =
  99.1 t/s". That row was mis-attributed: the 4-bit flagship (PXQ4, formerly PXQ6) is **18.7 GB and
  verifiably does not load on one 16 GB card** (OOM on load, re-tested). We could not tie the 99.1 figure to a reproducible
  config, so it is withdrawn rather than re-labeled. The verified single-card V100 tier is
  **PXQU-16** (14.0 GB): 98.5 t/s under the original published env, **101.3 t/s** with the
  2026-07-19 env additions below. The 4-bit flagship on these cards is a 2×16 GB (`-sm layer`) configuration.
- **1080 Ti int8 prefill (2026-07-19, opt-in):** `PXA_PXQ_INT8_PREFILL=1` routes PXQ prefill
  GEMMs through an int8 dp4a tile on sm_61: cold 5.8k-token PXQ2 prefill 251.0 → **709.0 t/s**
  (+182%, median of 3; 95% of the native IQ2_KS MMQ incumbent's 744.7 on the same card). Decode
  is byte-untouched (66.4/66.5 t/s off/on; temp-0 sha identical). Default OFF because the int8
  path is not bit-exact vs the fp16 pipeline (temp-0 64-tok continuation sha-identical in our
  gates; top-1 logits identical 3/3 spot-checks, top-5 tail-order shifts at p≈0.015).
- **1080 Ti prefill is published at `-ub 768`:** a ~1.9 GiB `-ub 2048` compute buffer cannot
  allocate next to a resident ~10 GiB tier on an 11 GB card (verified for both PXQ2 and the IQ2
  incumbent) — ub768 is the largest matched setting that fits. Decode is ub-insensitive.
- **q8_0 head:** the U16 decode rows use the q8_0 output-head build (`--output-tensor-type q8_0`,
  +123 MB vs the q6_K head): +5.2% decode on P100 (57.2 → 60.2) because the single lm_head GEMV
  was ~14% of the P100 decode wall on the int8-emulation path. `PXA_FUSE_DELTANET=3` adds the
  rest (60.2 → 62.4 on P100; 98.5 → 101.3 on V100 for the combined env).
- **`PXA_G2_ADDFUSE=1` (late 2026-07-19, bit-exact):** residual-add fusion measured on top of
  the rows above by paired interleaved A/B: V100 100.1 → **102.0** (+1.9%, quiet window);
  P100 62.25 → **63.0** (+1.2%). Its siblings NORMFUSE/QUANTFOLD measured no additional gain
  and REDFUSE measured a loss — all stay in-tree default-OFF (`docs/lab/LEVERS.md`).

## Model files

The tiers are published at `huggingface.co/poisonxa/PXA-Fusion2-35B-GGUF`
(`PXA-Fusion2-35B-{PXQ2,PXQ3,PXQ6,PXQ6-MTP}.gguf` + `mmproj-fusion2-f16.gguf`; the `PXQ6` files
are the 4-bit PXQ4 tier under its pre-rename filename — see `docs/RENAME-MAP.md`).
Verify integrity against `checksums.sha256` in this directory.

⚠ PXQ is a PXA-native format: **mainline llama.cpp cannot read these GGUFs** — build this fork.
⚠ Never round-trip PXQ tensors through mainline `gguf-py` (its size table silently truncates
the E16-row anchors).
