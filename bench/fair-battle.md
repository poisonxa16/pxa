# Fair battle — pxa vs upstream ik_llama.cpp (2026-07-19, rev 2)

Best-vs-best per card: upstream at its own documented best (pinned 2026-07-18 HEAD,
`GGML_CUDA_F16=ON` build per its docs, its best-fitting IQ_K quant, f16 KV), pxa at its
documented best (`docs/lab/LEVERS.md` recommended env; `PXA_PXQ_INT8_PREFILL=1` on sm_61). Same 35B
MoE architecture, same card, same protocol both sides.

**Protocol:** single `/completion`, cold 5,801-token prompt, `n_predict=200`, `temperature=0`,
`seed=42`, `cache_prompt=false`, median of 3, numbers from server `timings`.

**Rev 2 (same day):** the first pass ran `-b` = `-ub` and forced `-fa on` for everything. A follow-up
sweep showed both choices leave real speed on the table **for both engines**: `-b 2048` is
+4–25% prefill, and on these cards **`-fa off` is the cold-prefill regime while `-fa on` is the
decode regime** (see the regime table — the effect is symmetric, upstream gains from FA-off prefill
too). Rev 2 gives each side its best regime per metric. Nothing was re-measured for one side only.

## Headline (best config per side, per metric — prefill @ `-fa off`, decode @ `-fa on`, all `-b 2048`)

| card | quant (upstream vs pxq) | prefill t/s | decode t/s |
|---|---|---|---|
| Tesla P100 16 GB | IQ3_KS (14.2 GB) vs PXQU-16+q8head (14.1 GB) | 645 → **1,213 (+88%)** | 44.7 → **58.1 (+30%)** |
| Tesla V100 16 GB | IQ3_KS (14.2 GB) vs PXQU-16+q8head (14.1 GB) | 1,509 → **1,700 (+13%)** | 84.5 → **95.5 (+13%)** |
| GTX 1080 Ti 11 GB | IQ2_KS (10.1 GB) vs PXQ2 (10.7 GB) | **1,154** → 1,001 (−13%) | 52.2 → **65.4 (+25%)** |

> **Note:** the decode deltas in this table reflect the smaller PXQ quant class (PXQU-16 + q8_0 head, or PXQ2) **plus MTP speculative decode**, not the engine. The engine same-quant decode is **+2.7–3.3%** (V100 bit-identical) — see the Single-config view below. The engine win is prefill.

## Single-config view (chat serving: `-fa on -b 2048`, one server, no regime switching)

| card | upstream (prefill / decode) | pxa (prefill / decode) |
|---|---|---|
| P100 | 513 / 44.0 (ub2048) · 44.7 dec best (ub512) | **817 / 56.7** (ub2048) · **58.1** dec best (ub512) |
| V100 | 1,422 / 82.5 (ub512) | **1,589 / 94.1** (ub512) |
| 1080 Ti | 739 / 52.2 (ub768) | 667 / **65.4** (ub768) |

## The FA regime split (both engines, measured)

Flash-attention on these pre-Turing cards is a decode win but a cold-prefill loss — for upstream too:

| card / engine | prefill fa-on → fa-off | decode fa-on → fa-off |
|---|---|---|
| P100 pxq | 817 → **1,213** (+48%) | **56.7** → 41.1 (−28%) |
| P100 upstream | 513 → **645** (+26%) | **44.0** → 34.0 (−23%) |
| V100 pxq | 1,589 → **1,700** (+7%) | **94.1** → 76.6 (−19%) |
| V100 upstream | 1,422 → **1,509** (+6%) | **82.5** → 69.6 (−16%) |
| 1080 Ti pxq | 667 → **1,001** (+50%) | **65.4** → 34.2 (−48%) |
| 1080 Ti upstream | 739 → **1,154** (+56%) | **52.2** → 30.0 (−43%) |

Practical rule either engine's users can apply: prefill-heavy batch work (ingest, embedding prep,
summarize-once) → `-fa off`; interactive serving → `-fa on`. Measured at 5.8k-token fill; FA-off
attention memory grows with context, so re-check at your target ctx.

## Same-quant control (identical gguf on both builds)

pxa running upstream's own IQ_K ggufs — only the arch-level fusions differ
(`PXA_FUSE_DELTANET=3 PXA_G2_ADDFUSE=1`), matched config both sides:

| card | quant | upstream decode | pxa decode | Δ | output |
|---|---|---|---|---|---|
| V100 | IQ3_KS @ub512 | 84.5 | 87.2 | +3.2% | **bit-identical** (same temp-0 sha) |
| P100 | IQ3_KS @ub2048 | 44.0 | 45.2 | +2.7% | coherent, sha-stable runs |
| 1080 Ti | IQ2_KS @ub768 | 52.2 | 53.9 | +3.3% | coherent, sha-stable runs |

## Where upstream wins, and why (kept on the chart)

The 1080 Ti cold-prefill loss (−13%) is real: upstream's IQ2_KS MMQ int8 tile is a mature,
double-buffered, large-tile pipeline and its file is 6% smaller; our sm_61 int8 tile
(`PXA_PXQ_INT8_PREFILL`, first shipped this release) is a 64-thread single-buffered first cut that
reaches ~87–95% of it depending on config — and didn't exist at all a release ago (PXQ2 prefill on
that card was 251 t/s). Decode on the same card is +25% for pxa.

## Raw harness rows

```
# rev-1 rows (b = ub, fa on)
vik_V100_IQ3KS_ub512,OK,1154.8,84.5   | vik_V100_IQ3KS_ub2048,OOM (launch_fattn)
vik_P100_IQ3KS_ub512,OK,303.1,44.7    | vik_P100_IQ3KS_ub2048,OK,512.6,44.0
vik_1080Ti_IQ2KS_ub768,OK,702.8,52.2
pxq_V100_U16q8_ub512,OK,1268.6,95.5   | pxq_V100_U16q8_ub2048,OOM
pxq_P100_U16q8_ub512,OK,604.9,58.1    | pxq_P100_U16q8_ub2048,OK,816.8,56.7
pxq_1080Ti_PXQ2_ub768,OK,639.0,64.6
samequant_V100_IQ3KS_ub512,OK,1124.5,87.2 (sha matches upstream)
samequant_P100_IQ3KS_ub2048,OK,536.0,45.2
samequant_1080Ti_IQ2KS_ub768,OK,685.5,53.9
# rev-2 sweep rows (b 2048; fa as labeled)
dxV1_pxq_b2048_faon,OK,1588.5,94.1    | dxV2_pxq_b2048_faoff,OK,1699.5,76.6
dxV3_vik_b2048_faon,OK,1421.9,82.5    | dxV4_vik_b2048_faoff,OK,1508.7,69.6
dxP1_pxq_ub2048_faoff,OK,1213.1,41.1  | dxP2_vik_ub2048_faoff,OK,645.0,34.0
dxA_pxq2_b2048_faoff_c6144,OK,1000.6,34.2 | dxB_pxq2_b2048_faon_c8192,OK,667.3,65.4
dxC_pxq2_b768_faoff_c8192,OK,975.7,33.9
dxD_vik_iq2ks_b2048_faoff_c6144,OK,1153.6,30.0 | dxE_vik_iq2ks_b2048_faon_c8192,OK,738.8,52.2
```

## Codec-only re-run, 2026-09-02 (candidate engine, next release)

A separate comparison from everything above: same engine on both sides of every row, PXQ4 vs
MXFP4 at matched engine — isolating the codec, per `bench/fair/protocol.md`. `llama-server
/completion`, `temp=0`, `seed=42`, n=7 median, 1 warmup discarded, `cache_prompt=false`, unique
prompt per repeat, MTP off both sides, artifacts sha-checked against
`bench/fair/weights/MANIFEST.sha256`. Run on the engine described in
`RELEASE-NOTES-2026-09-02.md` and `RELEASE-NOTES-2026-09-07.md` (pipeline-scheduler fixes,
host-overhead cuts, the ported upstream correctness fixes) — the engine in this release.

### Dense, 2×P100 (Qwable-27B, `-c 32768 -b/-ub 2048 -fa on`)

| | PXQ4-core | MXFP4-lite | result |
|---|---|---|---|
| prefill @3,121 | 227.4 | 178.7 | **+27%** |
| prefill @20,801 | 203.3 | 163.7 | **+24%** |
| decode @8 | 18.3 | 14.4 | **+27%** |

### Dense, 2×V100 (Qwable-27B, `-c 32768 -b/-ub 2048 -fa on`) — the cell PXQ4 still loses

| | PXQ4-core (`PXA_PXQ_MMVQ=1`) | MXFP4-lite | result |
|---|---|---|---|
| prefill @3,121 | 798.1 | 364.0 | **+119%** |
| prefill @20,801 | 577.0 | 312.9 | **+84%** |
| decode @8 | 34.5 | 37.7 | **−8.3% — MXFP4 wins here** |

Prefill's lead widened since the original table because the candidate engine's NCCL P2P fix
(`docs/PXA-SM70-SERVING.md`) helps both quant types and PXQ4 was already ahead on prefill; the
decode loss is the same structural DP4A-scale-fixup story as the original table, unchanged by
the engine update.

### MoE, 2×V100 (35B-A3B class, `-c 32768 -b/-ub 2048 -fa on`) — expert-codec delta

| | PXA-Fusion4-35B PXQ4 | qwen36-35b MXFP4 | result |
|---|---|---|---|
| prefill @3,121 | 2,093.8 | 1,614.6 | **+30%** |
| prefill @20,801 | 1,467.2 | 1,256.8 | **+17%** |
| decode @8 | 218.4 | 188.0 | **+16%** |

Both files: `PXA-Fusion4-35B-PXQ4.gguf` from
`huggingface.co/poisonxa/PXA-Fusion4-35B-GGUF` and `qwen36-35b-MXFP4.gguf` from
`huggingface.co/poisonxa/PXA-bench-files-GGUF`, shas in `bench/fair/weights/MANIFEST.sha256`.

**Caveat, stated plainly:** the PXQ4 side is `PXA-Fusion4-35B`, a merge — not the same base
weights as the stock 35B-A3B model on the MXFP4 side. Same architecture and size class, not
byte-identical. The artifact's own codec census (`codec=PXQ4 (120 tensors; pxq4 120)`) shows only
its expert stacks are actually PXQ4; the rest is other types. This supersedes an earlier
MoE-decode row this repo withdrew after finding the same composition problem in a different
artifact — the fix here is disclosure, not a different codec: the number is real for
"PXQ4-quantized experts on this architecture," not for "PXQ4 the whole model."

## 2026-09-03 cross-engine, same file, three engines

Everything below is upstream ik_llama.cpp, mainline llama.cpp, and this engine, run on the exact
same weight file on the same cards, same day. Not a codec comparison: MXFP4 rows are the same
quant type on all three engines, so this isolates engine-level prefill and decode work. `temp=0`,
`seed=42`, `-c 32768 -b/-ub 2048 -fa on`, `/completion`, prefill sampled at 3,121 and 20,801
tokens, decode at batch 8. ik at HEAD `3c58ae37` (`GGML_CUDA_F16=ON`), mainline at `9400c89`
(2026-09-02). Model file: `Qwable-27B-MXFP4-lite.gguf`, sha per
`bench/fair/weights/MANIFEST.sha256`. Greedy 32-token continuation is byte-identical across all
three engines on the MXFP4 rows.

### 2x V100

The last three rows are the **serialized n=7** sweep on the folded build `24ebec4096`; the rows
above them are earlier development measurements, kept because they show where the engine's own
MXFP4 path was and what fixed it.

| engine | quant | prefill @3,121 | prefill @20,801 | decode @8 |
|---|---|---|---|---|
| upstream ik `3c58ae37` | MXFP4 | 470.8 | 395.3 | 37.4 |
| mainline `9400c89`, `-ub 2048` | MXFP4 | 939.6 | 1,123.7 | not captured |
| mainline `9400c89`, `-ub 512` | MXFP4 | 936.4 | 1,129.1 | not captured |
| ours, before the precision-routing fix | MXFP4 | 367 | 316 | 37.7 ‡ |
| ours, with the precision-routing fix | MXFP4 | 851 | 588 | 37.7 ‡ |
| ours, earlier build | PXQ4 | 831-838 | 577-592 | 38.6 ‡ |
| **ours, folded build, `-b 8192 -ub 2048`** | PXQ4 | **1,339.6** [1,334-1,342] | **1,281.9** [1,274-1,287] | **39.67** [39.62-39.68] |
| ours, folded build, `-b 8192 -ub 512` (pass a / b) | PXQ4 | 1,251.7 / 1,237.8 | 1,176.3 / 1,162.7 | 39.53 / 39.61 |

The pre-fix MXFP4 row is this engine's own MXFP4 path falling through to a plain `cublasSgemm`
instead of tensor cores on every fused-FFN GEMM (see `RELEASE-NOTES-2026-09-07.md`), not a codec
cost; the fix recovers most of the gap to mainline.

Against each competitor's best cell, the folded build is ahead everywhere: **+42.6%** at 3,121
tokens over mainline (1,339.6 against 939.6), **+13.5%** at 20,801 tokens over mainline (1,281.9
against 1,129.1), and **+6.1%** decode over upstream ik (39.67 against 37.38). The 20,801-token
cell is the one that changed: it was 81% of mainline earlier the same day and is now a win.

**What closed it was the prefill chunk size, and that was checked for fairness.** `-b` is the
chunk size, and the engine synchronizes on every chunk boundary — 14 of them over a 20,801-token
prompt at the default `-b 2048`, 3 at `-b 8192`. On this pair at `-ub 512` that is 1,097/1,092 t/s
against 1,207/1,204, with summed device utilisation going from 1.24-1.33 to 1.41-1.42; `-b 20480`
adds another 0.3%. Mainline was then run at the same chunk sizes on the same cards and **does not
gain from them**: 936/1,167 t/s at `-b 8192` and 942/1,142 at `-b 20480`, against 1,165 at its own
default. So the mainline rows above use whichever configuration was best for mainline, and
`-b 8192` is a launch flag that helps this engine's scheduler rather than a handicap on the other.
Greedy output was byte-identical across every chunk-size arm and the needle was recalled in all of
them. The engine's own default stays at `-b 2048` because a larger chunk costs compute buffer; see
`docs/KNOWN-ISSUES.md`.

Device overlap is still the structural weakness even though this cell now goes our way: the
scheduler host-synchronizes before each split's input copy, the default layer split is unbalanced
(0.58/0.36 device busy against mainline's roughly 1.1 ratio), and there is no cross-graph
pipelining. Larger chunks remove synchronizations; they do not fix that. See
`docs/DELTA-SINCE-IK.md`, "A scheduler difference from mainline, named".

**‡ Protocol.** The four unmarked folded-build cells are n=7 serialized, one arm at a time on a
quiet box, brackets showing observed min-max. Rows marked ‡ are earlier development measurements
at 2 runs per arm, and their decode figures come from a different arm of the same day than their
prefill figures; they are kept for the story they tell about the MXFP4 path, not as release
numbers.

### 2x P100

Ours is the **serialized n=7** sweep on the folded build `24ebec4096` at `-b 8192 -ub 256`, two
passes; competitors in the same session at their defaults, except the upstream ik row which is
from an earlier session.

| engine | quant | prefill @3,121 | prefill @20,801 | decode @8 |
|---|---|---|---|---|
| upstream ik `3c58ae37` | MXFP4 | 134.5 | 84.0 | 14.3 |
| mainline `9400c89` | MXFP4 | 209.1 | 254.7 | not captured (harness strips `n_past`) |
| ours, earlier build | MXFP4 | 181 | 165 | 14.3 |
| ours, earlier build, `-ub 2048` | PXQ4 | 231 | 205 | 18.3 |
| **ours, folded build, `-b 8192 -ub 256`** (pass a / b) | PXQ4 | **340.2** [339.9-340.3] / 339.1 | **316.8** [316.8-316.9] / 316.6 | **17.83** / 17.79 |

Against mainline: **+62.7%** at 3,121 tokens (340.2 against 209.1) and **+24.4%** at 20,801
(316.8 against 254.7). Against upstream ik: **+24.9%** decode (17.83 against 14.27). This pair sat
at 80% of mainline at 20,801 tokens before the chunk-size change, the same signature as the V100
pair had.

Note the `-ub` difference between the two pairs: `-ub 256` is right here and `-ub 2048` is right
on the V100 pair. The small-`-ub` result does not transfer, and neither does the large one — on
the earlier build at `-b 2048`, `-ub 256` gave 218.4 t/s at 3,121 tokens against 231 at `-ub 2048`
on this pair, and the order reverses at `-b 8192`. Measure it on your own cards rather than
carrying a number across.

### GTX 1080 Ti, published PXQ2 tier vs a fresh upstream build

Published `PXA-Fusion2-35B-PXQ2.gguf`, sha `c11b45ef` (`bench/checksums.sha256`; that is the
filename it carries in the Hugging Face repo -- `fusion2-35b-PXQ2.gguf` is the same bytes under
its pre-publication name), against upstream ik_llama.cpp's own IQ2_KS requantized fresh from
`pxa-35b-fusionv2-bf16.gguf` at ik HEAD `3c58ae37` (no separate file sha recorded for the fresh
requantize; the ik commit pins its quantizer).

**Protocol:** the folded build `24ebec4096`, **n=3**, `-b 2048 -ub 768 -ngl 99`, cold prefill at
`-fa off`, chat prefill and decode at `-fa on`, greedy shas stable 3/3 in every arm. Brackets are
the observed min-max across the three runs.

| engine | quant | cold prefill (`-fa off`) | chat prefill (`-fa on`) | decode, cold / chat |
|---|---|---|---|---|
| upstream ik `3c58ae37` | IQ2_KS | 1,132.1 [1,107.7-1,147.2] | 740.0 [727.5-746.4] | 30.4 / 53.3 |
| **ours, folded build** | PXQ2 | **1,306.0** [1,286.9-1,309.7] | 729.3 [726.3-730.4] | **32.1** / **59.2** |

Cold prefill **+15.4%** and chat decode **+11.1%** over upstream; cold decode +5.6%.

**Chat prefill is a tie, and is reported as one.** 729.3 against 740.0 is nominally −1.4%, but the
two spreads overlap — ours 726.3-730.4, upstream's 727.5-746.4 — so three runs per arm cannot
separate them. It is neither a win nor a loss at this sample size, and calling it either would be
reading noise. Chat prefill on this card is flash-attention dominated; the sm_61 FA path is the
lever that would move it, and it has not been written.

**These numbers need the sm_61 int8 prefill tile, and it is worth knowing how to confirm it is
on.** The same folded build with the tile inactive does 573.4 cold and 423.9 chat prefill — less
than half. The tile is armed by the default ENHANCE config level (no environment variable needed since
2026-09-03), which resolves `PXA_PXQ_INT8_PREFILL` to mode 1 on sm_61 silicon; the byte-keyed
2-bit weight decode it uses (`PXA_PXQ_I8_BLUT`) is on by default and is bit-identical either way.
Both print a line at startup:

```
PXA_PXQ_INT8_PREFILL: mode 1 (N13 dp4a int8 MMQ-tile prefill, sm_61 only; ...)
PXA_PXQ_I8_BLUT: ON (N14 byte-keyed 2-bit W-decode, bit-identical)
```

If those two lines are not in your server log, you are not measuring the configuration above.
`PXA_PXQ2_MMQ=1` (the N16 tile) makes no difference on this build — 1,301 / 728.6, inside the
spread — because the int8 path already carries that work.

An earlier branch build measured this cell at 1,252.8 cold prefill and a higher chat decode
figure under a protocol these rows do not reproduce line-for-line; the rows above are the folded
build under one stated protocol, and they replace it rather than sitting beside it.

## 2026-09-05 release build, bare command line (no `PXA_*` env, no `-b`/`-ub`)

Everything below is the `v2026.09.05` release binary with genuinely nothing exported and no
batch flags on the command line: `PXA_ENHANCE` is the default config level, and the server
auto-picks `-b`/`-ub` for the card set it finds, printed at startup
(`PXA_AUTO: batch defaults for ... -> -b ... -ub ...`). This is what an operator who reads no
documentation gets, superseding the hand-tuned `-b 8192` rows above as the release's headline
numbers; the hand-tuned rows stay for the story of how the chunk-size finding closed the
20,801-token cell.

### 2x V100

| engine | quant | prefill @3,121 | prefill @20,801 | decode @8 |
|---|---|---|---|---|
| upstream ik `3c58ae37` | MXFP4 | 470.8 | 395.3 | 37.4 |
| mainline `9400c89` | MXFP4 | 939.6 | 1,129.1 | not captured |
| **pxa, bare command line** | PXQ4 | **1,368.75** [1,367.73–1,369.78] n=2 | **1,299.55** [1,297.71–1,301.39] n=2 | **39.49** [39.48–39.50] n=5 |

Auto-picked `-b 8192 -ub 2048`, identical to the hand-tuned control's flags; greedy sha
`04b297ff3ce23d05`. Against ik's decode: **+5.6%**.

### 2x P100

| engine | quant | prefill @3,121 | prefill @20,801 | decode @8 |
|---|---|---|---|---|
| upstream ik `3c58ae37` | MXFP4 | 134.5 | 84.0 | 14.3 |
| mainline `9400c89` | MXFP4 | 209.1 | 254.7 | not captured |
| **pxa, bare command line** | PXQ4 | **337.6** | **315.3** | **18.1** |

Measured 2026-09-05 21:22 EDT on the release binary (`2729f12060`), GPUs 1 and 5, quiet box (the production seat
paused, nothing else running), no `PXA_*` environment and no `-b`/`-ub` (the engine chose
`-b 8192 -ub 256` itself): prefill n=3 (337.43–337.69 and 315.24–315.30), decode n=12 at fill 8
(18.13–18.15), greedy sha `433a1c4516d83d77`. Harness `defaults-bench/run-final-cell.sh p100`.
Against the n=7 reference on the previous fold (340.16 / 316.84 / 17.83, 2026-09-03 P100 table
above) that is −0.8 % / −0.5 % / +1.7 %, inside the reference's own spread.

### 1x GTX 1080 Ti

| engine | quant | cold prefill (`-fa off`) | chat prefill (`-fa on`) | decode, cold / chat |
|---|---|---|---|---|
| upstream ik `3c58ae37` | IQ2_KS | 1,132.1 | 740.0 | 30.4 / 53.3 |
| **pxa, bare command line** | PXQ2 | **1,363.5** [1,359.4–1,369.0] | 746.6 | 36.73 / **65.30** |

Auto-picked `-b 2048 -ub 768`, `--ctx-checkpoints 0`, declined speculation on this single-card
sm_61 pair. Cold sha is bit-identical to the hand-tuned control; chat differs only in
`--ctx-checkpoints` (0 vs 32). Cold prefill **+4.4%** over the previous hand-tuned cell, and
against ik: chat prefill is a **tie** (746.6 sits at the edge of ik's own 727–746 spread), chat
decode **+22%**.

These bare-command-line rows are the ones the release notes and README headline table quote;
`RELEASE-NOTES-2026-09-07.md` carries the full reproduction commands.
