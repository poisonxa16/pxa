# Release notes — 2026-08-09

Covers everything since the previous release (2026-07-31).

> ## ⚠ Read this before quoting any number below
>
> **Almost every performance lever in this release ships DEFAULT OFF.** A stock build of this tag
> behaves, on the codec decode path, essentially like the previous one. The headline **+19.9%
> decode** is real, reproducible, and gated — but it requires **one build flag and two environment
> variables**, listed in full under "Opt-in performance" below. Nothing here is a claim about what
> you get by simply upgrading.
>
> They ship off on purpose. Two of them (`PXQ_CANON_V2`, `PXQ6_CANON_CMAX`) change the canonical
> fp32 fold, which is a **re-baselining event** — every stored control becomes non-comparable. Two
> more (`PXA_FA_GQA_PACK`, `PXA_FA_GQA_QSMEM`) are not bit-exact against the shipped V-pass order
> and are held until a wider coherence gate is recorded. Turning them on is a decision, so it is
> yours to make, not ours to make silently.

Headline: **122B-A10B decode on 4× P100 went 27.69 → 33.20 t/s (+19.9%)**, prefill unchanged.
That is a landfill-Pascal fleet running a 122B at 33 t/s.

---

## ⚠ Breaking change — `--pxq-universal` now takes a map path

The three bundled tier-map presets (`12g`, `16g`, `16g-hq`) and the reference `.tiers` recipe files
have been **removed from the current tree**, along with the copies compiled into `llama-quantize`.
They were per-tensor bit allocations computed for specific model layouts — a recipe, not part of
the codec. (As for any change to any repository, earlier commits in the published history still
contain them; this is a change to what ships going forward, not a retraction of the past.)

```
# before
llama-quantize --imatrix m.imatrix --pxq-universal 16g in.gguf out.gguf PXQ_UNIVERSAL

# now
llama-quantize --imatrix m.imatrix --pxq-universal my-16gb.tiers in.gguf out.gguf PXQ_UNIVERSAL
```

The mechanism is unchanged and the map format is deliberately trivial — `#`-commented lines of
`regex=type`, one per expert tensor, types `pxq1|pxq2|pxq3|pxq4|pxq6`. `docs/PXQU-CONVERT.md` now
documents the format in full, plus the budget/composition splits behind the published 122B-A5B
builds, so you can author or generate a map for your own tensor names and VRAM budget. A bare name
still resolves against `$PXA_PXQU_DIR`. **Already-quantized PXQU GGUFs are unaffected** — this is a
quantize-time flag only.

## What is ON by default in this release

| | |
|---|---|
| `PXA_FA_TILE256` | **On.** Pre-Volta cards had no tile/mma prefill kernel for the D=256 head, so `Q->ne[0] == 256` forced the single-column VEC kernel at any batch size — every query column re-streamed the whole KV extent with no tile reuse. On a 122B-A10B (head 256) at fill 8881 on 4× P100 that kernel was **52.7% of prefill GPU time** (60 launches × 281 ms avg). Batch>8 D=256 attention now routes to the tile-f16 (ncols=16) kernel. **Decode (`ne1 <= 8`) is untouched.** `=0` reverts. |
| `PXA_AUTO_SPEC` | **Arms under `PXA_ENHANCE=1`** (`=1` forces on at any level, `=0` off, never under `PXA_REFERENCE`). See "Self-speculation" below. |
| correctness + robustness fixes | Always on — see that section. |

Everything else new in this release is opt-in.

## Opt-in performance — the 122B/P100 decode campaign

**How to turn it on** (Pascal; re-baseline your own controls afterwards, the fold changes):

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DPXQ6_CANON_CMAX=4
PXA_FA_GQA_PACK=4 PXA_FA_GQA_QSMEM=1 ./build/bin/llama-server …
```

Cell: 122B-A10B PXQU48-core, 4× P100, `-sm layer -c 32768 -np 1 -b 512 -ub 512 -fa on`, f16 KV,
`PXA_ENHANCE=1 PXA_P100_FP16_GEMM=1 PXA_ROUTER_FUSE=3`, temp 0, fill 8881, n=3 medians, both
binaries alternating in one run, engagement asserted from the server banner in every cell:

| cell | levers | deep decode | shallow | prefill |
|---|---|---:|---:|---:|
| baseline (shipped defaults) | none | 27.69 | 31.72 | 300.70 |
| + attention pack | `PXA_FA_GQA_PACK=4` + `QSMEM` + split target 160 | 30.67 | 31.94 | 300.76 |
| + chunk cap | `-DPXQ6_CANON_CMAX=4` | 30.56 | 35.62 | 300.59 |
| **both** | | **33.22** | 35.61 | 301.11 |
| both, repeat | | 33.19 | 35.56 | 300.70 |

**33.20 t/s from 27.69 — +19.9%, 36.12 → 30.12 ms/token. Prefill unchanged at ~300 t/s.** The two
levers overlap rather than add (both remove memory traffic): the chunk cap alone is +10.4%, the
attention pack + split retune adds +8.6% on top of it.

Both reorder fp32 accumulation, so neither is bit-exact by construction and both were gated:
perplexity **on the decode path** (`-b 1 -ub 1`, 6 chunks) 3.8109 → 3.8027, inside the stderr and
slightly lower; 256-token greedy generation from an 8881-token prompt **byte-identical**, plus four
greedy chat probes byte-identical.

⚠ The first version of that gate was **vacuous** and nearly passed as evidence: `llama-perplexity`
at `-b 512` evaluates 512-token batches, so `ne01 > 1` and the `ncols == 1` vector kernel
`PXA_FA_GQA_PACK` replaces never executes — both arms were trivially identical. **Gate decode
levers at `-b 1 -ub 1`.**

⚠ `PXQ6_CANON_CMAX=4` evidence is **sm_60 only**. On sm_70 the split target is 16×nsm, so the S the
driver selects differs and the analysis does not carry over. Measure before adopting it there.

## Opt-in performance — codec decode paths (all bit-exact, hash-verified)

All three default OFF. Same cell as above unless stated; every arm reproduced the standing temp-0
hash, which is the designed outcome for a sourcing-only change.

| lever | measured |
|---|---|
| `PXA_PXQ6_SHFL=1` | **+4.1%.** Sustained 2048 tokens at fill 14259, bracketed: CTL-a 33.20, SHFL 34.42, CTL-b 32.89 (control spread 0.9%). Prefill flat 403.7–403.9. SASS on the fired gateup loop: 315 → 257 instructions (−18%), LDS.U.32 68 → 6, SHFL.IDX 0 → 64, LSU transactions 80 → 18 (−77%), LDG/FFMA counts byte-identical. Mechanism: that loop is **LSU-pipe bound at ~83% of structural ceiling while DRAM sits at 50%**, so moving the codebook gathers onto the idle shuffle pipe is free throughput. |
| `PXA_PXQ3_PAIRLUT=1` | **+3.7% cold / +3.8% median.** fill 14259, n=3, bracketed CTL/LUT/CTL: 31.43 / 32.72 / 31.70 against a 0.86% control spread, `ENGAGE` asserted per arm. Registers 74 → 48 (3 → 5 blocks/SM). |
| `-DPXQ_CANON_V2=1` (build constant) | **+3.5% clean decode, +5.7%/+6.0% with the n-gram drafter** (30.31 → 31.36; 35.86 → 37.93 cold, 38.85 → 41.20 warm). A precision *improvement*, not a trade: the chained form is 2 FFMA (2 roundings, both products exact) versus FMUL+FFMA+FADD (3 roundings). The win **amplifies** under speculation because a drafted step carries `1+n_draft` rows through the same codec kernel. ⚠ Changes the fp32 fold — re-baselining event. ⚠ Token output is **not** a valid instrument for this class of change: greedy decoding only diverges when a rounding difference flips an argmax, so identical temp-0 output is expected and is a quality result, not a null arm. |

`PXA_PXQ6_QPF` also ships (1-deep slab prefetch, bit-exact by construction, default off) but is
**UNMEASURED** — no A/B is on record. Do not quote a number for it.

## Self-speculation

`PXA_AUTO_SPEC` arms the measured-best drafter per model family. One row ships
(`qwen35moe` → `ngram-mod:n_max=4,n_min=2`): **first-request decode +23.0% on code traffic, +4.6% on
prose, prefill neutral** (262.63 → 259.16). Cell as above, ~14–15k fill, control in the same run.

⚠ **Do not quote the median-of-3 figures (+38.8%/+35.7%).** The bench replays one prompt three
times, so reps 2–3 hit an already-warm n-gram table — prose acceptance 1.000 is the tell. Warm
numbers are real for multi-turn chat and agentic re-reads, but they are an upper bound, not the
default case. The gain tracks how repetitive the OUTPUT is: high on code and tool traffic, much
lower on prose.

**No `mtp` row ships.** On the same model MTP measured **−8.6%** at `n_max=1` and **−29.8%** at
`n_max=2` *despite* 0.800 acceptance — a drafter costing a transformer forward per cycle cannot pay
on bandwidth-bound sparse-MoE decode, where a k-token verify batch routes to up to k×8 distinct
experts and reads ~k× the expert bytes. `ngram-mod` drafts only on an n-gram match, so it costs
~nothing when it cannot predict. Do not generalise "speculation helps" from this. An explicit
`--spec-type` or `-md` always wins; only absence is filled.

## Correctness and robustness (always on)

- `ggml_validate_row_data` accepts the PXA slab types, so `--check-tensors` works on PXQ files.
- Out-of-range `PXA_ROUTER_FUSE` values are now **reported and the lever explicitly forced OFF**
  instead of silently folded to 0. A build predating mode 3 resolved `=3` to `0`, which made an A/B
  run two identical arms and report a result.
- Per-mode fired/declined engagement counters (`PXA_ROUTER_FUSE_DBG=1`); **the ratio is the
  assertion**, since a one-shot bool cannot distinguish "fired once" from "fires every token".
- `-sm graph` on the DeltaNet hybrid arches falls back to `-sm layer` with a warning: it produces
  fully degenerate output there. Root cause is the cross-device all-reduce not reaching consumers,
  not the recurrent state. `PXA_ALLOW_GRAPH_SPLIT_HYBRID=1` bypasses the guard, for debugging only.

## Documented dead ends (kept, default off, so nobody re-runs them)

`PXA_PXQ_REDUCE_BLK` (+0.22% against controls differing by 0.35% — below the noise floor),
`PXQ_GU_MINBLK` (+0.14% median), `PXA_PXQ6_ROWX2` (null in both regimes), `PXA_SMALLN_GEMV`
(**−28%** vs emulated-dp4a MMVQ — the sm_60 dense GEMV is instruction-economy bound, not bandwidth
bound), and the `__launch_bounds__` min-blocks forcing experiment (forced registers 10–45 below
natural, spilled, −27.7%).

## Lever inventory

Every `PXA_*` environment variable the engine reads was audited against the source in both
directions — 138 in total, none documented-but-absent and none present-but-undocumented. That
includes the ones that measured null, the one that is unmeasured, and seven flags the engine had
been reading with no write-up at all — among them `PXA_ALLOW_GRAPH_SPLIT_HYBRID`, which bypasses
a correctness guard and is now labelled as the footgun it is.

Two build-time constants (`PXQ6_CANON_CMAX`, `PXQ_GU_MINBLK`) are documented as constants rather
than env flags, because flipping them changes the canonical fp32 fold and is a re-baselining event.

⚠ `PXQ6_CANON_CMAX` default stays **16** even though **4** is worth +10.4% on the cell above: the
evidence is sm_60 only, and on sm_70 the split target is 16×nsm so the selected S differs. If you
run Pascal, build with `-DPXQ6_CANON_CMAX=4` and re-baseline.
