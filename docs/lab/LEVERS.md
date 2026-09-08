> ## STOP — users need exactly two envs
> **nothing** (the tune, config level ENHANCE, is the default since 2026-09-03; `PXA_ENHANCE=0` rolls it back, `PXA_REFERENCE=1` is the bit-exact baseline) and optionally **PXA_MODE=balance|max** (posture). Every other env below is an
> internal experiment record — including measured LOSSES kept so nobody rebuilds them blind. Setting lab knobs
> manually bypasses per-arch gates and typically degrades performance (e.g. RAGTAIL is a measured loss on sm_61).
> This file is the lab notebook, not a settings menu.

# PXA levers — every shipping `PXA_*` environment variable

The definitive reference for what each knob does, its default, its **measured** effect (with the
config the number came from), and its correctness gate class. Two gate classes appear below:

- **bit-exact** — flag ON produces byte-identical logits/output to flag OFF (proven by temp-0
  output sha identity and/or memcmp kernel gates; see `bench/determinism-gates.md`).
- **G3-class** — deterministic but *not* bit-identical to the reference path (different but
  equally valid arithmetic order/precision); gated by temp-0 coherence + semantic equivalence +
  top-k logit spot-checks before shipping.

Bench protocol for every decode number unless stated otherwise: `llama-server`, model fully
GPU-resident, 200-token temp-0 generations, median of ≥3, speed read from the server's
`timings.predicted_per_second` (`bench/speed-bench.sh`); paired interleaved A/B with a ≤1%
baseline-spread guard. "U16-q8out" = `fusion2-35b-U16-q8head.gguf` (PXQU-16 + q8_0 output head),
the artifact behind the published P100/V100 decode rows.

## ⚠ 0a. SUPERSEDED NUMBERS — read before trusting any row below (2026-07-28)

Three engine fixes in this release changed **what the incumbent code path is**, so rows measured
against the previous incumbent no longer describe the shipping binary. Corrections, in order of how
badly they would mislead you:

| row | what it says below | what it measures NOW | why |
|---|---|---|---|
| `PXA_PXQ_GEMM_2D=2` (sm_70) | +2.30% V100 prefill | **−18.6% on dense sm_70**, **+0.1% (neutral) on MoE sm_70** | It was measured against the *pre-coalescing* `k_pxq6_dequant_matrix`. That kernel wrote one thread per row — 32 sectors per 64 useful bytes — so beating it was easy. Now that the dequant is coalesced and cuBLAS keeps its HMMA GEMM, the half2 tile loses on Volta. **Ship arch-split: ON for sm_60 (+35% P100 dense prefill), hard-veto on cc>=700.** |
| `PXA_PXQ4_2D_SPLIT_TARGET` | tuned default, K8-2D split | **+7.6% decode on a 122B A5B MoE, no-op on a 35B MoE**. ~~changes greedy output at temp 0~~ **cured 2026-07-28 by PXQ_CANON_v1: S no longer affects output bits** | Workload-dependent. Arch-tuned defaults shipped 2026-07-28: 16×nsm on cc 7.0, 8×nsm on cc 6.0, 2×nsm elsewhere (from the sm70-stack sweeps). |
| dense prefill baselines (61.7 V100 / 77.4 P100) | quoted in several rows | **dead numbers** | The K≤11264 smem cliff made every wide-K tensor fall to dequant+cuBLAS per token; that fallback path costs 18×. Any plan written against those baselines is planning against a bug. |

**The general lesson, which applies to every row here:** a lever's measured value is relative to the
incumbent it replaces. When a fix improves the incumbent, every lever that was beating the old
incumbent must be re-measured. **Three of ours flipped sign or went to zero.**

### The cell matters more than the lever

The same binary change measured **+56.2% decode on a 122B A5B MoE** and **+12.3% on a 35B MoE**,
because the fraction recovered depends on how much stall was present to begin with. And the codec
comparison changes sign with the cell — by workload *and* by arch:

| cell (2×V100) | PXQ4 vs MXFP4 |
|---|---|
| **MoE** (Fusion4-35B, 17.9711 GiB vs MXFP4 17.7978 GiB — 0.97% larger) | ⚠ **decode figure WITHDRAWN** in this release (it read "+7.7%"), and the **+18.9% prefill carries the same caveat** — see "Withdrawn: MoE decode" in `README.md`. The artifact labelled PXQ4 is only 120 of its 443 quantized tensors PXQ4 (full census `F32:308 MXFP4:300 PXQ4:120 Q8_0:23 F16:2`; attention and the shared expert stayed MXFP4), so both rows compared *MXFP4-with-PXQ4-experts* against MXFP4 — not codec against codec. For a matched-bytes MoE codec contrast that was measured as such, use the `PXA_PXQ4_2D_SPLIT` row in §4: **2×P100 decode +15.1%** (rank-separated), **2×V100 decode −2.0%** (rank separation incomplete) |
| **dense** (Qwable-27B, genuine byte parity 4.2526 vs 4.2500 bpw) | −4.57% decode — PXQ loses |

The dense-decode ceiling is structural, not a tuning failure: MXFP4 dense decode runs at **76% of
HBM peak with DRAM traffic equal to weight bytes**, i.e. bandwidth-bound on weights. At equal bytes
the ceiling is a tie by construction, so a dense-decode ratio win requires **fewer bytes at
equal-or-better error**, not a faster kernel. MoE is not bound the same way, which is where the
kernel work has room — and even there the result is arch-split (2×P100 decode **+15.1%**, 2×V100
**−2.0%**; §4). **Quote the cell, not an average.**

---

## 0. Three config tiers — `PXA_REFERENCE` / default / `PXA_ENHANCE` (master switches, 2026-07-21)

One knob picks the whole posture; **every per-lever env var below still overrides its own
default** — the tier only moves what a lever defaults to when its own env is unset.

| tier | how to select | what you get |
|---|---|---|
| **REFERENCE** (level 0) | `PXA_REFERENCE=1` — **wins if both are set** | every PXA lever defaults OFF/0: the whole `PXA_PXQ6_GATE` family off, `PXA_FUSE_DELTANET=0` (eager path), `PXA_VOLTA_CUBLAS_NE11=0` (MMQ-always), no G3-class levers. The pure reference kernel/dispatch paths — the bit-exact audit and A/B baseline. |
| **DEFAULT** (level 1) | `PXA_ENHANCE=0` — **the rollback tier since 2026-09-03**, it used to be what you got with no env at all | the pre-2026-09-03 shipped defaults: §2's measured bit-exact winners ON + `PXA_VOLTA_CUBLAS_NE11=64` (sm_70). Behavior unchanged vs the 2026-07-21 ship. |
| **ENHANCE** (level 2) | **the default since 2026-09-03** (or `PXA_ENHANCE=1`, now a no-op) | default **+** the per-arch **measured** G3-class levers whose ship gates passed: `PXA_PXQ_INT8_PREFILL=1` on **sm_61 ONLY** (the cc==610 ship gate is unchanged; **+182% prefill** measured on the 1080 Ti, §4) and `PXA_SPEC_RELAXED=1` (spec lanes only, G3). **2026-07-29: plus the model-adaptive auto-set — see §0c** (`PXA_ROUTER_FUSE` gated pure-sm_70 × MoE, `PXA_PXQ_MMVQ` auto for PXQ4-bearing models on DP4A devices, `PXA_MTP_LAZY_WARMUP` default ON, server auto-samplers). |

Every claim is per-arch measured (the per-lever rows below carry the numbers + configs). The
CUDA backend prints one startup line with the level and the per-device decisions, e.g.
`PXA level=ENHANCE | dev0 Tesla V100-PCIE-16GB(sm_70): CUBLAS64 ON [+9.4% pf] | dev1 Tesla
P100-PCIE-16GB(sm_60): defaults [bit-exact set] | dev2 GeForce GTX 1080 Ti(sm_61):
INT8_PREFILL ON [+182% pf, G3] | spec: SPEC_RELAXED ON [G3, spec lanes]`.
Implementation: `ggml/src/ggml-cuda/pxa/pxa-enhance.cuh` (`common/sampling.cpp` mirrors the level
logic for `PXA_SPEC_RELAXED`, a non-CUDA TU).

## 0b. The two postures — `PXA_MODE=balance|max` + ADAPTIVE-UB (the ship UX, 2026-07-22)

The kernel levers below are the MEANS; these two named postures are the PRODUCT. One knob:

| posture | how to select | fa / ub it fills | goal |
|---|---|---|---|
| **BALANCE** (default, the daily) | no env, or `PXA_MODE=balance` | `-fa on`, adaptive ub (2048-class on 16 GB cards) | best **decode** AND best-possible prefill **in the fa-on regime** — carried by `PXA_FA_MASK_SKIP_TILE` (bit-identical fully-masked-tile skip in the Pascal **tile-f16** FA kernel — sm_60/`GGML_PREC_DEFAULT` only; tile-f32 paths, including ALL of sm_61, are not covered — see its §4 row). ⚠ corrected 2026-07-30: `PXA_FA_PREFILL_SPLIT` is **no longer a BALANCE carrier** — it has been experimental OPT-IN (default 0) since 2026-07-24 because its non-FA prefill chain inflates the compute buffer ~2.35× and OOMs 16 GB cards at ub2048; opt in with `PXA_FA_PREFILL_SPLIT=64` when VRAM headroom allows (see its §4 row). fa-on **decode is byte-untouched by construction** — a prefill lever that costs decode is a MAX-only lever, never BALANCE. |
| **MAX** (bulk ingest) | `PXA_MODE=max` | `-fa off`, largest-fitting ub {2048→1024→768→512} | absolute max prefill; decode secondary. SPLIT/MASK_SKIP are inert at fa-off; all fa-off prefill levers (fp16-GEMM sm_60, CUBLAS64 sm_70, INT8_PREFILL sm_61 under ENHANCE) engage. |

Rules of engagement:
- **Explicit CLI always wins**: `-fa`/`-ub` (or `LLAMA_ARG_FLASH_ATTN`/`LLAMA_ARG_UBATCH`) on the
  command line are never overridden — the mode only fills UNSET flags (`llama-server` only; the
  filling + adaptive-ub live in `examples/server/server.cpp`, the kernel-lever selection in
  `pxa_mode()` in `ggml/src/ggml-cuda/pxa/pxa-enhance.cuh`).
- `PXA_REFERENCE=1` still overrides everything to the pure reference path (posture stands down).
- Both postures imply the full measured ENHANCE-class lever set for the active tier; they differ
  only in fa + which prefill carriers engage + the adaptive-ub target.
- **ADAPTIVE-UB**: at server startup the free/total VRAM of every assigned CUDA device is probed
  and the largest ub in {2048,1024,768,512} that plausibly fits next to the model's per-device
  share is chosen (≈0.5 MiB/ub-token optimistic heuristic, capped at the card-type default);
  safe fallback = card-type default (**≥15 GiB card → 2048; 11 GB 1080Ti class → 768; else 512**
  — the 11 GB fallback is hardware-verified: ub2048/1024 compute buffers OOM next to a ~10 GB
  model, ub768 fits). The server logs the chosen `mode/fa/ub` + the reason at startup
  (`PXA posture: mode=… fa=… ub=… (…)`), and the CUDA startup line reports
  `| mode=balance [fa-on serving]` / `| mode=max [fa-off ingest]`.

**Measured per-card posture table** (fair-battle rev2, n_prompt=5432 cold, temp-0 median of 3;
PXQU-16+q8head on P100/V100, PXQ2 on the 1080 Ti — 2026-07-22 windows):

| card | BALANCE (fa-on) | MAX (fa-off, largest-fitting ub) |
|---|---|---|
| **V100 16 GB** (sm_70) | prefill **1627.7** / decode **91.0–92.8** @ ub512 (+6.0% prefill from CUBLAS64; fa-on ub2048 blows up at request time until SPLIT is silicon-verified — adaptive-ub or explicit `-ub 512` is the working fa-on ceiling on a near-full single card). **Decode headline with MTP: 108.3** @ ub1024 fa-on (mtp n1 + lazy, steady-state; base 91.6 same server run — see the MTP section; ub2048 + the MTP gguf OOMs single-card, ub1024 is the ceiling) | prefill **2358.2** / decode 77.3–78.3 @ ub2048 (**canonical close 2026-07-22**: WMMA v2 `=3` on the merged canonical build, +5.0% flag-attributable; clean confirm set 2383.5. Canonical flag-off base is now **2245.3** — the old 2149.6 row moved +4.5% from canonical churn (CUBLAS64-era), drift resolved. CUBLAS64 +9.6% banked inside; threshold 64 proven the true optimum: 48 ties, 32 loses, 96 forfeits the +5% [64,96) window) |
| **P100 16 GB** (sm_60) | **BALANCE (fa-on ub2048): prefill 1206 / decode 56.7** -- measured WITH `PXA_FA_PREFILL_SPLIT=64` explicitly set (+45% fa-on prefill, 834->1206, decode held; median of 3). ⚠ SPLIT is opt-in since 2026-07-24, so out-of-the-box BALANCE prefill is the 834-class number unless you set it | **MAX (fa-off ub2048): prefill 1170 / decode 41.5** (fp16-GEMM on, banked) |
| **1080 Ti 11 GB** (sm_61) | prefill 678 / decode **65.6** @ ub768 adaptive (ub2048/1024 physically OOM; SPLIT is the staged carrier toward the ~950–1001 fa-off class) | prefill **985** / decode 64.7 @ ub768 (reproduces published 1001 within 1.6%; ENHANCE INT8_PREFILL +830% is the carrier; I8-DBUF/BN128 maturation REFUTED — see §5) |

⚠ **HONESTY GATE (2026-07-22; amended 2026-07-30)**: `PXA_FA_MASK_SKIP_TILE` is compiled clean
and equivalence-argued by construction, but its staged silicon A/B (B1/B2/B3 sha-set +
decode-guard cells) has **not yet run** — it was defaulted ON per the posture directive; roll
back instantly with `PXA_FA_MASK_SKIP_TILE=0`. `PXA_FA_PREFILL_SPLIT` was demoted to opt-in
(default 0) on 2026-07-24 — see its §4 row — so it is no longer a shipped default awaiting
silicon; its A/B remains un-run and is owed when someone opts in.
Determinism note (new stack fact): temp-0 output has run-to-run sha flutter even on unmodified
binaries in cuBLAS-engaged configs — determinism gates must compare **sha sets / short-gen
exactness**, not single-run sha equality.

## 0c. Model-adaptive auto-set — (device × model) decisions + the decision ledger (2026-07-29)

The ENHANCE level is now adaptive to the **loaded model**, not just the device set. At model load the
loader registers a **model profile** with the backend (arch name, model class, expert count, MTP
head, vocab size, and a census of MMVQ-eligible PXQ4/PXQ4HQ tensors); every ENHANCE auto-set that
depends on the model re-resolves against it. **Since 2026-07-31 this layer is no longer
ENHANCE-only:** `PXA_PXQ_MMVQ` and `PXA_FUSE_DELTANET` auto-arm at **DEFAULT** as well, so a stock
run gets them without `PXA_ENHANCE=1` (that widening is the whole point — the MMVQ auto-set
previously self-armed on paper and never fired for anyone running the binary as shipped).
**REFERENCE** still turns every auto-set off, and **every per-lever env var still overrides its own
auto-decision at any level**. The startup decision ledger reports the real state at DEFAULT too.

Model classes: `dense` (no experts), `moe`, `moe-hybrid-recurrent` (Gated-DeltaNet class:
`qwen35moe`, `qwen3next`), `moe-hybrid-swa` (full/SWA interleave: `gpt-oss`, `laguna`).

**The decision ledger.** Silent failure is this engine's recurring defect class (a documented lever
absent from the binary, a silent `--custom-q` demote, 12 silently-ignored type flags). So every auto
decision is LOUD: once both the device topology and the model profile are known, the backend prints
one `PXA model=…` line plus one `PXA_AUTO: <LEVER>=<value> (<reason>; override <ENV>)` line per
lever — including levers that CANNOT engage, which say why (`OFF: dense model — no router GEMV
exists`, `INERT: no sm_60 device`). A lever that fails to engage never no-ops quietly.

What ENHANCE now auto-sets from the (device × model) pair, on top of the §0 table:

| decision | keyed on | auto value | basis |
|---|---|---|---|
| `PXA_ROUTER_FUSE` | pure-sm_70 topology **× MoE model** | 1 on pure-sm_70 × MoE; 0 on dense (no router GEMV exists), 0 on mixed (T4 measured −3.1%), 0 without sm_70 | +5.1..7.0% dec, single V100 (§4 row) |
| `PXA_PXQ_MMVQ` | model PXQ4/PXQ4HQ census **× DP4A capability** | 1 when the model carries PXQ4/PXQ4HQ tensors and an sm_70+ device is present; 2 on an all-sm_61 fleet; 0 on a pure-sm_60 fleet (no DP4A — the emulation loses) or when the census is 0 | +13.7% dense / +6.7% MoE-attn-PXQ4 decode (§4 row); fidelity-neutral paired |
| `PXA_MTP_LAZY_WARMUP` | MTP lever, self-gating on MTP-active runs | ON at ENHANCE (explicit `=0` rolls back; REFERENCE off) | temp-0 **bit-identical**; without it MTP costs −33% prefill (§6 MTP section) |
| server sampler defaults | gguf `general.architecture` (server-side, `PXA_AUTO_SAMPLERS`) | see §6 row | top_k=0 measured to HALVE decode on a ~201k vocab |
| server `-ts` fill (`PXA_AUTO_TS`) | exactly-2-device mixed sm_70+sm_60, `-ts` unset | `1.4,0.6` V100-heavy | +9.78% decode (T4 cell: PXQ4-35B split V100+P100, `-sm layer`, 588-tok prompt, 2026-07-23); other topologies deliberately untouched — the basis does not generalize |
| `PXA_FUSE_DELTANET` | ledger relevance only | unchanged (default 3) — ledger states `INERT on this arch` for non-DeltaNet models | +3.5% P100 decode where active |
| `PXA_PXQ_GEMM_2D` | sm_60 present **× dense × PXQ-bearing model** | auto mode 1 (sm_60-only kernel) | +35% P100 dense prefill (coalesced-binary cell, 2026-07-28); MoE deliberately stays OFF — the post-coalescing sm_60 MoE cell is unmeasured and the sm_70 flip (−18.6%) proves the coalescing fix can invert a win |

Implementation: `ggml_pxa_set_model_profile` (`ggml.h`/`ggml.c`, registered from `src/llama.cpp`
before any graph is built), consumed by `pxa-enhance.cuh` resolvers cached against the profile
generation (the profile can land after CUDA init in the server flow — one-shot statics would have
frozen the pre-model decision). The ledger prints from whichever of {CUDA init, model registration}
happens second.

## 1. Format enables (required to run PXQ files)

| var | default | what it does | notes |
|---|---|---|---|
| `PXA_PXQ6` | **on** | fused CUDA kernel family for the 4-bit tier (PXQ4, formerly PXQ6) and PXQ4-HQ; `=0` drops to the dequant→cuBLAS fallback | fallback is correct but ~2× slower decode; a failed table self-check auto-falls-back |
| `PXA_PXQ2` | **on** | the 2-bit (LM4) kernel family; `=0` disables | |
| `PXA_PXQ3` | **on** | the 3-bit (LM8 bit-plane) kernel family; `=0` disables | |
| `PXA_PXQ6R` | **on** | the 5-bit PXQ6 (LM32 x E16-row) kernel family; `=0` drops to dequant→cuBLAS | the quality tier (env name keeps the internal working name) |
| `PXA_PXQ1` (PXQ1, type id 248) | **on** | fused CUDA kernel family for the sub-2-bit tier: 1-bit sign codes × the shared E16-row scales (per-row fp16 anchor + frozen SUB16 4-bit subs), 2-level book {−1,+1}, type_size 5 (1 scale byte + 4 code bytes / 32 elems), ~1.26 bpw. `=0` drops to the dequant→cuBLAS fallback — the tier's original v1 serving path: correct, slower. Built for `--pxq-universal` mixed maps: the 24/32 GB stretch tiers put low-importance experts at 1-bit (`pxq1` lines in the tier map, e.g. a knapsack mix like 126×pxq1/18×pxq2 for a ≤24 GB 122B-A5B) | ⚠ **Corrected 2026-07-30: this row used to describe the tier as dequant-only and ungated. That ceased to be true on 2026-07-29**, when PXQ1 joined the shared decode/GEMM pickers as format P1 (`pxq6_pol_p1`, TAB / TAB_CS modes only — the nibble-addressed PRMT and PAIRLUT modes are forced off for a 1-bit code). The env is a real default-ON gate, read once. A one-shot **sign-book self-check** runs at first use (the PXQ1 book is `{−1,+1}`, so it needs its own `|v| ≤ 1` check rather than the `|v| < 1` one the 4-bit Lloyd-fit books use); on failure the fused kernels disable themselves and the fallback serves instead. Both outcomes print one line to stderr (`PXA_PXQ1 fused kernels: ON/OFF`). **Measured (122B-A10B PXQU24): decode 11.81 → 36.0 t/s.** Cell: 6 Tesla GPUs , ,  temp 0, n_predict 256, fill 6360 asserted identical across arms,  asserted, MMVQ=1, n=3 interleaved rounds, fresh container per arm, median. The  rollback arm (same binary, same rounds, one env var) is the causal proof. quantize: `llama-quantize … PXQ1` (uniform) or `pxq1` rules in a `--pxq-universal` map; fixed compiled-in book, no provenance KVs |
| `PXA_PXQ1_REP_GUARD` | auto (arms at ENHANCE) | **PXQ1 repetition guard** (2026-07-23): the 1-bit tier measurably loops on open-ended prompts (coding answers stay clean — measured on a 122B PXQU24 mixed file). When the loaded gguf contains ANY PXQ1 tensor AND the config level is **ENHANCE**, sampler init fills repetition defenses as DEFAULTS: `repeat_penalty` 1.0→1.15, `repeat_last_n` 64→256, DRY `dry_multiplier` 0→0.8 (base/allowed-length untouched). **User-set values always win — only knobs still at their defaults are filled.** `=0` forces off, `=1` forces on at any level. Visible as `PXQ1 content detected: repetition guard ON […]` at sampler init (the loader also logs detection at model load) | REFERENCE/DEFAULT levels leave sampling untouched; models with no PXQ1 tensors never trigger it. ⚠ **2026-07-30 — code/doc drift found, measured, and fixed:** on 2026-07-24 the code silently broadened the auto-arm to ANY PXQ tier (this row still said PXQ1-only). Measured harm on a high-quality tier (Laguna-S-2.1 PXQ4-core, 4×P100+1080Ti `-ts 8,8,8,8,3`, ctx 8192, `/v1/chat` temp 0 top_k 1, fresh server per arm, ×3 reps): DEFAULT answers 4183×391 = **1,635,553 (correct ×3)**; ENHANCE answered **1,635,593 (wrong ×3)** — the filled repeat/DRY penalties suppress the legitimate repeated digit. Both-direction conviction: ENHANCE+`PXA_REP_GUARD=0` → correct ×3; DEFAULT+`PXA_REP_GUARD=1` → wrong ×3. This flip was chased for a night as an "sm_61/1080Ti numerics bug" (INT8_PREFILL, MMVQ, graphs-off, and the 7-card topology were each A/B-exonerated — all null arms). The auto-arm is now PXQ1-content-only again (matching this row); the broadening's original motive (the Fusion4 prose attractor, H1) was separately root-caused to `PXA_PXQ4_2D_SPLIT` non-bit-exactness and fixed by `PXQ_CANON_v1`. `=1` still forces any-PXQ arming; fixed-binary verification: ENHANCE→correct ×3 + guard not applied, forced→wrong ×3 + `[FORCED]` log. ⚠ COROLLARY for past numbers: every gauntlet/eval served with `PXA_ENHANCE=1` on a PXQ file between 2026-07-24 and 2026-07-30 ran with repeat_penalty 1.15 / last_n 256 / DRY 0.8 silently filled for requests that did not set them |

All master gates are ON out of the box (they were mis-documented as "off" before 2026-07-21) —
zero-env users get the fused kernels on every PXQ / PXQ-UNIVERSAL file.

> **sm_86 / sm_89 (3090 / 4090 class, added 2026-07-23):** the canonical arch list is now
> `60;61;70;86;89` — binary-wide (every kernel, every quant: full 30xx/40xx support, not a
> PXQ-tier subset). The per-arch cc-gated levers (`PXA_ROUTER_FUSE` cc==7.0-only,
> `PXA_PXQ_INT8_PREFILL` cc==6.1-only, `PXA_PXQ6_WMMA` cc==7.0-only) fall through to their safe
> defaults on sm_86/89 — the build is correct on Ampere/Ada, just untuned (no arch-specific
> fast paths measured there yet).

> **deepseek2 / MLA posture auto-wire, cc-aware (2026-07-23, refined):** on a
> `deepseek2`-arch gguf (GLM-4.7-Flash / DeepSeek class, MLA attention) `llama-server`'s
> posture layer defaults `-mla 3` on every arch, and defaults `-fa` **per visible CUDA
> fleet compute capability**: **ON** everywhere EXCEPT when every visible device is cc 610
> (sm_61 / P40, GTX 10-series), where it defaults **OFF**. Reason: fa-off on MLA degrades
> catastrophically with context on most archs (community-measured P40: 37 → 3.3 t/s by
> 36k ctx), BUT a second community P40 measurement found the MLA fa-on kernel itself is
> 75-326% SLOWER decode on sm_61 (fp16-starved, 1:64 rate) than fa-off there — so sm_61
> flips the default. `-fa on` still defaults ON under `PXA_MODE=max` for deepseek2 on
> sm_60/70+ (logged: `PXA posture: mode=max but arch=deepseek2 — fa kept ON (MLA requires
> it)`); on an all-sm_61 fleet that max-mode exception does NOT fire. Explicit `-fa`/`-mla`
> always win on every arch; the fa-off warning is suppressed on an all-sm_61 fleet only.
> See docs/KNOWN-ISSUES.md.

> **RETIRED 2026-07-21:** type ids **250** (`PXQ4-LEGACY`, the lossless MXFP4-repack slab type;
> `PXA_PXQ4` gate) and **251** (`PXQ5`, the learned-book + SE8 legacy type; `PXA_PXQ5` /
> `PXA_PXQ5_FAST` gates) were removed from the fork entirely. Loading an old id-250/251 gguf now
> fails with a clean error at gguf load ("type id N — this type was retired 2026-07-21; requantize
> from your source model with llama-quantize PXQ4 or PXQ6") instead of running. ⚠ Any on-disk
> artifact quantized as PXQ5 (e.g. `*-PXQ5.gguf`) needs a pre-2026-07-21 binary or a requant.
> Same day, the CLI/display name **PXQ6 was re-pointed to the real 5-bit LM32 tier (gguf type id
> 256, ftype 257, ~5.27 bpw)** — the ladder is now strictly PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6
> (+ PXQ_UNIVERSAL); "PXQ6HQ" survives only as a deprecated `llama-quantize` alias for PXQ4-HQ.

## 2. Recommended-ON performance levers (all bit-exact) — **default ON since 2026-07-21**

These used to require an env; they are now the compiled-in defaults (`<var>=0` reverts any one of
them to the proven reference path — each is individually bit-exact, so reverting is purely a perf
rollback). The published per-card numbers are what a zero-env user now gets.

| var | default | what it does | measured | verdict |
|---|---|---|---|---|
| `PXA_PXQ6_KSPLIT` | **on** | K1: splits the gate/up decode GEMV over K-segments with a fixed-order workspace reducer (more blocks in flight on small-R launches) | part of the published kernel set (all published decode numbers use it) | **ON** |
| `PXA_PXQ6_VECX` | **on** | K2b: float4 activation loads in the decode mmv inner loop | part of the published kernel set | **ON** |
| `PXA_PXQ6_GUFUSE` | **on** | K3a: fuses the up+gate GEMV pair + GLU epilogue into one kernel | part of the published kernel set | **ON** |
| `PXA_PXQ6_SCATFUSE` | **on** | K3b: fuses the MoE scatter/accumulate epilogue | part of the published kernel set | **ON** |
| `PXA_PXQ6_RAGTAIL` | **on** | K4: skips store-masked FMA work on ragged tail tiles | part of the published kernel set | **ON** |
| `PXA_FUSE_DELTANET` | **3** | `=3` fuses the DeltaNet (linear-attention) decode glue-kernel chain | **+3.7% decode P100 U16** (57.2→60.2→62.4 with q8 head, published config) | **ON (=3)** |
| `PXA_G2_ADDFUSE` | **on** | G2-F4: re-enables ADD+FUSED_RMS_NORM pair fusion (ne0≥256) + fuses the residual add into the MUL_MULTI_ADD epilogue (experts first, residual last — bitwise-commutative-identical) | **+1.9% decode V100** (100.1→102.0, U16-q8out quiet-window) / **+1.2% P100** (62.25→63.0, same protocol) | **ON** |

The published per-card numbers in `bench/README.md` were measured with section-2 rows 1–6 ON
(`ADDFUSE` landed after; its gain stacks on top — see the cookbook).

## 3. Available but NOT in the recommended env (measured no-gain on the published configs)

| var | default | what it does | measured | verdict |
|---|---|---|---|---|
| `PXA_PXQ6_PAIRLUT` | off | K2a: 256-entry float2 byte-pair LUT for code expansion (bit-exact) | **+0.1% P100 U16-q8out** (62.3 vs 62.3) — an earlier +4.8% reading came from a 2×P100 `-ts 1,1` 4-bit-flagship config and does not transfer | OFF (harmless; config-specific) |
| `PXA_PXQ3_PAIRLUT` | off | K2b3: P3 (PXQ3 bit-plane tier) 64-entry float2 pair-LUT — replaces the per-pair bit-plane code reassembly (~10 int ops + 2 smem gathers) with BFE+BFE+BFI+LDS.64 keyed on the pair's 4 low-plane + 2 high-plane bits (bit-exact sourcing; distinct from K2a, which only fuses the two gathers on 4-bit tiers) | **+3.7% cold / +3.8% median, bit-exact.** Cell: 122B-A10B PXQU48-core, 4×P100, ub2048, fill 14259, n=3, bracketed CTL/LUT/CTL — CTL-a 31.43, LUT 32.72, CTL-b 31.70, against a 0.86% control spread; `ENGAGE` asserted 1/0/0 per arm. All three temp-0 hashes identical (sourcing-only change, so byte-identical output is the designed outcome and the hash is a valid gate here). Registers 74 → 48 (3 → 5 blocks/SM), STACK 0, LOCAL 0. Distinct from the byte-keyed `PXA_PXQ6_PAIRLUT` (+0.1%), which targets 4-bit tiers where key extraction is already one BFE | measured A/B, bracketed |
| `PXA_PXQ6_SHFL` | off | K2d: register-book __shfl decode (decode mmv family, every tier) — lane l holds book[l] in one register; each per-pair book gather becomes SHFL.IDX, moving the codebook traffic (~63% of decode LSU bytes on the fired gateup loop, which is LSU-pipe bound at ~83% of the structural ceiling while DRAM sits at 50%) onto the idle shuffle pipe at identical instruction count (bit-exact sourcing: same float from the same index, same FMA order; no fp16 round-trip, unlike K2c PRMT) | **+4.1%, bit-exact.** Cell: 122B-A10B PXQU48-core, 4×P100, no drafter, sustained 2048 tokens at fill 14259, bracketed — CTL-a 33.20, ROWX2 33.41, SHFL 34.42, CTL-b 32.89 (control spread 0.9%), +4.1% against a control mean of 33.05. Every arm reproduces the standing temp-0 hash. Prefill flat at 403.7–403.9 across all arms (decode-mmv family only; the dequant fallback and GEMM prefill call `POL::pair()` directly and are untouched). SASS on the fired `<p2,p2>` gateup loop: 315 → 257 instructions (−18%), LDS.U.32 68 → 6, SHFL.IDX 0 → 64, LSU transactions 80 → 18 (−77%), LSU bytes ~13.4 KB → ~5.4 KB (−59%), LDG/LDS.128/FFMA counts byte-identical | measured A/B, bracketed + SASS |
| `PXA_PXQ6_QPF` | off | Q1: one-slab-lookahead register prefetch twins of the fired decode kernels (1 = mmv family, 2 = gateup family, 3 = both) — the canonical chunk walk issues slab i+1's code words + scale bytes before slab i's unpack/FMA chain, overlapping the exposed per-warp DRAM latency (the residual after K2d: every issue-side lever pays ~0.2:1 and the code LDG was consumed by the next instruction with zero lookahead) with compute that was being issued anyway; walk order, fold boundaries, chains, ws slots identical — pure load reordering of read-only data, bit-exact; yields to an engaged R2 twin per site | ⚠ **UNMEASURED — no A/B on record.** Committed so the work is not lost; the A/B was still queued when the campaign changed hands. Default OFF, so the shipped path is unaffected either way. Expectation was +3–6% with a coverage caveat: it covers the PXQ6 type (13.75 of 32.4 GB of expert weight on that file) while the gate/up path is mostly PXQ3, so expect roughly proportional capture rather than the full figure. Bit-exact by construction — pure reordering of read-only loads. **Do not quote a number for this lever** | none |
| `PXA_PXQ6_ROWX2` | off | R2: two-rows-per-thread 128-thread twins of the decode mmv family (k_pxq6_mmv / mmv_ksplit_gen / gateup_mmv_ksplit_gen) — one warp-uniform x load (LDS.U.128, 29% of the per-iteration LSU bytes on the fired gateup loop, 12.8x LSU amplification overall) feeds TWO rows' chains, halving x LSU writeback per weight byte; grid, smem layout, red[]/ws slots and every per-row fp32 chain unchanged (bit-exact, temp-0 sha gate); tier-independent (p2/p3/p6/p6r all ride it) | **standalone NULL: +1.1% vs 0.9% control spread (122B PXQU48-core, 4x P100, 2048-tok sustained, all 6 sites ENGAGED, sha = baseline) — the x broadcast is only 29% of LSU bytes until K2d clears the codebook, after which it is ~69%; the stacked SHFL+ROWX2 arm is the live question (first stacked build VOIDed on the bv-default miswire, fixed 2026-08-04 — re-A/B pending)** | pending stacked A/B |
| `PXA_PXQ6_PIPE` | off | K5: sm_60 2-stage register prefetch in the decode mmv (bit-exact) | no measured gain on published configs | OFF |
| `PXA_PXQ6_KSPLIT_GEN` | **arch default: S=4 on sm_60/sm_70, off elsewhere** (2026-07-26; explicit env always wins, incl. `=0`) | K1b variant: generic S-chunk K-split (S in 2/4/8) on the **gate/up fused** decode path (`k_pxq6_gateup_mmv_ksplit_gen`). **Bit-exact since 2026-07-28 (PXQ_CANON_v1):** every decode-mmv variant (unsplit, K1 re-grid, S-split, FUSERED finisher, REDFUSE) now computes ONE fixed, shape-only summation order (NFIX fixed K-chunks, lane-major fold), so toggling this lever — or changing S — does not change output bits. (Historical: before that date the S-chunk fold differed from the reference kernel and this row was G3-class.) Why it wins where K1 does not: the K1 64-thread form spreads the SAME warp count over 4× the blocks, while this 256-thread S-chunk form multiplies warps-in-flight by S and divides the per-block x-stage by S | measured 2026-07-26, F35B pure-PXQ4 fill ~6k on the shipped-lever binary: 2×V100 decode 86.0 → 87.9/88.3 (**+2.6%**, S=8 worse at 86.5), 2×P100 59.0 → 60.8 (**+2.9%**). Numerics gated with the K8-2D split in one arm: 64-chunk paired NLL (35B pure-PXQ4, `ppl-eval-half.txt`), 0 NaN, aggregate −0.1% (inside noise, favors ON), max per-position 0.28% — reassociation jitter, zero-centered. Tier transfer (ship levers vs all-off, fill ~6k, screening grade): PXQ2 2×P100 n=2/arm 58.16–58.40 → 59.86–60.51 (+3.3% median); PXQU16 2×P100 n=2/arm 56.62–56.77 → 57.22–57.62 (+1.1%); PXQ2 single V100 (idx-pinned, arms interleaved A/B/A/B, sibling card under an ncu run — screening hygiene caveat) n=4/arm 86.98–88.80 → 93.17–93.44 = **+6.1% median, complete rank separation**. `PXA_PXQ6_LDCS=1` re-tested on the new geometry (2×P100, n=2): 60.13–60.71 vs 60.92 ship — flat/negative, stays OFF | S=4 ON (sm_60/70) |
| `PXA_MMVQ_MOE_NWARPS` | 1 | forces 2/4 warps in the routed MoE GEMV (mmvq) | "prove faster + shadow-clean before defaulting" — no win recorded | leave unset |

## 4. Opt-in levers with trade-offs (read before enabling)

| var | default | what it does | measured | gate class |
|---|---|---|---|---|
| `PXA_PXQ_INT8_PREFILL` | off | **V100 (sm_70) A/B 2026-07-21: −6.6% prefill vs the fp16 fused incumbent → KILLED for sm_70; gate stays sm_61.** `=1`: routes PXQ prefill GEMMs through an int8 dp4a MMQ-style tile on **sm_61 only** (codes→s8 via the snapped book, per-16/per-8 sub-scales folded into a per-tile fp32 rescale, activations q8-per-32). `=2` lifts the arch gate (TEST — sm_60 dp4a is emulated, never ship there) | **1080 Ti PXQ2 cold 5.8k-token prefill 251→709 t/s (+182%)**, 95% of the native-MMQ ceiling; decode byte-untouched (66.4/66.5) | **G3-class** (temp-0 64-tok continuation sha-identical in our gates; top-1 logits identical every spot-check; tail top-5 order can shift at p≈0.015). Flag OFF = byte-identical dispatch |
| `PXA_PXQ6_WMMA` | 0 (=3 is the KEEP arm) | V100 tensor-core prefill path (fp16 fragments; `=1` fp32-accum, `=2` fp16-accum twin, `=3` **v2: double-buffer + fused GLU + BN128**); auto-guarded to the 4-bit tier, cc 7.0 only | **CANONICAL CLOSE (V100 A/B 2026-07-22, v2 merged to the canonical tree, 5432-tok cold, ub2048 fa-off, median of 3): `=3` (v2) +5.0% flag-attributable prefill (2245.3→2358.2), decode within ±1% (77.9→77.3) — KEEP.** Clean confirm set corroborates at **2383.5** (+6.2%; matches the K6-worktree 2383.8 almost exactly) — 2358.2 is the conservative primary-set figure (pooled 6-clean-round median 2368.1). **Drift RESOLVED: the earlier "worktree base +4.8%" was NOT worktree-local — the canonical flag-off base itself is now 2245.3 vs the published 2149.6 (+4.5%, pre-existing canonical churn, e.g. the CUBLAS64-era changes), matching the worktree's 2253.7.** `=1` (v1) is FLAT (+0.2%, BUILD_K6) — the whole gain is the v2 rebuild; the old "+0.97%" v1 reading stands confirmed dead. ⚠ Bench hazard learned during the confirm: a sibling-card COLD MODEL LOAD contaminates prefill (one set collapsed to 2190–2261 wide-spread during an idx4 35B load; steady-state sibling decode at 76% util is harmless) — check for fresh containers on the other card before trusting a wide-spread set | G3-class (kernel documented not bit-exact; shas flutter per round even flag-off — CUBLAS64-on config, gate on coherence not sha); v1/v2-fp16acc keep OFF |
| `PXA_VOLTA_CUBLAS_NE11` | **64** | on sm_70, routes dense quantized GEMMs with `ne11 ≥ N` to fp16 cuBLAS (HMMA tensor cores) instead of DP4A MMQ; `=0` restores MMQ-always | **default-ON 2026-07-21: public PXQ2 single-V100 prefill median +9.4% (1949→2133), won all 3 interleaved rounds; decode untouched.** Earlier internal 35B +6.5% consistent. **Re-confirmed 2026-07-22 on PXQU-16+q8head: +9.6% MAX (1962→2149.6 fa-off ub2048) and +6.0% BALANCE fa-on ub512 prefill; the kernel-level break-even ladder (ub=ne11 sweep) proves 64 is the TRUE optimum — cuBLAS wins ≥64 (+5% at 64), ties at 48, loses at 32; 96 forfeits the [64,96) window** | G3-class (prefill numerics class changes; also run-to-run nondeterministic at temp-0 — gate on sha SETS); `=0` rollback |
| `PXA_VOLTA_CUBLAS_ID_NE11` | 0 | same idea for the routed (mul_mat_id) expert GEMMs on sm_70 | measured a LOSS on the configs tried — MXFP4 experts already ride fast MMQ | leave unset |
| `PXA_P100_FP16_GEMM` | on | sm_60 dense-GEMM prefill path: fp16 dequant + GemmEx-16F (GP100 has full-rate fp16) | `=0` rolls back to fp32 SGEMM (the old, slower path). Its gain is already banked in the published P100 1213/1169 fa-off numbers. **0a hygiene 2026-07-22: now level-aware — `PXA_REFERENCE=1` really turns it OFF on sm_60** (it used to stay silently ON and contaminate reference floors); explicit env still wins | G3-class; ON is the shipped default |
| `PXA_FA_MASK_SKIP_TILE` | **on** | skips fully-`-inf`-masked 64-wide KV tiles in the **tile-f16** FA kernel ONLY (port of the shipped wmma `PXA_FA_MASK_SKIP_v1`; the nb31 mask-stride lesson applied). A BALANCE carrier — but it engages ONLY where tile-f16 actually dispatches: **sm_60 (P100)** with `GGML_PREC_DEFAULT` FA precision, prefill-shape batches (Q rows > 8), under `-fa on`. NOT covered by THIS lever (these ride the tile-f32 kernel, whose equivalent skip is the separate opt-in `PXA_FA_MASK_SKIP_TILE_F32`, default off — see its row below): (a) **sm_61 — `fast_fp16_available()` excludes cc 610, so the 1080 Ti NEVER runs tile-f16 and this lever is inert there**; (b) F32-FA-precision archs (the gpt-oss/`OPENAI_MOE` class) on any pre-Volta card. Volta (sm_70) gets the equivalent skip from the wmma kernel (`PXA_FA_MASK_SKIP_v1`). Inert at fa-off. ⚠ scope corrected 2026-07-30 — this row previously also claimed sm_61 engagement, which the dispatch code contradicts | bit-identical BY CONSTRUCTION (skipped tiles contribute exactly zero); **⚠ staged B1 silicon A/B (sha-set + decode-guard, target P100 fa-on ub2048 pf ≥900) has NOT yet run** — defaulted ON per the 2026-07-22 posture directive; `=0` rolls back | bit-exact (by construction; silicon gate pending) |
| `PXA_FA_MASK_SKIP_TILE_F32` | off | skips fully-`-inf`-masked 32-wide KV tiles in the **tile-f32** FA kernel — the third instance of the shipped skip pattern (wmma `PXA_FA_MASK_SKIP_v1` → tile-f16 `PXA_FA_MASK_SKIP_TILE` → here). Engages ONLY where tile-f32 actually dispatches: **sm_61 (1080 Ti) prefill-shape batches** (Q rows > 8; `fast_fp16_available()` is false at cc 610 on any arch, so the 1080 Ti never runs tile-f16) and **forced-F32-FA-precision archs (the gpt-oss/`OPENAI_MOE` class)** at Q rows > 8 on sm_60 and via the sm_70 sink route. Decode (Q rows ≤ 8) rides the vec kernels — untouched by construction. Cost on non-skippable tiles: the mask scan reads ~12.5% of the tile's K/V bytes at D=128 (25% at D=64) — **2× the tile-f16 lever's relative cost** (the f32 KV tile is half as wide, 32 vs 64) — plus two extra block barriers per tile and +32 B smem/block (28800→28832 B at D=128/ncols=32; keeps the `__launch_bounds__` occupancy-2 fit under sm_60's 64 KB/SM) | **NOT yet measured on silicon — ships default OFF until its own A/B (sha-set + prefill + decode-guard) lands.** The surviving gain regime is np1 causal strictly-future tiles (every query block scans the full padded KV extent) plus np>1 co-resident-slot scans; given the 2× relative scan cost a measured net-negative is a live possibility. Bit-identical BY CONSTRUCTION when it engages: skipped tiles contribute exactly zero (kqmax initialized finite at −FLT_MAX/2 → `KQ_max_scale == expf(0) == 1.0f`, `expf(−inf) == +0.0f`, `VKQ += V·0` and `*= 1.0f` are IEEE-exact) — the f16 levers' term-by-term argument, transferred to fp32 arithmetic. `=1` opts in; `=0`/unset → off (off-path is byte-identical dispatch via a compile-time template split); `PXA_ENHANCE_DBG=1` logs the resolved state | bit-exact (by construction; silicon A/B not yet run — opt-in) |
| `PXA_FA_PREFILL_SPLIT` | **0 — EXPERIMENTAL OPT-IN ONLY** (⚠ row corrected 2026-07-30: it previously claimed "64 under BALANCE" while both resolvers have returned 0 since 2026-07-24/merged 07-28 — doc/code drift caught by the A6 audit) | per-ubatch FA regime dispatch (`src/llama-build-context.cpp` + resolver): graphs with `n_tokens ≥ N` build the non-FA batched-cuBLAS attention chain even under `-fa on` — prefill rides the fa-off math (the pre-Turing fast-prefill regime), decode/MTP-verify (< N) keep the byte-untouched FA branch. Values 1–8 clamp to 9 (decode/MTP-verify safety). **Why no longer a default:** the non-FA prefill chain inflates the compute buffer ~2.35× (1956 → 4607 MiB measured) and OOMs 16 GB cards at ub2048. The +45% P100 fa-on prefill (834→1206) is real but must be bought explicitly (`PXA_FA_PREFILL_SPLIT=64`) with the VRAM headroom to pay for it | decode byte-identical by construction; the staged B2/B3 silicon A/B has still not run | prefill G3-class (regime swap), decode bit-exact |
| `PXA_MXFP4_DEQ_V2` | on | fast coalesced smem-table MXFP4→f16 dequant kernel | 150→397 GB/s dequant, bit-identical output; `=0` rolls back | bit-exact |
| `PXA_PXQ6_PRMT` | off | K2c: prmt/byte-perm **register-LUT** book decode (4-bit tiers) — 16-entry book in 8 uniform registers, `__byte_perm` nibble→fp16, zero smem. Bit-exact (memcmp all-pass, all 4 tiers) | **−11% decode on V100** (re-screened −17% n=2 in the 2026-07-26 KSG4 ship config — same verdict, dead). ⚠ **CAUSE CORRECTED 2026-07-27: the original reading "Tesla decode is bandwidth-bound so it loses" is DISPROVEN by the ncu roofline** (`ncu-pxq4-mmv.log`, V100 decode instances of the mmv family): `dram__throughput` **5.96–35.62% of peak** — nowhere near a memory ceiling — with achieved occupancy **19.8–38.6%, Block Limit = Registers (6 blocks/SM)**. The kernel is **register-limited and latency-bound**; PRMT loses because its extra uniform-register book + integer ops worsen exactly the register/issue pressure that is the real constraint. This misreading aimed four sm_70 attacks at a bottleneck that was not there (`CUDA_GRAPH_V2` −2..−4%, `INT8_PREFILL` −6.6% prefill, PRMT −11%, `FUSERED` −1.6%). KEEP default-OFF: it is the correctness-proven **sm_80 Marlin-tier prerequisite**, not a Pascal/Volta lever | bit-exact |
| `PXA_PXQ6_LDCS` | off | K7: `ld.global.cs` (evict-first) on the decode weight code stream | **+0.5% V100 decode = noise** (below the <1% kill line); bit-exact + harmless, left OFF. May pay on tighter-L2 cards (P100 A/B pending) | bit-exact |
| `PXA_SPEC_RELAXED` (+`_PMIN`, default 0.05) | **level default: ON at ENHANCE (the default level), OFF at DEFAULT/REFERENCE**; `=0`/`=1` always wins | relaxed speculative acceptance: accept a draft token that lands in the target's post-filter candidate set with p ≥ PMIN (instead of exact-match only). Auto-disabled for grammar/mirostat/temp≤0 **2026-09-06 correction:** this row (and the two below, and the code comment at `common/sampling.cpp:1200`) said "off" while `pxa_spec_relaxed_enabled()` has returned `ggml_pxa_config_level() >= 2` since ENHANCE became the default; only §0 line 69 was right. At temp 0 the exact-match path runs regardless (greedy gates unaffected); at temp>0 on defaults the output legitimately differs from the drafter-off run. The server now prints `PXA_SPEC_RELAXED: ON (pmin=…)` when a spec stage is armed and this is live; gate arms pin `PXA_SPEC_RELAXED=0` | never A/B'd — window item; G3-class by design (output legitimately changes at temp>0) | G3-class |
| `PXA_PXQ4_2D` (+ `PXA_PXQ4_2D_MAX_NY`, default 8) | **on** (was `off` until BACKBONE_REV 2, 2026-07-26 — every rev-2 file carries PXQ 2D weights, so the old default would have put attention/shexp/dense-FFN on a per-token dequant→cuBLAS expansion; `=0` restores the old default for A/B) | Routes a plain 2D (non-MoE) MUL_MAT on a PXQ slab weight — attention projections, shared-expert FFN, `ssm_out`, `token_embd`, `nextn.eh_proj` — into the PXQ decode mmv (the E==1 case of the expert-stacked kernel, fed a one-entry zero-`ids` buffer) instead of dequant→cuBLAS. Only relevant to files quantized with `PXA_PXQ_NATIVE` beyond the routed experts (see §8). `MAX_NY` caps the token count claimed; above it the per-call dequant amortizes over the batch and the driver declines (do NOT raise it into prefill range — at ny=2048 the mmv would re-decode the weight once per token) | shared-experts-native file: decode **63.6 → 88.2 t/s** at the shexp conversion (recovers the −39.7% per-token dequant fallback). attn-native file, 2×V100 `-ts 1.05,0.95` fill 5739 n=8 interleaved: decode 85.7 t/s, prefill 1759.5 | bit-exact (same kernel instantiations, same fp32 chains as the MoE mmv); **ON for any PXQ-native-attn/shexp file** |
| `PXA_PXQ4_2D_KSPLIT` (+ `PXA_PXQ4_2D_KSPLIT_MINBLK`, default = device SM count) | off | K1-2D: re-grids the 2D decode mmv's four in-block k-segment chains (`kb ≡ kseg mod 4`) into four 64-thread blocks writing fp32 partials to the persistent KSPLIT workspace, plus a fixed-order reducer. Same total warps, 4× the blocks — the medicine for a launch whose `panels*ny` cannot cover the SMs. Fires only when `panels*ny < MINBLK`; `MINBLK=0` never splits, a huge value always splits (pure A/B knob, no rebuild). Inert unless `PXA_PXQ4_2D=1` | **FLAT on 2×V100, measured LOSS on 2×P100 — hence default OFF.** attn-native Fusion4-35B, fill 5739 / n_predict 256 / temp 0, one server instance per arm, medians of 3. **2×V100** (`-ts 1.05,0.95`, n_SM=80): off **85.34 / 85.06**, threshold-on (fires on the 40 `panels=64` + `panels=32` nodes) **84.88 / 84.90**, always-split (`MINBLK=100000`, all 82 nodes incl. `panels=128`) **85.14 / 85.24** — every arm inside the ±0.5% instance-to-instance drift measured between two *identical* code paths (pre-patch 85.09 vs lever-off 85.34). Prefill untouched (1744–1767 all arms; prefill declines at `ny > MAX_NY` long before reaching this code). **2×P100** (`-ts 1,1`, n_SM=56, medians of 2): off **57.86**, always-split **53.65 = −7.3%**. Verdict: the 2D decode mmv is **not** occupancy-limited at panels 32–128. Firing was proven, not assumed (the one-shot `PXA_PXQ4_2D_KSPLIT dev0: FIRING (panels=… )` stderr line); the 64-thread form's 4× redundant full-K smem staging (32 serial global loads per thread in the prologue vs 8 in the 256-thread form) plus one extra reducer launch per firing node cancels the extra-SM win. Un-tested shapes where the occupancy model still predicts a gain: shexp-native `panels=8`, and low-SM cards. **Independently re-measured (n=8/arm, 4 interleaved blocks, fresh server per arm per block, 2×V100): off 85.43 (IQR 0.59), threshold-on 84.58 (IQR 0.25) = −1.00% with complete rank separation (Mann-Whitney U 64/64, and separately within each block), always-split 85.09 = −0.41% (overlapping). At fill 50 — where the occupancy model predicts the *largest* win — off 93.07, threshold-on 92.30, always-split 93.37: no starvation win appears at any fill.** So at its shipped threshold this is a small but statistically clean regression, not merely flat | bit-exact: identical per-thread fp32 chains and identical ascending 4-term reduction as `k_pxq6_mmv` — temp-0 sha256 unchanged vs the pre-patch binary on 26 completions across 11 arms and both arches (sm_70 + sm_60), incl. the forced always-split arm. `=1` enables; unset is the exact pre-split single-launch behaviour. ⚠ **That sha gate is non-discriminating with the harness prompt** (5500 words of a repeating pangram): the identical hash `18a4c492…` is also produced by the *MXFP4-baseline model file*, i.e. by a different attention quantization entirely. It proves coherence and unchanged flag-off dispatch, not kernel equality. The bit-exactness claim rests on the construction argument (identical per-thread fp32 chains, identical fixed ascending 4-term reduction); no perplexity or logit-level gate was run on this lever |
| `PXA_PXQ4_2D_SPLIT` (+ `PXA_PXQ4_2D_SPLIT_TARGET`, default = 2× device SM count) | **on** (2026-07-26; `=0` restores the unsplit dispatch, which byte-reproduces pre-split output) | K8-2D: S-way K-chunk split for the 2D decode mmv (`k_pxq6_mmv_ksplit_gen` + `k_pxq_mmv_reduce_s`), the successor to `PXA_PXQ4_2D_KSPLIT` that attacks what that lever could not. nvprof gpu-trace on a rev-2-backbone file (Laguna-XS, 2×P100) localized the 2D decode tax to two launch shapes: R=2048/K=8192 (`attn_output`, dense `ffn_down`) = 32 blocks whose **33.8 KB full-K x-stage caps occupancy at 1 block/SM** → 127.2 µs/call = **87 GB/s**, and R=512/K=2048 (`ffn_{up,gate}_shexp`) = **8 blocks total** → 34.2 µs = **20 GB/s** — while the SAME dot path with a healthy grid (R=8192/K=2048 `attn_q`: 128 blocks, 9 KB) ran 53.0 µs = **210 GB/s and beat MXFP4's mmvq (161 GB/s) per byte on the same-shape tensor**. The split multiplies blocks by S (smallest power of two reaching `SPLIT_TARGET` blocks incl. the ny axis, capped at 16 and at ≥4 slabs per chunk) and divides the x-stage by S, fixing both starvation modes; S=1 shapes keep the bit-exact unsplit kernel unchanged. **PXQ_CANON_v1 (2026-07-28): the split is now BIT-EXACT vs unsplit.** Every variant writes raw per-(fixed-chunk, lane) partials (NFIX×KSEG slots per row, NFIX shape-only) and one canonical fold produces the value — identical bits at any S, on any device, with the lever on or off. This closed a real production defect: on the 122B with PXQ4 attention, the pre-canon split path turned a fabrication probe into a 962×-8-gram runaway that hit the 2048-token cap (SPLIT=0 answered cleanly in 343 tokens); the aggregate-NLL G3 gate that certified the lever was structurally blind to that failure mode (aggregate −0.1% while one near-tie flip enters a repetition attractor). ⚠ One-time re-baseline: the canonical order differs from BOTH pre-canon paths, so temp-0 output vs pre-2026-07-28 binaries changes once (PPL-verified neutral); from now on it is stable across split decisions. Unlike `PXA_PXQ4_2D_KSPLIT` (same warps spread thinner + 4× x-stage → measured flat/loss, stays OFF) this adds warps AND shrinks smem | PROTOCOL (fill 6401, n_predict 256, temp 0, **n=8/arm, 4 interleaved blocks, fresh server per arm per block**, 2×P100 `-ts 1,1`, all contrasts complete-rank-separated): rev-2 Laguna-XS decode unfixed **43.47** (IQR 0.15) → fixed **58.70** (0.34) = **+35.1%**, vs legacy-backbone 60.49 (0.24) — i.e. the rev-2 fidelity backbone's decode cost drops from **−28.1% to −3.0%**. Pure-PXQ4 F35B same protocol: 2×P100 decode MXFP4 53.10 (0.44) vs PXQ4-ship **60.92 (0.11) = +14.7%**, prefill 497 vs **783 = +57.5%**. 2×V100 same protocol (n=8/arm, ncu-stamped clean cards, complete rank separation): decode MXFP4 **89.16** (IQR 1.69) vs PXQ4-ship **85.17** (0.26) = **−4.5%** (from −23.7% pre-fix on the same config, screening n=3: 69.4), prefill 1654 vs 1618 = −2.2% (overlapping IQRs). **PAIRED RERUN on the shipping binary (2026-07-27, llama-server md5 `6940b347`, libggml `2996fc25` — the FUSERED-default-OFF build; both arms same run, same protocol): V100 decode MXFP4 89.12 (IQR 0.91) vs PXQ4 87.33 (0.46) = −2.0%, rank separation NOT complete (pxq max 88.02 > mxfp4 min 87.72); P100 52.56 (0.17) vs 60.52 (0.27) = +15.1%, rank-separated. The PXQ arm gained +2.5% between binaries with the MXFP4 control flat — the only source delta is the dormant FUSERED tail in the split kernels (codegen/register-allocation shift, not a designed change), so −2.0% is the current binary's honest figure and −4.5% is the prior binary's.** Short-fill smoke: Laguna 44.0 → 70.5. Numerics: 64-chunk paired NLL vs off-arm, 0 NaN, aggregate −0.1%, max per-position 0.28% (zero-centered reassociation jitter); temp-0 divergence begins at a near-tie token, off-arm byte-reproduces the pre-patch binary. **All of that is history as of 2026-07-28: split==unsplit bitwise (PXQ_CANON_v1), verified on Fusion4-35B-PXQ4N-attn across 5 dispatch variants (unsplit / default split / forced-max split / K1 re-grid / gateup-gen S=2) × 3 prompts × greedy-384 + fixed-seed temp-1.0 sampling — all hashes identical — and on the 122B runaway fabrication-probe regression (see the row text)** | bit-exact (PXQ_CANON_v1), ON |
| `PXA_PXQ_SPLIT_FUSERED` | **off — measured LOSS on both arches, kept as a documented dead end** (`=1` enables) | fence-and-flag fused finish for the S-split reducers: the last split block of each node performs the final chunk reduction in-kernel (threadFenceReduction pattern, same ascending order as the standalone reducers — output verified byte-identical), eliminating the `k_pxq_mmv_reduce_s` launch (230/token) and the gateup reduce launch (40/token). Motivated by an nvprof reading of GPU-busy parity between the codec arms with the deficit apparently in per-token launch count | protocol (F35B matched bytes, fill 5992, n=8/arm, 4 interleaved blocks, fresh server per arm per block, rank-separated): fused-ON decode **83.81 vs 85.17 = −1.6% (2×V100)**, **59.45 vs 60.92 = −2.4% (2×P100)**, MXFP4 control arms flat (+0.6%/−0.9%). Post-mortem: the finisher block reduces S·R floats ALONE while the rest of the machine drains — the wide reducer does the same work across R/256 parallel blocks in ~2.4 µs — and the launches it saves were already hidden by async submission. ⚠ Pattern note: this is the SECOND independent attack on decode launch overhead to come back negative on these cards (`PXA_CUDA_GRAPH_V2` was −2..−4% with captures verified firing). Decode launch count is NOT where the V100 deficit lives; do not attack it a third time without new evidence | byte-identical when enabled (verified); OFF |
| — (launch-bounds min-blocks experiment, NOT a lever — closed 2026-07-27) | reverted, not in tree | `__launch_bounds__(256, N)` min-blocks forcing on the three decode mmv kernels (`k_pxq6_mmv`, `k_pxq6_mmv_ksplit_gen`, `k_pxq6_gateup_mmv_ksplit_gen`), aimed at the ncu register finding (achieved occupancy 19.8–38.6%, Block Limit = Registers at 6 blocks/SM) | **decisive KILL at both tested points** (Laguna-XS rev-2, fill 6401, fresh server per arm per block): `(256,7)` n=8/arm interleaved — 2×V100 base 88.29 (IQR 0.64) vs 63.87 (1.04) = **−27.7%, rank-separated**; 2×P100 59.20 vs 58.52 = −1.2% (one P100 log tag reads `lb256x8` from a lazily-re-read script — the binary was the (256,7) snapshot, libggml `1d0efbd7`). `(256,8)` cut short at n=2 on the same cliff (63.6/63.9 vs 88.5/88.8, −28%). Verdict: the kernels' ~40+ live registers are load-bearing — capping to 36 or 32 regs/thread spills to local memory and destroys sm_70 decode. **Occupancy here cannot be bought with register caps**; the accidental +2.5% from the dormant FUSERED tail is therefore scheduling/ILP drift, not occupancy. **V100 codec lane CLOSED at −2.0%** (paired rerun, no rank separation — a median gap inside touching distributions; P100 +15.1% separates cleanly). Evidence to reopen on, if anyone does: `ncu-pxq4-mmv.log` (latency-bound, DRAM ≤36% of peak, register-limited) + this row | closed, reverted |
| — (Q-G2 wrel gate status, 2026-07-27) | — | `pxa-bench/pxqu_wrel.py` ships but is NOT runnable as shipped: it imports a numpy oracle (`pxqu_lab.py`) that is not part of this repository, and no equivalent is available. Q-G1 (`pxa-bench/pxqu_golden.py`, the stricter byte-parity gate) does build and runs PASS on all five input classes against the current-tree quantizers | Q-G1 is the running gate; Q-G2 is inert until its oracle is reconstructed |
| — (asymmetric FA KV cache types, `-ctk`/`-ctv` config, not an env — status row 2026-07-30) | — | mixing K and V cache quant types under `-fa on` is NOT generally runnable on the shipping build: it compiles with `GGML_CUDA_FA_ALL_QUANTS=OFF`, and a K/V pair without a compiled FA vec kernel does not fall back — it hard-aborts at request time (`on_no_fattn_vec_case`: "Unsupported KV type combination for head_size 128"). In particular **K=q8_0 / V=f16 at head-128 is NOT compiled** (that pair exists only in the `GGML_CUDA_FA_ALL_QUANTS` branch). The abort message itself lists the compiled head-128 pairs; the asymmetric ones available on this build are K=q8_0/V=q6_0 (7.5 BPV), K=q8_0/V=iq4_nl (6.5 BPV), and K=q6_0/V=q5_0 (6.0 BPV) | **NO measurement on record for ANY asymmetric K/V pair** — nothing here is benched. Recorded so "asymmetric KV as a free VRAM/quality trade" is not re-opened as a zero-cost idea: check the pair compiles, then bench in your own cell before trusting it | n/a (config status; unbenched) |
| `PXA_MTP_PREFETCH` | off (merged 2026-07-27 from the mtp-overlap lane) | async MTP companion commit: a dedicated `ctx_mtp` worker thread + `_async` variants of the accepted-hidden-rows commit; the wrapper submits the EXACT serial commit and returns immediately, overlapping the companion decode with `process_token` + the streaming write. Per-seq futures with drain calls at every other `ctx_mtp` touch point keep ordering; `=0` (default) is byte-for-byte the serial path | **measured NULL where measured (2026-07-27): F35B PXQ4-MTP, `--spec-type mtp:n_max=1`, fill ~6k, n=8/arm, 4 interleaved blocks, fresh server per arm per block, 2×V100 `-sm layer`: pf0 median 113.39 (111.33–114.72) vs pf1 113.31 (109.47–114.68) = −0.07%, fully overlapping, accept rate identical 0.96154 both arms.** ⚠ Caveat, prominently: 2×V100 `-sm layer` is the regime where this lever class is KNOWN to disappear (`PXA_SPEC_1ROW`: +6.6% single-card → +0.7% noise once split). Null where measured, **untested where it could pay** — the single-card dense-MTP verdict (27B vehicle) is owed | OFF |
| `PXA_F16_GEMV` | **on** (2026-07-26; `=0` restores the cuBLAS chain) | small-R F16 decode GEMV (`k_pxa_gemv_f16`): an F16×F32 `ne11==1` node misses every fast dispatch path (dmmv/mmvq need a quantized src0; batched-cuBLAS needs a real batch dim) and lands on the generic cuBLAS chain — convert x f32→f16 + `gemmSN_TN_kernel_half` + convert dst + a cpy = measured **31.5 µs + 3 helper kernels per call** (nvprof, 2×P100) for what is a ~0.3 MB tensor read. The rev-2 backbone's per-head `attn_gate` → `f16` promotion creates exactly this node per layer (40/token on Laguna-XS ≈ **1.5 ms/token**, the bulk of the `lite` recipe's measured decode cost). The kernel does the row dot directly: 4 warps/row, f16 weights × f32 activations, fp32 per-thread partials, fixed warp/smem reduction tree. Restricted to 2 ≤ R ≤ 512, contiguous, `ne11==1`, so it can never intercept a real dense projection; `ne01==1` stays with `mul_mat_1row` | isolated contribution inside the rev-2 Laguna fix above (70.5 with, ~68.5 without, short-fill n=1 — the 2D split dominates; the chain it removes is 1.5 ms/token of GPU time plus its CPU launch pressure). NOTE the replaced cuBLAS chain rounds x AND the result through f16; this path keeps both f32, so it is numerically DIFFERENT (strictly tighter) — behavior-gauntlet class, not sha class | G3-class, ON |
| `PXA_PXQ_GEMM_2D` <br>**⚠ mode 2 SUPERSEDED on sm_70 — see §0a; the sm_70 numbers in this row were measured against the pre-coalescing dequant and are −18.6% against the current one. Mode 2 is CLAMPED to sm_60 in code.** | 0 (off) | E==1 **prefill** GEMM for plain 2D PXQ weights (attention projections, shared-expert FFN, `ssm_out`, `token_embd`, `nextn.eh_proj`) in the `ny > PXA_PXQ4_2D_MAX_NY` window - the exact complement of `PXA_PXQ4_2D`'s decode claim, so on a gated arch a PXQ 2D weight never reaches dequant->cuBLAS at any batch size. Reuses `k_pxq6_gemm_grouped` unchanged with a one-expert tile map (`tile.e == 0` => `pxq6_panel()` degenerates to `W + p*stride`, the same degeneration the mmv driver gets from its one-entry zero-`ids` buffer); the map is built by a device kernel rather than the MoE driver's pageable H2D, so it stays legal under stream capture. `=1` sm_60 only, `=2` sm_60 + sm_70 (any fast-fp16 pre-Turing arch). Declines `GGML_PREC_F32` nodes - the cuBLAS fallback honours that with an fp32 SGEMM and this tile accumulates in half2 (MUL_MAT_ID never sets it, which is why the MoE driver has no such check). Only relevant to files quantized with `PXA_PXQ_NATIVE` beyond the routed experts (see §8) | attn-native Fusion4-35B, fill 5739 / n_predict 256 / temp 0, **n=8 per arm, 4 interleaved blocks, fresh server per arm per block**, `-c 8192 -b 2048 -ub 2048`: **2xV100** `-ts 1.05,0.95` mode 2 - prefill **1718.3 -> 1757.9 t/s = +2.30%**, decode 85.08 -> 85.41 (flat). **4xP100** `-ts 1,1,1,1` mode 1 - prefill **799.6 -> 806.7 t/s = +0.89%**, decode 57.24 -> 57.22 (flat). Complete rank separation within *both* the cold-rep and warm-rep strata on *both* arches (firing proven per device by the one-shot `PXA_PXQ_GEMM_2D dev0: FIRING (...)` stderr line), so the direction is real - but both are **below the +3% pre-registered keep line, hence default OFF**. Decode is untouched by construction (`ny <= MAX_NY` is declined here and owned by `PXA_PXQ4_2D`). WARNING: the design spec predicted a *multi-x regression* on sm_70 and would have shipped this sm_60-only; that was **wrong**, and why is worth keeping: it weighed this half2 tile (~15-20 TF) against cuBLAS HMMA (~90 TF) *alone*, but the incumbent is `k_pxq6_dequant_matrix` (32 scalar 2-byte stores per thread at a row stride of `K`) **plus** cuBLAS. Corollary for the prefill deficit: claiming *both* halves of the incumbent buys only +2.3%, so coalescing that dequant while KEEPING cuBLAS's HMMA GEMM should dominate this on Volta. **Independently re-measured (n=8/arm, 4 interleaved blocks, fresh server per arm per block): 2×V100 prefill 1747.2 → 1753.5 pooled, per-block paired median +1.53% (range −2.65%…+2.30%, 6/8 blocks positive) — i.e. the +2.30% above is the top of the range, not its centre; with `PXA_PXQ4_2D_KSPLIT=1` also on, 1727.1 → 1751.7 = +1.48%, 8/8 blocks positive. 4×P100 `-ts 1,1,1,1` 800.1 → 807.8 = +0.93% paired, 8/8 positive (confirms the +0.89%). Decode flat on both (−0.19% / −0.01%).** | **G3-class - gate on perplexity, not sha.** Strict-k half2 chain vs dequant + `cublasGemmEx(COMPUTE_16F)`: different accumulation order and intermediate handling. Gated: 200-chunk paired perplexity (`-c 512 -b 512 -ub 512 --seed 1`, `ppl-eval-half.txt`, 2xP100, same binary/model/cards) - **off 5.7880 +/-0.06762 vs on 5.7842 +/-0.06753**, delta -0.0038 = 18x inside the error bar. **The sm_70 path was gated separately** (mode 2 is the arm that enables it, and it was not covered by the sm_60 run above): 2×V100 `-ts 1.05,0.95`, 200 chunks, two independent replicates per arm — off **5.7713 ±0.06737** (both runs), on **5.7692 ±0.06732** (both runs), delta −0.0021 = 32× inside the error bar, firing proven per device. Flag OFF is byte-identical dispatch. ⚠ The temp-0 sha256 is NOT a meaningful gate here — the same hash is produced by the MXFP4-baseline *model file* — so the quality claim rests on the two perplexity runs alone |
| `PXA_PXQ6_FORCE_PREFILL` | off | TEST ONLY: bypasses the sm_60/70 prefill arch gate so correctness A/Bs can run on other archs | correctness testing only | never in production |
| `PXA_PARALLEL_LOAD` | off | **load-time lever, inert for inference math**: multi-threaded weight loading for `--no-mmap` loads — port of upstream ik PR#2057 (shared work-iterator worker pool; per-worker 16 MB pinned staging feeding `ggml_backend_tensor_set_async`; per-thread `llama_file` clones) + the PR#2102 memory-bound rider (CUDA-named buffers without the staging backend — split buffers, `--check-tensors` runs — load via a per-file lazy mmap + `MADV_DONTNEED` of each consumed range instead of unbounded whole-tensor read buffers). `=1` → 8 workers (upstream default), `2..64` → explicit count, unset/`0` → the byte-identical legacy serial path. **Inert under mmap (the default load mode): upstream's own rewrite serializes every tensor behind one mutex when mmap is on, so the gate logs one WARN and keeps the proven serial path — engaging it means launching with `--no-mmap`.** mmap/mlock/unmap-fragment machinery never runs in the parallel path by construction | 27B MXFP4 15.7 GB, 2×V100 `-ts 1.05,0.95` `-c 4096` fa-on, NVMe source, llama-cli `load time`, n=2/arm interleaved (2026-07-30): **cold-cache (fadvise-evicted) `--no-mmap` 10.07 s → 7.52 s (−25%); vs the prod-style default-mmap cold load 13.88 s = −46%**. Warm-cache: no-mmap serial 6.00 s, parallel 6.25 s (+4% — upload-stream-bound; `=16` worse at 6.62 s, keep 8), default-mmap warm 7.06 s. Host-RAM loads parallelize hardest: 2B Q8 `-ngl 0` 2.55 s → 0.62 s (−76%); 54.7 GB F16 MTP-graft host load 87 s → 68 s (−22%, mixed-cache cell). [EST] for the live recreate flow (llama-server, multi-card) — same loader code but not yet run there, and it requires adding `--no-mmap` to the launch | bit-exact (identical bytes into identical buffers): temp-0 sha identity ON vs OFF on CPU-host, single-V100 cuda-staging, `--check-tensors` (the PR#2102 branch), 3-shard split-GGUF (2B Q8 — all arms one sha, old-binary null arm included), 2×V100 27B MXFP4 (20/20 arms one sha across mmap/serial/parallel/16-worker, cold+warm), PXQ4-native 35B, and 54.7 GB F16 MTP-graft host cells; OFF = the untouched pre-port code path |

### New in 2026-07-28 — both default OFF

| lever | default | what it does | measured | gate |
|---|---|---|---|---|
| `PXA_PXQ_MMVQ` | `0` (off); **DEFAULT *and* ENHANCE auto-set 1 (or 2 on an all-sm_61 fleet) when the loaded model carries PXQ4/PXQ4HQ tensors and a DP4A-capable device is present — §0c (2026-07-29), widened from ENHANCE-only to DEFAULT 2026-07-31; REFERENCE still opts out; explicit env always wins**. ⚠ Before 2026-07-31 the auto-set required level 2, so a stock run never armed it and the entire sm_70 backbone benefit was unreachable without `PXA_ENHANCE=1` — the lever self-armed on paper and never fired. Observed live on a 299B serve: the startup ledger printed `level=DEFAULT — model-adaptive auto-set engages only under PXA_ENHANCE=1` | Routes the **decode** GEMV for the PXQ4 / PXQ4HQ tiers to the stock q8_1 **MMVQ** kernel (`mul_mat_vec_q`) instead of the bespoke PXQ 2D driver, and lets the fused up+gate pair reach `ggml_cuda_op_fused_mul_mat_vec_q_id` — one kernel walking the activation once for both, which is the fusion MXFP4 already gets and the per-operand divert throws away. Only the MMVQ batch window (`ny <= MMVQ_MAX_BATCH_SIZE`) is handed over; prefill keeps the 2D drivers untouched. `=1` cc>=CC_VOLTA, `=2` cc>=610 (real DP4A only — sm_60 would run the DP4A *emulation* and lose). Declines if `PXA_PXQ6_BOOK` / `PXA_PXQ6_SUB` / `PXA_PXQ6_SUB_HQ` are set: this TU holds frozen copies of the book/SUB tables and a runtime override would silently diverge from the fused kernels. | **MEASURED 2026-07-28.** Qwable-27B dense PXQ4core, 2×V100, `-c 4096`, n=7, same gguf in every arm: bespoke `mmv` **29.787** → `MMVQ=1` **33.861** = **+13.7%** (MXFP4 on the same config: 36.401, so the gap closes from −18.2% to −7.0%). Prefill also rises 541.3 → 554.7, free, since the flag only claims the decode batch window. `MMVQ=2` is identical to `=1` (33.854), so the DP4A-only variant adds nothing over the Volta gate. **PXQ4core gains far more than PXQ4HQ** (33.861 vs 30.969). On **MoE** (Fusion4-35B) the flag is a no-op unless attention is moved off PXQ6 first — PXQ6 is not MMVQ-registered, so `attn=PXQ6 + MMVQ` is +0.4% while `attn=PXQ4 + MMVQ` is **+6.7%** (93.20 → 99.47). **QUALITY: neutral.** Paired perplexity **at `-b 8 -ub 8`** (see the warning below), 30 chunks, held-out half: dense 5.2113±0.15741 → 5.2149±0.15776 (Δ +0.0036, 44× inside the bar); MoE 5.9029±0.17947 → 5.8998±0.17937 (Δ −0.0031, 58× inside). Opposite signs, each ~2% of its own bar — noise, not a systematic cost. ⚠⚠ **DO NOT gate this lever with default-batch perplexity.** `llama-perplexity` is pure prefill at `-b 512`; the MMVQ dispatch gate is `ne11 <= MMVQ_MAX_BATCH_SIZE` (=8), so at default batch the kernel **never fires** and both arms return bit-identical PPL (5.1599±0.05895 on ours) — a false PASS from a measurement in which the feature was off. The `b=8` arms DIFFER, which is what proves the kernel engaged. This applies to any decode-window lever, not just this one. Background: ncu on the bespoke PXQ mmv shows it **register-limited at 19.8–38.6% occupancy** — precisely the failure mode MMVQ's design avoids — and PXQ was absent from all three MMVQ registration points while MXFP4 rides it on sm_70. Cost is ~1 `vec_dot` plus 3 switch arms. Enable and measure on your own cell before trusting it. | G3-class — different accumulation path; gate on perplexity, not sha |
| `PXA_PXQ_KV` | unset (= `q8_0`, table default). ⚠ This row described a lever that was NOT in any shipped source until 2026-07-28 (docs/code drift — the audit found 0 hits in six branches and an unconditional pin in `llama-quantize.cpp`); it is now IMPLEMENTED on the codec-fixes branch exactly as documented | Overrides the BACKBONE_REV 2 pin that holds `attn_k` / `attn_v` / `attn_v_b` at `q8_0`. Accepts `q8_0｜pxq4｜pxq4hq｜pxq6｜mxfp4`. **Quantizer-side, not runtime.** ⚠ Three gates had to agree before this could work, and each failed *silently*: (1) `--custom-q` registers a rule but `pxa_pxq_backbone_type()` never receives `params`, so the table wins regardless; (2) the rev-2 table pins K/V; (3) `pxq4_tensor_eligible()` excludes attn_k/v **by name** and demoted the tier straight back. All three now clear when the env names a `pxq*` tier. | On Qwable-27B, `attn_k` per layer **10.00 MiB → 2.66 MiB** (vs 5.00 MiB at q8_0); 17 layers each of k and v. Speed effect **NOT yet measured** — the first arm built against a pre-patch binary and came out **byte-identical to its control**, i.e. it tested nothing. | Changes the artifact, not the kernel. **Diff the tier table of the arm against its control before benching** — that is what caught the null arm |

**Method note that generalises.** An A/B arm whose tier table matches its control in every class did
not test your flag; it tested nothing, and it will read as "the lever is worth zero". Dump and diff
tier tables before spending a bench window. Likewise, put a **known-answer coherence gate before any
timing** — an empty completion scores perfectly on tokens/sec — and make that gate tolerant of a
`<think>` block, or reasoning models will report false failures.

---

## 5. Documented dead ends (kept for reproducibility — measured no-gain or loss, default OFF)

| var | what it was | measured outcome |
|---|---|---|
| `PXA_G2_REDFUSE` | G2-F1: absorb the gateup ksplit-reduce + GLU into the down-mmv prologue | **−0.8% decode V100** — the 8× workspace re-read + GLU recompute costs more than the reduce it removes. KILL |
| `PXA_G2_NORMFUSE` | G2-F3: fused rms-norm emits a q8_1 sidecar so the mmvq chain skips `quantize_q8_1` | no measurable gain over ADDFUSE alone (P100 63.0→62.95); bit-exact, kept OFF |
| `PXA_G2_QUANTFOLD` | G2-F2: the DeltaNet out-gate kernel emits the q8_1 sidecar for `linear_attn_out` | same — no gain over ADDFUSE on the measured configs; bit-exact, kept OFF |
| `PXA_PASCAL_DMMV` | alternate Pascal DMMV dispatch experiment | measured loss; documented dead end |
| `PXA_PXQ6_BOOKX` *(kernel body NOT in this tree — see row)* | K9 (2026-08-0x, sm_70 decode track): stage the 16-entry book and the sub-scale table as 32 lane-private copies (`tab[idx*32 + lane]`) so lane L always reads bank L and the book LDS is conflict-free by construction | **KILL, −7.2%.** Qwable-27B PXQ4, one V100, tg128: **34.15 → 31.70**. The extra index arithmetic (`idx<<5 + lane`) costs more than the conflict replays it removes — i.e. the decode dot is **not** limited by shared-memory bandwidth, which is the reusable finding. Bit-exact by construction |
| `PXA_PXQ6_MMVX2` *(kernel body NOT in this tree — see row)* | K10: two slabs per loop trip so both weight loads issue before either FMA chain (same operand order, bit-exact) | **KILL, −3.4%.** Same rig: **34.15 → 33.00**; registers **40 → 52**, which drops resident blocks **6 → 4**. Stacked with K9: **30.09**. The register cost of the second in-flight slab outweighs the load-issue overlap |

> **Why K9/K10 have no code here.** Both landed on `sm70-codec` as commit `0227577`, against the *pre-canon* `k_pxq6_mmv` and its `pxq6_pick_mmv_ksplit_gen_x` variant selector. `PXQ_CANON_v2` since rewrote that kernel and its dispatch (which now carries the `rowx2`/`qpf` variants), so the 187-line probe body no longer applies and porting it would mean writing new, unmeasured kernel code for two levers that are **already known to be slower**. The measurements above are the part worth keeping — that was the commit's own stated purpose. The kernel bodies remain retrievable at **`git show archive/sm70-codec-K9-K10`** (tag on `0227577`), which is tagged precisely so they survive branch pruning. ⚠ **`sm70-codec` reads as reachable from the unified line but its code is NOT applied** — it is an ancestor of the abandoned `pxa/unified` union, which was recorded with `-s ours`; `grep PXA_PXQ6_BOOKX ggml/src/ggml-cuda/pxa/pxq6.cuh` returns nothing and that is correct, not a regression. Also in that commit and likewise not ported: folding the PXQ4-HQ per-row sub-scale pair into one 16-bit load, and a compile-time `pxq6_srcpair()` operand-sourcing route.

| `PXA_CONV_SILU_FUSE` *(env string no longer present in the tree — grep-audits should expect 0 source hits; see row text)* | B2 (2026-07-22, P100 track): fold the DeltaNet conv-output SiLU into the SSM_CONV kernel epilogue, on the theory that the PXA_PROFILE-measured "conv_output_silu 14.1% of decode, 400us/call" bucket was a width pathology | **KILL — audit disproved the premise, fused kernel confirmed it.** The UNARY already runs n_tokens-wide (ne=[8192,2]); the 400us/call is a **profiler artifact** (PXA_PROFILE syncs before/after every node, so the fixed sync round-trip is billed equally to a 16 KB elementwise op and a real GEMM). Full fusion built anyway per protocol: temp-0 sha bit-identical, decode 56.5→56.1 (−0.7% ub512) / 56.4→55.6 (−1.4% ub2048) on P100 PXQU-16. Closed per the B2 kill line; the code stays only in a removed side worktree — the ggml_ssm_conv API change was not worth carrying for a dead lever |
| `PXA_ROUTER_FUSE` | B3 phase 1 (2026-07-22 synthesis): dedicated warp-per-row F32 GEMV kernel for the MoE router logits (`ffn_gate_inp`, F32×F32 ne11==1), which misses every fast dispatch path and lands on a bare `cublasSgemm` — the #1 PXA_PROFILE decode bucket (22.7% P100 / 25.5% V100) | **PER-ARCH verdict, closed 2026-07-22: KILL on sm_60, KEEP on sm_70.** P100 fair-battle A/B (PXQU-16+q8head): decode 57.3→58.2 (+1.6% ub512), 57.2→57.9 (+1.2% ub2048) — under the pre-registered <+3% kill line; the profiler bucket was queue-gap absorption (same lesson as B2). **The owed V100 A/B ran 2026-07-22 (single V100, ub512 fa-on): decode 92.1→96.8 (+5.1%), REPRODUCED on a full second run of both arms 90.3→96.6 (+7.0%); the 6 on-rounds sat 96.3–96.9, very tight. Above the 3% line → ON is worth it on sm_70** — the fuse verdict does NOT transfer across arches; gate it per-arch (sm_70 only), do not blanket-apply either verdict. G3-class (fuse reorders FP math; ON arm shows run-to-run sha flutter — ULP logit changes flap expert ties). Kernel + dispatch live at `ggml-cuda.cu` (search PXA_ROUTER_FUSE), zero cost when unset. **Tier wiring DONE 2026-07-22 (integration): ENHANCE auto-enables mode 1 = the cc==700-only ship gate (resolver `pxa_router_fuse_mode_resolve()` in `pxa-enhance.cuh`, INT8_PREFILL pattern); REFERENCE/DEFAULT stay OFF. Env always wins: `PXA_ROUTER_FUSE=0` forces OFF at any level, `=1` the sm_70 ship gate, `=2` TEST all-arch (⚠ semantics change: explicit `=1` is now arch-gated — a Pascal re-bench needs `=2`). Startup line prints the per-dev decision.** **Auto-wire SILICON-VERIFIED 2026-07-22 (canonical close): with `PXA_ENHANCE=1` and NO fuse env, startup printed `PXA level=ENHANCE | dev0 Tesla V100(sm_70): CUBLAS64 ON ROUTER_FUSE ON [+5-7% dec, sm_70]` and decode matched the headline (108.1 vs 108.3). Standalone re-confirm on the merged build: +4.6% decode (91.6→95.8, ub1024 fa-on). ⚠ Stacking fact: the fuse does NOT add on top of MTP — mtp+fuse 107.6 == mtp-alone 108.3 within noise (the fused-router win is absorbed into MTP accept-rate round-to-round variance, ±10 t/s). ENHANCE remains the recommended production switch: it matches the MTP headline while auto-enabling the fuse for the no-MTP path for free** |
| MTP `n_max>=2` on P100 (config, not env) | deeper MTP draft depths on sm_60 | **measured LOSS**: n_max=2 decode 54.9→47.4 (−14%), accept/drafted-token collapses to 0.42; n_max=3,p_min=0.5 gets back to 51.5 but still under OFF. Cause = the B4 verify tax (verify(3+)≥1.65× on P100). Use n_max=1 (see §6) |
| MTP `n_max=2` on V100 (config, not env) | deeper MTP draft depth on sm_70 | **same shape of LOSS as P100 (2026-07-22, ub1024 fa-on)**: decode 92.7 vs base 94.1 — below OFF; per-drafted-token accept collapses 0.960→0.480 (97/202 — the 2nd draft token is almost never accepted, accepted-count identical to n1's 97/101). n_max=3 skipped per protocol (n2 < base). The smaller sm_70 verify tax does not rescue depth ≥2; **n_max=1 is the sweet spot on BOTH arches** |
| B6 np2 on a SINGLE V100 (config, not env) | two server slots on one card to overlap requests (canonical build, c16384 np2 fa-on ub512) | **KILL (2026-07-22): 2 concurrent identical requests aggregate 77.5 t/s decode vs 93.3 single-stream on the same np2 server = −17% AGGREGATE** (rounds 79.0/77.5/77.3, aggregate = sum of both slots' predicted_per_second). Heavily asymmetric per-slot (~49.8 / ~27.5): one slot's decode overlaps the sibling's COLD prefill and both lose — the cold-prefill-interleaved regime is where single-card np2 dies. Contrast: the production 35B runs np2 across TWO V100s (different regime, resident-KV siblings + MASK_SKIP) — that layout is unaffected by this kill. Single V100 serving = np1; concurrency belongs at the proxy queue, not the slot count |
| `PXA_MOE_FASTTG_MAX_NY=1` + MTP (grouped-verify combo) | route Ny>1 MTP verify batches to the A1 expert-grouped path | **measured BIG LOSS on P100**: n2 decode 48.1→30.3. The grouped path loses badly at tiny Ny on sm_60; leave the default (8, fast-TG path) |
| B13 GPU_TOPK_SAMPLER | GPU radix top-k=100 select + 100-pair D2H to kill the CPU sampler wall | **KILLED by measurement before build** (spec-decode track, 2026-07-22): sampler = 0.43 ms/tok at top_k=100 on P100 (1.8% of the 23.4 ms decode wall) — 18× under the 1 ms kill line. The gpt-oss top_k=0 lesson does not transfer to top_k=100 |
| B1 ngram spec activation (wikitext/synthetic-agentic matrix, P100) | in-tree `--spec-type ngram-mod / ngram-map-k4v` sweeps | **no clean keeper**: every ngram-mod config either regressed cold-prose >1%, regressed agentic hugely (−36..−45% at low hit rates — failed drafts are pure verify-tax on P100), or broke sha. Near-miss: `ngram-map-k4v:n_max=64,ngram_size_n=8,ngram_size_m=8,ngram_min_hits=2` = bit-exact + +2.5% on the synthetic agentic transcript, −1.8% cold — retest against a REAL tool-call transcript before any verdict |
| `PXA_SPEC_SMALLN` | B4 v1 (2026-07-22 synthesis, from-scratch kernel `pxa-smalln.cu`): warp-per-row multi-column dequant-FMA GEMV for the dense MXFP4/q8_0 backbone at ne11 2..8 on cc<70 — one weight pass, R fp32 accumulators, meant to beat emulated-dp4a MMVQ at spec-verify shapes | **KILL as built, measured honestly**: P100 single-card MTP n1 decode 63.3→49.9 (−21%), n2 48.1→39.1. Root cause: v1 reads the code stream in scalar 1-byte loads (16 B/warp-iter) vs mmvq's vectorized 128-bit streams — the mmvq verify tax at k=2 is only 1.28×, leaving less headroom than the B4 audit implied at k=4. Zero-cost when unset (sanity cell: no-MTP decode 57.0, baseline sha, path never fires at ne11=1). Kept in-tree default-OFF; the un-attempted next step is uint4 code loads (one thread = one full 16 B block) + half2 y staging — only worth a window if a future card/model shows verify(2) ≫ 1.3× |
| B4 SPEC_SMALLN cuBLAS-redirect variant | route P100 dense ne11 2..8 to the banked fp16-cuBLAS path | **KILL** (spec-decode track, 2026-07-22): verify(4)/verify(1) got WORSE, 1.646→2.28× — per-call dequant+setup dominates at tiny N. Superseded by the custom `PXA_SPEC_SMALLN` multi-column kernel (§6) |
| `PXA_CUDA_GRAPH_V2` (+`PXA_CUDA_GRAPH_LOG`, `PXA_CUDA_GRAPHS_PASCAL`) | keyed whole-token CUDA-graph replay cache (byte-identical output). Env-only opt-in — never default | **KILLED on both box arches with captures VERIFIED firing:** V100 −2..−4%; **P100 −3.9%** (65.0→62.5, public PXQ2, replays=396/400 tokens, 3 interleaved rounds). Decode is GPU-busy; replay bookkeeping is pure tax. ⚠ Measurement lesson: an earlier '+3.5% P100' reading was NOISE — the cc<Ampere arch gate silently kept captures at 0 (a graph env that captures nothing is a no-op). Never believe a graph number without `PXA_CUDA_GRAPH_LOG` showing captures>0 |
| `PXA_CUDA_GRAPH_BATCH` / `PXA_CUDA_GRAPH_MOE` / `_LRU` / `_REARM` / `_BATCH_MAX_NY` / `PXA_CUDA_GRAPHS_PASCAL` | earlier opt-in CUDA-graph capture experiments for small multi-token batches | no shipped gain; diagnostic lineage of GRAPH_V2. Per-knob mechanism + defaults are documented individually in the 2026-07-30 lever inventory below |
| `PXA_PXQ_I8_RAGTAIL` (1080 Ti track, 2026-07-22) | B17-adjacent: port of the fp16 kernel's K4 RAGTAIL (bit-exact ragged-tile FMA skip, `PXA_PXQ6_RAGTAIL`) into the sm_61 int8 dp4a prefill tile (`pxq6i8.cuh`/`k_pxqi8_gemm_grouped`). Diagnostic instrumentation (`PXA_PXQI8_DEBUG=1`) confirmed the theory's premise: at ub768/PXQ2, MoE routing (n_as=256 experts, ~6144 routed token-instances/ubatch) leaves **~91% of tiles ragged, averaging ~64% wasted row-slots** per tile — exactly the regime RAGTAIL should help. **Measured anyway: −2.9% prefill** (963.0 t/s vs 991.7 t/s baseline, fair-battle rev2 protocol, 1080Ti/PXQ2/ub768/fa-off, median of 3, `PXA_PXQ_INT8_PREFILL=1` both arms). **KILL.** Root cause: this tile is not FMA-throughput-bound (matches the prior 2026-07-22 DBUF/BN128 audit's "223-228 regs, 0 spills, 4 blocks/SM — ILP-saturated, not latency-bound" finding) — the per-`kb`-iteration smem stage/`__syncthreads()` pair dominates the wall, so skipping the FMA consumption loop saves nothing and the extra per-thread branch (`if (fma_on)`) is pure tax. Also confirms **B17 stream-K was correctly declined without a build**: the same debug pass showed grid = panels(8) × tiles(~259–276) ≈ 2100+ blocks per up/gate launch vs the 28-SM×4-blocks/SM = 112-block concurrency ceiling — many wave-serialized launches already, not SM-starved, so adding blocks via a K-split has no occupancy upside to capture. Both findings reframe **B18 MOE_ALIGN** for this fork: the fixed per-tile launch/staging/sync overhead (not ragged-row compute) is the likely wall, so a sort-and-pad port in the vLLM style is unlikely to pay here either without first attacking tile *count*/*fixed-cost*, not row occupancy — **not built this window** (recommend profiling the per-kb sync/stage cost with `ncu` before any further MoE-tile investment here; the existing tile design already avoids launching empty-token tiles, which is MOE_ALIGN's other headline benefit). Code kept in `pxq6i8.cuh` (gated OFF by default, zero cost when unset) as a clean, reusable diagnostic (`PXA_PXQI8_DEBUG`) + a validated-negative kernel variant so nobody re-builds this blind. |

### MTP speculative decode — MEASURED KEEPER on P100 AND V100 (2026-07-22; V100 battery closed same day)

**Config: `--spec-type mtp:n_max=1,p_min=0.0` + `PXA_MTP_LAZY_WARMUP=1`.** Model = an MTP-grafted
fusion2 (nextn tail layer, qwen35moe `nextn_predict_layers=1`). Measured, fair-battle rev2 (5432-tok
cold wikitext, n_predict=200, temp0, median of 3):

| cell | prefill | decode | vs OFF |
|---|---|---|---|
| P100 single-card (U16-q8out+MTP graft, ub512 fa-on) OFF | 919.6 | 56.8 | — |
| P100 single-card, mtp n1 + lazy | 923.8 | **63.4** | **decode +11.6%, prefill flat** |
| 2×P100 (PXQ4-MTP 19GB, ub512 fa-on) OFF | 938.2 | 54.9 | — |
| 2×P100, mtp n1 + lazy | 924.7 | **59.1** | **decode +7.7%** |
| V100 single-card (U16-q8out+MTP graft, ub1024 fa-on) OFF | 2131.9 | 94.1 | — |
| V100 single-card, mtp n1 + lazy | 2047.8 | **107.5** | **decode +14.2%, prefill −3.9% (fa-on)** |
| V100 prefill-regime control (fa-off ub1024) OFF | 2009.1 | 76.1 | — |
| V100 prefill-regime control, mtp n1 + lazy | 1966.8 | 97.4 | prefill −2.1% = flat within round noise |

Draft acceptance 0.78–0.79 per drafted token (wikitext continuation). Coherence verified (clean
technical prose at temp-0). G3-class: run-to-run sha flutters between a small sha set on the spec
arms (batch-shape fp flutter); MTP-OFF on the grafted model is **bit-identical to the ungrafted
baseline** (same sha 384ec84d3aa7c001) — the graft is output-transparent when speculation is off.
- **Without `PXA_MTP_LAZY_WARMUP=1` MTP costs −33% prefill** (938→624) — the lazy env is mandatory.
- **OOM: single-card 16GB + MTP + ub2048 does NOT fit** (cuMemCreate OOM on first prefill).
  MTP single-card runs ub<=1024 (measured ub1024: prefill 1110 / decode 63.0, accept 0.87 -- the
  single-card MTP balance posture); the 2-card split takes ub2048 fine.
- **n_max=1 is the P100 sweet spot** — deeper drafts lose (see §5 kill rows): the B4 verify tax
  (verify(2)/verify(1)=1.28×, verify(4)=1.65×, measured 2026-07-22) eats the extra accepts.
- Model artifacts (built this window via gguf surgery):
  `fusion2-35b-U16-q8out-MTP.gguf` (14.6GB, single-card; base
  U16-q8out + blk.40 tail/nextn from the ornith MXFP4-MTP donor) and
  `fusion2-35b-PXQ4-MTP-fixed.gguf` (19.2GB, 2-card; the retired-id-250 blk.40 experts of
  `fusion2-35b-PXQ6-MTP-clean` swapped back to the donor's plain MXFP4 — byte-size-identical swap).
- **V100 CLOSED 2026-07-22 (same two arms, single V100, ub1024 fa-on): mtp n1 + lazy = decode
  94.1→107.5 (+14.2%), accept 0.960 (97/101) on the reading-comprehension workload — the expected
  larger sm_70 win, confirmed.** Caveats, recorded honestly: (1) ~~the n1 decode rounds trended
  96.5→107.5→117.6 (warmup/clock ramp) — re-median on a warmed card before engraving~~ →
  **RESOLVED by the steady-state battery below: the ramp settles into a stable ~105–113 band;
  the old 107.5 median was honest.** (2) fa-on serving pays −3.9% prefill for the +14.2% decode
  (only the fa-off control regime is flat at −2.1%). (3) The 0.960 accept is prompt-dependent
  (P100 wikitext measured 0.78–0.79; the steady-state battery read 0.79–0.80 cumulative) — do not
  quote it as the general rate. (4) The MTP-vs-base temp-0 sha-identity gate could NOT be applied
  on this V100 build — the BASELINE itself flutters shas per round (CUBLAS64-on sm_70 nondet), so
  the correctness gate was temp-0 coherence (verified by reading the output), weaker than the P100
  window's bit-identity proof. (5) ~~ub2048 untested on V100~~ → **tested: single V100 + the
  13.6GB MTP gguf + ub2048 = OOM at context creation** (verbatim: `allocating 2004.00 MiB on
  device 0: cudaMalloc failed: out of memory … failed to allocate compute buffers`) — same shape
  as the P100 precedent; ub1024 is the single-card ceiling. n_max=2 is a KILL sub-arm on V100 too
  (92.7, accept 0.480 — see §5).
- **⭐ STEADY-STATE BATTERY = the canonical-close V100 decode numbers (2026-07-22, merged canonical
  build, single V100, ub1024 fa-on, np1).** Protocol upgrade: 5 rounds in ONE server run
  (`fb-cell5.sh`), median of the LAST 3 — back-to-back `fb-cell.sh` invocations tear down/reload
  the model between rounds and re-cool clocks, breaking steady-state. Cells:

  | cell | decode | prefill | notes |
  |---|---|---|---|
  | base (no spec, no fuse) | 91.6 | 2031.7 | rounds 91.6–92.0, flat — no ramp on the base |
  | mtp n1 + lazy | **108.3** | 2107.4 | **+18.2%**; accept 0.802 cumulative; steady 105.6–109.4 after a 2-round ramp |
  | router-fuse only | 95.8 | 2125.3 | +4.6% standalone (startup: `ROUTER_FUSE ON [+5-7% dec, sm_70]`) |
  | mtp + fuse (env) | 107.6 | 2013.6 | == mtp-alone within noise — the fuse does NOT stack on MTP (accept variance ±10 t/s dominates; last-3 spread 96.9–117.6) |
  | mtp + `PXA_ENHANCE=1` (no fuse env) | 108.1 | 2031.4 | **auto-wire verified** — startup printed `level=ENHANCE … ROUTER_FUSE ON` + `SPEC_RELAXED ON`, matches the headline |

  **The publishable V100 decode headline is 108.3 t/s (mtp n1 + lazy), +18.2% over the 91.6
  same-run base — supersedes the 107.5 3-round figure.** Recommended production switch =
  the ENHANCE level on top of the MTP config: statistically identical to the headline (108.1) and
  it auto-enables the sm_70 fuse for free. All cells coherence-gated (clean English temp-0
  continuations; shas flutter on every cell incl. base, per the CUBLAS64 sm_70 protocol note).

### PXA_SPEC_1ROW + PXA_CUBLAS_EAGER_INIT — MEASURED, both default ON (2026-07-23 fair-battle)

**Origin:** an r/unsloth MTP-prefetch report, checked out line-for-line against our dispatch — the
ne01==1 F32 shared-expert-gate GEMV at MTP spec-verify batch sizes (Ny=2..8) missed every fast
path (dmmv/mmvq/mmq need a quantized src0, batched-cublas needs `src1->ne[2]*ne[3] > 1`, the old
`mul_mat_1row` needed `ggml_nrows(src1)==1`, ROUTER_FUSE needs a `ne01` in [2,4096]) and fell
through to a bare `cublasSgemm` every spec-verify decode step — plus a separately-reported
intermittent `CUBLAS_STATUS_INVALID_VALUE` from cuBLAS's lazy workspace allocation on a
near-full card (external sm_86 report; not reproduced on our Tesla-only fleet).

| var | default | what it does | measured | gate class |
|---|---|---|---|---|
| `PXA_SPEC_1ROW` | **on** | extends the single-output-row GEMV (`mul_mat_1row`) from `ne11==1` only to spec-verify batch sizes `Ny<=8` (one CUDA block per token column instead of one block total); `=0` restores the old ne11==1-only dispatch, the missed shapes fall back to cuBLAS as before | **V100 single-card, ub1024 fa-on, MTP n1: +6.6% decode (110.64 vs 103.82 t/s off)** — 6-round fair-battle, median of rounds 3-6. **Flat/harmless on P100 ub512 (+0.06%, noise) and on a 2xV100 `-sm layer` split (+0.7%, noise; also flat under `-ts` PXQ4-MTP, 102.34 vs 101.66)** — the single-GPU win doesn't transfer once the model is split across cards, but nothing regresses. `Ny==1` launches the identical arithmetic DAG as before (bit-identical). `Ny` 2..8 replaces cuBLAS's tiled reduction with a per-thread fp32 dot product | **G3-class, gated like `PXA_PXQ_INT8_PREFILL` (logit-identity + accept-parity, not a sha gate).** Gate 1 (MTP off, temp-0 sha, 1 V100): 5/5 bit-identical, patched vs baseline. Gate 2 (MTP on, 3 arms, 1 V100): baseline == patched+`SPEC_1ROW=0` exact on all 5 prompts (identical sha + identical draft/accept counts, 0.8036 both); patched-default diverges from both on 2/5 prompts — confirmed via logprobs to be a genuine float-ULP near-tie flip (top-2 candidates within ~0.01-0.02 nats at the first diverging token) between the fp32 dot product and cuBLAS's tiled reduction, not a bug. Both diverged responses stay coherent and land on the same correct final answer, phrasing only. MTP accept-rate held (0.8036 off/baseline vs 0.8482 default-on; the higher total draft count is a longer path, not degraded acceptance) |
| `PXA_CUBLAS_EAGER_INIT` | **on** | creates each device's cuBLAS handle + a preallocated `cublasSetWorkspace` buffer (cuBLAS's own default size, 4 MiB pre-Hopper / 32 MiB Hopper+) at backend init, before weights fill VRAM, instead of lazily on first use; `=0` restores stock lazy creation. Alloc failure at init falls back to stock lazy behavior silently | **Perf-neutral by design, measured: P100 ub512 +0.05% (noise); V100 ub1024 −1.5% (110.64→112.36 off), within the fair-battle's own ~2.7% round-to-round noise band.** Measured VRAM cost ~12 MiB/device (a bit above the theoretical 4 MiB workspace, still trivial). All 13 server starts across both cards in the fair-battle came up healthy in 11-12s, no OOM introduced at any point (V100 ub1024 had 995 MiB headroom, P100 ub512 had 1577 MiB) — our Tesla cards never sat near-full enough to reproduce the lazy-alloc failure the fix targets, so the stability claim is unverified-but-free here, not disproven | bit-exact (handle/workspace plumbing only, no math path touched) |

Full protocol + raw per-round numbers: `bench/spec1row-fairbattle` results are summarized above;
the same-worktree drift-free A/B build (2 target files only) ran on `build-spec1row` vs
`build-baseline`, model PXA-Fusion4-35B-PXQU16-MTP.

## 6. Speculative-decode / MTP / server levers (engine features, model-dependent)

| var | default | what it does |
|---|---|---|
| `PXA_MTP_LAZY_WARMUP` | off; **ENHANCE default ON (2026-07-29, §0c; `=0` rolls back; REFERENCE off)** | `=1`: skips the per-prompt-batch MTP companion warmup and stops flagging every prompt token as an MTP output (large prefill win on MTP models; temp-0 bit-identical; eager mode byte-unchanged when unset) |
| `PXA_AUTO_SAMPLERS` | unset → engages at the ENHANCE level, which is the default (never under `PXA_REFERENCE=1`, and not at `PXA_ENHANCE=0`); `=0` forces off, `=1` forces on at any level | **Server-side model-family sampler DEFAULTS** (2026-07-29): fills only sampler fields the CLI left unset (per-field `--temp/--top-k/--top-p/--min-p` explicitness check; per-request params still win). Family table: `gpt-oss` → temp 1.0 / top_k **100** / top_p 1.0 (top_k=0 — the "official" gpt-oss setting — was measured to **HALVE decode**: 31 vs 65 t/s, full ~201k-vocab CPU sort per token, gpt-oss-120b 6-card cell 2026-07-04); `qwen35moe`/`qwen3next`/`qwen35` → the official Qwen no-think set temp 0.7 / top_k 20 / top_p 0.8 / min_p 0; `laguna` → the same Qwen-family set (HEURISTIC — no per-model sampler sweep yet, the log says so). Unknown arch → stock defaults kept, logged. Plus a generic guard: a resolved `top_k<=0` is clamped to 100 (loud). Every applied default logs value+reason+override. |
| `PXA_MTP_ADAPTIVE` (+`_K`) | off | adaptive draft-length cap for MTP speculation (per-slot acceptance feedback). **MEASURED NULL 2026-07-08 — the only A/B on record: decode Δ≈0** (K=1 vs K=0: reason 44.0/43.6, code 43.6/43.5, agentic 47.4/46.8 t/s — within ~1% noise), accept identical (MTP draft-accept 89–93% both arms), coherence clean (no ×N degeneration; trap gauntlet 24/24 handled). Cell: Qwen3.5-122B-A10B (community "abliterated" abliteration), MXFP4 MoE + Q8_0 output head, MTP-grafted; `build-adaptivek` binary (**PRE-canonical**); 6 Tesla (2×V100 + 4×P100), `-sm layer`, np1, ub2048, `--spec-type mtp:n_max=3,p_min=0.5`. Leave OFF. ⚠ pre-2026-07-28 binary — a kernel change can invalidate prior baselines (PXQ_CANON_v1 did exactly that once); re-measure on the canonical build before quoting anything beyond "null" |
| `PXA_NGRAM_RESET_STREAK` | 3 | streak threshold for the ngram-speculator map reset; `0` = never reset on acceptance (helps varied-writer models keep the map warm) |
| `PXA_NP_SPEC_GATE` | off | opt-in gate for speculation under np>1 (shelved feature; leave unset) |
| `PXA_SPEC_RELAXED` (+`_PMIN`) | level default (ON at ENHANCE, OFF below); `=0` pins exact-match | relaxed draft-acceptance experiment (G3-class; not recommended) |
| `PXA_SHARED_MTP_BATCH_COMMIT` | on | batches MTP commit work across slots; `=0` restores fully-serial behavior (rollback knob) |
| `PXA_MOE_FASTTG_MAX_NY` | 8 | max verify-batch Ny that stays on the per-token fast-TG path; `=1` routes Ny>1 MTP verify batches to the expert-grouped batched path (weights read once per traversal) |
| `PXA_MOE_GROUPED` / `_VERIFY` / `PXA_MOE_BATCHED_VERIFY` | off | A1 expert-grouped batched-MoE verify kernels + shadow-verify harness (G3-class; incompatible with graph capture). ⚠ fixed 2026-07-30: `PXA_MOE_GROUPED_VERIFY` was presence-tested (`=0` still enabled shadow-verify); it is now value-tested like every other lever |
| `PXA_PROMPT_INTERLEAVE` | on | co-decodes resident slots while another slot prefills; `=0` reverts to serialize-behind-decode (ops kill-switch) |
| `PXA_HEALTH_STALL_MS` | 60000 | `/health` reports stalled if a queued probe can't be served within the deadline (`0` = off) — keeps health honest instead of parking an HTTP worker |
| `PXA_WEDGE_EXIT_MS` | **180000** (⚠ doc previously said 0 — the code has always defaulted to 180000/ON) | in-server stall watchdog: one `llama_decode` stuck longer than this across 3 consecutive 5 s checks triggers the runtime-appropriate action (`0` = off). **Container-aware since 2026-07-30:** in a container → `_exit(42)` (the orchestrator restarts); bare metal → an in-process recovery attempt first (`llama_decode_stop()` soft-wedge abort at strike 3), then `_exit(41)` at strike 6 with a loud "no supervisor detected — restart manually" banner. Exit codes are distinct (41 bare / 42 container) so logs can tell the contracts apart. The monitor never forks or re-execs in either mode. |
| `PXA_IN_CONTAINER` | unset (auto-detect) | forces the container verdict used by `PXA_WEDGE_EXIT_v1`: `=1` container contract, `=0` bare-metal contract. Precedence: this override > auto-detection (`/.dockerenv`, `/run/.containerenv`, `container=` in `/proc/1/environ`, docker/containerd/kubepods/libpod/lxc in `/proc/1/cgroup`, container paths in `/proc/self/mountinfo`). The verdict + evidence is printed once at startup (`PXA_CONTAINER_AWARE_v1: runtime=...`) |
| `PXA_PORT_GUARD` | on | pre-bind probe: if a LIVE listener already answers on the target host:port the server refuses to start with an actionable error instead of silently coexisting/fighting (the 2026-07-30 bare-metal duplicate-server incident); `=0` bypasses |
| `GGML_NO_BACKTRACE` | unset | `=1`: on abort, skip the fork-a-debugger backtrace entirely (symbols-only). The fork path is now safe (`PXA_BT_NOFORK_v1`: child `_exit`s instead of `exit`ing — the old `exit()` could deadlock a fork child of the multithreaded CUDA process into an immortal duplicate server; parent waits max 15 s then SIGKILLs), but production servers may still prefer to skip the debugger attach, which stops the process while gdb runs |

## 7. Multi-GPU / partition levers

| var | default | what it does |
|---|---|---|
| `PXA_EXPERT_SHARD` | unset | comma list of device indices: shards expert tensors of the listed home devices across the group (consumed by the CUDA MoE up/gate shard branch). Unset = bit-identical stock placement |
| `PXA_REPLICATE_RECURRENT` | off | replicates recurrent (DeltaNet) state full-head per device instead of head-splitting — trades memory for no cross-device reduce |
| `PXA_REDUCE_CAPTURE` | off | allows the cross-device reduce inside graph capture once the per-device events exist |

## 8. Quantizer inputs (build-time, `llama-quantize`)

| var | what it does |
|---|---|
| `PXA_PXQ_COMPOSITION_OVERRIDE` (env) / `--pxq-composition-override` (CLI) | **SAFETY GATE, not a performance knob — read before ever setting it.** After every PXQ-target quantize the writer sums output bytes by type and asserts (a) PXQ-family bytes ≥ 50% of the file and (b) a uniform PXQ target actually produced bytes of its named tier. On failure it **deletes the mislabelled output** and aborts with `PXQ composition assertion: target <FTYPE> produced <X>% PXQ-family bytes…`. This assertion exists because exactly such a file shipped once: an artifact labelled "PXQ4" that was 27% PXQ / 68% MXFP4 with attention and the shared expert carrying none, which entered a public README as *"tier/codec the only variable"* and stood for weeks (the retracted 104.06). **If you trip this assertion and reach for the override, you are about to ship the same mislabel — rename the output honestly instead.** (Documented 2026-07-30; the guard itself shipped 2026-07-28, `src/llama-quantize.cpp`.) |
| `PXA_PXQ6_BOOK` / `PXA_PXQ2_BOOK` / `PXA_PXQ3_BOOK` / `PXA_PXQ6R_BOOK` | override the frozen codebook — for lab experiments; shipped books are compiled in and sha-pinned |
| `PXA_PXQ6_SUB` / `_SUB_HQ` / `PXA_PXQ2_SUB` / `PXA_PXQ3_SUB` | override the sub-scale LUTs (lab) |
| `PXA_PXQ6_ANCHOR_FIT` / `PXA_PXQ2_ANCHOR_FIT` / `PXA_PXQ3_ANCHOR_FIT` | anchor-fit strategy toggles in the native quantizers (lab; defaults are the shipped, gate-proven settings) |
| `PXA_PXQ_HEAD` | output-head type for the PXQ ftypes (`output.weight` only, NOT `token_embd`): `q8_0` (default) \| `q6_k` \| `f16`; unknown values warn and fall back to `q8_0`. Default q8_0 = **+3.0% P100 decode measured, all rounds** (q8_0 rides Pascal's fast DMMV path where K-quant heads ride the slow scalar path, and the head runs every token over the full ~151k vocab) — and q8_0 is higher precision than the old q6_k default, so speed AND quality. An explicit `--output-tensor-type` still wins |
| `PXA_PXQ_NATIVE` | widens PXQ slab-tier eligibility beyond the routed expert stacks (`*_exps.weight`, always eligible) for the “100% native PXQ, no inherited MXFP4” work. Comma-separated class list: `shexp` (`ffn_{up,gate,down}_shexp.weight`), `attn` (`attn_q` / `attn_qkv` / `attn_output` / `attn_gate.weight`), `embd` (`token_embd.weight`), `ssm` (`ssm_alpha` / `ssm_beta` / `ssm_out.weight` — **risky, recurrent state**), or `all`. Unset (default) = historical behaviour, routed experts only. The geometry gate still applies to every class (`rows % 64 == 0 && K % 32 == 0`); a tensor that fails it is demoted to MXFP4 by the caller, so a bad list can never produce an unloadable file. **Quality:** `attn` measured *better* than the tier it replaces — native-attn PXQ4 beats PXQ6 on a 4000-chunk paired perplexity (t = +15.64) while being 4 GB smaller. **Speed:** the resulting file needs `PXA_PXQ4_2D=1` at runtime or its attention nodes fall back to dequant→cuBLAS; even with it, 2×V100 decode is −10% and prefill −12% vs the MXFP4-attention file, while 4×P100 is at decode parity (+0.5%) and −2.5% prefill (see §4 and the kernel-parity report) ⚠ **Superseded as a default by `PXA_PXQ_BACKBONE` (BACKBONE_REV 2, 2026-07-26):** the rev-2 table now promotes `attn` / `shexp` / dense-FFN / `token_embd` in every PXQ tier without this lever, and to a *higher* tier than this lever gives (PXQ6 rather than the file's own expert tier). `PXA_PXQ_NATIVE` remains a research override for widening eligibility to classes the table does not claim (notably `ssm`). ⚠ **To reproduce a pre-2026-07-26 `PXQ4N-attn`-style artifact byte-for-byte you must ALSO set `PXA_PXQ_BACKBONE=legacy`** — otherwise the backbone table wins and you are measuring a different file than the one the historical number came from. A second latent trap, fixed the same day: `embd` never actually fired (its name test was a strict-suffix match that cannot match the top-level `token_embd.weight`), so any "embd is a null" result predating the fix is null **by construction**, not evidence |
| `PXA_PXQU_DIR` | directory a bare `--pxq-universal <name>` argument resolves against, as `$PXA_PXQU_DIR/<name>.tiers` (default `pxa-bench/pxq-universal/`). Absolute and relative paths bypass it |
| `--imatrix` (behaviour note, not a lever) | every PXQ codec **consumes** the importance matrix: the anchor pick and the code/sub-scale selection both minimise a diagonal-weighted SSE (`err += w[i]·e²`), per-expert columns when `imx_size == K·E` and a shared column when `== K`. Since 2026-07-26 a **dead-column guard** sits in front of it: an all-zero (or non-finite / negative) column would make *every* candidate score exactly `0.0`, silently collapsing the argmin into its tie-break, so such a column is fit **unweighted** instead and the total is reported at completion. This is the normal case for routed experts that never fired during calibration — expected on a 256-expert MoE — but a large count means the corpus is too thin for the expert count. Corpus guidance: ≥2000 chunks, and compute the imatrix with `-sm graph` (pure prefill, measured +163%) |
| `PXA_PXQ_BACKBONE` | **the backbone allocation table (BACKBONE_REV 2, default ON since 2026-07-26).** Before rev 2 every PXQ tier quantized its routed experts natively and then flattened *everything else* — attention, `token_embd`, shared experts, dense FFN — to flat MXFP4. Measured on Laguna-S-2.1 vs the stock `Q4_K_M` recipe from the same bf16 source: `attn_output` **3.2×** the error (0.1157 vs 0.0362), `attn_gate` **1.6×** at the worst absolute value (0.1609), `attn_q` / `token_embd` / `ffn_*_shexp` 1.6× each — while the PXQ6 file was **8% BIGGER** than the Q4_K_M it lost to. Rev 2 replaces the flatten with a per-class table: `attn_q`/`attn_qkv`/`attn_output`/per-channel `attn_gate`/`ffn_*_shexp`/dense `ffn_*` → the native PXQ tier above the experts (**PXQ4 for the PXQ4/PXQ4HQ/PXQ6 tiers as of 2026-07-31, was PXQ6**; PXQ4HQ for PXQ3; PXQ4 for PXQ2). **On a model with NO routed experts the named tier governs the backbone instead** — the promotion is defined relative to an expert tier and a dense model has none, so a PXQ6-tier dense quant under the new default emitted 140 tensors / 193.92 MiB of PXQ4 and ZERO pxq6 (composition-identical to the PXQ4 arm) and the composition assertion correctly refused to write it. Measured on Qwen3-0.6B-Q8_0. An explicit `core`/`hq`/`pxq6` still overrides on dense, with the assertion as the backstop; `attn_k`/`attn_v` → `q8_0` (unchanged, already at Q4_K_M parity); **per-HEAD `attn_gate` (`ne[1] ≤ 256`) → `f16`** — the Laguna killer, 72 softplus scalars per layer at 0.03% of the file whose 16% error becomes a 31–38% functional error on every head; `token_embd` → `q6_k` (a row *gather*, never a GEMM, so the Pascal "k-quants are slow" rule does not apply); `output` → `q8_0` via `PXA_PXQ_HEAD` (unchanged); geometry failures land on `q8_0` instead of a silent MXFP4 demotion. `ssm_*`, `nextn.*` (MTP), the router and the norms are untouched. **Cost: +0.45 GiB on a 62.0 GiB Laguna PXQ4 = +0.73%.** Values: unset/`1`/`v2` = rev 2 [default]; **`pxq6` = restore the pre-2026-07-31 PXQ6 backbone on the 4-/5-bit tiers** (use it to byte-reproduce any artifact built before 2026-07-31, exactly as `legacy` does for pre-2026-07-26); `legacy`/`0` = the exact pre-rev-2 recipe (byte-reproduces old artifacts — use it for any A/B against a pre-2026-07-26 file); `lite` = **only the promotions that are free at decode** — per-head `attn_gate`→`f16`, `token_embd`→`q6_k`, `attn_k`/`attn_v`→`q8_0`, geometry-fail→`q8_0` — leaving the dense GEMM backbone on MXFP4. This keeps the class that actually corrupted Laguna (the per-head gate) at essentially zero decode cost, and is the option to reach for if the full table's decode cost is unacceptable on your cards; `hq` = the pre-registered fallback, PXQ4HQ instead of PXQ6 on the 4-/5-bit tiers (≈82% of the modelled gain at +0.26 bpw instead of +1.02, and PXQ4HQ — unlike PXQ6 — has a CPU panel-dequant, so the file stays partial-offload capable); `universal` = additionally apply the table to `PXQ_UNIVERSAL` and `PXQ1` (off by default: a PXQU tier map is user-authored per-tensor and neither tier has measured backbone evidence). Recipe-level only — **no on-disk slab layout changes, so every previously shipped PXQ file still loads.** Every file records which table built it in the `pxa.pxq.backbone_rev` / `pxa.pxq.backbone_map` gguf KVs. ⚠ A rev-2 file needs `PXA_PXQ4_2D` at runtime (now default ON) or its promoted backbone falls back to a per-token dequant→cuBLAS — **measured 45.2 → 9.3 t/s, −79%**, on a rev-2 Laguna-XS on 2×P100. ✅ **RESOLVED 2026-07-31 by moving the 4-/5-bit default backbone from PXQ6 to PXQ4.** The costs below were all measured against a **PXQ6** backbone, which is not MMVQ-registered (`pxa_pxq_mmvq_type()` admits PXQ4/PXQ4HQ only), so it dropped the dense and shared-expert FFN off the fused decode path.

**The two halves only pay TOGETHER — ship both or neither.** Full 2×2 on HY3 (299B, 192 experts, 8×V100-SXM2, f16 KV, `-c 131072 -np 2 -b/-ub 2048`, NVLink order `0,2,3,7,6,4,5,1`; n=5 cached-prefill decode runs per arm, arms interleaved, fresh server per arm, one binary):

| backbone | MMVQ | decode t/s | vs default |
|---|---|---|---|
| PXQ6 (old default) | off | 19.61 | — |
| PXQ6 | armed | 19.60 | −0.05% |
| PXQ4 (new default) | off | 20.74 | +5.8% |
| PXQ4 | armed | **24.55** | **+25.2%** |

**Per-tier composition, verified on a 0.6B DENSE model** (dense on purpose: with no routed
experts every weight is backbone, so the quantizer own by-bytes composition report reads the
backbone table directly). Tiers whose named type only lands on experts legitimately produce zero
bytes of their own name there and the composition assertion aborts the WRITE -- that is a result,
not a failure, and the composition prints before the abort:

| arm | backbone produced | note |
|---|---|---|
| `PXQ4` default | pxq4 51.7% | the new default |
| `PXQ6` default | **pxq6 56.9%** | dense guard: the named tier governs when n_expert <= 1 |
| `PXQ2` default | pxq4 43.5% + **pxq4hq 8.4%** | unchanged; attn_output stays on the HQ tier |
| `PXQ3` default | pxq4hq 53.1% | unchanged |
| `PXQ2`/`PXQ3`/`PXQ6` `=core` | pxq4 51.7% (all three identical) | core reaches every tier |
| `PXQ1` `=core` | mxfp4 64.7% | silent no-op -- needs `core,universal`; now warns |
| `PXQ1` `=core,universal` | pxq4 51.7% | the BACKBONE allocates correctly — but read the next line before using this |

> ⚠ **`works` above means the backbone, NOT the tier.** Measured 2026-08-24 by the PXQ
> type-coverage sweep: on a DENSE model `PXQ1` produces **zero bytes of `pxq1`** even with
> the `universal` opt-in — the 51.7% is the pxq4 backbone, and the composition assertion
> then aborts the write. PXQ1 is an EXPERT-tier codec and has no dense path; that is a
> designed limit correctly caught by the assertion, not a bug. On a real MoE the native
> codec does fire on the routed expert stacks (measured: 80.9% pxq1, rc=0).
> Two further cautions on this tier: it has **no CPU codec** (GPU-only; a CPU-only or
> partially-offloaded run aborts), and every MoE artifact containing `pxq1` in that sweep
> generated incoherent output while every artifact without it was fine — isolated across
> six controls, with a missing imatrix as the only uncontrolled variable. Not enough to
> call the codec defective, and enough that PXQ1 should not be shipped to anyone until
> that is resolved. The per-type baseline matrix is kept with the serving stack.

| `PXQ4` `=pxq6` | pxq6 56.9% | byte-identical to the PXQ6 default: rollback holds |
| `PXQ4` `=hq` | pxq4hq 53.1% | identical to the PXQ3 default |

The ~19pp interaction is the whole result: the backbone swap makes tensors ELIGIBLE, the DEFAULT-level `PXA_PXQ_MMVQ` auto-set makes the kernel FIRE. Three independent confirmations of the mechanism: the eligible-tensor census (PXQ6-backbone file exposes **240**, PXQ4-backbone **645** — a 405 gap matching the backbone-tensor count exactly); the `PXA_PXQ_MMVQ: mode 1` dispatch line printing on the core+armed arm ONLY (it is resolved lazily, so on a PXQ6 backbone nothing eligible ever reaches it); and the flat PXQ6 pair itself. ⚠ Compare with the +30–35% from the 2026-07-28 sm_70 campaign on Qwable-27B / Fusion4-35B: same range, mildly diluted, because the always-resident backbone is a smaller share of per-token traffic as a model gets more expert-dominated. Size the effect against a model of comparable expert count, not across the 27B↔299B gap. Kept for the historical record: ⚠⚠ **Decode cost of the full table, as measured against the OLD PXQ6 backbone:** same file/cards/harness (n_predict 128, 4 reps, one instance — indicative, NOT the fill-5739 n≥8 interleaved standard), decode is legacy-backbone **71.6** t/s, `hq` **51.6** (−28%), full PXQ6 table **45.2** (−37%), and `PXA_PXQ4_2D_KSPLIT=1` makes it worse still at **32.1** (extending that lever's known P100 loss to the previously-untested `panels=8` shexp shapes — leave it OFF). Cause: in a 256-expert MoE only ~8/256 of the expert bytes are touched per token, so the always-resident backbone is roughly HALF the per-token decode traffic, and the PXQ 2D decode mmv is slower per byte than MXFP4's mmvq/DMMV on Pascal. Confirmed like-for-like on the real artifact — Laguna-S-2.1 PXQ4 built from the same bf16 source, 6 Tesla cards, `-c 4096`, n=1: shipped legacy-backbone **34.8** t/s vs rev-2 **26.6** t/s = **−23.6%**. Until that kernel is competitive, `lite` is the speed-preserving way to take the gate/embedding half of the fix |
| `PXA_PXQ_ERRBUDGET` (+ `_REF`, `_MAXELEM`) | **default off** (it costs a full dequant pass). Quantize-time acceptance gate: dequantizes each written tensor straight back, accumulates relative RMS error vs the f32 source **per tensor class**, prints a table at completion and writes `<output>.errbudget.tsv`. `PXA_PXQ_ERRBUDGET_REF=<tsv>` compares against a stored budget (e.g. the `Q4_K_M` control's) and WARNs at **>1.5×** on any class; `PXA_PXQ_ERRBUDGET_MAXELEM` caps per-tensor sampling to whole rows (default 64M elements, `0` = every element). This is the instrument that would have caught the corrupt Laguna artifacts on disk — the signature (`attn_gate` 0.1609, `attn_output` at 3.2× the control) was in the numbers before a single token was generated. **Limitation, stated not hidden:** the PXQ slab types are CUDA-only and have no CPU `to_float`, so expert classes report `n/a` — the report covers the BACKBONE, which is the entire surface the bug lived on and the entire surface rev 2 changes. Row-interleaved (`*_R4`) types are also reported `n/a` rather than decoded row-wise incorrectly |

## 9. Diagnostics (no effect on results; may cost speed — leave OFF in production)

`PXA_EXPERT_LOG` (per-request MoE expert-routing histograms, np1 only), `PXA_PROFILE` /
`PXA_PROFILE_EVERY` / `PXA_CKPT_PROF` / `PXA_DECODE_WALL_DBG` / `PXA_SHARD_TIMING` (timing
instrumentation), `PXA_GRAPH_DUMP` (graph node dump), `PXA_MOE_DEBUG`, `PXA_SPEC_DBG` /
`PXA_DRAFT_DBG` / `PXA_MTP_DBG` (speculation tracing), `PXA_BIGCOPY_DBG` (large D2H copy
tracing), `PXA_OP_SYNC_CHECK` / `PXA_SYNC_BISECT` (per-op sync bisection for debugging async
faults), `PXA_CHATPARSE_EVERY` (chat-parse cadence).

## 10. Architecture support: GLM-4.7-Flash (`glm4_moe_lite`)

`Glm4MoeLiteForCausalLM` (HF `model_type: glm4_moe_lite`, e.g. `zai-org/GLM-4.7-Flash`) is a
**different architecture from GLM-4.5/GLM-4.6** (`Glm4MoeForCausalLM`, `model_type: glm4_moe`,
loaded here by `build_glm4_moe()`). Flash is DeepSeek-V2/V3-lineage — MLA attention, sigmoid
(`noaux_tc`) MoE gating with a score-correction bias, and a NextN/MTP tail — not the GQA
attention `build_glm4_moe()` implements. Mapping it onto that graph produces shape mismatches
or silent rope garbage; it must NOT be treated as a `glm4_moe` variant.

- **Converts as `deepseek2` (MLA), not a new arch.** `convert_hf_to_gguf.py`:
  `Glm4MoeLiteModel(DeepseekV2Model)` (`model_arch = gguf.MODEL_ARCH.DEEPSEEK2`), registered
  `@Model.register("Glm4MoeLiteForCausalLM")`. Two gaps closed on top of the existing
  `DeepseekV2Model`: (1) the HF config omits `scoring_func` (uses `topk_method: noaux_tc`
  instead) where `DeepseekV2Model.set_gguf_parameters` hard-indexes it — the subclass
  `setdefault`s it to `"sigmoid"` (correct: the `e_score_correction_bias` tensor is present,
  and the loader's own GLM-4.7-Flash 47-layer heuristic already defaults to sigmoid gating
  when the KV is missing); (2) the tokenizer pre-hash `cdf5f353...` (GLM-4.7-Flash's own
  vocab, distinct from GLM-4.5's `9ca2dd61...`) is mapped to `res = "glm4"` — the fork's
  `tokenizer.ggml.pre == "glm4"` vocab path already existed and needed no changes. Everything
  else (expert 3D-stacking into `ffn_{gate,up,down}_exps`, `kv_b_proj` → `attn_kv_b` +
  transposed-split `attn_k_b`/`attn_v_b`, `e_score_correction_bias` → `exp_probs_b.bias`,
  `q_lora_rank`/`kv_lora_rank`/`key_length`/`value_length`/`rope.dimension_count` KV writes) was
  already correct in `DeepseekV2Model` — this is a converter-registration gap, not a loader gap.
- **Loads on the existing `LLM_ARCH_DEEPSEEK2` graph — no loader changes needed.** The loader
  already special-cases the 47-layer shape (`n_layer==47` → sigmoid gating default, `is_lite`
  heuristic correctly excludes it so `q_lora_rank` is read) and already builds `wk_b`/`wv_b` +
  shared-expert tensors generically. Verified end-to-end: a mainline-converted (unsloth)
  GLM-4.7-Flash GGUF loads clean and produces coherent temp-0 output under this fork; the
  fork's own converter output was verified separately against a synthetic fixture (below).
- **Run flags: `-fa on -mla >= 1` is REQUIRED for `-sm` multi-GPU on this (and any MLA) arch** —
  the multi-GPU graph path aborts otherwise. Bake `-fa on -mla 3` (or `-mla 1`) into any launch
  config; a temp-0 `-mla 1` vs `-mla 3` A/B on the same GGUF must be token-identical (divergence
  there is a graph bug, not an expected quant/precision difference).
- **NextN/MTP layer is skipped at conversion, by design.** `model.layers.<num_hidden_layers>.*`
  (the NextN/MTP block: `eh_proj`/`embed_tokens`/`enorm`/`hnorm`/`shared_head.{head,norm}` plus
  a full attn+MoE block) is dropped by the existing `DeepseekV2Model.modify_tensors` layer-index
  guard — this fork's native MTP tail (`build_glm4_moe_mtp()`) is hard-gated to `LLM_ARCH_GLM_DSA`
  only, so GLM-4.7-Flash gets no speculative-decode head from this pass; wiring the fork's tail
  to the NextN block is future work, not part of this conversion path.
- **PXQ4/PXQ6 quantize the routed experts natively, no code changes.** `pxq4_tensor_eligible`
  (name ends `_exps.weight`, `ne[1] % 64 == 0`, `ne[0] % 32 == 0`) is purely name/shape driven —
  GLM-4.7-Flash's `ffn_{gate,up}_exps` (`ne=[2048,1536,64]`) and `ffn_down_exps`
  (`ne=[1536,2048,64]`) qualify with zero arch-specific handling.
- **MLA small-tensor quant lever — `attn_v_b` is the one gap.** `llama-quantize`'s legacy
  loose substring match `name.find("attn_k") != npos` (written for classic single `attn_k`
  GQA tensors) incidentally also matches `attn_k_b`, `attn_kv_a_mqa`, and `attn_kv_b` (all
  contain `"attn_k"` as a substring) — combined with `n_expert >= 4` this already forces those
  three MLA tensors to `q8_0` for free, no code change needed. `attn_v_b` does **not** match
  (no `"attn_k"` substring, and the exact-suffix `"attn_v.weight"` check doesn't fire either),
  so it rides plain MXFP4 (bs32; legal, `ne[0]=192` on the real model is not 256-superblock
  divisible but MXFP4/q8_0 don't care). If a quality gate ever flags the MLA path, force
  `attn_v_b` to `q8_0` explicitly (`--attn-v-type q8_0` or a `--custom-quants` regex override);
  `attn_k_b`/`attn_kv_a_mqa`/`attn_kv_b` need no such override.
- **CI conversion smoke test:** `tests/test-glm47flash-convert.sh` +
  `tests/fixtures/glm4-moe-lite-tiny/` — a synthetic 2-layer (`num_hidden_layers=2`,
  `first_k_dense_replace=1`) fixture carrying every real tensor-name pattern including the
  NextN tail, the REAL tokenizer (so the `cdf5f353...` pre-hash actually fires), and a
  config.json that deliberately OMITS `scoring_func` (the exact key gap this conversion path
  must survive). Asserts an exact tensor-name-set match (36 tensors; nothing missing, nothing
  extra, no NextN leak), `general.architecture==deepseek2`, and the sigmoid/MLA/tokenizer KVs.
  CPU-only, no GPU, no model download — safe on every commit.
- **Verified on real weights (2026-07-22).** `llama-quantize --allow-requantize` q8_0 → PXQ4 on
  the real 30.159 B model: 138/138 `_exps` take PXQ4 native (embedded tensor type 252, E16-row
  scales, tier core/bs16), head → q8_0, `attn_k_b`/`attn_kv_a_mqa`/`attn_kv_b` keep q8_0,
  `attn_v_b` → MXFP4, routers + `exp_probs_b` stay F32 — exactly the fixture prediction. Output
  15.307 GiB / 4.360 bpw (tensor mix: 281 f32, 142 q8_0, 330 mxfp4, 138 pxq4). Loads split across
  2× P100-16GB (`-ngl 99 -sm layer -fa on -mla 3 -c 8192`, 8279+7234 MiB weights, 423 MiB KV) with
  temp-0 coherent output on capital-continuation / factual-QA / code-reasoning prompts. mla A/B on
  a 1.5k-token cold prompt (P100 pair): prefill 686 t/s (`-mla 3`) vs 139 t/s (`-mla 1`) — ~4.9× —
  decode parity (~10.5 t/s at 1.5k fill; 24–28 t/s near-empty). `-mla 3` is the setting to ship.

---

*Numbers cite their config; anything without a number has no published A/B on the shipped
configs — treat its default as the tested state. Full A/B raw logs live with the bench suite
release notes.*

## Updates — 2026-07-28 — PXQ_CANON_v1, merged sm70-stack levers, backbone rev additions

| lever | default | what it does | measured | gate |
|---|---|---|---|---|
| `PXA_PXQ_DENSE_GATEUP` | **on** (merged from `sm70-stack` 2026-07-28) | Dense `GGML_OP_FUSED_UP_GATE` nodes with PXQ-typed up/gate now reach the fused `k_pxq6_gateup_mmv` / `_ksplit_gen` family through the one-entry `{0}` ids idiom instead of splitting into two `k_pxq6_mmv` calls + an elementwise GLU. SILU only; declines (with a one-shot FIRING/DECLINED log) for anything it cannot serve. ⚠ NOT bit-exact vs the unfused path (GLU association: fused vs split-node order) — a PATH choice, deterministic under a fixed config; `PXA_PXQ_DENSE_GATEUP=0` restores the old path. | Qwable-27B dense, 2×V100: part of the −21.6% → −4.25% recovery; 2×P100 stack total **+45.8% vs MXFP4** (14.268 → 20.796 t/s). Composes with the S-split; under PXQ_CANON_v1 the S choice no longer affects its output bits. | behaviour-gauntlet (fused-vs-unfused association), ON |
| `PXA_PXQ_DENSE_GATEUP_SPLIT` / `PXA_PXQ_DENSE_GATEUP_TARGET` | split follows arch default (cc 7.0: never split; else shared target) | The fused gateup carries two weights per block, so its grid does 2× the work of the mmv grid; sm_70 is past the knee at typical dense shapes and never splits (+0.85%), sm_60 keeps the shared target (−8.6% if disabled). | measured on Qwable-27B PXQ4core | bit-exact under PXQ_CANON_v1 |
| `PXA_PXQ4_2D_SPLIT_TARGET` (defaults) | **16×nsm on cc 7.0, 8×nsm on cc 6.0, 2×nsm elsewhere** (2026-07-28, was 2×nsm everywhere) | Measured sweeps on both Teslas showed the old 2×nsm default sat far below the knee (V100: 160→27.89 t/s vs 1280→30.70; P100: 112→14.81 vs 448→19.17). Under PXQ_CANON_v1 the retune is output-neutral (S does not change bits) — pre-canon it silently changed greedy output, which is why it was never shipped as a default before. | see the `bd0977e` sweep table | bit-exact under PXQ_CANON_v1 |
| `PXA_PXQ_BACKBONE=core` (new token) | off (rev-2 table default) | GEMM backbone (attn_q/qkv/output/per-channel gate, shexp, dense FFN) at the byte-parity **PXQ4 core** tier (4.2526 bpw vs MXFP4 4.2500) instead of PXQ6 — every backbone class becomes `PXA_PXQ_MMVQ`-eligible, which is the sm_70 recipe (attn PXQ6+MMVQ = +0.4%, attn PXQ4+MMVQ = +6.7% MoE decode). Previously only expressible via `--custom-q`. | MoE fidelity/speed arms: the 2026-07-28 arm matrix | quantizer-side |
| `PXA_PXQ_KV` | unset (=q8_0) | **Implemented 2026-07-28** (was a docs-only phantom — the audit found 0 source hits and an unconditional pin). Overrides the rev-2 K/V pin; accepts `q8_0`/`pxq4`/`pxq4hq`/`pxq6`/`mxfp4`. `mxfp4` restores true byte parity vs a flat-MXFP4 legacy control (whose K/V are MXFP4, not q8_0 — the pin rationale was wrong for that comparison). | artifact effect verified earlier (attn_k 10.00 → 2.66 MiB/layer on Qwable); speed/quality of the pxq tiers on K/V still unmeasured | quantizer-side |

**Backbone rev additions (2026-07-28, same branch):** `ssm_alpha`/`ssm_beta` (geometrically impossible for the 64-row PXQ panel) and non-expert `nextn.*` MTP companions now land on **q8_0** instead of flat MXFP4 (zero-MXFP4 rule; +0.01% file size). **`ssm_out` was claimed by the backbone table on 2026-08-25** and no longer lands on flat MXFP4. It was the LAST class still doing so in an otherwise-passing PXQ run: measured on a 24-layer DeltaNet draft model, a plain `PXQ4` target put its 18 `blk.N.ssm_out.weight` tensors on `mxfp4` (19.12 MiB, 4.2% of the file) purely because the resolver returned “not mine” for them — and because the tensor is geometry-ELIGIBLE, the “geometry failure → q8_0, never a silent MXFP4 demotion” rule never covered it. After the change the same build reports `pxq4 138 tensors` and **no `mxfp4` row at all**, at 249.38 MiB vs 249.33 MiB — byte parity, as expected from PXQ4 core's 4.2526 bpw vs MXFP4's 4.2500. This also aligns the default recipe with the shipped 35B one, which was already building `ssm_out` as PXQ4 by hand (see the `core+ssm_out→PXQ4` ship-file row below). ⚠ **The direction is reasoned and byte-neutral, not measured**: no paired-perplexity `ssm_out` arm exists in this tree, and `ssm_out` feeds a recurrent state. `PXA_PXQ_BACKBONE=lite` (or `=legacy`) still leaves it on MXFP4, and an explicit `--custom-q` rule still wins, so both A/B arms remain buildable. **Explicit `--custom-q` PXQ targets are now honoured in every backbone mode whenever slab geometry allows** — before this, `PXA_PXQ_BACKBONE=legacy` + a custom PXQ rule silently demoted back to MXFP4 and the arm came out byte-identical to its control (two agents measured "nothing" this way).

**⚠ Gate-methodology warning (systemic, 2026-07-28):** any DECODE-window lever (dispatch gated on `ne11 <= 8`-class conditions) that was certified with default-batch `llama-perplexity` was certified by a run in which the lever NEVER EXECUTED — `-b 512` perplexity is pure prefill, both arms are bit-identical, and the gate returns a false PASS. Gate decode levers at `-b 8 -ub 8` (the arms DIFFERING is what proves the kernel engaged), or with temp-0 + fixed-seed-sampling generation hashes as PXQ_CANON_v1 was. Audit note: `PXA_PXQ4_2D_SPLIT`/`PXA_PXQ6_KSPLIT_GEN` were certified partly through 64-chunk NLL aggregates; their real generation-level failure mode (the 122B runaway) was invisible to that gate and is now covered by the canon bit-exactness regression instead.

## Updates — 2026-07-29 — ship-recipe measurements (all on the merged engine)

- **`PXA_PXQ_MMVQ` — measured for the ship decision (35B MoE, 2×V100 ts1.05,0.95, fill 6018, n=9/arm
  interleaved):** core-backbone file 75.97 → **93.43-93.89 t/s (+23%)**; ship file (core+ssm_out→PXQ4)
  75.07 → **95.17 (+27%, legacy-recipe parity)**. Fidelity, HONEST decode-window protocol (`-b 8 -ub 8`,
  300 paired chunks, V100 — the kernel provably engaged: series differ): **+0.053% ppl (t=17)** — real,
  precisely quantified, and 1/50th of the recipe's −2.72% quality win. **Ships ON in the 35B serve env.**
  ⚠ CAVEATS THAT NOW MATTER (default-on ship): (1) the MMVQ TU carries **frozen copies** of the book/SUB
  tables and does NOT honour `PXA_PXQ6_BOOK`/`_SUB`/`_SUB_HQ` runtime uploads — the dispatch declines when
  those envs are set, so an override experiment silently reverts to the bespoke path; (2) `=1` gates on
  cc≥7.0 — on sm_60 it is an intentional no-op (P100 keeps the bespoke path, which beats MXFP4 there);
  a "null" measured on P100s at `=1` is null by ARCH GATE, not evidence (this produced one false
  null in an earlier sweep).
- **`PXA_PXQ_KV=pxq4` — first-ever measurement (the lever was a docs-only phantom until 2026-07-28):**
  artifact verified (all 20 attn_k/v tensors land PXQ4, file −11 MB vs q8_0 pin); paired 1000-chunk ppl
  + V100 speed cells queued (`nll-A6-kv4.txt` / speed matrix follow-up) — row to be completed with those
  numbers before any public release.
- **`PXA_PXQ4_2D_SPLIT` / canon price:** a 2×2 arm matrix on the published 35B artifact priced
  PXQ_CANON_v1 at **−1.9% decode** on the bespoke MoE path (the ~27% regression hypothesis was tested
  and killed; the 104.06 figure it was anchored to was retracted as an artifact-composition error).
  Bit-exactness costs 1.9% and is worth it: it retires the entire config-dependent greedy-flip class
  (S, target, device SM count, lever toggles).

## Updates — 2026-07-30 — container-aware wedge exit + fork root cause + strict quantize type args

- **`PXA_BT_NOFORK_v1` (ggml.c):** root cause of a duplicate-server incident observed on a
  multi-GPU bare-metal host (2026-07-30). `ggml_abort` → `ggml_print_backtrace()` forks to attach gdb; with no
  debugger on PATH the child called `exit()` — atexit/dtor handlers in a fork child of a
  multithreaded CUDA process deadlock on mutexes held at fork time, leaving an immortal
  duplicate of the server (parent's argv, PPID = parent, holding a dup of the listening
  socket) while the parent blocked forever in `waitpid`. Fixes: child `_exit`s, parent waits
  ≤15 s then SIGKILLs, `GGML_NO_BACKTRACE` honored, `fork()` failure falls back to symbols.
- **`PXA_CONTAINER_AWARE_v1` + `PXA_IN_CONTAINER` + exit codes 41/42:** see the section-6 rows.
- **`PXA_PORT_GUARD_v1`:** see the section-6 row.
- **`PXA_UTF8_FINAL_v1` (server):** final responses no longer 500 when a generation stops
  mid-multibyte-codepoint (`n_predict` cap / `ignore_eos`) — the incomplete trailing UTF-8
  bytes are dropped from the final payload exactly as the streaming path already did
  per-chunk. Closes the "incomplete UTF-8 string; last byte: 0x.." open bug (2026-07-29).
- **`PXA_TYPEARG_STRICT_v1` (`llama-quantize`):** all 12 `--*-type` flags now HARD-FAIL on an
  unparseable type name instead of silently ignoring the flag (exit 0, wrong artifact — the
  2026-07-29 open bug), and type names match case-insensitively (`q6_k` == `q6_K`).

### 2026-07-30 — previously UNDOCUMENTED levers (the A14 inventory closed out)

Every row here existed in source with no doc row (flagged by the 2026-07-29 A14 audit).
Documented so the sm_70 dead ends are reproducible instead of folklore:

| var | default | where | what it does |
|---|---|---|---|
| `PXA_PXQ_MMVQ_ROWS` | **4** (accepts 1\|2\|4\|8\|16) | `pxq-mmvq.cuh` | rows-per-block for the PXQ MMVQ decode kernel. ⚠ ROWS≥8 is a MEASURED DEAD END on sm_70 (register-bound: ROWS=1 62 reg/0 B spill → ROWS=16 183 reg/256 B spill; the "37.89 t/s ROWS=8 beats MXFP4" figure was FABRICATED — no on-disk log). This knob is how those dead ends were expressed |
| `PXA_PXQ_MMVQ_VDR` | **2** (accepts 2\|4) | `pxq-mmvq.cuh` | values-per-dot-iteration for the same kernel. VDR=4 measured −4 to −6% — dead end, kept for reproducibility |
| `PXA_MOE_GROUPED_VERIFY` | off | `grouped_moe_verify.cuh` | shadow-verify for the A1 grouped MoE path: grouped writes a private scratch, per-token path stays authoritative, mismatches print. ⚠ **Fixed 2026-07-30: this was PRESENCE-tested (`=0` still enabled it); it is now value-tested like the rest of the performance levers.** Presence-testing survives only in the graph-family diagnostics (`PXA_CUDA_GRAPH_MOE`, `PXA_CUDA_GRAPH_LOG`, `PXA_CUDA_GRAPHS_PASCAL`) — for those, `unset` is the only "off" |
| `PXA_MTP_ADAPTIVE_K` | 0 | `common/speculative.cpp` | acceptance-EMA adaptive draft depth (companion to `PXA_MTP_ADAPTIVE`). Measured null 2026-07-08: decode Δ≈0, draft-accept 89–93% unchanged, coherence clean — 122B MXFP4-MTP cell on a pre-canonical binary; see the §6 row for the full cell + the re-measure caveat |
| `PXA_SPEC_RELAXED_PMIN` | 0.05 | `common/sampling.cpp` | p_min floor for `PXA_SPEC_RELAXED`, which is ON by level default at ENHANCE (see the §4 row); only consulted at temp>0 |
| `PXA_REP_GUARD` | level default | `common/sampling.cpp` | repetition-attractor guard; supersedes `PXA_PXQ1_REP_GUARD` (back-compat alias). ⚠ 2026-07-30: ENHANCE auto-arm narrowed back to PXQ1-bearing files only — the any-PXQ broadening measurably flipped a temp-0 exact answer on PXQ4 (see the §1 row for the full measurement); `=1` forces any-PXQ, `=0` disables |
| `PXA_ENHANCE_DBG` | off | `pxa-enhance.cuh` | prints the detected topology + every chosen ENHANCE config (diagnostic; use it to audit what the adaptive layer decided) |
| `PXA_CUDA_GRAPH_MOE` | off (**presence-tested** — any value, including `0`, enables it) | `ggml-cuda.cu` | relaxes the `MUL_MAT_ID` capturability guard for MoE decode. The stock guard refuses capture whenever `n_expert_used > 1`, which fires on every routed token (qwen35moe routes 8) so capture never engaged at all; with this set, the guard becomes the predicate the fast-TG dispatch itself uses (`ne2 > 8 || src1->ne2 != 1 || src0 not quantized`), because the expert ids are read on-device and only the host-grouping path truly synchronizes. Inert unless a graph path is enabled. **Unmeasured on its own**; the family was measured **null ×3 on an sm_70 dense cell**, and whole-token graph replay is an outright measured loss on both Tesla arches (see `PXA_CUDA_GRAPH_V2`, §5) |
| `PXA_CUDA_GRAPH_LRU` | **8** slots (floor 2) | `ggml-cuda.cu` | LRU cap on the per-context CUDA-graph cache; past the cap the least-recently-used graph is evicted. **Read only on the `PXA_CUDA_GRAPH_V2` path**, so inert in the shipped default (V2 is opt-in and measured negative). **Unmeasured: no A/B on record for any cap value** |
| `PXA_CUDA_GRAPH_REARM` | **256** evaluations | `ggml-cuda.cu` | replaces the legacy permanent `disable_due_to_too_many_updates`: after 4 consecutive evaluations that require a graph update, replay goes on cooldown for this many evaluations and is then re-armed, so early shape churn cannot kill replay for the life of the process. `=0` restores the legacy permanent disable. Read only on the `PXA_CUDA_GRAPH_V2` path. **Unmeasured: no A/B on record** |
| `PXA_CUDA_GRAPH_BATCH_MAX_NY` | **8** — the fast-TG dispatch ceiling; the env can only LOWER it (values `≥1` and `< current`), and `PXA_MOE_FASTTG_MAX_NY` lowers it too | `ggml-cuda.cu` | upper token-count (`ne2`) bound at which a MoE node is still treated as graph-capturable under the opt-in `PXA_CUDA_GRAPH_BATCH` family. Inert unless `PXA_CUDA_GRAPH_BATCH=1`. **Unmeasured: no A/B on record** |
| `PXA_PXQ_DISPATCH_DBG` | **0 (off)** | `ggml-cuda.cu` | **diagnostic only — changes no math, no dispatch, and costs one static bool when unset.** `=1` prints once per distinct `(fmt_up, fmt_gate)` pair the fused MoE drivers actually accept (`PXA_PXQ_DISPATCH: <site> fmt_up=… fmt_gate=…`, 3=PXQ4 4=PXQ4HQ 5=PXQ2 6=PXQ3 7=PXQ6R 8=PXQ1). It exists so that "the new tier reached the fused path" is a printed fact instead of an inference from a speed delta — a silent decline to the dequant→cuBLAS fallback looks exactly like a slow kernel |
| `PXA_PXQ6R_ANCHOR_FIT` / `PXA_PXQ6R_SUB` | — | `pxq6.cuh` | PXQ6R tier lab knobs (anchor-fit strategy / sub-scale LUT override) |

### 2026-07-30 late — sampler blast radius + the checkpoint flags documented

| var / flag | default | what it does |
|---|---|---|
| `PXA_SAMPLE_SOFTFAIL_v1` (behavior, no env to enable) / `PXA_SAMPLE_ABORT` | soft-fail ON; `PXA_SAMPLE_ABORT=1` restores the old fatal | An unsampleable probability vector (NaN-cascade from garbage logits) used to `GGML_ABORT` the WHOLE server at `llama-sampling.cpp` — one poisoned slot killing every co-resident generation, the same blast-radius class as the old ret=-3 unwind. Observed live 2026-07-30 on a multi-GPU bare-metal host serving an `hy_v3` model (checkpoint-restore → full-reprocess → "Failed to sample token" → abort → the abort-path fork hang). Now: forensic dump on first occurrence (`probabilities.txt` kept), loud `PXA_SAMPLE_SOFTFAIL_v1` log, deterministic fallback token (max finite logit, else first candidate) — the request may degenerate, the server survives |
| `--ctx-checkpoints N` | **32** per slot | max recurrent/KV context checkpoints per slot (0 disables checkpointing). ⚠ Load-bearing operational knowledge: **`--ctx-checkpoints 0` was the H2 mitigation** for the recurrent-checkpoint contamination on `qwen35moe` hybrids **on pre-2026-07-28 binaries**; the canonical tree carries `PXA_CKPT_HYBRID_ROLLBACK_v1` which fixes the contamination properly (rollback gate on `seq_pos_max`, checkpoint match on `cur.pos_max`). Any binary WITHOUT that fix (i.e. anything built before 2026-07-28) must either serve `cache_prompt:false` client-side or be upgraded — the 2026-07-30 fatal chain on a multi-GPU bare-metal host started exactly here. Note some older builds do not have this flag at all: check `--help` before relying on it |
| `--ctx-checkpoints-interval N` | 512 | min tokens between checkpoints |
| `--ctx-checkpoints-tolerance N` | 5 | tokens-before-full-prompt at which a checkpoint is created |
| `--ctx-checkpoints-eviction NAME` | variance | eviction strategy: `fifo`, `variance`, `auto`(=variance); variance keeps uniform positional coverage |

### 2026-07-30 latest — the "sm_61 arith flip" was the PXQ repetition guard (dead-end + fix, fully measured)

Verification class: **root-caused with both-direction A/B; fix verified on the shipped binary.**
Cell for every arm: Laguna-S-2.1 **PXQ4-core**, 4×P100 + 1080 Ti, `-ts 8,8,8,8,3`, `-c 8192
-np 1 -b 2048 -ub 2048 -fa on` f16 KV `--jinja`, `/v1/chat/completions` temp 0 / top_k 1 /
`cache_prompt:false`, fresh server per arm, 3 reps per arm (all arms ×3 identical).

| arm | env | 4183×391 | verdict |
|---|---|---|---|
| B | none (DEFAULT) | **1,635,553 ✓** | control |
| C | `PXA_ENHANCE=1` +MMVQ+graphs-off | 1,635,593 ✗ | repro of the "flip" |
| D | C + `PXA_PXQ_INT8_PREFILL=0` | 1,635,593 ✗ | **INT8_PREFILL exonerated** |
| E1 | `PXA_ENHANCE=1` alone | 1,635,593 ✗ | ENHANCE is the carrier |
| E2 | `GGML_CUDA_DISABLE_GRAPHS=1` alone | 1,635,553 ✓ | null |
| E3 | `PXA_PXQ_MMVQ=1` alone | 1,635,553 ✓ | null |
| F1 | ENHANCE + `PXA_SPEC_RELAXED=0` | 1,635,593 ✗ | exonerated |
| F2 | ENHANCE + `PXA_AUTO_SAMPLERS=0` | 1,635,593 ✗ | exonerated |
| G1 | ENHANCE + `PXA_REP_GUARD=0` | **1,635,553 ✓** | **conviction (direction 1)** |
| G2 | DEFAULT + `PXA_REP_GUARD=1` | 1,635,593 ✗ | **conviction (direction 2)** |
| H1 | fixed binary, ENHANCE | **1,635,553 ✓** | fix verified |
| H2 | fixed binary, `PXA_REP_GUARD=1` | 1,635,593 ✗ + `[FORCED]` log | force path intact |

Also measured: the ORIGINAL failing 7-card cell (2×V100+4×P100+1080Ti, `-ts 1,1,8,8,8,8,3`,
same file/env as the incident incl. graphs-off) answered **correct ×3** when served standalone
— the incident's 4/5 ran co-resident with a second model server on the V100s; with the guard armed
the flip is near-tie/cell-fragile, which is exactly why it masqueraded as a numerics bug.
**Dead ends recorded so nobody re-chases:** sm_61/1080Ti DP4A numerics, INT8_PREFILL,
MMVQ, CUDA-graphs-off, and 7-card topology are ALL exonerated for this symptom.

### 2026-07-30 — MTP draft-gen worst-case reserve clamp (upstream #2181 port, gated)

| var | default | where | what it does |
|---|---|---|---|
| `PXA_MTP_DRAFT_RESERVE_CLAMP` | **0 (off)** | `src/llama.cpp` (`llama_kv_cache_update_internal`) | `=1`: when a deferred K-shift/s-copy/defrag fires during an MTP **draft-gen** decode (`mtp_op_type == MTP_OP_DRAFT_GEN`), the post-update worst-case graph re-reserve is clamped from `min(n_ctx, n_ubatch)` tokens to **1** — a draft-gen step consumes exactly one token + one hidden-state vector, so the full-width MTP-graph reserve is a pure transient compute-buffer spike. Port of upstream ik_llama.cpp #2181 (`de55d9e2`). **Verification class: code-audited no-op in the current server, either state** — the trigger is structurally unreachable here: `PXA_HYBRID_CTX_SHIFT_v3` never marks the MTP companion's KV flags (companion shift mirroring was replaced by `seq_rm` + rebuild), the server only defrags the target ctx, and nothing `seq_cp`s the companion. **[EST] no measurable effect in the live config**; landed as defense-in-depth/upstream parity for paths that re-mark companion flags (the imatrix MTP draft loop, future arch ports, any revival of companion shift mirroring). Safe by construction: an under-reserve auto-reallocs on the next larger graph (ggml-alloc), and reservation size never touches computed values — no output-path effect. Not in ENHANCE. TU compile-verified on the canonical tree; no runtime A/B exists because no reachable cell exercises it |

### 2026-07-30 — Qwen-VL image preprocessing: stbir resizers + bicubic (upstream #1967/#1969 port, gated)

| var | default | where | what it does |
|---|---|---|---|
| `PXA_MTMD_STBIR` | **0 (off)** | `examples/mtmd/clip.cpp` (`img_tool` + the Qwen-VL preprocess case) | `=1` (value-tested; `=0` verified OFF) enables BOTH halves of the upstream image-preprocessing correction as one switch: (a) upstream #1967 (`574f22b3`) — `img_tool::resize_bilinear/resize_bicubic` route to the vendored `stb_image_resize2.h` v2.18 (`STBIR_FILTER_TRIANGLE` / `STBIR_FILTER_CATMULLROM`), which *filters the full support on downscale* where the legacy handwritten loops point-sample a 2×2/4×4 neighborhood and alias; (b) upstream #1969 (`19f08160`) — the QWEN2VL/QWEN25VL/QWEN3VL/GEMMA4V preprocess case selects bicubic instead of bilinear, matching the HF reference processors (PIL/torchvision bicubic = Catmull-Rom). Deliberately coupled: upstream's "bicubic" lands on the stbir filtered Catmull-Rom — #1969 alone on the legacy point-sampled bicubic would add aliasing+overshoot on downscale instead of matching the reference. When ON the resizer swap affects EVERY projector that resizes (minicpmv/idefics3/gemma3/internvl/…); the algo change affects only the Qwen-VL family + Gemma4V case. Gate OFF = byte-identical legacy behavior (legacy loop bodies untouched, algo falls back to bilinear). **Verification: gate-off bit-exact by construction; gate-on is a deliberate pixel change** — vision output is NOT byte-comparable across resizers, compare task success. Functional: Qwen2.5-VL-7B-Instruct Q4_K_M + f16 mmproj, `llama-mtmd-cli` temp 0, the bundled newspaper test image is read correctly in both gate states. **Measured cost (single-thread, in-process around the `clip_image_preprocess` resize; CPU-only static Release, example targets at default flags = SSE2 baseline, no `-mavx2` — upstream CMake parity; dual-Xeon host; RGB-noise BMP; target 532×364 via `--image-max-tokens 256` qwen25vl smart-resize):** 800×560 input → legacy 4.86 ms vs stbir **2.80 ms** (~1.7× faster with the heavier filter; ×2 runs); 6000×4000 input → legacy 5.60 ms vs stbir **140.9 ms** (median of 3). ⚠ Read the big-image number correctly: upstream bills this port as a SIMD *speedup*, but on heavy downscale stbir is ~25× SLOWER — legacy's flat ~5 ms is the aliasing bug itself (point-sampling reads <1% of input pixels at 11×; stbir reads all 24 MP, which is the correctness being bought). ~141 ms once per large image is noise against the vision-tower encode of the same image. `-mavx2` on the mtmd target is an unexplored follow-up lever (stbir carries AVX2 paths). Intended first consumer: the Qwen2.5-VL serving deployment — flip via its env only after a task-success (OCR/read) A/B on that service. Not in ENHANCE |

### 2026-08-02 — DeepSeek-V4 DSA sparse attention (head-512 flash attention on sm_70)

⚠ **NO MEASURED EFFECT YET. Both rows below are UNMEASURED and must not be quoted as a
speedup.** They are recorded here because a shipping gated flag has to land with its row,
and because leaving them undocumented is how `PXA_PXQ_KV` ended up with a fabricated
measured effect for a string that existed nowhere in the source. Verification so far is
compile + correctness only; the throughput A/B has not been run.

**What the honest A/B has to answer, stated before the numbers exist.** DSA lives only on
the `-fa on` path. Under `-fa on` on sm_70 every DeepSeek-V4 attention node that DSA does
*not* take still runs on the CPU backend (head 512 has no other CUDA kernel), while
`-fa off` already puts 100% of attention on the GPU densely and is measured 2.5x faster at
prefill (114.83 -> 287.42) and +40% at decode (11.05 -> 15.50). So the baseline to beat is
**`-fa off`, not `-fa on`**, and `-fa on` + DSA only has a route to winning if DSA covers
essentially *every* attention layer. That is what `PXA_DSA_MIN_RATIO` controls. If DSA
loses to `-fa off`, the correct outcome is to keep `-fa off` and say so.

**Numerics.** `build_dsv4_attn_mha()` calls `ggml_flash_attn_ext_set_prec(GGML_PREC_F32)`
on every DeepSeek-V4 attention node, and that request lives in `op_params[3]`. The
upstream ik kernel never reads it: both of its GEMMs were `cublasHgemm` (F16 accumulate)
and it stored the score matrix in F16 as well, so a PREC_F32 request was silently answered
in F16 — a different answer, not a slower one, and worst exactly where the request was
aimed (the QK contraction is 512 deep and the PV reduction grows with context). Here both
GEMMs accumulate in F32 (`CUBLAS_COMPUTE_32F`, the same idiom `ggml-cuda.cu` uses for
`GGML_PREC_F32` mul_mat), the score matrix is F32, and the PV GEMM writes `dst` directly
in F32. This is unconditional — F32 accumulation also satisfies `GGML_PREC_DEFAULT`, which
imposes no requirement — so there is no second numeric path to keep correct.

| var | default | where | what it does |
|---|---|---|---|
| `PXA_DSA_ATTN` | **0 (OFF)** | `ggml-cuda/dsa_attn.cu` (`ggml_cuda_dsa_attn_supported`), mirrored in `build_deepseek4.cpp` | Master switch for the sparse attention path; set `=1` to enable, anything else leaves it off. A `FLASH_ATTN_EXT` node carrying an index list in `src[5]` (from `GGML_OP_MASK_TO_IDX`) attends only the KV rows each query can see, via cuBLAS GEMM + plain CUDA kernels — no wmma/mma/cp.async and no `__CUDA_ARCH__` branches, so it is the only CUDA attention path that accepts head dim 512 on Volta. Read through the single predicate that both the dispatcher and the scheduler gate call, so it cannot half-apply — which is what makes a same-binary A/B possible; the graph-builder mirror shares the same default so `off` reproduces the **pre-port graph**, not merely a disabled kernel. **⚠ DEFAULT OFF ON PURPOSE.** Where DSA engages it is not bit-identical to the path it replaces, and it has no end-to-end measurement — no perplexity, no KL against the pre-port binary, no throughput number. An opt-in switch is what an unmeasured behaviour change is allowed to be. **Effect: UNMEASURED.** Verification class: **G3 at best, not bit-exact.** ⚠ **An on-vs-off comparison at low fill is VACUOUS**: below the engagement threshold the compressed mask is narrower than the index list, `worth_it` is false, `src[5]` is never set, and both arms run the identical pre-port path — so they are byte-identical by construction and the comparison measures the engine against itself. Any A/B must prove engagement **in the same run** (`GGML_SCHED_DEBUG=2` showing `FLASH_ATTN` on CUDA) or it is not evidence. |
| `PXA_DSA_MIN_RATIO` | **4** | `ggml-cuda/dsa_attn.cu`, mirrored in `build_deepseek4.cpp` | Minimum `n_kv : index-width` ratio before the sparse path is taken at prefill (batch > 16; decode only requires the list to be narrower than `n_kv`). ik hardcodes 4, measured on an RTX 3090. It is a throughput trade — gather cost against GEMM saved — not a correctness bound, and the crossover is hardware-specific, so it is exposed rather than inherited. **This is the knob that decides layer coverage.** At the default, on the 43-layer DS4-Flash checkpoint (`n_swa` 128, `indexer_top_k` 512, `compress_ratios` alternating 4/128 → 21 CSA + 20 HCA + 2 raw), prefill engagement is CSA-only and only above roughly 9k fill; decode engages far more readily. Lowering it toward 1 pulls HCA and raw layers in as well, which is the only configuration in which `-fa on` could plausibly beat `-fa off`. The graph builder mirrors the same value so it does not emit an index list the kernel would decline; **the kernel's copy is authoritative** and a disagreement wastes a node rather than miscomputing. **Effect: UNMEASURED at every value, including the default.** Moot unless `PXA_DSA_ATTN=1`. |

### 2026-07-30 — crash/correctness hygiene: nl-less-vocab sampling OOB, jinja for-loop scope leak, boolean-flag argv swallow (upstream #2188/#2018/#2129 port)

| var | default | where | what it does |
|---|---|---|---|
| *(no env — always on)* nl-token OOB guard | — | `common/sampling.cpp` (`llama_sampling_prepare_impl`) | Port of upstream ik #2188 (`8be93884`). The repetition-penalty block read `logits[llama_token_nl(...)]` unconditionally; on a vocab whose tokenizer maps `"\n"` to zero tokens the loader's fallback chain (`src/llama-vocab.cpp`: `linefeed_id = special_pad_id`, itself `LLAMA_TOKEN_NULL` on Falcon3-class BPE) yields id −1, making that `logits[-1]` — an OOB read one float before the logit row, a configuration-dependent segfault. Reachable from every token sampled through `common_sampler_sample` → `llama_sampling_prepare` (the read fires whenever any prev tokens exist, even with all penalties at 1.0, and `PXA_REP_GUARD`/ENHANCE arms real penalties on PXQ1 content). Now the token is fetched once and both the read and the `penalize_nl` restore are skipped at `LLAMA_TOKEN_NULL`. **Cannot change output on a vocab with a real newline token** (byte-identical block; upstream additionally measured 26 greedy comparisons across 8 models unchanged, and reproduced segfault→HTTP-200 on 4×P100 `-ngl 99` — our hardware class). Ungated: the old path is UB, there is no legacy behavior worth keeping. Relevance here: PXQ-UNIVERSAL quantizes arbitrary community models — exactly the odd-vocab traffic that trips this. No local crash repro was re-run (no nl-less-vocab model was available locally); verification class: **structural + suite-identical + upstream repro on our hardware class** |
| `PXA_JINJA_LEGACY_LOOP_SCOPE` | **0 (fix active)** | `common/jinja/runtime.cpp` (`for_statement::execute_impl`) | Port of upstream ik #2018 (`997b289d`). Each `{% for %}` iteration now executes in a fresh child scope, so a conditionally-skipped `{% set %}` no longer leaks the previous iteration's value forward (upstream's field case: a tool-call turn with reasoning followed by one without duplicated the stale reasoning into the second turn — wrong prompt, degraded output; goes straight at the tool-calling-reliability hard requirement). Matches standard Jinja2; cross-iteration accumulation still works via `namespace()` (mutations ride the shared object, which is why correctly-authored Qwen-family templates are unaffected). `=1` restores the old shared-scope behavior (value-tested; unset or `=0` = fixed). **Verified 2026-07-30 (CPU-only Release, `pxq-build:tools` container, worktree @ `9d564d7`):** ported upstream unit test passes (test-jinja 306 tests / 1427 assertions / 0 failures); with `=1` exactly that one test fails, rendering `[a][a][c]` — the leak, byte-for-byte the legacy semantics — and the other 305 tests are byte-identical; with `=0` output is byte-identical to unset (value-test proven). `test-chat-template` and `test-chat-auto-parser` outputs are **byte-identical before/after the port** (their pre-existing failures at `9d564d7` — the GLM whitespace-trim assert, 12/57 auto-parser cases — are unchanged, i.e. not loop-scope-related). No perf claim: cost is one flat scope copy per loop iteration, template-render only |
| *(no env — always on)* `--skip-chat-parsing` / `--no-prefill-assistant` argv fix | — | `common/common.cpp` (`gpt_params_find_arg`) | Port of upstream ik #2129 (`3e76852b`). Both boolean flags carried a stray `CHECK_ARG`, silently consuming the NEXT command-line argument (`--skip-chat-parsing --version` ate `--version` and fell through to loading the default model) and dying as `invalid parameter` when the flag was last on the line. Removed; the usage strings were already boolean-style so no help change. **Verified:** `--skip-chat-parsing --version` / `--no-prefill-assistant --version` now print the version and exit 0 (before: rc=1 default-model-load failure); each flag as the last argument is accepted (before: usage dump). Ungated: pure argv correctness |

## Updates — 2026-08-09 — levers added since the previous release

Every gated flag that ships in this release and did not have a row above. Defaults are unchanged
behaviour unless the row says otherwise.

| lever | default | what it does | measured |
|---|---|---|---|
| `PXA_AUTO_SPEC` | unset → arms at the ENHANCE level, which is the default; `=1` forces on at any level, `=0` forces off, never under `PXA_REFERENCE` | Arms the measured-best **self**-speculation drafter for the model family, so ENHANCE users get it without knowing the flag exists. One row ships: `qwen35moe` → `--spec-type ngram-mod:n_max=4,n_min=2`. An explicit `--spec-type`/`--spec-stage` or a draft model (`-md`) always wins — only absence is filled. Arms through the same parser `--spec-type` uses, so the auto path cannot drift from the flag path. Families with no measured row are left alone and say so on stderr | **First-request decode (cold n-gram table): code 24.44 → 30.05 t/s (+23.0%), prose 23.69 → 24.78 (+4.6%); prefill neutral (262.63 → 259.16).** Cell: 4×P100, Qwen3.5-122B-A10B (community "abliterated" abliteration) PXQU48-core, `-c 32768 -b 512 -ub 512 -fa on`, ~14–15k fill, control in the same run. ⚠ **Do not quote the median-of-3 figures (+38.8%/+35.7%)** — the bench replays one prompt three times, so reps 2–3 hit an already-warm n-gram table (prose acceptance 1.000 is the tell). Warm numbers are real for multi-turn chat and agentic re-reads, but they are an upper bound. ⚠ Gain tracks how repetitive the OUTPUT is: high on code/tool traffic, much lower on prose. **No `mtp` row ships**: on the same model MTP measured −8.6% (n_max=1) and −29.8% (n_max=2) despite 0.800 acceptance — a drafter costing a transformer forward per cycle cannot pay on bandwidth-bound sparse-MoE decode, where a k-token verify batch routes to up to k×8 distinct experts |
| `PXA_PXQ_REDUCE_BLK` | **256** (byte-identical to prior behaviour); accepts 32/64/128/256 | `blockDim.x` of both PXQ reducers. They are pure per-thread row maps, so this is **bit-identical at any block size** — it only changes how many SMs the launch reaches (64 spreads the dense reduce from 12 blocks to 48 on a 56-SM chip) | **MEASURED NULL:** 64 gave +0.22% on 122B/4×P100 while the two bracketing controls differed by 0.35% — below the noise floor, engagement proven by banner in both arms. The premise (1.87 ms/token in reduce) came from a `PXQ6_CANON_CMAX=16` profile; CMAX=4 had already recovered it. Kept as a lever so a future reduce-shape change can re-test with one env var |
| `PXA_PXQ_TOKBATCH` | **0 (off)**; 2 or 3 valid (1 is meaningless — that *is* the per-token kernel; >3 is register-unsafe). Invalid values clamp to off with a warning | Token-batched gateup verify: on an `Ny > 1` verify batch, folds TOK consecutive tokens into one block so the expert weight walk is shared across the group. Workspace sizing and the canonical reducer are identical to the per-token arm, so **output is bit-identical** — only launch shape and weight-read temporal locality change. Every decline (lever off, `Ny == 1`, no instantiation, smem cap) falls through to the per-token launch unchanged | ⚠ **No A/B on record.** Requires the gen split (`PXA_PXQ6_KSPLIT_GEN`, default S=4 on sm_60/70): with the gen split off the lever never engages and the `ENGAGED` banner does not print — **such an A/B is VOID, not null** |
| `PXA_FA_GQA_PACK` | **0 (off)**; NH ∈ {2,4,8} | sm_60 D=256 vec flash-attention that hands one block NH Q heads sharing a KV head, so the K row is fetched once and dotted against NH query vectors. Load traffic and load-instruction count both fall by NH at unchanged FMA count. Gated on `ncols == 1`, `K == V == f16`, `Dk == Dv == 256`, no logit softcap, and NH dividing both the head count and the gqa_ratio; anything else keeps the stock kernel | Stock `flash_attn_vec_ext_f32` re-reads the whole K/V cache once per Q head — on qwen35moe-122B-A10B (32 Q heads over 2 KV heads) an nsys decode capture put it at 4.82 ms of a 36.14 ms token (13.4%), issuing ~291 MB of K+V loads per layer against 18.2 MB of unique cache. Measured 122B PXQU48-core, 4×P100, `-fa on`, fill 8881, n=3 medians, control in the same run: **NH=4 +6.9%**, NH=8 +6.9% (occupancy cancels the extra traffic cut); at fill 15 unchanged (31.83 → 31.94), prefill unchanged (300.5 t/s every arm — prefill takes `cols_per_block > 1`). ⚠ **Not bit-exact** vs the shipped `PXA_FA_VEC_ILP` V pass (1-accumulator vs 4-accumulator order). **Default stays 0 until a coherence + ppl gate is recorded** |
| `PXA_FA_GQA_QSMEM` | **0 (off)** | Stages the packed GQA query rows in shared memory, laid out `[j][i][lane]` so consecutive lanes hit consecutive banks. Q is what costs the registers (NH×QPL float2 per lane = 64 registers at NH=8); staging it gives 95 reg / 19456 B at NH=8 (2 blocks/SM) and 72 reg / 10240 B at NH=4 (3 blocks/SM) | Same rig/protocol as `PXA_FA_GQA_PACK`, split target 160: NH=4 registers 30.30 → **NH=4 shared 30.63 t/s (+1.1%)**; NH=8 shared 29.59 (−2.3%). The occupancy×traffic product model predicted NH=8+shared would win and it did not — at NH=8 the inner dot issues 32 shared loads per K row per warp and shared-load throughput simply replaces global-load throughput as the bound. NH=8 keeps the twin so that boundary stays reproducible. Inherits the same not-bit-exact caveat |

**Build-time constants** (not env flags — flipping them is a re-baselining event, which is exactly why
they are compile-time):

| constant | default | what it does | measured |
|---|---|---|---|
| `PXQ_CANON_V2` | **0 (off)** — build with `-DPXQ_CANON_V2=1` to engage | Accumulation **shape**, not sourcing. `pxq6_dot32` accumulated as `t += a0*x0 + a1*x1`, which ptxas emits as FMUL + FFMA + FADD — 3 float ops and 3 roundings per pair, 48 per 32 weights (SASS-verified on the shipped sm_60 binary). ptxas cannot fix this itself: `--fmad=true` CONTRACTS but never REASSOCIATES, and nvcc never sets LLVM `reassoc` — the same property `PXQ_CANON_v1`'s split==unsplit bit-exactness already relies on. The chained form is 2 FFMA: **2 roundings instead of 3, both products computed exactly** (2-of-2 rather than 1-of-2) — a precision *improvement*, not a trade. Every variant flips together, so split == unsplit still holds by construction (same pattern as `PXQ6_CANON_CMAX`) | **+3.5% clean decode, +5.7% / +6.0% drafted.** Cell: 122B-A10B PXQU48-core, 4×P100, fill 14259, n=3, bracketed controls — no drafter 30.31 → 31.36; with the n-gram drafter 35.86/35.93 → 37.93 cold and 38.85 → 41.20 warm. The win **amplifies** under speculation rather than decaying: a drafted step carries `1+n_draft` rows on `blockIdx.z` through the same codec kernel, so instructions removed from the inner loop are saved ~k times over — codec instruction cuts and speculation multiply, they do not overlap. Engagement proven in SASS on the fired instantiation: FMUL 108→44, FFMA 68→132, FADD 120→56 — exactly −64/+64/−64, the designed signature. ⚠ Changes the fp32 fold, so it is a **re-baselining event**: regenerate every control on the engaged binary before comparing anything. ⚠ **Token output is NOT a valid instrument for this class of change** — greedy decoding only diverges when a rounding difference flips an argmax, so identical temp-0 output is EXPECTED and is a quality result, not evidence of a null arm |
| `PXQ6_CANON_CMAX` | **16** | The canonical chunk count, which is *also* the split workspace row width: every split node writes `NFIX*KSEG*R` floats of raw partials and the reducer reads them all back, whatever S the driver chose. Stays shape-only, so `split == unsplit` bit-exactness holds by construction at any value | nsys decode capture (122B PXQU48-core, 4×P100, fill 8881, 128 tokens, GQA pack NH=4 + split target 160): the reduce kernels alone are 1.87 ms/token of ~32.6 (`k_pxq_mmv_reduce_s` 1.157, `k_pxq6_gateup_reduce_gen` 0.708), with a matching write side inside the mmv kernels — and the driver never selects S > 4 on any node of this model (attn_gate 2; ssm_out / attn_output / ffn_down_shexp 4; MoE gateup 4). At CMAX 16, three quarters of every workspace row is written and read for nothing; **CMAX 4 leaves every grid geometry of this model unchanged while cutting the workspace 2–4× per node.** ⚠ Default stays 16: it changes the canonical fp32 fold (fewer, longer chunks) so it is a re-baselining event, and **the evidence is sm_60 only** — on sm_70 the split target is 16×nsm and the selected S differs |
| `PXQ_GU_MINBLK` | **0 (off)** | `__launch_bounds__` min-blocks-per-SM pin on the *fired* gateup instantiations (TAB mode 0 × VECX=1). `PXQ_CANON_v2` pushed them 64 → 74/75 registers, crossing the 65-register cliff (65536/(256×4) = 64 exactly) and costing 25% of resident warps on 40 of 48 layers; asking for min 4 blocks brings them back to ≤64. Only the fired shape is pinned — REFERENCE-only VECX=0 variants keep their natural allocation | **MEASURED NULL: +0.14% median.** Distinct from the killed cap-forcing experiment (that forced 36/32 registers, 10–45 below natural, and spilled: −27.7%); this asks −10 from 74, and any `LOCAL>0` in res-usage aborts the change |

### Levers previously read by the engine but missing from this table

Closing an inventory gap: these are all read via `getenv` in shipped source and had no row.

| lever | default | what it does | measured |
|---|---|---|---|
| `PXA_FA_TILE256` | **ON** (`=0` reverts to the vec route) | On pre-Volta cards the D=256 head had **no** tile/mma prefill kernel, so `Q->ne[0] == 256` forced the single-column VEC kernel at any batch size — every query column re-streamed the whole KV extent with no tile reuse. This routes batch>8 D=256 attention to the tile-f16 (ncols=16) kernel, restoring KQ-tile reuse. **Decode (`ne1 <= 8`) is untouched.** Prints its state at startup | Profiled on a 122B-A10B (qwen35moe, head 256), 4×P100, fill 8881: `flash_attn_vec_ext_f16` was **52.7% of prefill GPU time** (60 launches × 281 ms avg) before the change. The routing change is the fix for that profile; a paired before/after throughput A/B is not on record |
| `PXA_FA_TILE_VOLTA` | **off** (`=1` routes sm_70 to the tile kernel) | The tile flash-attention kernels already ship and are already dispatched — but only to cards without fp16 mma (sm_60). sm_70 has working WMMA, so Volta always took the WMMA path and tile-vs-WMMA on Volta could not be measured at all. This routes sm_70 **only** (gated on fp16 mma present and Turing+ mma absent) down the same tile path the sm_60 cards take; if tile does not support the shape it falls through to the normal selector rather than failing. With the env unset the dispatcher is byte-identical to before, so the WMMA path and `PXA_FA_MASK_SKIP_v1` remain the untouched default. Logs a banner on first dispatch when ON | **+6.9% prefill.** Dense 30B PXQ4, 2×V100, `-ngl 99 -sm layer -r 3 -p 2048 -n 128 -ub 2048 -fa 1`, OFF/ON/OFF so drift is visible: OFF `pp2048 1092.95 ± 1.02`, ON `1164.23 ± 4.22`, OFF-repeat `1085.44 ± 6.56`. Intervals are disjoint and the two OFF arms agree to 0.7%. **Decode unchanged** (`tg128` 35.60 / 35.60 / 35.74) — batch-1 never reaches this branch. Engagement was confirmed inside the same run by the banner, present only in the ON arm. Default OFF pending an output-fidelity check: tile accumulates differently from WMMA, so bit-identity is neither expected nor measured |
| `PXA_F32PREC_F16GEMM` | **off** (`=1` arms it) | A dense GEMM carrying `GGML_PREC_F32` fell through to a dequant + fp32 SGEMM instead of the f16-input / fp32-accumulate tensor-core path. This routes those GEMMs through GemmEx with fp32 accumulation, which is what the precision request actually asks for. | **+104% warm prefill** on a community k-quant file: 566 -> 1152 t/s, dense 30B Q4_K_XL, 2x V100, `pp2048 -ub 1024`. The affected kernel was 50.7% of that prefill wall at 20.1 ms/call over 408 calls. A PXQ4 file of the same model measures **unchanged** (1235 vs 1240) because it has no PREC_F32 cuBLAS ops -- so this lever is a k-quant remedy, not a PXQ one. |
| `PXA_PXQ_MMVQ_FUGSPLIT` | **off** (`=1` arms it) | Dense PXQ up/gate decode runs as one fused kernel whose register pressure caps occupancy at 31%. This splits it into two plain MMVQ launches plus a GLU epilogue. Bit-identical output. | **+14.7% decode** on sm_70: dense 30B PXQ4 `tg128` 35.80 -> 41.06/41.10 (n=3 reps x2 blocks, +/-0.01), 2x V100. That closes a 17% deficit against the equivalent community k-quant (41.99 on the same cards) to **-2.1%**. Costs about **-0.9% prefill**, so pair it with the attention lever rather than using it alone if prefill matters. |
| `PXA_DENSE_FUG_PREC_KQ` | **off** (`=1` arms it) | The fused up/gate node stores a unary-op id in `op_params[0]`, which the generic dense branch misread as a precision field -- the third instance of that alias. This sanitizes it so the precision request is read correctly. | Measured on a dense 30B k-quant, 2x V100, `pp2048 -ub 1024`. It is the *correct* fix for the alias, but `PXA_F32PREC_F16GEMM` above is the faster remedy for the same symptom, which is why both ship and both default off. |
| `PXA_PXQ2_V3` | **off** (`=1` selects the v3 book) | Selects a refitted PXQ2 codebook. The shipped PXQ2 book reuses a sub-block table frozen for a different book, which clips each row's peak weights; v3 refits against the model family. | Gated off by default because the refit is only a partial remedy on paper and a full requant was not run. Treat as experimental. |
| `PXA_PXQ_CEIL_V2` | **off** (`=1` arms it) | Second-generation reconstruction ceiling for the PXQ codecs, read by both the CUDA and CPU paths and by the loader so a file and its runtime agree. | Ships off; no throughput claim is made for it. It changes reconstruction, so any use must be fidelity-checked against the tier's own gate rather than assumed neutral. |
| `PXA_PXQ_KQW` | **off** | Quantizer-side knob affecting K/Q weight handling when writing a PXQ file. | Quantize-time only -- it cannot change the behaviour of an already-written file. No throughput claim; ships off. |
| `PXA_PXQ_SMIN` | **off** | Quantizer-side minimum-scale floor for the PXQ1 encoder. | Quantize-time only. No throughput claim; ships off. |
| `PXA_ALLOW_GRAPH_SPLIT_HYBRID` | **off — and this is a correctness guard, not a performance lever** | ⚠ **Footgun. `=1` bypasses a measured-defect guard and can produce fully degenerate output.** `-sm graph` on the DeltaNet hybrid arches (`qwen35moe`, `qwen3next`, `qwen35`) emits `!!!!…` at temp 0; the engine detects this and silently falls back to `-sm layer`. This env removes the fallback, for debugging only | Root-caused 2026-08-04: the DeltaNet head split is arithmetically exact (per-device partials sum to the `-sm layer` reference), but the **cross-device all-reduce never reaches the consumers** — each device normalises its own 1/n partial, so every device computes different router logits and picks a different top-8; NaN by layer 3, first token already `!`. `PXA_REDUCE_NCCL=0` and `PXA_REPLICATE_RECURRENT=1` are both byte-identically degenerate, so it is not the reduce transport or the recurrent carry. Not worth fixing at present: on 4×P100 `-sm graph` is **+64% prefill (255.6 → 419.0 t/s) but −17% decode (26.73 → 22.17)**, n=3 medians, 14259-token prompt, one binary/one run — consistent with 2×V100 (−31%) and an 8×V100 node (−4.6× decode). Fixing it buys prefill only, and only if the engine gains a per-phase split mode |
| `PXA_REDUCE_NCCL` | **ON** where NCCL is available (`=0` takes the non-NCCL reduce route) | Transport selection for the cross-device reduce | Used as a bisection control in the graph-split investigation above; **both routes were byte-identically degenerate**, which is what ruled the transport out. No throughput A/B on record |
| `PXA_REPLICATE_RECURRENT` | off (presence-tested) | Replicates the recurrent state across devices instead of splitting it | Same investigation: **measured null as a fix** — byte-identically degenerate. No throughput A/B |
| `PXA_PXQ_MOE_DOWN_SPLIT` | **0 (off)**; 1 = K1 kseg split, 2/4/8 = gen S | K-split for the fused MoE **down** projection, bit-exact. ⚠ Deliberately uses the MMV cap, not the gateup family's `PXQ6_GU_SPLIT_MAX`: at `K >= 2048` the GU cap changes the canonical fold (`canon_nfix(64,8) != canon_nfix(64,16)` → a silent bit-break), and the gen workspace row stride is a fixed `PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG` slots, so sizing with the GU cap would undersize it 2× → out-of-bounds writes | ⚠ **No A/B on record.** Prints an armed banner when engaged |
| `PXA_SHEXP_GATE_FUSE` | **off** | Folds the shared-expert gate GEMV into the `fused_mul_sigmoid` consumer. Bit-exact; unset or `0` reverts | ⚠ **No A/B on record.** Prints an ARMED banner when engaged |
| `PXA_SMALLN_GEMV` | **0 (off) everywhere; env-only, for interleaved A/Bs** | Scalar dequant-FMA GEMV route for small-n on pre-Volta | **MEASURED LOSS, kept as a documented dead end: −28% vs emulated-dp4a MMVQ.** Verdict: the sm_60 dense GEMV is **instruction-economy bound, not bandwidth bound** — mmvq's packed-q8_1 integer form wins even with dp4a emulated. A winning replacement must cut instructions per weight (half2 `__hfma2`, packed nibble decode), not merely avoid the int8 path |

**Diagnostics** (print-only; no effect on output, dispatch, or math — leave off in production):
`PXA_GEMV_DBG`, `PXA_GLUE_DBG`, `PXA_RDBG`, `PXA_ROUTER_FUSE_DBG`, `PXA_PXQ_DISPATCH_DBG`.
`PXA_ROUTER_FUSE_DBG=1` dumps per-mode fired/declined counters — **the ratio is the assertion**, since
a one-shot bool cannot distinguish "fired once" from "fires every token".

---

## Updates — 2026-08-16 — SWA flash-attention NaN fix + blast-radius levers

Four `PXA_*` vars shipped with the restore-then-truncate NaN fix and were missing from this file.
All measured on the 3-card reviewer cell (V100+2×P100, `-sm layer -ts 24,38,38 -c 262144 -b/-ub 2048
-ctk/-ctv f16 -fa on`, 30B MoE PXQ4, `n_swa = 2048`) unless a row says otherwise.

| var | default | what it does |
|---|---|---|
| `PXA_FA_SWA_SLICE` | **0 (slice OFF)** | `1` re-arms the old windowed SWA slice in `ggml-cuda/fattn.cu`. **Repro/benchmark only — it is UNSOUND with `np>1` or any slot reuse.** The slice assumed KV-cache *cell index* order equals *position* order, which holds only for one sequence laid into an empty cache. Otherwise `first = K->ne[1] - nton > 0` cuts the query's own cells out of the view, every sliced mask row is all `-inf`, and the fully-masked flash-attention output is NaN → **all logits non-finite**. |
| `PXA_FA_SWA_KEEP` | **1 (n_swa preserved)** | `0` re-zeroes `op_params[4]` (the old dispatch behaviour: full-range KV iteration). Default keeps `n_swa`, so the kernels' mask-driven `KV_min_max` scan — which reads the actual mask and is correct for **any** cell layout — bounds SWA decode iteration instead of the index-arithmetic slice. |
| `PXA_SWA_TRUNCATE_MIN_DISCARD` | `256` | server guard: minimum discarded tokens before a restore-then-truncate reuse is refused in favour of a clean full re-process. |
| `PXA_SOFTFAIL_MAX_CONSEC` | `16` | server breaker: consecutive sampler soft-fails on one slot before that generation is stopped. Bounds blast radius; does **not** restore correctness. |

`PXA_SWA_TRUNCATE_MAX_KEEP_FRAC` (default `0.5`) pairs with `MIN_DISCARD`: the guard fires only when
*both* `discarded > MIN_DISCARD` **and** `n_past < MAX_KEEP_FRAC × cache_size`.

### The slice is unreachable from `llama-perplexity` — gate this change with generations, not ppl/KL

Structural, and it decides the method. The slice engages only when

```
first = K->ne[1] − 256·ceil((max(256, Q->ne[1]) + n_swa)/256) > 0
```

In a perplexity run the whole chunk is one batch, so `Q->ne[1] = n_ctx` and
`nton = n_ctx + n_swa` rounded up, which is **always > `K->ne[1] = n_ctx`** ⇒ `first < 0` at **any**
`n_ctx`. It is a **decode**-path difference: at `Q->ne[1] = 1`, `nton = 2304`, so it engages once the
cache passes 2304 cells. A perplexity/KL run cannot exercise the changed code path at all — so
perplexity and KL-vs-parent are the *wrong* instrument here, independent of the della-corpus problem.

**Quality gate actually run (2026-08-16):** temp-0 greedy anchors, `seed 1234`, `-np 1`,
`cache_prompt:false`, 8 prompts of **12,389 tokens** each (cache far past the 2304-cell threshold, so
the slice *does* engage on the old arm), single sequence so the old path is in the regime where it
was sound. New (slice off, `n_swa` kept) vs old (`PXA_FA_SWA_SLICE=1`): **8/8 byte-identical
generations**, longest 364 chars. The change is output-identical wherever the old path was valid,
and it is the only one of the two that is correct at all under `np>1`.
*Coverage dropped:* 8 anchors, one cell, one model — not a corpus-wide KL. No Q8_0 parent exists on
disk for this lineage and `llama-perplexity` is not built in this tree, so a parent-relative KL was
not run.

### Measured before/after (same binary, same launch line, only the lever varies)

Adversarial workload = long prompt (6k–24k tok, topic rotating so the RAM cache fills with mutually
unrelated large states) immediately followed by a short unrelated prompt sharing only the chat
template, every 3rd cycle firing two shorts concurrently. Instance had already served real traffic.

| arm | slice | guard | breaker | correct | soft-fails | blast radius |
|---|---|---|---|---|---|---|
| **after (shipping)** | off | on | on(16) | **15/16** | **0** | — (53 guard fires, 140 truncating reuses) |
| **before (defect)** | **on** | off | off | 1/8-class | **101 in ~2 min** | one request ran to `p0=20476` **and still climbing** |
| breaker isolated | on | off | **on(1)** | 1/8 | 14 | **13 fires, every one `n_decoded=1`**; requests returned in 0.4–1.2 s |

The breaker row is the point of the breaker: same defect, same wrong answers, but a degenerate
generation is killed after **one** token instead of holding a slot to context-fill — which under
`-sm layer` is what starves every co-resident generation on the same cards.

---

## Updates — 2026-08-16 — split-mode, card-count and SWA_KEEP on the reviewer cell

Cell unless a row says otherwise: **Rig-A reviewer instance**, `muse-glimmer` 30B **dense** PXQ4 +
vision projector, build-fugv60, cards 4,1,6 (V100 sm_70 + 2×P100 sm_60), all pairs **PHB, no
NVLink/P2P**, `-ngl 99 -sm layer -ts 24,38,38 -c 262144 -b/-ub 2048 -ctk/-ctv f16 -np 2 -t 16
-fa on`, `PXA_ENHANCE=1`, `GGML_CUDA_NO_PINNED=1`. Single-stream measurement, prompt trimmed to an
EXACT token count, temp 0 / top_k 1 / seed 1234 / `cache_prompt:false`, `n_predict 128`,
fresh server per arm, arms interleaved with repeated baselines.

### Doc-vs-runtime contradictions found (fix the doc, not the runtime)
1. The 2026-08-16 block above describes this instance as a **"30B MoE PXQ4"**. The runtime ledger says
   `class=dense experts=0(top0)`. It is **dense**. The keeper script likewise calls it
   "dense full-attention".
2. "full-attention" is also wrong: the GGUF carries
   `muse-glimmer.attention.sliding_window_pattern arr[bool,52] = [true,true,true,false,...]`,
   i.e. **39 sliding-window layers + 13 full-attention layers** (period 4), with `n_swa = 2048`.
   `n_swa_pattern` printing as `1` is just the hparams default ("all layers non-SWA") left
   untouched because muse-glimmer takes the per-layer-array branch, not the scalar-period branch —
   the printed scalar is **not** the truth for this arch. Read `swa_layers`, not `n_swa_pattern`.
3. KV is allocated **full-size for all 52 layers** (3328+5632+4352 = 13312 MiB at `-c 262144`,
   = 262144 × 52 layers × 1024 B/token/layer). There is no iSWA/`--swa-full` split-cache path in
   this tree (`grep swa_full` in common/ + src/llama-context.cpp: no hits), so the 39 SWA layers
   each carry a full-context cache they can never address beyond their 2048-token window.
   **Upper bound if an SWA-aware cache were added: ~9.8 GiB of VRAM freed on this instance**
   (13 full layers still need 3328 MiB; 39 SWA layers need ~163 MiB). Not attempted here — it is
   a code change, and it would trade away prompt-cache reuse (`apply_checkp ... lack of cache data
   (likely due to SWA)` is already in the log). Recorded as an opportunity, not a lever.

### Benchmarking this box: two traps that void results
- ⚠ **`-p 127.0.0.1:8299:8299` collides with a live `node` service.** Bench ports 8355/8356 are
  free. Never send `docker run`'s own stderr to /dev/null in a harness — a daemon-level port clash
  otherwise presents as a model-load failure.
- ⚠⚠ **Parking the instance keeper's cron entry does NOT stop the keeper.** dcron has the next run
  queued, so the keeper fired 44 s and 29 s after the park in two separate campaigns and recreated
  Rig-A onto the cards mid-run; every subsequent arm then "OOMed" and looked like a config verdict.
  **Hold the keeper's own lock instead:** `exec 8>/tmp/pxa-instance-keeper.lk; flock -x 8` for the
  whole campaign (the keeper's own `flock -n -o 9 || exit 0` then makes every invocation a no-op),
  and close the fd before handing the instances back. Interleave repeated baseline arms — the repeat
  contradicting the identical first arm is what exposes this class of fault at all.

### Measured verdicts — 2026-08-16 speed campaign

Protocol for every row: fresh server per arm; container flags matching the instance keeper exactly
(`--runtime=nvidia`, `GGML_CUDA_NO_PINNED=1`, `PXA_ENHANCE=1`); prompt trimmed to an EXACT token
count via /tokenize+/detokenize; temp 0, top_k 1, seed 1234, `cache_prompt:false`, `n_predict 128`;
3 reps/arm; **baseline arms repeated and interleaved** so drift is visible. Raw data:
`<local-path>`; narrative: `speed-findings.md`.

| lever | cell / fill | verdict | gate |
|---|---|---|---|
| **`-sm graph`** (and `-sm attn`) | Rig-A 3-card, fill 8192 | **NULL — it never engages.** +0.0%/+0.2% prefill, +0.0%/+0.1% decode, byte-identical output. `LLM_ARCH_MUSE_GLIMMER` is **absent from `k_supported` in `is_model_split_supported()` (`src/llama.cpp:3208`)**, so `src/llama.cpp:3516` warns "not supported for this model" and silently serves `-sm layer`. Engagement disproof: the graph arm prints `split_mode_graph_scheduling = 0` / `graph splits = 4`, byte-identical to the layer arm | n/a (no-op) |
| **`PXA_FA_SWA_KEEP`** | Rig-A 3-card, **fill 24576** | **CONFIRMED, large.** KEEP=1 (shipped default) **17.60 t/s** vs KEEP=0 **14.79 t/s** = **+19.0% decode**; spread 0.2-0.3%. Prefill untouched (260.9/260.8). **The NaN correctness fix was not a performance cost — it paid ~19% decode at large fill.** | decode-path -> temp-0 output. **byte-identical** (both sha `e9167a08…`), 3 reps x 128 tok. Bit-exact is also structurally expected: masked KV contributes exactly zero. Engagement = the +19% itself (no log marker exists) |
| **Rig-A 3 cards -> 2 cards** (NEW) | cards 4,1 vs 4,1,6; fill 8192 | **BIGGEST WIN.** `-ts 45,55 -c 131072`: **prefill 289.4 -> 378.4 (+30.7%)**, **decode 19.74 -> 22.33 (+13.1%)**, and it **frees a whole P100 (card 6)**. Ladder in dev0's share: 36,64 +16.9/+8.2; 40,60 +22.4/+10.0; **45,55 +30.7/+13.1**. 40,60 repeat within 0.2%/0.4%. **COST: max context 262144 -> 131072** (65536/slot at np2; observed peak use ~25k). 50,50 untested, predicted OOM (dev0 also holds the 3670.81 MiB projector) | **byte-identical output vs the 3-card baseline** (sha `cb0b4a6d…`), 3 reps x 128 tok. Measured, not assumed — moving layers across an sm_70/sm_60 boundary can change kernels |
| **`--spec-type mtp:n_max=1`** on the single-P100 instance | that instance's card 5, fill 8352 | **NET LOSS: −6.5% decode.** MTP-on baselines 40.85/41.36/41.23 (mean 41.15) vs MTP-off **43.83**. With `GQA_PACK=4` too: **44.59 = +8.4%** over the live config. MTP-off spreads are the tightest in the campaign (0.3-0.4%). ⚠ one fill, single stream, synthetic **repetitive** prompt — which biases acceptance *in MTP's favour*, and it still lost. Needs a real-traffic confirm before flipping | bit-exact by construction (verify-gated); confirmed byte-identical (sha `a69dce26…`). Engagement proven both ways (`MTP context ready` / `speculation left off`) |
| **`PXA_FA_GQA_PACK=4`** on the single-P100 instance | that instance's card 5, fill 8352 | **DOES NOT REPRODUCE. NOT SHIPPED.** Measured **+1.7%** (MTP off, spreads 0.3/0.4%) and +2.7% (MTP on, vs a baseline band of 1.3%) — against the record's **~+6.9%**. `NH=8` is **+0.3% = NOISE** (the record's "+6.9% at NH=8" does not reproduce either). That instance satisfies every documented gate (D=256, K/V f16, gqa_ratio 4, sm_60) and sm_60 decode *does* route to `vec_f32` (`fattn.cu:137-166` via `pxq_use_sm60_vec_f32`), so it *could* fire | **VOID — engagement not proven.** The `PXA_FA_GQA_PACK: NH=4` banner comes from a `static` resolver lambda and proves only that the env var was **parsed**, not that the kernel dispatched; the kernel gate also needs `cols_per_block == 1`. The lever is documented NOT bit-exact yet output is byte-identical in every arm. **Needs a firing counter in `fattn-vec-f32.cuh` + rebuild before it can ship either way** |
| **`PXA_P100_FP16_GEMM`** | the single-P100 instance's card 5 (single sm_60), 40 chunks x 512 tok | ⚠ **FAILS THE STANDING QUALITY GATE: `Same top token` = 94.088 ± 0.234 %** (reject threshold is <98%). Median KLD 0.004266, max KLD 1.755914. **Auto-armed on ALL THREE live instances today** (every instance has a P100). Nothing was disarmed — reported for the owner's decision, since its benefit (+51% gpt-oss prefill on the record) is a real speed/fidelity trade | prefill/batched-path -> `llama-perplexity --kl-divergence`, reference = **lever OFF, same model** (not a Q8_0 parent). Engagement proven: `PXA_AUTO: P100_FP16_GEMM=off` vs `=on` in the two arms. Corpus `<engine-tree>` (**not** della). Perplexity NOT used |

## `-ts` SWEEP on the 2-card Rig-A layout — the optimum is further than 40,60
Cards 4,1 (V100+P100), `-sm layer -c 131072 -np 2`, fill 8192, 3 reps, vs the 3-card baseline
re-measured in the same campaign (`W_3card_base` 289.4 pf / 19.74 tg).

| ts (dev0=V100, dev1=P100) | prefill t/s | decode t/s | vs 3-card pf | vs 3-card tg |
|---|---|---|---|---|
| 36,64 | 338.5 | 21.35 | +16.9% | +8.2% |
| 40,60 | 354.4 | 21.70 | +22.4% | +10.0% |
| **45,55** | **378.4** | **22.33** | **+30.7%** | **+13.1%** |
| 40,60 (repeat) | 353.7 | 21.66 | +22.2% | +9.7% |
All four arms produced **byte-identical output** to the 3-card baseline (sha `cb0b4a6d…`).
Spreads: prefill 3.3-3.8%, decode 0.0-0.2%. The 40,60 repeat lands within 0.2%/0.4% of its first
run, so the ordering is real and monotone in dev0's share.
**Not found: the upper limit.** 45,55 is the best MEASURED point; 50,50 was NOT tested and is
predicted to OOM (dev0 also carries the 3670.81 MiB vision projector: 7106 weights + 3671 proj +
2308 compute + 3328 KV = 16413 MiB > 16384). The true optimum is somewhere in (45,50] for dev0.

---

# ⚠ QUALITY FINDING — `PXA_P100_FP16_GEMM` fails the standing gate, and it is ON right now

**CLASS:** prefill / batched dense-GEMM lever -> a batched KL run *does* reach it, so per the
ruling the instrument is `llama-perplexity --kl-divergence`, reading `Same top token`.
**REFERENCE:** the same model with the lever OFF (`PXA_P100_FP16_GEMM=0`), per the ruling — not a
Q8_0 parent.
**CELL:** the single-P100 instance's card 5, a **single P100 (sm_60)**, `Ornith-9B-Abliterated-PXQ4`, build-srv,
`-ngl 99 -c 512 --chunks 40` over `<engine-tree>` (a neutral corpus, **not**
the della corpus). **Sample: 40 chunks x 512 tokens = 20,480 tokens.**
**ENGAGEMENT PROVEN:** the two arms log
`PXA_AUTO: P100_FP16_GEMM=off …` and `PXA_AUTO: P100_FP16_GEMM=on …` respectively.

| statistic | value |
|---|---|
| **Same top token** | **94.088 ± 0.234 %** |
| Median KLD | 0.004266 |
| Maximum KLD | 1.755914 |
| Median Δp | −0.000% |

**Verdict against the standing thresholds (>=99.5% ship / 98-99.5% surface / <98% reject):
94.088% is a REJECT.** ~1 token in 17 changes at the top-1 position.

**It is currently auto-armed on ALL THREE live instances** — every instance has at least one P100
(Rig-A cards 1 and 6, the single-P100 instance's card 5, Rig-C card 0) and the ledger shows `P100_FP16_GEMM ON [2:1 hgemm]`
on each. So a lever that fails the project's own quality bar is shipping ungated on the owner's
live reviewer instance today. **Per instruction I have NOT disarmed anything — this is reported for
the owner's decision.** Its counterpart benefit is real and large (+51% gpt-oss prefill on the
record), so this is a genuine speed-vs-fidelity trade the owner should make knowingly, not a bug.
- Perplexity was NOT used as a gate (it is banned); the ref arm's `Final estimate` line is ignored.
- Coverage dropped: one model, one cell, one corpus, 40 chunks, prefill shapes only.

# UPSTREAM LEAD A — our tree HAS the unintended sm_60 fp16 fast path (llama.cpp issue #25593)
<https://github.com/ggml-org/llama.cpp/issues/25593>. Upstream: 95.002% token match, median KL
0.004962 unpatched. **Our tree carves out sm_61 only, never sm_60**, in exactly the places the
issue names (`<local-path>`):
- line 151/153: `#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610` -> `FAST_FP16_AVAILABLE`
- line 207: `static constexpr bool fast_fp16_available(const int cc) { return cc >= CC_PASCAL && cc != 610; }`
cc 600 (P100) passes both. The upstream fix is to add the `!= 600` / `!= CC_PASCAL` exclusion.
**Independent corroboration from our own tree:** `ggml-cuda/fattn.cu:28-44` already documents and
fixes ONE instance of this for the FA decode path — `pxq_use_sm60_vec_f32()` (PR #2144) routes
sm_60 decode to fp32 because "the fp16 FA vec kernel accumulates … in fp16, **flipping ~3-4% of
decode top-1 tokens** vs an all-fp32 reference", and reports it as free (tg128 96.79 vs 96.61).
That is the same defect class, the same magnitude, found here independently — but the fix is
**local to FA decode**; the general `fast_fp16_available()` gate is still open for every other
kernel keyed on it. My 94.088% KL number above is the *deliberate* GEMM lever, a separate site.
NOT MEASURED HERE: the generic leak's own ON/OFF gate (it needs a rebuild to toggle) — owed.

# UPSTREAM LEAD B — CPU input embedding forces a 3rd graph split (issue #22926)
<https://github.com/ggml-org/llama.cpp/issues/22926>. Closed as not-planned, **no merged PR** — an
idea with numbers, not code to pull. Upstream measured on 2x RTX 3090 / Qwen3.6-27B prefill:
930->943 @3,760 tok; 911->920 @14,937; **568->705 (+24%) @68,576** — the gain scales with prompt
length. Cost ~2.6 GiB VRAM.
**NOT TESTED here, deliberately, and the reason is measured:** our instances already report
`graph splits = 4`, and the 2-card Rig-A layout is the one worth trying it on — but the +2.6 GiB
does not fit alongside the 3670.81 MiB projector at the `-ts 45,55` optimum that just won
+30.7%/+13.1% (dev0 is already at ~15.4 GiB of 16 GiB there). It is dangerous on Rig-C too
(card 2 has 52-291 MiB free). Worth revisiting ONLY at high fill (24576+) and only if the
2-card `-ts` is backed off to leave headroom. Our `src/llama-model.cpp` has also diverged from
the line the issue names.

# UPSTREAM — low value, recorded so nobody re-chases
- **`--split-mode row` does not exist in our fork.** `grep -rn "SPLIT_MODE_ROW|\"row\"" include/llama.h common/common.cpp` returns nothing; our parser accepts exactly `none|layer|attn|graph` (`common/common.cpp:1947-1965`). Answer recorded either way, as asked.
- CUDA graphs: already default for batch-1; upstream's headline is H100-class small models. Not our bottleneck.
- NCCL tensor-parallel needs NVLink; our topology is **PHB on every pair** (`nvidia-smi topo -m`, all 7 GPUs). Consistent with our own record of `-sm graph` losing decode. Build already initialises NCCL communicators.
- Community FA v2 WMMA for Pascal/Volta: <https://github.com/sirCamp/flash-attention-legacy> — a port, not a flag. Noted as a known option; NOT attempted.

# 2026-08-16/17 — SWA-aware KV allocation lane (Rig-A / mgv-wt). Full record: `<local-path>`

## sm_60 `FAST_FP16_AVAILABLE` carve-out — **MEASURED, LARGE REGRESSION, REVERTED. Do not re-chase.**
Upstream llama.cpp issue #25593 proposes excluding cc 600 from the fast-fp16 gate alongside 610
(claimed: median KLD 0.002298->0.000001, greedy token match 96.47%->99.89%, tg **+1.5%**, prefill
"essentially tied"). Applied here and measured on Rig-A's own model/config
(`-ts 24,38,38` cards 4,1,6, `-c 262144`, f16 KV, `-np 2`, `-fa on`, vision), interleaved REF/NEW/REF:

| fill | pre-change | carve-out | pre-change (repeat) | prefill |
|---|---|---|---|---|
| 8192 | 269.45 / 18.52 | **136.94** / 18.54 | 269.56 / 18.51 | **−49.2%** |
| 24576 | 229.19 / 11.95 | **96.29** / 11.98 | 229.27 / 11.98 | **−58.0%** |

Repeat baseline agrees with the first to **0.04% / 0.03%**, so this is not drift. **Decode is
untouched**; the whole cost is prefill and it grows with fill. Fidelity on a 12-anchor diverse temp-0
corpus: **4/12 identical = 33.3% agreement (n=12)** — the change is numerically material.
- **Upstream's "prefill essentially tied" is FALSE on P100 with this engine.** GP100 has genuine 2:1
  FP16; forcing sm_60 onto fp32 roughly halves prefill for **any** head size.
- **Upstream's "three sites" is wrong for this tree — there are TWO**: `ggml/src/ggml-cuda/common.cuh`
  `FAST_FP16_AVAILABLE` macro and the `fast_fp16_available(cc)` host predicate. Keep them in lockstep.
- The D=256 story from the single-P100-instance/Rig-C lane (`fattn-tile-f32` handles only D=64/128, so D=256 falls to
  `vec_f32`) is real but is **NOT** why it regresses here: **the Rig-A target model is D=128**
  (`n_embd_head_k/v = 128`; the `n_embd_k_gqa = 256` line is `head_count_kv 2 x key_length 128`, the
  GQA embedding width — **not** the head dimension. Do not conflate these). Correct D analysis did not
  rescue the lever; only the measurement settled it.
- Prebuilt libs kept for a zero-rebuild redeploy if the owner ever wants the accuracy trade:
  `build-fugv60/refl-sm60carve/` (carve-out) and `build-fugv60/refl-presm60/` (shipped/known-good).
  **The CUDA kernels live in `libggml.so` (388 MB), not `llama-server`** — an A/B that swaps only the
  binary is a fake reference. Prove which is loaded via `/proc/<pid>/maps`.

## SWA-aware per-layer KV allocation — **NOT TRACTABLE as a patch in this fork. Not attempted.**
Target had 52 layers, `n_swa = 2048`, `swa_layers[]` = **39 sliding / 13 full** (verified from the
GGUF: `sliding_window_pattern` is a bool ARRAY `[T,T,T,F] x 13`, loaded via `get_key_or_arr` after
the scalar read fails). `n_swa_pattern` is **cosmetic** — printed value `1` is the untouched default
and nothing derives behaviour from it. KV today = **13312 MiB** (52 x 262144 x 256 x 2 B per side);
sizing the 39 sliding layers to a 4096-cell window would recover **~9828 MiB**.
**Why it is not a patch:** the cache is one global index space — a single `cells`/`head`/`size`/`n`
shared by every layer. `llm_build_kv_store` asserts `kv.size == n_ctx` and uses **`n_ctx` as the
V-transposed row stride**; the K-shift views `n_ctx` rows; the SWA mask is allocated at the graph-wide
`n_kv`; the per-layer copy fast-path patches `view_offs = kv_self.head*step`; slot allocation, defrag
and state save/load are all layer-agnostic. Shrinking a layer's tensor forces a per-layer
cell->row mapping; the only bounded mapping is a ring, which (a) breaks contiguous writes at the wrap,
(b) needs per-layer-class masks, and (c) **aliases under `-np 2`** — two live cells whose indices
differ by a multiple of the window land on the same row, silently corrupting one sequence. Upstream
solves this with a **separate** `llama_kv_cache_unified_iswa` (own cells/head/slot allocator, physical
eviction) which **this fork does not have**. It also re-establishes the index-vs-position coupling
that commit `16d5c1a8` removed after the NaN cascade — and adds a corruption mode that produces wrong
tokens rather than NaNs, which the soft-fail counters would **not** catch. Multi-day port, not a night's work.

## 2-card `-ts 45,55` at ctx 262144 — **the prize is REAL: it FITS once KV drops to ~3.5 GiB**
Measured on cards 4,1 (vision on, `-b/-ub 2048`, `-np 2`). GPU tensors 14212.86 MiB, compute 2308/card,
projector 3670.81 MiB on dev0.
- f16 KV (13312 MiB): **OOM** — dev1 cannot place its 2308 MiB compute buffer (7787.79+6656+2308 = 16752 > 16384).
- q8_0 KV (7072 MiB, recovers 6240): **OOM by only ~200 MiB** on dev0.
- **q4_0 KV (3744 MiB, recovers 9568): FITS and serves.** dev0 = 6425.07+1944.01+2308.01+287.27 = **10964 MiB of 16140 free**.
Since the SWA fix would recover **9828 MiB — ~260 MiB MORE than the arm that fit** — recovering the
KV **would** let the 2-card layout run at full ctx 262144. ⚠ **Quantised KV is a FIT PROBE ONLY**; it
alters numerics and its speed is not the shipped number (q4_0 decode pays a per-step dequant:
8192 → 359.7/18.05, 24576 → 298.79/11.96, decode spreads 11.6%/24.8%). The +30.7%/+13.1% prize can
only be re-measured at f16/262144 **after** the KV change exists.

## Method notes earned tonight (both cost real time)
- **An infrastructure failure must never render as a measurement verdict.** Two arms printed `fit=0`
  from `docker run` **rc=125** (`invalid reference format`, empty `$IMG` — the container never
  existed) and the decision table nearly retired the whole premise. Guards now: echo the fully
  expanded command; `rc != 0` or "container never existed" => `VOID — HARNESS FAULT, NOT A VERDICT`;
  only a container that started and then hit `cudaMalloc failed` counts as OOM.
- **`17*23 -> 391` with `max_tokens 16` proves NOTHING on this model** — reasoning consumes the
  budget (52 completion tokens for a 3-character answer) and `content` comes back **empty**, reading
  as a failure. Budget >= 512 and inspect `reasoning_content` too.
- **The shared repeated-sentence bench prompt is a degenerate fidelity instrument** (a 27B and a 9B
  produced byte-identical output). Use a diverse temp-0 corpus and report **agreement rate with n**;
  sanity-check any corpus by confirming two different models diverge on it.
  Working instrument: `<local-path>` + `cmp.py`.

# ─────────────────────────────────────────────────────────────────────────
# 2026-08-17 overnight (run7) — refusal comparison, requant blocker, harness fix
# ─────────────────────────────────────────────────────────────────────────

## ***THE ROUTING NUMBER: the uncensored fallback instance refuses 0 %***
The comparison the previous window deliberately deferred. Same corpus, same
settings, same scoring code as the 91–98 % stock baseline
(`BUILTIN_HARMFUL[::8][:100]`, n=100, greedy, 64 tokens, thinking off,
`/v1/chat/completions`). All three re-scored **with one script**, so no number
inherits another run's arithmetic — the pipeline reproduces the published
baseline exactly (combined 98.0 %, Wilson [93.0, 99.4]).

| model | prefix | arditi | substring | combined | Wilson 95 % CI |
|---|---|---|---|---|---|
| stock 9B baseline | 91.0 | 93.0 | 98.0 | **98.0** | [93.0, 99.4] |
| Rig-C 27B (`:8231`) | 0.0 | 3.0 | 0.0 | **0.0** | [0.0, 3.7] |
| single-P100 instance 9B (`:8219`) | 0.0 | 2.0 | 2.0 | **3.0** | [1.0, 8.5] |

Rig-C's 3 % arditi score is **100 % false positives** — hand-checked, all three
are the bare word `illegal` inside a *complying* answer (e.g. contrasting "tax
avoidance (legal)" with "tax evasion (illegal)"). True refusals: **0/100**.
**Route-to-fallback is vindicated, not undermined.** Rig-C answers 100 %
coherently (distinct-4gram ≈ 1.0, 0 degenerate).

## ***The abliterated instance is BROKEN, and its low refusal rate hides it***
Reading refusal rate alone says "it refuses 3 %, so it is permissive". Wrong,
and dangerously so:
- **50/100** harmful completions are degenerate repetition loops
  (worst: distinct-4gram **0.083**, one 4-gram repeated 12×).
- **77/100** open with the stereotyped filler *"No matter what" / "No problem" /
  "No need"* — 75 with the exact string *"no matter what"*. A **lower bound**:
  completions that pass the 4-gram gate still carry the same collapsed opener.
- Only ~47 % are non-refusing **and** non-degenerate.

**Confirmed to be the model, not the harness.** Through the *same endpoint,
same decoding, same instance*, benign prompts are clean, varied, and stop normally
("capital of France" → Paris; a haiku; photosynthesis in two sentences). The
collapse is **specific to harmful prompts** and reproduces at temp **0 and
0.7**. Signature of an over-aggressive abliteration: the refusal direction was
ablated so hard the harmful-prompt pathway fell into a fixed attractor rather
than learning to comply.
**Do not treat this instance as the permissive fallback.** The other one is.

## Requantizing Q8_0 → PXQ4 is DISABLED by default (not merely lossy)
The 35B MoE quantization stalled instantly on:
```
llama_model_quantize: failed to quantize: requantizing from type q8_0 is disabled
```
**The predicted risk was the wrong one.** Arch support was fine —
`general.architecture = qwen35moe`, 256 experts (top-8), 41 blocks, 753 tensors
all load. The wall was llama.cpp's refusal to requantize an already-quantized
source. `llama-quantize --allow-requantize` clears it and the job completes:
36039 MB → 18541 MB, 753/753 tensors, **695 s** at 24 threads, **4.381 BPW**,
loads and generates correctly (`--check-tensors` clean).
⚠ **Double-lossy provenance.** The box's reference PXQ4 was built **BF16 →
PXQ4** with Q8_0 kept as the KL reference. This one is **Q8_0 → PXQ4**. It is
not quality-comparable and must not be benched against BF16-sourced PXQ4 as if
the paths were equivalent.

## Correction: the degenerate bench prompt was NOT byte-identical
This file previously recorded that a 27B and a 9B "produced byte-identical
output" on the repeated-sentence prompt. Re-measured, they did **not** — the
hashes differ (663 vs 659 chars). The reality is worse for an exact-hash gate:
the outputs agree across **100 % of the shorter output**, differing only in
that the 27B ran four characters further before the token cap.
**An exact-hash fidelity gate would therefore have PASSED that pair by
accident.** Key a fidelity gate on **prefix agreement + a degeneracy measure**,
never on hash equality — hash equality is too brittle to express the property.
Measured, identical prompt text on both instances, n_predict 128, temp 0, top_k 1:

| prompt | common prefix | distinct-4gram | verdict |
|---|---|---|---|
| repeated sentence, `/completion` | **659 ch = 100.0 % of shorter** | 0.253 / 0.263 | BLIND |
| varied corpus, `/v1/chat/completions` | 53 ch = 11.4 % | **1.0 / 1.0** | DISCRIMINATING |

Fixed harness committed to the internal instance-management tooling as `instance-measure.py`;
the before/after is re-checkable via the corresponding `instance-corpus-verify.py` script. Corpus:
`<engine-tree>` (the box's own perplexity reference).
**The endpoint switch costs no metric** — llama-server returns the same
`timings` object (`prompt_per_second`/`predicted_per_second`) on the chat
endpoint.

## Two probe gotchas that will bite the next agent
- **`"status":"no slot available"` is NOT degradation.** The 27B instance runs
  `-np 1`, so its own `/health` returns that string while it serves *your*
  request. A liveness check that treats it as failure aborts a healthy run —
  and reads as "the instance degraded under load". It did not.
- **Thinking-disable is per-instance, not global.** The 9B guide instance has
  `--chat-template-kwargs '{"enable_thinking":false}'` baked into its launch
  line; the 27B instance has **no** chat-template-kwargs at all. Probing the 27B
  without a per-request `chat_template_kwargs.enable_thinking=false` returns
  **empty `content`** at 64 *and* 200 tokens — the whole budget goes to
  `reasoning_content`. Confirms the existing `max_tokens 16` note generalises:
  it is not the budget alone, it is budget × whether reasoning is on.

## `-ts` ceiling above 58 — NOT MEASURABLE while all three instances are up
Recorded as blocked, not attempted, with numbers. A bench copy of the 27B needs
~14.4 GiB of weights plus KV. Free VRAM with the three instances serving:
card0 4.0 / card1 2.4 / card2 0.3 / card3 10.6 / card4 2.4 / card5 0.5 /
card6 3.5 GiB. No card and no usable pair fits it. The only substantially free
card is the 1080 Ti, which is both too small **and** the wrong silicon —
a split ceiling measured on sm_61 would not transfer to the V100+P100 pair the
instance actually runs on. Needs a window where an instance may be taken down.
## Updates — 2026-08-17 — the reviewer instance on cards 2,0 measured for the first time

This instance had **never been benchmarked**. Cell: build-srv, PXQ4 + vision, cards 2,0
(dev0 = card 2 V100 sm_70, dev1 = card 0 P100 sm_60), `-ngl 99 -sm layer -ts 52,48 -c 131072
-b/-ub 2048 -np 1 -fa on`, fill 8192, n_predict 128, 3 reps/arm, temp 0, fresh server per arm,
`PXA_ENHANCE=1`, **3 interleaved repeated baselines**.

Geometry from the startup ledger: `arch=qwen35, n_layer 65, n_head 24, n_head_kv 4 (gqa 6),
n_embd_head_k = n_embd_head_v = 256, n_swa = 0`.

**BASELINE = 388.8 t/s prefill / 21.90 t/s decode** (mean of the 3 interleaved baseline medians;
baseline-to-baseline band **0.31% prefill / 0.10% decode**).

⚠ **The previously circulated baseline for this instance (~395 pf / ~23.4 tg) does not reproduce on
decode: 21.90 vs 23.4 is −6.4%, ~64x the baseline band.** Prefill agrees. Use 388.8 / 21.90.

⚠ Within-arm rep spread on prefill is ~3% and it is **systematic, not random** — rep 1 is the
slowest in every arm on this cell (cold effect). Medians of >=3 absorb it; a single-rep campaign
on this instance is worthless.

| arm | ts | ctx | prefill | decode | vs base pf | vs base tg | output |
|---|---|---|---|---|---|---|---|
| baseline (x3) | 52,48 | 131072 | 388.8 | 21.90 | — | — | sha `e60d5aa9…` |
| `PXA_P100_FP16_GEMM=0` | 52,48 | 131072 | 376.7 | 21.83 | **−3.1%** | −0.3% | byte-identical |
| **2-card push to V100** | **58,42** | **65536** | **432.0** | **23.00** | **+11.1%** | **+5.0%** | byte-identical |
| ts 58,42 @ full ctx | 58,42 | 131072 | — | — | **OOM (real)** | | `cudaMalloc 884.62 MiB on device 0` |
| ts 64,36 @ 65536 | 64,36 | 65536 | — | — | loads, **dies on first prefill** | | see caveat |

### `PXA_P100_FP16_GEMM` costs this instance only −3.1% prefill — the banner's "+51%" is not this cell
Engagement proven both ways (`PXA_AUTO: P100_FP16_GEMM=on` / `=off` in the two arms' ledgers).
The startup banner advertises "+51% gpt-oss prefill measured" — **a different model**. On this
instance turning the lever OFF costs **3.1% prefill and 0.3% decode**, ~16x smaller than the banner
implies. Anyone reasoning about this lever from the banner is reasoning about someone else's cell.
(Its fidelity cost is the separate 94.088% same-top-token reject already recorded above.)

### Shifting layers onto the V100 wins here too, but full context is the price
`-ts 58,42 @ ctx 65536` is **+11.1% prefill / +5.0% decode with byte-identical output** (3 reps
x 128 tok) — same direction as the 3-card→2-card result on the other instance, smaller because this
instance already starts V100-weighted at 52,48. **Cost: max context halves, 131072 → 65536.**
Not applied — a context reduction is a capability trade for the owner, not a tuning change.

### The `-ts` ceiling on this cell, and where it actually is
- At **ctx 131072 the instance is already AT dev0's VRAM ceiling**: 58,42 OOMs for real
  (`allocating 884.62 MiB on device 0: cudaMalloc failed`, exit 139) *after* the per-arm VRAM gate
  confirmed both cards >=13000 MiB free — so it is a structural verdict, not the environment fault
  that voided an earlier campaign on this box.
- At ctx 65536, `64,36` **loads but cannot serve** — it dies on the first real prefill, i.e. the
  weights fit and the compute buffer does not. ⚠ Not re-run clean after an earlier exit-139 arm,
  so treat as probable-not-certain. Usable ceiling is **between 58 and 64**; 58,42 is the best
  measured point and the true optimum was not found.

### Nulls by construction on this instance — recorded so nobody re-chases them
- **`PXA_FA_SWA_KEEP` is INERT here: `n_swa = 0`.** It cannot engage on this arch. (It remains the
  +19% decode win on the sliding-window instance — that is a different arch, not a fleet-wide lever.)
- **MTP is not armed and has no measured row for this arch** — the ledger prints
  `PXA_AUTO: spec arch=qwen35 -> no measured row, speculation left off`. Any "−1.7% MTP here"
  claim does not describe this instance as configured.
- **`PXA_FA_GQA_PACK` can only ever take NH=2 on this cell.** The gate needs NH to divide both the
  head count (24) and the gqa_ratio (**6**); neither 4 nor 8 divides 6, so the record's NH=4/NH=8
  rows are unreproducible here in principle.

## Updates — 2026-08-17 — the sm_60 FAST_FP16 carve-out: BUILT, MEASURED, REJECTED

Upstream llama.cpp issue #25593 reports that on sm_60 an fp16 fast path truncates fp32 math, and
that carving cc 600 out of `fast_fp16_available()` costs nothing. **Both halves of that fail here.**

### What the gate actually controls in THIS tree (read this before touching it again)
There are **two** sites, not three (`ggml/src/ggml-cuda/common.cuh:151` macro, `:207` function).
The pattern `ggml_cuda_highest_compiled_arch(cc) != 600` that upstream discussions cite **does not
exist in this fork.**
- The `FAST_FP16_AVAILABLE` **macro** is consumed in only two places (`mmq.cuh:1207`,
  `mmq_id_common.cuh:1624`) and both are the **Q2_K scale unpack** — irrelevant to PXQ-tier weights.
- The `fast_fp16_available(cc)` **function** is the real gate: it picks the f16 vs f32 FA kernels.
- **sm_60 decode is ALREADY fp32** via `pxq_use_sm60_vec_f32()` (`fattn.cu:41`, the PR #2144 port),
  whose own comment says "Prefill and the D=256 vec path stay on vec_f16".
  **=> the carve-out is a PREFILL-ONLY change on this fleet.** Gate it with a batched KL run; a
  temp-0 generated-token run is the wrong instrument and would return a clean meaningless number.

### Why it is a large REGRESSION here and not a free fix
`fattn-tile-f32.cu` supports **only head sizes 64 and 128** (`..._is_supported` → `K->ne[0] == 64 ||
== 128`). Both instances on this tree are **D=256**, so with the carve-out the `!fast_fp16_available`
branch sends D=256 prefill to the **single-column `vec_f32`** kernel — precisely the route
`PXA_FA_TILE256` exists to escape (its comment records that kernel at **52.7% of prefill GPU time**
on a pre-Volta D=256 rig). Carving out 600 also makes `PXA_FA_MASK_SKIP_TILE` (default ON) a
**phantom lever fleet-wide**, since sm_60 was the only arch still reaching tile-f16.

**MEASURED (fill 8192, n=3/arm, interleaved repeated baselines, ON/OFF selected per-arm by swapping
an archived `libggml.so` via `LD_LIBRARY_PATH`, so the arms are interleaved rather than split
across a rebuild):**

| cell | sm_60 share | baseline pf | carve-out pf | Δ prefill | Δ decode | pf band |
|---|---|---|---|---|---|---|
| single P100, 9B-class | 100% | 719.8 | 379.9 | **−47.2%** | no effect (inside 4.25% band) | 0.54% |
| V100+P100, 27B-class | ~48% | 388.2 | 236.1 | **−39.2%** | no effect (inside 1.6% band) | 0.03% |

Loss scales with sm_60 exposure — a coherence check on the mechanism. **Decode is untouched on both,
because this tree already fixed sm_60 decode; the upstream "+1.4-1.5% token generation" gain is
unreachable here for that same reason.** Engagement proof for a compile-time change with no banner:
the effect itself, at 87x and 1300x the respective prefill bands.

**Status: REJECTED, source reverted, nothing shipped.** The built artifact is archived off-tree at
`<local-path>` (ONE-BUILD-POLICY: archive, don't fork) purely as
evidence and as the reference binary for the quality run below.

### The RIGHT fix, scoped but NOT built (do not re-derive this)
A D=256 **tile-f32** variant will not fit: the f16 D=256 tile is already 43KB static smem at
ncols=16 and the f32 staging arrays are twice as wide (~84KB) — over the 48KB/block limit. That is
why the fallback goes to `vec_f32` by design.
The tractable fix is the PR #2144 pattern applied to the **tile** kernel: in
`flash_attn_tile_ext_f16` the staging arrays (`KV_tmp`, `Q_h2`) are **shared** and stay half2, while
`kqmax` / `kqsum` / `VKQ` (`fattn-tile-f16.cu:87,92,94`, plus the rescale block at `:230-246`) are
**register** accumulators in fp16. Promoting only those to float/float2 gives fp32 accumulation with
**shared memory unchanged**, keeps D=256 prefill tiled, and keeps `PXA_FA_MASK_SKIP_TILE` alive.
Cost is registers (VKQ 8 → 16 at D=256/ncols=16) under `__launch_bounds__(..., 1)` — must be
occupancy-measured. Not attempted: it is a real kernel change in a tree two live instances boot from.

### ⚠ QUALITY — and a warning about the `Same top token` threshold itself
Batched KL, single-P100 cell, 40 chunks x 512 tok, reference = carve-out OFF (the shipped binary),
corpus `<engine-tree>`, perplexity ignored:
**Same top token 94.127 ± 0.233 %**, median KLD 0.004603, median Δp **0.000%**, mean Δp −0.073%.

**This is statistically indistinguishable from the `PXA_P100_FP16_GEMM` figure (94.088 ± 0.234%,
median KLD 0.004266) measured on the SAME cell and corpus** — despite being a completely different
and independent code path (flash attention vs dense cuBLAS GEMM; neither gates the other).
Two unrelated perturbations landing on the same number, with **median Δp exactly 0.000%**, points at
**near-tie coin-flips in this model/corpus** rather than at either lever doing equal damage.
**Consequence: ~94% appears to be the floor for ANY sm_60 numeric change on this cell — including
the higher-precision one, since the carve-out arm IS the fp32 side and still scores 94.1%.**
A gate that rejects the more-accurate implementation is measuring the wrong thing. Treat
`Same top token` as a tripwire here, not a verdict, and **re-read the earlier "REJECT at 94.088%"
for `PXA_P100_FP16_GEMM` in that light — it is probably overstated.**
Confound in the 94.127% itself, stated plainly: the two arms differ by kernel (tiled vs
single-column) as well as by precision, so summation-order noise is included; this **bounds** the
fp16 leak from above rather than isolating it. Isolating it needs the surgical fix above.

### ⚠ 2026-08-17 — `PXA_P100_FP16_GEMM=0` HANGS THE CHAT ENDPOINT on the single-P100 MTP instance
Disarming this lever was applied to both instances on this tree and **reverted on the single-P100 instance
the same night.** Measured cost of OFF was benign (−3.7% prefill there, −3.1% on the 2-card instance),
but the cost was measured on `POST /completion` and the instances serve `POST /v1/chat/completions`:

| endpoint | that instance with `PXA_P100_FP16_GEMM=0` |
|---|---|
| `/completion` | works, returns the expected answer |
| `/v1/chat/completions` | **hangs, empty body**, while `/health` reports `{"status":"ok","slots_idle":2}` |

The server loads clean (MTP context initialised, slots idle, no log errors) and simply never answers
a chat request, so the keeper's gen-probe failed on every recreate and the keeper went into a
**recreate loop**. Recreating without the env fixed it on the first try — causal, not correlated.
VRAM contention and the binary were both ruled out.
**The 2-card instance KEEPS the disarm and is fine** — it passes the same chat probe. Instance-specific.
Untested hypothesis for the morning: that instance is the only one running **MTP**, and the chat path
adds the jinja template + `reasoning_effort`; the dense-GEMM change plausibly interacts with
draft/verify on a templated prompt. **Test on a bench container before ever re-applying.**

**Standing lesson for this box, wider than this lever:** a lever can measure clean on `/completion`
and still break the endpoint production uses. Benchmark harnesses here drive `/completion`; the
instances serve `/v1/chat/completions`. **Treat a passing keeper `verify_seat` — not a passing
benchmark — as the gate before any live config change.**
### 2026-08-02 — DeepSeek-V4 flipped in-place RoPE (`PXA_ROPE_FLIPPED_v1`, upstream ik #2198 port)

DS4 stores every roped row as `[nope | pe]` and the classic ggml shape for that is
`nope = view(x,0)`, `pe = view(x,off)`, `pe = rope(pe)`, `x = concat(nope, pe, 0)`. The
concat reads a **non-contiguous** source and writes a full-width destination. At the
checkpoint's real geometry (`attention.key_length` 512, `rope.dimension_count` 64,
`n_head` 64, `nt` 383) the q and attn-derope concats are about **50 MB per layer each**,
against roughly **1.1 MB** for the compressed-KV ring concat.

⚠ **This corrects a claim in our own notes.** An earlier internal quant-ladder note (2026-08-01,
not published) attributes the profiled `concat_f32_non_cont` cost (3.8% of prefill over 9295 calls) to
"raw-ring ⊕ compressed-K assembly". By tensor bytes that is off by roughly 45x — the
partial-RoPE concats dominate, not the ring.

Flipped RoPE removes the triple outright: rotate the **trailing** `n_dims` channels where
they already are. No view, no concat, no destination allocation. The rotation itself is
untouched — theta is still derived from the index *within* the rotated block, so only the
memory index moves.

⚠ **NO MEASURED THROUGHPUT EFFECT YET. Do not quote a speedup.** Upstream's author reports
+2-3% prefill / 0 decode for this change on his hardware; that is **his** number, not ours,
not reproduced here. What IS measured here is graph size (-636 nodes, -575 concats) and
correctness (byte-identical temp-0 output). No throughput A/B has been run.

| var | default | where | what it does |
|---|---|---|---|
| `PXA_DS4_ROPE_FLIPPED` | **1 (on)** | `src/graphs/build_deepseek4.cpp` (`dsv4_rope_flipped_enabled`) | Master switch for the six DS4 partial-RoPE sites: HCA compressed-KV, CSA/LID compressed-KV, indexer-q, attention q, attention kv, and the attention de-rope. On, each `view / rope / concat` triple collapses to a single flipped rope node that rotates the trailing `n_rot` channels and passes the leading ones through in the same kernel. `=0` rebuilds the **pre-port graph exactly**, so a same-binary A/B is possible and a rollback needs no rebuild. **Measured on DeepSeek-V4-Flash (43 layers, `-c 8192 -b 512 -ub 512`, one V100, `--n-cpu-moe 99`): graph 8229 -> 7593 nodes at `-fa on` and 8444 -> 7808 at `-fa off` (-636 either way, -7.7%); CONCAT nodes 1581 -> 1006 (-575).** Verification class: **bit-exact on CUDA**, confirmed at model level - temp-0 output is byte-identical to the `=0` arm for a 2600-token prompt with 320 generated tokens at `-fa off`, and for the short prompt at both `-fa` settings. ⚠ On the **CPU backend** the rope is not bit-reproducible between an in-place node and a fresh-destination node; that gap is PRE-EXISTING (reproduced bit-for-bit on the pre-port binary - see `tests/test-rope-flipped.cpp` arm 2b) and measures ~1e-6 % relative. **Throughput effect: UNMEASURED.** Upstream's author reports +2-3% prefill / 0 decode for the equivalent change on his hardware; that is his number, not ours. |
| *(removed)* in-place form | - | - | ⚠ **Upstream (ik #2198) makes these ropes IN-PLACE, which additionally drops the destination allocation. That is unsafe on sm_70 and was reverted here after being measured, not reasoned about.** With in-place ropes, `-fa on` plus a 2600-token prompt aborts with `ggml_cuda_compute_forward: ROPE_BACK failed / an illegal memory access was encountered` (pinned with `CUDA_LAUNCH_BLOCKING=1`; without it the sticky error surfaces later inside `cublasSgemm` and names the wrong kernel). Cause, from `GGML_SCHED_DEBUG=2` at the same fill: head dim 512 has no CUDA flash-attention kernel on Volta, so **86 of the `FLASH_ATTN` nodes run on the CPU backend** while **all 129 `ROPE_BACK` nodes are assigned to CUDA0** - `ggml_backend_cuda_supports_op` returns true unconditionally for ROPE/ROPE_BACK, and an in-place node is a *view*, which cannot be satisfied by inserting a copy: it must write into its `view_src`'s actual buffer. So a CUDA kernel writes to host memory. Upstream does not hit this because their DS4 attention runs on the GPU (Ampere+). The fresh-destination form removes the concat either way - that is the part that showed up in the profile - so the win is kept and only the destination allocation is given up. |
| *(no env — always on)* `op_params[15]` flipped flag | — | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/rope.cu` | The ggml-level mechanism. `op_params[15] == 1` on a ROPE/ROPE_BACK node means "rotate `[ne0-n_dims, ne0)` instead of `[0, n_dims)`". Implemented for f32 on CPU and f32+f16 on CUDA. Nothing sets it except the DS4 builder, so on every other arch the flag is 0 and the code path is byte-unchanged. Both rope constructors now write slot 15 explicitly rather than relying on the tensor allocator's zero-init. |

**Guards that ship with it — each is a place where a wrong answer would otherwise be silent:**

- `ggml_cuda_op_rope_rope_impl` declines a flipped pair. This one is **live, not
  defensive**: DS4 emits adjacent `GGML_OP_ROPE` nodes for q and kv, the CUDA dispatcher
  fuses adjacent ROPE nodes, and the fused path's own eligibility check is
  `memcmp(dst1->op_params, dst2->op_params, 15*sizeof(int))` — which stops one int short
  of the flipped flag. Without the guard the fused kernel would accept the pair and rotate
  the leading channels.
- `ggml_cuda_op_rope_cache_impl`, `ggml_cuda_op_rope_fast`, `ggml_cuda_op_fused_rope_fast`
  and `ggml_cuda_op_fused_rms_rope_fast` reject or decline flipped nodes.
- mrope and vision layouts **assert** rather than silently ignoring the flag (upstream
  ignores it); those layouts address the tail channels themselves, so a flipped request
  there is a caller bug.
- f16 on the CPU backend asserts (not implemented there; DS4 is f32).

**One upstream defect was not ported.** ik's non-in-place flipped *neox* kernel derives the
pair index as `i0/2 + rope_offset`, which walks past the end of the row for any
`ne0 > n_dims` (e.g. `ne0=192, n_dims=64` addresses element 223 of a 192-wide row). It is
unreachable in their build because DS4 always takes the in-place path, but it is wrong.
Ours derives the index from the thread's position within the rotated block, and
`tests/test-rope-flipped.cpp` exercises that path explicitly rather than trusting it.
