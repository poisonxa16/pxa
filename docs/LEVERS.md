
# Levers — the `PXA_*` environment variables you might actually use

This page covers two kinds of `PXA_*` variable: the ones that are **on by default** and do
something to your boot or your output, and the opt-in ones this release's tutorials and release
notes tell you about by name. For each one: what it does for you, and how to turn it off.

It does **not** cover the rest of the lab — every other `PXA_*` name in the source is an
experiment record, gated per architecture, kept for the paper trail, and it is usually *slower* if
you set it by hand. This page is deliberately short for that reason.

Three master switches sit above everything else on this page:

| variable | what it does |
|---|---|
| `PXA_ENHANCE=0` | rolls the whole set below back to the pre-2026-09-03 shipped behaviour |
| `PXA_REFERENCE=1` | every lever off, the bit-exact baseline. Useful when you suspect a lever caused a problem |
| `PXA_MODE=balance` (default) / `PXA_MODE=max` | `balance` is flash-attention-on serving; `max` is flash-attention-off for maximum prefill on a batch job (not for GLM/MLA models) |

---

## NEW in v2026.10: the engine picks its own settings, not just the launcher

Previously, `-sm`/`-b`/`-ub`/`-ts` defaults lived in the Python launcher (`pxa-launch`). As of this
release they also live **in the engine itself**, as one lever registry
(`common/pxa-registry.cpp`), so a bare `llama-server`/`pxa-server` invocation — no launcher, no
`PXA_*` set by hand, in a container or out of one — resolves the same defaults the launcher would
have picked. The boot log prints every choice it made as a `PXA_REGISTRY:` line, so the decision is
auditable, not inferred. `pxa-launch --doctor` (or `docker run <image> doctor`) prints the same
table and starts nothing.

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_SM` | rule | Unset `-sm` resolves to `-sm tensor` on a pair the capability check admits (two identical cards, an admitted architecture and tier — see Multi-GPU below), `-sm layer` elsewhere. Past a pair it picks tensor only on four identical P100s (measured 2026-09-27); tiers PXQ4 and PXQ-Next 4/4S8/5 (`PXA_AUTO_SM_PXQN=0`: PXQ4 only). | Pass `-sm layer` or `-sm tensor` yourself; an explicit `-sm` always wins. |
| `PXA_AUTO_BATCH` | rule | Unset `-b`/`-ub` resolve to the measured-good cell for your card set and file if one exists, else an adaptive VRAM ladder. Same mechanism that used to be described only as part of `PXA_ENHANCE`; it now has its own name in the boot log. | Pass `-b`/`-ub` yourself. |
| `PXA_AUTO_TS` | on for one mixed sm_70+sm_60 pair, `-sm layer` | On a mixed V100+P100 pair with no `-ts` given, sets `-ts 1.4,0.6` instead of splitting layers evenly across unequal cards. | Pass `-ts` yourself. |
| `PXA_AUTO_UB_LONG` | rule, 4x sm_60 only | On a 4-card P100 fleet: a dense file gets `-ub 256`, an expert (MoE) file gets `-ub 2048`, when `-ub` is unset. `=0` forces flat 2048 for both; `=1` forces 256. | Pass `-ub` yourself, or set `PXA_AUTO_UB_LONG=0`/`=1` to force one shape. |
| `PXA_AUTO_SAMPLERS` | on | Fills `--temp`/`--top-k`/`--top-p`/`--min-p` from a per-architecture-family table when you pass none, instead of the engine's old one-size defaults. | Pass any sampler flag yourself; passing one still lets the others come from the table. |

## Boot and context

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_CTX` | on | A boot with no `-c` no longer dies asking for the model's whole trained context window. It retries at half the size, down to a 4,096-token floor, and settles on the largest context that actually fits, printing which one it picked. | `PXA_AUTO_CTX=0` restores the old behaviour: fail immediately if you did not pass `-c`. Passing your own `-c N` always wins regardless. |
| `PXA_ENHANCE` | on (this is the default level) | Selects the measured-good kernel levers for the card(s) it finds — a mixed-card box gets a per-GPU decision — and fills `-b`/`-ub` from the measured cell for that card set when you have not passed them (see `PXA_AUTO_BATCH` above). Prints the whole decision at boot. | `PXA_ENHANCE=0` rolls back to the pre-2026-09-03 shipped set. `PXA_REFERENCE=1` is the bit-exact all-off baseline. |
| `PXA_CKPT_BUDGET` (+ `_COMPUTE_PCT`, `_FLOOR_MB`, `_MARGIN_MB`) | on | Part of why a 27B now fits **131k context on one 16 GB card**: clamps the MTP/speculative draft's per-step recurrent-checkpoint depth to what the card can actually afford, instead of a depth that merely fits at boot and OOMs later. Changes draft *depth* only, never arithmetic — output is unaffected at the depth it does run. | `PXA_CKPT_BUDGET=0` disables the clamp; only worth it for comparison, not for a card you plan to keep serving on. |

## Chat and reasoning

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_JINJA` | on for the architectures that need it (Gemma 4 and Qwen3.8) | Uses the model's own chat template instead of a built-in one that does not match it. For Gemma 4 this matters a lot: its turn markers look nothing like the built-in template's, so without this a plain boot would format every chat with the wrong template and nobody would notice until the answers looked odd. For Qwen3.8, without it a plain chat request returns the model's whole reasoning trace inside the answer instead of separating it out. | `PXA_AUTO_JINJA=0` declines and says why. An explicit `--jinja` or `--chat-template` always decides, in either direction. |
| `PXA_AUTO_REASONING` | on for Gemma 4 | Gemma 4's own template defaults reasoning **off**; the engine now follows that, so a plain chat request gets a direct answer instead of a reasoning trace that can eat a whole short answer budget. | `--reasoning on` per boot, or `"chat_template_kwargs": {"enable_thinking": true}` per request, asks for the thought back. `PXA_AUTO_REASONING=0` declines the automatic default and explains why. |

## Multi-GPU

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `-sm tensor` (a flag, not an env var, but documented here because it is the headline) | **the launcher's and the engine's default** (`--sm auto` / `PXA_AUTO_SM`) on a matched pair, architecture and tier it has hardware evidence for | Splits every large matrix across two identical cards instead of splitting the model by layer, so both cards work on the same token at once. Faster for one conversation at a time on a matched pair; refuses by name on an architecture or quantisation tier it has no hardware evidence for, or a card pair with no peer access, and falls back to the layer split cleanly when the launcher or engine chose it for you. **Pairs, and four identical P100s** — bug #206 (wrong tokens on a 4-way split inside a container with docker's 64 MiB `/dev/shm`) is fixed: a failed NCCL group now falls back to the peer route, and 4x P100 measured tensor decode +5% (PXQN4) to +41% (PXQ4) / prefill about 2x over layer (dense 27B, 2026-09-27). `PXA_TSPLIT_ALLOW_4WAY=0` restores the pair-only guard. Give a multi-card container `--shm-size=1g`. | Pass `-sm layer` to keep the previous default — a single card, a mixed pair, an unproven architecture/tier still resolve to it automatically. |
| `PXA_TSPLIT_SSM_OUT_PANEL` | **on** | This release's tensor-split headline: every PXQ tier's `ssm_out` tensor (the DeltaNet output projection) can now be sliced through a panel-aware path instead of being refused outright. Before this, `-sm tensor` refused every PXQ hybrid file by name at load, citing `blk.0.ssm_out.weight` — only a plain `Q8_0`-`ssm_out` file would boot. Measured, 2x P100, PXQ4/PXQ3/PXQ6/PXQ4-HQ/PXQ2 and the shipped mix27 file, decode ctrl median: **+23.5% to +43.2%** (ledger `pxq-ssm-out-dim0-ranges-panel`). Measured on pairs; four identical P100s also take the split by default (see above). **Refused by name, not offered:** an architecture with no split evidence — `qwen35moe` (Ornith) fails its own KLD gate on this split and is not admitted by default (ledger `pxa-tsplit-qwen35moe-kld-gate`, rejected). | `PXA_TSPLIT_SSM_OUT_PANEL=0` refuses the panel path by name and falls back to the old refusal behaviour for any file that would have needed it. |
| `PXA_TSPLIT_REDUCE=fused` | **on by default with `-sm tensor`** (engine default since v2026.10; `=off` restores the staged ring; groups that do not peer fall back automatically) | A faster way for two cards to add their tensor-split partial results together — one launch with a staged peer copy instead of a cross-device handshake. Output is byte-identical to the stock method. | Leave it unset. `=off` is for A/B runs only: it forces the staged ring and costs decode speed on a pair that peers (2x P100, 27B, 262k context: 24.7 vs 26.5–26.9 t/s), and the engine prints a warning when it is set on such a pair. A config written for v2026.09.20, where unset meant the staged ring, should drop `PXA_TSPLIT_REDUCE=off` and `PXA_TSPLIT_REDUCE_PREFILL=1`. |
| `PXA_TSPLIT_EPI` | **on** with the fused reduce at decode width | Ends each tensor-split phase in ONE launch per card instead of three or four: the residual add, the two-card sum and the next norm run in the reduce kernel itself. Output is byte-identical. The PXQN form of this epilogue ships in the release build (the closed PXQN library; PXA release tarball and images only). | `PXA_TSPLIT_EPI=0` runs the separate kernels exactly as before. |
| `PXA_TSPLIT_EPI_PUSH` | **on** with `PXA_TSPLIT_EPI` | The weight kernel that produces each card's partial also writes it straight into the other card's staging slot, so the card-to-card copy overlaps the matrix work. Byte-identical output. | `PXA_TSPLIT_EPI_PUSH=0` lets the reduce kernel do the copy. |
| `PXA_TSPLIT_EPI_NORM` | **on** for P100 (sm_60) cards with `PXA_TSPLIT_EPI` | Gives the one-launch tensor-split epilogue to classic files (PXQ4, k-quants, Q8_0): the residual add, the two-card sum and the next RMS norm run in the reduce kernel, and the norm node is skipped. Byte-identical output (the norm kernel's exact operation order). `=2` also arms it on other architectures. | `PXA_TSPLIT_EPI_NORM=0` runs the separate kernels. |
| `PXA_PXQ4_RB` | **on** (P100 / sm_60, one-token decode of classic PXQ4 weights) in the release build (the closed PXQN library; PXA release tarball and images only) | A faster P100 decode kernel for classic PXQ4 weights, including a one-launch up/gate + SiLU. Deterministic, not byte-identical to the open kernel. A build from source has no such kernel and decodes PXQ4 with the open kernel (correct, about 12% slower). | `PXA_PXQ4_RB=0` restores the open decode kernel. |
| `PXA_TSPLIT_REDUCE_PREFILL` | off; nothing sets it | Lets the fused tensor-split route take prefill-width reduces too. On two cards this pre-empts the faster prefill DMA route (`PXA_TSPLIT_PF`): 743 vs 824 t/s prefill at a 14.8k prompt on a V100 pair (dense 27B, llama-server), greedy output unchanged. | Leave it unset. Only useful for comparing routes, or on more than two cards where `PXA_TSPLIT_PF` does not apply. |
| `PXA_TSPLIT_FALLBACK` | off in the engine; set to `1` only when the split was auto-picked for you, not typed by name | If `-sm tensor` would refuse your card/model/tier combination, this demotes to `-sm layer` instead of stopping. The refusal message still prints either way. | Pass `-sm tensor` yourself (not auto) to keep the old stop-on-refusal behaviour, or `-sm layer` to skip the split entirely. |
| `PXA_TSPLIT_UNPROVEN_ARCH` | off | Runs `-sm tensor` on an architecture whose split builder exists but has never been measured here, instead of refusing it. This is how a new architecture gets its first measurement — it is not a quality guarantee for your model. `qwen35moe` specifically stays refused even with this set — see above, it is a measured KLD failure, not merely "unproven". | Leave it unset; unproven architectures are refused by name. |

## Weight streaming

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_STREAM_WEIGHTS=layers\|experts` | **off** | *Opt-in.* Keeps weights pinned in host RAM and streams them to the GPU once per graph through a double-buffered VRAM ring, instead of holding the whole file resident in VRAM. Its real wins are **capacity** — a file bigger than the card — and MoE experts at a wide micro-batch. On a 27B dense model that already fits the card, it is close to a wash at typical serving depths and a measured **+6%** at a 64k-token bracket, not the larger figure an earlier reading reported: that reading was mostly the Volta MMA q8 attention route firing, which the best resident configuration declines for a different reason, not the streaming path itself (ledger `pxa-stream-weights`, corrected 2026-09-25). | Leave it unset; weights stay resident. |
| `PXA_XCACHE` | **on for a MoE file that does not fit** (auto placement + a routing-counts file: `PXA_XCACHE_COUNTS=<csv>` or `<model>.expert-counts.csv` next to the GGUF); **never on a file that fits** | Instead of moving whole layers' experts to host RAM, keeps the most-routed experts resident and moves only the least-routed experts of a few layers (planner R4: the number of layers is picked by a measured decode-cost model). Counts come from `llama-imatrix` with `PXA_MOE_COUNTS_CSV=<csv>` on the same file; a counts file that does not match the model is refused as a whole. Bit-identical to whole-layer streaming (greedy-512 sha equal). Measured 2026-09-28, tg128/decode: 35B-A3B PXQ4 on one 16 GB V100 whole-layer 45.8 -> cache 57.0 t/s (auto L=25; 45.6 vs 37.3 on a busier box); Flash-Next PXQN (512 experts) on two P100s `-sm layer` whole-layer 1.56 -> cache 2.90 t/s, and with zero-copy cold experts (`PXA_STREAM_ZC_MOE=64`) 5.29 -> 12.43. On a file that fits, forcing it is slower (35B on two V100s: resident 84.3 vs 46.2), so it stays off there. Declines when MTP is loaded (not validated yet). Not planned under `-sm tensor`. | `PXA_XCACHE=0` forces whole layers; `PXA_XCACHE=1` also plans it under a forced `PXA_STREAM_WEIGHTS`/`PXA_STREAM_EXTRA_MB`. |

## Attention (V100)

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_FA_D256_VOLTA_TILE` (+ `_MINKV`, default 1280; `_MAXCOLS`, default 8) | **on** (2) on V100s, off at `PXA_REFERENCE` | A head-256 attention kernel for decode and the speculative verify step, taken once the conversation's KV cache reaches `_MINKV` tokens (below that, the old route runs — a short conversation costs nothing). Measured on a two-V100 long-context class: +9.3% on plain decode, up to +18.4% combined with the n-gram+MTP cascade. | `PXA_FA_D256_VOLTA_TILE=0` restores the old route exactly. `PXA_FA_D256_VOLTA_TILE=1` takes only the widest verify batches, skipping the width-1 (plain-decode) case. |
| `PXA_FA_MMA_VOLTA_Q8` | **on** | A matched q8_0 K/V cache at head 256 reaches a faster V100 prefill kernel instead of the f16 path. Measured +16.7% prefill on a two-V100 pair; decode and the f16 K/V path are untouched. A memory guard declines the q8_0 cache and falls through to the f16 path cleanly when a card is close to its ceiling — it never aborts. | `PXA_FA_MMA_VOLTA_Q8=0` keeps the f16 K/V path always. |
| `PXA_FA_QKV_DIRECT_VOLTA` | **on** on V100s, off at `PXA_REFERENCE` | Decode-width attention over a quantized (q4_0/q8_0) K/V cache reads the cache in place instead of converting the whole cache to f16 on every token. Removes most of the V100 decode-at-depth tax: measured 37.01 -> 51.70 t/s at 16k context on a V100 pair (dense 27B PXQN4); short-context decode unchanged. Output changes in the last bits (different summation order); greedy needle recall passes. | `PXA_FA_QKV_DIRECT_VOLTA=0` restores the previous WMMA route. |

## Speculative decoding (drafting)

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_SPEC` | on whenever there is about 2 GB of card memory to spare | Arms a cheap pattern-matching drafter that proposes the next few tokens when it spots a repeat, and the real model checks the proposal in one pass. Free-ish speed on repetitive text (code edits, template filling); close to a no-op on free prose. Never changes what token the model would have picked at temperature 0. | `--spec-type none` turns off drafting for that boot. `PXA_AUTO_SPEC=0` stops the engine arming a drafter for you at all. |
| `PXA_PXQ_MMVQ_COLS` | on **only if** its shape check passed on the binary you have (check the boot log) | A wider decode tile for the kernel a speculative verify batch of 3–8 tokens lands on. When it is on it is faster at that batch width; it never changes the output — bit-identical either way. | Nothing to turn off by hand; it is either on for your build or it is not, and either way it never changes what you get, only how fast. |
| `PXA_SPEC_FAST_VERIFY` | off (0) | Lets the speculative-decoding verify step (2–8 tokens at once) use cheaper arithmetic in the 4-bit weight kernels. On a V100 the verify kernel gets about 12% faster at 4 tokens. On a P100 it makes no measurable difference. | Leave it off unless you have measured it on your cards. With it on, greedy output with MTP on can differ slightly from MTP off. It is still the same every run. |
| `PXA_PXQN_RHT_NODE_NY` | on (1) | PXQN files (the closed PXQN library; PXA release tarball and images only): on a P100 the verify step (2–8 tokens) of the rotated layers is about 26% faster at 4 tokens; output unchanged bit for bit. | Set 0 only to compare against the old path. |
| `PXA_MTP_ZERO_BUDGET_SKIP` | on **only if** its check passed on the binary you have | A request that explicitly asks for zero draft tokens stops paying the bookkeeping cost of a drafter it will never use. | Same as above — nothing to set; it changes nothing observable either way. |
| `PXA_SPEC_POLICY` | **off** | *Opt-in.* Picks the trained-head drafter's depth per request instead of one fixed depth for every request. | Leave it unset; it is off in this release for the same reasons as last release (see `RELEASE-NOTES`). |
| `PXA_SPEC_SAMPLED` | **on** | The lossless sampling rule: above temperature 0, the drafter draws from its own distribution and the check accepts it with exactly the probability that makes the emitted token come out with the model's own distribution. Checked end to end on the one-card 27B: emitted-token frequencies with and without speculation agree (chi-square p 0.55, 1500 vs 1500 samples), while the old relaxed rule fails the same test (z 17.7). | Leave it on. `PXA_SPEC_SAMPLED=0` goes back to the older heuristic rule below. |
| `PXA_SPEC_BLOCK_VERIFY` | **off** (`=1` arms it, with `PXA_SPEC_SAMPLED`) | A sampled draft of two or more tokens is accepted as a block instead of token by token: same output distribution, never fewer kept tokens per step. Falls back to token by token under repetition penalties or a counting reasoning budget. At draft depth 1 the two rules are the same rule. | Leave it off: on the one-card 27B the draft depth is 1 (the rules coincide), and at depth 3 it measured no faster. |
| `PXA_SPEC_RELAXED` (+ `PXA_SPEC_RELAXED_PMIN`, default 0.05) | **off** (level default only with `PXA_SPEC_SAMPLED=0`) | The older acceptance rule above temperature 0: it accepts a draft token that holds at least 5% of the candidate mass, even when the model itself sampled something else. Fast; not the model's own distribution, which is why the lossless rule replaced it as the default. At temperature 0 this rule is not consulted. | Leave it off. `PXA_SPEC_RELAXED=1` turns it back on explicitly. |
| `PXA_CACHE_PARK_SPEC` | off | *Opt-in.* Parks a conversation that loses its server slot in host RAM instead of dropping it. | Leave it unset. |
| `PXA_GEMMA4_ASSISTANT` | off | *Opt-in.* Builds the Gemma 4 assistant/MTP drafter graph so it can be passed as a draft model. Greedy output with it on is **not** byte-identical to the same run without it. | Leave it unset, or `PXA_GEMMA4_ASSISTANT=0`. |

**131k context on one 16 GB card (mtpfit, this release):** a chunked K/V-to-f16 conversion
(`PXA_FA_F16_KV_CHUNK` and `PXA_FA_DEEP_QKV_TILE`, auto: only where the whole-tensor conversion
past 128 MiB would not fit on the card, so a card with room keeps the v2026.09.20 attention bit for
bit) plus the checkpoint-budget
clamp above are what make `-c 131072` fit and serve on one V100 or one P100 without MTP. **MTP
itself is not the default at 131k** — it loads and serves short prompts on the release binary, but
a deep prompt (~90–96k tokens) still runs out of memory in the flash-attention verify path, and
plain decode is faster than MTP decode at that depth on the files measured so far. Ledger:
`mtpfit-fa-kvchunk-auto`, `mtpfit-131k-onecard-fit`, `mtpfit-fa-deep-width1-vec`.

## Gemma 4

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_GEMMA4_MOE` | **on** | The 128-expert Gemma 4 MoE (`gemma-4-26B-A4B`) loads and runs by default, no switch. Gate table (retrieval, determinism, two-slot content check, soak) is in `KNOWN-ISSUES.md`. | `PXA_GEMMA4_MOE=0` brings back the old refusal, as a clean load-time error naming this switch. |
| `PXA_FA_D512_VOLTA` | on, on V100s | A dedicated attention kernel for Gemma 4's 512-wide attention heads. | `PXA_FA_D512_VOLTA=0` restores the old unfused path exactly. |
| `PXA_FA_D512_CHAIN_F32` | **on** | The unfused attention path for Gemma 4's 512-wide heads (every such layer on a P100 or 1080 Ti, and short contexts on a V100) accumulates in fp32 instead of fp16. Measured against a double-precision reference on a V100: attention output error (NMSE) 7.1e-5 -> 1.3e-8 with 6 context tokens, 2.0e-2 -> 1.8e-8 with wide attention scores. This is the fix for the short-context divergence previously blamed on the dedicated kernel, which was accurate all along. | `PXA_FA_D512_CHAIN_F32=0` restores the fp16 path byte for byte. |
| `PXA_PXQ_MOE_GELU` | on | Lets a PXQ file of Gemma 4's MoE reach the fast fused expert kernels. | `PXA_PXQ_MOE_GELU=0` forces the slow fallback path, useful only for comparison. |
| `PXA_MOE_DEVICE_MAP` | on, on a multi-card fleet | Builds the MoE expert-routing table on the GPU instead of round-tripping it through the host once per layer. | `PXA_MOE_DEVICE_MAP=0` turns it off. |
| `PXA_GEMMA4_ISWA` | **off** | *Opt-in.* A per-layer sliding-window KV cache for Gemma 4's 40 windowed attention layers. Not soaked at two concurrent slots. | Leave it unset. |
| `PXA_TSPLIT_GEMMA4` | **off** | *Opt-in.* Opens `-sm tensor` for Gemma 4, which otherwise demotes to `-sm layer`. 8–21% faster decode on a P100 pair; **23–38% slower** on a V100 pair (the head-256 attention kernel above only serves the `-sm layer` graph there). | Leave it unset; `-sm layer` runs, as it does by default. |

## Quantizing

This one belongs to the separate `pxq-quantize` download, not to the engine — it changes how a
PXQ file is **written**, never how one is read. Every PXQ file loads and runs the same either way.

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_PXQ_IMX` | **off** | *Opt-in.* Makes the PXQ tiers **consume** an importance matrix you pass with `--imatrix` instead of ignoring it. | Leave it unset: an offered `--imatrix` is dropped and the quantizer says so. See `QUANTIZING.md`. |

## Speed history (`/pxa/speed`)

The server keeps one small record per finished request (prefill and decode t/s, prompt size,
cache hits, draft acceptance) and charts it at `http://<host>:<port>/pxa/speed`. Local only:
nothing is sent anywhere. Details in `LAUNCHER.md`, "Speed history".

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_STATS` | on | Keeps the in-memory speed history behind `GET /pxa/stats` and the `/pxa/speed` page. | `PXA_STATS=0`: no history, both routes gone. |
| `PXA_STATS_MAX` | 10000 | How many records the history keeps (oldest dropped first; about 2.5 MB at the default). | Set a smaller number. |
| `PXA_STATS_FILE` | **off** (memory only) | *Opt-in.* Appends each record to a JSONL file and reloads it at start, so the chart survives a restart. `default` = `~/.cache/pxa/speed-stats.jsonl`. | Leave it unset. |
| `PXA_STATS_FILE_MB` | 8 | Rotates the history file at this size (`file` -> `file.1`, one old generation kept). | Only used with `PXA_STATS_FILE`. |

---

## What this page leaves out, on purpose

Every other `PXA_*` name you might find in a log line, a forum post, or an old command someone
pasted is a lab knob: an experiment record kept for reproducibility, gated to the one card or
architecture it was measured on, and very often a **measured loss** kept specifically so nobody
re-discovers it the hard way. Setting one by hand almost never makes your box faster. If you want
the full engineering reference — the mechanism behind each one, the measurement, the exact source
line — that detail lives in the private engineering tree and is not part of this release's public
documentation.
