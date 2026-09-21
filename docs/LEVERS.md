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

## Boot and context

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_CTX` | on | A boot with no `-c` no longer dies asking for the model's whole trained context window. It retries at half the size, down to a 4,096-token floor, and settles on the largest context that actually fits, printing which one it picked. | `PXA_AUTO_CTX=0` restores the old behaviour: fail immediately if you did not pass `-c`. Passing your own `-c N` always wins regardless. |
| `PXA_ENHANCE` | on (this is the default level) | Selects the measured-good kernel levers for the card(s) it finds — a mixed-card box gets a per-GPU decision — and fills `-b`/`-ub` from the measured cell for that card set when you have not passed them. Prints the whole decision at boot. | `PXA_ENHANCE=0` rolls back to the pre-2026-09-03 shipped set. `PXA_REFERENCE=1` is the bit-exact all-off baseline. |

## Chat and reasoning

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_JINJA` | on for the architectures that need it (currently Gemma 4 and, as of this release, Qwen3.8) | Uses the model's own chat template instead of a built-in one that does not match it. For Gemma 4 this matters a lot: its turn markers look nothing like the built-in template's, so without this a plain boot would format every chat with the wrong template and nobody would notice until the answers looked odd. For Qwen3.8, without it a plain chat request returns the model's whole reasoning trace inside the answer instead of separating it out. | `PXA_AUTO_JINJA=0` declines and says why. An explicit `--jinja` or `--chat-template` always decides, in either direction. |
| `PXA_AUTO_REASONING` | on for Gemma 4 | Gemma 4's own template defaults reasoning **off**; the engine now follows that, so a plain chat request gets a direct answer instead of a reasoning trace that can eat a whole short answer budget. | `--reasoning on` per boot, or `"chat_template_kwargs": {"enable_thinking": true}` per request, asks for the thought back. `PXA_AUTO_REASONING=0` declines the automatic default and explains why. |

## Multi-GPU

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `-sm tensor` (a flag, not an env var, but documented here because it is new) | **the launcher's default** (`--sm auto`) on a matched pair, architecture and tier it has hardware evidence for | Splits every large matrix across two identical cards instead of splitting the model by layer, so both cards work on the same token at once. Faster for one conversation at a time on a matched pair; refuses by name on an architecture or quantisation tier it has no hardware evidence for, or a card pair with no peer access, and falls back to the layer split cleanly when the launcher chose it for you. | Pass `-sm layer` to keep the previous default — a single card, a mixed pair, or an unproven architecture/tier still resolve to it automatically. |
| `PXA_TSPLIT_REDUCE=fused` | **opt-in**, and only meaningful under `-sm tensor` | A faster way for two cards to add their tensor-split partial results together — one launch with a staged peer copy instead of a cross-device handshake. Output is byte-identical to the stock method. | Leave it unset; the stock reduce runs instead. |
| `PXA_TSPLIT_REDUCE_PREFILL` | off in the engine; the launcher sets it to `1` with every `-sm tensor` | Lets the tensor-split routes take prefill-width reduces too, not just decode-width ones, instead of leaving those on the ring. Measured +2.8% prefill on a V100 pair (752 → 773 t/s, Qwen3.8-27B PXQ4, a 12,710-token prompt), rising to 849 t/s at `-b 2048 -ub 2048`; greedy output is unchanged. Prefill on the tensor split is still below the layer split's on this box — that gap is the split's own trade, not this lever's. | Leave it unset if you are running `-sm tensor` by hand and want the ring route at prefill size instead. |
| `PXA_TSPLIT_FALLBACK` | off in the engine; the launcher sets it to `1` only when it chose the split for you (`--sm auto`, not a typed `-sm tensor`) | If `-sm tensor` would refuse your card/model/tier combination, this demotes to `-sm layer` instead of stopping. The refusal message still prints either way. A split you asked for by name keeps the old stop-and-refuse contract; a split the launcher picked for you never gets to be the reason a boot fails. Proved on a non-admitted file: Gemma 4 with `-sm tensor` and this lever forced prints the refusal, then falls back and serves normally. | Pass `-sm tensor` yourself (not `--sm auto`) to keep the old stop-on-refusal behaviour, or `-sm layer` to skip the split entirely. |
| `PXA_TSPLIT_UNPROVEN_ARCH` | off | Runs `-sm tensor` on an architecture whose split builder exists but has never been measured here, instead of refusing it. This is how a new architecture gets its first measurement — it is not a quality guarantee for your model. | Leave it unset; unproven architectures are refused by name. |

## Attention (V100)

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_FA_D256_VOLTA_TILE` (+ `_MINKV`, default 1280; `_MAXCOLS`, default 8) | **on** (2) on V100s, off at `PXA_REFERENCE` | A head-256 attention kernel for decode and the speculative verify step, taken once the conversation's KV cache reaches `_MINKV` tokens (below that, the old route runs — a short conversation costs nothing). Measured on a two-V100 long-context class: +9.3% on plain decode, up to +18.4% combined with the n-gram+MTP cascade, narrowing what had been this release's one red board cell to within 1–4% of the competitor on the same bracket. | `PXA_FA_D256_VOLTA_TILE=0` restores the old route exactly. `PXA_FA_D256_VOLTA_TILE=1` takes only the widest verify batches, skipping the width-1 (plain-decode) case. |
| `PXA_FA_MMA_VOLTA_Q8` | **on** | A matched q8_0 K/V cache at head 256 reaches a faster V100 prefill kernel instead of the f16 path. Measured +16.7% prefill on a two-V100 pair; decode and the f16 K/V path are untouched. A memory guard declines the q8_0 cache and falls through to the f16 path cleanly when a card is close to its ceiling — it never aborts. | `PXA_FA_MMA_VOLTA_Q8=0` keeps the f16 K/V path always. |

## Speculative decoding (drafting)

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_AUTO_SPEC` | on whenever there is about 2 GB of card memory to spare | Arms a cheap pattern-matching drafter that proposes the next few tokens when it spots a repeat, and the real model checks the proposal in one pass. Free-ish speed on repetitive text (code edits, template filling); close to a no-op on free prose. Never changes what token the model would have picked at temperature 0. | `--spec-type none` turns off drafting for that boot. `PXA_AUTO_SPEC=0` stops the engine arming a drafter for you at all. |
| `PXA_PXQ_MMVQ_COLS` | on **only if** its shape check passed on the binary you have (check the boot log) | A wider decode tile for the kernel a speculative verify batch of 3–8 tokens lands on. When it is on it is faster at that batch width; it never changes the output — bit-identical either way. | Nothing to turn off by hand; it is either on for your build or it is not, and either way it never changes what you get, only how fast. |
| `PXA_MTP_ZERO_BUDGET_SKIP` | on **only if** its check passed on the binary you have | A request that explicitly asks for zero draft tokens stops paying the bookkeeping cost of a drafter it will never use. | Same as above — nothing to set; it changes nothing observable either way. |
| `PXA_SPEC_NGRAM_FEED_LAG` | **0** (the table is fed every step) | How stale the pattern-matching drafter's table is allowed to get before this release's fix. Left at its default, the table always has the freshest text. | `PXA_SPEC_NGRAM_FEED_LAG=32` restores the previous release's behaviour, where the table could lag up to 32 tokens behind — only useful for comparing against an older build. |
| `PXA_SPEC_POLICY` | **off** | *Opt-in.* Picks the trained-head drafter's depth per request, from the acceptance rule that request will actually run under, instead of one fixed depth for every request. | Leave it unset. It is off in this release because its first two-client test crashed the server; it is not something to turn on for anything you rely on yet. |
| `PXA_SPEC_SAMPLED` | **off**, labelled NEW | *Opt-in.* The lossless sampling rule: above temperature 0, the drafter draws from its own distribution and the check accepts it with exactly the probability that makes the emitted token come out with the model's own distribution — unlike the rule below, it never emits a token the model would not have sampled. | Leave it unset. It is off this release because two real defects were found in it (fixed in the pipeline, not yet in a shipped binary); the older, faster-but-lossy rule below still runs unless you turn this on. |
| `PXA_SPEC_RELAXED` (+ `PXA_SPEC_RELAXED_PMIN`, default 0.05) | **on** whenever you arm MTP drafting, above temperature 0 | The acceptance rule this release still uses by default above temperature 0: it accepts a draft token that holds at least 5% of the candidate mass, even when the model itself sampled something else. This is fast, and it is a heuristic, not the model's own distribution — see `PXA_SPEC_SAMPLED` above for the exact alternative. At temperature 0 this rule is not consulted; the exact-match check always runs there. | `PXA_SPEC_RELAXED=0` demands an exact match instead — slower on repetitive text, but the model's own choice every time. Running at temperature 0 has the same effect for that request. |
| `PXA_CACHE_PARK_SPEC` | off | *Opt-in.* Parks a conversation that loses its server slot in host RAM instead of dropping it, so the next turn does not re-read the whole conversation from scratch. Worth roughly +32% on a conversation switch when it applies. | Leave it unset. It is off because one server crash in this code path has never been explained. |
| `PXA_GEMMA4_ASSISTANT` | off | *Opt-in.* Builds the Gemma 4 assistant/MTP drafter graph so it can be passed as a draft model. Measured worth 11–17% decode where the assistant agrees with the target; greedy output with it on is **not** byte-identical to the same run without it. | Leave it unset. `PXA_GEMMA4_ASSISTANT=0`, or simply not setting it, refuses the assistant file at load with a message naming the switch. |

## Gemma 4

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_GEMMA4_MOE` | **on** | The 128-expert Gemma 4 MoE (`gemma-4-26B-A4B`) loads and runs by default, no switch. Gate table (retrieval, determinism, two-slot content check, soak) is in `KNOWN-ISSUES.md`. | `PXA_GEMMA4_MOE=0` brings back the old refusal, as a clean load-time error naming this switch. |
| `PXA_FA_D512_VOLTA` | on, on V100s | A dedicated attention kernel for Gemma 4's 512-wide attention heads. Without it those layers fall back to a much slower unfused chain whose cost grows with context far faster than the fused kernel's — this is the largest single speed difference on this model on Volta. | `PXA_FA_D512_VOLTA=0` restores the old unfused path exactly. |
| `PXA_PXQ_MOE_GELU` | on | Lets a PXQ file of Gemma 4's MoE reach the fast fused expert kernels. Without it, a PXQ file of this model silently falls back to a much slower generic expert loop — same answer, far slower. | `PXA_PXQ_MOE_GELU=0` forces the slow fallback path, useful only for comparison. |
| `PXA_MOE_DEVICE_MAP` | on, on a multi-card fleet | Builds the MoE expert-routing table on the GPU instead of round-tripping it through the host once per layer. On multiple V100s running Gemma 4's MoE this is worth a substantial prefill speedup; correctness is checked, not merely assumed — a host cross-check mode exists (`=2`) if you want to verify it on your own hardware. | `PXA_MOE_DEVICE_MAP=0` turns it off. |
| `PXA_GEMMA4_ISWA` | **off** | *Opt-in.* A per-layer sliding-window KV cache for Gemma 4's 40 windowed attention layers, instead of allocating every layer a full context's worth of cache. Cuts KV memory sharply (measured about −74% on one test) and is deterministic, but has not been soaked at two concurrent slots. | Leave it unset; the engine allocates full-context KV for every layer, which is correct but larger. |
| `PXA_TSPLIT_GEMMA4` | **off** | *Opt-in.* Opens `-sm tensor` (see Multi-GPU above) for Gemma 4, which otherwise demotes to `-sm layer` on this architecture. Measured on the 26B-A4B MoE with `PXA_TSPLIT_REDUCE=fused`: 8–21% faster decode on a P100 pair across short, mid, and long prompts, plus a few percent on prefill. On a V100 pair the same split is 23–38% *slower*, so it is not offered there yet — the head-256 attention kernel above only serves the `-sm layer` graph. A file with per-layer embeddings is refused either way; the released 26B-A4B has none. | Leave it unset; `-sm layer` runs, as it does by default. |

## Quantizing

This one belongs to the separate `pxq-quantize` download, not to the engine — it changes how a
PXQ file is **written**, never how one is read. Every PXQ file loads and runs the same either way.

| variable | default | what it does for you | how to turn it off |
|---|---|---|---|
| `PXA_PXQ_IMX` | **off** | *Opt-in.* Makes the PXQ tiers **consume** an importance matrix you pass with `--imatrix` instead of ignoring it. Collected on text shaped the way you actually use the model — chat-templated, not raw prose — it measurably improves the file at the same size and the same tier composition: Qwen3-0.6B at PXQ4, mean KL divergence 0.0818 → 0.0609 and same-top-token 86.1 % → 87.6 %; Gemma 4 26B-A4B at PXQ3, 0.383 → 0.320 and 80.6 % → 81.7 %. It is off by default because collecting the matrix costs a calibration pass of its own — about half an hour on a P100 pair for that Gemma file. | Leave it unset: an offered `--imatrix` is dropped and the quantizer says so. The full recipe, the cost and the tiers it has been measured on are in [`QUANTIZING.md`](QUANTIZING.md#the-imatrix-ignored-by-default-an-option-worth-taking) and [guide 04](tutorials/04-quantize-your-own-model.md#optional--calibrate-first-with-an-importance-matrix). |

---

## What this page leaves out, on purpose

Every other `PXA_*` name you might find in a log line, a forum post, or an old command someone
pasted is a lab knob: an experiment record kept for reproducibility, gated to the one card or
architecture it was measured on, and very often a **measured loss** kept specifically so nobody
re-discovers it the hard way. Setting one by hand almost never makes your box faster. If you want
the full engineering reference — the mechanism behind each one, the measurement, the exact source
line — that detail lives in the private engineering tree and is not part of this release's public
documentation.
