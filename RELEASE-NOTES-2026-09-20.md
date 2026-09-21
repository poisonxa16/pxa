<p align="center"><img src="docs/assets/pxa-network-banner.png" alt="PXA Network" width="760"></p>

# PXA v2026.09.20

On top of `v2026.09.13-rc3`.

**The short version: if you change nothing, this release behaves like the last one, only with
three bugs fixed.** Everything new and interesting in it — and there is a lot — ships **switched
off**, documented, with the number it measured and the command that turns it on. That is
deliberate. A release that makes a stranger's working setup behave differently is not a release,
it is a surprise. The new work gets defaults when it has been reviewed, gated and soaked, and this
note says for each piece exactly what is still missing. The one exception is Gemma 4's 128-expert
MoE, which is new this release and ships on by default — its own section below has the gate table.

There is also a correction to publish about how I measured decode speed. It is at the top, not the
bottom.

The reference for every lever, what it does, its default and how to turn it off is
[`docs/LEVERS.md`](docs/LEVERS.md).

---

## The correction — my speculative decode numbers were measured wrong

A speculative decoder keeps a table of text it has recently produced and uses it to guess what
comes next. My benchmark harness sent the same handful of prompts over and over inside one server
run, so from the second repetition onward the server was regenerating an answer it had already
written and the table could predict nearly every token of it. Acceptance climbed towards 100% and
the rate climbed with it — on one arm, 37 tokens/s on the first request and 120 by the sixth, for
the same prompt. Clearing the slot between repetitions does **not** clear that table.

Two honest things about it. **It affected both sides of every comparison**: the competitor ran
through the same harness in the same bracket, so the head-to-head shape was like-for-like even
where the absolute numbers were not. And **the effect is real, just misnamed** — an editor or a
tool-using client that regenerates text it has already seen genuinely does get that speed. What it
is not is the speed of a *first* answer, and a headline number should be a first answer.

Every decode figure in this note is re-measured:

- **a distinct prompt for every repetition**, generated fresh, so no repetition is ever a
  regeneration;
- three classes — free prose, a code-edit whose output re-quotes its own prompt, and a
  ~15,000-token long-context question — because a speculation change that helps one routinely
  hurts another;
- **the first-pass median is the headline**;
- six repetitions per cell where the number is a head-to-head claim, the competitor booted and
  driven in the same bracket on the same cards;
- **a plain control is really plain** — an arm with no `--spec-type` is not a control on this
  engine, because the automatic rule arms an n-gram drafter, so controls pin it off;
- **every speculative cell names its acceptance rule**, because a speculative rate without the
  rule beside it is not comparable to anything.

Prefill figures, the determinism and needle gates, and every lever A/B were unaffected: none of
them re-sends a prompt.

---

## What changed for someone who changes nothing

Three fixes, and nothing else.

1. **The speculative verify path wrote back through the wrong tile.** Checking a batch of draft
   tokens runs the model at width 2–8 instead of width 1. The decode kernel's write-back rule for
   that case clamped its tile backwards, so the widths speculation actually uses were the widths
   it handled worst. The fix is bit-identical by construction and it is on unconditionally.
2. **A prompt-cache bug that moved tokens it should not have.** Reviewed, six findings fixed, and
   its test passes. It is unreachable in the default configuration anyway; it is fixed regardless.
3. **The drafter's cascade constants, including how often the n-gram table is fed.** The table was
   allowed to lag up to 32 tokens behind the text; it is now fed every step, which is what the
   contract always said. Measured neutral on fresh text. `PXA_SPEC_NGRAM_FEED_LAG=32` restores the
   old behaviour if you want to compare.

Two more are on **only if their check passed on this binary** — check the boot log for
`PXA_AUTO:` or `PXA_PXQ_MMVQ_COLS` lines to see which side of that this build landed on:
`PXA_PXQ_MMVQ_COLS` (a wider verify tile on Volta, engages only at verify width 3–8) and
`PXA_MTP_ZERO_BUDGET_SKIP` (a request that asks for no drafting stops paying for the drafter).

**Everything else below is opt-in, with one exception added after this cut was first cut: the
tensor split now picks itself.** No environment variable in this release turns itself on, except
where a section says so explicitly (Gemma 4's MoE support, the handful of small first-boot fixes
under "What changed by itself", and `-sm tensor` below, which the launcher now chooses for you on
a matched pair it has evidence for).

---

## How to read every number

- **The cell, not an average.** Unless a row says otherwise: **2× Tesla V100 PCIe 16 GB (cc 7.0)**,
  both on PCIe x4 links, no NVLink; Qwen3.8-27B; this engine on the stock PXQ4 file, mainline
  `llama.cpp` @`4c9233c0` on the stock `UD-Q4_K_S`; `-c 16384`–`24576`, `-np 1`, 256 tokens
  generated, host timing off. Pascal rows say Pascal.
- **Two sampling shapes**, not interchangeable: greedy (temperature 0, top-k 1) and temperature
  1.0 with top-p 0.95 / top-k 20.
- **The competitor at its best, not at its defaults.** Mainline's fastest configuration on this
  pair is its own tensor split with its internal all-reduce, and that is what it is booted with.
- **A warm-up ramp is not a result**: every cell warms up before its first repetition, and a card
  that has just loaded a model reads faster over the first minute — judge an arm on its later,
  settled repetitions, not its first.

---

## `-sm tensor` — a tensor split for two identical cards, now the launcher's default

This engine has always split a model the ordinary way: card 0 takes the first half of the layers,
card 1 the second, and while one card works the other waits. `-sm tensor` splits each big matrix
instead, so both cards work on every token and the two halves are added back together once per
block. That addition is the whole cost of the idea, and most of this cut's kernel work is in it.

**Update, same day this section was first written: the launcher now picks this split for you.**
`--sm auto` (the default) resolves to `-sm tensor` on a pair of identical cards, an architecture
and quantisation tier it has hardware evidence for (currently Qwen3.8 at PXQ4), with the fused
all-reduce below armed automatically at both decode and prefill — no flag needed. It **stays on
`-sm layer`** for a single card, a mixed card pair, an unproven architecture or tier, a forced
`--ts`, or a `--workload longdoc` boot (that path runs flash attention off, and the split attention
builder has no non-FA path). Pass `-sm layer` yourself at any time to go back to the previous
default. The measurements and the "why opt-in" reasoning directly below predate that change and are
kept for the record; the current rule, its board numbers and the gate that cleared it are in
`docs/LAUNCHER.md` (§5b) and `bench/LEADERBOARD.md`.

```
llama-server -m <model>.gguf -ngl 99 -sm tensor -ts 1,1 ...
```

**It is admitted, not assumed.** The mode checks the file and the fleet before it engages and
refuses **by name** when the shape cannot be split correctly — a card pair with no peer access, an
architecture that has never been run through this split here, or a quantisation tier with no
hardware evidence. In this release the proven tiers are **PXQ4, PXQ3, and the standard K-quants
measured with it (`Q4_K_S`)**; every other tier and architecture is refused by name, with an
override for people who want to make the measurement themselves. A refusal prints what is wrong
and what to do about it:

```
PXA_TSPLIT: '-sm tensor' REFUSED
  why: a tensor split needs at least 2 devices and 1 is visible
  fix: use '-sm layer' on one card, or make a second GPU visible
  not falling back: '-sm tensor' was asked for explicitly. Set
  PXA_TSPLIT_FALLBACK=1 to demote to '-sm layer' instead of stopping.
```

**Measured, plain decode, Qwen3.8-27B PXQ4, greedy, fresh text, no other levers:**

| configuration | 2× V100 | 2× P100 |
|---|---|---|
| layer split — the default, what the last release shipped | 38.1 t/s | 17.8 t/s |
| `-sm tensor`, stock all-reduce | 44.2 t/s | 23.9 t/s |
| `-sm tensor`, fused all-reduce (`PXA_TSPLIT_REDUCE=fused`) | **50.0 t/s** | **25.9 t/s** |

Output is identical to the layer split — same greedy checksum on both card pairs.

The fused all-reduce is a second opt-in on top of the first. The stock way to add two cards'
partial results together is a handshake — each card signals, each waits on the other's event, the
host in the middle. The fused path does it in one launch with a staged peer copy and no back-edge
rendezvous. It is worth roughly **+13%** on top of the plain tensor split on the V100 pair.

**Why it is opt-in and not the default in this release.** It is proven at one request at a time, on
a P100 pair and the V100 pair. It has never been run at two concurrent requests and never soaked,
and on Volta it is a different numerical path from the layer split rather than the same arithmetic
in a different order. Two concurrent requests, a soak, and the gates decide whether the next
release calls it recommended.

**Four cards.** A four-device fused all-reduce is **not in this binary**; a four-card tensor split
uses the stock route.

## New, on by default on V100s: a head-256 attention kernel for long context, and a faster q8_0 prefill

Two V100 kernels ship this cut, both scoped to head-256 attention (Qwen3.8's shape):

**`PXA_FA_D256_VOLTA_TILE`** engages once a conversation's KV cache reaches 1,280 tokens (a short
chat never reaches it, and costs nothing). It is what a long conversation's decode step and this
engine's own speculative verify step run on above that floor. Measured on a two-V100 pair, first
pass, distinct prompt per repetition:

| class | before | after | change |
|---|---|---|---|
| long-context (~15k), plain decode | 31.90 t/s | 34.87 t/s | +9.3% |
| long-context, MTP depth 2 | 40.79 t/s | 47.28 t/s | +15.9% |
| long-context, n-gram → MTP cascade | 39.98 t/s | 47.35 t/s | +18.4% |

This was this release's one red board cell against mainline's own long-context speculative arm
(mainline 47.99 t/s, this engine 41.58 t/s the night before this kernel shipped). Re-measured in
the same bracket as mainline on the shipped package: this engine 47.78 t/s against mainline's own
46.07 t/s — the gap is closed. `PXA_FA_D256_VOLTA_TILE=0` restores the previous route exactly, and
the two `_MINKV` / `_MAXCOLS` knobs are documented in [`docs/LEVERS.md`](docs/LEVERS.md).

**`PXA_FA_MMA_VOLTA_Q8`** lets a matched q8_0 K/V cache at head 256 reach a faster prefill kernel on
V100 instead of the f16 path — **+16.7% prefill** on a two-V100 pair (1,070 → 1,249 t/s). Decode and
the f16 K/V path are untouched; a memory guard declines the q8_0 cache and falls through to f16
cleanly near a card's memory ceiling, rather than aborting. It shipped default on because the full
gate — 12/12 determinism at one and two slots, both needle lengths, six-way logit reproducibility —
passed on the packaged binary with it armed.

## New, opt-in: speculation that is lossless at every temperature

Speculative decoding drafts a few tokens cheaply and checks them against the real model. At
temperature 0 the check is simple: keep the draft token if it is the one the model would have
picked. Above temperature 0 it is not simple, and **this is where this release has something to
confess.**

**What the shipped engine actually does above temperature 0, unchanged from the last release.**
When you arm MTP drafting, a draft token is accepted if it holds at least 5% of the candidate
mass — even when the model sampled something else. Worse than the documentation said: that 5%
floor is compared against an **unnormalised, pre-temperature** weight, which is looser still. It
is fast. It is not the model's distribution. It has been in the engine for some time, the docs
described it too charitably, and the honest name for it is a *speed heuristic*, not an acceptance
rule.

**The lossless alternative is in this build, switched off, and labelled NEW**
(`PXA_SPEC_SAMPLED=1`). The drafter draws its token from its own distribution and the verifier
accepts it with probability `min(1, p/q)`, falling back to a residual draw on rejection. The
output distribution is the model's, exactly, at any temperature. Measured on the V100 pair with
`-sm tensor` armed, Qwen3.8-27B, 256 tokens, prose:

| | greedy | temperature 1.0 |
|---|---|---|
| no speculation | 47.80 | 47.80 |
| MTP depth 2, exact matching | **65.60** | 57.24 |
| MTP depth 2, **lossless sampled rule** | 64.48 | **64.54** |

At temperature 1.0 the lossless rule is worth roughly **+35%** over not speculating at all, and it
recovers almost all the ground exact matching loses there. That is the result this work exists
for: **speculation that never changes what the model would have said, and is still faster than not
speculating.**

**Why it is off.** Reviewers found two real defects in it — a dangling sampler pointer when a slot
is reused, and a stale draft distribution on a cascade. Both are being fixed. Neither is something
to hand a stranger in a release. It becomes the default when the fixes land, the end-to-end
distribution check passes on the cascade as well as on MTP alone, and it soaks; the 5%-floor
heuristic becomes the opt-in at that point.

**"Lossless" said precisely.** The acceptance rule never emits a token the target model did not
choose. It does **not** mean a speculative run is byte-identical to a non-speculative one:
verifying several tokens at once runs the model at a different batch width, and a different batch
width moves the last bits of a logit, which flips the occasional near-tie. Mainline has the same
property. Byte reproducibility is gated with speculation off, and the gate says so.

## New, opt-in: choosing the draft depth per request

Draft depth is not one number: a long repetitive completion wants a deep draft, a short prose
answer wants a shallow one or none. `PXA_SPEC_POLICY=1` decides per request on a file that carries
an MTP head. It is **off**, and the reason is specific rather than cautious: its first two-client
cell aborted the server, a review found five more real defects, it changes the VRAM reserve, and
it adds a multi-second stall to the first request. A user who changes nothing must not meet any of
that. A fix pass, a two-client soak, and VRAM/time-to-first-token numbers, then it gets a default.

## Not in this release, and named so nobody looks for it

Four pieces of this cut's work are **not in this binary at all**. They are finished or nearly
finished, they are measured, and they are parked for the next release because they moved a split
invariant somewhere else in the tree or were never reviewed, and this release does not ship a
moved invariant:

- the quantised K/V route through the fast Volta attention kernel (measured about +19% prefill at
  a 20,801-token fill, but its commit also changed the default attention path outside its own
  lever);
- the four-card fused all-reduce — a four-card tensor split uses the stock route;
- the per-op census instruments and the output-head placement lever;
- the Flash-Next kernel work (typed K/V gather, the small-ny F16 GEMV).

They are on the shelf, not in the bin. Next release.

## New, opt-in: parking a conversation instead of dropping it

A conversation that loses its slot is re-read from scratch on its next turn.
`PXA_CACHE_PARK_SPEC=1` parks the rest of it in host RAM: roughly **+32%** on a conversation
switch. It is off because one server crash in that work has never been explained, and an
unexplained crash is not something to default. The disk-backed store (`PXA_CACHE_DISK`) and the
prompt-end checkpoint (`PXA_CKPT_PROMPT_END`) are also off: measured wash and measured loss
respectively. They are documented so nobody has to re-derive that.

---

## Gemma 4 26B-A4B (the 128-expert mixture-of-experts model) — supported by default

Dense Gemma 4 (12B, and the smaller shapes) was already supported. The 128-expert sparse MoE
(`gemma-4-26B-A4B`) is new this release, and it ships **fully supported, with no switch** —
`PXA_GEMMA4_MOE=0` brings back the old refusal if you ever want it.

**What was measured**, two V100s, `-sm layer -fa -c 16384 -np 1`, greedy, a distinct prompt per
repetition, against stock llama.cpp built from source on the same cards and the same file:

| decode tok/s at prompt tokens | 14 | 2,370 | 6,363 | 12,710 |
|---|---:|---:|---:|---:|
| ours, Google's QAT `q4_0` file | 112.9 | 107.8 | 103.7 | 99.0 |
| ours, our own PXQ4 file | 106.7 | 101.9 | 97.9 | 93.7 |
| stock llama.cpp, same `q4_0` file, same cards | 99.6 | 96.0 | 93.8 | 92.9 |

| prefill tok/s at prompt tokens | 2,370 | 6,363 | 12,710 |
|---|---:|---:|---:|
| ours, Google's QAT `q4_0` file | 2,690 | 2,551 | 2,220 |
| ours, our own PXQ4 file | 2,347 | 2,116 | 1,799 |
| stock llama.cpp, same `q4_0` file, same cards | 1,278 | 1,422 | 1,440 |

Both files we serve lead stock llama.cpp at every depth measured, decode and prefill.

**What it passed before it was allowed to default on**, all on the same file and cards unless
noted: retrieval to 20,000 prompt tokens, determinism 12/12 byte-identical at one slot, a
two-slot content check (two different retrieval prompts sent at once, 8/8 answers carried their
own facts and never the other request's), and a 20-minute two-slot soak (502 completed, 125
cancelled, server alive, no trouble lines). The loader accepts either expert layout — Google's
merged tensor or the separate gate/up tensors this tree's own converter writes so the fused PXQ
expert kernels can use the file.

**Our own quant of it.** PXQ4 is **13.9 GB** and costs **+3.9%** perplexity against its own Q8_0
reference on chat-templated text; PXQ3 is **11.0 GB** and costs **+7.7%**. PXQ3 is the file for
**one card**: it boots at 16k context on a single 16 GB V100, where Google's `q4_0` file only fits
4k context on the same card. Say plainly what Google's file is and is not: it is a QAT
(quantization-aware trained) build, meaning different weights from a straight quantization of the
same checkpoint, so the row above is a comparison of two shipping files, not a quantizer-vs-
quantizer measurement.

**New, opt-in: `-sm tensor` for Gemma 4.** The tensor split above now works for this architecture
too — `PXA_TSPLIT_GEMMA4=1` opens it; without the switch, `-sm tensor` still demotes to `-sm layer`
on Gemma 4. Measured on the 26B-A4B with `PXA_TSPLIT_REDUCE=fused`: **8–21% faster decode on a P100
pair** across a 14-token, a 2,367-token and a 6,360-token prompt, plus a few percent on prefill. On
a **V100 pair it is 23–38% slower**, because the head-256 kernel above only serves the `-sm layer`
graph, so it is not offered there yet. A file with per-layer embeddings is refused either way; the
released 26B-A4B has none.

**What happens by itself, and how to turn each off** — every one of these prints a `PXA_AUTO:` (or
equivalent) line at boot:

- A chat request gets a direct answer instead of a forced reasoning trace — Google's own template
  defaults thinking off, and the engine now follows it. `--reasoning on`, or
  `"chat_template_kwargs": {"enable_thinking": true}` per request, asks for the thought back;
  `PXA_AUTO_REASONING=0` declines the rule.
- The model's own chat template is used automatically (its turn markers are nothing like Gemma
  3's, so the built-in template map would have been actively wrong, not just missing);
  `PXA_AUTO_JINJA=0` declines.
- On V100s, a new attention kernel serves the 512-wide attention heads; `PXA_FA_D512_VOLTA=0`
  restores the old unfused path.
- A PXQ file of this model reaches the fused expert kernels, and on more than one V100 the expert
  routing table is built on the device instead of round-tripping through the host;
  `PXA_PXQ_MOE_GELU=0` and `PXA_MOE_DEVICE_MAP=0` turn each off separately.

**What is still opt-in, on purpose:** the per-layer sliding-window KV cache (`PXA_GEMMA4_ISWA`,
which cuts KV memory on the 40 windowed layers but has not been soaked at two slots) and the
Gemma 4 assistant/MTP drafter (`PXA_GEMMA4_ASSISTANT=1`, which is not byte-identical to running
without it). `-sm layer` is the only supported split for this architecture — `-sm graph` and
`-sm attn` demote to it with a warning, and `-sm tensor` refuses it by name (this MoE has never
been measured through the tensor-split builder).

---

## The head-to-head, re-measured

Two V100s, Qwen3.8-27B, stock PXQ4 on our side and stock `UD-Q4_K_S` on mainline's, a distinct
prompt per repetition, medians of three, every speculative cell lossless. **Read the configuration
column** — several of the rows where this engine leads are rows where an **opt-in** is armed; out
of the box, the comparison is the plain-decode row.

| class | this engine | mainline, its own MTP config | configuration on our side |
|---|---|---|---|
| plain decode (prose) | 47.30 | 45.60 | default on both sides |
| plain decode (code-edit) | 45.65 | 45.50 | default on both sides |
| prose, greedy, MTP | 64.94 (+18.6%) | 54.77 | MTP depth 2 (opt-in) |
| prose, temp 1.0, MTP | 65.05 (+14.9%) | 56.61 | MTP depth 2 (opt-in) |
| code-edit, greedy, cascade | 130.50 (+68.0%) | 77.68 | n-gram → MTP cascade (opt-in) |
| code-edit, temp 1.0, cascade | 177.99 | 82.61 | **suspect — see note below, not for quoting yet** |
| long-context (~15.4k) decode, greedy | 57.70 | 57.26 | level; MTP depth 2 both sides |
| long-context (~15.4k) prefill | 829 t/s | 593 t/s (+39.8%) | plain, no speculation |

⚠ **The code-edit, temp-1.0, cascade row is marked suspect and is not a number to quote yet.** The
review that found the lossless-sampling bugs above also found that a cascade (n-gram in front of
MTP) with the old sampled-acceptance path could verify an n-gram draft token against a stale
distribution left over from an earlier MTP draft — exactly the arm that produced this cell, and
its being *faster* at temperature 1.0 than at greedy on the same class was the symptom that found
it. The greedy cell in the same row (130.50) is unaffected: at temperature 0 the exact-match rule
runs and the sampled path is never used.

**The codec is not the gap.** This engine reading mainline's own `UD-Q4_K_S` file on the same cards
in the same bracket reads plain decode within the spread of what it reads on its own PXQ4 file.
What the table above shows is the engine, not the quantization.

---

## Upgrade notes

- **The PXQ quantizer is now a separate download.** This release moves the code that *produces*
  PXQ files (`PXQ1`/`PXQ2`/`PXQ3`/`PXQ4`/`PXQ4-HQ`/`PXQ6`/`PXQ_UNIVERSAL`) out of this tree and
  into its own tool, `pxq-quantize` (`https://github.com/poisonxa16/pxq-quantize/releases`). The command-line arguments are the same
  ones you already know — only the program name changes. `llama-quantize` still does everything
  it always did except make a new PXQ file: every stock type, and reading/requantizing an
  *existing* PXQ file into something else (see "Leave PXQ", below) both still go through
  `llama-quantize` as before. Nothing about loading or running a PXQ file changes at all — every
  PXQ file you already have keeps working exactly as it did.
- **An importance matrix is now a documented option for the PXQ tiers, not a dead end.** Collected
  on chat-templated text and consumed with `PXA_PXQ_IMX=1`, it makes a measurably better file at
  the same size and the same tier composition — Qwen3-0.6B PXQ4 mean KL divergence 0.0818 → 0.0609
  and same-top-token 86.1 % → 87.6 %, Gemma 4 26B-A4B PXQ3 0.383 → 0.320 and 80.6 % → 81.7 %. The
  old guidance that called it net-negative was measured on raw text, out of distribution for a
  model you talk to; that reading is corrected in [`docs/QUANTIZING.md`](docs/QUANTIZING.md) and
  the recipe is an optional step in [guide 04](docs/tutorials/04-quantize-your-own-model.md). It
  stays **off by default** because collecting the matrix costs a calibration pass of its own —
  about half an hour on a P100 pair for that Gemma file.
- **If you converted Gemma 4 12B with the previous release's converter, your file has no chat
  template.** A converter defect dropped `tokenizer.chat_template` from the output; a chat request
  against that file will show the model's turn markers inside the answer instead of a clean reply.
  Re-convert from the original checkpoint with this release, or add the template to the file you
  already have with `gguf-py/scripts/gguf_new_metadata.py --chat-template`.
- Everything else in this release is opt-in (see above). If you pass none of the new flags or
  environment variables, this release behaves like the last one, with the three bug fixes at the
  top of this note.

---

## Gates this release is cited against

Greedy determinism 12/12 at one and two concurrent requests, on the default arm and on `-sm
tensor`, Volta and Pascal; needle recall at 3,121 and 20,801 tokens, including one under the split
with speculation armed; token-0 logit reproducibility; the end-to-end distribution check for the
lossless acceptance rule, with its lossy positive control; no-regression cells against the last
release's binary on the default configuration; a smoke test using the exact command lines the two
production seats use; a 20-minute two-client soak on the default configuration, on `-sm tensor`,
and with drafting armed; and every refusal message a user can meet, triggered on purpose and read.
Gemma 4's own gate table is in its section above and in `docs/KNOWN-ISSUES.md`.

**sm_61 (GTX 10-series, 1080 Ti).** The release builds for compute 6.0, 6.1 and 7.0. Every new
kernel in this cut is **compile-verified for sm_61**, including the template limits a wider tile
can overflow, and the packaged binary **boots and answers on a real 1080 Ti**: a small PXQ4 model
(Qwen3-0.6B), all 29/29 layers offloaded, `-fa on`, two coherent chat answers at roughly 220 t/s
decode, zero error lines. That card runs a production seat here, so a boot proof on one of this
release's larger models is still owed and needs its own maintenance window — this note names
exactly what was and was not measured rather than claiming more.

## Known limits

- `-sm tensor` needs two identical cards with peer access and a proven quantisation tier. It
  refuses anything else by name; the launcher now arms it for you on a pair it has evidence for
  (`--sm auto`, the default), and `-sm layer` still runs it by hand when you ask.
- With automatic drafting on, the same prompt can legitimately get a different greedy answer on
  two different requests — a verify batch runs at a different width than plain decode, and where
  two continuations are a genuine near-tie, the wider batch's arithmetic can land on either side of
  it. For repeatable output, run at `-np 1` and pass `--spec-type none`.
- Two server slots can resolve a genuine near-tie differently from each other. This is not one
  slot leaking another's content — a careful check (two different retrieval prompts sent at once)
  shows every answer carries only its own facts — it is ordinary batch-shape sensitivity that
  shows up only when the model's own answer was close to a coin flip.
- Gemma 4 is `-sm layer` only. The sliding-window KV cache lever and the Gemma 4 assistant/MTP
  drafter are opt-in.
- A q8_0 K/V cache still costs decode speed on Volta — `PXA_FA_MMA_VOLTA_Q8` (on by default) only
  reaches the faster kernel on **prefill**; decode and the f16 K/V path are unchanged.
- The temperature > 0 acceptance heuristic used unless you ask for the new rule is fast, not exact
  (see above); the lossless rule is opt-in this release and is the intended default next.
- Host RSS grows by up to about 8 GB over a long two-slot session and then stops. That is the
  server's prompt cache filling to its default `--cache-ram 8192` MiB cap, after which it evicts
  oldest-first (`cache size limit reached, removing oldest entry` in the server log); VRAM stays
  flat. Pass `--cache-ram <MiB>`, or `--cache-ram 0` to switch the cache off, if you need a smaller
  host footprint.
- The 1080 Ti (sm_61) build is compile-verified and boot-proven on a small PXQ4 model; a boot
  proof on one of this release's larger models is still owed.
- Everything from the last release's known-issues list that this release did not change still
  applies — see `docs/KNOWN-ISSUES.md` for the full, current list.

## What this release does not claim

- No claim that `-sm tensor` is proven under sustained load. It has had one soak at two concurrent
  requests: 20 minutes on the V100 pair with the fused reduce, no errors. That is one run on one
  pair of cards, which is why the mode stays opt-in.
- No claim that the temperature > 0 acceptance rule used by default (unless you turn on
  `PXA_SPEC_SAMPLED`) reproduces the model's own distribution — it does not, and this note says so
  plainly above.
- The code-edit, temperature-1.0, cascade cell in the head-to-head table is not a number to build
  on; it is flagged suspect and will be re-taken once the sampled-rule fix for cascades lands.
- No claim that Gemma 4's sliding-window KV lever or its assistant drafter are ready for a default;
  both are measured and opt-in, not defaults, this release.
- No four-card tensor split, and no claim about tensor-split behaviour on any architecture besides
  the ones named above as proven.

---

## Community

- Discord: <https://discord.gg/EqazvV9tf>
- If this is useful to you: <https://ko-fi.com/shatteredrealms1>

## Credits

Thanks to mistrjirka, a developer on the PXA Network Discord, for his help on this release.
