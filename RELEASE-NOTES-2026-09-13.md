<p align="center"><img src="docs/assets/pxa-network-banner.png" alt="PXA Network" width="760"></p>

# PXA v2026.09.13-rc3 — release candidate

Release candidate. `v2026.09.07-rc1` is still the last tag anyone outside this project has seen, so
this note covers everything since that tag: it carries forward what the unpublished `v2026.09.09-rc3`
and `v2026.09.11-rc3` candidates described — both were cut and gated but never published — and adds
this cut's work on top. The full
reference for every lever, its default, the configuration its number came from and its verification
class is [`docs/lab/LEVERS.md`](docs/lab/LEVERS.md); where a change has no number here, the number
lives in the row, and the row is the citation.

---

## How to read every number in this note

Say it once, at the top, so no line below has to repeat it.

- **The cell, not an average.** Every decode figure is **np1, temperature 0, `REPS 5`**, on the
  **2× Tesla V100 PCIe 16 GB pair (cc 7.0)** of the development box — **both cards on PCIe x4 links**,
  no NVLink. The four-card Tesla P100 (cc 6.0) rows say so explicitly. Nothing here is a claim about a
  card, a model or a batch shape that is not named.
- ⚠ **What the cross-engine tables carry, and what they do not.** Every cross-engine cell in this
  note runs **stock** Qwen3.8-27B on all three sides — this engine on the stock PXQ4 (`11d6dd2a`),
  mainline `llama.cpp` and upstream `ik_llama.cpp` on the stock `UD-Q4_K_S` (`b3853896`) — driven by
  the same client in the same bracket, with every quantised file on NVMe. The *weights* confound the
  earlier candidates carried is gone; what remains in a headline row is **codec and engine together**.
  One variable, not none.
  **The arm that separates them — this engine driving the competitors' own `Q4_K_S` file — has run on
  four P100s, and it reverses the prefill sign**: 198.52 t/s against mainline's 212.83 at 3,121 tokens
  and 172.63 against 266.28 at 20,801, i.e. **mainline is 7.2% and 54.2% faster at equal codec**. The
  Pascal prefill lead is the codec carrying an engine deficit, and it is published rather than left
  out — see [`bench/LEADERBOARD.md`](bench/LEADERBOARD.md). On the Volta pair that arm still has not
  run, so the pair's prefill rows stay codec-plus-engine. **No number in this note is a codec
  measurement.** The **same-file** comparisons — every lever A/B here — are unaffected: both arms of
  those are one file, so they isolate exactly what they claim to.

- **Three engines, and which one a comparison names matters.** This engine is a fork of
  `ik_llama.cpp`, which is itself a fork of `llama.cpp`, and the comparisons in this note are against
  both competitors at their **current heads** — mainline `llama.cpp` `4a899373`, upstream
  `ik_llama.cpp` `3bb386eb`. Where a row quotes mainline's pinned `fdb2c11c70` (2026-06-24) instead,
  it says so on the row; nothing quotes a pinned build silently. Upstream `ik_llama.cpp` is not a
  weaker configuration of mainline: it has `-sm attn` and `-sm graph` of its own, and MTP
  self-speculation, so any head-to-head that means anything has to run it **with those armed** — at
  its best, not at its defaults. Where this note credits a mechanism to upstream, that is a statement
  about who wrote it, not a hedge about the numbers.
- **Three prompt classes, and they do not move together.** **Control** is a *synthetic* repetitive
  prompt, where an n-gram table predicts the continuation almost perfectly — it measures the
  speculation ceiling and is the least like ordinary use. **Repetition** is a real repetitive workload
  (a code edit re-quoting its own buffer), partly predictable. **Prose** is free generation, where a
  table mostly cannot predict the next token, and it is the class closest to ordinary chat use. A
  speculation change that helps one routinely hurts another, so all three are quoted and none is
  averaged away. One number for "decode speed" on a speculative engine is a number with an unstated
  workload hidden inside it.
- **The bracket method.** A *window* is one lock on one card set with a no-drafter bracket arm at
  each end. If the two bracket readings differ by more than **1.50%**, the window is **VOID** and its
  numbers are shapes rather than results — something else on the box moved while the arms ran. Every
  figure below names the window it came from; VOID windows are quoted only where the direction is
  unambiguous, and are labelled.

---

## Read this first — the speculation default changed on every card family, and it takes every decode class

**On every card family from Pascal up (`cc >= 600`) the automatic decode default is now the n-gram stage alone** — long
(`n_max=64`), a two-token minimum, a 24-token lookback, and never wiped between steps — **instead of
the n-gram → MTP cascade.** No environment variable, no flag: a bare boot on a V100 pair or a P100 quad arms it.
It was measured on Volta first; the four-P100 measurement that widened it is further down this section.

Capture `pair-settle` (2026-09-14, cards 2 and 4, bracket drift **−0.63%** against a 1.50% band,
`REPS 6`, np 1), stock PXQ4 (`11d6dd2a`) on this side against mainline at its **current head**
(`4a899373`) running its own `ngram-mod` + MTP cascade — its best on this pair — on stock `UD-Q4_K_S`
(`b3853896`), both files on NVMe, both sides measured with a cold drafter state before every class
cell. This engine's arm is the **bare command line**: no `--spec-type` flag, the default arming
itself.

| class | this engine, new default | mainline, its own cascade | delta |
|---|---|---|---|
| control (a synthetic repetitive prompt — the speculation ceiling) | **147.89** | 125.07 | **+18.2%** |
| repetition (a real repetitive workload) | **76.65** | 60.54 | **+26.6%** |
| prose (free generation) | **141.02** | 83.27 | **+69.4%** |

**The drafter is not changing the output, only the rate**: this engine's greedy-decode hash matches its
own unspeculated arm, in the same bracket, on every class and every repetition.

**At two concurrent clients this engine now leads on all three classes, and it took three fixes on
the last night to get there.** The first two-client window found that the shipped n-gram default
could not serve a second slot on a 16 GB pair at all (see *Fixed*: the checkpoint budget priced one
slot's snapshot rows while the allocator claimed one set per slot). Once the budget priced every slot,
the pair afforded a 13-token draft at two slots and trailed mainline on two classes. Two more
measured changes to the same budget — a 256 MiB margin instead of 1024, and keeping a quarter of the
compute-buffer copy the budget held for the CUDA pool's growth instead of all of it (with a 512 MiB
floor on the whole reserve) — raised the two-slot draft to 23 tokens, and the cells read **control 222.10 against
mainline's 178.64 (+24.3%), repetition 144.17 against 115.55 (+24.8%) and prose
184.33 against 77.38 (+138.2%)**, sums over both slots in one valid bracket. The earlier
two-client row (163.69 / 125.44 / 148.75) was measured on the build that over-committed the card and
is withdrawn. Per-slot figures and the full accounting are in
[`bench/LEADERBOARD.md`](bench/LEADERBOARD.md).

**Why the long n-gram stage and not the cascade, stated as the trade it is.** The previous default put
an n-gram table in front of the trained MTP head: the table answers a model quoting its prompt, tool
JSON or a code edit, and the head answers what the table cannot. Two measurements retired it on Volta.
First, the head's stage costs prefill: on four P100s the shipping cascade reads ~48–50% of the
unspeculated prefill ceiling at *both* 3,121 and 20,801 tokens, while an n-gram-only chain matches the
unspeculated ceiling almost exactly — so the cost sits with the MTP stage, not with table indexing.
Second, a long table draft at a 24-token lookback wins every decode class on the pair outright, above.
The cascade remains available (`PXA_SPEC_AUTO_CHAIN=cascade`), and it remains the default on Pascal.

**Two defects had to be fixed before that default could ship, and both are worth reading.**

- **A capped response lost its last verified tokens.** The speculative path charged a whole verify
  step against the response budget *before* the loop that emits that step's tokens, so a response
  running into its `n_predict` limit mid-step dropped up to (step − 1) tokens off the end. At draft
  depth 1 that is at most one token and had gone unnoticed for months; at depth 64 it showed up
  immediately as the new default "diverging" from unspeculated output at temperature 0. It was an
  early **stop**, not a wrong token — the probe at the stopping prefix returned a byte-identical
  top-2 distribution on both arms — and the fix is one hunk. Afterwards all four arms produce
  byte-identical 512-token greedy output, and an A-B-A speed bracket puts every delta below the
  bracket's own drift: the default is exactly as fast as it measured, and nothing was being wrongly
  accepted.
- **A slot erase did not reset the drafter's persistent map.** `slots?action=erase` cleared the MTP
  companion and hidden carry but never the n-gram table, so a reused slot kept predicting out of the
  previous conversation's text — harmless for output, and fatal to any measurement that reuses a
  slot between cells. Erase now hard-resets a stage's persistent state as well: proved by draft and
  acceptance counters that are byte-identical across an erase (126 / 105) where they previously
  carried over (260 / 229), with the no-erase control unchanged.

**On four P100s the picture is different, and this release changes the default there too.** The
two-stage cascade is close to inert on that card set — the MTP stage costs prefill and returns almost
nothing at decode — while **the same long n-gram stage alone reads 95.57 t/s on the control class
against mainline's best measured 46.17, +107%** (against mainline's current head, measured in a
different bracket, +166%). So a bare boot now arms the table stage on its own on every card family
from Pascal up (the banner names both measurements); the flags it arms are
`--spec-type ngram:n_max=64,n_min=2,ngram_size_n=24` with `PXA_NGRAM_RESET_STREAK=0`, and
`PXA_SPEC_AUTO_CHAIN=cascade` restores the previous chain. The repetition
class there reads **21.25 against mainline's 19.54, +8.8%**, at REPS 6 in one bracket (capture
`quadrep2-w1`); the earlier REPS 3 reading of 17.60 against 18.33 undercounted this engine as much as it
overcounted the gap and is superseded — [`bench/LEADERBOARD.md`](bench/LEADERBOARD.md) carries both.

**And one thing about every speculative decoder, said plainly because it surprised me first.**
See *Known behaviour* below: speculative output is **not byte-reproducible at temperature 0**. A verify
step decodes M rows in one forward pass where plain decode always decodes one, kernels are selected by
batch width, and at a genuine near-tie between two continuations the argmax can land either way. It is
not a wrong accept, it is not specific to this codec or to four cards, and it is why speculative arms
are gated on fidelity rather than on a checksum.

---

## The adaptive draft length: measured, and not the default

The mechanism the class trade above asks for — an n-gram stage whose draft length grows on a high-acceptance
streak and falls back on misses — exists and has been measured. **It is not the decode default of this
release.** The default stays a **fixed**-length chain — the n-gram stage alone on every card family from
Pascal up, the n-gram → MTP cascade only where a visible device is older than that or where
`PXA_SPEC_AUTO_CHAIN=cascade` asks for it. Two classes are why, and both are stated before anything flattering.

**At one slot.** Window `speclen2 w1c`: one bracket, drift 1.17%, **stock against stock**, the V100
pair, medians of 5, np1. Greedy text is **identical across every drafted arm of this engine**
(`6d01fc27059a`) — identical to the *drafted default*, which is the honest form of that claim; it
differs from the no-drafter arm, and that difference is the verify-batch-shape non-losslessness
documented in [`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md).

| arm | control | repetition | prose |
|---|---|---|---|
| no drafter at all | 37.56 | 36.11 | 37.51 |
| mainline, its own `ngram-mod` + MTP cascade, tensor split | 127.01 | **50.15** | 61.20 |
| **this engine, the shipped cascade default** | 84.42 | 43.11 | 79.46 |
| this engine, adaptive draft length | **154.60** | 37.36 | **140.23** |

**Read the middle column first.** Against mainline's cascade the shipped default is **−14.0%** on
repetitive text and the adaptive arm is **−25.5%**. And the sharpest way to put the second number: on
that class the adaptive arm reads 37.36 against **36.11 for running no drafter at all** — **+3.5%**,
where mainline's cascade buys +38.9% on the same class in the same window. On repetitive text the
adaptive mechanism is not doing its job, and the cause is open.

**At two slots it gets worse, and this is what settled the decision.** Window `speclen2 np2`, valid,
drift 0.69%, two streams, aggregate tok/s. **This table has no mainline arm** — neither engine landed a
drafter arm on the control class in that window — so it compares this engine with itself only:

| arm (two streams, aggregate) | control | repetition | prose |
|---|---|---|---|
| no drafter at all | 49.80 | 46.51 | 49.61 |
| **this engine, the shipped cascade default** | 87.48 | **72.74** | **83.07** |
| this engine, adaptive draft length | **285.78** | 43.16 | 36.26 |

At two slots the adaptive arm is **below running no speculation at all** on prose — 36.26 against
49.61 — with draft acceptance halved, **0.473 against the cascade's 0.858**, and a bimodal spread
(min 24.83, max 141.34) that a median hides. A drafter that loses to no drafter is not a tuning
question.

**Two red classes. Both are OPEN, and they are open for different reasons.**

**The np2 prose collapse is root-caused.** The adaptive ramp kept **one** state shared across slots,
because the server shares a single speculation context — so one stream's miss reset the other stream's
length, which is exactly the bimodal shape the numbers show. A per-sequence fix exists. It is **not
built and not measured**, so nothing here claims it works; what changed is that this cell has a named
cause instead of a hypothesis.

**The repetition class at one slot is open, and the whole draft-length lever family is exhausted for
it.** Nine draft-length settings were measured — fixed 4, fixed 64/48, adaptive to 64, three fall
shapes, floors of 4 and 16, streaks of 1 and 2, a minimum-accepted-fraction rule — and **every one of
them lands between 37 and 44** on this class, against 50.57 for the shipping fixed cell and 61.62 for
mainline. Read the accept counts rather than the rates and the reason is plain: across those arms the
*proposals* swing from 1,593 to 2,304 — up 45% — while the tokens actually **accepted** sit flat
between 1,030 and 1,122, and throughput falls 27%. Extra proposal length buys nothing on this class
and costs linearly. A fall rule cannot fix it because it acts after the cost is paid; the growth
trigger is the thing that would have to change, and at ~0.65 acceptance per token a four-token
proposal is taken whole by luck often enough to keep ratcheting the length up. The only safe growth
trigger here is *do not grow* — which is the fixed-length cell that already ships.

**What makes that an exhaustion of the lever and not of the cell:** mainline drafts at the same long
64/48 on this class and **wins** it, 61.62 against 50.57, while proposing *more* tokens than this
engine's fixed cell does (1,806 against 1,593). Long drafts are not intrinsically slow on repetitive
text — they are slow **here**. That points upstream of the drafter, at the verify path, and there is
an independent finding against it: the verify batch reaches the matrix-vector kernel **two rows at a
time** rather than as one batch of the full draft width, so verification is priced per pair of rows
instead of per batch. A per-row price is exactly what makes proposal length a linear cost and rejected
proposals pure loss.

**So the cell stays red and OPEN, not exhausted.** The draft-length levers are done; **the open lever
is upstream of them — widening the verify batch so a long draft is verified in one pass.** That work
is live, and no claim is made for it here until it measures.

**What it buys where it works**, quoted after the costs and not before: +83.1% control and +76.5%
prose over the shipped default at one slot, +21.7% control and +129.1% prose over mainline's cascade,
and +226.7% control at two slots. Those are real and they are why the mechanism is being kept and
worked on rather than dropped. They are not a case for shipping it on by default while two classes are
red.

---

## The PXA attention split (`-sm attn`)

The `-sm attn` split mode originated in `ik_llama.cpp` (`ikawrakow`, 2025-12); **the PXA attention
split** is this project's version of it — the PXQ panel-aware cut, the hybrid (delta-net)
reduce-delivery fixes and demotion, the reduce route control, the model-exclusive admission and the
headroom balance: the parts that make it run on hybrid models, on the PXQ codec, on Pascal and Volta. The mode itself, the split builder, `ggml_reduce` and its transports, the
recurrent-layer split and the `qwen35` support it needs came into this line with the 2026-07-17
import.

What the mode does: instead of giving each card a contiguous run of layers, it splits the
**attention heads and the MTP verify batch** across the pair while the rest of the model stays
layer-split — worth having because a layer split leaves one card idle through the other's layers.
What the PXA attention split adds is the part that makes it **run at all here** — on a **panel
codec**, on a **hybrid recurrent architecture**, on **Pascal and Volta** — where before it did not.
Four pieces:

1. **A panel-aware split granularity.** A PXQ tensor cannot be cut on `K` at an arbitrary row: the
   codec keeps one fp16 anchor per 64-row panel header covering all of `K`. Splitting on panel
   boundaries is the only reason a panel codec can be cut at all; without it the slices are read at
   the wrong byte offsets, silently, because every assertion on that path is about sizes and PXQ
   satisfies those exactly.
2. **The hybrid (delta-net) reduce-delivery fixes**, their demotion guard and the consumer-side
   fallback — the reason a DeltaNet hybrid does not degenerate under a split graph.
3. **The reduce control surface**: the route selector, the pinned-host route, `PXA_REDUCE_TIME`,
   capture and the reduce debug output — the instruments that made the route choice below a
   measurement rather than a preference.
4. **The model-exclusive admission lever and its hparam fingerprint**, plus this cut's headroom
   balance between the split and the per-step checkpoint budget.

So: the PXA attention split is an upstream split mode made to work on this codec, on this
architecture class, on these cards — named because the working implementation on those models is
this project's, not because the idea is.

**What it measures, in valid windows:**

| cell | `-sm layer` | `-sm attn` | delta |
|---|---|---|---|
| default cascade, control (`casc-h2h`) | 85.65 | **103.87** | **+21.3%** |
| default cascade, prose (`casc-h2h`) | 81.92 | 77.56 | −5.3% |
| MTP head alone, control (`replanfix2-b1`) | 58.56 | **66.14** | **+12.9%** |
| MTP head alone, prose (`replanfix2-b1`) | 37.63 | 41.53 | +10.4% |
| 64/48 n-gram stage, prose (`replanfix2-c1`) | 65.62 | **106.99** | **+63%** |

That last row is the one to keep: at a long draft length the PXA attention split produces the **highest
prose figure measured on this box by any of the three engines**, 106.99 against mainline's 73.35 in
the same window — on different weights, as everywhere a mainline arm appears. Both arms above also set `PXA_REDUCE_NCCL=0`, which is a route choice and is quoted as part
of the configuration rather than attributed to the split.

**It ships behind an admission lever, `PXA_ATTN_SPLIT_QWEN38`, default off.** The load-time guard
demotes `-sm attn` to `-sm layer` on every DeltaNet hybrid architecture. This lever admits it for
**exactly one model shape** — `qwen35` carrying the Qwen3.8-27B hparam fingerprint, field by field —
and leaves every other file demoted exactly as before. It is admission only: no kernel, no reduce, no
`-ts` handling changes, and `-sm graph` is not admitted at all.

**The fidelity question that kept it a lever has now been measured, and the answer is why it stays
one.** The existing split fidelity gate scoped itself *out* of the MTP verify batch — the path every
number above runs on — so a new instrument was built for exactly that: `llama-perplexity` driven at
the speculative verify width, 12 chunks × 512, on the 2× V100 pair, on the **stock** PXQ4 file.
Against the layer split it reads **mean KLD 0.000684 on prose and 0.0106 on repetitive text** — the
same level the flash-attention on/off toggle reads on the same instrument (**0.000689 / 0.0098**),
which is the comparison that makes those numbers mean something rather than being two small decimals.
The **drafted rows are no worse than the accepted row**, and greedy text and draft counters are
**byte-identical to the layer split** on the harness.

It nonetheless **misses the pre-registered default-on criterion, on two tail statistics**: top-1
agreement is short by **2 rows in 3060**, and one worst-case logit is wider. That criterion was
registered before the numbers, so it is not renegotiated after them. **The PXA attention split
therefore ships as a measured option rather than a default** — worth **+21% control and +13% on the
MTP head alone** on the pair, and the best prose arm measured on this box. Turning it on by default is
**deferred until the rejection and checkpoint-restore path has an instrument of its own**: that path
is what a tail statistic is most likely to be reporting, and it has not been measured yet.

One further caveat belongs with the numbers rather than under them: **every arm in the table was taken
through the older blanket bypass**, because the model-exclusive admission is newer than the
measurements. The admission is sufficient on its own — the demotion branch is skipped when it fires —
but its path has not yet been exercised end to end on a serving load, so it prints the observed hparam
fingerprint against the wanted one when it declines, and an `ADMITTED` block when it fires. Read one
of those two on a first boot rather than assuming.

⚠ `PXA_ALLOW_GRAPH_SPLIT_HYBRID` is **not** this, and must not be used as if it were. That is a
blanket bypass that admits four hybrid architectures alike, its own lever row says in as many words
that it must not serve traffic, and its old speed figures are void because they were taken on a
configuration that was emitting NaN. The admission above is a whitelist of one model shape; the
bypass is the absence of a guard.

---

## Scheduling: the compute-buffer reserve stopped churning

On a speculative cycle the MTP head alternates between an N-row update pass and a 1-row draft pass.
Those two passes used to build **different graph shapes** — the update pass asks for fewer output rows
than it has, so it carries an `out_ids` row-slice; the 1-row pass asks for one row out of one, so the
slice was skipped — and two node counts on one context means the graph allocator's single plan can
never cover both. It re-planned, which is a full backend drain plus a compute-buffer reallocation,
**twice per accepted token**.

`PXA_MTP_STABLE_OUT_IDS` (default **on**) builds the slice unconditionally, where at one row out of
one the slice is the identity. Over a whole boot including prefill, reserve calls / coverage failures
/ allocator re-plans go from **242 / 174 / 176** to **7 / 5 / 11**, and the 229 width-5 re-reserves
per bench become one. The output does not move: greedy probe hash `a5467dd43144`, draft counters
`1013/1030` and `953/1158`, and the 512-token control generation all identical to the pre-fix arm.
Speed, window `replanfix2-b1`: **86.38 / 83.30** against **83.00 / 82.13** and **85.65 / 81.92** for
the same arm on the base build — **+1 to +4% control, +1.4% prose**, for a change whose point was
correctness of shape rather than speed.

A thrash guard (`PXA_RESERVE_GUARD`, default on, capped by `PXA_RESERVE_MAX`, default 24) latches the
mechanism off per context once it has re-reserved too often, and blacklists a shape after one failed
try — which is why the residual mismatch below costs a handful of events per boot rather than one per
cycle. `PXA_RESERVE_DEBUG=1` prints the node-by-node difference behind a coverage failure.

---

## Memory: a per-step checkpoint capacity the context can afford

A recurrent (DeltaNet/GDN) target checkpoints its state every speculation window, and the **per-step**
checkpoint buffers scale **linearly with the draft length**. On this V100 pair with the 27B PXQ4 file
they measure **377.58 + 226.55 MiB** at a 5-token capacity, **2996.02 + 1797.61** at 33 and
**5988.52 + 3593.11** at 65. At 65 the first card's budget is weights 7587.75 + KV 1245.52 + compute
432.93 + checkpoint 5988.52 = **15254 of 16384 MiB** — and the allocation *succeeds*. It leaves about
0.7 GB, the CUDA pool then grows, and the server dies with an out-of-memory a dozen speculative cycles
later. That is why a 64-token n-gram draft would not run at `-c 32768` on a 16 GB card, and because it
is a **startup** allocation, no scheduler escape hatch avoids it.

`PXA_CKPT_BUDGET` (default **on**) asks the question the allocator does not: after these buffers are
claimed, does every participating device still hold the compute buffers it has already reserved, plus
`PXA_CKPT_BUDGET_MARGIN_MB` (default 256)? The largest draft length for which the answer is yes
becomes the capacity, and every drafter stage is clamped to it. It allocates nothing to decide. **It
prints one line at every boot, clamp or no clamp**, because which capacity a run actually got is not
visible anywhere else, and two arms differing only in it have been compared as if they were like for
like:

```
common_speculative_init: recurrent checkpoint budget: the chain drafts up to 4 tokens (capacity 5), this context can checkpoint 5
common_speculative_init: recurrent checkpoint budget: the chain asks to draft 64 tokens, this context can checkpoint 33 - every stage is clamped to n_max=32
```

With it, the 64/48 arm in `replanfix2-c1` ran to completion at `-c 32768` on 16 GB cards with **exact**
per-step checkpoints — `this context can checkpoint 53`, drafts clamped to 52 — and posted the 185.88
control figure above. The same request under `-sm attn` resolved to capacity 20, because the split
leaves less headroom; that clamp is why the attention arm's control is modest rather than why it
crashed.

**Two things this budget got wrong until this cut, both found on the last night.** It priced the
capacity it was about to approve as *one* set of snapshot rows, while the allocator claims one set
**per slot**: at `-np 2` it approved 28 and the allocator took 56, leaving 30 MiB where 2.6 GB had been
promised, and the first real decode batch died in the CUDA pool with an out-of-memory. The fit test now
prices capacity × slots, returns the per-chain number, and the boot line names all three:

```
common_speculative_init: recurrent checkpoint budget: 65 tokens per chain x 2 state slot(s) = 130 snapshot rows would claim 12067.05 MiB on CUDA0 ... capacity 18 tokens per chain (36 snapshot rows over 2 state slot(s)) -> effective n_max = 17
```

And its margin was **1024 MiB**, which on this pair bought nothing but a shorter draft: with the budget
honest, 1024 afforded 13 tokens at two slots and 48 at one. **256** — measured to survive a 20,801-token
prompt on both slots at once, the 8k needle and concurrent requests at both slot counts — affords **17
and 56**. One-client control decode on the pair went from 163.57 to **196.47 t/s** in the same window
on that change alone, so the pair's one-client rows on the board, measured before it, understate the
shipped default. `PXA_CKPT_BUDGET_MARGIN_MB=1024` restores the old room.

**And the rest of the reserve was mis-sized in the other direction.** On top of the margin the budget kept
a full second copy of the compute buffers. That copy is not a duplicate of anything the scheduler owns —
it is a stand-in for the peak of the CUDA pool, which does grow during decode — and a full copy was far
more than that peak needs: on the pair it was 2963 MiB on the second card and it alone set the two-slot
draft at 14 tokens. Swept on the card, not reasoned about: keeping **none** of it dies at one slot (the
pool ran out at an 8k prompt), keeping a quarter survives a battery that opens the re-plan window four
times at both slot counts and is the fastest point measured on both shapes, keeping half or all is
safe and slower. `PXA_CKPT_BUDGET_COMPUTE_PCT` (default **25**) prices that fraction, and
`PXA_CKPT_BUDGET_FLOOR_MB` (default **512**) is a hard minimum on the whole reserve so that a device
with small compute buffers cannot fall back to the bare margin — the configuration that died. At the
defaults the two-slot draft is **23** tokens and the one-slot draft **60**; one-slot control on
the pair reads 214.10 t/s against 195.43 with the full copy kept. `PXA_CKPT_BUDGET_COMPUTE_PCT=100`
restores the old reserve.

> **`--recurrent-ckpt-mode gpu-fallback` is not the way to buy a longer draft.** It avoids the
> preallocation, and it is a measured loss: control-neutral, **prose −49%**, and it **changes the
> greedy output**. It is documented in [`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md) and it is not
> recommended.

---

## Decode

- **On a P100 (cc 6.0) the low PXQ tiers decode through a half2 inner loop by default.**
  `PXA_PXQ_MMV_H2` unset resolves to `3` (both low tiers) on `cc == 600` **exactly** and to `0`
  everywhere else — including the 1080 Ti, whose fp16 rate is 1/64 and for which the loop would be a
  loss. `PXA_PXQ_MMV_H2=0` restores the fp32 loop on a GP100.

  The mechanism: the shipped `pxq6_dot32` issues the same 16 pair decodes per 32 weights for every
  tier, so a 2-bit tier moves 39% fewer bytes than a 4-bit one and decodes at the same speed. That
  loop is ALU-bound, and the int8 route that fixes it on sm_70 needs DP4A, which GP100 does not have.
  GP100 does have full-rate `HFMA2`, and the LM4/LM8 books are fp16-exact (enforced by a startup
  self-check), so a half2 pair LUT carries **zero book error** — the entries are the same numbers, not
  roundings of them. Measured on 27B low-tier files, **2× P100 (cc 6.0)**, `srv n256`, medians of 7,
  in both arm orders, reporting the order-cancelled mean:

  | tier | np1 | np2 |
  |---|---|---|
  | PXQ2 uniform | +9.54% | +11.34% |
  | PXQ3 uniform | +14.42% | +16.06% |
  | PXQ2-attn4 | +4.27% | +5.59% |
  | PXQ3-balanced | +6.60% | +7.32% |

  Absolute anchor: PXQ2 uniform tg32 **18.587 → 20.385 t/s**. The lever is **not bit-exact against the
  fp32 arm by construction**, so it is held to the fidelity clause rather than to a hash: mean KL
  divergence no worse than the PXQ4 fused kernel's `0.000951` and same-top-p not below `98.710%`.
  Measured mean KLD **0.000026 – 0.000032** and same-top-p **99.64 – 99.75%**.

  ⚠ Gate this lever with `-b 8 -ub 8`, never the default batch. `llama-perplexity` is pure prefill and
  the dense 2D decode driver only accepts `ny <= 8`, so at `-b 512` both arms take the prefill GEMM and
  return bit-identical logits — a false pass on a run in which the kernel never executed. A greedy hash
  is not a pass condition either: the arithmetic differs by construction, so identical text would mean
  the kernel never engaged. And read `Mean KLD` from the per-arm log, never the run log's headline,
  which prints a **maximum** beside a bar that names a **mean**.

- **The low PXQ tiers get the split decode kernels PXQ4 already had.** `PXQ23_MMV_SPLIT` /
  `PXQ23_MMV_MT` remain on by default.
- **`PXA_PXQ23_MMVQ`** — PXQ2 and PXQ3 decode riding the same q8_1 MMVQ kernel PXQ4 uses. **Off by
  default and it stays off**: it is fast and it fails the fidelity clause — PXQ2 measures mean KLD
  `0.004624` against a comparator of `0.000951`, and PXQ3 `0.002186`. Neither is "no worse than". If
  +36% decode is worth +0.35% perplexity to you, the lever row documents the trade.

## Prefill

**Volta prefill against both competitors, at current heads.** Window `lb-fresh-pair` (2026-09-13),
2× V100 pair (cards 2 and 4), `REPS 5`, bracket drift **0.52%**. All three sides run **stock**
Qwen3.8-27B — this engine on the stock PXQ4 (`11d6dd2a`), mainline `4a899373` and upstream ik
`3bb386eb` on `UD-Q4_K_S` (`b3853896`) — every file on NVMe, every engine at its own best prefill
shape:

| prompt tokens | this engine, PXQ4 | mainline, `Q4_K_S` | upstream ik, `Q4_K_S` | delta vs best rival |
|---|---|---|---|---|
| 512 | **777.4** | 656.5 | 384.4 | **+18.4%** |
| 3,121 | **1,332.2** | 891.9 | 447.3 | **+49.4%** |
| 8,192 | **1,013.0** | 982.0 | 421.6 | **+3.2%** |
| 20,801 | **947.0** | 929.0 | 371.8 | **+1.9%** |

**What that table is, stated precisely: PXQ4 against `Q4_K_S` — codec *and* engine, together.** The
weights are the same on all three sides; two variables remain in it and neither is isolated. **The
equal-codec decomposition — this engine driving the same `Q4_K_S` file — has not run on this pair**,
and on four P100s, where it has, it reverses the sign. Until it runs here, read these rows as
engine-plus-codec.

**The margin narrows with prompt length and that is not smoothed over.** At 8,192 and 20,801 tokens
the lead over mainline is +3.2% and +1.9% — real, in a 0.52% bracket, and small. The large numbers in
this table are the short and mid classes.

"mainline, `Q4_K_S`" is its **layer** split, and the qualifier is load-bearing: its tensor split — the
configuration that wins its decode cells — is **slower** for prefill. ik's cells are its bare command
line, which beats its own MTP arm at every prompt length measured here. Each engine is quoted at its
own best prefill shape rather than at one shape chosen to suit this note.

**The four-card P100 prefill cell is green too, and it lives on the board rather than here**: 470.5
t/s at 3,121 tokens against mainline's 212.8 and 420.2 at 20,801 against 266.3, quoted at this
engine's fastest valid configuration (no speculation stage armed) — with the equal-codec row that
reverses the sign printed directly beneath it. Read it in
[`bench/LEADERBOARD.md`](bench/LEADERBOARD.md), where both halves are.

- **PXQ3's prefill dequant stops re-deriving what a 64-entry table already knows.**
- **`PXA_FA_TILE_V2`** — a flash-attention tile schedule that can issue a 128-bit shared load. Ships
  **off**; the lever row carries the measurement that decided it.

---

## Multi-GPU

- **A pinned-host route for the cross-device reduce, `PXA_REDUCE_PINNED`, ships off — as a measured
  loss with its numbers.** It stages the partials through pinned host memory with an in-kernel arrival
  spin, instead of the in-tree peer-BAR copy. It is **bit-identical on the card**: `tests/test-reduce-pinned.cu`
  passes 32/32 with zero differing elements on both devices at the decode shape, the MTP verify-batch
  shape, a vector tail and a shape its own predicate refuses (which falls back, still bit-identical).
  Per reduce, in the dominant bucket over 10,972 samples, it is **3.2× cheaper on the host** (13.08 µs
  enqueue against 41.42) and **1.43× dearer on the device** (66.83 µs span against 46.87) — and on a
  pair with working peer access the decode loop is device-bound between splits, so the enqueue saving
  buys nothing against the extra span across the ~130 reduces of a verify pass. The speed arms cost
  6–18% and were taken in a **VOID** window (1.88% drift) whose bias runs *against* the pinned arms,
  so the sign is safe and the size is not quoted. It is kept, off, with the instrument that measured
  it — `PXA_REDUCE_TIME`, the first per-reduce timer this operation has had on a card.
- **A companion finding, recorded because a retracted claim is worth as much as a confirmed one:** the
  scheduler's per-reduce backend drain, which that route was partly built to remove, **never runs on a
  two-card server**. Its instrument row is empty in both arms. The drain sits behind a `> 2 backends`
  guard, so it is live only on a wider asynchronous graph split, where it remains untested.

---

## Quantisation

- **The DeltaNet output projection is protected at `q8_0` on every PXQ level** (`PXA_PXQ_SSM_OUT`).
  `blk.N.ssm_out.weight` was moved onto the PXQ backbone in August on a byte-parity argument with no
  paired fidelity arm, and every PXQ level shipped it at four bits from that day. On Qwen3.8-27B,
  mean KLD against the file's own `Q8_0` source (wikitext-2, `-c 2048`, 10 chunks, one class changed):
  **0.433 → 0.0425**, same-top-p 93.05% → 93.26%, for **+5.10%** file size. Decode is unmoved — two
  P100s, np1, medians of 7: 18.29 / 18.10 / 17.83 / 17.81 t/s across `pxq4` / `mxfp4` / `q8_0` /
  `pxq6`, a 2.7% spread with no separation. `q8_0` and not the cheaper `pxq6` only because `pxq6` and
  `pxq4hq` have no vLLM decoder and a file is routed by its level, not its contents. It is a **floor**,
  not a pin: `--custom-q`, a PXQU map entry, the geometry demote and the MTP companion pin all still
  decide, and `PXA_PXQ_SSM_OUT=<type>|off` reproduces the old allocation for an A/B without a rebuild.
- **`pxa.pxq4hq.book` / `pxa.pxq4hq.sub`** — a mixed 4-bit file now describes itself, so a reader no
  longer has to guess which book and which sub-block layout produced it.
- **`pxq4hq` reaches the vLLM sidecar**, with a tier table, dispatch, and a converter that stops
  guessing.

---

## Models

### Gemma 4 — supported

**Dense Gemma 4 is supported** — 12B / 31B / E2B / E4B shapes. The architecture guard used to refuse
every `gemma4` file; it is now shape-aware and refuses only the 128-expert MoE, which has a heap
defect that is not yet fixed. Six converter defects were fixed along the way: Google's released dense
weights could not be converted at all before.

**Its eight 512-wide attention layers now run flash attention on the card, by default, and that is
this release's largest single speed change on any model.** With `-fa on`, a head size outside the
CUDA kernel set produced nodes the CUDA backend declines, and the scheduler placed them on the **CPU**
backend — at a cost proportional to the KV cache rather than to the prompt. The declined head size now
builds the unfused attention chain on the card instead. One P100, Gemma 4 12B PXQ4, `-c 16384 -fa -sm
layer -np 1 -b 2048 -ub 512, an 8,252-token prompt, and the shipping arm is a **bare boot with no
environment set at all**:

| | before (CPU placement) | after (on-card unfused chain) | |
|---|---|---|---|
| decode | 9.33 t/s | **20.76 t/s** | **+122.5%** |
| prefill | 109.2 t/s | **438.1 t/s** | **+301%** |

The opt-out (`PXA_FA_GPU_FALLBACK=0`) restores the old placement to within rep noise, and the
per-layer banner appears once in the bare log and zero times in the opt-out log — so the branch is
proved from the log rather than inferred from the timing. **The replaced path was also less accurate,
not only slower**: ggml's CPU flash attention accumulates P·V in an fp16 accumulator whenever V is
F16, which costs 1.54e-04 of rms(v) against ~5e-06 for an fp32 accumulator on the same 33 device-test
cases. The 8k retrieval still answers, and the greedy hash matches the old arm at 3,203 and 8,252
tokens.

**Two Gemma 4 levers ship OFF, and the reason is written down rather than implied.**

- **`PXA_FA_TILE_512`** — a 512/576-wide attention tile kernel. Its device test now passes all 33
  cases at 1.2e-07 nmse, after the *test's own reference* was found to be the defect (the same fp16
  accumulator above, plus an all-negative generator that hid it). **That pass is necessary and not
  sufficient**, which is the most useful sentence here: on real 8k logits the same binary sits ten
  times further from an fp32 truth than the incumbent path (symKL 1.122e-01 against 6.172e-03). Off
  until that is understood.
- **`PXA_GEMMA4_ISWA`** — a per-layer sliding-window KV cache, so the 40 sliding layers keep a fixed
  ring instead of a full-context allocation. The memory half works exactly as designed and is
  measured, not modelled: KV self size **1,376 MiB against 5,376 MiB**, −74.4%, hitting the prediction
  to the megabyte. It stays **off this release, but it is now deterministic**: identical requests used
  to return different logits once a prompt exceeded the window, because an emptied ring resumed at
  whatever cell the last eviction left; the ring now restarts at cell 0 on a full sequence removal, and
  the gate that found the defect passes on the fix (one server, five identical 3.2k-token requests: one
  sha and one distribution to nine decimal places; three fresh boots agree; the 8k needle survives;
  8k decode 20.4 t/s, prefill 418 t/s). It ships as an experimental lever; the default moves next
  release after the same gate at two slots.

Gates on one P100 with `google/gemma-4-12B-it-qat-q4_0`: retrieval answered at 3,121 / 8,267 / 21,358
prompt tokens through the model's own chat template; greedy identity 2/3 against stock `llama.cpp`
`82dbc4f01` — the same rate as the Qwen3-0.6B control; perplexity ratio 0.988; batches ≥ 1024 tokens
clean; determinism 7/7 at np1 and **2/7 at np2** (structural, and it reproduces with `PXA_REFERENCE=1`);
`llama-bench` pp512 207.1 t/s, tg64 17.1 t/s.

**The PXQ ladder for it, with the caveats the numbers need.** Against a `Q8_0` converted from the bf16
release, Gemma-4-12B **PXQ4 (6.98 GB)** measures mean KLD **1.337** on raw wikitext and **1.137**
chat-wrapped, against Google's own same-size QAT `q4_0` at **3.033** and **2.809** — **2.3× closer to
the full-precision reference at matched bytes**, and the PXQ4 file still retrieves correctly at 8,267
tokens. Three caveats travel with that, and they are not footnotes:

1. **PXQ3-balanced (5.92 GB) is where the two metrics disagree**: KLD 2.362 chat-wrapped, better than
   the QAT file, but perplexity **1452** against the QAT's 869. That tier is documented as not working
   rather than omitted.
2. Engine-versus-stock KLD on this model is **0.146–0.203** (same-top ≈ 88%) and is **reported and not
   interpreted**: it is neither the sliding window nor stock's CPU repack — both were tested and
   refuted — and it sits in a regime where *both* engines read this model above perplexity 500.
3. There is no stock CUDA build of it here, so there is no on-card A/B; `PXQ2-attn4` was not built, the
   31B was not run, and the MoE and MTP paths are still refused.

`PXA_AUTO_JINJA` resolves on for the Gemma 4 architectures.

---

### GLM-5.3-Flash — beta

**GLM-5.3-Flash runs on this box, coherently, on six cards and on all seven, and it is labelled beta
rather than supported.** It is the first release in which it runs at all.

- **Seven cards, `-c 24576`:** smoke 12/12 PASS; retrieval answered at 4,096 and 8,192 prompt tokens
  with both the code and the name found; determinism 10/12 byte-identical **at np1**.
- **Six cards, `-c 24576`, `-b 512 -ub 64`:** decode **13.82 t/s** at np1; prefill **52.4 / 51.8 /
  50.3 t/s** at 512 / 2,048 / 8,192 prompt tokens; smoke 12/12 PASS; determinism 10/10 at np1. This is
  the beta baseline, and it is a baseline rather than a result: no optimisation work has been done on
  this model.

**One defect had to be fixed to get there, and it was one line of intent rather than of arithmetic.**
At the first attention block the last prompt token could not attend to itself: the indexer's
incomplete key-pool tail was never selected, because the selection defaulted off when the file carries
no such key. It now defaults on whenever there is more than one pool, and a file key still overrides.
The effect is not subtle — at that block, per-token ratios against a reference move from 1.13 / 1.63 /
1.52 / 1.13 to 0.999 / 1.000 / 1.000 / 1.002, and six-prompt greedy identity goes from 5/6 to **6/6**.

**The stated limitations, all four, because "beta" is not a substitute for saying which.**

1. **One slot per server.** This graph boots at `-np 1` and only `-np 1` — it refuses mixed-sequence
   batches — so **run more servers for more clients**. No two-client throughput figure is quoted for
   this model, and any that appears elsewhere is two serialised streams summed, which reads
   near-identical to one stream rather than near-double.
2. **`-ub 64` is required at `-c 24576`.** With flash attention off the graph reserves an
   attention-scores tensor on **every** device, not once: 1,017–1,029 MiB per card at `-ub 128` and
   508–514 MiB at `-ub 64`, measured on all seven. `-b` does not move it; `-ub` is the only lever.
   The smaller micro-batch turns a 20,801-token prefill that aborted into one that completes, costs
   roughly 57 → 45 t/s of prefill, and costs **nothing** in output: the 512-token greedy hash is
   byte-identical across the two.
3. **The 20k-token determinism rows were not run.** The gates above are at 4k and 8k.
4. **Logit fidelity against a reference implementation is close but not identical, and the remainder
   is understood rather than fixed.** After the tail fix the layer-by-layer divergence is flat at the
   dump's print floor — there is no second structural defect — and what is left is MoE top-k routing
   amplifying sub-1% kernel arithmetic into an occasionally different expert choice. A 1e-3 tolerance
   on a top-1 probability is a bar no 288-expert router can hold across two different GEMM
   implementations; the honest replacement is a measured noise floor, and that is next release's work
   together with this model's optimisation.

### Qwen3.8-Flash-Next — loads again

**Its expert decode is the next thing to speed up, and the first attempt is on record as a lever that did not
pass.** Every PXQ2/PXQ3 byte in the Flash-Next file sits in the MoE expert tensors, and the half2 pair-LUT
decode loop this release ships for those tiers on P100 is dense-only, so on this file it touches nothing.
Extending it to the expert drivers (`PXA_PXQ_MMV_H2_MOE`, kept out of this release) reads **+2.8% /
+3.0% / +2.3%** on control / prose / repetition on the four-card seat, engaged on every card, and then fails
the token-0 spread gate at one slot: with the base binary bit-stable boot to boot (spread exactly 0 on eight
prompts), the lever moves the top-1 token on one prompt and the distribution by an order of magnitude over
the tolerance on two more. A reordered near-tie shows a small spread; a wrong function moves the token. It
is not in this release's binary; the finding and the kernel are in the next-release list.

**The Flash-Next PXQU-MTP file could not be loaded by the previous candidate, and can be now.** A
per-transformer-layer hparam array was sized against the block count, which in this tree *includes*
the grafted MTP head — so a file carrying exactly one entry per transformer layer was refused one
short. Sized correctly, the file loads and reaches tensor load; the same commit fixes the identical
comparison in the DeepSeek-V4 path. Proved on four P100s with its own negative control: the fixed
binary clears the check on the file that the previous commit refuses twelve seconds into the boot.

**The automatic speculation estimator now sizes the model by the bytes that will be resident**, not by
the whole file: it follows `-ngl` and honours `-ot` host pins, so a partially offloaded or
`-ot`-overridden boot is no longer declined by arithmetic. Proved on the four-P100 seat itself: the
banner reads a 12,611 MiB share against a hand-tuned tensor split whose entries sum to 12,612, and the
gate that used to print "cannot estimate free VRAM" now prints "OK" with 3,284 MiB of headroom. On this
architecture the default drafter is the match-gated table (`ngram-mod`, n_max 4): on ordinary code
traffic it is a wash (it drafts only when it can predict) and on repetitive output it gains; the long
n_max 64 drafter that the other card families use loses 13% here and is not armed.

---

## Fixed

- **The shipped n-gram default could abort with a CUDA out-of-memory at `-np 2` on a 16 GB Volta pair**
  (bug 153): the per-step checkpoint budget sized itself for one slot's snapshot rows while the allocator
  claimed one set per slot, and the over-commitment starved the first real decode graph. One slot was
  never affected. Fixed in the budget (see *Memory*); the effective draft length is printed at boot as
  `effective n_max`.
- **A capped response lost its last verified tokens** — the budget accounting defect described at the
  top of this note. One hunk; output identical, speed unchanged, and up to (step − 1) tokens per
  capped response recovered.
- **A slot erase did not reset a drafter's persistent map**, only its per-generation state, so a
  reused slot kept predicting out of the previous conversation.
- **The automatic speculation layer printed a two-stage cascade and built a one-stage chain.** The
  architecture whitelist never covered this model's MTP head, so a bare command line armed the n-gram
  table alone while the banner said otherwise. Fixed — and every board cell measured before the fix is
  now *relabelled* rather than renumbered: the throughput was real, its description was not.
- **A per-transformer-layer hparam array was sized by the block count**, which counts a grafted MTP
  head, so a correctly-built file was refused one entry short.
- **GLM-5.3-Flash: the last prompt token did not attend to itself** — the indexer's incomplete
  key-pool tail was never selected.
- **The 512/576 attention tile kernel's "depth error" was the device test's own reference**, not the
  kernel: the reference accumulated in fp16 and the generator was all-negative, so the error grew with
  depth for a reason that had nothing to do with the code under test. The test is fixed; the kernel's
  real-logit deviation is a separate open issue, below.
- **The MTP head's graph shape** — the reserve churn above.
- **The KV pool grid verifier had never compared the grid.** A verifier that could not fail is now one
  that can.
- **Two artifact gates were vacuous on a file with no native `pxq4`** — they passed by not executing.
  They now decline loudly.
- **A 16-byte shared load needed a built-in type**, or `ptxas` would not emit it at all.
- **A graph-diff instrument that could not see anything** — it read the real graph *after* the reserve
  had rebuilt it in the same buffer, so all 174 of its diffs compared the reserve with itself, and its
  node signature carried dimensions, which makes every node differ when the two graphs are built at
  different widths. Both fixed; it now names the two nodes behind a coverage failure.

## Packaging

- **`bench/gate/LAST-RUN.md` is no longer shipped.** It is the record of a gate run *on the build
  machine*, so it carried build-machine paths and internal model filenames. Every other file in that
  directory was copied by name; this was the one wildcard, so it shipped whatever happened to be in the
  directory when the package was cut. The gate *tooling* is unchanged and still ships.
- **The packager refuses to tar a package that leaks build-machine paths**, so the next occurrence
  fails the build rather than the audit.

---

## Levers whose default changed in this release

| lever | default | note |
|---|---|---|
| `PXA_PXQ_MMV_H2` | `3` on `cc == 600` exactly, `0` elsewhere | carried forward from the 09-11 candidate |
| auto speculation chain, `cc >= 600` on every visible device | **n-gram stage alone** (`n_max=64`, `n_min=2`, lookback 24, never wiped) | **new** — replaces the n-gram → MTP cascade on every card family from Pascal up (measured on a V100 pair and a P100 quad); `PXA_SPEC_AUTO_CHAIN=cascade` restores the previous default |
| `PXA_FA_GPU_FALLBACK` | `1` (on) | **new** — a declined flash-attention head size builds its unfused chain on the card instead of being placed on the CPU backend |
| the 4× P100 batch cell | chosen by the **file**, not only by the cards | **new** — a dense file resolves to `-b 2048 -ub 256`, an expert file to `-b 2048 -ub 2048`; the old flat answer was a 40–44% prefill loss on dense files |
| `PXA_MTP_STABLE_OUT_IDS` | `1` (on) | one graph shape for the MTP head; output byte-identical |
| `PXA_RESERVE_SHAPE` | `1` (on) | the reserve is built from the batch kind the context decodes |
| `PXA_RESERVE_GUARD` / `PXA_RESERVE_MAX` | `1` (on) / `24` | the thrash guard and its cap |
| `PXA_CKPT_BUDGET` / `PXA_CKPT_BUDGET_MARGIN_MB` | `1` (on) / **`256`** | **changed** — the margin was 1024; 256 survives the largest prompt on both slots of a 16 GB pair and affords 56 / 17 draft tokens at one / two slots instead of 48 / 13 (one-client control 163.57 → 196.47 on the pair) |
| `PXA_CKPT_BUDGET_COMPUTE_PCT` / `PXA_CKPT_BUDGET_FLOOR_MB` | **`25`** / **`512`** | **new** — the budget keeps a quarter of the compute-buffer copy it held for the CUDA pool, never less than 512 MiB in total; two-slot draft 14 → 23 tokens on a 16 GB pair, control 222.10 vs mainline 178.64; 0% dies at one slot, 25% is the fastest safe point on both shapes |

New and **off** by default: `PXA_FA_TILE_512` (measured, held by an open fidelity question — see
*Models*) and `PXA_GEMMA4_ISWA` (measured and now deterministic; experimental this release), `PXA_PXQ_MMV_TOK` (a token-batched dense decode
fold: +10.5% on control decode on a **pair** of P100s, a wash on four P100s and on V100s in this cut's
measurements, so it stays a lever — set it to 2 on a two-card Pascal box), `PXA_REDUCE_PINNED` (a measured loss, kept with its numbers),
`PXA_ATTN_SPLIT_QWEN38` (the attention-split admission), `PXA_RESERVE_DEBUG` and `PXA_REDUCE_TIME`
(diagnostics).

Every one of those has a row in [`docs/lab/LEVERS.md`](docs/lab/LEVERS.md) carrying its default, the
`getenv` site the default is decided at, the window each number came from, and its gate class.

---

## Known behaviour, known issues, and what this release does not claim

**Known behaviour — speculative output is not byte-reproducible at temperature 0.** A verify step
decodes M = 1 + draft rows in a single forward pass; plain decode always decodes one row. Matmul
kernels are selected by batch width, so the same logits are computed by different kernels at different
widths, and where two continuations are a genuine near-tie the argmax can land either way. Measured at
an aligned prefix on four P100s: 0.617 against 0.383 between two candidates, with every other
candidate below 5e-6 — a coin flip, not a wrong accept. It reproduces on two cards as well as four,
and on mainline's `Q4_K_S` file as well as this engine's PXQ4, so it is a property of speculative
decoding on this hardware and not of this codec or this card count. **Consequence for anyone gating a
build: a greedy checksum cannot pass for any speculative arm. Gate speculative configurations on
fidelity — top-1 agreement, KLD, logit spread — and keep byte-reproducibility gates for
`--spec-type none`, where this engine is exact.**

**Known issues carried into this release, each with the configuration it affects.**

- **The 4× P100 speculation multiplier trails the 2× V100 one**, and the cause is a kernel cost rather
  than a defect: a multi-row verify batch on four cards crosses a kernel-selection boundary that a
  one-row decode does not. The n-gram-alone arm above is what to run there meanwhile.
- **`PXA_GEMMA4_ISWA` is experimental and off by default.** Its determinism defect is fixed in this cut
  and gated at one slot (see *Models*); the two-slot gate is what stands between it and the default.
- **At two slots, identical greedy requests can return one of two outputs.** Measured on the 4× P100
  Flash-Next seat at `-np 2` with no speculation armed: six consecutive identical requests alternated
  between two completions, while `-np 1` returned six identical ones. The sampler is ruled out; the
  candidate is the fully-masked-tile skip that only engages when two slots share the cards. A greedy
  sha taken at `-np 2` is therefore not a correctness gate on this release; use `-np 1` for that.
- **`PXA_FA_TILE_512` deviates from an fp32 truth on real 8k logits**, ten times further than the path
  it would replace, despite passing its device test at 1.2e-07. Off by default.
- **The server's prompt cache saves nothing under `PXA_GEMMA4_ISWA` and reports success.** Only
  reachable with that off-by-default lever armed.
- **GLM-5.3-Flash is beta**: one slot per server, `-ub 64` at `-c 24576`, and a logit fidelity
  remainder that is understood but not closed. All four limits are in *Models*.

**What this release does not claim.**

- **No like-for-like codec comparison is claimed anywhere in this note.** The cross-engine tables run
  stock weights on every side, which removes the *weights* confound the earlier candidates carried,
  but they are still PXQ4 against `Q4_K_S` — **codec and engine together**. The decomposition that
  separates them has run on four P100s and **reverses the prefill sign** (mainline is 7.2% and 54.2%
  faster on its own file); on the Volta pair it has not run at all.
- **The 512-token Volta prefill class is quoted from a current window and the long classes are
  narrow.** At 8,192 and 20,801 tokens the lead over mainline is +3.2% and +1.9%. Small, real, and not
  rounded up.
- **The split mode is not this project's invention; this project's version of it is what runs on these
  models and cards.** It is not a default either: the attention split's verify-batch fidelity is
  measured and sits at the flash-attention toggle's level, but it misses the pre-registered
  default-on criterion on two tail statistics, so it ships as a measured option.
- **The adaptive draft length is not shipped as a default.** It is built and measured, and it is red on
  two classes — repetitive text at one slot, and prose at two slots, where it falls below running no
  speculation at all. The section above gives the numbers rather than the conclusion alone.
- **The two-client numbers for Qwen3.8-27B are new and the one that goes against this engine is in the same
  sentence as the two that do not.** The earlier two-client figures were withdrawn — the benchmark
  fired the same prompt on both slots and the drafting state was shared across them — and both causes
  are fixed. Re-measured with distinct prompts per slot and a drafter reset between cells, this engine
  leads on repetition and prose at two clients and **trails mainline on the synthetic control class**.
  For GLM-5.3-Flash there is still no two-client number at all, and the limitation there is
  structural — one slot per server.
- **Optimisation continues in the next release.** GLM-5.3-Flash and Gemma 4 have had correctness work
  and essentially no performance work; the 4× P100 speculation multiplier is a kernel question that is
  now named rather than guessed; and the equal-codec gap is on the board with nothing tried against it
  yet. `docs/NEXT-RELEASE.md` is the scoped list.

---

## Appendix — the artifacts behind the cross-engine numbers

Quoted so a reader can tell which file produced which row, and so a re-run can start from the same
bytes rather than from a model name.

| role | file | md5 | engine build |
|---|---|---|---|
| this engine, stock weights | Qwen3.8-27B, PXQ4 | `11d6dd2a` | `e68d088c28` / `03de222f92` |
| mainline, stock weights, current head | Qwen3.8-27B, `UD-Q4_K_S` | `b3853896` | `4a899373` |
| upstream ik, stock weights, current head | Qwen3.8-27B, `UD-Q4_K_S` | `b3853896` | `3bb386eb` |
| mainline, stock weights, pinned build | Qwen3.8-27B, `UD-Q4_K_S` | `b3853896` | `fdb2c11c70` (2026-06-24) |

All three engines read the same weights, so no cross-engine row here carries a *weights* confound. The
rows still differ in **codec and engine** — one variable fewer, not none. Where a table quotes the
pinned mainline build rather than its current head, it says so on the row. No table here is a codec
measurement; the arm that is one — this engine driving `Q4_K_S` — has run on four P100s only, and it
is published with the rest in [`bench/LEADERBOARD.md`](bench/LEADERBOARD.md).

---

## Credits

Thanks to mistrjirka, a developer on the PXA Network Discord, for his help on this release.

---

## Community

- Discord: <https://discord.gg/EqazvV9tf>
- Support the work (Ko-fi): <https://ko-fi.com/shatteredrealms1>
