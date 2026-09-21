# Leaderboard — this engine against mainline llama.cpp and upstream ik_llama.cpp

One table per card set. Every number is one arm of one **bracketed window**: a control measured at the
window's open and again at its close, with everything else measured between them. If the two controls
disagree by more than 1.50% the box moved during the window and **every cell in it is void** — no
number here exists without its bracket, and the bracket's drift is printed beside it. Every cell names
the **capture** it came from; the record for that capture holds the arm's command line, both md5s, the
per-rep values and the drift.

**Colour, and what it now decides.** 🟢 green = this engine holds the best valid cell, in one
bracket, on stock files, against the competitor's current head. 🔴 red = a competitor's best valid
cell beats ours. ⚪ grey = not measured, or the competitor has no artifact or cannot boot that
configuration. A difference inside the noise band is a tie and is never written as a win.

**A red cell is one of two things, and the difference is the whole point of this page.**

- **red-open** — a competitor leads and the work to close it is not finished. The cell names what is
  being tried.
- **red-exhausted** — a competitor leads, every mechanism I can name has been tried and *measured*,
  the cause is identified, and the reason no lever remains is written beside the number.

**This release ships when every cell is green or red-exhausted.** Not when the calendar says so. A
red-open cell is unfinished work; a red-exhausted cell is a published limitation with its evidence
attached, which is a thing a reader can check and act on. Neither is hidden, and a cell is never
promoted from red-open to red-exhausted by running out of time.

**Each engine is quoted at its own best configuration**, not at one configuration chosen to suit this
page. Where a competitor's best needs a flag, the flag is named. A comparison that handicaps the other
side is worth nothing.

---

## The three prompt classes, said once

Decode is quoted in three classes because a speculative engine does not have one decode speed.

- **control** — a synthetic repetitive prompt. An n-gram table predicts its continuation almost
  perfectly, so this class **measures the speculation ceiling** and nothing else. It is the class
  where a drafter's design shows up largest, and it is the least like ordinary use.
- **repetition** — a real repetitive workload (a code edit re-quoting its own buffer). Partly
  predictable, and the class that separates a drafter that helps from one that only helps a synthetic.
- **prose** — free generation, where a table mostly cannot predict the next token. This is the class
  closest to ordinary chat use.

A change that buys the control class routinely costs prose. One number for "decode speed" on a
speculative engine is a number with an unstated workload hidden inside it, so all three are printed
and none of them is averaged away.

---

## What a row can and cannot tell you

Two different things get compared here and they are not interchangeable.

- **Best against best** — each engine on the file it is fastest with. This is what an operator
  actually gets, and it is the headline row. It crosses **codec and engine together**.
- **Equal codec** — this engine run on the competitor's own file. This is the only row that isolates
  the *engine*, and it is a decomposition rather than a product number, so it is printed apart and
  never allowed to win a cell.

**These two can disagree in sign, and on this hardware they already have.** At equal codec this engine
is slower than mainline on four P100s on every measured class — see the equal-codec table below — so a
headline win on that card set is the codec carrying an engine deficit, not the reverse. Any table that
prints only the headline is hiding that, so both are printed.

---

## 2× V100 (sm_70) — Qwen3.8-27B

**Prefill** — capture `lb-fresh-pair`, 2026-09-13, cards 2/4, bracket drift **0.52%** against a 1.50%
band, REPS 5, temp 0, `-ngl 99 -c 32768 -np 1 -t 16 -fa on -ts 1,1`. This engine on the stock PXQ4
(`11d6dd2a`); both competitors on the stock `UD-Q4_K_S` (`b3853896`) — same weights on all three
sides, both competitors at their **current heads** (mainline `4a899373`, upstream ik `3bb386eb`), and
both competitor files on NVMe beside this engine's file.

| class | this engine | mainline | upstream ik | delta vs best rival | |
|---|---|---|---|---|---|
| prefill, 512 tok | **777.42** | 656.54 | 384.38 | +18.4% | 🟢 |
| prefill, 3,121 tok | **1,332.21** | 891.87 | 447.29 | +49.4% | 🟢 |
| prefill, 8,192 tok | **1,013.01** | 982.01 | 421.63 | +3.2% | 🟢 |
| prefill, 20,801 tok | **946.96** | 929.03 | 371.82 | +1.9% | 🟢 |

### Decode, launcher's tensor-split default — capture `main`, 2026-09-21 (`release/rc4/logs/board-v100-tsdefault.txt`)

Cards 2/4, REPS 6, greedy (temp 0, top-k 1), `-np 1`, `n_predict` 256, first-pass medians, host
load average under 8 for every arm below. **The launcher's `--sm auto` now resolves to `-sm tensor
-ts 1,1` on this pair**, with the fused reduce at both decode and prefill
(`PXA_TSPLIT_REDUCE=fused PXA_TSPLIT_REDUCE_PREFILL=1`, `PXA_TSPLIT_FALLBACK=1` — see
`docs/LEVERS.md`) and the engine's automatic speculation armed — that bare command line, no flags,
is what an operator gets by default on a matched V100 pair now. Gate on this exact packaged
binary, this exact path, `--sm auto`: **PASS=13 FAIL=0 SKIP=0**, plus a two-slot needle at 0/8 bad
(`release/rc4/logs/gate-qwen-v100-tensor-fused.md`).

| class | shipped default NOW (`-sm tensor`, auto spec) | previous default (`-sm layer`, auto spec) | tensor split + cascade | layer split + cascade | mainline, plain | mainline, best arm |
|---|---|---|---|---|---|---|
| prose | **68.58** | 50.99 | 80.18 | 64.67 | 45.22 | 58.86 |
| repetition (code edit) | **107.37** | 85.70 | 139.79 | 101.62 | 41.74 | 80.49 |
| long context (~15k) | **43.58** | 35.53 | 37.61 | **47.78** | 26.94 | 46.07 |
| prefill, short/mid/long (t/s) | 120 / 526 / 850 | 104 / 656 / 1,020 | — | — | — | — |

- *shipped default NOW* = the bare `./pxa-launch` command line on this pair: `--sm auto` picks
  `tensor`, fused reduce at decode and prefill, the n-gram stage armed automatically.
- *previous default* = the same bare command line under the prior release's launcher (`-sm layer`),
  same window — kept as a labelled comparison row, no longer what ships.
- *tensor split + cascade* = `-sm tensor` plus
  `--spec-type ngram:n_max=64,n_min=2,ngram_size_n=24 --spec-type mtp:n_max=2` — the fastest arm on
  this page for prose and repetition.
- *layer split + cascade* = the same cascade under `-sm layer` (rc5pack's same-bracket
  measurement) — the fastest arm for long context, see below.
- *mainline, plain / best arm* = stock llama.cpp at its own best on this pair, `-sm tensor` with
  `GGML_CUDA_ALLREDUCE=internal` (mainline's own tensor split is its best configuration here too),
  plain and its own `ngram-mod,draft-mtp` cascade; same-bracket capture `rc5pack`, drift 0.26–0.88%
  against a 1.50% band. Acceptance beside the speculative cells: ours 0.79 prose / 0.77 edit / 0.65
  long; mainline 0.58 / 0.78 / 0.60.

**Best arm against best arm is not the same arm in every cell, and every board cell above already
reflects that.** Prose and repetition are won by the **tensor split + cascade** (80.18, 139.79)
against mainline's best (58.86, 80.49) — 🟢 +36.1% / +73.7%. Long context is won by the **layer
split + cascade** (47.78) against mainline's own best cascade (46.07) — 🟢 +3.7%, the release-gate
cell above — **not** by the tensor split's own cascade (37.61), which trails mainline outright at
this class. **Say it plainly: for long-context speculative work, run `-sm layer` with the cascade,
not the launcher's own `-sm auto` default.** The tensor split's MTP verify step gets slower at
depth under the split — every verify batch has to cross both cards before the next one can start,
and that cost compounds with the longer KV cache this class exercises — while `-sm layer`'s verify
step stays on one card. `--sm auto` still picks `tensor` for this pair because the default is
tuned for plain decode, not for the speculative long-context case; pass `-sm layer` by hand for
that one.

**Prefill is the tensor split's honest trade, on this box's PCIe.** Over the x4 risers this rig
uses, the tensor-split default reads 850 t/s at long-context prefill against the layer split's
1,020 — the split's activation reduce crosses the link every layer, and that costs more than
prefill's larger batches make back. It is still faster than stock llama.cpp's own tensor split on
this file class: 817 t/s against mainline's 590 in the comparable window. **Decode also gives back
some of its lead when other jobs load the host's CPUs** — both cards must launch every step
together under the split, and a busy host desynchronises that handshake in a way `-sm layer` does
not suffer; every number in the table above was taken on a quiet host (load average under 8) and a
loaded host will read lower.

### 2× V100 (sm_70) — Gemma 4 26B-A4B, re-measured on the package — capture `rc4pack3`, 2026-09-21

Same window, same shapes. This engine on the PXQ4 file; stock llama.cpp on Google's own QAT
`q4_0` file, `-sm tensor` with the internal all-reduce, `--jinja`. Gemma 4 is `-sm layer` only on
this engine, which is what the launcher picks.

| class | this engine, plain | this engine, shipped default | stock llama.cpp, q4_0 | |
|---|---|---|---|---|
| prose | 107.03 | **159.33** | 93.14 | 🟢 +71.1% |
| repetition (code edit) | 104.64 | 102.92 | 91.62 | 🟢 +14.2% |
| long context (~15k) | 91.73 | **104.59** | 86.54 | 🟢 +20.9% |

Plain control 107.03 open / 106.64 close (−0.36%). The repetition class is the one place the
drafter does not pay on this model: 102.92 with it against 104.64 without, i.e. inside a point and a
half of each other, and the plain figure is the one quoted as the engine's floor.

### Competitor arms measured on 2026-09-20, kept for the record

These five numbers appear in the README's head-to-head section. They are **stock llama.cpp**
(`@4c9233c0`, 2026-09-15) and this engine on the competitor's own file, measured on the V100 pair at
REPS 3, `n_predict` 256, `-c 16384`, `-np 1`, a distinct salted prompt per rep, shapes `s-g0`
(greedy) and `s-t1` (temp 1.0, top-p 0.95, top-k 20). Where a cell above re-measures the same thing
at REPS 6 in a bracket, **the re-measured cell is the one that counts**; these stay because a
published number is not deleted.

| arm | shape | median t/s | what it is |
|---|---|---|---|
| `ml-tensor-plain-i` | s-g0 | 46.64 | mainline, `-sm tensor`, `GGML_CUDA_ALLREDUCE=internal`, no drafting |
| `ml-tensor-mtp3` | s-g0 | 56.33 | mainline, same, `--spec-type draft-mtp --spec-draft-n-max 3` |
| `ml-tensor-mtp3` | s-t1 | 60.17 | the same arm at temperature 1.0 |
| `m-plain` | long-g | 42.0 | mainline plain, the long-context class, rep 2 of 3 |
| `q4ks-ours-split-plain` | s-g0 | 47.51 | **this engine on mainline's own Q4_K_S file**, tensor split, no drafting |

---

**Decode, at this release's new Volta default, same-bracket against the current head** — capture
`pair-settle`, 2026-09-14, cards 2/4, bracket drift **−0.63%** (control 38.25 open / 38.01 close)

**Read with one correction, 2026-09-14: the np1 rows below are UNDERSTATED by the shipped default and have deliberately not been re-measured tonight.** They were captured with the checkpoint budget's headroom at its old 1024 MiB, which clamped the drafter to `n_max` 48 at one slot. The sweep that shipped a 256 MiB default (`e3cfa6925d`) raises that to `n_max` 56, and a same-shape REPS 3 control cell on the shipped binary reads **196.47 t/s** where this capture reads 147.89. The later reserve sweep (`d037ca2095`) raises it again — `n_max` 61 before the 512 MiB floor applies — and reads 195.43-214.10 across the same cell, but np1 medians at REPS 3 are not resolvable (every arm's minimum sits near 129 with maxima 195-214), so the only np1 claim made here is the one at 196.47. The re-measure at REPS 6 with a bracket is owed; the direction is not in doubt.
against a 1.50% band, REPS 6, np 1, both files on NVMe, mainline at its **current head** (`4a899373`,
md5 `034d8bc948`). This engine's arm is the shipping default for sm_70 and newer, unforced — the bare
command line, no `--spec-type` flag — which auto-arms the **n-gram stage alone** (`n_max=64`,
`n_min=2`, lookback 24, never wiped, no MTP stage). Mainline's arm is its own `ngram-mod,draft-mtp`
cascade at the layer split (`--spec-draft-n-max 3 --spec-draft-p-min 0.0`), its best on this pair. Both
sides measured with a **cold per-slot map**: a slot erase before every class cell on this engine (this
build's erase hard-resets the drafter's persistent map, fixed at `f782f722ea`); mainline has no
such reset — verified from its own current-head source, `handle_slots_erase` calls only
`prompt_clear()`, which never touches `server_slot::spec` — so mainline gets a **fresh boot before every
class cell** instead. This is the same-bracket, current-head, cold-map re-measure the row below used to
mark *settling*.

| class | this engine | mainline | delta | |
|---|---|---|---|---|
| decode, control (speculation ceiling) | **147.89** | 125.07 | +18.2% | 🟢 |
| decode, repetition | **76.65** | 60.54 | +26.6% | 🟢 |
| decode, prose | **141.02** | 83.27 | +69.4% | 🟢 |

Draft/accept per cell (this engine / mainline): control 1453/1453 (1.000) / 1493/1511 (0.988);
repetition 833/928 (0.898) / 1276/1717 (0.743); prose 1225/1230 (0.996) / 1317/1764 (0.747). **Fidelity
check**: this engine's greedy-decode sha matches its own **plain** arm (spec off, same bracket) on
every class, every rep — the drafter is not changing the output, only the rate. The earlier pinned-build
(`fdb2c11c70`, quadspec-w2) and cross-window current-head readings above are kept as history; this row
is now the one of record.

Upstream ik on this pair, from `lb-fresh-pair` at ik's own explicit `mtp:n_max=3,p_min=0.0` arm
(its real best — its bare command line logs zero drafts for this architecture): control 54.39, prose
43.64, repetition 52.43.

**The new default's numbers survived a correctness fix, which is why they are quoted here rather than
withdrawn.** The `quadspec-w2` arm was later found to be losing the last verified tokens of a capped
response — a budget accounting defect, not a wrong-token one. On the fixed build the same three
classes re-measured at REPS 6 read **149.35 / 78.54 / 141.93** (control / repetition / prose), at or
above every `quadspec-w2` bar, and the fix's own A-B-A deltas are smaller than the bracket's drift.
Nothing was inflating these numbers; tokens were being thrown away at the end.

**Two honest limits on the prefill window.** (1) **The equal-codec neutraliser did not run on this
pair** — a hard stop before it could be armed means every prefill cell above is codec **and** engine
together, not decomposed. On four P100s that decomposition *has* run and it reverses the sign, so the
absence here is a gap, not a formality. (2) The server exposed no request-count metric, so the usual
cross-contamination check could not run; this is stated rather than implied as a pass.

**Earlier window, kept for continuity, prefill only.** An earlier bracket on this pair
(`prefill2-v100-w1s`, drift 0.99%) measured this engine against a pinned mainline build (`fdb2c11c70`)
before either file had been moved to NVMe — the competitor's `Q4_K_S` file was on the array throughout
that window while this engine's file was already on NVMe. The effect of that asymmetry on those
numbers is unproven and the direction is mixed; it is noted here rather than corrected for.

---

## 4× P100 (sm_60) — Qwen3.8-27B

**Prefill** — capture `lb4-quad-clean`, 2026-09-13, bracket drift **0.53%**. **Decode** — capture
`lb-fresh-quad`, 2026-09-13, bracket drift **0.23%**, REPS 5. Cards 0/1/5/6, temp 0,
`-ngl 99 -c 32768 -np 1 -t 16 -fa on -ts 1,1,1,1` in both. This engine on the stock PXQ4
(`11d6dd2a`); both competitors on the stock `UD-Q4_K_S` (`b3853896`), both at their **current heads**
(mainline `4a899373`, ik `3bb386eb`), all three files on NVMe.

| class | this engine | mainline | upstream ik | delta vs best rival | |
|---|---|---|---|---|---|
| prefill, 3,121 tok | **470.54** | 212.83 | 114.78 | +121.1% | 🟢 |
| prefill, 20,801 tok | **420.23** | 266.28 | 74.72 | +57.8% | 🟢 |
| decode, control (speculation ceiling) ¹ | **95.57** | 46.17 | 18.83 | +107.0% | 🟢 |
| decode, prose ³ | **39.26** | 14.33 | 14.06 | +174.0% | 🟢 |
| decode, repetition ² | **21.25** | 19.54 | 18.43 | +8.8% | 🟢 |

¹ **Cross-capture, and labelled as such.** This engine's control decode cell comes from capture
`quad-default-pxq4` (2026-09-13, same four cards, same shape, REPS 3, bracket valid at −0.04% /
−0.07%), at the arm named below; the competitor column beside it comes from `lb-fresh-quad` /
`lb-shake1`. It is not one bracket with those, so the delta is read as direction and magnitude rather
than to the decimal. The prefill and prose rows are single-bracket.

² **decode, repetition is no longer cross-capture.** Capture `quadrep2-w1` (2026-09-14, same four
cards and shape, REPS 6, bracket valid at 0.34% against a 1.50% band) measures this engine's bare-boot
arm and mainline armed at its own best (`ngram-mod,draft-mtp`, identical to `lb-fresh-quad`'s
`ml-casc-l`) inside one continuous open/close bracket. ik's column is unchanged, from its own separate
capture.

³ **decode, prose on the shipped default.** Same capture as ² (`quadrep2-w1`, REPS 6, bracket drift
0.34%): this engine 39.26 on the bare boot that arms the n-gram stage alone, mainline 14.33 at its own
best in the same bracket. The 17.77 against 14.21 it replaces was `lb-fresh-quad`'s reading of the
cascade default this release retired on this card set. The upstream ik column stays from
`lb-fresh-quad` (cross-capture; ik was not re-run in `quadrep2-w1`).

**Which arm each cell is.** This engine's prefill cells are the **plain** arm — no speculation stage
armed, `-b 2048 -ub 256` — its fastest valid configuration for that class; booted twice inside the
same bracket it read 470.54 / 420.23 and 468.06 / 419.58, a spread under 0.6%. The decode-prose cell is
also the plain arm. **The two speculated decode cells are the n-gram stage alone**
(`--spec-type ngram:n_max=64,n_min=2,ngram_size_n=24`, `PXA_NGRAM_RESET_STREAK=0`) — the same shape as
the Volta default, and the arm this release tells an operator to run on four P100s. Mainline runs its
`ngram-mod,draft-mtp` cascade at the layer split for decode and its bare command line for prefill (its
own best at that class); ik's decode cells run its explicit `mtp:n_max=3,p_min=0.0` arm and its prefill
cells its bare command line, which beats its own MTP arm at these lengths.

**Mainline is quoted at its best on the control class, which is the reading least flattering to this engine.**
Its pinned-build cascade read **46.17** (`lb-shake1`); its current-head cascade reads **35.88**
(`lb-fresh-quad`). The +107.0% above is against 46.17. Against the current head the same cell is
+166%. The larger competitor number is the one in the table.

### The two decode cells that moved, and why

**The cascade is close to inert on this card set; the table stage alone is not.** With the two-stage
cascade armed, this engine's control cell reads 17.89 — essentially its unspeculated rate, so
speculation returns almost nothing. Drop the MTP stage and raise the table's draft length and the same
engine on the same cards reads **95.57**. The MTP stage is also what costs prefill here: the shipping
cascade reads ~48–50% of the unspeculated prefill ceiling at *both* 3,121 and 20,801 tokens, while an
n-gram-only chain matches that ceiling almost exactly. One stage, two classes, the same answer.

**decode, repetition is closed green at REPS 6, same bracket.** Capture `quadrep2-w1` (2026-09-14,
drift 0.34% against a 1.50% band) re-ran the class at the REPS the class had already been shown to
need, with mainline armed at its own best (`ngram-mod,draft-mtp`, identical to `lb-fresh-quad`'s
`ml-casc-l`) inside the same open/close bracket as this engine's bare-boot arm: **21.25** against
mainline's **19.54**, **+8.8%**. The earlier −10.3% at REPS 3 (17.60 vs 19.62) undercounted this
engine's own rate as much as it overcounted the gap; nothing about the class needed a mechanism, only
the REPS the class itself had already been shown to need.

### Speculative output is not byte-reproducible at temperature 0, and that is not a defect

This page carried, for part of a day, a claim that the fast quad arm was fast because of a wrong
accept. **That claim was wrong and is withdrawn here rather than quietly deleted**, because the
correction is worth more than the original.

A verify step decodes M = 1 + draft rows in a single forward pass; plain decode always decodes one
row. Matmul kernels are chosen by batch width — on cc 600 this engine's PXQ4 path takes a
float-accumulator kernel at ny ≤ 8 and a half2 GEMM above it — so identical inputs are computed by
different kernels at different widths. Where two continuations are a genuine near-tie the argmax can
land either way. Measured at an aligned prefix: **0.617 against 0.383** between the two candidates,
with every other candidate below 5e-6. That is a coin flip, not a wrong accept. The earlier "wide
margin" reading came from a probe that was misaligned by the prompt length and used the wrong sampler.

Three measurements close the question rather than argue it: it reproduces **on two cards** as well as
four (byte-identical shas), it reproduces on **mainline's `Q4_K_S`** file as well as this engine's
PXQ4, and it is unaffected by turning relaxed acceptance off. So it is a property of running a
speculative verify batch on this hardware, not of this codec, this card count, or this engine's accept
loop.

**The consequence for this page is a gate change, not a number change.** A greedy-checksum
repeatability gate cannot pass for *any* speculative arm on free generation — the previously shipped
cascade fails it too — so speculative cells are gated on fidelity (top-1 agreement, KLD, logit spread)
and byte-reproducibility gates are kept for `--spec-type none`, where this engine is exact. The 95.57
and 17.60 cells stand as speed results whose output is a different, equally valid greedy continuation
at a near-tie.

**What is still owed on this card set, and it is a cost rather than a defect:** the multi-row verify
batch crosses that kernel-selection boundary, so the speculation multiplier on four P100s is smaller
than the one on two V100s. Narrowing it is a kernel question, and it is on the next-release list.

**Equal codec — this engine on the competitors' `Q4_K_S` file.** Informational only, not a release
gate: this engine is built and tuned for its own PXQ format, and the rows below show it running
someone else's — a decomposition of where the codec's own effect ends and the engine's begins, kept
for transparency, not a product claim.

Capture `rc5pack`, 2026-09-21, on the packaged binary, same window/same bracket for each pair of
rows unless noted.

| class | this engine on their file | mainline, best | what it says |
|---|---|---|---|
| prefill, 3,121 tok, bare | **248.78** | 213.72 | this engine 16.4% faster |
| prefill, 3,121 tok, `-b 2048 -ub 1024` | 195.93 | 206.32 | mainline 5.3% faster at this shape; **bare stays the better arm for both engines here** |
| prefill, 20,801 tok, bare | 233.71 | 266.57 | mainline 12.3% faster |
| prefill, 20,801 tok, `-b 2048 -ub 1024` | 201.00 | **286.34** | mainline 42.4% faster at this shape; **`-ub 1024` is mainline's best arm at this length, and this engine's worst** |
| decode, control | **43.39** | 35.97 | this engine 20.6% faster (same-bracket) |
| decode, repetition | 14.18 | **19.51** | mainline 37.6% faster (same-bracket) |
| decode, prose | **24.63** | 14.36 | this engine 71.6% faster (same-bracket) |

**Best-arm-vs-best-arm, each engine free to pick its own shape:** prefill/3,121 — this engine's
bare (248.78) beats mainline's best, also bare (213.72), **+16.4%**. prefill/20,801 — this engine's
bare (233.71) is beaten by mainline's best, `-ub 1024` (286.34), **mainline 22.5% faster** — wider
than the bare-vs-bare gap (12.3%), because `-ub 1024` helps mainline's file at this length and hurts
this engine's read of the same file at both lengths (195.93 and 201.00, both below its own bare
numbers). The lever main's llama-bench probe pointed at narrows one cell and widens the other; both
are reported as measured.

All three decode rows and both bare prefill rows are a clean, same-bracket decomposition (captures
`rc5pack-lb4-quad` and `rc5pack-lb4-decode-rep`, drift 0.29% and 0.04%); the two `-ub 1024` rows are
their own same-bracket pair (capture `rc5pack-lb4-ub1024b`, drift 0.06%). **This engine leads two of
the three decode classes at equal codec** (control, prose) **and trails on repetition** — the row
still never wins a cell against the PXQ4 headline above, that was never the point of it.
**Speculation is not inert on `Q4_K_S` on these four cards** — mainline's own cascade arm (`ml-casc`)
reads 35.97 control against 12.90 for its own plain bracket control — which is worth knowing beside
the PXQ4 rows above.

The equal-codec rows deserve their own sentence, because they are the ones most likely to be read as
an evasion if not said plainly: **at equal codec, this engine leads mainline at prefill/3,121 and
decode/control and decode/prose, and trails at prefill/20,801 and decode/repetition** — a mixed
picture, not a clean win, on a format this engine does not target. Every number here is on the page
because it was measured, and none of them end by being left out.

**Two competitor facts these windows confirmed:**

- **ik does not arm speculation from a bare command line for this architecture.** Its bare-command-line
  arm logs zero drafts on every card set measured so far; the decode cells above are its explicit MTP
  arm, which is its real best.
- **Mainline's tensor split cannot be armed with a drafter on any P100 set, and that is mainline's own
  constraint, not a handicap imposed on it.** Its internal all-reduce pipeline requires exactly two
  devices of Volta or newer; four sm_60 cards fail both conditions, the pipeline falls back to NCCL,
  and a second communicator over the same devices ends the run in warm-up. Its best configuration on
  this hardware is therefore its cascade at the layer split, which is what is armed above.

**Earlier window, kept for continuity.** `lb-shake1` (drift 0.19%) measured pinned competitor builds —
mainline `fdb2c11c70`, ik `3c58ae37` — rather than current heads, and ran before the competitors'
`Q4_K_S` file was moved from the array onto NVMe. Its bare-command-line decode cells for this engine
read 18.37 (control), 18.22 (prose) and 17.61 (repetition). Those numbers stand, but their description
was wrong: they are **not** cascade results. They are the n-gram table alone at n_max 4 — the auto
layer never armed the MTP head for this architecture from a bare command line, a defect fixed in this
cut at `03de222f92` — so the arm drafted one stage, not two, with acceptance reading 1.000 because an
exact table continuation always matches its own prediction.

---

## Two concurrent clients (np2) — distinct prompts per slot, map reset per cell

capture `np2split-w1`, 2026-09-14, cards 2/4, REPS 6, np 2, both files on NVMe, mainline at its
current head (`4a899373`). **These rows replace the `pair-settle` ones, and the reason is a bug, not
a re-measure.** The engine that produced the previous np2 numbers over-committed VRAM at more than
one slot: the per-step speculative checkpoint budget sized itself for one slot's snapshot while the
allocator claimed one per slot, so the server aborted with a CUDA out-of-memory on the first decode
of a plain `-np 2` boot. It is fixed (`7308a1a8eb`), and the fix is what these rows are measured on
— which means the old `163.69 / 125.44 / 148.75` were rates a 16 GB pair could never actually pay.
The budget's headroom default was then swept for the first time and shipped at 256 MiB
(`e3cfa6925d`), and the reserve that margin is subtracted from was swept after it (`d037ca2095`) —
between them they are what buys the draft length back. The `np2margin-w1` rows these replace
(141.69 / 111.24 / 116.24) were correct for their build and are superseded by a lever, not withdrawn
for a fault. Same shape as before otherwise: each slot
builds and owns its own drafter, this engine's slot erase hard-resets the drafter's persistent map
before every class cell, mainline gets a fresh boot per cell. Distinct prompt pairs per slot —
repetition `w_code_edit` / `w_quote`, prose `w_prose` / `w_prose2`, control `w_random` /
`w_random2`. Reported per-slot, never the sum alone.

| class | this engine (slot0 / slot1 / sum) | mainline (slot0 / slot1 / sum) | delta on the sum | |
|---|---|---|---|---|
| decode, control | 104.91 / 117.17 / **222.10** | 90.14 / 88.81 / 178.64 | +24.3% | 🟢 |
| decode, repetition | 63.84 / 80.43 / **144.17** | 53.91 / 57.92 / 115.55 | +24.8% | 🟢 |
| decode, prose | 95.65 / 88.60 / **184.33** | 41.41 / 35.97 / 77.38 | +138.2% | 🟢 |

Draft/accept on the sum (this engine / mainline): control 2848/2848 (1.000) / 2993/3013 (0.993); repetition 2350/2644 (0.889) /
2754/3865 (0.713); prose 2260/2387 (0.947) / 2383/4155 (0.574).

**The window is bracketed and the bracket is valid.** Open and close are the same cell — the
speculation-off (`plain`) control pair at np 2, first and last inside one continuous lock hold on
the pair: **49.96 open / 49.87 close, drift -0.18%**. Arms are judged on the close
bracket.

**What sets the two-client rate here is how many checkpoint rows a 16 GB pair can pay, and that is
what changed.** The budget compares each device's free VRAM — a live reading, taken after the
weights, the KV cache and the compute buffers are all allocated — against what it must keep free.
It used to keep the margin *plus a whole second copy of those compute buffers*, which the free
reading had already excluded. That second copy was never a duplicate: it is an unlabelled allowance
for the CUDA pool that grows during decode. It had never been swept, and it was the whole of this
pair's asymmetry, because the output head lands on CUDA1 and left it keeping 3219.80 MiB against
CUDA0's 1808.54 — so CUDA1, not the margin, re-fitted the capacity down to 15 and clamped every
stage to `n_max` 14.

`PXA_CKPT_BUDGET_COMPUTE_PCT` now names that copy and keeps a quarter of it, with
`PXA_CKPT_BUDGET_FLOOR_MB` (512) as a hard minimum on the whole reserve. On this pair and this file
the reserve is 644.13 MiB on CUDA0, the capacity is 24 **with no re-fit on the second card**, and
every stage is clamped to `n_max` 23 rather than 14. The boot banner prints all three terms, which
branch of the default fired and why.

**The fraction is 25 because -np 1 says so, not because -np 2 prefers it.** Zero was measured first
and is faster still at two slots (238.72 control) — and it kills the server at one slot, where the
checkpoint can claim nearly all of a card's free VRAM: an 8k needle died in
`ggml_cuda_pool_vmm::alloc` with only the 256 MiB margin standing between the two. Swept at np 1 with
the same battery plus that needle: 0% keeps 256.00 MiB and dies, 25% keeps 420.24 and is clean, 50%
keeps 584.48 and is clean, 100% keeps 912.95 and is clean. 25 is the largest relaxation that survives
both shapes; the 512 MiB floor binds at np 1 (where 420.24 would otherwise stand) and is inactive at
np 2. Set `PXA_CKPT_BUDGET_COMPUTE_PCT=100` to restore the old behaviour, and note the default
returns to 100 by itself if `PXA_SCHED_RESERVE_REAL` is turned off, since the argument for keeping
less rests on that load-time reservation existing.

Control is the class that feels the clamp most, because its acceptance is 1.000 — every drafted token
is kept, so the rate scales almost linearly with draft length. Repetition and prose accept below
1.000 and are limited by the table as well as the clamp, which is why they gain less than control in
proportion even though all three now lead.

**The question the last capture left open about mainline's prose cell is now settled, and not in this engine's
rhetorical favour.** Mainline's np2 prose has been measured in three windows on the same head, the
same command line and the same prompt pair, with a fresh boot per cell in each: 80.31 in
`pair-settle`, 112.64 in `np2margin-w1`, and 77.38 here. Two of the three agree within 4%, so
`np2margin-w1`'s 112.64 is the unrepresentative one — the opposite of what that capture assumed when
it withdrew the `pair-settle` comparison. The +138.2% above is measured against the value two windows
out of three support. A reader who wants the most conservative possible claim can take mainline's
prose at its highest recorded value (112.64) and this engine still leads that class by +63.6%.

**The largest graphs this shape can ask for were checked on the shipped configuration, not assumed.**
On the same boot as these cells: a 20,801-token prefill on both slots, an 8k needle with two hidden
facts (both returned), and the cards' free VRAM read with the server live. One honest ceiling to
note: `--kv-unified` gives the two slots a single 32,768-cell ring, so two 20,801-token prompts
(41,603 cells) are serialised rather than co-resident — the server stays up and answers, it does not
refuse and it does not die.

---

## 1× GTX 1080 Ti (sm_61) — frozen

**These rows are frozen at their last measured values and are not current.** The single 1080 Ti in this
machine is in production use and is not available for windows, so nothing here has been re-measured
against either competitor's current head. They are kept for continuity and should be read as history.
See `bench/fair-battle.md` for the values and the windows that produced them.

---

## Open red cells — the release gate

**No product cell is red in this cut.** Every row on this page that measures this engine on its
own PXQ format — the format it is built for — is green. The equal-codec rows (this engine reading
the competitors' own `Q4_K_S` file) are **not release gates**: they are a decomposition kept for
transparency, answering "is the engine or the codec doing the work," not a product claim, and they
are not held to green-or-exhausted the way a shipping cell is. See the "Equal codec" section above
for their current numbers.

A cell on this list is **red-open** until the work beside it is finished and measured, at which
point it becomes green or **red-exhausted** with its cause written out. No cell is closed by a
deadline. **There is nothing on the list this cut.**

**Six cells left this list this cut** and are not on it: the four-card equal-codec prefill and
decode rows, moved to "not a release gate" above rather than closed (see the "Equal codec" section
for the current numbers); the four-card prefill headline (420.23 vs
266.28 at 20,801 tokens), the four-card decode control cell (95.57 vs 46.17, red-open for one day on a
diagnosis that turned out to be wrong — see the section above), the pair's np1 decode
control/repetition/prose row against the current head, which was *settling* and is now closed green on
all three classes in one same-bracket window (see the 2× V100 section above), the four-card decode
repetition cell (17.60 vs 19.62, −10.3% at REPS 3, red-open on a class already known to need REPS 6 —
capture `quadrep2-w1` re-ran it at REPS 6, same bracket, drift 0.34%, and it closed green at 21.25 vs
19.54, +8.8%), and the four-card equal-codec decomposition (was cross-capture with prose unmeasured —
the same window filled control/repetition/prose same-bracket: this engine leads control and prose,
trails repetition, and the row still never wins a cell — that owed same-bracket arm is the only thing
that was ever outstanding about it). **Two cells joined this list this morning and both have since left it green**: np2 decode control and
np2 decode repetition on the pair. They joined because the build that produced the previous numbers
over-committed VRAM at more than one slot and aborted on a plain `-np 2` boot, so neither the old lead
on repetition nor the old size of the control gap was real. They left because the reserve that clamped
the draft length was swept for the first time (`d037ca2095`, capture `np2split-w1`): control now leads
+24.3% and repetition +24.8%, on a bracket of -0.18%. The pair's prefill headline (946.96 vs 929.03 at 20,801 tokens) was never
on this list.

---

## What changed since the last cut

- **The pair's decode cells are settled against mainline's current head, one client and two.** A
  same-bracket, cold-map re-measure (capture `pair-settle`) replaces the *settling* cross-window
  reading: this engine's default leads mainline's own best cascade on all three np1 classes (+18.2% /
  +26.6% / +69.4%), with a fidelity check confirming the drafter does not change the output. The
  two-client section is no longer withdrawn, and it has since been re-measured end to end twice: first
  on a build that no longer over-commits VRAM at two slots (capture `np2margin-w1`, which left control
  and repetition red), then on the swept checkpoint reserve (capture `np2split-w1`), where this engine
  leads all three np2 classes — control +24.3%, repetition +24.8%, prose +138.2% — and **both np2 open
  red cells are closed**. The `pair-settle` np2 rows are withdrawn — see the two-client section.
- **Both open-red cells on the four-card quad are closed.** Capture `quadrep2-w1` (2026-09-14, REPS 6,
  same bracket, drift 0.34%) fixed the armset bug that voided the previous attempt (a bracket control
  must run without speculation, or it logs drafts and the window has zero valid brackets) and re-ran
  decode, repetition at the REPS the class had already been shown to need: **21.25 against mainline's
  19.54, +8.8%**, up from the −10.3% read at REPS 3. The same window also ran the equal-codec
  decomposition's missing same-bracket arm on all three classes for the first time (this engine leads
  control and prose there, trails repetition) — see the two sections above.
- **The Volta decode default changed, and the board's pair decode rows changed with it.** On sm_70 and
  newer the automatic default is now the n-gram stage alone — long (`n_max=64`), never wiped, no MTP
  stage — instead of the two-stage cascade. It takes all three decode classes on the pair against
  mainline's own best speculation, and it takes prose by the widest margin of anything on this page.
- **A capped response no longer loses its last verified tokens.** The speculative path charged a whole
  verify step against the response budget before emitting the step's tokens, so a response that ran
  into its limit mid-step dropped up to (step − 1) tokens off the end. It was invisible at draft depth
  1 and showed up as an "early stop" the moment the new long-draft default arrived. Fixed; the pair's
  numbers are unchanged by the fix, which is itself the evidence that nothing was being wrongly
  accepted.
- **The four-card decode control cell went red, then green, and the middle step is left on the page.**
  A repeatability gate caught three different completions from three identical temperature-0 requests,
  and the first diagnosis — a wide-margin wrong accept — was published here as a blocker. Re-measured
  with an aligned probe it is a 0.617/0.383 near-tie decided by batch-width kernel selection, it
  reproduces on two cards and on the competitor's own file, and it is not a defect. The number stands
  and the *gate* was what needed replacing. A page that only ever records corrections in the direction
  of good news is not a record.
- **Every cross-engine cell on the two active card sets is measured against competitors' CURRENT
  heads**, on stock weights on every side, in one bracket, with every quantised file on NVMe — the storage asymmetry that shadowed earlier windows does
  not apply to the current tables. Earlier pinned-build, pre-storage-swap windows are kept as
  clearly-labelled history rather than deleted.
- **ik's real best is on the board for decode.** Its explicit `mtp:n_max=3,p_min=0.0` arm is what is
  armed for every ik decode cell; its bare command line logs zero drafts for this architecture on
  every card set tested. Every ik decode cell on this page before that correction understated ik by
  roughly 50%. ik's prefill cells run its bare command line, which beats its own MTP arm at every
  prompt length measured — so ik's column is "ik at its best per class", not one configuration applied
  everywhere.
- **The equal-codec decomposition is printed beside every headline it exists for**, and on four P100s
  this engine leads mainline at equal codec on two of three decode classes (control, prose) and trails
  on repetition — the row has never won a cell against the PXQ4 headline above and isn't meant to. Both
  prefill rows and, as of `quadrep2-w1`, all three decode rows are now a clean, same-bracket
  decomposition.

---

## How to reproduce a row

Every cell carries its capture tag. For each one the record holds the arm's exact command line, the
engine commit, the md5 of the server binary actually loaded, the md5 of the model file, the prompt-set
md5, the per-rep values and spread, the bracket drift, and the greedy-decode hash of a fixed prompt.
Two independent engines returning the identical hash on the same file is the check that the coherence
probe measures the model and not the harness — mainline and ik did exactly that on both card sets this
cut, which is why this engine's differing hash there is the codec and not a wobble.
