# Changelog

This repository is the current home of the PXA engine. Releases before `v2026.09.07-rc1` were
published under the project's previous name at `poisonxa16/pxq_llama.cpp`, which stays online for
history. The engine is the same tree; only the project name changed. **PXQ** remains the name of
the codec — the quantizer and its GGUF tensor types — and nothing about the codec, its file
format or its tensor types changed with the rename.

Full notes for the current release: [`RELEASE-NOTES-2026-09-20.md`](RELEASE-NOTES-2026-09-20.md).

---

## v2026.09.20 — 2026-09-21

Full notes: [`RELEASE-NOTES-2026-09-20.md`](RELEASE-NOTES-2026-09-20.md). If you change nothing,
this release behaves like the last one with three correctness bugs fixed, plus one default change
described below (the launcher now picks the tensor split on a matched pair it has evidence for);
every other switch stays opt-in and is named, with the number it measured, in
[`docs/LEVERS.md`](docs/LEVERS.md).

- **Fixed:** the multi-column verify-path write-back, the prompt-cache token move, and the
  speculative cascade constants. All three could change output; none of them needs a switch.
- **New, now the launcher's default on a matched pair:** a tensor split for two identical cards
  (`-sm tensor`), with the fused all-reduce armed automatically at decode and prefill. `--sm auto`
  (the default) picks it for an architecture, tier and card pair it has hardware evidence for and
  falls back to the layer split cleanly otherwise; pass `-sm layer` yourself to keep the previous
  behaviour. Measured on a V100 pair and a P100 pair, described with the command that arms it.
- **New, on by default on V100s:** a head-256 attention kernel for long-context decode and the
  speculative verify step (`PXA_FA_D256_VOLTA_TILE`), and a q8_0 K/V cache that reaches a faster
  V100 prefill kernel (`PXA_FA_MMA_VOLTA_Q8`). +9–18% long-context decode and +16.7% prefill on a
  V100 pair; see [`docs/LEVERS.md`](docs/LEVERS.md).
- **New, opt-in:** `-sm tensor` for Gemma 4 (`PXA_TSPLIT_GEMMA4`) — 8–21% faster decode on a P100
  pair with `PXA_TSPLIT_REDUCE=fused`, slower on a V100 pair so it is not offered there yet.
- **Gemma 4** dense and MoE files load and run; the sliding-window KV lever and the MTP assistant
  drafter stay opt-in.
- **`pxq-quantize` moved out of this repository** into its own download. The engine still reads and
  runs every PXQ file; `llama-quantize` now refuses a PXQ target and says where the tool lives.
- **Importance matrices are a documented option for the PXQ tiers** (`PXA_PXQ_IMX=1`, default off):
  collected on chat-templated text they improve a PXQ4/PXQ3 file at the same size, correcting the
  earlier net-negative guidance that was measured on raw text.
- **Twelve step-by-step guides** under [`docs/tutorials/`](docs/tutorials/README.md), shipped inside
  the tarball as well.

## v2026.09.13-rc3 — 2026-09-14 (prerelease)

Full notes: [`RELEASE-NOTES-2026-09-13.md`](RELEASE-NOTES-2026-09-13.md). Every figure below is a cell
on [`bench/LEADERBOARD.md`](bench/LEADERBOARD.md) of this release; nothing that is not on that page
appears here. `v2026.09.09-rc3` is the last tag anyone outside this project has seen, so this entry
covers everything since then — including the `v2026.09.11-rc3` candidate, which was cut and gated but
never published.

**The decode default changed on every card family, and it is the headline.** On every card from Pascal
up, a bare command line now arms a **long n-gram stage on its own** — `n_max=64`, a two-token minimum,
a 24-token lookback, never wiped between steps — instead of the n-gram → trained-head cascade. No flag,
no environment variable.

- **Two V100 PCIe 16 GB cards, against mainline `llama.cpp` at its current head running its own best
  cascade, in one bracket at `REPS 6`**: decode **+18.2%** on a synthetic repetitive prompt, **+26.6%**
  on a real repetitive workload, **+69.4%** on free prose. Prefill on the same pair leads by
  **+18.4% / +49.4% / +3.2% / +1.9%** at 512 / 3,121 / 8,192 / 20,801 prompt tokens.
- **At two concurrent clients on that pair it leads all three classes as well** — **+24.3%**,
  **+24.8%**, **+138.2%** — after the fix below. The earlier two-client row is withdrawn on the board
  rather than quietly replaced.
- **Four Tesla P100s**: the table stage alone reads **+107%** on the repetitive control class against
  mainline's best measured cascade; prefill leads **+121.1%** at 3,121 and **+57.8%** at 20,801 tokens.
- **The drafter is not changing the output, only the rate**: the greedy hash of every speculated class
  matches its own unspeculated arm in the same bracket.
- `PXA_SPEC_AUTO_CHAIN=cascade` puts the old chain back.

**Two defects had to be fixed before that default could ship.** A capped response was losing the last
verified tokens of its final step — an early stop, not a wrong token, invisible at draft depth 1 and
obvious at depth 64. And `slots?action=erase` did not reset the drafter's persistent n-gram table, so a
reused slot kept predicting out of the previous conversation.

**Two concurrent clients on 16 GB cards.** The shipped default could not serve a second slot on a 16 GB
pair at all: the checkpoint budget priced one slot's snapshot rows while the allocator claimed a set per
slot. The budget now prices every slot, keeps a **256 MiB** margin instead of 1024, and holds a quarter
of the compute-buffer copy it used to hold whole, with a 512 MiB floor on the reserve. Two-slot draft
length went 13 → 23 tokens.

**Models.**

- **Dense Gemma 4 is supported** — 12B / 31B / E2B / E4B. The architecture guard is now shape-aware and
  refuses only the 128-expert MoE, which still has a heap defect. Six converter defects were fixed on
  the way: Google's released dense weights could not be converted at all before. Its eight 512-wide
  attention layers now run flash attention **on the card** instead of falling to the CPU backend — the
  largest single speed change in this release on any model.
- **GLM-5.3-Flash runs, as a beta.** Coherent on six cards and on all seven, with its limits written
  down rather than discovered later (one slot per server among them). It is the first release in which
  it runs at all, and no optimisation work has been done on it.
- **Qwen3.8-Flash-Next loads again.** A per-layer hparam array was sized against the wrong block count.

**What this release does not claim.** Run this engine on mainline's own `UD-Q4_K_S` file on four P100s
and **mainline's prefill is faster** — 7.2% at 3,121 tokens and 54.2% at 20,801. That equal-codec arm is
on the board in full rather than left out: the Pascal prefill lead is the codec carrying an engine
deficit, not the reverse. And **speculative output is not byte-reproducible at temperature 0** on any
engine, so speculative configurations are gated on fidelity — top-1 agreement, KLD, logit spread — and
byte-reproducibility gates are kept for `--spec-type none`, where this engine is exact.

**Packaging.** `bench/gate/LAST-RUN.md` is no longer shipped inside the tarball — it is a record of a
gate run on the build machine — and the packager now refuses to tar a package that leaks build-machine
paths at all.

Artifacts: a self-contained Linux x86_64 tarball (CUDA 12.8, `sm_60;61;70`, proven booting in a bare
`ubuntu:22.04` container with no toolkit, no `python3` and no `curl`).

---

## v2026.09.07-rc1 — 2026-09-07 (prerelease)

Documentation release candidate on top of `rc1`: **no compiled source changed** between the two,
which is why the A/B table measured on the `rc1` binary is quoted as this tag's own table.
What the documentation now carries that `rc1`'s did not:

- **The speculative rows are measured and published**, on the vLLM sidecar — our own V100 pair
  (`k=7` at 82.18 t/s against 47.02 plain, 1.75×, lossless) and an 8× V100-SXM2-32GB NVLink
  system against a specialist NVFP4 DFlash2 stack, with every row carrying *how it was taken*
  (alternated / single boot / n) in its own column.
- **A before/after section against our own last public release** `v2026.09.02`, including the
  −0.9% P100 decode regression as a stated regression, and the finding that the last public
  release is nondeterministic at temperature 0 on the 1080 Ti where this one is not.
- **What is NOT in this binary, stated plainly**: no `sm_60` token-folded verify kernel, no
  llama-engine DFlash speed row (with the measured reason), no Pascal vLLM Flash-Next seat.
- **Charts** for every headline comparison, generated from CSVs that ship in
  [`docs/data/`](docs/data).

Engine and sidecar content of the release itself:

- **The engine picks its own batch geometry per card.** A four-card P100 host gets
  `-b 2048 -ub 2048` with no flags at all — **+29% prefill @3k**, **+32% @20k** over the previous
  automatic choice, and level with what twelve hand-tuned environment levers used to buy.
- **The DeltaNet out-gate fusion race is root-caused and closed** — the last source of greedy
  nondeterminism on the V100 pair, and the cause of the previous release's six-answers-to-one-
  question behaviour on the 1080 Ti.
- **Release gate raised to 11/11** on both a hybrid MoE and a stock dense GGUF, including a
  token-0 logit-reproducibility arm at `np=1` and `np=2`.
- **Multi-slot admission fairness fixed**: a short chat arriving during a 100k-token prefill on
  another slot went from ~463 s to first token to ~2.8 s, with no measurable cost to the deep
  prefill.
- **DFlash and DFlash2 hand-ported** into the llama-engine line from `ik_llama.cpp`, acceptance
  byte-identical to the reference — **off by default and experimental in this binary**, with no
  speculative speed claim for it.
- **vLLM sidecar rebased onto 1Cat-vLLM v1.5.0** (521 upstream commits): MoE on Pascal fixed
  under custom all-reduce, and a live one-token completion defect on the Volta image fixed.
- **Two new vLLM quant tiers, PXQ2 and PXQ3**, alongside PXQ4 — one layout, three code widths,
  sharing kernels and books. CPU-exact and shipped; their on-GPU window did not run before the
  tag, so they are not a gated speed claim yet.
- **The Volta prefill regression is closed** — the v1.5.0 rebase is at parity on all four cells
  once run the way the seat runs it (compiled, `--block-size 256`).
- **Flash-Next under vLLM on Pascal** serves coherently after a MoE loader fix, and is explicitly
  **not** a speed row: 14.1 / 138 / 131 against the llama seat's 24.6 / 488 / 377 on the same
  four cards.
- **Ten levers measured and shipped OFF**, each written up with the number that killed it.
- Renamed: the project is **PXA**; **PXQ** stays the name of the codec.

Artifacts: a self-contained Linux x86_64 tarball (CUDA 12.8, `sm_60;61;70`, glibc floor 2.35,
proven booting in a bare `ubuntu:22.04` container), the PXQ4 DFlash2 drafter checkpoint, the
engine image `ghcr.io/poisonxa16/pxa`, and the vLLM sidecar images
`ghcr.io/poisonxa16/pxa-vllm:sm70` / `:sm60`.

---

## Earlier public releases

Three releases were published under the project's previous name. Their notes are carried in
this tree.

### v2026.09.02 — [notes](RELEASE-NOTES-2026-09-02.md)

A 24-hour measurement pass across both rigs the project runs day to day — the 4x Tesla P100 rig
(llama.cpp-lineage engine, hybrid MoE, PXQ_UNIVERSAL 4-bit) and the 2x Tesla V100 rig (vLLM-based
`sm_70` serving line, 27B dense-hybrid, PXQ4) — with every number traced to `bench/fair-battle.md`
or `docs/PXA-SM70-SERVING.md` under a fixed protocol (temperature 0, median of 7, one warmup
discarded, unique prompt per repeat). This is the release the "before" column of the current
before/after chart is measured against.

### v2026.08.31

**Unified KV ring (`--kv-unified`, `-kvu`).** By default a server with `-np N` divides its
context into N fixed slices, so a long request fails even when the cache is nearly empty; with
this lever every slot addresses the whole attention-KV ring, backed by admission control that
defers rather than evicts. On a 150k-context seat a single slot processed 86,401 tokens where the
static split capped it at 75,008. Also **elementwise chain fusion** (`PXA_EW_FUSE`, bit-identical
by construction, ~1.6% on a five-card seat, neutral-to-negative on a shared four-card seat, ships
off), Flash-Next MTP work and the mixed-tier recipe, a spec tracer, an async-commit path, and FA
tile f32 accumulation. Its own honest notes: speculative decoding was a net loss on P100 in those
measurements, and KV quantization is a memory lever, not a speed one (`q8_0` halves the cache and
`q4_0` quarters it; neither was faster than `f16` at any prompt length measured, at a cost of
2-16% shrinking as context grows).

### v2026.08.28-rc3 — [notes](docs/RELEASE-NOTES-2026.08.28-rc3.md)

**The first release cut from a single tree.** 47 commits: 19 on the distribution side and 28
carrying the Flash-Next architecture work, which until then lived on a branch with an unrelated
history — so neither its correctness fixes nor its performance levers had been in any release
build. Gated on the real model before the tag: 49/49 layers offloaded, KV exactly 3840.00 MiB,
PXQ2/PXQ3/PXQ6 fused kernels engaged.

---

## Before the first public release

The tree also carries the notes for the internal releases that preceded the first public tag.
They were never published as GitHub releases and are kept for the record:
[2026-08-20-rc1](docs/RELEASE-NOTES-2026.08.20-rc1.md),
[2026-08-11](RELEASE-NOTES-2026-08-11.md),
[2026-08-09](RELEASE-NOTES-2026-08-09.md),
[2026-07-24](RELEASE-NOTES-2026-07-24.md).
