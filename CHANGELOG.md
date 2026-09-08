# Changelog

This repository is the current home of the PXA engine. Releases before `v2026.09.07-rc1` were
published under the project's previous name at `poisonxa16/pxq_llama.cpp`, which stays online for
history. The engine is the same tree; only the project name changed. **PXQ** remains the name of
the codec — the quantizer and its GGUF tensor types — and nothing about the codec, its file
format or its tensor types changed with the rename.

Full notes for the current release: [`RELEASE-NOTES-2026-09-07.md`](RELEASE-NOTES-2026-09-07.md).

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
