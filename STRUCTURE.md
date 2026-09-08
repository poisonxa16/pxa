# How everything is structured now (2026-07-24)

## The one engine

**`poisonxa16/pxa`** (public, standalone) — THE engine and THE product. One codebase,
one name, in the `llama.cpp` -> `ik_llama.cpp` -> `pxa` lineage. Contains everything:

- Engine + all architectures (Laguna, Cohere2-MoE/North, Gemma-4, GLM, qwen35moe, deepseek, ...)
- PXQ codecs (PXQ1/2/4/6) + PXQU universal mixed-tier maps
- All speed levers (spec1row, router-fuse, enhance, spec-smalln, cublas-eager, volta-cublas,
  mtp-lazy-warmup, fuse-deltanet)
- All correctness ports (WMMA-K6, dmmv-OOB, fused-MoE indices, norm-BF16, graph-v2, + 26 upstream)
- Release packaging, docs, community credits (bradrlaw, Last-Guitar-5924)

**Box working tree:** the working tree (one pxq tree per machine).
**Branch:** `main`. **Binary:** `build-unified`. Everything — dev, bench, serve, gauntlet — runs
from this one build. Build with `--runtime=nvidia` (needs libcuda for the CUDA-driver-API link).

## The private mirror

**`poisonxa16/pxa-private`** (private) — same code + the secret universal-quant recipe
files (`.tiers`, `tier-maps.json`, `pxqu_wrel.py`, `pxqu_golden.py`, `pxa-bench/pxq-universal/`).
The recipes are `.gitignore`d so they never reach the public repo; on the private remote they are
force-added. One tree -> two remotes: `origin` = public clean, `private` = dirty backup.

## Upstream

**`ikawrakow/ik_llama.cpp`** (external) — the upstream we pull improvements from. Add as a remote to
`pxa` when cherry-picking upstream PRs; fold them straight into `main`. No staging fork.

## Retired / legacy

- **`poisonxa16/ik_llama.cpp` (the old fork) — DELETED 2026-07-24.** Content fully folded into
  `pxa`. Its campaign branches are bundled at
  `ik_llama-campaign-branches.bundle` in the offline archive.
- **A local legacy ik_llama tree** — kept ONLY because the live brains still
  run its `build-mmfast`. It retires the day the brains move onto the `pxa` unified build
  (a validated brain-engine swap; owner go required). After that, one engine, period.
- **Historical builds** live on the array: `<archive>/`
  (old worktrees, experiment binaries, the campaign bundle, MANIFEST.md). Never on the cards' cache.

## The law

One working tree, one build, everything runs from it. Archive, don't fork. See ONE-BUILD-POLICY.md.
The 8+ worktree / 2-repo sprawl was the root cause of the cross-history merge pain; it does not recur.
