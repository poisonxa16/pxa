# How everything is structured now (2026-07-24)

## The one engine

**`poisonxa16/pxa`** (public, standalone) — THE engine and THE product. One codebase,
one name, built on the ggml/llama.cpp code base (full lineage and credits in [`NOTICE`](NOTICE)). Contains everything:

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

A private backup remote carries the same code plus the generator internals that are not
published (`pxqu_wrel.py`, `pxqu_golden.py`, the bulk generated artifacts). The documented
reference budgets under `pxa-bench/pxq-universal/` — the `.tiers` files, `tier-maps.json` and
the determinism-gate scripts — DO ship here; see `.gitignore` for exactly what stays out.
One tree -> two remotes: `origin` = public clean, the other = dirty backup.

## The law

One working tree, one build, everything runs from it. Archive, don't fork. See [docs/ONE-BUILD-POLICY.md](docs/ONE-BUILD-POLICY.md).
The 8+ worktree / 2-repo sprawl was the root cause of the cross-history merge pain; it does not recur.
