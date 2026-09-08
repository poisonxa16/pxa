# ONE BUILD, ONE TRUTH — standing policy (2026-07-24)

There is exactly **one** working tree and **one** build. Everything runs from it.

- **Canonical tree**: this repo, branch `main`. **Binary**: one build directory, `build-unified`.
- ALL development, benchmarking, serving, and gauntlets run from this one build. No parallel worktrees,
  no per-experiment build dirs, no side-branches that outlive the day. Commit to `main` (or a same-day
  branch that merges straight back).
- The unified build already contains everything: Laguna / Cohere2-MoE / Gemma-4 archs, PXQ1/2/4/6 + PXQU
  codecs, all speed levers, the correctness ports, and community credits.
- Superseded/experimental builds are **archived off the working tree**, never kept live on the cards.

**Why**: an 8+ worktree sprawl on divergent bases was the root cause of a painful cross-history merge.
"Latest" must mean exactly one thing. Archive, don't fork.
