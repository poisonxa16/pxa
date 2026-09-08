# Fair-battle protocol (`bench/fair/`)

The rules every number under `bench/fair/` — and the three headline numbers `run.sh` prints —
must follow. This tightens (does not replace) the methodology already in
[`../fair-battle.md`](../fair-battle.md): same three shapes (engine-only / codec-only / product),
stricter reproducibility rules.

## Rules

1. **Endpoint:** `llama-server` `/completion` (not `/v1/chat/completions` — no chat-template or
   sampler defaults folded in; a raw prompt string in, a raw completion out).
2. **Sampling:** `temperature=0`, `seed=42`. Deterministic by construction — a rerun that doesn't
   reproduce is a bug in the harness or the build, not noise to average away.
3. **Repeats:** **n=7**, **median** reported. **1 warmup run discarded** before the 7 (first
   request after server start pays one-time cuBLAS/cuDNN autotune and page-fault cost that no
   later request repeats).
4. **Fill tokens named:** every reported cell states the exact prompt-token count it was measured
   at (a cold-prefill number at 512 tokens and one at 32k tokens are not comparable — pre-Turing
   attention memory and kernel choice both shift with fill).
5. **MTP:** speculative decode is either **on for both sides of a comparison, or off for both** —
   never on for one arm only. A codec or engine delta must not be a speculative-decode delta in
   disguise (see the README's "Engine-only, the honest number" for why this matters).
6. **Prompts:** every repeat (including the discarded warmup) uses a **unique prompt** of the
   stated token length — never the literal same string n times. A repeated literal prompt lets a
   KV/prompt cache silently turn a decode benchmark into a cache-hit benchmark; `cache_prompt`
   stays `false` for the same reason.
7. **Artifact sha check:** every GGUF used in a `bench/fair/` run must have its sha256 recorded in
   [`weights/MANIFEST.sha256`](weights/MANIFEST.sha256) **before** the run, and `run.sh` refuses
   to start unless `sha256sum -c weights/MANIFEST.sha256` passes against the files present. A
   benchmark run against an unverified file is not reproducible and is not reported.

## The three numbers

Every `bench/fair/` report reduces to three numbers, matching the shapes in
[`../fair-battle.md`](../fair-battle.md):

| number | isolates | shape |
|---|---|---|
| **engine-only** | the kernel/arch fixes | same GGUF, two engines (upstream ik_llama.cpp vs this engine) |
| **codec-only** | the PXQ codec | same engine, PXQ4 vs MXFP4 at matched bytes |
| **product** | what you'd actually run | best documented recipe per side (own quant, own levers) |

`run.sh` prints all three it can produce from the scripts that exist today
(`../speed-bench.sh`, `../measure.py`) and prints **"not automated yet"** for any number those
scripts cannot produce under this stricter protocol (in particular: neither script currently
speaks raw `/completion`, discards a warmup rep, or enforces unique-prompt-per-repeat — see
`run.sh`'s own comments for exactly which numbers that limits today). A missing number is
reported as missing, never backfilled with a number measured under a looser protocol.
