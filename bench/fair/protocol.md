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

`run.sh` measures all three itself, under the rules above — it does not hand off to
`../speed-bench.sh` or `../measure.py`, neither of which speaks raw `/completion`, discards a
warmup rep, or enforces a unique prompt per repeat. Any cell whose artifact or binary this repo
cannot name prints **`pending`**, and the run then exits **2** rather than 0, so a partial run
cannot be mistaken for a complete one. A missing number is reported as missing, never backfilled
with a number measured under a looser protocol.

Two shapes stay **out** of the three blocks, on purpose:

- an expert-codec comparison whose two sides are not the same base weights (the MoE decode row
  against MXFP4 in [`../fair-battle.md`](../fair-battle.md) is that: same architecture and size
  class, a different model). A rig file declaring `PRODUCT_SAME_BASE_WEIGHTS=no` is refused as a
  product row rather than printed with a footnote;
- multi-box, NVLink and sidecar-speculation rows. Different hardware or a different serving stack
  is a different table, not a fourth column in these three.

## How to reproduce the three README blocks

```bash
cd bench/fair
./run.sh --rig 2xv100 --plan     # exactly what it would run, starting nothing
./run.sh --rig 2xv100            # the three blocks
```

`--rig <name>` reads `rigs/<name>.env`: the models, flags and card indices of one machine. Three
ship — `4xp100`, `2xv100`, `1x1080ti` — and copying the closest one is how you add your own box.
Every value in a rig file is either taken from a file in this repo (the comment beside it says
which) or the literal word `pending`, which is what an unknown stays until someone measures it.

The engine-only block needs a second engine binary. The container image carries one at
`/opt/pxa/bin/upstream-ik-server`, built from the upstream commit every published comparison in
this repo was measured against and labelled with it (`org.pxa.upstream.ik.sha`); `run.sh` looks
there by default, and `UPSTREAM_BIN=/path/to/your/build` points it at your own build of that same
commit. Without it the block prints `upstream binary not present: <path>` and the run exits 2.

Stdout is the three blocks and nothing else — `./run.sh --rig <rig> > blocks.txt` gives you
something a README can paste — while progress and the reason behind every missing cell go to
stderr. Rule 5 is enforced rather than trusted: each arm's speculative-decode state is read from
its own startup log and command line, and two arms that disagree void the block and exit 2.
