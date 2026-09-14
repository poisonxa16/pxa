# Known issues

## Fixed 2026-09-04 — non-reproducible logits on hybrid (qwen35 / qwen3next) models under `-sm layer`

**Symptom.** Greedy (`temperature 0`) generation on a hybrid DeltaNet model split across two GPUs
occasionally produced a different completion for the same prompt — roughly 1 request in 4 on a
20,801-token prompt and 1 in 12 at `-np 2`, more often when the host was busy. Shorter prompts
looked perfectly stable.

**What it actually was.** The output shas were the tip of it. Asking the server for `n_probs 2` on
an identical prompt showed the *probabilities* differing on essentially every run in every
configuration, including the ones that passed 12/12 byte-identical: the first token's top-1
probability ranged over 0.4964–0.5339 in 48 runs at 3,121 tokens. The greedy text was stable only
where the top-2 margin was wider than that wobble.

**Cause.** `PXA_FUSE_DELTANET` bit 1, the out-gate fusion, folds `FUSED_RMS_NORM` and
`FUSED_MUL_UNARY(SILU)` into one kernel. That kernel runs one block per row and reads its row
twice — once for the RMS reduction, once for the store — with only a per-block `__syncthreads()`
between them. Unfused, the boundary between the two nodes is a grid-wide barrier, so every read
finishes before any write. Fused, when the allocator places the product on top of one of the
inputs at a **shifted** base, one block's store overwrites another block's not-yet-read input. The
outcome then depends on block scheduling, which is why it needed a long prompt, two cards and a
busy box to show up. Exact aliasing was always safe; only a shifted overlap raced.

**Fix.** Bit 1 is out of the default (`PXA_FUSE_DELTANET` default 55 → 53), and
`pxa_try_deltanet_outgate()` now declines on a shifted overlap using the same predicate the
`ADD`+`FUSED_RMS_NORM` fusion has used since 2026-09-02. It cost nothing to remove: −0.6% prefill,
+0.5…1.0% decode, inside the run-to-run spread.

**If you are on an older build.** `PXA_FUSE_DELTANET=53` in the environment is the complete
mitigation and needs no rebuild.

**How to check your own build.** `bench/gate/run-gate.sh` check 3b, or by hand: six identical
`/completion` requests with `n_predict 1`, `n_probs 2`, `temperature 0`, `cache_prompt false`, and
compare every returned probability. They must be identical to the last digit.

## Query-time sparse attention: reusing one selection for several tokens is deferred, not shipped

The mechanism that makes this lever worth having on long context — select the attended cells once
and reuse the selection for the next several tokens instead of re-selecting every token — is not
in this release. It cannot be added as a small change: dropping the selection/score/top-k/expand
chain for a token that is reusing a previous selection removes roughly 28 graph nodes per
sparse-attention layer, and this engine treats a change in node count as a signal to re-reserve
every compute buffer and rebuild the graph a second time. Doing that on every token that *does*
still need a fresh selection would cost more than the mechanism saves. A real design needs two
graph shapes and a cheap way to pick between them per token; pricing that switch is the next step,
not a rewrite of the selection itself. What ships instead is the selection machinery's own
overhead work — see `docs/lab/LEVERS.md`'s `PXA_QSA_GRID_INCR` / `PXA_QSA_FAST_TOPK` rows.

- **Status:** open; deferred rather than half-built. The lever still selects densely every token
  when it engages at all, gated off below a fill floor where dense attention is simply cheaper.

## A gate-multiply fusion does not fold its preceding bias-add above one token

An existing graph fusion collapses a gate's sigmoid (or SiLU) and the multiply that consumes it
into one kernel launch, and the machinery behind it already supports a multi-row batch — the
kernel indexes correctly for any number of tokens. The **graph builder** does not take that arm,
though: three shared-expert-gate call sites gate the fused form on the batch being exactly one
token wide, a stricter condition than the kernel actually needs. Above one token — every prefill
chunk, and every speculative-decode verify batch — each of those sites pays two launches (the
nonlinearity, then the multiply) where the fused kernel would do one, on every mixture-of-experts
layer. Decode is unaffected; it is already one token wide and already takes the fused arm.

Separately, and not the same gap: the fusion has never folded a *preceding* bias-add into the
same launch, on any batch width. That half only matters for a model whose feed-forward gate/up
carries a bias tensor, which neither shipped seat model does, so it is recorded rather than built.

Closing the multi-token half is not a free launch-count reduction to flip on: the fused kernel
computes the sigmoid and the multiply in one rounding step instead of two, which is the same
mathematics but not bit-identical to the unfused pair. A fix needs the full determinism gate at
both server topologies plus a needle check, not a bit-exactness proof.

- **Status:** open, scoped, not built. `PXA_FUSE_SIBLINGS_CENSUS=1` (see `docs/lab/LEVERS.md`)
  reports how often this exact `ADD` → fused-gate pattern occurs on a real graph, if you want to
  size it before touching it.

## MTP under two concurrent requests: a lazy-warmup step refuses a free draft token rather than guessing

Fixed as of this release, in the sense that the previous behaviour is gone: an MTP draft step used
to ask for the target model's hidden state at a given batch row without checking whether that row
was still addressable, and under a specific two-slot shape — one slot mid-prefill on a long
ubatch while another is verifying — it could silently get back the hidden state of a *different*
token, because that particular batch shape stores hidden rows packed by output position rather
than by the position they actually came from. That silent wrong-answer path is closed: the
addressing rule now refuses instead of guessing whenever the buffer shape does not support the
request.

What is left, and is not a bug: the refused case still costs something. When it fires, the MTP
draft loses its free carried-over token for that cycle — an error is logged
(`MTP hidden state is empty during speculation`) and the draft re-seeds from a fresh decode
instead, which is correct but not free. It only fires under the specific concurrent-prefill shape
described above, and only while that shape persists.

- **Status:** correctness fixed; the residual cost is accepted, not chased, in this release.

## Two defects found while chasing the hybrid re-entry `NaN`, left unfixed on purpose

Reading every unified-cache cell mutation while root-causing the attention-window sizing defect
(see `RELEASE-NOTES-2026-09-09.md`) turned up two more real issues, neither of which
could be tied to that failure and neither of which is fixed in this release — shipping an
unrelated fix alongside the one that was actually proven was judged worse than reporting both and
moving on:

- The scan behind `llama_kv_cache_seq_rm` resets a cell's source-sequence tag (`cell.src`) in a
  way that has not been checked against every caller's expectations.
- The server's prompt-cache lookup (`server_prompt_cache::load`) moves the token list out of
  **every** candidate entry it searches while looking for the best match, not only the one it
  ultimately picks — it is harmless today only because the default code path that consumes the
  result copies rather than relying on the (now-emptied) candidates it did not choose.

- **Status:** open, both. Neither has a reproduction tying it to observed bad output; both are
  latent until someone finds the call shape that reaches them.

## PXQ2 / PXQ3 on vLLM: the kernels are intentionally behind PXQ4's own tuning

PXQ2 and PXQ3 now convert and serve on the vLLM sidecar (`docs/VLLM.md`), through the same tiered
dispatch PXQ4's mixture-of-experts modules already used. Their linear-module kernels deliberately
do **not** carry the split-partials scratch-arena tuning PXQ4's own kernels use on the V100
serving path — that machinery exists to avoid warp starvation at PXQ4's specific matrix-vector
launch shapes, and the header comment for the PXQ2/PXQ3 kernels states plainly that there is no
equivalent starvation to fix at the shapes these two tiers actually run. This is a scoping
decision, not an oversight, and it means a PXQ2/PXQ3 linear module should not be expected to match
a PXQ4 module's tuning depth cycle for cycle at this point in the project.

- **Status:** working as scoped. Revisit only if a PXQ2/PXQ3-specific starvation pattern is ever
  measured.

## `--pxq-universal` quantize: harmless CUDA-driver noise + argument order

**Two things trip people up when building a PXQU (`--pxq-universal`) quant. Neither is a real bug.**

**1. Harmless container noise.** Running `llama-quantize` inside an `nvidia/cuda` container prints,
before any real output:

```
ERROR: driverInitFileInfo 578 result=11
ERROR: init 664 result=11
ERROR: init 250 result=11
```

This is emitted by the **NVIDIA container runtime's own driver probe**, not by `llama-quantize` —
the string does not exist anywhere in our source or binary. Quantization proceeds and completes
normally. Ignore it. (It also appears on a plain `--help`, and on any other quant type; it is not
specific to `--pxq-universal`.)

**2. `--pxq-universal` is a flag — put it before the positional filenames.** `llama-quantize`
parses option flags only until the first positional argument. If `--pxq-universal <map>.tiers`
comes *after* the input/output paths, `--pxq-universal` lands in the positional `type` slot and
you get:

```
main: invalid ftype '--pxq-universal'
```

Correct order (flag first, then `in out PXQ_UNIVERSAL`):

```bash
llama-quantize --pxq-universal my-16gb.tiers \
  model-bf16.gguf model-PXQU16.gguf PXQ_UNIVERSAL
```

The argument is a path to a `.tiers` map; see `docs/PXQU-CONVERT.md` for the format.

## The same three `ERROR: ... result=11` lines before `llama-server` starts

Same cause as above, same verdict: harmless. Inside an `nvidia/cuda` container the runtime's
driver probe prints

```
ERROR: driverInitFileInfo 578 result=11
ERROR: init 664 result=11
ERROR: init 250 result=11
```

and then `llama-server` prints its own first line, which in a container is

```
PXA_CONTAINER_AWARE_v1: runtime=container (/.dockerenv present (docker))
```

That second line is not an error either. It is the server saying it detected Docker, so that a
wedge or a fatal GPU fault will make it exit (codes 41/42) and let the container restart policy
bring it back, instead of sitting there answering health checks with a dead GPU. Bare-metal runs
print `runtime=host` instead. `PXA_IN_CONTAINER=0` or `=1` overrides the detection.

If the server stops after that line, the real reason is in the lines that follow it (model path,
VRAM, `-ngl`, or the `-ot` regex). Please paste from the banner to the exit when reporting.

## llama-imatrix crashes on CPU / partial-offload configs (pre-existing; fix pending)

**Symptom:** an imatrix capture run (`llama-imatrix`, or any `cb_eval`-based activation capture)
on a configuration that keeps some expert tensors on CPU — `-ngl < 99`, `--n-cpu-moe N`, or a pure
CPU run — hangs in a futex wait or dies with SIGSEGV / malloc corruption partway through the pass.
A control run *without* the capture callback on the same config is fine, and the same capture run
**fully GPU-resident (`-ngl 99`, all experts on GPU) works correctly**.

- **Scope:** the activation-collection path in the CPU backend for MoE expert tensors; inherited,
  not introduced by the PXQ kernels (reproduces with the PXQ envs unset). Plain generation and
  perplexity on partial-offload configs are unaffected.
- **Workaround:** run imatrix captures full-GPU-resident. For models too big for your card(s) at
  full residency, capture on a smaller-tier quant of the same model (importance statistics are
  approximately quant-independent) or on a multi-GPU `-sm layer` split — just keep `-ngl 99`.
- **Status:** upstream fix pending; tracked here so nobody burns a day rediscovering it.

## PXQ on CPU and under partial offload: not every tier is fast there

PXQ models now run on CPU and under partial offload, dense and MoE both. `-ngl 0`, `-ngl < 99`,
and `--n-cpu-moe N` are all fine, including through the CPU backend's fused MoE op. That is a
change from an earlier release, where a PXQ model with any expert layers left on the CPU aborted
at the first expert op; the abort is gone.

What differs by tier is speed, not correctness:

- **PXQ4 and PXQ4-HQ** get an AVX2 integer dot product built for the format
  (`ggml/src/pxq-dot.c`) on both the plain and fused-MoE CPU paths.
- **PXQ2, PXQ3, and PXQ6** fall back to a correct panel dequant (`ggml/src/pxq-cpu.c`): it works,
  it is not tuned for speed.

- **Consequence:** for a CPU-heavy or partial-offload deployment, prefer a PXQ4-family tier if
  you have the choice; for a fully GPU-resident deployment, pick the tier that fits your card
  entirely (see the README tier table); multi-GPU `-sm layer` splits are fine either way.
- **Also affects imatrix capture:** the CPU/partial-offload imatrix crash described in the entry
  above is a separate, pre-existing bug in the activation-collection path, not a PXQ property.
  It still applies regardless of quant type, so imatrix capture still needs full GPU residency.

## PXQ tensors and mainline gguf-py

Not a bug in this fork, but a standing trap: no gguf-py size table (mainline's or this fork's) can
express the E16-row per-row anchor, so a gguf-py read-modify-write **silently truncates** PXQ
tensors. Re-run `llama-quantize` from the bf16/f16 source instead. (See README.)

## deepseek2 / MLA (GLM-4.7-Flash class): -fa posture is cc-aware, not one-size-fits-all

MLA-attention models (gguf arch `deepseek2`) degrade **catastrophically** with context on
most silicon when flash-attention is off — community-measured on a P40: 37 t/s at low fill
collapsing to 3.3 t/s by 36k ctx; `-fa on` fixed it outright. The compressed-KV (MLA) path
without fa re-materializes full attention matrices whose cost grows with fill.

**But that P40 is sm_61 (GTX 10-series class) — a second community P40 measurement found
the MLA fa-on path itself is 75-326% SLOWER decode on sm_61 than fa-off, and the gap widens
with context.** sm_61 is fp16-starved (1:64 rate) and the MLA flash-attention kernel leans
on fp16, so the "fa on always" advice from the first measurement does not hold there. Root
cause reconciled: fa-off is bad on deepseek2 in general (re-materialized attention), but on
sm_61 specifically the fa-on kernel is even worse (fp16-starved), so fa-off ends up the net
win on that arch alone.

Since 2026-07-23 `llama-server`'s posture layer is **per-device-arch aware**:

- **`-mla 3` always defaults** on a `deepseek2` gguf, on every arch, when the CLI leaves it
  unset (unaffected by device cc — this part of the story hasn't changed).
- **`-fa` default depends on the visible CUDA fleet's compute capability:**
  - **Every visible device is cc 610 (sm_61 / P40, GTX 10-series)** → `-fa` defaults **OFF**
    (the measured-correct posture there), and the `PXA_MODE=max` fa-off-ingest exception does
    **not** force it back on. Logged: `PXA posture: arch=deepseek2 on sm_61 — fa default OFF
    (fp16-starved FA path; measured 75-326% slower decode with fa on), mla=3`.
  - **Any sm_60 (P100, full-rate fp16) or sm_70+ (Volta+, tensor cores) device present** (incl.
    a mixed fleet with an sm_61 card in it) → `-fa` defaults **ON** as before — including under
    `PXA_MODE=max` (logged: `PXA posture: mode=max but arch=deepseek2 — fa kept ON (MLA
    requires it)`).
- **Explicit `-fa on` / `-no-fa` / `-mla` always win**, in both directions, on every arch.
- **The fa-off warning is suppressed on an all-sm_61 fleet** (fa-off is the recommended config
  there) but still fires everywhere else:

```
WARNING: deepseek2/MLA with -fa off degrades severely with context; use -fa on -mla 3
```

Heed it on sm_60/sm_70+. On an all-sm_61 fleet, fa-off is not a warning-worthy state — it's
the default for a reason.

## Gemma 4: dense runs, the sparse MoE and the MTP assistant still fail clean at load

**Dense Gemma 4 runs on this engine** — `gemma4` with `n_expert = 0`. What was measured is the 12B
(`google/gemma-4-12B-it`, both Google's QAT `q4_0` GGUF and a `Q8_0` converted here from the bf16
release). The 31B is the same graph at a different size and is expected to work but was not run;
the small per-layer-embedding **E2B and E4B take a different path through the builder and are
untested** — they load, and nothing more than that is claimed. Two variants still refuse, and the
error now says which one you hit:

```
PXA runtime does not support this 'gemma4' variant (128-expert sparse Gemma-4 MoE); use stock llama.cpp. Dense Gemma-4 is supported. See docs/KNOWN-ISSUES.md.
```

- **The sparse MoE** (`gemma-4-26B-A4B`, 128 experts) — a real defect, not caution. On the affected
  expert config it corrupted the heap (`double free or corruption` on batches >= 1024 tokens) and
  fell back to roughly 8 t/s scalar decode. The refusal happens at graph-build time, which
  `llama_build_graph` runs at context creation inside a `try`/`catch`, so it surfaces as a clean
  load-time error rather than a corrupted process. **Workaround: run it on stock llama.cpp.**
- **The MTP assistant drafter** (`gemma4_mtp`, the `-it-assistant` files) — no known defect; it has
  simply never been measured here, and a path this engine has not measured does not ship enabled.
  **Workaround: run the dense target without a drafter.**

Demoted rather than refused: `-sm graph` and `-sm attn` fall back to `-sm layer` on any Gemma-4
file, with a warning. A graph-parallel (tensor-parallel) builder exists for this architecture but
no Gemma-4 file has been measured through it. `-sm layer` is the default, so this only affects an
explicit request.

Chat works out of the box: Gemma 4 ships its own jinja template, and the built-in template map has
no row for it — its turns are `<|turn>` / `<turn|>`, not Gemma 3's `<start_of_turn>`, so the
built-in `gemma` row would be wrong rather than merely missing. `llama-server` and `llama-cli`
therefore turn the jinja path on for this architecture by themselves and say so on stderr; an
explicit `--jinja` or `--chat-template` still decides, and `PXA_AUTO_JINJA=0` declines with a
reason. Before this, a launch with no flags logged `chat template parsing error` and formatted the
conversation with a template that was not the model's.

### What was measured, on `gemma-4-12B-it-qat-q4_0`

One Tesla P100 (sm_60) unless the row says CPU; stock llama.cpp `82dbc4f01` built from source is
the comparison engine. A Qwen3-0.6B-Q8_0 control was taken in the same session for every
cross-engine number, because this engine and stock are not byte-identical on any architecture.

| Gate | Result |
| --- | --- |
| Loads and serves | yes, on CPU and on one card; the boot line names the architecture, the expert count and every lever it turns on or declines |
| Greedy identity vs stock (3 prompts x 64 tokens, temp 0, CPU) | 2/3 byte-identical — the same rate the Qwen3-0.6B control gave in the same session; through `/v1/chat/completions` 2/3 against a control of 1/3 |
| Greedy identity, card vs CPU vs stock | prompt 1 byte-identical across all three; prompt 3 on the card reproduces stock's CPU answer exactly |
| Retrieval (a fact in the first line, asked for at the end) | answered correctly at **3,121 / 8,267 / 21,358 prompt tokens** |
| Perplexity vs stock (wikitext-2, `-c 2048`, 10 chunks, CPU) | ours 835.34 ± 52.31, stock 845.35 ± 52.90 — ratio **0.988** |
| Prefill batches >= 1024 tokens (`-b 2048 -ub 2048`) | clean: three rounds of 1,024 + 2,048 + 3,121-token prompts, server alive, no heap diagnostic in the log |
| Determinism, single slot (`-np 1`, 7 prompts, 3 boots) | **7/7** byte-identical |
| Determinism, two co-resident slots (`-np 2`) | **2/7** — see below |
| Every lever on vs `PXA_REFERENCE=1` | 2/3 byte-identical; the one difference has the reference build reproducing the CPU and stock answer |
| Speed, `llama-bench`, `-fa 1` | pp512 **207.1 ± 7.5 t/s**, tg64 **17.1 ± 1.8 t/s** |
| Speed, through the server | decode 29.8 t/s at short context, 9.1 t/s with 8k in the slot |

Retrieval has to be asked for the way the model is meant to be used. Gemma 4 is a turn-structured
model with a thinking channel: hand it a long document as **raw text** through `/completion` and it
degenerates into a repeating token instead of answering — and so does stock llama.cpp on the same
file, at a shorter length than it takes to break this engine. The numbers above come from
`/v1/chat/completions` with the model's own template, and with an answer budget large enough for
the model to finish thinking first: at 160 tokens it is still reasoning and the answer channel is
empty; at 1,200 the answer itself contains both identifiers.

### Concurrent slots are not bit-reproducible on Gemma 4

With two co-resident slots (`-np 2`), the same prompt at `temperature=0` can return different
completions in the two slots, and different completions from the single-slot reference: **2 of 7
prompts identical, against 7 of 7 at a single slot**, with the slots erased before every request so
every completion starts at the same KV placement. It reproduces with **every lever off**
(`PXA_REFERENCE=1`), so it is not one of this engine's optimisations — it is structural on this
architecture and is under investigation. Single-stream serving is unaffected. If you need
bit-reproducible output from Gemma 4 today, run it at `-np 1`.

### Size `-c` to what you actually need

This engine allocates the KV cache for **every** layer at the full context. Gemma 4 runs a
1,024-token sliding window on 40 of its 48 layers, so those layers need a window's worth of cache
and are given a whole context's worth instead: **336 KiB per token**, or 1,344 MiB at `-c 4096`,
where an engine with a per-layer window cache uses 544 MiB for the same model and context. Two
things follow, and both are measured:

- On a 16 GB card the practical ceiling is around 24k tokens for the 12B, not the 262k the model
  supports.
- Prefill cost tracks the **allocated** context, not the prompt: the same prompts run at 122 t/s at
  `-c 4096`, 68 t/s at `-c 18432` and 28 t/s at `-c 22528` on one P100. Flash attention is a wash
  either way (122.3 t/s with `-fa on`, 123.4 with it off).

A per-layer window cache is the fix and is not in this release.

### One number that is reported and not interpreted

Scored against stock llama.cpp's own logits, this engine's mean KL divergence on Gemma 4 is
**0.146** on the QAT `q4_0` file at `-c 2048`, **0.159** on the same file at `-c 1024`, and
**0.203** on a `Q8_0` file at `-c 1024` — top-token agreement 88.4–88.8% in all three. It is flat
in context, flat in quantization type, and unexplained. Two plausible explanations were tested and
**both are wrong**: it is not the sliding window (it does not change between `-c 1024`, where every
scored token has less than a window of history, and `-c 2048`, where every scored token has more),
and it is not stock's CPU weight-repacking path (it is no smaller on a `Q8_0` file, which stock
does not repack, than on the `q4_0` file, which it does).

Read it next to the regime it was taken in. **Both engines score this model above PPL 500 on
wikitext** — raw or wrapped in the model's own turn structure, `Q8_0` or `q4_0`, this engine or
stock. A model that uncertain will disagree with a second implementation about the top token
roughly one time in nine without either being wrong. Corpus metrics for Gemma 4 are recorded here
because they were measured, not because they are diagnostic; the quality evidence is the identity
rate against the control, the retrieval results, and the perplexity **ratio** of 0.988 against
stock on the identical file.

### A PXQ file for Gemma 4

`gemma-4-12B-it` converted from Google's bf16 release to `Q8_0` and quantized from there. Every
row below is scored against that `Q8_0`'s own logits, same corpus, card and binary, so the rows
are comparable to each other:

| File | Bytes | PPL | ln(PPL/base) | Mean KLD | Same top-1 |
| --- | --- | --- | --- | --- | --- |
| `Q8_0` (reference) | 12,669,627,712 | 546.47 | — | — | — |
| **PXQ4** | 6,981,972,000 | 600.20 | 0.101 | **1.337** | 57.4% |
| Google's QAT `q4_0` | 6,975,879,296 | 782.82 | 0.367 | 3.033 | 40.2% |

*(wikitext-2, `-c 2048`, 10 chunks, one P100.)* The whole tier set, on the same text wrapped in
the model's own turn structure (4 chunks, base PPL 533.99), against that same reference:

| File | Bytes | PPL | Mean KLD | Same top-1 |
| --- | --- | --- | --- | --- |
| **PXQ4** | 6,981,972,000 | 557.20 | **1.137** | 58.1% |
| PXQ3-balanced | 5,920,289,024 | 1,452.35 | 2.362 | 41.9% |
| Google's QAT `q4_0` | 6,975,879,296 | 869.30 | 2.809 | 41.2% |
| PXQ2-attn4 | 4,858,605,760 | 444,224.68 | 9.865 | 1.2% |

**Ship PXQ4 for this model.** PXQ3-balanced is the aggressive tier and its two metrics disagree —
better KL divergence than the QAT file at 15% fewer bytes, far worse perplexity. **PXQ2-attn4 does
not work on Gemma 4 at all**: agreeing with the reference on the next token 1.2% of the time is a
destroyed model, not a degraded one, and the profile's protection of attention cannot rescue it
because this architecture keeps half its bytes in 15,360-wide FFN projections on all 48 layers even
at two bits. It is measured here so that nobody tries it and reports a broken engine.

**At matched bytes the PXQ4 file is 2.3x closer to the full-precision reference than Google's own
4-bit file of this model** — and it still answers: booted from the PXQ4 file, it retrieves a fact
from the first line of a 3,121- and an 8,267-token document and states it in the answer. Three
things belong next to that claim. Google's QAT file is a *retrained* model rather than a
quantization of these weights, so it is a comparison of two shipping files and not a clean codec
A/B. The absolute KL divergences are large because of the regime described above and rank the
files against each other rather than standing alone. And PXQ3-balanced is the tier where the two
metrics disagree — better KL divergence than the QAT file at 15% fewer bytes, considerably worse
perplexity — so it is offered as the aggressive tier and nothing more.

The composition is what the quantizer chose with no flags: 240 tensors at PXQ4 (attention Q and
output, and all three FFN projections), 88 at `Q8_0` (attention K and V — 48 and 40, because the
eight full-attention layers share K as V and carry no V tensor at all), `token_embd` at `Q6_K`,
and 338 F32. **No tensor fell back for geometry**: every 2-D tensor in this architecture clears the
slab codec's 64-row, 32-wide requirement. The file was quantized from `Q8_0` rather than from
bf16, which is a second lossy pass and therefore strictly below what a single pass from the
original weights would give.

Scope: the `gemma4` / `gemma4_mtp` graph-build paths only. Every other architecture is unaffected;
this is a targeted refusal, not an allowlist.

Until 2026-09-10 the guard was a denylist **by architecture**: it refused every Gemma-4 file,
including the dense ones, which had therefore never been tested here. The condition is now the
shape that actually fails (`n_expert > 0`), so the dense models are refused no longer.

## `v2026.09.02`: delta-net state aliasing can produce garbage output (fixed in the next build)

**Symptom:** a request occasionally comes back as `!!!!` or another unsampleable, non-finite
completion, at a rate of roughly 1 in 36 requests on a plain single-copy-slot server, or on
almost every request if pipeline parallelism (a second scheduler copy slot) is enabled on a
multi-GPU layer split.

- **Root cause (corrected 2026-09-03):** the delta-net fusion cluster, not the state carry. On a
  match of its 23-node window the fusion writes the convolution tail through the concat node's
  data pointer and derives the delta-net state destination from that same pointer, instead of
  using the tensors the graph allocated. That is valid on exactly one graph shape — the in-place
  single-sequence concat into the recurrent cache row it was written against. On any other layout
  the same nodes still match, and whether the write is safe depends on whether the allocator gave
  it an exact alias or a shifted overlap of what it reads. A sibling fusion on the output gate
  skipped a write on an unverified sole-consumer assumption, with the same consequence. An earlier
  revision of this entry blamed the `np==1` state carry and the allocator; both were wrong, which
  is why the "fix" in that revision did not remove the fault — the fusion wrote past it.
- **Status:** fixed in the next engine build (see `RELEASE-NOTES-2026-09-07.md`) by guarding both
  fusions (`PXA_FUSE_DELTANET_WAR`, default on) so each fires only on the shape it is provably
  correct for, and by giving four unguarded node-adjacency fusions the same write-after-read and
  same-device checks. `v2026.09.02` has this defect live. The only workaround on that tag is
  `PXA_FUSE_DELTANET=0`, which costs the fusion's decode win but removes the fault; disabling
  pipeline parallelism lowers the odds without removing them.

## `v2026.09.02`: an ADD+RMS-norm fusion can make greedy output nondeterministic (fixed in the next build)

**Symptom:** at `temperature=0`, the same prompt against the same model file returns a different
completion on different runs. Seen on mixed-format MoE files (a model whose expert tensors are
not all the same quant type); not tied to one card architecture.

- **Root cause:** fusing an ADD into the RMS-norm that follows it removes the launch boundary
  that used to act as a barrier between the two kernels. When the norm's output overlaps either
  ADD input at a nonzero offset, the allocator can place the norm's output on the ADD inputs'
  freed, neighbor-merged memory, so the norm reads a location the ADD kernel is still writing.
- **Status:** fixed in the next engine build (see `RELEASE-NOTES-2026-09-07.md`) by declining the
  fusion whenever the norm output overlaps an ADD input at a nonzero offset; exact,
  index-for-index aliasing still fuses, so the fix has no measured cost. `v2026.09.02` has this
  defect live on affected files.

## 1080 Ti (11 GB): a from-scratch PXQ2 re-export may not fit where the published tier does

The published Fusion2-35B PXQ2 tier fits the documented bench protocol on an 11 GB 1080 Ti
(`-b 2048 -ub 768 -ngl 99`) with roughly 139 MB of VRAM to spare. A PXQ2 file you re-export
yourself from the same source weights can land noticeably larger on disk and resident, and a
file about 60 MB larger than the published tier no longer fits that same protocol on this card:
it fails to allocate its first request's compute buffers. If you are on an 11 GB card, prefer the
published PXQ2 tier over a fresh re-export, or drop `-ub` (or the context) until it fits.

## Multi-slot Flash-Next: the fast single-sequence prefill lever turns itself off above one slot

`PXA_DN_SHARE_INPUTS` builds the delta-net input views once per graph instead of once per layer.
It changes no arithmetic, only the allocation layout, and it is worth **+30.5% prefill at 20,801
tokens** on a 2× V100 layer split. It is **automatic**: on when the context is created with
`n_seq_max == 1`, off above that.

The reason it is not unconditional: the merged layout makes a residual defect reachable at
`-np 2` that is not reachable at `-np 1`. Measured, 12 slot-pinned greedy runs of a 20,801-token
needle prompt per arm — `-np 2` with every lever off: 12/12 identical. `-np 2` with
`PXA_DN_SHARE_INPUTS=1`: 9 identical, 2 corrupt, 1 a third signature. `-np 1` with the same lever:
12/12 identical. So the defect is only *exposed* by the shared views, not caused by them; the
layout is doing what a different memory layout would do anyway.

- **Consequence:** a server started with `-np 2 --kv-unified` is correct but gives up that prefill
  lever. A single-slot server keeps it. There is no correct way to force it on above one sequence
  today; `PXA_DN_SHARE_INPUTS=1` will do it and is not safe there.
- **Status:** open. The next step is to find what the merged layout makes reachable, not to
  change the lever.

## `PXA_SCHED_ASYNC_INPUTS` has one remaining site where a submitted graph outlives its call

The stream-ordered split-input staging ring is **default off** (`PXA_SCHED_ASYNC_INPUTS=0`). Two
call sites that let a submitted graph outlive its call were found and fixed (the scheduler slot
event must stay per split; `llama_decode` must be drained before a real-path reserve or a
reused-graph input rewrite). At least one more remains: a 12-run greedy gate at `-np 1` came back
11 clean and 1 corrupt with the lever on, and 12/12 clean with it off.

- **Consequence:** none at the default. Do not set `PXA_SCHED_ASYNC_INPUTS=1` on anything you
  care about. On its own the lever is worth +2.0% prefill at `-ub 2048` and nothing at `-ub 512`.
- **Status:** open.

## Two server slots at different KV offsets are not required to produce the same bytes

This is a property of a shared attention-KV ring, not a defect, and it is worth stating because it
looks exactly like one.

With `--kv-unified`, slot 0 holding a prompt at KV offset 0 and slot 1 holding the **same** prompt
at offset 3,266 return different — individually stable — completions to a byte-identical greedy
request. Erase slot 0 so slot 1 sits at offset 0 and the two become **byte-identical, including
every log-probability** (0.0 delta). Measured on the divergent pair: the first 12 tokens are
identical (mean |Δlogprob| 0.026, max 0.084) and the split happens at token 12 on a top-2 gap of
**0.005 nats**. Both arms recall a planted needle. The same structure appears with
flash-attention off, so it is `n_kv`-dependent reduction tiling in whichever attention path is
running. Mainline llama.cpp has the same property.

**How to compare slots fairly:** compare each slot against a reference **at the same KV
placement** — erase the other slots first (`--slot-save-path` plus
`/slots/<id>?action=erase`) — or compare with a log-probability tolerance rather than a hash.
Raw sha comparison across slots at different offsets is not a valid test and will report failures
that are not failures. A `-np 1` control run four times over gives 4/4 identical.

## `PXA_CUDA_GRAPH_DECODE=1` aborts on the second request into a warm slot

**Default is off**, and it should stay off. With CUDA-graph decode enabled, the second request
routed into an already-warm slot dies with an illegal memory access. It reproduces with none of
the 2026-09-03 build's new bits set, so this is the fork's pre-existing decode-graph path, which
an older architecture gate used to keep out of reach on these cards.

It is also slower where it does run: 37.2 / 36.6 / 36.0 / 37.9 t/s with graphs against 38.4 / 38.1
without, on 2× V100, with the counters confirming the path was live (87 captures, 4,569 replays).

- **Consequence:** none at the default. There is no reason to turn it on.
- **Status:** open, and low priority — the lever loses even when it works.

## `llama-gguf` and `llama-gguf-hash` fail to link on a full-target build

Pre-existing, unrelated to the engine, and it bites anyone who runs `cmake --build build` without
naming targets. Inside a build container started **without** `--runtime=nvidia`, CMake cannot
resolve `CUDA::cuda_driver`, and the link fails — for `llama-gguf` and `llama-gguf-hash` first,
because they come early in the target order, but for every CUDA-linking target including
`llama-server` if you get that far.

- **Workaround:** start the build container with `--runtime=nvidia` (or `--gpus all`), exactly as
  `BUILD-FROM-SOURCE.md` shows. That fixes every target. Alternatively, name the targets you
  actually need: `cmake --build build --target llama-server llama-cli llama-bench llama-quantize`.
- **Status:** open; the CMake dependency should not require a runtime GPU to resolve.

## Multi-GPU prefill: use `-b 8192`, because the default `-b 2048` costs about 10%

Not a defect, but the single most common way to leave multi-GPU prefill speed on the floor, so it
is written down here rather than only in a recipe.

`-b` is the prefill **chunk** size (`-ub` is the micro-batch inside a chunk). The engine
synchronizes on every chunk boundary, so a 20,801-token prompt at the default `-b 2048` pays 14 of
those; at `-b 8192` it pays 3. On a 2x V100 layer split at `-ub 512` that is 1,097/1,092 t/s
against 1,207/1,204 t/s, and summed device utilisation goes from 1.24-1.33 to 1.41-1.42. `-b 20480`
is another 0.3%, i.e. nothing.

- **Recommendation:** `-b 8192` on any multi-GPU layer split. On the 2x V100 pair the full
  shipping configuration is `-b 8192 -ub 2048`, and since 2026-09-03 the server fills exactly
  that on a 2x V100 pair when you do not pass `-b`/`-ub` yourself (2x P100 gets `-b 8192 -ub 256`,
  a single 1080 Ti `-b 2048 -ub 768`); see `docs/COOKBOOK.md`.
- **Cost:** compute buffer. A larger chunk needs a larger compute buffer, which is why the engine
  default stays at 2048 — an 11 GB card running a model that only just fits cannot afford it.
- **Not a fairness trick:** mainline llama.cpp was measured at the same chunk sizes on the same
  cards and does not gain from them (936/1,167 t/s at `-b 8192` and 942/1,142 at `-b 20480`,
  against 1,165 at its own default). The setting helps this engine's scheduler specifically.
- **Status:** documented, not defaulted.

## Pipeline parallelism is off for `qwen4exp` at large context, on purpose

- **Symptom (before the fix):** a 4x P100 Flash-Next seat at 150k context failed to create its
  context on the first boot, and the fallback failed too.
- **Cause:** the retry after a failed pipeline-parallel reserve created a second scheduler while
  the first still held its partial compute buffers, so there was no memory left for the fallback
  either. Fixed: the scheduler is freed before the retry.
- **Why the default changed anyway:** at 150k context the second copy's compute buffers are about
  2 GiB on one card and do not fit, and the seat's prefill did not benefit from pipelining in the
  4-card measurements. So `qwen4exp` no longer defaults to pipeline parallelism. `qwen35` keeps
  the default, where it is worth about +12% at 20k tokens on the V100 pair.
- **Override:** `PXA_PIPELINE_PP=1` forces it on for any architecture, if you have the VRAM and
  want to measure it yourself.
- **Status:** fixed and defaulted deliberately; not a limitation you need to work around.

## `PXA_DSA_ATTN=1`: a node the CUDA backend declines falls to the CPU backend

The sparse-attention path is requested per node, by attaching an index list to the flash-attention
node's `src[5]`. The CUDA backend now **declines** such a node when it cannot honour the request,
instead of running a dense kernel on it — a dense kernel handed a too-narrow index list does not
compute the dense answer, it computes a different one (measured 73.2% off in RMS).

Declining is the correct behaviour, but it has a consequence worth stating: the scheduler then
places that node on the CPU backend, which answers the *dense* question. So with the lever armed,
a node CUDA declines is silently answered by a different computation than the one that was asked
for — just slowly rather than wrongly.

Nothing reaches this today: the graph builder's own gate mirrors the backend's predicate, so no
shipping model attaches an index list the backend would decline. That is two places agreeing, not
a guarantee, and the lever ships **off**.

- **Status:** open, and bounded by the lever being off. The fix is for the request to be refused at
  graph-build time rather than resolved by a fallback at schedule time.

## `PXA_KV_INDEX_CHECK` is a debug instrument, and it is not bit-neutral

`PXA_KV_SEQ_RM_INDEX` itself is: with the index on and off, the same prompts produce byte-identical
answers, checked over three aged boots with the prompt cache driven to its ceiling.
`PXA_KV_INDEX_CHECK=1` — which rebuilds the index from scratch on every indexed removal and compares
— is a different matter: two of the twelve gate prompts (`p128`, `p1536`) moved against the control
with it on, while the on-vs-off comparison without it was exact.

That is the shape of an observer changing what it observes, not of the index being wrong, and it is
why the check is an instrument rather than a gate. Use it to find drift; do not use it in a run
whose output you intend to compare against anything.

- **Status:** by design, documented rather than fixed. `PXA_KV_SEQ_RM_INDEX` ships off; the check
  ships off with it.

## Four-card Flash-Next seat: identical within a boot, not always across boots

At one request at a time on the four-card hybrid-MoE seat, greedy output is **deterministic within
a boot** — the same prompt gives the same bytes every time, for as long as the server stays up.
Across boots it is not: five of the twelve fixed gate prompts, all of them 2,048 tokens or longer,
produce a different completion after a restart. Shorter prompts are stable across boots too.

That pattern — long prompts only, stable within a process, moving between processes — points at
something fixed at load or first use rather than at a race in the decode loop, but the cause is
still being narrowed and I would rather say that than guess in public.

- **Status:** open, under investigation. It does not affect a running server's reproducibility,
  which is what a client actually sees; it affects comparing a number taken today against one
  taken after a restart. Take both sides of any comparison within one boot.

## Fixed in this release — long prompts could answer differently, or not at all, on either card family

Long prompts on the GP100-class cards could answer differently, or not at all, depending on what
else the server was holding. The kernel responsible is the tile flash-attention kernel, which on
those cards handles BATCHED attention -- prompt processing, and any speculative verify batch
wider than eight columns. It kept its running softmax state in half precision, and it walks the keys in 64-cell tiles whose membership is
decided by the slot a request lands in rather than by the prompt itself — so the rounding of a
twenty-thousand-term accumulation moved when the placement moved.

I want to be blunt about how bad it was, because I only found out by measuring it: on a
20,859-token prompt, from one placement the first token was a newline at 96.5% and the model
answered; from another, four hundred cells further along, the end-of-text token won at 59.6% and it
returned nothing — and against a double-precision reference the half-precision arithmetic was 26%
off at *both* placements, so the good-looking one was not right either, it was lucky.

I checked the fix rather than assuming it, on both kinds of model, and the two answers are
different enough to be worth your time.

On a **dense** model — a 27B at PXQ4 on two GP100-class cards, 20,801-token prompt — the fixed
kernel is placement-invariant: start the request's KV band at cell 0 or at cell 101 and the first
token's distribution agrees to about one part in a thousand (top-1 0.0432 against 0.0431). The old
half-precision kernel was *also* placement-stable on that model, and still wrong: it put the top
token at 0.057 where fp32 says 0.043, and ranked a different token second. That is the 26% seen
live rather than against a reference — stable and wrong is the worst combination, because nothing
about it looks broken.

On a **mixture-of-experts** model the fix removes the failure but not the variation. On the
four-card seat both placements now return the same correct 32-token answer to the 20,859-token
needle prompt, where the old kernel returned nothing at the misaligned one — but their first-token
distributions still differ (0.70 against 0.93 for that token). The fixed kernel leaves rounding-order
residue at the 1e-7 level, and greedy top-10 expert routing plus recurrent layers amplify it: a
near-tie in the routing can land on the other side. That is inherent to greedy expert selection
under any arithmetic that is not bit-exact, not a defect left over. So on a MoE seat, if you need
reproducible output, erase both slots before a request or drive it with one slot (`-np 1`); on a
dense model you no longer need to.

The fix is not free: on the four-card seat that 20,859-token prompt takes 54.7 s to prefill instead
of 49.4 s, about 10% slower. I would make that trade again — the alternative is a fast answer that
is 26% off and occasionally empty.

The running state is fp32 now. Two consequences you should expect. Answers to long prompts on these
cards change against the previous release, on purpose. This is a GP100-class change only: the
V100-class cards run a different batched-attention kernel that this release does not touch, so
their output and their speed are unchanged for reasons that have nothing to do with this fix.
On the older cards it is not free — this is a correctness change, and I would
rather you get different bytes than confidently wrong ones. And the empty-answer class goes away
with it. `PXA_FA_TILE_F32ACC=0` restores the old arithmetic if you need to compare.

## Speculative decoding is lossless in its acceptance rule, not in its arithmetic

The acceptance rule only ever keeps a drafted token when the target model would have produced it,
so speculation cannot put a token in the output that the model did not choose. That is not the same
as producing the same bytes: with MTP speculation at the default depth, greedy output differed from
unspeculated greedy output on 5 of 6 fixed prompts on the 27B model. The verify batch's reduction
shape is not the single-row decode's, so a near-tie can land on the other side of the tie — and
wherever the draft sequence happened to be identical, the text was identical too, which is the
signature of arithmetic rather than of a rule being broken.

- **Status:** by design, and stated rather than hidden. If you need greedy text that is
  bit-reproducible against an unspeculated run, turn speculation off with
  `--spec-type none` (or leave `PXA_AUTO_SPEC=0` set, which stops the engine arming it for you).

## The vLLM sidecar's log-probabilities endpoint returns 500

Asking the sidecar for token log-probabilities fails with a server error. Generation itself is
unaffected — this is the logprobs response path only — and the llama-engine server's equivalent
endpoint works. It is being diagnosed; I would rather ship the line than have you find it.

- **Status:** open. If you need log-probabilities today, take them from the llama-engine server.

## Speculation changes the bytes, and the cascade changes more of them

Already stated above for MTP: speculation is lossless in its acceptance rule and not in its
arithmetic. The n-gram cascade is the same effect with a wider reach — at temperature 0, only 2 of
12 fixed prompts came back identical to the same server with no drafter at all. Every one of the
twelve is a valid greedy continuation; they are not the *same* valid continuation.

- **Status:** by design, and the reason is the verify batch's reduction shape rather than the
  drafter proposing anything the model would not have chosen. If you need greedy text that is
  bit-reproducible against an undrafted run, use `--spec-type none`.

## Fixed 2026-09-13 — the MTP head re-planned its graph twice per accepted token

**How it showed.** Not as an error. A boot looked healthy and decode looked ordinary; the only
visible trace was a stream of reserve lines in the log and, at longer draft lengths, an
out-of-memory a dozen speculative cycles into a run that had started fine. What was happening is
that the MTP head's two passes built **different graph shapes** — the N-row update pass asks for
fewer output rows than it has, so it carries an `out_ids` row-slice; the 1-row draft pass asks for
one row out of one, so the slice was skipped — and one context cannot hold two node counts under a
single allocator plan. Every alternation re-planned: a full backend drain plus a compute-buffer
reallocation, **twice per accepted token**. Over one bench that is 229 re-reserves at width 5.

Counted over a whole boot including prefill, reserve calls / coverage failures / allocator re-plans
were **242 / 174 / 176**; they are now **7 / 5 / 11**. The fix (`PXA_MTP_STABLE_OUT_IDS`, default on)
builds the slice unconditionally — at one row out of one the slice is the identity, so no value
changes — and a thrash guard (`PXA_RESERVE_GUARD`, cap `PXA_RESERVE_MAX`, default 24) latches the
mechanism off per context if it ever does start churning again.

- **Status:** fixed, default on, and the output is unchanged: greedy probe hash, draft counters and
  the full 512-token control generation are identical to the pre-fix arm. `PXA_MTP_STABLE_OUT_IDS=0`
  restores the old shape if you need to reproduce the churn.

## The main context still has two graph shapes, and the guard is what makes it cheap

The same cause survives in the main model, in a smaller form. Its last layer carries the same
`out_ids` row-slice (`GET_ROWS attn_out-63` and `GET_ROWS sainp_get_rows-63`), and the mirror image
of the MTP case applies: the reserve is widened to 2048 and asks for 2047 outputs, so it **builds**
the slice; a real decode ubatch is a verify batch in which every row is an output, so it **skips**
it. Two shapes, and the coverage test fails on the widths a verify batch actually takes (2, 4, 5, 44,
55 have all been observed).

The guard blacklists each shape after one failed try, so the cost is bounded at roughly **5 coverage
failures and 11 allocator re-plans per boot** — per boot, not per cycle. The same unconditional-slice
treatment that fixed the MTP head would work here and the identity argument is unchanged, but it
would add two `get_rows` to **every** main-model decode to save 11 re-plans per boot, and it would
need its own bitwise and speed proof. That trade was declined deliberately.

- **Status:** known, not fixed, bounded. `PXA_RESERVE_DEBUG=1` prints the two nodes if you want to
  see it for yourself; revisit only if a profile says those re-plans matter.

## Per-step recurrent checkpoints scale with the draft length, and on a 16 GB card that is the ceiling

A recurrent (DeltaNet/GDN) target checkpoints its state every speculation window. In the exact
`per-step` mode the buffers hold one conv+SSM snapshot **per drafted step**, so they scale linearly
with how many tokens the drafter proposes. Measured on a 2× V100 16 GB pair with the 27B PXQ4 file:

| capacity (`max_tokens`) | card 0 | card 1 |
|---|---|---|
| 5 (a 4-token draft) | 377.58 MiB | 226.55 MiB |
| 33 | 2996.02 MiB | 1797.61 MiB |
| 65 | 5988.52 MiB | 3593.11 MiB |

At 65 the first card's budget is weights 7587.75 + KV 1245.52 + compute 432.93 + checkpoint 5988.52 =
**15254 of 16384 MiB** — and the allocation *succeeds*, which is the trap. It leaves about 0.7 GB, the
CUDA pool grows during decode, and the first sizeable pool allocation then dies with an out-of-memory
about a dozen speculative cycles later, a long way from the line that caused it. Because it is a
**startup** allocation, no scheduler-side escape hatch avoids it.

`PXA_CKPT_BUDGET` (default on) now decides the capacity by affordability rather than by fit: after
these buffers are claimed, does every device still hold the compute buffers it has already reserved
plus `PXA_CKPT_BUDGET_MARGIN_MB` (default 256)? Every drafter stage is clamped to the answer, and the
answer is printed at **every** boot, clamp or no clamp:

```
common_speculative_init: recurrent checkpoint budget: the chain drafts up to 4 tokens (capacity 5), this context can checkpoint 5
common_speculative_init: recurrent checkpoint budget: the chain asks to draft 64 tokens, this context can checkpoint 33 - every stage is clamped to n_max=32
```

- **Status:** the crash is fixed; the **ceiling is not, and it is physics rather than a defect**. A
  16 GB card cannot checkpoint an arbitrarily long draft exactly, so on this pair a 64/48 n-gram stage
  resolves to a capacity in the fifties at `-sm layer` and around twenty under `-sm attn`, which leaves
  less headroom. **Read the budget line before comparing two arms**: two runs that differ only in the
  capacity they were granted have been compared as if they were like for like, and the clamp is not
  otherwise visible anywhere.

## `--recurrent-ckpt-mode gpu-fallback` changes the greedy output and halves prose throughput

The obvious way to buy a longer draft is to stop preallocating: `gpu-fallback` keeps small (~75 MiB
per card) shadow buffers whatever the draft length is, and it does avoid the out-of-memory above. It
is also a measured loss on both counts that matter. On the default cascade at `-sm layer`, in a valid
window (bracket drift 0.00%) with identical draft counters in both arms:

| mode | control | prose | greedy probe hash |
|---|---|---|---|
| `per-step` (the default that resolves) | 83.00 | **82.13** | `a5467dd43144` |
| `gpu-fallback` | 82.19 | **41.60** | `18c44e18e36f` |

Control is untouched and **prose halves**, which is the signature: prose rejects drafts far more often
than the repetitive class does, so it pays the restore cost far more often. And the hash moves — the
restored state is not the state that was saved, so this is a fidelity difference and not only a speed
one.

- **Status:** open, and **not recommended**. A rollback mode that alters the output is a defect, not a
  lever; the shadow buffers are most likely a narrower copy of the recurrent state. Use
  `PXA_CKPT_BUDGET` (on by default) to get a long draft with exact per-step checkpoints instead. The
  mode is kept because it is the only way to run a draft length this pair cannot checkpoint at all.

## Reproducing an upstream tensor-split number: it is sensitive to what else the host is doing

A note for anyone repeating the cross-engine comparisons in the release notes, because it cost time
here. Upstream `llama.cpp`'s two-card tensor split exchanges partials through pinned host memory with
an **in-kernel spin on a host-memory arrival token** — a design that is right for a pair with no peer
path, and one that holds a CPU busy while it waits. Its throughput therefore moves with host load in a
way this engine's peer-direct route does not: across three windows in one morning, with this engine's
own bracket arm holding steady, the same upstream cascade arm read **141.11**, **134.26** and
**114.47** tok/s on the control class — the 114 arm being the one that ran with two CUDA compiles on
the host.

- **Status:** not a defect in either engine; a measurement hazard. Quote an upstream tensor-split
  number only from a window whose bracket arms agree, run the box otherwise idle, and say which window
  the number came from. The same caution applies in reverse to any arm of this engine that is compared
  against one.
