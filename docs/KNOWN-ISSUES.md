# Known issues

## New this release — `-sm tensor` is proven at one request at a time only

The tensor split (`-sm tensor`) is new and opt-in in v2026.09.20. It is proven correct — same
output as the layer split — at one request at a time, on a P100 pair and a V100 pair, on the
quantisation tiers and architectures it admits (see `docs/COOKBOOK.md`). What it has **not** been
through yet:

- Two or more concurrent requests. It has never been run there and never soaked.
- On Volta it takes a genuinely different numerical path from the layer split (a different
  reduction order), not the same arithmetic reordered — so its output, while correct, is not
  expected to be byte-identical to the layer split's on that card family, even though both are
  valid.

**Status:** working as documented, not yet default. It refuses by name — a card pair with no peer
access, an architecture it has no hardware evidence for, a quantisation tier it has not been
proven on — rather than running silently and producing a plausible wrong answer. If you want to
prove out an architecture yourself, `PXA_TSPLIT_UNPROVEN_ARCH=1` runs it anyway; the boot prints a
loud warning that nothing here has been measured.

## New this release — the lossless speculative-sampling rule is off because of two real bugs

`PXA_SPEC_SAMPLED=1`, the new acceptance rule that reproduces the model's own distribution at any
temperature (see the release notes), is switched off by default in this binary. Two defects were
found in review, both being fixed:

- A dangling sampler pointer when a server slot is reused for a new conversation.
- On a cascade (an n-gram stage in front of the trained head), a drafted n-gram token can be
  checked against a stale distribution left behind by an earlier draft from the trained head,
  which is not the rule's design and produces an unreliable acceptance rate — the one place this
  showed up in this cut's own measurements is flagged directly in the release notes.

**Status:** open, fix in progress. Do not turn this lever on for anything you rely on yet; the
5%-floor heuristic that ships as the default above temperature 0 is documented in
[`LEVERS.md`](LEVERS.md) and in the release notes, including exactly what it does and does not
guarantee.

## 1080 Ti (sm_61): compile-verified and boot-proven this release

Every new kernel in this cut compiles for sm_61 and its template-limit checks pass. The packaged
`v2026.09.20` binary also boots and answers on a real GTX 1080 Ti: a small PXQ4 model
(Qwen3-0.6B) loaded with all 29/29 layers offloaded, `-fa on`, and answered two chat requests
coherently at roughly 220 t/s decode, with zero error lines in the boot log. That is a small-model
proof, not a measurement on this release's larger PXQ4 files — the card runs a production seat
here, so a bigger-model boot needs its own maintenance window.

## Fixed 2026-09-04 — non-reproducible logits on hybrid (qwen35 / qwen3next) models under `-sm layer`

**Symptom.** Greedy (`temperature 0`) generation on a hybrid DeltaNet model split across two GPUs
occasionally produced a different completion for the same prompt — roughly 1 request in 4 on a
20,801-token prompt and 1 in 12 at `-np 2`, more often when the host was busy. Shorter prompts
looked perfectly stable.

**Cause, briefly.** A graph fusion that folds two attention-output kernels into one could, on a
specific memory layout, write to a location a different block of the same kernel had not yet read
— a race that needed a long prompt, two cards and a busy box to show up at all.

**Fix.** The fusion now declines on that layout; it cost nothing to remove (well within run-to-run
noise on both prefill and decode).

**If you are on an older build.** `PXA_FUSE_DELTANET=53` in the environment is the complete
mitigation and needs no rebuild.

**How to check your own build.** `bench/gate/run-gate.sh` check 3b, or by hand: six identical
`/completion` requests with `n_predict 1`, `n_probs 2`, `temperature 0`, `cache_prompt false`, and
compare every returned probability. They must be identical to the last digit.

## MTP under two concurrent requests: a lazy-warmup step refuses a free draft token rather than guessing

An MTP draft step used to be able to silently read the wrong token's hidden state under a specific
two-slot shape (one slot mid-prefill on a long batch while another verifies). That silent
wrong-answer path is closed: the addressing rule now refuses instead of guessing whenever the
batch shape does not support the request.

What is left, and is not a bug: the refused case still costs something. When it fires, the MTP
draft loses its free carried-over token for that cycle and re-seeds from a fresh decode instead —
correct, but not free. It only fires under that specific concurrent-prefill shape.

- **Status:** correctness fixed; the residual cost is accepted, not chased, in this release.

## PXQ2 / PXQ3 on vLLM: the kernels are intentionally behind PXQ4's own tuning

PXQ2 and PXQ3 convert and serve on the vLLM sidecar (`docs/VLLM.md`) through the same tiered
dispatch PXQ4's own modules use. Their kernels deliberately do **not** carry the extra tuning
PXQ4's own kernels use on the V100 serving path — that tuning exists to fix a warp-starvation
pattern specific to PXQ4's own launch shapes, and no equivalent pattern exists at PXQ2/PXQ3's
shapes. This is a scoping decision, not an oversight: do not expect a PXQ2/PXQ3 linear module to
match a PXQ4 module's tuning depth cycle for cycle.

- **Status:** working as scoped.

## `pxq-quantize --pxq-universal`: harmless CUDA-driver noise + argument order

**Two things trip people up when building a PXQU (`--pxq-universal`) quant. Neither is a real bug.**

**1. Harmless container noise.** Running `pxq-quantize` inside an `nvidia/cuda` container prints,
before any real output:

```
ERROR: driverInitFileInfo 578 result=11
ERROR: init 664 result=11
ERROR: init 250 result=11
```

This is emitted by the **NVIDIA container runtime's own driver probe**, not by `pxq-quantize` —
the string does not exist anywhere in this project's source or binaries. Quantization proceeds and
completes normally. Ignore it. (It also appears on a plain `--help`, and on any other quant type;
it is not specific to `--pxq-universal`.)

**2. `--pxq-universal` is a flag — put it before the positional filenames.** Flag parsing stops at
the first positional argument. If `--pxq-universal <map>.tiers` comes *after* the input/output
paths, it lands in the positional `type` slot and you get an "invalid ftype" error instead of a
quantize run.

Correct order (flag first, then `in out PXQ_UNIVERSAL`):

```bash
./pxq-quantize --pxq-universal my-16gb.tiers \
  model-bf16.gguf model-PXQU16.gguf PXQ_UNIVERSAL
```

The argument is a path to a `.tiers` map; see `docs/PXQU-CONVERT.md` for the format.

## The same three `ERROR: ... result=11` lines before `llama-server` starts

Same cause as above, same verdict: harmless. Inside an `nvidia/cuda` container the runtime's
driver probe prints those three `ERROR:` lines and then `llama-server` prints its own first line,
which in a container is

```
PXA_CONTAINER_AWARE_v1: runtime=container (/.dockerenv present (docker))
```

That second line is not an error either — it is the server saying it detected Docker, so that a
wedge or a fatal GPU fault will make it exit and let the container restart policy bring it back,
instead of sitting there answering health checks with a dead GPU. Bare-metal runs print
`runtime=host` instead. `PXA_IN_CONTAINER=0` or `=1` overrides the detection.

If the server stops after that line, the real reason is in the lines that follow it (model path,
VRAM, `-ngl`, or the `-ot` regex). Please paste from the banner to the exit when reporting.

## `llama-imatrix` crashes on CPU / partial-offload configs (pre-existing; fix pending)

**Symptom:** an imatrix capture run on a configuration that keeps some expert tensors on CPU
(`-ngl < 99`, `--n-cpu-moe N`, or a pure CPU run) hangs or crashes partway through. A run **fully
GPU-resident (`-ngl 99`)** works correctly.

- **Scope:** the activation-collection path in the CPU backend for MoE expert tensors; inherited,
  not introduced by PXQ. Plain generation and perplexity on partial-offload configs are unaffected.
- **Workaround:** run imatrix captures full-GPU-resident, on a smaller-tier quant if the full model
  does not fit, or on a multi-GPU `-sm layer` split — just keep `-ngl 99`.
- **Status:** upstream fix pending.

## PXQ on CPU and under partial offload: not every tier is fast there

PXQ models run on CPU and under partial offload, dense and MoE both — `-ngl 0`, `-ngl < 99`, and
`--n-cpu-moe N` are all fine, including through the CPU backend's fused MoE op. What differs by
tier is speed, not correctness:

- **PXQ4 and PXQ4-HQ** get an AVX2 integer dot product built for the format, on both the plain and
  fused-MoE CPU paths.
- **PXQ2, PXQ3, and PXQ6** fall back to a correct panel dequant: it works, it is not tuned for
  speed.

- **Consequence:** for a CPU-heavy or partial-offload deployment, prefer a PXQ4-family tier if you
  have the choice; for a fully GPU-resident deployment, pick the tier that fits your card entirely
  (see the README tier table). Multi-GPU `-sm layer` splits are fine either way.
- **Also affects imatrix capture:** the CPU/partial-offload imatrix crash above is a separate,
  pre-existing bug, not a PXQ property, and it still applies regardless of quant type.

## PXQ tensors and mainline gguf-py

Not a bug in this fork, but a standing trap: no gguf-py size table (mainline's or this fork's) can
express the PXQ per-row anchor layout, so a gguf-py read-modify-write **silently truncates** PXQ
tensors. Re-run `pxq-quantize` from the bf16/f16 source instead of editing a PXQ file with a
generic GGUF tool.

## deepseek2 / MLA (GLM-4.7-Flash class): `-fa` posture is hardware-aware, not one-size-fits-all

MLA-attention models (gguf arch `deepseek2`) degrade **catastrophically** with context on most
silicon when flash-attention is off. But a P40 (sm_61, GTX 10-series class) is the exception: its
flash-attention kernel is fp16-starved on that hardware and is itself 75–326% *slower* decode
there than running with flash-attention off, and the gap widens with context. So:

- **`-mla 3` always defaults** on a `deepseek2` gguf, on every architecture.
- **`-fa` defaults OFF** when every visible CUDA device is sm_61 (P40, GTX 10-series) — the
  measured-correct posture there.
- **`-fa` defaults ON** on any sm_60 (P100) or sm_70+ (Volta and newer) device present, including a
  mixed fleet with an sm_61 card in it.
- **An explicit `-fa on` / `-no-fa` / `-mla` always wins**, in both directions, on every
  architecture.
- The "long context degrades with `-fa` off" warning is suppressed only on an all-sm_61 fleet,
  where `-fa` off is the recommended, not the dangerous, configuration.

Heed the warning on sm_60/sm_70+ hardware. On an all-sm_61 fleet, `-fa` off is not a warning-worthy
state — it is the default for a reason.

## Gemma 4 — dense and the 128-expert MoE both supported

**Dense Gemma 4** (`gemma4`, `n_expert = 0`) runs on this engine. What has been measured directly
is the 12B; the 31B is the same graph at a different size and is expected to work but was not run;
the smaller per-layer-embedding E2B/E4B shapes load and nothing more than that is claimed.

**The 128-expert sparse MoE** (`gemma-4-26B-A4B`) is **supported by default, no switch, as of this
release.** What it passed before that default shipped, Google's QAT `q4_0` file, two V100s,
`-sm layer`, stock llama.cpp built from source as the comparison engine on the same file and cards:

| Gate | Result |
| --- | --- |
| Retrieval (a fact planted in a long document) | answered at 3k, 8k and 20k prompt tokens, as raw text and through the chat template, thinking off and on |
| Determinism, single slot, greedy | 12/12 byte-identical |
| Perplexity (raw wikitext-2) | not worse than stock on the same file (raw, un-templated prose is far out of distribution for this model on both engines, so this reads as a floor, not an accuracy claim — see the chat-wrapped PXQ table below for a meaningful number) |
| Greedy chat replies vs stock, thinking off both sides, 5 questions | 4/5 byte-identical; the fifth is a confident, stable disagreement, not a near-tie |
| Two slots, two different retrieval prompts sent at once | 8/8 — every answer carried only its own facts |
| 20-minute two-slot soak, mixed request sizes, cancellations | 502 completed, 125 cancelled, server alive, no trouble lines |
| Other card sets | runs on one 16 GB V100 (PXQ3 fits `-c 16384`; Google's `q4_0` file fits `-c 4096` only), on a P100 pair, and on a P100 quad |

The loader accepts either expert tensor layout — Google's single merged tensor, or the separate
gate/up tensors this tree's own converter writes so the fused PXQ expert kernels can use the file.
`-sm layer` is the default and only supported split for this architecture out of the box; `-sm graph`
and `-sm attn` demote to it with a warning. `PXA_TSPLIT_GEMMA4=1` opens `-sm tensor` for this
architecture as an opt-in lever: measured on the 26B-A4B it decodes 8-21% faster than `-sm layer` on
a P100 pair with `PXA_TSPLIT_REDUCE=fused`, and it is slower on a V100 pair, where the head-256
attention kernel serves the `-sm layer` graph only — so it is not offered there yet. A file with
per-layer embeddings is refused. See `docs/LEVERS.md` for the numbers. `PXA_GEMMA4_MOE=0` brings
back the old refusal, as a clean load-time error naming the switch.

**The MTP assistant drafter** (`gemma4_mtp`, the `-it-assistant` files) is **opt-in**:
`PXA_GEMMA4_ASSISTANT=1` builds it, passed as the draft model. Measured on the 12B on one P100 it
is worth roughly +11% to +17% decode where the assistant agrees with the target and costs about as
much where it does not; greedy output with the drafter on is **not** byte-identical to the same
run without it, so it stays off by default. If you need reproducible output, run the target
without a drafter.

**Chat works out of the box.** Gemma 4's own chat template is used automatically — its turn
markers look nothing like Gemma 3's, so the built-in template map would be actively wrong rather
than merely missing (`PXA_AUTO_JINJA`, see `LEVERS.md`). Chat requests no longer default to
thinking: the model's own template default is thinking off, and the engine follows it
(`PXA_AUTO_REASONING`). Pass `--reasoning on`, or `"chat_template_kwargs": {"enable_thinking":
true}` per request, if you want the thought back.

### Concurrent slots are not bit-reproducible on Gemma 4

With two co-resident slots (`-np 2`), the same prompt at `temperature 0` can return a different
completion than the same prompt gets at one slot. This is **a near tie resolving differently, not
a broken answer or content crossing between slots** — a careful check (two different retrieval
prompts sent concurrently) shows every answer carries only its own content, every time. When two
slots prefill at the same moment, the shape of the batch each one lands in can differ slightly
from what it would get alone, and shape affects the order floating-point partials are summed in.
On an ordinary near tie that is invisible; on a genuine coin-flip token (which is what produced the
handful of short/empty replies that motivated this investigation) it can flip the outcome. It
reproduces with every lever off, so it is not one of this engine's optimisations — it is the
ordinary batch-shape sensitivity every engine of this family has, visible only when the model's
own answer was already a toss-up.

**If you need repeatable greedy output**, run at `-np 1` and turn the automatic drafter off
(`--spec-type none`) — a drafter changes the batch shape too, which is the same sensitivity by
another route. Under ordinary load this is rarely visible: the 20-minute two-slot soak above ran
502 requests with one empty reply.

### Size `-c` to what you actually need

This engine allocates the KV cache for **every** layer at the full context, even though Gemma 4
runs a narrow sliding window on most of its layers. Two consequences, both measured on the dense
12B: on a 16 GB card the practical ceiling is around 24k tokens, not the much larger window the
model supports; and prefill cost tracks the **allocated** context, not the prompt actually sent —
the same prompts ran at 122 t/s at `-c 4096`, 68 t/s at `-c 18432`, and 28 t/s at `-c 22528` on one
P100. A per-layer sliding-window cache exists as an opt-in lever (`PXA_GEMMA4_ISWA`, see
`LEVERS.md`) and is not the default this release.

### Our own PXQ quants

**Dense 12B**, converted from Google's bf16 release to `Q8_0` and quantized from there, scored
against that `Q8_0`'s own logits (wikitext-2):

| File | Bytes | PPL | Mean KLD | Same top-1 |
| --- | --- | --- | --- | --- |
| `Q8_0` (reference) | 12,669,627,712 | 546.47 | — | — |
| **PXQ4** | 6,981,972,000 | 600.20 | 1.337 | 57.4% |
| Google's QAT `q4_0` | 6,975,879,296 | 782.82 | 3.033 | 40.2% |

At matched bytes the PXQ4 file is closer to the full-precision reference than Google's own 4-bit
file of this model — and it still answers: it retrieves a fact from the first line of an
8,267-token document. Google's QAT file is a *retrained* model, not a quantization of the same
weights, so this is a comparison of two shipping files, not a clean codec A/B.

**The 128-expert MoE**, PXQ4 and PXQ3, scored against a Q8_0 reference on the checkpoint's own
chat-templated text (raw wikitext is far out of distribution for this model, same caveat as
above — do not score a PXQ file on raw text and conclude anything from it):

| File | Bytes | vs Q8_0 reference |
| --- | ---: | --- |
| **PXQ4** | 13.9 GB | +3.9% perplexity |
| **PXQ3** | 11.0 GB | +7.7% perplexity |

**PXQ3 is the one-card file**: it boots at 16k context on a single 16 GB V100, where Google's own
`q4_0` file of this model only fits 4k context on the same card. Say plainly what Google's file is
and is not: a QAT (quantization-aware trained) build, meaning different weights, not a
quantization of the checkpoint these numbers are scored against — so any row comparing against it
is a comparison of two shipping files, not a quantizer measurement.

## Attention-tile arithmetic on the older cards, and the fp32 fix

On the Pascal-class cards (P100, 1080 Ti), the flash-attention kernel that handles batched
attention (prompt processing, and any speculative verify batch wider than eight columns) used to
keep its running softmax state in half precision, and the resulting rounding depended on where in
the KV cache a request's band happened to start — so the same long prompt could occasionally
answer differently, or not at all, depending on what else the server was holding. On a dense model
the fix (the running state is now fp32) makes the kernel placement-invariant with no output change
otherwise; on a mixture-of-experts model the fix removes the failure but a tiny amount of
rounding-order variation remains, because greedy expert routing amplifies even a very small
arithmetic difference on a near tie — which is inherent to greedy routing under any arithmetic
that is not bit-exact, not a defect left over. If you need reproducible output on a MoE model
served on these cards, erase server slots before a request or run at `-np 1`; a dense model no
longer needs that.

This is a correctness change and it is not free: about 10% slower prefill on a long prompt on the
affected cards. `PXA_FA_TILE_F32ACC=0` restores the old, faster, occasionally-wrong arithmetic if
you need to compare against it. Volta-class cards run a different kernel this change does not
touch, so their output and speed are unaffected.

## Speculative output is not byte-reproducible, on either drafter, and the cascade goes further

The acceptance rule only ever keeps a drafted token when the target model would have produced it,
so speculation cannot put a token in the output the model did not choose — that is what "lossless"
means in this project's docs (see the release notes for the exact, careful version of that claim).
It is **not** the same as byte-identical output. Checking several drafted tokens at once runs the
model at a wider batch than plain decode does, a different batch width selects different kernels,
and where two continuations are a genuine near-tie the wider batch's arithmetic can land on either
side. With MTP speculation at the default depth, greedy output differed from unspeculated greedy
output on 5 of 6 fixed prompts on a 27B model; with the n-gram cascade the reach is wider still —
at temperature 0, the very same request sent to the very same server, unchanged, can come back
with a different valid completion each time, because which n-gram happens to match depends on what
the draft buffer holds when the request arrives.

- **Status:** by design, and stated rather than hidden. If you need output that is
  bit-reproducible — against an unspeculated run, or against itself on a repeated request — turn
  speculation off with `--spec-type none`, or leave `PXA_AUTO_SPEC=0` set so the engine does not
  arm a drafter for you.

## The vLLM sidecar's log-probabilities endpoint returns 500

Asking the sidecar for token log-probabilities fails with a server error. Generation itself is
unaffected — this is the logprobs response path only — and the llama-engine server's equivalent
endpoint works.

- **Status:** open. If you need log-probabilities today, take them from the llama-engine server.

## Per-step recurrent checkpoints scale with the draft length, and on a 16 GB card that is the ceiling

A recurrent (DeltaNet/GDN) target checkpoints its state every speculation window, and in the
default `per-step` mode those buffers scale **linearly** with how many tokens the drafter
proposes. On a 2× V100 16 GB pair with a 27B PXQ4 file, a 64-token draft asks for close to 9.6 GB
of checkpoint memory across the pair — enough that an unguarded boot could succeed at load time and
then die with an out-of-memory a dozen speculative cycles into a run, a long way from the line that
actually caused it.

The engine now works out, at boot, the longest draft it can checkpoint exactly given every card's
already-reserved memory, clamps every drafting stage to that capacity, and **prints the answer at
every boot, clamp or no clamp**:

```
common_speculative_init: recurrent checkpoint budget: the chain drafts up to 4 tokens (capacity 5), this context can checkpoint 5
common_speculative_init: recurrent checkpoint budget: the chain asks to draft 64 tokens, this context can checkpoint 33 - every stage is clamped to n_max=32
```

- **Status:** the crash is fixed; the ceiling itself is not a defect, it is physics — a 16 GB card
  cannot checkpoint an arbitrarily long draft exactly. **Read the budget line before comparing two
  runs**: two arms that differ only in the capacity they were granted are not comparable, and the
  clamp is not visible anywhere else.

## `--recurrent-ckpt-mode gpu-fallback` changes the greedy output and halves prose throughput

The obvious way to buy a longer draft than the budget above allows is to stop preallocating exact
checkpoints: `gpu-fallback` keeps small shadow buffers regardless of draft length. It does avoid
the out-of-memory, and it is a measured loss on both counts that matter — on one measured cascade,
prose throughput roughly halved and the greedy output hash changed, meaning the restored state was
not exactly the state that was saved.

- **Status:** open, and **not recommended**. A rollback mode that alters the output is a defect,
  not a lever. Use the automatic per-step budget above to get a long draft with exact checkpoints
  instead; this mode exists only for a draft length longer than any card in the pair can checkpoint
  at all.

## Reproducing an upstream tensor-split number: it is sensitive to what else the host is doing

A note for anyone repeating the cross-engine comparisons in the release notes. Upstream
`llama.cpp`'s two-card tensor split exchanges partials through pinned host memory with an
in-kernel spin on a host-memory arrival token — right for a pair with no peer path, but it holds a
CPU busy while it waits, so its throughput moves with host load in a way this engine's peer-direct
route does not. The same upstream arm was measured at three different numbers across three windows
in one morning on an otherwise-steady box, purely as a function of what else the host was doing.

- **Status:** not a defect in either engine; a measurement hazard. Quote an upstream tensor-split
  number only from a window whose control arms agree, run the box otherwise idle, and say which
  window the number came from.
