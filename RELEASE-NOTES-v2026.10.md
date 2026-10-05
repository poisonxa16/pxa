# PXA v2026.10 — release notes (draft)

On top of `v2026.09.20`.

---

## Headline: PXQN, the next-generation quant format

PXA v2026.10 introduces **PXQN**, a rotated, Hessian-rounded evolution of the PXQ codec for
Pascal and Volta: a one-sided block-128 Hadamard rotation plus GPTQ/LDLQ-style Hessian-aware
rounding, built into a second-generation `pxq-quantize`. Old PXQ files keep loading; PXQN is a
new format id alongside them, not a replacement for them.

**Speed next to size and quality — Qwen3.8-27B, plain decode (no drafting), 2026-09-28, final
binary.** Every speed figure below sits next to the file it was measured on: its size and its
KL divergence against the Q8_0 reference, scored on the **assistant tokens** of an in-distribution
chat set (what the model actually writes; user-turn tokens excluded).

| 2x P100, `-sm tensor` | PXA, PXQN4 | the fastest other Pascal build we tested, Q6_K |
|---|---:|---:|
| File size | 15,720,262,976 B (4.25 bpw) | 22,431,001,568 B (6.57 bpw) |
| KLD vs Q8_0, assistant tokens | 0.007930 (same top token 97.06%) | 0.002320 (same top token 98.28%) |
| Decode, tg128, t/s | **37.85** | 32.54 |
| Decode at 16k context, tg256, t/s | **36.72** | 31.53 |
| Prefill, pp512, t/s | **344.95** | 273.67 |
| Prefill, pp4096, t/s | **337.58** | 272.10 |
| Prefill, pp16384, t/s | **316.68** | 267.15 |

*llama-bench, both builds with the same flags (`-ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -b 2048
-ub 512 -t 8 -r 3`), same two cards, one bracket (ours / theirs / ours / theirs) after a 60 s
warm-up, judged on the closing pair, nothing else running on the box (prefill: `-p 512,4096,16384 -n 0`
in the same bracket). PXQN4 is a 30% smaller file than Q6_K, decodes 16% faster and reads a prompt 19
to 26% faster; Q6_K keeps the lower KLD — the trade is size and speed for a measured, small quality cost.*

| 2x V100, `-sm tensor`, PXA | file | KLD vs Q8_0, assistant tokens | decode tg128, t/s |
|---|---:|---:|---:|
| PXQN4 | 15,720,262,976 B (4.25 bpw) | 0.007930 | **56.00** (53.98 at 16k context) |
| PXQN5 | 18,785,381,696 B | 0.002211 (lower than Q6_K) | 48.80 |
| Q6_K | 22,431,001,568 B (6.57 bpw) | 0.002320 | 48.32 |

*PXQN4 row: final binary, quiet box, bracket close. PXQN5 and Q6_K rows and all KLD values: measured the same day on the
release-candidate binary one merge earlier (same decode kernels for these files at width 1).*

**MTP (the model's own draft head), first-pass numbers on the final binary.** On a multi-card
`-sm tensor` run of a file that carries an MTP head, a bare `llama-server` now uses MTP by default:
draft depth 3 with a top-1 confidence floor (0.8 on P100, 0.9 on V100) that stops a draft chain
early when the head is unsure. Opt out with `PXA_SPEC_AUTO_MTP=0` or `--spec-type none`.
Qwen3.8-27B PXQN4, greedy, 256 tokens:

| t/s, first request of each class | prose | code |
|---|---:|---:|
| 2x V100, plain | 56.64 | 56.38 |
| 2x V100, MTP (bare server default) | **77.14** | **107.71** |
| 2x P100, plain | 38.46 | 38.52 |
| 2x P100, MTP (bare server default) | **48.06** | **63.79** |
| 2x P100, the fastest other Pascal build we tested, Q6_K with its own MTP drafting | 45.92 | 64.00 |

*Fresh server per prompt class, prompt cache off, 60 s warm-up on an unrelated prompt, the first
request of each class quoted so nothing is replayed, three requests per cell, nothing else running.
Draft acceptance: 75.4% prose and 95.8% code on the P100 pair, 78.4% and 96.8% on the V100 pair.
On P100 our MTP is ahead of the other build's on prose and level on code (63.79 against 64.00).*

**Greedy output with MTP.** MTP verifies several tokens in one wider step, and at that width the
recurrent (DeltaNet) layers, norms and attention round slightly differently from single-token
decode. Greedy text with MTP on is usually, but not always, byte-identical to MTP off: on this
binary both classes matched on the P100 pair, and code matched on the V100 pair, while prose
drifted there. This behaviour predates this release and is not a quality
loss; if you need byte-reproducible greedy output, leave MTP off.

**PXQN in this release is compiled-only.** The PXQN kernels ship as a separate library,
`lib/libggml-pxqn.so`, loaded next to `libggml.so` (RPATH `$ORIGIN`); the source tree carries only
the open forwarding layer. The release tarball includes it. A build made from the public source tree
runs every older format and PXQ, but it can load PXQN files only with the release build's
`libggml-pxqn.so` next to it.

**vLLM images with PXQN.** The vLLM sidecar images gain PXQN: PXQN4 and PXQN5 on V100
(`pxa-vllm:sm70`), PXQN4 on P100 (`pxa-vllm:sm60`), from pre-converted checkpoints
(Qwen3.8-27B PXQN4 and PXQN5). The images are leak-checked and are published
separately from the engine tarball.

**Run MoE models that don't fit your cards.** The expert cache keeps the hot experts on the GPUs
and serves the cold ones from host memory, planned automatically at load. It switches itself on
only for a MoE file that does not fit and has its `<model>.expert-counts.csv` next to it; every
file that fits runs exactly as before. Flash-Next on two P100s: 10.67 t/s decode and 188.6 t/s
prompt reading (3k tokens), with byte-identical output across runs.


---

## What's new

**Flash-Next ships with PXQN experts.** Flash-Next (the qwen4exp MoE architecture) gets
LDLQ/Hessian-rounded PXQN experts alongside its dense PXQN Qwen3.8-27B sibling.
The shipped file is 98,660,237,024 bytes; its KL divergence against the reference on
assistant tokens is 0.0705 (same top token 91.2%), down from 0.0879 for the previous Flash-Next
file. On four P100s with the opt-in tensor-split recipe
(`PXA_TSPLIT_QWEN4EXP_HC=1`, see `docs/LEVERS.md`) it decodes 36.26 t/s (37.93 on prose) and reads
prompts at 510.3 t/s (3k tokens) and 474.4 t/s (16k tokens) on this release's final binary; the
launcher's default for this file stays the measured four-card layer recipe. Serve it from an
SSD, or set `PXA_PLE_MMAP=0`: memory-mapping its per-layer embedding table from a spinning disk
stalls the first long prompt.

**Tensor split reaches PXQ/PXQN hybrid files.** `-sm tensor` used to refuse every PXQ file with
a DeltaNet `ssm_out` tensor by name at load. A panel-aware row-range slicer
(`PXA_TSPLIT_SSM_OUT_PANEL`, on by default) fixes that for the **dense Qwen3.8-27B PXQ/PXQN
tiers** on a matched two-card pair, and on **four identical P100s**. Bug #206 (wrong tokens on a 4-way split
inside a container with docker's 64 MiB `/dev/shm`) is fixed: a failed NCCL group now falls back
to the peer route, so 4x P100 takes the tensor split by default; `PXA_TSPLIT_ALLOW_4WAY=0`
restores the pair-only rule. This is **not** "every PXQ file, any card count" — see Known Issues
for what is explicitly excluded (Gemma 4 is a separate opt-in lever, not this default;
`qwen35moe`/Ornith fails its own KLD admission gate and stays on `-sm layer`; four V100s and
three-card sets are unmeasured and stay on `-sm layer`). Greedy tokens are not always identical
between `-sm layer` and `-sm tensor` under this slicer — see Known Issues.

**Determinism.** The speculative-decode cascade's n-gram stage now honors its configured depth
instead of building itself with an effective `n_max` of 4 regardless of what was asked for; this ships under the
existing default and changes nothing for a user who passes no flags. The 12/12-byte-identical
greedy gate holds on this release's final binary at `np1` and at `np2` with the second slot idle
(slot-pinned, same KV placement), together with the token-0 logit-spread check (6/6 identical)
and needle recall at about 3k, 11k and 21k tokens, for Qwen3.8-27B PXQN4 on the shipping defaults
of a P100 pair and a V100 pair.
Concurrent-decode nondeterminism (a second slot actively decoding a different prompt at the same
time) is **not** fixed this release — see Known Issues, it is not new to PXQN or to any file.
**Server fixes.** A handful of crash-on-a-bad-request bugs are closed: an unbounded
`n_probs`/`top_logprobs` value used to reserve that many entries per token on the decode thread
and throw `bad_alloc` outside any try block, killing the server (now capped at 1024); `/slots/N` save/restore/erase could hang forever on error or
success (`server-slot-actions-hang-forever`); an uninitialised `stop` field on a slot could throw
out of the decode loop during chat-message parsing (`server-slot-stop-uninitialised-chat-parse`);
and the streaming handler could hang forever when the very first result was already the final
one (`server-stream-hangs-when-first-result-is-final`). **Parallel tool calls**: a first fix for
the `<tool_call>` marker match (exact-string compare) shipped with a regression that dropped
parallel tool calls from streamed content; the correction (`3a41a3bc1d`, bug #222) is merged
into this release's integration branch. **Prompt cache off by default on recurrent/hybrid
models**: CONFIRMED — the RAM prompt cache used to restore a similar-but-not-identical prior
prompt against a hybrid/recurrent model's state well enough to lose native tool-calling (bug
`seat-ram-prompt-cache-degrades-hybrid`); it is now off by default for models with recurrent
state unless `--cache-ram` is given explicitly (commit `5b4e83bbd3`, confirmed an ancestor of
this release's integration commit `dcb2b9dad7` — verified directly via `git merge-base
--is-ancestor`, 2026-09-27).

**The launcher picks its own settings, and so does the engine.** A bare `llama-server -m FILE`
now resolves the split mode, batch sizes and flash-attention setting from the same lever registry
the interactive launcher uses, so a bare invocation, `pxa-launch` and the container's no-argument
`ENTRYPOINT` pick the same settings. `pxa-launch --doctor` (or `docker run <image> doctor`)
prints the cards, driver, PCIe links, peer access, `/dev/shm` size and every default it would pick,
without starting anything. **PXA Control** (`pxa-launch --gui`) is the same launcher as a local web
page: cards, models, launch, chat and a speed history.

**Container note.** Multi-GPU PXA in Docker or an LXC container needs a larger `/dev/shm` than
the runtime default (64 MiB) — pass `--shm-size=1g` (`docker run --shm-size=1g ...`, or the
equivalent `shm_size` field in Docker Compose / LXC config). Single-GPU boots are unaffected.

**Ubuntu 22.04.** Two tarballs: the default (glibc 2.38, Ubuntu 24.04 and newer) and
`...-ubuntu22.04.tar.gz` (built with gcc 11 against glibc 2.35, runs on Ubuntu 22.04 and newer).
Same engine, same commit.

**Upgrading from v2026.09.20:** remove `PXA_TSPLIT_REDUCE` and `PXA_TSPLIT_REDUCE_PREFILL` from your
environment. The fused reduce is now the engine default and the prefill route is chosen
automatically; setting `PXA_TSPLIT_REDUCE=off` forces the older staged route (about 7% slower
decode) and the engine prints a one-time warning when you do.

**Multi-card troubleshooting (P2P).** The tensor split copies between cards directly (CUDA peer
access). Some boards, PCIe switches and riser cables report peer access that does not actually
work; the symptom is a hang, a crash or garbage output right after a multi-card load. Two ways
out: set `PXA_P2P=0` (the engine never enables peer access and stages every card-to-card copy
through host memory; free under `-sm layer`, slower under `-sm tensor`), or run the model on one
card. `pxa-launch --doctor` prints what peer access your cards report before you start anything.

---

## Known issues

- **`np2` slot/concurrency differences (#152/#279).** Two related but distinct open
  determinism gaps at `np>1`: identical greedy requests can return one of two fixed outputs
  depending on which of two co-resident, otherwise-idle slots serves them (#152), and separately,
  two greedy requests that are actually concurrent (overlapping in time, not just co-resident)
  can diverge from each other even though sequential requests on the same slots are
  byte-identical (#279). Neither is specific to PXQN, mix27, or Flash-Next — the previous
  default file shows the same class of failure on the same test. `np1` and `np2`-slot-pinned
  (idle second slot) remain 12/12.
- **Tensor split on more than two cards is measured on four P100s only.** Four identical P100s
  take `-sm tensor` by default (see What's new); four V100s, three cards, or a mixed set resolve
  to `-sm layer`. `PXA_TSPLIT_ALLOW_4WAY=0` puts a 4x P100 box back on the pair-only rule. Pass
  `--shm-size=1g` to any multi-card container (the 64 MiB docker default was bug #206's trigger).
- **Repeated context-checkpoint restores can drift a greedy multi-turn chat by the 4th turn.**
  On the default path (ordinary multi-turn continuation, no special flags), accumulated
  checkpoint-restore rounding in Qwen3.8-27B's DeltaNet recurrent state can flip the argmax by
  turn 4 against a cache-off reference (turns 0–2 match byte-for-byte). Open; no fix yet.
- **Six cosmetic (log/banner-only, no output impact) gaps carried to next release:**
  1. Two log-gated levers (`PXQ_MMV_H2` naming, `PXA_PXQ_MOE_MMV_ID`) can't show
     fired/declined/never-consulted on non-P100 cards under default settings — diagnostic only.
  2. Eleven default-on engine gates (six main ones: `PXA_DN_REDUCE_VIEWFIX`,
     `PXA_DN_SPLIT_INPUT_FIX`, `PXA_F16_GEMV_WIDE`, `PXA_SM_GRAPH_REDUCE_CONSUMER`,
     `PXA_SWA_HINT`, `PXA_VOLTA_F16_GEMM`, plus five more) print no boot-line disclosure;
     boot-line code changes are deferred by design so mid-cut binary changes don't invalidate an
     already-run gate.
  3. The PXQ imatrix-weighting ("KQW") quantize-time banner can never print under today's
     short-circuit check order, so a user quantizing with an imatrix sees only "imatrix
     IGNORED" and never learns KQW itself is default-on. KQW's default-on status is a measured
     win; only the log message is wrong.
  4. `timings.prompt_n` in a `/completion` response reports the full prompt length even when a
     RAM-prompt-cache restore skipped most of the prefill — metrics/observability only.
  5. The speculative-decode boot banner can announce a CASCADE-plus-MTP stage on any
     MTP-capable Qwen3.8-27B checkpoint when only the n-gram stage is actually armed — the
     served behavior (n-gram alone) matches what ships either way; only the banner overclaims.
  6. The auto batch-size picker prints a line only when a card set matches a measured cell; on
     any other card set it silently falls back to stock `-b`/`-ub` with no log line saying so.
- **The full off-by-default lever list** (every lab/experimental `PXA_*` env var that ships
  off, what it does, and why it isn't the default) lives in `docs/LEVERS.md` /
  `docs/lab/LEVERS.md` in the release package — see there rather than duplicating it here.

---

## Community

- Discord: <https://discord.gg/EqazvV9tf>
- Ko-fi: <https://ko-fi.com/shatteredrealms1>

---

## Credits

Thanks to the PXA Network Discord community — credited here by name only, never by code or
repository, as a standing rule:

- **Marcus** — reported that Ornith (`qwen35moe`) refuses `-sm tensor`, which started this
  release's investigation into the tensor split on that architecture (it stays on `-sm layer`: a
  measured KLD failure, see Known Issues).
- **mistrjirka** — publishes community PXQ quants of Ornith, Gemma 4 and Qwen3.8-27B.

---
