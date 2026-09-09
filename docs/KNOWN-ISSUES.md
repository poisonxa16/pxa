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

## Gemma-4 (128-expert Gemma MoE) is unsupported — fails clean at load

The PXA runtime is tuned for Qwen-family MoE. The **128-expert Gemma-4 MoE** path (`GEMMA4`
with per-layer embeddings, and the `GEMMA4_MTP` assistant) is **not supported**: on the affected
expert-config it corrupted the heap (`double free or corruption` on batches >= 1024 tokens) and
fell back to ~8 t/s scalar decode.

Rather than risk heap corruption, the runtime now **fails clean at model-load / context-creation**
with a clear error instead of attempting the build:

```
PXA runtime does not support arch 'gemma4' (128-expert Gemma-4 MoE); use stock llama.cpp. See docs/KNOWN-ISSUES.md.
```

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

- **Scope:** the `GEMMA4` / `GEMMA4_MTP` graph-build paths only. Every other architecture the fork
  supports (llama, qwen3moe, glm4_moe, openai_moe, deepseek2, minimax_m2, ...) is unaffected — this
  is a targeted denylist, not a Qwen-only allowlist.
- **Workaround:** run Gemma-4 on **stock llama.cpp**. The PXA runtime does not attempt Gemma-4
  optimization; the guard exists only to fail safely rather than corrupt memory.

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
