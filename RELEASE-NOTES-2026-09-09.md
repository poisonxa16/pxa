<p align="center"><img src="docs/assets/pxa-network-banner.png" alt="PXA Network" width="760"></p>

# PXA v2026.09.09-rc3 — release candidate

Release candidate. Supersedes `v2026.09.07-rc1`. There was a `v2026.09.09-rc2` in between —
built, gated on both seat models, and staged as a tarball and a container image — but I never
published it, so this release folds rc2's work into rc1's baseline and everything below is new
to anyone who has only seen the public repository. No compiled source in this note is
hypothetical: every fix cites the test that fails without it, and every lever below states
plainly whether it has a number behind it yet.

---

## Read this first

**Three correctness fixes land in this release, each with a regression test.** Two of them —
the multi-token-prediction head reading back a reused buffer, and a hybrid model's attention
window going empty after a forced re-process — were silent: no crash, no error, just a wrong
answer or a `NaN`, and only under a specific concurrency shape. The third — the indexed KV
removal path — was caught before it shipped and has stayed off by default until now. Read
[Three correctness fixes](#three-correctness-fixes-each-with-a-test-that-fails-without-it)
before assuming an older build's behaviour still holds.

**Multi-token-prediction speculation changes shape again in this release**, and the honest
state of its defaults is that the fix above reopened a measurement I had already closed once.
See [MTP: defaults and the two-slot shape](#mtp-defaults-and-the-two-slot-shape).

**A large batch of new performance levers land in this release, every one of them default
OFF.** None has a shipping-default number behind it yet; each is listed with what it does and
what would have to be true for it to ship on. See
[New levers, available and unmeasured](#new-levers-available-and-unmeasured).

**PXQ2 and PXQ3 now convert and serve on the vLLM sidecar**, not just on this engine. See
[PXQ2 / PXQ3 on the vLLM sidecar](#pxq2--pxq3-on-the-vllm-sidecar).

---

## Three correctness fixes, each with a test that fails without it

**The MTP head's feature row was being read back after the allocator had reused it.**
The multi-token-prediction head tags its FFN output `result_norm`, and that node is what the
decode picks by name as the embeddings source for any MTP op — so it is the row that gets read
back and handed to the next draft step as its conditioning hidden. It was never marked a graph
output, which leaves the graph allocator free to hand that buffer to a later node the moment the
head's own output projection has consumed it. The logits are computed *before* the reuse, which
is why the token sampled off that row was always right (0.954 accepted at depth 1) while any
chain built on the row itself collapsed (0.045 mean top-1 probability at depth 2) — an asymmetry
that read as a graph or row-addressing problem for two separate measurement passes before this
one. `tests/test-mtp-head-output.cpp` builds the head's tail at its real shape, allocates it
through the graph allocator exactly as a decode graph is allocated, and asserts both halves of
the signature: flagged, the row matches to `0.0`; unflagged, the row is off by `39.46` while the
logits still match to `0.0`. Guard: `PXA_MTP_HEAD_OUTPUT`, default **on**; `=0` reproduces the
defect for anyone who needs to confirm this is what an older build was doing.

**A hybrid slot re-entering its own cached prompt could take an attention window with no keys in
it.** With a unified KV cache and two slots in flight, one sequence re-processing a prompt from
position 0 while the other decodes lands its cells in a band that need not start at cell 0. The
attention window was sized from a *sampled* scan of the cell array — every 256th cell, with
flash attention on — which is sound only while occupancy is a contiguous prefix of the cache.
When the live band sat entirely between two sampled probe indices, every probe missed, the scan
answered zero, the window was clamped to 256, and all 133 mask rows of that batch were `-inf`:
the first flash-attention layer produced 272,384 `NaN`s out of 272,384, which then poisoned the
recurrent state carry and looked, for a while, like a recurrent-state bug rather than a masking
one. The fix is the exact scan this engine already applies to its sliding-window cache — the
sampled overload is deleted outright so it cannot come back. Measured before the fix: 80 sampler
soft-failures, 5 breaker trips, 5 truncated answers, on a 60-request drive designed to force the
re-process branch. After: seven independent runs, `0 / 0 / 0`. `tests/test-kv-cell-max.cpp`
keeps a verbatim copy of the retired sampled scan and fails if it is ever tempted back — one case
reproduces the exact measured cell layout, a second sweeps 3,000 random band layouts. Two
belt-and-braces guards also went in at the same time, in the CPU flash-attention and softmax
kernels: a fully-masked row now returns a finite zero instead of computing `0 * inf`. They are
inert on a correct graph and exist only so this specific failure shape can never again surface as
a silent `NaN` from a different call site. Cost: the exact scan is slower than the one it
replaced — about 422 microseconds in its worst case, once per micro-batch, roughly 1% of a
4-way-concurrent decode step and 0.4% of a 512-token prefill step — and it ships that way rather
than caching the result, because correctness came first and the caching form is still just an
idea on the ledger, not code.

**The KV removal index was answering from an incomplete whitelist.** An index that mirrors
cache edits from a hand-maintained list of call sites fails silently when the list is short: it
keeps answering, from contents that no longer describe the cache. Two real defects fell out of
walking every unified-cache cell mutation against that list: one bulk-edit call site invalidated
nothing at all, and a removal call with its start and end swapped could walk off the end of the
index's internal map instead of matching nothing, the way an equivalent full scan does. The list
is no longer load-bearing: every change to a cell's sequence set and every whole-cell copy now
bumps a tag epoch in the one place all such edits already pass through, and the index answers a
query only while its own epoch agrees with the cache's. A verification mode
(`PXA_KV_INDEX_CHECK=1`) runs the removal both the indexed way and the scanned way from the same
starting state and prints exactly where they'd disagree, rather than requiring a bisection.
`tests/test-kv-seq-rm-index.cpp` runs four arms of 60,000 steps each and passes; the two
defect-reproduction arms fail on the pre-fix code and pass on this one. The lever
(`PXA_KV_SEQ_RM_INDEX`) stays **off by default** in this release: the fix is proven on CPU and by
a live-server harness that drives a prompt cache to its eviction ceiling, but it has not yet
cleared a GPU determinism window on the model class that first found the earlier version of this
bug, and that window is what would flip the default, not this note.

---

## MTP: defaults and the two-slot shape

The head-output fix above changes what the MTP head actually hands the next draft step, which
means every acceptance-rate table measured before it — including the depth default this engine
shipped with internally — was measured against a drafter that was quietly worse than the one in
this binary. That table is being re-taken rather than assumed, on both server topologies a seat
actually runs (one request at a time, and two concurrent), because the two have disagreed before:
an earlier internal measurement found the previous shipped depth 31% slower than not
speculating at all once two requests were running concurrently, even though the same depth won
cleanly at one request at a time. The number that decides a default in this release is the one
that holds on **both** columns, not just the faster-to-measure one.

- **Draft depth default: 3**, for the dense `qwen35` family. Decode 256, median of 3, on two
  V100s serving a 27B PXQ4 model — one request at a time, then two concurrent clients:

  | arm | one client | two clients (aggregate) |
  |---|---|---|
  > **What these were measured on.** The benchmark prompt behind this table is 54 words drawn
  > from an 18-word vocabulary, generated with the end-of-text token ignored. That is an
  > accidentally *repetitive* workload — on an n-gram drafter it accepts essentially every
  > proposal — so treat these as speculation's best case, not its average. A drafter's whole
  > job is guessing what comes next, and text that repeats itself is the easiest text there
  > is. Numbers on prose, code and genuinely repetitive workloads are being measured
  > separately. **Plain, unspeculated decode is unaffected by this caveat**: it does not
  > depend on what the prompt looks like, so every non-speculation figure here stands as
  > written.

  | no speculation | 37.31 t/s | — |
  | **depth 3** | **54.42 t/s** | **71.89 t/s** |
  | depth 4 | 54.11 t/s | 35.54 t/s |

  Depth 4 buys nothing at one client and loses half the throughput at two. The default is the
  depth that holds on both columns, which is the whole point of measuring both. Two caveats I
  would want if I were reading this: on the arms whose overlap fell below 0.9 the two-client
  figure is partly serialised throughput rather than true concurrency — the ranking stands, and
  the arm I rejected had the *better* overlap (0.93) — and at two slots the output varies with
  batch shape and cold start, which is the documented near-tie class rather than a defect.
- **`PXA_MTP_ZERO_OUTPUT_COMMIT`** (the commit decode asks for no sampled output, trading one
  extra single-row draft decode for a full-graph commit pass collapsed to a KV-only one): resolved
  per architecture family from the same window as the depth default — and it is **off**. At depth
  3 it measures 52.52 t/s at one client and 63.63 aggregate at two, against 54.42 and 71.89
  without it. It led the earlier matrix for a reason that no longer exists: it was working around
  the stale read-back row that the head-output fix above closes. With the cause gone, what is left
  is its own cost — an extra single-row draft decode, about 2.9 ms a cycle. A workaround that
  outlives its bug is just overhead, and this one is now off by default.
- **`PXA_MTP_PMIN_TOPK`** (an MTP confidence floor now compares a top-10-renormalised
  probability instead of a full-vocabulary softmax, matching the scale a floor is meaningful at):
  default **10** at the shipped config level, **0** at the bit-exact reference level. Inert on
  the shipping path today, because the family default for the floor itself is `p_min = 0` — no
  probability is computed at all until an operator or a future default arms one.
- **`PXA_MTP_BATCH_SLOTS`** (one draft decode per step shared across every concurrent slot,
  instead of one independent draft chain per slot): default **off**, landed without a speed
  window yet. On today's per-slot chains, two concurrent slots pay roughly twice the draft-decode
  count of one; this is the lever that removes that multiplier when it ships.

**The two-slot (`np>1`) shape gets its own scrutiny in this release, not a rider on the `np1`
number.** Two things changed specifically for it:

- **Hidden-row addressing now refuses instead of guessing.** The target model's hidden states
  are stored densely, one row per raw batch position, only while every row of a batch actually
  asks for one. Under the batch-size-dependent warm-up path, a batch can instead store only the
  rows that asked for output — packed at the *end* of the buffer, not at the position their token
  occupied. A caller asking "give me the hidden for batch row 0" against that second buffer shape
  used to get back row 0 of the wrong tensor: a real hidden state, just for a different token,
  returned as if it were correct. The addressing rule is now a single, tested function with an
  explicit contract: a request against a batch-dense buffer resolves to that row; the same
  request against the output-indexed shape is refused rather than answered, and only the
  documented negative "last row produced" sentinel is honoured on both shapes. This closes a
  silent-corruption class that only two concurrent slots — one of them mid-prefill — could reach.
  The trade is that the refused case now costs a free draft token this release did not have a
  free way to reproduce correctly; an error is logged and speculation continues from a
  freshly-decoded hidden instead.
- **The determinism gate for every default in this release runs at `np1` *and* `np2`, on the
  same KV placement**, not `np1` alone. This is deliberate given the finding above: a shape that
  looks settled at one concurrent request is not the shape the seats actually run.

---

## The launcher no longer pre-empts the engine's own MTP default

`--spec mtp` with no explicit depth used to be rewritten by the launcher to a fixed depth before
the engine ever saw it — first to a depth that measured as a loss on both architectures, later to
a flat depth of one once deeper chains were found to lose for a different, since-fixed reason.
Both were the launcher guessing at a decision the engine now makes for itself: a bare `--spec
mtp` passes straight through, and the engine's own boot-time line names the depth, confidence
floor and commit shape it resolved for that file's architecture and why. The launcher keeps one
refusal — a depth of two or more on a sparse mixture-of-experts file, which has not been
re-measured since the dense-family fixes above landed — and that refusal now has the usual escape
hatch (`--accept-unmeasured`) instead of being a hard stop.

---

## New levers, available and unmeasured

Every lever below landed this release without a shipping-default speed number attached, so every
one of them ships **off**. That is not a hedge: it is the actual state of the evidence, and each
row says what already backs it (a CPU proof, a bit-exactness argument) and what does not (a speed
window).

| lever | default | what it does | evidence so far |
|---|---|---|---|
| `PXA_FUSE_SIBLINGS` (+ `PXA_SIB_MAX`) | off | Merges a run of independent, same-op, same-shape graph nodes (copy and row-normalisation nodes today) into one launch. Nothing is eliminated, only the launch count drops; a runtime check proves no node in a merged run can see another's write. | Bit-exact by construction; CPU parity proof in place. No GPU speed number yet — whether a given decode graph even contains a mergeable run is itself unmeasured. |
| `PXA_CPY_ROWS` | off | Resolves a copy kernel's source/destination indices once per row instead of once per element. Only engages on narrow, numerous rows (gated `ne0 <= 256 && rows >= 256`); a microbenchmark found the coarser mapping is a *loss* everywhere outside that shape, so the gate exists to keep it off where it would hurt. | Bit-exact by construction. Microbenchmarked on one card in isolation; not yet measured inside a real decode graph. |
| `PXA_FA_F16_KV_CHUNK` (+ `PXA_FA_KV_CHUNK_GRAN`) | off | On a card with no fp16 matrix-multiply hardware serving a **quantized** KV cache, converts that cache to fp16 in bounded chunks instead of one whole-tensor conversion, capping the scratch buffer at a fixed size regardless of context depth. This is a VRAM-headroom lever, not a speed lever, and prefill is expected to give back a small amount at depth in exchange. | CPU-proven merge arithmetic. **Has no effect at all on a server running an uncompressed (f16) KV cache**, which is how the shipped serving recipes run today — it only matters once a deployment adds a quantized KV cache at long context. |
| query-time sparse attention (`PXA_QSA` family) | off, and gated further by `PXA_QSA_MIN_FILL` (default 65,536 cells) | Selects a subset of cached keys instead of attending densely once the cache is deep enough that selection is cheaper than the attention it replaces. This release's work went into the overhead of the selection machinery itself (an append-only candidate pool instead of a full rebuild per token, one fused reduction instead of several, a single-launch top-k), not into the underlying idea, which shipped off in the previous release too. | Identity-proven: the selected path reproduces the dense reference's output exactly on the fixtures tested. The fill floor exists because, below it, dense attention is simply cheaper — the lever is not meant to engage there and does not. |
| `PXA_RS_RING` | off | Expresses a rejected draft token's recurrent-state rollback as an index move into a small ring of retained states, instead of a device-side copy of the whole state. | Landed without its own A/B window in this release. |
| `PXA_FUSE_DELTANET` (existing lever, mask unchanged) | on, mask **53** | One bit of this mask's fusion set — folding a recurrent-state gather into the multiply that immediately follows it — has been part of the shipped mask since early in this project but never actually ran: a device-identity check used to read the first bytes of a host-resident input as if they were a device id, which reliably declined the fusion. That guard is fixed generically (every fusion sharing the same check benefits), so this fusion now executes on real traffic for the first time. Nothing about the mask value changed; what changed is that part of it now does something. | Bit-exact by construction, and CPU-proven on both destinations the fused kernel writes. GPU speed delta on the seat this fires on is unmeasured — the expectation is small and positive, since the starting point is a kernel launch that was previously pure overhead. |
| `PXA_KV_SEQ_RM_INDEX` | off | See [Three correctness fixes](#three-correctness-fixes-each-with-a-test-that-fails-without-it) above — this is the fixed version of the lever, still off by default pending a GPU determinism window. | CPU-proven; live-server harness clean; GPU window pending. |

---

## PXQ2 / PXQ3 on the vLLM sidecar

The vLLM sidecar (`tools/vllm-pxq4/`) served only PXQ4 for tensor-parallel linear modules until
this release, even though the kernel libraries had carried PXQ2 and PXQ3 operators for longer
than that. A checkpoint declaring either of those tiers on a linear module raised a hard "tiered
LINEAR modules are not implemented" error at construction. That gap is closed: the sidecar
resolves a per-module tier the same way it already did for its mixture-of-experts modules,
allocates that tier's narrower slab stride, and dispatches to that tier's own kernels — with no
new fallback path and no new device-side scratch-arena management, because those kernels were
already written to be self-sufficient.

Both tiers now **convert** (the offline GGUF-to-vLLM converter decodes PXQ2/PXQ3 tensors instead
of refusing them at the header) and **serve** (a converted checkpoint loads and generates). Status
against this release's own gate, which is the same same-top-token-agreement, greedy-identity and
speed protocol every tier here is held to: **PXQ2 agrees with this engine's own server on 97.5%
of first tokens and 92.5% of exact 32-token continuations**, over 40 greedy prompts on the same
file; **PXQ3 agrees on 92.5% and 87.5%**. The non-default options are markedly worse — 82.5% /
45.0% and 70.0% / 37.5% — which is why they are options and not defaults. Read the bar honestly:
this is agreement between two of my own implementations of the same codec on the same weights, so
it measures whether the vLLM decoders reproduce the engine's, and it does not measure either
against the original model.

The full tier-support truth table — what converts, what serves, and which claim is actually
gated in this release — is [`docs/VLLM.md`](VLLM.md); that page is the single source of truth for
this and the prose above will not repeat numbers that live there.

---

## PXQ2 and PXQ3 decode faster on the vLLM side

The 2- and 3-bit tiers had no split-decode kernel, so their decode ran a path sized for wider
weights. They have one now, on by default (`PXQ23_MMV_SPLIT` / `PXQ23_MMV_MT`; `=0` restores the
old path). Same image, same file, the kernel the only variable — and the generated text is
byte-identical, so this is speed and nothing else:

| file | decode, one request | at two concurrent (per stream) |
|---|---|---|
| `PXQ3` uniform | 43.60 t/s, was 31.49 — **+38.4%** | 36.70, was 23.86 — **+53.8%** |
| `PXQ2` uniform | 51.49 t/s, was 37.07 — **+38.9%** | 40.52, was 28.81 — **+40.6%** |
| `PXQ2` attn4 | 49.75 t/s, was 41.34 — **+20.3%** | 35.20, was 27.88 — **+26.3%** |
| `PXQ3` balanced | 43.93 t/s, was 36.22 — **+21.3%** | 32.98, was 24.53 — **+34.4%** |

Prefill is unchanged. The two promoted profiles gain about half as much as the uniform files, and
that is exactly what you would expect: a promoted file is only part 2- or 3-bit, so only part of it
was ever on the slow path.

One honest note on the `attn4` row. An early boot of the split arm produced a different 256-token
continuation, which I could not reproduce: the re-run gives 49.93 and 35.34 with byte-identical
text, and the kernel is bit-exact against the old path at every shape I tested, with a wide margin
on the first token. This model is dense, so there is no expert routing to blame it on. I am
recording it rather than explaining it away — one unexplained boot is a thing to say out loud, not
a thing to leave out of a table because it did not happen again.

## Quantizer: a policy layer for where the extra bits go

Quantizing at a tier below PXQ4 (PXQ2, PXQ3) has always meant choosing which tensors get the
lower tier and which stay higher. That choice was previously baked into one fixed table per model
class. A `--pxq-policy` flag (`PXA_PXQ_POLICY` as an environment override) now selects between
named policies — **`uniform`** (the previous fixed behaviour, and the default), **`balanced`**,
and **`attn4`** — that promote the attention-block tensors (query/key/value projection, the
grouped-attention gate, and the recurrent output projection on hybrid models) by one tier or to a
flat PXQ4 floor, on top of the existing per-architecture backbone table. `--pxq-uniform` remains
as an explicit alias for the default. The policies are **available and unmeasured**: the default
is unchanged from the previous fixed behaviour precisely because nothing yet says a different one
is better, and a quality comparison across policies and model families is the next thing to
measure rather than something this release is claiming.

### The DeltaNet output projection is protected at `q8_0`

Measuring the policy layer turned up something the policy layer did not cause. On Qwen3.8-27B —
a hybrid model, 48 DeltaNet layers out of 65 — a file built today with no flags measured **mean
KLD 0.433** against its own `Q8_0` source, while the 23 August build of the same recipe measured
**0.060**. A tensor-by-tensor census of the two files put the entire gap in one class:
`blk.N.ssm_out.weight`, the DeltaNet output projection, 48 tensors and about 5% of the bytes.
That class had been moved onto the 4-bit backbone on a byte-parity argument with no fidelity
measurement behind it; the August file predated the move and still had it on `mxfp4`.

Measured properly, one class changed and nothing else (wikitext-2, `-c 2048`, 10 chunks, same
base logits): `pxq4` **0.433**, `mxfp4` 0.060, `pxq4hq` 0.049, `pxq6` 0.044, `q8_0` **0.042**
(same-top-p 93.05% → 93.26%). Decode does not pay for the difference — on two P100s at `-c 8192`,
np1 medians of seven, the four files land within 2.7% of each other with no separation.

**So `ssm_out` is now pinned to `q8_0` by default on every PXQ level and under every policy**, at
a cost of +5.1% file size. `pxq6` is a hair cheaper (+1.2%) for the same fidelity and was not
chosen: `pxq6` and `pxq4hq` have no vLLM decoder in this release, a file is routed to an engine
by its *level* rather than by its contents, and a `PXQ4` file carrying 48 `pxq6` tensors would be
handed to an engine that cannot read it. `q8_0` loads on both and is already in every PXQ file we
ship. It is a floor rather than a pin: `--custom-q 'ssm_out\.weight=<type>'` and
`PXA_PXQ_SSM_OUT=<type>` (or `=off`) still decide, and a deliberately flat-MXFP4 backbone
(`PXA_PXQ_BACKBONE=legacy`, `=lite`) is left alone. The resolved `pxa.pxq.backbone_map` KV in
every file says which one it got.

The same class was on the 4-bit tier at PXQ2 and PXQ3 as well — the profiles buy it up to the
PXQ4 cap, which reads like protection and is the same allocation — so the fix lands on those
levels too.

---

## Everything else since rc1 (the unpublished rc2 work, folded in)

`v2026.09.07-rc1` is the last tag anyone outside this project has seen. Everything in this
section shipped, was gated, and was staged as `rc2` — then held back rather than published, so
it is new to a reader of the public repository even though none of it is new work as of today.

- **`POST /slots/:id?action=erase` works without `--slot-save-path`.** Erasing a slot drops its
  cached prompt and KV in memory; it has nothing to do with slot *files*. The whole `/slots/:id`
  route was nevertheless registered only when `--slot-save-path` was set, so a caller could not
  erase a slot without pretending to want slot files, and got a bare 404 instead of an explanation.
  The route is now always registered, and only `save` and `restore` refuse without the path — with
  a message that says which flag is missing and that erase does not need it.

  Worth saying plainly, because it changes how some of this project's own numbers should be read:
  the caller this hit hardest was my determinism harness, whose stated premise is that every
  request lands at the same KV offset as its single-stream reference — which it arranges by erasing
  the other slots first. Those erase calls had been 404ing on every seat boot, and the harness
  swallowed the error. So an unknown share of the two-slot divergences recorded during this
  release's development was not a defect at all, but the KV-placement variation that the same
  harness documents as expected. The harness now treats a failed erase as fatal rather than as
  nothing, which is the only way a precondition is worth stating.

- **A 2-5% prefill regression on the V100 pair, closed.** A restructuring of the gated-recurrent
  kernel to support a per-key-channel forget gate (needed for a different model family) cost every
  gated-delta-net model extra registers and extra inner-loop instructions it did not need, because
  nothing told the compiler the per-channel path was dead for this family. Moving that choice from
  a runtime branch to a template parameter restores the pre-regression register count and
  instruction count exactly where the per-channel path is unused, by construction rather than by
  measurement — confirmed unchanged by the determinism gate, byte-identical reference hashes
  across the release binary, the regressed head, and the fix, on all twelve gate prompts.
- **A divergent-barrier bug in the vendored Volta flash-attention kernel, closed.** The kernel
  family backing flash attention on Volta cards was carried in from upstream at a commit that
  predated upstream's own fix for a barrier-convergence bug in the exact code shape this engine's
  compiled kernel configurations all use — every shape this engine instantiates takes the affected
  branch, not a rare corner of it. A compute-sanitizer synchronization check found roughly three
  thousand divergent-barrier reports on the pre-fix binary and zero after, with prefill and decode
  speed unchanged within run-to-run noise.
- **A double-normalization discrepancy between the CPU and GPU paths of the gated-recurrent
  attention norm, closed.** The query and key vectors of this engine's gated-delta-net attention
  were normalized once in the graph builder and then normalized again inside the recurrent op
  itself, on the CPU backend only — using two different formulas and two different numerical
  floors. The GPU backend only ever applied the first normalization. For an ordinary, well-scaled
  activation this is invisible: renormalizing an already-unit vector moves it by an amount many
  orders of magnitude below floating-point precision. It stops being invisible for a vector whose
  norm sits near either formula's protection floor, which is a real, if uncommon, state for a
  freshly-reset or heavily-masked position to be in. Fixed to one reference formula, applied once.
- **The indexed KV-removal path was found unsafe and shipped off**, before the proper fix above
  existed. This is the same lever documented in [Three correctness
  fixes](#three-correctness-fixes-each-with-a-test-that-fails-without-it) — it is listed here too
  because the sequence matters: it was live by default for part of this development cycle, was
  caught by exactly the kind of gate this project runs before every tag, and was shipped off while
  the real fix was built. No release with this lever on by default ever reached the public.
- **The determinism-gate harness itself had a silent-pass bug, found and closed.** A comparison
  script that feeds two engine builds identical prompts and diffs their output was calling this
  build's command-line tool with a flag from an older interface; the tool rejected the flag and
  exited immediately on both sides, and the script's own extractor treated two empty outputs as
  "identical" and reported a pass. Every identity-gate pass claimed before this fix should be
  treated as unproven, not as false — the re-run under the corrected harness (which now hard-fails
  on a nonzero exit code, an empty output, or a length mismatch beyond what the echoed prompt
  explains) reproduced every one of those passes genuinely.
- **Everything above merged into one engine head and re-gated as a unit**, on both seat models,
  before this release candidate's own work began on top of it.

---

## Performance: the one-card floor

Measured on the merged head this release builds on, one Tesla P100 16 GB, the standard published
protocol (temperature 0, unique prompt per repeat, median of 7 with one warm-up discarded):
**73.49 tokens/second decode**, **1,452.0 tokens/second prefill** at a roughly 5,800-token cold
prompt. Both are level with the `rc1` release binary measured back to back on the same box.
Published floor: **70 t/s decode, 1,300 t/s prefill or better** — see `START-HERE.md` and
[`docs/COOKBOOK.md`](COOKBOOK.md) for the full protocol and how to reproduce it on your own card.

---

**Two requests that arrive together are a separate question from placement.** Single-request
greedy output is reproducible across processes — three boots, twelve prompts, byte-identical — and
the placement-controlled two-slot check passes (13 of 13). But two requests that arrive at the same
moment can be processed in the same batch, and a near-tie token can then come out differently from
what that same prompt produces alone. During this release's gate that happened **once in five
trials, on one prompt out of twelve**, and never on the other eleven; four later trials on two
binaries, including one with the new accumulation turned off, did not reproduce it. It is
timing-dependent batch composition, not placement and not leftover state — and it is the same
effect that makes speculative decoding not byte-reproducible against an unspeculated run. If you
need reproducible bytes, send one request at a time.

## What "deterministic" means on the four-card seat

On the Flash-Next 4xP100 seat I can now say exactly what "deterministic" means here, because I
measured it instead of assuming it. Run the same sequence of requests against a fresh server and
you get the same bytes back: three independent boots, twelve prompts from 62 to 20,859 tokens,
greedy, byte-identical on all twelve, down to which slot the server picked and which prompts
stopped early. What is *not* stable is the same prompt asked at a different point in a server's
life. The attention KV is one shared ring and a request is placed wherever there is room, and it
turns out that where a request's band *starts* changes the answer -- not by a rounding error, by a
lot. With the ring empty, the 20,859-token needle prompt answers correctly and confidently: the
first token of the real answer sits at 0.965 against 0.033 for end-of-text. Put a 100-token
request on the other slot first, so the needle's band starts 101 cells in instead of on a tile
boundary, and the same prompt in the same process gives 0.633 against 0.348 -- and in the wild,
with a 20k prompt already resident, it goes to 0.396 against 0.596, which means the server answers
that question with nothing at all. I chased that to the end: it is not the other request's
content (two different 256-token fillers give bit-identical results), it is not state left behind
by a previous occupant (a filler on the same slot changes nothing at all), and it is not a
per-process lottery (every one of these numbers reproduces to nine decimal places in a fresh
process). It presented as the alignment of the band's starting cell, and the cause underneath that is
the one described above: the tile flash-attention kernel -- which on the GP100-class
cards handles batched attention -- accumulated its running softmax state in half precision, so a long prompt's arithmetic depended on which ring
cells its keys landed in. That is a bug, not a property, and it is fixed in this release. On a dense model the fixed kernel is
placement-invariant to about one part in a thousand; on a mixture-of-experts model two
placements can still answer differently at a near-tie, because greedy top-10 expert routing
amplifies rounding-order residue the fixed kernel leaves at the 1e-7 level. That last part is
inherent to greedy expert selection under non-bit-exact arithmetic, which is why the recipe
below still applies to MoE seats and no longer applies to dense ones. Until you are on that build: drive the seat with one slot (`-np 1`) or
erase the slots between requests, and do not compare runs taken at different points in a server's
history. One more thing worth saying, because it affected the numbers this project has been
quoting: the determinism gate's "erase the other slots first" step was calling an endpoint the
server only registered when `--slot-save-path` was passed, so it had been failing silently on
every seat boot. The endpoint no longer needs that flag, the gate now fails loudly instead of
skipping its own precondition, and the numbers here were taken after both.

## The release gate

Run on the exact binary in this release, on two V100s, with the strict setting that turns an
operator-chosen skip into a failure:

| arm | result |
|---|---|
| release gate, Volta family (`GATE_STRICT=1`) | **PASS 13, FAIL 0, SKIP 0** |
| seat gate, greedy determinism, 12 fixed prompts | **12/12** at one request and at two |
| coherence smoke | **12/12** |
| needle recall | **4/4** on both prompt lengths |
| non-finite logits / breaker trips / sampler soft-failures | **0 / 0 / 0** |

Speed against the previous build, measured back to back on the same two cards minutes apart —
not across sessions, which is the only way these numbers mean anything:

| | this release | previous | |
|---|---|---|---|
| prefill @512 | 684.94 t/s | 762.59 | see note |
| prefill @2,048 | 970.56 t/s | 971.05 | level |
| prefill @8,192 | 993.37 t/s | 999.77 | level |
| decode, one request | 34.52 t/s | 34.33 | level |
| decode, two requests | 50.31 t/s | 49.63 | level |

Everything is within about 1% except the 512-token prefill, and I am not claiming that one as a
regression: at n=3 the two arms' min-max ranges overlap, which is what warm-up noise looks like at
the shortest prompt. It is the row I would re-measure first if you care about short prefill.

On the four-card seat the same release gate is **PASS 13, FAIL 0, SKIP 0**, and its speed is level
with the previous build (prefill 350/514/459 t/s at 512/2,048/8,192 against 351/513/476; decode
23.98 and 23.77 against 23.85 and 23.37). The 8,192 row is 3.5% down and that is the fp32
accumulation being paid for: it costs nothing at 2,048, about 3% at 8,192 and around 10% at 20,859,
which is the shape you would expect from an accumulation whose error grows with the number of terms.

The seat's greedy determinism arm on that model reads **12/12 at one request and 4/12 at two**, and
I am publishing that number rather than the one that flatters. Most of those eight are the
concurrency class described above — two requests batched together, a near-tie landing differently —
and three of the twelve prompts return an empty answer on both builds, which is a pre-existing
behaviour of this model at those lengths and not something this release introduced. The single-request
column is the one a client actually experiences, and it is clean.

**Verified on the shipping artifact.** The 20,859-token needle prompt is now answered at every KV
placement tested — starts at cell 0, 101, 257, 4,097 and around 20,800 — with identical 32-token
text, where the old kernel returned nothing at the misaligned ones. The first token's probability
still ranges 0.68 to 0.93 across those placements, which is the mixture-of-experts amplification
described above and not a return of the defect. The twelve-prompt fixed sequence is byte-identical
across three fresh boots, none unstable.

The captured gate output is in [`bench/gate/LAST-RUN.md`](../bench/gate/LAST-RUN.md).

## What else merged, and where each lever stands

| lever | state | why |
|---|---|---|
| `PXQ23_MMV_SPLIT` / `PXQ23_MMV_MT` | **ON** | a measured win and bit-exact: the split-decode table above |
| `PXA_PXQ23_MMVQ` | **OFF**, both tiers | fast, but it fails this project's fidelity rule — see below |
| `PXA_FA_TILE_V2` | OFF | a new tile kernel, landed without a window of its own |
| `PXA_PXQ3_PAIRLUT_DQ` | OFF | bit-identity proven; the cost measurement has not decided it |
| n-gram draft policy | off by default | the alias and the auto-policy ship; arming it is the operator's call |
| `PXA_QSA_GRID_VERIFY` | unchanged | a correctness fix with its own test, not a new default |
| `PXA_SPEC_ADAPTIVE_LOAD` | ~~**ON**~~ → **OFF** (reverted 2026-09-11) | draft depth follows how many slots are decoding; the +37% that made it a default did not reproduce — see below |
| n-gram → MTP cascade | **ON** for MTP files | the table drafts first, the trained head answers what it cannot — see below |

**`PXA_PXQ23_MMVQ` is fast and it still ships off, and it is worth saying why.** It is a real
speed-up — roughly +36% decode on a uniform file at one request, more at two. But this project's
rule for turning a kernel on by default is that it must not cost more fidelity than the equivalent
4-bit path already does, and measured against that comparator it does: PXQ2's mean KL divergence is
about five times the reference and PXQ3's about twice, where the rule allows neither. So it ships
as something you can choose rather than something chosen for you. If you serve PXQ3 on a V100 and
+36% decode is worth +0.35% perplexity to you, `PXA_PXQ23_MMVQ=2` buys exactly that trade. I would
not make the same recommendation for PXQ2: its uniform operating point is not one this engine
sends you to anyway, which is what the quantizer's own policy default is for.

**PXQ4HQ on the sidecar** merged with its kernels and its tests. The tier cap it ships with is
deliberately left where the branch set it, pending the measurement that decides it — this section
gets its number when that window has run, and until then the cap is not a claim.

**One library, one build script, one tag.** The two kernel branches each shipped their own sidecar
library and build script. The merged tree has a single one — `build_pxq_v18.sh`, producing
`libpxq_<arch>_v18.so` — carrying both mechanisms, with the launcher, the tests and the notes all
naming that one version. Two libraries whose names differ by one digit is exactly how a box ends
up running something other than what its logs claim.

**Speculation can back off when the server is busy — but the default was reverted, and this
paragraph is the record of why.** A fixed draft depth is a bet that there is spare capacity to spend
on guessing, and under real concurrent load there isn't — every slot drafting at full depth competes
for the same batch. The depth scales with how many slots are actually decoding, full at one and zero
at the cutoff. Only the depth moves: acceptance stays exact, so it cannot change a token at any
temperature. `PXA_SPEC_ADAPTIVE_LOAD=1` arms it.

That mechanism was made the default on 2026-09-10 on a measured **78.93 t/s aggregate against 57.62
(+37%)** at four clients, with a **49.94 t/s** single-client control inside the base arm's own
**43.5–53.2** band. **On 2026-09-11 the same binary, arms, cards and arm order did not reproduce it,
and the default was reverted to OFF.** The number above is left in place rather than deleted because
the reason it was ever credible is the durable lesson: the 4-client arm's own control drifts **16.7%**
between runs and the server-TTFT arms **20.3%**, so a +37% reading from arms that noisy is not
resolvable at n=5 — and the single-client control, which read as the load-bearing half of the
argument, wanders as much as the effect it was meant to bracket. A control that noisy cannot license
a default in either direction. OFF is also what every release before 2026-09-10 shipped, so the
revert returns to a measured state instead of picking a new unmeasured one.

**An MTP model now gets both drafters, in series.** A file with a trained multi-token-prediction
head used to use only that head. It now gets an n-gram table in front of it: the table answers the
easy continuations — the model quoting the prompt, tool JSON, a code edit — and the head answers
everything the table cannot. The two are good at different things, and running them in series beats
choosing one, which is why the auto layer no longer chooses.

Two honest qualifications, because this default is workload-shaped:

- **At two busy slots, on the workload measured, the n-gram stage ALONE beat the cascade**: 91.0
  against 81.4 t/s aggregate. If you serve two saturated slots, measure your own traffic before
  assuming the cascade is the right shape — `--spec-type ngram` gives you that arm.
- The gain is largest where output repeats the prompt and smallest on fresh prose, which is what a
  table lookup is by nature. See the caveat above about the benchmark workload.

`--spec-type` and `PXA_AUTO_SPEC=0` still win; `PXA_SPEC_NGRAM_POLICY=0` keeps the MTP head alone.

## Known issues

See [`docs/KNOWN-ISSUES.md`](KNOWN-ISSUES.md) for the current list, including three items new in
this release: a query-time sparse-attention mechanism (selecting once and reusing the selection
for several tokens) that turned out to need a second graph shape this engine cannot cheaply
switch between, and so is deferred rather than shipped even experimentally; a graph-fusion rule
that does not yet fold a bias-add into the gate multiply it precedes once a batch carries more
than one token, which costs a handful of extra kernel launches per mixture-of-experts layer on a
prefill or a speculative-decode verify batch and nothing on ordinary decode; and two defects found
by reading, while chasing the hybrid re-entry `NaN` above, that could not be tied to that failure
and were deliberately left unfixed rather than shipped alongside a fix for something else — one in
the KV-removal scan, one in the server's prompt-cache lookup.

---

## Build

This release is built with **CUDA 12.8**, and building it from source needs **CUDA 12.x**. That is
not a preference: CUDA 13.0 removed offline compilation for the Pascal and Volta architectures
altogether, so a 13.x toolkit cannot produce the `sm_60` / `sm_61` / `sm_70` code this project
exists to run. If your distribution has moved to 13.x, keep a 12.x toolkit alongside it or build
in the release container, which pins 12.8.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build build -j
```
