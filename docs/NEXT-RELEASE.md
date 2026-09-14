# What is scoped for the next release

This list is the work this release deliberately did **not** do, written down at the moment the cut was
made rather than reconstructed afterwards. Every item names what is already measured, what is not, and
what would close it — so a reader can tell a decision from an omission, and so the next person to pick
one up starts from the evidence rather than from the idea.

It is a scope list, not a schedule. Nothing here carries a date.

---

## 1. The multi-row verify batch on four Pascal cards

**Status: cause named and measured; no lever tried.**

A speculative verify step decodes M = 1 + draft rows in one forward pass. Matmul kernels are chosen by
batch width, so on `cc 600` this engine's PXQ4 path takes a float-accumulator kernel at `ny ≤ 8` and a
half2 GEMM above it, and the K-split shape and the attention tile change with M as well. The
consequence is a **cost**, not a correctness problem: the speculation multiplier on four P100s is
materially smaller than the one on two V100s, even at high acceptance.

Closing it means making the multi-row path as cheap per row as the one-row path — a kernel question,
on a known boundary, with a measurement that already localises it. Widening the existing window
(`PXA_PXQ4_2D_MAX_NY`) has been tried and is **not** sufficient on its own.

**Also owed on the same card set:** the decode-repetition cell against mainline is marginal (−10.3%
against its current head, −4.1% against its pinned build) and was measured at REPS 3. That class has
since been shown to need **REPS 6** to be stable. Re-measure before arguing about a mechanism.

## 2. GLM-5.3-Flash and Gemma 4 optimisation

**Status: correctness done, performance work not started.**

Both models reached this release by being made *right*, not fast. GLM-5.3-Flash's six-card baseline —
13.82 t/s decode, 50–52 t/s prefill at `-c 24576` — is a starting line, and no lever has been measured
against it. Gemma 4 gained a large default-on win this cut (its wide attention layers moved off the CPU
backend) and nothing beyond it.

Specific items already identified and not attempted:

- **GLM-5.3-Flash at `-c 24576` on six cards has about 28 MiB free on the first device**, so no lever
  fits in that shape. Either a shape with headroom, or a memory lever first.
- **Teach the fit planner to size the compute buffer from `-ub`** instead of a fixed reserve, and then
  measure what the `-ub 64` context ceiling actually costs at a smaller context.
- **Does this graph support more than one sequence at all, and what would it cost?** Today it boots at
  `-np 1` only. This is a source question, not a card question.

## 3. The MTP head for GLM-5.3-Flash is a seven-card shape

**Status: not attempted — capacity.**

The MTP head did not fit alongside the six-card layout, and the seven-card layout is where it has room.
Arming it there has not been tried. It is the obvious next speed lever on that model and it is listed
here rather than implied by its absence.

## 4. The 512/576 attention tile kernel's real-logit deviation

**Status: open, with the instrument built and the wrong diagnosis already retired once.**

`PXA_FA_TILE_512` passes its device test on all 33 cases at 1.2e-07 nmse — after that test's *own*
reference turned out to be the earlier "depth error" (an fp16 accumulator and an all-negative
generator). The pass is necessary and not sufficient: on real 8k logits the same binary sits about ten
times further from an fp32 truth than the path it would replace (symKL 1.122e-01 against 6.172e-03),
and fp32 carriers made it worse rather than better. The lever stays off until that is explained.

## 5. A measured noise floor for the model-fidelity gate

**Status: the cause is identified; the gate's tolerance is the thing that needs replacing.**

GLM-5.3-Flash's remaining logit divergence against a reference implementation is not a second defect.
After the indexer tail fix the layer-by-layer divergence is flat at the diagnostic's print floor, and
what is left is top-k expert routing amplifying sub-1% kernel arithmetic into an occasionally different
expert choice. **A 1e-3 tolerance on a top-1 probability is a bar no 288-expert router can hold across
two different GEMM implementations** — it is an instrument with no stated floor.

The replacement is to *measure* the floor (two runs of the same engine, same file, same prompts, at the
shapes the gate uses) and gate on that, together with top-1 agreement and a spread bound, rather than
on an absolute tolerance nobody derived.

## 6. Audit every synthetic test fixture against a real file

**Status: one instance found the hard way; no audit done.**

A synthetic GGUF fixture wrote a metadata key that real released files do not carry. The identity gate
built on that fixture passed thousands of node comparisons — on a configuration the real file never
takes — while the real file quietly took the other branch and was wrong. The gate was not weak; it was
testing something else.

The work is mechanical and worth doing once, properly: for every synthetic fixture in the tree, list
the keys it writes, diff that list against the keys real files of that architecture actually carry, and
either make the fixture match or make the gate run both branches. **A fixture that is more complete
than reality is a check that silently does not run.**

## 7. Carried-over items that are measured and not shipped

- **The token-batched dense decode fold (`PXA_PXQ_MMV_TOK`)** is measured a win on Pascal and a small
  win on Volta, is bit-exact where it engages, and is **still off by default**. The default flip is a
  decision, not more measurement.
- **The automatic speculation estimator ignores `-ngl` and `-ot`**, so it declines speculation on a
  partially offloaded boot. A fix is written and validated against a real boot's own loader numbers to
  within 1 MiB; it is not in this cut.
- **The sliding-window KV cache for Gemma 4 (`PXA_GEMMA4_ISWA`)** delivers its predicted −74.4% KV
  footprint exactly and is **not deterministic** once a prompt exceeds the window. Its gate is named:
  12/12 at `np 1` **and** `np 2` before the default moves — run the gate before flipping the lever, not
  after.
- **The server's prompt cache saves nothing under that lever and reports success.** One fix, only
  reachable with an off-by-default lever armed.
- **Two concurrent clients** have no published number on any model. The harness defect that produced
  the previous ones is fixed on both sides — each slot owns its drafter, and a slot erase now
  hard-resets that drafter's persistent map — and the re-measurement on distinct prompts per slot has
  not landed.

## 8. The equal-codec gap on four Pascal cards

**Status: measured, clean, and nothing has been tried against it.**

Running this engine on the competitors' own `Q4_K_S` file, mainline is **7.2% faster at 3,121 prompt
tokens and 54.2% faster at 20,801**. The batching confound that used to inflate that reading is gone —
both sides are the bare command line — so the number is now a real, unexplained engine deficit on
somebody else's codec, and it is the last big red cell on `bench/LEADERBOARD.md` with no mechanism
proposed against it.

It matters beyond pride: every headline win on that card set is the codec carrying this deficit, and a
codec advantage is a smaller thing to rest on than an engine advantage.

---

## Flash-Next expert decode on P100: the half2 loop for the MoE drivers

**Measured:** on the four-P100 Flash-Next seat, the routed experts run at roughly 115 GB/s per active card
against about 550 achievable, and all of the file's PXQ2/PXQ3 bytes are expert weights. Routing the fused
MoE down projection and the gate/up split kernel onto the half2 pair-LUT loop (`PXA_PXQ_MMV_H2_MOE`, bits
for each site) reads +2.8% control, +3.0% prose, +2.3% repetition at REPS 3 in one bracket, sub-additive
across the two sites as an ALU-bound fix should be. Smoke 12/12 and the 8k needle pass.

**Why it is not the default:** the token-0 spread gate at one slot, with base compared against itself first
(spread exactly 0, so the instrument's floor is zero), shows the lever moving the top-1 token on one prompt
and the distribution by an order of magnitude over tolerance on two more. The fp16 group accumulation that
is exact enough on dense rows is not on the expert rows at these shapes. **What would close it:** an fp32
partial-sum variant of the expert loop (accumulate the 16-element groups in fp32, keep the pair-LUT decode),
re-gated the same way; the branch and the gate scripts are ready.

