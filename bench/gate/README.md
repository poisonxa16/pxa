# `bench/gate/` — the release gate

One command decides whether a build is shippable. It runs on any box with the engine built,
takes no container, and hard-codes no path.

```
MODEL=/path/to/model.gguf ./bench/gate/run-gate.sh
```

It exits **0 only if every check passed**. Any failure, and any check that could not be run
honestly, is printed as `FAIL` or `SKIP` with the reason. A `SKIP` does not fail the gate — but a
gate with skips has not proved what a clean gate proves, so read the summary line, not just the
exit code.

## What it checks

| # | Check | What a failure means |
|---|---|---|
| 1 | **Greedy determinism, np=1.** The same prompt `REPS` times (default 12) at `temperature 0`, `top_k 1`, `cache_prompt false`. Every output must be **byte-identical**. | The engine is not deterministic at batch 1. Anything from an uninitialised buffer to a race in a fused kernel lands here first. This is the single most sensitive check in the gate. |
| 2 | **Coherence.** A short factual completion (`prompts/coherence.txt`) must contain the expected answer. | The model is loaded and the dequant/GEMM path produces sense, not noise. Catches a codec or scale regression that determinism alone would happily reproduce identically. |
| 3 | **Needle recall.** Both campaign prompts (3,121 and 20,801 tokens) must recall *both* planted identifiers, `NEEDLE_REPS` times each (default 4), with a stable sha across runs. | Long-context attention or KV handling is wrong. A model that answers fluently but has lost the first line of a 20k prompt fails here and passes check 2. |
| 3b | **Logit reproducibility** (the primary determinism check). `LOGIT_REPS` runs (default 6) of the 3,121-token prompt at `n_predict 1`, `n_probs 2`, run once at np=1 and once pinned to slot 1 at np=2; every returned probability must be identical to the last digit. | The forward pass itself is nondeterministic. Checks 1, 3 and 4 compare the *argmax*; this one compares the numbers the argmax was taken over. A kernel race can wobble the logits on every single run and still produce a stable sha wherever the top-2 margin is wider than the wobble — which makes it look like a rare flake on long prompts and under load instead of what it is. Skipped, not failed, on a server that does not return `completion_probabilities`. |
| 4 | **Greedy determinism, np=2, other slot erased first.** Boots with `-np 2`, erases every slot, then runs `REPS` requests pinned to slot 1. All must be byte-identical, and (unless `NP2_CROSS_CHECK=0`) must equal the np=1 reference. | A multi-slot defect: state leaking between slots, a shared scratch buffer, a graph outliving its call. Skipped, not failed, if the server has no `-np` or no slot-erase endpoint. |
| 5 | **Unit tests.** `test-pxq-cpu-dot`, `test-kv-seq-shadow`, `test-narrow-kernel-parity`. | The CPU PXQ dot product, the KV sequence shadow, or a narrow-kernel lever has diverged from its reference. |

## The two rules that make the multi-slot arm meaningful

These are not style preferences. Both were established by chasing what looked like real defects
and turned out not to be, and both cost hours before they were written down.

### 0. A stable sha is not determinism

Check 3b exists because of 2026-09-03. The release gate on `rel/v2026.09.03` reported np=1 12/12
byte-identical and then flipped once in four runs at 20,801 tokens; the box was busy, and the flip
was nearly written off as a benign near-tie. It was not. Asking the same server for `n_probs 2`
showed that the token-0 probability of an identical prompt was **different on essentially every
run in every arm**, including the arms that passed 12/12: 0.4964–0.5339 across 48 runs at 3,121
tokens, 0.3937–0.4339 across 12 runs at 20,801. The cause was a write-after-read race in the
DeltaNet out-gate fusion (`PXA_FUSE_DELTANET` bit 1), and the greedy sha was stable only because
most top-2 margins are wider than 4e-2. The token that flipped had a margin of 4.9e-4.

So: **never conclude "benign near-tie" from output shas alone.** A near-tie in a deterministic
engine reproduces the same tie the same way every time. If the outputs move, the logits moved, and
the only honest next step is to look at them.

### 1. Twelve out of twelve, at np=1 **and** at np=2

A lever is not shippable on six clean runs, and it is not shippable on np=1 alone. The failure
mode that matters is *rare*: a fused DeltaNet lever that produced identical output 11 times in 12
at np=1 produced a second, different sha on the 12th, and a different lever produced 9 identical
plus two of one variant plus one of a third at np=2 while being 12/12 clean at np=1. Both were
real bugs — a kernel writing into storage it had inferred, rather than checked, was dead. So:

* **12 runs per arm, both arms.** Anything less than 12/12 identical in either arm is a fail, not
  a flake, and the lever goes back off by default until the mechanism is understood.
* Fixing it means *checking* the claim the kernel makes about the graph (is this tensor really
  dead? is this node really the sole consumer?), not tightening the pattern match that inferred
  it.
* When a lever is a bitmask, gate each bit. Only the bits that are guarded and meet 12/12 in both
  arms belong in the default mask; the fallback is mask 0.

### 2. Compare slots at the **same KV placement**, never across offsets

Two slots at *different* KV offsets legitimately produce different text from the same prompt, and
this is not a defect — not in this engine and not in mainline.

The reason: with a shared KV ring, a request in slot 1 sitting behind slot 0's content sees a
different `n_kv`, which changes the attention reduction tiling, which changes floating-point
summation order by a few ULPs. Almost always invisible; occasionally it lands on a near-tie. In
the measured case, slot 0 at offset 0 and slot 1 at offset 3266 agreed for their first 12 tokens
(mean |Δlogprob| 0.026, max 0.084) and then diverged at a token where the top-2 gap was
**0.005 nats** — a coin toss the tiling decided. Both answers were correct; both recalled the
needle. Erasing slot 0 so slot 1 also started at offset 0 made the two byte-identical with a
0.0 logprob delta. The same behaviour appears with flash attention off, so it is not specific to
one attention path.

So the rule is:

* **Erase the other slots before you compare.** Then a byte comparison is valid, and that is what
  `run-gate.sh` does — hence the `--slot-save-path` it passes at `-np 2`, which is what registers
  the `POST /slots/<n>?action=erase` route.
* **Never compare raw shas across slots at different offsets.** If you cannot erase, compare
  logprobs with a tolerance instead, or compare each slot only against a reference taken at its
  own placement.
* A sha difference between two *differently placed* slots is evidence of nothing. Chasing it costs
  a night.

## Configuration

Everything is an environment variable; every one has a default.

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | — | **Required.** Path to the `.gguf` under test. |
| `BIN` | auto | Directory holding `llama-server` and the test binaries. Auto-detects `build/bin`, `build-spd/bin`, `build-tok/bin`, `build/Release/bin` under the repo root. |
| `GPUS` | unset | Sets `CUDA_VISIBLE_DEVICES`, e.g. `GPUS=2,4`. Left alone if empty. |
| `HOST` / `PORT` | `127.0.0.1` / `18080` | The gate boots and kills its own server; pick a free port. |
| `NGL` / `CTX` / `BATCH` / `UBATCH` / `THREADS` | `99` / `32768` / `2048` / `2048` / `nproc` | Standard server sizing. |
| `FA` | `on` | Passed as `-fa $FA`. Set `FA=` (empty) to omit the flag on a build that has none. |
| `SERVER_ARGS` | empty | Anything extra, verbatim: `-ts`, `-ot`, `--kv-unified`, tensor overrides. |
| `REPS` | `12` | Determinism repetitions per arm. Do not lower it for a release gate — see rule 1. |
| `NEEDLE_REPS` | `4` | Needle repetitions per prompt. |
| `N_PREDICT` | `32` | Tokens generated per determinism/needle request. |
| `GATE_NP2` | `auto` | `auto` runs the np=2 arm if the server has `-np`; `0` disables it; `1` forces it. |
| `NP2_CROSS_CHECK` | `1` | Also require the np=2 slot to equal the np=1 reference. |
| `GATE_TESTS` | `1` | `0` skips the unit tests (e.g. on a box where they were not built). |
| `COHERENCE_EXPECT` | `paris` | Case-insensitive substring the coherence completion must contain. Change it with `prompts/coherence.txt`. |
| `BOOT_TIMEOUT` / `REQ_TIMEOUT` | `900` / `900` | Seconds. Raise `BOOT_TIMEOUT` for a cold model on spinning disks. |
| `PROMPTS` | `bench/gate/prompts` | Prompt directory. |
| `WORKDIR` | a fresh `mktemp -d` | Server logs and per-test output land here; the path is printed in the summary. |

## Examples

Two V100s, the release model, the defaults:

```
MODEL=/models/Qwable-27B-PXQ4core.gguf GPUS=2,4 ./bench/gate/run-gate.sh
```

A single card, smaller context, no unit tests built yet:

```
MODEL=/models/model.gguf GPUS=0 CTX=8192 UBATCH=512 GATE_TESTS=0 ./bench/gate/run-gate.sh
```

A CPU-only sanity pass:

```
MODEL=/models/model.gguf NGL=0 FA= GATE_NP2=0 ./bench/gate/run-gate.sh
```

Tensor split and an override, passed straight through:

```
MODEL=/models/big.gguf GPUS=0,1,5,6 \
  SERVER_ARGS="-ts 5079,12612,12612,11897 -ot per_layer_token_embd\.weight=CPU" \
  ./bench/gate/run-gate.sh
```

## The prompts

`prompts/needle3121.txt` and `prompts/needle20801.txt` are the two campaign long-context prompts:
a planted first line carrying two identifiers (`MAGENTA-7741`, `BLUE-HERON`), then ~3.1k / ~20.8k
tokens of deterministic word-salad filler drawn from a fixed 60-word vocabulary, then a question
asking for both identifiers back. The filler is synthetic — there is no prose, no real document
and nothing private in either file — which is exactly why they are safe to ship and why they make
a clean determinism fixture: the model has nothing to latch onto except the needle.

`prompts/coherence.txt` is a one-line factual completion. Keep it short: its job is to catch
"the engine is deterministic but the arithmetic is wrong", not to measure quality.

## Notes

* **Dependencies: `bash` and `python3`.** Nothing else. It deliberately does not use `curl` —
  CUDA build containers routinely ship without it, and the gate has to run wherever the engine
  was built, not only where someone remembered to install a client.
* The gate never touches a GPU it was not given. If several jobs share a box, hold whatever lock
  that box uses around the whole invocation — the gate boots a server and expects the cards to
  itself for the duration.
* It kills its own server on exit, including on Ctrl-C.
* `cache_prompt` is false on every request. A determinism check that reuses a prefix cache is
  measuring the cache, not the engine.
* `seed` is pinned and sampling is `temperature 0`, `top_k 1`. If a build needs a seed to be
  deterministic at temperature 0, that is itself the finding.

**Answer budget.** `N_PREDICT` defaults to 256. Thinking models (stock Qwen3.8 and friends) open every answer with a `<think>` block that eats about a hundred tokens, so a 32-token budget truncated every needle answer while all determinism arms were green; 256 is harmless for no-think models and recalls the needles on both kinds.
