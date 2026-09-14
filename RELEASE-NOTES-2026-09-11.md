<p align="center"><img src="docs/assets/pxa-network-banner.png" alt="PXA Network" width="760"></p>

# PXA v2026.09.11-rc3 — release candidate

Release candidate. Supersedes `v2026.09.09-rc3`, which is the last tag anyone outside this project
has seen. This note covers what changed since that tag and nothing else; the full reference for
every lever, its default, the configuration it was measured on and its verification class is
[`docs/lab/LEVERS.md`](docs/lab/LEVERS.md).

The distance from `v2026.09.09-rc3` is large: new model support, a changed decode default on one
architecture, and a speculation default that was declared in the code but unreachable in practice.
Every number quoted below was taken on the binaries in this package. Where a change has no number
in this note, that is because the number lives in the LEVERS row, not because it is absent — the
row is the citation.

---

## Read this first

**On a P100 (compute capability 6.0) the low PXQ tiers now decode through a half2 inner loop by
default.** `PXA_PXQ_MMV_H2` unset resolves to `3` (both low tiers) on `cc == 600` **exactly**, and
to `0` on every other architecture — so an unset environment gets the loop on a GP100 and the exact
fp32 decode loop everywhere else, including the 1080 Ti, whose fp16 rate is 1/64 and for which this
loop would be a loss. `PXA_PXQ_MMV_H2=0` restores the fp32 loop on a GP100.

The mechanism: the shipped `pxq6_dot32` issues the same 16 pair decodes per 32 weights for every
tier, so a 2-bit tier moves 39% fewer bytes than a 4-bit one and decodes at the same speed on a P100
(18.50 vs 18.01 t/s). That loop is ALU-bound, and the int8 route that fixes it on sm_70 needs DP4A,
which GP100 does not have. GP100 does have full-rate `HFMA2`, and the LM4/LM8 books are fp16-exact
(enforced by a startup self-check), so a half2 pair LUT carries **zero book error** — the entries are
the same numbers, not roundings of them.

Measured on 27B low-tier files, 2× P100 (cc 6.0), `srv n256`, medians of 7, in **both arm orders**,
reporting the order-cancelled mean because a decode-only kernel cannot move the prefill control and
whatever that control does is the drift:

| tier | np1 | np2 |
|---|---|---|
| PXQ2 uniform | +9.54% | +11.34% |
| PXQ3 uniform | +14.42% | +16.06% |
| PXQ2-attn4 | +4.27% | +5.59% |
| PXQ3-balanced | +6.60% | +7.32% |

Absolute anchor: PXQ2 uniform tg32 **18.587 → 20.385 t/s**.

This lever is **not bit-exact against the fp32 arm, by construction** — `x` is rounded to fp16 once
at stage time and the 16-element group accumulates in fp16 — so it is held to the project's fidelity
clause rather than to a hash: mean KL divergence against the file's own exact path no worse than the
PXQ4 fused kernel's `0.000951`, and same-top-p not below `98.710%`. Measured mean KLD
**0.000026 – 0.000032** (30–37× under the bar) and same-top-p **99.64 – 99.75%**.

⚠⚠ **Gate this lever with `-b 8 -ub 8`, never the default batch, and read the statistic the bar
names.** `llama-perplexity` is pure prefill and the dense 2D decode driver only accepts `ny <= 8`, so
at `-b 512` both arms take the prefill GEMM and return bit-identical logits — a false PASS on a run
in which the kernel never executed. In the other direction, a greedy hash is not a pass condition
either: the arithmetic differs by construction, so identical text would mean the kernel never
engaged. Use the kernel's own first-fire banner as the engagement check. And note that the run log's
headline prints a **maximum** KLD (0.020244 on one file) directly beside a bar that names a **mean**
(0.000951); those two numbers are 21× apart, the maximum exceeds the bar on two of three files, and
the mean is 30–37× under it. Read `Mean KLD` from the per-arm log, never the run log's headline.

**Also in this release: dense Gemma 4 is supported**, and the declared default speculation for MTP
models — an n-gram → MTP cascade — was unreachable in `v2026.09.09-rc3` because the guard asked a
different question. Both are described below.

---

## Decode

- **The low PXQ tiers get the split decode kernels PXQ4 already had.** `PXQ23_MMV_SPLIT` /
  `PXQ23_MMV_MT` remain on by default.
- **`PXA_PXQ23_MMVQ`** — PXQ2 and PXQ3 decode riding the same q8_1 MMVQ kernel PXQ4 uses. This is
  the case where the *smaller* file was stuck on the *slower* path. It is **off by default and it
  stays off**: it is fast and it fails the fidelity clause. PXQ2 measures mean KLD `0.004624`
  against a comparator of `0.000951` — 4.9× the bar — and PXQ3 `0.002186`, 2.3×. Neither is "no
  worse than". The full argument, including the confound that cannot be unconfounded, is in the
  LEVERS row. If +36% decode is worth +0.35% perplexity to you, the row documents the trade.
- **`PXA_PXQ_MMV_H2`** — see *Read this first*.

## Prefill

- **PXQ3's prefill dequant stops re-deriving what a 64-entry table already knows.**
- **`PXA_FA_TILE_V2`** — a flash-attention tile schedule that can issue a 128-bit shared load. Ships
  **off**; the LEVERS row carries the measurement that decided it.

## Speculation

- **A `--spec-type ngram` alias, and an MTP-head-aware auto policy** for the self-speculation
  family. The first honest measurement of that family is in the LEVERS row, including why its
  control was not a control.
- **An MTP file gets both drafters, in series, by default.**
- **The n-gram → MTP cascade default was declared but unreachable.** The guard read
  `nextn > 0 && pxa_spec_ngram_policy_on()`, and `pxa_spec_ngram_policy_on()` answers a *different*
  question than the one the default needed answered — so the cascade the code advertised was never
  selected, and the boot line said so plainly for anyone reading it. Fixed; the declared default is
  now reachable. Its verifying arm — an MTP file with `PXA_SPEC_NGRAM_POLICY` **unset**, confirming
  the boot line reads the cascade ON and that throughput matches the pinned-environment arm — is
  scheduled rather than done, and is called out here rather than left implied.
- **Load-adaptive draft depth (`PXA_SPEC_ADAPTIVE_LOAD`).** A fixed depth is a bet on spare
  capacity, which is what this lever is for. It was briefly made the default on a +37% claim and
  reverted when that claim did not reproduce — **neither state ever reached a published package, so
  the user-visible default is unchanged: off**, and `PXA_SPEC_ADAPTIVE_LOAD=1` arms it. Re-measured
  with an order-alternated harness — 16 alternating pairs, np4, 4 clients, 27B PXQ4, both arm
  orders — the effect is **+10.70% order-cancelled**, with the prefill control licensing the
  cancellation. It stays off, and the row states exactly what a flip would still require: an np1 arm
  establishing that the policy is **inert at one slot**, which is a design claim and not a measured
  one.

## Quantisation

- **The DeltaNet output projection is protected at `q8_0` on every PXQ level** (`PXA_PXQ_SSM_OUT`).
  Quantizer-side.
- **`pxa.pxq4hq.book` / `pxa.pxq4hq.sub`** — a mixed 4-bit file now describes itself, so a reader no
  longer has to guess which book and which sub-block layout produced it.
- **`pxq4hq` reaches the vLLM sidecar**, with a tier table, dispatch, and a converter that stops
  guessing.

## Models

- **Dense Gemma 4**: it runs, it converts from the released weights instead of refusing them, and it
  has a PXQ tier set. It refuses only the shapes that actually fail, and the tier that does not work
  is documented as not working rather than omitted. `PXA_AUTO_JINJA` resolves on for the Gemma 4
  architectures.

## Fixed

- **The KV pool grid verifier had never compared the grid.** A verifier that could not fail is now
  one that can.
- **Two artifact gates were vacuous on a file with no native `pxq4`** — they passed by not
  executing. They now decline loudly.
- **A 16-byte shared load needed a built-in type**, or `ptxas` would not emit it at all.

## Packaging

- **`bench/gate/LAST-RUN.md` is no longer shipped.** It is the record of a gate run *on the build
  machine*, so it carried build-machine paths and internal model filenames. Every other file in that
  directory was copied by name; this was the one wildcard, so it shipped whatever happened to be in
  the directory when the package was cut. It was absent from `v2026.09.09-rc3` for a reason nobody
  chose — that package was cut before the run it records had happened, so there was nothing to ship.
  The gate *tooling* is unchanged and still ships.
- **The packager now refuses to tar a package that leaks build-machine paths**, so the next
  occurrence fails the build rather than the audit.

---

## Levers whose default changed in this release

| lever | default | note |
|---|---|---|
| `PXA_PXQ_MMV_H2` | `3` on `cc == 600` exactly, `0` elsewhere | **new**, and the only new lever that ships on by default |

**No lever row in [`docs/lab/LEVERS.md`](docs/lab/LEVERS.md) was modified between `v2026.09.09-rc3`
and this release — the diff is additions only.** Every lever that existed at that tag carries the
same default it carried there.
