# Config cookbook — per-card recommended command lines

**This is what `tools/pxa-launch.py` does by hand.** The launcher reads the same rows out of
its own table, matches them to the cards you tick and the file you pick, and prints the flags,
the env and the source line before it starts anything — so if you just want a server running,
run [the launcher](LAUNCHER.md) and skip this page. Read this page when you want to drive it
yourself, when you want to see what a recipe is made of, or when your topology is not one the
launcher has a measured row for.

Copy-paste starting points for the cards this fork is tuned for. Every number is a measured
median (protocol: `bench/speed-bench.sh` — server-reported `timings.predicted_per_second`,
200-token temp-0 generations, median of ≥3, model fully GPU-resident; prefill = cold prompt at
the stated `-ub`). Weights: <https://huggingface.co/poisonxa/PXA-bench-files-GGUF> — first
published under `poisonxa/PXA-Fusion2-35B-GGUF`, a repository since renamed to
`poisonxa/PXA-Fusion4-35B-GGUF` that keeps the same bytes at revision
`2d7d9e7e5f445b1ea38a5686b8cf9d6bc9aa29df`.

**Every recipe below assumes a fully GPU-resident model** because that's what's fast and
what these numbers were measured on. PXQ itself no longer requires it: `-ngl < 99` and
`--n-cpu-moe` both work now, but only PXQ4 and PXQ4-HQ get a fast CPU path (an AVX2 integer dot,
`ggml/src/pxq-dot.c`); PXQ2, PXQ3, and PXQ6 fall back to a correct-but-slow dequant
(`ggml/src/pxq-cpu.c`) on CPU or under partial offload (see `docs/KNOWN-ISSUES.md`). For the
recipes here, pick the tier that fits your VRAM with ~2.6 GB headroom for compute buffer + KV.

The recommended env used by every recipe below:

```bash
export LD_LIBRARY_PATH=build/bin:build/src:build/ggml/src:build/examples/mtmd
```

That is the whole env. **The tune is on by default** (config level `ENHANCE`, since 2026-09-03):
the engine reads the card set at startup, selects the measured-good kernel levers for it, and
prints the decision (mixed-card boxes get a per-GPU line). You will see this line first:

```
PXA config level: ENHANCE (default; PXA_ENHANCE=0 for DEFAULT, PXA_REFERENCE=1 for REFERENCE)
```

`PXA_ENHANCE=1` is still accepted and is now a no-op — every published number was measured with
it set, and it is the default because shipping otherwise meant the binary people ran was not the
binary that was measured. To roll back to the pre-2026-09-03 shipped set, export
`PXA_ENHANCE=0`; for the bit-exact all-levers-off baseline, `PXA_REFERENCE=1`. Optionally add
`PXA_MODE=balance` (default, fa-on serving) or `PXA_MODE=max` (fa-off, max prefill).

**`-b` / `-ub` are also chosen for you** on the card sets the campaign measured — a 2× V100 pair
gets `-b 8192 -ub 2048`, a 2× P100 pair `-b 8192 -ub 256`, a single 1080 Ti `-b 2048 -ub 768` —
and the server says so at startup. Those three rows are the campaign's measured cells (dense 27B
PXQ4-core on the pairs, PXQ2 on the 1080 Ti); where a recipe below states its own `-b`/`-ub`,
that recipe measured them on its own model and its value wins:

```
PXA_AUTO: batch defaults for 2x P100 (sm_60) -> -b 8192 -ub 256 (measured cell; pass -b/-ub to override, PXA_ENHANCE=0 for the stock defaults)
```

Passing `-b` or `-ub` yourself always wins. Most recipes below still spell them out, because a
recipe should read as a complete command line and because several of them measured a value the
auto table does not cover; the 1080 Ti recipe is the one that now leaves them off, since the
engine's row for that card IS its measured row. Card sets with no measured cell keep the
adaptive-VRAM ladder.

> **Lab footnote:** everything the ENHANCE level arms for you — the PXQ6 kernel family
> (KSPLIT/VECX/GUFUSE/SCATFUSE), `PXA_FUSE_DELTANET=3`, `PXA_G2_ADDFUSE=1` (2026-07-19, +1.9% V100
> / +1.2% P100 decode, bit-exact), and the sm_61 `PXA_PXQ_INT8_PREFILL` carrier used in the 1080 Ti
> recipe below — is a hand-settable lab knob in its own right, each with its own measurement and
> gate class, in [`docs/LEVERS.md`](LEVERS.md). Setting any of them by hand
> bypasses the per-arch gate the ENHANCE level applies and is usually **slower**, not faster.

## Two FA regimes — pick by workload (read this before quoting a prefill number)

On these pre-Turing cards (P100/V100/1080 Ti), flash-attention is a **decode win but a
cold-prefill loss** — for this engine *and* for upstream ik_llama. You run **one** setting per
server, so choose by what you're doing. Measured (35B, cold 5.8k-token prompt, `-b 2048`, median
of 3; full sweep in `bench/fair-battle.md`):

| card | `-fa on` (interactive: chat/agent) | `-fa off` (batch: ingest/summarize/embed) |
|---|---|---|
| P100 | prefill **817** · decode **56.7** | prefill **1,213** · decode 41.1 |
| V100 | prefill **1,589** · decode **94.1** | prefill **1,700** · decode 76.6 |
| 1080 Ti | prefill **667** · decode **65.4** | prefill **1,001** · decode 34.2 |

- **Interactive serving → `-fa on`** (what the recipes below use). You get the full decode speed
  *and* a solid prefill in the same server — e.g. P100 gets **+59% prefill** vs upstream (the engine
  win). The accompanying **+30% decode** in that comparison comes from the smaller PXQ quant tier
  (PXQU-16 + q8_0 head vs upstream IQ3_KS) **plus MTP speculative decode**, not the kernel — the
  same-quant engine control is decode +2.7–3.3% (see `bench/fair-battle.md`).
- **Prefill-heavy batch → `-fa off`.** Prefill jumps 26–56% (this is where the "+88% P100
  prefill" headline comes from) but decode drops 16–48%. Use it for one-shot ingest/summarize
  passes where you barely decode.
- The recipes below are the interactive (`-fa on`) defaults. For a batch job, add `-fa off` and
  read the prefill from the right column above.

**Where that leaves this engine against mainline on prefill, measured 2026-09-13.** On a 2× V100 pair
with **stock** Qwen3.8-27B on both sides (this engine on the stock PXQ4, mainline `fdb2c11c70` on
`UD-Q4_K_S`), prompt processing reads **1,348 / 1,404 / 1,324** tok/s at 3,121 / 8,192 / 20,801 prompt
tokens against mainline's best prefill shape (its layer split) at **956 / 1,017 / 964** — **+41.1 /
+38.0 / +37.3%**, inside a 0.99% bracket. Read it for what it is: **PXQ4 against `Q4_K_S`, so codec and
engine together** — same weights on both sides, but the arm that would separate the two (this engine on
the same `Q4_K_S` file) has not been run **on this pair** (on four P100s it has, and it goes the other
way — see [`../bench/LEADERBOARD.md`](../bench/LEADERBOARD.md)). The 512-token class is not quoted at
all: its bracket moved −1.81% and mainline's own 512 arm spread 21% across repetitions. Mainline's
*tensor* split, the shape that wins its decode cells, is the slower one for prefill. The four-card
P100 prefill cell **is** measured on stock files ([`../bench/LEADERBOARD.md`](../bench/LEADERBOARD.md))
but against **pinned** competitor revisions, so it is provisional until they are rebuilt at their
current heads.

## 1× Tesla P100 16 GB — PXQU-16 (q8_0 head)

```bash
./build/bin/llama-server -m fusion2-35b-U16-q8head.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
Expected: **~62–63 t/s decode** (62.4 published; 63.0 with ADDFUSE), **827–843 t/s prefill**
@ ub2048. Decode is ub-insensitive — drop to `-b/-ub 512` if you want a smaller compute buffer.

## 1× Tesla V100 16 GB — PXQU-16 (q8_0 head)

Same command as the P100. Expected: **~101–102 t/s decode** (101.3 published; 102.0 with
ADDFUSE), **~1800–1900 t/s prefill** @ ub2048.

## 2× Tesla P100 (or V100) — PXQ4 flagship (18.7 GB, the `*-PXQ6.gguf` file)

```bash
./build/bin/llama-server -m PXA-Fusion2-35B-PXQ6.gguf \
  -c 8192 -np 1 -ngl 99 -sm layer -ts 1,1 -fa on -ctk f16 -ctv f16 -b 8192 -ub 2048 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
Expected: **55.7 t/s decode** (2×P100), **~843 t/s prefill**. Note the explicit `-b 8192 -ub 2048`:
this recipe pins them and therefore keeps them. Left unset on a P100 pair the engine would fill
`-b 8192 -ub 256`, which is the measured optimum for the *dense 27B PXQ4-core* cell at `-c 32768`,
not for this 35B MoE file at `-c 8192` — the auto table is per card set, and `-ub` still does not
transfer between models. Where a recipe here states a `-ub`, that recipe measured it; use it. The
4-bit flagship does NOT fit one 16 GB card — single-card 16 GB users want PXQU-16 instead. For the MTP variant
(`*-PXQ6-MTP.gguf`) you no longer have to pass anything: the server arms the n-gram → MTP cascade for a
file with an MTP head by itself (see *Speculation*, below). Passing `--spec-type mtp:n_max=3,p_min=0.5`
by hand still works and is what earlier releases documented — but an explicit `--spec-type` **replaces**
the auto choice, so that line now turns the n-gram stage **off**. Pass it only if you mean to.

### `-b 8192` on any multi-GPU split — read this before benchmarking a long prompt

`-b` is the prefill **chunk** size; `-ub` is the micro-batch inside a chunk. This engine
synchronizes at every chunk boundary, so a 20,801-token prompt at the engine default `-b 2048`
pays 14 of those and at `-b 8192` pays 3. On a 2× V100 layer split that is worth about **+10%**
long-prompt prefill (1,097 → 1,207 t/s at `-ub 512`) and it takes summed device utilisation from
1.24–1.33 to 1.41–1.42. `-b 20480` buys another 0.3%, i.e. nothing.

Since 2026-09-03 the engine sets it for you on the card sets the campaign measured — a 2× V100
pair, a 2× P100 pair, a single 1080 Ti — and only when you did not pass `-b`/`-ub` yourself. It is
not a blanket default because a larger chunk needs a larger compute buffer, and on a card where
the model only just fits that buffer is exactly what you do not have; outside those three cells
the adaptive-VRAM ladder still picks `-ub` and `-b` keeps the stock value. Set them by hand when
you have the headroom and you are serving long prompts across more than one card.

`-ub` does **not** transfer between configurations and there is no universal answer: with
`-b 8192`, `-ub 2048` is the best cell on a 2× V100 pair and `-ub 256` is the best on a 2× P100
pair — which is exactly why the engine only fills the cells it has measured. Measure it on your
own cards. The published pair numbers and the protocol behind them are in
[`bench/fair-battle.md`](../bench/fair-battle.md).

## 1× GTX 1080 Ti 11 GB — PXQ2 + int8 prefill tile

```bash
./build/bin/llama-server -m PXA-Fusion2-35B-PXQ2.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
No env, no `-b`/`-ub`, no `--ctx-checkpoints`, no `PXA_AUTO_SPEC=0`. The engine recognises a
single 1080 Ti and fills `-b 2048 -ub 768 --ctx-checkpoints 0` — the measured cell — and declines
to auto-arm speculation on this card. Pass any of those flags yourself and yours wins.

⚠ **The auto-spec decline is the thing keeping this card alive, and it is worth knowing why.**
The server auto-arms `--spec-type ngram-mod:n_max=4,n_min=2` for the `qwen35moe` family. That
drafter's context asks for a 254 MiB per-step checkpoint buffer and falls back to a 62.8 MiB
shadow, and 11 GiB has no room for it next to 9,907 MiB of weights, 160 MiB of KV and a 733 MiB
compute buffer: the seat loads, answers short prompts, and then dies mid-prefill on a 5.8k-token
prompt with `out of memory` (reproduced twice, 2026-09-04). Since 2026-09-03 the auto-arm
declines below 2 GiB of post-weights headroom and declines outright on a single-card sm_61 fleet,
and says so on stderr:

```
PXA_AUTO: spec DECLINED -- single-card sm_61 (11 GB class): the weights fill the card and a draft context OOMs mid-prefill
```

If you want the drafter anyway, `--spec-type ngram-mod:n_max=4,n_min=2` or `PXA_AUTO_SPEC=1`
forces it — on a card with the headroom, it is a real win.

Measured on the published PXQ2 tier, n=3, with nothing passed: cold prefill **1,363 t/s** at
`-fa off` and **747 t/s** chat prefill at `-fa on`, decode **65.3 t/s** chat and 36.7 cold. (The
older row for this card, 1,306 / 729 / 59.2 / 32.1, is the same engine launched with the levers
and the flags spelled out by hand.) Against
a fresh upstream ik_llama.cpp IQ2_KS build on the same card that is **+15.4%** cold prefill and
**+11.1%** chat decode, with chat prefill a tie (see [`bench/fair-battle.md`](../bench/fair-battle.md)).

⚠ **Those numbers depend on the sm_61 int8 prefill tile, which the default ENHANCE level arms
for you** — there is nothing to export. Without the tile the same build does 573 / 424 t/s, less
than half. Confirm it fired by looking for these two lines in the server log at startup; if they
are absent you are not running the configuration these numbers describe:

```
PXA_PXQ_INT8_PREFILL: mode 1 (N13 dp4a int8 MMQ-tile prefill, sm_61 only; ...)
PXA_PXQ_I8_BLUT: ON (N14 byte-keyed 2-bit W-decode, bit-identical)
```

`PXA_PXQ_INT8_PREFILL=1` sets the first explicitly if you have rolled the level back with
`PXA_ENHANCE=0`; it is a G3-class lever, see [`docs/LEVERS.md`](LEVERS.md) §4.
`PXA_PXQ_I8_BLUT` is already on by default and bit-identical — you do not need to set it. Use
`-ub 768`: a ub2048 compute buffer (~1.9 GiB) cannot allocate next to the resident model on 11 GB.
⚠ PXQU-12 (11.6 GB) does NOT fit an 11 GB card — it's a 12 GB tier; PXQ2 is the 1080 Ti tier.

## stock-gguf-on-pxq-engine — a stock quant on this engine, no PXQ file required

The engine fixes above — sm_60 fp16-GEMM, flash-attention regime routing, the MoE path, `np>1`
hybrid concurrency, wide f16 GEMV — apply to any stock GGUF you already have (Q4_K, MXFP4,
IQ_K, …). You do not need a PXQ file to get them:

```bash
./build/bin/llama-server -m your-model-Q4_K_M.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```

That's the whole recipe: point `-m` at a stock quant and run as normal — the tune is on by
default. **Engine-only numbers** (same GGUF, two engines — upstream
ik_llama.cpp vs this engine, matched config, isolating the kernel/arch fixes from any codec):
from the "Same-quant control" table in [`bench/fair-battle.md`](../bench/fair-battle.md) —
V100 decode 84.5 → 87.2 t/s (**+3.2%**, bit-identical output, same temp-0 sha), P100 decode 44.0
→ 45.2 t/s (**+2.7%**), 1080 Ti decode 52.2 → 53.9 t/s (**+3.3%**) — all on upstream's own IQ_K
ggufs, no PXQ tensor involved. The bigger engine-only win is prefill — see the README's
"Engine-only, the honest number" and `bench/fair-battle.md`'s regime tables for the fa-on/fa-off
split on the same stock files.

## 1× 12 GB card — PXQU-12

Same command shape as PXQU-16 with `fusion2-35b-U12.gguf`. Measured on the 16 GB Teslas:
58.4 t/s decode P100 / 97.6 V100 (see `bench/HEAD-TO-HEAD.md` §12 GB tier).

## Vision / MTP extras

- Vision: add `--mmproj mmproj-fusion2-f16.gguf` (projector loads on the first CUDA device).
- MTP speculative decode (flagship-MTP file only): armed automatically as the n-gram → MTP cascade;
  see *Speculation*, below. `--spec-type mtp:n_max=3,p_min=0.5` pins the head alone and, because an
  explicit `--spec-type` replaces the auto choice, drops the n-gram stage.

## 4xp100-flashnext — 4× Tesla P100, hybrid MoE

The lever set below needs the pipeline-scheduler fixes, the host-overhead cuts and the ported
upstream correctness fixes described in `RELEASE-NOTES-2026-09-02.md` and
`RELEASE-NOTES-2026-09-07.md`. Those are in this release; an older tagged binary will reject
several of these variables' effects silently by simply not having the code behind them.

```bash
./build/bin/llama-server -m your-flashnext-hybrid-PXQU.gguf \
  -ngl 99 -ts 5079,12612,12612,11897 \
  -ot 'per_layer_token_embd\.weight=CPU' \
  -c 150016 -b 2048 -ub 2048 -wgt 8 -t 16 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```

**There is no export block any more.** As of 2026-09-03 the eleven levers this recipe used to
export are armed by the default ENHANCE level, so the command line above IS the recipe:
`PXA_FA_GQA_PACK=4` (deep-fill decode) and `PXA_MOE_DEVICE_MAP=1` (the
device-side expert-routing table) arm on a multi-card sm_60 topology. **⚠ Neither carries a
verified *effect*:** `PXA_MOE_DEVICE_MAP`'s engagement and correctness ARE verified — its host
cross-check rebuilds the expert map and compares it element-by-element, reporting 48/48 rows
identical on the 4× P100 cell — but no A/B is on record for it; and `PXA_FA_GQA_PACK`'s effect
does not reproduce on the reviewer cell. Do not quote a throughput number for either.

⚠ **`PXA_FA_GQA_PACK` needs one more sentence, because it is the one lever here that is armed by
default and is *not* bit-exact.** Its **engagement is now proven** — a firing counter on the 4× P100
cell shows the default reaching the packed kernel (5 fires) while the `=0` control fires none. Its
**fidelity has still never been measured by an instrument that worked**: one perplexity arm never
dispatched the kernel at all, and the separate perplexity harness voided its own control (with the
lever set to `0` the OFF arm still fired 5 times on the `-b 1 -ub 1` path). Read nothing into either
KLD. If that matters for your deployment, turn this one off on its own — the override line below
does exactly that and leaves the rest armed.

The other nine (`PXA_KQ_MASK_PAD1`, `PXA_KV_SEQ_SOA`, `PXA_TOPK_RAW`,
`PXA_TOPK_MOE_MULTIROW`, `PXA_GETROWS_NARROW`, `PXA_CPY_FASTDIV`, `PXA_CONCAT_FLAT`,
`PXA_NORM_REGCACHE`, `PXA_SCHED_RESET_LAZY`) are host-side or index-math changes with no
architecture in them and arm everywhere at ENHANCE. Each stays individually overridable:
`PXA_FA_GQA_PACK=0` turns that one off without touching the rest, and `PXA_ENHANCE=0` rolls the
whole set back. The startup ledger prints every one of these decisions with its reason.

Expected, `-c 150016`, temp 0, n=7 median (1 warmup discarded), `/completion`:

| context fill | prefill | decode |
|---|---|---|
| ~3,000 tok | ~487 t/s | ~28 t/s |
| ~20,000 tok | ~411 t/s | — |
| ~86,000 tok | ~230 t/s | ~19.3 t/s |

Full raw reps, the arm-by-arm ladder, and what each lever's number depends on:
`RELEASE-NOTES-2026-09-02.md`.

> **Lab footnote:** this is the campaign's shipped set, not the whole lab. `PXA_FA_KEYS_PER_SPLIT`
> and `PXA_GEMV_RPB` were measured in the same run and are **negative** at this fill depth —
> left off deliberately, not omitted by oversight. See `docs/LEVERS.md` and
> `RELEASE-NOTES-2026-09-02.md`'s rejected-levers list before re-trying either.

## 2xv100-27b-vllm — 2× Tesla V100, 27B dense, vLLM sm_70 serving line

This recipe runs the separate vLLM-based sm_70 serving line, not the llama.cpp-based engine
above — see `docs/PXA-SM70-SERVING.md` for the full build and why the two exist. PXQ4 codec,
tensor-parallel across both cards.

```bash
export PXQ4_LIB=libpxq4_sm70_v12b.so
export PXQ4_MMV_MMA=1
export PXQ4_MMV_SPLIT_MAX_BLOCKS=300
export NCCL_P2P_LEVEL=SYS
export NCCL_BUFFSIZE=1048576

vllm serve your-27b-dense-hybrid-pxq4 \
  --quantization pxq4 --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --dtype float16 --enable-prefix-caching \
  --gpu-memory-utilization 0.88 --max-model-len 32768 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16]}'
```

`PXQ4_MMV_MMA=1` arms the v12b tensor-core decode path (batch ≥5); `NCCL_P2P_LEVEL=SYS` +
`NCCL_BUFFSIZE=1048576` fix the two V100s defaulting to a non-P2P NCCL path on a PCIe x4/PHB
topology, which was costing prefill far more than any kernel (`docs/PXA-SM70-SERVING.md`).
`--gpu-memory-utilization 0.88`, not the usual 0.92, because the inductor autotune pass OOMs at
0.92 once P2P is on. Quote the `cudagraph_capture_sizes` string exactly — bash brace-expands the
unquoted JSON.

Expected, temp 0, n=7 median, `/completion`:

| metric | value |
|---|---|
| prefill @3k | ~1,009 t/s |
| prefill @20k | ~984 t/s |
| decode, single stream | ~50.4 t/s |
| decode, aggregate @8 streams | ~190 t/s |
| decode, aggregate @16 streams | ~299 t/s |

Full raw reps and the arm-by-arm NCCL/GMU ladder: `RELEASE-NOTES-2026-09-02.md`.

## Speculation — the server picks the drafter chain for your cards

**You do not have to arm this.** When you have not passed `--spec-type`, `-md`, `PXA_AUTO_SPEC=0` or
`PXA_REFERENCE=1`, the server arms the chain that was measured best on the cards it can see, and the
boot log prints the chain it actually resolved — not a sentence about one.

### On sm_70 and newer (V100 and up): the n-gram stage ALONE, long and never wiped

```
PXA_AUTO: spec arch=qwen35 -> the n-gram stage ALONE, chain = ngram_mod
          (--spec-type ngram:n_max=64,n_min=2,ngram_size_n=24, PXA_NGRAM_RESET_STREAK=0 ...)
```

Measured 2026-09-13 on the 2× V100 pair, Qwen3.8-27B PXQ4, np1, 6 reps per cell, three output
classes, with a no-speculation bracket either side whose drift was 0.18% against a 1.50% band. tok/s,
**control / repetition / prose**:

| chain | control | repetition | prose |
|---|---|---|---|
| no speculation | 38.26 | 36.94 | 38.02 |
| **n-gram alone, `n_max=64 n_min=2` lookback 24, never wiped** | **148.12** | **75.70** | **140.97** |
| n-gram alone, `n_max=16 n_min=2` | 89.57 | 55.29 | 82.02 |
| the same 64/2/24 **with** an MTP stage behind it | 163.83 | 51.80 | 140.61 |
| mainline's n-gram + MTP cascade | 137.16 | 62.57 | 74.99 |

Against mainline: **+8.0% control, +21.0% repetition, +88.0% prose** — the only chain in two windows
that wins all three classes. Adding the trained head back buys control (+10.6%) and **loses repetition
by 46%**, because the head drafts unconditionally wherever the table declines and pays a sequential
companion decode for each of those tokens: its acceptance on that class is 0.572 against this chain's
0.898, on more than twice the proposals. The table drafts only on a match, which is why a 38-token
draft is affordable in front of nothing and ruinous behind a head.

Two honest caveats. The per-slot checkpoint budget clamped this chain's draft capacity to **38** in
that window, so these are not an `n_max=64` measurement — it won all three classes from a ceiling 20%
tighter than the MTP arm's. And no speculating arm carries a byte-identity claim: every one of them
returned more than one sha across its reps on a deterministic prompt. `PXA_REFERENCE=1` or
`--spec-type none` is how you ask for none of this.

### On sm_60 / sm_61 (P100, 1080 Ti) and on mixed fleets: the cascade, unchanged

An MTP file gets an n-gram table stage in front of the trained head, exactly as it did before:

```
PXA_AUTO: spec arch=qwen35 -> CASCADE, chain = ngram_mod -> mtp (qwen35.nextn_predict_layers=1)
```

The Volta result above was measured on a homogeneous V100 pair, and speculation was close to inert on
the 4× P100 quad in the same campaign, so Pascal and any mixed fleet keep today's chain until the quad
is measured. A model family with its own measured row in the auto table keeps that row on every card.

### Choosing the other chain

| you want | say |
|---|---|
| the previous auto chain on a Volta box | `PXA_SPEC_AUTO_CHAIN=cascade` |
| the n-gram-alone chain on a non-Volta box | `PXA_SPEC_AUTO_CHAIN=ngram` |
| the trained head with no table in front | `PXA_SPEC_NGRAM_POLICY=0` |
| no auto-armed speculation at all | `PXA_AUTO_SPEC=0` (or `PXA_ENHANCE=0`) |
| something else entirely | `--spec-type ...` — an explicit chain always wins |

**Serving two busy slots?** At `-np 2` on the workload that was measured, the n-gram stage alone was
faster than the cascade (91.0 against 81.4 t/s aggregate) even on the cards that default to the
cascade. If your traffic is concurrent, `--spec-type ngram` is worth an arm of your own.


### The n-gram stage's draft length, and how to raise it

The n-gram stage is built from the bare `ngram` **alias**, and an alias fills its own measured
defaults: the stage drafts **`n_max = 4`, `n_min = 2`** (the hash width beside it, `ngram_size_n = 16`,
is a quality knob and not a draft length — do not read it as one). Longer drafts are a per-stage
option:

```bash
# raise just the n-gram stage; the head keeps its own depth
--spec-type ngram:n_max=64,n_min=48 --spec-type mtp:n_max=3
```

**Know what you are buying.** Long drafts buy the repetitive class and pay for it in prose — on this
engine and on upstream alike. In one valid window on the V100 pair, control / prose:

| n-gram stage | control | prose |
|---|---|---|
| 4 (the default) | 86.7 | **83.8** |
| 32 | 128.5 | 57.4 |
| 64 / 48 | **185.9** | 65.6 |

So a longer fixed length is a bet on your workload, not a free win. If your traffic really is
repetitive — batch reformatting, diff application, structured extraction — raise it. If it is chat and
prose, leave it alone.

### Read the budget line at boot before you trust the length you asked for

A recurrent (DeltaNet/GDN) model checkpoints its state every speculation window, and the exact
`per-step` checkpoint buffers scale **linearly with the draft length** — on a 2× 16 GB pair with the
27B PXQ4 file, 0.6 GB across the pair at a 4-token draft and 9.6 GB at a 64-token one. Asking for more
than the cards can hold used to mean an out-of-memory a dozen cycles into the run. It no longer does:
the engine works out the largest capacity that still leaves every device its compute buffers plus a
margin, clamps every stage to it, and **prints the answer at every boot, clamp or no clamp**:

```
common_speculative_init: recurrent checkpoint budget: the chain drafts up to 4 tokens (capacity 5), this context can checkpoint 5
common_speculative_init: recurrent checkpoint budget: the chain asks to draft 64 tokens, this context can checkpoint 33 - every stage is clamped to n_max=32
```

**Two arms that differ only in this line are not comparable**, and the clamp is not visible anywhere
else. Read it before quoting a number. `PXA_CKPT_BUDGET=0` disables the clamp (and restores the
out-of-memory); `PXA_CKPT_BUDGET_MARGIN_MB` (default 256) sets the headroom it keeps.

⚠ **Do not reach for `--recurrent-ckpt-mode gpu-fallback` to buy a longer draft.** It avoids the
preallocation and it is a measured loss: control-neutral, **prose halved**, and it **changes the greedy
output**. See [`KNOWN-ISSUES.md`](KNOWN-ISSUES.md).

## The PXA attention split (`-sm attn`)

The `-sm attn` split mode originated in `ik_llama.cpp` (`ikawrakow`, 2025-12); **the PXA attention
split** is this project's version of it — the PXQ panel-aware cut, the hybrid (delta-net)
reduce-delivery fixes and demotion, the reduce route control, the model-exclusive admission and the
headroom balance: the parts that make it run on hybrid models, on the PXQ codec, on Pascal and Volta.

What the mode does: instead of giving each card a contiguous run of layers, it splits the **attention
heads and the MTP verify batch** across the pair while the rest of the model stays layer-split, which
is worth having because a layer split leaves one card idle through the other's layers. What the PXA
attention split adds is what makes it run here at all — on a PXQ file, on a DeltaNet hybrid, on
Pascal and Volta: a **panel-aware split granularity** (a PXQ tensor keeps one fp16 anchor per 64-row
panel header covering all of `K`, so it can only be cut on panel boundaries — cut anywhere else and
the slices are read at the wrong byte offsets in silence, because every assertion on that path is
about sizes and PXQ satisfies those exactly), the **hybrid reduce-delivery fixes** with their
demotion guard and consumer-side fallback, the **reduce route controls and timers**, and the
**model-exclusive admission lever** below.

```bash
# 2x V100, a Qwen3.8-27B PXQ4 file, the PXA attention split (-sm attn), default cascade
PXA_ATTN_SPLIT_QWEN38=1 PXA_REDUCE_NCCL=0 \
./build/bin/llama-server -m Qwen3.8-27B-PXQ4.gguf \
  -c 32768 -np 1 -ngl 99 -sm attn -ts 1,1 -fa on --kv-unified \
  --host 0.0.0.0 --port 8080
```

The admission is sufficient on its own — the demotion branch is skipped when it fires, so the blanket
bypass below is **not** part of this recipe. Confirm it on the first boot; the admission prints a block
that names itself, and if it does not appear, the file is not the one the fingerprint describes:

```
PXA_ATTN_SPLIT_QWEN38=1: '-sm attn' ADMITTED for Qwen3.8-27B
```

Measured against `-sm layer` on the same pair, same window, np1 temperature 0:

| cell | `-sm layer` | `-sm attn` |
|---|---|---|
| default cascade, control | 85.7 | **103.9** (+21%) |
| default cascade, prose | 81.9 | 77.6 (−5%) |
| MTP head alone, control | 58.6 | **66.1** (+13%) |
| 64/48 n-gram stage, prose | 65.6 | **107.0** (+63%) |

That last row is the interesting one: at a long draft length the PXA attention split produces the highest
prose figure measured on this box by any of the three engines.

⚠ **It is off by default, and the reason is specific rather than cautionary.** `PXA_ATTN_SPLIT_QWEN38`
admits `-sm attn` for **exactly one model shape** — `qwen35` carrying the Qwen3.8-27B hparam
fingerprint. Its verify-batch fidelity **has now been measured** on a purpose-built instrument
(`llama-perplexity` at the speculative verify width, 12 chunks × 512, stock PXQ4): mean KLD **0.000684**
on prose and **0.0106** on repetitive text against the layer split — the same level the
flash-attention on/off toggle reads on that instrument — with greedy text and draft counters
byte-identical. It **misses the pre-registered default-on criterion on two tail statistics** (top-1
agreement short by 2 rows in 3060, one wider worst-case logit), so it is a **measured option you may
opt into**, not a default, and the default waits on an instrument for the rejection and
checkpoint-restore path. If the lever is on and your file is not the admitted one, the boot prints the
fingerprint it observed against the one it wants, field by field, so it never fails silently.

⚠ **`PXA_ALLOW_GRAPH_SPLIT_HYBRID` is not this, and you do not need it here.** It is a blanket bypass
that admits four hybrid architectures alike, and its own lever row says it must not serve traffic.
The arms in the table above were taken with it, because they predate the admission; the admission is
what replaces it for this one model. `-sm graph` is **not** admitted by either and stays demoted.

⚠ **One honest caveat about the lever itself:** the numbers above were taken through the blanket
bypass, so the *admission path* has not yet been exercised end to end on a serving load. If the
fingerprint were off by a field the lever would be a no-op — which is exactly why it prints the
observed fingerprint against the wanted one when it declines, and prints the ADMITTED block when it
fires. Read one of those two blocks on your first boot; do not assume.

## Quantizing your own model

See the README "Quantize your own" section — pure tiers (`PXQ4`, `PXQ3`, `PXQ2`) or a
mixed-tier PXQU map (`--pxq-universal <map>.tiers`, `docs/PXQU-CONVERT.md`), plus:
- **`--output-tensor-type q8_0`** (recommended): +5.2% decode on P100 for +123 MB.
- **Imatrix doctrine:** the PXQ tiers **ignore** an offered imatrix (since 2026-08-24;
  consuming it measured worse than not consuming it — `docs/QUANTIZING.md`), so a calibration
  run buys a PXQ target nothing. For a **stock** target it still pays: quantizing a merged
  model, recompute the imatrix ON the merge (activation statistics are anchor-specific),
  full-GPU-resident (the CPU/partial-offload capture path crashes — `docs/KNOWN-ISSUES.md`).

## Leave PXQ, export and requantize to a stock type

For when you want a file stock llama.cpp or ik_llama.cpp can load, or a different bit width than
any PXQ tier offers:

```bash
./build/bin/llama-pxq-export in.gguf out-f16.gguf
./build/bin/llama-quantize --allow-requantize --i-know-this-is-double-lossy --imatrix m.imatrix out-f16.gguf out-Q4_K_M.gguf Q4_K_M
```

From the release tarball the same two commands are `./bin/llama-pxq-export` and
`./bin/llama-quantize` — both binaries are in the package, so leaving PXQ needs no build.

`llama-pxq-export` decodes the PXQ GGUF tensor by tensor into plain F16 (add `--type f32` for
F32, `--cpu` to decode without a GPU, `--verify` to cross-check the chunked decode); everything
that isn't a PXQ tensor copies through byte for byte. `llama-quantize` then requantizes that plain file the normal way — pass
`--i-know-this-is-double-lossy` even here: most PXQ files carry a few non-PXQ tensors (an
attention head or two, an embedding) already stored as `q8_0` or `mxfp4`, copied through
verbatim by the export step, and `llama-quantize` refuses to requantize an already-quantized
tensor without that flag. The result loads in stock llama.cpp, no PXQ support required on the
reading end — proved end to end, CPU-only, export then requantize then a stock `llama-cli`
load, on a real PXQ4 file.

Say the quality cost plainly, in the tool's own words (`llama-quantize --help`, on
`--i-know-this-is-double-lossy`): *"Two lossy passes compound: the second codec fits a grid to
weights that were already snapped, so quality is strictly below quantizing the original
F32/BF16 once."* Exporting to F16 first does not undo that: the weights already went through
one lossy PXQ pass, so `out-Q4_K_M.gguf` here is a second lossy pass on top of it, not the same
quality as quantizing the original BF16/F32 straight to Q4_K_M. Keep the original checkpoint if
you expect to want more than one target format.

(`llama-quantize --allow-requantize --i-know-this-is-double-lossy` also accepts a PXQ file
directly as `in.gguf`, skipping the export step; the two-step recipe above is the one to reach
for when you want the intermediate F16/F32 file too.)
