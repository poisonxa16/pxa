# The PXQ4 vLLM backend

PXA Network ships **two** runtimes for one quantization family.

| runtime | what it is | hardware | PXQ tiers |
|---|---|---|---|
| `pxa` (this repo) | the GGUF-native llama.cpp engine | sm_60 Pascal, sm_61, sm_70 Volta, and newer | **all of them** |
| `vllm-pxq4` (`tools/vllm-pxq4/`) | a vLLM quantization plugin | sm_70 Volta, sm_60 Pascal | **PXQ4 only** |

This document covers the second one: what it is, what it will and will not load, how
to convert a model for it, how to start a server, which knobs actually move the
numbers, and the two flags a server needs before it will answer a `tools=` request.

Choosing between the two engines for a given model, card set and concurrency is what
[`tools/pxa-launch.py`](../tools/pxa-launch.py) automates. See
[Choosing an engine](#choosing-an-engine) — you do not have to make this call by hand.

---

## 1. What the backend is

`vllm-pxq4` is an **out-of-tree vLLM quantization backend**. It plugs into stock vLLM
through the documented plugin surface and **patches zero lines of vLLM**:

- a `vllm.general_plugins` entry point, `pxq4 = pxq4_vllm:register`, which calls
  vLLM's `register_quantization_config` at startup;
- a `PXQ4Config` that a checkpoint can self-select via `override_quantization_method`,
  so `--quantization pxq4` is an assertion rather than a discovery;
- a `PXQ4LinearMethod` that owns weight creation, tensor-parallel sharding and the
  forward call for every module the checkpoint declares as PXQ4;
- a standalone torch extension (`libpxq4_*.so`) that registers the `pxq4::*` operators
  through `TORCH_LIBRARY`. It links **libtorch and cudart only** — no vLLM header, no
  vLLM object, no vLLM rebuild.

Because the kernel library is a plain torch extension, upgrading vLLM does not require
rebuilding it; upgrading **torch** does, because the ABI is torch's.

What you get from vLLM that llama.cpp does not offer: tensor and pipeline parallelism,
paged KV, continuous batching, CUDA-graph capture, and real data parallelism.
llama.cpp's `-sm layer` is a *serialized* multi-GPU pipeline, so concurrent requests
queue behind one another — which is the whole shape of the crossover in §3.

### What ships where

| path | contents |
|---|---|
| `tools/vllm-pxq4/src/` | plugin source, CUDA kernels, the GGUF→vLLM converter, the parity harness |
| `tools/vllm-pxq4/docs/` | the format spec, kernel notes, plugin-surface analysis and the design record |
| `pxa/pxq4/kernels/` | prebuilt kernel libraries, one per GPU arch + revision |
| `pxa/pxq4/sidecar/site-sm60/` | the `pxq4_vllm` plugin tree vLLM loads via `PYTHONPATH` |
| `scripts/pxa-serve-sm70.sh` | V100-class serving launcher, measured defaults |
| `scripts/pxa-serve-sm60.sh` | P100-class serving launcher, measured defaults |
| `tools/pxa-launch.py` | picks the engine, prints the evidence and the command |

`pxa/pxq4/MANIFEST.md` lists every shipped binary with size and md5.

---

## 2. Quant tier support — the matrix

**The vLLM backend implements three tiers: PXQ2, PXQ3 and PXQ4.** Everything else is a
llama.cpp job.

> **PXQ2/PXQ3 GPU gate status, `v2026.09.05`: not gated on GPU.** Correctness of the
> tiers themselves is CPU-gated green (`gpu_selftest.py` bit-exact against a host oracle;
> the converter and TP-shard-placement gates pass on CPU), and the code ships. The on-GPU
> window — same-top-token vs the llama engine (70/70 target), needle recall, and speed on
> one P100 and one V100 — **did not run before the tag**: the tier arms were served at
> `--max-model-len 8192` while the benchmark prompt is ~20k tokens, so the cells returned
> HTTP 400 rather than numbers, and the re-run did not get a window. The tiers are
> therefore available but **not a gated claim** in this release; the window is the first
> item of the next tier release. Raise `--max-model-len` before benchmarking them.
> 
> 

| tier | bits | slab bytes | `pxa` (llama.cpp) | `vllm-pxq` | note |
|---|---|---|---|---|---|
| **PXQ4** | 4 | 1088 | yes | **yes** | the original vLLM tier; its kernels are unchanged |
| **PXQ3** | 3 | 832 | yes | **yes** | bit-plane codes; added 2026-09-05 |
| **PXQ2** | 2 | 576 | yes | **yes** | added 2026-09-05 |
| PXQ4-HQ | 4 | 1152 | yes | no | bs8 sub-scales; no vLLM kernel |
| PXQ6 | 6 | 1088 | yes | no | GPU-only; no CPU codec |
| PXQ1 | 1 | 320 | yes (GPU only) | no | no dense path and no CPU codec |
| PXQ_UNIVERSAL | mixed | — | yes | see below | a per-tensor mix; servable only if every tier it uses is |

The three served tiers are **one layout with three code widths**. All of them use 64-row
panels, a 128 B fp16 row-anchor header per panel, 32-column slabs, a 64 B sub-scale SoA per
slab, the *same* SUB16 lookup table, and the same parity-locked reconstruction

    eff = fp32(anchor_fp16) * SUB16[nibble]        once per 16-element block
    w   = eff * fp32(book[code])                   per element

They differ in exactly two things: how many bits a code is (hence 8 / 12 / 16 code bytes per
row per slab, hence 576 / 832 / 1088 slab bytes) and how many entries the book has (4 / 8 /
16). Sharding rules, parameter shapes, loader semantics and the CUDA-graph capture argument
are identical across all three — which is why the tier is a table lookup in the kernels and
not three code paths.

**The slab stride is the tier, and it is checked rather than inferred, everywhere.** A PXQ2
tensor read at the PXQ4 stride is a well-formed array of the wrong bytes: it loads, it shards,
it passes every shape assertion, and it generates fluent nonsense. So the stride is asserted
against the parameter's own last dimension at `create_weights`, again inside every op, and
again against the checkpoint's own tensors at load.

Two further properties that matter operationally:

1. **The refusal is clean and early.** A non-PXQ4 tier is rejected at the conversion
   gate, not at load time and never at generation time. There is no silent
   wrong-output path on vLLM: you either get a converted checkpoint or an error.
2. **Mixed tiers ACROSS modules are supported; mixed tiers INSIDE one parameter are not.**
   This is the rule that makes a real artifact servable, so it is worth stating precisely.
   A vLLM *parameter* has one slab stride and one book, so everything that ends up inside one
   tensor must be one tier. Different parameters may be different tiers. Concretely, vLLM's
   FusedMoE keeps gate and up concatenated in `w13` and down in `w2`: gate and up must
   therefore match, and down need not. A file whose routed experts are PXQ2 for gate/up and
   PXQ3 for down — which is what the quantizer emits for the larger MoEs — is expressible
   without weakening the invariant. The converter enforces this per parameter and refuses to
   write anything else.

3. **A PXQ4 file is not uniformly PXQ4.** A real dense 27B artifact parses as
   325 pxq4 + 132 q8_0 + 1 q6_K + 360 f32 + 48 mxfp4 tensors. The backbone
   (attention k/v, the output head, norms) is deliberately held at higher precision by
   the allocation table. Any design that assumes one type throughout is wrong, and the
   converter enforces the consequence: **every fused vLLM module served by
   `PXQ4LinearMethod` is uniformly PXQ4 across all of its output partitions.** There is
   no mixed-precision fused module, ever. That is why the default policy leaves
   `self_attn.qkv_proj` in fp16 even though `attn_q` is already PXQ4 on disk.

### What a tiered checkpoint declares

A tiered checkpoint sets `quantization_config.quant_method` to **`pxq`** (the superset) and
adds two keys. A single-tier PXQ4 checkpoint keeps saying `pxq4` and is bit-for-bit
unaffected — as is every launcher that passes `--quantization pxq4`.

```json
"quant_method": "pxq",
"pxq_tiers":  { "mlp.experts.w13": "pxq2", "mlp.experts.w2": "pxq2",
                "self_attn.o_proj": "pxq4", "linear_attn.in_proj_qkvz": "pxq4" },
"tier_books": { "pxq2": [ ... 4 floats ... ], "pxq4": [ ... 16 floats ... ] },
"tier_sub":   [ ... 16 floats ... ]
```

`pxq_tiers` keys are module suffixes, matched longest-first, exactly like `pxq4_modules`; the
`.w13` / `.w2` suffix names a fused FusedMoE parameter rather than a module. The **on-disk
tensor keys do not change**: they are still `<module>.pxq4_slabs` and `<module>.pxq4_anchor`.
That is a wire-format name, like the `pxq4` torch namespace — the tier travels in
`config.json`, never in a key — and keeping it is what lets every checkpoint already in the
field load untouched.

`tier_books` is not bookkeeping. The quantizer's books are overridable at build time
(`PXA_PXQ2_V3`, `PXA_PXQ_CEIL_V2`, `PXA_PXQ*_BOOK`) and the GGUF records what was actually
used in `pxa.pxq2.book` / `pxa.pxq3.book`. Decoding a v3-book file with the v1 table is a
silent, uniform weight error across every expert in the model, with no shape, checksum or
load-time symptom. So the converter copies the file's own tables into `config.json`, and the
runtime **refuses to serve a PXQ2/PXQ3 module whose book the checkpoint did not record**
rather than falling back to the compiled-in default. The SUB16 table is shared by all three
tiers, so there is one of it, and the converter asserts every tier in a file agrees on it.

### The tier that fits: Fusion2 / Fusion4 35B on one card

The 35B `qwen35moe` artifacts are the reason PXQ2 and PXQ3 exist on this backend. Their tensor
inventory is not what the name suggests, and the numbers below are read off the file's own
tensor directory, not estimated:

| | count | bytes |
|---|---|---|
| `ffn_{gate,up,down}_exps` (routed experts) | 120 | **9.12 GB PXQ2** / 13.15 GB PXQ3 |
| mxfp4 (attention, GDN, shared experts, embeddings) | 291 | 1.01 GB |
| q8_0 (attn k/v) | 20 | 0.02 GB |
| q6_K (output head) | 1 | 0.42 GB |
| f32 (norms, routers) | 301 | 0.09 GB |

Every panel-format tensor in the file is a routed expert, and they are uniformly one tier.
So the whole question is what happens to the 1.01 GB of MXFP4 backbone, and there are two
honest answers. Both are provided, because one is the deliverable and the other is its
control:

* **`--policy m2`** — the routed experts stay at their on-disk tier as a **byte move**, and the
  MXFP4 backbone (shared experts, `attn_output`, `attn_qkv`+`attn_gate`, `ssm_out`) is
  **re-encoded to PXQ4** with the native encoder. This is the existing P2A lever applied to a
  different tensor set, not new numerics.
* **`--policy m2f`** — identical, except the MXFP4 backbone is decoded to fp16. No encoder
  needed and no new numerics anywhere, which is exactly what makes it the control arm for the
  same-top-token gate: it separates "does the PXQ2/PXQ3 expert path decode correctly" from
  "does re-encoding the backbone cost quality".

#### The 16 GiB arithmetic, so nobody expects PXQ3 on one card

Resident weight bytes of the converted checkpoint, including the 0.89 GB vision tower:

All four rows are the converter's own plan totals for the real files, not estimates:

| source tier | policy | routed experts | rest (incl. the 0.83 GiB vision tower) | total |
|---|---|---|---|---|
| PXQ2 | m2 (PXQ4 backbone) | 8.50 GiB | 3.73 GiB | **12.23 GiB** |
| PXQ2 | m2f (fp16 backbone) | 8.50 GiB | 5.39 GiB | 13.89 GiB |
| PXQ3 | m2 | 12.25 GiB | 3.73 GiB | **15.98 GiB** |
| PXQ3 | m2f | 12.25 GiB | 5.39 GiB | 17.64 GiB |

A P100 and a V100-PCIE are 16 GiB, of which roughly 0.8 GiB goes to the CUDA context and the
allocator before a weight is loaded. So:

* **PXQ2 runs at TP=1 on one card**, with about 3 GiB left for KV and activations. Use
  `m2`; `m2f` leaves under 1.5 GiB and is a gate arm, not a serving configuration.
* **PXQ3 does not fit one card, in either policy, and no amount of tuning changes that.**
  It runs at **TP=2**. This is arithmetic, not a limitation of the port.

### A worked example of the tier limit: Qwen3.8 Flash-Next

Flash-Next is the model most often asked about here, so the answer is written down
rather than rediscovered. Written up the same day the PXQ2/PXQ3 tiers above landed, and quoted verbatim:

> Qwen3.8 Flash-Next is not served by the vLLM line in this release, and the reason is
> the checkpoint, not the engine. The engine side is ready: the fork registers
> `Qwen4ExpForCausalLM`/`ForConditionalGeneration`/`MTP` and routes their DeltaNet layers
> through the same GDN module our Pascal port covers, and its exact fused top-k router is
> tuned for precisely this model's shape (512 experts, top-10). Three checkpoint facts
> block it. First, no vLLM-form (safetensors) Flash-Next exists on the box — every
> artifact is GGUF. Second, in those GGUF files every routed expert is PXQ2 or PXQ3,
> tiers the vLLM sidecar does not implement; only the shared experts and attention are
> PXQ4, and the converter deliberately refuses mixed-tier modules because a mixed module
> loads cleanly and generates subtly wrong text. Third, the smallest artifact is 91.9 GiB
> against 64 GB of P100 VRAM, and a uniform-PXQ4 conversion would be larger, not smaller;
> llama.cpp serves it only because it pages weights, which vLLM does not do. Flash-Next
> therefore stays on the llama.cpp seat, which is where it is fastest anyway.

**Reconciling with the tier landing above, same day.** The mail's second blocker —
"tiers the vLLM sidecar does not implement" — is exactly what
[the tier support above](#2-quant-tier-support--the-matrix) retires: the sidecar now
serves PXQ2 and PXQ3, in different `FusedMoE` parameters, which is precisely the
gate/up-vs-down split Flash-Next's GGUF files use. Blockers one (no vLLM-form checkpoint)
and three (91.9 GiB resident against 64 GB of P100 VRAM; `--cpu-offload-gb` would stream
weights over the bus every layer and lose to the llama.cpp seat it is meant to beat) still
stand on their own. Flash-Next therefore stays on `pxa`, which is also where it is
fastest — now for two reasons instead of three. The Fusion 35B family, which has the same
tier structure at a size that fits, is the model this backend serves instead.

`tools/pxa-launch.py` reads the tier from the **per-tensor ggml type histogram**, not
from a metadata key, and refuses a file this backend cannot serve before emitting a
command.

### Mixture-of-experts on this backend

Routed experts are served by the plugin's own `PXQ4MoEMethod`, not by vLLM's fused-MoE
kernels: the two stacked expert parameters are kept quantized in VRAM — at PXQ2, PXQ3 or PXQ4,
independently for `w13` and `w2` — and decoded per use.
That is the whole memory argument — a 35B MoE whose experts were dequantized to fp16
would not fit on the cards it is meant to run on.

Two paths exist inside it. A device-indexed path reads the routed expert id out of
device memory, so it performs no host synchronization and is legal inside a captured
CUDA graph; it is used for decode-shaped batches. A per-expert loop, which does
synchronize, handles prefill-shaped batches, where re-reading each expert's weights once
per row would lose to a dequantize-and-GEMM. The boundary between them is the capture
ladder (see [The capture ladder and speculative decoding](#the-capture-ladder-and-speculative-decoding)).

On Pascal, read [Custom all-reduce](#custom-all-reduce) before putting a MoE seat into
service. Correctness there is a gate you run, not a property you assume.

### Why PXQ4 here, honestly stated

PXQ4 is **not smaller** than AWQ. Measured like-for-like on the language-model body:
AWQ g128 asym = 4.156 bpw, PXQ4 = 4.254 bpw — PXQ4 is ~2.3% *larger* per tensor. Since
decode is bytes-read-per-GPU-per-token bound, size is not the argument.

The argument is **quality per bit**: an fp16 row anchor, a per-16-element sub-scale and
a non-uniform 16-entry codebook fit against an importance matrix is a better-conditioned
4 bits than uniform group quantization at essentially the same footprint. A head-to-head
throughput comparison against AWQ has **not** been run.

---

## 3. Choosing an engine

**`tools/pxa-launch.py` automates this decision.** It reads the model file, probes the
cards, applies a measured decision table and then prints the engine, the evidence and
the exact command before running anything:

```bash
tools/pxa-launch.py --model /path/to/model --np 8 --explain   # decide and print, run nothing
tools/pxa-launch.py --model /path/to/model --np 8             # decide and exec
```

It never picks silently, it refuses rather than dropping a parameter that does not
translate between engines, and it labels any branch it is extrapolating as
`[INFERRED]` or `UNMEASURED` instead of guessing quietly. Full behaviour:
[`docs/LAUNCHER.md`](LAUNCHER.md).

The decision is not just about the card. These are the measurements behind it, all on
one 2x Tesla P100-PCIE-16GB pair (sm_60), every boot correctness-gated before its
number was kept:

**Dense 27B PXQ4 — vLLM wins everything measured**

| metric | vLLM | llama.cpp | ratio |
|---|---|---|---|
| single-stream decode | 24.01 | 13.7 | 1.75x |
| aggregate decode @8 | ~70 | 12.4 | 5.6x |
| prefill | ~225 | 156.5 | 1.44x |

*Caveat carried with these: the llama.cpp side is a single boot, below this bench's own
two-boot bar, and the graphs-on dense arm was never launched. The direction is not in
doubt; the exact ratios are single-boot.*

**MoE 35B PXQ4 — the engines swap places at a sharp crossover**

| concurrency | llama.cpp | vLLM | winner |
|---|---|---|---|
| np=1 | 95.6 | 30.4 | llama.cpp 3.14x |
| np=4 | 75.93 | 64.82 | llama.cpp +17.1% |
| np=5 | 79.49 | 64.32 | llama.cpp +23.6% ← llama.cpp peaks |
| np=6 | 69.58 | 75.60 | **vLLM +8.7%** ← crossover |
| np=7 | 67.74 | 87.03 | vLLM +28.5% |
| np=8 | 62.42 | 95.81 | vLLM +53.5% |

Neither curve is monotonic and the flip is abrupt — llama.cpp *peaks* at np=5, above its
own np=4 value, then drops 12.5% in one step while vLLM climbs. The margin swings 32
points between np=5 and np=6. The launcher therefore stores the **table**, never a
fitted slope: a straight line from np=4 to np=8 puts the threshold too early and
misprices np=5 by ~14%.

Long-document prefill on the same MoE model favours llama.cpp by ~1.7–1.9x (1136 /
~1058 / ~1000 vs 567.6 / 595.8 / 594.4 tok/s), but that arm is **cross-harness with
unmatched prompt lengths** — directionally trusted, not controlled.

Rule of thumb, if you are choosing by hand: **dense model or high concurrency → vLLM;
MoE at low concurrency, or long-document prefill → llama.cpp.** Nothing above np=8 was
measured on either engine.

---

## 4. Hardware and images

| arch | cards | attention backend | status |
|---|---|---|---|
| sm_70 | Tesla V100 | `FLASH_ATTN_V100` | measured, serving |
| sm_60 | Tesla P100 | `PASCAL_SDPA` | measured, serving |
| sm_61 | GTX 1080 Ti, P40 | — | **not supported here** — use `pxa` |

**Pascal support is not free.** Stock vLLM compiles for compute capability 7.0 and up,
and the last PyTorch shipping sm_60 cubins is 2.7.1+cu126. Running on P100 therefore
needs its own image, its own torch, the opt-in `tools/vllm-pxq4/tools/patch_sm60_compile.py`,
and `TORCHDYNAMO_DISABLE=1`.

**Two thin images, not one fat one.** A single image spanning sm_60 and sm_70 was tried
and does not work, for a structural reason rather than a configuration one:
`VLLM_SKIP_C_STABLE=1` is required to build against torch 2.7.1, and it drops
`csrc/libtorch_stable/`, where an operator the V100 serving path calls unconditionally
lives. You cannot have sm_60 cubins and that operator in the same build. Each arch gets
an image pinned to the torch its cards need.

**Eligibility is a property of the image, not of the compute capability.** An image only
serves a card if it actually carries PXQ4 kernels for it; `pxa-launch.py` probes this
rather than assuming a capability floor.

### The engine base

The plugin is out-of-tree, but the **image** is not: it is built from our fork of
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a Volta-focused vLLM fork, and that
fork's version is what decides which attention, quantization and speculative-decode
paths exist at all.

| | |
|---|---|
| upstream base | 1Cat-vLLM **v1.5.0** |
| our delta | 22 commits: the Pascal (sm_60) port, the short-prefill guard, the build recipe, the branding, and the vendored PXQ4 sidecar |
| wheel | `pxa_vllm-1.5.1.dev22+g<sha>` |

Everything PXA-specific lives in commits on top of the tag, and the sidecar
(`pxa/pxq4/`) is a separate directory the upstream tree never touches, so a future
rebase is a rebase and not a merge.

**What moving to the v1.5.0 base brought in.** The upstream range is overwhelmingly
Volta kernel work, and almost all of it is reachable from our seats:

- the paged XQA decode kernel behind `FLASH_ATTN_V100` was rewritten, including one
  audited change that removes a redundant fp16 round-trip in the softmax-to-PV handoff
  on fp16 KV;
- the sm_70 Marlin GEMMs and the TurboMind AWQ kernel were rewritten — the path every
  non-PXQ4 tensor in a PXQ4 checkpoint takes (`VLLM_SM70_QUANT_BACKEND=marlin`);
- **two silent-wrong-answer fixes.** The classic AWQ CUDA kernels contain an assertion
  that compiles out in release builds, so on Volta they ran as empty kernels and
  returned NaN; that path now routes through a Triton dequant instead. And the
  Triton-MLA decode kernel asked for 100 KiB of shared memory on a card that has 96,
  so it failed to launch; its block size is now halved on compute 7.0;
- custom all-reduce gained a dtype fallback (a non-float tensor could previously be
  handed to a kernel that only understands float) and a CUDA-IPC safety fix for
  VMM-backed allocations;
- DFlash and DFlash2 speculative decoding arrived in full, wired through
  `FLASH_ATTN_V100` rather than bolted beside it.

**What it did not bring.** The new sm_70 NVFP4/MXFP4/FP8 MoE kernels are not reachable
from a PXQ4 checkpoint — routed experts go through this plugin's own MoE method, not
vLLM's fused-MoE kernels. Nothing in the range widens any capability gate to Pascal;
every guard added upstream is compute 7.0 exactly, or 7.0 and above.

### Images

| tag | arch | torch | base |
|---|---|---|---|
| `ghcr.io/poisonxa16/pxa-vllm:sm70` | sm_70 | 2.10.0+cu128 | v1.5.0 |
| `ghcr.io/poisonxa16/pxa-vllm:sm60` | sm_60 (+sm_70 cubins) | 2.7.1+cu126 | v1.5.0 |
| `ghcr.io/poisonxa16/pxa-vllm:sm70-v1cat-8f5d78e` | sm_70 | 2.10.0+cu128 | the previous base, kept for bisection |
| `ghcr.io/poisonxa16/pxa-vllm:sm60-v1cat-8f5d78e` | sm_60 | 2.7.1+cu126 | the previous base, kept for bisection |

The two current tags are built by `scripts/build-images.sh sm70` and `... sm60` from
the fork checkout. Each build ends with an in-image gate that reads the cubins out of
`torch`, `_C.abi3.so`, `_moe_C.abi3.so` and `flash_attn_v100` with `cuobjdump` and
fails the build if any of them lacks an architecture the build was asked for. A `.so`
filename proves nothing; that gate is why the arch claim in this table is a fact.

> **Resolved for `v2026.09.05`: the `sm70` prefill regression was a measurement mode.**
> The v1.5.0-based Volta image loses ~10% of prefill when it is run **eager** — and the
> seat does not run eager. Compiled boots were failing with `Constraints violated
> (inputs_embeds.size()[0], positions.size()[1])` because the benchmark harness left
> `PXQ4_TRACE_M=1` set: the sidecar's route trace hashes the batch dimension, which
> specialises a dynamic dim and makes Dynamo reject the graph. With the trace off, both
> the candidate and the previously-shipped image boot compiled. Compiled at
> `--block-size 256` the candidate is at parity — prefill 1,002.85 vs 1,010.0 @3k
> (-0.7%), 993.84 vs 987.3 @20k (+0.7%), decode 49.73 vs 49.71, agg@8 178.23 vs 178.33 —
> and byte-identical on the determinism gate (`sha c220beafa2d5`, 12/12 at `-np 1`, same
> sha pair at `-np 2`, logit spread 0.0 on both). `ghcr.io/poisonxa16/pxa-vllm:sm70` is
> therefore the v1.5.0 image (`sm70-v15c`), served with `--block-size 256`. The
> compile-safe guard shipped in the sidecar is hardening; it is not what fixed this.
> 

### Picking the kernel library

`pxa/pxq4/kernels/` holds one library per arch and revision. Each carries exactly one
CUDA device binary, for the arch in its filename, and **no PTX** — a library will not
run on an architecture it was not built for.

| library | arch | ops added | status |
|---|---|---|---|
| `libpxq_sm70_v13.so` | sm_70 | `pxq2_*` / `pxq3_*` (`dequant_out`, `mmv_out`, `linear_out`, `moe_mmv_out`), `pxq_selftest`, `pxq_set_book`, `pxq_set_sub`, `pxq_supported` | **required for any PXQ2/PXQ3 checkpoint** on sm_70. A superset of v12b: the PXQ4 kernels in it are the v12b sources, unmodified. Built from `kernels-src/build_pxq_v13.sh`. |
| `libpxq_sm60_v13.so` | sm_60 (+sm_70 cubin) | same | **required for any PXQ2/PXQ3 checkpoint** on sm_60 |
| `libpxq4_sm70_v12b.so` | sm_70 | `f16_mmv_out`, `mma_out` | **current** for sm_70 — the tensor-core decode path, armed at batch >= 5 by `PXQ4_MMV_MMA=1`. Built from `pxa/pxq4/kernels-src/build_v12b.sh`; see `pxa/pxq4/MANIFEST.md`. This is the library the 2xV100 recipe in `docs/COOKBOOK.md` names. |
| `libpxq4_sm70_v10.so` | sm_70 | `f16_mmv_out` | shipped for sm_70; superseded by v12b |
| `libpxq4_sm60_v10.so` | sm_60 | `f16_mmv_out` | **shipped** for sm_60 |
| `libpxq4_sm60_v11.so` | sm_60 | `gemm2d_out` | ~+34% prefill behind `PXQ4_GEMM2D`, **default off** — failed first-token quality at 87.5%. Do not enable without re-gating quality. |
| `libpxq4_sm60_v9.so` | sm_60 | `f16_mmv_out` | superseded by v10 |
| `libpxq4_sm70_v9.so` | sm_70 | `f16_mmv_out` | superseded by v10 |
| `libpxq4_sm60_v8.so` | sm_60 | `moe_mmv_out` | superseded |

> **The `libpxq_*_v13.so` libraries carry no "4" in the name**, because they are the first
> that are not PXQ4-only — and because `libpxq4_sm70_v13.so` already exists and is an
> unrelated PXQ4-only build. Do not rename either.
>
> Pointing a PXQ2/PXQ3 checkpoint at a v12b-or-earlier library is **not** a silent failure:
> the plugin asks the loaded library whether the tier's ops exist and refuses at layer
> construction with a message naming the library it needs.

> **`PXQ4_LIB` must always be set explicitly.** The fallback lookup names are fixed
> (`libpxq4_sm70.so`) regardless of the arch in the file, so with `PXQ4_LIB` unset the
> loader can reach an sm_70 / torch-2.10 library on a Pascal image and die part-way
> through model load with
> `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv` — not with a clear
> message. Both shipping launchers set it for you.

Identifying a library after the fact — the ops are registered through `TORCH_LIBRARY`
string schemas, so `nm` will not find them:

```bash
grep -aoE '^(moe_mmv_out|f16_mmv_out|gemm2d_out|pxq2_mmv_out|pxq3_mmv_out|pxq_selftest)$' \
     pxa/pxq4/kernels/libpxq4_sm60_v10.so | sort -u
cuobjdump --list-elf pxa/pxq4/kernels/libpxq4_sm60_v10.so   # expect one member: pxq4_kernel.sm_60.cubin
```

`pxq_selftest` present means the library serves PXQ2/PXQ3. It is also an entry point you can
run: before loading a model, `torch.ops.pxq4.pxq_selftest(254)` decodes a deterministic
synthetic panel set with a host oracle written from the format spec and requires the device
kernels to agree **bit-exactly** — dequant against the oracle, and the decode GEMV against a
host replay of the kernel's own canonical-chunk fold. Tier `252` additionally checks the
templated PXQ4 instantiation against the shipped PXQ4 kernel, which is the transcription gate
for the whole header family. `kernels-src/tests_pxq23/gpu_selftest.py` wraps all of that plus
a real-tensor check and an expert-indexing check; run it first in any window.

---

## 5. Converting a PXQ4 model for vLLM

vLLM cannot read a PXQ GGUF. Two independent blockers, neither patchable without
forking three packages: `gguf.GGMLQuantizationType(252)` raises inside
`GGUFReader._build_tensors`, killing the file open before a single tensor is yielded;
and vLLM's generic GGUF sharder slices rows assuming per-row-contiguous blocks, which
the 64-row panel interleave violates.

So conversion is **offline and explicit**. The converter is pure Python + numpy — no
torch, no CUDA, no vLLM, no GPU:

```bash
cd tools/vllm-pxq4/src
python -m gguf_to_vllm.convert \
  --gguf   /path/to/model-PXQ4.gguf \
  --ref-hf /path/to/reference-hf-checkpoint \
  --out    /path/to/model-PXQ4-vllm \
  --policy p1
```

| flag | meaning |
|---|---|
| `--gguf` | the PXQ4 GGUF to convert. Required. |
| `--ref-hf` | a reference HF checkpoint of the same model: source of `config.json`, tokenizer, the vision tower, and the key-set diff. Effectively mandatory for a servable output. |
| `--out` | output directory. Required unless `--dry-run`. |
| `--policy` | which modules are served as PXQ4 — see below. Default `p1`. |
| `--encoder` | path to `pxq4_encode.so`; required by the `p2*` policies, which re-encode tensors that are not PXQ4 on disk. |
| `--shard-size-gb` | safetensors shard size. Default 4.0. |
| `--dry-run` | plan the entire conversion from the GGUF header alone and run every structural check, without reading tensor data. |
| `--verify` / `--no-verify` | round-trip every native PXQ4 tensor and compare **bytes**. On by default. Leave it on. |
| `--emit-plan` | write the conversion plan as JSON. |

**Run `--dry-run` first.** It exercises everything except the byte-writing and will tell
you, in seconds and off the header alone, whether the artifact is convertible.

### Policies

| policy | serves as PXQ4 | needs `--encoder` |
|---|---|---|
| `p1` | what the dense artifact already carries as PXQ4 on disk | no |
| `p2a` | p1 + the GDN output projection (`ssm_out`, MXFP4 on disk → re-encoded) | yes |
| `p2c` | p2a + a uniformly PXQ4 fused QKV (re-encodes k/v) | yes |
| `m1` | the MoE policy: expert stacks, shared experts, `o_proj`, fused GDN in-projection | no |
| `m2` | the **tiered** MoE policy: routed experts stay at their on-disk tier (PXQ2 / PXQ3 / PXQ4) as a byte move; the MXFP4 backbone — shared experts, `o_proj`, fused GDN in-projection, `ssm_out` — is re-encoded to PXQ4 | yes |
| `m2f` | `m2` with the backbone left fp16. No encoder, no new numerics; the **control arm** for the same-top-token gate, and roughly 1.7 GiB larger | no |
| `p2b` | **blocked at the CLI** — it was p2a plus a PXQ4 LM head, and a 4-bit head is not servable by the engine side. Rather than silently emitting a checkpoint byte-identical to p2a under a name that promises more, the converter refuses it by name. |

Start with `p1` (or `m1` for MoE). It needs no encoder and no re-encoding. For a 35B MoE
whose routed experts are PXQ2 or PXQ3, `m2` is the serving policy and `m2f` is its control —
see [The tier that fits](#the-tier-that-fits-fusion2--fusion4-35b-on-one-card) for why the
1.7 GiB between them decides whether the model fits one card.

### What comes out

For every module the policy serves as PXQ4, **two** tensors and **no** `.weight`:

```
<module>.pxq4_slabs    uint8     [N/64, K/32, 1088]   C-contiguous
<module>.pxq4_anchor   float16   [N/64, 64]           C-contiguous
```

These are derived from the GGUF blob by a **pure split** — the header bytes and the slab
bytes of each panel, reinterpreted, with no value recomputed. That is why the emitted
checkpoint can be proven equal to the GGUF by a byte comparison rather than a numeric
tolerance, and why `--verify` can round-trip every tensor exactly.

Everything else is decoded to fp16 `<module>.weight`. `config.json` is copied from
`--ref-hf` with **only** its `quantization_config` rewritten, so every architectural
field stays byte-identical to what already runs.

There is exactly one place bytes move: **GDN head order.** ggml stores value-heads
repeat-major, HF stores them k-head-major, so every per-v-head axis is gathered into HF
order on the way out. It stays a byte move (a 128-row head block is exactly 2 panels, a
128-column block exactly 4 slabs, so no nibble, sub-scale or anchor value is touched)
and `--verify` undoes the gather before comparing. Both the reorder and its proof
against `--ref-hf` are **fatal if missing**, not warnings: an unpermuted GDN checkpoint
loads, shards, passes every byte gate, and generates fluent garbage.

### Gates before trusting a conversion

The GPU-free suites need no CUDA and no GPU:

```bash
cd tools/vllm-pxq4/src
bash build_hostsim.sh          # compiles the CPU kernel simulator, then runs the kernel suite
python3 test_pxq4_config.py    # quant config / plugin registration
python3 gguf_to_vllm_test.py   # converter, incl. a bit-exact gate against a C oracle
python3 test_pxq4_linear.py    # linear method (skips cleanly if vLLM is absent)
```

`build_hostsim.sh` compiles the **real** `pxq4_kernel.cuh`, unmodified, against a stub
`cuda_fp16.h` and emulates a CUDA launch — so the kernel suite exercises the shipping
kernel source rather than a reimplementation that could drift from it. **A C++ compiler
is required:** without `libpxq4_hostsim.so` the eight simulator-backed tests *fail*
rather than skip. Seeing `9/17` means a missing toolchain, not a kernel defect.

---

## 6. Starting a server

### The easy path — the shipping launchers

Both scripts boot a container, wait for `/health`, and then **read back** the settings
that fail silently if they do not take.

```bash
# Volta / Tesla V100 class
MODEL=/path/to/model-PXQ4-vllm scripts/pxa-serve-sm70.sh

# Pascal / Tesla P100 class
MODEL=/path/to/model-PXQ4-vllm scripts/pxa-serve-sm60.sh

# every parameter and its default
scripts/pxa-serve-sm70.sh --help
```

Only `MODEL` has no usable default. Everything else is an environment variable:
`CARDS` (comma-separated indices; TP size = how many you list), `PORT`, `BIND`
(defaults to `127.0.0.1`), `IMAGE`, `LIB`, `SITE`, `GMU`, `MML`, `MNS`, `LADDER`,
`SPLIT_MAX_BLOCKS`, `TOOL_PARSER`, `EXTRA_ARGS`, `BOOT_TIMEOUT`.

Two optional operator guards, both unset by default: `PXA_REQUIRE_HOST` refuses to run
unless `hostname` matches, and `PXA_RESERVED_CARDS` (sm_60 script) refuses if `CARDS`
intersects a list of indices you have reserved for other workloads.

### The underlying command

```
vllm serve <model> \
  --quantization pxq4 \
  --dtype float16 \
  --attention-backend FLASH_ATTN_V100          # PASCAL_SDPA on sm_60 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16]}'
```

with `PYTHONPATH` pointing at the `pxq4_vllm` plugin tree and `PXQ4_LIB` pointing at the
kernel library for the card.

> **Bash brace-expands that JSON.** `{"cudagraph_capture_sizes":[1,2,3,4]}` becomes
> `cudagraph_capture_sizes:[1` because of the commas inside the braces. Single-quote it
> for whichever shell finally parses it.

`vllm serve` is correct for a self-contained image. It is **wrong** for an image whose
python, torch and vLLM live on the host and are bind-mounted in — `vllm` is not on
`PATH` there, and the command must be
`<host python> -m vllm.entrypoints.openai.api_server --model ...` instead.
`pxa-launch.py` handles that distinction from a site-local JSON descriptor named by
`PXA_VLLM_HOST_ENV`; nothing site-specific is hardcoded.

### The exact container lines

These are the two lines the seats run, with nothing elided. Substitute the model path;
everything else is load-bearing and each value is explained in
[Tuning](#7-tuning-what-moves-the-number-and-by-how-much).

Volta, 2x Tesla V100, TP=2:

```bash
docker run -d --name pxa-pxq4-v100 --runtime=nvidia --network bridge \
  -p 127.0.0.1:8262:8262 --ipc=host --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=2,4 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH=/opt/pxa/pxq4/sidecar/site-union \
  -e PXQ4_LIB=/opt/pxa/pxq4/kernels/libpxq4_sm70_v12b.so \
  -e PXQ4_MMV_MMA=1 -e PXQ4_MMV_SPLIT_MAX_BLOCKS=300 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_BUFFSIZE=1048576 \
  -e VLLM_SM70_QUANT_BACKEND=marlin -e VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0 \
  -v /opt/pxa/pxq4:/opt/pxa/pxq4 -v /path/to/models:/models \
  ghcr.io/poisonxa16/pxa-vllm:sm70 \
  python -m vllm.entrypoints.openai.api_server \
    --model /models/<model>-PXQ4-vllm --quantization pxq4 \
    --attention-backend FLASH_ATTN_V100 --tensor-parallel-size 2 --dtype float16 \
    --enable-prefix-caching --trust-remote-code --block-size 256 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.88 --max-model-len 32768 \
    --max-num-seqs 16 --max-num-batched-tokens 4096 \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16]}' \
    --host 0.0.0.0 --port 8262
```

`--block-size 256` is not decoration: it is the block size the parity above was measured at,
and `tools/pxa-launch.py` now emits it by default for a GDN hybrid on `sm_70` (override with
`--vllm-block-size N`, suppress with `0`). It is deliberately **not** applied on `sm_60`, where
it has never been measured.

`NCCL_P2P_LEVEL=SYS` and the 1 MiB `NCCL_BUFFSIZE` are not cargo cult: on an all-PHB
box with every card on four PCIe lanes, prefill was dominated by all-reduce until they
were set, and `--gpu-memory-utilization 0.88` rather than a higher value is what leaves
room for the larger NCCL buffers.

Pascal, 2x Tesla P100, TP=2:

```bash
docker run -d --name pxa-pxq4-p100 --runtime=nvidia --network bridge \
  -p 127.0.0.1:8199:8199 --ipc=host --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=1,5 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCHDYNAMO_DISABLE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e PYTHONPATH=/opt/pxa/pxq4/sidecar/site-sm60 \
  -e PXQ4_LIB=/opt/pxa/pxq4/kernels/libpxq4_sm60_v10.so \
  -e PXQ4_MMV_SLICE_MAX=8 -e PXQ4_MMV_SPLIT_MAX_BLOCKS=300 \
  -e HOME=/tmp -e TMPDIR=/tmp \
  -v /opt/pxa/pxq4:/opt/pxa/pxq4 -v /path/to/models:/models \
  ghcr.io/poisonxa16/pxa-vllm:sm60 \
  python -m vllm.entrypoints.openai.api_server \
    --model /models/<model>-PXQ4-vllm --quantization pxq4 \
    --attention-backend PASCAL_SDPA --tensor-parallel-size 2 --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
    --host 0.0.0.0 --port 8199
```

**On a 16 GiB card carrying a large resident model, two of those values change**, and both
are memory-budget facts rather than tuning preferences (measured 2026-09-05, Fusion2-35B PXQ2,
12.23 GiB resident, one P100, TP=1):

```
    --gpu-memory-utilization 0.88 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
```

`0.88` because 0.96 leaves the allocator ~20 MiB and the PASCAL_SDPA prefill OOMs at 6.5k
tokens *after* the server is healthy; `--limit-mm-per-prompt` because a vision-capable
architecture otherwise reserves an encoder cache and profiles the vision tower, and the engine
refuses to start with `0.0 GiB` of KV cache. `tools/pxa-launch.py` applies both automatically
when the smallest selected card is ≤ 17 GiB; `--vllm-mm` opts image input back in.

`TORCHDYNAMO_DISABLE=1` decides whether the server starts at all on Pascal: without it,
the profiling run compiles the language model through Inductor and dies on a
capability-6.0 card. `FULL_DECODE_ONLY` is a correctness requirement on this arch rather
than a preference. Add `--disable-custom-all-reduce` for a MoE model until you have
gated it — see below.

### Speculative decoding

The v1.5.0 base carries DFlash and DFlash2 block-diffusion drafters, integrated through
`FLASH_ATTN_V100` rather than beside it, so the draft and verify passes use the same
attention backend as the target. A DFlash drafter checkpoint is a small model whose
config declares `dflash_config` and the target layers it reads hidden states from; the
fork's model registry maps `DFlashDraftModel` and `DFlash2DraftModel` directly.

```
  --speculative-config '{"method":"dflash",
                         "model":"/models/<drafter>",
                         "num_speculative_tokens":7,
                         "draft_sample_method":"probabilistic"}'
```

Two things decide whether it pays, and neither is the engine:

- **Lineage.** Acceptance is a property of how well the drafter's training distribution
  matches the target. A drafter trained on the stock model and pointed at a fine-tune of
  it collapses — measured on the llama.cpp side at roughly 7% acceptance against 38-45%
  for the matched pair, which turns speculation into a net loss. Match the drafter to
  the target's lineage, or do not use one.
- **The capture ladder**, above. Speculation multiplies the tokens per step by `k+1`.

Verification is exact rejection sampling, so a correct DFlash arm changes **no** greedy
token. The gate is therefore not "close enough": capture the same prompts with the
drafter on and off and require every completion to be byte-identical. A single
divergence is a verifier bug, not a quality trade.

> **Known limit, `v2026.09.05`: a drafter plus a 27B target leaves little context on a
> 16 GB card.** At `--max-model-len 4096` the drafter and a dense 27B target together
> leave roughly 5.8k tokens of usable context on a single V100. This release's DFlash
> window on this backend is therefore **acceptance- and byte-identity-gated only** —
> confirming the speculative path is correct and changes no output — with **no
> throughput cell** taken at a context length anyone would actually serve at. This is a
> memory-budget limit of running drafter + target together under vLLM's resident-weight
> model, not a defect in the DFlash integration itself; the llama-engine DFlash port
> (`RELEASE-NOTES-2026-09-07.md`, "DFlash") does not share this constraint.
> 

### Custom all-reduce

Custom all-reduce (CAR) is vLLM's own two-rank collective, used instead of NCCL for
small payloads. It is the single largest multi-GPU lever on this stack and it is **on
by default**.

- **Dense PXQ4 on Volta and on Pascal: leave it on.** It was byte-gated — twenty greedy
  completions identical to an NCCL reference boot — on both arches, and turning it off
  costs roughly half the single-stream decode rate on Pascal.
- **MoE on Pascal: now correct on the v1.5.0 base, and it is the faster arm.** Through
  2026-08, a CAR-enabled MoE seat on P100 had a recorded history of producing
  fluent-looking token soup from the first character, deterministically, while the same
  seat with `--disable-custom-all-reduce` was correct; the documented mitigation was to
  run MoE on Pascal with CAR off. As of the v1.5.0 rebase (2026-09-05) that arm is
  **byte-gated clean**: 20/20 completions identical between CAR-on and CAR-off, same
  first token in all 20, on a 35B MoE (`coder35-moe-pxq4-m1`, TP=2, GPUs 1/5). CAR on is
  also **faster** — +28.8% single-stream decode (29.80 vs 23.13 tok/s) and +6.6%
  aggregate @8 (97.81 vs 91.73 tok/s) over the CAR-off mitigation — so the two mandatory
  workarounds this project carried for MoE on Pascal (CAR off, PP=2) are retired for any
  model gated the same way. The collective itself was exonerated a release ago (tens of
  thousands of live reduces cross-checked bit-exact against NCCL); the leading candidate
  for what actually fixed it is the v1.5.0 base's dtype fallback in the all-reduce path,
  though the exact commit has not been bisected out of the 521-commit range. **Still gate
  your own model before trusting CAR on** — this result is for the model and library
  version stated above, not a blanket clearance for every MoE checkpoint.

The gate is cheap and it is the only thing that settles it: capture twenty greedy
completions with CAR off, capture the same twenty with CAR on, and require every one to
be byte-identical. Anything less than 20/20 means CAR off.

### The capture ladder and speculative decoding

`cudagraph_capture_sizes` is a list of **token** counts, not request counts. In ordinary
decode one running request contributes one token, so a ladder of `[1,2,4,8,16]` covers
sixteen concurrent requests. Under speculative decoding each request contributes
`k+1` tokens per step — the drafted tokens plus the bonus row — so the same sixteen
requests need `16 x (k+1)` in the ladder. Get this wrong and nothing breaks loudly: the
step simply falls out of the captured graphs and runs eager, and you lose the whole
CUDA-graph win while every gate still passes.

The plugin's own MoE path has the same constraint for the same reason. Its arenas are
sized once, eagerly, before capture and refuse to grow inside it, so the pre-capture
sweep is driven off the ladder the engine was actually given rather than an assumed
width. If you widen the ladder, you do not have to do anything; if you pin it with
`PXQ4_MOE_INDEXED_MAX_S`, you own the consequences.

---

## 7. Tuning: what moves the number, and by how much

Everything in this section was measured with each arm correctness-gated *before* its
speed number was recorded. A fast wrong answer is not a result.

### The three correctness keys — not tuning knobs

| setting | why |
|---|---|
| `cudagraph_mode: FULL_DECODE_ONLY` | The vLLM default (`FULL_AND_PIECEWISE`) **also captures prefill graphs** at the ladder sizes. A raw `/v1/completions` prompt short enough to fit one then prefills through a captured graph whose input buffer holds stale data and returns fluent garbage from character zero. Chat traffic never shows it, because the chat template pads every prompt past the captured sizes — which is exactly why arithmetic gates stayed green while the bug was live. |
| `custom_ops: ["none"]` | Mandatory wherever `FULL_DECODE_ONLY` is emitted on sm_60. Without it, PP≥2 + FDO is a **hard boot failure** (`CUDA error: an illegal memory access was encountered`, in `profile_run`). On sm_70 its necessity is unmeasured; it is emitted anyway, which is the safe direction. |
| `TORCHDYNAMO_DISABLE=1` (sm_60 only) | Load-bearing. Without it `profile_run` compiles the **language** model through Inductor and dies with `GPUTooOldForTriton` on a capability-6.0 card. This single flag decides whether the server starts at all. |

Do not "fix" the Triton problem by disabling Triton globally on Pascal — that was tried
and reverted. The warmup path then calls `triton.next_power_of_2` on
`TritonPlaceholder`, which does not define it, and shimming that method only moves the
failure to a real Triton kernel launch on the same path.

**The one-token completion defect — same species, different arch, now fixed on Volta.**
The `FULL_DECODE_ONLY` note above describes a short raw prompt hitting a stale *prefill*
graph. A sibling defect hit the *decode* graph on the shipped Volta (sm_70) image: a raw
`/v1/completions` prompt of exactly one token (e.g. `"Hello"`) returned a degenerate
`"!!!!…"` completion, because a one-token prompt's shape happened to match the captured
decode graph's query length and replayed it over stale input state; 5- and 12-token
prompts were clean, and chat-templated and 70-prompt gates never saw it because they pad
every prompt past the envelope. Live since at least 2026-08. **Fixed on the v1.5.0-based
candidate image** (carrying this project's short-prefill guard): a one-token probe is
now coherent where the previously-shipped image on the same cards returned `!!!!`, and a
70-prompt same-top-token capture against the shipped image agrees on the first token
70/70 (bar was 98.6%). A raw one-token probe is now part of the gate for this reason.

### The capture ladder — the largest silent loss

**The ladder must be passed explicitly.** When `cudagraph_capture_sizes` is `None` the
sm_70 branch hard-codes `[1, 2]`, so every batch above 2 concurrent runs eager —
roughly **4x slower**. Both launchers read the installed value back out of the startup
log and warn if it collapsed to `[1,2]`; a quoting slip is otherwise invisible.

The ladder should be powers of two covering `--max-num-seqs`. `[1,2,4,8]` is the
measured ladder on sm_60; widening past 8 is inference, not measurement.

### sm_70 (2x Tesla V100-PCIE-16GB, TP=2, dense 27B PXQ4)

| arm | decode | ms/tok | agg@4 | agg@8 |
|---|---|---|---|---|
| baseline | 48.74 | 20.52 | 106.46 | 134.36 |
| `PXQ4_MMV_SPLIT_MAX_BLOCKS=300` | **51.46** | 19.43 | 107.26 | 132.13 |
| `SPLIT=600` | 51.31 | 19.49 | 106.60 | 134.47 |
| `SPLIT=150` | 49.51 | 20.20 | 105.07 | 134.66 |
| `SPLIT=300`, GMU 0.92 | 51.32 | 19.49 | 104.85 | 133.76 |
| `SPLIT=300`, MNS=16, ladder `[1..8,16]` | 50.98 | 19.62 | **107.02** | **135.78** |

Two shipped profiles: `agg` (default; MNS=16) takes both aggregate crowns while still
clearing 50 tok/s single-stream. `PXA_PROFILE=single` (MNS=8) trades 0.5 tok/s of
concurrency headroom for the best single-stream figure.

| knob | value | effect |
|---|---|---|
| `PXQ4_MMV_SPLIT_MAX_BLOCKS` | **300** | routes `gate_up` from mono to split. **+2.7 tok/s** (48.74 → 51.46). 150 is worse, 600 is a wash. |
| `--gpu-memory-utilization` | **0.85** | *not* 0.90+. 0.98 with a pinned 12 GiB KV is a four-card, 32-GiB-per-card setting and **aborts** on a 16 GiB card at TP=2 before the model finishes loading. |
| `--max-num-seqs` | 16 (`agg`) / 8 (`single`) | drives the ladder; see the table above. |
| `f16 mmv hook` | n/a | does **not** arm on sm_70 — it is Pascal-specific. Expected, and costs nothing here. |

### sm_60 (2x Tesla P100-PCIE-16GB, TP=2)

| metric | measured |
|---|---|
| single-stream decode | 24.0 – 26.4 tok/s |
| aggregate @8 | 70.0 – 72.1 tok/s |
| aggregate @4 | 45.0 – 51.5 tok/s |
| long-doc prefill | ~218 tok/s |

| knob | value | effect |
|---|---|---|
| custom all-reduce | **left ON** | the dominant lever: **13.3 → 24.0 tok/s, ~1.8x**. Do *not* pass `--disable-custom-all-reduce` here. Byte-gated 20/20 against an NCCL reference on this exact library + tree + config. It remains unsafe for MoE models on Pascal — a different model class. |
| plugin tree with the fp16 mmv hook | `site-sm60` | **~+12%**, and its absence is *silent*. Count it, do not assume it — the log string is `fp16 mmv fast path armed`; grepping `f16 mmv` matches nothing and makes a working hook look absent. |
| `PXQ4_MMV_SPLIT_MAX_BLOCKS` | **300** | not 150. 150 is **bimodal**: four boots gave 24.58, 24.61, 19.75, 9.27. 300 gave 24.01 / 23.97 / 23.97 across three. A single good 150 sample looks like a win and is not. The mechanism is not understood; 300 ships on stability. |
| `--gpu-memory-utilization` | **0.90**, but **0.88 on a 16 GiB card with a large resident model** | 0.94 reintroduces the raw-prompt `!!!!` failure. Separately, and it bites at the other end: serving a 12.23 GiB model on one 16 GiB P100, **0.96 left the allocator 20 MiB and the PASCAL_SDPA prefill OOM'd at 6.5k tokens** — *after* the server came up healthy and answered short requests, which is the worst shape of failure because it looks like a working seat until someone sends a long prompt. 0.88 gave 1.87 GiB of KV cache (69,259 tokens) and a stable server on the same card and model. `pxa-launch.py` now caps the sm_60 default at 0.88 whenever the smallest selected card is ≤ 17 GiB. |
| `--limit-mm-per-prompt` | **`{"image":0,"video":0}`** on a multimodal checkpoint served text-only | A vision-capable architecture makes vLLM reserve a 16384-token encoder cache and profile the vision tower at maximum feature size. On a 16 GiB card holding 12.23 GiB of weights that is the difference between a working server and `0.22 GiB KV cache is needed, which is larger than the available KV cache memory (0.0 GiB)` — an engine that refuses to start, with a message pointing at `gpu_memory_utilization`, which is *not* where the memory went. The PXQ conversions of text-only GGUFs borrow their vision tower from a reference checkpoint and are gated text-only, so off is the honest default; `pxa-launch.py --vllm-mm` restores it. |
| `--max-num-seqs` / ladder | **8** / `[1,2,4,8]` | both required for correctness. With defaults at MNS=4 / ladder `[1,2,4]`, a raw 1-token prompt returns `!!!!` while "Paris" and `17*23` both still pass. MNS=16 is unverified on this arch. |
| `PXQ4_MMV_SLICE_MAX` | 8 | **a dead knob** at serving level (14.43 at 8 vs 14.24 at 16). Pinned only to stop it being re-litigated. |
| `PXQ4_GEMM2D=1` (v11 library) | **off** | reaches **300.1 tok/s prefill (+37%)** and **fails raw-prompt correctness**. Decode and the byte-gate are unaffected. A prefill win that changes what the model says is not a win. Recorded so it is not rediscovered as a fresh idea. |

### Parallelism

- **Dense → tensor parallel.** `--tensor-parallel-size <cards>`.
- **MoE → pipeline parallel.** `--pipeline-parallel-size <cards> --tensor-parallel-size 1`.
  PP=2 + `FULL_DECODE_ONLY` is the arm that produced every MoE number in §3.
- vLLM rejects a non-power-of-two parallel degree at startup.
- **Without P2P between cards, turn custom all-reduce off on MoE** — it costs ~18%
  versus NCCL there. On the sm_60 dense arm above it is the opposite: leaving it on is
  worth 1.8x. `pxa-launch.py` reads the topology rather than hardcoding either.

---

## 8. Tool calling

**A server will not answer a `tools=` request without two flags.** Omit them and every
such request returns **400**.

```
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

With the shipping launcher:

```bash
MODEL=/path/to/model-PXQ4-vllm TOOL_PARSER=qwen3_coder scripts/pxa-serve-sm70.sh
```

**The parser choice is not cosmetic.** For the Qwen-family coder templates it is
`qwen3_coder`, **not** `hermes`: hermes expects JSON inside `<tool_call>`, while this
template emits XML — `<function=NAME><parameter=K>V`. Choosing hermes does not error;
it returns **empty tool calls**, which is a far more expensive failure to notice. Match
the parser to the chat template your model actually ships.

> `tools/pxa-launch.py` does **not** emit these flags and has no passthrough for extra
> vLLM arguments. If you need tool calling, either run `--explain`, take the printed
> command and append the two flags, or use `scripts/pxa-serve-sm70.sh` with
> `TOOL_PARSER=`.

---

## 9. After the server is up

A launcher that execs the server cannot observe anything afterwards, so neither
launcher makes a health claim. **Passing a flag is not evidence the flag took effect** —
an image can parse `FULL_DECODE_ONLY`, boot healthy, and silently override it back to
`FULL_AND_PIECEWISE` from its own compile policy. Check:

1. **Capture mode.** Grep the server log for the *installed* cudagraph mode. If it is
   not `FULL_DECODE_ONLY`, the server is not healthy — shut it down. A derived image is
   the fix, not a flag.
2. **Capture ladder.** `grep -ao 'cudagraph_capture_sizes[^]]*]'` on the log. If it
   collapsed to `[1, 2]`, the explicit list did not take and you are eager above 2
   concurrent.
3. **The fp16 mmv hook, on sm_60.** `grep -c 'fp16 mmv fast path armed'`. Expect it on
   most layers; a low count costs ~12% and says nothing in the logs by itself.
4. **Per-device resident bytes.** Confirm the split actually happened.
5. **Short-prompt correctness, on a RAW, non-chat-templated prompt** — a 1-token and a
   5-token completion, before any number is trusted. Chat-templated traffic pads every
   prompt past the captured graph sizes, which is precisely how prefill-graph corruption
   survives arithmetic gating.

Both launchers do (2) and (3) for you and warn on failure.

### Failure modes that are silent by default

| symptom | cause |
|---|---|
| `undefined symbol: ..._ZNK3c1010TensorImpl15incref_pyobjectEv` mid-load | `PXQ4_LIB` unset or pointing at the wrong torch ABI |
| `GPUTooOldForTriton` at `profile_run` on P100 | `TORCHDYNAMO_DISABLE=1` missing |
| `CUDA error: an illegal memory access` in `profile_run`, zero tokens | `custom_ops: ["none"]` missing with PP≥2 + FDO |
| fluent garbage from character zero on a raw short prompt | prefill graphs captured — `cudagraph_mode` is not `FULL_DECODE_ONLY` |
| `!!!!` on a raw 1-token prompt, sane answers otherwise (sm_60) | MNS/ladder too small, or GMU ≥ 0.94 |
| ~4x slowdown above 2 concurrent | capture ladder collapsed to `[1,2]` |
| `tools=` requests return 400 | `--enable-auto-tool-choice` / `--tool-call-parser` missing |
| empty tool calls, no error | wrong `--tool-call-parser` for the template |
| ~12% less decode on sm_60, no error | fp16 mmv hook did not arm; check `PYTHONPATH` / `PXQ4_LIB` |

---

## 10. Correctness gates this backend had to pass

1. bit-exact dequant parity against a CPU reference;
2. single-linear-layer GEMM parity;
3. **sharded parity** — each per-rank slice must dequantize to exactly the unsharded
   result;
4. logprob parity against `pxa` on the same prompts at temperature 0;
5. an end-to-end throughput measurement on the target cards.

All five pass on both architectures. `-use_fast_math` is forbidden in any kernel build:
it would change the fp32 fold order, and bit-identity with the llama.cpp kernels is the
whole correctness argument.

---

## Further reading

| document | what it covers |
|---|---|
| [`tools/vllm-pxq4/README.md`](../tools/vllm-pxq4/README.md) | the package itself, attribution, build and test |
| [`docs/PXA-SM70-SERVING.md`](PXA-SM70-SERVING.md) | the full V100 sweep and every trap |
| [`docs/PXA-SM60-SERVING.md`](PXA-SM60-SERVING.md) | the full P100 sweep and every trap |
| [`docs/LAUNCHER.md`](LAUNCHER.md) | `pxa-launch.py`: the decision table, refusals and evidence |
| [`docs/lab/LEVERS.md`](lab/LEVERS.md) | the `PXA_*` levers on the llama.cpp engine |
| [`pxa/pxq4/README.md`](../pxa/pxq4/README.md) | kernel libraries, the `PXQ4_LIB` rule, rebuilding |
| `tools/vllm-pxq4/docs/01-pxq4-format-spec.md` | the PXQ4 on-disk format |
| `tools/vllm-pxq4/docs/09-chosen-design.md` | the design that was built, and its gates |

---

PXQ, PXQ4 and `pxa` are developed by **PXA Network** — <https://pxanetwork.com>.
The vLLM serving engine is Apache 2.0; this backend matches that licence. See
`tools/vllm-pxq4/LICENSE-NOTICE.md` for the full attribution chain.
