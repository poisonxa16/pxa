# PXQ llama.cpp — v2026.08.28-rc3

Release candidate. Supersedes **v2026.08.25-rc2**; **47 commits** — 19 on the
distribution side and 28 carrying the Flash-Next architecture work, which until now
lived on a separate branch. Branch `rc/unified`.

**This is the first release cut from a single tree.** The engine and the Flash-Next
work had unrelated git histories, so the architecture support, both correctness fixes
and the performance levers were not in any release build. They are now: the unified
tree builds clean and was gated on the real model before this tag was cut — 49/49
layers offloaded, KV exactly 3840.00 MiB, PXQ2/PXQ3/PXQ6 fused kernels engaged, nine
capitals correct, and decode at 26.86 tok/s, identical to the pre-merge measurement.

---

## Read this first

**This release adds a model architecture that upstream cannot load at all.**
Qwen3.8-Flash-Next declares `general.architecture = qwen4exp`. Current mainline
llama.cpp (`ghcr.io/ggml-org/llama.cpp:server-cuda`, ggml 0.19.0) answers:

```
llama_model_load: error loading model: unknown model architecture: 'qwen4exp'
```

Mainline registers `qwen3next` and has none of the `hc_*` hyper-connection or
`ple_*` tensors this architecture is built from. That is a statement about
coverage, not a benchmark: there is no engine-vs-engine number for this model
because there is no other engine to compare against.

**Two latent correctness bugs are fixed, and one of them affected every model.**
`inp_ple_rows` / `inp_ple_conv_hist` were read on every architecture but assigned
only by the qwen4exp graph builder, with no initialiser — so on any *other* model
the guard dereferenced a wild pointer. Measured 1 abort in 3 runs before, 0 in 15
after. The older binary was not correct, only lucky.

**The launcher will now pick a working config for a heterogeneous card pool.**
`--model` + `--cards` is enough; it derives the tensor split, the micro-batch and
the CPU offload itself, prints the evidence, and refuses rather than guessing.

Nothing here changes numerics on an existing PXQ4 instance.

---

## Both engines, one tree

This release carries the llama.cpp engine **and** the vLLM serving path, which
until now lived in a separate repository:

| piece | what it is |
|---|---|
| `tools/vllm-pxq4/` | the PXQ4 quantization backend / vLLM plugin |
| `scripts/pxa-serve-sm70.sh` | V100 instance, measured: 51.46 tok/s single, 135.78 aggregate @8 |
| `scripts/pxa-serve-sm60.sh` | P100 instance, measured: 24.0-26.4 single, 70-72 aggregate @8 |
| `scripts/pxa-serve-flashnext.sh` | Flash-Next PXQU instance, 4x P100 + 1080 Ti, 160k context |
| `pxa/pxq4/` | 6 kernel libraries, 75 kernel sources, README + MANIFEST with md5s |

Before `pxa/pxq4` was vendored, none of it was in version control anywhere — the
kernel sources existed only in a scratch directory on a single machine.

Both instances were restarted from this tree's own copies and verified by generating,
including tool calling, which needs `--enable-auto-tool-choice` and
`--tool-call-parser qwen3_coder`; without those every `tools=` request returns 400.

---

## New architecture: qwen4exp (Qwen3.8-Flash-Next)

48 layers, 512 experts with 10 used, hybrid attention, and two structures that do
not appear in any other model we serve.

| piece | what it is | why it needed new code |
|---|---|---|
| Per-layer embeddings (PLE) | `per_layer_token_embd`, 51.2e9 elements, ~51 GiB | a pure `GET_ROWS` gather — one lookup per token per head, no GEMM. Belongs in host RAM; the n-gram hash that indexes it needs int64 multiply and xor, which ggml has not got, so it is computed host-side |
| Hyper-connections | `hc_*`, a wide 10240 residual with 4 streams | the mixer **replaces** all norms; the head mixer **is** the output norm |
| GDN linear attention | 36 of 48 layers | only 12 layers keep a KV cache (`full_attention_interval=4`) |

**Converter** (`convert_qwen4exp.py`): HF safetensors → GGUF, streaming, with a
213-case self-test. Four layout rules were each established by byte-comparison
against a reference GGUF, and each one silently corrupts the model if missed:

- `PERM48 = arange(48).reshape(16,3).T.ravel()` on `ssm_a`, `ssm_dt.bias`,
  `ssm_alpha`, `ssm_beta`, every `attn_gate`, the v-portion of `attn_qkv` and
  `ssm_conv1d`, and the **input** dim of `ssm_out`
- `ssm_a = -exp(A_log)` under that permutation
- every RMSNorm weight is stored as `(w-1)` — `linear_attn.norm` is the sole exception
- `ssm_conv1d` permutes **only** its v-portion

Scope gap, stated plainly: the vision tower (333 tensors) and MTP (31 tensors) are
**not** converted.

---

## PXQ_UNIVERSAL: a mixed per-tensor tier map

A PXQU artifact carries different PXQ tiers on different tensors. The Flash-Next
build is 96.77 GiB / 1224 tensors:

| type | tensors | bytes | share |
|---|---:|---:|---:|
| q8_0 | 123 | 51974 MiB | 52.5% |
| pxq3 | 106 | 34606 MiB | 34.9% |
| pxq2 | 38 | 8574 MiB | 8.7% |
| pxq4 | 276 | 1460 MiB | 1.5% |
| f16 | 290 | 1220 MiB | 1.2% |
| q6_K | 2 | 995 MiB | 1.0% |
| f32 | 389 | 250 MiB | 0.3% |

The q8_0 share is almost entirely the CPU-resident PLE. **Excluding it, the
GPU-resident bytes are 94.7% PXQ-family** — which is why the build needs
`--pxq-composition-override` to pass a floor that measures the wrong thing for an
architecture with a large host-side table.

Two guards were added to the quantizer because both failures were hit for real:

- panel codecs are refused on row-gather tensors (they gather nonsense)
- any tensor with `ne0 % 32 != 0` is ruled to f32, with a preflight that refuses to
  start rather than dying 59 tensors in

**Quality caveat, unresolved:** the build emits 154 `pxq3 reconstruction ceiling
0.8957 x row absmax` warnings — row-peak weights are mildly clipped, so expect
elevated top-1% error. No imatrix was used, by instruction. Both correctness
batteries passed, but this is the first thing to suspect if quality is questioned.

---

## Correctness fixes

| fix | symptom | evidence |
|---|---|---|
| `inp_ple_rows` / `inp_ple_conv_hist` initialised | wild-pointer dereference on **any non-qwen4exp model**; a 0.6B dense model aborted on a PLE assert after an unrelated relink | 1 abort / 3 runs before, 0 / 15 after, correct output in all 15 |
| ROUTER_FUSE mode-3 base-pointer guard | the guard checked `ne00 % 4` and row stride `nb[1] % 16` but never that `src0->data` / `src1->data` were 16-byte aligned, while the kernel casts both to `float4*` — a misaligned-address **abort**, not a wrong number | default-OFF path; declines instead of faulting |

The first is the more important lesson: the failure moved with an unrelated
relink because the value tracked heap layout. **When a failure moves with a
relink, suspect uninitialised memory before suspecting the change.**

---

## Performance — measured

Instrument: `llama-cli`, `--temp 0 --ignore-eos`, 2 reps, 2931-token prompt for
prefill. Every earlier prefill figure on this model came off a 15-token prompt and
was fixed overhead, not throughput.

**Flash-Next PXQ4, six cards** (4x P100 + 2x V100, PLE on CPU):

| ub | prefill | decode |
|---|---|---|
| 512 (adaptive default) | 337–366 tok/s | 30.6–30.9 tok/s |
| **1024** | **430–439 tok/s** | 30.6–31.1 tok/s |
| 2048 | OOM — needs its own `-ts` | — |

`-ub 1024` is **+20% prefill for nothing**: the decode ranges overlap, so it is not
a trade. Decode reached 32.4 tok/s on a short prompt via two committed levers — a
byte-proportional six-card `-ts` rebalance (4 layers off the P100s onto the V100s)
and a dedicated small-K/large-R f16 GEMV for the hyper-connection up-projection.

**Flash-Next PXQU, four P100s:** 27.79 tok/s decode; prefill 411–414 (ub1024),
424–430 (ub2048).
**Five cards incl. the 1080 Ti, 160k context:** 26.86 tok/s, nine capitals correct,
clean EOS.

**PXQU is slower than PXQ4 and must not be sold as a speed win.** It is a footprint
win: 46.7 GiB of GPU-resident weights against PXQ4's 65.0, so four or five cards
instead of six, with 160k context.

**vLLM instances** (unchanged this cycle, restored in rc2): sm70 51.46 single /
135.78 aggregate @8; sm60 24.0–26.4 single / 70–72 aggregate @8.

---

## Launcher

`pxa-launch.py` now handles this architecture. Four defects — three of which
produced a confident wrong answer rather than saying UNMEASURED:

1. **KV/token was 4x too high on hybrid attention.** The flat
   `n_layer x n_kv x (d_k+d_v)` form assumes every layer caches KV. qwen4exp caches
   on 12 of 48. It priced 160k at 15.0 GiB instead of 3.75. Now reads
   `attention.compress_ratios`, falling back to `full_attention_interval`.
   The corrected formula predicts 24576 B/token and the engine allocated **exactly**
   3516.00 / 3840.00 / 6144.00 MiB at c=150016 / 163840 / 262144. Three exact hits.
2. **`R-17B` refused a model that fits.** It compared *file* bytes against VRAM, but
   50.66 of the 96.77 GiB is host-resident PLE; the GPU-resident remainder is
   46.10 GiB and runs on five cards.
3. **`-ot` was never emitted**, so the PLE was offloaded with everything else and the
   load died on a 16089.57 MiB `cudaMalloc`.
4. **No `-ts`**, so a heterogeneous pool got an even split — putting ~9.2 GiB on a
   card with 7.8 GiB free.

`auto_tensor_split()` now derives the split from real free VRAM, charges the head
card its larger compute buffer, reserves decode headroom, and **declines loudly**
if any card has no room. Validated against a hand-tuned split that was already
known good: the launcher independently produced `111,267,95,267,260` against
`110,268,94,268,261` — within 1 part in 1000 per card — and the instance was then
restarted on the launcher's own numbers and generated correctly.

Also: the sm_60 image recipe still named `libpxq4_sm60_v8.so`; bumped to v10, which
is what both shipping launchers use. **Not v11** — v11 is 37% faster on prefill and
fails raw-prompt correctness.

---

## Operational notes that cost real time

**A config that loads is not a config that runs.** At c=262144 the Flash-Next instance
loaded cleanly, printed every buffer, then died on the first token with
`CUDA error: out of memory` in `llama_decode` — the split had left one card 51 MiB.
Decode allocates transient buffers the init-time report does not include. Leave
~1200 MiB free per card *after* load. The same instance at 163840 keeps 807–1803 MiB
free per card and runs.

**Run this model from NVMe, never from a parity array.** `per_layer_token_embd` is
CPU-resident and mmap'd, so every decode token faults PLE pages off whatever disk
holds the file. From the array that is ~17 MB/s, and the process wedges in
uninterruptible `D` state inside the md driver where `SIGKILL` cannot be delivered
until the read drains.

**`-ub` and `-ts` are coupled.** llama.cpp folds a per-device compute allowance into
the `-ts` walk, so changing `-ub` repacks the layers. A split tuned at one `-ub` can
overflow a card at another — that is exactly how ub2048 OOM'd a six-card instance whose
split was tuned at ub512.

**Tool calling needs two flags that no launcher passed.** Without
`--enable-auto-tool-choice` and `--tool-call-parser`, every `tools=` request returns
400. The parser is `qwen3_coder`, **not** `hermes`: hermes expects JSON inside
`<tool_call>`, while this template emits XML `<function=NAME><parameter=K>V`.
Choosing hermes would have returned *empty* tool calls rather than an error.

---

## Known gaps

- vision tower and MTP tensors are not converted for qwen4exp
- 154 pxq3 clipping warnings in the PXQU build; no imatrix, by instruction
- ub2048 on the six-card PXQ4 instance needs its own `-ts`; not derived
- the hybrid-attention KV correction is MEASURED for qwen4exp only. `qwen35moe`
  takes the same correction but remains `[INFERRED]` — nobody booted it
- QSA top-k sparsity is still not wired: full-attention layers attend densely.
  Not a bug at short context; real at length
