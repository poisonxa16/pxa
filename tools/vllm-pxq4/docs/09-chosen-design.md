# 09 — PXQ4-in-vLLM: THE implementation plan

Status: definitive. Supersedes `08-design-minimal-risk.md`, `08-design-max-performance.md`,
`08-design-least-code.md`. Implementation agents build from this file.
No GPU was run. No container was restarted. Nothing under `/path/to/engine-repo` was modified.

---

## 0. LEAD WITH THE BLOCKER-SHAPED FACT

**PXQ4 at 4.25 bpw cannot deliver "110–120 tok/s". The realistic ceiling of this project on
this box is ≈ +9% over the incumbent's measured 92.8 tok/s peak, and reaching even that
requires re-encoding three tensor classes that are *not* PXQ4 in the artifact today.
The v1 that ships without re-encoding is a ~23% REGRESSION.**

I re-derived this independently from both checkpoints this session, and it confirms
`08-design-least-code`'s correction (max-performance got the direction right but the wrong
magnitude; minimal-risk's 5.225 GiB/GPU over-counts).

Measured inputs (FACT, both read this session):

| | source | bytes |
|---|---|---:|
| AWQ body (layers 0–63, excl. lm_head/embed/visual/mtp) | safetensors headers, `awq.py` this session | 12,691,100,928 (11.820 GiB) |
| AWQ `lm_head.weight` | **BF16, unquantized** — `ignore` list has `lm_head` | 2,542,796,800 |
| AWQ `embed_tokens` | BF16 | 2,542,796,800 |
| AWQ visual (333 tensors) | BF16, all in `ignore` | 921,460,192 |
| AWQ mtp (15 tensors) | BF16, unquantized | 849,398,784 |
| PXQ4 file, id-252 tensors (325) | `06-file-composition.md` | 12,231,950,336 (11.392 GiB) |

Per-tensor proof of the bpw gap: their `mlp.gate_proj` (17408×5120) costs
`weight_packed 44,564,480 + weight_scale 1,392,640 + weight_zero_point 348,160 = 46,305,280 B`
= **4.156 bpw** (group_size 128, 4-bit, asymmetric — `config.json quantization_config`, read
this session). Our PXQ4 `ffn_gate` of the same shape is **47,384,576 B = 4.2540 bpw**
(`4.25 + 16/K`, `ggml.h:465-467`). **PXQ4 is 2.33% larger per tensor than their AWQ.**

Decode bytes read per GPU at TP=4 (weights only; embed is a gather, not a read; MTP not loaded):

| policy | GiB/GPU/token | vs AWQ | PROJECTED peak tok/s |
|---|---:|---:|---:|
| **incumbent AWQ** (measured 92.8 peak / 57.4 median) | **3.547** | — | 92.8 (MEASURED) |
| **P1** — PXQ4 for ffn+o_proj+GDN-in_proj only, rest fp16 | 4.615 | +30.1% | **≈ 71.3** |
| **P2a** — + `linear_attn.out_proj` (ssm_out) re-encoded | 4.099 | +15.6% | ≈ 80.3 |
| **P2b** — + `lm_head` re-encoded | 3.664 | +3.3% | ≈ 89.8 |
| **P2c** — + `self_attn.qkv_proj` (needs k/v re-encoded) | **3.237** | **−8.7%** | **≈ 101.7** |

PROJECTION method, stated once: all figures scale the incumbent's own **measured** 92.8 tok/s
by the ratio of weight-bytes-read-per-GPU-per-token, i.e. they assume our kernels sustain the
same effective HBM bandwidth (462 GB/s/card implied) as their TurboMind sm70 GEMM. That
assumption is optimistic — `k_pxq6_mmv` is a scalar `__fmaf` kernel with no HMMA
(`pxq6.cuh:914-971`, `pxq6_dot32` `:634-674`), theirs is tensor-core. Treat +9% as a ceiling,
not a forecast. ASSUMPTIONS: TP=4, no MTP, KV unchanged, `lm_head` read in full every step.

### What this means for the decision
The bandwidth case for PXQ4-in-vLLM is **marginal at the PXQ4 tier**. The defensible reasons
to build it are:
1. **Quality-per-bit at ≈ equal bytes** — PXQ4 is imatrix-calibrated with a learned 16-entry
   codebook (`pxa.pxq6.book` in the file) versus their uniform int4/g128. If PXQ4 measurably
   beats AWQ on the seats' own eval at parity bytes, that is the product.
2. **It unlocks the lower tiers.** The bandwidth win only becomes large at PXQ3/PXQ2
   (ids 255/254). This port is the *vehicle*; PXQ4 is the first payload, not the last.
3. Their deployment structurally cannot take the `lm_head` win (`lm_head` is in their 311-entry
   `ignore` list). We can.

**GATE G0 (owner decision, no code, do it first): does PXQ4 beat the AWQ twin on quality at
parity bytes?** If it does not, stop — a ~+9% ceiling does not pay for ~2,500 LOC. This gate
needs no new code: llama.cpp PXQ4 is already running, their vLLM is already serving.

---

## 1. Which design was chosen, and what was grafted in

**Chosen spine: `minimal-risk`.** Its ordering — *risk-retired-per-unit-work*, with a
fully-dequantized fp16 checkpoint running on stock vLLM before one line of CUDA exists — is
the single most valuable idea in the three documents, and it is the only one that maximises
(a) probability of a correct first token and (d) fraction verifiable without a GPU. Its two
CPU-only bit-exactness gates retire >90% of format risk before the lease is ever needed.

**Grafted from `least-code`:**
- The economics correction above (verified; it is right and it is load-bearing).
- The GDN name mapping: the fork expects the projection **already split on disk** as
  `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a` (`qwen3_5.py:493-494,506-507`), so our
  `attn_qkv`→`in_proj_qkv` and `attn_gate`→`in_proj_z` map 1:1 with **no row permutation**.
- The insight that `lm_head` is the biggest single lever and should be sourced from the AWQ
  twin's **BF16** `lm_head.weight` (unquantized, in their ignore list) rather than from our
  Q8_0 copy, avoiding double quantization.

**Grafted from `max-performance`:**
- Re-encoding is **mandatory, not optional**. It is promoted from "phase 2 maybe" to the
  defined P2, with the encoder tool specified as a first-class component.
- The commitment to never set `_sm70_f16_prepared` and to set `_sm70_f16_forbidden`
  defensively (verified: `linear.py:61-63`).

**Rejected from `least-code`:** `ModelRegistry.register_model("Qwen3_5ForCausalLM", …)` to drop
the vision tower. **It is not 2 lines and not free.** `Qwen3_5ForCausalLMBase` is declared
`(nn.Module, HasInnerState, SupportsEagle3, SupportsLoRA, SupportsPP)` — **no `IsHybrid`**
(`qwen3_5.py:658-664`), while `Qwen3_5ForConditionalGeneration` declares it explicitly
(`qwen3_5.py:819`). `ModelConfig.is_hybrid` reads `self._model_info.is_hybrid`
(`config/model.py:1630-1631`) and drives mamba-state allocation at `:1764`. Registering the
text-only class would silently lose hybrid cache config. Also verified: `--language-model-only`
exists (`arg_utils.py:1211`) but only zeroes per-modality limits (`config/multimodal.py:315`);
it does **not** skip constructing `self.visual` (`qwen3_5.py:843`, unconditional).
**Decision: keep `Qwen3_5ForConditionalGeneration` and copy the 333 BF16 visual tensors
(921,460,192 B) verbatim from the AWQ twin into our checkpoint.** They cost ≈0.21 GiB/GPU
resident and **zero** decode bandwidth. Revisit in P3 only.

**Rejected from all three: the mixed-precision fused layer.** See §3.1 — the invariant
"every quantized linear is *uniformly* PXQ4" removes ~250 LOC of the riskiest code (custom
param loaders that silently mis-shard) and is achievable by policy alone.

---

## 2. Verified facts this plan rests on

Everything below was read in source **this session** unless marked otherwise.

### 2.1 The plugin seam
- `register_quantization_config(name)` — `quantization/__init__.py:57-102`. Appends to the
  runtime `QUANTIZATION_METHODS` list (`:46`), appends to
  `current_platform.supported_quantization`, hard-requires `issubclass(cls, QuantizationConfig)`,
  stores in `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` (`:54`), which **overrides built-ins** at
  lookup. Entry point group `vllm.general_plugins`, loaded in every engine-core and worker
  process (`arg_utils.py:749`, `v1/engine/core.py:108`, `v1/worker/worker_base.py:247`
  — per `03-vllm-plugin-surface.md`). **Zero patches to `/opt/1Cat-vLLM`.**

### 2.2 TP sharding is free, on both axes, with stock loaders
- `_ColumnvLLMParameter.load_column_parallel_weight` (`parameter.py:145-151`) narrows dim
  `output_dim` using `self.data.shape[output_dim]` — **panel units, automatically**.
- `_ColumnvLLMParameter.load_merged_column_weight` (`parameter.py:153-173`): if the param is a
  `PackedColumnParameter`/`PackedvLLMParameter` **and `packed_dim == output_dim`**, it divides
  `shard_offset`/`shard_size` by `packed_factor` before narrowing. With `packed_factor=64` a
  logical row offset becomes a panel offset.
- `MergedColumnParallelLinear.weight_loader_v2` (`linear.py:1140-1205`) computes
  `shard_offset = sum(output_sizes[:id]) // tp_size`, `shard_size = output_sizes[id] // tp_size`.
- `_load_fused_module_from_checkpoint` (`linear.py:1100-1138`) — the tuple-shard path used by
  `("in_proj_qkvz", "in_proj_qkv", (0,1,2))` — **also** packing-adjusts before
  `loaded_weight.narrow(param.output_dim, …)`. Verified integral for every real shape.
- `RowvLLMParameter.load_row_parallel_weight` (`parameter.py:220-230`) narrows dim `input_dim`
  using `self.data.shape[input_dim]` and **never consults `packed_factor`** — so with the slab
  tensor's dim 1 already in slab units, the K split lands on slab boundaries for free.
- A param that has `output_dim` but **no** `input_dim` falls through to
  `BasevLLMParameter.load_row_parallel_weight` → `_assert_and_load` (`parameter.py:93-103`)
  = **full copy**. That is exactly the required 128 B anchor-header duplication, for free.
- **The silent failure mode**: `_adjust_shard_indexes_for_packing` (`parameter.py:605-610`)
  does `round(shard_size // packed_factor)` — a misaligned offset **truncates without raising**,
  yielding a well-formed wrong slice and a model that loads cleanly with subtly wrong logits.
  Hard `%64`/`%32` asserts in `create_weights` are the only defence. Non-negotiable.

### 2.3 Real shard arithmetic (all exact, no truncation, at TP=2 and TP=4)
`MergedColumnParallelLinear in_proj_qkvz`, `output_sizes=[2048,2048,6144,6144]`
(`qwen3_5.py:214-224`, `key_dim=2048`, `value_dim=6144`), TP=4 →
offsets/sizes per rank `(0,512)(512,512)(1024,1536)(2560,1536)` → ÷64 → `(0,8)(8,8)(16,24)(40,24)`.
`gate_up_proj` `[17408,17408]` TP=4 → `(0,4352)(4352,4352)` → `(0,68)(68,68)`.
`QKVParallelLinear` TP=4 → q `(0,3072)`→`(0,48)`, k `(3072,256)`→`(48,4)`, v `(3328,256)`→`(52,4)`.
Row-parallel `o_proj` K=6144→1536 (48 slabs), `down_proj` K=17408→4352 (136 slabs),
`out_proj` K=6144→1536. **Zero remainders anywhere.**

### 2.4 The GDN split trap and its key
`_uses_split_gdn_input_projections(quant_config)` (`qwen3_5.py:127-157`) scans
`quant_config.modules_to_not_convert / ignored_layers / ignore` and `quant_config.config["ignore"]`,
returning True iff any entry is `linear_attn`, `*.linear_attn`, or contains
`linear_attn.in_proj_a` / `linear_attn.in_proj_b`. When True:
- `create_qkvz_proj` builds `output_sizes=[2048,2048,6144,6144]` (`qwen3_5.py:212-230`) —
  all multiples of 64 at TP=2 and TP=4;
- `create_ba_proj` builds a separate `in_proj_ba` with `output_sizes=[48,48]`;
- `load_weights` adds `("in_proj_ba","in_proj_b",0)` / `("in_proj_ba","in_proj_a",1)`
  (`qwen3_5.py:487-508`).
When False, the `48`-row `b`/`a` are folded into `in_proj_qkvz` → 12 rows/rank at TP=4 → the
packed shard arithmetic truncates. **Our config MUST expose `ignore` containing
`"linear_attn.in_proj_a"` and `"linear_attn.in_proj_b"`.** This is a hard requirement, not a
style choice.

### 2.5 `attn_q` needs no permutation
`Qwen3NextAttention.__init__` (`qwen3_next.py:502-513`) builds
`QKVParallelLinear(hidden_size, head_dim=256, total_num_heads=24*(1+attn_output_gate)=48,
total_num_kv_heads=4)` → q block = 12288 rows, matching our `attn_q`. The forward
(`qwen3_next.py:564-571`) does `q_gate.view(..., num_heads, -1); torch.chunk(2, dim=-1)`
= **per-head interleave `[q_h(256) | gate_h(256)]`**, identical to
`llama-build-context.cpp:2003-2007` (stride `2*row_size`, gate at `offset=row_size`).
A contiguous 3072-row (TP=4) slice = 6 whole `(q,gate)` head-pairs. **Converter emits `attn_q`
byte-for-byte with no reorder.**

### 2.6 The sm70 fast-path bypass
`_maybe_sm70_dense_forward` (`linear.py:56-96`) runs **before** `quant_method.apply()` in every
`forward()` (`:805`, `:1794`, `:612`, `:425`). It returns early if
`layer._sm70_f16_forbidden` (`:61-62`) and requires `layer._sm70_f16_prepared` (`:63`), which
only `UnquantizedLinearMethod.process_weights_after_loading` sets (`:408`). Our method must
**never** set `_sm70_f16_prepared` and **must** set `_sm70_f16_forbidden = True` in
`create_weights`. Note `_mark_default_sm70_dense_modules` (`qwen3_5.py:167-177`) sets
`_sm70_f16_force_enable = True` on every `qkv_proj`/`out_proj` — inert without `_prepared`,
but set the forbid flag anyway.

### 2.7 Toolchain (read this session, inside the running container)
Python 3.12.3 · torch **2.10.0+cu128** · nvcc **12.8** V12.8.93 · gcc 12.4.0 ·
vllm `0.1.dev1+g2ceb15066` at `/opt/vllm-venv/lib/python3.12/site-packages/vllm`, source at
`/opt/1Cat-vLLM`. **Container `/` is 100% full (207 G used, 0 avail); `/path/to/models` has 567 G.**
Nothing may be installed into the image. Everything we ship lives under `/path/to/models` and is
reached by `PYTHONPATH` + a hand-written `.dist-info` (see §7.4).

### 2.8 PXQ4 device format (re-read this session)
`pxq6_pol_p6` (`pxq6.cuh:317-346`): `SLAB=1088, HDR=128, CODE_OFF=64, NEFF=2`;
`row_effs` = `eff[0]=anch*sub[b&0xf]` (elems 0–15), `eff[1]=anch*sub[b>>4]` (elems 16–31);
`pair()` = `tab[byte&0xf], tab[byte>>4]`, LE byte `b` of the 16-byte code row.
`pxq6_panel_stride<POL>(kslabs) = HDR + kslabs*SLAB` (`pxq6.cuh:520-522`).
`k_pxq6_dequant_matrix` (`pxq6.cuh:680-726`): 1 block per slab, 64 threads, one row each,
stages a `[64][34]` tile then stores one warp per row along K.
`k_pxq6_mmv` (`pxq6.cuh:914-971`): `__launch_bounds__(256)`, block = 4 ksegs × 64 rows
(`PXQ4_MMV_KSEG=4`, `pxq4.cuh:114`), grid `(R/64, j, iy)`,
`extern __shared__ float pxq6_smem[]` sized **`(K + 256) * 4` bytes**, consumes
**fp32** activations and writes **fp32** output. CPU reference `pxa_deq_row_pxq6`
in `ggml/src/pxq-cpu.c` (`pxa_deq_pairs16` + panel walk).

---

## 3. Phase plan

### P1 — CORRECT (no re-encoding, no new numerics)
Serve as PXQ4 exactly the tensors that already are PXQ4 **and** sit in a uniformly-PXQ4 vLLM
module. Dequantize everything else to fp16 offline.

PXQ4-served modules (308 of the 325 id-252 tensors, 11,663,335,424 B):
`mlp.gate_up_proj` (130 ggml tensors) · `mlp.down_proj` (65) · `self_attn.o_proj` (17) ·
`linear_attn.in_proj_qkvz` (96 = 48 `attn_qkv` + 48 `attn_gate`).

fp16 in P1: `self_attn.qkv_proj` (because `attn_k`/`attn_v` are Q8_0 and the module is fused —
see §3.1) · `linear_attn.in_proj_ba` · `linear_attn.out_proj` (MXFP4) · `lm_head` (Q8_0) ·
`embed_tokens` (Q6_K) · `conv1d`, all norms, `visual.*`.

Result: **4.615 GiB/GPU decode read, ≈71 tok/s PROJECTED.** P1 is a correctness milestone.
Do not quote a throughput number for it without the word "regression" in the same sentence.

### P2 — COMPETITIVE (adds the encoder)
Re-encode to PXQ4, in this order (each is independently gateable):
- **P2a** `linear_attn.out_proj` — 48 tensors, MXFP4 → PXQ4, `−0.516 GiB/GPU`.
- **P2b** `lm_head` — 1 tensor, **source the AWQ twin's BF16 `lm_head.weight`**, not our Q8_0
  copy, `−0.435 GiB/GPU`. Note our own backbone table deliberately pins `output=q8_0`
  (`pxa.pxq.backbone_map`); **run a quality gate on a 4-bit LM head before committing.**
- **P2c** `attn_k`/`attn_v` — 34 tensors, Q8_0 → PXQ4, which makes `self_attn.qkv_proj`
  uniformly PXQ4 and lets `attn_q` (17 tensors, 568,614,912 B) finally be served.
  `−0.427 GiB/GPU`. Our backbone pins k/v to q8_0 for quality reasons: **gate this too.**
- Optional: `embed_tokens` Q6_K → PXQ4 (resident only, zero decode benefit).

Result at P2c: **3.237 GiB/GPU, ≈102 tok/s PROJECTED (+9.6%).**

### P3 — LATER (not planned here)
MTP (`blk.64` → `mtp.*`, spec decode) · WMMA prefill (`k_pxq6_gemm_grouped_wmma`,
`pxq6.cuh:2912`, currently MoE-only) · gufuse SwiGLU FFN fusion · PXQ3/PXQ2 tiers ·
text-only model class.

### 3.1 THE INVARIANT (read this twice)
> **Every vLLM linear module served by `PXQ4LinearMethod` is *uniformly* PXQ4 across all of its
> `output_partition_sizes`. There is no mixed-precision fused module, ever.**

`create_weights` asserts it. The converter enforces it. This is why P1 leaves `qkv_proj` in
fp16 rather than inventing a per-shard-dispatch parameter class: `QKVParallelLinear` is
hard-wired in `Qwen3NextAttention` (`qwen3_next.py:505`) with no split seam, so a PXQ4 `q`
next to fp16 `k`/`v` would require custom `load_qkv_weight` overrides — the single most likely
source of a silently mis-sharded, cleanly-loading, subtly-wrong model. P2c dissolves the
problem instead of coding around it. Cost of the deferral: 0.366 GiB/GPU during P1 only.

---

## 4. Repository layout

One git repo, **on this machine** at
`<scratch>/pxq-vllm/pxq4-vllm/`,
deployed to the DGX at `/path/to/models/pxa-vllm-pxq4/` (never to `/`, never to container `/`).

```
pxq4-vllm/
  pyproject.toml                     # §7.4
  README.md
  src/pxq4_vllm/
    __init__.py                      # register()  — the entry point
    config.py                        # PXQ4Config
    linear.py                        # PXQ4LinearMethod
    parameters.py                    # PXQ4SlabParameter, PXQ4AnchorParameter
    ops.py                           # torch.ops.pxq4 wrappers + register_fake + Workspace
    layout.py                        # SHARED: panel/slab arithmetic constants + helpers
    reference.py                     # SHARED: numpy bit-exact dequant (no torch, no CUDA)
    _lib/                            # libpxq4_sm70.so lands here (gitignored)
  csrc/
    pxq4_tables.h                    # vendored: PXQ6_* constants, book, sub
    pxq4_vendor.cuh                  # vendored: pol_p6, ldcodes, dot32, dequant, mmv (~500 ln)
    pxq4_ops.cu                      # host launchers + fp16<->fp32 staging kernels
    pxq4_torch.cpp                   # TORCH_LIBRARY(pxq4, …)
    CMakeLists.txt
  tools/pxq4_gguf/
    gguf_raw.py                      # raw struct GGUF reader (does NOT import `gguf`)
    dequant_ref.py                   # numpy q8_0 / q6_k / mxfp4 / f32 dequant
    namemap.py                       # ggml name -> HF name, single source of truth
    convert.py                       # CLI
  tools/pxq4_encode/                 # P2 only
    encode_shim.cpp                  # #includes mgv-wt's pxq6-quantize.inc.cpp (read-only)
    CMakeLists.txt
  tests/
    test_layout.py  test_shard.py  test_names.py  test_ops.py  test_e2e.py
```

`src/pxq4_vllm/layout.py` and `reference.py` are **imported by both the converter and the
tests**. They are the contract surface; they must not import torch or vllm.

---

## 5. Component A — the offline converter (`tools/pxq4_gguf/`)

Pure Python + numpy. No torch, no CUDA, no vLLM, no GPU. **Fully testable on this machine.**

### 5.1 Why not vLLM's GGUF loader
`gguf.GGMLQuantizationType(252)` raises inside `GGUFReader._build_tensors`, which kills the
whole file open; and vLLM's generic GGUF sharder slices rows assuming per-row-contiguous
blocks, which panel interleave violates. `gguf_raw.py` parses the header/KV/tensor-directory
by `struct` and `np.memmap`s the data section. (Existing working prototype:
`prototypes/pxq-vllm/tc.py`.)

### 5.2 CLI
```
python -m pxq4_gguf.convert \
  --gguf     /path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf \
  --ref-hf   /path/to/hf/<reference-hf-model> \
  --out      /path/to/models/pxa-models/Qwen3.8-27B-PXQ4-vllm \
  --policy   {p1,p2a,p2b,p2c} \
  [--encoder /path/to/models/pxa-vllm-pxq4/build/pxq4_encode.so]      # P2 only
  [--shard-size-gb 4]
```
`--ref-hf` is mandatory: the converter **copies** `config.json` (rewriting only
`quantization_config`), `tokenizer*`, `generation_config.json`, `preprocessor_config.json`,
`chat_template*`, and the **333 `model.visual.*` BF16 tensors** from the AWQ twin. Copying the
architectural config verbatim is deliberate — it guarantees every field the fork's
`Qwen3_5Config` reads is exactly what the incumbent already runs on.

### 5.3 Emitted tensors — THE CROSS-COMPONENT CONTRACT

For every module the policy marks PXQ4, emit **two** tensors and **no** `.weight`:

| key | dtype | shape | contiguity |
|---|---|---|---|
| `<module>.pxq4_slabs` | `uint8` | `[N//64, K//32, 1088]` | C-contiguous |
| `<module>.pxq4_anchor` | `float16` | `[N//64, 64]` | C-contiguous |

`N` = output rows (ggml `ne[1]`), `K` = input dim (ggml `ne[0]`).
Derivation from the GGUF tensor blob `B` of length `(N//64) * (128 + (K//32)*1088)`:

```python
P, S = N // 64, K // 32
a = np.frombuffer(B, dtype=np.uint8).reshape(P, 128 + S * 1088)
anchor = a[:, :128].copy().view('<f2')            # -> (P, 64)   float16
slabs  = a[:, 128:].copy().reshape(P, S, 1088)    # -> (P, S, 1088) uint8
```
**This is a pure split. No byte is reordered, no value is recomputed.** Round-trip
(`np.concatenate([anchor.view(u1), slabs.reshape(P,-1)], axis=1).tobytes() == B`) is asserted
per tensor by the converter.

For every other tensor: emit `<module>.weight` as `float16 [N, K]` (or the HF-native shape for
`conv1d`, norms, biases), dequantized by `dequant_ref.py`.

### 5.4 `namemap.py` — the complete map (single source of truth)

`L` = ggml block index. Full-attention blocks are `L % 4 == 3` (`qwen35.full_attention_interval=4`);
GDN blocks are `L % 4 != 3, L < 64`; `L == 64` is MTP.
HF prefix: `H = f"model.language_model.layers.{L}"`.

| ggml | HF | P1 type | shape (N,K) |
|---|---|---|---|
| `token_embd.weight` | `model.language_model.embed_tokens.weight` | fp16 | 248320×5120 |
| `output.weight` | `lm_head.weight` | fp16 (P2b: PXQ4) | 248320×5120 |
| `output_norm.weight` | `model.language_model.norm.weight` | fp16 | 5120 |
| `blk.L.attn_norm.weight` | `H.input_layernorm.weight` | fp16 | 5120 |
| `blk.L.post_attention_norm.weight` | `H.post_attention_layernorm.weight` | fp16 | 5120 |
| `blk.L.ffn_gate.weight` | `H.mlp.gate_proj.*` | **PXQ4** | 17408×5120 |
| `blk.L.ffn_up.weight` | `H.mlp.up_proj.*` | **PXQ4** | 17408×5120 |
| `blk.L.ffn_down.weight` | `H.mlp.down_proj.*` | **PXQ4** | 5120×17408 |
| *GDN (48)* | | | |
| `blk.L.attn_qkv.weight` | `H.linear_attn.in_proj_qkv.*` | **PXQ4** | 10240×5120 |
| `blk.L.attn_gate.weight` | `H.linear_attn.in_proj_z.*` | **PXQ4** | 6144×5120 |
| `blk.L.ssm_out.weight` | `H.linear_attn.out_proj.weight` | fp16 (P2a: PXQ4) | 5120×6144 |
| `blk.L.ssm_beta.weight` | `H.linear_attn.in_proj_b.weight` | fp16 | 48×5120 |
| `blk.L.ssm_alpha.weight` | `H.linear_attn.in_proj_a.weight` | fp16 | 48×5120 |
| `blk.L.ssm_conv1d.weight` | `H.linear_attn.conv1d.weight` | fp16 | →`(10240,1,4)` |
| `blk.L.ssm_a` | `H.linear_attn.A_log` | fp16 | 48 |
| `blk.L.ssm_dt.bias` | `H.linear_attn.dt_bias` | fp16 | 48 |
| `blk.L.ssm_norm.weight` | `H.linear_attn.norm.weight` | fp16 | 128 |
| *full-attn (17)* | | | |
| `blk.L.attn_q.weight` | `H.self_attn.q_proj.weight` | fp16 (P2c: PXQ4) | 12288×5120 |
| `blk.L.attn_k.weight` | `H.self_attn.k_proj.weight` | fp16 (P2c: PXQ4) | 1024×5120 |
| `blk.L.attn_v.weight` | `H.self_attn.v_proj.weight` | fp16 (P2c: PXQ4) | 1024×5120 |
| `blk.L.attn_output.weight` | `H.self_attn.o_proj.*` | **PXQ4** | 5120×6144 |
| `blk.L.attn_q_norm.weight` | `H.self_attn.q_norm.weight` | fp16 | 256 |
| `blk.L.attn_k_norm.weight` | `H.self_attn.k_norm.weight` | fp16 | 256 |
| *MTP `L==64`* | `mtp.*` — **P3, not emitted in P1/P2** | — | — |

Three mappings are **INFERENCE, not verified**, and gate G5 exists to catch them:
1. `ssm_beta`→`in_proj_b`, `ssm_alpha`→`in_proj_a` (name-led; `qwen3_5.py:265-272` splits
   `ba` as `b = ba[..., :48]`, `a = ba[..., 48:]`, and the stacked mapping is
   `(in_proj_ba, in_proj_b, 0)`, `(in_proj_ba, in_proj_a, 1)`).
2. `ssm_a` → `A_log` (whether ggml stores `A` or `log A`).
3. `ssm_conv1d` ggml `ne=(4,10240)` → HF `(10240,1,4)` reshape orientation.

`namemap.py` must expose:
```python
GGML_TO_HF: Callable[[str, dict], str | None]     # (ggml_name, gguf_kv) -> HF name or None
PXQ4_MODULES_P1: frozenset[str]                   # vLLM module suffixes served by PXQ4
HF_MODULE_OF: Callable[[str], str]                # HF tensor name -> owning vLLM module prefix
```
Note the two-into-one merges: `mlp.gate_proj` + `mlp.up_proj` are separate on disk (vLLM fuses
them into `gate_up_proj` via `packed_modules_mapping`, `qwen3_5.py:665-675`), and
`in_proj_qkv` + `in_proj_z` are separate on disk (fused into `in_proj_qkvz`). **The converter
emits them separately.** Do not pre-fuse.

### 5.5 `quantization_config` written into `config.json`
```json
{
  "quant_method": "pxq4",
  "pxq4_version": 1,
  "tier": "core",
  "type_id": 252,
  "panel_rows": 64, "slab_cols": 32, "slab_bytes": 1088, "header_bytes": 128,
  "book": [ ...16 floats, verbatim from gguf KV pxa.pxq6.book... ],
  "sub":  [ ...16 floats, verbatim from gguf KV pxa.pxq6.sub... ],
  "backbone_rev": 2,
  "backbone_map": "<verbatim pxa.pxq.backbone_map>",
  "pxq4_modules": ["mlp.gate_up_proj","mlp.down_proj","self_attn.o_proj",
                   "linear_attn.in_proj_qkvz"],
  "ignore": ["linear_attn.in_proj_a", "linear_attn.in_proj_b",
             "linear_attn.in_proj_ba", "linear_attn.out_proj",
             "self_attn.qkv_proj", "lm_head", "model.visual"]
}
```
`pxq4_modules` is an **allow-list of vLLM module suffixes**; `ignore` exists for the fork's
`_uses_split_gdn_input_projections` probe (`qwen3_5.py:127-157`) and as belt-and-braces.
P2 policies move entries from `ignore` to `pxq4_modules` (`p2c` adds `self_attn.qkv_proj`).
**The two entries `linear_attn.in_proj_a` / `linear_attn.in_proj_b` must never be removed.**

### 5.6 Converter self-checks (fail the run, do not warn)
1. Every PXQ4 tensor: `N % 64 == 0 and K % 32 == 0`; on-disk size ==
   `(N//64) * (128 + (K//32)*1088)`.
2. Split → rejoin round-trips to the original bytes.
3. `config["book"] == gguf_kv["pxa.pxq6.book"]` and likewise `sub`; both also compared against
   the vendored `csrc/pxq4_tables.h` values.
4. Emitted key set ⊇ (AWQ twin key set with `weight_packed|weight_scale|weight_zero_point|weight_shape`
   collapsed to `weight`), minus `mtp.*`, plus the `pxq4_slabs`/`pxq4_anchor` substitutions.
   Report any symmetric difference.
5. For every PXQ4 module: all of its `output_partition_sizes` at TP∈{1,2,4} are `% 64 == 0`,
   and `K/tp % 32 == 0` for row-parallel ones.
6. Every emitted module is uniformly one type (§3.1 invariant).

---

## 6. Component B — the runtime package (`src/pxq4_vllm/`)

### 6.1 `__init__.py`
```python
def register() -> None:
    """vllm.general_plugins entry point. Imports the config module, whose
    @register_quantization_config('pxq4') decorator does the registration.
    Must be idempotent: it runs in the engine-core process and in every TP worker."""
```

### 6.2 `layout.py` (no torch, no vllm)
```python
PANEL_ROWS   = 64
SLAB_COLS    = 32
SLAB_BYTES   = 1088
HEADER_BYTES = 128
CODE_OFF     = 64
CODE_BYTES   = 16
TYPE_ID      = 252

def panel_bytes(K: int) -> int                      # 128 + (K//32)*1088
def tensor_bytes(N: int, K: int) -> int             # (N//64)*panel_bytes(K)
def slab_shape(N: int, K: int) -> tuple[int,int,int]    # (N//64, K//32, 1088)
def anchor_shape(N: int) -> tuple[int,int]              # (N//64, 64)
def assert_geometry(N: int, K: int) -> None         # raises on %64 / %32 violation
def split_blob(blob: bytes|np.ndarray, N: int, K: int) -> tuple[np.ndarray, np.ndarray]
def join_blob(slabs: np.ndarray, anchor: np.ndarray) -> np.ndarray
```

### 6.3 `reference.py` (numpy only — the bit-exactness oracle)
```python
BOOK: np.ndarray   # float32[16], from ggml-pxq6-tables.h PXQ6_BOOK_INIT
SUB:  np.ndarray   # float32[16], PXQ6_SUB16_INIT

def dequant(slabs: np.ndarray, anchor: np.ndarray, *,
            book: np.ndarray = BOOK, sub: np.ndarray = SUB) -> np.ndarray:
    """slabs uint8[P,S,1088], anchor float16[P,64] -> float32[P*64, S*32].
    Must reproduce pxa_deq_row_pxq6 (ggml/src/pxq-cpu.c) EXACTLY in fp32:
      anch      = float32(anchor[p, r])                     # fp16 -> fp32 widening
      sb        = slabs[p, kb, r]
      eff[0]    = anch * sub[sb & 0xF]     # elements 0..15
      eff[1]    = anch * sub[sb >> 4]      # elements 16..31
      byte      = slabs[p, kb, 64 + 16*r + b]               # b = 0..15
      w[2b]     = eff[(2b)>>4]   * book[byte & 0xF]
      w[2b+1]   = eff[(2b+1)>>4] * book[byte >> 4]
    Output row index = p*64 + r, column index = kb*32 + element."""
```
The multiply order `(anchor * sub) * book` is load-bearing for bit-exactness — do not
reassociate to `anchor * (sub * book)`.

### 6.4 `parameters.py`
```python
from vllm.model_executor.parameter import PackedColumnParameter, PackedvLLMParameter

class PXQ4SlabParameter(PackedvLLMParameter):
    """uint8 [N/64, K/32, 1088]. Constructed with
       output_dim=0, input_dim=1, packed_dim=0, packed_factor=64.
    Column split  -> narrow(0) in whole panels (packing-adjusted by the stock loader).
    K split       -> narrow(1) in whole slabs  (row loader ignores packed_factor)."""

class PXQ4AnchorParameter(PackedColumnParameter):
    """float16 [N/64, 64]. Constructed with output_dim=0, packed_dim=0, packed_factor=64.
    Deliberately has NO input_dim: on a RowParallelLinear it falls to
    BasevLLMParameter._assert_and_load = full copy = the required header duplication."""
```
Neither class overrides any `load_*` method. **If an implementer finds themselves writing a
custom `load_qkv_weight` or `load_merged_column_weight`, the §3.1 invariant has been broken —
stop and re-read §3.1.**

### 6.5 `config.py`
```python
@register_quantization_config("pxq4")
class PXQ4Config(QuantizationConfig):
    def __init__(self, *, pxq4_modules: list[str], ignore: list[str],
                 book: list[float], sub: list[float],
                 pxq4_version: int = 1, tier: str = "core",
                 backbone_rev: int | None = None,
                 backbone_map: str | None = None) -> None: ...
        # MUST call super().__init__() (sets self.packed_modules_mapping, base_config.py:73-76)
        # MUST expose self.ignore as a plain list[str]  <-- qwen3_5.py:127-157 reads it

    def get_name(self) -> str:                        return "pxq4"
    def get_supported_act_dtypes(self) -> list:       return [torch.half]
    @classmethod
    def get_min_capability(cls) -> int:               return 70
    @staticmethod
    def get_config_filenames() -> list[str]:          return []
    @classmethod
    def from_config(cls, config: dict) -> "PXQ4Config": ...
    def get_quant_method(self, layer, prefix: str): ...
```
`get_quant_method` contract, exactly:
```
if isinstance(layer, LinearBase):
    if _matches(prefix, self.ignore):        return UnquantizedLinearMethod()
    if _matches(prefix, self.pxq4_modules):  return PXQ4LinearMethod(self)
    return UnquantizedLinearMethod()
return None                    # VocabParallelEmbedding / ParallelLMHead / attention
```
`_matches(prefix, pats)` = `any(p == prefix or prefix.endswith("." + p) or p in prefix
                                for p in pats)`. Ignore is checked **first**.
Returning `None` for non-`LinearBase` yields `UnquantizedEmbeddingMethod`
(`vocab_parallel_embedding.py:479-482`) — which is what we want in P1 and P2a/P2b-with-fp16-head.
Also implement `override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None)`
returning `"pxq4"` when `hf_quant_cfg.get("quant_method") == "pxq4"`, so the checkpoint
self-selects without a CLI flag.

### 6.6 `linear.py`
```python
class PXQ4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: PXQ4Config) -> None: ...

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       input_size: int, output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs) -> None:
        K = input_size_per_partition
        N = sum(output_partition_sizes)
        # --- the asserts that stand between us and a silently wrong model ---
        assert params_dtype is torch.float16, f"pxq4 requires fp16 acts, got {params_dtype}"
        assert K % 32 == 0,  f"pxq4 K={K} not %32 ({layer.prefix})"
        assert N % 64 == 0,  f"pxq4 N={N} not %64 ({layer.prefix})"
        for i, o in enumerate(output_partition_sizes):
            assert o % 64 == 0, f"pxq4 shard {i} size {o} not %64 ({layer.prefix})"
        wl = extra_weight_attrs["weight_loader"]
        slabs  = PXQ4SlabParameter(
            data=torch.empty(N // 64, K // 32, 1088, dtype=torch.uint8),
            output_dim=0, input_dim=1, packed_dim=0, packed_factor=64, weight_loader=wl)
        anchor = PXQ4AnchorParameter(
            data=torch.empty(N // 64, 64, dtype=torch.float16),
            output_dim=0, packed_dim=0, packed_factor=64, weight_loader=wl)
        layer.register_parameter("pxq4_slabs", slabs)
        layer.register_parameter("pxq4_anchor", anchor)
        layer.pxq4_N, layer.pxq4_K = N, K
        layer._sm70_f16_forbidden = True          # linear.py:61-62 ; NEVER set _sm70_f16_prepared
        PXQ4Workspace.reserve(dequant_elems=N * K, act_elems=PXQ4_MMV_MAX_M * max(N, K))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Assert contiguity and shapes only. Do NOT repack. Do NOT set _sm70_f16_prepared.
        ...

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor: ...
```
`apply` body, exactly:
```python
x2 = x.reshape(-1, x.shape[-1])
M, N = x2.shape[0], layer.pxq4_N
out = torch.empty((M, N), dtype=torch.float16, device=x2.device)
if M <= PXQ4_MMV_MAX_M:                       # default 8; env PXQ4_MMV_MAX_M
    torch.ops.pxq4.mmv_out(out, x2.contiguous(), layer.pxq4_slabs, layer.pxq4_anchor)
else:
    w = PXQ4Workspace.dequant_view(N, layer.pxq4_K)      # preallocated fp16 [N, K]
    torch.ops.pxq4.dequant_out(w, layer.pxq4_slabs, layer.pxq4_anchor)
    torch.mm(x2, w.t(), out=out)
if bias is not None:
    out.add_(bias)
return out.reshape(*x.shape[:-1], N)
```

### 6.7 `ops.py`
```python
PXQ4_MMV_MAX_M: int          # default 8 (matches PXA_PXQ4_2D_MAX_NY, ggml-cuda.cu:4021)

def load_library() -> None:  # torch.ops.load_library(_lib/libpxq4_sm70.so), idempotent

class PXQ4Workspace:
    """Per-device, module-level, allocated once BEFORE any CUDA-graph capture.
    Sized by the max over all layers seen in create_weights. Never resized in apply()."""
    @classmethod
    def reserve(cls, *, dequant_elems: int, act_elems: int) -> None: ...
    @classmethod
    def materialize(cls, device: torch.device) -> None: ...     # called once, pre-capture
    @classmethod
    def dequant_view(cls, N: int, K: int) -> torch.Tensor: ...  # fp16 [N,K] view, no alloc
    @classmethod
    def act_f32_view(cls, M: int, K: int) -> torch.Tensor: ...  # fp32 [M,K] staging
    @classmethod
    def out_f32_view(cls, M: int, N: int) -> torch.Tensor: ...  # fp32 [M,N] staging
```
Largest per-rank dequant buffer at TP=4 is `mlp.gate_up_proj` N=8704 × K=5120 fp16 = 85 MiB
(TP=2: 170 MiB). Budget for it.

`register_fake` implementations for both ops are **mandatory** — without them
`torch.compile` / piecewise capture will fail. Annotate the output as mutated
(`Tensor(a!)`) in the schema.

---

## 7. Component C — the CUDA extension (`csrc/`)

### 7.1 The op ABI (frozen; components B and C agree on exactly this)
```
pxq4::dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()
    out    : fp16, [N, K], contiguous, device cuda
    slabs  : uint8, [N/64, K/32, 1088], contiguous
    anchor : fp16,  [N/64, 64], contiguous
    N % 64 == 0, K % 32 == 0, N/64 == slabs.size(0) == anchor.size(0)

pxq4::mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()
    x      : fp16, [M, K], contiguous ; M <= PXQ4_MMV_MAX_M
    out    : fp16, [M, N], contiguous
    slabs, anchor as above

pxq4::version() -> int
```
Namespace is `pxq4`, **not** `_C` — we must not collide with the fork's own `torch.ops._C`.
Both ops preallocate nothing, allocate nothing, and are capture-safe.

### 7.2 Vendoring: what to copy, and the only three lines to change
Copy **verbatim** from `/path/to/engine-repo` (read-only source; do not edit that tree):

| from | symbols |
|---|---|
| `ggml/include/ggml-pxq6-tables.h:21-37` | `PXQ6_QK/TYPE_SIZE/BM/SLAB_BYTES/HDR_BYTES/ROW_META`, `PXQ6_BOOK_INIT`, `PXQ6_SUB16_INIT` |
| `ggml/src/ggml-cuda/pxa/pxq4.cuh:114` | `PXQ4_MMV_KSEG` (= 4) |
| `ggml/src/ggml-cuda/pxa/pxq6.cuh:317-346` | `struct pxq6_pol_p6` |
| `ggml/src/ggml-cuda/pxa/pxq6.cuh:436-464` | `pxq6_ldcodes` |
| `ggml/src/ggml-cuda/pxa/pxq6.cuh:520-526` | `pxq6_panel_stride`, `pxq6_panel` |
| `ggml/src/ggml-cuda/pxa/pxq6.cuh:634-674` | `pxq6_dot32`, `pxq6_pairx`, `pxq6_acc2`, `pxq6_mode` |
| `ggml/src/ggml-cuda/pxa/pxq6.cuh:680-726` | `k_pxq6_dequant_matrix` |
| `ggml/src/ggml-cuda/pxa/pxq6.cuh:914-971` | `k_pxq6_mmv` |
| the `pxq6_book_g` / `pxq6_sub16_g` `__device__` arrays and their upload helper | `pxq6.cuh:79 ff` |

Drop everything MoE (`gufuse`, `down_scat`, `grouped`), everything HQ (`pol_p6hq`), the
int8/MMVQ family, and the `pxa_pxq_fmt` ggml type mapper. Target ≈500 vendored lines.

**The only device-code edits (they change addressing, never arithmetic):**

In `k_pxq6_dequant_matrix`, replace
```cuda
const uint8_t * panel = wq + p*pxq6_panel_stride<POL>(kslabs);
const uint8_t * slab  = panel + POL::HDR + (size_t)kb*POL::SLAB;
const float anch = POL::HDR ? POL::anchor(panel, row) : 0.f;
```
with
```cuda
const uint8_t * slab = slabs + ((size_t)p*kslabs + kb)*POL::SLAB;
const float anch = __half2float(anchor[(size_t)p*64 + row]);
```
and add `const __half * __restrict__ anchor` to the signature, renaming `wq`→`slabs`.

In `k_pxq6_mmv`, replace
```cuda
const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;
...
pxq6_dot32<POL,MODE,VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch, ...)
```
with a `slabs`/`anchor` pair and `slabs + ((size_t)p*kslabs + kb)*POL::SLAB`. Delete the
`ids`/`e`/`n_as` expert machinery (we are always `e == 0`) and the `j` grid axis; keep
`iy` as the token axis. **`row_effs`, `ldcodes`, `pair`, `pxq6_acc2`, the `nfix` canonical
fold, and the reduction must be untouched** — that is the entire bit-exactness argument.

### 7.3 Host launchers (`pxq4_ops.cu`) — the non-obvious bits
- `dequant_out`: `grid = (N/64)*(K/32)`, `block = 64`, no dynamic smem.
- `mmv_out`: `grid = dim3(N/64, 1, M)`, `block = 256`,
  **dynamic smem = `(K + 256) * sizeof(float)`**. For `down_proj` at TP=1/2 this exceeds
  48 KiB (K=17408 → 70,656 B), so the launcher **must** call
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` once.
  V100 caps at 96 KiB/block; K=17408 fits, K≥24320 would not — add an explicit check that
  falls back to the dequant path.
- `k_pxq6_mmv` consumes **fp32 x** and writes **fp32 out** (`pxq6.cuh:920-923, 968-969`).
  v1 stages: `out_f32 = ws.out_f32_view(M,N)`, `x_f32 = ws.act_f32_view(M,K)`, two trivial
  elementwise convert kernels around the call. **Keep it this way** — llama.cpp also feeds
  f32 activations, so staging is what makes our result comparable to the shipping engine.
  A native-fp16 intake is a P3 optimization.
- `MODE` / `VECX`: read `pxa_pxq_mmv_mode()` in `ggml/src/ggml-cuda/ggml-cuda.cu` and hardcode
  the value that path selects for `cc == 700`. Record the number and its source line in a
  comment. Do not guess.
- Tables: upload `pxq6_book_g` / `pxq6_sub16_g` once per device at library init.

### 7.4 Build and deploy
Standalone `.so` linked against libtorch only; **no vLLM headers, no vLLM rebuild, no patch to
`/opt/1Cat-vLLM`.**
```
nvcc -O3 -std=c++17 -gencode arch=compute_70,code=sm_70 \
     --expt-relaxed-constexpr -Xcompiler -fPIC -shared
```
Build in a **throwaway container from the same image** with `-v /path/to/models:/path/to/models`,
writing only under `/path/to/models`. The production container's `/` is 100% full and must not be
written to; the production container must not be restarted.

Installation without touching the image:
```
/path/to/models/pxa-vllm-pxq4/site/
    pxq4_vllm/…                              (package)
    pxq4_vllm-0.1.0.dist-info/METADATA
    pxq4_vllm-0.1.0.dist-info/entry_points.txt   ->  [vllm.general_plugins]
                                                     pxq4 = pxq4_vllm:register
```
launched with `PYTHONPATH=/path/to/models/pxa-vllm-pxq4/site`. `importlib.metadata` discovers
`.dist-info` directories on `sys.path`, so the entry point resolves with nothing installed.
`pyproject.toml` stays in the repo for normal `pip install -e .` on machines with disk.

---

## 8. Gates — what must pass, in order

CPU-only gates run **on this machine, today**. GPU gates need the lease and are out of scope
for the current workflow.

| # | gate | needs | what it retires |
|---|---|---|---|
| **G0** | PXQ4 beats AWQ on quality at parity bytes | owner decision, no code | the project's reason to exist |
| **G1** | `reference.dequant()` == `pxa_deq_row_pxq6` (`pxq-cpu.c`), `np.array_equal` in **fp32**, on ≥3 real tensors of distinct shapes | CPU | >90% of format risk: panel math, slab offsets, nibble order, table values, multiply order |
| **G2** | split→join round-trips to the original GGUF bytes for all 325 PXQ4 tensors | CPU | converter layout |
| **G3** | `dequant(narrow(x)) == narrow(dequant(x))` bit-exact, **both axes**, TP∈{2,4}, on `gate_up`, `down`, `o_proj`, `in_proj_qkvz` | CPU | that the TP repack is a permutation; the silent-truncation class of bug |
| **G4** | emitted key set vs the AWQ twin's index: only the intended differences | CPU | name mapping |
| **G5** | S0 checkpoint (everything fp16, **no plugin, no CUDA**, stock vLLM) loads and its greedy continuations match llama.cpp on ≥20 prompts | GPU | name mapping semantics: `in_proj_b`/`in_proj_a` order, `A_log`, `conv1d` orientation, `q_proj` interleave |
| **G6** | `torch.ops.pxq4.dequant_out` == `reference.dequant` bit-exact in fp32 | GPU | the vendoring edits |
| **G7** | P1 checkpoint + plugin: per-layer output vs S0 within fp16 tolerance; then full-model same-top-token vs S0 ≥ 99.5% | GPU | `create_weights`/`apply`/sharding end to end |
| **G8** | `mmv_out` vs `dequant`+`mm` within tolerance; `cudagraph_mode=FULL_AND_PIECEWISE` captures; assert no `_sm70_f16_prepared` on any layer; assert zero allocations inside `apply()` | GPU | the capture/compile assumption (§10 risk 3) |
| **G9** | first throughput measurement. **Nothing before G9 may quote tok/s.** | GPU + lease | — |
| **G10** | P2 only: 4-bit `lm_head` and 4-bit `attn_k/v` each clear the seats' quality bar *before* they ship | GPU | that P2 buys speed without buying a quality regression |

**G1 and G3 together are the project.** They are pure numpy, they need no GPU, no lease, and
no container, and they retire the failure mode that the rest of the design cannot detect.

---

## 9. Parallel work split (four agents, four contracts)

| agent | owns | must not touch | blocked on |
|---|---|---|---|
| **A — converter** | `tools/pxq4_gguf/**`, `src/pxq4_vllm/layout.py`, `src/pxq4_vllm/reference.py` | `csrc/`, `linear.py`, `config.py` | nothing — start now |
| **B — runtime** | `src/pxq4_vllm/{__init__,config,linear,parameters,ops}.py` | `tools/`, `csrc/` | the §7.1 ABI (frozen here) |
| **C — kernels** | `csrc/**` | everything Python except `ops.py`'s docstring | the §7.1 ABI (frozen here) |
| **D — gates** | `tests/**` | all of the above | §5.3 + §6.3 + §7.1 (all frozen here) |

The three frozen contracts are: **§5.3** (what the converter emits), **§6.3** (`reference.dequant`
semantics), **§7.1** (the op ABI). Nothing else crosses a boundary. Agent A can reach G1–G4
with no input from B or C. Agent C can reach G6 against A's `reference.dequant` alone.

Estimated LOC: converter 850 · runtime python 420 · CUDA 900 (of which ~500 vendored verbatim)
· tests 350 · encoder shim (P2) 180. **≈2,700 total, ≈2,200 hand-written.
Files modified in `/opt/1Cat-vLLM` or `/path/to/engine-repo`: 0.**

---

## 10. Top risks

1. **The economics (§0).** P1 is a 23% regression and P2c is a +9% ceiling that assumes our
   scalar kernel matches their tensor-core kernel's effective bandwidth. If G0 does not show a
   quality win, this project should not be built. Say the regression first, every time, to
   everyone.
2. **Silent mis-sharding.** `round(shard_size // packed_factor)` (`parameter.py:605-610`)
   truncates without raising, and a PXQ4 weight row is not contiguous bytes — so a bad split
   loads cleanly and produces subtly wrong logits. Defences: the `%64`/`%32` asserts in
   `create_weights` (§6.6), the §3.1 uniformity invariant that removes the custom-loader class
   of bug entirely, and gate G3.
3. **UNVERIFIED: does an out-of-tree `torch.ops.pxq4.*` participate in `FULL_AND_PIECEWISE`
   capture the way `torch.ops._C` does?** Mitigations are `register_fake` + `Tensor(a!)`
   mutation annotations + the preallocated `PXQ4Workspace` (never allocate in `apply`), all
   specified above; the fallback is a `cudagraph_mode` change on the CLI. Tested at G8.
4. **`k_pxq6_mmv` is tuned for `ny <= 8`** (`ggml-cuda.cu:4021`) while vLLM's median decode
   batch is larger. If the crossover to dequant+`mm` lands below M≈16, only single-stream
   improves and the median case sees nothing. Fix is a batched mmv (~150 LOC), P3.
5. **Three unverified name mappings** (`in_proj_b`/`in_proj_a` order, `A_log`, `conv1d`
   orientation) each produce plausible output rather than an error. Gate G5 exists solely for
   these and must run before any kernel work is trusted.

---

## 11. Corrections to the brief, carried forward

- The artifact is **five** types, not four: `ssm_out` is **MXFP4 (ggml id 39)** on all 48 GDN
  layers, 802,160,640 B. There is **no F16 tensor in the file at all**.
- **PXQ4 is 2.33% larger per tensor than the incumbent's AWQ** (4.254 vs 4.156 bpw). The
  brief's "3.66 GiB/GPU vs their 4.64" compared different scopes; the like-for-like decode-read
  numbers are 4.615 (P1) / 3.237 (P2c) vs **3.547** (theirs).
- The incumbent's `lm_head` is **BF16 and unquantized** (in its 311-entry `ignore` list) — the
  single biggest lever available to us, and one they structurally do not have.
- `Qwen3_5ForCausalLM` exists (`qwen3_5.py:772`) but is **not registered** (`registry.py:560`
  lists only `Qwen3_5ForConditionalGeneration`) **and does not declare `IsHybrid`**. Registering
  it is not a free win. Copy the vision tower instead.
- `--language-model-only` exists but only zeroes multimodal limits; it does not skip building
  the tower.
- **MoE: void, and measured.** 866 tensors, zero `*_exps`, zero expert KVs. Zero effort, zero
  fallback, zero VRAM. Sky-35B is the MoE model.
