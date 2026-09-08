# 08 — PXQ4 in vLLM: least-code / maximum-leverage design

Target: `github.com/KewaiiGamer/1Cat-vLLM` @ `2ceb15066` (`v0.1.dev1+g2ceb15066`), 4x V100-SXM2 (sm_70), TP=4,
model Qwen3.8-27B (`qwen35`, dense hybrid, **no MoE**), artifact
`/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf`.

---

## 0. Read this first — the engineering works, the *stated goal* does not follow from it

The port is feasible with ~1,600 hand-written LOC, zero patches to Kewaii's tree, and two CUDA kernels.
**But the brief's throughput premise is arithmetically wrong, and I can show it from the two checkpoints
on disk.**

The brief says "PXQ4 sharded 4 ways = 3.66 GiB/GPU vs their AWQ 4.64 GiB → 110-120 tok/s". Those two
numbers are not like-for-like: 3.66 is `14.64/4` (our whole file, vocab included, vision excluded);
4.64 is their whole deployment (vocab **and** an 0.858 GiB vision tower included). Measured
like-for-like from the actual files:

| body = language-model layers 0-63 only, excl. lm_head / embed / MTP / vision | bytes | GiB |
|---|---:|---:|
| their AWQ W4A16 g128 body (sum of safetensors headers) | 12,691,100,928 | **11.820** |
| our PXQ4 file body (sum of GGUF tensor sizes) | 13,055,719,424 | **12.159** |

**Our body is 2.9% BIGGER than theirs.** That is not a measurement error, it is the format:
PXQ4 = `4.25 + 16/K` bpw = 4.253 bpw at K=5120 (`ggml.h:465-467`, reproduced byte-exactly by the census);
compressed-tensors pack-quantized 4-bit g128 asym = `4 + 16/128 + zp/128` ≈ 4.16-4.19 bpw. PXQ4 is
**not a smaller format than their AWQ.** It is a *better-conditioned* format at the same size
(fp16 row anchor + per-16 sub-scale + nonuniform PX16 book, quantized from Q8_0 with an imatrix).

Decode throughput on both engines is bytes-read-per-GPU-per-token (established fact in the brief). So:

| policy | resident/GPU | decode-read/GPU | vs AWQ | projected peak / median tok/s |
|---|---:|---:|---:|---:|
| incumbent AWQ (measured 92.8 / 57.4) | 4.354 GiB | **3.547 GiB** | 1.000 | 92.8 / 57.4 (MEASURED) |
| **A** PXQ4 native, everything else → fp16 | 4.782 | 4.190 | 1.181 | **78.6 / 48.6** ← a regression |
| **B** A + `ssm_out` on their existing sm70 MXFP4 kernel | 4.266 | 3.674 | 1.036 | 89.6 / 55.4 ← still a regression |
| **C** B + `lm_head` and `attn_k/v` at 4-bit | 3.774 | **3.182** | 0.897 | **103.5 / 64.0** |

(Byte columns are arithmetic on real tensor shapes — FACT. tok/s columns are PROJECTIONS: they assume
decode time scales linearly with per-GPU weight bytes read, that both engines stay at the same ~50% of
HBM peak, and that KV/activation traffic is unchanged. No GPU was run for this study.)

Three consequences that must shape the plan:

1. **The naive port (policy A) is 18% *slower* than what is already running on that box.** Any plan that
   stops at "PXQ4 kernel + fp16 for the other four types" ships a regression.
2. The single biggest lever is **not** PXQ4 at all — it is that their deployment carries an fp16
   `lm_head` (2.368 GiB, in their 311-entry `ignore` list) that is read on **every** decode step.
   Serving the head at 4 bits is worth 0.435 GiB/GPU/token, more than half of the entire policy-C win.
3. The honest value proposition is **"~10% faster than AWQ at better quality-per-bit, on an engine that is
   already 2-3x faster than our llama.cpp fork"** (their 92.8 vs our 47.96/63.76 with every lever on).
   The quality claim is the reason to do this. The speed claim is thin and must be re-derived after
   a real measurement, not promised.

Nothing below is blocked. But do not start until someone with authority agrees that ~10% + quality is
the goal, because policy C costs materially more than policy A.

---

## 1. The seam hunt (the design slant, answered)

I looked for a way to avoid a new quantization method entirely. Four candidates, three rejected.

**REJECTED — masquerade as compressed-tensors.** `CompressedTensorsConfig` parses `config_groups` into
typed schemes and dispatches in `_get_scheme_from_parts`; a PXQ4 scheme means a new
`schemes/compressed_tensors_pxq4.py` *plus* an `elif` in their dispatcher *plus* passing their
`num_bits`/`type`/`strategy`/`group_size` validators. That is a fork patch **and** more code than a
first-class config. No benefit.

**REJECTED — `VLLM_SM70_QUANT_BACKEND`.** It is `Literal["auto","marlin","turbomind"]` (`envs.py:115`,
`:789-790`, raises otherwise) consumed by hand-written `if`s inside the awq / awq_marlin / auto_gptq /
fp8 / compressed-tensors schemes (`envs.py:793-813`). It is a per-format tri-state override, not a
dispatch table. Not extensible without patching.

**REJECTED FOR v1, KEEP AS PHASE 2 — masquerade as NVFP4 through `nvfp4_sm70_prepare`.** This one is
genuinely tempting and deserves its rejection in writing, because it is the only path to tensor cores.
`prepare_nvfp4_linear` (`sm70_turbomind.py:256-287`, local copy) takes **arbitrary fp16 per-group
scales** — it folds `weight_scale * weight_global_scale` to fp16 offline (`:268-275`) — at **group 16**
(`NVFP4_GROUP_SIZE = 16`, `:16`), and `Config_NVF4` is instantiated at group 16 for sm_70
(`sm70_884_4.cu:117-123`). PXQ4's effective scale is exactly `anchor[row] * SUB16[s4[row][block]]` per
16 elements — it folds to an fp16 `[K/16, N]` tensor offline with no loss. So the scale half is a
perfect structural match, and the whole s884 HMMA mainloop would be inherited.

It fails on the codebook, and I can now prove it from the artifact itself. The file stores its book
(`pxa.pxq6.book`, read out of the GGUF this session):

```
[-0.98779, -0.73535, -0.55859, -0.41968, -0.30103, -0.19446, -0.09552, 0.0,
  0.08472,  0.17126,  0.26196,  0.36060,  0.47119,  0.60059, 0.76563, 1.0]
```

7 negatives, exactly one zero (index 7), 8 positives, max = 1.0 — **asymmetric, one zero**. E2M1 is
symmetric with **two** zeros (±0) and 14 distinct nonzero magnitudes. No scalar `c` makes
`PX16 == c * E2M1`; the zero multiplicity alone forbids it. Reusing the NVFP4 kernel therefore requires
editing the register-side 4-bit→half converter LUT (`Transform_HMMA_SIMT_B`, `kernels/core/mma.h:13-25`,
`arch/mma_sm70.h:12-36`) inside their vendored TurboMind — i.e. compiling a *variant* of
`csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` (8,492 lines) into our own `.so` against their headers.
Feasible (sources ship, Apache-2.0, `nvcc` 12.8 is in the container), high value for prefill, and
**explicitly out of scope for v1** — see §9, Option T.

**ACCEPTED — four real seams, all zero-patch:**

| seam | what it buys | cost |
|---|---|---|
| `@register_quantization_config("pxq4")` + `vllm.general_plugins` entry point (`quantization/__init__.py:57-101`, `plugins/__init__.py:14,69`, loaded at `arg_utils.py:749` and per-rank at `v1/worker/worker_base.py:247`) | the whole quant method, no fork patch | ~420 LOC |
| `ModelRegistry.register_model("Qwen3_5ForCausalLM", ...)` — their **own in-tree class** (`qwen3_5.py:772`), which `registry.py:560` never exposes (only `Qwen3_5ForConditionalGeneration`) | drops the unconditionally-constructed 0.858 GiB vision tower (`qwen3_5.py:843`) that our GGUF does not contain | **2 LOC** |
| vLLM's existing `narrow()`-based sharder, driven by `output_dim`/`input_dim`/`packed_dim`/`packed_factor` | column *and* row TP sharding of a panel-interleaved format with **no custom `weight_loader`** | 6 LOC of `set_weight_attrs` |
| their existing sm70 MXFP4 GEMM via `sm70_turbomind.prepare_mxfp4_linear` (`sm70_turbomind.py:229-253`) | serves the 48 `ssm_out` tensors at 4 bits with **zero new CUDA** | ~40 LOC py + ~50 LOC converter |

That last row is the difference between policy A and policy B — 0.5 GiB/GPU/token for ~90 LOC.

**One more seam this session found, which removes a whole class of converter work.** The fork's
`load_weights` expects the GDN input projection **already split on disk** as four separate tensors
(`qwen3_5.py:493-494, 506-507`):

```
("in_proj_qkvz", "in_proj_qkv", (0,1,2))   ("in_proj_qkvz", "in_proj_z", 3)
("in_proj_ba",   "in_proj_b",   0)         ("in_proj_ba",   "in_proj_a",  1)
```

and `create_qkvz_proj` builds `MergedColumnParallelLinear(output_sizes=[key_dim, key_dim, value_dim,
value_dim])` = `[2048, 2048, 6144, 6144]` — **plain concatenation, not head-interleaved**
(`qwen3_5.py:≈208-233`). Our GGUF is `attn_qkv` (5120x10240 = q2048|k2048|v6144) + `attn_gate`
(5120x6144 = z) + `ssm_beta`/`ssm_alpha` (48 each). That is a **1:1 name map with no row permutation and
no concatenation.** Confirmed against the incumbent's own checkpoint keys
(`model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed`, `...in_proj_z...`,
`...in_proj_a.weight`, `...in_proj_b.weight`). A head-interleaved layout would have forced a panel
permutation; it does not exist here.

---

## 2. Architecture in one line

```
GGUF (mixed: PXQ4 + q8_0 + q6_k + mxfp4 + f32)
   └─ offline converter (numpy, no vLLM, no gguf pkg) ──► HF-style safetensors dir
        ├─ *.pxq4_slabs  uint8 [N/64, K/32, 1088]      (panel bytes, verbatim)
        ├─ *.pxq4_anchor f16   [N/64, 64]              (panel headers)
        ├─ *.weight_packed / *.weight_scale            (ssm_out, mxfp4, their convention)
        ├─ *.weight      f16                           (norms, in_proj_a/b, embed)
        └─ config.json  {"quantization_config": {"quant_method": "pxq4", ...}}
                                    │
   pip package `pxq4-vllm` (entry_points: vllm.general_plugins) ─┘
        ├─ register_quantization_config("pxq4") → PXQ4Config
        ├─ ModelRegistry.register_model("Qwen3_5ForCausalLM", ...)
        └─ torch.ops.pxq4.{dequant, gemv}   ← our own .so, built beside vLLM, never rebuilding it
```

Everything else — GDN sm70 kernels, FLASH_ATTN_V100, FULL_AND_PIECEWISE graphs, paged KV, TP=4, the
MTP spec-decode model — is inherited untouched.

---

## 3. File-by-file

### 3.1 New — offline converter, `pxq4-convert/` (no vLLM import, runs on CPU anywhere)

| file | LOC | contents |
|---|---:|---|
| `gguf_raw.py` | 180 | Raw GGUF v3 header parser. **Reuse the scanner already written during the port** (`prototypes/pxq-vllm/tc.py`, `ggufscan.py`) — it already parses 866 tensors / 57 KVs of this exact file. Must NOT use the `gguf` PyPI package: `gguf.GGMLQuantizationType(252)` raises inside `GGUFReader._build_tensors`, killing the whole file open. |
| `dequant_ref.py` | 150 | Pure-numpy reference dequant for PXQ4 / q8_0 / q6_k / mxfp4. PXQ4: `panel = W + p*(128 + (K/32)*1088)`; `anchor[r] @ 2r`; slab `kb` at `+128 + kb*1088`; `scale_byte[r] @ slab+r` (lo nibble → elems 0-15, hi → 16-31); `code row r @ slab+64+16r`, byte b = `code(2b) | code(2b+1)<<4`; `w = f32(anchor[r]) * SUB16[s4] * BOOK[code]`. Book/sub read from the file's own `pxa.pxq6.book` / `pxa.pxq6.sub` KVs — **do not hard-code**, `PXA_PXQ6_BOOK`/`_SUB` can be overridden at build time. |
| `pxq4_encode.py` | 140 | numpy PXQ4 **encoder** (row absmax → fp16 anchor; per-16 SUB16 index search; per-element book search). Needed only for policy C (`lm_head`). Does not need bit-parity with the C quantizer — it produces a *new* tensor. |
| `mxfp4_repack.py` | 50 | ggml `block_mxfp4` (`ggml-common.h:181-187`: 1 byte E8M0 + 16 bytes nibbles per 32) → compressed-tensors `weight_packed` `[N,K/2]` uint8 + `weight_scale` `[N,K/32]`, the layout `unpack_mxfp4_weight` expects (`sm70_turbomind.py:118-126`). |
| `namemap.py` | 90 | GGUF → HF name map (§6.2). |
| `convert.py` | 200 | Driver: policy flags (`--ssm-out {fp16,mxfp4}`, `--lm-head {fp16,pxq4}`), streams tensors, writes safetensors shards + index + `config.json`. |
| **total** | **~810** | |

### 3.2 New — runtime package `pxq4_vllm/`

| file | LOC | contents |
|---|---:|---|
| `__init__.py` | 30 | `register()` entry point: `register_quantization_config("pxq4")(PXQ4Config)`, `ModelRegistry.register_model("Qwen3_5ForCausalLM", "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM")`, `import pxq4_vllm._C`. |
| `config.py` | 170 | `PXQ4Config` (§4.1). |
| `linear.py` | 150 | `PXQ4LinearMethod` (§4.2). |
| `mxfp4.py` | 60 | `SM70MXFP4LinearMethod`: `create_weights` declaring `weight_packed`/`weight_scale` with the same attrs compressed-tensors uses, `process_weights_after_loading` → `sm70_turbomind.prepare_mxfp4_linear(layer)`, `apply` → `sm70_turbomind.apply_prepared_linear`. Pure delegation. |
| `ops.py` | 40 | thin wrappers + `torch.library.register_fake` metas. |
| `csrc/pxq4_pxa.cuh` | ~500 **copied** + 60 edited | vendored slice of `ggml/src/ggml-cuda/pxa/pxq6.cuh` (§5). |
| `csrc/pxq4_ops.cu` | 300 | torch bindings, `TORCH_LIBRARY(pxq4, ...)`. |
| `setup.py` / `CMakeLists.txt` | 90 | §5.3. |
| **total** | **~1,400** (of which ~500 verbatim) | |

### 3.3 New — tests, `tests/`

`test_dequant_ref.py` (CPU numpy vs `pxa_pxq_dequant_2d`), `test_shard_equiv.py` (pure numpy, no GPU),
`test_cuda_dequant.py`, `test_gemv.py`, `test_layer.py`. ~220 LOC.

### 3.4 Existing files to patch

**None in `/opt/1Cat-vLLM`. None in `/path/to/engine-repo`.** Both trees are read-only in this design.
The only "patch" is deployment-side: the serving container must `pip install pxq4-vllm` (or mount it on
`PYTHONPATH`) and pass `--quantization pxq4` is *not even needed* — `override_quantization_method`
self-selects from `config.json`.

---

## 4. The Python side

### 4.1 `PXQ4Config`

```python
@register_quantization_config("pxq4")
class PXQ4Config(QuantizationConfig):
    def __init__(self, ignore, layer_map, quantize_lm_head):
        self.ignore = ignore              # MUST contain "linear_attn.in_proj_b"/"in_proj_a"  (see below)
        self.layer_map = layer_map        # suffix -> "pxq4" | "mxfp4" | "fp16"
        self.quantize_lm_head = quantize_lm_head

    def get_name(self): return "pxq4"                              # base_config.py:78, instance method
    def get_supported_act_dtypes(self): return [torch.half]        # :83  — fp16 ONLY, no bf16 on Volta
    @classmethod
    def get_min_capability(cls): return 70                         # :88  CLASSMETHOD (easy to get wrong)
    @staticmethod
    def get_config_filenames(): return []                          # :99
    @classmethod
    def from_config(cls, config): ...                              # :105
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        return "pxq4" if hf_quant_cfg.get("quant_method") == "pxq4" else None   # :111-130
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, ParallelLMHead):
            return PXQ4LinearMethod(self) if self.quantize_lm_head else UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            return None                                            # vocab_parallel_embedding.py:479-482
        if isinstance(layer, LinearBase):
            kind = self._kind_for(prefix)
            if kind == "pxq4":  return PXQ4LinearMethod(self)
            if kind == "mxfp4": return SM70MXFP4LinearMethod(self)
            return UnquantizedLinearMethod()   # NEVER None for a LinearBase — linear.py:492-495 raises
        return None
```

Capability gate: `get_min_capability() == 70` is checked once at `config/vllm.py:611-621`. The fork
already returns 70 from `awq_marlin.py:265-266` and `compressed_tensors.py:108-110`, so 70 is an accepted
value in this tree. We return it unconditionally and additionally refuse to load if
`current_platform.get_device_capability() != (7,0)` is false-y in a way that matters — actually we do
**not** refuse: the dequant path is arch-agnostic; only the fused GEMV is Volta-tuned. Declaring 70 lets
it run on anything ≥70.

**`ignore` is load-bearing, not cosmetic.** `_uses_split_gdn_input_projections(quant_config)`
(`qwen3_5.py:127-157`) introspects `modules_to_not_convert` / `ignored_layers` / `ignore` / `config["ignore"]`
for `linear_attn.in_proj_a` or `linear_attn.in_proj_b` (`:154-155`). If it returns False,
`create_qkvz_proj` appends `[num_v_heads, num_v_heads]` = `[48, 48]` to the fused projection
(`qwen3_5.py:≈216-218`), giving a 16,480-row fused param (16480/64 = 257.5 panels — **not panel-aligned
at all**) whose shard 4 is 12 rows/rank at TP=4. The whole design collapses. Our emitted `config.json`
therefore lists `linear_attn.in_proj_a` / `linear_attn.in_proj_b` in `ignore`, exactly as the incumbent
AWQ config does, and `get_quant_method` returns `UnquantizedLinearMethod()` for `in_proj_ba`.

### 4.2 `PXQ4LinearMethod`

```python
def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                   input_size, output_size, params_dtype, **extra):
    N = sum(output_partition_sizes); K = input_size_per_partition
    assert N % 64 == 0, f"pxq4: {N} output rows/partition is not a whole number of 64-row panels"
    assert K % 32 == 0, f"pxq4: {K} input cols/partition is not a whole number of 32-col slabs"
    assert params_dtype == torch.float16
    slabs  = Parameter(torch.empty(N//64, K//32, 1088, dtype=torch.uint8), requires_grad=False)
    anchor = Parameter(torch.empty(N//64, 64,          dtype=torch.float16), requires_grad=False)
    set_weight_attrs(slabs,  {"output_dim":0, "input_dim":1,
                              "packed_dim":0, "packed_factor":64, **extra})
    set_weight_attrs(anchor, {"output_dim":0,
                              "packed_dim":0, "packed_factor":64, **extra})
    layer.register_parameter("pxq4_slabs", slabs); layer.register_parameter("pxq4_anchor", anchor)
    layer.pxq4_N = int(N); layer.pxq4_K = int(K)     # python ints — apply() must never sync
```

**Why this is the whole sharding implementation.** vLLM has no sharder API; the *layer's* loader narrows
by declared attributes.

- **Column-parallel** (`ColumnParallelLinear.weight_loader`, `linear.py:775-778`) narrows `output_dim`
  only. `slabs.narrow(0, off/64, size/64)` = whole panels, a contiguous byte range. `anchor.narrow(0, …)`
  = the matching headers. `packed_dim == output_dim` triggers the `//packed_factor` divide at
  `linear.py:1053-1058`.
- **Row-parallel** (`RowParallelLinear.weight_loader`, `linear.py:1728-1761`) narrows `input_dim` only and
  **ignores `packed_factor`** — verified. `slabs.narrow(1, kb0, kbn)` = the slab subrange of every panel.
  The anchor has **no `input_dim`**, so it falls through to a full copy (assert at `linear.py:1760`) —
  which *is* the "duplicate the 128 B header on a K-split" step the format requires. Free.
- **Merged column-parallel with a tuple shard id** — the GDN `in_proj_qkv` case. Read this session at
  `linear.py:968-1035`: the tuple branch walks `output_sizes[first:last+1]`, divides both offset and size
  by `packed_factor` when `packed_dim == output_dim` (`:1013-1018`), narrows the loaded tensor, and
  recurses; the per-shard branch computes `shard_offset = sum(output_sizes[:id]) // tp_size` then
  `round(… // packed_factor)` (`:1053-1058`). For `[2048,2048,6144,6144]` at TP=4 every intermediate is
  exact: source offsets 0/2048/4096/10240 → 0/32/64/160 panels; dest offsets 0/512/1024/2560 → 0/8/16/40
  panels; sizes 512/512/1536/1536 → 8/8/24/24 panels. **No truncation anywhere.**

Use the **v1** loader path (do not add `"pxq4"` to `WEIGHT_LOADER_V2_SUPPORTED`, `linear.py:193-210`).

The silent failure mode: `round(x // packed_factor)` truncates a misaligned offset without raising
(`linear.py:1053-1058`, `parameter.py:606-609`), producing a well-formed *wrong* slice and a model that
loads cleanly with subtly wrong logits. The two asserts above are the only defence and are mandatory.

```python
def process_weights_after_loading(self, layer):
    layer.pxq4_slabs.data  = layer.pxq4_slabs.data.contiguous()
    layer.pxq4_anchor.data = layer.pxq4_anchor.data.contiguous()
    layer._sm70_f16_forbidden = True     # defensive: linear.py:57-58
    # NEVER set _sm70_f16_prepared (linear.py:62-63) — that is the fp16 dense fast path's key and it is
    # checked BEFORE quant_method.apply() at linear.py:425-427 and :805.

def apply(self, layer, x, bias=None):
    xf = x.reshape(-1, x.shape[-1])
    M = xf.shape[0]
    out = torch.empty((M, layer.pxq4_N), dtype=x.dtype, device=x.device)
    if M >= PXQ4_GEMM_M_THRESHOLD:                       # default 32, env-overridable
        w = torch.ops.pxq4.dequant(layer.pxq4_slabs, layer.pxq4_anchor,
                                   layer.pxq4_N, layer.pxq4_K)      # [N,K] fp16
        torch.mm(xf, w.t(), out=out)
    else:
        torch.ops.pxq4.gemv(out, xf, layer.pxq4_slabs, layer.pxq4_anchor,
                            layer.pxq4_N, layer.pxq4_K)
    if bias is not None: out.add_(bias)
    return out.reshape(x.shape[:-1] + (layer.pxq4_N,))
```

The `M` branch is host-side Python, which is **graph-capture safe** because capture is per-shape: a
captured decode graph always took the `gemv` arm at capture time. `torch.empty` inside `apply` is fine —
under capture it is serviced from the graph's private pool, exactly as
`sm70_turbomind.apply_prepared_linear` does (`sm70_turbomind.py:298-302`).

Note `_mark_default_sm70_dense_modules` (`qwen3_5.py:169-179`) sets `_sm70_f16_force_enable = True` on
every module whose last name component is `qkv_proj` or `out_proj` (`:160-166`). That flag is inert for
us because `_maybe_sm70_dense_forward` also requires `_sm70_f16_prepared`, which only
`UnquantizedLinearMethod.process_weights_after_loading` sets (`linear.py:56-96`, `:408`). Under policy A
`ssm_out` *is* unquantized fp16 and *will* take that fast path — which is free speed, not a hazard.

---

## 5. The CUDA side

### 5.1 Exactly two kernels, and why

`ggml/src/ggml-cuda/pxa/pxq6.cuh` is 3,601 lines and contains **10** `ggml_`/`GGML_` tokens, 4 of them in
comments and 6 in one host-side type mapper (`pxa_pxq_fmt`, `pxq6.cuh:3335-3345`). The `__global__`
kernels take `const uint8_t* W, const half* A, float* C`, plain ints, and a caller-supplied
`cudaStream_t` — **no ggml types at all**. This is a vendor-and-wrap, not a rewrite.
(Reminder: the id-252 kernels live in **`pxq6.cuh`**; `pxq4.cuh` documents the RETIRED id-250 format.)

1. **`pxq4::dequant`** — port of `k_pxq6_dequant_matrix`. `(slabs, anchor, N, K) -> half[N,K]`.
   Serves prefill (→ `torch.mm`/cuBLAS HMMA), the stage-2 fallback, and every correctness test.
2. **`pxq4::gemv`** — port of the `k_pxq6_mmv*` family. `(out[M,N] half, x[M,K] half, slabs, anchor, N, K)`
   for small `M`. **This is the only kernel that matters for the thesis**: it is the one that reads 4.25
   bpw instead of 16. One real edit: the ggml mmv consumes **fp32** activations and emits fp32
   (`pxq6.cuh:635, :930, :968`); vLLM hands us fp16. ~6 LOC.

**Deliberately NOT ported:**
- `k_pxq6_gemm_grouped` (the fused scalar `__hfma2` tile, `pxq6.cuh:2517-2625`). Our own note at
  `ggml-cuda.cu:4436-4444` measured it at **−18.6% on sm_70** versus coalesced-dequant + cuBLAS. The
  winning prefill shape keeps cuBLAS.
- The WMMA twin (`pxq6.cuh:2893-3218`) — wired only into the MoE driver (`ggml-cuda.cu:5056`), and this
  model has no MoE.
- `pxq-mmvq.cuh` — genuinely non-portable: it lives inside ggml's `mmvq-templates.cuh`, keys on
  `ggml_type` template parameters, and consumes `block_q8_1`.
- `pxq6_ksplit_workspace` (`pxq6.cuh:2480-2494`) — raw `cudaMalloc`, declines under stream capture.
  Omitting it is bit-identical, costs only occupancy on very large K. If it is ever needed, preallocate
  in `process_weights_after_loading`.

Shared-memory check against real TP=4 shapes (INFERENCE from shape arithmetic, not measured): the mmv
path stages `x` in dynamic smem capped at 46 KB (`ggml-cuda.cu:4262`). `ffn_down` K=4352 → 17.4 KB;
`ffn_gate`/`up` K=5120 → 20.5 KB. Fits. TP=2 also fits (`ffn_down` 34.8 KB). TP=1 (69.6 KB) does not and
would need the S-split path — irrelevant for the DGX, **relevant for the 2x V100 Unraid box only if
someone tries TP=1 there.**

### 5.2 The vendored slice

Copy into `csrc/pxq4_pxa.cuh`: the `PXQ6_*` geometry constants (`ggml-pxq6-tables.h:21-27`), the book/sub
`__device__` tables (`:33-44`, `pxq6.cuh:79`), the panel/slab addressing helpers (`pxq6.cuh:520-526`,
`:703-704`), the per-16 dequant expression (`pxq6.cuh:326-331`), `k_pxq6_dequant_matrix`, and the mmv
kernel. Edits: strip the `hq` (PXQ4HQ) branches, strip the expert dimension (`e*panels`) to a single
matrix, change the mmv activation type to `half`, replace the host dispatcher with a plain
`cudaStream_t` entry point. **The book/sub tables must be loaded from the checkpoint, not baked in** —
emit them as a `[16]` fp32 buffer per model and `cudaMemcpyToSymbol` once, or pass as kernel args, so a
model built with `PXA_PXQ6_BOOK` overridden still decodes correctly.

### 5.3 Build — beside vLLM, never rebuilding it

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
  name="pxq4-vllm", packages=["pxq4_vllm"],
  ext_modules=[CUDAExtension(
      name="pxq4_vllm._C",
      sources=["csrc/pxq4_ops.cu"],
      extra_compile_args={"cxx": ["-O3", "-std=c++17"],
                          "nvcc": ["-O3", "-std=c++17",
                                   "-gencode", "arch=compute_70,code=sm_70",
                                   "--expt-relaxed-constexpr"]})],
  cmdclass={"build_ext": BuildExtension},
  entry_points={"vllm.general_plugins": ["pxq4 = pxq4_vllm:register"]},
)
```

Torch registration in `pxq4_ops.cu`:

```cpp
TORCH_LIBRARY(pxq4, m) {
  m.def("dequant(Tensor slabs, Tensor anchor, int n, int k) -> Tensor");
  m.def("gemv(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor, int n, int k) -> ()");
}
TORCH_LIBRARY_IMPL(pxq4, CUDA, m) { m.impl("dequant", &pxq4_dequant); m.impl("gemv", &pxq4_gemv); }
```

**The thing that gets forgotten and breaks CUDA-graph capture is not the output allocation — it is the
`Tensor(a!) out` alias annotation and the `register_fake` meta.** Without a meta kernel, torch.compile
tracing (which `FULL_AND_PIECEWISE` runs before capture) fails *before* capture is ever attempted:

```python
@torch.library.register_fake("pxq4::dequant")
def _(slabs, anchor, n, k): return torch.empty((n, k), dtype=torch.float16, device=slabs.device)
@torch.library.register_fake("pxq4::gemv")
def _(out, x, slabs, anchor, n, k): return None
```

Also add `vllm::pxq4_gemv` to nothing — we do **not** need a `splitting_ops` entry; a leaf custom op
inside a piecewise region is captured normally.

Build environment, from this session's recon: `nvcc` 12.8, gcc, cmake, ninja and `torch 2.10.0+cu128`
are present in the container, and `site-packages/vllm` is a **copied** install (not editable-linked to
`/opt/1Cat-vLLM`, so edits there are inert at runtime — another reason the plugin is the right seam).
Two operational constraints: the production container's overlay is **100% full, 0 bytes available**, so
build in a *fresh* container with a volume under `/path/to/models`; and never write to `/` or host `/tmp` on
the DGX.

---

## 6. Weight loading

### 6.1 Option (b): offline converter. The other two options are dead or worse.

Option (a), vLLM's GGUF loader, is dead — **verified, not inferred**: `gguf.GGMLQuantizationType(252)`
raises `ValueError` inside `GGUFReader._build_tensors`, which runs over the whole tensor table in the
constructor. One PXQ4 tensor kills the file open before any tensor is yielded. Fixing it means forking
the upstream `gguf` PyPI package *plus* `quantization/gguf.py`'s five type sets *plus* vLLM's vendored
ggml `_custom_ops` — and still landing on a sharder that slices rows assuming per-row-contiguous blocks,
which panel interleave violates (`pxq-cpu.h:5-9`; ggml's own `to_float` is NULL for the same reason,
`ggml.c:1407-1414`). Option (c), teaching `llama-quantize` to emit safetensors, duplicates the quantizer
forever.

### 6.2 The five types and what happens to each

| ggml type | tensors | bytes | policy A | policy B/C | vLLM param |
|---|---:|---:|---|---|---|
| PXQ4 (252) | 325 | 11.392 GiB | **native**, bytes copied verbatim | same | `pxq4_slabs` + `pxq4_anchor` |
| Q8_0 (8) `attn_k`,`attn_v` | 34 | 0.176 | dequant → fp16 | (C) re-encode → PXQ4 | `weight` / `pxq4_*` |
| Q8_0 (8) `output` (lm_head) | 1 | 1.258 | dequant → fp16 | **(C) encode → PXQ4** ← the big lever | `pxq4_*` |
| Q8_0 (8) `ssm_alpha`,`ssm_beta`,`nextn.eh_proj` | 97 | 0.087 | dequant → fp16 | same | `weight` (in `ignore`) |
| Q6_K (14) `token_embd` | 1 | 0.971 | dequant → fp16 | same (not read at decode) | `weight` |
| **MXFP4 (39) `ssm_out`** | 48 | 0.747 | dequant → fp16 (+0.52 GiB/GPU) | **repack → their sm70 MXFP4 kernel** | `weight_packed`+`weight_scale` |
| F32 (0) norms/conv1d/A/dt | 360 | 0.010 | → fp16 (norms stay f32 where vLLM wants f32) | same | `weight` |

`ssm_out` being MXFP4 rather than PXQ4 is not an accident to be fixed: `ssm_out` is simply absent from
the backbone map's promote list (`pxa.pxq.backbone_map` = `attn_q,attn_qkv,attn_output,attn_gate_ch,
shexp,ffn_dense=tier+1; attn_k,attn_v=q8_0; attn_gate_head=f16; token_embd=q6_k; output=q8_0`), so it
fell through to the file's base tier.

Geometry gate: **all 325 PXQ4 tensors pass `rows%64 && K%32`**, at TP=1, 2 and 4, on their natural axis.
Six shapes only: 5120x6144 (48), 5120x10240 (48), 5120x12288 (17), 5120x17408 (130), 6144x5120 (17),
17408x5120 (65). `bytes = (rows/64)*(128 + (K/32)*1088)` reproduces all six on-disk sizes to the byte,
with no inter-tensor padding.

### 6.3 Name map (authoritative — taken from the incumbent checkpoint's own keys this session)

Prefix depends on the registered architecture. With `Qwen3_5ForCausalLM` (§1 seam 2) it is
`model.layers.N.`; with `Qwen3_5ForConditionalGeneration` it is `model.language_model.layers.N.`.

```
token_embd.weight                 -> model.embed_tokens.weight            (f16)
output.weight                     -> lm_head.weight | lm_head.pxq4_*      (fp16 | PXQ4, policy)
output_norm.weight                -> model.norm.weight
blk.N.attn_norm.weight            -> ...layers.N.input_layernorm.weight
blk.N.post_attention_norm.weight  -> ...layers.N.post_attention_layernorm.weight
# full-attn layers (3,7,...,63, +64)
blk.N.attn_q/k/v.weight           -> ...self_attn.{q,k,v}_proj.*
blk.N.attn_output.weight          -> ...self_attn.o_proj.*                (ROW-parallel, K=6144)
blk.N.attn_q_norm/attn_k_norm     -> ...self_attn.{q,k}_norm.weight
# GDN layers (il%4 != 3)
blk.N.attn_qkv.weight             -> ...linear_attn.in_proj_qkv.*         (10240 = q|k|v, NO permutation)
blk.N.attn_gate.weight            -> ...linear_attn.in_proj_z.*           (6144 = z)
blk.N.ssm_beta.weight             -> ...linear_attn.in_proj_b.weight      (f16, in `ignore`)
blk.N.ssm_alpha.weight            -> ...linear_attn.in_proj_a.weight      (f16, in `ignore`)
blk.N.ssm_out.weight              -> ...linear_attn.out_proj.*            (ROW-parallel, K=6144)
blk.N.ssm_conv1d.weight           -> ...linear_attn.conv1d.weight
blk.N.ssm_a                       -> ...linear_attn.A_log
blk.N.ssm_dt.bias                 -> ...linear_attn.dt_bias
blk.N.ssm_norm.weight             -> ...linear_attn.norm.weight
# MLP, all 65 blocks
blk.N.ffn_gate/up.weight          -> ...mlp.{gate,up}_proj.*   (COL, merged into gate_up_proj)
blk.N.ffn_down.weight             -> ...mlp.down_proj.*        (ROW, K=17408)
# blk.64 = MTP. `load_weights` skips names starting with "mtp." (qwen3_5.py:≈536-537); emit under
# `mtp.` only when the qwen3_5_mtp speculator is enabled, otherwise drop blk.64 in v1.
```

`attn_q` rows = 12288 = 2 x (24 x 256): it is **gate-fused** (`llama-build-context.cpp:2003-2007` takes
Q at `ggml_view_3d(..., 2*row_size, ..., 0)` and gate at `offset=row_size`, i.e. per-head interleave
`[q_h | gate_h]`, matching `qwen3_next.py:565-567`). Head pitch 256 rows = 4 panels, so the converter can
de-interleave into `q_proj` (6144) and a separate gate at **panel granularity, pure memcpy** — or leave
it fused if the vLLM module expects the fused form. Decide by reading the incumbent's
`self_attn.q_proj.weight_shape` value; the AWQ index shows `q_proj/k_proj/v_proj/o_proj` as separate
tensors, so the converter **must** de-interleave. (Its 12288 rows split as 6144+6144, both 96 panels.)

### 6.4 Emitted `config.json` (delta from the incumbent's)

```jsonc
{
  "architectures": ["Qwen3_5ForCausalLM"],       // needs our register_model; else ...ForConditionalGeneration
  "model_type": "qwen3_5",
  "dtype": "float16",                             // NOT bfloat16 — Volta has no bf16
  "quantization_config": {
    "quant_method": "pxq4",
    "pxq": {"backbone_rev": 2, "tier": "core", "version": 1,
            "book": [...16 f32...], "sub": [...16 f32...]},
    "ignore": ["linear_attn.in_proj_a", "linear_attn.in_proj_b", "lm_head"],   // lm_head only in policy A/B
    "layer_map": {"mlp.gate_proj":"pxq4", "mlp.up_proj":"pxq4", "mlp.down_proj":"pxq4",
                  "self_attn.q_proj":"pxq4", "self_attn.o_proj":"pxq4",
                  "self_attn.k_proj":"fp16", "self_attn.v_proj":"fp16",
                  "linear_attn.in_proj_qkv":"pxq4", "linear_attn.in_proj_z":"pxq4",
                  "linear_attn.out_proj":"mxfp4"}
  }
}
```

Everything else (`vision_config`, rope, ssm dims, tokenizer files) is copied from the incumbent's
directory — the two models are the same base, and our GGUF KVs agree with it (`block_count 65`,
`embedding_length 5120`, `feed_forward_length 17408`, `head_count 24/4`, `key_length/value_length 256`,
`ssm.group_count 16`, `ssm.inner_size 6144`, `ssm.time_step_rank 48`, `full_attention_interval 4`,
`nextn_predict_layers 1`).

---

## 7. MoE

**There is none, and the effort delta is zero.** Verified directly against the artifact by raw GGUF
header parse this session: 866 tensors, `general.architecture = qwen35`, **zero `*_exps` tensors and zero
expert KVs**; `ffn_gate`/`ffn_up`/`ffn_down` are dense on all 65 blocks. The brief's "40x256 routed
experts" belongs to **Sky-35B**, a different model. No `FusedMoEMethodBase`, no `RoutedExperts` path, no
dequant fallback, **0 GiB of fallback VRAM cost** — the line item does not exist.

If Sky-35B is targeted later, the seam is `FusedMoEMethodBase` (not `LinearMethodBase`), and the ggml
side is *already* built for it: PXQ4's addressing is natively `(expert, panel, K-block, row)`
(`pxq6.cuh:520-526`), `k_pxq6_gemm_grouped` (`:2517-2625`) is a grouped/MoE kernel with a tile map, and
the WMMA twin is wired to the MoE driver at `ggml-cuda.cu:5056`. The dense port above *removes* the
expert dimension; a MoE port would put it back rather than add anything new. A dequant fallback for a
hypothetical MoE would cost `n_experts * (gate+up+down bytes) * (16/4.25 - 1) / tp` — for a 40x256 layout
that is prohibitive (multiple GiB/GPU), so a fallback should never be the plan there; the grouped kernel
should be ported directly.

---

## 8. Staged plan — every stage has an offline correctness gate

A scheduling reality first: **stages S2+ need a GPU for unit tests.** The DGX lease is held by other
jobs and the Unraid V100s are serving live seats. S0/S1 gates are CPU-only and can start immediately;
S2-S5 need a small window (tens of MB, seconds) — not a benchmark, but not free either.

**S0 — fp16 reference checkpoint. No plugin, no CUDA, no PXQ4 at runtime.**
Converter dequantizes *everything* to fp16 and writes a plain HF directory.
- **Gate 0a (CPU, bit-exact):** numpy `dequant_ref.py` vs `pxa_deq_row_pxq6` via
  `pxa_pxq_dequant_2d` (`pxq-cpu.c:135-158`, public at `:219-225`) on every distinct shape.
  Must be **bit-exact in fp32** — the dequant is the parity-locked contract (`pxq-cpu.h:16-18`),
  unlike the GEMM kernels which are explicitly not bit-exact (fp16 MMA snap). This single test validates
  the layout reader, panel arithmetic and the book/sub tables — the bulk of the port's risk.
- **Gate 0b:** the fp16 checkpoint loads and generates coherent text under stock vLLM, proving the name
  map, the `Qwen3_5ForCausalLM` registration and the GDN split-projection config. ~27 GiB fp16 → 6.8
  GiB/GPU at TP=4; fits.
- **Gate 0c:** its logits on 32 fixed prompts match llama.cpp's PXQ4 logits to a stated tolerance.
  *This is the quality baseline for everything after.*

**S1 — plugin + sharding, dequant at load. Still no CUDA.**
`PXQ4LinearMethod.process_weights_after_loading` dequantizes the PXQ4 params to a plain fp16 `weight`
and `apply` is `F.linear`. Runtime footprint identical to S0; the *checkpoint* is now real PXQ4.
- **Gate 1a (CPU, no vLLM, no GPU):** `test_shard_equiv.py` — for every (tensor, TP∈{2,4}, rank), assert
  `dequant(narrow(param))` == `narrow(dequant(param))` **bit-exact in fp32**, for both the dim-0 (panel)
  and dim-1 (slab) narrows. This is the proof that the TP repack is a permutation and not a
  requantization. Note the format *could* be re-quantized per shard but must not be: `row0` seeds the
  deterministic tie-break `pxq_tie_take_hi` (`pxq6-quantize.inc.cpp:49`, used `:230`, warned `:416-418`),
  so re-quantizing a shard from row 0 yields **different bytes** for identical weights.
- **Gate 1b:** load at TP=4 and diff every layer's dequantized weight against S0's fp16 tensor —
  bit-exact. Catches every silent `//packed_factor` truncation.

**S2 — `pxq4::dequant` CUDA op.** Replace the numpy load-time dequant.
- **Gate 2a:** CUDA dequant vs numpy reference, **bit-exact fp32**, all six shapes plus a
  deliberately-misaligned shape that must assert.
- **Gate 2b:** end-to-end logits identical to S1.

**S3 — 4-bit resident + `pxq4::gemv`. The first stage that changes the memory picture.**
- **Gate 3a:** `gemv(x, W)` vs `x @ dequant(W).T` in fp32, max relative error under a stated bound
  (not bit-exact: fp32 accumulation order differs).
- **Gate 3b:** one full transformer block — instantiate the real vLLM modules on one GPU with the plugin,
  feed fixed random hidden states, compare PXQ4-served vs S0-fp16-served outputs. No model run, no
  scheduler, no KV cache.
- **Gate 3c:** full model, logits vs S0 on the 32 fixed prompts; same-top-token rate reported.
- **Gate 3d:** cudagraph capture succeeds with `FULL_AND_PIECEWISE` (this is where a missing
  `register_fake` or alias annotation shows up).

**S4 — policy B: `ssm_out` on their MXFP4 kernel.** −0.52 GiB/GPU/token.
- **Gate 4a:** `mxfp4_repack` + `mxfp4_sm70_prepare` + `mxfp4_gemm_sm70_out` vs numpy ggml-MXFP4 dequant
  → `mm`, relative-error bound. **This gate is where the "does ggml type-39 packing match their
  convention" question is answered**; if it fails, fall back to fp16 `ssm_out` (policy A cost) rather than
  double-quantizing.

**S5 — policy C: 4-bit `lm_head`.** −0.435 GiB/GPU/token, the largest single win.
- Source the head from the **incumbent's `lm_head.weight` (bf16, unquantized — it is in their `ignore`
  list)**, not from our q8_0 copy, to avoid a double quantization. Encode with `pxq4_encode.py`.
- **Gate 5a:** perplexity / same-top-token vs S0 on a held-out set. The backbone deliberately keeps
  `output` at q8_0 (`llama-quantize.cpp:1343-1345` region, `pxa.pxq.backbone_map`), so this is the one
  place the design knowingly departs from a decision our own quantizer made. If it fails the quality
  gate, policy B is the ceiling and the project lands at ~90 tok/s projected — i.e. **at parity with
  the incumbent, winning only on quality.**
- **Gate 5b:** `ParallelLMHead` sharding honours `packed_dim`/`packed_factor`. **Not verified.**
  `ParallelLMHead` uses `VocabParallelEmbedding`'s own loader with `shard_indices` and vocab padding, not
  the linear loader. 248,320/4 = 62,080 rows/rank = 970 panels, so the arithmetic is exact — but if that
  loader ignores `packed_factor`, attach an explicit `weight_loader` to the two params (~25 LOC).

**S6 — measurement.** Only now does anyone quote a tok/s number. Everything in §0 is a projection.

---

## 9. Risks, honestly

**Top 3.**

1. **The win is ~10%, only under policy C, and policy C is the riskiest stage.** §0's arithmetic is
   the whole business case: policy A is −15%, policy B is −3%, policy C is +10% and depends on
   4-bit-ing an LM head that our own backbone table deliberately protects at q8_0. If gate 5a fails we
   ship parity. Mitigation: run gate 5a *early* (it needs only the converter and llama.cpp, no vLLM),
   before committing to S3-S5. Kill criterion: if a 4-bit head costs more than [agreed] same-top-token
   points, stop at policy B and reframe the project as quality-at-parity.
2. **Numeric drift with no budget.** PXQ4 was quality-gated inside llama.cpp with its own attention,
   GDN and sampling. In vLLM the same weights meet FLASH_ATTN_V100, a different GDN implementation, fp16
   activations throughout, and a GEMV whose fp32 accumulation order differs from ggml's. The dequant is
   bit-exact (gate 0a/2a) but the *model* is not, and there is currently no agreed acceptance threshold.
   Mitigation: fix the threshold at S0 (gate 0c) before any kernel exists.
3. **`Qwen3_5ForCausalLM` may not stand alone.** `registry.py:560` exposes only
   `Qwen3_5ForConditionalGeneration`, whose `__init__` builds `Qwen3_VisionTransformer`
   **unconditionally** (`qwen3_5.py:843`) — `language_model_only` only zeroes multimodal input limits
   (`config/multimodal.py:315`), it does not skip construction. Our 2-line `register_model` of their
   in-tree `Qwen3_5ForCausalLM` (`qwen3_5.py:772`) is INFERENCE, not verified to load. Fallback: copy the
   333 vision tensors (0.858 GiB bf16 → fp16) out of the incumbent's safetensors into our checkpoint;
   costs ~0.21 GiB/GPU resident and **zero** decode-read (never touched on text-only requests), so the
   §0 table is unchanged either way. This is a cost, not a blocker.

**The rest.**

4. **CUDA-graph capture.** Missing `register_fake` / `Tensor(a!)` annotation fails tracing before capture
   (gate 3d). Also never port `pxq6_ksplit_workspace` (`pxq6.cuh:2480-2494`) — its raw `cudaMalloc`
   declines under capture.
5. **Silent shard truncation.** `round(x // packed_factor)` (`linear.py:1053-1058`,
   `parameter.py:606-609`) truncates without raising. The two asserts in `create_weights` and gate 1a/1b
   are the only defence. This is the failure mode that produces a model which loads cleanly and is
   subtly wrong.
6. **ggml MXFP4 ≠ their MXFP4 convention** (nibble order, E8M0 scale dtype). Unverified. Gate 4a decides;
   fallback is fp16 `ssm_out` and policy B becomes unreachable (ceiling drops to policy A).
7. **`_uses_split_gdn_input_projections` regression.** If our `ignore` list is ever dropped or renamed,
   the fused qkvzba projection becomes 16,480 rows (257.5 panels) and TP=4 puts 12 rows on shard 4. Add
   an explicit runtime assert in `PXQ4Config.__init__` that both `linear_attn.in_proj_a` and
   `in_proj_b` are present in `ignore`.
8. **Prefill scratch.** Dequant→cuBLAS materialises up to 44.6 MB fp16 per call at TP=4 (`ffn_gate`,
   4352x5120). Fine, but it makes prefill read 4.25 bpw *and* write+read 16 bpw. If prefill regresses
   versus the incumbent's TurboMind s884 path, that is expected and is what Option T exists for.
9. **The fork moves.** Pinned at `2ceb15066` (2026-08-15). `qwen3_5.py`'s `stacked_params_mapping` and
   `create_qkvz_proj` are private implementation detail; a rename breaks the converter's name map, not
   the kernels. Pin the fork commit in the package metadata and re-run gate 0b on every bump.
10. **Volta smem ceiling at TP=1** (§5.1): 69.6 KB needed vs 46 KB cap for `ffn_down` — TP=1 is
    unsupported without the S-split path. Document it; the 2x V100 Unraid box must use TP=2.

**Option T (phase 2, performance only, not in the LOC estimate).** Compile a variant of
`csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` with the PX16 book substituted into the register-side
converter, and drive it through `nvfp4_sm70_prepare`'s group-16 path with our folded
`anchor*SUB16` fp16 scales. That reaches real HMMA tensor cores for prefill and possibly for large-batch
decode. It needs their source tree at build time (public, Apache-2.0), and it is the only path that
makes prefill competitive. Scope it separately after S6 gives a real measurement.

---

## 10. LOC

| component | new | of which verbatim copy |
|---|---:|---:|
| offline converter | 810 | 0 (but ~180 reused from this study's scanner) |
| runtime python package | 450 | 0 |
| CUDA (vendored slice + shim) | 860 | ~500 |
| tests | 220 | 0 |
| **total** | **~2,340** | **~500** |
| **hand-written** | **~1,840** | |

Policy A alone (drop `pxq4_encode.py`, `mxfp4_repack.py`, `mxfp4.py`) is ~1,600 total / ~1,150
hand-written — and ships a 15% regression. The extra ~690 LOC *is* the project.

Patches to `/opt/1Cat-vLLM`: **0 files.** Patches to `/path/to/engine-repo`: **0 files.**

---

## 11. Not verified

- `Qwen3_5ForCausalLM` loading standalone via `register_model` (risk 3).
- ggml MXFP4 (type 39) packing vs the fork's `weight_packed`/`weight_scale` convention (risk 6).
- `ParallelLMHead` honouring `packed_dim`/`packed_factor` (gate 5b).
- Whether `attn_q`'s gate-fused 12288 rows must be de-interleaved into `q_proj` + a gate tensor, or
  whether the fork's `qwen3_5.py` expects the fused form. The AWQ index shows separate `q_proj`, which
  says de-interleave; not confirmed against the vLLM module's `output_sizes`.
- Whether `PXA_PXQ_KQW` / `ANCHOR_FIT` state is recorded in the GGUF (does not affect decode).
- Every tok/s figure in this document. **No GPU was run.**
