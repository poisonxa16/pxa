# 08 — PXQ4 as a vLLM quantization backend: minimal-risk design

Target: `1Cat-vLLM` fork @ `2ceb15066` (`v0.1.dev1+g2ceb15066.cu128`), 4x V100-SXM2-32GB (sm_70),
TP=4, `dtype=float16`. Artifact: `/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf`.
Slant: **minimal risk / fastest to first correct token.** No GPU runs were performed for this
document; every throughput number is a labelled PROJECTION.

Tags: **FACT** = read in source, file:line given. **INFERENCE** = derived from FACTs.
**ASSUMPTION** = not verified, flagged.

---

## 0. Verdict, and the one thing that must not be misread

**No blocker.** Every hard question is already answered by the recon: the format is pinned
(01), the kernels carry no ggml types and can be vendored (02), `register_quantization_config`
is a real out-of-tree hook (03), sm_70 is a first-class capability in this fork (04), the GGUF
loader path is dead and an offline converter replaces it (05), the artifact's type census is
measured (06), and TP=2/TP=4 sharding is byte-exact with no re-quantization (07).

**The thing that must not be misread: the minimal-risk v1 is a correctness milestone, not a
performance win, and at its most conservative it is a large performance *regression*.** The
brief's headline ("PXQ4 sharded 4 ways = 3.66 GiB/GPU vs their AWQ 4.64 → 110-120 tok/s") is
arithmetic on the PXQ4 tensors only. The measured census (06) says PXQ4 is 11.39 of the
artifact's 14.63 GiB; the other 3.24 GiB is Q6_K `token_embd`, Q8_0 `output`/`attn_k`/`attn_v`,
and MXFP4 `ssm_out`. A v1 that serves the non-PXQ4 tail as fp16 lands at **5.225 GiB/GPU at
TP=4** (PROJECTION, §9) — *above* the incumbent AWQ's 4.64. The 4-bit win only materialises when
`ssm_out` and `lm_head` are also served at 4 bits, which is phase 2.

So this design is explicitly staged, and the stages are ordered by *risk retired per unit of
work*, not by performance:

| stage | what runs | weights resident | risk retired |
|---|---|---|---|
| **S0** | converter emits an **all-fp16** safetensors checkpoint; **stock vLLM, no plugin** | 13.5 GiB/GPU fp16 | name map, shapes, dequant correctness, model wiring, GDN split |
| **S1** | plugin registered; PXQ4 params declared + TP-sharded; **dequant at load** (pure torch, no CUDA) | 13.5 GiB/GPU fp16 | registration, `get_quant_method` dispatch, panel/slab sharding, `_sm70_f16_prepared` hazard |
| **S2** | CUDA dequant op replaces the torch dequant; still dequant-at-load | 13.5 GiB/GPU fp16 | the vendored kernel, bit-exactly, against the CPU reference |
| **S3** | weights stay 4-bit resident; `k_pxq6_mmv` decode + dequant→`torch.mm` prefill | 5.225 GiB/GPU | first real bandwidth/VRAM win; CUDA-graph capture |
| **S4** | 4-bit `ssm_out` + 4-bit `lm_head`/`token_embd`; WMMA / gufuse prefill | 3.41 GiB/GPU | performance, phase 2 |

**S0 requires zero new CUDA and zero vLLM plugin code.** That is the fastest path to a first
correct token, and it is a checkpoint, not a throwaway: the converter it exercises is the same
converter S1-S4 use, with one flag flipped.

---

## 1. Architecture in one picture

```
OFFLINE (CPU, runs anywhere, no vLLM import)
  Qwen3.8-27B-PXQ4.gguf  ──►  pxq4_gguf2st.py  ──►  <outdir>/
    866 tensors, 5 ggml types      raw struct GGUF parser        model-*.safetensors
    {F32:360, Q8_0:132, Q6_K:1,    per-type handler              config.json  (quantization_config)
     MXFP4:48, PXQ4:325}           ggml→HF name map              pxq4_book.json (BOOK/SUB16 from KVs)
                                   --emit=fp16 | --emit=pxq4

RUNTIME (in-container, out-of-tree pip package `pxq4-vllm`)
  entry_points["vllm.general_plugins"]["pxq4"] = pxq4_vllm:register
        │  loaded at arg_utils.py:749, v1/engine/core.py:108, v1/worker/worker_base.py:247
        ▼
  @register_quantization_config("pxq4") class PXQ4Config      ── quantization/__init__.py:57
        │  get_quant_method(layer, prefix) → per-module dispatch
        ├─► PXQ4LinearMethod      (pxq4_slabs uint8 + pxq4_anchor fp16, panel/slab sharded)
        ├─► PXQ4MXFP4LinearMethod (phase 2; ssm_out)
        └─► UnquantizedLinearMethod()  (attn_k/v, in_proj_ba, lm_head, anything fp16)
        │
        ▼  torch.ops.pxq4.*   ←── pxq4_sm70_C.so   (our own TORCH_LIBRARY namespace,
                                   vendored slice of ggml/src/ggml-cuda/pxa/pxq6.cuh)
```

Nothing in `/opt/1Cat-vLLM` is patched. Nothing in `/path/to/engine-repo` is modified.

---

## 2. File-by-file manifest

### 2.1 New files — offline converter (no vLLM dependency)

| file | LOC | contents |
|---|---|---|
| `tools/pxq4_gguf/gguf_raw.py` | 130 | Raw GGUF v3 struct parser: magic/version/`n_tensors`/`n_kv`, KV table (all 13 GGUF value types incl. arrays), tensor table (name, n_dims, dims, **raw type id as an int — never an enum**), aligned data base. Derived from the working scanner already run against the artifact (`prototypes/pxq-vllm/ggufscan.py`). **This is the whole reason option (b) is cheap** — `gguf.GGMLQuantizationType(252)` raises `ValueError` inside `GGUFReader._build_tensors`, killing the entire file open (FACT, 05 §B1). |
| `tools/pxq4_gguf/pxq4_codec.py` | 120 | The PXQ4 (id 252) codec. `split_panels(buf,R,K) -> (anchor fp16 [R//64,64], slabs uint8 [R//64,K//32,1088])` — a pure byte de-interleave, the only transform the converter applies to PXQ4 tensors. `dequant(anchor,slabs) -> fp32 [R,K]` — the vectorised reference from 01 §6: `w = hdr[:,None,:,None] * eff * BOOK[nib]`, then `transpose(0,2,1,3).reshape(R,K)`. Constants `HDR=128`, `SLAB=1088`, `QK=32`, `BM=64`, `CODE_OFF=64` (FACT, `ggml/include/ggml-pxq6-tables.h:21-27`). BOOK/SUB16 are **read from the file's `pxa.pxq6.book` / `pxa.pxq6.sub` KVs** (FACT, `src/llama-quantize.cpp:1980-1983`; both confirmed present in the artifact, 06 §3), never hard-coded — `PXA_PXQ6_BOOK`/`PXA_PXQ6_SUB` can override at build time (`ggml-cuda/pxq6.cuh:288-302`). |
| `tools/pxq4_gguf/legacy_codec.py` | 170 | Dequantisers for the tail types: Q8_0 (id 8, 132 tensors), Q6_K (id 14, 1 tensor), MXFP4 (id 39, 48 tensors), F32/F16 passthrough. All standard ggml block formats, all row-contiguous — none of the panel machinery applies. |
| `tools/pxq4_gguf/namemap.py` | 220 | ggml→HF/vLLM name mapping + the fusion/split policy (§5.3). Ships a `--check-against <awq_twin_dir>` mode that diffs the produced key set against `/path/to/hf/<reference-hf-model>`'s key set and **fails the build on any mismatch**. This is the cheapest possible gate on the fiddliest part of the job. |
| `tools/pxq4_gguf/convert.py` | 200 | CLI driver. `--emit fp16` (stage S0) or `--emit pxq4` (S1+); `--tail-policy fp16|native`; writes sharded safetensors, `config.json` (copied from the AWQ twin with `quantization_config` replaced), and `pxq4_book.json`. |
| `tools/pxq4_gguf/verify.py` | 150 | The correctness gates of §7: dequant-vs-CPU-reference bit-exactness, shard-then-dequant vs dequant-then-shard bit-exactness, geometry asserts, byte-size reproduction `(R/64)*(128+(K/32)*1088)`. |

### 2.2 New files — runtime pip package `pxq4-vllm`

| file | LOC | contents |
|---|---|---|
| `pxq4_vllm/__init__.py` | 20 | `def register(): from . import config  # noqa` — the entry point callable. Nothing else at import time. |
| `pxq4_vllm/config.py` | 260 | `@register_quantization_config("pxq4") class PXQ4Config(QuantizationConfig)` — §3.1. |
| `pxq4_vllm/linear.py` | 300 | `PXQ4LinearMethod(LinearMethodBase)` — §3.2. Includes the pure-torch dequant used by S1 and as the permanent CPU/meta fallback. |
| `pxq4_vllm/mxfp4.py` | 90 | Phase 2 only: `PXQ4MXFP4LinearMethod`, delegating to the fork's already-built `torch.ops._C.mxfp4_sm70_prepare` / `mxfp4_gemm_sm70_out` (FACT, `csrc/torch_bindings.cpp:198-200`; `sm70_turbomind.py:229-253`). |
| `pxq4_vllm/ops.py` | 100 | `torch.ops.pxq4.*` wrappers + `@register_fake` meta kernels for **every** op. §4.4. |
| `pxq4_vllm/csrc/pxq4_pxa.cuh` | 560 | **Vendored verbatim** from `/path/to/engine-repo/ggml/src/ggml-cuda/pxa/pxq6.cuh` (~500 lines) + ~60 lines of edits. Slice list in §4.2. |
| `pxq4_vllm/csrc/pxq4_sm70.cu` | 320 | Torch shim: `TORCH_LIBRARY(pxq4, m)`, dtype/shape/device `TORCH_CHECK`s, stream plumbing, launch config. §4.3. |
| `setup.py` | 70 | `CUDAExtension` with `-gencode arch=compute_70,code=sm_70`. §4.5. |
| `pyproject.toml` | 30 | `[project.entry-points."vllm.general_plugins"] pxq4 = "pxq4_vllm:register"`. |

### 2.3 New files — tests

| file | LOC | contents |
|---|---|---|
| `tests/test_dequant_bitexact.py` | 120 | GPU dequant vs `pxa_pxq_dequant_2d` CPU reference, `torch.equal` in fp32. |
| `tests/test_shard_permutation.py` | 90 | shard→dequant vs dequant→shard, both axes, TP=2 and TP=4. |
| `tests/test_linear_layer.py` | 110 | One `RowParallelLinear` / one `MergedColumnParallelLinear`, real artifact tensors, vs fp16 golden. |
| `tests/cpuref/` | 90 | ctypes binding of `pxa_pxq_dequant_2d`; a 40-line standalone `Makefile` for `ggml/src/pxq-cpu.c`. |

### 2.4 Existing files to patch

**In `/opt/1Cat-vLLM` (Kewaii's fork): NONE.** This is a design requirement, not a preference —
`site-packages/vllm` is a *copied* install, not an editable link to `/opt/1Cat-vLLM`, so edits
there are inert at runtime anyway (FACT, 05 §A2).

**In `/path/to/engine-repo` (our llama.cpp tree): NONE.** Read-only by rule. The vendored kernel
slice is a copy, with the provenance canary comment (`pxq6.cuh:1-3`) preserved and a
`// VENDORED FROM mgv-wt@acf8f245 ggml/src/ggml-cuda/pxa/pxq6.cuh:<range>` banner per block so a
future re-sync is mechanical.

**Optional, phase 2 only:** if `ssm_out` is to be *re-quantized* from MXFP4 to PXQ4, that needs
one exported entry point around `pxq6_quantize_expert` (`src/pxq6-quantize.inc.cpp:287-289`,
currently `static` in an `.inc.cpp` — FACT, 07 §4). Do not attempt this in v1; the cheaper
phase-2 route is the fork's existing MXFP4 sm70 GEMM.

---

## 3. The Python side

### 3.1 `PXQ4Config`

Six abstract members (FACT, `base_config.py:70-163`), plus `override_quantization_method`
(`:111-130`) which is how the checkpoint self-selects.

```python
@register_quantization_config("pxq4")
class PXQ4Config(QuantizationConfig):
    def __init__(self, tensor_types: dict[str, str], book: list[float], sub: list[float],
                 backbone_rev: int, backbone_map: str, ignore: list[str],
                 runtime: str = "native"):          # "native" | "dequant"  (S1/S2 use "dequant")
        super().__init__()                          # sets self.packed_modules_mapping (base_config.py:76)
        self.tensor_types = tensor_types            # module-prefix suffix -> "pxq4"|"mxfp4"|"fp16"
        self.book, self.sub = book, sub             # 16+16 fp32, from pxa.pxq6.book / pxa.pxq6.sub
        self.backbone_rev, self.backbone_map = backbone_rev, backbone_map
        self.ignore = ignore                        # MUST contain linear_attn.in_proj_b / in_proj_a
        self.ignored_layers = ignore                # alias: qwen3_5.py:144 scans BOTH attribute names
        self.runtime = runtime

    def get_name(self):                 return "pxq4"                    # base_config.py:78, instance
    def get_supported_act_dtypes(self): return [torch.half]              # :83, instance. V100: no bf16
    @classmethod
    def get_min_capability(cls):        return 70                        # :88, CLASSMETHOD
    @staticmethod
    def get_config_filenames():         return []                        # :99, STATICMETHOD
    @classmethod
    def from_config(cls, config):       return cls(**_parse(config))     # :105, gets the raw dict
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        return "pxq4" if hf_quant_cfg.get("quant_method") == "pxq4" else None      # :111-130
    def get_quant_method(self, layer, prefix):                            # :150
        ...
```

**Capability gating.** `get_min_capability() -> 70`, unconditional. The single gate is
`vllm/config/vllm.py:611-621` (FACT, 04 §5): `capability < get_min_capability()` raises, then
`get_supported_act_dtypes()` is checked against `model_config.dtype`. This fork already returns
70 elsewhere — `awq_marlin.py:265-266` and `compressed_tensors.py:108-110` do it unconditionally.
Returning `[torch.half]` only makes `--dtype bfloat16` fail loudly, which is what we want on
Volta. We do **not** read `VLLM_SM70_QUANT_BACKEND`: it is a three-string `Literal`
(`envs.py:789-790`, `:115`) that raises on unknown values and is not a dispatch table
(FACT, 04 §1-2).

**Registration and load order.** `register_quantization_config` appends to the runtime
`QUANTIZATION_METHODS` list (`quantization/__init__.py:92`) and to
`current_platform.supported_quantization` (`:94-95`), and custom entries **override** built-ins
in `get_quantization_config` (`:279`). `load_general_plugins()` runs in the engine
(`arg_utils.py:749`, before `ModelConfig` is built), in the engine core (`v1/engine/core.py:108`)
and in **every TP worker** (`v1/worker/worker_base.py:247`) — so all 4 ranks re-register
independently. Selection: `ModelConfig._verify_quantization` (`config/model.py:1002-1080`) probes
custom names **first** (`:1038-1044`), so `config.json` → `"quantization_config": {"quant_method":
"pxq4", ...}` selects us with no CLI flag.

**Per-module dispatch — the mixed-type answer.** `get_quant_method` is called once per module
with the full dotted prefix; that *is* the per-tensor hook, and it is exactly how
compressed-tensors serves a mixed checkpoint today (`compressed_tensors.py:146-186`).

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, LinearBase):
        kind = self._lookup(prefix)                     # suffix match against tensor_types
        if kind == "pxq4":  return PXQ4LinearMethod(self)
        if kind == "mxfp4": return PXQ4MXFP4LinearMethod(self)     # phase 2
        return UnquantizedLinearMethod()                # NEVER None for a LinearBase
    return None                                         # embeddings -> UnquantizedEmbeddingMethod
```

Returning `None` for a `LinearBase` raises `ValueError("All linear layers should support quant
method.")` (FACT, `linear.py:492-495`); `None` is correct only for embeddings
(`vocab_parallel_embedding.py:479-482`).

**Mandatory: the GDN ignore list.** `_uses_split_gdn_input_projections(quant_config)`
(FACT, `qwen3_5.py:127-157`) scans `modules_to_not_convert` / `ignored_layers` / `ignore` /
`config["ignore"]` for entries equal to `linear_attn`, ending `.linear_attn`, or containing
`linear_attn.in_proj_a` / `linear_attn.in_proj_b`. If it returns False, `create_qkvz_proj`
appends `[num_v_heads, num_v_heads] = [48, 48]` to `output_sizes` (`qwen3_5.py:221-222`) and at
TP=4 those become **12-row shards — violating rows%64, unshardable**. The production AWQ twin's
`config.json` lists both per layer (FACT, 07 §5, 311 entries). Ours must too. Effect: `in_proj_qkvz`
becomes `[2048, 2048, 6144, 6144]` (→ `512/512/1536/1536` at TP=4, all %64 ✓) and `in_proj_ba`
becomes a separate `MergedColumnParallelLinear([48, 48])` served by `UnquantizedLinearMethod()`.

### 3.2 `PXQ4LinearMethod`

```python
class PXQ4LinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        N = sum(output_partition_sizes); K = input_size_per_partition
        # HARD ASSERTS — do not rely on round(x // packed_factor), it truncates silently
        for p in output_partition_sizes:
            if p % 64: raise ValueError(f"PXQ4 output partition {p} not a multiple of 64 rows")
        if K % 32: raise ValueError(f"PXQ4 input partition {K} not a multiple of 32 columns")
        wl = extra_weight_attrs["weight_loader"]

        slabs = Parameter(torch.empty(N // 64, K // 32, 1088, dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(slabs, dict(output_dim=0, input_dim=1,
                                     packed_dim=0, packed_factor=64, weight_loader=wl))
        anchor = Parameter(torch.empty(N // 64, 64, dtype=torch.float16), requires_grad=False)
        set_weight_attrs(anchor, dict(output_dim=0, packed_dim=0, packed_factor=64,
                                      weight_loader=wl))     # NOTE: no input_dim  (see §3.3)
        layer.register_parameter("pxq4_slabs", slabs)
        layer.register_parameter("pxq4_anchor", anchor)
        layer._sm70_f16_forbidden = True                     # linear.py:57-58, defensive
```

`apply` copies the template's CUDA-graph discipline verbatim
(`sm70_turbomind.py:290-339`): reshape to 2-D, preallocate `out` with `torch.empty`, call an
**out-variant** op, `out.add_(bias)`, reshape back.

```python
    def apply(self, layer, x, bias=None):
        st = layer._pxq4_state                       # plain Python ints, resolved at load time
        xf = x.reshape(-1, x.shape[-1])
        out = torch.empty((xf.shape[0], st.N), dtype=x.dtype, device=x.device)
        if xf.shape[0] <= st.mmv_max_ny:             # decode: 8, a Python int
            torch.ops.pxq4.pxq4_mmv_out(out, xf, layer.pxq4_slabs, layer.pxq4_anchor, st.K, st.N)
        else:                                        # prefill: dequant -> cuBLAS HMMA
            w = torch.empty((st.N, st.K), dtype=torch.float16, device=x.device)
            torch.ops.pxq4.pxq4_dequant_out(w, layer.pxq4_slabs, layer.pxq4_anchor)
            torch.mm(xf, w.t(), out=out)
        if bias is not None: out.add_(bias)
        return out.reshape(x.shape[:-1] + (st.N,))
```

`process_weights_after_loading` resolves `N`, `K`, `panels = N//64`, `kslabs = K//32` to Python
`int`s from `pxq4_slabs.shape` (**never** from a device tensor read in `apply()` — a `.item()`
in `apply` breaks CUDA-graph capture; FACT, the template resolves `k_ld`/`q_ld` with
`int(meta[0])` at load time, `sm70_turbomind.py:146-147`), stashes them on `layer._pxq4_state`,
and — in `runtime == "dequant"` mode (S1/S2) — replaces the two params with a single fp16
`weight` and swaps `apply` to `F.linear`. Note `K` **must** come from the post-shard tensor shape
(`pxq4_slabs.shape[1] * 32`), not from any header field, because a K-shard changes the panel
stride.

**It must never set `layer._sm70_f16_prepared`.** `_maybe_sm70_dense_forward` (`linear.py:56-96`)
is checked *before* `quant_method.apply()` (`:425-427`, `:805`, `:1789-1795`) and keys on that
attribute (`:62-63`), which only `UnquantizedLinearMethod.process_weights_after_loading` sets
(`:408`). `_mark_default_sm70_dense_modules` (`qwen3_5.py:168-181`) does set
`_sm70_f16_force_enable = True` on every `qkv_proj`/`out_proj`, but that flag alone is inert
(07 §6). Setting `_sm70_f16_forbidden` (`linear.py:57-58`) is belt-and-braces.

### 3.3 How TP sharding is declared — the whole mechanism

There is **no sharder API**. vLLM's loaders read three `getattr` attributes off each parameter
and call `narrow()` (FACT: column `linear.py:775-778`; row `linear.py:1749-1752`; packing divide
`linear.py:1053-1058` and `:1013-1018`; v2 equivalents `parameter.py:148-230`, `:605-616`).

| param | dtype / per-rank shape | attributes |
|---|---|---|
| `pxq4_slabs` | `uint8 [N/64, K/32, 1088]` | `output_dim=0, input_dim=1, packed_dim=0, packed_factor=64` |
| `pxq4_anchor` | `fp16 [N/64, 64]` | `output_dim=0, packed_dim=0, packed_factor=64` — **no `input_dim`** |

- **Column / merged-column / QKV split (output rows).** The loader divides `shard_offset` and
  `shard_size` by `packed_factor=64`, converting row units to **panel** units, then `narrow(0)`.
  A panel is a self-contained contiguous byte range (FACT, `pxq6.cuh:520-526`), so this is a pure
  memcpy of whole panels — byte-identical, zero overhead. Verified against real shapes (07 §3):
  `in_proj_qkvz` offsets `0/2048/4096/10240` → `0/32/64/160` panels, sizes at TP=4
  `512/512/1536/1536` → `8/8/24/24` panels; `gate_up` size `4352` → `68` panels. All integral at
  TP=2 and TP=4. Smallest object anywhere: 8 panels.
- **Row split (K).** `narrow(1)` on `pxq4_slabs` takes a contiguous slab subrange from every
  panel. Dim 1 is *already* in 32-column slab units and **the packing divide is never applied to
  the input dim** (FACT, `linear.py:1728-1761` reads only `input_dim`; `parameter.py:220-230`
  likewise) — so there is no unit mismatch. `pxq4_anchor` has no `input_dim`, so the row loader
  skips its narrow entirely (`linear.py:1749`, guarded `if input_dim is not None`) and copies it
  whole: **that is the 128 B header duplication, obtained for free.** Cost +0.60 MiB/rank total
  at TP=4 (07 §3).
- **Loader version: v1.** Do *not* add `PXQ4LinearMethod` to `WEIGHT_LOADER_V2_SUPPORTED`
  (`linear.py:193-210`). v1 honours `is_sharded_weight` (`linear.py:755-761`) as an escape hatch,
  and applies `packed_factor` in both merged branches; v2's fused-on-disk branch only applies
  packing for literal `PackedColumnParameter`/`PackedvLLMParameter` instances
  (`parameter.py:162`, `:186`). (If v2 is later preferred, subclass `PackedvLLMParameter` —
  attributes alone are not enough there.)
- **The silent failure to guard against.** `round(shard_size // packed_factor)`
  (`linear.py:1557-1559`, `parameter.py:606-609`) truncates a misaligned offset *without raising*,
  producing a well-formed wrong slice and subtly wrong logits. Hence the explicit `%64`/`%32`
  asserts in `create_weights`, and the converter-side gate of §7 C2.

`attn_q` needs **no permutation**: it is 12288 rows = per-head interleaved `[q_h | gate_h]`
(FACT, `src/llama-build-context.cpp:2003-2007` — `ggml_view_3d` with stride `2*row_size`, q at
offset 0, gate at `row_size`), matching what `qwen3_next.py:565-567` reconstructs. A contiguous
3072-row slice at TP=4 is semantically correct.

---

## 4. The CUDA side

### 4.1 Which kernels, and why so few

Only **two** device entry points are needed for the whole v1:

1. `k_pxq6_dequant_matrix<pxq6_pol_p6, half>` (FACT, `pxq6.cuh:681-726`; host wrapper `:728-741`)
   — grid `nslabs = (nrows/64)*kslabs`, block 64, writes row-major `[N][K]`.
2. `k_pxq6_mmv<pxq6_pol_p6, MODE, VECX>` (FACT, `pxq6.cuh:914-971`) — decode GEMV, block 256,
   built on `pxq6_dot32` (`:634-674`), which together with `pxq6_pol_p6` (`:317-346`) and
   `pxq6_ldcodes` (`:436-464`) *is* the PXQ4 decoder.

Deliberately **not** ported in v1:

- **The fused prefill GEMM** `k_pxq6_gemm_grouped` (`pxq6.cuh:2517-2625`). It already computes
  vLLM's exact contract (`A` row-major half `[M][K]` `:2524`, `C` row-major `[M][R]` `:2525,
  :2621-2623`, optional per-row bias `:2617`) — but it is a scalar `__hfma2` tile with no tensor
  cores, and **our own measurement note says folding the dequant into it is −18.6% on sm_70**
  versus coalesced-dequant + cuBLAS, with the explicit corollary that the winning shape keeps
  cuBLAS's HMMA (FACT, `ggml-cuda.cu:4436-4444`; the dense 2D driver is clamped to `cc < CC_VOLTA`
  at `ggml-cuda.cu:4533` for exactly this reason). So v1 prefill is `dequant → torch.mm`, which is
  both faster *and* less code.
- **The WMMA twin** `k_pxq6_gemm_grouped_wmma` (`pxq6.cuh:2912`, `__CUDA_ARCH__ >= 700 && < 750`)
  — reachable today only from the MoE driver (`ggml-cuda.cu:5056`), env-default OFF, and
  explicitly not bit-exact (`pxq6.cuh:53-59`). Phase 2, and it needs GPU time we do not have.
- **`pxq-mmvq.cuh`** — the one genuinely non-portable family: it lives inside ggml's
  `mmvq-templates.cuh`, keys on `ggml_type` template params, and consumes `block_q8_1`
  (`pxq-mmvq.cuh:134`). Porting it means porting ggml's q8_1 activation quantizer. Never.
- **The K-split mmv variants**, because `pxq6_ksplit_workspace` does a raw `cudaMalloc` and
  **declines under stream capture** (FACT, `pxq6.cuh:2480-2494`) — a direct hazard against
  `cudagraph_mode=FULL_AND_PIECEWISE`. They are bit-identical to the unsplit form
  (`pxq6.cuh:26-31`), so skipping them costs only occupancy.

**Shared-memory check** (INFERENCE from shape arithmetic, not measured): the mmv path stages all
of `x` in dynamic smem, capped at 46 KB (`ggml-cuda.cu:4262`). At TP=4 every tensor fits
(`ffn_down` K=4352 → 17.4 KB; `ffn_gate/up` K=5120 → 20.5 KB). At TP=2 also fits (`ffn_down`
K=8704 → 34.8 KB). At **TP=1 `ffn_down` (K=17408 → 69.6 KB) does not fit** — TP=1 is unsupported
in v1 and must raise a clear error, not silently misbehave.

### 4.2 The vendored slice (`pxq4_pxa.cuh`, ~500 copied + ~60 edited LOC)

Copy verbatim from `/path/to/engine-repo/ggml/src/ggml-cuda/pxa/pxq6.cuh` (**not** `pxq4.cuh` — that
file documents the retired id-250 MXFP4-repack format and contains no id-252 compute kernel at
all; FACT, `pxq4.cuh:1-17`, `:59-60`, `:117-119`):

| block | source range |
|---|---|
| table constants (subset of `ggml-pxq6-tables.h:21-27`) + `__device__ pxq6_book_g[16]`, `pxq6_sub16_g[16]` with their frozen `PXQ6_BOOK_INIT`/`PXQ6_SUB16_INIT` | `ggml-pxq6-tables.h:21-27,33-44`; `pxq6.cuh:78-80` |
| `pxq6_pol_p6` policy struct (`QK/BM/SLAB/HDR/CODE_OFF=64/NEFF=2/CODE_WORDS=4/CODE_BYTES=16`, `anchor()`, `row_effs()`) | `pxq6.cuh:317-346` |
| `pxq6_panel_stride`, `pxq6_panel` | `pxq6.cuh:519-527` |
| `pxq6_ldcodes`, `pxq6_acc2`, `pxq6_pairx` + mode structs | `pxq6.cuh:436-464` and the small helpers around `:400-520` |
| `pxq6_dot32` | `pxq6.cuh:634-674` |
| `k_pxq6_dequant_matrix` + `dequantize_row_pxq6_cuda` | `pxq6.cuh:681-741` |
| `k_pxq6_mmv` | `pxq6.cuh:914-971` |

Edits required (all enumerated in 02 §3; ~60 LOC):

1. Delete `#include` of `pxa-enhance.cuh` (pulls in `ggml.h` for `ggml_pxa_model_profile`); replace
   the `PXA_PXQ6_GATE` macro's `pxa_gate_default(x)` with the literal default.
2. Delete `pxa_pxq_fmt(ggml_type)` (`pxq6.cuh:3335-3345`) — we know it is PXQ4.
3. Delete the `PXQ6_PICK*` policy pickers (`pxq6.cuh:3375-3460`) and instantiate `pxq6_pol_p6`
   directly; this also drops the `pxq23.cuh` include and the P1/P2/P3 tables.
4. Freeze the ~25 `getenv` gates (`pxq6.cuh:96-247`) to compile-time constants, so a stray env var
   cannot silently change vLLM numerics.
5. Delete `pxq6_maybe_upload_tables` (`pxq6.cuh:274-306`) — it does `cudaSetDevice` mid-call and is
   not capture-safe. Keep the frozen `__device__` table symbols. **Add a load-time host-side
   assert that the file's `pxa.pxq6.book`/`pxa.pxq6.sub` KVs equal the compiled-in tables**, and
   refuse to load otherwise (this is the only place the env-override risk resurfaces).
6. `k_pxq6_mmv` takes fp32 activations (`pxq6.cuh:930`) and writes fp32 (`:968-969`); vLLM hands
   fp16. Change the stage to `xs[idx] = __half2float(xh[idx])` and the store to `__float2half_rn`
   (~6 LOC).
7. `CUDA_CHECK` → `C10_CUDA_KERNEL_LAUNCH_CHECK()`.

Nothing else. FACT: `pxq6.cuh` is 3601 lines containing exactly **10** `ggml_`/`GGML_` tokens, of
which 6 are in that one host-side type mapper and 4 are comments; the `__global__` kernels touch
zero ggml types (02 verdict).

### 4.3 The torch shim (`pxq4_sm70.cu`, ~320 LOC)

```cpp
TORCH_LIBRARY(pxq4, m) {
  m.def("pxq4_dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()");
  m.def("pxq4_mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor, int K, int N) -> ()");
}
TORCH_LIBRARY_IMPL(pxq4, CUDA, m) { m.impl("pxq4_dequant_out", &pxq4_dequant_out);
                                     m.impl("pxq4_mmv_out",     &pxq4_mmv_out); }
```

Discipline copied from the fork's own ops (FACT, `csrc/torch_bindings.cpp:215-218` uses exactly
this `Tensor(a!) out ... -> ()` out-variant form; `awq_sm70_gemm.cu:3050` takes the stream with
`at::cuda::getCurrentCUDAStream()`, `:3062-3075` `TORCH_CHECK`s only host-visible dtype/device
properties):

- `TORCH_CHECK` dtype/contiguity/device only — never read device memory.
- Stream from `at::cuda::getCurrentCUDAStream()`; no stream creation, no events, no
  `cudaDeviceSynchronize`, no host allocation inside the op.
- No scratch allocation inside the op (this is why the K-split kernels are excluded).
- Grid/block computed from the tensor *shapes*, which are host-side metadata.
- `slabs` is passed as `reinterpret_cast<const uint8_t*>(slabs.data_ptr())`; the panel base
  arithmetic uses `kslabs = slabs.size(1)` — post-shard, as required.

Our ops live in the `pxq4::` namespace, not `_C`. `torch.ops` is a global registry, so a separate
`.so` needs no vLLM symbol and no `_C` relink (INFERENCE, well-supported; 05 §A2).

### 4.4 `register_fake` — the single most likely thing to be forgotten

Under `cudagraph_mode=FULL_AND_PIECEWISE` the linear layers sit inside the `torch.compile` region.
**Without a meta kernel, tracing fails before capture is even attempted.** The fork registers one
for every sm70 op (FACT, `vllm/_sm70_ops.py:302-315` `@register_fake("_C::awq_gemm_sm70_out")`,
`:113-130`). Ours:

```python
@torch.library.register_fake("pxq4::pxq4_dequant_out")
def _(out, slabs, anchor): return None
@torch.library.register_fake("pxq4::pxq4_mmv_out")
def _(out, x, slabs, anchor, K, N): return None
```

Both return `None` because both are out-variants. The mutable-alias annotation `Tensor(a!)` in the
schema is equally load-bearing: it tells the piecewise-graph partitioner the op mutates `out` and
must not be reordered or DCE'd.

**Open item (ASSUMPTION, not verified):** whether an out-of-tree `TORCH_LIBRARY` namespace
participates in piecewise CUDA-graph capture alongside the `vllm::qwen_gdn_*` splitting ops
(`compilation.py:764-773`). Mitigation is cheap and staged: S1/S2 use dequant-at-load and hit no
custom op in the hot path at all, so this risk is only faced at S3 — by which point everything
else is proven. Fallback if it bites: `--enforce-eager` for a first correct run, then add our op
names to `splitting_ops` via the compilation config.

### 4.5 Build — without rebuilding vLLM

```python
# setup.py
CUDAExtension(
    name="pxq4_vllm._pxq4_C",
    sources=["pxq4_vllm/csrc/pxq4_sm70.cu"],
    extra_compile_args={"cxx": ["-O3", "-std=c++17",
                                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"],
                        "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                                 "-gencode", "arch=compute_70,code=sm_70"]})
```

FACT (05 §A2): the container has `nvcc` 12.8 at `/usr/local/cuda/bin/nvcc`, `gcc`/`g++`, `cmake`
and `ninja` in `/opt/vllm-venv/bin`, and torch `2.10.0+cu128` — so a `CUDAExtension` build matches
the runtime ABI exactly. The `_GLIBCXX_USE_CXX11_ABI` flag must agree with the prebuilt
`_C.abi3.so`; that is the one thing to verify on first build.

**Two operational blockers, both FACT (05 §A2), both with a fixed workaround:**

1. The production container's overlay is `207G used, 0 avail, 100%`. **You cannot build in
   `vllm-qwen38-27b-cyber-1`,** and you must not try — it is the box owner's production service.
   Build in a *fresh* container from the same image with a writable volume under `/path/to/models`,
   then `pip install` the resulting wheel into the deployment.
2. The DGX root filesystem is 100% full. Everything — build tree, wheel, converted checkpoint —
   lives under `/path/to/models`. Never `/` and never host `/tmp`.

No GPU lease is needed for any of the build steps or for stages S0-S2's offline gates.

---

## 5. Weight loading

### 5.1 Why the GGUF path is dead (settled, do not revisit)

FACT (05 §B1), measured in the container: `gguf.GGMLQuantizationType(252)` raises `ValueError`,
and the raise is inside `GGUFReader._build_tensors`, which walks the **whole** tensor table in the
constructor. One PXQ4 tensor kills the entire file open, before any tensor is yielded — reached
from `weight_utils.py:1273/1293/1337` via `gguf_loader.py:22-28`. Downstream is equally closed:
`quantization/gguf.py:13` imports the same enum; its five type sets (`:163-193`) exclude 252;
`GGUFLinearMethod.process_weights_after_loading` raises for anything outside them (`:485-491`);
and `_fused_mul_mat_gguf` (`:199-229`) dispatches into vLLM's own **vendored, older** ggml
`_custom_ops`, where 252 is unknown at the C level. And even if all of that were patched, the
`GGUFUninitializedParameter` sharder (`gguf.py:452-500`) slices assuming per-row-contiguous
blocks, which 64-row panel interleave violates — the same reason ggml's own `to_float`/`vec_dot`
are deliberately NULL for PXQ4 (`ggml.c:1407-1414`, `pxq-cpu.h:4-12`).

**Offline converter → safetensors makes the problem disappear rather than solving it,** and it
never touches Kewaii's tree.

### 5.2 The five types, and what the converter does with each

Measured census (FACT, 06 §1 — 866 tensors, 15,719,771,584 B on disk):

| id | type | count | bytes | GiB | converter action (v1) |
|---:|---|---:|---:|---:|---|
| 252 | **PXQ4** | 325 | 12,231,950,336 | 11.392 | **de-interleave** into `pxq4_anchor` fp16 `[N/64,64]` + `pxq4_slabs` uint8 `[N/64,K/32,1088]`. Byte-preserving; the only transform. |
| 8 | Q8_0 | 132 | 1,621,032,960 | 1.510 | dequant → fp16. `attn_k`/`attn_v` (17 each), `ssm_alpha`/`ssm_beta` (48 each, tiny), `output.weight` (the only big one), `nextn.eh_proj`. |
| 14 | Q6_K | 1 | 1,042,944,000 | 0.971 | `token_embd` → fp16. Embeddings are gathered, not GEMM'd. |
| 39 | **MXFP4** | 48 | 802,160,640 | 0.747 | `ssm_out` (GDN out_proj). **v1: dequant → fp16.** Phase 2: the fork's existing sm70 MXFP4 GEMM. |
| 0 | F32 | 360 | 10,686,464 | 0.010 | passthrough; norms stay f32. |

**No F16 tensors exist in this file at all** — the backbone table's `attn_gate_head=f16` rule never
fires because every `attn_gate` here is per-channel (`ne[1]=6144`). The design still handles f16 as
a passthrough case, because a future re-quantize could produce it.

Layout confirmed byte-exactly: `bytes = (rows/64)*(128 + (K/32)*1088)` reproduces all six distinct
PXQ4 shapes to the byte, with no inter-tensor padding (FACT, 06 §5). All 325 PXQ4 tensors pass
`rows%64==0 && K%32==0`, at TP=1, TP=2 and TP=4.

**The one genuinely mixed fused module: `qkv_proj`.** It fuses `attn_q` (PXQ4) + `attn_k` (Q8_0) +
`attn_v` (Q8_0), and vLLM gives one module exactly one quant method. v1's answer is the boring
one: **dequant `attn_q` to fp16 as well and serve the whole `qkv_proj` unquantized.** Cost:
+0.37 GiB/GPU at TP=4 (§9). This only affects the 17 full-attention blocks; the 48 GDN blocks use
`in_proj_qkvz`, which is uniformly PXQ4 (`attn_qkv` 252 + `attn_gate` 252) and stays quantized.
Phase 2 removes the cost by re-quantizing `attn_k`/`attn_v` to PXQ4 (both are geometrically
eligible: 1024 rows %64 ✓, 256/rank at TP=4 %64 ✓) via the quantizer's existing `PXA_PXQ_KV`
override (`llama-quantize.cpp:1355-1371`), producing a uniform-PXQ4 `qkv_proj`.

### 5.3 Name mapping and fusion

The mapping is the fiddliest part of the job and the cheapest to gate. Rule: **emit exactly the
key set the AWQ twin emits**, then diff. The twin loads today under this exact model code, so
matching it is definitive.

| GGUF | vLLM module | class | parallel |
|---|---|---|---|
| `blk.N.attn_q` (12288 = q\|gate per-head interleaved) + `attn_k` + `attn_v` | `...self_attn.qkv_proj` | `QKVParallelLinear` | column |
| `blk.N.attn_output` | `...self_attn.o_proj` | `RowParallelLinear` | **row (K)** |
| `blk.N.attn_qkv` (10240) + `blk.N.attn_gate` (6144) | `...linear_attn.in_proj_qkvz` (16384) | `MergedColumnParallelLinear([2048,2048,6144,6144])` | column |
| `blk.N.ssm_alpha` + `ssm_beta` (48 each) | `...linear_attn.in_proj_ba` | `MergedColumnParallelLinear([48,48])` | column, **fp16 always** |
| `blk.N.ssm_out` | `...linear_attn.out_proj` | `RowParallelLinear` | **row (K)** |
| `blk.N.ffn_gate` + `ffn_up` (17408 each) | `...mlp.gate_up_proj` | `MergedColumnParallelLinear` | column |
| `blk.N.ffn_down` | `...mlp.down_proj` | `RowParallelLinear` | **row (K)** |
| `token_embd` | `embed_tokens` | `VocabParallelEmbedding` | vocab |
| `output` | `lm_head` | `ParallelLMHead` | column |

`packed_modules_mapping` (`qwen3_5.py:665-675`, `:823-826`) is `qkv_proj←[q_proj,k_proj,v_proj]`,
`gate_up_proj←[gate_proj,up_proj]`, `in_proj_qkvz←[in_proj_qkv,in_proj_z]`,
`in_proj_ba←[in_proj_b,in_proj_a]` — so the converter emits the **unfused** checkpoint names and
vLLM's merged loaders do the fusing at load, shard-id by shard-id. Our PXQ4 params ride the same
path: checkpoint key `...linear_attn.in_proj_qkv.pxq4_slabs` resolves to the registered parameter
`pxq4_slabs` on the `in_proj_qkvz` module, exactly as AWQ's `.qweight` does.

Only **three** quantized classes are row-parallel (`o_proj`, GDN `out_proj`, `ffn_down`); those are
the only ones exercising the slab-subrange K-split. Everything else is whole-panel memcpy.

### 5.4 The emitted `config.json`

Copied from the AWQ twin with `quantization_config` replaced:

```json
{"quant_method": "pxq4",
 "pxq4_tier": "core", "pxq4_version": 1,
 "backbone_rev": 2,
 "backbone_map": "attn_q,attn_qkv,attn_output,attn_gate_ch,shexp,ffn_dense=tier+1;attn_k,attn_v=q8_0;attn_gate_head=f16;token_embd=q6_k;output=q8_0",
 "book": [ ...16 fp32 from pxa.pxq6.book... ],
 "sub":  [ ...16 fp32 from pxa.pxq6.sub...  ],
 "runtime": "native",
 "tensor_types": {"mlp.gate_up_proj": "pxq4", "mlp.down_proj": "pxq4",
                  "linear_attn.in_proj_qkvz": "pxq4", "self_attn.o_proj": "pxq4",
                  "linear_attn.out_proj": "fp16", "self_attn.qkv_proj": "fp16",
                  "linear_attn.in_proj_ba": "fp16", "lm_head": "fp16"},
 "ignore": ["model.language_model.layers.N.linear_attn.in_proj_b",
            "model.language_model.layers.N.linear_attn.in_proj_a", "lm_head", "..."]}
```

`backbone_rev` / `backbone_map` are carried verbatim from the file's `pxa.pxq.backbone_rev` /
`pxa.pxq.backbone_map` KVs (FACT, 06 §3) so the artifact's provenance survives conversion and a
mismatched table can be detected later.

---

## 6. MoE

**There are no routed experts. The "40×256 routed experts" line item is void, and this is
measured, not assumed.**

FACT (05 §0, 06 §1): the GGUF header of `/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf` was parsed
with a raw struct parser — 866 tensors, `general.architecture = qwen35`, **zero `*_exps` tensors
and zero expert KVs of any kind**. `ffn_gate`/`ffn_up`/`ffn_down` are dense on all 65 blocks
(6 distinct PXQ4 shapes, none expert-stacked). This is consistent with the brief's own statement
and with the AWQ twin's HF config. Sky-35B is the MoE model; this is not it.

Concretely: **no `FusedMoEMethodBase`, no `RoutedExperts` path, no fused expert-stack parameter,
no MoE dequant fallback, and therefore no MoE VRAM cost to quantify. Effort delta: zero.**

For completeness, if a future PXQ4 MoE artifact does need this: the format already carries experts
as its **outermost** axis (`pxq6_panel(W,e,panels,p,kslabs) = W + (e*panels+p)*panel_stride`,
FACT `pxq6.cuh:520-526`), so an expert stack is a contiguous run of per-expert panel blocks and an
expert-parallel split is another whole-panel memcpy. The MoE-shaped kernels
(`k_pxq6_gemm_gufuse` `:2631`, `k_pxq6_gemm_down_scat` `:2766`) already exist and are what the
dense 2D driver degenerates from (`ggml-cuda.cu:4493-4570` builds a one-expert tile map) — i.e.
the MoE case is *less* work for us than the dense case, not more. That is a note for the Sky-35B
port, not work in this one.

---

## 7. Staged plan with offline-verifiable checkpoints

Every gate below **C0-C5** is verifiable without a full model run. Gates C0-C3 need no GPU lease
at all; C4-C5 need only a single small allocation, not a benchmark.

### Stage S0 — fp16 reference checkpoint. No plugin, no CUDA.
Build `tools/pxq4_gguf` with `--emit fp16`: every tensor dequantized to fp16/f32, names mapped,
`config.json` carrying **no** `quantization_config`. Load with **stock vLLM in the fork**, TP=4.

- **C0 (CPU, no GPU):** for each of the six PXQ4 shapes, `pxq4_codec.dequant()` vs the CPU
  reference `pxa_pxq_dequant_2d` (`ggml/src/pxq-cpu.c:219-225`, bound with ctypes) — **`torch.equal`
  bit-exact in fp32**. This is the parity-locked contract (`pxq-cpu.h:16-18`: *"the dequant itself
  IS the parity-locked contract (fp32 eff/book products)"*), and passing it validates the panel
  arithmetic, the header split, the SoA scale plane, the nibble order and the BOOK/SUB16 tables in
  one shot — **>90% of the port's risk, retired before any GPU is touched.**
- **C1 (CPU):** converted key set == AWQ twin key set, exactly. Byte sizes reproduce
  `(R/64)*(128+(K/32)*1088)` for all 325 PXQ4 tensors.
- **C2 (GPU, load only):** the fp16 checkpoint loads at TP=4 and generates. Compare greedy
  continuations against our llama.cpp engine on the same prompts. This proves the name map, the
  GDN split, `in_proj_qkvz` fusion order, `attn_q` interleave, and the whole model wiring —
  with **zero** new code in the inference path.
  VRAM: 13.5 GiB/GPU at TP=4 (fits 32 GB with KV). **Does not fit 2×16 GB Unraid** — S0 is
  DGX-only.

### Stage S1 — plugin, sharding, dequant-at-load. Still no CUDA.
`--emit pxq4`, `runtime: "dequant"`. `PXQ4Config` + `PXQ4LinearMethod` registered; the two
parameters declared and TP-sharded by stock loaders; `process_weights_after_loading` runs the
**pure-torch** vectorised dequant into an fp16 `weight` and swaps `apply` to `F.linear`.

- **C3 (CPU/GPU, no model run):** shard-then-dequant vs dequant-then-shard, **bit-identical in
  fp32**, for both axes, at TP=2 and TP=4, on real tensors. This is the test that proves the TP
  repack is a *permutation*, not a re-quantization. Run it standalone against the safetensors —
  no vLLM needed.
- **C4 (load only):** the model loads at TP=4 under the plugin and reproduces S0's outputs
  **exactly** (same fp16 weights arrive by a different route). Any divergence localises to
  sharding or dispatch, with the kernel not yet in play.
  Also asserts: `_sm70_f16_prepared` never set; `get_quant_method` returns
  `UnquantizedLinearMethod()` (never `None`) for every `LinearBase`.

### Stage S2 — CUDA dequant op.
Vendor `pxq4_pxa.cuh`, build `pxq4_sm70_C.so`, replace the torch dequant with
`torch.ops.pxq4.pxq4_dequant_out`. Still dequant-at-load.

- **C5 (single allocation, no benchmark):** GPU `k_pxq6_dequant_matrix` vs the CPU reference,
  `torch.equal` bit-exact in fp32, on all six shapes and on sharded slices. Then re-run C4 —
  outputs must be unchanged from S0/S1.

### Stage S3 — 4-bit resident. First real win.
Stop dequantizing at load (`runtime: "native"`). Add `k_pxq6_mmv` (fp16-staged) for `M <= 8`;
prefill = `pxq4_dequant_out` → `torch.mm`.

- **C6:** per-layer numeric check — `pxq4_mmv_out` vs `dequant → torch.mm` with an fp16-appropriate
  tolerance (these paths are **not** bit-exact to each other by design: the GEMM snaps
  `__float2half_rn`, the mmv accumulates in fp32 — `pxq6.cuh:13-15`). Gate on logprob parity, not
  a hash.
- **C7:** CUDA-graph capture succeeds with `cudagraph_mode=FULL_AND_PIECEWISE`; if not, fall back
  to `--enforce-eager` for a correct run and treat capture as a separate work item.
- Only now is a throughput measurement meaningful — and it needs a GPU lease this study does
  not have.

### Stage S4 — phase 2 (performance), separately scheduled
1. `ssm_out` via the fork's existing sm70 MXFP4 GEMM (`sm70_turbomind.py:229-253`,
   `torch_bindings.cpp:198-200`) — **−0.52 GiB/GPU**, contingent on ggml's type-39 packing matching
   the fork's convention (**not verified**, check early).
2. `lm_head` + `token_embd` at 4 bits — **−0.87 GiB/GPU**.
3. `attn_k`/`attn_v` re-quantized to PXQ4 via `PXA_PXQ_KV`, giving a uniform-PXQ4 `qkv_proj` —
   **−0.37 GiB/GPU**.
4. Prefill: WMMA `k_pxq6_gemm_grouped_wmma` and/or the `gufuse` up+gate+SILU fusion
   (`pxq6.cuh:2631`; `pxq4_glu_apply` at `pxq4.cuh:87-98` already implements the exact
   SILU-swiglu branch a Qwen dense FFN needs). Both need GPU time and a quality gate.
5. MTP: `qwen3_5_mtp.py` spec decode is inherited free and is where the llama.cpp side got
   31.24 → 38.59 tok/s.

---

## 8. Risks, honestly

**R1 — v1 is slower than the incumbent, and someone will call that failure.** (Highest
probability, near certainty.) At TP=4, v1 = 5.225 GiB/GPU vs AWQ's 4.64 (§9), and the S3 decode
projection is ~90 tok/s against their **measured** 92.8 peak. The win is a phase-2 outcome, not a
v1 outcome. Mitigation: say it first, in every status update; the deliverable of v1 is *a correct
PXQ4 path in vLLM*, and the phase-2 items that flip the sign are enumerated and costed above
(−1.76 GiB/GPU total → 3.41 GiB/GPU → ~129 tok/s PROJECTION).

**R2 — silent mis-sharding.** `round(shard_size // packed_factor)` truncates a misaligned offset
without raising (`linear.py:1557-1559`, `parameter.py:606-609`), and a PXQ4 weight row is not
contiguous bytes, so a linear slice yields garbage rather than an error. A model that loads
cleanly and produces subtly wrong logits is the worst failure mode in this project. Mitigation:
the `%64`/`%32` asserts in `create_weights` (raise, never round), plus gate **C3**, plus the S0/S1
output-identity check at **C4**.

**R3 — the name map.** More conversions die on tensor naming than on numerics. `in_proj_qkvz`'s
fusion order (q,k,v then z), the `attn_q` per-head `[q|gate]` interleave, the MTP `blk.64` block,
and the `in_proj_ba` split are four independent chances to be wrong, and three of them produce
*plausible* output rather than an error. Mitigation: gate **C1** (key-set diff against the AWQ
twin, build fails on mismatch) and gate **C2** (greedy-continuation match against llama.cpp).
`attn_q` interleave is now FACT (`llama-build-context.cpp:2003-2007`), not assumption.

Beyond the top three:

**R4 — out-of-tree `TORCH_LIBRARY` under piecewise CUDA graphs** (ASSUMPTION, §4.4). Faced only
at S3; fallback `--enforce-eager`.

**R5 — MXFP4 packing convention mismatch** between ggml type 39 and the fork's `mxfp4_sm70_*`
(NOT VERIFIED, 05 §B4). Only affects phase-2 item 1; v1 fallback is fp16 `ssm_out`, already
costed.

**R6 — build/ABI.** `_GLIBCXX_USE_CXX11_ABI` disagreement with the prebuilt `_C.abi3.so`; the
production container cannot be built in (100% full overlay) and must not be touched. Mitigated by
building in a fresh container with a `/path/to/models` volume.

**R7 — vendoring forks the kernel.** Future fixes in `mgv-wt`'s `pxq6.cuh` will not propagate.
Mitigated by per-block provenance banners and the C0/C5 bit-exactness gates, which will catch any
drift immediately. Budget a periodic re-sync.

**R8 — TP=1 does not fit the mmv smem budget** (`ffn_down` K=17408 → 69.6 KB vs a 46 KB cap,
INFERENCE from `ggml-cuda.cu:4262`). Must raise a clear error, not silently degrade. TP=2 and
TP=4 both fit.

**R9 — the Unraid deployment (2×16 GB, TP=2).** S0/S1/S2 need 27 GiB/GPU and simply do not fit;
even S3 at 10.4 GiB/GPU leaves little KV headroom on 16 GB cards. Unraid is a phase-2 target
(V2 policy = 6.8 GiB/GPU), and its cards are currently occupied by live seats anyway.

**What would actually kill it:** nothing found. The three candidate killers were all checked and
all cleared — sharding (07: bit-identical both directions, both TP degrees), the plugin seam (03:
documented, tested, loaded in every rank), and the kernels' ggml entanglement (02: 10 tokens in
3601 lines, none in device code).

---

## 9. VRAM and throughput — PROJECTIONS, with their assumptions

Weight bytes per GPU. Method: exact byte formulas per type
(`pxq(rows,K) = (rows//64)*(128+(K//32)*1088)`, `q8 = rows*(K//32)*34`, `f16 = rows*K*2`,
`mxfp4 = rows*(K//32)*17`) applied to the measured per-tensor shapes from the artifact's tensor
directory, with column-parallel splits on rows and row-parallel splits on K, header duplication
included. Script: `prototypes/pxq-vllm/vram2.py`.

| policy | TP=4 | TP=2 | notes |
|---|---:|---:|---|
| S0/S1/S2 — all fp16 | ~13.5 GiB | ~27 GiB | correctness stages only; TP=2 does not fit 16 GB |
| **S3 v1** — PXQ4 for gate_up/down/qkvz/o_proj; fp16 tail | **5.225** | 10.445 | *above* the incumbent |
| S4a — v1 + 4-bit `ssm_out` + 4-bit head/embd | 3.84 | 7.7 | |
| **S4b V2** — everything 4-bit incl. k/v | **3.413** | 6.821 | |
| incumbent AWQ W4A16 (reported) | 4.64 | — | carries an **fp16 lm_head** (their ignore list, 07 §5) |

Decode-bandwidth projection. **Assumptions, all stated:** decode is purely weight-bandwidth-bound;
every weight except `token_embd` is read once per token; both engines achieve ~50% of the V100's
900 GB/s (= 450 GB/s effective), which is the brief's established measured fact; no MTP, no
batching, single stream.

| policy | bytes/GPU/token | projected tok/s | |
|---|---:|---:|---|
| S3 v1 | 4.63 GiB | **~90** | vs their **measured** 92.8 peak → parity at best, likely a small regression |
| S4a | 3.68 GiB | ~114 | |
| S4b V2 | 3.26 GiB | ~129 | |

Sanity check on the model itself: applying it to the incumbent's 4.64 GiB gives ~90 tok/s against
their measured 92.8 peak — the model reproduces a number we did not fit it to, which is the only
evidence available that these projections are in the right range. **No GPU run was performed for
this document; none of these are measurements.**

One further note on S3's prefill: `dequant → torch.mm` reads 4-bit weights, writes fp16, then
reads fp16 back, i.e. ~`B4 + 2*B_fp16` of traffic. At M=1 that is roughly **5× the bytes of a pure
fp16 decode** — which is precisely why `k_pxq6_mmv` (not the dequant path) must carry decode, and
why the `M <= 8` threshold in `apply()` is load-bearing rather than cosmetic. For prefill the
dequant amortises over M tokens and is the *faster* choice on sm_70, per our own −18.6% measurement
against the fused tile (`ggml-cuda.cu:4436-4444`).

---

## 10. LOC summary

| component | new/edited | copied verbatim |
|---|---:|---:|
| offline converter (`tools/pxq4_gguf/`, 6 files) | 990 | — |
| runtime package Python (`config`, `linear`, `ops`, `mxfp4`, `__init__`, packaging) | 800 | — |
| vendored kernel slice `pxq4_pxa.cuh` | 60 | 500 |
| torch shim `pxq4_sm70.cu` + `setup.py` | 390 | — |
| tests (4 files + CPU-ref binding) | 410 | — |
| **total** | **~2,650** | **~500** |

Of the 2,650, ~1,400 (converter + tests) is offline tooling with no CUDA and no vLLM import, and
~1,000 is the runtime plugin. Only **~250 lines** — the shim's launch/plumbing logic — are novel
GPU code; the numerics are copied, which is what makes bit-exact parity with the quality-gated
llama.cpp engine achievable at all.

---

## 11. Not verified / open

1. Whether an out-of-tree `TORCH_LIBRARY` namespace is capturable under
   `cudagraph_mode=FULL_AND_PIECEWISE` (§4.4). Faced at S3.
2. Whether ggml type-39 MXFP4 packing matches the fork's `mxfp4_sm70_prepare` convention (§8 R5).
   Phase-2 only.
3. `in_proj_qkvz` sub-projection order inside the GGUF `attn_qkv` tensor — inferred from
   2048+2048+6144=10240, **not** read from the quantizer's tensor-splitting code. Gate C1/C2
   catches it.
4. Whether `MergedColumnParallelLinear` with 4 `output_sizes` and 2 checkpoint names
   (`in_proj_qkvz←[in_proj_qkv, in_proj_z]`, `qwen3_5.py:665-675`) resolves shard ids as assumed.
   The AWQ twin does exactly this today, so the behaviour is proven — but our key-set diff (C1)
   is what confirms we match it.
5. The 46 KB dynamic-smem cap (`ggml-cuda.cu:4262`) was not re-read for this document; the TP=4 /
   TP=2 fit conclusions are shape arithmetic on top of it.
6. No GPU was run. Every tok/s and every GiB/GPU figure in §9 is a projection from byte counts.
