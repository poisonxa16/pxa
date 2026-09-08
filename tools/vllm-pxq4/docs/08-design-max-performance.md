# PXQ4 as a first-class quantization backend in 1Cat-vLLM (sm_70) — maximum-performance design

Target: `KewaiiGamer/1Cat-vLLM` @ `2ceb15066`, Qwen3.8-27B (gguf arch `qwen35`), 4x V100-32GB (DGX, TP=4)
and 2x V100-16GB (Unraid, TP=2). Artifact: `/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf`.
Every line citation below is to a file that was read (either `/path/to/engine-repo` @ `acf8f245`, or
`/opt/1Cat-vLLM` inside the running container). **No GPU was run in this study. Every throughput
number in this document is a PROJECTION and is labelled as such.**

---

## 0. Verdict, and the one decision that determines whether this is worth doing

**No blocker. The port is a ~2,100-LOC out-of-tree pip package plus one self-contained CUDA `.so`,
with ZERO patches to the vLLM fork.** The plugin seam (`register_quantization_config`,
`quantization/__init__.py:57-101`), the TP sharding declaration (`output_dim`/`input_dim`/
`packed_dim`/`packed_factor`, `linear.py:775-778`, `:1053-1058`, `:1749-1752`), the GDN kernels, the
FLASH_ATTN_V100 backend and the cudagraph machinery are all already in place and already running on
this exact box.

**But the headline premise in the brief is wrong, and the design turns on fixing it.** The artifact is
not uniformly PXQ4: only 11.39 of 14.63 GiB is (census §5.1). A backend that serves PXQ4 with our
kernel and promotes everything else to fp16 ("Policy A") lands at **4.864 GiB/GPU at TP=4 — worse than
the incumbent AWQ's 4.64 GiB/GPU**, i.e. a *regression*, because the whole thesis is
bytes-read-per-GPU-per-token.

The design therefore commits to **Policy MAX**: PXQ4 residency for every geometrically eligible
tensor, *including the two the current artifact does not cover* — `ssm_out` (48 tensors, currently
MXFP4 id 39) and `output.weight` / lm_head (currently Q8_0, 1.35 GiB). Both pass the `rows%64 &&
K%32` gate. That needs one extra offline step (a re-quantize of two tensor classes from the Q8_0
parent, §5.4), not a runtime feature.

**Projected outcome (PROJECTION; calibration and assumptions in §7):**

| | resident weights /GPU | read /GPU /decode step | projected peak single-stream |
|---|---|---|---|
| incumbent AWQ W4A16 (measured 92.8 tok/s peak) | 4.64 GiB | ~4.64 GiB | **92.8 (MEASURED)** |
| Policy A (PXQ4 + fp16 tail) | 4.864 GiB | 4.271 GiB | 100.7 |
| **Policy MAX (this design)** | **3.913 GiB** | **3.321 GiB** | **129.6** |
| Policy MAX + PXQ4 vocab tier (v2, §4.6) | 3.412 GiB | 3.321 GiB | 129.6 |

At TP=2 (Unraid 2x16 GB): Policy MAX = 7.815 GiB/GPU resident, leaving ~7 GiB/card for KV
(~110k tokens at 64 KiB/token) — it fits, projected 64.9 tok/s.

The second-order commitment that follows from "maximum performance": **weights stay in PXQ4 in VRAM
for the entire process lifetime.** There is no resident dequant anywhere in this design. Dequant
exists only as a *transient, bounded, preallocated* 42.5 MiB/rank tile used by the prefill path, and
that choice is itself evidence-driven, not lazy (§4.3).

---

## 1. Architecture

```
OFFLINE (CPU only, host box, no GPU)
  Qwen3.8-27B-PXQ4.gguf ──┐
  Qwen3.8-27B-Q8_0.gguf ──┤    tools/pxq4_convert.py
                          ├──> ├─ PXQ4 tensors: raw byte passthrough, panel/slab reshaped
  libpxq4enc.so (encoder) ┘    ├─ ssm_out + output.weight: dequant(Q8_0 parent) -> PXQ4 encode
  libpxq4ref.so (CPU deq)      ├─ attn_k/attn_v/ssm_alpha/beta/eh_proj (q8_0): -> fp16
                               ├─ token_embd (q6_k): -> fp16   [v2: -> PXQ4]
                               └─ norms (f32): -> f32
                          ==> out/  model-0000N-of-M.safetensors
                                    config.json  {"quantization_config": {"quant_method": "pxq4", ...}}
                                    pxq4_tiers.json (per-module tier + geometry, embedded in config)

RUNTIME (inside the existing container, nothing rebuilt)
  pip install pxq4_vllm-*.whl        # pure-python + one prebuilt libpxq4_sm70.so
      entry_points."vllm.general_plugins" -> pxq4_vllm:register
          register() imports config.py -> @register_quantization_config("pxq4") PXQ4Config
      torch.ops.load_library(libpxq4_sm70.so) -> torch.ops.pxq4.{gemv_out,gemm_out,dequant_out,...}

  vLLM engine start ──> load_general_plugins()  (arg_utils.py:749, v1/engine/core.py:108,
                                                v1/worker/worker_base.py:247 — every TP rank)
                   ──> ModelConfig._verify_quantization (model.py:1038-1066) probes custom
                       methods FIRST -> PXQ4Config.override_quantization_method -> "pxq4"
                   ──> qwen3_5.py builds the model; every LinearBase asks
                       PXQ4Config.get_quant_method(layer, prefix)
                   ──> PXQ4LinearMethod.create_weights declares pxq_slab/pxq_anchor with
                       output_dim/input_dim/packed_dim/packed_factor -> stock loaders shard
                   ──> process_weights_after_loading: freeze scalars, alloc workspace, NEVER
                       set _sm70_f16_prepared (set _sm70_f16_forbidden defensively)
                   ──> apply(): M<=MMV_MAX -> torch.ops.pxq4.gemv_out (fused, PXQ4-resident)
                                M> MMV_MAX -> dequant_out(tile) + cuBLAS HMMA mm
```

Two hard invariants that the whole design is built around:

1. **Every TP split is expressed in whole panels (64 rows, dim0) and whole slabs (32 columns, dim1).**
   Violation is silent — `round(shard_size // packed_factor)` truncates without raising
   (`linear.py:1055-1056`, `parameter.py:606-609`).
2. **The on-disk layout IS the kernel layout.** `process_weights_after_loading` performs no repack,
   no arithmetic, no allocation of weight-sized buffers. This is the property that lets us keep the
   weights at 4.25 bpw in VRAM and is the entire performance thesis.

---

## 2. File-by-file manifest

### 2.1 New repository `pxq4-vllm/` (all new files)

| file | LOC | contents |
|---|---|---|
| `pyproject.toml` | 40 | scikit-build-core backend; `[project.entry-points."vllm.general_plugins"] pxq4 = "pxq4_vllm:register"`; pin `torch==2.10.0`, `safetensors`, `numpy` |
| `CMakeLists.txt` | 70 | `find_package(Torch)`, `CUDA_ARCHITECTURES 70`, builds `libpxq4_sm70.so`; links **only** libtorch + CUDA runtime — no vLLM headers, no vLLM link (§4.7) |
| `csrc/pxq4_vendor.cuh` | **500 (copied verbatim)** | the id-252 slice of `mgv-wt/ggml/src/ggml-cuda/pxa/pxq6.cuh` (see §4.1 for the exact symbol list and line ranges) + the 32 table constants from `ggml/include/ggml-pxq6-tables.h:21-44`. Carries the provenance canary comment (`pxq6.cuh:1-3`) and a `PXQ4_VENDOR_SRC_COMMIT "acf8f245"` macro |
| `csrc/pxq4_vendor_edits.md` | 40 | the diff-log of the 60 edited lines, so a future re-sync is mechanical |
| `csrc/pxq4_kernels.cu` | 260 | new/edited kernels: fp16-in/fp16-out `k_pxq4_mmv_h` wrapper, `dst_t=half` instantiation of `k_pxq6_dequant_matrix`, `k_pxq4_gemm_grouped<half>`, (v2) `k_pxq4_embed_gather`, (v2) `k_pxq4_gate_up_silu` |
| `csrc/pxq4_ops.cpp` | 300 | `TORCH_LIBRARY(pxq4, …)` schemas, arg validation, stream plumbing, per-device workspace registry, launch-config policy table |
| `csrc/pxq4_workspace.cu` | 90 | one preallocated per-device dequant tile + one mmv reduction scratch; **no `cudaMalloc` at steady state** (this is what removes the `pxq6_ksplit_workspace` capture hazard, `pxq6.cuh:2480-2494`) |
| `pxq4_vllm/__init__.py` | 30 | `register()`: `torch.ops.load_library(...)`, import `config`, log the vendor commit |
| `pxq4_vllm/config.py` | 220 | `PXQ4Config(QuantizationConfig)` + tier map + `ignored_layers` |
| `pxq4_vllm/linear.py` | 300 | `PXQ4LinearMethod(LinearMethodBase)`, `PXQ4EmbeddingMethod` (v2), dispatch policy |
| `pxq4_vllm/ops.py` | 120 | thin python wrappers + `@torch.library.register_fake` metas for every op |
| `pxq4_vllm/policy.py` | 80 | M-threshold table, per-shape overrides, warmup-autotune cache (mirrors `warmup/awq_sm70_warmup.py` in spirit) |
| `tools/pxq4_convert.py` | 600 | GGUF reader (own parser, §5.1) → safetensors + config.json |
| `tools/pxq4_names.py` | 120 | ggml-name → vLLM/HF-name map, **generated and diffed against the incumbent AWQ checkpoint's `model.safetensors.index.json`** (§5.3) |
| `tools/enc/pxq4_enc.c` + `Makefile` | 60 | 40-line exported shim over `pxq6_quantize_expert` compiled from a **copy** of the mgv-wt tree (§5.4) |
| `tools/ref/pxq4_ref.c` + `Makefile` | 40 | standalone build of `ggml/src/pxq-cpu.c` (its only real dep is `GGML_COMPUTE_FP16_TO_FP32`) exporting `pxa_pxq_dequant_2d` for ctypes |
| `tests/*` | 250 | the six gates in §8 |
| **total** | **~2,120 new + 500 copied** | |

### 2.2 Files to patch in `/opt/1Cat-vLLM`

**None. Zero. The design requires no fork patch.** This is deliberate and is what makes the work
hand-offable to Kewaii as a package rather than a merge.

Three *optional* patches, each with a working no-patch fallback:

| optional patch | why you might want it | no-patch fallback |
|---|---|---|
| add `"pxq4"` to `QuantizationMethods` Literal (`quantization/__init__.py:12-45`) | typing/`--quantization pxq4` autocompletion only | the runtime list `QUANTIZATION_METHODS` is appended to by the decorator at `:92`; CLI already accepts it because plugins load in `add_cli_args` (`arg_utils.py:2772`) |
| add `pxq4::gemv_out` to `compilation.py:764-773` `splitting_ops` | if piecewise cudagraph capture rejects an out-of-tree namespace (§9 risk 3) | set `--cudagraph-mode PIECEWISE`, or pass `splitting_ops` through `CompilationConfig` on the CLI — a config value, not a code edit |
| opt `qkv_proj`/`out_proj` out of `_mark_default_sm70_dense_modules` (`qwen3_5.py:159-181`) | belt-and-braces | resolved: `_maybe_sm70_dense_forward` requires `_sm70_f16_prepared` (`linear.py:62-63`) which only a `process_weights_after_loading` sets; ours never will, and we set `_sm70_f16_forbidden` (`linear.py:57-58`) |

---

## 3. Python side

### 3.1 `PXQ4Config`

```python
# pxq4_vllm/config.py
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

@register_quantization_config("pxq4")                      # quantization/__init__.py:57
class PXQ4Config(QuantizationConfig):
    def __init__(self, tiers: dict[str, str], book: list[float], sub16: list[float],
                 backbone_rev: int, backbone_map: str, ignore: list[str]):
        super().__init__()                                  # base_config.py:73-76 — mandatory
        self.tiers = tiers            # module-prefix pattern -> "pxq4" | "fp16" | "skip"
        self.book, self.sub16 = book, sub16                 # from gguf KV pxa.pxq6.book/.sub
        self.backbone_rev, self.backbone_map = backbone_rev, backbone_map
        # BOTH names, because qwen3_5.py:144-148 scans several attribute spellings:
        self.ignore = self.ignored_layers = self.modules_to_not_convert = list(ignore)

    def get_name(self): return "pxq4"                       # base_config.py:78
    def get_supported_act_dtypes(self): return [torch.half] # :83  (V100: no bf16, no fp8)
    @classmethod
    def get_min_capability(cls): return 70                  # :88 — Volta; gate at config/vllm.py:611-621
    @staticmethod
    def get_config_filenames(): return []                   # :99 — we live in config.json
    @classmethod
    def from_config(cls, cfg): return cls(**_validate(cfg)) # :105, fed by weight_utils.py:302-321
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        return "pxq4" if hf_quant_cfg.get("quant_method") == "pxq4" else None   # :111-130

    def get_quant_method(self, layer, prefix):              # :150
        from .linear import PXQ4LinearMethod
        if isinstance(layer, LinearBase):
            tier = self._tier_for(prefix)
            if tier == "pxq4":  return PXQ4LinearMethod(self)
            return UnquantizedLinearMethod()   # NEVER None: linear.py:492-495 raises
        if isinstance(layer, VocabParallelEmbedding):
            return None            # -> UnquantizedEmbeddingMethod, vocab_parallel_embedding.py:479-482
        return None
```

Notes that are load-bearing:

* `get_min_capability() -> 70` unconditionally. The fork already established the precedent
  (`awq_marlin.py:265-266`, `compressed_tensors.py:108-110` return 70 unconditionally;
  `awq.py:177-184` conditionally). The single gate is `config/vllm.py:611-621`.
* We do **not** read `VLLM_SM70_QUANT_BACKEND` (`envs.py:793-813`). That is a routing knob inside
  *existing* configs (three-string `Literal`, no dispatch table); it is not a seam. PXQ4 is a
  first-class method name instead.
* `self.ignore` must contain `linear_attn.in_proj_b` and `linear_attn.in_proj_a`, **geometrically
  mandatory**: `_uses_split_gdn_input_projections` (`qwen3_5.py:127-157`) returns False otherwise,
  and the fused `in_proj_qkvz` then carries two 48-row shards = 12 rows/rank at TP=4, which can never
  satisfy `rows % 64`. The incumbent AWQ `config.json` lists exactly these; ours mirrors it.
* Checkpoint self-selection needs no CLI flag: custom methods are probed **first**
  (`model.py:1038-1044`, "custom overrides have preference").

### 3.2 `PXQ4LinearMethod`

```python
class PXQ4LinearMethod(LinearMethodBase):                   # linear.py:286-325
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra):
        N = sum(output_partition_sizes); K = input_size_per_partition
        # LOUD failure instead of silent truncation (linear.py:1055-1056 truncates)
        for n in output_partition_sizes:
            assert n % 64 == 0, f"PXQ4 shard {n} rows is not a multiple of 64 (panel)"
        assert K % 32 == 0,  f"PXQ4 K shard {K} is not a multiple of 32 (slab)"
        assert params_dtype == torch.half
        wl = extra["weight_loader"]

        slab = Parameter(torch.empty(N // 64, K // 32, 1088, dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(slab,   {"output_dim": 0, "input_dim": 1,
                                  "packed_dim": 0, "packed_factor": 64, "weight_loader": wl})
        anchor = Parameter(torch.empty(N // 64, 64, dtype=torch.half), requires_grad=False)
        set_weight_attrs(anchor, {"output_dim": 0,           # NO input_dim -> full copy on a K split
                                  "packed_dim": 0, "packed_factor": 64, "weight_loader": wl})
        layer.register_parameter("pxq_slab", slab)
        layer.register_parameter("pxq_anchor", anchor)
        layer.pxq_N, layer.pxq_K = N, K
```

Why this exact shaping (all FACT):

* **Column split (N):** `ColumnParallelLinear.weight_loader` narrows dim0 by
  `shard_size = param.shape[output_dim]` (`linear.py:775-778`) → whole panels, pure memcpy.
  For `MergedColumnParallelLinear`/`QKVParallelLinear` the offsets arrive in *row* units and are
  divided by `packed_factor=64` (`linear.py:1053-1058`, `:1013-1018`, qkv `:1556-1560`) → panel units.
  `adjust_marlin_shard` (`linear.py:218-232`) is the existing precedent for a tiled layout.
* **Row split (K):** narrows `pxq_slab` on `input_dim=1` in the param's own units — and the packing
  divide is **never** applied to the input dim (`linear.py:1749-1752`, `parameter.py:220-230`) — so
  slab units are correct as-is. `pxq_anchor` has no `input_dim`, so the loader skips the narrow
  (`linear.py:1747`) and copies it whole: **that IS the 128 B header duplication the format requires,
  obtained for free.**
* Use the **v1** loader path — do *not* add the class to `WEIGHT_LOADER_V2_SUPPORTED`
  (`linear.py:193-210`). v1 honours `is_sharded_weight` (`linear.py:755-761`) which v2 lacks, and v1
  applies `packed_factor` in both merged branches.
* The anchor is `[N/64, 64]`, not `[N]`, precisely so that both parameters take identical
  `packed_factor` treatment on merged/qkv loads. Flattened at first use.

```python
    def process_weights_after_loading(self, layer):
        # NO repack. The on-disk layout is the kernel layout. This is the whole point.
        layer.pxq_slab   = Parameter(layer.pxq_slab.data,   requires_grad=False)
        layer.pxq_anchor = Parameter(layer.pxq_anchor.data, requires_grad=False)
        layer._pxq = PXQ4State(                 # plain python ints — apply() must never sync
            panels=layer.pxq_N // 64, kslabs=layer.pxq_K // 32, N=layer.pxq_N, K=layer.pxq_K,
            mmv_max=policy.mmv_max_for(layer.pxq_N, layer.pxq_K))
        layer._sm70_f16_forbidden = True        # linear.py:57-58, defensive
        assert not getattr(layer, "_sm70_f16_prepared", False)   # linear.py:62-63 would bypass us
        pxq4_workspace.reserve(layer.pxq_N, layer.pxq_K, layer.pxq_slab.device)

    def apply(self, layer, x, bias=None):
        st = layer._pxq
        xf = x.reshape(-1, x.shape[-1])                       # sm70_turbomind.py:286-287 pattern
        M = xf.shape[0]
        out = torch.empty((M, st.N), dtype=x.dtype, device=x.device)   # graph-pool serviced
        if M <= st.mmv_max:
            torch.ops.pxq4.gemv_out(out, xf, layer.pxq_slab, layer.pxq_anchor, st.panels, st.kslabs)
        else:
            w = pxq4_workspace.tile(st.N, st.K, x.device)     # preallocated fp16 [N,K]
            torch.ops.pxq4.dequant_out(w, layer.pxq_slab, layer.pxq_anchor, st.panels, st.kslabs)
            torch.mm(xf, w.t(), out=out)                      # cuBLAS HMMA
        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], st.N)
```

The discipline copied from the template (`sm70_turbomind.py:284-339`): 2-D reshape, preallocated
output, `_out`-style custom op, `out.add_(bias)`, reshape back, and **all shape scalars converted to
python ints at load time** (`sm70_turbomind.py:146-147`) so `apply()` never touches a GPU value.

---

## 4. CUDA side

### 4.1 What is vendored verbatim (numerics stay bit-identical to the shipping engine)

From `mgv-wt/ggml/src/ggml-cuda/pxa/pxq6.cuh` — **`pxq4.cuh` is a decoy**, it documents the retired
id-250 MXFP4-repack format (`pxq4.cuh:1-12`, kernels deleted at `:59-60,117-119`).

| symbol | pxq6.cuh lines | role |
|---|---|---|
| `pxq6_book_g[16]` / `pxq6_sub16_g[16]` + `PXQ6_BOOK_INIT`/`PXQ6_SUB16_INIT` | `:78-80`, tables `ggml-pxq6-tables.h:21-44` | the two frozen 16-entry tables. **Keep the static initializers, delete `pxq6_maybe_upload_tables` (`:274-306`)** — it does `cudaMemcpyToSymbol` + `cudaSetDevice` mid-call, which is not capture-safe, and it only exists to honour env overrides we must freeze anyway |
| `pxq6_pol_p6` | `:317-346` | the id-252 policy struct: anchor read `:325-327`, `eff = anch*sub[nib]` `:329-334`, code addressing `:336-340` |
| `pxq6_ldcodes` | `:436-464` | one `uint4` load of a row's 16 code bytes |
| `pxq6_panel_stride` / `pxq6_panel` | `:519-527` | `stride = HDR + kslabs*SLAB`; `W + (e*panels+p)*stride`. **Global K and absolute row appear nowhere** — this is why a shard is a first-class tensor |
| `pxq6_dot32`, `pxq6_acc2`, `pxq6_pairx` | `:634-674` | the decoder: 32 K-values, one row, one thread |
| `k_pxq6_dequant_matrix<POL,dst_t>` | `:681-726` | store-coalesced dequant via `__shared__ dst_t tile[64][34]` (`:695,:719-725`) |
| `k_pxq6_mmv<POL,MODE,VECX>` | `:914-971` | the decode GEMV |
| `k_pxq6_gemm_grouped<POL,RAG,PIPE>` | `:2517-2625` | the fused tile GEMM (kept, default off, §4.3) |
| `pxq4_tile_info`, `k_pxq_tiles_2d` | `pxq4.cuh:30-35, 46-57` | device-side tile map, capture-legal by design (`pxq4.cuh:41-45`) |
| (v2) `k_pxq6_gemm_grouped_wmma`, `pxq6_wmma_*` | `:2884-3010, 3038-3070` | sm_70 HMMA twin |
| (v2) `k_pxq6_gemm_gufuse`, `pxq4_glu_apply` | `:2631-2762`, `pxq4.cuh:87-110` | fused up+gate+SiLU |

**Why this is safe to vendor:** `pxq6.cuh` is 3601 lines and contains exactly 10 `ggml_`/`GGML_`
tokens — 4 in comments, 6 in one host-side type mapper (`pxa_pxq_fmt`, `:3335-3345`). The
`__global__`s take `const uint8_t*`, `const half*`, `float*`, ints and a caller-supplied
`cudaStream_t`. There is no `ggml_tensor`, no pool, no ggml stream. The 4 `__CUDA_ARCH__` guards in
the file (`:2893, :2917, :3083, :3218`) are all on WMMA kernels.

**The ~60 edited lines**, enumerated so a re-sync is mechanical:
1. drop `#include "pxa-enhance.cuh"` and inline `pxa_gate_default` defaults for the `PXQ6_GATE`
   macros (`:96-247`) as compile-time constants — a stray env var must not be able to change vLLM
   numerics silently;
2. delete `pxa_pxq_fmt` (`:3335-3345`) and the `PXQ6_PICK*` pickers (`:3375-3460`); instantiate
   `pxq6_pol_p6` directly (this also drops the `pxq23.cuh` include and the P1/P2/P3 tables);
3. `dst_t`-template the GEMM store (`:2621-2623`) — the sibling `k_pxq6_gemm_gufuse` is already
   `dst_t`-templated (`:2631`), so this is a copy of an existing pattern;
4. fp16 activations in the mmv: `xs[idx] = x[idx]` (`:930`) → `__half2float(xh[idx])`, and the fp32
   store (`:968-969`) → `dst_t`; ~6 lines;
5. `CUDA_CHECK` → `C10_CUDA_KERNEL_LAUNCH_CHECK()`;
6. delete `pxq6_ksplit_workspace` (`:2480-2494`) — raw `cudaMalloc`, explicitly declines under stream
   capture. The K-split kernels are bit-identical to the unsplit form (`:26-31`); omitting them costs
   occupancy only. (v2 may reinstate them against our preallocated workspace.)

### 4.2 Op surface

```cpp
TORCH_LIBRARY(pxq4, m) {
  m.def("gemv_out(Tensor(a!) out, Tensor x, Tensor slab, Tensor anchor, int panels, int kslabs) -> ()");
  m.def("gemm_out(Tensor(a!) out, Tensor x, Tensor slab, Tensor anchor, int panels, int kslabs) -> ()");
  m.def("dequant_out(Tensor(a!) w,  Tensor slab, Tensor anchor, int panels, int kslabs) -> ()");
  // v2:
  m.def("gate_up_silu_out(Tensor(a!) out, Tensor x, Tensor sg, Tensor ag, Tensor su, Tensor au, int panels, int kslabs) -> ()");
  m.def("embed_gather(Tensor(a!) out, Tensor ids, Tensor slab, Tensor anchor, int panels, int kslabs) -> ()");
}
```

Every op: `Tensor(a!)` alias annotation on the output, `-> ()` return, plus a python-side
`@torch.library.register_fake` returning `None` (the pattern at `vllm/_sm70_ops.py:287-315`). **The
alias annotation and the fake kernel are the two things most likely to be forgotten**; without them
tracing fails before capture is even attempted.

Capture-safety rules enforced by construction: no allocation inside any op (workspace is reserved in
`process_weights_after_loading`), no `cudaSetDevice`, no `cudaMalloc`, no host sync, no `getenv` at
call time, launch config derived from the python ints passed in.

### 4.3 Kernel dispatch policy — and why prefill dequants

| regime | kernel | rationale |
|---|---|---|
| M ≤ `mmv_max` (default 8) | `pxq4::gemv_out` = `k_pxq6_mmv`, fp16-staged | this is the regime that decides the tok/s headline. Weights read straight from PXQ4 → 3.32 GiB/GPU/token |
| M > `mmv_max` | `dequant_out` → `torch.mm` (cuBLAS HMMA) | **our own measurement says so.** `ggml-cuda.cu:4436-4444` records the fused tile at **−18.6% on sm_70** vs coalesced-dequant + cuBLAS, and states the corollary explicitly: a fix that coalesces the dequant and *keeps* cuBLAS's HMMA should dominate on sm_70. `k_pxq6_gemm_grouped` is a scalar `__hfma2` tile with no tensor cores (`:2594-2611`) |
| v2 experiment | `gemm_out` (fused tile) and `gemm_wmma_out` | both compiled in, both default-off behind `PXQ4_PREFILL=fused|wmma|dequant`. Promotion requires a measured win on a leased card |

This is the one place where "maximum performance" and "fused everywhere" diverge, and the evidence
points away from fusing. **Crucially it does not cost VRAM residency**: the dequant target is a single
per-rank tile, sized to the largest shard — at TP=4 that is `ffn_gate/up` 4352×5120 fp16 = **42.5
MiB**, at TP=2 it is 8704×5120 = 85 MiB. Weights remain PXQ4 in VRAM. The extra traffic (write 42.5
MiB + read it back) is amortized over M tokens and is a ~2% overhead at M=512, ~13% at M=64.

`mmv_max` is not a guess to be shipped as one: `policy.py` runs a per-shape crossover sweep at warmup
(the fork already has a precedent, `vllm/model_executor/warmup/awq_sm70_warmup.py`) and caches the
result keyed by `(N, K, dtype)`. Default 8 comes from ggml's own `PXA_PXQ4_2D_MAX_NY`
(`ggml-cuda.cu:4021`) and is expected to be **too low for vLLM**, whose continuous batching decodes M
in the tens (§9 risk 1).

### 4.4 Shared-memory feasibility (INFERENCE from shape arithmetic, not measured)

`k_pxq6_mmv` stages the whole `x` vector in dynamic smem, capped at 46 KB (`ggml-cuda.cu:4262`).
Per-rank K at TP=4: `ffn_down` 4352 → 17.4 KB; `ffn_gate/up` 5120 → 20.5 KB; `attn_output` 1536 → 6 KB.
At TP=2: `ffn_down` 8704 → 34.8 KB. **Every tensor fits at TP=4 and TP=2.** At TP=1 `ffn_down`
(K=17408 → 69.6 KB) does not — TP=1 must fall back to the dequant path or to the (deleted) S-split
kernels. TP=1 is not a target deployment.

### 4.5 What we deliberately do NOT port

* `pxq-mmvq.cuh` / the ggml MMVQ family — it lives inside `mmvq-templates.cuh`, keys on `ggml_type`
  template params and consumes `block_q8_1` (`pxq-mmvq.cuh:134`). Porting it means porting ggml's q8_1
  activation quantizer. It is also explicitly not bit-exact (`pxq-mmvq.cuh:16-17`) and default-off.
* The int8/DP4A prefill twin (`pxq6i8.cuh`) — sm_61-first; on sm_70 the fp16 path is the right one.
* ggml's MMQ traits: `mmq.cu` contains **no** PXQ code at all (only the Volta cuBLAS threshold at
  `mmq.cu:250-267`). Keeping PXQ out of trait machinery built for row-contiguous block quants was a
  deliberate decision argued at `pxq6i8.cuh:3-14` — and the same argument is why we must not use
  vLLM's GGUF loader (`quantization/gguf.py`).
* The AWQ-specific substrate we do **not** inherit and must not assume in projections: the TP2 tile
  all-reduce overlap (`awq_sm70_gemm.cu:3208-3290`, gated on `_awq_sm70_prepared`,
  `linear.py:107-109`), `VLLM_SM70_AWQ_MLP_ENGINE` (default off), `VLLM_SM70_AWQ_PREFILL_EXACT_DENSE`
  (default on), the autotune cache, and `warmup/awq_sm70_warmup.py`. Each is a separate later port.

### 4.6 v2 kernels (designed now, built after v1 passes its gates)

1. **`gate_up_silu_out`** — `k_pxq6_gemm_gufuse` (`:2631-2762`) + `pxq4_glu_apply` (`pxq4.cuh:87-98`,
   which already implements exactly the `unary==0` SiLU-swiglu branch). A Qwen dense FFN *is* an
   up+gate pair with a SiLU-swiglu epilogue. Saves one full 17408-wide fp16 intermediate round-trip
   per token per layer.
2. **`embed_gather`** — PXQ4 vocab tier. A row `r` is `panel r/64`, row-in-panel `r%64`: anchor at
   `hdr + 2*(r%64)`, per slab a scale byte at `slab + r%64` and 16 code bytes at `slab + 64 + 16*(r%64)`.
   ~40 LOC. Drops `token_embd` from 606 MiB/GPU fp16 to 161 MiB/GPU (capacity only — an embedding
   gather is not bandwidth-bound), taking Policy MAX residency to **3.412 GiB/GPU**.
3. **WMMA prefill** (`:2912`) — sm_70-gated HMMA, currently reachable only from the MoE driver
   (`ggml-cuda.cu:5056`) and env-default off (`pxq6.cuh:233-237`). Explicitly not bit-exact
   (`pxq6.cuh:53-59`). Wire it to the one-expert degenerate tile map and race it against
   dequant+cuBLAS on a leased card.

### 4.7 Build — without rebuilding vLLM

The extension depends on **libtorch only**. It never includes a vLLM header and never links
`_C.abi3.so`; it registers into its own `torch.ops.pxq4` namespace. Consequences: vLLM is untouched,
and the wheel survives a vLLM upgrade as long as the torch ABI holds.

```cmake
cmake_minimum_required(VERSION 3.26)
project(pxq4_sm70 LANGUAGES CXX CUDA)
find_package(Torch REQUIRED)                       # from /opt/vllm-venv
set(CMAKE_CUDA_ARCHITECTURES 70)                   # V100 only; no fat binary
add_library(pxq4_sm70 SHARED csrc/pxq4_ops.cpp csrc/pxq4_kernels.cu csrc/pxq4_workspace.cu)
target_compile_options(pxq4_sm70 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70 --expt-relaxed-constexpr -lineinfo -O3>)
target_compile_definitions(pxq4_sm70 PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_link_libraries(pxq4_sm70 PRIVATE ${TORCH_LIBRARIES})
```

Operational constraints (FACT from recon): nvcc 12.8, gcc, cmake, ninja and torch 2.10.0+cu128 are
present in the container image; but **the production container's overlay is 100% full (0 bytes
available)** and the DGX root filesystem is full. Therefore: build in a **fresh** container from the
same image with a volume mounted under `/path/to/models`, never inside `vllm-qwen38-27b-cyber-1`, never
writing to `/` or host `/tmp`. Also note `site-packages/vllm` is a *copied* install, not
editable-linked to `/opt/1Cat-vLLM` — edits to `/opt/1Cat-vLLM` are inert at runtime, which is another
reason the no-patch design is the right one.

`_GLIBCXX_USE_CXX11_ABI` must match `torch._C._GLIBCXX_USE_CXX11_ABI`; mismatch is the classic
"undefined symbol: _ZN3c10..." failure at `load_library` time.

---

## 5. Weight loading: GGUF → vLLM parameters, mixed-type

### 5.1 The census we are converting (FACT, parsed from the artifact)

866 tensors, GGUF v3, 15,708,774,400 B of tensor data.

| id | type | count | bytes | disposition in this design |
|---:|---|---:|---:|---|
| 252 | **PXQ4** | 325 | 12,231,950,336 | **byte passthrough**, reshaped to `[R/64, K/32, 1088]` + `[R/64, 64]` |
| 8 | Q8_0 | 132 | 1,621,032,960 | `output.weight` (1,350,860,800) → **re-encoded PXQ4**; the rest (attn_k/v, ssm_alpha/beta, eh_proj) → **fp16** |
| 14 | Q6_K | 1 | 1,042,944,000 | `token_embd` → fp16 (v1) / **PXQ4** (v2) |
| 39 | MXFP4 | 48 | 802,160,640 | `ssm_out` → **re-encoded PXQ4** from the Q8_0 parent |
| 0 | F32 | 360 | 10,686,464 | passthrough (norms, conv1d, ssm_a/dt) |

No F16 tensors exist (the `attn_gate_head=f16` backbone rule never fires — every `attn_gate` here is
per-channel, ne[1]=6144). No expert tensors. Backbone KVs in the file:
`pxa.pxq.backbone_rev=2`, `pxa.pxq.backbone_map=attn_q,attn_qkv,attn_output,attn_gate_ch,shexp,ffn_dense=tier+1;attn_k,attn_v=q8_0;attn_gate_head=f16;token_embd=q6_k;output=q8_0`,
plus — critically — the 16-entry `pxa.pxq6.book` and `pxa.pxq6.sub` tables. **Read the tables from the
file; do not hard-code them** (they can be overridden at build time via `PXA_PXQ6_BOOK`/`PXA_PXQ6_SUB`,
`llama-quantize.cpp:1980-1983`). The converter asserts they match the vendored `PXQ6_BOOK_INIT` /
`PXQ6_SUB16_INIT` compiled into the `.so`, and refuses to convert if not.

Layout confirmed byte-exactly against all six PXQ4 shapes: `bytes = (R/64)*(128 + (K/32)*1088)`,
no inter-tensor padding, all 325 tensors pass `R%64 && K%32`.

### 5.2 Reader

Own ~200-line GGUF parser (already prototyped as `ggufscan.py`/`tc.py` during recon).
**Do not use the `gguf` PyPI package**: `gguf.GGMLQuantizationType(252)` raises `ValueError` inside
`GGUFReader._build_tensors`, which runs over the whole tensor table *in the constructor* — one PXQ4
tensor kills the file open before any tensor is yielded. And do not use `quantization/gguf.py`: its
sharder slices rows assuming per-row-contiguous blocks, which panel interleave violates
(`pxq-cpu.h:5-9`; ggml's own `to_float` is NULL for the same reason, `ggml.c:1407-1414`).

### 5.3 Name mapping — the rule, not a guess

The converter must emit the **exact tensor names the incumbent AWQ checkpoint uses**, because that
checkpoint demonstrably loads into `qwen3_5.py` today. `tools/pxq4_names.py` is therefore *generated*
by diffing against `/path/to/hf/<reference-hf-model>/model.safetensors.index.json`
and the converter **fails loudly on any name it cannot map**. Known structural points:

* `packed_modules_mapping` (`qwen3_5.py:665-675, 823-826`): `qkv_proj←[q_proj,k_proj,v_proj]`,
  `gate_up_proj←[gate_proj,up_proj]`, `in_proj_qkvz←[in_proj_qkv,in_proj_z]`,
  `in_proj_ba←[in_proj_b,in_proj_a]`. We emit the **unfused** names.
* GDN: ggml `attn_qkv` (10240 = 2048+2048+6144) is `in_proj_qkv`; ggml `attn_gate` (6144) is
  `in_proj_z`. `in_proj_ba` has no PXQ4 counterpart (it is f32/untouched in the artifact) and is
  served by `UnquantizedLinearMethod`.
* `attn_q` rows=12288 = 2×(24×256) is **gate-fused**, per-head interleaved `[q_h | gate_h]`
  (`llama-build-context.cpp:2003-2007` takes `ggml_view_3d(..., 2*row_size, ..., 0)` for Q and the
  same view at `offset=row_size` for gate; matches `qwen3_next.py:565-567`). **No converter
  permutation is needed** — but a column shard must split each 6144 half separately, which the stock
  merged loader does because the two halves are separate shard ids.
* Every PXQ4 weight becomes two checkpoint entries, `<module>.pxq_slab` and `<module>.pxq_anchor`,
  matching the registered parameter names (the same convention as AWQ's `qweight`/`scales`/`qzeros`).

### 5.4 The two re-encodes (this is what makes the design beat AWQ)

`ssm_out` and `output.weight` are not PXQ4 in the artifact because they are not in the backbone map's
class list. Both are geometrically eligible (`ssm_out` R=5120 %64, K=6144 %32; `output` R=248320 %64,
K=5120 %32) and both are large.

* **Source: the Q8_0 parent gguf, not the artifact.** Re-quantizing `ssm_out` from its MXFP4 form
  would be a double quantization; the PXQ4 file was itself made from Q8_0, so sourcing from Q8_0
  gives these two tensors *identical provenance* to the other 325.
* **Encoder:** build `libpxq4enc.so` from a **copy** of the mgv-wt tree at the pinned commit, adding a
  ~40-line exported shim over `pxq6_quantize_expert(src, dst, R, K, imx, tier, row0)`
  (`pxq6-quantize.inc.cpp:287-289`) — which already takes an arbitrary `[R,K]` fp32 block plus an
  absolute row offset, exactly our input shape. Pass `row0 = 0` for a whole tensor; `row0` seeds the
  deterministic tie-break `pxq_tie_take_hi` (`:49`, used `:230`, warned at `:416-418`), so it must be
  the true absolute row. **The production tree is not modified** — this is a copy, and the shim is
  additive.
* Sizes: PXQ4 `ssm_out` = 16,721,920 B ×48 = 802,652,160 B, i.e. **within 0.06% of the MXFP4 it
  replaces** — so this is not a size win, it is a *single-kernel-runtime* win that also removes the
  unverified "does ggml type-39 packing match the fork's MXFP4 convention?" risk entirely.
  PXQ4 `output.weight` = 675,927,040 B vs 1,350,860,800 Q8_0 vs 2,542,796,800 fp16 — **the big one**:
  at TP=4 the lm_head is 161 MiB/rank read *every decode step*, versus 606 MiB/rank if fp16. The
  incumbent AWQ config's 311-entry ignore list contains `lm_head`, so **they carry an fp16 lm_head and
  structurally cannot take this win.**
* Quality: this creates a *new artifact* and therefore needs the standard PXQ quality gate before
  deployment (same class of work as the open Sky-35B gate). That gate is CPU-side perplexity/
  same-top-token on a leased window — it is real work and it is outside this design's scope.

### 5.5 What each tier gets at runtime

| tier | tensors | vLLM method | why |
|---|---|---|---|
| PXQ4 | 325 + 48 (`ssm_out`) + 1 (`lm_head`) | `PXQ4LinearMethod` | our kernel |
| fp16 | attn_k, attn_v (34), ssm_alpha/beta (96), eh_proj (1) | `UnquantizedLinearMethod` | 356 MB + 47 MB + 105 MB total; per-GPU rounding error. The backbone keeps k/v at 8-bit *for quality reasons* (`llama-quantize.cpp:1355-1371`); promoting them to fp16 is strictly safer and costs 42.5 MiB/rank each |
| fp16 (v1) / PXQ4 (v2) | token_embd | `None` → `UnquantizedEmbeddingMethod` / `PXQ4EmbeddingMethod` | capacity only |
| f32 | 360 norms/conv1d/ssm_a/dt | untouched | not `LinearBase` |

`get_quant_method` must return `UnquantizedLinearMethod()` — never `None` — for a skipped
`LinearBase`; `None` raises (`linear.py:492-495`). `None` is correct only for embeddings
(`vocab_parallel_embedding.py:479-482`).

---

## 6. MoE

**This model has no MoE, and there is no MoE work in this project.** Verified twice, independently, in
the artifact itself: 866 tensors, zero `*_exps` tensors, zero expert KVs, `ffn_gate/up/down` dense on
all 65 blocks. The brief's "40×256 routed experts" belongs to **Sky-35B**, a different model. No
`FusedMoEMethodBase`, no `RoutedExperts` path, nothing to port. **Effort delta: zero. VRAM cost of a
dequant fallback: not applicable — there is nothing to fall back from.**

Forward-looking note, since it is nearly free to state: the vendored kernels *are* the MoE kernels.
`pxq6_panel(W, e, panels, p, kslabs)` (`pxq6.cuh:523-527`) has an expert index built in, and our own
`pxa_pxq_gemm_2d` (`ggml-cuda.cu:4493-4570`) already serves a plain 2D matmul by supplying a
degenerate one-expert tile map via `k_pxq_tiles_2d` — which is exactly the call pattern this vLLM port
uses. If PXQ4 is ever pointed at Sky-35B, the additional work is a `FusedMoEMethodBase` subclass plus
wiring `k_pxq6_gemm_gufuse` (`:2631-2762`) and `k_pxq6_gemm_down_scat` (`:2766-2880`) to vLLM's
expert-id/tile map — the kernels already exist and already expect that shape. Should a dequant
fallback ever be needed there, it must be **per-active-expert into a transient workspace**
(`n_active_experts × expert_shard_bytes`), never resident fp16 experts; sizing that requires Sky-35B's
per-expert shapes, which this study did not measure.

---

## 7. Performance model (PROJECTION — assumptions stated)

**Calibration.** The incumbent's measured 92.8 tok/s peak single-stream against a 4.64 GiB/GPU
resident W4A16 model implies an effective decode bandwidth of `4.64 GiB × 92.8 = 462 GB/s per card` —
51% of the V100's 900 GB/s, matching the "both engines run at ~50% of theoretical HBM" fact in the
brief. **This design's projections reuse that same 462 GB/s constant**, which makes them a like-for-
like scaling of a measured point on this exact box rather than a fresh guess.

**Assumptions (each one is a way the projection can be wrong):**
1. decode stays weight-bandwidth-bound, i.e. per-token bytes = per-GPU resident weight bytes minus
   the embedding table;
2. `k_pxq6_mmv` reaches the same fraction of peak bandwidth as their TurboMind GEMV — it is a
   different kernel with a different smem strategy (it stages all of `x`), so this is the weakest
   assumption in the model (§9 risk 1);
3. TP=4 all-reduce overhead is unchanged (we do not inherit their TP2 tile-overlap all-reduce);
4. MTP/ngram speculation is off; our llama.cpp-side MTP gains (+24% measured there) are not counted.

| configuration | resident /GPU | read /GPU /token | projected tok/s | delta vs incumbent |
|---|---|---|---|---|
| incumbent AWQ (MEASURED) | 4.64 GiB | — | **92.8 peak / 57.4 median** | — |
| Policy A: PXQ4 + fp16 tail | 4.864 GiB | 4.271 GiB | 100.7 | +8.5% |
| **Policy MAX v1** | **3.913 GiB** | **3.321 GiB** | **129.6** | **+40%** |
| Policy MAX + PXQ4 vocab (v2) | 3.412 GiB | 3.321 GiB | 129.6 | +40% (capacity only) |
| Policy MAX, TP=2 (Unraid 16 GB) | 7.815 GiB | 6.631 GiB | 64.9 | n/a |

Largest per-GPU items at TP=4 under Policy MAX: `ffn_gate` 734 MiB, `ffn_up` 734 MiB, `ffn_down` 734
MiB, `token_embd` 606 MiB (fp16, v1), `attn_qkv` 319 MiB, `attn_gate` 191 MiB, `ssm_out` 191 MiB,
`lm_head` 161 MiB.

Second-order upside not in the table, all requiring measurement: the fused gate_up+SiLU kernel (one
17408-wide fp16 round trip per token per layer), the WMMA prefill, and eventually MTP (the fork
already ships `qwen3_5_mtp.py` spec decode and the artifact already contains the `blk.64` MTP block —
we would be the only PXQ4 consumer with a 4-bit MTP block, since the incumbent ignores all `mtp.*`).

---

## 8. Staged plan — every stage has a checkpoint verifiable WITHOUT a model run

Stages S0–S3 need **no GPU at all** except a few seconds of device time for a single tiny allocation
in S2/S3; those two are the only pre-model GPU touches and must be scheduled inside a lease window
(this study performs none).

| stage | work | checkpoint (pass criterion) | GPU? |
|---|---|---|---|
| **S0** | repo skeleton, CMake, empty `TORCH_LIBRARY`, wheel, entry point | `python -c "import vllm; import pxq4_vllm"` inside a *fresh* container registers `"pxq4"` in `QUANTIZATION_METHODS`; `vllm serve --quantization pxq4` reaches "unknown model dir" rather than "unknown quantization" | no |
| **S1** | converter + `libpxq4ref.so` (CPU dequant reference) | (a) byte-accounting reproduces all 866 on-disk tensor sizes exactly, `bytes=(R/64)(128+(K/32)1088)` for all 325 PXQ4; (b) `pxa.pxq6.book`/`.sub` from the file == the vendored table constants; (c) every emitted name matches the incumbent AWQ index.json key set (modulo `.pxq_slab`/`.pxq_anchor` vs `.qweight`/`.scales`) | no |
| **S2** | vendored dequant kernel + `dequant_out` | **bit-exact gate**: `torch.equal(fp32(pxq4::dequant_out(t)), pxa_pxq_dequant_2d(t))` for ≥20 tensors covering all six shapes. Must be EXACT — `pxq-cpu.h:16-18` states the dequant is the parity-locked contract (fp32 `eff*book` products), and this single test validates the layout reader, panel arithmetic and table constants, i.e. >90% of the port's risk | seconds |
| **S3** | TP repack proof | shard-then-dequant == dequant-then-shard, **bit-identical in fp32**, for a column split at a 64-row boundary and a K split at a 32-column boundary, at TP=2 and TP=4. This proves the byte-gather is a permutation, not a requantization | seconds |
| **S4** | `gemv_out` + `gemm_out` + one `PXQ4LinearMethod` in isolation | build a standalone `ColumnParallelLinear`/`RowParallelLinear` (no model) with a converted `ffn_gate` shard; compare against `dequant→torch.mm` with fp16 tolerance (max-abs and cosine, not a hash — the GEMM/mmv paths snap products to fp16 and are explicitly *not* bit-exact vs the CPU reference). Also assert `layer._sm70_f16_prepared` is absent and `apply()` performs no `cudaMalloc` (poison the allocator) | minutes |
| **S5** | cudagraph capture | capture `apply()` under `torch.cuda.graph` and replay 100×, byte-identical outputs; then `torch.compile` with `cudagraph_mode=FULL_AND_PIECEWISE` on a 2-layer stub. **This is where the out-of-tree namespace assumption is tested** (§9 risk 3) | minutes |
| **S6** | one full transformer block (1 GDN + 1 full-attn) | logits of the converted block vs the same block run in fp16 from the Q8_0 parent; per-token top-1 agreement and KL. Catches name-mapping and merged-shard-order errors that S4 cannot | minutes |
| **S7** | whole model, TP=4, greedy | same-top-token vs the llama.cpp PXQ4 engine on a fixed 500-prompt set — the two engines share the *same weight bytes*, so disagreement is a port bug, not a quantization artifact. This is the strongest end-to-end gate available and it needs no reference fp16 model | lease window |
| **S8** | perf: `mmv_max` crossover sweep, v2 kernels, autotune cache | throughput vs the incumbent on the same harness (790-sample methodology) | lease window |

The re-encode of `ssm_out`/`output.weight` (§5.4) runs between S1 and S6 and carries its own,
separate **quality gate** (perplexity + same-top-token vs the Q8_0 parent) — it changes the model, not
just the runtime, and must not be conflated with the port's correctness gates.

---

## 9. Risks

Ranked by expected damage × probability. Each has a detection point and a kill criterion.

1. **`k_pxq6_mmv` does not hold its efficiency at vLLM's batch shapes.** Our mmv family is tuned for
   `ny ≤ 8` (`ggml-cuda.cu:4021, :4239`) and stages all of `x` in dynamic smem. vLLM with continuous
   batching decodes M in the tens-to-hundreds; above the crossover we fall to dequant+cuBLAS, which
   re-reads a 42.5 MiB fp16 tile and **gives back the entire bytes-per-token advantage** that
   justifies the project. *Detect:* S8 crossover sweep. *Mitigate:* the `x2` two-rows-per-thread
   variant (`pxq6.cuh:1949-2016`) and the WMMA path both raise the ceiling; a batched-mmv (M in
   registers, weights streamed once) is ~150 LOC and is the real fix. *Kill:* if the crossover lands
   below M≈16 and cannot be raised, the median-throughput case never sees the win and only the
   single-stream number improves.
2. **The VRAM premise depends on work outside vLLM.** Policy A (no re-encode) is a *regression* vs the
   incumbent's 4.64 GiB/GPU. Getting to 3.91 GiB requires re-quantizing `ssm_out` and `output.weight`,
   i.e. a new artifact plus a quality gate on a leased window. *Detect:* immediately — it is an input
   to the plan, not a discovery. *Mitigate:* the encoder shim is 40 LOC over an existing function that
   already accepts arbitrary `[R,K]` blocks. *Kill:* if a 4-bit lm_head fails the quality gate, keep
   it Q8_0 → 4.31 GiB/GPU, projected ~110 tok/s, still ahead but a much thinner margin.
3. **Out-of-tree op namespace under `FULL_AND_PIECEWISE` cudagraph capture — UNVERIFIED.** All the
   evidence we have is for ops in `torch.ops._C` with `_sm70_ops.py`-style fakes; whether
   `torch.ops.pxq4.*` participates identically in piecewise capture alongside `vllm::qwen_gdn_*`
   splitting ops is an assumption. *Detect:* S5, before any model work. *Mitigate:* `register_fake` +
   `Tensor(a!)` annotations (the two most-forgotten pieces); failing that, add our op to
   `splitting_ops` via CLI config, or run `cudagraph_mode=PIECEWISE`. *Kill:* none — worst case is a
   config change with a modest launch-overhead cost.
4. **Silent misalignment.** `round(shard_size // packed_factor)` truncates without raising
   (`linear.py:1055-1056`, `parameter.py:606-609`): a misaligned offset produces a well-formed *wrong*
   panel slice and a model that loads cleanly with subtly wrong logits. *Mitigate:* the explicit
   `%64`/`%32` asserts in `create_weights`, plus S3 and S7. This is the failure mode most likely to
   reach production undetected.
5. **Name/shard-order mapping.** The GDN fused `in_proj_qkvz` ordering, the gate-fused `attn_q`
   interleave, and the `in_proj_ba` split flag are three independent ways to load *plausible* garbage.
   *Mitigate:* generate the map from the incumbent checkpoint's index (§5.3), fail loudly on unmapped
   names, and gate at S6 (one block) rather than S7 (whole model), because a block-level diff localizes
   the error.
6. **Vendor drift.** We copy 500 lines of `pxq6.cuh` at `acf8f245`. Future llama.cpp-side kernel fixes
   will not propagate. *Mitigate:* `PXQ4_VENDOR_SRC_COMMIT` macro, `pxq4_vendor_edits.md`, and S2's
   bit-exact gate re-run on every re-sync (it is a 30-second test).
7. **Container/build friction.** Production overlay is 100% full; DGX `/` is 100% full;
   `site-packages/vllm` is a copy, not editable. *Mitigate:* fresh container + `/path/to/models` volume,
   never touch `vllm-qwen38-27b-cyber-1` beyond `docker exec`-to-read.
8. **fp16 accumulation in the GEMM path.** `k_pxq6_gemm_grouped` snaps products to fp16
   (`pxq6.cuh:2594-2611`); the WMMA twin is explicitly not bit-exact (`pxq6.cuh:53-59`). Long-context
   prefill accumulation error is a real (if bounded) quality question. *Mitigate:* v1 prefill is
   dequant→cuBLAS fp16 with fp32 accumulate, which is *better* than the fused path, not worse.
9. **TP=1 does not fit the mmv smem cap** (K=17408 → 69.6 KB vs 46 KB). Only relevant if someone tries
   a single-card deployment; falls back to dequant+mm automatically. INFERENCE, not measured.
10. **`_sm70_f16_force_enable`** is set on every `qkv_proj`/`out_proj` by
    `_mark_default_sm70_dense_modules` (`qwen3_5.py:159-181`). Resolved as inert (it requires
    `_sm70_f16_prepared`, `linear.py:62-63`), and we set `_sm70_f16_forbidden` defensively — but it is
    one attribute away from AWQ kernels reading PXQ4 bytes, so S4 asserts on it explicitly.

**What would make me abandon the project:** risk 1 landing badly *and* risk 2's quality gate failing
together — that combination leaves a single-stream-only win on a model that no longer fits the VRAM
story. Nothing found in recon suggests either is likely; both are measurable early (S8 and the
re-encode gate respectively), and neither requires the full port to be finished first.

---

## 10. Effort

| component | new LOC | copied LOC |
|---|---:|---:|
| vendored kernel slice (+60 edited) | 60 | 500 |
| new kernels (fp16 mmv, half dequant/gemm, v2 stubs) | 260 | — |
| torch C++ op layer + workspace | 390 | — |
| python: config, linear method, ops, policy | 750 | — |
| offline converter + name map + encoder/reference shims | 820 | — |
| tests (six gates) | 250 | — |
| build files | 110 | — |
| **total** | **~2,640** | **500** |

Sequenced: S0–S3 (converter + bit-exact gates, no GPU) is roughly half the LOC and carries most of the
risk retirement. S4–S6 is the vLLM integration. S7–S8 need leased GPU windows and are where every
number in §7 stops being a projection.
