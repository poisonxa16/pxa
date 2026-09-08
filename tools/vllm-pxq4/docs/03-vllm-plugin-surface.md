# vLLM quantization plugin surface — 1Cat-vLLM fork (as installed in `vllm-qwen38-27b-cyber-1`)

Source read at `/opt/1Cat-vLLM` inside the running container, git HEAD `2ceb15066518e7241cf3c57a71ddd39168cbc675` (Sat Aug 15 15:39:26 2026 +0100) — matches the reported `v0.1.dev1+g2ceb15066`. All line numbers below are from that tree. Nothing was modified, no container was restarted, no GPU was touched.

## 0. Verdict up front

There is **no blocker in the plugin surface**. `register_quantization_config("pxq4")` is a real, load-bearing hook (`quantization/__init__.py:57-104`, consumed at `:230`), the linear TP sharding is driven entirely by three *parameter attributes* (`output_dim`, `input_dim`, `packed_dim`+`packed_factor`) that stock loaders read with `getattr`, and PXQ4's 64-row panel / 32-column slab geometry maps onto those attributes exactly (`packed_factor=64` on the output dim, slab-count units on the input dim). No custom weight_loader is required.

**One real trap, with a clean fix** (FACT, `qwen3_5.py:127-157,207-229`): unless our config exposes an "ignore"-style list naming `linear_attn.in_proj_b`/`in_proj_a`, the GDN input projection is built as a *single fused* `MergedColumnParallelLinear` with `output_sizes=[key_dim, key_dim, value_dim, value_dim, num_v_heads, num_v_heads]`. `num_v_heads = 48`; at TP=4 those trailing shards are 12 rows each — **not** multiples of 64, so the panel-packed shard arithmetic (`shard_offset // packed_factor`) truncates and the layer cannot be PXQ4-sharded. Fix is one attribute (see §6.3).

Second trap (FACT, `linear.py:56-96,805,1795`): `_maybe_sm70_dense_forward()` is checked *before* `quant_method.apply()` in every `forward()`. It is gated on `layer._sm70_f16_prepared`, which is only ever set by `UnquantizedLinearMethod.process_weights_after_loading` (`linear.py:408`). Our method must never set that attribute; if it does, our kernel is silently bypassed.

---

## 1. The registry — what must be registered for the method string "pxq4"

FACT, `vllm/model_executor/layers/quantization/__init__.py`:

- `:12-45` `QuantizationMethods = Literal[...]` — a static `Literal` of built-in names. **Not extensible at runtime** (it is a typing construct).
- `:46` `QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))` — the *mutable* runtime list. This is what gets appended to.
- `:54` `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}` — the out-of-tree dict.
- `:57` `def register_quantization_config(quantization: str)` — decorator factory. Its `_wrapper` (`:83-102`):
  - `:92` appends the name to `QUANTIZATION_METHODS`,
  - `:94-95` appends it to `current_platform.supported_quantization` ("Automatically assume the custom quantization config is supported") — so no platform allow-list edit is needed,
  - `:97-100` **hard requirement**: `issubclass(cls, QuantizationConfig)` else `ValueError`,
  - `:101` stores it in `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`.
- `:107` `def get_quantization_config(quantization: str) -> type[QuantizationConfig]` — `:108-109` rejects anything not in `QUANTIZATION_METHODS`; builds the built-in `method_to_config` dict (`:192-272`); `:279` `method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)` — **customized entries override built-ins**; `:281` returns the class.

### How our module gets imported in time
FACT, `vllm/plugins/__init__.py:14` `DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"`, `:69` `load_general_plugins()`, gated by `envs.VLLM_PLUGINS` (`:32`).
Call sites: `engine/arg_utils.py:749` (inside `create_model_config`, immediately after `resolve_quantization_config` and **before** `ModelConfig` is constructed), `arg_utils.py:2772` (`AsyncEngineArgs.add_cli_args`, so `--quantization pxq4` is accepted by the CLI parser), `v1/engine/core.py:108`, `v1/worker/worker_base.py:247` (re-loaded in every engine-core and worker process — important for TP=4, each rank re-registers).

**Deliverable:** a pip-installable package `pxq4_vllm` with entry point
```
[project.entry-points."vllm.general_plugins"]
pxq4 = "pxq4_vllm:register"
```
whose `register()` imports the module containing `@register_quantization_config("pxq4") class PXQ4Config(QuantizationConfig)`. No fork patch required. (INFERENCE: this is the only supported no-patch path; the alternative is adding `"pxq4"` to `QuantizationMethods` + `method_to_config` in the fork, which is a 2-line patch if we would rather vendor it.)

---

## 2. `QuantizationConfig` ABC — exact signatures in this fork

FACT, `vllm/model_executor/layers/quantization/base_config.py:70-215`.

```python
class QuantizationConfig(ABC):
    def __init__(self):                     # :73  — MUST call super().__init__();
        super().__init__()                  #       it sets self.packed_modules_mapping = {}
        self.packed_modules_mapping: dict[str, list[str]] = dict()   # :76

    @abstractmethod
    def get_name(self) -> QuantizationMethods: ...                   # :78-81  (instance method)

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...     # :83-86  (instance method)

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int: ...                          # :88-97  (70 == Volta)

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]: ...                     # :99-103

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig": ...   # :105-109

    @abstractmethod
    def get_quant_method(self, layer: torch.nn.Module, prefix: str
                        ) -> QuantizeMethodBase | None: ...          # :150-163
```
Six abstract members. Everything else is optional, with defaults:

| member | line | default | do we need it |
|---|---|---|---|
| `override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None) -> QuantizationMethods \| None` | `:111-130` | `None` | **yes** — this is how a checkpoint self-selects `pxq4` (see §5) |
| `get_from_keys(config, keys)` / `get_from_keys_or(config, keys, default)` | `:132-148` | helpers | convenience |
| `get_cache_scale(name) -> str \| None` | `:165` | `None` | no |
| `apply_vllm_mapper(hf_to_vllm_mapper)` | `:168` | no-op | no |
| `maybe_update_config(model_name, hf_config=None, revision=None)` | `:181` | no-op | no |
| `is_mxfp4_quant(prefix, layer) -> bool` | `:200-215` | `False` | no |

`QuantizeMethodBase` (`:19-55`) — parent of `LinearMethodBase`:
- class attr `uses_meta_device: bool = False` (`:25`)
- `@abstractmethod create_weights(self, layer, *weight_args, **extra_weight_attrs)` (`:27-34`)
- `@abstractmethod apply(self, layer, *args, **kwargs) -> torch.Tensor` (`:36-41`)
- optional `embedding(self, layer, *args, **kwargs)` (`:44`) — required *only* if the method is returned for a bare `VocabParallelEmbedding` (`vocab_parallel_embedding.py:490-497` raises `NotImplementedError` otherwise; `method_has_implemented_embedding` at `base_config.py:58-67` is how it checks). Returning `None` from `get_quant_method` for embedding layers gives `UnquantizedEmbeddingMethod` (`vocab_parallel_embedding.py:479-482`) — that is what we want.
- optional `process_weights_after_loading(self, layer: nn.Module) -> None` (`:50-55`)

Reference implementation to read alongside: `AWQConfig`, `awq.py:136-347` (`get_name` `:170`, `get_supported_act_dtypes` `:173` returns `[torch.half]`, `get_min_capability` `:177` returns **70** when `sm70_tm.use_turbomind(...)`, `get_config_filenames` `:187`, `from_config` `:195`, `get_quant_method` `:204`).

---

## 3. `LinearMethodBase` — exact signatures

FACT, `vllm/model_executor/layers/linear.py:286-325`:

```python
class LinearMethodBase(QuantizeMethodBase):
    @abstractmethod
    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,      # K on this rank
                       output_partition_sizes: list[int],  # per-logical-matrix N on this rank
                       input_size: int,                    # global K
                       output_size: int,                   # global N
                       params_dtype: torch.dtype,
                       **extra_weight_attrs): ...          # :290-314

    @abstractmethod
    def apply(self, layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor: ...   # :316-325
```

`extra_weight_attrs` always contains `weight_loader` — the layer's bound loader, chosen at the call site:
- `ColumnParallelLinear.__init__` → `linear.py:697-709`
- `RowParallelLinear.__init__` → `linear.py:1695-1707`
- both pass `weight_loader = self.weight_loader_v2 if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader`

`WEIGHT_LOADER_V2_SUPPORTED` is a plain list at `linear.py:193-210` (opt-in by class *name*), with a decorator `register_weight_loader_v2_supported_method(cls)` at `linear.py:213-216`. **Our `PXQ4LinearMethod` should NOT be registered there** — see §4.4.

What `create_weights` must do: build `torch.nn.Parameter` (or `BasevLLMParameter`) objects, set the sharding attributes on them, and `layer.register_parameter(name, param)`. Canonical minimal example: `AWQLinearMethod.create_weights`, `awq.py:359-433` — three params (`qweight`, `qzeros`, `scales`), each carrying `input_dim` / `output_dim` / `packed_dim` / `packed_factor`, then `register_parameter` at `:431-433`.

`apply` contract we should copy (CUDA-graph-safe, preallocated out): `sm70_turbomind.apply_prepared_linear`, `sm70_turbomind.py:284-339` — reshape `x` to 2-D, `torch.empty((M, N), dtype=x.dtype, device=x.device)`, call an `_out` custom op, `out.add_(bias)`, reshape back.

`process_weights_after_loading(layer)` is where the repack to kernel layout happens (`sm70_turbomind.py:158-283`, four `prepare_*_linear` functions; state stashed on the layer via `setattr(layer, "_sm70_turbomind_linear", state)` `:151`). For PXQ4 this step is *almost free*: the on-disk layout **is** the kernel layout, so it only needs to free the loader parameters into plain `nn.Parameter(requires_grad=False)` and cache derived scalars.

---

## 4. THE TP HOOK — how quantized weights get sharded (the important part)

### 4.1 Where partition sizes come from
FACT:
- `ColumnParallelLinear` computes `self.output_size_per_partition` / `self.output_partition_sizes` and passes them into `create_weights` at `linear.py:697-704` (`input_size_per_partition=self.input_size` — column-parallel never splits K).
- `RowParallelLinear` passes `input_size_per_partition=self.input_size_per_partition`, `output_partition_sizes=self.output_partition_sizes` at `linear.py:1695-1701` (row-parallel never splits N).
- `MergedColumnParallelLinear.__init__` `linear.py:855-885`: stores `self.output_sizes`, and **asserts** `all(output_size % self.tp_size == 0 for output_size in output_sizes)` (`:872`).
- `QKVParallelLinear` is a `ColumnParallelLinear` subclass (`linear.py:1207`) with `_get_shard_offset_mapping` / `_get_shard_size_mapping` and `num_kv_head_replicas`.

### 4.2 The actual sharding mechanism — attributes, not methods
There is **no** "custom sharder" API. Sharding is performed by the *layer's* weight loader, which introspects the parameter with `getattr`. A quant method declares its layout by setting attributes on the parameters it creates in `create_weights`.

**v1 path (plain `Parameter` + `set_weight_attrs`, or any param the layer's `weight_loader` sees):**
- `ColumnParallelLinear.weight_loader` `linear.py:753-788`:
  `output_dim = getattr(param, "output_dim", None)` (`:754`); if set and not `is_sharded_weight`, `shard_size = param_data.shape[output_dim]; start_idx = self.tp_rank * shard_size; loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)` (`:775-778`).
- `RowParallelLinear.weight_loader` `linear.py:1728-1761`: identical with `input_dim` (`:1729`, narrow at `:1749-1752`).
- `MergedColumnParallelLinear.weight_loader` `linear.py:924-1098`: computes `shard_offset = sum(self.output_sizes[:id])`, `shard_size = self.output_sizes[id]`, both `//= self.tp_size` (`:1041-1044`), then **the packing hook**:
  ```
  packed_dim = getattr(param, "packed_dim", None)              # linear.py:1053
  if packed_dim == output_dim:                                 # :1054
      shard_size   = round(shard_size   // param.packed_factor) # :1055
      shard_offset = round(shard_offset // param.packed_factor) # :1056
      shard_size, shard_offset = adjust_marlin_shard(param, shard_size, shard_offset)  # :1058
  ```
  (the same block for the fused-on-disk branch at `:1013-1018`). `adjust_marlin_shard` at `linear.py:218-232` multiplies both by `param.marlin_tile_size` when present — **this is the precedent for a non-row-contiguous tiled layout**, and it is exactly the shape of hook PXQ4 needs.
- `QKVParallelLinear.weight_loader` `linear.py:1417-...`: same structure, offsets from the q/k/v shard maps, same `packed_dim`/`packed_factor` treatment.
- **Escape hatch:** `is_sharded_weight = getattr(param, "is_sharded_weight", False)` (`linear.py:755-761`, `1064-1070`, `1729-1735`) — when true the loader skips `narrow()` entirely and expects the file to already hold this rank's slice. Used by bitsandbytes. Available to us but undesirable (ties the artifact to a TP size).

**v2 path (`BasevLLMParameter` subclasses), `vllm/model_executor/parameter.py`:**
- `BasevLLMParameter.__init__` `:41-66` captures `weight_loader`, `self.tp_rank`, `self.tp_size`.
- `_ColumnvLLMParameter(output_dim)` `:129-201`:
  - `load_column_parallel_weight` `:148-154`: `shard_size = self.data.shape[self.output_dim]`, `narrow(output_dim, tp_rank*shard_size, shard_size)`.
  - `load_merged_column_weight` `:156-176` and `load_qkv_weight` `:178-201`: if the param is a `PackedColumnParameter`/`PackedvLLMParameter` **and** `packed_dim == output_dim`, call `adjust_shard_indexes_for_packing` first (`:162-168`, `:186-192`).
- `RowvLLMParameter(input_dim)` `:204-231`: `load_row_parallel_weight` `:220-231` narrows `input_dim` by `tp_rank*shard_size`. Note: **the packing adjustment is never applied to the input dim** — the input dim is taken in the parameter's own units.
- `ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter)` `:233-240` — both dims.
- `PackedvLLMParameter(ModelWeightParameter)` `:353-395`: ctor `(packed_factor: int | Fraction, packed_dim: int, marlin_tile_size: int | None = None, **kwargs)`.
- `_adjust_shard_indexes_for_packing(shard_size, shard_offset, packed_factor, marlin_tile_size)` `:605-616`: `shard_size // packed_factor`, `shard_offset // packed_factor`, then optional marlin tiling (`:601-602`).
- Dispatch: `ColumnParallelLinear.weight_loader_v2` `linear.py:790-796` → `param.load_column_parallel_weight`; `RowParallelLinear.weight_loader_v2` `linear.py:1763-1770` → `param.load_row_parallel_weight`; merged `linear.py:1140-1204`; qkv `linear.py:1371-1415`.
- `LinearBase.update_param_tp_status()` `linear.py:501-505` refreshes `tp_rank`/`tp_size` on every `BasevLLMParameter` after `create_weights`.

### 4.3 Does PXQ4 shard through this? **Yes**, with a two-parameter split.
INFERENCE (design), grounded in the established layout (`mgv-wt` `ggml/src/pxq-cpu.h:1-17`, `ggml/src/ggml-cuda/pxa/pxq6.cuh:8-18`): a PXQ4 tensor = per-64-row panels, each panel = 128 B fp16 anchor header + one 1088 B slab per 32-column block. The header is *not* interleaved with the slabs, so store it as its own parameter and the remaining bytes become a clean 3-D array:

| param | dtype / shape (per rank) | attributes |
|---|---|---|
| `pxq_slab` | `uint8 [N_part//64, K_part//32, 1088]` | `output_dim=0`, `input_dim=1`, `packed_dim=0`, `packed_factor=64` |
| `pxq_anchor` | `float16 [N_part]` (or `[N_part//64, 64]`) | `output_dim=0` only (no `input_dim`, no packing) |

- **Column split (N):** loader narrows `pxq_slab` on dim 0 after dividing row offsets by `packed_factor=64` → whole panels, a pure memcpy. `pxq_anchor` narrows on dim 0 in row units. Both exact.
- **Row split (K):** loader narrows `pxq_slab` on `input_dim=1` using the param's own shape — dim 1 is already in 32-column slab units, and packing is *not* applied to the input dim (`parameter.py:220-231`; `linear.py:1749-1752`), so no unit mismatch. `pxq_anchor` has no `input_dim`, so `RowParallelLinear.weight_loader` copies it whole = the required header duplication (`linear.py:1747`: the narrow is skipped when `input_dim is None`). Exact, and it is the byte-gather repack we already established, expressed for free by `narrow()`.
- **Merged / QKV:** `shard_offset`/`shard_size` come from `output_sizes` in row units, then `//64` via `packed_factor`. Requires every per-rank sub-shard to be a multiple of 64 rows. Verified against this model's shapes at TP=2 and TP=4 for `attn_q/qkv/gate/output`, `ffn_gate/up/down` — the only violation is the GDN b/a pair (§6.3).

Note the divide is `round(x // packed_factor)` (`linear.py:1055-1056`, `parameter.py:608-609`) — it **silently truncates** if a shard is not 64-aligned. Add an explicit `assert size % 64 == 0` in `create_weights` so a bad TP size fails loudly instead of loading garbage.

### 4.4 Which loader path to choose
Recommend **v1** (do *not* add `PXQ4LinearMethod` to `WEIGHT_LOADER_V2_SUPPORTED`, `linear.py:193-210`), using plain `torch.nn.Parameter` + `set_weight_attrs`. Reasons (FACT):
1. v1 honours `is_sharded_weight` (`linear.py:755-761,1064-1070,1729-1735`); v2 does not. Keeps a pre-sharded fallback available if a future PXQ tier breaks the clean-shard property.
2. v1's `MergedColumnParallelLinear.weight_loader` applies `packed_factor` in *both* branches (`:1013-1018` fused-on-disk, `:1053-1058` per-shard-id); the v2 fused-on-disk branch routes through `_load_fused_module_from_checkpoint` (`linear.py:1230-1268`) which only applies packing for `PackedColumnParameter`/`PackedvLLMParameter` instances.
3. `PackedvLLMParameter` would also work; if we go v2 we must subclass it, not just set attributes, because the v2 isinstance checks are literal (`parameter.py:162`, `:186`).

### 4.5 CUDA graph / torch.compile
FACT: the fork ships `cudagraph_mode=FULL_AND_PIECEWISE` and `splitting_ops` naming `vllm::qwen_gdn_*`. Custom ops are C++-registered into `torch.ops._C` with a Python-side `@register_fake` shim, e.g. `_sm70_ops.py:287-315` (`awq_gemm_sm70_out` + `_awq_gemm_sm70_out_fake` returning `None`). Out-of-tree we cannot add to `_C`; we must register our kernels in our own namespace (`torch.ops.pxq::pxq4_gemm_out`) via `TORCH_LIBRARY` in our own `.so` plus a `register_fake` for the meta/abstract impl. ASSUMPTION (not verified): a plugin-owned namespace participates in piecewise CUDA-graph capture the same way `_C` ops do — worth confirming before we commit to the op signature.

---

## 5. How the checkpoint selects the quant method

FACT, `vllm/config/model.py:1002-1080` (`ModelConfig._verify_quantization`):
- `:1003` `supported_quantization = me_quant.QUANTIZATION_METHODS` (the runtime list our decorator appended to).
- `:1008` `quant_cfg = self.model_arch_config.quantization_config` — i.e. `config.json["quantization_config"]`; `:1010` `quant_method = quant_cfg["quant_method"]`.
- `:1014-1032` a fixed `overrides` list of built-in probe order; `:1038-1044`: `quantization_methods = [q for q in supported_quantization if q not in overrides] + overrides` — with the comment *"Any custom overrides will be in quantization_methods so we place them at the start of the list so custom overrides have preference over the built-in ones."* **Our `"pxq4"` is probed first.**
- `:1045-1066` for each name: `method = me_quant.get_quantization_config(name)`, `method.override_quantization_method(quant_cfg, self.quantization, hf_config=self.hf_config)`; first non-`None` wins and sets `self.quantization`. `:1055-1064` raises if a *built-in* (`name in get_args(QuantizationMethods)`) returns an override without being in the `overrides` list — this check explicitly does **not** apply to custom names.
- `:1069-1078` otherwise `self.quantization` must equal `quant_cfg["quant_method"]` or it raises.

So: `config.json` carrying `"quantization_config": {"quant_method": "pxq4", ...}` selects us with no CLI flag; `--quantization pxq4` also works and must agree.

FACT, `vllm/model_executor/model_loader/weight_utils.py:263-...` (`get_quant_config`):
- `:268` `quant_cls = get_quantization_config(model_config.quantization)`;
- `:271-273` special-cases `"gguf"` (no config file → `quant_cls()`);
- `:275-282` reads `hf_config.quantization_config`, then `text_config.quantization_config`, then `compression_config`;
- `:302-321` `return quant_cls.from_config(hf_quant_config)` — the normal path, so our `from_config(dict)` receives the raw `quantization_config` dict verbatim;
- `:326-346` `hf_overrides["quantization_config_file"]` → `quant_cls.from_config_file(...)` and `quantization_config_dict_json` → `from_config_dict_json(...)`, both optional (`hasattr` guarded) — useful hooks if we ever want the PXQ backbone map in a sidecar file rather than `config.json`.

`get_config_filenames()` is the *other* discovery path (used further down `get_quant_config` for checkpoints with no `quantization_config` in `config.json`, AWQ-style `quant_config.json`, `awq.py:187-193`). We can return `[]` and rely on `config.json`.

---

## 6. Exactly what we must implement

### 6.1 Class list
```python
# pxq4_vllm/config.py
@register_quantization_config("pxq4")
class PXQ4Config(QuantizationConfig):
    def __init__(self, backbone_rev: int, backbone_map: dict[str, str],
                 ignored_layers: list[str] | None = None): super().__init__(); ...
    def get_name(self) -> str: return "pxq4"
    def get_supported_act_dtypes(self) -> list[torch.dtype]: return [torch.half]   # V100: no bf16
    @classmethod
    def get_min_capability(cls) -> int: return 70
    @staticmethod
    def get_config_filenames() -> list[str]: return []
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PXQ4Config": ...
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None): 
        return "pxq4" if hf_quant_cfg.get("quant_method") == "pxq4" else None
    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None: ...

class PXQ4LinearMethod(LinearMethodBase):        # panel-packed 4-bit path
    def __init__(self, quant_config: PXQ4Config): ...
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs): ...
    def process_weights_after_loading(self, layer) -> None: ...
    def apply(self, layer, x, bias=None) -> torch.Tensor: ...

class PXQ8LinearMethod(LinearMethodBase):        # q8_0 tier: attn_k / attn_v / geometry failures
    ...same four members...
```
Plus, from the mixed-checkpoint reality (`docs/lab/LEVERS.md` PXA_PXQ_BACKBONE rev2), `PXQ4Config.get_quant_method` must dispatch **per tensor class**:

| checkpoint tier | vLLM method to return |
|---|---|
| PXQ4 (attn_q/qkv/output/attn_gate per-channel, dense ffn_*) | `PXQ4LinearMethod` |
| q8_0 (attn_k, attn_v, geometry failures) | `PXQ8LinearMethod` (or dequantise offline to fp16 and return `UnquantizedLinearMethod()`) |
| f16 (per-head attn_gate `ne[1] <= 256`) | `UnquantizedLinearMethod()` (`linear.py:327`) |
| q6_k `token_embd`, q8_0 `output`/lm_head | `None` → `UnquantizedEmbeddingMethod` (`vocab_parallel_embedding.py:479-482`); dequantise to fp16 in the converter |
| `ssm_*`, `nextn.*` (MTP), routers, norms | not `LinearBase` / not quantised — untouched |

`get_quant_method` returning `None` for a `LinearBase` **raises** `ValueError("All linear layers should support quant method.")` (`linear.py:492-495`) — so for skipped *linear* layers we must return `UnquantizedLinearMethod()`, never `None`. `None` is only correct for embedding layers. `is_layer_skipped(prefix, ignored_layers, fused_mapping, *, skip_with_substr=False)` (`quantization/utils/quant_utils.py:489-...`) is the stock helper; AWQ calls it with `self.packed_modules_mapping` at `awq.py:205-211`.

### 6.2 Parameter layout (the sharding declaration) — recap
`pxq_slab uint8 [N//64, K//32, 1088]` with `{output_dim:0, input_dim:1, packed_dim:0, packed_factor:64}`, and `pxq_anchor float16 [N]` with `{output_dim:0}`. `weight_loader` taken from `extra_weight_attrs["weight_loader"]` and set on both (`awq.py:390,431-433` for the pattern). Nothing else is needed for TP=2 or TP=4, column or row.

### 6.3 Mandatory: force the GDN split projection
FACT `qwen3_5.py:127-157`: `_uses_split_gdn_input_projections(quant_config)` collects `getattr(quant_config, attr)` for `attr in ("modules_to_not_convert", "ignored_layers", "ignore")` plus `quant_config.config["ignore"]` (`:144-148`), and returns True iff some entry `== "linear_attn"`, `endswith(".linear_attn")`, or contains `"linear_attn.in_proj_a"` / `"linear_attn.in_proj_b"` (`:151-157`).

`PXQ4Config` must therefore expose e.g.
```python
self.ignored_layers = ["linear_attn.in_proj_b", "linear_attn.in_proj_a"]
```
Effect (`qwen3_5.py:207-244`): `in_proj_qkvz` becomes `output_sizes=[key_dim,key_dim,value_dim,value_dim]` (2048/2048/6144/6144 → 512/512/1536/1536 at TP=4, all multiples of 64 ✓) and `in_proj_ba` becomes a separate `MergedColumnParallelLinear(output_sizes=[48,48])` for which `get_quant_method` returns `UnquantizedLinearMethod()`. Without this the fused layer carries two 48-row shards (12 rows/rank at TP=4) and PXQ4 panel sharding is impossible for it.
`packed_modules_mapping` for checkpoint naming is at `qwen3_5.py:665-675` / `823-826`: `qkv_proj←[q_proj,k_proj,v_proj]`, `gate_up_proj←[gate_proj,up_proj]`, `in_proj_qkvz←[in_proj_qkv,in_proj_z]`, `in_proj_ba←[in_proj_b,in_proj_a]` — the offline converter must emit the *unfused* names.

### 6.4 Things that will silently break us
- Setting `layer._sm70_f16_prepared` → `_maybe_sm70_dense_forward` (`linear.py:56-96`) bypasses `apply()` (checked at `linear.py:425-427`, `805`, and in `ReplicatedLinear.forward`/`RowParallelLinear.forward`). Also note `_mark_default_sm70_dense_modules` (`qwen3_5.py:168-181`) sets `_sm70_f16_force_enable = True` on every module whose leaf name is `qkv_proj` or `out_proj` — confirm (not verified) that this force flag cannot promote a PXQ4 layer into the fp16 dense path.
- `VLLM_SM70_QUANT_BACKEND` (`envs.py:793-813`) is *not* our seam; it is a routing knob inside existing configs (`sm70_turbomind.quant_backend()` `:31-33`, consumed by `AWQConfig.get_min_capability` `awq.py:177-185`). We do not read it.
- Do **not** touch `gguf.py`: `get_quant_config` short-circuits `"gguf"` at `weight_utils.py:271-273` and the GGUF params use `UninitializedParameter` + `is_gguf_weight` special cases threaded through every loader (`linear.py:762-772`, `934-956`, `1425-1445`, `1736-1746`) whose sharding assumes per-row-contiguous blocks. Offline converter → safetensors, as already decided.

---

## 7. Open items (not verified)
- Whether an out-of-tree `TORCH_LIBRARY` namespace is capturable under `cudagraph_mode=FULL_AND_PIECEWISE` alongside `vllm::qwen_gdn_*` splitting ops.
- Whether `_sm70_f16_force_enable` (`qwen3_5.py:168-181`) can override a quantized `qkv_proj`/`out_proj`.
- Exact `envs.py` line numbers for `VLLM_SM70_QUANT_BACKEND` were taken from the brief, not re-read here.
- No throughput number is asserted anywhere in this document; no GPU was run.
