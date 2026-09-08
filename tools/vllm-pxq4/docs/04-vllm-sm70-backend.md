# 1Cat-vLLM sm_70 4-bit backend — reverse engineering

Target: `vllm-qwen38-27b-cyber-1` on the DGX build host, image `kewaii/vllm:latest`.
Source tree present and complete at `/opt/1Cat-vLLM` (git `2ceb150`, "feat: add Dockerfile"),
installed as a **non-editable wheel copy** at `/opt/vllm-venv/lib/python3.12/site-packages/vllm`
(`vllm.__file__` from `/` resolves to site-packages; version `0.1.dev1+g2ceb15066`).
`nvcc` 12.8.93 is present in the container at `/usr/local/cuda/bin/nvcc`.
Prebuilt wheels also at `/opt/1Cat-vLLM/dist-cu128-sm70/`.

**VERDICT: a clean seam exists.** Two of them, at different levels, and they compose.
Nothing found in this fork blocks adding a first-class `pxq4` quantization method on sm_70.
The one substantive constraint is described in §7 (scale granularity / codebook), and it is
a kernel-authoring constraint, not an architectural blocker.

---

## 1. VLLM_SM70_QUANT_BACKEND — where read, what it accepts

FACT. Defined and validated in `vllm/envs.py`:

- `vllm/envs.py:115` — type decl: `VLLM_SM70_QUANT_BACKEND: Literal["auto","marlin","turbomind"] = "auto"`
- `vllm/envs.py:789-790` — `SM70_QUANT_BACKENDS = ("auto","marlin","turbomind")`
- `vllm/envs.py:793-801` — `get_sm70_quant_backend()`; raises `ValueError` on anything else
- `vllm/envs.py:804-810` — `use_sm70_turbomind(default_enabled: bool)`:
  `"marlin" -> False`, `"turbomind" -> True`, `"auto" -> default_enabled`
- `vllm/envs.py:812-813` — `force_sm70_marlin()`
- `vllm/envs.py:1490` — registered in the env lambda table

Re-exported thinly at `vllm/model_executor/layers/quantization/sm70_turbomind.py:32-41`
(`quant_backend()`, `use_turbomind()`, `forces_marlin()`).

**"turbomind" does NOT select a backend object.** It is a tri-state *override of per-format
default booleans*. Each existing quant format has its own default:

| env | default | envs.py |
|---|---|---|
| `VLLM_SM70_AWQ_TURBOMIND` | 1 | `envs.py:116,1494-1495` |
| `VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND` | **0** | `envs.py:118,1503-1504` |
| `VLLM_SM70_FP8_TURBOMIND` | 1 | `envs.py:195,1798-1799` |
| `VLLM_SM70_NVFP4_TURBOMIND` | 1 | `envs.py:197,1811-1812` |
| `VLLM_SM70_MXFP4_TURBOMIND` | 1 | `envs.py:198,1814-1815` |

INFERENCE (well-supported): the running container sets `VLLM_SM70_QUANT_BACKEND=turbomind`
(confirmed in `docker inspect` Config.Env) precisely because this model is
**compressed-tensors**, whose per-format default is `0`. The global override is what flips
`compressed_tensors_wNa16` onto the TurboMind path. Setting `marlin` instead would route the
same checkpoint through `csrc/quantization/marlin/sm70_*`, which this fork also ships.

`turbomind` therefore means: *"prefer the vendored lmdeploy TurboMind sm70 s884 GEMM over the
sm70 Marlin port, for every quant format that has both."*

## 2. The seam — is it a dispatch layer we can extend?

FACT, and this is the important nuance:

**`VLLM_SM70_QUANT_BACKEND` is NOT an extensible backend registry.** It is a `Literal` of three
strings validated at `envs.py:795-800`; adding `"pxq4"` there would require editing their
`envs.py`, and it would still not do anything, because there is no dispatch table keyed on the
value. Every consumer is a hand-written `if` inside an existing quant config:

- `awq.py:177-183, 254, 443` (AWQLinearMethod)
- `awq_marlin.py:265-266, 300-313`
- `auto_gptq.py:454, 517-518`
- `fp8.py:158, 221, 230, 338-343, 434, 451, 464, 540, 576, 694, 811, 953`
- `compressed_tensors/schemes/compressed_tensors_wNa16.py:83, 230-268, 297-298`
- `.../compressed_tensors_w4a16_nvfp4.py:96, 124-125`
- `.../compressed_tensors_w4a4_mxfp4.py:44-50, 103, 134-135`
- `.../compressed_tensors_w4a4_nvfp4.py:46-53, 150, 183-184`
- `mxfp4.py:814-817`, `mxfp4_sm70_moe.py:30`

Note also that TurboMind is **not** registered as an `MPLinearKernel`. The mixed-precision
kernel registry (`vllm/model_executor/kernels/linear/mixed_precision/__init__.py:42-56`:
Marlin, Machete, Exllama, AllSpark, Conch, Triton, CPU, XPU, RDNA3, Dynamic4bit, Cutlass) has
**no TurboMind entry**. The fork deliberately bypassed that registry and bolted TurboMind into
each scheme's `process_weights_after_loading` / `apply`. So: `turbomind` is *hardwired into the
AWQ / GPTQ / compressed-tensors / fp8 paths*, per your question's second alternative.

**But the seam we actually need is a different and better one, and it is first-class:**

`vllm/model_executor/layers/quantization/__init__.py:57-101` — `register_quantization_config(name)`.
- appends to `QUANTIZATION_METHODS` (`__init__.py:88`), which is what
  `vllm/config/model.py:1002-1085` (`_verify_quantization`) validates `--quantization` against
- appends to `current_platform.supported_quantization` if non-empty (`__init__.py:89-92`).
  FACT: `vllm/platforms/cuda.py` does **not** override `supported_quantization`, so it is the
  base `[]` (`vllm/platforms/interface.py:141`) and the gate at `interface.py:710` is a no-op
  on CUDA. Nothing to fight.
- enforces `issubclass(cls, QuantizationConfig)` (`__init__.py:94-97`)
- documented as a supported public API: `docs/features/quantization/README.md:74-176`
  ("Out-of-Tree Quantization Plugins"), with a worked `MyQuantConfig` / `MyQuantLinearMethod`
  example and a required-methods table.
- regression test exists: `tests/quantization/test_register_quantization_config.py:63,104-113`

Import timing is solved: `load_general_plugins()` (entrypoint group `vllm.general_plugins`,
`vllm/plugins/__init__.py:14,69`) is called from `vllm/engine/arg_utils.py:747-749` and
`:2772`, from `vllm/v1/engine/core.py:106-108`, and from `vllm/v1/worker/worker_base.py:245-247`
— i.e. **before** `ModelConfig._verify_quantization` runs in the engine process and again in each
TP worker. A pip-installed `pxq4-vllm` package exposing a `vllm.general_plugins` entrypoint that
runs the decorator is sufficient; **no edit to their tree is required for registration.**

Contract we must satisfy (`base_config.py`): `get_name` (:79-82), `get_supported_act_dtypes`
(:84-87), `get_min_capability` (:89-96), `get_config_filenames` (:98-101),
`from_config` (:103-107), `get_quant_method(layer, prefix)`; and on the method object
`create_weights` / `apply` / `process_weights_after_loading`
(`QuantizeMethodBase`, `base_config.py:19-56`).

### The template to copy
`vllm/model_executor/layers/quantization/sm70_turbomind.py` (339 lines) is exactly the shape
of what we need, minus the config wrapper:

- `SM70TurboMindLinearState` dataclass (`:21-29`) — weight, scales, group_size, k_ld, q_ld,
  output_size, op_kind — stashed on the layer under `STATE_ATTR = "_sm70_turbomind_linear"` (`:17`)
- unpack helpers `_get_u4_slices` (`:81-92`), `unpack_compressed_weight` (`:106-109`) etc.
- `prepare_compressed_uint4_linear` (`:191-226`) — the offline→runtime repack, called from
  `process_weights_after_loading`
- `apply_prepared_linear` (`:290-339`) — **this is the CUDA-graph-safe apply**:
  `x.reshape(-1, K)` → preallocate `out = torch.empty((M, N), dtype=x.dtype)` → single custom
  op writing into `out` → `out.add_(bias)` → `out.reshape(out_shape)`. No allocation inside the
  op, no host sync, no dynamic shape. That is precisely our kernels' contract.
- the calling scheme zeroes out the original checkpoint params after repack
  (`compressed_tensors_wNa16.py:258-268`: replaces `weight_packed`/`weight_scale` with
  `torch.empty(0)` Parameters) so the fp16-unpacked staging tensor is freed. Copy this — it is
  what keeps the 4.64 GiB/GPU resident figure honest.

Python↔C++ shim: `vllm/_sm70_ops.py` (2123 lines) wraps every op with a `_op(name)` presence
check (`:23-29`) plus a `register_fake` meta implementation (e.g. `:56-75`) so `torch.compile`
and `FULL_AND_PIECEWISE` CUDA graph capture can trace it. We must supply fakes too.

## 3. How it copes with sm_70 lacking Ampere features

FACT: **native 4-bit-fed HMMA, not dequant-to-fp16-then-cuBLAS.** There is no cuBLAS call
anywhere in the backend — `grep -i cublas csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` returns
nothing across all 8492 lines.

The mechanism, bottom-up:

- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/mma.h:13-25` — inline PTX
  `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, guarded `#if TURBOMIND_ARCH_SM70`.
  This is the Volta-only HMMA shape; Ampere `m16n8k16` is never used.
- `.../gemm/arch/mma_sm70.h:12-36` — `SM70_MMA_884`, M=8 N=32 K=8, `FragA/FragB = Array<half,K>`,
  `FragC = Array<float,8>`; `fma()` issues two `mma_m8n8k4_row_col`.
  Weights arrive as **half fragments**: the 4-bit→half conversion happens in registers, in the
  operand transform stage (`Transform_HMMA_SIMT_B`, see `arch/config_sm70_s884.h:133-142`),
  immediately before the HMMA. Nothing is materialised to fp16 in HBM.
- Tiling/pipelining is Volta-shaped: `iterator_sm70.h`, `mainloop_sm70.h`, `scheduler_sm70.cuh`,
  `smem_copy_sm70.h` — i.e. no `cp.async`, no async barriers, no TMA. They wrote a Volta
  mainloop rather than trying to degrade an Ampere one.
- Kernel instantiations: `kernel/sm70_884_4.cu`, `sm70_884_8.cu`, `sm70_884_16.cu`. Registered
  in `csrc/sm70_turbomind/ops/tm_registry_sm70.cu:9-17` — a **replaced** `Registry` ctor that
  registers only `sm70_884_4/8/16()`; `Registry::Add` (`:19-37`) drops any kernel whose arch is
  incompatible or whose smem exceeds `sharedMemPerBlockOptin` (Volta = 96 KB), which is how the
  Ampere-sized tiles are silently excluded.
- Config aliases at `arch/config_sm70_s884.h:132-142` (`Config_U4_g`, grouped uint4),
  `:145-154` (`Config_MXF4`), `:156-166` (`Config_NVF4`), `:169+` (`Config_E4M3`) —
  all `Sm70_s884<Operand_A<half>, ..., Operand_B_Pack<T>, Transform_HMMA_SIMT_B,
  Operand_V_Pack<U>, ...>`. **Adding a weight format = adding one more `Operand_B_Pack` type +
  its register-side converter, then instantiating tile shapes.** That is the intended extension
  point of the vendored code.
- The dispatcher `awq_gemm_sm70_out` (`ops/awq_sm70_gemm.cu:3058-3196`) builds
  `MatrixLayout desc_A/desc_B/desc_V/desc_D`, `op.quant_b = {QuantType::kK, group_size}`, then
  `gemm.Run(...)`. Type checks at `:3067-3074`: **x fp16, weight int32, scales int32, out fp16** —
  exactly the shape contract in the brief.
- CUDA-graph awareness is explicit: `is_stream_capturing()` (`:1222`), autotune caches are
  refused during capture (`:1686` `TORCH_CHECK(!is_stream_capturing(stream), "cache miss during
  CUDA graph capture")`), and there is an export/import cache pair
  (`_C::sm70_gemm_export_cache` / `_C::sm70_gemm_import_cache`).

There are ALSO hand-written, non-TurboMind Volta kernels living in the same translation unit —
`nvfp4_raw_gemv_partial_kernel` (`:589`), `nvfp4_raw_gemv_h2_partial_kernel` (`:711`),
`nvfp4_raw_gemv_warp_kernel` (`:797`), plus reduce kernels, and
`sm70_f16_gate_mul_kernel` (`:1745`), `sm70_silu_and_mul_fp16_kernel` (`:48`).
INFERENCE: these are M=1 decode GEMV fast paths where the tensor-core GEMM loses to a plain
SIMT/dp4a-style kernel. **This is a second, much cheaper template for PXQ4**: a bespoke
decode-shaped GEMV kernel can be dropped in beside them without touching TurboMind's template
machinery at all.

## 4. Sources vs. prebuilt .so

FACT: **full sources shipped.** Not binary-only.

- `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` — 8492 lines, 376 KB. The whole Python-facing op
  surface (all `*_sm70_prepare`, all `*_gemm_sm70_out`, MoE variants, lm_head kernels).
- `csrc/sm70_turbomind/ops/tm_registry_sm70.cu` — 39 lines, the arch-gated kernel registry.
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/{core,utils,kernels/{gemm,core,attention}}` —
  vendored lmdeploy, Apache-2.0.
- `csrc/quantization/marlin/sm70_*.cu` — 11 files, the independent sm70 Marlin port
  (`CMakeLists.txt:445-456`).
- `csrc/torch_bindings.cpp:137-310+` — every op `def`/`impl`, guarded by
  `ENABLE_SM70_TURBOMIND=1` / `ENABLE_SM70_MARLIN=1` compile definitions.
- Build wiring: `CMakeLists.txt:110-115` (`CUDA_SUPPORTED_ARCHS` includes `7.0` — upstream
  vLLM dropped it), `:365-410` (SM70 TurboMind srcs + `set_gencode_flags_for_srcs` for arch 7.0),
  `:443-470` (SM70 Marlin, only when *no other* Marlin arch is in the build — note
  `elseif(MARLIN_SM70_ARCHS)` → "Skipping SM70 Marlin kernels in mixed Marlin arch build").
- The compiled artifact is `/opt/vllm-venv/.../vllm/_C.abi3.so` (164 MB). Verified via
  `torch._C._dispatch_get_all_op_names()`: **54 registered `sm70` ops**, including
  `_C::uint4_sm70_prepare`, `_C::awq_gemm_sm70_out`, `_C::awq_gemm_sm70_out_tile_reduce`,
  `_C::mxfp4_gemm_sm70_out`, `_C::nvfp4_gemm_sm70_out`, `_C::nvfp4_gemv_sm70_{raw,warp,h2}_out`,
  `_C::fp8_gemm_sm70_out{,_auto,_meta}`, `_C::sm70_f16_gemm{,_out}`,
  `_C::sm70_f16_lm_head_top1{,_tc}_out`, `_C::sm70_gemm_{export,import}_cache`,
  `_C::sm70_marlin_available`, plus 13 `awq_moe_*_sm70_*` and 8 `fp8_moe_*_sm70_*`, and
  `_C_custom_ar::sm70_tp{2,4}_all_reduce_gemma_rms_norm`.

**Consequence for us:** we have a choice, and the low-risk one is available.
(a) *Separate extension* — build `pxq4_sm70_C.so` as our own torch extension, register ops in
our own `pxq4::` namespace, ship it as a pip package alongside their wheel. No rebuild of their
164 MB `_C`, no fork of their tree, works against the wheel they already ship. Recommended.
(b) *In-tree* — add `csrc/pxq4_sm70/` + CMake block mirroring `CMakeLists.txt:365-410`. Better
if we intend to upstream to Kewaii, but requires a full rebuild.

Note (a) still costs a copy of the TurboMind headers if we want the s884 mainloop; those are
Apache-2.0 and header-mostly under `kernels/gemm/`, so vendoring is legal and mechanical.

## 5. get_min_capability gating — sm_70 is already a first-class citizen here

FACT: the gate is a single check, in the engine, once:
`vllm/config/vllm.py:601-633` (`_get_quantization_config`) —
```
capability = current_platform.get_device_capability().to_int()
if capability < quant_config.get_min_capability():
    raise ValueError(...)
```
followed by a `get_supported_act_dtypes()` check that raises if `model_config.dtype` is not
listed. Declared abstract at `base_config.py:89-96` ("E.g., 70 for Volta, 75 for Turing,
80 for Ampere").

This fork has already done the 75→70 work, and did it **conditionally on the turbomind knob**:

- `awq.py:177-184`: returns `70` if `use_turbomind(VLLM_SM70_AWQ_TURBOMIND) or forces_marlin()`,
  else falls back to the upstream Turing-and-up value.
- `compressed_tensors_wNa16.py:80-88`: same pattern —
  `70` if `use_turbomind(VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND) or forces_marlin()`, else `75`.
- `awq_marlin.py:265-266`: unconditional `return 70`.
- `compressed_tensors.py:108-110`: `CompressedTensorsConfig.get_min_capability()` → unconditional
  `return 70`. (Per-scheme values are then checked separately via `_check_scheme_supported`,
  `compressed_tensors.py:757`.)

**For PXQ4:** declare `get_min_capability() -> 70` unconditionally — we have no Ampere fallback
to protect and no reason to make it conditional. `get_supported_act_dtypes()` must return
`[torch.float16]` only (V100 has no bf16; the fork runs the production model at
`dtype=torch.float16`, and every sm70 op `TORCH_CHECK`s fp16 — `awq_sm70_gemm.cu:3067-3074`).
Note that returning only fp16 makes `vllm/config/vllm.py:622-627` raise a clear error if someone
passes `--dtype bfloat16`, which is the behaviour we want.

There is no second, hidden capability gate to satisfy — the mixed-precision `MPLinearKernel`
`get_min_capability`s (`kernels/linear/mixed_precision/*.py`) only apply if we route through that
registry, which (per §2) TurboMind itself does not, and neither should we.

## 6. GDN / FlashQLA-SM70 — inherited free? **CONFIRMED, with two precise caveats**

FACT — the kernels exist and are quantization-independent:

- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, 7013 lines. Ten custom ops
  registered via `direct_register_custom_op` at `:6833-6935`:
  `qwen_gdn_full_forward`, `qwen_gdn_output_projection`, `qwen_gdn_input_projection_core`,
  `qwen_gdn_input_projection`, `qwen_gdn_attention_core`, `..._standard`, `..._standard_spec`,
  `..._spec_commit`, `..._context`, `..._003_spec`. All ten are listed as
  `splitting_ops` in `vllm/config/compilation.py:764-773`, which is why they appear in the
  container's `FULL_AND_PIECEWISE` log line.
- The recurrent core kernels are a **separate package**, `flash_qla`
  (`/opt/1Cat-vLLM/flash_qla/ops/gated_delta_rule/...`), imported at
  `qwen_gdn_linear_attn.py:1529, 1668-1683, 1814, 2941, 3525` — TileLang
  (`chunk_gated_delta_rule_fwd_sm70_tilelang`) and a compiled `sm70/fused_fwd` extension.
  The log string "FlashQLA-SM70" is at `:1581`, the fp16-only guard at `:1934`
  ("FlashQLA-SM70 GDN prefill only runs on fp16 q/k/v tensors").
- Attention (non-GDN layers) is a third separate package: `flash-attention-v100/`, shipping
  `flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so` and `paged_kv_utils...so`.
- **The projections are called generically.** `qwen_gdn_input_projection`
  (`qwen_gdn_linear_attn.py:6768-6822`) does `mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)`
  and `ba, _ = self.in_proj_ba(hidden_states)` — plain `LinearBase.__call__`, which dispatches to
  whatever `quant_method` the layer was built with. There is **no** `sm70_tm.has_prepared_linear`
  check anywhere in the GDN file; that helper is referenced only from the five quant schemes
  (`auto_gptq.py:517`, `compressed_tensors_w4a4_mxfp4.py:134`, `..._w4a16_nvfp4.py:124`,
  `..._wNa16.py:297`, `..._w4a4_nvfp4.py:183`). The recurrent core receives fp16
  `mixed_qkv`/`b`/`a` tensors and never learns what produced them.

So the answer to your key assumption is **verified: yes, we inherit GDN, FlashQLA-SM70,
FLASH_ATTN_V100, paged KV, CUDA graphs, TP and MTP regardless of weight format.**

CAVEATS — two places where `quant_config` changes GDN *structure*, which our config must
answer correctly:

1. **`_uses_split_gdn_input_projections(quant_config)`** — `vllm/model_executor/models/qwen3_5.py:127-157`.
   It introspects `quant_config` for `modules_to_not_convert` / `ignored_layers` / `ignore`
   (and `quant_config.config["ignore"]`) and returns True if `linear_attn` / `in_proj_a` /
   `in_proj_b` are listed. That flag decides at `qwen3_5.py:204-245` whether `in_proj_qkvz` is
   `MergedColumnParallelLinear([key_dim, key_dim, value_dim, value_dim])` with a **separate**
   `in_proj_ba`, or one fused `[..., num_v_heads, num_v_heads]` projection with `in_proj_ba = None`.
   **Our `PXQ4Config` must expose an `ignored_layers`-shaped attribute** whose contents mirror
   what the offline converter actually left unquantized, or the module layout will not match the
   weights we produce. This is a direct consequence of the mixed-type backbone table
   (`docs/lab/LEVERS.md` PXA_PXQ_BACKBONE rev2): attn_k/attn_v→q8_0, per-head attn_gate→f16,
   token_embd→q6_k, output→q8_0, ssm_*/nextn.*/router/norms untouched.

2. **`maybe_disable_tp(quant_config)`** — `qwen_gdn_linear_attn.py:2363-2379`. Returns True
   (replicate `ba_proj` instead of TP-sharding it) only for
   `isinstance(quant_config, (AWQMarlinConfig, AutoGPTQConfig, INCConfig))`. The reason given
   is Marlin's `MIN_THREAD_N=64` vs `num_v_heads=64 / TP4 = 16`
   (ref: vllm-project/vllm#35924). `PXQ4Config` is none of those three, so `ba_proj` stays
   TP-sharded — which is correct for us **iff** our kernel tolerates N=16 per rank, or **iff**
   `ba_proj` is one of the tensors our backbone leaves in f16/q8_0.
   FACT from the brief: per-HEAD `attn_gate` with `ne[1] <= 256` is demoted to **f16** by the
   backbone table, and `ba_proj` output is `num_v_heads*2 = 128` rows. So this projection will
   not be PXQ4 at all and the constraint dissolves — but that must be checked against the actual
   artifact, not assumed. Also note `qwen3_5.py:167-176` force-marks `qkv_proj`/`out_proj` with
   `_sm70_f16_force_enable`, an unrelated dense-fp16 fast path we should leave alone.

Not verified: whether the GDN conv1d / `ssm_state` paths have any dtype coupling beyond fp16.

## 7. The one real constraint: scale granularity and the codebook

This is the only place where "copy the pattern" stops being mechanical.

FACT — registered sm70 s884 group sizes, from `kernel/sm70_884_4.cu`:
- `Config_U4_d` (per-channel uint4): `:41-56`
- `Config_U4_g` (grouped uint4): GroupSizeV = **128** (`:73-85`), **64** (`:87-92`), **32** (`:94-100`)
- `Config_MXF4`: GroupSizeV = **32** only (`:105-112`)
- `Config_NVF4`: GroupSizeV = **16** (`:115-123`)
- `grep "GroupSizeV=16"` on the U4 configs: **no match.**

PXQ4 (per the brief and `ggml/src/pxq-cpu.h:1-17`, `ggml/src/ggml-cuda/pxa/pxq6.cuh:8-18`) is
fp16 row anchor × frozen 4-bit SUB16 sub-scale **per 16 elements**, i.e. effective group 16.
So the grouped-uint4 TurboMind path cannot be reused as-is: its smallest registered group is 32.

**But NVFP4 is already instantiated at group 16 on sm_70** (`sm70_884_4.cu:117-123`,
`Config_NVF4<kColMajor, 0>` with `..., 1, 16, ...`), and NVFP4 is structurally the closest
existing analogue: 4-bit nonuniform codebook element (`Operand_B_Pack<fp4_e2m1_t>`,
`config_sm70_s884.h:160`) + a per-16 sub-scale (`Operand_V_Pack<uint16_t>`, `:162`) + a
per-tensor global scale that `prepare_nvfp4_linear` **folds into the fp16 scale table** before
the kernel ever sees it (`sm70_turbomind.py:267-278`:
`weight_scale.t().to(f32) * weight_global_scale` → `.to(torch.float16)`).

INFERENCE (needs verification against `ggml/src/ggml-cuda/pxa/pxq6.cuh` — I did not read the PXQ4
codebook in this task): the same fold works for us. PXQ4's per-row fp16 anchor is a *rank-1*
factor over the output dimension, and the sub-scale is per-(16-block, row); their product
`eff[block][row] = anchor[row] * sub16[block][row]` is exactly an NVFP4-shaped fp16 group-16
scale table. The offline converter emits that table directly, and the runtime `prepare` becomes
a pure layout swizzle with **no arithmetic and no re-quantization** — the same property the
brief already established for TP sharding. What remains genuinely new is the element decode:
if PX16 is a nonuniform 16-entry book, it replaces `fp4_e2m1_t`'s LUT with ours inside the
`Transform_HMMA_SIMT_B` register-side converter. That is one converter type + tile-shape
instantiations, mirroring `Config_NVF4`.

If that turns out harder than it looks, the fallback is §3's hand-written route:
a bespoke `pxq4_gemv_sm70_*_out` modelled on `nvfp4_raw_gemv_warp_kernel`
(`awq_sm70_gemm.cu:797`) for M=1 decode — which is the regime that actually decides the
tok/s number — with a slower generic path for prefill.

## 8. Concrete plan implied by all of the above

1. Offline converter (host-side, no vLLM): PXQ4 GGUF → a directory of safetensors +
   `config.json` carrying `quantization_config = {"quant_method": "pxq4", ...}` including an
   `ignore`/`ignored_layers` list naming every tensor the backbone table demoted (q8_0 / q6_k /
   f16 / untouched). Per the brief, do **not** touch `vllm/.../gguf.py` — type 252 is unknown to
   the vendored `gguf` package and its row-sharder violates panel interleave.
2. `pxq4_vllm` pip package, entrypoint group `vllm.general_plugins`, whose plugin function runs
   `@register_quantization_config("pxq4")` on `PXQ4Config(QuantizationConfig)`.
   `get_min_capability() -> 70`; `get_supported_act_dtypes() -> [torch.float16]`;
   `get_quant_method` returns `PXQ4LinearMethod` for `LinearBase` when the layer is in the PXQ4
   set, `UnquantizedLinearMethod` / a q8_0 method otherwise (the checkpoint is mixed-type —
   §6 caveat 1 depends on this being declared honestly).
3. `PXQ4LinearMethod` modelled line-for-line on `sm70_turbomind.py`: state dataclass on the
   layer, repack in `process_weights_after_loading`, preallocated-output custom op in `apply`,
   free the staging params afterwards (`compressed_tensors_wNa16.py:258-268`).
4. `pxq4_sm70_C` torch extension in our own op namespace + `register_fake` metas mirroring
   `vllm/_sm70_ops.py:56-75`, so `FULL_AND_PIECEWISE` capture works.
5. Kernel: first target M=1 decode GEMV (`nvfp4_raw_gemv_warp_kernel` shape); then, if the
   numbers justify it, the `Config_NVF4`-shaped s884 path at group 16.

Nothing in steps 1-4 requires modifying `/opt/1Cat-vLLM`. Step 5 optionally vendors
Apache-2.0 TurboMind headers.

## 9. What we do NOT inherit

The AWQ path carries a large amount of *format-specific* tuning that will not transfer and
should not be assumed in any projection:
`awq_gemm_sm70_out_tile_reduce` + the TP2 tile-all-reduce substrate
(`awq_sm70_gemm.cu:3208-3290`, `vllm/distributed/device_communicators/custom_all_reduce.py:631`);
the fused gate_up+SiLU epilogue behind `VLLM_SM70_AWQ_MLP_ENGINE` (`envs.py:139,1581-1583`;
`awq.py:467,534-570`; `qwen2.py:115`) — default **off**;
`VLLM_SM70_AWQ_PREFILL_EXACT_DENSE` (default **on**, `awq.py:472-521`), which dequantizes
selected TP4 prefill projections into a bounded fp16 workspace and runs
`_C::sm70_f16_gemm_out` instead;
the autotune/dispatch-policy machinery (`awq_sm70_gemm.cu:1030-1360`) and its exported cache;
and `vllm/model_executor/warmup/awq_sm70_warmup.py`. Each of these is a separate porting
decision for PXQ4, not a freebie.

Any throughput figure for PXQ4-in-vLLM remains a **PROJECTION**. No GPU run was performed in
this task; nothing was started, stopped, or modified on either box.
