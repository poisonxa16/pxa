# 05 — vLLM PXQ4 backend: the template, the loading path, and the MoE delta

All paths below are inside the running container `vllm-qwen38-27b-cyber-1` on the DGX build host, source
checkout `/opt/1Cat-vLLM` (fork `1cat_vllm-0.1.dev1+g2ceb15066.cu128`). Everything was read
read-only via `docker exec`. No GPU was used; no container was modified.

---

## 0. Headline

**MoE effort delta = ZERO.** FACT, from the artifact itself: I parsed the GGUF header of
`/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf` directly (raw struct parser, not the `gguf`
package). 866 tensors, `general.architecture = qwen35`, and there is **no `*_exps` tensor and
no expert KV of any kind**. Per-class tensor census below. There is no fused-MoE port, no
`FusedMoEMethodBase`, no `RoutedExperts` path to implement. Delete that line item.

**Two corrections to the brief, both load-bearing:**

1. `sm70_turbomind.py` is **not** a `QuantizationConfig` and does not register anything. It is
   a stateless helper module (repack + apply shim) that *existing* configs call from inside
   their own `process_weights_after_loading` / `apply_weights`. It has no `create_weights`, no
   parameter declarations, no TP-sharding logic. It is the right template for the **kernel-call
   half** of the job and tells us nothing about the **weight-declaration half**. The real
   template for that half is
   `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py:99-220`,
   which is the scheme actually serving this model today.

2. The PXQ4 artifact is **five** types, not the four the brief listed, and it contains **no f16
   tensor at all**. Measured histogram (ggml type id → count):
   `{0 (F32): 360, 8 (Q8_0): 132, 14 (Q6_K): 1, 39 (MXFP4): 48, 252 (PXQ4): 325}`.
   `GGML_TYPE_MXFP4 = 39` confirmed at `/path/to/engine-repo/ggml/include/ggml.h:424`.
   **`ssm_out.weight` (all 48 GDN layers) is MXFP4, not PXQ4** — the backbone map
   (`pxa.pxq.backbone_map` KV, read from the file) lists
   `attn_q,attn_qkv,attn_output,attn_gate_ch,shexp,ffn_dense=tier+1` and `ssm_out` is not in
   that class list, so it fell through to MXFP4. A vLLM backend that only implements PXQ4 will
   leave 48 GDN output projections unserved. See §B3.

---

## A. The template

### A1. Structure of `sm70_turbomind.py` (339 lines, read in full)

It exports three things:

* **Gating predicates** — `is_exact_sm70_cuda(tensor, enabled)` (`:44-47`) checks
  `torch.cuda.get_device_capability(tensor.device) == (7,0)` on an already-materialised weight;
  `is_exact_sm70_cuda_platform()` (`:50-57`) is the pre-weight variant used at quant-method
  selection time. `quant_backend()` / `use_turbomind()` / `forces_marlin()` (`:32-41`) delegate
  to `vllm.envs` — this is the `VLLM_SM70_QUANT_BACKEND` routing knob, and the brief is right
  that it is not the seam.
* **CPU-side unpackers** — `_get_u4_slices` (`:81-92`) and the four `unpack_*` functions
  (`:95-126`) turn the on-disk int32/uint8 nibble packing into a dense `uint8` `[K,N]` tensor,
  plus `scales` as fp16 and `zeros` as fp16. Note `unpack_compressed_weight` (`:106-109`)
  ends with `.t().contiguous()` — the repack op wants **K-major `[K,N]`**, the opposite of
  vLLM's stored `[N,K]`.
* **The state object + repack + apply.** `SM70TurboMindLinearState` (`:21-29`) is a plain
  dataclass holding `weight, scales, group_size, k_ld, q_ld, output_size, op_kind`. It is
  stashed on the layer under the attribute name `"_sm70_turbomind_linear"` (`STATE_ATTR`, `:17`)
  by `_store_state` (`:133-151`), *not* registered as a Parameter.

`prepare_compressed_uint4_linear` (`:191-226`) is the exact shape of what we copy:
unpack → `sm70_ops.uint4_sm70_prepare(qweight, scales, zeros, group_size, interleave_gated_silu)`
→ `_store_state`. The repack op returns a 3-list `[tm_weight, tm_scales, meta]`, and `meta` is a
2-element int64 tensor whose entries become `k_ld` and `q_ld` (`:146-147`) — i.e. the kernel's
leading dimensions are computed **on device by the repack op** and read back once at load time.

`apply_prepared_linear` (`:290-339`):

```python
state = getattr(layer, STATE_ATTR)
reshaped_x = x.reshape(-1, x.shape[-1])
out_shape = x.shape[:-1] + (state.output_size,)
out = torch.empty((reshaped_x.shape[0], state.output_size), dtype=x.dtype, device=x.device)
sm70_ops.awq_gemm_sm70_out(out, reshaped_x, state.weight, state.scales,
                           state.group_size, state.k_ld, state.q_ld)
if bias is not None: out.add_(bias)
return out.reshape(out_shape)
```

The dtype contract, enforced with `TORCH_CHECK` on the C++ side at
`csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:3058-3075`: `in_feats` fp16, `out` fp16, `tm_weight`
int32, `tm_scales` int32, all CUDA. `out` is an explicit first argument — the op is an
out-variant with no return value.

**Consumption pattern** (this is how our config plugs in, `compressed_tensors_wNa16.py:229-300`):
`process_weights_after_loading` calls `sm70_tm.should_prepare_turbomind(...)`, and on success
calls `prepare_compressed_uint4_linear(...)` and then **replaces every original Parameter with a
zero-element `torch.nn.Parameter`** (`:255-278`) so the pre-repack bytes are freed. `apply_weights`
(`:296-299`) then does `if sm70_tm.has_prepared_linear(layer): return sm70_tm.apply_prepared_linear(...)`.
Callers of the shim: `auto_gptq.py:454,517-518`, `compressed_tensors_wNa16.py:230,250,297-298`,
and the w4a4/w4a16 fp4 schemes.

### A2. How the custom op is registered and built — and can we avoid rebuilding vLLM?

**Registration chain (FACT):**

1. `csrc/torch_bindings.cpp:184-186` — `ops.def("uint4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, int group_size, bool interleave_gated_silu) -> Tensor[]"); ops.impl(..., torch::kCUDA, &uint4_sm70_prepare);`
   and `:215-218` — `ops.def("awq_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, int group_size, int k_ld, int q_ld, bool gated_silu) -> ()"); ops.impl(...)`.
   This is a **`TORCH_LIBRARY` block in the `_C` namespace** (schema strings use the `Tensor(a!)`
   mutable-alias annotation on `out`).
2. `csrc/ops.h:108,140,149` declares the C++ prototypes.
3. `CMakeLists.txt:366-412` — the whole thing is gated on
   `cuda_archs_loose_intersection(SM70_TURBOMIND_ARCHS "7.0" "${CUDA_ARCHS}")`, appends
   `csrc/sm70_turbomind/ops/tm_registry_sm70.cu`, `ops/awq_sm70_gemm.cu`, and ~20 vendored
   lmdeploy `.cu/.cc` files to **`VLLM_EXT_SRC`** (i.e. into `_C`), sets
   `-DENABLE_SM70_TURBOMIND=1` and compiles `torch_bindings.cpp` with the same define.
   `tm_registry_sm70.cu:9-18` is the TurboMind kernel registry, registering only
   `sm70_884_4/8/16`.
4. `vllm/_sm70_ops.py` is a thin typed wrapper: `_op(name)` (`:23-29`) does
   `getattr(torch.ops._C, name)` with a clear error if the build lacked arch 7.0, and every op
   gets a `@register_fake("_C::<name>")` meta kernel (e.g. `:302-315` for `awq_gemm_sm70_out`,
   `:113-130` for `uint4_sm70_prepare`). `current_platform.import_kernels()` (`:10`) forces
   `import vllm._C` (`vllm/platforms/interface.py:242-249`).

**Can we add an extension without rebuilding vLLM? YES — INFERENCE, but well-supported:**

* `torch.ops` is a global registry. A **separate** `.so` with its own
  `TORCH_LIBRARY(pxq4, m)` block registers under `torch.ops.pxq4.*` and needs no vLLM symbol,
  no vLLM header, and no `_C` relink. Nothing in the fork's op-lookup path requires our ops to
  live in `_C`; our config would call `torch.ops.pxq4.pxq4_gemm_sm70_out` instead of going
  through `_sm70_ops._op()`.
* The fork ships `vllm/plugins/__init__.py:14` `DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"`,
  loaded in **all** processes including workers, with `load_plugins_by_group` (`:28-60`)
  enumerating `importlib.metadata.entry_points`. So a standalone pip package
  (`pxq4-vllm`) can ship (a) the `.so`, (b) the `PXQ4Config` decorated with
  `@register_quantization_config("pxq4")` (`quantization/__init__.py:57-101`), and (c) an entry
  point in `vllm.general_plugins` whose callable just imports the config module. Zero edits to
  the fork.
* Toolchain is present **inside the container**: `nvcc` 12.8 (`/usr/local/cuda/bin/nvcc`),
  `gcc`/`g++`, `cmake` and `ninja` in `/opt/vllm-venv/bin`, torch `2.10.0+cu128`. So a
  `torch.utils.cpp_extension.CUDAExtension` build with `-gencode arch=compute_70,code=sm_70`
  matches the runtime ABI exactly.
* **Operational blockers, both real:**
  - `df -h /` inside the container reports `overlay 207G used 198G, 0 avail, 100%`. **You cannot
    build in this container.** Build in a fresh container from the same base image with a
    writable volume under `/path/to/models`, then install the resulting wheel.
  - `site-packages/vllm` is a **copied install**, not an editable link to `/opt/1Cat-vLLM`
    (`ls -la` shows a real directory, and `vllm.__file__` →
    `/opt/vllm-venv/lib/python3.12/site-packages/vllm/__init__.py`). Editing `/opt/1Cat-vLLM/vllm/*.py`
    changes nothing at runtime. Another reason to go out-of-tree.
* **Caveat (ASSUMPTION):** two `TORCH_LIBRARY` namespaces coexisting is standard PyTorch, but I
  did not build or load such an `.so` against this image — no GPU runs were permitted. The one
  thing to verify on first build is C++ ABI flag agreement (`_GLIBCXX_USE_CXX11_ABI`) with the
  prebuilt `_C.abi3.so`.

### A3. CUDA-graph-capture safety

What the template actually does, and what actually matters:

* **The output is allocated inside `apply()`** — `torch.empty(...)` at `sm70_turbomind.py:298`.
  This is *not* a violation. During `cudaStreamCaptureMode` capture, PyTorch's caching allocator
  services allocations from the graph's private memory pool; a plain `torch.empty` on the capture
  stream is legal and is what every vLLM quant path does. What is forbidden is a **host-side
  sync** or a **cudaMalloc that escapes the pool**.
* **The real rules the template obeys, which our kernel must too:**
  1. **Every scalar the kernel needs is a Python `int` captured at load time, never read from a
     device tensor in `apply()`.** `k_ld`/`q_ld` are pulled out of the `meta` tensor with
     `int(meta[0])`/`int(meta[1])` inside `_store_state` (`:146-147`) — i.e. the one and only
     device→host sync happens in `process_weights_after_loading`, long before capture. Our PXQ4
     equivalents (panel count, kslab count, K, N, book/sub table pointers) must be resolved the
     same way. A `.item()` in `apply()` will break capture.
  2. **No data-dependent control flow on tensor values.** The `op_kind` branch (`:305-336`) is on
     a Python string fixed at load time.
  3. **Out-variant custom op with the mutable-alias schema annotation** — `Tensor(a!) out`
     (`torch_bindings.cpp:215-216`). Required so the Inductor/piecewise-graph partitioner knows
     the op mutates `out` and does not reorder or DCE it.
  4. **A registered fake/meta kernel** — `@register_fake("_C::awq_gemm_sm70_out")`
     (`_sm70_ops.py:302-315`). Under `cudagraph_mode=FULL_AND_PIECEWISE` the linear layers sit
     inside the `torch.compile` region; without a meta kernel tracing fails before capture is
     even attempted. This is the single most likely thing to be forgotten.
  5. **No stream creation, no event sync, no `cudaDeviceSynchronize`, no host allocations, no
     `TORCH_CHECK` that reads device memory** inside the op. The C++ side only does
     `TORCH_CHECK` on dtype/device metadata (`awq_sm70_gemm.cu:3062-3075`) — all host-visible
     properties — and takes the stream with `at::cuda::getCurrentCUDAStream()` (`:3050`).
     Our PXQ kernels must not allocate scratch inside the op; any workspace must be a
     load-time-allocated persistent buffer passed in, or drawn from the caching allocator via
     `torch::empty` on the capture stream (acceptable, but a fixed persistent buffer is safer).
  6. **Weight tensors must be materialised and address-stable before capture.** The template
     achieves this by doing all repacking in `process_weights_after_loading` and then shrinking
     the originals to zero-element Parameters (`compressed_tensors_wNa16.py:255-278`).

* One thing to *avoid* copying: the template's `SM70TurboMindLinearState` holds the repacked
  weights as **plain dataclass fields, not `nn.Parameter`s** (`sm70_turbomind.py:142-151`). That
  is fine for capture but means the tensors are invisible to `model.parameters()`, memory
  accounting, and any sleep/wake or weight-reload path. For a first-class `pxq4` config, prefer
  re-registering the repacked buffers as `nn.Parameter(..., requires_grad=False)`.

### A4. TP sharding of packed weights — the exact mechanism, and how panel/slab plugs in

**Mechanism (FACT).** vLLM v2 weight loading is: `LinearBase.weight_loader_v2` dispatches to a
method **on the Parameter subclass**, not on the layer —
`linear.py:790-796` (`param.load_column_parallel_weight`), `:1763-1770`
(`param.load_row_parallel_weight`), `:1100-1180` / `:1322-1394` for the merged/QKV fused cases.
The Parameter classes live in `vllm/model_executor/parameter.py`:

* `_ColumnvLLMParameter.load_column_parallel_weight` (`parameter.py:148-154`):
  `loaded_weight.narrow(self.output_dim, tp_rank * shard_size, shard_size)` where
  `shard_size = self.data.shape[self.output_dim]`.
* `_ColumnvLLMParameter.load_merged_column_weight` (`:156-176`) and `.load_qkv_weight` (`:178-217`):
  same, but with `shard_offset`/`shard_size` first passed through
  `adjust_shard_indexes_for_packing` **iff** `packed_dim == output_dim`.
* `_adjust_shard_indexes_for_packing` (`:605-617`) is literally
  `shard_size //= packed_factor; shard_offset //= packed_factor`.
* `RowvLLMParameter.load_row_parallel_weight` (`:220-230`):
  `loaded_weight.narrow(self.input_dim, tp_rank * shard_size, shard_size)`,
  `shard_size = self.data.shape[self.input_dim]`. **No packing adjustment on the row path.**
* `BasevLLMParameter.load_row_parallel_weight` (`:102-103`) → `_assert_and_load` → **full copy,
  no narrowing**. This is the "replicated on every rank" behaviour.
* `PackedvLLMParameter(ModelWeightParameter)` (`:353-393`) carries `packed_factor`, `packed_dim`,
  and inherits both column and row loading.
* Declaration site to copy: `compressed_tensors_wNa16.py:139-215` — `PackedvLLMParameter(input_dim=1,
  output_dim=0, packed_dim=1, packed_factor=self.pack_factor, weight_loader=weight_loader,
  data=torch.empty(...))`, then `layer.register_parameter("weight_packed", weight)`.

**Everything the loader does is `narrow()` along one declared dim of one tensor.** So the panel/slab
layout has to be *shaped* so that panel/slab boundaries fall on a tensor dimension. It does.

**Proposed declaration (INFERENCE from the above, not yet built):** two parameters per PXQ4 weight.

| param | dtype | shape | attrs |
|---|---|---|---|
| `pxq4_anchor` | fp16 | `[N/64, 64]` | col-parallel: `PackedColumnParameter(output_dim=0, packed_dim=0, packed_factor=64)`; row-parallel: plain `BasevLLMParameter` (replicated) |
| `pxq4_slabs` | uint8 | `[N/64, K/32, 1088]` | `PackedvLLMParameter(output_dim=0, packed_dim=0, packed_factor=64, input_dim=1)` |

Why this works, term by term:

* **Column-parallel / merged-column / QKV (split output rows).** `narrow(0, ...)` on both tensors
  slices whole 64-row panels — exactly the "pure memcpy of whole panels" property. `packed_factor=64`
  on `output_dim=0` converts vLLM's row-unit `shard_offset`/`shard_size` into panel units at
  `parameter.py:161-165` and `:186-190`. Checked against the artifact's real shapes: fused
  `in_proj_qkvz` shard offsets `0/2048/4096/10240` → `0/32/64/160` panels; shard sizes at TP=4
  `512/512/1536/1536` → `8/8/24/24` panels. All integral. `ffn_gate_up` offset `17408 → 272`
  panels, size `4352 → 68`. All integral at TP=2 and TP=4.
* **Row-parallel (split K).** `narrow(1, ...)` on `pxq4_slabs` takes a contiguous slab subrange
  from every panel — the byte-gather is done *for* us by `narrow`, because dim 1 is the
  32-column-block axis. `shard_size = data.shape[1] = (K/tp)/32` and the offset is
  `tp_rank * shard_size` in slab units, so no packing adjustment is needed and none is applied
  (`parameter.py:220-230`) — which is exactly right. Meanwhile `pxq4_anchor` as a plain
  `BasevLLMParameter` hits `_assert_and_load` and is copied whole to every rank — the
  "duplicate the 128 B header" step, for free.
* **Numerics.** Bit-identical: no re-quantization, no scale recomputation, because
  `eff = anchor[row] * sub[block]` has no cross-K coupling
  (`ggml/src/ggml-cuda/pxa/pxq6.cuh:11-18`, `ggml/src/pxq-cpu.h:1-17`).
* **Consequence for the kernel:** the K-shard's slabs are contiguous in memory *within a panel*
  after `narrow`, but the panel stride changes (`K_full/32` → `K_shard/32` slabs). Our
  `pxq4_sm70_prepare` must therefore recompute the panel stride from the *post-shard* tensor
  shape, not from a header field. Read `K` from `pxq4_slabs.shape[1] * 32` at
  `process_weights_after_loading` time.

Geometry gate holds everywhere (rows%64, K%32) at TP=2 and TP=4 for every PXQ4 tensor class in
the artifact — that was established previously and the measured shapes below are consistent with it.

### A5. Which Linear class wraps which projection

The model is `Qwen3_5ForCausalLM` (`vllm/model_executor/models/qwen3_5.py:772`), built on
`qwen3_next.py`. Only the GDN piece is defined in `qwen3_5.py`; the rest is inherited.

| projection | class | parallel kind | our GGUF tensor | ggml type in artifact |
|---|---|---|---|---|
| full-attn `qkv_proj` | `QKVParallelLinear` (`qwen3_next.py:505`) | column (QKV) | `attn_q` + `attn_k` + `attn_v` | 252 / **8** / **8** |
| full-attn `o_proj` | `RowParallelLinear` (`qwen3_next.py:515`) | **row (K split)** | `attn_output` | 252 |
| GDN `in_proj_qkvz` | `MergedColumnParallelLinear`, `output_sizes=[key_dim,key_dim,value_dim,value_dim]` (`qwen3_5.py:214-229`) | column (merged, 4 shards) | `attn_qkv` (10240) **+** `attn_gate` (6144) | 252 + 252 |
| GDN `in_proj_ba` | `MergedColumnParallelLinear`, `output_sizes=[num_v_heads]*2` (`qwen3_5.py:231-244`) | column (merged, 2 shards) | `ssm_alpha` + `ssm_beta` | **8** (q8_0) |
| GDN `out_proj` | `RowParallelLinear` (`qwen_gdn_linear_attn.py:2181`) | **row (K split)** | `ssm_out` | **39 (MXFP4)** |
| GDN `conv1d` | `ColumnParallelLinear` w/ `mamba_v2_sharded_weight_loader` (`qwen_gdn_linear_attn.py:2103-2141`) | custom | `ssm_conv1d` | 0 (F32) |
| MLP gate/up | `MergedColumnParallelLinear` (via `Qwen2MoeMLP`) | column (merged, 2 shards) | `ffn_gate` + `ffn_up` | 252 + 252 |
| MLP down | `RowParallelLinear` | **row (K split)** | `ffn_down` | 252 |
| `lm_head` | `ParallelLMHead` | column | `output.weight` | **8** |
| `embed_tokens` | `VocabParallelEmbedding` | vocab-parallel | `token_embd.weight` | **14 (Q6_K)** |

**Two consequences you must design for:**

* **`in_proj_ba` must be split out, and the switch is our config's ignore list.**
  `_uses_split_gdn_input_projections(quant_config)` (`qwen3_5.py:127-158`) scans the config for
  attributes `modules_to_not_convert`, `ignored_layers`, `ignore`, plus `config["ignore"]`, and
  returns True iff some entry equals `"linear_attn"` or contains `"linear_attn.in_proj_b"`. The
  production AWQ config does exactly this — I read its `config.json` and it lists
  `model.language_model.layers.N.linear_attn.in_proj_b` and `.in_proj_a` for every GDN layer.
  If we do **not** expose an equivalent ignore list, `create_qkvz_proj` appends
  `[num_v_heads, num_v_heads]` to `output_sizes` (`qwen3_5.py:221-222`) and the b/a rows get
  folded into the quantized fused projection — where at TP=4 the per-rank shard is
  `48/4 = 12` rows, **violating rows%64**. So: `PXQ4Config` must carry an `ignore` list
  naming `linear_attn.in_proj_b` / `in_proj_a`, and `ssm_alpha`/`ssm_beta` (q8_0 in our file)
  must be served by a non-PXQ4 method. This is not optional.
* **Only three of the eight quantized classes are row-parallel** (`o_proj`, GDN `out_proj`,
  `ffn_down`). Those are the only ones needing the slab-subrange K-split. Everything else is
  whole-panel memcpy.

---

## B. Weight loading

### B1. Does the fork's GGUF loader carry a custom type id? **NO — hard failure, verified.**

FACT, measured inside the container:

```
gguf pkg: /opt/vllm-venv/lib/python3.12/site-packages/gguf/__init__.py
num types: 34, max id: 41, 252 in set: False
gguf.GGMLQuantizationType(252) -> ValueError: 252 is not a valid GGMLQuantizationType
```

And the failure point is unrecoverable at *file* granularity, not tensor granularity. I read
`gguf.gguf_reader.GGUFReader._build_tensors` from the installed package:

```python
ggml_type = GGMLQuantizationType(raw_dtype[0])          # <-- ValueError on 252
...
block_size, type_size = GGML_QUANT_SIZES[ggml_type]     # <-- KeyError even if the enum passed
```

This runs in the constructor's tensor-table pass, over **all** tensors, before anything is
yielded. One PXQ4 tensor kills the whole `GGUFReader(...)` call. `vllm` calls this from
`model_loader/weight_utils.py:1293` (`gguf_quant_weights_iterator`), `:1337`
(`..._multi`), and `:1273` (`get_gguf_weight_type_map`), all reached from
`model_loader/gguf_loader.py:22-28`.

Downstream is equally closed even if you patched the reader:

* `quantization/gguf.py:13` — `from gguf import GGMLQuantizationType as WeightType`, and
  `:163-193` builds `UNQUANTIZED_TYPES / STANDARD_QUANT_TYPES / KQUANT_TYPES / IMATRIX_QUANT_TYPES`
  → `DEQUANT_TYPES / MMVQ_QUANT_TYPES / MMQ_QUANT_TYPES`. 252 is in none of them.
* `gguf.py:485-491` (`GGUFLinearMethod.process_weights_after_loading`) raises
  `ValueError(f"Unsupported GGUF quantization type ...")` for anything outside those sets.
* `gguf.py:199-229` (`_fused_mul_mat_gguf`) dispatches to `ops.ggml_mul_mat_vec_a8` /
  `ggml_mul_mat_a8` / `ggml_dequantize` — vLLM's **own vendored ggml `_custom_ops`**, a different
  and much older ggml than our fork. 252 is unknown there too, at the C level.
* `gguf.py:469-470` stores the per-shard type as `torch.empty(..., dtype=torch.uint8)`. 252 fits
  in uint8 by luck; **PXQ6 = 256 would not.** Worth knowing before the ladder moves.
* Independently, the brief's warning stands: `GGUFUninitializedParameter` + `_create_padded_weight_param`
  (`gguf.py:452-500`) shard by assuming per-row-contiguous blocks, which 64-row panel interleave
  violates.

**Conclusion: option (a) "extend the GGUF loader" is dead as stated.** It is not one patch to the
fork; it is a patch to the third-party `gguf` pip package *plus* the fork's `gguf.py` *plus* the
vendored ggml C ops, and it lands us on a loader whose sharding model is wrong for our layout
anyway.

### B2. The three options, assessed

| | effort | risk | verdict |
|---|---|---|---|
| **(a) extend the GGUF loader to pass PXQ4 through as raw uint8** | high | high | **Reject.** Requires forking the upstream `gguf` PyPI package (`GGMLQuantizationType` enum + `GGML_QUANT_SIZES`), then patching `quantization/gguf.py`'s five type sets and its `_fused_mul_mat_gguf` dispatch, then *still* bypassing `GGUFUninitializedParameter`'s row-contiguous sharder because panel interleave breaks it. Every one of those is a vendored-file edit that a fork rebase will silently revert. It also drags in the whole `GGUFModelLoader` name-mapping machinery (`gguf_loader.py:108-175`) which is written around dense/MoE llama naming. |
| **(b) offline GGUF→safetensors converter, PXQ4 as raw uint8 + sidecar json** | **low-to-moderate** | **low** | **RECOMMENDED.** See B3. |
| **(c) re-quantize from HF weights into a vLLM-native layout** | very high | very high | **Reject.** Throws away the artifact we already have and validated, forces a second quantizer implementation to be kept bit-compatible with `pxq6.cuh`/`src/pxq6r-quantize.inc.cpp` forever, and every future PXQ tier change becomes a two-place change. It also buys nothing: the kernels want the panel/slab layout regardless of what file it arrived in. |

Option (b) also has a property the others don't: **it never touches the fork.** The converter is a
standalone script; the runtime side is an out-of-tree pip package (§A2). Kewaii's tree stays clean,
which matters if this is going to be offered upstream.

### B3. The recommended path in detail — and how it handles five types

**Converter (offline, CPU, runs anywhere with the file):**

1. Parse the GGUF header with **our own** reader (the raw struct parser I already wrote and ran —
   `prototypes/pxq-vllm/ggufscan.py`), *not* the `gguf` package. 30 lines: magic/version/counts,
   KV table, tensor table (name, ndim, dims, type id, offset), then `data` at the aligned base.
   This sidesteps the enum entirely and is the whole reason (b) is cheap.
2. For each tensor, emit into safetensors according to its ggml type:
   * **252 (PXQ4)** → **two** entries: `<name>.pxq4_anchor` fp16 `[N/64, 64]` and
     `<name>.pxq4_slabs` uint8 `[N/64, K/32, 1088]`. This is a pure byte de-interleave of the
     panel blob (header split from slabs); it is the *only* transform, and it is what makes
     vLLM's `narrow`-based sharder work (§A4).
   * **39 (MXFP4)** → `ssm_out` only. Either emit as raw uint8 + scales and serve it with the
     fork's already-shipped `prepare_mxfp4_linear` / `mxfp4_gemm_sm70_out`
     (`sm70_turbomind.py:229-253`, `torch_bindings.cpp:198-200`) — **the fork already has a
     working sm70 MXFP4 GEMM**, so this is nearly free — or dequantize the 48 tensors to fp16
     (they are 6144×5120 each, ~60 MiB fp16 apiece, 2.9 GiB total across 48 — too much, so use
     the MXFP4 path).
   * **8 (Q8_0)** → dequantize to fp16 on the CPU during conversion. 132 tensors: `attn_k`,
     `attn_v` (17 each, 5120×1024), `ssm_alpha`, `ssm_beta` (48 each, 5120×48 — tiny),
     `output.weight` (5120×248320 → 2.4 GiB fp16, the only big one), `eh_proj`. Q8_0 dequant is
     trivial and exact-ish; serving these as plain fp16 through `UnquantizedLinearMethod` costs
     nothing in complexity. **Alternative for `output.weight` if 2.4 GiB is unacceptable:** keep
     it Q8_0 and route the LM head through the fork's existing int8 path — but note it is
     column-parallel and small in the decode budget; fp16 is the low-risk default.
   * **14 (Q6_K)** → `token_embd` only. Dequantize to fp16 (5120×248320 → 2.4 GiB). Embeddings
     are gathered, not GEMM'd; there is no reason to keep them quantized.
   * **0 (F32)** → norms, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, 360 tensors, all 1-D or tiny.
     Copy through as f32 (or cast to fp16 to match `dtype=torch.float16`; keep norms in f32,
     vLLM handles that).
3. Rename ggml names → HF/vLLM names (`blk.N.ffn_gate` → `model.language_model.layers.N.mlp.gate_proj`,
   etc.). The mapping is already implied by the AWQ twin's tensor names — dump them from
   `/path/to/hf/<reference-hf-model>` and match one-for-one.
   **Fuse `attn_qkv` (10240 rows) + `attn_gate` (6144 rows) → `linear_attn.in_proj_qkvz`
   (16384 rows)** by concatenating panels along dim 0 — legal because panels are self-contained
   (`pxq-cpu.h:1-17`). Keep `ssm_alpha`/`ssm_beta` as `in_proj_a`/`in_proj_b` (separate, per §A5).
4. Write `config.json` copied from the AWQ twin with `quantization_config` replaced by:
   ```json
   {"quant_method": "pxq4",
    "pxq4_tier": "core", "pxq4_backbone_rev": "<from pxa.pxq.backbone_rev>",
    "tensor_types": {"<vllm module prefix>": "pxq4" | "mxfp4" | "fp16", ...},
    "ignore": ["...linear_attn.in_proj_b", "...linear_attn.in_proj_a", "lm_head", ...]}
   ```
   The `pxa.pxq6.book` and `pxa.pxq6.sub` KVs (16 fp32 each, read from the file) go in here too —
   the PX16 book and SUB16 table are the frozen constants the kernel needs.

**Runtime side — how per-tensor types are dispatched (this is the answer to §7):**

`QuantizationConfig.get_quant_method(self, layer, prefix)` (`base_config.py:150-163`) is called
**once per module, with the module's full dotted prefix**. That is the per-tensor hook, and it is
exactly how compressed-tensors already handles a mixed checkpoint:
`compressed_tensors.py:146-186` returns `UnquantizedLinearMethod()` when `get_scheme()` finds no
match for that prefix, and a `CompressedTensorsLinearMethod` when it does.

So `PXQ4Config.get_quant_method` becomes a three-way lookup against the sidecar `tensor_types`:

```python
def get_quant_method(self, layer, prefix):
    if not isinstance(layer, LinearBase):
        return None                      # VocabParallelEmbedding, Attention -> default
    kind = self._lookup(prefix)          # from config["tensor_types"], suffix-matched
    if kind == "pxq4":  return PXQ4LinearMethod(self)
    if kind == "mxfp4": return PXQ4MXFP4LinearMethod(self)   # reuses sm70_turbomind.prepare_mxfp4_linear
    return UnquantizedLinearMethod()     # fp16: attn_k/v, in_proj_a/b, lm_head
```

`self._lookup` must handle the **fused** modules: `qkv_proj` covers `attn_q`(252)+`attn_k`(8)+`attn_v`(8),
which is a genuinely mixed fused layer. **Simplest correct answer: dequantize `attn_q` to fp16 too**
and serve the whole `qkv_proj` unquantized. It is 17 layers × 5120×3072 = 0.5 GiB fp16 total — a
1.4% weight-budget cost against the alternative of a per-shard-type fused kernel. `gate_up_proj`
(`ffn_gate`+`ffn_up`) is uniformly 252, and `in_proj_qkvz` is uniformly 252, so those stay fused
and quantized. **INFERENCE, not measured:** this leaves ~92% of GEMM bytes on the PXQ4 path
(`ffn_gate/up/down` = 195 tensors of 325, plus `in_proj_qkvz`/`attn_gate` 96, plus `attn_output` 17).

**Loader:** none needed. Once the file is safetensors, the stock `DefaultModelLoader` handles it,
and our parameters get filled through the standard `weight_loader_v2` path in §A4. That is the
single biggest reason to prefer (b): **the entire GGUF loading problem disappears rather than being
solved.**

### B4. Residual risks with option (b)

1. **Converter output size.** Dequantizing `token_embd` (Q6_K) and `output.weight` (Q8_0) to fp16
   costs ~4.8 GiB on top of the 14.64 GiB artifact. Per-GPU at TP=4 this is fine (embeddings and
   LM head are vocab/column parallel), but the on-disk safetensors will be ~19 GiB. Confirm free
   space **under `/path/to/models`** — never `/` (100% full).
2. **Name mapping is the fiddly part**, not the bytes. Budget the time there, and validate by
   diffing the converted key set against the AWQ twin's key set before any load attempt.
3. **`in_proj_qkvz` panel concatenation** assumes the four sub-projections' row counts are each
   multiples of 64 in the *fused* tensor's coordinate system (2048/2048/6144/6144 — yes) and that
   the GGUF `attn_qkv` row order is q,k,v (INFERENCE from the 2048+2048+6144=10240 arithmetic;
   **not verified** against the quantizer's tensor-splitting code — check `attn_qkv` construction
   in the conversion tool before trusting it).
4. **MXFP4 `ssm_out` via the fork's existing path** assumes the fork's MXFP4 packing convention
   matches ggml's type-39 packing. **Not verified.** If it doesn't, the fallback is dequantizing
   those 48 tensors to fp16 (2.9 GiB) or extending our own kernel to a second format. Check this
   early — it is the one place where "the fork already has it" might not hold.

---

## Summary of what to build

1. `pxq4_gguf2st.py` — standalone converter, own GGUF header parser, five type handlers, name
   mapper, sidecar `quantization_config`. **No vLLM dependency.**
2. `pxq4-vllm` pip package:
   * `csrc/pxq4_sm70.cu` — `TORCH_LIBRARY(pxq4, ...)` with `pxq4_sm70_prepare(...) -> Tensor[]`
     and `pxq4_gemm_sm70_out(Tensor(a!) out, ...) -> ()`, ported from
     `ggml/src/ggml-cuda/pxa/pxq6.cuh` (**not** `pxq4.cuh` — that file documents the retired id-250
     format, per `ggml.h:458`). Built with `CUDAExtension`, `-gencode arch=compute_70,code=sm_70`.
   * `pxq4_vllm/ops.py` — `_op()` wrappers plus `@register_fake` meta kernels for both ops
     (mandatory for `FULL_AND_PIECEWISE`).
   * `pxq4_vllm/config.py` — `@register_quantization_config("pxq4") class PXQ4Config`, with the
     parameter declarations of §A4 and the three-way `get_quant_method` of §B3.
   * entry point in group `vllm.general_plugins`.
3. Build it in a **fresh** container from the same image with a writable volume under
   `/path/to/models` — the production container's overlay is 100% full and must not be written to.
