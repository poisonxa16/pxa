# 07 — PXQ4 tensor-parallel sharding: reconciled verdict

**This does not kill the project.** PXQ4 shards for tensor parallelism at TP=4 and TP=2 with
no re-quantization, no permutation of weight values, and bit-identical numerics in both split
directions. All three analyses reached that conclusion independently and I confirmed the
load-bearing claims against source. The genuine risk in this project is *not* sharding — it is
the mixed-type tail (§6) and the economics (§7).

---

## 1. Where the three agents AGREE (treat as established)

Unanimous, and each point independently re-verified below or in the recon docs:

| Claim | Status |
|---|---|
| Column-parallel (split output rows) at any multiple of **64** = whole-panel memcpy, zero overhead, zero permutation | AGREE, verified |
| Row-parallel (split K) at any multiple of **32** = duplicate the 128 B panel header + copy the contiguous slab run; bit-identical | AGREE, verified |
| **No per-shard re-quantization is required at any TP degree** | AGREE, verified |
| One converted checkpoint serves TP=1/2/4; no per-TP artifacts | AGREE |
| Every real tensor shape in this model passes both gates at TP=2 *and* TP=4 | AGREE, arithmetic reproduced |
| A weight row is **not** contiguous bytes → vLLM's generic GGUF sharder cannot be used | AGREE, verified |
| `in_proj_ba` (ssm_alpha/ssm_beta, 48 rows) must be split out of the fused GDN projection | AGREE, verified |
| Never set `_sm70_f16_prepared` / `_awq_sm70_prepared` | AGREE, verified |
| The checkpoint is mixed-type and any uniform-PXQ4 design fails at load | AGREE, verified |

Agreement here is strong evidence: three independent readings of the same source converged on
identical alignment constants and identical loader mechanics.

---

## 2. Disagreements, settled against source

### 2.1 attn_q row order — is `[q_h | gate_h]` per-head interleaved?
**column-parallel said YES (verified). moe-and-mixed said NOT VERIFIED and flagged it as one of
the two largest remaining unknowns. column-parallel is RIGHT.**

FACT, `/path/to/engine-repo/src/llama-build-context.cpp:1993-2007`
(`llm_build_mul_mat_qkv_gated`):
```
auto Qaux     = llm_build_lora_mm(lctx, ctx0, wq, cur);              // :1994
auto row_size = ggml_row_size(Qaux->type, n_embd_head_k);            // :2003
auto Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, Qaux, n_embd_head_k,
              Qaux->ne[0]/(2*n_embd_head_k), n_tokens, 2*row_size, Qaux->nb[1], 0));       // :2005
auto gate = ggml_cont_2d(ctx0, ggml_view_3d(ctx0, Qaux, n_embd_head_k,
              Qaux->ne[0]/(2*n_embd_head_k), n_tokens, 2*row_size, Qaux->nb[1], row_size), ...);// :2007
```
Stride `2*row_size` with q at offset 0 and gate at offset `row_size` is **per-head interleave**:
`[q_h0(256) | gate_h0(256) | q_h1(256) | ...]`, 24 pairs. This is the same order
`qwen3_next.py:565-567` reconstructs (`view(..., num_heads, -1)` then `chunk(2, dim=-1)`).
**No permutation in the converter.** A contiguous 3072-row (TP=4) / 6144-row (TP=2) slice is
semantically correct. The moe agent's fallback ("a 256-row = 4-panel reorder") is moot but was
the right shape of contingency.

### 2.2 Does the row-parallel loader consult `packed_factor`?
**row-parallel said NO (packed attrs are inert on the K axis). VERIFIED, on BOTH loader paths.**

- v1: `RowParallelLinear.weight_loader`, `/opt/1Cat-vLLM/vllm/model_executor/layers/linear.py:1728-1761`.
  Reads only `input_dim`; `packed_dim`/`packed_factor` never appear. The narrow is
  `shard_size = param_data.shape[input_dim]; start_idx = self.tp_rank*shard_size;
  loaded_weight.narrow(input_dim, start_idx, shard_size)` at :1749-1752, guarded by
  `if input_dim is not None` (:1749), then `assert param_data.shape == loaded_weight.shape` (:1760).
- v2: `RowvLLMParameter.load_row_parallel_weight`, `parameter.py:220-230` — identical, no packing math.
- The anchor parameter, declared with `output_dim` and **no** `input_dim`, therefore falls through
  to `BasevLLMParameter.load_row_parallel_weight` → `_assert_and_load` (`parameter.py:102-103, 93-97`),
  i.e. a **full copy**. That *is* the required 128 B header duplication, obtained with no custom loader.

So the two agents' parameter shapings are not in conflict — `packed_dim:0/packed_factor:64`
governs the column axis only (`linear.py:1556-1560`, `parameter.py:606-609`), and is ignored on the row axis.
Both recommend the same declaration and it is correct:
```
pxq_slab   uint8 [R//64, K//32, 1088]  {output_dim:0, input_dim:1, packed_dim:0, packed_factor:64}
pxq_anchor fp16  [R//64, 64]           {output_dim:0}      # NO input_dim
```

### 2.3 `ssm_out` — omitted by column-parallel, flagged by the other two. The other two are RIGHT.
FACT (tensor census, `out.json`): `blk.N.ssm_out.weight`, **48 tensors, ne=(6144,5120), ggml type 39
(MXFP4), 802,160,640 B**. It is the GDN out_proj and it is row-parallel. A PXQ4-only row-parallel
design leaves 48 of 65 blocks unserved. This is a scope gap in the column-parallel analysis, not a
contradiction — but it is real and it is the largest single VRAM item in the tail.

### 2.4 The "3.66 GiB/GPU" premise in the brief
**moe-and-mixed is RIGHT that the brief's figure is wrong, and I reproduced its arithmetic
independently.** Census byte totals: PXQ4 12,231,950,336 · Q8_0 1,621,032,960 · Q6_K 1,042,944,000 ·
MXFP4 802,160,640 · F32 10,686,464 = 15,708,774,400 B = 14.63 GiB (matches the artifact).
Only **11.39 GiB is PXQ4**; at TP=4 that is 2.848 GiB/GPU. Independent recomputation of the
moe agent's Policy A (everything non-PXQ4 → fp16) gives **5222.2 MB = 4.864 GiB/GPU at TP=4**,
matching their 4.863 to the third decimal. Policy A is *worse* than the incumbent AWQ 4.64 GiB/GPU.
See §7.

### 2.5 A fused `[k;v]` parameter would mis-split at TP=2 (moe agent)
Correct as a warning about a *custom* parameter, but it does not apply to stock vLLM:
`QKVParallelLinear.weight_loader` computes q/k/v offsets separately per `shard_id`
(`linear.py:1538-1546`) before the packing divide (`:1556-1560`) and narrows the *loaded* tensor at
`start_idx = shard_rank*shard_size` (`:1596-1600`). Only an invented fused-kv param loaded through a
plain column loader would hand rank0 all of K. Keep k and v as separate shard ids — which the stock
loader already does.

### 2.6 46 KB smem cap (kernels recon) — row-parallel agent could not verify it
Still **NOT VERIFIED** here; I did not re-check `ggml-cuda.cu:4262`. Immaterial to sharding: at TP=4
every staged-x figure is ≤20.5 KB and at TP=2 ≤34.8 KB, so both sit below even the conservative cap.
The direction of the effect is unambiguous — K-splitting only ever *reduces* the staged-x footprint.

---

## 3. BOTTOM LINE

### TP = 4 (DGX, 4×V100-32GB, NVLink): **SHARDS. No re-quantization.**
- Column-parallel: whole-panel memcpy. Every partition is a multiple of 64 rows.
  q|gate 3072 (48 panels) · k 256 (4) · v 256 (4) · GDN qkvz 512/512/1536/1536 (8/8/24/24,
  cumulative offsets 0/512/1024/2560 → 0/8/16/40 panels) · gate_up 4352 each (68 panels).
  Smallest object anywhere = 8 panels. Overhead **exactly zero bytes**.
- Row-parallel: header duplication only. ffn_down K 17408→4352 (136 slabs); attn_output
  6144→1536 (48); ssm_out 6144→1536. All %32. Cost +0.065% (ffn_down) / +0.184% (attn_output)
  = **0.60 MiB/rank total**.

### TP = 2 (Unraid, 2×V100-16GB, PHB/no NVLink): **SHARDS. No re-quantization.**
Identical conclusion, strictly more slack: q|gate 6144 (96 panels), k/v 512 (8), GDN 1024/1024/3072/3072
(offsets 0/16/32/80 panels), gate_up 8704 (136), ffn_down K 8704 (272 slabs), attn_output K 3072 (96).
Header-duplication cost 0.40 MiB/rank. **There is no shape in this model that passes at TP=2 and fails
at TP=4, or vice versa.** The only TP=2-specific item is a *performance* one, not a correctness one:
the AWQ down_proj tile-overlap all-reduce (`linear.py:100-119`, `:122-175`) gates on
`tp_size == 2 and layer._awq_sm70_prepared`, so we forgo it on the box where the all-reduce is most
expensive. That is a comms optimization gap, not a sharding blocker.

### Is per-shard re-quantization required? **NO. In either direction, at either degree.**
Structural proof, not assertion:
- The dequant contract has **zero cross-K and zero cross-row coupling**:
  `eff = fp32(anchor_fp16) * SUB16[s4]`, `w = eff * BOOK[code]`. The anchor is read **once** from the
  panel header before the K loop and never re-derived —
  `/path/to/engine-repo/ggml/src/pxq-cpu.c:141` (`const float anchor = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t*)panel)[r]);`),
  loop body `:143-157`. CUDA policy is identical (`ggml/src/ggml-cuda/pxa/pxq6.cuh:323-330`, consumed `:700-712`).
- The kernels take **`kslabs` as a runtime argument** and derive the panel stride from it:
  `pxq6_panel_stride<POL>(kslabs) = POL::HDR + kslabs*POL::SLAB` (`pxq6.cuh:519-522`) and
  `pxq6_panel(W,e,panels,p,kslabs) = W + (e*panels + p)*stride` (`pxq6.cuh:523-526`). Global K and
  absolute row appear **nowhere**. A shard handed to the kernel as its own base pointer with its own
  `kslabs` and `panels` is a first-class, self-describing PXQ4 tensor.
- Re-quantizing would *change the bytes* and forfeit parity with the quality-gated artifact:
  `pxq6_pick_anchor` takes the absmax over the **whole** K (`src/pxq6-quantize.inc.cpp:264-268`), so a
  K-shard would get a smaller anchor and every SUB16 index and code would be re-searched. It buys an
  unmeasured, structurally bounded gain (SUB16 spans 0.214600…0.987793, a 4.60× range —
  `ggml/include/ggml-pxq6-tables.h:40-44`) at the cost of a checkpoint per TP degree. **Do not do it.**

---

## 4. Q4 — could our quantizer re-quantize an already-split matrix if we had to?

**The algorithm can today; the tooling cannot.** Checked, not assumed.

FACT, `/path/to/engine-repo/src/pxq6-quantize.inc.cpp:287-289`:
```
static void pxq6_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                 const float * imx, int tier,
                                 int64_t row0 /*absolute row of src's first row*/)
```
It already takes an arbitrary `[R,K]` fp32 block plus an **absolute row offset**, and the threading
path already calls it on 64-row-aligned sub-ranges of a larger tensor
(`pxq6-quantize.inc.cpp:416-420`, jobs of `CHUNK` panels). So a column shard is exactly the input
shape it consumes; a K shard is exactly a different `K` argument. No new algorithm needed.

Three caveats that matter:
1. It is a **`static`** function inside an `.inc.cpp` compiled into `llama-quantize` — **no exported
   symbol, no public API.** `llama-quantize`'s CLI operates on whole GGUF files only. Using it on a
   split matrix needs a new exported entry point plus a small standalone driver. Call it new tooling,
   small.
2. `row0` **seeds the deterministic tie-break** — `pxq_tie_take_hi(row, blk)`
   (`pxq6-quantize.inc.cpp:49`, used at `:230`), with the in-source warning at `:416-418` that a
   chunk-local row "would flip the deterministic tie-break and make the artifact differ across thread
   counts." Consequence: **re-quantizing a column shard as its own tensor starting at row 0 produces
   different bytes from the parent for identical weights.** Another reason the memcpy is the right answer.
3. The eligibility gate is `ne[1] % 64 == 0 && ne[0] % 32 == 0` (`src/llama-quantize.cpp:1119-1122`,
   used by `pxq4_tensor_eligible` `:1485-1497`) — every shard in this model passes, so the gate is not
   the obstacle either.

Bottom line: **we are not blocked on quantizer capability, and we do not need to exercise it.**

---

## 5. Corrections to the brief's own framing

- **Filename trap confirmed.** `ggml/src/ggml-cuda/pxa/pxq4.cuh:1-12` documents an **E8M0 per-row scale
  byte, "numerics are EXACTLY MXFP4"** layout — that is the retired id-250 format, *not* what is in the
  artifact. The live id-252 codec is the fp16-anchor + SUB16 family: `pxa_pxq_dequant_row` dispatches
  `GGML_TYPE_PXQ4 → pxa_deq_row_pxq6(..., hq=false)` (`ggml/src/pxq-cpu.c`, switch in
  `pxa_pxq_dequant_row`), constants in `ggml/include/ggml-pxq6-tables.h:21-27`
  (`QK 32 / TYPE_SIZE 17 / BM 64 / SLAB 1088 / HDR 128 / ROW_META 2`). Port from **pxq6.cuh**.
- **The incumbent AWQ ignore list is more generous than assumed** (read from
  `/path/to/hf/<reference-hf-model>/config.json`, 311 entries):
  it contains `lm_head`, every `mtp.*` linear, all `visual.*`, and per-layer
  `linear_attn`, `linear_attn.norm`, `linear_attn.in_proj_b`, `linear_attn.in_proj_a`.
  So **their 4.64 GiB/GPU carries an fp16 lm_head and an fp16 MTP block** — confirming the moe agent's
  stated assumption, and meaning a PXQ4 lm_head is an advantage the incumbent structurally does not have.
  (Whether the bare `linear_attn` entry also exempts `in_proj_qkvz`/`out_proj` is **NOT VERIFIED** —
  the presence of explicit `in_proj_a/b` children suggests parent entries do not propagate.)
- The same list confirms `in_proj_b`/`in_proj_a` are ignored in production, which is exactly what
  `_uses_split_gdn_input_projections` matches on
  (`/opt/1Cat-vLLM/vllm/model_executor/models/qwen3_5.py:127-157`: scans
  `modules_to_not_convert` / `ignored_layers` / `ignore` / `config['ignore']` for `linear_attn`,
  `.linear_attn`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`). Our `PXQ4Config` must expose the
  same, or the fused GDN projection gains `[48, 48]` rows → 12/rank at TP=4, which can never satisfy
  `rows % 64`. **Geometrically mandatory, as the moe agent said.**

## 6. The sm70 fast-path hazard — resolved, and less scary than reported
All three warned about `_sm70_f16_force_enable`. Settled: `_mark_default_sm70_dense_modules`
(`qwen3_5.py:159-181`) does set `_sm70_f16_force_enable = True` on every module whose prefix ends in
`qkv_proj` or `out_proj` — i.e. on our column-parallel qkv_proj *and* the row-parallel GDN out_proj.
But `_maybe_sm70_dense_forward` (`linear.py:56-96`) returns `None` unless
**`_sm70_f16_prepared`** is set (`:62-63`), which only a `process_weights_after_loading` sets. Since our
`PXQ4LinearMethod` owns that hook and will never set it, the interception is **inert**. There is also an
explicit opt-out, `_sm70_f16_forbidden` (`:57-58`) — set it defensively on every PXQ4 layer.
Same structure for the AWQ down-tile path: it requires `_awq_sm70_prepared` (`linear.py:107-109`).
Both interceptors are called *before* `quant_method.apply()` (`linear.py:1789-1795`), so the flags
are the only thing standing between us and AWQ kernels reading PXQ4 bytes.

## 7. What actually threatens the project (not sharding)
The moe agent's headline stands and is the most important finding of the three reports:
**"PXQ4 kernel + fp16 everything else" is a regression, not a win.** At TP=4, Policy A = 4.864 GiB/GPU
vs the incumbent's 4.64. The premise only holds if `ssm_out` (0.75 GiB packed, 48 tensors) and the LM
head (`output.weight`, Q8_0, 1.35 GiB) are also served at 4 bits. Both are geometrically eligible for
PXQ4 (`ssm_out` 5120 rows %64, K 6144 %32; `output` 248320 rows %64, K 5120 %32). All throughput
numbers here and in the source reports are **PROJECTIONS** under the stated bandwidth-bound assumptions;
this study performed no GPU runs.

---

## 8. THE SINGLE MOST IMPORTANT CONSTRAINT

**Every tensor-parallel split must be expressed in whole panels (dim0, 64 rows) and whole slabs
(dim1, 32 columns), never in weight rows or columns — and the 128 B fp16 anchor header must travel
with its panel on a column split and be duplicated verbatim on a K split.**

This is the one constraint whose violation is **silent**. vLLM's packing adjustment is
`shard_size = round(shard_size // packed_factor); shard_offset = round(shard_offset // packed_factor)`
(`linear.py:1557-1559`, and `parameter.py:606-609` for the v2 path): a misaligned offset **truncates
without raising**, yielding a wrong-but-well-formed panel slice and a model that loads cleanly and
produces subtly wrong logits. And because a PXQ4 weight row is scattered across its panel's slabs
rather than contiguous (`ggml/src/pxq-cpu.h:5-11` — ggml's own `to_float`/`vec_dot` are NULL on purpose
for exactly this reason), any code that slices the tensor blob linearly produces garbage rather than an
error.

Enforcement, both cheap:
1. In `create_weights`, assert `output_size_per_partition % 64 == 0` and
   `input_size_per_partition % 32 == 0` per output partition, and raise — do not rely on `round(//)`.
2. Gate the converter with **shard-then-dequant vs dequant-then-shard, bit-identical in fp32**, against
   `pxa_pxq_dequant_2d` (`ggml/src/pxq-cpu.c`, which itself asserts `nrows % 64 == 0 && k % 32 == 0`).
   That single test proves the TP repack is a permutation and not a re-quantization, and simultaneously
   validates panel arithmetic, header handling and the book/SUB16 constants.
