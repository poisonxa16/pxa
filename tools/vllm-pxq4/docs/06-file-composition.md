# 06 — Actual type composition of Qwen3.8-27B-PXQ4.gguf

Source of truth: struct-parsed GGUF header of `/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf`
on the DGX (script: `/path/to/models/pxa-build-cache/tc.py`, local copy
`<scratch>/pxq-vllm/tc.py`,
raw JSON `.../pxq-vllm/out.json`). Only the header + tensor directory were read
(first 10,997,184 bytes); no tensor data was touched.

`GGUF` v3, `n_tensors = 866`, `n_kv = 57`, `general.alignment = 32`,
data section starts at byte 10,997,184, file size 15,719,771,584 (14.638 GiB).

## HEADLINE — one surprise vs the documented backbone

**`ssm_out.weight` is MXFP4 (ggml type 39), not PXQ4 and not q8_0.** 48 tensors,
802,160,640 bytes (0.75 GiB, 5.1% of the file). This is a **fifth** type the vLLM
backend must handle, and it was not in the brief's expected set. Everything else
matches the rev-2 backbone table. There are **zero expert tensors** — the MoE
question is closed by the directory itself: no `*_exps`, no `ffn_*_shexp`, no
`ffn_gate_inp`.

## 1. Type histogram (FACT — from the tensor directory; bytes derived from consecutive offsets, and independently confirmed against the closed-form size of every type, exact to the byte)

| id | name | count | bytes | GiB | % file |
|---:|---|---:|---:|---:|---:|
| 252 | **PXQ4** | 325 | 12,231,950,336 | 11.392 | 77.9% |
| 8 | Q8_0 | 132 | 1,621,032,960 | 1.510 | 10.3% |
| 14 | Q6_K | 1 | 1,042,944,000 | 0.971 | 6.6% |
| 39 | **MXFP4** | 48 | 802,160,640 | 0.747 | 5.1% |
| 0 | F32 | 360 | 10,686,464 | 0.010 | 0.07% |
| | **total** | **866** | **15,708,774,400** | **14.630** | |

Type-39 name resolved in-tree: `GGML_TYPE_MXFP4 = 39, // so we are compatible with
mainline` — `ggml/include/ggml.h:424`. `GGML_TYPE_PXQ4 = 252` — `ggml/include/ggml.h:469`.

Absent: PXQ4HQ (253), PXQ6 (256), PXQ1/2/3, F16, BF16. **No F16 tensors at all** —
the documented `attn_gate_head=f16` rule never fires on this model because every
`attn_gate` here is per-channel (ne[1]=6144 > 256).

`general.file_type = 38` (INFERENCE: a PXA-local PXQ4 ftype; the only `MOSTLY_*`
enumerator I could locate near it is `GGML_FTYPE_MOSTLY_MXFP4 = 25` at
`ggml/include/ggml.h:574`, so 38 is not verified by name — treat as opaque).

## 2. Per-pattern type map (FACT)

`blk.N` collapsed. `ne` printed GGUF-order = `ne[0] x ne[1]` = **K x rows**
(K = input/contraction dim, rows = output dim).

| tensor pattern | type | n | ne (K x rows) | which blocks |
|---|---|---:|---|---|
| `token_embd.weight` | **Q6_K** | 1 | 5120 x 248320 | — |
| `output.weight` | **Q8_0** | 1 | 5120 x 248320 | — |
| `output_norm.weight` | F32 | 1 | 5120 | — |
| `blk.N.attn_norm.weight` | F32 | 65 | 5120 | 0–64 |
| `blk.N.post_attention_norm.weight` | F32 | 65 | 5120 | 0–64 |
| `blk.N.ffn_gate.weight` | **PXQ4** | 65 | 5120 x 17408 | 0–64 |
| `blk.N.ffn_up.weight` | **PXQ4** | 65 | 5120 x 17408 | 0–64 |
| `blk.N.ffn_down.weight` | **PXQ4** | 65 | 17408 x 5120 | 0–64 |
| **GDN / DeltaNet layers (48)** | | | | il % 4 != 3, il<64 |
| `blk.N.attn_qkv.weight` | **PXQ4** | 48 | 5120 x 10240 | 0,1,2,4,5,6,… |
| `blk.N.attn_gate.weight` | **PXQ4** | 48 | 5120 x 6144 | same |
| `blk.N.ssm_out.weight` | **MXFP4** | 48 | 6144 x 5120 | same |
| `blk.N.ssm_alpha.weight` | Q8_0 | 48 | 5120 x 48 | same |
| `blk.N.ssm_beta.weight` | Q8_0 | 48 | 5120 x 48 | same |
| `blk.N.ssm_conv1d.weight` | F32 | 48 | 4 x 10240 | same |
| `blk.N.ssm_a` | F32 | 48 | 48 | same |
| `blk.N.ssm_dt.bias` | F32 | 48 | 48 | same |
| `blk.N.ssm_norm.weight` | F32 | 48 | 128 | same |
| **Full-attention layers (17)** | | | | il % 4 == 3, plus 64 |
| `blk.N.attn_q.weight` | **PXQ4** | 17 | 5120 x 12288 | 3,7,11,…,63,**64** |
| `blk.N.attn_k.weight` | **Q8_0** | 17 | 5120 x 1024 | same |
| `blk.N.attn_v.weight` | **Q8_0** | 17 | 5120 x 1024 | same |
| `blk.N.attn_output.weight` | **PXQ4** | 17 | 6144 x 5120 | same |
| `blk.N.attn_q_norm.weight` | F32 | 17 | 256 | same |
| `blk.N.attn_k_norm.weight` | F32 | 17 | 256 | same |
| **MTP block (blk.64 only)** | | | | |
| `blk.64.nextn.eh_proj.weight` | **Q8_0** | 1 | 10240 x 5120 | 64 |
| `blk.64.nextn.enorm.weight` | F32 | 1 | 5120 | 64 |
| `blk.64.nextn.hnorm.weight` | F32 | 1 | 5120 | 64 |
| `blk.64.nextn.shared_head_norm.weight` | F32 | 1 | 5120 | 64 |

Three block classes only (exact type+shape signatures, no per-layer variation
inside a class):
- **48 GDN blocks**: 0,1,2,4,5,6,8,9,10,… (every `il` with `il % 4 != 3`, `il < 64`)
- **16 full-attention blocks**: 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63
- **1 MTP block**: 64 — full-attention tensor set **plus** the four `nextn.*` tensors.

`qwen35.full_attention_interval = 4` is consistent with the observed `il % 4 == 3`
placement. Block 64 (the MTP/nextn draft layer) also carries a full-attention set.

**Fused expert stacks: NONE.** Zero tensors matching `*_exps*`, `*shexp*`,
`ffn_gate_inp`. Confirms the brief: this model has no MoE.

**`attn_q` ne[1] = 12288 = 2 × (24 heads × 256 head_dim) — it is gate-fused.** FACT
from the fork: `src/llama-load-tensors.cpp:4893`
`if (model.arch == LLM_ARCH_QWEN3NEXT || … || model.arch == LLM_ARCH_QWEN35) { for (auto & s : split_kq) s /= 2*gqa_ratio; }`
— i.e. the loader halves the Q split for qwen35 because Q carries a concatenated
gate. **A vLLM column shard of `attn_q` must split each 6144 half separately**, not
slice 12288 linearly.

Likewise `attn_qkv` ne[1] = 10240 = q 2048 + k 2048 + v 6144 (matches
`qwen35.ssm.inner_size = 6144`, `ssm.group_count = 16`, `ssm.state_size = 128`:
16 × 128 = 2048 for q and k). Shard per component.

## 3. Backbone KVs (FACT — verbatim)

```
pxa.pxq.backbone_rev = 2
pxa.pxq.backbone_map = attn_q,attn_qkv,attn_output,attn_gate_ch,shexp,ffn_dense=tier+1;attn_k,attn_v=q8_0;attn_gate_head=f16;token_embd=q6_k;output=q8_0
pxa.pxq6.version    = 1
pxa.pxq6.tier       = core
pxa.pxq6.book       = [-0.98779296875, -0.7353515625, -0.55859375, -0.419677734375,
                       -0.301025390625, -0.1944580078125, -0.09552001953125, 0.0,
                        0.084716796875, 0.1712646484375, 0.261962890625, 0.360595703125,
                        0.47119140625, 0.6005859375, 0.765625, 1.0]
pxa.pxq6.sub        = [0.2147216796875, 0.303466796875, 0.362060546875, 0.408935546875,
                        0.449951171875, 0.4873046875, 0.52294921875, 0.55859375,
                        0.59423828125, 0.6318359375, 0.671875, 0.71630859375,
                        0.7666015625, 0.82470703125, 0.89599609375, 0.98779296875]
```

**The 16-entry PX16 codebook and the 16-entry SUB16 sub-scale table are IN THE FILE.**
The vLLM dequant path must read `pxa.pxq6.book` / `pxa.pxq6.sub` from the GGUF
rather than hard-coding constants — the KV names carry `pxq6` even though the
tensor type is 252/PXQ4 (consistent with the brief's "id-252 kernels live in
pxq6.cuh" file-name trap).

Other load-relevant KVs: `general.architecture = qwen35`, `qwen35.block_count = 65`,
`embedding_length 5120`, `feed_forward_length 17408`, `head_count 24`,
`head_count_kv 4`, `key_length 256`, `value_length 256`, `context_length 262144`,
`nextn_predict_layers 1`, `rope.freq_base 1e7`, `rope.dimension_count 64`,
`rope.dimension_sections [11,11,10,0]` (mrope), `rms_eps 1e-6`,
`ssm.{conv_kernel 4, state_size 128, group_count 16, time_step_rank 48, inner_size 6144}`,
vocab 248320, `bos 248044 / eos 248046 / pad 248055`, `tokenizer.ggml.pre = qwen35`.
imatrix-quantized: `quantize.imatrix.dataset = unsloth_calibration_Qwen3.8-27B.txt`,
45 chunks, 496 entries.

## 4. Representative shapes and byte counts (FACT)

| tensor | type | ne (K x rows) | bytes | file offset |
|---|---|---|---:|---:|
| `output.weight` | Q8_0 | 5120 x 248320 | 1,350,860,800 | 0 |
| `token_embd.weight` | Q6_K | 5120 x 248320 | 1,042,944,000 | 1,350,881,280 |
| `blk.0.attn_gate.weight` | PXQ4 | 5120 x 6144 | 16,723,968 | 2,393,825,280 |
| `blk.0.attn_qkv.weight` | PXQ4 | 5120 x 10240 | 27,873,280 | — |
| `blk.0.ffn_down.weight` | PXQ4 | 17408 x 5120 | 47,360,000 | 2,438,443,008 |
| `blk.0.ffn_gate.weight` | PXQ4 | 5120 x 17408 | 47,384,576 | 2,485,803,008 |
| `blk.0.ffn_up.weight` | PXQ4 | 5120 x 17408 | 47,384,576 | 2,533,187,584 |
| `blk.3.attn_q.weight` | PXQ4 | 5120 x 12288 | 33,447,936 | — |
| `blk.0.ssm_out.weight` | MXFP4 | 6144 x 5120 | 16,711,680 | — |
| fused expert tensor | **does not exist** | — | — | — |

**The panel layout is confirmed byte-exactly.** Predicted PXQ4 size
`= (rows/64) * (128 + (K/32) * 1088)` reproduces the on-disk size of **all six**
distinct PXQ4 shapes to the byte (e.g. ffn_down: 80 panels × (128 + 544×1088) =
80 × 592,000 = 47,360,000 ✓; ffn_gate: 272 × (128 + 160×1088) = 272 × 174,208 =
47,384,576 ✓). There is **no inter-tensor padding** in the data section. This is an
independent confirmation of the 128 B fp16 anchor header + 1088 B/32-col slab
geometry from `ggml/src/pxq-cpu.h:1-17` and `ggml/src/ggml-cuda/pxa/pxq6.cuh:8-18`.
MXFP4 checks out too: 31,457,280 elems / 32 × 17 = 16,711,680 ✓.

## 5. Geometry gate on every PXQ4 tensor (FACT)

**Zero failures.** All 325 PXQ4 tensors satisfy `rows % 64 == 0 && K % 32 == 0`.

Six distinct PXQ4 shapes:

| ne (K x rows) | K | rows | n | rows%64 | K%32 | bytes each |
|---|---:|---:|---:|---:|---:|---:|
| 5120 x 6144 | 5120 | 6144 | 48 | 0 | 0 | 16,723,968 |
| 5120 x 10240 | 5120 | 10240 | 48 | 0 | 0 | 27,873,280 |
| 5120 x 12288 | 5120 | 12288 | 17 | 0 | 0 | 33,447,936 |
| 5120 x 17408 | 5120 | 17408 | 130 | 0 | 0 | 47,384,576 |
| 6144 x 5120 | 6144 | 5120 | 17 | 0 | 0 | 16,721,920 |
| 17408 x 5120 | 17408 | 5120 | 65 | 0 | 0 | 47,360,000 |

The tensors that *would* have failed the gate were already demoted by the
quantizer: `ssm_alpha`/`ssm_beta` are 5120 x **48** (48 % 64 = 48 ≠ 0) → q8_0, and
`attn_k`/`attn_v` are 5120 x 1024 (gate-clean, but pinned to q8_0 by the rev-2 table).

### TP shard check on the real shapes (FACT, arithmetic on the measured ne)

Every PXQ4 shape shards on the natural axis at both TP=2 and TP=4 with no
re-quantization:

| tensor | axis | TP2 shard | ok | TP4 shard | ok |
|---|---|---:|:--:|---:|:--:|
| attn_qkv 5120x10240 | col (rows) | q1024/k1024/v3072 | ✓ all %64=0 | q512/k512/v1536 | ✓ |
| attn_gate 5120x6144 | col | 3072 | ✓ | 1536 | ✓ |
| attn_q 5120x12288 (2×6144) | col, **per half** | 3072/half | ✓ | 1536/half | ✓ |
| ffn_gate/up 5120x17408 | col | 8704 | ✓ | 4352 (=68×64) | ✓ |
| attn_output 6144x5120 | row (K) | 3072 | ✓ %32=0 | 1536 | ✓ |
| ffn_down 17408x5120 | row (K) | 8704 | ✓ | 4352 | ✓ |

MXFP4 `ssm_out` 6144x5120 row-shards to K=3072/1536, both %32=0 (MXFP4 block=32) ✓.

## What this means for the vLLM backend

1. **Five types, not one.** `create_weights` / loader must dispatch PXQ4(252),
   Q8_0(8), Q6_K(14), **MXFP4(39)**, F32(0). Any design assuming uniform PXQ4
   fails at load on 181 of 866 tensors.
2. **MXFP4 is the newly-discovered work item.** 48 `ssm_out` GEMMs (6144→5120,
   one per GDN layer = 74% of layers) are MXFP4. INFERENCE: the rev-2 backbone
   table simply does not claim the `ssm` class (see `docs/lab/LEVERS.md:430` —
   `ssm_*` is only reachable via the `PXA_PXQ_NATIVE=ssm` research override, and
   `docs/lab/LEVERS.md:430` states an unclaimed/failing tensor "is demoted to MXFP4 by the
   caller"), so the quantizer's default fallback took it. **Mitigation is cheap:**
   `ssm_out` in vLLM is a RowParallelLinear whose weight can be dequantized offline
   to fp16 (0.75 GiB → 1.5 GiB total, 0.375 GiB/GPU at TP=4) or re-quantized —
   it does **not** require porting an MXFP4 kernel. Decide this in the converter,
   not in the kernel.
3. **Q8_0 is 10.3% of the file and unavoidable**: `output.weight` (1.26 GiB, the
   single largest Q8_0), `attn_k`/`attn_v` (34 tensors), `ssm_alpha`/`ssm_beta`
   (96 tensors), `nextn.eh_proj`. `token_embd` is Q6_K (0.97 GiB). The two vocab
   tensors alone are 2.24 GiB = 15.3% of the file and are **not** PXQ4.
4. **The codebook lives in the GGUF**, not in code — read `pxa.pxq6.book` /
   `pxa.pxq6.sub` at convert time.
5. **`attn_q` and `attn_qkv` are fused-multi-output.** Column shards must be taken
   per logical component (q|gate for attn_q; q|k|v for attn_qkv), which is exactly
   what vLLM's `QKVParallelLinear` / `MergedColumnParallelLinear` already express —
   and `qwen3_5.py:214-240` already wires the GDN path through
   `MergedColumnParallelLinear` (per brief; not re-verified here).
6. **Panel arithmetic is exact and there is no padding**, so a converter can compute
   every panel byte range in closed form: `panel_bytes(K) = 128 + (K/32)*1088`,
   `tensor_bytes = (rows/64) * panel_bytes(K)`.

### PROJECTION (labelled, no runs performed)
Per-GPU weight bytes at TP=4 if the file is ported as-is with `ssm_out` dequantized
to fp16 and `token_embd`/`output` replicated (not sharded): PXQ4 11.39/4 = 2.85 GiB
+ Q8_0-shardable ~0.09 GiB + ssm_out fp16 1.5/4 = 0.375 GiB + replicated vocab
2.24 GiB ≈ **5.6 GiB/GPU**, which is *worse* than the AWQ twin's 4.64 GiB. Shard
the vocab (`VocabParallelEmbedding` + `ParallelLMHead`, which vLLM does by default)
and it drops to ≈ **3.9 GiB/GPU**. ASSUMPTION: no activation/KV accounted, vocab
sharded row-wise at 248320/4. The brief's 3.66 GiB/GPU figure assumes uniform PXQ4
and is therefore ~7% optimistic; the 110–120 tok/s projection should be re-derived
from 3.9 GiB/GPU once the `ssm_out` decision is made.
