# Should the backbone go all-PXQ4?

Short answer: **no — the defensible change makes the file 491,520 B bigger.** The premise is
right in direction and wrong in magnitude, and wrong specifically on the tensors that carry
almost all of the saving.

## The ceiling, measured

A literal all-PXQ4 file saves **1,163,931,648 B = 7.404%** of the 15,719,771,584 B artifact
(14.6402 -> 13.5562 GiB; 4.5998 -> 4.2590 bpw). That is the entire prize. It decomposes badly:

| class | saving | share | verdict |
|---|---|---|---|
| `output.weight` (lm_head) | 674,933,760 B | 58.0% | possible; writes logits over a 248,320 vocab |
| `token_embd.weight` | 367,016,960 B | 31.5% | **engine-impossible — segfaults** |
| `attn_k` + `attn_v` (34) | 94,629,888 B | 8.1% | riskiest item on the board |
| `nextn.eh_proj` | 27,842,560 B | 2.4% | not worth a run |
| `ssm_out` (48, MXFP4) | **-491,520 B** | -0.04% | the only change with real evidence, and it COSTS bytes |

89.5% of the prize is two tensors. Remove those and everything else is 0.776% of the file.
77.81% of the artifact is already PXQ4 at 4.2526 bpw — there is no fat left in the
mixed-type table.

## `token_embd` -> PXQ4 crashes on the first token

PXQ slab types deliberately set `to_float` / `from_float` / `vec_dot` to NULL: a `to_float`
receives a single-row pointer, but a PXQ row's bytes are scattered across its panel's slabs.
`token_embd` is always a `GGML_OP_GET_ROWS` node. The CUDA backend's GET_ROWS switch lists
F16/F32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and returns false otherwise, so the node falls to the CPU
backend, which accepts it unconditionally and calls the NULL `to_float`.

**The quantizer will happily produce this file.** It passes the geometry gate, passes the
row-interleave guard (which contains zero PXQ entries), writes cleanly, loads cleanly, prints
a normal type histogram — and segfaults on the first token. This is a missing guard, not just
a bad idea.

## Physically impossible: the geometry gate

456 tensors, 35,753,984 B = 0.23% of the file, fail `rows % 64 == 0 && K % 32 == 0`:
96 x `ssm_alpha`/`ssm_beta` at ne=(5120,48) — 48 rows below the 64-row panel; 48 x
`ssm_conv1d` at ne=(4,10240) — K=4 cannot fill a 32-column block; 312 x 1-D f32 norms and
biases at rows=1. Identical across pxq1..pxq6 (all share blck_size 32, row_meta 2), so no
tier of the ladder reaches them. Only a format change would — a 16- or 48-row panel variant,
or row padding. This is why even a literal all-PXQ4 file floors at 4.259 bpw, not 4.254.

## The real win is the converter, not the model

These are different artifacts and must not be netted against each other.

**(A) Sidecar upcast** — the conversion decodes 547 tensors to fp16: 10,506,779,648 B out of
3,945,039,872 B of source, a **2.663x inflation**. It is policy-driven, not type-driven: 16
`attn_q` tensors are *already PXQ4 on disk* and are still decoded to fp16 (2,013,265,920 B,
19.2% of the whole upcast) purely because their fused `qkv_proj` sibling is q8_0. Separately,
the q8_0 head we deliberately paid for is inflated to fp16 — 1,191,936,000 B and 0.2775
GiB/GPU/token discarded for zero quality gain.

**(B) The GGUF** — total available -1,163,931,648 B; available *and* not reckless today
**+491,520 B**.

`ssm_out` is 5.1% of the model on disk but 28.7% of the sidecar upcast. `lm_head` and
`token_embd` are 89.5% of the disk saving but are unreachable in the sidecar for loader
reasons regardless of on-disk type. **Trading the model's measured quality pins to fix a
converter problem is paying in the wrong currency.**

## Corrections to earlier claims

- The `attn_k`/`attn_v` pin is **not vestigial**. The governing upstream rule for a non-MoE
  model is the `n_gqa() >= 4` catch-all, not the `n_expert >= 4` MoE heuristic, and this
  model's n_gqa is 24/4 = **6** — so it fires. Q4_K_M itself puts `attn_v` at Q6_K, so
  "parity with Q4_K_M" was never a 4-bit claim. The artifact's k/v payloads are md5-identical
  to the Q8_0 source, so 4 bits would be their first and only lossy hop. Only 17 of 65 blocks
  carry attention with 4 KV heads serving 24 query heads over a 262,144-token window — the
  model's sole exact-recall path, with no head redundancy to dilute error.
- pxq6 at 5.253 bpw is **below** q6_K's 6.5625, not above. The incumbent for `output.weight`
  is q8_0 at 8.5 bpw, so even the conservative fallback is a 38% bit cut.
- The FFN, not the mixed-type table, is where size actually lives: 195 tensors,
  9,238,394,880 B, **58.77% of the file**. At pxq3 that is -2,172,518,400 B -> 11.53 GiB and
  3.623 bpw, about **1.87x** the entire all-PXQ4 ceiling. Unmeasured at any rung below 4
  bits, and it would break the sidecar outright — no pxq3 kernel, Parameter class or config
  branch exists there, so 59% of the model would fall through to fp16.

## The finding that outranks the question

**The shipped artifact was quantized with no importance matrix.** Its `quantize.imatrix.*`
KVs are byte-identical to Unsloth's `Qwen3.8-27B-Q8_0.gguf` (chunks_count=45,
entries_count=496) and name a path that does not exist here; the recipe passes no
`--imatrix`. They were inherited. Every PXQ codec consumes the imatrix inside its anchor pick
and sub-scale argmin — with none, `w = 1` and the weighting is inert, across 77.81% of the
file. Our own guidance calls for >= 2000 chunks.

Costs zero bytes, reaches more of the model than every type decision combined. Note the
source is itself an already-quantized Q8_0, so we requantize from a lossy source; a bf16
original would be a second, independent win.

Chunk **length** matters as much as count: this model has recurrent linear-attention layers,
so at short chunk length the recurrent state never reaches its deployment regime and
importance for `ssm_out` and the qkv projections is fitted under the wrong state
distribution — green on every short benchmark, degraded at long context.

## Order of work

1. Fix the provenance writer. `pxa.pxq.backbone_map` is emitted as a hardcoded literal and is
   already false on the shipped file (it advertises `attn_gate_head=f16` on an artifact with
   zero f16 tensors). Until fixed, experimental arms are unattributable.
2. Give `errbudget` eyes. It skips interleaved / NULL-`to_float` types — i.e. every PXQ type,
   i.e. exactly what every change here writes. A panel-aware CPU dequant already exists
   (`pxa_pxq_dequant_row` in `pxq-cpu.c`) and is simply not wired in. It exists because the
   flat-MXFP4 backbone shipped two corrupt artifacts that looked normal on disk.
3. Rebuild with a real imatrix.
4. Ship p2a (done — see ENCODER_AND_P2A.md).
5. Decide k/v by measurement, evaluating the ladder pxq6 -> pxq4hq -> pxq4 and taking the
   highest tier that holds retrieval at >= 128k. A short-prompt perplexity A/B will look
   clean and is the trap: K/V error is written into the cache once and re-read by every
   subsequent token, so it compounds with context instead of averaging out.
6. Add a guard so the quantizer refuses to put a PXQ type on a GET_ROWS tensor.
