# Exporting a PXQ model back to a stock format (`llama-pxq-export`)

The PXQ types — `pxq1` (248), `pxq4` (252), `pxq4hq` (253), `pxq2` (254), `pxq3` (255),
`pxq6` (256) — are 64-row **panel-interleaved** formats. A single weight row's bytes are
scattered across the slabs of its panel, so the ggml type traits carry no `to_float` and no
`vec_dot`, and `llama-quantize` refuses to read them:

```
cannot requantize from a PXQ slab type (PXQ1/PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6: CUDA-only slab
layout, no CPU codec) — requantize from the original F32/BF16/Q8_0 source
```

There is no *per-row* CPU codec, and there cannot be one — that part of the refusal is
structural. But it made a PXQ artifact a **terminal** format: if the source checkpoint was gone,
the weights were locked to this engine.

Two things fix that, and both are here:

1. **`llama-pxq-export`** decodes every PXQ tensor and writes an ordinary GGUF (F16 or F32).
2. **`llama-quantize` now accepts a PXQ file as a source directly**, behind an explicit consent
   flag, because the panel-aware CPU decoder in `ggml/src/pxq-cpu.c` covers all six tiers as of
   2026-09-01. Neither path needs a GPU.

### Why there is still no ggml `to_float` for the PXQ types

`to_float` in the ggml type traits is handed **one row pointer** and a count. A PXQ row's bytes
are spread across the slabs of its 64-row panel, and the panel base is not recoverable from a row
pointer, so any `to_float` registered for these types would decode garbage — and would do so
silently, in every generic ggml path that assumes per-row decodability (`llama_tensor_dequantize_internal`
walks rows exactly this way; so do `dup`, `get_rows` and the chunked `mul_mat`). The traits keep
`to_float` NULL on purpose.

The CPU decoder therefore takes the addressing the format actually has: **the tensor base plus a
global row index** (`pxa_pxq_dequant_row` / `pxa_pxq_dequant_2d`, `ggml/src/pxq-cpu.h`).
`llama-quantize` calls that instead of the row-walking helper, and `mul_mat` /
`mul_mat_id` already had a panel-dequant early return for the four tiers that were covered — PXQ1
and PXQ6 joined it with this change, which also retires the two `no CPU vec_dot` abort stubs from
the reachable paths (they stay wired as a backstop against a NULL call).

## Usage

```
llama-pxq-export in.gguf out.gguf [--type f16|f32] [--device N] [--verify] [--chunk-mib N]
```

| flag | meaning |
|---|---|
| `--type f16` (default) | landing type for PXQ tensors |
| `--type f32` | same decode, no rounding at all — use it when the export exists only to be re-quantized and disk allows |
| `--device N` | CUDA device to decode on (default 0) |
| `--cpu` | decode with the CPU panel dequant instead of CUDA — same values (0 ULP), much slower, no GPU needed |
| `--verify` | re-decode each PXQ tensor whole and require the streamed chunked result to be bit-identical, and cross-check against the independent CPU panel dequant where one exists |
| `--chunk-mib N` | decode chunk budget, default 256 |

## The command chain

Two routes. The **one-step** route needs no intermediate file:

```
llama-quantize --allow-requantize --i-know-this-is-double-lossy \
    model-pxq4.gguf model-Q4_K_M.gguf Q4_K_M 8
```

The **two-step** route is what you want when the F16 itself is the deliverable (handing the model
to stock llama.cpp, measuring against a reference, or feeding a tool that is not this one):

```
# 1. PXQ artifact -> F16 GGUF  (--cpu if you have no GPU)
llama-pxq-export model-pxq4.gguf model-f16.gguf --type f16 --device 0

# 2. F16 GGUF -> anything stock (stock llama-quantize, or this fork's)
llama-quantize --allow-requantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M 8

# 3. the result loads in stock llama.cpp / ik_llama.cpp
llama-cli -m model-Q4_K_M.gguf -p "..." -n 64
```

`--allow-requantize` is required in step 2 only because the F16 file may still carry
non-PXQ quantized tensors that were copied through (MXFP4, `Q*_K`, the block-32 codecs the
quantizer picks for row-gather tables). Nothing in step 2 touches a PXQ layout — by then there
is none left. `--i-know-this-is-double-lossy` is not needed in step 2 either: an F16 source is
not a lossy source.

### `--i-know-this-is-double-lossy`

`llama-quantize` refuses a **Q8_0 or PXQ source** without it. Requantizing an already-quantized
tensor runs two lossy passes: the second codec fits its grid (and reads any imatrix) against
weights that were already snapped to the first grid, so the result is strictly worse than
quantizing the original F32/BF16 once. Q8_0 is near-lossless and the damage is small; a PXQ tier
is 1.26–5.27 bpw and the damage is not. Both are legitimate when the original checkpoint is gone
— they are not legitimate *by accident*, which is what the flag is for.

> This changes the documented PXQU flow in `docs/PXQU-CONVERT.md`, which quantizes from a Q8_0
> source: add `--i-know-this-is-double-lossy` to those command lines.

## What the export does, tensor by tensor

* **PXQ tensors** are uploaded to the CUDA device in whole 64-row panel chunks, decoded with
  `ggml_get_to_fp16_cuda()` / `ggml_get_to_fp32_cuda()` — the exact functions the
  `dequant → cuBLAS` serving fallback calls — and written as F16/F32.
* **Every other tensor is copied byte for byte.** F32 norms, biases, 1-D tensors, MXFP4 and
  `Q*_K` backbone tensors, the routing tables, the token embedding: unchanged bytes, unchanged
  type.
* **3-D MoE expert tensors need no special case.** A PXQ tensor is `E × (ne1/64)` contiguous
  64-row panels — panels row-major, experts outermost — so `nrows = ne1·ne2·ne3` decodes the
  whole tensor, and any 64-row-aligned prefix of the panel run decodes independently. That is
  exactly what makes the chunking legal, and it is why a per-expert loop is unnecessary.
* **Row-gather / `GET_ROWS` tables are never PXQ.** `per_layer_token_embd` (the qwen4exp /
  gemma PLE n-gram-hash table, and anything with `ne1 ≥ 1e6`) is ruled out of every panel codec
  by the quantizer as a *correctness* gate, not a preference — a panel row is unreadable in
  isolation, so such a file would load cleanly and gather nonsense. The exporter re-applies the
  same test and **refuses the file** if it ever finds a PXQ row-gather table, rather than
  producing a plausible wrong answer.
* **Short rows are never PXQ either.** `ne0 % 32 != 0` (e.g. `ssm_conv1d` / `ple_conv1d`, which
  are `[4, 10240]`) and `ne1 % 64 != 0` fall out of PXQ eligibility in the quantizer; they
  arrive here as F32/F16/`Q*` and are copied.
* **The output head** (`output.weight`) is an ordinary 2-D weight: decoded if it landed on a
  PXQ tier, copied otherwise. No special case.

> **Aside, found while building this** — `llama-quantize --token-embedding-type pxq4` (or any
> PXQ tier) produces a broken model. `token_embd` is read by `GET_ROWS`, one row at a time, and
> a PXQ row is unreadable in isolation. The automatic row-gather guard catches
> `per_layer_token_embd` by name and anything with `ne1 >= 1e6` by size, but a 151936-row vocab
> slips through both — and an explicit `--token-embedding-type` is an override anyway. Measured:
> the file quantizes and loads cleanly, and then generates nothing at all. Export decodes such a
> file correctly (the F16 it produces generates normal text), but the PXQ artifact itself was
> never usable. Do not send an embedding table to a panel codec.

## Metadata

* Every KV is preserved except:
  * `general.file_type` → `1` (`MOSTLY_F16`) or `0` (`ALL_F32`). This KV is a one-word summary;
    the per-tensor types in the tensor index remain exact, exactly as in any mixed file
    `llama-quantize` writes.
  * `general.quantization_version` → `GGML_QNT_VERSION` (2).
  * **`pxa.pxq*` KVs are dropped.** They are not inert: `pxa.pxq2.version` / `pxa.pxq3.version`
    are read by the model loader, which fires a loud `PXQ TABLE MISMATCH` warning whenever the
    runtime's env-selected codebooks differ from the file's — and it would keep firing on a file
    with no PXQ tensor left to decode. `pxa.pxq{2,3,6}.book` / `.sub` are the codebooks,
    `pxa.pxq.backbone_rev` / `.backbone_map` / `.backbone_overrides` record which tensor class
    got which tier, and `pxa.pxqu.version` marks a UNIVERSAL mix. All of it describes a layout
    the exported file no longer contains.
  * A single `pxa.pxq_export.source_types` string is added, recording the PXQ tiers the file
    came from (e.g. `pxq2:96,pxq3:48,pxq4:112`), so the provenance is not simply lost.
* Split files are refused: merge with `llama-gguf-split --merge` first.

## Codebook provenance is a hard error, not a warning

The PXQ2/PXQ3 codebooks are **env-armed at decode time** (`PXA_PXQ2_V3`, `PXA_PXQ_CEIL_V2`) and
the quantizer stamps the version it used into the file. Decoding a v2/v3 file with v1 tables
gives numbers that are wrong and entirely plausible. The model loader only *warns* about a
mismatch, because a running server can be restarted. An export **bakes** the mistake into a new
artifact, so `llama-pxq-export` treats it as fatal and tells you which env var to set. Custom
book overrides (`PXA_PXQ6_BOOK`, `PXA_PXQ2_BOOK`, …) are honoured but warned about.

## The numerical story — read this before you use the output

**Step 1 adds no quantization error.** A PXQ weight is `anchor × sub × book[code]` — an fp16 row
anchor, an fp16-snapped sub-scale and an fp16-snapped codebook entry — and the decode evaluates
that product in fp32. Every distinct stored code maps to exactly one value; no information the
file holds is discarded and no new grid is fitted.

Two precise statements, because the difference matters if you are chasing bits:

* `--type f32` is **exact**: the file holds the fp32 dequant itself. Nothing is rounded anywhere
  in step 1.
* `--type f16` (the default) rounds that fp32 product once on store — the *same* single rounding
  the serving path's `dequant → cuBLAS` fallback performs before every GEMM. So the exported F16
  holds exactly the numbers this engine multiplies on that path. The product of three fp16
  quantities is not in general an fp16 number, so this is one rounding, not zero; it is just the
  rounding the engine was already doing.

If the export exists only to be re-quantized, `--type f32` is the one that makes the two-step
chain numerically identical to a hypothetical single-step "PXQ straight to Q4_K_M" — that path
would dequantize to fp32 and quantize from fp32, which is precisely what
`--type f32` + `llama-quantize` does. `--type f16` costs one fp16 rounding ahead of the
K-quant fit; it halves the intermediate file, which on a 200 GB model is the deciding factor.

**Step 2 is a second, independent quantization of an already-quantized tensor.** Going
`BF16 → PXQ4 → F16 → Q4_K_M` is strictly worse than `BF16 → Q4_K_M`: the Q4_K_M grid is fitted to
weights that have already been snapped to the PXQ grid, so the two error terms compound and the
K-quant importance heuristics (and any imatrix) are looking at the wrong distribution.

> **If you still have the original BF16/F32 checkpoint, quantize from that.** Use this tool when
> you do not — to unlock an artifact whose source is gone, to hand a model to stock llama.cpp,
> or to get a reference F16 to measure against.

An imatrix computed on the exported F16 model is also measuring the PXQ-rounded weights, not the
original ones. It is still useful (it is the right importance for *this* model), but it is not
the imatrix you would have gotten from the source.

## Cost and limits

* **Coverage:** all six PXQ tiers — `pxq1`, `pxq2`, `pxq3`, `pxq4`, `pxq4hq`, `pxq6`. There is a
  CUDA dequant for every one; the export uses it directly.
* **A GPU is optional.** `--cpu` (and the direct `llama-quantize` route) use the panel dequant
  in `ggml/src/pxq-cpu.c`, which covers all six tiers and is validated to 0 ULP against the CUDA
  kernels. The CUDA path is the default because it is far faster; the CPU path is a scalar loop
  over every element.
* **Memory:** nothing is fully resident. Peak host and device footprint is one chunk in plus one
  chunk out (`--chunk-mib`, default 256 → well under 1 GiB of VRAM), regardless of model size.
  `--verify` is the exception: it holds one whole tensor, so use it on small models.
* **Size:** the output is much larger than the input — F16 is 16 bpw against PXQ's 1.26–5.27.
  Budget disk for it before starting a 200 GB export.
* **Compute parity is not bit-parity.** The exported F16 model does *not* have to produce
  token-identical output to the PXQ model at temperature 0: the weights are identical, but the
  PXQ fused kernels and cuBLAS F16 accumulate in a different order. Greedy continuations agree
  for a while and then drift, which is the ordinary consequence of a different summation order,
  not a defect in the export. Setting `PXA_PXQ6=0` on the PXQ side forces the
  `dequant → cuBLAS` fallback and brings the two much closer.

## Verification

`tests/test-pxq-export.sh` quantizes a small model to each PXQ tier, exports it, and checks that

1. the CUDA and CPU decoders agree to **0 ULP** on every tier,
2. every non-PXQ tensor is byte-identical between the PXQ file and the export,
3. the streamed chunked decode is bit-identical to a whole-tensor decode,
4. `--cpu` and the GPU path produce byte-identical files,
5. `llama-quantize` refuses a PXQ source without the consent flag and requantizes with it,
6. the exported file loads and produces a greedy continuation that tracks the PXQ model's.

## Follow-up: a PXQ CPU `vec_dot` / AVX2 matmul

The CPU path today is *dequantize the whole panel, then do an fp32 dot*
(`pxa_pxq_mul_mat_cpu`). Correct, and slow enough that it is a compatibility fallback rather than
a serving option. A real CPU kernel is a well-shaped piece of work and worth writing down:

* The 4-bit tiers store two codes per byte, and the book is 16 fp32 entries. Snap the book to
  int8 once per tensor (it is already fp16-snapped and symmetric) and the decode becomes a
  **`pshufb` nibble lookup**: load 16 bytes of the code row, `vpand`/`vpsrlw` into two nibble
  halves, two `vpshufb` against the broadcast 16-entry int8 book, and you have 32 int8 weights
  per 16 bytes loaded — the same shape as the Q4_K / IQ4_NL kernels in `iqk_mul_mat.cpp`.
* Pair that with **Q8_0 activations** (`vec_dot_type = GGML_TYPE_Q8_0`) and the inner product is
  `vpmaddubsw` + `vpmaddwd` + `vpaddd`, with the per-16-element `eff = anchor × SUB16[s4]`
  folded in at the end of each block as a single fp32 multiply — the E16-row scale structure is
  what makes that legal, since `eff` is constant across each 16-element run.
* PXQ3's bit-plane packing and PXQ6's 5th-bit plane need one extra `vpor` of a shifted plane
  before the `pshufb`; PXQ2 needs a 2-bit expand; PXQ1 is a sign mask (`vpsignb`).
* The blocker is not the arithmetic, it is the addressing: `vec_dot` gets a row pointer, and a
  PXQ row is panel-interleaved. So this cannot be a `vec_dot` trait either — it belongs in
  `pxa_pxq_mul_mat_cpu` as a panel-tiled kernel (decode a 64-row panel's slab into a register
  tile, dot it against the Q8 activation block), which is also the shape that reuses the panel
  layout instead of fighting it.

### Measured (Qwen3-0.6B-Q8_0, GTX 1080 Ti / sm_61, all six tiers)

| tier | tensors cross-checked | worst CUDA-vs-CPU diff | non-PXQ tensors byte-identical | `--cpu` vs GPU export |
|---|---|---|---|---|
| pxq4 | 196 | 0 ULP | 114 / 114 | byte-identical |
| pxq4hq | 196 | 0 ULP | 114 / 114 | byte-identical |
| pxq2 | 196 | 0 ULP | 114 / 114 | byte-identical |
| pxq3 | 196 | 0 ULP | 114 / 114 | byte-identical |
| pxq6 | 196 | 0 ULP | 114 / 114 | byte-identical |
| pxq1 | 196 | 0 ULP | 114 / 114 | byte-identical |

Chunked (64 MiB) decode equalled the whole-tensor decode with zero byte differences on every
tensor of every tier. On PXQ6 the exported F16 model's greedy continuation was identical to the
PXQ model's for the whole 24-token generation, both against the fused kernels and against the
`PXA_PXQ6=0` dequant fallback. Both the `--cpu` export and the direct
`PXQ6 → Q4_K_M` requantize were re-run with `CUDA_VISIBLE_DEVICES=` (no visible device) and
produced byte-identical output.

## Building

```
cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=1
cmake --build build -j 12 --target llama-pxq-export llama-quantize llama-cli
```
