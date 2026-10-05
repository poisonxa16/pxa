# Converting to PXQU (universal mixed-tier) quants

`PXQ_UNIVERSAL` (a.k.a. PXQU) is a per-tensor mixed quant: instead of one bit-width for the whole model, a
**tier map** assigns a quant type to each expert tensor by name. This lets a large MoE land in a target VRAM
budget by spending bits where they matter (high-importance experts) and squeezing the rest (down to the 1-bit
`pxq1` tier). The result is a model that fits fully GPU-resident on a card it otherwise couldn't.

## The flag

```
llama-quantize --allow-requantize \
  --pxq-universal <map>.tiers \
  <source-q8>.gguf  <out>.gguf  PXQ_UNIVERSAL  <threads>
```

- `--pxq-universal <map>.tiers` — path to the tier map (format below).
- **No `--imatrix`.** PXQU is built from the PXQ codecs, and those ignore an importance matrix
  by default (since 2026-08-24; consuming it measured worse than not consuming it — see
  [`QUANTIZING.md`](QUANTIZING.md)). Passing one prints `PXQ tiers: imatrix IGNORED` and stamps
  `quantize.imatrix.ignored_by` into the output. Where an imatrix *does* help a PXQU build is
  upstream of the quantizer: it is evidence for **you** when you write the tier map below.
- Source should be a near-lossless **Q8_0** gguf.

> ⚠ **Do not pass `--override-kv <arch>.expert_used_count=int:N` at quantize time.** Earlier
> revisions of this page suggested it, to pin routing to the model's real top-k. It bakes the key
> into the output GGUF as **INT32**, while the loader reads that key as **UINT32** — so the file
> quantizes without complaint and then fails at load with
> `key <arch>.expert_used_count has wrong type i32 but expected type u32`.
>
> Every normally-converted source already carries the correct value and type, so **no override is
> needed**. If you genuinely must set it, spell the type `uint:`. The failure is invisible until
> load because a *runtime* `--override-kv` goes through a different code path that never consults
> GGUF types — only a baked one is type-checked.

Files containing the `pxq1` (1-bit) tier require a build with the PXQ1 codec (this release); other tiers
(`pxq2/pxq3/pxq4/pxq6`) run on any current build.

## The tier map format

A tier map is a text file of `#`-commented lines, one `regex=type` rule per expert tensor:

```
# <budget> tier map — one rule per expert tensor
^blk\.0\.ffn_gate_exps\.weight$=pxq2
^blk\.0\.ffn_up_exps\.weight$=pxq2
^blk\.0\.ffn_down_exps\.weight$=pxq3
...
```

- The regex is matched against the GGUF tensor name; the type is one of
  `pxq1` / `pxq2` / `pxq3` / `pxq4` / `pxq6`.
- Blank lines and `#` comments are ignored. Tensors with no matching rule fall through to the
  backbone recipe for the requested ftype.
- A bare filename is resolved relative to `$PXA_PXQU_DIR` (default `pxa-bench/pxq-universal/`);
  an absolute or relative path is used as-is.

Write one for your own tensor names and VRAM budget: total the per-tensor byte cost at each tier
(bpw × elements), then spend the budget on the tensors your imatrix says matter most — that is a
decision *you* make in the map, not something the quantizer reads out of the matrix. As reference
points, these are the budget/composition splits behind our published 122B-A5B (48 layers × 256
experts) builds:

| Budget | Composition (experts) | ~Resident |
|---|---|---|
| 24 GB | 126× pxq1 · 18× pxq2 | ~23.5 GiB |
| 32 GB | 61× pxq1 · 57× pxq2 · 26× pxq3 | ~31.7 GiB |
| 48 GB | 61× pxq2 · 57× pxq3 · 26× pxq6 | ~45.8 GiB |

The tighter the budget, the more experts drop to `pxq1`, and the more the aggressive tiers benefit
from a no-think serving posture.

> A mostly-1-bit map is the aggressive edge. Always live-validate after quantizing — coherence is
> the gate.

## Building for 30-series / 40-series (Ampere / Ada)

The release binary already targets `sm_86`/`sm_89`. To build from source for these cards:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -DCMAKE_BUILD_TYPE=Release
```

For a mixed fleet, list every arch you run: `-DCMAKE_CUDA_ARCHITECTURES="60;61;70;86;89"` (Pascal → Ada).
The per-arch performance levers auto-select at runtime; no environment variable is needed.
