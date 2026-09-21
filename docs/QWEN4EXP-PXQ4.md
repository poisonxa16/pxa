# Quantizing qwen4exp (Qwen3.8 Flash Next) to PXQ4

This architecture has one property that breaks the normal GGUF workflow, and it
costs hours to discover the hard way. Read the first section before you start a
run.

## The one rule: `per_layer_token_embd` must already be block-quantized

`qwen4exp` carries a per-layer token embedding table (the "PLE"):

```
per_layer_token_embd.weight   160 x 320,001,536   = 51.2B of the model's ~180B parameters
```

It is a **row-gather (`GET_ROWS`) table**: it is read one row at a time, indexed
by token id. `llama-quantize` therefore **refuses to touch it** and copies it
through unchanged. Three independent reasons, all load-bearing:

* PXQ and MXFP4 are **panel codecs** — a row is spread across a panel with shared
  state. Such a table quantizes and loads cleanly, then **gathers nonsense**.
* The usual fallback is not available: `Q4_K` needs `ne0 % 256 == 0`, and this
  tensor has `ne0 = 160`. Only block-32 codecs are legal here.
* The dequantize path sizes one f32 buffer to the whole tensor. At 51.2B
  parameters that is ~205 GB of RAM; it OOMs long before it quantizes.

This is a correctness gate. It is **not** overridable by `--custom-q`.

**The consequence:** whatever width the PLE has in your input file is the width
it has in your output. If the input stores it at `bf16`, ~95 GiB of `bf16`
lands in the output, PXQ-family bytes fall to ~38%, and the run fails the 50%
composition floor. **The fix belongs to the input file, not to the quantize
run.**

Since the precondition check was added, this fails in seconds with a message
naming the tensor. Older builds ran for hours first and then deleted their own
output.

## Step 1 — convert with a block-32 PLE

`--ple-type q8_0` is the default. Do not change it if the file is destined for
PXQ.

```bash
python3 convert_qwen4exp.py /path/to/Qwen3.8-Flash-Next \
    --outfile /models/qwen3.8-flash-next-bf16.gguf \
    --ple-type q8_0
```

The PLE phase is the slow part (128 shards, ~95 GiB read). It streams one shard
at a time on purpose — concatenating them peaks near 300 GB of RSS and has taken
a host down. To parallelise:

```bash
PXA_CONVERT_WORKERS=12 python3 convert_qwen4exp.py ... --ple-type q8_0
```

Output is byte-identical at any worker count. Measured on a 21-disk array: 12
workers took the PLE phase from 47 s/shard to 29.7 s/shard (1.58x). It does not
scale past that — the source reads are the wall, not the CPU. 48 workers
deadlocked the md stripe cache and produced zero throughput. Stage the source on
NVMe if you need it materially faster.

`--ple-type bf16` exists for non-PXQ uses. It now warns at conversion time that
the result cannot be PXQ-quantized.

## Step 2 — quantize to PXQ4

```bash
./build/bin/llama-quantize \
    /models/qwen3.8-flash-next-bf16.gguf \
    /models/Qwen3.8-Flash-Next-PXQ4.gguf \
    PXQ4 "$(nproc)"
```

### Starting from an existing Q8_0 GGUF instead

A third-party Q8_0 GGUF works **if** its PLE is already `Q8_0` — check before you
start (see Verifying below). You must add `--allow-requantize`, because some
backbone tensors are converted `q8_0 -> f16` and requantizing is refused by
default:

```bash
./build/bin/llama-quantize --allow-requantize \
    /models/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
    /models/Qwen3.8-Flash-Next-PXQ4.gguf \
    PXQ4 "$(nproc)"
```

Pass the **first** shard of a split GGUF; the rest are found automatically.

If the engine was built with shared libraries and you run the binary directly
from the build tree, point the loader at them:

```bash
export LD_LIBRARY_PATH=/path/to/PXA/build/src:/path/to/PXA/build/ggml/src
```

Do not pass an imatrix. The PXQ tiers ignore one by default (since 2026-08-24 —
consuming it measured *worse* than not consuming it on PXQ4), so `--imatrix` here
buys nothing: the quantizer prints `PXQ tiers: imatrix IGNORED` and records
`quantize.imatrix.ignored_by` in the file rather than claiming a consumption that
did not happen. See [`QUANTIZING.md`](QUANTIZING.md) for the detail.

## Verifying before you spend hours

Check the PLE's stored type in any GGUF. `Q8_0` at ~51880 MiB is correct;
`BF16` at ~97656 MiB will fail:

```bash
python3 - <<'PY'
import sys, glob
sys.path.insert(0, "gguf-py")
from gguf import GGUFReader
for f in sorted(glob.glob("/models/Qwen3.8-Flash-Next-Q8_0-*.gguf")):
    try:
        r = GGUFReader(f)
    except Exception:
        continue
    for t in r.tensors:
        if "per_layer_token_embd" in t.name:
            print(f.split("/")[-1], t.name, t.tensor_type.name,
                  f"{t.n_bytes/2**20:.1f} MiB")
PY
```

## Reading a composition failure

Every run prints an output composition table. For a PXQ target the run fails if
PXQ-family bytes are under 50%, or if a uniform PXQ target emitted zero bytes of
the tier it names. Both mean the file's **name would misrepresent its contents**,
so every file the run wrote is removed.

A failure dominated by `bf16` is the PLE problem above — fix the input.

`--pxq-composition-override` (env `PXA_PXQ_COMPOSITION_OVERRIDE=1`) downgrades
the failure to a warning and keeps the file. It is the wrong tool for a bf16
PLE: it would ship a file that is ~60% `bf16` while its name claims PXQ4. Use it
only when you have already established that the composition is legitimate.

## Known-good reference numbers

For the ~180B Flash-Next at PXQ4:

| quantity | value |
|---|---|
| `per_layer_token_embd` at Q8_0 | 51880.1 MiB |
| `per_layer_token_embd` at bf16 (**fails**) | 97656.8 MiB |
| tensors quantized | 1224 |
| tier split (uncensored build) | 38 pxq2 / 106 pxq3 / 276 pxq4 |

## Serving

The PLE stays on the CPU — it is a gather table, and at ~51 GiB it will not fit
alongside the backbone on most cards:

```bash
llama-server -m Qwen3.8-Flash-Next-PXQ4.gguf \
    -ngl 99 -ot 'per_layer_token_embd\.weight=CPU' \
    -c 163840 -ub 1024 --jinja
```

Expect the gather to cost a host round-trip per layer per token. On multi-GPU
without NVLink, `-sm layer` additionally costs a host-bridge round-trip per
token — and it is still the split mode to use here.

`-sm graph` is not an alternative for this file. It splits the attention output
and the expert down projections along `K`, which a PXQ tensor cannot be cut on:
PXQ keeps one fp16 anchor per row in the 64-row panel header and that anchor
covers all of `K`, so a `K`-split would have to duplicate it. The engine refuses
`-sm graph` on a PXQ file at load, with an error naming the tensor. Graph mode is
independently guarded off for this architecture, which is a DeltaNet hybrid.
