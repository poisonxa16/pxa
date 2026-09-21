# Quantizing your own models to PXQ4

Both engines in this repository run PXQ4, and both are fed from the same place: a GGUF.
The llama.cpp engine loads that GGUF directly; the vLLM backend takes one more step that
converts it into a vLLM-loadable safetensors directory.

```
   HF checkpoint
        |  convert_hf_to_gguf.py            (convert_qwen4exp.py for Flash-Next)
        v
   f16/bf16 GGUF  --llama-quantize-->  Q8_0 GGUF  --llama-quantize-->  PXQ4 GGUF
                                                                          |
                                    +-------------------------------------+
                                    |                                     |
                              llama.cpp engine                  python -m gguf_to_vllm.convert
                              (run it directly)                           |
                                                                          v
                                                              vLLM safetensors directory
```

Everything below needs only `pxq-quantize` (a separate download — see the release page) and
Python. No GPU is required to quantize. `pxq-quantize` takes the same command line as
`llama-quantize` and is the tool that writes the PXQ tiers; the `llama-quantize` built from
this repo still handles every stock type and will tell you to fetch `pxq-quantize` if you ask
it for a PXQ one. Where a command below says `llama-quantize` with a PXQ type, run
`pxq-quantize` instead.

---

## 1. HF checkpoint to GGUF

```bash
python3 convert_hf_to_gguf.py /path/to/hf-model --outfile model-f16.gguf --outtype f16
```

**Flash-Next / qwen4exp models use a dedicated converter** — they are not handled by
`convert_hf_to_gguf.py`:

```bash
python3 convert_qwen4exp.py /path/to/hf-model --outfile model-f16.gguf --ple-type q8_0
```

`--ple-type q8_0` is the default and must stay that way for anything destined for
PXQ. That architecture's `per_layer_token_embd` is a 51.2B-parameter row-gather
table which the quantizer copies through **unchanged** — so if it arrives at
`bf16`, ~95 GiB of `bf16` lands in the output and the run cannot meet the PXQ
composition floor. See **[QWEN4EXP-PXQ4.md](QWEN4EXP-PXQ4.md)** for the full
picture, verification snippet, and worked examples.

## 2. GGUF to PXQ4

Quantize from a near-lossless source. Q8_0 first, then PXQ4, gives noticeably better
results than going straight from f16, and it makes re-quantizing to other tiers cheap:

```bash
./llama-quantize model-f16.gguf model-q8.gguf   Q8_0  $(nproc)
./llama-quantize model-q8.gguf  model-pxq4.gguf PXQ4  $(nproc)
```

Quantizing **from an existing Q8_0 GGUF** needs `--allow-requantize`; without it the
run stops at the first tensor that has to be converted out of `q8_0`:

```bash
./llama-quantize --allow-requantize model-q8.gguf model-pxq4.gguf PXQ4 $(nproc)
```

For a split GGUF, pass the **first** shard — the rest are found automatically. There is
no `--imatrix` in that command on purpose: the PXQ tiers ignore an importance matrix
(see below).

That file runs on the llama.cpp engine as-is. Stop here if that is all you need.

### The PXQ tiers

| Type | bpw | What it is |
|---|---|---|
| `PXQ1` | 1.26 | 1-bit sign x E16-row scales. Experts only; the sub-2-bit stretch tier. |
| `PXQ2` | 2.27 | LM4 x E16-row scales. Experts. |
| `PXQ3` | 3.27 | LM8 bit-plane x E16-row scales. Experts. |
| **`PXQ4`** | **4.27** | **PX16 book + E16-row scales. The default choice.** |
| `PXQ4-HQ` | 4.52 | PXQ4 with bs8 sub-scales. |
| `PXQ6` | 5.27 | LM32 5-bit x E16-row scales. The quality tier. |

`PXQ_UNIVERSAL` (PXQU) is a per-tensor mixed-tier mode driven by a tier map you write
yourself; see [`PXQU-CONVERT.md`](PXQU-CONVERT.md) for the flag and the map format.

### Tier profiles: what a level name means (`--pxq-policy`)

A level name has never described the whole file. On a model with routed experts it names the
**expert** tier, and the always-resident backbone — attention projections, the shared expert,
the DeltaNet output projection — is quantized a notch or two above it, because those bytes are
read on *every* token while a given expert is read on about one token in fifty. That mix is why
our MoE files hold up at 2 bits.

On a **dense** model there were no experts to name, so the level named everything: `PXQ2` on
Qwen3.8-27B put the attention projections at 2.27 bpw along with the FFN. That was never a
measured choice — it was what fell out of keeping the file honestly named — and `--pxq-policy`
is where I fixed it. The rule is the one the MoE side already used, stated once for both shapes:

> **the level names the tier of the bulk — routed experts on a MoE model, the FFN on a dense one
> — and the attention block is bought up from there.**

| profile | dense model | MoE model |
|---|---|---|
| `uniform` *(default for `PXQ4` and above)* | level for the whole GEMM backbone | the historical per-tier backbone table |
| `balanced` | FFN at the level; attention +1 notch; `attn_output` +2 | whole backbone at `pxq4` |
| `attn4` | FFN at the level; the whole attention block at `pxq4` | same as `balanced` |
| *(every profile)* | `ssm_out` at **`q8_0`** — see below | `ssm_out` at **`q8_0`** |

```bash
./llama-quantize --allow-requantize --pxq-policy balanced model-q8.gguf model-pxq2.gguf PXQ2 $(nproc)
./llama-quantize --allow-requantize --pxq-uniform          model-q8.gguf model-pxq2.gguf PXQ2 $(nproc)
```

What this costs on Qwen3.8-27B, exactly (attention is 21% of that model's parameters against the
FFN's 64%, so buying it up is cheap):

| level | `uniform` | `balanced` | `attn4` |
|---|---|---|---|
| PXQ2 | 8.93 GiB | 9.83 GiB (+10.0%) | 10.61 GiB (+18.8%) |
| PXQ3 | 11.79 GiB | 12.60 GiB (+7.0%) | *same as balanced* |

At level `pxq3` and above the two profiles converge — `balanced`'s +1/+2 both land on the cap —
so there is one file, not two.

**Nothing promotes past `pxq4`.** Two reasons, either sufficient: `pxq4hq` has no vLLM decoder
in this release, so a profile file containing one would run on one engine and not the other; and
`pxq4`/`pxq4hq` are the only types the fused MMVQ decode path admits, with `pxq4` the faster and
smaller of the two. `pxq4hq` and `pxq6` remain available as explicit choices (the `PXQ4-HQ` and
`PXQ6` levels, `--custom-q`, or a PXQU map) — they are just not somewhere a *default* sends you.

### The DeltaNet output projection is protected at `q8_0`

**`blk.N.ssm_out.weight` is quantized at `q8_0` by default on every PXQ level, whatever the
profile.** It is the one class in the file whose type you cannot read off the level name, and
this is why.

On a hybrid model — Qwen3.8-27B is 48 DeltaNet layers out of 65 — `ssm_out` is what carries the
recurrent block's whole output back into the residual stream. It sits where `attn_output` sits on
an attention layer, but it is reading a *recurrent state* rather than a softmax average, so there
is no averaging in front of it to absorb an error. Measured against the file's own `Q8_0` source
(wikitext-2, `-c 2048`, 10 chunks, one class changed and nothing else):

| `ssm_out` type | mean KLD vs source | same-top-p | file size |
|---|---|---|---|
| `pxq4` — what every level used to resolve to | **0.433** | 93.05% | — |
| `mxfp4` — the pre-2026-08-25 landing | 0.060 | 92.46% | −0.003% |
| `pxq4hq` | 0.049 | 93.23% | +0.30% |
| `pxq6` | 0.044 | 93.08% | +1.20% |
| **`q8_0` — the default** | **0.042** | **93.26%** | **+5.10%** |

Forty-eight tensors, about 5% of the file's bytes, and a **7x** mean-KL gap between the top and
bottom rows. Decode speed does not pay for it: on two P100s at `-c 8192`, np1 medians of 7, the
four files land at 18.29 / 18.10 / 17.83 / 17.81 t/s (`pxq4`, `mxfp4`, `q8_0`, `pxq6`) — a 2.7%
spread with no separation, so the type of this class does not move decode.

**Why `q8_0` and not the cheaper `pxq6`.** The same portability rule that caps the profiles at
`pxq4`: `pxq6` and `pxq4hq` have no vLLM decoder in this release. A file is routed to an engine
by its *level*, not by its contents, so a `PXQ4` file carrying 48 `pxq6` tensors would be handed
to an engine that cannot read them and fail to load with nothing in front of it to explain why.
`q8_0` loads on both, is already in every PXQ file we ship (`attn_k`/`attn_v` are pinned to it),
and measures the best of the five rows anyway. Portability costs 3.9% of file size here; it is
worth it.

**It is a floor, not a pin.** It lifts a landing that would otherwise sit on a low PXQ tier and
does nothing else, so all of these still win:

```bash
# an explicit type for the class — outranks the default, in either direction
./llama-quantize --allow-requantize --custom-q 'ssm_out\.weight=pxq6' src.gguf out.gguf PXQ4 $(nproc)
# same thing without a regex, and it accepts `off` to reproduce the pre-2026-09-10 allocation
PXA_PXQ_SSM_OUT=pxq4hq ./llama-quantize --allow-requantize src.gguf out.gguf PXQ4 $(nproc)
PXA_PXQ_SSM_OUT=off    ./llama-quantize --allow-requantize src.gguf out.gguf PXQ4 $(nproc)
```

A geometry failure's `q8_0`, the MTP companion block's `q8_0` and a deliberately flat-MXFP4
backbone (`PXA_PXQ_BACKBONE=legacy` or `=lite`) are all left exactly where they were — that last
one is the `mxfp4` row above, a separately measured operating point at 0.060, not the 0.433 this
floor exists for.

The resolved `pxa.pxq.backbone_map` KV says which one you got (`ssm_out=q8_0` for the default,
`ssm_out=custom:pxq6` if you overrode it), and the quantizer prints the table above at the point
where it decides, so a file's own log accounts for its bytes.

**The `PXQ2` base now defaults to `attn4`; every other level still defaults to `uniform`.** I do
not change a default recipe on reasoning alone, so this one waited for a measurement — and the
measurement was not close. Against a `PXQ4` anchor at perplexity 6.27, uniform `PXQ2` lands at
**30.01**: 4.8× the anchor, mean KL divergence 1.654, and it picks the anchor's top token on
**48.8%** of positions, which is a coin flip. The same base under `attn4` lands at **9.35** —
1.49×, KLD 0.453, 72.8% — for **10.1% more bytes**, and it decodes *faster* on both engines,
because the promoted attention tensors ride a quicker kernel than the 2-bit ones they replace.
A 4.8× default that nobody had measured is not the conservative choice; it is the unexamined one.

**`PXQ3` moved too, to `balanced`, and on a different kind of evidence.** Here perplexity does not
decide it: 6.6151 against uniform's 6.5884 is inside the ±0.086 noise of the measurement, so on
that number alone you would call it a tie. The distribution disagreement is not a tie — mean KL
divergence **0.0758 against 0.1056**, and it picks the anchor's top token on **88.833%** of
positions against **86.585%** — for **4.2% more bytes** and **+15.8% / +14%** decode on the two
engines. I am recording that the two measures point different ways, because a reader who checks
only perplexity will think this default was unmotivated.

**`PXQ4` and above did not move.** `PXQ4`'s attention is already at the cap, so every profile
produces the *same file* — there is nothing to choose. `PXQ4HQ` and `PXQ6` are simply unmeasured,
and I would rather say that than call them safe: unexamined is exactly what uniform `PXQ2` was
until somebody measured it.

`--pxq-policy uniform` (or `--pxq-uniform`) still reproduces the old allocation exactly, so every
file published before 2026-09-09 remains reproducible.

The first hardware arms have now run, on two V100s, dense Qwen3.8-27B, on both engines — see
[the promoted profiles in `VLLM.md`](VLLM.md#the-promoted-profiles-and-the-rule-that-decides-their-decode-speed).
The short version: `attn4` is 10.1% larger than `uniform` at PXQ2 and decodes **11.7% faster**
on vLLM and 10.3% faster on `llama-server`, because the fused decode path admits `pxq4` only
and `attn4` is what puts the attention block there; `balanced` at PXQ2 lands attention on
`pxq3`, off that path, and is slower than `uniform` despite being bigger. The fidelity half of
the question is closed by a per-token KL-divergence ladder against a PXQ4 anchor (wikitext-2,
30 chunks — full table in
[`VLLM.md`](VLLM.md#the-fidelity-question-closed-by-a-per-token-measurement)): under the
pre-registered rule (recommended over the uniform base tier iff mean KLD to the anchor is lower
**and** Same-top-p is not lower), `attn4` and `balanced` are both **recommended** at PXQ2 and
`balanced` is **recommended** at PXQ3. Because `attn4` wins its PXQ2 comparison, it becomes the
quantizer's default policy for `pxq2` bases — `pxq3` and above already converge `balanced` and
`attn4` to the same shape, so a `pxq3` base gets the same default. That is a one-line change in
`src/pxa-pxq-policy.h`; until it ships, `uniform` is what today's binaries still emit by
default.

Every PXQ file says which rule made it:

```
pxa.policy.rev      3
pxa.policy.name     balanced
pxa.policy.level    pxq2
pxa.policy.bulk     ffn
pxa.policy.summary  balanced: level pxq2 names the FFN tier; attention +1 notch, attn_output +2 (cap pxq4)
```

and the quantize log prints the summary line on every run, next to the resolved
`pxa.pxq.backbone_map` that says what each role actually landed on.

### Files made before rc3: the missing codebook KVs

Every PXQ tier is a codebook format — nothing can decode a `pxq3` tensor without
`pxa.pxq3.book` and `pxa.pxq3.sub`. Until 2026-09-09 the quantizer chose those keys from the
**requested** level rather than from the tiers it actually **emitted**, so any file whose
backbone tier differs from its level name is missing a codebook. A `PXQ2` MoE build emits a
`pxq4` backbone and stamped only `pxa.pxq2.*`; our own `PXA-Fusion2-35B-PXQ2` is an example.

The llama.cpp engine never noticed, because it decodes from its compiled-in tables — but the
vLLM converter reads the KVs, as it should, and cannot decode what is not described. Loading an
affected file now prints a `PXQ CODEBOOK KV MISSING` warning naming the tier and the key.

Fixing it does not need a re-quantize. This is metadata:

```bash
python3 tools/pxq-stamp-books.py --check  model.gguf              # what is missing
python3 tools/pxq-stamp-books.py          model.gguf fixed.gguf   # add the keys
python3 tools/pxq-stamp-books.py --verify model.gguf fixed.gguf   # per-tensor sha256 proof
```

The tensor bytes are copied verbatim — proven per tensor, not asserted (733/733 byte-identical
on the Fusion2 35B). The tool **refuses** a file whose `pxa.<tier>.version` says a non-default
book (`PXA_PXQ_CEIL_V2`, `PXA_PXQ2_V3`) was used and that does not carry that book: the table it
was built with is unknowable from the file, and writing the default there would produce
something confidently wrong. Re-quantize those instead.

Files written by this build carry a codebook for every tier they contain, and a run that somehow
emitted one without its book removes its own output and exits non-zero rather than shipping a
file nothing can read.

**You do not need an importance matrix for a PXQ tier** — the default ignores one if you pass it,
and a PXQ file made without one is the file this release was measured on. It is worth having as an
*option*, though: collected on text shaped the way you will actually use the model, an importance
matrix consumed with `PXA_PXQ_IMX=1` makes a measurably better file (numbers below). The default
stays off because collecting the matrix costs a calibration pass of its own. An imatrix also still
helps for the ordinary K-quants — and the release tarball carries `bin/llama-imatrix` for that.

### The imatrix: ignored by default, an option worth taking

Since 2026-08-24 every PXQ tier — `PXQ1`, `PXQ2`, `PXQ3`, `PXQ4`, `PXQ4-HQ`, `PXQ6` and
`PXQ_UNIVERSAL` — **drops an offered importance matrix instead of consuming it**, unless you
switch consumption on with `PXA_PXQ_IMX=1`. Pass `--imatrix` without that variable and the
quantizer prints

```
PXQ tiers: imatrix IGNORED (measured net-negative on the PXQ lattice; PXA_PXQ_IMX=1 to consume it)
```

once, and writes `quantize.imatrix.ignored_by` into the output file in place of the usual
`quantize.imatrix.*` provenance keys — so a PXQ artifact can never claim a consumption that did
not happen. (An auditor reading `quantize.imatrix.n_entries` gets `0`, which is the honest
answer.)

**About that message.** It is the wording from the 2026-08-24 measurement that set the default,
and this build still prints it; the reading behind it has since been corrected, and the correction
is worth your time. That A/B was run on **raw encyclopedia text** — out of distribution for a
model you are going to talk to — and on that text every way of consuming the matrix made a PXQ4
file slightly worse. Re-measured on **chat-templated** text, calibration and scoring both, the
same lever is a clear win at the tiers most people ship:

| model, tier | mean KL divergence | same top token | file size |
|---|---:|---:|---|
| Qwen3-0.6B, PXQ4, default | 0.0818 | 86.1 % | 399,573,408 B |
| Qwen3-0.6B, PXQ4, `PXA_PXQ_IMX=1` | **0.0609** | **87.6 %** | 399,573,472 B |
| Gemma 4 26B-A4B, PXQ3, default | 0.383 | 80.6 % | 11,013,679,104 B |
| Gemma 4 26B-A4B, PXQ3, `PXA_PXQ_IMX=1` | **0.320** | **81.7 %** | 11,013,679,392 B |

Same tier, same composition, same bytes to within the few dozen bytes of imatrix provenance the
header gains — the matrix does not change which tensor gets which tier, only which sub-scale and
anchor each block picks inside the tier it was already going to get.

So why is it still off by default? **Because the matrix is not free.** The quantize step itself
costs a second or two more, but collecting the matrix is a calibration pass over a corpus on a
GPU, and for a routed-expert (MoE) model that pass is the expensive part — about half an hour on
a pair of P100s for the Gemma 4 26B file above, and that was a *thin* calibration: 130 context
chunks, which left 6 of one layer's 128 experts unseen and those two tensors quantized unweighted.
A default has to be right for someone who just wants a file; an option can ask for half an hour.
The recipe is the optional step in
[guide 04](tutorials/04-quantize-your-own-model.md#optional--calibrate-first-with-an-importance-matrix).

Two things to know before you reach for it. **Calibrate in distribution**: the win came from
wrapping the calibration corpus in the model's own chat template, which is the whole point — a
matrix collected on raw text is what produced the original net-negative reading. And the evidence
is tier-dependent: PXQ4 and PXQ3 are clean wins on every metric measured; PXQ4-HQ improved on KL
divergence and top-token agreement but moved very slightly the wrong way on perplexity, and
PXQ1/PXQ2/PXQ6 have not been measured with it at all. (The switch lives in `pxq-quantize`, not in
the engine; the provenance-KV rewrite is in `examples/quantize/quantize.cpp`.)

An imatrix remains correct and worth having for **stock** tiers — including the `Q4_K_M`-style
requantize after `llama-pxq-export`, which is an ordinary k-quant and consumes it normally:

```bash
./llama-imatrix -m model-q8.gguf -f calibration.txt -o model.imatrix
./llama-quantize --imatrix model.imatrix model-q8.gguf model-q4km.gguf Q4_K_M $(nproc)
```

---

## 3. A PXQ GGUF to a vLLM-loadable directory

The vLLM backend does not read GGUF. Convert it. **PXQ2, PXQ3 and PXQ4 all convert**; the
other tiers (PXQ1, PXQ4-HQ, PXQ6) are llama.cpp-only and the converter refuses them by name
rather than skipping their tensors — see the truth table in [`VLLM.md`](VLLM.md#2-quant-tier-support--the-truth-table)
for which tier runs on which backend and what has actually been gated on a GPU.

```bash
cd tools/vllm-pxq4
python3 -m gguf_to_vllm.convert \
  --gguf   /models/model-pxq4.gguf \
  --ref-hf /path/to/original-hf-model \
  --out    /models/model-pxq4-vllm \
  --policy p1
```

- `--ref-hf` is the original HF checkpoint. The converter reads its `config.json` and
  tokenizer to build the output directory; it does not read its weights.
- `--policy` selects which modules are served from the panel tier rather than decoded to
  fp16. `p1` is the default and needs no encoder. On a **dense** artifact whose fused
  `qkv_proj` mixes a panel tier with q8_0, `p2c --fuse-uniform-pxq4` is what keeps that
  parameter quantized instead of fp16 — read the policy section of
  [`VLLM.md`](VLLM.md#policies) before choosing it: it costs one extra quantization pass.
- `--shard-size-gb` sets output shard size (default 4.0).
- `--dry-run` plans the whole conversion from the GGUF header alone and runs every
  structural self-check without writing bytes. **Do this first** — it takes seconds and
  catches name-mapping problems before you spend an hour writing 20 GB.

Pure Python and numpy: no torch, no CUDA, no vLLM, no GPU.

The result is a normal HF-style directory with sharded safetensors, the tokenizer, and a
`config.json` carrying `quantization_config.quant_method = "pxq4"` for a single-tier PXQ4
checkpoint, or `"pxq"` plus a `pxq_tiers` / `tier_books` pair when more than one tier is
present. `--quantization pxq` accepts both; `--quantization pxq4` accepts only the first.
Serve it:

```bash
docker run --rm --runtime=nvidia --gpus '"device=0,1"' \
  -v /models:/models -p 8000:8000 \
  ghcr.io/poisonxa16/pxa-vllm:sm60 \
    --model /models/model-pxq4-vllm --quantization pxq4 \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

Use the `sm70` tag on V100s. See [`tools/vllm-pxq4/README.md`](../tools/vllm-pxq4/README.md).

---

## Verifying before you trust it

Conversion is structural, not statistical: it can produce a file that loads and still be
wrong. Check the output before building anything on it.

```bash
python3 -m gguf_to_vllm.verify --gguf model-pxq4.gguf --vllm /models/model-pxq4-vllm
```

Then the only test that actually matters — ask it something and read the answer. A quant
that loads, reports sensible perplexity and then produces fluent nonsense is a failure
mode you will only catch by looking. Coherence is the gate.
