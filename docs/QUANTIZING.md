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

Everything below needs only `llama-quantize` (in the release tarball, or built from this
repo) and Python. No GPU is required to quantize.

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

For a split GGUF, pass the **first** shard — the rest are found automatically. An
imatrix is optional for PXQ4; if yours reports
`load_imatrix: failed reading name for entry N` it is unreadable by this build, and
dropping the flag still produces a valid artifact.

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

An imatrix is optional but helps the low tiers:

```bash
./llama-imatrix -m model-q8.gguf -f calibration.txt -o model.imatrix
./llama-quantize --imatrix model.imatrix model-q8.gguf model-pxq4.gguf PXQ4 $(nproc)
```

---

## 3. PXQ4 GGUF to a vLLM-loadable directory

The vLLM backend does not read GGUF. Convert it:

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
- `--policy` selects the tensor-name mapping. `p1` is the default.
- `--shard-size-gb` sets output shard size (default 4.0).
- `--dry-run` plans the whole conversion from the GGUF header alone and runs every
  structural self-check without writing bytes. **Do this first** — it takes seconds and
  catches name-mapping problems before you spend an hour writing 20 GB.

Pure Python and numpy: no torch, no CUDA, no vLLM, no GPU.

The result is a normal HF-style directory with `config.json` carrying
`quantization_config.quant_method = "pxq4"`, sharded safetensors, and the tokenizer.
Serve it:

```bash
docker run --rm --runtime=nvidia --gpus '"device=0,1"' \
  -v /models:/models -p 8000:8000 \
  ghcr.io/poisonxa16/pxa-vllm:sm60 \
    --model /models/model-pxq4-vllm --quantization pxq4 \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

Use the `sm70` tag on V100s. See [`../docker/vllm-pxq4/README.md`](../docker/vllm-pxq4/README.md).

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
