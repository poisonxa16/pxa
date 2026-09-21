# 05 — Quantize for the vLLM sidecar

There are two servers in this project. This page is about making a model file for the second one.

> **Honesty note.** The two `--dry-run` commands on this page I ran here on the day I wrote it,
> on real PXQ files, and the output below is what they printed. The full conversion and the
> serving that follows it are the commands I use on my own box, but I did not re-run them on the
> day of writing — so treat this page as the shape of the job rather than a transcript, and read
> what the dry run tells you before you commit hours to a conversion.

---

## 1. Which server do you actually want?

| | **the engine** (guides 01–04) | **the vLLM sidecar** |
|---|---|---|
| reads | `.gguf` files, every PXQ tier | a *converted folder*, PXQ4 mainly |
| best at | one or two people, lowest delay per token | many people at once |
| runs on | P100, 1080 Ti / P40, V100 | P100 and V100. **Not the 1080 Ti** |
| set-up | download a file, point at it | convert first, then serve |

The honest version is a bit more nuanced than "few users / many users":

- **A dense model?** The sidecar tends to win at *every* level of load, including a single
  conversation, because it can make both cards work on one token at once.
- **A mixture-of-experts model?** The engine wins while only a few people are using it, and the
  sidecar overtakes somewhere around six simultaneous conversations.
- **Feeding it very long documents?** The engine.
- **A GTX 1080 Ti?** The engine. The sidecar has no code for that card.

If you do not want to think about it, the launcher decides from measurements rather than
folklore:

```bash
./pxa-launch --explain -m your-model.gguf --gpus 0,1 --np 8
```

It prints `ENGINE = llama` or `ENGINE = vllm` with the reason, and runs nothing.

---

## 2. What you need before you start

1. **A PXQ4 `.gguf`** — made as in [guide 04](04-quantize-your-own-model.md), or downloaded.
2. **The original Hugging Face folder** the model came from. The converter copies the
   configuration, the tokenizer and anything vision-related out of it. You cannot skip this; a
   `.gguf` does not carry everything the sidecar's loader expects.
3. Disk for the output, which is roughly the same size as the input.

### Which tiers the sidecar can read

| tier | converts | serves | notes |
|---|---|---|---|
| **PXQ4** | yes | yes | the one to use |
| PXQ3 | yes | yes | works; its answers track the engine's closely but not exactly |
| PXQ2 | yes | yes | same caveat |
| PXQ4-HQ | **no** | no | no kernel for it on this side |
| PXQ6 | **no** | no | same |
| PXQ1 | **no** | no | same |

**So: quantize at PXQ4 if the destination is the sidecar.** The engine reads everything; the
sidecar is the fussy one, and it is fussy on purpose — a tier it cannot decode properly would
load, run, and be quietly wrong.

---

## 3. Dry-run it first. Always.

The conversion reads a whole model and writes another one. Check the plan before you spend the
time:

```bash
cd tools/vllm-pxq4
python3 -m gguf_to_vllm.convert --gguf ~/models/your-model-PXQ4.gguf --dry-run
```

```
gguf: 866 tensors, 62 KVs, arch=qwen35
  f32      360 tensors     10,686,464 B
  pxq4     325 tensors 12,231,950,336 B
  q6_K       1 tensors  1,042,944,000 B
  q8_0     180 tensors  3,225,354,240 B

plan (policy=p1): 1155 output tensors, 22,011,264,000 B
  dense    547 tensors  10,506,779,648 B
  pxq4     608 tensors  11,504,484,352 B
  skipped:   15 ggml tensors (15x MTP / not mapped in P1-P2 ...)
  WARNING: 2,013,265,920 B (1.88 GiB) of this plan is native PXQ on disk but served DENSE f16
  because policy p1 does not name the module ...

decode weight bytes read per GPU per token (PROJECTION INPUT, not a measurement):
  TP=2: 9.066 GiB/GPU  (9,734,902,784 B)
```

**What just happened.** It read only the file's header — seconds, not minutes — and told you
exactly what it would produce: how many tensors, how big, what it would skip, and which parts
would end up stored in full 16-bit precision rather than 4-bit. That last warning is the useful
one: it means a chunk of your model would take four times the memory it needs to. It is telling
you before you find out the hard way.

The last line is the sidecar's own estimate of how much weight data each card has to read per
token. It is arithmetic, not a measurement, but it is the number that decides your decode speed.

If the file is not convertible, the dry run says so plainly rather than producing something
broken:

```
policy p1 requires re-encoding blk.0.attn_qkv.weight (mxfp4 -> pxq4) but no --encoder was
given. Refusing to silently fall back to fp16: that would produce a checkpoint that loads,
runs, and is quietly slower than the policy claims.
```

---

## 4. Convert

```bash
cd tools/vllm-pxq4
python3 -m gguf_to_vllm.convert \
  --gguf   ~/models/your-model-PXQ4.gguf \
  --ref-hf ~/models/the-original-hf-folder \
  --out    ~/models/your-model-PXQ4-vllm \
  --policy p1
```

This takes a while — it is reading and rewriting every tensor in the model.

What you get is a **folder**, not a file:

```bash
ls ~/models/your-model-PXQ4-vllm
```

```
config.json  model-00001-of-000NN.safetensors  ...  tokenizer.json  tokenizer_config.json
```

Point the sidecar at the **folder**. This is the mistake everyone makes once: the sidecar cannot
read a `.gguf`, and handing it one gets you a confusing error a long way from the real cause.

The converter checks its own work as it goes and refuses rather than writing something it is not
sure about. If it refuses, read the message — like the engine's refusals, it names the problem
and the flag that addresses it.

> **If it went wrong**
> - **"the GGUF carries pxq4 tensors but no `pxa.pxq4.book` / `pxa.pxq4.sub`"** — your file was
>   made before those tables were written into files. Add `--books-from-defaults`, which
>   substitutes the built-in tables and records in the output that it did so. Any PXQ file
>   published before the 2026-09-09 release needs this.
> - **"ggml types this converter cannot decode"** — your file is at a tier the sidecar has no
>   kernel for (PXQ4-HQ, PXQ6, PXQ1). The message names them. Re-quantize at PXQ4, or just serve
>   the file with the engine, which reads every tier.
> - **"violates the §3.1 uniformity invariant"** — parts of the model that the sidecar fuses into
>   one block came out at different tiers. Quantize with a policy that keeps them uniform, or use
>   the engine.
> - **It ran out of disk** — the output is about the size of the input, and both exist at once.

---

## 5. Check it before you trust it

The converter ships its own checks. Run them once after you set up, so that a later failure is
about your model and not your installation:

```bash
cd tools/vllm-pxq4/src
python3 test_pxq4_config.py
python3 gguf_to_vllm_test.py
```

Then, when the sidecar is up ([guide 06](06-run-the-vllm-sidecar.md)), do the test that actually
matters: **ask it something and read the answer.** A checkpoint that loads, reports plausible
numbers and then produces fluent nonsense is a real failure mode, and the only thing that catches
it is your eyes.

One more honest note. On PXQ3 and PXQ2 the sidecar and the engine do not give byte-identical
answers on the same file — they agree on most tokens and not all, because they decode the same
bytes with different kernels. At PXQ4 they line up. If you need the two servers to agree exactly,
use PXQ4.

---

## Where next

- Start the sidecar → [06](06-run-the-vllm-sidecar.md)
- Both servers at once → [07](07-docker-compose.md)
- Making the PXQ4 file in the first place → [04](04-quantize-your-own-model.md)
