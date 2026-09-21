# 04 — Quantize your own model for the PXA engine

Take a model from Hugging Face and turn it into a file your card can actually hold. This is how
I make every file I publish.

Budget: half an hour for a small model, a few hours for a large one, almost all of it waiting.
It is CPU work — your graphics card sits idle for the whole thing.

---

## What you need first

**The source repository, not just the release.** This is one of two jobs the release tarball
cannot do on its own: the Hugging Face → GGUF converter is a Python script that lives in the repo.

```bash
git clone https://github.com/poisonxa16/pxa
```

**And `pxq-quantize`, a separate download, for the PXQ step.** As of this release, the tool that
*makes* a PXQ file (`pxq-quantize`) is not part of this source tree — get it from
`https://github.com/poisonxa16/pxq-quantize/releases`. Everything that *loads and runs* a PXQ file (the engine itself, and this
tree's own `llama-quantize` when you are reading or requantizing an existing PXQ file — see
[guide 10](10-when-something-goes-wrong.md) if that's what you're doing) is unaffected and needs
no separate download. `pxq-quantize` takes the exact same command-line arguments as the
`llama-quantize` commands you will see below; only the program name is different.

You also need Python with a few packages:

```bash
python3 -c "import torch, numpy, gguf, safetensors; print('ok')"
```

```
ok
```

If that fails: `pip install torch numpy gguf safetensors`. The CPU build of torch is fine — none
of this touches the GPU.

And you need the model itself, as a Hugging Face folder:

```bash
mkdir -p ~/models/my-model && cd ~/models/my-model
# download config.json, tokenizer files and every model-*.safetensors from the model's HF page
ls
```

```
config.json  generation_config.json  merges.txt  model-00001-of-00002.safetensors
model-00002-of-00002.safetensors  model.safetensors.index.json  tokenizer.json
tokenizer_config.json  vocab.json
```

> **You do not need an importance matrix.** Every other llama.cpp quantizing guide on the
> internet tells you to build a calibration corpus and run `llama-imatrix` first. **For PXQ,
> skip it.** The PXQ tiers ignore `--imatrix` — it measured worse, not better, on this codec —
> and the quantizer says so out loud if you pass one. An imatrix still helps for the ordinary
> K-quants, and `bin/llama-imatrix` is in the release package for that; for a PXQ tier it does
> nothing.

---

## Step 1 — Hugging Face folder to GGUF

```bash
cd ~/pxa/pxa-v2026.09.20/tools   # the tools/ folder inside the unpacked release
PYTHONPATH=$PWD/gguf-py python3 convert_hf_to_gguf.py ~/models/my-model \
  --outfile ~/models/my-model-f16.gguf \
  --outtype f16
```

```
INFO:hf-to-gguf:Writing the following files:
INFO:hf-to-gguf:/home/you/models/my-model-f16.gguf: n_tensors = 398, total_size = 4.0G
Writing: 100%|██████████| 4.00G/4.00G [00:16<00:00, 249Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to /home/you/models/my-model-f16.gguf
```

**What just happened.** Nothing was compressed yet. The converter read the model out of its
Hugging Face layout and rewrote it as one GGUF file at 16 bits per number — the same numbers, one
file instead of a folder. Everything after this works on GGUF files.

> **If it went wrong**
> - **`Model <SomethingForCausalLM> is not supported`** — that architecture has no converter
>   here. Not everything is supported; check the model's architecture in its `config.json`
>   against what the repo's converter knows.
> - **`ModuleNotFoundError: No module named 'gguf'`** — `pip install gguf`.
> - **It ran out of memory** — the converter loads the model a shard at a time, but a very large
>   model on a small machine can still struggle. Add swap.

---

## Step 2 — to Q8_0, the stepping stone

This step uses `llama-quantize` — it is not making a PXQ file yet, so the ordinary tool in this
release's `bin/` (or your build) is what you want.

```bash
cd ~/pxa/build/bin     # or wherever llama-quantize is; the release tarball has it in bin/
./llama-quantize ~/models/my-model-f16.gguf ~/models/my-model-q8.gguf Q8_0 16
```

```
llama_model_quantize_internal: ---- output composition (by bytes) ----
llama_model_quantize_internal:   q8_0         253 tensors    4075.68 MiB  100.0%
llama_model_quantize_internal:   f32          145 tensors       0.75 MiB    0.0%

main: quantize time =  8613.41 ms
```

The `16` on the end is how many CPU threads to use. Use however many cores you have.

**What just happened.** 8 bits per number instead of 16 — half the size, and close enough to the
original that nobody can tell. This step is not strictly required, but it makes everything
afterwards about twice as fast to read, and if you later want to try three different PXQ tiers
you do each one from this file instead of from the huge one.

---

## Step 3 — pick a tier

This is the only real decision. Each tier is a trade between size and quality:

| tier | bits per number | roughly, for a 27B model | use it when |
|---|---|---|---|
| `PXQ6` | 5.27 | ~18 GB | you have memory to spare and want the best answers |
| `PXQ4-HQ` | 4.52 | ~16 GB | a bit better than PXQ4 for a bit more size |
| **`PXQ4`** | **4.27** | **~15 GB** | **the default. Start here** |
| `PXQ3` | 3.27 | ~12 GB | PXQ4 does not fit and you can accept a small drop |
| `PXQ2` | 2.27 | ~8 GB | the only way to fit. Noticeably worse, still usable |

A quick way to choose: **take your card's memory, subtract about 2 GB for context and working
space, and pick the largest tier whose file fits in what is left.**

| your card | usually lands on |
|---|---|
| 11 GB (1080 Ti) | PXQ2 for a 27B–35B, PXQ4 for anything under about 14B |
| 16 GB (P100, V100) | PXQ4 up to about 27B, PXQ3 or PXQ2 above that |
| 32 GB (V100 32 GB) | PXQ4 on a 35B, PXQ6 if you want the quality |
| 2 × 16 GB | PXQ4 on a 35B; the two cards share it |

Names are case-insensitive but every doc writes them in capitals, so do that.

---

## Step 4 — quantize

This is the step that needs `pxq-quantize` (see "What you need first" above) — you are producing
a new PXQ file, not reading one.

```bash
./pxq-quantize --allow-requantize --i-know-this-is-double-lossy \
  ~/models/my-model-q8.gguf ~/models/my-model-PXQ4.gguf PXQ4 16
```

```
llama_model_quantize_internal: model size  =  4076.43 MB
llama_model_quantize_internal: quant size  =  2243.41 MB
llama_model_quantize_internal: ---- output composition (by bytes) ----
llama_model_quantize_internal:   pxq4         180 tensors    1747.12 MiB   77.9%
llama_model_quantize_internal:   q6_K           1 tensors     304.28 MiB   13.6%
llama_model_quantize_internal:   q8_0          72 tensors     191.25 MiB    8.5%
llama_model_quantize_internal:   f32          145 tensors       0.75 MiB    0.0%
```

4.0 GB in, 2.2 GB out, and it took a minute and a half on a busy machine.

**What just happened.** It squeezed the big rectangles of numbers down to 4-ish bits each, and
deliberately left some parts alone. The composition table at the end tells you what it actually
did: most of the file is now PXQ, and the rest is the parts it judged too important to squeeze —
the vocabulary table (`q6_K`) and a couple of attention pieces (`q8_0`). That mix is the recipe,
not a mistake.

Those two flags mean: `--allow-requantize` "yes, I know I am compressing something already
compressed", and `--i-know-this-is-double-lossy` "yes, I understand two rounds of rounding stack
up". If you would rather do one round, quantize straight from the f16 file and drop both flags —
it is slower to read but a touch cleaner.

> **If it went wrong**
>
> - **`PXQ composition assertion: target PXQ4 produced 36.4% PXQ-family bytes (floor 50%)` and
>   the output file is deleted.** This is the one you are most likely to hit on a *small* model,
>   and the quantizer is right to stop. On a small model the vocabulary tables — which never get
>   squeezed — are most of the file, so calling the result "PXQ4" would misdescribe it. I hit
>   this at 36.4% on a 0.6B model and 49.1% on a 1.7B. Your options, best first: **use a bigger
>   model** (PXQ exists for models too big for your card, and a 0.6B is not that); try one tier
>   up, which puts more bytes in the PXQ column; or use an ordinary `Q4_K_M` (with the ordinary
>   `llama-quantize`) for a model that small, which is what it is for.
>   `--pxq-composition-override` forces it through, and then the name on the file is not really
>   true.
>
> - **`REFUSING: the output filename says PXQ but the requested type is Q3_K (ftype 12), which is
>   NOT a PXQ tier. Nothing has been written.`** You passed a *number* instead of a name. Stock
>   type numbers run 0–38 and the PXQ tiers are 248 and 252–257, so a number from the wrong list
>   quietly produces the wrong thing. The message even lists the names for you. **Always pass the
>   name.** This guard exists because someone lost twelve minutes to exactly that.
>
> - **`NOTE: --imatrix supplied, but the PXQ tiers ignore it by default.`** Not an error — it is
>   telling you the calibration file you passed did nothing. Drop the flag, or keep it and add
>   `PXA_PXQ_IMX=1` to have it consumed — see the optional step below.
>
> - **It takes a very long time.** That is normal; it is reading and rewriting every number in
>   the model. Give it every core you have.

---

## Optional — calibrate first with an importance matrix

**Skip this the first time.** Everything above produces a good file on its own, and this step
costs you a calibration pass on a GPU before you can even start the quantize. Come back to it when
you care more about the last bit of quality than about the half hour.

An *importance matrix* is a recording of which parts of the model the calculation actually leans
on, gathered by running the full-size model over a sample of text. By default the PXQ tiers
**ignore** one, because the first time it was measured — on raw encyclopedia prose — it made files
slightly worse. Measured again on text shaped the way a chat model is really used, it is a clear
win at the same file size: mean KL divergence 0.0818 → 0.0609 and same-top-token 86.1 % → 87.6 %
on Qwen3-0.6B at PXQ4, and 0.383 → 0.320 and 80.6 % → 81.7 % on Gemma 4 26B-A4B at PXQ3.

**That "shaped the way you use it" is the whole trick.** Calibrate on raw text and you get the
old, unhelpful result back.

**1. Make a chat-shaped calibration corpus.** Take a few megabytes of ordinary prose you are
happy to calibrate on and wrap each passage in the model's own chat template, so the file you feed
the next command reads like turns, not like an encyclopedia:

```
<|im_start|>user
Tell me about the Severn bore.<|im_end|>
<|im_start|>assistant
The Severn bore is a tidal wave that travels up the River Severn …<|im_end|>
```

Use *your* model's markers — they differ per family, and the model's own
`tokenizer.chat_template` is the source of truth. Keep this text away from anything you will later
measure the file against; calibrating and scoring on the same passages flatters the result.

**2. Collect the matrix**, with the `llama-imatrix` in the engine package (this is the part that
wants a GPU):

```bash
./bin/llama-imatrix -m ~/models/my-model-q8.gguf \
  -f ~/models/calibration-chat.txt \
  -o ~/models/my-model-chat-calib.imatrix \
  -c 2048 -b 512 -ub 512 -ngl 99 -fa --chunks -1
```

It prints how many chunks it computed over. More is better: a couple of minutes is enough for a
small dense model, but a mixture-of-experts model needs enough text for every expert to be
selected — the Gemma 4 26B number above came from a 33-minute pass on two P100s that still left
6 of one layer's 128 experts unseen. The quantizer skips any tensor the matrix has no usable data
for and says which, so a thin calibration degrades gracefully instead of writing something wrong.

**3. Quantize with it consumed**, which is what `PXA_PXQ_IMX=1` turns on:

```bash
PXA_PXQ_IMX=1 ./pxq-quantize --imatrix ~/models/my-model-chat-calib.imatrix \
  --allow-requantize --i-know-this-is-double-lossy \
  ~/models/my-model-q8.gguf ~/models/my-model-PXQ4.gguf PXQ4 16
```

Without `PXA_PXQ_IMX=1` the `--imatrix` flag is accepted and ignored, and the quantizer says so —
if you do not see the matrix being loaded in the output, you left the variable off.

The output is the same tier, the same composition and the same size as before, give or take the
few dozen bytes of provenance the header gains, and it loads and runs exactly like any other PXQ
file. Measured where? PXQ4 and PXQ3 cleanly; PXQ4-HQ improved on two metrics of three; PXQ1, PXQ2
and PXQ6 have not been measured with it, so treat those as unknown rather than as a win.

---

## Step 5 — run it

```bash
./run-server.sh -m ~/models/my-model-PXQ4.gguf -ngl 99 -c 8192 -fa on
```

Check the log names your tier:

```
pxa: engine | codec=PXQ4 (180 tensors; pxq2 0, pxq3 0, pxq4 180, pxq6 0)
llm_load_tensors: offloaded 37/37 layers to GPU
```

Then ask it something, as in [guide 01](01-run-your-first-model.md).

You can also run a PXQ file on the CPU (`-ngl 0`) when the card is busy — it works and it is
slow, which is fine for checking that a file you just made is sound.

---

## Step 6 — check you did not break it

A file that loads is not a file that works. Do both of these.

### The cheap check, which catches most of it

**Ask it things and read the answers.** Five questions you know the answer to, at
`temperature: 0`. A broken quantization usually announces itself immediately: repeated tokens,
drifting off topic mid-sentence, confident answers that are not even the right *kind* of thing.

Include one very short, raw prompt — those are the sharpest test, because a chat template pads
your prompt and can hide the fault:

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The","n_predict":32,"temperature":0}'
```

Fluent text is a pass. Punctuation soup or an empty answer is a fail, and it is a fail worth
reporting even if a longer prompt works fine.

### The number, and how to read it

**Perplexity** scores how surprised the model is by ordinary text. Lower is better. On its own the
figure means nothing — what means something is **your quantized file compared against the file
you made it from, on exactly the same text**.

```bash
./llama-perplexity -m ~/models/my-model-q8.gguf   -f ~/text.txt -c 512 --chunks 8 -ngl 99
./llama-perplexity -m ~/models/my-model-PXQ4.gguf -f ~/text.txt -c 512 --chunks 8 -ngl 99
```

Any few hundred kilobytes of ordinary English will do for `~/text.txt`. Each run ends with:

```
Final estimate: PPL = <a number> +/- <a margin>
```

Read the **ratio** of the two, not either number:

| PXQ perplexity ÷ source perplexity | what it means |
|---|---|
| up to about **1.05×** | fine. This is what a good PXQ4 or PXQ3 looks like |
| **1.1× – 1.6×** | the expected cost of a low tier like PXQ2. Usable, and you will notice |
| **above about 2×** | something is wrong. Not "a low tier" — wrong |

If you want to be stricter, the same tool will compare two files token by token rather than in
aggregate: run the good file once with `--kl-divergence-base anchor.kld` to record its opinions,
then the candidate with `--kl-divergence-base anchor.kld --kl-divergence`. It reports a mean
divergence (smaller is better) and a "same top token" percentage — how often the two files would
have picked the same next word. Around 87% and up is a healthy cheaper tier; near 50% is a coin
flip and the file is not worth keeping.

**But the last word is still the cheap check.** A file can score respectably and still produce
fluent nonsense. Read the answers.

---

## Mixed tiers, when you are ready for them

You do not have to give the whole model the same tier. A **PXQU** file gives the parts that
matter most more bits and the rest fewer, so you get better answers for the same file size — it
is how I fit a 35B model into 16 GB.

You write a small text file of rules, one per line, and point `pxq-quantize` at it:

```bash
./pxq-quantize --allow-requantize --i-know-this-is-double-lossy \
  --pxq-universal ~/my-map.tiers \
  ~/models/my-model-q8.gguf ~/models/my-model-PXQU.gguf PXQ_UNIVERSAL 16
```

`--pxq-universal` must come **before** the file names — the quantizer stops reading flags at the
first one, and putting it last gets you `invalid ftype '--pxq-universal'`. Inside the map file
the tier names are **lowercase** (`pxq2`, `pxq3`, `pxq4`), unlike the name you pass on the
command line.

This is a deep end. Get a plain PXQ4 file working first.

---

## Where next

- Run it properly → [02](02-pick-settings-for-your-cards.md)
- Make it work with the other server → [05](05-quantize-for-vllm.md)
- The file is garbage → [10](10-when-something-goes-wrong.md)
