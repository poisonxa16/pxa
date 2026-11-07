# 04 — Quantize your own model for the PXA engine

Take a model from Hugging Face and turn it into a file your card can actually hold. You do this in
the **Encode** tab of PXA Control.

Budget: half an hour for a small model, a few hours for a large one, almost all of it waiting.
The Free encoder is CPU work, so your graphics card can stay busy with something else.

---

## What you need first

**PXA Control running.** Unpack the release tarball and start `./pxa`, as in [guide 01](01-run-your-first-model.md).
Control opens in your browser.

**The converter's Python packages.** They are not in the tarball. Install them once:

```bash
python3 -m venv ~/pxa-convert && ~/pxa-convert/bin/pip install -r tools/requirements-convert.txt
```

then start PXA Control with `PXA_CONVERT_PYTHON=~/pxa-convert/bin/python`:

```bash
PXA_CONVERT_PYTHON=~/pxa-convert/bin/python ./pxa
```

The list itself (`tools/requirements-convert.txt`) is in the release tarball. From a source
checkout use `requirements/requirements-convert_hf_to_gguf.txt` and
`requirements/requirements-convert_legacy_llama.txt`. The CPU build of torch is fine.

**The encoder.** If Encode says no quantizer was found, press **Get the encoder**. **Free** makes
the classic PXQ tiers and needs no key. **Pro** makes the PXQN tiers; it is a supporter reward
(unlock it with Store coins from a Ko-fi donation, or paste the key the PXA Network Discord gives
supporters with `/encoder`).

**Disk space.** Free disk for the source model, the converted file and the result. A Pro encode of a
27B model needs about 260 GB while it runs.

> **You do not need an importance matrix.** Skip the calibration steps that other quantizing guides
> start with. The PXQ tiers do not use one.

---

## Step 1 — Source

Open the **Encode** tab. It walks you through five steps: Source, Target, Checks, Run, Done.

In Source, give it a Hugging Face model name, or a folder or a GGUF file on this computer, and press
**Look it up**. Check the model's licence on its Hugging Face page first, and accept its terms there
if it asks.

Models that convert, load and generate in our tests: Qwen3 and Qwen2, Llama, Mistral, Gemma 3, Phi
and the Nemotron families. Gemma 3 vision towers are skipped (text only). If **Look it up** says the
architecture is not supported, that model has no converter here.

---

## Step 2 — Target: pick a tier

This is the only real decision. Encode shows which tiers fit your cards, with their quality class,
estimated file size and, where we have a measurement, the decode speed. The plan names a tier that
fits the GPU memory it sees. The README's tier table
(["PXQN tiers and what they are worth"](../../README.md)) has the measured quality for each.

A quick way to choose: **take your card's memory, subtract about 2 GB for context and working
space, and pick the largest tier whose file fits in what is left.**

| your card | usually lands on |
|---|---|
| 11 GB (1080 Ti) | the smallest tiers for a 27B–35B, a mid tier for anything under about 14B |
| 16 GB (P100, V100) | a mid tier up to about 27B, a smaller one above that |
| 32 GB (V100 32 GB) | a mid or high tier on a 35B |

Tier names are written in capitals everywhere (PXQ4, PXQN4 and so on).

---

## Step 3 — Checks and Run

Run the **Checks** and start the job in **Run**. It
shows progress, and a running job resumes after a crash, a reboot, or after you free disk space.
When it finishes, **Done** shows the file.

**It takes a very long time.** That is normal; it reads and rewrites every number in the model.

> **If it went wrong**
>
> - **Encode says it cannot convert the model.** The converter's packages are missing or Control was
>   not started with `PXA_CONVERT_PYTHON`. Run the `python3 -m venv` line above and start Control as shown.
> - **Encode says no quantizer was found.** Press **Get the encoder**.
> - **It stopped because the file would be mostly vocabulary tables.** On a very small model the
>   tables that never get squeezed are most of the file, so the name "PXQ4" would misdescribe it.
>   Use a bigger model (PXQ exists for models too big for your card), or try one tier up.
> - **It ran out of memory or disk.** Free some space and resume the job.
> - **It says a model is locked.** The file was encoded with Pro and locked. Paste your key into
>   the key field on the Encode tab.

---

## Step 4 — run it

Put the finished file where Control's **Home** can see it, pick it, and press **Start**. Check the
log names your tier:

```
pxa: engine | codec=PXQ4 (180 tensors; pxq2 0, pxq3 0, pxq4 180, pxq6 0)
```

Then ask it something, as in [guide 01](01-run-your-first-model.md).

---

## Step 5 — check you did not break it

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
./bin/llama-perplexity -m ~/models/my-model-source.gguf -f ~/text.txt -c 512 --chunks 8 -ngl 99
./bin/llama-perplexity -m ~/models/my-model-PXQ4.gguf -f ~/text.txt -c 512 --chunks 8 -ngl 99
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

## Where next

- Run it properly → [02](02-pick-settings-for-your-cards.md)
- Make it work with the other server → [05](05-quantize-for-vllm.md)
- The file is garbage → [10](10-when-something-goes-wrong.md)
