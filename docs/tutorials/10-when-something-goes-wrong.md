# 10 — When something goes wrong

The failures people actually hit, in the order they hit them. Find your symptom, then read
"where the logs are" and "how to ask for help" at the bottom before you give up.

---

## It will not start at all

**`error while loading shared libraries: libcuda.so.1`**
No NVIDIA driver on this machine, or the binary cannot see it. `nvidia-smi` is the test. This is
the correct failure for a box with no driver — nothing is wrong with the package.

**`version GLIBC_2.34 not found`, or `GLIBCXX_...`**
Your Linux is older than the binaries need. There is no flag for this. Use the container image
instead — it brings its own userspace — or move to a newer distro.

**`CUDA driver version is insufficient for CUDA runtime version`**
Your driver is older than 570. Upgrade the **driver**. Do not install a CUDA toolkit; the package
carries its own CUDA runtime, and installing a toolkit will not change this.

**`./run-server.sh: Permission denied`**
The extract lost the executable bit: `chmod +x run-server.sh pxa-launch bin/*`.

---

## Out of memory when it loads

Symptoms: it reads the file for a while and then dies, or it loads fully and then dies on the
very first token.

That second one catches people out, because the model "fit". It fit, and then the first token
needed working space that the load did not reserve.

Work through this in order:

1. **Check the card is actually empty.**

   ```bash
   nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
   ```

   ```
   index, name, memory.used [MiB], memory.total [MiB]
   0, Tesla P100-PCIE-16GB, 5 MiB, 16384 MiB
   ```

   A desktop session, a stale server or another container can quietly hold a gigabyte or three.

2. **Do the arithmetic before you wait again.**

   ```bash
   ./pxa-launch --explain -m your-model.gguf --gpus 0 --workload chat
   ```

   Read the `** VRAM estimate` line and the `** HEADROOM RULE` line under it. You need about
   1.2 GB free per card **after** everything loads. See
   [guide 02 §7](02-pick-settings-for-your-cards.md).

3. **Lower the context.** `-c 8192` instead of `-c 32768`. The KV cache is reserved up front and
   it is usually the thing that tipped you over. If you did not pass `-c` at all, the engine
   already retried smaller sizes by itself before giving up — see "What changed by itself" below.

4. **Shrink the cache instead of the context:** add `-ctk q8_0 -ctv q8_0`. Roughly halves it.

5. **Step down a tier.** A PXQ3 or PXQ2 file of the same model is much smaller. See
   [guide 04](04-quantize-your-own-model.md).

6. **Add a card**, if you have one. `CUDA_VISIBLE_DEVICES=0,1`.

---

## It refused, with a block of text

The engine refuses rather than silently doing something else. A refusal looks like this:

```
==================================================================
PXA_TSPLIT: '-sm tensor' REFUSED
  why: a tensor split needs at least 2 devices and 1 is visible
  fix: use '-sm layer' on one card, or make a second GPU visible
  not falling back: '-sm tensor' was asked for explicitly. Set
  PXA_TSPLIT_FALLBACK=1 to demote to '-sm layer' instead of stopping.
==================================================================
```

**Read it in this order: `fix:` first, then `why:` if you want to know what it was protecting you
from.** The `fix:` line is written for your exact situation and is usually the whole answer.

Refusals exist because the alternative is worse. The failures they catch are not crashes — they
are configurations that load, produce fluent text, and are quietly wrong, or that silently do
something other than what you asked. You would never notice. Stopping costs you one command.

The quantizer refuses in the same style, and its messages also tell you the flag that overrides
them. If you use an override, you are taking responsibility for checking the result.

---

## The first request takes forever, then everything is fine

That is normal and expected. Loading weights off disk, building the computation plan and warming
up the drafter all happen once. See [guide 03 §6](03-going-faster.md).

If it is *every* request that is slow, that is different — read on.

---

## Everything is slow, all the time

Check these four, in order:

1. **`offloaded N/N layers to GPU`** in the startup log. If it says `0/N`, the model is running
   on your CPU at roughly a fiftieth of the speed. Either `-ngl 99` is missing, or something has
   put a CUDA *stub* library on `LD_LIBRARY_PATH` and the real driver is being shadowed. Clear
   `LD_LIBRARY_PATH` and use `./run-server.sh`, which sets it correctly itself.

2. **`pxa: dev 0 <card>`** — is that the card you meant? On a mixed box,
   `CUDA_VISIBLE_DEVICES=0` can mean a different card from the one `nvidia-smi` calls 0 unless
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. `run-server.sh` and `pxa-launch` set it; a bare
   `bin/llama-server` does not.

3. **Is something else on the card?** `nvidia-smi`. If the engine saw less free memory at startup
   than you expect, it sized its batches down to stay safe, and the `PXA posture:` line in the log
   says what it chose.

4. **Are you measuring what you think you are measuring?** [Guide 09](09-measure-your-card.md)
   exists because this is the usual answer.

---

## Two different cards, and it is slower than one

Expected. With the ordinary layer split, every token has to visit both cards in turn, so the pair
runs at the pace of the slower one plus the cost of moving data between them. A fast card paired
with a slow one is not the average.

Two cards are for running a model that does not fit on one. If both cards are identical and you
want one conversation to go *faster*, that is what `-sm tensor` is for — see
[guide 02 §3](02-pick-settings-for-your-cards.md).

---

## Two server slots gave different answers to the same prompt

If it is a genuine near-tie in the model's own output, this is expected and explained in
[guide 08 §4](08-long-chats.md) and [guide 03 §4](03-going-faster.md) — it is not content
crossing between slots. If you need to check that for yourself, send two *different* prompts to
the two slots at once and confirm each answer only ever carries its own content; that is the
actual correctness property this engine guarantees. If you need repeatable output, compare one
slot at a time, at `-np 1`, with `--spec-type none`.

---

## The output is garbage

Tell these two apart first, because they have completely different causes.

**Fluent text that is factually wrong** — that is the model. Small models make things up
confidently. Try a bigger model or a higher tier.

**Not text at all** — repeated tokens, punctuation soup, `!!!!`, an empty answer to a one-word
prompt. That is a real fault and I want to hear about it. Common causes, in order:

1. **A quantization tier too low for that model.** PXQ2 is aggressive. Check the file against a
   higher tier before blaming anything else — [guide 04 §6](04-quantize-your-own-model.md) shows
   how to measure the damage rather than guess.
2. **A file that was damaged in transit.** Check the download against its published checksum.
3. **You overrode a refusal.** If you passed `PXA_TSPLIT_UNPROVEN_ARCH=1`,
   `--pxq-composition-override`, or `--pxq-name-override`, take it off and see if the problem goes
   away. That is what those refusals were for.
4. **A genuine bug.** Short, raw, non-chat prompts are the sharpest test — a chat template pads
   your prompt and can hide the fault:

   ```bash
   curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
     -d '{"prompt":"The","n_predict":32,"temperature":0}'
   ```

   If *that* comes back as nonsense while long prompts are fine, please report it. Do not work
   around it with a longer prompt — that is the bug hiding, not leaving.

---

## Undo: turning this release's new things off again

Most of what is new in this release is opt-in: if you did not turn it on, it is not running. If you
did turn something on and the machine started behaving oddly, here is the off switch for each.
Remove the flag or variable and restart the server; there is no state to clean up.

| you turned on | how to turn it off |
|---|---|
| `-sm tensor` (the tensor split) | remove `-sm tensor` from the command line |
| `PXA_TSPLIT_REDUCE=fused` (the faster tensor-split reduce) | remove the variable; only relevant with `-sm tensor` |
| `PXA_SPEC_POLICY=1` (the trained MTP guesser, per-request depth) | remove the variable |
| `PXA_SPEC_SAMPLED=1` (the new lossless sampling rule) | remove the variable |
| `PXA_CACHE_PARK_SPEC=1` (parking a conversation in host RAM) | remove the variable |
| `PXA_GEMMA4_ASSISTANT=1` (the Gemma 4 assistant drafter) | remove the variable |
| any `PXA_*` you copied off a web page | remove it. The engine picks these per card by itself |

A few things in this release **changed by themselves**, because the old behaviour was the thing
people tripped over. Each prints one `PXA_AUTO:` line at boot saying what it did, and each has an
off switch if you want the old behaviour back:

| what the engine now does by itself | how to get the old behaviour |
|---|---|
| A boot with no `-c` no longer dies asking for the model's whole trained context window; it settles on a context that fits and tells you | pass `-c N` yourself, or `PXA_AUTO_CTX=0` to fail as before |
| Gemma 4 answers a chat request directly instead of thinking first (that is the default the model ships with; a short answer budget used to come back empty) | `--reasoning on`, or per request `"chat_template_kwargs": {"enable_thinking": true}`; `PXA_AUTO_REASONING=0` restores the old default |
| Gemma 4 and Qwen3.8 chat go through the model's own template, so a Qwen3.8 reply's thinking arrives in `reasoning_content` and the answer in `content` (it used to arrive as one block inside `content` unless you passed `--jinja`) | `PXA_AUTO_JINJA=0` |
| Gemma 4 26B-A4B (the mixture-of-experts one) loads like any other model; it used to be refused | `PXA_GEMMA4_MOE=0` refuses it again |
| On V100s, Gemma 4 writes long answers faster (a new attention kernel for its 512-wide heads, used only once the conversation is past ~1,000 tokens) | `PXA_FA_D512_VOLTA=0` |
| A PXQ file of Gemma 4 26B-A4B runs through the fast expert kernels, and on two or more V100s the expert routing table is built on the device instead of the host | `PXA_PXQ_MOE_GELU=0`, `PXA_MOE_DEVICE_MAP=0` |

And two bigger hammers, for when you suspect the engine's automatic choices rather than a flag
you set:

```bash
# the plainer, older set of automatic choices
PXA_ENHANCE=0 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on

# everything off — the slow reference path. If the fault disappears here, it is one of the
# tricks, and that is exactly what I need to know.
PXA_REFERENCE=1 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

---

## Where the logs are

There is no log file. The server writes to the terminal, and it writes the important part to
**stderr** while the HTTP lines go to stdout — so capture both:

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on 2>&1 | tee pxa.log
```

With Docker:

```bash
docker logs pxa > pxa.log 2>&1
```

The first hundred lines are the half that matters. That block of `PXA config level:`,
`PXA level=`, `PXA posture:` and `PXA_AUTO:` lines is the engine telling you every decision it
made and why — it is the most useful thing you can send me.

---

## How to ask for help

Come to Discord: **<https://discord.gg/EqazvV9tf>**

Paste these five things. It is two minutes of your time and it usually turns a week of
back-and-forth into one reply:

1. **`./run-server.sh --version`**

   ```
   version: 5592 (518e0a7b7a)
   ```

2. **`nvidia-smi -L`** and the driver line from `nvidia-smi | head -3`.

3. **The exact command you ran.** Not a description of it — the line itself.

4. **The first ~100 lines of the log**, including the whole `PXA_AUTO:` block. Put it in a file
   or a paste, not in the chat.

5. **What you expected and what you got.** If it is a speed question, read
   [guide 09](09-measure-your-card.md) first — half of all "it's slow" reports turn out to be one
   of the three traps on that page, and you will find it yourself in ten minutes.

Please do report the ugly stuff. A model that produces punctuation soup on a one-word prompt is a
bug I want, and a bug report with those five items is one I can usually fix.

---

## Supporting the project

I build and test this on second-hand cards in my own house, and I give it away. If it saved you
from buying a new GPU and you would like to chip in:

**<https://ko-fi.com/shatteredrealms1>**

Entirely optional, and it will never gate anything. Telling someone else with an old card that
this exists helps just as much.
