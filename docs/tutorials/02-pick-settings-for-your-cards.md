# 02 — Pick settings for your cards

Guide 01 gave you one command that works. This page is about the handful of things worth
changing, and — just as important — the things that are already decided for you.

**The short version.** The engine reads your cards at startup and fills in the batch geometry,
the kernel choices and the drafter by itself, and prints every decision. The only three things
you normally type are which card, how much context, and where the model is.

---

## 1. Tell it which card to use

With more than one card in the box, `-ngl 99` spreads the model over **all of them**. To pin it
to one, name the card in front of the command:

```bash
nvidia-smi -L
```

```
GPU 0: Tesla P100-PCIE-16GB (UUID: GPU-aad5ef40-...)
GPU 1: Tesla P100-PCIE-16GB (UUID: GPU-1e365a2e-...)
GPU 2: Tesla V100-PCIE-16GB (UUID: GPU-9af9aaf0-...)
```

```bash
CUDA_VISIBLE_DEVICES=0 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

With Docker it is `--gpus '"device=0"'` instead.

**What just happened.** `CUDA_VISIBLE_DEVICES` hides every card except the ones you list. The
numbers mean what `nvidia-smi -L` says they mean, because both `run-server.sh` and `pxa-launch`
pin `CUDA_DEVICE_ORDER=PCI_BUS_ID` for you. Without that pin, CUDA sorts your cards
fastest-first and `0` can quietly be a different card from the one `nvidia-smi` calls 0.

Always check the log agrees with you:

```
pxa: dev 0 Tesla P100-PCIE-16GB cc 6.0 -> path: sm_60 fp16-hfma2
```

---

## 2. One card

Nothing to decide. This is guide 01's command:

```bash
CUDA_VISIBLE_DEVICES=0 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

Fit the model to the card rather than the other way round:

| your card | a sensible tier |
|---|---|
| 11 GB (GTX 1080 Ti) | PXQ2, or a smaller model at PXQ4 |
| 16 GB (P100, V100) | PXQ4 on a mid-size model; PXQ2/PXQU on a 35B |
| 32 GB (V100 32 GB) | PXQ4 on a 35B, or PXQ6 for the best quality |

The tiers are explained in [guide 04](04-quantize-your-own-model.md). If you are downloading
rather than making your own, the file name usually says which one it is.

**One card and a big mixture-of-experts model — the Gemma 4 case.** If the model you want is
Gemma 4 26B-A4B (the 128-expert MoE) and you have exactly one 16 GB card, ask specifically for the
**PXQ3** tier of it. It boots at 16,000 tokens of context on that one card, where Google's own
published file of the same model only fits 4,096 tokens on the same card — four times the
conversation room for a small quality cost. `-sm layer` is the only split this model supports, so
there is nothing to choose there either.

---

## 3. Two identical cards

You have two choices, and they do genuinely different things.

### The default: layer split

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

The first half of the model goes on card 0, the second half on card 1. This is what you get
without asking, it works with every model and every file format, and its point is **capacity**:
two 16 GB cards let you run a model that needs 30 GB.

It does not make one conversation faster. While card 0 is working card 1 is waiting, and vice
versa — the work goes round the two cards in turn.

### The opt-in: tensor split

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on -sm tensor
```

Here is what `-sm tensor` does, without the jargon. A model is mostly big rectangles of numbers.
The layer split gives card 0 some whole rectangles and card 1 the others. The tensor split
instead **cuts every rectangle down the middle**, gives the left half to card 0 and the right
half to card 1, and has both cards work on the same token at the same time. They swap their
half-answers over the PCIe bus after each step and add them together.

So the layer split is two people taking turns at one job, and the tensor split is two people
doing one job together. That is why the tensor split can make a single conversation faster and
the layer split cannot.

**The raw server binary still defaults to the layer split** — nothing above turns `-sm tensor` on
by itself when you drive `run-server.sh` by hand, which is what this guide's commands do. **The
launcher (`tools/pxa-launch.py`) is different: as of this release, `--sm auto` (its default)
chooses `-sm tensor` for you** on two identical cards, an architecture and tier it has hardware
evidence for, with the fused all-reduce armed automatically — see [guide 00](00-start-here.md) and
[`docs/LAUNCHER.md`](../LAUNCHER.md#5b-which-split-mode-and-who-chooses). It is proven at
one-user-at-a-time on the cards I own and has not been through a long soak test at two concurrent
requests, which is why it still stops rather than guessing on anything it has not measured (see the
refusal table below). Pass `-sm layer` yourself, to the launcher or to the raw binary, to keep the
old behaviour.

**When it is worth trying**: two cards of the same model, in the same machine, and you care about
one conversation being fast rather than many conversations at once. It is **not** offered for
every model — an architecture has to be proven through it on real hardware first (see the refusal
table below), and Gemma 4 is one that has not been, so it stays on the layer split for now.

**After you turn it on, check three things:**

1. It booted at all — see the refusals below.
2. Ask it something at `temperature: 0` and read the answer. It should be as good as the layer
   split's. Different wording is fine; word salad is not.
3. The speed actually went up. [Guide 09](09-measure-your-card.md) shows how to measure that
   without fooling yourself.

If any of the three disappoints you, drop `-sm tensor` and you are back on the old path.

### When it refuses, and what the refusal means

`-sm tensor` stops rather than quietly doing something else. It prints a block like this:

```
==================================================================
PXA_TSPLIT: '-sm tensor' REFUSED
  why: a tensor split needs at least 2 devices and 1 is visible
  fix: use '-sm layer' on one card, or make a second GPU visible
  not falling back: '-sm tensor' was asked for explicitly. Set
  PXA_TSPLIT_FALLBACK=1 to demote to '-sm layer' instead of stopping.
==================================================================
```

Always read the `fix:` line — it is written for exactly your situation. The ones you are likely
to meet:

| `why:` says | what it means | what to do |
|---|---|---|
| needs at least 2 devices | you gave it one card | drop `-sm tensor` |
| has a tensor-split builder but has never been RUN through this split here | your model's architecture has never been tested in this mode on real cards, so nothing knows whether the result would be right | use `-sm layer`. You can override it (the message tells you how) but then you are the one making the measurement |
| requires flash attention | you passed `-fa off` | pass `-fa on` |
| a tier / file format with no hardware evidence | this release admits the file types it has actually run through the split | use `-sm layer` for that file |

**Why I refuse instead of falling back.** The failure this protects you from is not a crash. It
is a model that loads, produces fluent text, and is quietly wrong — or a split that left half the
model uncut, so the speed number you take is not a tensor-split number at all. Stopping costs you
one command. Not stopping costs you a wrong answer you never notice.

If you would rather it fell back to the layer split instead of stopping, set
`PXA_TSPLIT_FALLBACK=1`. The refusal is still printed either way.

---

## 4. Four cards

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run-server.sh -m your-model.gguf -ngl 99 -c 32768 -fa on
```

Layer split, as usual, and four cards is where it earns its keep: it is how you run a model far
too big for any one card. The context can be much larger too, because the KV cache is spread
across four cards' memory instead of one.

Do not use `-sm tensor` on four cards in this release. It is written and it is not soaked; four
cards stay on the ordinary route.

---

## 5. Mixed cards

A P100 and a V100 in one box, or a 1080 Ti alongside a P100, will work — the engine makes a
**per-card** decision about which of its tricks to use, so each card gets what suits it.

```bash
CUDA_VISIBLE_DEVICES=0,2 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

Two honest warnings:

- **The whole thing runs at the pace of the slowest card**, because every token has to visit
  both. A fast card paired with a slow one is not the average of the two.
- **`-sm tensor` is for identical cards.** Cutting every rectangle in half makes sense only if
  both halves take the same time.

If the two cards are very different sizes, tell it how to divide the model with `-ts`, in
proportion to the memory you want each to hold:

```bash
CUDA_VISIBLE_DEVICES=0,2 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on -ts 11,16
```

The launcher works this ratio out for you from free memory; you only need `-ts` if you want to
override it.

---

## 6. Context length, and why it costs memory

`-c` is how many tokens the model can have in front of it — your whole conversation plus what it
is writing.

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 32768 -fa on
```

The catch is the **KV cache**: the model's working memory for the conversation, which lives in
VRAM and grows with every token of context you allow. It is reserved when the server starts, not
when you start typing, so `-c 32768` costs that memory the whole time even if you only ever send
three words.

Rules of thumb:

- **Chatting**: `-c 8192` is plenty and leaves room.
- **Feeding it documents**: raise it, and expect to drop a tier or use more cards.
- **It loads and then dies on the first token**: your context is too big. Halve `-c`.

You can also shrink the cache by storing it in 8 bits instead of 16:

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0
```

That roughly halves the cache's memory for a small quality cost. It is a fair trade when the
alternative is not fitting.

**If you do not pass `-c` at all**, the engine no longer fails outright the way an older build
would. It retries at smaller and smaller contexts until one actually fits and tells you which one
it picked — see [guide 10](10-when-something-goes-wrong.md) for the details and how to turn that
off if you would rather it failed loudly.

---

## 7. How to tell it fits, before you wait for a load

This is the single most useful thing on this page. The launcher will do the arithmetic and start
nothing:

```bash
./pxa-launch --explain -m your-model.gguf --gpus 0 --workload chat
```

Among the output, look for this line:

```
** VRAM estimate [INFERRED, never blocks]: weights 9.94 GiB + KV 0.08 GiB
   (= ctx 4096 x 20.0 KiB/tok) = 10.01 GiB vs 16.00 GiB free / 16.00 GiB total
   across 1 card(s). Compute buffers and fragmentation are NOT in this number.
** HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load.
```

Read it like this: **weights + KV, compared against free VRAM, and leave about 1.2 GB spare per
card on top.** If the total is within about 1.2 GB of your free memory, it will probably load and
then fall over on the first token — that last bit is the working space the number above does not
include.

Change `--gpus` and re-run to try a different card set; change `-c` to see what a bigger context
would cost. It takes two seconds and loads nothing.

The launcher also flatly refuses, before loading anything, if the weights alone cannot fit the
cards you picked, or if `-c` is larger than the model was trained for. Those two refusals are
arithmetic on the file's own bytes, so they are never wrong.

**What just happened.** `--explain` runs the whole decision process — which engine, which flags,
which card — prints it with the reason for each choice, and then stops instead of starting a
server. Drop `--explain` and the same decision actually runs.

> **If it went wrong**
> - **`--explain` says "REFUSING: -c ... exceeds this model's trained context"** — the model was
>   never trained that long. Lower `-c`.
> - **It fits on paper and OOMs in practice** — something else is on the card. Check with
>   `nvidia-smi`; a desktop session can quietly hold a gigabyte.
> - **The estimate looks wildly wrong** — it is labelled `[INFERRED]` for a reason: on some
>   architectures the per-token KV arithmetic is an approximation. Trust it as a guide, not a
>   guarantee, and keep the 1.2 GB of headroom.

---

## 8. The one flag people get wrong

`-fa` (flash attention) has two settings and they suit different jobs:

| | |
|---|---|
| `-fa on` | chatting, and serving several people. This is what you want almost always. |
| `-fa off` | feeding in very long documents, where the read phase dominates |

One server carries one setting, so if you do both, run two servers on two ports.

---

## Where next

- Making it faster without changing the answers → [03](03-going-faster.md)
- Long documents and follow-up questions → [08](08-long-chats.md)
- It will not fit no matter what → [04](04-quantize-your-own-model.md), and pick a lower tier
