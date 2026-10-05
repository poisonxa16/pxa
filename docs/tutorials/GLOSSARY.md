# Glossary

Every word in these guides that is not an ordinary English word, explained in two sentences.
If something here is still unclear, ask on Discord — <https://discord.gg/EqazvV9tf> — and I will
rewrite the entry.

---

**Model** — the file that does the thinking. It is a few gigabytes of numbers that were learned
from text, and running it is called *inference*.

**Weights** — those numbers. A model with more weights usually knows more and takes more memory.

**VRAM** — the memory on the graphics card itself, separate from your computer's normal RAM. A
16 GB card has 16 GB of VRAM, and the whole model has to fit in it on this engine.

**Quantization** — squashing the model's numbers into fewer bits each so the file gets smaller.
A model stored at 16 bits per number might be 70 GB; the same model at 4 bits is about 18 GB, fits
on a card you can afford, and answers almost identically.

**Bits per weight (bpw)** — how many bits each number gets after squashing. Fewer bits means a
smaller file and a slightly dumber model; the whole art is choosing the fewest bits you can get
away with.

**GGUF** — the file format this engine reads. One `.gguf` file holds the weights, the vocabulary
and the settings, so there is exactly one file to download and one file to point at.

**PXQ** — my own family of quantization formats, written as PXQ2, PXQ3, PXQ4, PXQ4-HQ and PXQ6.
They are laid out so the old Pascal and Volta cards can decode them quickly, which is the entire
reason this project exists. Made with `pxq-quantize`, a separate download from this engine.

**PXQU** — a *mixed* PXQ file, where different parts of the model get different numbers of bits.
The parts that matter most are stored more precisely, so you get better answers for the same file
size.

**Mixture of experts (MoE)** — a model built from many smaller sub-networks ("experts"), where
only a handful are consulted for any given token instead of the whole model. It lets a model have
a huge total parameter count while only paying the compute cost of a much smaller one per token;
Gemma 4 26B-A4B is an example this engine supports.

**Token** — a chunk of text, usually a short word or a piece of one. Models read and write tokens
rather than letters, and roughly 3 tokens is 4 characters of English.

**Context** — how many tokens the model can have in front of it at once: your whole conversation,
plus whatever it is writing. Set it with `-c`; a bigger context costs VRAM even when you are not
using it.

**KV cache** — the model's short-term memory for the conversation so far, held in VRAM. It grows
with every token in the context, which is why doubling `-c` can push a model that just fit into
running out of memory.

**Prefill** — the first phase of a request, where the model reads everything you sent it. It is
measured in tokens per second and it is fast, because the card can chew the whole prompt at once.

**Decode** — the second phase, where the model writes its answer one token at a time. It is much
slower than prefill per token, and it is the number you feel while watching text appear.

**Tokens per second (t/s)** — the speed. Prefill t/s is how fast it reads, decode t/s is how fast
it writes, and they are different numbers that are not comparable to each other.

**Layer split** (`-sm layer`) — the ordinary way to use two or more cards: put the first half of
the model on card one and the second half on card two. Only one card is working at any moment,
so two cards let you run a *bigger* model, not a faster one.

**Tensor split** (`-sm tensor`) — the other way: cut every part of the model down the middle so
both cards work on every token at the same time. It needs two identical cards that can talk to
each other directly, and when it works it makes the model *faster*, not just bigger. New this
release, and now the launcher's own default (`--sm auto`) on a matched pair it has hardware
evidence for; `-sm layer` still runs when you ask for it by name.

**Speculative decoding** — a speed trick where something small and quick guesses the next few
tokens and the real model checks them all in one go. Checking several guesses costs about as much
as producing one token, so when the guesses are good you get several tokens for the price of one.

**Drafter** — the thing doing the guessing. It can be a pattern-matcher that noticed you repeat
yourself, or a small extra head trained alongside the model.

**MTP** — "multi-token prediction", a small extra piece some models ship with whose only job is to
guess the next token or two. When a model has one, the engine can use it as the drafter and the
guesses are much better than pattern-matching alone.

**Lossless (as I use it)** — the answer is one the big model itself chose, not the drafter's
opinion. It does not mean the text is byte-for-byte identical to a run with the trick switched
off; see guide 03 for why, honestly.

**Flash attention** (`-fa on`) — a faster, more memory-efficient way of doing the attention step.
Leave it on for chatting; guide 02 says when to turn it off.

**Batch / micro-batch** (`-b` / `-ub`) — how many tokens the card is handed at once while reading
your prompt. The engine picks these for your card; you almost never need to touch them.

**Engine** — my llama.cpp-derived server. It is the low-latency one: best for one or two people
talking to it.

**Sidecar** — the vLLM-based server in this project. It is the throughput one: worse for one
person, much better when ten people are talking to it at once.

**OpenAI-compatible API** — the shape of HTTP requests that ChatGPT's API uses. Both my servers
speak it, so almost any chat app can point at them by changing one URL.

**Perplexity (PPL)** — a score for how surprised a model is by ordinary text. Lower is better, and
it is the cheapest way to tell whether a quantization run broke your model.

**KLD** — a score for how far a quantized model's opinions have drifted from the original's.
Smaller is better, and unlike perplexity it compares your file directly against the model you
started from.
