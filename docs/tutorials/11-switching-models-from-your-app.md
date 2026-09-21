# 11 — Switching models from your app

Every other guide in this set runs **one model per server**. That is the right default — it is
the lowest-latency configuration, and it is what every other page assumes. This page is for a
different shape: an app (a coding assistant, a tool that scripts several models, anything that lets the *caller*
pick a model by name in its request) that wants to choose between several models you have on disk,
without you hand-starting a different server for each one.

This engine does not do that by itself. What it does do — and what makes it possible — is print
the **exact command line** it would start a given model with, for your exact cards, without
guessing. The rest of this page is about handing that command to a small proxy that starts and
stops servers on demand, keyed by the model name in the request.

---

## 1. Get the exact command for each model

For every model you want switchable, ask the launcher what it would run — this starts nothing:

```bash
./pxa-launch --explain -m /models/qwen38-27b-pxq4.gguf --gpus 0,1
```

It prints the full command line: the engine binary, `-ngl`, the split mode, `-c`, `-b`/`-ub`, flash
attention, and the chat-template flags this specific model needs — the same reasoning
[guide 02](02-pick-settings-for-your-cards.md) walks through by hand, done for you. Copy that
line. Repeat for each model file, giving each its own port:

```bash
./pxa-launch --explain -m /models/qwen38-27b-pxq4.gguf --gpus 0,1 --port 8081
./pxa-launch --explain -m /models/gemma-4-26B-A4B-PXQ3.gguf --gpus 0 --port 8082
```

**Why not just write the commands by hand from guide 02?** You can — the launcher just saves you
re-deriving the right split mode, batch sizes and chat-template flags for each file, and it is the
same command a person running that model directly would get, so nothing about how the model
behaves changes because a proxy is in front of it.

---

## 2. Put a model-swapping proxy in front of them

You need something that: listens on one port, reads the `model` field of an incoming
OpenAI-compatible request, and starts (or reuses) the right backend command for that name,
stopping ones you are not using so idle models are not holding VRAM. This engine does not include
one — that is a separate, general-purpose piece, not specific to this project. **[llama-swap](https://github.com/mostlygeek/llama-swap)**
is a widely used one built for exactly this shape (it manages any OpenAI-compatible
command-line server, not only this one), and it is what these steps use as the example; any proxy
with the same job — one command per model, routed by name — works the same way.

A minimal config, one entry per model, each `cmd` being the exact line guide step 1 printed for
that model:

```yaml
models:
  "qwen38-27b":
    cmd: PXA_TSPLIT_REDUCE=fused PXA_TSPLIT_REDUCE_PREFILL=1 PXA_TSPLIT_FALLBACK=1 /path/to/run-server.sh -m /models/qwen38-27b-pxq4.gguf -ngl 99 -sm tensor -ts 1,1 -c 32768 -fa on --host 127.0.0.1 --port 8081
    proxy: http://127.0.0.1:8081

  "gemma4-26b-a4b":
    cmd: /path/to/run-server.sh -m /models/gemma-4-26B-A4B-PXQ3.gguf -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port 8082
    proxy: http://127.0.0.1:8082
```

The `qwen38-27b` line is what `--explain` now prints for a matched pair of identical cards on this
architecture and tier: `--sm auto` (the launcher's default) resolves to `-sm tensor -ts 1,1` with
the fused all-reduce armed at decode and prefill, not the `-sm layer` line an earlier printing of
this guide showed. Gemma 4 still resolves to `-sm layer` — that architecture is not in the set
`--sm auto` picks the split for yet (see `docs/LAUNCHER.md`).

Point your app at the proxy's port instead of this engine's port directly, and send whichever of
the two names above in the request's `"model"` field. The proxy starts that server on first use and
can stop the other to free the card between switches — read your chosen proxy's own docs for its
exact idle/unload behaviour, since that part is the proxy's decision, not this engine's.

**What this buys you, and what it costs.** One endpoint, several models, chosen per request instead
of per server you hand-started. The cost is the same cost as stopping one server and starting
another by hand: whichever model is not currently loaded has to load from disk on its first
request after a switch, exactly as in [guide 01](01-run-your-first-model.md) — a proxy does not
make that faster, it only automates the button-press.

---

## 3. Whole folder at once

If you have many model files and do not want to write one launcher command and one proxy entry by
hand for each, run the launcher once per file in a loop and collect the printed commands — there is
no single flag in this release that writes a whole proxy config file for you from a folder of
models. That is a reasonable thing to want, and it is on the list for a future release; until then,
step 1 above, repeated, is the way to get there.

---

## Where next

- What each flag in the printed command means → [02](02-pick-settings-for-your-cards.md)
- Running one model the ordinary way → [01](01-run-your-first-model.md)
- Something in a printed command does not work as shown here → [10](10-when-something-goes-wrong.md)
