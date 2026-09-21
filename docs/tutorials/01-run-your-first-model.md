# 01 — Run your first model

By the end of this page you will have a chat server running on your own card, and you will have
talked to it twice: once from the command line, once from a normal chat app.

Budget: fifteen minutes, nearly all of it downloading the model.

If you have not read [00 — Start here](00-start-here.md), read the "what you need" section of it
first. You need an NVIDIA driver and nothing else.

---

## Step 1 — get the release

Pick the **tarball** path or the **Docker** path. They end up in the same place.

### Tarball

```bash
mkdir -p ~/pxa && cd ~/pxa
curl -L -O https://github.com/poisonxa16/pxa/releases/download/v2026.09.20/pxa-v2026.09.20-linux-x86_64-cuda12.8-sm60_61_70.tar.gz
tar xzf pxa-v2026.09.20-linux-x86_64-cuda12.8-sm60_61_70.tar.gz
cd pxa-v2026.09.20
```

You should see a folder with these things in it:

```bash
ls
```

```
BUILD-FROM-SOURCE.md  LICENSE  README.md  START-HERE.md  VERSION
bench/  bin/  docs/  lib/  pxa-launch  run-server.sh  tools/
```

Check it can actually start:

```bash
./run-server.sh --version
```

```
version: 5592 (518e0a7b7a)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```

The build number and commit will be this release's, not the ones above.

### Docker

```bash
mkdir -p ~/models
docker pull ghcr.io/poisonxa16/pxa:v2026.09.20
```

Nothing to unpack. Models live in `~/models` on your machine and the container sees them at
`/models`.

**What just happened.** The tarball is the whole program: the server, its libraries, and its own
copy of the CUDA runtime. It does not install anything, it does not touch your system, and you
delete it by deleting the folder. The Docker image is the same binaries with a Linux userspace
wrapped around them, which is why it works on distros too old for the tarball.

> **If it went wrong**
> - **`version GLIBC_2.34 not found`** — your Linux is older than the binaries. Use the Docker
>   path instead; it carries its own userspace.
> - **`./run-server.sh: Permission denied`** — the extract lost the executable bit. `chmod +x
>   run-server.sh pxa-launch bin/*` fixes it.
> - **`tar: ... not in gzip format`** — the download was interrupted and you have an HTML error
>   page. Delete it and download again.

---

## Step 2 — get a model

Models are not in the release. You download one `.gguf` file and point at it.

```bash
cd ~/pxa/pxa-v2026.09.20     # or: cd ~/models, for the Docker path
curl -L -O https://huggingface.co/poisonxa/PXA-bench-files-GGUF/resolve/main/fusion2-35b-U16-q8head.gguf
```

That is a 35-billion-parameter model squeezed to about 14 GB, which is the right size for one
16 GB card. If your card is an 11 GB 1080 Ti, take the smaller file instead:

```bash
curl -L -O https://huggingface.co/poisonxa/PXA-bench-files-GGUF/resolve/main/fusion2-35b-PXQ2.gguf
```

Check the download is complete and not a half-file:

```bash
ls -l fusion2-35b-U16-q8head.gguf
```

```
-rw-r--r-- 1 you you 14018474880 Sep 20 16:20 fusion2-35b-U16-q8head.gguf
```

**What just happened.** A `.gguf` is one self-contained file: the weights, the vocabulary, and
the chat template all in one. There is nothing to unpack and no folder structure to get right.

> **If it went wrong**
> - **The file is a few kilobytes** — you downloaded an error page. Check the URL and retry.
> - **The download died halfway** — add `-C -` to `curl` to resume: `curl -L -C - -O <url>`.
> - **You want to check it properly** — the release carries the expected checksums in
>   `bench/fair/weights/MANIFEST.sha256`. `sha256sum <file>` and compare. A mismatch means
>   download again; do not try to debug anything else until it matches.

### Other models worth knowing about

This example file is a good first model, but it is not the only one. Two worth knowing early:

- **Qwen3.8-27B** is what most of the numbers in the project README were measured on, and it is
  the model to reach for if you want to reproduce a published figure yourself.
- **Gemma 4 26B-A4B** (Google's 128-expert mixture-of-experts model) runs here too, no switch
  needed. If you only have **one** 16 GB card, ask for the **PXQ3** tier of it specifically — it
  boots at 16,000 tokens of context on a single card, where Google's own file of the same model
  only fits 4,096 on the same card. [Guide 04](04-quantize-your-own-model.md) shows how to make
  your own PXQ file of any model you already have the weights for.

---

## Step 3 — start the server

### Tarball

```bash
./run-server.sh -m fusion2-35b-U16-q8head.gguf -ngl 99 -c 8192 -fa on
```

### Docker

```bash
docker run -d --name pxa \
    --gpus '"device=0"' \
    -p 8080:8080 \
    -v ~/models:/models:ro \
    ghcr.io/poisonxa16/pxa:v2026.09.20 \
    -m /models/fusion2-35b-U16-q8head.gguf \
    -ngl 99 -c 8192 -fa on
```

Either way you get a page of startup log. The lines worth your eyes are these:

```
PXA config level: ENHANCE (default; PXA_ENHANCE=0 for DEFAULT, PXA_REFERENCE=1 for REFERENCE)
PXA level=ENHANCE | dev0 Tesla P100-PCIE-16GB(sm_60): FP16_GEMM ON [2:1 hgemm] ...
PXA posture: mode=balance [fa-on serving, decode-first] fa=on (explicit) ub=2048 ...
pxa: engine | codec=PXQ2 (120 tensors; pxq2 120, pxq3 0, pxq4 0, pxq6 0)
pxa: dev 0 Tesla P100-PCIE-16GB cc 6.0 -> path: sm_60 fp16-hfma2
llm_load_tensors: offloaded 41/41 layers to GPU
```

Three of those are your health check:

1. **`offloaded 41/41 layers to GPU`** — the number before the slash must equal the number after
   it. `0/41` means the model is running on your CPU and will be about fifty times slower.
2. **`pxa: dev 0 <your card>`** — it found the card you meant.
3. **`codec=...`** — it recognised the file's format. `codec=off` means this is not one of my
   files, which is fine, just slower.

**What just happened.** Those three flags mean: `-ngl 99` put every layer of the model on the
graphics card, `-c 8192` gave the conversation room for 8192 tokens, and `-fa on` turned on the
faster attention maths. The engine worked out everything else — batch sizes, which of my GPU
tricks are worth using on your specific card — by reading your card at startup, and it printed
every decision it made. There is nothing else to configure.

The server listens on port 8080. With the tarball it binds to `127.0.0.1`, so it is reachable
from that machine only. With Docker it binds inside the container and `-p 8080:8080` is what
exposes it.

> **If it went wrong**
> - **`CUDA driver version is insufficient for CUDA runtime version`** — your driver is older
>   than 570. Upgrade the driver. Do not install a CUDA toolkit; the package has its own.
> - **`offloaded 0/41 layers`** — either you forgot `-ngl 99`, or something is shadowing the real
>   NVIDIA driver library. Check nothing else has put a `stubs` directory on `LD_LIBRARY_PATH`.
> - **Docker: `could not select device driver "" with capabilities: [[gpu]]`** — Docker cannot
>   see your card. Try the older form instead of `--gpus`: `--runtime=nvidia -e
>   NVIDIA_VISIBLE_DEVICES=0`. Only one of the two works on any given host.
> - **It loads for a while and then dies** — the model does not fit alongside its context. Drop
>   `-c 8192` to `-c 4096`, or use a smaller file. [Guide 02](02-pick-settings-for-your-cards.md)
>   shows how to check before you wait.

---

## Step 4 — talk to it from the command line

Open a second terminal. First, is it up?

```bash
curl -s http://127.0.0.1:8080/health
```

```
{"status":"ok","slots_idle":1,"slots_processing":0}
```

If you get "connection refused", it is still loading — a big model from a spinning disk can take
a couple of minutes. Wait and try again.

Now ask it something:

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital city of France is","n_predict":16,"temperature":0}'
```

You get a lump of JSON. The answer is in the `content` field:

```
{"content":" Paris. It is located in the north-central part of France, on the Seine", ...}
```

If reading raw JSON annoys you, pipe it through Python:

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital city of France is","n_predict":16,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])'
```

```
 Paris. It is located in the north-central part of France, on the Seine
```

**What just happened.** `/completion` is the raw endpoint: it takes your text and continues it,
with no chat formatting at all. `n_predict` is how many tokens to write, and `temperature: 0`
means "always pick the most likely next word", which makes the answer repeatable. It is the
simplest possible proof that the whole stack works.

> **If it went wrong**
> - **`Connection refused`** — still loading, or it crashed. Look at the server's terminal.
> - **The answer is repeated punctuation or gibberish** — that is a real bug and I want to hear
>   about it. Please do not paper over it with a longer prompt; see
>   [guide 10](10-when-something-goes-wrong.md) for what to send me.
> - **It answers something confidently wrong** — that is the model, not the engine. Small models
>   make things up. Try a bigger file.

---

## Step 5 — talk to it from a chat app

The server also speaks the same HTTP shape that OpenAI's API uses, which means almost any chat
app can point at it.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pxa","messages":[{"role":"user","content":"In one sentence: what is a GPU?"}],"max_tokens":60,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

You get a normal conversational answer. Some models start with a `<think>` block where they
reason out loud before answering — that is the model's own habit, not a fault.

To see what name the server is serving under:

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool | head -8
```

```
{
    "object": "list",
    "data": [
        {
            "id": "/path/to/your-model.gguf",
            "object": "model",
```

### Pointing an app at it

In any OpenAI-compatible app — Open WebUI, LibreChat, Continue, a script using the `openai`
Python package — set these three things:

| setting | value |
|---|---|
| Base URL / API base | `http://127.0.0.1:8080/v1` |
| API key | anything at all; the server does not check one unless you asked it to |
| Model name | whatever `/v1/models` printed, or just `pxa` |

If the app runs on a different machine, restart the server with `--host 0.0.0.0` so it listens on
the network, and use your machine's IP instead of `127.0.0.1`. Do that only on a network you
trust — there is no password on the port unless you add one with `--api-key`.

If you want an app to pick between **several** models on one server instead of hard-coding one,
see [guide 11](11-switching-models-from-your-app.md).

**What just happened.** `/v1/chat/completions` is the endpoint every chat app knows. The server
takes your messages, wraps them in whatever chat format this particular model was trained on
(that format is stored inside the `.gguf` file), and replies in the shape the app expects.

> **If it went wrong**
> - **The app says "model not found"** — use the exact `id` string that `/v1/models` printed, or
>   whatever name the app insists on. Most apps do not actually care.
> - **Tool calling / function calling returns an error** — start the server with `--jinja` so it
>   uses the model's own template. Without it, requests carrying `tools` fail.
> - **The reply is cut off mid-sentence** — raise `max_tokens`.

---

## Step 6 — stop it

Tarball: `Ctrl-C` in the terminal running it.

Docker:

```bash
docker stop pxa && docker rm pxa
```

Check the card is actually free again:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv
```

```
index, memory.used [MiB]
0, 5 MiB
```

---

## Where next

- More than one card, or wondering about context length → [02](02-pick-settings-for-your-cards.md)
- Want it faster → [03](03-going-faster.md)
- Want to quantize your own model → [04](04-quantize-your-own-model.md)
- An app that needs to pick between several models → [11](11-switching-models-from-your-app.md)
- Something broke → [10](10-when-something-goes-wrong.md)
