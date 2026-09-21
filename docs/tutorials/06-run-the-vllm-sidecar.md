# 06 — Run the vLLM sidecar

The sidecar is the second server. Use it when several people are talking to the box at once.

> **Honesty note.** I did not re-run the serving commands on this page on the day I wrote it —
> the cards were busy with other work. They are the lines I run on my own machine and they are
> what the shipping start-up scripts contain. The easiest and safest way to start it is
> [guide 07](07-docker-compose.md), which is one command and has been brought up end to end.

Before this page you need a **converted checkpoint folder** — see
[guide 05](05-quantize-for-vllm.md). The sidecar cannot read a `.gguf`.

Also: the sidecar runs on P100 (`sm_60`) and V100 (`sm_70`). **It does not run on a GTX 1080 Ti
or a P40.** Use the engine for those.

---

## Step 1 — the easy way

```bash
cd docker
docker compose --profile vllm up -d
```

Set `VLLM_MODEL_DIR` in `.env` to your converted folder first. Everything else has a working
default. [Guide 07](07-docker-compose.md) walks through it.

If that works, skip to step 3.

---

## Step 2 — the long way, one container at a time

Two things differ between the card families and both are load-bearing, so pick your line and
change only the paths in it.

### On two V100s

```bash
docker run -d --name pxa-vllm --runtime=nvidia \
  -p 127.0.0.1:8000:8000 --ipc=host --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v ~/models:/models \
  ghcr.io/poisonxa16/pxa-vllm:sm70-v2026.09.20 \
  python -m vllm.entrypoints.openai.api_server \
    --model /models/your-model-PXQ4-vllm --quantization pxq4 \
    --attention-backend FLASH_ATTN_V100 --tensor-parallel-size 2 --dtype float16 \
    --enable-prefix-caching --trust-remote-code \
    --gpu-memory-utilization 0.88 --max-model-len 32768 \
    --max-num-seqs 16 --max-num-batched-tokens 4096 \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16]}' \
    --host 0.0.0.0 --port 8000
```

### On two P100s

```bash
docker run -d --name pxa-vllm --runtime=nvidia \
  -p 127.0.0.1:8001:8001 --ipc=host --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCHDYNAMO_DISABLE=1 \
  -v ~/models:/models \
  ghcr.io/poisonxa16/pxa-vllm:sm60-v2026.09.20 \
  python -m vllm.entrypoints.openai.api_server \
    --model /models/your-model-PXQ4-vllm --quantization pxq4 \
    --attention-backend PASCAL_SDPA --tensor-parallel-size 2 --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
    --host 0.0.0.0 --port 8001
```

**The five things in there that are not decoration:**

| | why it is there |
|---|---|
| `--attention-backend PASCAL_SDPA` / `FLASH_ATTN_V100` | the attention code written for that card family. The wrong one will not run |
| `TORCHDYNAMO_DISABLE=1` (**P100 only**) | without it the server does not start at all — it tries to compile GPU code your card is too old for, and stops with `GPUTooOldForTriton` |
| `--compilation-config '...'` in **single quotes** | those are the graph sizes it pre-compiles. Leave it out and it quietly compiles only the two smallest, and everything above two simultaneous conversations runs several times slower with no error. The single quotes matter: without them your shell mangles the JSON |
| `--ipc=host --shm-size=16g` | Docker's default shared memory is 64 MB, which is not enough for two cards to talk |
| `--gpu-memory-utilization` | how much of the card it may fill. Higher is not better: too high and it fails at startup on a 16 GB card |

**It takes minutes to start.** It loads the model and then compiles and captures GPU graphs. That
is once per start, and it is why the health check is given a long window.

```bash
docker logs -f pxa-vllm
curl -s http://127.0.0.1:8000/health
```

---

## Step 3 — talk to it

The sidecar speaks the same OpenAI-compatible shape as the engine, so any chat app points at it
the same way:

| setting | value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | anything; none is required unless you started it with `--api-key` |
| Model name | the path you passed to `--model`, unless you set `--served-model-name` |

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/your-model-PXQ4-vllm","messages":[{"role":"user","content":"In one sentence: what is a GPU?"}],"max_tokens":60}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

The port is published on `127.0.0.1` only, so it is reachable from that machine. To let other
machines in, change the `-p` mapping to `-p 8000:8000` and mind who is on your network.

---

## Step 4 — the two flags a tool-calling client needs

If your client sends `tools` — anything that lets the model call a function, search the web, or
run code — you need **two extra flags on the server**, and they must be there at startup:

```
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Add them to the `api_server` line in step 2.

**Without them, every request carrying `tools` comes back as HTTP 400.** Plain chat keeps
working, which is what makes this confusing: the server looks healthy and one particular client
just fails.

**And the parser name matters.** For Qwen-family models it is `qwen3_coder`, not `hermes`. Both
names are accepted; picking the wrong one does not raise an error, it returns **empty tool
calls** — the model calls the tool, the parser cannot read the format it used, and your client
sees nothing. If tool calls silently do nothing, this is the first thing to check.

---

## Step 5 — when something looks wrong

The sidecar's failures are mostly quiet, which is the worst kind. These are the ones worth
recognising:

| what you see | what it is |
|---|---|
| `GPUTooOldForTriton` while starting, on a P100 | `TORCHDYNAMO_DISABLE=1` missing |
| fluent garbage from the very first character on a short prompt | the graph mode is wrong; `cudagraph_mode` must be `FULL_DECODE_ONLY` on P100 |
| `!!!!` on a one-word prompt but sane answers otherwise (P100) | the capture list is too small, or `--gpu-memory-utilization` is too high |
| everything fine until about three people use it, then roughly four times slower | the capture list collapsed to its default two entries. Pass `--compilation-config` explicitly |
| `tools=` requests return 400 | the two flags in step 4 |
| tool calls come back empty, no error | wrong `--tool-call-parser` |
| `undefined symbol: ...TensorImpl...` part way through loading | the kernel library and the container do not match. Use the image as shipped |
| it asks for log probabilities and gets a 500 | known, still open. Generation itself is unaffected; the engine's equivalent endpoint works |

For anything else, [guide 10](10-when-something-goes-wrong.md) applies here too, and Discord is
<https://discord.gg/EqazvV9tf>.

---

## Where next

- Both servers with one command → [07](07-docker-compose.md)
- Making a checkpoint it can read → [05](05-quantize-for-vllm.md)
- Deciding which server you want → [05 §1](05-quantize-for-vllm.md)
