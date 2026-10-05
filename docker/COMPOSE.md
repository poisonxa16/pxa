# Run PXA with Docker Compose

Two cards and a model file. Four commands. You end up with an OpenAI-compatible
endpoint on `http://localhost:8080`, which means anything that already talks to
OpenAI — Open WebUI, the assistant panel in an editor, the `openai` Python package,
`curl` — talks to it without knowing it is not OpenAI.

## What you need first

- **NVIDIA driver 535 or newer**, and the **NVIDIA Container Toolkit** so that Docker
  can see the cards. Check both with:
  ```bash
  nvidia-smi
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu:22.04 nvidia-smi -L
  ```
  If the first works and the second does not, install the container toolkit — that is
  the only missing piece and it is a five-minute job.

  If you are used to writing `--gpus all` and it fails with something about an
  **ldcache error**, that is the container toolkit's older hook and not your cards;
  the `--runtime=nvidia` form above is what this compose file uses for exactly that
  reason, and there is a note in the file about switching to the newer form if your
  Docker prefers it.
- **Docker Compose v2** (`docker compose version`, not the old `docker-compose`).
- **A model file.** A `.gguf`. If you do not have one yet, the release page links
  several with their hashes.

Cards this is built for: Tesla P100, GTX 1080 Ti, Tesla P40, Tesla V100. Newer cards
work too, but they are not what the numbers were measured on.

## The four commands

```bash
curl -LO https://raw.githubusercontent.com/poisonxa16/pxa/main/docker/docker-compose.yml
curl -Lo .env https://raw.githubusercontent.com/poisonxa16/pxa/main/docker/.env.example
nano .env                 # three lines to change; see below
docker compose up -d
```

The three lines in `.env`:

```ini
MODELS_DIR=/home/you/models      # the folder your model file is in
MODEL_FILE=your-model.gguf       # the file name inside that folder
GPUS=0,1                         # which cards, numbered as nvidia-smi numbers them
```

Then watch it come up. A big model takes a few minutes to load off disk — that is
normal and it is not stuck:

```bash
docker compose logs -f pxa
```

When the log settles, ask it something:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":40}'
```

To stop it: `docker compose down`.

## Which card is card 0

`nvidia-smi -L` lists the cards in this machine and numbers them. Those are the
numbers to put in `GPUS`. Both images pin `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that
those numbers mean the same thing inside the container as outside — without it CUDA
sorts cards by how fast they are instead, and on a machine with two different kinds of
card you quietly get a different GPU than the one you asked for.

`GPUS=0,1` gives the server both cards and it will split the model across them
in layers. If your two cards are identical and can talk to each other directly, try
adding `PXA_EXTRA_FLAGS=-sm tensor -ts 1,1` in `.env` — that splits each layer across
both cards instead, which is faster for one conversation at a time. If it does not
help on your pair, take it back out; the layer split is the safe default for a reason.

## If something goes wrong

**The container keeps restarting.** Almost always the model did not fit. Halve `CTX`
in `.env` (8192 to 4096), `docker compose up -d` again. If it still restarts, the
model itself is too big for the cards you gave it.

**`could not select device driver "nvidia"`** or **`unknown runtime specified nvidia`**.
The NVIDIA Container Toolkit is not installed, or the Docker daemon has not been
restarted since it was. `docker info | grep -i runtime` should list `nvidia`.

**It says unhealthy but it answers.** The health check's patience is `START_PERIOD` in
`.env`. Raise it; a model loading from a hard disk can take longer than five minutes.

**You changed the port and now the health check fails.** Change `PXA_PORT` in `.env`
and nothing else — the compose file wires that one value to the server, the published
port and the health check together. Passing `--port` as an extra flag changes only one
of the three.

## PXA Control in the container (opt-in)

PXA Control is the browser page for the server: it shows it live (speed, cards, requests), has
a chat box, and can stop it. Outside a container `pxa` opens it by itself; in a container it is
**off unless you ask**, so a compose file you already run never opens a new port.

To turn it on, set `PXA_CONTROL=1` in `.env` and uncomment the port line under `ports:` in the
`pxa` service:

```ini
PXA_CONTROL=1
```

```yaml
      - "${PXA_CONTROL_PORT:-7777}:7777"
```

Then `docker compose up -d` and read the address from the log; it carries the access token
(`?token=...`) the page asks for once, then keeps in a cookie:

```bash
docker compose logs pxa | grep "PXA Control"
```

A recreated container (`docker compose down` then `up`) makes a new token. To keep one, add
`PXA_CONTROL_TOKEN: ${PXA_CONTROL_TOKEN}` to the service's `environment:` and put 16 or more
letters, digits, `_` or `-` after `PXA_CONTROL_TOKEN=` in `.env`.

## Which engine — and the tool that answers it

There are two engines in this stack and they are good at different things.

| | the engine (`pxa`) | the vLLM sidecar (`pxa-vllm`) |
|---|---|---|
| reads | `.gguf` files directly | a converted checkpoint directory |
| best at | one conversation at a time | several people at once |
| multi-card | layer split, or `-sm tensor` on a matched pair | real tensor parallelism |
| cards | P100, 1080 Ti, P40, V100 | V100 (`sm70`), P100 (`sm60`) |
| start-up | seconds to a couple of minutes | several minutes — it compiles CUDA graphs |

Most people want the first one, which is why it is what `docker compose up` starts.

You do not have to decide from a table. Run:

```bash
docker compose --profile tools run --rm launcher
```

It reads the cards actually in this machine, the model files actually in your models
folder — family and quantization tier out of each file's own header — asks what you
want the server for, and prints the configuration it would use, including which of the
two engines and why.

## Running the vLLM sidecar

Only worth it if several people or agents will hit the endpoint at once. It needs a
**converted checkpoint directory**, not a `.gguf`; `docs/VLLM.md` covers making one.

Set `VLLM_ARCH` (`sm70` for V100, `sm60` for P100), `VLLM_MODEL_DIR`, and `VLLM_GPUS`
in `.env`, then:

```bash
docker compose --profile vllm up -d
```

It appears on port 8000 and speaks the same OpenAI API. Give it several minutes on the
first boot: it captures CUDA graphs at start-up, which is slow on these cards and fast
for everything afterwards.

The defaults in the compose file for the sidecar are measured serving settings, not
round numbers. Two are worth knowing before you change them:

- `VLLM_GMU=0.88` is the memory ceiling that a 27B model actually loads at on 16 GB
  cards at TP=2. Raising it does not give you more room; it aborts during load.
- `TORCHDYNAMO_DISABLE=1` is what lets the sidecar start at all on a P100. Leave it
  alone on Pascal.

## Running both at once

They are separate services on separate ports, so:

```bash
docker compose --profile vllm up -d
```

starts both. Give them **different cards** in `GPUS` and `VLLM_GPUS`. Two servers on
one card will fit right up until the moment one of them needs memory the other took,
and then the failure looks like a bug in whichever one asked second.

## Comparing against another engine

The engine image carries one engine. To check a claim against a different engine on your own
card, with the same driver and the same weight file, changing exactly one thing, build the other
engine's `llama-server` statically, mount it into the container and use `--entrypoint`:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8080:8080 \
  -v /your/models:/models:ro \
  -v /path/to/other-engine/llama-server:/opt/pxa/bin/upstream-server:ro \
  --entrypoint /opt/pxa/bin/upstream-server \
  ghcr.io/poisonxa16/pxa:latest \
  -m /models/your-model.gguf -ngl 99 -c 8192 --host 0.0.0.0 --port 8080
```

`bench/fair/run.sh` runs both arms for you; see `docker/README.md`.

## What is in the engine image

The release tarball, unpacked, and nothing else added: the same `llama-server`, the
same libraries, the same bundled CUDA runtime, the same docs and the same launcher.
The binaries in the image are byte-for-byte the ones the tarball ships, so the
published gate results apply to the image without a separate claim. `docker image
inspect` will tell you which release and which commit:

```bash
docker image inspect ghcr.io/poisonxa16/pxa:v3 \
  --format '{{json .Config.Labels}}' | python3 -m json.tool
```
