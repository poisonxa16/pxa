# 07 — The whole stack with Docker Compose

One file, one command, and both servers come up. This is the path to use if you want the thing
running permanently rather than in a terminal you have to keep open.

You need Docker, the NVIDIA Container Toolkit, and your model file somewhere on disk. If you have
not run the engine at all yet, do [guide 01](01-run-your-first-model.md) first — it is easier to
fix one thing than three.

---

## Step 1 — check Docker can see your card

This is the single most common failure, and it has nothing to do with this project, so test it
first:

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

You should get the same table `nvidia-smi` prints on the host:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.142      Driver Version: 580.142      CUDA Version: 13.0      |
|   0  Tesla P100-PCIE-16GB    ...
```

**What just happened.** Docker handed a container your real GPU. If `nvidia-smi` works on the
host but that command fails, the NVIDIA Container Toolkit is the only missing piece — install it,
restart the Docker daemon, and try again. Nothing below will work until this does.

---

## Step 2 — get the compose file and set three things

The release carries `docker/docker-compose.yml` and `docker/.env.example`. Copy the example and
fill it in:

```bash
cd docker
cp .env.example .env
```

Three variables have no default and you must set all three:

| variable | what it is | example |
|---|---|---|
| `MODELS_DIR` | the folder **on your machine** holding your `.gguf` files | `/home/you/models` |
| `MODEL_FILE` | just the file name inside that folder, not a path | `fusion2-35b-U16-q8head.gguf` |
| `GPUS` | which cards, as a JSON list, numbered the way `nvidia-smi` numbers them | `["0"]` or `["0","1"]` |

So a minimal `.env` is three lines:

```bash
MODELS_DIR=/home/you/models
MODEL_FILE=fusion2-35b-U16-q8head.gguf
GPUS=["0"]
```

Everything else already has a sensible default — the port (8080), the context length (8192), the
image version. Change them only if you need to.

**What just happened.** Your models folder gets mounted **read-only** inside the container at
`/models`, which is why `MODEL_FILE` is a bare file name: the container does not know or care
where the folder lives on your machine. If you forget `MODELS_DIR`, compose refuses to start and
tells you which variable is missing, rather than guessing and failing later with something
cryptic.

> **If it went wrong**
> - **`set MODELS_DIR in .env to the directory holding your model files`** — exactly what it
>   says. You copied `.env.example` but did not edit it, or you edited the example instead of
>   `.env`.
> - **The container starts and says it cannot find the model** — `MODEL_FILE` has a path in it.
>   It must be just the file name. Check the file really is directly inside `MODELS_DIR`.
> - **`GPUS` is rejected** — it is a JSON list of strings, quotes and all: `["0","1"]`, not
>   `0,1`.

---

## Step 3 — start the engine

```bash
docker compose up -d
```

```
[+] Running 2/2
 ✔ Network docker_default   Created
 ✔ Container docker-pxa-1   Started
```

Watch it come up:

```bash
docker compose logs -f pxa
```

Look for the same lines guide 01 taught you — `offloaded N/N layers to GPU`, and a
`pxa: dev 0 <your card>` line naming the card you meant. Then, once it settles:

```bash
curl -s http://127.0.0.1:8080/health
```

```
{"status":"ok","slots_idle":1,"slots_processing":0}
```

That is the whole stack running. Point any OpenAI-compatible app at
`http://127.0.0.1:8080/v1` exactly as in [guide 01 §5](01-run-your-first-model.md).

**Be patient the first time.** The health check is given **five minutes** to go green, because a
14 GB model coming off a slow disk genuinely takes minutes. Do not shorten that — if you do,
Docker decides the container failed and restarts it, which starts the load again, forever.

To change the port, change **`PXA_PORT` in `.env` and nothing else**. It drives the server, the
published port and the health check together; setting a port in two places is how people end up
with a container that is healthy inside and unreachable outside.

Stop it with:

```bash
docker compose down
```

> **If it went wrong**
> - **The container restarts over and over** — almost always the health check timing out on a
>   slow first load, or the model not fitting. `docker compose logs pxa` shows which.
> - **`could not select device driver`** — step 1 did not really pass. Go back to it.
> - **Healthy, but nothing on port 8080** — you changed the port in the command instead of in
>   `.env`.
> - **`Auto-detected mode as 'legacy'` / `ldcache error`** — an older form of Docker's GPU flag
>   (`--gpus`) hitting a container-toolkit hook that does not like this host. The compose file
>   already uses the other form (`runtime: nvidia`), which does not have this problem; if you are
>   testing with a bare `docker run --gpus ...` command by hand and hit this, swap in
>   `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=...` instead.

---

## Step 4 — add the vLLM sidecar (optional)

The sidecar is the second server, for when several people are using the box at once. It is in the
same compose file but it does not start by default:

```bash
docker compose --profile vllm up -d
```

That brings up both. The engine keeps `8080`; the sidecar gets `8000`.

Two things will bite you if nobody tells you:

1. **The sidecar cannot read a `.gguf`.** It needs a *converted checkpoint directory*, which is a
   folder, not a file. You set `VLLM_MODEL_DIR` to that folder. Making one is
   [guide 05](05-quantize-for-vllm.md). This is the single most common mistake with the sidecar,
   and it is why the sidecar is not in the default profile — the person following guide 01 has a
   `.gguf` and would hit a wall.
2. **Give the two servers different cards.** They will both try to fill the card you give them.
   Running the engine on card 0 and the sidecar on card 1 works; running both on card 0 does not.

The sidecar is also given **fifteen minutes** to become healthy, because it compiles and captures
GPU graphs at startup. That is normal and it only happens once per start.

See [guide 06](06-run-the-vllm-sidecar.md) for what it is for and how to talk to it.

---

## Step 5 — the configuration checker

There is a third service that starts no server. It reads your cards and your model, prints the
configuration it would use and why, and exits:

```bash
docker compose --profile tools run --rm launcher
```

Use it when something will not fit, or when you want to know what the engine decided before you
wait for a load. It is the same reasoning described in
[guide 02 §7](02-pick-settings-for-your-cards.md), in a container.

---

## What is in the file, and what you may safely change

| thing | what it does | change it? |
|---|---|---|
| `image: ghcr.io/poisonxa16/pxa:${PXA_VERSION:-...}` | which release to run | only to pin an older version |
| `MODELS_DIR` / `MODEL_FILE` / `GPUS` | the three you must set | yes |
| `PXA_PORT`, `VLLM_PORT` | host ports | yes, one place each |
| `CTX` | context length, `-c` | yes — see [guide 02 §6](02-pick-settings-for-your-cards.md) |
| `PXA_EXTRA_FLAGS` | anything else you want passed straight to the server, such as `-sm tensor -ts 1,1` on a matched pair | yes — see [guide 02 §3](02-pick-settings-for-your-cards.md) before adding `-sm tensor` |
| `shm_size` (4 GB engine, 16 GB sidecar) and `ipc: host` | shared memory between processes | **no.** Docker's 64 MB default is not enough for multiple cards and the failure is confusing |
| `start_period` (300 s / 900 s) | how long the health check waits before giving up | **no**, unless you are making it longer |
| the `runtime: nvidia` line | how the card is handed to the container | no — it is the form measured to work; see the errata note under step 3 |

---

## Where next

- Talking to the engine → [01 §5](01-run-your-first-model.md)
- Making a checkpoint the sidecar can read → [05](05-quantize-for-vllm.md)
- Something will not come up → [10](10-when-something-goes-wrong.md)
