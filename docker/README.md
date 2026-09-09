# pxa — container image

A runnable `llama-server` image for Pascal and Volta cards (Tesla P100, GTX 1080 Ti / P40,
Tesla V100 — CUDA compute capability 60, 61, 70). No build tooling required on the host:
pull the image, mount a GGUF, run.

Models are **not** part of the image. Mount a directory containing your `.gguf` file(s)
at `/models` and point `-m` at the file inside the container.

## Quick start — one card (P100 or V100)

```bash
docker run -d --name pxa \
    --gpus '"device=0"' \
    -p 8080:8080 \
    -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-model.gguf \
    -ngl 99 -c 8192 -b 2048 -ub 512
```

If your Docker install uses the older nvidia-docker2 runtime instead of the `--gpus`
flag, replace `--gpus '"device=0"'` with `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0`.

Then:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/completion -d '{"prompt": "The capital city of France is", "n_predict": 8}'
```

Changing the port takes **two** flags, not one — pass `--port` for the server and
`-e LLAMA_ARG_PORT` (same value) so the image's built-in `HEALTHCHECK` probes the
right place (see "A gotcha with `LLAMA_ARG_*` env vars" below for why):

```bash
docker run -d --name pxa \
    --gpus '"device=0"' \
    -e LLAMA_ARG_PORT=8390 \
    -p 8390:8390 \
    -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-model.gguf \
    -ngl 99 -c 8192 -b 2048 -ub 512 --port 8390 --host 0.0.0.0
```

## Measured recipes

`-b`/`-ub` are optional on every recipe below — if you leave them off, the engine's
PXA posture layer (`PXA_MODE=balance` by default) sizes `-ub` from each device's free
VRAM and matches `-b` to it automatically (16 GB-class cards -> 2048, 11 GB-class ->
768). The explicit values below are what was actually measured for each topology;
pass them if you want the exact benchmarked behavior rather than the auto pick.

Each recipe adds `-e PXA_ENHANCE=1` to arm the per-architecture G3-class levers
(int8 prefill on sm_61, router-fusion on sm_70) that the measured numbers below
include. Leaving it unset runs the DEFAULT tier instead — still correct, just not
what was benchmarked.

### 2x V100 (layer-split, flash attention on)

```bash
docker run -d --name pxa-v100x2 \
    --gpus '"device=0,1"' \
    -e PXA_ENHANCE=1 \
    -e LLAMA_ARG_PORT=8390 \
    -p 8390:8390 \
    --ipc=host --shm-size=4g \
    -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-model.gguf \
    -ngl 99 -ts 1,1 -sm layer \
    -c 20480 -b 8192 -ub 2048 -fa \
    --port 8390 --host 0.0.0.0
```

(`--gpus '"device=0,1"'` selects your two V100s by index — substitute the pair
`nvidia-smi -L` shows; on the older nvidia-docker2 runtime use `--runtime=nvidia -e
NVIDIA_VISIBLE_DEVICES=0,1` instead.)

### 2x P100 (layer-split)

```bash
docker run -d --name pxa-p100x2 \
    --gpus '"device=0,1"' \
    -e PXA_ENHANCE=1 \
    -e LLAMA_ARG_PORT=8390 \
    -p 8390:8390 \
    --ipc=host --shm-size=4g \
    -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-model.gguf \
    -ngl 99 -ts 1,1 -sm layer \
    -c 20480 -b 8192 -ub 256 \
    --port 8390 --host 0.0.0.0
```

### 1x GTX 1080 Ti / P40 (PXQ2, ~35B tier)

The 1080 Ti's 11 GB budget targets a PXQ2-quantized ~35B model. `--ctx-checkpoints
0` disables the mid-prompt checkpoint mechanism — on an 11 GB single card the
checkpoint buffers cost more headroom than the resumable-prefix win is worth.

```bash
docker run -d --name pxa-1080ti \
    --gpus '"device=0"' \
    -e PXA_ENHANCE=1 \
    -e LLAMA_ARG_PORT=8390 \
    -p 8390:8390 \
    -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-pxq2-35b-model.gguf \
    -ngl 99 -c 8192 -b 2048 -ub 768 --ctx-checkpoints 0 \
    --port 8390 --host 0.0.0.0
```

## The second binary — the upstream engine, for engine-only comparisons

The image ships two engines:

| path | what it is |
|---|---|
| `/usr/local/bin/llama-server` | this engine. The image's `ENTRYPOINT`, and what every recipe above runs. |
| `/opt/pxa/bin/upstream-ik-server` | upstream `ik_llama.cpp`, pinned to commit `3c58ae37`, built in the same container for the same three architectures with the same CUDA toolkit. |

That second binary is there for one reason: an *engine-only* number — same weight file,
same card, same driver, same CUDA runtime, only the binary differs — is the one comparison
that cannot be argued with, and it is also the one nobody can reproduce if they have to
build the other engine themselves first. `3c58ae37` is the revision every published
upstream row in this repo was measured against (`bench/fair-battle.md`, `docs/ENGINE.md`,
`docs/data/chart-data-2026-09-02.csv`).

Ask the image what it is carrying rather than trusting a doc:

```bash
docker inspect --format '{{index .Config.Labels "org.pxa.upstream.ik.sha"}}' ghcr.io/poisonxa16/pxa:latest
docker run --rm --entrypoint cat ghcr.io/poisonxa16/pxa:latest /opt/pxa/bin/upstream-ik-server.rev
```

### Running the comparison

Both arms, one after the other, same file and same flags. Use a file **both** engines can
read — a stock quant such as MXFP4 or a `Q*_K` type. A PXQ file loads only in this engine,
so a PXQ arm is a product comparison, not an engine-only one:

```bash
# this engine (the default entrypoint)
docker run --rm --gpus '"device=0"' -p 8390:8390 -e LLAMA_ARG_PORT=8390 \
    -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-stock-quant.gguf \
    -ngl 99 -c 32768 -b 2048 -ub 2048 -fa on --port 8390 --host 0.0.0.0

# upstream, same everything, one flag different
docker run --rm --gpus '"device=0"' -p 8390:8390 -e LLAMA_ARG_PORT=8390 \
    -v /path/to/your/models:/models:ro \
    --entrypoint /opt/pxa/bin/upstream-ik-server \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-stock-quant.gguf \
    -ngl 99 -c 32768 -b 2048 -ub 2048 -fa on --port 8390 --host 0.0.0.0
```

Measure each arm the same way — `temperature 0`, `/completion`, a unique prompt per repeat,
median of 7 with one warmup discarded, speculative decode either on for both arms or off for
both. That is `bench/fair/protocol.md`, and `bench/fair/run.sh --rig <your rig>` runs exactly
these two arms for you and prints the block. It looks for the upstream binary at
`/opt/pxa/bin/upstream-ik-server` by default, which is where this image puts it.

The upstream binary is statically linked on purpose: the image carries this engine's
`libllama.so`/`libggml.so` in `/usr/local/lib`, and a dynamically linked upstream server
would load *those* at run time and quietly benchmark this engine's kernels under upstream's
name.

## Flags that matter

| flag | meaning | notes |
|---|---|---|
| `-m` | path to the `.gguf` inside the container | required, no default — always point at `/models/...` |
| `-ngl` | number of layers offloaded to GPU | no image default; `99` offloads the whole model, lower it only if a layer must stay on CPU |
| `-ts` | tensor split across devices (comma list, relative weights) | only matters with more than one visible GPU |
| `-c` | context size (tokens) | no image default (engine default is small); always pass this explicitly |
| `-b` | logical batch size | no image default |
| `-ub` | physical micro-batch size | no image default; this is the batch size the pipelining and checkpoint levers below are tuned against |
| `-np` | parallel request slots | omit for a single-user server; add `--kv-unified` if you raise it |

### A gotcha with `LLAMA_ARG_*` env vars

This engine applies `LLAMA_ARG_*` environment variables *after* parsing the CLI
flags, so an env var always wins over the equivalent flag — not the other way
round. To avoid silently ignoring the CLI flags shown above, this image bakes in
only `LLAMA_ARG_HOST=0.0.0.0` (so the server binds outside the container) and
leaves `-ngl`/`-c`/`-b`/`-ub`/`--port` etc. entirely to you. Configure each
setting as either a CLI flag (as the examples above do) or a matching `-e
LLAMA_ARG_*` var — not both for the same setting, since the env value would
silently win. If you change `--port`, also pass `-e LLAMA_ARG_PORT=<port>` so
the container's built-in `HEALTHCHECK` (which reads that same env var) probes
the right port.

## Mounting a model directory

Any host directory with `.gguf` files works. Bind-mount it read-only at `/models`:

```bash
-v /path/to/your/models:/models:ro
```

Then reference the file by its in-container path: `-m /models/<file>.gguf`. For a
multi-part GGUF, mount the directory holding all the shard files and point `-m` at the
first shard — llama.cpp finds the rest by name.

## PXA_* environment gates

These are engine-internal levers, compiled into this build with the `rc/unified-final-20260903`
defaults below. You normally do **not** need to touch any of them — they're documented here
so you know what's active and how to turn one off if you hit something unexpected on your
card. Set with `-e NAME=value` on `docker run`.

| variable | default in this image | what it does |
|---|---|---|
| `PXA_PIPELINE_PP` | off, except **on** automatically for the qwen3.5 / qwen3.5-moe architectures | enables scheduler pipeline parallelism (`n_copies > 1`) across GPUs on a layer split. Set `PXA_PIPELINE_PP=0` to force off on any model, `=1` to force on. |
| `PXA_CKPT_PROMPT_POLICY` | on | uses the mainline-style "checkpoint near the end of the prompt" policy instead of one checkpoint per micro-batch, cutting host-side drain stalls during long-context prefill. Set `=0` to restore the old per-micro-batch checkpointing. |
| `PXA_CKPT_LEAN_SYNC` | on | drops two of the three device-synchronize drains per context checkpoint, keeping only the one that guards the bulk device-to-host read. Set `=0` for the conservative triple-drain behavior. |
| `PXA_DN_SHARE_INPUTS` | auto (`-1`) — resolves to **on** when `n_seq_max <= 1`, off otherwise | shares DeltaNet layer inputs across the recurrent-state graph when there is only one sequence in flight. Set `=1` / `=0` to force. |
| `PXA_FUSE_DELTANET` | `55` (bitmask) | enables the DeltaNet CUDA kernel fusion clusters that passed the same-top-token / bit-exact gates (bit0 fusion, bit1 safe-carry, bit2 no-op skip, bit4 safe-carry cluster, bit5 provable row absorb). Set `=0` to disable all DeltaNet fusion and fall back to the eager per-op path. |
| `PXA_ENHANCE` | unset (DEFAULT tier) | `=1` arms the ENHANCE config tier: per-architecture G3-class levers that passed their own ship gates (sm_61-only int8 prefill, sm_70-only router-GEMV fusion, relaxed spec-decode lanes). The "Measured recipes" numbers above assume `PXA_ENHANCE=1`; without it you get the still-correct DEFAULT tier. `PXA_REFERENCE=1` goes the other way — every PXA lever forced off, for A/B comparisons against the plain reference path. |

Every other `PXA_*` lever in the engine keeps its compiled-in default; see the upstream
repo's `LEVERS.md` for the full list if you're chasing something unusual on non-standard
hardware.

## Health check

The image's `HEALTHCHECK` polls `GET /health` every 30s once the server has had 60s to
start. `docker ps` will show `healthy` / `unhealthy` accordingly.

## Building the image yourself

```bash
docker build -f docker/Dockerfile \
    --build-arg CUDA_ARCHS="60;61;70" \
    -t pxa:local .
```

`CUDA_ARCHS` controls which SM targets get compiled in (`60`=P100, `61`=1080 Ti/P40,
`70`=V100). Narrow it to your exact card to shrink build time and binary size.

This builds **two** CUDA engines — this one and the pinned upstream one — so it is a long
build on a busy machine; the release image is built once, at tag time, on an idle box.
`--build-arg BUILD_JOBS=6` caps the compile jobs if you need the cores for something else.

Two more knobs, both for the second binary:

| build arg | default | what it does |
|---|---|---|
| `IK_SHA` | `3c58ae373a0081c884099f435fb16ca720852bf7` | the upstream commit to build. Pinned, never a branch: a comparison against a moving target is not a comparison. Point it somewhere else and the numbers in this repo no longer describe what you built — say which revision you used. |
| `IK_URL` | `https://github.com/ikawrakow/ik_llama.cpp` | where to fetch it from (a mirror, or a local bare repo). |

The upstream stage does a shallow, single-commit fetch keyed only on `IK_SHA`, so editing
this repo's source never refetches or rebuilds it. To build just that stage:

```bash
docker build -f docker/Dockerfile --target build-upstream --build-arg BUILD_JOBS=6 .
```
