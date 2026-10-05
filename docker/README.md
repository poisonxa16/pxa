# pxa — container image

A runnable `llama-server` image for Pascal and Volta cards (Tesla P100, GTX 1080 Ti / P40,
Tesla V100 — CUDA compute capability 60, 61, 70). No build tooling required on the host:
pull the image, mount a GGUF, run.

Models are **not** part of the image. Mount a directory containing your `.gguf` file(s)
at `/models` and point `-m` at the file inside the container.

## The entrypoint: one command, the engine picks its settings

The image's `ENTRYPOINT` is `pxa-entrypoint`. It has one rule, so a `docker run` line
you already use keeps its meaning:

| You run | What starts |
|---|---|
| `docker run IMAGE` (no arguments) | **pxa-launch** picks the cards, the model and the flags, prints why, and starts the server. The model is `$PXA_MODEL`, or the only `.gguf` under `/models`; the cards are `$PXA_GPUS`, or every card the container can see. |
| `docker run IMAGE -m /models/x.gguf ...` | Anything that starts with `-` is an **engine argument** and goes straight to `llama-server`, unchanged. |
| `docker run IMAGE doctor` | `pxa-launch --doctor`: cards, driver, P2P, the model file (tier, arch, sha256) and the defaults the engine would pick, with the reason for each. Starts nothing. |
| `docker run -p 7777:7777 IMAGE gui` | **PXA Control**, the launcher in a browser (rig, models, launch, speed, chat). In a container it listens on every interface and prints an access token; open the printed `?token=` address. See docs/LAUNCHER.md, "PXA Control (GUI)". |
| `docker run -e PXA_CONTROL=1 -p 8080:8080 -p 7777:7777 IMAGE ...` | Either of the first two, **plus PXA Control next to the server**: the server live, its speed, a chat box, Stop. Opt-in on purpose: without `PXA_CONTROL=1` no web port opens, exactly as before. It listens on every interface inside the container and needs the access token it prints once in the log (`docker logs <name>`; `-e PXA_CONTROL_TOKEN=<16+ characters>` keeps it fixed). |
| `docker run IMAGE launch --gpus 0,1 ...` | pxa-launch with your own arguments. |
| `docker run IMAGE llama-bench ...` (any program) | That program. |

Both of the first two get the same settings on the same cards. Since v2026.10 the engine
picks `-sm`, `-b`, `-ub`, `-fa`, `-ngl` (every layer on the cards) and `-c` (`-np` x 4096)
itself when you do not pass them, from the same registry pxa-launch reads (on two identical cards with a Qwen3.8 dense
PXQ4 or PXQ-Next 4 / 4S8 / 5 file, and on four identical P100s, it picks `-sm tensor`; other card sets stay on
`-sm layer`). A flag you pass always wins. The boot log prints every choice as a `PXA_REGISTRY:` line.

**More than one card: give the container a bigger `/dev/shm`.** Docker and podman give a container
64 MiB of `/dev/shm`; the multi-card reduce (NCCL's shared-memory transport) needs more on four
cards. Pass `--shm-size=1g` (or `--ipc=host`; compose: `shm_size: 1gb`). In LXC, mount a 1 GiB
tmpfs on `/dev/shm` (`lxc.mount.entry: tmpfs dev/shm tmpfs rw,nosuid,nodev,create=dir,size=1G 0 0`).
The entrypoint and `pxa-launch` print a warning when a multi-card container has less; the engine
itself falls back to a slower reduce route rather than serve wrong tokens (bug #206).

```bash
# no arguments: the launcher does everything (one model under /models)
docker run -d --name pxa --gpus '"device=0,1"' --shm-size=1g -p 8080:8080 \
    -v /path/to/your/models:/models:ro ghcr.io/poisonxa16/pxa:latest

# what would it pick, and why? (starts nothing)
docker run --rm --gpus '"device=0,1"' -v /path/to/your/models:/models:ro \
    ghcr.io/poisonxa16/pxa:latest doctor -m /models/your-model.gguf
```

`PXA_LAUNCH_EXTRA="--explain"` on the no-argument path prints what the launcher would run
and starts nothing. With engine arguments, the one difference from the launcher is `-c`:
a launcher recipe row passes its own context (for example 32768), the bare engine picks
`-np` x 4096 - pass `-c` yourself for more.

The same binaries answer to `pxa-server`, `pxa-bench`, `pxa-quantize`, `pxa-perplexity`
and `pxa-cli`; the `llama-*` names keep working.

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
docker run -d --name pxa-server \
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

## Comparing against another engine

The image ships one engine: `/usr/local/bin/llama-server` (the image's `ENTRYPOINT`, and what every
recipe above runs). `/opt/pxa/bin/llama-server` is a symlink to it — one binary, both documented
paths.

An *engine-only* number — same weight file, same card, same driver, same CUDA runtime, only the
binary differs — is the one comparison that cannot be argued with. To get one, bring the other
engine's `llama-server` yourself: build it statically (`-DBUILD_SHARED_LIBS=OFF`) with the same CUDA
toolkit and the same architectures, mount it into the container, and run it with `--entrypoint`.
Link it statically on purpose: the image carries this engine's `libllama.so`/`libggml.so` in
`/usr/local/lib`, and a dynamically linked second server would load *those* at run time and quietly
benchmark this engine's kernels under another engine's name.

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

# the other engine, same everything, one flag different
docker run --rm --gpus '"device=0"' -p 8390:8390 -e LLAMA_ARG_PORT=8390 \
    -v /path/to/your/models:/models:ro \
    -v /path/to/other-engine/llama-server:/opt/pxa/bin/upstream-server:ro \
    --entrypoint /opt/pxa/bin/upstream-server \
    ghcr.io/poisonxa16/pxa:latest \
    -m /models/your-stock-quant.gguf \
    -ngl 99 -c 32768 -b 2048 -ub 2048 -fa on --port 8390 --host 0.0.0.0
```

Measure each arm the same way — `temperature 0`, `/completion`, a unique prompt per repeat,
median of 7 with one warmup discarded, speculative decode either on for both arms or off for
both. That is `bench/fair/protocol.md`, and `bench/fair/run.sh --rig <your rig>` runs exactly
these two arms for you and prints the block. It looks for the second binary at
`/opt/pxa/bin/upstream-server` by default; set `UPSTREAM_BIN=/path/to/your/build` to point it
elsewhere.

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

## Encoding with the Pro encoder inside the container

A Pro encoder key is tied to at most two machines. A container has no `/etc/machine-id` of its own, so the encoder
identifies the machine by the GPUs it can see, and a container started with a different `NVIDIA_VISIBLE_DEVICES`
counts as another machine. Give it the host's id, read-only:

```bash
-v /etc/machine-id:/etc/machine-id:ro
```

Check first that the host has that file (`cat /etc/machine-id`): if it does not, Docker creates an empty directory
in its place. Running the engine needs none of this; only the Pro encoder looks at it.

## PXA_* environment gates

These are engine-internal levers, compiled into this build with the unified build of 2026-09-03
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

`--build-arg BUILD_JOBS=6` caps the compile jobs if you need the cores for something else; the
release image is built once, at tag time, on an idle box.
