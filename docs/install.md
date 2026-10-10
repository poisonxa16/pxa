# Install PXA

PXA ships as a tarball, a one-line installer, or a container image. All three carry the same
binaries; models are separate and are never inside them.

## Tarball (recommended)

Download from the [release page](https://github.com/poisonxa16/pxa/releases/latest). Use
`pxa-v3.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz`, or the `-ubuntu22.04` variant on Ubuntu 22.04.
Then:

```bash
tar xzf pxa-v3.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz
cd pxa-v3.1-linux-x86_64-cuda12.8-sm60_61_70
./pxa --doctor                 # checks your cards, driver and CPU; starts nothing
./pxa                          # opens PXA Control in your browser
```

`START-HERE.md` in the tarball goes through the first fifteen minutes, with what a good start looks
like. One line starts a server without the launcher:

```bash
./run-server.sh -m /path/to/model.gguf          # serves on http://127.0.0.1:8080
curl -s http://127.0.0.1:8080/health            # {"status":"ok"}
```

## One command

The installer checks your CPU, glibc and cards, downloads the build that matches, verifies its
checksum and unpacks it:

```bash
curl -fsSL https://raw.githubusercontent.com/poisonxa16/pxa/main/install.sh | bash
```

Add `-s -- --docker` after `bash` to pull the container image instead.

## Container

`ghcr.io/poisonxa16/pxa:v3.1` carries the same binaries. Models are not in the image; mount a folder
at `/models`.

```bash
docker run -d --name pxa --gpus '"device=0,1"' --shm-size=1g -p 8080:8080 \
    -v /path/to/models:/models:ro \
    -v pxa-cache:/work/.cache/pxa -e PXA_CACHE_DIR=/work/.cache/pxa \
    ghcr.io/poisonxa16/pxa:v3.1
```

The `pxa-cache` volume is where a session's learned expert counts are kept. Without that mount they
are deleted when the container is removed. More than one card needs `--shm-size=1g`.

If `--gpus` fails with `nvidia-container-cli: ldcache error` (some hosts, Unraid among them), use
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1` instead.

Container images and Compose: `docker/COMPOSE.md`.

## From source

`BUILD-FROM-SOURCE.md`. A source build runs classic PXQ files and standard quants. For PXQN files,
add the compiled PXQN library from the release page (section 4b of that guide says how).

## Updating

PXA Control can install a newer release, and from a terminal the same tool is `pxa-update`:

```bash
pxa-update check
pxa-update apply
pxa-update rollback
```

Stop a server from this install before `apply` or `rollback`. Neither touches your models, your
settings, or a model's `.expert-counts.csv`.
